import os
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from data.get_loader import get_loader
import yaml
from arch.rwkv4sr import RWKVIR
from typing import Optional, Dict, Any
from torch.profiler import profile, record_function, ProfilerActivity
from torcheval.metrics import PeakSignalNoiseRatio
from torchvision.io import read_image
import torchvision.transforms as T
import torch.nn.functional as F


class Trainer():
    def __init__(self, config_dir: str) -> None:
        self.config = self._load_config(config_dir)
        
        self.train_datasets = self.config['data']['train_dataset_dirs']

        self.device = self._resolve_device(self.config['device'])

        self.dataloader = get_loader(self.train_datasets, 
                                     self.config['data']['scale'], 
                                     self.config['data']['patch_size'],
                                     self.config['data']['batch_size'],
                                     self.device,
                                     self.config['data']['num_workers'],
                                     self.config['data']['persist_workers'],
                                     self.config['data']['pin_memory'],
                                     self.config['data']['prefetch_factor']
                                     )

        
        self.checkpoint_dir = self.config['model']['checkpoint_dir']
        self.checkpoint_freq = self.config['trainer']['checkpoint_freq']
        self.log_freq = self.config['trainer']['log_freq']
        self.checkpoint_dir = self.config['model']['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.global_step = 0
        self.start_iteration = 0
        self.lr_milestones = list(self.config['scheduler'].get('lr_milestones', []))
        self.lr_gamma = float(self.config['scheduler'].get('lr_gamma', 0.5))

        self.model = self._init_model()
        self.optimizer = self._init_optimizer()
        self.criterion = nn.L1Loss()
        self.scheduler = self._init_scheduler()
        self._setup_amp()
        self._init_metrics()

        if self.config['trainer']['logging']:
            self._log_init()

        resume_path = None
        if self.config['trainer']['resume']:
            resume_path = self._get_latest_checkpoint(self.checkpoint_dir)
        if resume_path:
            self._load_checkpoint(resume_path)




    @staticmethod
    def _load_config(config_dir: str) -> Dict[str, Any]:
        config = None
        try:
            with open(config_dir, 'r') as f:
                config = yaml.safe_load(f)
                print('config loaded successfully')
        except FileNotFoundError as e:
            raise FileNotFoundError('config yaml not found')
        except Exception as e:
            raise ValueError('error loading config from yaml')
        return config
    
    def _resolve_device(self, requested: Optional[str]) -> torch.device:
        if requested:
            return torch.device(requested)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    def _get_latest_checkpoint(self, checkpoint_dir: str) -> Optional[str]:
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        if not checkpoint_files:
            return None
        checkpoint_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)
        return os.path.join(checkpoint_dir, checkpoint_files[0])

    def _init_model(self) -> None:
        model_config = self.config['model']
        depths = [model_config['blocks_per_layer']] * model_config['residual_groups']
        model = RWKVIR(
            img_size=model_config['patch_size'],
            depths=depths,
            mlp_ratio=3.,
            patch_size=model_config['patch_size'],
            img_range=1,
            embed_dim=model_config['embed_dim'],
            upscale=model_config['scale'],
            upsampler=model_config['upsampler'],
            resi_connection=model_config['resi_connection']#,
            # n_head=model_config['num_heads']
        )
        return model.to(self.device)

    def _init_optimizer(self) -> None:
        opt_config = self.config['optimizer']
        name = opt_config['name'].lower()
        if name == 'adam':
            betas = tuple(opt_config.get('betas'))
            print(f'optimizer: {name} initialized successfully')
            return torch.optim.Adam(self.model.parameters(), 
                                    lr=opt_config['lr'], 
                                    betas=betas)
        if name == 'adamw':
            betas = tuple(opt_config.get('betas'))
            weight_decay = float(opt_config.get('weight_decay', 0.01))
            print(f'optimizer: {name} initialized successfully')
            return torch.optim.AdamW(self.model.parameters(), 
                                     lr=opt_config['lr'], 
                                     betas=betas,
                                     weight_decay=weight_decay)
        
        raise ValueError(f"Optimizer not defined in _init_optimizer(): {opt_config['name']}")
    
    def _init_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        if self.lr_milestones:
            milestones = [int (milestone) for milestone in self.lr_milestones]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                             milestones=milestones,
                                                             gamma=self.lr_gamma)
            print('scheduler initialized successfully')
            return scheduler
        return None
    
    def _log_init(self) -> None:
        self.run = wandb.init(
            entity=self.config['wandb']['entity'],
            project=self.config['wandb']['project'],
            name=self.config['wandb']['run_name'],
            id=self.config['wandb']['id'],
            resume=self.config['wandb']['resume'],
        )
        wandb.watch(self.model, 
                    log='gradients', 
                    log_freq=self.config['trainer']['log_freq'])
        print('wandb initialized successfully')

    def log(self, loss: float, lr: float, psnr: float) -> None:
        self.run.log({'loss': loss, 'lr': lr, 'psnr': psnr})

    def _setup_amp(self) -> None:
        dt = self.device.type
        self.autocast_device = dt
        self.scaler = None

        if dt == 'cuda':
            use_bf16 = torch.cuda.is_bf16_supported()
            self.autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
            print(f'using cuda autocast with dtype: {self.autocast_dtype}')
            self.scaler = torch.amp.GradScaler('cuda', enabled=not use_bf16)
        elif dt == 'mps':
            self.autocast_dtype = torch.float16
            self.scaler = None
            print('using mps autocast with dtype: torch.float16')
        else:
            self.autocast_dtype = torch.float32
            self.scaler = None
            print('using cpu autocast with dtype: torch.float32')

    def _init_metrics(self) -> None:
        self.psnr_metric = PeakSignalNoiseRatio().to(self.device)
        self.test_img_hr = read_image('training_data/test_HR.png').to(self.device)
        self.test_img_hr = T.ConvertImageDtype(torch.float32)(self.test_img_hr).unsqueeze(0)
        self.test_img_lr = read_image('training_data/test.png').to(self.device)
        self.test_img_lr = T.ConvertImageDtype(torch.float32)(self.test_img_lr).unsqueeze(0)

        print('PSNR metric initialized successfully')

    def calculate_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        self.psnr_metric.update(pred, target)
        psnr = self.psnr_metric.compute().item()
        self.psnr_metric.reset()
        return psnr

    def train(self) -> None:
        num_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'model has {num_parameters} trainable parameters')
        print(f'device used for training: {self.device}')
        print('starting training...')
        num_iterations = self.config['trainer']['num_iterations']

        remaining = max(0, num_iterations - self.global_step)
        pbar = tqdm(total=remaining, desc=f'Iteration {self.global_step}/{num_iterations}')

        while self.global_step < num_iterations:
            self.model.train()

            lr, hr = self.dataloader.next()
            # with torch.amp.autocast(device_type=self.autocast_device, dtype=self.autocast_dtype):
            preds = self.model(lr)
            loss = self.criterion(preds, hr)

            self.optimizer.zero_grad()

            # if self.scaler:
            #     self.scaler.scale(loss).backward()
            #     self.scaler.step(self.optimizer)
            #     self.scaler.update()
            # else:
            loss.backward()
            self.optimizer.step()

            self.global_step += 1
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            if self.global_step % self.log_freq == 0:
                self.model.eval()
                with torch.no_grad():
                    psnr = self.calculate_psnr(self.model(self.test_img_lr), self.test_img_hr)
                self.model.train()
                if self.config['trainer']['logging']:
                    self.log(loss.item(), current_lr, psnr)

                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{current_lr:.4f}', 'psnr': f'{psnr:.4f}'})

            if self.global_step % self.checkpoint_freq == 0:
                self.checkpoint(self.global_step)
            pbar.update(1)



    def checkpoint(self, iteration: int) -> None:
        state = {
            'iteration': iteration + 1,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }

        path = self.checkpoint_dir + '/' + f'iteration_{iteration}.pt'
        torch.save(state, path)
        print(f'checkpoint saved at {path} successfully')
    
    def _load_checkpoint(self, checkpoint_dir: str) -> None:
        checkpoint = torch.load(checkpoint_dir, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.start_iteration = checkpoint.get("iteration", 0)
        self.global_step = checkpoint.get("global_step", 0)
        print(f'checkpoint loaded from {checkpoint_dir} successfully, resuming from iteration {self.start_iteration}')




    

    
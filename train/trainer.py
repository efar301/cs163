import os
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from data.get_loader import get_loader
import yaml
from arch.model import RWKVSR
from typing import Optional, Dict, Any
from torch.profiler import profile, record_function, ProfilerActivity



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
        self.start_epoch = 0

        self.model = self._init_model()
        self.optimizer = self._init_optimizer()
        self.criterion = nn.L1Loss()
        self._setup_amp()

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
        model = RWKVSR(
            scale=model_config['scale'],
            in_channels=model_config['in_channels'],
            num_channels=model_config['num_channels'],
            num_blocks=model_config['num_blocks']
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
        
        raise ValueError(f"Optimizer not defined in _init_optimizer(): {opt_config['name']}")

    def _log_init(self) -> None:
        self.run = wandb.init(config={**self.config['wandb']})
        wandb.watch(self.model, log='gradients', log_freq=self.config['trainer']['log_freq'])
        print('wandb initialized successfully')

    def log(self, loss: float, lr: float) -> None:
        self.run.log({'loss': loss, 'lr': lr})

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


    def train(self) -> None:
        num_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'model has {num_parameters} trainable parameters')
        print(f'device used for training: {self.device}')
        print('starting training...')
        num_epochs = self.config['trainer']['num_epochs']

        for epoch in range(self.start_epoch, num_epochs):
            self.model.train()
            progress = tqdm(range(len(self.dataloader)), desc=f'Epoch {epoch + 1}/{num_epochs}')
            for _ in progress:
                lr, hr = self.dataloader.next()

                with torch.amp.autocast(device_type=self.autocast_device, dtype=self.autocast_dtype):
                    preds = self.model(lr)
                    loss = self.criterion(preds, hr)

                self.optimizer.zero_grad()

                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                self.global_step += 1
                current_lr = self.optimizer.param_groups[0]['lr']
                
                if self.global_step % self.log_freq == 0:
                    if self.config['trainer']['logging']:
                        self.log(loss.item(), current_lr)
                    progress.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{current_lr:.4f}'})
            
            if self.global_step % self.checkpoint_freq == 0:
                self.checkpoint(epoch)
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        #     for _ in range(30):  # warmup a bit
        #         lr, hr = self.dataloader.next()
        #         with torch.autocast(device_type=self.autocast_device, dtype=self.autocast_dtype):
        #             preds = self.model(lr); loss = self.criterion(preds, hr)
        #         self.optimizer.zero_grad(set_to_none=True); loss.backward(); self.optimizer.step()
        # print(prof.key_averages().table(sort_by="self_cpu_time_total"))

    def checkpoint(self, epoch: int) -> None:
        state = {
            'epoch': epoch + 1,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }

        path = self.checkpoint_dir + '/' + f'epoch_{epoch + 1}.pt'
        torch.save(state, path)
        print(f'checkpoint saved at {path} successfully')
    
    def _load_checkpoint(self, checkpoint_dir: str) -> None:
        checkpoint = torch.load(checkpoint_dir, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        print(f'checkpoint loaded from {checkpoint_dir} successfully, resuming from epoch {self.start_epoch}')




    

    
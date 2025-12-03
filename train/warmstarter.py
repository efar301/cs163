import os
import time
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from data.get_loader import get_loader
import yaml
from arch.idrwkv import IDRWKV
from typing import Optional, Dict, Any, Iterable, List, Tuple
from torch.profiler import profile, record_function, ProfilerActivity
from torchvision.io import read_image
import torchvision.transforms as T
import torch.nn.functional as F

from common.test_utils import calculate_psnr, calculate_ssim

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

torch.backends.cudnn.benchmark = True              
torch.backends.cuda.matmul.allow_tf32 = True       
torch.backends.cudnn.allow_tf32 = True

def _tensor_to_uint8(img_tensor: torch.Tensor):
    """Convert CHW tensor in [0, 1] to HWC uint8 BGR array."""
    img = img_tensor.detach().cpu()
    if img.dim() == 4:
        img = img.squeeze(0)
    if img.dim() == 3 and img.shape[0] in (1, 3):
        img = img.permute(1, 2, 0)
    img = img.clamp(0, 1).mul(255).round().to(torch.uint8).contiguous()
    return img.numpy()[..., ::-1]

class FFTLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion = torch.nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)

        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)

        return self.loss_weight * self.criterion(pred_fft, target_fft)
    
class SpecialLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(SpecialLoss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion = nn.L1Loss(reduction=reduction)
        self.fft_criterion = FFTLoss(loss_weight=0.01)

    def forward(self, pred, target):
        l1_loss = self.loss_weight * self.criterion(pred, target)
        fft_loss = self.fft_criterion(pred, target)
        return l1_loss + fft_loss


class WarmStarter():
    def __init__(self, config_dir: str) -> None:
        self.config = self._load_config(config_dir)
        self._seed_everything(self.config['seed'])
        
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
        self.ft_milestones = list(self.config.get('finetuning', {}).get('lr_milestones', []))

        self.model = self._init_model()
        # self._freeze_params()
        self.optimizer = self._init_optimizer()
        # self.criterion = nn.L1Loss()
        self.criterion = SpecialLoss()
        self.scheduler = self._init_scheduler()
        self._setup_amp()

        if self.config['trainer']['logging']:
            self._log_init()

        resume_path = None
        if self.config['trainer']['resume']:
            resume_path = self._get_latest_checkpoint(self.checkpoint_dir)
        if resume_path:
            self._load_checkpoint(resume_path)

    def _seed_everything(self, seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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


    def _init_optimizer(self) -> None:
        opt_config = self.config['optimizer']

        lr = opt_config['lr']

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        name = opt_config['name'].lower()
        if name == 'adam':
            betas = tuple(opt_config.get('betas'))
            print(f'optimizer: {name} initialized successfully')
            return torch.optim.Adam(trainable_params, 
                                    lr=lr, 
                                    betas=betas)
        
        raise ValueError(f"Optimizer not defined in _init_optimizer(): {opt_config['name']}")

    def _init_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        
        milestones = [int (milestone) for milestone in self.lr_milestones]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                            milestones=milestones,
                                                            gamma=self.lr_gamma)
        print('scheduler initialized successfully')
        return scheduler

    def _init_model(self) -> None:
        model_config = self.config['model']
        model = IDRWKV(
            img_size=model_config['patch_size'],
            num_blocks=model_config['num_blocks'],
            hidden_rate=model_config['hidden_rate'],
            patch_size=model_config['patch_size'],
            img_range=1,
            embed_dim=model_config['embed_dim'],
            upscale=model_config['scale'],
            n_head=model_config['num_heads'],
            with_cp=model_config['with_cp']
        )
        return model.to(self.device)
    
    def _freeze_params(self) -> None:
        for name, param in self.model.named_parameters():
            if name.startswith('upsample'):
                param.requires_grad = True
            else:
                param.requires_grad = False

        print('parameters frozen successfully')

    def _log_init(self) -> None:
        self.run = wandb.init(
            entity=self.config['wandb']['entity'],
            project=self.config['wandb']['project'],
            name=self.config['wandb']['run_name'],
            id=self.config['wandb']['id'],
            resume=self.config['wandb']['resume']
        )
        wandb.watch(self.model, 
                    log='gradients', 
                    log_freq=self.config['trainer']['log_freq'])
        print('wandb initialized successfully')

    def log(self, loss: float, lr: float, psnr: float, ssim: float) -> None:
        self.run.log({'loss': loss, 'lr': lr, 'psnr': psnr, 'ssim': ssim})

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

    def benchmark(
        self,
        datasets: Optional[Iterable[str]] = None,
        scales: Optional[Iterable[str]] = None,
        lr_root: str = 'testing_data/LR/LRBI',
        hr_root: str = 'testing_data/HR'
    ) -> List[Tuple[str, str, float, float, float]]:
        self.model.eval()

        if not os.path.isdir(lr_root) or not os.path.isdir(hr_root):
            raise FileNotFoundError('Benchmark folders not found.')

        dataset_names = list(datasets) if datasets else sorted(os.listdir(lr_root))
        scales = list(scales) if scales else [f'x{self.config["model"]["scale"]}']

        results: List[Tuple[str, str, float, float, float]] = []

        for dataset in dataset_names:
            for scale in scales:
                lr_dir = os.path.join(lr_root, dataset, scale)
                hr_dir = os.path.join(hr_root, dataset, scale)

                if not os.path.isdir(lr_dir) or not os.path.isdir(hr_dir):
                    continue

                lr_images = sorted(os.listdir(lr_dir))
                hr_images = set(os.listdir(hr_dir))
                image_pairs = []
                for lr_image in lr_images:
                    hr_img_name = lr_image.replace('_LRBI_', '_HR_')
                    if hr_img_name in hr_images:
                        image_pairs.append(
                            (os.path.join(lr_dir, lr_image), os.path.join(hr_dir, hr_img_name))
                        )

                if not image_pairs:
                    continue

                dataset_psnrs = []
                dataset_ssims = []
                crop_border = int(scale.replace('x', ''))

                for lr_path, hr_path in image_pairs:
                    lr = read_image(lr_path)
                    lr = T.ConvertImageDtype(torch.float32)(lr).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        predicted = self.model(lr).squeeze(0)

                    hr = read_image(hr_path)
                    hr = T.ConvertImageDtype(torch.float32)(hr)

                    predicted_arr = _tensor_to_uint8(predicted)
                    hr_arr = _tensor_to_uint8(hr)

                    psnr_val = calculate_psnr(
                        predicted_arr,
                        hr_arr,
                        crop_border=crop_border,
                        input_order='HWC',
                        test_y_channel=True
                    )
                    ssim_val = calculate_ssim(
                        predicted_arr,
                        hr_arr,
                        crop_border=crop_border,
                        input_order='HWC',
                        test_y_channel=True
                    )

                    dataset_psnrs.append(psnr_val)
                    dataset_ssims.append(ssim_val)

                avg_psnr = sum(dataset_psnrs) / len(dataset_psnrs)
                avg_ssim = sum(dataset_ssims) / len(dataset_ssims)
                # print(f'PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}')
                results.append((dataset, scale, avg_psnr, avg_ssim))

        self.model.train()
        return results

    def warmstart(self) -> None:
        num_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'model has {num_parameters} trainable parameters')
        print(f'device used for training: {self.device}')
        print('starting finetuning...')

        num_iterations = self.config['trainer']['num_iterations']

        remaining = max(0, num_iterations - self.global_step)
        pbar = tqdm(ncols=140, total=remaining, desc=f'Iteration {self.global_step}/{num_iterations}')

        while self.global_step < num_iterations:
            self.model.train()

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
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            if self.global_step % self.log_freq == 0:
                self.model.eval()
                with torch.no_grad():
                    results = self.benchmark(datasets=['Set5'], scales=[f'x{self.config["model"]["scale"]}'])
                    dataset, scale, psnr, ssim = results[0]
                if self.config['trainer']['logging']:
                    self.log(loss.item(), current_lr, psnr, ssim)

                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{current_lr:.6f}', 'psnr': f'{psnr:.4f}', 'ssim': f'{ssim:.4f}'})
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
            'finetuning': True,
            'warmstart': True
        }

        path = self.checkpoint_dir + '/' + f'iteration_{iteration}.pt'
        torch.save(state, path)
        print(f'\ncheckpoint saved at {path} successfully')

    def _load_checkpoint(self, checkpoint_dir: str) -> None:
        checkpoint = torch.load(checkpoint_dir, map_location=self.device)
        state_dict = checkpoint['model_state_dict']
        is_warmstart = bool(checkpoint.get('warmstart'))

        if not is_warmstart:
            body_state_dict = {
                k: v for k, v in state_dict.items()
                if not k.startswith('upsample')
            }

            missing, unexpected = self.model.load_state_dict(body_state_dict, strict=False)

            self.start_iteration = 0
            self.global_step = 0
            print(
                f'checkpoint loaded from {checkpoint_dir} successfully, '
                f'beginning upsample-only finetune from iteration {self.start_iteration}'
            )
        else:
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            self.start_iteration = checkpoint.get('iteration', 0)
            self.global_step = checkpoint.get('global_step', 0)
            print(
                f'checkpoint loaded from {checkpoint_dir} successfully, '
                f'resuming finetuning from iteration {self.start_iteration}'
            )


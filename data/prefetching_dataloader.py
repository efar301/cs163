# Adapted from the ASID DataPrefetcher found at
# https://github.com/saturnian77/ASID/blob/master/data_tools/dataloader_DIV2K_memory.py

import torch
from torch.utils.data import DataLoader
from typing import Tuple, Optional

def _resolve_device(requested: Optional[str]) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

class DataPrefetcher():
    def __init__(self, loader: DataLoader, device: Optional[str]) -> None:
        self.loader = loader
        self.device = _resolve_device(device)
        self.dataiter = iter(loader)
        self.stream = torch.cuda.Stream() if self.device.type == 'cuda' else None
        self.__preload__()
        print(f'data prefetcher using device: {self.device}')
        print('data prefetcher initialized successfully')

    def __preload__(self) -> None:
        try:
            self.lr, self.hr = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.loader)
            self.lr, self.hr = next(self.dataiter)

        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                self.hr = self.hr.cuda(non_blocking=True)
                self.lr = self.lr.cuda(non_blocking=True)
        else:
            self.hr = self.hr.to(self.device)
            self.lr = self.lr.to(self.device)

    def next(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        hr = self.hr
        lr = self.lr
        self.__preload__()
        return lr, hr
    
    def __len__(self):
        return len(self.loader) 
    

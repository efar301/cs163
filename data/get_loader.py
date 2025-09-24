# Adapted from the ASID DataPrefetcher
# # https://github.com/saturnian77/ASID/blob/master/data_tools/dataloader_DIV2K_memory.py

from torch.utils.data import DataLoader
from .image_pair_dataset import Paired_Dataset
from .prefetching_dataloader import DataPrefetcher
from typing import Tuple, Optional

def get_loader(image_directories: list, scale: int, patch_size: Tuple[int, int], batch_size: int, 
               device: Optional[str], num_workers: int, pin_memory: bool) -> DataPrefetcher:
    data_class = Paired_Dataset(image_directories, scale, patch_size)
    dataloader = DataLoader(data_class, batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    prefetcher = DataPrefetcher(dataloader, device=device)
    return prefetcher


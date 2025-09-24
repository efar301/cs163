import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T
from torchvision.transforms.functional import InterpolationMode
from torchvision.io import read_image
import os
from typing import Tuple

class Paired_Dataset(Dataset):
    def __init__(self, image_directories: list[str], scale: int, patch_size: Tuple[int, int]) -> None:
        super(Paired_Dataset, self).__init__()

        # list of each image path that we use
        self.image_paths = self.get_image_paths(image_directories)
        self.scale = scale
        self.patch_size = patch_size

    def get_image_paths(self, image_directories: list[str]) -> list:
        image_paths = []

        for directory in image_directories:
            for path in os.listdir(directory):
                if path.endswith('.png'):
                    image_paths.append(directory + '/' + path)

        return image_paths

    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[index]
        image = read_image(image_path)
        # normalize the image to [0,1]
        image = T.ToDtype(torch.float32, scale=True)(image)

        hr_crop = T.RandomCrop((self.scale * self.patch_size[0], self.scale * self.patch_size[1]))(image)

        aug = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomChoice([
                T.Identity(),
                T.RandomRotation((90, 90)),
                T.RandomRotation((180, 180)),
                T.RandomRotation((270, 270)),
            ])
        ])
        hr_tensor = aug(hr_crop)

        # bicubically downscale the hr image to the patch size
        lr_tensor = T.Resize(self.patch_size, interpolation=InterpolationMode.BICUBIC)(hr_tensor)
        return torch.clamp(hr_tensor, 0, 1), torch.clamp(lr_tensor, 0, 1)



            
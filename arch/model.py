# model will be here
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1, stride=1),
        )
    
    def forward(self, x):
        res = x
        x = self.ff(x)
        x = x + res
        return x

class RWKVSR(nn.Module):
    def __init__(self, scale: int, in_channels: int, num_channels: int, num_blocks: int) -> None:
        super().__init__()

        self.feat_extractor = nn.Conv2d(in_channels=in_channels, out_channels=num_channels, kernel_size=3, padding=1, stride=1)

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(ResBlock(num_channels=num_channels))

        self.pre_upscale = nn.Conv2d(in_channels=num_channels, out_channels=3*scale*scale, kernel_size=3, padding=1, stride=1)
        self.upscale = nn.PixelShuffle(scale)
    
    def forward(self, x: torch.Tensor) -> None:
        x = self.feat_extractor(x)
        res = x
       
        for block in self.blocks:
            x = block(x)

        x = self.pre_upscale(x)
        x = self.upscale(x)
        return x

if __name__ == '__main__':
    model = RWKVSR(2, 3, 64, 8)
    test = torch.randn(3, 100, 100)

    out = model(test)
    print(test.shape, out.shape)
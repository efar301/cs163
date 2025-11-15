# -----------------------------------------------------------------------------------
# RWKVIR: Image Restoration Using RWKV Transformer
# -----------------------------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, to_2tuple, trunc_normal_
from arch.wkv_6.vrwkv_6_9 import Block as RWKV
from einops import rearrange


def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

class ElementScale(nn.Module):
    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=requires_grad)

    def forward(self, x):
        return x * self.scale


class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )


    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCCA(nn.Module):
    def __init__(self, embed_dims, feedforward_channels, kernel_size=3):
        super(RCCA, self).__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels

        self.dwconv = nn.Conv2d(in_channels=self.feedforward_channels, out_channels=self.feedforward_channels,
                                kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=True,
                                groups=self.feedforward_channels)

        self.decompose = nn.Conv2d(in_channels=self.feedforward_channels, out_channels=1, kernel_size=1)
        self.sigma = ElementScale(self.feedforward_channels, init_value=1e-5, requires_grad=True)

        self.cca = CCALayer(self.feedforward_channels, self.feedforward_channels // 4)

        self.act = nn.GELU()
        self.decompose_act = nn.GELU()

    def feat_decompose(self, x):
        # x_d: [B, C, H, W] -> [B, 1, H, W]
        x = self.sigma(x - self.decompose_act(self.decompose(x)))
        return x

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.act(x)
        x1 = self.feat_decompose(x)
        x2 = self.cca(x)
        x = x1 + x2
        return x + input

# ESA from HNCT
class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m
    
    
class MBConv(nn.Module):
    def __init__(self, dim, expansion_rate=4):
        super().__init__()
        hidden_dim = int(dim * expansion_rate)
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)

class RWKVBlock(nn.Module):  # Add CNN block here for the RWKV improevment
    def __init__(self, dim, n_head, n_layer, layer_id, hidden_rate=4,
                post_norm=False, key_norm=False, with_cp=False, norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.RWKV = RWKV(n_embd=dim, n_head=n_head, n_layer=n_layer, layer_id=layer_id, hidden_rate=hidden_rate,
                        post_norm=post_norm, with_cp=with_cp)
        self.norm = norm_layer(dim)
        
    def forward(self, x, patch_resolution):
        # patch_resolution needs to be reset
        B, _, C = x.shape
        shortcut = x

        # RWKV
        x = self.RWKV(x, patch_resolution)
        
        # FFN
        return x + shortcut
        
  
class BasicLayer(nn.Module): #
    def __init__(self, dim, input_resolution, depth, n_head, n_layer, layer_id, hidden_rate=4,
                post_norm=False, with_cp=False, norm_layer=nn.LayerNorm,  use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # It will be used for the RWKV forward
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            RWKVBlock(dim=dim, n_head=n_head, n_layer=n_layer, layer_id=layer_id, hidden_rate=hidden_rate,
                    post_norm=post_norm, with_cp=with_cp, norm_layer=norm_layer)
            for i in range(depth)])
        

        

    def forward(self, x, x_size):   # The use of x_size needs to be changed!!
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        return x


class RWKVB(nn.Module):
    def __init__(self, dim, input_resolution, depth, n_head, n_layer, layer_id, hidden_rate=4,
                post_norm=False, with_cp=False,
                norm_layer=nn.LayerNorm, use_checkpoint=False,
                img_size=224, patch_size=4):
        super(RWKVB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.act = nn.GELU()
        

        # blocks for distillation
        self.conv1 = MBConv(dim, expansion_rate=2)
        self.block1 = RWKVBlock(dim=dim, n_head=n_head, n_layer=n_layer, layer_id=layer_id, hidden_rate=hidden_rate,
                                post_norm=post_norm, with_cp=with_cp, norm_layer=norm_layer)
        self.distill1 = nn.Conv2d(dim, dim // 2, 1)
        self.remain1 = nn.Conv2d(dim, dim // 2, 1)
        
        self.conv2 = MBConv(dim // 2, expansion_rate=2)
        self.block2 = RWKVBlock(dim=dim // 2, n_head=n_head // 2, n_layer=n_layer, layer_id=layer_id, hidden_rate=hidden_rate,
                                post_norm=post_norm, with_cp=with_cp, norm_layer=norm_layer)
        self.distill2 = nn.Conv2d(dim // 2, dim // 4, 1)
        self.remain2 = nn.Conv2d(dim // 2, dim // 4, 1)                                
        

        self.conv3 = MBConv(dim // 4, expansion_rate=2)
        self.block3 = RWKVBlock(dim=dim // 4, n_head=n_head // 4, n_layer=n_layer, layer_id=layer_id, hidden_rate=hidden_rate,
                                post_norm=post_norm, with_cp=with_cp, norm_layer=norm_layer)
        
        self.mixer = nn.Conv2d(dim, dim, 1, 1, 0)
        self.cca = RCCA(dim, dim)
        self.esa = ESA(dim, nn.Conv2d)

    def forward(self, x, x_size):
        H, W = x_size
        shortcut = x
        shortcut = rearrange(shortcut, 'b (h w) c -> b c h w', h=H, w=W)
        # full dim pass
        f = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        f = self.conv1(f)
        f = rearrange(f, 'b c h w -> b (h w) c', h=H, w=W)
        f = self.block1(f, x_size)
        f = rearrange(f, 'b (h w) c -> b c h w', h=H, w=W)
        d1 = self.distill1(f)
        r1 = self.remain1(f)

        # dim // 2 pass
        r1 = self.conv2(r1)
        r1 = rearrange(r1, 'b c h w -> b (h w) c', h=H, w=W)
        r1 = self.block2(r1, x_size)
        r1 = rearrange(r1, 'b (h w) c -> b c h w', h=H, w=W)
        d2 = self.distill2(r1)
        r2 = self.remain2(r1)

        # dim // 4 pass
        r2 = self.conv3(r2)
        r2 = rearrange(r2, 'b c h w -> b (h w) c', h=H, w=W)
        r2 = self.block3(r2, x_size)
        r2 = rearrange(r2, 'b (h w) c -> b c h w', h=H, w=W)

        # concat and process
        out = torch.cat((d1, d2, r2), dim=1)
        out = self.mixer(out)
        out = self.cca(out)
        out = out + shortcut
        out = self.esa(out)
        out = rearrange(out, 'b c h w -> b (h w) c', h=H, w=W)
        return out


        
class RWKVIR(nn.Module):
    def __init__(self, img_size=64, patch_size=1, in_chans=3,
                 embed_dim=64, depths=[6, 6, 6, 6], hidden_rate=4.,
                 post_norm=False, key_norm=False, with_cp=False,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, upscale=4, img_range=1., n_head=8,
                 **kwargs):
        super(RWKVIR, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.n_head = n_head
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.hidden_rate = hidden_rate
        ################################### 1, shallow feature extraction ###################################
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        ################################### 2, deep feature extraction ######################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # build RWKV blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RWKVB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=depths[i_layer],
                         n_head=self.n_head,
                         n_layer=self.num_layers,
                         hidden_rate=hidden_rate,
                         img_size=img_size,
                         patch_size=patch_size,
                         layer_id=i_layer,
                         post_norm=post_norm, 
                         with_cp=with_cp, norm_layer=norm_layer, 
                         use_checkpoint=use_checkpoint,
                         )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction

        self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        ################################ 3, high quality image reconstruction ################################

        self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                        (patches_resolution[0], patches_resolution[1]))


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if getattr(m, 'init_scale', 1) == 0:
                nn.init.constant_(m.weight, 0)
            else:
                trunc_normal_(m.weight, std=0.02 * getattr(m, 'init_scale', 1))
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        
        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        H, W = x.shape[2:]
        # x = self.check_image_size(x)
        
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range


        x = self.conv_first(x)
        x = self.forward_features(x)
        x = self.conv_after_body(x) + x
        x = self.upsample(x)


        x = x / self.img_range + self.mean
        x = x[:, :, :H*self.upscale, :W*self.upscale]
        return x.clamp(0, 1)

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9
        return flops


def buildRWKVIR():
    return RWKVIR(img_size=(64, 64), depths=[6, 6, 6, 6], 
                   mlp_ratio=2., patch_size=8,
                   img_range=1, embed_dim=64, upscale=2,
                   upsampler='pixelshuffledirect', resi_connection='1conv',)

# from fvcore.nn import FlopCountAnalysis, parameter_count_table

if __name__ == '__main__':
    upscale = 2
    height = 20
    width = 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RWKVIR(img_size=(height, width), depths=[1, 1, 1], 
                   hidden_rate=3, patch_size=8,
                   img_range=1, embed_dim=64, upscale=upscale, n_head=8).to(device)
    
    
    x = torch.randn((1, 3, height, width)).to(device)
    pred = model(x)
    print(pred.shape)
    print("x on:", x.device)
    print("model param on:", next(model.parameters()).device)
    print(f"number of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

#     flops = FlopCountAnalysis(model, x)
#     print("FLOPs: ", flops.total())
#     print(parameter_count_table(model))
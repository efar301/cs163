# -----------------------------------------------------------------------------------
# RWKVIRv4 This arch uses the RWKV4
# -----------------------------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import to_2tuple, trunc_normal_
from arch.wkv_4.vrwkv_4_11 import Block as RWKV
from einops.layers.torch import Rearrange, Reduce
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
    
class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, hidden_dim, bias = False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias = False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)
    
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
    
class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)



class RWKVBlock(nn.Module):  # Add CNN block here for the RWKV improevment
    def __init__(self, dim, input_resolution, n_layer, layer_id, drop_path=0., hidden_rate=4,
                init_values=None, post_norm=False, key_norm=False, with_cp=False,
                mlp_ratio=3., drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.RWKV = RWKV(n_embd=dim, n_layer=n_layer, layer_id=layer_id, drop_path=drop_path, hidden_rate=hidden_rate,
                        init_values=init_values, post_norm=post_norm, key_norm=key_norm, with_cp=with_cp)
        # self.RWKV = RWKV(n_embd=dim, n_layer=n_layer, layer_id=layer_id, hidden_rate=4, init_mode='local', key_norm=key_norm)
    def forward(self, x, patch_resolution):
        # RWKV
        x = self.RWKV(x, patch_resolution)
        
        # FFN
        return x
        

class BasicLayer(nn.Module): 
    def __init__(self, dim, input_resolution, depth, n_layer, layer_id, drop_path=0., hidden_rate=4,
                init_values=None, post_norm=False, key_norm=False, with_cp=False,
                mlp_ratio=3., drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # It will be used for the RWKV forward
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            RWKVBlock(dim=dim, input_resolution=input_resolution, n_layer=n_layer, layer_id=layer_id,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, 
                    hidden_rate=hidden_rate, init_values=init_values,
                    post_norm=post_norm, key_norm=key_norm, with_cp=with_cp,
                    mlp_ratio=mlp_ratio, drop=drop, act_layer=act_layer, norm_layer=norm_layer)
            for i in range(depth)])
        
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
        

    def forward(self, x, x_size):   # The use of x_size needs to be changed!!
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class RWKVB(nn.Module):
    def __init__(self, dim, input_resolution, depth, n_layer, layer_id, drop_path=0., hidden_rate=4,
                init_values=None, post_norm=False, key_norm=False, with_cp=False,
                mlp_ratio=3., drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                img_size=224, patch_size=4, resi_connection='1conv'):
        super(RWKVB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.block1 = RWKVBlock(dim=dim, input_resolution=input_resolution, n_layer=n_layer, layer_id=layer_id,
                    drop_path=drop_path[0], 
                    hidden_rate=hidden_rate, init_values=init_values,
                    post_norm=post_norm, key_norm=key_norm, with_cp=with_cp,
                    mlp_ratio=mlp_ratio, drop=drop, act_layer=act_layer, norm_layer=norm_layer)
        
        self.block2 = RWKVBlock(dim=dim//2, input_resolution=input_resolution, n_layer=n_layer, layer_id=layer_id,
                    drop_path=drop_path[0], 
                    hidden_rate=hidden_rate, init_values=init_values,
                    post_norm=post_norm, key_norm=key_norm, with_cp=with_cp,
                    mlp_ratio=mlp_ratio, drop=drop, act_layer=act_layer, norm_layer=norm_layer)
        
        self.block3 = RWKVBlock(dim=dim//4, input_resolution=input_resolution, n_layer=n_layer, layer_id=layer_id,
                    drop_path=drop_path[0], 
                    hidden_rate=hidden_rate, init_values=init_values,
                    post_norm=post_norm, key_norm=key_norm, with_cp=with_cp,
                    mlp_ratio=mlp_ratio, drop=drop, act_layer=act_layer, norm_layer=norm_layer)

        self.conv1 = MBConv(dim, expansion_rate=2)
        self.conv2 = MBConv(dim//2, expansion_rate=2)
        self.conv3 = MBConv(dim//4, expansion_rate=2)
        self.mixer = nn.Conv2d(dim, dim, 1, 1, 0)
        self.cca = CCALayer(dim)
        self.esa = ESA(dim, nn.Conv2d)


        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        

    def forward(self, x, x_size):  # x_size need to be used
        H, W = x_size
        shortcut = x
        shortcut = rearrange(shortcut, 'b (h w) c -> b c h w', h=H, w=W)
        x1 = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        x1 = self.conv1(x1)
        x1 = rearrange(x1, 'b c h w -> b (h w) c')
        x1 = self.block1(x1, x_size)
        x2, res1 = torch.chunk(x1, 2, dim=2)

        x2 = rearrange(x2, 'b (h w) c -> b c h w', h=H, w=W)
        x2 = self.conv2(x2)
        x2 = rearrange(x2, 'b c h w -> b (h w) c')
        x2 = self.block2(x2, x_size)
        x3, res2 = torch.chunk(x2, 2, dim=2)

        x3 = rearrange(x3, 'b (h w) c -> b c h w', h=H, w=W)
        x3 = self.conv3(x3)
        x3 = rearrange(x3, 'b c h w -> b (h w) c')
        x3 = self.block3(x3, x_size)

        comb = torch.cat([res1, res2, x3], dim=2) 
        comb = rearrange(comb, 'b (h w) c -> b c h w', h=H, w=W)
        comb = self.mixer(comb)
        comb = self.cca(comb)
        comb = comb + shortcut
        comb = self.esa(comb)
        comb = rearrange(comb, 'b c h w -> b (h w) c')
        return comb

class RWKVIR(nn.Module):
    def __init__(self, img_size=64, patch_size=1, in_chans=3,
                 embed_dim=64, depths=[6, 6, 6, 6], mlp_ratio=3., hidden_rate=4,
                 init_values=None, post_norm=False, key_norm=False, with_cp=False,
                 drop_rate=0.0, drop_path_rate=0.1, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, img_range=1., upsampler='', resi_connection='1conv',
                 **kwargs):
        super(RWKVIR, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        # self.window_size = window_size

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

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

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build RWKV blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RWKVB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=depths[i_layer],
                         drop=drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection,
                         n_layer=self.num_layers,
                         layer_id=i_layer, 
                         hidden_rate=hidden_rate,
                         init_values=init_values, post_norm=post_norm, 
                         key_norm=key_norm, with_cp=with_cp,
                         mlp_ratio=self.mlp_ratio,
                         act_layer=nn.GELU, norm_layer=norm_layer, 
                         downsample=None, use_checkpoint=use_checkpoint,
                         )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                             nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                             nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                             nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                             nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        #####################################################################################################
        ################################ 3, high quality image reconstruction ################################

            # for lightweight SR (to save parameters)
        self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch, (patches_resolution[0], patches_resolution[1]))


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
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
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        
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

        # shallow feature extraction
        x = self.conv_first(x)

        # deep feature extraction + conv
        x = self.conv_after_body(self.forward_features(x)) + x
        # high quality image reconstruction
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


# from fvcore.nn import FlopCountAnalysis, parameter_count_table
# import numpy as np

if __name__ == '__main__':
    upscale = 2
    height = 100
    width = 200
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RWKVIR(img_size=(height, width), depths=[6, 6, 6, 6, 6], 
                   hidden_rate=3, patch_size=8,
                   img_range=1, embed_dim=64, upscale=upscale,
                   upsampler='pixelshuffledirect', resi_connection='3conv',).to(device)
    
    
    x = torch.randn((2, 3, height, width)).to(device)
    pred = model(x)
    print(pred.shape)
    print("x on:", x.device)
    print("model param on:", next(model.parameters()).device)
    print(f"number of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
#     img_lq = np.empty([3,64,64], dtype = float, order = 'C')
#     x = torch.from_numpy(img_lq).float().unsqueeze(0).to('cuda')
#     flops = FlopCountAnalysis(model, x)
#     print("FLOPs: ", flops.total())
#     print(parameter_count_table(model))
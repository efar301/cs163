# adapted from
# https://github.com/YuzhenD/Resyn/blob/master/basicsr/module/base/vrwkv6.py 

from typing import Sequence
import math, os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.checkpoint as cp
from einops import rearrange

from timm.layers import DropPath


logger = logging.getLogger(__name__)


T_MAX = 4096 
HEAD_SIZE = 8

from torch.utils.cpp_extension import load

wkv6_cuda = load(name="wkv6",
                 sources=["cuda_v6/wkv6_op.cpp", 
                          "cuda_v6/wkv6_cuda.cu"],
                 verbose=True, extra_cuda_cflags=["-allow-unsupported-compiler", "-res-usage", "--use_fast_math",
                 "-O3", "-Xptxas=-O3", 
                 "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", 
                 f"-D_T_={T_MAX}"])

class WKV_6(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, H, r, k, v, w, u):
        with torch.no_grad():
            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H
            r32= r.float().contiguous()
            k32= k.float().contiguous()
            v32= v.float().contiguous()
            ew32 = (-torch.exp(w.float())).contiguous()
            u32 = u.float().contiguous()
            ctx.save_for_backward(r32, k32, v32, ew32, u32)
            y32 = torch.empty((B, T, C), device=r.device, dtype=torch.float32)
            wkv6_cuda.forward(B, T, C, H, r32, k32, v32, ew32, u32, y32)
            return y32.to(r.dtype)



    @staticmethod
    def backward(ctx, gy):
        with torch.no_grad():
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            r32, k32, v32, ew32, u32 = ctx.saved_tensors
            out_dtype = gy.dtype

            gy32 = gy.float().contiguous()
            gr32 = torch.empty((B, T, C), device=gy.device, dtype=torch.float32)
            gk32 = torch.empty_like(gr32)
            gv32 = torch.empty_like(gr32)
            gw32 = torch.empty_like(gr32)
            gu32 = torch.empty((B, C), device=gy.device, dtype=torch.float)

            wkv6_cuda.backward(B, T, C, H, r32, k32, v32, ew32, u32, gy32, gr32, gk32, gv32, gw32, gu32)
            gu32 = gu32.sum(0).view(H, C//H)

            return (None, None, None, None, gr32.to(out_dtype), gk32.to(out_dtype), gv32.to(out_dtype), gw32.to(out_dtype), gu32.to(out_dtype))


def RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u):
    return WKV_6.apply(B, T, C, H, r, k, v, w, u)


class OmniShift(nn.Module):
    def __init__(self, dim):
        super(OmniShift, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, groups=dim, bias=False)
        self.conv3x3 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.conv5x5 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, padding=2, groups=dim, bias=False) 
        self.alpha = nn.Parameter(torch.randn(4), requires_grad=True) 

    def forward(self, x):
        out1x1 = self.conv1x1(x)
        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(x) 
        
        out = self.alpha[0]*x + self.alpha[1]*out1x1 + self.alpha[2]*out3x3 + self.alpha[3]*out5x5
        return out

class VRWKV_SpatialMix_V6(nn.Module):
    def __init__(self, n_embd, n_head, n_layer, layer_id, with_cls_token=False, with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.attn_sz = n_embd

        self.n_head = n_head
        self.head_size = self.attn_sz // self.n_head
        assert self.head_size == HEAD_SIZE
        self.device = None
        self._init_weights()
        self.with_cls_token = with_cls_token
        self.shift_func = OmniShift(n_embd)


        self.key = nn.Linear(self.n_embd, self.attn_sz, bias=False)
        self.value = nn.Linear(self.n_embd, self.attn_sz, bias=False)
        self.receptance = nn.Linear(self.n_embd, self.attn_sz, bias=False)
        self.output = nn.Linear(self.attn_sz, n_embd, bias=False)
        self.output.init_scale = 0

        self.ln_x = nn.LayerNorm(self.attn_sz)
        self.with_cp = with_cp
        

    def _init_weights(self):
        with torch.no_grad():
            ratio_0_to_1 = self.layer_id / (self.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (self.layer_id / self.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, self.n_embd)
            for i in range(self.n_embd):
                ddd[0, 0, i] = i / self.n_embd

            # # fancy time_mix
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            # fancy time_decay
            decay_speed = torch.ones(self.attn_sz)
            for n in range(self.attn_sz):
                decay_speed[n] = -6 + 5 * (n / (self.attn_sz - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,self.attn_sz))

            TIME_DECAY_EXTRA_DIM = 16
            self.time_decay_w1 = nn.Parameter(torch.zeros(self.n_embd, TIME_DECAY_EXTRA_DIM))
            self.time_decay_w2 = nn.Parameter(torch.zeros(TIME_DECAY_EXTRA_DIM, self.attn_sz).uniform_(-1e-2, 1e-2))

            tmp = torch.zeros(self.attn_sz)
            for n in range(self.attn_sz):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (self.attn_sz - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))


    def jit_func(self, x, patch_resolution):
        # Mix x with the previous timestep to produce xk, xv, xr
        B, T, C = x.size()
        H, W = patch_resolution

        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        xx = self.shift_func(x) - x
        x = rearrange(x, 'b c h w -> b (h w) c', h=H, w=W)
        xx = rearrange(xx, 'b c h w -> b (h w) c', h=H, w=W)

        xw = x + xx * (self.time_maa_w)
        xk = x + xx * (self.time_maa_k)
        xv = x + xx * (self.time_maa_v)
        xr = x + xx * (self.time_maa_r)

        # xr, xk, xv, xw = x, x, x, x

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)

        ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        # [B, T, C]
        w = self.time_decay + ww

        return r, k, v, w
    
    def _scan(self, x, patch_resolution):
        B, T, C = x.size()
        H = self.n_head
        r, k, v, w = self.jit_func(x, patch_resolution)
        y = RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, self.time_faaaa)
        return y

    def jit_func_2(self, x):
        x = self.ln_x(x)
        x = self.output(x)
        return x

    def forward(self, x, patch_resolution=None):
        def _inner_forward(x):
            B, T, C = x.size()
            H, W = patch_resolution
            self.device = x.device

            # patch_resolution_run = patch_resolution
            # if self.layer_id % 2 == 0:
            #     x = rearrange(x, 'b (h w) c -> b (w h) c', h=H, w=W)
            #     patch_resolution_run = W, H
            
            
            # shortcut = x
            # r, k, v, w = self.jit_func(x, patch_resolution_run)
            # x = RUN_CUDA_RWKV6(B, T, C, self.n_head, r, k, v, w, u=self.time_faaaa)

            # if self.layer_id % 2 == 0:
            #     x = rearrange(x, 'b (w h) c -> b (h w) c', h=H, w=W)

            # scan rows -> transpose -> scan cols -> avg 
            row = self._scan(x, (H, W))
            col = self._scan(rearrange(x, 'b (h w) c -> b (w h) c', h=H, w=W), (W, H))
            col = rearrange(col, 'b (w h) c -> b (h w) c', h=H, w=W)
            out = 0.5 * (row + col)
            
            return self.jit_func_2(out)
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, n_head, n_layer, layer_id, hidden_rate=3, 
                 with_cls_token=False, with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.attn_sz = n_embd
        self.n_head = n_head
        self.head_size = self.attn_sz // self.n_head
        assert self.head_size == HEAD_SIZE
        self.with_cp = with_cp
        self._init_weights()
        self.with_cls_token = with_cls_token
        self.shift_func = OmniShift(n_embd)
        hidden_sz = int(hidden_rate * n_embd)
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)
        self.receptance.init_scale = 0
        self.value.init_scale = 0        


    def _init_weights(self):
        with torch.no_grad():
            ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer)) # 1 to ~0
            x = torch.ones(1, 1, self.n_embd)
            for i in range(self.n_embd):
                x[0, 0, i] = i / self.n_embd
            self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.spatial_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))


    def forward(self, x, patch_resolution=None):
        def _inner_forward(x):
            B, T, C = x.size()
            H, W = patch_resolution

            x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
            xx = self.shift_func(x) - x
            x = rearrange(x, 'b c h w -> b (h w) c', h=H, w=W)
            xx = rearrange(xx, 'b c h w -> b (h w) c', h=H, w=W)

            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
            
            k = self.key(xk)
            k = torch.square(torch.relu(k))
            kv = self.value(k)
            x = torch.sigmoid(self.receptance(xr)) * kv
            return x
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embd, n_head, n_layer, layer_id, hidden_rate=4,
                 post_norm=False, key_norm=False, with_cls_token=False, with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)

        self.att = VRWKV_SpatialMix_V6(n_embd, n_head, n_layer, layer_id, with_cls_token=with_cls_token)

        self.ffn = VRWKV_ChannelMix(n_embd, n_head, n_layer, layer_id, hidden_rate=hidden_rate, 
                                    with_cls_token=with_cls_token)

        self.post_norm = post_norm
        
        self.gamma1 = nn.Parameter(torch.ones(n_embd), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones(n_embd), requires_grad=True)
        self.with_cp = with_cp

    def forward(self, x, patch_resolution=None):
        def _inner_forward(x):
            if self.layer_id == 0:
                x = self.ln0(x)
            if self.post_norm:
                x = self.gamma1*x + self.att(self.ln1(x), patch_resolution)
                x = self.gamma2*x + self.ffn(self.ln2(x), patch_resolution)       
            else:
                x = self.gamma1*x + self.att(self.ln1(x), patch_resolution)
                x = self.gamma2*x + self.ffn(self.ln2(x), patch_resolution)
            return x
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


def resize_pos_embed(pos_embed,
                     src_shape,
                     dst_shape,
                     mode='bicubic',
                     num_extra_tokens=1):
    """Resize pos_embed weights.

    Args:
        pos_embed (torch.Tensor): Position embedding weights with shape
            [1, L, C].
        src_shape (tuple): The resolution of downsampled origin training
            image, in format (H, W).
        dst_shape (tuple): The resolution of downsampled new training
            image, in format (H, W).
        mode (str): Algorithm used for upsampling. Choose one from 'nearest',
            'linear', 'bilinear', 'bicubic' and 'trilinear'.
            Defaults to 'bicubic'.
        num_extra_tokens (int): The number of extra tokens, such as cls_token.
            Defaults to 1.

    Returns:
        torch.Tensor: The resized pos_embed of shape [1, L_new, C]
    """
    if src_shape[0] == dst_shape[0] and src_shape[1] == dst_shape[1]:
        return pos_embed
    assert pos_embed.ndim == 3, 'shape of pos_embed must be [1, L, C]'
    _, L, C = pos_embed.shape
    src_h, src_w = src_shape
    assert L == src_h * src_w + num_extra_tokens, \
        f"The length of `pos_embed` ({L}) doesn't match the expected " \
        f'shape ({src_h}*{src_w}+{num_extra_tokens}). Please check the' \
        '`img_size` argument.'
    extra_tokens = pos_embed[:, :num_extra_tokens]

    src_weight = pos_embed[:, num_extra_tokens:]
    src_weight = src_weight.reshape(1, src_h, src_w, C).permute(0, 3, 1, 2)

    # The cubic interpolate algorithm only accepts float32
    dst_weight = F.interpolate(
        src_weight.float(), size=dst_shape, align_corners=False, mode=mode)
    dst_weight = torch.flatten(dst_weight, 2).transpose(1, 2)
    dst_weight = dst_weight.to(src_weight.dtype)

    return torch.cat((extra_tokens, dst_weight), dim=1)



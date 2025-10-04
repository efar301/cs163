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
            assert r.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert w.is_contiguous()
            assert u.is_contiguous()
            ew = (-torch.exp(w.float())).contiguous()
            ctx.save_for_backward(r, k, v, ew, u)
            y = torch.empty((B, T, C), device=r.device, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            wkv6_cuda.forward(B, T, C, H, r, k, v, ew, u, y)
            return y

    @staticmethod
    def backward(ctx, gy):
        with torch.no_grad():
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            assert gy.is_contiguous()
            r, k, v, ew, u = ctx.saved_tensors
            gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            wkv6_cuda.backward(B, T, C, H, r, k, v, ew, u, gy, gr, gk, gv, gw, gu)
            gu = torch.sum(gu, 0).view(H, C//H)
            return (None, None, None, None, gr, gk, gv, gw, gu)

def RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u):
    return WKV_6.apply(B, T, C, H, r, k, v, w, u)

def q_shift_multihead(input, shift_pixel=1, head_dim=HEAD_SIZE, 
                      patch_resolution=None, with_cls_token=False):
    B, N, C = input.shape
    assert C % head_dim == 0
    assert head_dim % 4 == 0
    if with_cls_token:
        cls_tokens = input[:, [-1], :]
        input = input[:, :-1, :]
    input = input.transpose(1, 2).reshape(
        B, -1, head_dim, patch_resolution[0], patch_resolution[1])  # [B, n_head, head_dim H, W]
    B, _, _, H, W = input.shape
    output = torch.zeros_like(input)
    output[:, :, 0:int(head_dim):4, :, shift_pixel:W] = \
        input[:, :, 0:int(head_dim):4, :, 0:W-shift_pixel]
    output[:, :, 1:int(head_dim):4, :, 0:W-shift_pixel] = \
        input[:, :, 1:int(head_dim):4, :, shift_pixel:W]
    output[:, :, 2:int(head_dim):4, shift_pixel:H, :] = \
        input[:, :, 2:int(head_dim):4, 0:H-shift_pixel, :]
    output[:, :, 3:int(head_dim):4, 0:H-shift_pixel, :] = \
        input[:, :, 3:int(head_dim):4, shift_pixel:H, :]
    if with_cls_token:
        output = output.reshape(B, C, N-1).transpose(1, 2)
        output = torch.cat((output, cls_tokens), dim=1)
    else:
        output = output.reshape(B, C, N).transpose(1, 2)
    return output

class StarShift(nn.Module):
    def __init__(self, channels, n_head, head_dim, bottleneck_ratio=4):
        super().__init__()
        self.bottleneck_channels = channels // bottleneck_ratio
        def dilation(factor):
            return nn.Sequential(
                nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, 
                dilation=factor, padding=factor, groups=channels, bias=False),
                nn.SiLU()
            )
        


        self.n_head = n_head
        self.head_dim = head_dim
        self.conv1 = dilation(1)
        self.conv2 = dilation(2)
        self.conv3 = dilation(3)
        self.mixer = nn.Sequential(
            nn.Conv2d(in_channels=3*channels, out_channels=self.bottleneck_channels, kernel_size=1, bias=False),
            nn.Conv2d(in_channels=self.bottleneck_channels, out_channels=channels, kernel_size=1, bias=False),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x, patch_resolution):
        B, T, C = x.shape
        H, W = patch_resolution
        feat = x.reshape(B, H, W, C).permute(0, 3, 1 ,2).contiguous()
        y1, y2, y3 = self.conv1(feat), self.conv2(feat), self.conv3(feat)
        y = torch.cat((y1, y2, y3), dim=1)
        y = self.mixer(y)
        out = y.permute(0, 2, 3, 1).contiguous().reshape(B, T, C)
        return out

        

class VRWKV_SpatialMix_V6(nn.Module):
    def __init__(self, n_embd, n_head, n_layer, layer_id, shift_mode='q_shift_multihead',
                shift_pixel=1, init_mode='fancy', key_norm=False, with_cls_token=False, 
                 with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.attn_sz = n_embd

        self.n_head = n_head
        self.head_size = self.attn_sz // self.n_head
        assert self.head_size == HEAD_SIZE
        self.device = None
        self._init_weights(init_mode)
        self.with_cls_token = with_cls_token
        self.shift_pixel = shift_pixel
        # self.shift_mode = shift_mode
        # self.shift_func = eval(shift_mode)
        self.shift_func = StarShift(self.n_embd, self.n_head, self.head_size)


        self.key = nn.Linear(self.n_embd, self.attn_sz, bias=False)
        self.value = nn.Linear(self.n_embd, self.attn_sz, bias=False)
        self.receptance = nn.Linear(self.n_embd, self.attn_sz, bias=False)
        # if key_norm:
        #     self.key_norm = nn.LayerNorm(n_embd)
        # else:
        #     self.key_norm = None
        self.output = nn.Linear(self.attn_sz, n_embd, bias=False)
        self.output.init_scale = 0

        self.ln_x = nn.LayerNorm(self.attn_sz)
        self.with_cp = with_cp

        # self.conv_scale = nn.Parameter(torch.ones(n_embd))
        # self.conv1b3 = nn.Sequential(
		# 	nn.Conv2d(in_channels=n_embd, out_channels=n_embd, kernel_size=1, stride=1),
		# 	nn.InstanceNorm2d(n_embd),
		# 	nn.GELU(),
		# )
        # self.conv1a3 = nn.Sequential(
		# 	nn.Conv2d(in_channels=n_embd, out_channels=n_embd, kernel_size=1, stride=1),
		# 	nn.InstanceNorm2d(n_embd),
		# 	nn.GELU(),
		# )
        # self.conv33 = nn.Sequential(
		# 	nn.Conv2d(in_channels=n_embd, out_channels=n_embd, kernel_size=3, stride=1, padding=1, bias=False,
		# 			  groups=n_embd),
		# 	nn.InstanceNorm2d(n_embd),
		# 	nn.SiLU(),
		# )
        

    def _init_weights(self, init_mode):
        if init_mode=='fancy':
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
        else:
            raise NotImplementedError

    def jit_func(self, x, patch_resolution):
        # Mix x with the previous timestep to produce xk, xv, xr
        B, T, C = x.size()

        # xx = self.shift_func(x, self.shift_pixel, patch_resolution=patch_resolution, 
        #                      with_cls_token=self.with_cls_token) - x
        xx = self.shift_func(x, patch_resolution) - x

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

    def jit_func_2(self, x):
        x = self.ln_x(x)
        x = self.output(x)
        return x

    def forward(self, x, patch_resolution=None):
        def _inner_forward(x):
            B, T, C = x.size()
            self.device = x.device
            
            # shortcut = x
            r, k, v, w = self.jit_func(x, patch_resolution)
            x = RUN_CUDA_RWKV6(B, T, C, self.n_head, r, k, v, w, u=self.time_faaaa)
            
            # input_conv = shortcut.reshape([B,patch_resolution[0],patch_resolution[1],C]).permute(0, 3, 1, 2).contiguous()
            # out_33 = self.conv1a3(self.conv33(self.conv1b3(input_conv)))
            # output = out_33.permute(0, 2, 3, 1).contiguous()
            # x = output.reshape([B,T,C])*self.conv_scale + x
            
            return self.jit_func_2(x)
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, n_head, n_layer, layer_id, shift_mode='q_shift_multihead',
                 shift_pixel=1, hidden_rate=4, init_mode='fancy', key_norm=False, 
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
        self._init_weights(init_mode)
        self.with_cls_token = with_cls_token
        self.shift_pixel = shift_pixel
        # self.shift_mode = shift_mode
        # self.shift_func = eval(shift_mode)
        self.shift_func = StarShift(self.n_embd, self.n_head, self.head_size)

        hidden_rate = 3

        hidden_sz = hidden_rate * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        # if key_norm:
        #     self.key_norm = nn.LayerNorm(hidden_sz)
        # else:
        #     self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)
        self.receptance.init_scale = 0
        self.value.init_scale = 0
        
        # self.conv_scale = nn.Parameter(torch.ones(n_embd))
        # self.conv1b3 = nn.Sequential(
		# 	nn.Conv2d(in_channels=n_embd, out_channels=n_embd, kernel_size=1, stride=1),
		# 	nn.InstanceNorm2d(n_embd),
		# 	nn.GELU(),
		# )
        # self.conv1a3 = nn.Sequential(
		# 	nn.Conv2d(in_channels=n_embd, out_channels=n_embd, kernel_size=1, stride=1),
		# 	nn.InstanceNorm2d(n_embd),
		# 	nn.GELU(),
		# )
        # self.conv33 = nn.Sequential(
		# 	nn.Conv2d(in_channels=n_embd, out_channels=n_embd, kernel_size=3, stride=1, padding=1, bias=False,
		# 			  groups=n_embd),
		# 	nn.InstanceNorm2d(n_embd),
		# 	nn.SiLU(),
		# )

    def _init_weights(self, init_mode):
        if init_mode == 'fancy':
            with torch.no_grad(): # fancy init of time_mix
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer)) # 1 to ~0
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                # self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                # self.spatial_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_k = nn.Parameter(torch.ones(1, 1, self.n_embd) * 0.5)
                self.spatial_mix_r = nn.Parameter(torch.ones(1, 1, self.n_embd) * 0.5)
        else:
            raise NotImplementedError

    def forward(self, x, patch_resolution=None):
        def _inner_forward(x):
            # xx = self.shift_func(x, self.shift_pixel, patch_resolution=patch_resolution,
            #                      with_cls_token=self.with_cls_token)
            xx = self.shift_func(x, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
            # xx = self.shift_func(x, patch_resolution)
            B, T, C = x.size()
            # shortcut = x
            # input_conv = shortcut.reshape([B,patch_resolution[0],patch_resolution[1],C]).permute(0, 3, 1, 2).contiguous()
            # out_33 = self.conv1a3(self.conv33(self.conv1b3(input_conv)))
            # output = out_33.permute(0, 2, 3, 1).contiguous()
            # x = output.reshape([B,T,C])*self.conv_scale + x

            # xr, xk = x, x

            k = self.key(xk)
            k = torch.square(torch.relu(k))
            # if self.key_norm is not None:
            #     k = self.key_norm(k)
            kv = self.value(k)
            x = torch.sigmoid(self.receptance(xr)) * kv
            return x
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embd, n_head, n_layer, layer_id, shift_mode='q_shift_multihead',
                 shift_pixel=1, drop_path=0., hidden_rate=4, init_mode='fancy',
                 post_norm=False, key_norm=False, with_cls_token=False,
                 with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)

        self.att = VRWKV_SpatialMix_V6(n_embd, n_head, n_layer, layer_id, shift_mode,
                                       shift_pixel, init_mode, key_norm=key_norm,
                                       with_cls_token=with_cls_token)

        self.ffn = VRWKV_ChannelMix(n_embd, n_head, n_layer, layer_id, shift_mode,
                                    shift_pixel, hidden_rate, init_mode, key_norm=key_norm,
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
                x = self.gamma1*x + self.drop_path(self.ln1(self.att(x, patch_resolution)))
                x = self.gamma2*x + self.drop_path(self.ln2(self.ffn(x, patch_resolution)))       
            else:
                x = self.gamma1*x + self.drop_path(self.att(self.ln1(x), patch_resolution))
                x = self.gamma2*x + self.drop_path(self.ffn(self.ln2(x), patch_resolution))
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



# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence
import warnings
import math

import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.checkpoint as cp
from einops import rearrange

from timm.layers import DropPath

T_MAX = 4096 # increase this if your ctx_len is long [NOTE: TAKES LOTS OF VRAM!]
# it's possible to go beyond CUDA limitations if you slice the ctx and pass the hidden state in each slice

from torch.utils.cpp_extension import load
wkv_cuda = load(name="wkv", 
                sources=["cuda_v4/wkv_op.cpp", 
                          "cuda_v4/wkv_cuda.cu"],
                verbose=True, extra_cuda_cflags=['-res-usage', '--maxrregcount=60', '--use_fast_math', '-O3', '-Xptxas=-O3', f'-DTmax={T_MAX}'])


class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        # assert T <= T_MAX   # Try to make it unused 
        assert B * C % min(C, 1024) == 0

        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()

        ctx.save_for_backward(w, u, k, v)
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        wkv_cuda.backward(B, T, C,
                          w.float().contiguous(),
                          u.float().contiguous(),
                          k.float().contiguous(),
                          v.float().contiguous(),
                          gy.float().contiguous(),
                          gw, gu, gk, gv)
        if half_mode:
            gw = torch.sum(gw.half(), dim=0)
            gu = torch.sum(gu.half(), dim=0)
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            gw = torch.sum(gw.bfloat16(), dim=0)
            gu = torch.sum(gu.bfloat16(), dim=0)
            return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)


def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())


def q_shift(input, shift_pixel=1, gamma=1/4, patch_resolution=None):
    assert gamma <= 1/4
    B, N, C = input.shape
    input = input.transpose(1, 2).reshape(B, C, patch_resolution[0], patch_resolution[1])
    B, C, H, W = input.shape
    output = torch.zeros_like(input)
    output[:, 0:C:4, :, shift_pixel:W] = input[:, 0:C:4, :, 0:W-shift_pixel]
    output[:, 1:C:4, :, 0:W-shift_pixel] = input[:, 1:C:4, :, shift_pixel:W]
    output[:, 2:C:4, shift_pixel:H, :] = input[:, 2:C:4, 0:H-shift_pixel, :]
    output[:, 3:C:4, 0:H-shift_pixel, :] = input[:, 3:C:4, shift_pixel:H, :]
    # output[:, int(C*gamma*4):, ...] = input[:, int(C*gamma*4):, ...]
    return output.flatten(2).transpose(1, 2)


class VRWKV_SpatialMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=1, init_mode='fancy',
                 key_norm=False, with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd
        self._init_weights(init_mode)
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_pixel > 0:
            self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_v = None
            self.spatial_mix_r = None

        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(n_embd)
        else:
            self.key_norm = None
        self.output = nn.Linear(attn_sz, n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

        self.recurrence = 2 
        
        self.conv_scale = nn.Parameter(torch.ones(n_embd))
        # self.alpha = nn.Parameter(torch.randn(4), requires_grad=True) 
        
        self.conv1b3 = nn.Sequential(
			nn.Conv2d(in_channels=n_embd, out_channels=n_embd, kernel_size=1, stride=1),
			nn.InstanceNorm2d(n_embd),
			nn.GELU(),
		)
        self.conv1a3 = nn.Sequential(
			nn.Conv2d(in_channels=n_embd, out_channels=n_embd, kernel_size=1, stride=1),
			nn.InstanceNorm2d(n_embd),
			nn.GELU(),
		)
        self.conv33 = nn.Sequential(
			nn.Conv2d(in_channels=n_embd, out_channels=n_embd, kernel_size=3, padding=1, bias=False,
					  groups=n_embd),
			nn.InstanceNorm2d(n_embd),
			nn.SiLU(),
		)
        
        self.with_cp = with_cp
    
    def _init_weights(self, init_mode):
        if init_mode=='fancy':
            with torch.no_grad(): # fancy init
                ratio_0_to_1 = (self.layer_id / (self.n_layer - 1)) # 0 to 1
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer)) # 1 to ~0
                
                # fancy time_decay
                decay_speed = torch.ones(self.n_embd)
                for h in range(self.n_embd):
                    decay_speed[h] = -5 + 8 * (h / (self.n_embd-1)) ** (0.7 + 1.3 * ratio_0_to_1)
                self.spatial_decay = nn.Parameter(decay_speed)
                self.spatial_decay_2 = nn.Parameter(decay_speed)

                # fancy time_first
                zigzag = (torch.tensor([(i+1)%3 - 1 for i in range(self.n_embd)]) * 0.5)
                self.spatial_first = nn.Parameter(torch.ones(self.n_embd) * math.log(0.3) + zigzag)
                self.spatial_second = nn.Parameter(torch.ones(self.n_embd) * math.log(0.3) + zigzag)
                
                # fancy time_mix
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
                self.spatial_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))
        elif init_mode=='local':
            self.spatial_decay = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_first = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))
        elif init_mode=='global':
            self.spatial_decay = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_first = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        else:
            raise NotImplementedError
    
    def jit_func(self, x, patch_resolution):
        # Mix x with the previous timestep to produce xk, xv, xr
        B, T, C = x.size()
        if self.shift_pixel > 0:
            xx = self.shift_func(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xv = x
            xr = x

        # Use xk, xv, xr to produce k, v, r
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        return sr, k, v

    def forward(self, x, patch_resolution):
        B, T, C = x.size()
        self.device = x.device
        
        shortcut = x
        input_conv = shortcut.reshape([B,patch_resolution[0],patch_resolution[1],C]).permute(0, 3, 1, 2).contiguous()
        # input_mid = self.alpha[0]*input_conv + self.alpha[1]*self.conv1a3(input_conv) + self.alpha[2]*self.conv33(input_conv) + self.alpha[3]*self.conv55(input_conv)
        out_33 = self.conv1b3(self.conv33(self.conv1a3(input_conv)))
        output = out_33.permute(0, 2, 3, 1).contiguous()
        x = output.reshape([B,T,C])*self.conv_scale + x

        sr, k, v = self.jit_func(x, patch_resolution)
        
        # for j in range(self.recurrence):
        if self.layer_id%2==0:
            v = RUN_CUDA(B, T, C, self.spatial_decay / T, self.spatial_first / T, k, v)  # Here v is the rwkv
        else:
            h, w = patch_resolution
            k = rearrange(k, 'b (h w) c -> b (w h) c', h=h, w=w)
            v = rearrange(v, 'b (h w) c -> b (w h) c', h=h, w=w)
            v = RUN_CUDA(B, T, C, self.spatial_decay / T, self.spatial_first / T, k, v)
            v = rearrange(v, 'b (w h) c -> b (h w) c', h=h, w=w) 
        
        rwkv = v        
        if self.key_norm is not None:
            rwkv = self.key_norm(rwkv)
        rwkv = sr * rwkv + output.reshape([B,T,C])*self.conv_scale
        rwkv = self.output(rwkv)
        return rwkv


class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=1, hidden_rate=4, init_mode='fancy',
                 key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        # self._init_weights(init_mode)
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_pixel > 0:
            self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_r = None

        hidden_sz = hidden_rate * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0
        
        self.conv1x1 = nn.Conv2d(in_channels=n_embd, out_channels=n_embd, kernel_size=1, groups=n_embd, bias=False)
        self.conv3x3 = nn.Conv2d(in_channels=n_embd, out_channels=n_embd, kernel_size=3, padding=1, groups=n_embd, bias=False)
        self.conv5x5 = nn.Conv2d(in_channels=n_embd, out_channels=n_embd, kernel_size=5, padding=2, groups=n_embd, bias=False) 
        self.alpha = nn.Parameter(torch.randn(4), requires_grad=True)
        
        self.with_cp = False
    
    def _init_weights(self, init_mode):
        if init_mode == 'fancy':
            with torch.no_grad(): # fancy init of time_mix
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer)) # 1 to ~0
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        elif init_mode == 'local':
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))
        elif init_mode == 'global':
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        else:
            raise NotImplementedError

    def forward(self, x, patch_resolution):
        B, T, C = x.size()
        x = x.reshape([B,patch_resolution[0],patch_resolution[1],C]).permute(0, 3, 1, 2).contiguous()
        out1x1 = self.conv1x1(x)
        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(x) 
        out = self.alpha[0]*x + self.alpha[1]*out1x1 + self.alpha[2]*out3x3 + self.alpha[3]*out5x5
        out = out.permute(0, 2, 3, 1).contiguous()
        x = out.reshape([B,T,C])
        
        xk = x
        xr = x

        k = self.key(xk)
        k = torch.square(torch.relu(k))
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)

        rkv = torch.sigmoid(self.receptance(xr)) * kv
        return rkv


class Block(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift', init_mode='local',
                 channel_gamma=1/4, shift_pixel=1, drop_path=0., hidden_rate=4,
                 init_values=None, post_norm=False, key_norm=False, with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)

        self.att = VRWKV_SpatialMix(n_embd, n_layer,  layer_id, shift_mode, channel_gamma, 
                                   shift_pixel, init_mode=init_mode, key_norm=key_norm)
        self.ffn = VRWKV_ChannelMix(n_embd, n_layer, layer_id, shift_mode, channel_gamma, 
                                   shift_pixel, hidden_rate, init_mode=init_mode, key_norm=key_norm)
        # self.ffn = CAB(n_embd)
        
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
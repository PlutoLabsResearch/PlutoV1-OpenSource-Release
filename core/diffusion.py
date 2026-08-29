from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.codec import conv_cls


class TimeEmbedding(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),nn.SiLU(), nn.Linear(dim * 4,dim)
        )

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half,device=t.device) * (9.21 / max(half - 1,1))
        )
        ang = t[:, None] * freqs[None]
        return self.net(torch.cat([ang.sin(), ang.cos()], dim=-1))


class ConditionedBlock(nn.Module):

    def __init__(self,ndim, channels,cond_dim, tdim):
        super().__init__()
        Conv = conv_cls(ndim)
        k = 3 if ndim > 1 else 7
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = Conv(channels, channels, k, padding=k // 2)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = Conv(channels, channels, k, padding=k // 2)
        self.t_proj = nn.Linear(tdim, channels * 2)
        self.c_proj = Conv(cond_dim, channels * 2, 1)

    def forward(self, x, t_emb, cond):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)

        shape = (h.shape[0], -1) + (1,) * (h.dim() - 2)
        ts,tb = self.t_proj(t_emb).view(*shape, 2).unbind(-1) \
            if False else self.t_proj(t_emb).chunk(2, dim=-1)
        ts = ts.view(h.shape[0], -1,*([1] * (h.dim() - 2)))
        tb = tb.view(h.shape[0],-1, *([1] * (h.dim() - 2)))

        cs,cb = self.c_proj(cond).chunk(2,dim=1)
        cs = F.interpolate(cs, size=h.shape[2:], mode="nearest")
        cb = F.interpolate(cb, size=h.shape[2:],mode="nearest")

        h = h * (1 + ts + cs) + tb + cb
        h = F.silu(self.norm2(h))
        return x + self.conv2(h)


class DiffusionDecoder(nn.Module):

    def __init__(self, ndim=1, in_channels=1, channels=64, cond_dim=128, depth=6, tdim=128):
        super().__init__()
        Conv = conv_cls(ndim)
        k = 3 if ndim > 1 else 7
        self.ndim = ndim
        self.time = TimeEmbedding(tdim)
        self.inp = Conv(in_channels,channels, k, padding=k // 2)
        self.blocks = nn.ModuleList(
            ConditionedBlock(ndim, channels, cond_dim, tdim) for _ in range(depth)
        )
        self.out = nn.Sequential(
            nn.GroupNorm(8, channels),nn.SiLU(),
            Conv(channels, in_channels,k, padding=k // 2),
        )

    def forward(self,x_t, t, cond):
        temb = self.time(t)
        h = self.inp(x_t)
        for blk in self.blocks:
            h = blk(h,temb,cond)
        return self.out(h)

    def loss(self, x0, cond):
        b = x0.shape[0]
        t = torch.rand(b,device=x0.device)
        shape = (b,) + (1,) * (x0.dim() - 1)
        noise = torch.randn_like(x0)
        x_t = (1 - t.view(shape)) * noise + t.view(shape) * x0
        target = x0 - noise
        return F.mse_loss(self(x_t,t,cond), target)

    @torch.no_grad()
    def sample(self,cond, shape, steps=8, device=None):
        device = device or cond.device
        # rectified flow is near-straight, so a handful of steps is enough
        x = torch.randn(shape, device=device)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((shape[0],), i * dt, device=device)
            x = x + self(x,t, cond) * dt
        return x

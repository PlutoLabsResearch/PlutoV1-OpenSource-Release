from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class S4DKernel(nn.Module):
    def __init__(self, channels, state=64):
        super().__init__()
        self.channels, self.state = channels, state
        # log-spaced decay rates: some states remember briefly, others a long time
        self.log_A = nn.Parameter(torch.log(torch.linspace(0.001, 0.3, state).unsqueeze(0).repeat(channels, 1)))
        self.B = nn.Parameter(torch.randn(channels, state) * 0.5)
        self.C = nn.Parameter(torch.randn(channels, state) * 0.5)
        self.D = nn.Parameter(torch.ones(channels))

    def kernel(self, length, device, chunk=512):
        # built in chunks: the full (channels, state, length) tensor is hundreds
        # of MB per layer and grows with context, which caps usable sequence length
        A = -torch.exp(self.log_A)
        w = (self.C * self.B).unsqueeze(-1)
        out = []
        for i in range(0, length, chunk):
            t = torch.arange(i, min(i + chunk, length), device=device).view(1, 1, -1)
            out.append((w * torch.exp(A.unsqueeze(-1) * t)).sum(1))
        return torch.cat(out, dim=-1)

    def forward(self, x):
        L = x.shape[-1]
        n = 2 * L
        with torch.autocast("cuda", enabled=False):   # no complex bf16
            k = self.kernel(L, x.device)
            y = torch.fft.irfft(torch.fft.rfft(x.float(), n=n) * torch.fft.rfft(k.float(), n=n), n=n)[..., :L]
        return y.to(x.dtype) + x * self.D.view(1, -1, 1)


class SSMBlock(nn.Module):
    def __init__(self, dim, expand=2, state=64):
        super().__init__()
        inner = dim * expand
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, inner * 2)
        self.conv = nn.Conv1d(inner, inner, 4, padding=3, groups=inner)
        self.ssm = S4DKernel(inner, state)
        self.out_proj = nn.Linear(inner, dim)

    def forward(self, x):
        residual = x
        x, gate = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        h = self.conv(x.transpose(1, 2))[..., :x.shape[1]]
        h = self.ssm(F.silu(h)).transpose(1, 2)
        return residual + self.out_proj(h * F.silu(gate))


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.0):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} must be divisible by heads {heads}")
        self.heads, self.head_dim = heads, dim // heads
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=-1)
        q, k, v = (t.view(B, T, self.heads, self.head_dim).transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        return x + self.proj(out.transpose(1, 2).contiguous().view(B, T, C))


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.up = nn.Linear(dim, dim * mult * 2)
        self.down = nn.Linear(dim * mult, dim)

    def forward(self, x):
        a, b = self.up(self.norm(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(a) * b)


class HybridBackbone(nn.Module):
    def __init__(self, dim=512, depth=12, attn_every=6, heads=8, state=64, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.kinds = []
        for i in range(depth):
            # attention only every nth layer -- recall where it's needed, linear cost elsewhere
            if attn_every and (i + 1) % attn_every == 0:
                self.layers += [CausalSelfAttention(dim, heads, dropout), FeedForward(dim)]
                self.kinds += ["attn", "ff"]
            else:
                self.layers.append(SSMBlock(dim, state=state))
                self.kinds.append("ssm")

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def composition(self):
        a, s = self.kinds.count("attn"), self.kinds.count("ssm")
        return f"{s} SSM + {a} attention ({a / max(a + s, 1):.0%} quadratic layers)"

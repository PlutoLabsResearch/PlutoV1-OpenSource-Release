from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DEAD = 1e-2


class VectorQuantizer(nn.Module):
    def __init__(self, dim, size, decay=0.99):
        super().__init__()
        self.dim, self.size, self.decay = dim, size, decay
        self.register_buffer("codebook", torch.randn(size, dim) * 0.1)
        self.register_buffer("cluster_size", torch.zeros(size))
        self.register_buffer("ema_sum", torch.randn(size, dim) * 0.1)

    def forward(self, x):
        flat = x.reshape(-1, self.dim)
        d = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ self.codebook.t() + self.codebook.pow(2).sum(1)
        idx = d.argmin(1)
        quant = self.codebook[idx].view_as(x)

        if self.training:
            self._update(flat, idx)

        # straight-through: encoder sees quantization as identity
        commit = F.mse_loss(x, quant.detach())
        quant = x + (quant - x).detach()
        return quant, idx.view(x.shape[0],x.shape[1]), commit

    @torch.no_grad()
    def _update(self, flat, idx):
        onehot = F.one_hot(idx, self.size).type(flat.dtype)
        self.cluster_size.mul_(self.decay).add_(onehot.sum(0), alpha=1 - self.decay)
        self.ema_sum.mul_(self.decay).add_(onehot.t() @ flat, alpha=1 - self.decay)
        n = self.cluster_size.sum()
        w = ((self.cluster_size + 1e-5) / (n + self.size * 1e-5) * n).unsqueeze(1)
        self.codebook.copy_(self.ema_sum / w)

        # revive unused entries or the codebook collapses
        dead = self.cluster_size < DEAD
        if dead.any() and flat.shape[0] > 0:
            pick = torch.randint(0, flat.shape[0], (int(dead.sum()),), device=flat.device)
            self.codebook[dead] = flat[pick]
            self.cluster_size[dead] = 1.0

    def usage(self):
        return float((self.cluster_size > DEAD).float().mean())


class ResidualVQ(nn.Module):
    def __init__(self, dim, size, levels):
        super().__init__()
        self.levels, self.size = levels, size
        self.layers = nn.ModuleList(VectorQuantizer(dim, size) for _ in range(levels))

    def forward(self, x):
        residual, total = x, torch.zeros_like(x)
        all_idx, commit = [], 0.0
        for layer in self.layers:
            q, idx, c = layer(residual)
            residual = residual - q.detach()
            total = total + q
            all_idx.append(idx)
            commit += c
        return total, torch.stack(all_idx, dim=-1), commit / self.levels

    def decode_indices(self, idx):
        return sum(l.codebook[idx[..., i]] for i, l in enumerate(self.layers))

    def usage(self):
        return [l.usage() for l in self.layers]

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleCritic(nn.Module):

    def __init__(self, channels=16):
        super().__init__()
        c = channels
        self.layers = nn.ModuleList([
            nn.Conv1d(1, c, 15, padding=7),
            nn.Conv1d(c,c * 4, 41,stride=4, groups=4,padding=20),
            nn.Conv1d(c * 4,c * 16, 41, stride=4,groups=16, padding=20),
            nn.Conv1d(c * 16, c * 32, 41,stride=4, groups=16, padding=20),
            nn.Conv1d(c * 32, c * 32, 5, padding=2),
        ])
        self.out = nn.Conv1d(c * 32,1, 3,padding=1)

    def forward(self, x):
        feats = []
        for layer in self.layers:
            x = F.leaky_relu(layer(x),0.1)
            feats.append(x)
        return self.out(x), feats


class Critic(nn.Module):

    def __init__(self, scales=3, channels=16, lr=2e-4, adv_weight=1.0,fm_weight=2.0,warmup=2000):
        super().__init__()
        self.nets = nn.ModuleList(ScaleCritic(channels) for _ in range(scales))
        self.pool = nn.AvgPool1d(4, stride=2, padding=2)
        self.adv_weight = adv_weight
        self.fm_weight = fm_weight
        self.warmup = warmup
        self.steps = 0
        self._opt = None
        self.lr = lr

    def forward(self,x):
        verdicts,features = [],[]
        for i,net in enumerate(self.nets):
            if i:
                x = self.pool(x)
            v, f = net(x)
            verdicts.append(v)
            features.append(f)
        return verdicts, features

    def _optimizer(self):
        if self._opt is None:
            self._opt = torch.optim.AdamW(self.parameters(), lr=self.lr,betas=(0.8,0.95))
        return self._opt

    def active(self):
        return self.steps >= self.warmup

    def step_discriminator(self, real,fake):
        # useless against a codec that can't reconstruct yet
        self.steps += 1
        if not self.active():
            return 0.0
        rv,_ = self(real)
        fv, _ = self(fake.detach())
        loss = sum(F.relu(1 - r).mean() + F.relu(1 + f).mean() for r, f in zip(rv, fv)) / max(len(rv), 1)
        opt = self._optimizer()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        opt.step()
        return float(loss.item())

    def generator_terms(self, real,fake):
        if not self.active():
            zero = torch.zeros((), device=real.device)
            return zero, zero
        rv, rf = self(real)
        fv, ff = self(fake)
        adv = sum((-f).mean() for f in fv) / max(len(fv), 1)
        n = 0
        fm = 0.0
        for a, b in zip(rf, ff):
            for r, f in zip(a, b):
                fm = fm + F.l1_loss(f, r.detach())
                n += 1
        return self.adv_weight * adv,self.fm_weight * (fm / max(n, 1))

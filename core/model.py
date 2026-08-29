from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.backbone import HybridBackbone
from core.vocab import EOS, SEP, vocab_size


class Model(nn.Module):
    def __init__(self,codebook=512, levels=4, dim=512, depth=12, attn_every=4, heads=8, state=64, dropout=0.1,max_len=8192):
        super().__init__()
        self.codebook,self.levels = codebook,levels
        self.vocab = vocab_size(codebook, levels)
        self.max_len = max_len
        self.embed = nn.Embedding(self.vocab, dim)
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        self.drop = nn.Dropout(dropout)
        self.backbone = HybridBackbone(dim, depth,attn_every,heads,state,dropout)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, self.vocab,bias=False)
        self.head.weight = self.embed.weight
        self.apply(self._init)
        nn.init.normal_(self.pos, std=0.01)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids):
        h = self.drop(self.embed(ids) + self.pos[:, :ids.shape[1]])
        return self.head(self.norm(self.backbone(h)))

    def loss(self, ids, mask_prefix=True):
        logits = self(ids[:, :-1])
        target = ids[:, 1:].clone()
        if mask_prefix:
            # graded on the answer, not on repeating the prompt
            for b in range(ids.shape[0]):
                hit = (ids[b] == SEP).nonzero()
                if hit.numel():
                    target[b, :int(hit[0])] = -100
        return F.cross_entropy(logits.reshape(-1, self.vocab), target.reshape(-1), ignore_index=-100)

    @torch.no_grad()
    def generate(self, prompt, max_new=512, temperature=0.9,top_p=0.95, rep_penalty=1.2, rep_window=96, guidance=1.0,uncond=None, stop_at_eos=True):
        dev = next(self.parameters()).device
        out = (prompt if prompt.dim() == 2 else prompt.unsqueeze(0)).to(dev)
        if uncond is not None:
            uncond = uncond.to(dev)
        for _ in range(max_new):
            v = self(out[:, -self.max_len:])[:, -1].clone()

            if guidance != 1.0 and uncond is not None:
                u = uncond if uncond.dim() == 2 else uncond.unsqueeze(0)
                tail = out[:, prompt.shape[-1]:]
                ucat = torch.cat([u, tail],dim=1) if tail.numel() else u
                vu = self(ucat[:, -self.max_len:])[:, -1]
                v = vu + guidance * (v - vu)

            recent = out[0, -rep_window:].unique()
            v[0, recent] = torch.where(v[0, recent] > 0,v[0, recent] / rep_penalty, v[0,recent] * rep_penalty)
            p = F.softmax(v / max(temperature, 1e-5), dim=-1)
            sp,si = torch.sort(p, descending=True, dim=-1)
            sp = sp * ((sp.cumsum(-1) - sp) < top_p)
            sp = sp / sp.sum(-1, keepdim=True).clamp_min(1e-9)
            nxt = si.gather(-1, torch.multinomial(sp, 1))
            out = torch.cat([out, nxt], dim=1)
            if stop_at_eos and int(nxt.item()) == EOS:
                break
        return out

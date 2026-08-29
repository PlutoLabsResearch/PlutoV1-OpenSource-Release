from __future__ import annotations

import random
from dataclasses import dataclass

import torch


@dataclass
class Sample:
    prefix: torch.Tensor
    target: torch.Tensor
    synthetic: bool = False
    score: float = 1.0


class SelfImproving:
    def __init__(self, verifier=None, max_synthetic=0.3):
        if not 0.0 <= max_synthetic < 1.0:
            raise ValueError("max_synthetic must be in [0, 1)")
        self.verifier = verifier
        self.max_synthetic = max_synthetic
        self.real, self.synthetic = [], []
        self.rejected = 0

    def add(self, target, prefix=None):
        self.real.append(Sample(prefix if prefix is not None else torch.zeros(0, dtype=torch.long), target))

    def absorb(self, generated, prefix=None):
        score = self.verifier.score(generated) if self.verifier else 1.0
        bar = getattr(self.verifier, "threshold", None) if self.verifier else None
        total = len(self.real) + len(self.synthetic) + 1
        # real data must stay in the majority or the distribution drifts and collapses
        if (bar is not None and score < bar) or (len(self.synthetic) + 1) / total > self.max_synthetic:
            self.rejected += 1
            return False
        g = generated.squeeze(0) if generated.dim() == 3 else generated
        self.synthetic.append(Sample(prefix if prefix is not None else torch.zeros(0, dtype=torch.long), g, True, score))
        return True

    def batch(self, n, rng=None):
        rng = rng or random
        if not self.real and not self.synthetic:
            return []
        n_syn = min(len(self.synthetic), int(n * self.max_synthetic))
        out = [self.real[rng.randrange(len(self.real))] for _ in range(n - n_syn)] if self.real else []
        out += [self.synthetic[rng.randrange(len(self.synthetic))] for _ in range(n_syn)] if self.synthetic else []
        rng.shuffle(out)
        return out

    def stats(self):
        total = len(self.real) + len(self.synthetic)
        return {"real": len(self.real), "synthetic": len(self.synthetic),
                "share": round(len(self.synthetic) / total, 3) if total else 0.0,
                "cap": self.max_synthetic, "rejected": self.rejected}

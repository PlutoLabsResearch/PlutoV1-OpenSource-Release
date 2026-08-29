from __future__ import annotations

import re

import torch


class CodecVerifier:
    def __init__(self,codec):
        self.codec = codec
        self.real_mean = None
        self.threshold = None

    @torch.no_grad()
    def score(self, tokens):
        self.codec.eval()
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
        payload = self.codec.decode(tokens, (tokens.shape[1],))
        again,_ = self.codec.encode(payload)
        n = min(tokens.shape[1], again.shape[1])
        if n == 0:
            return 0.0
        return float((tokens[:,:n] == again[:, :n]).float().mean())

    @torch.no_grad()
    def calibrate(self,real_tokens, margin=0.8):
        # round-trip isn't idempotent, so calibrate against real data
        scores = [self.score(t) for t in real_tokens]
        if not scores:
            return 0.0
        self.real_mean = sum(scores) / len(scores)
        self.threshold = self.real_mean * margin
        return self.threshold

    def accepts(self, tokens):
        if self.threshold is None:
            raise RuntimeError("calibrate() on real data first")
        return self.score(tokens) >= self.threshold


class AnswerVerifier:
    def __init__(self,check=None):
        self.check = check or numeric_match

    def collect(self, model, problems,samples=8, max_new=320):
        from core.vocab import SEP,decode_bytes,encode_bytes
        dev = next(model.parameters()).device
        kept, attempts = [], 0
        for problem in problems:
            ids = torch.cat([encode_bytes(problem), torch.tensor([SEP])]).unsqueeze(0).to(dev)
            for _ in range(samples):
                gen = model.generate(ids,max_new=max_new, temperature=1.0)
                answer = decode_bytes(gen[0,ids.shape[1]:])
                attempts += 1
                if self.check(problem,answer):
                    kept.append(gen[0].cpu())
        return kept, attempts


def numeric_match(problem,answer):
    want = re.search(r"####\s*(-?[\d.]+)", problem)
    got = re.findall(r"-?\d+\.?\d*", answer)
    if not want or not got:
        return False
    try:
        return abs(float(got[-1]) - float(want.group(1))) < 1e-6
    except ValueError:
        return False

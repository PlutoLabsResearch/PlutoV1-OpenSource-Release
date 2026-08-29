from __future__ import annotations

import torch

BYTE_BASE = 0
BYTE_COUNT = 256

BOS = 256
EOS = 257
SEP = 258
PAD = 259
AUDIO_MARK = 260
IMAGE_MARK = 261
VIDEO_MARK = 262
TEXT_MARK = 263

CONTROL_END = 264


def vocab_size(codebook=512, levels=4):
    return CONTROL_END + codebook * levels


def encode_bytes(s):
    b = s.encode("utf-8") if isinstance(s,str) else s
    return torch.tensor(list(b),dtype=torch.long)


def decode_bytes(ids):
    raw = bytes(int(i) for i in ids.flatten().tolist() if i < BYTE_COUNT)
    return raw.decode("utf-8", errors="replace")


def encode_codec(tokens, codebook=512):
    T, levels = tokens.shape
    # each level gets its own block, otherwise residual depth is invisible
    offs = CONTROL_END + torch.arange(levels, device=tokens.device) * codebook
    return (tokens + offs).reshape(-1)


def decode_codec(ids,levels=4, codebook=512):
    ids = ids[ids >= CONTROL_END]
    usable = (ids.shape[0] // levels) * levels
    if usable == 0:
        return torch.zeros(0, levels, dtype=torch.long)
    grid = ids[:usable].reshape(-1, levels)
    offs = CONTROL_END + torch.arange(levels, device=ids.device) * codebook
    return (grid - offs).clamp(0, codebook - 1)


def sequence(*parts, sep_after=None, end=False):
    out = []
    for i, p in enumerate(parts):
        out.append(p if isinstance(p, torch.Tensor) else encode_bytes(p))
        if sep_after is not None and i == sep_after:
            out.append(torch.tensor([SEP], dtype=torch.long))
    if end:
        # without EOS the model never learns where a sample stops
        out.append(torch.tensor([EOS], dtype=torch.long))
    return torch.cat(out)

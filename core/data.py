from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def read_bytes(path):
    return np.frombuffer(Path(path).read_bytes(),dtype=np.uint8)


def bytes_to_floats(raw):
    return (raw.astype(np.float32) - 127.5) / 127.5


def floats_to_bytes(x):
    return np.clip(np.round(x * 127.5 + 127.5),0, 255).astype(np.uint8)


def load_file(path):
    return bytes_to_floats(read_bytes(path))


def activity(x,window):
    n = len(x) // window
    if n < 3:
        return np.zeros(0)
    blocks = x[:n * window].reshape(n, window).astype(np.float64)
    return np.sqrt((blocks**2).mean(axis=1))


def trim_quiet(x,thresh=0.03, window_ratio=1 / 64):
    x = np.asarray(x, dtype=np.float64)
    if x.size < 32:
        return x.astype(np.float32)

    window = max(1, int(x.size * window_ratio))
    a = activity(x, window)
    if a.size < 3:
        return x.astype(np.float32)

    # silence is training signal, and it teaches the wrong thing
    loud = a >= thresh * float(np.max(np.abs(x)))
    if not loud.any():
        return x.astype(np.float32)

    keep = [x[i * window : (i + 1) * window] for i, on in enumerate(loud) if on]
    return (np.concatenate(keep) if keep else x).astype(np.float32)


def chunk_1d(streams,length,hop=None):
    hop = hop or length
    out = []
    for s in streams:
        s = np.asarray(s,dtype=np.float32).ravel()
        for i in range(0,max(len(s) - length + 1,0), hop):
            out.append(s[i : i + length])
    if not out:
        return torch.empty(0, 1, length)
    return torch.from_numpy(np.stack(out)).unsqueeze(1)


def chunk_nd(items,has_channels=True):
    if not items:
        return torch.empty(0)
    arr = np.stack([np.asarray(i, dtype=np.float32) for i in items])
    # (H, W) is rank-2 with one channel; (C, H, W) already carries its own
    if not has_channels:
        arr = arr[:, None]
    return torch.from_numpy(arr)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

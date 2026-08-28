from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from core.data import trim_quiet

SR = 24_000
EXT = {".wav", ".mp3", ".flac", ".ogg"}


def load(path, trim=True):
    a, _ = sf.read(str(path), dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    return trim_quiet(a, thresh=0.03) if trim else a.astype(np.float32)


def load_pairs(folder, limit=None):
    out = []
    for f in sorted(Path(folder).iterdir()):
        if f.suffix.lower() not in EXT or f.stat().st_size == 0:
            continue
        txt = f.with_suffix(".txt")
        if not txt.exists():
            continue
        try:
            out.append((load(f), txt.read_text(encoding="utf-8").strip()))
        except Exception:
            continue
        if limit and len(out) >= limit:
            break
    return out


def resample(a, src, dst=SR):
    if src == dst:
        return a.astype(np.float32)
    n = int(round(len(a) * dst / src))
    return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)


def from_mic(seconds=4.0, sr=SR, device=None):
    import sounddevice as sd
    a = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32", device=device)
    sd.wait()
    return trim_quiet(a.ravel(), thresh=0.03)


def augment(a, rng, noise=0.005, speed=0.08, gain=0.15):
    # the corpus is synthetic and clean; real microphones are neither
    if speed:
        r = 1.0 + rng.uniform(-speed, speed)
        n = int(len(a) / r)
        a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)
    if gain:
        a = a * (1.0 + rng.uniform(-gain, gain))
    if noise:
        a = a + rng.normal(0, noise * float(np.abs(a).max() + 1e-6), len(a)).astype(np.float32)
    return np.clip(a, -1.0, 1.0).astype(np.float32)

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.quantize import ResidualVQ

CONV = {1: nn.Conv1d, 2: nn.Conv2d,3: nn.Conv3d}
DECONV = {1: nn.ConvTranspose1d, 2: nn.ConvTranspose2d,3: nn.ConvTranspose3d}


def conv_cls(ndim, transpose=False):
    t = DECONV if transpose else CONV
    if ndim not in t:
        raise ValueError(f"ndim must be 1, 2 or 3 (got {ndim})")
    return t[ndim]


def as_tuple(v, ndim):
    return tuple(v for _ in range(ndim)) if isinstance(v, int) else tuple(v)


class ResidualUnit(nn.Module):
    def __init__(self,ndim,channels,dilation):
        super().__init__()
        C = conv_cls(ndim)
        k = 3 if ndim > 1 else 7
        pad = (k - 1) * dilation // 2
        self.block = nn.Sequential(nn.ELU(), C(channels,channels, k, dilation=dilation, padding=pad), nn.ELU(), C(channels,channels, 1))

    def forward(self, x):
        return x + self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, ndim, in_ch, out_ch, stride):
        super().__init__()
        s = as_tuple(stride, ndim)
        C = conv_cls(ndim)
        down = C(in_ch, out_ch, tuple(2 * v for v in s), stride=s, padding=tuple(v // 2 for v in s))
        self.body = nn.Sequential(ResidualUnit(ndim,in_ch, 1),ResidualUnit(ndim,in_ch,3), nn.ELU(),down)

    def forward(self, x):
        return self.body(x)


class DecoderBlock(nn.Module):
    def __init__(self, ndim,in_ch, out_ch, stride):
        super().__init__()
        s = as_tuple(stride, ndim)
        up = conv_cls(ndim, True)(in_ch, out_ch, tuple(2 * v for v in s), stride=s,padding=tuple(v // 2 for v in s))
        self.body = nn.Sequential(nn.ELU(), up,ResidualUnit(ndim, out_ch, 1), ResidualUnit(ndim,out_ch, 3))

    def forward(self,x):
        return self.body(x)


class Codec(nn.Module):
    def __init__(self,ndim=1, in_channels=1, channels=32, latent=128, strides=(2,4,4, 5), codebook_size=512, levels=4):
        super().__init__()
        self.ndim, self.in_channels = ndim,in_channels
        C = conv_cls(ndim)
        k = 3 if ndim > 1 else 7

        hop = [1] * ndim
        for s in strides:
            for a, v in enumerate(as_tuple(s,ndim)):
                hop[a] *= v
        self.hop = tuple(hop)

        enc,ch = [C(in_channels, channels,k, padding=k // 2)], channels
        for s in strides:
            enc.append(EncoderBlock(ndim,ch, ch * 2,s))
            ch *= 2
        enc += [nn.ELU(), C(ch, latent,3, padding=1)]
        self.encoder = nn.Sequential(*enc)

        dec,ch2 = [C(latent, ch, 3,padding=1)], ch
        for s in reversed(strides):
            dec.append(DecoderBlock(ndim, ch2, ch2 // 2, s))
            ch2 //= 2
        dec += [nn.ELU(), C(ch2,in_channels, k, padding=k // 2)]
        self.decoder = nn.Sequential(*dec)
        self.quantizer = ResidualVQ(latent, codebook_size,levels)

    def pad_input(self, x):
        pads = []
        for a in reversed(range(self.ndim)):
            pads += [0,(-x.shape[2 + a]) % self.hop[a]]
        return F.pad(x, pads) if any(pads) else x

    @staticmethod
    def _fit(y,shape):
        if y.shape[2:] == shape[2:]:
            return y
        y = y[tuple([slice(None), slice(None)] + [slice(0, s) for s in shape[2:]])]
        pads = []
        for a in reversed(range(len(shape) - 2)):
            pads += [0,max(0, shape[2 + a] - y.shape[2 + a])]
        return F.pad(y,pads) if any(pads) else y

    # flatten last, so the convolutions still see spatial structure
    def _flatten(self, z):
        return z.flatten(2).transpose(1,2), tuple(z.shape[2:])

    def _unflatten(self, seq, grid):
        return seq.transpose(1, 2).reshape(seq.shape[0], -1, *grid)

    def encode(self,x):
        z,grid = self._flatten(self.encoder(self.pad_input(x)))
        _,idx, _ = self.quantizer(z)
        return idx, grid

    def decode(self, idx, grid):
        return self.decoder(self._unflatten(self.quantizer.decode_indices(idx), grid))

    def forward(self, x):
        z,grid = self._flatten(self.encoder(self.pad_input(x)))
        q, idx, commit = self.quantizer(z)
        recon = self.decoder(self._unflatten(q,grid))
        return self._fit(recon, x.shape), idx, commit

    def compression(self, shape):
        n_in, frames = 1, 1
        for a, s in enumerate(shape):
            n_in *= s
            frames *= max(1,s // self.hop[a])
        tokens = frames * self.quantizer.levels
        return f"{n_in} values -> {frames} frames x {self.quantizer.levels} tokens ({n_in / max(tokens, 1):.1f}x)"

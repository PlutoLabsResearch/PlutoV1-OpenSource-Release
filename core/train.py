from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from core.data import device
from core.vocab import PAD


def spectral_loss(a, b,sizes=(256, 512, 1024)):
    # phase-blind: never let this outweigh the waveform term
    total, used = 0.0, 0
    for n in sizes:
        if a.shape[-1] < n:
            continue
        win = torch.hann_window(n, device=a.device)
        sa = torch.stft(a.squeeze(1), n, n // 4, window=win, return_complex=True).abs()
        sb = torch.stft(b.squeeze(1), n, n // 4, window=win,return_complex=True).abs()
        total += F.l1_loss(sa, sb) / sa.abs().mean().clamp_min(1e-5)
        total += 0.5 * F.l1_loss(torch.log(sa + 1e-3), torch.log(sb + 1e-3))
        used += 1
    return total / max(used, 1)


def pad_batch(rows, dev):
    w = max(r.shape[0] for r in rows)
    out = torch.full((len(rows), w), PAD, dtype=torch.long)
    for i, r in enumerate(rows):
        out[i, :r.shape[0]] = r
    return out.to(dev)


class Trainer:
    def __init__(self, critic=None, buffer=None, lr=3e-4,weight_decay=0.05, wave_weight=50.0, log_every=500, amp=True):
        self.critic, self.buffer = critic, buffer
        self.lr, self.weight_decay = lr, weight_decay
        self.wave_weight = wave_weight
        self.log_every = log_every
        self.device = device()
        # bf16 needs no loss scaling, unlike fp16
        self.amp = amp and self.device.type == "cuda" and torch.cuda.is_bf16_supported()

    def fit_codec(self, codec, data, steps=6000, batch=16, use_spectral=None, save_best=None, eval_every=None):
        codec = codec.to(self.device)
        ndim = data.dim() - 2
        if use_spectral is None:
            use_spectral = ndim == 1
        opt = torch.optim.AdamW(codec.parameters(), lr=self.lr, betas=(0.8, 0.95))

        params = sum(p.numel() for p in codec.parameters()) / 1e6
        print(f"codec ndim={ndim} hop={codec.hop} {params:.2f}M n={data.shape[0]}{' adv' if self.critic else ''}")
        eval_every = eval_every or self.log_every
        best = float("inf")
        codec.train()

        for step in range(1, steps + 1):
            x = data[torch.randint(0, data.shape[0], (min(batch, data.shape[0]),))].to(self.device)
            recon, _, commit = codec(x)
            loss = self.wave_weight * (F.l1_loss(recon, x) + F.mse_loss(recon, x)) + 0.25 * commit
            if use_spectral:
                loss = loss + spectral_loss(recon, x)
            if self.critic is not None:
                self.critic.step_discriminator(x, recon)
                adv, fm = self.critic.generator_terms(x, recon)
                loss = loss + adv + fm

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
            opt.step()

            if step % eval_every and step != 1:
                continue
            with torch.no_grad():
                err = (F.mse_loss(recon, x).sqrt() / x.pow(2).mean().sqrt().clamp_min(1e-9)).item()
                corr = torch.corrcoef(torch.stack([x.flatten(), recon.flatten()]))[0, 1].item()
            mark = ""
            # adversarial training oscillates, so the last step is rarely the best one
            if err < best:
                best = err
                if save_best:
                    torch.save(codec.state_dict(), save_best)
                mark = "  *"
            print(f"{step:6d} loss {loss.item():7.3f}  err {err:.4f}  corr {corr:+.3f}  cb {np.mean(codec.quantizer.usage()):.0%}{mark}")

        codec.eval()
        return codec

    def fit_model(self, model, sequences, steps=6000, batch=12, ctx=512, held=None, eval_every=500, patience=8, save_best=None):
        model = model.to(self.device)
        usable = [s for s in sequences if s.shape[0] > 8]
        if not usable:
            raise ValueError("no usable sequences")

        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, betas=(0.9, 0.95), weight_decay=self.weight_decay)
        warm = min(max(0.05, 2.0 / max(steps, 1)), 0.5)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps, pct_start=warm)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"model {params:.2f}M {model.backbone.composition()} n={len(usable)}")

        # held-out bottoms out long before training ends
        best, best_step, since = float("inf"), 0, 0
        model.train()
        for step in range(1, steps + 1):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
                loss = model.loss(self._batch(usable, batch, ctx))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            if step % eval_every and step != 1:
                continue
            msg = f"{step:6d} train {np.exp(loss.item()):8.1f}"
            if held:
                hp = self.evaluate(model, held, ctx, batch)
                msg += f"  held {hp:8.1f}"
                if hp < best:
                    best, best_step, since = hp, step, 0
                    if save_best:
                        torch.save(model.state_dict(), save_best)
                    msg += "  *"
                else:
                    since += 1
            print(msg, flush=True)
            if held and since >= patience:
                print(f"stop {best:.1f} @ {best_step}")
                break

        model.eval()
        return model

    def _batch(self, seqs, n, ctx):
        pool = seqs
        if self.buffer is not None:
            drawn = [s.target for s in self.buffer.batch(n)]
            if drawn:
                pool = drawn
        rows = []
        for _ in range(n):
            s = pool[np.random.randint(len(pool))]
            if s.shape[0] > ctx:
                i = np.random.randint(0, s.shape[0] - ctx)
                s = s[i:i + ctx]
            rows.append(s)
        return pad_batch(rows, self.device)

    @torch.no_grad()
    def evaluate(self, model, held, ctx=512, batch=8, rounds=8):
        model.eval()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            losses = [model.loss(self._batch(held, batch, ctx)).item() for _ in range(rounds)]
        model.train()
        return float(np.exp(np.mean(losses)))

    def reinforce(self, model, problems, verifier, samples=8, rounds=1, max_new=320):
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr * 0.1, weight_decay=self.weight_decay)
        history = []
        for r in range(rounds):
            kept, attempts = verifier.collect(model, problems, samples, max_new)
            rate = len(kept) / max(attempts, 1)
            if rate < 0.02:
                print(f"{r} pass {rate:.1%} skip")
                history.append({"round": r, "pass_rate": rate, "loss": None})
                continue
            loss = self._train_on(model, kept, opt)
            print(f"{r} pass {rate:.1%} n={len(kept)} loss {loss:.4f}")
            history.append({"round": r, "pass_rate": rate, "loss": loss})
        return {"rounds": history}

    def _train_on(self, model, samples, opt):
        model.train()
        losses = []
        for i in range(0, len(samples), 4):
            loss = model.loss(pad_batch(samples[i:i + 4], self.device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        model.eval()
        return float(np.mean(losses)) if losses else float("nan")

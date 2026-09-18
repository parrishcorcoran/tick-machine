#!/usr/bin/env python3
"""EXACT TRANSFER OVER THE AUDIO WIRE. Gap coding on a slot-exact clock.

The tick is one audio sample (48 kHz): the receiver sees a slot or it does
not, nothing in between. A weight is the gap since the previous pulse (its
|angle| on a 4096 dial, plus a guard so pulses never overlap); its sign is the
pulse's polarity. No value is ever sent. Every recovered weight is checked
tick for tick against the angle that was sent, and the layer's output against
the matmul.

Mac headphone out -> cable -> USB dongle mic in (one channel).
    python3 tick_machine/audio_gap.py --neurons 4
"""
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, "/Users/abundancemachine/AnalogLLM")
from loopback.characterize import capture, FS      # noqa: E402  the same link as yesterday

GUARD = 8            # samples between pulses even for a zero weight
HALF = 2047
AMP = 0.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neurons", type=int, default=4)
    ap.add_argument("--layer", type=int, default=5)
    a = ap.parse_args()

    from transformers import GPT2LMHeadModel
    m = GPT2LMHeadModel.from_pretrained("gpt2")
    lin = m.transformer.h[a.layer].mlp.c_fc
    W = lin.weight.detach().double().numpy()[:, :a.neurons]          # (768, neurons)
    cyc = np.abs(W).max(0)
    q = np.round(W / cyc * HALF).astype(np.int64)                     # signed angles
    qs = q.T.reshape(-1)                                              # column by column
    n = qs.size

    # --- SENDER: pulses only. gap = |angle| + guard, polarity = sign.
    gaps = np.abs(qs) + GUARD
    pos = np.cumsum(np.concatenate([[FS // 10], gaps]))               # pilot at 0.1 s, then the weights
    sig = np.zeros(pos[-1] + FS // 10)
    sig[pos[0]] = AMP                                                 # pilot: known positive
    sig[pos[1:]] = AMP * np.where(qs >= 0, 1.0, -1.0)
    print(f"{n:,} weights of layer {a.layer} c_fc, {a.neurons} neurons: {n} pulses, "
          f"{sig.size/FS:.1f} s of wire, {n/(sig.size/FS):.0f} weights/s")

    # --- WIRE
    rec = capture(sig, None, None, pad=int(1.5 * FS))

    # --- RECEIVER: find pulses, read gaps, read polarity. Uses only what came back.
    # the pilot's recorded shape is the matched filter (pulse spreads over a few samples)
    e = np.convolve(np.abs(rec), np.ones(64) / 64, "same")
    p0 = int(np.argmax(e[:int(1.0 * FS)] > 0.25 * e.max()))
    p0 = p0 + int(np.argmax(np.abs(rec[p0:p0 + 64])))
    kern = rec[p0 - 6:p0 + 7].copy()
    polarity = np.sign(kern[6])                                       # the wire may invert
    kern *= polarity
    c = np.correlate(rec, kern, "same") / (kern @ kern)
    # peaks: local maxima of |c| above threshold, at least GUARD-2 apart
    thr = 0.4
    cand = np.flatnonzero(np.abs(c) > thr)
    peaks = []
    for i in cand:
        if c[i] == c[max(0, i - 3):i + 4][np.argmax(np.abs(c[max(0, i - 3):i + 4]))] and (not peaks or i - peaks[-1] >= GUARD - 2):
            peaks.append(i)
    peaks = np.array(peaks)
    print(f"  receiver found {peaks.size} pulses (sent {n + 1})")
    got_gaps = np.diff(peaks)
    got_mag = got_gaps - GUARD
    got_sign = np.where(c[peaks[1:]] >= 0, 1, -1)
    got = got_sign * got_mag
    k = min(got.size, n)
    exact = int(np.sum(got[:k] == qs[:k]))
    off = np.abs(got[:k] - qs[:k])
    print(f"  [MEASURED] tick-exact weights: {exact}/{k}   wrong: {k - exact}   worst off by {off.max() if k else 0} ticks")
    # the layer's output for these neurons, from the wire, vs the matmul
    x = np.random.default_rng(1).standard_normal(768)
    if k == n:
        Wg = (got.reshape(a.neurons, 768).T / HALF) * cyc
        y = x @ Wg; ref = x @ W
        print(f"  [MEASURED] layer output vs matmul: rel err {np.linalg.norm(y - ref) / np.linalg.norm(ref):.2e}"
              f"   (dial rounding alone: {np.linalg.norm(x @ ((q / HALF) * cyc) - ref) / np.linalg.norm(ref):.2e})")
    else:
        print("  pulse count mismatch -- decode not aligned; see counts above")


if __name__ == "__main__":
    main()

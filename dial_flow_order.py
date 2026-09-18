#!/usr/bin/env python3
"""THE STREAMING GPT-2 ON THE DIAL. One machine.

The dial turns; one turn is one tick. Every weight is an angle (4096 + lap)
and fires once per turn as one edge. Every neuron output is an angle on its
own dial with a lap counter. All 12 blocks are live on every turn, each
reading what the block before it emitted on the previous turn. The residual
stream is held, never cleared: the accumulators ring. The forks are read on
every turn while the dial is still turning; the token is whichever rings
loudest, and we watch it lock.

    python3 dial_flow.py --T 128
"""
import argparse
import math
import sys

import torch

sys.path.insert(0, "/Users/abundancemachine/tick-machine")
import edge_model as em            # noqa: E402  the dial machine: EdgeLinear, lag attention, dial embeddings

torch.set_num_threads(8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=128, help="ticks per cycle: how long the input takes to arrive")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--dial", type=int, default=65536, help="weight dial (4096 x 16 laps)")
    ap.add_argument("--act", type=int, default=4096, help="neuron-output dial")
    ap.add_argument("--order", default="fixed", help="fixed | shuffle: blocks fire in a random order each turn, in place")
    ap.add_argument("--perm", type=int, default=-1, help="seed: wire the 12 blocks in a random depth order (-1 = as trained)")
    ap.add_argument("--gelu", default="on", help="on | off (identity) | clip (limiter +-3) | relu")
    ap.add_argument("--stream", default="full", help="full | bits (one bit per crossing PER TICK: the edge nudges the receiver's dial up or down; the dial keeps the hidden state) | dial (every value crossing between blocks is one edge on a 4096 dial + laps) | dial1 (4096, one turn only, clipped) | sign (one bit: the edge direction only)")
    ap.add_argument("--input", default="pulses", help="pulses (arrives over T ticks, width-coded, so T is also its resolution) | step (full precision, all at once)")
    ap.add_argument("--init", default="zero", help="zero | noise: accumulators start with random junk on them")
    a = ap.parse_args()
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    ids = tok(a.prompt, return_tensors="pt").input_ids
    S = ids.shape[1]
    ref = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    with torch.no_grad():
        batch_tok = ref(ids).logits[0, -1].argmax().item()

    m, n_weights = em.build(a.dial, a.act, True, 16384, ids)        # weights, embeddings, head, outputs: all on dials
    tr = m.transformer
    blocks = list(tr.h)
    if a.gelu != "on":
        import torch.nn as nn
        f = {"off": nn.Identity(), "clip": nn.Hardtanh(-3, 3), "relu": nn.ReLU()}[a.gelu]
        for blk in blocks: blk.mlp.act = f
    if a.perm >= 0:
        p = torch.randperm(len(blocks), generator=torch.Generator().manual_seed(a.perm)).tolist(); blocks = [blocks[i] for i in p]; print("  depth order:", p)
    L = len(blocks)
    layers = [mod for blk in blocks for mod in (blk.attn.c_attn, blk.attn.c_proj, blk.mlp.c_fc, blk.mlp.c_proj)] + [m.lm_head]

    with torch.no_grad():
        x0 = tr.wte(ids) + tr.wpe(torch.arange(S))[None]             # the chord is struck (dial embeddings)
    # the input arrives as pulses: each line on for its width, integrated so far
    sx = x0.abs().amax(dim=-1, keepdim=True) + 1e-12
    width = torch.round(x0.abs() / sx * a.T)
    sign = torch.sign(x0)
    input_at = (lambda t: sign * sx * torch.clamp(torch.full_like(width, float(t)), max=width) / a.T) if a.input == "pulses" else (lambda t: x0)

    g = torch.Generator().manual_seed(1)
    units = None
    if a.stream != "full":
        with torch.no_grad():
            h = x0.clone(); units = []
            for blk in blocks:
                units.append(h[0].abs().amax(0).clamp(min=1e-6) / 2047); out = blk(h); h = out[0] if isinstance(out, tuple) else out
            units.append(h[0].abs().amax(0).clamp(min=1e-6) / 2047)
    PREV = [None] * (L + 1); EDGES = [0, 0]; EDGELOG = []
    R = [None] * (L + 1); STEP = [None] * (L + 1); LASTBIT = [None] * (L + 1)
    def as_edges(k, v):
        if a.stream == "full": return v
        if a.stream == "bits":                                   # the receiver's dial R holds the state; one bit per tick moves it
            if R[k] is None: R[k] = torch.zeros_like(v); STEP[k] = torch.full_like(v, float(units[k].mean()) * 16); LASTBIT[k] = torch.zeros_like(v)
            if R[k].shape != v.shape:                            # positions arrived: grow the dials
                R[k] = torch.zeros_like(v); STEP[k] = torch.full_like(v, float(units[k].mean()) * 16); LASTBIT[k] = torch.zeros_like(v)
            bit = torch.sign(v - R[k])                           # THE EDGE: up or down, nothing else crosses
            same = (bit == LASTBIT[k]) & (bit != 0)
            STEP[k] = torch.where(same, STEP[k] * 1.5, STEP[k] * 0.5).clamp(units[k] * 0.25, units[k] * 2047)
            R[k] = R[k] + bit * STEP[k]; LASTBIT[k] = bit
            return R[k]
        u = units[k]
        if a.stream == "dial":                                                                # angle + laps; an edge fires ONLY where the angle changed
            q = torch.round(v / u)
            if PREV[k] is None or PREV[k].shape != q.shape: EDGES[0] += q.numel(); EDGES[1] += q.numel()
            else: EDGES[0] += int((q != PREV[k]).sum()); EDGES[1] += q.numel()
            PREV[k] = q
            return q * u
        if a.stream == "dial1": return torch.clamp(torch.round(v / u), -2047, 2047) * u     # one turn, no laps
        if a.stream == "sign":  return torch.sign(v) * u * 2047 / 8                        # one bit: which way the edge went
    emitted = [torch.zeros_like(x0) if a.init == "zero" else torch.randn(x0.shape, generator=g) * x0.abs().mean() * 4 for _ in range(L + 1)]
    history, lock, curve = [], None, []
    edges_per_turn = sum(l.n_weights for l in layers) * S
    print(f"dial machine: {n_weights:,} weights as angles, {edges_per_turn:,} edges per turn ({S} positions), "
          f"{L} blocks live per turn, {a.T}-tick cycle")
    with torch.no_grad():
        for t in range(1, a.T + L + 2):
            if a.order == "fixed":
                new = [None] * (L + 1)
                new[0] = input_at(min(t, a.T))
                for k in range(1, L + 1):                                # every block, this turn, reads the last turn
                    out = blocks[k - 1](as_edges(k - 1, emitted[k - 1]))
                    new[k] = out[0] if isinstance(out, tuple) else out
                emitted = new
            else:                                                        # blocks fire in a random order, IN PLACE: whatever is on the dial below, now
                emitted[0] = input_at(min(t, a.T))
                for k in torch.randperm(L, generator=g).tolist():
                    out = blocks[k](emitted[k])
                    emitted[k + 1] = out[0] if isinstance(out, tuple) else out
            logits = m.lm_head(tr.ln_f(as_edges(L, emitted[L])))[0, -1]
            EDGELOG.append((t, EDGES[0], EDGES[1])); EDGES[0] = EDGES[1] = 0           # the forks, read while the dial turns
            top = logits.argmax().item()
            history.append(top)
            srt = logits.sort(descending=True)
            curve.append((t, logits[batch_tok].item(), srt.values[0].item() if top != batch_tok else srt.values[1].item(), tok.decode([top])))
            if lock is None and len(history) >= 8 and all(h == top for h in history[-8:]):
                lock = t - 7
    if a.stream == "dial":
        print("\n  edges that actually fired per turn (only where a dial's angle changed) / values crossing")
        for t, e, n in EDGELOG[:20] + EDGELOG[20::8]: print(f"   turn {t:3d}: {e:7,d} of {n:7,d}   {e/n:6.1%}   head says {tok.decode([history[t-1]])!r}")
    final = history[-1]
    lock = None                                                     # the lock is when the FINAL answer arrives and never leaves
    for i in range(len(history) - 1, -1, -1):
        if history[i] != final: break
        lock = i + 1
    total_edges = edges_per_turn * len(history)
    print(f"  batch answer:   {tok.decode([batch_tok])!r}")
    print(f"  dial answer:    {tok.decode([final])!r}   {'MATCH' if final == batch_tok else 'DIFFERS'}")
    print(f"  locked at tick {lock} of {a.T}" if lock else "  never locked")
    print(f"  edges: {total_edges:,} over {len(history)} turns; one bit each; no weight ever moved")
    print("\n  resonance: the answer's fork vs the loudest other, per tick")
    print(f"  {'tick':>5} {'answer':>9} {'other':>9} {'margin':>8}  head says")
    for t, ans, oth, w in curve[:24] + curve[24::12]:
        print(f"  {t:5d} {ans:9.3f} {oth:9.3f} {ans-oth:+8.3f}  {w!r}")


if __name__ == "__main__":
    main()

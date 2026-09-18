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
    L = len(blocks)
    layers = [mod for blk in blocks for mod in (blk.attn.c_attn, blk.attn.c_proj, blk.mlp.c_fc, blk.mlp.c_proj)] + [m.lm_head]

    with torch.no_grad():
        x0 = tr.wte(ids) + tr.wpe(torch.arange(S))[None]             # the chord is struck (dial embeddings)
    # the input arrives as pulses: each line on for its width, integrated so far
    sx = x0.abs().amax(dim=-1, keepdim=True) + 1e-12
    width = torch.round(x0.abs() / sx * a.T)
    sign = torch.sign(x0)
    input_at = lambda t: sign * sx * torch.clamp(torch.full_like(width, float(t)), max=width) / a.T

    emitted = [torch.zeros_like(x0) for _ in range(L + 1)]           # held across turns, never cleared
    history, lock, curve = [], None, []
    edges_per_turn = sum(l.n_weights for l in layers) * S
    print(f"dial machine: {n_weights:,} weights as angles, {edges_per_turn:,} edges per turn ({S} positions), "
          f"{L} blocks live per turn, {a.T}-tick cycle")
    with torch.no_grad():
        for t in range(1, a.T + L + 2):
            new = [None] * (L + 1)
            new[0] = input_at(min(t, a.T))
            for k in range(1, L + 1):                                # every block, this turn, reads the last turn
                out = blocks[k - 1](emitted[k - 1])
                new[k] = out[0] if isinstance(out, tuple) else out
            emitted = new
            logits = m.lm_head(tr.ln_f(emitted[L]))[0, -1]           # the forks, read while the dial turns
            top = logits.argmax().item()
            history.append(top)
            srt = logits.sort(descending=True)
            curve.append((t, logits[batch_tok].item(), srt.values[0].item() if top != batch_tok else srt.values[1].item(), tok.decode([top])))
            if lock is None and len(history) >= 8 and all(h == top for h in history[-8:]):
                lock = t - 7
    final = history[-1]
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

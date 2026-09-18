#!/usr/bin/env python3
"""HEDGED PIPELINE: branches that differ in WHEN they strike.

Branch 0 strikes after 4 turns of hold (eager, often wrong, far ahead).
Branch 1 after 8, branch 2 after 16, branch 3 after 32 (careful, slow).
All flow through the same dial machine as batch rows. When a position settles
(T+L turns after it started) every branch that struck the wrong note there is
reset to the settled prefix; the output is the frontier of the furthest branch
that was right. A miss costs the gap between two holds, not a full restart.

    python3 dial_hedge.py --tokens 32 --holds 4 8 16 32
"""
import argparse
import sys

import torch

sys.path.insert(0, "/Users/abundancemachine/tick-machine")
import edge_model as em            # noqa: E402

torch.set_num_threads(8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--holds", type=int, nargs="+", default=[4, 8, 16, 32])
    a = ap.parse_args()
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    ids = tok(a.prompt, return_tensors="pt").input_ids[0].tolist()
    ref = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    with torch.no_grad():
        batch = ref.generate(torch.tensor([ids]), max_new_tokens=a.tokens, do_sample=False, pad_token_id=50256)[0].tolist()
    m, _ = em.build(65536, 4096, True, 16384, torch.tensor([ids]))
    tr = m.transformer; blocks = list(tr.h); L = len(blocks)
    B = len(a.holds)
    P = len(ids) + a.tokens                                   # positions, fixed width; unstruck ones have no input yet
    seqs = [list(ids) + [0] * a.tokens for _ in range(B)]     # tokens per branch (0 = placeholder, no input)
    start = [[0] * len(ids) + [None] * a.tokens for _ in range(B)]
    final = list(ids) + [None] * a.tokens                     # settled notes, shared truth
    hist = [[] for _ in range(B)]
    emitted = None
    t, resets, wins = 0, 0, [0] * B
    with torch.no_grad():
        while final[-1] is None:
            t += 1
            x0 = tr.wte(torch.tensor(seqs)) + tr.wpe(torch.arange(P))[None]
            sx = x0.abs().amax(-1, keepdim=True) + 1e-12
            width = torch.round(x0.abs() / sx * a.T)
            prog = torch.tensor([[0 if s is None else min(max(t - s, 0), a.T) for s in st] for st in start],
                                dtype=torch.float32)[:, :, None]
            inp = torch.sign(x0) * sx * torch.minimum(prog.expand_as(width), width) / a.T
            if emitted is None:
                emitted = [torch.zeros_like(x0) for _ in range(L + 1)]
            new = [inp]
            for kk in range(1, L + 1):
                out = blocks[kk - 1](emitted[kk - 1])
                new.append(out[0] if isinstance(out, tuple) else out)
            emitted = new
            logits = m.lm_head(tr.ln_f(emitted[L]))            # (B, P, V)

            # settle: the oldest unsettled position, on the branch furthest along that agrees with the truth so far
            p = final.index(None)
            for b in sorted(range(B), key=lambda b: -sum(s is not None for s in start[b])):
                if start[b][p] is not None and t >= start[b][p] + a.T + L and seqs[b][:p] == final[:p]:
                    final[p] = logits[b, p - 1].argmax().item(); wins[b] += 1
                    break
            if final[p] is not None:
                for b in range(B):
                    if start[b][p] is not None and seqs[b][p] != final[p]:      # struck the wrong note: reset to truth
                        resets += 1
                        seqs[b] = final[:p + 1] + [0] * (P - p - 1)
                        start[b] = start[b][:p] + [t + 1] + [None] * (P - p - 1)
                        for kk in range(L + 1):
                            emitted[kk][b, p:] = 0
                        hist[b] = []
                    elif start[b][p] is None:                                    # slower branch never struck it: give it the truth
                        seqs[b][p] = final[p]; start[b][p] = t + 1

            # strike: each branch on its own hold
            for b in range(B):
                q = next((i for i, s in enumerate(start[b]) if s is None), None)
                if q is None:
                    continue
                top = logits[b, q - 1].argmax().item()
                hist[b].append(top)
                if t - start[b][q - 1] >= a.holds[b] and len(hist[b]) >= a.holds[b] and all(h == top for h in hist[b][-a.holds[b]:]):
                    seqs[b][q] = top; start[b][q] = t + 1; hist[b] = []
    gen = final[len(ids):]
    print(f"  batch:  {tok.decode(batch[len(ids):])!r}")
    print(f"  hedged: {tok.decode(gen)!r}   {'MATCH' if gen == batch[len(ids):] else 'DIFFERS'}")
    print(f"  holds {a.holds}: settled by branch {wins} (which branch was furthest and right), wrong strikes reset {resets}")
    print(f"  ticks/token: {t/a.tokens:.1f} all in (vs 141 one at a time)")


if __name__ == "__main__":
    main()

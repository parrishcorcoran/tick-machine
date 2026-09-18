#!/usr/bin/env python3
"""BRANCHED PIPELINE: when two forks are close, strike both and let both flow.

Each branch is a batch row through the same dial machine (weights read once
for all rows). When a position finalizes (T+L turns after it started), the
branch whose struck note matches the settled note survives and the others are
dropped. Only if no branch had it do we un-strike and restart. Exact by
construction; the branches are a head start on both sides of a near-tie.

    python3 dial_branch.py --tokens 32 --prompt "Once upon a time" --split 2.0 --k 2
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
    ap.add_argument("--lock", type=int, default=8)
    ap.add_argument("--split", type=float, default=2.0, help="branch when the lead is under this margin")
    ap.add_argument("--k", type=int, default=2, help="forks to branch on")
    ap.add_argument("--max-branches", type=int, default=8)
    a = ap.parse_args()
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    ids = tok(a.prompt, return_tensors="pt").input_ids[0].tolist()
    ref = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    with torch.no_grad():
        batch = ref.generate(torch.tensor([ids]), max_new_tokens=a.tokens, do_sample=False, pad_token_id=50256)[0].tolist()
    m, _ = em.build(65536, 4096, True, 16384, torch.tensor([ids]))
    tr = m.transformer; blocks = list(tr.h); L = len(blocks)

    # a branch: its own token list; all branches share the prefix up to the last finalized position
    branches = [list(ids)]
    start = [0] * len(ids)                       # per position (shared: every branch's position p started at the same tick)
    finalized_upto = len(ids)                    # positions < this are final
    emitted = None                               # list over layers of (B, S, D)
    history = []
    t, rollbacks, splits, max_alive = 0, 0, 0, 1
    gaps, last_commit = [], 0
    with torch.no_grad():
        while True:
            t += 1
            B, S = len(branches), len(branches[0])
            seqs = torch.tensor(branches)
            x0 = tr.wte(seqs) + tr.wpe(torch.arange(S))[None]
            sx = x0.abs().amax(-1, keepdim=True) + 1e-12
            width = torch.round(x0.abs() / sx * a.T)
            prog = torch.tensor([min(max(t - s, 0), a.T) for s in start], dtype=torch.float32)[None, :, None]
            inp = torch.sign(x0) * sx * torch.minimum(prog.expand_as(width), width) / a.T
            if emitted is None or emitted[0].shape[:2] != (B, S):
                old = emitted
                emitted = [torch.zeros_like(x0) for _ in range(L + 1)]
                if old is not None:
                    ob, os_ = old[0].shape[:2]
                    for kk in range(L + 1):
                        emitted[kk][:, :os_] = old[kk] if ob == B else old[kk][:1].expand(B, -1, -1)
            new = [inp]
            for kk in range(1, L + 1):
                out = blocks[kk - 1](emitted[kk - 1])
                new.append(out[0] if isinstance(out, tuple) else out)
            emitted = new
            logits = m.lm_head(tr.ln_f(emitted[L]))            # (B, S, V)

            # finalize the oldest struck position once it has settled
            p = finalized_upto
            if p < S and t >= start[p] + a.T + L:
                settled = logits[0, p - 1].argmax().item()      # all branches share the prefix before p
                keep = [i for i, br in enumerate(branches) if br[p] == settled]
                if keep:
                    if len(branches) > 1:
                        emitted = [e[keep] for e in emitted]; logits = logits[keep]
                    branches = [branches[i] for i in keep]
                    finalized_upto = p + 1
                else:
                    rollbacks += 1
                    print(f"  tick {t:4d}: position {p} settled to {tok.decode([settled])!r}; no branch had it -> un-strike", flush=True)
                    branches = [branches[0][:p] + [settled]]; start = start[:p] + [t + 1]
                    emitted = [e[:1, :p].clone() for e in emitted]; finalized_upto = p + 1; history = []
                    continue
                if len(branches) == 1:
                    emitted = [e[:1] for e in emitted]

            # strike the next note on the newest position of every branch
            top = logits[:, -1].argmax(-1).tolist()
            history.append(tuple(top))
            if (S - len(ids) + (1 if False else 0)) < a.tokens and t - start[-1] >= a.lock and all(h == tuple(top) for h in history[-a.lock:]):
                lead = logits[:, -1].topk(a.k, dim=-1)
                newb = []
                for i, br in enumerate(branches):
                    margin = (lead.values[i, 0] - lead.values[i, 1]).item()
                    if margin < a.split and len(newb) + a.k <= a.max_branches:
                        splits += 1
                        for c in lead.indices[i].tolist():
                            newb.append(br + [c])
                    else:
                        newb.append(br + [lead.indices[i, 0].item()])
                # replicate emitted rows for the new branches
                rows = []
                for i, br in enumerate(branches):
                    n = sum(1 for nb in newb if nb[:-1] == br)
                    rows += [i] * n
                emitted = [e[rows] for e in emitted]
                branches = newb; start.append(t + 1); history = []
                max_alive = max(max_alive, len(branches)); gaps.append(t - last_commit); last_commit = t
                print(f"  tick {t:4d}: position {S} struck {[tok.decode([b[-1]]) for b in branches]}  ({len(branches)} branch{'es' if len(branches)>1 else ''} flowing)", flush=True)
            if len(branches[0]) - len(ids) >= a.tokens and finalized_upto >= len(branches[0]):
                break
    gen = branches[0][len(ids):]
    print(f"\n  batch:    {tok.decode(batch[len(ids):])!r}")
    print(f"  branched: {tok.decode(gen)!r}   {'MATCH' if gen == batch[len(ids):] else 'DIFFERS'}")
    print(f"  splits {splits}, max branches alive {max_alive}, rollbacks {rollbacks}")
    print(f"  ticks/token: {sum(gaps)/len(gaps):.1f} between strikes, {t/a.tokens:.1f} all in (vs 141 one at a time)")


if __name__ == "__main__":
    main()

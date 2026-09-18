#!/usr/bin/env python3
"""PIPELINED: start the next token while this one is still flowing.

Same dial machine as dial_flow.py. Each position has its own start tick; its
input ramps in from there. When the newest position's fork has held for LOCK
turns, its token is committed and struck as the next input on the very next
turn, while every earlier position keeps flowing and settling behind it.
Several tokens are in flight at once. At the end everything settles and each
committed token is checked against the batch model and against what its own
position settled to.

    python3 dial_pipe.py --tokens 8 --T 128
"""
import argparse
import sys

import torch

sys.path.insert(0, "/Users/abundancemachine/tick-machine")
import edge_model as em            # noqa: E402

torch.set_num_threads(8)
LOCK = 8
MARGIN = 0.0        # strike only if the leading fork is ahead of the next by this much


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--lock", type=int, default=8)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--rollback", action="store_true", help="un-strike a note its position settles away from")
    a = ap.parse_args()
    global LOCK, MARGIN
    LOCK, MARGIN = a.lock, a.margin
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    ids = tok(a.prompt, return_tensors="pt").input_ids[0].tolist()
    ref = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    with torch.no_grad():
        batch = ref.generate(torch.tensor([ids]), max_new_tokens=a.tokens, do_sample=False, pad_token_id=50256)[0].tolist()
    m, _ = em.build(65536, 4096, True, 16384, torch.tensor([ids]))
    tr = m.transformer; blocks = list(tr.h); L = len(blocks)

    seq = list(ids)                                  # tokens present in the machine
    start = [0] * len(ids)                           # tick each position started arriving
    committed = []                                   # (position, tick committed)
    finalized, rollbacks = set(), 0
    emitted = None
    history = []                                     # top fork of the newest position, per tick
    t = 0
    with torch.no_grad():
        while True:
            t += 1
            S = len(seq)
            x0 = tr.wte(torch.tensor([seq])) + tr.wpe(torch.arange(S))[None]
            sx = x0.abs().amax(-1, keepdim=True) + 1e-12
            width = torch.round(x0.abs() / sx * a.T)
            prog = torch.tensor([min(max(t - s, 0), a.T) for s in start], dtype=torch.float32)[None, :, None]
            inp = torch.sign(x0) * sx * torch.minimum(prog.expand_as(width), width) / a.T
            if emitted is None or emitted[0].shape[1] != S:          # a new position joined: extend, keep the rest ringing
                old = emitted
                emitted = [torch.zeros_like(x0) for _ in range(L + 1)]
                if old is not None:
                    for k in range(L + 1):
                        emitted[k][:, :old[k].shape[1]] = old[k]
            new = [inp]
            for k in range(1, L + 1):
                out = blocks[k - 1](emitted[k - 1])
                new.append(out[0] if isinstance(out, tuple) else out)
            emitted = new
            logits = m.lm_head(tr.ln_f(emitted[L]))[0]
            top = logits[-1].argmax().item()
            history.append(top)
            if a.rollback:
                # the oldest struck-but-not-final position: final once T+L turns after it started
                for p, _ in committed:
                    if p in finalized or t < start[p - 1] + a.T + L:
                        continue
                    final_tok = logits[p - 1].argmax().item()
                    finalized.add(p)
                    if final_tok != seq[p]:
                        rollbacks += 1
                        print(f"  tick {t:4d}: position {p} settled to {tok.decode([final_tok])!r}, not {tok.decode([seq[p]])!r} "
                              f"-> un-strike, restart from there ({len(seq) - p - 1} later notes dropped)", flush=True)
                        seq = seq[:p] + [final_tok]; start = start[:p] + [t + 1]
                        committed = [(q, tc) for q, tc in committed if q < p] + [(p, t)]
                        finalized = {q for q in finalized if q < p}
                        for k in range(L + 1):
                            emitted[k] = emitted[k][:, :p].clone()
                        history = []
                    break
            newest_started = start[-1]
            lead = logits[-1].topk(2).values
            if (len(committed) < a.tokens and t - newest_started >= LOCK and all(h == top for h in history[-LOCK:])
                    and (lead[0] - lead[1]).item() >= MARGIN):
                seq.append(top); start.append(t + 1); committed.append((S, t)); history = []
                print(f"  tick {t:4d}: position {S} locked {tok.decode([top])!r} -> struck as input at tick {t+1}; "
                      f"{sum(1 for s in start if t - s < a.T + L)} positions still flowing", flush=True)
            if len(committed) == a.tokens and t >= start[-1] + a.T + L and (not a.rollback or len(finalized) == a.tokens):
                break
        settled = m.lm_head(tr.ln_f(emitted[L]))[0].argmax(-1).tolist()
    gen = seq[len(ids):]
    print(f"\n  batch:     {tok.decode(batch[len(ids):])!r}")
    print(f"  pipelined: {tok.decode(gen)!r}   {'MATCH' if gen == batch[len(ids):] else 'DIFFERS'}")
    changed = [p for p, _ in committed if settled[p - 1] != seq[p]]
    print(f"  committed tokens that their own position later settled away from: {len(changed)} of {len(committed)}")
    ticks = [tc for _, tc in committed]
    gaps = [ticks[0]] + [b - a_ for a_, b in zip(ticks, ticks[1:])]
    if a.rollback: print(f"  rollbacks: {rollbacks};  total ticks {t} for {a.tokens} tokens = {t/a.tokens:.1f} per token including settling")
    print(f"  ticks between commits: {gaps}  -> {sum(gaps)/len(gaps):.1f} per token pipelined, vs {a.T + L + 1} per token one at a time")


if __name__ == "__main__":
    main()

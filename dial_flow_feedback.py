#!/usr/bin/env python3
"""THE STREAM CARRIES THE WHOLE SEQUENCE. One operation per turn, every position at once.
Every generated position's input is the head's CURRENT ring at the position before it,
fed back every turn, before anything has locked. A new position is added every turn.
The wavefront corrects guesses that change. Measured: how many turns until all N
generated tokens equal the greedy reference and stay there -> tokens per turn.
    python3 dial_flow_feedback.py --case 0 --n 32
"""
import argparse, json, sys, torch
sys.path.insert(0, "/Users/abundancemachine/tick-machine")
import edge_model as em
torch.set_num_threads(8)
ap = argparse.ArgumentParser()
ap.add_argument("--case", type=int, default=0); ap.add_argument("--n", type=int, default=32)
ap.add_argument("--max-turns", type=int, default=200); ap.add_argument("--grow", type=int, default=1, help="new positions added per turn")
a = ap.parse_args()
from transformers import GPT2Tokenizer
tok = GPT2Tokenizer.from_pretrained("gpt2")
case = json.load(open("/Users/abundancemachine/tick-machine/gpt2_dial/reference.json"))[a.case]
ids, ref = list(case["ids"]), case["ref"][:a.n]; S = len(ids)
calib = tok("The capital of France is", return_tensors="pt").input_ids
m, _ = em.build(65536, 4096, True, 16384, calib); tr = m.transformer; blocks = list(tr.h); L = len(blocks)
guess = []                                   # current guess for each generated position
emitted = [None] * (L + 1)
first_all = None; history = []
with torch.no_grad():
    for t in range(1, a.max_turns + 1):
        if len(guess) < a.n and t > 1:
            for _ in range(a.grow):
                if len(guess) < a.n: guess.append(guess[-1] if guess else ids[-1])   # placeholder until the ring below it speaks
        seq = ids + guess
        new = [None] * (L + 1)
        new[0] = tr.wte(torch.tensor([seq])) + tr.wpe(torch.arange(len(seq)))[None]
        for k in range(1, L + 1):
            if emitted[k - 1] is None: break
            out = blocks[k - 1](emitted[k - 1]); new[k] = out[0] if isinstance(out, tuple) else out
        emitted = new
        if emitted[L] is not None:
            logits = m.lm_head(tr.ln_f(emitted[L]))[0]                     # every position's ring, this turn
            P = logits.shape[0]
            for j in range(len(guess)):                                     # position S+j reads the ring at S+j-1
                if S + j - 1 < P: guess[j] = int(logits[S + j - 1].argmax())
        ok = sum(int(g == r) for g, r in zip(guess, ref)); history.append(ok)
        if len(guess) == a.n and ok == a.n and first_all is None: first_all = t
        if first_all is not None and t >= first_all + 8 and all(h == a.n for h in history[-8:]): break
        if first_all is not None and history[-1] != a.n: first_all = None
print(f"case {a.case}: {case['prompt']!r}, {a.n} tokens, {a.grow} new position(s) per turn")
print("  turn: correct-of-N  " + " ".join(f"{t+1}:{h}" for t, h in enumerate(history) if (t + 1) % 4 == 0 or t + 1 == len(history)))
same = guess == ref
print(f"  final: {'IDENTICAL to fp32 greedy' if same else 'differs'}   {tok.decode(guess)!r}")
if first_all: print(f"  [MEASURED] all {a.n} tokens right and standing from turn {first_all}: {a.n/first_all:.2f} tokens per turn  (staged machine: {a.n} tokens = {a.n} full passes)")
else: print(f"  [MEASURED] never all right within {a.max_turns} turns; best {max(history)} of {a.n}")

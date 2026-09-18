#!/usr/bin/env python3
"""THE PILE MACHINE. No matmul anywhere in this file.

Every neuron is a PILE with a level on a 4096-tick dial (plus laps). The trained
notes are CHUTES from pile to pile. A pile POURS down its chutes only when its dial
moves by at least one tick (an EDGE); what it pours is the change. Bends are spill
laws applied by each pile to its own level. There are no layers, no turns: flips are
queued and processed in RANDOM order; the notes are never read except along the
chutes of a pile that flipped. The previous token's piles are left STANDING, so for
the next token only the changes flow. Previous positions' key/value piles stand too.

Check: same tokens as fp32 greedy GPT-2? How many flips and chute-walks per token?
    python3 piles.py --case 0 --n 8
"""
import argparse, json, math, sys, numpy as np, torch
rng = np.random.default_rng(0)
ap = argparse.ArgumentParser()
ap.add_argument("--case", type=int, default=0); ap.add_argument("--n", type=int, default=8)
ap.add_argument("--dial", type=int, default=4096); ap.add_argument("--order", default="random")
a = ap.parse_args()
from transformers import GPT2LMHeadModel, GPT2Tokenizer
tok = GPT2Tokenizer.from_pretrained("gpt2"); hf = GPT2LMHeadModel.from_pretrained("gpt2").eval()
sd = {k: v.detach().double().numpy() for k, v in hf.state_dict().items()}
case = json.load(open("/Users/abundancemachine/tick-machine/gpt2_dial/reference.json"))[a.case]
ids, ref = list(case["ids"]), case["ref"][:a.n]
HALF = a.dial // 2 - 1
L, D, NH, HD = 12, 768, 12, 64
COUNT = {"flips": 0, "walks": 0}

def order(idx):
    return rng.permutation(idx) if a.order == "random" else idx

class Piles:
    """A bank of piles fed by chutes W (in x out) from an upstream bank's EMITTED levels.
    level = bias + sum_i emitted_i * W[i]   -- maintained by pours, never recomputed."""
    def __init__(self, W, b, name):
        self.W, self.name = W, name
        self.level = b.copy() if b is not None else np.zeros(W.shape[1])
        self.emit_prev = np.zeros(W.shape[0])        # what upstream has poured so far
        self.unit = None                             # this bank's tick size (per pile), set from calibration
    def pour(self, emitted):                          # upstream emitted levels changed: pour the CHANGES down the chutes
        d = emitted - self.emit_prev
        flips = np.nonzero(d)[0]
        for i in order(flips):
            self.level += d[i] * self.W[i]           # one pile flipped: its grains run down its chutes
        COUNT["flips"] += len(flips); COUNT["walks"] += len(flips) * self.W.shape[1]
        self.emit_prev = emitted.copy()

def on_dial(x, unit):                                  # a pile's emitted level: its dial position, in ticks, with laps
    return np.round(x / unit) * unit

class Block:
    def __init__(self, k):
        p = f"transformer.h.{k}."
        self.ln1w, self.ln1b, self.ln2w, self.ln2b = sd[p+"ln_1.weight"], sd[p+"ln_1.bias"], sd[p+"ln_2.weight"], sd[p+"ln_2.bias"]
        self.qkv = Piles(sd[p+"attn.c_attn.weight"], sd[p+"attn.c_attn.bias"], "qkv")
        self.proj = Piles(sd[p+"attn.c_proj.weight"], sd[p+"attn.c_proj.bias"], "proj")
        self.fc = Piles(sd[p+"mlp.c_fc.weight"], sd[p+"mlp.c_fc.bias"], "fc")
        self.mproj = Piles(sd[p+"mlp.c_proj.weight"], sd[p+"mlp.c_proj.bias"], "mproj")
        self.K, self.V = [], []                        # standing key/value piles of earlier positions
        self.units = {}
    def unit(self, name, x):                           # calibration: the first token sets each bank's tick size
        if name not in self.units: self.units[name] = np.maximum(np.abs(x), 1e-6) / HALF
        return self.units[name]

def water(x, w, b):                                    # the depth's water level: LN as a pile-of-piles (sums only)
    m = x.mean(); v = ((x - m) ** 2).mean()
    return (x - m) / math.sqrt(v + 1e-5) * w + b

def gelu(x):                                           # the spill law
    return 0.5 * x * (1 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))

blocks = [Block(k) for k in range(L)]
lnfw, lnfb, wte, wpe = sd["transformer.ln_f.weight"], sd["transformer.ln_f.bias"], sd["transformer.wte.weight"], sd["transformer.wpe.weight"]
head = Piles(wte.T.copy(), None, "head")

def token_flow(pos, tid, new_pos):
    """One position flows through the piles. Piles are STANDING from the previous position."""
    x = wte[tid] + wpe[pos]
    for k, B in enumerate(blocks):
        n1 = water(x, B.ln1w, B.ln1b); e1 = on_dial(n1, B.unit("n1", n1))
        B.qkv.pour(e1)
        q, kk, v = B.qkv.level[:D], B.qkv.level[D:2*D], B.qkv.level[2*D:]
        if new_pos: B.K.append(kk.copy()); B.V.append(v.copy())
        else: B.K[-1], B.V[-1] = kk.copy(), v.copy()
        out = np.zeros(D)
        for h in range(NH):                            # each head: the query pile meets every standing key pile
            sl = slice(h*HD, (h+1)*HD); qh = q[sl]
            scores = np.array([(qh * Kt[sl]).sum() for Kt in B.K]) / math.sqrt(HD)   # a pour per standing position
            COUNT["walks"] += HD * len(B.K)
            alpha = np.exp(scores - scores.max()); alpha /= alpha.sum()               # spill across positions
            for s, Vt in enumerate(B.V): out[sl] += alpha[s] * Vt[sl]                # value piles pour by their share
            COUNT["walks"] += HD * len(B.V)
        eo = on_dial(out, B.unit("ao", out)); B.proj.pour(eo)
        x = x + B.proj.level
        n2 = water(x, B.ln2w, B.ln2b); e2 = on_dial(n2, B.unit("n2", n2)); B.fc.pour(e2)
        act = gelu(B.fc.level); ea = on_dial(act, B.unit("act", act)); B.mproj.pour(ea)
        x = x + B.mproj.level
    nf = water(x, lnfw, lnfb); ef = on_dial(nf, blocks[0].unit("nf", nf)); head.pour(ef)
    return int(np.argmax(head.level))

gen = []; per_tok = []
for i, t in enumerate(ids):                            # the prompt flows in, one position at a time (piles stand between)
    COUNT["flips"] = COUNT["walks"] = 0
    nxt = token_flow(i, t, True); per_tok.append((COUNT["flips"], COUNT["walks"]))
pos = len(ids)
while len(gen) < a.n:
    gen.append(nxt); COUNT["flips"] = COUNT["walks"] = 0
    nxt = token_flow(pos, nxt, True); per_tok.append((COUNT["flips"], COUNT["walks"])); pos += 1
same = gen == ref
print(f"pile machine, dial {a.dial}, flips in {a.order} order, {case['prompt']!r} + {a.n} tokens")
print(f"  {'IDENTICAL to fp32 greedy' if same else 'differs'}   {tok.decode(gen)!r}")
if not same: print(f"  reference:  {tok.decode(ref)!r}")
f, w = zip(*per_tok)
print(f"  [MEASURED] per token: flips {min(f):,}-{max(f):,} (mean {int(np.mean(f)):,});  chute-walks {min(w)/1e6:.0f}-{max(w)/1e6:.0f}M (mean {np.mean(w)/1e6:.0f}M) of 124M chutes")

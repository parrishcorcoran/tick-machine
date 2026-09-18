#!/usr/bin/env python3
"""RUN GPT-2 FROM THE CIRCLE ONLY. Loads gpt2_dial/circle.npz and nothing else: every note is an
unsigned integer on a 65,536 circle (angle + 16 laps, complements for negatives), every stream
value an integer number of ticks. A pile is an exact int64 stack of (input ticks x note); a gauge
turns a pile real only where a law (water level, bend, limiter, cleanup) needs it, and the law's
result goes straight back onto its dial as ticks. The residual stream is added as integer ticks.
    python3 circle_run.py --case 0 --n 4
"""
import argparse, json, math, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--case", type=int, default=0); ap.add_argument("--n", type=int, default=4); a = ap.parse_args()
from transformers import GPT2Tokenizer
tok = GPT2Tokenizer.from_pretrained("gpt2")
case = json.load(open("/Users/abundancemachine/tick-machine/gpt2_dial/reference.json"))[a.case]
ids, ref = list(case["ids"]), case["ref"][:a.n]
C = np.load("/Users/abundancemachine/tick-machine/gpt2_dial/circle.npz")
CIRCLE, L, D, NH, HD = 65536, 12, 768, 12, 64
u_res = C["u_res"]
class Notes:
    def __init__(self, name):
        circ = C[f"{name}.circle"].astype(np.int64)
        self.q = np.where(circ >= CIRCLE // 2, circ - CIRCLE, circ)             # complement -> signed position on the circle
        self.gauge, self.u_in, self.bias = C[f"{name}.gauge"], C[f"{name}.u_in"], C[f"{name}.bias"]
        import os; self.bias_real = C[f"{name}.bias_real"] if os.environ.get("BIAS_FLOAT") else None
        self.u_out = C[f"{name}.u_out"] if f"{name}.u_out" in C else None
    def stack(self, x_real):
        t = np.round(x_real / self.u_in).astype(np.int64)                        # the input as ticks on its dial (laps allowed)
        pile = t @ self.q + (0 if self.bias_real is not None else self.bias)     # exact integer stack of every landing
        y = pile * self.gauge + (self.bias_real if self.bias_real is not None else 0)   # real, for the law downstream
        return np.round(y / self.u_out).astype(np.int64) * self.u_out if self.u_out is not None else y   # back onto the output dial
def water(x, w, b): m = x.mean(); v = ((x - m) ** 2).mean(); return (x - m) / math.sqrt(v + 1e-5) * w + b
def gelu(x): return 0.5 * x * (1 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))
blocks = [dict(qkv=Notes(f"qkv.{k}"), proj=Notes(f"proj.{k}"), fc=Notes(f"fc.{k}"), mproj=Notes(f"mproj.{k}"),
               ln1=(C[f"ln.{k}.ln_1.weight"], C[f"ln.{k}.ln_1.bias"]), ln2=(C[f"ln.{k}.ln_2.weight"], C[f"ln.{k}.ln_2.bias"]), K=[], V=[]) for k in range(L)]
head = Notes("head"); lnf = (C["ln_f.weight"], C["ln_f.bias"]); wte_t, wpe_t = C["wte.ticks"], C["wpe.ticks"]
def to_res_ticks(y_real): return np.round(y_real / u_res).astype(np.int64)
def position(pos, tid):
    r = wte_t[tid] + wpe_t[pos]                                                  # the residual stream: integer ticks, stacked
    for B in blocks:
        qkv = B["qkv"].stack(water(r * u_res, *B["ln1"])); q, kk, v = qkv[:D], qkv[D:2*D], qkv[2*D:]
        B["K"].append(kk); B["V"].append(v); Kt = np.stack(B["K"]); Vt = np.stack(B["V"]); o = np.zeros(D)
        for h in range(NH):
            sl = slice(h*HD, (h+1)*HD); sc = (Kt[:, sl] * q[sl]).sum(1) / math.sqrt(HD); al = np.exp(sc - sc.max()); al /= al.sum(); o[sl] = (al[:, None] * Vt[:, sl]).sum(0)
        r = r + to_res_ticks(B["proj"].stack(o))
        r = r + to_res_ticks(B["mproj"].stack(gelu(B["fc"].stack(water(r * u_res, *B["ln2"])))))
    return int(np.argmax(head.stack(water(r * u_res, *lnf))))
gen = []
for i, t in enumerate(ids): nxt = position(i, t)
pos = len(ids)
while len(gen) < a.n: gen.append(nxt); nxt = position(pos, nxt); pos += 1
print(f"{case['prompt']!r} + {a.n}: {'IDENTICAL to fp32 greedy' if gen == ref else 'differs'}   {tok.decode(gen)!r}" + ("" if gen == ref else f"   (reference {tok.decode(ref)!r})"))

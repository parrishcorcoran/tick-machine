#!/usr/bin/env python3
"""THE LIGHT CONE ON THE CIRCLE. Everything is integer ticks now, so 'moved' is exact: a dial's
tick count changed. Kick ONE input phase (dimension `dim` of the last position) by k ticks and
watch, at every depth, how many dials moved and by how many ticks."""
import json, math, numpy as np, sys
from transformers import GPT2Tokenizer
tok = GPT2Tokenizer.from_pretrained("gpt2")
C = np.load("/Users/abundancemachine/tick-machine/gpt2_dial/circle.npz"); CIRCLE, L, D, NH, HD = 65536, 12, 768, 12, 64; u_res = C["u_res"]
class Notes:
    def __init__(s, n):
        c = C[f"{n}.circle"].astype(np.int64); s.q = np.where(c >= CIRCLE // 2, c - CIRCLE, c)
        s.gauge, s.u_in, s.bias = C[f"{n}.gauge"], C[f"{n}.u_in"], C[f"{n}.bias"]; s.u_out = C[f"{n}.u_out"] if f"{n}.u_out" in C else None
    def stack(s, x):
        t = np.round(x / s.u_in).astype(np.int64); y = (t @ s.q + s.bias) * s.gauge
        if s.u_out is None: return y, None
        k = np.round(y / s.u_out).astype(np.int64); return k * s.u_out, k
def water(x, w, b): m = x.mean(); v = ((x - m) ** 2).mean(); return (x - m) / math.sqrt(v + 1e-5) * w + b
def gelu(x): return 0.5 * x * (1 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))
blocks = [dict(qkv=Notes(f"qkv.{k}"), proj=Notes(f"proj.{k}"), fc=Notes(f"fc.{k}"), mproj=Notes(f"mproj.{k}"), ln1=(C[f"ln.{k}.ln_1.weight"], C[f"ln.{k}.ln_1.bias"]), ln2=(C[f"ln.{k}.ln_2.weight"], C[f"ln.{k}.ln_2.bias"])) for k in range(L)]
head = Notes("head"); lnf = (C["ln_f.weight"], C["ln_f.bias"]); wte_t, wpe_t = C["wte.ticks"], C["wpe.ticks"]
ids = json.load(open("/Users/abundancemachine/tick-machine/gpt2_dial/reference.json"))[0]["ids"]
def run(kick):                                   # kick: (768,) integer ticks added to the last position's chord; returns every dial's ticks
    Ks = [[] for _ in range(L)]; Vs = [[] for _ in range(L)]; rec = {}
    for pos, tid in enumerate(ids):
        r = wte_t[tid] + wpe_t[pos] + (kick if pos == len(ids) - 1 else 0)
        for k, B in enumerate(blocks):
            qkv, tq = B["qkv"].stack(water(r * u_res, *B["ln1"])); q, kk, v = qkv[:D], qkv[D:2*D], qkv[2*D:]
            Ks[k].append(kk); Vs[k].append(v); Kt = np.stack(Ks[k]); Vt = np.stack(Vs[k]); o = np.zeros(D)
            for h in range(NH):
                sl = slice(h*HD, (h+1)*HD); sc = (Kt[:, sl] * q[sl]).sum(1) / math.sqrt(HD); al = np.exp(sc - sc.max()); al /= al.sum(); o[sl] = (al[:, None] * Vt[:, sl]).sum(0)
            p, tp = B["proj"].stack(o); r = r + np.round(p / u_res).astype(np.int64)
            f, tf = B["fc"].stack(water(r * u_res, *B["ln2"])); m, tm = B["mproj"].stack(gelu(f)); r = r + np.round(m / u_res).astype(np.int64)
            if pos == len(ids) - 1: rec[k] = dict(qkv=tq, attn=tp, fc=tf, mlp=tm, res=r.copy())
    lg, _ = head.stack(water(r * u_res, *lnf)); rec["head"] = lg; return rec
base = run(np.zeros(D, dtype=np.int64)); dim = int(sys.argv[1]) if len(sys.argv) > 1 else 100
srt = np.sort(base["head"])[::-1]; print(f"baseline head: {tok.decode([int(np.argmax(base['head']))])!r}, margin over the next fork {srt[0]-srt[1]:.3f}")
print(f"\nONE phase kicked (input dim {dim}); per depth: share of dials that moved (exact), and among the moved, the mean |ticks| moved")
print(f"{'kick k':>7} | " + " | ".join(f"{('depth '+str(d)):>14}" for d in (0, 1, 2, 4, 7, 11)) + " |  head: top-1, Δmargin")
for k in (1, 4, 16, 64, 256, 1024, 2047):
    kick = np.zeros(D, dtype=np.int64); kick[dim] = k; r = run(kick); cells = []
    for d in (0, 1, 2, 4, 7, 11):
        mv = np.concatenate([(r[d][s] != base[d][s]) for s in ("qkv", "attn", "fc", "mlp", "res")])
        dt = np.concatenate([np.abs(r[d][s] - base[d][s]) for s in ("qkv", "attn", "fc", "mlp", "res")])
        cells.append(f"{mv.mean():6.1%} {dt[mv].mean() if mv.any() else 0:6.1f}")
    s2 = np.sort(r["head"])[::-1]; print(f"{k:>7} | " + " | ".join(f"{c:>14}" for c in cells) + f" |  {tok.decode([int(np.argmax(r['head']))])!r:8s} {s2[0]-s2[1]-(srt[0]-srt[1]):+.3f}")

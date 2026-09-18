#!/usr/bin/env python3
"""THE LAP MACHINE. One rotator sweeps 4096 angles. At tick theta EVERY chute in the whole
model whose note is theta lights at once (one superposed pour) and drops whatever change is
waiting at it. A negative note is the same chute pouring the other way (complement). Piles are
abelian. A pile whose dial moved leaves its change waiting at all its chutes; each drops it
when its own angle comes round. Spill laws (water level, attention, GELU) act on the piles as
they move. No matmul in this file. Measured: LAPS until the whole model is quiet, per token.
    python3 -u lap_machine.py --case 0 --n 1
"""
import argparse, json, math, sys, time, numpy as np, torch
ap = argparse.ArgumentParser()
ap.add_argument("--case", type=int, default=0); ap.add_argument("--n", type=int, default=1); ap.add_argument("--max-laps", type=int, default=30)
a = ap.parse_args()
from transformers import GPT2LMHeadModel, GPT2Tokenizer
tok = GPT2Tokenizer.from_pretrained("gpt2"); hf = GPT2LMHeadModel.from_pretrained("gpt2").eval()
sd = {k: v.detach().double().numpy() for k, v in hf.state_dict().items()}
case = json.load(open("/Users/abundancemachine/tick-machine/gpt2_dial/reference.json"))[a.case]
ids, ref = list(case["ids"]), case["ref"][:a.n]
DIAL, HALF = 4096, 2047; L, D, NH, HD, V = 12, 768, 12, 64, 50257
cal = {}
def hook(name):
    def f(mod, inp, out): cal[name] = inp[0].detach()[0].abs().amax(0).double().numpy()
    return f
for k, blk in enumerate(hf.transformer.h):
    blk.attn.c_attn.register_forward_hook(hook(f"n1.{k}")); blk.attn.c_proj.register_forward_hook(hook(f"ao.{k}"))
    blk.mlp.c_fc.register_forward_hook(hook(f"n2.{k}")); blk.mlp.c_proj.register_forward_hook(hook(f"act.{k}"))
hf.lm_head.register_forward_hook(hook("nf"))
with torch.no_grad(): hf(tok("The capital of France is", return_tensors="pt").input_ids)
unit = {k: (np.maximum(v, 1e-6) / HALF).astype(np.float32) for k, v in cal.items()}

t0 = time.time(); W = np.load("/Users/abundancemachine/tick-machine/gpt2_dial/wiring.npz")
SRC, DST, VAL, RB, OFF = W["src"], W["dst"], W["val"], W["rb"], W["off"]
E_off = dict(zip(W["E_names"], W["E_offs"].tolist())); R_off = dict(zip(W["R_names"], W["R_offs"].tolist()))
nE = max(E_off.values()) + D; nR = max(R_off.values()) + V; NB = 4*L + 1
EMIT = np.zeros(nE, dtype=np.float32); LEVEL = W["bias"].astype(np.float64); LAST = np.zeros(len(SRC), dtype=np.float32)
print(f"lap machine: {len(SRC)/1e6:.1f}M chutes loaded, sorted by angle ({time.time()-t0:.0f}s); {DIAL} ticks per lap", flush=True)

def tick(th):
    s, e = OFF[th], OFF[th+1]
    src = SRC[s:e]; d = EMIT[src] - LAST[s:e]; nz = np.flatnonzero(d)
    if len(nz) == 0: return 0, None
    LEVEL[:] += np.bincount(DST[s:e][nz], weights=(d[nz] * VAL[s:e][nz]), minlength=nR)
    LAST[s:e][nz] = EMIT[src[nz]]
    return len(nz), np.bincount(RB[s:e][nz], minlength=NB) > 0

def R(name): return R_off[name]
def on_dial(x, u): return (np.round(x / u) * u).astype(np.float32)
def water(x, w, b): m = x.mean(); v = ((x - m) ** 2).mean(); return (x - m) / math.sqrt(v + 1e-5) * w + b
def gelu(x): return 0.5 * x * (1 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))
def set_emit(name, x, u):
    o = E_off[name]; e = on_dial(x, u); n = int((e != EMIT[o:o+len(e)]).sum()); EMIT[o:o+len(e)] = e; return n
wte, wpe = sd["transformer.wte.weight"], sd["transformer.wpe.weight"]
LN = [(sd[f"transformer.h.{k}.ln_1.weight"], sd[f"transformer.h.{k}.ln_1.bias"], sd[f"transformer.h.{k}.ln_2.weight"], sd[f"transformer.h.{k}.ln_2.bias"]) for k in range(L)]
LNF = (sd["transformer.ln_f.weight"], sd["transformer.ln_f.bias"])
Kst = [[] for _ in range(L)]; Vst = [[] for _ in range(L)]
r_in = [np.zeros(D) for _ in range(L)]; r_changed = [False] * L

def spill(pos, changed):
    edges = 0
    for k in range(L):
        qkv_c, proj_c, fc_c, mproj_c = changed[4*k], changed[4*k+1], changed[4*k+2], changed[4*k+3]
        if qkv_c:
            o = R(f"qkv.{k}"); q, kk, v = LEVEL[o:o+D], LEVEL[o+D:o+2*D], LEVEL[o+2*D:o+3*D]
            Kst[k][pos], Vst[k][pos] = kk.copy(), v.copy()
            Kt = np.stack(Kst[k][:pos+1]); Vt = np.stack(Vst[k][:pos+1]); out = np.zeros(D)
            for h in range(NH):
                sl = slice(h*HD, (h+1)*HD); sc = (Kt[:, sl] * q[sl]).sum(1) / math.sqrt(HD); al = np.exp(sc - sc.max()); al /= al.sum()
                out[sl] = (al[:, None] * Vt[:, sl]).sum(0)
            edges += set_emit(f"ao.{k}", out, unit[f"ao.{k}"])
        if proj_c or r_changed[k]:
            r_changed[k] = False; o = R(f"proj.{k}")
            edges += set_emit(f"n2.{k}", water(r_in[k] + LEVEL[o:o+D], LN[k][2], LN[k][3]), unit[f"n2.{k}"]); mproj_c = True
        if fc_c:
            o = R(f"fc.{k}"); edges += set_emit(f"act.{k}", gelu(LEVEL[o:o+4*D]), unit[f"act.{k}"])
        if mproj_c:
            o = R(f"proj.{k}"); o2 = R(f"mproj.{k}"); r_next = r_in[k] + LEVEL[o:o+D] + LEVEL[o2:o2+D]
            if k + 1 < L:
                r_in[k+1] = r_next; r_changed[k+1] = True
                edges += set_emit(f"n1.{k+1}", water(r_next, LN[k+1][0], LN[k+1][1]), unit[f"n1.{k+1}"])
            else:
                edges += set_emit("nf", water(r_next, *LNF), unit["nf"])
    return edges

def run_position(pos, tid):
    for k in range(L):
        while len(Kst[k]) <= pos: Kst[k].append(np.zeros(D)); Vst[k].append(np.zeros(D))
    x = wte[tid] + wpe[pos]; r_in[0] = x; r_changed[0] = True
    set_emit("n1.0", water(x, LN[0][0], LN[0][1]), unit["n1.0"]); spill(pos, [False] * NB)
    laps = []
    for lap in range(a.max_laps):
        drops = edges = 0; t0 = time.time()
        for th in range(DIAL):
            n, changed = tick(th); drops += n
            if changed is not None: edges += spill(pos, changed)
        o = R("head"); say = int(np.argmax(LEVEL[o:o+V])); laps.append((drops, edges))
        print(f"    pos {pos} lap {lap+1}: chute drops {drops:>11,}   edges {edges:>8,}   head says {tok.decode([say])!r:12s} ({time.time()-t0:.1f}s)", flush=True)
        if drops == 0: break
    return say, laps

gen = []; report = []
for i, t in enumerate(ids):
    nxt, laps = run_position(i, t); report.append(len(laps))
pos = len(ids)
while len(gen) < a.n:
    gen.append(nxt); nxt, laps = run_position(pos, nxt); report.append(len(laps)); pos += 1
print(f"\n{case['prompt']!r} + {a.n}: {'IDENTICAL to fp32 greedy' if gen == ref else 'differs'}   {tok.decode(gen)!r}   (reference {tok.decode(ref)!r})")
print(f"  [MEASURED] laps until quiet, per position: {report}   (a lap = {DIAL} ticks; quiet = a lap with 0 drops)")

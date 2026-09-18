#!/usr/bin/env python3
"""THE REVERSE ROTATOR: an adder with a delay bucket in one. NO MULTIPLY BY A NOTE ANYWHERE.

A pile has a pour register R and a level. A chute is a wire from a source pile to this pile
that OPENS at one tick of the lap (its note = when) and stays open to the end of the lap.
When it opens, the source's change is added to R (an add). Every tick, R is added to the
level (the delay bucket integrating). So a source change e through a chute that opened
with `angle` ticks left in the lap sweeps out e x angle by the end of the lap: the product
is TIME, never computed. Laps of the note (magnitude beyond one turn) open at tick 0 and
pour for the whole turn. A negative note is the same wire pouring the other way (complement).
At the end of the lap every pile reads its level once (the reverse rotator sends it back as
one edge) and the next depth's chutes take the change on the next lap.
    python3 -u reverse_rotator.py --case 0 --positions 1
"""
import argparse, json, math, sys, time, numpy as np, torch
ap = argparse.ArgumentParser(); ap.add_argument("--case", type=int, default=0); ap.add_argument("--positions", type=int, default=1); ap.add_argument("--n", type=int, default=0); ap.add_argument("--max-laps", type=int, default=20)
a = ap.parse_args()
from transformers import GPT2LMHeadModel, GPT2Tokenizer
tok = GPT2Tokenizer.from_pretrained("gpt2"); hf = GPT2LMHeadModel.from_pretrained("gpt2").eval()
sd = {k: v.detach().double().numpy() for k, v in hf.state_dict().items()}
case = json.load(open("/Users/abundancemachine/tick-machine/gpt2_dial/reference.json"))[a.case]
ids = list(case["ids"])[:a.positions] if a.n == 0 else list(case["ids"]); ref = case["ref"][:a.n]
DIAL, HALF, WHALF = 4096, 2047, 32767; L, D, NH, HD, V = 12, 768, 12, 64, 50257
cal = {}
def hook(name):
    def f(mod, inp, out): cal[name] = inp[0].detach()[0].abs().amax(0).double().numpy()
    return f
for k, blk in enumerate(hf.transformer.h):
    blk.attn.c_attn.register_forward_hook(hook(f"n1.{k}")); blk.attn.c_proj.register_forward_hook(hook(f"ao.{k}")); blk.mlp.c_fc.register_forward_hook(hook(f"n2.{k}")); blk.mlp.c_proj.register_forward_hook(hook(f"act.{k}"))
hf.lm_head.register_forward_hook(hook("nf"))
with torch.no_grad(): hf(tok("The capital of France is", return_tensors="pt").input_ids)
unit = {k: (np.maximum(v, 1e-6) / HALF) for k, v in cal.items()}

# ---- the wiring: each chute = (source pile, destination pile, OPEN TICK, laps, sign). No note value is stored.
t0 = time.time(); E_off, R_off = {}, {}; nE = nR = 0
for k in range(L):
    for n, sz in ((f"n1.{k}", D), (f"ao.{k}", D), (f"n2.{k}", D), (f"act.{k}", 4*D)): E_off[n] = nE; nE += sz
    for n, sz in ((f"qkv.{k}", 3*D), (f"proj.{k}", D), (f"fc.{k}", 4*D), (f"mproj.{k}", D)): R_off[n] = nR; nR += sz
E_off["nf"] = nE; nE += D; R_off["head"] = nR; nR += V
GAUGE = np.ones(nR); BIAS = np.zeros(nR); parts = []
def wire(W, b, en, rn):
    nin, nout = W.shape; cyc = np.maximum(np.abs(W).max(0), 1e-12); q = np.round(W / cyc * WHALF).astype(np.int64)
    mag = np.abs(q); angle = mag % DIAL; laps = mag // DIAL; sgn = np.sign(q).astype(np.int8)
    open_tick = np.where(angle == 0, DIAL, DIAL - angle).astype(np.int16)    # opens with `angle` ticks left in the lap; angle 0 never opens
    src = np.repeat(np.arange(nin, dtype=np.int32), nout) + E_off[en]; dst = np.tile(np.arange(nout, dtype=np.int32), nin) + R_off[rn]
    parts.append((open_tick.ravel(), src, dst, laps.ravel().astype(np.int8), sgn.ravel(), (angle.ravel() == 0)))
    GAUGE[R_off[rn]:R_off[rn]+nout] = cyc / WHALF                          # the pile's own gauge (per column scale), applied at read-out
    if b is not None: BIAS[R_off[rn]:R_off[rn]+nout] = b
for k in range(L):
    p = f"transformer.h.{k}."
    wire(sd[p+"attn.c_attn.weight"], sd[p+"attn.c_attn.bias"], f"n1.{k}", f"qkv.{k}"); wire(sd[p+"attn.c_proj.weight"], sd[p+"attn.c_proj.bias"], f"ao.{k}", f"proj.{k}")
    wire(sd[p+"mlp.c_fc.weight"], sd[p+"mlp.c_fc.bias"], f"n2.{k}", f"fc.{k}"); wire(sd[p+"mlp.c_proj.weight"], sd[p+"mlp.c_proj.bias"], f"act.{k}", f"mproj.{k}")
wire(sd["transformer.wte.weight"].T.copy(), None, "nf", "head")
OPEN = np.concatenate([p[0] for p in parts]); order = np.argsort(OPEN, kind="stable")
SRC = np.concatenate([p[1] for p in parts])[order]; DST = np.concatenate([p[2] for p in parts])[order]; LAPS = np.concatenate([p[3] for p in parts])[order]; SGN = np.concatenate([p[4] for p in parts])[order]
OFF = np.searchsorted(OPEN[order], np.arange(DIAL + 1)); del OPEN, order, parts   # chutes at index DIAL (angle 0) are past the last tick
has_laps = np.flatnonzero(LAPS)                                             # chutes whose note is more than one turn: they also pour all lap
EMIT = np.zeros(nE, dtype=np.float32); LAST = np.zeros(len(SRC), dtype=np.float32); LAST_L = np.zeros(len(has_laps), dtype=np.float32); LEVEL = BIAS / GAUGE
SRC_L, DST_L, W_L = SRC[has_laps], DST[has_laps], (SGN[has_laps].astype(np.float64) * LAPS[has_laps])
print(f"reverse rotator: {len(SRC)/1e6:.1f}M chutes, each = (from, to, open tick, laps, sign); no note values stored ({time.time()-t0:.0f}s)", flush=True)

def on_dial(x, u): return (np.round(x / u) * u).astype(np.float32)
def water(x, w, b): m = x.mean(); v = ((x - m) ** 2).mean(); return (x - m) / math.sqrt(v + 1e-5) * w + b
def gelu(x): return 0.5 * x * (1 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))
wte, wpe = sd["transformer.wte.weight"], sd["transformer.wpe.weight"]
LN = [(sd[f"transformer.h.{k}.ln_1.weight"], sd[f"transformer.h.{k}.ln_1.bias"], sd[f"transformer.h.{k}.ln_2.weight"], sd[f"transformer.h.{k}.ln_2.bias"]) for k in range(L)]
LNF = (sd["transformer.ln_f.weight"], sd["transformer.ln_f.bias"])
Kst = [[] for _ in range(L)]; Vst = [[] for _ in range(L)]; r_in = [np.zeros(D) for _ in range(L)]
def lv(name, n): o = R_off[name]; return LEVEL[o:o+n] * GAUGE[o:o+n]     # read a bank of piles in real units (its gauge)
def set_emit(name, x, u):
    o = E_off[name]; e = on_dial(x, u); n = int((e != EMIT[o:o+len(e)]).sum()); EMIT[o:o+len(e)] = e; return n

def lap():
    """One turn of the rotator. Returns chute events (opens with a change waiting)."""
    R = np.zeros(nR); events = 0
    d_all = EMIT[SRC_L] - LAST_L                                             # tick 0: a note of `laps` turns pours `laps` copies for the whole turn
    nz = np.flatnonzero(d_all)
    if len(nz): R += np.bincount(DST_L[nz], weights=d_all[nz].astype(np.float64) * W_L[nz], minlength=nR); LAST_L[nz] = EMIT[SRC_L[nz]]; events += len(nz)
    for th in range(DIAL):
        s, e = OFF[th], OFF[th+1]
        if s < e:
            src = SRC[s:e]; d = EMIT[src] - LAST[s:e]; nz = np.flatnonzero(d)
            if len(nz):
                R += np.bincount(DST[s:e][nz], weights=d[nz].astype(np.float64) * SGN[s:e][nz], minlength=nR)   # the chute OPENS: an add, no note value
                LAST[s:e][nz] = EMIT[src[nz]]; events += len(nz)
        LEVEL[:] += R                                                        # the delay bucket: whatever is pouring, every tick
    return events

def readout(pos):
    """End of lap: every pile reads its level once and emits its dial (the reverse rotator's one edge back)."""
    edges = 0
    for k in range(L):
        q, kk, v = lv(f"qkv.{k}", 3*D)[:D], lv(f"qkv.{k}", 3*D)[D:2*D], lv(f"qkv.{k}", 3*D)[2*D:]
        Kst[k][pos], Vst[k][pos] = kk.copy(), v.copy(); Kt = np.stack(Kst[k][:pos+1]); Vt = np.stack(Vst[k][:pos+1]); out = np.zeros(D)
        for h in range(NH):
            sl = slice(h*HD, (h+1)*HD); sc = (Kt[:, sl] * q[sl]).sum(1) / math.sqrt(HD); al = np.exp(sc - sc.max()); al /= al.sum(); out[sl] = (al[:, None] * Vt[:, sl]).sum(0)
        edges += set_emit(f"ao.{k}", out, unit[f"ao.{k}"])
        rmid = r_in[k] + lv(f"proj.{k}", D); edges += set_emit(f"n2.{k}", water(rmid, LN[k][2], LN[k][3]), unit[f"n2.{k}"])
        edges += set_emit(f"act.{k}", gelu(lv(f"fc.{k}", 4*D)), unit[f"act.{k}"])
        r_next = rmid + lv(f"mproj.{k}", D)
        if k + 1 < L: r_in[k+1] = r_next; edges += set_emit(f"n1.{k+1}", water(r_next, LN[k+1][0], LN[k+1][1]), unit[f"n1.{k+1}"])
        else: edges += set_emit("nf", water(r_next, *LNF), unit["nf"])
    return edges

def run_position(pos, tid):
    for k in range(L):
        while len(Kst[k]) <= pos: Kst[k].append(np.zeros(D)); Vst[k].append(np.zeros(D))
    x = wte[tid] + wpe[pos]; r_in[0] = x; set_emit("n1.0", water(x, LN[0][0], LN[0][1]), unit["n1.0"])
    for n in range(a.max_laps):
        t0 = time.time(); ev = lap(); ed = readout(pos); say = int(np.argmax(lv("head", V)))
        print(f"    pos {pos} lap {n+1:2d}: chutes opened with a change {ev:>11,}   piles that moved a tick {ed:>7,}   head says {tok.decode([say])!r:12s} ({time.time()-t0:.1f}s)", flush=True)
        if ev == 0: return say, n + 1
    return say, a.max_laps

with torch.no_grad(): truth = hf(torch.tensor([ids])).logits[0]
gen = []; laps_per = []
for i, t in enumerate(ids):
    say, n = run_position(i, t); laps_per.append(n)
    print(f"  position {i} ({tok.decode([t])!r}): quiet after {n} laps; head {tok.decode([say])!r}  vs fp32 {tok.decode([int(truth[i].argmax())])!r}  {'MATCH' if say == int(truth[i].argmax()) else 'DIFFERS'}", flush=True)
pos = len(ids); nxt = say
while len(gen) < a.n:
    gen.append(nxt); nxt, n = run_position(pos, nxt); laps_per.append(n); pos += 1
if a.n: print(f"\n{case['prompt']!r} + {a.n}: {'IDENTICAL to fp32 greedy' if gen == ref else 'differs'}   {tok.decode(gen)!r}")
print(f"  [MEASURED] laps until quiet per position: {laps_per}; no multiply by a note anywhere; a note is only WHEN its chute opens")

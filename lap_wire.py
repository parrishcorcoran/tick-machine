#!/usr/bin/env python3
"""Wire the rotator ONCE: every chute of GPT-2 (src pile, dst pile, note value) sorted by
its note's ANGLE on the 4096 dial, so that a tick of the lap is one slice. Cached to disk."""
import time, numpy as np, torch
from transformers import GPT2LMHeadModel
t0 = time.time()
hf = GPT2LMHeadModel.from_pretrained("gpt2").eval(); sd = {k: v.detach().double().numpy() for k, v in hf.state_dict().items()}
DIAL, WHALF, L, D, V = 4096, 32767, 12, 768, 50257
E_off, R_off = {}, {}; nE = nR = 0
for k in range(L):
    for n, sz in ((f"n1.{k}", D), (f"ao.{k}", D), (f"n2.{k}", D), (f"act.{k}", 4*D)): E_off[n] = nE; nE += sz
    for n, sz in ((f"qkv.{k}", 3*D), (f"proj.{k}", D), (f"fc.{k}", 4*D), (f"mproj.{k}", D)): R_off[n] = nR; nR += sz
E_off["nf"] = nE; nE += D; R_off["head"] = nR; nR += V
bias = np.zeros(nR)
parts = []
def chutes(W, b, en, rn, bid):
    nin, nout = W.shape; cyc = np.maximum(np.abs(W).max(0), 1e-12); q = np.round(W / cyc * WHALF)
    ang = (np.abs(q).astype(np.int32) % DIAL).ravel().astype(np.int16)
    src = (np.repeat(np.arange(nin, dtype=np.int32), nout) + E_off[en]); dst = (np.tile(np.arange(nout, dtype=np.int32), nin) + R_off[rn])
    val = ((q / WHALF) * cyc).ravel().astype(np.float32); rb = np.full(nin * nout, bid, dtype=np.int8)
    if b is not None: bias[R_off[rn]:R_off[rn]+nout] = b
    parts.append((ang, src, dst, val, rb))
for k in range(L):
    p = f"transformer.h.{k}."
    chutes(sd[p+"attn.c_attn.weight"], sd[p+"attn.c_attn.bias"], f"n1.{k}", f"qkv.{k}", 4*k)
    chutes(sd[p+"attn.c_proj.weight"], sd[p+"attn.c_proj.bias"], f"ao.{k}", f"proj.{k}", 4*k+1)
    chutes(sd[p+"mlp.c_fc.weight"], sd[p+"mlp.c_fc.bias"], f"n2.{k}", f"fc.{k}", 4*k+2)
    chutes(sd[p+"mlp.c_proj.weight"], sd[p+"mlp.c_proj.bias"], f"act.{k}", f"mproj.{k}", 4*k+3)
chutes(sd["transformer.wte.weight"].T.copy(), None, "nf", "head", 4*L)
ANG = np.concatenate([p[0] for p in parts]); order = np.argsort(ANG, kind="stable")
out = {}
for i, name in enumerate(("ang", "src", "dst", "val", "rb")): out[name] = np.concatenate([p[i] for p in parts])[order]
out["off"] = np.searchsorted(out["ang"], np.arange(DIAL + 1)); del out["ang"]
out["bias"] = bias; out["E_names"] = np.array(list(E_off.keys())); out["E_offs"] = np.array(list(E_off.values())); out["R_names"] = np.array(list(R_off.keys())); out["R_offs"] = np.array(list(R_off.values()))
np.savez("/Users/abundancemachine/tick-machine/gpt2_dial/wiring.npz", **out)
print(f"wired {len(order)/1e6:.1f}M chutes by angle in {time.time()-t0:.0f}s; chutes per tick min {np.diff(out['off']).min():,} max {np.diff(out['off']).max():,}", flush=True)

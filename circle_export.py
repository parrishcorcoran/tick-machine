#!/usr/bin/env python3
"""EXPORT GPT-2 TO THE CIRCLE. Every weight becomes ONE unsigned integer on a 65,536 circle:
      number = angle (0..4095) + 4096 * laps (0..15);   negatives are complements (65536 - |q|)
The input's dial unit is folded into the notes first (a note is "per input tick"), and each
column's loudest note is a full 8 turns (32767), so one column gauge says what a tick is worth.
Biases are integer ticks on the same gauge. The residual stream, token and position embeddings
are integer ticks on per-dimension dials (4096 + laps). Layer-norm gains/biases stay as the laws
they are. Output: gpt2_dial/circle.npz  +  a summary of what the circle holds.
"""
import numpy as np, torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
tok = GPT2Tokenizer.from_pretrained("gpt2"); hf = GPT2LMHeadModel.from_pretrained("gpt2").eval()
sd = {k: v.detach().double().numpy() for k, v in hf.state_dict().items()}
import os
CIRCLE, DIAL, HALF, WHALF, HEAD_HALF = 65536, 4096, 2047, 32767, 8191
LN_HALF = int(os.environ.get("LN_HALF", "8191")); RES_HALF = int(os.environ.get("RES_HALF", "2047"))   # dials fed by a water level, and the residual stream
L, D = 12, 768
# --- calibration: one standard pass gives every dial its unit (per dimension max / 2047)
cal_in, cal_out, res = {}, {}, []
def hook(name):
    def f(mod, inp, out): cal_in[name] = inp[0].detach()[0].abs().amax(0).double().numpy(); cal_out[name] = out.detach()[0].abs().amax(0).double().numpy()
    return f
def rhook(mod, inp, out): res.append((out[0] if isinstance(out, tuple) else out).detach()[0].abs().amax(0).double().numpy())
for k, blk in enumerate(hf.transformer.h):
    blk.attn.c_attn.register_forward_hook(hook(f"qkv.{k}")); blk.attn.c_proj.register_forward_hook(hook(f"proj.{k}"))
    blk.mlp.c_fc.register_forward_hook(hook(f"fc.{k}")); blk.mlp.c_proj.register_forward_hook(hook(f"mproj.{k}")); blk.register_forward_hook(rhook)
hf.lm_head.register_forward_hook(hook("head"))
with torch.no_grad(): hf(tok("The capital of France is", return_tensors="pt").input_ids)
u_res = np.maximum(np.max(res, axis=0), 1e-6) / RES_HALF                        # the residual stream's dial, per dimension
out = {"u_res": u_res}
stats = {"notes": 0, "complements": 0, "laps": np.zeros(16, dtype=np.int64)}
def to_circle(name, W, b, in_half, out_half):
    u_in = np.maximum(cal_in[name], 1e-6) / in_half                          # input dial unit per dimension
    Wu = W * u_in[:, None]                                                    # note per input TICK
    cyc = np.maximum(np.abs(Wu).max(0), 1e-12); q = np.round(Wu / cyc * WHALF).astype(np.int64)   # |q| <= 32767
    circle = np.where(q < 0, CIRCLE + q, q).astype(np.uint16)                 # complements
    gauge = (cyc / WHALF)                                                     # value of one tick, per column
    out[f"{name}.circle"] = circle; out[f"{name}.gauge"] = gauge.astype(np.float64); out[f"{name}.u_in"] = u_in
    out[f"{name}.bias"] = (np.round(b / gauge).astype(np.int64) if b is not None else np.zeros(W.shape[1], dtype=np.int64))
    out[f"{name}.bias_real"] = (b if b is not None else np.zeros(W.shape[1]))
    if out_half: out[f"{name}.u_out"] = np.maximum(cal_out[name], 1e-6) / out_half
    stats["notes"] += q.size; stats["complements"] += int((q < 0).sum()); stats["laps"] += np.bincount((np.abs(q) // DIAL).ravel(), minlength=16)
for k in range(L):
    p = f"transformer.h.{k}."
    to_circle(f"qkv.{k}", sd[p+"attn.c_attn.weight"], sd[p+"attn.c_attn.bias"], LN_HALF, HALF)
    to_circle(f"proj.{k}", sd[p+"attn.c_proj.weight"], sd[p+"attn.c_proj.bias"], HALF, HALF)
    to_circle(f"fc.{k}", sd[p+"mlp.c_fc.weight"], sd[p+"mlp.c_fc.bias"], LN_HALF, HALF)
    to_circle(f"mproj.{k}", sd[p+"mlp.c_proj.weight"], sd[p+"mlp.c_proj.bias"], HALF, HALF)
    for n in ("ln_1.weight", "ln_1.bias", "ln_2.weight", "ln_2.bias"): out[f"ln.{k}.{n}"] = sd[p+n]
to_circle("head", sd["transformer.wte.weight"].T.copy(), None, HEAD_HALF, None)
out["ln_f.weight"], out["ln_f.bias"] = sd["transformer.ln_f.weight"], sd["transformer.ln_f.bias"]
out["wte.ticks"] = np.round(sd["transformer.wte.weight"] / u_res).astype(np.int64)     # token chords as ticks on the residual dials
out["wpe.ticks"] = np.round(sd["transformer.wpe.weight"] / u_res).astype(np.int64)
np.savez("/Users/abundancemachine/tick-machine/gpt2_dial/circle.npz", **out)
lap_share = stats["laps"] / stats["laps"].sum()
print(f"exported {stats['notes']/1e6:.1f}M notes as uint16 on a {CIRCLE} circle (angle 0..4095 + 16 laps); complements (negatives): {stats['complements']/stats['notes']:.1%}")
print("laps histogram (share of notes with 0,1,2,...,7 laps): " + " ".join(f"{i}:{lap_share[i]:.1%}" for i in range(8)))
print(f"token chords: {out['wte.ticks'].shape[0]:,} x 768 ticks, |ticks| max {np.abs(out['wte.ticks']).max():,} (laps beyond 2047 kept); positions: {out['wpe.ticks'].shape[0]:,} x 768")

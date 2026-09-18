#!/usr/bin/env python3
"""Export the circle model as DENSE 16-bit edges for the optical LM runtime.
An edge is the circle number itself: uint16 on a 65,536 circle = phase (12 bits) + laps (4 bits),
negatives as complements. Layout [nin][nout] so the input index is implicit (no index stored).
Plus per-bank gauges/units/biases, LN laws, token/position chords as ticks."""
import json, numpy as np
C = np.load("/Users/abundancemachine/tick-machine/gpt2_dial/circle.npz"); L, D = 12, 768
o = "/Users/abundancemachine/tick-machine/gpt2_dial/optical/"; man = []
def bank(n):
    c = C[f"{n}.circle"]; nin, nout = c.shape
    c.astype(np.uint16).tofile(o + f"{n}.edges.bin")
    C[f"{n}.gauge"].astype(np.float32).tofile(o + f"{n}.gauge.bin"); C[f"{n}.u_in"].astype(np.float32).tofile(o + f"{n}.u_in.bin")
    C[f"{n}.bias"].astype(np.int64).tofile(o + f"{n}.bias.bin"); (C[f"{n}.u_out"] if f"{n}.u_out" in C else np.zeros(nout)).astype(np.float32).tofile(o + f"{n}.u_out.bin")
    man.append(dict(name=n, nin=nin, nout=nout))
for k in range(L):
    for b in ("qkv", "proj", "fc", "mproj"): bank(f"{b}.{k}")
    for p in ("ln_1.weight", "ln_1.bias", "ln_2.weight", "ln_2.bias"): C[f"ln.{k}.{p}"].astype(np.float32).tofile(o + f"ln.{k}.{p}.bin")
bank("head"); C["ln_f.weight"].astype(np.float32).tofile(o + "ln_f.weight.bin"); C["ln_f.bias"].astype(np.float32).tofile(o + "ln_f.bias.bin")
C["u_res"].astype(np.float32).tofile(o + "u_res.bin"); C["wte.ticks"].astype(np.int32).tofile(o + "wte.ticks.bin"); C["wpe.ticks"].astype(np.int32).tofile(o + "wpe.ticks.bin")
json.dump(man, open(o + "manifest.json", "w")); tot = sum(b["nin"] * b["nout"] for b in man)
print(f"{tot/1e6:.1f}M edges at 16 bits = {tot*2/1e6:.0f} MB (the circle numbers themselves; index implicit)")

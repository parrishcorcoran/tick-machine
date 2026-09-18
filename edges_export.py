#!/usr/bin/env python3
"""THE MODEL AS EDGES. From circle.npz: every note becomes an EDGE with a fire tick. For pile j
(an output) its notes are sorted by fire tick so an integrator can walk them as the lap turns.
  word = i (12 bits) | fire tick (12 bits) | laps (4 bits) | sign (1 bit)
  fire tick = 4096 - angle  (the edge fires with `angle` ticks left in the lap: rate held for
  `angle` ticks = angle copies of the input, no multiply); angle 0 never fires.
  laps: a second list per pile, fired at tick 0 and held the whole turn, `laps` copies.
Writes gpt2_dial/edges/*.bin and a manifest."""
import json, numpy as np
C = np.load("/Users/abundancemachine/tick-machine/gpt2_dial/circle.npz"); CIRCLE, DIAL, L = 65536, 4096, 12
out = "/Users/abundancemachine/tick-machine/gpt2_dial/edges/"; man = {"banks": []}
def bank(name):
    c = C[f"{name}.circle"].astype(np.int64); nin, nout = c.shape
    q = np.where(c >= CIRCLE // 2, c - CIRCLE, c); mag = np.abs(q); angle = mag % DIAL; laps = mag // DIAL; sign = (q < 0).astype(np.int64)
    I, J = np.meshgrid(np.arange(nin), np.arange(nout), indexing="ij")
    fire = (DIAL - angle) % DIAL
    # angle edges: (j, fire) sorted; angle 0 excluded (fires nothing)
    m = angle > 0
    key = J[m] * DIAL + fire[m]; o = np.argsort(key, kind="stable")
    words = (I[m][o] | (fire[m][o] << 12) | (laps[m][o] << 24) | (sign[m][o] << 28)).astype(np.uint32)
    off = np.searchsorted(J[m][o], np.arange(nout + 1)).astype(np.uint32)
    # lap edges: notes with laps > 0, per pile
    ml = laps > 0; key2 = J[ml]; o2 = np.argsort(key2, kind="stable")
    lwords = (I[ml][o2] | (laps[ml][o2] << 24) | (sign[ml][o2] << 28)).astype(np.uint32)
    loff = np.searchsorted(key2[o2], np.arange(nout + 1)).astype(np.uint32)
    for nm, arr in (("notes", words), ("off", off), ("lapnotes", lwords), ("lapoff", loff), ("gauge", C[f"{name}.gauge"].astype(np.float32)),
                    ("u_in", C[f"{name}.u_in"].astype(np.float32)), ("bias", C[f"{name}.bias"].astype(np.int32)),
                    ("u_out", (C[f"{name}.u_out"] if f"{name}.u_out" in C else np.zeros(nout)).astype(np.float32))):
        arr.tofile(out + f"{name}.{nm}.bin")
    man["banks"].append(dict(name=name, nin=nin, nout=nout, edges=int(words.size), lap_edges=int(lwords.size)))
    return words.size, lwords.size
E = LE = 0
for k in range(L):
    for b in (f"qkv.{k}", f"proj.{k}", f"fc.{k}", f"mproj.{k}"): e, le = bank(b); E += e; LE += le
e, le = bank("head"); E += e; LE += le
for k in range(L):
    for n in ("ln_1.weight", "ln_1.bias", "ln_2.weight", "ln_2.bias"): C[f"ln.{k}.{n}"].astype(np.float32).tofile(out + f"ln.{k}.{n}.bin")
C["ln_f.weight"].astype(np.float32).tofile(out + "ln_f.weight.bin"); C["ln_f.bias"].astype(np.float32).tofile(out + "ln_f.bias.bin")
C["u_res"].astype(np.float32).tofile(out + "u_res.bin"); C["wte.ticks"].astype(np.int16).tofile(out + "wte.ticks.bin"); C["wpe.ticks"].astype(np.int16).tofile(out + "wpe.ticks.bin")
man["edges"], man["lap_edges"] = E, LE; json.dump(man, open(out + "manifest.json", "w"))
print(f"edges: {E/1e6:.1f}M angle edges + {LE/1e6:.1f}M lap edges, 4 bytes each, sorted by (pile, fire tick)")

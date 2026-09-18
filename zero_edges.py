#!/usr/bin/env python3
"""EDGES THAT NEVER FIRE. On the edge machine an input value is delivered as an edge;
an input that is exactly 0 ticks fires nothing, and every note wired to it is never
read. How many are there? Dial every EdgeLinear's INPUT on a per-dimension 4096
dial (unit = calibration max / 2047, the same rule as the A->B crossing that
was identical) and count exact zeros over the 3 prompts + their 32 greedy tokens."""
import json, torch, sys
sys.path.insert(0, "/Users/abundancemachine/tick-machine")
import edge_model as em
from transformers import GPT2Tokenizer
tok = GPT2Tokenizer.from_pretrained("gpt2")
calib = tok("The capital of France is", return_tensors="pt").input_ids
m, _ = em.build(65536, 4096, True, 16384, calib)
cases = json.load(open("gpt2_dial/reference.json"))
mods = [(n, mod) for n, mod in m.named_modules() if isinstance(mod, em.EdgeLinear)]
unit = {}; count = {}
def hook(name):
    def f(mod, inp):
        x = inp[0].detach().reshape(-1, inp[0].shape[-1]).abs()
        if name not in unit: unit[name] = x.amax(0).clamp(min=1e-6) / 2047; return   # calibration
        z = (torch.round(x / unit[name]) == 0).float().mean().item()
        count.setdefault(name, []).append(z)
    return f
for n, mod in mods: mod.register_forward_pre_hook(hook(n))
with torch.no_grad():
    m(calib)                                                        # sets units
    for c in cases: m(torch.tensor([c["ids"] + c["ref"]]))
kinds = {"attn.c_attn": "attn (768x2304)", "attn.c_proj": "proj (768x768)", "mlp.c_fc": "fc (768x3072)", "mlp.c_proj": "mproj (3072x768)", "lm_head": "head (768x50257)"}
tot_w = 0; tot_skip = 0
print("share of input ticks that are exactly 0 (their notes never read), per note table:")
for k, label in kinds.items():
    zs = [sum(v)/len(v) for n, v in count.items() if n.endswith(k)]
    if not zs: continue
    mod = next(mod for n, mod in mods if n.endswith(k)); nw = mod.n_weights * len(zs); z = sum(zs)/len(zs)
    tot_w += nw; tot_skip += nw * z
    print(f"  {label:18s} x{len(zs):<3d} zero inputs {z:6.1%}   (per block: {' '.join(f'{v:.0%}' for v in zs)})")
print(f"[MEASURED] notes never read: {tot_skip/1e6:.1f}M of {tot_w/1e6:.1f}M = {tot_skip/tot_w:.1%}  (input dial 4096, unit = calib max/2047)")

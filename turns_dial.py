#!/usr/bin/env python3
"""GPT-2 AS 49 TURNS ON THE CIRCLE. The simple version.
  notes  : integers on a 65,536 circle (12-bit angle + 16 laps), per-column gauge; negatives are
           complements (a computer's signed integer IS the complement pair on a circle)
  inputs : integers = ticks on the input's dial (4096 + laps; the head's input at 16,384)
  a turn : one circular correlation of the two integer waveforms; the outputs are exact integers
           (rounded, so the FFT's float dust is gone) with laps; a gauge turns them real only
           where a bend or water level needs a real number
  outputs: snapped back onto their own dial (4096 + laps per neuron)
    python3 turns_dial.py --case 0 --n 2
"""
import argparse, json, math, sys, time, numpy as np, torch
ap = argparse.ArgumentParser(); ap.add_argument("--case", type=int, default=0); ap.add_argument("--n", type=int, default=2); a = ap.parse_args()
from transformers import GPT2LMHeadModel, GPT2Tokenizer
tok = GPT2Tokenizer.from_pretrained("gpt2"); hf = GPT2LMHeadModel.from_pretrained("gpt2").eval()
sd = {k: v.detach().double().numpy() for k, v in hf.state_dict().items()}
case = json.load(open("/Users/abundancemachine/tick-machine/gpt2_dial/reference.json"))[a.case]
ids, ref = list(case["ids"]), case["ref"][:a.n]
L, D, NH, HD = 12, 768, 12, 64
WHALF, IN_HALF, HEAD_IN_HALF, OUT_HALF = 32767, 2047, 8191, 2047       # notes: 4096x16; inputs: 4096(+laps); head input: 16384; outputs: 4096(+laps)

# calibration (one standard pass): every input dimension's tick and every output neuron's tick
cal_in, cal_out = {}, {}
def hook(name):
    def f(mod, inp, out):
        cal_in[name] = inp[0].detach()[0].abs().amax(0).double().numpy(); cal_out[name] = out.detach()[0].abs().amax(0).double().numpy()
    return f
for k, blk in enumerate(hf.transformer.h):
    blk.attn.c_attn.register_forward_hook(hook(f"qkv.{k}")); blk.attn.c_proj.register_forward_hook(hook(f"proj.{k}"))
    blk.mlp.c_fc.register_forward_hook(hook(f"fc.{k}")); blk.mlp.c_proj.register_forward_hook(hook(f"mproj.{k}"))
hf.lm_head.register_forward_hook(hook("head"))
with torch.no_grad(): hf(tok("The capital of France is", return_tensors="pt").input_ids)

class Turn:
    def __init__(self, W, b, name, in_half, out_half):
        self.nin, self.nout = W.shape; self.N = self.nin * self.nout
        self.u_in = np.maximum(cal_in[name], 1e-6) / in_half                 # the input's tick, per dimension
        Wu = W * self.u_in[:, None]                                            # a note per (input tick, neuron)
        cyc = np.maximum(np.abs(Wu).max(0), 1e-12)                             # each neuron's column: its loudest note is a full turn
        self.q = np.round(Wu / cyc * WHALF).astype(np.int64)                   # integer notes on the 65,536 circle (complements for negatives)
        self.gauge = cyc / WHALF                                               # what one tick of the note is worth, per neuron
        self.spectrum = np.fft.rfft(self.q.T.reshape(-1).astype(np.float64))   # recorded once
        self.b = b; self.offsets = np.arange(self.nout) * self.nin
        self.u_out = (np.maximum(cal_out[name], 1e-6) / out_half) if out_half else None
    def shoot(self, x_real):
        t = np.round(x_real / self.u_in)                                       # the input as ticks (with laps)
        X = np.zeros(self.N); X[:self.nin] = t
        turn = np.fft.irfft(np.conj(np.fft.rfft(X)) * self.spectrum, self.N)   # one turn
        acc = np.round(turn[self.offsets])                                     # exact integers with laps: the pile after the lap
        y = acc * self.gauge + (self.b if self.b is not None else 0)           # gauge -> real, for the laws
        return np.round(y / self.u_out) * self.u_out if self.u_out is not None else y   # onto the output dial

t0 = time.time(); blocks = []
for k in range(L):
    p = f"transformer.h.{k}."
    blocks.append(dict(ln1=(sd[p+"ln_1.weight"], sd[p+"ln_1.bias"]), ln2=(sd[p+"ln_2.weight"], sd[p+"ln_2.bias"]),
        qkv=Turn(sd[p+"attn.c_attn.weight"], sd[p+"attn.c_attn.bias"], f"qkv.{k}", IN_HALF, OUT_HALF), proj=Turn(sd[p+"attn.c_proj.weight"], sd[p+"attn.c_proj.bias"], f"proj.{k}", IN_HALF, OUT_HALF),
        fc=Turn(sd[p+"mlp.c_fc.weight"], sd[p+"mlp.c_fc.bias"], f"fc.{k}", IN_HALF, OUT_HALF), mproj=Turn(sd[p+"mlp.c_proj.weight"], sd[p+"mlp.c_proj.bias"], f"mproj.{k}", IN_HALF, OUT_HALF), K=[], V=[]))
head = Turn(sd["transformer.wte.weight"].T.copy(), None, "head", HEAD_IN_HALF, None)
lnf = (sd["transformer.ln_f.weight"], sd["transformer.ln_f.bias"]); wte, wpe = sd["transformer.wte.weight"], sd["transformer.wpe.weight"]
print(f"49 circles of integer notes recorded in {time.time()-t0:.0f}s", flush=True)

def water(x, w, b): m = x.mean(); v = ((x - m) ** 2).mean(); return (x - m) / math.sqrt(v + 1e-5) * w + b
def gelu(x): return 0.5 * x * (1 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))

def position(pos, tid):
    x = wte[tid] + wpe[pos]
    for B in blocks:
        qkv = B["qkv"].shoot(water(x, *B["ln1"])); q, kk, v = qkv[:D], qkv[D:2*D], qkv[2*D:]
        B["K"].append(kk); B["V"].append(v); Kt = np.stack(B["K"]); Vt = np.stack(B["V"]); out = np.zeros(D)
        for h in range(NH):
            sl = slice(h*HD, (h+1)*HD); sc = (Kt[:, sl] * q[sl]).sum(1) / math.sqrt(HD); al = np.exp(sc - sc.max()); al /= al.sum()
            out[sl] = (al[:, None] * Vt[:, sl]).sum(0)
        x = x + B["proj"].shoot(out)
        x = x + B["mproj"].shoot(gelu(B["fc"].shoot(water(x, *B["ln2"]))))
    return int(np.argmax(head.shoot(water(x, *lnf))))

gen = []; times = []
for i, t in enumerate(ids):
    t0 = time.time(); nxt = position(i, t); times.append(time.time() - t0)
pos = len(ids)
while len(gen) < a.n:
    gen.append(nxt); t0 = time.time(); nxt = position(pos, nxt); times.append(time.time() - t0); pos += 1
    print(f"  token {len(gen)}: {tok.decode([gen[-1]])!r}  ({times[-1]:.1f}s)", flush=True)
print(f"\n{case['prompt']!r} + {a.n}: {'IDENTICAL to fp32 greedy' if gen == ref else 'differs'}   {tok.decode(gen)!r}   (reference {tok.decode(ref)!r})")
print(f"  [MEASURED] integer notes 4096x16 laps, integer input ticks 4096(+laps), head input 16384, outputs on 4096(+laps) dials; {np.mean(times):.1f}s per position")

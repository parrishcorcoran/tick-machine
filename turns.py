#!/usr/bin/env python3
"""GPT-2 AS 49 TURNS. Every linear layer is ONE circular convolution: its rows of notes laid
end to end around one circle, one slot per weight; the input on the same circle from slot 0;
one correlation; each neuron read at its row's offset. The notes' spectrum is recorded ONCE
(record the pattern); per token each layer is FFT(input) x spectrum -> inverse FFT (shoot it).
No matmul in this file. The bends, water levels and the limiter over the past are per-neuron laws.
    python3 turns.py --case 0 --n 4
"""
import argparse, json, math, sys, time, numpy as np, torch
ap = argparse.ArgumentParser(); ap.add_argument("--case", type=int, default=0); ap.add_argument("--n", type=int, default=4); a = ap.parse_args()
from transformers import GPT2LMHeadModel, GPT2Tokenizer
tok = GPT2Tokenizer.from_pretrained("gpt2"); hf = GPT2LMHeadModel.from_pretrained("gpt2").eval()
sd = {k: v.detach().double().numpy() for k, v in hf.state_dict().items()}
case = json.load(open("/Users/abundancemachine/tick-machine/gpt2_dial/reference.json"))[a.case]
ids, ref = list(case["ids"]), case["ref"][:a.n]
L, D, NH, HD = 12, 768, 12, 64

class Turn:
    """One linear layer as one circle. Recorded once; shot per token."""
    def __init__(self, W, b):
        self.nin, self.nout = W.shape; self.N = self.nin * self.nout
        self.spectrum = np.fft.rfft(W.T.reshape(-1))            # the notes, end to end, recorded as a pattern
        self.b = b; self.offsets = np.arange(self.nout) * self.nin
    def shoot(self, x):
        X = np.zeros(self.N); X[:self.nin] = x                    # the input on the circle
        turn = np.fft.irfft(np.conj(np.fft.rfft(X)) * self.spectrum, self.N)   # one turn: every lag
        out = turn[self.offsets]                                  # read each neuron at its row's start
        return out + self.b if self.b is not None else out

t0 = time.time(); blocks = []
for k in range(L):
    p = f"transformer.h.{k}."
    blocks.append(dict(ln1=(sd[p+"ln_1.weight"], sd[p+"ln_1.bias"]), ln2=(sd[p+"ln_2.weight"], sd[p+"ln_2.bias"]),
        qkv=Turn(sd[p+"attn.c_attn.weight"], sd[p+"attn.c_attn.bias"]), proj=Turn(sd[p+"attn.c_proj.weight"], sd[p+"attn.c_proj.bias"]),
        fc=Turn(sd[p+"mlp.c_fc.weight"], sd[p+"mlp.c_fc.bias"]), mproj=Turn(sd[p+"mlp.c_proj.weight"], sd[p+"mlp.c_proj.bias"]), K=[], V=[]))
head = Turn(sd["transformer.wte.weight"].T.copy(), None)
lnf = (sd["transformer.ln_f.weight"], sd["transformer.ln_f.bias"]); wte, wpe = sd["transformer.wte.weight"], sd["transformer.wpe.weight"]
print(f"49 circles recorded in {time.time()-t0:.0f}s: {sum(t.N for B in blocks for t in (B['qkv'],B['proj'],B['fc'],B['mproj']))/1e6:.0f}M slots in the blocks + {head.N/1e6:.1f}M in the head", flush=True)

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
    print(f"  token {len(gen)}: {tok.decode([gen[-1]])!r}  ({times[-1]:.1f}s, 49 turns)", flush=True)
print(f"\n{case['prompt']!r} + {a.n}: {'IDENTICAL to fp32 greedy' if gen == ref else 'differs'}   {tok.decode(gen)!r}")
print(f"  [MEASURED] {np.mean(times):.1f}s per position in numpy float64; every layer one circular convolution, spectra recorded once")

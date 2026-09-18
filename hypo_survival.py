#!/usr/bin/env python3
"""HYPOTHESIS SURVIVAL. K candidate next tokens ride through GPT-2 as ONE vector
at the new position. Do their rings survive to the head?

  context = prompt + greedy tokens so far.  K hypotheses = the top-K next tokens
  at that context (what the tree would pose).  clean: run each hypothesis on its
  own, take its argmax -> n_i.  Then two superposed forms, one pass each:
    chord : new position = mean_i emb(c_i)           (no keys; read the chord's rings)
    keyed : new position = mean_i key_i (*) (emb(c_i)+pos) - pos, circular-conv keys with
            unit-magnitude random phases; unbind the block-11 residual with key_i^-1,
            then ln_f + head -> rings for hypothesis i
  Survival = does n_i show up: chord top-1 == n_1 (the greedy path), n_i in chord top-K;
  keyed argmax_i == n_i, n_i in keyed top-5.  fp32 HF GPT-2 (the dial model is identical to it).
"""
import json, sys, numpy as np, torch
from transformers import GPT2LMHeadModel
torch.manual_seed(0); np.random.seed(0)
m = GPT2LMHeadModel.from_pretrained("gpt2").eval(); tr = m.transformer
cases = json.load(open("/Users/abundancemachine/tick-machine/gpt2_dial/reference.json"))
cap = {}
tr.h[-1].register_forward_hook(lambda mod, i, o: cap.__setitem__("res", o[0]))

def key(n):
    ph = np.exp(1j * np.random.uniform(0, 2*np.pi, n//2 + 1)); ph[0] = 1; ph[-1] = 1
    return torch.tensor(np.fft.irfft(ph, n), dtype=torch.float32)
def bind(k, x):   return torch.fft.irfft(torch.fft.rfft(k) * torch.fft.rfft(x), x.shape[-1])
def unbind(k, x): return torch.fft.irfft(torch.fft.rfft(x) / torch.fft.rfft(k), x.shape[-1])

@torch.no_grad()
def run_embeds(past_ids, new_vec):
    e = torch.cat([tr.wte(torch.tensor(past_ids)), new_vec[None]], 0)[None]
    logits = m(inputs_embeds=e).logits[0, -1]
    return logits, cap["res"][0, -1].clone()

Ks = [2, 4, 8, 16, 32]
stats = {K: dict(n=0, chord_top1=0, chord_topK=0, keyed_arg=0, keyed_top5=0, ctx=0) for K in Ks}
with torch.no_grad():
    for c in cases:
        for j in range(0, 32, 4):
            ctx = c["ids"] + c["ref"][:j]
            base = m(torch.tensor([ctx])).logits[0, -1]
            pos = tr.wpe(torch.tensor(len(ctx)))
            for K in Ks:
                cand = base.topk(K).indices.tolist()
                # clean: each hypothesis alone (a batch of K rows, the exact answer)
                rows = torch.tensor([ctx + [t] for t in cand])
                clean = m(rows).logits[:, -1].argmax(-1).tolist()
                # chord
                emb = tr.wte(torch.tensor(cand))
                lg, _ = run_embeds(ctx, emb.mean(0))
                topK = lg.topk(K).indices.tolist()
                st = stats[K]; st["ctx"] += 1; st["n"] += K
                st["chord_top1"] += int(lg.argmax().item() == clean[0])
                st["chord_topK"] += sum(int(n in topK) for n in clean)
                # keyed
                keys = [key(768) for _ in range(K)]
                sup = torch.stack([bind(k, e + pos) for k, e in zip(keys, emb)]).mean(0) - pos
                _, res = run_embeds(ctx, sup)
                for i in range(K):
                    r = unbind(keys[i], res) * 1.0
                    lgi = m.lm_head(tr.ln_f(r))
                    st["keyed_arg"] += int(lgi.argmax().item() == clean[i])
                    st["keyed_top5"] += int(clean[i] in lgi.topk(5).indices.tolist())
print("hypothesis survival through 12 blocks, 3 prompts x 8 contexts, fp32 GPT-2")
print(f"{'K':>3} {'contexts':>8} | chord: greedy-path top-1 {'':>4} n_i in top-K {'':>4} | keyed: argmax==n_i {'':>2} n_i in top-5")
for K in Ks:
    s = stats[K]
    print(f"{K:>3} {s['ctx']:>8} | {s['chord_top1']:>3}/{s['ctx']:<3} = {s['chord_top1']/s['ctx']:5.0%}   {s['chord_topK']:>4}/{s['n']:<4} = {s['chord_topK']/s['n']:5.0%}   | {s['keyed_arg']:>4}/{s['n']:<4} = {s['keyed_arg']/s['n']:5.0%}   {s['keyed_top5']:>4}/{s['n']:<4} = {s['keyed_top5']/s['n']:5.0%}")

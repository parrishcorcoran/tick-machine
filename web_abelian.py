#!/usr/bin/env python3
"""IS THE WEB ABELIAN? Drop each grain of the last position's input down the web ALONE
(prompt context fixed), record where it lands at the bottom (the rings). If the web is
abelian, the rings for the whole input are the SUM of the rings of the grains:
    rings(x) == rings(0) + sum_i [ rings(x_i e_i) - rings(0) ]
That sum IS the one table, top to bottom. Compare its argmax to the real one."""
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
tok = GPT2Tokenizer.from_pretrained("gpt2"); m = GPT2LMHeadModel.from_pretrained("gpt2").eval(); tr = m.transformer
prompts = ["The capital of France is", "Once upon a time", "In 1492, Columbus", "The quick brown fox jumps over the", "My favorite color is"]
@torch.no_grad()
def rings(ctx_emb, last):                                  # last: (B, 768) inputs for the last position
    B = last.shape[0]; e = torch.cat([ctx_emb.expand(B, -1, -1), last[:, None]], 1)
    return m(inputs_embeds=e).logits[:, -1]
print("web abelian test: each grain of the last position dropped alone, rings summed, vs the real rings")
for p in prompts:
    ids = tok(p, return_tensors="pt").input_ids[0]; S = len(ids)
    with torch.no_grad():
        ctx = tr.wte(ids[:-1])[None]; x = tr.wte(ids[-1:])[0]          # the last token's grains (768 of them)
        true = rings(ctx, x[None])[0]
        zero = rings(ctx, torch.zeros(1, 768))[0]
        grains = torch.diag(x)                                          # grain i alone, at its real size
        alone = rings(ctx, grains)                                      # (768, V)
        summed = zero + (alone - zero).sum(0)
    t1, s1 = int(true.argmax()), int(summed.argmax())
    top5 = len(set(true.topk(5).indices.tolist()) & set(summed.topk(5).indices.tolist()))
    cos = torch.nn.functional.cosine_similarity(true - true.mean(), summed - summed.mean(), dim=0).item()
    print(f"  {p!r:38s} real {tok.decode([t1])!r:10s} summed-grains {tok.decode([s1])!r:10s} {'MATCH' if t1==s1 else 'DIFFERS'}   top5 overlap {top5}/5   ring-shape cos {cos:.3f}")

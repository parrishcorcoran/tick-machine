"""ONE EDGE = A ROW OF g ANGLES. Each layer's dial holds 4096 rows of g weights
(a rotating codebook, fitted by k-means); every group of g weights is sent as
one edge selecting the closest row. Bits per weight: 12/g. What does GPT-2 lose?"""
import sys, math, torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
torch.manual_seed(0); torch.set_num_threads(8)
g = int(sys.argv[1]) if len(sys.argv) > 1 else 4
K = 4096

def fit_book(W, g, iters=12):
    cyc = W.abs().amax(0).clamp(min=1e-12)                  # per-column scale, as on the dial
    X = (W / cyc).T.reshape(-1, g)                          # rows of g along each column
    idx = torch.randperm(X.shape[0])[:K]; C = X[idx].clone()
    for _ in range(iters):
        a = torch.cdist(X, C).argmin(1)
        for k in range(K):
            m = a == k
            if m.any(): C[k] = X[m].mean(0)
    a = torch.cdist(X, C).argmin(1)
    Wq = C[a].reshape(W.shape[1], W.shape[0]).T * cyc
    return Wq, float((Wq - W).norm() / W.norm())

tok = GPT2Tokenizer.from_pretrained("gpt2")
ref = GPT2LMHeadModel.from_pretrained("gpt2").eval()
m = GPT2LMHeadModel.from_pretrained("gpt2").eval()
errs = []
with torch.no_grad():
    for blk in m.transformer.h:
        for mod in (blk.attn.c_attn, blk.attn.c_proj, blk.mlp.c_fc, blk.mlp.c_proj):
            Wq, e = fit_book(mod.weight.detach(), g); mod.weight.data = Wq; errs.append(e)
text = open("/Users/abundancemachine/AnalogLLM/results/data/shakespeare.txt").read()[:3000]
ids = tok(text, return_tensors="pt").input_ids[:, :512]
with torch.no_grad():
    lr = ref(ids, labels=ids).loss.item(); lm = m(ids, labels=ids).loss.item()
    agree = (ref(ids).logits.argmax(-1) == m(ids).logits.argmax(-1)).float().mean().item()
    same = 0
    for p in ["The capital of France is", "Once upon a time", "In 1492, Columbus"]:
        i = tok(p, return_tensors="pt").input_ids
        same += torch.equal(ref.generate(i, max_new_tokens=12, do_sample=False, pad_token_id=50256)[0],
                            m.generate(i, max_new_tokens=12, do_sample=False, pad_token_id=50256)[0])
print(f"[MEASURED] one edge = {g} angles ({12/g:.1f} bits/weight): per-layer weight error {sum(errs)/len(errs):.3f} (rel), "
      f"perplexity x{math.exp(lm-lr):.3f}, top-1 agreement {agree:.3f}, identical prompts {same}/3")

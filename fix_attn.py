"""Tick attention with the causal mask HF applies inside eager attention."""
import torch
import transformers.models.gpt2.modeling_gpt2 as g
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import tick_machine as tm
torch.set_num_threads(4)
tok = GPT2Tokenizer.from_pretrained("gpt2")
ids = tok("The capital of France is", return_tensors="pt").input_ids
ref = GPT2LMHeadModel.from_pretrained("gpt2").eval()
with torch.no_grad():
    rt = ref.generate(ids, max_new_tokens=14, do_sample=False, pad_token_id=50256)
print("fp32            :", repr(tok.decode(rt[0][ids.shape[1]:])))

def tick_attn_masked(module, q, k, v, attention_mask, scaling=None, dropout=0.0, **kw):
    if scaling is None:
        scaling = q.size(-1) ** -0.5
    sc = torch.matmul(q, k.transpose(-1, -2)) * scaling
    # the causal mask HF's eager path applies from module.bias
    Lq, Lk = q.shape[-2], k.shape[-2]
    causal = module.bias[:, :, Lk - Lq:Lk, :Lk]
    sc = torch.where(causal, sc, torch.finfo(sc.dtype).min)
    if attention_mask is not None:
        sc = sc + attention_mask[:, :, :, :Lk]
    best = sc.amax(-1, keepdim=True)
    st = DIAL / 64
    lag = torch.round((best - sc) * 1.4426950408889634 * st) / st
    w = torch.exp2(-lag)
    w = w.masked_fill(~torch.isfinite(w), 0.0)
    w = w / (w.sum(-1, keepdim=True) + 1e-9)
    return torch.matmul(w.type(v.dtype), v).transpose(1, 2), w

for DIAL in (4096, 65536):
    g.eager_attention_forward = tick_attn_masked
    m = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()
    for blk in m.transformer.h:
        for path in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"):
            par, leaf = path.split(".")
            mod = getattr(getattr(blk, par), leaf)
            setattr(getattr(blk, par), leaf, tm.TickLinear(mod.weight.detach(), mod.bias.detach(), DIAL))
    m.lm_head = tm.TickLinear(m.lm_head.weight.detach().T.contiguous(), None, DIAL)
    with torch.no_grad():
        o = m.generate(ids, max_new_tokens=14, do_sample=False, pad_token_id=50256)
    first = next((i for i in range(14) if o[0][ids.shape[1]+i] != rt[0][ids.shape[1]+i]), None)
    print(f"tick both {DIAL:5d} :", repr(tok.decode(o[0][ids.shape[1]:])), "| first differing:", first,
          "  <-- IDENTICAL" if first is None else "")

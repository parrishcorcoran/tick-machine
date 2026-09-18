"""Which part diverges: the tick linears, or the tick attention?"""
import torch
import transformers.models.gpt2.modeling_gpt2 as g
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import tick_machine as tm
torch.set_num_threads(4)
tok = GPT2Tokenizer.from_pretrained("gpt2")
ids = tok("The capital of France is", return_tensors="pt").input_ids
orig_attn = g.eager_attention_forward
ref = GPT2LMHeadModel.from_pretrained("gpt2").eval()
with torch.no_grad():
    rt = ref.generate(ids, max_new_tokens=14, do_sample=False, pad_token_id=50256)
print("fp32          :", repr(tok.decode(rt[0][ids.shape[1]:])))

def run(name, linears, attn, dial=4096):
    g.eager_attention_forward = orig_attn
    m = tm.build_model(dial) if attn else None
    if not attn:
        g.eager_attention_forward = orig_attn
        m = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()
        if linears:
            for blk in m.transformer.h:
                for path in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"):
                    par, leaf = path.split(".")
                    mod = getattr(getattr(blk, par), leaf)
                    setattr(getattr(blk, par), leaf, tm.TickLinear(mod.weight.detach(), mod.bias.detach(), dial))
            m.lm_head = tm.TickLinear(m.lm_head.weight.detach().T.contiguous(), None, dial)
    else:
        if not linears:   # tick attention only: rebuild with real linears
            g.eager_attention_forward = None
            m = tm.build_model(dial)              # sets tick attn, tick linears
            fresh = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()
            for blk, fb in zip(m.transformer.h, fresh.transformer.h):
                blk.attn.c_attn, blk.attn.c_proj = fb.attn.c_attn, fb.attn.c_proj
                blk.mlp.c_fc, blk.mlp.c_proj = fb.mlp.c_fc, fb.mlp.c_proj
            m.lm_head = fresh.lm_head
    with torch.no_grad():
        o = m.generate(ids, max_new_tokens=14, do_sample=False, pad_token_id=50256)
    first = next((i for i in range(14) if o[0][ids.shape[1]+i] != rt[0][ids.shape[1]+i]), None)
    print(f"{name:14s}:", repr(tok.decode(o[0][ids.shape[1]:])), "| first differing token:", first)

run("tick linears", True, False)
run("tick attention", False, True)
run("both (dial 65536)", True, True, dial=65536)

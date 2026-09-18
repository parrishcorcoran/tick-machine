import json, torch
from transformers import GPT2LMHeadModel
m = GPT2LMHeadModel.from_pretrained("gpt2").eval()
cases = json.load(open("/Users/abundancemachine/tick-machine/gpt2_dial/reference.json"))
print("control: how often n_i is in the CONTEXT's own top-K (no new position at all) and in the top-1 hypothesis's clean top-K")
with torch.no_grad():
  for K in [2,4,8,16,32]:
    inbase=inh1=n=0; distinct=0
    for c in cases:
      for j in range(0,32,4):
        ctx=c["ids"]+c["ref"][:j]; base=m(torch.tensor([ctx])).logits[0,-1]; cand=base.topk(K).indices.tolist()
        lg=m(torch.tensor([ctx+[t] for t in cand])).logits[:,-1]; clean=lg.argmax(-1).tolist()
        bt=base.topk(K).indices.tolist(); h1=lg[0].topk(K).indices.tolist()
        inbase+=sum(int(x in bt) for x in clean); inh1+=sum(int(x in h1) for x in clean); n+=K; distinct+=len(set(clean))
    print(f"K={K:>2}: n_i in context top-K {inbase/n:4.0%}   n_i in hypothesis-1's clean top-K {inh1/n:4.0%}   distinct n_i per context {distinct/24:.1f} of {K}")

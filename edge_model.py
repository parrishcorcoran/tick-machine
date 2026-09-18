#!/usr/bin/env python3
"""THE WHOLE MODEL ON THE DIAL.

Every weight is an angle at DIAL resolution -- linear layers, token embeddings,
position embeddings, the head (tied). Every neuron output is snapped to the
dial too: an angle, plus a lap counter for values past one turn. One turn per
layer per token; one edge per weight.

The arithmetic of a turn is exactly the matmul (edge_transfer.py, step 3), so
each turn is computed as the matmul of the dial-valued weights. What is
measured here is what the model LOSES on the dial, and where:

    --lap on/off       is the lap counter necessary?
    --act 4096/0       neuron outputs on the dial, or left continuous
    --head 4096/16384  the head's resolution

Checks: greedy generation identical to fp32 on several prompts; top-1 agreement
per token; perplexity ratio on a text sample.
"""
import argparse
import math

import torch
import torch.nn as nn
import transformers.models.gpt2.modeling_gpt2 as g
from transformers import GPT2LMHeadModel, GPT2Tokenizer

torch.set_num_threads(4)


def to_dial(x, unit, dial, lap):
    """Snap to the dial. unit = value of one tick. lap on: any number of turns.
    lap off: one turn only, values past it are clipped."""
    n = torch.round(x / unit)
    if not lap:
        half = dial // 2 - 1
        n = n.clamp(-half, half)
    return n * unit


class EdgeLinear(nn.Module):
    """Weights as angles (per-output scale), output snapped to the dial."""

    def __init__(self, W, b, dial, act_dial, lap, calib_scale=None):
        super().__init__()
        half = dial // 2 - 1
        cyc = W.abs().amax(0).clamp(min=1e-12)
        q = torch.round(W / cyc * half)
        self.register_buffer("Wd", (q / half) * cyc)             # what the dial shows, per weight
        self.register_buffer("b", b)
        self.n_weights = W.numel()
        self.act_dial, self.lap = act_dial, lap
        self.unit = None                                          # set from calibration
        self.calib = []

    def forward(self, x):
        y = x @ self.Wd
        if self.b is not None:
            y = y + self.b
        if self.act_dial:
            if self.unit is None:                                 # calibration pass
                # per NEURON: each output dimension gets its own dial, like each
                # weight column got its own scale
                self.calib.append(y.detach().abs().reshape(-1, y.shape[-1]).amax(0))
                return y
            y = to_dial(y, self.unit, self.act_dial, self.lap)
        return y


def dial_embedding(emb, dial):
    half = dial // 2 - 1
    W = emb.weight.detach()
    cyc = W.abs().amax(0).clamp(min=1e-12)                       # per dimension
    emb.weight.data = (torch.round(W / cyc * half) / half) * cyc


def build(dial, act_dial, lap, head_dial, calib_ids):
    def lag_attn(module, q, k, v, attention_mask, scaling=None, dropout=0.0, **kw):
        if scaling is None:
            scaling = q.size(-1) ** -0.5
        sc = torch.matmul(q, k.transpose(-1, -2)) * scaling
        Lq, Lk = q.shape[-2], k.shape[-2]
        causal = module.bias[:, :, Lk - Lq:Lk, :Lk]
        sc = torch.where(causal, sc, torch.finfo(sc.dtype).min)
        if attention_mask is not None:
            sc = sc + attention_mask[:, :, :, :Lk]
        best = sc.amax(-1, keepdim=True)
        st = dial / 64
        lag = torch.round((best - sc) * 1.4426950408889634 * st) / st
        w = torch.exp2(-lag)
        w = w.masked_fill(~torch.isfinite(w), 0.0)
        w = w / (w.sum(-1, keepdim=True) + 1e-9)
        return torch.matmul(w.type(v.dtype), v).transpose(1, 2), w

    g.eager_attention_forward = lag_attn
    m = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()
    layers = []
    for blk in m.transformer.h:
        for path in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"):
            par, leaf = path.split(".")
            mod = getattr(getattr(blk, par), leaf)
            el = EdgeLinear(mod.weight.detach(), mod.bias.detach(), dial, act_dial, lap)
            setattr(getattr(blk, par), leaf, el)
            layers.append(el)
    dial_embedding(m.transformer.wte, dial)                      # embeddings on the dial (tied head follows)
    dial_embedding(m.transformer.wpe, dial)
    head = EdgeLinear(m.transformer.wte.weight.detach().T.contiguous(), None, dial, head_dial, lap)
    m.lm_head = head
    layers.append(head)
    # calibration: one fp-ish pass sets each layer's tick unit from its largest output
    with torch.no_grad():
        m(calib_ids)
    for el in layers:
        if el.act_dial:
            el.unit = torch.stack(el.calib).amax(0).clamp(min=1e-6) / (el.act_dial // 2 - 1)
    return m, sum(el.n_weights for el in layers)


PROMPTS = ["The capital of France is", "Once upon a time", "In 1492, Columbus",
           "The theory of relativity states that", "My favorite food is"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dial", type=int, default=4096)
    ap.add_argument("--act", type=int, default=4096, help="neuron output resolution, 0 = continuous")
    ap.add_argument("--head", type=int, default=16384)
    ap.add_argument("--lap", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=12)
    a = ap.parse_args()
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    ref = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    text = open("/Users/abundancemachine/AnalogLLM/results/data/shakespeare.txt").read()[:3000]
    tids = tok(text, return_tensors="pt").input_ids[:, :512]
    m, n_w = build(a.dial, a.act, bool(a.lap), a.head, tids[:, :128])

    print(f"dial {a.dial}, neuron outputs {a.act or 'continuous'}, head {a.head}, lap {'on' if a.lap else 'off'}")
    print(f"  weights on the dial: {n_w:,}  -> edges per token per layer-turn: one each")
    same = 0
    with torch.no_grad():
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt").input_ids
            r = ref.generate(ids, max_new_tokens=a.tokens, do_sample=False, pad_token_id=50256)
            o = m.generate(ids, max_new_tokens=a.tokens, do_sample=False, pad_token_id=50256)
            eq = torch.equal(r[0], o[0])
            same += eq
            print(f"  {'SAME' if eq else 'diff'}  {p!r:40s} {tok.decode(o[0][ids.shape[1]:])!r}")
        lr = ref(tids, labels=tids).loss.item()
        lm = m(tids, labels=tids).loss.item()
        agree = (ref(tids).logits.argmax(-1) == m(tids).logits.argmax(-1)).float().mean().item()
    print(f"  generation identical on {same}/{len(PROMPTS)} prompts")
    print(f"  [MEASURED] perplexity {math.exp(lm):.3f} vs fp32 {math.exp(lr):.3f}  (x{math.exp(lm-lr):.4f})   "
          f"next-token top-1 agreement {agree:.4f}")


if __name__ == "__main__":
    main()

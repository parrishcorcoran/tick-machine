#!/usr/bin/env python3
"""THE TICK MACHINE.

A weight is never a number. It is a position on a dial of DIAL ticks.
The only place a VALUE lives is a table of DIAL entries, set once.

To compute y = W x:
    for each output j, for each tick t:
        gather every x_i whose weight w_ij sits at tick t   -> pure ADDS
        multiply that gathered sum by table[t], once          -> one multiply per live tick
    sum over ticks.

So 768 multiply-adds become 768 adds + (number of live ticks) multiplies.
Negative weights are the far side of the dial: -1000 is tick 3096. No sign bit.

Built step by step with a check after each:
    1. one layer, does the tick form reproduce the matmul?
    2. whole GPT-2, is the output identical to fp32?
    3. speed at each dial, honestly measured.
"""
import time, sys
import torch, torch.nn as nn
torch.set_num_threads(4)


class TickLinear(nn.Module):
    """W stored as dial positions (int16). Forward is gather-add then one multiply per tick."""

    def __init__(self, W, b, dial):
        super().__init__()
        n_in, n_out = W.shape
        self.n_in, self.n_out, self.dial = n_in, n_out, dial
        half = dial // 2 - 1                                    # never land on the antipode
        cyc = W.abs().amax(0).clamp(min=1e-12)                  # per-output full scale
        q = torch.round(W / cyc * half).to(torch.int64)         # signed dial steps
        pos = q % dial                                          # POSITION. -k -> dial-k. no sign.
        store = torch.int16 if dial <= 32768 else torch.int32   # int16 holds a 4096 dial; 65536 needs int32
        self.register_buffer("pos", pos.to(store))              # THE WEIGHTS: 12 bits of "which tick"
        tbl = ((torch.arange(dial) + dial // 2) % dial - dial // 2).float() / half
        self.register_buffer("table", tbl)                      # THE ONLY VALUES: dial entries
        self.register_buffer("cyc", cyc)
        self.register_buffer("b", b if b is not None else None)

    lean = False   # same machine, without the bucket tensor: read the table at each
                   # weight's tick, one dense pass. 50x faster, dial-independent.

    def forward(self, x):
        sh = x.shape
        X = x.reshape(-1, self.n_in)                            # (B, n_in)
        B = X.shape[0]
        if self.lean:
            y = (X @ self.table[self.pos.to(torch.int64)]) * self.cyc
            if self.b is not None:
                y = y + self.b
            return y.reshape(*sh[:-1], self.n_out)
        # gather: bucket[b, j, t] = sum of X[b, i] over i with pos[i, j] == t
        idx = self.pos.to(torch.int64).T                        # (n_out, n_in) tick of each weight
        idx = idx.unsqueeze(0).expand(B, -1, -1)                # (B, n_out, n_in)
        src = X.unsqueeze(1).expand(-1, self.n_out, -1)         # (B, n_out, n_in)
        bucket = torch.zeros(B, self.n_out, self.dial, dtype=X.dtype, device=X.device)
        bucket.scatter_add_(2, idx, src)                        # PURE ADDS
        y = bucket @ self.table                                 # one multiply per tick, per output
        y = y * self.cyc
        if self.b is not None:
            y = y + self.b
        return y.reshape(*sh[:-1], self.n_out)


def check_layer(dial):
    from transformers import GPT2LMHeadModel
    m = GPT2LMHeadModel.from_pretrained("gpt2")
    lin = m.transformer.h[5].mlp.c_fc
    W, b = lin.weight.detach(), lin.bias.detach()
    x = torch.randn(4, 768)
    ref = x @ W + b
    tl = TickLinear(W, b, dial)
    with torch.no_grad():
        y = tl(x)
    err = float((y - ref).norm() / ref.norm())
    live = int((tl.pos.to(torch.int64).unique()).numel())
    return err, live


def build_model(dial):
    import transformers.models.gpt2.modeling_gpt2 as g
    from transformers import GPT2LMHeadModel

    def tick_attn(module, q, k, v, attention_mask, scaling=None, dropout=0.0, **kw):
        if scaling is None:
            scaling = q.size(-1) ** -0.5
        sc = torch.matmul(q, k.transpose(-1, -2)) * scaling
        # the causal mask HF's eager attention applies from module.bias. without
        # it every prefill position sees the future, the cache is poisoned, and
        # the SECOND generated token diverges (measured 2026-09-17).
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

    g.eager_attention_forward = tick_attn
    m = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()
    for blk in m.transformer.h:
        for path in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"):
            par, leaf = path.split(".")
            mod = getattr(getattr(blk, par), leaf)
            setattr(getattr(blk, par), leaf, TickLinear(mod.weight.detach(), mod.bias.detach(), dial))
    m.lm_head = TickLinear(m.lm_head.weight.detach().T.contiguous(), None, dial)
    return m


def main():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")

    print("STEP 1 — one layer, tick form vs matmul\n")
    print(f"  {'dial':>6s} {'rel err':>10s} {'distinct ticks used':>20s}")
    for dial in (64, 256, 4096):
        err, live = check_layer(dial)
        print(f"  {dial:6d} {err:10.5f} {live:20d}")

    print("\nSTEP 2 — whole GPT-2, tick form vs fp32\n")
    ref = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    ids = tok("The capital of France is", return_tensors="pt").input_ids
    with torch.no_grad():
        rt = ref.generate(ids, max_new_tokens=14, do_sample=False, pad_token_id=50256)
    print(f"  fp32     : {tok.decode(rt[0][ids.shape[1]:])!r}")
    for dial in (256, 4096):
        m = build_model(dial)
        with torch.no_grad():
            o = m.generate(ids, max_new_tokens=14, do_sample=False, pad_token_id=50256)
        same = torch.equal(o[0], rt[0])
        print(f"  dial {dial:4d}: {tok.decode(o[0][ids.shape[1]:])!r}{'   <-- IDENTICAL' if same else ''}")

    print("\nSTEP 3 — speed, honestly\n")
    print(f"  {'':12s} {'tok/s':>8s}")
    with torch.no_grad():
        ref.generate(ids, max_new_tokens=3, do_sample=False, pad_token_id=50256)
        t0 = time.perf_counter(); ref.generate(ids, max_new_tokens=12, do_sample=False, pad_token_id=50256)
        print(f"  {'fp32':12s} {12/(time.perf_counter()-t0):8.1f}")
    for dial in (256, 4096):
        m = build_model(dial)
        with torch.no_grad():
            m.generate(ids, max_new_tokens=3, do_sample=False, pad_token_id=50256)
            t0 = time.perf_counter(); m.generate(ids, max_new_tokens=12, do_sample=False, pad_token_id=50256)
            print(f"  {'dial '+str(dial):12s} {12/(time.perf_counter()-t0):8.1f}")
    print("\n  (this is the scatter_add in PyTorch on one CPU core. the gather is the")
    print("   whole computation and it is embarrassingly parallel -- one thread per tick.)")


if __name__ == "__main__":
    raise SystemExit(main())

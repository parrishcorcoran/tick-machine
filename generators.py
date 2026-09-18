"""#4: W as r masks and r bindings: W = sum_r diag(u_r) C_r, C_r circulant (a
convolution = a binding). Fit by alternating least squares on real GPT-2
squares; report the relative error for r = 4, 8, 16. In the arbitrary basis
of the residual stream (their caveat), so a poor fit is weak evidence, a good
one is strong."""
import sys, torch, math
from transformers import GPT2LMHeadModel
torch.set_num_threads(4); torch.manual_seed(0)
d = 768
F = torch.fft.fft(torch.eye(d)) / math.sqrt(d)        # unitary DFT
def fit(W, r, iters=40):
    W = W.to(torch.complex64)
    u = torch.randn(r, d, dtype=torch.complex64) / math.sqrt(r)
    c = torch.randn(r, d, dtype=torch.complex64)
    Fh = F.conj().T
    for _ in range(iters):
        # given c: rows i: W[i,:] = sum_r u[r,i] * (Fh diag(c_r) F)[i,:]
        C = torch.stack([Fh @ (c[k][:, None] * F) for k in range(r)])         # (r, d, d)
        A = C.permute(1, 0, 2)                                                # (d, r, d): per row i, r vectors
        u = torch.linalg.lstsq(A.transpose(1, 2), W[:, :, None]).solution[:, :, 0].T   # (r, d)
        # given u: columns of W Fh: (W Fh)[:, j] = sum_r u[r] * Fh[:, j] * c[r, j]
        WF = W @ Fh
        B = torch.stack([u[k][:, None] * Fh for k in range(r)])              # (r, d, d) -> per j: (d, r)
        Bj = B.permute(2, 1, 0)                                               # (d_j, d_i, r)
        c = torch.linalg.lstsq(Bj, WF.T[:, :, None]).solution[:, :, 0].T      # (r, d)
    C = torch.stack([Fh @ (c[k][:, None] * F) for k in range(r)])
    Wf = sum(u[k][:, None] * C[k] for k in range(r))
    return float((Wf - W).norm() / W.norm())
m = GPT2LMHeadModel.from_pretrained("gpt2")
for name, W in (("h5 attn.c_proj", m.transformer.h[5].attn.c_proj.weight),
                ("h5 mlp.c_fc[:, :768]", m.transformer.h[5].mlp.c_fc.weight[:, :768]),
                ("h5 mlp.c_proj[:768]", m.transformer.h[5].mlp.c_proj.weight[:768]),
                ("random control", torch.randn(768, 768) / 27)):
    W = W.detach().float()
    print(f"  {name:24s}", "  ".join(f"r={r}: rel err {fit(W, r):.3f}" for r in (4, 8, 16)), flush=True)

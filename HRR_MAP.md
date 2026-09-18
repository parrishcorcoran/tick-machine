# Standard LM operations <-> HRR operations, 1:1, from the sources (2026-09-18)

`[cited]` = the source says it. `[identity]` = an algebraic identity, exact, no approximation.
`[not mapped]` = no source maps it.

## A. The HRR alphabet (Plate 1995; FHRR; qFHRR arXiv 2604.25939)

| operation | HRR (real vectors) | FHRR (unit phasors) | qFHRR (integer phase index, K bins) |
|---|---|---|---|
| bind      | x (*) y = F^-1(F(x) . F(y))  circular convolution | add phases | r = (q_a + q_b) mod K      (eq. 3) |
| unbind    | x* (*) z, x*_k = x_{-k mod n} (involution ~ inverse) | subtract phases | q_a = (r - q_b) mod K      (eq. 4) |
| bundle    | x + y (superposition) | sum phasors, project back to unit circle | R_i = sum cos, I_i = sum sin; q = round(K/2pi atan2(I,R)) mod K (eq. 7-8) |
| similarity| x . y | mean cos(phase difference) | 1/D sum cos(2pi/K (q_a - q_b))  (eq. 5) |
| cleanup   | argmax_i (x . codebook_i) | same | same, by table |
| fractional bind | x^p via phases | multiply phases by p | round(p q) mod K (eq. 9) |

qFHRR fidelity to continuous FHRR: K=16 (4 bits) bind 0.987, bundle 0.973. `[cited]`
Our dial is qFHRR with K = 4096 (12 bits) plus a lap counter: bind = add angles mod 4096.

## B. The one exact bridge between a matmul and HRR  `[identity]`

1. Circular convolution with a fixed y IS multiplication by the circulant matrix C_y
   (Hrrformer, footnote 1). So: bind = one circulant matmul.
2. Any dense n x n matrix is a sum of its n wrapped diagonals:
       W = sum_{j=0}^{n-1} D_j P^j,   P = cyclic shift, D_j = diag(W[i, (i+j) mod n])
   and P^j x = x (*) delta_j (binding with a unit impulse at j). Therefore
       W x = sum_j  D_j . ( x (*) delta_j )
   = n impulse-bindings (shifts), each scaled per-dimension, bundled. Exact.
   A dense matmul is n bindings + n diagonal scalings + 1 bundle. No approximation.

## C. Transformer <-> VSA, as the literature states it

| LM operation | HRR operation | source | exact? |
|---|---|---|---|
| token embedding row | codebook item (atomic vector) | Plate's cleanup memory, by construction | identity |
| positional embedding | (not mapped; RoPE noted as "differentiable permutation") | 2512.14709 | not mapped |
| Q = X W_q | role vector | 2512.14709 | approx |
| K = X W_k | stored role (address) | 2512.14709 | approx |
| V = X W_v | filler | 2512.14709 | approx |
| Q K^T / sqrt(d) | similarity (dot) between roles | 2512.14709 eq.(2) | approx |
| softmax | soft unbinding / cleanup (denoises the superposition) | 2512.14709; Hrrformer sec.3 | approx |
| sum_j alpha_ij v_j | bundling of retrieved fillers | 2512.14709 eq.(4) | approx |
| residual add x + attn + mlp | superposition (bundle) of bound structures | 2512.14709 eq.(5) | identity (it is +) |
| multi-head | parallel binding channels | 2512.14709 | approx |
| layer norm | "maintains geometry" ~ FHRR's project-to-unit-circle | 2512.14709; FUNDAMENTALS 12 | approx |
| MLP c_fc, c_proj | **not mapped** ("learned nonlinear re-encodings") | 2512.14709 | not mapped |
| GELU | **not mapped** | all sources | not mapped |
| unembedding argmax(wte . h) | cleanup memory (nearest codebook item) | Plate | identity |

Conditions the mapping needs (2512.14709): near-orthogonal keys, sparse attention, normalisation.

## D. Attention written entirely in HRR (Hrrformer, arXiv 2305.19534)

  beta  = sum_t  k_t (*) v_t              bundle of bindings, one vector for all T pairs   (eq. 1)
  v_hat = q^dagger (*) beta               unbind with the query                            (eq. 2)
  a_t   = cos(v_t, v_hat)                 similarity of each value to what came out        (eq. 3)
  w     = softmax(a)                      cleanup: removes the superposition noise
  out   = [w_1 v_1, ..., w_T v_T]                                                          (eq. 4)
"an alternative (but not mathematically equivalent) form of self-attention" `[cited]`, O(T H log H).

GHRR (arXiv 2405.09689): bind = elementwise product of block-unitary matrices; attention =
[softmax(Re[Q * K^dagger]) * V]; softmax kept; replacing attention in a transformer improved
perplexity ~5% (WikiText-2 29.16 -> 27.5). `[cited]`

## E. What this says, plainly

- Everything linear is HRR exactly: residual = bundle, matmul = n shift-bindings + scalings + bundle,
  embedding/unembedding = codebook/cleanup. On the dial all of it is angle-adds.
- Attention is HRR by structure (bind keys to values, unbind with the query, cleanup), and every
  source calls the correspondence approximate; the one paper that makes it exact-by-construction
  (Hrrformer) changes attention to do so.
- No source maps the MLP or GELU to any HRR operation. The nearest thing in the HRR alphabet to a
  bend is FHRR's projection back to the unit circle (a limiter), which is LayerNorm's cousin, not GELU's.

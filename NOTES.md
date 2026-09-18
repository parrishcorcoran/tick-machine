# Tick machine — measured 2026-09-17, MacBook Air, PyTorch CPU

`[MEASURED]` unless marked.

**Step 1, one layer (GPT-2 h.5 mlp.c_fc, 768 in).** Tick form vs matmul:
dial 64: rel err 0.032 · 256: 0.0081 · 4096: 0.00049.
Distinct ticks used: 63 / 255 / 4094 — every tick is live. So at dial 4096 a
768-input layer does ~4094 multiplies per output, not 768. The multiply saving
exists only when inputs greatly outnumber the dial. The adds are still adds.

**Step 2, whole GPT-2, 14 greedy tokens, "The capital of France is".**
- tick linears only, dial 4096: **identical to fp32**.
- as first written (tick attention without the causal mask): diverges at
  token 2 — prefill positions saw the future and poisoned the cache.
- with the causal mask (now in the script): **identical to fp32 at dial 4096**.
- dial 65536 needs int32 positions; int16 wraps at 32768 (fixed).

**Step 3, speed.** fp32 51.5 tok/s · dial 256 4.3 · dial 4096 1.8 — measured
on a CPU shared with four other training jobs, so pessimistic by ~2–3×. Not yet
measured idle. The cost is PyTorch's `scatter_add_` over (B, n_out, dial).

## The edge machine, whole model — 2026-09-17 `[MEASURED]`

`edge_transfer.py`: one weight, one edge, one bit in 4096 ticks; the receiver
reads the dial when the bit arrives and gets the weight back to dial
resolution. One neuron: 768 edges = 768 weights. One GPT-2 layer, one turn:
2,359,296 edges, one per weight; error vs matmul 3.3% at dial 64 (the 3%),
0.8% at 256, 0.05% at 4096. At a 168 ps tick a 4096 turn is 688 ns; 49 turns
= 34 us/token `[DERIVED]`.

`edge_model.py`: every weight an angle (linear layers, wte, wpe, tied head);
every neuron output an angle on its own dial; attention as lags. Whole GPT-2,
5 prompts x 12 greedy tokens, plus 512 tokens of text for perplexity and
next-token agreement against fp32.

| weights | outputs | lap | identical | ppl ratio | top-1 |
|---|---|---|---|---|---|
| 4096 | continuous | — | 4/5 | 0.9995 | 98.8% |
| 4096 | 4096, one unit per layer | on | 1/5 | 0.94 | 91.4% |
| 4096 | 4096, per neuron | off | 1/5 | 1.10 | 76.8% |
| 4096 | 4096, per neuron | on | 3/5 | 1.0007 | 98.6% |
| 4096 | 16384, per neuron | on | 4/5 | 0.9992 | 98.4% |
| **65536** | **4096, per neuron** | **on** | **5/5** | **0.9995** | **99.8%** |

- The lap counter is necessary, on outputs and on weights.
- Each neuron needs its own dial; one dial per layer loses 7 points.
- The last bit is in the weights: dial + lap (65536) makes generation identical.
- The head wants more than 4096 (90.0% at 4096 vs 91.4% at 16384, same setting).

## Step 1 on a real link — 2026-09-17 `[MEASURED]`

`edge_link.c`: two processes, shared clock (mach_absolute_time, 42 ns
resolution). The sender holds 768 weights as ticks and fires one edge each by
flipping a shared-memory flag at its tick; the receiver timestamps arrivals
and reads the dial. No value is ever transmitted.

| tick | dial | wrong of 768 | dot-product rel err | weights/s per line |
|---|---|---|---|---|
| 200 ns | flat 4096 | 3–4 (≤ 3 ticks off) | 7.7e-6 – 3.3e-5 | 1,221 |
| 200 ns | nested 64×64 | 5–6, one high-digit slip (256–1024 ticks) | 1.5e-3 – 7.7e-3 | 39,062 |
| 100 ns | either | most | — | — |

- The tick floor between two processes on this OS is ~200 ns (shared-memory
  edge jitter ~0.5 us; a pipe jitters 79 us). A real-time thread policy did
  not lower it, and deadlocks with 4+ spinning pairs.
- Nested radix: 32x throughput, but a one-tick slip in the high digit costs
  64 ticks of value. The high digit needs protecting (guard band or redundancy).
- Two lines on two cores scale perfectly (same wall time, twice the weights).
- Cost of carrying bits as time on a CPU: ~1e3–1e5 weights/s per line vs
  ~1e11/s reading 12-bit values from memory. GPT-2 is 124M weights/token.
  The link is exact; it is fast only with picosecond ticks and millions of
  lines in parallel, i.e. hardware with stationary weights.

## Gap coding on the link — 2026-09-17 `[MEASURED]`

`gap_link.c`: the dial is a counter that resets at every edge. A weight is the
gap since the previous edge (|angle|); its sign is the edge's direction (flag
up = +, down = −). No lap is waited for. GPT-2's linear weights have mean
|angle| 420 ticks on a 4096 dial (median 336, 99% under 1517), so the
predicted gain over the flat dial is 4096/421 = 9.7x.

| tick | wrong of 768 | worst slip | dot rel err | weights/s per line |
|---|---|---|---|---|
| 200 ns | 1 | 7 ticks | 4.1e-4 | 12,878 (flat: 1,221 — **10.6x**) |
| 100 ns | 4 | 40 | 2.3e-3 | 25,755 |
| 42 ns | 215 | 32 | 3.5e-3 | 61,322 |

- No radix, so no fragile high digit; a slip is a few ticks and does not
  accumulate (a late edge shortens one gap and lengthens the next).
- Stacks with nesting: gap-code the nested digits (~40 ticks/weight, ~100x).
- The flat-dial control path in gap_link.c has a decode bug (all wrong);
  the flat numbers quoted are from edge_link.c, which was validated.

## The streaming GPT-2 on the dial — 2026-09-17 `[MEASURED]`

`dial_flow.py`: flowgpt.py's loop (all 12 blocks live every tick, each
reading the previous tick's emission, input arriving as pulses over a 128-tick
cycle) with every weight and embedding as an angle (4096 x 16 laps), every
neuron output on its own 4096 dial with a lap counter, attention as lags, the
residual stream held across turns, forks read on every turn.

- "The capital of France is": answer ' the', MATCH; **locked at tick 15 of
  128**, the same tick as the fp32 flow.
- 617,660,160 edges per turn (123.5M weights x 5 positions), one bit each;
  87.1 G edges over 141 turns. No weight moved.
- Resonance curve (answer's fork minus loudest other): negative and swinging
  through tick 9 (-2.9 ... -10.1), crosses zero at tick 10, holds from 15,
  peaks +1.68 at tick 25, relaxes to a steady +0.3 to +0.6 for the rest of the
  cycle. The absolute loudness swings to -178 at tick 2 and settles near -100.

## Pipelined tokens on the dial — 2026-09-17 `[MEASURED]`

`dial_pipe.py`: when the newest position's fork has held 8 turns its token is
struck as the next input on the next turn while everything else keeps flowing.
A position is FINAL T+L turns after it started (the flow then equals the batch
model). With --rollback, a final note that differs from the struck one
replaces it and everything after is re-struck.

"The capital of France is", 8 tokens, T=128:
- no rollback: 22.8 ticks/token (6.2x over 141), 7/8 tokens match; position 10
  struck ',' and later settled to ' Republic' -> the stream diverged.
- rollback: 1 un-strike (2 later notes dropped and re-struck), final output
  MATCHES the batch model; 38.1 ticks/token between commits, 55.8 including
  the final settling. Exact by construction; the early strikes are a head
  start that was right 7 times in 8.

"Once upon a time", 32 tokens, T=128, rollback on: output MATCHES exactly;
11 rollbacks in 32 (the 8-turn hold misses ~1 in 3, all near-ties: ' great'
vs ' the', ',' vs '.'); 68.6 ticks/token all in = 2.1x over one at a time.
Each miss costs a full 141-tick settle. The strike rule is the knob.

## Six ideas for the weight transfer — 2026-09-17 `[MEASURED]`

| # | idea | result |
|---|---|---|
| 1 | the head is the embedding table read the other way; the receiver holds it | edges/token 123.5M -> 84.9M, free, exact |
| 2 | sort each column once (wiring), send steps | mean step 3.74 ticks on real GPT-2 (4.74 ticks/weight). On the link (`sorted_link.c`, miss-tolerant receiver, warm pilots, nominal tick): 200 ns: 0 wrong x3, error 0.0, 438,648 weights/s per line (360x flat, 34x gap coding); 100 ns: 0-4 wrong, 877k/s; 42 ns: ~226 wrong by <=4 ticks, 5e-4, 2.09M/s |
| 4 | r masks + r bindings per matrix (ALS fit) | rel err 0.989/0.978/0.957 at r=4/8/16 vs random 0.990/0.980/0.961 — no structure found in the stored basis |
| 7 | residue dials 61/64/67 with a vote | at 200 ns 3 wrong but dot error 29%: a wrong remainder resolves to a value off by thousands; worse than gap coding |
| 8 | rows as f(pos_i - pos_j) | rel err 0.985 vs random 0.989 — no interval structure in the stored basis |
| 10 | four-symbol edges (+1,-1,+2,-2 = 1,-1,0,-0) | lap flag free at 200 ns (0 wrong); at 100 ns a missed edge fakes a double |

Bugs found on the way, all in the receiver: a missed edge must be detected
from the counter jump (else every later index shifts); the first flag touch
is late (cold line) so pilots must be warm; sender and receiver share one
clock so the tick is nominal, not calibrated from jittery pilots.

## Sorted steps as a door, and a correction — 2026-09-17

`[MEASURED]` sorted-row steps on GPT-2: entropy 3.10 bits/weight; a simple
4-bit code stores 84.9M weights in 49 MB (4.65 bits/weight). A batch-1
decode+accumulate pass over them (`steps_gemv.c`, 10 threads): 12.8 ms/token
= 78 tok/s, compute-bound in the nibble decoder, not memory-bound.

**Correction.** I projected ~3,000 tok/s through RAM from 3.1 bits/weight.
That forgot the wiring: sorting moves ~9 bits/weight of "which input" out of
the value and into the permutation, and on a CPU the permutation is data the
accumulator must read every token (219 MB/token in the gather version). Steps
+ permutation ~ 12 bits: the same door. The 3.1 bits/weight is real only
where the order is physical and set once (pins, taps, the agreed order on the
two-core link). The SSD delivers into a CPU, so it does not get it either.

Dial 64 + lap-in-symbol on the link: exact at 100 ns (0/2/1 wrong), 1.01M
weights/s per line; a missed edge now costs the rest of the row (laps are
cumulative) — recoverable from the counter's absolute value, not yet done.

## Metal kernels — 2026-09-17 `[MEASURED]` (Apple M4 GPU, 110,592 rows x 768, batch 1)

`metal_ticks.swift` (MSL compiled at runtime):

| kernel | bytes/pass | GB/s | tok/s ceiling | note |
|---|---|---|---|---|
| ticks16 (4096 dial + laps, int16) | 170 MB | 83.6 | **492** | identical model; max diff 2.5e-7 |
| ticks12 (12-bit packed, 2 per 3 bytes) | 127 MB | 80.3 | **631** | 4096 dial, no laps |
| steps4 (sorted 4-bit steps + order) | 219 MB | 34.4 | 157 | decode-bound; small mismatch (7.7e-3) unresolved |

The GPU reads packed ticks at the memory wall; no matrix is ever built. The
single-stream ceiling at full resolution on this laptop is set by
bytes/weight x 85M / 100 GB/s, and 12-bit packed is the best of the exact
encodings a CPU/GPU can read, because the sorted order costs more to read
than the steps save.

## GPT-2 on the dial, end to end on the Apple GPU — 2026-09-17 `[MEASURED]`

`gpt2_metal.swift` + `gpt2_dial/` (exported by the script in this session):
every linear layer and the head as int16 ticks (4096 x 16 laps, per-output
scale), embeddings on the dial, LayerNorm / GELU / attention with KV cache /
head as Metal kernels, greedy, batch 1, one command buffer per token.

- 3 prompts x 32 tokens: **identical to fp32 HF greedy, every token.**
- **160.9 tok/s** single stream on the M4 GPU, 247 MB of ticks per token
  (39.7 GB/s of weights). The pure weight-pass ceiling for this encoding
  measured earlier is 492 tok/s; the gap is dispatch overhead (~100 dispatches
  per token), a one-thread-per-head attention kernel and a one-threadgroup
  LayerNorm, all fixable.
- One real bug on the way: Metal's fast tanh returns NaN for |x| > ~44
  (inf/inf); GELU needs the argument clamped. Everything else matched numpy
  to 1e-6 stage by stage.

## Is anything on the M4 faster than RAM for weights? — 2026-09-17 `[MEASURED]`

`resident.swift`: int16-tick GEMV over 1–247 MB of weights, 40 passes back to
back. 4 MB: 100 GB/s; 8 MB: 105; 16–247 MB: 80–86. No size runs faster than
RAM's ~100 GB/s; small sizes are bounded by ~40 us dispatch latency. There is
no on-chip store on this chip that a weight-stationary layout could use. The
Neural Engine shares the same memory and accepts only CoreML int8/fp16.
Single-stream at full resolution on this laptop is therefore bounded by
bytes/token over ~100 GB/s: ~490 tok/s at 16-bit ticks, ~540 at 12-bit packed;
161 tok/s measured end to end with unfused kernels.

## Two machine-side measurements — 2026-09-17 `[MEASURED]`

`onchip.swift`: a slice of weights re-read many times INSIDE one dispatch
(no dispatch floor). 2 MB: 107 GB/s; 4 MB: 130; 8 MB: 124; 16 MB: 84;
32-64 MB: 70-74. On-chip memory on the M4 GPU gives at most ~1.3x RAM for
slices of 8 MB or less, and nothing above that. Corrects the earlier test
(which was dispatch-floor-limited) without changing the conclusion: there is
no on-chip store here that a stationary weight layout could use.

Entropy of the dial angles themselves (per-column scale, 4096 dial, no
sorting): 11.09 bits per weight. The notes carry 11 of the 12 stored bits;
a perfect lossless code would lift the RAM ceiling from 785 to 850 tok/s (8%).
The dial distribution is nearly uniform, so coding the angles is not a lever.

## The chain — 2026-09-18 `[MEASURED]`

`pair.c`: one round trip. Side A holds 64 real GPT-2 neurons' notes (h5 c_fc,
768 each) and fires them as sorted-step edges; side B stamps, reads the dial,
adds; then sends each sum back as ONE edge (angle in timing, sign and laps as
the counter step). 400 ns tick, 3 copies + median: 64/64 sums exact (one run),
40-50/64 on runs where the OS parked the receiving core (bursts of 10k+
missed edges). The loop closes; the substrate's scheduler is the limit.

`chain.c --sections`: the owner's chain, 1 -> (2,3) -> back (4). Each side
runs its layers from its OWN notes (nothing crosses for them); only the
activations cross, 3 copies, one edge each, laps as counter step. Four real
GPT-2 64x64 slices (h5/h6 c_fc and c_proj), GELU and requantisation to the
dial at each layer, integer reference chain:
  **64/64 exact at every layer, three runs, 400 ns and 200 ns; 2 hops,
  192 edges per hop, 0.03-0.05 s.**
Because the adds are abelian, each section's adds happen whenever its
activations arrive; nothing waits on anything but the hop. Notes are
stationary. What crosses per hop is 3 x D edges regardless of how many notes
the section holds.

## 2026-09-18  sequence numbers + single-value repair on the line (libedge.c, sections.py)

Every edge now carries its sequence number beside it (seqw = flag+32, written just
before the flag). The receiver trusts an edge only if the edge before it in the
stream was its predecessor, so a miss can never shift a value onto the wrong
index; it just leaves a hole. B reports the hole indices (up to 8 per reply),
A refires only those values (edge_repair_send / edge_repair_recv), B rechecks
the checksum.

[MEASURED] full two-section GPT-2 run, 3 prompts x 32 tokens, tick 200 ns, 3 copies:
  all three prompts IDENTICAL to fp32 greedy; 25.0 tok/s; 0 full resends, 0 values
  repaired (results/sections4.log). Previous run (no sequence numbers) had to resend
  whole 2,318-edge vectors at steps 7-16 and was killed.
[MEASURED] forced-drop test (EDGE_DROP=5, copies=1): receiver lists [5, 6] missing
  (6 is untrusted because its predecessor never arrived), repair fills both exactly,
  0 wrong elsewhere. One run of the same test with copies=1 showed an unrelated
  natural miss (no vote at 1 copy); the 3-copy production path had none.

## 2026-09-18  hypothesis survival through superposition (hypo_survival.py, hypo_control.py)

K candidate next tokens (the context's top-K) ride through fp32 GPT-2 as ONE
vector at the new position. 3 prompts x 8 contexts. n_i = each hypothesis's own
clean argmax (K-row batch = exact answer).

[MEASURED]
  K   chord top-1 == greedy path   n_i in chord top-K   keyed (circ-conv keys, unbind block-11 residual): argmax / top-5
  2        75%                          62%                     0% / 0%
  4        67%                          52%                     0% / 0%
  8        54%                          53%                     0% / 0%
 16        42%                          51%                     0% / 0%
 32        38%                          52%                     0% / 0%
control:  n_i in hypothesis-1's OWN clean top-K: 60/50/43/41/41%;  n_i in the context's top-K: 8-35%.
Reading: the chord's rings are ~ the loudest hypothesis's rings plus common tokens;
superposing K adds little over running hypothesis 1 alone. Keys do not pass a dense W
(W(k*x) != k*(Wx)), so keyed hypotheses cannot be told apart at the head: 0%.
The sum is abelian and passes; GELU/LN/softmax are where hypotheses collapse into each other.

## 2026-09-18  rotators with their own notes (rotators.c)

N CPU threads, each lapping its own int16 note slice (acc += note*x), slice either
resident (1 MB, in the core's cache) or RAM-sized (64 MB). -O3, 32-bit lanes.
[MEASURED]
  resident 1 MB:  1 rot  9.3 G notes/s | 4 rot 42.0 G/s (84 GB/s)  | 10 rot 76.2 G/s (152 GB/s)  -> 615 GPT-2 laps/s IF all 124M notes were held this way
  RAM 64 MB:      1 rot 11.7 G notes/s | 4 rot 37.4 G/s (75 GB/s)  | 10 rot 55.6 G/s (111 GB/s)  -> 448 laps/s
Reading: one CPU rotator's adder tops out near 10 G notes/s whether its notes are
resident or in RAM; 10 rotators together exceed the bus only when resident (152 vs
111 GB/s). Resident capacity on this box is ~10-20 MB against 185 MB of notes.

## 2026-09-18  edges that never fire (zero_edges.py)
Inputs to every note table dialed at 4096 (unit = calib max/2047); an input of 0 ticks
would fire no edge and its notes would never be read.
[MEASURED] 0.1% of notes (0.1M of 123.5M) over 3 prompts x 32 tokens. mproj is the only
table with any (0.2%). At 4096 resolution this lever is empty.

## 2026-09-18  GPU cyclers with their own notes (cyclers.swift)
One cycler = one SIMD group holding 3840 int16 notes (5 neurons x 768); laps them for
T=2000 tokens against 768 activations shared per threadgroup. Two kernels: notes in a
register array (unrolled), and notes re-read every lap by address (offs[t]=0 so the
compiler cannot hoist).
[MEASURED] register-array kernel: ~0-1 G notes/s at every size (the array spills to private
  memory; a failed implementation, not a result).
[MEASURED] re-read-every-lap kernel: 32 cyc 20 G notes/s | 128 cyc 81 | 256 cyc 160 | 512-4096 cyc 144-153
  = ~300 GB/s equivalent, 3x the RAM bus, up to 31 MB of notes -- because threadgroups run
  in waves and each wave's notes sit in the GPU's near memory across its 2000 laps. Notes
  kept near the cycler lap at 3x the bus. What stays near the GPU across a WHOLE token
  pass is bounded by that memory: ~8 MB (onchip.swift, 2026-09-17).

## 2026-09-18  starting order does not matter (dial_flow_order.py)
The streaming dial machine (all blocks live every turn) with the blocks fired in a
RANDOM order each turn, in place, and/or random junk left on every accumulator at start.
[MEASURED] 'The capital of France is', T=128:
  fixed order, zero start:      ' the' MATCH, locked tick 15
  shuffled order, zero start:   ' the' MATCH, locked tick 7
  fixed order, noise start:     ' the' MATCH, locked tick 15
  shuffled order, noise start:  ' the' MATCH, locked tick 8
[MEASURED] but the WIRING order (which block's dials feed which) is not free: 12 blocks
  wired in a random depth order, 5 seeds -> ' fast', ' in', ' the', ' months', ' funky'.
  1 of 5 matched. Firing order and starting state are free; the trained wiring is not.
[MEASURED] GELU in the streaming dial machine (same prompt, T=128): gelu off (identity) -> ','
  DIFFERS (locked 13); clip +-3 -> '-' DIFFERS (locked 12); relu -> ',' DIFFERS (locked 10).
  The bend is doing something: without it the dials lock, but on the wrong token.

## 2026-09-18  the streams as edges only (dial_flow_order.py --stream)
Every value crossing between blocks in the streaming machine becomes one edge:
  dial  = angle on a 4096 dial + laps (unit = per-dim max/2047 from one plain pass)
  dial1 = 4096, one turn only (clipped)
  sign  = ONE BIT: only the edge's direction
[MEASURED]
  'The capital of France is': full ' the' lock 15 | dial ' the' lock 15 | dial1 ' the' lock 20 | sign ' the' lock 10, margin frozen at +0.504 from tick 15 on
  dial stream, 4 more prompts: ',' MATCH 22 | ' was' MATCH 11 | ' fence' MATCH 34 | ' the' MATCH 18   -> 5/5
  sign stream, 4 more prompts: ' to' (want ','), ' of' (' was'), ',' (' fence'), ',' (' the') -> 1/5, all locked at 13
Reading: the streams can be edges (angle + laps) with nothing lost, 5/5. One bit per
crossing locks hard and fast but on the wrong token 4 of 5 times.
[MEASURED] --stream bits: ONE BIT per crossing PER TICK. The receiver's dial holds the state
  across turns; each tick the sender fires only a direction (up/down vs what the receiver
  holds); the step adapts (x1.5 same direction, x0.5 on reversal, clamped to [unit/4, unit*2047]).
  5/5 MATCH. True lock (final answer arrives and never leaves; the earlier "lock" numbers in
  this file mean "first 8-turn run of any token", which can be a false lock):
  ' the' 69 | ',' 53 | ' was' 57 | ' fence' 60 | ' the' 43  of 128.
  Nothing but one bit per value per tick crosses between neurons; the hidden state lives on
  the receiving dials and is never sent.

## 2026-09-18  an edge only where something changed (dial_flow_order.py --stream dial, edge count)
The angle stream, counting an edge only where a crossing dial's angle differs from
last turn (the line holds its level otherwise). Input at full precision, all at once
(--input step; the default pulse input is width-coded so T is also its resolution:
T=1 -> 1-bit input -> ' of', T=2 -> ' (', T=4 -> ' the', T=8 -> ' now', both full and dial streams).
[MEASURED] 'The capital of France is', full-precision step input:
  turn 1 100% | 3 92% | 5 77% | 7 62% | 9 46% | 11 31% | 13 15% | turn 14 on: 0 edges
  A wavefront: each turn one more depth goes quiet (49,920/13 ~ 3,840 fewer edges per
  turn = one block's crossing). Answer ' the' MATCH at turn 13; after that nothing fires,
  the lines hold, the dials hold the state. Total edges to the answer: 13 turns x ~half
  = ~325k crossings, then zero for as long as the input stands.

## 2026-09-18  two predictions, both wrong (dial_flow_order.py, dial_flow_feedback.py)
Prediction 1: lock turn is set by how close the top two forks are, not by depth.
[MEASURED] full-precision step input, angle stream, 5 prompts: lock 13, 11, 13, 13, 13 with
  final margins +0.55, +1.90, +0.38, +0.73, +0.72. Lock = depth (12 blocks + head), margin
  irrelevant. WRONG.
Prediction 2: with the head fed back every turn, the whole sequence in flight, ~1 token/turn.
[MEASURED] case 0, 32 tokens, one new position per turn, 200 turns: correct-of-32 climbs
  0,2,5,7,8,9,11,14,15,17,19,21 ... never all 32; ~0.1 tokens/turn, i.e. ~1 per depth (13).
  A position's input is the ring below it, and that ring is right only after ITS input has
  settled 13 turns earlier; guesses feed guesses. WRONG: the sequence axis does not collapse
  by feedback alone; each token still costs a depth of turns.

## 2026-09-18  is the web abelian? (web_abelian.py)
Each of the 768 grains of the last token dropped down the web ALONE (prompt context fixed,
real grain size), rings recorded, then summed: rings(0) + sum_i [rings(x_i e_i) - rings(0)].
If the web were abelian this sum equals the real rings and IS the one top-to-bottom table.
[MEASURED] 5 prompts: 0/5 tokens match (' is' for ' the', 'a' for ',', 'nown' for ' was',
  ' table' for ' fence', ' I' for ' the'); top-5 overlap 0-1 of 5; ring-shape cosine -0.4..0.86.
  With GPT-2's trained notes, where a grain lands depends on the other grains. The web
  does not collapse to one table by superposition of grains.

## 2026-09-18  THE PILE MACHINE (piles.py) -- no matmul anywhere
Every neuron a pile on a 4096 dial (+laps); trained notes are chutes; a pile pours the
CHANGE down its chutes only when its dial moves a tick; bends are per-pile spill laws;
flips processed in random order; previous token's piles and earlier positions' K/V piles
left standing (only changes flow).
[MEASURED] 3 prompts x 8 tokens: IDENTICAL to fp32 greedy, all three, random flip order.
[MEASURED] per token: ~65,200 flips (= every pile, all 12 depths), 124M chute-walks (= every chute).
  The signal per token is 65k edges. The work per token is 124M chute-walks, because each
  edge's grains run down every chute of that pile, and in software a chute-walk is a note
  read. Standing water did not make it sparse: a new token moves every pile by more than a tick.

## 2026-09-18  THE REVERSE ROTATOR (reverse_rotator.py) -- adder + delay bucket in one, no multiply by a note
A chute = (from pile, to pile, OPEN TICK, laps, sign). No note value stored anywhere. When a
chute opens, the source's change is ADDED to the pile's pour register; every tick the register
is added to the level (the delay bucket). A change e through a chute that opened with `angle`
ticks left sweeps out e x angle by lap end: the product is time. Multi-turn notes open at tick
0 and pour `laps` copies for the whole turn. Negative note = same wire, reversed. Every pile
reads its level once at lap end and emits its dial (one edge back). Wiring 6 s; a lap 1-2.7 s.
[MEASURED] 'The capital of France is', position 0 ('The'): head = '\n' = fp32 from lap ~15;
  position 1 (' capital'): head = ' of' = fp32 from lap ~30. Both MATCH.
  Not quiet by 40 laps: piles still moving a tick 1,184 (pos 0) / 3,135 (pos 1) at lap 40, decaying
  ~15%/lap: tick flip-flops at dial boundaries drain down the depths as transients. The right
  token stands long before the tail dies. Chute events per lap ~120-195M early (bias water then
  the front), falling to ~15-35M by lap 40.
Bugs fixed on the way: laps double-counted (added to R and to LEVEL); angle-0 chutes opened for a
full turn; emissions/last-poured kept in float32 so unchanged dials give exactly zero delta.

## 2026-09-18  block swaps, done properly (correction of the "wiring is not free" claim)
Next-token agreement with the untouched GPT-2 over 66 contexts (3 reference prompts along
their greedy continuations + 3 long sentences at several cut points):
[MEASURED]
  adjacent swap (i,i+1):  0:11%  1:73%  2:79%  3:82%  4:77%  5:82%  6:79%  7:86%  8:88%  9:85%  10:74%
  skip one block:         0:5%   1:83%  2:85%  3:82%  4:82%  5:80%  6:82%  7:80%  8:86%  9:83%  10:74%  11:65%
  reverse blocks 3..8: 50% | reverse 2..9: 27% | shuffle 3..8 (5 draws): 56-70% | shuffle 1..10: 20-50% | shuffle all 12: 0-2%
Reading: the middle blocks commute for the next token most of the time (swap or drop any one:
~80%); only block 0 is fixed in place (swap or skip it: 5-11%). The earlier 1-of-5 result used
full random permutations including block 0 and was the wrong test. The order of the middle is
approximately free, as the block-swap literature says; the ends are not.

## 2026-09-18  ONE LAYER = ONE CIRCULAR CONVOLUTION, EXACT
Lay every neuron's row of notes end to end around one circle (one slot per weight); lay the
input on the same circle from slot 0 (zeros elsewhere); one circular correlation; read the
value at each row's offset = that neuron's output. No row is a rotation of another; the rows
are superposed one after another in time.
[MEASURED] GPT-2 block 5, random input, float64 FFT:
  fc   768->3072: circle 2.36M slots, one turn 47 ms, max |conv - layer| 2.5e-14  EXACT
  proj 768->768:  circle 0.59M slots, one turn  7 ms, max error 1.2e-14            EXACT
  qkv  768->2304: circle 1.77M slots, one turn 24 ms, max error 2.7e-14            EXACT
The notes' spectrum (rfft of the waveform) is fixed and can be recorded once: "record the
pattern". Per token per layer: one FFT of the input, one pointwise multiply, one inverse FFT:
"shoot it". The whole model is 49 such turns (48 layers + head, head circle 38.6M slots).
Earlier "circulant share = random" tested the WRONG thing (rows as rotations of one row).

## 2026-09-18  GPT-2 AS 49 TURNS (turns.py) -- every linear layer one circular convolution
Spectra of all 49 note waveforms recorded once (7 s; 85M slots in the blocks + 38.6M in the
head). Per position: 49 turns, each = FFT(input on the circle) x recorded spectrum -> inverse
FFT, read at row offsets. No matmul in the file.
[MEASURED] 'The capital of France is' + 4 tokens: IDENTICAL to fp32 greedy (' the capital of the').
  12.3 s per position in numpy float64 (the head's 38.6M-slot circle is most of it).

## 2026-09-18  THE SIMPLE VERSION: 49 turns on the circle, integers only (turns_dial.py)
Notes: integers on a 65,536 circle (12-bit angle + 16 laps), per-neuron gauge, input tick folded
into the note; negatives as complements (signed integers). Inputs: integer ticks on a 4096 dial
(+laps), head input on 16,384. A turn: one circular correlation of two integer waveforms; outputs
rounded to exact integers (with laps) -- the FFT's float dust removed; gauge -> real only for the
bends and water levels; outputs snapped back to per-neuron 4096 dials (+laps).
[MEASURED] 'The capital of France is' + 2: IDENTICAL to fp32 greedy (' the capital'). 12.3 s/position numpy.
  'Once upon a time' + 2: IDENTICAL (', the').  'In 1492, Columbus' + 2: IDENTICAL (' was the').  3/3.

## 2026-09-18  the same turn on the other axis
axis 1: each neuron's notes contiguous around the circle; the input as one contiguous chord at slot 0; neurons read at spaced offsets.
axis 2: each input's fan-out contiguous around the circle; the input as spaced impulses (x_i at slot i*nout); the output read as one contiguous chord in the first nout slots.
[MEASURED] block 5 fc / mproj / qkv: both axes EXACT (max error 2e-14 .. 5e-14). Same circle, same
notes, turned 90 degrees: which of input/output is spread out in time and which is a chord.

## 2026-09-18  each neuron's outputs: one segment per neuron, two turns on one circle
An MLP neuron owns ONE contiguous 768-slot segment of a 2.36M circle. Its input notes (fc
column) fill the segment in wave_in; its output notes (mproj row) fill the same segment in
wave_out. Turn 1 (axis 1): chord in at slot 0 -> every neuron's sum lands at its own segment
start. Bend applied in place there. Turn 2 (axis 2): those fired values, already at j*768,
are the impulses -> the next chord comes out contiguous in slots 0..767. Nothing moved between
the turns: a neuron's outputs run down the neuron's own segment.
[MEASURED] blocks 0, 5, 11: EXACT (2e-13 .. 1e-12).

## 2026-09-18  the "tree" between depths: pairs or bundle?
value notes of neuron j at depth k (mproj row) against key notes of neuron m at depth k+1 (fc
column, ln_2 gain folded): one coupling per (j, m). Share of a row's energy in its loudest partners:
[MEASURED]                 top-1    top-10   top-100   loudest/rms
  random 3072x768x3072     0.46%    3.4%     20.6%     3.7
  depth 0->1               1.99%    8.7%     31.2%     6.9
  depth 3->4               0.99%    5.7%     26.4%     5.3
  depth 5->6               1.12%    6.2%     26.6%     5.5
  depth 8->9               1.16%    5.8%     24.8%     5.5
  depth 10->11             1.90%    7.0%     25.4%     6.6
Reading: a neuron's output spreads over hundreds of next-depth neurons (top-100 hold a quarter of
it), ~2x more focused than random, nowhere near a tree. Between depths the coupling is a bundle
into the stream, not neuron-to-neuron pairs; each neuron's pair is (its key, the stream).

## 2026-09-18  collapsing a neuron onto the neurons above it: another turn
Path from depth k's fired neurons to depth k+1's neurons: value notes -> stream -> water level ->
key notes. The water level is centering (linear, (I - J/n)) times one scalar gain per position
(1/sigma of the stream). So the path is ONE 3072x3072 coupling C(sigma) = V (I-J/n) diag(g/sigma) K,
with the rest of the stream passing through the same lens.
[MEASURED] depth 2->3 and 6->7: collapsed = standard to 2e-14 (EXACT); C run as one turn on a
9.4M-slot circle: EXACT. Depth-to-depth is another matrix, hence another turn, with one scalar
(the water level's gain) carried per position.

## 2026-09-18  THE FRUSTUM: two rungs of the spiral with NO reading between them
Ring 1 (nin->nh): neuron j's notes at slots [j*nin, (j+1)*nin); its result lands at j*nin.
Ring 2 (nh->nout) laid on a circle `nout` times wider: note W2[j,m] at slot j*nin + m*N1, every
other slot silent. Ring 2 then only ever hears ring 1's neuron slots; the cross-talk at all
other lags never aligns with a note. The output m is read at lag m*N1.
[MEASURED] small instances (8->16->8 ... 32->96->32), float64 FFT:
  two turns, NO reading/masking between them:            EXACT (1e-15)
  both rungs as ONE fused turn (linear, spectrum X*conj(W1)*W2): EXACT (1e-16)
  the bend applied to the WHOLE ring-1 signal (junk included), no neuron reads itself, then ring 2: EXACT (1e-15)
Reading: on a widening circle the neuron never has to read itself; the geometry selects. The
earlier failure (cos 0.07) was the two rings on the SAME circumference. Cost: the circle grows by
the fan-out per fused rung (GPT-2 fc->mproj: 768*3072*768 = 1.8G slots; each further rung x768).
In a rotator that is a lap `nout` times longer, i.e. the cone's rings grow geometrically.

## 2026-09-18  the light cone of one note (per-neuron ticks = each neuron's max over the prompt / 2047)
One input dimension of the last position kicked by k of its own ticks; share of neurons per depth
that moved >= 1 of their ticks:
[MEASURED]  k=1: 0% everywhere. k=16: 0.5% at depth 0, 0 after. k=256: 77% (d0) -> 59 -> 45 -> 35 -> 29
  -> 24 -> 19 -> 17 -> 15 -> 16 -> 13 -> 12.6% (d11). k=2047 (full scale): 96.5% (d0) -> 81% (d11);
  moved >= 16 ticks: 59% (d0) -> 32 -> 20 -> 12 -> 7 -> 5 -> 3 -> 2.5 -> 2.9 -> 3.1 -> 2.0 -> 1.7% (d11).
  Head: 0% of logits moved a tick in every case.
Reading: the cone of a single note NARROWS with depth. Small notes (<= 16 ticks) are swallowed at
depth 0; a loud note reaches most neurons once but its disturbance decays ~2x per depth. The
medium damps single notes; only what many notes agree on climbs. The head hears chords, not notes.
[MEASURED] interference, two notes (dims 100, 500) at 256 ticks: neurons moved by (a+b) but by
neither alone: 1.2% (d0) rising to ~9-11% (d5-d11); moved by a or b alone but not by (a+b): 8-12%
at every depth. Roughly a tenth of the ring at every depth is constructive-or-destructive
interference between just two notes.

## 2026-09-18  how many harmonics does a layer need?
The layer as one convolution costs one operation per HARMONIC of its note waveform. Spectrum of
the row-concatenated waveform, bins sorted by energy:
[MEASURED] GPT-2 fc/mproj/qkv, blocks 0, 5, 11: 50% of energy needs 18% of bins, 90% needs 58%,
  99% needs 86%, 99.9% needs 95.5% -- identical to a random 768x3072 matrix (18.6/58.7/86.2/95.5).
  Keeping the top 50% of harmonics: layer error 16-39%; top 25%: 34-81%; top 10%: 61-101%.
Reading: GPT-2's note waveforms are white on the circle. The HRR form needs every harmonic, so
for these notes it costs the same as the matmul (N log N vs N in software; one turn in a medium).
The matmul -> HRR simplification is real only for notes MADE of few harmonics, i.e. a model
trained in that form. Same fact as the 11.09-bit entropy of the dial angles, from the other side.

## 2026-09-18  THE MODEL ON THE CIRCLE, WHOLE (circle_export.py, circle_run.py)
Every one of the 123.5M weights exported as ONE unsigned 16-bit integer on a 65,536 circle:
angle 0..4095 + 4096 x laps (0..15), negatives as complements (65536 - |q|), the input dial's unit
folded into the note, one gauge per column (what a tick is worth). Biases as integer ticks on the
same gauge. Token chords and positions as integer ticks on per-dimension residual dials. The
residual stream is stacked as integer ticks. Only the laws (water level, bend, limiter) see reals.
[MEASURED] 50.0% of notes are complements; laps: 91.4% zero, 5.9% one, 1.6% two, 0.6% three, ~1% more.
[MEASURED] circle_run.py loads only circle.npz (566 MB): 3/3 prompts x 4 tokens IDENTICAL to fp32.

## 2026-09-18  the light cone ON THE CIRCLE (light_cone_circle.py) -- exact tick changes
One input phase (dim 100 of the last position) kicked by k ticks; per depth, share of dials whose
tick count changed (exact) and mean |ticks| moved among them; head top-1 and margin change.
[MEASURED]  k=1:   75-82% of dials moved at every depth, by ~2-4 ticks; head ' the', margin +0.001
            k=16:  96% (d0, 16 ticks) -> 92% (d1, 9) -> 88% (d2, 6) -> 84% (d4, 4) -> 80% (d7, 3) -> 83% (d11, 3)
            k=256: 99.7% (d0, 246 ticks) -> 99.3% (d1, 125) -> 99.3% (d2, 76) -> 98.7% (d4, 36) -> 98.3% (d7, 24) -> 98.1% (d11, 19); margin -0.002
            k=1024: ~100% everywhere, 923 -> 233 ticks; head ' the', margin +0.129
            k=2047: 100%, 1609 -> 987 ticks; head flips to ' a', margin -0.604
Reading: on the circle the cone is FULL WIDTH from depth 0: one phase touches ~all dials at once.
What falls with depth is the AMPLITUDE of the disturbance (~13x from d0 to d11 at k=256), not its
reach. And up to k=256 the head's fork does not move (|dmargin| <= 0.005) although the entire ring
moved by tens of ticks: the disturbance is out of tune with the chord and never reaches the fork.
(The earlier float-space cone with bank-level ticks was mis-thresholded; this is the real one.)
  dim 500 gives the same picture: k=1 71-82% moved by 2-4 ticks; k=256 99.7% (198 ticks) -> 98.5% (26 ticks); head unmoved (<= 0.003) until k=1024; at 2047 margin -0.671 but still ' the'.

## 2026-09-18  where the time goes on the simulated medium (circle machine, per position)
[MEASURED] 570 ms per position, 124M landings each:
  superposing (input ticks x notes stacked into piles)  567.1 ms   99.6%
  laws (water level, bend)                                 0.8 ms    0.1%
  attention (resonance with the past, limiter)             1.1 ms    0.2%
  reading the output (gauge, dial, argmax)                 0.5 ms    0.1%
  block 5 fc, one chord: direct superposition 30.6 ms (2.36M int64 landings, numpy) vs FFT turn
  27.6 ms (all 2.36M lags, spectrum pre-recorded): same integers. For one chord the FFT buys nothing.
Reading: after the superposing, everything is free (0.4% of the time). The simulation's entire
cost is that it has to superpose by hand; a medium's waves do that part on their own.

## 2026-09-18  THE EDGE FIELD ON THE GPU (edges_export.py, edge_field.swift)
Every note exported as an EDGE: word = input index | fire tick (4096 - angle) | laps | sign, sorted
per pile by fire tick (123.1M angle edges + 10.7M lap edges, 4 bytes each, 589 MB). Every pile is
a GPU lane with an integrator: for t in 0..4095 { fire every edge whose tick is t: rate += x*dir;
level += rate }. Lap edges fire at tick 0 and are held the whole turn. The pile after the lap is
an exact int64 with laps; gauged on the CPU; the laws run on the CPU in double and go back to dials.
[MEASURED] 3 prompts x 4 tokens: ALL IDENTICAL to fp32 greedy.
  106.4 ms per position wall; 104.0 ms of it inside the 49 laps on the GPU; 200,704 ticks per
  position (49 x 4096); 134M edges fired per position. Effective tick on the M4 GPU as coded:
  104 ms / 200,704 = 0.52 us per tick (every lane steps every tick; the tick is the constraint).
  Reading the output: free (the pile is the answer). Data movement per position: the input ticks
  into each bank (~75k ints) and the piles back (~133k ints); the 134M edges never move.

## 2026-09-18  parallelising the edge field (edge_field.swift)
Start: 106 ms/position (9.4 tok/s): one bank's lanes at a time, each lane waiting 4096 serial ticks.
[MEASURED] ticks in parallel (64 blocks of 64 ticks per pile, prefix over blocks): 53 ms. Identical.
[MEASURED] coalesced edge layout (neighbouring lanes read neighbouring words, 1.27 GB padded): 53 ms
  (head 16.5 -> 9.6 ms, the rest slightly worse). Not memory-bound.
[MEASURED] CPU<->GPU round trip: 0.164 ms per command buffer; 49 per token = 8 ms. Kernels inside one
  buffer cost 5 us each. GPU-side execution was 38 of the 51 ms: the kernel itself was slow.
[MEASURED] collapsing the tick wait by doubling (x added n times via shift-adds): 38 -> 35 ms GPU.
  The wait loop was not the cost; the per-lane serial chain of dependent loads was (mproj, 48 edges
  per lane, slowest).
[MEASURED] one threadgroup (256 lanes) per pile sharing its sorted edge list, contributions by
  doubling, group sum: GPU 15.5 ms, wall 26.4 ms per position = 38 tok/s. Identical, 3 prompts x 4.
  Remaining: ~8 ms of round trips (laws on the CPU between laps) + ~2 ms conversions; GPU 15.5 ms vs
  a 5.4 ms memory floor (536 MB of edges per token at 100 GB/s) -- the 12-step doubling per edge.

## 2026-09-18  how close the forks are (prompt 3, fp32 margins per generated token)
[MEASURED] ' was' 0.374 | ' the' 0.859 | ' first' 2.169 | ' American' 0.759 | ' to' 0.522 | ' conquer' 0.077 |
  ' the' 1.936 | ' Atlantic' 0.012 (runner-up ' New').  On rings near -100, 0.012 is ~1 part in 10,000.
The circle export (notes with the input unit folded in, water-level outputs and the residual as
integer ticks) flipped token 6 (0.077) at 2047 and token 8 (0.012) at 8191 on this prompt;
double-precision laws (circle_run.py) and the GPU (optical_lm.swift) agree with each other, so
it is the representation's rounding, not the runtime. piles.py / edge_model (per-column notes,
no input-tick dial, float residual) happened to land on ' Atlantic'. A fork of 0.012 is below any
12-bit dial's resolution; "identical" there is luck in either direction. Criterion from here:
identical wherever the fp32 margin exceeds the dial's resolution; report the margin where it doesn't.

## 2026-09-18  OPTICAL LM (optical_export.py, optical_lm.swift): the three steps
Dense 16-bit edges (the circle number itself, index implicit): 247 MB. Laws on the GPU (water
level, bend, limiter, residual adds), one command buffer per token. Two firings: one op per edge,
or x added |c| times by doubling (shift-adds only).
[MEASURED] 3 prompts x 8 tokens: prompts 1-2 identical; prompt 3 identical to token 7, token 8 is the
  0.012 fork (' New' for ' Atlantic'). 23/24.
  one-op firing:  5.30 ms per position, 4.98 ms on the GPU  ->  189 tok/s
  doubling:      17.42 ms per position, 16.88 ms on the GPU ->   57 tok/s
  (from 106 ms this morning; the memory floor for 247 MB of edges at 100 GB/s is 2.5 ms.)

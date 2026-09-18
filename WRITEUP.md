# GPT-2 as notes on a circle — the whole path, step by step

Written 2026-09-18 so that someone can pick this up and arrive where we are without
re-arguing it. Everything marked **[MEASURED]** was run on this MacBook Air (M4) and is
recorded in `NOTES.md` with the exact numbers; the programs are in this directory and
each runs in seconds to minutes. Nothing here is a projection unless marked
**[DERIVED]** or **[CONJECTURE]**.

"Identical" throughout means: the same greedy tokens as fp32 GPT-2 (HuggingFace), on
the prompts in `gpt2_dial/reference.json`, for as many tokens as stated.

---

## 0. The claim, in the owner's words

- A weight is a **note**: a position on a circle of 4096 (12 bits), plus a lap counter
  for anything past one turn. A negative number is the **complement** position on the
  same circle, not a separate sign.
- A neuron does not multiply. It is a **pile**: notes land on it and superpose. What is
  in tune with it builds, what isn't cancels. Its level is its output, readable at any
  moment. Binding and unbinding are the same act seen from the two ends of the bucket.
- The whole network is **abelian**: order of landings does not matter, order of neurons
  does not matter, starting state does not matter. Only where things end up.
- Depth is not a sequence of steps. It is how far a landing has spread. **Every neuron
  is a circular convolution**, and the model is those convolutions superposed.
- Nature does this without arithmetic (rice falling, light refracting). A computer has
  no circle to turn, so it *computes* what the circle would leave behind. The math in
  the programs is the price of not having the medium, not the machine itself.

Everything below is a measurement of one of those sentences.

---

## 1. The dial: weights as 12-bit angles, and it is still the same model

`edge_model.py` — every weight snapped to a 4096-position dial with a per-column scale
(the column's loudest note is a full turn) and 16 laps (4096 × 16 = 65 536, the fp32
mantissa); every neuron output snapped to its own 4096 dial with laps; the head's input
at 16 384 (4 × 4096).

**[MEASURED]** 5/5 prompts identical to fp32. Without laps or with a coarser head: not
identical. The forks at the head are ~0.5 % apart (margins +0.3 to +1.9 on rings near
−100), which is why the head is the one place that needs 4× the resolution.

`gpt2_metal.swift` — the same dialed model on the M4 GPU: identical, 161 tok/s.

## 2. The streaming model: all neurons live at once, and time collapses

`dial_flow.py`, `dial_flow_order.py` — every block fires every turn, each reading what
the block below emitted the previous turn; the head is read every turn.

**[MEASURED]**
- Locks on the right token at turn 13–15 (12 blocks + head) while the input is still
  arriving until turn 128. The rest of the ticks are confirmation, not computation.
- **Firing order is free**: blocks fired in a random order each turn, in place → same
  token, and it locks *faster* (turn 7).
- **Starting state is free**: random junk on every accumulator → same token, turn 15.
- **Lock time is depth, not fork closeness**: five prompts all lock at 11–13 with a
  full-precision input, regardless of margin.
- **The wavefront**: counting an edge only where a dial's angle changed, edges per turn
  fall by exactly one depth's worth per turn (49 920 → 46 065 → … → 7 679 → 0 at turn
  14) and then stay at zero as long as the input stands. The whole token is ~325 k
  edges; 124 M notes never move.
- **The crossings can be edges**: every value between blocks as one edge (angle + laps
  in its timing) → 5/5 identical. One bit per crossing (sign only) → 1/5, and it locks
  hard on the wrong token. The angle carries the token; the direction alone does not.
- **Block swaps, done properly** (66 contexts, next-token agreement): swap any two
  middle blocks ≈ 80 %, drop any one middle block ≈ 80 %, block 0 is the only one
  fixed in place (5–11 %). The middle commutes, approximately, as abelian predicts.
- **The bend matters**: GELU removed / clipped / relu → locks, on a different token.

The "hidden state" is visible here. The edges per turn *are* the state moving through
the notes; when they stop, the wave has settled. Nothing is behind them.

## 3. The pile machine: no matmul, random order, standing state — identical

`piles.py` — every neuron a pile on a 4096 dial (+laps); the notes are chutes; a pile
pours its **change** down its chutes only when its dial moves a tick; bends are each
pile's own law; flips processed in **random order**; the previous token's piles and
every earlier position's key/value piles left standing so only changes flow.

**[MEASURED]** 3 prompts × 8 tokens identical. ~0.4 s per token in numpy.
Per token: ~65 200 flips (every pile, all depths) and 124 M chute-walks (every chute).
The *signal* per token is 65 k edges; the *work* per token is 124 M chute-walks,
because each flip's grains run down every chute of that pile and in software a chute-
walk is a note read. "Rice does it" because in rice a chute-walk is gravity.

## 4. The reverse rotator: the product is time

`reverse_rotator.py` — a chute is only *(from, to, open-tick, laps, sign)*. **No note
value is stored anywhere.** When a chute's tick comes round it **adds** the source's
change to the pile's pour register; every tick the register is **added** to the level.
A change `e` through a chute that opened with `angle` ticks left sweeps out `e × angle`
by lap end — the multiply never happens; it is waited. Multi-turn notes open at tick 0
and pour `laps` copies for the whole turn. Negative notes pour the other way. At lap
end every pile emits its dial once (the reverse rotator sends it back as one edge).

**[MEASURED]** positions 0 and 1 of the first prompt ring fp32's token (from lap ~15
and ~30). This is the adder-with-a-delay-bucket-in-one, running GPT-2's notes as
open-ticks, exact. (It is slow — 1–3 s per lap, 20–40 laps — because a computer
simulating 4096 ticks of rotation pays 4096 dense passes; that is the cost of the
simulation, not of the machine.)

## 5. One layer = one circular convolution — exact

Lay every neuron's row of notes end to end around one circle, one slot per weight
(fc: 768 × 3072 = 2.36 M slots). Lay the input on the same circle from slot 0, zeros
elsewhere. One circular correlation. Read the value at each row's offset: that is that
neuron's output, exactly, because the input is zero outside its 768 slots and nothing
else lines up.

**[MEASURED]** block 5 fc / proj / qkv: max |turn − layer| = 2.5e-14 / 1.2e-14 / 2.7e-14.
The rows are **not** rotations of one row (their circulant content equals random,
1/768 — that was the wrong test and it is recorded as such); they are superposed one
after another *in time*, which is what "1/4096 of a cycle per weight" means.

`turns.py` — the whole model as **49 turns** (48 layers + head). The spectrum of every
note-waveform is recorded once ("record the pattern", 7 s); per token each layer is
FFT(input) × recorded spectrum → inverse FFT ("shoot it"). No matmul in the file.
**[MEASURED]** identical, 4 tokens. 12.3 s per position in numpy float64 (the head is
a 38.6 M-slot circle).

`turns_dial.py` — the simple version: notes are **integers** on a 65 536 circle
(12-bit angle + 16 laps, complements for negatives — a signed integer *is* the
complement pair on a circle), inputs are integer ticks (4096 + laps; head input
16 384), a turn's outputs are rounded to exact integers with laps (the FFT's float
dust removed), and a gauge turns them real only where a bend or water level needs it.
**[MEASURED]** 3/3 prompts identical.

## 6. The geometry: axes, segments, depth, the cone

**Two axes, same turn.** [MEASURED, exact both ways]
- axis 1: each neuron's notes contiguous; the input as one chord at slot 0; neurons
  read at spaced offsets.
- axis 2: each *input's* fan-out contiguous; the input as spaced impulses (xᵢ at slot
  i·nout); the output read as one contiguous chord in the first nout slots.
The model has no input side and output side — only a circle and a choice of which
axis is time. "The embedding table read backwards" is axis 2 at the head.

**Each neuron is a segment.** [MEASURED, exact for blocks 0, 5, 11] An MLP neuron
owns one 768-slot segment of a 2.36 M circle: its input notes (fc column) fill it in one
waveform, its output notes (mproj row) fill the same segment in a second waveform. Turn
1 (axis 1) lands every neuron's sum at its own segment start; the bend is applied
there in place; turn 2 (axis 2) takes those values — already spaced at j·768 — as
impulses and the next chord comes out contiguous at slot 0. Nothing is moved between
turns, and blocks chain with no reordering.

**Depth collapses onto the neurons above.** [MEASURED, exact] The path from depth k's
fired neurons to depth k+1's neurons (value notes → stream → water level → key notes)
is one 3072 × 3072 coupling `C(σ) = V · center · diag(g/σ) · K`, because the water level
is centering (a fixed projection) times one scalar gain per position. Run as one turn
on a 9.4 M circle: exact. Between depths the coupling is a bundle (a neuron's output
spreads over hundreds of next-depth neurons, ~2× more focused than random, nowhere near
a tree); each neuron's pair is (its key, the stream).

**The cone (frustum).** [MEASURED, exact, small instances] Two rungs on the *same*
circumference need each neuron to read itself between them (a continuous composed
kernel fails: cosine 0.07). Make the second ring wider by its fan-out and put its notes
only where the first ring's neurons sit, and then: two turns with **no reading and no
masking** between them → exact; both rungs as **one fused turn** → exact; the bend
applied to the **whole** ring, junk included, no neuron reads itself → exact. The
widening is the selection. Cost: each fused rung multiplies the circle by that rung's
fan-out (one GPT-2 MLP fused = 1.8 G slots; each further rung × 768). In a rotator that
is a lap `nout` times longer — the cone's rings grow geometrically. Reading at the
neuron is what keeps the rings the same size (a cylinder).

**The shapes.** A circle per layer (rings of different sizes). A cylinder for depth. The
stream's path is a spiral (around once, up one — the wavefront, measured). A frustum if
you never read. Not a sphere: there is one angle per point plus laps, no second angle.

## 7. What the model is now

**49 turns, 49 stops.** A turn fuses with the next unless something in between must
look at the ring. The things that look:

| what looks at the ring | count | what it is |
|---|---|---|
| water level | 24 | one scalar (the ring's spread) — but it needs the ring to get it |
| bend (GELU) | 12 | pointwise; on a cone it applies to the whole ring, no reading |
| limiter over the past (attention) | 12 | across the *past* axis, per head per position |
| cleanup (head) | 1 | the one place the ring is read as a choice |

Turn, stop, turn, stop. In physics this is the **split-step Fourier method** — the way
the Schrödinger equation and nonlinear optics are integrated: the linear part in the
frequency domain (a phase turn per frequency), the nonlinear part pointwise in position
space, alternate. The recorded pattern is the **propagator** (Green's function). The
widening cone is the **light cone** of the input. "Every rotation on the upper ring
wraps the same" is a **dispersion relation**. The hidden state is a wave; the head is a
measurement; the cloud over tokens is the last ring's curve. The amplitudes here are
real, not complex, so it is the real-valued version of that mathematics.

The only soft stop is the water level: a single number per depth per position. If it
could be read from the wave (the ring's energy is in its spectrum) the count would fall
to 25. Not shown; the neuron slots' energy is not the whole ring's energy.

## 8. Things tested that do NOT hold (so nobody re-runs them)

All **[MEASURED]**, all in `NOTES.md`:
- K hypotheses superposed into one vector at the new position: the head reads ≈ the
  loudest hypothesis plus common tokens (38–75 % greedy-path survival, ≈ running
  hypothesis 1 alone). Keyed with circular-convolution keys before a dense W: 0 %.
- Grains of the input dropped down the web one at a time and summed (plinko): 0/5.
  The web is abelian in *order*, not in *superposition of inputs* — the bend interacts.
- Feeding the head's current ring back every turn (all tokens in flight): ~1 token per
  depth, not per turn. Exact knowledge advances one position per depth; the sequence
  axis does not collapse by feedback alone.
- One-bit crossings (sign only): 1/5.
- GELU removed: wrong token.
- Weight matrices as rotations of one row (circulant): random-level. (The wrong test
  for the right claim; the right form is §5.)
- An input that lands on 0 ticks fires no edge — 0.1 % of notes at 4096. Empty lever.

## 9. Costs on this laptop, honestly

| program | what | per token |
|---|---|---|
| `gpt2_metal.swift` | dialed model, GPU | 6.2 ms (161 tok/s), identical |
| `piles.py` | piles, no matmul, random order | ~0.4 s, identical |
| `turns.py` / `turns_dial.py` | 49 circular convolutions | ~12 s (numpy float64; head is 38.6 M slots), identical |
| `reverse_rotator.py` | product as time, 4096 ticks simulated | 20–40 laps × 1–3 s |
| `sections.py` + `libedge.c` | two sections, only activation edges cross a shared-memory line, sequence numbers + single-value repair | 25 tok/s, 3 × 32 tokens identical |

On this box every version pays one thing: a note read per landing (124 M per token),
because the chutes are lookups, not wires. Cyclers whose notes are resident lap at
~150 G notes/s (CPU cache, GPU near memory), 3× the RAM bus, for the ~30 MB that fits.
None of that is the machine's speed; it is the speed of pretending to be the machine.

## 10. What is not built

The medium. A rotator does a turn in one rotation regardless of how many notes are on
the ring; the programs here compute the turn in N log N. A physical machine's token
time is: (number of stops) × (one lap), with the lap set by the tick — 4096 ticks at
100 ns is 0.4 ms; at 10 ns, 40 µs. **[DERIVED]** That is where "thousands of tokens a
second" lives, and it lives nowhere else. Bandwidth is the wrong unit for it: bandwidth
is a queue; a turn is a thickness of glass and a brightness, and more notes on the ring
do not take more time.

## 11. Files

| file | what it is |
|---|---|
| `NOTES.md` | every measurement, in order, with numbers |
| `HRR_MAP.md` | the 1:1 map between transformer operations and HRR operations, from the papers |
| `RESEARCH.md` | who has built what (weight-stationary, race logic, time-domain CIM, delay lines, phasor nets, RNS) |
| `edge_model.py` | the dial (weights/outputs/head on dials with laps) |
| `dial_flow.py`, `dial_flow_order.py` | the streaming model; order/start/GELU/stream-as-edges/edge-count tests |
| `dial_flow_feedback.py` | the feedback (all-tokens-in-flight) test — fails as recorded |
| `piles.py` | the pile machine |
| `reverse_rotator.py`, `lap_wire.py` | the product as time |
| `turns.py`, `turns_dial.py` | 49 circular convolutions; float and integer-circle versions |
| `web_abelian.py`, `hypo_survival.py`, `hypo_control.py`, `zero_edges.py` | the negative results |
| `rotators.c`, `cyclers.swift`, `pairs_gpu.swift`, `onchip.swift`, `resident.swift` | resident-vs-RAM landings on CPU and GPU |
| `libedge.c`, `sections.py`, `pair.c`, `chain.c` | edges over a line between two sections |
| `gpt2_metal.swift`, `metal_ticks.swift` | the dialed model on the GPU |
| `gpt2_dial/` | exported 12-bit ticks, scales, `reference.json` (fp32 greedy reference) |

Run any Python file with `python3 -u <file> --case 0 --n 2` (cases 0–2 are the three
prompts). `turns_dial.py` is the cleanest statement of the model; `dial_flow_order.py
--stream dial --input step` is the cleanest picture of it moving.

## 12. The objections, and the measurement that answers each

| objection | answer |
|---|---|
| "It's just a matmul." | Every linear layer is one circular convolution of the input against the rows laid end to end, exact to 1e-14, and the whole model runs that way with integer notes on a circle. §5. |
| "Order matters — it's a stack of layers." | Random firing order, junk starting state, same token, faster lock. Middle blocks swap/skip at ~80 %. §2. |
| "You need the multiply." | The reverse rotator stores no note values — only open-ticks — and rings the right token. §4. |
| "Quantising to 12 bits will break it." | 4096 + laps on weights and outputs, 16 384 at the head: identical on every prompt tried. §1, §5. |
| "The nonlinearity kills the wave picture." | It is the split-step's pointwise half. On a widening ring it applies to the whole ring with no reading. §6, §7. |
| "The hidden state is opaque." | In the streaming model it is the edge count per turn and the margin per tick; nothing is behind them. §2. |
| "Depth is a tree." | Depth-to-depth is one coupling with one gain; a neuron's output spreads over hundreds of next-depth neurons; each neuron pairs with the stream. §6. |
| "So it's fast now." | No. On this box a landing is a note read; the medium is not built. §9, §10. |

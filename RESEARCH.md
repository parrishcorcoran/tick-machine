# Who has built "record the weights as a pattern in a line, then shoot it"

Compiled 2026-09-17 from search results (abstracts and summaries, not full
papers). `[cited]` names the source; my reading of it is marked `[reading]`.

## 1. The weights never move: Cerebras, Groq, TPU (commercial, shipping)

The single-stream problem is exactly the one they solved. A GPU re-reads every
weight from DRAM for every token; Groq and Cerebras keep the whole model in
on-chip SRAM so the weight never crosses a memory bus. Cerebras: 44 GB of SRAM
on one wafer, 21 PB/s internal, **~3,000 tok/s single stream** on a 120B
model; Groq: 230 MB per chip, hundreds of chips per model, ~500 tok/s.
`[cited]` Google's TPU is "weight-stationary": weights loaded once into a
systolic array, activations stream through. `[cited]`

`[reading]` This is the dial machine's thesis made of SRAM: the weight sits
where it is used, and the per-token traffic is the activations, not the
weights. Their 3,000 tok/s is what "bandwidth is no longer the issue" looks
like when it works. The cost is that the medium holding the weights is a
wafer.

## 2. Race logic: values as arrival times (UCSB, 2014-2019)

Madhavan, Sherwood, Strukov: information is a **timing delay**, not a level;
computation is which signal arrives first. MIN, MAX and ADD-BY-CONSTANT come
free from delay chaining. Built for dynamic programming (edit distance),
later Boosted Race Trees (ASPLOS 2019 best paper), and "hybrid temporal
computing" with deterministic summations (2025). `[cited]`

`[reading]` This is the edge machine's alphabet in the literature: an edge
whose *time* is its value, and a min/max/add-constant algebra — the tropical
semiring already in the timing-substrate repo. What race logic does not do is
the dense multiply-accumulate; the 2025 "deterministic summations" paper is
the closest to adding it.

## 3. Time-domain compute-in-memory (2019-2025)

Pulse-width-modulated multiply-accumulate in silicon: weights held in SRAM or
NAND flash, activations as pulse widths, sums as accumulated charge or time.
Reported 60-300 TOPS/W, bit-scalable 1-8 bits; one 28 nm chip runs AlexNet
and VGG at 1-8 bit. `[cited]`

`[reading]` Same machine, in production silicon, at 8 bits or fewer. Nobody
in these results reports 12 bits in the time domain; that is above the
demonstrated range, not a proven ceiling (PRIOR_ART.md says the same).

## 4. Delay lines and recirculating loops: the literal "record and replay"

Photonic neural networks keep memory as light circulating in a waveguide
loop; weight waveforms for different layers are separated by the loop delay
(~400 ns); iterative solvers run as pulses that recirculate and evolve by
propagation alone. `[cited]` Radio-frequency memories store a waveform in a
frequency-shifting recirculating delay line. `[cited]`

`[reading]` This is "record the pattern in a line and shoot it": the line IS
the recording. A pattern circulating in a loop is replayed every lap for
free, and taps along it read it as it passes. It is the 1940s mercury delay
line with light instead of sound. Capacity is set by loop length x bandwidth
(the number worked out in the AnalogLLM plan: GPT-2 needs ~18 km of fibre).

## 5. Spikes: weights as delays, values as first-spike times (2020-2024)

Time-to-first-spike coding: one spike, its time is the value; 15x lower power
and 5.7x faster decisions than rate coding in one hardware study. Synaptic
**delays** trained as parameters alongside or instead of weights ("pure
synaptic-delay training"). `[cited]`

`[reading]` The closest published thing to "a weight is when to fire". They
train small networks this way; nobody in these results converts a pretrained
transformer.

## 6. Stochastic / bit-stream computing

A value is the density of ones in a bit stream; multiply is an AND gate;
precision comes from stream length. Recent FPGA work: 0.14% accuracy loss on
MNIST at 99.7% energy saving. `[cited]`

`[reading]` One bit per cycle, exactly, but the value is in the *count*, not
the timing, so 12-bit precision costs 4096 bits of stream: the unary tax the
radix table in the timing-substrate README measured at 341x. The dial's
advantage over this is that one edge carries the whole 12 bits by position.

## What this says about the question

"Record the weights as a pattern and replay it" exists in three forms, and
all three are the same statement: **the recording medium is the line.**

| form | who | what holds the pattern | replay cost |
|---|---|---|---|
| SRAM next to the multiplier | Cerebras, Groq, TPU | transistors, on-chip | zero — nothing moves |
| a loop the pattern circulates in | photonic delay-line nets, RF memory | length of fibre or waveguide | zero — it comes round every lap |
| a stored level crossed by a ramp | time-domain CIM, NAND-flash PWM | a charge in a cell | zero — the ramp reads it in place |

None of them replays the pattern *through a bus*; that would be reading the
weights again. The pattern is replayed by being somewhere the computation
can touch it without moving it. On a laptop the only such place is cache,
which is 16 MB against 185 MB of GPT-2 at 12 bits. That is the whole gap, and
it is why the same idea gives 3,000 tok/s on a wafer and 540 on this Mac.

# Angles, notes and abelian spread across accelerators (2026-09-17, second search)

Searched from the owner's frame: value = angle, multiply = add angles,
sum = superposition, order free. What exists, and what it says about
spreading the machine across parallel units.

## 1. Spiking phasor networks — the dial machine, published
FHRR with the phase of each complex number encoded as a spike TIME within a
cycle; weights are phase shifters; a layer computes new phases from weighted
superpositions; the same network runs "with or without a temporal variable".
Orchard, *Efficient Hyperdimensional Computing with Spiking Phasors* (Neural
Computation 2024); *Deep Phasor Networks* (2021); *Deep Learning in Spiking
Phasor Neural Networks* (2022). `[cited]`
`[reading]` This is the edge machine's alphabet in a journal: an angle is when
the spike comes. They train from scratch; nobody puts a pretrained transformer
on it. Spread: one phasor per neuron, a layer per unit, superposition between.

## 2. Residue number systems — the abelian spread, exactly
A value is split into remainders on coprime moduli; every channel computes
independently with no carries; the answer is recombined only at the end.
RNSiM, RNSnet (in-memory), a photonic RNS accelerator with WDM, and
*Leveraging RNS for High-Precision Analog DNN Accelerators* (2023): low
precision channels compose into high-precision exact results. `[cited]`
`[reading]` This is the owner's "smaller complement dials + lap counter"
and "we don't care about anything but the end", as hardware. It is the
route by which 8-bit units (the Neural Engine's int8, a GPU's int8 dot
product) can do 12-bit dial arithmetic EXACTLY: run each residue channel on a
separate unit, combine at the end. On the two-core link the residues failed
because timing slips corrupt a remainder; on digital units there are no
slips. Costs no fewer bits per weight (residues sum to ~12 bits), buys
precision on cheap units, not bandwidth.

## 3. Logarithmic number systems — multiply is add, and the price
Multiply = add of logs; inference tolerates low LNS precision; patents for
LNS inference accelerators. The catch every paper names: ADDITION is not
closed in LNS and must be approximated. `[cited]`
`[reading]` The mirror of the phasor story: angles make multiply free and put
the cost on addition. FHRR pays that cost as the limiter (superpose, then
back to the circle). Same trade, two names.

## 4. Adds-only networks — "all we are is adds" has precedent
AdderNet (ResNet-50 at 76.8% top-1 with additions only), ShiftAddNet (shift +
add, up to 196x energy saving). `[cited]` Trained from scratch; not a
conversion of a pretrained model.

## 5. Superposition of models — abelian storage
Cheung et al., *Superposition of Many Models into One* (NeurIPS 2019): many
task models stored in one parameter set, each bound with a context (phase)
key, retrieved by unbinding. `[cited]` `[reading]` Abelian superposition used
as memory: the bind/unbind of FUNDAMENTALS 14 on real networks. Exact for
linear layers with orthogonal keys; the nonlinearities leak between models.

## 6. Photonic tensor cores — notes superposing in a medium
Accumulation in the wavelength domain: each wavelength a channel, weighting by
microrings, the sum taken by a photodetector; multidomain (time x wavelength x
microwave) multiplexing at 34 TOPS/mm^2. `[cited]` `[reading]` Spread across
accelerators = spread across wavelengths; the adder is the detector; the nests
are the multiplexing domains. The chord accumulator, in light.

## What this says for spreading across units on this laptop
The abelian spread that is buildable in software today is (2): residues.
Split every dial value into remainders on small coprime moduli, run the
whole linear algebra as independent low-precision integer channels, recombine
at the end. That turns low-precision parallel units into an exact 12-bit dial
machine. It does not reduce bytes per weight, so it does not move the RAM
wall; it moves the PRECISION wall, which is the one that kept the Neural
Engine and int8 paths off the table.

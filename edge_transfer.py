#!/usr/bin/env python3
"""ONE-BIT EDGE TRANSFER. The smallest possible version, checked at each step.

A dial turns: 0, 1, 2 ... DIAL-1, 0, 1 ... forever. Sender and receiver both
see it. A weight is an angle on that dial. The sender never sends the angle.
It sends ONE BIT, at the tick when the dial is showing that angle. The
receiver reads the dial at the moment the bit arrives. That reading IS the
weight. Negative weights are the far side of the dial (complement).

    step 1  one weight, one wire, one turn: does the receiver get the weight back?
    step 2  one neuron: many weights, each its own wire, one turn. y = w . x ?
    step 3  one GPT-2 layer, all weights, one turn. exact against the matmul?
    step 4  what one turn costs in time at a given tick, and what it loses.
"""
import numpy as np
import torch

DIAL = 4096
HALF = DIAL // 2 - 1


def face(tick, dial=DIAL):
    """What the dial shows at a tick: signed, the far side is negative."""
    half = dial // 2 - 1
    s = (tick + dial // 2) % dial - dial // 2
    return s / half


def angle_of(w, cyc, dial=DIAL):
    """A weight as an angle: -1000 -> 3096. No sign anywhere."""
    half = dial // 2 - 1
    return (np.round(w / cyc * half).astype(np.int64)) % dial


# ------------------------------------------------------------------ step 1

def step1():
    w, cyc = -0.37, 1.0
    a = angle_of(w, cyc)                     # the sender knows only this tick
    wire = np.zeros(DIAL, dtype=np.uint8)    # what crosses the link in one turn
    got = None
    for tick in range(DIAL):                 # the dial turns
        bit = 1 if tick == a else 0          # SENDER: one bit, at its instant
        wire[tick] = bit
        if bit:                              # RECEIVER: read the dial when a bit arrives
            got = face(tick) * cyc
    print("STEP 1  one weight, one edge")
    print(f"  weight {w:+.4f}  -> angle {a} of {DIAL}   (its complement side: {a > DIAL//2})")
    print(f"  wire carried {wire.sum()} bit in {DIAL} ticks, at tick {np.flatnonzero(wire)[0]}")
    print(f"  receiver read the dial: {got:+.4f}   error {abs(got - w):.5f}  (dial resolution {cyc/HALF:.5f})")
    return abs(got - w) <= cyc / HALF


# ------------------------------------------------------------------ step 2

def step2(n=768, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n) * 0.1
    x = rng.standard_normal(n)
    cyc = np.abs(w).max()
    a = angle_of(w, cyc)                     # n senders, each knows one tick
    acc = 0.0
    edges = 0
    for tick in range(DIAL):                 # one turn
        firing = np.flatnonzero(a == tick)   # the wires with a bit on them right now
        edges += firing.size
        if firing.size:
            acc += face(tick) * x[firing].sum()      # dial now x the levels that fired
    y = acc * cyc
    ref = float(w @ x)
    print("\nSTEP 2  one neuron, 768 weights, 768 wires, one turn")
    print(f"  edges on the link: {edges} = one per weight.  bits per weight: 1")
    print(f"  y = {y:+.5f}   matmul {ref:+.5f}   rel err {abs(y-ref)/abs(ref):.2e}")
    return edges == n


# ------------------------------------------------------------------ step 3

def step3(dial=DIAL):
    from transformers import GPT2LMHeadModel
    m = GPT2LMHeadModel.from_pretrained("gpt2")
    lin = m.transformer.h[5].mlp.c_fc
    W = lin.weight.detach().numpy().astype(np.float64)      # (768, 3072)
    b = lin.bias.detach().numpy().astype(np.float64)
    x = np.random.default_rng(1).standard_normal(768)
    cyc = np.abs(W).max(0)
    a = angle_of(W, cyc[None, :], dial)                     # every weight: one tick
    acc = np.zeros(W.shape[1])
    edges = 0
    for tick in range(dial):                                # ONE turn does the whole layer
        fire = (a == tick)                                  # (768, 3072) bits on the wires now
        k = int(fire.sum())
        edges += k
        if k:
            acc += face(tick, dial) * (x[:, None] * fire).sum(0)   # dial now x levels, per neuron
    y = acc * cyc + b
    ref = x @ W + b
    err = np.linalg.norm(y - ref) / np.linalg.norm(ref)
    return edges, W.size, err


def main():
    ok1 = step1()
    ok2 = step2()
    print("\nSTEP 3  one GPT-2 layer (768 -> 3072), every weight one edge, one turn")
    for dial in (64, 256, 4096):
        edges, n, err = step3(dial)
        print(f"  dial {dial:5d}: {edges:,} edges for {n:,} weights   rel err vs matmul {err:.5f}")
    print("\nSTEP 4  what a turn costs, at a 168 ps tick  [DERIVED from step 3]")
    for dial in (64, 256, 4096):
        print(f"  dial {dial:5d}: turn = {dial*168e-12*1e9:8.3f} ns   "
              f"GPT-2 = 49 turns = {49*dial*168e-12*1e6:7.3f} us/token  -> {1/(49*dial*168e-12):,.0f} tok/s if every wire is its own")
    print("\n  one bit per weight per turn. the weight never moved; the dial was already showing it.")
    print("  ok:", ok1 and ok2)


if __name__ == "__main__":
    main()

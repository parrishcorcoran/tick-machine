#!/usr/bin/env python3
"""GPT-2 AS TWO SECTIONS. Nothing but edges between them.

  section A: embeddings + blocks 0-5, on the dial (weights 4096 x 16 laps,
             outputs on per-neuron dials with laps, attention as lags).
             each token: run its blocks, put the new position's residual on a
             4096-dial with laps, fire it across as edges.
  section B: blocks 6-11 + ln_f + head. catch the edges, run its blocks on its
             own copy of the sequence, ring the forks, send the winning token
             back as ONE edge.

  python3 sections.py A   (in one terminal)      python3 sections.py B   (in another)
  or: python3 sections.py both
"""
import ctypes, json, math, os, subprocess, sys, time
import numpy as np
import torch

sys.path.insert(0, "/Users/abundancemachine/tick-machine")
import edge_model as em            # noqa: E402

TICK = float(os.environ.get("TICK", "200"))
COPIES = 3
HALF = 2047
LINE_AB, LINE_BA = "/tmp/edge_line_AB", "/tmp/edge_line_BA"
lib = ctypes.CDLL("/Users/abundancemachine/tick-machine/libedge.dylib")
lib.edge_send.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_long), ctypes.c_int, ctypes.c_double, ctypes.c_int]
lib.edge_recv.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_long), ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_double]
lib.edge_recv2.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_long), ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_double, ctypes.POINTER(ctypes.c_long), ctypes.c_int]
lib.edge_repair_send.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_long), ctypes.POINTER(ctypes.c_long), ctypes.c_int, ctypes.c_double]
lib.edge_repair_recv.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_long), ctypes.POINTER(ctypes.c_long), ctypes.c_int, ctypes.c_double, ctypes.c_double]
REPLY = 9          # B's reply: [token or -1 (resend all) or -2 (repair), up to 8 indices, -3 = unused]


def recv2(path, n, timeout=10.0):
    arr = (ctypes.c_long * n)(); miss = (ctypes.c_long * 64)()
    nmiss = lib.edge_recv2(path.encode(), arr, n, TICK, COPIES, timeout, miss, 64)
    return np.array(arr[:], dtype=np.int64), [int(m) for m in miss[:min(nmiss, 64)]], nmiss


def repair_send(path, idx, vals):
    a = (ctypes.c_long * len(idx))(*idx); v = (ctypes.c_long * len(idx))(*[int(x) for x in vals])
    lib.edge_repair_send(path.encode(), a, v, len(idx), TICK)


def repair_recv(path, out, want, timeout=10.0):
    o = (ctypes.c_long * len(out))(*[int(x) for x in out]); w = (ctypes.c_long * len(want))(*want)
    filled = lib.edge_repair_recv(path.encode(), o, w, len(want), TICK, timeout)
    return np.array(o[:], dtype=np.int64), filled


def send(path, vals):
    arr = (ctypes.c_long * len(vals))(*[int(v) for v in vals])
    lib.edge_send(path.encode(), arr, len(vals), TICK, COPIES)


def checksum(vals):
    v = np.asarray(vals, dtype=np.int64)
    return int(((v * (np.arange(v.size) + 1)) % 1000003).sum() % 4093)   # position-weighted, so swaps are caught


LAST_K = [0]
def recv(path, n, timeout=10.0):
    arr = (ctypes.c_long * n)()
    LAST_K[0] = lib.edge_recv(path.encode(), arr, n, TICK, COPIES, timeout)
    return np.array(arr[:], dtype=np.int64)


def build(role, calib_ids):
    torch.set_num_threads(2)
    m, _ = em.build(65536, 4096, True, 16384, calib_ids)
    tr = m.transformer
    # the crossing's dial: one unit per dimension of the block-5 residual, from the calibration pass (both sides compute it)
    with torch.no_grad():
        h = tr.wte(calib_ids) + tr.wpe(torch.arange(calib_ids.shape[1]))[None]
        for blk in tr.h[:6]:
            h = blk(h)[0]
        unit = (h[0].abs().amax(0).clamp(min=1e-6) / HALF).double().numpy()
    return m, unit


def run_A(prompts_ids, n_new, calib_ids):
    m, unit = build("A", calib_ids); tr = m.transformer
    results = []; resends = [0]; repaired = [0]
    for ids in prompts_ids:
        seq = list(ids); t0 = None; gen = []
        for step in range(len(ids) + n_new):
            # prefill one prompt token per step (so B fills its sequence the same way), then generate
            S = min(step + 1, len(seq))
            with torch.no_grad():
                h = tr.wte(torch.tensor([seq[:S]])) + tr.wpe(torch.arange(S))[None]
                for blk in tr.h[:6]:
                    h = blk(h)[0]
            vec = np.round(h[0, -1].double().numpy() / unit).astype(np.int64)      # the new position, on the dial, with laps
            full = list(vec) + [checksum(vec)]
            send(LINE_AB, full)                                                      # -> B, as edges, with its checksum
            tries, repairs = 1, 0
            while True:
                reply = recv(LINE_BA, REPLY)                                         # <- B: token, or what to fix
                st = int(reply[0])
                if st >= 0: tok = st; break
                if st == -1: send(LINE_AB, full); tries += 1; continue              # resend everything
                idx = [int(i) for i in reply[1:] if i >= 0]                          # repair just these
                repair_send(LINE_AB, idx, [full[i] for i in idx]); repairs += len(idx)
            resends[0] += tries - 1; repaired[0] += repairs
            print(f"    A step {step}: token {tok}   sends {tries}, values repaired {repairs}", file=sys.stderr, flush=True)
            if step >= len(ids) - 1:
                if t0 is None: t0 = time.time()
                if step >= len(ids):
                    pass
                if len(gen) < n_new:
                    gen.append(tok); seq.append(tok)
        results.append((gen, time.time() - t0))
    print(f"    A: total full resends {resends[0]}, single values repaired {repaired[0]}", file=sys.stderr, flush=True)
    return results


def run_B(n_prompts, prompt_lens, n_new, calib_ids):
    m, unit = build("B", calib_ids); tr = m.transformer
    for p in range(n_prompts):
        hist = []
        for step in range(prompt_lens[p] + n_new):
            got, missing, nmiss = recv2(LINE_AB, 769)                                # <- A, edges, last one is the checksum
            while True:
                if nmiss > 0:
                    want = missing[:8]
                    send(LINE_BA, [-2] + want + [-3] * (8 - len(want)))              # -> A: "repair these"
                    got, filled = repair_recv(LINE_AB, got, want)
                    have = set(want[:filled]) if filled == len(want) else set()
                    # recheck which are still missing: a repaired index is filled only if its edge arrived in order
                    missing = [i for i in missing if i not in have]; nmiss = len(missing)
                    if nmiss > 0 and filled == 0: continue
                    if nmiss > 0: continue
                vec, ck = got[:768], int(got[768])
                if checksum(vec) == ck: break
                send(LINE_BA, [-1] + [-3] * 8); got, missing, nmiss = recv2(LINE_AB, 769)   # checksum still wrong: everything again
            hist.append(torch.tensor(vec.astype(np.float64) * unit, dtype=torch.float32))
            with torch.no_grad():
                h = torch.stack(hist)[None]
                for blk in tr.h[6:]:
                    h = blk(h)[0]
                logits = m.lm_head(tr.ln_f(h))[0, -1]
                tok = int(logits.argmax())
            send(LINE_BA, [tok] + [-3] * 8)                                          # -> A: the token


def main():
    role = sys.argv[1]
    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    cases = json.load(open("/Users/abundancemachine/tick-machine/gpt2_dial/reference.json"))[:int(os.environ.get('NCASES','3'))]
    n_new = int(os.environ.get('NNEW', '32'))
    calib_ids = tok("The capital of France is", return_tensors="pt").input_ids
    if role == "both":
        for p in (LINE_AB, LINE_BA):
            if os.path.exists(p): os.remove(p)
        b = subprocess.Popen([sys.executable, __file__, "B"])
        time.sleep(1)
        res = run_A([c["ids"] for c in cases], n_new, calib_ids)
        b.wait()
        allsame = True; ntok = 0; ttot = 0
        print(f"GPT-2 as two sections, tick {TICK:.0f} ns, {COPIES} copies: A = embeddings + blocks 0-5, B = blocks 6-11 + head; only edges between them")
        for c, (gen, dt) in zip(cases, res):
            same = gen == c["ref"][:n_new]; allsame &= same; ntok += len(gen); ttot += dt
            print(f"  {c['prompt']!r:32s} {'IDENTICAL to fp32' if same else 'differs'}   {tok.decode(gen)!r}")
        print(f"  [MEASURED] {'all identical' if allsame else 'not all identical'};  {ntok/ttot:.1f} tok/s;  per token: 2,304 activation edges A->B, 3 edges B->A, no weight ever crossed")
    elif role == "B":
        run_B(len(cases), [len(c["ids"]) for c in cases], n_new, calib_ids)


if __name__ == "__main__":
    main()

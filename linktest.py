import ctypes, os, sys, time, numpy as np, subprocess
lib = ctypes.CDLL("/Users/abundancemachine/tick-machine/libedge.dylib")
lib.edge_send.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_long), ctypes.c_int, ctypes.c_double, ctypes.c_int]
lib.edge_recv.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_long), ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_double]
P=b"/tmp/edge_line_T"; TICK=200.0
def send(vals): arr=(ctypes.c_long*len(vals))(*vals); return lib.edge_send(P,arr,len(vals),TICK,3)
def recv(n): arr=(ctypes.c_long*n)(); k=lib.edge_recv(P,arr,n,TICK,3,5.0); return list(arr), k
rng=np.random.default_rng(0)
tests=[[262],[0],[-1],[50256],list(rng.integers(-40000,40000,769))]
if sys.argv[1]=="B":
    for t in tests:
        got,k=recv(len(t)); print(f"  B: n={len(t):3d} edges seen {k:4d}  exact {sum(1 for a,b in zip(got,t) if a==b)}/{len(t)}  first {got[:3]} vs {t[:3]}", flush=True)
else:
    b=subprocess.Popen([sys.executable,__file__,"B"]); time.sleep(0.5)
    for t in tests: r=send(t); print(f"  A: sent n={len(t):3d} ({r} edges)", flush=True)
    b.wait()

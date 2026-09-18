// OPTICAL LM. GPT-2 as notes on a circle, run as an edge field on the GPU, one command buffer per token.
//   edges : dense uint16 [nin][nout] -- the circle number itself: phase (12 bits) + laps (4 bits), negatives as complements
//   a lap : one threadgroup per 32 piles; 8 lanes per pile stream its edges; an edge held for n ticks adds x n times
//           (DOUBLING: by shift-adds; otherwise: one op). The pile is an exact int64 with laps.
//   laws  : water level, bend, limiter over the past, residual adds -- small kernels on the GPU, fp32, results back onto dials
//   read  : the head's piles, gauged; argmax on the CPU. Everything else never leaves the GPU.
//   swiftc -O -o optical_lm optical_lm.swift && ./optical_lm [doubling]
import Foundation
import Metal

let DOUBLING = CommandLine.arguments.contains("doubling")
let src = """
#include <metal_stdlib>
using namespace metal;
#define SUB 8
#define PILES 32
inline long fire(int x, int c) {                            // an edge (circle number c) meets input x: x held for |c| ticks
    int v = c < 32768 ? c : c - 65536; int m = v < 0 ? -v : v; long xv = x;
#ifdef DOUBLING
    long acc = 0; for (int bit = 0; bit < 15; bit++) { if (m & (1 << bit)) acc += xv; xv <<= 1; }
#else
    long acc = xv * m;
#endif
    return v < 0 ? -acc : acc;
}
kernel void lap(device const ushort* edges [[buffer(0)]], device const int* x [[buffer(1)]], device const long* bias [[buffer(2)]], device long* out [[buffer(3)]],
                constant uint& nin [[buffer(4)]], constant uint& nout [[buffer(5)]], uint t [[thread_position_in_threadgroup]], uint g [[threadgroup_position_in_grid]]) {
    threadgroup long part[SUB][PILES];
    uint pile = g * PILES + (t % PILES), sub = t / PILES;
    long acc = 0;
    if (pile < nout) for (uint i = sub; i < nin; i += SUB) acc += fire(x[i], (int)edges[i * nout + pile]);
    part[sub][t % PILES] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sub == 0 && pile < nout) { long s = bias[pile]; for (uint k = 0; k < SUB; k++) s += part[k][t % PILES]; out[pile] = s; }
}
kernel void gauge_dial(device const long* level [[buffer(0)]], device const float* gauge [[buffer(1)]], device const float* uout [[buffer(2)]], device float* y [[buffer(3)]],
                       constant uint& n [[buffer(4)]], uint j [[thread_position_in_grid]]) {
    if (j >= n) return; long v = level[j]; long hi = v >> 20, lo = v & 0xFFFFF;      // keep the integer's precision through the float
    float r = ((float)hi * 1048576.0f + (float)lo) * gauge[j];
    y[j] = uout[j] > 0 ? rint(r / uout[j]) * uout[j] : r;                             // back onto its dial
}
kernel void to_ticks(device const float* y [[buffer(0)]], device const float* uin [[buffer(1)]], device int* x [[buffer(2)]], constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < n) x[i] = (int)rint(y[i] / uin[i]); }
kernel void gelu(device float* y [[buffer(0)]], constant uint& n [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i < n) { float v = y[i]; float a = clamp(0.7978845608f * (v + 0.044715f * v * v * v), -10.0f, 10.0f); y[i] = 0.5f * v * (1.0f + tanh(a)); } }
kernel void resadd(device int* res [[buffer(0)]], device const float* y [[buffer(1)]], device const float* ures [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < 768) res[i] += (int)rint(y[i] / ures[i]); }
kernel void water(device const int* res [[buffer(0)]], device const float* ures [[buffer(1)]], device const float* w [[buffer(2)]], device const float* b [[buffer(3)]],
                  device float* y [[buffer(4)]], uint t [[thread_position_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float ps[8], pq[8]; float v[3]; float s = 0, q = 0;
    for (int k = 0; k < 3; k++) { v[k] = (float)res[t + 256 * k] * ures[t + 256 * k]; s += v[k]; }
    s = simd_sum(s); if (lane == 0) ps[sg] = s; threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = (ps[0]+ps[1]+ps[2]+ps[3]+ps[4]+ps[5]+ps[6]+ps[7]) / 768.0f;
    for (int k = 0; k < 3; k++) q += (v[k] - mean) * (v[k] - mean);
    q = simd_sum(q); if (lane == 0) pq[sg] = q; threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = rsqrt((pq[0]+pq[1]+pq[2]+pq[3]+pq[4]+pq[5]+pq[6]+pq[7]) / 768.0f + 1e-5f);
    for (int k = 0; k < 3; k++) { uint i = t + 256 * k; y[i] = (v[k] - mean) * inv * w[i] + b[i]; }
}
kernel void store_kv(device const float* qkv [[buffer(0)]], device float* kv [[buffer(1)]], constant uint& pos [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < 2304) kv[pos * 2304 + i] = qkv[i]; }
kernel void attn(device const float* qkv [[buffer(0)]], device const float* kv [[buffer(1)]], device float* o [[buffer(2)]], constant uint& pos [[buffer(3)]], uint h [[thread_position_in_grid]]) {
    if (h >= 12) return; float sc[64]; float mx = -1e30f;
    for (uint s = 0; s <= pos; s++) { float d = 0; for (uint k = 0; k < 64; k++) d += qkv[h*64 + k] * kv[s*2304 + 768 + h*64 + k]; sc[s] = d / 8.0f; mx = max(mx, sc[s]); }
    float z = 0; for (uint s = 0; s <= pos; s++) { sc[s] = exp(sc[s] - mx); z += sc[s]; }
    for (uint k = 0; k < 64; k++) { float a = 0; for (uint s = 0; s <= pos; s++) a += sc[s] / z * kv[s*2304 + 1536 + h*64 + k]; o[h*64 + k] = a; }
}
"""
setvbuf(stdout, nil, _IONBF, 0)
let dir = "/Users/abundancemachine/tick-machine/gpt2_dial/optical/"
let dev = MTLCreateSystemDefaultDevice()!, q = dev.makeCommandQueue()!
let opts = MTLCompileOptions(); if DOUBLING { opts.preprocessorMacros = ["DOUBLING": 1 as NSNumber] }
let lib = try! dev.makeLibrary(source: src, options: opts)
func pipe(_ n: String) -> MTLComputePipelineState { return try! dev.makeComputePipelineState(function: lib.makeFunction(name: n)!) }
let pLap = pipe("lap"), pGauge = pipe("gauge_dial"), pTicks = pipe("to_ticks"), pGelu = pipe("gelu"), pRes = pipe("resadd"), pWater = pipe("water"), pStore = pipe("store_kv"), pAttn = pipe("attn")
func load<T>(_ name: String, _ t: T.Type) -> [T] { let d = try! Data(contentsOf: URL(fileURLWithPath: dir + name)); return d.withUnsafeBytes { Array($0.bindMemory(to: T.self)) } }
func buf<T>(_ a: [T]) -> MTLBuffer { return dev.makeBuffer(bytes: a, length: max(a.count, 1) * MemoryLayout<T>.stride, options: .storageModeShared)! }
func fbuf(_ n: Int) -> MTLBuffer { return dev.makeBuffer(length: max(n, 1) * 8, options: .storageModeShared)! }
struct Bank { let nin, nout: Int; let edges, gauge, uIn, uOut, bias, x, level, y: MTLBuffer }
let man = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: dir + "manifest.json"))) as! [[String: Any]]
var banks = [String: Bank](); var bytes = 0
for b in man {
    let n = b["name"] as! String, nin = b["nin"] as! Int, nout = b["nout"] as! Int
    let e = load("\(n).edges.bin", UInt16.self); bytes += e.count * 2
    banks[n] = Bank(nin: nin, nout: nout, edges: buf(e), gauge: buf(load("\(n).gauge.bin", Float.self)), uIn: buf(load("\(n).u_in.bin", Float.self)), uOut: buf(load("\(n).u_out.bin", Float.self)),
                    bias: buf(load("\(n).bias.bin", Int64.self)), x: fbuf(nin), level: fbuf(nout), y: fbuf(nout))
}
let L = 12, D = 768, V = 50257, MAXPOS = 128
let uRes = buf(load("u_res.bin", Float.self)), wte = load("wte.ticks.bin", Int32.self), wpe = load("wpe.ticks.bin", Int32.self)
let ln1w = (0..<L).map { buf(load("ln.\($0).ln_1.weight.bin", Float.self)) }, ln1b = (0..<L).map { buf(load("ln.\($0).ln_1.bias.bin", Float.self)) }
let ln2w = (0..<L).map { buf(load("ln.\($0).ln_2.weight.bin", Float.self)) }, ln2b = (0..<L).map { buf(load("ln.\($0).ln_2.bias.bin", Float.self)) }
let lnfw = buf(load("ln_f.weight.bin", Float.self)), lnfb = buf(load("ln_f.bias.bin", Float.self))
let res = fbuf(D), nrm = fbuf(D), o = fbuf(D); let kv = (0..<L).map { _ in dev.makeBuffer(length: MAXPOS * 2304 * 4, options: .storageModePrivate)! }
print("optical LM on \(dev.name): \(bytes / 1_000_000) MB of 16-bit edges resident; firing by \(DOUBLING ? "doubling (shift-adds)" : "one op"); one command buffer per token")

func enc(_ cb: MTLCommandBuffer, _ ps: MTLComputePipelineState, _ bufs: [MTLBuffer], _ consts: [UInt32], threads: Int, tg: Int, groups: Bool = false) {
    let e = cb.makeComputeCommandEncoder()!; e.setComputePipelineState(ps)
    for (i, b) in bufs.enumerated() { e.setBuffer(b, offset: 0, index: i) }
    for (k, c) in consts.enumerated() { var v = c; e.setBytes(&v, length: 4, index: bufs.count + k) }
    if groups { e.dispatchThreadgroups(MTLSize(width: threads, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1)) }
    else { e.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1)) }
    e.endEncoding()
}
func lapOf(_ cb: MTLCommandBuffer, _ bk: Bank, from y: MTLBuffer, dial: Bool) {           // y (real) -> ticks -> one lap -> gauge (-> dial)
    enc(cb, pTicks, [y, bk.uIn, bk.x], [UInt32(bk.nin)], threads: bk.nin, tg: 256)
    enc(cb, pLap, [bk.edges, bk.x, bk.bias, bk.level], [UInt32(bk.nin), UInt32(bk.nout)], threads: (bk.nout + 31) / 32, tg: 256, groups: true)
    enc(cb, pGauge, [bk.level, bk.gauge, dial ? bk.uOut : buf([Float](repeating: 0, count: bk.nout)), bk.y], [UInt32(bk.nout)], threads: bk.nout, tg: 256)
}
var gpuTime = 0.0
func position(_ pos: Int, _ tid: Int) -> Int {
    let rp = res.contents().bindMemory(to: Int32.self, capacity: D); for i in 0..<D { rp[i] = wte[tid * D + i] + wpe[pos * D + i] }
    let cb = q.makeCommandBuffer()!
    for k in 0..<L {
        let bq = banks["qkv.\(k)"]!, bp = banks["proj.\(k)"]!, bf = banks["fc.\(k)"]!, bm = banks["mproj.\(k)"]!
        enc(cb, pWater, [res, uRes, ln1w[k], ln1b[k], nrm], [], threads: 1, tg: 256, groups: true)
        lapOf(cb, bq, from: nrm, dial: true)
        enc(cb, pStore, [bq.y, kv[k]], [UInt32(pos)], threads: 2304, tg: 256)
        enc(cb, pAttn, [bq.y, kv[k], o], [UInt32(pos)], threads: 12, tg: 12)
        lapOf(cb, bp, from: o, dial: true); enc(cb, pRes, [res, bp.y, uRes], [], threads: D, tg: 256)
        enc(cb, pWater, [res, uRes, ln2w[k], ln2b[k], nrm], [], threads: 1, tg: 256, groups: true)
        lapOf(cb, bf, from: nrm, dial: true); enc(cb, pGelu, [bf.y], [UInt32(bf.nout)], threads: bf.nout, tg: 256)
        lapOf(cb, bm, from: bf.y, dial: true); enc(cb, pRes, [res, bm.y, uRes], [], threads: D, tg: 256)
    }
    let bh = banks["head"]!
    enc(cb, pWater, [res, uRes, lnfw, lnfb, nrm], [], threads: 1, tg: 256, groups: true); lapOf(cb, bh, from: nrm, dial: false)
    let t0 = Date(); cb.commit(); cb.waitUntilCompleted(); gpuTime += Date().timeIntervalSince(t0)
    let lg = bh.y.contents().bindMemory(to: Float.self, capacity: V); var best = 0; for t in 1..<V where lg[t] > lg[best] { best = t }; return best
}
let ref = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: "/Users/abundancemachine/tick-machine/gpt2_dial/reference.json"))) as! [[String: Any]]
let NNEW = Int(ProcessInfo.processInfo.environment["NNEW"] ?? "8") ?? 8; var allSame = true; var positions = 0
_ = position(0, 464); gpuTime = 0                                                          // warm
let tAll = Date()
for c in ref {
    let ids = c["ids"] as! [Int], want = Array((c["ref"] as! [Int]).prefix(NNEW))
    var nxt = 0; for (i, t) in ids.enumerated() { nxt = position(i, t); positions += 1 }
    var gen = [Int](); var pos = ids.count
    while gen.count < NNEW { gen.append(nxt); nxt = position(pos, nxt); pos += 1; positions += 1 }
    let same = gen == want; allSame = allSame && same
    print("  \(c["prompt"] as! String) + \(NNEW): \(same ? "IDENTICAL to fp32 greedy" : "differs") \(gen)\(same ? "" : "  (reference \(want))")")
}
let wall = Date().timeIntervalSince(tAll)
print(String(format: "[MEASURED] %@; %d positions; %.2f ms per position (%.0f tok/s); %.2f ms of it on the GPU in one command buffer; 49 laps x 4096 ticks; %d M edges", allSame ? "all identical" : "not all identical", positions, wall / Double(positions) * 1e3, Double(positions) / wall, gpuTime / Double(positions) * 1e3, bytes / 2_000_000))

// THE EDGE FIELD. Every pile is a GPU lane with an integrator: `level += rate` every tick of a
// 4096-tick lap. Every note is an EDGE that fires at its phase: when the lap reaches its fire
// tick, `rate += input × direction` -- an add, no note value, no multiply. Lap edges fire at tick
// 0 and are held the whole turn. After the lap the pile IS the answer; reading it is free.
// 49 laps per token (48 banks + head). The laws (water level, bend, limiter over the past) run on
// the CPU between laps in double precision and put their result straight back onto its dial.
//   swiftc -O -o edge_field edge_field.swift && ./edge_field
import Foundation
import Metal

let src = """
#include <metal_stdlib>
using namespace metal;
// TICKS IN PARALLEL. One lane per (pile, block of 64 ticks): it walks only its block's edges,
// 64 serial ticks, and reports the block's rate S and its local integral LOC. A second kernel per
// pile adds the blocks up with a prefix: the rate carried in from earlier blocks is held for the
// block's 64 ticks (a shift, not a multiply). Adds and shifts only; nothing waits 4096 ticks.
kernel void blocks(device const uint* notes [[buffer(0)]], device const uint* blkoff [[buffer(1)]], device const int* x [[buffer(2)]],
                   device int* S [[buffer(3)]], device long* LOC [[buffer(4)]], constant uint& nout [[buffer(5)]], uint gid [[thread_position_in_grid]]) {
    uint j = gid >> 6, b = gid & 63; if (j >= nout) return;
    uint p = blkoff[j * 65 + b], e = blkoff[j * 65 + b + 1];
    int r = 0; long local = 0; int t0 = (int)b * 64;
    for (int t = t0; t < t0 + 64; t++) {
        while (p < e && (int)((notes[p] >> 12) & 4095) == t) { uint w = notes[p]; int i = w & 4095; int sg = ((w >> 28) & 1) ? -1 : 1; r += x[i] * sg; p++; }
        local += r; }
    S[gid] = r; LOC[gid] = local;
}
kernel void gather(device const int* S [[buffer(0)]], device const long* LOC [[buffer(1)]], device const uint* lapnotes [[buffer(2)]], device const uint* lapoff [[buffer(3)]],
                   device const int* x [[buffer(4)]], device const int* bias [[buffer(5)]], device long* out [[buffer(6)]], constant uint& nout [[buffer(7)]], uint j [[thread_position_in_grid]]) {
    if (j >= nout) return;
    long rate0 = 0;
    for (uint p = lapoff[j]; p < lapoff[j+1]; p++) { uint w = lapnotes[p]; int i = w & 4095; int laps = (w >> 24) & 15; int sg = ((w >> 28) & 1) ? -1 : 1; rate0 += (long)x[i] * sg * laps; }
    long level = rate0 << 12; long prefix = 0;                              // lap edges held all 4096 ticks
    for (uint b = 0; b < 64; b++) { level += (prefix << 6) + LOC[j * 64 + b]; prefix += (long)S[j * 64 + b]; }
    out[j] = level + (long)bias[j];
}
kernel void blocks2(device const uint* notes [[buffer(0)]], device const uint* base [[buffer(1)]], device const uint* K [[buffer(2)]], device const int* x [[buffer(3)]],
                    device int* S [[buffer(4)]], device long* LOC [[buffer(5)]], constant uint& nout [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
    uint b = gid / nout, j = gid - b * nout; if (b >= 64) return;             // neighbouring lanes = neighbouring piles, same block: coalesced
    uint k = 0, Kb = K[b]; device const uint* row = notes + base[b] + j;
    int r = 0; long local = 0; int t0 = (int)b * 64;
    uint w = (k < Kb) ? row[0] : 0xFFFFFFFFu;
    for (int t = t0; t < t0 + 64; t++) {
        while (w != 0xFFFFFFFFu && (int)((w >> 12) & 4095) == t) { int i = w & 4095; int sg = ((w >> 28) & 1) ? -1 : 1; r += x[i] * sg; k++; w = (k < Kb) ? row[k * nout] : 0xFFFFFFFFu; }
        local += r; }
    S[j * 64 + b] = r; LOC[j * 64 + b] = local;
}
// NO TICK LOOP. Same lane per (pile, block); it walks its block's edges once. An edge fired with n
// ticks left in the block adds x to the block's integral n times -- done by doubling (shift-adds),
// not by stepping n ticks. Adds and shifts only; the wait is collapsed, not skipped.
kernel void blocks3(device const uint* notes [[buffer(0)]], device const uint* base [[buffer(1)]], device const uint* K [[buffer(2)]], device const int* x [[buffer(3)]],
                    device int* S [[buffer(4)]], device long* LOC [[buffer(5)]], constant uint& nout [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
    uint b = gid / nout, j = gid - b * nout; if (b >= 64) return;
    uint Kb = K[b]; device const uint* row = notes + base[b] + j; int t_end = (int)b * 64 + 64;
    int r = 0; long local = 0;
    for (uint k = 0; k < Kb; k++) {
        uint w = row[k * nout]; if (w == 0xFFFFFFFFu) break;
        int v = x[w & 4095]; if ((w >> 28) & 1) v = -v;
        int n = t_end - (int)((w >> 12) & 4095);                                // ticks this edge is held within the block
        r += v; long acc = 0; long vv = v;                                        // v added n times, by doubling
        for (int bit = 0; bit < 7; bit++) { if (n & (1 << bit)) acc += vv; vv <<= 1; }
        local += acc; }
    S[j * 64 + b] = r; LOC[j * 64 + b] = local;
}
// ONE THREADGROUP PER PILE. 256 lanes share the pile's edge list (sorted, contiguous): lane l takes
// edges l, l+256, l+512, ... so neighbouring lanes read neighbouring words and each lane's chain is
// only a handful of loads. An edge held for n ticks adds x n times, by doubling. The group sums.
kernel void lap4(device const uint* notes [[buffer(0)]], device const uint* off [[buffer(1)]],
                 device const uint* lapnotes [[buffer(2)]], device const uint* lapoff [[buffer(3)]],
                 device const int* x [[buffer(4)]], device const int* bias [[buffer(5)]], device long* out [[buffer(6)]],
                 uint j [[threadgroup_position_in_grid]], uint l [[thread_position_in_threadgroup]]) {
    threadgroup long part[256];
    long local = 0;
    for (uint p = off[j] + l; p < off[j+1]; p += 256) {
        uint w = notes[p]; long v = x[w & 4095]; if ((w >> 28) & 1) v = -v;
        int n = 4096 - (int)((w >> 12) & 4095);                                   // ticks this edge is held: fire tick to the end of the lap
        long acc = 0; for (int bit = 0; bit < 12; bit++) { if (n & (1 << bit)) acc += v; v <<= 1; }   // v added n times, by doubling
        local += acc; }
    for (uint p = lapoff[j] + l; p < lapoff[j+1]; p += 256) {                       // lap edges: held all 4096 ticks
        uint w = lapnotes[p]; long v = x[w & 4095]; if ((w >> 28) & 1) v = -v; int laps = (w >> 24) & 15;
        long acc = 0; for (int bit = 0; bit < 4; bit++) { if (laps & (1 << bit)) acc += v; v <<= 1; }
        local += acc << 12; }
    part[l] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (l == 0) { long s = 0; for (uint i = 0; i < 256; i++) s += part[i]; out[j] = s + (long)bias[j]; }
}
kernel void lap(device const uint* notes [[buffer(0)]], device const uint* off [[buffer(1)]],
                device const uint* lapnotes [[buffer(2)]], device const uint* lapoff [[buffer(3)]],
                device const int* x [[buffer(4)]], device const int* bias [[buffer(5)]], device const float* gauge [[buffer(6)]],
                device long* out [[buffer(7)]], constant uint& nout [[buffer(8)]], uint j [[thread_position_in_grid]]) {
    if (j >= nout) return;
    long level = 0; long rate = 0;
    for (uint p = lapoff[j]; p < lapoff[j+1]; p++) {                       // lap edges: fire at tick 0, held the whole turn
        uint w = lapnotes[p]; int i = w & 4095; int laps = (w >> 24) & 15; int s = ((w >> 28) & 1) ? -1 : 1;
        rate += (long)x[i] * s * laps; }
    uint p = off[j], e = off[j+1];
    for (int t = 0; t < 4096; t++) {                                        // the lap
        while (p < e && (int)((notes[p] >> 12) & 4095) == t) {              // every edge whose phase is now: fire
            uint w = notes[p]; int i = w & 4095; int s = ((w >> 28) & 1) ? -1 : 1; rate += (long)x[i] * s; p++; }
        level += rate;                                                      // the integrator: the only arithmetic
    }
    out[j] = level + (long)bias[j];                                        // the pile: an exact integer with laps; gauged on the CPU
}
"""
setvbuf(stdout, nil, _IONBF, 0)
let dir = "/Users/abundancemachine/tick-machine/gpt2_dial/edges/"
let dev = MTLCreateSystemDefaultDevice()!, q = dev.makeCommandQueue()!
let lib = try! dev.makeLibrary(source: src, options: nil); let ps = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "lap")!)
let psB = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "blocks3")!), psG = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "gather")!), ps4 = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "lap4")!)
func load<T>(_ name: String, _ t: T.Type) -> [T] { let d = try! Data(contentsOf: URL(fileURLWithPath: dir + name)); return d.withUnsafeBytes { Array($0.bindMemory(to: T.self)) } }
func buf<T>(_ a: [T]) -> MTLBuffer { return dev.makeBuffer(bytes: a, length: max(a.count, 1) * MemoryLayout<T>.stride, options: .storageModeShared)! }
struct Bank { let name: String; let nin, nout: Int; let notes, off, lapnotes, lapoff, bias, gauge: MTLBuffer; let uIn, uOut, g: [Double]; let blkoff, notes2, base, K, S, LOC: MTLBuffer; let x, out: MTLBuffer; let edges: Int }
let man = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: dir + "manifest.json"))) as! [String: Any]
var banks = [String: Bank](); var totalEdges = 0
for b in man["banks"] as! [[String: Any]] {
    let n = b["name"] as! String, nin = b["nin"] as! Int, nout = b["nout"] as! Int
    let bk = Bank(name: n, nin: nin, nout: nout, notes: buf(load("\(n).notes.bin", UInt32.self)), off: buf(load("\(n).off.bin", UInt32.self)),
                  lapnotes: buf(load("\(n).lapnotes.bin", UInt32.self)), lapoff: buf(load("\(n).lapoff.bin", UInt32.self)),
                  bias: buf(load("\(n).bias.bin", Int32.self)), gauge: buf(load("\(n).gauge.bin", Float.self)),
                  uIn: load("\(n).u_in.bin", Float.self).map { Double($0) }, uOut: load("\(n).u_out.bin", Float.self).map { Double($0) }, g: load("\(n).gauge.bin", Float.self).map { Double($0) },
                  blkoff: buf(load("\(n).blkoff.bin", UInt32.self)), notes2: buf(load("\(n).notes2.bin", UInt32.self)), base: buf(load("\(n).base.bin", UInt32.self)), K: buf(load("\(n).K.bin", UInt32.self)), S: dev.makeBuffer(length: nout * 64 * 4, options: .storageModePrivate)!, LOC: dev.makeBuffer(length: nout * 64 * 8, options: .storageModePrivate)!,
                  x: dev.makeBuffer(length: nin * 4, options: .storageModeShared)!, out: dev.makeBuffer(length: nout * 8, options: .storageModeShared)!,
                  edges: (b["edges"] as! Int) + (b["lap_edges"] as! Int))
    banks[n] = bk; totalEdges += bk.edges
}
let L = 12, D = 768, NH = 12, HD = 64, V = 50257
let uRes = load("u_res.bin", Float.self).map { Double($0) }
let wte = load("wte.ticks.bin", Int16.self), wpe = load("wpe.ticks.bin", Int16.self)
func ln(_ k: Int, _ n: String) -> [Double] { return load("ln.\(k).\(n).bin", Float.self).map { Double($0) } }
let lnW1 = (0..<L).map { ln($0, "ln_1.weight") }, lnB1 = (0..<L).map { ln($0, "ln_1.bias") }, lnW2 = (0..<L).map { ln($0, "ln_2.weight") }, lnB2 = (0..<L).map { ln($0, "ln_2.bias") }
let lnFW = load("ln_f.weight.bin", Float.self).map { Double($0) }, lnFB = load("ln_f.bias.bin", Float.self).map { Double($0) }
print("edge field on \(dev.name): \(banks.count) banks, \(Double(totalEdges) / 1e6) M edges resident in GPU memory; a lap = 4096 ticks")

var gpuTime = 0.0; var bankTime = [String: Double]()
var gpuExec = 0.0
func runLap(_ bk: Bank, _ xReal: [Double]) -> [Double] {                     // input onto its dial as ticks, one lap, pile out (gauged)
    let xp = bk.x.contents().bindMemory(to: Int32.self, capacity: bk.nin)
    for i in 0..<bk.nin { xp[i] = Int32((xReal[i] / bk.uIn[i]).rounded()) }
    let cb = q.makeCommandBuffer()!; var n = UInt32(bk.nout)
    let e1 = cb.makeComputeCommandEncoder()!; e1.setComputePipelineState(ps4)                       // one threadgroup per pile, 256 lanes share its edges
    for (i, b) in [bk.notes, bk.off, bk.lapnotes, bk.lapoff, bk.x, bk.bias, bk.out].enumerated() { e1.setBuffer(b, offset: 0, index: i) }
    _ = n; e1.dispatchThreadgroups(MTLSize(width: bk.nout, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)); e1.endEncoding()
    let t0 = Date(); cb.commit(); cb.waitUntilCompleted(); let dt = Date().timeIntervalSince(t0); gpuTime += dt; gpuExec += cb.gpuEndTime - cb.gpuStartTime; bankTime[bk.name.split(separator: ".")[0].description, default: 0] += dt
    let op = bk.out.contents().bindMemory(to: Int64.self, capacity: bk.nout)
    return (0..<bk.nout).map { j in let y = Double(op[j]) * bk.g[j]; return bk.uOut[j] > 0 ? (y / bk.uOut[j]).rounded() * bk.uOut[j] : y }   // gauge, then back onto its dial
}
func water(_ x: [Double], _ w: [Double], _ b: [Double]) -> [Double] {
    let m = x.reduce(0, +) / Double(x.count); let v = x.map { ($0 - m) * ($0 - m) }.reduce(0, +) / Double(x.count); let s = (v + 1e-5).squareRoot()
    return (0..<x.count).map { ($0 < x.count) ? (x[$0] - m) / s * w[$0] + b[$0] : 0 } }
func gelu(_ x: [Double]) -> [Double] { return x.map { 0.5 * $0 * (1 + tanh(0.7978845608028654 * ($0 + 0.044715 * $0 * $0 * $0))) } }
var Ks = Array(repeating: [[Double]](), count: L), Vs = Array(repeating: [[Double]](), count: L)
func position(_ pos: Int, _ tid: Int) -> Int {
    var r = (0..<D).map { Int(wte[tid * D + $0]) + Int(wpe[pos * D + $0]) }                    // the residual stream: integer ticks
    for k in 0..<L {
        let qkv = runLap(banks["qkv.\(k)"]!, water((0..<D).map { Double(r[$0]) * uRes[$0] }, lnW1[k], lnB1[k]))
        let qv = Array(qkv[0..<D]), kv = Array(qkv[D..<2*D]), vv = Array(qkv[2*D..<3*D]); Ks[k].append(kv); Vs[k].append(vv)
        var o = [Double](repeating: 0, count: D)
        for h in 0..<NH {                                                                     // resonance with the past, limiter, pull
            var sc = [Double](); for s in 0..<Ks[k].count { var d = 0.0; for i in h*HD..<(h+1)*HD { d += qv[i] * Ks[k][s][i] }; sc.append(d / 8.0) }
            let mx = sc.max()!; let ex = sc.map { exp($0 - mx) }; let z = ex.reduce(0, +)
            for s in 0..<Vs[k].count { for i in h*HD..<(h+1)*HD { o[i] += ex[s] / z * Vs[k][s][i] } } }
        let p = runLap(banks["proj.\(k)"]!, o); for i in 0..<D { r[i] += Int((p[i] / uRes[i]).rounded()) }
        let f = runLap(banks["fc.\(k)"]!, water((0..<D).map { Double(r[$0]) * uRes[$0] }, lnW2[k], lnB2[k]))
        let m = runLap(banks["mproj.\(k)"]!, gelu(f)); for i in 0..<D { r[i] += Int((m[i] / uRes[i]).rounded()) }
    }
    let lg = runLap(banks["head"]!, water((0..<D).map { Double(r[$0]) * uRes[$0] }, lnFW, lnFB))
    var best = 0; for t in 1..<V where lg[t] > lg[best] { best = t }; return best
}
let ref = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: "/Users/abundancemachine/tick-machine/gpt2_dial/reference.json"))) as! [[String: Any]]
let NNEW = 4; var allSame = true; var positions = 0; let tAll = Date(); gpuTime = 0
for c in ref {
    let ids = c["ids"] as! [Int], want = Array((c["ref"] as! [Int]).prefix(NNEW)); Ks = Array(repeating: [], count: L); Vs = Array(repeating: [], count: L)
    var nxt = 0; for (i, t) in ids.enumerated() { nxt = position(i, t); positions += 1 }
    var gen = [Int](); var pos = ids.count
    while gen.count < NNEW { gen.append(nxt); nxt = position(pos, nxt); pos += 1; positions += 1 }
    let same = gen == want; allSame = allSame && same
    print("  \(c["prompt"] as! String) + \(NNEW): \(same ? "IDENTICAL to fp32 greedy" : "differs") tokens \(gen)\(same ? "" : "  (reference \(want))")")
}
let wall = Date().timeIntervalSince(tAll)
for (k, v) in bankTime.sorted(by: { $0.value > $1.value }) { print(String(format: "    %-6@ %6.1f ms per position  (%d banks)", k, v / Double(positions) * 1e3, k == "head" ? 1 : 12)) }
print(String(format: "  GPU-side execution inside the laps: %.1f ms per position (the rest of the lap time is CPU-side overhead)", gpuExec / Double(positions) * 1e3))
print(String(format: "[MEASURED] %@; %d positions; %.1f ms per position wall, of which %.1f ms in the 49 laps on the GPU; %d ticks per position (49 laps x 4096); %.0fM edges fired per position",
             allSame ? "all identical" : "not all identical", positions, wall / Double(positions) * 1e3, gpuTime / Double(positions) * 1e3, 49 * 4096, Double(totalEdges) / 1e6))

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
func load<T>(_ name: String, _ t: T.Type) -> [T] { let d = try! Data(contentsOf: URL(fileURLWithPath: dir + name)); return d.withUnsafeBytes { Array($0.bindMemory(to: T.self)) } }
func buf<T>(_ a: [T]) -> MTLBuffer { return dev.makeBuffer(bytes: a, length: max(a.count, 1) * MemoryLayout<T>.stride, options: .storageModeShared)! }
struct Bank { let name: String; let nin, nout: Int; let notes, off, lapnotes, lapoff, bias, gauge: MTLBuffer; let uIn, uOut, g: [Double]; let x, out: MTLBuffer; let edges: Int }
let man = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: dir + "manifest.json"))) as! [String: Any]
var banks = [String: Bank](); var totalEdges = 0
for b in man["banks"] as! [[String: Any]] {
    let n = b["name"] as! String, nin = b["nin"] as! Int, nout = b["nout"] as! Int
    let bk = Bank(name: n, nin: nin, nout: nout, notes: buf(load("\(n).notes.bin", UInt32.self)), off: buf(load("\(n).off.bin", UInt32.self)),
                  lapnotes: buf(load("\(n).lapnotes.bin", UInt32.self)), lapoff: buf(load("\(n).lapoff.bin", UInt32.self)),
                  bias: buf(load("\(n).bias.bin", Int32.self)), gauge: buf(load("\(n).gauge.bin", Float.self)),
                  uIn: load("\(n).u_in.bin", Float.self).map { Double($0) }, uOut: load("\(n).u_out.bin", Float.self).map { Double($0) }, g: load("\(n).gauge.bin", Float.self).map { Double($0) },
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

var gpuTime = 0.0
func runLap(_ bk: Bank, _ xReal: [Double]) -> [Double] {                     // input onto its dial as ticks, one lap, pile out (gauged)
    let xp = bk.x.contents().bindMemory(to: Int32.self, capacity: bk.nin)
    for i in 0..<bk.nin { xp[i] = Int32((xReal[i] / bk.uIn[i]).rounded()) }
    let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(ps)
    for (i, b) in [bk.notes, bk.off, bk.lapnotes, bk.lapoff, bk.x, bk.bias, bk.gauge, bk.out].enumerated() { e.setBuffer(b, offset: 0, index: i) }
    var n = UInt32(bk.nout); e.setBytes(&n, length: 4, index: 8)
    let tg = min(ps.maxTotalThreadsPerThreadgroup, 256)
    e.dispatchThreads(MTLSize(width: bk.nout, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1)); e.endEncoding()
    let t0 = Date(); cb.commit(); cb.waitUntilCompleted(); gpuTime += Date().timeIntervalSince(t0)
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
print(String(format: "[MEASURED] %@; %d positions; %.1f ms per position wall, of which %.1f ms in the 49 laps on the GPU; %d ticks per position (49 laps x 4096); %.0fM edges fired per position",
             allSame ? "all identical" : "not all identical", positions, wall / Double(positions) * 1e3, gpuTime / Double(positions) * 1e3, 49 * 4096, Double(totalEdges) / 1e6))

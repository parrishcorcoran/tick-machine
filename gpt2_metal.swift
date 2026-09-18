// GPT-2 on the dial, on the Apple GPU. Every linear layer and the head are
// int16 ticks (4096 x 16 laps, per-output scale) read straight off the dial;
// the embeddings are on the dial too. LayerNorm, GELU, attention (KV cache),
// residuals and the head run as Metal kernels. Greedy decoding, batch 1.
// Checked token for token against fp32 HF greedy; timed on generated tokens.
import Foundation
import Metal

let src = """
#include <metal_stdlib>
using namespace metal;
constant int D = 768;

// y[row] = sum_i x[i] * W[row, i] * cyc[row]/32767 + b[row]  (+ resid[row] if given)
kernel void gemv_ticks(device const short* W [[buffer(0)]], device const float* x [[buffer(1)]],
                       device const float* cyc [[buffer(2)]], device const float* b [[buffer(3)]],
                       device float* y [[buffer(4)]], constant int& nin [[buffer(5)]],
                       constant int& addres [[buffer(6)]], device const float* res [[buffer(7)]],
                       uint row [[thread_position_in_grid]]) {
    device const short* w = W + (ulong)row * nin;
    float4 acc = 0;
    for (int i = 0; i < nin; i += 4)
        acc += float4(w[i], w[i+1], w[i+2], w[i+3]) * float4(x[i], x[i+1], x[i+2], x[i+3]);
    float v = (acc.x + acc.y + acc.z + acc.w) * cyc[row] / 32767.0f + b[row];
    y[row] = addres ? v + res[row] : v;
}
kernel void layernorm(device const float* x [[buffer(0)]], device const float* g [[buffer(1)]],
                      device const float* b [[buffer(2)]], device float* y [[buffer(3)]],
                      uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float sh[256];
    float s = 0; for (int i = tid; i < D; i += 256) s += x[i];
    sh[tid] = s; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int o = 128; o > 0; o >>= 1) { if (tid < o) sh[tid] += sh[tid + o]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float mean = sh[0] / D; threadgroup_barrier(mem_flags::mem_threadgroup);
    float v = 0; for (int i = tid; i < D; i += 256) { float d = x[i] - mean; v += d * d; }
    sh[tid] = v; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int o = 128; o > 0; o >>= 1) { if (tid < o) sh[tid] += sh[tid + o]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float inv = rsqrt(sh[0] / D + 1e-5f);
    for (int i = tid; i < D; i += 256) y[i] = (x[i] - mean) * inv * g[i] + b[i];
}
kernel void gelu(device float* x [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    float v = x[i]; float a = clamp(0.7978845608f * (v + 0.044715f * v * v * v), -10.0f, 10.0f);   // fast tanh is inf/inf past ~44
    x[i] = 0.5f * v * (1.0f + tanh(a));
}
// one thread per head: attention of the new position over the cache (positions 0..pos)
kernel void attend(device const float* qkv [[buffer(0)]], device float* K [[buffer(1)]], device float* V [[buffer(2)]],
                   device float* out [[buffer(3)]], constant int& pos [[buffer(4)]],
                   uint h [[thread_position_in_grid]]) {
    const int HD = 64;
    // write this position's k, v into the cache
    for (int d = 0; d < HD; d++) { K[(ulong)pos * D + h * HD + d] = qkv[D + h * HD + d]; V[(ulong)pos * D + h * HD + d] = qkv[2 * D + h * HD + d]; }
    float sc[1024]; float best = -1e30f;
    for (int p = 0; p <= pos; p++) { float s = 0; for (int d = 0; d < HD; d++) s += qkv[h * HD + d] * K[(ulong)p * D + h * HD + d]; s *= 0.125f; sc[p] = s; best = max(best, s); }
    float tot = 0; for (int p = 0; p <= pos; p++) { sc[p] = exp(sc[p] - best); tot += sc[p]; }
    for (int d = 0; d < HD; d++) { float a = 0; for (int p = 0; p <= pos; p++) a += sc[p] * V[(ulong)p * D + h * HD + d]; out[h * HD + d] = a / tot; }
}
kernel void embed(device const float* wte [[buffer(0)]], device const float* wpe [[buffer(1)]], device float* x [[buffer(2)]],
                  constant int& tokid [[buffer(3)]], constant int& pos [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    x[i] = wte[(ulong)tokid * D + i] + wpe[(ulong)pos * D + i];
}
"""

let D = 768, V = 50257, L = 12, MAXPOS = 1024
let dev = MTLCreateSystemDefaultDevice()!
let lib = try! dev.makeLibrary(source: src, options: nil)
let q = dev.makeCommandQueue()!
func pipe(_ n: String) -> MTLComputePipelineState { try! dev.makeComputePipelineState(function: lib.makeFunction(name: n)!) }
let pGemv = pipe("gemv_ticks"), pLN = pipe("layernorm"), pGelu = pipe("gelu"), pAtt = pipe("attend"), pEmb = pipe("embed")

let dir = "/Users/abundancemachine/tick-machine/gpt2_dial/"
func load(_ name: String) -> MTLBuffer {
    let d = try! Data(contentsOf: URL(fileURLWithPath: dir + name))
    return d.withUnsafeBytes { dev.makeBuffer(bytes: $0.baseAddress!, length: d.count, options: .storageModeShared)! }
}
func fbuf(_ n: Int) -> MTLBuffer { dev.makeBuffer(length: n * 4, options: .storageModeShared)! }
struct Lin { let w, cyc, b: MTLBuffer; let nin, nout: Int }
func lin(_ p: String, _ nin: Int, _ nout: Int) -> Lin { Lin(w: load(p + "_w.i16"), cyc: load(p + "_cyc.f32"), b: load(p + "_b.f32"), nin: nin, nout: nout) }
struct Block { let ln1g, ln1b, ln2g, ln2b: MTLBuffer; let attn, proj, fc, mproj: Lin; let K, Vc: MTLBuffer }
let blocks: [Block] = (0..<L).map { k in
    Block(ln1g: load("h\(k)_ln1_g.f32"), ln1b: load("h\(k)_ln1_b.f32"), ln2g: load("h\(k)_ln2_g.f32"), ln2b: load("h\(k)_ln2_b.f32"),
          attn: lin("h\(k)_attn", D, 3 * D), proj: lin("h\(k)_proj", D, D), fc: lin("h\(k)_fc", D, 4 * D), mproj: lin("h\(k)_mproj", 4 * D, D),
          K: fbuf(MAXPOS * D), Vc: fbuf(MAXPOS * D))
}
let lnfG = load("lnf_g.f32"), lnfB = load("lnf_b.f32"), wte = load("wte.f32"), wpe = load("wpe.f32")
let headW = load("head_w.i16"), headCyc = load("head_cyc.f32"), zeroB = fbuf(V)
let x = fbuf(D), h1 = fbuf(D), qkv = fbuf(3 * D), att = fbuf(D), x2 = fbuf(D), h2 = fbuf(D), ff = fbuf(4 * D), logits = fbuf(V)
var ticks = 0
let weightBytes = blocks.reduce(0) { $0 + ($1.attn.w.length + $1.proj.w.length + $1.fc.w.length + $1.mproj.w.length) } + headW.length

func gemv(_ e: MTLComputeCommandEncoder, _ l: Lin, _ xin: MTLBuffer, _ y: MTLBuffer, res: MTLBuffer?, b: MTLBuffer? = nil, cyc: MTLBuffer? = nil) {
    e.setComputePipelineState(pGemv)
    e.setBuffer(l.w, offset: 0, index: 0); e.setBuffer(xin, offset: 0, index: 1); e.setBuffer(cyc ?? l.cyc, offset: 0, index: 2)
    e.setBuffer(b ?? l.b, offset: 0, index: 3); e.setBuffer(y, offset: 0, index: 4)
    var nin = Int32(l.nin), add = Int32(res == nil ? 0 : 1)
    e.setBytes(&nin, length: 4, index: 5); e.setBytes(&add, length: 4, index: 6); e.setBuffer(res ?? y, offset: 0, index: 7)
    e.dispatchThreads(MTLSize(width: l.nout, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    e.memoryBarrier(scope: .buffers)
}
func ln(_ e: MTLComputeCommandEncoder, _ xin: MTLBuffer, _ g: MTLBuffer, _ b: MTLBuffer, _ y: MTLBuffer) {
    e.setComputePipelineState(pLN); e.setBuffer(xin, offset: 0, index: 0); e.setBuffer(g, offset: 0, index: 1); e.setBuffer(b, offset: 0, index: 2); e.setBuffer(y, offset: 0, index: 3)
    e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    e.memoryBarrier(scope: .buffers)
}

func run1(_ f: (MTLComputeCommandEncoder) -> Void) { let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!; f(e); e.endEncoding(); cb.commit(); cb.waitUntilCompleted() }
func debugBlocks(_ tokid: Int, _ pos: Int) -> Int {
    var t = Int32(tokid), p = Int32(pos)
    run1 { e in e.setComputePipelineState(pEmb); e.setBuffer(wte, offset: 0, index: 0); e.setBuffer(wpe, offset: 0, index: 1); e.setBuffer(x, offset: 0, index: 2); e.setBytes(&t, length: 4, index: 3); e.setBytes(&p, length: 4, index: 4); e.dispatchThreads(MTLSize(width: D, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)) }
    for (bi, blk) in blocks.enumerated() {
        run1 { e in ln(e, x, blk.ln1g, blk.ln1b, h1) }
        run1 { e in gemv(e, blk.attn, h1, qkv, res: nil) }
        run1 { e in e.setComputePipelineState(pAtt); e.setBuffer(qkv, offset: 0, index: 0); e.setBuffer(blk.K, offset: 0, index: 1); e.setBuffer(blk.Vc, offset: 0, index: 2); e.setBuffer(att, offset: 0, index: 3); e.setBytes(&p, length: 4, index: 4); e.dispatchThreads(MTLSize(width: 12, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 12, height: 1, depth: 1)) }
        run1 { e in gemv(e, blk.proj, att, x2, res: x) }
        run1 { e in ln(e, x2, blk.ln2g, blk.ln2b, h2) }
        run1 { e in gemv(e, blk.fc, h2, ff, res: nil) }
        run1 { e in e.setComputePipelineState(pGelu); e.setBuffer(ff, offset: 0, index: 0); e.dispatchThreads(MTLSize(width: 4 * D, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)) }
        run1 { e in gemv(e, blk.mproj, ff, x, res: x2) }
        dump(x, D, "blk\(bi)_x.f32"); dump(att, D, "blk\(bi)_att.f32"); dump(qkv, 3 * D, "blk\(bi)_qkv.f32")
    }
    return 0
}
func debugBlock0(_ tokid: Int, _ pos: Int) -> Int {
    let blk = blocks[0]; var p = Int32(pos)
    run1 { e in ln(e, x, blk.ln1g, blk.ln1b, h1) }; dump(h1, D, "d1_ln1.f32")
    run1 { e in gemv(e, blk.attn, h1, qkv, res: nil) }; dump(qkv, 3 * D, "d2_qkv.f32")
    run1 { e in e.setComputePipelineState(pAtt); e.setBuffer(qkv, offset: 0, index: 0); e.setBuffer(blk.K, offset: 0, index: 1); e.setBuffer(blk.Vc, offset: 0, index: 2); e.setBuffer(att, offset: 0, index: 3); e.setBytes(&p, length: 4, index: 4); e.dispatchThreads(MTLSize(width: 12, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 12, height: 1, depth: 1)) }; dump(att, D, "d3_att.f32")
    run1 { e in gemv(e, blk.proj, att, x2, res: x) }; dump(x2, D, "d4_x2.f32")
    run1 { e in ln(e, x2, blk.ln2g, blk.ln2b, h2) }; dump(h2, D, "d5_ln2.f32")
    run1 { e in gemv(e, blk.fc, h2, ff, res: nil) }; dump(ff, 4 * D, "d6_fc.f32")
    run1 { e in e.setComputePipelineState(pGelu); e.setBuffer(ff, offset: 0, index: 0); e.dispatchThreads(MTLSize(width: 4 * D, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)) }; dump(ff, 4 * D, "d7_gelu.f32")
    run1 { e in gemv(e, blk.mproj, ff, x, res: x2) }; dump(x, D, "d8_x.f32")
    return 0
}

/// one token through the whole model at position pos; returns the argmax of the logits
var dumpDir: String? = nil
func dump(_ b: MTLBuffer, _ n: Int, _ name: String) {
    guard let d = dumpDir else { return }
    let data = Data(bytes: b.contents(), count: n * 4); try! data.write(to: URL(fileURLWithPath: d + name))
}
func step(_ tokid: Int, _ pos: Int) -> Int {
    let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
    var t = Int32(tokid), p = Int32(pos)
    e.setComputePipelineState(pEmb); e.setBuffer(wte, offset: 0, index: 0); e.setBuffer(wpe, offset: 0, index: 1); e.setBuffer(x, offset: 0, index: 2)
    e.setBytes(&t, length: 4, index: 3); e.setBytes(&p, length: 4, index: 4)
    e.dispatchThreads(MTLSize(width: D, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    e.memoryBarrier(scope: .buffers)
    for (bi, blk) in blocks.enumerated() {
        if false { e.endEncoding(); cb.commit(); cb.waitUntilCompleted(); dump(x, D, "d0_embed.f32"); return debugBlock0(tokid, pos) }
        ln(e, x, blk.ln1g, blk.ln1b, h1)
        gemv(e, blk.attn, h1, qkv, res: nil)
        e.setComputePipelineState(pAtt); e.setBuffer(qkv, offset: 0, index: 0); e.setBuffer(blk.K, offset: 0, index: 1); e.setBuffer(blk.Vc, offset: 0, index: 2)
        e.setBuffer(att, offset: 0, index: 3); e.setBytes(&p, length: 4, index: 4)
        e.dispatchThreads(MTLSize(width: 12, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 12, height: 1, depth: 1))
        e.memoryBarrier(scope: .buffers)
        gemv(e, blk.proj, att, x2, res: x)                       // x2 = x + proj(att)
        ln(e, x2, blk.ln2g, blk.ln2b, h2)
        gemv(e, blk.fc, h2, ff, res: nil)
        e.setComputePipelineState(pGelu); e.setBuffer(ff, offset: 0, index: 0)
        e.dispatchThreads(MTLSize(width: 4 * D, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        e.memoryBarrier(scope: .buffers)
        gemv(e, blk.mproj, ff, x, res: x2)                       // x = x2 + mproj(gelu(ff))
    }
    ln(e, x, lnfG, lnfB, h1)
    gemv(e, Lin(w: headW, cyc: headCyc, b: zeroB, nin: D, nout: V), h1, logits, res: nil)
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    dump(x, D, "x_final.f32"); dump(logits, V, "logits.f32"); dump(qkv, 3 * D, "qkv_last.f32"); dump(h1, D, "h1_final.f32")
    let lg = logits.contents().bindMemory(to: Float.self, capacity: V)
    var best = 0; for i in 1..<V where lg[i] > lg[best] { best = i }
    return best
}

struct Case: Decodable { let prompt: String; let ids: [Int]; let ref: [Int] }
let cases = try! JSONDecoder().decode([Case].self, from: Data(contentsOf: URL(fileURLWithPath: dir + "reference.json")))
print("GPT-2 on the dial, Apple GPU (\(dev.name)): \(weightBytes / 1_000_000) MB of int16 ticks per token, 4096 x 16 laps, batch 1")
var allSame = true; var totalGen = 0.0; var totalTime = 0.0
for (ci, c) in cases.enumerated() {
    var pos = 0; var next = 0
    for (ti, t) in c.ids.enumerated() { dumpDir = (ci == 0 && ti == 0) ? dir : nil; next = step(t, pos); pos += 1 }
    dumpDir = nil
    var gen: [Int] = []; let t0 = Date()
    for _ in 0..<c.ref.count { gen.append(next); next = step(next, pos); pos += 1 }
    let dt = Date().timeIntervalSince(t0); totalGen += Double(gen.count); totalTime += dt
    let same = gen == c.ref; allSame = allSame && same
    let firstDiff = (0..<min(gen.count, c.ref.count)).first { gen[$0] != c.ref[$0] }
    print(String(format: "  %-28@ %@  %5.1f tok/s%@", c.prompt as NSString, same ? "IDENTICAL to fp32 (32 tokens)" : "differs", Double(gen.count) / dt, same ? "" : "   first differing token \(firstDiff ?? -1)"))
}
print(String(format: "  [MEASURED] %@   %.1f tok/s over %.0f generated tokens   (%.1f GB/s of weights)", allSame ? "all prompts identical to fp32" : "not all identical", totalGen / totalTime, totalGen, Double(weightBytes) * totalGen / totalTime / 1e9))

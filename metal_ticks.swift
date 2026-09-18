// Metal kernels for the dial machine, batch 1, GPT-2-sized (110,592 rows x 768).
// Three questions the CPU could not settle:
//   1. ticks16 : int16 ticks read directly by the GPU, no matrix ever materialised.
//   2. ticks12 : 12-bit ticks packed 2 per 3 bytes. 1.5 bytes/weight.
//   3. steps4  : sorted rows as 4-bit steps (15 = escape) + per-row order; the GPU decodes.
// Each kernel: one thread per output row, accumulate over 768 inputs, y[row] = acc * cyc[row].
// Reported: ms per full pass and the tokens/s ceiling that implies.
import Foundation
import Metal

let src = """
#include <metal_stdlib>
using namespace metal;
constant int NIN = 768;

kernel void ticks16(device const short* W [[buffer(0)]], device const float* x [[buffer(1)]],
                    device const float* cyc [[buffer(2)]], device float* y [[buffer(3)]],
                    uint row [[thread_position_in_grid]]) {
    device const short* w = W + (ulong)row * NIN;
    float4 acc = 0;
    for (int i = 0; i < NIN; i += 4) {
        float4 wv = float4(w[i], w[i+1], w[i+2], w[i+3]);   // the dial's face is linear: tick/half
        acc += wv * float4(x[i], x[i+1], x[i+2], x[i+3]);
    }
    y[row] = (acc.x + acc.y + acc.z + acc.w) * cyc[row] / 32767.0f;
}

kernel void ticks12(device const uchar* P [[buffer(0)]], device const float* x [[buffer(1)]],
                    device const float* cyc [[buffer(2)]], device float* y [[buffer(3)]],
                    uint row [[thread_position_in_grid]]) {
    device const uchar* p = P + (ulong)row * (NIN * 3 / 2);
    float acc = 0;
    for (int i = 0; i < NIN; i += 2) {
        uint b0 = p[0], b1 = p[1], b2 = p[2]; p += 3;
        int a = int((b0 | ((b1 & 0xF) << 8)) << 20) >> 20;         // 12-bit signed, sign-extended
        int b = int(((b1 >> 4) | (b2 << 4)) << 20) >> 20;
        acc += float(a) * x[i] + float(b) * x[i+1];
    }
    y[row] = acc * cyc[row] / 2047.0f;
}

kernel void steps4(device const uchar* C [[buffer(0)]], device const uint* off [[buffer(1)]],
                   device const short* perm [[buffer(2)]], device const float* x [[buffer(3)]],
                   device const float* cyc [[buffer(4)]], device float* y [[buffer(5)]],
                   uint row [[thread_position_in_grid]]) {
    device const uchar* c = C + off[row];
    device const short* pm = perm + (ulong)row * NIN;
    int v = -2047; float acc = 0; uint bit = 0;
    for (int i = 0; i < NIN; i++) {
        uint nib = (c[bit >> 3] >> (bit & 4)) & 15; bit += 4;
        if (nib == 15) {
            uint big = (c[bit >> 3] >> (bit & 4)) & 15; bit += 4;
            big |= ((c[bit >> 3] >> (bit & 4)) & 15) << 4; bit += 4;
            big |= ((c[bit >> 3] >> (bit & 4)) & 15) << 8; bit += 4;
            v += int(big);
        } else v += int(nib);
        acc += float(v) * x[pm[i]];
    }
    y[row] = acc * cyc[row] / 2047.0f;
}
"""

let NIN = 768, ROWS = 110_592
let dev = MTLCreateSystemDefaultDevice()!
let lib = try! dev.makeLibrary(source: src, options: nil)
let q = dev.makeCommandQueue()!
func pipe(_ n: String) -> MTLComputePipelineState { try! dev.makeComputePipelineState(function: lib.makeFunction(name: n)!) }

// ---- data with GPT-2's statistics: gaussian rows, per-row scale, sorted steps
var rng = SystemRandomNumberGenerator()
func gauss() -> Float { let u = Float.random(in: 1e-7...1, using: &rng), v = Float.random(in: 0...1, using: &rng); return sqrt(-2*log(u))*cos(2*Float.pi*v) }
let x = (0..<NIN).map { _ in Float.random(in: -0.5...0.5, using: &rng) }
var w16 = [Int16](repeating: 0, count: ROWS*NIN)
var p12 = [UInt8](repeating: 0, count: ROWS*NIN*3/2)
var code = [UInt8](repeating: 0, count: ROWS*NIN)      // upper bound
var off = [UInt32](repeating: 0, count: ROWS+1)
var perm = [Int16](repeating: 0, count: ROWS*NIN)
var cyc = [Float](repeating: 1, count: ROWS)
var bit = 0
var row = [Float](repeating: 0, count: NIN)
for r in 0..<ROWS {
    var mx: Float = 0
    for i in 0..<NIN { row[i] = gauss(); mx = max(mx, abs(row[i])) }
    cyc[r] = mx * 0.02
    var q12 = [Int](repeating: 0, count: NIN)
    for i in 0..<NIN { q12[i] = Int((row[i]/mx*2047).rounded()); w16[r*NIN+i] = Int16(q12[i]*16) }
    for i in stride(from: 0, to: NIN, by: 2) {
        let a = UInt32(bitPattern: Int32(q12[i])) & 0xFFF, b = UInt32(bitPattern: Int32(q12[i+1])) & 0xFFF
        let base = r*NIN*3/2 + i*3/2
        p12[base] = UInt8(a & 0xFF); p12[base+1] = UInt8((a >> 8) | ((b & 0xF) << 4)); p12[base+2] = UInt8(b >> 4)
    }
    off[r] = UInt32(bit >> 3)
    let ord = (0..<NIN).sorted { q12[$0] < q12[$1] }
    var prev = -2047
    func putNib(_ n: Int) { if bit & 4 != 0 { code[bit>>3] = (code[bit>>3] & 0x0F) | UInt8(n << 4) } else { code[bit>>3] = UInt8(n) }; bit += 4 }
    for i in 0..<NIN {
        let s = q12[ord[i]] - prev; prev = q12[ord[i]]; perm[r*NIN+i] = Int16(ord[i])
        if s < 15 { putNib(s) } else { putNib(15); putNib(s & 15); putNib((s >> 4) & 15); putNib((s >> 8) & 15) }
    }
}
off[ROWS] = UInt32(bit >> 3)
let bitsPerWeight = Double(bit) / Double(ROWS*NIN)

func buf<T>(_ a: [T]) -> MTLBuffer { a.withUnsafeBytes { dev.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! } }
let bW16 = buf(w16), bP12 = buf(p12), bCode = buf(Array(code[0..<(bit>>3)+16])), bOff = buf(off), bPerm = buf(perm)
let bX = buf(x), bCyc = buf(cyc), bY = dev.makeBuffer(length: ROWS*4, options: .storageModeShared)!

func time(_ name: String, _ ps: MTLComputePipelineState, _ bufs: [MTLBuffer], _ mb: Double) {
    var best = 1e9
    for _ in 0..<8 {
        let cb = q.makeCommandBuffer()!, enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(ps)
        for (i, b) in bufs.enumerated() { enc.setBuffer(b, offset: 0, index: i) }
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        enc.dispatchThreads(MTLSize(width: ROWS, height: 1, depth: 1), threadsPerThreadgroup: tg)
        enc.endEncoding()
        let t0 = Date(); cb.commit(); cb.waitUntilCompleted(); let d = Date().timeIntervalSince(t0)
        best = min(best, d)
    }
    print(String(format: "  %-8@ %7.2f ms/pass   %6.1f MB moved  %6.1f GB/s   -> %5.0f tok/s ceiling", name as NSString, best*1e3, mb, mb/1e3/best, 1/best))
}
// correctness: compare ticks12 and steps4 against ticks16 on a few rows
func ref(_ r: Int) -> Float { var a: Float = 0; for i in 0..<NIN { a += Float(w16[r*NIN+i]) * x[i] }; return a * cyc[r] / 32767 }
print("GPT-2-sized batch-1 pass on the Apple GPU (\(dev.name)), \(ROWS) rows x \(NIN); sorted steps at \(String(format: "%.2f", bitsPerWeight)) bits/weight")
time("ticks16", pipe("ticks16"), [bW16, bX, bCyc, bY], Double(ROWS*NIN*2)/1e6)
let y16 = bY.contents().bindMemory(to: Float.self, capacity: ROWS)
let e16 = (0..<8).map { abs(y16[$0*1000] - ref($0*1000)) }.max()!
time("ticks12", pipe("ticks12"), [bP12, bX, bCyc, bY], Double(ROWS*NIN*3/2)/1e6)
let y12 = bY.contents().bindMemory(to: Float.self, capacity: ROWS)
let e12 = (0..<8).map { abs(y12[$0*1000] - ref($0*1000)) }.max()!
time("steps4", pipe("steps4"), [bCode, bOff, bPerm, bX, bCyc, bY], Double(bit/8 + ROWS*NIN*2)/1e6)
let ys = bY.contents().bindMemory(to: Float.self, capacity: ROWS)
let es = (0..<8).map { abs(ys[$0*1000] - ref($0*1000)) }.max()!
print(String(format: "  agreement with the reference (max abs diff on sampled rows): ticks16 %.2e  ticks12 %.2e  steps4 %.2e", e16, e12, es))

// Weight-stationary on this chip: the same int16-tick GEMV pass, over weight
// sets of different sizes, repeated so that anything that fits on-chip stays
// there. Effective GB/s far above RAM's ~100 means the weights were resident.
import Foundation
import Metal
let src = """
#include <metal_stdlib>
using namespace metal;
kernel void ticks16(device const short* W [[buffer(0)]], device const float* x [[buffer(1)]], device float* y [[buffer(2)]],
                    constant int& nin [[buffer(3)]], uint row [[thread_position_in_grid]]) {
    device const short* w = W + (ulong)row * nin; float4 acc = 0;
    for (int i = 0; i < nin; i += 4) acc += float4(w[i], w[i+1], w[i+2], w[i+3]) * float4(x[i], x[i+1], x[i+2], x[i+3]);
    y[row] = (acc.x + acc.y + acc.z + acc.w) / 32767.0f;
}
"""
let dev = MTLCreateSystemDefaultDevice()!, q = dev.makeCommandQueue()!
let ps = try! dev.makeComputePipelineState(function: try! dev.makeLibrary(source: src, options: nil).makeFunction(name: "ticks16")!)
let NIN = 768
let x = dev.makeBuffer(length: NIN * 4, options: .storageModeShared)!
print("int16-tick weight pass on \(dev.name), batch 1, repeated 40x back to back (resident if it fits):")
for mb in [1, 4, 8, 16, 32, 64, 128, 247] {
    let rows = mb * 1_000_000 / (NIN * 2)
    let W = dev.makeBuffer(length: rows * NIN * 2, options: .storageModeShared)!, y = dev.makeBuffer(length: rows * 4, options: .storageModeShared)!
    var nin = Int32(NIN)
    let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(ps); e.setBuffer(W, offset: 0, index: 0); e.setBuffer(x, offset: 0, index: 1); e.setBuffer(y, offset: 0, index: 2); e.setBytes(&nin, length: 4, index: 3)
    for _ in 0..<40 { e.dispatchThreads(MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)) }
    e.endEncoding(); let t0 = Date(); cb.commit(); cb.waitUntilCompleted(); let d = Date().timeIntervalSince(t0) / 40
    let gbs = Double(rows * NIN * 2) / d / 1e9
    print(String(format: "  %4d MB of weights: %7.3f ms/pass  %7.1f GB/s effective   -> %6.0f passes/s", mb, d * 1e3, gbs, 1 / d))
}

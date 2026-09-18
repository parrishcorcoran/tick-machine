// Is on-chip memory faster than RAM for the weights? Keep a slice on the GPU
// and re-read it R times INSIDE one dispatch (no re-dispatch floor). If the
// slice stays on-chip, effective GB/s climbs well above RAM's ~100.
import Foundation
import Metal
let src = """
#include <metal_stdlib>
using namespace metal;
kernel void reread(device const short* W [[buffer(0)]], device const float* x [[buffer(1)]], device float* y [[buffer(2)]],
                   constant int& nin [[buffer(3)]], constant int& reps [[buffer(4)]], uint row [[thread_position_in_grid]]) {
    device const short* w = W + (ulong)row * nin; float acc = 0;
    for (int r = 0; r < reps; r++) { float4 a = 0; float xr = x[r & 511];
        for (int i = 0; i < nin; i += 4) a += float4(w[i], w[i+1], w[i+2], w[i+3]) * float4(x[i], x[i+1], x[i+2], x[i+3]) * xr;
        acc += a.x + a.y + a.z + a.w; }
    y[row] = acc / 32767.0f;
}
"""
let dev = MTLCreateSystemDefaultDevice()!, q = dev.makeCommandQueue()!
let ps = try! dev.makeComputePipelineState(function: try! dev.makeLibrary(source: src, options: nil).makeFunction(name: "reread")!)
let NIN = 768; let x = dev.makeBuffer(length: NIN * 4, options: .storageModeShared)!
print("weights re-read inside one dispatch on \(dev.name): effective GB/s (above ~100 = came from on-chip memory)")
for mb in [1, 2, 4, 8, 16, 32, 64] {
    let rows = mb * 1_000_000 / (NIN * 2); let W = dev.makeBuffer(length: rows * NIN * 2, options: .storageModeShared)!, y = dev.makeBuffer(length: rows * 4, options: .storageModeShared)!
    let reps = max(4, 512 / mb); var nin = Int32(NIN), r = Int32(reps)
    let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(ps); e.setBuffer(W, offset: 0, index: 0); e.setBuffer(x, offset: 0, index: 1); e.setBuffer(y, offset: 0, index: 2); e.setBytes(&nin, length: 4, index: 3); e.setBytes(&r, length: 4, index: 4)
    e.dispatchThreads(MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)); e.endEncoding()
    let t0 = Date(); cb.commit(); cb.waitUntilCompleted(); let d = Date().timeIntervalSince(t0)
    print(String(format: "  %3d MB slice, re-read %3dx: %7.2f ms  -> %7.1f GB/s effective", mb, reps, d * 1e3, Double(rows * NIN * 2) * Double(reps) / d / 1e9))
}

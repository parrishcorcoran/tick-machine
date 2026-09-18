// PAIRS INSIDE THE GPU. Each threadgroup is one node pair: a STORAGE node (SIMD group 0)
// that keeps K notes per lane in its registers, and a COMPUTE node (SIMD group 1) that
// adds. The wire between them is that threadgroup's own threadgroup memory: nobody
// else's. Per token the storage node fires all its notes over the wire, the compute
// node adds note*x and sends the sum back over the reverse wire. Hundreds of pairs run
// side by side, each on its own wire. Control: storage reads its notes from RAM each lap.
import Foundation
import Metal
func build(_ K: Int) -> String { return """
#include <metal_stdlib>
using namespace metal;
#define K \(K)
#define NIN 768
kernel void pair(device const short* W [[buffer(0)]], device const float* xs [[buffer(1)]], device float* out [[buffer(2)]],
                 constant int& T [[buffer(3)]], constant int& NX [[buffer(4)]], device const int* offs [[buffer(5)]], constant int& stream [[buffer(6)]],
                 uint tid [[thread_index_in_threadgroup]], uint gid [[threadgroup_position_in_grid]],
                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup short wire[32 * K];          // storage -> compute, this pair's own
    threadgroup float back[1];               // compute -> storage
    threadgroup float x[NIN];
    device const short* mine = W + (ulong)gid * 32 * K + lane;   // this pair's notes: lane l holds mine[32*k]
    short w[K];
    if (sg == 0) {
#pragma clang loop unroll(full)
        for (int k = 0; k < K; k++) w[k] = mine[32 * k]; }      // read once, kept
    float acc = 0;
    for (int t = 0; t < T; t++) {
        for (uint i = tid; i < NIN; i += 64) x[i] = xs[(t % NX) * NIN + i];
        if (sg == 0) {                                            // STORAGE fires its notes over the wire
            if (stream) {
                int o = offs[t];
#pragma clang loop unroll(full)
                for (int k = 0; k < K; k++) wire[32 * k + lane] = mine[32 * k + o]; }   // ... or streams them from RAM
            else {
#pragma clang loop unroll(full)
                for (int k = 0; k < K; k++) wire[32 * k + lane] = w[k]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 1) {                                            // COMPUTE adds what came over the wire
            float a = 0;
#pragma clang loop unroll(full)
            for (int k = 0; k < K; k++) a += float(wire[32 * k + lane]) * x[(32 * k + lane) % NIN];
            a = simd_sum(a); if (lane == 0) back[0] = a; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0 && lane == 0) acc += back[0];                 // STORAGE receives the sum back
    }
    if (tid == 0) out[gid] = acc;
}
"""}
setvbuf(stdout, nil, _IONBF, 0)
let dev = MTLCreateSystemDefaultDevice()!, q = dev.makeCommandQueue()!
let NIN = 768, NX = 64, T = 1000, maxPairs = 8192
let xs = dev.makeBuffer(length: NX * NIN * 4, options: .storageModeShared)!
let xp = xs.contents().bindMemory(to: Float.self, capacity: NX * NIN); for i in 0..<(NX * NIN) { xp[i] = Float((i * 31) % 97) / 97 }
let out = dev.makeBuffer(length: maxPairs * 4, options: .storageModeShared)!
let offs = dev.makeBuffer(length: T * 4, options: .storageModeShared)!; memset(offs.contents(), 0, T * 4)
print("\(dev.name): node pairs inside the GPU, each with its own wire (threadgroup memory) and its own notes (registers)")
for K in [32, 64, 128] {
    let lib: MTLLibrary
    do { lib = try dev.makeLibrary(source: build(K), options: nil) } catch { print("K=\(K) compile:", error); continue }
    let ps = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "pair")!)
    let W = dev.makeBuffer(length: maxPairs * 32 * K * 2 + 4096, options: .storageModeShared)!
    let wp = W.contents().bindMemory(to: Int16.self, capacity: maxPairs * 32 * K); for i in 0..<(maxPairs * 32 * K) { wp[i] = Int16(truncatingIfNeeded: i &* 7919) }
    print("\n  K=\(K): each pair keeps \(32 * K) notes (\(32 * K * 2) B); max threads/threadgroup \(ps.maxTotalThreadsPerThreadgroup)")
    print("   pairs |  notes kept in the node   |  notes streamed from RAM   | wires in flight")
    for pairs in [64, 256, 1024, 4096, 8192] {
        var rate = [Double]()
        for stream in [0, 1] {
            var t = Int32(T), nx = Int32(NX), st = Int32(stream); var d = 0.0
            for rep in 0..<2 {
                let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
                e.setComputePipelineState(ps); e.setBuffer(W, offset: 0, index: 0); e.setBuffer(xs, offset: 0, index: 1); e.setBuffer(out, offset: 0, index: 2)
                e.setBytes(&t, length: 4, index: 3); e.setBytes(&nx, length: 4, index: 4); e.setBuffer(offs, offset: 0, index: 5); e.setBytes(&st, length: 4, index: 6)
                e.dispatchThreadgroups(MTLSize(width: pairs, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)); e.endEncoding()
                let t0 = Date(); cb.commit(); cb.waitUntilCompleted(); if rep == 1 { d = Date().timeIntervalSince(t0) }
            }
            rate.append(Double(pairs * 32 * K) * Double(T) / d)
        }
        print(String(format: "   %5d | %7.1f G notes/s (%5.0f GB/s) | %7.1f G notes/s (%5.0f GB/s) | %d x %d B kept = %.1f MB", pairs, rate[0] / 1e9, rate[0] * 2 / 1e9, rate[1] / 1e9, rate[1] * 2 / 1e9, pairs, 32 * K * 2, Double(pairs * 32 * K * 2) / 1e6))
    }
}

// GPU CYCLERS WITH THEIR OWN NOTES. One cycler = one SIMD group (32 lanes). Each
// cycler loads ~4096 notes (5 neurons x 768, int16 ticks) into its REGISTERS once,
// then laps them for T tokens: for every token it reads the 768 activations (shared
// by all cyclers of the threadgroup) and puts out 5 sums. The notes never touch
// memory again. Control: the same cycler re-reading its notes from RAM every lap.
import Foundation
import Metal
let src = """
#include <metal_stdlib>
using namespace metal;
#define NIN 768
#define NEUR 5
#define K 24
kernel void lap_resident(device const short* W [[buffer(0)]], device const float* xs [[buffer(1)]], device float* out [[buffer(2)]],
                         constant int& T [[buffer(3)]], constant int& NX [[buffer(4)]],
                         uint tid [[thread_index_in_threadgroup]], uint gid [[threadgroup_position_in_grid]], uint tpg [[threads_per_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float x[NIN];
    uint c = gid * (tpg / 32) + sg;                       // this cycler
    short w[NEUR][K];                                     // its own notes, in registers
#pragma unroll
    for (int n = 0; n < NEUR; n++)
#pragma unroll
        for (int k = 0; k < K; k++) w[n][k] = W[((ulong)c * NEUR + n) * NIN + lane + 32 * k];
    float acc[NEUR] = {0, 0, 0, 0, 0};
    for (int t = 0; t < T; t++) {
        device const float* xv = xs + (t % NX) * NIN;
        for (uint i = tid; i < NIN; i += tpg) x[i] = xv[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
        for (int n = 0; n < NEUR; n++) { float a = 0;
#pragma unroll
            for (int k = 0; k < K; k++) a += float(w[n][k]) * x[lane + 32 * k];
            acc[n] += simd_sum(a); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) for (int n = 0; n < NEUR; n++) out[c * NEUR + n] = acc[n];
}
kernel void lap_ram(device const short* W [[buffer(0)]], device const float* xs [[buffer(1)]], device float* out [[buffer(2)]],
                    constant int& T [[buffer(3)]], constant int& NX [[buffer(4)]], device const int* offs [[buffer(5)]],
                    uint tid [[thread_index_in_threadgroup]], uint gid [[threadgroup_position_in_grid]], uint tpg [[threads_per_threadgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float x[NIN];
    uint c = gid * (tpg / 32) + sg;
    float acc[NEUR] = {0, 0, 0, 0, 0};
    for (int t = 0; t < T; t++) {
        device const float* xv = xs + (t % NX) * NIN;
        for (uint i = tid; i < NIN; i += tpg) x[i] = xv[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int n = 0; n < NEUR; n++) { float a = 0;
            device const short* wr = W + ((ulong)c * NEUR + n) * NIN + lane + offs[t];   // notes from RAM, every lap (offs = 0, unknown to the compiler)
            for (int k = 0; k < K; k++) a += float(wr[32 * k]) * x[lane + 32 * k];
            acc[n] += simd_sum(a); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) for (int n = 0; n < NEUR; n++) out[c * NEUR + n] = acc[n];
}
"""
setvbuf(stdout, nil, _IONBF, 0)
let dev = MTLCreateSystemDefaultDevice()!, q = dev.makeCommandQueue()!
print("device ok"); let lib: MTLLibrary
do { lib = try dev.makeLibrary(source: src, options: nil) } catch { print("metal compile:", error); exit(1) }
print("library ok")
let psR = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "lap_resident")!)
let psM = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "lap_ram")!)
print("pipelines ok")
let NIN = 768, NEUR = 5, NX = 64, T = 2000
let maxC = 4096
let W = dev.makeBuffer(length: maxC * NEUR * NIN * 2, options: .storageModeShared)!
let wp = W.contents().bindMemory(to: Int16.self, capacity: maxC * NEUR * NIN); for i in 0..<(maxC * NEUR * NIN) { wp[i] = Int16(truncatingIfNeeded: i &* 7919) }
let xs = dev.makeBuffer(length: NX * NIN * 4, options: .storageModeShared)!
let xp = xs.contents().bindMemory(to: Float.self, capacity: NX * NIN); for i in 0..<(NX * NIN) { xp[i] = Float((i * 31) % 97) / 97 }
let out = dev.makeBuffer(length: maxC * NEUR * 4, options: .storageModeShared)!
let offs = dev.makeBuffer(length: T * 4, options: .storageModeShared)!; memset(offs.contents(), 0, T * 4)
print("\(dev.name): cyclers each holding \(NEUR * NIN) notes; registers per thread allow \(psR.maxTotalThreadsPerThreadgroup) threads/threadgroup (resident), \(psM.maxTotalThreadsPerThreadgroup) (ram)")
func run(_ ps: MTLComputePipelineState, _ cyc: Int) -> Double {
    let tpg = min(ps.maxTotalThreadsPerThreadgroup, 1024) / 32 * 32; let cpg = tpg / 32
    let groups = (cyc + cpg - 1) / cpg
    var t = Int32(T), nx = Int32(NX)
    for rep in 0..<2 {
        let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
        e.setComputePipelineState(ps); e.setBuffer(W, offset: 0, index: 0); e.setBuffer(xs, offset: 0, index: 1); e.setBuffer(out, offset: 0, index: 2)
        e.setBytes(&t, length: 4, index: 3); e.setBytes(&nx, length: 4, index: 4); e.setBuffer(offs, offset: 0, index: 5)
        e.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1)); e.endEncoding()
        let t0 = Date(); cb.commit(); cb.waitUntilCompleted(); let d = Date().timeIntervalSince(t0)
        if rep == 1 { return Double(groups * cpg) * Double(NEUR * NIN) * Double(T) / d }
    }
    return 0
}
print(" cyclers | notes in registers (own 4096)      | notes from RAM every lap")
for cyc in [32, 64, 128, 256, 512, 1024, 2048, 4096] {
    let r = run(psR, cyc), m = run(psM, cyc)
    print(String(format: "%8d | %7.0f G notes/s  (=%5.0f GB/s)  | %7.0f G notes/s  (=%5.0f GB/s)   resident/RAM = %.1fx", cyc, r / 1e9, r * 2 / 1e9, m / 1e9, m * 2 / 1e9, r / m))
}

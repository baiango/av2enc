# hello_cuda — remote CUDA baseline

Standalone benchmark used to establish the RTX 3060 baseline for av2enc. It is
deliberately not part of the root CMake project and must not be included in the
encoder build.

Contents:

- `vector_add` float kernel
- naive direct 2D integer forward DCT for 8x8, 16x16, 32x32 blocks
- 16x16 SAD (one warp per block, 8 warps per CUDA block)
- D2D, pinned H2D, pageable H2D, and pinned D2H copy bandwidth
- CPU scalar references for every kernel; `main` asserts exact equality

**The integer DCT here is a throughput baseline only.** It is a direct
`O(N^4)` 2D sum with a Q6 fixed-point cosine basis, not the AV2 transform and
not suitable for real encoding. It exists to measure integer ALU and shared
memory throughput for block transforms.

## Build

```sh
cd /workspace/av2enc/bench/hello_cuda
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Run

```sh
./build/hello_cuda_bench          # correctness + median-of-10 benchmarks
./build/hello_cuda_bench --tiny   # correctness only (handy under ncu)
```

The process exits non-zero if any correctness check fails.

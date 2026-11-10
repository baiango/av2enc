# Architecture — av2enc

## Language / toolchain (locked)

- Host: **C++20**
- GPU: **CUDA C++**, CUDA 12.8, `sm_86`
- Build: CMake >= 3.24 + Ninja
- Warnings: `-Wall -Wextra -Wpedantic -Wconversion -Wshadow`
- No exceptions in hot paths; use error codes / `std::optional`
- Prefer `std::span`, `std::byte`, `std::unique_ptr`, RAII
- No third-party codec code. Standard library + CUDA + CUB only.

## Process split

```
Y4M/I420 input
  -> FrameScheduler
  -> Tile/BlockEncoder
       -> intra mode search        (GPU)
       -> forward transform + quant (GPU)
       -> reconstruction           (GPU)
  -> EntropyWriter                    (CPU, serial per tile)
  -> OBU/IVFWriter                    (CPU)
```

CPU owns:
- bitstream syntax and OBU framing
- entropy coder and CDF state
- rate control / QP selection
- reference frame management
- validation harness

GPU owns:
- forward/inverse transforms
- quantization / dequantization
- SAD/SATD and mode distortion
- intra prediction candidates
- later: motion estimation, motion compensation, loop filters

## First milestone (M0)

Still-picture encoder, 8-bit 4:2:0, one tile:

- fixed 64x64 superblocks
- fixed 32x32 luma coding blocks (chroma 16x16)
- DC intra only
- DCT 32x32 only
- scalar quantization with fixed QP
- no loop filters, film grain, SVC, palette, CfL
- output: IVF containing one frame

M0 is done when `avmdec` decodes the IVF and the decoded YUV is close to the
source. Bit-exactness of our reconstruction vs the decoder is the next gate.

## Module layout

```
src/
  main.cpp                 CLI
  io/y4m.{h,cpp}           Y4M reader (8-bit 4:2:0 first)
  io/ivf.{h,cpp}           IVF muxer
  bitstream/bitwriter.*    MSB-first bit writer
  bitstream/obu.*          OBU header + payload framing
  entropy/range_coder.*    multi-symbol range coder (CPU)
  entropy/cdf.*            CDF tables + adaptation
  cpu/transform.*          scalar reference DCT/quant
  cpu/intra.*              scalar reference DC prediction
  cuda/transform.cu        GPU transform/quant kernels
  cuda/intra.cu            GPU intra search kernels
tests/
  y4m_test.cpp
  ivf_test.cpp
  bitwriter_test.cpp
  range_coder_test.cpp
```

## CUDA rules

1. Every kernel has a scalar CPU reference; tests assert equality on randomized
   blocks before the kernel is used in the encoder.
2. No `cudaMalloc` / `cudaFree` in per-frame or per-block paths. Preallocate a
   persistent workspace.
3. Use pinned host memory for frame I/O; batch transfers once per frame.
4. One CUDA block per superblock for search kernels; warp reductions for
   SAD/SATD; shared memory for reference windows.
5. Use `cudaGraph` for the per-frame sequence once the kernel set stabilizes.
6. Compile with `-lineinfo`; profile with Nsight Systems/Compute.
7. `CMAKE_CUDA_ARCHITECTURES=86`.

## Validation

- `avmdec` and `dav2d` are black boxes. Do not read their encoder-side source.
- Unit tests: IVF structure, Y4M parse, bit writer, range-coder round-trip.
- Integration: encode one 64x64 frame, decode with `avmdec`, compare PSNR.
- Keep test media tiny; the remote disk is 16GB and non-persistent.

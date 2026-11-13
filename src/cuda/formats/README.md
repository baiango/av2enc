# CUDA format device chain

Status (2026-11-13): **10-bit 4:2:0 / 4:2:2 / 4:4:4 / 4:0:0 device chain
verified bit-exact against the product CPU references on `sm_86`.**

This directory owns the format/sample half of the GPU still path:

| Stage | Implementation | CPU reference |
|---|---|---|
| frame / plane geometry | `format_device.{h,cpp}` | `fmt::make_frame_geometry()` |
| sample conversion (8→10, native 10/12 upload, 10→8, 10/12 download, padding) | `high_bitdepth.cu`, `pipeline/still/hbd/hbd_frame.cu` | `fmt::convert_420_8_to_picture()`, `fmt::convert_to_420_8()`, `fmt::replicate_borders()` |
| DC prediction | `high_bitdepth_gpu.cu` (`dc_predict_kernel`) | `hbd::encode_still()` / `hbd_still.cpp` |
| forward DCT + quant | `pipeline/still/hbd/hbd_tx.cu` | `txv::forward_transform()` |
| dequant + inverse + reconstruction | `pipeline/still/hbd/hbd_tx.cu`, `high_bitdepth_gpu.cu` | `hbd::dequantize()` / `hbd::inverse_block()` |
| deblock / loop-filter plane geometry | `format_device.{h,cpp}` (`FilterGeometry`) | `fmt::make_frame_geometry()`, `deblock::deblock_plane_*()` |
| bitstream-facing buffers | `format_device.h` (`BitstreamLayout`) + `HighBitDepthBlock`/levels seam | `hbd::encode_still()` |

## Geometry contract

`FrameLayout` gives three extents per plane:

- `coded_*` — the decoder's `MiCols*4` plane (prediction/reconstruction target);
- `storage_*` — the 64-aligned coded extent used by the high-bit-depth device
  buffers (a partial 64x64 edge leaf has room);
- `padded_*` — the format layer's 64-aligned visible extent (filter/plane
  helpers).

`FilterGeometry` derives the §7.17 deblock plane extents, MI grid and interior
4x4/8x8 edge counts for both sample container widths (8-bit → 1 byte, 10/12-bit
→ 2 bytes).  Plane block rectangles mirror `fmt::block_rect()` for every
BLOCK_SIZE entry, including the `BLOCK_INVALID` chroma shapes.

`BitstreamLayout` is the capacity contract between the GPU block stage and the
CPU entropy writer: 1024 raster levels per block (adjusted 32x32 maximum), one
EOB per plane, and one descriptor per leaf.

## Device block stage

`high_bitdepth_gpu.cu` runs:

1. upload a coded 10-bit `fmt::Picture` plane (`cudaMemcpy2D` into the arena);
2. §7.13.2.10 DC prediction with the full §7.13.3.22 `Div_Lut` (needed for
   4:2:2 chroma 16x32, `w + h = 48`);
3. `hbd_tx_batch()` with `bit_depth = 10` and the per-job `quant_step =
   qlookup(base_q_idx)` (the default 528 is the QP-90 step);
4. `clip1(pred + residual)` reconstruction into the padded storage plane;
5. download scan-ordered levels, EOBs and reconstructed planes, then hand them
   to the existing CPU entropy/OBU writers.

All buffers are arena-owned; the stage contains no `cudaMalloc`/`cudaFree`.
`required_device_bytes()` accounts for each plane's separate arena alignment.

## Tests (run on the RTX 3060, `sm_86`)

```sh
# Geometry vs fmt/deblock references (host only)
./src/cuda/formats/build_format_device_test.sh
# => format_device_test: PASS (1064426 checks)

# Full 10-bit block path vs hbd::encode_still: q=1..303, 420/422/444/400,
# 64x64 exact plus 100x100 and 640x360 odd/partial edges
./src/cuda/formats/build_high_bitdepth_gpu_test.sh
# => high_bitdepth_cuda_test: PASS (27147456 comparisons)

# Sample conversion + HBD transforms
./src/cuda/pipeline/still/hbd/build_hbd_test.sh
# => hbd_test: PASS (19312 checks)
# => hbd_cuda_test: PASS (5244850 checks)

# 8-bit / 10-bit chroma transforms (unchanged baseline)
./src/cuda/formats/chroma/build_and_test.sh
# => 1514866 checks, PASS
./src/cuda/formats/chroma/build_gpu_test.sh
# => 1742235 checks, PASS
```

## Known integration gap

The product CLI still routes `--still --bit-depth 10 --chroma ...` through the
CPU `hbd::encode_still()`; the GPU stage is a standalone component (not in the
root CMake target).  Wiring it needs:

1. add `high_bitdepth_gpu.cu`, `format_device.cpp`, and the `hbd`/`hbd_depth`/
   `hbd_headers` host support to `AV2ENC_STILL_GPU_SOURCES`/host support in the
   root `CMakeLists.txt` (integrator-owned);
2. make `still_gpu.cu` accept `uint16_t` planes and dispatch this entry point
   for depth 10 (M1 core is still 8-bit);
3. extend the deblock kernel from 8-bit 4:2:0 to `FilterGeometry` +
   `sample_bytes` (geometry is ready; the filter arithmetic still clips at 8
   bits).

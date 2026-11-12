// CUDA high-bit-depth sample path for the format component.
//
// These are the three operations that touch every sample when a 10-bit /
// 4:2:2 / 4:4:4 frame enters or leaves the GPU:
//
//   * expand 8-bit samples into the LSB-aligned 16-bit container
//   * reduce 10/12-bit samples back to 8-bit (round-to-nearest)
//   * replicate visible edge samples into the MI-aligned coded padding
//
// The scalar CPU reference for all three lives in `src/encoder/format/`:
// `expand_8_to_depth()`, `reduce_to_8()`, and `replicate_borders()`.  The test
// asserts bit-exact equality; no kernel may diverge from those functions.
//
// This is a component, not the pipeline.  Buffers are caller-owned; the test
// harness uses cudaMalloc, but encoder code must use `src/cuda/arena.*`.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace av2enc::cuda::hb {

// dst[y * dst_stride + x] = uint16(src[y * src_stride_bytes + x]) << shift.
// `shift` is BitDepth - 8 (0 for 8-bit input).  Strides are in bytes for the
// 8-bit source and in samples for the 16-bit destination.
cudaError_t expand_8_to_16(const std::uint8_t* src, std::uint16_t* dst,
                           int width, int height, int src_stride_bytes,
                           int dst_stride_samples, int shift,
                           cudaStream_t stream = nullptr);

// dst[y * dst_stride_bytes + x] = round_to_8(src[y * src_stride + x], bit_depth)
// with saturation at 255.  `bit_depth` is the source depth (8/10/12).
cudaError_t reduce_16_to_8(const std::uint16_t* src, std::uint8_t* dst,
                           int width, int height, int src_stride_samples,
                           int dst_stride_bytes, int bit_depth,
                           cudaStream_t stream = nullptr);

// Fills the coded rectangle [0,coded_width) x [0,coded_height) outside the
// visible rectangle by replicating the nearest visible sample.  The visible
// region itself is left untouched.  Matches `fmt::replicate_borders()`.
cudaError_t replicate_borders_16(std::uint16_t* plane, int stride,
                                 int coded_width, int coded_height,
                                 int visible_width, int visible_height,
                                 cudaStream_t stream = nullptr);

}  // namespace av2enc::cuda::hb

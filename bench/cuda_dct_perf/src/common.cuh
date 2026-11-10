// Shared host/device declarations for the standalone M0 DCT precision and
// performance benchmark.  This directory is not part of the root CMake project.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status_ = (call);                                            \
    if (status_ != cudaSuccess) {                                            \
      std::fprintf(stderr, "CUDA error: %s (%s) at %s:%d\n", #call,          \
                   cudaGetErrorString(status_), __FILE__, __LINE__);         \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

// Fixed-point layout for the integer variant.  The row pass stores
// round(row_dot * scale_w * 2^kQrow); the column pass accumulates in int32 and
// applies one 64-bit multiplier for quantization.  See README.md.
constexpr int kQrow = 20;
constexpr int kQshift = 60;

// Everything the kernels need for one transform size.  The double fields are
// computed with the exact expression order used by
// src/encoder/m0_still.cpp:forward_dct()/encode_residual_plane().
struct BlockParams {
  int n = 0;
  double scale_w = 0.0;
  double scale_h = 0.0;
  double quant_factor = 0.0;
  int dq_denom = 1;
  // Fixed-point row multiplier and final quant multiplier, plus the
  // uncorrected multiplier used to demonstrate why the correction is needed.
  std::int32_t row_mul = 0;
  std::int64_t quant_mul = 0;
  std::int64_t quant_mul_naive = 0;
};

enum class Variant : int {
  kDouble = 0,
  kFloat = 1,
  kFixed = 2,
  kHybrid = 3,
};
constexpr int kVariantCount = 4;

inline const char* variant_name(Variant v) {
  switch (v) {
    case Variant::kDouble:
      return "double";
    case Variant::kFloat:
      return "float";
    case Variant::kFixed:
      return "fixed";
    case Variant::kHybrid:
      return "hybrid";
  }
  return "?";
}

// Copies the normative AV2 Dct_Kernel16/32 tables into device memory once.
void init_device_tables(const std::int16_t* kernel32,
                        const std::int16_t* kernel16);
void shutdown_device_tables();

// Architecture of the kernel image selected by the driver (86 on the 3060).
int loaded_binary_version();

// One CUDA block per transform block.
//   variant     : arithmetic under test
//   n           : 32 or 16
//   warps       : threads per CUDA block / 32 (4, 8, 32 for n=32; 2, 4, 8 for n=16)
//   write_coeff : also store the unquantized coefficients (double/float only)
//   d_coeff     : double* for kDouble/kHybrid, float* for kFloat, ignored for kFixed
//   fixed_naive : use the uncorrected fixed-point quant multiplier
void launch_transform(Variant variant, int n, int warps, bool write_coeff,
                      const std::int16_t* d_in, int nblocks,
                      const BlockParams& params, void* d_coeff,
                      std::int32_t* d_q, bool fixed_naive,
                      cudaStream_t stream);

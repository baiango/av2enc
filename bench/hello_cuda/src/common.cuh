#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

// Fixed-point scale for the naive integer DCT basis.
// Basis coefficients are round(cos(...) * 2^kDctQ). With Q=6 the maximum
// basis magnitude is 31, so a direct 32x32 2D accumulation of 8-bit input
// stays well inside int32 (worst case ~2.7e8).
constexpr int kDctQ = 6;
constexpr int kDctMaxSize = 32;
constexpr int kDctNumSizes = 3;
constexpr int kDctSizes[kDctNumSizes] = {8, 16, 32};

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status_ = (call);                                            \
    if (status_ != cudaSuccess) {                                            \
      std::fprintf(stderr, "CUDA error: %s (%s) at %s:%d\n", #call,          \
                   cudaGetErrorString(status_), __FILE__, __LINE__);         \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

// Fills the host-side basis tables and uploads them to device constant memory.
void init_dct_basis();

// Architecture of the kernel image the driver selected for this device
// (binaryVersion: 86 on RTX 3060, 120 on RTX 5070). Used to confirm which
// fatbin slice actually executes.
int loaded_binary_version();

// Returns a 32x32 row-major table for size index 0 (8), 1 (16), 2 (32).
const int32_t* host_dct_basis(int size_index);

void launch_vector_add(const float* a, const float* b, float* out, int n,
                       cudaStream_t stream);

// 16-byte vectorized device-to-device copy (SM path, for comparison with the
// copy-engine cudaMemcpyAsync path). `bytes` must be a multiple of 16.
void launch_copy(const void* in, void* out, size_t bytes, cudaStream_t stream);

void launch_dct(const int32_t* in, int32_t* out, int nblocks, int n,
                cudaStream_t stream);

// One warp per 16x16 block; 8 blocks per CUDA block.
void launch_sad16(const uint8_t* a, const uint8_t* b, int32_t* out, int nblocks,
                  cudaStream_t stream);

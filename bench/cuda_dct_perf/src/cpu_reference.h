// CPU transcription of the M0 forward DCT + quantization and the normative
// inverse transform.  Copied line-for-line from src/encoder/m0_still.cpp
// (read-only reference); do not change the arithmetic order here.
#pragma once

#include "common.cuh"

#include <cstdint>

namespace ref {

// Same constants as src/encoder/m0_still.cpp.
constexpr int kQuantStep = 528;
constexpr int kQuantTableBits = 3;

BlockParams make_block_params(int n);

// Forward DCT + quantization.  `tmp`, `coeff` and `q` are preallocated for n*n.
void forward_dct_quant(const std::int16_t* block, int n,
                       const BlockParams& params, double* tmp, double* coeff,
                       std::int32_t* q);

// Dequantization + normative inverse DCT, matching encode_residual_plane().
void inverse_dct_from_q(const std::int32_t* q, int n,
                        const BlockParams& params, int* residual);

int clip1(int value);
int round2(int value, int n);

}  // namespace ref

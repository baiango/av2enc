// AV2 v1.0.0 DIP linear model weights.
//
// Normative source: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7), §9.3 default CDF tables, table
// `Dip_Weights[6][64][11]` (PDF pages 1095-1100). The PDF is the only
// artefact that prints the numbers; the HTML specification omits them.
//
// Generated from spec/AV2_Spec_v1.0.0.pdf by
// src/encoder/dip/generate_dip_weights.py; do not hand-edit.
//
// Checksum (SHA-256 of the comma-separated values in raster order, mode
// major, 64 output positions, 11 features):
//   e5deedf1ed42fb944e233733c735593b04a3ae7116b51045bc692a11f53d04f7

#pragma once

#include <cstdint>

namespace av2enc::dip {

// Dip_Weights[mode][i * 8 + j][feature]. Values are signed Q10 model
// coefficients: the predictor computes (sum + 512) >> 10.  The initializer
// body is shared verbatim with the CUDA device table in
// src/cuda/dip/dip_device.cuh so the CPU and GPU cannot drift.
inline constexpr std::int16_t kDipWeights[6][64][11] = {
#include "encoder/dip/dip_weights_data.inc"
};

}  // namespace av2enc::dip

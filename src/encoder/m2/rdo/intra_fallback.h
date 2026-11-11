// Intra fallback rate for the M2 P-frame RDO layer.
//
// The intra fallback in `m2_frame.cpp` is DC prediction plus a DCT residual.
// Its coefficient rate is the M1 TransformStreamRate model, so this layer does
// not reimplement coefficient probabilities.  The intra syntax term comes from
// the M1 intra/chroma syntax helpers through `InterRateModel`.
//
// The P-frame fallback is a single DC mode, so the caller passes the M1
// `TxPlan` geometry (adjusted transform width/height, raster levels, EOB) for
// Y, U and V.

#pragma once

#include <cstdint>
#include <span>
#include <string>

namespace av2enc::m2::rdo {

struct IntraPlaneRateInput {
  int plane = 0;  // 0 luma, 1 U, 2 V
  int tx_size = 0;
  int adj_w = 0;
  int adj_h = 0;
  std::span<const std::int32_t> levels_raster;  // adj_w * adj_h, raster
  int eob = 0;
};

// Sum of `TransformStreamRate::block_bits()` over the three planes.  Returns
// false on an invalid geometry or level span and sets `error`.
bool intra_coefficient_bits(std::span<const IntraPlaneRateInput> planes,
                            double* bits, std::string* error = nullptr);

}  // namespace av2enc::m2::rdo

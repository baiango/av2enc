// AV2 v1.0.0 GDF (§7.20.5) encoder decision and application.
//
// The pixel filter is the recovered normative reference in
// src/encoder/filters/spec_tables/gdf.cpp.  This file only:
//   * maps the §5.18.7.9 GdfBlks grid to the §7.20.1 stripe-aligned sample
//     regions the decoder actually filters;
//   * applies it to a frame (one call per unit x 64-line stripe);
//   * decides `use_gdf` from reconstruction-vs-source SSE, per GDF unit and
//     per 64x64 block, and costs the signalled bits with the real §8.2 coder.
//
// Spec-only: no encoder or decoder state outside the passed buffers.
#ifndef AV2ENC_ENCODER_FILTERS_GDF_STREAM_GDF_DECISION_H_
#define AV2ENC_ENCODER_FILTERS_GDF_STREAM_GDF_DECISION_H_

#include "codec/filters/gdf/gdf_syntax.h"

#include <cstdint>
#include <string>
#include <vector>

namespace av2enc::filters::gdf {

// A read-only 16-bit luma plane.
struct GdfPlane {
  const std::uint16_t* data = nullptr;
  int width = 0;
  int height = 0;
  int stride = 0;
};

// §7.20.2 current/CdefFrame luma planes, 16-bit samples.  For a frame with
// CDEF/CCSO disabled both pointers are the same reconstruction buffer.
struct GdfSourceFrame {
  const std::uint16_t* curr = nullptr;
  const std::uint16_t* cdef = nullptr;
  int stride = 0;
};

// §7.20.1 variables used by the filter (not gdf_params inputs).
struct GdfFilterInputs {
  bool frame_is_intra = true;
  int base_q_idx = 0;
  int bit_depth = 8;
  int num_total_refs = 0;
  int order_hints[2] = {0, 0};
  int frame_distance[2] = {0, 0};
  int restricted_oh = 0;
};

// One contiguous part of a GDF unit that belongs to a single 64-line stripe.
// §7.20.2 clamps source reads to the current stripe (two lines beyond it), so a
// unit that spans a stripe boundary must be filtered in more than one call.
struct GdfStripeRegion {
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
  int stripe_start_y = 0;
  int stripe_end_y = 0;
};

// Fills `out` (capacity 8) and returns the number of regions for grid cell
// (row, col).  Regions are the exact sample ranges whose §7.20.1 GdfBlks index
// is (row, col): 64/128 sample units aligned with the offset (-8) stripes, with
// the last row extended to MiRows - 1 by the spec's Min(MiRowEnd - 1, ...).
int GdfUnitRegions(const GdfUnitLayout& layout, int row, int col,
                   GdfStripeRegion out[8]);

// Applies GDF to `out` (which must already contain a copy of CdefFrame).  When
// `params.per_block` is false every unit is filtered and `unit_flags` is
// ignored; otherwise it must hold layout.cols * layout.rows flags.
bool ApplyGdfToFrame(const GdfSourceFrame& source, const GdfFilterInputs& inputs,
                     const FrameGdfContext& context,
                     const FrameGdfParams& params,
                     const std::vector<std::uint8_t>& unit_flags,
                     std::uint16_t* out, int out_stride, std::string* error);

struct GdfDecisionOptions {
  GdfFilterInputs filter;
  // Search gdf_pic_qc_idx x gdf_pic_scale_idx (0..3 each) instead of using the
  // two fixed values.
  bool search_pic_indices = true;
  int pic_qc_idx = 0;
  int pic_scale_idx = 0;
  // Mirrors the frame header's disable_cdf_update for the standalone use_gdf
  // bit-cost measurement.
  bool disable_cdf_update = true;
  // SSE added per signalled bit.  0 = quality-only (enable every improving
  // unit); larger values trade filter gain for bit rate.
  double lambda_bits = 0.0;
};

struct GdfDecision {
  FrameGdfParams params;
  // Row-major GdfBlks grid for the chosen mode (empty when per_block == 0).
  std::vector<std::uint8_t> unit_flags;
  // Row-major 64x64 evaluation blocks inside each unit (regions are
  // stripe-aligned, so these are unit-local 64x64 quadrants).
  std::vector<std::uint8_t> block64_flags;
  int block64_cols = 0;
  int block64_rows = 0;

  std::int64_t sse_before = 0;
  std::int64_t sse_after = 0;
  int enabled_units = 0;
  int total_units = 0;
  std::int64_t header_bits = 0;   // sequence + frame GDF bits
  std::int64_t symbol_bytes = 0;  // standalone §8.2 payload for use_gdf
  std::int64_t symbol_bits = 0;   // symbol_bytes * 8
  double psnr_before = 0.0;
  double psnr_after = 0.0;
  int chosen_qc_idx = 0;
  int chosen_scale_idx = 0;
};

// Chooses gdf_pic_qc_idx / gdf_pic_scale_idx, gdf_per_block and the GdfBlks
// flags from reconstruction-vs-source SSE.  `recon` is the §7.20 input
// (CurrFrame/CdefFrame); `original` is the source frame the SSE is measured
// against.  `context.enable_gdf` must be true.
bool DecideGdf(const GdfSourceFrame& recon, const GdfPlane& original,
               const FrameGdfContext& context,
               const GdfDecisionOptions& options, GdfDecision* decision,
               std::string* error);

// Exact byte/bit cost of a standalone use_gdf symbol sequence in coding order.
// Used by the decision and pinned by the unit tests.  `disable_cdf_update`
// mirrors the frame header bit.
bool MeasureUseGdfSymbols(const GdfUnitLayout& layout,
                          const FrameGdfParams& params, int sb_size,
                          const std::vector<std::uint8_t>& flags,
                          bool disable_cdf_update, std::int64_t* bytes,
                          std::int64_t* bits, std::string* error);

}  // namespace av2enc::filters::gdf

#endif  // AV2ENC_ENCODER_FILTERS_GDF_STREAM_GDF_DECISION_H_

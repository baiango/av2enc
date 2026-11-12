// §7.20.5 Apply GDF filter process, using the recovered normative tables.
//
// The tables in gdf_tables.h come from the official gdf_*.h attachments
// (see raw/SHA256SUMS and generate_spec_tables.py).  This is a scalar CPU
// reference; it is not wired into the M1/M2 encode path.
//
// Scope: luma only, as the specification defines GDF.  BitDepth 8, 10 and 12
// are computed with the spec's own shifts; 10-bit is the only high bit depth
// signalable in v1.0.0.

#ifndef AV2ENC_ENCODER_FILTERS_SPEC_TABLES_GDF_H_
#define AV2ENC_ENCODER_FILTERS_SPEC_TABLES_GDF_H_

#include <cstdint>

#include "codec/filters/tables/luma_source.h"

namespace av2enc::filters::spec {

// §5.18.7.9 GDF params syntax / §7.20.1 derivation inputs.
struct GdfParams {
  bool frame_is_intra = true;
  int base_q_idx = 0;
  int bit_depth = 8;
  int gdf_pic_qc_idx = 0;   // f(2), 0..3
  int gdf_pic_scale_idx = 0;  // f(2), 0..3; GdfPixScale = 1 + this
};

// §7.20.1: the two table indices for one luma block.
struct GdfBlockIndices {
  int qp_idx = 0;       // 0..5
  int ref_dst_idx = 0;  // 0..5; 0 means "intra"
};

// §7.20.1 derivation.  `order_hints` and `frame_distance` are the first
// Min(num_total_refs, 2) entries.  `restricted_oh` is the spec's RESTRICTED_OH.
GdfBlockIndices GdfBlockTableIndices(const GdfParams& params,
                                     const int* order_hints,
                                     const int* frame_distance,
                                     int num_total_refs,
                                     int restricted_oh);

// §7.20.5 apply_gdf_filter(x, y, qpIdx, refDstIdx, w, h, stripeEndY).
//
// `lr` is LrFrame[0]; `src` supplies get_gdf_sample() via §7.20.2.  `x` and
// `y` are the top-left luma sample of the unit.  `w` and `h` are even.
//
// `stripe_end_y` is the spec's unclippedStripeEndY.  The v1.0.0 pseudocode
// does not read it (get_gdf_sample calls get_luma_source_sample); it is kept
// for signature fidelity and sanity-checked by the caller.
void ApplyGdfFilter(const LumaSource& src, std::uint16_t* lr, int lr_stride,
                    int x, int y, int qp_idx, int ref_dst_idx, int w, int h,
                    int stripe_end_y, int gdf_pix_scale, int bit_depth);

}  // namespace av2enc::filters::spec

#endif  // AV2ENC_ENCODER_FILTERS_SPEC_TABLES_GDF_H_

// Compound prediction and mode application for the multi-reference layer.
//
// The building blocks live elsewhere and are reused unchanged:
//   * src/encoder/inter/mc/     translational MC, subpel filters, MV stack
//   * src/encoder/inter/advanced/compound.*  §7.13.3.14/.27-.30 blending
//
// What this file adds:
//   * a 1/16-sample compound prediction plane produced with the MC filters
//     (mc::predict_plane rounds to integers at InterRound1 = 11; compound
//     prediction needs InterRound1 = 7 and 1/16 sample output);
//   * the per-plane reference start positions (§7.13.3.14 get_mask) so the
//     implicit-mask blend is exact;
//   * a compact mode descriptor that maps 1:1 to the §5.20.7.16 symbols and
//     to adv::CompoundJob;
//   * CWP weight/index conversion for both signallable rows.
//
// No bitstream syntax is written here.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "encoder/inter/advanced/compound.h"
#include "encoder/inter/mc/mc.h"

namespace av2enc::m2::multiref {

namespace adv = av2enc::inter::adv;
namespace mc = av2enc::inter::mc;

// CWP weights are `Cwp_Weighting_Factor[is_same_side][idx]` (§5.20.7.6).
inline constexpr int kMaxCwpNum = adv::kMaxCwpNum;
inline constexpr int kCwpEqual = adv::kCwpEqual;

int cwp_weight_from_index(int index, bool same_side);
// First signalable index for `weight`, or -1.  Searches index 0..kMaxCwpNum-1.
int cwp_index_from_weight(int weight, bool same_side);

// §5.20.7.18 read_wedge_mode() inverse.  `quad*5 + angle` is the 0..19 wedge
// angle index and `dist` the 0..3 distance.
struct WedgeSyntax {
  int quad = 0;
  int angle = 0;
  int dist = 0;
};

// Decoder-side lookups used by the encoder to keep its decisions signallable.
bool wedge_index_to_syntax(int wedge_index, WedgeSyntax& out);
int wedge_syntax_to_index(int quad, int angle, int dist);
// Bit count of read_wedge_mode() for the given index (0 if not signalable).
int wedge_syntax_bits(int wedge_index);

// Everything §5.20.7.16 can signal for a compound block.
struct CompoundModeParams {
  adv::CompoundType type = adv::CompoundType::kAverage;
  // CWP: signalled index and the resulting weight (8 == CWP_EQUAL).
  int cwp_index = 0;
  int cwp_weight = kCwpEqual;
  int wedge_index = 0;
  int wedge_sign = 0;
  int mask_type = 0;
  // Derived by the decoder (enable_imp_msk_bld && CWP_EQUAL && ...).  When set,
  // the average blend uses get_mask() instead of the CWP weights.
  bool use_imp_mask = false;
};

bool same_mode(const CompoundModeParams& a, const CompoundModeParams& b);

// RDO-free proxy for the number of §5.20.7.16 mode symbols, used by the
// decision API.  This is an encoder model, not the real range-coder cost.
int compound_mode_bits(const CompoundModeParams& params, bool same_side,
                       bool include_comp_mode = true);

// One plane of 1/16-sample predictions, block-local (stride == width).
struct PredPlane16 {
  int width = 0;
  int height = 0;
  int stride = 0;
  std::vector<std::int16_t> samples;

  const std::int16_t* row(int i) const {
    return samples.data() + static_cast<std::ptrdiff_t>(i) * stride;
  }
};

// Two reference predictions for one luma block plus the per-plane reference
// start positions the implicit-mask blend needs.
struct CompoundPrediction {
  int bsize = 0;
  mc::BlockRect luma_rect{};
  PredPlane16 pred[2][3];
  int sub_x = 1;
  int sub_y = 1;
  // Frame MI dimensions, needed by §7.13.3.14 get_mask().
  int mi_cols = 0;
  int mi_rows = 0;
  // RefStartX/Y from §7.13.3.14 (plane samples, per reference, per plane).
  int ref_start_x[2][3] = {};
  int ref_start_y[2][3] = {};

  int plane_width(int plane) const { return pred[0][plane].width; }
  int plane_height(int plane) const { return pred[0][plane].height; }
};

// §7.13.3.18 compound form: InterRound1 = 7, no Clip1.  `mv` is 1/8 luma
// samples.  `rect` is in plane samples.
bool predict_plane_compound_1_16(const mc::RefFrame& ref, int plane,
                                 const mc::BlockRect& rect, mc::Mv mv,
                                 mc::InterpFilter filter, std::int16_t* dst,
                                 int dst_stride, std::string& error);

// Predicts both references for Y/U/V.  References are unscaled.
bool build_compound_prediction(const mc::RefFrame& ref0,
                               const mc::RefFrame& ref1,
                               const mc::BlockRect& luma_rect, int bsize,
                               mc::Mv mv0, mc::Mv mv1,
                               mc::InterpFilter filter, int mi_cols,
                               int mi_rows, CompoundPrediction& out,
                               std::string& error);

// True when the implicit-mask blend applies to this block (§7.13.3.14):
// enable_imp_msk_bld, CWP_EQUAL, YMode != GLOBAL_GLOBALMV, both references
// unscaled, no compound warp.  `global_global_mv` is the caller's YMode test.
bool implicit_mask_eligible(bool seq_enable_imp_msk_bld,
                            const CompoundModeParams& params,
                            bool global_global_mv, bool refs_scaled,
                            bool compound_warp);

// Applies one mode to the three planes and writes integer samples.
//
//   * `params` selects AVERAGE (+CWP / implicit mask), WEDGE or DIFFWTD.
//   * `dst[p]` points at the destination plane origin with `dst_stride[p]`;
//     writes are clipped to `luma_rect` (and its chroma subsample).
//   * The luma mask used by wedge/diffwtd is generated internally and
//     subsampled for chroma exactly as §7.13.3.30 requires.
//   * `params.use_imp_mask` is taken as given; derive it with
//     implicit_mask_eligible().
bool apply_compound_prediction(const CompoundPrediction& prediction,
                               const CompoundModeParams& params,
                               std::uint8_t* dst[3], int dst_stride[3],
                               std::string& error);

// Convenience: SSE of a mode against the source luma block.
std::int64_t compound_luma_sse(const CompoundPrediction& prediction,
                               const CompoundModeParams& params,
                               const std::uint8_t* src, int src_stride,
                               std::uint8_t* scratch, int scratch_stride);

}  // namespace av2enc::m2::multiref

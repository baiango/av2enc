// AV2 v1.0.0 temporal inter tools — refine-MV.
//
// Implements §5.20.7.17 `read_refinemv()`, §7.13.3.3 `predict_refinemv()`,
// §7.13.3.4/.5 `get_ref_area()`, and §7.13.3.6 `search_refinemv()`.
//
// The search is a 24-neighbour integer-offset search around the starting MVs.
// Both the search and the final prediction use the normative two-pass subpel
// arithmetic; the search uses BILINEAR single-reference rounding (8-bit), the
// final prediction uses EIGHTTAP_SHARP at 1/16 precision and a compound
// average.

#ifndef AV2ENC_ENCODER_INTER_TEMPORAL_REFINEMV_H_
#define AV2ENC_ENCODER_INTER_TEMPORAL_REFINEMV_H_

#include <cstdint>

#include "codec/inter/temporal/temporal_types.h"

namespace av2enc::inter::temporal {

// §5.20.7.17 read_refinemv().  `ctx.syntax_bit` is the signalled bit when the
// decision is switchable (ignored otherwise).
TemporalStatus refine_mv_syntax(const RefineMvConfig& ctx, RefineMvSyntax* out);

// §5.20.7.17 is_refinemv_allowed_mode().
bool refine_mv_allowed_mode(const RefineMvConfig& ctx);
// §5.20.7.17 is_switchable_refinemv().
bool refine_mv_switchable(const RefineMvConfig& ctx);
// §5.20.7.17 is_refinemv_allowed_reference().
bool refine_mv_allowed_reference(const RefineMvConfig& ctx);

// §7.13.3.3 offset_refinemv().
MvPair offset_refinemv(const MvPair& src, int dx, int dy);

struct RefAreaInput {
  int w = 0;
  int h = 0;
  int sub_x = 0;
  int sub_y = 0;
  int x = 0;  // plane sample coordinates of the block origin
  int y = 0;
  int frame_w = 0;  // luma FrameWidth / FrameHeight
  int frame_h = 0;
  int ref_mi_cols[2] = {0, 0};
  int ref_mi_rows[2] = {0, 0};
  int ref_luma_w[2] = {0, 0};  // RefFrameWidth[refIdx]
  int ref_luma_h[2] = {0, 0};
  Mv cand_mvs[2];
  int num_refs = 2;
};

// §7.13.3.4/.5 get_ref_area(), one RefArea per reference list.
TemporalStatus compute_ref_area(const RefAreaInput& in, RefArea out[2]);

struct RefineSearchInput {
  int x = 0;  // luma block origin
  int y = 0;
  int w = 0;
  int h = 0;
  int frame_w = 0;
  int frame_h = 0;
  int ref_mi_cols[2] = {0, 0};
  int ref_mi_rows[2] = {0, 0};
  int ref_luma_w[2] = {0, 0};
  int ref_luma_h[2] = {0, 0};
  PlaneView ref[2];
  Mv cand_mvs[2];
  int num_refs = 2;
  bool tip_pred = false;
  bool switchable = true;
  bool use_ref_area = true;
  RefArea area[2]{};
  std::uint8_t* scratch0 = nullptr;  // (w+4) * (h+4) bytes
  std::uint8_t* scratch1 = nullptr;
  int scratch_stride = 0;
};

struct RefineSearchResult {
  int dx = 0;
  int dy = 0;
  MvPair offset_mvs;
  std::int64_t best_sad = 0;
};

// §7.13.3.6 search_refinemv().  Returns dx = dy = 0 for all early-exit cases.
TemporalStatus search_refinemv(const RefineSearchInput& in,
                               RefineSearchResult* out);

struct RefinePredictInput {
  PlaneView ref[2];
  int num_refs = 2;
  int x = 0;  // plane sample coordinates
  int y = 0;
  int w = 0;
  int h = 0;
  int sub_x = 0;
  int sub_y = 0;
  int frame_w = 0;
  int frame_h = 0;
  MvPair offset_mvs;  // result of offset_refinemv(), 1/8 luma
  int cwp_weight = kCwpEqual;
  bool use_ref_area = true;
  RefArea area[2]{};
  std::uint8_t* dst = nullptr;
  int dst_stride = 0;
  std::int16_t* scratch0 = nullptr;
  std::int16_t* scratch1 = nullptr;
  int scratch_stride = 0;
};

// Final §7.13.3.3 prediction with EIGHTTAP_SHARP and the compound average.
TemporalStatus refine_predict_block(const RefinePredictInput& in);

}  // namespace av2enc::inter::temporal

#endif  // AV2ENC_ENCODER_INTER_TEMPORAL_REFINEMV_H_

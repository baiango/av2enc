#include "encoder/m2/multiref/compound_predict.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <limits>

#include "encoder/inter/advanced/tables/wedge_tables.h"
#include "encoder/inter/mc/mc_core.h"

namespace av2enc::m2::multiref {
namespace {

using adv::CompoundType;

// Same block-size limits as the MC reference (M2 codes at most 64x64 leaves).
constexpr int kMaxBlockDim = mc::kMaxBlockDim;
constexpr int kMaxIntermediateRows = mc::kMaxIntermediateRows;

bool plane_rect_ok(const mc::RefFrame& ref, int plane,
                   const mc::BlockRect& rect) {
  return plane >= 0 && plane < 3 && rect.w > 0 && rect.h > 0 &&
         rect.w <= kMaxBlockDim && rect.h <= kMaxBlockDim && rect.x >= 0 &&
         rect.y >= 0 && rect.x + rect.w <= ref.plane_width(plane) &&
         rect.y + rect.h <= ref.plane_height(plane);
}

}  // namespace

int cwp_weight_from_index(int index, bool same_side) {
  if (index < 0 || index >= kMaxCwpNum) return kCwpEqual;
  return adv::cwp_weighting_factor(index, same_side);
}

int cwp_index_from_weight(int weight, bool same_side) {
  for (int i = 0; i < kMaxCwpNum; ++i) {
    if (adv::cwp_weighting_factor(i, same_side) == weight) return i;
  }
  return -1;
}

bool wedge_index_to_syntax(int wedge_index, WedgeSyntax& out) {
  if (wedge_index < 0 || wedge_index >= adv::kWedgeTypes) return false;
  for (int angle = 0; angle < adv::kWedgeAngles; ++angle) {
    for (int dist = 0; dist < adv::kNumWedgeDist; ++dist) {
      if (adv::tables::kWedgeAngleDist2Index[angle][dist] == wedge_index) {
        out.quad = angle / 5;
        out.angle = angle % 5;
        out.dist = dist;
        return true;
      }
    }
  }
  return false;
}

int wedge_syntax_to_index(int quad, int angle, int dist) {
  if (quad < 0 || quad > 3 || angle < 0 || angle > 4 || dist < 0 ||
      dist > 3) {
    return -1;
  }
  const int wedge_angle = quad * 5 + angle;
  return adv::tables::kWedgeAngleDist2Index[wedge_angle][dist];
}

int wedge_syntax_bits(int wedge_index) {
  WedgeSyntax syntax;
  if (!wedge_index_to_syntax(wedge_index, syntax)) return 0;
  // read_wedge_mode(): wedge_quad (4 symbols = 2 bits), wedge_angle
  // (5 symbols = 3 bits), then dist1 (4 symbols) or dist2 (3 symbols) - both
  // are 2 bits.  The raw wedge_sign bit is counted by compound_mode_bits().
  return 2 + 3 + 2;
}

bool same_mode(const CompoundModeParams& a, const CompoundModeParams& b) {
  return a.type == b.type && a.cwp_index == b.cwp_index &&
         a.cwp_weight == b.cwp_weight && a.wedge_index == b.wedge_index &&
         a.wedge_sign == b.wedge_sign && a.mask_type == b.mask_type &&
         a.use_imp_mask == b.use_imp_mask;
}

int compound_mode_bits(const CompoundModeParams& params, bool same_side,
                       bool include_comp_mode) {
  int bits = 0;
  if (include_comp_mode) bits += 1;  // comp_mode (reference_select == 1)
  if (params.type == CompoundType::kAverage) {
    if (!same_side) bits += 1;  // is_joint
    bits += 4;                  // compound_mode_non_joint (5 symbols)
    bits += 1;                  // comp_group_idx = 0 (AVERAGE)
    if (params.cwp_weight != kCwpEqual) {
      const int index = cwp_index_from_weight(params.cwp_weight, same_side);
      bits += index < 0 ? 0 : index;  // cwp_idx: index 1-bits, last implied
    }
    // `use_imp_mask` is a prediction derivation, not a syntax shortcut.
    return bits;
  }
  // Masked compound: comp_group_idx then compound_type.
  if (!same_side) bits += 1;
  bits += 4;  // compound_mode_non_joint
  bits += 1;  // comp_group_idx = 1
  bits += 2;  // compound_type (3 symbols: WEDGE/DIFFWTD/AVERAGE)
  if (params.type == CompoundType::kWedge) {
    bits += 1;  // wedge_sign
    bits += wedge_syntax_bits(params.wedge_index);
  } else {
    bits += 1;  // mask_type
  }
  return bits;
}

bool predict_plane_compound_1_16(const mc::RefFrame& ref, int plane,
                                 const mc::BlockRect& rect, mc::Mv mv,
                                 mc::InterpFilter filter, std::int16_t* dst,
                                 int dst_stride, std::string& error) {
  error.clear();
  if (dst == nullptr || dst_stride < rect.w) {
    error = "compound prediction destination too small";
    return false;
  }
  if (!plane_rect_ok(ref, plane, rect)) {
    error = "compound prediction block outside the reference plane";
    return false;
  }
  const int sub_x = plane > 0 ? 1 : 0;
  const int sub_y = plane > 0 ? 1 : 0;
  const mc::McSampleLocation loc =
      mc::mc_derive_sample_location(rect.x, rect.y, mv, sub_x, sub_y);
  const int intermediate_rows = mc::mc_intermediate_rows(rect.h, loc.step_y);
  if (intermediate_rows > kMaxIntermediateRows || rect.w > kMaxBlockDim) {
    error = "compound prediction block too large";
    return false;
  }
  const int last_x = ref.plane_width(plane) - 1;
  const int last_y = ref.plane_height(plane) - 1;
  const mc::InterpFilter filter_h = mc::mc_pass_filter(filter, rect.w <= 4);
  const mc::InterpFilter filter_v = mc::mc_pass_filter(filter, rect.h <= 4);
  const std::int16_t* taps_h =
      &mc::kSubpelFilters[static_cast<int>(filter_h)][0][0];
  const std::int16_t* taps_v =
      &mc::kSubpelFilters[static_cast<int>(filter_v)][0][0];

  const std::uint8_t* src = ref.plane[plane].data;
  const int stride = ref.plane[plane].stride;

  std::vector<std::int16_t> intermediate(
      static_cast<std::size_t>(intermediate_rows) *
      static_cast<std::size_t>(rect.w));

  for (int r = 0; r < intermediate_rows; ++r) {
    const int row = mc::mc_clip3(0, last_y, (loc.start_y >> mc::kScaleSubpelBits) + r - 3);
    for (int c = 0; c < rect.w; ++c) {
      const std::int32_t p_x = loc.start_x + loc.step_x * c;
      const int col = (p_x >> mc::kScaleSubpelBits) - 3;
      const int phase = static_cast<int>((p_x >> 6) & mc::kSubpelMask);
      const std::int32_t hsum =
          mc::mc_horizontal_sum(src, stride, row, col, 0, last_x,
                                taps_h + phase * mc::kSubpelTaps);
      intermediate[static_cast<std::size_t>(r) *
                       static_cast<std::size_t>(rect.w) +
                   static_cast<std::size_t>(c)] =
          static_cast<std::int16_t>(mc::mc_round2(hsum, mc::kInterRound0));
    }
  }

  for (int r = 0; r < rect.h; ++r) {
    const std::int32_t p_y = (loc.start_y & 1023) + loc.step_y * r;
    const int idx = static_cast<int>(p_y >> mc::kScaleSubpelBits);
    const int phase = static_cast<int>((p_y >> 6) & mc::kSubpelMask);
    const std::int16_t* fv = taps_v + phase * mc::kSubpelTaps;
    std::int16_t* out_row =
        dst + static_cast<std::ptrdiff_t>(r) * dst_stride;
    for (int c = 0; c < rect.w; ++c) {
      std::int32_t sum = 0;
      for (int t = 0; t < 8; ++t) {
        sum += static_cast<std::int32_t>(fv[t]) *
               static_cast<std::int32_t>(
                   intermediate[static_cast<std::size_t>(idx + t) *
                                    static_cast<std::size_t>(rect.w) +
                                static_cast<std::size_t>(c)]);
      }
      // Compound prediction: InterRound1 = 7, InterPostRound = 4 later.
      out_row[c] = static_cast<std::int16_t>(
          mc::mc_round2(sum, adv::kInterRound1Compound));
    }
  }
  return true;
}

bool build_compound_prediction(const mc::RefFrame& ref0,
                               const mc::RefFrame& ref1,
                               const mc::BlockRect& luma_rect, int bsize,
                               mc::Mv mv0, mc::Mv mv1,
                               mc::InterpFilter filter, int mi_cols,
                               int mi_rows, CompoundPrediction& out,
                               std::string& error) {
  error.clear();
  out = CompoundPrediction{};
  if ((luma_rect.x & 1) != 0 || (luma_rect.y & 1) != 0 ||
      (luma_rect.w & 1) != 0 || (luma_rect.h & 1) != 0) {
    error = "compound 4:2:0 requires an even luma rectangle";
    return false;
  }
  if (bsize < 0 || !adv::bsize_has_wedge(bsize)) {
    // A block without wedge support is still valid for AVERAGE/DIFFWTD; the
    // bsize value itself is only used by the wedge mask.
  }
  out.bsize = bsize;
  out.luma_rect = luma_rect;
  out.mi_cols = mi_cols;
  out.mi_rows = mi_rows;
  const mc::RefFrame* refs[2] = {&ref0, &ref1};
  const mc::Mv mvs[2] = {mv0, mv1};
  for (int plane = 0; plane < 3; ++plane) {
    const int sub_x = plane > 0 ? 1 : 0;
    const int sub_y = plane > 0 ? 1 : 0;
    const mc::BlockRect rect{luma_rect.x >> sub_x, luma_rect.y >> sub_y,
                             luma_rect.w >> sub_x, luma_rect.h >> sub_y};
    for (int list = 0; list < 2; ++list) {
      PredPlane16& dst = out.pred[list][plane];
      dst.width = rect.w;
      dst.height = rect.h;
      dst.stride = rect.w;
      dst.samples.resize(static_cast<std::size_t>(rect.w) *
                         static_cast<std::size_t>(rect.h));
      if (!predict_plane_compound_1_16(*refs[list], plane, rect, mvs[list],
                                       filter, dst.samples.data(), dst.stride,
                                       error)) {
        return false;
      }
      const mc::McSampleLocation loc = mc::mc_derive_sample_location(
          rect.x, rect.y, mvs[list], sub_x, sub_y);
      out.ref_start_x[list][plane] = loc.start_x >> mc::kScaleSubpelBits;
      out.ref_start_y[list][plane] = loc.start_y >> mc::kScaleSubpelBits;
    }
  }
  return true;
}

bool implicit_mask_eligible(bool seq_enable_imp_msk_bld,
                            const CompoundModeParams& params,
                            bool global_global_mv, bool refs_scaled,
                            bool compound_warp) {
  return seq_enable_imp_msk_bld && params.type == CompoundType::kAverage &&
         params.cwp_weight == kCwpEqual && !global_global_mv && !refs_scaled &&
         !compound_warp;
}

bool apply_compound_prediction(const CompoundPrediction& prediction,
                               const CompoundModeParams& params,
                               std::uint8_t* dst[3], int dst_stride[3],
                               std::string& error) {
  error.clear();
  if (dst == nullptr || dst[0] == nullptr || dst_stride[0] <= 0) {
    error = "compound destination missing";
    return false;
  }
  if (prediction.bsize < 0 || prediction.bsize >= adv::tables::kBlockSizes) {
    error = "invalid block size";
    return false;
  }
  const int luma_w = prediction.luma_rect.w;
  if (params.type == CompoundType::kWedge &&
      !adv::bsize_has_wedge(prediction.bsize)) {
    error = "wedge mode is not signalable for this block size";
    return false;
  }
  // The luma mask is generated by the plane-0 job and subsampled by chroma.
  std::vector<std::uint8_t> luma_mask(
      static_cast<std::size_t>(luma_w) *
      static_cast<std::size_t>(prediction.luma_rect.h));

  for (int plane = 0; plane < 3; ++plane) {
    const int sub_x = plane > 0 ? 1 : 0;
    const int sub_y = plane > 0 ? 1 : 0;
    const PredPlane16& p0 = prediction.pred[0][plane];
    const PredPlane16& p1 = prediction.pred[1][plane];
    if (p0.width <= 0 || p0.height <= 0 || p1.width != p0.width ||
        p1.height != p0.height) {
      error = "compound prediction planes are inconsistent";
      return false;
    }
    if (dst[plane] == nullptr || dst_stride[plane] <= 0) {
      error = "compound destination plane missing";
      return false;
    }
    adv::CompoundJob job{};
    job.plane = plane;
    job.sub_x = sub_x;
    job.sub_y = sub_y;
    job.bsize = prediction.bsize;
    job.bit_depth = 8;
    job.rect = adv::BlockRect{prediction.luma_rect.x >> sub_x,
                             prediction.luma_rect.y >> sub_y,
                             p0.width, p0.height};
    job.p0 = p0.samples.data();
    job.p0_stride = p0.stride;
    job.p1 = p1.samples.data();
    job.p1_stride = p1.stride;
    job.dst = dst[plane];
    job.dst_stride = dst_stride[plane];
    job.type = params.type;
    job.cwp_weight = params.cwp_weight;
    job.wedge_index = params.wedge_index;
    job.wedge_sign = params.wedge_sign;
    job.mask_type = params.mask_type;
    job.mi_cols = prediction.mi_cols;
    job.mi_rows = prediction.mi_rows;
    job.ref_start_x[0] = prediction.ref_start_x[0][plane];
    job.ref_start_x[1] = prediction.ref_start_x[1][plane];
    job.ref_start_y[0] = prediction.ref_start_y[0][plane];
    job.ref_start_y[1] = prediction.ref_start_y[1][plane];
    job.imp_mask_eligible =
        params.use_imp_mask && params.type == CompoundType::kAverage ? 1 : 0;
    if (plane == 0) {
      job.mask_out = luma_mask.data();
      job.mask_out_stride = luma_w;
    } else if (params.type == CompoundType::kWedge ||
               params.type == CompoundType::kDiffWtd) {
      job.luma_mask = luma_mask.data();
      job.luma_mask_stride = luma_w;
    }
    if (!adv::apply_compound(job)) {
      error = "adv::apply_compound rejected the job";
      return false;
    }
  }
  return true;
}

std::int64_t compound_luma_sse(const CompoundPrediction& prediction,
                               const CompoundModeParams& params,
                               const std::uint8_t* src, int src_stride,
                               std::uint8_t* scratch, int scratch_stride) {
  if (src == nullptr || scratch == nullptr || src_stride <= 0 ||
      scratch_stride <= 0) {
    return std::numeric_limits<std::int64_t>::max();
  }
  const PredPlane16& p0 = prediction.pred[0][0];
  const PredPlane16& p1 = prediction.pred[1][0];
  if (p0.width <= 0 || p0.height <= 0 || p1.width != p0.width ||
      p1.height != p0.height) {
    return std::numeric_limits<std::int64_t>::max();
  }
  if (params.type == adv::CompoundType::kWedge &&
      !adv::bsize_has_wedge(prediction.bsize)) {
    return std::numeric_limits<std::int64_t>::max();
  }
  // The predictions are block-local, so run the blend into a block-local
  // scratch with a (0, 0) rectangle.  `source` is the block origin.
  adv::CompoundJob job{};
  job.plane = 0;
  job.sub_x = 0;
  job.sub_y = 0;
  job.bsize = prediction.bsize;
  job.bit_depth = 8;
  job.rect = adv::BlockRect{0, 0, prediction.luma_rect.w,
                            prediction.luma_rect.h};
  job.p0 = p0.samples.data();
  job.p0_stride = p0.stride;
  job.p1 = p1.samples.data();
  job.p1_stride = p1.stride;
  job.dst = scratch;
  job.dst_stride = scratch_stride;
  job.type = params.type;
  job.cwp_weight = params.cwp_weight;
  job.wedge_index = params.wedge_index;
  job.wedge_sign = params.wedge_sign;
  job.mask_type = params.mask_type;
  job.mi_cols = prediction.mi_cols;
  job.mi_rows = prediction.mi_rows;
  job.ref_start_x[0] = prediction.ref_start_x[0][0];
  job.ref_start_x[1] = prediction.ref_start_x[1][0];
  job.ref_start_y[0] = prediction.ref_start_y[0][0];
  job.ref_start_y[1] = prediction.ref_start_y[1][0];
  job.imp_mask_eligible =
      params.use_imp_mask && params.type == adv::CompoundType::kAverage ? 1 : 0;
  if (!adv::apply_compound(job)) {
    return std::numeric_limits<std::int64_t>::max();
  }
  return adv::block_sse(scratch, scratch_stride, 0, 0, src, src_stride, 0, 0,
                        prediction.luma_rect.w, prediction.luma_rect.h);
}

}  // namespace av2enc::m2::multiref

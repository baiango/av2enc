// AV2 §7.13.3.18 inter motion compensation — CPU reference implementation.

#include "codec/inter/mc/mc.h"

#include <array>
#include <cstddef>

namespace av2enc::inter::mc {
namespace {

constexpr int kMaxIntermediate =
    kMaxIntermediateRows * kMaxBlockDim;  // int16 samples, see mc_core.h

bool valid_rect(const BlockRect& rect, int width, int height) {
  if (rect.w <= 0 || rect.h <= 0 || rect.x < 0 || rect.y < 0) {
    return false;
  }
  if (rect.x + rect.w > width || rect.y + rect.h > height) {
    return false;
  }
  return true;
}

McStatus validate_plane(const RefFrame& ref, int plane, const BlockRect& rect,
                        InterpFilter filter, std::uint8_t* dst,
                        int dst_stride) {
  if (plane < 0 || plane > 2) {
    return McStatus::kInvalidArgument;
  }
  const int filter_index = static_cast<int>(filter);
  if (filter_index < 0 || filter_index >= kSubpelFilterTypes) {
    return McStatus::kInvalidArgument;
  }
  if (ref.mi_cols <= 0 || ref.mi_rows <= 0) {
    return McStatus::kInvalidArgument;
  }
  if (ref.plane[plane].data == nullptr) {
    return McStatus::kInvalidArgument;
  }
  const int plane_w = ref.plane_width(plane);
  const int plane_h = ref.plane_height(plane);
  if (ref.plane[plane].stride < plane_w) {
    return McStatus::kInvalidArgument;
  }
  if (!valid_rect(rect, plane_w, plane_h)) {
    return McStatus::kInvalidArgument;
  }
  if (rect.w > kMaxBlockDim || rect.h > kMaxBlockDim) {
    return McStatus::kBlockTooLarge;
  }
  if (dst == nullptr || dst_stride < rect.w) {
    return McStatus::kInvalidArgument;
  }
  return McStatus::kOk;
}

Mv clamp_mv_bounds(Mv mv) {
  mv.y = mc_clip3(kMvLow + 1, kMvUpp - 1, mv.y);
  mv.x = mc_clip3(kMvLow + 1, kMvUpp - 1, mv.x);
  return mv;
}

// §5.20.7.13 mv_clamp_to_integer.
std::int32_t mv_clamp_to_integer(std::int32_t v) {
  if (v < kMvLow + 1) {
    return kMvLow + 8;
  }
  if (v > kMvUpp - 1) {
    return kMvUpp - 8;
  }
  return v;
}

bool mi_inside(int row, int col, const MvStackParams& p) {
  return row >= 0 && col >= 0 && row < p.mi_rows && col < p.mi_cols;
}

const MiInfo& mi_at(const MvStackParams& p, int row, int col) {
  return p.mi[static_cast<std::size_t>(row) *
                  static_cast<std::size_t>(p.mi_cols) +
              static_cast<std::size_t>(col)];
}

// One entry of the spatial scan.  `mi` is the current block's MI position.
struct StackBuilder {
  const MvStackParams& p;
  MvStack* stack;
  int prune_count = 0;

  bool same_mv(Mv a, Mv b) const { return a == b; }

  void search_stack(Mv cand, int mv_row, int mv_col, int weight) {
    int found = -1;
    if (prune_count < kMaxPrNum) {
      for (int i = 0; i < stack->count; ++i) {
        ++prune_count;
        if (same_mv(cand, stack->cand[i].mv[0])) {
          found = i;
          break;
        }
      }
    }
    if (found >= 0) {
      stack->cand[found].weight += weight;
      return;
    }
    if (stack->count < kMaxRefMvStackSize) {
      StackCandidate& c = stack->cand[stack->count];
      c.mv[0] = cand;
      c.mv[1] = cand;
      c.weight = weight;
      c.row_offset = mv_row - p.mi_row;
      c.col_offset = mv_col - p.mi_col;
      ++stack->count;
    }
  }

  // §7.12.2.10 Add reference motion vector (single-reference case).
  void add_reference_mv(int mv_row, int mv_col, int weight) {
    const MiInfo& m = mi_at(p, mv_row, mv_col);
    if (!m.is_inter) {
      return;
    }
    for (int list = 0; list < 2; ++list) {
      if (m.ref[list] != p.ref_frame) {
        continue;
      }
      const Mv cand{m.mv[list][0], m.mv[list][1]};
      search_stack(cand, mv_row, mv_col, weight);
    }
  }

  void scan_point(int delta_row, int delta_col) {
    int mv_row = p.mi_row + delta_row;
    int mv_col = p.mi_col + delta_col;
    const bool sb_border = (p.mi_row & (p.sb_h4 - 1)) == 0;
    if (delta_row < 0 && sb_border) {
      mv_col = (mv_col >> 1) << 1;
    }
    const int adj_delta_col = mv_col - p.mi_col;
    if (!mi_inside(mv_row, mv_col, p)) {
      return;
    }
    const MiInfo& m = mi_at(p, mv_row, mv_col);
    if (m.ref[0] == kRefNone) {  // not yet coded
      return;
    }
    int weight = 1;
    if (delta_row == -1 && adj_delta_col == -1) {
      weight = 0;
    } else if (adj_delta_col < -1) {
      weight = 0;
    }
    add_reference_mv(mv_row, mv_col, weight);
  }

  void scan_point_if_valid(int delta_row, int delta_col) {
    const int mv_row = p.mi_row + delta_row;
    const int mv_col = p.mi_col + delta_col;
    if (!mi_inside(mv_row, mv_col, p)) {
      return;
    }
    if (p.mi_col - 1 < 0) {
      return;
    }
    if (mi_at(p, mv_row, mv_col).col_base ==
        mi_at(p, mv_row, p.mi_col - 1).col_base) {
      return;
    }
    scan_point(delta_row, delta_col);
  }

  void scan_col(int delta_col) {
    delta_col += p.mi_col & 1;
    scan_point_if_valid(p.bh4 - 1, delta_col);
    if (p.bh4 > 1) {
      scan_point_if_valid(0, delta_col);
    }
  }

  void insert_mvp_candidate(int y_cand, int x_cand) {
    if (y_cand >= stack->count || x_cand >= stack->count) {
      return;
    }
    const Mv cand{stack->cand[y_cand].mv[0].y, stack->cand[x_cand].mv[0].x};
    if (prune_count < kMaxPrNum) {
      for (int i = 0; i < stack->count; ++i) {
        ++prune_count;
        if (same_mv(cand, stack->cand[i].mv[0])) {
          return;
        }
      }
    }
    if (stack->count >= kMaxRefMvStackSize) {
      return;
    }
    StackCandidate& c = stack->cand[stack->count];
    c.mv[0] = cand;
    c.mv[1] = cand;
    c.weight = 1;
    c.row_offset = 0;
    c.col_offset = 0;
    ++stack->count;
  }

  void sort_nearest() {
    const int end = stack->num_nearest;
    for (int start = 0; start + 1 < end; ++start) {
      int max_weight = stack->cand[start].weight;
      int max_idx = start;
      for (int idx = start + 1; idx < end; ++idx) {
        if (max_weight < stack->cand[idx].weight) {
          max_weight = stack->cand[idx].weight;
          max_idx = idx;
        }
      }
      if (max_idx != start) {
        const StackCandidate tmp = stack->cand[start];
        stack->cand[start] = stack->cand[max_idx];
        stack->cand[max_idx] = tmp;
      }
    }
  }
};

}  // namespace

const char* mc_status_string(McStatus status) {
  switch (status) {
    case McStatus::kOk:
      return "ok";
    case McStatus::kInvalidArgument:
      return "invalid argument";
    case McStatus::kUnsupportedSubsampling:
      return "unsupported chroma subsampling";
    case McStatus::kBlockTooLarge:
      return "block too large";
  }
  return "unknown";
}

McStatus predict_plane(const RefFrame& ref, int plane, const BlockRect& rect,
                       Mv mv, InterpFilter filter, std::uint8_t* dst,
                       int dst_stride) {
  const McStatus status =
      validate_plane(ref, plane, rect, filter, dst, dst_stride);
  if (status != McStatus::kOk) {
    return status;
  }

  const int sub_x = plane > 0 ? 1 : 0;
  const int sub_y = plane > 0 ? 1 : 0;
  const McSampleLocation loc =
      mc_derive_sample_location(rect.x, rect.y, mv, sub_x, sub_y);
  const int intermediate_rows = mc_intermediate_rows(rect.h, loc.step_y);
  if (intermediate_rows > kMaxIntermediateRows ||
      rect.w > kMaxBlockDim) {
    return McStatus::kBlockTooLarge;
  }

  const int first_x = 0;
  const int first_y = 0;
  const int last_x = (ref.plane_width(plane)) - 1;
  const int last_y = (ref.plane_height(plane)) - 1;

  const InterpFilter filter_h = mc_pass_filter(filter, rect.w <= 4);
  const InterpFilter filter_v = mc_pass_filter(filter, rect.h <= 4);
  const std::int16_t* taps_h = &kSubpelFilters[static_cast<int>(filter_h)][0][0];
  const std::int16_t* taps_v = &kSubpelFilters[static_cast<int>(filter_v)][0][0];

  const std::uint8_t* src = ref.plane[plane].data;
  const int stride = ref.plane[plane].stride;

  std::array<std::int16_t, kMaxIntermediate> intermediate{};

  // Horizontal pass, storable intermediate (spec §7.13.3.18 first loop).
  for (int r = 0; r < intermediate_rows; ++r) {
    const int row =
        mc_clip3(first_y, last_y, (loc.start_y >> kScaleSubpelBits) + r - 3);
    for (int c = 0; c < rect.w; ++c) {
      const std::int32_t p_x = loc.start_x + loc.step_x * c;
      const int col = (p_x >> kScaleSubpelBits) - 3;
      const int phase = static_cast<int>((p_x >> 6) & kSubpelMask);
      const std::int32_t hsum = mc_horizontal_sum(
          src, stride, row, col, first_x, last_x,
          taps_h + phase * kSubpelTaps);
      intermediate[static_cast<std::size_t>(r) *
                       static_cast<std::size_t>(rect.w) +
                   static_cast<std::size_t>(c)] =
          static_cast<std::int16_t>(mc_round2(hsum, kInterRound0));
    }
  }

  // Vertical pass.
  for (int r = 0; r < rect.h; ++r) {
    const std::int32_t p_y = (loc.start_y & 1023) + loc.step_y * r;
    const int idx = static_cast<int>(p_y >> kScaleSubpelBits);
    const int phase = static_cast<int>((p_y >> 6) & kSubpelMask);
    const std::int16_t* fv = taps_v + phase * kSubpelTaps;
    for (int c = 0; c < rect.w; ++c) {
      std::int32_t sum = 0;
      for (int t = 0; t < 8; ++t) {
        sum += static_cast<std::int32_t>(fv[t]) *
               static_cast<std::int32_t>(
                   intermediate[static_cast<std::size_t>(idx + t) *
                                    static_cast<std::size_t>(rect.w) +
                                static_cast<std::size_t>(c)]);
      }
      dst[static_cast<std::size_t>(r) * static_cast<std::size_t>(dst_stride) +
          static_cast<std::size_t>(c)] =
          mc_clip1(mc_round2(sum, kInterRound1));
    }
  }
  return McStatus::kOk;
}

McStatus predict_block_420(const RefFrame& ref, const BlockRect& luma_rect,
                           Mv mv, InterpFilter filter, MutablePlane dst[3]) {
  if (dst == nullptr) {
    return McStatus::kInvalidArgument;
  }
  if ((luma_rect.x & 1) != 0 || (luma_rect.y & 1) != 0 ||
      (luma_rect.w & 1) != 0 || (luma_rect.h & 1) != 0) {
    return McStatus::kUnsupportedSubsampling;
  }
  const BlockRect chroma_rect{luma_rect.x >> 1, luma_rect.y >> 1,
                              luma_rect.w >> 1, luma_rect.h >> 1};
  McStatus status =
      predict_plane(ref, 0, luma_rect, mv, filter, dst[0].data, dst[0].stride);
  if (status != McStatus::kOk) {
    return status;
  }
  for (int plane = 1; plane <= 2; ++plane) {
    status = predict_plane(ref, plane, chroma_rect, mv, filter, dst[plane].data,
                           dst[plane].stride);
    if (status != McStatus::kOk) {
      return status;
    }
  }
  return McStatus::kOk;
}

McStatus build_mv_stack(const MvStackParams& params, MvStack* stack) {
  if (stack == nullptr || params.mi == nullptr || params.mi_cols <= 0 ||
      params.mi_rows <= 0 || params.bw4 <= 0 || params.bh4 <= 0 ||
      params.sb_h4 <= 0 || !is_inter_ref(params.ref_frame) ||
      params.use_intrabc) {
    return McStatus::kInvalidArgument;
  }
  if ((params.sb_h4 & (params.sb_h4 - 1)) != 0) {
    return McStatus::kInvalidArgument;
  }
  if (params.mi_row < 0 || params.mi_col < 0 ||
      params.mi_row >= params.mi_rows || params.mi_col >= params.mi_cols) {
    return McStatus::kInvalidArgument;
  }

  *stack = MvStack{};
  StackBuilder b{params, stack};
  const bool sb_border = (params.mi_row & (params.sb_h4 - 1)) == 0;

  // Steps 7-13: right/bottom, above/right, left, above, below-left,
  // above-right, above-left.  Temporal scan (steps 5-6, 12) is out of scope.
  b.scan_point(params.bh4 - 1, -1);
  const int above_right_col = params.bw4 - 1 - (sb_border ? 1 : 0);
  b.scan_point(-1, above_right_col > 0 ? above_right_col : 0);
  if (params.bh4 >= 2) {
    b.scan_point(0, -1);
  }
  if (params.bw4 >= (sb_border ? 4 : 2)) {
    b.scan_point(-1, 0);
  }
  if (params.bh4 <= 16) {
    b.scan_point(params.bh4, -1);
  }
  if (params.bw4 <= 16) {
    b.scan_point(-1, sb_border ? ((params.bw4 > 2) ? params.bw4 : 2)
                               : params.bw4);
  }
  b.scan_point(-1, -1);

  // Step 14: boundary between the immediate neighbourhood and the extra scan.
  stack->num_nearest = stack->count;

  // Step 15.
  b.scan_col(-3);

  // Steps 16-17.
  const bool use_sort =
      params.reorder == DrlReorder::kAlways ||
      (params.reorder == DrlReorder::kConstraint && stack->num_nearest >= 4);
  if (use_sort) {
    b.sort_nearest();
  }

  // Step 20: derived-SMVP stack is empty in this subset.

  // Step 21 (start): §7.12.2.20 clamps the existing candidates before the
  // global-MV duplicate check and the MVP combinations.
  for (int i = 0; i < stack->count; ++i) {
    for (int list = 0; list < 2; ++list) {
      stack->cand[i].mv[list] =
          clamp_mv_to_picture(clamp_mv_bounds(stack->cand[i].mv[list]),
                              params);
    }
  }

  // Step 21: extra search - add the global MV if it is not already present.
  if (stack->count < kMaxRefMvStackSize) {
    bool found = false;
    if (b.prune_count < kMaxPrNum) {
      for (int i = 0; i < stack->count; ++i) {
        ++b.prune_count;
        if (params.global_mv == stack->cand[i].mv[0]) {
          found = true;
          break;
        }
      }
    }
    if (!found) {
      StackCandidate& c = stack->cand[stack->count];
      c.mv[0] = params.global_mv;
      c.mv[1] = params.global_mv;
      c.weight = 1;
      c.row_offset = 0;
      c.col_offset = 0;
      ++stack->count;
    }
  }

  // Step 21 (continued): MVP combination candidates.
  const int num = stack->count;
  if (num > 1) {
    b.insert_mvp_candidate(0, 1);
    b.insert_mvp_candidate(1, 0);
  }
  if (num > 2) {
    b.insert_mvp_candidate(0, 2);
    b.insert_mvp_candidate(2, 0);
    b.insert_mvp_candidate(1, 2);
    b.insert_mvp_candidate(2, 1);
  }

  // Step 22: clamping.
  for (int i = 0; i < stack->count; ++i) {
    for (int list = 0; list < 2; ++list) {
      stack->cand[i].mv[list] =
          clamp_mv_to_picture(clamp_mv_bounds(stack->cand[i].mv[list]),
                              params);
    }
  }
  return McStatus::kOk;
}

Mv clamp_mv_to_picture(Mv mv, const MvStackParams& params) {
  const std::int32_t low_y =
      -(params.mi_row + params.bh4) * kMiSize * 8 - kMvBorder;
  const std::int32_t high_y =
      (params.mi_rows - params.mi_row) * kMiSize * 8 + kMvBorder;
  const std::int32_t low_x =
      -(params.mi_col + params.bw4) * kMiSize * 8 - kMvBorder;
  const std::int32_t high_x =
      (params.mi_cols - params.mi_col) * kMiSize * 8 + kMvBorder;
  mv.y = mc_clip3(low_y, high_y, mv.y);
  mv.x = mc_clip3(low_x, high_x, mv.x);
  return mv;
}

McStatus resolve_mode(const MvStack& stack, const MvStackParams& params,
                      const ModeDecision& decision, ResolvedMode* out) {
  if (out == nullptr) {
    return McStatus::kInvalidArgument;
  }
  if (stack.count <= 0 || decision.drl_index < 0 ||
      decision.drl_index >= stack.count) {
    return McStatus::kInvalidArgument;
  }
  if (decision.ref_frame != params.ref_frame) {
    return McStatus::kInvalidArgument;
  }

  ResolvedMode result;
  result.drl_index = decision.drl_index;
  result.ref_frame = decision.ref_frame;
  result.filter = decision.filter;

  const Mv pred = stack.cand[decision.drl_index].mv[0];
  result.pred_mv = pred;

  switch (decision.mode) {
    case InterMode::kSkipMerge:
    case InterMode::kNearmv:
      // assign_mv applies mv_clamp_to_integer to PredMvs + 0 as well.
      result.mv.y = mv_clamp_to_integer(pred.y);
      result.mv.x = mv_clamp_to_integer(pred.x);
      break;
    case InterMode::kNewmv: {
      Mv mvd = decision.mvd;
      if (decision.me_absolute) {
        mvd.y = decision.me_mv.y - pred.y;
        mvd.x = decision.me_mv.x - pred.x;
      }
      result.mvd = mvd;
      result.mv.y = mv_clamp_to_integer(pred.y + mvd.y);
      result.mv.x = mv_clamp_to_integer(pred.x + mvd.x);
      break;
    }
    default:
      return McStatus::kInvalidArgument;
  }

  *out = result;
  return McStatus::kOk;
}

std::int64_t sad_block(const std::uint8_t* a, int a_stride,
                       const std::uint8_t* b, int b_stride, int w, int h) {
  std::int64_t sad = 0;
  for (int r = 0; r < h; ++r) {
    for (int c = 0; c < w; ++c) {
      const int av = a[static_cast<std::size_t>(r) *
                           static_cast<std::size_t>(a_stride) +
                       static_cast<std::size_t>(c)];
      const int bv = b[static_cast<std::size_t>(r) *
                           static_cast<std::size_t>(b_stride) +
                       static_cast<std::size_t>(c)];
      sad += (av > bv ? av - bv : bv - av);
    }
  }
  return sad;
}

McStatus select_merge_candidate(const MvStack& stack, const RefFrame& ref,
                                std::int8_t ref_frame,
                                const BlockRect& luma_rect,
                                InterpFilter filter, const std::uint8_t* src,
                                int src_stride, std::uint8_t* scratch,
                                int scratch_stride, MergeScore* out) {
  if (out == nullptr || src == nullptr || scratch == nullptr ||
      src_stride < luma_rect.w || scratch_stride < luma_rect.w ||
      stack.count <= 0) {
    return McStatus::kInvalidArgument;
  }
  if (!is_inter_ref(ref_frame)) {
    return McStatus::kInvalidArgument;
  }
  MergeScore best;
  for (int i = 0; i < stack.count; ++i) {
    const McStatus status =
        predict_plane(ref, 0, luma_rect, stack.cand[i].mv[0], filter, scratch,
                      scratch_stride);
    if (status != McStatus::kOk) {
      return status;
    }
    const std::int64_t sad =
        sad_block(scratch, scratch_stride, src, src_stride, luma_rect.w,
                  luma_rect.h);
    if (best.drl_index < 0 || sad < best.sad) {
      best.drl_index = i;
      best.sad = sad;
    }
  }
  *out = best;
  return McStatus::kOk;
}

}  // namespace av2enc::inter::mc

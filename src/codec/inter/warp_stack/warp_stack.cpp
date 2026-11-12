#include "codec/inter/warp_stack/warp_stack.h"

#include <algorithm>
#include <cstring>

namespace av2enc::inter::warp {
namespace {

int floor_log2(int v) {
  int n = 0;
  while (v > 1) {
    v >>= 1;
    ++n;
  }
  return n;
}

std::int64_t clamp_i64(std::int64_t v) {
  if (v > 2147483647LL) return 2147483647LL;
  if (v < -2147483648LL) return -2147483648LL;
  return v;
}

int ls_product(int a, int b) { return ((a * b) >> 2) + (a + b); }

bool min_dim_at_least(const WarpMi& mi, int threshold) {
  const int w = std::max<int>(1, mi.w4) * kMiSize;
  const int h = std::max<int>(1, mi.h4) * kMiSize;
  return std::min(w, h) >= threshold;
}

// §7.12.2.13 get_mv() without TIP/global-motion syntax.  `ref_list` is -1 when
// the caller does not want the global-MV substitution.
inter::mc::Mv get_mv(const WarpStackParams& params, const WarpMi& mi,
                     int ref_list, int cand_list) {
  const bool global_mode =
      mi.y_mode == kYModeGlobalMv || mi.y_mode == kYModeGlobalGlobalMv;
  if (ref_list >= 0 && global_mode && params.global_mv_nonidentity &&
      mi.ref[cand_list] != kRefTip && min_dim_at_least(mi, 8)) {
    return params.global_mv[ref_list];
  }
  return inter::mc::Mv{mi.sub_mv[cand_list][0], mi.sub_mv[cand_list][1]};
}

void search_stack(const WarpStackParams& params, inter::mc::Mv cand, int mv_row,
                  int mv_col, int weight, WarpStack* stack, int* prune_count) {
  int found = -1;
  if (*prune_count < kMaxPrNum) {
    for (int i = 0; i < stack->count; ++i) {
      ++*prune_count;
      if (cand == stack->cand[i].mv[0]) {
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
    WarpCandidate& c = stack->cand[stack->count];
    c.mv[0] = cand;
    c.mv[1] = cand;
    c.weight = weight;
    c.row_offset = mv_row - params.mi_row;
    c.col_offset = mv_col - params.mi_col;
    c.cwp = kCwpEqual;
    ++stack->count;
  }
}

void add_global_candidate(const WarpStackParams& params, WarpStack* stack,
                          int* prune_count) {
  if (stack->count >= kMaxRefMvStackSize) return;
  bool found = false;
  if (*prune_count < kMaxPrNum) {
    for (int i = 0; i < stack->count; ++i) {
      ++*prune_count;
      if (params.global_mv[0] == stack->cand[i].mv[0]) {
        found = true;
        break;
      }
    }
  }
  if (found) return;
  WarpCandidate& c = stack->cand[stack->count];
  c.mv[0] = params.global_mv[0];
  c.mv[1] = params.global_mv[0];
  c.weight = 1;
  c.row_offset = 0;
  c.col_offset = 0;
  c.cwp = kCwpEqual;
  ++stack->count;
}

void insert_mvp_candidate(const WarpStackParams& /*params*/, int y_cand,
                          int x_cand, WarpStack* stack, int* prune_count) {
  if (y_cand >= stack->count || x_cand >= stack->count) return;
  if (stack->count >= kMaxRefMvStackSize) return;
  const inter::mc::Mv cand{stack->cand[y_cand].mv[0].y,
                           stack->cand[x_cand].mv[0].x};
  if (*prune_count < kMaxPrNum) {
    for (int i = 0; i < stack->count; ++i) {
      ++*prune_count;
      if (cand == stack->cand[i].mv[0]) return;
    }
  }
  WarpCandidate& c = stack->cand[stack->count];
  c.mv[0] = cand;
  c.mv[1] = cand;
  c.weight = 1;
  c.row_offset = 0;
  c.col_offset = 0;
  c.cwp = kCwpEqual;
  ++stack->count;
}

void scan_point_if_valid(const WarpStackParams& params, int delta_row,
                         int delta_col, WarpStack* stack, int* prune_count) {
  const int mv_row = params.mi_row + delta_row;
  const int mv_col = params.mi_col + delta_col;
  if (!params.grid().inside(mv_row, mv_col)) return;
  if (params.mi_col - 1 < 0) return;
  const int other_col = params.mi_col - 1;
  if (params.grid().at(mv_row, mv_col).col_base ==
      params.grid().at(mv_row, other_col).col_base) {
    return;
  }
  scan_point(params, delta_row, delta_col, stack, prune_count);
}

}  // namespace

const char* status_string(Status status) {
  switch (status) {
    case Status::kOk:
      return "ok";
    case Status::kInvalidArgument:
      return "invalid argument";
    case Status::kInvalidBlockSize:
      return "invalid block size";
    case Status::kUnsupportedCompound:
      return "compound stack is out of scope";
    case Status::kUnsupportedTemporal:
      return "temporal candidates are out of scope";
    case Status::kUnsupportedIntrabc:
      return "intra block copy is out of scope";
  }
  return "unknown";
}

// ---------------------------------------------------------------------------
// §7.11.4
// ---------------------------------------------------------------------------

void scan_point_warp_context(const WarpContextParams& params, int delta_row,
                             int delta_col, WarpContext* out) {
  if (out == nullptr) return;
  const int is_sb_border = (params.mi_row & (params.sb_h4 - 1)) == 0 ? 1 : 0;
  if (delta_row < 0 && is_sb_border) {
    delta_col -= params.mi_col & 1;
  }
  const int mv_row = params.mi_row + delta_row;
  const int mv_col = params.mi_col + delta_col;
  if (!params.grid.inside(mv_row, mv_col)) return;
  const WarpMi& n = params.grid.at(mv_row, mv_col);
  if (!n.written || !n.is_inter) return;

  if (n.ref[0] == params.ref_frame[0] || n.ref[1] == params.ref_frame[0]) {
    if (!out->warp_sample_found[0]) {
      out->extra.found = true;
      out->extra.delta_row = delta_row;
      out->extra.delta_col = delta_col;
    }
    out->warp_sample_found[0] = true;
    if (n.motion_mode >= kMotionLocalWarp) {
      ++out->warp_mv_count;
    }
  }
  if (n.ref[0] == params.ref_frame[1] || n.ref[1] == params.ref_frame[1]) {
    out->warp_sample_found[1] = true;
  }
}

WarpContext compute_warp_context(const WarpContextParams& params) {
  WarpContext out{};
  if (params.grid.mi == nullptr || params.grid.mi_cols <= 0 ||
      params.grid.mi_rows <= 0 || params.sb_h4 <= 0 ||
      (params.sb_h4 & (params.sb_h4 - 1)) != 0 || params.bw4 <= 0 ||
      params.bh4 <= 0) {
    return out;
  }
  const int sb = (params.mi_row & (params.sb_h4 - 1)) == 0 ? 1 : 0;
  scan_point_warp_context(params, params.bh4 - 1, -1, &out);
  scan_point_warp_context(params, -1, sb ? std::max(0, params.bw4 - 2)
                                         : params.bw4 - 1,
                          &out);
  scan_point_warp_context(params, 0, -1, &out);
  if (params.bw4 >= (sb ? 4 : 2)) {
    scan_point_warp_context(params, -1, 0, &out);
  }
  return out;
}

// ---------------------------------------------------------------------------
// §7.12.2.4 warp corner process
// ---------------------------------------------------------------------------

inter::mc::Mv warp_sub_block_mv(const ia::WarpParams& params, int x, int y,
                                int i8, int j8) {
  const int src_x = x + j8 * 8 + 4;
  const int src_y = y + i8 * 8 + 4;
  const std::int64_t dst_x = std::int64_t{params.p[2]} * src_x +
                             std::int64_t{params.p[3]} * src_y + params.p[0];
  const std::int64_t dst_y = std::int64_t{params.p[4]} * src_x +
                             std::int64_t{params.p[5]} * src_y + params.p[1];
  const std::int64_t shift = std::int64_t{src_y} << ia::kWarpedModelPrecBits;
  const std::int64_t shift_x = std::int64_t{src_x} << ia::kWarpedModelPrecBits;
  int mv_y = static_cast<int>(clamp_i64(
      ia::round2_signed64(dst_y - shift, ia::kWarpedModelPrecBits - 3)));
  int mv_x = static_cast<int>(clamp_i64(
      ia::round2_signed64(dst_x - shift_x, ia::kWarpedModelPrecBits - 3)));
  mv_y = ia::clip3(ia::kMvLow + 1, ia::kMvUpp - 1, mv_y);
  mv_x = ia::clip3(ia::kMvLow + 1, ia::kMvUpp - 1, mv_x);
  return inter::mc::Mv{mv_y, mv_x};
}

inter::mc::Mv get_warp_motion_vector_xy_pos(const ia::WarpParams& params,
                                            int pos_row, int pos_col, int mi_row,
                                            int mi_col, int bw4, int bh4,
                                            int mi_rows, int mi_cols) {
  const int y = pos_row * kMiSize;
  const int x = pos_col * kMiSize;
  const std::int64_t xc = std::int64_t{params.p[2]} * x +
                          std::int64_t{params.p[3]} * y + params.p[0] -
                          (std::int64_t{x} << ia::kWarpedModelPrecBits);
  const std::int64_t yc = std::int64_t{params.p[4]} * x +
                          std::int64_t{params.p[5]} * y + params.p[1] -
                          (std::int64_t{y} << ia::kWarpedModelPrecBits);
  int mv_y = static_cast<int>(
      clamp_i64(ia::round2_signed64(yc, ia::kWarpedModelPrecBits - 3)));
  int mv_x = static_cast<int>(
      clamp_i64(ia::round2_signed64(xc, ia::kWarpedModelPrecBits - 3)));
  mv_y = ia::clip3(ia::kMvLow + 1, ia::kMvUpp - 1, mv_y);
  mv_x = ia::clip3(ia::kMvLow + 1, ia::kMvUpp - 1, mv_x);
  mv_y = ia::clamp_mv_row(mv_y, mi_row, bh4, mi_rows);
  mv_x = ia::clamp_mv_col(mv_x, mi_col, bw4, mi_cols);
  return inter::mc::Mv{mv_y, mv_x};
}

bool warp_corner_process(const WarpGrid& grid, int sb_h4, int mi_row, int mi_col,
                         int bw4, int bh4, std::int8_t ref_frame0, int delta_row,
                         int delta_col, int adjust_col, CornerResult* out) {
  if (out == nullptr || grid.mi == nullptr) return false;
  const int is_sb_border = (mi_row & (sb_h4 - 1)) == 0 ? 1 : 0;
  const int mv_row = mi_row + delta_row;
  int mv_col = mi_col + delta_col;
  delta_col += adjust_col;
  int mv_col2;
  if (delta_row < 0 && is_sb_border) {
    mv_col2 = (mi_col - (mi_col & 1)) + (delta_col - (delta_col & 1));
  } else {
    mv_col2 = mi_col + delta_col;
  }
  // The guard is written after `deltaCol += adjustCol` in §7.12.2.4.
  if (is_sb_border == 1 && delta_col == 0 && bw4 <= 2) return false;
  if (!grid.inside(mv_row, mv_col2)) return false;

  for (int ref = 0; ref <= 1; ++ref) {
    const WarpMi& n = grid.at(mv_row, mv_col2);
    if (!n.written || !n.is_inter) continue;
    if (n.ref[ref] != ref_frame0) continue;
    const int idx = out->corners_found;
    out->corner_pts[idx][0] = (mv_row + 1) * kMiSize;
    out->corner_pts[idx][1] = (mv_col + 1) * kMiSize;
    if (n.motion_mode >= kMotionLocalWarp) {
      if (ref > 0) return false;
      const inter::mc::Mv corner =
          get_warp_motion_vector_xy_pos(n.warp[ref], mv_row + 1, mv_col + 1,
                                        mi_row, mi_col, bw4, bh4, grid.mi_rows,
                                        grid.mi_cols);
      out->corner_mvs[idx][0] = corner.y;
      out->corner_mvs[idx][1] = corner.x;
    } else {
      out->corner_mvs[idx][0] = n.sub_mv[ref][0];
      out->corner_mvs[idx][1] = n.sub_mv[ref][1];
    }
    ++out->corners_found;
    return true;
  }
  return false;
}

CornerResult generate_points_from_corners(const WarpGrid& grid, int sb_h4,
                                          int mi_row, int mi_col, int bw4,
                                          int bh4, std::int8_t ref_frame0,
                                          int iter, WarpStack* stack) {
  CornerResult out{};
  if (grid.mi == nullptr || bw4 <= 0 || bh4 <= 0) return out;
  warp_corner_process(grid, sb_h4, mi_row, mi_col, bw4, bh4, ref_frame0, -1, -1,
                      iter, &out);
  warp_corner_process(grid, sb_h4, mi_row, mi_col, bw4, bh4, ref_frame0, -1,
                      bw4 - 1, iter, &out);
  warp_corner_process(grid, sb_h4, mi_row, mi_col, bw4, bh4, ref_frame0,
                      bh4 - 1, -1, 0, &out);
  if (out.corners_found != 3) {
    out.all_mvs_same = false;
    return out;
  }

  out.all_mvs_same = true;
  for (int n = 0; n < out.corners_found; ++n) {
    for (int c = 0; c < 2; ++c) {
      const std::int64_t value =
          (std::int64_t{out.corner_pts[n][c]} << ia::kWarpedModelPrecBits) +
          (std::int64_t{out.corner_mvs[n][c]} << kGmTransOnlyPrecDiff);
      out.ref_pts[n][c] = static_cast<int>(clamp_i64(value));
      if (out.corner_mvs[n][c] != out.corner_mvs[0][c]) {
        out.all_mvs_same = false;
      }
    }
  }
  if (out.all_mvs_same) return out;
  for (int n = 0; n < out.corners_found; ++n) {
    for (int c = 0; c < 2; ++c) {
      if (out.ref_pts[n][c] < 0) {
        out.negative_ref_pts = true;
        return out;
      }
    }
  }

  const int width_log2 = floor_log2(bw4 * kMiSize);
  const int height_log2 = floor_log2(bh4 * kMiSize);
  const int y0 = out.corner_pts[0][0];
  const int x0 = out.corner_pts[0][1];

  ia::WarpParams wmmat{};
  wmmat.p[0] = 0;
  wmmat.p[1] = 0;
  wmmat.p[2] = (out.ref_pts[1][1] - out.ref_pts[0][1]) >> width_log2;
  wmmat.p[4] = (out.ref_pts[1][0] - out.ref_pts[0][0]) >> width_log2;
  wmmat.p[3] = (out.ref_pts[2][1] - out.ref_pts[0][1]) >> height_log2;
  wmmat.p[5] = (out.ref_pts[2][0] - out.ref_pts[0][0]) >> height_log2;

  const std::int64_t wmmat0 = std::int64_t{out.ref_pts[0][1]} -
                              std::int64_t{wmmat.p[2]} * x0 -
                              std::int64_t{wmmat.p[3]} * y0;
  const std::int64_t wmmat1 = std::int64_t{out.ref_pts[0][0]} -
                              std::int64_t{wmmat.p[4]} * x0 -
                              std::int64_t{wmmat.p[5]} * y0;

  out.wmmat = ia::reduce_warp_model(wmmat);
  out.wmmat.p[0] = ia::clip3(-ia::kWarpedModelTransClamp,
                             ia::kWarpedModelTransClamp -
                                 (1 << ia::kWarpParamReduceBits),
                             static_cast<int>(clamp_i64(wmmat0)));
  out.wmmat.p[1] = ia::clip3(-ia::kWarpedModelTransClamp,
                             ia::kWarpedModelTransClamp -
                                 (1 << ia::kWarpParamReduceBits),
                             static_cast<int>(clamp_i64(wmmat1)));
  if (stack != nullptr) {
    insert_warp_candidate(out.wmmat, stack);
    out.inserted = true;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Warp parameter bank
// ---------------------------------------------------------------------------

void WarpParamBank::reset() {
  for (int i = 0; i < kWarpParamBankSize; ++i) params_[i] = ia::WarpParams{};
  start_ = 0;
  size_ = 0;
}

void WarpParamBank::record(const ia::WarpParams& params) {
  const int write = (start_ + size_) % kWarpParamBankSize;
  params_[write] = params;
  if (size_ == kWarpParamBankSize) {
    start_ = (start_ + 1) % kWarpParamBankSize;
  } else {
    ++size_;
  }
}

const ia::WarpParams& WarpParamBank::slot(int physical) const {
  const int wrapped =
      ((physical % kWarpParamBankSize) + kWarpParamBankSize) % kWarpParamBankSize;
  return params_[wrapped];
}

// ---------------------------------------------------------------------------
// §7.12.2.11 / §7.12.2.9
// ---------------------------------------------------------------------------

void insert_warp_candidate(const ia::WarpParams& params, WarpStack* stack) {
  if (stack == nullptr) return;
  if (stack->num_warp_found >= kMaxWarpRefCandidates) return;
  stack->warp_params[stack->num_warp_found] = params;
  ++stack->num_warp_found;
}

void add_warp_motion_vector(const WarpGrid& grid, int /*mi_row*/, int /*mi_col*/,
                            std::int8_t ref_frame0, bool derive_wrl, int mv_row,
                            int mv_col, WarpStack* stack) {
  if (stack == nullptr || !derive_wrl) return;
  if (!grid.inside(mv_row, mv_col)) return;
  const WarpMi& n = grid.at(mv_row, mv_col);
  if (!n.written || !n.is_inter) return;
  if (n.motion_mode < kMotionLocalWarp) return;
  if (n.ref[0] != ref_frame0) return;
  insert_warp_candidate(n.warp[0], stack);
}

// ---------------------------------------------------------------------------
// §7.12.2.6 scan point / §7.12.2.10 add reference MV
// ---------------------------------------------------------------------------

void add_reference_mv(const WarpStackParams& params, int mv_row, int mv_col,
                      int weight, WarpStack* stack, int* prune_count) {
  if (!params.grid().inside(mv_row, mv_col)) return;
  const WarpMi& m = params.grid().at(mv_row, mv_col);
  if (!m.written || !m.is_inter) return;
  for (int list = 0; list < 2; ++list) {
    if (m.ref[list] == params.ref_frame[0]) {
      const inter::mc::Mv cand = get_mv(params, m, 0, list);
      search_stack(params, cand, mv_row, mv_col, weight, stack, prune_count);
    }
  }
}

void scan_point(const WarpStackParams& params, int delta_row, int delta_col,
                WarpStack* stack, int* prune_count) {
  if (stack == nullptr || prune_count == nullptr) return;
  const int is_sb_border = (params.mi_row & (params.sb_h4 - 1)) == 0 ? 1 : 0;
  int mv_row = params.mi_row + delta_row;
  int mv_col = params.mi_col + delta_col;
  if (delta_row < 0 && is_sb_border) {
    mv_col = (mv_col >> 1) << 1;
    delta_col = mv_col - params.mi_col;
  }
  if (!params.grid().inside(mv_row, mv_col)) return;
  if (!params.grid().at(mv_row, mv_col).written) return;

  int weight = 1;
  if (delta_row == -1 && delta_col == -1) {
    weight = 0;
  } else if (delta_col < -1) {
    weight = 0;
  }

  // §7.12.2.6: the warp candidate is offered before the stack-full bail-out,
  // so a full MV stack does not suppress the warp stack.
  add_warp_motion_vector(params.grid(), params.mi_row, params.mi_col,
                         params.ref_frame[0], params.derive_wrl, mv_row, mv_col,
                         stack);
  if (stack->count >= kMaxRefMvStackSize) return;
  add_reference_mv(params, mv_row, mv_col, weight, stack, prune_count);
}

void scan_col(const WarpStackParams& params, int delta_col, WarpStack* stack,
              int* prune_count) {
  // §7.12.2.5: the offset is adjusted only for a 1-MI-wide block.
  if (params.bw4 == 1) delta_col += params.mi_col & 1;
  scan_point_if_valid(params, params.bh4 - 1, delta_col, stack, prune_count);
  if (params.bh4 > 1) {
    scan_point_if_valid(params, 0, delta_col, stack, prune_count);
  }
}

// ---------------------------------------------------------------------------
// §7.12.2.19 sorting (one move-to-front) / §7.12.2.23 clamping
// ---------------------------------------------------------------------------

void sort_stack_range(WarpStack* stack, int start, int end) {
  if (stack == nullptr || start < 0 || end > stack->count || start + 1 >= end) {
    return;
  }
  int max_weight = stack->cand[start].weight;
  int max_idx = start;
  for (int idx = start + 1; idx < end; ++idx) {
    if (max_weight < stack->cand[idx].weight) {
      max_weight = stack->cand[idx].weight;
      max_idx = idx;
    }
  }
  if (max_idx == start) return;
  const WarpCandidate tmp = stack->cand[start];
  stack->cand[start] = stack->cand[max_idx];
  stack->cand[max_idx] = tmp;
}

namespace {

inter::mc::Mv picture_clamp(const WarpStackParams& p, inter::mc::Mv mv) {
  const std::int32_t low_y =
      -(p.mi_row + p.bh4) * kMiSize * 8 - mc::kMvBorder;
  const std::int32_t high_y =
      (p.mi_rows - p.mi_row) * kMiSize * 8 + mc::kMvBorder;
  const std::int32_t low_x =
      -(p.mi_col + p.bw4) * kMiSize * 8 - mc::kMvBorder;
  const std::int32_t high_x =
      (p.mi_cols - p.mi_col) * kMiSize * 8 + mc::kMvBorder;
  mv.y = ia::clip3(low_y, high_y, mv.y);
  mv.x = ia::clip3(low_x, high_x, mv.x);
  return mv;
}

}  // namespace

void clamp_stack(const WarpStackParams& params, WarpStack* stack) {
  if (stack == nullptr) return;
  for (int i = 0; i < stack->count; ++i) {
    stack->cand[i].mv[0] = picture_clamp(params, stack->cand[i].mv[0]);
    stack->cand[i].mv[1] = picture_clamp(params, stack->cand[i].mv[1]);
  }
}

// ---------------------------------------------------------------------------
// §7.12.2 builder (single-reference spatial subset + warp half)
// ---------------------------------------------------------------------------

Status build_warp_stack(const WarpStackParams& params, WarpStack* stack) {
  if (stack == nullptr || params.mi == nullptr || params.mi_cols <= 0 ||
      params.mi_rows <= 0 || params.mi_row < 0 || params.mi_col < 0 ||
      params.mi_row >= params.mi_rows || params.mi_col >= params.mi_cols) {
    return Status::kInvalidArgument;
  }
  if (params.bw4 <= 0 || params.bh4 <= 0 || params.bw4 > 16 || params.bh4 > 16) {
    return Status::kInvalidBlockSize;
  }
  if (params.sb_h4 <= 0 || (params.sb_h4 & (params.sb_h4 - 1)) != 0) {
    return Status::kInvalidArgument;
  }
  if (!is_inter_ref_value(params.ref_frame[0])) {
    return Status::kInvalidArgument;
  }
  if (params.is_compound) return Status::kUnsupportedCompound;
  if (params.use_intrabc) return Status::kUnsupportedIntrabc;
  if (params.use_ref_frame_mvs || params.use_temporal_first) {
    return Status::kUnsupportedTemporal;
  }
  if (params.max_drl_bits_minus_1 < 0 || params.max_drl_bits_minus_1 > 4) {
    return Status::kInvalidArgument;
  }

  *stack = WarpStack{};
  for (int i = 0; i < kMaxWarpRefCandidates; ++i) {
    stack->warp_params[i] = ia::WarpParams{};  // Default_Warp_Params
  }

  const int is_sb_border = (params.mi_row & (params.sb_h4 - 1)) == 0 ? 1 : 0;

  // Steps 3-4: generate points from corners.
  if (params.derive_wrl) {
    generate_points_from_corners(params.grid(), params.sb_h4, params.mi_row,
                                 params.mi_col, params.bw4, params.bh4,
                                 params.ref_frame[0], 0, stack);
    if (stack->num_warp_found == 0 && params.bw4 <= 16) {
      generate_points_from_corners(params.grid(), params.sb_h4, params.mi_row,
                                   params.mi_col, params.bw4, params.bh4,
                                   params.ref_frame[0], 1, stack);
    }
  }

  int prune_count = 0;

  // Steps 6-13.
  scan_point(params, params.bh4 - 1, -1, stack, &prune_count);
  scan_point(params, -1,
             std::max(0, params.bw4 - 1 - is_sb_border), stack, &prune_count);
  if (params.bh4 >= 2) {
    scan_point(params, 0, -1, stack, &prune_count);
  }
  if (params.bw4 >= (is_sb_border ? 4 : 2)) {
    scan_point(params, -1, 0, stack, &prune_count);
  }
  if (params.bh4 <= 16) {
    scan_point(params, params.bh4, -1, stack, &prune_count);
  }
  if (params.bw4 <= 16) {
    scan_point(params, -1,
               is_sb_border ? std::max(2, params.bw4) : params.bw4, stack,
               &prune_count);
  }
  scan_point(params, -1, -1 - is_sb_border, stack, &prune_count);

  // Step 14.
  stack->num_nearest = stack->count;

  // Step 15.
  scan_col(params, -3, stack, &prune_count);

  // Steps 16-17.
  const bool use_sort =
      params.reorder == DrlReorder::kAlways ||
      (params.reorder == DrlReorder::kConstraint &&
       !params.use_temporal_first && stack->num_nearest >= 4);
  stack->sorted = use_sort;
  if (use_sort) {
    sort_stack_range(stack, 0, stack->num_nearest);
  }

  // Steps 18-20 are temporal/derived/ref-bank candidates: out of scope.

  // Step 21: extra search.
  clamp_stack(params, stack);
  add_global_candidate(params, stack, &prune_count);
  const bool mvp_gate = params.bw4 * kMiSize > 32 && params.bh4 * kMiSize > 32;
  if (mvp_gate) {
    const int num = stack->count;
    if (num > 1) {
      insert_mvp_candidate(params, 0, 1, stack, &prune_count);
      insert_mvp_candidate(params, 1, 0, stack, &prune_count);
    }
    if (num > 2) {
      insert_mvp_candidate(params, 0, 2, stack, &prune_count);
      insert_mvp_candidate(params, 2, 0, stack, &prune_count);
      insert_mvp_candidate(params, 1, 2, stack, &prune_count);
      insert_mvp_candidate(params, 2, 1, stack, &prune_count);
    }
  }
  if (params.derive_wrl) {
    if (params.warp_bank != nullptr) {
      const int count = params.warp_bank->size();
      for (int i = count - 1; i >= 0; --i) {
        insert_warp_candidate(params.warp_bank->at(i), stack);
      }
    }
    insert_warp_candidate(params.gm_params, stack);
    for (int i = 0; i < 2; ++i) {
      insert_warp_candidate(ia::WarpParams{}, stack);
    }
  }

  // Step 22.
  clamp_stack(params, stack);
  return Status::kOk;
}

// ---------------------------------------------------------------------------
// §7.12.3 find warp samples
// ---------------------------------------------------------------------------

bool above_sample_stored(const WarpSampleParams& params, int delta_col) {
  if (!params.grid.inside(params.mi_row - 1, params.mi_col + delta_col)) {
    return false;
  }
  const int is_sb_border = (params.mi_row & (params.sb_h4 - 1)) == 0 ? 1 : 0;
  if (!is_sb_border) return true;
  if (((params.mi_col + delta_col) % 2) == 0) return true;
  const WarpMi& n = params.grid.at(params.mi_row - 1, params.mi_col + delta_col);
  const int src_w4 = std::max<int>(1, n.w4);
  if (src_w4 == 1) return false;
  return params.mi_col + delta_col + 1 < params.mi_cols;
}

void add_warp_sample(const WarpSampleParams& params, int delta_row, int delta_col,
                     WarpSampleSet* out) {
  if (out == nullptr || out->num_samples >= kLeastSquaresSamplesMax) return;
  const int mv_row = params.mi_row + delta_row;
  const int mv_col = params.mi_col + delta_col;
  if (!params.grid.inside(mv_row, mv_col)) return;
  const WarpMi& n = params.grid.at(mv_row, mv_col);
  if (!n.written) return;
  const std::int8_t target = params.ref_frame[params.ref];
  for (int list = 0; list < 2; ++list) {
    if (n.ref[list] != target) continue;
    const int cand_w4 = std::max<int>(1, n.w4);
    const int cand_h4 = std::max<int>(1, n.h4);
    const int cand_row = n.row_base;
    const int cand_col = n.col_base;
    const int mid_y = cand_row * kMiSize + cand_h4 * 2 - 1;
    const int mid_x = cand_col * kMiSize + cand_w4 * 2 - 1;
    int* cand = out->cand[out->num_samples];
    cand[0] = mid_y * 8;
    cand[1] = mid_x * 8;
    cand[2] = mid_y * 8 + n.mv[list][0];
    cand[3] = mid_x * 8 + n.mv[list][1];
    ++out->num_samples;
    if (out->num_samples >= kLeastSquaresSamplesMax) return;
  }
}

bool find_warp_samples(const WarpSampleParams& params, WarpSampleSet* out) {
  if (out == nullptr || params.grid.mi == nullptr || params.w4 <= 0 ||
      params.h4 <= 0 || params.ref < 0 || params.ref > 1 ||
      params.mi_row < 0 || params.mi_col < 0 ||
      params.mi_row >= params.mi_rows || params.mi_col >= params.mi_cols) {
    return false;
  }
  *out = WarpSampleSet{};
  int do_top_left = 1;
  int do_top_right = 1;
  const bool avail_u = params.avail_u && params.mi_row > 0;
  const bool avail_l = params.avail_l && params.mi_col > 0;

  if (avail_u) {
    const int col_offset = params.grid.at(params.mi_row - 1, params.mi_col).col_base -
                           params.mi_col;
    if (col_offset < 0) do_top_left = 0;
    int i = col_offset;
    const int limit = std::min(params.w4, params.mi_cols - params.mi_col);
    while (i < limit) {
      const WarpMi& n = params.grid.at(params.mi_row - 1, params.mi_col + i);
      const int src_w = std::max<int>(1, n.w4);
      if (above_sample_stored(params, i)) {
        add_warp_sample(params, -1, i, out);
      }
      i += src_w;
    }
    do_top_right = (i == params.w4) && (i < params.mi_cols - params.mi_col);
  }
  if (avail_l) {
    const int row_offset =
        params.grid.at(params.mi_row, params.mi_col - 1).row_base - params.mi_row;
    if (row_offset < 0) do_top_left = 0;
    int i = row_offset;
    const int limit = std::min(params.h4, params.mi_rows - params.mi_row);
    while (i < limit) {
      const WarpMi& n = params.grid.at(params.mi_row + i, params.mi_col - 1);
      const int src_h = std::max<int>(1, n.h4);
      add_warp_sample(params, i, -1, out);
      i += src_h;
    }
  }
  if (do_top_left && above_sample_stored(params, -1)) {
    add_warp_sample(params, -1, -1, out);
  }
  if (do_top_right && params.w4 <= 16 && above_sample_stored(params, params.w4)) {
    add_warp_sample(params, -1, params.w4, out);
  }
  return out->num_samples > 0;
}

// ---------------------------------------------------------------------------
// §7.13.3.23 local warp estimation
// ---------------------------------------------------------------------------

namespace {

int floor_log2_64(std::int64_t v) {
  int n = 0;
  while (v > 1) {
    v >>= 1;
    ++n;
  }
  return n;
}

// §7.13.3.22 with a 64-bit divisor.  The spec's integers are unbounded; the
// determinant of the normal equations for a 64x64 block exceeds 2^31, so the
// divisor must not be narrowed (adv::resolve_divisor() takes int and is only
// used by the component's own fitter).
struct Div64 {
  int shift = 0;
  std::int64_t factor = 0;
};

Div64 resolve_divisor_64(std::int64_t d) {
  const std::int64_t magnitude = d < 0 ? -d : d;
  const int n = floor_log2_64(magnitude);
  const std::int64_t e = magnitude - (std::int64_t{1} << n);
  const int f = (n > ia::kDivLutBits)
                    ? static_cast<int>(
                          ia::round2_signed64(e, n - ia::kDivLutBits))
                    : static_cast<int>(e << (ia::kDivLutBits - n));
  Div64 out;
  out.shift = n + ia::kDivLutPrecBits;
  out.factor = d < 0 ? -ia::div_lut_entry(f) : ia::div_lut_entry(f);
  return out;
}

}  // namespace

// §7.13.3.23 warp estimation.  This is a direct transcription of the spec's
// least-squares normal equations (not adv::warp_estimate, which narrows the
// determinant to int).  The samples and block geometry come from §7.12.3.
ia::WarpParams estimate_warp_from_samples(const WarpSampleSet& samples, int mid_x,
                                          int mid_y, const int block_mv[2]) {
  const std::int64_t suy = std::int64_t{mid_y} * 8;
  const std::int64_t sux = std::int64_t{mid_x} * 8;
  const std::int64_t duy = suy + block_mv[0];
  const std::int64_t dux = sux + block_mv[1];

  std::int64_t a00 = 0;
  std::int64_t a01 = 0;
  std::int64_t a11 = 0;
  std::int64_t bx0 = 0;
  std::int64_t bx1 = 0;
  std::int64_t by0 = 0;
  std::int64_t by1 = 0;
  for (int i = 0; i < samples.num_samples; ++i) {
    const int sy = samples.cand[i][0] - static_cast<int>(suy);
    const int sx = samples.cand[i][1] - static_cast<int>(sux);
    const int dy = samples.cand[i][2] - static_cast<int>(duy);
    const int dx = samples.cand[i][3] - static_cast<int>(dux);
    if (std::abs(sx - dx) < kLsMvMax && std::abs(sy - dy) < kLsMvMax) {
      a00 += ls_product(sx, sx) + 8;
      a01 += ls_product(sx, sy) + 4;
      a11 += ls_product(sy, sy) + 8;
      bx0 += ls_product(sx, dx) + 8;
      bx1 += ls_product(sy, dx) + 4;
      by0 += ls_product(sx, dy) + 4;
      by1 += ls_product(sy, dy) + 8;
    }
  }

  ia::WarpParams params{};
  const std::int64_t det = a00 * a11 - a01 * a01;
  if (det == 0) {
    params.p[2] = 1 << ia::kWarpedModelPrecBits;
    params.p[3] = 0;
    params.p[4] = 0;
    params.p[5] = 1 << ia::kWarpedModelPrecBits;
    ia::get_warp_translation(params, mid_x, mid_y, block_mv);
    return params;
  }

  Div64 ds = resolve_divisor_64(det);
  int div_shift = ds.shift - ia::kWarpedModelPrecBits;
  std::int64_t div_factor = ds.factor;
  if (div_shift < 0) {
    div_factor <<= -div_shift;
    div_shift = 0;
  }

  auto diag = [&](std::int64_t v) -> int {
    const std::int64_t rounded = ia::round2_signed64(v * div_factor, div_shift);
    return static_cast<int>(
        rounded > 2147483647LL ? 2147483647LL
                               : (rounded < -2147483648LL ? -2147483648LL : rounded));
  };
  params.p[2] = diag(a11 * bx0 - a01 * bx1);
  params.p[3] = diag(-a01 * bx0 + a00 * bx1);
  params.p[4] = diag(a11 * by0 - a01 * by1);
  params.p[5] = diag(-a01 * by0 + a00 * by1);
  params = ia::reduce_warp_model(params);
  ia::get_warp_translation(params, mid_x, mid_y, block_mv);
  return params;
}

ia::WarpParams warp_model_with_translation(ia::WarpParams params, int mid_x,
                                           int mid_y, const int block_mv[2]) {
  ia::get_warp_translation(params, mid_x, mid_y, block_mv);
  return params;
}

std::optional<LocalWarpResult> derive_local_warp(const LocalWarpParams& params,
                                                 int ref) {
  if (params.grid.mi == nullptr || params.w4 <= 0 || params.h4 <= 0 || ref < 0 ||
      ref > 1) {
    return std::nullopt;
  }
  WarpSampleParams sp;
  sp.mi_row = params.mi_row;
  sp.mi_col = params.mi_col;
  sp.w4 = params.w4;
  sp.h4 = params.h4;
  sp.mi_cols = params.mi_cols;
  sp.mi_rows = params.mi_rows;
  sp.sb_h4 = params.sb_h4;
  sp.grid = params.grid;
  sp.ref_frame[0] = params.ref_frame[0];
  sp.ref_frame[1] = params.ref_frame[1];
  sp.ref = ref;
  sp.avail_u = params.avail_u;
  sp.avail_l = params.avail_l;

  LocalWarpResult out;
  find_warp_samples(sp, &out.samples);

  const int mid_y = params.mi_row * kMiSize + params.h4 * 2 - 1;
  const int mid_x = params.mi_col * kMiSize + params.w4 * 2 - 1;

  // Reproduce the determinant to report the degenerate branch.
  const int suy = mid_y * 8;
  const int sux = mid_x * 8;
  const int duy = suy + params.block_mv.y;
  const int dux = sux + params.block_mv.x;
  std::int64_t a00 = 0;
  std::int64_t a01 = 0;
  std::int64_t a11 = 0;
  for (int i = 0; i < out.samples.num_samples; ++i) {
    const int sy = out.samples.cand[i][0] - suy;
    const int sx = out.samples.cand[i][1] - sux;
    const int dy = out.samples.cand[i][2] - duy;
    const int dx = out.samples.cand[i][3] - dux;
    if (std::abs(sx - dx) < kLsMvMax && std::abs(sy - dy) < kLsMvMax) {
      a00 += ls_product(sx, sx) + 8;
      a01 += ls_product(sx, sy) + 4;
      a11 += ls_product(sy, sy) + 8;
    }
  }
  out.det_zero = (a00 * a11 - a01 * a01) == 0;

  const int block_mv[2] = {params.block_mv.y, params.block_mv.x};
  out.params = estimate_warp_from_samples(out.samples, mid_x, mid_y, block_mv);
  return out;
}

}  // namespace av2enc::inter::warp

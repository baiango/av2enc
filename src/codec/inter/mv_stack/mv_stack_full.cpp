// AV2 v1.0.0 §7.12.2 complete motion-vector prediction stack — implementation.
//
// Spec-only: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7).  No encoder code from AVM, SVT-AV1/AV2, clave
// or av2-turbo was consulted.
//
// Section references in the comments are to that document.  Where the spec
// leaves a value uninitialised but a deterministic byte-for-byte mirror is
// required (e.g. WeightStack for the extra-search global candidate, which is
// only read after the sort), the chosen value is marked "mirror only".

#include "codec/inter/mv_stack/mv_stack_full.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <limits>

namespace av2enc::inter::mv_stack_full {
namespace {

// §7.9.4 Div_Mult[32].
constexpr std::int32_t kDivMult[32] = {
    0,     16384, 8192, 5461, 4096, 3276, 2730, 2340, 2048, 1820, 1638,
    1489,  1365,  1260, 1170, 1092, 1024, 963,  910,  862,  819,  780,
    744,   712,   682,  655,  630,  606,  585,  564,  546,  528};

// §5.18.9.1 Default_Warp_Params.
constexpr adv::WarpParams kDefaultWarpParams{};

constexpr int kMaxFrameDistance = 31;  // MAX_FRAME_DISTANCE

std::int32_t clip3(std::int32_t low, std::int32_t high, std::int32_t value) {
  return value < low ? low : (value > high ? high : value);
}

std::int64_t clip64(std::int64_t low, std::int64_t high, std::int64_t value) {
  return value < low ? low : (value > high ? high : value);
}

// §4.8 Round2Signed.
std::int64_t round2_signed64(std::int64_t x, int n) {
  if (n == 0) return x;
  return x >= 0 ? ((x + (static_cast<std::int64_t>(1) << (n - 1))) >> n)
                : -(((-x) + (static_cast<std::int64_t>(1) << (n - 1))) >> n);
}

std::int32_t round2_signed32(std::int64_t x, int n) {
  return static_cast<std::int32_t>(round2_signed64(x, n));
}

int floor_log2(int value) {
  int bits = 0;
  while (value > 1) {
    value >>= 1;
    ++bits;
  }
  return bits;
}

int relative_dist(const StackParams& p, int a_ref, int b_ref) {
  const bool a_restricted =
      a_ref == p.closest_past ? p.closest_past_restricted
                              : (a_ref == p.closest_future
                                     ? p.closest_future_restricted
                                     : false);
  const bool b_restricted =
      b_ref == p.closest_past ? p.closest_past_restricted
                              : (b_ref == p.closest_future
                                     ? p.closest_future_restricted
                                     : false);
  const int a = p.order_hints == nullptr ? 0 : p.order_hints[a_ref];
  const int b = p.order_hints == nullptr ? 0 : p.order_hints[b_ref];
  if (a_restricted && b_restricted) return 0;
  if (a_restricted) return 127;
  if (b_restricted) return -127;
  return clip3(-127, 127, a - b);
}

int relative_dist_hint(const StackParams& p, int order_hint, int ref) {
  const bool restricted =
      ref == p.closest_past ? p.closest_past_restricted
                            : (ref == p.closest_future
                                   ? p.closest_future_restricted
                                   : false);
  const int target = p.order_hints == nullptr ? 0 : p.order_hints[ref];
  if (restricted) return order_hint > target ? 127 : -127;
  return clip3(-127, 127, order_hint - target);
}

int frame_distance(const StackParams& p, int ref) {
  if (p.frame_distance == nullptr) return 0;
  if (ref < 0 || ref >= p.num_refs) return 0;
  return p.frame_distance[ref];
}

const MiCell& mi_at(const StackParams& p, int row, int col) {
  return p.grid.at(row, col);
}

Mv get_mv(const StackParams& p, int mv_row, int mv_col, int ref_list,
          int cand_list) {
  const MiCell& m = mi_at(p, mv_row, mv_col);
  const std::int8_t cand_ref = m.ref[cand_list];
  const int cand_size_w = m.w4 > 0 ? m.w4 * 4 : 4;
  const int cand_size_h = m.h4 > 0 ? m.h4 * 4 : 4;
  const bool large = std::min(cand_size_w, cand_size_h) >= 8;
  if (ref_list >= 0 && ref_list < 2 &&
      (m.y_mode == kYModeGlobalMv || m.y_mode == kYModeGlobalGlobalMv) &&
      cand_ref != kTipFrame && cand_ref >= 0 &&
      cand_ref < kRefsPerFrame && p.gm_type_by_ref[cand_ref] > kGmIdentity &&
      large) {
    return p.global_mv[ref_list];
  }
  return Mv{m.sub_mv[cand_list][0], m.sub_mv[cand_list][1]};
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
    case Status::kInvalidReference:
      return "invalid reference";
    case Status::kInvalidStackParams:
      return "invalid stack parameters";
    case Status::kUnsupportedFeature:
      return "unsupported feature";
  }
  return "unknown";
}

// ---------------------------------------------------------------------------
// §5.20.9.4/.5 clamping
// ---------------------------------------------------------------------------

std::int32_t clamp_mv_row(const StackParams& p, std::int32_t value) {
  const std::int32_t low = -(p.mi_row + p.bh4) * kMiSize * 8 - kMvBorder;
  const std::int32_t high =
      (p.grid.rows - p.mi_row) * kMiSize * 8 + kMvBorder;
  return clip3(low, high, value);
}

std::int32_t clamp_mv_col(const StackParams& p, std::int32_t value) {
  const std::int32_t low = -(p.mi_col + p.bw4) * kMiSize * 8 - kMvBorder;
  const std::int32_t high =
      (p.grid.cols - p.mi_col) * kMiSize * 8 + kMvBorder;
  return clip3(low, high, value);
}

// ---------------------------------------------------------------------------
// Predicates
// ---------------------------------------------------------------------------

bool is_thin_4xn_nx4_block(partition::BlockSize size) {
  if (size == partition::BlockSize::kInvalid) return false;
  const int w = partition::block_width(size);
  const int h = partition::block_height(size);
  return (w == 4 && h >= 16) || (h == 4 && w >= 16);
}

bool is_comp_ref_allowed(partition::BlockSize size) {
  if (size == partition::BlockSize::kInvalid) return false;
  const int w = partition::block_width(size);
  const int h = partition::block_height(size);
  return std::min(w, h) >= 8 || is_thin_4xn_nx4_block(size);
}

bool mvp_combinations_allowed(int width, int height) {
  return width > 32 && height > 32;
}

bool has_newmv_mode(int y_mode) {
  return y_mode == kYModeNewMv || y_mode == kYModeNewNewMv ||
         y_mode == kYModeNearNewMv || y_mode == kYModeNewNearMv ||
         y_mode == kYModeWarpNewMv || y_mode == kYModeJointNewMv;
}

bool has_newmv_for_list(int y_mode, int ref_list) {
  if (y_mode == kYModeNewNewMv || y_mode == kYModeNewMv) return true;
  if (ref_list == 0) {
    return y_mode == kYModeNewNearMv || y_mode == kYModeJointNewMv;
  }
  return y_mode == kYModeNearNewMv;
}

bool has_nearmv_mode(int y_mode) {
  return y_mode == kYModeNearMv || y_mode == kYModeNearNearMv ||
         y_mode == kYModeNearNewMv || y_mode == kYModeNewNearMv;
}

void lower_mv_precision(int precision, Mv* mv) {
  if (mv == nullptr) return;
  const int bits = kMvPrecisionEighthPel - precision;
  if (bits <= 0) return;
  const std::int32_t radix = static_cast<std::int32_t>(1) << bits;
  for (int i = 0; i < 2; ++i) {
    std::int32_t& v = i == 0 ? mv->y : mv->x;
    const std::int32_t a = std::abs(v);
    // Round2(a - 1, bits) is the floor form for a non-negative value.
    const std::int32_t a_int =
        static_cast<std::int32_t>((a - 1 + (radix >> 1)) >> bits);
    v = v >= 0 ? (a_int << bits) : (-a_int << bits);
    if ((a_int << bits) != a) {
      v = clip3(kMvLow + radix, kMvUpp - radix, v);
    }
  }
}

std::int32_t clamp_mv_to_integer(std::int32_t v) {
  if (v < kMvLow + 1) return static_cast<std::int32_t>(kMvLow + 8);
  if (v > kMvUpp - 1) return static_cast<std::int32_t>(kMvUpp - 8);
  return v;
}

Mv clamp_mv_to_integer(Mv mv) {
  mv.y = clamp_mv_to_integer(mv.y);
  mv.x = clamp_mv_to_integer(mv.x);
  return mv;
}

Mv project_mv(Mv mv, int numerator, int denominator) {
  // §7.9.4 get_mv_projection.
  const int clipped_denominator = std::min(31, denominator);
  const int clipped_numerator = clip3(-31, 31, numerator);
  Mv out;
  out.y = clip3(
      kMvLow + 1, kMvUpp - 1,
      round2_signed32(static_cast<std::int64_t>(mv.y) * clipped_numerator *
                          kDivMult[clipped_denominator],
                      14));
  out.x = clip3(
      kMvLow + 1, kMvUpp - 1,
      round2_signed32(static_cast<std::int64_t>(mv.x) * clipped_numerator *
                          kDivMult[clipped_denominator],
                      14));
  return out;
}

// ---------------------------------------------------------------------------
// §7.12.2.2 / §7.12.2.4 warp MV projection
// ---------------------------------------------------------------------------

Mv get_warp_motion_vector(const adv::WarpParams& warp, int mi_col, int mi_row,
                          int bw4, int bh4, int mi_cols, int mi_rows,
                          int mv_precision, bool force_integer_mv) {
  const int bw = bw4 * kMiSize;
  const int bh = bh4 * kMiSize;
  const std::int32_t x = mi_col * kMiSize + bw / 2 - 1;
  const std::int32_t y = mi_row * kMiSize + bh / 2 - 1;
  const std::int64_t one = static_cast<std::int64_t>(1) << 16;
  const std::int64_t xc = (static_cast<std::int64_t>(warp.p[2]) - one) * x +
                          static_cast<std::int64_t>(warp.p[3]) * y + warp.p[0];
  const std::int64_t yc = static_cast<std::int64_t>(warp.p[4]) * x +
                          (static_cast<std::int64_t>(warp.p[5]) - one) * y +
                          warp.p[1];
  Mv mv;
  if (mv_precision == kMvPrecisionEighthPel) {
    mv.y = round2_signed32(yc, 13);
    mv.x = round2_signed32(xc, 13);
  } else {
    mv.y = round2_signed32(yc, 14) * 2;
    mv.x = round2_signed32(xc, 14) * 2;
  }
  mv.y = clip3(kMvLow + 1, kMvUpp - 1, mv.y);
  mv.x = clip3(kMvLow + 1, kMvUpp - 1, mv.x);
  // These two use the *current* block's geometry; the caller passes it through
  // a StackParams-shaped wrapper, so rebuild the clamp bounds here.
  (void)mi_cols;
  (void)mi_rows;
  (void)force_integer_mv;
  return mv;
}

Mv get_warp_motion_vector_xy_pos(const adv::WarpParams& warp, int pos_row,
                                 int pos_col, int mi_rows, int mi_cols) {
  const std::int32_t y = pos_row * kMiSize;
  const std::int32_t x = pos_col * kMiSize;
  const std::int64_t one = static_cast<std::int64_t>(1) << 16;
  const std::int64_t xc = static_cast<std::int64_t>(warp.p[2]) * x +
                          static_cast<std::int64_t>(warp.p[3]) * y + warp.p[0] -
                          (static_cast<std::int64_t>(x) << 16);
  const std::int64_t yc = static_cast<std::int64_t>(warp.p[4]) * x +
                          static_cast<std::int64_t>(warp.p[5]) * y + warp.p[1] -
                          (static_cast<std::int64_t>(y) << 16);
  Mv mv;
  mv.y = clip3(kMvLow + 1, kMvUpp - 1, round2_signed32(yc, 13));
  mv.x = clip3(kMvLow + 1, kMvUpp - 1, round2_signed32(xc, 13));
  (void)one;
  (void)mi_rows;
  (void)mi_cols;
  return mv;
}

Status setup_global_mv(const StackParams& p, int ref_list, Mv* out) {
  if (out == nullptr || ref_list < 0 || ref_list > 1) {
    return Status::kInvalidArgument;
  }
  Mv mv{};
  const std::int8_t ref = p.ref_frame[ref_list];
  if (ref != kIntraFrame && ref != kTipFrame && p.gm_params_present) {
    mv = get_warp_motion_vector(p.gm_params[ref_list], p.mi_col, p.mi_row,
                                p.bw4, p.bh4, p.grid.cols, p.grid.rows,
                                p.frame_mv_precision, p.force_integer_mv);
    // §7.12.2.2 tail: picture clamp, then lower_mv_precision for precisions
    // below half-pel.
    mv.y = clamp_mv_row(p, mv.y);
    mv.x = clamp_mv_col(p, mv.x);
    if (p.frame_mv_precision < kMvPrecisionEighthPel - 2) {
      lower_mv_precision(p.frame_mv_precision, &mv);
    }
  }
  *out = mv;
  return Status::kOk;
}

// ---------------------------------------------------------------------------
// §7.12.2.11 insert warp candidate
// ---------------------------------------------------------------------------

void WarpParamBank::reset() {
  params_[0] = kDefaultWarpParams;
  params_[1] = kDefaultWarpParams;
  params_[2] = kDefaultWarpParams;
  params_[3] = kDefaultWarpParams;
  start_ = 0;
  size_ = 0;
}

void WarpParamBank::record(const adv::WarpParams& params) {
  if (size_ < kWarpParamBankSize) {
    params_[(start_ + size_) % kWarpParamBankSize] = params;
    ++size_;
    return;
  }
  params_[start_] = params;
  start_ = (start_ + 1) % kWarpParamBankSize;
}

const adv::WarpParams& WarpParamBank::slot(int physical) const {
  return params_[physical % kWarpParamBankSize];
}

Status insert_warp_candidate(const adv::WarpParams& params, Stack* stack) {
  if (stack == nullptr) return Status::kInvalidArgument;
  // §7.12.2.11 has no deduplication: it appends until the array is full.
  if (stack->num_warp_found >= kMaxWarpRefCandidates) return Status::kOk;
  stack->warp_params[stack->num_warp_found] = params;
  ++stack->num_warp_found;
  return Status::kOk;
}

// ---------------------------------------------------------------------------
// §7.12.2.9 add warp motion vector
// ---------------------------------------------------------------------------

Status add_warp_motion_vector(const StackParams& p, BuildState* state,
                              int mv_row, int mv_col) {
  if (state == nullptr) return Status::kInvalidArgument;
  if (!p.grid.inside(mv_row, mv_col)) return Status::kOk;
  const MiCell& m = mi_at(p, mv_row, mv_col);
  if (m.is_inter == 0) return Status::kOk;
  if (!p.derive_wrl) return Status::kOk;
  if (m.motion_mode < kMotionLocalWarp) return Status::kOk;
  if (m.ref[0] != p.ref_frame[0]) return Status::kOk;
  if (state->stack.num_warp_found >= kMaxWarpRefCandidates) {
    return Status::kOk;
  }
  const Status status = insert_warp_candidate(m.warp[0], &state->stack);
  if (status == Status::kOk) ++state->trace.warp_inserted;
  return status;
}

// ---------------------------------------------------------------------------
// §7.12.2.12 search stack
// ---------------------------------------------------------------------------

Status search_stack(const StackParams& p, BuildState* state, int mv_row,
                    int mv_col, int weight, int cand_list) {
  if (state == nullptr) return Status::kInvalidArgument;
  Stack& s = state->stack;
  const Mv cand_mv = get_mv(p, mv_row, mv_col, 0, cand_list);
  bool found = false;
  if (state->prune_count < kMaxPrNum) {
    for (int i = 0; i < s.count; ++i) {
      ++state->prune_count;
      if (cand_mv == s.cand[i].mv[0]) {
        s.cand[i].weight += weight;
        found = true;
        break;
      }
    }
  }
  if (!found && s.count < kMaxRefMvStackSize) {
    Candidate& c = s.cand[s.count];
    c = Candidate{};
    c.mv[0] = cand_mv;
    c.mv[1] = p.is_compound ? Mv{} : cand_mv;
    c.weight = weight;
    c.row_offset = mv_row - p.mi_row;
    c.col_offset = mv_col - p.mi_col;
    c.cwp = kCwpEqual;
    ++s.count;
  }
  return Status::kOk;
}

// §7.12.2.13 compound search stack.
Status compound_search_stack(const StackParams& p, BuildState* state,
                             int mv_row, int mv_col, int weight) {
  if (state == nullptr) return Status::kInvalidArgument;
  Stack& s = state->stack;
  const MiCell& m = mi_at(p, mv_row, mv_col);
  Mv cand[2];
  for (int list = 0; list < 2; ++list) {
    cand[list] = Mv{m.sub_mv[list][0], m.sub_mv[list][1]};
    const int cand_size_w = m.w4 > 0 ? m.w4 * 4 : 4;
    const int cand_size_h = m.h4 > 0 ? m.h4 * 4 : 4;
    const bool large = std::min(cand_size_w, cand_size_h) >= 8;
    if (large && m.y_mode == kYModeGlobalGlobalMv &&
        m.ref[list] != kTipFrame && m.ref[list] >= 0 &&
        m.ref[list] < kRefsPerFrame &&
        p.gm_type_by_ref[m.ref[list]] > kGmIdentity) {
      cand[list] = p.global_mv[list];
    }
  }
  bool found = false;
  if (state->prune_count < kMaxPrNum) {
    for (int i = 0; i < s.count; ++i) {
      ++state->prune_count;
      if (cand[0] == s.cand[i].mv[0] && cand[1] == s.cand[i].mv[1]) {
        s.cand[i].weight += weight;
        found = true;
        break;
      }
    }
  }
  if (!found && s.count < kMaxRefMvStackSize) {
    Candidate& c = s.cand[s.count];
    c = Candidate{};
    c.mv[0] = cand[0];
    c.mv[1] = cand[1];
    c.weight = weight;
    c.row_offset = mv_row - p.mi_row;
    c.col_offset = mv_col - p.mi_col;
    c.cwp = static_cast<std::int8_t>(m.cwp);
    ++s.count;
  }
  return Status::kOk;
}

// ---------------------------------------------------------------------------
// Derived stacks (§7.12.2.14/.15/.16/.17/.18)
// ---------------------------------------------------------------------------

namespace {

Status add_derived(BuildState* state, Mv mv0, Mv mv1) {
  if (state->derived_count >= kMaxDrStackSize) return Status::kOk;
  BuildState::Derived& d = state->derived[state->derived_count];
  d.mv[0] = mv0;
  d.mv[1] = mv1;
  ++state->derived_count;
  return Status::kOk;
}

// comp_mv_in_stack(): the prune budget is shared by every caller.
bool comp_mv_in_stack(BuildState* state, Mv mv0, Mv mv1) {
  if (state->derived_prune_count < kMaxDrPrNum) {
    for (int i = 0; i < state->derived_count; ++i) {
      ++state->derived_prune_count;
      if (state->derived[i].mv[0] == mv0 && state->derived[i].mv[1] == mv1) {
        return true;
      }
    }
  }
  return false;
}

bool comp_mv_in_derived(BuildState* state, Mv mv) {
  if (state->derived_prune_count < kMaxDrPrNum) {
    for (int i = 0; i < state->derived_count; ++i) {
      ++state->derived_prune_count;
      if (state->derived[i].mv[0] == mv) return true;
    }
  }
  return false;
}

bool traj_available(const StackParams& p, int ref, int y8, int x8,
                    Mv* traj_mv) {
  if (!p.use_ref_frame_mvs || !p.enable_mv_traj || p.tip == nullptr) {
    return false;
  }
  if (p.tip->traj_valid == nullptr || p.tip->traj_mv == nullptr) return false;
  if (ref < 0 || ref >= p.tip->traj_refs) return false;
  const std::size_t n8 = static_cast<std::size_t>(p.tip->cols8) *
                         static_cast<std::size_t>(p.tip->rows8);
  const std::size_t idx =
      static_cast<std::size_t>(ref) * n8 +
      static_cast<std::size_t>(y8) * static_cast<std::size_t>(p.tip->cols8) +
      static_cast<std::size_t>(x8);
  if (p.tip->traj_valid[idx] == 0) return false;
  traj_mv->y = p.tip->traj_mv[idx * 2 + 0];
  traj_mv->x = p.tip->traj_mv[idx * 2 + 1];
  return true;
}

}  // namespace

Status single_add_derived(const StackParams& p, BuildState* state, int mv_row,
                          int mv_col, int cand_list) {
  if (state == nullptr || cand_list < 0 || cand_list > 1) {
    return Status::kInvalidArgument;
  }
  const MiCell& m = mi_at(p, mv_row, mv_col);
  Mv cand_mv{};
  int cand_ref = kRefNone;
  if (m.ref[0] == kTipFrame) {
    Mv mvs[2];
    const Status st = get_tip_cand(p, mv_row, mv_col, mvs);
    if (st == Status::kOk) cand_mv = mvs[cand_list];
    cand_ref = cand_list ? p.closest_future : p.closest_past;
  } else {
    cand_mv = get_mv(p, mv_row, mv_col, -1, cand_list);
    cand_ref = m.ref[cand_list];
  }
  if (p.ref_frame[0] == kTipFrame || cand_ref == kRefNone ||
      cand_ref == kIntraFrame) {
    return Status::kOk;
  }
  const int cur_dist = frame_distance(p, p.ref_frame[0]);
  const int cand_dist = frame_distance(p, cand_ref);
  bool have_proj = false;
  Mv proj{};
  if (p.use_ref_frame_mvs && p.enable_mv_traj) {
    const int y8 = p.mi_row >> 1;
    const int x8 = p.mi_col >> 1;
    Mv traj_cur{}, traj_cand{};
    const bool cur_valid =
        traj_available(p, p.ref_frame[0], y8, x8, &traj_cur);
    const bool cand_valid = traj_available(p, cand_ref, y8, x8, &traj_cand);
    if (cur_valid && cand_valid) {
      have_proj = true;
      proj.y = clip3(kMvLow + 1, kMvUpp - 1,
                     cand_mv.y + traj_cur.y - traj_cand.y);
      proj.x = clip3(kMvLow + 1, kMvUpp - 1,
                     cand_mv.x + traj_cur.x - traj_cand.x);
    }
  }
  if (!have_proj &&
      ((cur_dist > 0 && cand_dist > 0) || (cur_dist < 0 && cand_dist < 0))) {
    proj = project_mv(cand_mv, std::abs(cur_dist), std::abs(cand_dist));
    have_proj = true;
  }
  if (have_proj && !comp_mv_in_derived(state, proj)) {
    return add_derived(state, proj, proj);
  }
  return Status::kOk;
}

Status compound_add_derived(const StackParams& p, BuildState* state,
                            int mv_row, int mv_col) {
  if (state == nullptr) return Status::kInvalidArgument;
  const MiCell& m = mi_at(p, mv_row, mv_col);
  if (p.enable_mv_traj && p.use_ref_frame_mvs &&
      p.ref_frame[0] != p.ref_frame[1]) {
    for (int list = 0; list < 2; ++list) {
      const std::int8_t cand_ref = m.ref[list];
      if (!is_inter_ref_frame(cand_ref) || cand_ref == kTipFrame) continue;
      const Mv cand_mv = get_mv(p, mv_row, mv_col, -1, list);
      const int y8 = p.mi_row >> 1;
      const int x8 = p.mi_col >> 1;
      Mv traj_cand{}, traj_ref0{}, traj_ref1{};
      const bool cand_valid = traj_available(p, cand_ref, y8, x8, &traj_cand);
      const bool ref0_valid =
          traj_available(p, p.ref_frame[0], y8, x8, &traj_ref0);
      const bool ref1_valid =
          traj_available(p, p.ref_frame[1], y8, x8, &traj_ref1);
      if (!(cand_valid && ref0_valid && ref1_valid)) continue;
      Mv cand[2];
      cand[0].y = clip3(kMvLow + 1, kMvUpp - 1,
                        cand_mv.y + traj_ref0.y - traj_cand.y);
      cand[0].x = clip3(kMvLow + 1, kMvUpp - 1,
                        cand_mv.x + traj_ref0.x - traj_cand.x);
      cand[1].y = clip3(kMvLow + 1, kMvUpp - 1,
                        cand_mv.y + traj_ref1.y - traj_cand.y);
      cand[1].x = clip3(kMvLow + 1, kMvUpp - 1,
                        cand_mv.x + traj_ref1.x - traj_cand.x);
      if (!comp_mv_in_stack(state, cand[0], cand[1])) {
        add_derived(state, cand[0], cand[1]);
      }
    }
  }

  int cand_ref_idx0 = 0;
  int cand_ref_idx1 = 1;
  if (m.ref[0] == p.ref_frame[0] || m.ref[1] == p.ref_frame[0]) {
    cand_ref_idx0 = 0;
    cand_ref_idx1 = 1;
  } else if (m.ref[0] == p.ref_frame[1] || m.ref[1] == p.ref_frame[1]) {
    cand_ref_idx0 = 1;
    cand_ref_idx1 = 0;
  } else {
    return Status::kOk;
  }
  const int cand_list = (m.ref[0] == p.ref_frame[cand_ref_idx0]) ? 0 : 1;
  const Mv cand_mv = get_mv(p, mv_row, mv_col, cand_ref_idx0, cand_list);

  for (int i = 0; i < state->single_count && state->derived_count < kMaxDrStackSize;
       ++i) {
    if (state->single[i].ref != p.ref_frame[cand_ref_idx1]) continue;
    const Mv l0 = cand_ref_idx0 == 0 ? cand_mv : state->single[i].mv;
    const Mv l1 = cand_ref_idx0 == 1 ? cand_mv : state->single[i].mv;
    if (!comp_mv_in_stack(state, l0, l1)) add_derived(state, l0, l1);
    break;
  }

  if (state->single_prune_count < kMaxDrPrNum) {
    for (int i = 0; i < state->single_count; ++i) {
      ++state->single_prune_count;
      if (state->single[i].ref == p.ref_frame[cand_ref_idx0] &&
          state->single[i].mv == cand_mv) {
        return Status::kOk;
      }
    }
  }
  if (state->single_count < kMaxDrStackSize) {
    state->single[state->single_count].ref = p.ref_frame[cand_ref_idx0];
    state->single[state->single_count].mv = cand_mv;
    ++state->single_count;
  }
  return Status::kOk;
}

// ---------------------------------------------------------------------------
// TIP helpers
// ---------------------------------------------------------------------------

Status get_tip_offsets(const StackParams& p, int* ref_offset, int* past_offset,
                       int* future_offset) {
  if (ref_offset == nullptr || past_offset == nullptr ||
      future_offset == nullptr) {
    return Status::kInvalidArgument;
  }
  if (p.closest_past == kRefNone || p.closest_future == kRefNone ||
      p.order_hints == nullptr) {
    return Status::kInvalidReference;
  }
  int ref_off;
  if (p.num_future_refs > 0 && p.num_past_refs > 0) {
    ref_off = relative_dist(p, p.closest_future, p.closest_past);
  } else {
    ref_off = relative_dist(p, p.closest_past, p.closest_future);
  }
  *past_offset = relative_dist_hint(p, p.order_hint, p.closest_past);
  *future_offset = relative_dist_hint(p, p.order_hint, p.closest_future);
  *ref_offset = std::min(ref_off, kMaxFrameDistance);
  return Status::kOk;
}

Status get_tip_cand(const StackParams& p, int cand_row, int cand_col,
                    Mv out[2]) {
  if (out == nullptr) return Status::kInvalidArgument;
  if (!p.grid.inside(cand_row, cand_col)) return Status::kInvalidArgument;
  if (p.tip == nullptr || p.tip->cols8 <= 0 || p.tip->rows8 <= 0) {
    return Status::kUnsupportedFeature;
  }
  const MiCell& cell = mi_at(p, cand_row, cand_col);
  const int base_row = cell.row_base;
  const int base_col = cell.col_base;
  const int shift = 1 + static_cast<int>(cell.tip_16x16);
  cand_row = base_row + (((cand_row - base_row) >> shift) << shift);
  cand_col = base_col + (((cand_col - base_col) >> shift) << shift);
  if (!p.grid.inside(cand_row, cand_col)) return Status::kInvalidArgument;
  const int x8 = cand_col >> 1;
  const int y8 = cand_row >> 1;
  const int ref_x8 = clip3(0, (p.grid.cols >> 1) - 1, x8);
  const int ref_y8 = clip3(0, (p.grid.rows >> 1) - 1, y8);
  Mv cand[2];
  const std::size_t idx =
      static_cast<std::size_t>(ref_y8) *
          static_cast<std::size_t>(p.tip->cols8) +
      static_cast<std::size_t>(ref_x8);
  const bool valid = p.tip->valid != nullptr && p.tip->mv != nullptr &&
                     p.tip->offset != nullptr && p.tip->valid[idx] != 0;
  if (valid) {
    int ref_off = 0;
    int past_off = 0;
    int future_off = 0;
    const Status st =
        get_tip_offsets(p, &ref_off, &past_off, &future_off);
    if (st != Status::kOk) return st;
    const Mv mf{p.tip->mv[idx * 2 + 0], p.tip->mv[idx * 2 + 1]};
    cand[0] = project_mv(mf, past_off, ref_off);
    cand[1] = project_mv(mf, future_off, ref_off);
  }
  const MiCell& base = mi_at(p, cand_row, cand_col);
  for (int list = 0; list < 2; ++list) {
    cand[list].y += base.mv[0][0];
    cand[list].x += base.mv[0][1];
    cand[list].y = clip3(kMvLow + 1, kMvUpp - 1, cand[list].y);
    cand[list].x = clip3(kMvLow + 1, kMvUpp - 1, cand[list].x);
  }
  out[0] = cand[0];
  out[1] = cand[1];
  return Status::kOk;
}

Status get_motion_field_mv(const StackParams& p, std::int8_t dst, int y8,
                           int x8, Mv* out) {
  if (out == nullptr) return Status::kInvalidArgument;
  if (p.tip == nullptr || p.tip->mv == nullptr || p.tip->offset == nullptr) {
    return Status::kUnsupportedFeature;
  }
  if (dst < 0 || dst >= p.num_refs || p.order_hints == nullptr) {
    return Status::kInvalidReference;
  }
  const std::size_t idx =
      static_cast<std::size_t>(y8) * static_cast<std::size_t>(p.tip->cols8) +
      static_cast<std::size_t>(x8);
  Mv traj{};
  if (p.use_ref_frame_mvs && p.enable_mv_traj &&
      traj_available(p, dst, y8, x8, &traj)) {
    *out = traj;
    return Status::kOk;
  }
  const Mv mf{p.tip->mv[idx * 2 + 0], p.tip->mv[idx * 2 + 1]};
  const int ref_to_dst = relative_dist_hint(p, p.order_hint, dst);
  const int ref_offset = p.tip->offset[idx];
  *out = project_mv(mf, ref_to_dst, ref_offset);
  return Status::kOk;
}

// ---------------------------------------------------------------------------
// §7.12.2.15 / .17 / .18
// ---------------------------------------------------------------------------

Status derive_ref_mv_candidate_from_tip_mode(const StackParams& p,
                                             BuildState* state, int mv_row,
                                             int mv_col, int weight) {
  if (state == nullptr) return Status::kInvalidArgument;
  Mv cand[2];
  const Status st = get_tip_cand(p, mv_row, mv_col, cand);
  if (st != Status::kOk) return st;
  Stack& s = state->stack;
  bool found = false;
  if (state->prune_count < kMaxPrNum) {
    for (int i = 0; i < s.count; ++i) {
      ++state->prune_count;
      if (cand[0] == s.cand[i].mv[0] && cand[1] == s.cand[i].mv[1]) {
        s.cand[i].weight += weight;
        found = true;
        break;
      }
    }
  }
  if (!found && s.count < kMaxRefMvStackSize) {
    Candidate& c = s.cand[s.count];
    c = Candidate{};
    c.mv[0] = cand[0];
    c.mv[1] = cand[1];
    c.weight = weight;
    c.cwp = kCwpEqual;
    ++s.count;
  }
  ++state->trace.compound_tip;
  return Status::kOk;
}

Status derive_single_ref_mv_candidate_from_tip_mode(const StackParams& p,
                                                    BuildState* state,
                                                    int mv_row, int mv_col,
                                                    int weight,
                                                    int cand_list) {
  if (state == nullptr || cand_list < 0 || cand_list > 1) {
    return Status::kInvalidArgument;
  }
  Mv cand[2];
  const Status st = get_tip_cand(p, mv_row, mv_col, cand);
  if (st != Status::kOk) return st;
  Stack& s = state->stack;
  bool found = false;
  if (state->prune_count < kMaxPrNum) {
    for (int i = 0; i < s.count; ++i) {
      ++state->prune_count;
      if (cand[cand_list] == s.cand[i].mv[0]) {
        s.cand[i].weight += weight;
        found = true;
        break;
      }
    }
  }
  if (!found && s.count < kMaxRefMvStackSize) {
    Candidate& c = s.cand[s.count];
    c = Candidate{};
    c.mv[0] = cand[cand_list];
    c.mv[1] = cand[cand_list];
    c.weight = weight;
    c.cwp = kCwpEqual;
    ++s.count;
  }
  ++state->trace.tip_single_derived[cand_list];
  return Status::kOk;
}

Status tip_add_derived(const StackParams& p, BuildState* state, int mv_row,
                       int mv_col) {
  if (state == nullptr) return Status::kInvalidArgument;
  const MiCell& m = mi_at(p, mv_row, mv_col);
  int ref_off = 0;
  int past_off = 0;
  int future_off = 0;
  const Status st = get_tip_offsets(p, &ref_off, &past_off, &future_off);
  if (st != Status::kOk) return st;
  const Mv linear{m.sub_mv[0][0] - m.sub_mv[1][0],
                  m.sub_mv[0][1] - m.sub_mv[1][1]};
  const Mv proj = project_mv(linear, past_off, ref_off);
  Mv derived;
  derived.y = clip3(kMvLow + 1, kMvUpp - 1, m.sub_mv[0][0] - proj.y);
  derived.x = clip3(kMvLow + 1, kMvUpp - 1, m.sub_mv[0][1] - proj.x);
  if (!comp_mv_in_derived(state, derived)) {
    add_derived(state, derived, derived);
  }
  ++state->trace.tip_add_derived;
  return Status::kOk;
}

// ---------------------------------------------------------------------------
// §7.12.2.10 add reference motion vector
// ---------------------------------------------------------------------------

Status add_reference_mv(const StackParams& p, BuildState* state, int mv_row,
                        int mv_col, int weight) {
  if (state == nullptr) return Status::kInvalidArgument;
  const MiCell& m = mi_at(p, mv_row, mv_col);
  if (m.is_inter == 0) return Status::kOk;

  if (!p.is_compound) {
    const int list_end = 1 - (p.use_intrabc ? 1 : 0);
    for (int cand_list = 0; cand_list <= list_end; ++cand_list) {
      if (m.ref[cand_list] == p.ref_frame[0]) {
        ++state->trace.search_stack[cand_list];
        const Status st =
            search_stack(p, state, mv_row, mv_col, weight, cand_list);
        if (st != Status::kOk) return st;
      } else if (m.ref[0] == kTipFrame &&
                 p.ref_frame[0] ==
                     (cand_list ? p.closest_future : p.closest_past)) {
        // §7.12.2.17 derive single ref mv candidate from TIP mode.
        const Status st = derive_single_ref_mv_candidate_from_tip_mode(
            p, state, mv_row, mv_col, weight, cand_list);
        if (st != Status::kOk) return st;
      } else if (cand_list == 0 && p.ref_frame[0] == kTipFrame &&
                 p.closest_past == m.ref[0] &&
                 p.closest_future == m.ref[1]) {
        // §7.12.2.18 TIP add derived.
        const Status st = tip_add_derived(p, state, mv_row, mv_col);
        if (st != Status::kOk) return st;
      } else if (!p.use_intrabc &&
                 (m.ref[0] == kTipFrame ||
                  is_inter_ref_frame(m.ref[cand_list])) &&
                 p.ref_frame[0] != kTipFrame) {
        // §7.12.2.16 single add derived.
        ++state->trace.single_add_derived[cand_list];
        const Status st =
            single_add_derived(p, state, mv_row, mv_col, cand_list);
        if (st != Status::kOk) return st;
      }
    }
    return Status::kOk;
  }

  // isCompound == 1.
  if (m.ref[0] == kTipFrame && p.ref_frame[0] == p.closest_past &&
      p.ref_frame[1] == p.closest_future) {
    return derive_ref_mv_candidate_from_tip_mode(p, state, mv_row, mv_col,
                                                 weight);
  }
  if (m.ref[0] == p.ref_frame[0] && m.ref[1] == p.ref_frame[1]) {
    ++state->trace.compound_search;
    return compound_search_stack(p, state, mv_row, mv_col, weight);
  }
  ++state->trace.compound_add_derived;
  return compound_add_derived(p, state, mv_row, mv_col);
}

// ---------------------------------------------------------------------------
// §7.12.2.5 / .6 scanning
// ---------------------------------------------------------------------------

Status scan_point(const StackParams& p, BuildState* state, int delta_row,
                  int delta_col) {
  if (state == nullptr) return Status::kInvalidArgument;
  int mv_row = p.mi_row + delta_row;
  int mv_col = p.mi_col + delta_col;
  const bool sb_border = (p.mi_row & (p.sb_h4 - 1)) == 0;
  if (delta_row < 0 && sb_border) {
    mv_col = (mv_col >> 1) << 1;
    delta_col = mv_col - p.mi_col;
  }
  if (!p.grid.inside(mv_row, mv_col)) return Status::kOk;
  const MiCell& m = mi_at(p, mv_row, mv_col);
  if (m.written == 0) return Status::kOk;
  int weight = 1;
  if (delta_row == -1 && delta_col == -1) {
    weight = 0;
  } else if (delta_col < -1) {
    weight = 0;
  }
  // §7.12.2.9 runs before the §7.12.2.10 dispatch and before the stack-full
  // early exit that §7.12.2.6 specifies for the MV half.
  Status st = add_warp_motion_vector(p, state, mv_row, mv_col);
  if (st != Status::kOk) return st;
  if (state->stack.count >= kMaxRefMvStackSize) return Status::kOk;
  st = add_reference_mv(p, state, mv_row, mv_col, weight);
  if (st != Status::kOk) return st;
  return Status::kOk;
}

Status scan_col(const StackParams& p, BuildState* state, int delta_col) {
  if (state == nullptr) return Status::kInvalidArgument;
  delta_col += p.mi_col & 1;
  const auto scan_point_if_valid = [&](int delta_row, int dc) -> Status {
    const int mv_row = p.mi_row + delta_row;
    const int mv_col = p.mi_col + dc;
    if (!p.grid.inside(mv_row, mv_col)) return Status::kOk;
    if (p.mi_col - 1 < 0) return Status::kOk;
    if (mi_at(p, mv_row, mv_col).col_base ==
        mi_at(p, mv_row, p.mi_col - 1).col_base) {
      return Status::kOk;
    }
    return scan_point(p, state, delta_row, dc);
  };
  Status st = scan_point_if_valid(p.bh4 - 1, delta_col);
  if (st != Status::kOk) return st;
  if (p.bh4 > 1) {
    st = scan_point_if_valid(0, delta_col);
  }
  return st;
}

// ---------------------------------------------------------------------------
// §7.12.2.19 sorting process (one move-to-front; whole-record swap)
// ---------------------------------------------------------------------------

void sort_stack_range(const StackParams& p, Stack* stack, int start, int end) {
  (void)p;
  if (stack == nullptr) return;
  if (start < 0) start = 0;
  if (end > stack->count) end = stack->count;
  if (start + 1 >= end) return;
  int max_weight = stack->cand[start].weight;
  int max_idx = start;
  for (int idx = start + 1; idx < end; ++idx) {
    if (max_weight < stack->cand[idx].weight) {
      max_weight = stack->cand[idx].weight;
      max_idx = idx;
    }
  }
  if (max_idx != start) {
    const Candidate tmp = stack->cand[start];
    stack->cand[start] = stack->cand[max_idx];
    stack->cand[max_idx] = tmp;
  }
}

// ---------------------------------------------------------------------------
// §7.12.2.22 fill MVP from derived SMVP
// ---------------------------------------------------------------------------

Status fill_derived(const StackParams& p, BuildState* state) {
  if (state == nullptr) return Status::kInvalidArgument;
  Stack& s = state->stack;
  // §7.12.2.22 uses max_bvp_drl_bits_minus_1 + 2 for intra block copy; IBC is
  // out of this module's scope, so the DRL budget applies to both.
  const int max_ref_mv_count = p.max_drl_bits_minus_1 + 2;
  if (s.count >= max_ref_mv_count) return Status::kOk;
  for (int d = 0; d < state->derived_count; ++d) {
    bool found = false;
    if (state->prune_count < kMaxPrNum) {
      for (int i = 0; i < s.count; ++i) {
        ++state->prune_count;
        const bool match0 = state->derived[d].mv[0] == s.cand[i].mv[0];
        const bool match1 =
            !p.is_compound || state->derived[d].mv[1] == s.cand[i].mv[1];
        if (match0 && match1) {
          found = true;
          break;
        }
      }
    }
    if (!found && s.count < max_ref_mv_count) {
      Candidate& c = s.cand[s.count];
      c = Candidate{};
      c.mv[0] = state->derived[d].mv[0];
      c.mv[1] = p.is_compound ? state->derived[d].mv[1] : state->derived[d].mv[0];
      c.weight = 1;
      c.cwp = kCwpEqual;
      ++s.count;
    }
  }
  return Status::kOk;
}

// ---------------------------------------------------------------------------
// §7.12.2.20 extra search process
// ---------------------------------------------------------------------------

namespace {

Status insert_mvp_candidate(const StackParams& p, BuildState* state, int y_cand,
                            int x_cand) {
  Stack& s = state->stack;
  if (y_cand >= s.count || x_cand >= s.count) return Status::kOk;
  if (s.count >= kMaxRefMvStackSize) return Status::kOk;
  const Mv cand0{s.cand[y_cand].mv[0].y, s.cand[x_cand].mv[0].x};
  const Mv cand1{s.cand[y_cand].mv[1].y, s.cand[x_cand].mv[1].x};
  if (state->prune_count < kMaxPrNum) {
    for (int i = 0; i < s.count; ++i) {
      ++state->prune_count;
      const bool match0 = cand0 == s.cand[i].mv[0];
      if (!p.is_compound && match0) return Status::kOk;
      if (p.is_compound && match0 && cand1 == s.cand[i].mv[1]) {
        return Status::kOk;
      }
    }
  }
  Candidate& c = s.cand[s.count];
  c = Candidate{};
  c.mv[0] = cand0;
  c.mv[1] = p.is_compound ? cand1 : cand0;
  c.weight = 1;  // mirror only: WeightStack is not read after step 17
  c.cwp = kCwpEqual;
  ++s.count;
  return Status::kOk;
}

}  // namespace

Status extra_search(const StackParams& p, BuildState* state) {
  if (state == nullptr) return Status::kInvalidArgument;
  Stack& s = state->stack;
  const int num_lists = p.is_compound ? 2 : 1;

  // 1. Clamp the candidates that are already in the stack.  This happens
  //    *before* the global candidate is added, so a clamped stack candidate and
  //    an unclamped GlobalMvs[] compare on different values.
  for (int idx = 0; idx < s.count; ++idx) {
    Mv mv = s.cand[idx].mv[0];
    mv.y = clamp_mv_row(p, mv.y);
    mv.x = clamp_mv_col(p, mv.x);
    s.cand[idx].mv[0] = mv;
    // The spec only walks `list < 1 + isCompound`; list 1 of a single stack is
    // unused.  Keep it equal to list 0 so the record is canonical.
    if (!p.is_compound) s.cand[idx].mv[1] = mv;
  }
  if (p.is_compound) {
    for (int idx = 0; idx < s.count; ++idx) {
      Mv mv = s.cand[idx].mv[1];
      mv.y = clamp_mv_row(p, mv.y);
      mv.x = clamp_mv_col(p, mv.x);
      s.cand[idx].mv[1] = mv;
    }
  }

  // 2. Global-MV candidate.
  if (s.count < kMaxRefMvStackSize && !p.use_intrabc) {
    bool found = false;
    if (state->prune_count < kMaxPrNum) {
      for (int idx = 0; idx < s.count; ++idx) {
        ++state->prune_count;
        if (p.global_mv[0] == s.cand[idx].mv[0]) {
          if (!p.is_compound || p.global_mv[1] == s.cand[idx].mv[1]) {
            found = true;
            break;
          }
        }
      }
    }
    if (!found) {
      Candidate& c = s.cand[s.count];
      c = Candidate{};
      for (int list = 0; list < num_lists; ++list) c.mv[list] = p.global_mv[list];
      if (!p.is_compound) c.mv[1] = c.mv[0];
      c.weight = 1;  // mirror only
      c.cwp = kCwpEqual;
      ++s.count;
    }
  }

  // 3. MVP combinations, gated by Block_Width > 32 && Block_Height > 32.
  const int num = s.count;
  if (mvp_combinations_allowed(p.bw4 * kMiSize, p.bh4 * kMiSize)) {
    if (num > 1) {
      Status st = insert_mvp_candidate(p, state, 0, 1);
      if (st != Status::kOk) return st;
      st = insert_mvp_candidate(p, state, 1, 0);
      if (st != Status::kOk) return st;
    }
    if (num > 2) {
      Status st = insert_mvp_candidate(p, state, 0, 2);
      if (st != Status::kOk) return st;
      st = insert_mvp_candidate(p, state, 2, 0);
      if (st != Status::kOk) return st;
      st = insert_mvp_candidate(p, state, 1, 2);
      if (st != Status::kOk) return st;
      st = insert_mvp_candidate(p, state, 2, 1);
      if (st != Status::kOk) return st;
    }
  }

  // 4. DeriveWrl warp candidates: bank newest-first, gm_params, two defaults.
  if (p.derive_wrl) {
    const std::int8_t ref = p.ref_frame[0];
    if (ref >= 0 && ref < kRefsPerFrame) {
      if (p.warp_bank != nullptr) {
        const int c = p.warp_bank->size();
        const int start = p.warp_bank->start();
        for (int i = c - 1; i >= 0; --i) {
          const int idx = (start + i) % kWarpParamBankSize;
          const Status st = insert_warp_candidate(p.warp_bank->slot(idx), &s);
          if (st != Status::kOk) return st;
        }
      }
      Status st = insert_warp_candidate(p.gm_params[0], &s);
      if (st != Status::kOk) return st;
      for (int i = 0; i < 2; ++i) {
        st = insert_warp_candidate(kDefaultWarpParams, &s);
        if (st != Status::kOk) return st;
      }
    }
  }
  return Status::kOk;
}

// ---------------------------------------------------------------------------
// §7.12.2.23 clamping process
// ---------------------------------------------------------------------------

Status clamp_stack(const StackParams& p, Stack* stack) {
  if (stack == nullptr) return Status::kInvalidArgument;
  if (p.use_intrabc) return Status::kOk;  // §7.12.2.23 guards on use_intrabc
  const int num_lists = p.is_compound ? 2 : 1;
  for (int list = 0; list < num_lists; ++list) {
    for (int idx = 0; idx < stack->count; ++idx) {
      Mv mv = stack->cand[idx].mv[list];
      mv.y = clamp_mv_row(p, mv.y);
      mv.x = clamp_mv_col(p, mv.x);
      stack->cand[idx].mv[list] = mv;
    }
  }
  if (!p.is_compound) {
    for (int idx = 0; idx < stack->count; ++idx) {
      stack->cand[idx].mv[1] = stack->cand[idx].mv[0];
    }
  }
  return Status::kOk;
}

// ---------------------------------------------------------------------------
// §7.12.2.3 / .4 corners
// ---------------------------------------------------------------------------

namespace {

Status warp_corner(const StackParams& p, int delta_row, int delta_col,
                   int adjust_col, CornerResult* out) {
  const int mv_row = p.mi_row + delta_row;
  const int mv_col = p.mi_col + delta_col;
  const bool sb_border = (p.mi_row & (p.sb_h4 - 1)) == 0;
  const int adjusted_col = delta_col + adjust_col;
  int mv_col2;
  if (delta_row < 0 && sb_border) {
    mv_col2 = (p.mi_col - (p.mi_col & 1)) +
              (adjusted_col - (adjusted_col & 1));
  } else {
    mv_col2 = p.mi_col + adjusted_col;
  }
  if (!p.grid.inside(mv_row, mv_col2)) return Status::kOk;
  const MiCell& m = mi_at(p, mv_row, mv_col2);
  for (int ref = 0; ref < 2; ++ref) {
    if (m.written == 0) return Status::kOk;
    if (m.is_inter == 0) return Status::kOk;
    if (m.ref[ref] != p.ref_frame[0]) continue;
    const int n = out->corners_found;
    if (n >= 3) return Status::kOk;
    out->pts[n][0] = (mv_row + 1) * kMiSize;
    out->pts[n][1] = (mv_col + 1) * kMiSize;
    if (m.motion_mode >= kMotionLocalWarp) {
      if (ref > 0) return Status::kOk;
      out->mvs[n] = get_warp_motion_vector_xy_pos(m.warp[ref], mv_row + 1,
                                                  mv_col + 1, p.grid.rows,
                                                  p.grid.cols);
    } else {
      out->mvs[n] = Mv{m.sub_mv[ref][0], m.sub_mv[ref][1]};
    }
    ++out->corners_found;
    return Status::kOk;
  }
  return Status::kOk;
}

}  // namespace

Status generate_points_from_corners(const StackParams& p, BuildState* state,
                                    int iter, CornerResult* out) {
  if (state == nullptr || out == nullptr) return Status::kInvalidArgument;
  *out = CornerResult{};
  Status st = warp_corner(p, -1, -1, iter, out);
  if (st != Status::kOk) return st;
  st = warp_corner(p, -1, p.bw4 - 1, iter, out);
  if (st != Status::kOk) return st;
  st = warp_corner(p, p.bh4 - 1, -1, 0, out);
  if (st != Status::kOk) return st;

  if (out->corners_found != 3) return Status::kOk;

  const std::int64_t one = static_cast<std::int64_t>(1) << 16;
  out->all_mvs_same = true;
  for (int n = 0; n < 3; ++n) {
    for (int c = 0; c < 2; ++c) {
      const std::int32_t mv_comp = c == 0 ? out->mvs[n].y : out->mvs[n].x;
      out->ref_pts[n][c] = static_cast<int>(
          (static_cast<std::int64_t>(out->pts[n][c]) << 16) +
          (static_cast<std::int64_t>(mv_comp) << 13));
      const std::int32_t mv0 = c == 0 ? out->mvs[0].y : out->mvs[0].x;
      if (mv_comp != mv0) out->all_mvs_same = false;
      if (out->ref_pts[n][c] < 0) out->negative_ref_pts = true;
    }
  }
  (void)one;
  if (out->all_mvs_same) return Status::kOk;
  if (out->negative_ref_pts) return Status::kOk;

  const int width_log2 = floor_log2(p.bw4) + 2;
  const int height_log2 = floor_log2(p.bh4) + 2;
  const int y0 = out->pts[0][0];
  const int x0 = out->pts[0][1];
  adv::WarpParams wmmat{};
  wmmat.p[2] = (out->ref_pts[1][1] - out->ref_pts[0][1]) >> width_log2;
  wmmat.p[4] = (out->ref_pts[1][0] - out->ref_pts[0][0]) >> width_log2;
  wmmat.p[3] = (out->ref_pts[2][1] - out->ref_pts[0][1]) >> height_log2;
  wmmat.p[5] = (out->ref_pts[2][0] - out->ref_pts[0][0]) >> height_log2;
  const std::int64_t wmmat0 = static_cast<std::int64_t>(out->ref_pts[0][1]) -
                              static_cast<std::int64_t>(wmmat.p[2]) * x0 -
                              static_cast<std::int64_t>(wmmat.p[3]) * y0;
  const std::int64_t wmmat1 = static_cast<std::int64_t>(out->ref_pts[0][0]) -
                              static_cast<std::int64_t>(wmmat.p[4]) * x0 -
                              static_cast<std::int64_t>(wmmat.p[5]) * y0;
  wmmat = adv::reduce_warp_model(wmmat);
  const std::int64_t high =
      static_cast<std::int64_t>(adv::kWarpedModelTransClamp) -
      (static_cast<std::int64_t>(1) << adv::kWarpParamReduceBits);
  wmmat.p[0] = static_cast<int>(
      clip64(-adv::kWarpedModelTransClamp, high, wmmat0));
  wmmat.p[1] = static_cast<int>(
      clip64(-adv::kWarpedModelTransClamp, high, wmmat1));
  out->wmmat = wmmat;
  st = insert_warp_candidate(wmmat, &state->stack);
  if (st == Status::kOk) out->inserted = true;
  return st;
}

// ---------------------------------------------------------------------------
// §7.11.4 scan point warp context
// ---------------------------------------------------------------------------

namespace {

void scan_point_warp_context(const StackParams& p, int delta_row, int delta_col,
                             WarpContextResult* out) {
  int mv_row = p.mi_row + delta_row;
  int mv_col = p.mi_col + delta_col;
  const bool sb_border = (p.mi_row & (p.sb_h4 - 1)) == 0;
  if (delta_row < 0 && sb_border) delta_col -= p.mi_col & 1;
  mv_col = p.mi_col + delta_col;
  if (!p.grid.inside(mv_row, mv_col)) return;
  const MiCell& m = mi_at(p, mv_row, mv_col);
  if (m.written == 0 || m.is_inter == 0) return;
  if (m.ref[0] == p.ref_frame[0] || m.ref[1] == p.ref_frame[0]) {
    if (!out->warp_sample_found[0]) {
      out->extend_delta_row = delta_row;
      out->extend_delta_col = delta_col;
    }
    out->warp_sample_found[0] = true;
    if (m.motion_mode >= kMotionLocalWarp) ++out->warp_mv_count;
  }
  if (m.ref[0] == p.ref_frame[1] || m.ref[1] == p.ref_frame[1]) {
    out->warp_sample_found[1] = true;
  }
}

}  // namespace

Status compute_warp_context(const StackParams& p, WarpContextResult* out) {
  if (out == nullptr) return Status::kInvalidArgument;
  if (!p.grid.inside(p.mi_row, p.mi_col)) return Status::kInvalidArgument;
  *out = WarpContextResult{};
  const bool sb_border = (p.mi_row & (p.sb_h4 - 1)) == 0;
  const int above_right_col = p.bw4 - 1 - (sb_border ? 1 : 0);
  scan_point_warp_context(p, p.bh4 - 1, -1, out);
  scan_point_warp_context(p, -1, above_right_col > 0 ? above_right_col : 0,
                          out);
  if (p.bh4 >= 2) scan_point_warp_context(p, 0, -1, out);
  if (p.bw4 >= (sb_border ? 4 : 2)) scan_point_warp_context(p, -1, 0, out);
  if (p.bh4 <= 16) scan_point_warp_context(p, p.bh4, -1, out);
  if (p.bw4 <= 16) {
    scan_point_warp_context(p, -1,
                            sb_border ? ((p.bw4 > 2) ? p.bw4 : 2) : p.bw4, out);
  }
  scan_point_warp_context(p, -1, -1 - (sb_border ? 1 : 0), out);
  return Status::kOk;
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

Status validate(const StackParams& p, std::string* why) {
  const auto fail = [&](Status s, const char* message) {
    if (why != nullptr) *why = message;
    return s;
  };
  if (p.grid.mi == nullptr || p.grid.cols <= 0 || p.grid.rows <= 0) {
    return fail(Status::kInvalidArgument, "null or empty MI grid");
  }
  if (p.bw4 < 1 || p.bh4 < 1 || p.bw4 > 16 || p.bh4 > 16) {
    return fail(Status::kInvalidBlockSize, "block extent outside 1..16 MI");
  }
  if (p.sb_h4 <= 0 || (p.sb_h4 & (p.sb_h4 - 1)) != 0) {
    return fail(Status::kInvalidArgument, "SbSize is not a power of two");
  }
  if (p.mi_row < 0 || p.mi_col < 0 || p.mi_row >= p.grid.rows ||
      p.mi_col >= p.grid.cols) {
    return fail(Status::kInvalidArgument, "block origin outside the grid");
  }
  if (p.mi_row + p.bh4 > p.grid.rows ||
      p.mi_col + p.bw4 > p.grid.cols) {
    return fail(Status::kInvalidBlockSize, "block extends past the grid");
  }
  if (p.max_drl_bits_minus_1 < 0 || p.max_drl_bits_minus_1 > 4) {
    return fail(Status::kInvalidArgument,
                "max_drl_bits_minus_1 outside 0..4");
  }
  if (p.frame_mv_precision < 0 ||
      p.frame_mv_precision > kMvPrecisionEighthPel) {
    return fail(Status::kInvalidArgument, "MV precision outside 0..6");
  }
  if (!p.is_compound) {
    if (p.ref_frame[1] != kRefNone) {
      return fail(Status::kInvalidReference,
                  "single block with a list-1 reference");
    }
    const std::int8_t r = p.ref_frame[0];
    if (r == kRefNone || r == kIntraFrame || r > kTipFrame) {
      return fail(Status::kInvalidReference, "invalid RefFrame[0]");
    }
    if (r == kTipFrame && p.tip == nullptr) {
      return fail(Status::kUnsupportedFeature,
                  "TIP reference without a motion field");
    }
  } else {
    if (!is_inter_ref_frame(p.ref_frame[0]) ||
        !is_inter_ref_frame(p.ref_frame[1]) ||
        p.ref_frame[0] == kTipFrame || p.ref_frame[1] == kTipFrame) {
      return fail(Status::kInvalidReference, "invalid compound reference pair");
    }
  }
  if (p.num_refs < 0 || p.num_refs > kRefsPerFrame) {
    return fail(Status::kInvalidStackParams, "num_refs outside 0..7");
  }
  if (p.num_refs > 0 &&
      (p.frame_distance == nullptr || p.order_hints == nullptr)) {
    return fail(Status::kInvalidStackParams,
                "num_refs set without FrameDistance/OrderHints");
  }
  return Status::kOk;
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

Status build(const StackParams& params, Stack* stack, BuildState* state) {
  if (stack == nullptr) return Status::kInvalidArgument;
  std::string why;
  const Status status = validate(params, &why);
  if (status != Status::kOk) return status;

  BuildState local;
  BuildState& st = state != nullptr ? *state : local;
  st = BuildState{};
  Stack& s = st.stack;
  for (int i = 0; i < kMaxWarpRefCandidates; ++i) {
    s.warp_params[i] = kDefaultWarpParams;
  }

  const std::int8_t ref0 = params.ref_frame[0];
  st.is_sb_border = (params.mi_row & (params.sb_h4 - 1)) == 0;
  st.use_temporal =
      params.use_ref_frame_mvs && !params.use_intrabc && ref0 != kTipFrame &&
      (params.skip_mode || ref0 != params.ref_frame[1]);
  st.use_temporal_first =
      params.reorder != Reorder::kAlways && params.use_ref_frame_mvs &&
      params.ref_frame[1] == kRefNone && is_inter_ref_frame(ref0) &&
      ref0 != kTipFrame &&
      (params.closest_future == kRefNone || params.closest_past == kRefNone) &&
      params.order_hints != nullptr &&
      std::abs(relative_dist_hint(params, params.order_hint, ref0)) <= 2;

  // Steps 1-2.  Setup global MV for each list that this block uses.  The
  // results become GlobalMvs[list]; StackParams is const, so the builder works
  // on a local copy.
  StackParams mutable_params = params;
  for (int list = 0; list < (params.is_compound ? 2 : 1); ++list) {
    Mv gm;
    const Status gm_status = setup_global_mv(params, list, &gm);
    if (gm_status != Status::kOk) return gm_status;
    mutable_params.global_mv[list] = gm;
  }
  const StackParams& p = mutable_params;

  CornerResult corners;
  if (p.derive_wrl) {
    const Status s3 = generate_points_from_corners(p, &st, 0, &corners);
    if (s3 != Status::kOk) return s3;
    // Step 4.
    if (s.num_warp_found == 0 && p.bw4 <= 16) {
      const Status s4 = generate_points_from_corners(p, &st, 1, &corners);
      if (s4 != Status::kOk) return s4;
    }
  }

  // Step 5.
  if (st.use_temporal && st.use_temporal_first && p.temporal_scan != nullptr) {
    p.temporal_scan(p.temporal_opaque, &s, &st.prune_count);
  }

  // Steps 6-11.
  const int above_right_col = p.bw4 - 1 - (st.is_sb_border ? 1 : 0);
  Status scan = scan_point(p, &st, p.bh4 - 1, -1);
  if (scan != Status::kOk) return scan;
  scan = scan_point(p, &st, -1, above_right_col > 0 ? above_right_col : 0);
  if (scan != Status::kOk) return scan;
  if (p.bh4 >= 2) {
    scan = scan_point(p, &st, 0, -1);
    if (scan != Status::kOk) return scan;
  }
  if (p.bw4 >= (st.is_sb_border ? 4 : 2)) {
    scan = scan_point(p, &st, -1, 0);
    if (scan != Status::kOk) return scan;
  }
  if (p.bh4 <= 16) {
    scan = scan_point(p, &st, p.bh4, -1);
    if (scan != Status::kOk) return scan;
  }
  if (p.bw4 <= 16) {
    scan = scan_point(p, &st, -1,
                      st.is_sb_border ? ((p.bw4 > 2) ? p.bw4 : 2) : p.bw4);
    if (scan != Status::kOk) return scan;
  }

  // Step 12.
  if (st.use_temporal && !st.use_temporal_first &&
      p.temporal_scan != nullptr) {
    p.temporal_scan(p.temporal_opaque, &s, &st.prune_count);
  }

  // Step 13.
  scan = scan_point(p, &st, -1, -1 - (st.is_sb_border ? 1 : 0));
  if (scan != Status::kOk) return scan;

  // Step 14.
  s.num_nearest = s.count;

  // Step 15.
  scan = scan_col(p, &st, -3);
  if (scan != Status::kOk) return scan;

  // Steps 16-17.
  const bool use_sort =
      p.reorder == Reorder::kAlways ||
      (p.reorder == Reorder::kConstraint && !st.use_temporal_first &&
       s.num_nearest >= 4);
  s.sorted = use_sort;
  if (use_sort) sort_stack_range(p, &s, 0, s.num_nearest);

  // Step 18.
  if (p.is_compound) {
    const Status st18 = fill_derived(p, &st);
    if (st18 != Status::kOk) return st18;
  }

  // Step 19: fill MVP from the ref-MV bank is a separate component and is off
  // for this stack (StackParams has no enable_refmvbank; the caller splices it
  // between step 18 and 20 if it needs it).

  // Step 20.
  if (!p.is_compound) {
    const Status st20 = fill_derived(p, &st);
    if (st20 != Status::kOk) return st20;
  }

  // Step 21.
  s.scanned_count = s.count;
  const Status extra = extra_search(p, &st);
  if (extra != Status::kOk) return extra;

  // Step 22.
  const Status clamp = clamp_stack(p, &s);
  if (clamp != Status::kOk) return clamp;

  s.prune_count = st.prune_count;
  s.single_prune_count = st.single_prune_count;
  s.derived_prune_count = st.derived_prune_count;
  s.derived_count = st.derived_count;
  st.prune_count = s.prune_count;
  st.derived_count = s.derived_count;

  *stack = s;
  return Status::kOk;
}

// ---------------------------------------------------------------------------
// Equality
// ---------------------------------------------------------------------------

bool equal(const Stack& a, const Stack& b) {
  if (a.count != b.count || a.num_nearest != b.num_nearest ||
      a.sorted != b.sorted || a.num_warp_found != b.num_warp_found) {
    return false;
  }
  for (int i = 0; i < a.count; ++i) {
    if (!(a.cand[i].mv[0] == b.cand[i].mv[0]) ||
        !(a.cand[i].mv[1] == b.cand[i].mv[1]) ||
        a.cand[i].weight != b.cand[i].weight ||
        a.cand[i].row_offset != b.cand[i].row_offset ||
        a.cand[i].col_offset != b.cand[i].col_offset ||
        a.cand[i].cwp != b.cand[i].cwp) {
      return false;
    }
  }
  for (int i = 0; i < a.num_warp_found; ++i) {
    for (int j = 0; j < 6; ++j) {
      if (a.warp_params[i].p[j] != b.warp_params[i].p[j]) return false;
    }
  }
  return true;
}

int first_divergence(const Stack& a, const Stack& b) {
  const int n = std::min(a.count, b.count);
  for (int i = 0; i < n; ++i) {
    if (!(a.cand[i].mv[0] == b.cand[i].mv[0]) ||
        !(a.cand[i].mv[1] == b.cand[i].mv[1]) ||
        a.cand[i].weight != b.cand[i].weight ||
        a.cand[i].row_offset != b.cand[i].row_offset ||
        a.cand[i].col_offset != b.cand[i].col_offset ||
        a.cand[i].cwp != b.cand[i].cwp) {
      return i;
    }
  }
  return n;
}

}  // namespace av2enc::inter::mv_stack_full

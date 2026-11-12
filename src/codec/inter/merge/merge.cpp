// AV2 skip / merge / DRL semantics — implementation.
//
// Spec-only: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7).  No encoder code from AVM, SVT-AV1/AV2,
// clave, or av2-turbo was consulted.

#include "codec/inter/merge/merge.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>

namespace av2enc::video::merge {
namespace {

constexpr std::int8_t kCwpEqual = 8;

// §7.9.4 Div_Mult[32].
constexpr std::int32_t kDivMult[32] = {
    0,     16384, 8192, 5461, 4096, 3276, 2730, 2340, 2048, 1820, 1638,
    1489,  1365,  1260, 1170, 1092, 1024, 963,  910,  862,  819,  780,
    744,   712,   682,  655,  630,  606,  585,  564,  546,  528};

int block_width_px(BlockSize size) { return partition::block_width(size); }
int block_height_px(BlockSize size) { return partition::block_height(size); }

bool valid_block_size(BlockSize size) {
  return size != BlockSize::kInvalid && partition::is_valid_block_size(size);
}

std::int32_t clip3(std::int32_t low, std::int32_t high, std::int32_t value) {
  return value < low ? low : (value > high ? high : value);
}

std::int64_t round2_signed64(std::int64_t x, int n) {
  if (n == 0) {
    return x;
  }
  return x >= 0 ? ((x + (static_cast<std::int64_t>(1) << (n - 1))) >> n)
                : -(((-x) + (static_cast<std::int64_t>(1) << (n - 1))) >> n);
}

int relative_dist_restricted(int a, bool a_restricted, int b,
                             bool b_restricted) {
  if (a_restricted && b_restricted) {
    return 0;
  }
  if (a_restricted) {
    return 127;
  }
  if (b_restricted) {
    return -127;
  }
  return clip3(-127, 127, a - b);
}

// §7.9.4 get_mv_projection for the derived-stack path.
inter::mc::Mv project_mv(inter::mc::Mv mv, int numerator, int denominator) {
  const int clipped_denominator = std::min(31, denominator);
  const int clipped_numerator = clip3(-31, 31, numerator);
  inter::mc::Mv out;
  out.y = clip3(
      inter::mc::kMvLow + 1, inter::mc::kMvUpp - 1,
      static_cast<std::int32_t>(round2_signed64(
          static_cast<std::int64_t>(mv.y) * clipped_numerator *
              kDivMult[clipped_denominator],
          14)));
  out.x = clip3(
      inter::mc::kMvLow + 1, inter::mc::kMvUpp - 1,
      static_cast<std::int32_t>(round2_signed64(
          static_cast<std::int64_t>(mv.x) * clipped_numerator *
              kDivMult[clipped_denominator],
          14)));
  return out;
}

bool has_newmv_mode(int y_mode) {
  return y_mode == kYModeNewMv || y_mode == kYModeNewNewMv ||
         y_mode == kYModeNearNewMv || y_mode == kYModeNewNearMv ||
         y_mode == kYModeWarpNewMv || y_mode == kYModeJointNewMv;
}

bool has_newmv_for_list(int y_mode, int ref_list) {
  if (y_mode == kYModeNewNewMv || y_mode == kYModeNewMv) {
    return true;
  }
  if (ref_list == 0) {
    return y_mode == kYModeNewNearMv || y_mode == kYModeJointNewMv;
  }
  return y_mode == kYModeNearNewMv;
}

bool mi_inside(int row, int col, const StackParams& p) {
  return row >= 0 && col >= 0 && row < p.mi_rows && col < p.mi_cols;
}

const inter::mc::MiInfo& mi_at(const StackParams& p, int row, int col) {
  return p.mi[static_cast<std::size_t>(row) *
                  static_cast<std::size_t>(p.mi_cols) +
              static_cast<std::size_t>(col)];
}

inter::mc::Mv get_mv(const inter::mc::MiInfo& mi, int list) {
  return inter::mc::Mv{mi.mv[list][0], mi.mv[list][1]};
}

// ---------------------------------------------------------------------------
// Stack builder
// ---------------------------------------------------------------------------

struct BuildOptions {
  bool gate_mvp_combinations = true;
  bool mc_above_left_scan = false;
  bool include_derived = true;
  bool extra_mv_low_clamp = false;
};

struct DerivedCandidate {
  inter::mc::Mv mv[2];
};

struct SingleCandidate {
  std::int8_t ref = kRefNone;
  inter::mc::Mv mv;
};

class StackBuilder {
 public:
  StackBuilder(const StackParams& params, MvStack* stack,
               const BuildOptions& options)
      : p_(params), stack_(stack), options_(options) {}

  void run() {
    const bool sb_border = (p_.mi_row & (p_.sb_h4 - 1)) == 0;

    // Steps 6-13.
    scan_point(p_.bh4 - 1, -1);
    const int above_right_col = p_.bw4 - 1 - (sb_border ? 1 : 0);
    scan_point(-1, above_right_col > 0 ? above_right_col : 0);
    if (p_.bh4 >= 2) {
      scan_point(0, -1);
    }
    if (p_.bw4 >= (sb_border ? 4 : 2)) {
      scan_point(-1, 0);
    }
    if (p_.bh4 <= 16) {
      scan_point(p_.bh4, -1);
    }
    if (p_.bw4 <= 16) {
      scan_point(-1, sb_border ? ((p_.bw4 > 2) ? p_.bw4 : 2) : p_.bw4);
    }
    if (options_.mc_above_left_scan) {
      // Current inter::mc behaviour: the spec call is
      // scan_point(-1, -1 - isSbBorder).
      scan_point(-1, -1);
    } else {
      scan_point(-1, -1 - (sb_border ? 1 : 0));
    }

    // Step 14.
    stack_->num_nearest = stack_->count;

    // Step 15.
    scan_col(-3);

    // Steps 16-17.
    const bool use_sort =
        p_.reorder == DrlReorder::kAlways ||
        (p_.reorder == DrlReorder::kConstraint && !p_.use_temporal_first &&
         stack_->num_nearest >= 4);
    stack_->sorted = use_sort;
    if (use_sort) {
      sort_nearest();
    }

    // Steps 18 and 20: fill mvp from derived smvp.  Temporal tools are off in
    // M2; the derived stack can still be non-empty for multi-reference blocks
    // because §7.12.2.16 projects a neighbour MV onto the current reference.
    if (options_.include_derived) {
      fill_derived();
    }

    // Step 21: pre-clamp existing candidates.
    clamp_all();

    // Step 21: add a global-MV candidate if it is not already present.
    if (stack_->count < kMaxRefMvStackSize) {
      bool found = false;
      if (prune_count_ < kMaxPrNum) {
        for (int i = 0; i < stack_->count; ++i) {
          ++prune_count_;
          const bool match0 = p_.global_mv[0] == stack_->cand[i].mv[0];
          const bool match1 =
              !p_.is_compound || p_.global_mv[1] == stack_->cand[i].mv[1];
          if (match0 && match1) {
            found = true;
            break;
          }
        }
      }
      if (!found) {
        StackCandidate& c = stack_->cand[stack_->count];
        c.mv[0] = p_.global_mv[0];
        c.mv[1] = p_.is_compound ? p_.global_mv[1] : p_.global_mv[0];
        c.weight = 1;
        c.row_offset = 0;
        c.col_offset = 0;
        c.cwp = kCwpEqual;
        ++stack_->count;
      }
    }

    // Step 21: MVP-combination candidates, gated by Block_Width > 32 and
    // Block_Height > 32.
    const bool gate = options_.gate_mvp_combinations
                          ? mvp_combinations_allowed(block_width_for_params(),
                                                     block_height_for_params())
                          : true;
    if (gate) {
      const int num = stack_->count;
      if (num > 1) {
        insert_mvp_candidate(0, 1);
        insert_mvp_candidate(1, 0);
      }
      if (num > 2) {
        insert_mvp_candidate(0, 2);
        insert_mvp_candidate(2, 0);
        insert_mvp_candidate(1, 2);
        insert_mvp_candidate(2, 1);
      }
    }

    // Step 22.
    clamp_all();
  }

 private:
  int block_width_for_params() const { return p_.bw4 * 4; }
  int block_height_for_params() const { return p_.bh4 * 4; }

  bool same_mv(const inter::mc::Mv& a, const inter::mc::Mv& b) const {
    return a == b;
  }

  void search_stack(inter::mc::Mv cand0, inter::mc::Mv cand1, int mv_row,
                    int mv_col, int weight) {
    int found = -1;
    if (prune_count_ < kMaxPrNum) {
      for (int i = 0; i < stack_->count; ++i) {
        ++prune_count_;
        const bool match0 = same_mv(cand0, stack_->cand[i].mv[0]);
        const bool match1 =
            !p_.is_compound || same_mv(cand1, stack_->cand[i].mv[1]);
        if (match0 && match1) {
          found = i;
          break;
        }
      }
    }
    if (found >= 0) {
      stack_->cand[found].weight += weight;
      return;
    }
    if (stack_->count < kMaxRefMvStackSize) {
      StackCandidate& c = stack_->cand[stack_->count];
      c.mv[0] = cand0;
      c.mv[1] = p_.is_compound ? cand1 : cand0;
      c.weight = weight;
      c.row_offset = mv_row - p_.mi_row;
      c.col_offset = mv_col - p_.mi_col;
      c.cwp = kCwpEqual;
      ++stack_->count;
    }
  }

  void add_derived(inter::mc::Mv mv0, inter::mc::Mv mv1) {
    if (derived_count_ >= kMaxDrStackSize) {
      return;
    }
    derived_[derived_count_].mv[0] = mv0;
    derived_[derived_count_].mv[1] = mv1;
    ++derived_count_;
  }

  bool comp_mv_in_derived(inter::mc::Mv mv0, inter::mc::Mv mv1) {
    if (derived_prune_count_ < kMaxDrPrNum) {
      for (int i = 0; i < derived_count_; ++i) {
        ++derived_prune_count_;
        if (derived_[i].mv[0] == mv0 && derived_[i].mv[1] == mv1) {
          return true;
        }
      }
    }
    return false;
  }

  // §7.12.2.16 Single add derived process (non-temporal subset).
  void single_add_derived(int mv_row, int mv_col, int cand_list) {
    const inter::mc::MiInfo& m = mi_at(p_, mv_row, mv_col);
    const std::int8_t cand_ref = m.ref[cand_list];
    if (!is_inter_ref(cand_ref) || cand_ref == kTipFrame) {
      return;
    }
    if (p_.frame_distance == nullptr || p_.num_refs <= 0) {
      return;
    }
    const int cur_ref = p_.ref_frame[0];
    if (cur_ref < 0 || cur_ref >= p_.num_refs || cand_ref >= p_.num_refs) {
      return;
    }
    const int cur_dist = p_.frame_distance[cur_ref];
    const int cand_dist = p_.frame_distance[cand_ref];
    if (!((cur_dist > 0 && cand_dist > 0) ||
          (cur_dist < 0 && cand_dist < 0))) {
      return;
    }
    const inter::mc::Mv cand_mv = get_mv(m, cand_list);
    const inter::mc::Mv projected =
        project_mv(cand_mv, std::abs(cur_dist), std::abs(cand_dist));
    if (!comp_mv_in_derived(projected, projected)) {
      add_derived(projected, projected);
    }
  }

  // §7.12.2.14 Compound add derived process (non-temporal subset).
  void compound_add_derived(int mv_row, int mv_col) {
    const inter::mc::MiInfo& m = mi_at(p_, mv_row, mv_col);
    int cand_ref_idx0 = 0;
    int cand_ref_idx1 = 1;
    if (m.ref[0] == p_.ref_frame[0] || m.ref[1] == p_.ref_frame[0]) {
      cand_ref_idx0 = 0;
      cand_ref_idx1 = 1;
    } else if (m.ref[0] == p_.ref_frame[1] || m.ref[1] == p_.ref_frame[1]) {
      cand_ref_idx0 = 1;
      cand_ref_idx1 = 0;
    } else {
      return;
    }
    const int cand_list = (m.ref[0] == p_.ref_frame[cand_ref_idx0]) ? 0 : 1;
    const inter::mc::Mv cand_mv = get_mv(m, cand_list);

    for (int i = 0; i < single_count_ && derived_count_ < kMaxDrStackSize;
         ++i) {
      if (single_[i].ref != p_.ref_frame[cand_ref_idx1]) {
        continue;
      }
      const inter::mc::Mv l0 =
          cand_ref_idx0 == 0 ? cand_mv : single_[i].mv;
      const inter::mc::Mv l1 =
          cand_ref_idx0 == 1 ? cand_mv : single_[i].mv;
      if (!comp_mv_in_derived(l0, l1)) {
        add_derived(l0, l1);
      }
      break;
    }

    if (single_prune_count_ < kMaxDrPrNum) {
      for (int i = 0; i < single_count_; ++i) {
        ++single_prune_count_;
        if (single_[i].ref == p_.ref_frame[cand_ref_idx0] &&
            single_[i].mv == cand_mv) {
          return;
        }
      }
    }
    if (single_count_ < kMaxDrStackSize) {
      single_[single_count_].ref = p_.ref_frame[cand_ref_idx0];
      single_[single_count_].mv = cand_mv;
      ++single_count_;
    }
  }

  void compound_search_stack(int mv_row, int mv_col, int weight) {
    const inter::mc::MiInfo& m = mi_at(p_, mv_row, mv_col);
    search_stack(get_mv(m, 0), get_mv(m, 1), mv_row, mv_col, weight);
  }

  void add_reference_mv(int mv_row, int mv_col, int weight) {
    const inter::mc::MiInfo& m = mi_at(p_, mv_row, mv_col);
    if (!m.is_inter) {
      return;
    }
    if (!p_.is_compound) {
      for (int list = 0; list < 2; ++list) {
        if (m.ref[list] == p_.ref_frame[0]) {
          const inter::mc::Mv mv = get_mv(m, list);
          search_stack(mv, mv, mv_row, mv_col, weight);
        } else if (is_inter_ref(m.ref[list])) {
          single_add_derived(mv_row, mv_col, list);
        }
      }
      return;
    }

    if (m.ref[0] == kTipFrame || m.ref[1] == kTipFrame) {
      return;  // TIP is disabled in M2.
    }
    if (m.ref[0] == p_.ref_frame[0] && m.ref[1] == p_.ref_frame[1]) {
      compound_search_stack(mv_row, mv_col, weight);
    } else {
      compound_add_derived(mv_row, mv_col);
    }
  }

  void scan_point(int delta_row, int delta_col) {
    int mv_row = p_.mi_row + delta_row;
    int mv_col = p_.mi_col + delta_col;
    const bool sb_border = (p_.mi_row & (p_.sb_h4 - 1)) == 0;
    if (delta_row < 0 && sb_border) {
      mv_col = (mv_col >> 1) << 1;
      delta_col = mv_col - p_.mi_col;
    }
    if (!mi_inside(mv_row, mv_col, p_)) {
      return;
    }
    if (mi_at(p_, mv_row, mv_col).ref[0] == kRefNone) {
      return;  // not yet coded
    }
    int weight = 1;
    if (delta_row == -1 && delta_col == -1) {
      weight = 0;
    } else if (delta_col < -1) {
      weight = 0;
    }
    add_reference_mv(mv_row, mv_col, weight);
  }

  void scan_point_if_valid(int delta_row, int delta_col) {
    const int mv_row = p_.mi_row + delta_row;
    const int mv_col = p_.mi_col + delta_col;
    if (!mi_inside(mv_row, mv_col, p_)) {
      return;
    }
    if (p_.mi_col - 1 < 0) {
      return;
    }
    if (mi_at(p_, mv_row, mv_col).col_base ==
        mi_at(p_, mv_row, p_.mi_col - 1).col_base) {
      return;
    }
    scan_point(delta_row, delta_col);
  }

  void scan_col(int delta_col) {
    delta_col += p_.mi_col & 1;
    scan_point_if_valid(p_.bh4 - 1, delta_col);
    if (p_.bh4 > 1) {
      scan_point_if_valid(0, delta_col);
    }
  }

  void sort_nearest() {
    const int end = stack_->num_nearest;
    for (int start = 0; start + 1 < end; ++start) {
      int max_weight = stack_->cand[start].weight;
      int max_idx = start;
      for (int idx = start + 1; idx < end; ++idx) {
        if (max_weight < stack_->cand[idx].weight) {
          max_weight = stack_->cand[idx].weight;
          max_idx = idx;
        }
      }
      if (max_idx != start) {
        const StackCandidate tmp = stack_->cand[start];
        stack_->cand[start] = stack_->cand[max_idx];
        stack_->cand[max_idx] = tmp;
      }
    }
  }

  void fill_derived() {
    const int max_ref_mv_count = p_.max_drl_bits_minus_1 + 2;
    if (stack_->count >= max_ref_mv_count) {
      return;
    }
    for (int d = 0; d < derived_count_; ++d) {
      bool found = false;
      if (prune_count_ < kMaxPrNum) {
        for (int i = 0; i < stack_->count; ++i) {
          ++prune_count_;
          const bool match0 =
              same_mv(derived_[d].mv[0], stack_->cand[i].mv[0]);
          const bool match1 =
              !p_.is_compound ||
              same_mv(derived_[d].mv[1], stack_->cand[i].mv[1]);
          if (match0 && match1) {
            found = true;
            break;
          }
        }
      }
      if (!found && stack_->count < max_ref_mv_count) {
        StackCandidate& c = stack_->cand[stack_->count];
        c.mv[0] = derived_[d].mv[0];
        c.mv[1] = p_.is_compound ? derived_[d].mv[1] : derived_[d].mv[0];
        c.weight = 1;
        c.row_offset = 0;
        c.col_offset = 0;
        c.cwp = kCwpEqual;
        ++stack_->count;
      }
    }
  }

  void insert_mvp_candidate(int y_cand, int x_cand) {
    if (y_cand >= stack_->count || x_cand >= stack_->count) {
      return;
    }
    const inter::mc::Mv cand0{stack_->cand[y_cand].mv[0].y,
                              stack_->cand[x_cand].mv[0].x};
    const inter::mc::Mv cand1{stack_->cand[y_cand].mv[1].y,
                              stack_->cand[x_cand].mv[1].x};
    if (stack_->count >= kMaxRefMvStackSize) {
      return;
    }
    if (prune_count_ < kMaxPrNum) {
      for (int i = 0; i < stack_->count; ++i) {
        ++prune_count_;
        const bool match0 = same_mv(cand0, stack_->cand[i].mv[0]);
        if (!p_.is_compound && match0) {
          return;
        }
        if (p_.is_compound && match0 &&
            same_mv(cand1, stack_->cand[i].mv[1])) {
          return;
        }
      }
    }
    StackCandidate& c = stack_->cand[stack_->count];
    c.mv[0] = cand0;
    c.mv[1] = p_.is_compound ? cand1 : cand0;
    c.weight = 1;
    c.row_offset = 0;
    c.col_offset = 0;
    c.cwp = kCwpEqual;
    ++stack_->count;
  }

  inter::mc::Mv picture_clamp(inter::mc::Mv mv) const {
    const std::int32_t low_y =
        -(p_.mi_row + p_.bh4) * inter::mc::kMiSize * 8 - inter::mc::kMvBorder;
    const std::int32_t high_y =
        (p_.mi_rows - p_.mi_row) * inter::mc::kMiSize * 8 +
        inter::mc::kMvBorder;
    const std::int32_t low_x =
        -(p_.mi_col + p_.bw4) * inter::mc::kMiSize * 8 - inter::mc::kMvBorder;
    const std::int32_t high_x =
        (p_.mi_cols - p_.mi_col) * inter::mc::kMiSize * 8 +
        inter::mc::kMvBorder;
    mv.y = clip3(low_y, high_y, mv.y);
    mv.x = clip3(low_x, high_x, mv.x);
    return mv;
  }

  inter::mc::Mv mv_low_clamp(inter::mc::Mv mv) const {
    mv.y = clip3(inter::mc::kMvLow + 1, inter::mc::kMvUpp - 1, mv.y);
    mv.x = clip3(inter::mc::kMvLow + 1, inter::mc::kMvUpp - 1, mv.x);
    return mv;
  }

  void clamp_all() {
    for (int i = 0; i < stack_->count; ++i) {
      for (int list = 0; list < 2; ++list) {
        inter::mc::Mv mv = stack_->cand[i].mv[list];
        if (options_.extra_mv_low_clamp) {
          mv = mv_low_clamp(mv);
        }
        stack_->cand[i].mv[list] = picture_clamp(mv);
      }
    }
  }

  const StackParams& p_;
  MvStack* stack_;
  BuildOptions options_;
  int prune_count_ = 0;
  DerivedCandidate derived_[kMaxDrStackSize]{};
  int derived_count_ = 0;
  int derived_prune_count_ = 0;
  SingleCandidate single_[kMaxDrStackSize]{};
  int single_count_ = 0;
  int single_prune_count_ = 0;
};

Status validate_stack_params(const StackParams& p, bool allow_compound) {
  if (p.mi == nullptr || p.mi_cols <= 0 || p.mi_rows <= 0) {
    return Status::kInvalidArgument;
  }
  if (p.bw4 < kMinBlockMi || p.bh4 < kMinBlockMi || p.bw4 > kMaxBlockMi ||
      p.bh4 > kMaxBlockMi) {
    return Status::kInvalidBlockSize;
  }
  if (p.sb_h4 <= 0 || (p.sb_h4 & (p.sb_h4 - 1)) != 0) {
    return Status::kInvalidArgument;
  }
  if (p.mi_row < 0 || p.mi_col < 0 || p.mi_row >= p.mi_rows ||
      p.mi_col >= p.mi_cols) {
    return Status::kInvalidArgument;
  }
  if (p.use_intrabc) {
    return Status::kUnsupportedFeature;
  }
  if (p.max_drl_bits_minus_1 < 0 || p.max_drl_bits_minus_1 > 4) {
    return Status::kInvalidArgument;
  }
  if (p.is_compound) {
    if (!allow_compound) {
      return Status::kUnsupportedFeature;
    }
    if (!is_inter_ref(p.ref_frame[0]) || !is_inter_ref(p.ref_frame[1]) ||
        p.ref_frame[0] == kTipFrame || p.ref_frame[1] == kTipFrame) {
      return Status::kInvalidArgument;
    }
  } else {
    if (!is_inter_ref(p.ref_frame[0]) || p.ref_frame[0] == kTipFrame) {
      return Status::kInvalidArgument;
    }
    if (p.ref_frame[1] != kRefNone) {
      return Status::kInvalidArgument;
    }
  }
  return Status::kOk;
}

Status build_impl(const StackParams& params, const BuildOptions& options,
                  MvStack* stack) {
  if (stack == nullptr) {
    return Status::kInvalidArgument;
  }
  const Status status = validate_stack_params(params, true);
  if (status != Status::kOk) {
    return status;
  }
  *stack = MvStack{};
  StackBuilder builder(params, stack, options);
  builder.run();
  return Status::kOk;
}

bool same_stack(const MvStack& a, const MvStack& b) {
  if (a.count != b.count) {
    return false;
  }
  for (int i = 0; i < a.count; ++i) {
    if (!(a.cand[i].mv[0] == b.cand[i].mv[0]) ||
        !(a.cand[i].mv[1] == b.cand[i].mv[1]) ||
        a.cand[i].weight != b.cand[i].weight ||
        a.cand[i].row_offset != b.cand[i].row_offset ||
        a.cand[i].col_offset != b.cand[i].col_offset) {
      return false;
    }
  }
  return true;
}

int first_stack_divergence(const MvStack& a, const MvStack& b) {
  const int n = std::min(a.count, b.count);
  for (int i = 0; i < n; ++i) {
    if (!(a.cand[i].mv[0] == b.cand[i].mv[0]) ||
        !(a.cand[i].mv[1] == b.cand[i].mv[1]) ||
        a.cand[i].weight != b.cand[i].weight ||
        a.cand[i].row_offset != b.cand[i].row_offset ||
        a.cand[i].col_offset != b.cand[i].col_offset) {
      return i;
    }
  }
  return n;
}

MvStack stack_from_mc(const inter::mc::MvStack& mc) {
  MvStack out;
  out.count = mc.count;
  out.num_nearest = mc.num_nearest;
  for (int i = 0; i < mc.count && i < kMaxRefMvStackSize; ++i) {
    out.cand[i].mv[0] = mc.cand[i].mv[0];
    out.cand[i].mv[1] = mc.cand[i].mv[1];
    out.cand[i].weight = mc.cand[i].weight;
    out.cand[i].row_offset = mc.cand[i].row_offset;
    out.cand[i].col_offset = mc.cand[i].col_offset;
    out.cand[i].cwp = kCwpEqual;
  }
  return out;
}

inter::mc::DrlReorder to_mc_reorder(DrlReorder reorder) {
  switch (reorder) {
    case DrlReorder::kDisabled:
      return inter::mc::DrlReorder::kNever;
    case DrlReorder::kConstraint:
      return inter::mc::DrlReorder::kConstraint;
    case DrlReorder::kAlways:
      return inter::mc::DrlReorder::kAlways;
  }
  return inter::mc::DrlReorder::kNever;
}

DrlReorder normalize_reorder(DrlReorder reorder) {
  switch (reorder) {
    case DrlReorder::kDisabled:
    case DrlReorder::kConstraint:
    case DrlReorder::kAlways:
      return reorder;
  }
  return DrlReorder::kDisabled;
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
    case Status::kInvalidRefCount:
      return "invalid reference count";
    case Status::kInvalidDrlIndex:
      return "invalid DRL index";
    case Status::kTruncatedDrlBits:
      return "truncated DRL bits";
    case Status::kInvalidSkipMode:
      return "invalid skip_mode";
    case Status::kUnsupportedFeature:
      return "unsupported feature";
  }
  return "unknown";
}

// ---------------------------------------------------------------------------
// Predicates
// ---------------------------------------------------------------------------

bool is_inter_ref(std::int8_t ref) {
  return ref != kIntraFrame && ref != kRefNone;
}

bool is_thin_4xn_nx4_block(BlockSize size) {
  if (!valid_block_size(size)) {
    return false;
  }
  const int w = block_width_px(size);
  const int h = block_height_px(size);
  return (w == 4 && h >= 16) || (h == 4 && w >= 16);
}

bool is_comp_ref_allowed(BlockSize size) {
  if (!valid_block_size(size)) {
    return false;
  }
  const int w = block_width_px(size);
  const int h = block_height_px(size);
  return std::min(w, h) >= 8 || is_thin_4xn_nx4_block(size);
}

bool mvp_combinations_allowed(int width, int height) {
  return width > 32 && height > 32;
}

bool mvp_combinations_allowed(BlockSize size) {
  if (!valid_block_size(size)) {
    return false;
  }
  return mvp_combinations_allowed(block_width_px(size), block_height_px(size));
}

int relative_dist(int a, int b) { return clip3(-127, 127, a - b); }

// ---------------------------------------------------------------------------
// Frame-level skip_mode_params
// ---------------------------------------------------------------------------

Status derive_skip_mode_params(const SkipModeFrameParams& params,
                               SkipModeFrameResult* result) {
  if (result == nullptr) {
    return Status::kInvalidArgument;
  }
  if (params.num_total_refs < 1 || params.num_total_refs > kMaxSupportedRefs) {
    return Status::kInvalidRefCount;
  }
  SkipModeFrameResult out;
  out.allowed = false;
  out.frame[0] = 0;
  out.frame[1] = 0;
  out.num_total_refs = params.num_total_refs;

  if (!params.frame_is_intra && !params.switch_frame) {
    out.allowed = true;
    out.frame[0] = 0;
    out.frame[1] = params.num_total_refs > 1 ? 1 : 0;
    if (params.num_total_refs > 1) {
      if (!params.refs[0].valid || !params.refs[1].valid) {
        return Status::kInvalidArgument;
      }
      int cur_to_ref0 =
          std::abs(relative_dist_restricted(
              params.order_hint, params.order_hint_restricted,
              params.refs[0].order_hint, params.refs[0].restricted));
      int cur_to_ref1 =
          std::abs(relative_dist_restricted(
              params.order_hint, params.order_hint_restricted,
              params.refs[1].order_hint, params.refs[1].restricted));
      if (params.refs[0].restricted) {
        cur_to_ref0 = 0;
      }
      if (params.refs[1].restricted) {
        cur_to_ref1 = 0;
      }
      if (std::abs(cur_to_ref0 - cur_to_ref1) > 1) {
        out.frame[1] = 0;
      }
    }
  }
  *result = out;
  return Status::kOk;
}

Status validate_skip_mode_present(const SkipModeFrameResult& result,
                                  bool skip_mode_present) {
  if (skip_mode_present && !result.allowed) {
    return Status::kInvalidSkipMode;
  }
  return Status::kOk;
}

Status derive_skip_mode_frames(const SkipModeFrameResult& result,
                               const SkipModeFrameInputs& inputs,
                               std::int8_t out[2]) {
  if (out == nullptr) {
    return Status::kInvalidArgument;
  }
  if (!result.allowed) {
    return Status::kInvalidSkipMode;
  }
  if (inputs.neighbours.count < 0 || inputs.neighbours.count > 2) {
    return Status::kInvalidArgument;
  }
  for (int n = 0; n < inputs.neighbours.count; ++n) {
    const std::int8_t ref0 = inputs.neighbours.ref[n][0];
    const std::int8_t ref1 = inputs.neighbours.ref[n][1];
    if (ref0 == kTipFrame) {
      if (inputs.closest_past == kRefNone ||
          inputs.closest_future == kRefNone) {
        return Status::kInvalidArgument;
      }
      out[0] = std::min(inputs.closest_past, inputs.closest_future);
      out[1] = std::max(inputs.closest_past, inputs.closest_future);
      return Status::kOk;
    }
    if (is_inter_ref(ref0) && is_inter_ref(ref1)) {
      if (ref0 >= result.num_total_refs || ref1 >= result.num_total_refs) {
        return Status::kInvalidRefCount;
      }
      out[0] = ref0;
      out[1] = ref1;
      return Status::kOk;
    }
    if (is_inter_ref(ref0)) {
      break;
    }
  }
  out[0] = result.frame[0];
  out[1] = result.frame[1];
  return Status::kOk;
}

Status skip_mode_syntax_allowed(const BlockSkipModeInputs& inputs,
                                bool* allowed) {
  if (allowed == nullptr) {
    return Status::kInvalidArgument;
  }
  if (!valid_block_size(inputs.size)) {
    return Status::kInvalidBlockSize;
  }
  *allowed = inputs.skip_mode_present && !inputs.seg_lvl_skip &&
             !inputs.seg_lvl_globalmv && !inputs.region_intra &&
             is_comp_ref_allowed(inputs.size);
  return Status::kOk;
}

// ---------------------------------------------------------------------------
// DRL
// ---------------------------------------------------------------------------

std::vector<std::uint8_t> encode_drl_index(int index, int start, int m) {
  std::vector<std::uint8_t> bits;
  if (m < 1 || m > kMaxDrlIndex || start < 0 || start > m + 1 ||
      index < 0 || index > m) {
    return bits;
  }
  if (start <= m && index < start) {
    return bits;
  }
  if (start == m + 1 && index != m) {
    return bits;
  }
  if (index < m) {
    for (int idx = start; idx < index; ++idx) {
      bits.push_back(1);
    }
    bits.push_back(0);
  } else {
    for (int idx = start; idx < m; ++idx) {
      bits.push_back(1);
    }
  }
  return bits;
}

Status decode_drl_index(std::span<const std::uint8_t> bits, int start, int m,
                        int* index, std::size_t* consumed) {
  if (index == nullptr || consumed == nullptr || m < 1 ||
      m > kMaxDrlIndex || start < 0 || start > m + 1) {
    return Status::kInvalidArgument;
  }
  std::size_t pos = 0;
  for (int idx = start; idx < m; ++idx) {
    if (pos >= bits.size()) {
      return Status::kTruncatedDrlBits;
    }
    const std::uint8_t bit = bits[pos++];
    if (bit == 0) {
      *index = idx;
      *consumed = pos;
      return Status::kOk;
    }
  }
  *index = m;
  *consumed = pos;
  return Status::kOk;
}

int drl_cdf_row(int idx) {
  if (idx < 0) {
    return 0;
  }
  return std::min(idx, 2);
}

bool has_second_drl(int y_mode, bool skip_mode, bool use_optflow) {
  return (y_mode == kYModeNearNearMv || y_mode == kYModeNearNewMv) &&
         !skip_mode && !use_optflow;
}

Status derive_second_drl_start(int y_mode, bool same_refs, int refmv_idx0,
                               int* start) {
  if (start == nullptr || refmv_idx0 < 0 ||
      refmv_idx0 > kMaxDrlIndex) {
    return Status::kInvalidDrlIndex;
  }
  *start = (same_refs && y_mode == kYModeNearNearMv) ? refmv_idx0 + 1 : 0;
  return Status::kOk;
}

// ---------------------------------------------------------------------------
// single_mode
// ---------------------------------------------------------------------------

Status decode_single_mode(int single_mode, SingleMode* out) {
  if (out == nullptr) {
    return Status::kInvalidArgument;
  }
  switch (single_mode) {
    case 0:
      *out = SingleMode::kNearMv;
      return Status::kOk;
    case 1:
      *out = SingleMode::kGlobalMv;
      return Status::kOk;
    case 2:
      *out = SingleMode::kNewMv;
      return Status::kOk;
    default:
      return Status::kInvalidArgument;
  }
}

int single_mode_y_mode(int single_mode) { return kYModeNearMv + single_mode; }

bool single_mode_has_drl(SingleMode mode) {
  return mode == SingleMode::kNearMv || mode == SingleMode::kNewMv;
}

bool single_mode_has_mvd(SingleMode mode) { return mode == SingleMode::kNewMv; }

// ---------------------------------------------------------------------------
// Mode contexts
// ---------------------------------------------------------------------------

int skip_mode_context(const NeighborModes& neighbours) {
  int ctx = 0;
  if (neighbours.above.available && neighbours.above.skip_mode) {
    ++ctx;
  }
  if (neighbours.left.available && neighbours.left.skip_mode) {
    ++ctx;
  }
  return ctx;
}

int skip_flag_context(const NeighborModes& neighbours, bool skip_mode) {
  int ctx = 0;
  if (neighbours.above.available && neighbours.above.skip_flag) {
    ++ctx;
  }
  if (neighbours.left.available && neighbours.left.skip_flag) {
    ++ctx;
  }
  if (skip_mode) {
    ctx += 3;  // SKIP_CONTEXTS >> 1
  }
  return ctx;
}

Status derive_new_mv_context(const NewMvContextParams& params, int* ctx) {
  if (ctx == nullptr || params.mi == nullptr || params.mi_cols <= 0 ||
      params.mi_rows <= 0 || params.bw4 <= 0 || params.bh4 <= 0 ||
      params.sb_h4 <= 0) {
    return Status::kInvalidArgument;
  }
  int new_mv_count = 0;
  const bool sb_border = (params.mi_row & (params.sb_h4 - 1)) == 0;

  const auto inside = [&](int row, int col) {
    return row >= 0 && col >= 0 && row < params.mi_rows &&
           col < params.mi_cols;
  };
  const auto scan = [&](int delta_row, int delta_col) -> int {
    const int mv_row = params.mi_row + delta_row;
    const int mv_col = params.mi_col + delta_col;
    if (!inside(mv_row, mv_col)) {
      return 0;
    }
    const std::size_t idx =
        static_cast<std::size_t>(mv_row) *
            static_cast<std::size_t>(params.mi_cols) +
        static_cast<std::size_t>(mv_col);
    const inter::mc::MiInfo& mi = params.mi[idx];
    if (!mi.is_inter) {
      return 0;
    }
    const int cand_mode =
        params.y_mode == nullptr ? 0 : static_cast<int>(params.y_mode[idx]);
    int found = 0;
    if (!params.is_compound) {
      for (int list = 0; list < 2; ++list) {
        if (mi.ref[list] == params.ref_frame[0]) {
          if (has_newmv_for_list(cand_mode, list)) {
            new_mv_count = std::min(3, new_mv_count + 1);
          }
          found = 1;
          break;
        }
      }
    } else {
      if (mi.ref[0] == params.ref_frame[0] &&
          mi.ref[1] == params.ref_frame[1]) {
        found = 1;
      }
      if (found > 0 && has_newmv_mode(cand_mode)) {
        new_mv_count = std::min(3, new_mv_count + found);
      }
    }
    return found;
  };

  const int left_a = scan(params.bh4 - 1, -1);
  const int above_a = scan(-1, params.bw4 - 1);
  const int left_b = scan(0, -1);
  const int above_b =
      (params.bw4 >= (sb_border ? 4 : 2)) ? scan(-1, 0) : 0;
  const int nearest = ((above_a != 0 || above_b != 0) ? 1 : 0) +
                      ((left_a != 0 || left_b != 0) ? 1 : 0);
  *ctx = nearest + (new_mv_count > 0 ? 2 : 0);
  return Status::kOk;
}

// ---------------------------------------------------------------------------
// Stack API
// ---------------------------------------------------------------------------

Status build_mv_stack(const StackParams& params, MvStack* stack) {
  BuildOptions options;
  return build_impl(params, options, stack);
}

Status audit_mc_stack(const StackParams& params, McStackAudit* audit) {
  if (audit == nullptr) {
    return Status::kInvalidArgument;
  }
  *audit = McStackAudit{};
  if (params.is_compound) {
    audit->mc_supported = false;
    return Status::kOk;
  }
  const Status status = validate_stack_params(params, false);
  if (status != Status::kOk) {
    return status;
  }

  MvStack spec;
  BuildOptions spec_options;
  Status s = build_impl(params, spec_options, &spec);
  if (s != Status::kOk) {
    return s;
  }
  audit->mc_supported = true;
  audit->expected_count = spec.count;

  inter::mc::MvStackParams mp;
  mp.mi_row = params.mi_row;
  mp.mi_col = params.mi_col;
  mp.bw4 = params.bw4;
  mp.bh4 = params.bh4;
  mp.sb_h4 = params.sb_h4;
  mp.mi_cols = params.mi_cols;
  mp.mi_rows = params.mi_rows;
  mp.mi = params.mi;
  mp.ref_frame = params.ref_frame[0];
  mp.global_mv = params.global_mv[0];
  mp.use_intrabc = false;
  mp.reorder = to_mc_reorder(normalize_reorder(params.reorder));

  inter::mc::MvStack mc_stack;
  const inter::mc::McStatus mc_status =
      inter::mc::build_mv_stack(mp, &mc_stack);
  if (mc_status != inter::mc::McStatus::kOk) {
    audit->mc_call_ok = false;
    return Status::kOk;
  }
  audit->mc_call_ok = true;
  audit->mc_count = mc_stack.count;
  const MvStack mc = stack_from_mc(mc_stack);

  if (same_stack(mc, spec)) {
    audit->matches_spec = true;
    return Status::kOk;
  }
  audit->first_divergence = first_stack_divergence(mc, spec);
  audit->matches_spec = false;

  // Instrumented variants: the shared component's observed deviations are
  // (1) MVP combinations are not gated, (2) step 13 scans (-1,-1) instead of
  // (-1,-1-isSbBorder), (3) derived-SMVP candidates are not implemented, and
  // (4) MV_LOW/MV_UPP clamping is applied before the picture clamp.
  BuildOptions mc_style;
  mc_style.gate_mvp_combinations = false;
  mc_style.mc_above_left_scan = true;
  mc_style.include_derived = false;
  mc_style.extra_mv_low_clamp = true;
  MvStack mc_expected;
  s = build_impl(params, mc_style, &mc_expected);
  if (s != Status::kOk) {
    return s;
  }
  if (same_stack(mc, mc_expected)) {
    // Isolate each observed deviation so the audit report does not
    // cross-attribute one difference to another.
    BuildOptions variant = mc_style;
    variant.gate_mvp_combinations = true;
    MvStack gated_mc;
    s = build_impl(params, variant, &gated_mc);
    if (s != Status::kOk) {
      return s;
    }
    audit->mvp_combinations_not_gated = !same_stack(mc_expected, gated_mc);

    variant = mc_style;
    variant.mc_above_left_scan = false;
    MvStack spec_scan;
    s = build_impl(params, variant, &spec_scan);
    if (s != Status::kOk) {
      return s;
    }
    audit->above_left_sb_border_scan_diff =
        !same_stack(mc_expected, spec_scan);

    variant = mc_style;
    variant.include_derived = true;
    MvStack with_derived;
    s = build_impl(params, variant, &with_derived);
    if (s != Status::kOk) {
      return s;
    }
    audit->derived_candidates_missing = !same_stack(mc_expected, with_derived);

    variant = mc_style;
    variant.extra_mv_low_clamp = false;
    MvStack no_low_clamp;
    s = build_impl(params, variant, &no_low_clamp);
    if (s != Status::kOk) {
      return s;
    }
    audit->extra_mv_low_clamp = !same_stack(mc_expected, no_low_clamp);
  }
  return Status::kOk;
}

Status build_stack_checked(const StackParams& params, MvStack* stack,
                           McStackAudit* audit) {
  const Status status = build_mv_stack(params, stack);
  if (status != Status::kOk) {
    return status;
  }
  if (audit != nullptr) {
    const Status audit_status = audit_mc_stack(params, audit);
    if (audit_status != Status::kOk) {
      return audit_status;
    }
  }
  return Status::kOk;
}

// ---------------------------------------------------------------------------
// Resolution
// ---------------------------------------------------------------------------

std::int32_t clamp_mv_to_integer(std::int32_t v) {
  if (v < inter::mc::kMvLow + 1) {
    return inter::mc::kMvLow + 8;
  }
  if (v > inter::mc::kMvUpp - 1) {
    return inter::mc::kMvUpp - 8;
  }
  return v;
}

inter::mc::Mv clamp_mv_to_integer(inter::mc::Mv mv) {
  mv.y = clamp_mv_to_integer(mv.y);
  mv.x = clamp_mv_to_integer(mv.x);
  return mv;
}

Status validate_mode_syntax(const ModeSyntax& syntax, const StackParams& params,
                            const MvStack& stack, std::string* why) {
  const auto fail = [&](Status status, const char* message) {
    if (why != nullptr) {
      *why = message;
    }
    return status;
  };
  if (!syntax.is_inter) {
    return fail(Status::kInvalidArgument, "mode is not inter");
  }
  const int m = params.max_drl_bits_minus_1 + 1;
  const auto drl_ok = [&]() {
    return syntax.drl_index >= 0 && syntax.drl_index <= m &&
           syntax.drl_index < stack.count &&
           syntax.drl_index <= kMaxDrlIndex;
  };

  if (syntax.skip_mode) {
    if (syntax.y_mode != kYModeNearNearMv) {
      return fail(Status::kInvalidSkipMode, "skip_mode must imply NEAR_NEARMV");
    }
    if (!params.is_compound || !is_inter_ref(params.ref_frame[1])) {
      return fail(Status::kInvalidSkipMode,
                  "skip_mode requires two references");
    }
    if (syntax.ref_frame[0] != params.ref_frame[0] ||
        syntax.ref_frame[1] != params.ref_frame[1]) {
      return fail(Status::kInvalidSkipMode,
                  "skip_mode reference pair mismatch");
    }
    if (!is_inter_ref(syntax.ref_frame[0]) ||
        !is_inter_ref(syntax.ref_frame[1])) {
      return fail(Status::kInvalidSkipMode,
                  "skip_mode reference is not inter");
    }
    if (syntax.mvd.y != 0 || syntax.mvd.x != 0) {
      return fail(Status::kInvalidSkipMode, "skip_mode has an MVD");
    }
    if (!drl_ok()) {
      return fail(Status::kInvalidDrlIndex,
                  "skip_mode DRL index outside the stack");
    }
    return Status::kOk;
  }

  if (syntax.ref_frame[1] != kRefNone) {
    // Compound prediction without skip_mode needs two independent DRL
    // indices and per-list MVD modes (§5.20.7.6 has_second_drl).  That is
    // outside the M2 skip/merge scope; do not silently resolve it as if both
    // lists shared RefMvIdx.
    return fail(Status::kUnsupportedFeature,
                "compound non-skip mode is not implemented");
  }

  if (params.is_compound) {
    return fail(Status::kInvalidArgument,
                "single-reference syntax on a compound stack");
  }
  if (syntax.ref_frame[0] != params.ref_frame[0]) {
    return fail(Status::kInvalidArgument, "reference frame mismatch");
  }
  if (syntax.y_mode != single_mode_y_mode(
                            static_cast<int>(syntax.single_mode))) {
    return fail(Status::kInvalidArgument,
                "YMode does not match single_mode");
  }
  switch (syntax.single_mode) {
    case SingleMode::kGlobalMv:
      if (syntax.mvd.y != 0 || syntax.mvd.x != 0) {
        return fail(Status::kInvalidArgument, "GLOBALMV has an MVD");
      }
      return Status::kOk;
    case SingleMode::kNearMv:
      if (syntax.mvd.y != 0 || syntax.mvd.x != 0) {
        return fail(Status::kInvalidArgument, "NEARMV has an MVD");
      }
      if (!drl_ok()) {
        return fail(Status::kInvalidDrlIndex,
                    "NEARMV DRL index outside the stack");
      }
      return Status::kOk;
    case SingleMode::kNewMv:
      if (!drl_ok()) {
        return fail(Status::kInvalidDrlIndex,
                    "NEWMV DRL index outside the stack");
      }
      return Status::kOk;
  }
  return fail(Status::kInvalidArgument, "bad single_mode");
}

Status resolve_mode(const MvStack& stack, const StackParams& params,
                    const ModeSyntax& syntax, ResolvedMode* out) {
  if (out == nullptr) {
    return Status::kInvalidArgument;
  }
  const Status status = validate_mode_syntax(syntax, params, stack, nullptr);
  if (status != Status::kOk) {
    return status;
  }
  ResolvedMode result;
  result.skip_mode = syntax.skip_mode;
  result.skip_flag = syntax.skip_flag;
  result.y_mode = syntax.y_mode;
  result.drl_index = syntax.drl_index;
  result.ref_frame[0] = syntax.ref_frame[0];
  result.ref_frame[1] = syntax.ref_frame[1];
  result.cwp = kCwpEqual;

  if (syntax.skip_mode || syntax.ref_frame[1] != kRefNone) {
    const StackCandidate& c =
        stack.cand[static_cast<std::size_t>(syntax.drl_index)];
    result.compound = true;
    result.pred_mv[0] = c.mv[0];
    result.pred_mv[1] = c.mv[1];
    result.mv[0] = clamp_mv_to_integer(c.mv[0]);
    result.mv[1] = clamp_mv_to_integer(c.mv[1]);
    result.mvd[0] = inter::mc::Mv{};
    result.mvd[1] = inter::mc::Mv{};
    result.cwp = c.cwp;
    *out = result;
    return Status::kOk;
  }

  switch (syntax.single_mode) {
    case SingleMode::kGlobalMv:
      result.pred_mv[0] = params.global_mv[0];
      result.mv[0] = clamp_mv_to_integer(params.global_mv[0]);
      break;
    case SingleMode::kNearMv:
      result.pred_mv[0] =
          stack.cand[static_cast<std::size_t>(syntax.drl_index)].mv[0];
      result.mv[0] = clamp_mv_to_integer(result.pred_mv[0]);
      break;
    case SingleMode::kNewMv:
      result.pred_mv[0] =
          stack.cand[static_cast<std::size_t>(syntax.drl_index)].mv[0];
      result.mvd[0] = syntax.mvd;
      result.mv[0].y = clamp_mv_to_integer(
          static_cast<std::int32_t>(static_cast<std::int64_t>(
                                        result.pred_mv[0].y) +
                                    static_cast<std::int64_t>(syntax.mvd.y)));
      result.mv[0].x = clamp_mv_to_integer(
          static_cast<std::int32_t>(static_cast<std::int64_t>(
                                        result.pred_mv[0].x) +
                                    static_cast<std::int64_t>(syntax.mvd.x)));
      break;
  }
  result.pred_mv[1] = inter::mc::Mv{};
  result.mvd[1] = inter::mc::Mv{};
  result.mv[1] = result.mv[0];
  *out = result;
  return Status::kOk;
}

Status resolve_single_mode(const MvStack& stack, const StackParams& params,
                           SingleMode mode, int drl_index,
                           inter::mc::Mv mvd, ResolvedMode* out) {
  ModeSyntax syntax;
  syntax.is_inter = true;
  syntax.skip_mode = false;
  syntax.single_mode = mode;
  syntax.y_mode = single_mode_y_mode(static_cast<int>(mode));
  syntax.drl_index = drl_index;
  syntax.mvd = mvd;
  syntax.ref_frame[0] = params.ref_frame[0];
  syntax.ref_frame[1] = kRefNone;
  return resolve_mode(stack, params, syntax, out);
}

Status resolve_skip_mode(const MvStack& stack, const StackParams& params,
                         const SkipModeFrameResult& skip_params,
                         const SkipModeFrameInputs& skip_inputs, int drl_index,
                         bool skip_flag, ResolvedMode* out) {
  if (out == nullptr || !params.is_compound) {
    return Status::kInvalidSkipMode;
  }
  std::int8_t frames[2] = {kRefNone, kRefNone};
  const Status status =
      derive_skip_mode_frames(skip_params, skip_inputs, frames);
  if (status != Status::kOk) {
    return status;
  }
  if (frames[0] != params.ref_frame[0] || frames[1] != params.ref_frame[1]) {
    return Status::kInvalidSkipMode;
  }
  ModeSyntax syntax;
  syntax.is_inter = true;
  syntax.skip_flag = skip_flag;
  syntax.skip_mode = true;
  syntax.y_mode = kYModeNearNearMv;
  syntax.drl_index = drl_index;
  syntax.ref_frame[0] = frames[0];
  syntax.ref_frame[1] = frames[1];
  return resolve_mode(stack, params, syntax, out);
}

Status choose_drl_index_for_mv(const MvStack& stack, inter::mc::Mv mv,
                               int start, int m, int* index, bool* found) {
  if (index == nullptr || found == nullptr || start < 0 ||
      start > kMaxDrlIndex || m < 0 || m > kMaxDrlIndex) {
    return Status::kInvalidArgument;
  }
  *found = false;
  *index = start;
  const int end = std::min(stack.count - 1, m);
  for (int i = start; i <= end; ++i) {
    if (stack.cand[static_cast<std::size_t>(i)].mv[0] == mv) {
      *index = i;
      *found = true;
      break;
    }
  }
  return Status::kOk;
}

}  // namespace av2enc::video::merge

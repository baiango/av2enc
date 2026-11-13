#ifndef AV2ENC_CUDA_INTER_MV_STACK_MV_STACK_DEVICE_CUH_
#define AV2ENC_CUDA_INTER_MV_STACK_MV_STACK_DEVICE_CUH_

// Device transcription of the AV2 v1.0.0 §7.12.2 spatial MV stack, the
// multi-reference pair builder and the compound merge-candidate set.
//
// SHARED WITH EMULATION.  This header is compiled twice:
//   * by nvcc for the kernels in `mv_stack.cu`;
//   * by the host compiler for the emulation parity test.
// The CPU reference in `src/codec/inter/merge/merge.cpp` and
// `src/codec/inter/multiref/mvstack_multi.cpp` is authoritative; the emulation
// test compares every stack field, every pair-stack field and every merge
// candidate against it.
//
// Why a transcription: `merge.cpp` also owns `std::vector`/`std::span`/
// `std::string` syntax helpers, so it cannot be compiled into a kernel.  Only
// the POD stack core is restated here; all §4.8/§7.9 arithmetic is copied from
// the reference statement by statement.
//
// Every function is `AV2ENC_MVS_HD` and every table has device storage, so a
// kernel can call the same code the host test runs.

#include <cstdint>

#include "codec/inter/mc/mc_types.h"
#include "codec/inter/merge/merge.h"

#if defined(__CUDACC__)
#define AV2ENC_MVS_HD __host__ __device__ __forceinline__
#else
#define AV2ENC_MVS_HD inline
#endif

#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
#define AV2ENC_MVS_DEV_TABLE __device__ constexpr
#else
#define AV2ENC_MVS_DEV_TABLE inline constexpr
#endif

namespace av2enc::cuda::mvstack {

namespace vm = av2enc::video::merge;
namespace imc = av2enc::inter::mc;

inline constexpr int kMvsCwpEqual = 8;
// One compound stack (<= 6) plus every cross-list pair (<= 6*5), minus
// duplicates.  40 is a safe fixed cap for the result set.
inline constexpr int kMaxMergeCandidates = 40;

// §7.9.4 Div_Mult[32].  Runtime-indexed, so it needs device storage.
AV2ENC_MVS_DEV_TABLE std::int32_t kMvsDivMult[32] = {
    0,     16384, 8192, 5461, 4096, 3276, 2730, 2340, 2048, 1820, 1638,
    1489,  1365,  1260, 1170, 1092, 1024, 963,  910,  862,  819,  780,
    744,   712,   682,  655,  630,  606,  585,  564,  546,  528};

AV2ENC_MVS_HD std::int32_t mvs_clip3(std::int32_t low, std::int32_t high,
                                            std::int32_t value) {
  return value < low ? low : (value > high ? high : value);
}

AV2ENC_MVS_HD std::int64_t mvs_round2_signed64(std::int64_t x, int n) {
  if (n == 0) {
    return x;
  }
  return x >= 0 ? ((x + (static_cast<std::int64_t>(1) << (n - 1))) >> n)
                : -(((-x) + (static_cast<std::int64_t>(1) << (n - 1))) >> n);
}

AV2ENC_MVS_HD bool mvs_mv_eq(imc::Mv a, imc::Mv b) {
  return a.y == b.y && a.x == b.x;
}

// §7.9.4 get_mv_projection.
AV2ENC_MVS_HD imc::Mv mvs_project_mv(imc::Mv mv, int numerator,
                                            int denominator) {
  const int clipped_denominator = denominator < 31 ? denominator : 31;
  const int clipped_numerator = mvs_clip3(-31, 31, numerator);
  imc::Mv out;
  out.y = mvs_clip3(
      imc::kMvLow + 1, imc::kMvUpp - 1,
      static_cast<std::int32_t>(mvs_round2_signed64(
          static_cast<std::int64_t>(mv.y) * clipped_numerator *
              kMvsDivMult[clipped_denominator],
          14)));
  out.x = mvs_clip3(
      imc::kMvLow + 1, imc::kMvUpp - 1,
      static_cast<std::int32_t>(mvs_round2_signed64(
          static_cast<std::int64_t>(mv.x) * clipped_numerator *
              kMvsDivMult[clipped_denominator],
          14)));
  return out;
}

AV2ENC_MVS_HD bool mvs_is_inter_ref(std::int8_t ref) {
  return ref != vm::kIntraFrame && ref != vm::kRefNone;
}

AV2ENC_MVS_HD bool mvs_has_newmv_mode(int y_mode) {
  return y_mode == vm::kYModeNewMv || y_mode == vm::kYModeNewNewMv ||
         y_mode == vm::kYModeNearNewMv || y_mode == vm::kYModeNewNearMv ||
         y_mode == vm::kYModeWarpNewMv || y_mode == vm::kYModeJointNewMv;
}

AV2ENC_MVS_HD bool mvs_has_newmv_for_list(int y_mode, int ref_list) {
  if (y_mode == vm::kYModeNewNewMv || y_mode == vm::kYModeNewMv) {
    return true;
  }
  if (ref_list == 0) {
    return y_mode == vm::kYModeNewNearMv || y_mode == vm::kYModeJointNewMv;
  }
  return y_mode == vm::kYModeNearNewMv;
}

AV2ENC_MVS_HD bool mvs_mi_inside(int row, int col,
                                        const vm::StackParams& p) {
  return row >= 0 && col >= 0 && row < p.mi_rows && col < p.mi_cols;
}

AV2ENC_MVS_HD const imc::MiInfo& mvs_mi_at(const vm::StackParams& p,
                                                  int row, int col) {
  return p.mi[static_cast<std::size_t>(row) *
                  static_cast<std::size_t>(p.mi_cols) +
              static_cast<std::size_t>(col)];
}

AV2ENC_MVS_HD imc::Mv mvs_get_mv(const imc::MiInfo& mi, int list) {
  return imc::Mv{mi.mv[list][0], mi.mv[list][1]};
}

// Mirrors merge.cpp validate_stack_params(allow_compound = true).
AV2ENC_MVS_HD vm::Status mvs_validate(const vm::StackParams& p) {
  if (p.mi == nullptr || p.mi_cols <= 0 || p.mi_rows <= 0) {
    return vm::Status::kInvalidArgument;
  }
  if (p.bw4 < vm::kMinBlockMi || p.bh4 < vm::kMinBlockMi ||
      p.bw4 > vm::kMaxBlockMi || p.bh4 > vm::kMaxBlockMi) {
    return vm::Status::kInvalidBlockSize;
  }
  if (p.sb_h4 <= 0 || (p.sb_h4 & (p.sb_h4 - 1)) != 0) {
    return vm::Status::kInvalidArgument;
  }
  if (p.mi_row < 0 || p.mi_col < 0 || p.mi_row >= p.mi_rows ||
      p.mi_col >= p.mi_cols) {
    return vm::Status::kInvalidArgument;
  }
  if (p.use_intrabc) {
    return vm::Status::kUnsupportedFeature;
  }
  if (p.max_drl_bits_minus_1 < 0 || p.max_drl_bits_minus_1 > 4) {
    return vm::Status::kInvalidArgument;
  }
  if (p.is_compound) {
    if (!mvs_is_inter_ref(p.ref_frame[0]) ||
        !mvs_is_inter_ref(p.ref_frame[1]) ||
        p.ref_frame[0] == vm::kTipFrame || p.ref_frame[1] == vm::kTipFrame) {
      return vm::Status::kInvalidArgument;
    }
  } else {
    if (!mvs_is_inter_ref(p.ref_frame[0]) ||
        p.ref_frame[0] == vm::kTipFrame) {
      return vm::Status::kInvalidArgument;
    }
    if (p.ref_frame[1] != vm::kRefNone) {
      return vm::Status::kInvalidArgument;
    }
  }
  return vm::Status::kOk;
}

// ---------------------------------------------------------------------------
// §7.12.2 stack builder (spatial subset, temporal tools off)
// ---------------------------------------------------------------------------

struct MvsDerivedCandidate {
  imc::Mv mv[2];
};

struct MvsSingleCandidate {
  std::int8_t ref = vm::kRefNone;
  imc::Mv mv;
};

class MvsStackBuilder {
 public:
  AV2ENC_MVS_HD MvsStackBuilder(const vm::StackParams& params,
                                vm::MvStack* stack)
      : p_(params), stack_(stack) {}

  AV2ENC_MVS_HD void run() {
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
    scan_point(-1, -1 - (sb_border ? 1 : 0));

    // Step 14.
    stack_->num_nearest = stack_->count;

    // Step 15.
    scan_col(-3);

    // Steps 16-17.
    const bool use_sort =
        p_.reorder == vm::DrlReorder::kAlways ||
        (p_.reorder == vm::DrlReorder::kConstraint && !p_.use_temporal_first &&
         stack_->num_nearest >= 4);
    stack_->sorted = use_sort;
    if (use_sort) {
      sort_nearest();
    }

    // Steps 18 and 20.
    fill_derived();

    // Step 21: pre-clamp.
    clamp_all();

    // Step 21: global-MV candidate.
    if (stack_->count < vm::kMaxRefMvStackSize) {
      bool found = false;
      if (prune_count_ < vm::kMaxPrNum) {
        for (int i = 0; i < stack_->count; ++i) {
          ++prune_count_;
          const bool match0 = mvs_mv_eq(p_.global_mv[0], stack_->cand[i].mv[0]);
          const bool match1 =
              !p_.is_compound ||
              mvs_mv_eq(p_.global_mv[1], stack_->cand[i].mv[1]);
          if (match0 && match1) {
            found = true;
            break;
          }
        }
      }
      if (!found) {
        vm::StackCandidate& c = stack_->cand[stack_->count];
        c.mv[0] = p_.global_mv[0];
        c.mv[1] = p_.is_compound ? p_.global_mv[1] : p_.global_mv[0];
        c.weight = 1;
        c.row_offset = 0;
        c.col_offset = 0;
        c.cwp = kMvsCwpEqual;
        ++stack_->count;
      }
    }

    // Step 21: MVP combinations, gated by Block_Width/Height > 32.
    const bool gate = p_.bw4 * 4 > 32 && p_.bh4 * 4 > 32;
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
  AV2ENC_MVS_HD void search_stack(imc::Mv cand0, imc::Mv cand1, int mv_row,
                                  int mv_col, int weight) {
    int found = -1;
    if (prune_count_ < vm::kMaxPrNum) {
      for (int i = 0; i < stack_->count; ++i) {
        ++prune_count_;
        const bool match0 = mvs_mv_eq(cand0, stack_->cand[i].mv[0]);
        const bool match1 =
            !p_.is_compound || mvs_mv_eq(cand1, stack_->cand[i].mv[1]);
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
    if (stack_->count < vm::kMaxRefMvStackSize) {
      vm::StackCandidate& c = stack_->cand[stack_->count];
      c.mv[0] = cand0;
      c.mv[1] = p_.is_compound ? cand1 : cand0;
      c.weight = weight;
      c.row_offset = mv_row - p_.mi_row;
      c.col_offset = mv_col - p_.mi_col;
      c.cwp = kMvsCwpEqual;
      ++stack_->count;
    }
  }

  AV2ENC_MVS_HD void add_derived(imc::Mv mv0, imc::Mv mv1) {
    if (derived_count_ >= vm::kMaxDrStackSize) {
      return;
    }
    derived_[derived_count_].mv[0] = mv0;
    derived_[derived_count_].mv[1] = mv1;
    ++derived_count_;
  }

  AV2ENC_MVS_HD bool comp_mv_in_derived(imc::Mv mv0, imc::Mv mv1) {
    if (derived_prune_count_ < vm::kMaxDrPrNum) {
      for (int i = 0; i < derived_count_; ++i) {
        ++derived_prune_count_;
        if (mvs_mv_eq(derived_[i].mv[0], mv0) &&
            mvs_mv_eq(derived_[i].mv[1], mv1)) {
          return true;
        }
      }
    }
    return false;
  }

  // §7.12.2.16 Single add derived process (non-temporal subset).
  AV2ENC_MVS_HD void single_add_derived(int mv_row, int mv_col, int cand_list) {
    const imc::MiInfo& m = mvs_mi_at(p_, mv_row, mv_col);
    const std::int8_t cand_ref = m.ref[cand_list];
    if (!mvs_is_inter_ref(cand_ref) || cand_ref == vm::kTipFrame) {
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
    const imc::Mv cand_mv = mvs_get_mv(m, cand_list);
    const imc::Mv projected = mvs_project_mv(
        cand_mv, cur_dist < 0 ? -cur_dist : cur_dist,
        cand_dist < 0 ? -cand_dist : cand_dist);
    if (!comp_mv_in_derived(projected, projected)) {
      add_derived(projected, projected);
    }
  }

  // §7.12.2.14 Compound add derived process (non-temporal subset).
  AV2ENC_MVS_HD void compound_add_derived(int mv_row, int mv_col) {
    const imc::MiInfo& m = mvs_mi_at(p_, mv_row, mv_col);
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
    const imc::Mv cand_mv = mvs_get_mv(m, cand_list);

    for (int i = 0; i < single_count_ && derived_count_ < vm::kMaxDrStackSize;
         ++i) {
      if (single_[i].ref != p_.ref_frame[cand_ref_idx1]) {
        continue;
      }
      const imc::Mv l0 = cand_ref_idx0 == 0 ? cand_mv : single_[i].mv;
      const imc::Mv l1 = cand_ref_idx0 == 1 ? cand_mv : single_[i].mv;
      if (!comp_mv_in_derived(l0, l1)) {
        add_derived(l0, l1);
      }
      break;
    }

    if (single_prune_count_ < vm::kMaxDrPrNum) {
      for (int i = 0; i < single_count_; ++i) {
        ++single_prune_count_;
        if (single_[i].ref == p_.ref_frame[cand_ref_idx0] &&
            mvs_mv_eq(single_[i].mv, cand_mv)) {
          return;
        }
      }
    }
    if (single_count_ < vm::kMaxDrStackSize) {
      single_[single_count_].ref = p_.ref_frame[cand_ref_idx0];
      single_[single_count_].mv = cand_mv;
      ++single_count_;
    }
  }

  AV2ENC_MVS_HD void compound_search_stack(int mv_row, int mv_col, int weight) {
    const imc::MiInfo& m = mvs_mi_at(p_, mv_row, mv_col);
    search_stack(mvs_get_mv(m, 0), mvs_get_mv(m, 1), mv_row, mv_col, weight);
  }

  AV2ENC_MVS_HD void add_reference_mv(int mv_row, int mv_col, int weight) {
    const imc::MiInfo& m = mvs_mi_at(p_, mv_row, mv_col);
    if (!m.is_inter) {
      return;
    }
    if (!p_.is_compound) {
      for (int list = 0; list < 2; ++list) {
        if (m.ref[list] == p_.ref_frame[0]) {
          const imc::Mv mv = mvs_get_mv(m, list);
          search_stack(mv, mv, mv_row, mv_col, weight);
        } else if (mvs_is_inter_ref(m.ref[list])) {
          single_add_derived(mv_row, mv_col, list);
        }
      }
      return;
    }

    if (m.ref[0] == vm::kTipFrame || m.ref[1] == vm::kTipFrame) {
      return;  // TIP has its own stack (src/cuda/codec/inter/tip/).
    }
    if (m.ref[0] == p_.ref_frame[0] && m.ref[1] == p_.ref_frame[1]) {
      compound_search_stack(mv_row, mv_col, weight);
    } else {
      compound_add_derived(mv_row, mv_col);
    }
  }

  AV2ENC_MVS_HD void scan_point(int delta_row, int delta_col) {
    int mv_row = p_.mi_row + delta_row;
    int mv_col = p_.mi_col + delta_col;
    const bool sb_border = (p_.mi_row & (p_.sb_h4 - 1)) == 0;
    if (delta_row < 0 && sb_border) {
      mv_col = (mv_col >> 1) << 1;
      delta_col = mv_col - p_.mi_col;
    }
    if (!mvs_mi_inside(mv_row, mv_col, p_)) {
      return;
    }
    if (mvs_mi_at(p_, mv_row, mv_col).ref[0] == vm::kRefNone) {
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

  AV2ENC_MVS_HD void scan_point_if_valid(int delta_row, int delta_col) {
    const int mv_row = p_.mi_row + delta_row;
    const int mv_col = p_.mi_col + delta_col;
    if (!mvs_mi_inside(mv_row, mv_col, p_)) {
      return;
    }
    if (p_.mi_col - 1 < 0) {
      return;
    }
    if (mvs_mi_at(p_, mv_row, mv_col).col_base ==
        mvs_mi_at(p_, mv_row, p_.mi_col - 1).col_base) {
      return;
    }
    scan_point(delta_row, delta_col);
  }

  AV2ENC_MVS_HD void scan_col(int delta_col) {
    delta_col += p_.mi_col & 1;
    scan_point_if_valid(p_.bh4 - 1, delta_col);
    if (p_.bh4 > 1) {
      scan_point_if_valid(0, delta_col);
    }
  }

  AV2ENC_MVS_HD void sort_nearest() {
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
        const vm::StackCandidate tmp = stack_->cand[start];
        stack_->cand[start] = stack_->cand[max_idx];
        stack_->cand[max_idx] = tmp;
      }
    }
  }

  AV2ENC_MVS_HD void fill_derived() {
    const int max_ref_mv_count = p_.max_drl_bits_minus_1 + 2;
    if (stack_->count >= max_ref_mv_count) {
      return;
    }
    for (int d = 0; d < derived_count_; ++d) {
      bool found = false;
      if (prune_count_ < vm::kMaxPrNum) {
        for (int i = 0; i < stack_->count; ++i) {
          ++prune_count_;
          const bool match0 =
              mvs_mv_eq(derived_[d].mv[0], stack_->cand[i].mv[0]);
          const bool match1 =
              !p_.is_compound ||
              mvs_mv_eq(derived_[d].mv[1], stack_->cand[i].mv[1]);
          if (match0 && match1) {
            found = true;
            break;
          }
        }
      }
      if (!found && stack_->count < max_ref_mv_count) {
        vm::StackCandidate& c = stack_->cand[stack_->count];
        c.mv[0] = derived_[d].mv[0];
        c.mv[1] = p_.is_compound ? derived_[d].mv[1] : derived_[d].mv[0];
        c.weight = 1;
        c.row_offset = 0;
        c.col_offset = 0;
        c.cwp = kMvsCwpEqual;
        ++stack_->count;
      }
    }
  }

  AV2ENC_MVS_HD void insert_mvp_candidate(int y_cand, int x_cand) {
    if (y_cand >= stack_->count || x_cand >= stack_->count) {
      return;
    }
    const imc::Mv cand0{stack_->cand[y_cand].mv[0].y,
                        stack_->cand[x_cand].mv[0].x};
    const imc::Mv cand1{stack_->cand[y_cand].mv[1].y,
                        stack_->cand[x_cand].mv[1].x};
    if (stack_->count >= vm::kMaxRefMvStackSize) {
      return;
    }
    if (prune_count_ < vm::kMaxPrNum) {
      for (int i = 0; i < stack_->count; ++i) {
        ++prune_count_;
        const bool match0 = mvs_mv_eq(cand0, stack_->cand[i].mv[0]);
        if (!p_.is_compound && match0) {
          return;
        }
        if (p_.is_compound && match0 &&
            mvs_mv_eq(cand1, stack_->cand[i].mv[1])) {
          return;
        }
      }
    }
    vm::StackCandidate& c = stack_->cand[stack_->count];
    c.mv[0] = cand0;
    c.mv[1] = p_.is_compound ? cand1 : cand0;
    c.weight = 1;
    c.row_offset = 0;
    c.col_offset = 0;
    c.cwp = kMvsCwpEqual;
    ++stack_->count;
  }

  AV2ENC_MVS_HD imc::Mv picture_clamp(imc::Mv mv) const {
    const std::int32_t low_y =
        -(p_.mi_row + p_.bh4) * imc::kMiSize * 8 - imc::kMvBorder;
    const std::int32_t high_y =
        (p_.mi_rows - p_.mi_row) * imc::kMiSize * 8 + imc::kMvBorder;
    const std::int32_t low_x =
        -(p_.mi_col + p_.bw4) * imc::kMiSize * 8 - imc::kMvBorder;
    const std::int32_t high_x =
        (p_.mi_cols - p_.mi_col) * imc::kMiSize * 8 + imc::kMvBorder;
    mv.y = mvs_clip3(low_y, high_y, mv.y);
    mv.x = mvs_clip3(low_x, high_x, mv.x);
    return mv;
  }

  AV2ENC_MVS_HD void clamp_all() {
    for (int i = 0; i < stack_->count; ++i) {
      for (int list = 0; list < 2; ++list) {
        stack_->cand[i].mv[list] = picture_clamp(stack_->cand[i].mv[list]);
      }
    }
  }

  const vm::StackParams& p_;
  vm::MvStack* stack_;
  int prune_count_ = 0;
  MvsDerivedCandidate derived_[vm::kMaxDrStackSize]{};
  int derived_count_ = 0;
  int derived_prune_count_ = 0;
  MvsSingleCandidate single_[vm::kMaxDrStackSize]{};
  int single_count_ = 0;
  int single_prune_count_ = 0;
};

// Single-reference or compound stack build.  Matches merge::build_mv_stack.
AV2ENC_MVS_HD vm::Status mvs_build_stack(const vm::StackParams& params,
                                                vm::MvStack* stack) {
  if (stack == nullptr) {
    return vm::Status::kInvalidArgument;
  }
  const vm::Status status = mvs_validate(params);
  if (status != vm::Status::kOk) {
    return status;
  }
  *stack = vm::MvStack{};
  MvsStackBuilder builder(params, stack);
  builder.run();
  return vm::Status::kOk;
}

// ---------------------------------------------------------------------------
// Multi-reference: two single-list stacks + the compound two-list stack.
// Matches multiref::build_pair_stacks.
// ---------------------------------------------------------------------------

struct MvsPairJob {
  vm::StackParams params;
  std::int8_t ref0 = 0;
  std::int8_t ref1 = 0;
};

struct MvsPairStacks {
  vm::MvStack list[2];
  vm::MvStack compound;
  int num_nearest[2] = {0, 0};
  int ok[2] = {0, 0};
  int compound_ok = 0;
};

AV2ENC_MVS_HD void mvs_build_pair_stacks(const MvsPairJob& job,
                                                MvsPairStacks* out) {
  *out = MvsPairStacks{};
  const std::int8_t refs[2] = {job.ref0, job.ref1};
  for (int list = 0; list < 2; ++list) {
    vm::StackParams local = job.params;
    local.ref_frame[0] = refs[list];
    local.ref_frame[1] = vm::kRefNone;
    local.is_compound = false;
    const vm::Status status = mvs_build_stack(local, &out->list[list]);
    out->ok[list] = status == vm::Status::kOk ? 1 : 0;
    if (out->ok[list] == 0) {
      return;
    }
    out->num_nearest[list] = out->list[list].num_nearest;
  }
  vm::StackParams compound = job.params;
  compound.ref_frame[0] = job.ref0;
  compound.ref_frame[1] = job.ref1;
  compound.is_compound = true;
  const vm::Status status = mvs_build_stack(compound, &out->compound);
  out->compound_ok = status == vm::Status::kOk ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Compound merge candidate set.  Matches
// multiref::build_compound_merge_candidates.
// ---------------------------------------------------------------------------

struct MvsMergeCandidate {
  imc::Mv mv[2];
  int drl_index[2] = {0, 0};
  int weight = 0;
  int single_index = 0;
};

struct MvsMergeSet {
  MvsMergeCandidate cand[kMaxMergeCandidates];
  int count = 0;
};

AV2ENC_MVS_HD bool mvs_merge_seen(const MvsMergeSet& set, imc::Mv mv0,
                                         imc::Mv mv1) {
  for (int i = 0; i < set.count; ++i) {
    if (mvs_mv_eq(set.cand[i].mv[0], mv0) &&
        mvs_mv_eq(set.cand[i].mv[1], mv1)) {
      return true;
    }
  }
  return false;
}

AV2ENC_MVS_HD void mvs_push_merge(MvsMergeSet* set, int i0, int i1,
                                         imc::Mv mv0, imc::Mv mv1,
                                         int weight0, int weight1) {
  if (set->count >= kMaxMergeCandidates) {
    return;
  }
  if (mvs_merge_seen(*set, mv0, mv1)) {
    return;
  }
  MvsMergeCandidate& c = set->cand[set->count];
  c.mv[0] = mv0;
  c.mv[1] = mv1;
  c.drl_index[0] = i0;
  c.drl_index[1] = i1;
  c.weight = weight0 + weight1;
  c.single_index = i0 == i1 ? 1 : 0;
  ++set->count;
}

AV2ENC_MVS_HD void mvs_build_merge_set(const MvsPairStacks& stacks,
                                              MvsMergeSet* set) {
  *set = MvsMergeSet{};
  if (stacks.ok[0] == 0 || stacks.ok[1] == 0) {
    return;
  }
  if (stacks.compound_ok != 0) {
    for (int i = 0; i < stacks.compound.count; ++i) {
      const vm::StackCandidate& c = stacks.compound.cand[i];
      mvs_push_merge(set, i, i, c.mv[0], c.mv[1], c.weight, c.weight);
    }
  }
  for (int i = 0; i < stacks.list[0].count; ++i) {
    for (int j = 0; j < stacks.list[1].count; ++j) {
      if (i == j) {
        continue;
      }
      const vm::StackCandidate& a = stacks.list[0].cand[i];
      const vm::StackCandidate& b = stacks.list[1].cand[j];
      mvs_push_merge(set, i, j, a.mv[0], b.mv[0], a.weight, b.weight);
    }
  }
}

}  // namespace av2enc::cuda::mvstack

#endif  // AV2ENC_CUDA_INTER_MV_STACK_MV_STACK_DEVICE_CUH_

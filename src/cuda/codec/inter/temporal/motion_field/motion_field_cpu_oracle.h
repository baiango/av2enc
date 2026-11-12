// Host-side oracle for the §7.22 block-to-grid storage loop.
//
// The CPU reference exposes `store_motion_field_entry()` (the post-MV ordering,
// allow-list and compression) but the mvMethod 0/1 composition lives in the
// `motion_field_estimation()` caller.  This helper reproduces that composition
// with the CPU arithmetic and then calls the CPU entry function, giving the
// GPU test an independent oracle for `mf_store_block()`.
//
// Test-only header; not included by any kernel or production TU.

#ifndef AV2ENC_CUDA_MOTION_FIELD_MOTION_FIELD_CPU_ORACLE_H_
#define AV2ENC_CUDA_MOTION_FIELD_MOTION_FIELD_CPU_ORACLE_H_

#include <algorithm>
#include <cstdint>

#include "cuda/codec/inter/temporal/motion_field/motion_field_device.cuh"
#include "codec/inter/temporal/motion_field.h"
#include "codec/inter/temporal/temporal_math.h"

namespace av2enc::cuda::motion_field {

inline void mf_test_cpu_store_block(const MfBlockStoreInput& in,
                                    std::int8_t* out_ref,
                                    std::int16_t* out_mv) {
  using CpuMv = inter::temporal::Mv;
  if (!in.enable_ref_frame_mvs) return;
  const int w8 = in.mi_cols >> 1;
  const int h8 = in.mi_rows >> 1;
  int bw4 = in.bw4;
  int bh4 = in.bh4;
  const int n = (bw4 <= 2 && bh4 <= 2 && !in.tip_frame_as_output) ? 4 : 8;
  bw4 = std::min(in.mi_cols - in.c, bw4);
  bh4 = std::min(in.mi_rows - in.r, bh4);
  if (bw4 <= 0 || bh4 <= 0) return;

  int ref0 = in.refs[0];
  int ref1 = in.refs[1];
  const bool is_wedge = inter::temporal::is_inter_ref(ref0) &&
                        inter::temporal::is_inter_ref(ref1) &&
                        ref0 != inter::temporal::kTipFrame && in.is_wedge != 0;
  const bool tip_pred = ref0 == inter::temporal::kTipFrame;
  int wtd = 0;
  const bool have_weight =
      inter::temporal::tip_weight(in.tip_global_wtd_index, &wtd);
  if (tip_pred) {
    ref0 = in.closest_past;
    ref1 = in.closest_future;
  }
  if ((tip_pred || in.tip_frame_as_output) && have_weight && wtd == 16) {
    ref1 = inter::temporal::kRefNone;
  }
  int ref_oh[2] = {in.ref_order_hint[0], in.ref_order_hint[1]};
  if (tip_pred) {
    ref_oh[0] = in.tip_past_order_hint;
    ref_oh[1] = in.tip_future_order_hint;
  }

  inter::temporal::TipConfig tip;
  tip.num_past_refs = in.num_past_refs;
  tip.num_future_refs = in.num_future_refs;
  tip.order_hint = in.order_hint;
  tip.closest_past_order_hint = in.tip_past_order_hint;
  tip.closest_future_order_hint = in.tip_future_order_hint;

  const int rows = (bh4 + 1) >> 1;
  const int cols = (bw4 + 1) >> 1;
  for (int i8 = 0; i8 < rows; ++i8) {
    for (int j8 = 0; j8 < cols; ++j8) {
      const int x8 = (in.c >> 1) + j8;
      const int y8 = (in.r >> 1) + i8;
      const int row = in.r + (i8 << 1);
      const int col = in.c + (j8 << 1);
      int refs[2] = {inter::temporal::kRefNone, inter::temporal::kRefNone};
      CpuMv mvs[2] = {CpuMv{0, 0}, CpuMv{0, 0}};
      inter::temporal::MvPair tip_cand;
      if (tip_pred) {
        inter::temporal::TipCandInput ci;
        ci.cand_row = row;
        ci.cand_col = col;
        const std::size_t mi_idx =
            static_cast<std::size_t>(row) * in.mi_cols + col;
        ci.base_row = in.mi_row_base != nullptr ? in.mi_row_base[mi_idx] : row;
        ci.base_col = in.mi_col_base != nullptr ? in.mi_col_base[mi_idx] : col;
        ci.is_16x16 =
            in.tip_sizes16 != nullptr && in.tip_sizes16[mi_idx] != 0;
        const std::size_t midx =
            (static_cast<std::size_t>(row) * in.mi_cols + col) * 4;
        ci.base_mv =
            CpuMv{in.src.mvs != nullptr ? in.src.mvs[midx + 0] : 0,
                  in.src.mvs != nullptr ? in.src.mvs[midx + 1] : 0};
        inter::temporal::MotionField field;
        field.w8 = in.field.w8;
        field.h8 = in.field.h8;
        field.mv = in.field.mv;
        field.valid = in.field.valid;
        field.offset = in.field.offset;
        ci.field = in.field.mv != nullptr ? &field : nullptr;
        tip_cand = inter::temporal::get_tip_cand(ci, tip);
      }
      for (int list = 0; list < 2; ++list) {
        const int ref_idx = list == 0 ? ref0 : ref1;
        if (!inter::temporal::is_inter_ref(ref_idx)) continue;
        int mv_row = 0;
        int mv_col = 0;
        if (in.mv_method > 0) {
          const bool use_refine = in.src.use_refinemv || tip_pred;
          const std::int32_t* plane =
              use_refine ? in.src.refine_mvs : in.src.mvs;
          if (plane == nullptr) continue;
          if (use_refine) {
            const std::size_t idx =
                ((static_cast<std::size_t>(i8 << 1) * in.src.delta_stride4 +
                  (j8 << 1)) *
                     4 +
                 list * 2);
            mv_row = plane[idx + 0];
            mv_col = plane[idx + 1];
          } else {
            const std::size_t idx =
                (static_cast<std::size_t>(row) * in.mi_cols + col) * 4 +
                list * 2;
            mv_row = plane[idx + 0];
            mv_col = plane[idx + 1];
          }
          if (in.mv_method == 1) {
            if (in.src.mv_deltas == nullptr) continue;
            if (n == 4 && !tip_pred) {
              int tr = 0;
              int tc = 0;
              for (int a = 0; a < 2; ++a) {
                for (int b = 0; b < 2; ++b) {
                  const std::size_t idx =
                      ((static_cast<std::size_t>(a) * in.src.delta_stride4 + b) *
                           4 +
                       list * 2);
                  tr += in.src.mv_deltas[idx + 0];
                  tc += in.src.mv_deltas[idx + 1];
                }
              }
              mv_row += inter::temporal::round2_signed(tr, 3);
              mv_col += inter::temporal::round2_signed(tc, 3);
            } else {
              const std::size_t idx =
                  ((static_cast<std::size_t>(i8 << 1) * in.src.delta_stride4 +
                    (j8 << 1)) *
                       4 +
                   list * 2);
              mv_row += inter::temporal::round2_signed(in.src.mv_deltas[idx + 0], 1);
              mv_col += inter::temporal::round2_signed(in.src.mv_deltas[idx + 1], 1);
            }
          }
        } else {
          if (tip_pred) {
            mv_row = tip_cand.mv[list].row;
            mv_col = tip_cand.mv[list].col;
          } else if (in.src.motion_mode >= 2 && !in.src.force_integer_mv) {
            mv_row = in.warp_mv[list].row;
            mv_col = in.warp_mv[list].col;
          } else if (in.src.is_global_mv_cand && !in.src.force_integer_mv) {
            mv_row = in.global_mv[list].row;
            mv_col = in.global_mv[list].col;
          } else {
            if (in.src.mvs == nullptr) continue;
            const std::size_t idx =
                (static_cast<std::size_t>(row) * in.mi_cols + col) * 4 +
                list * 2;
            mv_row = in.src.mvs[idx + 0];
            mv_col = in.src.mvs[idx + 1];
          }
        }
        if (inter::temporal::abs_int(mv_row) <=
                inter::temporal::kRefMvsLimit &&
            inter::temporal::abs_int(mv_col) <=
                inter::temporal::kRefMvsLimit) {
          refs[list] = ref_idx;
          mvs[list] = CpuMv{mv_row, mv_col};
        }
      }
      inter::temporal::MotionFieldStoreEntry entry;
      entry.refs[0] = refs[0];
      entry.refs[1] = refs[1];
      entry.mv[0] = mvs[0];
      entry.mv[1] = mvs[1];
      entry.is_wedge = is_wedge;
      entry.mask = in.mask;
      entry.mask_stride = in.mask_stride;
      entry.i8 = i8;
      entry.j8 = j8;
      entry.order_hint = in.order_hint;
      entry.ref_order_hint[0] = ref_oh[0];
      entry.ref_order_hint[1] = ref_oh[1];
      inter::temporal::RefMvField view;
      view.w8 = w8;
      view.h8 = h8;
      view.ref = out_ref;
      view.mv = out_mv;
      (void)inter::temporal::store_motion_field_entry(entry, &view, y8, x8);
    }
  }
}

}  // namespace av2enc::cuda::motion_field

#endif  // AV2ENC_CUDA_MOTION_FIELD_MOTION_FIELD_CPU_ORACLE_H_

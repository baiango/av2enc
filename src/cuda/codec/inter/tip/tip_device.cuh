// Shared AV2 TIP arithmetic — host and device.
//
// This header is compiled three ways:
//   * by nvcc inside the kernels (`tip.cu`),
//   * by nvcc in the real parity test (`tip_test.cu`, host side),
//   * by the host compiler with CUDA keywords stubbed in
//     (`tip_emulation_test.cpp`).
//
// The integer operations are transcriptions of
// `src/encoder/inter/temporal/temporal_math.h`,
// `src/encoder/inter/temporal/temporal_predict.h` and the §7.10 loops of
// `src/encoder/inter/temporal/motion_field.cpp`.  §4.8 helpers and the §7.13.3.17
// sample-location derivation are reused from `advanced_common.h` /
// `warp_common.h`, so there is only one implementation of those.

#ifndef AV2ENC_CUDA_TIP_TIP_DEVICE_CUH_
#define AV2ENC_CUDA_TIP_TIP_DEVICE_CUH_

#include <cstddef>
#include <cstdint>

#include "tip.h"

#include "codec/inter/advanced/warp_common.h"
#include "codec/inter/mc/mc_core.h"
#include "codec/inter/temporal/temporal_math.h"

#if defined(__CUDACC__)
#define AV2ENC_TIP_HD __host__ __device__
#else
#define AV2ENC_TIP_HD
#endif

namespace av2enc::cuda::tip {
namespace dev {

using inter::temporal::kInterPostRoundCompound;
using inter::temporal::kInterRound0;
using inter::temporal::kInterRound1Compound;
using inter::temporal::kInterRound1Single;
using inter::temporal::kMaxFrameDistance;
using inter::temporal::kMiSize;
using inter::temporal::kMvLow;
using inter::temporal::kMvUpp;
using inter::temporal::kRefMvsLimit;
using inter::temporal::kScaleSubpelBits;
using inter::temporal::kSubpelBits;

// §4.8 / §7.9 helpers.  `abs_int`, `clip3`, `round2`, `round2_signed`,
// `round2_signed64` and `get_relative_dist` are the same host/device functions
// the CPU reference uses (advanced_common.h).
AV2ENC_TIP_HD inline int abs_int(int value) { return inter::adv::abs_int(value); }
AV2ENC_TIP_HD inline int clip3(int low, int high, int value) {
  return inter::adv::clip3(low, high, value);
}
AV2ENC_TIP_HD inline int round2(int value, int n) {
  return inter::adv::round2(value, n);
}
AV2ENC_TIP_HD inline int round2_signed(int value, int n) {
  return inter::adv::round2_signed(value, n);
}
AV2ENC_TIP_HD inline std::int64_t round2_signed64(std::int64_t value, int n) {
  return inter::adv::round2_signed64(value, n);
}
AV2ENC_TIP_HD inline int get_relative_dist(int a, int b) {
  return inter::adv::get_relative_dist(a, b);
}
AV2ENC_TIP_HD inline int clip1(int value, int bit_depth) {
  return inter::adv::clip1(value, bit_depth);
}

AV2ENC_TIP_HD inline std::size_t mf_index(int w8, int y8, int x8) {
  return static_cast<std::size_t>(y8) * static_cast<std::size_t>(w8) +
         static_cast<std::size_t>(x8);
}

// ---------------------------------------------------------------------------
// §7.9.4/.5 projection
// ---------------------------------------------------------------------------

// §7.9.4 Div_Mult[32].  This is a constexpr function rather than a
// namespace-scope array so nvcc never has to materialise a host array in
// device code (the host/device `inline constexpr` array form fails there).
AV2ENC_TIP_HD constexpr std::int32_t div_mult(int distance) {
  switch (distance) {
    case 0: return 0;
    case 1: return 16384;
    case 2: return 8192;
    case 3: return 5461;
    case 4: return 4096;
    case 5: return 3276;
    case 6: return 2730;
    case 7: return 2340;
    case 8: return 2048;
    case 9: return 1820;
    case 10: return 1638;
    case 11: return 1489;
    case 12: return 1365;
    case 13: return 1260;
    case 14: return 1170;
    case 15: return 1092;
    case 16: return 1024;
    case 17: return 963;
    case 18: return 910;
    case 19: return 862;
    case 20: return 819;
    case 21: return 780;
    case 22: return 744;
    case 23: return 712;
    case 24: return 682;
    case 25: return 655;
    case 26: return 630;
    case 27: return 606;
    case 28: return 585;
    case 29: return 564;
    case 30: return 546;
    case 31: return 528;
  }
  return 0;
}

AV2ENC_TIP_HD inline Mv mv_projection(Mv mv, int numerator, int denominator) {
  const int clipped_denominator =
      denominator < 0 ? 0
                      : (denominator < kMaxFrameDistance ? denominator
                                                         : kMaxFrameDistance);
  const int clipped_numerator =
      clip3(-kMaxFrameDistance, kMaxFrameDistance, numerator);
  const std::int64_t scale =
      static_cast<std::int64_t>(div_mult(clipped_denominator));
  Mv out;
  out.row = static_cast<std::int32_t>(clip3(
      kMvLow + 1, kMvUpp - 1,
      static_cast<int>(round2_signed64(static_cast<std::int64_t>(mv.row) *
                                           clipped_numerator * scale,
                                       14))));
  out.col = static_cast<std::int32_t>(clip3(
      kMvLow + 1, kMvUpp - 1,
      static_cast<int>(round2_signed64(static_cast<std::int64_t>(mv.col) *
                                           clipped_numerator * scale,
                                       14))));
  return out;
}

AV2ENC_TIP_HD inline Mv mv_projection_clamp(Mv mv, int numerator,
                                            int denominator) {
  Mv out = mv_projection(mv, numerator, denominator);
  out.row = clip3(-kRefMvsLimit, kRefMvsLimit, out.row);
  out.col = clip3(-kRefMvsLimit, kRefMvsLimit, out.col);
  return out;
}

// ---------------------------------------------------------------------------
// §7.9.2 / §7.9.6 / §7.9.7 / §7.9.8
// ---------------------------------------------------------------------------

AV2ENC_TIP_HD inline int tmvp_shift(int sb_64x64, int proj_step) {
  return (sb_64x64 != 0 || proj_step == 1) ? 3 : 4;
}
AV2ENC_TIP_HD inline int tmvp_unit(int x8, int sb_64x64, int proj_step) {
  return x8 >> tmvp_shift(sb_64x64, proj_step);
}
AV2ENC_TIP_HD inline int tmvp_phase(int x8, int sb_64x64, int proj_step) {
  return tmvp_unit(x8, sb_64x64, proj_step) % 3;
}
AV2ENC_TIP_HD inline bool tmvp_avail(int base8, int loc8, int max8,
                                     int sb_64x64, int proj_step) {
  if (loc8 >= max8) {
    return false;
  }
  return tmvp_unit(base8, sb_64x64, proj_step) ==
         tmvp_unit(loc8, sb_64x64, proj_step);
}

struct SampledPosition {
  bool valid = false;
  int x8 = 0;
  int y8 = 0;
};

AV2ENC_TIP_HD inline int project_no_constraint(int v8, std::int32_t delta,
                                               int dst_sign, int max8,
                                               bool* pos_valid) {
  std::int32_t offset8;
  if (delta >= 0) {
    offset8 = delta >> (3 + 1 + inter::temporal::kMiSizeLog2);
  } else {
    offset8 = -((-delta) >> (3 + 1 + inter::temporal::kMiSizeLog2));
  }
  v8 += dst_sign * offset8;
  if (v8 < 0 || v8 >= max8) {
    *pos_valid = false;
  }
  return v8;
}

AV2ENC_TIP_HD inline SampledPosition sampled_position(int x8, int y8,
                                                      int dst_sign, Mv proj_mv,
                                                      int mi_cols, int mi_rows,
                                                      int proj_step) {
  SampledPosition out;
  out.valid = true;
  const int y = project_no_constraint(y8, proj_mv.row, dst_sign, mi_rows >> 1,
                                      &out.valid);
  const int x = project_no_constraint(x8, proj_mv.col, dst_sign, mi_cols >> 1,
                                      &out.valid);
  out.x8 = x;
  out.y8 = y;
  if (proj_step == 2) {
    out.x8 -= out.x8 & 1;
    out.y8 -= out.y8 & 1;
  }
  return out;
}

AV2ENC_TIP_HD inline bool check_block_position(bool pos_valid, int base_x8,
                                               int base_y8, int pos_x8,
                                               int pos_y8, int sb_64x64,
                                               int proj_step) {
  if (!pos_valid) {
    return false;
  }
  const int shift = tmvp_shift(sb_64x64, proj_step);
  for (int ord = 0; ord < 2; ++ord) {
    const int v8 = ord != 0 ? base_x8 : base_y8;
    const int sb_off8 = 1 << shift;
    const int max_off8 = proj_step > 1 ? (ord != 0 ? sb_off8 : 0)
                                       : (ord != 0 ? (sb_off8 >> 1) : 0);
    const int base8 = (v8 >> shift) << shift;
    const int pos8 = ord != 0 ? pos_x8 : pos_y8;
    if (pos8 < base8 - max_off8 || pos8 >= base8 + sb_off8 + max_off8) {
      return false;
    }
  }
  return true;
}

// §7.22/§7.13.3.2 get_tip_offsets().
AV2ENC_TIP_HD inline TipOffsets tip_offsets(const TipMotionFieldConfig& cfg) {
  TipOffsets out;
  if (cfg.num_future_refs > 0 && cfg.num_past_refs > 0) {
    out.ref_offset = get_relative_dist(cfg.closest_future_order_hint,
                                       cfg.closest_past_order_hint);
  } else {
    out.ref_offset = get_relative_dist(cfg.closest_past_order_hint,
                                       cfg.closest_future_order_hint);
  }
  out.past_offset = get_relative_dist(cfg.order_hint, cfg.closest_past_order_hint);
  out.future_offset =
      get_relative_dist(cfg.order_hint, cfg.closest_future_order_hint);
  out.ref_offset = out.ref_offset < kMaxFrameDistance ? out.ref_offset
                                                      : kMaxFrameDistance;
  return out;
}

// §7.10.5 calc_avg().
AV2ENC_TIP_HD inline int calc_avg(int n, int d) {
  if (d == 1) {
    return n;
  }
  if (d == 2) {
    return round2_signed(n, 1);
  }
  if (d == 3) {
    return round2_signed(n * 85, 8);
  }
  return round2_signed(n, 2);
}

// ---------------------------------------------------------------------------
// §7.10.2 temporal scale
// ---------------------------------------------------------------------------

AV2ENC_TIP_HD inline void scale_cell(const TipField& field,
                                     int ref_offset, int y8, int x8) {
  const std::size_t pos = mf_index(field.w8, y8, x8);
  const Mv mv{field.mv[pos * 2 + 0], field.mv[pos * 2 + 1]};
  if (field.valid[pos] != 0) {
    const int start_offset = field.offset[pos];
    const Mv scaled = mv_projection_clamp(mv, ref_offset, start_offset);
    field.mv[pos * 2 + 0] = scaled.row;
    field.mv[pos * 2 + 1] = scaled.col;
    field.valid[pos] = 1;
  }
  field.offset[pos] = ref_offset;
}

// ---------------------------------------------------------------------------
// §7.10.3 hole fill (one TMVP superblock, reference row-major order)
// ---------------------------------------------------------------------------

// kTipDirs[5] = {-1,0}, {0,-1}, {1,0}, {0,1}, {0,0}.  Accessors keep the
// table out of device code for the same reason as div_mult().
AV2ENC_TIP_HD constexpr int tip_dir_row(int dir) {
  return dir == 0 ? -1 : (dir == 2 ? 1 : 0);
}
AV2ENC_TIP_HD constexpr int tip_dir_col(int dir) {
  return dir == 1 ? -1 : (dir == 3 ? 1 : 0);
}

AV2ENC_TIP_HD inline void hole_fill_superblock(const TipField& field,
                                               const TipMotionFieldConfig& cfg,
                                               int y8, int x8) {
  const int step = cfg.proj_step;
  const int sb8 = 1 << tmvp_shift(cfg.sb_64x64, cfg.proj_step);
  const int end_row = (y8 + sb8) < field.h8 ? (y8 + sb8) : field.h8;
  const int end_col = (x8 + sb8) < field.w8 ? (x8 + sb8) : field.w8;
  for (int row8 = y8; row8 < end_row; row8 += step) {
    for (int col8 = x8; col8 < end_col; col8 += step) {
      for (int dir = 0; dir < 4; ++dir) {
        const int dst_row = row8 + tip_dir_row(dir) * step;
        const int dst_col = col8 + tip_dir_col(dir) * step;
        if (dst_row < y8 || dst_row >= end_row || dst_col < x8 ||
            dst_col >= end_col ||
            field.valid[mf_index(field.w8, dst_row, dst_col)] != 0) {
          continue;
        }
        const std::size_t src = mf_index(field.w8, row8, col8);
        const std::size_t dst = mf_index(field.w8, dst_row, dst_col);
        field.valid[dst] = field.valid[src];
        field.mv[dst * 2 + 0] = field.mv[src * 2 + 0];
        field.mv[dst * 2 + 1] = field.mv[src * 2 + 1];
        field.offset[dst] = field.offset[src];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// §7.10.4 block-average filter
// ---------------------------------------------------------------------------

// §7.10.4 Weight_Div_Mult[1..5].  constexpr accessor, not a host array.
AV2ENC_TIP_HD constexpr int weight_div_mult(int count) {
  return count == 1 ? 65536
         : count == 2 ? 32768
         : count == 3 ? 21845
         : count == 4 ? 16384
         : count == 5 ? 13107
                      : 0;
}

AV2ENC_TIP_HD inline void block_average_cell(const TipField& field,
                                             const TipMotionFieldConfig& cfg,
                                             const TipFieldScratch& scratch,
                                             int y8, int x8) {
  const int step = cfg.proj_step;
  const int sb8 = 1 << tmvp_shift(cfg.sb_64x64, cfg.proj_step);
  const int base_row = (y8 / sb8) * sb8;
  const int base_col = (x8 / sb8) * sb8;
  const int end_row = (base_row + sb8) < field.h8 ? (base_row + sb8) : field.h8;
  const int end_col = (base_col + sb8) < field.w8 ? (base_col + sb8) : field.w8;
  int mv[2] = {0, 0};
  int count = 0;
  for (int dir = 0; dir < 5; ++dir) {
    const int dst_row = y8 + tip_dir_row(dir) * step;
    const int dst_col = x8 + tip_dir_col(dir) * step;
    if (dst_row < base_row || dst_row >= end_row || dst_col < base_col ||
        dst_col >= end_col ||
        field.valid[mf_index(field.w8, dst_row, dst_col)] == 0) {
      continue;
    }
    const std::size_t src = mf_index(field.w8, dst_row, dst_col);
    mv[0] += field.mv[src * 2 + 0];
    mv[1] += field.mv[src * 2 + 1];
    ++count;
  }
  const std::size_t dst = mf_index(field.w8, y8, x8);
  if (count == 0) {
    scratch.avg_valid[dst] = 0;
    scratch.avg_mv[dst * 2 + 0] = -(1 << 15);
    scratch.avg_mv[dst * 2 + 1] = -(1 << 15);
  } else {
    scratch.avg_valid[dst] = 1;
    scratch.avg_mv[dst * 2 + 0] =
        round2_signed(mv[0] * weight_div_mult(count), 16);
    scratch.avg_mv[dst * 2 + 1] =
        round2_signed(mv[1] * weight_div_mult(count), 16);
  }
}

AV2ENC_TIP_HD inline void copy_average_cell(const TipField& field,
                                            const TipFieldScratch& scratch,
                                            int y8, int x8) {
  const std::size_t pos = mf_index(field.w8, y8, x8);
  field.mv[pos * 2 + 0] = scratch.avg_mv[pos * 2 + 0];
  field.mv[pos * 2 + 1] = scratch.avg_mv[pos * 2 + 1];
  field.valid[pos] = scratch.avg_valid[pos];
}

// ---------------------------------------------------------------------------
// §7.10.5 sample-gap fill (ProjStep == 2)
// ---------------------------------------------------------------------------

AV2ENC_TIP_HD inline void fill_sample_gap_cell(const TipField& field,
                                               const TipMotionFieldConfig& cfg,
                                               int y8, int x8, int dy, int dx) {
  if (field.valid[mf_index(field.w8, y8, x8)] == 0 ||
      y8 + dy == field.h8 || x8 + dx == field.w8) {
    return;
  }
  const int cur_offset = field.offset[mf_index(field.w8, y8, x8)];
  int count = 0;
  int avg[2] = {0, 0};
  for (int i = 0; i < 4; ++i) {
    const int cand_x = i & 1;
    const int cand_y = i >> 1;
    const bool available =
        dy >= cand_y && dx >= cand_x &&
        tmvp_avail(x8, x8 + 2 * cand_x, field.w8, cfg.sb_64x64,
                   cfg.proj_step) &&
        tmvp_avail(y8, y8 + 2 * cand_y, field.h8, cfg.sb_64x64,
                   cfg.proj_step) &&
        field.valid[mf_index(field.w8, y8 + 2 * cand_y, x8 + 2 * cand_x)] != 0;
    if (!available) {
      continue;
    }
    ++count;
    const std::size_t src =
        mf_index(field.w8, y8 + 2 * cand_y, x8 + 2 * cand_x);
    Mv proj;
    if (i == 0) {
      proj = Mv{field.mv[src * 2 + 0], field.mv[src * 2 + 1]};
    } else {
      proj = mv_projection_clamp(
          Mv{field.mv[src * 2 + 0], field.mv[src * 2 + 1]}, cur_offset,
          field.offset[src]);
    }
    avg[0] += proj.row;
    avg[1] += proj.col;
  }
  const std::size_t dst = mf_index(field.w8, y8 + dy, x8 + dx);
  field.offset[dst] = cur_offset;
  field.mv[dst * 2 + 0] = calc_avg(avg[0], count);
  field.mv[dst * 2 + 1] = calc_avg(avg[1], count);
  field.valid[dst] = 1;
}

// ---------------------------------------------------------------------------
// §7.10.6 build TIP block
// ---------------------------------------------------------------------------

AV2ENC_TIP_HD inline void derive_tip_block(
    const TipField& field, const TipMotionFieldConfig& cfg,
    const TipOffsets& offsets, int bw4, int bh4, Mv global_mv, int row,
    int col, TipPlanBlock* out, Mv* mv_grid, Mv* mv_grid1) {
  const int y8 = row >> 1;
  const int x8 = col >> 1;
  Mv local[2];
  const std::size_t pos = mf_index(field.w8, y8, x8);
  if (field.valid[pos] == 0) {
    local[0] = Mv{0, 0};
    local[1] = Mv{0, 0};
  } else {
    const Mv field_mv{field.mv[pos * 2 + 0], field.mv[pos * 2 + 1]};
    local[0] = mv_projection(field_mv, offsets.past_offset,
                             offsets.ref_offset);
    local[1] = mv_projection(field_mv, offsets.future_offset,
                             offsets.ref_offset);
  }
  out->mi_row = row;
  out->mi_col = col;
  out->bw4 = bw4;
  out->bh4 = bh4;
  out->mv[0] = Mv{local[0].row + global_mv.row, local[0].col + global_mv.col};
  out->mv[1] = Mv{local[1].row + global_mv.row, local[1].col + global_mv.col};
  out->motion_field_valid = field.valid[pos] != 0 ? 1 : 0;
  for (int i = 0; i < bh4; ++i) {
    for (int j = 0; j < bw4; ++j) {
      const int r = row + i;
      const int c = col + j;
      if (r < cfg.mi_rows && c < cfg.mi_cols) {
        const std::size_t index =
            static_cast<std::size_t>(r) *
                static_cast<std::size_t>(cfg.mi_cols) +
            static_cast<std::size_t>(c);
        mv_grid[index] = out->mv[0];
        if (mv_grid1 != nullptr) {
          mv_grid1[index] = out->mv[1];
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// §7.13.3.18 two-pass subpel prediction
// ---------------------------------------------------------------------------

template <typename Sample>
AV2ENC_TIP_HD inline std::int32_t horizontal_sum(
    const Sample* ref, int stride, int row, int col, int first_x, int last_x,
    const std::int16_t* filter) {
  std::int32_t sum = 0;
  for (int t = 0; t < 8; ++t) {
    const int cx = clip3(first_x, last_x, col + t);
    sum += static_cast<std::int32_t>(filter[t]) *
           static_cast<std::int32_t>(
               ref[static_cast<std::size_t>(row) *
                       static_cast<std::size_t>(stride) +
                   static_cast<std::size_t>(cx)]);
  }
  return sum;
}

template <typename Sample>
AV2ENC_TIP_HD inline std::int32_t predict_sample(
    const Sample* ref, int stride, int first_x, int first_y, int last_x,
    int last_y, std::int32_t start_x, std::int32_t start_y, std::int32_t step_x,
    std::int32_t step_y, InterpFilter filter_h, InterpFilter filter_v,
    int round1, int bit_depth, bool clip_result, int r, int c) {
  const std::int32_t p_x = start_x + step_x * c;
  const int col = static_cast<int>(p_x >> kScaleSubpelBits) - 3;
  const int phase_h = static_cast<int>((p_x >> 6) & inter::mc::kSubpelMask);
  const std::int16_t* fh =
      &inter::mc::kSubpelFilters[static_cast<int>(filter_h)][phase_h][0];

  const std::int32_t p_y = (start_y & 1023) + step_y * r;
  const int row_base = static_cast<int>(p_y >> kScaleSubpelBits);
  const int phase_v = static_cast<int>((p_y >> 6) & inter::mc::kSubpelMask);
  const std::int16_t* fv =
      &inter::mc::kSubpelFilters[static_cast<int>(filter_v)][phase_v][0];

  std::int32_t sum_v = 0;
  for (int t = 0; t < 8; ++t) {
    const int row = clip3(
        first_y, last_y,
        static_cast<int>(start_y >> kScaleSubpelBits) + row_base + t - 3);
    const std::int32_t hsum =
        horizontal_sum(ref, stride, row, col, first_x, last_x, fh);
    const std::int32_t intermediate = round2(hsum, kInterRound0);
    sum_v += static_cast<std::int32_t>(fv[t]) * intermediate;
  }
  const std::int32_t result = round2(sum_v, round1);
  return clip_result ? clip1(result, bit_depth) : result;
}

// One independent TIP prediction job.  `tid`/`nthr` let a CUDA block or a host
// emulation loop share the same code.
template <typename Sample>
AV2ENC_TIP_HD inline void predict_job_thread(const TipPredictJob& job, int tid,
                                             int nthr) {
  if (job.w <= 0 || job.h <= 0 || job.dst == nullptr) {
    return;
  }
  const bool single = job.num_refs == 1 || job.cwp_weight == 16;
  const InterpFilter filter = static_cast<InterpFilter>(job.filter);
  const InterpFilter filter_h =
      inter::mc::mc_pass_filter(filter, job.w <= 4);
  const InterpFilter filter_v =
      inter::mc::mc_pass_filter(filter, job.h <= 4);

  int first_x[2] = {0, 0};
  int first_y[2] = {0, 0};
  int last_x[2] = {0, 0};
  int last_y[2] = {0, 0};
  const int lists = single ? 1 : 2;
  for (int list = 0; list < lists; ++list) {
    if (job.use_ref_area != 0) {
      first_x[list] = job.area_first_x[list];
      first_y[list] = job.area_first_y[list];
      last_x[list] = job.area_last_x[list];
      last_y[list] = job.area_last_y[list];
    } else {
      first_x[list] = 0;
      first_y[list] = 0;
      last_x[list] = job.ref[list].width - 1;
      last_y[list] = job.ref[list].height - 1;
    }
  }

  inter::adv::MvScaleOutput loc[2];
  for (int list = 0; list < lists; ++list) {
    int mv_pair[2] = {job.mv[list].row, job.mv[list].col};
    const int ref_luma_w = job.ref[list].width << job.sub_x;
    const int ref_luma_h = job.ref[list].height << job.sub_y;
    loc[list] = inter::adv::motion_vector_scaling(
        job.sub_x > 0 ? 1 : 0, job.sub_x, job.sub_y, job.x, job.y, mv_pair,
        false, ref_luma_w, ref_luma_h, job.frame_w, job.frame_h);
  }

  Sample* dst = static_cast<Sample*>(job.dst);
  const int plane_samples = job.w * job.h;
  for (int i = tid; i < plane_samples; i += nthr) {
    const int r = i / job.w;
    const int c = i - r * job.w;
    std::int32_t value = 0;
    if (single) {
      const Sample* ref = static_cast<const Sample*>(job.ref[0].data);
      value = predict_sample(ref, job.ref[0].stride, first_x[0], first_y[0],
                             last_x[0], last_y[0], loc[0].start_x,
                             loc[0].start_y, loc[0].step_x, loc[0].step_y,
                             filter_h, filter_v, kInterRound1Single,
                             job.bit_depth, true, r, c);
    } else {
      const Sample* ref0 = static_cast<const Sample*>(job.ref[0].data);
      const Sample* ref1 = static_cast<const Sample*>(job.ref[1].data);
      const std::int32_t v0 =
          predict_sample(ref0, job.ref[0].stride, first_x[0], first_y[0],
                         last_x[0], last_y[0], loc[0].start_x, loc[0].start_y,
                         loc[0].step_x, loc[0].step_y, filter_h, filter_v,
                         kInterRound1Compound, job.bit_depth, false, r, c);
      const std::int32_t v1 =
          predict_sample(ref1, job.ref[1].stride, first_x[1], first_y[1],
                         last_x[1], last_y[1], loc[1].start_x, loc[1].start_y,
                         loc[1].step_x, loc[1].step_y, filter_h, filter_v,
                         kInterRound1Compound, job.bit_depth, false, r, c);
      value = clip1(round2(job.cwp_weight * v0 +
                               (16 - job.cwp_weight) * v1,
                           4 + kInterPostRoundCompound),
                    job.bit_depth);
    }
    dst[static_cast<std::size_t>(r) *
            static_cast<std::size_t>(job.dst_stride) +
        static_cast<std::size_t>(c)] = static_cast<Sample>(value);
  }
}

// Builds the per-plane job for one plan block (mirrors predict_tip_plan()).
AV2ENC_TIP_HD inline bool plan_block_job(const TipPlanBlock& block,
                                         const TipPlanPredictConfig& cfg,
                                         int plane, TipPredictJob* job) {
  const int w = (block.bw4 * kMiSize) >> cfg.sub_x[plane];
  const int h = (block.bh4 * kMiSize) >> cfg.sub_y[plane];
  const int x = (block.mi_col * kMiSize) >> cfg.sub_x[plane];
  const int y = (block.mi_row * kMiSize) >> cfg.sub_y[plane];
  const int vis_w = (w < cfg.dst_width[plane] - x) ? w
                                                   : cfg.dst_width[plane] - x;
  const int vis_h = (h < cfg.dst_height[plane] - y) ? h
                                                    : cfg.dst_height[plane] - y;
  if (vis_w <= 0 || vis_h <= 0) {
    return false;
  }
  *job = TipPredictJob{};
  job->ref[0] = cfg.ref[plane][0];
  job->ref[1] = cfg.ref[plane][1];
  const std::size_t dst_offset =
      (static_cast<std::size_t>(y) *
           static_cast<std::size_t>(cfg.dst_stride[plane]) +
       static_cast<std::size_t>(x)) *
      static_cast<std::size_t>(cfg.sample_bytes);
  job->dst = static_cast<std::uint8_t*>(cfg.dst[plane]) + dst_offset;
  job->dst_stride = cfg.dst_stride[plane];
  job->x = x;
  job->y = y;
  job->w = vis_w;
  job->h = vis_h;
  job->sub_x = cfg.sub_x[plane];
  job->sub_y = cfg.sub_y[plane];
  job->frame_w = cfg.frame_w;
  job->frame_h = cfg.frame_h;
  job->mv[0] = block.mv[0];
  job->mv[1] = block.mv[1];
  job->num_refs = 2;
  job->cwp_weight = cfg.cwp_weight;
  job->filter = cfg.filter;
  job->bit_depth = cfg.bit_depth;
  job->sample_bytes = cfg.sample_bytes;
  job->use_ref_area = 0;
  return true;
}

}  // namespace dev
}  // namespace av2enc::cuda::tip

#endif  // AV2ENC_CUDA_TIP_TIP_DEVICE_CUH_

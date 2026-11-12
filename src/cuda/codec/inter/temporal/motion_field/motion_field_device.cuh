// Shared AV2 temporal motion-field arithmetic — host and device.
//
// This header is compiled three ways:
//   * by nvcc inside the kernels (`motion_field.cu`),
//   * by nvcc in the real GPU parity test (`motion_field_test.cu`, host side),
//   * by the host compiler with CUDA keywords stubbed in
//     (`motion_field_emulation_test.cpp`).
//
// Every function is a transcription of the CPU reference:
//   src/encoder/inter/temporal/temporal_math.h
//   src/encoder/inter/temporal/motion_field.cpp
//   src/encoder/inter/temporal/ref_mv_bank.cpp
// using the same §4.8 helpers from advanced_common.h.  There is one
// implementation of the integer arithmetic; only the loop placement differs
// between the host emulation and the kernels.
//
// Ordering notes for the GPU port:
//   * §7.9 projection writes are order-dependent (the replace rule consults
//     the field written by earlier source cells), so one thread applies one
//     whole projection job.  Jobs are launched in process-list order.
//   * §7.9 trajectory gap fill reads only lattice anchors and writes only the
//     three interpolated neighbours, so it is safely cell-parallel.
//   * §7.22 block storage writes disjoint 8x8 cells per block descriptor.
//   * §7.12.2 stack generation is per-block state; one thread per block.

#ifndef AV2ENC_CUDA_MOTION_FIELD_MOTION_FIELD_DEVICE_CUH_
#define AV2ENC_CUDA_MOTION_FIELD_MOTION_FIELD_DEVICE_CUH_

#include <cstddef>
#include <cstdint>

#include "cuda/codec/inter/temporal/motion_field/motion_field.h"
#include "codec/inter/advanced/advanced_common.h"

#if defined(__CUDACC__) || defined(__CUDA__)
#define AV2ENC_MF_HD __host__ __device__ __forceinline__
#else
#define AV2ENC_MF_HD inline
#endif

namespace av2enc::cuda::motion_field {

namespace t = ::av2enc::inter::temporal;

// Device-safe §9.2 constants.  These are functions, not namespace-scope
// variables, so device code never forms a host symbol reference; the
// static_asserts pin them to the shared CPU header.

AV2ENC_MF_HD constexpr int mfkRefNone() { return -1; }
static_assert(mfkRefNone() == t::kRefNone);
AV2ENC_MF_HD constexpr int mfkRefMvsLimit() { return 2047; }
static_assert(mfkRefMvsLimit() == t::kRefMvsLimit);
AV2ENC_MF_HD constexpr int mfkRestrictedOrderHint() { return -1; }
static_assert(mfkRestrictedOrderHint() == t::kRestrictedOrderHint);
AV2ENC_MF_HD constexpr int mfkRefMvBankSize() { return 4; }
static_assert(mfkRefMvBankSize() == t::kRefMvBankSize);
AV2ENC_MF_HD constexpr int mfkMaxRefMvStackSize() { return 6; }
static_assert(mfkMaxRefMvStackSize() == t::kMaxRefMvStackSize);
AV2ENC_MF_HD constexpr int mfkMaxFrameDistance() { return 31; }
static_assert(mfkMaxFrameDistance() == t::kMaxFrameDistance);
AV2ENC_MF_HD constexpr int mfkMiSize() { return 4; }
static_assert(mfkMiSize() == t::kMiSize);
AV2ENC_MF_HD constexpr int mfkMvUpp() { return 65536; }
static_assert(mfkMvUpp() == t::kMvUpp);
AV2ENC_MF_HD constexpr int mfkMvLow() { return -65536; }
static_assert(mfkMvLow() == t::kMvLow);
AV2ENC_MF_HD constexpr int mfkRefsPerFrame() { return 7; }
static_assert(mfkRefsPerFrame() == t::kRefsPerFrame);
AV2ENC_MF_HD constexpr int mfkMaxPrNum() { return 16; }
static_assert(mfkMaxPrNum() == t::kMaxPrNum);
AV2ENC_MF_HD constexpr int mfkTipFrame() { return 7; }
static_assert(mfkTipFrame() == t::kTipFrame);
AV2ENC_MF_HD constexpr int mfkMiSizeLog2() { return 2; }
static_assert(mfkMiSizeLog2() == t::kMiSizeLog2);
AV2ENC_MF_HD constexpr int mfkCwpEqual() { return 8; }
static_assert(mfkCwpEqual() == t::kCwpEqual);
AV2ENC_MF_HD constexpr int mfkBankRefsPerFrame() { return 9; }
static_assert(mfkBankRefsPerFrame() == t::kBankRefsPerFrame);
AV2ENC_MF_HD constexpr int mfkLastInterRef() { return 6; }
static_assert(mfkLastInterRef() == t::kLastInterRef);
AV2ENC_MF_HD constexpr int mfkFirstInterRef() { return 0; }
static_assert(mfkFirstInterRef() == t::kFirstInterRef);

// ---------------------------------------------------------------------------
// §4.8 helpers (single implementation, shared with every other component).
// ---------------------------------------------------------------------------

AV2ENC_MF_HD int mf_abs(int value) { return inter::adv::abs_int(value); }
AV2ENC_MF_HD int mf_clip3(int low, int high, int value) {
  return inter::adv::clip3(low, high, value);
}
AV2ENC_MF_HD int mf_round2(int value, int n) {
  return inter::adv::round2(value, n);
}
AV2ENC_MF_HD int mf_round2_signed(int value, int n) {
  return inter::adv::round2_signed(value, n);
}
AV2ENC_MF_HD std::int64_t mf_round2_signed64(std::int64_t value, int n) {
  return inter::adv::round2_signed64(value, n);
}
AV2ENC_MF_HD int mf_get_relative_dist(int a, int b) {
  return inter::adv::get_relative_dist(a, b);
}
AV2ENC_MF_HD int mf_min(int a, int b) { return a < b ? a : b; }
AV2ENC_MF_HD int mf_max(int a, int b) { return a > b ? a : b; }

AV2ENC_MF_HD bool mf_is_inter_ref(int ref) {
  return ref >= mfkFirstInterRef() && ref <= mfkLastInterRef();
}

// GetMsb(x) = FloorLog2(x) for x > 0, 0 for x == 0.
AV2ENC_MF_HD int mf_get_msb(std::uint32_t value) {
  int result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}

// Flat index helpers (identical to motion_field.h).
AV2ENC_MF_HD std::size_t mf_index(int w8, int y8, int x8) {
  return static_cast<std::size_t>(y8) * static_cast<std::size_t>(w8) +
         static_cast<std::size_t>(x8);
}
AV2ENC_MF_HD std::size_t mf_traj_mv_index(int w8, int h8, int rf,
                                                 int y8, int x8, int comp) {
  return ((static_cast<std::size_t>(rf) * static_cast<std::size_t>(h8) +
           static_cast<std::size_t>(y8)) *
              static_cast<std::size_t>(w8) +
          static_cast<std::size_t>(x8)) *
             2 +
         static_cast<std::size_t>(comp);
}
AV2ENC_MF_HD std::size_t mf_traj_pos_index(int k, int num_refs, int w8,
                                                   int h8, int rf, int y8,
                                                   int x8, int comp) {
  return (((static_cast<std::size_t>(k) * static_cast<std::size_t>(num_refs) +
            static_cast<std::size_t>(rf)) *
               static_cast<std::size_t>(h8) +
           static_cast<std::size_t>(y8)) *
              static_cast<std::size_t>(w8) +
          static_cast<std::size_t>(x8)) *
             2 +
         static_cast<std::size_t>(comp);
}

// ---------------------------------------------------------------------------
// §7.9.4/.5 projection and §7.22 compression.
// ---------------------------------------------------------------------------

// §7.9.4 Div_Mult[32].  A function, not a namespace-scope array: NVCC cannot
// reference a host `constexpr` array from device code ("undefined in device
// code") when the index is not a compile-time constant.
AV2ENC_MF_HD int mf_div_mult(int index) {
  switch (index) {
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
    default: return 0;
  }
}

AV2ENC_MF_HD Mv mf_get_mv_projection(Mv mv, int numerator,
                                            int denominator) {
  const int clipped_denominator =
      denominator < 0 ? 0
                      : (denominator < mfkMaxFrameDistance()
                             ? denominator
                             : mfkMaxFrameDistance());
  const int clipped_numerator =
      mf_clip3(-mfkMaxFrameDistance(), mfkMaxFrameDistance(), numerator);
  const std::int64_t scale =
      static_cast<std::int64_t>(mf_div_mult(clipped_denominator));
  Mv out;
  out.row = static_cast<std::int32_t>(mf_clip3(
      mfkMvLow() + 1, mfkMvUpp() - 1,
      static_cast<int>(mf_round2_signed64(
          static_cast<std::int64_t>(mv.row) * clipped_numerator * scale, 14))));
  out.col = static_cast<std::int32_t>(mf_clip3(
      mfkMvLow() + 1, mfkMvUpp() - 1,
      static_cast<int>(mf_round2_signed64(
          static_cast<std::int64_t>(mv.col) * clipped_numerator * scale, 14))));
  return out;
}

AV2ENC_MF_HD Mv mf_get_mv_projection_clamp(Mv mv, int numerator,
                                                  int denominator) {
  Mv out = mf_get_mv_projection(mv, numerator, denominator);
  out.row = mf_clip3(-mfkRefMvsLimit(), mfkRefMvsLimit(), out.row);
  out.col = mf_clip3(-mfkRefMvsLimit(), mfkRefMvsLimit(), out.col);
  return out;
}

AV2ENC_MF_HD std::int16_t mf_compression_mv(std::int32_t v) {
  const std::int32_t a = v < 0 ? -v : v;
  const int msb = mf_get_msb(static_cast<std::uint32_t>(a));
  const int step_log2 = msb > 4 ? msb - 4 : 0;
  const std::int32_t c = (a >> step_log2) + (step_log2 << 4);
  return static_cast<std::int16_t>(v < 0 ? -c : c);
}

AV2ENC_MF_HD std::int32_t mf_uncompression_mv(std::int16_t v) {
  const std::int32_t c =
      v < 0 ? -static_cast<std::int32_t>(v) : static_cast<std::int32_t>(v);
  const int step_log2 = (c >> 4) > 1 ? (c >> 4) - 1 : 0;
  const std::int32_t a = (c - (step_log2 << 4)) << step_log2;
  return v < 0 ? -a : a;
}

// ---------------------------------------------------------------------------
// §7.9.2 TMVP units, §7.9.6 sampled position, §7.9.8 block position.
// ---------------------------------------------------------------------------

AV2ENC_MF_HD int mf_tmvp_shift(int sb_64x64, int proj_step) {
  return (sb_64x64 != 0 || proj_step == 1) ? 3 : 4;
}
AV2ENC_MF_HD int mf_tmvp_unit(int x8, int sb_64x64, int proj_step) {
  return x8 >> mf_tmvp_shift(sb_64x64, proj_step);
}
AV2ENC_MF_HD int mf_tmvp_phase(int x8, int sb_64x64, int proj_step) {
  return mf_tmvp_unit(x8, sb_64x64, proj_step) % 3;
}
AV2ENC_MF_HD bool mf_tmvp_avail(int base8, int loc8, int max8,
                                       int sb_64x64, int proj_step) {
  if (loc8 >= max8) {
    return false;
  }
  return mf_tmvp_unit(base8, sb_64x64, proj_step) ==
         mf_tmvp_unit(loc8, sb_64x64, proj_step);
}

struct MfSampledPosition {
  bool valid = false;
  int x8 = 0;
  int y8 = 0;
};

// §7.9.7 project_no_constraint.
AV2ENC_MF_HD int mf_project_no_constraint(int v8, std::int32_t delta,
                                                 int dst_sign, int max8,
                                                 bool* pos_valid) {
  std::int32_t offset8;
  if (delta >= 0) {
    offset8 = delta >> (3 + 1 + mfkMiSizeLog2());
  } else {
    offset8 = -((-delta) >> (3 + 1 + mfkMiSizeLog2()));
  }
  v8 += dst_sign * offset8;
  if (v8 < 0 || v8 >= max8) {
    *pos_valid = false;
  }
  return v8;
}

// §7.9.6 get_sampled_position.
AV2ENC_MF_HD MfSampledPosition mf_get_sampled_position(
    int x8, int y8, int dst_sign, Mv proj_mv, int mi_cols, int mi_rows,
    int proj_step) {
  MfSampledPosition out;
  out.valid = true;
  const int y = mf_project_no_constraint(y8, proj_mv.row, dst_sign,
                                         mi_rows >> 1, &out.valid);
  const int x = mf_project_no_constraint(x8, proj_mv.col, dst_sign,
                                         mi_cols >> 1, &out.valid);
  out.x8 = x;
  out.y8 = y;
  if (proj_step == 2) {
    out.x8 -= out.x8 & 1;
    out.y8 -= out.y8 & 1;
  }
  return out;
}

// §7.9.8 check_block_position.
AV2ENC_MF_HD bool mf_check_block_position(bool pos_valid, int base_x8,
                                                 int base_y8, int pos_x8,
                                                 int pos_y8, int sb_64x64,
                                                 int proj_step) {
  if (!pos_valid) {
    return false;
  }
  const int shift = mf_tmvp_shift(sb_64x64, proj_step);
  for (int ord = 0; ord < 2; ++ord) {
    const int v8 = ord != 0 ? base_x8 : base_y8;
    const int sb_off8 = 1 << shift;
    const int max_off8 = proj_step > 1
                             ? (ord != 0 ? sb_off8 : 0)
                             : (ord != 0 ? (sb_off8 >> 1) : 0);
    const int base8 = (v8 >> shift) << shift;
    const int pos8 = ord != 0 ? pos_x8 : pos_y8;
    if (pos8 < base8 - max_off8 || pos8 >= base8 + sb_off8 + max_off8) {
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// TIP weights / offsets and calc_avg.
// ---------------------------------------------------------------------------

// Tip_Weighting_Factor[8], again as a switch so device code never forms the
// address of a host constexpr array.
AV2ENC_MF_HD int mf_tip_weighting_factor(int index) {
  switch (index) {
    case 0: return 8;
    case 1: return 12;
    case 2: return 16;
    case 3: return 18;
    case 4: return 20;
    case 5: return 4;
    case 6: return 6;
    case 7: return -4;
    default: return 0;
  }
}

AV2ENC_MF_HD bool mf_tip_weight(int index, int* weight) {
  if (index < 0 || index > 7) {
    return false;
  }
  *weight = mf_tip_weighting_factor(index);
  return true;
}

struct MfTipOffsets {
  int ref_offset = 0;
  int past_offset = 0;
  int future_offset = 0;
};

AV2ENC_MF_HD MfTipOffsets mf_get_tip_offsets(
    int num_past_refs, int num_future_refs, int order_hint,
    int closest_past_order_hint, int closest_future_order_hint) {
  MfTipOffsets out;
  if (num_future_refs > 0 && num_past_refs > 0) {
    out.ref_offset = mf_get_relative_dist(closest_future_order_hint,
                                          closest_past_order_hint);
  } else {
    out.ref_offset = mf_get_relative_dist(closest_past_order_hint,
                                          closest_future_order_hint);
  }
  out.past_offset = mf_get_relative_dist(order_hint, closest_past_order_hint);
  out.future_offset =
      mf_get_relative_dist(order_hint, closest_future_order_hint);
  out.ref_offset =
      out.ref_offset < mfkMaxFrameDistance() ? out.ref_offset : mfkMaxFrameDistance();
  return out;
}

AV2ENC_MF_HD int mf_calc_avg(int n, int d) {
  if (d == 1) {
    return n;
  }
  if (d == 2) {
    return mf_round2_signed(n, 1);
  }
  if (d == 3) {
    return mf_round2_signed(n * 85, 8);
  }
  return mf_round2_signed(n, 2);
}

// ---------------------------------------------------------------------------
// §7.9.2 trajectory intersection (writes the trajectory arrays).
// ---------------------------------------------------------------------------

AV2ENC_MF_HD void mf_check_traj_intersect(
    MfProjectionParams& p, int src_frame, int x8, int y8, int end_frame,
    Mv mv) {
  MfTraj* traj = &p.traj;
  if (traj->mv == nullptr || traj->valid == nullptr || traj->pos == nullptr ||
      traj->pos_valid == nullptr || end_frame == mfkRefNone()) {
    return;
  }
  const int num_refs = traj->num_refs;
  const int w8 = p.mi_cols >> 1;
  const int h8 = p.mi_rows >> 1;
  for (int k = 0; k < 3; ++k) {
    const std::size_t pv_idx =
        ((static_cast<std::size_t>(k) * static_cast<std::size_t>(num_refs) +
          static_cast<std::size_t>(src_frame)) *
             static_cast<std::size_t>(h8) +
         static_cast<std::size_t>(y8)) *
            static_cast<std::size_t>(w8) +
        static_cast<std::size_t>(x8);
    if (traj->pos_valid[pv_idx] == 0) {
      continue;
    }
    const int traj_y8 = traj->pos[pv_idx * 2 + 0];
    const int traj_x8 = traj->pos[pv_idx * 2 + 1];
    const std::size_t end_idx =
        (static_cast<std::size_t>(end_frame) * static_cast<std::size_t>(h8) +
         static_cast<std::size_t>(traj_y8)) *
            static_cast<std::size_t>(w8) +
        static_cast<std::size_t>(traj_x8);
    if (traj->valid[end_idx] != 0) {
      continue;
    }
    for (int c = 0; c < 2; ++c) {
      const std::int32_t v =
          traj->mv[mf_traj_mv_index(w8, h8, src_frame, traj_y8, traj_x8, c)] +
          (c == 0 ? mv.row : mv.col);
      traj->mv[mf_traj_mv_index(w8, h8, end_frame, traj_y8, traj_x8, c)] =
          static_cast<std::int32_t>(
              mf_clip3(-mfkRefMvsLimit(), mfkRefMvsLimit(), static_cast<int>(v)));
    }
    traj->valid[end_idx] = 1;
    const MfSampledPosition pos = mf_get_sampled_position(
        traj_x8, traj_y8, 1,
        Mv{traj->mv[mf_traj_mv_index(w8, h8, end_frame, traj_y8, traj_x8, 0)],
           traj->mv[mf_traj_mv_index(w8, h8, end_frame, traj_y8, traj_x8, 1)]},
        p.mi_cols, p.mi_rows, p.proj_step);
    if (mf_check_block_position(pos.valid, traj_x8, traj_y8, pos.x8, pos.y8,
                                p.sb_64x64, p.proj_step)) {
      const std::size_t tp_idx = mf_traj_pos_index(
          k, num_refs, w8, h8, end_frame, pos.y8, pos.x8, 0);
      traj->pos[tp_idx] = static_cast<std::int16_t>(traj_y8);
      traj->pos[tp_idx + 1] = static_cast<std::int16_t>(traj_x8);
      traj->pos_valid[tp_idx / 2] = 1;
    }
  }

  const MfSampledPosition end = mf_get_sampled_position(
      x8, y8, 1, mv, p.mi_cols, p.mi_rows, p.proj_step);
  if (!end.valid) {
    return;
  }
  for (int k = 0; k < 3; ++k) {
    const std::size_t tp_idx =
        (((static_cast<std::size_t>(k) * static_cast<std::size_t>(num_refs) +
           static_cast<std::size_t>(end_frame)) *
              static_cast<std::size_t>(h8) +
          static_cast<std::size_t>(end.y8)) *
             static_cast<std::size_t>(w8) +
         static_cast<std::size_t>(end.x8)) *
        2;
    if (traj->pos_valid[tp_idx / 2] == 0) {
      continue;
    }
    const int traj_y8 = traj->pos[tp_idx + 0];
    const int traj_x8 = traj->pos[tp_idx + 1];
    if (!mf_check_block_position(true, traj_x8, traj_y8, x8, y8, p.sb_64x64,
                                 p.proj_step)) {
      continue;
    }
    const std::size_t src_idx =
        (static_cast<std::size_t>(src_frame) * static_cast<std::size_t>(h8) +
         static_cast<std::size_t>(traj_y8)) *
            static_cast<std::size_t>(w8) +
        static_cast<std::size_t>(traj_x8);
    if (traj->valid[src_idx] != 0) {
      continue;
    }
    for (int c = 0; c < 2; ++c) {
      const std::int32_t v =
          traj->mv[mf_traj_mv_index(w8, h8, end_frame, traj_y8, traj_x8, c)] -
          (c == 0 ? mv.row : mv.col);
      traj->mv[mf_traj_mv_index(w8, h8, src_frame, traj_y8, traj_x8, c)] =
          static_cast<std::int32_t>(
              mf_clip3(-mfkRefMvsLimit(), mfkRefMvsLimit(), static_cast<int>(v)));
    }
    traj->valid[src_idx] = 1;
    const MfSampledPosition pos = mf_get_sampled_position(
        traj_x8, traj_y8, 1,
        Mv{traj->mv[mf_traj_mv_index(w8, h8, src_frame, traj_y8, traj_x8, 0)],
           traj->mv[mf_traj_mv_index(w8, h8, src_frame, traj_y8, traj_x8, 1)]},
        p.mi_cols, p.mi_rows, p.proj_step);
    if (mf_check_block_position(pos.valid, traj_x8, traj_y8, pos.x8, pos.y8,
                                p.sb_64x64, p.proj_step)) {
      const std::size_t dst_idx = mf_traj_pos_index(
          k, num_refs, w8, h8, src_frame, pos.y8, pos.x8, 0);
      traj->pos[dst_idx] = static_cast<std::int16_t>(traj_y8);
      traj->pos[dst_idx + 1] = static_cast<std::int16_t>(traj_x8);
      traj->pos_valid[dst_idx / 2] = 1;
    }
  }
}

// ---------------------------------------------------------------------------
// §7.9 projection job: one full source-cell / trajectory sweep.
// ---------------------------------------------------------------------------

// Takes the parameters by value: the struct's pointers still address the
// caller's field/trajectory arrays, and the local copy keeps the trajectory
// writes race-free without const_cast.
AV2ENC_MF_HD void mf_project_job(MfProjectionParams p) {
  const int w8 = p.mi_cols >> 1;
  const int h8 = p.mi_rows >> 1;
  if (w8 <= 0 || h8 <= 0 || p.saved.mv == nullptr || p.saved.ref == nullptr ||
      p.saved.w8 != w8 || p.saved.h8 != h8 || p.field.mv == nullptr ||
      p.field.valid == nullptr || p.field.offset == nullptr) {
    return;
  }
  const int ref_to_cur = mf_get_relative_dist(p.src_order_hint, p.order_hint);

  int start_ref_map[mfkRefsPerFrame()];
  for (int k = 0; k < mfkRefsPerFrame(); ++k) {
    start_ref_map[k] = mfkRefNone();
  }
  for (int k = 0; k < p.num_saved; ++k) {
    const int saved_oh = p.saved_order_hint[k];
    for (int rf = 0; rf < p.num_total_refs; ++rf) {
      if (saved_oh == p.ref_order_hint[rf] &&
          p.ref_order_hint[rf] != p.src_order_hint &&
          !(saved_oh == mfkRestrictedOrderHint() ||
            p.ref_order_hint[rf] == mfkRestrictedOrderHint())) {
        start_ref_map[k] = rf;
        break;
      }
    }
  }

  const int list = p.is_bwd ? 1 : 0;
  for (int y8 = 0; y8 < h8; y8 += p.proj_step) {
    for (int x8 = 0; x8 < w8; x8 += p.proj_step) {
      const std::size_t pos = mf_index(w8, y8, x8);
      const std::int8_t src_ref =
          p.saved.ref[pos * 2 + static_cast<std::size_t>(list)];
      if (!mf_is_inter_ref(src_ref) || src_ref >= mfkRefsPerFrame()) {
        continue;
      }
      Mv mv2{
          mf_uncompression_mv(
              p.saved.mv[(pos * 2 + static_cast<std::size_t>(list)) * 2 + 0]),
          mf_uncompression_mv(
              p.saved.mv[(pos * 2 + static_cast<std::size_t>(list)) * 2 + 1])};
      const int mapped_ref = start_ref_map[static_cast<int>(src_ref)];
      if (p.enable_mv_traj != 0) {
        mf_check_traj_intersect(p, p.src, x8, y8, mapped_ref, mv2);
      }
      int ref_offset = mf_get_relative_dist(p.src_order_hint,
                                            p.saved_order_hint[src_ref]);
      if (p.saved_order_hint[src_ref] == mfkRestrictedOrderHint()) {
        ref_offset = 0;
      }
      bool pos_valid = mf_abs(ref_offset) <= mfkMaxFrameDistance();
      if (p.is_bwd != 0) {
        ref_offset = -ref_offset;
      }
      if (!pos_valid || ref_offset < 0) {
        continue;
      }
      Mv mv = mv2;
      const Mv proj_mv =
          mf_get_mv_projection(mv, ref_to_cur * p.dst_sign, ref_offset);
      const MfSampledPosition sampled =
          p.is_bwd != 0
              ? mf_get_sampled_position(x8, y8, 1, proj_mv, p.mi_cols,
                                        p.mi_rows, p.proj_step)
              : mf_get_sampled_position(x8, y8, p.dst_sign, proj_mv, p.mi_cols,
                                        p.mi_rows, p.proj_step);
      pos_valid = sampled.valid;
      pos_valid = mf_check_block_position(pos_valid, x8, y8, sampled.x8,
                                          sampled.y8, p.sb_64x64,
                                          p.proj_step);
      if (!pos_valid) {
        continue;
      }
      const std::size_t dst = mf_index(w8, sampled.y8, sampled.x8);
      const bool replace =
          p.field.valid[dst] == 0 ||
          (p.target_frame != mfkRefNone() &&
           p.target_frame == start_ref_map[static_cast<int>(src_ref)] &&
           p.field.offset[dst] != ref_offset);
      if (!replace) {
        continue;
      }
      if (p.enable_mv_traj != 0 && p.traj.mv != nullptr &&
          p.traj.valid != nullptr && p.traj.pos != nullptr &&
          p.traj.pos_valid != nullptr) {
        MfTraj* traj = &p.traj;
        const int phase = mf_tmvp_phase(sampled.x8, p.sb_64x64, p.proj_step);
        const std::size_t tp_idx = mf_traj_pos_index(
            phase, traj->num_refs, w8, h8, p.src, y8, x8, 0);
        traj->pos[tp_idx] = static_cast<std::int16_t>(sampled.y8);
        traj->pos[tp_idx + 1] = static_cast<std::int16_t>(sampled.x8);
        traj->pos_valid[tp_idx / 2] = 1;
        for (int c = 0; c < 2; ++c) {
          const std::int32_t v = c == 0 ? proj_mv.row : proj_mv.col;
          traj->mv[mf_traj_mv_index(w8, h8, p.src, sampled.y8, sampled.x8,
                                    c)] =
              static_cast<std::int32_t>(mf_clip3(
                  -mfkRefMvsLimit(), mfkRefMvsLimit(), -static_cast<int>(v)));
        }
        traj->valid[(static_cast<std::size_t>(p.src) *
                         static_cast<std::size_t>(h8) +
                     static_cast<std::size_t>(sampled.y8)) *
                        static_cast<std::size_t>(w8) +
                    static_cast<std::size_t>(sampled.x8)] = 1;
        const int end_frame = start_ref_map[static_cast<int>(src_ref)];
        if (end_frame != mfkRefNone()) {
          const Mv end_mv = mf_get_mv_projection(
              mv, ref_offset - ref_to_cur * p.dst_sign, ref_offset);
          for (int c = 0; c < 2; ++c) {
            const std::int32_t v = c == 0 ? end_mv.row : end_mv.col;
            traj->mv[mf_traj_mv_index(w8, h8, end_frame, sampled.y8,
                                      sampled.x8, c)] =
                static_cast<std::int32_t>(
                    mf_clip3(-mfkRefMvsLimit(), mfkRefMvsLimit(),
                             static_cast<int>(v)));
          }
          traj->valid[(static_cast<std::size_t>(end_frame) *
                           static_cast<std::size_t>(h8) +
                       static_cast<std::size_t>(sampled.y8)) *
                          static_cast<std::size_t>(w8) +
                      static_cast<std::size_t>(sampled.x8)] = 1;
          const MfSampledPosition target =
              mf_get_sampled_position(x8, y8, 1, mv, p.mi_cols, p.mi_rows,
                                      p.proj_step);
          if (mf_check_block_position(target.valid, sampled.x8, sampled.y8,
                                      target.x8, target.y8, p.sb_64x64,
                                      p.proj_step)) {
            const std::size_t tp_idx2 = mf_traj_pos_index(
                phase, traj->num_refs, w8, h8, end_frame, target.y8,
                target.x8, 0);
            traj->pos[tp_idx2] = static_cast<std::int16_t>(sampled.y8);
            traj->pos[tp_idx2 + 1] = static_cast<std::int16_t>(sampled.x8);
            traj->pos_valid[tp_idx2 / 2] = 1;
          }
        }
      }
      if (p.is_bwd != 0) {
        mv.row = -mv.row;
        mv.col = -mv.col;
      }
      p.field.valid[dst] = 1;
      p.field.mv[dst * 2 + 0] = mv.row;
      p.field.mv[dst * 2 + 1] = mv.col;
      p.field.offset[dst] = ref_offset;
    }
  }
}

// ---------------------------------------------------------------------------
// §7.9 trajectory gap fill (fill_traj_mv), one anchor cell.
// ---------------------------------------------------------------------------

AV2ENC_MF_HD void mf_traj_fill_cell(MfTraj* traj, int rf, int y8,
                                           int x8, int w8, int h8, int sb_64x64,
                                           int proj_step) {
  if (traj == nullptr || traj->mv == nullptr || traj->valid == nullptr ||
      proj_step != 2) {
    return;
  }
  for (int pass = 0; pass < 3; ++pass) {
    const int dy = pass == 0 ? 0 : 1;
    const int dx = pass == 0 ? 1 : (pass == 1 ? 0 : 1);
    if (y8 + dy == h8 || x8 + dx == w8) {
      continue;
    }
    const std::size_t cur =
        (static_cast<std::size_t>(rf) * static_cast<std::size_t>(h8) +
         static_cast<std::size_t>(y8)) *
            static_cast<std::size_t>(w8) +
        static_cast<std::size_t>(x8);
    if (traj->valid[cur] == 0) {
      continue;
    }
    int count = 1;
    int avg[2] = {
        traj->mv[mf_traj_mv_index(w8, h8, rf, y8, x8, 0)],
        traj->mv[mf_traj_mv_index(w8, h8, rf, y8, x8, 1)]};
    const bool r_avail =
        dx > 0 &&
        mf_tmvp_avail(x8, x8 + 2, w8, sb_64x64, proj_step) &&
        traj->valid[(static_cast<std::size_t>(rf) *
                         static_cast<std::size_t>(h8) +
                     static_cast<std::size_t>(y8)) *
                        static_cast<std::size_t>(w8) +
                    static_cast<std::size_t>(x8 + 2)] != 0;
    const bool b_avail =
        dy > 0 &&
        mf_tmvp_avail(y8, y8 + 2, h8, sb_64x64, proj_step) &&
        traj->valid[(static_cast<std::size_t>(rf) *
                         static_cast<std::size_t>(h8) +
                     static_cast<std::size_t>(y8 + 2)) *
                        static_cast<std::size_t>(w8) +
                    static_cast<std::size_t>(x8)] != 0;
    const bool br_avail =
        dy > 0 && dx > 0 &&
        mf_tmvp_avail(x8, x8 + 2, w8, sb_64x64, proj_step) &&
        mf_tmvp_avail(y8, y8 + 2, h8, sb_64x64, proj_step) &&
        traj->valid[(static_cast<std::size_t>(rf) *
                         static_cast<std::size_t>(h8) +
                     static_cast<std::size_t>(y8 + 2)) *
                        static_cast<std::size_t>(w8) +
                    static_cast<std::size_t>(x8 + 2)] != 0;
    if (r_avail) {
      ++count;
      avg[0] += traj->mv[mf_traj_mv_index(w8, h8, rf, y8, x8 + 2, 0)];
      avg[1] += traj->mv[mf_traj_mv_index(w8, h8, rf, y8, x8 + 2, 1)];
    }
    if (b_avail) {
      ++count;
      avg[0] += traj->mv[mf_traj_mv_index(w8, h8, rf, y8 + 2, x8, 0)];
      avg[1] += traj->mv[mf_traj_mv_index(w8, h8, rf, y8 + 2, x8, 1)];
    }
    if (br_avail) {
      ++count;
      avg[0] += traj->mv[mf_traj_mv_index(w8, h8, rf, y8 + 2, x8 + 2, 0)];
      avg[1] += traj->mv[mf_traj_mv_index(w8, h8, rf, y8 + 2, x8 + 2, 1)];
    }
    for (int c = 0; c < 2; ++c) {
      traj->mv[mf_traj_mv_index(w8, h8, rf, y8 + dy, x8 + dx, c)] =
          static_cast<std::int32_t>(mf_calc_avg(avg[c], count));
    }
    traj->valid[(static_cast<std::size_t>(rf) *
                     static_cast<std::size_t>(h8) +
                 static_cast<std::size_t>(y8 + dy)) *
                    static_cast<std::size_t>(w8) +
                static_cast<std::size_t>(x8 + dx)] = 1;
  }
}

// ---------------------------------------------------------------------------
// §7.12.2.8 get_motion_field_mv.
// ---------------------------------------------------------------------------

AV2ENC_MF_HD Mv mf_get_motion_field_mv(const MfField& field,
                                              const MfTraj* traj,
                                              int dst_ref, int order_hint,
                                              int dst_order_hint,
                                              bool use_traj, int y8, int x8) {
  const std::size_t pos = mf_index(field.w8, y8, x8);
  if (use_traj && traj != nullptr && dst_ref >= 0 &&
      dst_ref < traj->num_refs && traj->valid != nullptr &&
      traj->mv != nullptr) {
    const std::size_t idx =
        (static_cast<std::size_t>(dst_ref) * static_cast<std::size_t>(field.h8) +
         static_cast<std::size_t>(y8)) *
            static_cast<std::size_t>(field.w8) +
        static_cast<std::size_t>(x8);
    if (traj->valid[idx] != 0) {
      return Mv{traj->mv[mf_traj_mv_index(field.w8, field.h8, dst_ref, y8, x8,
                                          0)],
                traj->mv[mf_traj_mv_index(field.w8, field.h8, dst_ref, y8, x8,
                                          1)]};
    }
  }
  const Mv mv{field.mv[pos * 2 + 0], field.mv[pos * 2 + 1]};
  const int ref_offset = field.offset[pos];
  const int ref_to_dst = mf_get_relative_dist(order_hint, dst_order_hint);
  return mf_get_mv_projection(mv, ref_to_dst, ref_offset);
}

// ---------------------------------------------------------------------------
// §7.12.2.7/.8 temporal scan.
// ---------------------------------------------------------------------------

struct MfScanInput {
  int mi_row = 0;
  int mi_col = 0;
  int mi_cols = 0;
  int mi_rows = 0;
  int tile_mi_col_start = 0;
  int tile_mi_col_end = 0;
  int tile_mi_row_start = 0;
  int tile_mi_row_end = 0;
  int bw4 = 1;
  int bh4 = 1;
  int ref0 = mfkRefNone();
  int ref1 = mfkRefNone();
  int ref0_order_hint = mfkRestrictedOrderHint();
  int ref1_order_hint = mfkRestrictedOrderHint();
  int order_hint = 0;
  int is_compound = 0;
  int use_traj = 0;
  const MfField* field = nullptr;
  const MfTraj* traj = nullptr;
};

AV2ENC_MF_HD void mf_temporal_sample(const MfScanInput& in,
                                            int delta_row, int delta_col,
                                            StackMv* stack, int* num_mv_found,
                                            int* prune_count) {
  if (*num_mv_found >= mfkMaxRefMvStackSize()) {
    return;
  }
  const int mv_row = in.mi_row + delta_row;
  const int mv_col = in.mi_col + delta_col;
  const int row_start =
      in.tile_mi_row_end > in.tile_mi_row_start ? in.tile_mi_row_start : 0;
  const int row_end = in.tile_mi_row_end > in.tile_mi_row_start
                          ? in.tile_mi_row_end
                          : in.mi_rows;
  const int col_start =
      in.tile_mi_col_end > in.tile_mi_col_start ? in.tile_mi_col_start : 0;
  const int col_end = in.tile_mi_col_end > in.tile_mi_col_start
                          ? in.tile_mi_col_end
                          : in.mi_cols;
  if (mv_row < row_start || mv_row >= row_end || mv_col < col_start ||
      mv_col >= col_end) {
    return;
  }
  const int x8 = mv_col >> 1;
  const int y8 = mv_row >> 1;
  // Odd MI extents have a trailing 4x4 row/column with no 8x8 field cell
  // (the field is MiRows>>1 x MiCols>>1).  Sampling there must be a no-op;
  // otherwise the device reads past the field allocation.
  if (x8 < 0 || x8 >= in.field->w8 || y8 < 0 || y8 >= in.field->h8) {
    return;
  }
  const std::size_t pos = mf_index(in.field->w8, y8, x8);
  if (in.field->valid[pos] == 0) {
    return;
  }
  if (in.is_compound == 0) {
    const Mv cand = mf_get_motion_field_mv(*in.field, in.traj, in.ref0,
                                           in.order_hint, in.ref0_order_hint,
                                           in.use_traj != 0, y8, x8);
    int idx = *num_mv_found;
    if (*prune_count >= mfkMaxPrNum()) {
      idx = *num_mv_found;
    } else {
      for (int i = 0; i < *num_mv_found; ++i) {
        ++(*prune_count);
        if (cand == stack[i].mv[0]) {
          idx = i;
          break;
        }
      }
    }
    const int weight = mf_abs(mf_get_relative_dist(in.order_hint,
                                                   in.ref0_order_hint)) <= 2
                           ? 2
                           : 1;
    if (idx < *num_mv_found) {
      stack[idx].weight += weight;
    } else {
      stack[*num_mv_found].mv[0] = cand;
      stack[*num_mv_found].mv[1] = cand;
      stack[*num_mv_found].weight = weight;
      stack[*num_mv_found].cwp = mfkCwpEqual();
      ++(*num_mv_found);
    }
  } else {
    const Mv cand0 = mf_get_motion_field_mv(*in.field, in.traj, in.ref0,
                                            in.order_hint, in.ref0_order_hint,
                                            in.use_traj != 0, y8, x8);
    const Mv cand1 = mf_get_motion_field_mv(*in.field, in.traj, in.ref1,
                                            in.order_hint, in.ref1_order_hint,
                                            in.use_traj != 0, y8, x8);
    int idx = *num_mv_found;
    if (*prune_count >= mfkMaxPrNum()) {
      idx = *num_mv_found;
    } else {
      for (int i = 0; i < *num_mv_found; ++i) {
        ++(*prune_count);
        if (cand0 == stack[i].mv[0] && cand1 == stack[i].mv[1]) {
          idx = i;
          break;
        }
      }
    }
    if (idx < *num_mv_found) {
      stack[idx].weight += 1;
    } else {
      stack[*num_mv_found].mv[0] = cand0;
      stack[*num_mv_found].mv[1] = cand1;
      stack[*num_mv_found].weight = 1;
      stack[*num_mv_found].cwp = mfkCwpEqual();
      ++(*num_mv_found);
    }
  }
}

AV2ENC_MF_HD void mf_temporal_scan(const MfScanInput& in, StackMv* stack,
                                          int* num_mv_found,
                                          int* prune_count) {
  if (in.field == nullptr || stack == nullptr || num_mv_found == nullptr ||
      prune_count == nullptr || in.bw4 <= 0 || in.bh4 <= 0) {
    return;
  }
  const int step_w4 = in.bw4 >= 16 ? 4 : 2;
  const int step_h4 = in.bh4 >= 16 ? 4 : 2;
  const int row_end = mf_min(in.bh4, 16);
  const int col_end = mf_min(in.bw4, 16);
  const int delta_row = row_end - step_h4;
  const int delta_col = col_end - step_w4;
  const int start_found = *num_mv_found;
  if (delta_row >= 0 && delta_col >= 0) {
    mf_temporal_sample(in, delta_row, delta_col, stack, num_mv_found,
                       prune_count);
  }
  if ((row_end >= 3 * step_h4 || col_end >= 3 * step_w4) &&
      start_found == *num_mv_found) {
    mf_temporal_sample(in, row_end >> 1, col_end >> 1, stack, num_mv_found,
                       prune_count);
  }
}

// ---------------------------------------------------------------------------
// §7.22 get_tip_cand.
// ---------------------------------------------------------------------------

struct MfTipCandInput {
  int cand_row = 0;  // MI units
  int cand_col = 0;
  int base_row = 0;
  int base_col = 0;
  int is_16x16 = 0;
  Mv base_mv{};
  const MfField* field = nullptr;
  int num_past_refs = 0;
  int num_future_refs = 0;
  int order_hint = 0;
  int closest_past_order_hint = mfkRestrictedOrderHint();
  int closest_future_order_hint = mfkRestrictedOrderHint();
};

AV2ENC_MF_HD MvPair mf_get_tip_cand(const MfTipCandInput& in) {
  MvPair out;
  if (in.field == nullptr || in.field->w8 <= 0 || in.field->h8 <= 0 ||
      in.field->mv == nullptr || in.field->valid == nullptr) {
    out.mv[0] = in.base_mv;
    out.mv[1] = in.base_mv;
    return out;
  }
  const int base_row = in.base_row;
  const int base_col = in.base_col;
  const int shift = in.is_16x16 != 0 ? 2 : 1;
  const int cand_row =
      base_row + (((in.cand_row - base_row) >> shift) << shift);
  const int cand_col =
      base_col + (((in.cand_col - base_col) >> shift) << shift);
  const int x8 = cand_col >> 1;
  const int y8 = cand_row >> 1;
  out.mv[0] = Mv{0, 0};
  out.mv[1] = Mv{0, 0};
  const int ref_x8 = mf_clip3(0, in.field->w8 - 1, x8);
  const int ref_y8 = mf_clip3(0, in.field->h8 - 1, y8);
  const std::size_t pos = mf_index(in.field->w8, ref_y8, ref_x8);
  if (in.field->valid[pos] != 0) {
    const MfTipOffsets offsets = mf_get_tip_offsets(
        in.num_past_refs, in.num_future_refs, in.order_hint,
        in.closest_past_order_hint, in.closest_future_order_hint);
    const Mv field_mv{in.field->mv[pos * 2 + 0], in.field->mv[pos * 2 + 1]};
    out.mv[0] =
        mf_get_mv_projection(field_mv, offsets.past_offset, offsets.ref_offset);
    out.mv[1] = mf_get_mv_projection(field_mv, offsets.future_offset,
                                     offsets.ref_offset);
  }
  for (int list = 0; list < 2; ++list) {
    out.mv[list].row = mf_clip3(mfkMvLow() + 1, mfkMvUpp() - 1,
                                out.mv[list].row + in.base_mv.row);
    out.mv[list].col = mf_clip3(mfkMvLow() + 1, mfkMvUpp() - 1,
                                out.mv[list].col + in.base_mv.col);
  }
  return out;
}

// ---------------------------------------------------------------------------
// §7.22 storage entry (ordering, wedge allow-list, compression).
// ---------------------------------------------------------------------------

struct MfStoreEntry {
  int refs[2] = {mfkRefNone(), mfkRefNone()};
  Mv mv[2];
  bool is_wedge = false;
  const std::uint8_t* mask = nullptr;
  int mask_stride = 0;
  int i8 = 0;
  int j8 = 0;
  int order_hint = 0;
  int ref_order_hint[2] = {mfkRestrictedOrderHint(), mfkRestrictedOrderHint()};
};

AV2ENC_MF_HD void mf_store_motion_field_entry(
    const MfStoreEntry& in, std::int8_t* out_ref, std::int16_t* out_mv, int w8,
    int h8, int y8, int x8) {
  // Same contract as the CPU helper: i8/j8 are the 8x8 sub-block index inside
  // a block, so they must be in [0, 7] (the wedge mask is 64x64).  Blocks whose
  // nominal size exceeds 16 MI are outside the §7.22/§9.2 block table; the CPU
  // entry helper rejects them and the GPU must too.
  if (out_ref == nullptr || out_mv == nullptr || y8 < 0 || x8 < 0 ||
      y8 >= h8 || x8 >= w8 || in.i8 < 0 || in.i8 > 7 || in.j8 < 0 ||
      in.j8 > 7) {
    return;
  }
  const std::size_t pos = mf_index(w8, y8, x8);
  int refs[2] = {in.refs[0], in.refs[1]};
  Mv mvs[2] = {in.mv[0], in.mv[1]};
  std::uint8_t allow[2] = {1, 1};
  if (in.is_wedge && in.mask != nullptr) {
    int count0 = 0;
    int count1 = 0;
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 8; ++j) {
        const int m =
            in.mask[static_cast<std::size_t>(in.i8 * 8 + i) *
                        static_cast<std::size_t>(in.mask_stride) +
                    static_cast<std::size_t>(in.j8 * 8 + j)];
        if (m > 60) {
          ++count0;
        }
        if (m < 4) {
          ++count1;
        }
      }
    }
    if (count0 >= 60) {
      allow[1] = 0;
    } else if (count1 >= 60) {
      allow[0] = 0;
    }
  }

  int final_refs[2] = {mfkRefNone(), mfkRefNone()};
  Mv final_mv[2];
  final_mv[0] = Mv{0, 0};
  final_mv[1] = Mv{0, 0};
  for (int list = 0; list < 2; ++list) {
    if (!mf_is_inter_ref(refs[list])) {
      continue;
    }
    if (mf_abs(mvs[list].row) <= mfkRefMvsLimit() &&
        mf_abs(mvs[list].col) <= mfkRefMvsLimit() && allow[list] != 0) {
      final_refs[list] = refs[list];
      final_mv[list] = mvs[list];
    }
  }
  if (final_refs[0] != mfkRefNone() && final_refs[1] == mfkRefNone()) {
    final_refs[1] = final_refs[0];
    final_mv[1] = final_mv[0];
  } else if (final_refs[1] != mfkRefNone() && final_refs[0] == mfkRefNone()) {
    final_refs[0] = final_refs[1];
    final_mv[0] = final_mv[1];
  } else if (final_refs[0] != mfkRefNone() && final_refs[1] != mfkRefNone()) {
    const int rel0 = mf_get_relative_dist(in.ref_order_hint[0], in.order_hint);
    const int rel1 = mf_get_relative_dist(in.ref_order_hint[1], in.order_hint);
    bool to_switch;
    if (rel0 < 0 && rel1 < 0) {
      to_switch =
          mf_get_relative_dist(in.ref_order_hint[0], in.ref_order_hint[1]) < 0;
    } else if (rel0 > 0 && rel1 > 0) {
      to_switch =
          mf_get_relative_dist(in.ref_order_hint[0], in.ref_order_hint[1]) < 0;
    } else {
      to_switch = rel0 > 0 && rel1 < 0;
    }
    if (to_switch) {
      const int tr = final_refs[0];
      final_refs[0] = final_refs[1];
      final_refs[1] = tr;
      const Mv tm = final_mv[0];
      final_mv[0] = final_mv[1];
      final_mv[1] = tm;
    }
  }
  for (int list = 0; list < 2; ++list) {
    out_ref[pos * 2 + static_cast<std::size_t>(list)] =
        static_cast<std::int8_t>(final_refs[list]);
    out_mv[(pos * 2 + static_cast<std::size_t>(list)) * 2 + 0] =
        mf_compression_mv(final_mv[list].row);
    out_mv[(pos * 2 + static_cast<std::size_t>(list)) * 2 + 1] =
        mf_compression_mv(final_mv[list].col);
  }
}

// Clipped block geometry shared by the §7.22 writer, the ordering-stamp pass
// and the host-emulation oracle.  Returns false for an empty block.
AV2ENC_MF_HD bool mf_block_geometry(const MfBlockStoreInput& in, int* w8,
                                    int* h8, int* bw4_out, int* bh4_out) {
  const int cols = in.mi_cols >> 1;
  const int rows = in.mi_rows >> 1;
  int bw4 = mf_min(in.mi_cols - in.c, in.bw4);
  int bh4 = mf_min(in.mi_rows - in.r, in.bh4);
  if (cols <= 0 || rows <= 0 || bw4 <= 0 || bh4 <= 0) {
    return false;
  }
  *w8 = cols;
  *h8 = rows;
  *bw4_out = bw4;
  *bh4_out = bh4;
  return true;
}

// §7.22 ordering support.  A 4x4 block can share an 8x8 cell with an adjacent
// odd-origin 4x4 block, so a parallel "one thread per block" writer would let
// threads race on the same cell.  The GPU path runs a stamp pass first
// (`atomicMax` of the descriptor index per cell); each block then writes only
// the cells whose stamp equals its own index, i.e. the last descriptor in
// array order wins, exactly like the CPU driver's call order.  Host emulation
// calls this helper directly (plain max) to build the same map.
AV2ENC_MF_HD void mf_mark_block_cells(const MfBlockStoreInput& in,
                                      int* stamp, int index) {
  int w8 = 0;
  int h8 = 0;
  int bw4 = 0;
  int bh4 = 0;
  if (stamp == nullptr || in.enable_ref_frame_mvs == 0 ||
      !mf_block_geometry(in, &w8, &h8, &bw4, &bh4)) {
    return;
  }
  const int rows = (bh4 + 1) >> 1;
  const int cols = (bw4 + 1) >> 1;
  for (int i8 = 0; i8 < rows; ++i8) {
    const int y8 = (in.r >> 1) + i8;
    if (y8 < 0 || y8 >= h8 || i8 > 7) {
      continue;
    }
    for (int j8 = 0; j8 < cols; ++j8) {
      const int x8 = (in.c >> 1) + j8;
      // The entry helper rejects i8/j8 outside [0, 7]; those cells must not
      // acquire a stamp, or the block would shadow the previous real writer.
      if (x8 < 0 || x8 >= w8 || j8 > 7) {
        continue;
      }
      const std::size_t cell = mf_index(w8, y8, x8);
      if (stamp[cell] < index) {
        stamp[cell] = index;
      }
    }
  }
}

// §7.22 block-to-subsampled-grid writes, mvMethod 0/1.  When `cell_stamp` is
// non-null, only cells whose last-writer stamp equals `block_index` are
// written; that is the GPU ordering pass.  Host code passes the defaults.
AV2ENC_MF_HD void mf_store_block(const MfBlockStoreInput& in,
                                        std::int8_t* out_ref,
                                        std::int16_t* out_mv,
                                        const int* cell_stamp = nullptr,
                                        int block_index = -1) {
  if (in.enable_ref_frame_mvs == 0 || out_ref == nullptr || out_mv == nullptr) {
    return;
  }
  int w8 = 0;
  int h8 = 0;
  int bw4 = 0;
  int bh4 = 0;
  if (!mf_block_geometry(in, &w8, &h8, &bw4, &bh4)) {
    return;
  }
  // n is derived from the nominal block size, before the edge clip.
  const int n =
      (in.bw4 <= 2 && in.bh4 <= 2 && in.tip_frame_as_output == 0) ? 4 : 8;

  int ref0 = in.refs[0];
  int ref1 = in.refs[1];
  const bool is_wedge = mf_is_inter_ref(ref0) && mf_is_inter_ref(ref1) &&
                        ref0 != mfkTipFrame() && in.is_wedge != 0;
  const bool tip_pred = ref0 == mfkTipFrame();
  int tip_weight = 0;
  const bool have_tip_weight =
      mf_tip_weight(in.tip_global_wtd_index, &tip_weight);
  if (tip_pred) {
    ref0 = in.closest_past;
    ref1 = in.closest_future;
  }
  if ((tip_pred || in.tip_frame_as_output != 0) && have_tip_weight &&
      tip_weight == 16) {
    ref1 = mfkRefNone();
  }
  int ref_oh[2] = {in.ref_order_hint[0], in.ref_order_hint[1]};
  if (tip_pred) {
    ref_oh[0] = in.tip_past_order_hint;
    ref_oh[1] = in.tip_future_order_hint;
  }

  const int rows = (bh4 + 1) >> 1;  // Round2(bh4, 1)
  const int cols = (bw4 + 1) >> 1;
  for (int i8 = 0; i8 < rows; ++i8) {
    for (int j8 = 0; j8 < cols; ++j8) {
      const int x8 = (in.c >> 1) + j8;
      const int y8 = (in.r >> 1) + i8;
      // Odd MI origins can map a 4x4 block past the last 8x8 cell row/column
      // (the CPU entry helper rejects those cells too).
      if (x8 < 0 || x8 >= w8 || y8 < 0 || y8 >= h8) {
        continue;
      }
      const int row = in.r + (i8 << 1);
      const int col = in.c + (j8 << 1);
      int refs[2] = {mfkRefNone(), mfkRefNone()};
      Mv mfmvs[2];
      mfmvs[0] = Mv{0, 0};
      mfmvs[1] = Mv{0, 0};
      MvPair tip_cand;
      tip_cand.mv[0] = Mv{0, 0};
      tip_cand.mv[1] = Mv{0, 0};
      if (tip_pred) {
        MfTipCandInput tc;
        tc.cand_row = row;
        tc.cand_col = col;
        const std::size_t mi_idx =
            static_cast<std::size_t>(row) *
                static_cast<std::size_t>(in.mi_cols) +
            static_cast<std::size_t>(col);
        tc.base_row = in.mi_row_base != nullptr ? in.mi_row_base[mi_idx] : row;
        tc.base_col = in.mi_col_base != nullptr ? in.mi_col_base[mi_idx] : col;
        tc.is_16x16 = in.tip_sizes16 != nullptr ? in.tip_sizes16[mi_idx] : 0;
        const std::size_t mvs_idx =
            (static_cast<std::size_t>(row) *
                 static_cast<std::size_t>(in.mi_cols) +
             static_cast<std::size_t>(col)) *
                4;
        tc.base_mv = Mv{in.src.mvs != nullptr
                            ? in.src.mvs[mvs_idx + 0]
                            : 0,
                        in.src.mvs != nullptr ? in.src.mvs[mvs_idx + 1] : 0};
        tc.field = in.field.mv != nullptr ? &in.field : nullptr;
        tc.num_past_refs = in.num_past_refs;
        tc.num_future_refs = in.num_future_refs;
        tc.order_hint = in.order_hint;
        tc.closest_past_order_hint = in.tip_past_order_hint;
        tc.closest_future_order_hint = in.tip_future_order_hint;
        tip_cand = mf_get_tip_cand(tc);
      }
      for (int list = 0; list < 2; ++list) {
        const int ref_idx = list == 0 ? ref0 : ref1;
        if (!mf_is_inter_ref(ref_idx)) {
          continue;
        }
        int mv_row = 0;
        int mv_col = 0;
        if (in.mv_method > 0) {
          const bool use_refine = in.src.use_refinemv != 0 || tip_pred;
          const std::int32_t* plane =
              use_refine ? in.src.refine_mvs : in.src.mvs;
          if (plane == nullptr) {
            continue;
          }
          if (use_refine) {
            const std::size_t idx =
                ((static_cast<std::size_t>(i8 << 1) *
                      static_cast<std::size_t>(in.src.delta_stride4) +
                  static_cast<std::size_t>(j8 << 1)) *
                     4 +
                 static_cast<std::size_t>(list) * 2);
            mv_row = plane[idx + 0];
            mv_col = plane[idx + 1];
          } else {
            const std::size_t idx =
                (static_cast<std::size_t>(row) *
                     static_cast<std::size_t>(in.mi_cols) +
                 static_cast<std::size_t>(col)) *
                    4 +
                static_cast<std::size_t>(list) * 2;
            mv_row = plane[idx + 0];
            mv_col = plane[idx + 1];
          }
          if (in.mv_method == 1) {
            if (in.src.mv_deltas == nullptr) {
              continue;
            }
            if (n == 4 && !tip_pred) {
              int total_row = 0;
              int total_col = 0;
              for (int a = 0; a < 2; ++a) {
                for (int b = 0; b < 2; ++b) {
                  const std::size_t idx =
                      ((static_cast<std::size_t>(a) *
                            static_cast<std::size_t>(in.src.delta_stride4) +
                        static_cast<std::size_t>(b)) *
                           4 +
                       static_cast<std::size_t>(list) * 2);
                  total_row += in.src.mv_deltas[idx + 0];
                  total_col += in.src.mv_deltas[idx + 1];
                }
              }
              mv_row += mf_round2_signed(total_row, 1 + 2);
              mv_col += mf_round2_signed(total_col, 1 + 2);
            } else {
              const std::size_t idx =
                  ((static_cast<std::size_t>(i8 << 1) *
                        static_cast<std::size_t>(in.src.delta_stride4) +
                    static_cast<std::size_t>(j8 << 1)) *
                       4 +
                   static_cast<std::size_t>(list) * 2);
              mv_row += mf_round2_signed(in.src.mv_deltas[idx + 0], 1);
              mv_col += mf_round2_signed(in.src.mv_deltas[idx + 1], 1);
            }
          }
        } else {
          if (tip_pred) {
            mv_row = tip_cand.mv[list].row;
            mv_col = tip_cand.mv[list].col;
          } else if (in.src.motion_mode >= 2 &&
                     in.src.force_integer_mv == 0) {
            mv_row = in.warp_mv[list].row;
            mv_col = in.warp_mv[list].col;
          } else if (in.src.is_global_mv_cand != 0 &&
                     in.src.force_integer_mv == 0) {
            mv_row = in.global_mv[list].row;
            mv_col = in.global_mv[list].col;
          } else {
            if (in.src.mvs == nullptr) {
              continue;
            }
            const std::size_t idx =
                (static_cast<std::size_t>(row) *
                     static_cast<std::size_t>(in.mi_cols) +
                 static_cast<std::size_t>(col)) *
                    4 +
                static_cast<std::size_t>(list) * 2;
            mv_row = in.src.mvs[idx + 0];
            mv_col = in.src.mvs[idx + 1];
          }
        }
        if (mf_abs(mv_row) <= mfkRefMvsLimit() &&
            mf_abs(mv_col) <= mfkRefMvsLimit()) {
          refs[list] = ref_idx;
          mfmvs[list] = Mv{mv_row, mv_col};
        }
      }

      if (cell_stamp != nullptr &&
          cell_stamp[mf_index(w8, y8, x8)] != block_index) {
        continue;
      }
      MfStoreEntry entry;
      entry.refs[0] = refs[0];
      entry.refs[1] = refs[1];
      entry.mv[0] = mfmvs[0];
      entry.mv[1] = mfmvs[1];
      entry.is_wedge = is_wedge;
      entry.mask = in.mask;
      entry.mask_stride = in.mask_stride;
      entry.i8 = i8;
      entry.j8 = j8;
      entry.order_hint = in.order_hint;
      entry.ref_order_hint[0] = ref_oh[0];
      entry.ref_order_hint[1] = ref_oh[1];
      mf_store_motion_field_entry(entry, out_ref, out_mv, w8, h8, y8, x8);
    }
  }
}

// ---------------------------------------------------------------------------
// §7.12.2.21 / §5.20.2.2 reference-MV bank.
// ---------------------------------------------------------------------------

AV2ENC_MF_HD int mf_get_rmb_list_index(const int ref_frames[2]) {
  if (!mf_is_inter_ref(ref_frames[0])) {
    return 8;
  }
  if (!mf_is_inter_ref(ref_frames[1]) && ref_frames[0] <= 5) {
    return ref_frames[0];
  }
  if (ref_frames[0] == 0 && ref_frames[1] == 0) {
    return 6;
  }
  if (ref_frames[0] == 0 && ref_frames[1] == 1) {
    return 7;
  }
  return 8;
}

AV2ENC_MF_HD bool mf_rmb_params_equal(const RefMvBankEntry& a,
                                             const RefMvBankEntry& b) {
  for (int i = 1; i < 6; ++i) {
    if (a.p[i] != b.p[i]) {
      return false;
    }
  }
  return true;
}

AV2ENC_MF_HD void mf_update_ref_mv_count(RefMvBank* bank, int mi_row,
                                                int mi_col, int bw4, int bh4,
                                                int sb_size4,
                                                int chroma_part) {
  if (bank == nullptr || chroma_part != 0 || sb_size4 <= 0) {
    return;
  }
  int unit_size4 = sb_size4 >> 3;
  if (unit_size4 < 1) {
    unit_size4 = 1;
  }
  const int width_units = mf_max(bw4 / unit_size4, 1);
  const int height_units = mf_max(bh4 / unit_size4, 1);
  const int unit_count = width_units * height_units;
  if (mi_row % sb_size4 == 0 && mi_col % sb_size4 == 0) {
    bank->remain_hits = mf_max(unit_count, 4);
    bank->unit_hits = 0;
  } else if (mi_row % unit_size4 == 0 && mi_col % unit_size4 == 0) {
    bank->remain_hits += unit_count;
    bank->unit_hits = 0;
  }
}

AV2ENC_MF_HD bool mf_update_ref_mv_bank(
    RefMvBank* bank, int mi_row, int mi_col, int bw4, int bh4, int sb_size4,
    int chroma_part, const int ref_frames[2], const Mv mvs[2], int cwp_idx,
    bool from_within_sb) {
  if (bank == nullptr) {
    return false;
  }
  if (from_within_sb) {
    mf_update_ref_mv_count(bank, mi_row, mi_col, bw4, bh4, sb_size4,
                           chroma_part);
    if (bank->remain_hits == 0 || bank->unit_hits >= 16) {
      return false;
    }
    --bank->remain_hits;
    ++bank->unit_hits;
  }
  ++bank->bank_hits;

  const int r0 = ref_frames[0];
  const int r1 = ref_frames[1];
  if (!mf_is_inter_ref(r0)) {
    return false;
  }
  const bool is_compound = mf_is_inter_ref(r1);
  const int ref = mf_get_rmb_list_index(ref_frames);
  RefMvBankEntry p;
  p.p[0] = cwp_idx;
  p.p[1] = is_compound ? r0 + (r1 + 1) * mfkBankRefsPerFrame() : r0;
  p.p[2] = mvs[0].row;
  p.p[3] = mvs[0].col;
  p.p[4] = is_compound ? mvs[1].row : 0;
  p.p[5] = is_compound ? mvs[1].col : 0;

  const int count = bank->size[ref];
  const int start = bank->start[ref];
  int found = -1;
  for (int i = 0; i < count; ++i) {
    const int idx = (start + i) % mfkRefMvBankSize();
    if (mf_rmb_params_equal(bank->bank[ref][idx], p)) {
      found = i;
      break;
    }
  }
  if (found >= 0) {
    const int found_idx = (start + found) % mfkRefMvBankSize();
    const RefMvBankEntry tmp = bank->bank[ref][found_idx];
    for (int i = found; i < count - 1; ++i) {
      const int idx0 = (start + i) % mfkRefMvBankSize();
      const int idx1 = (start + i + 1) % mfkRefMvBankSize();
      bank->bank[ref][idx0] = bank->bank[ref][idx1];
    }
    const int tail = (start + count - 1) % mfkRefMvBankSize();
    bank->bank[ref][tail] = tmp;
    return true;
  }

  const int idx = (start + count) % mfkRefMvBankSize();
  bank->bank[ref][idx] = p;
  if (count < mfkRefMvBankSize()) {
    bank->size[ref] = count + 1;
  } else {
    bank->start[ref] = (start + 1) % mfkRefMvBankSize();
  }
  return true;
}

AV2ENC_MF_HD void mf_reset_ref_mv_bank(RefMvBank* bank, int mi_row,
                                              int mi_col, int sb_size4,
                                              bool frame_is_intra, bool top_row,
                                              const RefMvBankRowSource* sources,
                                              int row_mi_cols) {
  if (bank == nullptr) {
    return;
  }
  bank->bank_hits = 0;
  bank->remain_hits = 0;
  bank->unit_hits = 0;
  if (frame_is_intra || top_row) {
    return;
  }
  if (sources == nullptr || row_mi_cols <= 0) {
    return;
  }
  int cand_col = mi_col;
  int row_hits = 0;
  while (cand_col < row_mi_cols && cand_col < mi_col + sb_size4 &&
         row_hits < 4) {
    const int cand_col2 = (cand_col >> 1) << 1;
    const RefMvBankRowSource& src =
        sources[static_cast<std::size_t>(cand_col2)];
    if (src.is_inter != 0) {
      ++row_hits;
      const int ref_frames[2] = {src.ref[0], src.ref[1]};
      const Mv mvs[2] = {Mv{src.mv[0][0], src.mv[0][1]},
                         Mv{src.mv[1][0], src.mv[1][1]}};
      (void)mf_update_ref_mv_bank(bank, mi_row, mi_col, 1, 1, sb_size4, 0,
                                  ref_frames, mvs, src.cwp_idx, false);
    }
    const int step = src.bw4 > 0 ? src.bw4 : 1;
    cand_col += step;
  }
}

AV2ENC_MF_HD void mf_check_rmb_cand(
    const RefMvBankEntry& entry, const int cand_mvs[2][2], bool is_compound,
    int mi_row, int mi_col, int mi_cols, int mi_rows, int bw, int bh,
    int max_out, StackMv* stack, int* num_found, int* prune_count) {
  if (*num_found >= max_out) {
    return;
  }
  if (*prune_count < mfkMaxPrNum()) {
    for (int i = 0; i < *num_found; ++i) {
      ++(*prune_count);
      if (cand_mvs[0][0] == stack[i].mv[0].row &&
          cand_mvs[0][1] == stack[i].mv[0].col &&
          (!is_compound || (cand_mvs[1][0] == stack[i].mv[1].row &&
                            cand_mvs[1][1] == stack[i].mv[1].col))) {
        return;
      }
    }
  }
  for (int i = 0; i < 1 + (is_compound ? 1 : 0); ++i) {
    const int ref_y = mi_row * mfkMiSize() + cand_mvs[i][0] / 8;
    const int ref_x = mi_col * mfkMiSize() + cand_mvs[i][1] / 8;
    if (ref_x <= -bw || ref_y <= -bh || ref_x >= mi_cols * mfkMiSize() ||
        ref_y >= mi_rows * mfkMiSize()) {
      return;
    }
  }
  for (int i = 0; i < 1 + (is_compound ? 1 : 0); ++i) {
    stack[*num_found].mv[i].row = cand_mvs[i][0];
    stack[*num_found].mv[i].col = cand_mvs[i][1];
  }
  if (!is_compound) {
    stack[*num_found].mv[1] = stack[*num_found].mv[0];
  }
  stack[*num_found].cwp = static_cast<int>(entry.p[0]);
  ++(*num_found);
}

AV2ENC_MF_HD void mf_fill_mvp_from_ref_mv_bank(
    const RefMvBank& bank, const int ref_frames[2], bool is_compound,
    bool use_intrabc, int max_drl_bits_minus_1, int mi_row, int mi_col,
    int mi_cols, int mi_rows, int bw, int bh, int max_out, StackMv* stack,
    int* num_found, int* prune_count) {
  if (stack == nullptr || num_found == nullptr || prune_count == nullptr) {
    return;
  }
  const int ref = mf_get_rmb_list_index(ref_frames);
  const int key = is_compound ? ref_frames[0] + (ref_frames[1] + 1) *
                                                  mfkBankRefsPerFrame()
                              : ref_frames[0];
  int max_ref_mv_count = max_drl_bits_minus_1 + 2;
  if (use_intrabc) {
    max_ref_mv_count = max_drl_bits_minus_1 + 2;
  }
  if (max_ref_mv_count > mfkMaxRefMvStackSize()) {
    max_ref_mv_count = mfkMaxRefMvStackSize();
  }
  if (max_ref_mv_count > max_out) {
    max_ref_mv_count = max_out;
  }
  if (max_ref_mv_count > 6) {
    max_ref_mv_count = 6;
  }
  const int count = bank.size[ref];
  const int start = bank.start[ref];
  for (int i = count - 1; i >= 0 && *num_found < max_ref_mv_count; --i) {
    const int idx = (start + i) % mfkRefMvBankSize();
    const RefMvBankEntry& entry = bank.bank[ref][idx];
    if (entry.p[1] != key) {
      continue;
    }
    int cand_mvs[2][2] = {{0, 0}, {0, 0}};
    for (int list = 0; list < 2; ++list) {
      for (int comp = 0; comp < 2; ++comp) {
        cand_mvs[list][comp] =
            static_cast<int>(entry.p[2 + list * 2 + comp]);
      }
    }
    mf_check_rmb_cand(entry, cand_mvs, is_compound, mi_row, mi_col, mi_cols,
                      mi_rows, bw, bh, max_out, stack, num_found,
                      prune_count);
  }
}

// §7.12.2.19 sorting process: one move-to-front of the maximum weight.
AV2ENC_MF_HD void mf_sort_stack_range(StackMv* stack, int count,
                                             int start, int end) {
  if (stack == nullptr) {
    return;
  }
  if (start < 0) {
    start = 0;
  }
  if (end > count) {
    end = count;
  }
  if (start + 1 >= end) {
    return;
  }
  int max_weight = stack[start].weight;
  int max_idx = start;
  for (int idx = start + 1; idx < end; ++idx) {
    if (max_weight < stack[idx].weight) {
      max_weight = stack[idx].weight;
      max_idx = idx;
    }
  }
  if (max_idx != start) {
    const StackMv tmp = stack[start];
    stack[start] = stack[max_idx];
    stack[max_idx] = tmp;
  }
}

// One merge-stack work item: §7.12.2.7/.8 temporal candidates, then the
// §7.12.2.21 bank appends, then the §7.12.2.19 sort.  Shared by the kernel and
// the host-emulation parity test.
AV2ENC_MF_HD void mf_merge_block(const MfStackBlockInput& in,
                                        const RefMvBank* bank,
                                        StackMv* stack_out,
                                        int* num_found_out, int* prune_out) {
  StackMv stack[mfkMaxRefMvStackSize()];
  for (int i = 0; i < mfkMaxRefMvStackSize(); ++i) {
    stack[i] = in.stack[i];
  }
  int num_found = in.num_found;
  int prune = in.prune_count;

  if (in.use_temporal != 0 && in.field.mv != nullptr &&
      in.field.valid != nullptr && in.field.offset != nullptr) {
    MfScanInput scan;
    scan.mi_row = in.mi_row;
    scan.mi_col = in.mi_col;
    scan.mi_cols = in.mi_cols;
    scan.mi_rows = in.mi_rows;
    scan.tile_mi_col_start = in.tile_mi_col_start;
    scan.tile_mi_col_end = in.tile_mi_col_end;
    scan.tile_mi_row_start = in.tile_mi_row_start;
    scan.tile_mi_row_end = in.tile_mi_row_end;
    scan.bw4 = in.bw4;
    scan.bh4 = in.bh4;
    scan.ref0 = in.ref0;
    scan.ref1 = in.ref1;
    scan.ref0_order_hint = in.ref0_order_hint;
    scan.ref1_order_hint = in.ref1_order_hint;
    scan.order_hint = in.order_hint;
    scan.is_compound = in.is_compound;
    scan.use_traj = in.use_traj;
    scan.field = &in.field;
    scan.traj = in.traj.mv != nullptr ? &in.traj : nullptr;
    mf_temporal_scan(scan, stack, &num_found, &prune);
  }

  if (in.use_refmvbank != 0 && bank != nullptr) {
    const int ref_frames[2] = {in.ref0, in.ref1};
    mf_fill_mvp_from_ref_mv_bank(*bank, ref_frames, in.is_compound != 0,
                                 in.use_intrabc != 0, in.max_drl_bits_minus_1,
                                 in.mi_row, in.mi_col, in.mi_cols, in.mi_rows,
                                 in.bw4 * mfkMiSize(), in.bh4 * mfkMiSize(),
                                 mfkMaxRefMvStackSize(), stack, &num_found,
                                 &prune);
  }

  if (in.reorder != 0) {
    mf_sort_stack_range(stack, num_found, 0, in.num_nearest);
  }

  for (int i = 0; i < mfkMaxRefMvStackSize(); ++i) {
    stack_out[i] = stack[i];
  }
  if (num_found_out != nullptr) {
    *num_found_out = num_found;
  }
  if (prune_out != nullptr) {
    *prune_out = prune;
  }
}

}  // namespace av2enc::cuda::motion_field

#endif  // AV2ENC_CUDA_MOTION_FIELD_MOTION_FIELD_DEVICE_CUH_

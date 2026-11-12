// AV2 §7.16 Deblocking filter for TIP — shared host/device core.
//
// Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7), §7.16 (pages 528–529), invoking §7.17.3
// filter maximum width, §7.17.7.1 sample filtering and §7.17.7.2 filter
// choice.
//
// This header is compiled three ways:
//   * by nvcc for sm_86 / sm_120, as the body of every device kernel;
//   * by the host compiler during the emulation parity test, with the CUDA
//     keywords absent (AV2ENC_TIP_HD becomes `inline`);
//   * transitively by the CPU reference through the §7.17 core header, which
//     the arithmetic below reuses (filter_max_width, get_q, round2, clip3,
//     the normative tables).
//
// §7.16 in one paragraph: the edge lattice is the TIP build block grid
// (`tipSize` 8 or 16 luma samples), not the coding/transform grid.  There is
// no mode-info, no block/TX/sub-PU decision, no skip/lossless/segmentation and
// no per-cell qindex.  One (qThr, side) pair is derived per plane from
// baseFilterLevel, outside the traversal.  The vertical pass visits x = k*sw
// and y = 4*m; the horizontal pass visits x = 4*m and y = k*sh, with literal
// 4-sample groups (not scaled by chroma subsampling).  The vertical tile edge
// is the vertical pass's sbEdge; the horizontal pass's sbEdge is the 64x64
// grid edge, so a horizontal tile edge gates only the skip decision.
//
// Bit depth: the sample kernels are templated on Sample/BitDepth (8 and 10
// bits).  Only Clip1 changes with bit depth; the §7.17.7 arithmetic is
// bit-depth agnostic once `side` has been derived with the bit-depth offsets
// in §7.16.  The 8-bit instantiation is byte-for-byte the §7.17 kernel
// arithmetic, so it is bit-exact with the CPU reference.
//
// This file is device-safe: it includes no CUDA runtime header and performs no
// allocation.  Buffer sizing helpers are host-inline so the emulation test can
// check the arena plan without a GPU.

#ifndef AV2ENC_CUDA_TIP_DEBLOCK_TIP_DEBLOCK_DEVICE_H_
#define AV2ENC_CUDA_TIP_DEBLOCK_TIP_DEBLOCK_DEVICE_H_

#include <cstddef>
#include <cstdint>

#include "encoder/filters/deblock/deblock_core.h"
#include "encoder/filters/tip_deblock/tip_deblock_core.h"

#if defined(__CUDACC__)
#define AV2ENC_TIP_HD __host__ __device__ __forceinline__
#else
#define AV2ENC_TIP_HD inline
#endif

namespace av2enc::cuda::tip_deblock {

// Arena allocations are 256-byte aligned (Arena::kAlignment).  Kept as a
// local constant so this header stays CUDA-runtime-free; tip_deblock.cu
// static_asserts that the two agree.
inline constexpr std::size_t kTipArenaAlignment = 256;

// MI_SIZE / MI_SIZE_LOG2 (§9.2).  §7.16 converts plane coordinates to luma MI
// coordinates for the tile-edge tests.
inline constexpr int kTipMiSize = 4;
inline constexpr int kTipMiSizeLog2 = 2;

// ---------------------------------------------------------------------------
// Frame geometry and arena sizing (host-inline, no CUDA types).
// ---------------------------------------------------------------------------
struct TipDeblockConfig {
  int mi_cols = 0;
  int mi_rows = 0;
  int subsampling_x = 1;
  int subsampling_y = 1;
  int bit_depth = 8;
  // Plane strides in samples.  A zero entry means "the derived plane width".
  int stride[3] = {0, 0, 0};
  // Tile starts in luma MI units (MiColStarts / MiRowStarts).  The arena frame
  // allocates one uint16 per entry when the count is positive.
  int num_tile_cols = 0;
  int num_tile_rows = 0;
};

inline int tip_plane_shift_x(const TipDeblockConfig& config, int plane) {
  return (plane == 0) ? 0 : config.subsampling_x;
}

inline int tip_plane_shift_y(const TipDeblockConfig& config, int plane) {
  return (plane == 0) ? 0 : config.subsampling_y;
}

inline int tip_deblock_plane_width(const TipDeblockConfig& config,
                                   int plane) {
  if (plane < 0 || plane > 2 || config.mi_cols <= 0) {
    return 0;
  }
  return (config.mi_cols * kTipMiSize) >> tip_plane_shift_x(config, plane);
}

inline int tip_deblock_plane_height(const TipDeblockConfig& config,
                                    int plane) {
  if (plane < 0 || plane > 2 || config.mi_rows <= 0) {
    return 0;
  }
  return (config.mi_rows * kTipMiSize) >> tip_plane_shift_y(config, plane);
}

inline int tip_deblock_plane_stride(const TipDeblockConfig& config,
                                    int plane) {
  if (plane < 0 || plane > 2) {
    return 0;
  }
  return (config.stride[plane] > 0) ? config.stride[plane]
                                    : tip_deblock_plane_width(config, plane);
}

inline int tip_sample_bytes(int bit_depth) { return (bit_depth == 10) ? 2 : 1; }

inline std::size_t tip_align_up(std::size_t bytes,
                                std::size_t alignment = kTipArenaAlignment) {
  const std::size_t mask = alignment - 1;
  return (bytes + mask) & ~mask;
}

inline std::size_t tip_deblock_plane_bytes(const TipDeblockConfig& config,
                                           int plane) {
  if (plane < 0 || plane > 2) {
    return 0;
  }
  const std::size_t stride =
      static_cast<std::size_t>(tip_deblock_plane_stride(config, plane));
  const std::size_t height =
      static_cast<std::size_t>(tip_deblock_plane_height(config, plane));
  return stride * height * static_cast<std::size_t>(tip_sample_bytes(config.bit_depth));
}

// §7.16 is in-place: the filter reads and writes only the three plane buffers.
// There is no per-launch scratch.  The function exists so the plan is explicit
// and so a future staging step cannot silently grow the footprint; the arena
// entry points still bracket the launch with mark()/rewind().
inline std::size_t tip_deblock_scratch_bytes(const TipDeblockConfig&) {
  return 0;
}

// Bytes a caller must have available in one persistent Arena before
// tip_deblock_arena_frame_init() for this config:
//
//   three aligned plane buffers
// + aligned MiColStarts / MiRowStarts arrays
// + scratch (0 today)
//
// The arena's own base-pointer alignment can consume up to
// Arena::kAlignment - 1 extra bytes, so size the arena with
// `tip_deblock_required_device_bytes(config) + Arena::kAlignment` of slack.
inline std::size_t tip_deblock_required_device_bytes(
    const TipDeblockConfig& config) {
  std::size_t total = 0;
  for (int plane = 0; plane < 3; ++plane) {
    total += tip_align_up(tip_deblock_plane_bytes(config, plane));
  }
  total += tip_align_up(static_cast<std::size_t>(config.num_tile_cols) *
                        sizeof(std::uint16_t));
  total += tip_align_up(static_cast<std::size_t>(config.num_tile_rows) *
                        sizeof(std::uint16_t));
  total += tip_align_up(tip_deblock_scratch_bytes(config));
  return total;
}

// ---------------------------------------------------------------------------
// Device options: the subset of the CPU TipDeblockOptions the kernels need.
// The host maps the public options struct onto this POD (tip size is derived
// by the shared select_tip_size(), never recomputed on the device).
// ---------------------------------------------------------------------------
struct TipDeblockDeviceOptions {
  int base_q_idx = 0;
  int delta_q_u_ac = 0;
  int delta_q_v_ac = 0;
  int base_uv_ac_delta_q = 0;
  int bit_depth = 8;
  int tip_size = 16;  // 8 (BLOCK_8X8) or 16 (BLOCK_16X16)
  int disable_loopfilters_across_tiles = 0;
};

// Per-plane §7.16 strength, derived once per plane.  Mirrors
// av2enc::tip_deblock::derive_plane_info() exactly.
struct TipPlaneInfo {
  int base_filter_level = 0;
  int q_thr = 0;
  int side = 0;
};

AV2ENC_TIP_HD TipPlaneInfo derive_plane_info(
    int plane, const TipDeblockDeviceOptions& options) {
  int base_filter_level = options.base_q_idx;
  if (plane == 1) {
    base_filter_level += options.delta_q_u_ac + options.base_uv_ac_delta_q;
  } else if (plane == 2) {
    base_filter_level += options.delta_q_v_ac + options.base_uv_ac_delta_q;
  }

  // No q_clamped() special case, no segment id, no DfDeltaQ term.  With
  // baseFilterLevel == 0, get_q(0, 0) returns Ac_Qlookup[0] (64), so qThr is
  // 1; `side` is 0 and filter_choice() returns 0.  QUANT_TABLE_BITS is 3
  // (§9.2), the same literal the §7.17 core uses.
  const int q_thr =
      av2enc::deblock::round2(av2enc::deblock::get_q(base_filter_level, 0), 3) >>
      6;
  const int q_ind = av2enc::deblock::clip3(
      0, av2enc::deblock::kMaxSideTable - 1,
      base_filter_level - 24 * (options.bit_depth - 8));
  const int side =
      av2enc::deblock::imax(
          av2enc::deblock::kSideThresholds[q_ind] +
              (1 << (12 - options.bit_depth)),
          0) >>
      (13 - options.bit_depth);

  TipPlaneInfo info;
  info.base_filter_level = base_filter_level;
  info.q_thr = q_thr;
  info.side = side;
  return info;
}

// ---------------------------------------------------------------------------
// Frame view.  The plane pointers are typed per bit depth; the host launcher
// reinterpret_casts the arena buffers once.
// ---------------------------------------------------------------------------
template <typename Sample>
struct TipDeblockFrameView {
  Sample* plane[3] = {nullptr, nullptr, nullptr};
  int stride[3] = {0, 0, 0};
  int mi_cols = 0;
  int mi_rows = 0;
  int sub_x = 1;
  int sub_y = 1;
  const std::uint16_t* tile_col_starts = nullptr;
  int num_tile_cols = 0;
  const std::uint16_t* tile_row_starts = nullptr;
  int num_tile_rows = 0;
};

// ---------------------------------------------------------------------------
// Pixel helpers.
// ---------------------------------------------------------------------------
template <typename Sample>
AV2ENC_TIP_HD int tip_sample_at(const Sample* plane, int stride, int x, int y) {
  return static_cast<int>(
      plane[static_cast<std::ptrdiff_t>(y) * stride + x]);
}

template <typename Sample, int BitDepth>
AV2ENC_TIP_HD void tip_write_sample(Sample* plane, int stride, int x, int y,
                                    int value) {
  constexpr int high = (1 << BitDepth) - 1;
  const int clipped = value < 0 ? 0 : (value > high ? high : value);
  plane[static_cast<std::ptrdiff_t>(y) * stride + x] =
      static_cast<Sample>(clipped);
}

// §7.16 is_vert_tile_edge(x, subX): col = (x << subX) >> MI_SIZE_LOG2.
AV2ENC_TIP_HD bool tip_is_vert_tile_edge(
    const std::uint16_t* starts, int count, int x, int sub_x) {
  const int col = (x << sub_x) >> kTipMiSizeLog2;
  return av2enc::deblock::is_tile_start(starts, count, col);
}

// §7.16 is_horz_tile_edge(y, subY): row = (y << subY) >> MI_SIZE_LOG2.
AV2ENC_TIP_HD bool tip_is_horz_tile_edge(
    const std::uint16_t* starts, int count, int y, int sub_y) {
  const int row = (y << sub_y) >> kTipMiSizeLog2;
  return av2enc::deblock::is_tile_start(starts, count, row);
}

// §7.17.7.2 filter choice, shared by §7.16 with count = MI_SIZE and the
// plane-coordinate (x, y).  Returns the chosen width or 0.
template <typename Sample, int BitDepth>
AV2ENC_TIP_HD int tip_filter_choice(const Sample* plane, int stride, int x,
                                    int y, int q_thr, int side_thr, int dx,
                                    int dy, int max_width_neg,
                                    int max_width_pos) {
  if (q_thr == 0 || side_thr == 0) {
    return 0;
  }
  constexpr int count = kTipMiSize;
  const int max_samples_pos =
      av2enc::deblock::clip3(3, 8, max_width_pos + 1);
  const int max_samples_neg =
      av2enc::deblock::clip3(3, 8, max_width_neg + 1);
  constexpr int kOff = 8;

  int s[16];
  int t[16];
  const int x2 = x + (count - 1) * dy;
  const int y2 = y + (count - 1) * dx;

  for (int dist = 0; dist < max_samples_pos; ++dist) {
    s[kOff + dist] = tip_sample_at(plane, stride, x + dist * dx, y + dist * dy);
    t[kOff + dist] =
        tip_sample_at(plane, stride, x2 + dist * dx, y2 + dist * dy);
  }
  for (int dist = 0; dist < max_samples_neg; ++dist) {
    s[kOff - dist - 1] =
        tip_sample_at(plane, stride, x - (dist + 1) * dx, y - (dist + 1) * dy);
    t[kOff - dist - 1] =
        tip_sample_at(plane, stride, x2 - (dist + 1) * dx,
                      y2 - (dist + 1) * dy);
  }

  int second_deriv[4];  // dist = -2..1 stored at [dist + 2]
  for (int dist = -2; dist < 2; ++dist) {
    const int sp0 = s[kOff + dist - 1];
    const int sq0 = s[kOff + dist];
    const int sq1 = s[kOff + dist + 1];
    const int tp0 = t[kOff + dist - 1];
    const int tq0 = t[kOff + dist];
    const int tq1 = t[kOff + dist + 1];
    const int deriv_s = av2enc::deblock::iabs(sp0 - (sq0 << 1) + sq1);
    const int deriv_t = av2enc::deblock::iabs(tp0 - (tq0 << 1) + tq1);
    second_deriv[dist + 2] = (deriv_s + deriv_t + 1) >> 1;
  }
  const int sd_m2 = second_deriv[0];
  const int sd_m1 = second_deriv[1];
  const int sd_0 = second_deriv[2];
  const int sd_1 = second_deriv[3];

  if (sd_m2 > side_thr || sd_1 > side_thr) {
    return 0;
  }
  if (max_width_pos == 1) {
    return 1;
  }

  const int side_thr2 = side_thr >> 2;
  if (sd_m2 > side_thr2 || sd_1 > side_thr2) {
    return 1;
  }
  if (sd_m1 + sd_0 > q_thr * 4) {
    return 1;
  }

  const int side_thr3 = side_thr >> 3;
  if (sd_m2 > side_thr3 || sd_1 > side_thr3) {
    return 2;
  }
  if (sd_m1 + sd_0 > q_thr * 3) {
    return 2;
  }

  const int end_thr = (side_thr * 3) >> 4;
  if (max_width_neg > 2) {
    const int deriv_s = av2enc::deblock::iabs(
        s[kOff - 1] - s[kOff - 4] - 3 * (s[kOff - 1] - s[kOff - 2]));
    const int deriv_t = av2enc::deblock::iabs(
        t[kOff - 1] - t[kOff - 4] - 3 * (t[kOff - 1] - t[kOff - 2]));
    if (((deriv_s + deriv_t + 1) >> 1) > end_thr) {
      return 2;
    }
  }
  {
    const int deriv_s = av2enc::deblock::iabs(
        s[kOff] - s[kOff + 3] - 3 * (s[kOff] - s[kOff + 1]));
    const int deriv_t = av2enc::deblock::iabs(
        t[kOff] - t[kOff + 3] - 3 * (t[kOff] - t[kOff + 1]));
    if (((deriv_s + deriv_t + 1) >> 1) > end_thr) {
      return 2;
    }
  }

  if (max_width_pos == 3) {
    return 3;
  }

  const int transition = (sd_m1 + sd_0) << 4;
  int prev_dist = 3;
  for (int dist = 4; dist <= max_width_pos; dist += 2) {
    const int q_thr4 = q_thr * av2enc::deblock::kQFirst[dist - 4];
    const int end_thr4 = (side_thr * dist) >> 4;
    if (transition > q_thr4) {
      return prev_dist;
    }
    const int dist2 = av2enc::deblock::imin(7, dist);
    if (max_width_neg >= dist2) {
      const int deriv_s = av2enc::deblock::iabs(
          s[kOff - 1] - s[kOff - dist2 - 1] -
          dist2 * (s[kOff - 1] - s[kOff - 2]));
      const int deriv_t = av2enc::deblock::iabs(
          t[kOff - 1] - t[kOff - dist2 - 1] -
          dist2 * (t[kOff - 1] - t[kOff - 2]));
      if (((deriv_s + deriv_t + 1) >> 1) > end_thr4) {
        return prev_dist;
      }
    }
    {
      const int deriv_s = av2enc::deblock::iabs(
          s[kOff] - s[kOff + dist2] -
          dist2 * (s[kOff] - s[kOff + 1]));
      const int deriv_t = av2enc::deblock::iabs(
          t[kOff] - t[kOff + dist2] -
          dist2 * (t[kOff] - t[kOff + 1]));
      if (((deriv_s + deriv_t + 1) >> 1) > end_thr4) {
        return prev_dist;
      }
    }
    prev_dist = dist;
  }
  return max_width_pos;
}

// §7.17.7.1 sample filtering.  §7.16 always passes prevLossless =
// currLossless = 0 and Min(width, maxWidthNeg/Pos).
template <typename Sample, int BitDepth>
AV2ENC_TIP_HD void tip_sample_filtering(
    Sample* plane, int stride, int x, int y, int q_thr, int dx, int dy,
    int max_width_neg, int max_width_pos, int prev_lossless,
    int curr_lossless) {
  const int width = av2enc::deblock::imax(max_width_neg, max_width_pos);
  const int q0 = tip_sample_at(plane, stride, x, y);
  const int q1 = tip_sample_at(plane, stride, x + dx, y + dy);
  const int p0 = tip_sample_at(plane, stride, x - dx, y - dy);
  const int p1 = tip_sample_at(plane, stride, x - 2 * dx, y - 2 * dy);

  const int q_thr_clamp =
      q_thr * av2enc::deblock::kQThreshMults[width - 1];
  int delta_m2 = p1 - q1 + 3 * (q0 - p0);
  delta_m2 *= 4;
  delta_m2 = av2enc::deblock::clip3(-q_thr_clamp, q_thr_clamp, delta_m2);

  const int delta_m2_neg =
      delta_m2 * av2enc::deblock::kWMult[max_width_neg - 1];
  const int delta_m2_pos =
      delta_m2 * av2enc::deblock::kWMult[max_width_pos - 1];

  for (int i = 0; i < width; ++i) {
    const int diff_neg = av2enc::deblock::round2(
        delta_m2_neg * (max_width_neg - i), 3 + av2enc::deblock::kDfShift);
    const int diff_pos = av2enc::deblock::round2(
        delta_m2_pos * (max_width_pos - i), 3 + av2enc::deblock::kDfShift);
    const int qy = y + i * dy;
    const int qx = x + i * dx;
    if (!curr_lossless) {
      tip_write_sample<Sample, BitDepth>(
          plane, stride, qx, qy,
          tip_sample_at(plane, stride, qx, qy) - diff_pos);
    }
    if (i < max_width_neg && !prev_lossless) {
      const int pi = -i - 1;
      const int py = y + pi * dy;
      const int px = x + pi * dx;
      tip_write_sample<Sample, BitDepth>(
          plane, stride, px, py,
          tip_sample_at(plane, stride, px, py) + diff_neg);
    }
  }
}

// ---------------------------------------------------------------------------
// One §7.16 edge.  `pass == 0` is the vertical-boundary pass (dx=1, dy=0);
// `pass == 1` is horizontal (dx=0, dy=1).  The decision record is identical
// to the CPU inspection path; `Apply` selects writing or not.
// ---------------------------------------------------------------------------
template <bool Apply, typename Sample, int BitDepth>
AV2ENC_TIP_HD av2enc::tip_deblock::TipDeblockEdgeInfo tip_edge_view(
    const TipDeblockFrameView<Sample>& frame,
    const TipDeblockDeviceOptions& options, int plane, int pass, int x, int y) {
  av2enc::tip_deblock::TipDeblockEdgeInfo info{};
  if (plane < 0 || plane > 2 || pass < 0 || pass > 1) {
    return info;
  }

  const int sub_x = (plane != 0) ? frame.sub_x : 0;
  const int sub_y = (plane != 0) ? frame.sub_y : 0;
  const int dx = (pass == 0) ? 1 : 0;
  const int dy = (pass == 0) ? 0 : 1;
  const int sw = options.tip_size >> sub_x;
  const int sh = options.tip_size >> sub_y;
  const int w = (frame.mi_cols * kTipMiSize) >> sub_x;
  const int h = (frame.mi_rows * kTipMiSize) >> sub_y;

  // The §7.16 loops only visit these positions.
  if (pass == 0) {
    if (x <= 0 || x >= w || (x % sw) != 0 || y < 0 || (y % 4) != 0 ||
        y + 3 >= h) {
      return info;
    }
  } else {
    if (y <= 0 || y >= h || (y % sh) != 0 || x < 0 || (x % 4) != 0 ||
        x + 3 >= w) {
      return info;
    }
  }
  info.on_screen = 1;

  const TipPlaneInfo plane_info = derive_plane_info(plane, options);
  info.q_thr = plane_info.q_thr;
  info.side = plane_info.side;

  if (pass == 0) {
    info.tile_vert_edge =
        tip_is_vert_tile_edge(frame.tile_col_starts, frame.num_tile_cols, x,
                              sub_x)
            ? 1
            : 0;
    // The vertical pass passes vertTileEdge as sbEdge.
    info.sb_edge = info.tile_vert_edge;
    info.filter_size = sw;
    info.tile_skipped =
        (options.disable_loopfilters_across_tiles && info.tile_vert_edge) ? 1
                                                                        : 0;
  } else {
    info.tile_horz_edge =
        tip_is_horz_tile_edge(frame.tile_row_starts, frame.num_tile_rows, y,
                              sub_y)
            ? 1
            : 0;
    // The horizontal pass passes horz64Edge as sbEdge; a horizontal tile edge
    // only gates the skip decision.
    info.sb_edge = ((((y << sub_y) % 64) == 0)) ? 1 : 0;
    info.filter_size = sh;
    info.tile_skipped =
        (options.disable_loopfilters_across_tiles && info.tile_horz_edge) ? 1
                                                                        : 0;
  }

  // filterSize is sw/sh, unclipped: §7.16 has no screen-edge clip.
  av2enc::deblock::filter_max_width(plane, info.filter_size, info.sb_edge,
                                   &info.max_width_pos, &info.max_width_neg);

  if (info.tile_skipped) {
    return info;
  }

  Sample* data = frame.plane[plane];
  const int stride = frame.stride[plane];
  info.width = tip_filter_choice<Sample, BitDepth>(
      data, stride, x, y, info.q_thr, info.side, dx, dy, info.max_width_neg,
      info.max_width_pos);
  if (info.width <= 0) {
    return info;
  }
  info.apply_filter = 1;

  if (Apply) {
    const int neg = av2enc::deblock::imin(info.width, info.max_width_neg);
    const int pos = av2enc::deblock::imin(info.width, info.max_width_pos);
    for (int i = 0; i < kTipMiSize; ++i) {
      const int sx = x + dy * i;
      const int sy = y + dx * i;
      tip_sample_filtering<Sample, BitDepth>(data, stride, sx, sy, info.q_thr,
                                             dx, dy, neg, pos, 0, 0);
    }
  }
  return info;
}

// One plane/pass traversal, exactly the §7.16 loop nest.
template <typename Sample, int BitDepth>
AV2ENC_TIP_HD void tip_plane_pass_view(
    const TipDeblockFrameView<Sample>& frame,
    const TipDeblockDeviceOptions& options, int plane, int pass) {
  if (plane < 0 || plane > 2 || pass < 0 || pass > 1) {
    return;
  }
  const int sub_x = (plane != 0) ? frame.sub_x : 0;
  const int sub_y = (plane != 0) ? frame.sub_y : 0;
  const int sw = options.tip_size >> sub_x;
  const int sh = options.tip_size >> sub_y;
  const int w = (frame.mi_cols * kTipMiSize) >> sub_x;
  const int h = (frame.mi_rows * kTipMiSize) >> sub_y;

  if (pass == 0) {
    // Literal §7.16: y += 4 (not 4 << subY), x += sw.
    for (int y = 0; y < h; y += 4) {
      for (int x = 0; x < w; x += sw) {
        if (x > 0) {
          (void)tip_edge_view<true, Sample, BitDepth>(frame, options, plane,
                                                      pass, x, y);
        }
      }
    }
  } else {
    // Literal §7.16: x += 4 (not 4 << subX), y += sh.
    for (int x = 0; x < w; x += 4) {
      for (int y = 0; y < h; y += sh) {
        if (y > 0) {
          (void)tip_edge_view<true, Sample, BitDepth>(frame, options, plane,
                                                      pass, x, y);
        }
      }
    }
  }
}

// Full §7.16 traversal (plane, then vertical, then horizontal).  The CUDA
// launcher interleaves the planes across two kernel launches instead, which is
// equivalent because the three planes are independent and each plane keeps
// vertical-before-horizontal ordering.
template <typename Sample, int BitDepth>
AV2ENC_TIP_HD void tip_frame_view(const TipDeblockFrameView<Sample>& frame,
                                  const TipDeblockDeviceOptions& options) {
  for (int plane = 0; plane < 3; ++plane) {
    tip_plane_pass_view<Sample, BitDepth>(frame, options, plane, 0);
    tip_plane_pass_view<Sample, BitDepth>(frame, options, plane, 1);
  }
}

}  // namespace av2enc::cuda::tip_deblock

#endif  // AV2ENC_CUDA_TIP_DEBLOCK_TIP_DEBLOCK_DEVICE_H_

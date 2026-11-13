// Host-emulation parity test for the shared AV2 §7.16 TIP deblocking core.
//
// Compiles tip_deblock_device.h as ordinary C++ (AV2ENC_TIP_HD -> inline) and
// compares it against:
//   * the CPU reference in src/encoder/filters/tip_deblock/ for 8-bit
//     (all 256 base_q_idx values, both tip sizes, all planes and passes,
//      tile gating, random frames, and a real reconstructed frame);
//   * an independent transcription of the §7.16/§7.17.7 arithmetic for 10-bit,
//     which the CPU reference does not implement.
//
// No GPU or nvcc is required.  The real GPU-vs-CPU test is tip_deblock_test.cu.
//
// Usage: tip_deblock_emulation_test [repo_root]

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "tip_deblock_device.h"

#include "codec/filters/deblock/deblock_core.h"
#include "codec/filters/tip_deblock/tip_deblock.h"

namespace ctd = av2enc::cuda::tip_deblock;
namespace cpu = av2enc::tip_deblock;

namespace {

using cpu::InterpFilter;
using cpu::TipDeblockOptions;
using cpu::TipDeblockStatus;
using cpu::TipFrameMode;

int g_checks = 0;
int g_failures = 0;
std::uint64_t g_samples_compared = 0;
std::uint64_t g_samples_changed = 0;

#define CHECK(cond, msg)                                                   \
  do {                                                                     \
    ++g_checks;                                                            \
    if (!(cond)) {                                                         \
      ++g_failures;                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, (msg)); \
    }                                                                      \
  } while (0)

int random_range(std::mt19937& rng, int low, int high) {
  const std::uint32_t span = static_cast<std::uint32_t>(high - low + 1);
  return low + static_cast<int>(rng() % span);
}

// ---------------------------------------------------------------------------
// Test frame, templated on the sample type so 8- and 10-bit share the guard /
// checksum / comparison logic.
// ---------------------------------------------------------------------------
template <typename Sample>
struct FrameT {
  int mi_cols = 0;
  int mi_rows = 0;
  int guard_cols = 8;
  int guard_rows = 2;
  int stride[3] = {0, 0, 0};
  int height[3] = {0, 0, 0};
  Sample guard = 0;
  std::vector<Sample> data[3];
  std::vector<std::uint16_t> tile_cols;
  std::vector<std::uint16_t> tile_rows;

  int plane_width(int plane) const {
    const int luma = mi_cols * 4;
    return (plane == 0) ? luma : luma / 2;
  }
  int plane_height(int plane) const {
    const int luma = mi_rows * 4;
    return (plane == 0) ? luma : luma / 2;
  }
  Sample* plane(int plane) { return data[plane].data(); }
  const Sample* plane(int plane) const { return data[plane].data(); }
  std::size_t index(int plane, int y, int x) const {
    return static_cast<std::size_t>(y) *
               static_cast<std::size_t>(stride[plane]) +
           static_cast<std::size_t>(x);
  }
};

template <typename Sample>
FrameT<Sample> make_frame(int mi_cols, int mi_rows, Sample guard) {
  FrameT<Sample> frame;
  frame.mi_cols = mi_cols;
  frame.mi_rows = mi_rows;
  frame.guard = guard;
  frame.stride[0] = mi_cols * 4 + frame.guard_cols;
  frame.stride[1] = mi_cols * 2 + frame.guard_cols;
  frame.stride[2] = frame.stride[1];
  frame.height[0] = mi_rows * 4 + frame.guard_rows;
  frame.height[1] = mi_rows * 2 + frame.guard_rows;
  frame.height[2] = frame.height[1];
  for (int plane = 0; plane < 3; ++plane) {
    frame.data[plane].assign(
        static_cast<std::size_t>(frame.stride[plane]) *
            static_cast<std::size_t>(frame.height[plane]),
        guard);
  }
  return frame;
}

template <typename Sample, typename Fn>
void fill_plane(FrameT<Sample>& frame, int plane, Fn fn) {
  const int width = frame.plane_width(plane);
  const int height = frame.plane_height(plane);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      frame.plane(plane)[frame.index(plane, y, x)] =
          static_cast<Sample>(fn(x, y));
    }
  }
}

template <typename Sample>
bool guard_intact(const FrameT<Sample>& frame) {
  for (int plane = 0; plane < 3; ++plane) {
    const int width = frame.plane_width(plane);
    const int height = frame.plane_height(plane);
    for (int y = 0; y < height + frame.guard_rows; ++y) {
      const int start = (y < height) ? width : 0;
      for (int x = start; x < frame.stride[plane]; ++x) {
        if (frame.plane(plane)[frame.index(plane, y, x)] != frame.guard) {
          return false;
        }
      }
    }
  }
  return true;
}

template <typename Sample>
bool same_planes(const FrameT<Sample>& a, const FrameT<Sample>& b, int* plane,
                 int* px, int* py) {
  for (int p = 0; p < 3; ++p) {
    const std::size_t count = a.data[p].size();
    for (std::size_t i = 0; i < count; ++i) {
      ++g_samples_compared;
      if (a.data[p][i] != b.data[p][i]) {
        if (plane != nullptr) {
          *plane = p;
          *px = static_cast<int>(i % static_cast<std::size_t>(a.stride[p]));
          *py = static_cast<int>(i / static_cast<std::size_t>(a.stride[p]));
        }
        return false;
      }
    }
  }
  return true;
}

template <typename Sample>
std::uint64_t count_changed(const FrameT<Sample>& before,
                            const FrameT<Sample>& after) {
  std::uint64_t changed = 0;
  for (int plane = 0; plane < 3; ++plane) {
    const int width = before.plane_width(plane);
    const int height = before.plane_height(plane);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        if (before.plane(plane)[before.index(plane, y, x)] !=
            after.plane(plane)[after.index(plane, y, x)]) {
          ++changed;
        }
      }
    }
  }
  return changed;
}

// ---------------------------------------------------------------------------
// CPU / device views.
// ---------------------------------------------------------------------------
cpu::TipDeblockFrame cpu_view(FrameT<std::uint8_t>& frame) {
  cpu::TipDeblockFrame view{};
  for (int plane = 0; plane < 3; ++plane) {
    view.planes[plane].data = frame.data[plane].data();
    view.planes[plane].stride = frame.stride[plane];
  }
  view.mi_cols = frame.mi_cols;
  view.mi_rows = frame.mi_rows;
  view.subsampling_x = 1;
  view.subsampling_y = 1;
  view.tile_col_starts =
      frame.tile_cols.empty() ? nullptr : frame.tile_cols.data();
  view.num_tile_cols = static_cast<int>(frame.tile_cols.size());
  view.tile_row_starts =
      frame.tile_rows.empty() ? nullptr : frame.tile_rows.data();
  view.num_tile_rows = static_cast<int>(frame.tile_rows.size());
  return view;
}

template <typename Sample>
ctd::TipDeblockFrameView<Sample> device_view(FrameT<Sample>& frame) {
  ctd::TipDeblockFrameView<Sample> view{};
  for (int plane = 0; plane < 3; ++plane) {
    view.plane[plane] = frame.data[plane].data();
    view.stride[plane] = frame.stride[plane];
  }
  view.mi_cols = frame.mi_cols;
  view.mi_rows = frame.mi_rows;
  view.sub_x = 1;
  view.sub_y = 1;
  view.tile_col_starts =
      frame.tile_cols.empty() ? nullptr : frame.tile_cols.data();
  view.num_tile_cols = static_cast<int>(frame.tile_cols.size());
  view.tile_row_starts =
      frame.tile_rows.empty() ? nullptr : frame.tile_rows.data();
  view.num_tile_rows = static_cast<int>(frame.tile_rows.size());
  return view;
}

// ---------------------------------------------------------------------------
// Options.
// ---------------------------------------------------------------------------
TipDeblockOptions base_options() {
  TipDeblockOptions options;
  options.tip_frame_mode = TipFrameMode::kAsOutput;
  options.apply_deblocking_filter_tip = true;
  options.enable_df_sub_pu = true;
  options.allow_df_sub_pu = true;
  options.disable_loopfilters_across_tiles = false;
  options.base_q_idx = 255;
  options.delta_q_u_ac = 0;
  options.delta_q_v_ac = 0;
  options.base_uv_ac_delta_q = 0;
  options.bit_depth = 8;
  options.enable_tip_refinemv = false;
  options.tip_interp_filter = InterpFilter::kEighttapSharp;
  return options;
}

TipDeblockOptions tip8_options() {
  TipDeblockOptions options = base_options();
  options.enable_tip_refinemv = true;
  options.tip_interp_filter = InterpFilter::kEighttapSharp;
  return options;
}

ctd::TipDeblockDeviceOptions device_options(const TipDeblockOptions& options) {
  ctd::TipDeblockDeviceOptions device;
  device.base_q_idx = options.base_q_idx;
  device.delta_q_u_ac = options.delta_q_u_ac;
  device.delta_q_v_ac = options.delta_q_v_ac;
  device.base_uv_ac_delta_q = options.base_uv_ac_delta_q;
  device.bit_depth = options.bit_depth;
  device.tip_size = static_cast<int>(cpu::tip_deblock_tip_size(options));
  device.disable_loopfilters_across_tiles =
      options.disable_loopfilters_across_tiles ? 1 : 0;
  return device;
}

// ---------------------------------------------------------------------------
// Independent strength transcription (§7.16), covering 8- and 10-bit.
// ---------------------------------------------------------------------------
struct RefPlaneInfo {
  int base_filter_level = 0;
  int q_thr = 0;
  int side = 0;
};

int ref_clip3(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

RefPlaneInfo ref_plane_info(int plane, int base_q_idx, int delta_u,
                            int delta_v, int delta_uv, int bit_depth) {
  int base = base_q_idx;
  if (plane == 1) {
    base += delta_u + delta_uv;
  } else if (plane == 2) {
    base += delta_v + delta_uv;
  }
  int q = base;
  if (!(q == 0)) {
    q = ref_clip3(1, 255, q);
  }
  const int ac = (q < 25)
                     ? av2enc::deblock::kAcQlookup[q]
                     : (av2enc::deblock::kAcQlookup[((q - 1) % 24) + 1]
                        << ((q - 1) / 24));
  RefPlaneInfo info;
  info.base_filter_level = base;
  info.q_thr = ((ac + (1 << 2)) >> 3) >> 6;
  const int q_ind =
      ref_clip3(0, av2enc::deblock::kMaxSideTable - 1,
                base - 24 * (bit_depth - 8));
  const int raw = av2enc::deblock::kSideThresholds[q_ind] +
                  (1 << (12 - bit_depth));
  info.side = std::max(raw, 0) >> (13 - bit_depth);
  return info;
}

void test_tip_size_and_strength() {
  const InterpFilter filters[] = {InterpFilter::kEighttap,
                                  InterpFilter::kEighttapSmooth,
                                  InterpFilter::kEighttapSharp};
  for (InterpFilter filter : filters) {
    for (int refine = 0; refine <= 1; ++refine) {
      TipDeblockOptions options = base_options();
      options.tip_interp_filter = filter;
      options.enable_tip_refinemv = (refine != 0);
      const cpu::TipSize expected =
          (refine != 0 && filter == InterpFilter::kEighttapSharp)
              ? cpu::TipSize::kBlock8x8
              : cpu::TipSize::kBlock16x16;
      CHECK(cpu::tip_deblock_tip_size(options) == expected,
            "CPU tipSize selection");
      CHECK(device_options(options).tip_size ==
                static_cast<int>(expected),
            "device tipSize mapping");
    }
  }

  struct Deltas {
    int u;
    int v;
    int uv;
  };
  const Deltas deltas[] = {{0, 0, 0}, {-37, 23, -11}, {32, -32, 16}};
  const int bit_depths[] = {8, 10};
  for (const Deltas& delta : deltas) {
    for (int bit_depth : bit_depths) {
      TipDeblockOptions options = base_options();
      options.bit_depth = bit_depth;
      options.delta_q_u_ac = delta.u;
      options.delta_q_v_ac = delta.v;
      options.base_uv_ac_delta_q = delta.uv;
      ctd::TipDeblockDeviceOptions device = device_options(options);
      for (int q = 0; q <= 255; ++q) {
        options.base_q_idx = q;
        device.base_q_idx = q;
        for (int plane = 0; plane < 3; ++plane) {
          const RefPlaneInfo expected =
              ref_plane_info(plane, q, delta.u, delta.v, delta.uv, bit_depth);
          const ctd::TipPlaneInfo actual =
              ctd::derive_plane_info(plane, device);
          CHECK(actual.base_filter_level == expected.base_filter_level,
                "strength baseFilterLevel");
          CHECK(actual.q_thr == expected.q_thr, "strength qThr");
          CHECK(actual.side == expected.side, "strength side");
          if (bit_depth == 8) {
            const cpu::TipDeblockPlaneInfo cpu_info =
                cpu::tip_deblock_plane_info(options, plane);
            CHECK(cpu_info.base_filter_level == actual.base_filter_level,
                  "CPU/device baseFilterLevel");
            CHECK(cpu_info.q_thr == actual.q_thr, "CPU/device qThr");
            CHECK(cpu_info.side == actual.side, "CPU/device side");
          }
        }
      }
    }
  }
  std::cout << "  tipSize + strength (all 256 base_q_idx, 8/10-bit)\n";
}

// ---------------------------------------------------------------------------
// Independent 10-bit §7.17.7 transcription.
// ---------------------------------------------------------------------------
int ref10_sample(const std::uint16_t* plane, int stride, int x, int y) {
  return static_cast<int>(
      plane[static_cast<std::ptrdiff_t>(y) * stride + x]);
}

void ref10_store(std::uint16_t* plane, int stride, int x, int y, int value) {
  plane[static_cast<std::ptrdiff_t>(y) * stride + x] =
      static_cast<std::uint16_t>(std::clamp(value, 0, 1023));
}

int ref10_filter_choice(const std::uint16_t* plane, int stride, int x, int y,
                        int q_thr, int side_thr, int dx, int dy,
                        int max_width_neg, int max_width_pos) {
  if (q_thr == 0 || side_thr == 0) {
    return 0;
  }
  const int samples_pos = std::clamp(max_width_pos + 1, 3, 8);
  const int samples_neg = std::clamp(max_width_neg + 1, 3, 8);
  int pos[16] = {};
  int neg[16] = {};
  const int base = 8;
  const int x_end = x + 3 * dy;
  const int y_end = y + 3 * dx;
  for (int d = 0; d < samples_pos; ++d) {
    pos[base + d] = ref10_sample(plane, stride, x + d * dx, y + d * dy);
    neg[base + d] =
        ref10_sample(plane, stride, x_end + d * dx, y_end + d * dy);
  }
  for (int d = 1; d <= samples_neg; ++d) {
    pos[base - d] = ref10_sample(plane, stride, x - d * dx, y - d * dy);
    neg[base - d] =
        ref10_sample(plane, stride, x_end - d * dx, y_end - d * dy);
  }

  int deriv[4] = {};
  for (int d = -2; d <= 1; ++d) {
    const int i = d + 2;
    const int a = pos[base + d - 1];
    const int b = pos[base + d];
    const int c = pos[base + d + 1];
    const int e = neg[base + d - 1];
    const int f = neg[base + d];
    const int g = neg[base + d + 1];
    deriv[i] = (std::abs(a - 2 * b + c) + std::abs(e - 2 * f + g) + 1) >> 1;
  }
  if (deriv[0] > side_thr || deriv[3] > side_thr) {
    return 0;
  }
  if (max_width_pos == 1) {
    return 1;
  }
  if (deriv[0] > (side_thr >> 2) || deriv[3] > (side_thr >> 2)) {
    return 1;
  }
  if (deriv[1] + deriv[2] > q_thr * 4) {
    return 1;
  }
  if (deriv[0] > (side_thr >> 3) || deriv[3] > (side_thr >> 3)) {
    return 2;
  }
  if (deriv[1] + deriv[2] > q_thr * 3) {
    return 2;
  }

  const int end_thr = (side_thr * 3) >> 4;
  if (max_width_neg > 2) {
    const int ds = std::abs(pos[base - 1] - pos[base - 4] -
                            3 * (pos[base - 1] - pos[base - 2]));
    const int dt = std::abs(neg[base - 1] - neg[base - 4] -
                            3 * (neg[base - 1] - neg[base - 2]));
    if (((ds + dt + 1) >> 1) > end_thr) {
      return 2;
    }
  }
  {
    const int ds = std::abs(pos[base] - pos[base + 3] -
                            3 * (pos[base] - pos[base + 1]));
    const int dt = std::abs(neg[base] - neg[base + 3] -
                            3 * (neg[base] - neg[base + 1]));
    if (((ds + dt + 1) >> 1) > end_thr) {
      return 2;
    }
  }
  if (max_width_pos == 3) {
    return 3;
  }

  const int transition = (deriv[1] + deriv[2]) << 4;
  int prev = 3;
  for (int d = 4; d <= max_width_pos; d += 2) {
    if (transition > q_thr * av2enc::deblock::kQFirst[d - 4]) {
      return prev;
    }
    const int reach = std::min(7, d);
    const int end4 = (side_thr * d) >> 4;
    if (max_width_neg >= reach) {
      const int ds = std::abs(pos[base - 1] - pos[base - reach - 1] -
                              reach * (pos[base - 1] - pos[base - 2]));
      const int dt = std::abs(neg[base - 1] - neg[base - reach - 1] -
                              reach * (neg[base - 1] - neg[base - 2]));
      if (((ds + dt + 1) >> 1) > end4) {
        return prev;
      }
    }
    {
      const int ds = std::abs(pos[base] - pos[base + reach] -
                              reach * (pos[base] - pos[base + 1]));
      const int dt = std::abs(neg[base] - neg[base + reach] -
                              reach * (neg[base] - neg[base + 1]));
      if (((ds + dt + 1) >> 1) > end4) {
        return prev;
      }
    }
    prev = d;
  }
  return max_width_pos;
}

void ref10_sample_filtering(std::uint16_t* plane, int stride, int x, int y,
                            int q_thr, int dx, int dy, int max_width_neg,
                            int max_width_pos) {
  const int width = std::max(max_width_neg, max_width_pos);
  const int q0 = ref10_sample(plane, stride, x, y);
  const int q1 = ref10_sample(plane, stride, x + dx, y + dy);
  const int p0 = ref10_sample(plane, stride, x - dx, y - dy);
  const int p1 = ref10_sample(plane, stride, x - 2 * dx, y - 2 * dy);
  const int clamp = q_thr * av2enc::deblock::kQThreshMults[width - 1];
  int delta = 4 * (p1 - q1 + 3 * (q0 - p0));
  delta = std::clamp(delta, -clamp, clamp);
  const int scaled_neg = delta * av2enc::deblock::kWMult[max_width_neg - 1];
  const int scaled_pos = delta * av2enc::deblock::kWMult[max_width_pos - 1];
  for (int i = 0; i < width; ++i) {
    const int diff_pos =
        (scaled_pos * (max_width_pos - i) + (1 << 10)) >> 11;
    const int diff_neg =
        (scaled_neg * (max_width_neg - i) + (1 << 10)) >> 11;
    const int qx = x + i * dx;
    const int qy = y + i * dy;
    ref10_store(plane, stride, qx, qy,
                ref10_sample(plane, stride, qx, qy) - diff_pos);
    if (i < max_width_neg) {
      const int px = x - (i + 1) * dx;
      const int py = y - (i + 1) * dy;
      ref10_store(plane, stride, px, py,
                  ref10_sample(plane, stride, px, py) + diff_neg);
    }
  }
}

void ref10_edge(FrameT<std::uint16_t>& frame, const ctd::TipDeblockConfig& cfg,
                const ctd::TipDeblockDeviceOptions& options, int plane,
                int pass, int x, int y) {
  const int sub_x = (plane != 0) ? cfg.subsampling_x : 0;
  const int sub_y = (plane != 0) ? cfg.subsampling_y : 0;
  const int sw = options.tip_size >> sub_x;
  const int sh = options.tip_size >> sub_y;
  const RefPlaneInfo info =
      ref_plane_info(plane, options.base_q_idx, options.delta_q_u_ac,
                     options.delta_q_v_ac, options.base_uv_ac_delta_q,
                     options.bit_depth);
  if (pass == 0) {
    if (x <= 0 || (x % sw) != 0) {
      return;
    }
    const bool tile =
        av2enc::deblock::is_tile_start(frame.tile_cols.data(),
                                       static_cast<int>(frame.tile_cols.size()),
                                       (x << sub_x) >> 2);
    if (tile && options.disable_loopfilters_across_tiles != 0) {
      return;
    }
    int pos = 0;
    int neg = 0;
    av2enc::deblock::filter_max_width(plane, sw, tile ? 1 : 0, &pos, &neg);
    const int width = ref10_filter_choice(frame.plane(plane),
                                          frame.stride[plane], x, y, info.q_thr,
                                          info.side, 1, 0, neg, pos);
    if (width <= 0) {
      return;
    }
    for (int i = 0; i < 4; ++i) {
      ref10_sample_filtering(frame.plane(plane), frame.stride[plane], x, y + i,
                             info.q_thr, 1, 0, std::min(width, neg),
                             std::min(width, pos));
    }
    return;
  }
  if (y <= 0 || (y % sh) != 0) {
    return;
  }
  const bool tile =
      av2enc::deblock::is_tile_start(frame.tile_rows.data(),
                                     static_cast<int>(frame.tile_rows.size()),
                                     (y << sub_y) >> 2);
  if (tile && options.disable_loopfilters_across_tiles != 0) {
    return;
  }
  int pos = 0;
  int neg = 0;
  const int horz64 = (((y << sub_y) % 64) == 0) ? 1 : 0;
  av2enc::deblock::filter_max_width(plane, sh, horz64, &pos, &neg);
  const int width = ref10_filter_choice(frame.plane(plane), frame.stride[plane],
                                        x, y, info.q_thr, info.side, 0, 1,
                                        neg, pos);
  if (width <= 0) {
    return;
  }
  for (int i = 0; i < 4; ++i) {
    ref10_sample_filtering(frame.plane(plane), frame.stride[plane], x + i, y,
                           info.q_thr, 0, 1, std::min(width, neg),
                           std::min(width, pos));
  }
}

void ref10_frame(FrameT<std::uint16_t>& frame,
                 const ctd::TipDeblockDeviceOptions& options) {
  ctd::TipDeblockConfig cfg;
  cfg.mi_cols = frame.mi_cols;
  cfg.mi_rows = frame.mi_rows;
  cfg.bit_depth = 10;
  for (int plane = 0; plane < 3; ++plane) {
    const int sw = options.tip_size >> ((plane == 0) ? 0 : 1);
    const int sh = options.tip_size >> ((plane == 0) ? 0 : 1);
    const int width = frame.plane_width(plane);
    const int height = frame.plane_height(plane);
    for (int y = 0; y < height; y += 4) {
      for (int x = 0; x < width; x += sw) {
        if (x > 0) {
          ref10_edge(frame, cfg, options, plane, 0, x, y);
        }
      }
    }
    for (int x = 0; x < width; x += 4) {
      for (int y = 0; y < height; y += sh) {
        if (y > 0) {
          ref10_edge(frame, cfg, options, plane, 1, x, y);
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Sample-kernel parity (independent of the edge traversal).
// ---------------------------------------------------------------------------
void test_sample_kernels() {
  std::mt19937 rng(0x7160716u);
  for (int iteration = 0; iteration < 3000; ++iteration) {
    FrameT<std::uint8_t> a = make_frame<std::uint8_t>(8, 8, 0xEEu);
    FrameT<std::uint16_t> b = make_frame<std::uint16_t>(8, 8, 0x3EEu);
    fill_plane(a, 0, [&rng](int, int) { return random_range(rng, 0, 255); });
    fill_plane(b, 0, [&rng](int, int) { return random_range(rng, 0, 1023); });

    const int pos_set[] = {1, 3, 4, 6, 8};
    const int neg_set[] = {1, 2, 3, 4, 6, 8};
    const int pos = pos_set[rng() % 5];
    const int neg = neg_set[rng() % 6];
    const int q_thr = random_range(rng, 0, 400);
    const int side = random_range(rng, 0, 4000);
    const int x = random_range(rng, 8, 24);
    const int y = random_range(rng, 8, 16);
    const int dx = (rng() % 2) != 0 ? 1 : 0;
    const int dy = (dx == 0) ? 1 : 0;

    // 8-bit: device header vs the §7.17 CPU kernel it must match bit-exactly.
    const int width8 = ctd::tip_filter_choice<std::uint8_t, 8>(
        a.plane(0), a.stride[0], x, y, q_thr, side, dx, dy, neg, pos);
    const int ref_width8 = av2enc::deblock::filter_choice(
        a.plane(0), a.stride[0], x, y, q_thr, side, dx, dy, neg, pos);
    CHECK(width8 == ref_width8, "8-bit filter_choice matches §7.17 kernel");
    if (width8 > 0) {
      FrameT<std::uint8_t> copy = a;
      FrameT<std::uint8_t> kernel_copy = a;
      av2enc::deblock::sample_filtering(
          copy.plane(0), copy.stride[0], x, y, q_thr, dx, dy,
          std::min(width8, neg), std::min(width8, pos), 0, 0, 8);
      ctd::tip_sample_filtering<std::uint8_t, 8>(
          kernel_copy.plane(0), kernel_copy.stride[0], x, y, q_thr, dx, dy,
          std::min(width8, neg), std::min(width8, pos), 0, 0);
      for (std::size_t i = 0; i < copy.data[0].size(); ++i) {
        ++g_samples_compared;
        CHECK(copy.data[0][i] == kernel_copy.data[0][i],
              "8-bit sample_filtering matches §7.17 kernel");
        if (copy.data[0][i] != kernel_copy.data[0][i]) {
          break;
        }
      }
    }

    // 10-bit: device header vs the independent transcription.
    const int width10 = ctd::tip_filter_choice<std::uint16_t, 10>(
        b.plane(0), b.stride[0], x, y, q_thr, side, dx, dy, neg, pos);
    const int ref_width10 = ref10_filter_choice(
        b.plane(0), b.stride[0], x, y, q_thr, side, dx, dy, neg, pos);
    CHECK(width10 == ref_width10, "10-bit filter_choice matches reference");
    if (width10 > 0) {
      FrameT<std::uint16_t> copy = b;
      FrameT<std::uint16_t> kernel_copy = b;
      ref10_sample_filtering(copy.plane(0), copy.stride[0], x, y, q_thr, dx,
                             dy, std::min(width10, neg),
                             std::min(width10, pos));
      ctd::tip_sample_filtering<std::uint16_t, 10>(
          kernel_copy.plane(0), kernel_copy.stride[0], x, y, q_thr, dx, dy,
          std::min(width10, neg), std::min(width10, pos), 0, 0);
      bool same = true;
      for (std::size_t i = 0; i < copy.data[0].size(); ++i) {
        ++g_samples_compared;
        if (copy.data[0][i] != kernel_copy.data[0][i]) {
          same = false;
          break;
        }
      }
      CHECK(same, "10-bit sample_filtering matches reference");
    }
  }
  std::cout << "  sample kernels (3000 randomized windows, 8/10-bit)\n";
}

// ---------------------------------------------------------------------------
// All base_q_idx x both tip sizes x all planes/passes, CPU vs device.
// ---------------------------------------------------------------------------
void fill_structured(FrameT<std::uint8_t>& frame) {
  std::mt19937 rng(0x1616u + static_cast<unsigned>(frame.mi_cols * 31 +
                                                   frame.mi_rows));
  for (int plane = 0; plane < 3; ++plane) {
    fill_plane(frame, plane, [&rng](int x, int y) {
      const int wobble = random_range(rng, -6, 6);
      const int base = ((x / 3 + y / 5) & 1) ? 80 : 176;
      return std::clamp(base + wobble + (x % 7) * 2, 0, 255);
    });
  }
  // Deliberate discontinuities on the visited lattice (both tip sizes and both
  // chroma phases land on these).
  for (int plane = 0; plane < 3; ++plane) {
    const int width = frame.plane_width(plane);
    const int height = frame.plane_height(plane);
    const int split_x = width / 2;
    const int split_y = height / 2;
    fill_plane(frame, plane, [split_x, split_y](int x, int y) {
      if (x >= split_x) {
        return 200;
      }
      if (y >= split_y) {
        return 48;
      }
      return 112;
    });
  }
}

void compare_frame_pair(FrameT<std::uint8_t>& cpu_frame,
                        FrameT<std::uint8_t>& device_frame, int line) {
  int plane = 0;
  int px = 0;
  int py = 0;
  const bool same = same_planes(cpu_frame, device_frame, &plane, &px, &py);
  CHECK(same, "CPU/device plane contents match");
  if (!same) {
    std::fprintf(stderr, "  first diff: test line %d plane %d (%d,%d)\n",
                 line, plane, px, py);
  }
}

void test_all_q_plane_passes() {
  std::uint64_t changed_total = 0;
  struct DeltaSet {
    int u;
    int v;
    int uv;
    bool disable_tiles;
  };
  const DeltaSet sets[] = {{0, 0, 0, false},
                           {-37, 23, -11, false},
                           {12, -9, 5, true}};
  for (const DeltaSet& set : sets) {
    for (int tip8 = 0; tip8 <= 1; ++tip8) {
      for (int q = 0; q <= 255; ++q) {
        const FrameT<std::uint8_t> base = [&] {
          FrameT<std::uint8_t> frame = make_frame<std::uint8_t>(10, 8, 0xEEu);
          fill_structured(frame);
          frame.tile_cols = {0, 5};
          frame.tile_rows = {0, 3};
          return frame;
        }();
        FrameT<std::uint8_t> cpu_frame = base;
        FrameT<std::uint8_t> device_frame = base;
        TipDeblockOptions options = tip8 ? tip8_options() : base_options();
        options.base_q_idx = q;
        options.delta_q_u_ac = set.u;
        options.delta_q_v_ac = set.v;
        options.base_uv_ac_delta_q = set.uv;
        options.disable_loopfilters_across_tiles = set.disable_tiles;
        const ctd::TipDeblockDeviceOptions device = device_options(options);

        cpu::TipDeblockFrame cpu_ref = cpu_view(cpu_frame);
        ctd::TipDeblockFrameView<std::uint8_t> device_ref =
            device_view(device_frame);

        // Interleave the CPU and device calls pass by pass: the device result
        // must match the CPU after every single plane/pass, not just the end.
        for (int plane = 0; plane < 3; ++plane) {
          for (int pass = 0; pass < 2; ++pass) {
            const TipDeblockStatus status =
                cpu::tip_deblock_plane_pass(cpu_ref, options, plane, pass);
            CHECK(status == TipDeblockStatus::kOk, "CPU plane/pass status");
            ctd::tip_plane_pass_view<std::uint8_t, 8>(device_ref, device, plane,
                                                      pass);
            compare_frame_pair(cpu_frame, device_frame, __LINE__);
          }
        }
        CHECK(guard_intact(device_frame), "guards intact after plane/pass");
        changed_total += count_changed(base, cpu_frame);
      }
    }
  }
  g_samples_changed += changed_total;
  std::cout << "  all 256 base_q_idx x 2 tips x 3 planes x 2 passes vs CPU ("
            << changed_total << " filtered samples)\n";
}

// ---------------------------------------------------------------------------
// Full-frame parity, including the 64x64-grid sbEdge clamp.
// ---------------------------------------------------------------------------
void test_full_frame_all_q() {
  std::uint64_t changed_total = 0;
  for (int tip8 = 0; tip8 <= 1; ++tip8) {
    for (int disable = 0; disable <= 1; ++disable) {
      for (int q = 0; q <= 255; ++q) {
        FrameT<std::uint8_t> base = make_frame<std::uint8_t>(12, 32, 0xEEu);
        fill_structured(base);
        base.tile_cols = {0, 4};
        base.tile_rows = {0, 9};
        FrameT<std::uint8_t> cpu_frame = base;
        FrameT<std::uint8_t> device_frame = base;
        TipDeblockOptions options = tip8 ? tip8_options() : base_options();
        options.base_q_idx = q;
        options.delta_q_u_ac = q % 17 - 8;
        options.delta_q_v_ac = 7 - q % 13;
        options.base_uv_ac_delta_q = q % 5 - 2;
        options.disable_loopfilters_across_tiles = (disable != 0);

        cpu::TipDeblockFrame cpu_ref = cpu_view(cpu_frame);
        const TipDeblockStatus status =
            cpu::tip_deblock_frame(cpu_ref, options);
        CHECK(status == TipDeblockStatus::kOk, "CPU full-frame status");
        ctd::tip_frame_view<std::uint8_t, 8>(device_view(device_frame),
                                             device_options(options));
        compare_frame_pair(cpu_frame, device_frame, __LINE__);
        CHECK(guard_intact(device_frame), "guards intact after full frame");
        changed_total += count_changed(base, cpu_frame);
      }
    }
  }
  g_samples_changed += changed_total;
  std::cout << "  full frame (256 q x 2 tips x tile gating, 64-grid sbEdge, "
            << changed_total << " filtered samples)\n";
}

// ---------------------------------------------------------------------------
// Random frames, 8-bit CPU parity and 10-bit reference parity.
// ---------------------------------------------------------------------------
template <typename Sample>
void randomize_frame(FrameT<Sample>& frame, std::mt19937& rng, int max_value) {
  for (int plane = 0; plane < 3; ++plane) {
    fill_plane(frame, plane, [&rng, max_value](int x, int y) {
      const std::uint32_t noise =
          static_cast<std::uint32_t>(random_range(rng, 0, max_value));
      const int texture = ((x * 3) ^ (y * 5)) & 31;
      const int value = static_cast<int>(noise) + texture - 16;
      return std::clamp(value, 0, max_value);
    });
  }
  const int width = frame.plane_width(0);
  const int height = frame.plane_height(0);
  if (width > 16) {
    const int split = 8 + 8 * random_range(rng, 0, 1);
    fill_plane(frame, 0, [split, max_value](int x, int) {
      return (x < split) ? 24 : max_value - 24;
    });
  }
  if (height > 16) {
    const int split = 8 + 8 * random_range(rng, 0, 1);
    fill_plane(frame, 1, [split, max_value](int, int y) {
      return (y < split) ? max_value - 24 : 24;
    });
  }
}

void test_random_8bit() {
  std::mt19937 rng(0x7F16u);
  std::uint64_t changed = 0;
  for (int iteration = 0; iteration < 120; ++iteration) {
    const int mi_cols = 8 + 2 * random_range(rng, 0, 8);
    const int mi_rows = 8 + 2 * random_range(rng, 0, 8);
    FrameT<std::uint8_t> base = make_frame<std::uint8_t>(mi_cols, mi_rows, 0xEEu);
    randomize_frame(base, rng, 255);
    base.tile_cols = {0};
    base.tile_rows = {0};
    if ((rng() % 3) == 0) {
      base.tile_cols.push_back(static_cast<std::uint16_t>(
          1 + random_range(rng, 0, mi_cols - 2)));
    }
    if ((rng() % 3) == 0) {
      base.tile_rows.push_back(static_cast<std::uint16_t>(
          1 + random_range(rng, 0, mi_rows - 2)));
    }
    std::sort(base.tile_cols.begin(), base.tile_cols.end());
    std::sort(base.tile_rows.begin(), base.tile_rows.end());

    const bool tip8 = (rng() % 2) != 0;
    TipDeblockOptions options = tip8 ? tip8_options() : base_options();
    options.base_q_idx = random_range(rng, 0, 255);
    options.delta_q_u_ac = random_range(rng, -32, 32);
    options.delta_q_v_ac = random_range(rng, -32, 32);
    options.base_uv_ac_delta_q = random_range(rng, -16, 16);
    options.disable_loopfilters_across_tiles = (rng() % 2) != 0;

    FrameT<std::uint8_t> cpu_frame = base;
    FrameT<std::uint8_t> device_frame = base;
    cpu::TipDeblockFrame cpu_ref = cpu_view(cpu_frame);
    const TipDeblockStatus status = cpu::tip_deblock_frame(cpu_ref, options);
    CHECK(status == TipDeblockStatus::kOk, "random CPU status");
    ctd::tip_frame_view<std::uint8_t, 8>(device_view(device_frame),
                                         device_options(options));
    compare_frame_pair(cpu_frame, device_frame, __LINE__);
    CHECK(guard_intact(device_frame), "random guards intact");
    changed += count_changed(base, cpu_frame);

    // Determinism of the shared core.
    FrameT<std::uint8_t> rerun = base;
    ctd::tip_frame_view<std::uint8_t, 8>(device_view(rerun),
                                         device_options(options));
    int plane = 0;
    int px = 0;
    int py = 0;
    (void)plane;
    (void)px;
    (void)py;
    CHECK(rerun.data[0] == device_frame.data[0] &&
              rerun.data[1] == device_frame.data[1] &&
              rerun.data[2] == device_frame.data[2],
          "random device run is deterministic");
  }
  g_samples_changed += changed;
  std::cout << "  random 8-bit frames (120, CPU parity, " << changed
            << " samples changed)\n";
}

void test_random_10bit() {
  std::mt19937 rng(0x10B17u);
  std::uint64_t changed = 0;
  for (int iteration = 0; iteration < 60; ++iteration) {
    const int mi_cols = 8 + 2 * random_range(rng, 0, 8);
    const int mi_rows = 8 + 2 * random_range(rng, 0, 8);
    FrameT<std::uint16_t> base =
        make_frame<std::uint16_t>(mi_cols, mi_rows, 0x3EEu);
    randomize_frame(base, rng, 1023);
    base.tile_cols = {0};
    base.tile_rows = {0};
    if ((rng() % 3) == 0) {
      base.tile_cols.push_back(static_cast<std::uint16_t>(
          1 + random_range(rng, 0, mi_cols - 2)));
    }
    if ((rng() % 3) == 0) {
      base.tile_rows.push_back(static_cast<std::uint16_t>(
          1 + random_range(rng, 0, mi_rows - 2)));
    }
    std::sort(base.tile_cols.begin(), base.tile_cols.end());
    std::sort(base.tile_rows.begin(), base.tile_rows.end());

    const bool tip8 = (rng() % 2) != 0;
    TipDeblockOptions options = tip8 ? tip8_options() : base_options();
    options.bit_depth = 10;
    options.base_q_idx = random_range(rng, 0, 255);
    options.delta_q_u_ac = random_range(rng, -32, 32);
    options.delta_q_v_ac = random_range(rng, -32, 32);
    options.base_uv_ac_delta_q = random_range(rng, -16, 16);
    options.disable_loopfilters_across_tiles = (rng() % 2) != 0;

    FrameT<std::uint16_t> device_frame = base;
    FrameT<std::uint16_t> ref_frame = base;
    ctd::tip_frame_view<std::uint16_t, 10>(device_view(device_frame),
                                           device_options(options));
    ref10_frame(ref_frame, device_options(options));
    CHECK(same_planes(ref_frame, device_frame, nullptr, nullptr, nullptr),
          "10-bit device matches independent reference");
    CHECK(guard_intact(device_frame), "10-bit guards intact");
    changed += count_changed(base, device_frame);
  }
  g_samples_changed += changed;
  std::cout << "  random 10-bit frames (60, reference parity, " << changed
            << " samples changed)\n";
}

// ---------------------------------------------------------------------------
// All base_q_idx on a 10-bit frame, both tips.
// ---------------------------------------------------------------------------
void test_all_q_10bit() {
  std::uint64_t changed_total = 0;
  const FrameT<std::uint16_t> base = [] {
    FrameT<std::uint16_t> frame =
        make_frame<std::uint16_t>(10, 16, 0x3EEu);
    std::mt19937 rng(0x10A11u);
    randomize_frame(frame, rng, 1023);
    frame.tile_cols = {0, 5};
    frame.tile_rows = {0, 4};
    return frame;
  }();
  for (int tip8 = 0; tip8 <= 1; ++tip8) {
    for (int q = 0; q <= 255; ++q) {
      TipDeblockOptions options = tip8 ? tip8_options() : base_options();
      options.bit_depth = 10;
      options.base_q_idx = q;
      options.delta_q_u_ac = q % 19 - 9;
      options.delta_q_v_ac = 11 - q % 23;
      options.base_uv_ac_delta_q = q % 7 - 3;
      FrameT<std::uint16_t> device_frame = base;
      FrameT<std::uint16_t> ref_frame = base;
      ctd::tip_frame_view<std::uint16_t, 10>(device_view(device_frame),
                                             device_options(options));
      ref10_frame(ref_frame, device_options(options));
      CHECK(same_planes(ref_frame, device_frame, nullptr, nullptr, nullptr),
            "10-bit all-q parity");
      CHECK(guard_intact(device_frame), "10-bit all-q guards intact");
      changed_total += count_changed(base, device_frame);
    }
  }
  g_samples_changed += changed_total;
  std::cout << "  10-bit all 256 base_q_idx x 2 tips vs reference ("
            << changed_total << " filtered samples)\n";
}

// ---------------------------------------------------------------------------
// Arena sizing plan (host side; executed on the GPU box by the .cu test too).
// ---------------------------------------------------------------------------
void test_arena_plan() {
  ctd::TipDeblockConfig config;
  config.mi_cols = 20;
  config.mi_rows = 12;
  config.bit_depth = 8;
  config.num_tile_cols = 2;
  config.num_tile_rows = 2;
  const std::size_t plane0 =
      ctd::tip_align_up(ctd::tip_deblock_plane_bytes(config, 0));
  const std::size_t plane1 =
      ctd::tip_align_up(ctd::tip_deblock_plane_bytes(config, 1));
  const std::size_t plane2 =
      ctd::tip_align_up(ctd::tip_deblock_plane_bytes(config, 2));
  CHECK(plane0 >= static_cast<std::size_t>(80 * 48), "luma plane bytes");
  CHECK(plane1 >= static_cast<std::size_t>(40 * 24), "chroma plane bytes");
  CHECK(plane2 == plane1, "symmetric chroma plane bytes");
  const std::size_t tiles =
      ctd::tip_align_up(2 * sizeof(std::uint16_t)) * 2;
  CHECK(ctd::tip_deblock_required_device_bytes(config) ==
            plane0 + plane1 + plane2 + tiles,
        "required bytes = planes + tile starts + scratch");
  CHECK(ctd::tip_deblock_scratch_bytes(config) == 0, "in-place scratch is 0");

  ctd::TipDeblockConfig ten = config;
  ten.bit_depth = 10;
  CHECK(ctd::tip_deblock_required_device_bytes(ten) >
            ctd::tip_deblock_required_device_bytes(config),
        "10-bit doubles plane footprint");

  ctd::TipDeblockConfig strided = config;
  strided.stride[0] = 96;
  CHECK(ctd::tip_deblock_required_device_bytes(strided) >
            ctd::tip_deblock_required_device_bytes(config),
        "stride override increases the plan");

  CHECK(ctd::tip_deblock_plane_width(config, 0) == 80, "plane width luma");
  CHECK(ctd::tip_deblock_plane_width(config, 1) == 40, "plane width chroma");
  CHECK(ctd::tip_deblock_plane_height(config, 0) == 48, "plane height luma");
  CHECK(ctd::tip_deblock_plane_stride(strided, 0) == 96, "stride override");
  CHECK(ctd::tip_deblock_plane_stride(config, 0) == 80, "derived stride");
  std::cout << "  arena plan (" << ctd::tip_deblock_required_device_bytes(config)
            << " bytes at 80x48 8-bit, 2+2 tile starts)\n";
}

// ---------------------------------------------------------------------------
// Real reconstructed frame (decoded.y4m), when present.
// ---------------------------------------------------------------------------
bool load_y4m_first_frame(const std::string& path,
                          FrameT<std::uint8_t>* out) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return false;
  }
  std::string line;
  if (!std::getline(in, line) || line.rfind("YUV4MPEG2", 0) != 0) {
    return false;
  }
  std::istringstream header(line);
  std::string token;
  int width = 0;
  int height = 0;
  bool c420 = false;
  while (header >> token) {
    if (token.size() > 1) {
      const char tag = token[0];
      const std::string value = token.substr(1);
      if (tag == 'W') width = std::stoi(value);
      if (tag == 'H') height = std::stoi(value);
      if (tag == 'C' && value == "420") c420 = true;
    }
  }
  if (width <= 0 || height <= 0 || !c420 || (width % 8) != 0 ||
      (height % 8) != 0) {
    return false;
  }
  if (!std::getline(in, line) || line.rfind("FRAME", 0) != 0) {
    return false;
  }
  FrameT<std::uint8_t> frame =
      make_frame<std::uint8_t>(width / 4, height / 4, 0xEEu);
  std::vector<std::uint8_t> row(static_cast<std::size_t>(width));
  for (int y = 0; y < height; ++y) {
    in.read(reinterpret_cast<char*>(row.data()), width);
    if (!in) {
      return false;
    }
    std::memcpy(frame.plane(0) + frame.index(0, y, 0), row.data(),
                static_cast<std::size_t>(width));
  }
  row.assign(static_cast<std::size_t>(width / 2), 0);
  for (int y = 0; y < height / 2; ++y) {
    in.read(reinterpret_cast<char*>(row.data()), width / 2);
    if (!in) {
      return false;
    }
    std::memcpy(frame.plane(1) + frame.index(1, y, 0), row.data(),
                static_cast<std::size_t>(width / 2));
  }
  for (int y = 0; y < height / 2; ++y) {
    in.read(reinterpret_cast<char*>(row.data()), width / 2);
    if (!in) {
      return false;
    }
    std::memcpy(frame.plane(2) + frame.index(2, y, 0), row.data(),
                static_cast<std::size_t>(width / 2));
  }
  *out = std::move(frame);
  return true;
}

void test_real_frame(const std::string& repo_root) {
  FrameT<std::uint8_t> loaded;
  if (!load_y4m_first_frame(repo_root + "/decoded.y4m", &loaded)) {
    std::cout << "  real frame: skipped (decoded.y4m not usable)\n";
    return;
  }
  const std::uint64_t changed_total =
      [&] {
        std::uint64_t total = 0;
        for (int tip8 = 0; tip8 <= 1; ++tip8) {
          for (int q : {0, 64, 128, 255}) {
            FrameT<std::uint8_t> cpu_frame = loaded;
            FrameT<std::uint8_t> device_frame = loaded;
            TipDeblockOptions options =
                tip8 ? tip8_options() : base_options();
            options.base_q_idx = q;
            cpu::TipDeblockFrame cpu_ref = cpu_view(cpu_frame);
            const TipDeblockStatus status =
                cpu::tip_deblock_frame(cpu_ref, options);
            CHECK(status == TipDeblockStatus::kOk, "real frame CPU status");
            ctd::tip_frame_view<std::uint8_t, 8>(device_view(device_frame),
                                                 device_options(options));
            compare_frame_pair(cpu_frame, device_frame, __LINE__);
            CHECK(guard_intact(device_frame), "real frame guards intact");
            total += count_changed(loaded, cpu_frame);
          }
        }
        return total;
      }();
  g_samples_changed += changed_total;
  std::cout << "  real frame 640x360 (8 q/tip combinations, " << changed_total
            << " changed samples)\n";
}

}  // namespace

int main(int argc, char** argv) {
  const std::string repo_root = (argc > 1) ? argv[1] : ".";
  std::cout << "AV2 §7.16 TIP deblocking CUDA emulation parity\n";
  test_tip_size_and_strength();
  test_sample_kernels();
  test_all_q_plane_passes();
  test_full_frame_all_q();
  test_random_8bit();
  test_random_10bit();
  test_all_q_10bit();
  test_arena_plan();
  test_real_frame(repo_root);
  std::cout << "\nchecks: " << g_checks << "  failures: " << g_failures
            << "  samples compared: " << g_samples_compared
            << "  filtered samples: " << g_samples_changed << "\n";
  return (g_failures == 0) ? 0 : 1;
}

// CPU-vs-GPU parity test for the M1 chroma decision/prediction module.
//
// Two layers:
//   1. synthetic: random reconstruction/source states over all M1 chroma block
//      sizes, partial coded bounds, SB-top lines and ds_filter values, scored
//      against the CPU components (av2enc::intra::prepare_edges + predict_intra for DC,
//      cfl::compute_luma/derive_implicit_alpha/apply_prediction for CfL).
//   2. real frames: run the actual CPU M1 encoder block loop on 64x64 and
//      640x360 frames and compare, block by block, the GPU decision/prediction
//      against m1_still.cpp's `choose_uv_mode`.
//
// The real-frame layer includes m1_still.cpp with `private` made public for the
// test translation unit only (the file itself is not edited). It replays
// `Encoder::encode()` block by block using the encoder's own methods, and at
// the end asserts the replayed reconstruction equals an independent
// `encode_m1_still_picture`/`Encoder::encode()` run, so a harness divergence
// cannot silently pass.

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Pre-include everything m1_still.cpp includes so `#define private public`
// cannot reach the standard library's own headers.
#include "encoder/cfl/cfl.h"
#include "encoder/filters/deblock/deblock.h"
#include "encoder/intra_modes/intra_modes.h"
#include "encoder/m0_cdf_tables.h"
#include "encoder/m0_tables.h"
#include "encoder/m1/m1_cdf_tables.h"
#include "encoder/m1/m1_headers.h"
#include "encoder/m1/m1_still.h"
#include "encoder/m1/m1_syntax.h"
#include "encoder/m1/m1_tables.h"
#include "encoder/partition/partition.h"
#include "encoder/tx_variants/tx_variants.h"
#include "entropy/av2/av2_entropy.h"
#include "io/y4m.h"

#define private public
#include "encoder/m1/m1_still.cpp"
#undef private

#include "cuda/m1/chroma/m1_chroma.h"

namespace {

using av2enc::cuda::m1::M1ChromaJob;
using av2enc::cuda::m1::M1ChromaOptions;
using av2enc::cuda::m1::M1ChromaResult;

#define CUDA_CHECK(expr)                                                    \
  do {                                                                      \
    const cudaError_t status_ = (expr);                                     \
    if (status_ != cudaSuccess) {                                           \
      std::fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #expr,  \
                   cudaGetErrorString(status_));                            \
      std::exit(1);                                                         \
    }                                                                       \
  } while (0)

struct Rng {
  std::uint64_t state;
  explicit Rng(std::uint64_t seed) : state(seed == 0 ? 1 : seed) {}
  std::uint64_t next() {
    std::uint64_t x = state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    state = x;
    return x;
  }
  int range(int n) { return n <= 0 ? 0 : static_cast<int>(next() % static_cast<std::uint64_t>(n)); }
};

int clamp3(int lo, int hi, int v) { return v < lo ? lo : (v > hi ? hi : v); }

// ---------------------------------------------------------------------------
// Failure accounting
// ---------------------------------------------------------------------------
struct Stats {
  long long blocks = 0;
  long long cfl_blocks = 0;
  long long failures = 0;
  long long decision_mismatch = 0;
  long long alpha_mismatch = 0;
  long long sse_mismatch = 0;
  long long pred_mismatch = 0;
  long long max_sse_delta = 0;
  long long max_alpha_delta = 0;
  long long max_pred_delta = 0;
  // Geometry coverage (real-frame layers).
  long long partial_blocks = 0;   // block extends past the coded plane
  long long sb_boundary_blocks = 0;  // x or y on a 32-sample chroma SB edge
  long long non_top_sb_blocks = 0;   // sb_top_y > 0 (CfL SB-top clamp)
  long long negative_alpha_blocks = 0;

  void note_failure() { ++failures; }
};

bool check_int(const char* what, long long got, long long want, Stats& s,
               int cx, int cy, int w, int h) {
  const long long delta = got > want ? got - want : want - got;
  if (delta != 0) {
    if (s.failures < 10) {
      std::fprintf(stderr,
                   "  MISMATCH %s block x=%d y=%d %dx%d got=%lld want=%lld\n",
                   what, cx, cy, w, h, got, want);
    }
    s.note_failure();
    return false;
  }
  return true;
}

bool check_bytes(const char* what, const std::uint8_t* got,
                 const std::uint8_t* want, int count, int width, int height,
                 int cx, int cy, Stats& s) {
  int first = -1;
  int max_delta = 0;
  for (int i = 0; i < count; ++i) {
    const int d = std::abs(static_cast<int>(got[i]) - static_cast<int>(want[i]));
    if (d != 0 && first < 0) first = i;
    max_delta = std::max(max_delta, d);
  }
  if (first >= 0) {
    if (s.failures < 10) {
      std::fprintf(stderr,
                   "  MISMATCH %s block x=%d y=%d %dx%d first=(%d,%d) "
                   "got=%u want=%u max|d|=%d\n",
                   what, cx, cy, width, height, first % width, first / width,
                   static_cast<unsigned>(got[first]),
                   static_cast<unsigned>(want[first]), max_delta);
    }
    s.max_pred_delta = std::max<long long>(s.max_pred_delta, max_delta);
    s.note_failure();
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// CPU reference for one synthetic block: the same component calls the CPU M1
// path makes, driven by a hand-built state.
// ---------------------------------------------------------------------------
struct CpuChroma {
  int dc_u = 0;
  int dc_v = 0;
  int alpha_u = 0;
  int alpha_v = 0;
  bool is_cfl = false;
  long long sse_dc_u = 0;
  long long sse_dc_v = 0;
  long long sse_cfl_u = 0;
  long long sse_cfl_v = 0;
  std::vector<std::uint8_t> pred_u;
  std::vector<std::uint8_t> pred_v;
};

long long sse_plane(const std::vector<std::uint8_t>& pred,
                    const std::vector<std::uint8_t>& src, int w, int h) {
  long long sse = 0;
  for (int i = 0; i < w * h; ++i) {
    const int d = static_cast<int>(src[static_cast<std::size_t>(i)]) -
                  static_cast<int>(pred[static_cast<std::size_t>(i)]);
    sse += static_cast<long long>(d) * d;
  }
  return sse;
}

std::uint8_t source_clamped(const std::vector<std::uint8_t>& plane, int stride,
                            int max_x, int max_y, int x, int y) {
  return plane[static_cast<std::size_t>(clamp3(0, max_y, y)) *
                   static_cast<std::size_t>(stride) +
               static_cast<std::size_t>(clamp3(0, max_x, x))];
}

CpuChroma cpu_block(int w, int h, int x, int y, int coded_w, int coded_h,
                    int sb_top_y, int luma_max_x, int luma_max_y,
                    const std::vector<std::uint8_t>& luma, int luma_stride,
                    const std::vector<std::uint8_t>& chroma_u,
                    const std::vector<std::uint8_t>& chroma_v, int chroma_stride,
                    const std::vector<std::uint8_t>& src_u,
                    const std::vector<std::uint8_t>& src_v, int src_stride,
                    int src_max_x, int src_max_y, int ds_filter,
                    int num4_above_right, int num4_below_left,
                    bool enabled_cfl) {
  CpuChroma out;
  out.pred_u.assign(static_cast<std::size_t>(w) * h, 0);
  out.pred_v.assign(static_cast<std::size_t>(w) * h, 0);

  // --- DC prediction through the real CPU intra path -----------------------
  const auto prepare = [&](const std::uint8_t* plane,
                           av2enc::intra::IntraEdges& edges) {
    av2enc::intra::NeighborRequest request;
    request.recon = av2enc::intra::PlaneView{plane, chroma_stride};
    request.x = x;
    request.y = y;
    request.w = w;
    request.h = h;
    request.max_x = coded_w - 1;
    request.max_y = coded_h - 1;
    request.have_left = x > 0;
    request.have_above = y > 0;
    request.num4_above_right = num4_above_right;
    request.num4_below_left = num4_below_left;
    request.color_plane = 1;
    return av2enc::intra::prepare_edges(request, 8, edges);
  };
  av2enc::intra::PredictParams params;
  params.mode = av2enc::intra::Mode::kDc;
  params.angle_delta = 0;
  params.enable_edge_filter = true;
  av2enc::intra::IntraEdges edges_u;
  av2enc::intra::IntraEdges edges_v;
  if (prepare(chroma_u.data(), edges_u) != av2enc::intra::IntraStatus::kOk ||
      prepare(chroma_v.data(), edges_v) != av2enc::intra::IntraStatus::kOk) {
    std::fprintf(stderr, "cpu_block: prepare_edges failed\n");
    std::exit(1);
  }
  if (av2enc::intra::predict_intra(edges_u, params, out.pred_u.data(),
                                   static_cast<std::size_t>(w)) !=
          av2enc::intra::IntraStatus::kOk ||
      av2enc::intra::predict_intra(edges_v, params, out.pred_v.data(),
                                   static_cast<std::size_t>(w)) !=
          av2enc::intra::IntraStatus::kOk) {
    std::fprintf(stderr, "cpu_block: predict_intra failed\n");
    std::exit(1);
  }
  out.dc_u = out.pred_u[0];
  out.dc_v = out.pred_v[0];
  const std::vector<std::uint8_t> dc_pred_u = out.pred_u;
  const std::vector<std::uint8_t> dc_pred_v = out.pred_v;

  // DC candidate is scored on U only (choose_uv_mode).
  {
    std::vector<std::uint8_t> src_u_block(static_cast<std::size_t>(w) * h);
    std::vector<std::uint8_t> src_v_block(static_cast<std::size_t>(w) * h);
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        src_u_block[static_cast<std::size_t>(i) * w + j] = source_clamped(
            src_u, src_stride, src_max_x, src_max_y, x + j, y + i);
        src_v_block[static_cast<std::size_t>(i) * w + j] = source_clamped(
            src_v, src_stride, src_max_x, src_max_y, x + j, y + i);
      }
    }
    out.sse_dc_u = sse_plane(out.pred_u, src_u_block, w, h);
    out.sse_dc_v = sse_plane(out.pred_v, src_v_block, w, h);  // diagnostic only
  }

  if (!enabled_cfl) {
    out.sse_cfl_u = 0;
    out.sse_cfl_v = 0;
    return out;
  }

  // --- CfL through the CPU reference --------------------------------------
  std::vector<std::uint8_t> u_key = chroma_u;
  std::vector<std::uint8_t> v_key = chroma_v;
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      const std::size_t index = static_cast<std::size_t>(y + i) *
                                    static_cast<std::size_t>(chroma_stride) +
                                static_cast<std::size_t>(x + j);
      u_key[index] = static_cast<std::uint8_t>(out.dc_u);
      v_key[index] = static_cast<std::uint8_t>(out.dc_v);
    }
  }
  av2enc::cfl::Block ub;
  ub.luma = luma.data();
  ub.luma_stride = luma_stride;
  ub.luma_max_x = luma_max_x;
  ub.luma_max_y = luma_max_y;
  ub.sb_top_y = sb_top_y;
  ub.chroma = u_key.data();
  ub.chroma_stride = chroma_stride;
  ub.x = x;
  ub.y = y;
  ub.w = w;
  ub.h = h;
  ub.coded_width = coded_w;
  ub.coded_height = coded_h;
  ub.have_above = av2enc::cfl::cfl_avail_above(y, 1) ? 1 : 0;
  ub.have_left = av2enc::cfl::cfl_avail_left(x, 1) ? 1 : 0;
  ub.bit_depth = 8;
  ub.ds_filter = ds_filter;
  ub.sub_x = 1;
  ub.sub_y = 1;
  av2enc::cfl::Block vb = ub;
  vb.chroma = v_key.data();

  av2enc::cfl::Scratch scratch;
  av2enc::cfl::compute_luma(ub, scratch);
  out.alpha_u = av2enc::cfl::derive_implicit_alpha(ub, scratch);
  av2enc::cfl::compute_luma(vb, scratch);
  out.alpha_v = av2enc::cfl::derive_implicit_alpha(vb, scratch);
  av2enc::cfl::apply_prediction(ub, out.alpha_u, scratch);
  av2enc::cfl::apply_prediction(vb, out.alpha_v, scratch);

  std::vector<std::uint8_t> src_u_block(static_cast<std::size_t>(w) * h);
  std::vector<std::uint8_t> src_v_block(static_cast<std::size_t>(w) * h);
  out.pred_u.assign(static_cast<std::size_t>(w) * h, 0);
  out.pred_v.assign(static_cast<std::size_t>(w) * h, 0);
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      const std::size_t key = static_cast<std::size_t>(y + i) *
                                  static_cast<std::size_t>(chroma_stride) +
                              static_cast<std::size_t>(x + j);
      out.pred_u[static_cast<std::size_t>(i) * w + j] = u_key[key];
      out.pred_v[static_cast<std::size_t>(i) * w + j] = v_key[key];
      src_u_block[static_cast<std::size_t>(i) * w + j] = source_clamped(
          src_u, src_stride, src_max_x, src_max_y, x + j, y + i);
      src_v_block[static_cast<std::size_t>(i) * w + j] = source_clamped(
          src_v, src_stride, src_max_x, src_max_y, x + j, y + i);
    }
  }
  out.sse_cfl_u = sse_plane(out.pred_u, src_u_block, w, h);
  out.sse_cfl_v = sse_plane(out.pred_v, src_v_block, w, h);

  // Same tie-break as choose_uv_mode(): DC first, strict `<`. The CPU path
  // returns the DC prediction when DC wins, so restore it here.
  const long long cfl_total = out.sse_cfl_u + out.sse_cfl_v;
  out.is_cfl = cfl_total < out.sse_dc_u;
  if (!out.is_cfl) {
    out.pred_u = dc_pred_u;
    out.pred_v = dc_pred_v;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Synthetic parity cases
// ---------------------------------------------------------------------------
struct DeviceBuffers {
  std::uint8_t* luma = nullptr;
  std::uint8_t* chroma_u = nullptr;
  std::uint8_t* chroma_v = nullptr;
  std::uint8_t* src_u = nullptr;
  std::uint8_t* src_v = nullptr;
  std::uint8_t* pred_u = nullptr;
  std::uint8_t* pred_v = nullptr;
  M1ChromaResult* result = nullptr;
  M1ChromaJob* jobs = nullptr;
  M1ChromaResult* results = nullptr;
  int luma_cap = 0;
  int chroma_cap = 0;
  int src_cap = 0;
};

void synthetic_case(Rng& rng, int w, int h, std::vector<std::uint8_t>& luma,
                    int& luma_stride, int& luma_max_x, int& luma_max_y,
                    std::vector<std::uint8_t>& chroma_u,
                    std::vector<std::uint8_t>& chroma_v, int& chroma_stride,
                    std::vector<std::uint8_t>& src_u,
                    std::vector<std::uint8_t>& src_v, int& src_stride,
                    int& coded_w, int& coded_h, int& x, int& y,
                    int& src_max_x, int& src_max_y, int& sb_top_y,
                    int& ds_filter, int& num4_above_right,
                    int& num4_below_left) {
  static const int kOrigins[] = {0, 2, 4, 6, 8, 10, 12, 16, 20, 32};
  y = kOrigins[rng.range(10)];
  x = kOrigins[rng.range(10)];
  coded_w = x + w + rng.range(9);
  coded_h = y + h + rng.range(9);
  luma_stride = 2 * coded_w + rng.range(17);
  luma_max_x = 2 * coded_w - 1;
  luma_max_y = 2 * coded_h - 1;
  sb_top_y = ((y * 2) / 64) * 64;
  luma.assign(static_cast<std::size_t>(luma_stride) *
                  static_cast<std::size_t>(2 * coded_h + 16),
              0);
  for (auto& v : luma) v = static_cast<std::uint8_t>(rng.range(256));

  chroma_stride = coded_w + rng.range(17);
  const int chroma_h = coded_h + 8;
  chroma_u.assign(static_cast<std::size_t>(chroma_stride) * chroma_h, 0);
  chroma_v.assign(chroma_u.size(), 0);
  const bool correlated = rng.range(100) < 60;
  if (correlated) {
    // Chroma tracks the (filter-0 downsampled) luma with opposite slopes, so
    // the derived alpha is non-zero and CfL can win. That exercises the CfL
    // branch, alpha sign, and the U+V scoring path.
    const int luma_rows = 2 * coded_h + 16;
    for (int cy = 0; cy < chroma_h; ++cy) {
      for (int cx = 0; cx < chroma_stride; ++cx) {
        const int lx = std::min(cx * 2, luma_stride - 2);
        const int ly = std::min(cy * 2, luma_rows - 2);
        const int lum = (static_cast<int>(luma[static_cast<std::size_t>(ly) *
                                                 luma_stride + lx]) +
                         static_cast<int>(luma[static_cast<std::size_t>(ly) *
                                                  luma_stride + lx + 1]) +
                         static_cast<int>(
                             luma[static_cast<std::size_t>(ly + 1) *
                                      luma_stride + lx]) +
                         static_cast<int>(
                             luma[static_cast<std::size_t>(ly + 1) *
                                      luma_stride + lx + 1])) /
                        4;
        const int noise_u = rng.range(5) - 2;
        const int noise_v = rng.range(5) - 2;
        const std::size_t index = static_cast<std::size_t>(cy) * chroma_stride + cx;
        chroma_u[index] = static_cast<std::uint8_t>(
            clamp3(0, 255, 128 + (lum - 128) / 2 + noise_u));
        chroma_v[index] = static_cast<std::uint8_t>(
            clamp3(0, 255, 128 - (lum - 128) / 3 + noise_v));
      }
    }
  } else {
    for (auto& v : chroma_u) v = static_cast<std::uint8_t>(rng.range(256));
    for (auto& v : chroma_v) v = static_cast<std::uint8_t>(rng.range(256));
  }

  src_max_x = std::max(0, coded_w - rng.range(4));
  src_max_y = std::max(0, coded_h - rng.range(4));
  src_stride = src_max_x + 1;
  src_u.assign(static_cast<std::size_t>(src_stride) *
                   static_cast<std::size_t>(src_max_y + 1),
               0);
  src_v.assign(src_u.size(), 0);
  if (correlated) {
    // Source is the reconstruction target the chroma data was built from, so
    // the SSE decision is the realistic one (CfL often beats DC).
    for (int sy = 0; sy <= src_max_y; ++sy) {
      for (int sx = 0; sx <= src_max_x; ++sx) {
        const std::size_t src_index =
            static_cast<std::size_t>(sy) * src_stride + sx;
        const std::size_t key =
            static_cast<std::size_t>(sy) * chroma_stride + sx;
        src_u[src_index] = chroma_u[key];
        src_v[src_index] = chroma_v[key];
      }
    }
  } else {
    for (auto& v : src_u) v = static_cast<std::uint8_t>(rng.range(256));
    for (auto& v : src_v) v = static_cast<std::uint8_t>(rng.range(256));
  }

  ds_filter = rng.range(4);
  num4_above_right = rng.range(w / 4 + 1);
  num4_below_left = rng.range(h / 4 + 1);
}

void run_synthetic(DeviceBuffers& dev, Stats& stats) {
  static const int kSizes[] = {4, 8, 16, 32};
  Rng rng(0x9e3779b97f4a7c15ULL);
  const int cases = 600;
  M1ChromaJob last_job;
  M1ChromaOptions last_options;
  M1ChromaResult last_got;
  bool have_last = false;
  for (int c = 0; c < cases; ++c) {
    const int w = kSizes[rng.range(4)];
    const int h = kSizes[rng.range(4)];
    std::vector<std::uint8_t> luma, chroma_u, chroma_v, src_u, src_v;
    int luma_stride = 0, luma_max_x = 0, luma_max_y = 0;
    int chroma_stride = 0, src_stride = 0;
    int coded_w = 0, coded_h = 0, x = 0, y = 0;
    int src_max_x = 0, src_max_y = 0, sb_top_y = 0;
    int ds_filter = 0, num4_above_right = 0, num4_below_left = 0;
    synthetic_case(rng, w, h, luma, luma_stride, luma_max_x, luma_max_y,
                   chroma_u, chroma_v, chroma_stride, src_u, src_v, src_stride,
                   coded_w, coded_h, x, y, src_max_x, src_max_y, sb_top_y,
                   ds_filter, num4_above_right, num4_below_left);

    const bool enable_cfl = (c % 7) != 0;
    const CpuChroma cpu = cpu_block(
        w, h, x, y, coded_w, coded_h, sb_top_y, luma_max_x, luma_max_y, luma,
        luma_stride, chroma_u, chroma_v, chroma_stride, src_u, src_v,
        src_stride, src_max_x, src_max_y, ds_filter, num4_above_right,
        num4_below_left, enable_cfl);

    const std::size_t luma_bytes = luma.size();
    const std::size_t chroma_bytes = chroma_u.size();
    const std::size_t src_bytes = src_u.size();
    if (static_cast<int>(luma_bytes) > dev.luma_cap ||
        static_cast<int>(chroma_bytes) > dev.chroma_cap ||
        static_cast<int>(src_bytes) > dev.src_cap) {
      std::fprintf(stderr, "device buffer too small\n");
      std::exit(1);
    }
    CUDA_CHECK(cudaMemcpy(dev.luma, luma.data(), luma_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.chroma_u, chroma_u.data(), chroma_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.chroma_v, chroma_v.data(), chroma_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.src_u, src_u.data(), src_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.src_v, src_v.data(), src_bytes,
                          cudaMemcpyHostToDevice));

    M1ChromaJob job;
    job.luma = dev.luma;
    job.luma_stride = luma_stride;
    job.luma_max_x = luma_max_x;
    job.luma_max_y = luma_max_y;
    job.sb_top_y = sb_top_y;
    job.chroma_u = dev.chroma_u;
    job.chroma_v = dev.chroma_v;
    job.chroma_stride = chroma_stride;
    job.pred_u = dev.pred_u;
    job.pred_v = dev.pred_v;
    job.pred_stride = 0;
    job.src_u = dev.src_u;
    job.src_v = dev.src_v;
    job.src_stride_u = src_stride;
    job.src_stride_v = src_stride;
    job.src_max_x = src_max_x;
    job.src_max_y = src_max_y;
    job.x = x;
    job.y = y;
    job.w = w;
    job.h = h;
    job.coded_width = coded_w;
    job.coded_height = coded_h;
    job.num4_above_right = num4_above_right;
    job.num4_below_left = num4_below_left;

    M1ChromaOptions options;
    options.enable_cfl = enable_cfl ? 1 : 0;
    options.ds_filter = ds_filter;

    {
      const cudaError_t status =
          av2enc::cuda::m1::chroma_decide(job, options, dev.result);
      if (status != cudaSuccess) {
        std::fprintf(stderr,
                     "chroma_decide rejected case %d: %s x=%d y=%d %dx%d "
                     "coded=%dx%d luma_stride=%d chroma_stride=%d "
                     "luma_max=%d,%d src=%dx%d stride=%d cfl=%d\n",
                     c, cudaGetErrorString(status), x, y, w, h, coded_w,
                     coded_h, luma_stride, chroma_stride, luma_max_x,
                     luma_max_y, src_max_x + 1, src_max_y + 1, src_stride,
                     enable_cfl ? 1 : 0);
        std::exit(1);
      }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    M1ChromaResult got;
    CUDA_CHECK(cudaMemcpy(&got, dev.result, sizeof(got),
                          cudaMemcpyDeviceToHost));
    std::vector<std::uint8_t> pred_u(static_cast<std::size_t>(w) * h);
    std::vector<std::uint8_t> pred_v(pred_u.size());
    CUDA_CHECK(cudaMemcpy(pred_u.data(), dev.pred_u, pred_u.size(),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pred_v.data(), dev.pred_v, pred_v.size(),
                          cudaMemcpyDeviceToHost));

    ++stats.blocks;
    if (cpu.is_cfl) ++stats.cfl_blocks;
    check_int("is_cfl", got.is_cfl, cpu.is_cfl ? 1 : 0, stats, x, y, w, h);
    check_int("uv_mode", got.uv_mode, 0, stats, x, y, w, h);
    check_int("angle_delta", got.angle_delta, 0, stats, x, y, w, h);
    check_int("dc_u", got.dc_u, cpu.dc_u, stats, x, y, w, h);
    check_int("dc_v", got.dc_v, cpu.dc_v, stats, x, y, w, h);
    check_int("sse_dc_u", got.sse_dc_u, cpu.sse_dc_u, stats, x, y, w, h);
    check_int("sse_dc_v", got.sse_dc_v, cpu.sse_dc_v, stats, x, y, w, h);
    if (enable_cfl) {
      check_int("alpha_u", got.alpha_u, cpu.alpha_u, stats, x, y, w, h);
      check_int("alpha_v", got.alpha_v, cpu.alpha_v, stats, x, y, w, h);
      check_int("sse_cfl_u", got.sse_cfl_u, cpu.sse_cfl_u, stats, x, y, w, h);
      check_int("sse_cfl_v", got.sse_cfl_v, cpu.sse_cfl_v, stats, x, y, w, h);
      const long long win = cpu.is_cfl ? cpu.sse_cfl_u + cpu.sse_cfl_v
                                       : cpu.sse_dc_u;
      check_int("sse", got.sse, win, stats, x, y, w, h);
    } else {
      check_int("sse", got.sse, 0, stats, x, y, w, h);
    }
    const std::vector<std::uint8_t>& want_u = cpu.pred_u;
    const std::vector<std::uint8_t>& want_v = cpu.pred_v;
    if (!check_bytes("pred_u", pred_u.data(), want_u.data(), w * h, w, h, x, y,
                     stats)) {
      ++stats.pred_mismatch;
      if (stats.failures < 10) {
        std::fprintf(stderr,
                     "    ctx cfl=%d/%d alpha=%d/%d cpu=%d/%d dc=%d/%d "
                     "cpu_dc=%d/%d ds=%d\n",
                     got.is_cfl, cpu.is_cfl ? 1 : 0, got.alpha_u, got.alpha_v,
                     cpu.alpha_u, cpu.alpha_v, got.dc_u, got.dc_v, cpu.dc_u,
                     cpu.dc_v, ds_filter);
      }
    }
    if (!check_bytes("pred_v", pred_v.data(), want_v.data(), w * h, w, h, x, y,
                     stats)) {
      ++stats.pred_mismatch;
    }
    last_job = job;
    last_options = options;
    last_got = got;
    have_last = true;
  }

  // Batch path: the same job repeated must produce independent, identical
  // results (result indexing, no shared state) even with no pred outputs.
  if (have_last) {
    const int batch = 16;
    std::vector<M1ChromaJob> jobs(static_cast<std::size_t>(batch), last_job);
    for (M1ChromaJob& job : jobs) {
      job.pred_u = nullptr;
      job.pred_v = nullptr;
    }
    CUDA_CHECK(cudaMemcpy(dev.jobs, jobs.data(),
                          sizeof(M1ChromaJob) * static_cast<std::size_t>(batch),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(av2enc::cuda::m1::chroma_decide_batch(
        dev.jobs, batch, last_options, dev.results));
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<M1ChromaResult> results(static_cast<std::size_t>(batch));
    CUDA_CHECK(cudaMemcpy(results.data(), dev.results,
                          sizeof(M1ChromaResult) * static_cast<std::size_t>(batch),
                          cudaMemcpyDeviceToHost));
    bool batch_ok = true;
    for (const M1ChromaResult& r : results) {
      if (r.winner != last_got.winner || r.is_cfl != last_got.is_cfl ||
          r.alpha_u != last_got.alpha_u || r.alpha_v != last_got.alpha_v ||
          r.sse != last_got.sse || r.sse_dc_u != last_got.sse_dc_u ||
          r.sse_cfl_u != last_got.sse_cfl_u ||
          r.sse_cfl_v != last_got.sse_cfl_v) {
        batch_ok = false;
      }
    }
    if (!batch_ok) {
      std::fprintf(stderr, "  MISMATCH batch vs single-block result\n");
      stats.note_failure();
    }

    // Policy/validation behaviour.
    M1ChromaOptions directional = last_options;
    directional.enable_uv_directional = 1;
    if (av2enc::cuda::m1::chroma_decide(last_job, directional, dev.result) !=
        cudaErrorNotSupported) {
      std::fprintf(stderr, "  MISMATCH directional policy was not rejected\n");
      stats.note_failure();
    }
    M1ChromaOptions full_uv = last_options;
    full_uv.enable_uv_full = 1;
    if (av2enc::cuda::m1::chroma_decide(last_job, full_uv, dev.result) !=
        cudaErrorNotSupported) {
      std::fprintf(stderr, "  MISMATCH uv_full policy was not rejected\n");
      stats.note_failure();
    }
    M1ChromaJob bad = last_job;
    bad.w = 2;
    if (av2enc::cuda::m1::chroma_decide(bad, last_options, dev.result) !=
        cudaErrorInvalidValue) {
      std::fprintf(stderr, "  MISMATCH invalid block size was not rejected\n");
      stats.note_failure();
    }
    std::printf("batch/policy checks   %s\n", stats.failures == 0 ? "ok" : "FAIL");
  }
}

// ---------------------------------------------------------------------------
// Real-frame layer: replay the M1 encoder block loop with a GPU hook.
// ---------------------------------------------------------------------------
struct GpuPlanes {
  std::uint8_t* luma = nullptr;
  std::uint8_t* chroma_u = nullptr;
  std::uint8_t* chroma_v = nullptr;
  std::uint8_t* src_u = nullptr;
  std::uint8_t* src_v = nullptr;
  std::uint8_t* pred_u = nullptr;
  std::uint8_t* pred_v = nullptr;
  M1ChromaResult* result = nullptr;

  int luma_stride = 0;
  int luma_height = 0;
  int chroma_stride = 0;
  int chroma_height = 0;
  int chroma_visible_w = 0;
  int chroma_visible_h = 0;
};

void upload_rect(std::uint8_t* dst, int dst_stride, const std::uint8_t* src,
                 int src_stride, int x, int y, int w, int h, int plane_w,
                 int plane_h) {
  if (w <= 0 || h <= 0 || x >= plane_w || y >= plane_h) return;
  const int copy_w = std::min(w, plane_w - x);
  const int copy_h = std::min(h, plane_h - y);
  if (copy_w <= 0 || copy_h <= 0) return;
  CUDA_CHECK(cudaMemcpy2D(dst + static_cast<std::size_t>(y) * dst_stride + x,
                          static_cast<std::size_t>(dst_stride),
                          src + static_cast<std::size_t>(y) * src_stride + x,
                          static_cast<std::size_t>(src_stride),
                          static_cast<std::size_t>(copy_w),
                          static_cast<std::size_t>(copy_h),
                          cudaMemcpyHostToDevice));
}

M1ChromaJob make_job(const av2enc::m1::Encoder& enc, const GpuPlanes& gpu,
                     int cx0, int cy0, int cw, int ch) {
  M1ChromaJob job;
  job.luma = gpu.luma;
  job.luma_stride = gpu.luma_stride;
  job.luma_max_x = enc.mi_cols_ * 4 - 1;
  job.luma_max_y = enc.mi_rows_ * 4 - 1;
  job.sb_top_y = (cy0 * 2 / 64) * 64;
  job.chroma_u = gpu.chroma_u;
  job.chroma_v = gpu.chroma_v;
  job.chroma_stride = gpu.chroma_stride;
  job.pred_u = gpu.pred_u;
  job.pred_v = gpu.pred_v;
  job.pred_stride = 0;
  job.src_u = gpu.src_u;
  job.src_v = gpu.src_v;
  job.src_stride_u = gpu.chroma_visible_w;
  job.src_stride_v = gpu.chroma_visible_w;
  job.src_max_x = gpu.chroma_visible_w - 1;
  job.src_max_y = gpu.chroma_visible_h - 1;
  job.x = cx0;
  job.y = cy0;
  job.w = cw;
  job.h = ch;
  job.coded_width = enc.mi_cols_ * 2;
  job.coded_height = enc.mi_rows_ * 2;
  job.num4_above_right = enc.count_top_right(1, cx0 >> 2, cy0 >> 2, cw >> 2);
  job.num4_below_left = enc.count_bottom_left(1, cx0 >> 2, cy0 >> 2, ch >> 2);
  return job;
}

bool replay_node(av2enc::m1::Encoder& enc,
                 const av2enc::partition::PartitionNode& node,
                 const GpuPlanes& gpu, const M1ChromaOptions& options,
                 Stats& stats, std::string& error,
                 std::vector<M1ChromaJob>& timing_jobs);

bool replay_block(av2enc::m1::Encoder& enc,
                  const av2enc::partition::PartitionNode& node,
                  const GpuPlanes& gpu, const M1ChromaOptions& options,
                  Stats& stats, std::string& error,
                  std::vector<M1ChromaJob>& timing_jobs) {
  using av2enc::partition::BlockSize;
  const int r = node.mi_row;
  const int c = node.mi_col;
  const int w = av2enc::partition::block_width(node.size);
  const int h = av2enc::partition::block_height(node.size);
  const int w4 = w >> 2;
  const int h4 = h >> 2;
  const int x0 = c * 4;
  const int y0 = r * 4;

  av2enc::m1::ModeChoice y;
  if (!enc.choose_luma_mode(x0, y0, w, h, y)) {
    error = enc.error_;
    return false;
  }
  std::vector<std::uint8_t> luma_pred;
  if (!enc.predict_block(0, x0, y0, w, h, y.mode, y.angle_delta, luma_pred)) {
    error = enc.error_;
    return false;
  }
  av2enc::m1::TxPlan luma_plan;
  if (!enc.compute_tx(0, x0, y0, w, h, luma_pred, luma_plan)) {
    error = enc.error_;
    return false;
  }
  enc.write_recon(0, x0, y0, w, h, luma_plan);
  upload_rect(gpu.luma, gpu.luma_stride, enc.recon(0).samples.data(),
              enc.recon(0).width, x0, y0, w, h, enc.recon(0).width,
              enc.recon(0).height);

  const BlockSize chroma_size =
      av2enc::partition::chroma_plane_residual_size(enc.geometry_, node.size);
  if (!av2enc::partition::is_valid_block_size(chroma_size)) {
    error = "invalid chroma residual size";
    return false;
  }
  const int cw = av2enc::partition::block_width(chroma_size);
  const int ch = av2enc::partition::block_height(chroma_size);
  const int cx0 = x0 / 2;
  const int cy0 = y0 / 2;

  // CPU expected values captured from the pre-decision state.
  av2enc::cfl::Scratch scratch;
  av2enc::cfl::Block ub;
  av2enc::cfl::Block vb;
  enc.make_cfl_block(1, cx0, cy0, cw, ch, ub);
  enc.make_cfl_block(2, cx0, cy0, cw, ch, vb);
  av2enc::cfl::compute_luma(ub, scratch);
  const int cpu_alpha_u = av2enc::cfl::derive_implicit_alpha(ub, scratch);
  av2enc::cfl::compute_luma(vb, scratch);
  const int cpu_alpha_v = av2enc::cfl::derive_implicit_alpha(vb, scratch);

  const M1ChromaJob job = make_job(enc, gpu, cx0, cy0, cw, ch);
  {
    const cudaError_t status =
        av2enc::cuda::m1::chroma_decide(job, options, gpu.result);
    if (status != cudaSuccess) {
      std::fprintf(stderr,
                   "chroma_decide rejected frame block: %s x=%d y=%d %dx%d "
                   "coded=%dx%d chroma_stride=%d luma_max=%d,%d src=%dx%d "
                   "sb_top=%d num4=%d,%d\n",
                   cudaGetErrorString(status), cx0, cy0, cw, ch,
                   job.coded_width, job.coded_height, job.chroma_stride,
                   job.luma_max_x, job.luma_max_y, job.src_max_x + 1,
                   job.src_max_y + 1, job.sb_top_y, job.num4_above_right,
                   job.num4_below_left);
      std::exit(1);
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  M1ChromaResult got;
  CUDA_CHECK(cudaMemcpy(&got, gpu.result, sizeof(got), cudaMemcpyDeviceToHost));
  std::vector<std::uint8_t> got_u(static_cast<std::size_t>(cw) * ch);
  std::vector<std::uint8_t> got_v(got_u.size());
  CUDA_CHECK(cudaMemcpy(got_u.data(), gpu.pred_u, got_u.size(),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(got_v.data(), gpu.pred_v, got_v.size(),
                        cudaMemcpyDeviceToHost));

  av2enc::m1::UvChoice uv;
  if (!enc.choose_uv_mode(cx0, cy0, cw, ch, y.av2_mode, y.angle_delta, uv)) {
    error = enc.error_;
    return false;
  }

  // CPU predictions/SSE for comparison, using the same CPU routines.
  std::vector<std::uint8_t> u_dc;
  std::vector<std::uint8_t> v_dc;
  if (!enc.predict_block(1, cx0, cy0, cw, ch, av2enc::intra::Mode::kDc, 0,
                         u_dc) ||
      !enc.predict_block(2, cx0, cy0, cw, ch, av2enc::intra::Mode::kDc, 0,
                         v_dc)) {
    error = enc.error_;
    return false;
  }
  const long long cpu_sse_dc_u =
      enc.sse_against_source(1, cx0, cy0, cw, ch, u_dc);
  const long long cpu_sse_dc_v =
      enc.sse_against_source(2, cx0, cy0, cw, ch, v_dc);

  long long cpu_sse_cfl_u = 0;
  long long cpu_sse_cfl_v = 0;
  std::vector<std::uint8_t> cfl_pred_u(static_cast<std::size_t>(cw) * ch);
  std::vector<std::uint8_t> cfl_pred_v(cfl_pred_u.size());
  if (options.enable_cfl != 0) {
    for (int yy = 0; yy < ch; ++yy) {
      for (int xx = 0; xx < cw; ++xx) {
        cfl_pred_u[static_cast<std::size_t>(yy) * cw + xx] =
            enc.recon(1).at(cx0 + xx, cy0 + yy);
        cfl_pred_v[static_cast<std::size_t>(yy) * cw + xx] =
            enc.recon(2).at(cx0 + xx, cy0 + yy);
      }
    }
    cpu_sse_cfl_u = enc.sse_against_source(1, cx0, cy0, cw, ch, cfl_pred_u);
    cpu_sse_cfl_v = enc.sse_against_source(2, cx0, cy0, cw, ch, cfl_pred_v);
  }

  ++stats.blocks;
  if (uv.is_cfl) ++stats.cfl_blocks;
  if (cx0 + cw > job.coded_width || cy0 + ch > job.coded_height) {
    ++stats.partial_blocks;
  }
  if (cx0 % 32 == 0 || cy0 % 32 == 0) ++stats.sb_boundary_blocks;
  if (job.sb_top_y > 0) ++stats.non_top_sb_blocks;
  if (got.alpha_u < 0 || got.alpha_v < 0) ++stats.negative_alpha_blocks;
  if (got.is_cfl != (uv.is_cfl ? 1 : 0)) {
    std::fprintf(stderr,
                 "  DECISION x=%d y=%d %dx%d got is_cfl=%d want %d "
                 "(dc_u=%lld cfl=%lld+%lld)\n",
                 cx0, cy0, cw, ch, got.is_cfl, uv.is_cfl ? 1 : 0,
                 cpu_sse_dc_u, cpu_sse_cfl_u, cpu_sse_cfl_v);
    ++stats.decision_mismatch;
    stats.note_failure();
  }
  check_int("uv_mode", got.uv_mode, uv.av2_mode, stats, cx0, cy0, cw, ch);
  check_int("angle_delta", got.angle_delta, uv.angle_delta, stats, cx0, cy0, cw,
            ch);
  if (options.enable_cfl != 0) {
    if (got.alpha_u != cpu_alpha_u || got.alpha_v != cpu_alpha_v) {
      std::fprintf(stderr,
                   "  ALPHA x=%d y=%d %dx%d got %d/%d want %d/%d\n", cx0, cy0,
                   cw, ch, got.alpha_u, got.alpha_v, cpu_alpha_u, cpu_alpha_v);
      ++stats.alpha_mismatch;
      stats.note_failure();
    }
    if (got.sse_cfl_u != cpu_sse_cfl_u || got.sse_cfl_v != cpu_sse_cfl_v) {
      std::fprintf(stderr,
                   "  SSE-CFL x=%d y=%d %dx%d got %lld/%lld want %lld/%lld\n",
                   cx0, cy0, cw, ch, got.sse_cfl_u, got.sse_cfl_v,
                   cpu_sse_cfl_u, cpu_sse_cfl_v);
      ++stats.sse_mismatch;
      stats.note_failure();
    }
    const long long want_win =
        uv.is_cfl ? cpu_sse_cfl_u + cpu_sse_cfl_v : cpu_sse_dc_u;
    const long long delta = got.sse > want_win ? got.sse - want_win
                                               : want_win - got.sse;
    stats.max_sse_delta = std::max(stats.max_sse_delta, delta);
    if (delta != 0) {
      std::fprintf(stderr, "  SSE x=%d y=%d %dx%d got %lld want %lld\n", cx0,
                   cy0, cw, ch, got.sse, want_win);
      ++stats.sse_mismatch;
      stats.note_failure();
    }
  } else if (got.sse != 0) {
    std::fprintf(stderr, "  SSE x=%d y=%d %dx%d got %lld want 0\n", cx0, cy0,
                 cw, ch, got.sse);
    ++stats.sse_mismatch;
    stats.note_failure();
  }
  if (got.sse_dc_u != cpu_sse_dc_u || got.sse_dc_v != cpu_sse_dc_v) {
    std::fprintf(stderr,
                 "  SSE-DC x=%d y=%d %dx%d got %lld/%lld want %lld/%lld\n",
                 cx0, cy0, cw, ch, got.sse_dc_u, got.sse_dc_v, cpu_sse_dc_u,
                 cpu_sse_dc_v);
    ++stats.sse_mismatch;
    stats.note_failure();
  }

  const std::vector<std::uint8_t>& want_u = uv.is_cfl ? cfl_pred_u : u_dc;
  const std::vector<std::uint8_t>& want_v = uv.is_cfl ? cfl_pred_v : v_dc;
  if (!check_bytes("pred_u", got_u.data(), want_u.data(), cw * ch, cw, ch,
                   cx0, cy0, stats)) {
    ++stats.pred_mismatch;
  }
  if (!check_bytes("pred_v", got_v.data(), want_v.data(), cw * ch, cw, ch,
                   cx0, cy0, stats)) {
    ++stats.pred_mismatch;
  }

  timing_jobs.push_back(job);

  // --- Finish the block exactly like encode_block() -----------------------
  std::vector<std::uint8_t> u_pred;
  std::vector<std::uint8_t> v_pred;
  if (uv.is_cfl) {
    if (!enc.predict_block(1, cx0, cy0, cw, ch, av2enc::intra::Mode::kDc, 0,
                           u_pred) ||
        !enc.predict_block(2, cx0, cy0, cw, ch, av2enc::intra::Mode::kDc, 0,
                           v_pred)) {
      error = enc.error_;
      return false;
    }
    for (int yy = 0; yy < ch; ++yy) {
      for (int xx = 0; xx < cw; ++xx) {
        enc.recon_mut(1).at(cx0 + xx, cy0 + yy) =
            u_pred[static_cast<std::size_t>(yy) * cw + xx];
        enc.recon_mut(2).at(cx0 + xx, cy0 + yy) =
            v_pred[static_cast<std::size_t>(yy) * cw + xx];
      }
    }
    av2enc::cfl::Scratch scratch2;
    av2enc::cfl::Block ub2;
    av2enc::cfl::Block vb2;
    enc.make_cfl_block(1, cx0, cy0, cw, ch, ub2);
    enc.make_cfl_block(2, cx0, cy0, cw, ch, vb2);
    av2enc::cfl::compute_luma(ub2, scratch2);
    const int alpha_u = av2enc::cfl::derive_implicit_alpha(ub2, scratch2);
    av2enc::cfl::compute_luma(vb2, scratch2);
    const int alpha_v = av2enc::cfl::derive_implicit_alpha(vb2, scratch2);
    av2enc::cfl::apply_prediction(ub2, alpha_u, scratch2);
    av2enc::cfl::apply_prediction(vb2, alpha_v, scratch2);
    for (int yy = 0; yy < ch; ++yy) {
      for (int xx = 0; xx < cw; ++xx) {
        u_pred[static_cast<std::size_t>(yy) * cw + xx] =
            enc.recon(1).at(cx0 + xx, cy0 + yy);
        v_pred[static_cast<std::size_t>(yy) * cw + xx] =
            enc.recon(2).at(cx0 + xx, cy0 + yy);
      }
    }
  } else {
    if (!enc.predict_block(1, cx0, cy0, cw, ch, uv.mode, uv.angle_delta,
                           u_pred) ||
        !enc.predict_block(2, cx0, cy0, cw, ch, uv.mode, uv.angle_delta,
                           v_pred)) {
      error = enc.error_;
      return false;
    }
  }
  av2enc::m1::TxPlan u_plan;
  av2enc::m1::TxPlan v_plan;
  if (!enc.compute_tx(1, cx0, cy0, cw, ch, u_pred, u_plan) ||
      !enc.compute_tx(2, cx0, cy0, cw, ch, v_pred, v_plan)) {
    error = enc.error_;
    return false;
  }
  enc.write_recon(1, cx0, cy0, cw, ch, u_plan);
  enc.write_recon(2, cx0, cy0, cw, ch, v_plan);
  upload_rect(gpu.chroma_u, gpu.chroma_stride, enc.recon(1).samples.data(),
              enc.recon(1).width, cx0, cy0, cw, ch, enc.recon(1).width,
              enc.recon(1).height);
  upload_rect(gpu.chroma_v, gpu.chroma_stride, enc.recon(2).samples.data(),
              enc.recon(2).width, cx0, cy0, cw, ch, enc.recon(2).width,
              enc.recon(2).height);

  // Context update (same as encode_block; entropy coding is not replayed).
  const int joint = av2enc::m1::syntax::y_mode_joint(y.av2_mode, y.angle_delta);
  for (int yy = 0; yy < h4; ++yy) {
    const int rr = r + yy;
    if (rr >= enc.mi_rows_) break;
    for (int xx = 0; xx < w4; ++xx) {
      const int cc = c + xx;
      if (cc >= enc.mi_cols_) break;
      enc.intra_joint(rr, cc) = joint;
    }
  }
  const int cr = r / 2;
  const int cc0 = c / 2;
  const int cw4 = cw / 4;
  const int ch4 = ch / 4;
  const int chroma_mi_rows = enc.mi_rows_ / 2;
  const int chroma_mi_cols = enc.mi_cols_ / 2;
  for (int yy = 0; yy < ch4; ++yy) {
    const int rr = cr + yy;
    if (rr >= chroma_mi_rows) break;
    for (int xx = 0; xx < cw4; ++xx) {
      const int cc = cc0 + xx;
      if (cc >= chroma_mi_cols) break;
      enc.uv_smooth(rr, cc) =
          (!uv.is_cfl && av2enc::m1::y_mode_is_smooth(uv.av2_mode)) ? 1 : 0;
    }
  }
  for (int plane = 0; plane < 3; ++plane) {
    enc.decoded_[plane].fill(
        plane == 0 ? c : c / 2, plane == 0 ? r : r / 2,
        plane == 0 ? w4 : cw4, plane == 0 ? h4 : ch4);
  }
  return true;
}

bool replay_node(av2enc::m1::Encoder& enc,
                 const av2enc::partition::PartitionNode& node,
                 const GpuPlanes& gpu, const M1ChromaOptions& options,
                 Stats& stats, std::string& error,
                 std::vector<M1ChromaJob>& timing_jobs) {
  if (node.partition == av2enc::partition::PartitionType::kNone) {
    return replay_block(enc, node, gpu, options, stats, error, timing_jobs);
  }
  for (const av2enc::partition::PartitionNode& child : node.children) {
    if (!replay_node(enc, child, gpu, options, stats, error, timing_jobs)) {
      return false;
    }
  }
  return true;
}

bool replay_frame(av2enc::m1::Encoder& enc, const GpuPlanes& gpu,
                  const M1ChromaOptions& options, Stats& stats,
                  std::string& error, std::vector<M1ChromaJob>& timing_jobs) {
  using namespace av2enc::partition;
  FrameGeometry geometry = enc.geometry_;
  DecisionOptions poptions;
  poptions.min_block_size_px = enc.options_.min_block_size;
  poptions.max_depth = enc.options_.max_partition_depth;
  poptions.allow_rect = true;
  poptions.allow_split = false;
  if (enc.options_.force_single_block) {
    poptions.max_depth = 0;
    poptions.min_block_size_px = 64;
  }
  const double lambda =
      0.85 * std::pow(2.0, (enc.options_.base_q_idx - 12) / 6.0);
  const double signal_cost = lambda * 2.0;
  const auto cost_fn = [&](const PartitionNodeInfo& info) {
    NodeCosts costs;
    costs.leaf = static_cast<double>(
        enc.block_variance_cost_at(info.mi_row, info.mi_col, info.size));
    if (std::getenv("AV2ENC_M1_SQUARE_ONLY") != nullptr &&
        block_width(info.size) != block_height(info.size)) {
      costs.leaf += 1.0e18;
    }
    if (std::getenv("AV2ENC_M1_LEAF32") != nullptr &&
        (block_width(info.size) > 32 || block_height(info.size) > 32)) {
      costs.leaf += 1.0e18;
    }
    costs.none_signal = 0.0;
    costs.split_signal = signal_cost;
    costs.horz_signal = signal_cost;
    costs.vert_signal = signal_cost;
    return costs;
  };
  const FrameDecision frame = decide_frame(geometry, cost_fn, poptions);
  if (frame.status != DecisionStatus::kOk) {
    error = "partition decision failed: " + frame.error;
    return false;
  }
  if (!validate_frame(geometry, frame, error)) {
    error = "partition tree invalid: " + error;
    return false;
  }
  for (const PartitionNode& sb : frame.superblocks) {
    for (int plane = 0; plane < 3; ++plane) {
      enc.decoded_[plane].reset(
          plane == 0 ? enc.mi_cols_ : enc.mi_cols_ / 2,
          plane == 0 ? enc.mi_rows_ : enc.mi_rows_ / 2);
    }
    enc.sb_mi_row_ = sb.mi_row;
    enc.sb_mi_col_ = sb.mi_col;
    enc.sb_cols_plane_[0] = std::min(16, enc.mi_cols_ - sb.mi_col);
    enc.sb_rows_plane_[0] = std::min(16, enc.mi_rows_ - sb.mi_row);
    enc.sb_cols_plane_[1] = std::min(8, enc.mi_cols_ / 2 - sb.mi_col / 2);
    enc.sb_rows_plane_[1] = std::min(8, enc.mi_rows_ / 2 - sb.mi_row / 2);
    enc.sb_cols_plane_[2] = enc.sb_cols_plane_[1];
    enc.sb_rows_plane_[2] = enc.sb_rows_plane_[1];
    if (!replay_node(enc, sb, gpu, options, stats, error, timing_jobs)) {
      return false;
    }
  }
  return true;
}

struct FrameCase {
  av2enc::m1::M1Options cpu_options;
  M1ChromaOptions gpu_options;
};

void run_frame_case(const av2enc::m1::StillPicture& picture, const FrameCase& c,
                    const std::string& label, Stats& stats,
                    std::vector<M1ChromaJob>& timing_jobs) {
  std::string error;
  av2enc::m1::Encoder reference(picture, c.cpu_options);
  av2enc::m1::SyntaxStream ref_stream;
  if (!reference.encode(&ref_stream, error)) {
    std::fprintf(stderr, "%s: reference encode failed: %s\n", label.c_str(),
                 error.c_str());
    std::exit(1);
  }

  av2enc::m1::Encoder enc(picture, c.cpu_options);
  GpuPlanes gpu;
  gpu.luma_stride = enc.recon(0).width;
  gpu.luma_height = enc.recon(0).height;
  gpu.chroma_stride = enc.recon(1).width;
  gpu.chroma_height = enc.recon(1).height;
  gpu.chroma_visible_w = static_cast<int>((picture.width + 1U) / 2U);
  gpu.chroma_visible_h = static_cast<int>((picture.height + 1U) / 2U);
  const std::size_t luma_bytes = static_cast<std::size_t>(gpu.luma_stride) *
                                 static_cast<std::size_t>(gpu.luma_height);
  const std::size_t chroma_bytes = static_cast<std::size_t>(gpu.chroma_stride) *
                                   static_cast<std::size_t>(gpu.chroma_height);
  const std::size_t src_bytes =
      static_cast<std::size_t>(gpu.chroma_visible_w) *
      static_cast<std::size_t>(gpu.chroma_visible_h);
  CUDA_CHECK(cudaMalloc(&gpu.luma, luma_bytes));
  CUDA_CHECK(cudaMalloc(&gpu.chroma_u, chroma_bytes));
  CUDA_CHECK(cudaMalloc(&gpu.chroma_v, chroma_bytes));
  CUDA_CHECK(cudaMalloc(&gpu.src_u, src_bytes));
  CUDA_CHECK(cudaMalloc(&gpu.src_v, src_bytes));
  CUDA_CHECK(cudaMalloc(&gpu.pred_u, 64U * 64U));
  CUDA_CHECK(cudaMalloc(&gpu.pred_v, 64U * 64U));
  CUDA_CHECK(cudaMalloc(&gpu.result, sizeof(M1ChromaResult)));
  CUDA_CHECK(cudaMemset(gpu.luma, 0, luma_bytes));
  CUDA_CHECK(cudaMemset(gpu.chroma_u, 0, chroma_bytes));
  CUDA_CHECK(cudaMemset(gpu.chroma_v, 0, chroma_bytes));
  CUDA_CHECK(cudaMemcpy(gpu.src_u, picture.u.data(), src_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(gpu.src_v, picture.v.data(), src_bytes,
                        cudaMemcpyHostToDevice));

  if (!replay_frame(enc, gpu, c.gpu_options, stats, error, timing_jobs)) {
    std::fprintf(stderr, "%s: replay failed: %s\n", label.c_str(),
                 error.c_str());
    std::exit(1);
  }

  // Harness self-check: the replayed encoder must equal a plain encode().
  bool recon_ok = true;
  for (int plane = 0; plane < 3; ++plane) {
    const auto& a = enc.recon(plane);
    const auto& b = reference.recon(plane);
    if (a.width != b.width || a.height != b.height ||
        a.samples != b.samples) {
      recon_ok = false;
      std::fprintf(stderr, "%s: replay recon plane %d differs from encode()\n",
                   label.c_str(), plane);
    }
  }
  if (!recon_ok) {
    stats.note_failure();
  }

  std::printf("%-30s blocks=%lld cfl=%lld partial=%lld sb-edge=%lld "
              "sb-top>0=%lld neg-alpha=%lld recon-harness=%s\n",
              label.c_str(), stats.blocks, stats.cfl_blocks,
              stats.partial_blocks, stats.sb_boundary_blocks,
              stats.non_top_sb_blocks, stats.negative_alpha_blocks,
              recon_ok ? "exact" : "DIFF");

  CUDA_CHECK(cudaFree(gpu.luma));
  CUDA_CHECK(cudaFree(gpu.chroma_u));
  CUDA_CHECK(cudaFree(gpu.chroma_v));
  CUDA_CHECK(cudaFree(gpu.src_u));
  CUDA_CHECK(cudaFree(gpu.src_v));
  CUDA_CHECK(cudaFree(gpu.pred_u));
  CUDA_CHECK(cudaFree(gpu.pred_v));
  CUDA_CHECK(cudaFree(gpu.result));
}

void time_batch(const std::vector<M1ChromaJob>& host_jobs,
                const M1ChromaOptions& options, const char* label) {
  if (host_jobs.empty()) return;
  const int count = static_cast<int>(host_jobs.size());
  M1ChromaJob* d_jobs = nullptr;
  M1ChromaResult* d_results = nullptr;
  std::uint8_t* pool_u = nullptr;
  std::uint8_t* pool_v = nullptr;
  CUDA_CHECK(cudaMalloc(&d_jobs, sizeof(M1ChromaJob) * count));
  CUDA_CHECK(cudaMalloc(&d_results, sizeof(M1ChromaResult) * count));
  CUDA_CHECK(cudaMalloc(&pool_u, static_cast<std::size_t>(count) * 64U * 64U));
  CUDA_CHECK(cudaMalloc(&pool_v, static_cast<std::size_t>(count) * 64U * 64U));
  std::vector<M1ChromaJob> jobs = host_jobs;
  for (int i = 0; i < count; ++i) {
    jobs[static_cast<std::size_t>(i)].pred_u =
        pool_u + static_cast<std::size_t>(i) * 64U * 64U;
    jobs[static_cast<std::size_t>(i)].pred_v =
        pool_v + static_cast<std::size_t>(i) * 64U * 64U;
    jobs[static_cast<std::size_t>(i)].pred_stride = 64;
  }
  CUDA_CHECK(cudaMemcpy(d_jobs, jobs.data(), sizeof(M1ChromaJob) * count,
                        cudaMemcpyHostToDevice));

  const int iters = 100;
  for (int i = 0; i < 10; ++i) {
    CUDA_CHECK(av2enc::cuda::m1::chroma_decide_batch(d_jobs, count, options,
                                                     d_results));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(av2enc::cuda::m1::chroma_decide_batch(d_jobs, count, options,
                                                     d_results));
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  const double per_frame = static_cast<double>(ms) / iters;
  const double per_block_us = per_frame * 1000.0 / static_cast<double>(count);
  std::printf("timing %-22s %8d blocks  %8.3f ms/frame  %7.3f us/block\n",
              label, count, per_frame, per_block_us);

  // Single-block entry point: launch/validation latency (one job per call).
  M1ChromaOptions single_options = options;
  M1ChromaResult* d_single = nullptr;
  CUDA_CHECK(cudaMalloc(&d_single, sizeof(M1ChromaResult)));
  for (int i = 0; i < 10; ++i) {
    CUDA_CHECK(av2enc::cuda::m1::chroma_decide(jobs[0], single_options,
                                               d_single));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(av2enc::cuda::m1::chroma_decide(jobs[0], single_options,
                                               d_single));
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  std::printf("timing %-22s %8d calls   %8.3f us/call (launch latency)\n",
              "single-block", iters, static_cast<double>(ms) * 1000.0 / iters);
  CUDA_CHECK(cudaFree(d_single));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_jobs));
  CUDA_CHECK(cudaFree(d_results));
  CUDA_CHECK(cudaFree(pool_u));
  CUDA_CHECK(cudaFree(pool_v));
}

// ---------------------------------------------------------------------------
// Input frames
// ---------------------------------------------------------------------------
struct FrameStorage {
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
  std::uint32_t width = 0;
  std::uint32_t height = 0;

  av2enc::m1::StillPicture picture() const {
    av2enc::m1::StillPicture p;
    p.width = width;
    p.height = height;
    p.y = y;
    p.u = u;
    p.v = v;
    return p;
  }
};

FrameStorage synthetic_frame(std::uint32_t width, std::uint32_t height) {
  FrameStorage f;
  f.width = width;
  f.height = height;
  const std::uint32_t cw = width / 2;
  const std::uint32_t ch = height / 2;
  f.y.resize(static_cast<std::size_t>(width) * height);
  f.u.resize(static_cast<std::size_t>(cw) * ch);
  f.v.resize(f.u.size());
  const auto hash = [](std::uint32_t x, std::uint32_t y) {
    std::uint32_t h = x * 374761393U + y * 668265263U;
    h = (h ^ (h >> 13)) * 1274126177U;
    return h ^ (h >> 16);
  };
  for (std::uint32_t y = 0; y < height; ++y) {
    for (std::uint32_t x = 0; x < width; ++x) {
      const std::uint32_t base = (x * 255U) / (width - 1U);
      const std::uint32_t wave =
          static_cast<std::uint32_t>(32.0 * std::sin(x * 0.11) *
                                     std::cos(y * 0.07));
      const std::uint32_t noise = hash(x, y) % 40U;
      f.y[static_cast<std::size_t>(y) * width + x] =
          static_cast<std::uint8_t>((base + wave + noise) & 0xFFU);
    }
  }
  for (std::uint32_t y = 0; y < ch; ++y) {
    for (std::uint32_t x = 0; x < cw; ++x) {
      const std::uint32_t base = (x * 255U) / (cw > 1 ? cw - 1U : 1U);
      const std::uint32_t noise = hash(x + 7U, y + 13U) % 90U;
      f.u[static_cast<std::size_t>(y) * cw + x] =
          static_cast<std::uint8_t>((base + noise) & 0xFFU);
      f.v[static_cast<std::size_t>(y) * cw + x] =
          static_cast<std::uint8_t>((255U - base + noise / 2U) & 0xFFU);
    }
  }
  return f;
}

std::optional<FrameStorage> load_y4m(const std::string& path) {
  av2enc::Y4mReader reader(path);
  std::string error;
  if (!reader.open(error)) {
    std::fprintf(stderr, "y4m open failed: %s\n", error.c_str());
    return std::nullopt;
  }
  av2enc::Y4mFrame frame;
  if (reader.read_frame(frame, error) != av2enc::Y4mReadResult::kFrame) {
    std::fprintf(stderr, "y4m read failed: %s\n", error.c_str());
    return std::nullopt;
  }
  FrameStorage f;
  f.width = frame.width();
  f.height = frame.height();
  f.y.assign(frame.y().begin(), frame.y().end());
  f.u.assign(frame.u().begin(), frame.u().end());
  f.v.assign(frame.v().begin(), frame.v().end());
  return f;
}

void clear_encoder_env() {
  static const char* kNames[] = {
      "AV2ENC_M1_FORCE_Y",   "AV2ENC_M1_FORCE_Y2", "AV2ENC_M1_BASE_ONLY",
      "AV2ENC_M1_NO_DIR",    "AV2ENC_M1_NO_EDGE",  "AV2ENC_M1_UV_DC",
      "AV2ENC_M1_LUMA_DC",   "AV2ENC_M1_TRACE",    "AV2ENC_M1_SQUARE_ONLY",
      "AV2ENC_M1_LEAF32",    "AV2ENC_M1_ZERO_DC",  "AV2ENC_M1_YMODE_CTX0",
      "AV2ENC_M1_YMODE_CTX2", "AV2ENC_M1_DCSIGN_CTX", "AV2ENC_M1_NO_TXTYPE",
  };
  for (const char* name : kNames) {
    unsetenv(name);
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> y4m_paths;
  bool run_full_intra_64 = true;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--y4m" && i + 1 < argc) {
      y4m_paths.emplace_back(argv[++i]);
    } else if (arg == "--no-full-intra") {
      run_full_intra_64 = false;
    } else if (arg == "-h" || arg == "--help") {
      std::printf("usage: %s [--y4m <file>]... [--no-full-intra]\n", argv[0]);
      return 0;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
      return 2;
    }
  }
  clear_encoder_env();

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  std::printf("m1_chroma_test: %s cc %d.%d, CUDA %d\n", prop.name, prop.major,
              prop.minor, CUDART_VERSION);

  // --- synthetic parity ----------------------------------------------------
  DeviceBuffers dev;
  dev.luma_cap = 512 * 512;
  dev.chroma_cap = 256 * 256;
  dev.src_cap = 128 * 128;
  CUDA_CHECK(cudaMalloc(&dev.luma, static_cast<std::size_t>(dev.luma_cap)));
  CUDA_CHECK(cudaMalloc(&dev.chroma_u,
                        static_cast<std::size_t>(dev.chroma_cap)));
  CUDA_CHECK(cudaMalloc(&dev.chroma_v,
                        static_cast<std::size_t>(dev.chroma_cap)));
  CUDA_CHECK(cudaMalloc(&dev.src_u, static_cast<std::size_t>(dev.src_cap)));
  CUDA_CHECK(cudaMalloc(&dev.src_v, static_cast<std::size_t>(dev.src_cap)));
  CUDA_CHECK(cudaMalloc(&dev.pred_u, 64U * 64U));
  CUDA_CHECK(cudaMalloc(&dev.pred_v, 64U * 64U));
  CUDA_CHECK(cudaMalloc(&dev.result, sizeof(M1ChromaResult)));
  CUDA_CHECK(cudaMalloc(&dev.jobs, sizeof(M1ChromaJob) * 16));
  CUDA_CHECK(cudaMalloc(&dev.results, sizeof(M1ChromaResult) * 16));

  Stats stats;
  run_synthetic(dev, stats);
  std::printf("synthetic            blocks=%lld cfl=%lld failures=%lld\n",
              stats.blocks, stats.cfl_blocks, stats.failures);

  CUDA_CHECK(cudaFree(dev.luma));
  CUDA_CHECK(cudaFree(dev.chroma_u));
  CUDA_CHECK(cudaFree(dev.chroma_v));
  CUDA_CHECK(cudaFree(dev.src_u));
  CUDA_CHECK(cudaFree(dev.src_v));
  CUDA_CHECK(cudaFree(dev.pred_u));
  CUDA_CHECK(cudaFree(dev.pred_v));
  CUDA_CHECK(cudaFree(dev.result));
  CUDA_CHECK(cudaFree(dev.jobs));
  CUDA_CHECK(cudaFree(dev.results));

  // --- real frames ---------------------------------------------------------
  struct NamedFrame {
    std::string label;
    FrameStorage storage;
  };
  std::vector<NamedFrame> frames;
  for (const std::string& path : y4m_paths) {
    auto loaded = load_y4m(path);
    if (!loaded.has_value()) return 1;
    const std::string label = std::to_string(loaded->width) + "x" +
                              std::to_string(loaded->height) + " " +
                              std::filesystem::path(path).filename().string();
    frames.push_back(NamedFrame{label, std::move(*loaded)});
  }
  if (frames.empty()) {
    frames.push_back(NamedFrame{"64x64 synthetic", synthetic_frame(64, 64)});
    frames.push_back(NamedFrame{"640x360 synthetic", synthetic_frame(640, 360)});
  }
  // Note: dimensions that are not a multiple of 8 make MiCols*4 > width, but
  // the CPU partition planner rejects those trees ("spec-forced partition
  // conflicts with DecisionOptions"), so no such frame is used here. Partial
  // blocks are covered by the 640x360 bottom edge (a 32x32 chroma block at
  // y=160 extends past the 180-row coded chroma plane).

  for (NamedFrame& frame : frames) {
    // Default policy: DC/CfL, luma DC (fast) then full intra on small frames.
    FrameCase base;
    base.cpu_options = av2enc::m1::M1Options{};
    base.gpu_options = M1ChromaOptions{};
    Stats frame_stats;
    std::vector<M1ChromaJob> timing_jobs;
    run_frame_case(frame.storage.picture(), base, frame.label + " dc/cfl",
                   frame_stats, timing_jobs);
    if (frame.storage.width >= 320) {
      time_batch(timing_jobs, base.gpu_options,
                 (frame.label + " batch").c_str());
    }

    if (run_full_intra_64 && frame.storage.width <= 64 &&
        frame.storage.height <= 64) {
      FrameCase full = base;
      full.cpu_options.luma_dc_only = false;
      Stats full_stats;
      std::vector<M1ChromaJob> full_jobs;
      run_frame_case(frame.storage.picture(), full,
                     frame.label + " full-intra", full_stats, full_jobs);
      frame_stats.blocks += full_stats.blocks;
      frame_stats.cfl_blocks += full_stats.cfl_blocks;
      frame_stats.failures += full_stats.failures;
      frame_stats.decision_mismatch += full_stats.decision_mismatch;
      frame_stats.alpha_mismatch += full_stats.alpha_mismatch;
      frame_stats.sse_mismatch += full_stats.sse_mismatch;
      frame_stats.pred_mismatch += full_stats.pred_mismatch;
    }

    if (frame.storage.width >= 320) {
      FrameCase dc_only = base;
      dc_only.cpu_options.enable_cfl = false;
      dc_only.cpu_options.force_dc = true;
      dc_only.gpu_options.enable_cfl = 0;
      Stats dc_stats;
      std::vector<M1ChromaJob> dc_jobs;
      run_frame_case(frame.storage.picture(), dc_only,
                     frame.label + " dc-only", dc_stats, dc_jobs);
      frame_stats.blocks += dc_stats.blocks;
      frame_stats.cfl_blocks += dc_stats.cfl_blocks;
      frame_stats.failures += dc_stats.failures;
      frame_stats.decision_mismatch += dc_stats.decision_mismatch;
      frame_stats.alpha_mismatch += dc_stats.alpha_mismatch;
      frame_stats.sse_mismatch += dc_stats.sse_mismatch;
      frame_stats.pred_mismatch += dc_stats.pred_mismatch;
    }

    stats.blocks += frame_stats.blocks;
    stats.cfl_blocks += frame_stats.cfl_blocks;
    stats.failures += frame_stats.failures;
    stats.decision_mismatch += frame_stats.decision_mismatch;
    stats.alpha_mismatch += frame_stats.alpha_mismatch;
    stats.sse_mismatch += frame_stats.sse_mismatch;
    stats.pred_mismatch += frame_stats.pred_mismatch;
  }

  std::printf("total blocks=%lld cfl=%lld failures=%lld "
              "(decision=%lld alpha=%lld sse=%lld pred=%lld)\n",
              stats.blocks, stats.cfl_blocks, stats.failures,
              stats.decision_mismatch, stats.alpha_mismatch,
              stats.sse_mismatch, stats.pred_mismatch);
  std::printf("RESULT: %s\n", stats.failures == 0 ? "PASS" : "FAIL");
  return stats.failures == 0 ? 0 : 1;
}

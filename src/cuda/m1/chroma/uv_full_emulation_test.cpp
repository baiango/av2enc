// Host emulation parity test for the M1 GPU full-UV chroma search.
//
// Compiles m1_chroma_device.cuh with CUDA keywords stubbed and one emulated
// thread, then compares the decision, all SSE diagnostics, and every
// prediction byte against a CPU reference built from the real CPU intra and
// CfL components (the same routines m1_still.cpp::choose_uv_mode uses).
//
// This is not a substitute for the on-device test (m1_chroma_test.cu), but it
// executes the exact device search code on a machine without nvcc.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define __device__
#define __host__
#define __forceinline__ inline
#define __shared__ static

struct EmuDim {
  unsigned x = 0;
  unsigned y = 0;
  unsigned z = 0;
};
static EmuDim threadIdx;
static EmuDim blockIdx;
static EmuDim blockDim;
static inline void __syncthreads() {}
static inline long long __shfl_down_sync(unsigned, long long value, int) {
  return value;
}

#include "m1_chroma_device.cuh"

#include "encoder/cfl/cfl.h"
#include "encoder/intra_modes/intra_modes.h"

namespace {

using av2enc::cuda::m1::ChromaUvShared;
using av2enc::cuda::m1::M1ChromaJob;
using av2enc::cuda::m1::M1ChromaOptions;
using av2enc::cuda::m1::M1ChromaResult;
using av2enc::cuda::m1::kM1ChromaCfl;
using av2enc::cuda::m1::kM1ChromaDc;
using av2enc::cuda::m1::kM1ChromaUvDirectional;
using av2enc::intra::IntraStatus;
using av2enc::intra::Mode;

int g_checks = 0;
int g_failures = 0;

std::uint32_t g_rng = 0x243f6a88u;
std::uint32_t next_random() {
  g_rng ^= g_rng << 13;
  g_rng ^= g_rng >> 17;
  g_rng ^= g_rng << 5;
  return g_rng;
}
int random_range(int low, int high) {
  return low + static_cast<int>(next_random() %
                                static_cast<std::uint32_t>(high - low + 1));
}

int clamp3(int lo, int hi, int v) {
  return v < lo ? lo : (v > hi ? hi : v);
}

std::uint8_t source_at(const std::uint8_t* plane, int stride, int max_x,
                       int max_y, int x, int y) {
  return plane[static_cast<std::size_t>(clamp3(0, max_y, y)) * stride +
               static_cast<std::size_t>(clamp3(0, max_x, x))];
}

long long sse_block(const std::uint8_t* src, int stride, int sx, int sy,
                    int max_x, int max_y, const std::vector<std::uint8_t>& pred,
                    int w, int h) {
  long long sse = 0;
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      const int s = source_at(src, stride, max_x, max_y, sx + j, sy + i);
      const int p = pred[static_cast<std::size_t>(i) * w + j];
      const int d = s - p;
      sse += static_cast<long long>(d) * d;
    }
  }
  return sse;
}

bool cpu_predict(const M1ChromaJob& job, int plane, int mode, int delta,
                 std::vector<std::uint8_t>& out) {
  av2enc::intra::NeighborRequest request;
  request.recon = av2enc::intra::PlaneView{
      plane == 1 ? job.chroma_u : job.chroma_v, job.chroma_stride};
  request.x = job.x;
  request.y = job.y;
  request.w = job.w;
  request.h = job.h;
  request.max_x = job.coded_width - 1;
  request.max_y = job.coded_height - 1;
  request.have_left = job.x > 0;
  request.have_above = job.y > 0;
  request.num4_above_right = job.num4_above_right;
  request.num4_below_left = job.num4_below_left;
  request.color_plane = 1;
  av2enc::intra::IntraEdges edges;
  if (av2enc::intra::prepare_edges(request, 8, edges) != IntraStatus::kOk) {
    return false;
  }
  av2enc::intra::PredictParams params;
  params.mode = static_cast<Mode>(mode);
  params.angle_delta = delta;
  params.enable_edge_filter = job.enable_edge_filter != 0;
  params.above_neighbor_is_smooth = job.filter_type_above != 0;
  params.left_neighbor_is_smooth = job.filter_type_left != 0;
  out.resize(static_cast<std::size_t>(job.w) * job.h);
  return av2enc::intra::predict_intra(edges, params, out.data(), job.w) ==
         IntraStatus::kOk;
}

bool cfl_reference(const M1ChromaJob& job, int ds_filter,
                   std::vector<std::uint8_t>& pred_u,
                   std::vector<std::uint8_t>& pred_v, int& alpha_u,
                   int& alpha_v) {
  pred_u.assign(static_cast<std::size_t>(job.w) * job.h, 0);
  pred_v.assign(pred_u.size(), 0);
  std::vector<std::uint8_t> dc_u;
  std::vector<std::uint8_t> dc_v;
  if (!cpu_predict(job, 1, 0, 0, dc_u) || !cpu_predict(job, 2, 0, 0, dc_v)) {
    return false;
  }
  // Copy the reconstructed chroma planes so apply_prediction cannot mutate the
  // job state (the CPU path temporarily writes DC into its recon planes).
  const std::size_t chroma_size =
      static_cast<std::size_t>(job.chroma_stride) *
      static_cast<std::size_t>(job.coded_height + 96);
  std::vector<std::uint8_t> cu(chroma_size);
  std::vector<std::uint8_t> cv(chroma_size);
  std::memcpy(cu.data(), job.chroma_u, chroma_size);
  std::memcpy(cv.data(), job.chroma_v, chroma_size);
  for (int i = 0; i < job.h; ++i) {
    for (int j = 0; j < job.w; ++j) {
      const int index = (job.y + i) * job.chroma_stride + job.x + j;
      cu[static_cast<std::size_t>(index)] =
          dc_u[static_cast<std::size_t>(i) * job.w + j];
      cv[static_cast<std::size_t>(index)] =
          dc_v[static_cast<std::size_t>(i) * job.w + j];
    }
  }

  av2enc::cfl::Block ub;
  ub.luma = job.luma;
  ub.luma_stride = job.luma_stride;
  ub.luma_max_x = job.luma_max_x;
  ub.luma_max_y = job.luma_max_y;
  ub.sb_top_y = job.sb_top_y;
  ub.chroma = cu.data();
  ub.chroma_stride = job.chroma_stride;
  ub.x = job.x;
  ub.y = job.y;
  ub.w = job.w;
  ub.h = job.h;
  ub.coded_width = job.coded_width;
  ub.coded_height = job.coded_height;
  ub.have_above = av2enc::cfl::cfl_avail_above(job.y, 1) ? 1 : 0;
  ub.have_left = av2enc::cfl::cfl_avail_left(job.x, 1) ? 1 : 0;
  ub.bit_depth = 8;
  ub.ds_filter = ds_filter;
  ub.sub_x = 1;
  ub.sub_y = 1;
  av2enc::cfl::Block vb = ub;
  vb.chroma = cv.data();

  av2enc::cfl::Scratch scratch;
  av2enc::cfl::compute_luma(ub, scratch);
  alpha_u = av2enc::cfl::derive_implicit_alpha(ub, scratch);
  alpha_v = av2enc::cfl::derive_implicit_alpha(vb, scratch);
  av2enc::cfl::apply_prediction(ub, alpha_u, scratch);
  av2enc::cfl::apply_prediction(vb, alpha_v, scratch);
  for (int i = 0; i < job.h; ++i) {
    for (int j = 0; j < job.w; ++j) {
      const int index = (job.y + i) * job.chroma_stride + job.x + j;
      pred_u[static_cast<std::size_t>(i) * job.w + j] =
          cu[static_cast<std::size_t>(index)];
      pred_v[static_cast<std::size_t>(i) * job.w + j] =
          cv[static_cast<std::size_t>(index)];
    }
  }
  return true;
}

struct CpuExpectation {
  M1ChromaResult result;
  std::vector<std::uint8_t> pred_u;
  std::vector<std::uint8_t> pred_v;
};

bool cpu_full_uv(const M1ChromaJob& job, const M1ChromaOptions& opt,
                 CpuExpectation& expect) {
  const bool y_directional = job.y_mode >= 1 && job.y_mode <= 8;
  long long best = 0x7fffffffffffffffLL;
  int best_mode = 0;
  int best_delta = 0;
  bool best_cfl = false;
  long long sse_dc_u = 0;
  long long sse_dc_v = 0;
  long long sse_cfl_u = 0;
  long long sse_cfl_v = 0;
  std::vector<std::uint8_t> pred;

  for (int mode = 0; mode < 13; ++mode) {
    const int delta = (y_directional && mode == job.y_mode) ? job.y_angle_delta : 0;
    std::vector<std::uint8_t> u;
    std::vector<std::uint8_t> v;
    if (!cpu_predict(job, 1, mode, delta, u) ||
        !cpu_predict(job, 2, mode, delta, v)) {
      return false;
    }
    const long long su = sse_block(job.src_u, job.src_stride_u, job.x, job.y,
                                   job.src_max_x, job.src_max_y, u, job.w,
                                   job.h);
    const long long sv = sse_block(job.src_v, job.src_stride_v, job.x, job.y,
                                   job.src_max_x, job.src_max_y, v, job.w,
                                   job.h);
    if (mode == 0) {
      sse_dc_u = su;
      sse_dc_v = sv;
    }
    if (su + sv < best) {
      best = su + sv;
      best_mode = mode;
      best_delta = delta;
      best_cfl = false;
      expect.pred_u = std::move(u);
      expect.pred_v = std::move(v);
    }
  }

  int alpha_u = 0;
  int alpha_v = 0;
  std::vector<std::uint8_t> cfl_u;
  std::vector<std::uint8_t> cfl_v;
  if (!cfl_reference(job, opt.ds_filter, cfl_u, cfl_v, alpha_u, alpha_v)) {
    return false;
  }
  if (opt.enable_cfl != 0) {
    sse_cfl_u = sse_block(job.src_u, job.src_stride_u, job.x, job.y,
                          job.src_max_x, job.src_max_y, cfl_u, job.w, job.h);
    sse_cfl_v = sse_block(job.src_v, job.src_stride_v, job.x, job.y,
                          job.src_max_x, job.src_max_y, cfl_v, job.w, job.h);
    if (sse_cfl_u + sse_cfl_v < best) {
      best = sse_cfl_u + sse_cfl_v;
      best_cfl = true;
      best_mode = 0;
      best_delta = 0;
      expect.pred_u = cfl_u;
      expect.pred_v = cfl_v;
    }
  }

  expect.result.winner = best_cfl
                             ? kM1ChromaCfl
                             : (best_mode == 0 ? kM1ChromaDc
                                               : kM1ChromaUvDirectional);
  expect.result.uv_mode = best_mode;
  expect.result.angle_delta = best_delta;
  expect.result.is_cfl = best_cfl ? 1 : 0;
  expect.result.alpha_u = alpha_u;
  expect.result.alpha_v = alpha_v;
  expect.result.dc_u = 0;  // filled by the caller from DC prediction
  expect.result.dc_v = 0;
  expect.result.sse_dc_u = sse_dc_u;
  expect.result.sse_dc_v = sse_dc_v;
  expect.result.sse_cfl_u = sse_cfl_u;
  expect.result.sse_cfl_v = sse_cfl_v;
  expect.result.sse = best;
  return true;
}

void run_case() {
  M1ChromaJob job{};
  M1ChromaOptions opt{};
  opt.enable_uv_full = 1;
  opt.enable_cfl = random_range(0, 1);
  opt.ds_filter = random_range(0, 3);

  const int dims[] = {4, 8, 16, 32, 64};
  job.w = dims[random_range(0, 4)];
  job.h = dims[random_range(0, 4)];
  job.coded_width = random_range(job.w, 96);
  job.coded_height = random_range(job.h, 96);
  job.x = random_range(0, job.coded_width - job.w);
  job.y = random_range(0, job.coded_height - job.h);
  job.chroma_stride = job.coded_width + 32;
  const int chroma_rows = job.coded_height + 96;
  const int luma_stride = job.chroma_stride * 2;
  const int luma_rows = chroma_rows * 2;

  std::vector<std::uint8_t> luma(static_cast<std::size_t>(luma_stride) * luma_rows);
  std::vector<std::uint8_t> cu(static_cast<std::size_t>(job.chroma_stride) * chroma_rows);
  std::vector<std::uint8_t> cv(cu.size());
  for (std::uint8_t& v : luma) v = static_cast<std::uint8_t>(next_random() & 0xFFu);
  for (std::uint8_t& v : cu) v = static_cast<std::uint8_t>(next_random() & 0xFFu);
  for (std::uint8_t& v : cv) v = static_cast<std::uint8_t>(next_random() & 0xFFu);

  job.luma = luma.data();
  job.luma_stride = luma_stride;
  job.luma_max_x = job.coded_width * 2 - 1;
  job.luma_max_y = job.coded_height * 2 - 1;
  job.sb_top_y = (job.y * 2 / 64) * 64;
  job.chroma_u = cu.data();
  job.chroma_v = cv.data();
  job.chroma_stride = job.chroma_stride;
  job.src_max_x = job.coded_width - 1;
  job.src_max_y = job.coded_height - 1;
  job.src_stride_u = job.coded_width;
  job.src_stride_v = job.coded_width;
  std::vector<std::uint8_t> src_u(static_cast<std::size_t>(job.coded_width) * job.coded_height);
  std::vector<std::uint8_t> src_v(src_u.size());
  for (std::uint8_t& v : src_u) v = static_cast<std::uint8_t>(next_random() & 0xFFu);
  for (std::uint8_t& v : src_v) v = static_cast<std::uint8_t>(next_random() & 0xFFu);
  job.src_u = src_u.data();
  job.src_v = src_v.data();
  std::vector<std::uint8_t> pred_u(static_cast<std::size_t>(job.w) * job.h);
  std::vector<std::uint8_t> pred_v(pred_u.size());
  job.pred_u = pred_u.data();
  job.pred_v = pred_v.data();
  job.y_mode = random_range(0, 12);
  job.y_angle_delta = random_range(-3, 3);
  job.filter_type_above = random_range(0, 1);
  job.filter_type_left = random_range(0, 1);
  job.enable_edge_filter = random_range(0, 1);
  job.num4_above_right = random_range(0, job.w / 4 + 2);
  job.num4_below_left = random_range(0, job.h / 4 + 2);

  CpuExpectation expect;
  if (!cpu_full_uv(job, opt, expect)) {
    std::printf("CPU reference failed\n");
    ++g_checks;
    ++g_failures;
    return;
  }
  // Fill the DC diagnostics the CPU path would report.
  std::vector<std::uint8_t> dc_u;
  std::vector<std::uint8_t> dc_v;
  if (cpu_predict(job, 1, 0, 0, dc_u) && cpu_predict(job, 2, 0, 0, dc_v)) {
    expect.result.dc_u = dc_u[0];
    expect.result.dc_v = dc_v[0];
  }

  M1ChromaResult got{};
  static ChromaUvShared shared;
  blockDim.x = 1;
  threadIdx.x = 0;
  av2enc::cuda::m1::chroma_uv_full_search(job, opt, &shared, &got);
  blockDim.x = 0;

  ++g_checks;
  const bool decision_mismatch =
      got.winner != expect.result.winner || got.uv_mode != expect.result.uv_mode ||
      got.angle_delta != expect.result.angle_delta ||
      got.is_cfl != expect.result.is_cfl || got.sse != expect.result.sse ||
      got.sse_dc_u != expect.result.sse_dc_u ||
      got.sse_dc_v != expect.result.sse_dc_v ||
      got.sse_cfl_u != expect.result.sse_cfl_u ||
      got.sse_cfl_v != expect.result.sse_cfl_v ||
      got.alpha_u != expect.result.alpha_u ||
      got.alpha_v != expect.result.alpha_v || got.dc_u != expect.result.dc_u ||
      got.dc_v != expect.result.dc_v;
  if (decision_mismatch) {
    ++g_failures;
    if (g_failures <= 20) {
      std::printf(
          "FAIL decision: winner %d/%d uv %d/%d delta %d/%d cfl %d/%d "
          "sse %lld/%lld dc %lld,%lld/%lld,%lld cfl %lld,%lld/%lld,%lld "
          "alpha %d,%d/%d,%d dcval %d,%d/%d,%d\n",
          got.winner, expect.result.winner, got.uv_mode, expect.result.uv_mode,
          got.angle_delta, expect.result.angle_delta, got.is_cfl,
          expect.result.is_cfl, got.sse, expect.result.sse, got.sse_dc_u,
          got.sse_dc_v, expect.result.sse_dc_u, expect.result.sse_dc_v,
          got.sse_cfl_u, got.sse_cfl_v, expect.result.sse_cfl_u,
          expect.result.sse_cfl_v, got.alpha_u, got.alpha_v,
          expect.result.alpha_u, expect.result.alpha_v, got.dc_u, got.dc_v,
          expect.result.dc_u, expect.result.dc_v);
    }
    return;
  }

  for (std::size_t i = 0; i < pred_u.size(); ++i) {
    if (pred_u[i] != expect.pred_u[i] || pred_v[i] != expect.pred_v[i]) {
      ++g_failures;
      if (g_failures <= 20) {
        std::printf("FAIL pred sample %zu U %u/%u V %u/%u\n", i,
                    static_cast<unsigned>(pred_u[i]),
                    static_cast<unsigned>(expect.pred_u[i]),
                    static_cast<unsigned>(pred_v[i]),
                    static_cast<unsigned>(expect.pred_v[i]));
      }
      return;
    }
  }
}

}  // namespace

int main() {
  for (int trial = 0; trial < 4000; ++trial) {
    run_case();
  }
  std::printf("chroma_uv_emulation_test: %s (%d checks)\n",
              g_failures == 0 ? "PASS" : "FAIL", g_checks);
  return g_failures == 0 ? 0 : 1;
}

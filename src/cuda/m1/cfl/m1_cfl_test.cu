// M1 GPU CfL stage test: CPU vs GPU bit-exactness.
//
// The CPU reference is src/encoder/cfl/cfl.cpp (the correctness oracle); the
// kernels are src/cuda/cfl/cfl.cu, reached through the M1 adapter in
// m1_cfl.cu. Coverage:
//
//   1. Geometry sweep 4x4 .. 64x64, all four cfl_ds_filter_index values,
//      every chroma availability combination (x/y 0..3 included), padded and
//      coded luma bounds, and partial blocks whose coded size crosses the
//      coded chroma grid.
//   2. Derived alpha 0..511: a constructed 4x4 luma/chroma family walks every
//      reachable implicitAlpha and every one is checked against the CPU and
//      against the GPU-reported alpha. U and V carry different alphas in the
//      same pair where the luma geometry allows it.
//   3. Explicit alpha: every Q8 value -512..512 on an 8x8 block (U = +q,
//      V = -q), plus the signalable multiples of 32.
//   4. Real frames (Y4M): 64x64 and 640x360, plus a 632x356 crop of the latter
//      for right-edge coded partials. Blocks are enumerated with M1 leaf
//      geometries (64/32/16/8 and rectangular), chroma DC prediction is
//      produced by src/encoder/intra_modes, and every block is replayed with
//      the exact frame state M1 had when it coded that block.
//   5. Timing of one whole-frame 640x360 batch.
//
// Build/run with build_m1_cfl_test.sh on the sm_86 box. Test-only device
// allocations are cudaMalloc; the stage itself never allocates or synchronizes.

#include "m1_cfl.h"

#include "encoder/cfl/cfl.h"
#include "encoder/intra_modes/intra_modes.h"
#include "encoder/partition/partition.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using av2enc::cfl::Block;
using av2enc::cuda::CflJob;
using av2enc::cuda::m1::M1CflFrame;
using av2enc::cuda::m1::M1CflPair;
using av2enc::cuda::m1::M1CflPairExplicit;
using av2enc::cuda::m1::M1CflRect;

constexpr int kLumaStride = 192;
constexpr int kLumaRows = 192;
constexpr int kChromaStride = 128;
constexpr int kChromaRows = 128;
constexpr std::size_t kLumaPlane =
    static_cast<std::size_t>(kLumaStride) * kLumaRows;
constexpr std::size_t kChromaPlane =
    static_cast<std::size_t>(kChromaStride) * kChromaRows;

#define CUDA_CHECK(expr)                                                     \
  do {                                                                       \
    const cudaError_t err_ = (expr);                                         \
    if (err_ != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error: %s (%d) from %s at %s:%d\n",         \
                   cudaGetErrorString(err_), static_cast<int>(err_), #expr,  \
                   __FILE__, __LINE__);                                      \
      std::exit(1);                                                          \
    }                                                                        \
  } while (false)

struct Counters {
  long long cases = 0;
  long long samples = 0;
  long long mismatches = 0;
  long long alpha_checks = 0;
  long long alpha_mismatches = 0;
  int printed = 0;
};

Counters g;

void note(const char* what, long long index, int expected, int actual) {
  if (g.printed >= 20) {
    return;
  }
  std::printf("  MISMATCH [%s] index %lld expected %d got %d\n", what, index,
              expected, actual);
  ++g.printed;
}

std::uint8_t* at(std::vector<std::uint8_t>& plane, int stride, int row,
                 int col) {
  return &plane[static_cast<std::size_t>(row) *
                    static_cast<std::size_t>(stride) +
                static_cast<std::size_t>(col)];
}

// ---------------------------------------------------------------------------
// Y4M input
// ---------------------------------------------------------------------------

struct Y4mFrame {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
};

bool load_y4m(const std::string& path, Y4mFrame& out, std::string& error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    error = "cannot open " + path;
    return false;
  }
  std::string header;
  std::getline(in, header);
  if (header.rfind("YUV4MPEG2", 0) != 0) {
    error = "not a Y4M file: " + path;
    return false;
  }
  bool chroma_420 = false;
  std::size_t pos = 0;
  while (pos < header.size()) {
    while (pos < header.size() && header[pos] == ' ') {
      ++pos;
    }
    if (pos >= header.size()) {
      break;
    }
    const char tag = header[pos];
    const std::size_t start = ++pos;
    while (pos < header.size() && header[pos] != ' ') {
      ++pos;
    }
    const std::string value = header.substr(start, pos - start);
    if (tag == 'W') {
      out.width = std::atoi(value.c_str());
    } else if (tag == 'H') {
      out.height = std::atoi(value.c_str());
    } else if (tag == 'C') {
      chroma_420 = value == "420" || value == "420jpeg" ||
                   value == "420mpeg2" || value == "420paldv";
    }
  }
  if (out.width <= 0 || out.height <= 0 || !chroma_420) {
    error = "only 8-bit 4:2:0 Y4M is supported: " + path;
    return false;
  }
  std::string frame_header;
  std::getline(in, frame_header);
  if (frame_header.rfind("FRAME", 0) != 0) {
    error = "missing FRAME marker: " + path;
    return false;
  }
  const std::size_t luma_size =
      static_cast<std::size_t>(out.width) * out.height;
  const std::size_t chroma_size =
      static_cast<std::size_t>((out.width + 1) / 2) *
      static_cast<std::size_t>((out.height + 1) / 2);
  out.y.resize(luma_size);
  out.u.resize(chroma_size);
  out.v.resize(chroma_size);
  in.read(reinterpret_cast<char*>(out.y.data()),
          static_cast<std::streamsize>(luma_size));
  in.read(reinterpret_cast<char*>(out.u.data()),
          static_cast<std::streamsize>(chroma_size));
  in.read(reinterpret_cast<char*>(out.v.data()),
          static_cast<std::streamsize>(chroma_size));
  if (!in) {
    error = "truncated Y4M frame: " + path;
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Batch replay
//
// One BatchCase is one M1 block: the U/V chroma planes hold exactly the
// samples CfL may read (DC prediction written, reconstructed neighbours) plus
// the shared reconstructed luma. Cases are uploaded and replayed in a single
// stage launch, which is what "batch U+V and independent blocks" means here.
// ---------------------------------------------------------------------------

struct BatchCase {
  std::vector<std::uint8_t> luma_own;  // used when luma_ref is null
  const std::vector<std::uint8_t>* luma_ref = nullptr;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
  M1CflPair host{};  // host pointers patched by run_batch
  int alpha_u = 0;
  int alpha_v = 0;
};

// Runs the CPU reference on one case and compares with the GPU result. The
// caller passes the GPU chroma plane bytes and the derived alphas (when
// `derived`).
void compare_case(BatchCase& c, const std::vector<std::uint8_t>& gpu_u,
                  const std::vector<std::uint8_t>& gpu_v, bool derived,
                  int gpu_alpha_u, int gpu_alpha_v) {
  Block bu = c.host.u;
  Block bv = c.host.v;
  std::vector<std::uint8_t> cpu_u = c.u;
  std::vector<std::uint8_t> cpu_v = c.v;
  bu.chroma = cpu_u.data();
  bv.chroma = cpu_v.data();

  av2enc::cfl::Scratch scratch;
  av2enc::cfl::compute_luma(bu, scratch);
  const int alpha_u =
      derived ? av2enc::cfl::derive_implicit_alpha(bu, scratch) : c.alpha_u;
  av2enc::cfl::apply_prediction(bu, alpha_u, scratch);
  av2enc::cfl::compute_luma(bv, scratch);
  const int alpha_v =
      derived ? av2enc::cfl::derive_implicit_alpha(bv, scratch) : c.alpha_v;
  av2enc::cfl::apply_prediction(bv, alpha_v, scratch);

  ++g.cases;
  if (derived) {
    ++g.alpha_checks;
    if (gpu_alpha_u != alpha_u) {
      ++g.alpha_mismatches;
      note("derived_alpha_u", g.cases,
           alpha_u, gpu_alpha_u);
    }
    ++g.alpha_checks;
    if (gpu_alpha_v != alpha_v) {
      ++g.alpha_mismatches;
      note("derived_alpha_v", g.cases,
           alpha_v, gpu_alpha_v);
    }
  }

  const int x = bu.x;
  const int y = bu.y;
  const int w = bu.w;
  const int h = bu.h;
  const int stride = bu.chroma_stride;
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      const std::size_t index = static_cast<std::size_t>(y + i) *
                                    static_cast<std::size_t>(stride) +
                                static_cast<std::size_t>(x + j);
      ++g.samples;
      if (gpu_u[index] != cpu_u[index]) {
        ++g.mismatches;
        note("prediction_u", g.cases, cpu_u[index], gpu_u[index]);
      }
      ++g.samples;
      if (gpu_v[index] != cpu_v[index]) {
        ++g.mismatches;
        note("prediction_v", g.cases, cpu_v[index], gpu_v[index]);
      }
    }
  }
}

// `shared_luma` is the plane used by every case whose luma_ref is set (real
// frames). Cases with luma_own get their own device copy.
void run_batch(std::vector<BatchCase>& cases,
               const std::vector<std::uint8_t>* shared_luma, bool derived) {
  const int count = static_cast<int>(cases.size());
  if (count == 0) {
    return;
  }

  // Concatenate the per-case planes and record their offsets.
  std::vector<std::uint8_t> luma_own_all;
  std::vector<std::uint8_t> u_all;
  std::vector<std::uint8_t> v_all;
  std::vector<std::size_t> u_offset(cases.size());
  std::vector<std::size_t> v_offset(cases.size());
  std::vector<std::size_t> luma_offset(cases.size());
  for (std::size_t i = 0; i < cases.size(); ++i) {
    BatchCase& c = cases[i];
    luma_offset[i] = luma_own_all.size();
    if (c.luma_ref == nullptr) {
      luma_own_all.insert(luma_own_all.end(), c.luma_own.begin(),
                          c.luma_own.end());
    }
    u_offset[i] = u_all.size();
    u_all.insert(u_all.end(), c.u.begin(), c.u.end());
    v_offset[i] = v_all.size();
    v_all.insert(v_all.end(), c.v.begin(), c.v.end());
  }

  // Device luma base: the shared plane first, then per-case copies.
  std::uint8_t* d_luma_base = nullptr;
  const std::size_t shared_bytes =
      shared_luma != nullptr ? shared_luma->size() : 0;
  CUDA_CHECK(cudaMalloc(&d_luma_base, shared_bytes + luma_own_all.size()));
  if (shared_luma != nullptr) {
    CUDA_CHECK(cudaMemcpy(d_luma_base, shared_luma->data(), shared_bytes,
                          cudaMemcpyHostToDevice));
  }
  if (!luma_own_all.empty()) {
    CUDA_CHECK(cudaMemcpy(d_luma_base + shared_bytes, luma_own_all.data(),
                          luma_own_all.size(), cudaMemcpyHostToDevice));
  }
  std::uint8_t* d_u = nullptr;
  std::uint8_t* d_v = nullptr;
  CUDA_CHECK(cudaMalloc(&d_u, u_all.size()));
  CUDA_CHECK(cudaMalloc(&d_v, v_all.size()));
  CUDA_CHECK(cudaMemcpy(d_u, u_all.data(), u_all.size(),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, v_all.data(), v_all.size(),
                        cudaMemcpyHostToDevice));

  std::vector<M1CflPair> device_pairs(cases.size());
  std::vector<M1CflPairExplicit> device_explicit(cases.size());
  for (std::size_t i = 0; i < cases.size(); ++i) {
    BatchCase& c = cases[i];
    c.host.u.chroma = c.u.data();
    c.host.v.chroma = c.v.data();
    const std::uint8_t* luma_ptr =
        c.luma_ref != nullptr ? d_luma_base
                              : d_luma_base + shared_bytes + luma_offset[i];
    M1CflPair& dp = device_pairs[i];
    dp = c.host;
    dp.u.luma = luma_ptr;
    dp.v.luma = luma_ptr;
    dp.u.chroma = d_u + u_offset[i];
    dp.v.chroma = d_v + v_offset[i];
    device_explicit[i].u.block = dp.u;
    device_explicit[i].u.alpha_q8 = c.alpha_u;
    device_explicit[i].v.block = dp.v;
    device_explicit[i].v.alpha_q8 = c.alpha_v;
  }

  std::vector<int> alphas(static_cast<std::size_t>(count) * 2, 0);
  M1CflPair* d_pairs = nullptr;
  M1CflPairExplicit* d_explicit = nullptr;
  int* d_alpha = nullptr;
  CUDA_CHECK(cudaMalloc(&d_pairs, sizeof(M1CflPair) * cases.size()));
  CUDA_CHECK(cudaMalloc(&d_explicit,
                        sizeof(M1CflPairExplicit) * cases.size()));
  CUDA_CHECK(cudaMalloc(&d_alpha, sizeof(int) * alphas.size()));
  CUDA_CHECK(cudaMemcpy(d_pairs, device_pairs.data(),
                        sizeof(M1CflPair) * cases.size(),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_explicit, device_explicit.data(),
                        sizeof(M1CflPairExplicit) * cases.size(),
                        cudaMemcpyHostToDevice));

  if (derived) {
    CUDA_CHECK(av2enc::cuda::m1::m1_cfl_predict_pairs(d_pairs, count, d_alpha));
  } else {
    CUDA_CHECK(av2enc::cuda::m1::m1_cfl_predict_pairs_explicit(d_explicit, count));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<std::uint8_t> gpu_u(u_all.size());
  std::vector<std::uint8_t> gpu_v(v_all.size());
  CUDA_CHECK(cudaMemcpy(gpu_u.data(), d_u, gpu_u.size(),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(gpu_v.data(), d_v, gpu_v.size(),
                        cudaMemcpyDeviceToHost));
  if (derived) {
    CUDA_CHECK(cudaMemcpy(alphas.data(), d_alpha, sizeof(int) * alphas.size(),
                          cudaMemcpyDeviceToHost));
  }

  for (std::size_t i = 0; i < cases.size(); ++i) {
    std::vector<std::uint8_t> cu(cases[i].u.size());
    std::vector<std::uint8_t> cv(cases[i].v.size());
    std::copy(gpu_u.begin() + static_cast<std::ptrdiff_t>(u_offset[i]),
              gpu_u.begin() + static_cast<std::ptrdiff_t>(u_offset[i] +
                                                          cu.size()),
              cu.begin());
    std::copy(gpu_v.begin() + static_cast<std::ptrdiff_t>(v_offset[i]),
              gpu_v.begin() + static_cast<std::ptrdiff_t>(v_offset[i] +
                                                          cv.size()),
              cv.begin());
    compare_case(cases[i], cu, cv, derived,
                 derived ? alphas[2 * i] : 0,
                 derived ? alphas[2 * i + 1] : 0);
  }

  CUDA_CHECK(cudaFree(d_pairs));
  CUDA_CHECK(cudaFree(d_explicit));
  CUDA_CHECK(cudaFree(d_alpha));
  CUDA_CHECK(cudaFree(d_u));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_luma_base));
}

// ---------------------------------------------------------------------------
// Synthetic geometry sweep
// ---------------------------------------------------------------------------

void fill_luma_pattern(std::vector<std::uint8_t>& plane) {
  for (int row = 0; row < kLumaRows; ++row) {
    for (int col = 0; col < kLumaStride; ++col) {
      *at(plane, kLumaStride, row, col) =
          static_cast<std::uint8_t>((row * 7 + col * 11 + (row * col) / 13) &
                                    0xFF);
    }
  }
}

void fill_chroma_pattern(std::vector<std::uint8_t>& plane, int salt) {
  for (int row = 0; row < kChromaRows; ++row) {
    for (int col = 0; col < kChromaStride; ++col) {
      *at(plane, kChromaStride, row, col) = static_cast<std::uint8_t>(
          (row * 5 + col * 3 + salt + (row * col) / 7) & 0xFF);
    }
  }
}

void add_synth_case(std::vector<BatchCase>& cases, int coded_w, int coded_h,
                    int w, int h, int x, int y, int filter, int luma_max_x,
                    int luma_max_y) {
  BatchCase c;
  c.luma_own.resize(kLumaPlane);
  fill_luma_pattern(c.luma_own);
  c.u.resize(kChromaPlane);
  c.v.resize(kChromaPlane);
  fill_chroma_pattern(c.u, 0);
  fill_chroma_pattern(c.v, 97);

  M1CflFrame frame;
  frame.luma = c.luma_own.data();
  frame.luma_stride = kLumaStride;
  frame.luma_max_x = luma_max_x;
  frame.luma_max_y = luma_max_y;
  frame.bit_depth = 8;
  frame.ds_filter = filter;

  M1CflRect ur;
  ur.chroma = c.u.data();
  ur.chroma_stride = kChromaStride;
  ur.x = x;
  ur.y = y;
  ur.w = w;
  ur.h = h;
  ur.coded_width = coded_w;
  ur.coded_height = coded_h;
  M1CflRect vr = ur;
  vr.chroma = c.v.data();
  c.host = av2enc::cuda::m1::make_m1_cfl_pair(frame, ur, vr);
  c.alpha_u = 32 * (1 + (x / 4) % 8);
  c.alpha_v = -32 * (1 + (y / 4) % 8);
  cases.push_back(std::move(c));
}

void test_geometry_sweep() {
  std::printf("== synthetic geometry sweep ==\n");
  std::vector<BatchCase> derived;
  const int sizes[] = {4, 8, 16, 32};
  const int offsets[] = {0, 1, 2, 3, 4, 8, 16, 32, 64};
  for (int filter = 0; filter <= 3; ++filter) {
    for (int size : sizes) {
      for (int x : offsets) {
        for (int y : offsets) {
          if (x + size > kChromaStride || y + size > kChromaRows) {
            continue;
          }
          // Coded plane sometimes stops inside the block, which drives the
          // right/bottom prevSample replication.
          const int coded_w = (x + size > 64) ? 64 : (x + size);
          const int coded_h = (y + size > 64) ? 64 : (y + size);
          add_synth_case(derived, coded_w, coded_h, size, size, x, y, filter,
                         kLumaStride - 1, kLumaRows - 1);
        }
      }
    }
  }
  // 64-wide blocks need a valid chroma stride and exercise the >32 step rule.
  const int big[] = {0, 1, 4, 8, 16, 32};
  for (int filter = 0; filter <= 3; ++filter) {
    for (int x : big) {
      for (int y : big) {
        add_synth_case(derived, 128, 128, 64, 64, x, y, filter,
                       kLumaStride - 1, kLumaRows - 1);
        // Coded bounds clamp to the 64x64 luma region so L and the luma
        // average read padding.
        add_synth_case(derived, 64, 64, 64, 64, x, y, filter, 95, 95);
      }
    }
  }
  run_batch(derived, nullptr, true);

  std::printf("== synthetic explicit alpha ==\n");
  run_batch(derived, nullptr, false);
}

void test_explicit_sweep() {
  std::printf("== explicit alpha Q8 sweep (-512..512) ==\n");
  std::vector<BatchCase> cases;
  for (int q = -512; q <= 512; ++q) {
    BatchCase c;
    c.luma_own.resize(kLumaPlane);
    fill_luma_pattern(c.luma_own);
    c.u.resize(kChromaPlane);
    c.v.resize(kChromaPlane);
    fill_chroma_pattern(c.u, 11);
    fill_chroma_pattern(c.v, 203);
    M1CflFrame frame;
    frame.luma = c.luma_own.data();
    frame.luma_stride = kLumaStride;
    frame.luma_max_x = kLumaStride - 1;
    frame.luma_max_y = kLumaRows - 1;
    frame.ds_filter = (q < 0 ? -q : q) & 3;
    M1CflRect ur;
    ur.chroma = c.u.data();
    ur.chroma_stride = kChromaStride;
    ur.x = 16;
    ur.y = 16;
    ur.w = 8;
    ur.h = 8;
    ur.coded_width = 96;
    ur.coded_height = 96;
    M1CflRect vr = ur;
    vr.chroma = c.v.data();
    c.host = av2enc::cuda::m1::make_m1_cfl_pair(frame, ur, vr);
    c.alpha_u = q;
    c.alpha_v = -q;
    cases.push_back(std::move(c));
  }
  run_batch(cases, nullptr, false);
}

// ---------------------------------------------------------------------------
// Derived alpha 0..511
//
// Construction: a 4x4 chroma block at (8, 8) with above+left available. The
// eight luma-line values are lx = 3k + k * [-3,-2,-1,0,1,2,3,0], which has
// mean 3k and der = 28k^2. The eight chroma neighbour values are chosen so
// nor = k * (s4 + 2*s5 + 3*s6); sweeping k and T hits every implicitAlpha
// 0..511. The test searches the constructions with the CPU oracle, then
// replays all 512 pairs on the GPU in one launch.
// ---------------------------------------------------------------------------

constexpr int kCoverLumaStride = 64;
constexpr int kCoverChromaStride = 32;

void fill_cover_planes(int k, int t_u, int t_v, std::vector<std::uint8_t>& luma,
                       std::vector<std::uint8_t>& u,
                       std::vector<std::uint8_t>& v) {
  const int m = 3 * k;
  const int c[8] = {-3, -2, -1, 0, 1, 2, 3, 0};
  int lx[8];
  for (int i = 0; i < 8; ++i) {
    lx[i] = m + k * c[i];
  }
  std::fill(luma.begin(), luma.end(), 0);
  // Above line: rows 14,15 at cols 16..23 (2x2 taps of filter 0).
  for (int i = 0; i < 4; ++i) {
    for (int dy = 0; dy < 2; ++dy) {
      for (int dx = 0; dx < 2; ++dx) {
        *at(luma, kCoverLumaStride, 14 + dy, 16 + 2 * i + dx) =
            static_cast<std::uint8_t>(lx[i]);
      }
    }
  }
  // Left line: rows 16..23 at cols 14,15.
  for (int j = 0; j < 4; ++j) {
    for (int dy = 0; dy < 2; ++dy) {
      for (int dx = 0; dx < 2; ++dx) {
        *at(luma, kCoverLumaStride, 16 + 2 * j + dy, 14 + dx) =
            static_cast<std::uint8_t>(lx[4 + j]);
      }
    }
  }

  const auto fill_chroma = [](std::vector<std::uint8_t>& plane, int t,
                              int salt) {
    std::fill(plane.begin(), plane.end(), 0);
    const int s6 = std::min(255, t / 3);
    const int s5 = std::min(255, (t - 3 * s6) / 2);
    const int s4 = t - 3 * s6 - 2 * s5;
    // Above row (y-1 = 7), cols x..x+3 = 8..11.
    for (int i = 0; i < 4; ++i) {
      *at(plane, kCoverChromaStride, 7, 8 + i) = 0;
    }
    // Left column (x-1 = 7), rows y..y+3 = 8..11.
    *at(plane, kCoverChromaStride, 8, 7) = static_cast<std::uint8_t>(s4);
    *at(plane, kCoverChromaStride, 9, 7) = static_cast<std::uint8_t>(s5);
    *at(plane, kCoverChromaStride, 10, 7) = static_cast<std::uint8_t>(s6);
    *at(plane, kCoverChromaStride, 11, 7) = 0;
    // DC prediction inside the block (not read by the alpha derivation).
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        *at(plane, kCoverChromaStride, 8 + i, 8 + j) =
            static_cast<std::uint8_t>((salt + i * 17 + j * 29) & 0xFF);
      }
    }
  };
  fill_chroma(u, t_u, 64);
  fill_chroma(v, t_v, 192);
}

M1CflPair make_cover_pair(const std::vector<std::uint8_t>& luma,
                        std::vector<std::uint8_t>& u,
                        std::vector<std::uint8_t>& v) {
  M1CflFrame frame;
  frame.luma = luma.data();
  frame.luma_stride = kCoverLumaStride;
  frame.luma_max_x = kCoverLumaStride - 1;
  frame.luma_max_y = kCoverLumaStride - 1;
  M1CflRect ur;
  ur.chroma = u.data();
  ur.chroma_stride = kCoverChromaStride;
  ur.x = 8;
  ur.y = 8;
  ur.w = 4;
  ur.h = 4;
  ur.coded_width = 24;
  ur.coded_height = 24;
  M1CflRect vr = ur;
  vr.chroma = v.data();
  return av2enc::cuda::m1::make_m1_cfl_pair(frame, ur, vr);
}

void test_derived_alpha_coverage() {
  std::printf("== derived alpha 0..511 coverage ==\n");
  struct Construct {
    int k = 0;
    int t = 0;
  };
  std::map<int, Construct> first;
  std::vector<std::vector<std::uint8_t>> luma_by_k(41);
  std::vector<std::uint8_t> u(kCoverChromaStride * kCoverChromaStride);
  std::vector<std::uint8_t> v(u.size());

  for (int k = 1; k <= 40; ++k) {
    luma_by_k[k].assign(kCoverLumaStride * kCoverLumaStride, 0);
    std::vector<std::uint8_t> dummy_u(u.size());
    std::vector<std::uint8_t> dummy_v(v.size());
    for (int t = 0; t <= 1530; ++t) {
      fill_cover_planes(k, t, t, luma_by_k[k], dummy_u, dummy_v);
      M1CflPair pair = make_cover_pair(luma_by_k[k], dummy_u, dummy_v);
      av2enc::cfl::Scratch scratch;
      av2enc::cfl::compute_luma(pair.u, scratch);
      const int alpha = av2enc::cfl::derive_implicit_alpha(pair.u, scratch);
      if (alpha >= 0 && alpha <= 511 && first.find(alpha) == first.end()) {
        first[alpha] = Construct{k, t};
      }
    }
  }
  if (static_cast<int>(first.size()) != 512) {
    std::fprintf(stderr,
                 "derived-alpha construction reached %zu/512 values; extend "
                 "the k/T family\n",
                 first.size());
    std::exit(1);
  }

  // Alternatives reachable with the same k, so U and V can carry different
  // alphas while sharing one luma plane (the M1 block contract).
  std::vector<std::vector<int>> by_k(41);
  for (const auto& entry : first) {
    by_k[static_cast<std::size_t>(entry.second.k)].push_back(entry.first);
  }

  std::vector<BatchCase> cases;
  cases.reserve(512);
  for (int alpha = 0; alpha <= 511; ++alpha) {
    const Construct cu = first[alpha];
    const std::vector<int>& group = by_k[static_cast<std::size_t>(cu.k)];
    const auto it = std::find(group.begin(), group.end(), alpha);
    const int beta = group[(static_cast<std::size_t>(it - group.begin()) + 1) %
                           group.size()];
    const Construct cv = first[beta];

    BatchCase c;
    c.luma_own = luma_by_k[static_cast<std::size_t>(cu.k)];
    c.u.assign(kCoverChromaStride * kCoverChromaStride, 0);
    c.v.assign(kCoverChromaStride * kCoverChromaStride, 0);
    fill_cover_planes(cu.k, cu.t, cv.t, c.luma_own, c.u, c.v);
    c.host = make_cover_pair(c.luma_own, c.u, c.v);
    cases.push_back(std::move(c));
  }

  // CPU sanity: the constructions really produce the intended alphas.
  for (int alpha = 0; alpha <= 511; ++alpha) {
    BatchCase& c = cases[static_cast<std::size_t>(alpha)];
    av2enc::cfl::Scratch scratch;
    Block bu = c.host.u;
    std::vector<std::uint8_t> cu = c.u;
    bu.chroma = cu.data();
    av2enc::cfl::compute_luma(bu, scratch);
    const int got = av2enc::cfl::derive_implicit_alpha(bu, scratch);
    if (got != alpha) {
      std::fprintf(stderr, "coverage construction %d produced alpha %d\n",
                   alpha, got);
      std::exit(1);
    }
  }

  run_batch(cases, nullptr, true);
  std::printf("   exercised derived alphas 0..511 (512 pairs, %zu cases)\n",
              cases.size());
}

// ---------------------------------------------------------------------------
// Real frames
// ---------------------------------------------------------------------------

struct Frame {
  int width = 0;
  int height = 0;
  int mi_cols = 0;
  int mi_rows = 0;
  int luma_stride = 0;
  int luma_rows = 0;
  int chroma_stride = 0;
  int chroma_rows = 0;
  int luma_max_x = 0;
  int luma_max_y = 0;
  std::vector<std::uint8_t> luma;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
};

bool build_frame(const Y4mFrame& src, int width, int height, Frame& f,
                 std::string& error) {
  if (width <= 0 || height <= 0 || width > src.width || height > src.height) {
    error = "bad crop for frame";
    return false;
  }
  f.width = width;
  f.height = height;
  av2enc::partition::FrameGeometry geometry;
  geometry.width = width;
  geometry.height = height;
  geometry.sb_size = 64;
  geometry.has_chroma = true;
  geometry.chroma_subsampling_x = true;
  geometry.chroma_subsampling_y = true;
  if (!av2enc::partition::validate_geometry(geometry, error)) {
    return false;
  }
  f.mi_cols = av2enc::partition::mi_cols(geometry);
  f.mi_rows = av2enc::partition::mi_rows(geometry);
  f.luma_stride = (f.mi_cols * 4 + 63) & ~63;
  f.luma_rows = (f.mi_rows * 4 + 63) & ~63;
  f.chroma_stride = f.luma_stride / 2;
  f.chroma_rows = f.luma_rows / 2;
  f.luma_max_x = f.mi_cols * 4 - 1;
  f.luma_max_y = f.mi_rows * 4 - 1;
  // dav2d starts CurrFrame at 0; CfL reads the padding, so use 0, not 128.
  f.luma.assign(static_cast<std::size_t>(f.luma_stride) * f.luma_rows, 0);
  f.u.assign(static_cast<std::size_t>(f.chroma_stride) * f.chroma_rows, 0);
  f.v.assign(f.u.size(), 0);

  const int chroma_w = (width + 1) / 2;
  const int chroma_h = (height + 1) / 2;
  for (int row = 0; row < height; ++row) {
    std::memcpy(at(f.luma, f.luma_stride, row, 0),
                &src.y[static_cast<std::size_t>(row) * src.width],
                static_cast<std::size_t>(width));
  }
  for (int row = 0; row < chroma_h; ++row) {
    std::memcpy(at(f.u, f.chroma_stride, row, 0),
                &src.u[static_cast<std::size_t>(row) * chroma_w],
                static_cast<std::size_t>(chroma_w));
    std::memcpy(at(f.v, f.chroma_stride, row, 0),
                &src.v[static_cast<std::size_t>(row) * chroma_w],
                static_cast<std::size_t>(chroma_w));
  }
  return true;
}

struct Leaf {
  int lx = 0;
  int ly = 0;
  int lw = 0;
  int lh = 0;
};

av2enc::partition::BlockSize size_from_dims(int w, int h) {
  using av2enc::partition::BlockSize;
  for (int i = 0; i < av2enc::partition::kBlockSizeCount; ++i) {
    const auto size = static_cast<BlockSize>(i);
    if (av2enc::partition::is_valid_block_size(size) &&
        av2enc::partition::block_width(size) == w &&
        av2enc::partition::block_height(size) == h) {
      return size;
    }
  }
  return BlockSize::kInvalid;
}

void emit_leaves(int sbx, int sby, int pattern, std::vector<Leaf>& out) {
  const int x = sbx * 64;
  const int y = sby * 64;
  switch (pattern) {
    case 0:
      out.push_back(Leaf{x, y, 64, 64});
      break;
    case 1:
      for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
          out.push_back(Leaf{x + dx * 32, y + dy * 32, 32, 32});
        }
      }
      break;
    case 2:
      for (int dy = 0; dy < 4; ++dy) {
        for (int dx = 0; dx < 4; ++dx) {
          out.push_back(Leaf{x + dx * 16, y + dy * 16, 16, 16});
        }
      }
      break;
    case 3:
      for (int dy = 0; dy < 8; ++dy) {
        for (int dx = 0; dx < 8; ++dx) {
          out.push_back(Leaf{x + dx * 8, y + dy * 8, 8, 8});
        }
      }
      break;
    case 4:
      for (int dx = 0; dx < 2; ++dx) {
        out.push_back(Leaf{x + dx * 32, y, 32, 64});
      }
      break;
    default:
      for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
          const int qx = x + dx * 32;
          const int qy = y + dy * 32;
          if ((dx + dy) % 2 == 0) {
            out.push_back(Leaf{qx, qy, 32, 16});
            out.push_back(Leaf{qx, qy + 16, 32, 16});
          } else {
            out.push_back(Leaf{qx, qy, 16, 32});
            out.push_back(Leaf{qx + 16, qy, 16, 32});
          }
        }
      }
      break;
  }
}

bool dc_predict(std::vector<std::uint8_t>& plane, int stride, int x, int y,
                int w, int h, int max_x, int max_y, int color_plane,
                std::vector<std::uint8_t>& out) {
  out.resize(static_cast<std::size_t>(w) * h);
  av2enc::intra::NeighborRequest request;
  request.recon = av2enc::intra::PlaneView{plane.data(), stride};
  request.x = x;
  request.y = y;
  request.w = w;
  request.h = h;
  request.max_x = max_x;
  request.max_y = max_y;
  request.have_left = x > 0;
  request.have_above = y > 0;
  request.color_plane = color_plane;
  av2enc::intra::PredictParams params;
  params.mode = av2enc::intra::Mode::kDc;
  params.angle_delta = 0;
  params.enable_edge_filter = true;
  return av2enc::intra::predict_block(request, 8, params, out.data(),
                                      static_cast<std::size_t>(w)) ==
         av2enc::intra::IntraStatus::kOk;
}

struct RealRun {
  std::size_t pairs = 0;
  std::vector<BatchCase> cases;
  // Coverage evidence for the report/test gate.
  int chroma4x4 = 0;
  int chroma32x32 = 0;
  int partial_bottom = 0;
  int partial_right = 0;
  int sb_top_clamp = 0;
  int no_above = 0;
  int no_left = 0;
};

bool build_real_cases(Frame& f, RealRun& run, std::string& error) {
  av2enc::partition::FrameGeometry geometry;
  geometry.width = f.width;
  geometry.height = f.height;
  geometry.sb_size = 64;
  geometry.has_chroma = true;
  geometry.chroma_subsampling_x = true;
  geometry.chroma_subsampling_y = true;

  const int sb_cols = (f.mi_cols * 4 + 63) / 64;
  const int sb_rows = (f.mi_rows * 4 + 63) / 64;
  for (int sby = 0; sby < sb_rows; ++sby) {
    for (int sbx = 0; sbx < sb_cols; ++sbx) {
      const int pattern = (sbx * 3 + sby * 5) % 6;
      std::vector<Leaf> leaves;
      emit_leaves(sbx, sby, pattern, leaves);
      std::sort(leaves.begin(), leaves.end(), [](const Leaf& a, const Leaf& b) {
        return a.ly != b.ly ? a.ly < b.ly : a.lx < b.lx;
      });
      for (const Leaf& leaf : leaves) {
        const auto luma_size = size_from_dims(leaf.lw, leaf.lh);
        if (!av2enc::partition::is_valid_block_size(luma_size)) {
          continue;
        }
        const auto chroma_size =
            av2enc::partition::chroma_plane_residual_size(geometry, luma_size);
        if (!av2enc::partition::is_valid_block_size(chroma_size)) {
          continue;
        }
        const int cw = av2enc::partition::block_width(chroma_size);
        const int ch = av2enc::partition::block_height(chroma_size);
        const int cx = leaf.lx / 2;
        const int cy = leaf.ly / 2;
        // The superblock grid overhangs the coded plane when MiCols*4 is not a
        // multiple of 64. M1 never codes a leaf that starts outside the coded
        // plane; partially overhanging leaves are kept (they exercise the
        // right/bottom prevSample replication).
        if (cx >= f.mi_cols * 2 || cy >= f.mi_rows * 2) {
          continue;
        }
        if (cy + ch > f.chroma_rows || cx + cw > f.chroma_stride) {
          continue;  // leaf outside the padded plane (should not happen)
        }

        std::vector<std::uint8_t> dcu;
        std::vector<std::uint8_t> dcv;
        if (!dc_predict(f.u, f.chroma_stride, cx, cy, cw, ch, f.mi_cols * 2 - 1,
                        f.mi_rows * 2 - 1, 1, dcu) ||
            !dc_predict(f.v, f.chroma_stride, cx, cy, cw, ch, f.mi_cols * 2 - 1,
                        f.mi_rows * 2 - 1, 2, dcv)) {
          error = "intra DC prediction failed";
          return false;
        }
        for (int i = 0; i < ch; ++i) {
          std::memcpy(at(f.u, f.chroma_stride, cy + i, cx),
                      &dcu[static_cast<std::size_t>(i) * cw],
                      static_cast<std::size_t>(cw));
          std::memcpy(at(f.v, f.chroma_stride, cy + i, cx),
                      &dcv[static_cast<std::size_t>(i) * cw],
                      static_cast<std::size_t>(cw));
        }

        // Snapshot the exact state CfL sees: every row up to the block bottom
        // (above neighbour + coded block) and reconstructed neighbours.
        BatchCase c;
        c.luma_ref = &f.luma;
        const std::size_t rows = static_cast<std::size_t>(cy + ch);
        c.u.resize(rows * static_cast<std::size_t>(f.chroma_stride));
        c.v.resize(c.u.size());
        std::memcpy(c.u.data(), f.u.data(), c.u.size());
        std::memcpy(c.v.data(), f.v.data(), c.v.size());

        M1CflFrame frame;
        frame.luma = f.luma.data();
        frame.luma_stride = f.luma_stride;
        frame.luma_max_x = f.luma_max_x;
        frame.luma_max_y = f.luma_max_y;
        M1CflRect ur;
        ur.chroma = c.u.data();
        ur.chroma_stride = f.chroma_stride;
        ur.x = cx;
        ur.y = cy;
        ur.w = cw;
        ur.h = ch;
        ur.coded_width = f.mi_cols * 2;
        ur.coded_height = f.mi_rows * 2;
        M1CflRect vr = ur;
        vr.chroma = c.v.data();
        c.host = av2enc::cuda::m1::make_m1_cfl_pair(frame, ur, vr);
        c.alpha_u = 32 * ((leaf.lx / 8) % 8);
        c.alpha_v = -32 * (1 + (leaf.ly / 8) % 8);
        if (!av2enc::cuda::m1::m1_cfl_valid_pair(c.host, &error)) {
          return false;
        }

        if (cw == 4 && ch == 4) {
          ++run.chroma4x4;
        }
        if (cw == 32 && ch == 32) {
          ++run.chroma32x32;
        }
        if (cy + ch > f.mi_rows * 2) {
          ++run.partial_bottom;
        }
        if (cx + cw > f.mi_cols * 2) {
          ++run.partial_right;
        }
        // filtered_above() clamps its rows to max(sbTop - 1, ...). A block
        // whose luma top is an SB boundary with an available above neighbour
        // reads the clamped row.
        if (cy > 0 && (cy * 2) % 64 == 0 && c.host.u.have_above != 0) {
          ++run.sb_top_clamp;
        }
        if (c.host.u.have_above == 0) {
          ++run.no_above;
        }
        if (c.host.u.have_left == 0) {
          ++run.no_left;
        }

        // Advance the frame the way M1 does: CPU CfL reconstructs the block
        // before the next block's DC prediction reads it.
        Block bu = c.host.u;
        Block bv = c.host.v;
        std::vector<std::uint8_t> cu = c.u;
        std::vector<std::uint8_t> cv = c.v;
        bu.chroma = cu.data();
        bv.chroma = cv.data();
        av2enc::cfl::Scratch scratch;
        av2enc::cfl::compute_luma(bu, scratch);
        const int alpha_u = av2enc::cfl::derive_implicit_alpha(bu, scratch);
        av2enc::cfl::apply_prediction(bu, alpha_u, scratch);
        av2enc::cfl::compute_luma(bv, scratch);
        const int alpha_v = av2enc::cfl::derive_implicit_alpha(bv, scratch);
        av2enc::cfl::apply_prediction(bv, alpha_v, scratch);
        for (int i = 0; i < ch; ++i) {
          std::memcpy(at(f.u, f.chroma_stride, cy + i, cx),
                      &cu[static_cast<std::size_t>(cy + i) *
                              f.chroma_stride + cx],
                      static_cast<std::size_t>(cw));
          std::memcpy(at(f.v, f.chroma_stride, cy + i, cx),
                      &cv[static_cast<std::size_t>(cy + i) *
                              f.chroma_stride + cx],
                      static_cast<std::size_t>(cw));
        }
        run.cases.push_back(std::move(c));
      }
    }
  }
  run.pairs = run.cases.size();
  return true;
}

void test_real_frame(const Y4mFrame& src, int width, int height,
                     const char* label) {
  std::printf("== real frame %s (%dx%d) ==\n", label, width, height);
  Frame f;
  std::string error;
  if (!build_frame(src, width, height, f, error)) {
    std::fprintf(stderr, "build_frame: %s\n", error.c_str());
    std::exit(1);
  }
  RealRun run;
  if (!build_real_cases(f, run, error)) {
    std::fprintf(stderr, "build_real_cases: %s\n", error.c_str());
    std::exit(1);
  }
  std::printf("   %zu M1 block pairs (mi %dx%d, coded %dx%d chroma)\n",
              run.pairs, f.mi_cols, f.mi_rows, f.mi_cols * 2, f.mi_rows * 2);
  std::printf(
      "   coverage: 4x4=%d 32x32=%d partial_bottom=%d partial_right=%d "
      "sb_top_clamp=%d no_above=%d no_left=%d\n",
      run.chroma4x4, run.chroma32x32, run.partial_bottom, run.partial_right,
      run.sb_top_clamp, run.no_above, run.no_left);
  if (width == 640 && height == 360 &&
      (run.partial_bottom == 0 || run.sb_top_clamp == 0 ||
       run.chroma4x4 == 0 || run.chroma32x32 == 0)) {
    std::fprintf(stderr, "640x360 coverage is incomplete\n");
    std::exit(1);
  }
  if (width == 632 && run.partial_right == 0) {
    std::fprintf(stderr, "crop coverage is incomplete\n");
    std::exit(1);
  }
  run_batch(run.cases, &f.luma, true);

  // Explicit-alpha replay over the same real blocks.
  for (std::size_t i = 0; i < run.cases.size(); ++i) {
    run.cases[i].alpha_u = 32 * static_cast<int>((i * 5) % 9) - 128;
    run.cases[i].alpha_v = 32 * static_cast<int>((i * 7) % 9) - 128;
  }
  run_batch(run.cases, &f.luma, false);
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

void time_frame(const Y4mFrame& src) {
  std::printf("== timing 640x360 whole-frame batch ==\n");
  Frame f;
  std::string error;
  if (!build_frame(src, 640, 360, f, error)) {
    std::fprintf(stderr, "timing build_frame: %s\n", error.c_str());
    return;
  }
  RealRun run;
  if (!build_real_cases(f, run, error)) {
    std::fprintf(stderr, "timing build_real_cases: %s\n", error.c_str());
    return;
  }
  const int count = static_cast<int>(run.cases.size());
  std::vector<M1CflPair> host_pairs(static_cast<std::size_t>(count));
  std::vector<std::uint8_t> u_all;
  std::vector<std::uint8_t> v_all;
  std::vector<std::size_t> u_off(run.cases.size());
  std::vector<std::size_t> v_off(run.cases.size());
  std::size_t total = 0;
  for (std::size_t i = 0; i < run.cases.size(); ++i) {
    u_off[i] = total;
    total += run.cases[i].u.size();
    v_off[i] = total;
    total += run.cases[i].v.size();
  }
  u_all.resize(total);
  v_all.resize(total);
  for (std::size_t i = 0; i < run.cases.size(); ++i) {
    std::copy(run.cases[i].u.begin(), run.cases[i].u.end(),
              u_all.begin() + static_cast<std::ptrdiff_t>(u_off[i]));
    std::copy(run.cases[i].v.begin(), run.cases[i].v.end(),
              v_all.begin() + static_cast<std::ptrdiff_t>(v_off[i]));
  }

  std::uint8_t* d_luma = nullptr;
  std::uint8_t* d_u = nullptr;
  std::uint8_t* d_v = nullptr;
  M1CflPair* d_pairs = nullptr;
  int* d_alpha = nullptr;
  CUDA_CHECK(cudaMalloc(&d_luma, f.luma.size()));
  CUDA_CHECK(cudaMalloc(&d_u, u_all.size()));
  CUDA_CHECK(cudaMalloc(&d_v, v_all.size()));
  CUDA_CHECK(cudaMalloc(&d_pairs, sizeof(M1CflPair) * host_pairs.size()));
  CUDA_CHECK(cudaMalloc(&d_alpha, sizeof(int) * 2 * host_pairs.size()));
  CUDA_CHECK(cudaMemcpy(d_luma, f.luma.data(), f.luma.size(),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_u, u_all.data(), u_all.size(),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, v_all.data(), v_all.size(),
                        cudaMemcpyHostToDevice));
  for (std::size_t i = 0; i < run.cases.size(); ++i) {
    host_pairs[i] = run.cases[i].host;
    host_pairs[i].u.luma = d_luma;
    host_pairs[i].v.luma = d_luma;
    host_pairs[i].u.chroma = d_u + u_off[i];
    host_pairs[i].v.chroma = d_v + v_off[i];
  }
  CUDA_CHECK(cudaMemcpy(d_pairs, host_pairs.data(),
                        sizeof(M1CflPair) * host_pairs.size(),
                        cudaMemcpyHostToDevice));

  constexpr int kWarmup = 5;
  constexpr int kIterations = 50;
  for (int i = 0; i < kWarmup; ++i) {
    CUDA_CHECK(av2enc::cuda::m1::m1_cfl_predict_pairs(d_pairs, count, d_alpha));
  }
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < kIterations; ++i) {
    CUDA_CHECK(av2enc::cuda::m1::m1_cfl_predict_pairs(d_pairs, count, d_alpha));
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0F;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  const double per_frame = static_cast<double>(ms) / kIterations;
  std::printf("   %d pairs (%d plane jobs), %.3f ms/frame, %.1f ns/plane-job\n",
              count, count * 2, per_frame,
              count > 0 ? per_frame * 1.0e6 / (2.0 * count) : 0.0);
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_luma));
  CUDA_CHECK(cudaFree(d_u));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_pairs));
  CUDA_CHECK(cudaFree(d_alpha));
}

void test_api_edges() {
  std::printf("== API edge cases ==\n");
  // No-op and invalid-pointer handling.
  if (av2enc::cuda::m1::m1_cfl_predict_pairs(nullptr, 0) != cudaSuccess) {
    std::fprintf(stderr, "m1_cfl_predict_pairs(nullptr, 0) should be a no-op\n");
    std::exit(1);
  }
  if (av2enc::cuda::m1::m1_cfl_predict_pairs(nullptr, 1) != cudaErrorInvalidValue) {
    std::fprintf(stderr, "m1_cfl_predict_pairs(nullptr, 1) should be invalid\n");
    std::exit(1);
  }
  if (av2enc::cuda::m1::m1_cfl_predict_pairs_explicit(nullptr, 2) !=
      cudaErrorInvalidValue) {
    std::fprintf(stderr, "m1_cfl_predict_pairs_explicit(nullptr, 2) invalid\n");
    std::exit(1);
  }

  // m1_cfl_valid_pair rejects a U/V geometry mismatch.
  std::vector<std::uint8_t> luma(kCoverLumaStride * kCoverLumaStride, 128);
  std::vector<std::uint8_t> u(kCoverChromaStride * kCoverChromaStride, 64);
  std::vector<std::uint8_t> v(u.size(), 64);
  M1CflPair pair = make_cover_pair(luma, u, v);
  std::string error;
  if (!av2enc::cuda::m1::m1_cfl_valid_pair(pair, &error)) {
    std::fprintf(stderr, "m1_cfl_valid_pair rejected a valid pair: %s\n",
                 error.c_str());
    std::exit(1);
  }
  pair.v.y += 1;
  if (av2enc::cuda::m1::m1_cfl_valid_pair(pair, &error)) {
    std::fprintf(stderr, "m1_cfl_valid_pair accepted mismatched U/V geometry\n");
    std::exit(1);
  }

  // The alpha readback is optional: one pair through the ordinary bounded
  // path with d_alpha_q8 == nullptr must still match the CPU.
  std::vector<std::uint8_t> luma2(kCoverLumaStride * kCoverLumaStride, 0);
  std::vector<std::uint8_t> u2(kCoverChromaStride * kCoverChromaStride, 0);
  std::vector<std::uint8_t> v2(u2.size(), 0);
  fill_cover_planes(7, 700, 800, luma2, u2, v2);
  M1CflPair host = make_cover_pair(luma2, u2, v2);

  std::uint8_t* d_luma = nullptr;
  std::uint8_t* d_u = nullptr;
  std::uint8_t* d_v = nullptr;
  M1CflPair* d_pair = nullptr;
  CUDA_CHECK(cudaMalloc(&d_luma, luma2.size()));
  CUDA_CHECK(cudaMalloc(&d_u, u2.size()));
  CUDA_CHECK(cudaMalloc(&d_v, v2.size()));
  CUDA_CHECK(cudaMalloc(&d_pair, sizeof(M1CflPair)));
  CUDA_CHECK(cudaMemcpy(d_luma, luma2.data(), luma2.size(),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_u, u2.data(), u2.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, v2.data(), v2.size(), cudaMemcpyHostToDevice));
  M1CflPair device = host;
  device.u.luma = d_luma;
  device.v.luma = d_luma;
  device.u.chroma = d_u;
  device.v.chroma = d_v;
  CUDA_CHECK(cudaMemcpy(d_pair, &device, sizeof(M1CflPair),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(av2enc::cuda::m1::m1_cfl_predict_pairs(d_pair, 1, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<std::uint8_t> gpu_u(u2.size());
  CUDA_CHECK(cudaMemcpy(gpu_u.data(), d_u, gpu_u.size(),
                        cudaMemcpyDeviceToHost));

  av2enc::cfl::Scratch scratch;
  Block cpu_u = host.u;
  std::vector<std::uint8_t> cpu_u_plane = u2;
  cpu_u.chroma = cpu_u_plane.data();
  av2enc::cfl::compute_luma(cpu_u, scratch);
  const int alpha = av2enc::cfl::derive_implicit_alpha(cpu_u, scratch);
  av2enc::cfl::apply_prediction(cpu_u, alpha, scratch);
  for (int i = 0; i < cpu_u.h; ++i) {
    for (int j = 0; j < cpu_u.w; ++j) {
      const std::size_t index =
          static_cast<std::size_t>(cpu_u.y + i) * cpu_u.chroma_stride +
          static_cast<std::size_t>(cpu_u.x + j);
      if (gpu_u[index] != cpu_u_plane[index]) {
        std::fprintf(stderr, "nullptr alpha readback changed the prediction\n");
        std::exit(1);
      }
    }
  }
  CUDA_CHECK(cudaFree(d_luma));
  CUDA_CHECK(cudaFree(d_u));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_pair));

  std::printf("   no-op, null, alpha-optional and validation paths OK\n");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr,
                 "usage: %s <64x64.y4m> <640x360.y4m>\n", argv[0]);
    return 2;
  }
  Y4mFrame small;
  Y4mFrame large;
  std::string error;
  if (!load_y4m(argv[1], small, error) || !load_y4m(argv[2], large, error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 2;
  }
  if (small.width != 64 || small.height != 64 || large.width != 640 ||
      large.height != 360) {
    std::fprintf(stderr, "unexpected Y4M dimensions\n");
    return 2;
  }

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  std::printf("m1_cfl_test: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
  test_api_edges();
  test_geometry_sweep();
  test_explicit_sweep();
  test_derived_alpha_coverage();
  test_real_frame(small, 64, 64, "testsrc2 64x64");
  test_real_frame(large, 640, 360, "testsrc2 640x360");
  test_real_frame(large, 632, 356, "640x360 crop 632x356");
  time_frame(large);

  if (g.mismatches == 0 && g.alpha_mismatches == 0) {
    std::printf(
        "m1_cfl_test: %lld cases, %lld samples, %lld derived alphas, all "
        "exact\n",
        g.cases, g.samples, g.alpha_checks);
    return 0;
  }
  std::fprintf(stderr,
               "m1_cfl_test: %lld/%lld samples and %lld/%lld alphas FAILED\n",
               g.mismatches, g.samples, g.alpha_mismatches, g.alpha_checks);
  return 1;
}

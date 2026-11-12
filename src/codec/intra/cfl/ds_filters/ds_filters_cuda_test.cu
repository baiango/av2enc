// CPU-vs-GPU verification for the CfL down-sampling index.
//
// The existing CUDA CfL path (src/cuda/cfl/) already honours cfl::Block::
// ds_filter and shares ds_weight() with the CPU reference.  This test does not
// edit that path; it proves the seam used by this component:
//
//   encoder search -> int cfl_ds_filter_index
//                  -> Block.ds_filter (the only ABI field)
//                  -> cfl_predict() / cfl_predict_derived()
//
// For every filter and every edge geometry it compares three predictions:
//   * ds_filters.cpp (this component),
//   * cfl.cpp (existing CPU reference),
//   * cfl.cu (existing GPU kernel).
//
// Build/run on the CUDA box with build_cuda_test.sh.  There is no nvcc on the
// macOS reference host, so this file is not part of the default CPU test run.

#include "ds_filters.h"

#include "../../../../cuda/codec/intra/cfl/cfl.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

using av2enc::cfl::Block;
using av2enc::cfl::Scratch;
using av2enc::cfl::SearchMode;
using av2enc::cfl::SourcePlane;
using av2enc::cfl::ds::FilterChoice;
using av2enc::cfl::ds::LumaScratch;
using av2enc::cfl::ds::SearchOptions;

int g_checks = 0;
int g_failures = 0;

#define CUDA_CHECK(expr)                                                   \
  do {                                                                     \
    const cudaError_t status = (expr);                                     \
    ++g_checks;                                                            \
    if (status != cudaSuccess) {                                           \
      ++g_failures;                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s -> %s\n", __FILE__, __LINE__,   \
                   #expr, cudaGetErrorString(status));                     \
    }                                                                      \
  } while (0)

#define CHECK(cond)                                                       \
  do {                                                                    \
    ++g_checks;                                                           \
    if (!(cond)) {                                                        \
      ++g_failures;                                                       \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    }                                                                     \
  } while (0)

struct Fixture {
  int w;
  int h;
  int x;
  int y;
  int coded_w;
  int coded_h;
  int sb_top_y;
  int have_above;
  int have_left;
  std::vector<std::uint8_t> luma;
  std::vector<std::uint8_t> chroma;
  int luma_stride;
  int chroma_stride;
};

Fixture make_fixture(int w, int h, int x, int y, int coded_w, int coded_h,
                     int sb_top_y, int luma_w, int luma_h, int seed) {
  Fixture f{};
  f.w = w;
  f.h = h;
  f.x = x;
  f.y = y;
  f.coded_w = coded_w;
  f.coded_h = coded_h;
  f.sb_top_y = sb_top_y;
  f.have_above = av2enc::cfl::cfl_avail_above(y, 1) ? 1 : 0;
  f.have_left = av2enc::cfl::cfl_avail_left(x, 1) ? 1 : 0;
  f.luma_stride = luma_w;
  f.chroma_stride = std::max(coded_w, x + w);
  const int chroma_h = std::max(coded_h, y + h);
  f.luma.resize(static_cast<std::size_t>(luma_w) *
                static_cast<std::size_t>(luma_h));
  f.chroma.resize(static_cast<std::size_t>(f.chroma_stride) *
                  static_cast<std::size_t>(chroma_h));
  for (std::size_t i = 0; i < f.luma.size(); ++i) {
    f.luma[i] = static_cast<std::uint8_t>((i * 37 + seed * 11) & 255);
  }
  for (std::size_t i = 0; i < f.chroma.size(); ++i) {
    f.chroma[i] = static_cast<std::uint8_t>((i * 29 + seed * 7 + 64) & 255);
  }
  return f;
}

Block host_block(const Fixture& f, int filter, std::uint8_t* chroma) {
  Block b{};
  b.luma = f.luma.data();
  b.luma_stride = f.luma_stride;
  b.luma_max_x = f.luma_stride - 1;
  b.luma_max_y = static_cast<int>(f.luma.size() / f.luma_stride) - 1;
  b.sb_top_y = f.sb_top_y;
  b.chroma = chroma;
  b.chroma_stride = f.chroma_stride;
  b.x = f.x;
  b.y = f.y;
  b.w = f.w;
  b.h = f.h;
  b.coded_width = f.coded_w;
  b.coded_height = f.coded_h;
  b.have_above = f.have_above;
  b.have_left = f.have_left;
  b.bit_depth = 8;
  b.ds_filter = filter;
  b.sub_x = 1;
  b.sub_y = 1;
  return b;
}

void run_case(const Fixture& f, int filter, int alpha, bool derived) {
  std::vector<std::uint8_t> ours = f.chroma;
  std::vector<std::uint8_t> reference = f.chroma;

  Block ours_block = host_block(f, filter, ours.data());
  Block ref_block = host_block(f, filter, reference.data());

  LumaScratch luma;
  av2enc::cfl::ds::compute_luma(ours_block, filter, luma);
  Scratch ref_scratch;
  av2enc::cfl::compute_luma(ref_block, ref_scratch);

  int chosen_alpha = alpha;
  if (derived) {
    chosen_alpha = av2enc::cfl::ds::derive_implicit_alpha(ours_block, luma);
    CHECK(chosen_alpha ==
          av2enc::cfl::derive_implicit_alpha(ref_block, ref_scratch));
  }
  av2enc::cfl::ds::apply_prediction(ours_block, chosen_alpha, luma);
  av2enc::cfl::apply_prediction(ref_block, chosen_alpha, ref_scratch);
  CHECK(std::memcmp(ours.data(), reference.data(), ours.size()) == 0);

  std::uint8_t* d_luma = nullptr;
  std::uint8_t* d_chroma = nullptr;
  int* d_alpha = nullptr;
  CUDA_CHECK(cudaMalloc(&d_luma, f.luma.size()));
  CUDA_CHECK(cudaMalloc(&d_chroma, f.chroma.size()));
  CUDA_CHECK(cudaMalloc(&d_alpha, sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_luma, f.luma.data(), f.luma.size(),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_chroma, f.chroma.data(), f.chroma.size(),
                        cudaMemcpyHostToDevice));

  Block device = host_block(f, filter, d_chroma);
  device.luma = d_luma;
  if (derived) {
    CUDA_CHECK(av2enc::cuda::cfl_predict_derived(device, d_alpha));
  } else {
    CUDA_CHECK(av2enc::cuda::cfl_predict(device, chosen_alpha));
  }
  CHECK(cudaDeviceSynchronize() == cudaSuccess);

  std::vector<std::uint8_t> gpu = f.chroma;
  CUDA_CHECK(cudaMemcpy(gpu.data(), d_chroma, f.chroma.size(),
                        cudaMemcpyDeviceToHost));
  CHECK(std::memcmp(gpu.data(), ours.data(), ours.size()) == 0);
  if (derived) {
    int gpu_alpha = 0;
    CUDA_CHECK(cudaMemcpy(&gpu_alpha, d_alpha, sizeof(int),
                          cudaMemcpyDeviceToHost));
    CHECK(gpu_alpha == chosen_alpha);
  }
  CUDA_CHECK(cudaFree(d_alpha));
  CUDA_CHECK(cudaFree(d_chroma));
  CUDA_CHECK(cudaFree(d_luma));
}

}  // namespace

int main() {
  const Fixture fixtures[] = {
      make_fixture(4, 4, 0, 0, 32, 32, 0, 64, 64, 1),
      make_fixture(8, 8, 8, 8, 32, 32, 0, 64, 64, 2),
      make_fixture(16, 8, 32, 32, 64, 64, 64, 128, 128, 3),
      make_fixture(8, 8, 24, 24, 30, 30, 0, 64, 64, 4),
      make_fixture(4, 8, 4, 4, 32, 32, 0, 64, 64, 5),
  };

  for (const Fixture& f : fixtures) {
    for (int filter = 0; filter < 4; ++filter) {
      run_case(f, filter, 64, false);
      run_case(f, filter, 0, true);
    }
  }

  // The encoder seam: choose an index on the host, then apply it on the GPU
  // without any kernel change.
  const Fixture f = fixtures[2];
  std::vector<std::uint8_t> u_chroma = f.chroma;
  std::vector<std::uint8_t> v_chroma = f.chroma;
  Block u = host_block(f, 0, u_chroma.data());
  Block v = host_block(f, 0, v_chroma.data());
  SourcePlane su{};
  su.plane = f.chroma.data();
  su.stride = f.chroma_stride;
  su.max_x = f.coded_w - 1;
  su.max_y = f.coded_h - 1;
  SourcePlane sv = su;
  Scratch scratch;
  const std::optional<FilterChoice> choice = av2enc::cfl::ds::search_filter(
      u, v, su, sv, SearchOptions{}, SearchMode::kExhaustiveSse, scratch);
  CHECK(choice.has_value());
  if (choice.has_value()) {
    run_case(f, choice->index, 0, true);
    std::printf("cuda seam: selected cfl_ds_filter_index=%d\n", choice->index);
  }

  std::printf("ds_filters_cuda_test: %d checks, %d failures\n", g_checks,
              g_failures);
  return g_failures == 0 ? 0 : 1;
}

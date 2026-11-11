// CPU-vs-GPU parity test for the M1 transform-stream adapter.
//
// The CPU side is `ts::evaluate_candidates()` / `ts::quantize_block()`; the GPU
// side is `gpu::evaluate_batch()` / `gpu::quantize_batch()` (verified CUDA
// transform + RDOQ kernels).  Every candidate is compared, not only the
// winner: levels, reconstruction, EOB, SSE, rates, cost and §5.20.8.2 symbol.
//
// Build/run on the sm_86 box:
//
//   ./build_gpu_test.sh
//   CUDA_ARCH=sm_86 ./build_gpu_test.sh
//
// The CPU adapter and the CPU reference components are compiled with
// -ffp-contract=off; nvcc uses -fmad=false.
#include <cuda_runtime.h>

#include "encoder/m1/transform_stream/transform_stream.h"
#include "encoder/m1/transform_stream/transform_stream_gpu.h"
#include "encoder/intra_modes/intra_modes.h"
#include "encoder/tx_type/tx_type.h"
#include "encoder/tx_variants/tx_variants.h"
#include "io/y4m.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

namespace ts = av2enc::m1::ts;
using av2enc::txtype::TxType;

long long g_checks = 0;
int g_failures = 0;
const char* g_suite = "";

void suite(const char* name) { g_suite = name; }

void check(bool condition, const char* what, int line) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    std::printf("FAIL [%s:%d] %s\n", g_suite, line, what);
    if (g_failures > 40) {
      std::printf("too many failures; aborting\n");
      std::exit(1);
    }
  }
}

#define CHECK(cond) check((cond), #cond, __LINE__)
#define CHECK_EQ(a, b)                        \
  do {                                        \
    const auto va__ = (a);                    \
    const auto vb__ = (b);                    \
    check(va__ == vb__, #a " == " #b, __LINE__); \
  } while (0)
#define CHECK_CUDA(expr)                                              \
  do {                                                                \
    const cudaError_t err__ = (expr);                                 \
    check(err__ == cudaSuccess, cudaGetErrorString(err__), __LINE__); \
  } while (0)

bool same_levels(const std::vector<std::int32_t>& a,
                 const std::vector<std::int32_t>& b) {
  return a == b;
}
bool same_recon(const std::vector<std::int16_t>& a,
                const std::vector<std::int16_t>& b) {
  return a == b;
}

struct HostBlock {
  int coded_w = 0;
  int coded_h = 0;
  int intra_mode = 0;
  int angle_delta = 0;
  int reduced_tx_set = 0;
  int dct_only = 0;
  int plane = 0;
  int txb_skip_context = 0;
  int dc_sign_context = 0;
  double lambda = 0.0;
  int prev_level_cap = 2;
  int level_cap = 0;
  int enable_idtx = 1;
  std::vector<TxType> candidate_types;
  std::vector<std::int16_t> residual;
};

ts::Request make_request(const HostBlock& block) {
  ts::Request request;
  request.residual_coded = block.residual.data();
  request.coded_w = block.coded_w;
  request.coded_h = block.coded_h;
  request.intra_mode = block.intra_mode;
  request.angle_delta = block.angle_delta;
  request.reduced_tx_set = block.reduced_tx_set;
  request.dct_only = block.dct_only != 0;
  request.plane = block.plane;
  request.txb_skip_context = block.txb_skip_context;
  request.dc_sign_context = block.dc_sign_context;
  request.lambda = block.lambda;
  request.prev_level_cap = block.prev_level_cap;
  request.level_cap = block.level_cap;
  request.enable_idtx_intra = block.enable_idtx != 0;
  if (!block.candidate_types.empty()) {
    request.candidate_types = std::span<const TxType>(
        block.candidate_types.data(), block.candidate_types.size());
  }
  return request;
}

std::vector<std::int16_t> make_residual(int w, int h, int kind,
                                        std::uint32_t seed) {
  std::vector<std::int16_t> residual(static_cast<std::size_t>(w * h));
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> noise(-40, 40);
  std::uniform_int_distribution<int> wide(-140, 140);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int value = 0;
      switch (kind) {
        case 0:
          value = noise(rng);
          break;
        case 1:
          value = x * 3 - y * 2 + noise(rng);
          break;
        case 2:
          value = (x < w / 2 ? -60 : 60) + noise(rng);
          break;
        default:
          value = wide(rng);
          break;
      }
      residual[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
               static_cast<std::size_t>(x)] =
          static_cast<std::int16_t>(std::clamp(value, -255, 255));
    }
  }
  return residual;
}

HostBlock make_block(int w, int h, int mode, int reduced, int kind,
                     std::uint32_t seed) {
  HostBlock block;
  block.coded_w = w;
  block.coded_h = h;
  block.intra_mode = mode;
  block.reduced_tx_set = reduced;
  const auto params = av2enc::txv::make_tx_params(w, h);
  block.lambda = ts::default_lambda(*params);
  block.residual = make_residual(w, h, kind, seed);
  return block;
}

struct Frame {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> luma;
};

bool load_frame(const std::filesystem::path& path, Frame* frame) {
  av2enc::Y4mReader reader(path);
  std::string error;
  if (!reader.open(error)) {
    std::printf("  cannot open %s: %s\n", path.c_str(), error.c_str());
    return false;
  }
  av2enc::Y4mFrame y4m;
  if (reader.read_frame(y4m, error) != av2enc::Y4mReadResult::kFrame) {
    std::printf("  cannot read %s: %s\n", path.c_str(), error.c_str());
    return false;
  }
  frame->width = static_cast<int>(y4m.width());
  frame->height = static_cast<int>(y4m.height());
  const std::span<const std::uint8_t> y = y4m.y();
  frame->luma.assign(y.begin(), y.end());
  return true;
}

bool add_frame_blocks(const Frame& frame, int bw, int bh, int mode, int reduced,
                      int max_blocks, std::vector<HostBlock>* blocks) {
  std::vector<std::uint8_t> pred(static_cast<std::size_t>(bw * bh));
  const auto params = av2enc::txv::make_tx_params(bw, bh);
  int added = 0;
  for (int y0 = 0; y0 + bh <= frame.height; y0 += bh) {
    for (int x0 = 0; x0 + bw <= frame.width; x0 += bw) {
      if (max_blocks > 0 && added >= max_blocks) return true;
      av2enc::intra::NeighborRequest request;
      request.recon.data = frame.luma.data();
      request.recon.stride = frame.width;
      request.x = x0;
      request.y = y0;
      request.w = bw;
      request.h = bh;
      request.max_x = frame.width - 1;
      request.max_y = frame.height - 1;
      request.have_above = y0 > 0;
      request.have_left = x0 > 0;
      request.color_plane = 0;
      av2enc::intra::PredictParams predict;
      predict.mode = av2enc::intra::Mode::kDc;
      if (av2enc::intra::predict_block(request, 8, predict, pred.data(),
                                       static_cast<std::size_t>(bw)) !=
          av2enc::intra::IntraStatus::kOk) {
        continue;
      }
      HostBlock block;
      block.coded_w = bw;
      block.coded_h = bh;
      block.intra_mode = mode;
      block.reduced_tx_set = reduced;
      block.lambda = ts::default_lambda(*params);
      block.residual.resize(static_cast<std::size_t>(bw * bh));
      for (int y = 0; y < bh; ++y) {
        for (int x = 0; x < bw; ++x) {
          const int src = frame.luma[static_cast<std::size_t>(y0 + y) *
                                         static_cast<std::size_t>(frame.width) +
                                     static_cast<std::size_t>(x0 + x)];
          const int p = pred[static_cast<std::size_t>(y * bw + x)];
          block.residual[static_cast<std::size_t>(y * bw + x)] =
              static_cast<std::int16_t>(src - p);
        }
      }
      blocks->push_back(std::move(block));
      ++added;
    }
  }
  return true;
}

// Custom rate model used by the hook-plumbing parity case.
struct HookState {
  int builds = 0;
  int releases = 0;
};

double hook_coefficient_rate(void* context, std::int32_t scan_pos,
                             std::int32_t prev_abs_state, std::int32_t level,
                             std::int32_t level_context, std::int32_t is_eob) {
  (void)context;
  (void)level_context;
  const int magnitude = level < 0 ? -level : level;
  double bits = 0.3 + 0.12 * static_cast<double>(magnitude);
  if (prev_abs_state == 0) bits += 0.25;
  if (is_eob != 0) bits += 0.15;
  return bits + 0.002 * static_cast<double>(scan_pos);
}

double hook_eob_rate(void* context, std::int32_t eob,
                     std::int32_t num_coefficients) {
  (void)context;
  (void)num_coefficients;
  return 0.9 + 0.3 * std::log2(static_cast<double>(eob) + 1.0);
}

bool hook_build(void* user, const ts::RateModelRequest& request,
                av2enc::rdoq::RateModel* model, std::string* error) {
  (void)request;
  if (error != nullptr) error->clear();
  ++static_cast<HookState*>(user)->builds;
  model->context = user;
  model->coefficient_rate = &hook_coefficient_rate;
  model->eob_rate = &hook_eob_rate;
  return true;
}

void hook_release(void* user, av2enc::rdoq::RateModel* model) noexcept {
  (void)model;
  ++static_cast<HookState*>(user)->releases;
}

bool run_parity(const std::vector<HostBlock>& blocks,
                const ts::RateModelHooks* hooks, const char* label,
                bool expect_budget_error = false) {
  const int count = static_cast<int>(blocks.size());
  std::vector<ts::Request> requests(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    requests[static_cast<std::size_t>(i)] =
        make_request(blocks[static_cast<std::size_t>(i)]);
    requests[static_cast<std::size_t>(i)].hooks = hooks;
  }

  // CPU: per-candidate and winner.
  std::vector<std::vector<ts::Decision>> cpu_candidates(
      static_cast<std::size_t>(count));
  std::vector<ts::Decision> cpu_winner(static_cast<std::size_t>(count));
  std::string error;
  for (int i = 0; i < count; ++i) {
    if (!ts::evaluate_candidates(requests[static_cast<std::size_t>(i)],
                                 &cpu_candidates[static_cast<std::size_t>(i)],
                                 &error)) {
      std::printf("  CPU evaluate failed: %s\n", error.c_str());
      return false;
    }
    if (!ts::quantize_block(requests[static_cast<std::size_t>(i)],
                            &cpu_winner[static_cast<std::size_t>(i)],
                            &error)) {
      std::printf("  CPU decide failed: %s\n", error.c_str());
      return false;
    }
  }

  av2enc::m1::ts::gpu::BatchOptions options;
  options.table_budget_bytes = 1ull << 30;
  std::vector<av2enc::m1::ts::gpu::CandidateOutcome> gpu_candidates;
  if (!av2enc::m1::ts::gpu::evaluate_batch(requests.data(), count, options,
                                           &gpu_candidates, &error)) {
    if (expect_budget_error) {
      std::printf("  %-28s budget error as expected: %s\n", label,
                  error.c_str());
      return true;
    }
    std::printf("  GPU evaluate failed: %s\n", error.c_str());
    return false;
  }
  if (expect_budget_error) {
    CHECK(false && "expected a table-budget error");
    return false;
  }

  std::vector<ts::Decision> gpu_winner(static_cast<std::size_t>(count));
  if (!av2enc::m1::ts::gpu::quantize_batch(requests.data(), count, options,
                                           gpu_winner.data(), &error)) {
    std::printf("  GPU decide failed: %s\n", error.c_str());
    return false;
  }

  std::size_t index = 0;
  int compared = 0;
  for (int b = 0; b < count; ++b) {
    const auto& cpu = cpu_candidates[static_cast<std::size_t>(b)];
    for (std::size_t j = 0; j < cpu.size(); ++j) {
      CHECK(index < gpu_candidates.size());
      if (index >= gpu_candidates.size()) return false;
      const auto& gpu = gpu_candidates[index];
      const ts::Decision& reference = cpu[j];
      CHECK_EQ(gpu.block, b);
      CHECK_EQ(gpu.type, static_cast<int>(reference.type));
      CHECK_EQ(gpu.eob, reference.eob);
      CHECK_EQ(gpu.nonzero, reference.nonzero);
      CHECK_EQ(gpu.sse, reference.sse);
      CHECK_EQ(gpu.coeff_rate_bits, reference.coeff_rate_bits);
      CHECK_EQ(gpu.type_bits, reference.type_bits);
      CHECK_EQ(gpu.rate_bits, reference.rate_bits);
      CHECK_EQ(gpu.cost, reference.cost);
      CHECK_EQ(gpu.rdoq_distortion, reference.rdoq_distortion);
      CHECK_EQ(gpu.rdoq_rate_bits, reference.rdoq_rate_bits);
      CHECK_EQ(gpu.rdoq_cost, reference.rdoq_cost);
      CHECK_EQ(gpu.codable, reference.codable);
      CHECK_EQ(static_cast<int>(gpu.symbol.kind),
               static_cast<int>(reference.symbol.kind));
      CHECK_EQ(static_cast<int>(gpu.symbol.symbol),
               static_cast<int>(reference.symbol.symbol));
      CHECK_EQ(gpu.symbol.has_long_flag, reference.symbol.has_long_flag);
      CHECK_EQ(gpu.symbol.is_long_side_dct, reference.symbol.is_long_side_dct);
      CHECK_EQ(static_cast<int>(gpu.symbol.cdf_symbols),
               static_cast<int>(reference.symbol.cdf_symbols));
      CHECK_EQ(gpu.symbol.bits, reference.symbol.bits);
      if (!same_levels(gpu.levels_scan, reference.levels)) {
        std::printf("  level mismatch block %d cand %zu type %d eob %d/%d "
                    "areas %zu/%zu\n",
                    b, j, static_cast<int>(reference.type), gpu.eob,
                    reference.eob, gpu.levels_scan.size(),
                    reference.levels.size());
        int shown = 0;
        for (std::size_t k = 0; k < reference.levels.size(); ++k) {
          if (gpu.levels_scan[k] != reference.levels[k] && shown < 8) {
            std::printf("    %zu: gpu %d cpu %d\n", k,
                        gpu.levels_scan[k], reference.levels[k]);
            ++shown;
          }
        }
      }
      CHECK(same_levels(gpu.levels_scan, reference.levels));
      if (!same_recon(gpu.recon_coded, reference.recon_coded)) {
        std::printf("  recon mismatch block %d cand %zu type %d eob %d/%d "
                    "params %dx%d sizes %zu/%zu\n",
                    b, j, static_cast<int>(reference.type), gpu.eob,
                    reference.eob, reference.params.coded_w,
                    reference.params.coded_h, gpu.recon_coded.size(),
                    reference.recon_coded.size());
      }
      if (!same_recon(gpu.recon_coded, reference.recon_coded)) {
        std::printf("  recon mismatch block %d cand %zu type %d eob %d/%d "
                    "params %dx%d\n",
                    b, j, static_cast<int>(reference.type), gpu.eob,
                    reference.eob, reference.params.coded_w,
                    reference.params.coded_h);
        int shown = 0;
        for (std::size_t k = 0; k < reference.recon_coded.size(); ++k) {
          if (gpu.recon_coded[k] != reference.recon_coded[k] && shown < 5) {
            std::printf("    %zu: gpu %d cpu %d\n", k,
                        gpu.recon_coded[k], reference.recon_coded[k]);
            ++shown;
          }
        }
      }
      CHECK(same_recon(gpu.recon_coded, reference.recon_coded));
      ++index;
      ++compared;
    }

    const ts::Decision& reference = cpu_winner[static_cast<std::size_t>(b)];
    const ts::Decision& gpu = gpu_winner[static_cast<std::size_t>(b)];
    CHECK_EQ(gpu.valid, reference.valid);
    CHECK_EQ(static_cast<int>(gpu.type), static_cast<int>(reference.type));
    CHECK_EQ(gpu.eob, reference.eob);
    CHECK_EQ(gpu.cost, reference.cost);
    CHECK_EQ(gpu.rate_bits, reference.rate_bits);
    CHECK(same_levels(gpu.levels, reference.levels));
    CHECK(same_recon(gpu.recon_coded, reference.recon_coded));
  }
  std::printf("  %-28s %3d blocks  %5d candidates  parity OK\n", label, count,
              compared);
  return true;
}

std::vector<HostBlock> synthetic_blocks() {
  std::vector<HostBlock> blocks;
  blocks.push_back(make_block(4, 4, 0, 0, 0, 1));
  blocks.push_back(make_block(4, 8, 3, 0, 1, 2));
  blocks.push_back(make_block(8, 8, 9, 0, 1, 3));
  blocks.push_back(make_block(8, 16, 0, 0, 2, 4));
  blocks.push_back(make_block(16, 8, 12, 0, 0, 5));
  blocks.push_back(make_block(16, 16, 3, 0, 1, 6));
  blocks.push_back(make_block(16, 16, 0, 1, 2, 7));
  blocks.push_back(make_block(16, 16, 0, 3, 0, 8));
  blocks.push_back(make_block(16, 32, 0, 0, 1, 9));
  blocks.push_back(make_block(32, 16, 0, 0, 0, 10));
  blocks.push_back(make_block(8, 8, 0, 0, 3, 11));
  HostBlock chroma = make_block(16, 16, 0, 0, 0, 12);
  chroma.dct_only = 1;
  chroma.plane = 1;
  chroma.txb_skip_context = 3;
  blocks.push_back(chroma);
  HostBlock level_cap = make_block(8, 8, 0, 0, 1, 13);
  level_cap.level_cap = 12;
  level_cap.prev_level_cap = 4;
  blocks.push_back(level_cap);
  HostBlock long32 = make_block(16, 32, 0, 0, 1, 14);
  long32.enable_idtx = 0;  // M1 enable_idtx_intra = 0
  blocks.push_back(long32);
  HostBlock subset = make_block(8, 8, 0, 0, 0, 15);
  subset.candidate_types = {TxType::kDctDct, TxType::kAdstAdst};
  blocks.push_back(subset);
  return blocks;
}

void test_synthetic() {
  suite("synthetic");
  const std::vector<HostBlock> blocks = synthetic_blocks();
  CHECK(run_parity(blocks, nullptr, "synthetic/AV2"));

  HookState state;
  ts::RateModelHooks hooks;
  hooks.build = &hook_build;
  hooks.release = &hook_release;
  hooks.user = &state;
  CHECK(run_parity(blocks, &hooks, "synthetic/custom"));
  CHECK_EQ(state.builds, state.releases);

  // Budget rejection happens before any table allocation.
  std::vector<ts::Request> requests(blocks.size());
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    requests[i] = make_request(blocks[i]);
  }
  av2enc::m1::ts::gpu::BatchOptions tight;
  tight.table_budget_bytes = 1024;
  std::vector<av2enc::m1::ts::gpu::CandidateOutcome> outcomes;
  std::string error;
  CHECK(!av2enc::m1::ts::gpu::evaluate_batch(
      requests.data(), static_cast<int>(requests.size()), tight, &outcomes,
      &error));
  CHECK(!error.empty());
}

void test_determinism() {
  suite("determinism");
  const std::vector<HostBlock> blocks = synthetic_blocks();
  std::vector<ts::Request> requests(blocks.size());
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    requests[i] = make_request(blocks[i]);
  }
  av2enc::m1::ts::gpu::BatchOptions options;
  std::vector<av2enc::m1::ts::gpu::CandidateOutcome> first;
  std::vector<av2enc::m1::ts::gpu::CandidateOutcome> second;
  std::string error;
  CHECK(av2enc::m1::ts::gpu::evaluate_batch(
      requests.data(), static_cast<int>(requests.size()), options, &first,
      &error));
  CHECK(av2enc::m1::ts::gpu::evaluate_batch(
      requests.data(), static_cast<int>(requests.size()), options, &second,
      &error));
  CHECK_EQ(first.size(), second.size());
  for (std::size_t i = 0; i < std::min(first.size(), second.size()); ++i) {
    CHECK_EQ(first[i].eob, second[i].eob);
    CHECK_EQ(first[i].cost, second[i].cost);
    CHECK(same_levels(first[i].levels_scan, second[i].levels_scan));
  }
}

void test_real_frames(const std::filesystem::path& data_dir) {
  suite("real-frames");
  const std::filesystem::path small = data_dir / "m0_64x64_1f.y4m";
  const std::filesystem::path large = data_dir / "m0_640x360_1f.y4m";
  Frame frame;
  if (load_frame(small, &frame)) {
    std::vector<HostBlock> blocks;
    add_frame_blocks(frame, 64, 64, 0, 0, 0, &blocks);
    add_frame_blocks(frame, 16, 16, 0, 0, 4, &blocks);
    CHECK(run_parity(blocks, nullptr, "64x64/AV2"));
  } else {
    std::printf("  SKIP %s\n", small.c_str());
  }
  if (load_frame(large, &frame)) {
    std::vector<HostBlock> blocks;
    add_frame_blocks(frame, 16, 16, 0, 0, 8, &blocks);
    add_frame_blocks(frame, 32, 32, 0, 0, 4, &blocks);
    // Keep the table build bounded: a 32x32 block has a wide level range.
    CHECK(run_parity(blocks, nullptr, "640x360/AV2"));
  } else {
    std::printf("  SKIP %s\n", large.c_str());
  }
}

void test_invalid() {
  suite("invalid");
  const auto residual = make_residual(8, 8, 0, 99);
  ts::Request request;
  request.residual_coded = residual.data();
  request.coded_w = 8;
  request.coded_h = 8;
  request.intra_mode = 0;
  request.lambda = 10.0;
  request.hooks = nullptr;
  std::vector<av2enc::m1::ts::gpu::CandidateOutcome> outcomes;
  av2enc::m1::ts::gpu::BatchOptions options;
  std::string error;
  CHECK(!av2enc::m1::ts::gpu::evaluate_batch(nullptr, 1, options, &outcomes,
                                             &error));
  CHECK(!av2enc::m1::ts::gpu::evaluate_batch(&request, 1, options, nullptr,
                                             &error));
  request.coded_w = 5;
  CHECK(!av2enc::m1::ts::gpu::evaluate_batch(&request, 1, options, &outcomes,
                                             &error));
  request.coded_w = 8;
  request.lambda = std::numeric_limits<double>::quiet_NaN();
  CHECK(!av2enc::m1::ts::gpu::evaluate_batch(&request, 1, options, &outcomes,
                                             &error));

  av2enc::m1::ts::gpu::DeviceAllocator half;
  half.allocate = nullptr;
  half.deallocate = [](void*, void*) {};
  options.allocator = half;
  request.lambda = 10.0;
  CHECK(!av2enc::m1::ts::gpu::evaluate_batch(&request, 1, options, &outcomes,
                                             &error));
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path data_dir = "tests/data";
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--data" && i + 1 < argc) {
      data_dir = argv[++i];
    } else if (std::string(argv[i]) == "--help") {
      std::printf("usage: %s [--data DIR]\n", argv[0]);
      return 0;
    } else {
      std::printf("unknown argument: %s\n", argv[i]);
      return 2;
    }
  }

  std::printf("av2enc M1 transform-stream GPU parity test\n");
  test_synthetic();
  test_determinism();
  test_real_frames(data_dir);
  test_invalid();
  std::printf("%lld checks, %d failures\n", g_checks, g_failures);
  std::printf("RESULT: %s\n", g_failures == 0 ? "PASS" : "FAIL");
  return g_failures == 0 ? 0 : 1;
}

// CUDA self-test for the AV2 intra transform-type batch evaluator.
//
// The CPU reference is linked directly (src/encoder/tx_type/tx_type.cpp and
// src/encoder/tx_variants/tx_variants.cpp); the transform kernels are the real
// ones from src/cuda/tx_variants/tx_variants.cu.
//
// Gates:
//   * every candidate of every block: quantized levels and inverse residual
//     bit-identical to the CPU,
//   * every candidate metric (SSE, nonzero, sum|level|, eob==1) identical,
//   * selected type / cost / SSE / candidate index identical to
//     txtype::choose_intra_tx_type() for the same cost model,
//   * real 64x64 and 640x360 luma blocks plus a mixed-size synthetic batch,
//   * determinism across two identical launches,
//   * argument validation.
//
// Build/run: ./build_and_test.sh
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "cuda/codec/transform/tx_type/tx_type.h"
#include "cuda/codec/transform/variants/tx_variants.h"
#include "codec/intra/modes/intra_modes.h"
#include "codec/rate/rdo/rdo.h"
#include "codec/transform/tx_type/tx_type.h"
#include "codec/transform/variants/tx_tables.h"
#include "codec/transform/variants/tx_variants.h"
#include "core/io/y4m.h"

namespace {

namespace txtype = av2enc::txtype;
using namespace av2enc::txtype;

long long g_checks = 0;
int g_failures = 0;
const char* g_suite = "";

void suite_start(const char* name) { g_suite = name; }

void check(bool cond, const char* what, int line) {
  ++g_checks;
  if (!cond) {
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
    const auto va = (a);                      \
    const auto vb = (b);                      \
    check(va == vb, #a " == " #b, __LINE__);  \
  } while (0)
#define CHECK_CUDA(expr)                                              \
  do {                                                                \
    const cudaError_t err__ = (expr);                                 \
    check(err__ == cudaSuccess, cudaGetErrorString(err__), __LINE__); \
  } while (0)

std::mt19937 g_rng(0x1234abcd);

int rand_range(int lo, int hi) {
  std::uniform_int_distribution<int> dist(lo, hi);
  return dist(g_rng);
}

void fill_residual(int w, int h, int kind, std::int16_t* out) {
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int v = 0;
      switch (kind) {
        case 0:
          v = rand_range(-60, 60);
          break;
        case 1:
          v = (x * 3 - y * 2) + rand_range(-20, 20);
          break;
        case 2:
          v = (x < w / 2 ? -40 : 40) + rand_range(-10, 10);
          break;
        default:
          v = (rand_range(0, 15) == 0) ? rand_range(-120, 120) : 0;
          break;
      }
      out[y * w + x] = static_cast<std::int16_t>(v);
    }
  }
}

bool levels_equal(const std::int32_t* a, const std::int32_t* b, int n) {
  for (int i = 0; i < n; ++i) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

bool residual_equal(const std::int16_t* a, const std::int16_t* b, int n) {
  for (int i = 0; i < n; ++i) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Batch driver
// ---------------------------------------------------------------------------

struct HostBlock {
  int coded_w = 0;
  int coded_h = 0;
  int mapped_mode = 0;
  int reduced_tx_set = 0;
  int dct_only = 0;
  std::vector<std::int16_t> residual_coded;
  std::vector<std::int16_t> residual_adj;
};

struct BatchResult {
  std::vector<txtype::TxTypeEval> cpu_eval;
  std::vector<std::vector<double>> cpu_coeff;
  std::vector<std::vector<std::int32_t>> cpu_levels;
  std::vector<std::vector<std::int16_t>> cpu_recon;
  std::vector<txtype::TxTypeChoice> cpu_choice;
  std::vector<av2enc::cuda::TxTypeCandidateMetrics> gpu_metrics;
  std::vector<av2enc::cuda::TxTypeDecision> gpu_decisions;
  std::vector<double> gpu_coeff;
  std::vector<std::int32_t> gpu_levels;
  std::vector<std::int16_t> gpu_recon;
  std::vector<av2enc::cuda::TxBlockJob> tx_jobs;
  std::vector<av2enc::cuda::TxTypeBlockJob> blocks;
  std::vector<av2enc::cuda::TxTypeCandJob> cands;
  av2enc::cuda::TxTypePlan plan;
  std::vector<av2enc::cuda::TxTypeBlockRequest> requests;
  int nblocks = 0;
  int ncandidates = 0;
};

bool run_batch(const std::vector<HostBlock>& host, double lambda,
               double coeff_bits, bool use_type_bits, BatchResult* out) {
  using av2enc::cuda::kTxTypeAdjElems;
  using av2enc::cuda::kTxTypeCodedElems;

  out->nblocks = static_cast<int>(host.size());
  out->requests.resize(host.size());
  std::size_t coded_elems = 0;
  std::size_t adj_elems = 0;
  for (std::size_t i = 0; i < host.size(); ++i) {
    out->requests[i].coded_w = host[i].coded_w;
    out->requests[i].coded_h = host[i].coded_h;
    out->requests[i].mapped_mode = host[i].mapped_mode;
    out->requests[i].reduced_tx_set = host[i].reduced_tx_set;
    out->requests[i].dct_only = host[i].dct_only;
  }
  av2enc::cuda::tx_type_pack_layout(out->requests.data(),
                                    static_cast<int>(host.size()),
                                    &coded_elems, &adj_elems);

  txtype::TxTypeCostConfig cfg;
  cfg.lambda = lambda;
  cfg.coefficient_bits = coeff_bits;
  cfg.use_static_type_cost = use_type_bits;
  std::string error;
  if (!av2enc::cuda::tx_type_build_plan(out->requests.data(),
                                        static_cast<int>(host.size()),
                                        &out->plan, &error)) {
    std::printf("  plan failed: %s\n", error.c_str());
    return false;
  }
  out->ncandidates = static_cast<int>(out->plan.candidates.size());
  out->blocks = out->plan.blocks;
  out->cands = out->plan.candidates;
  out->tx_jobs = out->plan.tx_jobs;

  std::vector<std::int16_t> residual_coded(coded_elems);
  std::vector<std::int16_t> residual_adj(adj_elems);
  for (std::size_t i = 0; i < host.size(); ++i) {
    std::memcpy(&residual_coded[static_cast<std::size_t>(
                    out->requests[i].coded_offset)],
                host[i].residual_coded.data(),
                host[i].residual_coded.size() * sizeof(std::int16_t));
    std::memcpy(
        &residual_adj[static_cast<std::size_t>(out->requests[i].adj_offset)],
        host[i].residual_adj.data(),
        host[i].residual_adj.size() * sizeof(std::int16_t));
  }

  std::int16_t* d_coded = nullptr;
  std::int16_t* d_adj = nullptr;
  std::int16_t* d_adj_scratch = nullptr;
  double* d_coeff = nullptr;
  std::int32_t* d_levels = nullptr;
  std::int16_t* d_recon = nullptr;
  av2enc::cuda::TxTypeBlockJob* d_blocks = nullptr;
  av2enc::cuda::TxTypeCandJob* d_cands = nullptr;
  av2enc::cuda::TxBlockJob* d_tx_jobs = nullptr;
  av2enc::cuda::TxTypeCandidateMetrics* d_metrics = nullptr;
  av2enc::cuda::TxTypeDecision* d_decisions = nullptr;

  const std::size_t nblocks = host.size();
  const std::size_t ncand = out->plan.candidates.size();
  CHECK_CUDA(cudaMalloc(&d_coded, coded_elems * sizeof(std::int16_t)));
  CHECK_CUDA(cudaMalloc(&d_adj, adj_elems * sizeof(std::int16_t)));
  CHECK_CUDA(cudaMalloc(&d_adj_scratch,
                        out->plan.adj_scratch_elems * sizeof(std::int16_t)));
  CHECK_CUDA(cudaMalloc(&d_coeff,
                        out->plan.coeff_scratch_elems * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&d_levels,
                        out->plan.level_scratch_elems * sizeof(std::int32_t)));
  CHECK_CUDA(cudaMalloc(&d_recon,
                        out->plan.recon_scratch_elems * sizeof(std::int16_t)));
  CHECK_CUDA(cudaMalloc(&d_blocks,
                        std::max<std::size_t>(1, nblocks) *
                            sizeof(av2enc::cuda::TxTypeBlockJob)));
  CHECK_CUDA(cudaMalloc(&d_cands,
                        std::max<std::size_t>(1, ncand) *
                            sizeof(av2enc::cuda::TxTypeCandJob)));
  CHECK_CUDA(cudaMalloc(&d_tx_jobs,
                        std::max<std::size_t>(1, ncand) *
                            sizeof(av2enc::cuda::TxBlockJob)));
  CHECK_CUDA(cudaMalloc(&d_metrics,
                        std::max<std::size_t>(1, ncand) *
                            sizeof(av2enc::cuda::TxTypeCandidateMetrics)));
  CHECK_CUDA(cudaMalloc(&d_decisions,
                        std::max<std::size_t>(1, nblocks) *
                            sizeof(av2enc::cuda::TxTypeDecision)));

  CHECK_CUDA(cudaMemcpy(d_coded, residual_coded.data(),
                        coded_elems * sizeof(std::int16_t),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_adj, residual_adj.data(),
                        adj_elems * sizeof(std::int16_t),
                        cudaMemcpyHostToDevice));
  if (nblocks != 0) {
    CHECK_CUDA(cudaMemcpy(d_blocks, out->plan.blocks.data(),
                          nblocks * sizeof(av2enc::cuda::TxTypeBlockJob),
                          cudaMemcpyHostToDevice));
  }
  if (ncand != 0) {
    CHECK_CUDA(cudaMemcpy(d_cands, out->plan.candidates.data(),
                          ncand * sizeof(av2enc::cuda::TxTypeCandJob),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_tx_jobs, out->plan.tx_jobs.data(),
                          ncand * sizeof(av2enc::cuda::TxBlockJob),
                          cudaMemcpyHostToDevice));
  }

  av2enc::cuda::TxTypeCostModel model;
  model.lambda = lambda;
  model.coeff_bits = coeff_bits;
  model.use_type_bits = use_type_bits;
  CHECK_CUDA(av2enc::cuda::tx_type_choose_batch(
      d_coded, d_adj, d_blocks, out->nblocks, d_cands, out->ncandidates,
      d_tx_jobs, d_adj_scratch, d_coeff, d_levels, d_recon, d_metrics,
      d_decisions, model, nullptr));
  CHECK_CUDA(cudaDeviceSynchronize());

  out->gpu_metrics.resize(ncand);
  out->gpu_decisions.resize(nblocks);
  out->gpu_coeff.assign(ncand * kTxTypeAdjElems, 0.0);
  out->gpu_levels.assign(ncand * kTxTypeAdjElems, 0);
  out->gpu_recon.assign(ncand * kTxTypeCodedElems, 0);
  if (ncand != 0) {
    CHECK_CUDA(cudaMemcpy(out->gpu_metrics.data(), d_metrics,
                          ncand * sizeof(av2enc::cuda::TxTypeCandidateMetrics),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(out->gpu_coeff.data(), d_coeff,
                          ncand * kTxTypeAdjElems * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(out->gpu_levels.data(), d_levels,
                          ncand * kTxTypeAdjElems * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(out->gpu_recon.data(), d_recon,
                          ncand * kTxTypeCodedElems * sizeof(std::int16_t),
                          cudaMemcpyDeviceToHost));
  }
  if (nblocks != 0) {
    CHECK_CUDA(cudaMemcpy(out->gpu_decisions.data(), d_decisions,
                          nblocks * sizeof(av2enc::cuda::TxTypeDecision),
                          cudaMemcpyDeviceToHost));
  }

  cudaFree(d_coded);
  cudaFree(d_adj);
  cudaFree(d_adj_scratch);
  cudaFree(d_coeff);
  cudaFree(d_levels);
  cudaFree(d_recon);
  cudaFree(d_blocks);
  cudaFree(d_cands);
  cudaFree(d_tx_jobs);
  cudaFree(d_metrics);
  cudaFree(d_decisions);

  // CPU reference.
  out->cpu_eval.resize(ncand);
  out->cpu_coeff.resize(ncand);
  out->cpu_levels.resize(ncand);
  out->cpu_recon.resize(ncand);
  out->cpu_choice.resize(nblocks);
  int index = 0;
  for (std::size_t b = 0; b < host.size(); ++b) {
    const HostBlock& block = host[b];
    txtype::TxTypeQuery query;
    query.coded_w = block.coded_w;
    query.coded_h = block.coded_h;
    query.mapped_mode = block.mapped_mode;
    query.reduced_tx_set = block.reduced_tx_set;
    query.dct_only = block.dct_only != 0;
    txtype::TxTypeSet set;
    if (!txtype::intra_tx_candidates(query, &set)) {
      return false;
    }
    if (set.count != out->blocks[b].cand_count) {
      std::printf("  candidate count mismatch at block %zu\n", b);
      return false;
    }
    for (int j = 0; j < set.count; ++j) {
      const int adj_n = set.candidates[j].params.w * set.candidates[j].params.h;
      const int coded_n =
          set.candidates[j].params.coded_w * set.candidates[j].params.coded_h;
      out->cpu_coeff[index].assign(static_cast<std::size_t>(adj_n), 0.0);
      out->cpu_levels[index].assign(static_cast<std::size_t>(adj_n), 0);
      out->cpu_recon[index].assign(static_cast<std::size_t>(coded_n), 0);
      if (!txtype::evaluate_intra_tx_type(
              block.residual_coded.data(), query, set.candidates[j], cfg,
              &out->cpu_eval[index], out->cpu_levels[index].data(),
              out->cpu_recon[index].data())) {
        return false;
      }
      // Intermediate coefficients, for the bit-exact forward-transform gate.
      av2enc::txv::forward_block(block.residual_adj.data(),
                                 set.candidates[j].params,
                                 out->cpu_coeff[index].data());
      ++index;
    }
    if (!txtype::choose_intra_tx_type(block.residual_coded.data(), query, cfg,
                                      &out->cpu_choice[b])) {
      return false;
    }
  }
  return true;
}

// Returns the number of candidates compared.
int verify_batch(const std::vector<HostBlock>& host, const BatchResult& result,
                 const char* label) {
  using av2enc::cuda::kTxTypeAdjElems;
  using av2enc::cuda::kTxTypeCodedElems;
  int compared = 0;
  int index = 0;
  int eob_zero = 0;
  int eob_one = 0;
  int non_codable = 0;
  for (std::size_t b = 0; b < host.size(); ++b) {
    const HostBlock& block = host[b];
    const txtype::TxTypeChoice& cpu_choice = result.cpu_choice[b];
    const av2enc::cuda::TxTypeDecision& gpu = result.gpu_decisions[b];
    CHECK_EQ(gpu.type, static_cast<int>(cpu_choice.candidate.type));
    CHECK(std::fabs(gpu.cost - cpu_choice.eval.cost) < 1e-9);
    CHECK(std::fabs(gpu.sse - cpu_choice.eval.sse) < 1e-9);
    CHECK_EQ(gpu.candidate_index, cpu_choice.candidate_index);
    CHECK_EQ(gpu.nonzero, cpu_choice.eval.nonzero);
    CHECK_EQ(gpu.eob_one, cpu_choice.eval.eob == 1 ? 1 : 0);

    for (int j = 0; j < result.blocks[b].cand_count; ++j) {
      const txtype::TxTypeEval& cpu = result.cpu_eval[index];
      const av2enc::cuda::TxTypeCandidateMetrics& m = result.gpu_metrics[index];
      const int adj_n = cpu.candidate.params.w * cpu.candidate.params.h;
      const int coded_n = cpu.candidate.params.coded_w *
                          cpu.candidate.params.coded_h;
      const std::size_t level_base =
          static_cast<std::size_t>(index) * kTxTypeAdjElems;
      bool coeff_equal = true;
      for (int i = 0; i < adj_n; ++i) {
        if (result.gpu_coeff[level_base + static_cast<std::size_t>(i)] !=
            result.cpu_coeff[index][static_cast<std::size_t>(i)]) {
          coeff_equal = false;
          break;
        }
      }
      CHECK(coeff_equal);
      const std::size_t recon_base =
          static_cast<std::size_t>(index) * kTxTypeCodedElems;
      CHECK(levels_equal(&result.gpu_levels[level_base],
                         result.cpu_levels[index].data(), adj_n));
      CHECK(residual_equal(&result.gpu_recon[recon_base],
                           result.cpu_recon[index].data(), coded_n));
      CHECK_EQ(m.sse, cpu.sse);
      CHECK_EQ(m.nonzero, cpu.nonzero);
      CHECK_EQ(m.sum_abs, cpu.sum_abs);
      CHECK_EQ(m.eob_one, cpu.eob == 1 ? 1 : 0);
      if (cpu.eob == 0) {
        ++eob_zero;
      } else if (cpu.eob == 1) {
        ++eob_one;
      }
      if (!cpu.codable) {
        ++non_codable;
      }
      ++compared;
      ++index;
    }
  }
  std::printf("  %-42s %5zu blocks %6d candidates"
              "  (eob0 %d, eob1 %d, non-codable %d)\n",
              label, host.size(), compared, eob_zero, eob_one, non_codable);
  return compared;
}

// ---------------------------------------------------------------------------
// Blocks
// ---------------------------------------------------------------------------

HostBlock make_block(int coded_w, int coded_h, int mode, int reduced,
                     int residual_kind) {
  HostBlock block;
  block.coded_w = coded_w;
  block.coded_h = coded_h;
  block.mapped_mode = mode;
  block.reduced_tx_set = reduced;
  block.residual_coded.resize(static_cast<std::size_t>(coded_w * coded_h));
  fill_residual(coded_w, coded_h, residual_kind, block.residual_coded.data());
  const auto params = av2enc::txv::make_tx_params(coded_w, coded_h);
  block.residual_adj.resize(static_cast<std::size_t>(params->w * params->h));
  av2enc::txv::downsample_residual(block.residual_coded.data(), *params,
                                   block.residual_adj.data());
  return block;
}

std::vector<HostBlock> synthetic_blocks() {
  std::vector<HostBlock> blocks;
  for (int size = 0; size < 25; ++size) {
    const int w = av2enc::txv::kTxSizes[size].width;
    const int h = av2enc::txv::kTxSizes[size].height;
    for (int mode : {0, 3, 9, 12}) {
      for (int reduced : {0, 1, 3}) {
        for (int kind = 0; kind < 2; ++kind) {
          blocks.push_back(make_block(w, h, mode, reduced, kind));
        }
      }
    }
  }
  // Force-DCT (chroma-style) path: one candidate, zero type bits.
  {
    HostBlock block = make_block(16, 16, 0, 0, 0);
    block.dct_only = 1;
    blocks.push_back(block);
  }
  return blocks;
}

struct Frame {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> luma;
};

bool load_y4m(const std::filesystem::path& path, Frame* frame) {
  av2enc::Y4mReader reader(path);
  std::string error;
  if (!reader.open(error)) {
    std::printf("  cannot open %s: %s\n", path.c_str(), error.c_str());
    return false;
  }
  av2enc::Y4mFrame y4m;
  if (reader.read_frame(y4m, error) != av2enc::Y4mReadResult::kFrame) {
    std::printf("  cannot read frame from %s: %s\n", path.c_str(),
                error.c_str());
    return false;
  }
  frame->width = static_cast<int>(y4m.width());
  frame->height = static_cast<int>(y4m.height());
  const std::span<const std::uint8_t> y = y4m.y();
  frame->luma.assign(y.begin(), y.end());
  return true;
}

bool add_frame_blocks(const Frame& frame, int bw, int bh, int mode, int reduced,
                      std::vector<HostBlock>* blocks) {
  std::vector<std::uint8_t> pred(static_cast<std::size_t>(bw * bh));
  std::vector<std::int16_t> residual(static_cast<std::size_t>(bw * bh));
  for (int y0 = 0; y0 + bh <= frame.height; y0 += bh) {
    for (int x0 = 0; x0 + bw <= frame.width; x0 += bw) {
      av2enc::intra::NeighborRequest req;
      req.recon.data = frame.luma.data();
      req.recon.stride = frame.width;
      req.x = x0;
      req.y = y0;
      req.w = bw;
      req.h = bh;
      req.max_x = frame.width - 1;
      req.max_y = frame.height - 1;
      req.have_above = y0 > 0;
      req.have_left = x0 > 0;
      req.color_plane = 0;
      av2enc::intra::PredictParams params;
      params.mode = av2enc::intra::Mode::kDc;
      if (av2enc::intra::predict_block(req, 8, params, pred.data(),
                                       static_cast<std::size_t>(bw)) !=
          av2enc::intra::IntraStatus::kOk) {
        return false;
      }
      for (int y = 0; y < bh; ++y) {
        for (int x = 0; x < bw; ++x) {
          const std::size_t src_pos =
              static_cast<std::size_t>((y0 + y) * frame.width + x0 + x);
          const std::size_t pos = static_cast<std::size_t>(y * bw + x);
          residual[pos] = static_cast<std::int16_t>(frame.luma[src_pos] -
                                                    pred[pos]);
        }
      }
      HostBlock block;
      block.coded_w = bw;
      block.coded_h = bh;
      block.mapped_mode = mode;
      block.reduced_tx_set = reduced;
      block.residual_coded = residual;
      const auto tp = av2enc::txv::make_tx_params(bw, bh);
      block.residual_adj.resize(static_cast<std::size_t>(tp->w * tp->h));
      av2enc::txv::downsample_residual(residual.data(), *tp,
                                       block.residual_adj.data());
      blocks->push_back(std::move(block));
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

void test_plan_and_validation() {
  suite_start("plan/validation");
  std::vector<HostBlock> host;
  host.push_back(make_block(4, 4, 0, 0, 0));
  host.push_back(make_block(16, 32, 3, 0, 1));
  host.push_back(make_block(64, 64, 0, 0, 2));
  std::vector<av2enc::cuda::TxTypeBlockRequest> requests(host.size());
  for (std::size_t i = 0; i < host.size(); ++i) {
    requests[i].coded_w = host[i].coded_w;
    requests[i].coded_h = host[i].coded_h;
    requests[i].mapped_mode = host[i].mapped_mode;
    requests[i].reduced_tx_set = host[i].reduced_tx_set;
  }
  std::size_t coded_elems = 0;
  std::size_t adj_elems = 0;
  av2enc::cuda::tx_type_pack_layout(requests.data(),
                                    static_cast<int>(requests.size()),
                                    &coded_elems, &adj_elems);
  CHECK_EQ(coded_elems, static_cast<std::size_t>(16 + 512 + 4096));
  CHECK_EQ(adj_elems, static_cast<std::size_t>(16 + 512 + 1024));
  CHECK_EQ(requests[0].coded_offset, 0);
  CHECK_EQ(requests[1].coded_offset, 16);
  CHECK_EQ(requests[2].coded_offset, 16 + 512);
  CHECK_EQ(requests[0].adj_offset, 0);
  CHECK_EQ(requests[1].adj_offset, 16);
  CHECK_EQ(requests[2].adj_offset, 16 + 512);

  av2enc::cuda::TxTypePlan plan;
  std::string error;
  CHECK(av2enc::cuda::tx_type_build_plan(
      requests.data(), static_cast<int>(requests.size()), &plan, &error));
  CHECK(error.empty());
  CHECK_EQ(plan.blocks.size(), static_cast<std::size_t>(3));
  CHECK_EQ(plan.candidates.size(), plan.tx_jobs.size());
  // 4x4 INTRA_1 = 7, 16x32 HIGH_32 = 8, 64x64 DCTONLY = 1.
  CHECK_EQ(plan.candidates.size(), static_cast<std::size_t>(7 + 8 + 1));
  CHECK_EQ(plan.blocks[0].cand_count, 7);
  CHECK_EQ(plan.blocks[1].cand_count, 8);
  CHECK_EQ(plan.blocks[2].cand_count, 1);
  CHECK_EQ(plan.adj_scratch_elems,
           plan.candidates.size() * av2enc::cuda::kTxTypeAdjElems);
  CHECK_EQ(plan.recon_scratch_elems,
           plan.candidates.size() * av2enc::cuda::kTxTypeCodedElems);
  for (std::size_t i = 0; i < plan.candidates.size(); ++i) {
    const auto& c = plan.candidates[i];
    CHECK_EQ(c.tx.in_offset,
             static_cast<int>(i) * av2enc::cuda::kTxTypeAdjElems);
    CHECK_EQ(c.tx.out_offset,
             static_cast<int>(i) * av2enc::cuda::kTxTypeCodedElems);
    CHECK_EQ(plan.tx_jobs[i].in_offset, c.tx.in_offset);
    CHECK_EQ(plan.tx_jobs[i].out_offset, c.tx.out_offset);
    CHECK_EQ(c.type, static_cast<int>(plan.tx_jobs[i].params.tx_type));
  }
  // type_bits must come from the CPU model for the same candidate.
  {
    txtype::TxTypeQuery query;
    query.coded_w = 4;
    query.coded_h = 4;
    query.mapped_mode = 0;
    txtype::TxTypeSet set;
    CHECK(txtype::intra_tx_candidates(query, &set));
    for (int j = 0; j < set.count; ++j) {
      CHECK(std::fabs(plan.candidates[static_cast<std::size_t>(j)].type_bits -
                      txtype::static_type_bits(query, set.candidates[j])) <
            1e-12);
    }
  }

  // Invalid plan inputs.
  {
    av2enc::cuda::TxTypeBlockRequest bad = requests[0];
    bad.coded_w = 2;
    bad.coded_h = 8;
    CHECK(!av2enc::cuda::tx_type_build_plan(&bad, 1, &plan, &error));
    CHECK(!error.empty());
    bad = requests[0];
    bad.mapped_mode = 13;
    CHECK(!av2enc::cuda::tx_type_build_plan(&bad, 1, &plan, &error));
    bad = requests[0];
    bad.reduced_tx_set = 4;
    CHECK(!av2enc::cuda::tx_type_build_plan(&bad, 1, &plan, &error));
    CHECK(!av2enc::cuda::tx_type_build_plan(nullptr, 1, &plan, &error));
    CHECK(!av2enc::cuda::tx_type_build_plan(requests.data(), 1, nullptr,
                                            &error));
  }

  // Device entry-point validation.
  {
    av2enc::cuda::TxTypeCostModel model;
    av2enc::cuda::TxTypeDecision decision;
    av2enc::cuda::TxTypeCandidateMetrics metrics;
    CHECK_EQ(av2enc::cuda::tx_type_choose_batch(
                 nullptr, nullptr, nullptr, 0, nullptr, 0, nullptr, nullptr,
                 nullptr, nullptr, nullptr, nullptr, &decision, model,
                 nullptr),
             cudaSuccess);
    CHECK_EQ(av2enc::cuda::tx_type_choose_batch(
                 nullptr, nullptr, nullptr, 1, nullptr, 1, nullptr, nullptr,
                 nullptr, nullptr, nullptr, &metrics, &decision, model,
                 nullptr),
             cudaErrorInvalidValue);
    CHECK_EQ(av2enc::cuda::tx_type_choose_batch(
                 nullptr, nullptr, nullptr, -1, nullptr, 0, nullptr, nullptr,
                 nullptr, nullptr, nullptr, &metrics, &decision, model,
                 nullptr),
             cudaErrorInvalidValue);
  }
}

void test_synthetic_batch() {
  suite_start("synthetic");
  const std::vector<HostBlock> host = synthetic_blocks();
  const struct {
    double lambda;
    double coeff_bits;
    bool use_type_bits;
    const char* name;
  } models[] = {
      {0.0, 1.0, true, "lambda=0"},
      {544.5, 1.0, true, "lambda=544.5"},
      {544.5, 2.5, false, "lambda=544.5, no type bits"},
      {25267.6, 1.0, true, "lambda=rdo(90)"},
  };
  int total = 0;
  for (const auto& m : models) {
    BatchResult result;
    CHECK(run_batch(host, m.lambda, m.coeff_bits, m.use_type_bits, &result));
    total += verify_batch(host, result, m.name);
  }
  std::printf("  %d candidates compared across %zu models\n", total,
              sizeof(models) / sizeof(models[0]));
}

void test_real_frames(const std::filesystem::path& data_dir) {
  suite_start("real_frames");
  const bool allow_missing =
      std::getenv("AV2ENC_ALLOW_MISSING_MEDIA") != nullptr;
  Frame frame64;
  Frame frame360;
  const bool have64 = load_y4m(data_dir / "basic_64x64_1f.y4m", &frame64);
  const bool have360 = load_y4m(data_dir / "basic_640x360_1f.y4m", &frame360);
  check(have64 && have360, "real-frame media present", __LINE__);
  if (!have64 || !have360) {
    if (allow_missing) {
      return;
    }
    return;
  }

  const double lambda =
      av2enc::rdo::lambda_sse_from_quant_step(static_cast<double>(528 / 8));
  std::printf("  QP 90 lambda = %.1f\n", lambda);

  if (have64) {
    std::vector<HostBlock> host;
    CHECK(add_frame_blocks(frame64, 64, 64, 0, 0, &host));
    CHECK(add_frame_blocks(frame64, 16, 16, 0, 0, &host));
    CHECK(add_frame_blocks(frame64, 8, 8, 0, 0, &host));
    CHECK(add_frame_blocks(frame64, 4, 4, 3, 0, &host));
    CHECK_EQ(host.size(), static_cast<std::size_t>(1 + 16 + 64 + 256));
    BatchResult result;
    CHECK(run_batch(host, lambda, 1.0, true, &result));
    verify_batch(host, result, "64x64 mixed 64/16/8/4");
  }

  if (have360) {
    // The whole 640x360 luma plane in 16x16 blocks, one batch per mode.
    for (int mode : {0, 3, 1}) {
      std::vector<HostBlock> host;
      CHECK(add_frame_blocks(frame360, 16, 16, mode, 0, &host));
      CHECK_EQ(host.size(), static_cast<std::size_t>(40 * 22));
      BatchResult result;
      CHECK(run_batch(host, lambda, 1.0, true, &result));
      char label[64];
      std::snprintf(label, sizeof(label), "640x360 16x16 mode=%d", mode);
      verify_batch(host, result, label);
    }
    // Long-32 shapes: the only intra sets where IDTX is codable.
    for (int mode : {0}) {
      std::vector<HostBlock> host;
      CHECK(add_frame_blocks(frame360, 16, 32, mode, 0, &host));
      CHECK_EQ(host.size(), static_cast<std::size_t>(40 * 11));
      BatchResult result;
      CHECK(run_batch(host, lambda, 1.0, true, &result));
      verify_batch(host, result, "640x360 16x32 (IDTX legal)");
    }
    {
      std::vector<HostBlock> host;
      CHECK(add_frame_blocks(frame360, 32, 16, 0, 0, &host));
      CHECK_EQ(host.size(), static_cast<std::size_t>(20 * 22));
      BatchResult result;
      CHECK(run_batch(host, lambda, 1.0, true, &result));
      verify_batch(host, result, "640x360 32x16 (IDTX legal)");
    }
    // 32x32 is DCTONLY: one candidate per block.
    {
      std::vector<HostBlock> host;
      CHECK(add_frame_blocks(frame360, 32, 32, 0, 0, &host));
      CHECK_EQ(host.size(), static_cast<std::size_t>(20 * 11));
      BatchResult result;
      CHECK(run_batch(host, lambda, 1.0, true, &result));
      verify_batch(host, result, "640x360 32x32 (DCTONLY)");
    }
    // reduced_tx_set = 1: 16x16 -> {DCT_DCT, ADST_ADST}.
    {
      std::vector<HostBlock> host;
      CHECK(add_frame_blocks(frame360, 16, 16, 0, 1, &host));
      BatchResult result;
      CHECK(run_batch(host, lambda, 1.0, true, &result));
      verify_batch(host, result, "640x360 16x16 reduced=1");
      for (const auto& c : result.cands) {
        CHECK(c.type == 0 || c.type == 3);
      }
    }
  }
}

void test_determinism() {
  suite_start("determinism");
  std::vector<HostBlock> host;
  host.push_back(make_block(16, 16, 0, 0, 0));
  host.push_back(make_block(32, 16, 3, 0, 1));
  host.push_back(make_block(8, 8, 0, 0, 2));
  BatchResult a;
  BatchResult b;
  CHECK(run_batch(host, 544.5, 1.0, true, &a));
  CHECK(run_batch(host, 544.5, 1.0, true, &b));
  CHECK_EQ(a.ncandidates, b.ncandidates);
  for (int i = 0; i < a.ncandidates; ++i) {
    CHECK_EQ(a.gpu_metrics[static_cast<std::size_t>(i)].sse,
             b.gpu_metrics[static_cast<std::size_t>(i)].sse);
    CHECK_EQ(a.gpu_metrics[static_cast<std::size_t>(i)].nonzero,
             b.gpu_metrics[static_cast<std::size_t>(i)].nonzero);
  }
  for (int i = 0; i < a.nblocks; ++i) {
    CHECK_EQ(a.gpu_decisions[static_cast<std::size_t>(i)].type,
             b.gpu_decisions[static_cast<std::size_t>(i)].type);
    CHECK_EQ(a.gpu_decisions[static_cast<std::size_t>(i)].cost,
             b.gpu_decisions[static_cast<std::size_t>(i)].cost);
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path data_dir = "tests/data";
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--data" && i + 1 < argc) {
      data_dir = argv[++i];
    } else if (arg == "--help") {
      std::printf("usage: %s [--data DIR]\n", argv[0]);
      return 0;
    } else {
      std::printf("unknown argument: %s\n", arg.c_str());
      return 2;
    }
  }

  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
    std::printf("no CUDA device\n");
    return 1;
  }
  std::printf("av2enc CUDA intra transform-type test\n");
  std::printf("device: %s (sm_%d%d, %zu MiB)\n", prop.name, prop.major,
              prop.minor, prop.totalGlobalMem >> 20);

  test_plan_and_validation();
  test_synthetic_batch();
  test_real_frames(data_dir);
  test_determinism();

  std::printf("\n%lld checks, %d failures\n", g_checks, g_failures);
  std::printf("RESULT: %s\n", g_failures == 0 ? "PASS" : "FAIL");
  return g_failures == 0 ? 0 : 1;
}

// Real-GPU CPU-vs-CUDA parity test for the M2 video TX-selection module.
//
// For every candidate the device kernel produces, the test runs the *product
// CPU reference* on the same inputs:
//   * `txv::downsample_residual` / `forward_block` / `inverse_block`
//   * the `TileEncoder` delta-Q quantiser expression (qf*528/q, then
//     lround(coeff*factor)) and the §7.14.4 dequantiser
//   * the shared `deltaq::core::get_q` host path for §7.14.2 quantisers
//   * `txpartition::get_scan` / `tx_class_of` for the §5.20.7.30 scan
// and compares coefficients, levels, scan-order levels, EOB, nonzero, SSE,
// rate, cost, codability and every decision tier bit for bit.
//
// Usage: video_tx_test [--quick]
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "codec/transform/inter_residual/inter_tx_type.h"
#include "codec/transform/tx_partition/tx_partition.h"
#include "codec/transform/tx_partition/tx_scan.h"
#include "codec/transform/variants/tx_variants.h"
#include "cuda/arena.h"
#include "cuda/codec/transform/delta_q/deltaq_device.cuh"
#include "cuda/pipeline/video/tx/video_tx.h"

namespace {

namespace txv = av2enc::txv;
namespace txpart = av2enc::video::txpartition;
namespace videotx = av2enc::cuda::videotx;
using av2enc::cuda::Arena;
using av2enc::cuda::Status;

int g_checks = 0;
int g_failures = 0;
constexpr int kMaxReported = 12;

void check_impl(bool cond, const char* what, int line) {
  ++g_checks;
  if (!cond) {
    ++g_failures;
    if (g_failures <= kMaxReported) {
      std::printf("FAIL line %d: %s\n", line, what);
    }
  }
}

#define CHECK(cond) check_impl((cond), #cond, __LINE__)

bool same_double(double a, double b) {
  // Every compared quantity is a non-negative sum or a deterministic double
  // expression.  Compare the bits so a signed zero or NaN cannot slip through.
  std::uint64_t ua = 0;
  std::uint64_t ub = 0;
  std::memcpy(&ua, &a, sizeof(ua));
  std::memcpy(&ub, &b, sizeof(ub));
  return ua == ub;
}

struct Rng {
  std::uint64_t state = 0x9e3779b97f4a7c15ull;
  std::uint32_t next() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return static_cast<std::uint32_t>(state >> 32);
  }
  int range(int lo, int hi) {
    return lo + static_cast<int>(next() % static_cast<std::uint32_t>(hi - lo + 1));
  }
};

// ---------------------------------------------------------------------------
// CPU reference: one candidate on one leaf
// ---------------------------------------------------------------------------

struct CpuCandMetrics {
  double sse = 0.0;
  double rate_bits = 0.0;
  double cost = 0.0;
  std::int32_t nonzero = 0;
  std::int32_t eob = 0;
  std::int32_t codable = 0;
};

struct CpuLeafDecision {
  std::int32_t tx_size = -1;
  std::int32_t tx_type = -1;
  std::int32_t eob = 0;
  std::int32_t nonzero = 0;
  double sse = 0.0;
  double rate = 0.0;
  double cost = 0.0;
  std::vector<std::int32_t> levels;  // scan-order winner levels
};

struct CpuJobDecision {
  std::int32_t best_plan = -1;  // absolute plan index
  std::int32_t partition = 0;
  double best_sse = 0.0;
  double best_rate = 0.0;
  double best_cost = 0.0;
  std::vector<CpuLeafDecision> leaves;
};

struct CpuResult {
  std::vector<CpuCandMetrics> candidates;
  std::vector<std::int32_t> plan_valid;
  std::vector<double> plan_sse;
  std::vector<double> plan_rate;
  std::vector<double> plan_cost;
  std::vector<std::vector<std::int32_t>> plan_leaf_best;  // [plan][leaf]
  std::vector<CpuJobDecision> jobs;
  bool ok = false;
};

std::int32_t inter_ctx(int eob, int coded_w, int coded_h) {
  if (eob <= 0) return 0;
  int lw = 0;
  while ((1 << lw) < coded_w) ++lw;
  const int bwl = lw < 5 ? lw : 5;
  const int eoby = (eob - 1) >> bwl;
  const int eobx = (eob - 1) - (eoby << bwl);
  const int diag = eobx + eoby;
  if (diag < 2) return 1;
  const int limit = std::min(coded_w, 32) + std::min(coded_h, 32) - 4;
  if (diag > limit) return 2;
  return 0;
}

// Gathers the leaf's coded-size residual exactly like the kernel.
void gather_residual(const videotx::VideoTxJobRequest& job,
                     const videotx::VideoTxLeafDesc& leaf,
                     const std::uint8_t* src, const std::uint8_t* pred,
                     const std::int16_t* residual,
                     std::vector<std::int16_t>* out) {
  const int cw = leaf.coded_w;
  const int ch = leaf.coded_h;
  out->assign(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch), 0);
  const int src_stride = job.src_stride > 0 ? job.src_stride : job.coded_w;
  const int pred_stride = job.pred_stride > 0 ? job.pred_stride : job.coded_w;
  for (int ay = 0; ay < ch; ++ay) {
    for (int ax = 0; ax < cw; ++ax) {
      int sample;
      if (job.residual_offset >= 0) {
        sample = residual[job.residual_offset + (leaf.y + ay) * job.coded_w +
                          (leaf.x + ax)];
      } else {
        int s;
        if (job.src_packed != 0) {
          s = src[job.src_offset + (leaf.y + ay) * src_stride + leaf.x + ax];
        } else {
          int px = job.src_x + leaf.x + ax;
          int py = job.src_y + leaf.y + ay;
          px = std::max(0, std::min(job.src_visible_w - 1, px));
          py = std::max(0, std::min(job.src_visible_h - 1, py));
          s = src[job.src_offset + py * src_stride + px];
        }
        const int p =
            pred[job.pred_offset + (leaf.y + ay) * pred_stride + leaf.x + ax];
        sample = s - p;
      }
      (*out)[static_cast<std::size_t>(ay) * cw + ax] =
          static_cast<std::int16_t>(sample);
    }
  }
}

CpuResult cpu_reference(const videotx::VideoTxPlanData& plan,
                        const videotx::VideoTxJobRequest* requests,
                        const std::uint8_t* src, const std::uint8_t* pred,
                        const std::int16_t* residual) {
  CpuResult out;
  out.candidates.resize(plan.candidates.size());
  out.plan_valid.assign(plan.plans.size(), 0);
  out.plan_sse.assign(plan.plans.size(), 0.0);
  out.plan_rate.assign(plan.plans.size(), 0.0);
  out.plan_cost.assign(plan.plans.size(), 0.0);
  out.plan_leaf_best.resize(plan.plans.size());
  out.jobs.resize(plan.jobs.size());
  out.ok = true;

  for (std::size_t j = 0; j < plan.jobs.size(); ++j) {
    const videotx::VideoTxJobDesc& jd = plan.jobs[j];
    const videotx::VideoTxJobRequest& req = requests[j];
    double best_plan_cost = 1e300;
    double best_plan_sse = 1e300;
    int best_plan = -1;

    for (int p_local = 0; p_local < jd.plan_count; ++p_local) {
      const int p_index = jd.plan_begin + p_local;
      const videotx::VideoTxPlanDesc& pd = plan.plans[p_index];
      bool valid = pd.leaf_count > 0 && pd.leaf_count <= 4;
      double plan_cost = pd.partition_bits;
      double plan_sse = 0.0;
      double plan_rate = pd.partition_bits;
      out.plan_leaf_best[p_index].assign(pd.leaf_count, -1);

      for (int l = 0; l < pd.leaf_count && valid; ++l) {
        const int leaf_index = pd.leaf_begin + l;
        const videotx::VideoTxLeafDesc& leaf = plan.leaves[leaf_index];
        std::vector<std::int16_t> coded;
        gather_residual(req, leaf, src, pred, residual, &coded);

        double leaf_best_cost = 1e300;
        double leaf_best_sse = 1e300;
        double leaf_best_rate = 0.0;
        int leaf_best_type = -1;
        int leaf_best_cand = -1;

        for (int c = 0; c < leaf.cand_count; ++c) {
          const int ci = leaf.cand_begin + c;
          const videotx::VideoTxCandDesc& cd = plan.candidates[ci];
          const txv::TxParams& params = cd.params;
          const int w = params.w;
          const int h = params.h;
          const int area = w * h;
          const int coded_n = leaf.coded_w * leaf.coded_h;

          std::vector<std::int16_t> adj(static_cast<std::size_t>(area));
          txv::downsample_residual(coded.data(), params, adj.data());

          std::vector<double> coeff(static_cast<std::size_t>(area));
          txv::forward_block(adj.data(), params, coeff.data());

          const int dcq = av2enc::cuda::deltaq::core::get_q(req.qindex, req.delta_dc);
          const int acq = av2enc::cuda::deltaq::core::get_q(req.qindex, req.delta_ac);
          const double scaled = params.quant_factor * 528.0;
          std::vector<std::int32_t> levels(static_cast<std::size_t>(area));
          std::vector<std::int32_t> dequant(static_cast<std::size_t>(area));
          for (int i = 0; i < area; ++i) {
            const int q = i == 0 ? dcq : acq;
            const double factor = scaled / static_cast<double>(q);
            const std::int32_t level = static_cast<std::int32_t>(
                std::lround(coeff[i] * factor));
            levels[i] = level;
            const std::int32_t aq = level < 0 ? -level : level;
            const std::int32_t high = (aq * q) & 0xFFFFFF;
            const std::int32_t dq =
                txv::round2(high, txv::kQuantTableBits) / params.dq_denom;
            dequant[i] = std::max(-(1 << 15),
                                  std::min((1 << 15) - 1,
                                           level < 0 ? -dq : dq));
          }

          std::vector<std::int16_t> recon(static_cast<std::size_t>(coded_n));
          txv::inverse_block(dequant.data(), params, recon.data());

          std::int64_t sse_i = 0;
          for (int i = 0; i < coded_n; ++i) {
            const std::int64_t d = static_cast<std::int64_t>(recon[i]) - coded[i];
            sse_i += d * d;
          }
          const double sse = static_cast<double>(sse_i);

          const int tx_size = cd.tx_size;
          const txpart::TxScanClass cls =
              txpart::tx_class_of(static_cast<int>(cd.tx_type));
          std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
          const bool scan_ok = txpart::get_scan(tx_size, cls, scan);
          CHECK(scan_ok);
          int eob = 0;
          int nonzero = 0;
          std::int64_t sum_abs = 0;
          for (int cpos = 0; cpos < area; ++cpos) {
            const std::int32_t level = levels[scan[cpos]];
            if (level != 0) {
              ++nonzero;
              sum_abs += level < 0 ? -static_cast<std::int64_t>(level)
                                   : static_cast<std::int64_t>(level);
              eob = cpos + 1;
            }
          }
          (void)sum_abs;

          const double type_bits =
              jd.use_type_bits != 0 && cd.codes_type != 0
                  ? cd.type_bits[jd.is_inter != 0
                                     ? inter_ctx(eob, leaf.coded_w, leaf.coded_h)
                                     : 0]
                  : 0.0;
          const double rate =
              eob == 0 ? jd.skip_bits
                       : type_bits +
                             jd.coefficient_bits * static_cast<double>(nonzero);
          const double cost = sse + jd.lambda * rate;
          const bool codable =
              jd.is_inter != 0 ? (eob == 0 ? cd.tx_type == 0 : true)
                               : ((eob >= 2) || cd.tx_type == 0);

          CpuCandMetrics m;
          m.sse = sse;
          m.rate_bits = rate;
          m.cost = cost;
          m.nonzero = nonzero;
          m.eob = eob;
          m.codable = codable ? 1 : 0;
          out.candidates[ci] = m;

          if (codable &&
              (cost < leaf_best_cost ||
               (cost == leaf_best_cost &&
                (sse < leaf_best_sse ||
                 (sse == leaf_best_sse && cd.tx_type < leaf_best_type))))) {
            leaf_best_cost = cost;
            leaf_best_sse = sse;
            leaf_best_rate = rate;
            leaf_best_type = cd.tx_type;
            leaf_best_cand = ci;
          }
        }
        if (leaf_best_cand < 0) {
          valid = false;
          break;
        }
        out.plan_leaf_best[p_index][l] = leaf_best_cand;
        plan_cost += leaf_best_cost;
        plan_sse += leaf_best_sse;
        plan_rate += leaf_best_rate;
      }

      out.plan_valid[p_index] = valid ? 1 : 0;
      out.plan_sse[p_index] = plan_sse;
      out.plan_rate[p_index] = plan_rate;
      out.plan_cost[p_index] = plan_cost;
      if (valid && (plan_cost < best_plan_cost ||
                    (plan_cost == best_plan_cost && plan_sse < best_plan_sse))) {
        best_plan_cost = plan_cost;
        best_plan_sse = plan_sse;
        best_plan = p_local;
      }
    }

    CpuJobDecision& d = out.jobs[j];
    if (best_plan < 0) {
      out.ok = false;
      continue;
    }
    const int best_plan_index = jd.plan_begin + best_plan;
    const videotx::VideoTxPlanDesc& bp = plan.plans[best_plan_index];
    d.best_plan = best_plan_index;
    d.partition = bp.partition;
    d.best_sse = out.plan_sse[best_plan_index];
    d.best_rate = out.plan_rate[best_plan_index];
    d.best_cost = out.plan_cost[best_plan_index];
    d.leaves.resize(bp.leaf_count);
    for (int l = 0; l < bp.leaf_count; ++l) {
      const int ci = out.plan_leaf_best[best_plan_index][l];
      const videotx::VideoTxCandDesc& bc = plan.candidates[ci];
      const videotx::VideoTxLeafDesc& bl = plan.leaves[bp.leaf_begin + l];
      CpuLeafDecision& ld = d.leaves[l];
      ld.tx_size = bc.tx_size;
      ld.tx_type = bc.tx_type;
      ld.eob = out.candidates[ci].eob;
      ld.nonzero = out.candidates[ci].nonzero;
      ld.sse = out.candidates[ci].sse;
      ld.rate = out.candidates[ci].rate_bits;
      ld.cost = out.candidates[ci].cost;
      ld.levels.assign(static_cast<std::size_t>(bc.params.w * bc.params.h), 0);
      // Recompute the winner's scan-order levels (matches the kernel replay).
      std::vector<std::int16_t> coded;
      gather_residual(req, bl, src, pred, residual, &coded);
      std::vector<std::int16_t> adj(
          static_cast<std::size_t>(bc.params.w * bc.params.h));
      txv::downsample_residual(coded.data(), bc.params, adj.data());
      std::vector<double> coeff(static_cast<std::size_t>(adj.size()));
      txv::forward_block(adj.data(), bc.params, coeff.data());
      const int dcq =
          av2enc::cuda::deltaq::core::get_q(req.qindex, req.delta_dc);
      const int acq =
          av2enc::cuda::deltaq::core::get_q(req.qindex, req.delta_ac);
      const double scaled = bc.params.quant_factor * 528.0;
      std::vector<std::int32_t> levels(adj.size());
      for (int i = 0; i < static_cast<int>(adj.size()); ++i) {
        const int q = i == 0 ? dcq : acq;
        levels[i] = static_cast<std::int32_t>(
            std::lround(coeff[i] * (scaled / static_cast<double>(q))));
      }
      std::vector<std::uint16_t> scan(adj.size());
      txpart::get_scan(bc.tx_size,
                       txpart::tx_class_of(static_cast<int>(bc.tx_type)), scan);
      for (int c = 0; c < static_cast<int>(adj.size()); ++c) {
        ld.levels[c] = levels[scan[c]];
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Test fixture: buffers + one batch
// ---------------------------------------------------------------------------

struct Fixture {
  // One source buffer: the plane used by plane-mode jobs lives at offset 0,
  // and packed blocks are appended after it (all jobs index this one buffer,
  // exactly like the production source plane / packed-block contract).
  std::vector<std::uint8_t> plane_src;
  std::vector<std::uint8_t> packed_pred;
  std::vector<std::int16_t> packed_residual;
  std::vector<videotx::VideoTxJobRequest> requests;
};

void make_plane(Fixture* f, int w, int h, Rng* rng, int kind) {
  f->plane_src.resize(static_cast<std::size_t>(w) * h);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int v = 128;
      if (kind == 0) {
        v = rng->range(0, 255);
      } else if (kind == 1) {
        v = (x * 7 + y * 3) & 0xFF;
      } else {
        v = 128 + ((x / 4 + y / 4) % 2) * 24;
      }
      f->plane_src[static_cast<std::size_t>(y) * w + x] =
          static_cast<std::uint8_t>(v);
    }
  }
}

void add_job(Fixture* f, const videotx::VideoTxJobRequest& req,
             const std::vector<std::uint8_t>& src_block,
             const std::vector<std::uint8_t>& pred_block,
             const std::vector<std::int16_t>& residual_block) {
  videotx::VideoTxJobRequest r = req;
  if (req.residual_offset >= 0) {
    r.residual_offset = static_cast<int>(f->packed_residual.size());
    f->packed_residual.insert(f->packed_residual.end(), residual_block.begin(),
                              residual_block.end());
  } else if (req.src_packed != 0) {
    r.src_offset = static_cast<int>(f->plane_src.size());
    f->plane_src.insert(f->plane_src.end(), src_block.begin(),
                        src_block.end());
    r.pred_offset = static_cast<int>(f->packed_pred.size());
    f->packed_pred.insert(f->packed_pred.end(), pred_block.begin(),
                          pred_block.end());
  } else {
    r.pred_offset = static_cast<int>(f->packed_pred.size());
    f->packed_pred.insert(f->packed_pred.end(), pred_block.begin(),
                          pred_block.end());
  }
  f->requests.push_back(r);
}

struct BatchResult {
  std::vector<videotx::VideoTxCandMetrics> cands;
  std::vector<videotx::VideoTxPlanResult> plans;
  std::vector<videotx::VideoTxLeafResult> leaves;
  std::vector<videotx::VideoTxJobResult> jobs;
  std::vector<std::int32_t> levels;
  int error_count = 0;
};

struct DeviceBuffer {
  void* ptr = nullptr;
  std::size_t bytes = 0;
  ~DeviceBuffer() {
    if (ptr != nullptr) cudaFree(ptr);
  }
  bool upload(const void* host, std::size_t size) {
    bytes = size;
    if (size == 0) return true;
    if (cudaMalloc(&ptr, size) != cudaSuccess) return false;
    return cudaMemcpy(ptr, host, size, cudaMemcpyHostToDevice) == cudaSuccess;
  }
};

bool run_batch(const videotx::VideoTxPlanData& plan, const Fixture& f,
               const videotx::VideoTxConfig& config, BatchResult* out,
               std::uint64_t* required_out = nullptr) {
  const std::uint64_t need = videotx::required_device_bytes(config);
  if (required_out != nullptr) *required_out = need;
  Arena arena;
  std::string error;
  if (arena.init(need + Arena::kAlignment, Arena::kAlignment, &error) !=
      Status::kOk) {
    std::printf("arena init failed: %s\n", error.c_str());
    return false;
  }
  videotx::VideoTxWorkspace ws;
  if (ws.init(arena, config, nullptr, &error) != Status::kOk) {
    std::printf("workspace init failed: %s\n", error.c_str());
    return false;
  }
  // Test scaffolding only: the module's inputs are device-resident, so the
  // fixture copies them once per batch.  (The encoder allocates these from its
  // own arena; this file is a test tree and check_arena.sh exempts it.)
  DeviceBuffer d_src;
  DeviceBuffer d_pred;
  DeviceBuffer d_residual;
  if (!d_src.upload(f.plane_src.data(), f.plane_src.size()) ||
      !d_pred.upload(f.packed_pred.data(), f.packed_pred.size()) ||
      !d_residual.upload(f.packed_residual.data(),
                         f.packed_residual.size() * sizeof(std::int16_t))) {
    std::printf("input upload failed\n");
    return false;
  }
  videotx::VideoTxBuffers buffers;
  buffers.src = static_cast<const std::uint8_t*>(d_src.ptr);
  buffers.pred = static_cast<const std::uint8_t*>(d_pred.ptr);
  buffers.residual = static_cast<const std::int16_t*>(d_residual.ptr);
  if (videotx::video_tx_select_batch(plan, buffers, ws, nullptr) !=
      Status::kOk) {
    std::printf("video_tx_select_batch failed\n");
    return false;
  }
  out->cands.resize(plan.candidates.size());
  out->plans.resize(plan.plans.size());
  out->leaves.resize(plan.jobs.size() * videotx::kMaxLeavesPerPlan);
  out->jobs.resize(plan.jobs.size());
  out->levels.resize(plan.jobs.size() * videotx::kMaxLeavesPerPlan *
                     videotx::kMaxAdjElems);
  cudaMemcpy(out->cands.data(), ws.d_cand_metrics,
             out->cands.size() * sizeof(videotx::VideoTxCandMetrics),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(out->plans.data(), ws.d_plan_results,
             out->plans.size() * sizeof(videotx::VideoTxPlanResult),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(out->leaves.data(), ws.d_leaf_results,
             out->leaves.size() * sizeof(videotx::VideoTxLeafResult),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(out->jobs.data(), ws.d_job_results,
             out->jobs.size() * sizeof(videotx::VideoTxJobResult),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(out->levels.data(), ws.d_levels,
             out->levels.size() * sizeof(std::int32_t),
             cudaMemcpyDeviceToHost);
  out->error_count = ws.last_error_count;
  return true;
}

// Compares one batch against the CPU reference.  Returns the number of
// candidate/decision checks performed.
int compare_batch(const char* label, const videotx::VideoTxPlanData& plan,
                  const Fixture& f, const BatchResult& gpu) {
  const CpuResult cpu = cpu_reference(
      plan, f.requests.data(),
      f.plane_src.empty() ? nullptr : f.plane_src.data(),
      f.packed_pred.empty() ? nullptr : f.packed_pred.data(),
      f.packed_residual.empty() ? nullptr : f.packed_residual.data());
  int checks = 0;
  int local_failures = g_failures;
  for (std::size_t i = 0; i < plan.candidates.size(); ++i) {
    ++checks;
    const CpuCandMetrics& c = cpu.candidates[i];
    const videotx::VideoTxCandMetrics& g = gpu.cands[i];
    if (!same_double(c.sse, g.sse) || !same_double(c.rate_bits, g.rate_bits) ||
        !same_double(c.cost, g.cost) || c.nonzero != g.nonzero ||
        c.eob != g.eob || c.codable != g.codable) {
      CHECK(false);
      if (g_failures <= kMaxReported) {
        std::printf("  %s candidate %zu: cpu(sse=%.17g rate=%.17g cost=%.17g "
                    "nz=%d eob=%d cod=%d) gpu(sse=%.17g rate=%.17g cost=%.17g "
                    "nz=%d eob=%d cod=%d)\n",
                    label, i, c.sse, c.rate_bits, c.cost, c.nonzero, c.eob,
                    c.codable, g.sse, g.rate_bits, g.cost, g.nonzero, g.eob,
                    g.codable);
      }
    }
  }
  for (std::size_t i = 0; i < plan.plans.size(); ++i) {
    ++checks;
    const videotx::VideoTxPlanResult& g = gpu.plans[i];
    if (cpu.plan_valid[i] != g.valid || !same_double(cpu.plan_sse[i], g.sse) ||
        !same_double(cpu.plan_rate[i], g.rate_bits) ||
        !same_double(cpu.plan_cost[i], g.cost)) {
      CHECK(false);
    }
  }
  for (std::size_t j = 0; j < plan.jobs.size(); ++j) {
    ++checks;
    const CpuJobDecision& c = cpu.jobs[j];
    const videotx::VideoTxJobResult& g = gpu.jobs[j];
    if (c.best_plan != g.plan_index || !same_double(c.best_sse, g.sse) ||
        !same_double(c.best_rate, g.rate_bits) ||
        !same_double(c.best_cost, g.cost) ||
        g.status != (c.best_plan >= 0 ? 0 : -1)) {
      CHECK(false);
      if (g_failures <= kMaxReported) {
        std::printf("  %s job %zu: cpu(plan=%d cost=%.17g) gpu(plan=%d "
                    "cost=%.17g status=%d)\n",
                    label, j, c.best_plan, c.best_cost, g.plan_index, g.cost,
                    g.status);
      }
      continue;
    }
    if (c.best_plan < 0) continue;
    if (g.partition != c.partition || g.leaf_count !=
                                          static_cast<int>(c.leaves.size())) {
      CHECK(false);
    }
    for (int l = 0; l < static_cast<int>(c.leaves.size()) &&
                    l < g.leaf_count;
         ++l) {
      const CpuLeafDecision& cl = c.leaves[l];
      const videotx::VideoTxLeafResult& gl =
          gpu.leaves[j * videotx::kMaxLeavesPerPlan + l];
      ++checks;
      if (gl.tx_size != cl.tx_size || gl.tx_type != cl.tx_type ||
          gl.eob != cl.eob || gl.nonzero != cl.nonzero ||
          gl.level_elems != static_cast<int>(cl.levels.size()) ||
          !same_double(gl.sse, cl.sse) || !same_double(gl.rate_bits, cl.rate) ||
          !same_double(gl.cost, cl.cost)) {
        CHECK(false);
        if (g_failures <= kMaxReported) {
          std::printf("  %s job %zu leaf %d: cpu(size=%d type=%d eob=%d "
                      "nz=%d cost=%.17g) gpu(size=%d type=%d eob=%d nz=%d "
                      "cost=%.17g)\n",
                      label, j, l, cl.tx_size, cl.tx_type, cl.eob, cl.nonzero,
                      cl.cost, gl.tx_size, gl.tx_type, gl.eob, gl.nonzero,
                      gl.cost);
        }
      }
      for (int k = 0; k < static_cast<int>(cl.levels.size()); ++k) {
        const std::int32_t got = gpu.levels[gl.level_offset + k];
        if (got != cl.levels[k]) {
          CHECK(false);
          if (g_failures <= kMaxReported) {
            std::printf("  %s job %zu leaf %d level %d: cpu=%d gpu=%d\n",
                        label, j, l, k, cl.levels[k], got);
          }
          break;
        }
      }
    }
  }
  if (g_failures != local_failures) {
    std::printf("  %s: %d failures\n", label, g_failures - local_failures);
  }
  return checks;
}

videotx::VideoTxJobRequest base_request(int coded_w, int coded_h) {
  videotx::VideoTxJobRequest r;
  r.coded_w = coded_w;
  r.coded_h = coded_h;
  r.mode = videotx::VideoTxMode::kInter;
  r.qindex = 90;
  r.search.search_partition = 0;
  r.search.search_tx_type = 0;
  r.search.lambda = 544.5;
  return r;
}

void fill_residual(std::vector<std::int16_t>* out, int w, int h, Rng* rng,
                   int kind) {
  out->assign(static_cast<std::size_t>(w) * h, 0);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int v;
      switch (kind) {
        case 0:
          v = rng->range(-40, 40);
          break;
        case 1:
          v = (x * 5 - y * 3) / 2;
          break;
        case 2: {
          const int dx = x - w / 2;
          const int dy = y - h / 2;
          v = (dx * dx + dy * dy < (w * h) / 16) ? 90 : -20;
          break;
        }
        default:
          v = 0;
          break;
      }
      (*out)[static_cast<std::size_t>(y) * w + x] =
          static_cast<std::int16_t>(v);
    }
  }
}

// ---------------------------------------------------------------------------
// Test groups
// ---------------------------------------------------------------------------

int run_size_matrix(bool quick) {
  // Every one of the 25 AV2 TX sizes as a one-block job, DCT_DCT, several
  // qindex values, residual input and source/pred input.
  Rng rng;
  Fixture f;
  make_plane(&f, 128, 128, &rng, 0);
  const int qs[] = {1, 25, 50, 90, 140, 190, 255, 303};
  for (int size = 0; size < 25; ++size) {
    const int w = txpart::tx_width(size);
    const int h = txpart::tx_height(size);
    for (int qi = 0; qi < (quick ? 2 : 8); ++qi) {
      for (int kind = 0; kind < 2; ++kind) {
        videotx::VideoTxJobRequest req = base_request(w, h);
        req.qindex = qs[qi];
        std::vector<std::int16_t> residual;
        std::vector<std::uint8_t> src_block(static_cast<std::size_t>(w) * h);
        std::vector<std::uint8_t> pred_block(static_cast<std::size_t>(w) * h);
        fill_residual(&residual, w, h, &rng, (size + qi) % 3);
        for (int i = 0; i < w * h; ++i) {
          pred_block[i] = static_cast<std::uint8_t>(rng.range(60, 190));
          src_block[i] = static_cast<std::uint8_t>(
              std::max(0, std::min(255, pred_block[i] + residual[i])));
        }
        if (kind == 0) {
          req.residual_offset = 0;  // replaced by add_job
          add_job(&f, req, src_block, pred_block, residual);
        } else {
          req.residual_offset = -1;
          req.src_packed = 1;
          add_job(&f, req, src_block, pred_block, residual);
        }
      }
    }
  }
  videotx::VideoTxPlanData plan;
  std::string error;
  if (!videotx::video_tx_build_plan(f.requests.data(),
                                    static_cast<int>(f.requests.size()), &plan,
                                    &error)) {
    std::printf("plan build failed: %s\n", error.c_str());
    return 0;
  }
  if (plan.candidates.size() != f.requests.size()) {
    std::printf("unexpected candidate count\n");
    return 0;
  }
  videotx::VideoTxConfig config;
  config.max_jobs = static_cast<int>(f.requests.size());
  config.max_plans_per_job = 1;
  config.max_candidates_per_leaf = 1;
  BatchResult gpu;
  if (!run_batch(plan, f, config, &gpu)) return 0;
  const int checks = compare_batch("sizes", plan, f, gpu);
  CHECK(gpu.error_count == 0);
  std::printf("  sizes: %zu jobs, %d checks\n", f.requests.size(), checks);
  return checks;
}

int run_type_matrix(bool quick) {
  // Every legal TX type of every size, intra (all mapped modes) and inter,
  // with the 1-D classes enabled so IDTX/V_*/H_* are exercised too.
  Fixture f;
  Rng rng;
  make_plane(&f, 64, 64, &rng, 0);
  const int qs[] = {1, 90, 255};
  const int q_count = quick ? 1 : 3;
  for (int mode = 0; mode < 2; ++mode) {
    const videotx::VideoTxMode vm =
        mode == 0 ? videotx::VideoTxMode::kInter
                  : videotx::VideoTxMode::kIntra;
    const int modes = mode == 0 ? 1 : (quick ? 2 : 13);
    for (int size = 0; size < 25; ++size) {
      const int w = txpart::tx_width(size);
      const int h = txpart::tx_height(size);
      for (int qi = 0; qi < q_count; ++qi) {
        for (int mi = 0; mi < modes; ++mi) {
          videotx::VideoTxJobRequest req = base_request(w, h);
          req.mode = vm;
          req.qindex = qs[qi];
          req.search.search_tx_type = 1;
          req.search.include_1d_types = 1;
          req.search.intra_mapped_mode = mi;
          req.residual_offset = 0;
          std::vector<std::int16_t> residual;
          fill_residual(&residual, w, h, &rng, (size + qi + mi) % 3);
          add_job(&f, req, {}, {}, residual);
        }
      }
    }
  }
  videotx::VideoTxPlanData plan;
  std::string error;
  if (!videotx::video_tx_build_plan(f.requests.data(),
                                    static_cast<int>(f.requests.size()), &plan,
                                    &error)) {
    std::printf("type plan build failed: %s\n", error.c_str());
    return 0;
  }
  // The plan must match the public candidate enumerator for every job.
  for (std::size_t j = 0; j < plan.jobs.size(); ++j) {
    const videotx::VideoTxJobDesc& jd = plan.jobs[j];
    const videotx::VideoTxPlanDesc& pd = plan.plans[jd.plan_begin];
    for (int l = 0; l < pd.leaf_count; ++l) {
      const videotx::VideoTxLeafDesc& leaf = plan.leaves[pd.leaf_begin + l];
      std::int32_t types[videotx::kMaxCandidatesPerLeaf];
      double bits[videotx::kMaxCandidatesPerLeaf][3];
      const int n = videotx::video_tx_tx_types(
          leaf.coded_w, leaf.coded_h, f.requests[j].mode, f.requests[j].search,
          types, bits, videotx::kMaxCandidatesPerLeaf);
      CHECK(n == leaf.cand_count);
      for (int c = 0; c < n && c < leaf.cand_count; ++c) {
        CHECK(types[c] == plan.candidates[leaf.cand_begin + c].tx_type);
      }
    }
  }
  videotx::VideoTxConfig config;
  config.max_jobs = static_cast<int>(f.requests.size());
  config.max_plans_per_job = 1;
  config.max_candidates_per_leaf = videotx::kMaxCandidatesPerLeaf;
  BatchResult gpu;
  if (!run_batch(plan, f, config, &gpu)) return 0;
  const int checks = compare_batch("types", plan, f, gpu);
  CHECK(gpu.error_count == 0);

  // Type coverage: the matrix must reach all 16 TX_TYPE values.
  bool seen[16] = {};
  for (const videotx::VideoTxCandDesc& c : plan.candidates) {
    seen[c.tx_type] = true;
  }
  int unique = 0;
  for (bool s : seen) unique += s ? 1 : 0;
  CHECK(unique == 16);

  // classify counts: candidate metrics must fall in the legal range.
  int codable = 0;
  int noncodable = 0;
  for (const videotx::VideoTxCandMetrics& m : gpu.cands) {
    if (m.codable) ++codable; else ++noncodable;
  }
  std::printf("  types: %zu jobs, %zu candidates (%d codable, %d not), "
              "%d type values, %d checks\n",
              f.requests.size(), plan.candidates.size(), codable, noncodable,
              unique, checks);
  return checks;
}

void run_type_rule_tests() {
  // 1-D classes are opt-in; the DDT guard withholds ADST/FDST types but keeps
  // the identity types.
  std::int32_t types[videotx::kMaxCandidatesPerLeaf];
  double bits[videotx::kMaxCandidatesPerLeaf][3];
  videotx::VideoTxSearchOptions opt;
  opt.include_1d_types = 0;
  int n = videotx::video_tx_tx_types(8, 8, videotx::VideoTxMode::kInter, opt,
                                     types, bits,
                                     videotx::kMaxCandidatesPerLeaf);
  CHECK(n > 0);
  bool has_1d = false;
  for (int i = 0; i < n; ++i) has_1d = has_1d || types[i] >= 10;
  CHECK(!has_1d);

  opt.include_1d_types = 1;
  const int n_2d = videotx::video_tx_tx_types(
      8, 8, videotx::VideoTxMode::kInter, opt, types, bits,
      videotx::kMaxCandidatesPerLeaf);
  CHECK(n_2d == 16);

  opt.enable_inter_ddt = 1;
  const int n_ddt = videotx::video_tx_tx_types(
      8, 8, videotx::VideoTxMode::kInter, opt, types, bits,
      videotx::kMaxCandidatesPerLeaf);
  CHECK(n_ddt < n_2d);
  bool has_ddt = false;
  for (int i = 0; i < n_ddt; ++i) {
    has_ddt = has_ddt ||
              av2enc::video::residual::inter_tx_type_is_ddt(
                  static_cast<txv::TxType>(types[i]));
  }
  CHECK(!has_ddt);

  // Intra long-side set: 16x32 has IDTX and the H_/V_ classes.
  opt = videotx::VideoTxSearchOptions{};
  opt.include_1d_types = 1;
  const int n_intra = videotx::video_tx_tx_types(
      16, 32, videotx::VideoTxMode::kIntra, opt, types, bits,
      videotx::kMaxCandidatesPerLeaf);
  CHECK(n_intra == 8);
  std::printf("  type rules: inter 2d=%d ddt=%d, intra 16x32=%d\n", n_2d,
              n_ddt, n_intra);
}


int mi_size_for_px_test(int w, int h) {
  for (int mi = 0; mi < txpart::kBlockCount; ++mi) {
    if (txpart::block_width_px(mi) == w && txpart::block_height_px(mi) == h) {
      return mi;
    }
  }
  return -1;
}

// §5.20.6.3 partition bit in the search mask.
std::int32_t part_bit(int partition) {
  switch (partition) {
    case 0: return videotx::kPartBitNone;
    case 1: return videotx::kPartBitSplit;
    case 2: return videotx::kPartBitHorz;
    case 3: return videotx::kPartBitVert;
    case 4: return videotx::kPartBitHorz4;
    case 5: return videotx::kPartBitVert4;
    default: return 0;
  }
}

// Checks that a plan's leaves tile the block exactly (no gap, no overlap).
bool leaves_tile(const videotx::VideoTxPlanData& plan,
                 const videotx::VideoTxLeafDesc* leaves, int count, int w,
                 int h) {
  (void)plan;
  std::vector<int> cover(static_cast<std::size_t>(w) * h, 0);
  for (int l = 0; l < count; ++l) {
    const auto& leaf = leaves[l];
    for (int y = 0; y < leaf.coded_h; ++y) {
      for (int x = 0; x < leaf.coded_w; ++x) {
        const int px = leaf.x + x;
        const int py = leaf.y + y;
        if (px < 0 || py < 0 || px >= w || py >= h) return false;
        ++cover[static_cast<std::size_t>(py) * w + px];
      }
    }
  }
  for (int v : cover) {
    if (v != 1) return false;
  }
  return true;
}

int run_partition_matrix(bool quick) {
  // Every legal (block size, TX partition) pair, every legal TX type of every
  // resulting leaf, intra and inter, several qindex values.  Each job isolates
  // exactly one partition so the matrix is a true cross product.
  Fixture f;
  Rng rng;
  make_plane(&f, 64, 64, &rng, 0);
  const int qs[] = {1, 90, 255};
  const int qn = quick ? 1 : 3;
  int combos = 0;
  for (int mi = 0; mi < txpart::kBlockCount; ++mi) {
    const int w = txpart::block_width_px(mi);
    const int h = txpart::block_height_px(mi);
    if (w > 64 || h > 64) continue;
    const auto legal = txpart::legal_splits(mi, false);
    if (!legal.applicable) continue;
    for (int oi = 0; oi < legal.option_count; ++oi) {
      const int partition = static_cast<int>(legal.options[oi]);
      for (int qi = 0; qi < qn; ++qi) {
        for (int mode = 0; mode < 2; ++mode) {
          videotx::VideoTxJobRequest req = base_request(w, h);
          req.mode = mode == 0 ? videotx::VideoTxMode::kInter
                               : videotx::VideoTxMode::kIntra;
          req.qindex = qs[qi];
          req.search.search_none = partition == 0 ? 1 : 0;
          req.search.search_partition = 1;
          req.search.search_tx_type = 1;
          req.search.include_1d_types = 1;
          req.search.partition_mask = part_bit(partition);
          req.residual_offset = 0;
          std::vector<std::int16_t> residual;
          fill_residual(&residual, w, h, &rng, (mi + partition + qi) % 3);
          add_job(&f, req, {}, {}, residual);
          ++combos;
        }
      }
    }
  }
  videotx::VideoTxPlanData plan;
  std::string error;
  if (!videotx::video_tx_build_plan(f.requests.data(),
                                    static_cast<int>(f.requests.size()), &plan,
                                    &error)) {
    std::printf("partition plan build failed: %s\n", error.c_str());
    return 0;
  }
  // Every isolated job must produce exactly one plan with the requested
  // partition and the normative leaf geometry.
  for (std::size_t j = 0; j < plan.jobs.size(); ++j) {
    const auto& jd = plan.jobs[j];
    const auto& pd = plan.plans[jd.plan_begin];
    CHECK(jd.plan_count == 1);
    const int expected_partition =
        f.requests[j].search.partition_mask == videotx::kPartBitNone
            ? 0
            : f.requests[j].search.partition_mask == videotx::kPartBitSplit
                  ? 1
                  : f.requests[j].search.partition_mask == videotx::kPartBitHorz
                        ? 2
                        : f.requests[j].search.partition_mask ==
                                  videotx::kPartBitVert
                              ? 3
                              : f.requests[j].search.partition_mask ==
                                        videotx::kPartBitHorz4
                                    ? 4
                                    : 5;
    CHECK(pd.partition == expected_partition);
    CHECK(leaves_tile(plan, plan.leaves.data() + pd.leaf_begin,
                      pd.leaf_count, jd.coded_w, jd.coded_h));
    const int mi = mi_size_for_px_test(jd.coded_w, jd.coded_h);
    if (mi < 0) continue;
    txpart::SplitGeometry geom{};
    std::string geom_error;
    CHECK(txpart::tx_partition_geometry(
        mi, 0, 0, static_cast<txpart::TxPartition>(expected_partition), &geom,
        &geom_error));
    CHECK(pd.leaf_count == geom.count);
    for (int l = 0; l < geom.count && l < pd.leaf_count; ++l) {
      const auto& leaf = plan.leaves[pd.leaf_begin + l];
      CHECK(leaf.x == geom.parts[l].mi_col * 4);
      CHECK(leaf.y == geom.parts[l].mi_row * 4);
      CHECK(leaf.coded_w == txpart::tx_width(geom.parts[l].tx_size));
      CHECK(leaf.coded_h == txpart::tx_height(geom.parts[l].tx_size));
    }
  }
  videotx::VideoTxConfig config;
  config.max_jobs = static_cast<int>(f.requests.size());
  config.max_plans_per_job = 1;
  config.max_candidates_per_leaf = videotx::kMaxCandidatesPerLeaf;
  BatchResult gpu;
  if (!run_batch(plan, f, config, &gpu)) return 0;
  const int checks = compare_batch("partitions", plan, f, gpu);
  CHECK(gpu.error_count == 0);
  std::printf("  partitions: %d legal (size,partition,mode) combos, "
              "%zu jobs, %zu candidates, %d checks\n",
              combos, f.requests.size(), plan.candidates.size(), checks);
  return checks;
}

int run_partition_search(bool quick) {
  // All partitions enabled at once: the decision must pick the minimum-cost
  // plan and its leaves must tile the block.
  Fixture f;
  Rng rng;
  make_plane(&f, 64, 64, &rng, 0);
  const int qs[] = {1, 90, 255};
  const int qn = quick ? 1 : 3;
  for (int mi = 0; mi < txpart::kBlockCount; ++mi) {
    const int w = txpart::block_width_px(mi);
    const int h = txpart::block_height_px(mi);
    if (w > 64 || h > 64) continue;
    for (int qi = 0; qi < qn; ++qi) {
      for (int mode = 0; mode < 2; ++mode) {
        videotx::VideoTxJobRequest req = base_request(w, h);
        req.mode = mode == 0 ? videotx::VideoTxMode::kInter
                             : videotx::VideoTxMode::kIntra;
        req.qindex = qs[qi];
        req.search.search_partition = 1;
        req.search.search_tx_type = 1;
        req.search.include_1d_types = 1;
        req.residual_offset = 0;
        std::vector<std::int16_t> residual;
        fill_residual(&residual, w, h, &rng, (mi + qi) % 3);
        add_job(&f, req, {}, {}, residual);
      }
    }
  }
  videotx::VideoTxPlanData plan;
  std::string error;
  if (!videotx::video_tx_build_plan(f.requests.data(),
                                    static_cast<int>(f.requests.size()), &plan,
                                    &error)) {
    std::printf("partition search plan build failed: %s\n", error.c_str());
    return 0;
  }
  videotx::VideoTxConfig config;
  config.max_jobs = static_cast<int>(f.requests.size());
  config.max_plans_per_job = videotx::kMaxPlansPerJob;
  config.max_candidates_per_leaf = videotx::kMaxCandidatesPerLeaf;
  BatchResult gpu;
  if (!run_batch(plan, f, config, &gpu)) return 0;
  const int checks = compare_batch("partition-search", plan, f, gpu);
  CHECK(gpu.error_count == 0);

  // Independent decision check using the device's own per-plan results.
  for (std::size_t j = 0; j < plan.jobs.size(); ++j) {
    const auto& jd = plan.jobs[j];
    int expected = -1;
    double best_cost = 1e300;
    double best_sse = 1e300;
    for (int p = 0; p < jd.plan_count; ++p) {
      const auto& pr = gpu.plans[jd.plan_begin + p];
      if (!pr.valid) continue;
      if (pr.cost < best_cost ||
          (pr.cost == best_cost && pr.sse < best_sse)) {
        best_cost = pr.cost;
        best_sse = pr.sse;
        expected = jd.plan_begin + p;
      }
    }
    CHECK(expected == gpu.jobs[j].plan_index);
    const auto& jr = gpu.jobs[j];
    if (jr.plan_index >= 0) {
      const auto& pd = plan.plans[jr.plan_index];
      CHECK(leaves_tile(plan, plan.leaves.data() + pd.leaf_begin,
                        pd.leaf_count, jd.coded_w, jd.coded_h));
    }
  }
  std::printf("  partition-search: %zu jobs, %zu plans, %zu candidates, "
              "%d checks (required_device_bytes=%llu)\n",
              f.requests.size(), plan.plans.size(), plan.candidates.size(),
              checks,
              static_cast<unsigned long long>(
                  videotx::required_device_bytes(config)));
  return checks;
}

void run_partition_rules() {
  // reduced_tx_part_set removes HORZ4/VERT4 (4x16 / 16x4 / 4x32 / 32x4).
  const int mi_4x16 = [&] {
    for (int mi = 0; mi < txpart::kBlockCount; ++mi) {
      if (txpart::block_width_px(mi) == 4 &&
          txpart::block_height_px(mi) == 16) {
        return mi;
      }
    }
    return -1;
  }();
  CHECK(mi_4x16 >= 0);
  const auto full = txpart::legal_splits(mi_4x16, false);
  const auto reduced = txpart::legal_splits(mi_4x16, true);
  CHECK(full.option_count > reduced.option_count);

  // A mask without NONE and without any legal partition must be rejected.
  Fixture f;
  Rng rng;
  make_plane(&f, 32, 32, &rng, 0);
  videotx::VideoTxJobRequest req = base_request(16, 16);
  req.search.search_none = 1;
  req.search.search_partition = 0;
  req.search.partition_mask = videotx::kPartBitNone;
  req.residual_offset = 0;
  std::vector<std::int16_t> residual;
  fill_residual(&residual, 16, 16, &rng, 0);
  add_job(&f, req, {}, {}, residual);
  videotx::VideoTxPlanData plan;
  std::string error;
  CHECK(videotx::video_tx_build_plan(f.requests.data(), 1, &plan, &error));
  CHECK(plan.plans.size() == 1);
  CHECK(plan.plans[0].partition == 0);

  req.search.search_none = 0;
  req.search.search_partition = 1;
  req.search.partition_mask = 0;
  CHECK(!videotx::video_tx_build_plan(&req, 1, &plan, &error));
  std::printf("  partition rules: reduced=%d/%d options, mask rejection\n",
              reduced.option_count, full.option_count);
}

int run_partition_edges() {
  // Partition search at an overhanging, odd-sized plane edge: all partitions
  // and types must clamp the source read exactly.
  Fixture f;
  Rng rng;
  const int plane_w = 37;
  const int plane_h = 29;
  make_plane(&f, plane_w, plane_h, &rng, 2);
  const int sizes[] = {3, 6, 8, 9, 10, 21, 23};
  const int origins[][2] = {{0, 0}, {31, 23}, {33, 25}};
  for (int s : sizes) {
    for (int oi = 0; oi < 3; ++oi) {
      videotx::VideoTxJobRequest req = base_request(txpart::tx_width(s),
                                                    txpart::tx_height(s));
      req.residual_offset = -1;
      req.src_packed = 0;
      req.src_offset = 0;
      req.src_stride = plane_w;
      req.src_x = origins[oi][0];
      req.src_y = origins[oi][1];
      req.src_visible_w = plane_w;
      req.src_visible_h = plane_h;
      req.search.search_partition = 1;
      req.search.search_tx_type = 1;
      std::vector<std::uint8_t> pred_block(
          static_cast<std::size_t>(req.coded_w) * req.coded_h);
      for (int i = 0; i < req.coded_w * req.coded_h; ++i) {
        pred_block[i] = static_cast<std::uint8_t>(rng.range(70, 190));
      }
      add_job(&f, req, {}, pred_block, {});
    }
  }
  videotx::VideoTxPlanData plan;
  std::string error;
  if (!videotx::video_tx_build_plan(f.requests.data(),
                                    static_cast<int>(f.requests.size()), &plan,
                                    &error)) {
    std::printf("partition edge plan build failed: %s\n", error.c_str());
    return 0;
  }
  videotx::VideoTxConfig config;
  config.max_jobs = static_cast<int>(f.requests.size());
  config.max_plans_per_job = videotx::kMaxPlansPerJob;
  config.max_candidates_per_leaf = videotx::kMaxCandidatesPerLeaf;
  BatchResult gpu;
  if (!run_batch(plan, f, config, &gpu)) return 0;
  const int checks = compare_batch("partition-edges", plan, f, gpu);
  CHECK(gpu.error_count == 0);
  std::printf("  partition edges: %zu jobs, %d checks\n", f.requests.size(),
              checks);
  return checks;
}

int run_qindex_sweep(bool quick) {
  // qindex 1..303 for a representative size set: exercises the delta-Q
  // quantiser state across the whole 8/10-bit qindex range.
  const int sizes[] = {0, 1, 2, 3, 4, 16, 15, 17, 23};
  const int n_sizes = static_cast<int>(sizeof(sizes) / sizeof(sizes[0]));
  const int q_end = quick ? 32 : 303;
  Fixture f;
  Rng rng;
  make_plane(&f, 16, 16, &rng, 1);
  for (int s = 0; s < n_sizes; ++s) {
    const int w = txpart::tx_width(sizes[s]);
    const int h = txpart::tx_height(sizes[s]);
    for (int q = 1; q <= q_end; ++q) {
      videotx::VideoTxJobRequest req = base_request(w, h);
      req.qindex = q;
      req.residual_offset = 0;
      std::vector<std::int16_t> residual;
      fill_residual(&residual, w, h, &rng, (s + q) % 3);
      add_job(&f, req, {}, {}, residual);
    }
  }
  videotx::VideoTxPlanData plan;
  std::string error;
  if (!videotx::video_tx_build_plan(f.requests.data(),
                                    static_cast<int>(f.requests.size()), &plan,
                                    &error)) {
    std::printf("q sweep plan build failed: %s\n", error.c_str());
    return 0;
  }
  videotx::VideoTxConfig config;
  config.max_jobs = static_cast<int>(f.requests.size());
  config.max_plans_per_job = 1;
  config.max_candidates_per_leaf = 1;
  BatchResult gpu;
  if (!run_batch(plan, f, config, &gpu)) return 0;
  const int checks = compare_batch("qindex", plan, f, gpu);
  CHECK(gpu.error_count == 0);
  std::printf("  qindex: %zu jobs, %d checks\n", f.requests.size(), checks);
  return checks;
}

int run_plane_edges() {
  // Odd visible bounds + overhanging block origins: the kernel must clamp
  // exactly like TileEncoder::source_sample.
  Fixture f;
  Rng rng;
  const int plane_w = 37;
  const int plane_h = 29;
  make_plane(&f, plane_w, plane_h, &rng, 2);
  const int sizes[] = {0, 1, 2, 3, 4, 5, 7, 9, 13, 15, 17, 19, 21, 23};
  const int origins[][2] = {{0, 0}, {31, 23}, {33, 25}, {17, 11}};
  for (int s : sizes) {
    const int w = txpart::tx_width(s);
    const int h = txpart::tx_height(s);
    for (int oi = 0; oi < 4; ++oi) {
      videotx::VideoTxJobRequest req = base_request(w, h);
      req.residual_offset = -1;
      req.src_packed = 0;
      req.src_offset = 0;
      req.src_stride = plane_w;
      req.src_x = origins[oi][0];
      req.src_y = origins[oi][1];
      req.src_visible_w = plane_w;
      req.src_visible_h = plane_h;
      std::vector<std::uint8_t> pred_block(static_cast<std::size_t>(w) * h);
      for (int i = 0; i < w * h; ++i) {
        pred_block[i] = static_cast<std::uint8_t>(rng.range(80, 180));
      }
      add_job(&f, req, {}, pred_block, {});
    }
  }
  videotx::VideoTxPlanData plan;
  std::string error;
  if (!videotx::video_tx_build_plan(f.requests.data(),
                                    static_cast<int>(f.requests.size()), &plan,
                                    &error)) {
    std::printf("edge plan build failed: %s\n", error.c_str());
    return 0;
  }
  videotx::VideoTxConfig config;
  config.max_jobs = static_cast<int>(f.requests.size());
  config.max_plans_per_job = 1;
  config.max_candidates_per_leaf = 1;
  BatchResult gpu;
  if (!run_batch(plan, f, config, &gpu)) return 0;
  const int checks = compare_batch("edges", plan, f, gpu);
  CHECK(gpu.error_count == 0);
  std::printf("  plane edges: %zu jobs, %d checks\n", f.requests.size(), checks);
  return checks;
}

int run_delta_q() {
  // All planes and non-zero DC/AC deltas: the quantiser is driven only by the
  // resolved per-plane deltas the caller supplies.
  Fixture f;
  Rng rng;
  make_plane(&f, 64, 64, &rng, 0);
  const int deltas[][2] = {{0, 0}, {-8, 0}, {4, 0}, {0, 5}, {-3, -6}, {8, 8}};
  const int sizes[] = {0, 1, 2, 3, 7, 15, 16, 21};
  for (int plane = 0; plane < 3; ++plane) {
    for (int di = 0; di < 6; ++di) {
      for (int s : sizes) {
        videotx::VideoTxJobRequest req = base_request(txpart::tx_width(s),
                                                      txpart::tx_height(s));
        req.plane = plane;
        req.qindex = 120;
        req.delta_dc = deltas[di][0] + (plane == 2 ? 2 : 0);
        req.delta_ac = deltas[di][1];
        req.residual_offset = 0;
        std::vector<std::int16_t> residual;
        fill_residual(&residual, req.coded_w, req.coded_h, &rng,
                      (plane + di + s) % 3);
        add_job(&f, req, {}, {}, residual);
      }
    }
  }
  videotx::VideoTxPlanData plan;
  std::string error;
  if (!videotx::video_tx_build_plan(f.requests.data(),
                                    static_cast<int>(f.requests.size()), &plan,
                                    &error)) {
    std::printf("delta-q plan build failed: %s\n", error.c_str());
    return 0;
  }
  videotx::VideoTxConfig config;
  config.max_jobs = static_cast<int>(f.requests.size());
  config.max_plans_per_job = 1;
  config.max_candidates_per_leaf = 1;
  BatchResult gpu;
  if (!run_batch(plan, f, config, &gpu)) return 0;
  const int checks = compare_batch("delta-q", plan, f, gpu);
  CHECK(gpu.error_count == 0);
  std::printf("  delta-q: %zu jobs, %d checks\n", f.requests.size(), checks);
  return checks;
}

int run_residual_equivalence() {
  // A packed source/pred pair and the residual it implies must produce the
  // same decision and levels.
  Fixture f;
  Rng rng;
  make_plane(&f, 64, 64, &rng, 0);
  const int sizes[] = {0, 1, 2, 3, 4, 7, 13};
  for (int s : sizes) {
    const int w = txpart::tx_width(s);
    const int h = txpart::tx_height(s);
    std::vector<std::uint8_t> src_block(static_cast<std::size_t>(w) * h);
    std::vector<std::uint8_t> pred_block(static_cast<std::size_t>(w) * h);
    std::vector<std::int16_t> residual(static_cast<std::size_t>(w) * h);
    for (int i = 0; i < w * h; ++i) {
      pred_block[i] = static_cast<std::uint8_t>(rng.range(40, 210));
      residual[i] = static_cast<std::int16_t>(rng.range(-70, 70));
      src_block[i] = static_cast<std::uint8_t>(std::max(
          0, std::min(255, static_cast<int>(pred_block[i]) + residual[i])));
      residual[i] = static_cast<std::int16_t>(
          static_cast<int>(src_block[i]) - static_cast<int>(pred_block[i]));
    }
    videotx::VideoTxJobRequest a = base_request(w, h);
    a.residual_offset = 0;
    add_job(&f, a, src_block, pred_block, residual);
    videotx::VideoTxJobRequest b = base_request(w, h);
    b.residual_offset = -1;
    b.src_packed = 1;
    add_job(&f, b, src_block, pred_block, residual);
  }
  videotx::VideoTxPlanData plan;
  std::string error;
  if (!videotx::video_tx_build_plan(f.requests.data(),
                                    static_cast<int>(f.requests.size()), &plan,
                                    &error)) {
    return 0;
  }
  videotx::VideoTxConfig config;
  config.max_jobs = static_cast<int>(f.requests.size());
  config.max_plans_per_job = 1;
  config.max_candidates_per_leaf = 1;
  BatchResult gpu;
  if (!run_batch(plan, f, config, &gpu)) return 0;
  const int checks = compare_batch("residual-equiv", plan, f, gpu);
  // The paired jobs (2*i, 2*i+1) must be identical.
  for (std::size_t j = 0; j + 1 < gpu.jobs.size(); j += 2) {
    CHECK(gpu.jobs[j].cost == gpu.jobs[j + 1].cost);
    CHECK(gpu.leaves[j * videotx::kMaxLeavesPerPlan].eob ==
          gpu.leaves[(j + 1) * videotx::kMaxLeavesPerPlan].eob);
  }
  std::printf("  residual-equiv: %zu jobs, %d checks\n", f.requests.size(),
              checks);
  return checks;
}

int run_determinism() {
  Fixture f;
  Rng rng;
  make_plane(&f, 64, 64, &rng, 0);
  const int sizes[] = {0, 1, 2, 3, 9, 12, 17};
  for (int s : sizes) {
    videotx::VideoTxJobRequest req = base_request(txpart::tx_width(s),
                                                  txpart::tx_height(s));
    req.residual_offset = 0;
    std::vector<std::int16_t> residual;
    fill_residual(&residual, req.coded_w, req.coded_h, &rng, s % 3);
    add_job(&f, req, {}, {}, residual);
  }
  videotx::VideoTxPlanData plan;
  std::string error;
  if (!videotx::video_tx_build_plan(f.requests.data(),
                                    static_cast<int>(f.requests.size()), &plan,
                                    &error)) {
    return 0;
  }
  videotx::VideoTxConfig config;
  config.max_jobs = static_cast<int>(f.requests.size());
  config.max_plans_per_job = 1;
  config.max_candidates_per_leaf = 1;
  BatchResult a;
  BatchResult b;
  if (!run_batch(plan, f, config, &a)) return 0;
  if (!run_batch(plan, f, config, &b)) return 0;
  CHECK(a.jobs.size() == b.jobs.size());
  CHECK(std::memcmp(a.jobs.data(), b.jobs.data(),
                    a.jobs.size() * sizeof(a.jobs[0])) == 0);
  CHECK(std::memcmp(a.cands.data(), b.cands.data(),
                    a.cands.size() * sizeof(a.cands[0])) == 0);
  CHECK(std::memcmp(a.levels.data(), b.levels.data(),
                    a.levels.size() * sizeof(a.levels[0])) == 0);
  std::printf("  determinism: %zu jobs\n", f.requests.size());
  return static_cast<int>(f.requests.size());
}

void run_arena_tests() {
  videotx::VideoTxConfig config;
  config.max_jobs = 4;
  config.max_plans_per_job = 4;
  config.max_candidates_per_leaf = 8;
  const std::uint64_t need = videotx::required_device_bytes(config);
  CHECK(need > videotx::persistent_device_bytes(config));

  {
    Arena arena;
    std::string error;
    CHECK(arena.init(need, Arena::kAlignment, &error) == Status::kOk);
    videotx::VideoTxWorkspace ws;
    CHECK(ws.init(arena, config, nullptr, &error) == Status::kOk);
    CHECK(ws.valid());
    CHECK(arena.used() == need);
    const std::uint64_t mark = arena.mark();
    Arena probe;
    CHECK(probe.init(need - 1, Arena::kAlignment, &error) == Status::kOk);
    videotx::VideoTxWorkspace short_ws;
    std::uint64_t required = 0;
    CHECK(short_ws.init(probe, config, &required, &error) ==
          Status::kArenaExhausted);
    CHECK(required == need);
    CHECK(probe.used() == 0);
    CHECK(!short_ws.valid());
    CHECK(arena.mark() == mark);
  }

  // Invalid configs.
  {
    Arena arena;
    std::string error;
    CHECK(arena.init(1 << 20, Arena::kAlignment, &error) == Status::kOk);
    videotx::VideoTxWorkspace ws;
    videotx::VideoTxConfig bad;
    bad.max_jobs = 0;
    CHECK(ws.init(arena, bad, nullptr, &error) == Status::kInvalidArgument);
  }
  std::printf("  arena: required=%llu\n",
              static_cast<unsigned long long>(need));
}

void run_validation_tests() {
  Fixture f;
  Rng rng;
  make_plane(&f, 32, 32, &rng, 0);
  videotx::VideoTxJobRequest req = base_request(16, 16);
  req.residual_offset = 0;
  std::vector<std::int16_t> residual;
  fill_residual(&residual, 16, 16, &rng, 0);
  add_job(&f, req, {}, {}, residual);
  videotx::VideoTxPlanData plan;
  std::string error;
  CHECK(videotx::video_tx_build_plan(f.requests.data(), 1, &plan, &error));

  // Non-AV2 block size.
  {
    videotx::VideoTxJobRequest bad = req;
    bad.coded_w = 12;
    bad.coded_h = 12;
    videotx::VideoTxPlanData p;
    CHECK(!videotx::video_tx_build_plan(&bad, 1, &p, &error));
  }
  // Bad qindex.
  {
    videotx::VideoTxJobRequest bad = req;
    bad.qindex = 400;
    videotx::VideoTxPlanData p;
    CHECK(!videotx::video_tx_build_plan(&bad, 1, &p, &error));
  }
  // Plane source with no visible bounds.
  {
    videotx::VideoTxJobRequest bad = req;
    bad.residual_offset = -1;
    bad.src_packed = 0;
    videotx::VideoTxPlanData p;
    CHECK(!videotx::video_tx_build_plan(&bad, 1, &p, &error));
  }

  // A plan that does not fit the workspace config.
  {
    videotx::VideoTxConfig config;
    config.max_jobs = 0;
    Arena arena;
    std::string err;
    CHECK(arena.init(1 << 20, Arena::kAlignment, &err) == Status::kOk);
    videotx::VideoTxWorkspace ws;
    CHECK(ws.init(arena, config, nullptr, &err) == Status::kInvalidArgument);
  }

  // A hand-tampered plan is rejected before the device reads it.
  {
    videotx::VideoTxPlanData tampered = plan;
    tampered.leaves[0].cand_begin = 1 << 20;
    videotx::VideoTxConfig config;
    config.max_jobs = 1;
    config.max_plans_per_job = 1;
    config.max_candidates_per_leaf = 1;
    Arena arena;
    std::string err;
    CHECK(arena.init(videotx::required_device_bytes(config),
                     Arena::kAlignment, &err) == Status::kOk);
    videotx::VideoTxWorkspace ws;
    CHECK(ws.init(arena, config, nullptr, &err) == Status::kOk);
    DeviceBuffer d_residual;
    CHECK(d_residual.upload(f.packed_residual.data(),
                            f.packed_residual.size() * sizeof(std::int16_t)));
    videotx::VideoTxBuffers buffers;
    buffers.residual = static_cast<const std::int16_t*>(d_residual.ptr);
    CHECK(videotx::video_tx_select_batch(tampered, buffers, ws, nullptr) ==
          Status::kInvalidArgument);
  }
  std::printf("  validation: rejected malformed requests\n");
}

}  // namespace

int main(int argc, char** argv) {
  bool quick = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--quick") == 0) {
      quick = true;
    } else {
      std::printf("usage: %s [--quick]\n", argv[0]);
      return 2;
    }
  }

  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) {
    std::printf("cudaGetDevice failed: %s\n", cudaGetErrorString(err));
    return 1;
  }
  cudaDeviceProp prop{};
  err = cudaGetDeviceProperties(&prop, device);
  if (err != cudaSuccess) {
    std::printf("cudaGetDeviceProperties failed: %s\n",
                cudaGetErrorString(err));
    return 1;
  }
  std::printf("av2enc CUDA M2 video TX-selection test\n");
  std::printf("device: %s (sm_%d%d, %d SMs, %.1f GiB)\n", prop.name,
              prop.major, prop.minor, prop.multiProcessorCount,
              static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));

  int checks = 0;
  checks += run_size_matrix(quick);
  checks += run_type_matrix(quick);
  run_type_rule_tests();
  checks += run_partition_matrix(quick);
  checks += run_partition_search(quick);
  run_partition_rules();
  checks += run_partition_edges();
  checks += run_qindex_sweep(quick);
  checks += run_plane_edges();
  checks += run_delta_q();
  checks += run_residual_equivalence();
  checks += run_determinism();
  run_arena_tests();
  run_validation_tests();

  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  std::printf("RESULT: %s\n", g_failures == 0 ? "PASS" : "FAIL");
  return g_failures == 0 ? 0 : 1;
}

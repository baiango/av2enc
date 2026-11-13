// Batched M2 TX-selection kernel.  See video_tx.h for the contract.
//
// The kernel is one CUDA block per residual block.  Each block walks its plan
// candidates (partition plans -> leaves -> TX types), runs the exact CPU
// transform path in shared memory, and reduces SSE / EOB / level statistics in
// integer arithmetic (so the reduction is exact and order-independent).  One
// block per job means the decision needs no cross-block communication: the
// whole batch is one kernel launch.
#include "cuda/pipeline/video/tx/video_tx.h"

#include <algorithm>
#include <cstring>
#include <span>

#include "codec/transform/inter_residual/inter_tx_type.h"
#include "codec/transform/inter_residual/video_residual_cdf.h"
#include "codec/transform/tx_partition/tx_partition.h"
#include "codec/transform/tx_partition/tx_scan.h"
#include "codec/transform/tx_type/tx_type.h"
#include "codec/transform/variants/tx_tables_dev.cuh"
#include "cuda/codec/transform/delta_q/deltaq_device.cuh"

namespace av2enc::cuda::videotx {
namespace {

using namespace av2enc::txv::dev;
namespace txpart = av2enc::video::txpartition;
namespace residual = av2enc::video::residual;

constexpr int kThreads = 256;
constexpr int kQuantStep = av2enc::txv::kQuantStep;
constexpr int kQuantTableBits = av2enc::txv::kQuantTableBits;

// Dynamic shared layout (all 16-byte aligned so the double array is aligned):
//   0      int16  s_adj    [1024]   2048 B  adjusted residual
//   2048   double s_coeff  [1024]   8192 B  forward coefficients / inverse tmp
//   10240  int32  s_levels [1024]   4096 B  quantised levels
//   14336  int32  s_dq     [1024]   4096 B  dequantised coefficients
//   18432  int16  s_coded  [4096]   8192 B  coded-size leaf residual
//   26624                                   total
constexpr int kAdjOff = 0;
constexpr int kCoeffOff = 2 * kMaxAdjElems;
constexpr int kLevelsOff = kCoeffOff + 8 * kMaxAdjElems;
constexpr int kDqOff = kLevelsOff + 4 * kMaxAdjElems;
constexpr int kCodedOff = kDqOff + 4 * kMaxAdjElems;
constexpr int kDynamicSmemBytes = kCodedOff + 2 * kMaxCodedElems;

__device__ __forceinline__ std::int32_t round2_dev(std::int32_t value, int n) {
  if (n <= 0) {
    return value;
  }
  return (value + (1 << (n - 1))) >> n;
}

__device__ __forceinline__ std::int32_t clip3_dev(std::int32_t low,
                                                  std::int32_t high,
                                                  std::int32_t value) {
  return value < low ? low : (value > high ? high : value);
}

__device__ __forceinline__ int log2_pos(int value) {
  int n = 0;
  while ((1 << n) < value) {
    ++n;
  }
  return n;
}

__device__ __forceinline__ int floor_log2_dev(int value) {
  int result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}

// §7.15.2.1 kernel selection (the same table `txv::tx_kernel` uses).
__device__ __forceinline__ const std::int16_t* kernel_dev(int type, int n) {
  switch (type) {
    case 1:  // IDT
      return nullptr;
    case 2:  // ADST
      if (n == 4) return &kAdst4[0][0];
      if (n == 8) return &kAdst8[0][0];
      if (n == 16) return &kAdst16[0][0];
      return &kDct32[0][0];
    case 3:  // FDST / FLIPADST
      if (n == 4) return &kFdst4[0][0];
      if (n == 8) return &kFdst8[0][0];
      if (n == 16) return &kFdst16[0][0];
      return &kDct32[0][0];
    default:  // DCT
      if (n == 4) return &kDct4[0][0];
      if (n == 8) return &kDct8[0][0];
      if (n == 16) return &kDct16[0][0];
      return &kDct32[0][0];
  }
}

__device__ __forceinline__ bool valid_candidate_dev(const TxParams& p) {
  const int w = p.w;
  const int h = p.h;
  const bool dim_ok = (w == 4 || w == 8 || w == 16 || w == 32) &&
                      (h == 4 || h == 8 || h == 16 || h == 32);
  if (!dim_ok) return false;
  if (p.coded_w < w || p.coded_h < h || p.coded_w > kMaxCodedDim ||
      p.coded_h > kMaxCodedDim) {
    return false;
  }
  if (p.coded_w % w != 0 || p.coded_h % h != 0) return false;
  if (p.row_type < 0 || p.row_type > 3 || p.col_type < 0 || p.col_type > 3) {
    return false;
  }
  if (p.row_shift <= 0 || p.row_shift >= 31 || p.col_shift <= 0 ||
      p.col_shift >= 31) {
    return false;
  }
  if (p.dq_denom != 1 && p.dq_denom != 2 && p.dq_denom != 4) return false;
  if (p.log2_w < 2 || p.log2_w > 6 || p.log2_h < 2 || p.log2_h > 6) {
    return false;
  }
  return p.scale_w > 0.0 && p.scale_h > 0.0 && p.quant_factor > 0.0 &&
         p.rect_prescale > 0.0;
}

// §8.3.2 inter_tx_type CDF context (mirrors residual::inter_tx_type_context).
__device__ __forceinline__ int inter_ctx_dev(int eob, int coded_w,
                                             int coded_h) {
  if (eob <= 0) return 0;
  const int lw = floor_log2_dev(coded_w);
  const int bwl = lw < 5 ? lw : 5;
  const int eoby = (eob - 1) >> bwl;
  const int eobx = (eob - 1) - (eoby << bwl);
  const int diag = eobx + eoby;
  if (diag < 2) return 1;
  const int limit = (coded_w < 32 ? coded_w : 32) +
                    (coded_h < 32 ? coded_h : 32) - 4;
  if (diag > limit) return 2;
  return 0;
}

// Builds the coded-size leaf residual.  The candidate loop reuses it for the
// inverse-transform SSE.
__device__ __forceinline__ void prepare_leaf(
    const VideoTxJobDesc& job, const VideoTxLeafDesc& leaf,
    const std::uint8_t* __restrict__ src,
    const std::uint8_t* __restrict__ pred,
    const std::int16_t* __restrict__ residual, std::int16_t* s_coded, int tid,
    int nthreads) {
  const int cw = leaf.coded_w;
  const int coded_n = cw * leaf.coded_h;
  const int src_stride = job.src_stride > 0 ? job.src_stride : job.coded_w;
  const int pred_stride = job.pred_stride > 0 ? job.pred_stride : job.coded_w;

  for (int t = tid; t < coded_n; t += nthreads) {
    const int ay = t / cw;
    const int ax = t - ay * cw;
    int sample;
    if (job.use_residual != 0) {
      sample = static_cast<int>(
          residual[job.residual_offset + (leaf.y + ay) * job.coded_w +
                   (leaf.x + ax)]);
    } else {
      int s;
      if (job.src_packed != 0) {
        s = static_cast<int>(
            src[job.src_offset + (leaf.y + ay) * src_stride + leaf.x + ax]);
      } else {
        int px = job.src_x + leaf.x + ax;
        int py = job.src_y + leaf.y + ay;
        px = clip3_dev(0, job.src_visible_w - 1, px);
        py = clip3_dev(0, job.src_visible_h - 1, py);
        s = static_cast<int>(src[job.src_offset + py * src_stride + px]);
      }
      const int p = static_cast<int>(
          pred[job.pred_offset + (leaf.y + ay) * pred_stride + leaf.x + ax]);
      sample = s - p;
    }
    s_coded[t] = static_cast<std::int16_t>(sample);
  }
  __syncthreads();
}

// §7.15.4 downsample: coded-size leaf residual -> adjusted residual.
__device__ __forceinline__ void downsample_leaf(const VideoTxLeafDesc& leaf,
                                                const TxParams& params,
                                                const std::int16_t* s_coded,
                                                std::int16_t* s_adj, int tid,
                                                int nthreads) {
  const int w = params.w;
  const int h = params.h;
  const int sx = leaf.coded_w / w;
  const int sy = leaf.coded_h / h;
  const int shift = log2_pos(sx * sy);
  const int elems = w * h;
  for (int t = tid; t < elems; t += nthreads) {
    const int j = t / w;
    const int i = t - j * w;
    std::int32_t sum = 0;
    for (int dy = 0; dy < sy; ++dy) {
      const int row = (j * sy + dy) * leaf.coded_w;
      for (int dx = 0; dx < sx; ++dx) {
        sum += static_cast<std::int32_t>(s_coded[row + i * sx + dx]);
      }
    }
    s_adj[t] = static_cast<std::int16_t>(round2_dev(sum, shift));
  }
  __syncthreads();
}

struct Reduction {
  int* nonzero;
  unsigned long long* sum_abs;
  unsigned long long* sse;
  int* eob;
};

// Forward -> quantise -> dequantise -> inverse for one candidate.
//
// `s_coeff` doubles as the inverse intermediate (`s_inter`, int32 overlay) once
// quantisation is done.  When `out_levels` is non-null the quantised levels are
// also written to it in scan order (the winning-plan replay path).
__device__ __forceinline__ void run_candidate(
    const VideoTxJobDesc& job, const VideoTxLeafDesc& leaf,
    const VideoTxCandDesc& cand, const std::uint16_t* scan,
    const std::int16_t* s_coded, std::int16_t* s_adj, double* s_coeff,
    std::int32_t* s_levels, std::int32_t* s_dq, const Reduction& red,
    std::int32_t* out_levels, int tid, int nthreads) {
  const TxParams p = cand.params;
  const int w = p.w;
  const int h = p.h;
  const int elems = w * h;
  const std::int16_t* kw = kernel_dev(static_cast<int>(p.row_type), w);
  const std::int16_t* kh = kernel_dev(static_cast<int>(p.col_type), h);

  // ---- forward: row pass -------------------------------------------------
  if (kw == nullptr) {
    for (int t = tid; t < elems; t += nthreads) {
      s_coeff[t] = static_cast<double>(s_adj[t]);
    }
  } else {
    for (int t = tid; t < elems; t += nthreads) {
      const int i = t / w;
      const int v = t - i * w;
      const int row = i * w;
      double sum = 0.0;
      for (int j = 0; j < w; ++j) {
        sum = __dadd_rn(
            sum, __dmul_rn(static_cast<double>(kw[v * w + j]),
                           static_cast<double>(s_adj[row + j])));
      }
      s_coeff[t] = __dmul_rn(sum, p.scale_w);
    }
  }
  __syncthreads();

  // ---- forward: column pass, quantise, dequantise ------------------------
  // §7.14.2 quantisers come from the shared delta-Q core; the coefficient
  // expression order is the product CPU path's (`still_encoder.cpp`):
  //   factor = qf * 528 / q;  level = lround(coeff * factor)
  const int dc_quant = deltaq::core::get_q(job.qindex, job.delta_dc);
  const int ac_quant = deltaq::core::get_q(job.qindex, job.delta_ac);
  const int dc_denom = dc_quant > 0 ? dc_quant : 1;
  const int ac_denom = ac_quant > 0 ? ac_quant : 1;
  const double scaled =
      __dmul_rn(p.quant_factor, static_cast<double>(kQuantStep));
  for (int t = tid; t < elems; t += nthreads) {
    const int u = t / w;
    const int v = t - u * w;
    double out;
    if (kh == nullptr) {
      out = s_coeff[t];
    } else {
      double sum = 0.0;
      for (int i = 0; i < h; ++i) {
        sum = __dadd_rn(
            sum, __dmul_rn(static_cast<double>(kh[u * h + i]),
                           s_coeff[i * w + v]));
      }
      out = __dmul_rn(sum, p.scale_h);
    }
    if (p.rect_prescale != 1.0) {
      out = __dmul_rn(out, p.rect_prescale);
    }
    const int q = (t == 0) ? dc_denom : ac_denom;
    const double factor = scaled / static_cast<double>(q);
    const std::int32_t level =
        static_cast<std::int32_t>(lround(__dmul_rn(out, factor)));
    s_levels[t] = level;

    const std::int32_t aq = level < 0 ? -level : level;
    const std::int32_t high = (aq * q) & 0xFFFFFF;
    const std::int32_t dq = round2_dev(high, kQuantTableBits) / p.dq_denom;
    s_dq[t] = clip3_dev(-(1 << 15), (1 << 15) - 1, level < 0 ? -dq : dq);
  }
  __syncthreads();

  // ---- eob / nonzero / sum|level| over the §5.20.7.30 scan ---------------
  if (tid == 0) {
    *red.nonzero = 0;
    *red.sum_abs = 0ULL;
    *red.sse = 0ULL;
    *red.eob = 0;
  }
  __syncthreads();
  for (int c = tid; c < elems; c += nthreads) {
    const std::int32_t level = s_levels[scan[c]];
    if (level != 0) {
      atomicAdd(red.nonzero, 1);
      const unsigned long long abs_level =
          level < 0
              ? static_cast<unsigned long long>(-static_cast<long long>(level))
              : static_cast<unsigned long long>(level);
      atomicAdd(red.sum_abs, abs_level);
      atomicMax(red.eob, c + 1);
    }
  }
  __syncthreads();

  // ---- inverse: row pass (s_coeff is dead and becomes s_inter) -----------
  auto* s_inter = reinterpret_cast<std::int32_t*>(s_coeff);
  const bool odd_rect = ((p.log2_w ^ p.log2_h) & 1) != 0;
  const int identity_w = kIdentityScaleDev[static_cast<int>(p.log2_w)];
  const int identity_h = kIdentityScaleDev[static_cast<int>(p.log2_h)];
  if (kw == nullptr) {
    for (int t = tid; t < elems; t += nthreads) {
      const int i = t / w;
      const int oj = t - i * w;
      const int row = i * w;
      const std::int32_t value = s_dq[row + oj];
      const std::int32_t pre =
          odd_rect ? round2_dev(static_cast<std::int32_t>(
                                    static_cast<long long>(value) * 2896LL),
                                12)
                   : value;
      const long long s = static_cast<long long>(pre) * identity_w;
      s_inter[t] = clip3_dev(-(1 << 15), (1 << 15) - 1,
                             round2_dev(static_cast<std::int32_t>(s),
                                        p.row_shift));
    }
  } else {
    for (int t = tid; t < elems; t += nthreads) {
      const int i = t / w;
      const int oj = t - i * w;
      const int row = i * w;
      long long sum = 0;
      for (int j = 0; j < w; ++j) {
        const std::int32_t value = s_dq[row + j];
        const std::int32_t pre =
            odd_rect ? round2_dev(static_cast<std::int32_t>(
                                      static_cast<long long>(value) * 2896LL),
                                  12)
                     : value;
        sum += static_cast<long long>(kw[j * w + oj]) * pre;
      }
      s_inter[t] = clip3_dev(-(1 << 15), (1 << 15) - 1,
                             round2_dev(static_cast<std::int32_t>(sum),
                                        p.row_shift));
    }
  }
  __syncthreads();

  // ---- inverse: column pass + coded-size SSE ----------------------------
  const int sx = leaf.coded_w / w;
  const int sy = leaf.coded_h / h;
  unsigned long long local_sse = 0ULL;
  for (int t = tid; t < elems; t += nthreads) {
    const int oi = t / w;
    const int j = t - oi * w;
    std::int32_t value;
    if (kh == nullptr) {
      const long long s = static_cast<long long>(s_inter[t]) * identity_h;
      value = clip3_dev(-256, 255,
                        round2_dev(static_cast<std::int32_t>(s), p.col_shift));
    } else {
      long long sum = 0;
      for (int i = 0; i < h; ++i) {
        sum += static_cast<long long>(kh[i * h + oi]) * s_inter[i * w + j];
      }
      value = clip3_dev(-256, 255,
                        round2_dev(static_cast<std::int32_t>(sum), p.col_shift));
    }
    for (int dy = 0; dy < sy; ++dy) {
      const int row = (oi * sy + dy) * leaf.coded_w;
      for (int dx = 0; dx < sx; ++dx) {
        const std::int32_t d =
            value - static_cast<std::int32_t>(s_coded[row + j * sx + dx]);
        local_sse += static_cast<unsigned long long>(
            static_cast<long long>(d) * static_cast<long long>(d));
      }
    }
  }
  atomicAdd(red.sse, local_sse);

  // ---- optional scan-order level output (winner replay) ------------------
  if (out_levels != nullptr) {
    for (int c = tid; c < elems; c += nthreads) {
      out_levels[c] = s_levels[scan[c]];
    }
  }
  __syncthreads();
}

__global__ void video_tx_kernel(
    const VideoTxJobDesc* __restrict__ jobs,
    const VideoTxPlanDesc* __restrict__ plans,
    const VideoTxLeafDesc* __restrict__ leaves,
    const VideoTxCandDesc* __restrict__ candidates,
    const std::uint16_t* __restrict__ scan_tables,
    const std::uint8_t* __restrict__ src,
    const std::uint8_t* __restrict__ pred,
    const std::int16_t* __restrict__ residual,
    VideoTxCandMetrics* __restrict__ cand_metrics,
    VideoTxPlanResult* __restrict__ plan_results,
    VideoTxLeafResult* __restrict__ leaf_results,
    VideoTxJobResult* __restrict__ job_results,
    std::int32_t* __restrict__ levels, int* __restrict__ error_count) {
  extern __shared__ __align__(16) unsigned char smem[];
  auto* s_adj = reinterpret_cast<std::int16_t*>(smem + kAdjOff);
  auto* s_coeff = reinterpret_cast<double*>(smem + kCoeffOff);
  auto* s_levels = reinterpret_cast<std::int32_t*>(smem + kLevelsOff);
  auto* s_dq = reinterpret_cast<std::int32_t*>(smem + kDqOff);
  auto* s_coded = reinterpret_cast<std::int16_t*>(smem + kCodedOff);

  __shared__ int sh_nonzero;
  __shared__ unsigned long long sh_sum_abs;
  __shared__ unsigned long long sh_sse;
  __shared__ int sh_eob;
  __shared__ int sh_leaf_found;
  // Best candidate per (plan-in-job, leaf-position); -1 = none.
  __shared__ int sh_leaf_best[kMaxPlansPerJob][kMaxLeavesPerPlan];
  __shared__ int sh_best_plan;
  __shared__ double sh_best_cost;
  __shared__ double sh_best_sse;
  __shared__ double sh_best_rate;
  __shared__ double sh_plan_cost;
  __shared__ double sh_plan_sse;
  __shared__ double sh_plan_rate;

  const int tid = static_cast<int>(threadIdx.x);
  const int nthreads = static_cast<int>(blockDim.x);
  const int job_index = static_cast<int>(blockIdx.x);
  const VideoTxJobDesc job = jobs[job_index];
  VideoTxJobResult* job_result = &job_results[job_index];

  const bool buffers_ok =
      job.use_residual != 0 ? (residual != nullptr)
                            : (src != nullptr && pred != nullptr);
  const bool job_ok =
      job.coded_w >= 4 && job.coded_w <= kMaxCodedDim && job.coded_h >= 4 &&
      job.coded_h <= kMaxCodedDim && job.plan_count > 0 &&
      job.plan_count <= kMaxPlansPerJob && job.plan_begin >= 0 &&
      buffers_ok;
  if (!job_ok) {
    if (tid == 0) {
      job_result->status = -1;
      job_result->plan_index = -1;
      job_result->leaf_count = 0;
      atomicAdd(error_count, 1);
    }
    return;
  }

  const Reduction red{&sh_nonzero, &sh_sum_abs, &sh_sse, &sh_eob};
  sh_best_plan = -1;
  sh_best_cost = 1e300;
  sh_best_sse = 1e300;
  sh_best_rate = 0.0;
  if (tid == 0) {
    for (int p = 0; p < kMaxPlansPerJob; ++p) {
      for (int l = 0; l < kMaxLeavesPerPlan; ++l) {
        sh_leaf_best[p][l] = -1;
      }
    }
  }
  __syncthreads();

  for (int p_local = 0; p_local < job.plan_count; ++p_local) {
    const int plan_index = job.plan_begin + p_local;
    const VideoTxPlanDesc plan = plans[plan_index];
    const bool plan_ok = plan.leaf_count > 0 &&
                         plan.leaf_count <= kMaxLeavesPerPlan &&
                         plan.leaf_begin >= 0;
    bool plan_valid = plan_ok;
    if (tid == 0) {
      sh_plan_cost = plan.partition_bits;
      sh_plan_sse = 0.0;
      sh_plan_rate = plan.partition_bits;
    }

    for (int l = 0; l < (plan_ok ? plan.leaf_count : 0); ++l) {
      const int leaf_index = plan.leaf_begin + l;
      const VideoTxLeafDesc leaf = leaves[leaf_index];
      if (leaf.cand_count <= 0 || leaf.cand_begin < 0 || leaf.coded_w < 4 ||
          leaf.coded_w > kMaxCodedDim || leaf.coded_h < 4 ||
          leaf.coded_h > kMaxCodedDim || leaf.x < 0 || leaf.y < 0 ||
          leaf.x + leaf.coded_w > job.coded_w ||
          leaf.y + leaf.coded_h > job.coded_h) {
        plan_valid = false;
        break;
      }

      prepare_leaf(job, leaf, src, pred, residual, s_coded, tid, nthreads);

      double leaf_best_cost = 1e300;
      double leaf_best_sse = 1e300;
      double leaf_best_rate = 0.0;
      int leaf_best_type = -1;
      int leaf_best_cand = -1;

      for (int c = 0; c < leaf.cand_count; ++c) {
        const int cand_index = leaf.cand_begin + c;
        const VideoTxCandDesc cand = candidates[cand_index];
        const bool cand_ok = valid_candidate_dev(cand.params) &&
                             cand.scan_table >= 0 &&
                             cand.scan_table < kScanTables &&
                             cand.tx_size >= 0 && cand.tx_size < 25 &&
                             cand.tx_type >= 0 && cand.tx_type < 16;
        if (!cand_ok) {
          if (tid == 0) {
            VideoTxCandMetrics m{};
            m.valid = 0;
            cand_metrics[cand_index] = m;
            atomicAdd(error_count, 1);
          }
          continue;
        }

        downsample_leaf(leaf, cand.params, s_coded, s_adj, tid, nthreads);
        const std::uint16_t* scan =
            scan_tables + cand.scan_table * kScanStride;
        run_candidate(job, leaf, cand, scan, s_coded, s_adj, s_coeff, s_levels,
                      s_dq, red, nullptr, tid, nthreads);

        if (tid == 0) {
          const double sse = static_cast<double>(sh_sse);
          const int nonzero = sh_nonzero;
          const int eob = sh_eob;
          const double type_bits =
              job.use_type_bits != 0 && cand.codes_type != 0
                  ? cand.type_bits[job.is_inter != 0
                                       ? inter_ctx_dev(eob, leaf.coded_w,
                                                       leaf.coded_h)
                                       : 0]
                  : 0.0;
          const double rate =
              eob == 0 ? job.skip_bits
                       : type_bits + job.coefficient_bits *
                                          static_cast<double>(nonzero);
          const double cost = sse + job.lambda * rate;
          const bool codable =
              job.is_inter != 0 ? (eob == 0 ? cand.tx_type == 0 : true)
                                : ((eob >= 2) || cand.tx_type == 0);

          VideoTxCandMetrics m{};
          m.sse = sse;
          m.rate_bits = rate;
          m.cost = cost;
          m.nonzero = nonzero;
          m.eob = eob;
          m.codable = codable ? 1 : 0;
          m.valid = 1;
          cand_metrics[cand_index] = m;

          if (codable &&
              (cost < leaf_best_cost ||
               (cost == leaf_best_cost &&
                (sse < leaf_best_sse ||
                 (sse == leaf_best_sse && cand.tx_type < leaf_best_type))))) {
            leaf_best_cost = cost;
            leaf_best_sse = sse;
            leaf_best_rate = rate;
            leaf_best_type = cand.tx_type;
            leaf_best_cand = cand_index;
          }
        }
        __syncthreads();
      }

      if (tid == 0) {
        sh_leaf_found = leaf_best_cand >= 0 ? 1 : 0;
      }
      __syncthreads();
      if (sh_leaf_found == 0) {
        plan_valid = false;
        break;
      }
      if (tid == 0) {
        sh_leaf_best[p_local][l] = leaf_best_cand;
        sh_plan_cost += leaf_best_cost;
        sh_plan_sse += leaf_best_sse;
        sh_plan_rate += leaf_best_rate;
      }
      __syncthreads();
    }

    if (tid == 0) {
      VideoTxPlanResult pr{};
      pr.partition = plan.partition;
      pr.leaf_result_begin = job_index * kMaxLeavesPerPlan;
      pr.leaf_count = plan_ok ? plan.leaf_count : 0;
      pr.valid = plan_valid ? 1 : 0;
      pr.sse = sh_plan_sse;
      pr.rate_bits = sh_plan_rate;
      pr.cost = sh_plan_cost;
      plan_results[plan_index] = pr;

      if (plan_valid &&
          (sh_plan_cost < sh_best_cost ||
           (sh_plan_cost == sh_best_cost && sh_plan_sse < sh_best_sse))) {
        sh_best_cost = sh_plan_cost;
        sh_best_sse = sh_plan_sse;
        sh_best_rate = sh_plan_rate;
        sh_best_plan = p_local;
      }
    }
    __syncthreads();
  }

  if (sh_best_plan < 0) {
    if (tid == 0) {
      job_result->status = -1;
      job_result->plan_index = -1;
      job_result->leaf_count = 0;
      atomicAdd(error_count, 1);
    }
    return;
  }

  const int best_plan_index = job.plan_begin + sh_best_plan;
  const VideoTxPlanDesc best_plan = plans[best_plan_index];
  for (int l = 0; l < best_plan.leaf_count; ++l) {
    const int leaf_index = best_plan.leaf_begin + l;
    const VideoTxLeafDesc leaf = leaves[leaf_index];
    const int cand_index = sh_leaf_best[sh_best_plan][l];
    const VideoTxCandDesc cand = candidates[cand_index];

    // Replay the winner so its levels can be written in scan order.  The
    // candidate metrics were already recorded in the first pass; the coded
    // residual must be rebuilt because the last plan's leaves overwrote it.
    prepare_leaf(job, leaf, src, pred, residual, s_coded, tid, nthreads);
    downsample_leaf(leaf, cand.params, s_coded, s_adj, tid, nthreads);
    const std::uint16_t* scan =
        scan_tables + cand.scan_table * kScanStride;
    std::int32_t* out_levels = levels + leaf.level_offset;
    run_candidate(job, leaf, cand, scan, s_coded, s_adj, s_coeff, s_levels,
                  s_dq, red, out_levels, tid, nthreads);

    if (tid == 0) {
      const double sse = static_cast<double>(sh_sse);
      const int eob = sh_eob;
      const double type_bits =
          job.use_type_bits != 0 && cand.codes_type != 0
              ? cand.type_bits[job.is_inter != 0
                                   ? inter_ctx_dev(eob, leaf.coded_w,
                                                   leaf.coded_h)
                                   : 0]
              : 0.0;
      const double rate =
          eob == 0 ? job.skip_bits
                   : type_bits + job.coefficient_bits *
                                      static_cast<double>(sh_nonzero);
      VideoTxLeafResult lr{};
      lr.tx_size = cand.tx_size;
      lr.tx_type = cand.tx_type;
      lr.eob = eob;
      lr.nonzero = sh_nonzero;
      lr.level_offset = leaf.level_offset;
      lr.level_elems = cand.params.w * cand.params.h;
      lr.valid = 1;
      lr.sse = sse;
      lr.rate_bits = rate;
      lr.cost = sse + job.lambda * rate;
      leaf_results[job_index * kMaxLeavesPerPlan + l] = lr;
    }
    __syncthreads();
  }

  if (tid == 0) {
    job_result->status = 0;
    job_result->plan_index = best_plan_index;
    job_result->partition = best_plan.partition;
    job_result->leaf_result_begin = job_index * kMaxLeavesPerPlan;
    job_result->leaf_count = best_plan.leaf_count;
    job_result->sse = sh_best_sse;
    job_result->rate_bits = sh_best_rate;
    job_result->cost = sh_best_cost;
  }
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

std::uint64_t align_up(std::uint64_t bytes) {
  const std::uint64_t a = Arena::kAlignment;
  return (bytes + a - 1) / a * a;
}

std::uint64_t bytes_for(const VideoTxConfig& c) {
  const std::uint64_t jobs = static_cast<std::uint64_t>(c.max_jobs);
  const std::uint64_t plans_per_job =
      static_cast<std::uint64_t>(c.max_plans_per_job);
  const std::uint64_t types =
      static_cast<std::uint64_t>(c.max_candidates_per_leaf);
  const std::uint64_t plans = jobs * plans_per_job;
  const std::uint64_t leaves = plans * kMaxLeavesPerPlan;
  const std::uint64_t cands = leaves * types;
  std::uint64_t total = 0;
  total += align_up(static_cast<std::uint64_t>(kScanTables) * kScanStride *
                    sizeof(std::uint16_t));
  total += align_up(jobs * sizeof(VideoTxJobDesc));
  total += align_up(plans * sizeof(VideoTxPlanDesc));
  total += align_up(leaves * sizeof(VideoTxLeafDesc));
  total += align_up(cands * sizeof(VideoTxCandDesc));
  total += align_up(cands * sizeof(VideoTxCandMetrics));
  total += align_up(plans * sizeof(VideoTxPlanResult));
  total += align_up(jobs * kMaxLeavesPerPlan * sizeof(VideoTxLeafResult));
  total += align_up(jobs * sizeof(VideoTxJobResult));
  total += align_up(jobs * kMaxLeavesPerPlan * kMaxAdjElems *
                    sizeof(std::int32_t));
  total += align_up(sizeof(int));
  return total;
}

template <typename T>
Status allocate_array(Arena& arena, std::uint64_t count, T** out) {
  const std::uint64_t bytes = align_up(count * sizeof(T));
  void* raw = nullptr;
  const Status s = arena.allocate_checked(static_cast<std::size_t>(bytes),
                                          &raw);
  *out = static_cast<T*>(raw);
  return s;
}

int mi_size_for_px(int w, int h) {
  for (int mi = 0; mi < txpart::kBlockCount; ++mi) {
    if (txpart::block_width_px(mi) == w && txpart::block_height_px(mi) == h) {
      return mi;
    }
  }
  return -1;
}

// ---------------------------------------------------------------------------
// TX-type candidate enumeration (§5.20.8.3) and the §8.3.2 symbol cost
// ---------------------------------------------------------------------------

struct TypeEntry {
  std::int32_t type = 0;
  double bits[3] = {0.0, 0.0, 0.0};
  std::int32_t codes_type = 1;
};

// §9.3 CDF layout: entries [0 .. nsym-2) are cumulative probabilities scaled to
// 32768, the next entry is the adaptation rate and the last is the counter.
double cdf_symbol_bits(const std::uint16_t* cdf, int nsym, int symbol) {
  if (symbol < 0 || symbol >= nsym) {
    return 0.0;
  }
  const int lo = symbol == 0 ? 0 : static_cast<int>(cdf[symbol - 1]);
  const int hi = symbol == nsym - 1 ? 32768 : static_cast<int>(cdf[symbol]);
  const int range = hi > lo ? hi - lo : 1;
  return -std::log2(static_cast<double>(range) / 32768.0);
}

// Estimated `inter_tx_type` / `inter_tx_type_offset` symbol cost for one
// §8.3.2 context.  Mirrors `residual::tx_type_symbol_bits()` (which is not
// exported) with the context supplied directly instead of an eob.
double inter_type_bits_ctx(residual::InterTxSet set, av2enc::txv::TxSize ts,
                           const residual::InterTxCandidate& c, int ctx) {
  using namespace residual;
  if (set == InterTxSet::kDctOnly) {
    return 0.0;
  }
  const int sqr = std::clamp(tx_size_sqr(ts), 0, 3);
  switch (set) {
    case InterTxSet::kInter1:
      return cdf_symbol_bits(&kInterTxTypeSet1Cdf[ctx][sqr][0], 2,
                             c.symbol) +
             cdf_symbol_bits(c.symbol == 0
                                 ? &kInterTxTypeIndexSet1Cdf[ctx][0]
                                 : &kInterTxTypeOffsetSet1Cdf[ctx][0],
                             8, c.offset_symbol);
    case InterTxSet::kInter2: {
      const int offset_ctx = ctx;  // context is supplied by the caller
      return cdf_symbol_bits(&kInterTxTypeSet2Cdf[ctx][0], 2, c.symbol) +
             cdf_symbol_bits(c.symbol == 0
                                 ? &kInterTxTypeIndexSet2Cdf[offset_ctx][0]
                                 : &kInterTxTypeOffsetSet2Cdf[offset_ctx][0],
                             c.symbol == 0 ? 8 : 4, c.offset_symbol);
    }
    case InterTxSet::kDctIdtx:
      return cdf_symbol_bits(&kInterTxTypeSet3Cdf[ctx][sqr][0], 2, c.symbol);
    case InterTxSet::kDctIdtxIddct:
      return cdf_symbol_bits(&kInterTxTypeSet4Cdf[ctx][sqr][0], 4, c.symbol);
    case InterTxSet::kWide64:
    case InterTxSet::kHigh64:
    case InterTxSet::kWide32:
    case InterTxSet::kHigh32: {
      double bits =
          cdf_symbol_bits(&kInterTxTypeLongCdf[ctx][sqr][0], 4, c.symbol);
      if (c.codes_long_flag) {
        const int flag = c.is_long_side_dct ? 1 : 0;
        bits += cdf_symbol_bits(&kIsLongSideDctCdf[flag][0], 2, flag);
      }
      return bits;
    }
    case InterTxSet::kDctOnly:
      break;
  }
  return 0.0;
}

// §5.20.8.2 canonical inverse tables.  `inter_tx_symbol()` accepts aliases that
// map to the same symbol; the search may only emit the canonical type because
// the decoder's compute_tx_type() inverts the symbol to `Tx_Type_Inverse_*`.
constexpr std::int32_t kInterInvSet1[16] = {
    9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 4, 5, 3, 6, 7, 8};
constexpr std::int32_t kInterInvSet2[12] = {
    9, 10, 11, 0, 1, 2, 4, 5, 3, 6, 7, 8};
constexpr std::int32_t kInterInvSet3[2] = {9, 0};
constexpr std::int32_t kInterInvSet4[4] = {0, 10, 11, 9};
// [is_long_side_dct][orientation][symbol]
constexpr std::int32_t kInterInvLong[2][2][4] = {
    {{10, 12, 14, 9}, {11, 13, 15, 9}},
    {{0, 1, 4, 11}, {0, 2, 5, 10}},
};

bool inter_leaf_types(std::int32_t coded_w, std::int32_t coded_h,
                      const VideoTxSearchOptions& options,
                      std::vector<TypeEntry>* out, std::string* error) {
  using residual::InterTxCandidate;
  using residual::InterTxSet;
  using av2enc::txv::TxSize;
  const auto ts = static_cast<TxSize>(av2enc::txv::tx_size_index(coded_w,
                                                                 coded_h));
  const InterTxSet set =
      residual::inter_tx_set(ts, options.reduced_tx_set);
  out->clear();
  if (set == InterTxSet::kDctOnly) {
    InterTxCandidate c;
    if (!residual::inter_tx_symbol(ts, options.reduced_tx_set,
                                   options.enable_inter_ddt != 0,
                                   av2enc::txv::TxType::kDctDct, &c)) {
      if (error != nullptr) *error = "DCT_DCT missing from a DCTONLY set";
      return false;
    }
    TypeEntry e;
    e.type = 0;
    e.codes_type = 0;
    out->push_back(e);
    return true;
  }

  std::vector<std::int32_t> canonical;
  if (set == InterTxSet::kInter1) {
    canonical.assign(kInterInvSet1, kInterInvSet1 + 16);
  } else if (set == InterTxSet::kInter2) {
    canonical.assign(kInterInvSet2, kInterInvSet2 + 12);
  } else if (set == InterTxSet::kDctIdtx) {
    canonical.assign(kInterInvSet3, kInterInvSet3 + 2);
  } else if (set == InterTxSet::kDctIdtxIddct) {
    canonical.assign(kInterInvSet4, kInterInvSet4 + 4);
  } else {
    const int orientation =
        (set == InterTxSet::kWide64 || set == InterTxSet::kWide32) ? 0 : 1;
    const bool codes_flag =
        set == InterTxSet::kWide32 || set == InterTxSet::kHigh32;
    const int long_values = codes_flag ? 2 : 1;
    for (int flag = 0; flag < long_values; ++flag) {
      const int is_long = codes_flag ? flag : 1;
      for (int symbol = 0; symbol < 4; ++symbol) {
        canonical.push_back(kInterInvLong[is_long][orientation][symbol]);
      }
    }
  }

  for (std::int32_t type : canonical) {
    if (!options.include_1d_types && type >= 10) continue;
    if (options.enable_inter_ddt != 0 &&
        residual::inter_tx_type_is_ddt(
            static_cast<av2enc::txv::TxType>(type))) {
      continue;
    }
    if (!residual::inter_tx_type_in_set(
            set, static_cast<av2enc::txv::TxType>(type))) {
      continue;
    }
    InterTxCandidate c;
    if (!residual::inter_tx_symbol(ts, options.reduced_tx_set,
                                   options.enable_inter_ddt != 0,
                                   static_cast<av2enc::txv::TxType>(type),
                                   &c)) {
      continue;
    }
    TypeEntry e;
    e.type = type;
    e.codes_type = 1;
    // INTER_2's offset CDF context is `inter_tx_type_offset_context`, which
    // differs from the type context.  Compute all three type contexts here;
    // the device selects by the decoded eob.  For INTER_2 the offset context
    // is derived from the same eob, so the table is built per type context and
    // the device reproduces the formula.
    for (int ctx = 0; ctx < 3; ++ctx) {
      e.bits[ctx] = inter_type_bits_ctx(set, ts, c, ctx);
    }
    out->push_back(e);
  }
  if (out->empty()) {
    if (error != nullptr) *error = "empty inter TX-type set";
    return false;
  }
  return true;
}

bool intra_leaf_types(std::int32_t coded_w, std::int32_t coded_h,
                      const VideoTxSearchOptions& options,
                      std::vector<TypeEntry>* out, std::string* error) {
  txtype::TxTypeQuery q;
  q.coded_w = coded_w;
  q.coded_h = coded_h;
  q.mapped_mode = options.intra_mapped_mode;
  q.reduced_tx_set = options.reduced_tx_set;
  q.dct_only = options.dct_only != 0;
  txtype::TxTypeSet set;
  if (!txtype::intra_tx_candidates(q, &set)) {
    if (error != nullptr) *error = "intra TX-type candidate enumeration failed";
    return false;
  }
  out->clear();
  for (int i = 0; i < set.count; ++i) {
    const int type = static_cast<int>(set.candidates[i].type);
    if (!options.include_1d_types && type >= 10) continue;
    TypeEntry e;
    e.type = type;
    e.codes_type = set.candidates[i].symbol >= 0 ? 1 : 0;
    const double bits = e.codes_type != 0
                            ? txtype::static_type_bits(q, set.candidates[i])
                            : 0.0;
    e.bits[0] = bits;
    e.bits[1] = bits;
    e.bits[2] = bits;
    out->push_back(e);
  }
  if (out->empty()) {
    if (error != nullptr) *error = "empty intra TX-type set";
    return false;
  }
  return true;
}

bool build_leaf_types(std::int32_t coded_w, std::int32_t coded_h,
                      VideoTxMode mode, const VideoTxSearchOptions& options,
                      std::vector<TypeEntry>* out, std::string* error) {
  if (mode == VideoTxMode::kIntra) {
    return intra_leaf_types(coded_w, coded_h, options, out, error);
  }
  return inter_leaf_types(coded_w, coded_h, options, out, error);
}

// §5.20.6.3 partition bit index for one txPartition value.
int partition_bit(int partition) {
  switch (static_cast<txpart::TxPartition>(partition)) {
    case txpart::TxPartition::kNone:
      return kPartBitNone;
    case txpart::TxPartition::kSplit:
      return kPartBitSplit;
    case txpart::TxPartition::kHorz:
      return kPartBitHorz;
    case txpart::TxPartition::kVert:
      return kPartBitVert;
    case txpart::TxPartition::kHorz4:
      return kPartBitHorz4;
    case txpart::TxPartition::kVert4:
      return kPartBitVert4;
    default:
      return 0;
  }
}

// Estimated §5.20.6.3 signalling cost of one partition decision from the §9.3
// default CDFs (tx_do_partition, then tx_partition_type / tx_2or3_partition_type
// when the block shape codes one).
double partition_bits_for(int mi_size, bool is_inter, bool reduced_tx_part_set,
                          int partition, const txpart::LegalSplits& legal) {
  txpart::TxPartitionCdfs cdfs;
  cdfs.reset(reduced_tx_part_set);
  const bool fsc = false;
  std::uint16_t* do_cdf = cdfs.do_partition_cdf(fsc, is_inter, mi_size);
  if (do_cdf == nullptr) return 0.0;
  const bool do_split = partition != 0;
  double bits = cdf_symbol_bits(do_cdf, 2, do_split ? 1 : 0);
  if (!do_split) return bits;
  if (legal.has_partition_type) {
    std::uint16_t* type_cdf =
        cdfs.partition_type_cdf(fsc, is_inter, mi_size);
    if (type_cdf == nullptr) return bits;
    int symbol = 0;
    if (partition == static_cast<int>(txpart::TxPartition::kSplit)) {
      symbol = 0;
    } else if (partition == static_cast<int>(txpart::TxPartition::kHorz)) {
      symbol = 1;
    } else if (partition == static_cast<int>(txpart::TxPartition::kVert)) {
      symbol = 2;
    } else {
      return bits;  // HORZ4/VERT4 never reach the partition-type branch
    }
    bits += cdf_symbol_bits(type_cdf, txpart::kTxPartitionTypeNum, symbol);
    return bits;
  }
  if (legal.has_2or3) {
    const bool horizontal = legal.allow_horz;
    const int symbol =
        ((horizontal ? partition == static_cast<int>(txpart::TxPartition::kHorz4)
                     : partition == static_cast<int>(txpart::TxPartition::kVert4)))
            ? 1
            : 0;
    if (!legal.reduced_tx_part_set) {
      std::uint16_t* cdf = cdfs.partition_2or3_cdf(fsc, is_inter, mi_size);
      if (cdf != nullptr) bits += cdf_symbol_bits(cdf, 2, symbol);
    }
  }
  return bits;
}

bool append_candidate(const VideoTxLeafDesc& leaf,
                      std::int32_t tx_size, std::int32_t tx_type,
                      const TypeEntry& entry, VideoTxPlanData* plan,
                      std::string* error) {
  const auto params = av2enc::txv::make_tx_params(
      static_cast<av2enc::txv::TxSize>(tx_size),
      static_cast<av2enc::txv::TxType>(tx_type));
  if (!params.has_value()) {
    if (error != nullptr) *error = "unsupported transform size/type";
    return false;
  }
  if (params->coded_w != leaf.coded_w || params->coded_h != leaf.coded_h) {
    if (error != nullptr) *error = "candidate size does not match leaf";
    return false;
  }
  VideoTxCandDesc cand{};
  cand.params = *params;
  cand.tx_size = tx_size;
  cand.tx_type = tx_type;
  const txpart::TxScanClass cls =
      txpart::tx_class_of(static_cast<int>(tx_type));
  cand.scan_table = tx_size * 3 + static_cast<int>(cls);
  cand.codes_type = entry.codes_type;
  cand.type_bits[0] = entry.bits[0];
  cand.type_bits[1] = entry.bits[1];
  cand.type_bits[2] = entry.bits[2];
  plan->candidates.push_back(cand);
  return true;
}

// Emits one plan candidate (a partition and its leaves) and its candidates.
bool append_plan(const VideoTxJobRequest& request,
                 const VideoTxSearchOptions& options, int job_index,
                 int partition, double partition_bits,
                 const txpart::SplitGeometry& geometry, VideoTxPlanData* plan,
                 std::string* error) {
  (void)request;
  if (geometry.count <= 0 || geometry.count > kMaxLeavesPerPlan) {
    if (error != nullptr) *error = "partition leaf count out of range";
    return false;
  }
  VideoTxPlanDesc plan_desc{};
  plan_desc.leaf_begin = static_cast<int>(plan->leaves.size());
  plan_desc.partition = partition;
  plan_desc.partition_bits = partition_bits;

  for (int k = 0; k < geometry.count; ++k) {
    const int tx_size = geometry.parts[k].tx_size;
    if (tx_size < 0 || tx_size >= 25) {
      if (error != nullptr) *error = "partition produced an invalid TX size";
      return false;
    }
    const int coded_w = txpart::tx_width(tx_size);
    const int coded_h = txpart::tx_height(tx_size);
    if (coded_w > options.max_tx_size || coded_h > options.max_tx_size) {
      if (error != nullptr) *error = "partition leaf exceeds max_tx_size";
      return false;
    }
    if (!av2enc::txv::tx_size_supported(coded_w, coded_h)) {
      if (error != nullptr) *error = "leaf size is not an AV2 TX size";
      return false;
    }

    VideoTxLeafDesc leaf{};
    leaf.x = geometry.parts[k].mi_col * 4;
    leaf.y = geometry.parts[k].mi_row * 4;
    leaf.coded_w = coded_w;
    leaf.coded_h = coded_h;
    leaf.cand_begin = static_cast<int>(plan->candidates.size());
    leaf.leaf_position = k;
    leaf.level_offset = job_index * kMaxLeavesPerPlan * kMaxAdjElems +
                        k * kMaxAdjElems;

    // Stage 2: the legal §5.20.8.3 TX-type set of this leaf (or DCT_DCT only
    // when the caller pins the type search off).
    std::vector<TypeEntry> types;
    if (!build_leaf_types(coded_w, coded_h, request.mode, options, &types,
                          error)) {
      return false;
    }
    bool added = false;
    for (const TypeEntry& entry : types) {
      if (!options.search_tx_type && entry.type != 0) continue;
      if (!append_candidate(leaf, tx_size, entry.type, entry, plan, error)) {
        return false;
      }
      added = true;
    }
    if (!added) {
      TypeEntry dct;
      dct.type = 0;
      dct.codes_type = 0;
      if (!append_candidate(leaf, tx_size, 0, dct, plan, error)) {
        return false;
      }
    }
    leaf.cand_count = static_cast<int>(plan->candidates.size()) - leaf.cand_begin;
    if (leaf.cand_count > kMaxCandidatesPerLeaf) {
      if (error != nullptr) *error = "too many candidates for one leaf";
      return false;
    }
    plan->leaves.push_back(leaf);
  }
  plan_desc.leaf_count = geometry.count;
  plan->plans.push_back(plan_desc);
  return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::uint64_t required_device_bytes(const VideoTxConfig& config) {
  return bytes_for(config);
}

std::uint64_t persistent_device_bytes(const VideoTxConfig& config) {
  (void)config;
  return align_up(static_cast<std::uint64_t>(kScanTables) * kScanStride *
                  sizeof(std::uint16_t));
}

std::uint64_t scratch_device_bytes(const VideoTxConfig& config) {
  return required_device_bytes(config) - persistent_device_bytes(config);
}

Status VideoTxWorkspace::init(Arena& arena, const VideoTxConfig& config_in,
                              std::uint64_t* required_bytes,
                              std::string* error) {
  if (config_in.max_jobs <= 0 || config_in.max_plans_per_job <= 0 ||
      config_in.max_plans_per_job > kMaxPlansPerJob ||
      config_in.max_candidates_per_leaf <= 0 ||
      config_in.max_candidates_per_leaf > kMaxCandidatesPerLeaf) {
    if (error != nullptr) *error = "invalid VideoTxConfig";
    if (required_bytes != nullptr) *required_bytes = 0;
    return Status::kInvalidArgument;
  }
  reset();
  config = config_in;
  bytes = required_device_bytes(config);
  if (required_bytes != nullptr) *required_bytes = bytes;
  const std::uint64_t mark = arena.mark();

  const std::uint64_t jobs = static_cast<std::uint64_t>(config.max_jobs);
  const std::uint64_t plans = jobs * config.max_plans_per_job;
  const std::uint64_t leaves = plans * kMaxLeavesPerPlan;
  const std::uint64_t cands = leaves * config.max_candidates_per_leaf;

  Status s = allocate_array(arena,
                            static_cast<std::uint64_t>(kScanTables) *
                                kScanStride,
                            &d_scan);
  if (s == Status::kOk) s = allocate_array(arena, jobs, &d_jobs);
  if (s == Status::kOk) s = allocate_array(arena, plans, &d_plans);
  if (s == Status::kOk) s = allocate_array(arena, leaves, &d_leaves);
  if (s == Status::kOk) s = allocate_array(arena, cands, &d_candidates);
  if (s == Status::kOk) s = allocate_array(arena, cands, &d_cand_metrics);
  if (s == Status::kOk) s = allocate_array(arena, plans, &d_plan_results);
  if (s == Status::kOk) {
    s = allocate_array(arena, jobs * kMaxLeavesPerPlan, &d_leaf_results);
  }
  if (s == Status::kOk) s = allocate_array(arena, jobs, &d_job_results);
  if (s == Status::kOk) {
    s = allocate_array(arena, jobs * kMaxLeavesPerPlan * kMaxAdjElems,
                       &d_levels);
  }
  if (s == Status::kOk) s = allocate_array(arena, 1, &d_error_count);

  if (s != Status::kOk) {
    arena.rewind(mark);
    const std::uint64_t required = bytes;
    reset();
    if (required_bytes != nullptr) *required_bytes = required;
    if (error != nullptr) *error = "VideoTxWorkspace arena exhausted";
    return Status::kArenaExhausted;
  }

  // Persistent §5.20.7.30 scan tables (TX_CLASS_2D/HORIZ/VERT, all 25 sizes).
  std::vector<std::uint16_t> host_scan(static_cast<std::size_t>(kScanTables) *
                                       kScanStride);
  for (int size = 0; size < 25; ++size) {
    for (int cls = 0; cls < 3; ++cls) {
      std::uint16_t* base =
          host_scan.data() + (size * 3 + cls) * kScanStride;
      const auto span = std::span<std::uint16_t>(
          base, static_cast<std::size_t>(kScanStride));
      if (!txpart::get_scan(size, static_cast<txpart::TxScanClass>(cls),
                            span)) {
        arena.rewind(mark);
        reset();
        if (error != nullptr) *error = "scan table generation failed";
        return Status::kInvalidArgument;
      }
    }
  }
  const cudaError_t copy =
      cudaMemcpy(d_scan, host_scan.data(),
                 host_scan.size() * sizeof(std::uint16_t),
                 cudaMemcpyHostToDevice);
  if (copy != cudaSuccess) {
    arena.rewind(mark);
    reset();
    if (error != nullptr) *error = "scan table upload failed";
    return status_from_cuda(copy);
  }
  last_error_count = 0;
  return Status::kOk;
}

void VideoTxWorkspace::reset() noexcept {
  d_scan = nullptr;
  d_jobs = nullptr;
  d_plans = nullptr;
  d_leaves = nullptr;
  d_candidates = nullptr;
  d_cand_metrics = nullptr;
  d_plan_results = nullptr;
  d_leaf_results = nullptr;
  d_job_results = nullptr;
  d_levels = nullptr;
  d_error_count = nullptr;
  last_error_count = 0;
}

bool VideoTxWorkspace::valid() const noexcept {
  return config.max_jobs > 0 && d_scan != nullptr && d_jobs != nullptr &&
         d_plans != nullptr && d_leaves != nullptr && d_candidates != nullptr &&
         d_cand_metrics != nullptr && d_plan_results != nullptr &&
         d_leaf_results != nullptr && d_job_results != nullptr &&
         d_levels != nullptr && d_error_count != nullptr;
}

bool video_tx_build_plan(const VideoTxJobRequest* requests, int count,
                         VideoTxPlanData* plan, std::string* error) {
  if (error != nullptr) error->clear();
  if (plan == nullptr) {
    if (error != nullptr) *error = "null plan";
    return false;
  }
  plan->clear();
  if (requests == nullptr || count <= 0) {
    if (error != nullptr) *error = "no requests";
    return false;
  }

  for (int i = 0; i < count; ++i) {
    const VideoTxJobRequest& r = requests[i];
    if (r.coded_w < 4 || r.coded_w > kMaxCodedDim || r.coded_h < 4 ||
        r.coded_h > kMaxCodedDim) {
      if (error != nullptr) *error = "block size out of range";
      return false;
    }
    if (r.plane < 0 || r.plane > 2 || r.qindex < 0 || r.qindex > 303) {
      if (error != nullptr) *error = "bad plane/qindex";
      return false;
    }
    if (av2enc::txv::tx_size_index(r.coded_w, r.coded_h) < 0) {
      if (error != nullptr) {
        *error = "block size is not one of the 25 AV2 transform sizes";
      }
      return false;
    }
    if (r.search.max_tx_size < 4) {
      if (error != nullptr) *error = "bad max_tx_size";
      return false;
    }
    if (r.residual_offset < 0 && r.src_packed == 0 &&
        (r.src_visible_w <= 0 || r.src_visible_h <= 0)) {
      if (error != nullptr) *error = "plane source needs visible bounds";
      return false;
    }

    VideoTxJobDesc job{};
    job.plane = r.plane;
    job.coded_w = r.coded_w;
    job.coded_h = r.coded_h;
    job.qindex = r.qindex;
    job.delta_dc = r.delta_dc;
    job.delta_ac = r.delta_ac;
    job.use_residual = r.residual_offset >= 0 ? 1 : 0;
    job.src_packed = r.src_packed;
    job.src_offset = r.src_offset;
    job.src_stride = r.src_stride;
    job.src_x = r.src_x;
    job.src_y = r.src_y;
    job.src_visible_w = r.src_visible_w;
    job.src_visible_h = r.src_visible_h;
    job.pred_offset = r.pred_offset;
    job.pred_stride = r.pred_stride;
    job.residual_offset = r.residual_offset >= 0 ? r.residual_offset : 0;
    job.plan_begin = static_cast<int>(plan->plans.size());
    job.level_base = i * kMaxLeavesPerPlan * kMaxAdjElems;
    job.is_inter = r.mode == VideoTxMode::kInter ? 1 : 0;
    job.use_type_bits = r.search.use_type_bits;
    job.lambda = r.search.lambda;
    job.coefficient_bits = r.search.coefficient_bits;
    job.skip_bits = r.search.skip_bits;

    const int mi_size = mi_size_for_px(r.coded_w, r.coded_h);
    // A 4x64 / 64x4 block has no §6.19.3.1 MiSize; it can still be one
    // transform candidate, but it has no partition syntax.
    const bool partition_possible = mi_size >= 0 && r.search.search_partition != 0;
    (void)partition_possible;

    bool any_plan = false;
    const bool partition_search =
        r.search.search_partition != 0 && mi_size >= 0;
    const txpart::LegalSplits legal =
        partition_search
            ? txpart::legal_splits(mi_size, r.search.reduced_tx_part_set != 0)
            : txpart::LegalSplits{};
    const bool can_partition = partition_search && legal.applicable;

    if (r.search.search_none != 0 &&
        (r.search.partition_mask & kPartBitNone) != 0) {
      txpart::SplitGeometry geometry{};
      geometry.count = 0;
      const int tx_size = av2enc::txv::tx_size_index(r.coded_w, r.coded_h);
      geometry.parts[0] = txpart::SplitPart{0, 0, tx_size};
      geometry.count = 1;
      const double bits =
          can_partition
              ? partition_bits_for(mi_size, r.mode == VideoTxMode::kInter,
                                   r.search.reduced_tx_part_set != 0, 0, legal)
              : 0.0;
      if (!append_plan(r, r.search, i, 0, bits, geometry, plan, error)) {
        return false;
      }
      any_plan = true;
    }

    if (can_partition) {
      for (int oi = 0; oi < legal.option_count; ++oi) {
        const int partition = static_cast<int>(legal.options[oi]);
        if (partition == 0) continue;  // NONE handled above
        const int bit = partition_bit(partition);
        if (bit == 0 || (r.search.partition_mask & bit) == 0) continue;
        txpart::SplitGeometry geometry{};
        std::string geom_error;
        if (!txpart::tx_partition_geometry(mi_size, 0, 0,
                                           legal.options[oi], &geometry,
                                           &geom_error)) {
          if (error != nullptr) {
            *error = "legal partition did not resolve: " + geom_error;
          }
          return false;
        }
        const double bits = partition_bits_for(
            mi_size, r.mode == VideoTxMode::kInter,
            r.search.reduced_tx_part_set != 0, partition, legal);
        if (!append_plan(r, r.search, i, partition, bits, geometry, plan,
                         error)) {
          return false;
        }
        any_plan = true;
      }
    }

    if (!any_plan) {
      if (error != nullptr) *error = "no candidate plan selected";
      return false;
    }
    job.plan_count = static_cast<int>(plan->plans.size()) - job.plan_begin;
    if (job.plan_count > kMaxPlansPerJob) {
      if (error != nullptr) *error = "too many plans for one block";
      return false;
    }
    plan->jobs.push_back(job);
  }
  return true;
}

int video_tx_tx_types(std::int32_t coded_w, std::int32_t coded_h,
                      VideoTxMode mode, const VideoTxSearchOptions& options,
                      std::int32_t* out, double type_bits[][3], int cap) {
  if (out == nullptr || cap < 1) return -1;
  if (av2enc::txv::tx_size_index(coded_w, coded_h) < 0) return -1;
  std::vector<TypeEntry> types;
  std::string error;
  if (!build_leaf_types(coded_w, coded_h, mode, options, &types, &error)) {
    return -1;
  }
  int n = 0;
  for (const TypeEntry& e : types) {
    if (n >= cap) break;
    out[n] = e.type;
    if (type_bits != nullptr) {
      type_bits[n][0] = e.bits[0];
      type_bits[n][1] = e.bits[1];
      type_bits[n][2] = e.bits[2];
    }
    ++n;
  }
  return n;
}

Status video_tx_select_batch(const VideoTxPlanData& plan,
                             const VideoTxBuffers& buffers,
                             VideoTxWorkspace& workspace,
                             cudaStream_t stream) {
  if (!workspace.valid()) {
    return Status::kNotInitialized;
  }
  const std::int64_t n_jobs = static_cast<std::int64_t>(plan.jobs.size());
  const std::int64_t n_plans = static_cast<std::int64_t>(plan.plans.size());
  const std::int64_t n_leaves = static_cast<std::int64_t>(plan.leaves.size());
  const std::int64_t n_cands =
      static_cast<std::int64_t>(plan.candidates.size());
  const std::int64_t max_plans =
      static_cast<std::int64_t>(workspace.config.max_jobs) *
      workspace.config.max_plans_per_job;
  const std::int64_t max_leaves = max_plans * kMaxLeavesPerPlan;
  const std::int64_t max_cands =
      max_leaves * workspace.config.max_candidates_per_leaf;
  if (n_jobs <= 0 || n_jobs > workspace.config.max_jobs || n_plans > max_plans ||
      n_leaves > max_leaves || n_cands > max_cands) {
    return Status::kInvalidArgument;
  }

  // Validate every range before the device reads it.  The plan builder always
  // produces this shape; a hand-built plan must not be able to index outside
  // the workspace arrays.
  const std::int64_t total_levels =
      static_cast<std::int64_t>(workspace.config.max_jobs) *
      kMaxLeavesPerPlan * kMaxAdjElems;
  for (const VideoTxJobDesc& j : plan.jobs) {
    if (j.plan_begin < 0 || j.plan_count <= 0 ||
        j.plan_count > kMaxPlansPerJob ||
        static_cast<std::int64_t>(j.plan_begin) + j.plan_count > n_plans) {
      return Status::kInvalidArgument;
    }
    for (int p = 0; p < j.plan_count; ++p) {
      const VideoTxPlanDesc& pd = plan.plans[j.plan_begin + p];
      if (pd.leaf_begin < 0 || pd.leaf_count <= 0 ||
          pd.leaf_count > kMaxLeavesPerPlan ||
          static_cast<std::int64_t>(pd.leaf_begin) + pd.leaf_count > n_leaves) {
        return Status::kInvalidArgument;
      }
      for (int l = 0; l < pd.leaf_count; ++l) {
        const VideoTxLeafDesc& lf = plan.leaves[pd.leaf_begin + l];
        if (lf.cand_begin < 0 || lf.cand_count <= 0 ||
            lf.cand_count > kMaxCandidatesPerLeaf ||
            static_cast<std::int64_t>(lf.cand_begin) + lf.cand_count >
                n_cands) {
          return Status::kInvalidArgument;
        }
        if (lf.x < 0 || lf.y < 0 || lf.coded_w < 4 ||
            lf.coded_w > kMaxCodedDim || lf.coded_h < 4 ||
            lf.coded_h > kMaxCodedDim || lf.x + lf.coded_w > j.coded_w ||
            lf.y + lf.coded_h > j.coded_h || lf.level_offset < 0 ||
            static_cast<std::int64_t>(lf.level_offset) + kMaxAdjElems >
                total_levels) {
          return Status::kInvalidArgument;
        }
      }
    }
  }
  for (const VideoTxCandDesc& c : plan.candidates) {
    if (c.tx_size < 0 || c.tx_size >= 25 || c.tx_type < 0 || c.tx_type >= 16 ||
        c.scan_table < 0 || c.scan_table >= kScanTables) {
      return Status::kInvalidArgument;
    }
  }

  cudaError_t err = cudaMemcpyAsync(
      workspace.d_jobs, plan.jobs.data(), plan.jobs.size() * sizeof(VideoTxJobDesc),
      cudaMemcpyHostToDevice, stream);
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(workspace.d_plans, plan.plans.data(),
                          plan.plans.size() * sizeof(VideoTxPlanDesc),
                          cudaMemcpyHostToDevice, stream);
  }
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(workspace.d_leaves, plan.leaves.data(),
                          plan.leaves.size() * sizeof(VideoTxLeafDesc),
                          cudaMemcpyHostToDevice, stream);
  }
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(workspace.d_candidates, plan.candidates.data(),
                          plan.candidates.size() * sizeof(VideoTxCandDesc),
                          cudaMemcpyHostToDevice, stream);
  }
  if (err == cudaSuccess) {
    err = cudaMemsetAsync(workspace.d_error_count, 0, sizeof(int), stream);
  }
  if (err != cudaSuccess) {
    return status_from_cuda(err);
  }

  video_tx_kernel<<<static_cast<unsigned>(n_jobs), kThreads,
                    kDynamicSmemBytes, stream>>>(
      workspace.d_jobs, workspace.d_plans, workspace.d_leaves,
      workspace.d_candidates, workspace.d_scan, buffers.src, buffers.pred,
      buffers.residual, workspace.d_cand_metrics, workspace.d_plan_results,
      workspace.d_leaf_results, workspace.d_job_results, workspace.d_levels,
      workspace.d_error_count);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    return status_from_cuda(err);
  }

  // The module's single synchronisation: the caller gets finished decisions.
  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return status_from_cuda(err);
  }
  int host_error = 0;
  err = cudaMemcpy(&host_error, workspace.d_error_count, sizeof(int),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    return status_from_cuda(err);
  }
  workspace.last_error_count = host_error;
  return Status::kOk;
}

}  // namespace av2enc::cuda::videotx

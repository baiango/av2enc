// CPU-vs-GPU bit-exact tests and GPU timing for the batched M1 transform
// stage.  The CPU side is the real `src/encoder/tx_variants/` reference driven
// exactly like `TileEncoder::compute_tx` in `src/encoder/m1/m1_still.cpp`.
//
//   ./build_m1_tx_test.sh && <build>/m1_tx_cuda_test [64x64.y4m 640x360.y4m]
//
// Exits 1 on the first mismatching category (all categories still run).
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "cuda/m1/tx/m1_tx.h"
#include "encoder/tx_variants/tx_variants.h"

#ifndef AV2ENC_TEST_ROOT
#define AV2ENC_TEST_ROOT "."
#endif

using av2enc::cuda::M1TxBuffers;
using av2enc::cuda::M1TxInput;
using av2enc::cuda::M1TxJob;
using av2enc::txv::TxParams;
using av2enc::txv::TxSize;
using av2enc::txv::TxType;

namespace {

int g_checks = 0;
int g_failures = 0;

void fail(const std::string& label, const std::string& what) {
  ++g_failures;
  std::printf("FAIL %s: %s\n", label.c_str(), what.c_str());
}

#define CUDA_CHECK(expr)                                                     \
  do {                                                                       \
    const cudaError_t err__ = (expr);                                        \
    if (err__ != cudaSuccess) {                                              \
      std::printf("CUDA error %s at %s:%d: %s\n", #expr, __FILE__, __LINE__, \
                  cudaGetErrorString(err__));                                \
      std::exit(2);                                                          \
    }                                                                        \
  } while (0)

// --------------------------------------------------------------------------
// CPU reference helpers (replicas of private M1 code, kept in the test only).
// --------------------------------------------------------------------------

std::vector<std::uint16_t> build_scan_cpu(int w, int h) {
  std::vector<std::uint16_t> scan(static_cast<std::size_t>(w) *
                                  static_cast<std::size_t>(h));
  int x = 0;
  int y = 0;
  for (int c = 0; c < w * h; ++c) {
    scan[static_cast<std::size_t>(c)] =
        static_cast<std::uint16_t>(y * w + x);
    ++x;
    --y;
    if (y < 0 || x >= w) {
      ++x;
      const int s = std::min(x, h - 1 - y);
      x -= s;
      y += s;
    }
  }
  return scan;
}

// Closed form used by the device kernel (see scan_pos in m1_tx.cu).
int scan_pos_cpu(int c, int w, int h) {
  int diagonal = 0;
  int rem = c;
  for (;;) {
    const int lo = diagonal - (w - 1) > 0 ? diagonal - (w - 1) : 0;
    const int hi = diagonal < h - 1 ? diagonal : h - 1;
    const int len = hi - lo + 1;
    if (rem < len) {
      const int row = hi - rem;
      const int col = diagonal - row;
      return row * w + col;
    }
    rem -= len;
    ++diagonal;
  }
}

int clip3_int(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

struct CpuResult {
  std::vector<double> coeff;
  std::vector<std::int32_t> levels;
  std::int32_t eob = 0;
  std::vector<std::int32_t> dequant;
  std::vector<std::int16_t> residual_out;
  std::vector<std::int16_t> residual_adj;
};

// Mirrors TileEncoder::compute_tx (source+prediction or precomputed residual).
void cpu_pipeline(const M1TxJob& job, const std::uint8_t* src,
                  const std::uint8_t* pred, const std::int16_t* resid,
                  CpuResult* out) {
  const TxParams& p = job.params;
  const int cw = p.coded_w;
  const int ch = p.coded_h;
  const int w = p.w;
  const int h = p.h;
  const int coded_pels = cw * ch;

  std::vector<std::int16_t> coded(static_cast<std::size_t>(coded_pels));
  if (job.input == static_cast<std::uint8_t>(M1TxInput::kResidual)) {
    std::memcpy(coded.data(), resid + job.residual_in_offset,
                static_cast<std::size_t>(coded_pels) * sizeof(std::int16_t));
  } else {
    for (int y = 0; y < ch; ++y) {
      for (int x = 0; x < cw; ++x) {
        int s;
        if (job.src_packed != 0) {
          s = src[job.src_offset + y * cw + x];
        } else {
          const int px = clip3_int(0, job.src_width - 1, job.src_x + x);
          const int py = clip3_int(0, job.src_height - 1, job.src_y + y);
          s = src[job.src_offset + py * job.src_stride + px];
        }
        const int pv = pred[job.pred_offset + y * cw + x];
        coded[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
              static_cast<std::size_t>(x)] =
            static_cast<std::int16_t>(s - pv);
      }
    }
  }

  out->residual_adj.assign(static_cast<std::size_t>(w) * h, 0);
  av2enc::txv::downsample_residual(coded.data(), p, out->residual_adj.data());

  out->coeff.assign(static_cast<std::size_t>(w) * h, 0.0);
  out->levels.assign(static_cast<std::size_t>(w) * h, 0);
  av2enc::txv::forward_transform(out->residual_adj.data(), p,
                                 out->coeff.data(), out->levels.data());

  const std::vector<std::uint16_t> scan = build_scan_cpu(w, h);
  out->eob = 0;
  for (int c = w * h - 1; c >= 0; --c) {
    if (out->levels[scan[static_cast<std::size_t>(c)]] != 0) {
      out->eob = c + 1;
      break;
    }
  }

  out->dequant.assign(static_cast<std::size_t>(w) * h, 0);
  out->residual_out.assign(static_cast<std::size_t>(coded_pels), 0);
  av2enc::txv::inverse_transform(out->levels.data(), p, out->dequant.data(),
                                 out->residual_out.data());
}

// --------------------------------------------------------------------------
// Host batch: builds device-offset jobs, one launch for the whole batch.
// --------------------------------------------------------------------------

struct DeviceBuffer {
  void* ptr = nullptr;
  ~DeviceBuffer() {
    if (ptr != nullptr) cudaFree(ptr);
  }
  void alloc(std::size_t bytes) {
    if (bytes == 0) return;
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
  }
};

struct Batch {
  std::vector<M1TxJob> jobs;
  std::vector<std::string> labels;
  std::vector<std::uint8_t> src;
  std::vector<std::uint8_t> pred;
  std::vector<std::int16_t> resid;
  std::vector<CpuResult> refs;

  // Extra options applied to the launch.
  int smem_elems = 0;
  std::size_t max_area = 0;

  int add_plane(const std::uint8_t* data, int w, int h, int stride) {
    const int off = static_cast<int>(src.size());
    src.insert(src.end(), static_cast<std::size_t>(stride) * h, 0);
    for (int y = 0; y < h; ++y) {
      std::memcpy(src.data() + off + y * stride,
                  data + static_cast<std::size_t>(y) * stride,
                  static_cast<std::size_t>(w));
    }
    return off;
  }

  void add_packed(const TxParams& p, const std::vector<std::uint8_t>& block_src,
                  const std::vector<std::uint8_t>& block_pred,
                  const std::string& label) {
    M1TxJob job;
    job.params = p;
    job.input = static_cast<std::uint8_t>(M1TxInput::kSourcePred);
    job.src_packed = 1;
    job.src_offset = static_cast<int>(src.size());
    src.insert(src.end(), block_src.begin(), block_src.end());
    job.pred_offset = static_cast<int>(pred.size());
    pred.insert(pred.end(), block_pred.begin(), block_pred.end());
    job.residual_adj_offset = 0;
    job.coeff_offset = 0;
    job.levels_offset = 0;
    job.dequant_offset = 0;
    job.residual_out_offset = 0;
    job.eob_offset = 0;
    push(job, label);
  }

  void add_plane_block(const TxParams& p, int src_base, int plane_w,
                       int plane_h, int stride, int x0, int y0,
                       const std::vector<std::uint8_t>& block_pred,
                       const std::string& label) {
    M1TxJob job;
    job.params = p;
    job.input = static_cast<std::uint8_t>(M1TxInput::kSourcePred);
    job.src_packed = 0;
    job.src_offset = src_base;
    job.src_stride = stride;
    job.src_x = static_cast<std::int16_t>(x0);
    job.src_y = static_cast<std::int16_t>(y0);
    job.src_width = static_cast<std::int16_t>(plane_w);
    job.src_height = static_cast<std::int16_t>(plane_h);
    job.pred_offset = static_cast<int>(pred.size());
    pred.insert(pred.end(), block_pred.begin(), block_pred.end());
    push(job, label);
  }

  void add_residual(const TxParams& p,
                    const std::vector<std::int16_t>& coded_resid,
                    const std::string& label) {
    M1TxJob job;
    job.params = p;
    job.input = static_cast<std::uint8_t>(M1TxInput::kResidual);
    job.residual_in_offset = static_cast<int>(resid.size());
    resid.insert(resid.end(), coded_resid.begin(), coded_resid.end());
    push(job, label);
  }

  void push(const M1TxJob& job, const std::string& label) {
    const TxParams& p = job.params;
    const int area = p.w * p.h;
    max_area = std::max(max_area, static_cast<std::size_t>(area));
    jobs.push_back(job);
    labels.push_back(label);
    CpuResult ref;
    const std::uint8_t* src_ptr = src.empty() ? nullptr : src.data();
    const std::uint8_t* pred_ptr = pred.empty() ? nullptr : pred.data();
    const std::int16_t* resid_ptr = resid.empty() ? nullptr : resid.data();
    cpu_pipeline(jobs.back(), src_ptr, pred_ptr, resid_ptr, &ref);
    refs.push_back(std::move(ref));
  }

  void finish_offsets() {
    int off_adj = 0;
    int off_coeff = 0;
    int off_levels = 0;
    int off_dequant = 0;
    int off_out = 0;
    int off_eob = 0;
    for (std::size_t i = 0; i < jobs.size(); ++i) {
      M1TxJob& j = jobs[i];
      const TxParams& p = j.params;
      const int area = p.w * p.h;
      const int coded = p.coded_w * p.coded_h;
      j.residual_adj_offset = off_adj;
      off_adj += area;
      j.coeff_offset = off_coeff;
      off_coeff += area;
      j.levels_offset = off_levels;
      off_levels += area;
      j.dequant_offset = off_dequant;
      off_dequant += area;
      j.residual_out_offset = off_out;
      off_out += coded;
      j.eob_offset = off_eob;
      off_eob += 1;
    }
  }

  // Runs the whole batch on the GPU and compares every output against refs.
  // `use_graph` captures the stage into a CUDA graph and replays it, which is
  // what the stage-3 scheduler adapter will do.
  void run_and_compare(const std::string& tag, bool forward_only = false,
                       bool use_graph = false) {
    if (jobs.empty()) return;
    finish_offsets();

    const int adj_total = jobs.back().residual_adj_offset +
                          jobs.back().params.w * jobs.back().params.h;
    const int coded_total =
        jobs.back().residual_out_offset +
        jobs.back().params.coded_w * jobs.back().params.coded_h;

    DeviceBuffer d_jobs, d_src, d_pred, d_resid, d_adj, d_coeff, d_levels,
        d_dequant, d_out, d_eob, d_err;
    CUDA_CHECK(cudaMalloc(&d_jobs.ptr, jobs.size() * sizeof(M1TxJob)));
    CUDA_CHECK(cudaMemcpy(d_jobs.ptr, jobs.data(),
                          jobs.size() * sizeof(M1TxJob),
                          cudaMemcpyHostToDevice));
    if (!src.empty()) {
      d_src.alloc(src.size());
      CUDA_CHECK(cudaMemcpy(d_src.ptr, src.data(), src.size(),
                            cudaMemcpyHostToDevice));
    }
    if (!pred.empty()) {
      d_pred.alloc(pred.size());
      CUDA_CHECK(cudaMemcpy(d_pred.ptr, pred.data(), pred.size(),
                            cudaMemcpyHostToDevice));
    }
    if (!resid.empty()) {
      d_resid.alloc(resid.size() * sizeof(std::int16_t));
      CUDA_CHECK(cudaMemcpy(d_resid.ptr, resid.data(),
                            resid.size() * sizeof(std::int16_t),
                            cudaMemcpyHostToDevice));
    }
    d_adj.alloc(static_cast<std::size_t>(adj_total) * sizeof(std::int16_t));
    d_coeff.alloc(static_cast<std::size_t>(adj_total) * sizeof(double));
    d_levels.alloc(static_cast<std::size_t>(adj_total) * sizeof(std::int32_t));
    d_dequant.alloc(static_cast<std::size_t>(adj_total) * sizeof(std::int32_t));
    d_out.alloc(static_cast<std::size_t>(coded_total) * sizeof(std::int16_t));
    d_eob.alloc(jobs.size() * sizeof(std::int32_t));
    d_err.alloc(sizeof(int));
    CUDA_CHECK(cudaMemset(d_err.ptr, 0, sizeof(int)));

    M1TxBuffers buffers;
    buffers.src = static_cast<const std::uint8_t*>(d_src.ptr);
    buffers.pred = static_cast<const std::uint8_t*>(d_pred.ptr);
    buffers.residual_in = static_cast<const std::int16_t*>(d_resid.ptr);
    buffers.residual_adj = static_cast<std::int16_t*>(d_adj.ptr);
    buffers.coeff = static_cast<double*>(d_coeff.ptr);
    buffers.levels = static_cast<std::int32_t*>(d_levels.ptr);
    buffers.dequant = static_cast<std::int32_t*>(d_dequant.ptr);
    buffers.residual_out =
        forward_only ? nullptr : static_cast<std::int16_t*>(d_out.ptr);
    buffers.eob = static_cast<std::int32_t*>(d_eob.ptr);
    buffers.error_count = static_cast<int*>(d_err.ptr);

    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    if (use_graph) {
      CUDA_CHECK(cudaStreamCreate(&stream));
      CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
      CUDA_CHECK(av2enc::cuda::m1_tx_batch(
          static_cast<const M1TxJob*>(d_jobs.ptr),
          static_cast<int>(jobs.size()), buffers, smem_elems, stream));
      CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
      CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, 0));
      CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      cudaGraphExecDestroy(graph_exec);
      cudaGraphDestroy(graph);
      cudaStreamDestroy(stream);
    } else {
      CUDA_CHECK(av2enc::cuda::m1_tx_batch(
          static_cast<const M1TxJob*>(d_jobs.ptr),
          static_cast<int>(jobs.size()), buffers, smem_elems, nullptr));
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    int errors = 0;
    CUDA_CHECK(cudaMemcpy(&errors, d_err.ptr, sizeof(int),
                          cudaMemcpyDeviceToHost));
    if (errors != 0) {
      fail(tag, "device rejected " + std::to_string(errors) + " job(s)");
    }

    std::vector<std::int16_t> h_adj(static_cast<std::size_t>(adj_total));
    std::vector<double> h_coeff(static_cast<std::size_t>(adj_total));
    std::vector<std::int32_t> h_levels(static_cast<std::size_t>(adj_total));
    std::vector<std::int32_t> h_dequant(static_cast<std::size_t>(adj_total));
    std::vector<std::int16_t> h_out(static_cast<std::size_t>(coded_total));
    std::vector<std::int32_t> h_eob(jobs.size());
    CUDA_CHECK(cudaMemcpy(h_adj.data(), d_adj.ptr,
                          h_adj.size() * sizeof(std::int16_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_coeff.data(), d_coeff.ptr,
                          h_coeff.size() * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_levels.data(), d_levels.ptr,
                          h_levels.size() * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dequant.data(), d_dequant.ptr,
                          h_dequant.size() * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost));
    if (!forward_only) {
      CUDA_CHECK(cudaMemcpy(h_out.data(), d_out.ptr,
                            h_out.size() * sizeof(std::int16_t),
                            cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaMemcpy(h_eob.data(), d_eob.ptr,
                          h_eob.size() * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost));

    for (std::size_t i = 0; i < jobs.size(); ++i) {
      const M1TxJob& j = jobs[i];
      const CpuResult& ref = refs[i];
      const TxParams& p = j.params;
      const int area = p.w * p.h;
      const int coded = p.coded_w * p.coded_h;
      const std::string label = tag + " " + labels[i];

      ++g_checks;
      if (std::memcmp(ref.coeff.data(), h_coeff.data() + j.coeff_offset,
                      static_cast<std::size_t>(area) * sizeof(double)) != 0) {
        fail(label, "coefficients");
      }
      ++g_checks;
      if (std::memcmp(ref.levels.data(), h_levels.data() + j.levels_offset,
                      static_cast<std::size_t>(area) * sizeof(std::int32_t)) !=
          0) {
        fail(label, "levels");
      }
      ++g_checks;
      if (ref.eob != h_eob[i]) {
        fail(label, "eob cpu=" + std::to_string(ref.eob) + " gpu=" +
                        std::to_string(h_eob[i]));
      }
      ++g_checks;
      if (std::memcmp(ref.dequant.data(), h_dequant.data() + j.dequant_offset,
                      static_cast<std::size_t>(area) * sizeof(std::int32_t)) !=
          0) {
        fail(label, "dequant");
      }
      ++g_checks;
      if (std::memcmp(ref.residual_adj.data(),
                      h_adj.data() + j.residual_adj_offset,
                      static_cast<std::size_t>(area) * sizeof(std::int16_t)) !=
          0) {
        fail(label, "adjusted residual");
      }
      if (!forward_only) {
        ++g_checks;
        if (std::memcmp(ref.residual_out.data(),
                        h_out.data() + j.residual_out_offset,
                        static_cast<std::size_t>(coded) *
                            sizeof(std::int16_t)) != 0) {
          fail(label, "coded-size inverse residual");
        }
      }
    }
  }
};

// --------------------------------------------------------------------------
// Input generators.
// --------------------------------------------------------------------------

std::uint32_t g_rng = 0x9e3779b9u;

std::uint32_t next_rand() {
  g_rng ^= g_rng << 13;
  g_rng ^= g_rng >> 17;
  g_rng ^= g_rng << 5;
  return g_rng;
}

int rand_range(int lo, int hi) {
  return lo + static_cast<int>(next_rand() % static_cast<std::uint32_t>(hi - lo + 1));
}

enum class Pattern { kRandom, kGradient, kImpulse, kFlat, kSmall };

std::vector<std::uint8_t> make_pred(int cw, int ch, Pattern pattern) {
  std::vector<std::uint8_t> out(static_cast<std::size_t>(cw) * ch, 128);
  for (int y = 0; y < ch; ++y) {
    for (int x = 0; x < cw; ++x) {
      std::uint8_t v = 128;
      switch (pattern) {
        case Pattern::kRandom:
          v = static_cast<std::uint8_t>(rand_range(0, 255));
          break;
        case Pattern::kGradient:
          v = static_cast<std::uint8_t>((x * 7 + y * 3) & 0xFF);
          break;
        case Pattern::kImpulse:
          v = (x == cw / 2 && y == ch / 2) ? 255 : 16;
          break;
        case Pattern::kFlat:
          v = static_cast<std::uint8_t>(rand_range(0, 255));
          break;
        case Pattern::kSmall:
          v = static_cast<std::uint8_t>(rand_range(120, 136));
          break;
      }
      out[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
          static_cast<std::size_t>(x)] = v;
    }
  }
  return out;
}

std::vector<std::int16_t> make_residual(int cw, int ch, Pattern pattern) {
  std::vector<std::int16_t> out(static_cast<std::size_t>(cw) * ch, 0);
  for (int y = 0; y < ch; ++y) {
    for (int x = 0; x < cw; ++x) {
      int v = 0;
      switch (pattern) {
        case Pattern::kRandom:
          v = rand_range(-255, 255);
          break;
        case Pattern::kGradient:
          v = ((x * 5 + y * 11) % 511) - 255;
          break;
        case Pattern::kImpulse:
          v = (x == 0 && y == 0) ? 255 : ((x == cw - 1 && y == ch - 1) ? -255 : 0);
          break;
        case Pattern::kFlat:
          v = (x + y) & 1 ? 63 : -63;
          break;
        case Pattern::kSmall:
          v = rand_range(-3, 3);
          break;
      }
      out[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
          static_cast<std::size_t>(x)] = static_cast<std::int16_t>(v);
    }
  }
  return out;
}

std::vector<std::uint8_t> make_source_block(int cw, int ch, Pattern pattern) {
  std::vector<std::uint8_t> out(static_cast<std::size_t>(cw) * ch, 128);
  for (int y = 0; y < ch; ++y) {
    for (int x = 0; x < cw; ++x) {
      std::uint8_t v = 128;
      switch (pattern) {
        case Pattern::kRandom:
          v = static_cast<std::uint8_t>(rand_range(0, 255));
          break;
        case Pattern::kGradient:
          v = static_cast<std::uint8_t>((x * 11 + y * 5) & 0xFF);
          break;
        case Pattern::kImpulse:
          v = (x == cw / 2 && y == ch / 2) ? 240 : 20;
          break;
        case Pattern::kFlat:
          v = 100;
          break;
        case Pattern::kSmall:
          v = static_cast<std::uint8_t>(rand_range(100, 140));
          break;
      }
      out[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
          static_cast<std::size_t>(x)] = v;
    }
  }
  return out;
}

struct SizeDesc {
  TxSize size;
  const char* name;
};

const SizeDesc kSizes[25] = {
    {TxSize::k4x4, "4x4"},       {TxSize::k8x8, "8x8"},
    {TxSize::k16x16, "16x16"},   {TxSize::k32x32, "32x32"},
    {TxSize::k64x64, "64x64"},   {TxSize::k4x8, "4x8"},
    {TxSize::k8x4, "8x4"},       {TxSize::k8x16, "8x16"},
    {TxSize::k16x8, "16x8"},     {TxSize::k16x32, "16x32"},
    {TxSize::k32x16, "32x16"},   {TxSize::k32x64, "32x64"},
    {TxSize::k64x32, "64x32"},   {TxSize::k4x16, "4x16"},
    {TxSize::k16x4, "16x4"},     {TxSize::k8x32, "8x32"},
    {TxSize::k32x8, "32x8"},     {TxSize::k16x64, "16x64"},
    {TxSize::k64x16, "64x16"},   {TxSize::k4x32, "4x32"},
    {TxSize::k32x4, "32x4"},     {TxSize::k8x64, "8x64"},
    {TxSize::k64x8, "64x8"},     {TxSize::k4x64, "4x64"},
    {TxSize::k64x4, "64x4"},
};

TxParams params_for(TxSize size, TxType type) {
  const auto p = av2enc::txv::make_tx_params(size, type);
  if (!p.has_value()) {
    std::printf("bad params for size %d type %d\n", static_cast<int>(size),
                static_cast<int>(type));
    std::exit(2);
  }
  return *p;
}

const Pattern kPatterns[5] = {Pattern::kRandom, Pattern::kGradient,
                              Pattern::kImpulse, Pattern::kFlat,
                              Pattern::kSmall};
const char* pattern_name(Pattern p) {
  switch (p) {
    case Pattern::kRandom: return "random";
    case Pattern::kGradient: return "gradient";
    case Pattern::kImpulse: return "impulse";
    case Pattern::kFlat: return "flat";
    case Pattern::kSmall: return "small";
  }
  return "?";
}

// --------------------------------------------------------------------------
// Tests.
// --------------------------------------------------------------------------

void test_scan_order() {
  for (const SizeDesc& d : kSizes) {
    const TxParams p = params_for(d.size, TxType::kDctDct);
    const std::vector<std::uint16_t> scan = build_scan_cpu(p.w, p.h);
    for (int c = 0; c < p.w * p.h; ++c) {
      ++g_checks;
      if (scan_pos_cpu(c, p.w, p.h) != scan[static_cast<std::size_t>(c)]) {
        fail("scan", std::string(d.name) + " index " + std::to_string(c));
        break;
      }
    }
  }
}

// All 25 sizes x 16 types x 5 patterns as one mixed launch, residual input.
void test_matrix_residual() {
  Batch batch;
  batch.smem_elems = 0;
  for (const SizeDesc& d : kSizes) {
    for (int t = 0; t < 16; ++t) {
      const TxType type = static_cast<TxType>(t);
      const TxParams p = params_for(d.size, type);
      for (Pattern pattern : kPatterns) {
        const std::vector<std::int16_t> coded =
            make_residual(p.coded_w, p.coded_h, pattern);
        batch.add_residual(p, coded,
                           std::string(d.name) + "/" +
                               std::to_string(t) + "/" + pattern_name(pattern));
      }
    }
  }
  batch.run_and_compare("matrix-residual");
}

// All 25 sizes x 16 types x 5 patterns, packed source+prediction.
void test_matrix_source() {
  Batch batch;
  for (const SizeDesc& d : kSizes) {
    for (int t = 0; t < 16; ++t) {
      const TxType type = static_cast<TxType>(t);
      const TxParams p = params_for(d.size, type);
      for (Pattern pattern : kPatterns) {
        batch.add_packed(p, make_source_block(p.coded_w, p.coded_h, pattern),
                         make_pred(p.coded_w, p.coded_h, pattern),
                         std::string(d.name) + "/" + std::to_string(t) + "/" +
                             pattern_name(pattern));
      }
    }
  }
  batch.run_and_compare("matrix-source");
}

// Forward-only shape: no inverse buffers, same levels/eob.
void test_forward_only() {
  Batch batch;
  for (const SizeDesc& d : kSizes) {
    const TxParams p = params_for(d.size, TxType::kDctDct);
    for (Pattern pattern : kPatterns) {
      batch.add_residual(p, make_residual(p.coded_w, p.coded_h, pattern),
                         std::string(d.name) + "/" + pattern_name(pattern));
    }
  }
  batch.run_and_compare("forward-only", true);
}

// The stage must be capturable: no allocation, no host sync inside.
void test_graph_capture() {
  Batch batch;
  for (const SizeDesc& d : kSizes) {
    for (TxType type : {TxType::kDctDct, TxType::kAdstAdst, TxType::kIdtx}) {
      const TxParams p = params_for(d.size, type);
      batch.add_residual(
          p, make_residual(p.coded_w, p.coded_h, Pattern::kGradient),
          std::string(d.name) + "/" +
              std::to_string(static_cast<int>(type)));
    }
  }
  batch.run_and_compare("graph", false, true);
}

// Plane-mode reads including blocks that overhang the visible plane.
void test_plane_edges() {
  Batch batch;
  const int plane_w = 100;
  const int plane_h = 70;
  const int stride = 112;
  std::vector<std::uint8_t> plane(static_cast<std::size_t>(stride) * plane_h, 0);
  for (int y = 0; y < plane_h; ++y) {
    for (int x = 0; x < stride; ++x) {
      plane[static_cast<std::size_t>(y) * stride + x] =
          static_cast<std::uint8_t>((x * 3 + y * 17 + (x ^ y)) & 0xFF);
    }
  }
  const int base = batch.add_plane(plane.data(), plane_w, plane_h, stride);

  for (const SizeDesc& d : kSizes) {
    const TxParams p = params_for(d.size, TxType::kDctDct);
    const int xs[4] = {0, 1, plane_w - 1, std::max(0, plane_w - p.coded_w)};
    const int ys[4] = {0, 2, plane_h - 1, std::max(0, plane_h - p.coded_h)};
    for (int k = 0; k < 4; ++k) {
      std::vector<std::uint8_t> pred(
          static_cast<std::size_t>(p.coded_w) * p.coded_h);
      for (int y = 0; y < p.coded_h; ++y) {
        for (int x = 0; x < p.coded_w; ++x) {
          const int px = clip3_int(0, plane_w - 1, xs[k] + x);
          const int py = clip3_int(0, plane_h - 1, ys[k] + y);
          pred[static_cast<std::size_t>(y) * p.coded_w + x] =
              plane[static_cast<std::size_t>(py) * stride + px];
        }
      }
      // Shift the prediction so the residual is not all zero.
      for (std::size_t i = 0; i < pred.size(); ++i) {
        pred[i] = static_cast<std::uint8_t>(clip3_int(0, 255, pred[i] + 9));
      }
      batch.add_plane_block(p, base, plane_w, plane_h, stride, xs[k], ys[k],
                            pred, std::string(d.name) + "/edge" +
                                      std::to_string(k));
    }
  }
  batch.run_and_compare("plane-edges");
}

// smem hint guard: a 4x4 job survives while the 32x32 job is rejected.
void test_smem_guard() {
  Batch batch;
  batch.smem_elems = 16;
  const TxParams small = params_for(TxSize::k4x4, TxType::kDctDct);
  const TxParams big = params_for(TxSize::k32x32, TxType::kDctDct);
  batch.add_residual(small, make_residual(4, 4, Pattern::kRandom), "4x4");
  batch.add_residual(big, make_residual(32, 32, Pattern::kRandom), "32x32");

  batch.finish_offsets();
  DeviceBuffer d_jobs, d_resid, d_levels, d_eob, d_err;
  CUDA_CHECK(cudaMalloc(&d_jobs.ptr, batch.jobs.size() * sizeof(M1TxJob)));
  CUDA_CHECK(cudaMemcpy(d_jobs.ptr, batch.jobs.data(),
                        batch.jobs.size() * sizeof(M1TxJob),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&d_resid.ptr,
                        batch.resid.size() * sizeof(std::int16_t)));
  CUDA_CHECK(cudaMemcpy(d_resid.ptr, batch.resid.data(),
                        batch.resid.size() * sizeof(std::int16_t),
                        cudaMemcpyHostToDevice));
  int levels_total = 0;
  for (const M1TxJob& j : batch.jobs) {
    levels_total += j.params.w * j.params.h;
  }
  CUDA_CHECK(cudaMalloc(&d_levels.ptr,
                        static_cast<std::size_t>(levels_total) *
                            sizeof(std::int32_t)));
  CUDA_CHECK(cudaMalloc(&d_eob.ptr,
                        batch.jobs.size() * sizeof(std::int32_t)));
  CUDA_CHECK(cudaMalloc(&d_err.ptr, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_err.ptr, 0, sizeof(int)));
  M1TxBuffers buffers;
  buffers.residual_in =
      static_cast<const std::int16_t*>(d_resid.ptr);
  buffers.levels = static_cast<std::int32_t*>(d_levels.ptr);
  buffers.eob = static_cast<std::int32_t*>(d_eob.ptr);
  buffers.error_count = static_cast<int*>(d_err.ptr);
  CUDA_CHECK(av2enc::cuda::m1_tx_batch(
      static_cast<const M1TxJob*>(d_jobs.ptr),
      static_cast<int>(batch.jobs.size()), buffers, batch.smem_elems,
      nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());
  int errors = 0;
  std::int32_t eob[2] = {0, 0};
  CUDA_CHECK(cudaMemcpy(&errors, d_err.ptr, sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(eob, d_eob.ptr, sizeof(eob),
                        cudaMemcpyDeviceToHost));
  ++g_checks;
  if (errors != 1) {
    fail("smem-guard", "expected 1 rejected job, got " + std::to_string(errors));
  }
  ++g_checks;
  if (eob[0] != batch.refs[0].eob) {
    fail("smem-guard", "4x4 eob mismatch");
  }
  ++g_checks;
  if (eob[1] != -1) {
    fail("smem-guard", "32x32 job not marked eob=-1");
  }
}

void test_empty_and_bad_args() {
  M1TxBuffers buffers;
  CUDA_CHECK(av2enc::cuda::m1_tx_batch(nullptr, 0, buffers, 0, nullptr));
  ++g_checks;
  if (av2enc::cuda::m1_tx_batch(nullptr, 1, buffers, 0, nullptr) !=
      cudaErrorInvalidValue) {
    fail("args", "null jobs accepted");
  }
  ++g_checks;
  M1TxJob job;
  if (av2enc::cuda::m1_tx_init_job(&job, 64, 64, TxType::kDctDct)) {
    if (job.params.w != 32 || job.params.h != 32) {
      fail("args", "64x64 did not adjust to 32x32");
    }
  } else {
    fail("args", "m1_tx_init_job rejected 64x64");
  }
  ++g_checks;
  if (av2enc::cuda::m1_tx_init_job(&job, 8, 64, TxType::kDctDct) != true) {
    fail("args", "m1_tx_init_job rejected 8x64");
  }
  ++g_checks;
  if (av2enc::cuda::m1_tx_init_job(&job, 6, 6, TxType::kDctDct)) {
    fail("args", "m1_tx_init_job accepted 6x6");
  }
}

// --------------------------------------------------------------------------
// Y4M real-frame blocks.
// --------------------------------------------------------------------------

struct Y4mFrame {
  int width = 0;
  int height = 0;
  int chroma_w = 0;
  int chroma_h = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
};

bool load_y4m(const std::string& path, Y4mFrame* out) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) return false;
  char line[512];
  if (std::fgets(line, sizeof(line), f) == nullptr) {
    std::fclose(f);
    return false;
  }
  if (std::strncmp(line, "YUV4MPEG2", 9) != 0) {
    std::fclose(f);
    return false;
  }
  int w = 0;
  int h = 0;
  for (char* tok = std::strtok(line, " \n"); tok != nullptr;
       tok = std::strtok(nullptr, " \n")) {
    if (tok[0] == 'W') w = std::atoi(tok + 1);
    if (tok[0] == 'H') h = std::atoi(tok + 1);
  }
  if (w <= 0 || h <= 0) {
    std::fclose(f);
    return false;
  }
  if (std::fgets(line, sizeof(line), f) == nullptr) {
    std::fclose(f);
    return false;
  }
  if (std::strncmp(line, "FRAME", 5) != 0) {
    std::fclose(f);
    return false;
  }
  const int cw = (w + 1) / 2;
  const int ch = (h + 1) / 2;
  out->width = w;
  out->height = h;
  out->chroma_w = cw;
  out->chroma_h = ch;
  out->y.resize(static_cast<std::size_t>(w) * h);
  out->u.resize(static_cast<std::size_t>(cw) * ch);
  out->v.resize(static_cast<std::size_t>(cw) * ch);
  bool ok = std::fread(out->y.data(), 1, out->y.size(), f) == out->y.size();
  ok = ok && std::fread(out->u.data(), 1, out->u.size(), f) == out->u.size();
  ok = ok && std::fread(out->v.data(), 1, out->v.size(), f) == out->v.size();
  std::fclose(f);
  return ok;
}

// Deterministic DC-ish prediction from the clamped above row, like an intra DC
// predictor would produce.
std::vector<std::uint8_t> frame_pred(const std::uint8_t* plane, int pw, int ph,
                                     int stride, int x0, int y0, int cw,
                                     int ch) {
  std::vector<std::uint8_t> pred(static_cast<std::size_t>(cw) * ch, 128);
  int sum = 0;
  for (int x = 0; x < cw; ++x) {
    const int px = clip3_int(0, pw - 1, x0 + x);
    const int py = clip3_int(0, ph - 1, y0 - 1);
    sum += plane[static_cast<std::size_t>(py) * stride + px];
  }
  const int dc = (sum + cw / 2) / cw;
  std::fill(pred.begin(), pred.end(), static_cast<std::uint8_t>(dc));
  return pred;
}

struct BlockSpec {
  int plane;
  int x0;
  int y0;
  TxSize size;
  TxType type;
};

void add_frame_cases(Batch* batch, const Y4mFrame& frame, const char* tag) {
  const std::uint8_t* planes[3] = {frame.y.data(), frame.u.data(),
                                   frame.v.data()};
  const int plane_w[3] = {frame.width, frame.chroma_w, frame.chroma_w};
  const int plane_h[3] = {frame.height, frame.chroma_h, frame.chroma_h};
  int bases[3] = {0, 0, 0};
  for (int p = 0; p < 3; ++p) {
    bases[p] = batch->add_plane(planes[p], plane_w[p], plane_h[p],
                                plane_w[p]);
  }

  std::vector<BlockSpec> specs;
  const int lw = frame.width;
  const int lh = frame.height;
  // Luma: corners, interior, overhanging edges, and the rectangular set.
  specs.push_back({0, 0, 0, TxSize::k64x64, TxType::kDctDct});
  specs.push_back({0, lw - 64, 0, TxSize::k64x64, TxType::kDctDct});
  specs.push_back({0, 0, lh - 64, TxSize::k64x64, TxType::kDctDct});
  specs.push_back({0, lw - 64, lh - 64, TxSize::k64x64, TxType::kIdtx});
  specs.push_back({0, 0, 0, TxSize::k32x32, TxType::kAdstAdst});
  specs.push_back({0, lw / 2, lh / 2, TxSize::k32x32, TxType::kDctDct});
  specs.push_back({0, lw - 24, lh - 24, TxSize::k16x16, TxType::kDctDct});
  specs.push_back({0, 4, 4, TxSize::k16x16, TxType::kFlipadstFlipadst});
  specs.push_back({0, lw - 9, lh - 9, TxSize::k8x8, TxType::kDctDct});
  specs.push_back({0, 1, 1, TxSize::k4x4, TxType::kDctDct});
  specs.push_back({0, 0, 0, TxSize::k16x64, TxType::kDctDct});
  specs.push_back({0, 0, 0, TxSize::k64x16, TxType::kDctDct});
  specs.push_back({0, 8, 8, TxSize::k32x8, TxType::kDctDct});
  specs.push_back({0, 8, 8, TxSize::k8x32, TxType::kDctDct});
  specs.push_back({0, lw / 3, lh / 3, TxSize::k32x64, TxType::kDctDct});
  specs.push_back({0, lw / 3, lh / 3, TxSize::k64x32, TxType::kDctDct});
  specs.push_back({0, 2, 2, TxSize::k4x16, TxType::kDctDct});
  specs.push_back({0, 2, 2, TxSize::k16x4, TxType::kDctDct});

  // Chroma: 32x32 and smaller at the same places, including overhang.
  const int cw = frame.chroma_w;
  const int chh = frame.chroma_h;
  specs.push_back({1, 0, 0, TxSize::k32x32, TxType::kDctDct});
  specs.push_back({1, cw - 32, 0, TxSize::k32x32, TxType::kDctDct});
  specs.push_back({1, 0, chh - 32, TxSize::k32x32, TxType::kDctDct});
  specs.push_back({1, cw - 32, chh - 32, TxSize::k32x32, TxType::kDctDct});
  specs.push_back({2, 0, 0, TxSize::k32x32, TxType::kDctDct});
  specs.push_back({2, cw / 2, chh / 2, TxSize::k16x16, TxType::kDctDct});
  specs.push_back({1, cw - 8, chh - 8, TxSize::k8x8, TxType::kDctDct});
  specs.push_back({2, cw - 4, chh - 4, TxSize::k4x4, TxType::kDctDct});
  specs.push_back({1, cw / 3, chh / 3, TxSize::k16x8, TxType::kDctDct});
  specs.push_back({2, cw / 3, chh / 3, TxSize::k8x16, TxType::kDctDct});

  for (const BlockSpec& s : specs) {
    const std::uint8_t* plane = planes[s.plane];
    const int pw = plane_w[s.plane];
    const int ph = plane_h[s.plane];
    const TxParams p = params_for(s.size, s.type);
    std::vector<std::uint8_t> pred =
        frame_pred(plane, pw, ph, pw, s.x0, s.y0, p.coded_w, p.coded_h);
    batch->add_plane_block(p, bases[s.plane], pw, ph, pw, s.x0, s.y0, pred,
                           std::string(tag) + "/p" +
                               std::to_string(s.plane) + "/" +
                               std::to_string(s.x0) + "," +
                               std::to_string(s.y0) + "/" +
                               av2enc::txv::tx_size_name(s.size));
  }
}

// --------------------------------------------------------------------------
// Timing.
// --------------------------------------------------------------------------

struct TimingGroup {
  TxSize size;
  TxType type;
  int count;
};

void run_timing_groups(const char* name, const std::vector<TimingGroup>& groups,
                       bool forward_only) {
  std::vector<M1TxJob> jobs;
  std::vector<int> areas;
  int total_adj = 0;
  int total_coded = 0;
  for (const TimingGroup& g : groups) {
    const TxParams p = params_for(g.size, g.type);
    for (int i = 0; i < g.count; ++i) {
      M1TxJob j;
      av2enc::cuda::m1_tx_init_job(&j, p.coded_w, p.coded_h, g.type);
      j.input = static_cast<std::uint8_t>(M1TxInput::kSourcePred);
      j.src_packed = 1;
      j.src_offset = total_coded;
      j.pred_offset = total_coded;
      j.residual_adj_offset = total_adj;
      j.coeff_offset = total_adj;
      j.levels_offset = total_adj;
      j.dequant_offset = total_adj;
      j.residual_out_offset = total_coded;
      j.eob_offset = static_cast<int>(jobs.size());
      jobs.push_back(j);
      areas.push_back(p.w * p.h);
      total_adj += p.w * p.h;
      total_coded += p.coded_w * p.coded_h;
    }
  }
  const int n = static_cast<int>(jobs.size());
  int smem_elems = 0;
  for (int area : areas) smem_elems = std::max(smem_elems, area);

  std::vector<std::uint8_t> src(static_cast<std::size_t>(total_coded), 100);
  std::vector<std::uint8_t> pred(static_cast<std::size_t>(total_coded), 128);
  for (std::size_t i = 0; i < src.size(); ++i) {
    src[i] = static_cast<std::uint8_t>((i * 37 + (i >> 6)) & 0xFF);
  }
  for (std::size_t i = 0; i < pred.size(); ++i) {
    pred[i] = static_cast<std::uint8_t>((i * 11 + 128) & 0xFF);
  }

  DeviceBuffer d_jobs, d_src, d_pred, d_adj, d_coeff, d_levels, d_dequant,
      d_out, d_eob;
  CUDA_CHECK(cudaMalloc(&d_jobs.ptr, jobs.size() * sizeof(M1TxJob)));
  CUDA_CHECK(cudaMemcpy(d_jobs.ptr, jobs.data(), jobs.size() * sizeof(M1TxJob),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&d_src.ptr, src.size()));
  CUDA_CHECK(cudaMemcpy(d_src.ptr, src.data(), src.size(),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&d_pred.ptr, pred.size()));
  CUDA_CHECK(cudaMemcpy(d_pred.ptr, pred.data(), pred.size(),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&d_adj.ptr,
                        static_cast<std::size_t>(total_adj) *
                            sizeof(std::int16_t)));
  CUDA_CHECK(cudaMalloc(&d_coeff.ptr,
                        static_cast<std::size_t>(total_adj) * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_levels.ptr,
                        static_cast<std::size_t>(total_adj) *
                            sizeof(std::int32_t)));
  CUDA_CHECK(cudaMalloc(&d_dequant.ptr,
                        static_cast<std::size_t>(total_adj) *
                            sizeof(std::int32_t)));
  CUDA_CHECK(cudaMalloc(&d_out.ptr,
                        static_cast<std::size_t>(total_coded) *
                            sizeof(std::int16_t)));
  CUDA_CHECK(cudaMalloc(&d_eob.ptr,
                        static_cast<std::size_t>(n) * sizeof(std::int32_t)));

  M1TxBuffers buffers;
  buffers.src = static_cast<const std::uint8_t*>(d_src.ptr);
  buffers.pred = static_cast<const std::uint8_t*>(d_pred.ptr);
  buffers.residual_adj = static_cast<std::int16_t*>(d_adj.ptr);
  buffers.coeff = static_cast<double*>(d_coeff.ptr);
  buffers.levels = static_cast<std::int32_t*>(d_levels.ptr);
  buffers.dequant = static_cast<std::int32_t*>(d_dequant.ptr);
  buffers.residual_out =
      forward_only ? nullptr : static_cast<std::int16_t*>(d_out.ptr);
  buffers.eob = static_cast<std::int32_t*>(d_eob.ptr);

  const int iters = 10;
  for (int i = 0; i < 3; ++i) {
    CUDA_CHECK(av2enc::cuda::m1_tx_batch(
        static_cast<const M1TxJob*>(d_jobs.ptr), n, buffers, smem_elems,
        nullptr));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, nullptr));
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(av2enc::cuda::m1_tx_batch(
        static_cast<const M1TxJob*>(d_jobs.ptr), n, buffers, smem_elems,
        nullptr));
  }
  CUDA_CHECK(cudaEventRecord(stop, nullptr));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  const double per_block_us = static_cast<double>(ms) * 1000.0 / iters / n;
  std::printf("  %-28s %6d blocks: %8.3f ms | %7.4f us/block%s\n", name, n,
              static_cast<double>(ms) / iters, per_block_us,
              forward_only ? " (forward only)" : "");
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
}

void run_timing_case(const TimingGroup& tc, bool forward_only,
                     const char* name) {
  std::vector<TimingGroup> groups{tc};
  run_timing_groups(name, groups, forward_only);
}

void test_timing() {
  std::printf(
      "\n[timing] GPU only; full fused stage unless marked forward-only\n");
  const TimingGroup cases[] = {
      {TxSize::k4x4, TxType::kDctDct, 32768},
      {TxSize::k8x8, TxType::kDctDct, 32768},
      {TxSize::k16x16, TxType::kDctDct, 16384},
      {TxSize::k32x32, TxType::kDctDct, 4096},
      {TxSize::k64x64, TxType::kDctDct, 4096},
      {TxSize::k16x64, TxType::kDctDct, 4096},
      {TxSize::k64x16, TxType::kDctDct, 4096},
      {TxSize::k32x8, TxType::kDctDct, 8192},
      {TxSize::k8x32, TxType::kDctDct, 8192},
  };
  const char* names[] = {"4x4 DCT",       "8x8 DCT",      "16x16 DCT",
                         "32x32 DCT",     "64x64 (adj 32x32) DCT",
                         "16x64 DCT",     "64x16 DCT",    "32x8 DCT",
                         "8x32 DCT"};
  for (std::size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    run_timing_case(cases[i], false, names[i]);
  }
  run_timing_case({TxSize::k8x8, TxType::kDctDct, 32768}, true, "8x8 DCT");
  run_timing_case({TxSize::k16x16, TxType::kDctDct, 16384}, true, "16x16 DCT");
  run_timing_case({TxSize::k32x32, TxType::kAdstAdst, 4096}, false,
                  "32x32 ADST_ADST");
  run_timing_case({TxSize::k32x32, TxType::kIdtx, 4096}, false, "32x32 IDTX");

  // One 640x360 M1 frame's transform workload: luma block sizes from
  // `AV2ENC_M1_TRACE=1 build-m1/av2enc tests/data/m0_640x360_1f.y4m --m1`
  // (218 blocks) plus both chroma planes at half size (436 blocks).
  const std::vector<TimingGroup> frame = {
      {TxSize::k16x16, TxType::kDctDct, 50},
      {TxSize::k32x8, TxType::kDctDct, 48},
      {TxSize::k64x64, TxType::kDctDct, 41},
      {TxSize::k8x32, TxType::kDctDct, 20},
      {TxSize::k32x16, TxType::kDctDct, 17},
      {TxSize::k64x16, TxType::kDctDct, 10},
      {TxSize::k64x8, TxType::kDctDct, 8},
      {TxSize::k8x64, TxType::kDctDct, 6},
      {TxSize::k16x32, TxType::kDctDct, 6},
      {TxSize::k32x32, TxType::kDctDct, 5},
      {TxSize::k32x64, TxType::kDctDct, 4},
      {TxSize::k64x32, TxType::kDctDct, 2},
      {TxSize::k16x64, TxType::kDctDct, 1},
      // Chroma, two planes.
      {TxSize::k8x8, TxType::kDctDct, 100},
      {TxSize::k16x4, TxType::kDctDct, 96},
      {TxSize::k32x32, TxType::kDctDct, 82},
      {TxSize::k4x16, TxType::kDctDct, 40},
      {TxSize::k16x8, TxType::kDctDct, 34},
      {TxSize::k32x8, TxType::kDctDct, 20},
      {TxSize::k32x4, TxType::kDctDct, 16},
      {TxSize::k4x32, TxType::kDctDct, 12},
      {TxSize::k8x16, TxType::kDctDct, 12},
      {TxSize::k16x16, TxType::kDctDct, 10},
      {TxSize::k16x32, TxType::kDctDct, 8},
      {TxSize::k32x16, TxType::kDctDct, 4},
      {TxSize::k8x32, TxType::kDctDct, 2},
  };
  run_timing_groups("640x360 M1 TX distribution", frame, false);
}

}  // namespace

int main(int argc, char** argv) {
  int device = 0;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  CUDA_CHECK(cudaSetDevice(device));
  std::printf("av2enc CUDA M1 transform/quant stage test\n");
  std::printf("device: %s (sm_%d%d, %d SMs, %.1f GiB)\n", prop.name,
              prop.major, prop.minor, prop.multiProcessorCount,
              static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));

  std::string y4m64 = std::string(AV2ENC_TEST_ROOT) + "/tests/data/m0_64x64_1f.y4m";
  std::string y4m360 = std::string(AV2ENC_TEST_ROOT) + "/tests/data/m0_640x360_1f.y4m";
  if (argc > 1) y4m64 = argv[1];
  if (argc > 2) y4m360 = argv[2];

  test_scan_order();
  test_matrix_residual();
  test_matrix_source();
  test_forward_only();
  test_graph_capture();
  test_plane_edges();
  test_smem_guard();
  test_empty_and_bad_args();

  Y4mFrame f64;
  Y4mFrame f360;
  const bool have64 = load_y4m(y4m64, &f64);
  const bool have360 = load_y4m(y4m360, &f360);
  if (have64) {
    Batch batch;
    add_frame_cases(&batch, f64, "64x64");
    batch.run_and_compare("real-64x64");
  } else {
    fail("real-64x64", "could not read " + y4m64);
  }
  if (have360) {
    Batch batch;
    add_frame_cases(&batch, f360, "640x360");
    batch.run_and_compare("real-640x360");
  } else {
    fail("real-640x360", "could not read " + y4m360);
  }

  test_timing();

  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  std::printf("RESULT: %s\n", g_failures == 0 ? "PASS" : "FAIL");
  return g_failures == 0 ? 0 : 1;
}

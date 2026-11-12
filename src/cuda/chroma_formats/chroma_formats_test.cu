// Real-GPU test for the CUDA chroma-format component.
//
// Runs the arena-backed kernels for every format (4:2:0, 4:2:2, 4:4:4,
// 4:0:0), every reachable chroma residual size, every plane and both bit
// depths (8/10), and compares each stage bit-exactly with the CPU references:
//
//   * av2enc::chroma::make_frame_shape / chroma_mi_size / make_tx_shape
//   * av2enc::chroma::forward_dct / quantize_coefficients / dequantize_block /
//     inverse_dct                              (8-bit)
//   * av2enc::hbd::dequantize / inverse_block  (10-bit)
//   * the prediction / residual / reconstruction transcription in
//     chroma_formats_reference.h (built on chroma::approx_divide/round2)
//
// It also checks the arena contract:
//   * required_device_bytes() == persistent + largest scratch,
//   * init grows the arena by exactly the persistent bytes,
//   * two back-to-back launches with mark()/rewind() leave used() flat and
//     peak_used() unchanged on the second launch,
//   * a one-byte-short arena returns kArenaTooSmall with the required size and
//     leaves the bump pointer untouched.
//
// Memory: every device buffer comes from av2enc::cuda::Arena.  There is no
// cudaMalloc / cudaFree / cudaMallocHost in this file or in the component.
//
// Build/run: ./build_gpu_test.sh   (sm_86 default, needs an NVIDIA GPU)
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "cuda/arena.h"
#include "cuda/chroma_formats/chroma_formats.h"
#include "chroma_formats_reference.h"

#include "cuda/m1/hbd/hbd_cpu.h"
#include "encoder/format/chroma/chroma_format.h"
#include "encoder/format/chroma/chroma_residual.h"
#include "encoder/tx_variants/tx_variants.h"

namespace {

using av2enc::chroma::BlockSize;
using av2enc::chroma::ChromaFormat;
using av2enc::cuda::Arena;
using av2enc::cuda::chroma::ChromaStatus;
using av2enc::cuda::chroma::Config;
using av2enc::cuda::chroma::Format;
using av2enc::cuda::chroma::FrameGeom;
using av2enc::cuda::chroma::ResidualSize;
using av2enc::cuda::chroma::TxShape;
using av2enc::cuda::chroma::Workspace;
using av2enc::cuda::chroma::testref::dc_predict;
using av2enc::cuda::chroma::testref::make_residual;
using av2enc::cuda::chroma::testref::reconstruct;

long long g_checks = 0;
int g_failures = 0;

void expect(bool condition, const char* what, const char* detail,
            long long got = 0, long long want = 0) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    if (g_failures <= 30) {
      std::printf("FAIL: %s (%s got=%lld want=%lld)\n", what, detail, got,
                  want);
    }
  }
}

#define CHECK(cond) expect((cond), #cond, "", 0, 0)
#define CHECK_EQ(a, b)                                                     \
  do {                                                                     \
    const long long va_ = static_cast<long long>(a);                       \
    const long long vb_ = static_cast<long long>(b);                       \
    expect(va_ == vb_, #a " == " #b, "", va_, vb_);                        \
  } while (0)

bool cuda_ok(cudaError_t error) { return error == cudaSuccess; }
ChromaFormat cpu_format(Format format) {
  return static_cast<ChromaFormat>(static_cast<std::uint8_t>(format));
}

template <typename T>
bool upload(const std::vector<T>& host, T* device) {
  if (host.empty() || device == nullptr) return false;
  return cuda_ok(cudaMemcpy(device, host.data(), host.size() * sizeof(T),
                            cudaMemcpyHostToDevice));
}

template <typename T>
bool download(const T* device, std::vector<T>& host) {
  if (host.empty() || device == nullptr) return false;
  return cuda_ok(cudaMemcpy(host.data(), device, host.size() * sizeof(T),
                            cudaMemcpyDeviceToHost));
}

std::uint32_t g_rng = 0x6a09e667u;
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

// ---------------------------------------------------------------------------
// One full block pipeline case on the GPU
// ---------------------------------------------------------------------------

void run_case(Workspace& workspace, int bit_depth, int plane,
              int nominal_w, int nominal_h, int x0, int y0) {
  const FrameGeom& geometry = workspace.geometry();
  const auto& plane_geom = geometry.planes[plane];
  const int stride = plane_geom.padded_w;
  const int sample_count = stride * plane_geom.padded_h;
  const int max_x = stride - 1;
  const int max_y = plane_geom.padded_h - 1;
  const int have_left = x0 > 0 ? 1 : 0;
  const int have_above = y0 > 0 ? 1 : 0;
  const int sample_max = (1 << bit_depth) - 1;

  TxShape shape;
  CHECK(av2enc::cuda::chroma::make_tx_shape(nominal_w, nominal_h, &shape));
  av2enc::chroma::TxShape cpu_shape;
  CHECK(av2enc::chroma::make_tx_shape(nominal_w, nominal_h, cpu_shape));
  const int elems = shape.adj_w * shape.adj_h;

  std::vector<std::uint16_t> src(static_cast<std::size_t>(sample_count));
  std::vector<std::uint16_t> recon(static_cast<std::size_t>(sample_count));
  for (auto& value : src) {
    value = static_cast<std::uint16_t>(random_range(0, sample_max));
  }
  for (auto& value : recon) {
    value = static_cast<std::uint16_t>(random_range(0, sample_max));
  }
  CHECK(upload(src, workspace.src_plane(plane)));
  CHECK(upload(recon, workspace.recon_plane(plane)));

  // --- prediction -------------------------------------------------------
  av2enc::cuda::chroma::PredictJob predict_job;
  predict_job.recon = workspace.recon_plane(plane);
  predict_job.stride = stride;
  predict_job.max_x = plane_geom.coded_w - 1;
  predict_job.max_y = plane_geom.coded_h - 1;
  predict_job.x0 = x0;
  predict_job.y0 = y0;
  predict_job.nominal_w = nominal_w;
  predict_job.nominal_h = nominal_h;
  predict_job.have_left = have_left;
  predict_job.have_above = have_above;
  predict_job.bit_depth = bit_depth;
  predict_job.pred = workspace.pred_block(0);
  std::string error;
  CHECK_EQ(static_cast<int>(workspace.predict(&predict_job, 1, nullptr,
                                              &error).code),
           static_cast<int>(ChromaStatus::Code::kOk));
  std::vector<std::uint16_t> device_pred(
      static_cast<std::size_t>(nominal_w) * nominal_h, 0);
  CHECK(download(workspace.pred_block(0), device_pred));
  const int cpu_pred = dc_predict(recon, stride, plane_geom.coded_w,
                                  plane_geom.coded_h, x0, y0, nominal_w,
                                  nominal_h, have_left, have_above, bit_depth);
  for (std::size_t i = 0; i < device_pred.size(); ++i) {
    CHECK_EQ(device_pred[i], cpu_pred);
  }

  // --- forward residual -------------------------------------------------
  std::vector<std::int16_t> cpu_residual;
  make_residual(src, stride, max_x, max_y, x0, y0, cpu_pred, shape,
                cpu_residual);

  av2enc::cuda::chroma::ResidualJob residual_job;
  residual_job.src = workspace.src_plane(plane);
  residual_job.src_stride = stride;
  residual_job.max_x = max_x;
  residual_job.max_y = max_y;
  residual_job.x0 = x0;
  residual_job.y0 = y0;
  residual_job.pred = cpu_pred;
  residual_job.bit_depth = bit_depth;
  residual_job.shape = shape;
  residual_job.residual_adj = workspace.residual_adj(0);
  CHECK_EQ(static_cast<int>(workspace.make_residuals(&residual_job, 1,
                                                     nullptr, &error).code),
           static_cast<int>(ChromaStatus::Code::kOk));
  std::vector<std::int16_t> device_residual(
      static_cast<std::size_t>(elems), 0);
  CHECK(download(workspace.residual_adj(0), device_residual));
  for (int i = 0; i < elems; ++i) {
    CHECK_EQ(device_residual[static_cast<std::size_t>(i)],
             cpu_residual[static_cast<std::size_t>(i)]);
  }

  // --- forward transform + quantisation --------------------------------
  std::vector<double> cpu_coeff(static_cast<std::size_t>(elems), 0.0);
  av2enc::chroma::forward_dct(cpu_residual.data(), cpu_shape.adj_w,
                              cpu_shape.adj_h, cpu_shape.odd_rect,
                              cpu_coeff.data());
  std::vector<std::int32_t> cpu_levels(static_cast<std::size_t>(elems), 0);
  av2enc::chroma::quantize_coefficients(
      cpu_coeff.data(), cpu_shape.seg_eob,
      av2enc::chroma::quant_factor_for(cpu_shape), cpu_levels.data());

  av2enc::cuda::chroma::ForwardJob forward_job;
  forward_job.shape = shape;
  forward_job.residual_adj = workspace.residual_adj(0);
  forward_job.coeff = workspace.coeff(0);
  forward_job.levels = workspace.levels(0);
  CHECK_EQ(static_cast<int>(workspace.forward(&forward_job, 1, nullptr,
                                              &error).code),
           static_cast<int>(ChromaStatus::Code::kOk));
  std::vector<double> device_coeff(static_cast<std::size_t>(elems), 0.0);
  std::vector<std::int32_t> device_levels(static_cast<std::size_t>(elems), 0);
  CHECK(download(workspace.coeff(0), device_coeff));
  CHECK(download(workspace.levels(0), device_levels));
  for (int i = 0; i < elems; ++i) {
    CHECK(std::memcmp(&device_coeff[static_cast<std::size_t>(i)],
                      &cpu_coeff[static_cast<std::size_t>(i)],
                      sizeof(double)) == 0);
    CHECK_EQ(device_levels[static_cast<std::size_t>(i)],
             cpu_levels[static_cast<std::size_t>(i)]);
  }

  // --- dequantisation ---------------------------------------------------
  const auto params_opt = av2enc::txv::make_tx_params(
      nominal_w, nominal_h, av2enc::txv::TxType::kDctDct);
  CHECK(params_opt.has_value());
  const av2enc::txv::TxParams params = *params_opt;
  std::vector<std::int32_t> cpu_dequant(static_cast<std::size_t>(elems), 0);
  if (bit_depth == 8) {
    av2enc::chroma::dequantize_block(cpu_levels.data(), elems,
                                     cpu_shape.dq_denom, cpu_dequant.data());
  } else {
    av2enc::hbd::dequantize(cpu_levels.data(), params, cpu_dequant.data(),
                            bit_depth);
  }
  av2enc::cuda::chroma::DequantJob dequant_job;
  dequant_job.shape = shape;
  dequant_job.bit_depth = bit_depth;
  dequant_job.levels = workspace.levels(0);
  dequant_job.dequant = workspace.dequant(0);
  CHECK_EQ(static_cast<int>(workspace.dequantize(&dequant_job, 1, nullptr,
                                                 &error).code),
           static_cast<int>(ChromaStatus::Code::kOk));
  std::vector<std::int32_t> device_dequant(
      static_cast<std::size_t>(elems), 0);
  CHECK(download(workspace.dequant(0), device_dequant));
  for (int i = 0; i < elems; ++i) {
    CHECK_EQ(device_dequant[static_cast<std::size_t>(i)],
             cpu_dequant[static_cast<std::size_t>(i)]);
  }

  // --- inverse ----------------------------------------------------------
  std::vector<std::int32_t> cpu_back(static_cast<std::size_t>(elems), 0);
  if (bit_depth == 8) {
    std::vector<int> back(static_cast<std::size_t>(elems), 0);
    av2enc::chroma::inverse_dct(cpu_dequant.data(), cpu_shape, back.data());
    for (int i = 0; i < elems; ++i) {
      cpu_back[static_cast<std::size_t>(i)] =
          back[static_cast<std::size_t>(i)];
    }
  } else {
    std::vector<std::int16_t> coded(
        static_cast<std::size_t>(params.coded_w) * params.coded_h, 0);
    av2enc::hbd::inverse_block(cpu_dequant.data(), params, coded.data(),
                               bit_depth);
    for (int j = 0; j < shape.adj_h; ++j) {
      for (int i = 0; i < shape.adj_w; ++i) {
        const std::size_t coded_index =
            static_cast<std::size_t>(j * shape.upscale_y) *
                static_cast<std::size_t>(params.coded_w) +
            static_cast<std::size_t>(i * shape.upscale_x);
        cpu_back[static_cast<std::size_t>(j) * shape.adj_w + i] =
            coded[coded_index];
      }
    }
  }
  av2enc::cuda::chroma::InverseJob inverse_job;
  inverse_job.shape = shape;
  inverse_job.bit_depth = bit_depth;
  inverse_job.dequant = workspace.dequant(0);
  inverse_job.residual_adj = workspace.residual_adj(0);
  CHECK_EQ(static_cast<int>(workspace.inverse(&inverse_job, 1, nullptr,
                                              &error).code),
           static_cast<int>(ChromaStatus::Code::kOk));
  std::vector<std::int16_t> device_back(static_cast<std::size_t>(elems), 0);
  CHECK(download(workspace.residual_adj(0), device_back));
  for (int i = 0; i < elems; ++i) {
    CHECK_EQ(device_back[static_cast<std::size_t>(i)],
             cpu_back[static_cast<std::size_t>(i)]);
  }

  // --- reconstruction ---------------------------------------------------
  std::vector<std::uint16_t> cpu_recon = recon;
  reconstruct(cpu_back, cpu_pred, shape, bit_depth, cpu_recon, stride, max_x,
              max_y, x0, y0);
  av2enc::cuda::chroma::ReconstructJob recon_job;
  recon_job.shape = shape;
  recon_job.bit_depth = bit_depth;
  recon_job.pred = cpu_pred;
  recon_job.residual_adj = workspace.residual_adj(0);
  recon_job.dst = workspace.recon_plane(plane);
  recon_job.dst_stride = stride;
  recon_job.max_x = max_x;
  recon_job.max_y = max_y;
  recon_job.x0 = x0;
  recon_job.y0 = y0;
  CHECK_EQ(static_cast<int>(workspace.reconstruct(&recon_job, 1, nullptr,
                                                  &error).code),
           static_cast<int>(ChromaStatus::Code::kOk));
  std::vector<std::uint16_t> device_recon(
      static_cast<std::size_t>(sample_count), 0);
  CHECK(download(workspace.recon_plane(plane), device_recon));
  for (std::size_t i = 0; i < cpu_recon.size(); ++i) {
    CHECK_EQ(device_recon[i], cpu_recon[i]);
  }
}

// ---------------------------------------------------------------------------
// All 25 transform sizes x both depths, independent of format geometry
// ---------------------------------------------------------------------------

void test_transforms(Workspace& workspace) {
  const int dims[] = {4, 8, 16, 32, 64};
  for (int bit_depth : {8, 10}) {
    for (int width : dims) {
      for (int height : dims) {
        TxShape shape;
        CHECK(av2enc::cuda::chroma::make_tx_shape(width, height, &shape));
        av2enc::chroma::TxShape cpu_shape;
        CHECK(av2enc::chroma::make_tx_shape(width, height, cpu_shape));
        const int elems = shape.adj_w * shape.adj_h;
        const int span = bit_depth == 8 ? 300 : 1200;

        std::vector<std::int16_t> residual(static_cast<std::size_t>(elems));
        for (auto& value : residual) {
          value = static_cast<std::int16_t>(random_range(-span, span));
        }
        CHECK(upload(residual, workspace.residual_adj(0)));

        av2enc::cuda::chroma::ForwardJob forward_job;
        forward_job.shape = shape;
        forward_job.residual_adj = workspace.residual_adj(0);
        forward_job.coeff = workspace.coeff(0);
        forward_job.levels = workspace.levels(0);
        std::string error;
        CHECK_EQ(static_cast<int>(workspace.forward(&forward_job, 1, nullptr,
                                                    &error).code),
                 static_cast<int>(ChromaStatus::Code::kOk));

        std::vector<double> cpu_coeff(static_cast<std::size_t>(elems), 0.0);
        av2enc::chroma::forward_dct(residual.data(), cpu_shape.adj_w,
                                    cpu_shape.adj_h, cpu_shape.odd_rect,
                                    cpu_coeff.data());
        std::vector<std::int32_t> cpu_levels(static_cast<std::size_t>(elems),
                                             0);
        av2enc::chroma::quantize_coefficients(
            cpu_coeff.data(), cpu_shape.seg_eob,
            av2enc::chroma::quant_factor_for(cpu_shape), cpu_levels.data());
        std::vector<double> device_coeff(static_cast<std::size_t>(elems), 0.0);
        std::vector<std::int32_t> device_levels(
            static_cast<std::size_t>(elems), 0);
        CHECK(download(workspace.coeff(0), device_coeff));
        CHECK(download(workspace.levels(0), device_levels));
        for (int i = 0; i < elems; ++i) {
          CHECK(std::memcmp(&device_coeff[static_cast<std::size_t>(i)],
                            &cpu_coeff[static_cast<std::size_t>(i)],
                            sizeof(double)) == 0);
          CHECK_EQ(device_levels[static_cast<std::size_t>(i)],
                   cpu_levels[static_cast<std::size_t>(i)]);
        }

        const auto params_opt = av2enc::txv::make_tx_params(
            width, height, av2enc::txv::TxType::kDctDct);
        CHECK(params_opt.has_value());
        const av2enc::txv::TxParams params = *params_opt;
        std::vector<std::int32_t> cpu_dequant(static_cast<std::size_t>(elems),
                                              0);
        if (bit_depth == 8) {
          av2enc::chroma::dequantize_block(cpu_levels.data(), elems,
                                           cpu_shape.dq_denom,
                                           cpu_dequant.data());
        } else {
          av2enc::hbd::dequantize(cpu_levels.data(), params,
                                  cpu_dequant.data(), bit_depth);
        }
        av2enc::cuda::chroma::DequantJob dequant_job;
        dequant_job.shape = shape;
        dequant_job.bit_depth = bit_depth;
        dequant_job.levels = workspace.levels(0);
        dequant_job.dequant = workspace.dequant(0);
        CHECK_EQ(static_cast<int>(workspace.dequantize(&dequant_job, 1,
                                                       nullptr, &error).code),
                 static_cast<int>(ChromaStatus::Code::kOk));
        std::vector<std::int32_t> device_dequant(
            static_cast<std::size_t>(elems), 0);
        CHECK(download(workspace.dequant(0), device_dequant));
        for (int i = 0; i < elems; ++i) {
          CHECK_EQ(device_dequant[static_cast<std::size_t>(i)],
                   cpu_dequant[static_cast<std::size_t>(i)]);
        }

        av2enc::cuda::chroma::InverseJob inverse_job;
        inverse_job.shape = shape;
        inverse_job.bit_depth = bit_depth;
        inverse_job.dequant = workspace.dequant(0);
        inverse_job.residual_adj = workspace.residual_adj(0);
        CHECK_EQ(static_cast<int>(workspace.inverse(&inverse_job, 1, nullptr,
                                                    &error).code),
                 static_cast<int>(ChromaStatus::Code::kOk));
        std::vector<std::int16_t> device_back(static_cast<std::size_t>(elems),
                                              0);
        CHECK(download(workspace.residual_adj(0), device_back));

        if (bit_depth == 8) {
          std::vector<int> cpu_back(static_cast<std::size_t>(elems), 0);
          av2enc::chroma::inverse_dct(cpu_dequant.data(), cpu_shape,
                                      cpu_back.data());
          for (int i = 0; i < elems; ++i) {
            CHECK_EQ(device_back[static_cast<std::size_t>(i)],
                     cpu_back[static_cast<std::size_t>(i)]);
          }
        } else {
          std::vector<std::int16_t> coded(
              static_cast<std::size_t>(params.coded_w) * params.coded_h, 0);
          av2enc::hbd::inverse_block(cpu_dequant.data(), params, coded.data(),
                                     bit_depth);
          for (int j = 0; j < shape.adj_h; ++j) {
            for (int i = 0; i < shape.adj_w; ++i) {
              const std::size_t coded_index =
                  static_cast<std::size_t>(j * shape.upscale_y) *
                      static_cast<std::size_t>(params.coded_w) +
                  static_cast<std::size_t>(i * shape.upscale_x);
              CHECK_EQ(
                  device_back[static_cast<std::size_t>(j) * shape.adj_w + i],
                  coded[coded_index]);
            }
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Every format / plane / residual size / depth on the GPU
// ---------------------------------------------------------------------------

void test_pipelines(Workspace& workspace) {
  const FrameGeom& geometry = workspace.geometry();
  const std::pair<int, int> luma_blocks[] = {
      {64, 64}, {32, 32}, {32, 16}, {32, 8}};
  for (int bit_depth : {8, 10}) {
    for (int plane = 0; plane < geometry.num_planes; ++plane) {
      for (const auto& block : luma_blocks) {
        int nominal_w = block.first;
        int nominal_h = block.second;
        if (plane > 0) {
          const ResidualSize residual =
              av2enc::cuda::chroma::chroma_residual_size(
                  block.first, block.second, geometry.format);
          CHECK(residual.valid);
          if (!residual.valid) continue;
          nominal_w = residual.width;
          nominal_h = residual.height;
        }
        const auto& plane_geom = geometry.planes[plane];
        if (nominal_w > plane_geom.padded_w ||
            nominal_h > plane_geom.padded_h) {
          continue;
        }
        const int max_x = std::min(4, plane_geom.coded_w - nominal_w);
        const int max_y = std::min(4, plane_geom.coded_h - nominal_h);
        run_case(workspace, bit_depth, plane, nominal_w, nominal_h, 0, 0);
        if (max_x > 0) {
          run_case(workspace, bit_depth, plane, nominal_w, nominal_h, max_x, 0);
        }
        if (max_y > 0) {
          run_case(workspace, bit_depth, plane, nominal_w, nominal_h, 0, max_y);
        }
        if (max_x > 0 && max_y > 0) {
          run_case(workspace, bit_depth, plane, nominal_w, nominal_h, max_x,
                   max_y);
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Depth-clipping plane copy
// ---------------------------------------------------------------------------

void test_copy(Workspace& workspace, int bit_depth) {
  const FrameGeom& geometry = workspace.geometry();
  for (int plane = 0; plane < geometry.num_planes; ++plane) {
    const auto& plane_geom = geometry.planes[plane];
    const int stride = plane_geom.padded_w;
    const int width = std::min(stride, 64);
    const int height = std::min(plane_geom.padded_h, 64);
    std::vector<std::uint16_t> src(
        static_cast<std::size_t>(stride) * plane_geom.padded_h);
    for (auto& value : src) {
      value = static_cast<std::uint16_t>(
          random_range(0, (1 << (bit_depth + 1))));
    }
    CHECK(upload(src, workspace.src_plane(plane)));
    std::vector<std::uint16_t> zeros(4096, 0);
    CHECK(upload(zeros, workspace.pred_block(0)));

    av2enc::cuda::chroma::CopyJob job;
    job.src = workspace.src_plane(plane);
    job.src_stride = stride;
    job.dst = workspace.pred_block(0);
    job.dst_stride = width;
    job.width = width;
    job.height = height;
    job.bit_depth = bit_depth;
    std::string error;
    CHECK_EQ(static_cast<int>(
                 workspace.copy_planes(&job, 1, nullptr, &error).code),
             static_cast<int>(ChromaStatus::Code::kOk));
    std::vector<std::uint16_t> device(width * height, 0);
    CHECK(download(workspace.pred_block(0), device));
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const std::size_t source_index =
            static_cast<std::size_t>(y) * stride +
            static_cast<std::size_t>(x);
        CHECK_EQ(device[static_cast<std::size_t>(y) * width + x],
                 av2enc::cuda::chroma::testref::clip1(
                     static_cast<int>(src[source_index]), bit_depth));
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Arena contract
// ---------------------------------------------------------------------------

void test_arena_contract(const Config& config, std::uint64_t required) {
  Arena arena;
  CHECK(av2enc::cuda::is_ok(arena.init(required + 4096)));
  Workspace workspace;
  std::string error;
  std::uint64_t needed = 0;
  CHECK_EQ(static_cast<int>(
               workspace.init(arena, config, &error, &needed).code),
           static_cast<int>(ChromaStatus::Code::kOk));
  CHECK_EQ(needed, required);
  CHECK_EQ(arena.used(),
           av2enc::cuda::chroma::persistent_device_bytes(config));

  // Two back-to-back launches of a two-job forward batch: no growth.
  TxShape shape_a;
  TxShape shape_b;
  CHECK(av2enc::cuda::chroma::make_tx_shape(8, 8, &shape_a));
  CHECK(av2enc::cuda::chroma::make_tx_shape(32, 64, &shape_b));
  std::vector<std::int16_t> zeros(1024, 0);
  for (int slot = 0; slot < 2; ++slot) {
    CHECK(upload(zeros, workspace.residual_adj(slot)));
  }
  av2enc::cuda::chroma::ForwardJob jobs[2];
  jobs[0].shape = shape_a;
  jobs[0].residual_adj = workspace.residual_adj(0);
  jobs[0].coeff = workspace.coeff(0);
  jobs[0].levels = workspace.levels(0);
  jobs[1].shape = shape_b;
  jobs[1].residual_adj = workspace.residual_adj(1);
  jobs[1].coeff = workspace.coeff(1);
  jobs[1].levels = workspace.levels(1);

  const std::uint64_t mark = arena.mark();
  CHECK_EQ(static_cast<int>(
               workspace.forward(jobs, 2, nullptr, &error).code),
           static_cast<int>(ChromaStatus::Code::kOk));
  const std::uint64_t peak1 = arena.peak_used();
  CHECK_EQ(arena.used(), mark);
  CHECK_EQ(static_cast<int>(
               workspace.forward(jobs, 2, nullptr, &error).code),
           static_cast<int>(ChromaStatus::Code::kOk));
  CHECK_EQ(arena.used(), mark);
  CHECK_EQ(arena.peak_used(), peak1);
  std::printf("  arena contract: required=%llu used=%llu peak=%llu\n",
              static_cast<unsigned long long>(required),
              static_cast<unsigned long long>(arena.used()),
              static_cast<unsigned long long>(arena.peak_used()));

  // Invalid arguments.
  av2enc::cuda::chroma::PredictJob bad_predict;
  CHECK_EQ(static_cast<int>(
               workspace.predict(&bad_predict, 1, nullptr, &error).code),
           static_cast<int>(ChromaStatus::Code::kInvalidArgument));
  CHECK_EQ(static_cast<int>(
               workspace.forward(jobs, 0, nullptr, &error).code),
           static_cast<int>(ChromaStatus::Code::kInvalidArgument));
  CHECK_EQ(static_cast<int>(workspace.forward(jobs,
                                              config.max_blocks + 1, nullptr,
                                              &error).code),
           static_cast<int>(ChromaStatus::Code::kInvalidArgument));
  CHECK_EQ(static_cast<int>(
               workspace.forward(nullptr, 1, nullptr, &error).code),
           static_cast<int>(ChromaStatus::Code::kInvalidArgument));

  // One byte short.
  Arena small;
  CHECK(av2enc::cuda::is_ok(small.init(required - 1)));
  Workspace short_ws;
  std::string short_error;
  std::uint64_t short_needed = 0;
  const ChromaStatus status =
      short_ws.init(small, config, &short_error, &short_needed);
  CHECK_EQ(static_cast<int>(status.code),
           static_cast<int>(ChromaStatus::Code::kArenaTooSmall));
  CHECK_EQ(status.required_bytes, required);
  CHECK_EQ(short_needed, required);
  CHECK_EQ(small.used(), 0);
  CHECK(!short_ws.initialized());
}

}  // namespace

int main() {
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
    std::printf("no CUDA device\n");
    return 1;
  }
  std::printf("av2enc CUDA chroma-format test\n");
  std::printf("device: %s (sm_%d%d, %d SMs)\n", prop.name, prop.major,
              prop.minor, prop.multiProcessorCount);

  const Format formats[] = {Format::k420, Format::k422, Format::k444,
                            Format::k400};
  Config base;
  base.width = 64;
  base.height = 64;
  base.format = Format::k444;  // largest geometry
  base.bit_depth = 8;
  base.max_blocks = 4;

  std::uint64_t arena_bytes = 0;
  for (Format format : formats) {
    Config config = base;
    config.format = format;
    arena_bytes = std::max(
        arena_bytes, av2enc::cuda::chroma::required_device_bytes(config));
  }
  Arena arena;
  CHECK(av2enc::cuda::is_ok(arena.init(arena_bytes + 4096)));

  for (Format format : formats) {
    Config config = base;
    config.format = format;
    const std::uint64_t required =
        av2enc::cuda::chroma::required_device_bytes(config);
    const std::uint64_t persistent =
        av2enc::cuda::chroma::persistent_device_bytes(config);
    const std::uint64_t scratch =
        av2enc::cuda::chroma::scratch_device_bytes(config);
    CHECK_EQ(required, persistent + scratch);
    CHECK_EQ(required % Arena::kAlignment, 0);

    arena.reset();
    Workspace workspace;
    std::string error;
    std::uint64_t needed = 0;
    CHECK_EQ(static_cast<int>(
                 workspace.init(arena, config, &error, &needed).code),
             static_cast<int>(ChromaStatus::Code::kOk));
    CHECK_EQ(needed, required);
    CHECK_EQ(arena.used(), persistent);
    std::printf("  %-5s required=%llu persistent=%llu scratch=%llu\n",
                av2enc::cuda::chroma::format_name(format),
                static_cast<unsigned long long>(required),
                static_cast<unsigned long long>(persistent),
                static_cast<unsigned long long>(scratch));

    const long long before = g_checks;
    test_pipelines(workspace);
    test_copy(workspace, 8);
    test_copy(workspace, 10);
    std::printf("  %-5s pipeline+copy: checks=%lld\n",
                av2enc::cuda::chroma::format_name(format),
                g_checks - before);
  }

  // All 25 transform sizes x both depths.  The sweep is format-independent,
  // so one 4:4:4 workspace is enough.
  arena.reset();
  Workspace transform_workspace;
  std::string error;
  CHECK_EQ(static_cast<int>(
               transform_workspace.init(arena, base, &error).code),
           static_cast<int>(ChromaStatus::Code::kOk));
  const long long tx_before = g_checks;
  test_transforms(transform_workspace);
  std::printf("  transforms (25 sizes x 2 depths): checks=%lld\n",
              g_checks - tx_before);

  Config contract_config = base;
  contract_config.format = Format::k422;
  const std::uint64_t contract_required =
      av2enc::cuda::chroma::required_device_bytes(contract_config);
  test_arena_contract(contract_config, contract_required);

  arena.release();
  if (g_failures == 0) {
    std::printf("chroma_formats GPU: %lld checks, PASS\n", g_checks);
    return 0;
  }
  std::printf("chroma_formats GPU: %lld checks, %d FAILURES\n", g_checks,
              g_failures);
  return 1;
}

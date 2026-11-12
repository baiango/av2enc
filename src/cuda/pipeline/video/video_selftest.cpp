// Backend selftest for the inter residual chain.
//
// The baseline M2 decisions code every inter leaf with skip_flag = 1 unless the
// optional inter-residual tool is enabled, so this test keeps that path
// honest: it runs MC -> residual -> downsample -> forward/quant -> inverse ->
// reconstruction on the linked backend and compares every level and every
// reconstructed sample of Y, U and V with the CPU transform reference.
//
//   ./build_and_test.sh --selftest 64       # emulation backend
//   ./build_gpu_gate.sh --selftest 128      # CUDA backend

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "codec/inter/mc/mc.h"
#include "pipeline/video/video_frame.h"
#include "codec/transform/variants/tx_variants.h"

#include "video_backend.h"
#include "video_gpu.h"

namespace av2enc::cuda::video {
namespace {

using detail::Backend;
using detail::InterJob;
using detail::PlaneTx;

std::uint32_t g_rng = 0x9e3779b9u;
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

int clamped_sample(const std::vector<std::uint8_t>& plane, int stride, int w,
                   int h, int x, int y) {
  const int cx = std::clamp(x, 0, w - 1);
  const int cy = std::clamp(y, 0, h - 1);
  return plane[static_cast<std::size_t>(cy) * static_cast<std::size_t>(stride) +
               static_cast<std::size_t>(cx)];
}

bool cpu_tx_plane(const std::vector<std::uint8_t>& source, int src_stride,
                  int src_w, int src_h, int x0, int y0, int w, int h,
                  const std::vector<std::uint8_t>& pred, PlaneTx& out,
                  std::string& error) {
  const auto params_opt =
      av2enc::txv::make_tx_params(w, h, av2enc::txv::TxType::kDctDct);
  if (!params_opt.has_value()) {
    error = "selftest bad transform size";
    return false;
  }
  const av2enc::txv::TxParams& params = *params_opt;
  const std::size_t coded =
      static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
  std::vector<std::int16_t> residual_coded(coded);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const int src = clamped_sample(source, src_stride, src_w, src_h,
                                     x0 + x, y0 + y);
      const int p = pred[static_cast<std::size_t>(y) *
                             static_cast<std::size_t>(w) +
                         static_cast<std::size_t>(x)];
      residual_coded[static_cast<std::size_t>(y) *
                         static_cast<std::size_t>(w) +
                     static_cast<std::size_t>(x)] =
          static_cast<std::int16_t>(src - p);
    }
  }
  std::vector<std::int16_t> residual_adj(
      static_cast<std::size_t>(params.w) * static_cast<std::size_t>(params.h));
  av2enc::txv::downsample_residual(residual_coded.data(), params,
                                   residual_adj.data());
  std::vector<double> coeff(
      static_cast<std::size_t>(params.w) * static_cast<std::size_t>(params.h));
  out.levels.assign(coeff.size(), 0);
  av2enc::txv::forward_transform(residual_adj.data(), params, coeff.data(),
                                 out.levels.data());
  std::vector<std::int32_t> dequant(coeff.size());
  std::vector<std::int16_t> recon_residual(coded);
  av2enc::txv::inverse_transform(out.levels.data(), params, dequant.data(),
                                 recon_residual.data());
  out.recon.resize(coded);
  for (std::size_t i = 0; i < coded; ++i) {
    const int v = static_cast<int>(pred[i]) + recon_residual[i];
    out.recon[i] = static_cast<std::uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
  }
  return true;
}

bool cpu_reference_420(
    const std::array<std::vector<std::uint8_t>, 3>& source, int width,
    int height, const InterJob& job,
    const av2enc::inter::mc::RefFrame& ref_frame,
    std::array<PlaneTx, 3>& out, std::string& error) {
  const int cw = job.w / 2;
  const int ch = job.h / 2;
  std::vector<std::uint8_t> y(
      static_cast<std::size_t>(job.w) * static_cast<std::size_t>(job.h));
  std::vector<std::uint8_t> u(static_cast<std::size_t>(cw) *
                              static_cast<std::size_t>(ch));
  std::vector<std::uint8_t> v(u.size());
  av2enc::inter::mc::MutablePlane dst[3];
  dst[0] = av2enc::inter::mc::MutablePlane{y.data(), job.w};
  dst[1] = av2enc::inter::mc::MutablePlane{u.data(), cw};
  dst[2] = av2enc::inter::mc::MutablePlane{v.data(), cw};
  const av2enc::inter::mc::BlockRect rect{job.x, job.y, job.w, job.h};
  if (av2enc::inter::mc::predict_block_420(
          ref_frame, rect, av2enc::inter::mc::Mv{job.mv_y, job.mv_x},
          av2enc::inter::mc::InterpFilter::kEighttap,
          dst) != av2enc::inter::mc::McStatus::kOk) {
    error = "selftest reference MC failed";
    return false;
  }
  const std::vector<std::uint8_t>* preds[3] = {&y, &u, &v};
  const int coords[3][4] = {
      {job.x, job.y, job.w, job.h},
      {job.x / 2, job.y / 2, cw, ch},
      {job.x / 2, job.y / 2, cw, ch}};
  for (int plane = 0; plane < 3; ++plane) {
    const int stride = plane == 0 ? width : width / 2;
    const int ph = plane == 0 ? height : height / 2;
    if (!cpu_tx_plane(source[static_cast<std::size_t>(plane)], stride,
                      stride, ph, coords[plane][0], coords[plane][1],
                      coords[plane][2], coords[plane][3],
                      *preds[static_cast<std::size_t>(plane)],
                      out[static_cast<std::size_t>(plane)], error)) {
      return false;
    }
  }
  return true;
}

}  // namespace

bool video_inter_residual_selftest(int trials, std::string& error) {
  error.clear();
  if (trials <= 0) trials = 64;
  constexpr int kWidth = 64;
  constexpr int kHeight = 64;
  constexpr int kChromaW = kWidth / 2;
  constexpr int kChromaH = kHeight / 2;
  std::array<std::vector<std::uint8_t>, 3> source;
  std::array<std::vector<std::uint8_t>, 3> reference;
  source[0].resize(static_cast<std::size_t>(kWidth) * kHeight);
  reference[0].resize(source[0].size());
  source[1].resize(static_cast<std::size_t>(kChromaW) * kChromaH);
  reference[1].resize(source[1].size());
  source[2].resize(source[1].size());
  reference[2].resize(source[1].size());
  for (int plane = 0; plane < 3; ++plane) {
    for (std::size_t i = 0; i < source[static_cast<std::size_t>(plane)].size();
         ++i) {
      source[static_cast<std::size_t>(plane)][i] =
          static_cast<std::uint8_t>(next_random() & 0xFFu);
      reference[static_cast<std::size_t>(plane)][i] =
          static_cast<std::uint8_t>(next_random() & 0xFFu);
    }
  }

  av2enc::video::VideoInputFrame host_source;
  host_source.width = kWidth;
  host_source.height = kHeight;
  host_source.y = std::span<const std::uint8_t>(source[0]);
  host_source.u = std::span<const std::uint8_t>(source[1]);
  host_source.v = std::span<const std::uint8_t>(source[2]);

  av2enc::video::VideoReconFrame host_reference;
  host_reference.width = kWidth;
  host_reference.height = kHeight;
  host_reference.y = reference[0];
  host_reference.u = reference[1];
  host_reference.v = reference[2];

  std::unique_ptr<Backend> backend = detail::make_backend();
  if (!backend) {
    error = "no backend";
    return false;
  }
  backend->reset_stats();
  if (!backend->begin_frame(host_source, kWidth / 4, kHeight / 4,
                            &host_reference, error)) {
    return false;
  }

  av2enc::inter::mc::RefFrame ref_frame;
  ref_frame.plane[0] =
      av2enc::inter::mc::PlaneView{reference[0].data(), kWidth};
  ref_frame.plane[1] =
      av2enc::inter::mc::PlaneView{reference[1].data(), kChromaW};
  ref_frame.plane[2] =
      av2enc::inter::mc::PlaneView{reference[2].data(), kChromaW};
  ref_frame.mi_cols = kWidth / 4;
  ref_frame.mi_rows = kHeight / 4;

  const int sizes[][2] = {{16, 16}, {32, 32}, {64, 64}, {32, 16}, {16, 32}};
  for (int t = 0; t < trials; ++t) {
    const int index = random_range(0, 4);
    InterJob job;
    job.w = sizes[index][0];
    job.h = sizes[index][1];
    // mc_predict_block_420 requires even luma origins, like the partition
    // grid (4-sample alignment in the encoder).
    job.x = random_range(0, (kWidth - job.w) / 2) * 2;
    job.y = random_range(0, (kHeight - job.h) / 2) * 2;
    job.mv_y = random_range(-16, 16);
    job.mv_x = random_range(-16, 16);

    std::array<PlaneTx, 3> got;
    if (!backend->inter_residual_420(job, got, error)) return false;
    std::array<PlaneTx, 3> want;
    if (!cpu_reference_420(source, kWidth, kHeight, job, ref_frame, want,
                           error)) {
      return false;
    }
    for (int plane = 0; plane < 3; ++plane) {
      if (got[static_cast<std::size_t>(plane)].levels !=
          want[static_cast<std::size_t>(plane)].levels) {
        error = "inter residual levels differ (plane " +
                std::to_string(plane) + ", trial " + std::to_string(t) + ")";
        return false;
      }
      if (got[static_cast<std::size_t>(plane)].recon !=
          want[static_cast<std::size_t>(plane)].recon) {
        error = "inter residual recon differs (plane " +
                std::to_string(plane) + ", trial " + std::to_string(t) + ")";
        return false;
      }
    }
  }

  if (!backend->end_frame(error)) return false;
  std::printf(
      "m2 inter-residual selftest: PASS (%d trials, backend=%s, "
      "mc=%llu tx=%llu resid=%llu recon=%llu)\n",
      trials, backend->name(),
      static_cast<unsigned long long>(backend->stats().mc_launches),
      static_cast<unsigned long long>(backend->stats().tx_launches),
      static_cast<unsigned long long>(backend->stats().residual_launches),
      static_cast<unsigned long long>(backend->stats().recon_launches));
  return true;
}

}  // namespace av2enc::cuda::video

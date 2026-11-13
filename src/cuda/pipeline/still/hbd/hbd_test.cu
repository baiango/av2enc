// CPU-vs-GPU bit-exact tests for the 10-bit M1 path.
//
// Build/run on the CUDA box (sm_86):
//   ./build_hbd_test.sh
//
// Covered:
//   * 4:2:0 geometry (visible vs coded padding) and 8->10 upload against
//     `fmt::convert_420_8_to_picture()` for even and odd dimensions;
//   * 10->8 download against `fmt::convert_to_420_8()`;
//   * a mixed-size, mixed-TX_TYPE batch through `hbd_tx_batch()` against
//     `hbd::encode_block()` / `encode_block_residual()`: levels, EOB, dequant
//     and inverse residual must match element-for-element at 8/10/12-bit.
//
// The comparison is per element, not a checksum or MD5.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "cuda/pipeline/still/hbd/hbd_cpu.h"
#include "cuda/pipeline/still/hbd/hbd_frame.h"
#include "cuda/pipeline/still/hbd/hbd_tx.h"
#include "formats/format.h"
#include "codec/transform/variants/tx_variants.h"

namespace {

namespace hbd = av2enc::hbd;
namespace chbd = av2enc::cuda::hbd;
namespace fmt = av2enc::fmt;
namespace txv = av2enc::txv;

int g_failures = 0;
long long g_checks = 0;

#define CHECK(condition)                                                     \
  do {                                                                       \
    if (!(condition)) {                                                      \
      ++g_failures;                                                          \
      std::printf("%s:%d: CHECK failed: %s\n", __FILE__, __LINE__,           \
                  #condition);                                               \
    }                                                                        \
    ++g_checks;                                                              \
  } while (false)

#define CUDA_TRY(expr)                                                       \
  do {                                                                       \
    const cudaError_t err_ = (expr);                                         \
    if (err_ != cudaSuccess) {                                               \
      std::printf("%s:%d: CUDA error: %s\n", __FILE__, __LINE__,             \
                  cudaGetErrorString(err_));                                 \
      std::exit(1);                                                          \
    }                                                                        \
  } while (false)

template <typename T>
T* device_copy(const std::vector<T>& host) {
  if (host.empty()) {
    return nullptr;
  }
  T* dev = nullptr;
  CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&dev), host.size() * sizeof(T)));
  CUDA_TRY(cudaMemcpy(dev, host.data(), host.size() * sizeof(T),
                      cudaMemcpyHostToDevice));
  return dev;
}

template <typename T>
void download(T* dev, std::vector<T>* host) {
  if (host->empty()) {
    return;
  }
  CUDA_TRY(cudaMemcpy(host->data(), dev, host->size() * sizeof(T),
                      cudaMemcpyDeviceToHost));
}

// ---------------------------------------------------------------------------
// Sample path
// ---------------------------------------------------------------------------

struct Source420 {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
};

Source420 make_source(int width, int height, std::mt19937* rng) {
  Source420 source;
  source.width = width;
  source.height = height;
  const int chroma_w = (width + 1) / 2;
  const int chroma_h = (height + 1) / 2;
  std::uniform_int_distribution<int> dist(0, 255);
  source.y.resize(static_cast<std::size_t>(width) * height);
  source.u.resize(static_cast<std::size_t>(chroma_w) * chroma_h);
  source.v.resize(source.u.size());
  for (std::uint8_t& value : source.y) value = static_cast<std::uint8_t>(dist(*rng));
  for (std::uint8_t& value : source.u) value = static_cast<std::uint8_t>(dist(*rng));
  for (std::uint8_t& value : source.v) value = static_cast<std::uint8_t>(dist(*rng));
  return source;
}

fmt::Source420_8 source_view(const Source420& source) {
  fmt::Source420_8 view;
  view.width = static_cast<std::uint32_t>(source.width);
  view.height = static_cast<std::uint32_t>(source.height);
  view.y = {source.y.data(), static_cast<std::uint32_t>(source.width)};
  view.u = {source.u.data(),
            static_cast<std::uint32_t>((source.width + 1) / 2)};
  view.v = {source.v.data(),
            static_cast<std::uint32_t>((source.width + 1) / 2)};
  return view;
}

void test_frame_upload_download() {
  std::mt19937 rng(0x420u);
  for (const auto& dims : {std::pair<int, int>{64, 64},
                           std::pair<int, int>{67, 43},
                           std::pair<int, int>{640, 360},
                           std::pair<int, int>{641, 361}}) {
    const int width = dims.first;
    const int height = dims.second;
    const Source420 source = make_source(width, height, &rng);
    const fmt::Source420_8 view = source_view(source);

    fmt::Picture picture;
    const fmt::ConvertResult converted = fmt::convert_420_8_to_picture(
        view, fmt::ChromaFormat::k420, fmt::BitDepth::k10, picture);
    CHECK(converted.ok());
    if (!converted.ok()) continue;

    chbd::Hbd420Frame frame;
    std::string error;
    CHECK(chbd::hbd_make_420_frame_10(static_cast<std::uint32_t>(width),
                                      static_cast<std::uint32_t>(height),
                                      &frame, &error));
    CHECK(frame.total_samples == picture.geometry.total_samples);
    CHECK(frame.frame_bytes == picture.geometry.frame_bytes);

    // `hbd_upload_420_8()` consumes device-resident source planes.  Upload the
    // three 8-bit planes first; passing the host vectors directly would make
    // the kernel dereference host addresses (illegal access).
    const int chroma_w = (width + 1) / 2;
    std::uint8_t* src_dev_y = nullptr;
    std::uint8_t* src_dev_u = nullptr;
    std::uint8_t* src_dev_v = nullptr;
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&src_dev_y),
                        source.y.size()));
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&src_dev_u),
                        source.u.size()));
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&src_dev_v),
                        source.v.size()));
    CUDA_TRY(cudaMemcpy(src_dev_y, source.y.data(), source.y.size(),
                        cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemcpy(src_dev_u, source.u.data(), source.u.size(),
                        cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemcpy(src_dev_v, source.v.data(), source.v.size(),
                        cudaMemcpyHostToDevice));

    fmt::Source420_8 dev_view = view;
    dev_view.y.data = src_dev_y;
    dev_view.u.data = src_dev_u;
    dev_view.v.data = src_dev_v;
    // Keep the compiler honest about the stride contract (bytes, not samples).
    dev_view.y.stride = static_cast<std::uint32_t>(width);
    dev_view.u.stride = static_cast<std::uint32_t>(chroma_w);
    dev_view.v.stride = static_cast<std::uint32_t>(chroma_w);

    std::uint16_t* dev_frame = nullptr;
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&dev_frame),
                        frame.total_samples * sizeof(std::uint16_t)));
    CUDA_TRY(cudaMemset(dev_frame, 0, frame.frame_bytes));
    CUDA_TRY(chbd::hbd_upload_420_8(dev_view, frame, dev_frame));
    CUDA_TRY(cudaDeviceSynchronize());
    CUDA_TRY(cudaFree(src_dev_y));
    CUDA_TRY(cudaFree(src_dev_u));
    CUDA_TRY(cudaFree(src_dev_v));

    std::vector<std::uint16_t> actual(frame.total_samples, 0);
    CUDA_TRY(cudaMemcpy(actual.data(), dev_frame, frame.frame_bytes,
                        cudaMemcpyDeviceToHost));

    // Compare every coded sample, including replicated padding.
    for (int plane = 0; plane < 3; ++plane) {
      const chbd::HbdPlane420& gp = frame.planes[plane];
      const fmt::PlaneBuffer& cp = picture.planes[plane];
      for (std::uint32_t y = 0; y < gp.coded_height; ++y) {
        for (std::uint32_t x = 0; x < gp.coded_width; ++x) {
          CHECK(actual[gp.offset + static_cast<std::size_t>(y) * gp.stride + x] ==
                cp.at(x, y));
        }
      }
    }

    // Border replication on an unpadded device frame must match the CPU
    // `fmt::replicate_borders()` result (the upload path already filled
    // padding, so zero it first and re-run the standalone kernel).
    fmt::Picture unpadded = picture;
    for (int plane = 0; plane < 3; ++plane) {
      const chbd::HbdPlane420& gp = frame.planes[plane];
      for (std::uint32_t y = 0; y < gp.coded_height; ++y) {
        for (std::uint32_t x = 0; x < gp.coded_width; ++x) {
          if (x >= gp.visible_width || y >= gp.visible_height) {
            unpadded.planes[plane].at(x, y) = 0;
          }
        }
      }
    }
    for (int plane = 0; plane < 3; ++plane) {
      const chbd::HbdPlane420& gp = frame.planes[plane];
      CUDA_TRY(cudaMemcpy(dev_frame + gp.offset,
                          unpadded.planes[plane].samples.data(),
                          static_cast<std::size_t>(gp.coded_width) *
                              gp.coded_height * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice));
    }
    CUDA_TRY(chbd::hbd_replicate_420_borders(frame, dev_frame));
    CUDA_TRY(cudaDeviceSynchronize());
    CUDA_TRY(cudaMemcpy(actual.data(), dev_frame, frame.frame_bytes,
                        cudaMemcpyDeviceToHost));
    for (int plane = 0; plane < 3; ++plane) {
      const chbd::HbdPlane420& gp = frame.planes[plane];
      const fmt::PlaneBuffer& cp = picture.planes[plane];
      for (std::uint32_t y = 0; y < gp.coded_height; ++y) {
        for (std::uint32_t x = 0; x < gp.coded_width; ++x) {
          CHECK(actual[gp.offset +
                       static_cast<std::size_t>(y) * gp.stride + x] ==
                cp.at(x, y));
        }
      }
    }

    // Download visible 10-bit -> 8-bit and compare with the CPU conversion.
    fmt::Picture420_8 expected;
    const fmt::ConvertResult reverse = fmt::convert_to_420_8(picture, expected);
    CHECK(reverse.ok());
    if (!reverse.ok()) {
      CUDA_TRY(cudaFree(dev_frame));
      continue;
    }

    std::vector<std::uint8_t> out_y(static_cast<std::size_t>(width) * height,
                                    0);
    std::vector<std::uint8_t> out_u(
        static_cast<std::size_t>((width + 1) / 2) * ((height + 1) / 2), 0);
    std::vector<std::uint8_t> out_v(out_u.size(), 0);
    std::uint8_t* out_dev_y = nullptr;
    std::uint8_t* out_dev_u = nullptr;
    std::uint8_t* out_dev_v = nullptr;
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&out_dev_y), out_y.size()));
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&out_dev_u), out_u.size()));
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&out_dev_v), out_v.size()));
    chbd::HbdPlane8View views[3];
    views[0] = {out_dev_y, width, width, height};
    views[1] = {out_dev_u, (width + 1) / 2, (width + 1) / 2, (height + 1) / 2};
    views[2] = {out_dev_v, (width + 1) / 2, (width + 1) / 2, (height + 1) / 2};
    CUDA_TRY(chbd::hbd_download_420_8(frame, dev_frame, views));
    CUDA_TRY(cudaDeviceSynchronize());
    CUDA_TRY(cudaMemcpy(out_y.data(), out_dev_y, out_y.size(),
                        cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaMemcpy(out_u.data(), out_dev_u, out_u.size(),
                        cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaMemcpy(out_v.data(), out_dev_v, out_v.size(),
                        cudaMemcpyDeviceToHost));
    CHECK(out_y == expected.y.data);
    CHECK(out_u == expected.u.data);
    CHECK(out_v == expected.v.data);

    CUDA_TRY(cudaFree(out_dev_y));
    CUDA_TRY(cudaFree(out_dev_u));
    CUDA_TRY(cudaFree(out_dev_v));
    CUDA_TRY(cudaFree(dev_frame));
  }
}

// ---------------------------------------------------------------------------
// Transform path
// ---------------------------------------------------------------------------

struct TxCase {
  int coded_w = 0;
  int coded_h = 0;
  txv::TxType type = txv::TxType::kDctDct;
  bool residual_mode = false;
  bool packed = true;
  std::vector<std::uint16_t> src;
  std::vector<std::uint16_t> pred;
  std::vector<std::int16_t> residual;
  int src_stride = 0;
  int src_width = 0;
  int src_height = 0;
  int x0 = 0;
  int y0 = 0;
  hbd::BlockResult expected;
};

struct DeviceBatch {
  std::uint16_t* src = nullptr;
  std::uint16_t* pred = nullptr;
  std::int16_t* residual_in = nullptr;
  std::int16_t* residual_adj = nullptr;
  double* coeff = nullptr;
  std::int32_t* levels = nullptr;
  std::int32_t* dequant = nullptr;
  std::int16_t* residual_out = nullptr;
  std::int32_t* eob = nullptr;
  int* error_count = nullptr;
  std::vector<std::int32_t> h_levels;
  std::vector<std::int32_t> h_dequant;
  std::vector<std::int16_t> h_residual_out;
  std::vector<std::int32_t> h_eob;
};

void run_depth(int bit_depth) {
  std::mt19937 rng(0x10B17u + static_cast<unsigned>(bit_depth));
  const int sample_high = (1 << bit_depth) - 1;
  std::uniform_int_distribution<int> sample_dist(0, sample_high);
  std::uniform_int_distribution<int> residual_dist(-(1 << bit_depth),
                                                  (1 << bit_depth) - 1);

  const int sizes[][2] = {{4, 4},   {8, 8},   {16, 16}, {32, 32}, {64, 64},
                          {4, 8},   {8, 4},   {8, 16},  {16, 8},  {16, 32},
                          {32, 16}, {4, 16},  {16, 4},  {8, 32},  {32, 8},
                          {16, 64}, {64, 16}, {4, 32},  {32, 4},  {8, 64},
                          {64, 8},  {64, 32}, {32, 64}};
  const txv::TxType types[] = {txv::TxType::kDctDct, txv::TxType::kAdstAdst,
                               txv::TxType::kIdtx, txv::TxType::kDctAdst,
                               txv::TxType::kDctFlipadst};

  std::vector<TxCase> cases;
  int index = 0;
  for (const auto& size : sizes) {
    TxCase c;
    c.coded_w = size[0];
    c.coded_h = size[1];
    c.type = types[static_cast<std::size_t>(index) % 5];
    c.residual_mode = (index % 3) == 1;
    c.packed = (index % 2) == 0;
    ++index;
    const int cw = c.coded_w;
    const int ch = c.coded_h;

    c.pred.resize(static_cast<std::size_t>(cw) * ch);
    for (std::uint16_t& value : c.pred) {
      value = static_cast<std::uint16_t>(sample_dist(rng));
    }
    if (c.residual_mode) {
      c.residual.resize(static_cast<std::size_t>(cw) * ch);
      for (std::int16_t& value : c.residual) {
        value = static_cast<std::int16_t>(residual_dist(rng));
      }
    }

    if (c.packed) {
      c.src.resize(static_cast<std::size_t>(cw) * ch);
      for (std::uint16_t& value : c.src) {
        value = static_cast<std::uint16_t>(sample_dist(rng));
      }
      c.src_stride = cw;
      c.src_width = cw;
      c.src_height = ch;
      c.x0 = 0;
      c.y0 = 0;
    } else {
      // Non-tight plane with the block at a non-zero origin, so the GPU's
      // clamping path is compared too.
      c.src_width = cw + 5;
      c.src_height = ch + 3;
      c.x0 = 2;
      c.y0 = 1;
      c.src_stride = c.src_width;
      c.src.resize(static_cast<std::size_t>(c.src_width) * c.src_height);
      for (std::uint16_t& value : c.src) {
        value = static_cast<std::uint16_t>(sample_dist(rng));
      }
    }

    const auto params = txv::make_tx_params(cw, ch, c.type);
    CHECK(params.has_value());
    if (!params) continue;
    bool ok;
    if (c.residual_mode) {
      ok = hbd::encode_block_residual(c.residual.data(), c.pred.data(),
                                      *params, bit_depth, &c.expected);
    } else {
      ok = hbd::encode_block(c.src.data(), c.src_stride, c.src_width,
                             c.src_height, c.x0, c.y0, c.pred.data(), *params,
                             bit_depth, &c.expected);
    }
    CHECK(ok);
    cases.push_back(std::move(c));
  }

  // Build one mixed batch with explicit element offsets.
  std::vector<chbd::HbdTxJob> jobs;
  std::vector<std::uint16_t> all_src;
  std::vector<std::uint16_t> all_pred;
  std::vector<std::int16_t> all_residual;
  std::size_t src_off = 0;
  std::size_t pred_off = 0;
  std::size_t residual_off = 0;
  std::size_t adj_off = 0;
  std::size_t coeff_off = 0;
  std::size_t level_off = 0;
  std::size_t dequant_off = 0;
  std::size_t out_off = 0;
  int max_elems = 0;

  for (const TxCase& c : cases) {
    const auto params = txv::make_tx_params(c.coded_w, c.coded_h, c.type);
    CHECK(params.has_value());
    if (!params) continue;
    const int area = params->w * params->h;
    max_elems = std::max(max_elems, area);

    chbd::HbdTxJob job{};
    job.params = *params;
    job.input = c.residual_mode
                    ? static_cast<std::uint8_t>(chbd::HbdTxInput::kResidual)
                    : static_cast<std::uint8_t>(chbd::HbdTxInput::kSourcePred);
    job.src_packed = c.packed ? 1 : 0;
    job.src_x = static_cast<std::int16_t>(c.x0);
    job.src_y = static_cast<std::int16_t>(c.y0);
    job.src_width = static_cast<std::int16_t>(c.src_width);
    job.src_height = static_cast<std::int16_t>(c.src_height);
    job.src_stride = c.src_stride;

    job.src_offset = static_cast<std::int32_t>(src_off);
    all_src.insert(all_src.end(), c.src.begin(), c.src.end());
    src_off += c.src.size();

    job.pred_offset = static_cast<std::int32_t>(pred_off);
    all_pred.insert(all_pred.end(), c.pred.begin(), c.pred.end());
    pred_off += c.pred.size();

    job.residual_in_offset = static_cast<std::int32_t>(residual_off);
    if (c.residual_mode) {
      all_residual.insert(all_residual.end(), c.residual.begin(),
                          c.residual.end());
      residual_off += c.residual.size();
    }

    job.residual_adj_offset = static_cast<std::int32_t>(adj_off);
    adj_off += static_cast<std::size_t>(area);
    job.coeff_offset = static_cast<std::int32_t>(coeff_off);
    coeff_off += static_cast<std::size_t>(area);
    job.levels_offset = static_cast<std::int32_t>(level_off);
    level_off += static_cast<std::size_t>(area);
    job.dequant_offset = static_cast<std::int32_t>(dequant_off);
    dequant_off += static_cast<std::size_t>(area);
    job.residual_out_offset = static_cast<std::int32_t>(out_off);
    out_off += static_cast<std::size_t>(c.coded_w * c.coded_h);
    job.eob_offset = static_cast<std::int32_t>(jobs.size());

    jobs.push_back(job);
  }

  if (jobs.empty()) return;

  DeviceBatch dev;
  dev.src = device_copy(all_src);
  dev.pred = device_copy(all_pred);
  if (!all_residual.empty()) {
    dev.residual_in = device_copy(all_residual);
  }
  CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&dev.residual_adj),
                      adj_off * sizeof(std::int16_t)));
  CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&dev.coeff),
                      coeff_off * sizeof(double)));
  CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&dev.levels),
                      level_off * sizeof(std::int32_t)));
  CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&dev.dequant),
                      dequant_off * sizeof(std::int32_t)));
  CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&dev.residual_out),
                      out_off * sizeof(std::int16_t)));
  CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&dev.eob),
                      jobs.size() * sizeof(std::int32_t)));
  CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&dev.error_count), sizeof(int)));
  CUDA_TRY(cudaMemset(dev.error_count, 0, sizeof(int)));

  chbd::HbdTxBuffers buffers;
  buffers.src = dev.src;
  buffers.pred = dev.pred;
  buffers.residual_in = dev.residual_in;
  buffers.residual_adj = dev.residual_adj;
  buffers.coeff = dev.coeff;
  buffers.levels = dev.levels;
  buffers.dequant = dev.dequant;
  buffers.residual_out = dev.residual_out;
  buffers.eob = dev.eob;
  buffers.error_count = dev.error_count;

  chbd::HbdTxJob* dev_jobs = device_copy(jobs);
  CUDA_TRY(chbd::hbd_tx_batch(dev_jobs, static_cast<int>(jobs.size()), buffers,
                              bit_depth, max_elems));
  CUDA_TRY(cudaDeviceSynchronize());

  int error_count = 0;
  CUDA_TRY(cudaMemcpy(&error_count, dev.error_count, sizeof(int),
                      cudaMemcpyDeviceToHost));
  CHECK(error_count == 0);

  dev.h_levels.assign(level_off, 0);
  dev.h_dequant.assign(dequant_off, 0);
  dev.h_residual_out.assign(out_off, 0);
  dev.h_eob.assign(jobs.size(), 0);
  download(dev.levels, &dev.h_levels);
  download(dev.dequant, &dev.h_dequant);
  download(dev.residual_out, &dev.h_residual_out);
  download(dev.eob, &dev.h_eob);

  for (std::size_t i = 0; i < jobs.size(); ++i) {
    const TxCase& c = cases[i];
    const chbd::HbdTxJob& job = jobs[i];
    const auto params = txv::make_tx_params(c.coded_w, c.coded_h, c.type);
    if (!params) continue;
    const int area = params->w * params->h;
    CHECK(dev.h_eob[i] == c.expected.eob);
    for (int t = 0; t < area; ++t) {
      const std::size_t at = static_cast<std::size_t>(job.levels_offset) + t;
      CHECK(dev.h_levels[at] == c.expected.levels[static_cast<std::size_t>(t)]);
      const std::size_t dq = static_cast<std::size_t>(job.dequant_offset) + t;
      CHECK(dev.h_dequant[dq] ==
            c.expected.dequant[static_cast<std::size_t>(t)]);
    }
    const int coded = c.coded_w * c.coded_h;
    for (int t = 0; t < coded; ++t) {
      const std::size_t at =
          static_cast<std::size_t>(job.residual_out_offset) + t;
      CHECK(dev.h_residual_out[at] ==
            c.expected.residual[static_cast<std::size_t>(t)]);
    }
  }

  CUDA_TRY(cudaFree(dev.src));
  CUDA_TRY(cudaFree(dev.pred));
  if (dev.residual_in != nullptr) CUDA_TRY(cudaFree(dev.residual_in));
  CUDA_TRY(cudaFree(dev.residual_adj));
  CUDA_TRY(cudaFree(dev.coeff));
  CUDA_TRY(cudaFree(dev.levels));
  CUDA_TRY(cudaFree(dev.dequant));
  CUDA_TRY(cudaFree(dev.residual_out));
  CUDA_TRY(cudaFree(dev.eob));
  CUDA_TRY(cudaFree(dev.error_count));
  CUDA_TRY(cudaFree(dev_jobs));
}

void test_geometry_agreement() {
  for (const auto& dims : {std::pair<int, int>{64, 64},
                           std::pair<int, int>{641, 361}}) {
    fmt::Picture picture;
    std::string error;
    const fmt::Format format{static_cast<std::uint32_t>(dims.first),
                             static_cast<std::uint32_t>(dims.second),
                             fmt::ChromaFormat::k420, fmt::BitDepth::k10};
    CHECK(fmt::allocate_picture(format, picture, error));
    chbd::Hbd420Frame frame;
    CHECK(chbd::hbd_make_420_frame(format, &frame, &error));
    CHECK(frame.total_samples == picture.geometry.total_samples);
    for (int plane = 0; plane < 3; ++plane) {
      CHECK(frame.planes[plane].coded_width ==
            picture.geometry.planes[plane].coded_width);
      CHECK(frame.planes[plane].visible_width ==
            picture.geometry.planes[plane].visible_width);
      CHECK(frame.planes[plane].offset ==
            picture.geometry.planes[plane].offset);
    }
  }
  chbd::Hbd420Frame frame;
  std::string error;
  const fmt::Format four_four_four{64, 64, fmt::ChromaFormat::k444,
                                   fmt::BitDepth::k10};
  CHECK(!chbd::hbd_make_420_frame(four_four_four, &frame, &error));
}

// 10/12-bit native picture conversion for every chroma format: the generic
// frame layout must match `fmt::make_frame_geometry()`, upload must replicate
// visible edges into coded padding, and download must return the visible
// rectangle unchanged.
void test_generic_frame_upload_download() {
  std::mt19937 rng(0x10B1u);
  std::uniform_int_distribution<int> dist(0, 1023);
  const fmt::ChromaFormat formats[] = {
      fmt::ChromaFormat::k400, fmt::ChromaFormat::k420,
      fmt::ChromaFormat::k422, fmt::ChromaFormat::k444};
  const std::pair<int, int> dims[] = {{64, 64},   {67, 43},
                                      {100, 100}, {640, 360}};
  for (const fmt::ChromaFormat chroma : formats) {
    for (const auto& size : dims) {
      fmt::Format format;
      format.width = static_cast<std::uint32_t>(size.first);
      format.height = static_cast<std::uint32_t>(size.second);
      format.chroma = chroma;
      format.depth = fmt::BitDepth::k10;
      fmt::Picture picture;
      std::string error;
      CHECK(fmt::allocate_picture(format, picture, error));
      if (picture.geometry.num_planes == 0) continue;
      for (int plane = 0; plane < picture.geometry.num_planes; ++plane) {
        fmt::PlaneBuffer& p = picture.planes[plane];
        for (std::uint32_t y = 0; y < p.visible_height; ++y) {
          for (std::uint32_t x = 0; x < p.visible_width; ++x) {
            p.at(x, y) = static_cast<std::uint16_t>(dist(rng));
          }
        }
        fmt::replicate_borders(p);
      }

      chbd::Hbd420Frame frame;
      CHECK(chbd::hbd_make_frame(format, &frame, &error));
      CHECK(frame.num_planes == picture.geometry.num_planes);
      CHECK(frame.total_samples == picture.geometry.total_samples);
      CHECK(frame.frame_bytes == picture.geometry.frame_bytes);
      for (int plane = 0; plane < frame.num_planes; ++plane) {
        const chbd::HbdPlane420& fp = frame.planes[plane];
        const fmt::PlaneGeometry& cp = picture.geometry.planes[plane];
        CHECK(fp.coded_width == cp.coded_width);
        CHECK(fp.coded_height == cp.coded_height);
        CHECK(fp.visible_width == cp.visible_width);
        CHECK(fp.visible_height == cp.visible_height);
      }

      std::uint16_t* dev_frame = nullptr;
      CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&dev_frame),
                          frame.total_samples * sizeof(std::uint16_t)));
      CUDA_TRY(cudaMemset(dev_frame, 0, frame.frame_bytes));
      CUDA_TRY(chbd::hbd_upload_picture_10(picture, frame, dev_frame));
      CUDA_TRY(cudaDeviceSynchronize());

      std::vector<std::uint16_t> actual(frame.total_samples, 0);
      CUDA_TRY(cudaMemcpy(actual.data(), dev_frame, frame.frame_bytes,
                          cudaMemcpyDeviceToHost));
      for (int plane = 0; plane < frame.num_planes; ++plane) {
        const chbd::HbdPlane420& fp = frame.planes[plane];
        const fmt::PlaneBuffer& cp = picture.planes[plane];
        for (std::uint32_t y = 0; y < fp.coded_height; ++y) {
          for (std::uint32_t x = 0; x < fp.coded_width; ++x) {
            CHECK(actual[fp.offset + static_cast<std::size_t>(y) * fp.stride +
                         x] == cp.at(x, y));
          }
        }
      }

      std::vector<std::uint16_t> out[3];
      chbd::HbdPlane16View views[3];
      for (int plane = 0; plane < frame.num_planes; ++plane) {
        const chbd::HbdPlane420& fp = frame.planes[plane];
        out[plane].assign(static_cast<std::size_t>(fp.visible_width) *
                              fp.visible_height,
                          0);
        views[plane] = {out[plane].data(), static_cast<int>(fp.visible_width),
                        static_cast<int>(fp.visible_width),
                        static_cast<int>(fp.visible_height)};
      }
      CUDA_TRY(chbd::hbd_download_picture_10(frame, dev_frame, views));
      CUDA_TRY(cudaDeviceSynchronize());
      for (int plane = 0; plane < frame.num_planes; ++plane) {
        const chbd::HbdPlane420& fp = frame.planes[plane];
        const fmt::PlaneBuffer& cp = picture.planes[plane];
        for (std::uint32_t y = 0; y < fp.visible_height; ++y) {
          for (std::uint32_t x = 0; x < fp.visible_width; ++x) {
            CHECK(out[plane][static_cast<std::size_t>(y) * fp.visible_width +
                             x] == cp.at(x, y));
          }
        }
      }
      CUDA_TRY(cudaFree(dev_frame));
    }
  }
}

}  // namespace

int main() {
  test_geometry_agreement();
  test_frame_upload_download();
  test_generic_frame_upload_download();
  run_depth(8);
  run_depth(10);
  run_depth(12);

  if (g_failures != 0) {
    std::printf("hbd_cuda_test: %d failure(s), %lld checks\n", g_failures,
                g_checks);
    return 1;
  }
  std::printf("hbd_cuda_test: PASS (%lld checks)\n", g_checks);
  return 0;
}

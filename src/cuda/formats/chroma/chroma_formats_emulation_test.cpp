// Host emulation of the shared chroma-format device bodies.
//
// This TU compiles `chroma_formats_device.cuh` with the CUDA keywords stubbed
// and compares every result with the real CPU components:
//
//   * av2enc::chroma::make_frame_shape / chroma_mi_size / make_tx_shape
//     (geometry, Subsampled_Size, nominal -> adjusted shape);
//   * av2enc::chroma::forward_dct / quantize_coefficients /
//     dequantize_block / inverse_dct (8-bit reference);
//   * av2enc::hbd::dequantize / inverse_block (10-bit reference, the project's
//     verified high-bit-depth oracle) plus av2enc::txv::make_tx_params.
//
// The prediction / residual / reconstruction reference transcribes
// `TileWriter::encode_residual_plane()` in chroma_still.cpp and calls the same
// public chroma helpers (approx_divide, round2) for its arithmetic.
//
// It also exercises the arena contract with the real Arena implementation
// (built against a host cuda_runtime stub): sizing from
// required_device_bytes(), a two-cycle mark()/rewind() scratch reuse check with
// no net growth, and a one-byte-short negative test that must report
// kArenaTooSmall with the exact required size.
//
// Build/run: ./build_and_test.sh
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

// CUDA keyword stubs: the shared header contains no CUDA-specific code, so
// the host compiler builds exactly the same bytes.
#define __device__
#define __forceinline__ inline

#include "cuda/formats/chroma/chroma_formats.h"
#include "cuda/formats/chroma/chroma_formats_device.cuh"
#include "chroma_formats_reference.h"

#include "cuda/pipeline/still/hbd/hbd_cpu.h"
#include "formats/chroma/chroma_format.h"
#include "formats/chroma/chroma_residual.h"
#include "codec/transform/variants/tx_variants.h"

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
#define CHECK_DOUBLE(a, b)                                                 \
  do {                                                                     \
    expect(std::memcmp(&(a), &(b), sizeof(double)) == 0, #a " bit-exact " #b, \
           "", 0, 0);                                                      \
  } while (0)

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

ChromaFormat cpu_format(Format format) {
  return static_cast<ChromaFormat>(static_cast<std::uint8_t>(format));
}

// ---------------------------------------------------------------------------
// Geometry / Subsampled_Size / transform shape parity
// ---------------------------------------------------------------------------

void test_geometry() {
  const Format formats[] = {Format::k420, Format::k422, Format::k444,
                            Format::k400};
  const std::pair<int, int> sizes[] = {{64, 64},   {640, 360}, {100, 100},
                                       {101, 99},  {65, 63},   {1, 1},
                                       {128, 64},  {63, 65}};
  for (Format format : formats) {
    for (const auto& size : sizes) {
      const FrameGeom ours =
          av2enc::cuda::chroma::make_frame_geometry(size.first, size.second,
                                                    format);
      const av2enc::chroma::FrameShape cpu =
          av2enc::chroma::make_frame_shape(size.first, size.second,
                                           cpu_format(format));
      CHECK_EQ(ours.width, cpu.width);
      CHECK_EQ(ours.height, cpu.height);
      CHECK_EQ(ours.mi_cols, cpu.mi_cols);
      CHECK_EQ(ours.mi_rows, cpu.mi_rows);
      CHECK_EQ(ours.coded_w, cpu.coded_width);
      CHECK_EQ(ours.coded_h, cpu.coded_height);
      CHECK_EQ(ours.padded_w, cpu.padded_width);
      CHECK_EQ(ours.padded_h, cpu.padded_height);
      CHECK_EQ(ours.num_planes, cpu.num_planes);
      CHECK_EQ(ours.monochrome, cpu.monochrome);
      CHECK_EQ(ours.info.idc, cpu.info.idc);
      CHECK_EQ(ours.info.sub_x, cpu.info.subsampling_x);
      CHECK_EQ(ours.info.sub_y, cpu.info.subsampling_y);
      CHECK_EQ(ours.info.profile_idc, cpu.info.profile_idc);
      for (int plane = 0; plane < 3; ++plane) {
        const auto& p = ours.planes[plane];
        const auto& c = cpu.planes[static_cast<std::size_t>(plane)];
        CHECK_EQ(p.present, c.present);
        CHECK_EQ(p.sub_x, c.subsampling_x);
        CHECK_EQ(p.sub_y, c.subsampling_y);
        CHECK_EQ(p.visible_w, c.visible_width);
        CHECK_EQ(p.visible_h, c.visible_height);
        CHECK_EQ(p.coded_w, c.coded_width);
        CHECK_EQ(p.coded_h, c.coded_height);
        CHECK_EQ(p.padded_w, c.padded_width);
        CHECK_EQ(p.padded_h, c.padded_height);
      }
    }
  }
  // Plane origin helper.
  for (int sub = 0; sub <= 1; ++sub) {
    for (int mi = 0; mi < 40; ++mi) {
      CHECK_EQ(av2enc::cuda::chroma::plane_origin_x(mi, sub),
               (mi >> sub) * 4);
      CHECK_EQ(av2enc::cuda::chroma::plane_origin_y(mi, sub),
               (mi >> sub) * 4);
    }
  }
}

void test_residual_sizes() {
  const Format formats[] = {Format::k420, Format::k422, Format::k444,
                            Format::k400};
  for (Format format : formats) {
    for (int index = 0; index < 29; ++index) {
      const ResidualSize ours =
          av2enc::cuda::chroma::chroma_residual_size_block(index, format);
      const BlockSize block = static_cast<BlockSize>(index);
      const auto cpu = av2enc::chroma::chroma_mi_size(block,
                                                      cpu_format(format));
      CHECK_EQ(ours.valid, cpu.has_value());
      if (ours.valid && cpu.has_value()) {
        CHECK_EQ(ours.width, av2enc::chroma::block_width(*cpu));
        CHECK_EQ(ours.height, av2enc::chroma::block_height(*cpu));
        CHECK_EQ(ours.block_index, index);
      }
      const int width = av2enc::cuda::chroma::block_width(index);
      const int height = av2enc::cuda::chroma::block_height(index);
      const ResidualSize by_dims =
          av2enc::cuda::chroma::chroma_residual_size(width, height, format);
      CHECK_EQ(by_dims.valid, ours.valid);
      CHECK_EQ(by_dims.width, ours.width);
      CHECK_EQ(by_dims.height, ours.height);
      CHECK_EQ(av2enc::cuda::chroma::is_chroma_offset_for_subsize(
                   width, height, format),
               av2enc::chroma::is_chroma_offset_for_subsize(
                   block, cpu_format(format)));
    }
  }
}

void test_tx_shapes() {
  const int dims[] = {4, 8, 16, 32, 64};
  for (int width : dims) {
    for (int height : dims) {
      TxShape ours;
      CHECK(av2enc::cuda::chroma::make_tx_shape(width, height, &ours));
      av2enc::chroma::TxShape cpu;
      CHECK(av2enc::chroma::make_tx_shape(width, height, cpu));
      CHECK_EQ(ours.nominal_w, cpu.nominal_w);
      CHECK_EQ(ours.nominal_h, cpu.nominal_h);
      CHECK_EQ(ours.adj_w, cpu.adj_w);
      CHECK_EQ(ours.adj_h, cpu.adj_h);
      CHECK_EQ(ours.upscale_x, cpu.upscale_x);
      CHECK_EQ(ours.upscale_y, cpu.upscale_y);
      CHECK_EQ(ours.tx_size_ctx, cpu.tx_size_ctx);
      CHECK_EQ(ours.eob_multisize, cpu.eob_multisize);
      CHECK_EQ(ours.seg_eob, cpu.seg_eob);
      CHECK_EQ(ours.tx_pels, cpu.tx_pels);
      CHECK_EQ(ours.dq_denom, cpu.dq_denom);
      CHECK_EQ(ours.row_shift, cpu.row_shift);
      CHECK_EQ(ours.col_shift, cpu.col_shift);
      CHECK_EQ(ours.odd_rect, cpu.odd_rect);
      const double cpu_quant = av2enc::chroma::quant_factor_for(cpu);
      CHECK_DOUBLE(ours.quant_factor, cpu_quant);
    }
  }
  // Rejections mirror the CPU rule.
  TxShape rejected;
  CHECK(!av2enc::cuda::chroma::make_tx_shape(2, 4, &rejected));
  CHECK(!av2enc::cuda::chroma::make_tx_shape(4, 128, &rejected));
  CHECK(!av2enc::cuda::chroma::make_tx_shape(6, 8, &rejected));
}

// ---------------------------------------------------------------------------
// Transform parity over every coded size (all 25 TX sizes), both depths
// ---------------------------------------------------------------------------

void test_transforms() {
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

        // Forward DCT + quantisation (bit-depth independent CPU reference).
        std::vector<double> cpu_coeff(static_cast<std::size_t>(elems), 0.0);
        av2enc::chroma::forward_dct(residual.data(), cpu_shape.adj_w,
                                    cpu_shape.adj_h, cpu_shape.odd_rect,
                                    cpu_coeff.data());
        std::vector<std::int32_t> cpu_levels(static_cast<std::size_t>(elems),
                                             0);
        av2enc::chroma::quantize_coefficients(
            cpu_coeff.data(), cpu_shape.seg_eob,
            av2enc::chroma::quant_factor_for(cpu_shape), cpu_levels.data());

        std::vector<double> dev_coeff(static_cast<std::size_t>(elems), 0.0);
        std::vector<double> tmp(static_cast<std::size_t>(elems), 0.0);
        std::vector<std::int32_t> dev_levels(static_cast<std::size_t>(elems),
                                             0);
        av2enc::cuda::chroma::detail::forward_dct_dev(
            residual.data(), shape.adj_w, shape.adj_h, shape.odd_rect,
            shape.scale_w, shape.scale_h, shape.rect_prescale,
            dev_coeff.data(), tmp.data());
        av2enc::cuda::chroma::detail::quantize_dev(
            dev_coeff.data(), elems, shape.quant_factor, dev_levels.data());

        for (int i = 0; i < elems; ++i) {
          CHECK_DOUBLE(dev_coeff[static_cast<std::size_t>(i)],
                       cpu_coeff[static_cast<std::size_t>(i)]);
          CHECK_EQ(dev_levels[static_cast<std::size_t>(i)],
                   cpu_levels[static_cast<std::size_t>(i)]);
        }

        // Dequantisation.
        std::vector<std::int32_t> cpu_dequant(static_cast<std::size_t>(elems),
                                              0);
        const auto params_opt =
            av2enc::txv::make_tx_params(width, height,
                                        av2enc::txv::TxType::kDctDct);
        CHECK(params_opt.has_value());
        const av2enc::txv::TxParams params = *params_opt;
        if (bit_depth == 8) {
          av2enc::chroma::dequantize_block(cpu_levels.data(), elems,
                                           cpu_shape.dq_denom,
                                           cpu_dequant.data());
        } else {
          av2enc::hbd::dequantize(cpu_levels.data(), params,
                                  cpu_dequant.data(), bit_depth);
        }
        std::vector<std::int32_t> dev_dequant(static_cast<std::size_t>(elems),
                                              0);
        av2enc::cuda::chroma::detail::dequantize_dev(
            dev_levels.data(), elems, shape.dq_denom, bit_depth,
            dev_dequant.data());
        for (int i = 0; i < elems; ++i) {
          CHECK_EQ(dev_dequant[static_cast<std::size_t>(i)],
                   cpu_dequant[static_cast<std::size_t>(i)]);
        }

        // Inverse at the adjusted size.
        std::vector<std::int16_t> dev_back(static_cast<std::size_t>(elems), 0);
        std::vector<std::int32_t> intermediate(
            static_cast<std::size_t>(elems), 0);
        av2enc::cuda::chroma::detail::inverse_dct_dev(
            dev_dequant.data(), shape, bit_depth, dev_back.data(),
            intermediate.data());

        if (bit_depth == 8) {
          std::vector<int> cpu_back(static_cast<std::size_t>(elems), 0);
          av2enc::chroma::inverse_dct(cpu_dequant.data(), cpu_shape,
                                      cpu_back.data());
          for (int i = 0; i < elems; ++i) {
            CHECK_EQ(dev_back[static_cast<std::size_t>(i)],
                     cpu_back[static_cast<std::size_t>(i)]);
          }
        } else {
          // The 10-bit oracle emits the coded-size residual; the adjusted
          // value lives at the top-left of each duplication cell.
          std::vector<std::int16_t> hbd_coded(
              static_cast<std::size_t>(params.coded_w) * params.coded_h, 0);
          av2enc::hbd::inverse_block(cpu_dequant.data(), params,
                                     hbd_coded.data(), bit_depth);
          for (int j = 0; j < shape.adj_h; ++j) {
            for (int i = 0; i < shape.adj_w; ++i) {
              const std::size_t coded_index =
                  static_cast<std::size_t>(j * shape.upscale_y) *
                      static_cast<std::size_t>(params.coded_w) +
                  static_cast<std::size_t>(i * shape.upscale_x);
              CHECK_EQ(dev_back[static_cast<std::size_t>(j) * shape.adj_w + i],
                       hbd_coded[coded_index]);
            }
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Full block pipeline for every format / plane / residual size / depth
// ---------------------------------------------------------------------------

void run_pipeline_case(Format format, int bit_depth, int plane, int nominal_w,
                       int nominal_h, const FrameGeom& geometry, int x0,
                       int y0) {
  (void)format;  // geometry already carries the format
  const auto& plane_geom = geometry.planes[plane];
  const int stride = plane_geom.padded_w;
  const int max_x = stride - 1;
  const int max_y = plane_geom.padded_h - 1;
  const int coded_w = plane_geom.coded_w;
  const int coded_h = plane_geom.coded_h;
  const int have_left = x0 > 0 ? 1 : 0;
  const int have_above = y0 > 0 ? 1 : 0;

  TxShape shape;
  CHECK(av2enc::cuda::chroma::make_tx_shape(nominal_w, nominal_h, &shape));
  av2enc::chroma::TxShape cpu_shape;
  CHECK(av2enc::chroma::make_tx_shape(nominal_w, nominal_h, cpu_shape));

  const int sample_max = (1 << bit_depth) - 1;
  std::vector<std::uint16_t> src(
      static_cast<std::size_t>(stride) * plane_geom.padded_h);
  std::vector<std::uint16_t> recon(
      static_cast<std::size_t>(stride) * plane_geom.padded_h);
  for (auto& value : src) {
    value = static_cast<std::uint16_t>(random_range(0, sample_max));
  }
  for (auto& value : recon) {
    value = static_cast<std::uint16_t>(random_range(0, sample_max));
  }

  // DC prediction.
  const int cpu_pred = dc_predict(recon, stride, coded_w, coded_h, x0, y0,
                                      nominal_w, nominal_h, have_left,
                                      have_above, bit_depth);
  const int dev_pred = av2enc::cuda::chroma::detail::dc_predict_dev(
      recon.data(), stride, coded_w - 1, coded_h - 1, x0, y0, nominal_w,
      nominal_h, have_left, have_above, bit_depth);
  CHECK_EQ(dev_pred, cpu_pred);

  // Forward residual.
  std::vector<std::int16_t> cpu_residual;
  make_residual(src, stride, max_x, max_y, x0, y0, cpu_pred, shape,
                    cpu_residual);
  std::vector<std::int16_t> dev_residual(
      static_cast<std::size_t>(shape.adj_w) * shape.adj_h, 0);
  av2enc::cuda::chroma::detail::make_residual_dev(
      src.data(), stride, max_x, max_y, x0, y0, dev_pred, shape,
      dev_residual.data());
  for (std::size_t i = 0; i < cpu_residual.size(); ++i) {
    CHECK_EQ(dev_residual[i], cpu_residual[i]);
  }

  // Forward transform + quantisation against the CPU chroma module.
  const int elems = shape.adj_w * shape.adj_h;
  std::vector<double> cpu_coeff(static_cast<std::size_t>(elems), 0.0);
  av2enc::chroma::forward_dct(cpu_residual.data(), cpu_shape.adj_w,
                              cpu_shape.adj_h, cpu_shape.odd_rect,
                              cpu_coeff.data());
  std::vector<std::int32_t> cpu_levels(static_cast<std::size_t>(elems), 0);
  av2enc::chroma::quantize_coefficients(
      cpu_coeff.data(), cpu_shape.seg_eob,
      av2enc::chroma::quant_factor_for(cpu_shape), cpu_levels.data());

  std::vector<double> dev_coeff(static_cast<std::size_t>(elems), 0.0);
  std::vector<double> tmp(static_cast<std::size_t>(elems), 0.0);
  std::vector<std::int32_t> dev_levels(static_cast<std::size_t>(elems), 0);
  av2enc::cuda::chroma::detail::forward_dct_dev(
      dev_residual.data(), shape.adj_w, shape.adj_h, shape.odd_rect,
      shape.scale_w, shape.scale_h, shape.rect_prescale, dev_coeff.data(),
      tmp.data());
  av2enc::cuda::chroma::detail::quantize_dev(
      dev_coeff.data(), elems, shape.quant_factor, dev_levels.data());
  for (int i = 0; i < elems; ++i) {
    CHECK_DOUBLE(dev_coeff[static_cast<std::size_t>(i)],
                 cpu_coeff[static_cast<std::size_t>(i)]);
    CHECK_EQ(dev_levels[static_cast<std::size_t>(i)],
             cpu_levels[static_cast<std::size_t>(i)]);
  }

  // Dequantisation: 8-bit chroma module, 10-bit hbd oracle.
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
  std::vector<std::int32_t> dev_dequant(static_cast<std::size_t>(elems), 0);
  av2enc::cuda::chroma::detail::dequantize_dev(
      dev_levels.data(), elems, shape.dq_denom, bit_depth,
      dev_dequant.data());
  for (int i = 0; i < elems; ++i) {
    CHECK_EQ(dev_dequant[static_cast<std::size_t>(i)],
             cpu_dequant[static_cast<std::size_t>(i)]);
  }

  // Inverse.
  std::vector<std::int32_t> cpu_back(static_cast<std::size_t>(elems), 0);
  std::vector<std::int16_t> dev_back(static_cast<std::size_t>(elems), 0);
  std::vector<std::int32_t> intermediate(static_cast<std::size_t>(elems), 0);
  if (bit_depth == 8) {
    std::vector<int> back(static_cast<std::size_t>(elems), 0);
    av2enc::chroma::inverse_dct(cpu_dequant.data(), cpu_shape, back.data());
    for (int i = 0; i < elems; ++i) {
      cpu_back[static_cast<std::size_t>(i)] = back[static_cast<std::size_t>(i)];
    }
    av2enc::cuda::chroma::detail::inverse_dct_dev(
        dev_dequant.data(), shape, bit_depth, dev_back.data(),
        intermediate.data());
  } else {
    std::vector<std::int16_t> coded(
        static_cast<std::size_t>(params.coded_w) * params.coded_h, 0);
    av2enc::hbd::inverse_block(cpu_dequant.data(), params, coded.data(),
                               bit_depth);
    av2enc::cuda::chroma::detail::inverse_dct_dev(
        dev_dequant.data(), shape, bit_depth, dev_back.data(),
        intermediate.data());
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
  for (int i = 0; i < elems; ++i) {
    CHECK_EQ(dev_back[static_cast<std::size_t>(i)],
             cpu_back[static_cast<std::size_t>(i)]);
  }

  // Reconstruction into the padded plane.
  std::vector<std::uint16_t> cpu_recon = recon;
  std::vector<std::uint16_t> dev_recon = recon;
  reconstruct(cpu_back, cpu_pred, shape, bit_depth, cpu_recon, stride,
                  max_x, max_y, x0, y0);
  av2enc::cuda::chroma::detail::reconstruct_block_dev(
      dev_back.data(), dev_pred, shape, bit_depth, dev_recon.data(), stride,
      max_x, max_y, x0, y0);
  for (std::size_t i = 0; i < cpu_recon.size(); ++i) {
    CHECK_EQ(dev_recon[i], cpu_recon[i]);
  }
}

void test_pipelines() {
  const Format formats[] = {Format::k420, Format::k422, Format::k444,
                            Format::k400};
  const std::pair<int, int> luma_blocks[] = {
      {64, 64}, {32, 32}, {32, 16}, {32, 8}};
  for (Format format : formats) {
    const FrameGeom geometry =
        av2enc::cuda::chroma::make_frame_geometry(64, 64, format);
    for (int bit_depth : {8, 10}) {
      for (int plane = 0; plane < geometry.num_planes; ++plane) {
        for (const auto& block : luma_blocks) {
          int nominal_w = block.first;
          int nominal_h = block.second;
          if (plane > 0) {
            const ResidualSize residual =
                av2enc::cuda::chroma::chroma_residual_size(
                    block.first, block.second, format);
            CHECK(residual.valid);
            if (!residual.valid) continue;
            nominal_w = residual.width;
            nominal_h = residual.height;
          }
          const auto& plane_geom = geometry.planes[plane];
          if (nominal_w > plane_geom.padded_w ||
              nominal_h > plane_geom.padded_h) {
            continue;  // not reachable for the 64x64 test frame
          }
          const int max_x = std::min(4, plane_geom.coded_w - nominal_w);
          const int max_y = std::min(4, plane_geom.coded_h - nominal_h);
          run_pipeline_case(format, bit_depth, plane, nominal_w, nominal_h,
                            geometry, 0, 0);
          if (max_x > 0) {
            run_pipeline_case(format, bit_depth, plane, nominal_w, nominal_h,
                              geometry, max_x, 0);
          }
          if (max_y > 0) {
            run_pipeline_case(format, bit_depth, plane, nominal_w, nominal_h,
                              geometry, 0, max_y);
          }
          if (max_x > 0 && max_y > 0) {
            run_pipeline_case(format, bit_depth, plane, nominal_w, nominal_h,
                              geometry, max_x, max_y);
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Depth-clipping copy, every present plane / format / depth
// ---------------------------------------------------------------------------

void test_copy() {
  const Format formats[] = {Format::k420, Format::k422, Format::k444,
                            Format::k400};
  for (Format format : formats) {
    const FrameGeom geometry =
        av2enc::cuda::chroma::make_frame_geometry(64, 64, format);
    for (int bit_depth : {8, 10}) {
      for (int plane = 0; plane < geometry.num_planes; ++plane) {
        const auto& plane_geom = geometry.planes[plane];
        const int stride = plane_geom.padded_w;
        const int width = std::min(plane_geom.padded_w, 17);
        const int height = std::min(plane_geom.padded_h, 13);
        std::vector<std::uint16_t> src(
            static_cast<std::size_t>(stride) * plane_geom.padded_h);
        // Deliberately include out-of-range values so the depth clip is
        // exercised (the device stores uint16 and must clip to BitDepth).
        for (auto& value : src) {
          value = static_cast<std::uint16_t>(
              random_range(0, (1 << (bit_depth + 1))));
        }
        std::vector<std::uint16_t> dst(src.size(), 0);
        av2enc::cuda::chroma::detail::copy_rect_dev(
            src.data(), stride, dst.data(), stride, width, height, bit_depth);
        for (int y = 0; y < height; ++y) {
          for (int x = 0; x < width; ++x) {
            const std::size_t index =
                static_cast<std::size_t>(y) * stride +
                static_cast<std::size_t>(x);
            CHECK_EQ(dst[index],
                     av2enc::cuda::chroma::testref::clip1(
                         static_cast<int>(src[index]), bit_depth));
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Arena contract
// ---------------------------------------------------------------------------

void test_arena() {
  const Config config{64, 64, Format::k422, 8, 4};
  const std::uint64_t required =
      av2enc::cuda::chroma::required_device_bytes(config);
  const std::uint64_t persistent =
      av2enc::cuda::chroma::persistent_device_bytes(config);
  const std::uint64_t scratch =
      av2enc::cuda::chroma::scratch_device_bytes(config);
  CHECK(required > 0);
  CHECK_EQ(required, persistent + scratch);
  CHECK_EQ(required % Arena::kAlignment, 0);
  CHECK_EQ(persistent % Arena::kAlignment, 0);
  CHECK_EQ(scratch % Arena::kAlignment, 0);

  // Invalid configurations report zero bytes and kInvalidArgument.
  Config invalid = config;
  invalid.max_blocks = 0;
  CHECK_EQ(av2enc::cuda::chroma::required_device_bytes(invalid), 0);
  invalid = config;
  invalid.bit_depth = 12;
  CHECK_EQ(av2enc::cuda::chroma::required_device_bytes(invalid), 0);
  invalid = config;
  invalid.format = static_cast<Format>(99);
  CHECK_EQ(av2enc::cuda::chroma::required_device_bytes(invalid), 0);

  Arena arena;
  CHECK(av2enc::cuda::is_ok(arena.init(required + 4096)));
  Workspace workspace;
  std::string error;
  std::uint64_t needed = 0;
  const ChromaStatus status =
      workspace.init(arena, config, &error, &needed);
  CHECK_EQ(static_cast<int>(status.code),
           static_cast<int>(ChromaStatus::Code::kOk));
  CHECK_EQ(needed, required);
  CHECK(workspace.initialized());
  CHECK_EQ(arena.used(), persistent);
  CHECK_EQ(workspace.persistent_bytes(), persistent);
  CHECK_EQ(workspace.scratch_bytes(), scratch);

  // Two scratch cycles: flat used(), stable peak (the launchers use exactly
  // this allocate/rewind bracket).
  const std::uint64_t mark = arena.mark();
  for (int cycle = 0; cycle < 2; ++cycle) {
    void* raw = nullptr;
    const auto allocated = arena.allocate_checked(scratch, &raw);
    CHECK(av2enc::cuda::is_ok(allocated));
    CHECK(raw != nullptr);
    CHECK_EQ(arena.used(), mark + scratch);
    CHECK(arena.rewind(mark));
    CHECK_EQ(arena.used(), mark);
  }
  const std::uint64_t peak = arena.peak_used();
  CHECK(peak >= mark + scratch);

  // One byte short: kArenaTooSmall, exact requirement, no residue.
  Arena small;
  CHECK(av2enc::cuda::is_ok(small.init(required - 1)));
  Workspace short_ws;
  std::string short_error;
  std::uint64_t short_needed = 0;
  const ChromaStatus short_status =
      short_ws.init(small, config, &short_error, &short_needed);
  CHECK_EQ(static_cast<int>(short_status.code),
           static_cast<int>(ChromaStatus::Code::kArenaTooSmall));
  CHECK_EQ(short_status.required_bytes, required);
  CHECK_EQ(short_needed, required);
  CHECK_EQ(small.used(), 0);
  CHECK(!short_ws.initialized());

  // Invalid config through init.
  Workspace invalid_ws;
  CHECK_EQ(static_cast<int>(invalid_ws.init(arena, invalid).code),
           static_cast<int>(ChromaStatus::Code::kInvalidArgument));

  // All four formats size and initialise; monochrome allocates one plane.
  for (Format format : {Format::k420, Format::k422, Format::k444,
                        Format::k400}) {
    Config cfg = config;
    cfg.format = format;
    const std::uint64_t need =
        av2enc::cuda::chroma::required_device_bytes(cfg);
    Arena probe;
    CHECK(av2enc::cuda::is_ok(probe.init(need)));
    Workspace ws;
    CHECK_EQ(static_cast<int>(ws.init(probe, cfg).code),
             static_cast<int>(ChromaStatus::Code::kOk));
    CHECK_EQ(probe.used(),
             av2enc::cuda::chroma::persistent_device_bytes(cfg));
    if (format == Format::k400) {
      CHECK(ws.src_plane(1) == nullptr);
      CHECK(ws.recon_plane(2) == nullptr);
    } else {
      CHECK(ws.src_plane(1) != nullptr);
      CHECK(ws.src_plane(2) != nullptr);
    }
  }

  std::printf(
      "  arena: required=%llu persistent=%llu scratch=%llu peak=%llu\n",
      static_cast<unsigned long long>(required),
      static_cast<unsigned long long>(persistent),
      static_cast<unsigned long long>(scratch),
      static_cast<unsigned long long>(peak));
}

}  // namespace

int main() {
  std::printf("av2enc CUDA chroma-format emulation test\n");
  test_geometry();
  std::printf("  geometry: checks=%lld\n", g_checks);
  test_residual_sizes();
  std::printf("  Subsampled_Size: checks=%lld\n", g_checks);
  test_tx_shapes();
  std::printf("  tx shapes: checks=%lld\n", g_checks);
  test_transforms();
  std::printf("  transforms (25 sizes x 2 depths): checks=%lld\n", g_checks);
  test_pipelines();
  std::printf("  format/plane pipelines: checks=%lld\n", g_checks);
  test_copy();
  std::printf("  depth-clipping copy: checks=%lld\n", g_checks);
  test_arena();

  if (g_failures == 0) {
    std::printf("chroma_formats emulation: %lld checks, PASS\n", g_checks);
    return 0;
  }
  std::printf("chroma_formats emulation: %lld checks, %d FAILURES\n", g_checks,
              g_failures);
  return 1;
}

#include "pipeline/still/high_bitdepth/high_bitdepth_still.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "cuda/pipeline/still/hbd/hbd_cpu.h"
#include "pipeline/still/high_bitdepth/high_bitdepth_headers.h"
#include "codec/transform/variants/tx_variants.h"

namespace av2enc::high_bitdepth {
namespace {

int round2(int value, int n) {
  if (n <= 0) return value;
  return (value + (1 << (n - 1))) >> n;
}

int floor_log2(int value) {
  int result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}

// §7.13.2.11 / §7.13.3.22.  M10 denominators are powers of two, for which
// resolve_divisor() gives f = 0 and Div_Lut[0] = 512.
int approx_divide(int numerator, int denominator) {
  const int shift = 9 + floor_log2(denominator);
  return round2(numerator * 512, shift);
}

int clip3(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

int padded_extent(int coded) { return (coded + 63) & ~63; }

struct PlaneView {
  std::uint16_t* samples = nullptr;
  int stride = 0;
  int coded_width = 0;
  int coded_height = 0;
};

// §7.13.2.10 DC prediction.  Reads only reconstructed samples; coordinates are
// clipped to the *coded* plane, because the decoder reconstructs the full
// MiCols*4 x MiRows*4 plane before cropping to the visible rectangle.
void dc_predict(const PlaneView& recon, int x0, int y0, int w, int h,
                bool have_left, bool have_above, std::uint16_t* pred) {
  const int max_x = recon.coded_width - 1;
  const int max_y = recon.coded_height - 1;
  int sum_left = 0;
  int sum_above = 0;
  if (have_left) {
    for (int i = 0; i < h; ++i) {
      const int y = clip3(0, max_y, y0 + i);
      sum_left += recon.samples[static_cast<std::size_t>(y) * recon.stride +
                                static_cast<std::size_t>(x0 - 1)];
    }
  }
  if (have_above) {
    for (int j = 0; j < w; ++j) {
      const int x = clip3(0, max_x, x0 + j);
      sum_above += recon.samples[static_cast<std::size_t>(y0 - 1) *
                                     recon.stride +
                                 static_cast<std::size_t>(x)];
    }
  }

  int value = hbd::dc_pred_bias(10);  // 512
  if (have_left && have_above) {
    value = approx_divide(sum_left + sum_above, w + h);
  } else if (have_left) {
    value = round2(sum_left, floor_log2(h));
  } else if (have_above) {
    value = round2(sum_above, floor_log2(w));
  }
  value = hbd::clip1(value, 10);
  for (int i = 0; i < w * h; ++i) {
    pred[i] = static_cast<std::uint16_t>(value);
  }
}

void append_scan_levels(const std::int32_t* levels, int w, int h, int eob,
                        std::vector<std::int32_t>* out) {
  std::vector<std::uint16_t> scan(static_cast<std::size_t>(w) * h);
  hbd::build_scan(w, h, scan.data());
  for (int c = 0; c < eob; ++c) {
    out->push_back(levels[scan[static_cast<std::size_t>(c)]]);
  }
}

}  // namespace

std::vector<HighBitDepthBlockDesc> make_block_plan(std::uint32_t width,
                                          std::uint32_t height) {
  std::vector<HighBitDepthBlockDesc> blocks;
  const int mi_cols = static_cast<int>(fmt::mi_cols(width));
  const int mi_rows = static_cast<int>(fmt::mi_rows(height));
  for (int r = 0; r < mi_rows; r += 16) {
    for (int c = 0; c < mi_cols; c += 16) {
      const bool edge = (r + 16 > mi_rows) || (c + 16 > mi_cols);
      if (edge) {
        blocks.push_back(HighBitDepthBlockDesc{static_cast<std::uint16_t>(r),
                                      static_cast<std::uint16_t>(c), 64});
      } else {
        // PARTITION_VERT -> two 32x64 columns; each PARTITION_HORZ -> two
        // 32x32 leaves.  Decode order is TL, BL, TR, BR.
        for (int cc = c; cc < c + 16; cc += 8) {
          for (int rr = r; rr < r + 16; rr += 8) {
            blocks.push_back(HighBitDepthBlockDesc{static_cast<std::uint16_t>(rr),
                                          static_cast<std::uint16_t>(cc), 32});
          }
        }
      }
    }
  }
  return blocks;
}

bool high_bitdepth_geometry_supported(std::uint32_t width, std::uint32_t height,
                            std::string* reason) {
  if (width == 0 || height == 0) {
    if (reason != nullptr) *reason = "M10 dimensions must be non-zero";
    return false;
  }
  const std::uint32_t mi_cols = fmt::mi_cols(width);
  const std::uint32_t mi_rows = fmt::mi_rows(height);
  const std::uint32_t col_remainder = mi_cols % 16U;
  const std::uint32_t row_remainder = mi_rows % 16U;
  if (col_remainder != 0U && col_remainder <= 8U) {
    if (reason != nullptr) {
      *reason = "M10 fixed tree needs the last superblock column to have more "
                "than 8 MI; this width forces an implied 64x64 partition";
    }
    return false;
  }
  if (row_remainder != 0U && row_remainder <= 8U) {
    if (reason != nullptr) {
      *reason = "M10 fixed tree needs the last superblock row to have more "
                "than 8 MI; this height forces an implied 64x64 partition";
    }
    return false;
  }
  return true;
}

bool encode_high_bitdepth_cpu(const fmt::Picture& picture, const HighBitDepthOptions& options,
                    HighBitDepthResult& result, std::string& error) {
  error.clear();
  result = HighBitDepthResult{};
  if (picture.format.depth != fmt::BitDepth::k10 ||
      picture.format.chroma != fmt::ChromaFormat::k420) {
    error = "M10 requires a 10-bit 4:2:0 picture";
    return false;
  }
  if (!high_bitdepth_geometry_supported(picture.format.width, picture.format.height,
                              &error)) {
    return false;
  }
  if (!fmt::samples_in_range(picture)) {
    error = "M10 input samples must be in [0,1023]";
    return false;
  }
  const auto config_check = [&]() {
    HighBitDepthConfig config;
    config.max_frame_width = static_cast<std::uint16_t>(picture.format.width);
    config.max_frame_height = static_cast<std::uint16_t>(picture.format.height);
    config.base_q_idx = options.base_q_idx;
    config.disable_cdf_update = options.disable_cdf_update;
    std::string reason;
    if (!validate_config(config, &reason)) {
      error = reason;
      return false;
    }
    return true;
  };
  if (!config_check()) return false;

  const int mi_cols = static_cast<int>(picture.geometry.mi_cols);
  const int mi_rows = static_cast<int>(picture.geometry.mi_rows);
  const int coded_luma_w = mi_cols * 4;
  const int coded_luma_h = mi_rows * 4;
  const int coded_chroma_w = coded_luma_w / 2;
  const int coded_chroma_h = coded_luma_h / 2;

  // Decoder CurrFrame starts at zero.  Storage is 64-aligned so a partial
  // 64x64 edge leaf can be reconstructed without touching unrelated memory.
  HighBitDepthPlane recon[3];
  recon[0].width = padded_extent(coded_luma_w);
  recon[0].height = padded_extent(coded_luma_h);
  recon[1].width = recon[2].width = recon[0].width / 2;
  recon[1].height = recon[2].height = recon[0].height / 2;
  for (int plane = 0; plane < 3; ++plane) {
    recon[plane].coded_width = plane == 0 ? coded_luma_w : coded_chroma_w;
    recon[plane].coded_height = plane == 0 ? coded_luma_h : coded_chroma_h;
    const fmt::PlaneGeometry& pg = picture.geometry.planes[plane];
    recon[plane].visible_width = static_cast<int>(pg.visible_width);
    recon[plane].visible_height = static_cast<int>(pg.visible_height);
    recon[plane].samples.assign(
        static_cast<std::size_t>(recon[plane].width) *
            static_cast<std::size_t>(recon[plane].height),
        0);
  }

  const std::vector<HighBitDepthBlockDesc> plan =
      make_block_plan(picture.format.width, picture.format.height);

  std::vector<HighBitDepthBlock> blocks;
  blocks.reserve(plan.size());
  std::vector<std::int32_t> levels;
  std::vector<std::uint16_t> pred(64 * 64);

  for (const HighBitDepthBlockDesc& desc : plan) {
    const int x0 = static_cast<int>(desc.mi_col) * 4;
    const int y0 = static_cast<int>(desc.mi_row) * 4;
    const int luma = desc.luma_size;
    const int chroma = luma / 2;
    const int coded_w[3] = {luma, chroma, chroma};
    const int coded_h[3] = {luma, chroma, chroma};

    HighBitDepthBlock block;
    block.mi_row = desc.mi_row;
    block.mi_col = desc.mi_col;
    block.luma_size = desc.luma_size;
    block.levels_offset = static_cast<std::uint32_t>(levels.size());

    for (int plane = 0; plane < 3; ++plane) {
      const auto params_opt = txv::make_tx_params(
          coded_w[plane], coded_h[plane], txv::TxType::kDctDct);
      if (!params_opt.has_value()) {
        error = "unsupported M10 transform size";
        return false;
      }
      const txv::TxParams& params = *params_opt;
      block.tx_size[plane] = static_cast<std::uint8_t>(params.tx_size);

      const int px0 = plane == 0 ? x0 : x0 / 2;
      const int py0 = plane == 0 ? y0 : y0 / 2;
      const bool have_left = px0 > 0;
      const bool have_above = py0 > 0;
      PlaneView view{recon[plane].samples.data(), recon[plane].width,
                     recon[plane].coded_width, recon[plane].coded_height};
      dc_predict(view, px0, py0, coded_w[plane], coded_h[plane], have_left,
                 have_above, pred.data());

      const fmt::PlaneBuffer& src = picture.planes[plane];
      hbd::BlockResult br;
      if (!hbd::encode_block(
              src.samples.data(), static_cast<int>(src.stride),
              static_cast<int>(src.visible_width),
              static_cast<int>(src.visible_height), px0, py0, pred.data(),
              params, 10, &br)) {
        error = "hbd::encode_block failed";
        return false;
      }
      if (std::getenv("HIGH_BITDEPTH_ZERO_RESIDUAL") != nullptr) {
        // Debug isolation: code an all-skip frame and use DC prediction as the
        // reconstruction.  Not a production mode.
        br.eob = 0;
        std::fill(br.recon.begin(), br.recon.end(), 0);
        for (int i = 0; i < coded_w[plane] * coded_h[plane]; ++i) {
          br.recon[static_cast<std::size_t>(i)] = pred[static_cast<std::size_t>(i)];
        }
      }
      block.eob[plane] = static_cast<std::uint32_t>(br.eob);
      append_scan_levels(br.levels.data(), params.w, params.h, br.eob, &levels);

      // Store exactly the part of the coded-size reconstruction inside the
      // coded plane, like decode_block().  Samples past the frame's coded
      // extent are reconstructed by the decoder but never stored.
      for (int y = 0; y < coded_h[plane]; ++y) {
        const int py = py0 + y;
        if (py >= recon[plane].coded_height) break;
        for (int x = 0; x < coded_w[plane]; ++x) {
          const int px = px0 + x;
          if (px >= recon[plane].coded_width) break;
          recon[plane].samples[static_cast<std::size_t>(py) *
                                   recon[plane].width +
                               static_cast<std::size_t>(px)] =
              br.recon[static_cast<std::size_t>(y) * coded_w[plane] + x];
        }
      }
    }
    blocks.push_back(block);
  }

  std::vector<std::uint8_t> frame;
  std::uint32_t tile_bytes = 0;
  if (!assemble_high_bitdepth_stream(picture, options, blocks.data(), blocks.size(),
                           levels.data(), levels.size(), &frame, &tile_bytes,
                           &error)) {
    return false;
  }

  result.tile_bytes = tile_bytes;
  result.frame_bytes = static_cast<std::uint32_t>(frame.size());
  result.frame = std::move(frame);
  result.blocks = std::move(blocks);
  result.levels = std::move(levels);
  for (int plane = 0; plane < 3; ++plane) {
    result.recon[plane] = std::move(recon[plane]);
  }
  return true;
}

bool convert_8bit_420_to_picture(const std::uint8_t* y, const std::uint8_t* u,
                                 const std::uint8_t* v, std::uint32_t width,
                                 std::uint32_t height, std::uint32_t y_stride,
                                 fmt::Picture* picture, std::string* error) {
  if (picture == nullptr || y == nullptr || u == nullptr || v == nullptr) {
    if (error != nullptr) *error = "null 8-bit plane";
    return false;
  }
  const std::uint32_t cw = (width + 1U) / 2U;
  fmt::Source420_8 source;
  source.width = width;
  source.height = height;
  source.y = fmt::ConstPlane8{y, y_stride};
  source.u = fmt::ConstPlane8{u, cw};
  source.v = fmt::ConstPlane8{v, cw};
  const auto converted = fmt::convert_420_8_to_picture(
      source, fmt::ChromaFormat::k420, fmt::BitDepth::k10, *picture);
  if (!converted.ok()) {
    if (error != nullptr) *error = converted.error;
    return false;
  }
  return true;
}

}  // namespace av2enc::high_bitdepth

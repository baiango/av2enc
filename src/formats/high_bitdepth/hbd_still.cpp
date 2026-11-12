#include "formats/high_bitdepth/hbd_still.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "cuda/pipeline/still/hbd/hbd_cpu.h"
#include "formats/high_bitdepth/hbd_depth.h"
#include "formats/high_bitdepth/hbd_headers.h"
#include "formats/high_bitdepth/hbd_reader.h"
#include "pipeline/still/high_bitdepth/high_bitdepth_headers.h"
#include "codec/transform/variants/tx_variants.h"

namespace av2enc::hbd {
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

// §7.13.2.11 approx_divide() with the normative §7.13.3.22 Div_Lut.  The M10
// DC block sizes (w+h 32/64/128) all resolve to Div_Lut[0] = 512, but 4:2:2
// chroma blocks (16x32 -> 48) and any non-power-of-two sum need the full table.
constexpr int kDivLutBits = 7;
constexpr int kDivLutPrecBits = 9;
constexpr int kDivLut[129] = {
    512, 508, 504, 500, 496, 493, 489, 485, 482, 478, 475, 471, 468, 465, 462,
    458, 455, 452, 449, 446, 443, 440, 437, 434, 431, 428, 426, 423, 420, 417,
    415, 412, 410, 407, 405, 402, 400, 397, 395, 392, 390, 388, 386, 383, 381,
    379, 377, 374, 372, 370, 368, 366, 364, 362, 360, 358, 356, 354, 352, 350,
    349, 347, 345, 343, 341, 340, 338, 336, 334, 333, 331, 329, 328, 326, 324,
    323, 321, 320, 318, 317, 315, 314, 312, 311, 309, 308, 306, 305, 303, 302,
    301, 299, 298, 297, 295, 294, 293, 291, 290, 289, 287, 286, 285, 284, 282,
    281, 280, 279, 278, 277, 275, 274, 273, 272, 271, 270, 269, 267, 266, 265,
    264, 263, 262, 261, 260, 259, 258, 257, 256};

int approx_divide(int numerator, int denominator) {
  const int abs_denominator = denominator < 0 ? -denominator : denominator;
  if (abs_denominator == 0) return 0;
  const int n = floor_log2(abs_denominator);
  const int e = abs_denominator - (1 << n);
  const int f =
      n > kDivLutBits ? round2(e, n - kDivLutBits)
                      : e << (kDivLutBits - n);
  const int scale = kDivLut[f < 128 ? f : 128];
  return round2(numerator * (denominator < 0 ? -scale : scale),
                n + kDivLutPrecBits);
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

// §7.13.2.10 DC prediction with the 10-bit neutral value (512) and Clip1 1023.
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

  int value = depth::sample_ranges().dc_bias;  // 512
  if (have_left && have_above) {
    value = approx_divide(sum_left + sum_above, w + h);
  } else if (have_left) {
    value = round2(sum_left, floor_log2(h));
  } else if (have_above) {
    value = round2(sum_above, floor_log2(w));
  }
  value = depth::clip1(value);
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

// §7.14.4 dequant with the frame qlookup step and the 10-bit clip.
void dequantize_qindex(const std::int32_t* levels, const txv::TxParams& params,
                       int quant_step, std::int32_t* dequant) {
  const int n = params.w * params.h;
  for (int i = 0; i < n; ++i) {
    const std::int32_t qc = levels[i];
    const std::int32_t abs_qc = qc < 0 ? -qc : qc;
    const std::int32_t dq_high =
        (abs_qc * static_cast<std::int32_t>(quant_step)) & 0xFFFFFF;
    const std::int32_t dq = txv::round2(dq_high, txv::kQuantTableBits) /
                            static_cast<std::int32_t>(params.dq_denom);
    const std::int32_t value = qc < 0 ? -dq : dq;
    dequant[i] = depth::clip_dequant(value);
  }
}

}  // namespace

bool qindex_encodable(std::uint16_t base_q_idx, std::string* reason) {
  const auto fail = [reason](const char* text) {
    if (reason != nullptr) *reason = text;
    return false;
  };
  if (base_q_idx == 0) {
    return fail("base_q_idx = 0 is lossless; not implemented");
  }
  const int maxq = depth::max_q(depth::kBitDepth);
  if (base_q_idx > static_cast<unsigned>(maxq)) {
    return fail("base_q_idx exceeds MaxQ = 303 at 10-bit");
  }
  return true;
}

bool encode_still(const fmt::Picture& picture, const HbdOptions& options,
                  HbdResult& result, std::string& error) {
  error.clear();
  result = HbdResult{};
  if (picture.format.depth != fmt::BitDepth::k10) {
    error = "hbd requires 10-bit samples";
    return false;
  }
  if (fmt::chroma_format_idc(picture.format.chroma) < 0) {
    error = "hbd requires a valid chroma format";
    return false;
  }
  if (!av2enc::high_bitdepth::high_bitdepth_geometry_supported(picture.format.width, picture.format.height,
                                   &error)) {
    return false;
  }
  if (!fmt::samples_in_range(picture)) {
    error = "hbd input samples must be in [0,1023]";
    return false;
  }
  if (!qindex_encodable(options.base_q_idx, &error)) return false;

  const int quant_step = depth::qlookup(options.base_q_idx);
  const int num_planes = fmt::num_planes(picture.format.chroma);
  const int sub_x = fmt::subsampling_x(picture.format.chroma);
  const int sub_y = fmt::subsampling_y(picture.format.chroma);

  const int mi_cols = static_cast<int>(picture.geometry.mi_cols);
  const int mi_rows = static_cast<int>(picture.geometry.mi_rows);
  const int coded_luma_w = mi_cols * 4;
  const int coded_luma_h = mi_rows * 4;

  av2enc::high_bitdepth::HighBitDepthPlane recon[3];
  const int padded_luma_w = padded_extent(coded_luma_w);
  const int padded_luma_h = padded_extent(coded_luma_h);
  for (int plane = 0; plane < 3; ++plane) {
    if (plane >= num_planes) continue;
    const fmt::PlaneGeometry& pg = picture.geometry.planes[plane];
    const int shift_x = plane == 0 ? 0 : sub_x;
    const int shift_y = plane == 0 ? 0 : sub_y;
    recon[plane].coded_width = coded_luma_w >> shift_x;
    recon[plane].coded_height = coded_luma_h >> shift_y;
    recon[plane].width = padded_luma_w >> shift_x;
    recon[plane].height = padded_luma_h >> shift_y;
    recon[plane].visible_width = static_cast<int>(pg.visible_width);
    recon[plane].visible_height = static_cast<int>(pg.visible_height);
    recon[plane].samples.assign(
        static_cast<std::size_t>(recon[plane].width) *
            static_cast<std::size_t>(recon[plane].height),
        0);
  }

  const std::vector<av2enc::high_bitdepth::HighBitDepthBlockDesc> plan =
      av2enc::high_bitdepth::make_block_plan(picture.format.width, picture.format.height);

  std::vector<av2enc::high_bitdepth::HighBitDepthBlock> blocks;
  blocks.reserve(plan.size());
  std::vector<std::int32_t> levels;
  std::vector<std::uint16_t> pred(64 * 64);
  std::vector<std::int16_t> residual;
  std::vector<std::int16_t> residual_adj;
  std::vector<std::int32_t> block_levels;
  std::vector<std::int32_t> dequant;
  std::vector<std::int16_t> residual_out;
  std::vector<double> coeff;

  for (const av2enc::high_bitdepth::HighBitDepthBlockDesc& desc : plan) {
    const int x0 = static_cast<int>(desc.mi_col) * 4;
    const int y0 = static_cast<int>(desc.mi_row) * 4;
    const int luma = desc.luma_size;
    const int coded_w[3] = {luma, luma >> sub_x, luma >> sub_x};
    const int coded_h[3] = {luma, luma >> sub_y, luma >> sub_y};

    av2enc::high_bitdepth::HighBitDepthBlock block;
    block.mi_row = desc.mi_row;
    block.mi_col = desc.mi_col;
    block.luma_size = desc.luma_size;
    block.levels_offset = static_cast<std::uint32_t>(levels.size());
    block.tx_size[0] = block.tx_size[1] = block.tx_size[2] = 0;
    block.eob[0] = block.eob[1] = block.eob[2] = 0;

    for (int plane = 0; plane < num_planes; ++plane) {
      auto params_opt = txv::make_tx_params(coded_w[plane], coded_h[plane],
                                            txv::TxType::kDctDct);
      if (!params_opt.has_value()) {
        error = "unsupported hbd transform size";
        return false;
      }
      txv::TxParams params = *params_opt;
      block.tx_size[plane] = static_cast<std::uint8_t>(params.tx_size);

      // §7.14.2: the transform component bakes the QP-90 step (528) into
      // quant_factor.  Scale it so forward quantisation matches qlookup.
      if (quant_step != txv::kQuantStep) {
        params.quant_factor *=
            static_cast<double>(txv::kQuantStep) /
            static_cast<double>(quant_step);
      }

      const int px0 = plane == 0 ? x0 : (x0 >> sub_x);
      const int py0 = plane == 0 ? y0 : (y0 >> sub_y);
      const bool have_left = px0 > 0;
      const bool have_above = py0 > 0;
      PlaneView view{recon[plane].samples.data(), recon[plane].width,
                     recon[plane].coded_width, recon[plane].coded_height};
      dc_predict(view, px0, py0, coded_w[plane], coded_h[plane], have_left,
                 have_above, pred.data());

      // Coded-size residual from the visible source plane, with the decoder's
      // edge clamp (same rule as hbd::encode_block).
      const fmt::PlaneBuffer& src = picture.planes[plane];
      const int cw = coded_w[plane];
      const int ch = coded_h[plane];
      const int coded_pels = cw * ch;
      residual.assign(static_cast<std::size_t>(coded_pels), 0);
      for (int y = 0; y < ch; ++y) {
        const int sy = std::clamp(py0 + y, 0,
                                  static_cast<int>(src.visible_height) - 1);
        for (int x = 0; x < cw; ++x) {
          const int sx = std::clamp(px0 + x, 0,
                                    static_cast<int>(src.visible_width) - 1);
          const int s = static_cast<int>(
              src.samples[static_cast<std::size_t>(sy) * src.stride +
                          static_cast<std::size_t>(sx)]);
          const int p = static_cast<int>(
              pred[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
                   static_cast<std::size_t>(x)]);
          residual[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
                   static_cast<std::size_t>(x)] =
              static_cast<std::int16_t>(s - p);
        }
      }

      const int adj_area = params.w * params.h;
      residual_adj.assign(static_cast<std::size_t>(adj_area), 0);
      block_levels.assign(static_cast<std::size_t>(adj_area), 0);
      coeff.assign(static_cast<std::size_t>(adj_area), 0.0);
      hbd::downsample(residual.data(), params, residual_adj.data());
      hbd::forward(residual_adj.data(), params, coeff.data(),
                   block_levels.data());
      const int eob = hbd::eob_from_levels(block_levels.data(), params.w,
                                           params.h);
      block.eob[plane] = static_cast<std::uint32_t>(eob);

      dequant.assign(static_cast<std::size_t>(adj_area), 0);
      dequantize_qindex(block_levels.data(), params, quant_step,
                        dequant.data());
      residual_out.assign(static_cast<std::size_t>(coded_pels), 0);
      hbd::inverse_block(dequant.data(), params, residual_out.data(),
                         depth::kBitDepth);

      append_scan_levels(block_levels.data(), params.w, params.h, eob, &levels);

      // Reconstruction = Clip1(prediction + residual), stored exactly inside
      // the coded plane like decode_block().
      for (int y = 0; y < ch; ++y) {
        const int py = py0 + y;
        if (py >= recon[plane].coded_height) break;
        for (int x = 0; x < cw; ++x) {
          const int px = px0 + x;
          if (px >= recon[plane].coded_width) break;
          const int value =
              static_cast<int>(pred[static_cast<std::size_t>(y) *
                                        static_cast<std::size_t>(cw) +
                                    static_cast<std::size_t>(x)]) +
              static_cast<int>(residual_out[static_cast<std::size_t>(y) *
                                                static_cast<std::size_t>(cw) +
                                            static_cast<std::size_t>(x)]);
          recon[plane].samples[static_cast<std::size_t>(py) *
                                   recon[plane].width +
                               static_cast<std::size_t>(px)] =
              static_cast<std::uint16_t>(depth::clip1(value));
        }
      }
    }
    blocks.push_back(block);
  }

  // Reuse the verified M10 entropy writer for the tile payload.  The payload
  // depends on the coefficient-CDF slice, not on the header, so pass the slice
  // for the real qindex while keeping the M10 header config pinned at q=90
  // (its bytes are discarded).  The hbd header below carries the real qindex.
  av2enc::high_bitdepth::HighBitDepthOptions payload_options;
  payload_options.base_q_idx = 90;
  payload_options.disable_cdf_update = options.disable_cdf_update;
  payload_options.qctx = depth::qctx_slice(options.base_q_idx);
  std::vector<std::uint8_t> payload;
  std::uint32_t tile_bytes = 0;
  if (!av2enc::high_bitdepth::assemble_high_bitdepth_tile_payload(
          picture, payload_options, blocks.data(), blocks.size(),
          levels.data(), levels.size(), &payload, &tile_bytes, &error)) {
    return false;
  }

  HbdConfig config;
  config.max_frame_width = static_cast<std::uint16_t>(picture.format.width);
  config.max_frame_height = static_cast<std::uint16_t>(picture.format.height);
  config.depth = fmt::BitDepth::k10;
  config.chroma = picture.format.chroma;
  config.base_q_idx = options.base_q_idx;
  config.disable_cdf_update = options.disable_cdf_update;
  if (!validate_config(config, &error)) return false;

  result.frame = assemble_still_picture(config, payload);
  result.tile_bytes = tile_bytes;
  result.frame_bytes = static_cast<std::uint32_t>(result.frame.size());
  result.quant_step = quant_step;
  result.blocks = std::move(blocks);
  result.levels = std::move(levels);
  for (int plane = 0; plane < num_planes; ++plane) {
    result.recon[plane] = std::move(recon[plane]);
  }
  return true;
}

}  // namespace av2enc::hbd

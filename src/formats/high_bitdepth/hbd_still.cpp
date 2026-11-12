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

// §7.13.2.11 approx_divide().  The DC block sizes used here have a power-of-two
// denominator (w+h is 32/64/128), for which resolve_divisor() gives f = 0 and
// Div_Lut[0] = 512.  This mirrors the verified M10 predictor exactly.
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
  if (depth::qctx_slice(base_q_idx) != 0) {
    return fail("encode seam: the reused M10 entropy writer only has the "
                "frozen coefficient-CDF slice 0 (base_q_idx <= 90).  "
                "base_q_idx 91..303 needs a qctx-aware entropy writer; the "
                "header writer and depth helpers already support it.");
  }
  return true;
}

bool encode_still(const fmt::Picture& picture, const HbdOptions& options,
                  HbdResult& result, std::string& error) {
  error.clear();
  result = HbdResult{};
  if (picture.format.depth != fmt::BitDepth::k10 ||
      picture.format.chroma != fmt::ChromaFormat::k420) {
    error = "hbd requires a 10-bit 4:2:0 picture";
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

  const int mi_cols = static_cast<int>(picture.geometry.mi_cols);
  const int mi_rows = static_cast<int>(picture.geometry.mi_rows);
  const int coded_luma_w = mi_cols * 4;
  const int coded_luma_h = mi_rows * 4;
  const int coded_chroma_w = coded_luma_w / 2;
  const int coded_chroma_h = coded_luma_h / 2;

  av2enc::high_bitdepth::HighBitDepthPlane recon[3];
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
    const int chroma = luma / 2;
    const int coded_w[3] = {luma, chroma, chroma};
    const int coded_h[3] = {luma, chroma, chroma};

    av2enc::high_bitdepth::HighBitDepthBlock block;
    block.mi_row = desc.mi_row;
    block.mi_col = desc.mi_col;
    block.luma_size = desc.luma_size;
    block.levels_offset = static_cast<std::uint32_t>(levels.size());

    for (int plane = 0; plane < 3; ++plane) {
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

      const int px0 = plane == 0 ? x0 : x0 / 2;
      const int py0 = plane == 0 ? y0 : y0 / 2;
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

  // Reuse the verified M10 entropy/OBU assembly for the tile payload.  M10
  // validates its own config with base_q_idx = 90, but the payload does not
  // depend on the header qindex (the CDF slice is frozen at slice 0 and we
  // only encode q <= 90 here), so the payload bytes are exactly what the
  // decoder needs.  We then re-frame them with the real qindex header.
  av2enc::high_bitdepth::HighBitDepthOptions payload_options;
  payload_options.base_q_idx = 90;
  payload_options.disable_cdf_update = options.disable_cdf_update;
  std::vector<std::uint8_t> high_bitdepth_frame;
  std::uint32_t high_bitdepth_tile_bytes = 0;
  if (!av2enc::high_bitdepth::assemble_high_bitdepth_stream(picture, payload_options, blocks.data(),
                                blocks.size(), levels.data(), levels.size(),
                                &high_bitdepth_frame, &high_bitdepth_tile_bytes, &error)) {
    return false;
  }

  HbdParsedStream parsed;
  if (!parse_stream(high_bitdepth_frame, &parsed, &error)) {
    error = "internal: cannot parse the reused M10 payload: " + error;
    return false;
  }

  HbdConfig config;
  config.max_frame_width = static_cast<std::uint16_t>(picture.format.width);
  config.max_frame_height = static_cast<std::uint16_t>(picture.format.height);
  config.depth = fmt::BitDepth::k10;
  config.base_q_idx = options.base_q_idx;
  config.disable_cdf_update = options.disable_cdf_update;
  if (!validate_config(config, &error)) return false;

  result.frame = assemble_still_picture(config, parsed.tile_payload);
  result.tile_bytes = static_cast<std::uint32_t>(parsed.tile_payload.size());
  result.frame_bytes = static_cast<std::uint32_t>(result.frame.size());
  result.quant_step = quant_step;
  result.blocks = std::move(blocks);
  result.levels = std::move(levels);
  for (int plane = 0; plane < 3; ++plane) {
    result.recon[plane] = std::move(recon[plane]);
  }
  (void)high_bitdepth_tile_bytes;
  return true;
}

}  // namespace av2enc::hbd

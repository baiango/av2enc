#include "formats/high_bitdepth/hbd_still.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "codec/filters/deblock/deblock.h"
#include "cuda/pipeline/still/hbd/hbd_cpu.h"
#include "formats/high_bitdepth/hbd_cdef.h"
#include "formats/high_bitdepth/hbd_ccso.h"
#include "formats/high_bitdepth/hbd_deblock.h"
#include "formats/high_bitdepth/hbd_gdf.h"
#include "formats/high_bitdepth/hbd_lr.h"
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

  // §7.20 chroma LR is gated at 4:2:0.  At 4:2:2/4:4:4 the decoded chroma
  // differs from the encoder reconstruction by 1 LSB on a handful of samples
  // at 64-line stripe boundaries (the CfL luma-tap source path); keep the
  // combination rejected by name instead of shipping a non-exact stream.
  // Reproducer: av2enc C444p10.y4m -o x.ivf --still --bit-depth 10
  //   --chroma 444 --still-chroma-lr --qp 303 --recon r.y4m
  if (options.enable_chroma_lr && num_planes > 1 &&
      !(sub_x == 1 && sub_y == 1)) {
    error =
        "chroma loop restoration is only decoder-exact at 10-bit 4:2:0; "
        "4:2:2/4:4:4 chroma LR differs from the decoder at stripe "
        "boundaries.  Use --chroma 420 or omit --still-chroma-lr";
    return false;
  }

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

  // §7.17 deblocking at the 10-bit sample depth (opt-in, so every pinned
  // 10-bit stream stays byte-identical).  The MI metadata comes from the fixed
  // M10 block plan: one transform per block, no skips in an intra still.  The
  // filter runs after every block, on the same pre-filter reconstruction the
  // decoder predicted from.
  if (options.enable_deblock) {
    DeblockGrids grids;
    if (!build_deblock_grids(picture.format, blocks, options.base_q_idx, &grids,
                             &error)) {
      return false;
    }
    if (!apply_deblock_16(grids, recon, &error)) return false;
  }

  // §7.18 CDEF (opt-in).  CurrFrame is the post-deblock reconstruction; the
  // filter writes a separate CdefFrame, exactly like the decoder, and that
  // picture becomes the output and the reference samples.
  HbdCdefPlan cdef_plan;
  // §7.19/§7.20 sample the deblocked frame *before* CDEF as CurrFrame, so keep
  // an independent copy whenever a later stage needs it.
  std::vector<std::uint16_t> curr_samples[3];
  const bool need_curr = options.enable_ccso || options.enable_cdef ||
                         options.enable_wiener || options.enable_chroma_lr ||
                         options.enable_pc_wiener || options.enable_gdf;
  if (need_curr) {
    for (int plane = 0; plane < num_planes; ++plane) {
      curr_samples[plane] = recon[plane].samples;
    }
  }
  if (options.enable_cdef) {
    HbdFilterFrame16 filter_frame;
    int src_stride[3] = {0, 0, 0};
    const std::uint16_t* src_planes[3] = {nullptr, nullptr, nullptr};
    std::vector<std::uint16_t> cdef_samples[3];
    for (int plane = 0; plane < num_planes; ++plane) {
      filter_frame.src[plane] = curr_samples[plane].data();
      filter_frame.ref[plane] = picture.planes[plane].samples.data();
      filter_frame.src_stride[plane] = recon[plane].width;
      filter_frame.ref_stride[plane] = picture.planes[plane].stride;
      filter_frame.visible_w[plane] = recon[plane].visible_width;
      filter_frame.visible_h[plane] = recon[plane].visible_height;
      src_stride[plane] = recon[plane].width;
      src_planes[plane] = curr_samples[plane].data();
    }
    filter_frame.mi_cols = mi_cols;
    filter_frame.mi_rows = mi_rows;
    filter_frame.sub_x = sub_x;
    filter_frame.sub_y = sub_y;
    filter_frame.num_planes = num_planes;
    filter_frame.bit_depth = depth::kBitDepth;
    if (!plan_cdef(filter_frame, options.cdef_damping,
                   options.cdef_num_strengths, &cdef_plan, &error)) {
      return false;
    }
    std::uint16_t* dst_planes[3] = {nullptr, nullptr, nullptr};
    for (int plane = 0; plane < num_planes; ++plane) {
      cdef_samples[plane].assign(recon[plane].samples.size(), 0);
      dst_planes[plane] = cdef_samples[plane].data();
    }
    if (!apply_cdef16(cdef_plan, picture.format, src_planes, dst_planes,
                      src_stride, mi_cols, mi_rows, &error)) {
      return false;
    }
    for (int plane = 0; plane < num_planes; ++plane) {
      recon[plane].samples.swap(cdef_samples[plane]);
    }
  }

  // §7.19 CCSO (opt-in).  CurrFrame is the pre-CDEF picture; CdefFrame is the
  // current reconstruction and is modified in place.
  HbdCcsoPlan ccso_plan;
  if (options.enable_ccso) {
    HbdCcsoFrame16 filter_frame;
    int cdef_stride[3] = {0, 0, 0};
    const std::uint16_t* curr_planes[3] = {nullptr, nullptr, nullptr};
    std::uint16_t* out_planes[3] = {nullptr, nullptr, nullptr};
    for (int plane = 0; plane < num_planes; ++plane) {
      filter_frame.curr[plane] = curr_samples[plane].data();
      filter_frame.cdef[plane] = recon[plane].samples.data();
      filter_frame.ref[plane] = picture.planes[plane].samples.data();
      filter_frame.curr_stride[plane] = recon[plane].width;
      filter_frame.cdef_stride[plane] = recon[plane].width;
      filter_frame.ref_stride[plane] = picture.planes[plane].stride;
      filter_frame.visible_w[plane] = recon[plane].visible_width;
      filter_frame.visible_h[plane] = recon[plane].visible_height;
      cdef_stride[plane] = recon[plane].width;
      curr_planes[plane] = curr_samples[plane].data();
      out_planes[plane] = recon[plane].samples.data();
    }
    filter_frame.mi_cols = mi_cols;
    filter_frame.mi_rows = mi_rows;
    filter_frame.sub_x = sub_x;
    filter_frame.sub_y = sub_y;
    filter_frame.num_planes = num_planes;
    filter_frame.bit_depth = depth::kBitDepth;
    if (!plan_ccso(filter_frame, options.ccso_luma_size_log2,
                   options.ccso_edge_only, &ccso_plan, &error)) {
      return false;
    }
    if (!apply_ccso16(ccso_plan, picture.format, curr_planes, out_planes,
                      cdef_stride, cdef_stride, mi_cols, mi_rows, &error)) {
      return false;
    }
  }

  // §7.20 loop restoration (Wiener NS and/or PC-Wiener) and §7.20.5 GDF.
  // LrFrame is a separate buffer: LR reads CurrFrame/CdefFrame and writes it,
  // then GDF rewrites its luma from CdefFrame.
  HbdLrPlan lr_plan;
  HbdGdfPlan gdf_plan;
  std::vector<std::uint16_t> lr_samples[3];
  std::vector<std::uint8_t> lr_tx_skip;
  const bool any_lr = options.enable_wiener || options.enable_chroma_lr ||
                      options.enable_pc_wiener;
  if (any_lr || options.enable_gdf) {
    HbdLrFrame16 lr_frame;
    for (int plane = 0; plane < num_planes; ++plane) {
      lr_frame.curr[plane] = curr_samples[plane].data();
      lr_frame.cdef[plane] = recon[plane].samples.data();
      lr_frame.ref[plane] = picture.planes[plane].samples.data();
      lr_frame.curr_stride[plane] = recon[plane].width;
      lr_frame.cdef_stride[plane] = recon[plane].width;
      lr_frame.ref_stride[plane] = picture.planes[plane].stride;
      lr_frame.visible_w[plane] = recon[plane].visible_width;
      lr_frame.visible_h[plane] = recon[plane].visible_height;
    }
    lr_frame.mi_cols = mi_cols;
    lr_frame.mi_rows = mi_rows;
    lr_frame.sub_x = sub_x;
    lr_frame.sub_y = sub_y;
    lr_frame.num_planes = num_planes;
    lr_frame.bit_depth = depth::kBitDepth;
    lr_frame.base_q_idx = options.base_q_idx;

    // §7.20.1 LrTxSkip: skip == 0 everywhere, luma eob == 0 marks the cell.
    lr_tx_skip.assign(static_cast<std::size_t>(mi_cols) *
                          static_cast<std::size_t>(mi_rows),
                      0);
    if (options.enable_pc_wiener) {
      for (const auto& block : blocks) {
        if (block.eob[0] != 0) continue;
        const int w4 = block.luma_size / 4;
        for (int yy = 0; yy < w4; ++yy) {
          for (int xx = 0; xx < w4; ++xx) {
            const int r = block.mi_row + yy;
            const int c = block.mi_col + xx;
            if (r < mi_rows && c < mi_cols) {
              lr_tx_skip[static_cast<std::size_t>(r) * mi_cols + c] = 1;
            }
          }
        }
      }
    }

    if (any_lr) {
      if (!plan_lr(lr_frame, options.enable_wiener,
                   options.enable_chroma_lr, options.enable_pc_wiener,
                   options.base_q_idx, &lr_plan, &error)) {
        return false;
      }
      lr_plan.config.lr_tx_skip = lr_tx_skip;
      lr_plan.config.tx_skip_cols = mi_cols;
      lr_plan.config.tx_skip_rows = mi_rows;
      if (options.enable_pc_wiener) {
        std::vector<std::uint16_t> scratch(
            static_cast<std::size_t>(recon[0].width) *
            static_cast<std::size_t>(recon[0].height));
        plan_pc_wiener16(lr_frame, &lr_plan, lr_tx_skip, scratch.data());
      }
    }

    // LrFrame buffers.
    std::uint16_t* lr_ptr[3] = {nullptr, nullptr, nullptr};
    int lr_stride[3] = {0, 0, 0};
    for (int plane = 0; plane < num_planes; ++plane) {
      lr_samples[plane].assign(recon[plane].samples.size(), 0);
      lr_ptr[plane] = lr_samples[plane].data();
      lr_stride[plane] = recon[plane].width;
    }
    if (any_lr && lr_plan.config.enabled) {
      if (!apply_wiener16(lr_frame, lr_plan.config, lr_ptr, lr_stride,
                          &error)) {
        return false;
      }
      if (options.enable_pc_wiener && lr_plan.config.plane[0].enabled) {
        apply_pc_wiener16(lr_frame, lr_plan.config,
                          lr_plan.config.lr_tx_skip, lr_ptr[0],
                          lr_stride[0]);
      }
    } else {
      for (int plane = 0; plane < num_planes; ++plane) {
        lr_samples[plane] = recon[plane].samples;
      }
    }

    if (options.enable_gdf) {
      HbdGdfFrame16 gdf_frame;
      gdf_frame.curr = curr_samples[0].data();
      gdf_frame.cdef = recon[0].samples.data();
      gdf_frame.ref = picture.planes[0].samples.data();
      gdf_frame.curr_stride = recon[0].width;
      gdf_frame.cdef_stride = recon[0].width;
      gdf_frame.ref_stride = picture.planes[0].stride;
      gdf_frame.width = recon[0].visible_width;
      gdf_frame.height = recon[0].visible_height;
      gdf_frame.mi_cols = mi_cols;
      gdf_frame.mi_rows = mi_rows;
      gdf_frame.bit_depth = depth::kBitDepth;
      gdf_frame.base_q_idx = options.base_q_idx;
      if (!plan_gdf(gdf_frame, &gdf_plan, &error)) return false;
      if (!apply_gdf16(gdf_plan, gdf_frame, lr_ptr[0], lr_stride[0],
                       &error)) {
        return false;
      }
    }

    for (int plane = 0; plane < num_planes; ++plane) {
      recon[plane].samples.swap(lr_samples[plane]);
    }
  }

  // Reuse the verified M10 entropy writer for the tile payload.  The payload
  // depends on the coefficient-CDF slice, not on the header, so pass the slice
  // for the real qindex while keeping the M10 header config pinned at q=90
  // (its bytes are discarded).  The hbd header below carries the real qindex.
  av2enc::high_bitdepth::HighBitDepthOptions payload_options;
  payload_options.base_q_idx = 90;
  payload_options.disable_cdf_update = options.disable_cdf_update;
  payload_options.qctx = depth::qctx_slice(options.base_q_idx);
  if (cdef_plan.enabled && cdef_plan.num_strengths > 1) {
    payload_options.cdef_idx = cdef_plan.index.data();
    payload_options.cdef_num_strengths = cdef_plan.num_strengths;
  }
  if (lr_plan.config.enabled) payload_options.lr = &lr_plan.config;
  if (ccso_plan.config.enabled) {
    payload_options.ccso_luma_size_log2 = ccso_plan.config.ccso_luma_size_log2;
    for (int plane = 0; plane < num_planes; ++plane) {
      payload_options.ccso_plane_enabled[plane] =
          ccso_plan.config.plane[plane].enabled;
      payload_options.ccso_blks[plane] = ccso_plan.config.blks[plane].data();
      payload_options.ccso_blk_cols[plane] = ccso_plan.config.blk_cols[plane];
      payload_options.ccso_blk_rows[plane] = ccso_plan.config.blk_rows[plane];
    }
  }
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
  config.apply_deblock_y_vert = options.enable_deblock;
  config.apply_deblock_y_horz = options.enable_deblock;
  config.apply_deblock_u =
      options.enable_deblock && num_planes > 1;
  config.apply_deblock_v =
      options.enable_deblock && num_planes > 2;
  if (cdef_plan.enabled) {
    config.enable_cdef = true;
    config.cdef_damping = cdef_plan.damping;
    config.cdef_num_strengths = cdef_plan.num_strengths;
    for (int i = 0; i < cdef_plan.num_strengths; ++i) {
      config.cdef_strengths[i] = cdef_plan.strengths[i];
    }
  }
  if (ccso_plan.config.enabled) config.ccso = &ccso_plan.config;
  if (lr_plan.config.enabled) config.lr = &lr_plan.config;
  if (gdf_plan.config.enabled) config.gdf = &gdf_plan.config;
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

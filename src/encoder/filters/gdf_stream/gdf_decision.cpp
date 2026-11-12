#include "encoder/filters/gdf_stream/gdf_decision.h"

#include "encoder/filters/spec_tables/gdf.h"
#include "encoder/filters/spec_tables/luma_source.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace av2enc::filters::gdf {
namespace {

bool fail(std::string* error, const std::string& message) {
  if (error != nullptr) *error = message;
  return false;
}

bool valid_plane(const GdfPlane& plane) {
  return plane.data != nullptr && plane.width > 0 && plane.height > 0 &&
         plane.stride >= plane.width;
}

spec::GdfBlockIndices TableIndices(const FrameGdfParams& params,
                                   const GdfFilterInputs& inputs) {
  spec::GdfParams sp;
  sp.frame_is_intra = inputs.frame_is_intra;
  sp.base_q_idx = inputs.base_q_idx;
  sp.bit_depth = inputs.bit_depth;
  sp.gdf_pic_qc_idx = params.pic_qc_idx;
  return spec::GdfBlockTableIndices(
      sp, inputs.order_hints, inputs.frame_distance, inputs.num_total_refs,
      inputs.restricted_oh);
}

double Psnr(std::int64_t sse, std::int64_t samples, int bit_depth) {
  if (samples <= 0) return 0.0;
  if (sse <= 0) return 99.0;
  const double maxv = static_cast<double>((1 << bit_depth) - 1);
  const double mse = static_cast<double>(sse) / static_cast<double>(samples);
  return 10.0 * std::log10(maxv * maxv / mse);
}

}  // namespace

int GdfUnitRegions(const GdfUnitLayout& layout, int row, int col,
                   GdfStripeRegion out[8]) {
  if (out == nullptr || layout.blk_size <= 0) return 0;
  if (row < 0 || row >= layout.rows || col < 0 || col >= layout.cols) return 0;
  const int unit_mi = layout.blk_size / 4;
  int start_mi = unit_mi * row - 2;
  int end_mi = start_mi + unit_mi - 1;
  // §7.20.1 clamps stripeRow to MiRowEnd - 1, which pulls the tail of the frame
  // into the last GdfBlks row.
  if (row == layout.rows - 1) end_mi = layout.mi_rows - 1;
  if (end_mi >= layout.mi_rows) end_mi = layout.mi_rows - 1;
  if (start_mi > end_mi) return 0;

  int ys = std::max(0, start_mi * 4);
  int ye = end_mi * 4 + 3;
  if (ye >= layout.height) ye = layout.height - 1;
  const int xs = col * layout.blk_size;
  if (xs >= layout.width) return 0;
  const int xe = std::min(xs + layout.blk_size - 1, layout.width - 1);

  int count = 0;
  int stripe = (ys + 8) / 64;
  while (stripe * 64 - 8 <= ye && count < 8) {
    const int unclipped_start = stripe * 64 - 8;
    const int stripe_start = std::max(0, unclipped_start);
    const int stripe_end = std::min(layout.height - 1, unclipped_start + 63);
    const int region_y = std::max(ys, stripe_start);
    const int region_end = std::min(ye, stripe_end);
    if (region_y <= region_end) {
      GdfStripeRegion& region = out[count++];
      region.x = xs;
      region.y = region_y;
      region.w = xe - xs + 1;
      region.h = region_end - region_y + 1;
      region.stripe_start_y = stripe_start;
      region.stripe_end_y = stripe_end;
    }
    ++stripe;
  }
  return count;
}

bool ApplyGdfToFrame(const GdfSourceFrame& source,
                     const GdfFilterInputs& inputs,
                     const FrameGdfContext& context,
                     const FrameGdfParams& params,
                     const std::vector<std::uint8_t>& unit_flags,
                     std::uint16_t* out, int out_stride, std::string* error) {
  if (source.curr == nullptr || out == nullptr) {
    return fail(error, "GDF: null luma buffer");
  }
  if (source.stride < context.width || out_stride < context.width) {
    return fail(error, "GDF: luma stride smaller than width");
  }
  if ((context.width & 1) != 0 || (context.height & 1) != 0) {
    return fail(error, "GDF: luma dimensions must be even");
  }
  if (inputs.bit_depth != 8 && inputs.bit_depth != 10 &&
      inputs.bit_depth != 12) {
    return fail(error, "GDF: BitDepth must be 8, 10 or 12");
  }
  if (!params.frame_enable) return fail(error, "GDF: frame is not enabled");

  const GdfUnitLayout layout = DeriveGdfUnitLayout(context);
  if (layout.blk_size <= 0) {
    return fail(error, "GDF: frame layout is disabled");
  }
  if (params.blk_size != 0 && params.blk_size != layout.blk_size) {
    return fail(error, "GDF: GdfBlkSize does not match the frame context");
  }
  if (params.per_block) {
    if (unit_flags.size() !=
        static_cast<std::size_t>(layout.cols) *
            static_cast<std::size_t>(layout.rows)) {
      return fail(error, "GDF: GdfBlks grid has the wrong size");
    }
  }

  const spec::GdfBlockIndices indices = TableIndices(params, inputs);
  const std::uint16_t* cdef =
      source.cdef != nullptr ? source.cdef : source.curr;

  for (int row = 0; row < layout.rows; ++row) {
    for (int col = 0; col < layout.cols; ++col) {
      if (params.per_block) {
        const std::size_t index =
            static_cast<std::size_t>(row) *
                static_cast<std::size_t>(layout.cols) +
            static_cast<std::size_t>(col);
        if (unit_flags[index] == 0) continue;
      }
      GdfStripeRegion regions[8];
      const int count = GdfUnitRegions(layout, row, col, regions);
      for (int r = 0; r < count; ++r) {
        const GdfStripeRegion& region = regions[r];
        if ((region.w & 1) != 0 || (region.h & 1) != 0) {
          return fail(error, "GDF: stripe region is not even sized");
        }
        spec::LumaSource src;
        src.curr = source.curr;
        src.cdef = cdef;
        src.curr_stride = source.stride;
        src.cdef_stride = source.stride;
        src.luma_start_x = 0;
        src.luma_end_x = context.width - 1;
        src.luma_start_y = 0;
        src.luma_end_y = context.height - 1;
        src.luma_stripe_start_y = region.stripe_start_y;
        src.luma_stripe_end_y = region.stripe_end_y;
        spec::ApplyGdfFilter(src, out, out_stride, region.x, region.y,
                             indices.qp_idx, indices.ref_dst_idx, region.w,
                             region.h, region.stripe_end_y, params.pix_scale,
                             inputs.bit_depth);
      }
    }
  }
  return true;
}

bool MeasureUseGdfSymbols(const GdfUnitLayout& layout,
                          const FrameGdfParams& params, int sb_size,
                          const std::vector<std::uint8_t>& flags,
                          bool disable_cdf_update, std::int64_t* bytes,
                          std::int64_t* bits, std::string* error) {
  if (bytes == nullptr || bits == nullptr) {
    return fail(error, "GDF: null bit-count output");
  }
  *bytes = 0;
  *bits = 0;
  if (!params.per_block) return true;
  if (flags.size() != static_cast<std::size_t>(layout.cols) *
                          static_cast<std::size_t>(layout.rows)) {
    return fail(error, "GDF: GdfBlks grid has the wrong size");
  }

  const std::size_t symbols = flags.size();
  std::vector<std::uint8_t> buffer(4U * symbols + 64U);
  av2::Encoder encoder;
  encoder.disable_cdf_update = disable_cdf_update ? 1 : 0;
  av2::Status status = encoder.init_symbol(buffer.data(), buffer.size());
  if (status != av2::Status::kOk) {
    return fail(error, "GDF: symbol coder init failed");
  }
  GdfBlockCoder coder;
  FrameGdfParams per_block = params;
  per_block.per_block = true;
  const int sb_mi = sb_size / 4;
  const int sb_rows = (layout.mi_rows + sb_mi - 1) / sb_mi;
  const int sb_cols = (layout.mi_cols + sb_mi - 1) / sb_mi;
  int coded_symbols = 0;
  for (int sb_row = 0; sb_row < sb_rows; ++sb_row) {
    for (int sb_col = 0; sb_col < sb_cols; ++sb_col) {
      const int mi_row = sb_row * sb_mi;
      const int mi_col = sb_col * sb_mi;
      GdfSymbolPosition position;
      if (!GdfSymbolAt(layout, per_block, sb_size, mi_row, mi_col, &position)) {
        continue;
      }
      const std::size_t index =
          static_cast<std::size_t>(position.row) *
              static_cast<std::size_t>(layout.cols) +
          static_cast<std::size_t>(position.col);
      bool coded = false;
      if (!coder.WriteAt(encoder, layout, per_block, sb_size, mi_row, mi_col,
                         flags[index] != 0, &coded)) {
        return fail(error, "GDF: symbol encode failed");
      }
      if (!coded) return fail(error, "GDF: unit origin did not code a symbol");
      ++coded_symbols;
    }
  }
  if (coded_symbols != static_cast<int>(symbols)) {
    return fail(error, "GDF: GdfBlks grid was not fully coded");
  }
  std::size_t payload_bytes = 0;
  status = encoder.exit_symbol(&payload_bytes);
  if (status != av2::Status::kOk) {
    return fail(error, "GDF: symbol coder exit failed");
  }
  *bytes = static_cast<std::int64_t>(payload_bytes);
  *bits = static_cast<std::int64_t>(payload_bytes) * 8;
  return true;
}

bool DecideGdf(const GdfSourceFrame& recon, const GdfPlane& original,
               const FrameGdfContext& context,
               const GdfDecisionOptions& options, GdfDecision* decision,
               std::string* error) {
  if (decision == nullptr) return fail(error, "GDF: null decision");
  *decision = GdfDecision{};
  if (!context.enable_gdf) {
    return fail(error, "GDF: decision requires enable_gdf");
  }
  if (!options.search_pic_indices &&
      (options.pic_qc_idx < 0 || options.pic_qc_idx >= kGdfNumPicQcIdx ||
       options.pic_scale_idx < 0 ||
       options.pic_scale_idx >= kGdfNumPicScaleIdx)) {
    return fail(error, "GDF: fixed gdf_pic_qc_idx/gdf_pic_scale_idx out of range");
  }
  if (recon.curr == nullptr || !valid_plane(original)) {
    return fail(error, "GDF: decision needs recon and original luma");
  }
  if (original.width != context.width || original.height != context.height) {
    return fail(error, "GDF: original luma size does not match the context");
  }
  if (static_cast<std::uint16_t>(original.width) == 0U) {
    return fail(error, "GDF: empty frame");
  }

  const GdfUnitLayout layout = DeriveGdfUnitLayout(context);
  if (layout.blk_size <= 0) {
    return fail(error, "GDF: frame layout is disabled");
  }
  const std::int64_t samples =
      static_cast<std::int64_t>(context.width) * context.height;
  const int sub = layout.blk_size / 64;
  const int block_cols = layout.cols * sub;
  const int block_rows = layout.rows * sub;
  const std::size_t unit_count =
      static_cast<std::size_t>(layout.cols) * static_cast<std::size_t>(layout.rows);

  const int frame_enable_bits = context.single_picture_header_flag ? 0 : 1;
  const int seq_bits = 1 + (context.sb_size == 64 ? 1 : 0);
  const int base_header_bits =
      seq_bits + frame_enable_bits + 4;  // qc (2) + scale (2)

  std::vector<int> qc_values;
  std::vector<int> scale_values;
  if (options.search_pic_indices) {
    for (int q = 0; q < kGdfNumPicQcIdx; ++q) qc_values.push_back(q);
    for (int s = 0; s < kGdfNumPicScaleIdx; ++s) scale_values.push_back(s);
  } else {
    qc_values.push_back(options.pic_qc_idx);
    scale_values.push_back(options.pic_scale_idx);
  }

  std::vector<std::uint16_t> filtered(
      static_cast<std::size_t>(context.width) *
      static_cast<std::size_t>(context.height));
  const std::uint16_t* cdef =
      recon.cdef != nullptr ? recon.cdef : recon.curr;

  double best_cost = 0.0;
  bool have_best = false;

  for (const int qc : qc_values) {
    for (const int scale : scale_values) {
      FrameGdfParams base;
      base.frame_enable = true;
      base.per_block = false;
      base.pic_qc_idx = qc;
      base.pic_scale_idx = scale;
      base.pix_scale = 1 + scale;
      base.blk_size = layout.blk_size;
      base.cols = layout.cols;
      base.rows = layout.rows;
      base.per_block_present = layout.per_block_present;

      for (int y = 0; y < context.height; ++y) {
        std::copy_n(cdef + static_cast<std::ptrdiff_t>(y) * recon.stride,
                    context.width,
                    filtered.begin() + static_cast<std::ptrdiff_t>(y) *
                                          context.width);
      }
      if (!ApplyGdfToFrame(recon, options.filter, context, base,
                           std::vector<std::uint8_t>{}, filtered.data(),
                           context.width, error)) {
        return false;
      }

      std::vector<std::int64_t> unit_before(unit_count, 0);
      std::vector<std::int64_t> unit_after(unit_count, 0);
      std::vector<std::int64_t> block_before(
          static_cast<std::size_t>(block_cols) *
              static_cast<std::size_t>(block_rows),
          0);
      std::vector<std::int64_t> block_after(block_before.size(), 0);

      for (int row = 0; row < layout.rows; ++row) {
        for (int col = 0; col < layout.cols; ++col) {
          const std::size_t unit_index =
              static_cast<std::size_t>(row) *
                  static_cast<std::size_t>(layout.cols) +
              static_cast<std::size_t>(col);
          const int unit_x = col * layout.blk_size;
          const int unit_y = std::max(0, row * layout.blk_size - 8);
          GdfStripeRegion regions[8];
          const int count = GdfUnitRegions(layout, row, col, regions);
          for (int r = 0; r < count; ++r) {
            const GdfStripeRegion& region = regions[r];
            for (int y = region.y; y < region.y + region.h; ++y) {
              const std::uint16_t* orig_row =
                  original.data +
                  static_cast<std::ptrdiff_t>(y) * original.stride;
              const std::uint16_t* rec_row =
                  recon.curr + static_cast<std::ptrdiff_t>(y) * recon.stride;
              const std::uint16_t* out_row =
                  filtered.data() +
                  static_cast<std::ptrdiff_t>(y) * context.width;
              int by = (y - unit_y) / 64;
              by = std::clamp(by, 0, sub - 1);
              for (int x = region.x; x < region.x + region.w; ++x) {
                const std::int64_t o = orig_row[x];
                const std::int64_t e = o - rec_row[x];
                const std::int64_t f = o - out_row[x];
                const std::int64_t before = e * e;
                const std::int64_t after = f * f;
                unit_before[unit_index] += before;
                unit_after[unit_index] += after;
                int bx = (x - unit_x) / 64;
                bx = std::clamp(bx, 0, sub - 1);
                const std::size_t block_index =
                    static_cast<std::size_t>(row * sub + by) *
                        static_cast<std::size_t>(block_cols) +
                    static_cast<std::size_t>(col * sub + bx);
                block_before[block_index] += before;
                block_after[block_index] += after;
              }
            }
          }
        }
      }

      // Mode A: one frame-wide decision, no per-block symbols.
      std::int64_t sse_after = 0;
      for (const std::int64_t value : unit_after) sse_after += value;
      double cost = static_cast<double>(sse_after) +
                    options.lambda_bits * static_cast<double>(base_header_bits);
      bool chosen_mode_per_block = false;
      std::vector<std::uint8_t> chosen_flags;

      // Mode B: per-unit flags, only when gdf_per_block is coded.
      if (layout.per_block_present) {
        std::vector<std::uint8_t> flags(unit_count, 0);
        int enabled = 0;
        for (std::size_t i = 0; i < unit_count; ++i) {
          const std::int64_t gain = unit_before[i] - unit_after[i];
          if (static_cast<double>(gain) >
              options.lambda_bits * 1.0) {
            flags[i] = 1;
            ++enabled;
          }
        }
        std::int64_t sse_per_block = 0;
        for (std::size_t i = 0; i < unit_count; ++i) {
          sse_per_block += flags[i] != 0 ? unit_after[i] : unit_before[i];
        }
        std::int64_t symbol_bytes = 0;
        std::int64_t symbol_bits = 0;
        FrameGdfParams per_block = base;
        per_block.per_block = true;
        if (!MeasureUseGdfSymbols(layout, per_block, context.sb_size, flags,
                                  options.disable_cdf_update, &symbol_bytes,
                                  &symbol_bits, error)) {
          return false;
        }
        const int header = base_header_bits + 1;  // gdf_per_block bit
        const double per_block_cost =
            static_cast<double>(sse_per_block) +
            options.lambda_bits *
                static_cast<double>(header + symbol_bits);
        if (per_block_cost < cost) {
          cost = per_block_cost;
          sse_after = sse_per_block;
          chosen_mode_per_block = true;
          chosen_flags = std::move(flags);
        }
        (void)enabled;
      }

      if (!have_best || cost < best_cost) {
        have_best = true;
        best_cost = cost;
        std::int64_t sse_before = 0;
        for (const std::int64_t value : unit_before) sse_before += value;

        decision->params = base;
        decision->params.per_block = chosen_mode_per_block;
        decision->chosen_qc_idx = qc;
        decision->chosen_scale_idx = scale;
        decision->block64_cols = block_cols;
        decision->block64_rows = block_rows;
        decision->unit_flags = std::move(chosen_flags);
        decision->total_units = static_cast<int>(unit_count);
        decision->enabled_units = chosen_mode_per_block
                                      ? static_cast<int>(std::count(
                                            decision->unit_flags.begin(),
                                            decision->unit_flags.end(),
                                            std::uint8_t{1}))
                                      : static_cast<int>(unit_count);
        decision->sse_before = sse_before;
        decision->sse_after = sse_after;
        decision->header_bits =
            base_header_bits + (chosen_mode_per_block ? 1 : 0);
        decision->symbol_bytes = 0;
        decision->symbol_bits = 0;
        if (chosen_mode_per_block) {
          FrameGdfParams per_block = base;
          per_block.per_block = true;
          if (!MeasureUseGdfSymbols(layout, per_block, context.sb_size,
                                    decision->unit_flags,
                                    options.disable_cdf_update,
                                    &decision->symbol_bytes,
                                    &decision->symbol_bits, error)) {
            return false;
          }
        }
        decision->psnr_before = Psnr(sse_before, samples, options.filter.bit_depth);
        decision->psnr_after = Psnr(sse_after, samples, options.filter.bit_depth);

        decision->block64_flags.assign(block_before.size(), 0);
        for (std::size_t i = 0; i < block_before.size(); ++i) {
          const int block_row = static_cast<int>(i) / block_cols;
          const int block_col = static_cast<int>(i) % block_cols;
          const int unit_row = block_row / sub;
          const int unit_col = block_col / sub;
          const std::size_t unit_index =
              static_cast<std::size_t>(unit_row) *
                  static_cast<std::size_t>(layout.cols) +
              static_cast<std::size_t>(unit_col);
          const bool unit_on = !chosen_mode_per_block ||
                               decision->unit_flags[unit_index] != 0;
          if (unit_on && block_after[i] < block_before[i]) {
            decision->block64_flags[i] = 1;
          }
        }
      }
    }
  }

  if (!have_best) return fail(error, "GDF: no candidate evaluated");

  // A regular frame may code gdf_frame_enable = 0.  A single-picture frame
  // infers 1 and cannot turn GDF off at frame level.
  if (!context.single_picture_header_flag) {
    std::int64_t sse_before = 0;
    for (int y = 0; y < context.height; ++y) {
      const std::uint16_t* orig_row =
          original.data + static_cast<std::ptrdiff_t>(y) * original.stride;
      const std::uint16_t* rec_row =
          recon.curr + static_cast<std::ptrdiff_t>(y) * recon.stride;
      for (int x = 0; x < context.width; ++x) {
        const std::int64_t d =
            static_cast<std::int64_t>(orig_row[x]) - rec_row[x];
        sse_before += d * d;
      }
    }
    const double disable_cost =
        static_cast<double>(sse_before) +
        options.lambda_bits * static_cast<double>(seq_bits + 1);
    if (disable_cost < best_cost) {
      decision->params = FrameGdfParams{};
      decision->unit_flags.clear();
      decision->block64_flags.assign(
          static_cast<std::size_t>(block_cols) *
              static_cast<std::size_t>(block_rows),
          0);
      decision->block64_cols = block_cols;
      decision->block64_rows = block_rows;
      decision->chosen_qc_idx = 0;
      decision->chosen_scale_idx = 0;
      decision->enabled_units = 0;
      decision->total_units = static_cast<int>(unit_count);
      decision->sse_before = sse_before;
      decision->sse_after = sse_before;
      decision->header_bits = seq_bits + 1;
      decision->symbol_bytes = 0;
      decision->symbol_bits = 0;
      decision->psnr_before = Psnr(sse_before, samples, options.filter.bit_depth);
      decision->psnr_after = decision->psnr_before;
    }
  }
  return true;
}

}  // namespace av2enc::filters::gdf

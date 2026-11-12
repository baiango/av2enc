#include "encoder/filters/gdf_stream/gdf_syntax.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>

namespace av2enc::filters::gdf {
namespace {

using m1::filters_stream::BitReader;

bool fail(std::string* error, const char* message) {
  if (error != nullptr) *error = message;
  return false;
}

bool valid_sb_size(int sb_size) { return sb_size == 64 || sb_size == 128; }

bool valid_pic_index(int value) {
  return value >= 0 && value < kGdfNumPicQcIdx;
}

}  // namespace

int DeriveGdfBlkSize(int sb_size, bool unit_matches_sb_size,
                     const std::vector<int>& mi_col_starts,
                     const std::vector<int>& mi_row_starts) {
  int gdf_blk_size = std::max(sb_size, kGdfMinSize);
  if (unit_matches_sb_size) {
    gdf_blk_size = sb_size;
  } else if (sb_size == 64) {
    int a = 0;
    for (const int start : mi_col_starts) a |= start;
    for (const int start : mi_row_starts) a |= start;
    if ((a & 16) != 0) gdf_blk_size = 64;
  }
  return gdf_blk_size;
}

int GdfUnitCols(int mi_cols, int blk_size) {
  if (blk_size <= 0 || mi_cols <= 0) return 0;
  return (mi_cols * 4 + blk_size - 1) / blk_size;
}

int GdfUnitRows(int mi_rows, int blk_size) {
  if (blk_size <= 0 || mi_rows <= 0) return 0;
  return (mi_rows * 4 + blk_size - 1) / blk_size;
}

GdfUnitLayout DeriveGdfUnitLayout(const FrameGdfContext& context) {
  GdfUnitLayout layout;
  layout.mi_cols = context.mi_cols;
  layout.mi_rows = context.mi_rows;
  layout.width = context.width;
  layout.height = context.height;
  if (context.coded_lossless || !context.enable_gdf ||
      !valid_sb_size(context.sb_size)) {
    return layout;
  }
  layout.blk_size =
      DeriveGdfBlkSize(context.sb_size, context.gdf_unit_matches_sb_size,
                       context.mi_col_starts, context.mi_row_starts);
  layout.cols = GdfUnitCols(context.mi_cols, layout.blk_size);
  layout.rows = GdfUnitRows(context.mi_rows, layout.blk_size);
  layout.per_block_present =
      context.mi_cols * 4 > layout.blk_size ||
      context.mi_rows * 4 > layout.blk_size ||
      (context.disable_loopfilters_across_tiles &&
       (context.tile_rows > 1 || context.tile_cols > 1));
  return layout;
}

std::size_t WriteSequenceGdfFlags(BitWriter& writer,
                                  const SequenceGdfFlags& flags, int sb_size,
                                  std::string* error) {
  if (!valid_sb_size(sb_size)) {
    fail(error, "GDF: sb_size must be 64 or 128");
    return 0;
  }
  const std::size_t start = writer.bit_size();
  writer.put_bit(flags.enable_gdf);
  if (flags.enable_gdf && sb_size == 64) {
    writer.put_bit(flags.gdf_unit_matches_sb_size);
  }
  return writer.bit_size() - start;
}

bool ReadSequenceGdfFlags(BitReader& reader, int sb_size,
                          SequenceGdfFlags* flags, std::string* error) {
  if (flags == nullptr) return fail(error, "GDF: null sequence flags");
  if (!valid_sb_size(sb_size)) {
    return fail(error, "GDF: sb_size must be 64 or 128");
  }
  bool enable = false;
  if (!reader.read_bit(&enable)) return fail(error, "GDF: truncated enable_gdf");
  bool matches = false;
  if (enable && sb_size == 64) {
    if (!reader.read_bit(&matches)) {
      return fail(error, "GDF: truncated gdf_unit_matches_sb_size");
    }
  }
  flags->enable_gdf = enable;
  flags->gdf_unit_matches_sb_size = matches;
  return true;
}

std::size_t WriteFrameGdfParams(BitWriter& writer,
                                const FrameGdfContext& context,
                                FrameGdfParams* params, std::string* error) {
  if (params == nullptr) {
    fail(error, "GDF: null frame params");
    return 0;
  }
  const bool requested_enable = params->frame_enable;

  params->frame_enable = false;
  params->per_block = false;
  params->blk_size = 0;
  params->cols = 0;
  params->rows = 0;
  params->pix_scale = 1;
  params->per_block_present = false;

  if (context.coded_lossless || !context.enable_gdf) return 0;
  if (!valid_sb_size(context.sb_size)) {
    fail(error, "GDF: sb_size must be 64 or 128");
    return 0;
  }

  const std::size_t start = writer.bit_size();
  if (context.single_picture_header_flag) {
    params->frame_enable = true;
  } else {
    params->frame_enable = requested_enable;
    writer.put_bit(params->frame_enable);
    if (!params->frame_enable) return writer.bit_size() - start;
  }

  const GdfUnitLayout layout = DeriveGdfUnitLayout(context);
  params->blk_size = layout.blk_size;
  params->cols = layout.cols;
  params->rows = layout.rows;
  params->per_block_present = layout.per_block_present;

  if (layout.per_block_present) {
    writer.put_bit(params->per_block);
  } else {
    params->per_block = false;
  }

  if (!valid_pic_index(params->pic_qc_idx) ||
      !valid_pic_index(params->pic_scale_idx)) {
    fail(error, "GDF: gdf_pic_qc_idx/gdf_pic_scale_idx must be in 0..3");
    return writer.bit_size() - start;
  }
  writer.put_bits(static_cast<std::uint64_t>(params->pic_qc_idx), 2U);
  writer.put_bits(static_cast<std::uint64_t>(params->pic_scale_idx), 2U);
  params->pix_scale = 1 + params->pic_scale_idx;
  return writer.bit_size() - start;
}

bool ReadFrameGdfParams(BitReader& reader, const FrameGdfContext& context,
                        FrameGdfParams* params, std::string* error) {
  if (params == nullptr) return fail(error, "GDF: null frame params");
  *params = FrameGdfParams{};
  if (context.coded_lossless || !context.enable_gdf) return true;
  if (!valid_sb_size(context.sb_size)) {
    return fail(error, "GDF: sb_size must be 64 or 128");
  }

  if (context.single_picture_header_flag) {
    params->frame_enable = true;
  } else {
    bool enable = false;
    if (!reader.read_bit(&enable)) {
      return fail(error, "GDF: truncated gdf_frame_enable");
    }
    params->frame_enable = enable;
    if (!enable) return true;
  }

  const GdfUnitLayout layout = DeriveGdfUnitLayout(context);
  params->blk_size = layout.blk_size;
  params->cols = layout.cols;
  params->rows = layout.rows;
  params->per_block_present = layout.per_block_present;

  if (layout.per_block_present) {
    bool per_block = false;
    if (!reader.read_bit(&per_block)) {
      return fail(error, "GDF: truncated gdf_per_block");
    }
    params->per_block = per_block;
  }

  std::uint32_t qc = 0;
  std::uint32_t scale = 0;
  if (!reader.read_bits(2U, &qc)) {
    return fail(error, "GDF: truncated gdf_pic_qc_idx");
  }
  if (!reader.read_bits(2U, &scale)) {
    return fail(error, "GDF: truncated gdf_pic_scale_idx");
  }
  if (qc >= static_cast<std::uint32_t>(kGdfNumPicQcIdx) ||
      scale >= static_cast<std::uint32_t>(kGdfNumPicScaleIdx)) {
    return fail(error, "GDF: gdf_pic_qc_idx/gdf_pic_scale_idx out of range");
  }
  params->pic_qc_idx = static_cast<int>(qc);
  params->pic_scale_idx = static_cast<int>(scale);
  params->pix_scale = 1 + params->pic_scale_idx;
  return true;
}

bool GdfSymbolAt(const GdfUnitLayout& layout, const FrameGdfParams& params,
                 int sb_size, int mi_row, int mi_col,
                 GdfSymbolPosition* position) {
  if (position == nullptr) return false;
  if (!params.frame_enable || !params.per_block) return false;
  if (layout.blk_size <= 0 || !valid_sb_size(sb_size)) return false;
  const int sb_size4 = sb_size / 4;  // Num_4x4_Blocks_Wide[SbSize]
  if (sb_size4 <= 0) return false;
  if (mi_row % sb_size4 != 0 || mi_col % sb_size4 != 0) return false;
  const int sb_row = mi_row / sb_size4;
  const int sb_col = mi_col / sb_size4;
  const int sb_per_gdf = layout.blk_size / sb_size;
  if (sb_per_gdf <= 0) return false;
  if (sb_col % sb_per_gdf != 0 || sb_row % sb_per_gdf != 0) return false;
  const int row = sb_row / sb_per_gdf;
  const int col = sb_col / sb_per_gdf;
  if (row < 0 || row >= layout.rows || col < 0 || col >= layout.cols) {
    return false;
  }
  position->row = row;
  position->col = col;
  return true;
}

void GdfBlockCoder::Reset() {
  cdf_[0] = kDefaultUseGdfCdf[0];
  cdf_[1] = kDefaultUseGdfCdf[1];
  cdf_[2] = kDefaultUseGdfCdf[2];
}

bool GdfBlockCoder::WriteAt(av2::Encoder& encoder, const GdfUnitLayout& layout,
                            const FrameGdfParams& params, int sb_size,
                            int mi_row, int mi_col, bool use_gdf, bool* coded) {
  if (coded != nullptr) *coded = false;
  GdfSymbolPosition position;
  if (!GdfSymbolAt(layout, params, sb_size, mi_row, mi_col, &position)) {
    return true;
  }
  const av2::Status status =
      encoder.encode_symbol(cdf_, 2, use_gdf ? 1 : 0);
  if (status != av2::Status::kOk) return false;
  if (coded != nullptr) *coded = true;
  return true;
}

bool GdfBlockCoder::ReadAt(av2::Decoder& decoder, const GdfUnitLayout& layout,
                           const FrameGdfParams& params, int sb_size,
                           int mi_row, int mi_col, bool* use_gdf, bool* coded) {
  if (coded != nullptr) *coded = false;
  if (use_gdf == nullptr) return false;
  *use_gdf = false;
  GdfSymbolPosition position;
  if (!GdfSymbolAt(layout, params, sb_size, mi_row, mi_col, &position)) {
    return true;
  }
  int symbol = 0;
  const av2::Status status = decoder.decode_symbol(cdf_, 2, &symbol);
  if (status != av2::Status::kOk) return false;
  *use_gdf = symbol != 0;
  if (coded != nullptr) *coded = true;
  return true;
}

}  // namespace av2enc::filters::gdf

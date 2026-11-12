// Minimal still-picture stream that signals `morph_pred` on IBC blocks.
//
// This exists only to put the new §5.20.5.4 symbol and the §7.13.3.26 process
// in front of a decoder. It is not the production M1 encoder and is not wired
// into src/main.cpp or m1_syntax.cpp.
//
// The stream is the same shape as the M1 IBC gate writer
// (src/encoder/m1/ibc_stream):
//
//   * 8-bit 4:2:0, one closed-loop-key frame, one tile, 64x64 superblocks;
//   * PARTITION_NONE / HORZ / VERT;
//   * DC intra blocks (y/uv DC, palette flag 0, EOB=0) or IBC blocks;
//   * an IBC block writes use_intrabc, skip_flag, intrabc_info, then
//     morph_pred (when the sequence/frame conditions hold).
//
// The sequence header sets enable_bawp = 1, which is the only sequence bit
// that makes morph_pred signallable. The frame header sets
// allow_screen_content_tools = 1, force_integer_mv = 1, allow_intrabc = 1.
// In the single-picture branch enable_imp_msk_bld is inferred 0.
//
// Reconstruction follows the decoder order: DC prediction or IBC copy first,
// then morphological prediction in place on luma. Chroma is untouched by
// morph_pred.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "encoder/ibc/ibc.h"
#include "encoder/m1/ibc_stream/ibc_syntax.h"

namespace av2enc::inter::morph {

struct MorphStreamConfig {
  int width = 512;
  int height = 128;
  std::uint8_t base_q_idx = 90;
  std::uint8_t seq_level_idx = 7;
  bool disable_cdf_update = true;
  // Gate writer switches. morph_pred requires both; the non-screen mode is
  // only used by the intra-syntax comparison tests.
  bool allow_screen_content_tools = true;
  bool allow_intrabc = true;
  bool enable_bawp = true;  // the only sequence bit that enables morph_pred
  // Reject BVs whose -1,-1 reference border leaves the tile. dav2d --strict 1
  // requires this; morph_pred=1 additionally requires it by §6.19.5.4.
  bool strict_border = true;
  bool enable_intra_edge_filter = false;
};

struct MorphStreamBlock {
  int mi_row = 0;
  int mi_col = 0;
  int bsize = 11;  // partition::BlockSize; 11 == BLOCK_64X32
  bool ibc = false;
  ibc::BlockVector bv{};
  m1::ibc_stream::BvSyntax syntax;
  bool skip = true;  // IBC only
  bool morph_pred = false;
  // AV2 YMode for intra blocks: 0 = DC_PRED, 9 = SMOOTH_PRED. The gate uses
  // SMOOTH to create a non-flat reconstruction without residual coding.
  int y_mode = 0;
};

struct MorphStreamSb {
  enum class Partition { kNone, kHorz, kVert };
  Partition partition = Partition::kNone;
  std::vector<MorphStreamBlock> leaves;
};

struct MorphStreamPlan {
  MorphStreamConfig config;
  std::vector<MorphStreamSb> sbs;  // raster order
};

struct MorphStreamResult {
  std::vector<std::uint8_t> frame;  // TD + SEQ + CLK
  std::vector<std::uint8_t> recon_y;
  std::vector<std::uint8_t> recon_u;
  std::vector<std::uint8_t> recon_v;
  std::size_t tile_bytes = 0;
  int ibc_blocks = 0;
  int morph_symbols = 0;  // IBC blocks where the symbol is present
  int morph_blocks = 0;   // morph_pred == 1
  int morph_ctx[3] = {0, 0, 0};
  int morph_ctx_value[3][2] = {};  // [ctx][symbol]
  int morph_fitted = 0;   // morph_pred=1 blocks with alpha/beta other than identity
  int morph_alpha_min = 256;
  int morph_alpha_max = 256;
};

// Validates every block (including §6.19.5.4 for morph_pred=1) before emitting
// a bit. Returns false with a diagnosis in `error` for any illegal plan.
bool encode_morph_stream(const MorphStreamPlan& plan, MorphStreamResult& out,
                         std::string& error);

bool write_morph_ivf(const MorphStreamResult& result,
                     const MorphStreamConfig& config, const std::string& path,
                     std::string& error);
bool write_morph_y4m(const MorphStreamResult& result,
                     const MorphStreamConfig& config, const std::string& path,
                     std::string& error);

}  // namespace av2enc::inter::morph

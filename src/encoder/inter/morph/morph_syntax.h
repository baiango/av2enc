// `morph_pred S()` — AV2 v1.0.0 §5.20.5.4 and §8.3.2.
//
// Syntax (inside read_intrabc_info(), after assign_mv(0)):
//
//     if ( FrameIsIntra && allow_screen_content_tools && enable_bawp ) {
//         morph_pred S()
//     } else {
//         morph_pred = 0
//     }
//
// CDF: TileMorphPredCdf[ctx], ctx = sum of MorphPreds[NPos[n]] over the
// §5.20.4.1 neighbour list (at most two entries, both in the current
// superblock row). The default table is
//
//     Default_Morph_Pred_Cdf[3][2 + 1] = {
//         {31715, 55, 0}, {19667, 1, 0}, {10555, 91, 0}
//     };
//
// §6.19.5.4 makes `morph_pred == 1` conformant only when
// is_offset_mv_valid(-1, -1) == 1. That check is supplied by the caller and
// enforced by morph_pred_allowed().
//
// Sequence-header interaction: in the single_picture_header_flag branch that
// every still-picture stream in this tree uses, §5.4.6 writes `enable_bawp`
// but infers `enable_imp_msk_bld = 0`. `enable_imp_msk_bld` governs only
// implicit compound mask blending (§7.13.3.14) and has no effect on
// morph_pred in AV2 v1.0.0. MorphPredGate carries the flag so tests can pin
// that fact; it is never consulted for legality.

#pragma once

#include <cstdint>
#include <string>

#include "entropy/av2/av2_entropy.h"

namespace av2enc::inter::morph {

struct MorphCdfs {
  // [ctx][cdf0, rate, count]
  std::uint16_t morph_pred[3][3];
};

// §9.3 Default_Morph_Pred_Cdf.
void init_morph_cdfs(MorphCdfs& cdfs) noexcept;

// §5.20.4.1 add_neighbor() list for NPos (buffered = false). The first two
// inside/same-SB-row entries win.
struct MorphNeighbours {
  int count = 0;
  int row[2] = {0, 0};
  int col[2] = {0, 0};
};

// `bw4`/`bh4` are the block dimensions in 4x4 units; `sb_size` is in luma
// samples (64 for the streams in this tree).
MorphNeighbours morph_neighbours(int mi_row, int mi_col, int bw4, int bh4,
                                 int mi_cols, int mi_rows, int sb_size) noexcept;

// Sum of MorphPreds over NPos (0..2). `grid` is mi_cols * mi_rows, row-major.
int morph_symbol_context(const std::uint8_t* morph_grid, int mi_cols,
                         int mi_rows, int mi_row, int mi_col, int bw4,
                         int bh4, int sb_size) noexcept;

bool write_morph_pred(av2::Encoder& enc, MorphCdfs& cdfs, int ctx, bool value,
                      std::string& error);
bool read_morph_pred(av2::Decoder& dec, MorphCdfs& cdfs, int ctx, bool& value,
                     std::string& error);

// The three conditions in the syntax table. When false, morph_pred is
// inferred 0 and no symbol is coded.
struct MorphPredGate {
  bool enable_bawp = false;
  bool allow_screen_content_tools = false;
  bool frame_is_intra = true;
  bool use_intrabc = false;
  // No normative effect on morph_pred; see the header comment.
  bool enable_imp_msk_bld = false;
};

bool morph_pred_present(const MorphPredGate& gate) noexcept;

// Syntax condition + use_intrabc + the §6.19.5.4 conformance condition
// is_offset_mv_valid(-1, -1) == 1.
bool morph_pred_allowed(const MorphPredGate& gate, bool offset_mv_valid,
                        std::string* reason = nullptr);

}  // namespace av2enc::inter::morph

// AV2 v1.0.0 M1 multiple-reference-line (MRL) per-block syntax.
//
// Normative source: AV2 Bitstream & Decoding Process Specification v1.0.0
//   §5.20.5.5  read_intra_y_mode(): after YMode/AngleDeltaY,
//                if (enable_mrls && is_directional_mode(YMode)) {
//                  mrl_index; if (mrl_index > 0) mrl_sec_index;
//                } else { mrl_index = 0 }
//   §5.20.4.1  decode_block(): UsesMrls[r][c] =
//                mrl_index > 0 ? (mrl_sec_index ? 2 : 1) : 0
//   §8.3.x     TileMrlIndexCdf[ctx] / TileMrlSecIndexCdf[ctx]
//   §9.3       Default_Mrl_Index_Cdf[3][5],
//              Default_Mrl_Sec_Index_Cdf[3][3]
//
// This file owns *only* those two symbols, their contexts, and the validity
// rules.  It does not decide anything, does not write partitions, and does not
// reconstruct: it consumes one leaf at a time in decode order, exactly like the
// M1 syntax writer's write_block().  The prediction rules live in
// src/encoder/mrl/ (MrlIndex) and the interaction helpers in mrl_interaction.h.
//
// Context derivation (§8.3, via §5.20.4.1 add_neighbor):
//
//   NNum = 0
//   add_neighbor(r + bh4 - 1, c - 1)   // below-left
//   add_neighbor(r - 1, c + bw4 - 1)   // above-right
//   add_neighbor(r, c - 1)             // left
//   add_neighbor(r - 1, c)             // above
//
//   add_neighbor(nRow, nCol) keeps at most two positions, requires
//   is_inside(nRow, nCol) and excludes a neighbour in a different superblock
//   row:
//     aboveSbBoundary = (MiRow >> Mi_Width_Log2[SbSize]) !=
//                       (nRow  >> Mi_Width_Log2[SbSize])
//
//   mrl_index ctx     = sum over NPos of (UsesMrls[NPos] > 0)     // 0..2
//   mrl_sec_index ctx = sum over NPos of (UsesMrls[NPos] == 2)    // 0..2
//
// MRL is luma-only: chroma forces MrlIndex = 0 (§7.13.2.1), and the syntax is
// gated on is_directional_mode(YMode).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "codec/entropy/av2/av2_entropy.h"

namespace av2enc::still::mrl_stream {

// MRL_INDEX_CONTEXTS (§3) and the number of mrl_index values (0..3).
inline constexpr int kMrlIndexContexts = 3;
inline constexpr int kMaxMrlIndex = 3;
inline constexpr int kMrlIndexSymbols = kMaxMrlIndex + 1;  // 4

// Default_Mrl_Index_Cdf[3][4 + 1] (§9.3), byte-for-byte.
inline constexpr std::uint16_t kDefaultMrlIndexCdf[kMrlIndexContexts]
                                                  [kMrlIndexSymbols + 1] = {
    {29573, 31193, 32023, 78, 0},
    {21812, 27066, 30279, 75, 0},
    {16076, 23806, 28762, 1, 0},
};

// Default_Mrl_Sec_Index_Cdf[3][2 + 1] (§9.3), byte-for-byte.
inline constexpr std::uint16_t kDefaultMrlSecIndexCdf[kMrlIndexContexts][3] = {
    {19678, 6, 0},
    {12287, 6, 0},
    {9574, 6, 0},
};

// One mutable copy per tile.  Both arrays adapt in place unless the entropy
// coder has disable_cdf_update set.
struct MrlCdfs {
  std::uint16_t index[kMrlIndexContexts][kMrlIndexSymbols + 1];
  std::uint16_t sec[kMrlIndexContexts][3];
};

void init_mrl_cdfs(MrlCdfs& cdfs) noexcept;

// A block described in 4x4 mode-info units.  `bw4`/`bh4` are
// Num_4x4_Blocks_Wide/High of MiSize.
struct MrlBlockSyntax {
  int mi_row = 0;
  int mi_col = 0;
  int bw4 = 0;
  int bh4 = 0;

  int width() const { return bw4 * 4; }
  int height() const { return bh4 * 4; }
};

struct NeighborPos {
  int row = 0;
  int col = 0;
};

// §5.20.4.1 add_neighbor() over the four probes.  Returns NNum (0..2) and
// fills out[0..NNum).  `sb_size_mi` is the superblock size in 4x4 mode-info
// units (16 for 64x64, 32 for 128x128); the spec's
// `MiRow >> Mi_Width_Log2[SbSize]` test is equivalent to division by it.
int collect_npos(const MrlBlockSyntax& block, int mi_rows, int mi_cols,
                 int sb_size_mi, NeighborPos out[2]) noexcept;

// UsesMrls[] grid plus the two context derivations.  Cells start at 0, matching
// the decoder's cleared frame state.  update() is called once per coded block
// (including non-directional blocks, which store 0).
class MrlContext {
 public:
  MrlContext() = default;
  MrlContext(int mi_rows, int mi_cols, int sb_size_mi);

  void reset(int mi_rows, int mi_cols, int sb_size_mi);

  int mi_rows() const { return mi_rows_; }
  int mi_cols() const { return mi_cols_; }
  int sb_size_mi() const { return sb_size_mi_; }

  // 0 outside the frame.  Values are 0, 1 (mrl_index > 0) or 2 (+ secondary).
  int uses_at(int mi_row, int mi_col) const noexcept;

  // §8.3 contexts from the NPos set of the *current* block (before update).
  int index_context(const MrlBlockSyntax& block) const noexcept;
  int sec_context(const MrlBlockSyntax& block) const noexcept;

  // Stores UsesMrls = mrl_index > 0 ? (mrl_sec_index ? 2 : 1) : 0 over the
  // block footprint.  Clips to the frame like the decoder's r/c loops.
  void update(const MrlBlockSyntax& block, int mrl_index,
              bool mrl_sec_index) noexcept;

 private:
  int mi_rows_ = 0;
  int mi_cols_ = 0;
  int sb_size_mi_ = 16;
  std::vector<std::uint8_t> uses_;
};

// The two per-block syntax values.  `mrl_sec_index` is the raw syntax bit; use
// mrl_secondary_applies() before feeding it to a predictor.
struct MrlDecision {
  int mrl_index = 0;
  bool mrl_sec_index = false;
};

enum class MrlSyntaxStatus : std::uint8_t {
  kOk = 0,
  kInvalidMode,       // non-directional YMode with mrl_index != 0 / secondary
  kInvalidIndex,      // mrl_index outside 0..3
  kInvalidSecondary,  // encoder tried to signal secondary at 4x4 (or with 0)
  kEntropyError,      // av2::Encoder/Decoder failure
};

const char* mrl_syntax_status_name(MrlSyntaxStatus status) noexcept;

// §5.20.5.5 is_directional_mode(YMode): 1..8 (V, H, D45, D135, D113, D157,
// D203, D67).  0 = DC_PRED and 9..12 are non-directional.
bool is_directional_y_mode(int y_mode) noexcept;

// The decoder's `not4x4` test.  §7.13.2.7 applies the secondary blend only when
// MrlIndex > 0, mrl_sec_index == 1 and the block is not exactly 4x4.
bool mrl_secondary_applies(int block_w, int block_h, int mrl_index,
                           bool mrl_sec_index) noexcept;

// Exact writer.  Writes nothing for a non-directional mode (and requires the
// decision to be {0,false}); otherwise writes mrl_index, then mrl_sec_index
// when mrl_index > 0.  Updates `context` on success.
//
// Validity, before any bit is written:
//   * non-directional       -> mrl_index must be 0, secondary must be false
//   * mrl_index             -> 0..3
//   * mrl_index == 0        -> secondary must be false
//   * secondary at 4x4      -> rejected (the decoder ignores the bit, so an
//                              encoder must not spend it; core
//                              mrl::validate_params() also rejects it)
MrlSyntaxStatus write_mrl_syntax(av2::Encoder& encoder, MrlCdfs& cdfs,
                                 MrlContext& context,
                                 const MrlBlockSyntax& block, int y_mode,
                                 const MrlDecision& decision,
                                 std::string* error);

// Exact reader.  Mirrors the syntax above and rebuilds the same UsesMrls
// state.  At 4x4 a decoded mrl_sec_index == 1 is *accepted* (the bit is
// syntactically present and the decoder ignores its effect); callers must use
// mrl_secondary_applies() to feed prediction.
MrlSyntaxStatus read_mrl_syntax(av2::Decoder& decoder, MrlCdfs& cdfs,
                                MrlContext& context,
                                const MrlBlockSyntax& block, int y_mode,
                                MrlDecision* decision, std::string* error);

}  // namespace av2enc::still::mrl_stream

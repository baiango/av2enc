// AV2 v1.0.0 §7.13.3.15 — save subpu size process.
//
// Normative source: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7), §7.13.3.15, p. 484:
//
//   subX = (plane > 0) ? SubsamplingX : 0
//   subY = (plane > 0) ? SubsamplingY : 0
//   subPuSz = find_tx_size(w, h)
//   lumaRow = y >> (2 - subY)
//   lumaCol = x >> (2 - subX)
//   for (r = 0; r < h >> (MI_SIZE_LOG2 - subY); r++)
//     for (c = 0; c < w >> (MI_SIZE_LOG2 - subX); c++) {
//       SubPuColBase[plane > 0][lumaRow + r][lumaCol + c] = lumaCol
//       SubPuRowBase[plane > 0][lumaRow + r][lumaCol + c] = lumaRow
//       SubPuSize[plane > 0][lumaRow + r][lumaCol + c] = subPuSz
//     }
//
// §7.13.3.14 Predict subblock invokes this process once per plane, after the
// per-reference-list prediction loop, with the subblock origin and size.  It
// early-terminates when the region is the whole prediction unit:
//
//   if (w == PuWidth && h == PuHeight) return;
//
// `find_tx_size(w, h)` (§5.20.6.3) linearly searches Tx_Width/Tx_Height over
// TX_SIZES_ALL for an exact match and returns TX_INVALID otherwise.
//
// Why this is a "missing bit": the production M2 decoder model
// (`src/encoder/filters/deblock/deblock_core.h:118-127`) stores a single
// `sub_pu_size` per MI cell and documents it as `Max_Tx_Size_Rect[MiSize]`.
// That is the value used when `allow_df_sub_pu == 0` (intra, or an inter
// block with no sub-PU partition).  §7.13.3.15 instead stores
// `find_tx_size(w, h)` of the *predicted subblock* and a base coordinate.
// With `allow_df_sub_pu == 1` the two disagree for every sub-PU.  The M2 path
// never enables that flag (`m2_headers.cpp` writes `enable_df_sub_pu = 0`), so
// the reference is dormant, but it is the value a conforming decoder derives.
//
// Reference-MV path call site: §7.13.3.14 predict_subblock, which is reached
// from TIP/OPFL/refine-MV/compound sub-PU loops.  In this tree the analogues
// are `src/encoder/inter/temporal/refinemv.cpp` (offset MVs),
// `src/encoder/inter/temporal/opfl.cpp` (OpflMvs) and the per-plane predictor
// `src/encoder/m2/m2_frame.cpp:488` (`inter::mc::predict_plane`).  The latter
// is where save_subPU would be called after prediction for a real decoder
// model.
//
// This file is a self-contained reference: it owns the TX_SIZES_ALL tables so
// the test can cross-check them against the deblock component.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace av2enc::m1::missing_bits {

// MI_SIZE_LOG2 (§3).
inline constexpr int kMiSizeLog2 = 2;
inline constexpr int kMiSize = 1 << kMiSizeLog2;

// AV2 TX_SIZE enum (TX_SIZES_ALL order, §6.19.6.1).
enum class TxSize : std::uint8_t {
  k4x4 = 0,
  k8x8 = 1,
  k16x16 = 2,
  k32x32 = 3,
  k64x64 = 4,
  k4x8 = 5,
  k8x4 = 6,
  k8x16 = 7,
  k16x8 = 8,
  k16x32 = 9,
  k32x16 = 10,
  k32x64 = 11,
  k64x32 = 12,
  k4x16 = 13,
  k16x4 = 14,
  k8x32 = 15,
  k32x8 = 16,
  k16x64 = 17,
  k64x16 = 18,
  k4x32 = 19,
  k32x4 = 20,
  k8x64 = 21,
  k64x8 = 22,
  k4x64 = 23,
  k64x4 = 24,
  kCount = 25,
};

inline constexpr int kTxSizesAll = 25;
inline constexpr int kTxInvalid = -1;

// Tx_Width[TX_SIZES_ALL] / Tx_Height[TX_SIZES_ALL] (§7.16.2 tables, p. 614).
inline constexpr int kTxWidth[kTxSizesAll] = {
    4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64, 4,
    32, 8, 64, 4, 64};
inline constexpr int kTxHeight[kTxSizesAll] = {
    4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16, 32,
    4, 64, 8, 64, 4};

constexpr int tx_width(TxSize size) noexcept {
  return kTxWidth[static_cast<int>(size)];
}
constexpr int tx_height(TxSize size) noexcept {
  return kTxHeight[static_cast<int>(size)];
}

// §5.20.6.3 find_tx_size(): returns the TX_SIZE index whose Tx_Width and
// Tx_Height match exactly, or kTxInvalid.
int find_tx_size(int w, int h) noexcept;

const char* tx_size_name(TxSize size) noexcept;

// Arrays indexed exactly like the decoder's per-MI state:
//   SubPuSize[plane > 0][row][col]
//   SubPuColBase[plane > 0][row][col]
//   SubPuRowBase[plane > 0][row][col]
// The grid is `mi_rows x mi_cols` in *luma* MI units for both planes.
struct SubPuGrid {
  int mi_rows = 0;
  int mi_cols = 0;
  // [plane_is_chroma][row * mi_cols + col]
  std::vector<std::uint8_t> size[2];
  std::vector<std::int32_t> col_base[2];
  std::vector<std::int32_t> row_base[2];

  void reset(int rows, int cols);
  const std::uint8_t& size_at(int plane, int row, int col) const;
  const std::int32_t& col_base_at(int plane, int row, int col) const;
  const std::int32_t& row_base_at(int plane, int row, int col) const;
};

// Full §7.13.3.15 process.
//
// plane: 0 luma, 1 U, 2 V.
// x,y,w,h: region origin/extent in *plane* samples.
// subsampling_x/y: 0 or 1 (chroma only; forced 0 for plane 0).
// pu_width/pu_height: PuWidth/PuHeight in *plane* samples.  The process
//   returns true without touching the grid when w == pu_width && h == pu_height.
//
// Returns false (and sets *error when non-null) for an out-of-range plane, a
// region that has no matching TX size, or a grid write outside mi_rows/mi_cols.
bool save_subpu(int plane, int x, int y, int w, int h, int subsampling_x,
                int subsampling_y, int pu_width, int pu_height,
                SubPuGrid& grid, std::string* error = nullptr) noexcept;

}  // namespace av2enc::m1::missing_bits

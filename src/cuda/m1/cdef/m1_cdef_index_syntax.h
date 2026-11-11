// §5.20.10.1 read_cdef(): encoder counterpart + decoder mirror.
//
// This is the piece of M1 syntax that CDEF needs and that the current M1
// writer does not have.  It is deliberately independent of the entropy coder
// and of the syntax writer: both are injected as callbacks, so the M1 owner can
// call it from `m1_syntax.cpp` with `m1::syntax`'s own `sym()` helper and the
// component test can call it with `av2::Encoder` / `av2::Decoder`.
//
// Normative behaviour reproduced here:
//
//   read_cdef() {
//     if ((skip_flag && !cdef_on_skip_txfm_frame_enable) || !cdef_frame_enable)
//       return
//     cdefSize4 = 16                       // Num_4x4_Blocks_Wide[BLOCK_64X64]
//     cdefMask4 = ~(cdefSize4 - 1)
//     r = MiRow & cdefMask4
//     c = MiCol & cdefMask4
//     if (cdef_idx[r][c] == -1) {
//       if (CdefStrengths == 1) cdef_idx[r][c] = 0
//       else {
//         cdef_index0 S()
//         if (cdef_index0) cdef_idx[r][c] = 0
//         else if (CdefStrengths == 2) cdef_idx[r][c] = 1
//         else { cdef_index_minus_1 S(); cdef_idx[r][c] = ... + 1 }
//       }
//     }
//     // copy to all 64x64 origins covered by the current block
//   }
//
// Two consequences that matter for the encoder:
//   * With CdefStrengths == 1 nothing is coded for any block: every block is
//     forced to index 0 and CDEF cannot be skipped per block.
//   * With CdefStrengths >= 2 the index is always in [0, CdefStrengths); there
//     is no `-1` syntax.  "Do not filter this block" therefore requires an
//     all-zero strength set in the header (see cdef_search.h).
//
// The cdef_index0 context is the §8.3.1 derivation from the already-coded
// 64x64 neighbours.  It is derived here exactly once, in one place.
#ifndef AV2ENC_CUDA_M1_CDEF_INDEX_SYNTAX_H_
#define AV2ENC_CUDA_M1_CDEF_INDEX_SYNTAX_H_

#include <cstdint>
#include <vector>

namespace av2enc::cuda::m1 {

// cdef_idx grid indexed like the spec: [mi_row][mi_col] in luma MI units.
// Only the 64x64 origin entries are meaningful; everything else stays -1.
struct CdefIndexGrid {
  int mi_cols = 0;
  int mi_rows = 0;
  std::vector<std::int8_t> values;  // mi_cols * mi_rows, initialised to -1

  void reset(int cols, int rows) {
    mi_cols = cols;
    mi_rows = rows;
    values.assign(static_cast<std::size_t>(cols) *
                      static_cast<std::size_t>(rows),
                  static_cast<std::int8_t>(-1));
  }
  std::int8_t at(int r, int c) const {
    return values[static_cast<std::size_t>(r) *
                      static_cast<std::size_t>(mi_cols) +
                  static_cast<std::size_t>(c)];
  }
  void set(int r, int c, std::int8_t v) {
    values[static_cast<std::size_t>(r) *
                static_cast<std::size_t>(mi_cols) +
            static_cast<std::size_t>(c)] = v;
  }
  // One entry per 64x64 block, row-major (what the filter stage consumes).
  std::vector<std::int8_t> origins(int n64_cols, int n64_rows) const {
    std::vector<std::int8_t> out(static_cast<std::size_t>(n64_cols) *
                                 static_cast<std::size_t>(n64_rows));
    for (int br = 0; br < n64_rows; ++br) {
      for (int bc = 0; bc < n64_cols; ++bc) {
        const int r = br * 16 < mi_rows ? br * 16 : mi_rows - 1;
        const int c = bc * 16 < mi_cols ? bc * 16 : mi_cols - 1;
        out[static_cast<std::size_t>(br) * static_cast<std::size_t>(n64_cols) +
            static_cast<std::size_t>(bc)] = at(r, c);
      }
    }
    return out;
  }
};

// Mutable CDF pointers.  `index0` is 4x3 (CDEF_STRENGTH_INDEX0_CTX = 4),
// `minus1` is CdefStrengths entries and is unused for strengths <= 2.
struct CdefIndexCdfs {
  std::uint16_t* index0 = nullptr;
  std::uint16_t* minus1 = nullptr;
};

// §8.3.1 cdef_index0 context.  `mi_row`/`mi_col` are the current coding
// block's origin in luma MI units; `tile_mi_col_start`/`tile_mi_row_start` are
// MiColStart/MiRowStart for that block (0 for a single tile).
inline int cdef_index0_context(const CdefIndexGrid& grid, int sb_width_log2,
                               int tile_mi_col_start, int tile_mi_row_start,
                               int mi_row, int mi_col) {
  constexpr int kCdefSize4 = 16;
  constexpr int kCdefMask4 = ~(kCdefSize4 - 1);
  int ctx = 0;
  int cnt = 0;
  const int left_col = (mi_col - kCdefSize4) & kCdefMask4;
  const int left_row = mi_row & kCdefMask4;
  if (left_col >= tile_mi_col_start && left_row >= 0 &&
      left_row < grid.mi_rows && left_col >= 0 && left_col < grid.mi_cols) {
    ctx += grid.at(left_row, left_col) == 0 ? 1 : 0;
    ++cnt;
  }
  const int above_col = mi_col & kCdefMask4;
  const int above_row = (mi_row - kCdefSize4) & kCdefMask4;
  const int cur_sb_row = mi_row >> sb_width_log2;
  const int above_sb_row = above_row >> sb_width_log2;
  if (above_row >= tile_mi_row_start && above_sb_row == cur_sb_row &&
      above_row >= 0 && above_row < grid.mi_rows && above_col >= 0 &&
      above_col < grid.mi_cols) {
    ctx += grid.at(above_row, above_col) == 0 ? 1 : 0;
    ++cnt;
  }
  if (ctx != 0 && cnt == ctx) {
    ++ctx;
  }
  return ctx;
}

// One 64x64 block's index, §5.20.10.1.
//
// Encoder side.  `emit(cdf, n, symbol)` must match `av2::Encoder::encode_symbol`
// (or the M1 syntax writer's `sym()`), i.e. it writes one arithmetic symbol and
// updates the CDF in place.  Returns false when `idx` is not codeable with the
// supplied CdefStrengths.
//
// The caller invokes this once per coding block; a repeat call for a 64x64
// block whose origin was already coded is a no-op, mirroring
// `cdef_idx[r][c] != -1`.
template <typename Emit>
bool cdef_encode_block_index(CdefIndexGrid* grid, int strengths,
                             int sb_width_log2, int tile_mi_col_start,
                             int tile_mi_row_start, int mi_row, int mi_col,
                             int idx, const CdefIndexCdfs& cdfs, Emit&& emit) {
  constexpr int kCdefSize4 = 16;
  constexpr int kCdefMask4 = ~(kCdefSize4 - 1);
  if (grid == nullptr || strengths < 1 || strengths > 8) {
    return false;
  }
  const int r = mi_row & kCdefMask4;
  const int c = mi_col & kCdefMask4;
  if (r < 0 || c < 0 || r >= grid->mi_rows || c >= grid->mi_cols) {
    return false;
  }
  if (grid->at(r, c) != -1) {
    return true;  // already coded by an earlier block of this 64x64
  }
  if (strengths == 1) {
    if (idx != 0) {
      return false;
    }
    grid->set(r, c, 0);
    return true;
  }
  if (idx < 0 || idx >= strengths) {
    return false;
  }
  const int ctx =
      cdef_index0_context(*grid, sb_width_log2, tile_mi_col_start,
                          tile_mi_row_start, mi_row, mi_col);
  const int index0 = idx == 0 ? 1 : 0;
  if (!emit(cdfs.index0 + ctx * 3, 2, index0)) {
    return false;
  }
  if (index0 == 0) {
    if (strengths == 2) {
      // cdef_idx == 1 without a further symbol.
    } else {
      if (!emit(cdfs.minus1, strengths - 1, idx - 1)) {
        return false;
      }
    }
  }
  grid->set(r, c, static_cast<std::int8_t>(idx));
  return true;
}

// Decoder mirror.  `read(cdf, n)` returns the symbol or a negative value on
// failure.  Writes the index into `*idx_out` and into the grid.
template <typename Read>
bool cdef_decode_block_index(CdefIndexGrid* grid, int strengths,
                             int sb_width_log2, int tile_mi_col_start,
                             int tile_mi_row_start, int mi_row, int mi_col,
                             const CdefIndexCdfs& cdfs, Read&& read,
                             int* idx_out) {
  constexpr int kCdefSize4 = 16;
  constexpr int kCdefMask4 = ~(kCdefSize4 - 1);
  if (grid == nullptr || strengths < 1 || strengths > 8 || idx_out == nullptr) {
    return false;
  }
  const int r = mi_row & kCdefMask4;
  const int c = mi_col & kCdefMask4;
  if (r < 0 || c < 0 || r >= grid->mi_rows || c >= grid->mi_cols) {
    return false;
  }
  if (grid->at(r, c) != -1) {
    *idx_out = grid->at(r, c);
    return true;
  }
  int idx = 0;
  if (strengths != 1) {
    const int ctx =
        cdef_index0_context(*grid, sb_width_log2, tile_mi_col_start,
                            tile_mi_row_start, mi_row, mi_col);
    const int index0 = read(cdfs.index0 + ctx * 3, 2);
    if (index0 < 0) {
      return false;
    }
    if (index0 != 0) {
      idx = 0;
    } else if (strengths == 2) {
      idx = 1;
    } else {
      const int minus1 = read(cdfs.minus1, strengths - 1);
      if (minus1 < 0) {
        return false;
      }
      idx = minus1 + 1;
    }
  }
  grid->set(r, c, static_cast<std::int8_t>(idx));
  *idx_out = idx;
  return true;
}

}  // namespace av2enc::cuda::m1

#endif  // AV2ENC_CUDA_M1_CDEF_INDEX_SYNTAX_H_

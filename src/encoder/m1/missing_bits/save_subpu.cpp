// AV2 v1.0.0 §7.13.3.15 — save subpu size process.

#include "encoder/m1/missing_bits/save_subpu.h"

#include <cstddef>

namespace av2enc::m1::missing_bits {
namespace {

const char* const kNames[kTxSizesAll] = {
    "TX_4X4",   "TX_8X8",   "TX_16X16", "TX_32X32", "TX_64X64",
    "TX_4X8",   "TX_8X4",   "TX_8X16",  "TX_16X8",  "TX_16X32",
    "TX_32X16", "TX_32X64", "TX_64X32", "TX_4X16",  "TX_16X4",
    "TX_8X32",  "TX_32X8",  "TX_16X64", "TX_64X16", "TX_4X32",
    "TX_32X4",  "TX_8X64",  "TX_64X8",  "TX_4X64",  "TX_64X4"};

bool fail(std::string* error, const char* message) {
  if (error != nullptr) {
    *error = message;
  }
  return false;
}

}  // namespace

int find_tx_size(int w, int h) noexcept {
  for (int index = 0; index < kTxSizesAll; ++index) {
    if (kTxWidth[index] == w && kTxHeight[index] == h) {
      return index;
    }
  }
  return kTxInvalid;
}

const char* tx_size_name(TxSize size) noexcept {
  const int index = static_cast<int>(size);
  if (index < 0 || index >= kTxSizesAll) {
    return "TX_INVALID";
  }
  return kNames[index];
}

void SubPuGrid::reset(int rows, int cols) {
  mi_rows = rows;
  mi_cols = cols;
  const std::size_t count =
      static_cast<std::size_t>(rows < 0 ? 0 : rows) *
      static_cast<std::size_t>(cols < 0 ? 0 : cols);
  for (int plane = 0; plane < 2; ++plane) {
    size[plane].assign(count, 0);
    col_base[plane].assign(count, 0);
    row_base[plane].assign(count, 0);
  }
}

const std::uint8_t& SubPuGrid::size_at(int plane, int row, int col) const {
  return size[plane][static_cast<std::size_t>(row) *
                         static_cast<std::size_t>(mi_cols) +
                     static_cast<std::size_t>(col)];
}

const std::int32_t& SubPuGrid::col_base_at(int plane, int row,
                                           int col) const {
  return col_base[plane][static_cast<std::size_t>(row) *
                             static_cast<std::size_t>(mi_cols) +
                         static_cast<std::size_t>(col)];
}

const std::int32_t& SubPuGrid::row_base_at(int plane, int row,
                                           int col) const {
  return row_base[plane][static_cast<std::size_t>(row) *
                             static_cast<std::size_t>(mi_cols) +
                         static_cast<std::size_t>(col)];
}

bool save_subpu(int plane, int x, int y, int w, int h, int subsampling_x,
                int subsampling_y, int pu_width, int pu_height,
                SubPuGrid& grid, std::string* error) noexcept {
  if (plane < 0 || plane > 2) {
    return fail(error, "plane out of range");
  }
  if (x < 0 || y < 0 || w <= 0 || h <= 0) {
    return fail(error, "invalid prediction region");
  }
  if (pu_width < 0 || pu_height < 0) {
    return fail(error, "invalid prediction unit size");
  }

  // §7.13.3.15 early termination.
  if (w == pu_width && h == pu_height) {
    return true;
  }

  const int sub_x = (plane > 0) ? (subsampling_x != 0 ? 1 : 0) : 0;
  const int sub_y = (plane > 0) ? (subsampling_y != 0 ? 1 : 0) : 0;

  const int sub_pu_size = find_tx_size(w, h);
  if (sub_pu_size == kTxInvalid) {
    return fail(error, "find_tx_size found no matching TX size");
  }

  const int luma_row = y >> (2 - sub_y);
  const int luma_col = x >> (2 - sub_x);
  const int rows = h >> (kMiSizeLog2 - sub_y);
  const int cols = w >> (kMiSizeLog2 - sub_x);
  if (rows <= 0 || cols <= 0) {
    return fail(error, "prediction region smaller than one MI cell");
  }

  const int plane_index = plane > 0 ? 1 : 0;
  if (luma_row < 0 || luma_col < 0 || luma_row + rows > grid.mi_rows ||
      luma_col + cols > grid.mi_cols) {
    return fail(error, "SubPu arrays write outside MiRows/MiCols");
  }

  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      const std::size_t index =
          static_cast<std::size_t>(luma_row + r) *
              static_cast<std::size_t>(grid.mi_cols) +
          static_cast<std::size_t>(luma_col + c);
      grid.size[plane_index][index] = static_cast<std::uint8_t>(sub_pu_size);
      grid.col_base[plane_index][index] = luma_col;
      grid.row_base[plane_index][index] = luma_row;
    }
  }
  return true;
}

}  // namespace av2enc::m1::missing_bits

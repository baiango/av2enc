// §5.20.7.30 scan / row-column / LF-limit generation.  See tx_scan.h.
#include "encoder/m2/tx_partition/tx_scan.h"

#include <algorithm>

#include "encoder/m2/tx_partition/tx_partition.h"
#include "encoder/m2/tx_partition/tx_partition_tables.h"

namespace av2enc::m2::txpartition {
namespace {

constexpr int kMaxAdjDim = 32;

bool valid_tx_size(int tx_size) { return tx_size >= 0 && tx_size < 25; }

}  // namespace

const char* tx_scan_class_name(TxScanClass cls) {
  switch (cls) {
    case TxScanClass::k2D:
      return "TX_CLASS_2D";
    case TxScanClass::kHoriz:
      return "TX_CLASS_HORIZ";
    case TxScanClass::kVert:
      return "TX_CLASS_VERT";
  }
  return "unknown";
}

TxScanClass tx_class_of(int tx_type) {
  // §5.20.7.30: V_DCT / V_ADST / V_FLIPADST are TX_CLASS_VERT,
  // H_DCT / H_ADST / H_FLIPADST are TX_CLASS_HORIZ, everything else
  // (including IDTX) is TX_CLASS_2D.
  switch (tx_type) {
    case 10:  // V_DCT
    case 12:  // V_ADST
    case 14:  // V_FLIPADST
      return TxScanClass::kVert;
    case 11:  // H_DCT
    case 13:  // H_ADST
    case 15:  // H_FLIPADST
      return TxScanClass::kHoriz;
    default:
      return TxScanClass::k2D;
  }
}

int tx_scan_width(int tx_size) {
  if (!valid_tx_size(tx_size)) return 0;
  return std::min(kTxPartitionTxWidth[tx_size], kMaxAdjDim);
}

int tx_scan_height(int tx_size) {
  if (!valid_tx_size(tx_size)) return 0;
  return std::min(kTxPartitionTxHeight[tx_size], kMaxAdjDim);
}

int tx_scan_area(int tx_size) {
  return tx_scan_width(tx_size) * tx_scan_height(tx_size);
}

bool get_scan(int tx_size, TxScanClass cls, std::span<std::uint16_t> out) {
  const int w = tx_scan_width(tx_size);
  const int h = tx_scan_height(tx_size);
  const int area = w * h;
  if (area <= 0 || out.size() < static_cast<std::size_t>(area)) {
    return false;
  }
  int c = 0;
  switch (cls) {
    case TxScanClass::kVert:
      // Row-major: for each row (y) walk all columns (x).
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          out[static_cast<std::size_t>(c++)] =
              static_cast<std::uint16_t>(y * w + x);
        }
      }
      return true;
    case TxScanClass::kHoriz:
      // Column-major: for each column (x) walk all rows (y).
      for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) {
          out[static_cast<std::size_t>(c++)] =
              static_cast<std::uint16_t>(y * w + x);
        }
      }
      return true;
    case TxScanClass::k2D: {
      // Literal §5.20.7.30 diagonal walk.
      int x = 0;
      int y = 0;
      for (int i = 0; i < area; ++i) {
        out[static_cast<std::size_t>(c++)] =
            static_cast<std::uint16_t>(y * w + x);
        ++x;
        --y;
        if (y < 0 || x >= w) {
          ++x;
          const int s = std::min(x, h - 1 - y);
          x -= s;
          y += s;
        }
      }
      return true;
    }
  }
  return false;
}

bool get_tx_row_col(int pos, int tx_size, int* row, int* col) {
  if (row == nullptr || col == nullptr || !valid_tx_size(tx_size)) {
    return false;
  }
  const int area = tx_scan_area(tx_size);
  if (pos < 0 || pos >= area) return false;
  // §5.20.7.30 uses Tx_Width_Log2[Adjusted_Tx_Size[txSz]]: the row/col are in
  // the adjusted (capped-32) grid, not the coded grid.
  const int adj = adjusted_tx_size(tx_size);
  if (adj < 0) return false;
  const int bwl = kTxPartitionTxWidthLog2[adj];
  const int r = pos >> bwl;
  *row = r;
  *col = pos - (r << bwl);
  return true;
}

bool get_lf_limits(int row, int col, TxScanClass cls, int plane) {
  switch (cls) {
    case TxScanClass::k2D:
      return plane == 0 ? (row + col) < 4 : (row + col) < 1;
    case TxScanClass::kHoriz:
      return plane == 0 ? col < 2 : col < 1;
    case TxScanClass::kVert:
      return plane == 0 ? row < 2 : row < 1;
  }
  return false;
}

}  // namespace av2enc::m2::txpartition

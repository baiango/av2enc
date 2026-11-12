// AV2 v1.0.0 §5.20.7.30 get_scan(), get_tx_row_col(), get_lf_limits().
//
// `spec/tables/scan.json` only contains the four square TX_CLASS_2D scans
// (4x4, 8x8, 16x16, 32x32).  §5.20.7.30 defines the scan for every transform
// size and all three transform classes, so this file generates the missing
// rectangular / HORIZ / VERT scans from the normative algorithm.
//
// The generated scan is over the *adjusted* transform grid:
// w = Min(Tx_Width,32), h = Min(Tx_Height,32).  That is the grid
// `get_tx_row_col()` uses and
// the same `w`/`h` the coefficient writer iterates.
#pragma once

#include <cstdint>
#include <span>

namespace av2enc::video::txpartition {

// §9.1 TX_CLASS_* values (the spec numbers are the enum values).
enum class TxScanClass : std::uint8_t {
  k2D = 0,     // TX_CLASS_2D
  kHoriz = 1,  // TX_CLASS_HORIZ
  kVert = 2,   // TX_CLASS_VERT
};

const char* tx_scan_class_name(TxScanClass cls);

// §5.20.7.30 get_tx_class(txType) for the 16 TX_TYPE values.
TxScanClass tx_class_of(int tx_type);

// Min(Tx_Width,32) / Min(Tx_Height,32) for a TX size index 0..24.
int tx_scan_width(int tx_size);
int tx_scan_height(int tx_size);
int tx_scan_area(int tx_size);

// Writes w*h raster positions into `out` in scan order.  `out` must have at
// least tx_scan_area(tx_size) entries.  Returns false for an invalid size.
bool get_scan(int tx_size, TxScanClass cls, std::span<std::uint16_t> out);

// §5.20.7.30 get_tx_row_col(pos, txSz) on the adjusted grid.
bool get_tx_row_col(int pos, int tx_size, int* row, int* col);

// §5.20.7.30 get_lf_limits(row, col, txClass, plane).
bool get_lf_limits(int row, int col, TxScanClass cls, int plane);

}  // namespace av2enc::video::txpartition

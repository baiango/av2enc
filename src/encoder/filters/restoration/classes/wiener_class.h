// §7.20.4 pixel-classification support for the encoder-side Wiener search.
//
// This module implements the *algorithm* of §7.20.4 up to the final table
// lookups.  The three Pc_Wiener_* lookup tables that turn the 12-bit LUT input
// into a FilterClass (Pc_Wiener_Lut_To_Class) and that map FilterClass to a
// sub-class (Pc_Wiener_Sub_Classify2) are NOT printed anywhere in the
// specification text: §9.8 says only that they are "defined in the ... header
// file".  They are never invented here.  Callers that own the normative
// attachments pass them in; without them the multi-class frame-filter path
// returns kMissingNormativeTable and only NumFilterClasses == 1 remains
// available (for one class the subclass lookup is not consulted at all).
//
// The remaining constants and arithmetic (features, Mode_Weights,
// Mode_Offsets, Pc_Wiener_Normalizer, get_qval_given_tskip) are in the spec
// text and live in wiener_tables.h.

#ifndef AV2ENC_ENCODER_FILTERS_RESTORATION_CLASSES_WIENER_CLASS_H_
#define AV2ENC_ENCODER_FILTERS_RESTORATION_CLASSES_WIENER_CLASS_H_

#include <cstdint>
#include <vector>

#include "encoder/filters/restoration/classes/wiener_tables.h"
#include "encoder/filters/restoration/wiener_core.h"

namespace av2enc::restoration::classes {

enum class WienerTableStatus {
  kOk = 0,
  kMissingNormativeTable,
  kInvalidTable,
  kInvalidArgument,
};

const char* wiener_table_status_string(WienerTableStatus status);

// Caller-owned normative tables.  Sizes:
//   lut_to_class  : Pc_Wiener_Lut_To_Class[4096]
//   sub_classify2 : Pc_Wiener_Sub_Classify2[4][8][256] (row-major)
//   sub_classify  : Pc_Wiener_Sub_Classify[4][256]      (PC-Wiener only)
//   pc_filters    : Pc_Wiener_Filters[4][64][13]         (PC-Wiener only)
struct PcWienerTables {
  const std::uint8_t* lut_to_class = nullptr;
  const std::uint8_t* sub_classify2 = nullptr;
  const std::uint8_t* sub_classify = nullptr;
  const std::int16_t* pc_filters = nullptr;

  bool has_lut_to_class() const { return lut_to_class != nullptr; }
  bool has_sub_classify2() const { return sub_classify2 != nullptr; }
  bool has_sub_classify() const { return sub_classify != nullptr; }
  bool has_pc_filters() const { return pc_filters != nullptr; }

  std::uint8_t lut_class(int lut_input) const {
    return lut_to_class[lut_input & (kPcWienerLutEntries - 1)];
  }

  // Pc_Wiener_Sub_Classify2[filter_set][num_classes_idx][cls].
  std::uint8_t subclass(int filter_set, int num_classes_idx, int cls) const {
    const int index = ((filter_set * 8 + num_classes_idx) *
                           kNumPcWienerLutClasses +
                       (cls & (kNumPcWienerLutClasses - 1)));
    return sub_classify2[index];
  }

  // §7.20.4 filt = Pc_Wiener_Sub_Classify[index][cls] (PC-Wiener only).
  std::uint8_t pc_filter(int filter_set, int cls) const {
    return sub_classify[filter_set * kNumPcWienerLutClasses +
                        (cls & (kNumPcWienerLutClasses - 1))];
  }
};

struct WienerClassifierContext {
  const LrCoreFrame* core = nullptr;
  // LrTxSkip[y >> 2][x >> 2] on the luma MI grid.  The encoder knows this from
  // its transform decision; null means every 4x4 block was coded (tskip = 0).
  const std::uint8_t* lr_tx_skip = nullptr;
  int base_q_idx = 0;
  int bit_depth = 8;
  // Tile/frame bounds used by get_tx_skip().  For a single tile these are 0
  // and mi_rows.
  int mi_row_start = 0;
  int mi_row_end = 0;
  // MiColEnd for get_features()'s BlockEndX.  For a single tile this is
  // mi_cols.
  int mi_col_end = 0;
};

// §7.20.4 for one 4x4 luma block.  Requires Pc_Wiener_Lut_To_Class.
WienerTableStatus wiener_derive_filter_class(const LrCoreFrame& core,
                                             const LrBlock& block,
                                             const PcWienerTables& tables,
                                             const WienerClassifierContext& ctx,
                                             std::uint8_t* filter_class);

// Computes FilterClass on the whole luma MI grid.  Blocks that §7.20.1 skips
// (lossless, BRU-inactive, outside a tile) are assigned class 0 because no
// pixel-classified filter runs there.
WienerTableStatus wiener_derive_filter_class_grid(
    const LrCoreFrame& core, const PcWienerTables& tables,
    const WienerClassifierContext& ctx,
    std::vector<std::uint8_t>* filter_class_grid);

// Maps FilterClass -> coefficient class with Pc_Wiener_Sub_Classify2.
// `filter_set` is get_filter_set_index(base_q_idx); `num_classes_idx` is the
// f(3) value.  Requires the table when subclasses are actually needed.
WienerTableStatus wiener_build_subclass_grid(
    const std::uint8_t* filter_class_grid, int mi_cols, int mi_rows,
    int filter_set, int num_classes, int num_classes_idx,
    const PcWienerTables& tables, std::vector<std::uint8_t>* subclass_grid);

}  // namespace av2enc::restoration::classes

#endif  // AV2ENC_ENCODER_FILTERS_RESTORATION_CLASSES_WIENER_CLASS_H_

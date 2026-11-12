// MrlIndex ↔ edge filter / transform-type interaction for M1.
//
// The prediction core in src/encoder/mrl/ already implements the rules:
//
//   §7.13.2.7   intra edge filtering is applied only when MrlIndex == 0;
//   §5.20.7.29  compute_tx_type() adds Mrl_Index_To_Delta[MrlIndex] to pAngle
//               before wide_angle_mapping(), so the luma TX type depends on MRL.
//
// What this layer adds is the single place that *derives* those inputs, so the
// M1 syntax writer / RDO path cannot forget either rule:
//
//   * mrl_allows_edge_filter() / make_predict_params(): edge filtering is
//     forced off for MrlIndex > 0 regardless of enable_intra_edge_filter;
//   * mapped_intra_mode_for_mrl(): `txtype::map_intra_mode()` plus
//     Mrl_Index_To_Delta before the wide-angle map;
//   * make_tx_type_query() / tx_type_symbol_for_mrl(): the §5.20.8.2 query fed
//     to src/encoder/tx_type/ with the correct mapped mode.
//
// Nothing here predicts or writes bits.
#pragma once

#include <cstdint>

#include "codec/intra/mrl/core/mrl.h"
#include "codec/transform/tx_type/tx_type.h"

namespace av2enc::still::mrl_stream {

// Mrl_Index_To_Delta[4] = { 0, 1, -1, 0 } (§5.20.7.29 constants table).
int mrl_angle_delta(int mrl_index) noexcept;

// Mode_To_Angle[YMode] + AngleDeltaY * ANGLE_STEP + Mrl_Index_To_Delta.
// Returns -1 for a non-directional mode or an out-of-range delta/index.
int mrl_p_angle(int y_mode, int angle_delta, int mrl_index) noexcept;

// pAngle followed by §5.20.7.29 wide_angle_mapping() for coded_w x coded_h.
// Returns the mapped intra mode index 0..12, or -1 for invalid input.
// For mrl_index == 0 this is exactly txtype::map_intra_mode().
int mapped_intra_mode_for_mrl(int y_mode, int angle_delta, int mrl_index,
                              int coded_w, int coded_h) noexcept;

// §7.13.2.7: edge filtering only for the adjacent line.
bool mrl_allows_edge_filter(int mrl_index) noexcept;

// Builds the mrl::Params for prediction.  `sequence_enable_edge_filter` is the
// sequence-header enable_intra_edge_filter bit; the returned
// enable_edge_filter is false whenever mrl_index > 0.  Returns false for an
// invalid mode/delta/index or a secondary flag that cannot apply.
bool make_predict_params(int y_mode, int angle_delta, int mrl_index,
                         bool mrl_sec_index, bool sequence_enable_edge_filter,
                         bool above_neighbor_is_smooth,
                         bool left_neighbor_is_smooth, mrl::Params* params,
                         int block_w = 64, int block_h = 64) noexcept;

// §5.20.8.2 / §5.20.7.29 query for an intra luma TX block.  On success fills
// `query` with mapped_mode = mapped_intra_mode_for_mrl(...).  Returns false for
// an invalid mode/delta/index/size combination.
bool make_tx_type_query(int y_mode, int angle_delta, int mrl_index,
                        int coded_w, int coded_h, int reduced_tx_set,
                        bool dct_only, txtype::TxTypeQuery* query) noexcept;

// Convenience wrapper around txtype::tx_type_symbol() that takes the MRL-aware
// query inputs directly.
bool tx_type_symbol_for_mrl(int y_mode, int angle_delta, int mrl_index,
                            int coded_w, int coded_h, int reduced_tx_set,
                            bool dct_only, txtype::TxType type, int* symbol,
                            bool* is_long_side_dct) noexcept;

}  // namespace av2enc::still::mrl_stream

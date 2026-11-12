// AV2 inter transform-type sets, candidates and symbol mapping (§5.20.8.2/.3).
//
// This is the inter counterpart of `src/encoder/tx_type/` (which is intra-only:
// it says so in its header).  The pieces the inter path needs and intra does
// not:
//
//   * `get_tx_set()` with `is_inter = 1`: for reduced_tx_set = 0 the sets are
//     TX_SET_INTER_1 (4x4..8x16), TX_SET_INTER_2 (16x16), TX_SET_DCT_IDTX
//     (32x32), TX_SET_{WIDE,HIGH}_{32,64} for the long rectangular sizes and
//     TX_SET_DCTONLY for 32x64/64x32/64x64.
//   * The inter inversion tables `Tx_Type_Inter_Inv_Set1/2/3/4` and
//     `Tx_Type_Inv_Long`, i.e. two symbols (`inter_tx_type` and
//     `inter_tx_type_offset`) for INTER_1/INTER_2 and one symbol elsewhere.
//   * The `inter_tx_type` CDF context, which depends on the EOB diagonals, and
//     the `inter_tx_type_offset` context, which uses the *unclamped* tx
//     dimensions (the two differ in the spec; do not merge them).
//   * The `enable_inter_ddt` search-space guard: §5.4.8 allows DDT at the
//     sequence level and §7.15.4.1 only substitutes ADST/FDST with DDTX/FDDT
//     when useDdt.  The tables are independent of the flag.  This component
//     reconstructs only the plain ADST/FDST 1-D kernels, so with
//     `enable_inter_ddt = 1` it withholds the types whose kernels would change
//     at the decoder.  Identity types (IDTX, V/H_DCT) are not DDT types.
//
// Nothing here touches the transforms themselves: candidates carry
// `txv::TxParams` built by `txv::make_tx_params()`.
#pragma once

#include <cstdint>

#include "codec/transform/variants/tx_variants.h"

namespace av2enc::video::residual {

using txv::TxSize;
using txv::TxType;

// §6.19.8.2 transform set values, split by is_inter (the two enums overlap:
// intra 5/6 are INTER_1/INTER_2 for inter blocks).  Only the inter-relevant
// names are modelled.
enum class InterTxSet : std::int8_t {
  kDctOnly = 0,      // TX_SET_DCTONLY
  kWide64 = 1,       // TX_SET_WIDE_64
  kHigh64 = 2,       // TX_SET_HIGH_64
  kWide32 = 3,       // TX_SET_WIDE_32
  kHigh32 = 4,       // TX_SET_HIGH_32
  kInter1 = 5,       // TX_SET_INTER_1
  kInter2 = 6,       // TX_SET_INTER_2
  kDctIdtx = 7,      // TX_SET_DCT_IDTX
  kDctIdtxIddct = 8, // TX_SET_DCT_IDTX_IDDCT
};

const char* inter_tx_set_name(InterTxSet set);

// §5.20.7.30 Tx_Size_Sqr / Tx_Size_Sqr_Up as TX_SIZE enum values (4x4 = 0).
int tx_size_sqr(TxSize tx_size);
int tx_size_sqr_up(TxSize tx_size);

// §5.20.8.3 get_tx_set(txSz, plane = 0) for an inter block.
// `reduced_tx_set` is the frame-header field (M2 writes 0).
InterTxSet inter_tx_set(TxSize tx_size, int reduced_tx_set);

// One codable inter TX type and the symbols needed to signal it.
struct InterTxCandidate {
  TxType type = TxType::kDctDct;
  // Value of the `is_long_side_dct` symbol for WIDE_32/HIGH_32 (coded) and the
  // implied value 1 for WIDE_64/HIGH_64 (not coded).
  bool is_long_side_dct = true;
  bool codes_long_flag = false;  // WIDE_32 / HIGH_32 only
  // `inter_tx_type` symbol. -1 when the set codes no type at all.
  std::int8_t symbol = -1;
  // `inter_tx_type_offset` symbol (INTER_1 / INTER_2 only), else -1.
  std::int8_t offset_symbol = -1;
  txv::TxParams params;
};

inline constexpr int kMaxInterTxCandidates = 16;

struct InterTxSetInfo {
  InterTxSet set = InterTxSet::kDctOnly;
  int count = 0;
  InterTxCandidate candidates[kMaxInterTxCandidates];
};

// Enumerates the codable (type, long-side) pairs for `tx_size`.
//
// Candidate order is the §5.20.8.2 symbol order, which is what the writer
// needs; the decision path must not depend on it.  The enumeration follows
// `Tx_Type_In_Set_Inter` (§5.20.7.29) and additionally applies two
// encoder-side capability guards (not spec restrictions): the 1-D scan
// classes (V/H_DCT, V/H_ADST, V/H_FLIPADST) are out of scope until the
// class-specific scans are implemented, and `enable_inter_ddt = true`
// withholds ADST/FDST types because the decoder would use DDTX/FDDT kernels
// this component does not model.  Returns false for an unsupported TX size or
// a set that is not an inter set.
bool inter_tx_candidates(TxSize tx_size, int reduced_tx_set,
                         bool enable_inter_ddt, InterTxSetInfo* out);

// Inverse of the candidate table: the symbols needed to signal `type`.
// Returns false only when `type` is outside the decoder's set for this size
// (§5.20.7.29 aliases it to DCT_DCT).  `enable_inter_ddt` does not change the
// symbol mapping.
bool inter_tx_symbol(TxSize tx_size, int reduced_tx_set, bool enable_inter_ddt,
                     TxType type, InterTxCandidate* out);

// True when §7.15.4.1 would swap an ADST/FDST 1-D component for DDTX/FDDT
// (enable_inter_ddt && is_inter && sz != 4).  Identity types are not DDT.
bool inter_tx_type_is_ddt(TxType type);

// §5.20.7.29 Tx_Type_In_Set_Inter, i.e. whether the decoder's
// compute_tx_type() would keep this type for the set.  Types outside the set
// are aliased to DCT_DCT by the decoder, so an encoder must never signal one.
bool inter_tx_type_in_set(InterTxSet set, TxType type);

// CDF context for `inter_tx_type` (§8.3.2): 0/1/2 from the EOB diagonal.
// eob must be >= 1 (the syntax is only reached for a non-zero block).
int inter_tx_type_context(int eob, TxSize tx_size);

// CDF context for `inter_tx_type_offset` (§8.3.2).  Note the spec drops the
// `Min(., 32)` clamps here, unlike inter_tx_type_context().
int inter_tx_type_offset_context(int eob, TxSize tx_size);

// §5.20.7.29 compute_tx_type() for a chroma plane: `enable_chroma_dctonly`
// forces DCT_DCT, otherwise the luma TX type is reused when it is in the set.
TxType inter_chroma_tx_type(TxType luma_tx_type, TxSize chroma_tx_size,
                            int reduced_tx_set, bool enable_chroma_dctonly);

}  // namespace av2enc::video::residual

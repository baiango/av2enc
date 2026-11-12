// DDT constraint validation for M2.
//
// DDT has no syntax of its own beyond `enable_inter_ddt`; whether it changes a
// block follows from the signalled `inter_tx_type`, the TX size, the plane and
// a handful of sequence/frame tools.  This module turns that derivation into a
// single checked call so the encoder cannot emit a stream whose reconstruction
// disagrees with the decoder.
//
// Legal combinations (all citations are AV2 v1.0.0):
//   * §7.15.4.1: DDT applies only when is_inter = 1, use_intrabc = 0, and the
//     signalled 1D type is ADST/FDST at adjusted length 8 or 16.  Length 32 is
//     allowed by the predicate but aliases Dct_Kernel32; length 4 is excluded.
//   * §5.20.8.3 get_tx_set() + §5.20.7.29 Tx_Type_In_Set_Inter determine which
//     2D types can be signalled for a size:
//       - TX_SET_INTER_1 (4x4..16x16 non-16x16-square, all rect): all 16 types
//       - TX_SET_INTER_2 (16x16): types 0..11
//       - TX_SET_WIDE_32 / HIGH_32: {0,1,4,9,10,11,12,14} / {0,2,5,9,10,11,13,15}
//       - TX_SET_WIDE_64 / HIGH_64: {0,1,4,15} / {0,2,5,14}
//       - TX_SET_DCT_IDTX / DCT_IDTX_IDDCT / DCTONLY: DCT/IDT only, no DDT
//     reduced_tx_set != 0 selects DCT_IDTX/DCT_IDTX_IDDCT for inter, so DDT
//     requires reduced_tx_set == 0.
//   * Chroma: compute_tx_type() returns DCT_DCT when enable_chroma_dctonly = 1,
//     and otherwise reuses the luma TxType only if it is in the chroma size's
//     set.  DDT is allowed on chroma only in the latter case.
//   * Lossless/FSC: compute_tx_type() returns IDTX/DCT_DCT, never ADST/FDST, so
//     DDT never applies.  A caller that pairs lossless with ADST is rejected.
//
// Tool interactions (no extra constraint, documented because the task asks):
//   * TCQ (§7.14.4) is eligible when get_tx_class(PlaneTxType) == 2D.  DDT does
//     not change PlaneTxType, so TCQ+DDT coexist exactly as TCQ+ADST/FDST did.
//     DDTX/FDDT at length 8/16 alias ADST/FDST in the TX set, not in the class.
//   * IST (§7.15.3): inter IST is signalled only for TxType == DCT_DCT, so a
//     DDT block can never carry sec_tx_type != 0.  validation rejects the pair.
//   * QM (§7.14.4): useQm requires PlaneTxType < IDTX and keys the matrix by
//     txSz, not by 1D type.  DDT therefore leaves QM indexing untouched; a DDT
//     block can use QM exactly like the underlying ADST/FDST type.
#pragma once

#include <cstdint>
#include <string>

#include "codec/transform/ddt/ddt.h"

namespace av2enc::video::ddt {

enum class Status : std::int8_t {
  kOk = 0,             // DDT is active and the block is legal
  kDisabled,           // enable_inter_ddt == 0
  kNotInter,           // intra block
  kIntrabc,            // use_intrabc == 1 (is_inter is also 1)
  kLossless,           // lossless/FSC: compute_tx_type cannot return ADST/FDST
  kUnsupportedSize,    // not one of the 25 AV2 TX sizes
  kTypeNotInSet,       // signalled luma type is not in Ti_Type_In_Set_Inter
  kChromaDctOnly,      // enable_chroma_dctonly forced chroma to DCT_DCT
  kNoDdtComponent,     // legal type, but no ADST/FDST at length 8/16
  kIstConflict,        // sec_tx_type != 0 together with a DDT-active type
};

const char* status_name(Status status);

// One block's signalled state.  `tx_size` is the plane's TX size (chroma sizes
// are the subsampled sizes); `tx_type` is the luma TxType for every plane,
// because chroma derives its type from luma.
struct BlockSignal {
  int tx_size = 0;   // txv::TxSize
  int tx_type = 0;   // txv::TxType of the luma block
  int plane = 0;     // 0 = Y, 1 = U, 2 = V
  bool is_inter = false;
  bool use_intrabc = false;
  bool lossless = false;
  bool fsc_mode = false;
  bool enable_inter_ddt = false;
  bool enable_chroma_dctonly = false;
  int reduced_tx_set = 0;  // frame-header field (§5.20.7.27), M2 writes 0
  bool sec_tx_type = false;  // inter IST active on this block
  bool allow_tcq = false;    // frame-header allow_tcq (§7.14.4)
  bool using_qmatrix = false;
  int bit_depth = 8;
};

struct Validation {
  Status status = Status::kDisabled;
  bool ddt_active = false;
  Plan plan;
  TxType plane_tx_type = TxType::kDctDct;  // compute_tx_type() result
  bool tcq_eligible = false;   // get_tx_class(plane_tx_type) == 2D
  bool qm_eligible = false;    // using_qmatrix && plane_tx_type < IDTX
  bool ist_eligible = false;   // !sec_tx_type (inter IST requires DCT_DCT)
  std::string reason;
};

Validation validate(const BlockSignal& signal);

// Exposed for tests: the plane's effective TxType after compute_tx_type().
TxType plane_tx_type(const BlockSignal& signal);

// §5.20.8.2 get_tx_class() == TX_CLASS_2D.
bool tx_class_is_2d(TxType type);

}  // namespace av2enc::video::ddt

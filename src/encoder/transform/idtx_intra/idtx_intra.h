// AV2 intra identity-transform coverage: planner + legality.
//
// Normative source: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7):
//
//   §5.4.8   sequence_transform_quant_entropy_config(): enable_fsc infers
//            enable_idtx_intra = 1; otherwise enable_idtx_intra is coded
//   §5.20.7.29 compute_tx_type(): plane 0 returns TxTypes[]; chroma with
//            enable_chroma_dctonly == 0 derives the type from UVMode through
//            Mode_To_Txfm, then `is_tx_type_in_set()`, else DCT_DCT
//   §5.20.8.2 transform_type(): symbol sets, eob == 1 override,
//            reduced_tx_set == 2 intra override, IST applicability
//   §5.20.8.3 get_tx_set(): plane-dependent reduced set
//   §7.14.4  useQm = using_qmatrix && PlaneTxType < IDTX && segLvl < NUM_CUSTOM_QMS
//   §7.14.4  useFsc and the TCQ extra shift
//
// What this component owns
// ------------------------
// The component in `src/encoder/tx_type/` enumerates the full M1 intra sets
// (reduced_tx_set 0, no identity filter).  This planner adds the absent pieces:
//
//   * the §5.4.8 `enable_idtx_intra` bit, including the `enable_fsc` inference
//     that removes the bit from the stream;
//   * the identity-type legality rule (`enable_idtx_intra == 0` forbids IDTX,
//     V_DCT, H_DCT, V_ADST, H_ADST, V_FLIPADST, H_FLIPADST on every plane);
//   * `reduced_tx_set` 1..3 and the per-set legal-type filtering, including
//     the `reduced_tx_set == 2` intra DCT_DCT override;
//   * chroma non-DCT type derivation when `enable_chroma_dctonly == 0`;
//   * the QM (`tx_type < IDTX`) and IST (DCT_DCT / ADST_ADST only) predicates.
//
// It decides nothing about coefficients: it answers "which TX_TYPE may this
// (plane, size, mode, eob, sequence flags) combination signal, with which
// §5.20.8.2 symbols".  The type-syntax writer/reader is in `type_syntax.h`.
#pragma once

#include <cstdint>
#include <string>

#include "encoder/tx_variants/tx_variants.h"

namespace av2enc::idtx {

using txv::TxType;

// §5.4.8 sequence flags that change transform-type legality.  Plain values;
// the caller fills them from its header configuration.
struct SeqFlags {
  bool enable_fsc = false;
  // Raw enable_idtx_intra bit.  Ignored (and inferred to 1) when enable_fsc.
  bool enable_idtx_intra = false;
  bool enable_intra_ist = false;
  bool enable_inter_ist = false;
  bool enable_chroma_dctonly = true;
  bool using_qmatrix = false;
  bool allow_tcq = false;
  bool enable_cctx = false;
  bool enable_parity_hiding = false;
  bool monochrome = false;
  bool single_picture_header = true;
  // Frame-level (§5.18.2) reduced_tx_set for plane 0.
  int reduced_tx_set = 0;
};

// Effective values after the §5.4.8 inference chain.
struct DerivedSeqFlags {
  bool enable_idtx_intra = false;  // effective value (fsc forces 1)
  bool idtx_bit_coded = true;      // false when enable_fsc == 1
  bool idtx_allowed = false;       // identity 1D transforms may be signalled
};
DerivedSeqFlags derive_sequence_flags(const SeqFlags& flags);

// §6.19.8.2 intra transform sets, ordered exactly like Tx_Type_In_Set_Intra.
enum class Set : std::int8_t {
  kDctOnly = 0,
  kWide64 = 1,
  kHigh64 = 2,
  kWide32 = 3,
  kHigh32 = 4,
  kIntra1 = 5,
  kIntra2 = 6,
  // Not an intra set: §5.20.8.3 returns TX_SET_DCT_IDTX for
  // reduced_tx_set == 2, and transform_type() forces DCT_DCT for intra before
  // any symbol is read.  Kept so set_name()/get_tx_set() stay faithful.
  kDctIdtx = 9,
};
const char* set_name(Set set);

// Identity types are IDTX and every one-axis-type pair (§7.15.2.3 kernels).
inline bool is_identity_type(TxType type) {
  return static_cast<int>(type) >= static_cast<int>(TxType::kIdtx);
}

// §5.20.7.29 get_tx_class(): 0 = TX_CLASS_2D, 1 = TX_CLASS_HORIZ,
// 2 = TX_CLASS_VERT.  IDTX and the two-axis types are TX_CLASS_2D.
int tx_class(TxType type);

// Tx_Type_In_Set_Intra[set][type] for the 7 intra sets.
bool intra_set_allows(Set set, TxType type);

// One transform block query.
struct Query {
  int coded_w = 0;  // 4..64, one of the 25 TX sizes
  int coded_h = 0;
  int plane = 0;  // 0 luma, 1 U, 2 V
  // §5.20.7.29 wide_angle_mapping output for plane 0 (0..12).
  int mapped_mode = 0;
  // Signalled intra mode before mapping (0..12); the IST PAETH exclusion uses
  // the signalled value, not the mapped one.
  int y_mode = 0;
  // Signalled UV mode 0..13 and AngleDeltaUV (-3..3), used only for plane > 0.
  int uv_mode = 0;
  int uv_angle_delta = 0;
  // Residual EOB after the eob syntax.  0 means txb_skip (transform_type is
  // not parsed), 1 forces DCT_DCT, >= 2 codes the set symbol.
  int eob = 0;
  SeqFlags flags;
};

// One legal type and the symbols needed to signal it.
struct Candidate {
  TxType type = TxType::kDctDct;
  // Symbol index for the intra_tx_type CDF, or -1 when the set codes none.
  int symbol = -1;
  bool is_long_side_dct = false;  // signalled flag (32) or implied 1 (64)
  bool codes_long_flag = false;   // true only for WIDE_32 / HIGH_32
};

struct TypeSet {
  Set set = Set::kDctOnly;
  int count = 0;
  Candidate candidates[8];
};

// §5.20.8.3 get_tx_set(txSz, plane) for an intra block.  `query.plane`
// selects `reduced_tx_set` (0) or `enable_chroma_dctonly` (1/2).  Returns
// false for a size that is not one of the 25 AV2 transform sizes or an
// out-of-range reduced set.
bool get_tx_set(const Query& query, Set* set);

// Legal candidates in §5.20.8.2 symbol order.  For plane > 0 this is the
// *syntax* capability of the chroma set; the effective chroma type is the
// derived one from derive_chroma_type().  `eob <= 1` always yields exactly
// DCT_DCT with symbol -1.
bool enumerate_types(const Query& query, TypeSet* out,
                     std::string* error = nullptr);

// One concrete decision: the effective PlaneTxType plus its coding.
struct TypePlan {
  TypeSet set;
  TxType type = TxType::kDctDct;
  bool type_syntax = false;  // a §5.20.8.2 symbol is coded
  int symbol = -1;
  bool is_long_side_dct = false;
  bool codes_long_flag = false;
};

// Validates `requested` against the legal set and fills the plan.  A request
// for a type the set cannot signal fails (`error` says why): identity types
// with enable_idtx_intra == 0, wrong reduced set, a 1D type at an INTRA_2 /
// DCT-only size, etc.
bool plan_type(const Query& query, TxType requested, TypePlan* plan,
               std::string* error = nullptr);

// Deterministic default: DCT_DCT (legal in every intra set).  For chroma with
// enable_chroma_dctonly == 0 pass the derived type instead.
TxType default_type(const Query& query);

// §5.20.7.29 chroma derivation.  Returns the effective chroma PlaneTxType:
// DCT_DCT when enable_chroma_dctonly == 1 or the DCT-only size rule applies,
// else Mode_To_Txfm[mapped UVMode] filtered through is_tx_type_in_set().
// `uv_mode` 13 (UV_CFL_PRED) maps to DCT_DCT.  Returns false only for an
// invalid size / mode / angle.
bool derive_chroma_type(int uv_mode, int uv_angle_delta, int coded_w,
                        int coded_h, const SeqFlags& flags, TxType* type,
                        Set* set = nullptr);

// §5.20.7.29 Mode_To_Txfm[UVmode] (14 entries).
TxType mode_to_txfm(int uv_mode);

// Interaction with the tools that key off the transform type.
struct ToolUse {
  int tx_class = 0;
  bool allow_fsc_intra = false;  // enable_idtx_intra && w,h <= FSC_MAX
  bool use_fsc = false;          // §7.14.4
  bool use_qm = false;           // §7.14.4, requires PlaneTxType < IDTX
  bool use_ist = false;          // §5.20.8.2
  int ist_eob_limit = 0;
  bool tcq_extra_shift = false;  // §7.14.4, TX_CLASS_2D includes IDTX
  bool parity_hiding = false;    // §7.14/§5.20.8.2, excludes IDTX
};

// `seg_level` is the §7.14.4 segLvl, `qm_data_present` the QmDataPresent bit
// (accepted for symmetry; `useUserQm` additionally needs tw/th <= 8 and is not
// modelled here).  This planner is intra-only, so the `|| is_inter` disjunct in
// useFsc cannot fire; an inter caller must add it.
ToolUse derive_tool_use(const Query& query, TxType plane_type, int seg_level,
                        bool qm_data_present, bool lossless,
                        bool fsc_mode);

// Constants from §3 that the predicates above use.
inline constexpr int kFscMax = 32;
inline constexpr int kIst4x4Height = 8;
inline constexpr int kIst8x8Height = 32;
inline constexpr int kIst8x8HeightRed = 20;
inline constexpr int kNumCustomQms = 15;

}  // namespace av2enc::idtx

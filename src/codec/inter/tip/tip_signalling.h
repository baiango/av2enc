// AV2 v1.0.0 TIP frame-level signalling — writer + reader.
//
// Scope (AV2 Bitstream & Decoding Process Specification v1.0.0, 2026-05-28,
// commit 38f28e7):
//
//   §5.4.6   sequence_inter_config() temporal region, starting at
//            enable_ref_frame_mvs and ending after enable_tip_refinemv.
//   §5.18.2  frame_header_info() TIP fields: use_ref_frame_mvs,
//            tmvp_sample_step_minus_1, TipFrameMode, frame_opfl_refine_type()
//            (§5.18.3.2), allow_tip_hole_fill, tip_global_wtd_index,
//            TipGlobalMv, tip_sharp/tip_regular.
//   §5.2.1/.3 OBU payload framing for OBU_LEADING_TIP / OBU_REGULAR_TIP
//            (non-extensible OBU: trailing_one_bit, no obu_extension_flag bit).
//   §6.2.2   Table 6.1 OBU types 13 and 14.
//   §6.20    Table 6.20 TipFrameMode values.
//
// The sequence slice written here is *spliceable*: every field between
// enable_ref_frame_mvs and enable_tip_refinemv is present in spec order, so the
// caller can insert it at the position documented in README.md ("M2
// integration sites") without a second set of bit positions.  The temporal
// component (`inter/temporal/temporal_syntax.*`) keeps its own slice for its
// standalone tests; the semantics are identical, this file owns the OBU-level
// mapping and the EnableTipOutput derivation.
//
// This component never writes M2-owned headers.  It produces the bits and the
// OBU bytes; the gate in gate/ assembles a complete stream from them.

#ifndef AV2ENC_ENCODER_INTER_TIP_STREAM_TIP_SIGNALLING_H_
#define AV2ENC_ENCODER_INTER_TIP_STREAM_TIP_SIGNALLING_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/bitstream/bitwriter.h"
#include "core/bitstream/basic_headers.h"
#include "codec/inter/temporal/temporal_syntax.h"
#include "codec/inter/temporal/temporal_types.h"

namespace av2enc::inter::tip {

using temporal::BitReader;
using temporal::InterpFilter;
using temporal::Mv;
using temporal::RefineType;
using temporal::TemporalStatus;
using temporal::TipFrameMode;

// ---------------------------------------------------------------------------
// §6.2.2 Table 6.1 — the two TIP OBU types.
// ---------------------------------------------------------------------------
enum class TipObuType : std::uint8_t {
  kLeadingTip = 13,  // OBU_LEADING_TIP
  kRegularTip = 14,  // OBU_REGULAR_TIP
};

inline constexpr std::uint8_t kTipObuLeading = 13;
inline constexpr std::uint8_t kTipObuRegular = 14;

inline constexpr bool is_tip_obu_type(std::uint8_t obu_type) {
  return obu_type == kTipObuLeading || obu_type == kTipObuRegular;
}
inline constexpr bool is_tip_obu_type(TipObuType type) {
  return is_tip_obu_type(static_cast<std::uint8_t>(type));
}
// §5.1.2 is_tip_frame(): obu_type == OBU_LEADING_TIP || OBU_REGULAR_TIP.
inline constexpr bool is_tip_frame_obu(std::uint8_t obu_type) {
  return is_tip_obu_type(obu_type);
}

// The OBU writer in bitstream/basic_headers.h only knows the M0 set.  A TIP OBU
// uses the same §5.2.2 header layout; the type is carried through the enum.
inline constexpr av2enc::basic::ObuType tip_obu_type(TipObuType type) {
  return static_cast<av2enc::basic::ObuType>(static_cast<std::uint8_t>(type));
}
inline constexpr TipObuType tip_obu_kind(std::uint8_t obu_type) {
  return obu_type == kTipObuLeading ? TipObuType::kLeadingTip
                                    : TipObuType::kRegularTip;
}

// ---------------------------------------------------------------------------
// §5.4.6 sequence_inter_config() temporal region.
// ---------------------------------------------------------------------------
struct TipSequenceSyntax {
  // Field order below is the bitstream order, not a logical grouping.
  bool enable_ref_frame_mvs = false;
  bool reduced_ref_frame_mvs_mode = false;
  int order_hint_bits_minus_1 = 3;  // OrderHintBits = value + 1
  bool enable_refmvbank = false;
  bool disable_drl_reorder = true;
  bool constrain_drl_reorder = false;
  bool explicit_ref_frame_map = true;
  bool explicit_num_ref_frames = true;
  int num_ref_frames_minus_1 = 7;
  int long_term_frame_id_bits = 0;
  int seq_max_drl_bits_minus_1 = 4;      // ns(MAX_REF_MV_STACK_SIZE - 1 = 5)
  bool allow_frame_max_drl_bits = false;
  int seq_max_bvp_drl_bits_minus_1 = 0;  // ns(MAX_REF_BV_STACK_SIZE - 1 = 3)
  bool allow_frame_max_bvp_drl_bits = false;
  int num_same_ref_compound = 0;
  bool enable_tip = false;
  bool disable_tip_output = false;
  bool enable_tip_hole_fill = false;
  bool enable_mv_traj = false;
  bool enable_bawp = false;
  bool enable_cwp = false;
  bool enable_imp_msk_bld = false;
  bool enable_df_sub_pu = false;
  bool enable_tip_explicit_qp = false;
  RefineType enable_opfl_refine = RefineType::kNone;
  bool enable_refinemv = false;
  bool enable_tip_refinemv = false;

  // §5.4.6: EnableTipOutput = enable_tip ? !disable_tip_output : 0.
  constexpr bool enable_tip_output() const {
    return enable_tip && !disable_tip_output;
  }
};

// Writes the slice.  `writer` must be positioned at `enable_ref_frame_mvs`.
TemporalStatus write_tip_sequence_syntax(BitWriter& writer,
                                         const TipSequenceSyntax& syntax);
// Reads the slice and fills every field, including the derived conditions.
TemporalStatus read_tip_sequence_syntax(BitReader& reader,
                                        TipSequenceSyntax* syntax);

// ---------------------------------------------------------------------------
// §5.18.2 frame_header_info() TIP fields.
// ---------------------------------------------------------------------------
struct TipFrameSyntax {
  // Inputs (bitstream-position conditions).
  bool use_ref_frame_mvs = false;
  int num_total_refs = 0;
  bool sb_64x64 = true;
  bool enable_tip = false;
  bool enable_tip_refinemv = false;
  bool enable_tip_output = true;  // EnableTipOutput from the sequence header
  RefineType enable_opfl_refine = RefineType::kNone;
  // Signalled choice when enable_opfl_refine == REFINE_AUTO (non-TIP frames).
  RefineType requested_opfl_refine = RefineType::kNone;
  bool enable_refinemv = false;
  bool enable_tip_hole_fill = false;
  int num_past_refs = 0;
  int num_future_refs = 0;
  bool is_tip_frame = false;  // is_tip_frame(): OBU_LEADING_TIP/REGULAR_TIP
  // Requested values (writer).
  TipFrameMode requested_mode = TipFrameMode::kDisabled;
  int tip_global_wtd_index = 0;
  Mv tip_global_mv{};
  bool tip_sharp = true;
  bool tip_regular = false;

  // Resolved outputs (reader fills them; the writer derives the same values).
  int tmvp_sample_step_minus_1 = 0;
  TipFrameMode mode = TipFrameMode::kDisabled;
  RefineType opfl_refine_type = RefineType::kNone;
  bool allow_tip_hole_fill = false;
  InterpFilter interp_filter = InterpFilter::kEighttapSharp;

  // True when the §5.18.2 TIP block is present at all.
  constexpr bool has_tip_block() const {
    return enable_tip && use_ref_frame_mvs && num_total_refs >= 2;
  }
};

// Writes tmvp_sample_step_minus_1 (when present) followed by the §5.18.2 TIP
// block, or frame_opfl_refine_type() alone when the block is absent.
TemporalStatus write_tip_frame_syntax(BitWriter& writer,
                                      const TipFrameSyntax& syntax);
TemporalStatus read_tip_frame_syntax(BitReader& reader,
                                     TipFrameSyntax* syntax);

// §5.18.3.2 frame_opfl_refine_type() in isolation.
TemporalStatus write_tip_opfl_refine_type(BitWriter& writer,
                                          const TipFrameSyntax& syntax);
TemporalStatus read_tip_opfl_refine_type(BitReader& reader,
                                         const TipFrameSyntax& syntax,
                                         RefineType* out);

// Resolves TipFrameMode / opfl_refine_type / interp filter for a config
// without touching a writer.  Used by the writer, the reader and the tests.
TemporalStatus resolve_tip_frame_syntax(TipFrameSyntax* syntax);

// ---------------------------------------------------------------------------
// §7.10.1 setup-TIP conformance conditions.
// ---------------------------------------------------------------------------
// `setup_tip_motion_field` is only conforming when every one of these holds.
// `past_*`/`future_*` refer to ref_frame_idx[ClosestPast] /
// ref_frame_idx[ClosestFuture]; type eligibility is
// `RefFrameType != INTRA_ONLY_FRAME && != KEY_FRAME` and size eligibility is
// `RefFrameWidth/Height == FrameWidth/Height` (§7.9.1 p405).
struct TipSetupConditions {
  bool switch_frame = false;
  bool closest_past_present = false;
  bool closest_future_present = false;
  TipFrameMode mode = TipFrameMode::kDisabled;
  bool use_ref_frame_mvs = false;
  bool has_both_refs = false;
  int num_past_refs = 0;
  bool past_type_eligible = false;
  bool future_type_eligible = false;
  bool past_size_eligible = true;
  bool future_size_eligible = true;
};

// Fills the two "eligible" triples from a decoded reference-frame table.
TipSetupConditions make_tip_setup_conditions(
    bool switch_frame, const TipFrameSyntax& frame, int closest_past,
    int closest_future, bool past_type_eligible, bool future_type_eligible,
    bool past_size_eligible, bool future_size_eligible);

// Returns kOk when §7.10.1's requirements are all satisfied; on failure
// `reason` (when non-null) points to a static description.
TemporalStatus check_tip_setup_conditions(const TipSetupConditions& conditions,
                                          const char** reason = nullptr);

// ---------------------------------------------------------------------------
// OBU framing (§5.2.1/.3, Table 6.1).
// ---------------------------------------------------------------------------
// Appends §5.2.3 trailing_bits to the payload writer.  TIP OBUs are not
// extensible, so there is no payload obu_extension_flag bit.
void finish_tip_obu_payload(BitWriter* writer);

// Complete Annex B OBU (leb128 length + §5.2.2 header + payload) for a TIP
// frame whose frame-header bits were already written into `writer`.
std::vector<std::uint8_t> make_tip_frame_obu(TipObuType type,
                                             BitWriter* writer);

// Convenience: the §5.18.2 TIP slice alone, with §5.2.3 trailing bits.  A
// complete TIP OBU also needs the surrounding frame_header_info() fields
// (cur_mfh_id .. refresh_frame_flags, see README "frame header" ordering);
// the gate writes those and then calls finish_tip_obu_payload().
std::vector<std::uint8_t> make_tip_slice_bytes(const TipFrameSyntax& syntax,
                                               TemporalStatus* status);

}  // namespace av2enc::inter::tip

#endif  // AV2ENC_ENCODER_INTER_TIP_STREAM_TIP_SIGNALLING_H_

#ifndef AV2ENC_INTER_SEF_SEF_H_
#define AV2ENC_INTER_SEF_SEF_H_

// Show-existing-frame (SEF) syntax and bookkeeping for AV2 v1.0.0.
//
// Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7), §6.2.2 Table 6.1, §5.18.1/§5.18.2,
// §6.17.2, §7.21.1-§7.21.6 and §7.23. Section numbers below refer to that
// document.
//
// A show-existing-frame OBU re-outputs a frame that is still present in a
// reference frame slot. It carries *no tile data*: the whole OBU payload is
// `frame_header(1)`, whose SEF path returns immediately after
// `frame_to_show_map_idx` / `derive_sef_order_hint` / `sef_order_hint`
// (§5.18.2):
//
//   obu_header()                      §5.2.2
//   cur_mfh_id                  uvlc()
//   seq_header_id_in_frame_header uvlc()
//   ShowExistingFrame = is_sef()      == 1, implied by obu_type
//   frame_to_show_map_idx       f(CeilLog2(NumRefFrames))
//   derive_sef_order_hint       f(1)
//   if (!derive_sef_order_hint)
//     sef_order_hint            f(OrderHintBits)
//   film_grain_config()                §5.18.10.1 (immediate_output_frame = 1)
//   /* return; refresh_frame_flags = 0, immediate_output_frame = 1 */
//   trailing_bits()                   §5.2.3
//
// OBU types (§6.2.2 Table 6.1): OBU_LEADING_SEF = 11, OBU_REGULAR_SEF = 12.
// `is_sef()` (§5.2.1) is the predicate over those two values. The two types
// are bit-identical apart from `obu_type`; `IsRegular` (§5.18.2) only changes
// the open-loop-key bookkeeping (`OlkEncountered` / `OlkTUOrderHint`).
//
// Output semantics (§5.18.1 + §7.2 + §7.21):
//   * ShowExistingFrame == 1 skips loop filtering and §7.23 entirely: the SEF
//     has refresh_frame_flags = 0, so it neither stores nor invalidates any
//     reference slot.
//   * decode_frame_wrapup invokes output_frame_buffers() (§7.21.6) with
//     `derive_sef_order_hint ? frame_to_show_map_idx : -1`.
//   * For derive_sef_order_hint == 1 the referenced frame is output at its own
//     OrderHint, then §7.21.3 outputs successive implicit frames.
//   * For derive_sef_order_hint == 0, output_process(-1) takes the
//     `mixedOutput` branch (§7.21.1) and outputs the referenced FrameStore as
//     if it had the signalled `sef_order_hint`, then outputs successive frames
//     after that hint.
//
// Conformance rules for the reference (§6.17.2, derive_sef_order_hint == 1):
//   * the referenced slot must be valid,
//   * the referenced frame must not already have been shown,
//   * RefImplicitOutputFrame[slot] == 0,
//   * RefImmediateOutputFrame[slot] == 0.
// Those rules make derive=1 usable only for a *hidden* frame. A frame that was
// already output (the normal M2 case) must use derive=0. The pinned dav2d
// additionally requires the explicit hint to reconstruct to the referenced
// frame's OrderHint (src/obu.c: `if (cpoc != poc) goto error`), so
// make_show_header() always signals the reference's own hint.
//
// This directory owns syntax/validation only. It does not code tiles and does
// not modify M1/M2 core.

#include "core/bitstream/bitwriter.h"
#include "pipeline/grain/film_grain_syntax.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace av2enc::inter::sef {

// §6.2.2 Table 6.1.
enum class SefKind : std::uint8_t {
  kLeading = 11,  // OBU_LEADING_SEF
  kRegular = 12,  // OBU_REGULAR_SEF
};

inline constexpr std::uint8_t kObuLeadingSef = 11;
inline constexpr std::uint8_t kObuRegularSef = 12;

// §3: NUM_REF_FRAMES, and the AV2 display-order-hint range bound used by
// get_disp_order_hint() (§5.18.2): the result must be less than
// (1 << (DISPLAY_ORDER_HINT_BITS - 1)).
inline constexpr int kNumRefFramesMax = 16;
inline constexpr unsigned kDisplayOrderHintBits = 16;

enum class SefError : std::uint8_t {
  kOk = 0,
  kBadNumRefFrames,          // num_ref_frames outside 1..16
  kBadOrderHintBits,         // order_hint_bits outside 1..16
  kBadMapIdx,                // frame_to_show_map_idx >= num_ref_frames
  kInvalidReference,         // slot is invalid / has no frame store
  kReferenceAlreadyShown,    // derive=1: frame has already been output
  kReferenceNotHidden,       // derive=1: immediate/implicit output flag set
  kOrderHintNotSignalable,   // derive=0: no LSBs reconstruct the target hint
  kOrderHintMismatch,        // derive=0: hint != referenced frame's OrderHint
  kHintOutOfRange,           // get_disp_order_hint() >= 1<<15
  kUnsupportedMfh,           // cur_mfh_id != 0 (needs multi-frame-header state)
  kTruncated,                // reader ran past the payload
  kBadTrailingBits,          // missing trailing_one_bit / non-zero tail
  kBadObuHeader,             // not an SEF OBU / bad length prefix
  kBadObuLength,             // leb128 size disagrees with the byte span
};

const char* sef_error_name(SefError error);
const char* sef_kind_name(SefKind kind);

// ---------------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------------

struct SefHeader {
  SefKind kind = SefKind::kRegular;
  // cur_mfh_id uvlc(). M2 always codes 0 (the sequence header). A non-zero
  // value needs a multi-frame header and is rejected by validate_sef_reference.
  std::uint32_t cur_mfh_id = 0;
  // seq_header_id_in_frame_header uvlc(). M2 always codes 0.
  std::uint32_t seq_header_id = 0;
  // f(CeilLog2(NumRefFrames)). The slot must still hold the wanted frame.
  std::uint8_t frame_to_show_map_idx = 0;
  // 1: OrderHint = RefOrderHint[idx] (the frame's own hint). This is the only
  // conformant form for a hidden frame and the only form that needs no hint
  // bits.
  bool derive_sef_order_hint = true;
  // Only present/coded when derive_sef_order_hint == false. This is the
  // OrderHintBits-wide LSB value, not necessarily the absolute OrderHint.
  std::uint32_t sef_order_hint = 0;
  // §5.18.10.1 film_grain_config(). The SEF path sets
  // immediate_output_frame = 1 before the call, so a sequence with
  // film_grain_params_present = 1 codes apply_grain f(1) here and, when it is
  // set, fgm_id f(3) + grain_seed f(16). The model itself must already be
  // present in the decoder's slots (the caller emits one type-23 OBU before
  // this OBU when it plans a new model).
  filters::grain::FilmGrainConfig grain;
};

// Sequence-level values the SEF header depends on. M2 uses
// num_ref_frames = 2 and order_hint_bits = 4.
struct SefParams {
  std::uint8_t num_ref_frames = 2;
  std::uint8_t order_hint_bits = 4;
  // §5.4.2 sequence bit. When false film_grain_config() codes nothing.
  bool film_grain_params_present = false;
};

// Number of bits in frame_header_info()'s SEF path, excluding trailing_bits.
// M2 (2 refs, 4 hint bits, derive=1) => 4 bits; derive=0 => 8 bits.
std::size_t sef_frame_header_bits(const SefParams& params,
                                  const SefHeader& header);

// Writes the §5.18.2 SEF path of frame_header_info(): cur_mfh_id uvlc(0),
// seq_header_id_in_frame_header uvlc(0), then the SEF fields. Does not write
// trailing_bits or the OBU header. Returns the number of bits written.
std::size_t write_sef_frame_header(BitWriter& writer, const SefParams& params,
                                   const SefHeader& header);

// Complete Annex B.2 OBU: leb128(length) obu_header() payload.
// `make_sef_obu` appends §5.2.3 trailing_bits (one 1 bit then zeros).
std::vector<std::uint8_t> make_sef_obu(const SefParams& params,
                                       const SefHeader& header);

// TD + SEF OBU. This is one complete temporal unit / IVF frame: §7.3.3 makes
// an SEF its own coded output frame unit, so it must not share a temporal unit
// with a tile-group OBU.
std::vector<std::uint8_t> make_sef_temporal_unit(const SefParams& params,
                                                 const SefHeader& header);

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

// Parses a complete SEF OBU and verifies its leb128 framing and trailing bits.
// On success `bytes_consumed` is the OBU size (prefix + header + payload).
bool parse_sef_obu(std::span<const std::uint8_t> bytes, const SefParams& params,
                   SefHeader& header, std::size_t& bytes_consumed,
                   SefError* error);

// Parses the payload of a SEF OBU (frame_header syntax + trailing_bits).
// `bits_consumed` is the number of frame_header syntax bits, i.e. the bit
// position before trailing_bits. The trailing pattern is verified against the
// end of the payload.
bool parse_sef_frame_header(std::span<const std::uint8_t> payload,
                            const SefParams& params, SefHeader& header,
                            std::size_t& bits_consumed, SefError* error);

// ---------------------------------------------------------------------------
// Reference slots and output bookkeeping (§7.23 / §7.21)
// ---------------------------------------------------------------------------

// Decoder-visible state of one reference slot. `frame_store_id` is an encoder
// side identity used by tests and the gate to prove byte identity; the
// decoder only has FrameStore samples.
struct RefSlotState {
  bool valid = false;
  std::uint32_t order_hint = 0;
  std::uint32_t frame_store_id = 0;
  bool immediate_output_frame = false;
  bool implicit_output_frame = false;
  // Set by output_process (§7.21.1). For derive=1 an SEF may not reference a
  // slot whose frame has already been output.
  bool shown = false;
};

struct SlotMap {
  std::uint8_t num_ref_frames = 2;
  std::array<RefSlotState, kNumRefFramesMax> slots{};

  const RefSlotState* slot(std::uint8_t index) const;
  // Largest RefOrderHint over valid, showable slots; 0 when none qualify.
  // This is get_max_disp_order_hint(1) without the layer checks (one embedded
  // layer): hidden frames (immediate=0 and implicit=0) are skipped.
  std::uint32_t max_disp_order_hint() const;
};

// The state a coded frame writes in §7.23. `refresh_frame_flags` is the raw
// bitmask. For KEY_FRAME/SWITCH_FRAME only the first refreshed slot is marked
// valid (RefValid = first); the remaining refreshed slots are invalidated.
struct RefreshFrame {
  bool key_frame = false;
  std::uint32_t order_hint = 0;
  std::uint32_t frame_store_id = 0;
  bool immediate_output_frame = true;
  bool implicit_output_frame = false;
  std::uint32_t refresh_frame_flags = 0;
};

// Applies §7.23 for one coded frame. Does not output anything: call
// mark_shown() for the slots emitted by output_process().
void apply_refresh(SlotMap& map, const RefreshFrame& frame);

// Marks a slot's frame as already output (§7.21.1). A later derive=1 SEF may
// no longer reference it.
void mark_shown(SlotMap& map, std::uint8_t index);

// Validates the reference rules of §6.17.2. This is the spec gate:
//   * slot index in range and RefValid == 1,
//   * derive=1: frame not yet shown, RefImplicitOutputFrame == 0,
//     RefImmediateOutputFrame == 0,
//   * derive=0: get_disp_order_hint(sef_order_hint) is in range.
// It deliberately does not require the explicit hint to equal the referenced
// frame's OrderHint; the pinned decoder does (see validate_dav2d_compat).
bool validate_sef_reference(const SlotMap& map, const SefParams& params,
                            const SefHeader& header, SefError* error);

// §5.18.2 get_disp_order_hint() for the SEF path. `order_hint_lsbs` is the
// raw OrderHintBits-wide value; `max_disp` is get_max_disp_order_hint(1) before
// the current frame updates any slot.
std::uint32_t disp_order_hint(std::uint32_t order_hint_lsbs,
                              std::uint32_t max_disp_order_hint,
                              std::uint8_t order_hint_bits);

// Finds the OrderHintBits-wide value the decoder must read to reconstruct
// `target_order_hint`. Returns false when the target is outside the
// reconstructible window.
bool sef_order_hint_lsbs(std::uint32_t target_order_hint,
                         std::uint32_t max_disp_order_hint,
                         std::uint8_t order_hint_bits,
                         std::uint32_t& order_hint_lsbs);

// Builds a header that re-shows slot `index`:
//   * derive=1 when the slot is hidden and not yet shown (§6.17.2),
//   * otherwise derive=0 with sef_order_hint chosen so the decoder
//     reconstructs the referenced frame's OrderHint.
// Returns false with a SefError when the slot cannot be shown.
bool make_show_header(const SlotMap& map, const SefParams& params,
                      std::uint8_t index, SefKind kind, SefHeader& header,
                      SefError* error);

// ---------------------------------------------------------------------------
// Pinned-dav2d compatibility
// ---------------------------------------------------------------------------

// dav2d's SEF parse (src/obu.c) rejects derive=0 unless the explicit hint
// reconstructs to the referenced frame's frame_offset. It also drops an output
// whose frame_offset equals the last queued output (dpb_poc), which is the
// normal case when an SEF re-shows an already-output frame; `--outputinvisible
// 1` bypasses that dedup. These two helpers make those behaviours explicit so
// the gate can state exactly what it expects.
bool validate_dav2d_compat(const SlotMap& map, const SefParams& params,
                           const SefHeader& header, SefError* error);

// True when the pinned decoder will suppress the extra output unless run with
// --outputinvisible 1: the referenced frame has already been output and the
// SEF uses the explicit-hint path (derive=0).
bool dav2d_needs_output_invisible(const RefSlotState& slot);

// ---------------------------------------------------------------------------
// Small helpers shared by the gate / tests
// ---------------------------------------------------------------------------

// ceil(log2(value)), §4.8 CeilLog2. value == 1 -> 0.
unsigned ceil_log2(std::uint32_t value);

}  // namespace av2enc::inter::sef

#endif  // AV2ENC_INTER_SEF_SEF_H_

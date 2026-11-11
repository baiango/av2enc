// AV2 v1.0.0 multi-frame header (MFH): writer, reader, validation and
// frame-header inheritance semantics.
//
// Normative reference: spec/AV2_Spec_v1.0.0.pdf (commit 38f28e7).
//
//   §5.7    multi_frame_header_obu() syntax
//   §6.7    Multi frame header OBU semantics   (the task brief calls it §6.6;
//           this PDF revision numbers it §6.7)
//   §7.3.3  coded output frame unit: zero or more MFH OBUs before the frame
//   §7.3.4  coded non-output frame unit: same placement
//   §7.3.8.7 MFH availability (mfh_id_minus_1 before cur_mfh_id)
//   §5.18.2 frame_header_info(): cur_mfh_id, seq_header_id_in_frame_header
//   §5.18.4.1 frame_size(): mfh_frame_width/height_minus_1
//   §5.18.5.2 deblocking_filter_params(): mfh_deblocking_filter_update
//   §5.18.7.1 segmentation_params(): mfh_seg_info_present_flag
//   §6.2.2  Table 6.1: OBU_MULTI_FRAME_HEADER == 3
//   §6.2.1  MAX_MFH_NUM == 16, MAX_SEQ_NUM == 16, MAX_SEGMENTS == 16,
//           SEG_LVL_MAX == 3, MAXQ_BITS == 351
//
// An MFH is a sequence-scoped "template" that several frame headers can
// reference.  A frame selects it with cur_mfh_id; when it does, the sequence
// header is loaded through MfhSeqHeaderId[cur_mfh_id] (so the frame header
// omits seq_header_id_in_frame_header) and three groups of fields may be
// inherited instead of coded per frame:
//
//   * frame size   -- used by frame_size() when frame_size_override_flag == 0
//   * deblocking   -- apply_deblocking_filter[] when the update flag is set
//                     (the frame header then codes no apply bits; the
//                     per-plane df_delta_q_present/df_delta_q bits remain)
//   * segment info -- the 8/16-segment feature table reused by
//                     segmentation_params() when the sequence/MFH ext_seg
//                     flags match and the MFH allows (or forces) reuse
//
// This module is self-contained: it does not edit the M1/M2 core, main.cpp or
// CMakeLists.txt.  The M2-specific adapter used by the decoder gate lives in
// mfh_m2_adapter.h.
#pragma once

#include "bitstream/bitwriter.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace av2enc::frame::mfh {

// §6.2.1 / Table 6.1 constants.
inline constexpr std::uint32_t kMaxMfhNum = 16;   // MAX_MFH_NUM
inline constexpr std::uint32_t kMaxSeqNum = 16;   // MAX_SEQ_NUM
inline constexpr int kMaxSegments = 16;           // MAX_SEGMENTS
inline constexpr int kSegLvlMax = 3;              // SEG_LVL_MAX
inline constexpr int kSegLvlAltQ = 0;             // SEG_LVL_ALT_Q
inline constexpr int kSegLvlSkip = 1;             // SEG_LVL_SKIP
inline constexpr int kSegLvlGlobalMv = 2;         // SEG_LVL_GLOBALMV
inline constexpr int kSegAltQBits = 9;            // Segmentation_Feature_Bits[0]
inline constexpr int kSegAltQMax = 351;           // Segmentation_Feature_Max[0]
inline constexpr std::uint8_t kObuTypeMultiFrameHeader = 3;

// One entry of seg_info() (§5.4.9) for one segment and one feature level.
struct SegmentFeature {
  bool enabled = false;
  std::int32_t value = 0;  // clipped value, as stored in FeatureData[][]
};

// seg_info( numSegments ) (§5.4.9).  Only the first `segments()` rows are
// coded; the rest are required to stay at their default (disabled, 0) values.
struct SegInfo {
  bool ext_seg = false;  // mfh_ext_seg_flag: 16 segments instead of 8
  std::array<std::array<SegmentFeature, kSegLvlMax>, kMaxSegments> feature{};

  int segments() const { return ext_seg ? 16 : 8; }
};

// §5.7 multi_frame_header_obu() payload (everything after the OBU header).
struct MultiFrameHeader {
  std::uint32_t seq_header_id = 0;  // mfh_seq_header_id, < MAX_SEQ_NUM
  std::uint32_t mfh_id = 1;         // mfh_id_minus_1 + 1, in [1, MAX_MFH_NUM)
  std::uint8_t tlayer_id = 0;       // obu_tlayer_id (2 bits) -> MfhTLayerId
  // obu_mlayer_id is inferred 0 because §6.2.1 requires the OBU header's
  // obu_header_extension_flag to be 0 in conforming bitstreams.

  bool frame_size_present = false;  // mfh_frame_size_present_flag
  // 1..16; 0 means "derive the minimum width that fits frame_width - 1".
  std::uint8_t frame_width_bits = 0;   // mfh_frame_width_bits_minus_1 + 1
  std::uint8_t frame_height_bits = 0;  // mfh_frame_height_bits_minus_1 + 1
  std::uint32_t frame_width = 0;       // mfh_frame_width_minus_1 + 1, luma
  std::uint32_t frame_height = 0;      // mfh_frame_height_minus_1 + 1, luma

  bool deblocking_filter_update = false;  // mfh_deblocking_filter_update
  // mfh_apply_deblocking_filter[mfh_id][0..3] = Y vertical, Y horizontal,
  // U (both directions), V (both directions).
  std::array<bool, 4> apply_deblocking_filter{};

  bool seg_info_present = false;  // mfh_seg_info_present_flag
  bool allow_seg_info_change = false;  // mfh_allow_seg_info_change
  SegInfo seg_info;                    // mfh_ext_seg_flag lives in seg_info
};

// Field-by-field validation of a value that is about to be written.  The
// writer is strict: it rejects values that would be silently clipped or
// ignored by the syntax (for example frame_width != 0 without the present
// flag).
bool validate_multi_frame_header(const MultiFrameHeader& header,
                                 std::string* error);

// §5.7 syntax bits only: no OBU header, no §5.2.3 trailing_bits.  Returns the
// number of bits written; on validation failure returns 0 and writes nothing.
std::size_t write_multi_frame_header_syntax(BitWriter& writer,
                                            const MultiFrameHeader& header,
                                            std::string* error);

// Complete Annex B.2 OBU:
//   leb128(1 + payload_bytes) obu_header() payload
// where the payload is the §5.7 syntax followed by the §5.2.1 extensible-OBU
// trailer:
//   obu_extension_flag f(1)   // always 0; §6.2.1 forbids 1
//   trailing_bits()
// OBU_MULTI_FRAME_HEADER is in is_extensible_obu() (§5.2.1), so the flag is
// mandatory even though conforming streams always code 0.
// The OBU header is byte 0x0C for tlayer 0 (obu_header_extension_flag 0,
// type 3).
std::vector<std::uint8_t> make_multi_frame_header_obu(
    const MultiFrameHeader& header, std::string* error);

// One OBU in an Annex B.2 byte stream.
struct ObuView {
  std::uint8_t obu_type = 0;
  std::uint8_t tlayer_id = 0;
  std::uint8_t mlayer_id = 0;  // inferred 0 when extension_flag == 0
  std::uint8_t xlayer_id = 0;  // inferred 0 when extension_flag == 0
  bool extension_flag = false;
  std::size_t offset = 0;        // offset of the leb128 size prefix
  std::size_t payload_offset = 0;
  std::size_t payload_size = 0;
  std::size_t total_size = 0;    // leb128 + header (+ extension) + payload
};

// Parses one Annex B.2 OBU starting at `offset` (§5.2.1/§5.2.2).  `bytes` may
// contain a whole temporal unit; the caller advances by `out->total_size`.
bool parse_obu(std::span<const std::uint8_t> bytes, std::size_t offset,
               ObuView* out, std::string* error);

// Parses and validates a §5.7 MFH payload (excluding the OBU header).  The
// §5.2.1 extensible-OBU trailer is checked: obu_extension_flag must be 0 and
// trailing_bits() must be a one bit followed by zero padding.  `out->tlayer_id`
// is not set here: the OBU header carries it.
bool parse_multi_frame_header_payload(std::span<const std::uint8_t> payload,
                                      MultiFrameHeader* out, std::string* error);

// Parses one MFH OBU at `offset`; rejects a non-MFH OBU type.  On success
// copies obu_tlayer_id into out->tlayer_id and fills `obu` when non-null.
bool parse_multi_frame_header_obu(std::span<const std::uint8_t> bytes,
                                  std::size_t offset, MultiFrameHeader* out,
                                  ObuView* obu, std::string* error);

// Decoder-side store for MfhSeqHeaderId / MfhTLayerId / MfhMLayerId plus the
// per-slot MFH payload.  Entries persist until overwritten; the spec defines
// no CLK-time reset for MFHs, so reset() is only for a new decoding session.
class MfhStore {
 public:
  void reset();
  bool insert(const MultiFrameHeader& header, std::string* error);
  const MultiFrameHeader* find(std::uint32_t mfh_id) const;
  bool parse_and_insert(std::span<const std::uint8_t> bytes, std::size_t offset,
                        MultiFrameHeader* parsed, std::string* error);
  std::size_t size() const { return count_; }

 private:
  std::array<std::optional<MultiFrameHeader>, kMaxMfhNum> slots_{};
  std::size_t count_ = 0;
};

// Where a resolved frame-header field came from.  Used by tests and by an
// encoder that wants to report the real cost of an MFH.
enum class FieldSource : std::uint8_t {
  kAbsent = 0,   // not applicable / no field
  kSequence,     // sequence header value (max frame size, seq seg info)
  kMfh,          // inherited from the multi-frame header
  kExplicit,     // coded in the frame header (overrides the MFH)
  kForced,       // forced by another normative rule (e.g. CodedLossless)
};

const char* to_string(FieldSource source);

// Sequence-level inputs needed by the inheritance rules.
struct SequenceFrameFields {
  std::uint16_t max_frame_width = 0;
  std::uint16_t max_frame_height = 0;
  bool monochrome = false;              // NumPlanes == 1
  bool enable_ext_seg = false;          // §5.4.4
  bool seq_seg_info_present = false;    // §5.4.4
  bool seq_allow_seg_info_change = false;
  SegInfo seq_seg_info;
  // Sequence headers known to be available (§7.3.8.6).  Index ==
  // seq_header_id.  The frame header and MFH both require their referenced id
  // to be present.
  std::array<bool, kMaxSeqNum> sequence_headers{};
};

// Optional layer dependency maps (§6.4.2).  When `deps` is null a conservative
// single-embedded-layer default is applied: the frame must be in embedded layer
// 0 and its temporal layer must be at least MfhTLayerId (temporal layer t may
// depend on layers <= t).  Provide the real maps for anything else.
struct LayerDependencyMap {
  bool mlayer[8][8]{};      // MLayerDependencyMap[m][k]
  bool tlayer[8][4][4]{};   // TLayerDependencyMap[m][t][k]

  static LayerDependencyMap identity();
};

// Frame-header fields that participate in MFH inheritance.
struct FrameHeaderFields {
  std::uint32_t cur_mfh_id = 0;  // §5.18.2; 0 means "no MFH"
  std::uint32_t seq_header_id = 0;  // only used when cur_mfh_id == 0
  // obu_mlayer_id / obu_tlayer_id of the frame header, for the §7.3.8.7
  // layer dependency check against MfhMLayerId / MfhTLayerId.
  std::uint8_t obu_mlayer_id = 0;
  std::uint8_t obu_tlayer_id = 0;

  bool frame_size_override = false;  // §5.18.2
  std::uint32_t frame_width = 0;     // explicit, when override is true
  std::uint32_t frame_height = 0;

  bool coded_lossless = false;  // CodedLossless short-circuits deblocking

  // When the MFH supplies deblocking flags these explicit values are absent
  // from the frame header; callers may leave them at false.
  std::array<bool, 4> apply_deblocking_filter{};

  bool segmentation_enabled = false;  // first bit of segmentation_params()

  // segmentation_params() control flow when a reuse decision is coded
  // (allowChange is true).  `reuse_seg_info` is the coded bit; when no bit is
  // coded the resolver derives it from `haveSegParams`.
  bool reuse_seg_info = false;
  // New segment table when reuse_seg_info resolves to false.
  SegInfo seg_info;
};

// Effective frame-header values after applying §5.18.
struct ResolvedFrameFields {
  std::uint32_t seq_header_id = 0;  // MfhSeqHeaderId or the explicit id
  std::uint32_t frame_width = 0;    // computed FrameWidth
  std::uint32_t frame_height = 0;   // computed FrameHeight
  FieldSource frame_size_source = FieldSource::kAbsent;

  std::array<bool, 4> apply_deblocking_filter{};
  FieldSource deblocking_source = FieldSource::kAbsent;
  // True when the frame header codes the two/three deblocking apply bits.
  bool deblocking_bits_present = false;
  // Per-plane df_delta_q_present/df_delta_q syntax is still present for every
  // plane whose resolved apply flag is 1, even when the flags came from the
  // MFH (§5.18.5.2).
  bool deblocking_delta_q_bits_present[4] = {false, false, false, false};

  bool segmentation_enabled = false;
  int max_segments = 8;  // from sequence enable_ext_seg
  // True when segmentation_params() sends reuse_seg_info.
  bool reuse_seg_info_bit_present = false;
  // True when segmentation_params() copies MFH/sequence segment data instead
  // of parsing a new seg_info().
  bool reuse_seg_info = false;
  FieldSource seg_info_source = FieldSource::kAbsent;
  SegInfo seg_info;  // effective FeatureEnabled/FeatureData
};

// Applies the §5.18.2 / §5.18.4.1 / §5.18.5.2 / §5.18.7.1 inheritance and
// precedence rules and enforces the bitstream-conformance requirements that
// can be checked from the values in hand.
bool resolve_frame_fields(const MfhStore& mfh_store,
                          const SequenceFrameFields& sequence,
                          const FrameHeaderFields& frame,
                          const LayerDependencyMap* deps,
                          ResolvedFrameFields* out, std::string* error);

// Reads every MFH OBU in a byte range (temporal unit or whole stream) and
// inserts each one into `store`.  Non-MFH OBUs are skipped.  Useful for a
// decoder front end and tests.
bool scan_multi_frame_headers(std::span<const std::uint8_t> bytes,
                              MfhStore* store, std::size_t* mfh_count,
                              std::string* error);

}  // namespace av2enc::frame::mfh

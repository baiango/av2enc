#pragma once

#include "bitstream/bitwriter.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

// AV2 v1.0.0 OBU/header writer for the M0 still picture.
//
// Normative reference: spec/AV2_Spec_v1.0.0.pdf (commit 38f28e7). Section
// numbers below refer to that document. This file writes only headers; tile
// coefficient syntax (§5.20) belongs to the entropy stage.
//
// NOTE: docs/minimal-bitstream.md §2.2 lists first bytes 0x10/0x08/0x20 for
// TD/SEQ/CLK. Those are AV1's OBU header byte values (obu_type << 3). AV2
// §5.2.2 defines obu_header_extension_flag f(1), obu_type f(5),
// obu_tlayer_id f(2), so the AV2 bytes are 0x08/0x04/0x10. The pinned
// testdata/sample_480x270_130f.ivf starts with "01 08 19 04 ...", confirming
// the AV2 layout. TODO(docs): fix the hex column in minimal-bitstream.md.
namespace av2enc::m0 {

// OBU types used by M0 (§6.2.2, Table 6.1).
enum class ObuType : std::uint8_t {
  kSequenceHeader = 1,
  kTemporalDelimiter = 2,
  kClosedLoopKey = 4,
  kRegularTileGroup = 7,
};

// obu_header() fields (§5.2.2). With `extension_flag == false`, `mlayer_id`
// and `xlayer_id` are not written; the decoder infers 0, except for
// OBU_TEMPORAL_DELIMITER/OBU_MSDO which infer GLOBAL_XLAYER_ID.
struct ObuHeader {
  ObuType type = ObuType::kTemporalDelimiter;
  // obu_tlayer_id (§5.2.2); 2 bits. Some documents call this temporal_id.
  std::uint8_t tlayer_id = 0;
  // obu_header_extension_flag (§5.2.2); 1 bit.
  bool extension_flag = false;
  // Only present when extension_flag is true.
  std::uint8_t mlayer_id = 0;  // 3 bits
  std::uint8_t xlayer_id = 0;  // 5 bits
};

inline constexpr std::size_t kObuHeaderBytes = 1;
inline constexpr std::size_t kObuHeaderExtensionBytes = 1;

// Writes obu_header() (§5.2.2). Returns the number of bits written (8 or 16).
// AV2 v1.0.0 has no obu_has_size_field bit in the header; Annex B.2 carries
// the OBU length as a leb128() prefix outside open_bitstream_unit().
std::size_t write_obu_header(BitWriter& writer, const ObuHeader& header);

// Appends leb128(value) (§4.11.6). Annex B.2 length prefixes are byte-aligned
// and must be <= 2^32 - 1.
void append_leb128(std::vector<std::uint8_t>& out, std::uint32_t value);
std::size_t leb128_size(std::uint32_t value);

// Builds an Annex B.2 length-delimited OBU:
//
//   leb128(num_bytes_in_obu) obu_header() payload
//
// num_bytes_in_obu counts the header and any extension byte, but not the
// leb128 prefix itself (docs/minimal-bitstream.md §2.2, §5.2.1).
std::vector<std::uint8_t> make_obu(const ObuHeader& header,
                                   std::span<const std::uint8_t> payload);

// M0 still-picture configuration. Width/height are the maximum frame size,
// i.e. max_frame_width_minus_1 + 1 (§6.4.1).
struct StillPictureConfig {
  std::uint16_t max_frame_width = 64;
  std::uint16_t max_frame_height = 64;
  std::uint8_t seq_level_idx = 7;  // 7 = level 5.1 (Annex A.4); M0 pins this.
  std::uint8_t base_q_idx = 90;    // §5.18.6.1; must be >= 1 (0 is lossless).
  // §5.18.2 / §8.2.6. M0 defaults to frozen CDFs (disable_cdf_update = 1) so
  // the pinned regression bitstreams do not change.
  bool disable_cdf_update = true;
  // Opt-in normative CDF adaptation. Default off. When true the frame header
  // codes disable_cdf_update = 0 and the encoder adapts after every symbol.
  bool enable_cdf_adaptation = false;

  // Explicit adaptation enable overrides the legacy disable field; otherwise
  // the field is the raw frame-header bit. This keeps old callers that set
  // disable_cdf_update = false adaptive while defaulting new callers to frozen.
  bool cdf_update_disabled() const noexcept {
    return enable_cdf_adaptation ? false : disable_cdf_update;
  }
};

// Exact syntax lengths for the two M0 frame sizes. See write_sequence_header()
// and write_frame_header(). The frame-header difference comes from tile_info():
// a 256x256 frame has 4x4 superblocks, so one increment_tile_cols_log2 and one
// increment_tile_rows_log2 flag are present (both 0 for one tile). The extra
// two bits are easy to miss; docs/minimal-bitstream.md §9 omits them. TODO(docs).
inline constexpr std::size_t kSequenceHeaderSyntaxBits64x64 = 83;
inline constexpr std::size_t kSequenceHeaderSyntaxBits256x256 = 87;
// Payload bytes include §5.2.1's extensible-OBU obu_extension_flag and the
// §5.2.3 trailing one bit. For 256x256 the 87 syntax bits + 2 bits no longer
// fit in 11 bytes, so the payload grows to 12.
inline constexpr std::size_t kSequenceHeaderPayloadBytes64x64 = 11;
inline constexpr std::size_t kSequenceHeaderPayloadBytes256x256 = 12;
inline constexpr std::size_t kFrameHeaderSyntaxBits64x64 = 22;
inline constexpr std::size_t kFrameHeaderSyntaxBits256x256 = 24;
inline constexpr std::size_t kTileGroupHeaderBytes64x64 = 3;
inline constexpr std::size_t kTileGroupHeaderBytes256x256 = 4;

// Writes sequence_header_obu() (§5.4.1) syntax bits only: no §5.2.1 OBU
// extension flag and no trailing_bits. Returns 83 bits for 64x64, 87 for
// 256x256.
//
// single_picture_header_flag = 1 forces still_picture = 1, max_tlayer_id = 0,
// max_mlayer_id = 0, SeqMaxMlayerCnt = 1, OrderHintBits = 0, NumRefFrames = 2,
// enable_avg_cdf = 1 and avg_cdf_type = 1 (§5.4.1, §5.4.6, §5.4.8). It also
// removes seq_tier, decoder model info and the inter-DDT/TCQ-per-frame bits.
std::size_t write_sequence_header(BitWriter& writer, const StillPictureConfig& config);

// sequence_header_obu() plus §5.2.1's extensible-OBU obu_extension_flag, the
// §5.2.3 trailing_bits, the §5.2.2 OBU header and Annex B.2 leb128 framing.
//
// The 64x64 payload is 11 bytes: 83 syntax bits + extension flag 0 + trailing
// 1 + 3 zero bits. The 256x256 payload is 12 bytes (87 syntax bits + the same
// two bits + 7 zeros). TODO(docs): minimal-bitstream.md §2.2 omits the
// extension flag, and its §10 "8 bytes" note is stale; §3's 83-bit syntax
// count and 11-byte 64x64 payload are correct.
std::vector<std::uint8_t> make_sequence_header_obu(const StillPictureConfig& config);

// Writes frame_header(isFirst=1) -> frame_header_info() (§5.18.1-§5.18.2) for
// one OBU_CLOSED_LOOP_KEY intra frame, without the leading is_first_tile_group
// bit and without byte_alignment(). 22 bits for 64x64, 24 for 256x256.
std::size_t write_frame_header(BitWriter& writer, const StillPictureConfig& config);

// Writes the §5.19 tile_group_obu() header for one tile:
//
//   is_first_tile_group f(1)
//   frame_header(isFirst = 1)
//   [tile_start_and_end_present_flag absent because NumTiles == 1]
//   byte_alignment()                       (§5.2.4: zero bits only)
//
// 3 bytes (24 bits) for 64x64, 4 bytes for 256x256.
//
// TODO(docs): minimal-bitstream.md §5 correctly says one zero pad bit and three
// total bytes; §10 says "byte_alignment 2 zero bits", which would make four
// bytes. §5.19 + §5.2.4 are normative: is_first(1) + frame_header(22) = 23
// bits, so exactly one zero bit is appended.
std::size_t write_tile_group_header(BitWriter& writer, const StillPictureConfig& config);

// Builds the OBU_CLOSED_LOOP_KEY payload and frames it. `tile_payload` must be
// the complete §5.20 tile payload: range-coded symbols followed by the
// §8.2.4 exit_symbol() trailing 1 + zeros. There is deliberately no OBU-level
// trailing_bits: tile groups are `usedArith` in §5.2.1.
std::vector<std::uint8_t> make_closed_loop_key_obu(
    const StillPictureConfig& config, std::span<const std::uint8_t> tile_payload);

// Placeholder for the not-yet-landed normative §8.2 range coder. A single 0x00
// byte is intentionally not a conforming tile: it contains no exit_symbol()
// trailing 1 bit, and §8.2.2 init_symbol() needs at least 2 bytes
// (Min(sz*8, 15) bits). A stream built with it is therefore clearly incomplete
// rather than a plausible-looking but silently wrong stream.
inline constexpr std::array<std::uint8_t, 1> kPlaceholderTilePayload{0x00};

// Temporal delimiter OBU (§5.5): empty payload, so §5.2.1 adds no
// trailing_bits. The resulting bytes are 01 08 for AV2 (see namespace note).
std::vector<std::uint8_t> make_temporal_delimiter_obu();

// Assembles one M0 temporal unit in the §7.3.3 order TD -> SEQ -> CLK. The
// result is ready for IVF framing; src/io/ivf.* owns the IVF wrapper.
std::vector<std::uint8_t> assemble_m0_still_picture(
    const StillPictureConfig& config, std::span<const std::uint8_t> tile_payload);

// Same as above with kPlaceholderTilePayload. NOT decodable until the entropy
// stage produces a normative §5.20 tile payload.
std::vector<std::uint8_t> assemble_m0_still_picture_placeholder(
    const StillPictureConfig& config);

}  // namespace av2enc::m0

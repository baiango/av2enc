#include "bitstream/m0_headers.h"

#include "test_common.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using av2enc::BitWriter;
using av2enc::m0::ObuHeader;
using av2enc::m0::ObuType;
using av2enc::m0::StillPictureConfig;

// ---------------------------------------------------------------------------
// Compile-time contract. These are the numbers the M0 header writer must hit.
// ---------------------------------------------------------------------------

static_assert(static_cast<std::uint8_t>(ObuType::kSequenceHeader) == 1);
static_assert(static_cast<std::uint8_t>(ObuType::kTemporalDelimiter) == 2);
static_assert(static_cast<std::uint8_t>(ObuType::kClosedLoopKey) == 4);
static_assert(static_cast<std::uint8_t>(ObuType::kRegularTileGroup) == 7);

static_assert(av2enc::m0::kSequenceHeaderSyntaxBits64x64 == 83);
static_assert(av2enc::m0::kSequenceHeaderSyntaxBits256x256 == 87);
static_assert(av2enc::m0::kSequenceHeaderPayloadBytes64x64 == 11);
static_assert(av2enc::m0::kSequenceHeaderPayloadBytes256x256 == 12);
static_assert(av2enc::m0::kFrameHeaderSyntaxBits64x64 == 22);
static_assert(av2enc::m0::kFrameHeaderSyntaxBits256x256 == 24);
static_assert(av2enc::m0::kTileGroupHeaderBytes64x64 == 3);
static_assert(av2enc::m0::kTileGroupHeaderBytes256x256 == 4);

// §5.4.1 field-by-field encoding for the 64x64 M0 still picture:
//   1                 seq_header_id = uvlc(0)
//   00000             seq_profile_idc = 0
//   1                 single_picture_header_flag
//   00111             seq_level_idx = 7 (level 5.1)
//   1                 chroma_format_idc = uvlc(0) (4:2:0)
//   010               bit_depth_idc = uvlc(1) (8-bit)
//   0101 0101         frame_width/height_bits_minus_1 = 5
//   111111 111111     max_frame_width/height_minus_1 = 63
//   0 ...             crop, SB size, SDP, segments, intra tools, inter tools,
//                     transform/quant/entropy, filters (all zero)
//   1                 disable_drl_reorder
//   0                 seq_max_bvp_drl_bits_minus_1 = ns(3) value 0
//   1                 equal_ac_dc_q
//   10111             base_uv_ac_delta_q = 23 (=> BaseUVAcDeltaQ = 0)
//   0                 uv_ac_delta_q_enabled
//   0 ...             loop filters, seq_tile_info_present_flag, film grain
constexpr std::string_view kSequenceBits64 =
    "10000010011110100101010111111111111100000000000000000100000000000001101110000000000";
static_assert(kSequenceBits64.size() == av2enc::m0::kSequenceHeaderSyntaxBits64x64);

// Same syntax with max_frame_width_minus_1/max_frame_height_minus_1 = 255
// (8-bit fields instead of 6-bit).
constexpr std::string_view kSequenceBits256 =
    "100000100111101001110111111111111111111100000000000000000100000000000001101110000000000";
static_assert(kSequenceBits256.size() == av2enc::m0::kSequenceHeaderSyntaxBits256x256);

// §5.18.2 frame_header_info() for a CLK, 8-bit, QP 90, one-tile frame.
// Default M0 configuration: frozen CDFs (disable_cdf_update = 1).
constexpr std::string_view kFrameBits64 = "1100110101101000000000";
static_assert(kFrameBits64.size() == av2enc::m0::kFrameHeaderSyntaxBits64x64);

// Same frame with enable_cdf_adaptation = true (disable_cdf_update = 0).
constexpr std::string_view kFrameBits64Adaptive = "1100010101101000000000";
static_assert(kFrameBits64Adaptive.size() ==
              av2enc::m0::kFrameHeaderSyntaxBits64x64);

// 256x256 adds increment_tile_cols_log2 = 0 and increment_tile_rows_log2 = 0.
constexpr std::string_view kFrameBits256 = "110011000101101000000000";
static_assert(kFrameBits256.size() == av2enc::m0::kFrameHeaderSyntaxBits256x256);

// §5.19: is_first_tile_group + frame_header + byte_alignment.
constexpr std::string_view kTileGroupBits64 =
    "111001101011010000000000";  // 1 + 22 + 1 zero bit
static_assert(kTileGroupBits64.size() == 24);

constexpr std::array<std::uint8_t, 11> kSequencePayload64{
    0x82, 0x7a, 0x55, 0xff, 0xf0, 0x00, 0x04, 0x00, 0x1b, 0x80, 0x08};

constexpr std::array<std::uint8_t, 13> kSequenceObu64{
    0x0c, 0x04, 0x82, 0x7a, 0x55, 0xff, 0xf0,
    0x00, 0x04, 0x00, 0x1b, 0x80, 0x08};

constexpr std::array<std::uint8_t, 12> kSequencePayload256{
    0x82, 0x7a, 0x77, 0xff, 0xff, 0x00,
    0x00, 0x40, 0x01, 0xb8, 0x00, 0x80};

// OBU framing for the M0 64x64 still picture:
//   01 08                           TD (leb128 size 1, header 0x08)
//   0c 04 <11-byte payload>         SEQ (size 12, header 0x04)
//   05 10 e2 b4 00 00               CLK + placeholder tile byte
// The pinned sample testdata/sample_480x270_130f.ivf begins with the same
// TD bytes 01 08 and uses 0x04 / 0x10 for its sequence-header / CLK headers.
constexpr std::array<std::uint8_t, 21> kAssembledPlaceholder64{
    0x01, 0x08, 0x0c, 0x04, 0x82, 0x7a, 0x55, 0xff, 0xf0, 0x00, 0x04,
    0x00, 0x1b, 0x80, 0x08, 0x05, 0x10, 0xe6, 0xb4, 0x00, 0x00};

std::string bit_string(const BitWriter& writer) {
  std::string out;
  out.reserve(writer.bit_size());
  const std::span<const std::uint8_t> bytes = writer.bytes();
  for (std::size_t i = 0; i < writer.bit_size(); ++i) {
    const unsigned bit_in_byte = static_cast<unsigned>(i & 7U);
    const bool bit = ((bytes[i / 8U] >> (7U - bit_in_byte)) & 1U) != 0U;
    out.push_back(bit ? '1' : '0');
  }
  return out;
}

std::string to_hex(std::span<const std::uint8_t> bytes) {
  static constexpr char kDigits[] = "0123456789abcdef";
  std::string out;
  out.reserve(bytes.size() * 3U);
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    if (i != 0) {
      out.push_back(' ');
    }
    out.push_back(kDigits[bytes[i] >> 4U]);
    out.push_back(kDigits[bytes[i] & 0x0FU]);
  }
  return out;
}

bool expect_bytes(std::span<const std::uint8_t> actual,
                  std::span<const std::uint8_t> expected) {
  if (actual.size() == expected.size() &&
      std::equal(actual.begin(), actual.end(), expected.begin())) {
    return true;
  }
  std::cerr << "  actual:   " << to_hex(actual) << '\n';
  std::cerr << "  expected: " << to_hex(expected) << '\n';
  return false;
}

bool parse_leb128(std::span<const std::uint8_t> bytes,
                  std::size_t& offset,
                  std::uint32_t& value) {
  value = 0;
  unsigned shift = 0;
  for (unsigned i = 0; i < 5U; ++i) {
    if (offset >= bytes.size()) {
      return false;
    }
    const std::uint8_t byte = bytes[offset++];
    value |= static_cast<std::uint32_t>(byte & 0x7FU) << shift;
    if ((byte & 0x80U) == 0U) {
      return true;
    }
    shift += 7U;
  }
  return false;
}

bool header_matches(std::uint8_t header,
                    ObuType type,
                    std::uint8_t tlayer_id = 0,
                    bool extension_flag = false) {
  const unsigned ext = (header >> 7U) & 1U;
  const unsigned type_bits = (header >> 2U) & 0x1FU;
  const unsigned tlayer_bits = header & 0x03U;
  return ext == (extension_flag ? 1U : 0U) &&
         type_bits == static_cast<unsigned>(type) && tlayer_bits == tlayer_id;
}

struct ParsedObu {
  std::uint32_t size = 0;
  std::size_t header_offset = 0;
  std::uint8_t header = 0;
  std::size_t payload_offset = 0;
  std::uint32_t payload_size = 0;
};

bool parse_obu(std::span<const std::uint8_t> bytes, std::size_t& offset, ParsedObu& out) {
  std::uint32_t size = 0;
  if (!parse_leb128(bytes, offset, size)) {
    return false;
  }
  if (offset >= bytes.size()) {
    return false;
  }
  out.size = size;
  out.header_offset = offset;
  out.header = bytes[offset++];
  const std::size_t header_bytes = 1U + (((out.header >> 7U) & 1U) != 0U ? 1U : 0U);
  if (size < header_bytes || out.header_offset + size > bytes.size()) {
    return false;
  }
  out.payload_offset = out.header_offset + header_bytes;
  out.payload_size = size - static_cast<std::uint32_t>(header_bytes);
  offset = out.header_offset + size;
  return true;
}

// ---------------------------------------------------------------------------
// OBU header and leb128 framing
// ---------------------------------------------------------------------------

void test_obu_header_layout() {
  // §5.2.2: obu_header_extension_flag f(1), obu_type f(5), obu_tlayer_id f(2).
  // The docs' 0x10/0x08/0x20 values are the AV1 layout; AV2 gives 0x08/0x04/0x10.
  {
    BitWriter writer;
    CHECK(av2enc::m0::write_obu_header(writer, ObuHeader{ObuType::kTemporalDelimiter}) == 8U);
    CHECK(writer.byte_size() == 1U);
    CHECK(writer.bytes()[0] == 0x08U);
  }
  {
    BitWriter writer;
    CHECK(av2enc::m0::write_obu_header(writer, ObuHeader{ObuType::kSequenceHeader}) == 8U);
    CHECK(writer.bytes()[0] == 0x04U);
  }
  {
    BitWriter writer;
    CHECK(av2enc::m0::write_obu_header(writer, ObuHeader{ObuType::kClosedLoopKey}) == 8U);
    CHECK(writer.bytes()[0] == 0x10U);
  }
  {
    // Layout exercise only; an M0 OBU_CLOSED_LOOP_KEY must have tlayer_id = 0.
    ObuHeader header;
    header.type = ObuType::kClosedLoopKey;
    header.tlayer_id = 1;
    header.extension_flag = true;
    header.mlayer_id = 2;
    header.xlayer_id = 3;

    BitWriter writer;
    CHECK(av2enc::m0::write_obu_header(writer, header) == 16U);
    CHECK(writer.byte_size() == 2U);
    // 1 00100 01 = 0x91, then 010 00011 = 0x43.
    CHECK(writer.bytes()[0] == 0x91U);
    CHECK(writer.bytes()[1] == 0x43U);
  }
}

void test_leb128() {
  struct Case {
    std::uint32_t value;
    std::array<std::uint8_t, 5> bytes;
    std::size_t size;
  };
  const std::array<Case, 7> cases{{
      {0U, {0x00, 0, 0, 0, 0}, 1},
      {1U, {0x01, 0, 0, 0, 0}, 1},
      {12U, {0x0c, 0, 0, 0, 0}, 1},
      {127U, {0x7f, 0, 0, 0, 0}, 1},
      {128U, {0x80, 0x01, 0, 0, 0}, 2},
      {300U, {0xac, 0x02, 0, 0, 0}, 2},
      {16384U, {0x80, 0x80, 0x01, 0, 0}, 3},
  }};

  for (const Case& test_case : cases) {
    std::vector<std::uint8_t> out;
    av2enc::m0::append_leb128(out, test_case.value);
    CHECK(av2enc::m0::leb128_size(test_case.value) == test_case.size);
    CHECK(out.size() == test_case.size);
    CHECK(expect_bytes(out, std::span<const std::uint8_t>(test_case.bytes.data(),
                                                          test_case.size)));

    std::size_t offset = 0;
    std::uint32_t decoded = 0;
    CHECK(parse_leb128(out, offset, decoded));
    CHECK(decoded == test_case.value);
    CHECK(offset == out.size());
  }
}

void test_obu_framing() {
  const std::array<std::uint8_t, 2> payload{0xaa, 0xbb};

  {
    // Framing only: num_bytes_in_obu = 1 header byte + 2 payload bytes = 3.
    const std::vector<std::uint8_t> obu =
        av2enc::m0::make_obu(ObuHeader{ObuType::kClosedLoopKey}, payload);
    const std::array<std::uint8_t, 4> expected{0x03, 0x10, 0xaa, 0xbb};
    CHECK(expect_bytes(obu, expected));
  }
  {
    // Extension: num_bytes_in_obu = 1 + 1 + 2 = 4; obuPayloadSize = 4-1-1 = 2.
    ObuHeader header;
    header.type = ObuType::kClosedLoopKey;
    header.tlayer_id = 1;
    header.extension_flag = true;
    header.mlayer_id = 2;
    header.xlayer_id = 3;

    const std::vector<std::uint8_t> obu = av2enc::m0::make_obu(header, payload);
    const std::array<std::uint8_t, 5> expected{0x04, 0x91, 0x43, 0xaa, 0xbb};
    CHECK(expect_bytes(obu, expected));
  }
}

void test_temporal_delimiter_obu() {
  // §5.5: empty payload. §5.2.1 does not call trailing_bits() when
  // obuPayloadSize == 0, so the OBU is exactly two bytes.
  const std::vector<std::uint8_t> obu = av2enc::m0::make_temporal_delimiter_obu();
  const std::array<std::uint8_t, 2> expected{0x01, 0x08};
  CHECK(expect_bytes(obu, expected));

  std::size_t offset = 0;
  ParsedObu parsed;
  CHECK(parse_obu(obu, offset, parsed));
  CHECK(parsed.size == 1U);
  CHECK(parsed.header == 0x08U);
  CHECK(header_matches(parsed.header, ObuType::kTemporalDelimiter));
  CHECK(parsed.payload_size == 0U);
  CHECK(offset == obu.size());
}

// ---------------------------------------------------------------------------
// Sequence header
// ---------------------------------------------------------------------------

void test_sequence_header_64x64() {
  const StillPictureConfig config;  // 64x64, level 7, QP 90

  BitWriter writer;
  const std::size_t bits = av2enc::m0::write_sequence_header(writer, config);
  CHECK(bits == 83U);
  CHECK(writer.bit_size() == 83U);
  CHECK(writer.byte_size() == 11U);
  CHECK(bit_string(writer) == std::string(kSequenceBits64));

  // §5.2.1: sequence headers are extensible OBUs, so an obu_extension_flag
  // (0) precedes §5.2.3 trailing_bits. 83 + 1 + 1 = 85 -> 88 with zero pad.
  writer.put_bit(false);  // obu_extension_flag
  writer.put_bit(true);   // trailing_one_bit
  writer.align_to_byte();
  CHECK(writer.bit_size() == 88U);
  CHECK(writer.byte_size() == av2enc::m0::kSequenceHeaderPayloadBytes64x64);
  // Bits 80..87: three syntax zero bits, ext flag 0, trailing 1, 3 pad zeros.
  CHECK(bit_string(writer).substr(80U) == "00001000");
  CHECK(expect_bytes(writer.bytes(), kSequencePayload64));

  const std::vector<std::uint8_t> obu = av2enc::m0::make_sequence_header_obu(config);
  CHECK(obu.size() == 13U);
  CHECK(expect_bytes(obu, kSequenceObu64));

  std::size_t offset = 0;
  ParsedObu parsed;
  CHECK(parse_obu(obu, offset, parsed));
  CHECK(parsed.size == 12U);  // includes the 1-byte OBU header
  CHECK(parsed.header == 0x04U);
  CHECK(header_matches(parsed.header, ObuType::kSequenceHeader));
  CHECK(parsed.payload_size == 11U);
  // Last byte 0x08 = syntax zero bits, obu_extension_flag = 0, trailing 1,
  // then three zero pad bits.
  CHECK(obu[parsed.payload_offset + parsed.payload_size - 1U] == 0x08U);
  CHECK(offset == obu.size());
}

void test_sequence_header_256x256() {
  StillPictureConfig config;
  config.max_frame_width = 256;
  config.max_frame_height = 256;

  BitWriter writer;
  CHECK(av2enc::m0::write_sequence_header(writer, config) ==
        av2enc::m0::kSequenceHeaderSyntaxBits256x256);
  CHECK(bit_string(writer) == std::string(kSequenceBits256));

  writer.put_bit(false);  // obu_extension_flag
  writer.put_bit(true);   // trailing_one_bit
  writer.align_to_byte();
  CHECK(writer.byte_size() == av2enc::m0::kSequenceHeaderPayloadBytes256x256);
  // Bits 87..95: ext flag 0, trailing 1, 7 pad zeros.
  CHECK(bit_string(writer).substr(87U) == "010000000");
  CHECK(expect_bytes(writer.bytes(), kSequencePayload256));

  const std::vector<std::uint8_t> obu = av2enc::m0::make_sequence_header_obu(config);
  std::array<std::uint8_t, 14> expected{0x0d, 0x04};
  std::copy(kSequencePayload256.begin(), kSequencePayload256.end(), expected.begin() + 2);
  CHECK(expect_bytes(obu, expected));

  std::size_t offset = 0;
  ParsedObu parsed;
  CHECK(parse_obu(obu, offset, parsed));
  CHECK(parsed.size == 13U);  // 1-byte OBU header + 12-byte payload
  CHECK(parsed.payload_size == av2enc::m0::kSequenceHeaderPayloadBytes256x256);
  CHECK(offset == obu.size());
}

// ---------------------------------------------------------------------------
// Frame header and tile-group header
// ---------------------------------------------------------------------------

void test_frame_header_64x64() {
  const StillPictureConfig config;

  BitWriter writer;
  CHECK(av2enc::m0::write_frame_header(writer, config) == 22U);
  CHECK(writer.bit_size() == 22U);
  CHECK(writer.byte_size() == 3U);
  CHECK(bit_string(writer) == std::string(kFrameBits64));
  CHECK(writer.bytes()[0] == 0xcdU);
  CHECK(writer.bytes()[1] == 0x68U);
  CHECK(writer.bytes()[2] == 0x00U);
}

void test_frame_header_256x256() {
  StillPictureConfig config;
  config.max_frame_width = 256;
  config.max_frame_height = 256;

  BitWriter writer;
  CHECK(av2enc::m0::write_frame_header(writer, config) == 24U);
  CHECK(bit_string(writer) == std::string(kFrameBits256));
}

void test_frame_header_adaptive_cdf() {
  StillPictureConfig config;
  config.enable_cdf_adaptation = true;

  BitWriter writer;
  CHECK(av2enc::m0::write_frame_header(writer, config) == 22U);
  CHECK(bit_string(writer) == std::string(kFrameBits64Adaptive));
}

void test_frame_header_disable_cdf_update() {
  StillPictureConfig config;
  config.disable_cdf_update = true;

  BitWriter writer;
  CHECK(av2enc::m0::write_frame_header(writer, config) == 22U);
  CHECK(bit_string(writer) == std::string(kFrameBits64));

  // The explicit adaptation opt-in overrides the legacy disable field.
  config.enable_cdf_adaptation = true;
  BitWriter adaptive_writer;
  CHECK(av2enc::m0::write_frame_header(adaptive_writer, config) == 22U);
  CHECK(bit_string(adaptive_writer) == std::string(kFrameBits64Adaptive));
}

void test_tile_group_header_64x64() {
  const StillPictureConfig config;

  BitWriter writer;
  CHECK(av2enc::m0::write_tile_group_header(writer, config) == 24U);
  CHECK(writer.byte_aligned());
  CHECK(writer.byte_size() == 3U);
  CHECK(bit_string(writer) == std::string(kTileGroupBits64));

  const std::array<std::uint8_t, 3> expected{0xe6, 0xb4, 0x00};
  CHECK(expect_bytes(writer.bytes(), expected));
}

void test_tile_group_header_256x256() {
  StillPictureConfig config;
  config.max_frame_width = 256;
  config.max_frame_height = 256;

  BitWriter writer;
  CHECK(av2enc::m0::write_tile_group_header(writer, config) == 32U);
  CHECK(writer.byte_size() == 4U);
  const std::array<std::uint8_t, 4> expected{0xe6, 0x2d, 0x00, 0x00};
  CHECK(expect_bytes(writer.bytes(), expected));
}

void test_closed_loop_key_obu() {
  const StillPictureConfig config;

  {
    // Header only, no tile data yet: payload is the 3-byte tile-group header,
    // and num_bytes_in_obu = 1 + 3 = 4.
    const std::vector<std::uint8_t> obu = av2enc::m0::make_closed_loop_key_obu(config, {});
    const std::array<std::uint8_t, 5> expected{0x04, 0x10, 0xe6, 0xb4, 0x00};
    CHECK(expect_bytes(obu, expected));
  }
  {
    const std::vector<std::uint8_t> obu = av2enc::m0::make_closed_loop_key_obu(
        config, std::span<const std::uint8_t>(av2enc::m0::kPlaceholderTilePayload));
    const std::array<std::uint8_t, 6> expected{0x05, 0x10, 0xe6, 0xb4, 0x00, 0x00};
    CHECK(expect_bytes(obu, expected));
  }

  std::size_t offset = 0;
  ParsedObu parsed;
  const std::vector<std::uint8_t> obu = av2enc::m0::make_closed_loop_key_obu(
      config, std::span<const std::uint8_t>(av2enc::m0::kPlaceholderTilePayload));
  CHECK(parse_obu(obu, offset, parsed));
  CHECK(parsed.size == 5U);
  CHECK(parsed.header == 0x10U);
  CHECK(header_matches(parsed.header, ObuType::kClosedLoopKey));
  CHECK(parsed.payload_size == 4U);
  CHECK(parsed.payload_offset == 2U);  // one leb128 byte + one header byte
  CHECK(obu[parsed.payload_offset + 0U] == 0xe6U);
  CHECK(obu[parsed.payload_offset + 1U] == 0xb4U);
  CHECK(obu[parsed.payload_offset + 2U] == 0x00U);
  CHECK(obu[parsed.payload_offset + 3U] == 0x00U);
  CHECK(offset == obu.size());
}

// ---------------------------------------------------------------------------
// Full M0 temporal unit
// ---------------------------------------------------------------------------

void test_assemble_m0_still_picture() {
  const StillPictureConfig config;
  const std::vector<std::uint8_t> stream =
      av2enc::m0::assemble_m0_still_picture_placeholder(config);

  CHECK(stream.size() == 21U);
  CHECK(expect_bytes(stream, kAssembledPlaceholder64));

  // Parse the three OBUs in §7.3.3 order: TD, SEQ, CLK.
  std::size_t offset = 0;
  ParsedObu td;
  CHECK(parse_obu(stream, offset, td));
  CHECK(td.size == 1U);
  CHECK(td.header == 0x08U);
  CHECK(header_matches(td.header, ObuType::kTemporalDelimiter));
  CHECK(td.payload_size == 0U);

  ParsedObu seq;
  CHECK(parse_obu(stream, offset, seq));
  CHECK(seq.size == 12U);
  CHECK(seq.header == 0x04U);
  CHECK(header_matches(seq.header, ObuType::kSequenceHeader));
  CHECK(seq.payload_size == 11U);
  CHECK(expect_bytes(std::span<const std::uint8_t>(stream).subspan(
                         seq.payload_offset, seq.payload_size),
                     kSequencePayload64));

  ParsedObu clk;
  CHECK(parse_obu(stream, offset, clk));
  CHECK(clk.size == 5U);
  CHECK(clk.header == 0x10U);
  CHECK(header_matches(clk.header, ObuType::kClosedLoopKey));
  CHECK(clk.payload_size == 4U);
  CHECK(stream[clk.payload_offset + 0U] == 0xe6U);
  CHECK(stream[clk.payload_offset + 1U] == 0xb4U);
  CHECK(stream[clk.payload_offset + 2U] == 0x00U);
  CHECK(stream[clk.payload_offset + 3U] == 0x00U);
  CHECK(stream.back() == 0x00U);  // placeholder: no §8.2.4 exit_symbol() tail
                                  // and shorter than the §8.2.2 2-byte minimum
  CHECK(offset == stream.size());
}

}  // namespace

int main() {
  test_obu_header_layout();
  test_leb128();
  test_obu_framing();
  test_temporal_delimiter_obu();
  test_sequence_header_64x64();
  test_sequence_header_256x256();
  test_frame_header_64x64();
  test_frame_header_256x256();
  test_frame_header_adaptive_cdf();
  test_frame_header_disable_cdf_update();
  test_tile_group_header_64x64();
  test_tile_group_header_256x256();
  test_closed_loop_key_obu();
  test_assemble_m0_still_picture();
  return test::finish("m0_headers_test");
}

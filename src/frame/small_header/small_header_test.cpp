// Unit tests for src/encoder/syntax/small_header/.
//
// Every syntax structure is tested three ways:
//   * exact writer bits against a pinned bit string;
//   * writer -> bytes -> independent reader round-trip (bit count included);
//   * validation / truncation error paths.
//
// Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7), §5.4.1-§5.4.4, §5.4.9, §5.4.10,
// §5.18.4.4, §5.18.7.2-§5.18.7.7, §6.19.3.2, §6.19.4.1.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "core/bitstream/bitwriter.h"
#include "frame/small_header/small_header.h"

namespace {

using namespace av2enc::syntax::small_header;
using av2enc::BitWriter;

int g_checks = 0;
int g_failures = 0;

#define CHECK(cond)                                                        \
  do {                                                                     \
    ++g_checks;                                                            \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                                        \
    }                                                                      \
  } while (0)

#define CHECK_EQ(a, b)                                                        \
  do {                                                                        \
    ++g_checks;                                                               \
    const auto va = (a);                                                      \
    const auto vb = (b);                                                      \
    if (!(va == vb)) {                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s == %s\n", __FILE__, __LINE__, #a,   \
                   #b);                                                       \
      ++g_failures;                                                           \
    }                                                                         \
  } while (0)

std::vector<std::uint8_t> pack_bits(const std::string& bits) {
  std::vector<std::uint8_t> out((bits.size() + 7U) / 8U, 0U);
  for (std::size_t i = 0; i < bits.size(); ++i) {
    if (bits[i] == '1') {
      out[i >> 3U] |= static_cast<std::uint8_t>(0x80U >> (i & 7U));
    }
  }
  return out;
}

std::string hex(const std::vector<std::uint8_t>& bytes) {
  static const char* kHex = "0123456789abcdef";
  std::string out;
  for (const std::uint8_t byte : bytes) {
    out.push_back(kHex[byte >> 4U]);
    out.push_back(kHex[byte & 0x0fU]);
  }
  return out;
}

std::vector<std::uint8_t> finish(BitWriter& writer) {
  writer.align_to_byte();
  const std::span<const std::uint8_t> bytes = writer.bytes();
  return std::vector<std::uint8_t>(bytes.begin(), bytes.end());
}

std::string bits_of(const std::vector<std::uint8_t>& bytes,
                    std::size_t bit_count) {
  std::string out;
  out.reserve(bit_count);
  for (std::size_t i = 0; i < bit_count; ++i) {
    out.push_back(((bytes[i >> 3U] >> (7U - (i & 7U))) & 1U) != 0U ? '1' : '0');
  }
  return out;
}

bool same_segment_info(const SegmentInfo& a, const SegmentInfo& b) {
  return a == b;
}

// ---------------------------------------------------------------------------
// Cropping
// ---------------------------------------------------------------------------

void test_cropping() {
  {
    BitWriter writer;
    CroppingWindow window;
    std::string error;
    CHECK(write_cropping_window(writer, window, &error));
    CHECK(error.empty());
    const auto bytes = finish(writer);
    CHECK_EQ(hex(bytes), std::string("00"));
    CHECK_EQ(writer.bit_size(), std::size_t{8});
  }
  {
    CroppingWindow window;
    window.present = true;
    window.left = 1;
    window.right = 2;
    window.top = 3;
    window.bottom = 4;
    BitWriter writer;
    std::string error;
    CHECK(write_cropping_window(writer, window, &error));
    CHECK_EQ(writer.bit_size(), std::size_t{17});
    const auto bytes = finish(writer);
    // 1 010 011 00100 00101, then seven zero pad bits.
    CHECK_EQ(hex(bytes), std::string("a64280"));

    BitReader reader(bytes, 17);
    CroppingWindow parsed;
    CHECK(read_cropping_window(reader, &parsed, &error));
    CHECK_EQ(reader.bit_pos(), std::size_t{17});
    CHECK(parsed.present);
    CHECK_EQ(parsed.left, 1U);
    CHECK_EQ(parsed.right, 2U);
    CHECK_EQ(parsed.top, 3U);
    CHECK_EQ(parsed.bottom, 4U);
  }
  {
    CroppingWindow window;
    window.present = true;
    window.left = 8;
    window.right = 8;
    window.top = 8;
    window.bottom = 8;
    std::string error;
    CHECK(validate_cropping_window(window, 64, 64, &error));
    DisplaySize size;
    CHECK(derive_display_size(window, 64, 64, 64, 64, &size, &error));
    CHECK_EQ(size.width, 48U);
    CHECK_EQ(size.height, 48U);
  }
  {
    // §5.18.4.4 scales the maximum-frame offsets to a smaller FrameWidth.
    CroppingWindow window;
    window.present = true;
    window.left = 16;
    window.right = 16;
    window.top = 8;
    window.bottom = 8;
    DisplaySize size;
    std::string error;
    CHECK(derive_display_size(window, 1920, 1080, 1280, 720, &size, &error));
    CHECK_EQ(size.width, 1260U);   // 1270 - 10
    CHECK_EQ(size.height, 710U);   // 715 - 5
  }
  {
    CroppingWindow window;
    window.present = true;
    window.left = 64;
    std::string error;
    CHECK(!validate_cropping_window(window, 64, 64, &error));
    CHECK(!error.empty());
  }
  {
    CroppingWindow window;
    window.present = true;
    window.left = 60;
    window.right = 60;
    std::string error;
    CHECK(!validate_cropping_window(window, 64, 64, &error));
    CHECK(!error.empty());
  }
  {
    CroppingWindow window;
    window.present = true;
    window.left = 1;
    const std::vector<std::uint8_t> bytes = pack_bits("1010");
    BitReader reader(bytes, 4);
    CroppingWindow parsed;
    std::string error;
    CHECK(!read_cropping_window(reader, &parsed, &error));
    CHECK(reader.failed());
    CHECK(!error.empty());
  }
}

// ---------------------------------------------------------------------------
// PB aspect ratio
// ---------------------------------------------------------------------------

void test_pb_aspect_ratio() {
  const char* kExpected[] = {"00", "80", "c0"};
  for (int reduce = 0; reduce <= 1; ++reduce) {
    for (int log2 = 0; log2 <= (reduce ? 1 : 0); ++log2) {
      PbAspectRatio config;
      config.reduce = reduce != 0;
      config.max_pb_aspect_ratio_log2_minus_1 = static_cast<std::uint8_t>(log2);
      std::string error;
      CHECK(validate_pb_aspect_ratio(config, &error));
      CHECK_EQ(max_pb_aspect_ratio(config), reduce ? 2 << log2 : 8);

      BitWriter writer;
      CHECK(write_pb_aspect_ratio(writer, config, &error));
      const std::size_t expected_bits = reduce ? 2U : 1U;
      CHECK_EQ(writer.bit_size(), expected_bits);
      const auto bytes = finish(writer);
      CHECK_EQ(hex(bytes), std::string(kExpected[reduce ? 1 + log2 : 0]));

      BitReader reader(bytes, expected_bits);
      PbAspectRatio parsed;
      CHECK(read_pb_aspect_ratio(reader, &parsed, &error));
      CHECK_EQ(reader.bit_pos(), expected_bits);
      CHECK_EQ(parsed.reduce, config.reduce);
      CHECK_EQ(parsed.max_pb_aspect_ratio_log2_minus_1,
               config.max_pb_aspect_ratio_log2_minus_1);
    }
  }
  {
    PbAspectRatio config;
    config.max_pb_aspect_ratio_log2_minus_1 = 1;
    std::string error;
    CHECK(!validate_pb_aspect_ratio(config, &error));
    CHECK(!error.empty());
  }
  {
    PbAspectRatio config;
    config.reduce = true;
    config.max_pb_aspect_ratio_log2_minus_1 = 2;
    std::string error;
    CHECK(!validate_pb_aspect_ratio(config, &error));
    CHECK(!error.empty());
  }
  // §6.19.4.1 leaf conformance.
  CHECK(pb_leaf_aspect_allowed(16, 16, 2));
  CHECK(pb_leaf_aspect_allowed(16, 8, 2));
  CHECK(!pb_leaf_aspect_allowed(16, 4, 2));
  CHECK(pb_leaf_aspect_allowed(8, 2, 4));
  CHECK(!pb_leaf_aspect_allowed(8, 2, 2));
  CHECK(pb_leaf_aspect_allowed(64, 8, 8));
  CHECK(!pb_leaf_aspect_allowed(64, 4, 8));
  // §6.19.3.2 fragment: too thin rejects NONE; 8:1 rejects all partitions.
  CHECK(!pb_partition_none_allowed_by_aspect(8, 2, 2));
  CHECK(pb_non_none_partition_allowed_by_aspect(8, 2, 2));
  CHECK(!pb_partition_none_allowed_by_aspect(64, 4, 2));
  CHECK(!pb_non_none_partition_allowed_by_aspect(64, 4, 2));
}

// ---------------------------------------------------------------------------
// df_par_bits_minus_2
// ---------------------------------------------------------------------------

void test_df_par() {
  const char* kExpected[] = {"00", "40", "80", "c0"};
  for (std::uint8_t value = 0; value <= 3; ++value) {
    DfPar config;
    config.df_par_bits_minus_2 = value;
    std::string error;
    CHECK(validate_df_par(config, &error));
    CHECK_EQ(config.bits(), 2 + static_cast<int>(value));
    CHECK_EQ(config.min_value(), -(1 << (config.bits() - 1)));
    CHECK_EQ(config.max_value(), (1 << (config.bits() - 1)) - 1);

    BitWriter writer;
    CHECK(write_df_par(writer, config, &error));
    CHECK_EQ(writer.bit_size(), std::size_t{2});
    const auto bytes = finish(writer);
    CHECK_EQ(hex(bytes), std::string(kExpected[value]));

    BitReader reader(bytes, 2);
    DfPar parsed;
    CHECK(read_df_par(reader, &parsed, &error));
    CHECK_EQ(parsed.df_par_bits_minus_2, value);
  }
  {
    DfPar config;
    config.df_par_bits_minus_2 = 2;  // 4 bits: -8..7
    CHECK(df_delta_q_representable(-8, config));
    CHECK(df_delta_q_representable(7, config));
    CHECK(!df_delta_q_representable(8, config));
    CHECK(!df_delta_q_representable(-9, config));
  }
  {
    DfPar config;  // 2 bits: -2..1
    DfDeltaQSet values;
    values.present[0] = true;
    values.value[0] = 2;
    std::string error;
    CHECK(!validate_df_delta_q(config, values, &error));
    CHECK(!error.empty());
    values.value[0] = 1;
    CHECK(validate_df_delta_q(config, values, &error));
    values.present[1] = true;
    values.value[1] = 2;
    CHECK(!validate_df_delta_q(config, values, &error));
  }
  {
    DfPar config;
    config.df_par_bits_minus_2 = 4;
    std::string error;
    CHECK(!validate_df_par(config, &error));
    CHECK(!error.empty());
  }
  {
    const std::vector<std::uint8_t> bytes = pack_bits("0");
    BitReader reader(bytes, 1);
    DfPar config;
    std::string error;
    CHECK(!read_df_par(reader, &config, &error));
    CHECK(!error.empty());
  }
}

// ---------------------------------------------------------------------------
// Sequence segments
// ---------------------------------------------------------------------------

SegmentInfo empty_segment_info() { return SegmentInfo{}; }

void test_segments() {
  {
    SequenceSegmentConfig config;
    std::string error;
    CHECK(validate_sequence_segment_config(config, &error));
    BitWriter writer;
    CHECK(write_sequence_segment_config(writer, config, &error));
    CHECK_EQ(writer.bit_size(), std::size_t{2});
    const auto bytes = finish(writer);
    CHECK_EQ(hex(bytes), std::string("00"));

    BitReader reader(bytes, 2);
    SequenceSegmentConfig parsed;
    CHECK(read_sequence_segment_config(reader, &parsed, &error));
    CHECK_EQ(reader.bit_pos(), std::size_t{2});
    CHECK(!parsed.enable_ext_seg);
    CHECK(!parsed.present);
    CHECK_EQ(parsed.max_segments(), 8);
  }
  {
    SequenceSegmentConfig config;
    config.present = true;
    config.info = empty_segment_info();
    std::string error;
    BitWriter writer;
    CHECK(write_sequence_segment_config(writer, config, &error));
    // enable_ext_seg 0, present 1, allow 0, 8*3 zero feature_enabled bits.
    CHECK_EQ(writer.bit_size(), std::size_t{27});
    CHECK_EQ(bits_of(finish(writer), 27), std::string("010") + std::string(24, '0'));
  }
  {
    SequenceSegmentConfig config;
    config.enable_ext_seg = true;
    config.present = true;
    config.allow_change = true;
    config.info = empty_segment_info();
    std::string error;
    BitWriter writer;
    CHECK(write_sequence_segment_config(writer, config, &error));
    CHECK_EQ(writer.bit_size(), std::size_t{51});
    CHECK_EQ(bits_of(finish(writer), 51),
             std::string("111") + std::string(48, '0'));
    CHECK_EQ(config.max_segments(), 16);
  }
  {
    SequenceSegmentConfig config;
    config.present = true;
    config.info = empty_segment_info();
    config.info.enabled[0][0] = true;
    config.info.value[0][0] = -1;
    std::string error;
    BitWriter writer;
    CHECK(write_sequence_segment_config(writer, config, &error));
    // 010 | seg0: 1 1111111111 0 0 | seg1..7: 21 zero bits.
    const std::string expected = "010" + std::string("1") +
                                 std::string(10, '1') + "00" +
                                 std::string(21, '0');
    const std::size_t bit_count = writer.bit_size();
    const auto bytes = finish(writer);
    CHECK_EQ(bits_of(bytes, bit_count), expected);
    CHECK_EQ(bit_count, std::size_t{37});

    BitReader reader(bytes, 37);
    SequenceSegmentConfig parsed;
    CHECK(read_sequence_segment_config(reader, &parsed, &error));
    CHECK_EQ(reader.bit_pos(), std::size_t{37});
    CHECK(parsed.info.enabled[0][0]);
    CHECK_EQ(parsed.info.value[0][0], -1);
    CHECK(same_segment_info(parsed.info, config.info));
  }
  {
    SequenceSegmentConfig config;
    config.present = true;
    config.info = empty_segment_info();
    config.info.enabled[0][0] = true;
    config.info.value[0][0] = 352;
    std::string error;
    BitWriter writer;
    CHECK(!write_sequence_segment_config(writer, config, &error));
    CHECK(!error.empty());
  }
  {
    SequenceSegmentConfig config;
    config.present = true;
    config.info = empty_segment_info();
    config.info.enabled[0][1] = true;
    config.info.value[0][1] = 1;
    std::string error;
    BitWriter writer;
    CHECK(!write_sequence_segment_config(writer, config, &error));
    CHECK(!error.empty());
  }
  {
    SequenceSegmentConfig config;
    config.allow_change = true;  // not coded when present == 0
    std::string error;
    BitWriter writer;
    CHECK(!write_sequence_segment_config(writer, config, &error));
    CHECK(!error.empty());
  }
  {
    const std::vector<std::uint8_t> bytes = pack_bits("00");
    BitReader reader(bytes, 1);  // truncated after enable_ext_seg
    SequenceSegmentConfig parsed;
    std::string error;
    CHECK(!read_sequence_segment_config(reader, &parsed, &error));
    CHECK(!error.empty());
  }
}

void test_frame_segmentation_persistence() {
  SequenceSegmentConfig sequence;
  sequence.enable_ext_seg = false;
  sequence.present = true;
  sequence.allow_change = false;
  sequence.info = empty_segment_info();
  sequence.info.enabled[2][0] = true;
  sequence.info.value[2][0] = 7;

  {
    FrameSegmentRequest request;
    request.enabled = true;
    request.reuse_sequence = true;
    FrameSegmentState state;
    BitWriter writer;
    std::string error;
    CHECK(write_frame_segmentation_params(writer, sequence, request, true,
                                          false, &state, &error));
    // Only segmentation_enabled; DerivedPrimaryRefFrame == PRIMARY_REF_NONE
    // infers map update, and allow_change == 0 forces reuse with no bit.
    CHECK_EQ(writer.bit_size(), std::size_t{1});
    CHECK_EQ(hex(finish(writer)), std::string("80"));
    CHECK(state.reused_sequence);
    CHECK_EQ(state.info.value[2][0], 7);
    CHECK(state.update_map);
    CHECK(!state.temporal_update);

    const auto bytes = pack_bits("1");
    BitReader reader(bytes, 1);
    FrameSegmentState parsed;
    CHECK(read_frame_segmentation_params(reader, sequence, true, false,
                                         &parsed, &error));
    CHECK_EQ(reader.bit_pos(), std::size_t{1});
    CHECK(parsed.reused_sequence);
    CHECK_EQ(parsed.info.value[2][0], 7);
  }
  {
    FrameSegmentRequest request;
    request.enabled = true;
    request.reuse_sequence = false;  // forbidden by the sequence header
    FrameSegmentState state;
    BitWriter writer;
    std::string error;
    CHECK(!write_frame_segmentation_params(writer, sequence, request, true,
                                           false, &state, &error));
    CHECK(!error.empty());
  }

  sequence.allow_change = true;
  {
    FrameSegmentRequest request;
    request.enabled = true;
    request.reuse_sequence = true;
    FrameSegmentState state;
    BitWriter writer;
    std::string error;
    CHECK(write_frame_segmentation_params(writer, sequence, request, true,
                                          false, &state, &error));
    // segmentation_enabled + reuse_seg_info.
    CHECK_EQ(writer.bit_size(), std::size_t{2});
    CHECK_EQ(bits_of(finish(writer), 2), std::string("11"));
    CHECK(state.reused_sequence);
    CHECK_EQ(state.info.value[2][0], 7);
  }
  {
    FrameSegmentRequest request;
    request.enabled = true;
    request.reuse_sequence = false;
    request.info = empty_segment_info();
    request.info.enabled[0][0] = true;
    request.info.value[0][0] = 3;
    FrameSegmentState state;
    BitWriter writer;
    std::string error;
    CHECK(write_frame_segmentation_params(writer, sequence, request, false,
                                          false, &state, &error));
    // 1 (enabled) 0 (reuse) then seg_info(8): seg0 j0 enabled + su(10)=3,
    // j1/j2 disabled, segments 1..7 disabled. Then DerivedPrimaryRefFrame is
    // not NONE, so update_map/temporal bits follow.
    const std::string expected =
        "10" + std::string("1") + "0000000011" + "00" +
        std::string(21, '0') + "10";
    const std::size_t bit_count = writer.bit_size();
    const auto bytes = finish(writer);
    CHECK_EQ(bits_of(bytes, bit_count), expected);
    CHECK(!state.reused_sequence);
    CHECK_EQ(state.info.value[0][0], 3);
    CHECK(state.update_map);
    CHECK(!state.temporal_update);

    BitReader reader(bytes);
    FrameSegmentState parsed;
    CHECK(read_frame_segmentation_params(reader, sequence, false, false,
                                         &parsed, &error));
    CHECK_EQ(reader.bit_pos(), bit_count);
    CHECK(!parsed.reused_sequence);
    CHECK(same_segment_info(parsed.info, request.info));
    CHECK(parsed.update_map);
    CHECK(!parsed.temporal_update);
  }
  {
    // Sequence has no segment info: the frame must code seg_info(8).
    SequenceSegmentConfig absent;
    FrameSegmentRequest request;
    request.enabled = true;
    request.info = empty_segment_info();
    FrameSegmentState state;
    BitWriter writer;
    std::string error;
    CHECK(write_frame_segmentation_params(writer, absent, request, true, false,
                                          &state, &error));
    CHECK_EQ(writer.bit_size(), std::size_t{1 + 24});
    CHECK(!state.reused_sequence);
    const std::size_t bit_count = writer.bit_size();
    const auto bytes = finish(writer);
    BitReader reader(bytes, bit_count);
    FrameSegmentState parsed;
    CHECK(read_frame_segmentation_params(reader, absent, true, false, &parsed,
                                         &error));
    CHECK_EQ(reader.bit_pos(), bit_count);
    CHECK(!parsed.reused_sequence);
    CHECK(same_segment_info(parsed.info, request.info));
  }
  {
    // enable_ext_seg selects seg_info(16) even when sequence info is absent.
    SequenceSegmentConfig absent;
    absent.enable_ext_seg = true;
    FrameSegmentRequest request;
    request.enabled = true;
    request.info = empty_segment_info();
    FrameSegmentState state;
    BitWriter writer;
    std::string error;
    CHECK(write_frame_segmentation_params(writer, absent, request, true, false,
                                          &state, &error));
    CHECK_EQ(writer.bit_size(), std::size_t{1 + 48});
  }
  {
    FrameSegmentRequest request;
    request.enabled = false;
    FrameSegmentState state;
    BitWriter writer;
    std::string error;
    CHECK(write_frame_segmentation_params(writer, sequence, request, false,
                                          false, &state, &error));
    CHECK_EQ(writer.bit_size(), std::size_t{1});
    CHECK(state.info.enabled[0][0] == false);
  }
  {
    // Non-key frame map-update bits: update_map=1 temporal_update=1.
    FrameSegmentRequest request;
    request.enabled = true;
    request.reuse_sequence = true;
    request.update_map = true;
    request.temporal_update = true;
    FrameSegmentState state;
    BitWriter writer;
    std::string error;
    CHECK(write_frame_segmentation_params(writer, sequence, request, false,
                                          false, &state, &error));
    // 1 enabled, 1 reuse, 1 update, 1 temporal.
    CHECK_EQ(bits_of(finish(writer), 4), std::string("1111"));
    CHECK(state.temporal_update);
  }
}

// ---------------------------------------------------------------------------
// Tiles
// ---------------------------------------------------------------------------

void test_superblock_helpers() {
  CHECK(valid_sb_size(SbSize::k64x64));
  CHECK(valid_sb_size(SbSize::k128x128));
  CHECK(valid_sb_size(SbSize::k256x256));
  CHECK(!valid_sb_size(static_cast<SbSize>(32)));
  CHECK_EQ(sb_width_4x4(SbSize::k64x64), 16);
  CHECK_EQ(sb_width_4x4(SbSize::k128x128), 32);
  CHECK_EQ(sb_width_4x4(SbSize::k256x256), 64);
  CHECK_EQ(sb_shift(SbSize::k64x64), 4);
  CHECK_EQ(sb_shift(SbSize::k128x128), 5);
  CHECK_EQ(sb_shift(SbSize::k256x256), 6);
  CHECK(sequence_superblock_size(false, false) == SbSize::k64x64);
  CHECK(sequence_superblock_size(false, true) == SbSize::k128x128);
  CHECK(sequence_superblock_size(true, true) == SbSize::k256x256);
  CHECK_EQ(tile_log2(1, 1), 0);
  CHECK_EQ(tile_log2(1, 2), 1);
  CHECK_EQ(tile_log2(64, 128), 1);
  CHECK_EQ(tile_log2(2304, 8704), 2);
  CHECK(uniform_eligible(0, 1));
  CHECK(uniform_eligible(1, 2));
  CHECK(!uniform_eligible(1, 1));
  CHECK(uniform_eligible(1, 3));
}

void test_tile_params() {
  {
    TileParamsRequest request;
    TileParams params;
    BitWriter writer;
    std::string error;
    CHECK(write_tile_params(writer, 64, 64, SbSize::k64x64, SbSize::k64x64, 7,
                            0, request, &params, &error));
    CHECK_EQ(writer.bit_size(), std::size_t{1});
    CHECK_EQ(hex(finish(writer)), std::string("80"));
    CHECK(params.uniform_spacing);
    CHECK_EQ(params.sb_cols, 1);
    CHECK_EQ(params.sb_rows, 1);
    CHECK_EQ(params.tile_cols, 1);
    CHECK_EQ(params.tile_rows, 1);
    CHECK_EQ(params.tile_cols_log2, 0);
    CHECK_EQ(params.tile_rows_log2, 0);
    CHECK_EQ(params.sb_col_starts.size(), std::size_t{1});
    CHECK_EQ(params.sb_col_starts[0], 0);
    CHECK_EQ(params.sb_shift, 4);
  }
  {
    TileParamsRequest request;
    TileParams params;
    BitWriter writer;
    std::string error;
    CHECK(write_tile_params(writer, 640, 360, SbSize::k64x64, SbSize::k64x64,
                            7, 0, request, &params, &error));
    // uniform=1, increment_tile_cols_log2=0, increment_tile_rows_log2=0.
    CHECK_EQ(writer.bit_size(), std::size_t{3});
    CHECK_EQ(bits_of(finish(writer), 3), std::string("100"));
    CHECK_EQ(params.sb_cols, 10);
    CHECK_EQ(params.sb_rows, 6);
    CHECK_EQ(params.tile_cols, 1);
    CHECK_EQ(params.tile_rows, 1);
  }
  {
    // 8192-wide frame: level 5.1 forces minLog2TileCols/minLog2TileRows = 1,
    // so the increment bits are 1 then 0 (the writer stops at the minimum).
    TileParamsRequest request;
    TileParams params;
    BitWriter writer;
    std::string error;
    CHECK(write_tile_params(writer, 8192, 4320, SbSize::k64x64,
                            SbSize::k64x64, 7, 0, request, &params, &error));
    CHECK_EQ(writer.bit_size(), std::size_t{3});
    CHECK_EQ(bits_of(finish(writer), 3), std::string("100"));
    CHECK_EQ(params.sb_cols, 128);
    CHECK_EQ(params.sb_rows, 68);
    CHECK_EQ(params.tile_cols, 2);
    CHECK_EQ(params.tile_rows, 2);
    CHECK_EQ(params.tile_cols_log2, 1);
    CHECK_EQ(params.tile_rows_log2, 1);
    CHECK_EQ(params.sb_col_starts[0], 0);
    CHECK_EQ(params.sb_col_starts[1], 64);
    CHECK_EQ(params.sb_row_starts[0], 0);
    CHECK_EQ(params.sb_row_starts[1], 34);

    const auto bytes = finish(writer);
    BitReader reader(bytes, 3);
    TileParams parsed;
    CHECK(read_tile_params(reader, 8192, 4320, SbSize::k64x64,
                           SbSize::k64x64, 7, 0, &parsed, &error));
    CHECK_EQ(reader.bit_pos(), std::size_t{3});
    CHECK_EQ(parsed.tile_cols, 2);
    CHECK_EQ(parsed.tile_rows, 2);
    CHECK(parsed.sb_col_starts == params.sb_col_starts);
    CHECK(parsed.sb_row_starts == params.sb_row_starts);
  }
  {
    // Non-uniform spacing: 10x6 superblocks, three columns (3,3,4) and three
    // rows (2,2,2). This is the §5.18.7.3 else branch.
    TileParamsRequest request;
    request.width_in_sbs = {3, 3, 4};
    request.height_in_sbs = {2, 2, 2};
    TileParams params;
    BitWriter writer;
    std::string error;
    CHECK(write_tile_params(writer, 640, 360, SbSize::k64x64, SbSize::k64x64,
                            7, 0, request, &params, &error));
    CHECK_EQ(params.tile_cols, 3);
    CHECK_EQ(params.tile_rows, 3);
    CHECK(!params.uniform_spacing);
    CHECK_EQ(params.sb_col_starts[0], 0);
    CHECK_EQ(params.sb_col_starts[1], 3);
    CHECK_EQ(params.sb_col_starts[2], 6);
    CHECK_EQ(params.sb_row_starts[0], 0);
    CHECK_EQ(params.sb_row_starts[1], 2);
    CHECK_EQ(params.sb_row_starts[2], 4);

    const std::size_t bit_count = writer.bit_size();
    const auto bytes = finish(writer);
    BitReader reader(bytes, bit_count);
    TileParams parsed;
    CHECK(read_tile_params(reader, 640, 360, SbSize::k64x64,
                           SbSize::k64x64, 7, 0, &parsed, &error));
    CHECK_EQ(reader.bit_pos(), bit_count);
    CHECK(parsed.sb_col_starts == params.sb_col_starts);
    CHECK(parsed.sb_row_starts == params.sb_row_starts);
    CHECK_EQ(parsed.tile_cols_log2, 2);
    CHECK_EQ(parsed.tile_rows_log2, 2);
  }
  {
    TileParamsRequest request;
    request.tile_cols_log2 = 99;
    TileParams params;
    BitWriter writer;
    std::string error;
    CHECK(!write_tile_params(writer, 640, 360, SbSize::k64x64,
                             SbSize::k64x64, 7, 0, request, &params, &error));
    CHECK(!error.empty());
  }
  {
    TileParamsRequest request;
    request.width_in_sbs = {2, 3, 4};
    request.height_in_sbs = {2, 2, 2};
    TileParams params;
    BitWriter writer;
    std::string error;
    CHECK(!write_tile_params(writer, 640, 360, SbSize::k64x64,
                             SbSize::k64x64, 7, 0, request, &params, &error));
    CHECK(!error.empty());
  }
  {
    // Reserved level/tier combination must be rejected, not guessed.
    TileParamsRequest request;
    TileParams params;
    BitWriter writer;
    std::string error;
    CHECK(!write_tile_params(writer, 640, 360, SbSize::k64x64,
                             SbSize::k64x64, 22, 0, request, &params, &error));
    CHECK(!error.empty());
  }
}

void test_sequence_tile_config() {
  {
    SequenceTileConfig config;
    BitWriter writer;
    std::string error;
    CHECK(write_sequence_tile_config(writer, &config, 64, 64, SbSize::k64x64,
                                     7, 0, &error));
    CHECK_EQ(writer.bit_size(), std::size_t{1});
    CHECK_EQ(hex(finish(writer)), std::string("00"));
    const auto absent_tile_bytes = pack_bits("0");
    BitReader reader(absent_tile_bytes, 1);
    SequenceTileConfig parsed;
    CHECK(read_sequence_tile_config(reader, &parsed, 64, 64, SbSize::k64x64,
                                    7, 0, &error));
    CHECK_EQ(reader.bit_pos(), std::size_t{1});
    CHECK(!parsed.present);
  }
  {
    SequenceTileConfig config;
    config.present = true;
    BitWriter writer;
    std::string error;
    CHECK(write_sequence_tile_config(writer, &config, 64, 64, SbSize::k64x64,
                                     7, 0, &error));
    // present=1, allow_change=0, tile_params uniform=1.
    CHECK_EQ(writer.bit_size(), std::size_t{3});
    CHECK_EQ(bits_of(finish(writer), 3), std::string("101"));
    CHECK_EQ(config.params.tile_cols, 1);
    CHECK_EQ(config.params.tile_rows, 1);

    const auto bytes = finish(writer);
    BitReader reader(bytes, 3);
    SequenceTileConfig parsed;
    CHECK(read_sequence_tile_config(reader, &parsed, 64, 64, SbSize::k64x64,
                                    7, 0, &error));
    CHECK_EQ(reader.bit_pos(), std::size_t{3});
    CHECK(parsed.present);
    CHECK_EQ(parsed.params.tile_cols, 1);
  }
  {
    SequenceTileConfig config;
    config.present = true;
    config.allow_change = true;
    BitWriter writer;
    std::string error;
    CHECK(write_sequence_tile_config(writer, &config, 640, 360,
                                     SbSize::k64x64, 7, 0, &error));
    // 1 present, 1 allow, then uniform + two zero increments = 11100.
    CHECK_EQ(writer.bit_size(), std::size_t{5});
    CHECK_EQ(bits_of(finish(writer), 5), std::string("11100"));
    CHECK_EQ(config.params.sb_cols, 10);
    CHECK_EQ(config.params.sb_rows, 6);
  }
  {
    SequenceTileConfig config;
    config.present = true;
    config.request.tile_cols_log2 = 1;
    config.request.tile_rows_log2 = 1;
    BitWriter writer;
    std::string error;
    CHECK(write_sequence_tile_config(writer, &config, 640, 360,
                                     SbSize::k64x64, 7, 0, &error));
    CHECK_EQ(config.params.tile_cols, 2);
    CHECK_EQ(config.params.tile_rows, 2);
  }
  {
    // seqSbSize comes from get_seq_sb_size(): a 128x128 sequence superblock
    // changes the tile_params frame grid, not the sequence syntax ordering.
    SequenceTileConfig config;
    config.present = true;
    BitWriter writer;
    std::string error;
    CHECK(write_sequence_tile_config(writer, &config, 640, 360,
                                     SbSize::k128x128, 7, 0, &error));
    CHECK_EQ(writer.bit_size(), std::size_t{5});
    CHECK_EQ(config.params.uniform_sb_size, SbSize::k128x128);
    CHECK_EQ(config.params.sb_cols, 5);
    CHECK_EQ(config.params.sb_rows, 3);
    CHECK_EQ(config.params.tile_cols, 1);
    CHECK_EQ(config.params.tile_rows, 1);
    CHECK_EQ(config.params.sb_shift, 5);
    CHECK(validate_sequence_tile_config(config, 640, 360, SbSize::k128x128,
                                        7, 0, &error));
    CHECK(!validate_sequence_tile_config(config, 640, 360, SbSize::k64x64, 7,
                                         0, &error));
    CHECK(!error.empty());

    const auto bytes = finish(writer);
    BitReader reader(bytes, 5);
    SequenceTileConfig parsed;
    CHECK(read_sequence_tile_config(reader, &parsed, 640, 360,
                                    SbSize::k128x128, 7, 0, &error));
    CHECK_EQ(reader.bit_pos(), std::size_t{5});
    CHECK_EQ(parsed.params.sb_cols, 5);
    CHECK_EQ(parsed.params.sb_rows, 3);
  }
  {
    SequenceTileConfig config;
    config.allow_change = true;  // not coded with present == 0
    BitWriter writer;
    std::string error;
    CHECK(!write_sequence_tile_config(writer, &config, 64, 64,
                                      SbSize::k64x64, 7, 0, &error));
    CHECK(!error.empty());
  }
}

void test_frame_tile_info() {
  SequenceTileConfig sequence;
  sequence.present = true;
  {
    BitWriter writer;
    std::string error;
    CHECK(write_sequence_tile_config(writer, &sequence, 64, 64,
                                     SbSize::k64x64, 7, 0, &error));
  }
  {
    FrameTileRequest request;
    FrameTileInfo info;
    BitWriter writer;
    std::string error;
    CHECK(write_frame_tile_info(writer, sequence, 64, 64, SbSize::k64x64, 7,
                                0, request, &info, &error));
    // allow_tile_info_change == 0 -> the decoder infers reuse, no bits.
    CHECK_EQ(writer.bit_size(), std::size_t{0});
    CHECK(info.reused_sequence);
    CHECK_EQ(info.params.tile_cols, 1);
    CHECK_EQ(info.params.tile_rows, 1);

    BitReader reader(std::span<const std::uint8_t>{}, 0);
    FrameTileInfo parsed;
    CHECK(read_frame_tile_info(reader, sequence, 64, 64, SbSize::k64x64, 7, 0,
                               &parsed, &error));
    CHECK_EQ(reader.bit_pos(), std::size_t{0});
    CHECK(parsed.reused_sequence);
    CHECK_EQ(parsed.params.tile_rows, 1);
  }
  sequence.allow_change = true;
  {
    FrameTileRequest request;
    request.reuse_sequence = true;
    FrameTileInfo info;
    BitWriter writer;
    std::string error;
    CHECK(write_frame_tile_info(writer, sequence, 64, 64, SbSize::k64x64, 7,
                                0, request, &info, &error));
    CHECK_EQ(writer.bit_size(), std::size_t{1});
    CHECK_EQ(hex(finish(writer)), std::string("80"));
    CHECK(info.reused_sequence);
  }
  {
    FrameTileRequest request;
    request.reuse_sequence = false;
    FrameTileInfo info;
    BitWriter writer;
    std::string error;
    CHECK(write_frame_tile_info(writer, sequence, 64, 64, SbSize::k64x64, 7,
                                0, request, &info, &error));
    // reuse_tile_info=0 followed by tile_params uniform=1.
    CHECK_EQ(writer.bit_size(), std::size_t{2});
    CHECK_EQ(bits_of(finish(writer), 2), std::string("01"));
    CHECK(!info.reused_sequence);
    CHECK_EQ(info.params.tile_cols, 1);
  }
  {
    // Uniform sequence params stay eligible at a different frame size and
    // uniform_spacing() recomputes the starts from that frame's MiCols/MiRows.
    FrameTileRequest request;
    request.reuse_sequence = true;
    FrameTileInfo info;
    BitWriter writer;
    std::string error;
    CHECK(write_frame_tile_info(writer, sequence, 1280, 720, SbSize::k64x64,
                                7, 0, request, &info, &error));
    CHECK_EQ(writer.bit_size(), std::size_t{1});
    CHECK(info.reused_sequence);
    CHECK_EQ(info.params.sb_cols, 20);
    CHECK_EQ(info.params.sb_rows, 12);
    CHECK_EQ(info.params.tile_cols, 1);
    CHECK_EQ(info.params.sb_col_starts[0], 0);
  }
  {
    // A sequence without tile params always codes the full frame tile_params.
    SequenceTileConfig absent;
    FrameTileRequest request;
    FrameTileInfo info;
    BitWriter writer;
    std::string error;
    CHECK(write_frame_tile_info(writer, absent, 64, 64, SbSize::k64x64, 7, 0,
                                request, &info, &error));
    CHECK_EQ(writer.bit_size(), std::size_t{1});
    CHECK_EQ(hex(finish(writer)), std::string("80"));
    CHECK(!info.reused_sequence);
  }
  {
    // Non-uniform sequence params are only eligible at the same superblock
    // grid; here the 1280x720 frame needs 20x12 rather than 10x6.
    SequenceTileConfig nonuniform;
    nonuniform.present = true;
    nonuniform.request.width_in_sbs = {5, 5};
    nonuniform.request.height_in_sbs = {3, 3};
    BitWriter seq_writer;
    std::string error;
    CHECK(write_sequence_tile_config(seq_writer, &nonuniform, 640, 360,
                                     SbSize::k64x64, 7, 0, &error));
    FrameTileRequest request;
    request.reuse_sequence = true;
    FrameTileInfo info;
    BitWriter writer;
    CHECK(write_frame_tile_info(writer, nonuniform, 1280, 720, SbSize::k64x64,
                                7, 0, request, &info, &error));
    // eligible == false, so reuse_tile_info is not coded at all; the frame
    // writes a fresh uniform tile_params for 20x12 superblocks (uniform flag,
    // then one zero increment bit for each direction).
    CHECK(!info.reused_sequence);
    CHECK_EQ(writer.bit_size(), std::size_t{3});
    CHECK_EQ(info.params.sb_cols, 20);
  }
}

// ---------------------------------------------------------------------------
// Reader robustness
// ---------------------------------------------------------------------------

void test_reader_truncation() {
  const std::span<const std::uint8_t> empty;
  {
    BitReader reader(empty, 0);
    bool bit = false;
    CHECK(!reader.read_bit(&bit));
    CHECK(reader.failed());
  }
  {
    BitReader reader(empty, 0);
    std::uint32_t value = 0;
    CHECK(!reader.read_bits(1, &value));
    CHECK(reader.failed());
  }
  {
    BitReader reader(empty, 0);
    std::uint32_t value = 0;
    CHECK(!reader.read_uvlc(&value));
    CHECK(reader.failed());
  }
  {
    BitReader reader(empty, 0);
    std::int32_t value = 0;
    CHECK(!reader.read_su(1, &value));
    CHECK(reader.failed());
  }
  {
    // su(10) sign extension: 0x3ff -> -1, 0x001 -> 1.
    const auto minus_one = pack_bits("1111111111");
    BitReader reader(minus_one, 10);
    std::int32_t value = 0;
    CHECK(reader.read_su(10, &value));
    CHECK_EQ(value, -1);
    const auto plus_one = pack_bits("0000000001");
    BitReader reader2(plus_one, 10);
    CHECK(reader2.read_su(10, &value));
    CHECK_EQ(value, 1);
  }
}

}  // namespace

int main() {
  test_cropping();
  test_pb_aspect_ratio();
  test_df_par();
  test_segments();
  test_frame_segmentation_persistence();
  test_superblock_helpers();
  test_tile_params();
  test_sequence_tile_config();
  test_frame_tile_info();
  test_reader_truncation();

  if (g_failures != 0) {
    std::fprintf(stderr, "small_header_test: %d failure(s) / %d checks\n",
                 g_failures, g_checks);
    return 1;
  }
  std::printf("small_header_test: PASS (%d checks)\n", g_checks);
  return 0;
}

// CPU tests for the standalone 10-bit (M10) still picture.
//
// Covered here:
//   * 10-bit header fields: bit_depth_idc = 0, MaxQ = 303, base_q_idx f(9)
//   * rejected configurations (8/12-bit, q > 90, q > 303, 0)
//   * geometry guard for the fixed 32x32/64x64 tree
//   * block plan order
//   * end-to-end encode of a deterministic 10-bit picture: block stream,
//     coefficient magnitudes above 255 (escape path), recon range, OBU order
//
// The dav2d decode + 0-LSB recon gate is `gate_10bit_still.sh`.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "encoder/format/format.h"
#include "encoder/m10/m10_headers.h"
#include "encoder/m10/m10_still.h"
#include "bitstream/bitwriter.h"

namespace {

int failures = 0;
int checks = 0;

void expect(bool condition, const std::string& what) {
  ++checks;
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
  }
}

av2enc::fmt::Picture make_picture(std::uint32_t width, std::uint32_t height,
                                  bool noise) {
  av2enc::fmt::Format format;
  format.width = width;
  format.height = height;
  format.chroma = av2enc::fmt::ChromaFormat::k420;
  format.depth = av2enc::fmt::BitDepth::k10;
  av2enc::fmt::Picture picture;
  std::string error;
  if (!av2enc::fmt::allocate_picture(format, picture, error)) {
    std::fprintf(stderr, "allocate_picture: %s\n", error.c_str());
    std::abort();
  }
  std::uint32_t state = 0x12345678U;
  const auto next = [&state]() {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
  };
  const std::uint32_t cw = (width + 1U) / 2U;
  const std::uint32_t ch = (height + 1U) / 2U;
  for (std::uint32_t y = 0; y < height; ++y) {
    for (std::uint32_t x = 0; x < width; ++x) {
      std::uint32_t value =
          (x * 9U + y * 5U + ((x ^ y) * 3U) + 128U) & 1023U;
      if (noise) {
        value = (value + (next() % 257U)) & 1023U;
      }
      picture.planes[0].at(x, y) = static_cast<std::uint16_t>(value);
    }
  }
  for (std::uint32_t y = 0; y < ch; ++y) {
    for (std::uint32_t x = 0; x < cw; ++x) {
      const std::uint32_t base = (x * 17U + y * 7U + 256U) & 1023U;
      picture.planes[1].at(x, y) = static_cast<std::uint16_t>(base);
      picture.planes[2].at(x, y) =
          static_cast<std::uint16_t>((base + (noise ? (next() % 97U) : 0U)) &
                                     1023U);
    }
  }
  av2enc::fmt::replicate_borders(picture.planes[0]);
  av2enc::fmt::replicate_borders(picture.planes[1]);
  av2enc::fmt::replicate_borders(picture.planes[2]);
  return picture;
}

void test_config() {
  av2enc::m10::M10Config config;
  config.max_frame_width = 64;
  config.max_frame_height = 64;
  config.depth = av2enc::fmt::BitDepth::k10;
  config.base_q_idx = 90;
  std::string error;
  expect(av2enc::m10::validate_config(config, &error), "10-bit q90 config");

  av2enc::m10::M10Config bad = config;
  bad.depth = av2enc::fmt::BitDepth::k8;
  expect(!av2enc::m10::validate_config(bad, &error), "8-bit rejected");
  bad.depth = av2enc::fmt::BitDepth::k12;
  expect(!av2enc::m10::validate_config(bad, &error), "12-bit rejected");
  bad = config;
  bad.base_q_idx = 0;
  expect(!av2enc::m10::validate_config(bad, &error), "q0 rejected");
  bad.base_q_idx = 60;
  expect(!av2enc::m10::validate_config(bad, &error),
         "q60 rejected (fixed QP-90 quantizer)");
  bad.base_q_idx = 91;
  expect(!av2enc::m10::validate_config(bad, &error), "q91 rejected (slice 1)");
  bad.base_q_idx = 304;
  expect(!av2enc::m10::validate_config(bad, &error), "q304 rejected (MaxQ)");

  const auto counts = av2enc::m10::header_bit_counts(config);
  expect(counts.sequence_syntax_bits == 81, "64x64 seq syntax bits = 81");
  expect(counts.frame_syntax_bits == 23, "64x64 frame syntax bits = 23");
  expect(counts.tile_group_bits == 24, "64x64 tile group bits = 24");

  av2enc::m10::M10Config hd = config;
  hd.max_frame_width = 640;
  hd.max_frame_height = 360;
  const auto hd_counts = av2enc::m10::header_bit_counts(hd);
  expect(hd_counts.sequence_syntax_bits == 88, "640x360 seq syntax bits = 88");
  expect(hd_counts.frame_syntax_bits == 25, "640x360 frame syntax bits = 25");
  expect(hd_counts.tile_group_bits == 32, "640x360 tile group bits = 32");

  // bit_depth_idc is the single bit `1` (uvlc(0)) after chroma_format_idc.
  av2enc::BitWriter writer;
  av2enc::m10::write_sequence_header(writer, config);
  const std::vector<std::uint8_t> bytes = writer.take_bytes();
  expect(bytes.size() == 11, "64x64 sequence payload = 11 bytes");
  const auto bit_at = [&bytes](std::size_t index) {
    return (bytes[index / 8] >> (7 - (index % 8))) & 1U;
  };
  // 1 (seq_header_id) + 5 (profile) + 1 (single) + 5 (level) + 1 (chroma) = 13
  expect(bit_at(13) == 1, "bit_depth_idc = 0 (single uvlc bit)");
  expect(bit_at(14) == 0, "bit 14 continues width_bits_minus_1");

  // frame_header(): base_q_idx is f(9) at bit 6 for a 64x64 one-tile frame
  // (uvlc(0) cur_mfh_id, uvlc(0) seq_header_id, 4 flag bits, tile flag).
  av2enc::BitWriter frame_writer;
  av2enc::m10::write_frame_header(frame_writer, config);
  const std::vector<std::uint8_t> frame_bytes = frame_writer.take_bytes();
  const auto frame_bit = [&frame_bytes](std::size_t index) {
    return (frame_bytes[index / 8] >> (7 - (index % 8))) & 1U;
  };
  std::uint32_t q_value = 0;
  for (int i = 0; i < 9; ++i) {
    q_value = (q_value << 1U) | frame_bit(static_cast<std::size_t>(6 + i));
  }
  expect(q_value == 90U, "frame base_q_idx f(9) raw value 90");
}

void test_geometry() {
  std::string reason;
  expect(av2enc::m10::m10_geometry_supported(64, 64, &reason), "64x64 supported");
  expect(av2enc::m10::m10_geometry_supported(640, 360, &reason),
         "640x360 supported");
  expect(av2enc::m10::m10_geometry_supported(100, 100, &reason),
         "100x100 supported");
  expect(av2enc::m10::m10_geometry_supported(1920, 1080, &reason),
         "1920x1080 supported");
  expect(av2enc::m10::m10_geometry_supported(3840, 2160, &reason),
         "3840x2160 supported");
  expect(!av2enc::m10::m10_geometry_supported(129, 100, &reason),
         "129x100 rejected (implied partition)");
  expect(!av2enc::m10::m10_geometry_supported(641, 361, &reason),
         "641x361 rejected (implied partition)");
}

void test_plan() {
  const auto plan = av2enc::m10::make_block_plan(64, 64);
  expect(plan.size() == 4, "64x64 plan has 4 blocks");
  const int expected[4][2] = {{0, 0}, {8, 0}, {0, 8}, {8, 8}};
  for (int i = 0; i < 4 && i < static_cast<int>(plan.size()); ++i) {
    expect(plan[static_cast<std::size_t>(i)].mi_row == expected[i][0] &&
               plan[static_cast<std::size_t>(i)].mi_col == expected[i][1],
           "block order TL,BL,TR,BR");
    expect(plan[static_cast<std::size_t>(i)].luma_size == 32,
           "64x64 uses 32x32 leaves");
  }
  const auto edge = av2enc::m10::make_block_plan(100, 100);
  expect(edge.size() == 7, "100x100 plan has 6 full + 1 edge block");
  expect(edge.back().luma_size == 64, "partial superblock is a 64x64 leaf");
}

void test_encode(std::uint32_t width, std::uint32_t height, bool noise,
                 int expected_blocks) {
  const av2enc::fmt::Picture picture = make_picture(width, height, noise);
  av2enc::m10::M10Options options;
  options.base_q_idx = 90;
  av2enc::m10::M10Result result;
  std::string error;
  expect(av2enc::m10::encode_m10_cpu(picture, options, result, error),
         "encode " + std::to_string(width) + "x" + std::to_string(height) +
             ": " + error);
  if (!error.empty()) return;
  expect(static_cast<int>(result.blocks.size()) == expected_blocks,
         "block count");
  expect(!result.frame.empty() && result.frame.size() == result.frame_bytes,
         "frame bytes");
  expect(result.tile_bytes > 0 && result.tile_bytes < result.frame_bytes,
         "tile bytes inside frame");
  // §7.3.3 order: OBU_TEMPORAL_DELIMITER (0x01 0x08), then sequence header.
  expect(result.frame.size() > 6 && result.frame[0] == 0x01 &&
             result.frame[1] == 0x08,
         "frame starts with temporal delimiter");
  expect(result.frame[3] == 0x04, "sequence header OBU follows TD");

  int max_level = 0;
  for (std::int32_t level : result.levels) {
    max_level = std::max(max_level, std::abs(level));
  }
  expect(max_level <= 32767, "levels fit int16");
  for (int plane = 0; plane < 3; ++plane) {
    const auto& recon = result.recon[plane];
    for (int y = 0; y < recon.visible_height; ++y) {
      for (int x = 0; x < recon.visible_width; ++x) {
        const std::uint16_t value =
            recon.samples[static_cast<std::size_t>(y) * recon.width + x];
        if (value > 1023) {
          expect(false, "recon sample exceeds 1023");
          return;
        }
      }
    }
  }
  std::printf("  %ux%u: blocks=%zu levels=%zu max|level|=%d frame=%u B\n",
              width, height, result.blocks.size(), result.levels.size(),
              max_level, result.frame_bytes);
  if (noise) {
    // 10-bit coefficient magnitudes are up to ~4x the 8-bit range, so this
    // exercises the base/br/escape path with levels above 255.
    expect(max_level > 255, "noise case exercises levels above 255");
  }
}

}  // namespace

int main() {
  test_config();
  test_geometry();
  test_plan();
  test_encode(64, 64, false, 4);
  test_encode(64, 64, true, 4);
  test_encode(100, 100, true, 7);
  test_encode(640, 360, true, 210);
  if (failures == 0) {
    std::printf("m10_test: PASS (%d checks)\n", checks);
    return 0;
  }
  std::fprintf(stderr, "m10_test: %d failure(s) in %d checks\n", failures,
               checks);
  return 1;
}

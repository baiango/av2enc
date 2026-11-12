// Unit tests for the 10-bit `src/encoder/hbd/` component.
//
// Covered:
//   * depth-dependent sample ranges (Clip1, DC bias, edge defaults, dequant /
//     inverse clips, residual range, deblock q offset, CfL ranges)
//   * §7.14.2 qlookup / MaxQ and §8.3.1 coefficient-CDF slice selection
//   * §7.17 deblock qThr/side at 8-bit vs 10-bit
//   * §7.13.5 CfL luma average / scaled-luma clip
//   * sequence/frame header bit goldens (bit_depth_idc = 0, f(9) base_q_idx)
//   * writer <-> reader round-trip for q = 1, 64, 90, 140, 191, 303
//   * deblock-enabled header round-trip
//   * 64x64 encode: M10 byte-equivalence at q = 90, qlookup effect at q = 64,
//     reader round-trip, bit golden, geometry guard, qindex seam
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/bitstream/bitwriter.h"
#include "formats/high_bitdepth/hbd_depth.h"
#include "formats/high_bitdepth/hbd_headers.h"
#include "formats/high_bitdepth/hbd_reader.h"
#include "formats/high_bitdepth/hbd_still.h"
#include "formats/format.h"
#include "pipeline/still/high_bitdepth/high_bitdepth_still.h"

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

void expect_int(long long got, long long want, const std::string& what) {
  ++checks;
  if (got != want) {
    std::fprintf(stderr, "FAIL: %s (got %lld, want %lld)\n", what.c_str(), got,
                 want);
    ++failures;
  }
}

std::uint64_t fnv1a(const std::vector<std::uint8_t>& bytes) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (std::uint8_t byte : bytes) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  }
  return hash;
}

std::string hex(const std::vector<std::uint8_t>& bytes) {
  static const char* kDigits = "0123456789abcdef";
  std::string out;
  out.reserve(bytes.size() * 2);
  for (std::uint8_t byte : bytes) {
    out.push_back(kDigits[byte >> 4U]);
    out.push_back(kDigits[byte & 0x0fU]);
  }
  return out;
}

av2enc::fmt::Picture make_picture(std::uint32_t width, std::uint32_t height) {
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
  std::uint32_t state = 0x9E3779B9U;
  const auto next = [&state]() {
    state ^= state << 13U;
    state ^= state >> 17U;
    state ^= state << 5U;
    return state;
  };
  const std::uint32_t cw = (width + 1U) / 2U;
  const std::uint32_t ch = (height + 1U) / 2U;
  for (std::uint32_t y = 0; y < height; ++y) {
    for (std::uint32_t x = 0; x < width; ++x) {
      const std::uint32_t v =
          (x * 5U + y * 3U + ((x ^ y) << 1U) + (next() & 63U)) & 1023U;
      picture.planes[0].at(x, y) = static_cast<std::uint16_t>(v);
    }
  }
  for (std::uint32_t y = 0; y < ch; ++y) {
    for (std::uint32_t x = 0; x < cw; ++x) {
      picture.planes[1].at(x, y) =
          static_cast<std::uint16_t>((x * 11U + y * 7U + (next() & 31U)) &
                                     1023U);
      picture.planes[2].at(x, y) =
          static_cast<std::uint16_t>((x * 3U + y * 13U + (next() & 31U)) &
                                     1023U);
    }
  }
  av2enc::fmt::replicate_borders(picture.planes[0]);
  av2enc::fmt::replicate_borders(picture.planes[1]);
  av2enc::fmt::replicate_borders(picture.planes[2]);
  return picture;
}

av2enc::hbd::HbdConfig config_for(std::uint32_t w, std::uint32_t h,
                                  std::uint16_t q) {
  av2enc::hbd::HbdConfig config;
  config.max_frame_width = static_cast<std::uint16_t>(w);
  config.max_frame_height = static_cast<std::uint16_t>(h);
  config.base_q_idx = q;
  return config;
}

void test_depth_ranges() {
  using namespace av2enc::hbd::depth;
  const SampleRanges r = sample_ranges();
  expect_int(r.bit_depth, 10, "depth.bit_depth");
  expect_int(r.sample_max, 1023, "depth.sample_max");
  expect_int(r.dc_bias, 512, "depth.dc_bias");
  expect_int(r.intra_above_default, 511, "depth.above_default");
  expect_int(r.intra_left_default, 513, "depth.left_default");
  expect_int(r.intra_corner_default, 512, "depth.corner_default");
  expect_int(r.dequant_min, -(1 << 17), "depth.dequant_min");
  expect_int(r.dequant_max, (1 << 17) - 1, "depth.dequant_max");
  expect_int(r.inverse_row_min, -(1 << 17), "depth.inverse_row_min");
  expect_int(r.inverse_row_max, (1 << 17) - 1, "depth.inverse_row_max");
  expect_int(r.inverse_col_min, -1024, "depth.inverse_col_min");
  expect_int(r.inverse_col_max, 1023, "depth.inverse_col_max");
  expect_int(r.residual_min, -1024, "depth.residual_min");
  expect_int(r.residual_max, 1023, "depth.residual_max");
  expect_int(r.deblock_qindex_offset, 48, "depth.deblock_offset");
  expect_int(r.cfl_luma_avg_bias, 4096, "depth.cfl_avg_bias");
  expect_int(r.cfl_luma_avg_max, 8191, "depth.cfl_avg_max");
  expect_int(r.cfl_shift, 11, "depth.cfl_shift");

  expect_int(clip1(-1), 0, "clip1 low");
  expect_int(clip1(1024), 1023, "clip1 high");
  expect_int(clip1(1023), 1023, "clip1 max");
  expect_int(clip_dequant(-(1 << 17) - 1), -(1 << 17), "clip_dequant low");
  expect_int(clip_dequant(1 << 17), (1 << 17) - 1, "clip_dequant high");
  expect_int(clip_inverse_row(std::int64_t{1} << 40), (1 << 17) - 1,
             "clip_inverse_row high");
  expect_int(clip_inverse_col(-(1 << 11)), -1024, "clip_inverse_col low");
  expect(residual_representable(-1024), "residual min representable");
  expect(residual_representable(1023), "residual max representable");
  expect(!residual_representable(1024), "residual 1024 not representable");
}

void test_qindex_selection() {
  using namespace av2enc::hbd::depth;
  expect_int(max_q(8), 255, "max_q(8)");
  expect_int(max_q(10), 303, "max_q(10)");
  expect_int(max_q(12), -1, "max_q(12) reserved");
  expect_int(base_q_idx_bits(8), 8, "base_q_idx_bits(8)");
  expect_int(base_q_idx_bits(10), 9, "base_q_idx_bits(10)");

  expect_int(qlookup(0), 64, "qlookup(0)");
  expect_int(qlookup(1), 40, "qlookup(1)");
  expect_int(qlookup(25), 80, "qlookup(25)");
  expect_int(qlookup(26), 82, "qlookup(26)");
  expect_int(qlookup(90), 528, "qlookup(90)");
  expect_int(qlookup(91), 544, "qlookup(91)");
  expect_int(qlookup(140), 2240, "qlookup(140)");
  expect_int(qlookup(303), 245760, "qlookup(303)");
  expect_int(get_q(0, 0), 64, "get_q(0,0)");
  expect_int(get_q(90, 0), 528, "get_q(90,0)");
  expect_int(get_q(303, 0), 245760, "get_q(303,0)");
  expect_int(get_q(300, 10), get_q(303, 0), "get_q clamps to MaxQ");

  const int bounds[][2] = {{1, 0},    {90, 0},  {91, 1},  {140, 1},
                           {141, 2},  {190, 2}, {191, 3}, {303, 3}};
  for (const auto& entry : bounds) {
    expect_int(qctx_slice(entry[0]), entry[1],
               "qctx_slice(" + std::to_string(entry[0]) + ")");
  }
}

void test_deblock_ranges() {
  using namespace av2enc::hbd::depth;
  // 8-bit and 10-bit do not produce the same side for the same raw qindex:
  // the 10-bit index is offset by 48.  At q=90 the 8-bit side is 1 and the
  // 10-bit side is 0 (the §7.17 apply condition then disables filtering).
  const DeblockStrength r8 = deblock_strength(90, 8, 0, 0);
  const DeblockStrength r10 = deblock_strength(90, 10, 0, 0);
  expect_int(r8.q_ind, 90, "deblock 8-bit q_ind(90)");
  expect_int(r8.side, 1, "deblock 8-bit side(90)");
  expect_int(r10.q_ind, 42, "deblock 10-bit q_ind(90)");
  expect_int(r10.side, 0, "deblock 10-bit side(90)");
  expect_int(r10.q_thr, 1, "deblock 10-bit qThr(90)");

  const DeblockStrength r140 = deblock_strength(140, 10, 0, 0);
  expect_int(r140.q_ind, 92, "deblock 10-bit q_ind(140)");
  expect_int(r140.side, 4, "deblock 10-bit side(140)");

  const DeblockStrength r303 = deblock_strength(303, 10, 0, 0);
  expect_int(r303.q_ind, 255, "deblock 10-bit q_ind(303)");
  expect_int(r303.side, 160, "deblock 10-bit side(303)");

  // A chroma AC delta changes the clamped lvl.
  const DeblockStrength rc = deblock_strength(90, 10, 48, 0);
  expect_int(rc.q_ind, 90, "deblock 10-bit q_ind(90+48)");
}

void test_cfl_ranges() {
  using namespace av2enc::hbd::depth;
  expect_int(cfl_luma_average(0, 0), 4096, "cfl default lumaAvg");
  expect_int(cfl_luma_average(8 * 100 * 128, 128), 800, "cfl lumaAvg");
  expect_int(cfl_luma_average(8 * 1024 * 128, 128), 8191, "cfl lumaAvg clamp");
  // scaledLuma = Round2Signed(alpha_q8*(L-lumaAvg), 11).
  expect_int(cfl_scaled_luma(32, 1000, 512), 8, "cfl scaledLuma");
  expect_int(cfl_scaled_luma(-32, 1000, 512), -8, "cfl scaledLuma negative");
  expect_int(cfl_apply(1020, 32, 1000, 512), 1023, "cfl clip high");
  expect_int(cfl_apply(2, -32, 1000, 512), 0, "cfl clip low");
  expect_int(cfl_apply(512, 0, 1000, 512), 512, "cfl zero alpha");
}

void test_header_bits() {
  using namespace av2enc::hbd;
  const HbdConfig c64 = config_for(64, 64, 90);
  const HbdConfig c360 = config_for(640, 360, 90);
  const HbdConfig c4k = config_for(3840, 2160, 90);

  const HeaderBitCounts h64 = header_bit_counts(c64);
  const HeaderBitCounts h360 = header_bit_counts(c360);
  const HeaderBitCounts h4k = header_bit_counts(c4k);
  expect_int(static_cast<long long>(h64.sequence_syntax_bits), 81,
             "header bits 64x64 sequence");
  expect_int(static_cast<long long>(h64.frame_syntax_bits), 23,
             "header bits 64x64 frame");
  expect_int(static_cast<long long>(h64.tile_group_bits), 24,
             "header bits 64x64 tile group");
  expect_int(static_cast<long long>(h360.sequence_syntax_bits), 88,
             "header bits 640x360 sequence");
  expect_int(static_cast<long long>(h360.frame_syntax_bits), 25,
             "header bits 640x360 frame");
  expect_int(static_cast<long long>(h360.tile_group_bits), 32,
             "header bits 640x360 tile group");
  expect_int(static_cast<long long>(h4k.sequence_syntax_bits), 93,
             "header bits 4K sequence");
  expect_int(static_cast<long long>(h4k.frame_syntax_bits), 25,
             "header bits 4K frame");
  expect_int(static_cast<long long>(h4k.tile_group_bits), 32,
             "header bits 4K tile group");

  // Pinned sequence OBU bytes.  bit_depth_idc = 0 (uvlc "1" one-bit code) and
  // base_q_idx f(9) are the two 10-bit differences from the 8-bit writer.
  const std::string seq64 = hex(make_sequence_header_obu(c64));
  const std::string seq360 = hex(make_sequence_header_obu(c360));
  const std::string seq4k = hex(make_sequence_header_obu(c4k));
  // Bit goldens.  The 64x64 sequence OBU is 13 bytes total (leb128 size +
  // OBU header + 11 payload bytes); bit_depth_idc = 0 makes uvlc(0) a single
  // '1' bit, one bit shorter than the 8-bit uvlc(1) = "010".
  expect(seq64 == "0c04827d57ffc00010006e0020", "golden seq OBU 64x64");
  expect(seq360 == "0d04827e627fb380002000dc0040", "golden seq OBU 640x360");
  expect(seq4k == "0d04827eefbfe1bc00010006e002", "golden seq OBU 4K");
  {
    av2enc::BitWriter writer;
    write_tile_group_header(writer, c64);
    expect(hex(writer.take_bytes()) == "e65a00",
           "golden tile-group header 64x64 q90");
    av2enc::BitWriter writer360;
    write_tile_group_header(writer360, config_for(640, 360, 90));
    expect(hex(writer360.take_bytes()) == "e6168000",
           "golden tile-group header 640x360 q90");
  }

  // Round-trip every qindex class through the reader.  This also proves the
  // header writer supports the full 10-bit MaxQ range even though the reused
  // tile entropy is slice-0 only.
  const std::uint16_t qindexes[] = {1, 64, 90, 140, 191, 303};
  for (std::uint16_t q : qindexes) {
    HbdConfig config = config_for(640, 360, q);
    const std::vector<std::uint8_t> frame =
        assemble_still_picture(config, std::span<const std::uint8_t>{});
    HbdParsedStream parsed;
    std::string error;
    expect(parse_stream(frame, &parsed, &error),
           "reader parses q=" + std::to_string(q) + ": " + error);
    expect(parsed.has_sequence && parsed.has_frame, "reader sees seq+frame");
    expect_int(parsed.sequence.bit_depth_idc, 0, "reader bit_depth_idc = 0");
    expect_int(parsed.sequence.chroma_format_idc, 0, "reader 4:2:0");
    expect_int(parsed.sequence.max_frame_width, 640, "reader width");
    expect_int(parsed.sequence.max_frame_height, 360, "reader height");
    expect_int(parsed.frame.base_q_idx, q, "reader base_q_idx");
    expect(parsed.frame.disable_cdf_update, "reader disable_cdf_update");
    expect_int(parsed.frame.reduced_tx_set, 0, "reader reduced_tx_set");
    expect(!parsed.frame.apply_deblocking_filter[0], "reader deblock off");
    expect(parsed.tile_payload.empty(), "reader empty tile payload");
  }

  // Deblock-enabled frame header: all four planes + four df_delta bits.
  HbdConfig deblock = config_for(640, 360, 140);
  deblock.apply_deblock_y_vert = true;
  deblock.apply_deblock_y_horz = true;
  deblock.apply_deblock_u = true;
  deblock.apply_deblock_v = true;
  const std::vector<std::uint8_t> frame =
      assemble_still_picture(deblock, std::span<const std::uint8_t>{});
  HbdParsedStream parsed;
  std::string error;
  expect(parse_stream(frame, &parsed, &error),
         "reader parses deblock header: " + error);
  expect(parsed.frame.apply_deblocking_filter[0], "deblock y vert");
  expect(parsed.frame.apply_deblocking_filter[1], "deblock y horz");
  expect(parsed.frame.apply_deblocking_filter[2], "deblock u");
  expect(parsed.frame.apply_deblocking_filter[3], "deblock v");
  expect_int(parsed.frame.base_q_idx, 140, "deblock header q");

  // Configuration validation: 12-bit, q=0, q>303, zero dimensions.
  {
    HbdConfig bad = c64;
    bad.depth = av2enc::fmt::BitDepth::k12;
    expect(!validate_config(bad, &error), "reject 12-bit");
    bad = c64;
    bad.depth = av2enc::fmt::BitDepth::k8;
    expect(!validate_config(bad, &error), "reject 8-bit");
    bad = c64;
    bad.base_q_idx = 0;
    expect(!validate_config(bad, &error), "reject q=0");
    bad = c64;
    bad.base_q_idx = 304;
    expect(!validate_config(bad, &error), "reject q=304");
    bad = c64;
    bad.max_frame_width = 0;
    expect(!validate_config(bad, &error), "reject zero width");
    bad = c64;
    bad.apply_deblock_u = true;  // chroma-only deblock cannot be signalled
    expect(!validate_config(bad, &error), "reject chroma-only deblock");
  }
}

void test_still_encode() {
  using namespace av2enc::hbd;
  const av2enc::fmt::Picture picture = make_picture(64, 64);

  // Encoding seam: 1..90 only (slice-0 entropy), header handles 1..303.
  std::string reason;
  expect(qindex_encodable(90, &reason), "q=90 encodable");
  expect(qindex_encodable(1, &reason), "q=1 encodable");
  expect(!qindex_encodable(91, &reason), "q=91 seam rejected");
  expect(!qindex_encodable(0, &reason), "q=0 rejected");
  expect(!qindex_encodable(304, &reason), "q=304 rejected");

  // q=90 must be byte-identical to the verified M10 path.
  HbdOptions options;
  options.base_q_idx = 90;
  HbdResult hbd90;
  std::string error;
  expect(encode_still(picture, options, hbd90, error),
         "hbd encode q=90: " + error);
  expect_int(hbd90.quant_step, 528, "hbd q=90 step");

  av2enc::high_bitdepth::HighBitDepthOptions high_bitdepth_options;
  high_bitdepth_options.base_q_idx = 90;
  av2enc::high_bitdepth::HighBitDepthResult high_bitdepth_result;
  expect(av2enc::high_bitdepth::encode_high_bitdepth_cpu(picture, high_bitdepth_options, high_bitdepth_result, error),
         "m10 encode: " + error);
  expect(hbd90.frame == high_bitdepth_result.frame, "hbd q=90 frame == M10 frame");
  for (int plane = 0; plane < 3; ++plane) {
    expect(hbd90.recon[plane].samples == high_bitdepth_result.recon[plane].samples,
           "hbd q=90 recon plane " + std::to_string(plane) + " == M10");
  }

  // Reader round-trip of the real stream.
  HbdParsedStream parsed;
  expect(parse_stream(hbd90.frame, &parsed, &error),
         "reader parses encoded stream: " + error);
  expect_int(parsed.frame.base_q_idx, 90, "encoded stream q");
  expect(!parsed.tile_payload.empty(), "encoded stream has tile data");
  expect_int(static_cast<long long>(parsed.tile_payload.size()),
             static_cast<long long>(hbd90.tile_bytes),
             "reader tile payload size == result.tile_bytes");

  // Bit golden for the deterministic 64x64 picture (this picture is the M10
  // generator source, so the gate also pins the full IVF MD5
  // 1b5018610a86b8d79aba161e95770ae6).
  expect_int(hbd90.tile_bytes, 2713, "golden 64x64 q90 tile bytes");
  expect_int(hbd90.frame_bytes, 2734, "golden 64x64 q90 frame bytes");
  expect(fnv1a(hbd90.frame) == 0x2cc8c9b8c0699524ULL,
         "golden 64x64 q90 frame FNV-1a");

  // qlookup actually changes the stream and the reconstruction, and the
  // header carries the raw qindex.
  HbdOptions options64;
  options64.base_q_idx = 64;
  HbdResult hbd64;
  expect(encode_still(picture, options64, hbd64, error),
         "hbd encode q=64: " + error);
  expect_int(hbd64.quant_step, 248, "hbd q=64 step");
  expect(hbd64.frame != hbd90.frame, "q=64 frame differs from q=90");
  HbdParsedStream parsed64;
  expect(parse_stream(hbd64.frame, &parsed64, &error),
         "reader parses q=64 stream: " + error);
  expect_int(parsed64.frame.base_q_idx, 64, "q=64 stream header q");
  expect(hbd64.recon[0].samples != hbd90.recon[0].samples,
         "q=64 recon differs from q=90");

  expect_int(hbd64.tile_bytes, 3647, "golden 64x64 q64 tile bytes");
  expect_int(hbd64.frame_bytes, 3668, "golden 64x64 q64 frame bytes");
  expect(fnv1a(hbd64.frame) == 0x9cd3cf9cad15b699ULL,
         "golden 64x64 q64 frame FNV-1a");

  // Every encoded sample must be inside Clip1's range.
  for (int plane = 0; plane < 3; ++plane) {
    for (std::uint16_t sample : hbd90.recon[plane].samples) {
      if (sample > 1023) {
        expect(false, "recon sample within [0,1023]");
        break;
      }
    }
  }

  // Geometry guard: the M10 fixed tree rejects an implied-partition edge.
  av2enc::fmt::Picture odd = make_picture(100, 100);
  HbdResult odd_result;
  expect(encode_still(odd, options, odd_result, error),
         "100x100 geometry accepted");
  av2enc::fmt::Picture bad = make_picture(129, 100);
  expect(!encode_still(bad, options, odd_result, error),
         "129x100 geometry rejected");
}

}  // namespace

int main() {
  test_depth_ranges();
  test_qindex_selection();
  test_deblock_ranges();
  test_cfl_ranges();
  test_header_bits();
  test_still_encode();

  if (failures == 0) {
    std::printf("hbd_test: PASS (%d checks)\n", checks);
    return 0;
  }
  std::fprintf(stderr, "hbd_test: %d failure(s) in %d checks\n", failures,
               checks);
  return 1;
}

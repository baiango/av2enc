#include "pipeline/still/still_headers.h"

#include "core/bitstream/bitwriter.h"
#include "test_common.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using av2enc::BitWriter;
using av2enc::still::StillConfig;

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

std::string frame_bits(const StillConfig& config, std::size_t* bits = nullptr) {
  BitWriter writer;
  const std::size_t written = av2enc::still::write_frame_header(writer, config);
  if (bits != nullptr) {
    *bits = written;
  }
  return bit_string(writer);
}

struct DeblockCase {
  const char* name;
  StillConfig config;
  std::string_view expected;
};

// Frame-header prefix common to every case below (64x64, QP 90, frozen CDFs):
//   cur_mfh_id                       uvlc(0)          "1"
//   seq_header_id_in_frame_header    uvlc(0)          "1"
//   allow_screen_content_tools       f(1)             "0"
//   allow_intrabc                    f(1)             "0"
//   disable_cdf_update               f(1)             "1"
//   uniform_tile_spacing_flag        f(1)             "1"
//   base_q_idx                       f(8)             01011010 (90)
//   segmentation_enabled             f(1)             "0"
//   using_qmatrix                    f(1)             "0"
//   delta_q_present                  f(1)             "0"
// Then deblocking_filter_params() (§5.18.5.2):
//   apply_deblocking_filter[0]       f(1)
//   apply_deblocking_filter[1]       f(1)
//   apply_deblocking_filter[2..3]    f(1) each, only if [0] || [1]
//   df_delta_q_present[i]            f(1) per enabled plane
// Then:
//   tx_mode_select                   f(1)            "0"
//   reduced_tx_set                   f(2)            "00"
constexpr std::string_view kAllOn64 =
    "1100110101101000011110000000";  // 4 apply flags + 4 delta bits
constexpr std::string_view kAllOff64 =
    "1100110101101000000000";  // 2 apply flags, no chroma flags, no delta bits
constexpr std::string_view kYVertOnly64 =
    "1100110101101000010000000";  // 4 apply flags (0100) + 1 delta bit
constexpr std::string_view kYHorzU64 =
    "11001101011010000011000000";  // 4 apply flags (0110) + 2 delta bits
constexpr std::string_view kLumaOnly64 =
    "11001101011010000110000000";  // 4 apply flags (1100) + 2 delta bits

// 640x360 uses 10x6 superblocks, so tile_info() adds
// increment_tile_cols_log2 = 0 and increment_tile_rows_log2 = 0.
constexpr std::string_view kAllOn640 =
    "110011000101101000011110000000";
constexpr std::string_view kAllOff640 =
    "110011000101101000000000";

StillConfig base_config(std::uint16_t width, std::uint16_t height) {
  StillConfig config;
  config.max_frame_width = width;
  config.max_frame_height = height;
  config.base_q_idx = 90;
  config.disable_cdf_update = true;
  config.apply_deblock_y_vert = false;
  config.apply_deblock_y_horz = false;
  config.apply_deblock_u = false;
  config.apply_deblock_v = false;
  return config;
}

void test_deblock_delta_bits() {
  {
    StillConfig config = base_config(64, 64);
    config.apply_deblock_y_vert = true;
    config.apply_deblock_y_horz = true;
    config.apply_deblock_u = true;
    config.apply_deblock_v = true;
    std::size_t bits = 0;
    const std::string actual = frame_bits(config, &bits);
    CHECK(actual == kAllOn64);
    CHECK(bits == kAllOn64.size());
    CHECK(bits == 28U);  // 24 base bits + 4 df_delta_q_present bits
  }
  {
    StillConfig config = base_config(64, 64);
    std::size_t bits = 0;
    const std::string actual = frame_bits(config, &bits);
    CHECK(actual == kAllOff64);
    CHECK(bits == kAllOff64.size());
    CHECK(bits == 22U);  // no chroma apply flags, no df_delta_q_present bits
  }
  {
    StillConfig config = base_config(64, 64);
    config.apply_deblock_y_vert = true;
    std::size_t bits = 0;
    const std::string actual = frame_bits(config, &bits);
    CHECK(actual == kYVertOnly64);
    CHECK(bits == kYVertOnly64.size());
  }
  {
    StillConfig config = base_config(64, 64);
    config.apply_deblock_y_horz = true;
    config.apply_deblock_u = true;
    std::size_t bits = 0;
    const std::string actual = frame_bits(config, &bits);
    CHECK(actual == kYHorzU64);
    CHECK(bits == kYHorzU64.size());
  }
  {
    StillConfig config = base_config(64, 64);
    config.apply_deblock_y_vert = true;
    config.apply_deblock_y_horz = true;
    std::size_t bits = 0;
    const std::string actual = frame_bits(config, &bits);
    CHECK(actual == kLumaOnly64);
    CHECK(bits == kLumaOnly64.size());
  }
  {
    StillConfig config = base_config(640, 360);
    config.apply_deblock_y_vert = true;
    config.apply_deblock_y_horz = true;
    config.apply_deblock_u = true;
    config.apply_deblock_v = true;
    std::size_t bits = 0;
    const std::string actual = frame_bits(config, &bits);
    CHECK(actual == kAllOn640);
    CHECK(bits == kAllOn640.size());
    CHECK(bits == 30U);  // 26 base bits + 4 df_delta_q_present bits
  }
  {
    StillConfig config = base_config(640, 360);
    std::size_t bits = 0;
    const std::string actual = frame_bits(config, &bits);
    CHECK(actual == kAllOff640);
    CHECK(bits == kAllOff640.size());
  }
}

// Every enabled plane adds exactly one df_delta_q_present bit and nothing else.
void test_delta_count_is_one_per_enabled_plane() {
  const bool values[] = {false, true};
  for (bool yv : values) {
    for (bool yh : values) {
      for (bool u : values) {
        for (bool v : values) {
          StillConfig config = base_config(64, 64);
          config.apply_deblock_y_vert = yv;
          config.apply_deblock_y_horz = yh;
          config.apply_deblock_u = u;
          config.apply_deblock_v = v;
          const int chroma_present = (yv || yh) ? 2 : 0;
          const int enabled = static_cast<int>(yv) + static_cast<int>(yh) +
                              ((yv || yh) ? static_cast<int>(u) +
                                                static_cast<int>(v)
                                          : 0);
          const std::size_t expected_bits = static_cast<std::size_t>(
              17U + 2U + static_cast<std::size_t>(chroma_present) +
              static_cast<std::size_t>(enabled) + 3U);
          std::size_t bits = 0;
          (void)frame_bits(config, &bits);
          CHECK(bits == expected_bits);
        }
      }
    }
  }
}

// assemble_still_picture() must emit the same frame-header bits inside the
// CLK OBU, immediately after is_first_tile_group = 1.
void test_assembled_clk_starts_with_frame_header() {
  StillConfig config = base_config(64, 64);
  config.apply_deblock_y_vert = true;
  config.apply_deblock_y_horz = true;
  config.apply_deblock_u = true;
  config.apply_deblock_v = true;
  const std::vector<std::uint8_t> tile = {0x80};
  const std::vector<std::uint8_t> stream =
      av2enc::still::assemble_still_picture(config, tile);
  CHECK(!stream.empty());
  // TD (01 08), then the sequence OBU, then the CLK OBU with leb128 size.
  CHECK(stream.size() > 6U);
  CHECK(stream[0] == 0x01U);
  CHECK(stream[1] == 0x08U);
  const std::vector<std::uint8_t> seq = av2enc::still::make_sequence_header_obu(config);
  const std::size_t clk_offset = 2U + seq.size();
  CHECK(clk_offset + 2U < stream.size());
  // leb128 is 2 bytes for these streams.
  std::size_t clk_header = clk_offset;
  while ((stream[clk_header] & 0x80U) != 0U) {
    ++clk_header;
  }
  ++clk_header;  // first byte after the leb128 length
  CHECK(stream[clk_header] == 0x10U);  // obu_type = CLOSED_LOOP_KEY
  const std::size_t payload = clk_header + 1U;
  CHECK(payload < stream.size());

  BitWriter assembled;
  for (std::size_t i = payload; i < stream.size(); ++i) {
    assembled.put_bits(stream[i], 8U);
  }
  const std::string assembled_bits = bit_string(assembled);
  const std::string expected = std::string("1") + std::string(kAllOn64);
  CHECK(assembled_bits.substr(0, expected.size()) == expected);
}

}  // namespace

int main() {
  test_deblock_delta_bits();
  test_delta_count_is_one_per_enabled_plane();
  test_assembled_clk_starts_with_frame_header();
  return test::finish("still_headers_test");
}

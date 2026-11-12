// Standalone tests for the M1 tiled still writer.
//
// No CUDA, no decoder, no external dependencies. The tests cover geometry,
// tile_info() bit layout, tile_size_bytes selection, the §5.20.1 interleaved
// tile-group payload, 1x1 parity with the M1 core writers, per-tile CDF/context
// reset (identical tile content must produce identical payloads), and partial
// tile geometry.

#include "pipeline/tiles/stream/tiles_stream.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <span>
#include <string>
#include <vector>

#include "core/bitstream/basic_headers.h"
#include "pipeline/still/still_headers.h"

namespace {

using av2enc::BitWriter;
using av2enc::still::tiles_stream::Config;
using av2enc::still::tiles_stream::Frame;
using av2enc::still::tiles_stream::TilePayloadView;
using av2enc::tiles::TileLayout;
using av2enc::tiles::TileLayoutPreset;

int g_failures = 0;

void check(bool condition, const char* expression, const char* file, int line) {
  if (!condition) {
    ++g_failures;
    std::cerr << file << ':' << line << ": CHECK failed: " << expression << '\n';
  }
}

#define CHECK(expression) ::check((expression), #expression, __FILE__, __LINE__)

// MSB-first bit reader for the syntax written by BitWriter.
class BitReader {
 public:
  explicit BitReader(std::span<const std::uint8_t> data) : data_(data) {}

  bool bit() {
    if (bit_ >= data_.size() * 8U) {
      ok_ = false;
      return false;
    }
    const std::uint8_t byte = data_[bit_ / 8U];
    const bool value = ((byte >> (7U - static_cast<unsigned>(bit_ % 8U))) & 1U) != 0U;
    ++bit_;
    return value;
  }

  std::uint32_t bits(unsigned count) {
    std::uint32_t value = 0;
    for (unsigned i = 0; i < count; ++i) {
      value = (value << 1U) | (bit() ? 1U : 0U);
    }
    return value;
  }

  void skip(std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) bit();
  }

  void align_to_byte() { bit_ = (bit_ + 7U) & ~static_cast<std::size_t>(7U); }

  std::size_t position() const { return bit_; }
  bool ok() const { return ok_; }

 private:
  std::span<const std::uint8_t> data_;
  std::size_t bit_ = 0;
  bool ok_ = true;
};

Config make_config(int width, int height, TileLayoutPreset preset) {
  Config config;
  config.width = width;
  config.height = height;
  config.preset = preset;
  return config;
}

TileLayout plan_or_fail(int width, int height, TileLayoutPreset preset) {
  TileLayout layout;
  std::string error;
  if (!av2enc::still::tiles_stream::build_layout(width, height, preset, 64, 7, &layout, &error)) {
    CHECK(false && "layout unexpectedly rejected");
    std::cerr << "  " << error << '\n';
  }
  return layout;
}

void test_layouts() {
  struct Expect {
    int width;
    int height;
    TileLayoutPreset preset;
    int tiles;
  };
  const Expect cases[] = {
      {640, 360, TileLayoutPreset::k1x1, 1},     {640, 360, TileLayoutPreset::k2x1, 2},
      {640, 360, TileLayoutPreset::k2x2, 4},     {640, 360, TileLayoutPreset::k4x2, 8},
      {640, 360, TileLayoutPreset::k8x2, 16},    {1920, 1080, TileLayoutPreset::k1x1, 1},
      {1920, 1080, TileLayoutPreset::k2x1, 2},   {1920, 1080, TileLayoutPreset::k2x2, 4},
      {1920, 1080, TileLayoutPreset::k4x2, 8},   {1920, 1080, TileLayoutPreset::k8x2, 16},
      {1920, 1080, TileLayoutPreset::k16x2, 32},
  };
  for (const Expect& item : cases) {
    const TileLayout layout = plan_or_fail(item.width, item.height, item.preset);
    CHECK(layout.tile_count() == item.tiles);
    CHECK(layout.width == item.width);
    CHECK(layout.height == item.height);
    std::uint64_t pixels = 0;
    for (const av2enc::tiles::TileRect& rect : layout.tiles) {
      CHECK((rect.pixel_width() & 1) == 0);
      CHECK((rect.pixel_height() & 1) == 0);
      CHECK(av2enc::tiles::sb_cols(rect.pixel_width(), layout.sb_size) == rect.sb_cols);
      CHECK(av2enc::tiles::sb_rows(rect.pixel_height(), layout.sb_size) == rect.sb_rows);
      pixels += rect.pixels();
    }
    CHECK(pixels ==
          static_cast<std::uint64_t>(item.width) * static_cast<std::uint64_t>(item.height));
  }

  // 640x360 has only 10 superblock columns; a 16-column uniform grid cannot be
  // represented and must fail rather than silently change the layout.
  {
    TileLayout layout;
    std::string error;
    CHECK(!av2enc::still::tiles_stream::build_layout(640, 360, TileLayoutPreset::k16x2, 64, 7, &layout,
                                                  &error));
    CHECK(!error.empty());
  }
}

void test_tile_info() {
  struct Case {
    int width;
    int height;
    TileLayoutPreset preset;
  };
  const Case cases[] = {
      {640, 360, TileLayoutPreset::k1x1}, {640, 360, TileLayoutPreset::k2x1},
      {640, 360, TileLayoutPreset::k2x2}, {640, 360, TileLayoutPreset::k4x2},
      {640, 360, TileLayoutPreset::k8x2}, {1920, 1080, TileLayoutPreset::k16x2},
  };
  for (const Case& item : cases) {
    const TileLayout layout = plan_or_fail(item.width, item.height, item.preset);
    const int tile_size_bytes = layout.tile_count() > 1 ? 3 : 0;
    BitWriter writer;
    av2enc::still::tiles_stream::write_tile_info(writer, layout, tile_size_bytes);
    const std::size_t written_bits = writer.bit_size();
    const std::vector<std::uint8_t> bytes = writer.take_bytes();
    BitReader reader(bytes);

    CHECK(reader.bit());  // uniform_tile_spacing_flag
    int cols_log2 = layout.min_log2_tile_cols;
    while (cols_log2 < layout.max_log2_tile_cols) {
      const bool increment = reader.bit();
      if (!increment) break;
      ++cols_log2;
    }
    CHECK(cols_log2 == layout.cols_log2);

    int rows_log2 = std::max(layout.min_log2_tiles - layout.cols_log2, 0);
    while (rows_log2 < layout.max_log2_tile_rows) {
      const bool increment = reader.bit();
      if (!increment) break;
      ++rows_log2;
    }
    CHECK(rows_log2 == layout.rows_log2);

    if (layout.tile_count() > 1) {
      CHECK(static_cast<int>(reader.bits(2)) + 1 == tile_size_bytes);
    }
    CHECK(reader.ok());
    CHECK(reader.position() == written_bits);
  }
}

void test_tile_size_selection() {
  const auto select = [](std::initializer_list<std::uint32_t> sizes, std::string* error) {
    const std::vector<std::uint32_t> values(sizes);
    return av2enc::still::tiles_stream::select_tile_size_bytes(values, error);
  };
  std::string error;
  CHECK(select({2}, &error) == 1);
  CHECK(select({2, 255}, &error) == 1);
  CHECK(select({256}, &error) == 1);
  CHECK(select({257}, &error) == 2);
  CHECK(select({65535, 65536}, &error) == 2);
  CHECK(select({65537}, &error) == 3);
  CHECK(select({16777216}, &error) == 3);
  CHECK(select({16777217}, &error) == 4);
  CHECK(select({4294967295U}, &error) == 4);
  CHECK(select({1}, &error) == 0);
  CHECK(select({}, &error) == 0);
  CHECK(!error.empty());
}

void test_interleaved_assembly() {
  const Config config = make_config(640, 360, TileLayoutPreset::k4x2);
  const TileLayout layout = plan_or_fail(640, 360, TileLayoutPreset::k4x2);
  CHECK(layout.tile_count() == 8);

  const std::uint32_t sizes[] = {2, 300, 4, 5, 6, 70000, 8, 9};
  std::vector<std::vector<std::uint8_t>> payloads;
  std::vector<TilePayloadView> views;
  for (std::size_t i = 0; i < std::size(sizes); ++i) {
    payloads.emplace_back(sizes[i], static_cast<std::uint8_t>(i + 1));
    views.push_back(TilePayloadView{std::span<const std::uint8_t>(payloads.back())});
  }

  std::vector<std::uint8_t> clk_payload;
  int tile_size_bytes = 0;
  std::string error;
  CHECK(av2enc::still::tiles_stream::make_tile_group_payload(layout, config, views, &clk_payload,
                                                          &tile_size_bytes, &error));
  CHECK(tile_size_bytes == 3);

  BitWriter header_writer;
  av2enc::still::tiles_stream::write_frame_header(header_writer, config, layout, tile_size_bytes);
  const std::size_t header_bits =
      1U + header_writer.bit_size() + (layout.tile_count() > 1 ? 1U : 0U);
  const std::size_t header_bytes = (header_bits + 7U) / 8U;

  std::size_t offset = header_bytes;
  for (std::size_t i = 0; i + 1 < payloads.size(); ++i) {
    CHECK(offset + static_cast<std::size_t>(tile_size_bytes) <= clk_payload.size());
    std::uint32_t value = 0;
    for (int b = 0; b < tile_size_bytes; ++b) {
      value |= static_cast<std::uint32_t>(clk_payload[offset + static_cast<std::size_t>(b)])
               << (8 * b);
    }
    CHECK(value == sizes[i] - 1U);
    offset += static_cast<std::size_t>(tile_size_bytes);
    CHECK(offset + sizes[i] <= clk_payload.size());
    CHECK(std::equal(payloads[i].begin(), payloads[i].end(),
                     clk_payload.begin() + static_cast<std::ptrdiff_t>(offset)));
    offset += sizes[i];
  }
  CHECK(offset + sizes[payloads.size() - 1] == clk_payload.size());
  CHECK(std::equal(payloads.back().begin(), payloads.back().end(),
                   clk_payload.begin() + static_cast<std::ptrdiff_t>(offset)));

  // The tile-group header is: is_first_tile_group, frame_header(),
  // tile_start_and_end_present_flag = 0, byte_alignment.
  BitReader reader(clk_payload);
  CHECK(reader.bit());  // is_first_tile_group
  reader.skip(header_writer.bit_size());
  CHECK(!reader.bit());  // tile_start_and_end_present_flag
  reader.align_to_byte();
  CHECK(reader.position() / 8U == header_bytes);

  // Invalid inputs are explicit failures, never a silently different layout.
  {
    std::vector<TilePayloadView> short_views(views.begin(), views.end() - 1);
    CHECK(!av2enc::still::tiles_stream::make_tile_group_payload(
        layout, config, short_views, &clk_payload, &tile_size_bytes, &error));
  }
  {
    std::vector<std::uint8_t> tiny{1};
    std::vector<TilePayloadView> small_views(static_cast<std::size_t>(layout.tile_count()),
                                             TilePayloadView{std::span<const std::uint8_t>(tiny)});
    CHECK(!av2enc::still::tiles_stream::make_tile_group_payload(
        layout, config, small_views, &clk_payload, &tile_size_bytes, &error));
  }
}

void test_core_header_parity() {
  struct Case {
    int width;
    int height;
    bool deblock;
  };
  const Case cases[] = {
      {64, 64, false},
      {640, 360, false},
      {640, 360, true},
      {1920, 1080, false},
  };
  for (const Case& item : cases) {
    Config config = make_config(item.width, item.height, TileLayoutPreset::k2x2);
    config.enable_deblock = item.deblock;
    std::string error;
    CHECK(av2enc::still::tiles_stream::core_header_parity(config, &error));
    if (!error.empty()) std::cerr << "  " << error << '\n';
  }

  // 1x1 end-to-end OBU parity with av2enc::still::assemble_still_picture() when the
  // loop-filter-across-tiles flag is left at the core value (0).
  Config config = make_config(256, 256, TileLayoutPreset::k1x1);
  config.disable_loopfilters_across_tiles = false;
  const TileLayout layout = plan_or_fail(256, 256, TileLayoutPreset::k1x1);
  const std::vector<std::uint8_t> tile_payload = {0x12, 0x34, 0x56, 0x78};
  std::vector<TilePayloadView> views{TilePayloadView{std::span<const std::uint8_t>(tile_payload)}};
  std::vector<std::uint8_t> clk_payload;
  int tile_size_bytes = 0;
  std::string error;
  CHECK(av2enc::still::tiles_stream::make_tile_group_payload(layout, config, views, &clk_payload,
                                                          &tile_size_bytes, &error));
  CHECK(tile_size_bytes == 1);

  av2enc::still::StillConfig m1;
  m1.max_frame_width = 256;
  m1.max_frame_height = 256;
  m1.seq_level_idx = config.seq_level_idx;
  m1.base_q_idx = config.base_q_idx;
  m1.disable_cdf_update = true;
  m1.enable_intra_edge_filter = config.enable_intra_edge_filter;
  m1.enable_cfl_intra = config.enable_cfl_intra;
  m1.enable_chroma_dctonly = config.enable_chroma_dctonly;
  m1.apply_deblock_y_vert = config.enable_deblock;
  m1.apply_deblock_y_horz = config.enable_deblock;
  m1.apply_deblock_u = config.enable_deblock;
  m1.apply_deblock_v = config.enable_deblock;

  const std::vector<std::uint8_t> core = av2enc::still::assemble_still_picture(m1, tile_payload);

  // Rebuild the same temporal unit from our components.
  const std::vector<std::uint8_t> clk =
      av2enc::basic::make_obu(av2enc::basic::ObuHeader{av2enc::basic::ObuType::kClosedLoopKey}, clk_payload);
  const std::vector<std::uint8_t> td =
      av2enc::basic::make_obu(av2enc::basic::ObuHeader{av2enc::basic::ObuType::kTemporalDelimiter},
                           std::span<const std::uint8_t>{});
  const std::vector<std::uint8_t> seq = av2enc::still::tiles_stream::make_sequence_header_obu(config);
  std::vector<std::uint8_t> ours;
  ours.insert(ours.end(), td.begin(), td.end());
  ours.insert(ours.end(), seq.begin(), seq.end());
  ours.insert(ours.end(), clk.begin(), clk.end());
  CHECK(ours == core);
}

void fill_periodic_planes(av2enc::still::StillPicture* picture, std::vector<std::uint8_t>* y,
                          std::vector<std::uint8_t>* u, std::vector<std::uint8_t>* v) {
  const int width = static_cast<int>(picture->width);
  const int height = static_cast<int>(picture->height);
  y->resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
  u->resize(static_cast<std::size_t>(width / 2) * static_cast<std::size_t>(height / 2));
  v->resize(u->size());
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      (*y)[static_cast<std::size_t>(row) * static_cast<std::size_t>(width) +
           static_cast<std::size_t>(col)] =
          static_cast<std::uint8_t>(((col % 64) * 7 + (row % 64) * 3) & 0xFF);
    }
  }
  for (int row = 0; row < height / 2; ++row) {
    for (int col = 0; col < width / 2; ++col) {
      const std::uint8_t value =
          static_cast<std::uint8_t>(((col % 32) * 5 + (row % 32) * 11 + 17) & 0xFF);
      (*u)[static_cast<std::size_t>(row) * static_cast<std::size_t>(width / 2) +
           static_cast<std::size_t>(col)] = value;
      (*v)[static_cast<std::size_t>(row) * static_cast<std::size_t>(width / 2) +
           static_cast<std::size_t>(col)] = static_cast<std::uint8_t>(value ^ 0x5AU);
    }
  }
  picture->y = *y;
  picture->u = *u;
  picture->v = *v;
}

void test_per_tile_reset_and_origins() {
  // 128x128 2x2 gives four 64x64 tiles. With a 64x64-periodic source every tile
  // sees identical input; any CDF/context or above/left carry-over from the
  // previous tile would make the payloads differ.
  av2enc::still::StillPicture picture;
  picture.width = 128;
  picture.height = 128;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
  fill_periodic_planes(&picture, &y, &u, &v);

  const Config config = make_config(128, 128, TileLayoutPreset::k2x2);
  Frame frame;
  std::string error;
  CHECK(av2enc::still::tiles_stream::encode(picture, config, &frame, &error));
  if (!error.empty()) std::cerr << "  " << error << '\n';
  CHECK(frame.tiles.size() == 4);
  if (frame.tiles.size() == 4) {
    for (std::size_t i = 1; i < frame.tiles.size(); ++i) {
      CHECK(frame.tiles[i].payload == frame.tiles[0].payload);
      CHECK(frame.tiles[i].recon_y == frame.tiles[0].recon_y);
      CHECK(frame.tiles[i].local_sb_cols == 1);
      CHECK(frame.tiles[i].local_sb_rows == 1);
    }
  }
  CHECK(!frame.frame.empty());
  CHECK(!frame.clk_payload.empty());
  CHECK(frame.tile_size_bytes >= 1 && frame.tile_size_bytes <= 4);

  av2enc::still::tiles_stream::Reconstruction recon;
  CHECK(av2enc::still::tiles_stream::assemble_reconstruction(frame, &recon, &error));
  CHECK(recon.width == 128);
  CHECK(recon.height == 128);
  CHECK(recon.y.size() == 128U * 128U);
  CHECK(recon.u.size() == 64U * 64U);
  CHECK(recon.v.size() == 64U * 64U);

  // Determinism: the same input and config produce the same bytes.
  Frame again;
  CHECK(av2enc::still::tiles_stream::encode(picture, config, &again, &error));
  CHECK(again.frame == frame.frame);
}

void test_partial_tiles() {
  // 320x180 4x2: the planner produces a partial superblock row (128 / 52 px)
  // and non-uniform column widths (128, 64, 64, 64).
  const int width = 320;
  const int height = 180;
  av2enc::still::StillPicture picture;
  picture.width = static_cast<std::uint32_t>(width);
  picture.height = static_cast<std::uint32_t>(height);
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
  fill_periodic_planes(&picture, &y, &u, &v);

  const Config config = make_config(width, height, TileLayoutPreset::k4x2);
  Frame frame;
  std::string error;
  CHECK(av2enc::still::tiles_stream::encode(picture, config, &frame, &error));
  if (!error.empty()) std::cerr << "  " << error << '\n';
  CHECK(frame.tiles.size() == 8);
  if (frame.tiles.size() == 8) {
    CHECK(frame.layout.tiles[0].pixel_width() == 128);
    CHECK(frame.layout.tiles[1].pixel_width() == 64);
    CHECK(frame.layout.tiles[0].pixel_height() == 64);
    CHECK(frame.layout.tiles[4].pixel_height() == 116);
    CHECK(frame.tiles[0].local_sb_rows == 1);
    CHECK(frame.tiles[4].local_sb_rows == 2);
    CHECK(frame.layout.boundaries.back().has_partial_superblock_row);
  }

  av2enc::still::tiles_stream::Reconstruction recon;
  CHECK(av2enc::still::tiles_stream::assemble_reconstruction(frame, &recon, &error));
  CHECK(recon.width == width);
  CHECK(recon.height == height);
  CHECK(recon.y.size() == static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
}

void test_invalid_config() {
  av2enc::still::StillPicture picture;
  picture.width = 128;
  picture.height = 128;
  std::vector<std::uint8_t> y(128U * 128U, 0);
  std::vector<std::uint8_t> u(64U * 64U, 0);
  std::vector<std::uint8_t> v(64U * 64U, 0);
  picture.y = y;
  picture.u = u;
  picture.v = v;

  Frame frame;
  std::string error;
  {
    Config config = make_config(128, 128, TileLayoutPreset::k1x1);
    config.base_q_idx = 0;
    CHECK(!av2enc::still::tiles_stream::encode(picture, config, &frame, &error));
  }
  {
    Config config = make_config(130, 128, TileLayoutPreset::k2x2);
    CHECK(!av2enc::still::tiles_stream::encode(picture, config, &frame, &error));
  }
  {
    Config config = make_config(128, 128, TileLayoutPreset::k2x2);
    config.sb_size = 128;
    CHECK(!av2enc::still::tiles_stream::encode(picture, config, &frame, &error));
  }
  {
    Config config = make_config(128, 128, TileLayoutPreset::k2x2);
    picture.width = 126;
    picture.height = 126;
    CHECK(!av2enc::still::tiles_stream::encode(picture, config, &frame, &error));
    picture.width = 128;
    picture.height = 128;
  }
}

}  // namespace

int main() {
  test_layouts();
  test_tile_info();
  test_tile_size_selection();
  test_interleaved_assembly();
  test_core_header_parity();
  test_per_tile_reset_and_origins();
  test_partial_tiles();
  test_invalid_config();

  if (g_failures != 0) {
    std::cerr << "tiles_stream_test: " << g_failures << " failure(s)\n";
    return EXIT_FAILURE;
  }
  std::cout << "tiles_stream_test: PASS\n";
  return EXIT_SUCCESS;
}

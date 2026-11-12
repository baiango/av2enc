// Tests for the M1 palette syntax adapter (§5.20.8 / §8.3.2).
#include "codec/intra/palette/stream/palette_stream.h"

#include "codec/intra/palette/core/scc_cdf.h"
#include "codec/partition/tree/partition.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace av2 = av2enc::av2;

namespace {

using av2enc::still::palette_stream::MinimalPaletteLeaf;
using av2enc::still::palette_stream::PaletteModeInfo;
using av2enc::still::palette_stream::PaletteSyntaxReader;
using av2enc::still::palette_stream::PaletteSyntaxWriter;
using av2enc::still::palette_stream::ScreenContentFlags;
using av2enc::palette::PaletteCdfState;
using av2enc::palette::PaletteParams;
using av2enc::palette::PaletteResult;

int g_checks = 0;
int g_failures = 0;

void check(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  }
}

std::string hex(const std::vector<std::uint8_t>& bytes) {
  static constexpr char kDigits[] = "0123456789abcdef";
  std::string out;
  for (std::uint8_t b : bytes) {
    out.push_back(kDigits[b >> 4]);
    out.push_back(kDigits[b & 15]);
  }
  return out;
}

// Exactly `colours` distinct luma values, spread over the whole block.
std::vector<std::uint8_t> make_pattern(int w, int h, int colours) {
  static constexpr std::uint8_t kValues[8] = {0, 32, 90, 160, 200, 230, 250,
                                               255};
  std::vector<std::uint8_t> y(
      static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0);
  const int area = w * h;
  for (int r = 0; r < h; ++r) {
    for (int c = 0; c < w; ++c) {
      int idx = 0;
      if (colours > 1) {
        idx = ((r * w + c) * colours) / area;
        if (idx >= colours) idx = colours - 1;
        if (idx > 7) idx = 7;
      }
      y[static_cast<std::size_t>(r) * static_cast<std::size_t>(w) +
        static_cast<std::size_t>(c)] = kValues[idx];
    }
  }
  return y;
}

// Vertical stripes: adjacent columns differ so direction 1 is cheaper.
std::vector<std::uint8_t> make_vertical_stripes(int w, int h, int stripes) {
  static constexpr std::uint8_t kValues[8] = {10, 50, 90, 130, 170, 210, 240,
                                               255};
  std::vector<std::uint8_t> y(
      static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0);
  for (int r = 0; r < h; ++r) {
    for (int c = 0; c < w; ++c) {
      const int idx = (c * stripes) / w;
      y[static_cast<std::size_t>(r) * static_cast<std::size_t>(w) +
        static_cast<std::size_t>(c)] = kValues[idx % 8];
    }
  }
  return y;
}

PaletteResult make_result(std::initializer_list<int> colors, int w, int h,
                          const std::vector<std::uint8_t>& map, int direction) {
  PaletteResult result;
  result.has_palette = true;
  result.size = static_cast<int>(colors.size());
  int i = 0;
  for (int colour : colors) {
    result.colors[static_cast<std::size_t>(i++)] =
        static_cast<std::uint8_t>(colour);
  }
  result.color_map = map;
  result.direction = direction;
  result.onscreen_width = w;
  result.onscreen_height = h;
  return result;
}

struct TestBlock {
  int mi_row = 0;
  int mi_col = 0;
  int bsize = 0;
  int block_w = 0;
  int block_h = 0;
  int visible_w = 0;
  int visible_h = 0;
  std::vector<std::uint8_t> src;
  bool expect_palette = true;
};

void compare_cdfs(const PaletteCdfState& a, const PaletteCdfState& b,
                  const std::string& name) {
  check(std::memcmp(a.y_mode, b.y_mode, sizeof(a.y_mode)) == 0,
        name + ": y_mode CDF parity");
  check(std::memcmp(a.y_size, b.y_size, sizeof(a.y_size)) == 0,
        name + ": y_size CDF parity");
  check(std::memcmp(a.color2, b.color2, sizeof(a.color2)) == 0,
        name + ": color2 CDF parity");
  check(std::memcmp(a.color3, b.color3, sizeof(a.color3)) == 0,
        name + ": color3 CDF parity");
  check(std::memcmp(a.color4, b.color4, sizeof(a.color4)) == 0,
        name + ": color4 CDF parity");
  check(std::memcmp(a.color5, b.color5, sizeof(a.color5)) == 0,
        name + ": color5 CDF parity");
  check(std::memcmp(a.color6, b.color6, sizeof(a.color6)) == 0,
        name + ": color6 CDF parity");
  check(std::memcmp(a.color7, b.color7, sizeof(a.color7)) == 0,
        name + ": color7 CDF parity");
  check(std::memcmp(a.color8, b.color8, sizeof(a.color8)) == 0,
        name + ": color8 CDF parity");
  check(std::memcmp(a.identity, b.identity, sizeof(a.identity)) == 0,
        name + ": identity CDF parity");
}

bool run_sequence(const ScreenContentFlags& flags, int mi_cols, int mi_rows,
                  const std::vector<TestBlock>& blocks, bool adaptive,
                  const std::string& name) {
  PaletteSyntaxWriter writer(flags);
  PaletteSyntaxReader reader(flags);
  std::string error;
  if (!writer.reset(mi_cols, mi_rows, error)) {
    check(false, name + ": writer reset: " + error);
    return false;
  }
  if (!reader.reset(mi_cols, mi_rows, error)) {
    check(false, name + ": reader reset: " + error);
    return false;
  }

  PaletteCdfState enc_cdfs;
  PaletteCdfState dec_cdfs;
  av2enc::palette::init_palette_cdfs(enc_cdfs);
  av2enc::palette::init_palette_cdfs(dec_cdfs);

  std::vector<std::uint8_t> buffer(1U << 20);
  av2::Encoder enc;
  check(enc.init_symbol(buffer.data(), buffer.size()) == av2::Status::kOk,
        name + ": encoder init");
  enc.disable_cdf_update = adaptive ? 0 : 1;

  std::vector<PaletteResult> analysed(blocks.size());
  std::vector<PaletteModeInfo> infos(blocks.size());
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    const TestBlock& block = blocks[i];
    std::uint8_t cache[16] = {};
    const int cache_n =
        writer.cache(block.mi_row, block.mi_col, block.mi_col > 0, cache);
    if (!av2enc::palette::analyze_palette(
            block.src.data(), block.block_w, block.block_w, block.block_h,
            block.visible_w, block.visible_h, cache_n > 0 ? cache : nullptr,
            cache_n, PaletteParams{}, analysed[i], error)) {
      check(false, name + ": analyze: " + error);
      return false;
    }
    check(analysed[i].has_palette == block.expect_palette,
          name + ": expected palette presence");
    const PaletteResult* payload = analysed[i].has_palette ? &analysed[i] : nullptr;
    if (!writer.write(enc, enc_cdfs, block.mi_row, block.mi_col, block.bsize,
                      block.block_w, block.block_h, block.visible_w,
                      block.visible_h, payload, 0, &infos[i], error)) {
      check(false, name + ": write: " + error);
      return false;
    }
  }
  std::size_t bytes = 0;
  check(enc.exit_symbol(&bytes) == av2::Status::kOk, name + ": exit_symbol");
  check(bytes > 0, name + ": tile payload is non-empty");

  av2::Decoder dec;
  check(dec.init_symbol(buffer.data(), buffer.size(), bytes) ==
            av2::Status::kOk,
        name + ": decoder init");
  dec.disable_cdf_update = adaptive ? 0 : 1;

  for (std::size_t i = 0; i < blocks.size(); ++i) {
    const TestBlock& block = blocks[i];
    std::uint8_t cache[16] = {};
    const int cache_n =
        reader.cache(block.mi_row, block.mi_col, block.mi_col > 0, cache);
    check(cache_n == infos[i].cache_len, name + ": cache length parity");
    PaletteResult decoded;
    PaletteModeInfo info;
    if (!reader.read(dec, dec_cdfs, block.mi_row, block.mi_col, block.bsize,
                     block.block_w, block.block_h, block.visible_w,
                     block.visible_h, 0, decoded, &info, error)) {
      check(false, name + ": read: " + error);
      return false;
    }
    check(decoded.has_palette == analysed[i].has_palette,
          name + ": decoded palette presence");
    if (analysed[i].has_palette) {
      check(decoded.size == analysed[i].size, name + ": decoded size");
      for (int k = 0; k < analysed[i].size; ++k) {
        check(decoded.colors[static_cast<std::size_t>(k)] ==
                  analysed[i].colors[static_cast<std::size_t>(k)],
              name + ": decoded colour");
      }
      check(decoded.direction == analysed[i].direction,
            name + ": decoded direction");
      check(decoded.color_map == analysed[i].color_map,
            name + ": decoded colour map");
      check(info.identity_rows == analysed[i].identity_rows,
            name + ": decoded identity rows");
      check(info.direction_signalled == infos[i].direction_signalled,
            name + ": direction-signalled parity");
      check(info.size == infos[i].size, name + ": mode-info size parity");
      check(info.cache_used == infos[i].cache_used,
            name + ": mode-info cache-use parity");
    }
  }
  check(dec.exit_symbol() == av2::Status::kOk, name + ": decoder trailing bits");
  compare_cdfs(enc_cdfs, dec_cdfs, name);
  check(writer.state().sizes == reader.state().sizes,
        name + ": per-MI palette sizes parity");
  check(writer.state().colors == reader.state().colors,
        name + ": per-MI palette colours parity");
  return true;
}

void test_screen_content_flags() {
  ScreenContentFlags flags = av2enc::still::palette_stream::palette_screen_content_flags();
  std::string error;
  check(av2enc::still::palette_stream::validate_screen_content_flags(flags, error),
        "palette screen flags valid");

  ScreenContentFlags bad = flags;
  bad.allow_screen_content_tools = false;
  check(!av2enc::still::palette_stream::validate_screen_content_flags(bad, error),
        "palette without screen tools rejected");

  ScreenContentFlags ibc = flags;
  ibc.allow_intrabc = true;
  ibc.allow_global_intrabc = false;
  ibc.allow_local_intrabc = false;
  check(!av2enc::still::palette_stream::validate_screen_content_flags(ibc, error),
        "IBC without a reference region rejected");

  // §5.18.3.3 bits: allow_screen_content_tools + force_integer_mv.
  {
    av2enc::BitWriter writer;
    const std::size_t bits =
        av2enc::still::palette_stream::write_frame_screen_content_params(writer,
                                                                     flags);
    check(bits == 2, "frame screen-content params are 2 bits");
    writer.align_to_byte();
    const auto bytes = writer.take_bytes();
    check(bytes.size() == 1 && bytes[0] == 0xC0,
          "frame screen-content params bits == 11");
  }
  {
    ScreenContentFlags off = flags;
    off.allow_screen_content_tools = false;
    off.allow_palette = false;
    off.force_integer_mv = false;
    av2enc::BitWriter writer;
    const std::size_t bits =
        av2enc::still::palette_stream::write_frame_screen_content_params(writer,
                                                                     off);
    check(bits == 1, "disabled frame screen-content params are 1 bit");
    writer.align_to_byte();
    const auto bytes = writer.take_bytes();
    check(bytes.size() == 1 && bytes[0] == 0x00,
          "disabled frame screen-content params bit == 0");
  }
  // §5.4.7: single_picture_header_flag infers both values -> zero bits.
  {
    av2enc::BitWriter writer;
    const std::size_t bits =
        av2enc::still::palette_stream::write_sequence_screen_content_params(
            writer, flags);
    check(bits == 0, "single-picture sequence screen-content params are absent");
    check(writer.bit_size() == 0, "single-picture sequence writes no bits");
  }
  // Non-single-picture form (documented for a future header writer).
  {
    ScreenContentFlags seq = flags;
    seq.single_picture_header = false;
    av2enc::BitWriter writer;
    const std::size_t bits =
        av2enc::still::palette_stream::write_sequence_screen_content_params(
            writer, seq);
    check(bits == 2, "non-single sequence choose bits");
    writer.align_to_byte();
    const auto bytes = writer.take_bytes();
    check(bytes.size() == 1 && bytes[0] == 0xC0,
          "non-single sequence choose screen/integer == 11");
  }
}

void test_palette_allowed() {
  const ScreenContentFlags flags =
      av2enc::still::palette_stream::palette_screen_content_flags();
  struct Case {
    int bsize;
    bool expected;
  };
  const Case cases[] = {
      {0, false},   // 4x4
      {1, false},   // 4x8
      {2, false},   // 8x4
      {3, true},    // 8x8
      {4, true},    // 8x16
      {5, true},    // 16x8
      {6, true},    // 16x16
      {9, true},    // 32x32
      {10, true},   // 32x64
      {11, true},   // 64x32
      {12, true},   // 64x64
      {13, false},  // 64x128: height > 64
      {14, false},  // 128x64: width > 64
      {19, true},   // 4x16: spec condition is MiSize >= BLOCK_8X8
      {20, true},   // 16x4
      {27, true},   // 8x64
      {28, true},   // 64x8
  };
  for (const Case& c : cases) {
    const auto size = static_cast<av2enc::partition::BlockSize>(c.bsize);
    const int w = av2enc::partition::block_width(size);
    const int h = av2enc::partition::block_height(size);
    check(av2enc::still::palette_stream::palette_allowed_for_block(flags, c.bsize,
                                                                w, h) ==
              c.expected,
          "palette allowed bsize " + std::to_string(c.bsize));
  }
  ScreenContentFlags no_palette = flags;
  no_palette.allow_palette = false;
  check(!av2enc::still::palette_stream::palette_allowed_for_block(no_palette, 12,
                                                               64, 64),
        "allow_palette=0 disables the tool");
  check(!av2enc::still::palette_stream::palette_allowed_for_block(flags, 12, 0,
                                                               64),
        "zero width rejected");
}

void test_cdf_defaults() {
  PaletteCdfState cdfs;
  av2enc::palette::init_palette_cdfs(cdfs);
  check(std::memcmp(cdfs.y_mode, av2enc::scc::kPaletteYModeCdf,
                    sizeof(cdfs.y_mode)) == 0,
        "Default_Palette_Y_Mode_Cdf pinned");
  check(std::memcmp(cdfs.y_size, av2enc::scc::kPaletteYSizeCdf,
                    sizeof(cdfs.y_size)) == 0,
        "Default_Palette_Y_Size_Cdf pinned");
  check(std::memcmp(cdfs.color2, av2enc::scc::kPaletteSize2YColorCdf,
                    sizeof(cdfs.color2)) == 0,
        "Default_Palette_Size_2_Y_Color_Cdf pinned");
  check(std::memcmp(cdfs.color3, av2enc::scc::kPaletteSize3YColorCdf,
                    sizeof(cdfs.color3)) == 0,
        "Default_Palette_Size_3_Y_Color_Cdf pinned");
  check(std::memcmp(cdfs.color4, av2enc::scc::kPaletteSize4YColorCdf,
                    sizeof(cdfs.color4)) == 0,
        "Default_Palette_Size_4_Y_Color_Cdf pinned");
  check(std::memcmp(cdfs.color5, av2enc::scc::kPaletteSize5YColorCdf,
                    sizeof(cdfs.color5)) == 0,
        "Default_Palette_Size_5_Y_Color_Cdf pinned");
  check(std::memcmp(cdfs.color6, av2enc::scc::kPaletteSize6YColorCdf,
                    sizeof(cdfs.color6)) == 0,
        "Default_Palette_Size_6_Y_Color_Cdf pinned");
  check(std::memcmp(cdfs.color7, av2enc::scc::kPaletteSize7YColorCdf,
                    sizeof(cdfs.color7)) == 0,
        "Default_Palette_Size_7_Y_Color_Cdf pinned");
  check(std::memcmp(cdfs.color8, av2enc::scc::kPaletteSize8YColorCdf,
                    sizeof(cdfs.color8)) == 0,
        "Default_Palette_Size_8_Y_Color_Cdf pinned");
  check(std::memcmp(cdfs.identity, av2enc::scc::kIdentityRowYCdf,
                    sizeof(cdfs.identity)) == 0,
        "Default_Identity_Row_Y_Cdf pinned");

  check(av2enc::still::palette_stream::y_mode_cdf(cdfs) == cdfs.y_mode,
        "y_mode_cdf points at the palette table");
  check(av2enc::still::palette_stream::y_size_cdf(cdfs) == cdfs.y_size,
        "y_size_cdf points at the palette table");
  for (int size = 2; size <= 8; ++size) {
    for (int ctx = 0; ctx < 5; ++ctx) {
      const std::uint16_t* cdf =
          av2enc::still::palette_stream::color_cdf(cdfs, size, ctx);
      check(cdf != nullptr, "color_cdf valid size/context");
      check(cdf != nullptr && cdf[0] < 32768, "color_cdf is addressable");
    }
  }
  check(av2enc::still::palette_stream::color_cdf(cdfs, 1, 0) == nullptr,
        "color_cdf rejects size 1");
  check(av2enc::still::palette_stream::color_cdf(cdfs, 8, 5) == nullptr,
        "color_cdf rejects context 5");
  for (int ctx = 0; ctx < 4; ++ctx) {
    check(av2enc::still::palette_stream::identity_cdf(cdfs, ctx) ==
              cdfs.identity[ctx],
          "identity_cdf context");
  }
  check(av2enc::still::palette_stream::identity_cdf(cdfs, 4) == nullptr,
        "identity_cdf rejects context 4");
  check(av2enc::still::palette_stream::palette_size_symbol(2) == 0 &&
            av2enc::still::palette_stream::palette_size_symbol(8) == 6,
        "palette_size_symbol maps 2..8 to 0..6");
}

void test_color_order() {
  // 4x4 map: left/above ranks colour 1 first.
  const std::vector<std::uint8_t> map = {
      0, 0, 0, 0,
      0, 1, 1, 0,
      0, 1, 1, 0,
      0, 0, 0, 0,
  };
  PaletteResult result = make_result({10, 20}, 4, 4, map, 0);
  av2enc::still::palette_stream::ColorOrderSample sample;
  check(av2enc::still::palette_stream::color_order_at(result, 4, 1, 1, sample),
        "color_order_at succeeds");
  check(sample.hash >= 0 && sample.hash <= 8, "color order hash in range");
  check(sample.context_index >= 0, "color order context maps to a CDF");
  check(sample.symbol >= 0 && sample.symbol < 2, "color order symbol in range");
  check(sample.order[static_cast<std::size_t>(sample.symbol)] == map[1 * 4 + 1],
        "color order symbol indexes the map colour");
  check(!av2enc::still::palette_stream::color_order_at(result, 4, 0, 9, sample),
        "color_order_at rejects out-of-range columns");
}

void test_round_trips() {
  const ScreenContentFlags flags =
      av2enc::still::palette_stream::palette_screen_content_flags();
  struct Size {
    int w;
    int h;
    int bsize;
    bool allowed;
  };
  const Size sizes[] = {
      {4, 4, 0, false},   {4, 8, 1, false},   {8, 4, 2, false},
      {8, 8, 3, true},    {8, 16, 4, true},   {16, 8, 5, true},
      {16, 16, 6, true},  {16, 32, 7, true},  {32, 16, 8, true},
      {32, 32, 9, true},  {32, 64, 10, true}, {64, 32, 11, true},
      {64, 64, 12, true}, {4, 16, 19, true},  {16, 4, 20, true},
      {8, 64, 27, true},  {64, 8, 28, true},
  };
  for (const Size& size : sizes) {
    if (!size.allowed) {
      // 4x4/4x8/8x4 are below BLOCK_8X8: the writer must refuse a palette.
      PaletteSyntaxWriter writer(flags);
      std::string error;
      check(writer.reset(16, 16, error), "reject-size writer reset");
      std::vector<std::uint8_t> buffer(1U << 12);
      av2::Encoder enc;
      enc.init_symbol(buffer.data(), buffer.size());
      enc.disable_cdf_update = 1;
      PaletteCdfState cdfs;
      av2enc::palette::init_palette_cdfs(cdfs);
      const auto src = make_pattern(size.w, size.h, 2);
      PaletteResult result;
      check(av2enc::palette::analyze_palette(src.data(), size.w, size.w,
                                             size.h, size.w, size.h, nullptr, 0,
                                             PaletteParams{}, result, error),
            "reject-size analyze");
      check(!writer.write(enc, cdfs, 0, 0, size.bsize, size.w, size.h, size.w,
                          size.h, &result, 0, nullptr, error),
            "palette rejected below BLOCK_8X8");
      continue;
    }
    for (int colours = 1; colours <= 8; ++colours) {
      TestBlock block;
      block.mi_row = 0;
      block.mi_col = 0;
      block.bsize = size.bsize;
      block.block_w = size.w;
      block.block_h = size.h;
      block.visible_w = size.w;
      block.visible_h = size.h;
      block.src = make_pattern(size.w, size.h, colours);
      const std::string name = std::to_string(size.w) + "x" +
                               std::to_string(size.h) + " c" +
                               std::to_string(colours);
      run_sequence(flags, 16, 16, {block}, false, name);
    }
  }
  // Adaptive CDFs must stay in sync for a short cache sequence.
  {
    TestBlock a;
    a.mi_row = 0;
    a.mi_col = 0;
    a.bsize = 9;
    a.block_w = 32;
    a.block_h = 32;
    a.visible_w = 32;
    a.visible_h = 32;
    a.src = make_pattern(32, 32, 3);
    TestBlock b = a;
    b.mi_row = 8;
    b.src = make_pattern(32, 32, 3);
    run_sequence(flags, 16, 16, {a, b}, true, "adaptive cache sequence");
  }
}

void test_partial_blocks() {
  const ScreenContentFlags flags =
      av2enc::still::palette_stream::palette_screen_content_flags();
  struct Partial {
    int w;
    int h;
    int bsize;
    int vw;
    int vh;
    int colours;
  };
  const Partial partials[] = {
      {16, 16, 6, 10, 6, 5},
      {32, 32, 9, 17, 9, 8},
      {64, 64, 12, 40, 20, 3},
      {64, 32, 11, 33, 17, 4},
  };
  for (const Partial& p : partials) {
    TestBlock block;
    block.bsize = p.bsize;
    block.block_w = p.w;
    block.block_h = p.h;
    block.visible_w = p.vw;
    block.visible_h = p.vh;
    block.src = make_pattern(p.w, p.h, p.colours);
    const std::string name = "partial " + std::to_string(p.vw) + "x" +
                             std::to_string(p.vh) + " in " +
                             std::to_string(p.w) + "x" + std::to_string(p.h);
    run_sequence(flags, 16, 16, {block}, false, name);
  }
}

void test_direction_and_identity() {
  const ScreenContentFlags flags =
      av2enc::still::palette_stream::palette_screen_content_flags();
  {
    TestBlock block;
    block.bsize = 6;
    block.block_w = 16;
    block.block_h = 16;
    block.visible_w = 16;
    block.visible_h = 16;
    block.src = make_vertical_stripes(16, 16, 2);
    PaletteResult result;
    std::string error;
    check(av2enc::palette::analyze_palette(
              block.src.data(), 16, 16, 16, 16, 16, nullptr, 0, PaletteParams{},
              result, error),
          "vertical stripes analyze");
    check(result.direction == 1, "vertical stripes pick column direction");
    run_sequence(flags, 16, 16, {block}, false, "vertical stripes");
  }
  {
    // A single flat colour makes every line identical: row 0 -> 1, rows 1.. ->
    // 2 (same as previous line).
    TestBlock block;
    block.bsize = 6;
    block.block_w = 16;
    block.block_h = 16;
    block.visible_w = 16;
    block.visible_h = 16;
    block.src.assign(16 * 16, 77);
    PaletteResult result;
    std::string error;
    check(av2enc::palette::analyze_palette(
              block.src.data(), 16, 16, 16, 16, 16, nullptr, 0, PaletteParams{},
              result, error),
          "flat block analyze");
    check(result.direction == 0, "flat block uses row direction");
    check(result.identity_rows.size() == 16, "flat block identity count");
    check(result.identity_rows[0] == 1, "first flat row is identity 1");
    for (std::size_t i = 1; i < result.identity_rows.size(); ++i) {
      check(result.identity_rows[i] == 2,
            "repeated flat row is identity 2");
    }
    run_sequence(flags, 16, 16, {block}, false, "flat identity rows");
  }
  {
    // 64x64 does not code palette_direction.
    TestBlock block;
    block.bsize = 12;
    block.block_w = 64;
    block.block_h = 64;
    block.visible_w = 64;
    block.visible_h = 64;
    block.src = make_vertical_stripes(64, 64, 2);
    PaletteResult result;
    std::string error;
    check(av2enc::palette::analyze_palette(
              block.src.data(), 64, 64, 64, 64, 64, nullptr, 0, PaletteParams{},
              result, error),
          "64x64 stripes analyze");
    check(result.direction == 0, "64x64 direction is pinned to rows");
    run_sequence(flags, 16, 16, {block}, false, "64x64 pinned direction");
  }
}

void test_cache_hits() {
  const ScreenContentFlags flags =
      av2enc::still::palette_stream::palette_screen_content_flags();
  TestBlock first;
  first.mi_row = 0;
  first.mi_col = 0;
  first.bsize = 9;
  first.block_w = 32;
  first.block_h = 32;
  first.visible_w = 32;
  first.visible_h = 32;
  first.src = make_pattern(32, 32, 3);
  TestBlock second = first;
  second.mi_row = 8;
  second.src = make_pattern(32, 32, 3);

  PaletteSyntaxWriter writer(flags);
  std::string error;
  check(writer.reset(16, 16, error), "cache writer reset");
  std::uint8_t cache[16] = {};
  int cache_n = writer.cache(0, 0, false, cache);
  check(cache_n == 0, "first block has an empty cache");
  PaletteResult first_result;
  check(av2enc::palette::analyze_palette(
            first.src.data(), 32, 32, 32, 32, 32, nullptr, 0, PaletteParams{},
            first_result, error),
        "cache first analyze");
  PaletteCdfState cdfs;
  av2enc::palette::init_palette_cdfs(cdfs);
  std::vector<std::uint8_t> buffer(1U << 16);
  av2::Encoder enc;
  enc.init_symbol(buffer.data(), buffer.size());
  enc.disable_cdf_update = 1;
  PaletteModeInfo first_info;
  check(writer.write(enc, cdfs, 0, 0, 9, 32, 32, 32, 32, &first_result, 0,
                     &first_info, error),
        "cache first write");
  cache_n = writer.cache(8, 0, false, cache);
  check(cache_n == first_result.size, "second block sees the above palette");
  PaletteResult second_result;
  check(av2enc::palette::analyze_palette(
            second.src.data(), 32, 32, 32, 32, 32, cache, cache_n,
            PaletteParams{}, second_result, error),
        "cache second analyze");
  check(second_result.cache_used == second_result.size,
        "second palette is fully cached");
  PaletteModeInfo second_info;
  check(writer.write(enc, cdfs, 8, 0, 9, 32, 32, 32, 32, &second_result, 0,
                     &second_info, error),
        "cache second write");
  check(second_info.cache_len == first_result.size,
        "cache length reported in mode info");
  check(second_info.cache_used == second_result.size,
        "cache use reported in mode info");
  std::size_t bytes = 0;
  check(enc.exit_symbol(&bytes) == av2::Status::kOk, "cache exit_symbol");

  av2::Decoder dec;
  dec.init_symbol(buffer.data(), buffer.size(), bytes);
  dec.disable_cdf_update = 1;
  PaletteSyntaxReader reader(flags);
  check(reader.reset(16, 16, error), "cache reader reset");
  PaletteCdfState dec_cdfs;
  av2enc::palette::init_palette_cdfs(dec_cdfs);
  PaletteResult decoded_first;
  PaletteResult decoded_second;
  check(reader.read(dec, dec_cdfs, 0, 0, 9, 32, 32, 32, 32, 0, decoded_first,
                    nullptr, error),
        "cache first read");
  check(reader.read(dec, dec_cdfs, 8, 0, 9, 32, 32, 32, 32, 0, decoded_second,
                    nullptr, error),
        "cache second read");
  check(decoded_second.color_map == second_result.color_map,
        "cache second map round-trips");
  check(writer.state().colors == reader.state().colors,
        "cache per-MI colours round-trip");
}

void test_reachability_and_luma_only() {
  const ScreenContentFlags flags =
      av2enc::still::palette_stream::palette_screen_content_flags();
  PaletteSyntaxWriter writer(flags);
  std::string error;
  check(writer.reset(16, 16, error), "reachability writer reset");
  std::vector<std::uint8_t> buffer(1U << 12);
  av2::Encoder enc;
  enc.init_symbol(buffer.data(), buffer.size());
  enc.disable_cdf_update = 1;
  PaletteCdfState cdfs;
  av2enc::palette::init_palette_cdfs(cdfs);

  const auto src = make_pattern(8, 8, 2);
  PaletteResult result;
  check(av2enc::palette::analyze_palette(src.data(), 8, 8, 8, 8, 8, nullptr, 0,
                                         PaletteParams{}, result, error),
        "reachability analyze");
  // Non-DC luma mode means palette_mode_info() is not reached at all.
  check(!writer.write(enc, cdfs, 0, 0, 3, 8, 8, 8, 8, &result, 1, nullptr,
                      error),
        "palette rejected when luma mode is not DC_PRED");
  check(writer.write(enc, cdfs, 0, 0, 3, 8, 8, 8, 8, nullptr, 1, nullptr,
                     error),
        "non-DC luma mode writes no palette symbol");
  std::size_t bytes = 0;
  check(enc.exit_symbol(&bytes) == av2::Status::kOk,
        "reachability exit_symbol");
  // A symbol-free §8.2 stream is the 15-bit trailing marker plus padding, so
  // compare against a control encoder that never wrote a symbol.
  std::vector<std::uint8_t> control_buffer(1U << 12);
  av2::Encoder control;
  control.init_symbol(control_buffer.data(), control_buffer.size());
  control.disable_cdf_update = 1;
  std::size_t control_bytes = 0;
  check(control.exit_symbol(&control_bytes) == av2::Status::kOk,
        "control exit_symbol");
  check(bytes == control_bytes, "non-DC block consumes no palette bits");

  // Luma-only: there is no U/V palette API or state.
  check(av2enc::palette::chroma_rule() ==
            av2enc::palette::ChromaRule::kIntraAfterLuma,
        "AV2 v2 palette is luma-only; chroma is intra");
}

void test_no_palette_symbol() {
  const ScreenContentFlags flags =
      av2enc::still::palette_stream::palette_screen_content_flags();
  PaletteSyntaxWriter writer(flags);
  PaletteSyntaxReader reader(flags);
  std::string error;
  check(writer.reset(16, 16, error), "no-palette writer reset");
  check(reader.reset(16, 16, error), "no-palette reader reset");
  PaletteCdfState enc_cdfs;
  PaletteCdfState dec_cdfs;
  av2enc::palette::init_palette_cdfs(enc_cdfs);
  av2enc::palette::init_palette_cdfs(dec_cdfs);
  std::vector<std::uint8_t> buffer(1U << 12);
  av2::Encoder enc;
  enc.init_symbol(buffer.data(), buffer.size());
  // Keep CDF adaptation on so the has_palette_y=0 symbol is observable.
  enc.disable_cdf_update = 0;
  PaletteModeInfo info;
  check(writer.write(enc, enc_cdfs, 0, 0, 3, 8, 8, 8, 8, nullptr, 0, &info,
                     error),
        "no-palette write succeeds");
  check(!info.has_palette_y, "no-palette mode info is clear");
  check(enc_cdfs.y_mode[2] == 1,
        "has_palette_y=0 consumes the palette Y-mode symbol");
  std::size_t bytes = 0;
  check(enc.exit_symbol(&bytes) == av2::Status::kOk, "no-palette exit_symbol");

  av2::Decoder dec;
  dec.init_symbol(buffer.data(), buffer.size(), bytes);
  dec.disable_cdf_update = 0;
  PaletteResult decoded;
  check(reader.read(dec, dec_cdfs, 0, 0, 3, 8, 8, 8, 8, 0, decoded, &info,
                    error),
        "no-palette read succeeds");
  check(!decoded.has_palette, "no-palette decoded result is clear");
  check(dec_cdfs.y_mode[2] == 1,
        "decoder consumes the has_palette_y=0 symbol");
  check(dec.exit_symbol() == av2::Status::kOk,
        "no-palette decoder trailing bits");
}

// ---------------------------------------------------------------------------
// Pinned golden bitstreams
// ---------------------------------------------------------------------------

struct Golden {
  const char* name;
  std::vector<std::uint8_t> bytes;
  const char* expected_hex;
};

struct PaletteStateSeed {
  int mi_row = 0;
  int mi_col = 0;
  int block_w = 0;
  int block_h = 0;
  PaletteResult result;
};

std::vector<std::uint8_t> encode_golden(const ScreenContentFlags& flags,
                                        PaletteResult& result, int mi_row,
                                        int mi_col, int bsize, int w, int h,
                                        PaletteStateSeed* seed) {
  PaletteSyntaxWriter writer(flags);
  std::string error;
  if (!writer.reset(16, 16, error)) return {};
  if (seed != nullptr) {
    av2enc::palette::store_palette_state(writer.mutable_state(), seed->mi_row,
                                        seed->mi_col, seed->block_w,
                                        seed->block_h, seed->result);
  }
  std::vector<std::uint8_t> buffer(1U << 16);
  av2::Encoder enc;
  if (enc.init_symbol(buffer.data(), buffer.size()) != av2::Status::kOk) {
    return {};
  }
  enc.disable_cdf_update = 1;
  PaletteCdfState cdfs;
  av2enc::palette::init_palette_cdfs(cdfs);
  if (!writer.write(enc, cdfs, mi_row, mi_col, bsize, w, h, w, h, &result, 0,
                    nullptr, error)) {
    std::fprintf(stderr, "golden encode failed: %s\n", error.c_str());
    return {};
  }
  std::size_t bytes = 0;
  if (enc.exit_symbol(&bytes) != av2::Status::kOk) return {};
  return std::vector<std::uint8_t>(
      buffer.begin(),
      buffer.begin() + static_cast<std::ptrdiff_t>(bytes));
}

void test_goldens() {
  const ScreenContentFlags flags =
      av2enc::still::palette_stream::palette_screen_content_flags();

  // G1: 8x8, 2 colours, row direction, no cache.
  std::vector<std::uint8_t> map1(64, 0);
  for (int r = 0; r < 8; ++r) {
    for (int c = 0; c < 8; ++c) {
      map1[static_cast<std::size_t>(r) * 8 + static_cast<std::size_t>(c)] =
          static_cast<std::uint8_t>(((r / 2 + c / 2) & 1));
    }
  }
  PaletteResult g1 = make_result({0, 200}, 8, 8, map1, 0);
  // G2: 16x16, 3 colours, column direction, vertical stripes.
  std::vector<std::uint8_t> map2(256, 0);
  for (int r = 0; r < 16; ++r) {
    for (int c = 0; c < 16; ++c) {
      map2[static_cast<std::size_t>(r) * 16 + static_cast<std::size_t>(c)] =
          static_cast<std::uint8_t>(c / 6);
    }
  }
  PaletteResult g2 = make_result({10, 90, 200}, 16, 16, map2, 1);
  // G3: 32x32, 3 colours with two colours reused from the above palette.
  PaletteResult above = make_result({40, 80}, 32, 32, std::vector<std::uint8_t>(1024, 0), 0);
  PaletteStateSeed seed;
  seed.mi_row = 0;
  seed.mi_col = 0;
  seed.block_w = 64;
  seed.block_h = 32;
  seed.result = above;
  std::vector<std::uint8_t> map3(1024, 0);
  for (int r = 0; r < 32; ++r) {
    for (int c = 0; c < 32; ++c) {
      map3[static_cast<std::size_t>(r) * 32 + static_cast<std::size_t>(c)] =
          static_cast<std::uint8_t>((r / 8) % 3);
    }
  }
  PaletteResult g3 = make_result({40, 80, 120}, 32, 32, map3, 0);

  const Golden goldens[] = {
      {"8x8/2 colours", encode_golden(flags, g1, 0, 0, 3, 8, 8, nullptr),
       "ea856975c1bd54b790deea0000"},
      {"16x16/3 colours columns", encode_golden(flags, g2, 0, 0, 6, 16, 16,
                                                nullptr),
       "f066f4b4dd468c0000"},
      {"32x32/3 colours cache", encode_golden(flags, g3, 8, 0, 9, 32, 32,
                                               &seed),
       "f3cf0f5e8009700000"},
  };
  for (const Golden& golden : goldens) {
    check(!golden.bytes.empty(), std::string(golden.name) + ": encodes");
    const std::string actual = hex(golden.bytes);
    if (actual != golden.expected_hex) {
      std::fprintf(stderr, "GOLDEN %s: actual=%s expected=%s\n", golden.name,
                   actual.c_str(), golden.expected_hex);
    }
    check(actual == golden.expected_hex,
          std::string(golden.name) + ": pinned bytes");
  }
}

// ---------------------------------------------------------------------------
// Minimal M1-compatible stream
// ---------------------------------------------------------------------------

MinimalPaletteLeaf make_leaf(int mi_row, int mi_col, int bsize, int w, int h,
                             int colours) {
  MinimalPaletteLeaf leaf;
  leaf.mi_row = mi_row;
  leaf.mi_col = mi_col;
  leaf.bsize = bsize;
  leaf.palette = PaletteResult{};
  const auto src = make_pattern(w, h, colours);
  std::string error;
  if (!av2enc::palette::analyze_palette(src.data(), w, w, h, w, h, nullptr, 0,
                                         PaletteParams{}, leaf.palette, error)) {
    std::fprintf(stderr, "minimal leaf analyze failed: %s\n", error.c_str());
  }
  return leaf;
}

void test_minimal_stream() {
  using av2enc::still::palette_stream::MinimalPaletteFramePlan;
  using av2enc::still::palette_stream::MinimalPaletteFrameResult;
  using av2enc::still::palette_stream::MinimalPaletteSuperblock;

  const ScreenContentFlags flags =
      av2enc::still::palette_stream::palette_screen_content_flags();

  {
    MinimalPaletteFramePlan plan;
    plan.flags = flags;
    plan.width = 64;
    plan.height = 64;
    MinimalPaletteSuperblock sb;
    sb.partition = MinimalPaletteSuperblock::Partition::kNone;
    sb.leaves.push_back(make_leaf(0, 0, 12, 64, 64, 4));
    plan.sbs.push_back(sb);
    MinimalPaletteFrameResult result;
    std::string error;
    check(av2enc::still::palette_stream::encode_minimal_palette_frame(plan, result,
                                                                   error),
          "minimal 64x64 palette stream encodes: " + error);
    check(result.tile_bytes > 0, "minimal stream tile bytes");
    check(result.transform_skip_eob0, "minimal stream is transform-skip");
    check(result.recon_y.size() == 64 * 64, "minimal stream recon size");
    // Palette prediction is exact on the synthetic screen content.
    const auto& leaf = plan.sbs[0].leaves[0];
    std::vector<std::uint8_t> pred(64 * 64);
    av2enc::palette::predict_palette_block(leaf.palette, 64, 64, pred.data(),
                                           64);
    check(std::memcmp(pred.data(), result.recon_y.data(), pred.size()) == 0,
          "minimal stream recon equals palette prediction");
  }
  {
    // HORZ: second 64x32 leaf sees the first leaf's palette in the cache.
    MinimalPaletteFramePlan plan;
    plan.flags = flags;
    plan.width = 64;
    plan.height = 64;
    MinimalPaletteSuperblock sb;
    sb.partition = MinimalPaletteSuperblock::Partition::kHorz;
    sb.leaves.push_back(make_leaf(0, 0, 11, 64, 32, 3));
    sb.leaves.push_back(make_leaf(8, 0, 11, 64, 32, 3));
    plan.sbs.push_back(sb);
    MinimalPaletteFrameResult result;
    std::string error;
    check(av2enc::still::palette_stream::encode_minimal_palette_frame(plan, result,
                                                                   error),
          "minimal HORZ palette stream encodes: " + error);
    check(result.tile_bytes > 0, "minimal HORZ tile bytes");
  }
  {
    // VERT: second 32x64 leaf sees the first leaf's palette on its left.
    MinimalPaletteFramePlan plan;
    plan.flags = flags;
    plan.width = 64;
    plan.height = 64;
    MinimalPaletteSuperblock sb;
    sb.partition = MinimalPaletteSuperblock::Partition::kVert;
    sb.leaves.push_back(make_leaf(0, 0, 10, 32, 64, 3));
    sb.leaves.push_back(make_leaf(0, 8, 10, 32, 64, 3));
    plan.sbs.push_back(sb);
    MinimalPaletteFrameResult result;
    std::string error;
    check(av2enc::still::palette_stream::encode_minimal_palette_frame(plan, result,
                                                                   error),
          "minimal VERT palette stream encodes: " + error);
    check(result.tile_bytes > 0, "minimal VERT tile bytes");
  }
  {
    // Two 64x64 superblocks: the second sees the first on its left.
    MinimalPaletteFramePlan plan;
    plan.flags = flags;
    plan.width = 128;
    plan.height = 64;
    for (int sb_col = 0; sb_col < 2; ++sb_col) {
      MinimalPaletteSuperblock sb;
      sb.partition = MinimalPaletteSuperblock::Partition::kNone;
      sb.leaves.push_back(make_leaf(0, sb_col * 16, 12, 64, 64, 3));
      plan.sbs.push_back(sb);
    }
    MinimalPaletteFrameResult result;
    std::string error;
    check(av2enc::still::palette_stream::encode_minimal_palette_frame(plan, result,
                                                                   error),
          "minimal 128x64 palette stream encodes: " + error);
    check(result.tile_bytes > 0, "minimal 128x64 tile bytes");
  }
  {
    // A 4x4 palette leaf is not signallable.
    MinimalPaletteFramePlan plan;
    plan.flags = flags;
    plan.width = 64;
    plan.height = 64;
    MinimalPaletteSuperblock sb;
    sb.partition = MinimalPaletteSuperblock::Partition::kNone;
    sb.leaves.push_back(make_leaf(0, 0, 0, 4, 4, 2));
    plan.sbs.push_back(sb);
    MinimalPaletteFrameResult result;
    std::string error;
    check(!av2enc::still::palette_stream::encode_minimal_palette_frame(
              plan, result, error),
          "4x4 minimal palette leaf is rejected");
  }
  {
    ScreenContentFlags off = flags;
    off.allow_palette = false;
    MinimalPaletteFramePlan plan;
    plan.flags = off;
    MinimalPaletteFrameResult result;
    std::string error;
    check(!av2enc::still::palette_stream::encode_minimal_palette_frame(
              plan, result, error),
          "minimal stream requires allow_palette");
  }
}

}  // namespace

int main() {
  test_screen_content_flags();
  test_palette_allowed();
  test_cdf_defaults();
  test_color_order();
  test_round_trips();
  test_partial_blocks();
  test_direction_and_identity();
  test_cache_hits();
  test_reachability_and_luma_only();
  test_no_palette_symbol();
  test_goldens();
  test_minimal_stream();
  std::printf("palette_stream_test: %d checks, %d failures\n", g_checks,
              g_failures);
  return g_failures == 0 ? 0 : 1;
}

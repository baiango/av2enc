// Palette intra CPU tests (§5.20.8, §7.13.4).
#include "codec/intra/palette/core/palette.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  }
}

using av2enc::palette::PaletteCdfState;
using av2enc::palette::PaletteParams;
using av2enc::palette::PaletteResult;
using av2enc::palette::PaletteState;

std::vector<std::uint8_t> make_pattern(int w, int h, int colours) {
  std::vector<std::uint8_t> y(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0);
  static constexpr std::uint8_t kColors[8] = {0, 32, 90, 160, 200, 230, 250,
                                              255};
  for (int r = 0; r < h; ++r) {
    for (int c = 0; c < w; ++c) {
      std::size_t p = static_cast<std::size_t>(r) * static_cast<std::size_t>(w) + static_cast<std::size_t>(c);
      if (colours <= 2) {
        y[p] = ((r / 4 + c / 4) & 1) ? 200 : 32;
      } else {
        const int band = (r * colours) / h;
        y[p] = kColors[band % 8];
        if (r > h / 2 && c > w / 2) y[p] = kColors[(band + 1) % colours];
      }
    }
  }
  return y;
}

bool round_trip(const std::vector<std::uint8_t>& src, int w, int h, int vw,
                int vh, const std::uint8_t* cache, int cache_n,
                bool use_cache, PaletteCdfState* cdfs,
                const std::string& name) {
  PaletteResult analysed;
  std::string error;
  if (!av2enc::palette::analyze_palette(src.data(), w, w, h, vw, vh, cache,
                                        cache_n, PaletteParams{}, analysed,
                                        error)) {
    check(false, name + ": analyze: " + error);
    return false;
  }
  check(analysed.has_palette, name + ": has_palette");
  check(analysed.size >= 2 && analysed.size <= 8, name + ": size range");
  for (int i = 1; i < analysed.size; ++i) {
    check(analysed.colors[static_cast<std::size_t>(i - 1)] <
              analysed.colors[static_cast<std::size_t>(i)],
          name + ": palette sorted and distinct");
  }

  std::vector<std::uint8_t> buffer(1U << 18);
  av2enc::av2::Encoder enc;
  check(enc.init_symbol(buffer.data(), buffer.size()) ==
            av2enc::av2::Status::kOk,
        name + ": encoder init");
  enc.disable_cdf_update = 1;
  PaletteCdfState enc_cdfs = *cdfs;
  if (!av2enc::palette::encode_palette_y(enc, enc_cdfs, analysed, w, h, vw, vh,
                                         use_cache ? cache : nullptr,
                                         use_cache ? cache_n : 0, error)) {
    check(false, name + ": encode: " + error);
    return false;
  }
  std::size_t bytes = 0;
  check(enc.exit_symbol(&bytes) == av2enc::av2::Status::kOk,
        name + ": exit_symbol");

  av2enc::av2::Decoder dec;
  check(dec.init_symbol(buffer.data(), buffer.size(), bytes) ==
            av2enc::av2::Status::kOk,
        name + ": decoder init");
  dec.disable_cdf_update = 1;
  PaletteResult decoded;
  PaletteCdfState dec_cdfs = *cdfs;
  if (!av2enc::palette::decode_palette_y(dec, dec_cdfs, w, h, vw, vh,
                                         use_cache ? cache : nullptr,
                                         use_cache ? cache_n : 0, decoded,
                                         error)) {
    check(false, name + ": decode: " + error);
    return false;
  }
  // CDF adaptation is disabled, so encoder and decoder CDF state must agree.
  check(std::memcmp(enc_cdfs.color2, dec_cdfs.color2, sizeof(enc_cdfs.color2)) ==
            0,
        name + ": CDF state parity");
  check(decoded.has_palette, name + ": decoded has_palette");
  check(decoded.size == analysed.size, name + ": decoded size");
  for (int i = 0; i < analysed.size; ++i) {
    check(decoded.colors[static_cast<std::size_t>(i)] ==
              analysed.colors[static_cast<std::size_t>(i)],
          name + ": decoded colour");
  }
  check(decoded.direction == analysed.direction, name + ": direction");
  check(decoded.color_map == analysed.color_map, name + ": colour map");
  check(dec.exit_symbol() == av2enc::av2::Status::kOk, name + ": trailing bits");

  // Prediction is exact for the analysed palette.
  std::vector<std::uint8_t> pred(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0);
  av2enc::palette::predict_palette_block(analysed, w, h, pred.data(), w);
  int diff = 0;
  for (int r = 0; r < vh; ++r) {
    for (int c = 0; c < vw; ++c) {
      if (pred[static_cast<std::size_t>(r) * static_cast<std::size_t>(w) + static_cast<std::size_t>(c)] !=
          src[static_cast<std::size_t>(r) * static_cast<std::size_t>(w) + static_cast<std::size_t>(c)]) {
        ++diff;
      }
    }
  }
  check(diff == 0, name + ": palette prediction is exact");
  return true;
}

void test_context_function() {
  // A 3x3 map: left=2, above-left=1, above=2 -> scores {0,1,4,...}.  Verify the
  // ranking and the hash mapping rather than a magic number.
  std::uint8_t map[9] = {0, 2, 2, 1, 2, 2, 0, 1, 1};
  std::uint8_t order[8];
  const int hash =
      av2enc::palette::palette_color_context(map, 3, 2, 2, 4, order);
  check(hash >= 0 && hash <= 8, "context hash in range");
  check(av2enc::palette::palette_color_context_index(hash) >= 0,
        "context hash maps to a CDF");
  check(order[0] == 2, "context ranks the most frequent neighbour first");
  // Inverse of ColorOrder.
  for (int i = 0; i < 4; ++i) {
    check(av2enc::palette::palette_symbol_for_color(order[i], order, 4) == i,
          "ColorOrder inverse");
  }
  // The colour-order tie-break prefers the left neighbour.
  std::uint8_t map2[4] = {0, 1, 1, 0};
  const int hash2 =
      av2enc::palette::palette_color_context(map2, 2, 1, 1, 2, order);
  check(order[0] == 1, "left/above tie-break");
  check(av2enc::palette::palette_color_context_index(hash2) >= 0,
        "tie-break hash valid");
}

void test_cache() {
  PaletteState state;
  state.reset(16, 16);  // 64x64 frame in MI units
  PaletteResult above;
  above.has_palette = true;
  above.size = 3;
  above.colors[0] = 10;
  above.colors[1] = 20;
  above.colors[2] = 30;
  PaletteResult left;
  left.has_palette = true;
  left.size = 2;
  left.colors[0] = 40;
  left.colors[1] = 50;
  av2enc::palette::store_palette_state(state, 2, 1, 64, 64, above);
  av2enc::palette::store_palette_state(state, 3, 0, 64, 64, left);
  std::uint8_t cache[16] = {};
  // Above row is used only when (mi_row * 4) % 64 != 0; row 3 is inside
  // a 64x64 SB (12 % 64 != 0).
  const int n = av2enc::palette::get_palette_cache(state, 3, 1, true, cache);
  check(n == 5, "cache length");
  check(cache[0] == 10 && cache[1] == 40 && cache[2] == 20 && cache[3] == 50 &&
            cache[4] == 30,
        "cache interleaves above/left");
  // At the top of a superblock the above palette is not used.
  PaletteResult left_top;
  left_top.has_palette = true;
  left_top.size = 2;
  left_top.colors[0] = 60;
  left_top.colors[1] = 70;
  av2enc::palette::store_palette_state(state, 0, 0, 64, 64, left_top);
  const int n_top =
      av2enc::palette::get_palette_cache(state, 0, 1, true, cache);
  check(n_top == 2 && cache[0] == 60 && cache[1] == 70,
        "cache skips above at the SB top row");
}

void test_direction_and_sizes() {
  PaletteCdfState cdfs;
  av2enc::palette::init_palette_cdfs(cdfs);
  const int sizes[][2] = {{8, 8},  {16, 16}, {32, 32}, {64, 64},
                          {64, 32}, {32, 64}, {8, 32},  {32, 8}};
  for (const auto& s : sizes) {
    const int w = s[0];
    const int h = s[1];
    const auto src = make_pattern(w, h, 4);
    round_trip(src, w, h, w, h, nullptr, 0, false, &cdfs,
               "block " + std::to_string(w) + "x" + std::to_string(h));
  }
  // Vertical stripes are cheaper column-by-column for a 16x16 block.
  std::vector<std::uint8_t> stripes(static_cast<std::size_t>(16 * 16));
  for (int r = 0; r < 16; ++r) {
    for (int c = 0; c < 16; ++c) {
      stripes[static_cast<std::size_t>(r) * 16 +
                static_cast<std::size_t>(c)] = (c & 1) ? 240 : 10;
    }
  }
  PaletteResult pr;
  std::string error;
  check(av2enc::palette::analyze_palette(stripes.data(), 16, 16, 16, 16, 16,
                                         nullptr, 0, PaletteParams{}, pr,
                                         error),
        "stripes analyze");
  check(pr.direction == 1, "vertical stripes choose column direction");
  // A 64x64 block cannot signal palette_direction.
  const auto big = make_pattern(64, 64, 4);
  check(av2enc::palette::analyze_palette(big.data(), 64, 64, 64, 64, 64,
                                         nullptr, 0, PaletteParams{}, pr,
                                         error),
        "64x64 analyze");
  check(pr.direction == 0, "64x64 direction pinned to rows");
}

void test_edge_and_partial() {
  PaletteCdfState cdfs;
  av2enc::palette::init_palette_cdfs(cdfs);
  const int w = 64;
  const int h = 64;
  auto src = make_pattern(w, h, 3);
  const int vw = 40;
  const int vh = 20;
  PaletteResult pr;
  std::string error;
  check(av2enc::palette::analyze_palette(src.data(), w, w, h, vw, vh, nullptr,
                                         0, PaletteParams{}, pr, error),
        "partial analyze");
  // Last onscreen column/row replicated exactly.
  for (int r = 0; r < vh; ++r) {
    for (int c = vw; c < w; ++c) {
      check(pr.color_map[static_cast<std::size_t>(r) * static_cast<std::size_t>(w) + static_cast<std::size_t>(c)] ==
                pr.color_map[static_cast<std::size_t>(r) * static_cast<std::size_t>(w) + static_cast<std::size_t>(vw - 1)],
            "partial column replication");
    }
  }
  for (int r = vh; r < h; ++r) {
    for (int c = 0; c < w; ++c) {
      check(pr.color_map[static_cast<std::size_t>(r) * static_cast<std::size_t>(w) + static_cast<std::size_t>(c)] ==
                pr.color_map[static_cast<std::size_t>(vh - 1) * static_cast<std::size_t>(w) + static_cast<std::size_t>(c)],
            "partial row replication");
    }
  }
  round_trip(src, w, h, vw, vh, nullptr, 0, false, &cdfs, "partial 64x64");
}

void test_delta_limit() {
  // A palette containing 126 and 255 exposes the CeilLog2(range) shrink; the
  // analysis must drop/merge rather than emit an unrepresentable delta.
  std::vector<std::uint8_t> y(static_cast<std::size_t>(64 * 64));
  for (int r = 0; r < 64; ++r) {
    for (int c = 0; c < 64; ++c) {
      y[static_cast<std::size_t>(r) * 64 +
            static_cast<std::size_t>(c)] =
          (r < 16) ? 0 : (r < 32) ? 126 : (r < 48) ? 200 : 255;
    }
  }
  PaletteResult pr;
  std::string error;
  const bool ok = av2enc::palette::analyze_palette(
      y.data(), 64, 64, 64, 64, 64, nullptr, 0, PaletteParams{}, pr, error);
  check(ok, "delta-limit analyze succeeds");
  check(pr.has_palette, "delta-limit keeps a palette");
  PaletteCdfState cdfs;
  av2enc::palette::init_palette_cdfs(cdfs);
  std::vector<std::uint8_t> buffer(1U << 16);
  av2enc::av2::Encoder enc;
  enc.init_symbol(buffer.data(), buffer.size());
  enc.disable_cdf_update = 1;
  const bool encoded = av2enc::palette::encode_palette_y(
      enc, cdfs, pr, 64, 64, 64, 64, nullptr, 0, error);
  check(encoded, "delta-limit encodes: " + error);
}

void test_chroma_rule() {
  check(av2enc::palette::chroma_rule() ==
            av2enc::palette::ChromaRule::kIntraAfterLuma,
        "AV2 v1.0.0 palette is luma-only; chroma uses normal intra");
  // The syntax itself has no U/V palette entries: a palette result carries one
  // colour array and one map only.
  PaletteResult pr;
  check(pr.colors.size() == 8, "single luma palette array");
}

void test_cache_round_trip() {
  PaletteCdfState cdfs;
  av2enc::palette::init_palette_cdfs(cdfs);
  PaletteState state;
  state.reset(16, 16);
  auto src = make_pattern(64, 64, 4);
  PaletteResult above;
  std::string error;
  check(av2enc::palette::analyze_palette(src.data(), 64, 32, 32, 32, 32,
                                         nullptr, 0, PaletteParams{}, above,
                                         error),
        "cache source analyze");
  // Store a 32x32 palette at MI (1,0) so the block at (0,2) sees it above.
  av2enc::palette::store_palette_state(state, 0, 2, 64, 32, above);
  std::uint8_t cache[16] = {};
  const int n =
      av2enc::palette::get_palette_cache(state, 8, 2, false, cache);
  check(n == above.size, "cache from above palette");
  const auto block = make_pattern(64, 64, 4);
  round_trip(block, 64, 64, 64, 64, cache, n, true, &cdfs, "cache round trip");
}

}  // namespace

int main() {
  test_context_function();
  test_cache();
  test_direction_and_sizes();
  test_edge_and_partial();
  test_delta_limit();
  test_chroma_rule();
  test_cache_round_trip();
  std::printf("palette_test: %d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}

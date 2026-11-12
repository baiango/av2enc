// AV2 palette intra — CPU reference and encoder-side analysis.
//
// Spec-only: AV2 v1.0.0 §5.20.8.1 (palette mode info), §5.20.8.2 (palette
// cache), §5.20.8.4 (palette tokens), §5.20.8.5 (colour context), §7.13.4
// (palette prediction), §8.3.2 (CDF selection).
//
// What this module owns
// ---------------------
//   * palette colour selection (exact for <=8 colours, median-cut + Lloyd for
//     photographic input), palette size 2..8, sorted ascending as the decoder
//     does after `sort(palette_colors_y, 0, PaletteSizeY-1)`;
//   * the above/left palette cache (`get_palette_cache`) and cache reuse
//     decisions;
//   * colour-map generation and the normative color-index contexts;
//   * colour-map syntax round-trip (encoder + local decoder) with the real
//     §8.2 range coder and §9.3 palette CDFs;
//   * palette prediction and the residual interface (the palette itself is a
//     prediction; AV2 always codes a residual afterwards, so "transform-skip"
//     is the eob=0 case, not a separate syntax element);
//   * luma/chroma rules and edge/partial-block replication.
//
// Luma-only rule (verified against the v1.0.0 syntax)
// --------------------------------------------------
// `palette_mode_info()` is invoked for `PlaneStart == 0` and has no U/V
// branches.  Chroma therefore uses the normal intra path (DC/CfL/directional)
// after the luma palette block is reconstructed.  `chroma_rule()` records this
// and the tests assert that no chroma palette syntax exists.
//
// No bitstream assembly here: see `scc_stream.h` (shared with the IBC module).
// No AVM/SVT/clave code; the spec is the only reference.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "codec/entropy/av2/av2_entropy.h"
#include "codec/intra/palette/core/palette_core.h"

namespace av2enc::palette {

// ABGR / luma-only palette entry values are 8-bit here (8-bit 4:2:0 scope).
struct Rect {
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
};

struct PaletteParams {
  // Maximum palette size.  §5.20.8.1 codes PaletteSizeY = 2..8.
  int max_colors = 8;
  int min_colors = 2;
  // If the block has <= max_colors distinct values, use them exactly.  This is
  // the screen-content case (flat UI fills, text, icons).
  bool exact_when_possible = true;
  // Ordered 4x4 dithering when the block needs quantisation.  Off by default:
  // it raises PSNR at equal palette size but makes the map less compressible.
  bool dither = false;
  // Choose the palette direction / identity rows that minimise the exact
  // colour-map symbol cost.
  bool choose_direction = true;
  // Reject a palette that does not beat this many estimated bits / pixel of
  // the DC+residual proxy. 0 disables the guard.
  double max_bits_per_pixel = 0.0;
};

struct PaletteCdfState {
  std::uint16_t y_mode[3];
  std::uint16_t y_size[8];
  std::uint16_t color2[5][3];
  std::uint16_t color3[5][4];
  std::uint16_t color4[5][5];
  std::uint16_t color5[5][6];
  std::uint16_t color6[5][7];
  std::uint16_t color7[5][8];
  std::uint16_t color8[5][9];
  std::uint16_t identity[4][4];
};

void init_palette_cdfs(PaletteCdfState& cdfs) noexcept;

// Frame-level per-4x4 palette state.  §5.20.8.2 reads the *neighbouring MI's*
// palette size/colours, not the current block's, so the state is stored at MI
// granularity exactly as the decoder does.
struct PaletteState {
  int mi_cols = 0;
  int mi_rows = 0;
  std::vector<std::uint8_t> sizes;
  std::vector<std::array<std::uint8_t, kPaletteMaxSize>> colors;

  void reset(int cols, int rows);
  bool valid() const { return mi_cols > 0 && mi_rows > 0; }
};

// §5.20.8.2 get_palette_cache: interleaves above and left palette colours.
// `avail_l` is the block's AvailL.  Returns the number of cache entries and
// writes them to `cache` (capacity >= 16).
int get_palette_cache(const PaletteState& state, int mi_row, int mi_col,
                      bool avail_l, std::uint8_t cache[16]) noexcept;

struct PaletteResult {
  bool has_palette = false;
  int size = 0;                       // 2..8 when has_palette
  std::array<std::uint8_t, kPaletteMaxSize> colors{};
  std::vector<std::uint8_t> color_map;  // w*h, index into colors
  int direction = 0;                    // 0 = rows, 1 = columns
  std::vector<std::uint8_t> identity_rows;  // one per outer line
  int distinct_colors = 0;
  std::int64_t sse = 0;               // sum of squared error vs source
  int map_bits = 0;                   // exact-ish colour-map symbol cost
  int mode_bits = 0;                  // size/colour/cache cost
  int cache_used = 0;                 // number of cache entries reused
  int onscreen_width = 0;
  int onscreen_height = 0;
};

// Palette analysis for one luma block.
//
// `src` is the visible source luma plane; `stride` is its row stride.
// `cache`/`cache_n` come from get_palette_cache() for this block.
// `block_w`/`block_h` are the *coded* block dimensions;
// `visible_w`/`visible_h` clamp partial edge blocks.  Samples outside the
// visible area are never read; the colour map is replicated to the full block
// exactly as §5.20.8.4 does after the token loop.
bool analyze_palette(const std::uint8_t* src, int stride, int block_w,
                     int block_h, int visible_w, int visible_h,
                     const std::uint8_t* cache, int cache_n,
                     const PaletteParams& params, PaletteResult& out,
                     std::string& error);

// Store the chosen palette into frame state for every MI the block covers.
void store_palette_state(PaletteState& state, int mi_row, int mi_col,
                         int block_w, int block_h,
                         const PaletteResult& result) noexcept;

// §7.13.4 prediction from the colour map (full block resolution for
// TX_MODE_LARGEST).
void predict_palette_block(const PaletteResult& result, int block_w,
                           int block_h, std::uint8_t* dst, int dst_stride);

// Residual after palette prediction.  For an exact palette this is all zero;
// the caller may transform it (IDTX/transform-skip is simply eob=0 when the
// block is represented by the palette alone).
void palette_residual(const std::uint8_t* src, int src_stride,
                      const PaletteResult& result, int block_w, int block_h,
                      std::int16_t* residual, int residual_stride);

// ---------------------------------------------------------------------------
// Palette bitstream syntax (exact §8.2 coding)
// ---------------------------------------------------------------------------

// Write §5.20.8.1 (`has_palette_y` + size + cache/literal/delta colours) and
// §5.20.8.4 (`palette_direction`, identity rows, colour indices).
//
// `w`/`h` are the coded block dimensions; the token loop clamps to
// `visible_w`/`visible_h` per the onscreen formulas.  The encoder mutates
// `cdfs` exactly like the decoder.
bool encode_palette_y(av2::Encoder& enc, PaletteCdfState& cdfs,
                      const PaletteResult& result, int w, int h, int visible_w,
                      int visible_h, const std::uint8_t* cache, int cache_n,
                      std::string& error);

// Local decoder used by the tests (and to self-check the encoder before the
// dav2d gate).  `decode_palette_y` consumes the same symbols and rebuilds the
// colour map; the caller compares it with the analysis result.
bool decode_palette_y(av2::Decoder& dec, PaletteCdfState& cdfs, int w, int h,
                      int visible_w, int visible_h, const std::uint8_t* cache,
                      int cache_n, PaletteResult& out, std::string& error);

// Estimated palette syntax cost in bits, computed against copies of the CDFs
// with adaptation disabled.  Does not mutate `cdfs`.
int estimate_palette_bits(const PaletteResult& result, int w, int h,
                          int visible_w, int visible_h,
                          const std::uint8_t* cache, int cache_n,
                          const PaletteCdfState& cdfs);

// ---------------------------------------------------------------------------
// Luma/chroma rules and edge handling
// ---------------------------------------------------------------------------
enum class ChromaRule {
  // AV2 v1.0.0: no chroma palette.  Chroma is intra-predicted normally.
  kIntraAfterLuma,
};

ChromaRule chroma_rule() noexcept;

// §5.20.8.4 replication after the token loop: columns beyond onscreenWidth
// copy the last onscreen column, rows beyond onscreenHeight copy the last
// onscreen row.  Exposed for tests of partial/edge blocks.
void replicate_edge_color_map(std::uint8_t* map, int stride, int block_w,
                              int block_h, int onscreen_w, int onscreen_h);

// ---------------------------------------------------------------------------
// Small helpers shared with tests / the CUDA module.
// ---------------------------------------------------------------------------
int palette_size_index(int size) noexcept;  // PaletteSizeY -> 0..6

// Collect distinct values in the block (visible area only).
std::vector<std::uint8_t> block_histogram(const std::uint8_t* src, int stride,
                                          int w, int h);

}  // namespace av2enc::palette

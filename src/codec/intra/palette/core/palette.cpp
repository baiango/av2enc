#include "codec/intra/palette/core/palette.h"

#include "codec/intra/palette/core/scc_cdf.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>

namespace av2enc::palette {
namespace {

template <typename T, std::size_t N>
void copy_array(T (&dst)[N], const T (&src)[N]) {
  std::memcpy(dst, src, sizeof(dst));
}

std::uint16_t* color_cdf(PaletteCdfState& cdfs, int size) {
  switch (size) {
    case 2:
      return &cdfs.color2[0][0];
    case 3:
      return &cdfs.color3[0][0];
    case 4:
      return &cdfs.color4[0][0];
    case 5:
      return &cdfs.color5[0][0];
    case 6:
      return &cdfs.color6[0][0];
    case 7:
      return &cdfs.color7[0][0];
    default:
      return &cdfs.color8[0][0];
  }
}

// §8.2.5 inverse: n raw bits, most significant first.
bool encode_literal(av2::Encoder& enc, int bits, std::uint32_t value) {
  for (int i = bits - 1; i >= 0; --i) {
    if (enc.encode_bool(((value >> i) & 1U) != 0U) != av2::Status::kOk) {
      return false;
    }
  }
  return true;
}

// Median-cut palette for blocks with more than 8 distinct values.  Deterministic
// and dependency-free; the exact-palette path is what screen content uses.
void median_cut(const std::uint8_t* src, int stride, int w, int h, int max_colors,
                std::array<std::uint8_t, kPaletteMaxSize>& colors, int& size) {
  std::vector<std::uint8_t> values;
  values.reserve(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      values.push_back(src[static_cast<std::size_t>(y) *
                               static_cast<std::size_t>(stride) +
                           static_cast<std::size_t>(x)]);
    }
  }
  if (values.empty()) {
    size = 0;
    return;
  }
  struct Box {
    std::size_t begin = 0;
    std::size_t end = 0;
    int lo = 0;
    int hi = 0;
  };
  std::sort(values.begin(), values.end());
  std::vector<Box> boxes;
  boxes.push_back(Box{0, values.size(), values.front(), values.back()});
  while (static_cast<int>(boxes.size()) < max_colors) {
    // Split the box with the largest range; stable tie-break by index.
    std::size_t pick = boxes.size();
    int best_range = -1;
    for (std::size_t i = 0; i < boxes.size(); ++i) {
      const int range = boxes[i].hi - boxes[i].lo;
      if (boxes[i].end - boxes[i].begin > 1 && range > best_range) {
        best_range = range;
        pick = i;
      }
    }
    if (pick == boxes.size()) break;
    Box& box = boxes[pick];
    const std::size_t mid = (box.begin + box.end) / 2;
    Box left{box.begin, mid, values[box.begin], values[mid - 1]};
    Box right{mid, box.end, values[mid], values[box.end - 1]};
    box = left;
    boxes.push_back(right);
  }
  size = static_cast<int>(boxes.size());
  for (int i = 0; i < size; ++i) {
    const Box& box = boxes[static_cast<std::size_t>(i)];
    std::int64_t sum = 0;
    for (std::size_t k = box.begin; k < box.end; ++k) {
      sum += values[k];
    }
    const std::int64_t count =
        static_cast<std::int64_t>(box.end - box.begin);
    colors[static_cast<std::size_t>(i)] =
        static_cast<std::uint8_t>((sum + count / 2) / count);
  }
  std::sort(colors.begin(), colors.begin() + size);
}

// Fill the palette to at least min_colors with adjacent unused values so the
// syntax always codes a legal 2..8 entry palette.  Duplicates are disallowed
// because §5.20.8.1 delta codes a strictly increasing run.
void fill_palette(std::array<std::uint8_t, kPaletteMaxSize>& colors, int& size,
                  int min_colors) {
  while (size < min_colors) {
    bool added = false;
    for (int delta = 1; delta <= 255 && !added; ++delta) {
      for (int dir = -1; dir <= 1 && !added; dir += 2) {
        const int candidate = colors[0] + dir * delta;
        if (candidate < 0 || candidate > 255) continue;
        bool used = false;
        for (int i = 0; i < size; ++i) {
          if (colors[static_cast<std::size_t>(i)] == candidate) used = true;
        }
        if (used) continue;
        colors[static_cast<std::size_t>(size)] =
            static_cast<std::uint8_t>(candidate);
        ++size;
        added = true;
      }
    }
    if (!added) break;
    std::sort(colors.begin(), colors.begin() + size);
  }
}

// Simulate §5.20.8.1's delta loop.  `remaining` is the sorted run that must be
// coded after the cache slots; returns the smallest extra-bit count 0..3 that
// makes the run representable, or -1.
int palette_extra_bits_for(const std::uint8_t* remaining, int count) {
  if (count <= 1) return 0;
  for (int extra = 0; extra <= 3; ++extra) {
    int palette_bits = 5 + extra;  // minBits = BitDepth - 3 = 5
    bool ok = true;
    for (int i = 1; i < count; ++i) {
      const int delta = static_cast<int>(remaining[i]) -
                        static_cast<int>(remaining[i - 1]) - 1;
      if (delta < 0 || delta >= (1 << palette_bits)) {
        ok = false;
        break;
      }
      const int range = 255 - static_cast<int>(remaining[i]);
      palette_bits = std::min(palette_bits, palette_ceil_log2(range));
    }
    if (ok) return extra;
  }
  return -1;
}

// Build the cache subset and the remaining sorted run for the chosen palette.
// `used_cache` receives a bit per cache entry (1 = reuse).
int split_cache_and_run(const std::uint8_t* colors, int size,
                        const std::uint8_t* cache, int cache_n,
                        std::uint8_t* used_cache, std::uint8_t* remaining) {
  int idx = 0;
  for (int i = 0; i < cache_n && idx < size; ++i) {
    used_cache[i] = 0;
    for (int c = idx; c < size; ++c) {
      if (colors[c] == cache[i]) {
        used_cache[i] = 1;
        ++idx;
        break;
      }
    }
  }
  int run = 0;
  for (int c = 0; c < size; ++c) {
    bool cached = false;
    for (int i = 0; i < cache_n; ++i) {
      if (used_cache[i] && cache[i] == colors[c]) cached = true;
    }
    if (!cached) remaining[run++] = colors[c];
  }
  return run;
}

}  // namespace

void init_palette_cdfs(PaletteCdfState& cdfs) noexcept {
  copy_array(cdfs.y_mode, scc::kPaletteYModeCdf);
  copy_array(cdfs.y_size, scc::kPaletteYSizeCdf);
  copy_array(cdfs.color2, scc::kPaletteSize2YColorCdf);
  copy_array(cdfs.color3, scc::kPaletteSize3YColorCdf);
  copy_array(cdfs.color4, scc::kPaletteSize4YColorCdf);
  copy_array(cdfs.color5, scc::kPaletteSize5YColorCdf);
  copy_array(cdfs.color6, scc::kPaletteSize6YColorCdf);
  copy_array(cdfs.color7, scc::kPaletteSize7YColorCdf);
  copy_array(cdfs.color8, scc::kPaletteSize8YColorCdf);
  copy_array(cdfs.identity, scc::kIdentityRowYCdf);
}

void PaletteState::reset(int cols, int rows) {
  mi_cols = cols > 0 ? cols : 0;
  mi_rows = rows > 0 ? rows : 0;
  const std::size_t count = static_cast<std::size_t>(mi_cols) *
                            static_cast<std::size_t>(mi_rows);
  sizes.assign(count, 0);
  colors.assign(count, std::array<std::uint8_t, kPaletteMaxSize>{});
}

int get_palette_cache(const PaletteState& state, int mi_row, int mi_col,
                      bool avail_l, std::uint8_t cache[16]) noexcept {
  if (!state.valid() || mi_row < 0 || mi_col < 0 || mi_row >= state.mi_rows ||
      mi_col >= state.mi_cols) {
    return 0;
  }
  int above_n = 0;
  if ((mi_row * 4) % 64 != 0 && mi_row > 0) {
    above_n = state.sizes[static_cast<std::size_t>(mi_row - 1) *
                              static_cast<std::size_t>(state.mi_cols) +
                          static_cast<std::size_t>(mi_col)];
  }
  int left_n = 0;
  if (avail_l && mi_col > 0) {
    left_n = state.sizes[static_cast<std::size_t>(mi_row) *
                             static_cast<std::size_t>(state.mi_cols) +
                         static_cast<std::size_t>(mi_col - 1)];
  }
  int above_idx = 0;
  int left_idx = 0;
  int n = 0;
  while (above_idx < above_n || left_idx < left_n) {
    if (above_idx < above_n) {
      const auto& entry =
          state.colors[static_cast<std::size_t>(mi_row - 1) *
                           static_cast<std::size_t>(state.mi_cols) +
                       static_cast<std::size_t>(mi_col)];
      cache[n++] = entry[static_cast<std::size_t>(above_idx++)];
    }
    if (left_idx < left_n) {
      const auto& entry =
          state.colors[static_cast<std::size_t>(mi_row) *
                           static_cast<std::size_t>(state.mi_cols) +
                       static_cast<std::size_t>(mi_col - 1)];
      cache[n++] = entry[static_cast<std::size_t>(left_idx++)];
    }
  }
  return n;
}

std::vector<std::uint8_t> block_histogram(const std::uint8_t* src, int stride,
                                          int w, int h) {
  std::array<bool, 256> seen{};
  std::vector<std::uint8_t> values;
  if (src == nullptr || w <= 0 || h <= 0) return values;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const std::uint8_t v = src[static_cast<std::size_t>(y) *
                                     static_cast<std::size_t>(stride) +
                                 static_cast<std::size_t>(x)];
      if (!seen[v]) {
        seen[v] = true;
        values.push_back(v);
      }
    }
  }
  std::sort(values.begin(), values.end());
  return values;
}

void replicate_edge_color_map(std::uint8_t* map, int stride, int block_w,
                              int block_h, int onscreen_w, int onscreen_h) {
  if (map == nullptr || onscreen_w <= 0 || onscreen_h <= 0) return;
  // §5.20.8.4: for each onscreen row, copy the last onscreen column right.
  for (int i = 0; i < onscreen_h; ++i) {
    const std::uint8_t last = map[static_cast<std::size_t>(i) *
                                      static_cast<std::size_t>(stride) +
                                  static_cast<std::size_t>(onscreen_w - 1)];
    for (int j = onscreen_w; j < block_w; ++j) {
      map[static_cast<std::size_t>(i) * static_cast<std::size_t>(stride) +
          static_cast<std::size_t>(j)] = last;
    }
  }
  // Then copy the last onscreen row down over the full block width.
  for (int i = onscreen_h; i < block_h; ++i) {
    for (int j = 0; j < block_w; ++j) {
      map[static_cast<std::size_t>(i) * static_cast<std::size_t>(stride) +
          static_cast<std::size_t>(j)] =
          map[static_cast<std::size_t>(onscreen_h - 1) *
                  static_cast<std::size_t>(stride) +
              static_cast<std::size_t>(j)];
    }
  }
}

bool analyze_palette(const std::uint8_t* src, int stride, int block_w,
                     int block_h, int visible_w, int visible_h,
                     const std::uint8_t* cache, int cache_n,
                     const PaletteParams& params, PaletteResult& out,
                     std::string& error) {
  out = PaletteResult{};
  error.clear();
  if (src == nullptr || block_w <= 0 || block_h <= 0 || visible_w <= 0 ||
      visible_h <= 0 || visible_w > block_w || visible_h > block_h) {
    error = "invalid palette block geometry";
    return false;
  }
  if (params.max_colors < 2 || params.max_colors > kPaletteMaxSize) {
    error = "palette size must be 2..8";
    return false;
  }
  if (cache_n < 0 || cache_n > 16) {
    error = "invalid palette cache length";
    return false;
  }
  out.onscreen_width = visible_w;
  out.onscreen_height = visible_h;

  const std::vector<std::uint8_t> distinct =
      block_histogram(src, stride, visible_w, visible_h);
  out.distinct_colors = static_cast<int>(distinct.size());

  std::array<std::uint8_t, kPaletteMaxSize> colors{};
  int size = 0;
  if (params.exact_when_possible &&
      static_cast<int>(distinct.size()) <= params.max_colors) {
    size = static_cast<int>(distinct.size());
    for (int i = 0; i < size; ++i) {
      colors[static_cast<std::size_t>(i)] = distinct[static_cast<std::size_t>(i)];
    }
    fill_palette(colors, size, std::max(2, params.min_colors));
  } else {
    median_cut(src, stride, visible_w, visible_h, params.max_colors, colors,
               size);
    if (size < 2) {
      fill_palette(colors, size, 2);
    }
  }
  if (size < 2 || size > kPaletteMaxSize) {
    error = "palette analysis produced an illegal size";
    return false;
  }
  std::sort(colors.begin(), colors.begin() + size);

  // §5.20.8.1 delta codes palette_colors_y as a strictly increasing run, and
  // median_cut can return the same cluster value twice (two boxes with equal
  // means).  Collapse duplicates before the representability check; the cache
  // split below is value-based and would otherwise hide the duplicate.
  {
    int unique = 0;
    for (int i = 0; i < size; ++i) {
      if (unique == 0 || colors[static_cast<std::size_t>(unique - 1)] !=
                             colors[static_cast<std::size_t>(i)]) {
        colors[static_cast<std::size_t>(unique++)] =
            colors[static_cast<std::size_t>(i)];
      }
    }
    size = unique;
    if (size < 2) {
      fill_palette(colors, size, 2);
      std::sort(colors.begin(), colors.begin() + size);
    }
  }

  // The luma palette must be delta-representable.  If the chosen set is not,
  // merge the nearest colours until it is (screen content is normally exact).
  for (;;) {
    std::uint8_t used_cache[16] = {};
    std::uint8_t remaining[kPaletteMaxSize] = {};
    const int run =
        split_cache_and_run(colors.data(), size, cache, cache_n, used_cache,
                            remaining);
    const int extra = palette_extra_bits_for(remaining, run);
    if (extra >= 0) break;
    if (size <= 2) {
      error = "palette is not delta-representable";
      return false;
    }
    // Drop the entry that contributes least (smallest neighbour gap).
    int drop = 1;
    int smallest_gap = 256;
    for (int i = 0; i < size; ++i) {
      const int lo = i == 0 ? -1 : colors[static_cast<std::size_t>(i - 1)];
      const int hi =
          i == size - 1 ? 256 : colors[static_cast<std::size_t>(i + 1)];
      const int gap = hi - lo;
      if (gap < smallest_gap) {
        smallest_gap = gap;
        drop = i;
      }
    }
    for (int i = drop; i + 1 < size; ++i) {
      colors[static_cast<std::size_t>(i)] =
          colors[static_cast<std::size_t>(i + 1)];
    }
    --size;
  }

  out.size = size;
  out.has_palette = true;
  for (int i = 0; i < size; ++i) {
    out.colors[static_cast<std::size_t>(i)] = colors[static_cast<std::size_t>(i)];
  }

  // Build the colour map for the visible area, then replicate to the block.
  out.color_map.assign(static_cast<std::size_t>(block_w) *
                           static_cast<std::size_t>(block_h),
                       0);
  for (int y = 0; y < visible_h; ++y) {
    for (int x = 0; x < visible_w; ++x) {
      int value = src[static_cast<std::size_t>(y) *
                          static_cast<std::size_t>(stride) +
                      static_cast<std::size_t>(x)];
      if (params.dither) {
        static constexpr int kDither[4][4] = {
            {0, 8, 2, 10}, {12, 4, 14, 6}, {3, 11, 1, 9}, {15, 7, 13, 5}};
        value = palette_clip1(value + (kDither[y & 3][x & 3] - 8) / 8);
      }
      const int idx = palette_nearest_color(out.colors.data(), size, value);
      out.color_map[static_cast<std::size_t>(y) *
                        static_cast<std::size_t>(block_w) +
                    static_cast<std::size_t>(x)] =
          static_cast<std::uint8_t>(idx);
    }
  }
  replicate_edge_color_map(out.color_map.data(), block_w, block_w, block_h,
                           visible_w, visible_h);

  // Choose direction / identity rows by exact syntax cost.  A 64-wide or
  // 64-tall block does not code palette_direction, so it is pinned to rows.
  const bool can_signal_direction = block_w < 64 && block_h < 64;
  if (!can_signal_direction) {
    out.direction = 0;
    out.identity_rows.assign(static_cast<std::size_t>(visible_h), 0);
    for (int i = 0; i < visible_h; ++i) {
      out.identity_rows[static_cast<std::size_t>(i)] =
          static_cast<std::uint8_t>(palette_classify_line(
              out.color_map.data(), block_w, 0, i, visible_w, i > 0, size));
    }
  } else {
    std::int64_t best_bits = std::numeric_limits<std::int64_t>::max();
    for (int direction = 0; direction < 2; ++direction) {
      const int outer = direction ? visible_w : visible_h;
      const int inner = direction ? visible_h : visible_w;
      PaletteResult candidate = out;
      candidate.direction = direction;
      candidate.identity_rows.assign(static_cast<std::size_t>(outer), 0);
      for (int i = 0; i < outer; ++i) {
        candidate.identity_rows[static_cast<std::size_t>(i)] =
            static_cast<std::uint8_t>(palette_classify_line(
                candidate.color_map.data(), block_w, direction, i, inner,
                i > 0, size));
      }
      const int bits = estimate_palette_bits(candidate, block_w, block_h,
                                             visible_w, visible_h, cache,
                                             cache_n, PaletteCdfState{});
      if (bits < best_bits) {
        best_bits = bits;
        out.direction = direction;
        out.identity_rows = candidate.identity_rows;
      }
    }
  }

  out.map_bits = estimate_palette_bits(out, block_w, block_h, visible_w,
                                       visible_h, cache, cache_n,
                                       PaletteCdfState{});
  out.mode_bits = out.map_bits;  // kept split-compatible; see header
  out.cache_used = 0;
  {
    std::uint8_t used_cache[16] = {};
    std::uint8_t remaining[kPaletteMaxSize] = {};
    split_cache_and_run(out.colors.data(), out.size, cache, cache_n, used_cache,
                        remaining);
    for (int i = 0; i < cache_n; ++i) out.cache_used += used_cache[i] ? 1 : 0;
  }

  // Prediction error of the final palette.
  out.sse = 0;
  for (int y = 0; y < visible_h; ++y) {
    for (int x = 0; x < visible_w; ++x) {
      const int idx = out.color_map[static_cast<std::size_t>(y) *
                                        static_cast<std::size_t>(block_w) +
                                    static_cast<std::size_t>(x)];
      const int d = src[static_cast<std::size_t>(y) *
                            static_cast<std::size_t>(stride) +
                        static_cast<std::size_t>(x)] -
                    out.colors[static_cast<std::size_t>(idx)];
      out.sse += static_cast<std::int64_t>(d) * d;
    }
  }

  if (params.max_bits_per_pixel > 0.0) {
    const double bpp = static_cast<double>(out.map_bits) /
                       static_cast<double>(block_w * block_h);
    if (bpp > params.max_bits_per_pixel) out.has_palette = false;
  }
  return true;
}

void store_palette_state(PaletteState& state, int mi_row, int mi_col,
                         int block_w, int block_h,
                         const PaletteResult& result) noexcept {
  if (!state.valid() || !result.has_palette) return;
  const int w4 = block_w / 4;
  const int h4 = block_h / 4;
  for (int y = 0; y < h4; ++y) {
    const int r = mi_row + y;
    if (r >= state.mi_rows) break;
    for (int x = 0; x < w4; ++x) {
      const int c = mi_col + x;
      if (c >= state.mi_cols) break;
      const std::size_t idx = static_cast<std::size_t>(r) *
                                  static_cast<std::size_t>(state.mi_cols) +
                              static_cast<std::size_t>(c);
      state.sizes[idx] = static_cast<std::uint8_t>(result.size);
      state.colors[idx] = result.colors;
    }
  }
}

void predict_palette_block(const PaletteResult& result, int block_w,
                           int block_h, std::uint8_t* dst, int dst_stride) {
  if (dst == nullptr || block_w <= 0 || block_h <= 0) return;
  palette_predict_tx(result.colors.data(), result.color_map.data(), block_w,
                     0, 0, dst, dst_stride, block_w, block_h);
}

void palette_residual(const std::uint8_t* src, int src_stride,
                      const PaletteResult& result, int block_w, int block_h,
                      std::int16_t* residual, int residual_stride) {
  for (int y = 0; y < block_h; ++y) {
    for (int x = 0; x < block_w; ++x) {
      const int idx = result.color_map[static_cast<std::size_t>(y) *
                                           static_cast<std::size_t>(block_w) +
                                       static_cast<std::size_t>(x)];
      const int pred = result.colors[static_cast<std::size_t>(idx)];
      const int value = src[static_cast<std::size_t>(y) *
                                static_cast<std::size_t>(src_stride) +
                            static_cast<std::size_t>(x)];
      residual[static_cast<std::size_t>(y) *
                   static_cast<std::size_t>(residual_stride) +
               static_cast<std::size_t>(x)] =
          static_cast<std::int16_t>(value - pred);
    }
  }
}

int palette_size_index(int size) noexcept { return size - 2; }

ChromaRule chroma_rule() noexcept { return ChromaRule::kIntraAfterLuma; }

// ---------------------------------------------------------------------------
// Colour-map token coding (§5.20.8.4)
// ---------------------------------------------------------------------------
namespace {

bool encode_ns(av2::Encoder& enc, int n, int value) {
  const int w = palette_floor_log2(n) + 1;
  const int m = (1 << w) - n;
  const int v = value < m ? value : (value + m) >> 1;
  for (int i = w - 2; i >= 0; --i) {
    if (enc.encode_bool(((v >> i) & 1) != 0) != av2::Status::kOk) return false;
  }
  if (v >= m) {
    if (enc.encode_bool(((value + m) & 1) != 0) != av2::Status::kOk) {
      return false;
    }
  }
  return true;
}

bool read_ns(av2::Decoder& dec, int n, int& value) {
  const int w = palette_floor_log2(n) + 1;
  const int m = (1 << w) - n;
  int v = 0;
  for (int i = 0; i < w - 1; ++i) {
    bool bit = false;
    if (dec.read_bool(&bit) != av2::Status::kOk) return false;
    v = (v << 1) | (bit ? 1 : 0);
  }
  if (v < m) {
    value = v;
    return true;
  }
  bool extra = false;
  if (dec.read_bool(&extra) != av2::Status::kOk) return false;
  value = (v << 1) - m + (extra ? 1 : 0);
  return true;
}

bool encode_color_map(av2::Encoder& enc, PaletteCdfState& cdfs,
                      const PaletteResult& result, int w, int h, int visible_w,
                      int visible_h) {
  (void)h;
  const int direction = result.direction;
  const int outer_lim = direction ? visible_w : visible_h;
  const int inner_lim = direction ? visible_h : visible_w;
  const int n = result.size;
  const auto map_at = [&](int r, int c) {
    return static_cast<int>(
        result.color_map[static_cast<std::size_t>(r) *
                             static_cast<std::size_t>(w) +
                         static_cast<std::size_t>(c)]);
  };
  int prev_identity = kPaletteRowFlagContexts - 1;
  for (int i = 0; i < outer_lim; ++i) {
    // Recompute rather than trust the caller: identity 2 is illegal at i == 0.
    const int identity = palette_classify_line(
        result.color_map.data(), w, direction, i, inner_lim, i > 0, n);
    if (i < static_cast<int>(result.identity_rows.size()) &&
        result.identity_rows[static_cast<std::size_t>(i)] != identity) {
      return false;  // analysis and syntax disagree; fail loudly
    }
    if (enc.encode_symbol(&cdfs.identity[prev_identity][0], 3, identity) !=
        av2::Status::kOk) {
      return false;
    }
    for (int j = 0; j < inner_lim; ++j) {
      int r = 0;
      int c = 0;
      palette_coord(direction, i, j, &r, &c);
      if (identity == 2) {
        const int pr = direction ? r : r - 1;
        const int pc = direction ? c - 1 : c;
        if (map_at(r, c) != map_at(pr, pc)) return false;
      } else if (identity == 1 && j > 0) {
        const int pr = direction ? r - 1 : r;
        const int pc = direction ? c : c - 1;
        if (map_at(r, c) != map_at(pr, pc)) return false;
      } else if (r == 0 && c == 0) {
        if (!encode_ns(enc, n, map_at(0, 0))) {
          return false;
        }
      } else {
        std::uint8_t order[8];
        const int hash = palette_color_context(result.color_map.data(), w, r, c,
                                               n, order);
        const int ctx = palette_color_context_index(hash);
        if (ctx < 0) return false;
        const int symbol = palette_symbol_for_color(map_at(r, c), order, n);
        if (symbol < 0) return false;
        if (enc.encode_symbol(color_cdf(cdfs, n) + ctx * (n + 1), n, symbol) !=
            av2::Status::kOk) {
          return false;
        }
      }
    }
    prev_identity = identity;
  }
  return true;
}

}  // namespace

bool encode_palette_y(av2::Encoder& enc, PaletteCdfState& cdfs,
                      const PaletteResult& result, int w, int h, int visible_w,
                      int visible_h, const std::uint8_t* cache, int cache_n,
                      std::string& error) {
  error.clear();
  if (!result.has_palette) {
    return enc.encode_symbol(&cdfs.y_mode[0], 2, 0) == av2::Status::kOk;
  }
  if (result.size < 2 || result.size > kPaletteMaxSize) {
    error = "palette size out of range";
    return false;
  }
  if (enc.encode_symbol(&cdfs.y_mode[0], 2, 1) != av2::Status::kOk) {
    error = "palette has_palette_y failed";
    return false;
  }
  if (enc.encode_symbol(&cdfs.y_size[0], 7, result.size - 2) !=
      av2::Status::kOk) {
    error = "palette size symbol failed";
    return false;
  }

  std::uint8_t used_cache[16] = {};
  std::uint8_t remaining[kPaletteMaxSize] = {};
  const int run = split_cache_and_run(result.colors.data(), result.size, cache,
                                      cache_n, used_cache, remaining);
  int idx = 0;
  for (int i = 0; i < cache_n && idx < result.size; ++i) {
    const bool use = used_cache[i] != 0;
    if (enc.encode_bool(use) != av2::Status::kOk) {
      error = "palette cache bit failed";
      return false;
    }
    if (use) ++idx;
  }
  if (idx < result.size) {
    const std::uint32_t literal = remaining[0];
    if (!encode_literal(enc, 8, literal)) {
      error = "palette literal failed";
      return false;
    }
    ++idx;
  }
  if (idx < result.size) {
    const int extra = palette_extra_bits_for(remaining, run);
    if (extra < 0) {
      error = "palette run is not delta-representable";
      return false;
    }
    if (!encode_literal(enc, 2, static_cast<std::uint32_t>(extra))) {
      error = "palette extra-bits failed";
      return false;
    }
    int palette_bits = 5 + extra;
    int prev = remaining[0];
    for (int i = 1; i < run; ++i) {
      const int color = remaining[i];
      const int delta = color - prev - 1;
      if (delta < 0 || delta >= (1 << palette_bits)) {
        error = "palette delta does not fit paletteBits";
        return false;
      }
      if (!encode_literal(enc, palette_bits,
                          static_cast<std::uint32_t>(delta))) {
        error = "palette delta failed";
        return false;
      }
      prev = color;
      const int range = 255 - color;
      palette_bits = std::min(palette_bits, palette_ceil_log2(range));
    }
  }

  if (w < 64 && h < 64) {
    if (!encode_literal(enc, 1,
                        static_cast<std::uint32_t>(result.direction))) {
      error = "palette direction failed";
      return false;
    }
  }
  if (!encode_color_map(enc, cdfs, result, w, h, visible_w, visible_h)) {
    error = "palette colour map failed";
    return false;
  }
  return true;
}

bool decode_palette_y(av2::Decoder& dec, PaletteCdfState& cdfs, int w, int h,
                      int visible_w, int visible_h, const std::uint8_t* cache,
                      int cache_n, PaletteResult& out, std::string& error) {
  error.clear();
  out = PaletteResult{};
  int has_palette = 0;
  if (dec.decode_symbol(&cdfs.y_mode[0], 2, &has_palette) != av2::Status::kOk) {
    error = "palette has_palette_y decode failed";
    return false;
  }
  if (has_palette == 0) return true;
  int size_symbol = 0;
  if (dec.decode_symbol(&cdfs.y_size[0], 7, &size_symbol) !=
      av2::Status::kOk) {
    error = "palette size decode failed";
    return false;
  }
  const int size = size_symbol + 2;
  out.has_palette = true;
  out.size = size;

  int idx = 0;
  for (int i = 0; i < cache_n && idx < size; ++i) {
    bool use = false;
    if (dec.read_bool(&use) != av2::Status::kOk) {
      error = "palette cache decode failed";
      return false;
    }
    if (use) {
      out.colors[static_cast<std::size_t>(idx++)] =
          cache != nullptr ? cache[i] : 0;
    }
  }
  if (idx < size) {
    std::uint32_t literal = 0;
    if (dec.read_literal(8, &literal) != av2::Status::kOk) {
      error = "palette literal decode failed";
      return false;
    }
    out.colors[static_cast<std::size_t>(idx++)] =
        static_cast<std::uint8_t>(literal);
  }
  if (idx < size) {
    std::uint32_t extra = 0;
    if (dec.read_literal(2, &extra) != av2::Status::kOk) {
      error = "palette extra-bits decode failed";
      return false;
    }
    int palette_bits = 5 + static_cast<int>(extra);
    while (idx < size) {
      std::uint32_t delta = 0;
      if (dec.read_literal(palette_bits, &delta) != av2::Status::kOk) {
        error = "palette delta decode failed";
        return false;
      }
      const int color = palette_clip1(
          static_cast<int>(out.colors[static_cast<std::size_t>(idx - 1)]) +
          static_cast<int>(delta) + 1);
      out.colors[static_cast<std::size_t>(idx++)] =
          static_cast<std::uint8_t>(color);
      const int range = 255 - color;
      palette_bits = std::min(palette_bits, palette_ceil_log2(range));
    }
  }
  std::sort(out.colors.begin(), out.colors.begin() + size);

  int direction = 0;
  if (w < 64 && h < 64) {
    std::uint32_t bit = 0;
    if (dec.read_literal(1, &bit) != av2::Status::kOk) {
      error = "palette direction decode failed";
      return false;
    }
    direction = static_cast<int>(bit);
  }
  out.direction = direction;
  out.color_map.assign(static_cast<std::size_t>(w) * static_cast<std::size_t>(h),
                       0);
  const int outer_lim = direction ? visible_w : visible_h;
  const int inner_lim = direction ? visible_h : visible_w;
  int prev_identity = kPaletteRowFlagContexts - 1;
  for (int i = 0; i < outer_lim; ++i) {
    int identity = 0;
    if (dec.decode_symbol(&cdfs.identity[prev_identity][0], 3, &identity) !=
        av2::Status::kOk) {
      error = "palette identity row decode failed";
      return false;
    }
    if (identity == 2 && i == 0) {
      error = "palette identity_row_y == 2 at i == 0";
      return false;
    }
    for (int j = 0; j < inner_lim; ++j) {
      int r = 0;
      int c = 0;
      palette_coord(direction, i, j, &r, &c);
      if (identity == 2) {
        const int pr = direction ? r : r - 1;
        const int pc = direction ? c - 1 : c;
        out.color_map[static_cast<std::size_t>(r) * static_cast<std::size_t>(w) +
                      static_cast<std::size_t>(c)] =
            out.color_map[static_cast<std::size_t>(pr) *
                              static_cast<std::size_t>(w) +
                          static_cast<std::size_t>(pc)];
      } else if (identity == 1 && j > 0) {
        const int pr = direction ? r - 1 : r;
        const int pc = direction ? c : c - 1;
        out.color_map[static_cast<std::size_t>(r) * static_cast<std::size_t>(w) +
                      static_cast<std::size_t>(c)] =
            out.color_map[static_cast<std::size_t>(pr) *
                              static_cast<std::size_t>(w) +
                          static_cast<std::size_t>(pc)];
      } else if (r == 0 && c == 0) {
        int value = 0;
        if (!read_ns(dec, size, value)) {
          error = "palette first index decode failed";
          return false;
        }
        out.color_map[0] = static_cast<std::uint8_t>(value);
      } else {
        std::uint8_t order[8];
        const int hash = palette_color_context(out.color_map.data(), w, r, c,
                                               size, order);
        const int ctx = palette_color_context_index(hash);
        if (ctx < 0) {
          error = "palette colour context hash has no CDF";
          return false;
        }
        int symbol = 0;
        if (dec.decode_symbol(color_cdf(cdfs, size) + ctx * (size + 1), size,
                              &symbol) != av2::Status::kOk) {
          error = "palette colour index decode failed";
          return false;
        }
        out.color_map[static_cast<std::size_t>(r) * static_cast<std::size_t>(w) +
                      static_cast<std::size_t>(c)] = order[symbol];
      }
    }
    prev_identity = identity;
  }
  replicate_edge_color_map(out.color_map.data(), w, w, h, visible_w, visible_h);
  return true;
}

int estimate_palette_bits(const PaletteResult& result, int w, int h,
                          int visible_w, int visible_h,
                          const std::uint8_t* cache, int cache_n,
                          const PaletteCdfState& cdfs) {
  (void)h;
  PaletteCdfState scratch = cdfs;
  if (scratch.y_mode[0] == 0 && scratch.y_size[0] == 0) {
    init_palette_cdfs(scratch);
  }
  static thread_local std::vector<std::uint8_t> buffer(1U << 20);
  av2::Encoder enc;
  if (enc.init_symbol(buffer.data(), buffer.size()) != av2::Status::kOk) {
    return std::numeric_limits<int>::max();
  }
  enc.disable_cdf_update = 1;
  PaletteResult copy = result;
  std::string error;
  if (!encode_palette_y(enc, scratch, copy, w, h, visible_w, visible_h, cache,
                        cache_n, error)) {
    return std::numeric_limits<int>::max();
  }
  std::size_t bytes = 0;
  if (enc.exit_symbol(&bytes) != av2::Status::kOk) {
    return std::numeric_limits<int>::max();
  }
  return static_cast<int>(bytes) * 8;
}

}  // namespace av2enc::palette

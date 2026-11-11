// Shared primitives for the BRU writer/reader.  These are pure syntax helpers
// (no stream state) so using them from both sides does not couple the writer
// and reader state machines.
#pragma once

#include "bru.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

namespace av2enc::inter::bru::detail {

inline int ceil_log2(std::uint32_t value) {
  if (value <= 1U) return 0;
  int bits = 0;
  std::uint32_t v = value - 1U;
  while (v != 0U) {
    v >>= 1U;
    ++bits;
  }
  return bits;
}

inline int bit_width(std::uint32_t value) {
  int bits = 0;
  while (value != 0U) {
    value >>= 1U;
    ++bits;
  }
  return bits;
}

inline int floor_log2(std::uint32_t value) {
  int bits = -1;
  while (value != 0U) {
    value >>= 1U;
    ++bits;
  }
  return bits;
}

inline void put_uvlc(BitWriter& writer, std::uint32_t value) {
  unsigned leading_zeros = 0;
  while (value >= ((1ULL << (leading_zeros + 1U)) - 1ULL)) {
    ++leading_zeros;
    assert(leading_zeros < 32U);
  }
  writer.put_bits(0U, leading_zeros);
  writer.put_bit(true);
  const std::uint64_t payload =
      static_cast<std::uint64_t>(value) - ((1ULL << leading_zeros) - 1ULL);
  writer.put_bits(payload, leading_zeros);
}

inline void put_ns(BitWriter& writer, std::uint32_t n, std::uint32_t value) {
  assert(n >= 2U);
  assert(value < n);
  unsigned w = 0;
  while ((1U << w) < n) ++w;
  const std::uint32_t m = (1U << w) - n;
  if (value < m) {
    writer.put_bits(value, w - 1U);
  } else {
    const std::uint32_t x = value + m;
    writer.put_bits(x >> 1U, w - 1U);
    writer.put_bit((x & 1U) != 0U);
  }
}

inline void put_trailing_bits(BitWriter& writer) {
  writer.put_bit(true);
  writer.align_to_byte();
}

// le(n) from §4.11: little-endian fixed-width unsigned integer.
inline void put_le(BitWriter& writer, std::uint32_t value, int bytes) {
  for (int i = 0; i < bytes; ++i) {
    writer.put_bits((value >> (8 * i)) & 0xFFU, 8U);
  }
}

// §5.18.7.7.
inline int tile_log2(int blk_size, int target) {
  if (blk_size <= 0) return -1;
  int k = 0;
  while ((blk_size << k) < target) ++k;
  return k;
}

// §9.2 Tile_Width_Scaling_Factor / Tile_Area_Scaling_Factor for tier 0.
// Only the factors are needed; the level-31 special case is handled by the
// caller.  Mirrors the table used by encoder/tiles/tile_layout.cpp.
inline bool tile_scaling_factors(int seq_level_idx, int& width_factor,
                                 int& area_factor) {
  if (seq_level_idx < 0 || seq_level_idx >= 22) return false;
  width_factor = (seq_level_idx <= 17) ? 4 : 8;
  area_factor =
      (seq_level_idx <= 13) ? 4 : ((seq_level_idx <= 17) ? 8 : 16);
  return true;
}

inline constexpr int kMaxTileWidthPixels = 4096;
inline constexpr int kMaxTileAreaPixels = 4096 * 2304;
inline constexpr int kMaxTileColumns = 64;
inline constexpr int kMaxTileRows = 64;

// §5.18.7.5 uniform_spacing() for 64x64 superblocks.
struct UniformSpacing {
  std::vector<int> starts;
  int count = 1;
};

inline UniformSpacing uniform_spacing(int tile_log2_count, int mis,
                                      int sb_size) {
  UniformSpacing out;
  const int sb4x4 = sb_size / 4;
  const int sb_shift = ceil_log2(static_cast<std::uint32_t>(sb4x4));
  const int sbs = (mis + sb4x4 - 1) >> sb_shift;
  const int full_sbs = mis >> sb_shift;
  const int tile_sb = full_sbs >> tile_log2_count;
  const int extra_sbs =
      (tile_sb == 0) ? sbs : (full_sbs - (tile_sb << tile_log2_count));
  int start = 0;
  int i = 0;
  const int target = 1 << tile_log2_count;
  for (; i < target && start < sbs; ++i) {
    out.starts.push_back(start);
    start += tile_sb;
    if (i < extra_sbs) ++start;
  }
  out.count = i;
  return out;
}

// §9.2 / §5.18.7.3: minimum log2 tile columns for a level.  Our frames are
// small enough that the level bound is always inactive, but deriving it keeps
// the writer honest for future callers.
inline int min_log2_tile_cols(const SequenceConfig& c, int sb_cols) {
  if (c.seq_level_idx == 31) return 0;
  int width_factor = 4;
  int area_factor = 4;
  if (!tile_scaling_factors(c.seq_level_idx, width_factor, area_factor)) {
    return 0;
  }
  const int sb_shift = 4;  // 64x64 superblocks
  const int max_tile_width_sb =
      (width_factor * kMaxTileWidthPixels) >> (sb_shift + 4);
  return std::max(0, tile_log2(max_tile_width_sb, sb_cols));
}

inline int min_log2_tiles(const SequenceConfig& c, int sb_cols, int sb_rows,
                          int min_log2_cols) {
  if (c.seq_level_idx == 31) return min_log2_cols;
  int width_factor = 4;
  int area_factor = 4;
  if (!tile_scaling_factors(c.seq_level_idx, width_factor, area_factor)) {
    return min_log2_cols;
  }
  const int sb_shift = 4;
  const int max_tile_area_sb =
      (area_factor * kMaxTileAreaPixels) >> (2 * (sb_shift + 2) + 2);
  const int area_log2 = tile_log2(max_tile_area_sb, sb_rows * sb_cols);
  return std::max(min_log2_cols, area_log2);
}

inline int max_log2_tile_cols(int sb_cols) {
  return tile_log2(1, std::min(sb_cols, kMaxTileColumns));
}

inline int max_log2_tile_rows(int sb_rows) {
  return tile_log2(1, std::min(sb_rows, kMaxTileRows));
}

}  // namespace av2enc::inter::bru::detail

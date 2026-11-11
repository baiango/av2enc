#include "encoder/syntax/small_header/small_header.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace av2enc::syntax::small_header {
namespace {

void set_error(std::string* error, std::string message) {
  if (error != nullptr) *error = std::move(message);
}

void put_uvlc(BitWriter& writer, std::uint32_t value) {
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

void put_ns(BitWriter& writer, unsigned n, std::uint32_t value) {
  if (n <= 1U) return;
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

void put_su(BitWriter& writer, unsigned n, std::int32_t value) {
  if (n == 0U) return;
  const std::uint32_t mask = (1U << n) - 1U;
  writer.put_bits(static_cast<std::uint32_t>(value) & mask, n);
}

struct SbGeometry {
  int mi_cols = 0;
  int mi_rows = 0;
  int sb4x4 = 0;
  int sb_shift_value = 0;
  int sb_cols = 0;
  int sb_rows = 0;
};

SbGeometry geometry_for(int frame_width, int frame_height, SbSize sb_size) {
  SbGeometry g;
  g.mi_cols = 2 * ((frame_width + 7) >> 3);
  g.mi_rows = 2 * ((frame_height + 7) >> 3);
  g.sb4x4 = sb_width_4x4(sb_size);
  g.sb_shift_value = sb_shift(sb_size);
  g.sb_cols = (g.mi_cols + g.sb4x4 - 1) >> g.sb_shift_value;
  g.sb_rows = (g.mi_rows + g.sb4x4 - 1) >> g.sb_shift_value;
  return g;
}

struct LevelFactors {
  int width = 4;
  int area = 4;
  bool reserved = false;
};

// §9.2 Tile_Width_Scaling_Factor / Tile_Area_Scaling_Factor.
LevelFactors level_factors(int seq_level_idx, int seq_tier) {
  LevelFactors f;
  if (seq_level_idx < 0 || seq_level_idx > 31 || (seq_tier != 0 && seq_tier != 1)) {
    f.reserved = true;
    return f;
  }
  if (seq_level_idx == 31) {
    f.width = 1;
    f.area = 1;
    return f;
  }
  if (seq_level_idx >= 22) {
    f.reserved = true;
    return f;
  }
  if (seq_tier == 0) {
    f.width = (seq_level_idx <= 17) ? 4 : 8;
    f.area = (seq_level_idx <= 13) ? 4 : ((seq_level_idx <= 17) ? 8 : 16);
  } else {
    f.width = (seq_level_idx <= 13) ? 4 : ((seq_level_idx <= 17) ? 8 : 16);
    f.area = (seq_level_idx <= 13) ? 4 : ((seq_level_idx <= 17) ? 16 : 32);
  }
  return f;
}

inline constexpr int kMaxTileWidthPixels = 4096;       // MAX_TILE_WIDTH
inline constexpr int kMaxTileAreaPixels = 4096 * 2304;  // MAX_TILE_AREA
inline constexpr int kMaxTileCols = 64;                 // MAX_TILE_COLS
inline constexpr int kMaxTileRows = 64;                 // MAX_TILE_ROWS

bool derive_tile_limits(const SbGeometry& g, int seq_level_idx, int seq_tier,
                        TileParams* params, std::string* error) {
  const LevelFactors factors = level_factors(seq_level_idx, seq_tier);
  if (factors.reserved) {
    set_error(error, "reserved seq_level_idx/seq_tier combination");
    return false;
  }
  if (seq_level_idx == 31) {
    params->max_tile_width_sb = g.sb_cols;
    params->max_tile_area_sb = g.sb_cols * g.sb_rows;
  } else {
    const int shift = g.sb_shift_value;
    params->max_tile_width_sb =
        (factors.width * kMaxTileWidthPixels) >> (shift + 4);
    params->max_tile_area_sb =
        (factors.area * kMaxTileAreaPixels) >> (2 * (shift + 2) + 2);
  }
  if (params->max_tile_width_sb <= 0 || params->max_tile_area_sb <= 0) {
    set_error(error, "level yields a non-positive tile bound");
    return false;
  }
  params->min_log2_tile_cols =
      tile_log2(params->max_tile_width_sb, g.sb_cols);
  params->max_log2_tile_cols =
      tile_log2(1, std::min(g.sb_cols, kMaxTileCols));
  params->max_log2_tile_rows =
      tile_log2(1, std::min(g.sb_rows, kMaxTileRows));
  params->min_log2_tiles =
      std::max(params->min_log2_tile_cols,
               tile_log2(params->max_tile_area_sb, g.sb_rows * g.sb_cols));
  if (params->min_log2_tile_cols < 0 || params->max_log2_tile_cols < 0 ||
      params->max_log2_tile_rows < 0 || params->min_log2_tiles < 0) {
    set_error(error, "failed to derive tile limits");
    return false;
  }
  return true;
}

// §5.18.7.5 uniform_spacing().
std::vector<int> uniform_starts(int mis, SbSize sb_size, int tile_log2_value,
                                int* tile_count) {
  const int sb4x4 = sb_width_4x4(sb_size);
  const int shift = sb_shift(sb_size);
  const int sbs = (mis + sb4x4 - 1) >> shift;
  const int full_sbs = mis >> shift;
  const int tile_sb = full_sbs >> tile_log2_value;
  const int extra =
      (tile_sb == 0) ? sbs : full_sbs - (tile_sb << tile_log2_value);
  std::vector<int> starts;
  int start = 0;
  int i = 0;
  const int tiles = 1 << tile_log2_value;
  for (; i < tiles && start < sbs; ++i) {
    starts.push_back(start);
    start += tile_sb;
    if (i < extra) ++start;
  }
  if (tile_count != nullptr) *tile_count = i;
  return starts;
}

bool validate_tile_starts(const std::vector<int>& starts, int sb_count,
                          int tile_rows_or_cols, std::string* error) {
  if (tile_rows_or_cols <= 0 ||
      static_cast<int>(starts.size()) != tile_rows_or_cols) {
    set_error(error, "tile start count disagrees with the tile count");
    return false;
  }
  if (starts.empty() || starts.front() != 0) {
    set_error(error, "tile starts must begin at zero");
    return false;
  }
  for (std::size_t i = 1; i < starts.size(); ++i) {
    if (starts[i] <= starts[i - 1]) {
      set_error(error, "tile starts are not strictly increasing");
      return false;
    }
  }
  if (starts.back() >= sb_count) {
    set_error(error, "tile start lies outside the superblock grid");
    return false;
  }
  return true;
}

// §5.18.7.4 reuse_tile_params().
bool reuse_tile_params(const TileParams& sequence, const SbGeometry& g,
                       SbSize sb_size, TileParams* params,
                       std::string* error) {
  const int mi_cols = g.mi_cols;
  const int mi_rows = g.mi_rows;
  if (sequence.uniform_spacing) {
    int tile_cols = 0;
    int tile_rows = 0;
    params->sb_col_starts =
        uniform_starts(mi_cols, sequence.uniform_sb_size,
                       sequence.tile_cols_log2, &tile_cols);
    params->sb_row_starts =
        uniform_starts(mi_rows, sequence.uniform_sb_size,
                       sequence.tile_rows_log2, &tile_rows);
    params->tile_cols = tile_cols;
    params->tile_rows = tile_rows;
    params->tile_cols_log2 = tile_log2(1, tile_cols);
    params->tile_rows_log2 = tile_log2(1, tile_rows);
    params->sb_shift = sb_shift(sequence.uniform_sb_size);
  } else {
    params->sb_col_starts = sequence.sb_col_starts;
    params->sb_row_starts = sequence.sb_row_starts;
    params->tile_cols = sequence.tile_cols;
    params->tile_rows = sequence.tile_rows;
    params->tile_cols_log2 = tile_log2(1, params->tile_cols);
    params->tile_rows_log2 = tile_log2(1, params->tile_rows);
    params->sb_shift = sb_shift(sb_size);
  }
  params->uniform_spacing = sequence.uniform_spacing;
  params->uniform_sb_size = sequence.uniform_sb_size;
  params->sb_size = sb_size;
  params->frame_width = mi_cols * 4;
  params->frame_height = mi_rows * 4;
  params->sb_cols = g.sb_cols;
  params->sb_rows = g.sb_rows;
  if (!validate_tile_starts(params->sb_col_starts, g.sb_cols,
                            params->tile_cols, error) ||
      !validate_tile_starts(params->sb_row_starts, g.sb_rows,
                            params->tile_rows, error)) {
    return false;
  }
  return true;
}

bool eligible_sequence_tile_params(const SequenceTileConfig& sequence,
                                   const SbGeometry& g) {
  if (!sequence.present) return false;
  if (sequence.params.uniform_spacing) {
    return uniform_eligible(sequence.params.tile_rows_log2, g.sb_rows) &&
           uniform_eligible(sequence.params.tile_cols_log2, g.sb_cols);
  }
  return sequence.params.sb_cols == g.sb_cols &&
         sequence.params.sb_rows == g.sb_rows;
}

}  // namespace

// ---------------------------------------------------------------------------
// BitReader
// ---------------------------------------------------------------------------

bool BitReader::read_bit(bool* bit) {
  if (bit == nullptr || bit_pos_ >= bit_size_) {
    failed_ = true;
    return false;
  }
  const std::size_t byte = bit_pos_ >> 3U;
  const unsigned shift = 7U - static_cast<unsigned>(bit_pos_ & 7U);
  *bit = ((bytes_[byte] >> shift) & 1U) != 0U;
  ++bit_pos_;
  return true;
}

bool BitReader::read_bits(unsigned num_bits, std::uint32_t* value) {
  if (value == nullptr || num_bits > 32U || bits_left() < num_bits) {
    failed_ = true;
    return false;
  }
  std::uint32_t out = 0;
  for (unsigned i = 0; i < num_bits; ++i) {
    bool bit = false;
    if (!read_bit(&bit)) return false;
    out = (out << 1U) | (bit ? 1U : 0U);
  }
  *value = out;
  return true;
}

bool BitReader::read_uvlc(std::uint32_t* value) {
  if (value == nullptr) {
    failed_ = true;
    return false;
  }
  unsigned leading_zeros = 0;
  bool bit = false;
  for (;;) {
    if (!read_bit(&bit)) return false;
    if (bit) break;
    ++leading_zeros;
    if (leading_zeros >= 32U) {
      failed_ = true;
      return false;
    }
  }
  std::uint32_t payload = 0;
  if (!read_bits(leading_zeros, &payload)) return false;
  const std::uint64_t base = (1ULL << leading_zeros) - 1ULL;
  const std::uint64_t result = base + payload;
  if (result > std::numeric_limits<std::uint32_t>::max()) {
    failed_ = true;
    return false;
  }
  *value = static_cast<std::uint32_t>(result);
  return true;
}

bool BitReader::read_ns(unsigned n, std::uint32_t* value) {
  if (value == nullptr || n == 0U) {
    failed_ = true;
    return false;
  }
  if (n == 1U) {
    *value = 0;
    return true;
  }
  unsigned w = 0;
  while ((1U << w) < n) ++w;
  const std::uint32_t m = (1U << w) - n;
  std::uint32_t v = 0;
  if (!read_bits(w - 1U, &v)) return false;
  if (v < m) {
    *value = v;
    return true;
  }
  bool extra = false;
  if (!read_bit(&extra)) return false;
  const std::uint32_t result = (v << 1U) - m + (extra ? 1U : 0U);
  if (result >= n) {
    failed_ = true;
    return false;
  }
  *value = result;
  return true;
}

bool BitReader::read_su(unsigned n, std::int32_t* value) {
  if (value == nullptr || n == 0U || n > 31U) {
    failed_ = true;
    return false;
  }
  std::uint32_t raw = 0;
  if (!read_bits(n, &raw)) return false;
  const std::uint32_t sign = 1U << (n - 1U);
  std::int64_t signed_value = static_cast<std::int64_t>(raw);
  if ((raw & sign) != 0U) {
    signed_value -= static_cast<std::int64_t>(1ULL << n);
  }
  *value = static_cast<std::int32_t>(signed_value);
  return true;
}

bool BitReader::align_to_byte() {
  while ((bit_pos_ & 7U) != 0U) {
    bool bit = false;
    if (!read_bit(&bit)) return false;
    if (bit) {
      failed_ = true;
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Cropping window
// ---------------------------------------------------------------------------

bool validate_cropping_window(const CroppingWindow& window,
                              std::uint32_t max_frame_width,
                              std::uint32_t max_frame_height,
                              std::string* error) {
  if (max_frame_width == 0 || max_frame_height == 0) {
    set_error(error, "max frame dimensions must be non-zero");
    return false;
  }
  if (!window.present) return true;
  if (window.left > max_frame_width - 1U) {
    set_error(error, "seq_cropping_win_left_offset exceeds max width - 1");
    return false;
  }
  if (window.right > max_frame_width - 1U) {
    set_error(error, "seq_cropping_win_right_offset exceeds max width - 1");
    return false;
  }
  if (window.top > max_frame_height - 1U) {
    set_error(error, "seq_cropping_win_top_offset exceeds max height - 1");
    return false;
  }
  if (window.bottom > max_frame_height - 1U) {
    set_error(error, "seq_cropping_win_bottom_offset exceeds max height - 1");
    return false;
  }
  DisplaySize size;
  return derive_display_size(window, max_frame_width, max_frame_height,
                             max_frame_width, max_frame_height, &size, error);
}

bool write_cropping_window(BitWriter& writer, const CroppingWindow& window,
                           std::string* error) {
  // The writer validates the uvlc() field values but cannot know the maximum
  // frame size; callers with limits should call validate_cropping_window().
  if (window.present) {
    if (window.left > 0xFFFFU || window.right > 0xFFFFU ||
        window.top > 0xFFFFU || window.bottom > 0xFFFFU) {
      set_error(error, "cropping offset exceeds the 16-bit frame-size domain");
      return false;
    }
  }
  writer.put_bit(window.present);
  if (window.present) {
    put_uvlc(writer, window.left);
    put_uvlc(writer, window.right);
    put_uvlc(writer, window.top);
    put_uvlc(writer, window.bottom);
  }
  return true;}

bool read_cropping_window(BitReader& reader, CroppingWindow* window,
                          std::string* error) {
  if (window == nullptr) {
    set_error(error, "null CroppingWindow");
    return false;
  }
  CroppingWindow out;
  if (!reader.read_bit(&out.present)) {
    set_error(error, "truncated seq_cropping_window_present_flag");
    return false;
  }
  if (out.present) {
    if (!reader.read_uvlc(&out.left) || !reader.read_uvlc(&out.right) ||
        !reader.read_uvlc(&out.top) || !reader.read_uvlc(&out.bottom)) {
      set_error(error, "truncated cropping window offsets");
      return false;
    }
  }
  *window = out;
  return true;
}

bool derive_display_size(const CroppingWindow& window,
                         std::uint32_t max_frame_width,
                         std::uint32_t max_frame_height,
                         std::uint32_t frame_width,
                         std::uint32_t frame_height, DisplaySize* size,
                         std::string* error) {
  if (size == nullptr) {
    set_error(error, "null DisplaySize");
    return false;
  }
  if (max_frame_width == 0 || max_frame_height == 0 || frame_width == 0 ||
      frame_height == 0) {
    set_error(error, "frame dimensions must be non-zero");
    return false;
  }
  if (frame_width > max_frame_width || frame_height > max_frame_height) {
    set_error(error, "frame size exceeds the sequence maximum");
    return false;
  }
  if (!window.present) {
    size->width = frame_width;
    size->height = frame_height;
    return true;
  }
  const std::uint64_t max_w = max_frame_width;
  const std::uint64_t max_h = max_frame_height;
  const std::uint64_t fw = frame_width;
  const std::uint64_t fh = frame_height;
  const std::uint64_t crop_left = (static_cast<std::uint64_t>(window.left) * fw) / max_w;
  const std::uint64_t crop_right =
      fw - ((static_cast<std::uint64_t>(window.right) * fw) / max_w);
  const std::uint64_t crop_top = (static_cast<std::uint64_t>(window.top) * fh) / max_h;
  const std::uint64_t crop_bottom =
      fh - ((static_cast<std::uint64_t>(window.bottom) * fh) / max_h);
  if (crop_left >= crop_right || crop_top >= crop_bottom) {
    set_error(error, "cropping window produces an empty display rectangle");
    return false;
  }
  size->width = static_cast<std::uint32_t>(crop_right - crop_left);
  size->height = static_cast<std::uint32_t>(crop_bottom - crop_top);
  return true;
}

// ---------------------------------------------------------------------------
// PB aspect ratio
// ---------------------------------------------------------------------------

int max_pb_aspect_ratio(const PbAspectRatio& config) {
  if (!config.reduce) return 8;
  return 1 << (static_cast<int>(config.max_pb_aspect_ratio_log2_minus_1) + 1);
}

bool validate_pb_aspect_ratio(const PbAspectRatio& config,
                              std::string* error) {
  if (!config.reduce) {
    if (config.max_pb_aspect_ratio_log2_minus_1 != 0U) {
      set_error(error,
                "max_pb_aspect_ratio_log2_minus_1 is not coded when "
                "reduce_pb_aspect_ratio is 0");
      return false;
    }
    return true;
  }
  if (config.max_pb_aspect_ratio_log2_minus_1 > 1U) {
    set_error(error, "max_pb_aspect_ratio_log2_minus_1 must be 0 or 1");
    return false;
  }
  return true;
}

bool write_pb_aspect_ratio(BitWriter& writer, const PbAspectRatio& config,
                           std::string* error) {
  if (!validate_pb_aspect_ratio(config, error)) return false;
  writer.put_bit(config.reduce);
  if (config.reduce) {
    writer.put_bit(config.max_pb_aspect_ratio_log2_minus_1 != 0U);
  }
  return true;
}

bool read_pb_aspect_ratio(BitReader& reader, PbAspectRatio* config,
                          std::string* error) {
  if (config == nullptr) {
    set_error(error, "null PbAspectRatio");
    return false;
  }
  PbAspectRatio out;
  if (!reader.read_bit(&out.reduce)) {
    set_error(error, "truncated reduce_pb_aspect_ratio");
    return false;
  }
  if (out.reduce) {
    bool bit = false;
    if (!reader.read_bit(&bit)) {
      set_error(error, "truncated max_pb_aspect_ratio_log2_minus_1");
      return false;
    }
    out.max_pb_aspect_ratio_log2_minus_1 = bit ? 1U : 0U;
  }
  *config = out;
  return true;
}

bool pb_leaf_aspect_allowed(int bw4, int bh4, int max_ratio) {
  if (bw4 <= 0 || bh4 <= 0 || max_ratio <= 0) return false;
  const long long bw = bw4;
  const long long bh = bh4;
  const long long ratio = max_ratio;
  return bw <= bh * ratio && bh <= bw * ratio;
}

bool pb_partition_none_allowed_by_aspect(int bw4, int bh4, int max_ratio) {
  return pb_leaf_aspect_allowed(bw4, bh4, max_ratio);
}

bool pb_non_none_partition_allowed_by_aspect(int bw4, int bh4, int max_ratio) {
  if (bw4 <= 0 || bh4 <= 0 || max_ratio <= 0) return false;
  const long long bw = bw4;
  const long long bh = bh4;
  const long long ratio = max_ratio;
  if (bw <= bh * ratio && bh <= bw * ratio) return true;
  // §6.19.3.2: if either side reaches an 8:1 ratio, the fragment rejects the
  // partition entirely; otherwise only PARTITION_NONE is rejected.
  return !(bw >= bh * 8 || bh >= bw * 8);
}

// ---------------------------------------------------------------------------
// df_par_bits_minus_2
// ---------------------------------------------------------------------------

bool validate_df_par(const DfPar& config, std::string* error) {
  if (config.df_par_bits_minus_2 > 3U) {
    set_error(error, "df_par_bits_minus_2 is f(2), so it must be 0..3");
    return false;
  }
  return true;
}

bool write_df_par(BitWriter& writer, const DfPar& config, std::string* error) {
  if (!validate_df_par(config, error)) return false;
  writer.put_bits(config.df_par_bits_minus_2, 2U);
  return true;
}

bool read_df_par(BitReader& reader, DfPar* config, std::string* error) {
  if (config == nullptr) {
    set_error(error, "null DfPar");
    return false;
  }
  std::uint32_t value = 0;
  if (!reader.read_bits(2U, &value)) {
    set_error(error, "truncated df_par_bits_minus_2");
    return false;
  }
  config->df_par_bits_minus_2 = static_cast<std::uint8_t>(value);
  return true;
}

bool df_delta_q_representable(int value, const DfPar& config) {
  if (!validate_df_par(config, nullptr)) return false;
  return value >= config.min_value() && value <= config.max_value();
}

bool validate_df_delta_q(const DfPar& config, const DfDeltaQSet& values,
                         std::string* error) {
  if (!validate_df_par(config, error)) return false;
  for (int i = 0; i < 4; ++i) {
    if (!values.present[static_cast<std::size_t>(i)]) continue;
    const int value = values.value[static_cast<std::size_t>(i)];
    if (!df_delta_q_representable(value, config)) {
      set_error(error,
                "DfDeltaQ[" + std::to_string(i) + "] = " +
                    std::to_string(value) + " does not fit df_par_bits_minus_2" +
                    " = " + std::to_string(config.df_par_bits_minus_2) +
                    " (range " + std::to_string(config.min_value()) + ".." +
                    std::to_string(config.max_value()) + ")");
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Segments
// ---------------------------------------------------------------------------

bool operator==(const SegmentInfo& a, const SegmentInfo& b) {
  return a.enabled == b.enabled && a.value == b.value;
}

bool validate_segment_info(const SegmentInfo& info, int num_segments,
                           std::string* error) {
  if (num_segments != kBaseSegments && num_segments != kMaxSegments) {
    set_error(error, "num_segments must be 8 or 16");
    return false;
  }
  for (int i = 0; i < num_segments; ++i) {
    for (int j = 0; j < kSegLvlMax; ++j) {
      const std::size_t si = static_cast<std::size_t>(i);
      const std::size_t sj = static_cast<std::size_t>(j);
      if (!info.enabled[si][sj]) {
        if (info.value[si][sj] != 0) {
          set_error(error, "a disabled segment feature must have value 0");
          return false;
        }
        continue;
      }
      if (j == 0) {
        if (info.value[si][sj] < -kSegFeatureMax ||
            info.value[si][sj] > kSegFeatureMax) {
          set_error(error,
                    "SEG_LVL_ALT_Q value lies outside [-MAXQ_BITS, MAXQ_BITS]");
          return false;
        }
      } else if (info.value[si][sj] != 0) {
        set_error(error,
                  "SEG_LVL_SKIP/SEG_LVL_GLOBALMV carry zero value bits");
        return false;
      }
    }
  }
  for (int i = num_segments; i < kMaxSegments; ++i) {
    for (int j = 0; j < kSegLvlMax; ++j) {
      const std::size_t si = static_cast<std::size_t>(i);
      const std::size_t sj = static_cast<std::size_t>(j);
      if (info.enabled[si][sj] || info.value[si][sj] != 0) {
        set_error(error, "segment state outside MaxSegments must be zero");
        return false;
      }
    }
  }
  return true;
}

bool write_segment_info(BitWriter& writer, const SegmentInfo& info,
                        int num_segments, std::string* error) {
  if (!validate_segment_info(info, num_segments, error)) return false;
  for (int i = 0; i < num_segments; ++i) {
    for (int j = 0; j < kSegLvlMax; ++j) {
      const std::size_t si = static_cast<std::size_t>(i);
      const std::size_t sj = static_cast<std::size_t>(j);
      const bool enabled = info.enabled[si][sj];
      writer.put_bit(enabled);
      if (!enabled) continue;
      if (j == 0) {
        // Segmentation_Feature_Bits[0] == 9, Signed == 1 -> su(10).
        put_su(writer, 1U + kSegFeatureBits,
               static_cast<std::int32_t>(info.value[si][sj]));
      }
      // j == SEG_LVL_SKIP / SEG_LVL_GLOBALMV: f(0), no bits.
    }
  }
  return true;
}

bool read_segment_info(BitReader& reader, int num_segments, SegmentInfo* info,
                       std::string* error) {
  if (info == nullptr) {
    set_error(error, "null SegmentInfo");
    return false;
  }
  if (num_segments != kBaseSegments && num_segments != kMaxSegments) {
    set_error(error, "num_segments must be 8 or 16");
    return false;
  }
  SegmentInfo out;
  for (int i = 0; i < num_segments; ++i) {
    for (int j = 0; j < kSegLvlMax; ++j) {
      const std::size_t si = static_cast<std::size_t>(i);
      const std::size_t sj = static_cast<std::size_t>(j);
      bool enabled = false;
      if (!reader.read_bit(&enabled)) {
        set_error(error, "truncated seg_info feature_enabled");
        return false;
      }
      out.enabled[si][sj] = enabled;
      if (!enabled) continue;
      if (j == 0) {
        std::int32_t value = 0;
        if (!reader.read_su(1U + kSegFeatureBits, &value)) {
          set_error(error, "truncated SEG_LVL_ALT_Q feature_value");
          return false;
        }
        // §5.4.9 clips, but a conforming encoder writes the clipped domain.
        if (value < -kSegFeatureMax) value = -kSegFeatureMax;
        if (value > kSegFeatureMax) value = kSegFeatureMax;
        out.value[si][sj] = value;
      } else {
        out.value[si][sj] = 0;
      }
    }
  }
  *info = out;
  return true;
}

bool validate_sequence_segment_config(const SequenceSegmentConfig& config,
                                      std::string* error) {
  if (!config.present) {
    if (config.allow_change) {
      set_error(error,
                "seq_allow_seg_info_change is not coded when "
                "seq_seg_info_present_flag is 0");
      return false;
    }
    if (!validate_segment_info(config.info, kMaxSegments, error)) return false;
    return true;
  }
  return validate_segment_info(config.info, config.max_segments(), error);
}

bool write_sequence_segment_config(BitWriter& writer,
                                   const SequenceSegmentConfig& config,
                                   std::string* error) {
  if (!validate_sequence_segment_config(config, error)) return false;
  writer.put_bit(config.enable_ext_seg);
  writer.put_bit(config.present);
  if (!config.present) return true;
  writer.put_bit(config.allow_change);
  return write_segment_info(writer, config.info, config.max_segments(), error);
}

bool read_sequence_segment_config(BitReader& reader,
                                  SequenceSegmentConfig* config,
                                  std::string* error) {
  if (config == nullptr) {
    set_error(error, "null SequenceSegmentConfig");
    return false;
  }
  SequenceSegmentConfig out;
  if (!reader.read_bit(&out.enable_ext_seg)) {
    set_error(error, "truncated enable_ext_seg");
    return false;
  }
  if (!reader.read_bit(&out.present)) {
    set_error(error, "truncated seq_seg_info_present_flag");
    return false;
  }
  if (out.present) {
    if (!reader.read_bit(&out.allow_change)) {
      set_error(error, "truncated seq_allow_seg_info_change");
      return false;
    }
    if (!read_segment_info(reader, out.max_segments(), &out.info, error)) {
      return false;
    }
  }
  *config = out;
  return true;
}

bool write_frame_segmentation_params(BitWriter& writer,
                                     const SequenceSegmentConfig& config,
                                     const FrameSegmentRequest& request,
                                     bool derived_primary_ref_none,
                                     bool frame_type_switch,
                                     FrameSegmentState* state,
                                     std::string* error) {
  if (state == nullptr) {
    set_error(error, "null FrameSegmentState");
    return false;
  }
  if (request.enabled) {
    const bool will_reuse =
        config.present && (!config.allow_change || request.reuse_sequence);
    if (config.present && !config.allow_change && !request.reuse_sequence) {
      set_error(
          error,
          "seq_allow_seg_info_change == 0 forbids a frame override request");
      return false;
    }
    if (!will_reuse &&
        !validate_segment_info(request.info, config.max_segments(), error)) {
      return false;
    }
  }
  FrameSegmentState out;
  writer.put_bit(request.enabled);
  if (!request.enabled) {
    *state = out;
    return true;
  }

  out.have_sequence_params = config.present;
  const bool allow_change = config.present && config.allow_change;
  bool reuse = config.present;
  if (allow_change) {
    reuse = request.reuse_sequence;
    writer.put_bit(reuse);
  } else if (config.present && !request.reuse_sequence) {
    set_error(error,
              "seq_allow_seg_info_change == 0 forbids a frame override request");
    return false;
  }
  if (reuse) {
    if (!config.present) {
      set_error(error, "frame segmentation cannot reuse absent sequence info");
      return false;
    }
    out.info = config.info;
    out.reused_sequence = true;
  } else {
    if (config.present && !config.allow_change) {
      set_error(error,
                "seq_allow_seg_info_change == 0 forbids a frame seg_info()");
      return false;
    }
    if (!validate_segment_info(request.info, config.max_segments(), error)) {
      return false;
    }
    if (!write_segment_info(writer, request.info, config.max_segments(),
                            error)) {
      return false;
    }
    out.info = request.info;
  }

  if (derived_primary_ref_none) {
    out.update_map = true;
    out.temporal_update = false;
  } else {
    out.update_map = request.update_map;
    writer.put_bit(out.update_map);
    if (out.update_map && !frame_type_switch) {
      out.temporal_update = request.temporal_update;
      writer.put_bit(out.temporal_update);
    } else {
      out.temporal_update = false;
    }
  }
  *state = out;
  return true;
}

bool read_frame_segmentation_params(BitReader& reader,
                                    const SequenceSegmentConfig& config,
                                    bool derived_primary_ref_none,
                                    bool frame_type_switch,
                                    FrameSegmentState* state,
                                    std::string* error) {
  if (state == nullptr) {
    set_error(error, "null FrameSegmentState");
    return false;
  }
  FrameSegmentState out;
  bool enabled = false;
  if (!reader.read_bit(&enabled)) {
    set_error(error, "truncated segmentation_enabled");
    return false;
  }
  if (!enabled) {
    *state = out;
    return true;
  }
  out.have_sequence_params = config.present;
  const bool allow_change = config.present && config.allow_change;
  bool reuse = config.present;
  if (allow_change) {
    if (!reader.read_bit(&reuse)) {
      set_error(error, "truncated reuse_seg_info");
      return false;
    }
  }
  if (reuse) {
    if (!config.present) {
      set_error(error, "reuse_seg_info without sequence segment info");
      return false;
    }
    out.info = config.info;
    out.reused_sequence = true;
  } else {
    if (!read_segment_info(reader, config.max_segments(), &out.info, error)) {
      return false;
    }
  }
  if (derived_primary_ref_none) {
    out.update_map = true;
    out.temporal_update = false;
  } else {
    if (!reader.read_bit(&out.update_map)) {
      set_error(error, "truncated segmentation_update_map");
      return false;
    }
    if (out.update_map && !frame_type_switch) {
      if (!reader.read_bit(&out.temporal_update)) {
        set_error(error, "truncated segmentation_temporal_update");
        return false;
      }
    } else {
      out.temporal_update = false;
    }
  }
  *state = out;
  return true;
}

// ---------------------------------------------------------------------------
// Superblock / tile helpers
// ---------------------------------------------------------------------------

bool valid_sb_size(SbSize size) {
  return size == SbSize::k64x64 || size == SbSize::k128x128 ||
         size == SbSize::k256x256;
}

int sb_width_4x4(SbSize size) {
  switch (size) {
    case SbSize::k64x64:
      return 16;
    case SbSize::k128x128:
      return 32;
    case SbSize::k256x256:
      return 64;
  }
  return 0;
}

int sb_shift(SbSize size) {
  switch (size) {
    case SbSize::k64x64:
      return 4;
    case SbSize::k128x128:
      return 5;
    case SbSize::k256x256:
      return 6;
  }
  return -1;
}

const char* sb_size_name(SbSize size) {
  switch (size) {
    case SbSize::k64x64:
      return "64x64";
    case SbSize::k128x128:
      return "128x128";
    case SbSize::k256x256:
      return "256x256";
  }
  return "?";
}

SbSize sequence_superblock_size(bool use_256x256_superblock,
                                bool use_128x128_superblock) {
  if (use_256x256_superblock) return SbSize::k256x256;
  if (use_128x128_superblock) return SbSize::k128x128;
  return SbSize::k64x64;
}

int tile_log2(int block_size, int target) {
  if (block_size <= 0) return -1;
  int k = 0;
  while ((block_size << k) < target) {
    ++k;
    if (k > 30) return -1;
  }
  return k;
}

bool uniform_eligible(int tile_log2_value, int sb_num) {
  if (tile_log2_value < 0 || tile_log2_value > 30 || sb_num <= 0) return false;
  const int tile_num = 1 << tile_log2_value;
  const int tile_width = (sb_num + tile_num - 1) >> tile_log2_value;
  const int last_tile_width = sb_num - (tile_num - 1) * tile_width;
  return tile_width >= 1 && last_tile_width >= 1;
}

// ---------------------------------------------------------------------------
// tile_params()
// ---------------------------------------------------------------------------

bool write_tile_params(BitWriter& writer, int frame_width, int frame_height,
                       SbSize uniform_sb_size, SbSize sb_size,
                       int seq_level_idx, int seq_tier,
                       const TileParamsRequest& request, TileParams* params,
                       std::string* error) {
  if (params == nullptr) {
    set_error(error, "null TileParams");
    return false;
  }
  if (frame_width <= 0 || frame_height <= 0) {
    set_error(error, "frame dimensions must be positive");
    return false;
  }
  if (frame_width > 65535 || frame_height > 65535) {
    set_error(error, "frame dimensions exceed the 16-bit sequence domain");
    return false;
  }
  if (!valid_sb_size(uniform_sb_size) || !valid_sb_size(sb_size)) {
    set_error(error, "invalid superblock size");
    return false;
  }
  const SbGeometry g = geometry_for(frame_width, frame_height, sb_size);
  TileParams out;
  out.frame_width = frame_width;
  out.frame_height = frame_height;
  out.uniform_sb_size = uniform_sb_size;
  out.sb_size = sb_size;
  out.sb_cols = g.sb_cols;
  out.sb_rows = g.sb_rows;
  if (!derive_tile_limits(g, seq_level_idx, seq_tier, &out, error)) {
    return false;
  }
  const std::size_t start = writer.bit_size();

  const bool non_uniform =
      !request.width_in_sbs.empty() || !request.height_in_sbs.empty();
  if (non_uniform) {
    if (request.width_in_sbs.empty() || request.height_in_sbs.empty()) {
      set_error(error, "non-uniform spacing needs both width and height lists");
      return false;
    }
    if (request.width_in_sbs.size() > static_cast<std::size_t>(kMaxTileCols) ||
        request.height_in_sbs.size() > static_cast<std::size_t>(kMaxTileRows)) {
      set_error(error, "non-uniform spacing exceeds MAX_TILE_COLS/ROWS");
      return false;
    }
    int sum = 0;
    for (int width : request.width_in_sbs) {
      if (width < 1 || width > out.max_tile_width_sb) {
        set_error(error, "tile width outside 1..maxTileWidthSb");
        return false;
      }
      sum += width;
    }
    if (sum != g.sb_cols) {
      set_error(error, "non-uniform tile widths do not cover sbCols");
      return false;
    }
    writer.put_bit(false);  // uniform_tile_spacing_flag
    int widest = 1;
    int start_sb = 0;
    for (int width : request.width_in_sbs) {
      out.sb_col_starts.push_back(start_sb);
      const unsigned n = static_cast<unsigned>(
          std::min(g.sb_cols - start_sb, out.max_tile_width_sb));
      put_ns(writer, n, static_cast<std::uint32_t>(width - 1));
      widest = std::max(width, widest);
      start_sb += width;
    }
    out.tile_cols = static_cast<int>(out.sb_col_starts.size());
    out.tile_cols_log2 = tile_log2(1, out.tile_cols);
    const int max_tile_area_sb =
        out.min_log2_tiles > 0
            ? (g.sb_rows * g.sb_cols) >> (out.min_log2_tiles + 1)
            : g.sb_rows * g.sb_cols;
    const int max_tile_height_sb = std::max(max_tile_area_sb / widest, 1);
    sum = 0;
    start_sb = 0;
    for (int height : request.height_in_sbs) {
      if (height < 1 || height > max_tile_height_sb) {
        set_error(error, "tile height outside 1..maxTileHeightSb");
        return false;
      }
      sum += height;
    }
    if (sum != g.sb_rows) {
      set_error(error, "non-uniform tile heights do not cover sbRows");
      return false;
    }
    for (int height : request.height_in_sbs) {
      out.sb_row_starts.push_back(start_sb);
      const unsigned n = static_cast<unsigned>(
          std::min(g.sb_rows - start_sb, max_tile_height_sb));
      put_ns(writer, n, static_cast<std::uint32_t>(height - 1));
      start_sb += height;
    }
    out.tile_rows = static_cast<int>(out.sb_row_starts.size());
    out.tile_rows_log2 = tile_log2(1, out.tile_rows);
    out.uniform_spacing = false;
    out.sb_shift = g.sb_shift_value;
  } else {
    int tile_cols_log2 = request.tile_cols_log2;
    int tile_rows_log2 = request.tile_rows_log2;
    if (tile_cols_log2 < 0) tile_cols_log2 = out.min_log2_tile_cols;
    if (tile_cols_log2 < out.min_log2_tile_cols ||
        tile_cols_log2 > out.max_log2_tile_cols) {
      set_error(error, "tile_cols_log2 outside the level/frame limits");
      return false;
    }
    writer.put_bit(true);  // uniform_tile_spacing_flag
    int written_cols_log2 = out.min_log2_tile_cols;
    while (written_cols_log2 < out.max_log2_tile_cols) {
      const bool increment = written_cols_log2 < tile_cols_log2;
      writer.put_bit(increment);
      if (!increment) break;
      ++written_cols_log2;
    }
    if (written_cols_log2 != tile_cols_log2) {
      set_error(error, "requested tile_cols_log2 is not reachable");
      return false;
    }
    int tile_cols = 0;
    out.sb_col_starts =
        uniform_starts(g.mi_cols, uniform_sb_size, tile_cols_log2, &tile_cols);
    out.tile_cols = tile_cols;
    out.tile_cols_log2 = tile_log2(1, tile_cols);
    if (out.tile_cols <= 0) {
      set_error(error, "uniform spacing produced zero tile columns");
      return false;
    }

    const int min_log2_tile_rows =
        std::max(out.min_log2_tiles - out.tile_cols_log2, 0);
    if (tile_rows_log2 < 0) tile_rows_log2 = min_log2_tile_rows;
    if (tile_rows_log2 < min_log2_tile_rows ||
        tile_rows_log2 > out.max_log2_tile_rows) {
      set_error(error, "tile_rows_log2 outside the level/frame limits");
      return false;
    }
    int written_rows_log2 = min_log2_tile_rows;
    while (written_rows_log2 < out.max_log2_tile_rows) {
      const bool increment = written_rows_log2 < tile_rows_log2;
      writer.put_bit(increment);
      if (!increment) break;
      ++written_rows_log2;
    }
    if (written_rows_log2 != tile_rows_log2) {
      set_error(error, "requested tile_rows_log2 is not reachable");
      return false;
    }
    int tile_rows = 0;
    out.sb_row_starts =
        uniform_starts(g.mi_rows, uniform_sb_size, tile_rows_log2, &tile_rows);
    out.tile_rows = tile_rows;
    out.tile_rows_log2 = tile_log2(1, tile_rows);
    if (out.tile_rows <= 0) {
      set_error(error, "uniform spacing produced zero tile rows");
      return false;
    }
    out.uniform_spacing = true;
    out.sb_shift = sb_shift(uniform_sb_size);
  }
  if (!validate_tile_starts(out.sb_col_starts, g.sb_cols, out.tile_cols,
                            error) ||
      !validate_tile_starts(out.sb_row_starts, g.sb_rows, out.tile_rows,
                            error)) {
    return false;
  }
  out.syntax_bits = writer.bit_size() - start;
  *params = std::move(out);
  return true;
}

bool read_tile_params(BitReader& reader, int frame_width, int frame_height,
                      SbSize uniform_sb_size, SbSize sb_size,
                      int seq_level_idx, int seq_tier, TileParams* params,
                      std::string* error) {
  if (params == nullptr) {
    set_error(error, "null TileParams");
    return false;
  }
  if (frame_width <= 0 || frame_height <= 0) {
    set_error(error, "frame dimensions must be positive");
    return false;
  }
  if (!valid_sb_size(uniform_sb_size) || !valid_sb_size(sb_size)) {
    set_error(error, "invalid superblock size");
    return false;
  }
  const SbGeometry g = geometry_for(frame_width, frame_height, sb_size);
  TileParams out;
  out.frame_width = frame_width;
  out.frame_height = frame_height;
  out.uniform_sb_size = uniform_sb_size;
  out.sb_size = sb_size;
  out.sb_cols = g.sb_cols;
  out.sb_rows = g.sb_rows;
  if (!derive_tile_limits(g, seq_level_idx, seq_tier, &out, error)) {
    return false;
  }
  const std::size_t start = reader.bit_pos();
  bool uniform = false;
  if (!reader.read_bit(&uniform)) {
    set_error(error, "truncated uniform_tile_spacing_flag");
    return false;
  }
  if (uniform) {
    int tile_cols_log2 = out.min_log2_tile_cols;
    while (tile_cols_log2 < out.max_log2_tile_cols) {
      bool increment = false;
      if (!reader.read_bit(&increment)) {
        set_error(error, "truncated increment_tile_cols_log2");
        return false;
      }
      if (!increment) break;
      ++tile_cols_log2;
    }
    int tile_cols = 0;
    out.sb_col_starts =
        uniform_starts(g.mi_cols, uniform_sb_size, tile_cols_log2, &tile_cols);
    out.tile_cols = tile_cols;
    out.tile_cols_log2 = tile_log2(1, tile_cols);
    const int min_log2_tile_rows =
        std::max(out.min_log2_tiles - out.tile_cols_log2, 0);
    int tile_rows_log2 = min_log2_tile_rows;
    while (tile_rows_log2 < out.max_log2_tile_rows) {
      bool increment = false;
      if (!reader.read_bit(&increment)) {
        set_error(error, "truncated increment_tile_rows_log2");
        return false;
      }
      if (!increment) break;
      ++tile_rows_log2;
    }
    int tile_rows = 0;
    out.sb_row_starts =
        uniform_starts(g.mi_rows, uniform_sb_size, tile_rows_log2, &tile_rows);
    out.tile_rows = tile_rows;
    out.tile_rows_log2 = tile_log2(1, tile_rows);
    out.sb_shift = sb_shift(uniform_sb_size);
  } else {
    int start_sb = 0;
    int widest = 1;
    while (start_sb < g.sb_cols) {
      out.sb_col_starts.push_back(start_sb);
      const unsigned n = static_cast<unsigned>(
          std::min(g.sb_cols - start_sb, out.max_tile_width_sb));
      std::uint32_t value = 0;
      if (!reader.read_ns(n == 0U ? 1U : n, &value)) {
        set_error(error, "truncated width_in_sbs_minus_1");
        return false;
      }
      const int width = static_cast<int>(value) + 1;
      widest = std::max(width, widest);
      start_sb += width;
      if (static_cast<int>(out.sb_col_starts.size()) > kMaxTileCols) {
        set_error(error, "too many tile columns");
        return false;
      }
    }
    out.tile_cols = static_cast<int>(out.sb_col_starts.size());
    out.tile_cols_log2 = tile_log2(1, out.tile_cols);
    const int max_tile_area_sb =
        out.min_log2_tiles > 0
            ? (g.sb_rows * g.sb_cols) >> (out.min_log2_tiles + 1)
            : g.sb_rows * g.sb_cols;
    const int max_tile_height_sb = std::max(max_tile_area_sb / widest, 1);
    start_sb = 0;
    while (start_sb < g.sb_rows) {
      out.sb_row_starts.push_back(start_sb);
      const unsigned n = static_cast<unsigned>(
          std::min(g.sb_rows - start_sb, max_tile_height_sb));
      std::uint32_t value = 0;
      if (!reader.read_ns(n == 0U ? 1U : n, &value)) {
        set_error(error, "truncated height_in_sbs_minus_1");
        return false;
      }
      const int height = static_cast<int>(value) + 1;
      start_sb += height;
      if (static_cast<int>(out.sb_row_starts.size()) > kMaxTileRows) {
        set_error(error, "too many tile rows");
        return false;
      }
    }
    out.tile_rows = static_cast<int>(out.sb_row_starts.size());
    out.tile_rows_log2 = tile_log2(1, out.tile_rows);
    out.sb_shift = g.sb_shift_value;
  }
  out.uniform_spacing = uniform;
  if (!validate_tile_starts(out.sb_col_starts, g.sb_cols, out.tile_cols,
                            error) ||
      !validate_tile_starts(out.sb_row_starts, g.sb_rows, out.tile_rows,
                            error)) {
    return false;
  }
  out.syntax_bits = reader.bit_pos() - start;
  *params = std::move(out);
  return true;
}

// ---------------------------------------------------------------------------
// sequence_tile_config() / tile_info() reuse
// ---------------------------------------------------------------------------

bool validate_sequence_tile_config(const SequenceTileConfig& config,
                                   int max_frame_width, int max_frame_height,
                                   SbSize sb_size, int seq_level_idx,
                                   int seq_tier, std::string* error) {
  if (!config.present) {
    if (config.allow_change) {
      set_error(error,
                "allow_tile_info_change is not coded when "
                "seq_tile_info_present_flag is 0");
      return false;
    }
    return true;
  }
  if (config.params.uniform_sb_size != sb_size) {
    set_error(error,
              "sequence tile params were computed for a different "
              "get_seq_sb_size()");
    return false;
  }
  const SbGeometry g = geometry_for(max_frame_width, max_frame_height, sb_size);
  if (config.params.sb_cols != g.sb_cols ||
      config.params.sb_rows != g.sb_rows) {
    set_error(error, "sequence tile params disagree with the maximum frame size");
    return false;
  }
  if (!validate_tile_starts(config.params.sb_col_starts, g.sb_cols,
                            config.params.tile_cols, error) ||
      !validate_tile_starts(config.params.sb_row_starts, g.sb_rows,
                            config.params.tile_rows, error)) {
    return false;
  }
  if (config.params.tile_cols_log2 != tile_log2(1, config.params.tile_cols) ||
      config.params.tile_rows_log2 != tile_log2(1, config.params.tile_rows)) {
    set_error(error, "sequence tile log2 values disagree with the tile counts");
    return false;
  }
  (void)seq_level_idx;
  (void)seq_tier;
  return true;
}

bool write_sequence_tile_config(BitWriter& writer,
                                SequenceTileConfig* config,
                                int max_frame_width, int max_frame_height,
                                SbSize sb_size, int seq_level_idx,
                                int seq_tier, std::string* error) {
  if (config == nullptr) {
    set_error(error, "null SequenceTileConfig");
    return false;
  }
  if (!config->present && config->allow_change) {
    set_error(error,
              "allow_tile_info_change is not coded when "
              "seq_tile_info_present_flag is 0");
    return false;
  }
  writer.put_bit(config->present);
  if (!config->present) return true;
  writer.put_bit(config->allow_change);
  if (!write_tile_params(writer, max_frame_width, max_frame_height, sb_size,
                         sb_size, seq_level_idx, seq_tier, config->request,
                         &config->params, error)) {
    return false;
  }
  return validate_sequence_tile_config(*config, max_frame_width,
                                       max_frame_height, sb_size, seq_level_idx,
                                       seq_tier, error);
}

bool read_sequence_tile_config(BitReader& reader,
                               SequenceTileConfig* config,
                               int max_frame_width, int max_frame_height,
                               SbSize sb_size, int seq_level_idx,
                               int seq_tier, std::string* error) {
  if (config == nullptr) {
    set_error(error, "null SequenceTileConfig");
    return false;
  }
  SequenceTileConfig out;
  if (!reader.read_bit(&out.present)) {
    set_error(error, "truncated seq_tile_info_present_flag");
    return false;
  }
  if (out.present) {
    if (!reader.read_bit(&out.allow_change)) {
      set_error(error, "truncated allow_tile_info_change");
      return false;
    }
    if (!read_tile_params(reader, max_frame_width, max_frame_height, sb_size,
                          sb_size, seq_level_idx, seq_tier, &out.params,
                          error)) {
      return false;
    }
  }
  *config = out;
  return true;
}

bool write_frame_tile_info(BitWriter& writer,
                           const SequenceTileConfig& sequence,
                           int frame_width, int frame_height, SbSize sb_size,
                           int seq_level_idx, int seq_tier,
                           const FrameTileRequest& request,
                           FrameTileInfo* info, std::string* error) {
  if (info == nullptr) {
    set_error(error, "null FrameTileInfo");
    return false;
  }
  if (frame_width <= 0 || frame_height <= 0) {
    set_error(error, "frame dimensions must be positive");
    return false;
  }
  if (!valid_sb_size(sb_size)) {
    set_error(error, "invalid superblock size");
    return false;
  }
  const SbGeometry g = geometry_for(frame_width, frame_height, sb_size);
  const std::size_t start = writer.bit_size();
  FrameTileInfo out;
  const bool have = sequence.present;
  const bool eligible = eligible_sequence_tile_params(sequence, g);
  bool reuse = false;
  if (have && eligible) {
    if (sequence.allow_change) {
      reuse = request.reuse_sequence;
      writer.put_bit(reuse);
    } else {
      reuse = true;
    }
  }
  if (reuse) {
    if (!reuse_tile_params(sequence.params, g, sb_size, &out.params, error)) {
      return false;
    }
    out.reused_sequence = true;
  } else {
    if (!write_tile_params(writer, frame_width, frame_height,
                           sequence.present ? sequence.params.uniform_sb_size
                                            : sb_size,
                           sb_size, seq_level_idx, seq_tier, request.params,
                           &out.params, error)) {
      return false;
    }
  }
  out.syntax_bits = writer.bit_size() - start;
  *info = std::move(out);
  return true;
}

bool read_frame_tile_info(BitReader& reader,
                          const SequenceTileConfig& sequence,
                          int frame_width, int frame_height, SbSize sb_size,
                          int seq_level_idx, int seq_tier, FrameTileInfo* info,
                          std::string* error) {
  if (info == nullptr) {
    set_error(error, "null FrameTileInfo");
    return false;
  }
  const SbGeometry g = geometry_for(frame_width, frame_height, sb_size);
  const std::size_t start = reader.bit_pos();
  FrameTileInfo out;
  const bool have = sequence.present;
  const bool eligible = eligible_sequence_tile_params(sequence, g);
  bool reuse = false;
  if (have && eligible) {
    if (sequence.allow_change) {
      if (!reader.read_bit(&reuse)) {
        set_error(error, "truncated reuse_tile_info");
        return false;
      }
    } else {
      reuse = true;
    }
  }
  if (reuse) {
    if (!reuse_tile_params(sequence.params, g, sb_size, &out.params, error)) {
      return false;
    }
    out.reused_sequence = true;
  } else {
    const SbSize uniform_sb =
        sequence.present ? sequence.params.uniform_sb_size : sb_size;
    if (!read_tile_params(reader, frame_width, frame_height, uniform_sb,
                          sb_size, seq_level_idx, seq_tier, &out.params,
                          error)) {
      return false;
    }
  }
  out.syntax_bits = reader.bit_pos() - start;
  *info = std::move(out);
  return true;
}

}  // namespace av2enc::syntax::small_header

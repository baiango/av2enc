#include "core/conformance/conformance.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <utility>

namespace av2enc::conformance {
namespace {

constexpr int kMiSize = 4;
constexpr int kMaxTileWidth = 4096;              // MAX_TILE_WIDTH, p35
constexpr int64_t kMaxTileArea = 4096LL * 2304;  // MAX_TILE_AREA, p35
constexpr int64_t kParallelismRate = 547430400;  // p1110

int log2_exact(int value) {
  if (value <= 0) {
    return -1;
  }
  int k = 0;
  while ((1 << k) < value) {
    ++k;
    if (k > 30) {
      return -1;
    }
  }
  return (1 << k) == value ? k : -1;
}

std::string number(int64_t value) { return std::to_string(value); }

std::string decimal(double value, int precision = 3) {
  std::ostringstream out;
  out.precision(precision);
  out << std::fixed << value;
  return out.str();
}

std::string percent_rate(double observed, double limit) {
  if (limit <= 0.0) {
    return "?";
  }
  return decimal(100.0 * observed / limit, 1) + "%";
}

// §5.18.7.5 uniform_spacing() starts. Mirrors the geometry planner so the
// validator and the writer agree on tile boundaries. Returns the actual starts;
// a very small frame can produce fewer starts than 1 << tile_log2.
std::vector<int> uniform_starts(int mi, int sb_size, int tile_log2_value) {
  const int sb4x4 = sb_size / 4;
  const int sb_shift = log2_exact(sb4x4);
  if (sb_shift < 0) {
    return {};
  }
  const int sbs = (mi + sb4x4 - 1) >> sb_shift;
  const int full_sbs = mi >> sb_shift;
  const int tile_sb = full_sbs >> tile_log2_value;
  const int extra =
      (tile_sb == 0) ? sbs : full_sbs - (tile_sb << tile_log2_value);

  std::vector<int> starts;
  const int tile_count = 1 << tile_log2_value;
  int start = 0;
  for (int i = 0; i < tile_count && start < sbs; ++i) {
    starts.push_back(start);
    start += tile_sb;
    if (i < extra) {
      ++start;
    }
  }
  return starts;
}

// §5.18.7.3 TileWidth/TileHeight in luma samples.
std::vector<int> tile_extents(const std::vector<int>& starts, int mi_total,
                              int sb_size) {
  const int sb4x4 = sb_size / 4;
  std::vector<int> extents;
  extents.reserve(starts.size());
  for (std::size_t i = 0; i < starts.size(); ++i) {
    const int mi_start = starts[i] * sb4x4;
    const int mi_end = (i + 1 < starts.size())
                           ? starts[i + 1] * sb4x4
                           : mi_total;
    extents.push_back((mi_end - mi_start) * kMiSize);
  }
  return extents;
}

int64_t decoded_ref_frames(const Config& config) {
  if (config.single_picture_header_flag) {
    return 2;  // §5.4.1: single_picture_header_flag infers NumRefFrames = 2.
  }
  if (!config.explicit_num_ref_frames) {
    return 8;  // §5.4.6: inferred when explicit_num_ref_frames == 0.
  }
  return config.num_ref_frames;
}

bool is_video(const Config& config) {
  return !config.still_picture && !config.single_picture_header_flag;
}

int derived_interop_point(const Config& config) {
  if (config.interop_point >= 0) {
    return config.interop_point;
  }
  const ProfileInfo* profile = find_profile(config.seq_profile_idc);
  return profile == nullptr ? -1 : profile->interop_point;
}

}  // namespace

// ---------------------------------------------------------------------------
// TileGrid
// ---------------------------------------------------------------------------

TileGrid TileGrid::uniform(int width, int height, int cols, int rows,
                           int sb_size) {
  TileGrid grid;
  grid.cols = cols;
  grid.rows = rows;
  if (width <= 0 || height <= 0 || cols <= 0 || rows <= 0) {
    grid.representable = false;
    return grid;
  }
  const int cols_log2 = log2_exact(cols);
  const int rows_log2 = log2_exact(rows);
  const int sb4x4 = sb_size / 4;
  if (cols_log2 < 0 || rows_log2 < 0 || sb4x4 <= 0) {
    grid.representable = false;
    return grid;
  }
  const int mi_cols = 2 * ((width + 7) >> 3);
  const int mi_rows = 2 * ((height + 7) >> 3);
  const std::vector<int> col_starts =
      uniform_starts(mi_cols, sb_size, cols_log2);
  const std::vector<int> row_starts =
      uniform_starts(mi_rows, sb_size, rows_log2);
  if (static_cast<int>(col_starts.size()) != cols ||
      static_cast<int>(row_starts.size()) != rows) {
    grid.representable = false;
    return grid;
  }
  const std::vector<int> col_extents =
      tile_extents(col_starts, mi_cols, sb_size);
  const std::vector<int> row_extents =
      tile_extents(row_starts, mi_rows, sb_size);
  // Raster order: index = row * cols + col, one entry per tile.
  grid.tile_widths.reserve(static_cast<std::size_t>(cols) *
                           static_cast<std::size_t>(rows));
  grid.tile_heights.reserve(static_cast<std::size_t>(cols) *
                            static_cast<std::size_t>(rows));
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      grid.tile_widths.push_back(
          col_extents[static_cast<std::size_t>(col)]);
      grid.tile_heights.push_back(
          row_extents[static_cast<std::size_t>(row)]);
    }
  }
  return grid;
}

int64_t TileGrid::tile_count() const {
  return static_cast<int64_t>(cols) * static_cast<int64_t>(rows);
}

int64_t TileGrid::max_tile_area() const {
  if (tile_widths.size() != tile_heights.size()) {
    return 0;
  }
  int64_t best = 0;
  for (std::size_t i = 0; i < tile_widths.size(); ++i) {
    const int64_t area = static_cast<int64_t>(tile_widths[i]) *
                         static_cast<int64_t>(tile_heights[i]);
    best = std::max(best, area);
  }
  return best;
}

bool TileGrid::structurally_valid(std::string* why) const {
  if (cols <= 0 || rows <= 0) {
    if (why != nullptr) {
      *why = "tile grid has a non-positive dimension";
    }
    return false;
  }
  if (cols > 64 || rows > 64) {
    if (why != nullptr) {
      *why = "tile grid exceeds MAX_TILE_COLS / MAX_TILE_ROWS (64x64)";
    }
    return false;
  }
  if (log2_exact(cols) < 0 || log2_exact(rows) < 0) {
    if (why != nullptr) {
      *why = "tile grid dimensions are not powers of two";
    }
    return false;
  }
  if (tile_widths.empty() && tile_heights.empty()) {
    if (cols != 1 || rows != 1) {
      if (why != nullptr) {
        *why = "tile grid is missing its tile dimensions";
      }
      return false;
    }
    return true;
  }
  if (static_cast<int>(tile_widths.size()) != cols * rows ||
      static_cast<int>(tile_heights.size()) != cols * rows) {
    if (why != nullptr) {
      *why = "tile grid does not have cols * rows entries";
    }
    return false;
  }
  for (int width : tile_widths) {
    if (width <= 0) {
      if (why != nullptr) {
        *why = "tile grid has a non-positive tile width";
      }
      return false;
    }
  }
  for (int height : tile_heights) {
    if (height <= 0) {
      if (why != nullptr) {
        *why = "tile grid has a non-positive tile height";
      }
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Level resolution
// ---------------------------------------------------------------------------

ResolvedLevel resolve_level(const Config& config) {
  ResolvedLevel out;
  out.seq_level_idx = config.seq_level_idx;
  out.seq_tier = config.seq_tier;
  if (level_is_maximum(config.seq_level_idx)) {
    out.maximum_parameters = true;
    return out;
  }
  const LevelInfo* level = find_level(config.seq_level_idx);
  if (level == nullptr || level_is_reserved(config.seq_level_idx)) {
    return out;
  }
  const TileScaleFactor factor =
      tile_scale_factor(config.seq_level_idx, config.seq_tier);
  if (factor.reserved || config.seq_tier < 0 || config.seq_tier > 1) {
    return out;
  }

  out.info = level;
  out.valid = true;
  out.width_scaling_factor = factor.width;
  out.area_scaling_factor = factor.area;
  out.max_tile_width_luma =
      static_cast<int64_t>(factor.width) * kMaxTileWidth / 4;
  out.max_tile_area_luma =
      static_cast<int64_t>(factor.area) * kMaxTileArea / 4;
  out.max_num_frame_headers_per_sec =
      static_cast<int64_t>(level->max_header_rate) *
      (1 + (static_cast<int64_t>(config.seq_tier) << 1));
  out.max_tiles = level->max_tiles;
  out.max_tile_cols = level->max_tile_cols;
  out.min_comp_basis = config.seq_tier == 0
                           ? static_cast<double>(level->main_cr)
                           : static_cast<double>(level->high_cr);

  const double mbps =
      config.seq_tier == 0 ? level->main_mbps : level->high_mbps;
  int profile_factor = 0;
  int pic_factor = 0;
  double bitrate_factor = 0.0;
  out.profile_scaling_known =
      profile_scaling(config.seq_profile_idc, &profile_factor, &pic_factor,
                      &bitrate_factor);
  const double used_bitrate_factor =
      out.profile_scaling_known ? bitrate_factor : 1.0;  // last-resort display
  out.max_bitrate_bps = mbps * 1.0e6 * used_bitrate_factor;
  out.max_buffer_size_bits = out.max_bitrate_bps;  // ×1 second

  // Informational (no DecodeCount == 2 reservation). max_level_ref_frames()
  // applies the reservation before the cap, as p1109 specifies.
  const int64_t frame_pixels =
      static_cast<int64_t>(config.width) * static_cast<int64_t>(config.height);
  if (frame_pixels > 0) {
    int64_t base = (8 * level->max_pic_size) / frame_pixels;
    const int64_t cap =
        static_cast<int64_t>(8) << (config.explicit_num_ref_frames ? 1 : 0);
    base = std::max<int64_t>(0, std::min(base, cap));
    out.max_level_ref_frames = static_cast<int>(base);
  }
  return out;
}

int64_t max_level_ref_frames(const Config& config,
                             bool decode_count_2_lowers_limit) {
  const ResolvedLevel level = resolve_level(config);
  if (!level.valid) {
    return -1;
  }
  const int64_t frame_pixels =
      static_cast<int64_t>(config.width) * static_cast<int64_t>(config.height);
  if (frame_pixels <= 0) {
    return -1;
  }
  // p1109: Min((8 * MaxPicSize) / (max_frame_width * max_frame_height)
  //         [- 1 for the DecodeCount == 2 reservation],
  //         (8 << explicit_num_ref_frames)).
  int64_t base = (8 * level.info->max_pic_size) / frame_pixels;
  if (decode_count_2_lowers_limit) {
    base -= 1;
  }
  const int64_t cap =
      static_cast<int64_t>(8) << (config.explicit_num_ref_frames ? 1 : 0);
  return std::max<int64_t>(0, std::min(base, cap));
}

int64_t max_display_luma_samples_per_second(const Config& config) {
  const ResolvedLevel level = resolve_level(config);
  return level.valid ? level.info->max_display_rate : -1;
}

int64_t max_luma_samples_for_parsing_time(const Config& config,
                                          double frame_parsing_time) {
  const ResolvedLevel level = resolve_level(config);
  if (!level.valid || !(frame_parsing_time > 0.0)) {
    return -1;
  }
  return static_cast<int64_t>(
      std::floor(frame_parsing_time *
                 static_cast<double>(level.info->max_decode_rate)));
}

int64_t max_tiles_for_parsing_time(const Config& config,
                                   double frame_parsing_time) {
  const ResolvedLevel level = resolve_level(config);
  if (!level.valid || !(frame_parsing_time > 0.0)) {
    return -1;
  }
  const double scaled =
      static_cast<double>(level.max_tiles) * 120.0 * frame_parsing_time;
  const double bounded = std::max(1.0, scaled);
  const double limit = std::min(static_cast<double>(level.max_tiles), bounded);
  return static_cast<int64_t>(std::floor(limit));
}

double max_compressed_size_bytes(const Config& config, int64_t luma_samples,
                                 double frame_parsing_time) {
  const ResolvedLevel level = resolve_level(config);
  int pic_factor = 0;
  if (!level.valid || !(frame_parsing_time > 0.0) || luma_samples < 0 ||
      !profile_scaling(config.seq_profile_idc, nullptr, &pic_factor, nullptr) ||
      level.min_comp_basis <= 0.0) {
    return 0.0;
  }
  // p1110: Min((LumaSampleCount * PicSizeProfileFactor >> 3) * 1.25,
  //           (FrameParsingTime * MaxDecodeRate * PicSizeProfileFactor >> 3)
  //           / MinCompBasis)
  const int64_t first_term_int = (luma_samples * pic_factor) >> 3;
  const double first_term = static_cast<double>(first_term_int) * 1.25;
  const double decode_samples =
      std::floor(frame_parsing_time *
                 static_cast<double>(level.info->max_decode_rate) *
                 static_cast<double>(pic_factor) / 8.0);
  const double second_term = decode_samples / level.min_comp_basis;
  return std::min(first_term, second_term);
}

double max_frame_symbol_count(const Config& config,
                              double frame_parsing_time) {
  const ResolvedLevel level = resolve_level(config);
  int pic_factor = 0;
  if (!level.valid || !(frame_parsing_time > 0.0) ||
      !profile_scaling(config.seq_profile_idc, nullptr, &pic_factor, nullptr) ||
      level.min_comp_basis <= 0.0) {
    return 0.0;
  }
  const double per_symbol =
      8.0 / (9.0 * level.min_comp_basis) + 1.0 / 48.0;
  return frame_parsing_time *
         static_cast<double>(level.info->max_decode_rate) *
         static_cast<double>(pic_factor) * per_symbol;
}

int64_t max_tile_size_luma_samples(const Config& config,
                                   int64_t frame_headers_per_second) {
  const ResolvedLevel level = resolve_level(config);
  if (!level.valid || frame_headers_per_second <= 0) {
    return 0;
  }
  // p1110: MaxTileSizeInLumaSamples * NumFrameHeadersPerSec <=
  //        (ASF * 547,430,400) / 4.
  const int64_t numerator =
      static_cast<int64_t>(level.area_scaling_factor) * kParallelismRate / 4;
  return numerator / frame_headers_per_second;
}

double frame_parsing_time_from_fps(double fps, bool decode_count_2) {
  if (!(fps > 0.0)) {
    return 0.0;
  }
  const double base = 1.0 / fps;
  return decode_count_2 ? base / 2.0 : base;
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

bool Result::conformant() const {
  for (const Violation& violation : violations) {
    if (violation.severity == Severity::kError) {
      return false;
    }
  }
  return true;
}

bool Result::has_check(std::string_view check) const {
  for (const Violation& violation : violations) {
    if (std::string_view(violation.check) == check) {
      return true;
    }
  }
  return false;
}

int Result::error_count() const {
  int count = 0;
  for (const Violation& violation : violations) {
    if (violation.severity == Severity::kError) {
      ++count;
    }
  }
  return count;
}

int Result::warning_count() const {
  int count = 0;
  for (const Violation& violation : violations) {
    if (violation.severity == Severity::kWarning) {
      ++count;
    }
  }
  return count;
}

std::string Result::summary() const {
  std::ostringstream out;
  out << error_count() << " error(s), " << warning_count()
      << " warning(s), " << not_evaluated.size() << " check(s) not evaluated";
  return out.str();
}

// ---------------------------------------------------------------------------
// validate()
// ---------------------------------------------------------------------------

Result validate(const Config& config, const StreamStats& stats) {
  Result result;
  auto add = [&result](std::string check, std::string section, int page,
                       Severity severity, std::string detail) {
    Violation violation;
    violation.check = std::move(check);
    violation.section = std::move(section);
    violation.page = page;
    violation.severity = severity;
    violation.detail = std::move(detail);
    result.violations.push_back(std::move(violation));
  };
  auto skip = [&result](std::string check) {
    result.not_evaluated.push_back(std::move(check));
  };

  // -------------------------------------------------------------------------
  // §A.2 profiles (Table A.1/A.2, p1101-1102)
  // -------------------------------------------------------------------------
  const ProfileInfo* profile = find_profile(config.seq_profile_idc);
  if (profile == nullptr) {
    add("A.2.ProfileRange", "A.2", 1101, Severity::kError,
        "seq_profile_idc = " + number(config.seq_profile_idc) +
            " is outside 0..31");
  } else if (profile->reserved) {
    add("A.2.ReservedProfile", "A.2", 1101, Severity::kError,
        "seq_profile_idc = " + number(config.seq_profile_idc) +
            " is Reserved in Table A.1 (5..30 must not be signalled)");
  } else {
    if (!profile_allows_chroma(*profile, config.chroma_format_idc)) {
      add("A.2.ProfileChroma", "A.2", 1101, Severity::kError,
          std::string(profile_name(config.seq_profile_idc)) +
              " does not allow chroma_format_idc = " +
              number(config.chroma_format_idc));
    }
  }

  if (config.bit_depth_idc != 0 && config.bit_depth_idc != 1) {
    add("A.2.BitDepthIdc", "A.2", 1101, Severity::kError,
        "bit_depth_idc = " + number(config.bit_depth_idc) +
            " is reserved; Table A.1 allows only 0 (10-bit) and 1 (8-bit)");
  }
  if (config.chroma_format_idc < 0 || config.chroma_format_idc > 3) {
    add("A.2.ChromaFormat", "A.2", 1101, Severity::kError,
        "chroma_format_idc = " + number(config.chroma_format_idc) +
            " is outside Table 6.2's 0..3");
  }

  if (config.seq_profile_idc == 31) {
    // §A.2 p1102: for the Configurable profile the constraints are derived
    // from chroma_format_idc, bit_depth_idc and SeqMaxMlayerCnt. There is no
    // Table A.2 row, so the size/symbol/bitrate formulas cannot be evaluated.
    add("A.2.ConfigurableProfile", "A.2", 1102, Severity::kWarning,
        "seq_profile_idc = 31 (Configurable): no Table A.2 scaling factors; "
        "PicSizeProfileFactor/BitrateProfileFactor checks are reported as not "
        "evaluated");
    if (config.seq_max_mlayer_cnt_minus_1 < config.num_embedded_layers - 1) {
      add("A.2.ConfigurableLayers", "A.2", 1102, Severity::kError,
          "SeqMaxMlayerCnt = " +
              number(config.seq_max_mlayer_cnt_minus_1 + 1) +
              " is smaller than the " + number(config.num_embedded_layers) +
              " embedded layers in the stream");
    }
  }

  // -------------------------------------------------------------------------
  // §A.3 interoperability points and multi-sequence configurations
  // (Tables A.3/A.4/A.5/A.6, p1102-1104)
  // -------------------------------------------------------------------------
  const bool multistream = config.num_extended_layers > 1;
  int iop = derived_interop_point(config);
  if (profile != nullptr && !profile->configurable && config.interop_point >= 0 &&
      profile->interop_point >= 0 && config.interop_point != profile->interop_point) {
    add("A.3.InteropProfileMismatch", "A.3", 1102, Severity::kError,
        std::string(profile_name(config.seq_profile_idc)) +
            " implies IOP " + number(profile->interop_point) +
            " but interop_point = " + number(config.interop_point) +
            " was supplied");
  }
  if (iop < 0) {
    if (multistream || config.num_embedded_layers > 1) {
      add("A.3.InteropPointMissing", "A.3", 1102, Severity::kError,
          "a multi-layer stream has no interop point: seq_profile_idc = " +
              number(config.seq_profile_idc) +
              " does not determine one (Table A.1) and interop_point = -1");
    }
  } else if (interop_is_reserved(iop)) {
    add("A.3.ReservedInteropPoint", "A.3", 1102, Severity::kError,
        "interop point " + number(iop) + " is Reserved (3..14)");
  } else {
    const InteropPointInfo* info = find_interop_point(iop);
    if (info != nullptr && !info->reserved) {
      if (config.num_extended_layers < info->min_extended_layers ||
          config.num_extended_layers > info->max_extended_layers) {
        add("A.3.InteropExtendedLayers", "A.3", 1102, Severity::kError,
            "IOP " + number(iop) + " allows " +
                number(info->min_extended_layers) + ".." +
                number(info->max_extended_layers) + " extended layers, got " +
                number(config.num_extended_layers));
      }
      if (config.num_embedded_layers < info->min_embedded_layers ||
          config.num_embedded_layers > info->max_embedded_layers) {
        add("A.3.InteropEmbeddedLayers", "A.3", 1102, Severity::kError,
            "IOP " + number(iop) + " allows " +
                number(info->min_embedded_layers) + ".." +
                number(info->max_embedded_layers) + " embedded layers, got " +
                number(config.num_embedded_layers));
      }
      const int64_t layers =
          static_cast<int64_t>(config.num_extended_layers) *
          static_cast<int64_t>(config.num_embedded_layers);
      if (layers > info->max_layers) {
        add("A.3.InteropLayerCount", "A.3", 1102, Severity::kError,
            "IOP " + number(iop) + " allows at most " +
                number(info->max_layers) + " layers, got " + number(layers));
      }
      if (!info->combination_allowed &&
          config.num_extended_layers > 1 && config.num_embedded_layers > 1) {
        add("A.3.InteropCombination", "A.3", 1102, Severity::kError,
            "IOP " + number(iop) +
                " does not allow the combination of >1 extended and >1 "
                "embedded layers");
      }
    }

    // Table A.4. IOP 2's "MSDO plus Local LCR, or Global LCR" is handled
    // explicitly because it is a disjunction, not a per-OBU flag.
    const bool ext = config.num_extended_layers > 1;
    const bool emb = config.num_embedded_layers > 1;
    if (iop == 0) {
      if (ext && !config.msdo_present) {
        add("A.3.Iop0MsdoRequired", "A.3", 1103, Severity::kError,
            "IOP 0 with more than one extended layer requires an MSDO OBU");
      }
      if (!ext && config.msdo_present) {
        add("A.3.Iop0MsdoProhibited", "A.3", 1103, Severity::kError,
            "IOP 0 with one extended layer prohibits an MSDO OBU");
      }
    } else if (iop == 1) {
      if (ext && !emb && !config.msdo_present) {
        add("A.3.Iop1MsdoRequired", "A.3", 1103, Severity::kError,
            "IOP 1 with more than one extended layer requires an MSDO OBU");
      }
      if (!ext && config.msdo_present) {
        add("A.3.Iop1MsdoProhibited", "A.3", 1103, Severity::kError,
            "IOP 1 with one extended layer prohibits an MSDO OBU");
      }
      if (emb && !config.local_lcr_present) {
        add("A.3.Iop1LcrRequired", "A.3", 1103, Severity::kError,
            "IOP 1 with more than one embedded layer requires a local LCR "
            "OBU");
      }
    } else if (iop == 2) {
      if (!ext && !emb) {
        if (config.msdo_present) {
          add("A.3.Iop2MsdoProhibited", "A.3", 1103, Severity::kError,
              "IOP 2 with one extended and one embedded layer prohibits an "
              "MSDO OBU");
        }
      } else if (ext && !emb) {
        if (!config.msdo_present && !config.global_lcr_present) {
          add("A.3.Iop2MsdoOrGlobalLcr", "A.3", 1103, Severity::kError,
              "IOP 2 with more than one extended layer requires MSDO or a "
              "global LCR");
        }
      } else if (!ext && emb) {
        if (!(config.local_lcr_present || config.global_lcr_present)) {
          add("A.3.Iop2LcrRequired", "A.3", 1103, Severity::kError,
              "IOP 2 with more than one embedded layer requires an LCR OBU "
              "(global or local)");
        }
      } else if (!config.global_lcr_present &&
                 !(config.msdo_present && config.local_lcr_present)) {
        add("A.3.Iop2Combination", "A.3", 1103, Severity::kError,
            "IOP 2 with >1 extended and >1 embedded layers requires global "
            "LCR, or MSDO plus local LCR");
      }
    }
  }

  if (config.lcr_config_idc >= 0) {
    if (multi_seq_is_reserved(config.lcr_config_idc)) {
      add("A.3.MultiSeqConfigReserved", "A.3", 1104, Severity::kError,
          "lcr_config_idc = " + number(config.lcr_config_idc) +
              " is Reserved (3..63 in Table A.5)");
    } else {
      const MultiSeqConfigInfo* seq =
          find_multi_seq_config(config.lcr_config_idc);
      if (seq == nullptr) {
        add("A.3.MultiSeqConfigRange", "A.3", 1104, Severity::kError,
            "lcr_config_idc = " + number(config.lcr_config_idc) +
                " is outside 0..63");
      } else {
        if (!multi_seq_allows_profile(*seq, config.seq_profile_idc)) {
          add("A.3.MultiSeqProfile", "A.3", 1104, Severity::kError,
              std::string(seq->label) + " does not allow " +
                  profile_name(config.seq_profile_idc) +
                  " (Table A.6 profile list)");
        }
        if (!multi_seq_allows_chroma(*seq, config.chroma_format_idc)) {
          add("A.3.MultiSeqChroma", "A.3", 1104, Severity::kError,
              std::string(seq->label) + " does not allow chroma_format_idc = " +
                  number(config.chroma_format_idc) + " (Table A.6)");
        }
        if (config.bit_depth_idc != 0 && config.bit_depth_idc != 1) {
          add("A.3.MultiSeqBitDepth", "A.3", 1104, Severity::kError,
              std::string(seq->label) +
                  " allows only bit_depth_idc 0..1 (Table A.6)");
        }
      }
    }
    if (!config.global_lcr_present && !config.local_lcr_present) {
      add("A.3.MultiSeqConfigWithoutObu", "A.3", 1104, Severity::kError,
          "lcr_config_idc = " + number(config.lcr_config_idc) +
              " is set but no LCR OBU is signalled");
    }
  }

  // -------------------------------------------------------------------------
  // §A.4 level/tier legality (Table A.7, p1104-1106)
  // -------------------------------------------------------------------------
  if (config.seq_level_idx < 0 || config.seq_level_idx > 31) {
    add("A.4.LevelRange", "A.4", 1104, Severity::kError,
        "seq_level_idx = " + number(config.seq_level_idx) +
            " is outside 0..31");
  }
  if (config.seq_tier < 0 || config.seq_tier > 1) {
    add("A.4.SeqTierRange", "A.4", 1106, Severity::kError,
        "seq_tier = " + number(config.seq_tier) + " is outside 0..1");
  } else if (config.seq_tier == 1 && config.seq_level_idx < 4) {
    add("A.4.SeqTierLevel", "A.4", 1106, Severity::kError,
        "seq_tier = 1 may only be signalled for level 4.0 and above; "
        "seq_level_idx = " +
            number(config.seq_level_idx) + " is level " +
            level_name(config.seq_level_idx));
  }
  if (config.single_picture_header_flag && config.seq_tier != 0) {
    add("A.4.SeqTierNotSignalled", "A.4", 1104, Severity::kError,
        "single_picture_header_flag = 1 removes seq_tier from the sequence "
        "header (p1104), so seq_tier = " +
            number(config.seq_tier) + " cannot be expressed");
  }

  if (level_is_reserved(config.seq_level_idx)) {
    add("A.4.ReservedLevel", "A.4", 1105, Severity::kError,
        "seq_level_idx = " + number(config.seq_level_idx) +
            " is Reserved (22..30, Table A.7)");
    skip("A.4.* numeric limits (reserved level)");
    return result;
  }
  if (level_is_maximum(config.seq_level_idx)) {
    // p1111: "If seq_level_idx is equal to 31 ... there are no level-based
    // constraints on the bitstream." Profile constraints still apply.
    skip("A.4.* numeric limits (seq_level_idx = 31, maximum parameters)");
    return result;
  }

  const ResolvedLevel level = resolve_level(config);
  if (!level.valid) {
    add("A.4.LevelTileScaling", "A.4", 1110, Severity::kError,
        "no tile scaling factor exists for seq_level_idx = " +
            number(config.seq_level_idx) + ", seq_tier = " +
            number(config.seq_tier));
    skip("A.4.* numeric limits (no tile scaling factor)");
    return result;
  }

  // -------------------------------------------------------------------------
  // §A.4 frame and tile geometry (p1110)
  // -------------------------------------------------------------------------
  if (config.width < 16) {
    add("A.4.MinFrameWidth", "A.4", 1110, Severity::kError,
        "FrameWidth = " + number(config.width) + " < 16");
  }
  if (config.height < 16) {
    add("A.4.MinFrameHeight", "A.4", 1110, Severity::kError,
        "FrameHeight = " + number(config.height) + " < 16");
  }
  const int64_t frame_pixels =
      static_cast<int64_t>(config.width) * static_cast<int64_t>(config.height);
  if (frame_pixels > level.info->max_pic_size) {
    add("A.4.MaxPicSize", "A.4", 1110, Severity::kError,
        "FrameWidth * FrameHeight = " + number(config.width) + " * " +
            number(config.height) + " = " + number(frame_pixels) +
            " > MaxPicSize " + number(level.info->max_pic_size) + " (level " +
            level.info->name + ")");
  }
  if (config.width > level.info->max_hsize) {
    add("A.4.MaxHSize", "A.4", 1110, Severity::kError,
        "FrameWidth = " + number(config.width) + " > MaxHSize " +
            number(level.info->max_hsize) + " (level " + level.info->name +
            ")");
  }
  if (config.height > level.info->max_vsize) {
    add("A.4.MaxVSize", "A.4", 1110, Severity::kError,
        "FrameHeight = " + number(config.height) + " > MaxVSize " +
            number(level.info->max_vsize) + " (level " + level.info->name +
            ")");
  }

  TileGrid grid = config.tiles;
  if (grid.tile_widths.empty() && grid.tile_heights.empty()) {
    // Convenience: cols/rows alone mean a uniform grid. Explicit non-uniform
    // widths must be supplied as matching raster-order vectors.
    grid = TileGrid::uniform(config.width, config.height, grid.cols,
                             grid.rows, config.sb_size);
  }

  if (!grid.representable) {
    add("A.4.TileGridRepresentable", "A.4", 1110, Severity::kError,
        "tile grid " + number(grid.cols) + "x" + number(grid.rows) +
            " is not representable by §5.18.7.5 uniform_spacing() at " +
            number(config.width) + "x" + number(config.height));
  } else {
    std::string grid_error;
    if (!grid.structurally_valid(&grid_error)) {
      add("A.4.TileGridInvalid", "A.4", 1110, Severity::kError,
          "tile geometry is unusable: " + grid_error);
    } else {
      const int64_t tiles = grid.tile_count();
      if (tiles > level.max_tiles) {
        add("A.4.MaxTiles", "A.4", 1110, Severity::kError,
            "NumTiles = " + number(tiles) + " > MaxTiles " +
                number(level.max_tiles) + " (level " + level.info->name + ")");
      }
      if (grid.cols > level.max_tile_cols) {
        add("A.4.MaxTileCols", "A.4", 1110, Severity::kError,
            "TileCols = " + number(grid.cols) + " > MaxTileCols " +
                number(level.max_tile_cols) + " (level " + level.info->name +
                ", Table A.9)");
      }

      int too_wide = 0;
      int too_narrow = 0;
      int too_large = 0;
      int first_wide = -1;
      int first_large = -1;
      for (int row = 0; row < grid.rows; ++row) {
        for (int col = 0; col < grid.cols; ++col) {
          const std::size_t index =
              static_cast<std::size_t>(row) * static_cast<std::size_t>(grid.cols) +
              static_cast<std::size_t>(col);
          const int64_t tile_width = grid.tile_widths[index];
          const int64_t tile_height = grid.tile_heights[index];
          if (tile_width > level.max_tile_width_luma) {
            if (first_wide < 0) {
              first_wide = static_cast<int>(index);
            }
            ++too_wide;
          }
          // RightMostTile == 0 for every column except the last; the last
          // column is clipped at MiCols and is exempt from the 64-sample floor.
          if (col + 1 < grid.cols && tile_width < 64) {
            ++too_narrow;
          }
          if (tile_width * tile_height > level.max_tile_area_luma) {
            if (first_large < 0) {
              first_large = static_cast<int>(index);
            }
            ++too_large;
          }
        }
      }
      if (too_wide > 0) {
        add("A.4.MaxTileWidth", "A.4", 1110, Severity::kError,
            number(too_wide) +
                " tile(s) exceed TileWidth <= Tile_Width_Scaling_Factor * "
                "MAX_TILE_WIDTH / 4 = " +
                number(level.max_tile_width_luma) +
                " (level " + level.info->name + ", tier " +
                number(config.seq_tier) + "); first failing tile index " +
                number(first_wide));
      }
      if (too_narrow > 0) {
        add("A.4.MinTileWidth", "A.4", 1110, Severity::kError,
            number(too_narrow) +
                " non-rightmost tile(s) have TileWidth < 64");
      }
      if (too_large > 0) {
        add("A.4.MaxTileArea", "A.4", 1110, Severity::kError,
            number(too_large) +
                " tile(s) exceed TileWidth * TileHeight <= "
                "Tile_Area_Scaling_Factor * 4096 * 2304 / 4 = " +
                number(level.max_tile_area_luma) + " (level " +
                level.info->name + ", tier " + number(config.seq_tier) +
                "); first failing tile index " + number(first_large));
      }
    }
  }

  // -------------------------------------------------------------------------
  // §A.4 display rate, header rate, reference frames (p1109-1110)
  // -------------------------------------------------------------------------
  const bool video = is_video(config);
  if (!video) {
    skip("A.4.TotalDisplayLumaSampleRate (still picture)");
    skip("A.4.NumFrameHeadersPerSec (still picture)");
    skip("A.4 per-frame decode budget (still picture)");
    skip("A.4.MaxTileSizeInLumaSamples * NumFrameHeadersPerSec (still picture)");
    skip("A.4.MaxBitrate / MaxBufferSize (still picture)");
  } else {
    // TotalDisplayLumaSampleRate.
    int64_t display_samples = stats.tu_display_luma_samples;
    if (display_samples <= 0) {
      const int64_t output_frames =
          std::max<int64_t>(1, stats.output_frames_in_tu);
      display_samples = frame_pixels * output_frames;
    }
    double duration = stats.tu_duration_seconds;
    if (!(duration > 0.0) && stats.fps > 0.0) {
      duration = static_cast<double>(std::max(1, stats.output_frames_in_tu)) /
                 stats.fps;
    }
    if (duration > 0.0) {
      const double rate = static_cast<double>(display_samples) / duration;
      if (rate > static_cast<double>(level.info->max_display_rate)) {
        add("A.4.MaxDisplayRate", "A.4", 1110, Severity::kError,
            "TotalDisplayLumaSampleRate = " + decimal(rate, 1) +
                " > MaxDisplayRate " +
                number(level.info->max_display_rate) + " (" + percent_rate(
                    rate, static_cast<double>(level.info->max_display_rate)) +
                " of the level budget, level " + level.info->name + ")");
      }
    } else {
      skip("A.4.TotalDisplayLumaSampleRate (no duration / fps)");
    }

    // NumFrameHeadersPerSec.
    int64_t headers = stats.max_frame_headers_per_second;
    if (headers <= 0 && stats.fps > 0.0) {
      headers = static_cast<int64_t>(std::ceil(stats.fps));
    }
    if (headers > 0) {
      if (headers > level.max_num_frame_headers_per_sec) {
        add("A.4.NumFrameHeadersPerSec", "A.4", 1110, Severity::kError,
            "NumFrameHeadersPerSec = " + number(headers) +
                " > MaxNumFrameHeadersPerSec = MaxHeaderRate * "
                "(1 + (seq_tier << 1)) = " +
                number(level.max_num_frame_headers_per_sec) + " (level " +
                level.info->name + ", tier " + number(config.seq_tier) + ")");
      }
    } else {
      skip("A.4.NumFrameHeadersPerSec (no measured header count / fps)");
    }

    // MaxTileSizeInLumaSamples * NumFrameHeadersPerSec.
    int64_t max_tile_size = 0;
    for (const FrameSample& frame : stats.frames) {
      if (frame.max_tile_size_luma_samples > 0) {
        max_tile_size =
            std::max(max_tile_size, frame.max_tile_size_luma_samples);
      }
    }
    if (max_tile_size <= 0) {
      max_tile_size = grid.max_tile_area();
    }
    if (max_tile_size > 0 && headers > 0) {
      const int64_t allowed =
          max_tile_size_luma_samples(config, headers);
      if (max_tile_size > allowed) {
        add("A.4.MaxTileSizeHeaderRate", "A.4", 1110, Severity::kError,
            "MaxTileSizeInLumaSamples * NumFrameHeadersPerSec = " +
                number(max_tile_size) + " * " + number(headers) + " = " +
                number(max_tile_size * headers) +
                " > (Tile_Area_Scaling_Factor * 547430400) / 4, which allows "
                "MaxTileSizeInLumaSamples <= " +
                number(allowed) + " at this header rate");
      }
    } else {
      skip("A.4.MaxTileSizeInLumaSamples * NumFrameHeadersPerSec");
    }

    // Per-frame decode budget (p1110, "For a particular Frame with
    // ShowExistingFrame equal to 0").
    if (stats.frames.empty()) {
      skip("A.4.LumaSampleCount (no FrameStats)");
      skip("A.4.NumTiles <= MaxTiles * 120 * FrameParsingTime (no FrameStats)");
      skip("A.4.CompressedSize (no FrameStats)");
      skip("A.4.FrameSymbolCount (no FrameStats)");
    } else {
      bool reported_compressed = false;
      bool reported_symbols = false;
      for (std::size_t i = 0; i < stats.frames.size(); ++i) {
        const FrameSample& frame = stats.frames[i];
        if (frame.show_existing_frame) {
          continue;
        }
        int64_t luma = frame.luma_sample_count;
        if (luma <= 0) {
          luma = frame_pixels;
        }
        const bool decode_count_2 =
            frame.decode_count_2 ||
            (config.allow_global_intrabc && config.inloop_filtering_enabled);
        double parsing_time = frame.frame_parsing_time_seconds;
        if (!(parsing_time > 0.0)) {
          parsing_time =
              frame_parsing_time_from_fps(stats.fps, decode_count_2);
        }
        const std::string where = " (frame " + number(static_cast<int64_t>(i)) +
                                  ")";
        if (parsing_time > 0.0) {
          const int64_t max_luma =
              max_luma_samples_for_parsing_time(config, parsing_time);
          if (max_luma >= 0 && luma > max_luma) {
            add("A.4.LumaSampleCount", "A.4", 1110, Severity::kError,
                "LumaSampleCount = " + number(luma) +
                    " > FrameParsingTime * MaxDecodeRate = " +
                    number(max_luma) + where);
          }
          const int64_t max_tiles =
              max_tiles_for_parsing_time(config, parsing_time);
          if (max_tiles >= 0 && grid.tile_count() > max_tiles) {
            add("A.4.NumTilesParsingTime", "A.4", 1110, Severity::kError,
                "NumTiles = " + number(grid.tile_count()) +
                    " > Min(MaxTiles, Max(1, MaxTiles * 120 * "
                    "FrameParsingTime)) = " +
                    number(max_tiles) + where);
          }
          if (frame.frame_obu_bytes > 0) {
            const int64_t compressed =
                std::max<int64_t>(0, frame.frame_obu_bytes - 128);
            const double allowed =
                max_compressed_size_bytes(config, luma, parsing_time);
            if (allowed > 0.0 && static_cast<double>(compressed) > allowed) {
              add("A.4.CompressedSize", "A.4", 1110, Severity::kError,
                  "CompressedSize = " + number(compressed) +
                      " B (after the 128 B allowance) > " +
                      decimal(allowed, 1) + " B" + where);
            }
          } else if (!reported_compressed) {
            skip("A.4.CompressedSize (frame_obu_bytes not measured)");
            reported_compressed = true;
          }
          if (frame.symbol_count > 0) {
            const double allowed =
                max_frame_symbol_count(config, parsing_time);
            if (allowed > 0.0 &&
                static_cast<double>(frame.symbol_count) > allowed) {
              add("A.4.FrameSymbolCount", "A.4", 1110, Severity::kError,
                  "FrameSymbolCount = " + number(frame.symbol_count) + " > " +
                      decimal(allowed, 1) + where);
            }
          } else if (!reported_symbols) {
            skip("A.4.FrameSymbolCount (symbol_count not measured)");
            reported_symbols = true;
          }
        } else {
          skip("A.4 per-frame decode budget (no FrameParsingTime / fps)");
          break;
        }
      }
    }

    // MaxBitrate and MaxBufferSize (p1110-1111 + p1125 profile factor).
    if (!level.profile_scaling_known) {
      skip("A.4.MaxBitrate (no Table A.2 BitrateProfileFactor)");
      skip("A.4.MaxBufferSize (no Table A.2 BitrateProfileFactor)");
    } else {
      if (stats.max_bitrate_bps_over_1s > 0.0) {
        if (stats.max_bitrate_bps_over_1s > level.max_bitrate_bps) {
          add("A.4.MaxBitrate", "A.4", 1110, Severity::kError,
              "bitrate over the busiest 1 s = " +
                  decimal(stats.max_bitrate_bps_over_1s / 1.0e6, 3) +
                  " Mbit/s > MaxMbps * 1e6 * BitrateProfileFactor = " +
                  decimal(level.max_bitrate_bps / 1.0e6, 3) + " Mbit/s (level " +
                  level.info->name + ", tier " + number(config.seq_tier) + ")");
        }
      } else {
        skip("A.4.MaxBitrate (no measured 1 s bitrate)");
      }
      if (stats.peak_buffer_occupancy_bits > 0.0) {
        if (stats.peak_buffer_occupancy_bits > level.max_buffer_size_bits) {
          add("A.4.MaxBufferSize", "A.4", 1110, Severity::kError,
              "peak smoothing-buffer occupancy = " +
                  decimal(stats.peak_buffer_occupancy_bits, 0) +
                  " bits > MaxBufferSize = MaxBitrate * 1 s * "
                  "BitrateProfileFactor = " +
                  decimal(level.max_buffer_size_bits, 0) + " bits");
        }
      } else {
        skip("A.4.MaxBufferSize (no measured buffer occupancy)");
      }
    }
  }

  // NumRefFrames applies to video streams; a still's inferred value is 2.
  bool lowers_ref_limit = false;
  const bool tools_force_decode_count_2 =
      config.allow_global_intrabc && config.inloop_filtering_enabled;
  for (const FrameSample& frame : stats.frames) {
    const bool decode_count_2 = frame.decode_count_2 || tools_force_decode_count_2;
    if (decode_count_2 && !frame.show_existing_frame &&
        (config.max_mlayer_id != 0 || !frame.closed_loop_key)) {
      lowers_ref_limit = true;
      break;
    }
  }
  if (stats.frames.empty() && tools_force_decode_count_2 &&
      config.max_mlayer_id != 0) {
    lowers_ref_limit = true;
  }
  const int64_t ref_limit =
      max_level_ref_frames(config, lowers_ref_limit);
  if (ref_limit < 0) {
    skip("A.4.MaxLevelRefFrames (level not evaluable)");
  } else {
    const int64_t refs = stats.num_ref_frames > 0
                             ? stats.num_ref_frames
                             : decoded_ref_frames(config);
    if (refs > ref_limit) {
      add("A.4.MaxLevelRefFrames", "A.4", 1109, Severity::kError,
          "NumRefFrames = " + number(refs) + " > MaxLevelRefFrames = " +
              number(ref_limit) + " (8 * MaxPicSize / frame pixels" +
              (lowers_ref_limit ? " - 1 for DecodeCount == 2" : "") +
              ", capped at 8 << explicit_num_ref_frames)");
    }
  }

  // -------------------------------------------------------------------------
  // E.4.4: without timing and the decoder-model flags the claimed level
  // cannot be verified by either operating mode (p1127).
  // -------------------------------------------------------------------------
  if (video) {
    const bool timing = stats.timing_info_signalled ||
                        config.ci_timing_info_present;
    const bool decoder_model = stats.decoder_model_info_signalled ||
                               config.decoder_model_info_present;
    if (!timing || !decoder_model) {
      std::string missing;
      if (!timing) {
        missing += "ci_timing_info_present_flag = 0";
      }
      if (!decoder_model) {
        if (!missing.empty()) {
          missing += ", ";
        }
        missing += "decoder_model_info_present_flag = 0";
      }
      add("E.4.4.LevelConformanceUnverifiable", "E.4.4", 1127,
          Severity::kWarning,
          "level " + std::string(level.info->name) + " conformance cannot be "
          "verified by either decoder-model operating mode: " + missing);
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// Audit coverage: the 33 ABSENT rows of docs/spec-audit/14-conformance.md
// ---------------------------------------------------------------------------

const std::vector<CoverageEntry>& absent_constraint_coverage() {
  static const std::vector<CoverageEntry> kCoverage = {
      // §A.2/A.3 profiles (3 absent)
      {1, "A.2 profiles",
       "ProfileScalingFactor / PicSizeProfileFactor / BitrateProfileFactor "
       "must multiply the level and decoder-model numbers",
       "A.2 / p1102",
       "table:A.2 factors, applied by A.4.CompressedSize, "
       "A.4.FrameSymbolCount, A.4.MaxBitrate, A.4.MaxBufferSize"},
      {2, "A.3 interoperability points",
       "Interop-point layer counts (Table A.3) and mandatory MSDO/LCR mix "
       "(Table A.4)",
       "A.3 / p1102-1103",
       "check:A.3.Interop* (extended/embedded counts, combination, MSDO/LCR)"},
      {3, "A.2 profiles",
       "Configurable profile (31): constraints from chroma_format_idc, "
       "bit_depth_idc, SeqMaxMlayerCnt",
       "A.2 / p1102",
       "check:A.2.ConfigurableProfile, A.2.ConfigurableLayers"},

      // §A.4 levels (16 absent)
      {4, "A.4 levels", "FrameWidth * FrameHeight <= MaxPicSize", "A.4 / p1110",
       "check:A.4.MaxPicSize"},
      {5, "A.4 levels",
       "FrameWidth <= MaxHSize and FrameHeight <= MaxVSize", "A.4 / p1110",
       "check:A.4.MaxHSize, A.4.MaxVSize"},
      {6, "A.4 levels",
       "FrameWidth >= 16 and FrameHeight >= 16", "A.4 / p1110",
       "check:A.4.MinFrameWidth, A.4.MinFrameHeight"},
      {7, "A.4 levels", "NumTiles <= MaxTiles", "A.4 / p1110",
       "check:A.4.MaxTiles"},
      {8, "A.4 levels", "TileCols <= MaxTileCols", "A.4 / p1110",
       "check:A.4.MaxTileCols"},
      {9, "A.4 levels",
       "TileWidth >= 64 when RightMostTile == 0", "A.4 / p1110",
       "check:A.4.MinTileWidth"},
      {10, "A.4 levels",
       "TotalDisplayLumaSampleRate <= MaxDisplayRate", "A.4 / p1110",
       "check:A.4.MaxDisplayRate"},
      {11, "A.4 levels",
       "NumFrameHeadersPerSec <= MaxHeaderRate * (1 + (seq_tier << 1))",
       "A.4 / p1109-1110", "check:A.4.NumFrameHeadersPerSec"},
      {12, "A.4 levels", "NumRefFrames <= MaxLevelRefFrames",
       "A.4 / p1109-1110", "check:A.4.MaxLevelRefFrames"},
      {13, "A.4 levels",
       "LumaSampleCount <= FrameParsingTime * MaxDecodeRate", "A.4 / p1110",
       "check:A.4.LumaSampleCount"},
      {14, "A.4 levels",
       "NumTiles <= Min(MaxTiles, Max(1, MaxTiles*120*FrameParsingTime))",
       "A.4 / p1110", "check:A.4.NumTilesParsingTime"},
      {15, "A.4 levels",
       "CompressedSize <= Min((LumaSampleCount*PicSizeProfileFactor>>3)*1.25, "
       "(FrameParsingTime*MaxDecodeRate*PicSizeProfileFactor>>3)/MinCompBasis)",
       "A.4 / p1110", "check:A.4.CompressedSize"},
      {16, "A.4 levels",
       "FrameSymbolCount <= FrameParsingTime*MaxDecodeRate*"
       "PicSizeProfileFactor*(8/(9*MinCompBasis)+1/48)",
       "A.4 / p1110", "check:A.4.FrameSymbolCount"},
      {17, "A.4 levels",
       "MaxTileSizeInLumaSamples * NumFrameHeadersPerSec <= "
       "ASF*547430400/4",
       "A.4 / p1110", "check:A.4.MaxTileSizeHeaderRate"},
      {18, "A.4 levels",
       "MaxBitrate = MaxMbps * 1e6, MaxBufferSize = MaxBitrate * 1 s",
       "A.4 / p1110-1111, p1125",
       "check:A.4.MaxBitrate, A.4.MaxBufferSize"},
      {19, "A.4 levels",
       "seq_tier = 1 only for level >= 4.0", "A.4 NOTE / p1106",
       "check:A.4.SeqTierLevel, A.4.SeqTierNotSignalled"},

      // Annex D alpha/resampling (7 absent; no emission path)
      {20, "Annex D alpha",
       "Last atlas segment must not be an alpha segment", "§5.9.4 / p79",
       "N/A: no atlas (ats_*) writer exists; the syntax infers 0"},
      {21, "Annex D alpha",
       "ats_num_atlas_segments_minus_1 < MAX_NUM_ATLAS_SEGMENTS; unique "
       "ats_input_stream_id",
       "§6.9.6 / p293", "N/A: no atlas writer exists"},
      {22, "Annex D alpha", "D.2 chroma format determination", "D.2 / p1116",
       "N/A: informative annex; decoder/composition side"},
      {23, "Annex D alpha",
       "D.3 array initialization and background colour", "D.3 / p1117",
       "N/A: informative annex; no background colour is emitted"},
      {24, "Annex D alpha", "D.4 spatial mapping", "D.4 / p1118",
       "N/A: informative annex; no atlas segment placement is emitted"},
      {25, "Annex D alpha", "D.5 spatial mapping with alpha", "D.5 / p1119",
       "N/A: no alpha plane path in the encoder"},
      {26, "Annex D alpha",
       "D.5.1 frame resampling and D.5.2 monochrome resampling",
       "D.5.1-D.5.2 / p1120-1121",
       "N/A: sections are explicit placeholders outside the annex scope"},

      // Annex E timing/decoder model (7 absent)
      {27, "Annex E timing",
       "Timing + decoder-model flags are required to verify level conformance",
       "E.4.1/E.4.4 / p1126-1128",
       "check:E.4.4.LevelConformanceUnverifiable (warning; no BRT is emitted)"},
      {28, "Annex E timing",
       "seq_initial_display_delay_minus_1 / "
       "ops_initial_display_delay_minus_1 and InitialPresentationDelay",
       "E.5.7 / p1131",
       "timing: needs a BRT/CI OBU; no writer emits one"},
      {29, "Annex E timing",
       "TimeToDecode = lumaSamples / MaxDecodeRate (DecodeCount 2 reserve)",
       "E.5.6 / p1131",
       "check:A.4.MaxLevelRefFrames applies the DecodeCount reserve; "
       "full TimeToDecode needs timing"},
      {30, "Annex E timing",
       "PresentationInterval >= Max(pixels*outputFrames/MaxDisplayRate, "
       "MinFrameTime)",
       "E.7.6 / p1138",
       "check:A.4.MaxDisplayRate covers the rate half; MinFrameTime is a "
       "timing-model check"},
      {31, "Annex E timing",
       "Removal[i] + TimeToDecode[i] <= PresentationTime[i]",
       "E.7.7 / p1138", "timing: needs Removal[]/PresentationTime[]"},
      {32, "Annex E timing",
       "PresentationTime[j] non-decreasing after each RAP", "E.7.1 / p1136",
       "timing: needs PresentationTime[]; M2 sets immediate_output_frame = 1"},
      {33, "Annex E timing",
       "Decoder model must not raise DECODE_FRAME_BUF_UNAVAILABLE / "
       "DECODE_EXISTING_FRAME_BUF_EMPTY / DISPLAY_FRAME_LATE",
       "E.6/E.7.9 / p1134-1138",
       "timing: needs the frame-buffer model; A.4.MaxLevelRefFrames covers the "
       "DPB size half"},
  };
  return kCoverage;
}

}  // namespace av2enc::conformance

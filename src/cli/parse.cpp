// Command-line argument loop.  Moved verbatim from src/main.cpp: the flag
// order and the early-return order decide which error is reported first, so
// neither is reordered or rewritten here.
#include "cli/options.h"
#include "frame/metadata/cli/metadata_cli.h"

#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <ios>
#include <string>
#include <string_view>

bool parse_uint64(std::string_view text, std::uint64_t& value) {
  if (text.empty()) {
    return false;
  }
  const char* begin = text.data();
  const char* end = begin + text.size();
  std::uint64_t parsed = 0;
  const std::from_chars_result result = std::from_chars(begin, end, parsed);
  if (result.ec != std::errc{} || result.ptr != end) {
    return false;
  }
  value = parsed;
  return true;
}

bool parse_int(std::string_view text, int& value) {
  if (text.empty()) {
    return false;
  }
  const char* begin = text.data();
  const char* end = begin + text.size();
  int parsed = 0;
  const std::from_chars_result result = std::from_chars(begin, end, parsed);
  if (result.ec != std::errc{} || result.ptr != end) {
    return false;
  }
  value = parsed;
  return true;
}

bool parse_double(std::string_view text, double& value) {
  if (text.empty()) {
    return false;
  }
  std::string copy(text);
  char* end = nullptr;
  const double parsed = std::strtod(copy.c_str(), &end);
  if (end == nullptr || *end != '\0' || !std::isfinite(parsed)) {
    return false;
  }
  value = parsed;
  return true;
}

// --tiles accepts a total tile count or an explicit cols x rows grid and
// canonicalises it to "CxR". The tiled writer only implements the verified
// uniform layouts; anything else is rejected instead of being rounded to a
// different grid.
bool canonical_tile_layout(std::string_view text, std::string& canonical) {
  struct Alias {
    std::string_view name;
    std::string_view canonical;
  };
  static constexpr Alias kAliases[] = {
      {"1", "1x1"},   {"1x1", "1x1"}, {"2", "2x1"},   {"2x1", "2x1"},
      {"4", "2x2"},   {"2x2", "2x2"}, {"8", "4x2"},   {"4x2", "4x2"},
      {"16", "8x2"},  {"8x2", "8x2"}, {"32", "16x2"}, {"16x2", "16x2"},
  };
  for (const Alias& alias : kAliases) {
    if (text == alias.name) {
      canonical = alias.canonical;
      return true;
    }
  }
  return false;
}

ParseStatus parse_arguments(int argc, char** argv, Options& options, std::string& error) {
  for (int i = 1; i < argc; ++i) {
    const std::string_view argument = argv[i];

    if (argument == "-h" || argument == "--help") {
      options.help = true;
      return ParseStatus::kHelp;
    }

    if (argument.size() > 1 && argument.front() == '-') {
      options.specified.push_back(argument);
    }

    if (argument == "-o" || argument == "--output") {
      if (i + 1 >= argc) {
        error = "missing value after '" + std::string(argument) + "'";
        return ParseStatus::kError;
      }
      options.output = argv[++i];
      continue;
    }

    if (argument == "--frames") {
      if (i + 1 >= argc) {
        error = "missing value after '--frames'";
        return ParseStatus::kError;
      }
      std::uint64_t frames = 0;
      if (!parse_uint64(argv[++i], frames) || frames == 0) {
        error = "--frames must be a positive integer";
        return ParseStatus::kError;
      }
      options.max_frames = frames;
      continue;
    }

    if (argument == "--qp") {
      if (i + 1 >= argc) {
        error = "missing value after '--qp'";
        return ParseStatus::kError;
      }
      int qp = 0;
      if (!parse_int(argv[++i], qp) || qp < 0 || qp > 303) {
        error = "--qp must be an integer in [0, 303] (255 is MaxQ at 8-bit, "
                "303 at 10-bit)";
        return ParseStatus::kError;
      }
      options.qp = qp;
      continue;
    }

    if (argument == "--adaptive-cdf") {
      options.adaptive_cdf = true;
      continue;
    }

    if (argument == "--avg-cdf") {
      options.avg_cdf = true;
      continue;
    }

    if (argument == "--cdf-continuity") {
      options.cdf_continuity = true;
      continue;
    }

    if (argument == "--video-cdf-per-frame-qp") {
      options.video_cdf_per_frame_qp = true;
      continue;
    }

    if (argument == "--video-timing") {
      options.video_timing = true;
      continue;
    }

    if (argument == "--video-decoder-model") {
      // §5.4.13 nested block; also needs the §5.4.1 timing block.
      options.video_timing = true;
      options.video_decoder_model = true;
      if (i + 1 < argc && argv[i + 1][0] != '-') {
        const std::string_view value = argv[++i];
        const std::size_t first = value.find(':');
        const std::size_t second = first == std::string_view::npos
                                       ? std::string_view::npos
                                       : value.find(':', first + 1);
        std::uint64_t delay = 0;
        std::uint64_t edelay = 0;
        std::uint64_t low = 0;
        if (first == std::string_view::npos ||
            second == std::string_view::npos ||
            !parse_uint64(value.substr(0, first), delay) ||
            !parse_uint64(value.substr(first + 1, second - first - 1),
                          edelay) ||
            !parse_uint64(value.substr(second + 1), low) || low > 1 ||
            delay > 0xFFFFFFFFu || edelay > 0xFFFFFFFFu) {
          error =
              "--video-decoder-model expects decoder_delay:encoder_delay:"
              "low_delay_flag";
          return ParseStatus::kError;
        }
        options.decoder_buffer_delay = static_cast<std::uint32_t>(delay);
        options.encoder_buffer_delay = static_cast<std::uint32_t>(edelay);
        options.low_delay_mode = low != 0;
      }
      continue;
    }

    if (argument == "--video-brt") {
      options.video_brt = true;
      // Optional non-dependent br_time in rg(4) DecCT ticks, 0..511.
      if (i + 1 < argc && argv[i + 1][0] != '-') {
        std::uint64_t tick = 0;
        if (!parse_uint64(argv[++i], tick) || tick > 511) {
          error = "--video-brt expects a br_time in 0..511";
          return ParseStatus::kError;
        }
        options.brt_time = static_cast<std::uint32_t>(tick);
      }
      continue;
    }

    if (argument == "--video-mfh") {
      options.video_mfh = true;
      continue;
    }

    if (argument == "--video-tier") {
      if (i + 1 >= argc) {
        error = "missing value after '--video-tier'";
        return ParseStatus::kError;
      }
      const std::string_view value = argv[++i];
      if (value == "high") {
        options.video_tier_high = true;
      } else if (value == "main" || value == "0") {
        options.video_tier_high = false;
      } else {
        error = "--video-tier expects 'high' or 'main'";
        return ParseStatus::kError;
      }
      continue;
    }

    if (argument == "--video-multi-sequence") {
      options.video_multi_sequence = true;
      continue;
    }

    if (argument == "--video-lcr-full") {
      options.video_lcr_full = true;
      continue;
    }

    if (argument == "--still") {
      options.m1 = true;
      continue;
    }

    if (argument == "--still-gpu") {
      options.still_gpu = true;
      continue;
    }

    if (argument == "--tiles") {
      if (i + 1 >= argc) {
        error = "missing value after '--tiles'";
        return ParseStatus::kError;
      }
      std::string canonical;
      if (!canonical_tile_layout(argv[++i], canonical)) {
        error = "--tiles must be 2/4/8/16/32 or one of 1x1, 2x1, 2x2, 4x2, "
                "8x2, 16x2";
        return ParseStatus::kError;
      }
      options.tiles = canonical;
      continue;
    }

    if (argument == "--cpu-threads") {
      if (i + 1 >= argc) {
        error = "missing value after '--cpu-threads'";
        return ParseStatus::kError;
      }
      int value = 0;
      if (!parse_int(argv[++i], value) || value < 0 || value > 256) {
        error = "--cpu-threads must be an integer in [0, 256] "
                "(0 = auto, 1 = serial)";
        return ParseStatus::kError;
      }
      options.cpu_threads = static_cast<unsigned>(value);
      options.cpu_threads_given = true;
      continue;
    }

    if (argument == "--gpu-plan") {
      if (i + 1 >= argc) {
        error = "missing value after '--gpu-plan'";
        return ParseStatus::kError;
      }
      int gpus = 0;
      if (!parse_int(argv[++i], gpus) || gpus < 1 || gpus > 8) {
        error = "--gpu-plan must be an integer in [1, 8]";
        return ParseStatus::kError;
      }
      options.gpu_plan = gpus;
      continue;
    }

    if (argument == "--effort") {
      if (i + 1 >= argc) {
        error = "missing value after '--effort'";
        return ParseStatus::kError;
      }
      int value = 0;
      if (!parse_int(argv[++i], value) || value < 10 || value > 10000) {
        error = "--effort must be an integer in [10, 10000] (log scale)";
        return ParseStatus::kError;
      }
      options.effort_budget = value;
      options.effort_given = true;
      continue;
    }

    if (argument == "--cpu-used") {
      if (i + 1 >= argc) {
        error = "missing value after '--cpu-used'";
        return ParseStatus::kError;
      }
      int value = 0;
      if (!parse_int(argv[++i], value) || value < 0 || value > 8) {
        error = "--cpu-used must be an integer in [0, 8]";
        return ParseStatus::kError;
      }
      options.cpu_used = value;
      options.cpu_used_given = true;
      continue;
    }

    if (argument == "--effort-plan") {
      options.effort_plan = true;
      continue;
    }

    if (argument == "--bit-depth") {
      if (i + 1 >= argc) {
        error = "missing value after '--bit-depth'";
        return ParseStatus::kError;
      }
      int depth = 0;
      if (!parse_int(argv[++i], depth) || (depth != 8 && depth != 10)) {
        error = "--bit-depth must be 8 or 10 (12-bit is not signalable in AV2 "
                "v1.0.0)";
        return ParseStatus::kError;
      }
      options.bit_depth = depth;
      continue;
    }

    if (argument == "--chroma") {
      if (i + 1 >= argc) {
        error = "missing value after '--chroma'";
        return ParseStatus::kError;
      }
      const std::string value = argv[++i];
      if (value == "420" || value == "4:2:0") {
        options.chroma = "420";
      } else if (value == "422" || value == "4:2:2") {
        options.chroma = "422";
      } else if (value == "444" || value == "4:4:4") {
        options.chroma = "444";
      } else if (value == "400" || value == "4:0:0" || value == "mono") {
        options.chroma = "400";
      } else {
        error = "--chroma must be 420, 422, 444 or 400";
        return ParseStatus::kError;
      }
      continue;
    }

    if (argument == "--recon") {
      if (i + 1 >= argc) {
        error = "missing value after '--recon'";
        return ParseStatus::kError;
      }
      options.recon = argv[++i];
      continue;
    }

    if (argument == "--no-cfl") {
      options.still_cfl = false;
      continue;
    }

    if (argument == "--no-deblock") {
      options.still_deblock = false;
      continue;
    }
    if (argument == "--still-deblock") {
      options.still_deblock = true;
      continue;
    }

    if (argument == "--still-single") {
      options.still_single = true;
      continue;
    }

    if (argument == "--still-dc") {
      options.still_dc = true;
      continue;
    }

    if (argument == "--still-full-intra") {
      options.still_full_intra = true;
      continue;
    }

    if (argument == "--still-uv-full") {
      options.still_uv_full = true;
      continue;
    }

    if (argument == "--still-rdo-intra") {
      options.still_rdo_intra = true;
      continue;
    }

    if (argument == "--still-rdo-scale") {
      if (i + 1 >= argc ||
          !parse_double(argv[++i], options.still_rdo_scale) ||
          options.still_rdo_scale <= 0.0) {
        error = "--still-rdo-scale requires a positive number";
        return ParseStatus::kError;
      }
      continue;
    }

    if (argument == "--still-rdoq") {
      options.still_rdoq = true;
      continue;
    }

    if (argument == "--still-rdoq-scale") {
      if (i + 1 >= argc ||
          !parse_double(argv[++i], options.still_rdoq_scale) ||
          options.still_rdoq_scale <= 0.0) {
        error = "--still-rdoq-scale requires a positive number";
        return ParseStatus::kError;
      }
      continue;
    }

    if (argument == "--still-tx-type") {
      options.still_tx_type = true;
      continue;
    }

    if (argument == "--still-tx-type-scale") {
      if (i + 1 >= argc ||
          !parse_double(argv[++i], options.still_tx_type_scale) ||
          options.still_tx_type_scale <= 0.0) {
        error = "--still-tx-type-scale requires a positive number";
        return ParseStatus::kError;
      }
      continue;
    }

    if (argument == "--still-cdef") {
      options.still_cdef = true;
      continue;
    }

    if (argument == "--still-cdef-on-skip-txfm") {
      options.still_cdef_on_skip_txfm = true;
      continue;
    }

    if (argument == "--still-ccso") {
      options.still_ccso = true;
      continue;
    }

    if (argument == "--still-sb128") {
      options.still_sb128 = true;
      continue;
    }

    if (argument == "--still-sb256") {
      options.still_sb256 = true;
      continue;
    }

    if (argument == "--still-ext-partitions") {
      options.still_ext_partitions = true;
      continue;
    }

    if (argument == "--still-sub8") {
      options.still_sub8 = true;
      continue;
    }

    if (argument == "--still-sdp") {
      options.still_sdp = true;
      continue;
    }

    if (argument == "--still-segmentation") {
      options.still_segmentation = true;
      continue;
    }

    if (argument == "--still-delta-q") {
      options.still_delta_q = true;
      continue;
    }

    if (argument == "--still-per-segment-q") {
      options.still_per_segment_q = true;
      continue;
    }

    if (argument == "--still-seg-delta") {
      if (i + 1 >= argc) {
        error = "missing value after '--still-seg-delta'";
        return ParseStatus::kError;
      }
      int delta = 0;
      if (!parse_int(argv[++i], delta) || delta < 0 || delta > 64) {
        error = "--still-seg-delta must be an integer in [0, 64]";
        return ParseStatus::kError;
      }
      options.still_seg_delta = delta;
      continue;
    }

    if (argument == "--still-qm") {
      options.still_qm = true;
      continue;
    }

    if (argument == "--still-user-qm") {
      options.still_qm = true;
      options.still_user_qm = true;
      continue;
    }

    if (argument == "--still-dc-ac-delta-q") {
      options.still_dc_ac_delta_q = true;
      continue;
    }

    if (argument == "--still-content-interpretation") {
      options.still_content_interpretation = true;
      continue;
    }

    if (argument == "--metadata-timecode") {
      options.metadata_timecode = true;
      continue;
    }

    if (argument == "--metadata-scan-type") {
      options.metadata_scan_type = true;
      continue;
    }

    if (argument == "--metadata-cll") {
      options.metadata_cll = true;
      // Optional value max_cll:max_fall; the no-argument form is the neutral
      // 0:0 ("no upper bound signalled") unit.
      if (i + 1 < argc && argv[i + 1][0] != '-') {
        const std::string_view value = argv[++i];
        const std::size_t colon = value.find(':');
        std::uint64_t max_cll = 0;
        std::uint64_t max_fall = 0;
        if (colon == std::string_view::npos ||
            !parse_uint64(value.substr(0, colon), max_cll) ||
            !parse_uint64(value.substr(colon + 1), max_fall) ||
            max_cll > 65535 || max_fall > 65535) {
          error =
              "--metadata-cll expects max_cll:max_fall with each value in "
              "0..65535";
          return ParseStatus::kError;
        }
        options.metadata_max_cll = static_cast<std::uint16_t>(max_cll);
        options.metadata_max_fall = static_cast<std::uint16_t>(max_fall);
      }
      continue;
    }

    if (argument == "--metadata-mdcv") {
      options.metadata_mdcv = true;
      // No value = all-zero "unknown/unspecified" (every §6.16.6 range is
      // defined to mean that).  `bt2020` is the ST 2086 BT.2020/D65 preset;
      // otherwise ten comma-separated integers
      //   gx,gy,bx,by,rx,ry,wx,wy,luminance_max,luminance_min
      if (i + 1 < argc && argv[i + 1][0] != '-') {
        const std::string_view value = argv[++i];
        if (value == "bt2020") {
          auto& m = options.metadata_mdcv_value;
          m.primary_chromaticity_x = {8500, 6550, 35400};
          m.primary_chromaticity_y = {39850, 2300, 14600};
          m.white_point_chromaticity_x = 15635;
          m.white_point_chromaticity_y = 16450;
          m.luminance_max = 10000000;
          m.luminance_min = 50;
        } else {
          std::uint64_t fields[10] = {};
          std::size_t count = 0;
          std::size_t start = 0;
          while (count < 10) {
            const std::size_t comma = value.find(',', start);
            const std::string_view item =
                value.substr(start, comma == std::string_view::npos
                                        ? std::string_view::npos
                                        : comma - start);
            if (!parse_uint64(item, fields[count])) {
              error =
                  "--metadata-mdcv expects a preset name or ten comma-separated "
                  "integers (gx,gy,bx,by,rx,ry,wx,wy,max,min)";
              return ParseStatus::kError;
            }
            ++count;
            if (comma == std::string_view::npos) break;
            start = comma + 1;
          }
          if (count != 10) {
            error =
                "--metadata-mdcv expects a preset name or ten comma-separated "
                "integers (gx,gy,bx,by,rx,ry,wx,wy,max,min)";
            return ParseStatus::kError;
          }
          for (int k = 0; k < 8; ++k) {
            if (fields[k] > 65535) {
              error = "--metadata-mdcv chromaticities must fit 16 bits";
              return ParseStatus::kError;
            }
          }
          if (fields[8] > 0xFFFFFFFFu || fields[9] > 0xFFFFFFFFu) {
            error = "--metadata-mdcv luminances must fit 32 bits";
            return ParseStatus::kError;
          }
          auto& m = options.metadata_mdcv_value;
          m.primary_chromaticity_x = {static_cast<std::uint16_t>(fields[0]),
                                      static_cast<std::uint16_t>(fields[2]),
                                      static_cast<std::uint16_t>(fields[4])};
          m.primary_chromaticity_y = {static_cast<std::uint16_t>(fields[1]),
                                      static_cast<std::uint16_t>(fields[3]),
                                      static_cast<std::uint16_t>(fields[5])};
          m.white_point_chromaticity_x = static_cast<std::uint16_t>(fields[6]);
          m.white_point_chromaticity_y = static_cast<std::uint16_t>(fields[7]);
          m.luminance_max = static_cast<std::uint32_t>(fields[8]);
          m.luminance_min = static_cast<std::uint32_t>(fields[9]);
        }
      }
      continue;
    }

    if (argument == "--metadata-icc") {
      options.metadata_icc = true;
      options.metadata_icc_data = av2enc::metadata::cli::default_icc_profile();
      // Optional path to a real profile.  A profile header is 128 bytes and
      // avmdec rejects shorter payloads, so that is the minimum accepted here.
      if (i + 1 < argc && argv[i + 1][0] != '-') {
        const std::string path = argv[++i];
        std::ifstream input(path, std::ios::binary);
        if (!input) {
          error = "--metadata-icc cannot open '" + path + "'";
          return ParseStatus::kError;
        }
        input.seekg(0, std::ios::end);
        const std::streamoff size = input.tellg();
        input.seekg(0, std::ios::beg);
        if (size < 128) {
          error = "--metadata-icc profile must be at least 128 bytes";
          return ParseStatus::kError;
        }
        options.metadata_icc_data.assign(static_cast<std::size_t>(size), 0);
        if (!input.read(reinterpret_cast<char*>(options.metadata_icc_data.data()),
                        size)) {
          error = "--metadata-icc cannot read '" + path + "'";
          return ParseStatus::kError;
        }
      }
      continue;
    }

    if (argument == "--metadata-temporal-point") {
      options.metadata_temporal_point = true;
      // Optional uleb128 value 0..2^32-1; no argument emits the neutral 0.
      if (i + 1 < argc && argv[i + 1][0] != '-') {
        std::uint64_t ticks = 0;
        if (!parse_uint64(argv[++i], ticks) || ticks > 0xFFFFFFFFu) {
          error = "--metadata-temporal-point expects a value in 0..2^32-1";
          return ParseStatus::kError;
        }
        options.metadata_frame_presentation_time =
            static_cast<std::uint32_t>(ticks);
      }
      continue;
    }

    if (argument == "--metadata-frame-hash") {
      options.metadata_frame_hash = true;
      // Optional "per-plane" (three plane digests); the default is one MD5
      // over Y+U+V in plane order.
      if (i + 1 < argc && argv[i + 1][0] != '-') {
        const std::string_view value = argv[++i];
        if (value == "per-plane") {
          options.metadata_frame_hash_per_plane = true;
        } else if (value == "frame") {
          options.metadata_frame_hash_per_plane = false;
        } else {
          error = "--metadata-frame-hash expects 'per-plane' or 'frame'";
          return ParseStatus::kError;
        }
      }
      continue;
    }

    if (argument == "--metadata-banding") {
      options.metadata_banding = true;
      // Neutral by default (1-byte unit, both presence flags 0).  `source`
      // sets source_banding_present_flag and still codes no extra syntax.
      if (i + 1 < argc && argv[i + 1][0] != '-') {
        const std::string_view value = argv[++i];
        if (value == "neutral") {
          options.metadata_banding_value = {};
        } else if (value == "source") {
          options.metadata_banding_value = {};
          options.metadata_banding_value.source_banding_present_flag = true;
        } else {
          error = "--metadata-banding expects 'neutral' or 'source'";
          return ParseStatus::kError;
        }
      }
      continue;
    }

    if (argument == "--metadata-user-data") {
      options.metadata_user_data = true;
      options.metadata_user_data_value = {};
      // Optional "uuid[:hex-payload]".  The UUID is 32 hex digits (dashes are
      // ignored); no argument emits the zero UUID with an empty payload.
      if (i + 1 < argc && argv[i + 1][0] != '-') {
        const std::string_view value = argv[++i];
        const auto hex_value = [](char c) -> int {
          if (c >= '0' && c <= '9') return c - '0';
          if (c >= 'a' && c <= 'f') return c - 'a' + 10;
          if (c >= 'A' && c <= 'F') return c - 'A' + 10;
          return -1;
        };
        const std::size_t colon = value.find(':');
        const std::string_view uuid_text = value.substr(0, colon);
        std::string uuid_hex;
        for (const char c : uuid_text) {
          if (c != '-') uuid_hex.push_back(c);
        }
        if (uuid_hex.size() != 32) {
          error = "--metadata-user-data UUID must be 32 hex digits";
          return ParseStatus::kError;
        }
        for (std::size_t k = 0; k < 16; ++k) {
          const int hi = hex_value(uuid_hex[k * 2]);
          const int lo = hex_value(uuid_hex[k * 2 + 1]);
          if (hi < 0 || lo < 0) {
            error = "--metadata-user-data UUID must be hexadecimal";
            return ParseStatus::kError;
          }
          options.metadata_user_data_value.uuid[k] =
              static_cast<std::uint8_t>((hi << 4) | lo);
        }
        if (colon != std::string_view::npos) {
          const std::string_view hex = value.substr(colon + 1);
          if (hex.size() % 2 != 0) {
            error = "--metadata-user-data payload must be byte-aligned hex";
            return ParseStatus::kError;
          }
          for (std::size_t k = 0; k < hex.size(); k += 2) {
            const int hi = hex_value(hex[k]);
            const int lo = hex_value(hex[k + 1]);
            if (hi < 0 || lo < 0) {
              error = "--metadata-user-data payload must be hexadecimal";
              return ParseStatus::kError;
            }
            options.metadata_user_data_value.payload.push_back(
                static_cast<std::uint8_t>((hi << 4) | lo));
          }
        }
      }
      continue;
    }

    if (argument == "--lossless") {
      options.lossless = true;
      continue;
    }

    if (argument == "--lossless-fsc") {
      options.lossless_force_fsc = true;
      continue;
    }

    if (argument == "--still-delta-q-res") {
      if (i + 1 >= argc) {
        error = "missing value after '--still-delta-q-res'";
        return ParseStatus::kError;
      }
      int res = 0;
      if (!parse_int(argv[++i], res) || res < 0 || res > 3) {
        error = "--still-delta-q-res must be an integer in [0, 3]";
        return ParseStatus::kError;
      }
      options.still_delta_q_res = res;
      continue;
    }

    if (argument == "--still-ist") {
      options.still_ist = true;
      continue;
    }

    if (argument == "--still-ist-scale") {
      if (i + 1 >= argc ||
          !parse_double(argv[++i], options.still_ist_scale) ||
          options.still_ist_scale <= 0.0) {
        error = "--still-ist-scale requires a positive number";
        return ParseStatus::kError;
      }
      continue;
    }

    if (argument == "--still-tcq") {
      options.still_tcq = true;
      continue;
    }

    if (argument == "--still-cctx") {
      options.still_cctx = true;
      continue;
    }

    if (argument == "--still-tx-partition") {
      options.still_tx_partition = true;
      continue;
    }

    if (argument == "--still-idtx") {
      options.still_idtx = true;
      continue;
    }

    if (argument == "--still-ddt") {
      options.still_ddt = true;
      continue;
    }

    if (argument == "--still-tcq-scale") {
      if (i + 1 >= argc ||
          !parse_double(argv[++i], options.still_tcq_scale) ||
          options.still_tcq_scale <= 0.0) {
        error = "--still-tcq-scale requires a positive number";
        return ParseStatus::kError;
      }
      continue;
    }

    if (argument == "--still-qm-level") {
      if (i + 1 >= argc) {
        error = "missing value after '--still-qm-level'";
        return ParseStatus::kError;
      }
      int level = 0;
      if (!parse_int(argv[++i], level) || level < 0 || level > 14) {
        error = "--still-qm-level must be an integer in [0, 14]";
        return ParseStatus::kError;
      }
      options.still_qm_level = level;
      continue;
    }

    if (argument == "--still-ibp") {
      options.still_ibp = true;
      continue;
    }

    if (argument == "--still-mrl") {
      options.still_mrl = true;
      continue;
    }

    if (argument == "--still-cfl-ds") {
      if (i + 1 >= argc) {
        error = "missing value after '--still-cfl-ds'";
        return ParseStatus::kError;
      }
      int ds = 0;
      if (!parse_int(argv[++i], ds) || ds < 0 || ds > 3) {
        error = "--still-cfl-ds must be an integer in [0, 3]";
        return ParseStatus::kError;
      }
      options.still_cfl_ds = ds;
      continue;
    }

    if (argument == "--still-cfl-explicit") {
      options.still_cfl_explicit = true;
      continue;
    }

    if (argument == "--still-dip") {
      options.still_dip = true;
      continue;
    }

    if (argument == "--still-palette") {
      options.still_palette = true;
      continue;
    }

    if (argument == "--still-ibc") {
      options.still_ibc = true;
      continue;
    }

    if (argument == "--still-ibc-search") {
      if (i + 1 >= argc) {
        error = "missing value after '--still-ibc-search'";
        return ParseStatus::kError;
      }
      int step = 0;
      if (!parse_int(argv[++i], step) || step < 1 || step > 64) {
        error = "--still-ibc-search must be an integer in [1, 64]";
        return ParseStatus::kError;
      }
      options.still_ibc_search_step = step;
      continue;
    }

    if (argument == "--still-mhccp") {
      options.still_mhccp = true;
      continue;
    }

    if (argument == "--still-ccso-edge") {
      options.still_ccso = true;
      options.still_ccso_edge = true;
      continue;
    }

    if (argument == "--still-wiener") {
      options.still_wiener = true;
      continue;
    }

    if (argument == "--still-chroma-lr") {
      options.still_chroma_lr = true;
      continue;
    }

    if (argument == "--still-pc-wiener") {
      options.still_pc_wiener = true;
      continue;
    }

    if (argument == "--still-gdf") {
      options.still_gdf = true;
      continue;
    }

    if (argument == "--still-grain") {
      options.still_grain = true;
      continue;
    }

    if (argument == "--still-grain-adaptive") {
      options.still_grain_adaptive = true;
      continue;
    }

    if (argument == "--video") {
      options.video = true;
      continue;
    }

    if (argument == "--video-gpu") {
      // Implies --video: the GPU path is the same multi-frame sequence with a
      // device backend.  It is rejected by name in a build without the target
      // (see AV2ENC_VIDEO_GPU_CALLABLE below).
      options.video = true;
      options.video_gpu = true;
      continue;
    }

    if (argument == "--p-frames") {
      if (i + 1 >= argc) {
        error = "missing value after '--p-frames'";
        return ParseStatus::kError;
      }
      if (!parse_uint64(argv[++i], options.p_frames)) {
        error = "--p-frames must be a non-negative integer";
        return ParseStatus::kError;
      }
      continue;
    }

    if (argument == "--b-frames") {
      if (i + 1 >= argc) {
        error = "missing value after '--b-frames'";
        return ParseStatus::kError;
      }
      if (!parse_uint64(argv[++i], options.b_frames)) {
        error = "--b-frames must be a non-negative integer";
        return ParseStatus::kError;
      }
      continue;
    }

    if (argument == "--video-gops") {
      if (i + 1 >= argc) {
        error = "missing value after '--video-gops'";
        return ParseStatus::kError;
      }
      int value = 0;
      if (!parse_int(argv[++i], value) || value <= 0) {
        error = "--video-gops must be a positive integer";
        return ParseStatus::kError;
      }
      options.video_gops = value;
      continue;
    }

    if (argument == "--video-refs") {
      if (i + 1 >= argc) {
        error = "missing value after '--video-refs'";
        return ParseStatus::kError;
      }
      int refs = 0;
      if (!parse_int(argv[++i], refs) || refs < 1 ||
          refs > av2enc::video::kRefsPerFrame) {
        error = "--video-refs must be an integer in [1, " +
                std::to_string(av2enc::video::kRefsPerFrame) + "]";
        return ParseStatus::kError;
      }
      options.video_refs = refs;
      continue;
    }

    if (argument == "--video-num-ref-frames") {
      if (i + 1 >= argc) {
        error = "missing value after '--video-num-ref-frames'";
        return ParseStatus::kError;
      }
      int frames = 0;
      if (!parse_int(argv[++i], frames) || frames < 4 || frames > 8) {
        error = "--video-num-ref-frames must be an integer in [4, 8]";
        return ParseStatus::kError;
      }
      options.video_num_ref_frames = frames;
      continue;
    }

    if (argument == "--video-implicit-ref-map") {
      options.video_implicit_ref_map = true;
      continue;
    }

    if (argument == "--video-subpel-eighth") {
      options.video_subpel_eighth = true;
      continue;
    }

    if (argument == "--video-mv-precision") {
      if (i + 1 >= argc) {
        error = "missing value after '--video-mv-precision'";
        return ParseStatus::kError;
      }
      const std::string_view value = argv[++i];
      if (value == "eighth") {
        options.video_mv_precision = 0;
      } else if (value == "quarter") {
        options.video_mv_precision = 1;
      } else if (value == "half") {
        options.video_mv_precision = 2;
      } else {
        error = "--video-mv-precision must be eighth, quarter or half";
        return ParseStatus::kError;
      }
      continue;
    }

    if (argument == "--video-adaptive-mvd") {
      options.video_adaptive_mvd = true;
      continue;
    }

    if (argument == "--video-mvd-sign-derive") {
      options.video_mvd_sign_derive = true;
      continue;
    }

    if (argument == "--video-flex-mvres") {
      options.video_flex_mvres = true;
      continue;
    }

    if (argument == "--video-compound") {
      options.video_compound = true;
      continue;
    }

    if (argument == "--video-compound-wedge") {
      options.video_compound_wedge = true;
      continue;
    }

    if (argument == "--video-compound-diffwtd") {
      options.video_compound_diffwtd = true;
      continue;
    }

    if (argument == "--video-cwp-weighted") {
      options.video_cwp_weighted = true;
      continue;
    }

    if (argument == "--video-imp-mask") {
      options.video_imp_mask = true;
      continue;
    }

    if (argument == "--video-df-sub-pu") {
      options.video_df_sub_pu = true;
      continue;
    }

    if (argument == "--video-compound-interintra") {
      options.video_compound_interintra = true;
      continue;
    }

    if (argument == "--video-inter-residual") {
      options.video_inter_residual = true;
      continue;
    }

    if (argument == "--video-edge-inter") {
      options.video_edge_inter = true;
      continue;
    }

    if (argument == "--video-edge-mask") {
      options.video_edge_mask = true;
      continue;
    }

    if (argument == "--video-leaf-rd") {
      options.video_leaf_rd = true;
      continue;
    }

    if (argument == "--fast-decode" || argument == "--nightly") {
      // --nightly is the pre-2026-11-20 name for this aggregate; it stays an
      // exact alias so every existing gate and recipe keeps its stream.
      options.fast_decode = true;
      continue;
    }

    if (argument == "--nightly-b") {
      options.nightly_b = true;
      continue;
    }

    if (argument == "--video-rd-driver") {
      options.video_rd_driver = true;
      continue;
    }

    if (argument == "--video-eff-perc") {
      options.video_eff_perc = true;
      continue;
    }

    if (argument == "--video-tail-tx") {
      options.video_tail_tx = true;
      continue;
    }

    if (argument == "--video-head-tx") {
      options.video_head_tx = true;
      continue;
    }

    if (argument == "--video-rd-preset") {
      if (i + 1 >= argc) {
        error = "missing value after '--video-rd-preset'";
        return ParseStatus::kError;
      }
      options.video_rd_preset = argv[++i];
      continue;
    }

    if (argument == "--video-tx-search") {
      options.video_tx_search = true;
      continue;
    }

    if (argument == "--video-key-intra-txpart") {
      options.video_key_intra_txpart = true;
      continue;
    }

    if (argument == "--video-tx-context-rate") {
      options.video_tx_context_rate = true;
      continue;
    }

    if (argument == "--video-partition-rd") {
      options.video_partition_rd = true;
      continue;
    }

    if (argument == "--video-partition-rd-depth") {
      if (i + 1 >= argc) {
        error = "missing value after '--video-partition-rd-depth'";
        return ParseStatus::kError;
      }
      std::uint64_t depth = 0;
      if (!parse_uint64(argv[++i], depth) || depth < 1 || depth > 3) {
        error = "--video-partition-rd-depth must be 1, 2 or 3";
        return ParseStatus::kError;
      }
      options.video_partition_depth = static_cast<int>(depth);
      continue;
    }

    if (argument == "--video-partition-rd-lambda") {
      if (i + 1 >= argc) {
        error = "missing value after '--video-partition-rd-lambda'";
        return ParseStatus::kError;
      }
      char* end = nullptr;
      const double value = std::strtod(argv[++i], &end);
      if (end == argv[i] || *end != '\0' || !(value > 0.0)) {
        error = "--video-partition-rd-lambda needs a positive number";
        return ParseStatus::kError;
      }
      options.video_partition_lambda = value;
      continue;
    }

    if (argument == "--video-sdp") {
      options.video_sdp = true;
      continue;
    }

    if (argument == "--video-ddt") {
      options.video_ddt = true;
      continue;
    }

    if (argument == "--video-tile-flush-off") {
      options.video_tile_flush_off = true;
      continue;
    }

    if (argument == "--video-ref-signal-off") {
      options.video_ref_signal_off = true;
      continue;
    }

    if (argument == "--video-edge-inter-off") {
      options.video_edge_inter_off = true;
      continue;
    }

    if (argument == "--video-edge-mask-off") {
      options.video_edge_mask_off = true;
      continue;
    }

    if (argument == "--video-bawp-off") {
      options.video_bawp_off = true;
      continue;
    }

    if (argument == "--video-compound-off") {
      options.video_compound_off = true;
      continue;
    }

    if (argument == "--video-drift-refresh-off") {
      options.video_drift_refresh_off = true;
      continue;
    }

    if (argument == "--video-cdf-chain-off") {
      options.video_cdf_chain_off = true;
      continue;
    }

    if (argument == "--video-key-qp-offset-off") {
      options.video_key_qp_offset_off = true;
      continue;
    }

    if (argument == "--video-deblock-off") {
      options.video_deblock_off = true;
      continue;
    }

    if (argument == "--video-loop-filters-off") {
      options.video_loop_filters_off = true;
      continue;
    }

    if (argument == "--video-tx-search-off") {
      options.video_tx_search_off = true;
      continue;
    }

    if (argument == "--video-rdoq-off") {
      options.video_rdoq_off = true;
      continue;
    }

    if (argument == "--video-partition-rd-off") {
      options.video_partition_rd_off = true;
      continue;
    }

    if (argument == "--video-tile-group-off") {
      options.video_tile_group_off = true;
      continue;
    }

    if (argument == "--video-ccso-header-off") {
      options.video_ccso_header_off = true;
      continue;
    }

    if (argument == "--video-key-tx-lambda") {
      if (i + 1 >= argc) {
        error = "missing value after '--video-key-tx-lambda'";
        return ParseStatus::kError;
      }
      char* end = nullptr;
      const double value = std::strtod(argv[++i], &end);
      if (end == argv[i] || *end != '\0' || !(value > 0.0)) {
        error = "--video-key-tx-lambda needs a positive number";
        return ParseStatus::kError;
      }
      options.video_key_tx_lambda = value;
      continue;
    }

    if (argument == "--video-leaf-rd-lambda") {
      if (i + 1 >= argc) {
        error = "missing value after '--video-leaf-rd-lambda'";
        return ParseStatus::kError;
      }
      char* end = nullptr;
      const double value = std::strtod(argv[++i], &end);
      if (end == argv[i] || *end != '\0' || !(value > 0.0)) {
        error = "--video-leaf-rd-lambda needs a positive number";
        return ParseStatus::kError;
      }
      options.video_leaf_rd_lambda = value;
      continue;
    }

    if (argument == "--video-qp-lambda") {
      options.video_qp_lambda = true;
      continue;
    }

    if (argument == "--video-tx-lambda") {
      if (i + 1 >= argc) {
        error = "missing value after '--video-tx-lambda'";
        return ParseStatus::kError;
      }
      char* end = nullptr;
      const double value = std::strtod(argv[++i], &end);
      if (end == argv[i] || *end != '\0' || !(value > 0.0)) {
        error = "--video-tx-lambda needs a positive number";
        return ParseStatus::kError;
      }
      options.video_tx_lambda = value;
      continue;
    }

    if (argument == "--video-deblock") {
      options.video_deblock = true;
      continue;
    }

    if (argument == "--video-segmentation") {
      options.video_segmentation = true;
      continue;
    }

    if (argument == "--video-delta-q") {
      options.video_delta_q = true;
      continue;
    }

    if (argument == "--video-delta-q-res") {
      if (i + 1 >= argc ||
          !parse_int(argv[++i], options.video_delta_q_res) ||
          options.video_delta_q_res < 0 || options.video_delta_q_res > 3) {
        error = "--video-delta-q-res must be an integer in [0, 3]";
        return ParseStatus::kError;
      }
      continue;
    }

    if (argument == "--video-aq") {
      options.video_aq = true;
      continue;
    }

    if (argument == "--video-aq-strength") {
      if (i + 1 >= argc || !parse_double(argv[++i], options.video_aq_strength) ||
          !(options.video_aq_strength >= 0.0) ||
          options.video_aq_strength > 16.0) {
        error = "--video-aq-strength must be a number in [0, 16]";
        return ParseStatus::kError;
      }
      continue;
    }

    if (argument == "--video-aq-bias") {
      options.video_aq_bias_given = true;
      if (i + 1 >= argc || !parse_int(argv[++i], options.video_aq_bias) ||
          options.video_aq_bias < -8 || options.video_aq_bias > 8) {
        error = "--video-aq-bias must be an integer in [-8, 8]";
        return ParseStatus::kError;
      }
      continue;
    }

    if (argument == "--video-aq-clamp") {
      if (i + 1 >= argc || !parse_int(argv[++i], options.video_aq_clamp) ||
          options.video_aq_clamp < 1 || options.video_aq_clamp > 8) {
        error = "--video-aq-clamp must be an integer in [1, 8]";
        return ParseStatus::kError;
      }
      continue;
    }

    if (argument == "--video-per-segment-q") {
      options.video_per_segment_q = true;
      continue;
    }

    if (argument == "--video-seg-delta") {
      if (i + 1 >= argc || !parse_int(argv[++i], options.video_seg_delta) ||
          options.video_seg_delta < -351 || options.video_seg_delta > 351) {
        error = "--video-seg-delta must be an integer in [-351, 351]";
        return ParseStatus::kError;
      }
      continue;
    }

    if (argument == "--video-uv-delta-q") {
      if (i + 1 >= argc ||
          !parse_int(argv[++i], options.video_uv_delta_q_value) ||
          options.video_uv_delta_q_value < -63 ||
          options.video_uv_delta_q_value > 63) {
        error = "--video-uv-delta-q must be an integer in [-63, 63]";
        return ParseStatus::kError;
      }
      options.video_uv_delta_q = true;
      continue;
    }

    if (argument == "--video-seg-skip") {
      options.video_seg_skip = true;
      options.video_segmentation = true;
      continue;
    }

    if (argument == "--video-seg-globalmv") {
      options.video_seg_globalmv = true;
      options.video_segmentation = true;
      continue;
    }

    if (argument == "--video-seg-temporal") {
      options.video_seg_temporal = true;
      options.video_segmentation = true;
      continue;
    }

    if (argument == "--video-segments") {
      if (i + 1 >= argc || !parse_int(argv[++i], options.video_segments) ||
          options.video_segments < 1 || options.video_segments > 16) {
        error = "--video-segments must be an integer in [1, 16]";
        return ParseStatus::kError;
      }
      if (options.video_segments > 1) options.video_segmentation = true;
      continue;
    }

    if (argument == "--video-gme") {
      options.video_gme = true;
      continue;
    }

    if (argument == "--video-warp") {
      options.video_warp = true;
      options.video_gme = true;
      continue;
    }

    if (argument == "--video-bawp") {
      options.video_bawp = true;
      continue;
    }

    if (argument == "--video-bawp-carrier") {
      options.video_bawp = true;
      options.video_bawp_carrier = true;
      continue;
    }

    if (argument == "--video-ref-mvs") {
      options.video_ref_mvs = true;
      continue;
    }

    if (argument == "--video-ref-mvs-stress") {
      options.video_ref_mvs = true;
      options.video_ref_mvs_stress = true;
      continue;
    }

    if (argument == "--video-short-refresh") {
      options.video_short_refresh = true;
      continue;
    }

    if (argument == "--video-ref-signal-compact") {
      options.video_ref_signal_compact = true;
      continue;
    }

    if (argument == "--video-refmvbank") {
      options.video_refmvbank = true;
      continue;
    }

    if (argument == "--video-mv-traj") {
      options.video_mv_traj = true;
      continue;
    }

    if (argument == "--video-same-ref-compound") {
      options.video_same_ref_compound = true;
      continue;
    }

    if (argument == "--video-bru") {
      options.video_bru = true;
      continue;
    }

    if (argument == "--video-morph") {
      options.video_morph = true;
      continue;
    }

    if (argument == "--video-palette") {
      options.video_palette = true;
      continue;
    }

    if (argument == "--video-ibc") {
      options.video_ibc = true;
      continue;
    }

    if (argument == "--video-scc-force") {
      options.video_scc_force = true;
      continue;
    }

    if (argument == "--video-force-integer-mv") {
      options.video_force_integer_mv = true;
      continue;
    }

    if (argument == "--video-reduce-pb-aspect") {
      options.video_reduce_pb_aspect = true;
      options.video_pb_aspect_log2_minus_1 = 1;  // 4:1
      continue;
    }

    if (argument == "--video-pb-aspect-2") {
      options.video_reduce_pb_aspect = true;
      options.video_pb_aspect_log2_minus_1 = 0;  // 2:1
      continue;
    }

    if (argument == "--video-seq-seg-info") {
      options.video_seq_seg_info = true;
      options.video_seq_seg_frozen = false;
      continue;
    }

    if (argument == "--video-seq-seg-info-frozen") {
      options.video_seq_seg_info = true;
      options.video_seq_seg_frozen = true;
      continue;
    }

    if (argument == "--video-tile-group-compact") {
      options.video_tile_group_compact = true;
      continue;
    }

    if (argument == "--video-opfl") {
      options.video_opfl = true;
      continue;
    }

    if (argument == "--video-refine-mv") {
      options.video_refine_mv = true;
      continue;
    }

    if (argument == "--video-compound-near-near") {
      options.video_compound_near_near = true;
      continue;
    }

    if (argument == "--video-tip") {
      options.video_tip = true;
      continue;
    }

    if (argument == "--video-tip-no-hole-fill") {
      options.video_tip_no_hole_fill = true;
      continue;
    }

    if (argument == "--video-tip-block-ref") {
      options.video_tip_block_ref = true;
      continue;
    }

    if (argument == "--video-cdef") {
      options.video_cdef = true;
      continue;
    }

    if (argument == "--video-cdef-rd") {
      options.video_cdef_rd = true;
      options.video_cdef = true;
      continue;
    }

    if (argument == "--video-cdef-rd-lambda") {
      if (i + 1 >= argc) {
        error = "missing value after '--video-cdef-rd-lambda'";
        return ParseStatus::kError;
      }
      double lambda = 0.0;
      if (!parse_double(argv[++i], lambda) || lambda < 0.0) {
        error = "--video-cdef-rd-lambda must be a non-negative number";
        return ParseStatus::kError;
      }
      options.video_cdef_rd_lambda = lambda;
      options.video_cdef_rd = true;
      options.video_cdef = true;
      continue;
    }

    if (argument == "--video-cdef-strength") {
      if (i + 1 >= argc) {
        error = "missing value after '--video-cdef-strength'";
        return ParseStatus::kError;
      }
      int strength = 0;
      if (!parse_int(argv[++i], strength) || strength < 1 || strength > 8) {
        error = "--video-cdef-strength must be an integer in [1, 8]";
        return ParseStatus::kError;
      }
      options.video_cdef_strength = strength;
      options.video_cdef = true;
      continue;
    }

    if (argument == "--video-cdef-damping") {
      if (i + 1 >= argc) {
        error = "missing value after '--video-cdef-damping'";
        return ParseStatus::kError;
      }
      int damping = 0;
      if (!parse_int(argv[++i], damping) || damping < 3 || damping > 6) {
        error = "--video-cdef-damping must be an integer in [3, 6]";
        return ParseStatus::kError;
      }
      options.video_cdef_damping = damping;
      continue;
    }

    if (argument == "--video-cdef-search") {
      if (i + 1 >= argc) {
        error = "missing value after '--video-cdef-search'";
        return ParseStatus::kError;
      }
      const std::string_view mode = argv[++i];
      if (mode == "full") {
        options.video_cdef_search = 0;
      } else if (mode == "ladder") {
        options.video_cdef_search = 1;
      } else if (mode == "step") {
        options.video_cdef_search = 2;
      } else if (mode == "fast") {
        options.video_cdef_search = 3;
      } else {
        error = "--video-cdef-search must be full, ladder, step or fast";
        return ParseStatus::kError;
      }
      options.video_cdef = true;
      continue;
    }

    if (argument == "--video-ccso") {
      options.video_ccso = true;
      continue;
    }

    if (argument == "--video-wiener") {
      options.video_wiener = true;
      continue;
    }

    if (argument == "--video-chroma-lr") {
      options.video_chroma_lr = true;
      continue;
    }

    if (argument == "--video-gdf") {
      options.video_gdf = true;
      continue;
    }

    if (argument == "--video-ccso-bands") {
      if (i + 1 >= argc) {
        error = "missing value after '--video-ccso-bands'";
        return ParseStatus::kError;
      }
      int bands = 0;
      if (!parse_int(argv[++i], bands) || bands < 0 || bands > 6) {
        error = "--video-ccso-bands must be an integer in [0, 6]";
        return ParseStatus::kError;
      }
      options.video_ccso_bands = bands;
      options.video_ccso = true;
      continue;
    }

    if (argument == "--video-ccso-rd") {
      options.video_ccso_rd = true;
      options.video_ccso = true;
      continue;
    }

    if (argument == "--video-ccso-lambda") {
      if (i + 1 >= argc) {
        error = "missing value after '--video-ccso-lambda'";
        return ParseStatus::kError;
      }
      double lambda = 0.0;
      if (!parse_double(argv[++i], lambda) || !(lambda > 0.0)) {
        error = "--video-ccso-lambda needs a positive SSE-per-bit number";
        return ParseStatus::kError;
      }
      options.video_ccso_lambda = lambda;
      options.video_ccso_rd = true;
      options.video_ccso = true;
      continue;
    }

    if (argument == "--video-ccso-refit") {
      if (i + 1 >= argc) {
        error = "missing value after '--video-ccso-refit'";
        return ParseStatus::kError;
      }
      int passes = 0;
      if (!parse_int(argv[++i], passes) || passes < 0 || passes > 4) {
        error = "--video-ccso-refit must be an integer in [0, 4]";
        return ParseStatus::kError;
      }
      options.video_ccso_refit_passes = passes;
      options.video_ccso_rd = true;
      options.video_ccso = true;
      continue;
    }

    if (argument == "--video-ccso-unit-bits") {
      if (i + 1 >= argc) {
        error = "missing value after '--video-ccso-unit-bits'";
        return ParseStatus::kError;
      }
      double bits = 0.0;
      if (!parse_double(argv[++i], bits) || !(bits >= 0.0)) {
        error = "--video-ccso-unit-bits needs a non-negative bit count";
        return ParseStatus::kError;
      }
      options.video_ccso_unit_bits = bits;
      options.video_ccso_rd = true;
      options.video_ccso = true;
      continue;
    }

    if (argument == "--video-ccso-header-rd") {
      options.video_ccso_header_rd = true;
      options.video_ccso_rd = true;
      options.video_ccso = true;
      continue;
    }

    if (argument == "--video-intra-full") {
      options.video_intra_full = true;
      continue;
    }

    if (argument == "--video-pc-wiener") {
      options.video_pc_wiener = true;
      continue;
    }

    if (argument == "--video-cfl-ds-filter") {
      if (i + 1 >= argc) {
        error = "missing value after '--video-cfl-ds-filter'";
        return ParseStatus::kError;
      }
      int index = 0;
      if (!parse_int(argv[++i], index) || index < 0 || index > 3) {
        error = "--video-cfl-ds-filter must be an integer in [0, 3]";
        return ParseStatus::kError;
      }
      options.video_cfl_ds_filter = index;
      continue;
    }

    if (argument == "--video-cfl") {
      options.video_cfl = true;
      continue;
    }

    if (argument == "--video-qm") {
      options.video_qm = true;
      continue;
    }

    if (argument == "--video-user-qm") {
      options.video_qm = true;
      options.video_user_qm = true;
      continue;
    }

    if (argument == "--video-tcq") {
      options.video_tcq = true;
      continue;
    }

    if (argument == "--video-rdoq") {
      options.video_rdoq = true;
      continue;
    }

    if (argument == "--video-rdoq-scale") {
      if (i + 1 >= argc) {
        error = "missing value after '--video-rdoq-scale'";
        return ParseStatus::kError;
      }
      double scale = 0.0;
      if (!parse_double(argv[++i], scale) || scale <= 0.0) {
        error = "--video-rdoq-scale must be a positive number";
        return ParseStatus::kError;
      }
      options.video_rdoq_scale = scale;
      options.video_rdoq = true;
      continue;
    }

    if (argument == "--video-cctx") {
      options.video_cctx = true;
      continue;
    }

    if (argument == "--video-grain") {
      options.video_grain = true;
      continue;
    }

    if (argument == "--video-lossless") {
      options.video = true;
      options.video_lossless = true;
      continue;
    }

    if (argument == "--video-lossless-b") {
      options.video = true;
      options.video_lossless = true;
      options.video_lossless_b = true;
      continue;
    }

    if (argument == "--video-lossless-intra-period") {
      if (i + 1 >= argc) {
        error = "missing value after '--video-lossless-intra-period'";
        return ParseStatus::kError;
      }
      int value = 0;
      if (!parse_int(argv[++i], value) || value < 0) {
        error = "--video-lossless-intra-period must be a non-negative integer";
        return ParseStatus::kError;
      }
      options.video_lossless_intra_period = value;
      continue;
    }

    if (argument == "--video-idtx") {
      options.video_idtx = true;
      continue;
    }

    if (argument == "--video-mrl") {
      options.video_mrl = true;
      continue;
    }

    if (argument == "--video-ibp") {
      options.video_ibp = true;
      continue;
    }

    if (argument == "--video-dip") {
      options.video_dip = true;
      continue;
    }

    if (argument == "--video-ist") {
      options.video_ist = true;
      continue;
    }

    if (argument == "--video-idtx-1d") {
      options.video_idtx_1d = true;
      continue;
    }

    if (argument == "--video-mhccp") {
      options.video_mhccp = true;
      options.video_cfl = true;
      continue;
    }

    if (argument == "--video-drl-reorder") {
      options.video_drl_reorder = true;
      continue;
    }

    if (argument == "--video-drl-reorder-always") {
      options.video_drl_reorder = true;
      options.video_drl_reorder_always = true;
      continue;
    }

    if (argument == "--video-tlayers") {
      if (i + 1 >= argc) {
        error = "missing value after '--video-tlayers'";
        return ParseStatus::kError;
      }
      int value = 0;
      if (!parse_int(argv[++i], value) || value <= 0) {
        error = "--video-tlayers must be a positive integer";
        return ParseStatus::kError;
      }
      options.video_tlayers = value;
      continue;
    }

    if (argument == "--video-tlayer-qp-offsets") {
      if (i + 1 >= argc) {
        error = "missing value after '--video-tlayer-qp-offsets'";
        return ParseStatus::kError;
      }
      const std::string_view value = argv[++i];
      std::size_t start = 0;
      for (int layer = 0; layer < 4; ++layer) {
        const std::size_t comma = value.find(',', start);
        const std::string_view part =
            value.substr(start, comma == std::string_view::npos
                                    ? std::string_view::npos
                                    : comma - start);
        int offset = 0;
        if (!parse_int(part, offset) || offset < -63 || offset > 63) {
          error = "--video-tlayer-qp-offsets expects four comma-separated "
                  "integers in [-63, 63] (layers 0..3)";
          return ParseStatus::kError;
        }
        options.video_tlayer_qp_offset[static_cast<std::size_t>(layer)] =
            offset;
        if (comma == std::string_view::npos) {
          if (layer != 3) {
            error = "--video-tlayer-qp-offsets expects four comma-separated "
                    "integers (layers 0..3)";
            return ParseStatus::kError;
          }
          break;
        }
        if (layer == 3) {
          error = "--video-tlayer-qp-offsets expects exactly four values "
                  "(layers 0..3)";
          return ParseStatus::kError;
        }
        start = comma + 1;
      }
      options.video_tlayer_qp_given = true;
      continue;
    }

    if (argument == "--video-tlayer-anchor-residual") {
      options.video_tlayer_anchor_residual = true;
      continue;
    }

    if (argument == "--video-tlayer-refresh-threshold") {
      if (i + 1 >= argc) {
        error = "missing value after '--video-tlayer-refresh-threshold'";
        return ParseStatus::kError;
      }
      char* end = nullptr;
      const double value = std::strtod(argv[++i], &end);
      if (end == argv[i] || *end != '\0' || value < 0.0 ||
          value >= 30.0) {
        error = "--video-tlayer-refresh-threshold needs a number in [0, 30) "
                "dB (0 forces every layer-0 anchor)";
        return ParseStatus::kError;
      }
      options.video_tlayer_refresh_threshold = value;
      options.video_tlayer_anchor_residual = true;
      continue;
    }

    if (argument == "--video-tlayer-refresh-budget") {
      if (i + 1 >= argc) {
        error = "missing value after '--video-tlayer-refresh-budget'";
        return ParseStatus::kError;
      }
      int value = 0;
      if (!parse_int(argv[++i], value) || value < 0) {
        error = "--video-tlayer-refresh-budget needs a non-negative integer "
                "(extra payload bytes)";
        return ParseStatus::kError;
      }
      options.video_tlayer_refresh_budget = value;
      options.video_tlayer_anchor_residual = true;
      continue;
    }

    if (argument == "--video-tlayer-drop") {
      if (i + 1 >= argc) {
        error = "missing value after '--video-tlayer-drop'";
        return ParseStatus::kError;
      }
      int value = 0;
      if (!parse_int(argv[++i], value) || value < 0) {
        error = "--video-tlayer-drop must be a non-negative integer";
        return ParseStatus::kError;
      }
      options.video_tlayer_drop = value;
      continue;
    }

    if (argument == "--video-long-term") {
      options.video_long_term = true;
      continue;
    }

    if (argument == "--video-olk") {
      options.video_long_term = true;
      options.video_olk = true;
      continue;
    }

    if (argument == "--video-leading") {
      if (i + 1 >= argc) {
        error = "missing value after '--video-leading'";
        return ParseStatus::kError;
      }
      int value = 0;
      if (!parse_int(argv[++i], value) || value < 1 || value > 7) {
        error = "--video-leading must be an integer in [1, 7]";
        return ParseStatus::kError;
      }
      options.video_leading = value;
      continue;
    }

    if (argument == "--video-sef") {
      options.video_sef = true;
      continue;
    }

    if (argument == "--video-switch") {
      options.video_switch = true;
      continue;
    }

    if (argument == "--video-ras") {
      options.video_ras = true;
      continue;
    }

    if (argument == "--video-recon") {
      if (i + 1 >= argc) {
        error = "missing value after '--video-recon'";
        return ParseStatus::kError;
      }
      options.video_recon = argv[++i];
      continue;
    }

    if (argument == "--video-force-single") {
      options.video_force_single = true;
      continue;
    }

    if (argument == "--video-skip-collapse") {
      options.video_skip_collapse = true;
      continue;
    }

    if (argument == "--video-sb128") {
      options.video_sb128 = true;
      continue;
    }

    if (argument == "--video-sb256") {
      options.video_sb256 = true;
      continue;
    }

    if (argument == "--video-ext-partitions") {
      options.video_ext_partitions = true;
      continue;
    }

    if (argument == "--rc") {
      if (i + 1 >= argc) {
        error = "missing value after '--rc'";
        return ParseStatus::kError;
      }
      const std::string value = argv[++i];
      if (value != "cqp" && value != "crf" && value != "vbr") {
        error = "--rc must be cqp, crf or vbr";
        return ParseStatus::kError;
      }
      options.rc_mode = value;
      options.rc_given = true;
      continue;
    }

    if (argument == "--crf") {
      if (i + 1 >= argc) {
        error = "missing value after '--crf'";
        return ParseStatus::kError;
      }
      double value = 0.0;
      if (!parse_double(argv[++i], value) || value < 0.0 || value > 63.0) {
        error = "--crf must be a number in [0, 63]";
        return ParseStatus::kError;
      }
      options.rc_crf = value;
      options.rc_crf_given = true;
      continue;
    }

    if (argument == "--bitrate") {
      if (i + 1 >= argc) {
        error = "missing value after '--bitrate'";
        return ParseStatus::kError;
      }
      double value = 0.0;
      if (!parse_double(argv[++i], value) || value <= 0.0) {
        error = "--bitrate must be a positive number of kbps";
        return ParseStatus::kError;
      }
      options.rc_bitrate_kbps = value;
      options.rc_bitrate_given = true;
      continue;
    }

    if (argument == "--video-key-qp-offset") {
      if (i + 1 >= argc) {
        error = "missing value after '--video-key-qp-offset'";
        return ParseStatus::kError;
      }
      int offset = 0;
      if (!parse_int(argv[++i], offset) || offset < -63 || offset > 63) {
        error = "--video-key-qp-offset must be an integer in [-63, 63]";
        return ParseStatus::kError;
      }
      options.video_key_qp_offset = offset;
      options.video_key_qp_offset_given = true;
      continue;
    }

    // --video-tile-flush-min: implementation, then docs.  Delimited block so
    // it can be merged without collisions.
    if (argument == "--video-tile-flush-min") {
      options.video_tile_flush_min = true;
      continue;
    }
    // --video-tile-flush-min block end.

    // --video-drift-refresh: implementation, then docs.  Delimited block so it
    // can be merged without collisions.
    if (argument == "--video-drift-refresh") {
      options.video_drift_refresh = true;
      continue;
    }

    if (argument == "--video-drift-refresh-threshold") {
      if (i + 1 >= argc) {
        error = "missing value after '--video-drift-refresh-threshold'";
        return ParseStatus::kError;
      }
      double db = 0.0;
      if (!parse_double(argv[++i], db) || db <= 0.0 || db >= 30.0) {
        error = "--video-drift-refresh-threshold must be a drift allowance in "
                "(0, 30) dB";
        return ParseStatus::kError;
      }
      options.video_drift_refresh_threshold = db;
      continue;
    }

    if (argument == "--video-drift-refresh-qp") {
      if (i + 1 >= argc) {
        error = "missing value after '--video-drift-refresh-qp'";
        return ParseStatus::kError;
      }
      int drop = 0;
      if (!parse_int(argv[++i], drop) || drop < 0 || drop > 64) {
        error = "--video-drift-refresh-qp must be an integer in [0, 64]";
        return ParseStatus::kError;
      }
      options.video_drift_refresh_qp = drop;
      continue;
    }

    if (argument == "--video-drift-refresh-budget") {
      if (i + 1 >= argc) {
        error = "missing value after '--video-drift-refresh-budget'";
        return ParseStatus::kError;
      }
      int bytes = 0;
      if (!parse_int(argv[++i], bytes) || bytes < 0) {
        error = "--video-drift-refresh-budget must be a non-negative byte "
                "count";
        return ParseStatus::kError;
      }
      options.video_drift_refresh_budget = bytes;
      continue;
    }
    // --video-drift-refresh block end.

    if (argument == "--svc-layers") {
      if (i + 1 >= argc) {
        error = "missing value after '--svc-layers'";
        return ParseStatus::kError;
      }
      options.svc_layers = argv[++i];
      options.svc_given = true;
      continue;
    }

    if (argument == "--svc-temporal") {
      if (i + 1 >= argc) {
        error = "missing value after '--svc-temporal'";
        return ParseStatus::kError;
      }
      int value = 0;
      if (!parse_int(argv[++i], value) || value < 1 || value > 4) {
        error = "--svc-temporal must be an integer in [1, 4]";
        return ParseStatus::kError;
      }
      options.svc_temporal = value;
      options.svc_given = true;
      continue;
    }

    if (argument == "--svc-embedded") {
      if (i + 1 >= argc) {
        error = "missing value after '--svc-embedded'";
        return ParseStatus::kError;
      }
      int value = 0;
      if (!parse_int(argv[++i], value) || value < 1 || value > 8) {
        error = "--svc-embedded must be an integer in [1, 8]";
        return ParseStatus::kError;
      }
      options.svc_embedded = value;
      options.svc_given = true;
      continue;
    }

    if (argument == "--svc-inter-layer") {
      // Named rejection: AV2 v1.0.0 cannot carry inter-layer prediction.
      error =
          "--svc-inter-layer is not a conforming AV2 v1.0.0 feature: §6.8.2 "
          "requires lcr_dependent_xlayers_flag = 0, so no cross-xlayer "
          "reference or resampling can be signalled and there is no decoder "
          "process to gate.  Repro: av2enc in.y4m -o out.ivf --video --frames "
          "3 --svc-layers 0=640x360,1=1280x720 --svc-inter-layer.  Use "
          "--svc-embedded N for cross-mlayer (quality) refinement";
      return ParseStatus::kError;
    }

    if (argument == "--svc-lcr-id") {
      if (i + 1 >= argc) {
        error = "missing value after '--svc-lcr-id'";
        return ParseStatus::kError;
      }
      int value = 0;
      if (!parse_int(argv[++i], value) || value < 1 || value > 7) {
        error = "--svc-lcr-id must be an integer in [1, 7]";
        return ParseStatus::kError;
      }
      options.svc_lcr_id = value;
      options.svc_given = true;
      continue;
    }

    if (argument == "--oppoint") {
      if (i + 1 >= argc) {
        error = "missing value after '--oppoint'";
        return ParseStatus::kError;
      }
      int value = 0;
      if (!parse_int(argv[++i], value) || value < 0 || value > 6) {
        error = "--oppoint must be an integer in [0, 6]";
        return ParseStatus::kError;
      }
      options.svc_oppoint = value;
      options.svc_given = true;
      continue;
    }

    if (argument == "--alllayers") {
      options.svc_all_layers = true;
      options.svc_given = true;
      continue;
    }

    if (argument == "--spatial-layers") {
      if (i + 1 >= argc) {
        error = "missing value after '--spatial-layers'";
        return ParseStatus::kError;
      }
      int value = 0;
      if (!parse_int(argv[++i], value) || value < 1 || value > 30) {
        error = "--spatial-layers must be an integer in [1, 30]";
        return ParseStatus::kError;
      }
      options.svc_spatial_layers = value;
      options.svc_given = true;
      continue;
    }

    if (argument == "--video-min-block") {
      if (i + 1 >= argc) {
        error = "missing value after '--video-min-block'";
        return ParseStatus::kError;
      }
      int value = 0;
      if (!parse_int(argv[++i], value) || value < 4 || value > 64) {
        error = "--video-min-block must be an integer in [4, 64]";
        return ParseStatus::kError;
      }
      options.video_min_block = value;
      continue;
    }

    if (argument == "--video-max-depth") {
      if (i + 1 >= argc) {
        error = "missing value after '--video-max-depth'";
        return ParseStatus::kError;
      }
      int value = 0;
      if (!parse_int(argv[++i], value) || value < 0 || value > 8) {
        error = "--video-max-depth must be an integer in [0, 8]";
        return ParseStatus::kError;
      }
      options.video_max_depth = value;
      continue;
    }

    if (argument == "--video-me-range") {
      if (i + 1 >= argc) {
        error = "missing value after '--video-me-range'";
        return ParseStatus::kError;
      }
      int value = 0;
      if (!parse_int(argv[++i], value) || value < 1 || value > 256) {
        error = "--video-me-range must be an integer in [1, 256]";
        return ParseStatus::kError;
      }
      options.video_me_range = value;
      continue;
    }

    if (argument == "--video-trace") {
      options.video_trace = true;
      continue;
    }

    if (argument == "--still-ccso-bands") {
      if (i + 1 >= argc) {
        error = "missing value after '--still-ccso-bands'";
        return ParseStatus::kError;
      }
      int bands = 0;
      if (!parse_int(argv[++i], bands) || bands < 0 || bands > 6) {
        error = "--still-ccso-bands must be an integer in [0, 6]";
        return ParseStatus::kError;
      }
      options.still_ccso_bands = bands;
      continue;
    }

    if (argument == "--still-cdef-strengths") {
      if (i + 1 >= argc) {
        error = "missing value after '--still-cdef-strengths'";
        return ParseStatus::kError;
      }
      int strengths = 0;
      if (!parse_int(argv[++i], strengths) || strengths < 1 ||
          strengths > 8) {
        error = "--still-cdef-strengths must be an integer in [1, 8]";
        return ParseStatus::kError;
      }
      options.still_cdef_strengths = strengths;
      continue;
    }

    if (argument == "--still-depth") {
      if (i + 1 >= argc) {
        error = "missing value after '--still-depth'";
        return ParseStatus::kError;
      }
      int depth = 0;
      if (!parse_int(argv[++i], depth) || depth < 0 || depth > 8) {
        error = "--still-depth must be an integer in [0, 8]";
        return ParseStatus::kError;
      }
      options.still_depth = depth;
      continue;
    }

    if (argument == "--vram-cap") {
      if (i + 1 >= argc ||
          !parse_uint64(argv[++i], options.vram_cap_mib)) {
        error = "--vram-cap requires a non-negative MiB value";
        return ParseStatus::kError;
      }
      continue;
    }

    if (!argument.empty() && argument.front() == '-') {
      error = "unknown option '" + std::string(argument) + "'";
      return ParseStatus::kError;
    }

    if (!options.input.empty()) {
      error = "multiple input files specified";
      return ParseStatus::kError;
    }
    options.input = argv[i];
  }
  return ParseStatus::kOk;
}

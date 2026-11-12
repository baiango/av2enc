// still_dump: encode the first frame of a Y4M with the M1 still-picture encoder
// and dump the encoder-side reconstruction plus the exact StillOptions used.
//
// The dump is the reference half of tools/still_verify/verify.py: the IVF is
// produced by the real CLI, this tool reproduces the same encode and exposes
// StillResult.recon_{y,u,v}, which the CLI does not write anywhere.
//
// IMPORTANT: the options below must match the CLI flags for the same run.
// A depth/deblock/CfL mismatch silently compares two different encoders and
// produces "bitstream" errors that are really harness errors.
//
// Dump format (little-endian irrelevant, all bytes):
//   "StillDUMP1\n"
//   <one-line JSON header>\n
//   Y plane (width*height)
//   U plane (cw*ch)
//   V plane (cw*ch)
// JSON header carries dimensions, the StillOptions fields, result sizes and the
// AV2ENC_STILL_* environment variables that were visible to the library.

#include "pipeline/still/still_encoder.h"
#include "core/io/y4m.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

void usage(std::ostream& out) {
  out << "Usage: still_dump -i <input.y4m> -o <recon.dump> [options]\n"
         "\n"
         "Encodes the first Y4M frame and dumps StillResult.recon_y/u/v.\n"
         "Defaults mirror av2enc::still::StillOptions{}.\n"
         "\n"
         "Options (must match the CLI encode being verified):\n"
         "  -i, --input <file>      input Y4M (required, 8-bit 4:2:0)\n"
         "  -o, --output <file>     dump output path (required)\n"
         "  --depth N               max_partition_depth, 0..8 (CLI --still-depth)\n"
         "  --no-deblock            disable the deblocking filter\n"
         "  --full-intra            luma_dc_only = false (CLI --still-full-intra)\n"
         "  --no-cfl                disable CfL\n"
         "  --uv-full               enable the full chroma intra search\n"
         "  --cdef                  enable CDEF (CLI --still-cdef)\n"
         "  --cdef-strengths N      CDEF strength sets 1..8 (default 4)\n"
         "  --rdo-intra             rate-aware luma intra (CLI --still-rdo-intra)\n"
         "  --rdo-scale S           lambda sse_scale (default 0.125)\n"
         "  --rdoq                  enable coefficient RDOQ (CLI --still-rdoq)\n"
         "  --rdoq-scale S          RDOQ lambda sse_scale (default 0.125)\n"
         "  --tx-type               intra luma TX_TYPE search (CLI --still-tx-type)\n"
         "  --tx-type-scale S       lambda sse_scale (default 0.125)\n"
         "  --ccso                  enable band-offset-only CCSO (CLI --still-ccso)\n"
         "  --ccso-edge             edge-classified CCSO (CLI --still-ccso-edge)\n"
         "  --ccso-bands N          CCSO max_band_log2 0..6 (default 6)\n"
         "  --sb128                 use 128x128 superblocks (CLI --still-sb128)\n"
         "  --sb256                 signal 256x256 sequence SBs (CLI --still-sb256)\n"
         "  --ext-partitions        extended HORZ_3/VERT_3/4-way (--still-ext-partitions)\n"
         "  --qm                    signal a spec-defined QM set (CLI --still-qm)\n"
         "  --qm-level L            QM default matrix level 0..14 (default 3)\n"
         "  --ist                   enable intra secondary transform (CLI --still-ist)\n"
         "  --delta-q               enable DC/AC delta-Q search (CLI --still-dc-ac-delta-q)\n"
         "  --wiener                luma Wiener NS loop restoration\n"
         "  --chroma-lr             chroma Wiener NS loop restoration\n"
         "  --pc-wiener             luma PC-Wiener loop restoration\n"
         "  --gdf                   luma guided detail filter\n"
         "  --grain                 signal film grain (decoder synthesis)\n"
         "  --still-single             force one PARTITION_NONE superblock\n"
         "  --still-dc                 force DC intra for luma\n"
         "  --min-block-size N      partition floor in pixels (default 8)\n"
         "  --qp N                  base_q_idx (M1 pins 90)\n"
         "  --adaptive-cdf          enable CDF adaptation\n"
         "  --no-edge-filter        disable intra edge filtering\n"
         "  --print-options         print the resolved options as one JSON line\n"
         "  -h, --help\n"
         "\n"
         "Environment (read by the library, not by this tool):\n"
         "  AV2ENC_STILL_FORCE_Y=<mode>:<delta>, AV2ENC_STILL_FORCE_Y2, "
         "AV2ENC_STILL_BASE_ONLY,\n"
         "  AV2ENC_STILL_NO_DIR, AV2ENC_STILL_NO_EDGE, AV2ENC_STILL_UV_DC, "
         "AV2ENC_STILL_TRACE,\n"
         "  AV2ENC_STILL_UV_FULL, AV2ENC_STILL_FORCE_UV=<mode>:<delta>\n";
}

std::string json_escape(const std::string& in) {
  std::string out;
  out.reserve(in.size() + 8);
  for (const char c : in) {
    switch (c) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[7];
          std::snprintf(buf, sizeof(buf), "\\u%04x",
                        static_cast<unsigned>(static_cast<unsigned char>(c)));
          out += buf;
        } else {
          out += c;
        }
    }
  }
  return out;
}

// Writes the JSON fields for the env vars the M1 library consults. Kept as a
// literal list so a new knob is an explicit, visible change here.
std::string env_json() {
  static constexpr const char* kNames[] = {
      "AV2ENC_STILL_FORCE_Y",  "AV2ENC_STILL_FORCE_Y2", "AV2ENC_STILL_BASE_ONLY",
      "AV2ENC_STILL_NO_DIR",   "AV2ENC_STILL_NO_EDGE",  "AV2ENC_STILL_UV_DC",
      "AV2ENC_STILL_LUMA_DC",  "AV2ENC_STILL_TRACE",    "AV2ENC_STILL_SQUARE_ONLY",
      "AV2ENC_STILL_LEAF32",   "AV2ENC_STILL_ZERO_DC",  "AV2ENC_STILL_YMODE_CTX0",
      "AV2ENC_STILL_YMODE_CTX2", "AV2ENC_STILL_DCSIGN_CTX", "AV2ENC_STILL_NO_TXTYPE",
      "AV2ENC_STILL_UV_FULL",  "AV2ENC_STILL_FORCE_UV",
  };
  std::string out = "{";
  bool first = true;
  for (const char* name : kNames) {
    const char* value = std::getenv(name);
    if (value == nullptr) continue;
    if (!first) out += ',';
    first = false;
    out += '"';
    out += name;
    out += "\":\"";
    out += json_escape(value);
    out += '"';
  }
  out += '}';
  return out;
}

bool parse_int(std::string_view text, int& value) {
  if (text.empty()) return false;
  try {
    std::size_t used = 0;
    const int parsed = std::stoi(std::string(text), &used);
    if (used != text.size()) return false;
    value = parsed;
    return true;
  } catch (...) {
    return false;
  }
}

bool parse_double(std::string_view text, double& value) {
  if (text.empty()) return false;
  try {
    std::size_t used = 0;
    const double parsed = std::stod(std::string(text), &used);
    if (used != text.size() || !std::isfinite(parsed)) return false;
    value = parsed;
    return true;
  } catch (...) {
    return false;
  }
}

struct Cli {
  std::filesystem::path input;
  std::filesystem::path output;
  av2enc::still::StillOptions options;
  bool depth_overridden = false;
  bool print_options = false;
  bool help = false;
};

// Returns 0 on success, 2 on usage error.
int parse_args(int argc, char** argv, Cli& cli) {
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    const auto next_value = [&](const char* what, std::string& out) {
      if (i + 1 >= argc) {
        std::cerr << "still_dump: missing value after " << what << "\n";
        return false;
      }
      out = argv[++i];
      return true;
    };
    std::string value;
    if (arg == "-i" || arg == "--input") {
      if (!next_value("--input", value)) return 2;
      cli.input = value;
    } else if (arg == "-o" || arg == "--output") {
      if (!next_value("--output", value)) return 2;
      cli.output = value;
    } else if (arg == "--depth" || arg == "--still-depth") {
      if (!next_value("--depth", value)) return 2;
      int depth = 0;
      if (!parse_int(value, depth) || depth < 0 || depth > 8) {
        std::cerr << "still_dump: --depth must be an integer in [0, 8]\n";
        return 2;
      }
      cli.options.max_partition_depth = depth;
      cli.depth_overridden = true;
    } else if (arg == "--min-block-size") {
      if (!next_value("--min-block-size", value)) return 2;
      int size = 0;
      if (!parse_int(value, size) || size < 4 || size > 64) {
        std::cerr << "still_dump: --min-block-size must be in [4, 64]\n";
        return 2;
      }
      cli.options.min_block_size = size;
    } else if (arg == "--qp") {
      if (!next_value("--qp", value)) return 2;
      int qp = 0;
      if (!parse_int(value, qp) || qp < 0 || qp > 255) {
        std::cerr << "still_dump: --qp must be in [0, 255]\n";
        return 2;
      }
      cli.options.base_q_idx = static_cast<std::uint8_t>(qp);
    } else if (arg == "--no-deblock") {
      cli.options.enable_deblock = false;
    } else if (arg == "--full-intra") {
      cli.options.luma_dc_only = false;
    } else if (arg == "--no-cfl") {
      cli.options.enable_cfl = false;
    } else if (arg == "--uv-full") {
      cli.options.uv_full = true;
    } else if (arg == "--cdef") {
      cli.options.enable_cdef = true;
    } else if (arg == "--rdo-intra") {
      cli.options.rdo_intra = true;
    } else if (arg == "--rdo-scale") {
      if (!next_value("--rdo-scale", value)) return 2;
      double scale = 0.0;
      if (!parse_double(value, scale) || scale <= 0.0) {
        std::cerr << "still_dump: --rdo-scale must be positive\n";
        return 2;
      }
      cli.options.rdo_sse_scale = scale;
    } else if (arg == "--ccso") {
      cli.options.enable_ccso = true;
    } else if (arg == "--sb128") {
      cli.options.sb128 = true;
    } else if (arg == "--sb256") {
      cli.options.sb256 = true;
    } else if (arg == "--ext-partitions") {
      cli.options.ext_partitions = true;
    } else if (arg == "--ccso-edge") {
      cli.options.enable_ccso = true;
      cli.options.ccso_edge = true;
    } else if (arg == "--wiener") {
      cli.options.enable_wiener = true;
    } else if (arg == "--chroma-lr") {
      cli.options.enable_chroma_lr = true;
    } else if (arg == "--pc-wiener") {
      cli.options.enable_pc_wiener = true;
    } else if (arg == "--gdf") {
      cli.options.enable_gdf = true;
    } else if (arg == "--grain") {
      cli.options.enable_grain = true;
    } else if (arg == "--ccso-bands") {
      if (!next_value("--ccso-bands", value)) return 2;
      int bands = 0;
      if (!parse_int(value, bands) || bands < 0 || bands > 6) {
        std::cerr << "still_dump: --ccso-bands must be in [0, 6]\n";
        return 2;
      }
      cli.options.ccso_bands_log2 = bands;
    } else if (arg == "--tx-type") {
      cli.options.tx_type = true;
    } else if (arg == "--tx-type-scale") {
      if (!next_value("--tx-type-scale", value)) return 2;
      double scale = 0.0;
      if (!parse_double(value, scale) || scale <= 0.0) {
        std::cerr << "still_dump: --tx-type-scale must be positive\n";
        return 2;
      }
      cli.options.tx_type_sse_scale = scale;
    } else if (arg == "--rdoq") {
      cli.options.rdoq = true;
    } else if (arg == "--rdoq-scale") {
      if (!next_value("--rdoq-scale", value)) return 2;
      double scale = 0.0;
      if (!parse_double(value, scale) || scale <= 0.0) {
        std::cerr << "still_dump: --rdoq-scale must be positive\n";
        return 2;
      }
      cli.options.rdoq_sse_scale = scale;
    } else if (arg == "--cdef-strengths") {
      if (!next_value("--cdef-strengths", value)) return 2;
      int strengths = 0;
      if (!parse_int(value, strengths) || strengths < 1 || strengths > 8) {
        std::cerr << "still_dump: --cdef-strengths must be in [1, 8]\n";
        return 2;
      }
      cli.options.cdef_num_strengths = strengths;
    } else if (arg == "--ist") {
      cli.options.ist = true;
    } else if (arg == "--delta-q") {
      cli.options.delta_q = true;
    } else if (arg == "--qm") {
      cli.options.qm = true;
    } else if (arg == "--qm-level") {
      if (!next_value("--qm-level", value)) return 2;
      int level = 0;
      if (!parse_int(value, level) || level < 0 || level > 14) {
        std::cerr << "still_dump: --qm-level must be in [0, 14]\n";
        return 2;
      }
      cli.options.qm_level = level;
    } else if (arg == "--still-single") {
      cli.options.force_single_block = true;
    } else if (arg == "--still-dc") {
      cli.options.force_dc = true;
    } else if (arg == "--adaptive-cdf") {
      cli.options.adaptive_cdf = true;
    } else if (arg == "--no-edge-filter") {
      cli.options.enable_intra_edge_filter = false;
    } else if (arg == "--print-options") {
      cli.print_options = true;
    } else if (arg == "-h" || arg == "--help") {
      cli.help = true;
      return 0;
    } else {
      std::cerr << "still_dump: unknown argument: " << arg << "\n";
      return 2;
    }
  }
  if (cli.help) return 0;
  if (cli.options.rdo_intra && !cli.options.force_dc) {
    cli.options.luma_dc_only = false;
  }
  if (cli.input.empty()) {
    std::cerr << "still_dump: missing -i <input.y4m>\n";
    return 2;
  }
  if (cli.output.empty()) {
    std::cerr << "still_dump: missing -o <output.dump>\n";
    return 2;
  }
  return -1;  // parsed, continue
}

std::string options_json(const Cli& cli, std::uint32_t width,
                         std::uint32_t height,
                         const av2enc::still::StillResult& result) {
  const av2enc::still::StillOptions& o = cli.options;
  const std::uint32_t cw = (width + 1U) / 2U;
  const std::uint32_t ch = (height + 1U) / 2U;
  std::string j = "{";
  j += "\"width\":" + std::to_string(width);
  j += ",\"height\":" + std::to_string(height);
  j += ",\"chroma_width\":" + std::to_string(cw);
  j += ",\"chroma_height\":" + std::to_string(ch);
  j += ",\"base_q_idx\":" + std::to_string(o.base_q_idx);
  j += ",\"enable_cfl\":";
  j += o.enable_cfl ? "true" : "false";
  j += ",\"enable_deblock\":";
  j += o.enable_deblock ? "true" : "false";
  j += ",\"enable_intra_edge_filter\":";
  j += o.enable_intra_edge_filter ? "true" : "false";
  j += ",\"adaptive_cdf\":";
  j += o.adaptive_cdf ? "true" : "false";
  j += ",\"min_block_size\":" + std::to_string(o.min_block_size);
  j += ",\"max_partition_depth\":" + std::to_string(o.max_partition_depth);
  j += ",\"depth_overridden\":";
  j += cli.depth_overridden ? "true" : "false";
  j += ",\"force_single_block\":";
  j += o.force_single_block ? "true" : "false";
  j += ",\"force_dc\":";
  j += o.force_dc ? "true" : "false";
  j += ",\"luma_dc_only\":";
  j += o.luma_dc_only ? "true" : "false";
  j += ",\"uv_full\":";
  j += o.uv_full ? "true" : "false";
  j += ",\"rdo_intra\":";
  j += o.rdo_intra ? "true" : "false";
  j += ",\"rdo_sse_scale\":" + std::to_string(o.rdo_sse_scale);
  j += ",\"rdoq\":";
  j += o.rdoq ? "true" : "false";
  j += ",\"rdoq_sse_scale\":" + std::to_string(o.rdoq_sse_scale);
  j += ",\"tx_type\":";
  j += o.tx_type ? "true" : "false";
  j += ",\"tx_type_sse_scale\":" + std::to_string(o.tx_type_sse_scale);
  j += ",\"enable_ccso\":";
  j += o.enable_ccso ? "true" : "false";
  j += ",\"ccso_bands_log2\":" + std::to_string(o.ccso_bands_log2);
  j += ",\"sb128\":";
  j += o.sb128 ? "true" : "false";
  j += ",\"sb256\":";
  j += o.sb256 ? "true" : "false";
  j += ",\"ext_partitions\":";
  j += o.ext_partitions ? "true" : "false";
  j += ",\"qm\":";
  j += o.qm ? "true" : "false";
  j += ",\"qm_level\":" + std::to_string(o.qm_level);
  j += ",\"ist\":";
  j += o.ist ? "true" : "false";
  j += ",\"delta_q\":";
  j += o.delta_q ? "true" : "false";
  j += ",\"ccso_edge\":";
  j += o.ccso_edge ? "true" : "false";
  j += ",\"enable_wiener\":";
  j += o.enable_wiener ? "true" : "false";
  j += ",\"enable_chroma_lr\":";
  j += o.enable_chroma_lr ? "true" : "false";
  j += ",\"enable_pc_wiener\":";
  j += o.enable_pc_wiener ? "true" : "false";
  j += ",\"enable_gdf\":";
  j += o.enable_gdf ? "true" : "false";
  j += ",\"enable_grain\":";
  j += o.enable_grain ? "true" : "false";
  j += ",\"enable_cdef\":";
  j += o.enable_cdef ? "true" : "false";
  j += ",\"cdef_num_strengths\":" + std::to_string(o.cdef_num_strengths);
  j += ",\"tile_bytes\":" + std::to_string(result.tile_bytes);
  j += ",\"frame_bytes\":" + std::to_string(result.frame_bytes);
  j += ",\"leaves\":" + std::to_string(result.leaves);
  j += ",\"env\":" + env_json();
  j += '}';
  return j;
}

bool write_all(const std::filesystem::path& path, const std::string& header,
               const std::vector<std::uint8_t>& y,
               const std::vector<std::uint8_t>& u,
               const std::vector<std::uint8_t>& v) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) return false;
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  const auto write_plane = [&out](const std::vector<std::uint8_t>& plane) {
    if (plane.empty()) return true;
    out.write(reinterpret_cast<const char*>(plane.data()),
              static_cast<std::streamsize>(plane.size()));
    return static_cast<bool>(out);
  };
  return write_plane(y) && write_plane(u) && write_plane(v) &&
         static_cast<bool>(out);
}

}  // namespace

int main(int argc, char** argv) {
  Cli cli;
  const int parse_status = parse_args(argc, argv, cli);
  if (parse_status == 0) {
    usage(std::cout);
    return 0;
  }
  if (parse_status > 0) return parse_status;

  av2enc::Y4mReader reader(cli.input);
  std::string error;
  if (!reader.open(error)) {
    std::cerr << "still_dump: " << error << "\n";
    return 1;
  }
  av2enc::Y4mFrame frame;
  const av2enc::Y4mReadResult read_result = reader.read_frame(frame, error);
  if (read_result == av2enc::Y4mReadResult::kEof) {
    std::cerr << "still_dump: input has no frames\n";
    return 1;
  }
  if (read_result == av2enc::Y4mReadResult::kError) {
    std::cerr << "still_dump: " << error << "\n";
    return 1;
  }

  av2enc::still::StillPicture picture;
  picture.width = frame.width();
  picture.height = frame.height();
  picture.y = frame.y();
  picture.u = frame.u();
  picture.v = frame.v();

  av2enc::still::StillResult result;
  if (!av2enc::still::encode_still_picture(picture, cli.options, result, error)) {
    std::cerr << "still_dump: " << error << "\n";
    return 1;
  }

  const std::string json = options_json(cli, picture.width, picture.height,
                                        result);
  const std::string header = "StillDUMP1\n" + json + "\n";
  if (!write_all(cli.output, header, result.recon_y, result.recon_u,
                 result.recon_v)) {
    std::cerr << "still_dump: failed to write " << cli.output << "\n";
    return 1;
  }

  if (cli.print_options) {
    std::cout << json << "\n";
  }
  std::cout << "still_dump: " << picture.width << "x" << picture.height
            << " depth=" << cli.options.max_partition_depth
            << " deblock=" << (cli.options.enable_deblock ? 1 : 0)
            << " full_intra=" << (cli.options.luma_dc_only ? 0 : 1)
            << " tile=" << result.tile_bytes << "B"
            << " frame=" << result.frame_bytes << "B -> " << cli.output
            << "\n";
  return 0;
}

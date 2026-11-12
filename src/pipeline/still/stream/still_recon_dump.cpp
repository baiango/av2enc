// Encoder-reconstruction dump harness for the integrated still-picture path.
//
// The CLI only writes the bitstream.  Gate scripts need the encoder's own
// reconstruction to prove that a decoder reproduces it sample for sample, which
// is the only way to catch a prediction/syntax mismatch that still decodes.
// This harness mirrors the still-option mapping in src/main.cpp and writes the
// three reconstruction planes as Y4M 4:2:0.
//
// Usage:
//   still_recon_dump -i in.y4m -o recon.y4m [--still] [tool flags...]
//
// It is a test tool: it is not installed and it never writes a bitstream.

#include "core/io/y4m.h"
#include "pipeline/still/still_encoder.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

bool parse_int(std::string_view text, int& value) {
  if (text.empty()) return false;
  char* end = nullptr;
  std::string copy(text);
  const long parsed = std::strtol(copy.c_str(), &end, 10);
  if (end == nullptr || *end != '\0') return false;
  value = static_cast<int>(parsed);
  return true;
}

bool parse_double(std::string_view text, double& value) {
  if (text.empty()) return false;
  std::string copy(text);
  char* end = nullptr;
  const double parsed = std::strtod(copy.c_str(), &end);
  if (end == nullptr || *end != '\0') return false;
  value = parsed;
  return true;
}

void usage() {
  std::cerr
      << "usage: still_recon_dump -i in.y4m -o recon.y4m [--still] [flags]\n"
         "  mirrors the src/main.cpp option mapping for the CPU still path\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path input;
  std::filesystem::path output;
  int qp = 90;
  av2enc::still::StillOptions options;

  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      usage();
      return 0;
    }
    if (arg == "-i" || arg == "--input") {
      if (i + 1 >= argc) {
        usage();
        return 2;
      }
      input = argv[++i];
      continue;
    }
    if (arg == "-o" || arg == "--output") {
      if (i + 1 >= argc) {
        usage();
        return 2;
      }
      output = argv[++i];
      continue;
    }
    if (arg == "--qp") {
      if (i + 1 >= argc || !parse_int(argv[++i], qp)) return 2;
      continue;
    }
    if (arg == "--still") continue;
    if (arg == "--no-cfl") {
      options.enable_cfl = false;
      continue;
    }
    if (arg == "--no-deblock") {
      options.enable_deblock = false;
      continue;
    }
    if (arg == "--still-single") {
      options.force_single_block = true;
      continue;
    }
    if (arg == "--still-dc") {
      options.force_dc = true;
      continue;
    }
    if (arg == "--still-full-intra") {
      options.luma_dc_only = false;
      continue;
    }
    if (arg == "--still-uv-full") {
      options.uv_full = true;
      continue;
    }
    if (arg == "--still-rdo-intra") {
      options.rdo_intra = true;
      options.luma_dc_only = false;
      continue;
    }
    if (arg == "--still-rdo-scale") {
      if (i + 1 >= argc || !parse_double(argv[++i], options.rdo_sse_scale)) {
        return 2;
      }
      continue;
    }
    if (arg == "--still-rdoq") {
      options.rdoq = true;
      continue;
    }
    if (arg == "--still-rdoq-scale") {
      if (i + 1 >= argc || !parse_double(argv[++i], options.rdoq_sse_scale)) {
        return 2;
      }
      continue;
    }
    if (arg == "--still-tx-type") {
      options.tx_type = true;
      continue;
    }
    if (arg == "--still-tx-type-scale") {
      if (i + 1 >= argc || !parse_double(argv[++i], options.tx_type_sse_scale)) {
        return 2;
      }
      continue;
    }
    if (arg == "--still-cdef") {
      options.enable_cdef = true;
      continue;
    }
    if (arg == "--still-cdef-strengths") {
      if (i + 1 >= argc || !parse_int(argv[++i], options.cdef_num_strengths)) {
        return 2;
      }
      continue;
    }
    if (arg == "--still-ccso") {
      options.enable_ccso = true;
      continue;
    }
    if (arg == "--still-ccso-bands") {
      if (i + 1 >= argc || !parse_int(argv[++i], options.ccso_bands_log2)) {
        return 2;
      }
      continue;
    }
    if (arg == "--still-ibp") {
      options.enable_ibp = true;
      continue;
    }
    if (arg == "--still-mrl") {
      options.enable_mrl = true;
      options.luma_dc_only = false;
      continue;
    }
    if (arg == "--still-cfl-ds") {
      if (i + 1 >= argc || !parse_int(argv[++i], options.cfl_ds_filter)) {
        return 2;
      }
      continue;
    }
    if (arg == "--still-cfl-explicit") {
      options.cfl_explicit = true;
      continue;
    }
    if (arg == "--still-dip") {
      options.enable_dip = true;
      continue;
    }
    if (arg == "--still-palette") {
      options.palette = true;
      continue;
    }
    if (arg == "--still-ibc") {
      options.ibc = true;
      continue;
    }
    if (arg == "--still-ibc-search") {
      if (i + 1 >= argc || !parse_int(argv[++i], options.ibc_search_step)) {
        return 2;
      }
      continue;
    }
    if (arg == "--still-mhccp") {
      options.mhccp = true;
      continue;
    }
    if (arg == "--still-sb128") {
      options.sb128 = true;
      continue;
    }
    if (arg == "--still-sb256") {
      options.sb256 = true;
      continue;
    }
    if (arg == "--still-tcq") {
      options.tcq = true;
      continue;
    }
    if (arg == "--still-ist") {
      options.ist = true;
      continue;
    }
    if (arg == "--still-cctx") {
      options.cctx = true;
      continue;
    }
    if (arg == "--still-tx-partition") {
      options.tx_partition = true;
      continue;
    }
    if (arg == "--still-idtx") {
      options.idtx = true;
      continue;
    }
    if (arg == "--still-qm") {
      options.qm = true;
      continue;
    }
    if (arg == "--still-user-qm") {
      options.qm = true;
      options.user_qm = true;
      continue;
    }
    if (arg == "--still-depth") {
      if (i + 1 >= argc || !parse_int(argv[++i], options.max_partition_depth)) {
        return 2;
      }
      continue;
    }
    if (arg == "--still-per-segment-q") {
      options.per_segment_q = true;
      continue;
    }
    if (arg == "--still-seg-delta") {
      if (i + 1 >= argc || !parse_int(argv[++i], options.per_segment_q_delta)) {
        return 2;
      }
      continue;
    }
    std::cerr << "still_recon_dump: unknown option '" << arg << "'\n";
    return 2;
  }

  if (input.empty() || output.empty()) {
    usage();
    return 2;
  }

  av2enc::Y4mReader reader(input);
  std::string error;
  if (!reader.open(error)) {
    std::cerr << "still_recon_dump: " << error << "\n";
    return 1;
  }
  av2enc::Y4mFrame frame;
  if (reader.read_frame(frame, error) != av2enc::Y4mReadResult::kFrame) {
    std::cerr << "still_recon_dump: " << (error.empty() ? "no frames" : error)
              << "\n";
    return 1;
  }

  av2enc::still::StillPicture picture;
  picture.width = frame.width();
  picture.height = frame.height();
  picture.y = frame.y();
  picture.u = frame.u();
  picture.v = frame.v();

  options.base_q_idx = static_cast<std::uint8_t>(qp);
  av2enc::still::StillResult result;
  if (!av2enc::still::encode_still_picture(picture, options, result, error)) {
    std::cerr << "still_recon_dump: " << error << "\n";
    return 1;
  }

  const std::uint32_t width = frame.width();
  const std::uint32_t height = frame.height();
  const std::uint32_t chroma_w = (width + 1U) / 2U;
  const std::uint32_t chroma_h = (height + 1U) / 2U;

  std::ofstream out(output, std::ios::binary);
  if (!out) {
    std::cerr << "still_recon_dump: cannot open " << output << "\n";
    return 1;
  }
  const av2enc::Y4mHeader& header = reader.header();
  out << "YUV4MPEG2 W" << width << " H" << height << " F"
      << (header.fps_num == 0 ? 25U : header.fps_num) << ":"
      << (header.fps_den == 0 ? 1U : header.fps_den) << " Ip A0:0 "
      << header.colorspace << "\n";
  out << "FRAME\n";
  const auto write_plane = [&out](const std::vector<std::uint8_t>& plane) {
    out.write(reinterpret_cast<const char*>(plane.data()),
              static_cast<std::streamsize>(plane.size()));
  };
  if (result.recon_y.size() !=
          static_cast<std::size_t>(width) * static_cast<std::size_t>(height) ||
      result.recon_u.size() !=
          static_cast<std::size_t>(chroma_w) * static_cast<std::size_t>(chroma_h) ||
      result.recon_v.size() !=
          static_cast<std::size_t>(chroma_w) * static_cast<std::size_t>(chroma_h)) {
    std::cerr << "still_recon_dump: reconstruction plane size mismatch\n";
    return 1;
  }
  write_plane(result.recon_y);
  write_plane(result.recon_u);
  write_plane(result.recon_v);
  if (!out) {
    std::cerr << "still_recon_dump: write failed\n";
    return 1;
  }
  return 0;
}

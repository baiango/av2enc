// Minimal CLI for the chroma-format still writer.
//
// Usage:
//   chroma_still <input.y4m> -o <out.ivf> [--format 420|422|444|400]
//                [--recon <out.y4m>] [--qp 90]
//
// The input is the same 8-bit 4:2:0 Y4M the M0/M1 tools accept.  Chroma is
// upsampled by the existing format component (sample replication) to the
// requested format.  One still frame is written.
#include "formats/chroma/chroma_headers.h"
#include "formats/chroma/chroma_still.h"
#include "core/io/ivf.h"
#include "core/io/y4m.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

namespace {

struct Options {
  std::filesystem::path input;
  std::filesystem::path output;
  std::filesystem::path recon;
  av2enc::chroma::ChromaFormat format = av2enc::chroma::ChromaFormat::k422;
  int qp = 90;
  bool inspect = false;
  bool help = false;
};

void usage(std::ostream& out) {
  out << "usage: chroma_still <input.y4m> -o <out.ivf> [options]\n"
         "\n"
         "  --format <420|422|444|400>  chroma format (default 422)\n"
         "  --recon <out.y4m>           write the reconstructed visible frame\n"
         "  --qp <N>                    base_q_idx (default 90; 1..255, but\n"
         "                              this writer only pins the qctx-0 slice)\n"
         "  --inspect                   parse and print the sequence header\n"
         "  -h, --help                  show this help\n";
}

bool parse_format(std::string_view text, av2enc::chroma::ChromaFormat& format) {
  const auto parsed = av2enc::chroma::chroma_format_from_name(text);
  if (!parsed) {
    return false;
  }
  format = *parsed;
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string_view argument = argv[i];
    if (argument == "-h" || argument == "--help") {
      options.help = true;
      break;
    }
    if (argument == "-o" || argument == "--output") {
      if (i + 1 >= argc) {
        std::cerr << "missing value after -o\n";
        return 2;
      }
      options.output = argv[++i];
      continue;
    }
    if (argument == "--recon") {
      if (i + 1 >= argc) {
        std::cerr << "missing value after --recon\n";
        return 2;
      }
      options.recon = argv[++i];
      continue;
    }
    if (argument == "--format") {
      if (i + 1 >= argc || !parse_format(argv[++i], options.format)) {
        std::cerr << "--format must be 420, 422, 444 or 400\n";
        return 2;
      }
      continue;
    }
    if (argument == "--inspect") {
      options.inspect = true;
      continue;
    }
    if (argument == "--qp") {
      if (i + 1 >= argc) {
        std::cerr << "missing value after --qp\n";
        return 2;
      }
      char* end = nullptr;
      const long value = std::strtol(argv[++i], &end, 10);
      if (end == nullptr || *end != '\0' || value < 1 || value > 255) {
        std::cerr << "--qp must be in [1, 255]\n";
        return 2;
      }
      options.qp = static_cast<int>(value);
      continue;
    }
    if (!argument.empty() && argument.front() == '-') {
      std::cerr << "unknown option " << argument << "\n";
      return 2;
    }
    if (!options.input.empty()) {
      std::cerr << "multiple input files\n";
      return 2;
    }
    options.input = argv[i];
  }

  if (options.help) {
    usage(std::cout);
    return 0;
  }
  if (options.input.empty() || options.output.empty()) {
    usage(std::cerr);
    return 2;
  }
  if (options.qp != 90) {
    std::cerr << "chroma_still pins --qp 90: the generated CDF tables are the "
                 "qctx-0 slice and the quantiser arithmetic is validated for "
                 "qlookup(90) = 528\n";
    return 2;
  }

  std::string error;
  av2enc::Y4mReader reader(options.input);
  if (!reader.open(error)) {
    std::cerr << "av2enc: " << error << "\n";
    return 1;
  }
  const av2enc::Y4mHeader& header = reader.header();
  av2enc::Y4mFrame frame;
  const av2enc::Y4mReadResult read = reader.read_frame(frame, error);
  if (read != av2enc::Y4mReadResult::kFrame) {
    std::cerr << "av2enc: " << (error.empty() ? "no frame" : error) << "\n";
    return 1;
  }

  av2enc::chroma::ChromaPicture picture;
  if (!av2enc::chroma::make_chroma_picture(
          static_cast<int>(frame.width()), static_cast<int>(frame.height()),
          options.format, frame.y(), frame.u(), frame.v(), picture, error)) {
    std::cerr << "av2enc: " << error << "\n";
    return 1;
  }

  av2enc::chroma::ChromaStillOptions still_options;
  still_options.base_q_idx = static_cast<std::uint8_t>(options.qp);

  av2enc::chroma::ChromaStillResult result;
  if (!av2enc::chroma::encode_chroma_still_picture(picture, still_options,
                                                   result, error)) {
    std::cerr << "av2enc: " << error << "\n";
    return 1;
  }

  av2enc::IvfHeader ivf;
  ivf.width = static_cast<std::uint16_t>(frame.width());
  ivf.height = static_cast<std::uint16_t>(frame.height());
  ivf.fps_num = header.fps_num;
  ivf.fps_den = header.fps_den;

  av2enc::IvfWriter writer;
  if (!writer.open(options.output, ivf, error)) {
    std::cerr << "av2enc: " << error << "\n";
    return 1;
  }
  if (!writer.write_frame(result.frame, 0, error)) {
    std::cerr << "av2enc: " << error << "\n";
    return 1;
  }
  if (!writer.close(error)) {
    std::cerr << "av2enc: " << error << "\n";
    return 1;
  }

  if (!options.recon.empty()) {
    std::FILE* file = std::fopen(options.recon.string().c_str(), "wb");
    if (file == nullptr) {
      std::cerr << "av2enc: cannot open " << options.recon << "\n";
      return 1;
    }
    const bool ok = av2enc::chroma::write_recon_y4m(
        result, file, header.fps_num, header.fps_den, error);
    std::fclose(file);
    if (!ok) {
      std::cerr << "av2enc: " << error << "\n";
      return 1;
    }
  }

  std::cout << "chroma_still: " << av2enc::chroma::chroma_format_info(options.format).name
            << " " << frame.width() << "x" << frame.height()
            << " base_q_idx " << options.qp << ", " << result.leaves
            << " leaf blocks, tile " << result.tile_bytes
            << " bytes, temporal unit " << result.frame_bytes << " bytes\n";

  if (options.inspect) {
    av2enc::chroma::SequenceHeaderInfo info;
    if (!av2enc::chroma::find_sequence_header(result.frame, info, error)) {
      std::cerr << "av2enc: " << error << "\n";
      return 1;
    }
    const char* chroma_name =
        av2enc::chroma::chroma_format_info(info.chroma).name;
    std::cout << "sequence_header: chroma_format_idc=" << info.chroma_format_idc
              << " (" << chroma_name << ")"
              << " Monochrome=" << (info.monochrome ? 1 : 0)
              << " SubsamplingX=" << info.subsampling_x
              << " SubsamplingY=" << info.subsampling_y
              << " NumPlanes=" << info.num_planes
              << " bit_depth_idc=" << info.bit_depth_idc
              << " (" << info.bit_depth << "-bit)"
              << " profile=" << info.seq_profile_idc
              << " max=" << info.max_frame_width << "x"
              << info.max_frame_height
              << " enable_sdp=" << (info.enable_sdp ? 1 : 0)
              << " enable_cfl_intra=" << (info.enable_cfl_intra ? 1 : 0)
              << " enable_mhccp=" << (info.enable_mhccp ? 1 : 0)
              << " enable_chroma_dctonly="
              << (info.enable_chroma_dctonly ? 1 : 0)
              << " separate_uv_delta_q="
              << (info.separate_uv_delta_q ? 1 : 0)
              << "\n";
  }
  return 0;
}

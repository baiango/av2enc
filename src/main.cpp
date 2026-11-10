#include "bitstream/m0_headers.h"
#include "encoder/m0_still.h"
#include "io/ivf.h"
#include "io/y4m.h"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

struct Options {
  std::filesystem::path input;
  std::filesystem::path output;
  std::uint64_t max_frames = 0;  // 0 = all frames
  int qp = 90;
  bool adaptive_cdf = false;
  bool help = false;
};

enum class ParseStatus {
  kOk,
  kHelp,
  kError,
};

void print_usage(std::ostream& out) {
  out << "Usage: av2enc <input.y4m> -o <output.ivf> [--frames N] [--qp Q]\n"
         "              [--adaptive-cdf]\n"
         "\n"
         "M0 still-picture encoder: 8-bit 4:2:0, one key frame, one tile,\n"
         "64x64 superblocks, 32x32 DCT, DC intra only, no loop filters.\n"
         "Writes one AV2 frame (TD + SEQ + CLK) into an IVF with fourcc AV02.\n"
         "Input dimensions must be even (4:2:0); edge superblocks may be partial.\n"
         "\n"
         "Options:\n"
         "  -o, --output <file>  Output IVF path (required)\n"
         "  --frames <N>         M0 encodes one frame; values > 1 are clamped to 1\n"
         "  --qp <Q>             base_q_idx (default: 90; M0 pins Q = 90)\n"
         "  --adaptive-cdf       Enable normative §8.2.6 CDF adaptation\n"
         "                       (default off: frozen CDFs, pinned bitstream)\n"
         "  -h, --help           Show this help\n";
}

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

ParseStatus parse_arguments(int argc, char** argv, Options& options, std::string& error) {
  for (int i = 1; i < argc; ++i) {
    const std::string_view argument = argv[i];

    if (argument == "-h" || argument == "--help") {
      options.help = true;
      return ParseStatus::kHelp;
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
      if (!parse_int(argv[++i], qp) || qp < 0 || qp > 255) {
        error = "--qp must be an integer in [0, 255]";
        return ParseStatus::kError;
      }
      options.qp = qp;
      continue;
    }

    if (argument == "--adaptive-cdf") {
      options.adaptive_cdf = true;
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

  if (options.input.empty()) {
    error = "no input file specified";
    return ParseStatus::kError;
  }
  if (options.output.empty()) {
    error = "no output file specified (use -o <output.ivf>)";
    return ParseStatus::kError;
  }
  return ParseStatus::kOk;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  std::string error;
  const ParseStatus status = parse_arguments(argc, argv, options, error);
  if (status == ParseStatus::kHelp) {
    print_usage(std::cout);
    return 0;
  }
  if (status == ParseStatus::kError) {
    std::cerr << "av2enc: " << error << "\n\n";
    print_usage(std::cerr);
    return 2;
  }

  av2enc::Y4mReader reader(options.input);
  if (!reader.open(error)) {
    std::cerr << "av2enc: " << error << "\n";
    return 1;
  }

  const av2enc::Y4mHeader& y4m = reader.header();
  const std::uint16_t max_dimension = std::numeric_limits<std::uint16_t>::max();
  if (y4m.width > max_dimension || y4m.height > max_dimension) {
    std::cerr << "av2enc: error: Y4M dimensions " << y4m.width << "x" << y4m.height
              << " exceed the IVF limit of 65535x65535\n";
    return 1;
  }

  av2enc::IvfHeader ivf;
  ivf.width = static_cast<std::uint16_t>(y4m.width);
  ivf.height = static_cast<std::uint16_t>(y4m.height);
  ivf.fps_num = y4m.fps_num;
  ivf.fps_den = y4m.fps_den;

  av2enc::IvfWriter writer;
  if (!writer.open(options.output, ivf, error)) {
    std::cerr << "av2enc: " << error << "\n";
    return 1;
  }

  const auto fail_encode = [&](const std::string& message) {
    std::string close_error;
    writer.close(close_error);
    std::cerr << "av2enc: error: " << message << "\n";
    if (!close_error.empty()) {
      std::cerr << "av2enc: " << close_error << "\n";
    }
    return 1;
  };

  if (y4m.width % 2U != 0U || y4m.height % 2U != 0U) {
    std::cerr << "av2enc: error: M0 4:2:0 requires even dimensions, got "
              << y4m.width << "x" << y4m.height << "\n";
    return 1;
  }
  if (options.qp != 90) {
    std::cerr << "av2enc: error: M0 pins --qp 90 (coefficient CDF qctx slice 0)\n";
    return 1;
  }

  av2enc::Y4mFrame frame;
  const av2enc::Y4mReadResult read_result = reader.read_frame(frame, error);
  if (read_result == av2enc::Y4mReadResult::kEof) {
    return fail_encode("input has no frames");
  }
  if (read_result == av2enc::Y4mReadResult::kError) {
    return fail_encode(error);
  }

  av2enc::m0::StillPicture picture;
  picture.width = frame.width();
  picture.height = frame.height();
  picture.y = frame.y();
  picture.u = frame.u();
  picture.v = frame.v();

  av2enc::m0::StillPictureConfig config;
  config.max_frame_width = static_cast<std::uint16_t>(y4m.width);
  config.max_frame_height = static_cast<std::uint16_t>(y4m.height);
  config.base_q_idx = static_cast<std::uint8_t>(options.qp);
  // Frozen CDFs are the default M0 bitstream. Adaptive coding is an opt-in
  // normative §8.2.6 mode used to test the CDF-update path.
  config.enable_cdf_adaptation = options.adaptive_cdf;

  av2enc::m0::StillEncodeResult encoded;
  if (!av2enc::m0::encode_still_picture(picture, config, encoded, error)) {
    return fail_encode(error);
  }
  if (!writer.write_frame(encoded.frame, 0, error)) {
    return fail_encode(error);
  }

  if (!writer.close(error)) {
    std::cerr << "av2enc: " << error << "\n";
    return 1;
  }

  std::cout << "av2enc: M0 still-picture encode complete\n"
            << "  input:   " << options.input.string() << " (" << y4m.width << "x"
            << y4m.height << ", " << y4m.colorspace << ", " << y4m.fps_num << "/"
            << y4m.fps_den << " fps)\n"
            << "  output:  " << options.output.string() << " (" << writer.bytes_written()
            << " bytes IVF)\n"
            << "  frames:  1, base_q_idx " << options.qp << ", tile "
            << encoded.tile_bytes << " bytes, temporal unit " << encoded.frame_bytes
            << " bytes\n";
  return 0;
}

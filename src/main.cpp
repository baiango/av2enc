#include "io/ivf.h"
#include "io/y4m.h"

#include <array>
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
  int qp = 32;
  bool help = false;
};

enum class ParseStatus {
  kOk,
  kHelp,
  kError,
};

void print_usage(std::ostream& out) {
  out << "Usage: av2enc <input.y4m> -o <output.ivf> [--frames N] [--qp Q]\n"
         "\n"
         "M0 scaffold: reads 8-bit 4:2:0 Y4M and writes placeholder IVF frames.\n"
         "No AV2 bitstream syntax is emitted yet.\n"
         "\n"
         "Options:\n"
         "  -o, --output <file>  Output IVF path (required)\n"
         "  --frames <N>         Encode at most N frames (default: all)\n"
         "  --qp <Q>             Quantizer for the future encoder (default: 32;\n"
         "                       recorded in the placeholder payload only)\n"
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

void append_le32(std::vector<std::uint8_t>& out, std::uint32_t value) {
  out.push_back(static_cast<std::uint8_t>(value & 0xFFU));
  out.push_back(static_cast<std::uint8_t>((value >> 8U) & 0xFFU));
  out.push_back(static_cast<std::uint8_t>((value >> 16U) & 0xFFU));
  out.push_back(static_cast<std::uint8_t>((value >> 24U) & 0xFFU));
}

void append_le64(std::vector<std::uint8_t>& out, std::uint64_t value) {
  for (unsigned i = 0; i < 8; ++i) {
    out.push_back(static_cast<std::uint8_t>((value >> (8U * i)) & 0xFFU));
  }
}

// Placeholder payload so the IVF container and CLI plumbing can be exercised
// before the real AV2 syntax exists. It is NOT an AV2 frame and must be
// replaced by the bitstream/entropy path in the next milestone.
std::vector<std::uint8_t> make_placeholder_payload(std::uint64_t frame_index,
                                                   const av2enc::Y4mFrame& frame,
                                                   int qp) {
  constexpr std::array<std::uint8_t, 8> kMagic = {'A', 'V', '2', 'E',
                                                  'N', 'C', 'P', '0'};
  std::vector<std::uint8_t> payload;
  payload.reserve(28);
  payload.insert(payload.end(), kMagic.begin(), kMagic.end());
  append_le64(payload, frame_index);
  append_le32(payload, frame.width());
  append_le32(payload, frame.height());
  append_le32(payload, static_cast<std::uint32_t>(qp));
  return payload;
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

  av2enc::Y4mFrame frame;
  std::uint64_t frame_count = 0;
  while (options.max_frames == 0 || frame_count < options.max_frames) {
    const av2enc::Y4mReadResult result = reader.read_frame(frame, error);
    if (result == av2enc::Y4mReadResult::kEof) {
      break;
    }
    if (result == av2enc::Y4mReadResult::kError) {
      return fail_encode(error);
    }

    const std::vector<std::uint8_t> payload =
        make_placeholder_payload(frame_count, frame, options.qp);
    if (!writer.write_frame(payload, frame_count, error)) {
      return fail_encode(error);
    }
    ++frame_count;
  }

  if (!writer.close(error)) {
    std::cerr << "av2enc: " << error << "\n";
    return 1;
  }

  std::cout << "av2enc: placeholder encode complete (no AV2 syntax emitted)\n"
            << "  input:  " << options.input.string() << " (" << y4m.width << "x" << y4m.height
            << ", " << y4m.colorspace << ", " << y4m.fps_num << "/" << y4m.fps_den << " fps)\n"
            << "  output: " << options.output.string() << " (" << writer.bytes_written()
            << " bytes IVF)\n"
            << "  frames: " << frame_count << ", qp " << options.qp
            << " (recorded, not applied)\n";
  return 0;
}

#include "bitstream/m0_headers.h"
#include "encoder/m0_still.h"
#include "encoder/m1/m1_still.h"
#include "io/ivf.h"
#include "io/y4m.h"

#if AV2ENC_HAVE_CUDA
#include "cuda/arena.h"
#include "cuda/context.h"
#endif

#include <charconv>
#include <algorithm>
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
  bool m1 = false;
  bool m1_cfl = true;
  bool m1_deblock = true;
  bool m1_single = false;
  bool m1_dc = false;
  int m1_depth = -1;  // -1 = component default
  bool skip_vram_check = false;
  bool help = false;
};

enum class ParseStatus {
  kOk,
  kHelp,
  kError,
};

void print_usage(std::ostream& out) {
  out << "Usage: av2enc <input.y4m> -o <output.ivf> [--frames N] [--qp Q]\n"
         "              [--adaptive-cdf] [--m1] [--no-cfl] [--no-deblock]\n"
         "\n"
         "M0 still-picture encoder (default): 8-bit 4:2:0, one key frame,\n"
         "one tile, 64x64 superblocks, 32x32 DCT, DC intra only, no filters.\n"
         "M1 (--m1): partition tree, full intra modes, variable TX sizes,\n"
         "CfL and deblock. Both write one AV2 frame (TD + SEQ + CLK) into an\n"
         "IVF with fourcc AV02.\n"
         "\n"
         "Options:\n"
         "  -o, --output <file>  Output IVF path (required)\n"
         "  --frames <N>         Encodes one frame; values > 1 are clamped to 1\n"
         "  --qp <Q>             base_q_idx (default: 90; both paths pin Q = 90)\n"
         "  --adaptive-cdf       Enable normative §8.2.6 CDF adaptation\n"
         "  --m1                 Enable the integrated M1 still-picture path\n"
         "  --no-cfl             M1: disable chroma-from-luma\n"
         "  --no-deblock         M1: disable the deblocking filter\n"
         "  --skip-vram-check    EXPERIMENT ONLY: remove the 7,500 MB arena cap\n"
         "                       and size the CUDA arena from the device\n"
         "                       (do not use for target/CI verification)\n"
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

#if AV2ENC_HAVE_CUDA
// Explicit-experiment path: remove the 7,500 MB process cap and size the
// persistent arena from the selected device. The target/CI path never calls
// this; it keeps Arena::kDefaultCapacityBytes and the 7,500 MB cap.
bool configure_arena_from_device(std::string& error) {
  using av2enc::cuda::Arena;
  Arena::disable_cap();  // must happen before any Arena::init()

  av2enc::cuda::DeviceInfo info;
  const av2enc::cuda::Status status = av2enc::cuda::query_device(0, &info);
  if (status != av2enc::cuda::Status::kOk) {
    Arena::set_cap_bytes(0);  // restore the default 7,500 MB cap
    error = std::string("--skip-vram-check: query_device(0) failed: ") +
            av2enc::cuda::status_name(status);
    return false;
  }

  constexpr std::uint64_t kHeadroomBytes = 512ull * Arena::kMiB;
  const auto minus_headroom = [](std::uint64_t bytes) -> std::uint64_t {
    return bytes > kHeadroomBytes ? bytes - kHeadroomBytes : 0;
  };
  std::uint64_t capacity =
      std::min(minus_headroom(info.total_vram_bytes),
               minus_headroom(info.free_vram_bytes));
  if (capacity < Arena::kMiB) {
    capacity = Arena::kMiB;
  }
  // Store the chosen value as the process cap so later Arena::init() calls
  // cannot exceed it. (The 7,500 MB hard cap is intentionally gone here.)
  Arena::set_cap_bytes(capacity);

  std::cout << "av2enc: WARNING: --skip-vram-check is an explicit experiment\n"
            << "  device:   " << info.name << " (" << info.cc_major << "."
            << info.cc_minor << ", " << info.total_vram_bytes / Arena::kMiB
            << " MiB total, " << info.free_vram_bytes / Arena::kMiB
            << " MiB free)\n"
            << "  capacity: " << capacity / Arena::kMiB
            << " MiB (min(total, free) - 512 MiB; default is "
            << Arena::kDefaultCapacityBytes / Arena::kMiB << " MiB)\n"
            << "  target/CI verification must not use this flag\n";
  return true;
}
#endif

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

    if (argument == "--m1") {
      options.m1 = true;
      continue;
    }

    if (argument == "--no-cfl") {
      options.m1_cfl = false;
      continue;
    }

    if (argument == "--no-deblock") {
      options.m1_deblock = false;
      continue;
    }

    if (argument == "--m1-single") {
      options.m1_single = true;
      continue;
    }

    if (argument == "--m1-dc") {
      options.m1_dc = true;
      continue;
    }

    if (argument == "--m1-depth") {
      if (i + 1 >= argc) {
        error = "missing value after '--m1-depth'";
        return ParseStatus::kError;
      }
      int depth = 0;
      if (!parse_int(argv[++i], depth) || depth < 0 || depth > 8) {
        error = "--m1-depth must be an integer in [0, 8]";
        return ParseStatus::kError;
      }
      options.m1_depth = depth;
      continue;
    }

    if (argument == "--skip-vram-check") {
      options.skip_vram_check = true;
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

  if (options.skip_vram_check) {
#if AV2ENC_HAVE_CUDA
    if (!configure_arena_from_device(error)) {
      std::cerr << "av2enc: " << error << "\n";
      return 1;
    }
#else
    std::cerr << "av2enc: note: --skip-vram-check has no effect in this "
                 "CPU-only build\n";
#endif
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

  std::uint32_t tile_bytes = 0;
  std::uint32_t frame_bytes = 0;
  if (options.m1) {
    av2enc::m1::StillPicture picture;
    picture.width = frame.width();
    picture.height = frame.height();
    picture.y = frame.y();
    picture.u = frame.u();
    picture.v = frame.v();

    av2enc::m1::M1Options m1_options;
    m1_options.base_q_idx = static_cast<std::uint8_t>(options.qp);
    m1_options.enable_cfl = options.m1_cfl;
    m1_options.enable_deblock = options.m1_deblock;
    m1_options.adaptive_cdf = options.adaptive_cdf;
    m1_options.force_single_block = options.m1_single;
    m1_options.force_dc = options.m1_dc;
    if (options.m1_depth >= 0) {
      m1_options.max_partition_depth = options.m1_depth;
    }

    av2enc::m1::M1Result encoded;
    if (!av2enc::m1::encode_m1_still_picture(picture, m1_options, encoded,
                                             error)) {
      return fail_encode(error);
    }
    if (!writer.write_frame(encoded.frame, 0, error)) {
      return fail_encode(error);
    }
    tile_bytes = encoded.tile_bytes;
    frame_bytes = encoded.frame_bytes;
  } else {
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
    tile_bytes = encoded.tile_bytes;
    frame_bytes = encoded.frame_bytes;
  }

  if (!writer.close(error)) {
    std::cerr << "av2enc: " << error << "\n";
    return 1;
  }

  std::cout << "av2enc: " << (options.m1 ? "M1" : "M0")
            << " still-picture encode complete\n"
            << "  input:   " << options.input.string() << " (" << y4m.width
            << "x" << y4m.height << ", " << y4m.colorspace << ", "
            << y4m.fps_num << "/" << y4m.fps_den << " fps)\n"
            << "  output:  " << options.output.string() << " ("
            << writer.bytes_written() << " bytes IVF)\n"
            << "  frames:  1, base_q_idx " << options.qp << ", tile "
            << tile_bytes << " bytes, temporal unit " << frame_bytes
            << " bytes\n";
  return 0;
}

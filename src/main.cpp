#include "bitstream/m0_headers.h"
#include "encoder/m0_still.h"
#include "encoder/m1/m1_still.h"
#include "io/ivf.h"
#include "io/y4m.h"

#include <charconv>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#if AV2ENC_HAVE_CUDA
#include "cuda/arena.h"
#include "cuda/context.h"

// M1 GPU pipeline (owner: orchestrator, src/cuda/m1/).  Include the contract
// header when it is visible.  The call itself is only linked when CMake also
// found the orchestrator entry point (src/cuda/m1/m1_gpu.{cu,cpp}); until then
// --m1-gpu reports itself as unavailable at runtime.
#if __has_include("cuda/m1/m1_gpu.h")
#include "cuda/m1/m1_gpu.h"
#define AV2ENC_M1_GPU_HEADER 1
#else
#define AV2ENC_M1_GPU_HEADER 0
// TODO(agent 1/agent 9): replace this mirror with the real header once
// src/cuda/m1/m1_gpu.h lands in the tree.  Field names follow
// docs/m1-gpu.md (GpuOptions) and the orchestrator header.
namespace av2enc::cuda::m1gpu {

struct GpuOptions {
  std::uint8_t base_q_idx = 90;
  bool enable_cfl = true;
  bool enable_deblock = true;
  bool enable_edge_filter = true;
  bool luma_dc_only = true;
  int min_block_size = 8;
  int max_partition_depth = 4;
  bool adaptive_cdf = false;
  bool force_single_block = false;
  bool force_dc = false;
  int device = 0;
  std::uint32_t vram_cap_mb = 0;
  bool allow_cpu_fallback = true;
  bool verify_against_cpu = true;
  bool verbose = false;
};

struct GpuResult {
  std::uint32_t tile_bytes = 0;
  std::uint32_t frame_bytes = 0;
  std::vector<std::uint8_t> frame;
  std::vector<std::uint8_t> recon_y;
  std::vector<std::uint8_t> recon_u;
  std::vector<std::uint8_t> recon_v;
  bool frame_bytes_from_cpu = true;
};

bool encode_m1_gpu(const av2enc::m1::StillPicture& picture,
                   const GpuOptions& options, GpuResult& result,
                   std::string& error);

}  // namespace av2enc::cuda::m1gpu
#endif

#if AV2ENC_M1_GPU_HEADER && defined(AV2ENC_HAVE_M1_GPU) && AV2ENC_HAVE_M1_GPU
#define AV2ENC_M1_GPU_CALLABLE 1
#else
#define AV2ENC_M1_GPU_CALLABLE 0
#endif
#else
#define AV2ENC_M1_GPU_CALLABLE 0
#endif

namespace {

struct Options {
  std::filesystem::path input;
  std::filesystem::path output;
  std::uint64_t max_frames = 0;  // 0 = all frames
  int qp = 90;
  bool adaptive_cdf = false;
  bool m1 = false;
  bool m1_gpu = false;
  bool m1_cfl = true;
  bool m1_deblock = true;
  bool m1_single = false;
  bool m1_dc = false;
  bool m1_full_intra = false;
  bool m1_uv_full = false;
  bool m1_rdo_intra = false;
  double m1_rdo_scale = 0.125;
  bool m1_rdoq = false;
  double m1_rdoq_scale = 0.125;
  bool m1_tx_type = false;
  double m1_tx_type_scale = 0.125;
  bool m1_cdef = false;
  int m1_cdef_strengths = 4;
  bool m1_ccso = false;
  int m1_ccso_bands = 6;
  int m1_depth = -1;  // -1 = component default
  std::uint64_t vram_cap_mib = 0;  // 0 = largest reservation the device can serve
  bool help = false;
};

enum class ParseStatus {
  kOk,
  kHelp,
  kError,
};

void print_usage(std::ostream& out) {
  out << "Usage: av2enc <input.y4m> -o <output.ivf> [--frames N] [--qp Q]\n"
         "              [--adaptive-cdf] [--m1] [--m1-gpu] [--no-cfl]\n"
         "              [--no-deblock] [--m1-full-intra] [--m1-depth D]\n"
         "\n"
         "M0 still-picture encoder (default): 8-bit 4:2:0, one key frame,\n"
         "one tile, 64x64 superblocks, 32x32 DCT, DC intra only, no filters.\n"
         "M1 (--m1): partition tree, full intra modes, variable TX sizes,\n"
         "CfL and deblock. M1 GPU (--m1-gpu) uses the same M1 syntax with the\n"
         "block pipeline on the GPU; it requires a CUDA build and implies --m1.\n"
         "All paths write one AV2 frame (TD + SEQ + CLK) into an IVF with\n"
         "fourcc AV02.\n"
         "\n"
         "Options:\n"
         "  -o, --output <file>  Output IVF path (required)\n"
         "  --frames <N>         Encodes one frame; values > 1 are clamped to 1\n"
         "  --qp <Q>             base_q_idx (default: 90; both paths pin Q = 90)\n"
         "  --adaptive-cdf       Enable normative §8.2.6 CDF adaptation\n"
         "  --m1                 Enable the integrated CPU M1 still-picture path\n"
         "  --m1-gpu             Enable the CUDA M1 path (default off; implies\n"
         "                       --m1; CPU-only builds reject the flag)\n"
         "  --no-cfl             M1/M1 GPU: disable chroma-from-luma\n"
         "  --no-deblock         M1/M1 GPU: disable the deblocking filter\n"
         "  --m1-full-intra      Enable the directional/smooth intra search\n"
         "                       (SSE-only mode decision; DC is the default)\n"
         "  --m1-uv-full         M1: full chroma intra search (DC/CfL default)\n"
         "  --m1-rdo-intra       M1: rate-aware luma intra mode decision\n"
         "  --m1-rdo-scale S     lambda sse_scale for --m1-rdo-intra (0.125)\n"
         "  --m1-rdoq            M1: full-trellis coefficient RDOQ\n"
         "  --m1-rdoq-scale S    lambda sse_scale for --m1-rdoq (0.125)\n"
         "  --m1-tx-type         M1: intra luma TX_TYPE search\n"
         "  --m1-tx-type-scale S lambda sse_scale for --m1-tx-type (0.125)\n"
         "  --m1-cdef            M1: enable CDEF (post-deblock, opt-in)\n"
         "  --m1-cdef-strengths N  CDEF strength sets 1..8 (default 4)\n"
         "  --m1-ccso            M1: enable band-offset-only CCSO (post-CDEF)\n"
         "  --m1-ccso-bands N    CCSO max_band_log2 0..6 (default 6 = 64 bands)\n"
         "  --m1-depth <D>       M1/M1 GPU: partition depth cap [0, 8]\n"
         "  --vram-cap <MiB>     Upper bound for the CUDA arena; default 0 takes\n"
         "                       the largest reservation the device can serve\n"
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

#if AV2ENC_HAVE_CUDA
// Nothing to query here: the pipeline sizes the arena from the device at first
// use. This only records the optional upper bound.
bool apply_vram_cap(std::uint64_t cap_mib, std::string& error) {
  if (cap_mib > std::numeric_limits<std::uint64_t>::max() /
                   av2enc::cuda::Arena::kMiB) {
    error = "--vram-cap is too large";
    return false;
  }
  av2enc::cuda::Arena::set_capacity_cap_bytes(
      cap_mib * av2enc::cuda::Arena::kMiB);
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

    if (argument == "--m1-gpu") {
      options.m1_gpu = true;
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

    if (argument == "--m1-full-intra") {
      options.m1_full_intra = true;
      continue;
    }

    if (argument == "--m1-uv-full") {
      options.m1_uv_full = true;
      continue;
    }

    if (argument == "--m1-rdo-intra") {
      options.m1_rdo_intra = true;
      continue;
    }

    if (argument == "--m1-rdo-scale") {
      if (i + 1 >= argc ||
          !parse_double(argv[++i], options.m1_rdo_scale) ||
          options.m1_rdo_scale <= 0.0) {
        error = "--m1-rdo-scale requires a positive number";
        return ParseStatus::kError;
      }
      continue;
    }

    if (argument == "--m1-rdoq") {
      options.m1_rdoq = true;
      continue;
    }

    if (argument == "--m1-rdoq-scale") {
      if (i + 1 >= argc ||
          !parse_double(argv[++i], options.m1_rdoq_scale) ||
          options.m1_rdoq_scale <= 0.0) {
        error = "--m1-rdoq-scale requires a positive number";
        return ParseStatus::kError;
      }
      continue;
    }

    if (argument == "--m1-tx-type") {
      options.m1_tx_type = true;
      continue;
    }

    if (argument == "--m1-tx-type-scale") {
      if (i + 1 >= argc ||
          !parse_double(argv[++i], options.m1_tx_type_scale) ||
          options.m1_tx_type_scale <= 0.0) {
        error = "--m1-tx-type-scale requires a positive number";
        return ParseStatus::kError;
      }
      continue;
    }

    if (argument == "--m1-cdef") {
      options.m1_cdef = true;
      continue;
    }

    if (argument == "--m1-ccso") {
      options.m1_ccso = true;
      continue;
    }

    if (argument == "--m1-ccso-bands") {
      if (i + 1 >= argc) {
        error = "missing value after '--m1-ccso-bands'";
        return ParseStatus::kError;
      }
      int bands = 0;
      if (!parse_int(argv[++i], bands) || bands < 0 || bands > 6) {
        error = "--m1-ccso-bands must be an integer in [0, 6]";
        return ParseStatus::kError;
      }
      options.m1_ccso_bands = bands;
      continue;
    }

    if (argument == "--m1-cdef-strengths") {
      if (i + 1 >= argc) {
        error = "missing value after '--m1-cdef-strengths'";
        return ParseStatus::kError;
      }
      int strengths = 0;
      if (!parse_int(argv[++i], strengths) || strengths < 1 ||
          strengths > 8) {
        error = "--m1-cdef-strengths must be an integer in [1, 8]";
        return ParseStatus::kError;
      }
      options.m1_cdef_strengths = strengths;
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

    if (argument == "--vram-cap") {
      if (i + 1 >= argc ||
          !parse_uint64(argv[++i], options.vram_cap_mib) ||
          options.vram_cap_mib == 0) {
        error = "--vram-cap requires a positive MiB value";
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

  if (options.vram_cap_mib != 0) {
#if AV2ENC_HAVE_CUDA
    if (!apply_vram_cap(options.vram_cap_mib, error)) {
      std::cerr << "av2enc: " << error << "\n";
      return 2;
    }
#else
    std::cerr << "av2enc: note: --vram-cap has no effect in this "
                 "CPU-only build\n";
#endif
  }

#if !AV2ENC_M1_GPU_CALLABLE
  // Accept the flag in every build but fail before touching the input/output
  // so the caller gets a precise reason instead of a link-time or decode-time
  // surprise.
  if (options.m1_gpu) {
#if AV2ENC_HAVE_CUDA
    std::cerr << "av2enc: error: --m1-gpu is not available in this build: "
                 "the M1 GPU entry point (src/cuda/m1/m1_gpu.cu) has not "
                 "landed yet. Use --m1 for the CPU M1 path.\n";
#else
    std::cerr << "av2enc: error: --m1-gpu requires a CUDA build; this is a "
                 "CPU-only reference build. Use --m1 for the CPU M1 path.\n";
#endif
    return 2;
  }
#endif

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

  // M0's edge recursion cannot code blocks below 32x32 in one dimension (it
  // would need 16x16 / 16x8 / 8x16 / 8x32 / 8x8 coders). Rather than emit a
  // desynchronised stream, use the M1 still path for those geometries and say
  // so on stderr. Explicit --m1 / --m1-gpu are unaffected.
  bool m0_fallback_to_m1 = false;
  if (!options.m1 && !options.m1_gpu) {
    std::string reason;
    if (!av2enc::m0::geometry_supported(frame.width(), frame.height(),
                                        reason)) {
      std::cerr << "av2enc: note: M0 cannot represent " << frame.width() << "x"
                << frame.height() << " (" << reason
                << "); falling back to the M1 still-picture path\n";
      m0_fallback_to_m1 = true;
    }
  }

  if (options.m1 || options.m1_gpu || m0_fallback_to_m1) {
    av2enc::m1::StillPicture picture;
    picture.width = frame.width();
    picture.height = frame.height();
    picture.y = frame.y();
    picture.u = frame.u();
    picture.v = frame.v();

    if (options.m1_gpu) {
#if AV2ENC_M1_GPU_CALLABLE
      // GpuOptions mirrors M1Options and adds orchestrator controls; see
      // src/cuda/m1/m1_gpu.h and docs/m1-gpu.md.
      av2enc::cuda::m1gpu::GpuOptions gpu_options;
      gpu_options.base_q_idx = static_cast<std::uint8_t>(options.qp);
      gpu_options.enable_cfl = options.m1_cfl;
      gpu_options.enable_deblock = options.m1_deblock;
      gpu_options.enable_edge_filter = true;
      gpu_options.luma_dc_only = !options.m1_full_intra;
      gpu_options.min_block_size = 8;
      if (options.m1_depth >= 0) {
        gpu_options.max_partition_depth = options.m1_depth;
      }
      gpu_options.adaptive_cdf = options.adaptive_cdf;
      gpu_options.force_single_block = options.m1_single;
      gpu_options.force_dc = options.m1_dc;
      gpu_options.uv_full = options.m1_uv_full;
      gpu_options.enable_cdef = options.m1_cdef;
      gpu_options.rdoq = options.m1_rdoq;
      gpu_options.tx_type = options.m1_tx_type;
      gpu_options.rdo_intra = options.m1_rdo_intra;
      gpu_options.vram_cap_mb = static_cast<std::uint32_t>(
          std::min<std::uint64_t>(options.vram_cap_mib,
                                  std::numeric_limits<std::uint32_t>::max()));

      av2enc::cuda::m1gpu::GpuResult encoded;
      if (!av2enc::cuda::m1gpu::encode_m1_gpu(picture, gpu_options, encoded,
                                              error)) {
        return fail_encode(error);
      }
      if (encoded.frame_bytes_from_cpu) {
        std::cerr << "av2enc: note: frame_bytes_from_cpu=1; frame bytes came "
                     "from the CPU M1 reference (linked stage hooks: "
                  << av2enc::cuda::m1gpu::linked_stage_count() << ")\n";
      }
      if (!writer.write_frame(encoded.frame, 0, error)) {
        return fail_encode(error);
      }
      tile_bytes = encoded.tile_bytes;
      frame_bytes = encoded.frame_bytes;
#else
      return fail_encode("--m1-gpu is not available in this build");
#endif
    } else {
      av2enc::m1::M1Options m1_options;
      m1_options.base_q_idx = static_cast<std::uint8_t>(options.qp);
      m1_options.enable_cfl = options.m1_cfl;
      m1_options.enable_deblock = options.m1_deblock;
      m1_options.adaptive_cdf = options.adaptive_cdf;
      m1_options.force_single_block = options.m1_single;
      m1_options.force_dc = options.m1_dc;
      m1_options.luma_dc_only = !options.m1_full_intra && !options.m1_rdo_intra;
      m1_options.uv_full = options.m1_uv_full;
      m1_options.rdo_intra = options.m1_rdo_intra;
      m1_options.rdo_sse_scale = options.m1_rdo_scale;
      m1_options.rdoq = options.m1_rdoq;
      m1_options.rdoq_sse_scale = options.m1_rdoq_scale;
      m1_options.tx_type = options.m1_tx_type;
      m1_options.tx_type_sse_scale = options.m1_tx_type_scale;
      m1_options.enable_cdef = options.m1_cdef;
      m1_options.cdef_num_strengths = options.m1_cdef_strengths;
      m1_options.enable_ccso = options.m1_ccso;
      m1_options.ccso_bands_log2 = options.m1_ccso_bands;
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
    }
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

  const char* path_name =
      options.m1_gpu
          ? "M1 GPU"
          : (options.m1 ? "M1"
                        : (m0_fallback_to_m1 ? "M1 (M0 fallback)" : "M0"));
  std::cout << "av2enc: " << path_name
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

#include "core/bitstream/basic_headers.h"
#include "codec/entropy/lossless/lossless.h"
#include "pipeline/still/basic_encoder.h"
#include "pipeline/still/still_encoder.h"
#include "core/io/ivf.h"
#include "core/io/y4m.h"

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

// Still GPU pipeline (owner: orchestrator, src/cuda/pipeline/still/).  Include
// the contract header when it is visible.  The call itself is only linked when
// CMake also found the orchestrator entry point; until then --still-gpu reports
// itself as unavailable at runtime.
#if __has_include("cuda/pipeline/still/still_gpu.h")
#include "cuda/pipeline/still/still_gpu.h"
#define AV2ENC_STILL_GPU_HEADER 1
#else
#define AV2ENC_STILL_GPU_HEADER 0
// TODO(agent 1/agent 9): replace this mirror with the real header once
// src/cuda/m1/still_gpu.h lands in the tree.  Field names follow
// docs/m1-gpu.md (GpuOptions) and the orchestrator header.
namespace av2enc::cuda::still_gpu {

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

bool encode_still_gpu(const av2enc::still::StillPicture& picture,
                   const GpuOptions& options, GpuResult& result,
                   std::string& error);

}  // namespace av2enc::cuda::still_gpu
#endif

#if AV2ENC_STILL_GPU_HEADER && defined(AV2ENC_HAVE_STILL_GPU) && AV2ENC_HAVE_STILL_GPU
#define AV2ENC_STILL_GPU_CALLABLE 1
#else
#define AV2ENC_STILL_GPU_CALLABLE 0
#endif
#else
#define AV2ENC_STILL_GPU_CALLABLE 0
#endif

namespace {

struct Options {
  std::filesystem::path input;
  std::filesystem::path output;
  std::uint64_t max_frames = 0;  // 0 = all frames
  int qp = 90;
  bool adaptive_cdf = false;
  bool m1 = false;
  bool still_gpu = false;
  bool still_cfl = true;
  bool still_deblock = true;
  bool still_single = false;
  bool still_dc = false;
  bool still_full_intra = false;
  bool still_uv_full = false;
  bool still_rdo_intra = false;
  double still_rdo_scale = 0.125;
  bool still_rdoq = false;
  double still_rdoq_scale = 0.125;
  bool still_tx_type = false;
  double still_tx_type_scale = 0.125;
  bool still_cdef = false;
  int still_cdef_strengths = 4;
  bool still_ccso = false;
  int still_ccso_bands = 6;
  bool still_sb128 = false;
  bool still_sb256 = false;
  bool still_ext_partitions = false;
  bool still_segmentation = false;
  bool still_delta_q = false;
  int still_delta_q_res = 0;
  bool lossless = false;
  bool lossless_force_fsc = false;
  bool still_content_interpretation = false;
  bool still_qm = false;
  int still_qm_level = 3;
  bool still_ist = false;
  double still_ist_scale = 0.125;
  bool still_dc_ac_delta_q = false;
  bool still_ibp = false;
  bool still_mrl = false;
  int still_cfl_ds = 0;
  bool still_cfl_explicit = false;
  bool still_dip = false;
  bool still_palette = false;
  bool still_mhccp = false;
  bool still_ccso_edge = false;
  bool still_wiener = false;
  bool still_chroma_lr = false;
  bool still_pc_wiener = false;
  bool still_gdf = false;
  bool still_grain = false;
  int still_depth = -1;  // -1 = component default
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
         "              [--adaptive-cdf] [--still] [--still-gpu] [--no-cfl]\n"
         "              [--no-deblock] [--still-full-intra]\n"
         "\n"
         "Basic still-picture encoder (default): 8-bit 4:2:0, one key frame,\n"
         "one tile, 64x64 superblocks, 32x32 DCT, DC intra only, no filters.\n"
         "Still (--still): partition tree, full intra modes, variable TX sizes,\n"
         "CfL and deblock. Still GPU (--still-gpu) uses the same still syntax with the\n"
         "block pipeline on the GPU; it requires a CUDA build and implies --still.\n"
         "All paths write one AV2 frame (TD + SEQ + CLK) into an IVF with\n"
         "fourcc AV02.\n"
         "\n"
         "Options:\n"
         "  -o, --output <file>  Output IVF path (required)\n"
         "  --frames <N>         Encodes one frame; values > 1 are clamped to 1\n"
         "  --qp <Q>             base_q_idx (default: 90; Still CPU accepts 1..255,\n"
         "                       Basic and --still-gpu pin 90)\n"
         "  --adaptive-cdf       Enable normative §8.2.6 CDF adaptation\n"
         "  --still                 Enable the integrated CPU still-picture path\n"
         "  --still-gpu             Enable the CUDA still path (default off; implies\n"
         "                       --still; CPU-only builds reject the flag)\n"
         "  --no-cfl             Still/Still GPU: disable chroma-from-luma\n"
         "  --no-deblock         Still/Still GPU: disable the deblocking filter\n"
         "  --still-full-intra      Enable the directional/smooth intra search\n"
         "                       (SSE-only mode decision; DC is the default)\n"
         "  --still-uv-full         M1: full chroma intra search (DC/CfL default)\n"
         "  --still-rdo-intra       M1: rate-aware luma intra mode decision\n"
         "  --still-rdoq            M1: full-trellis coefficient RDOQ\n"
         "  --still-tx-type         M1: intra luma TX_TYPE search\n"
         "  --still-cdef            M1: enable CDEF (post-deblock, opt-in)\n"
         "  --still-ccso            M1: enable band-offset-only CCSO (post-CDEF)\n"
         "  --still-sb128           Still: 128x128 superblocks (default 64x64)\n"
         "  --still-sb256           Still: signal 256x256 sequence superblocks\n"
         "                       (intra frames still code 128x128; default off)\n"
         "  --still-ext-partitions  Still: HORZ_3/VERT_3 and uneven 4-way\n"
         "                       extended partitions (CPU-only; default off)\n"
         "  --still-segmentation    M1: signal a neutral 8-segment table (opt-in)\n"
         "  --still-delta-q         M1: signal neutral per-superblock delta-Q (opt-in)\n"
         "  --still-content-interpretation  M1: emit a default content interpretation OBU\n"
         "  --lossless              Q=0 lossless still picture (DPCM intra, 8x8 blocks)\n"
         "  --lossless-fsc          Lossless: force forward-skip-coding coefficient path\n"
         "  --still-qm              M1: signal a spec-defined quantizer matrix\n"
         "                       (default matrices, level 3; opt-in)\n"
         "  --still-ist             M1: intra secondary transform (luma, opt-in)\n"
         "  --still-dc-ac-delta-q   M1: DC/AC delta-Q frame search (opt-in)\n"
         "  --still-ibp             M1: enable intra bi-prediction (§5.4.5)\n"
         "  --still-mrl             M1: enable multiple reference lines for\n"
         "                       directional luma (implies the directional search)\n"
         "  --still-cfl-ds <N>      M1: cfl_ds_filter_index 0..3 (default 0;\n"
         "                       3 behaves like 0)\n"
         "  --still-cfl-explicit    M1: allow explicit CfL alpha (CFL_EXPLICIT)\n"
         "  --still-dip             M1: enable DIP as a luma DC alternative\n"
         "  --still-palette         M1: enable luma palette intra (screen content;\n"
         "                       sets allow_screen_content_tools)\n"
         "  --still-mhccp           M1: enable MHCCP (CFL_MULTI) chroma prediction\n"
         "  --still-ccso-edge       M1: edge-classified CCSO + per-unit skip\n"
         "  --still-wiener          M1: luma Wiener NS loop restoration (post-CCSO)\n"
         "  --still-chroma-lr       M1: chroma Wiener NS loop restoration\n"
         "  --still-pc-wiener       M1: luma PC-Wiener loop restoration\n"
         "  --still-gdf             M1: guided detail filter (pre-Wiener)\n"
         "  --still-grain           M1: signal film grain (decoder synthesis)\n"
         "  --vram-cap <MiB>     Upper bound for the CUDA arena; default 0 takes\n"
         "                       the largest reservation the device can serve\n"
         "  -h, --help           Show this help\n"
         "\n"
         "Debug options (test-only; not encoding-quality options):\n"
         "  --still-single          Force one coding block per superblock\n"
         "  --still-dc              Force DC-only intra\n"
         "  --still-depth <D>       Still/Still GPU: partition depth cap [0, 8]\n"
         "  --still-rdo-scale S     lambda sse_scale for --still-rdo-intra (0.125)\n"
         "  --still-rdoq-scale S    lambda sse_scale for --still-rdoq (0.125)\n"
         "  --still-tx-type-scale S lambda sse_scale for --still-tx-type (0.125)\n"
         "  --still-cdef-strengths N  CDEF strength sets 1..8 (default 4)\n"
         "  --still-ccso-bands N    CCSO max_band_log2 0..6 (default 6 = 64 bands)\n"
         "  --still-delta-q-res N   delta_q_res for --still-delta-q, 0..3 (default 0)\n"
         "  --still-qm-level L     QM default matrix level 0..14 (default 3)\n"
         "  --still-ist-scale S    lambda sse_scale for --still-ist (0.125)\n";
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

    if (argument == "--still") {
      options.m1 = true;
      continue;
    }

    if (argument == "--still-gpu") {
      options.still_gpu = true;
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

    if (argument == "--still-segmentation") {
      options.still_segmentation = true;
      continue;
    }

    if (argument == "--still-delta-q") {
      options.still_delta_q = true;
      continue;
    }

    if (argument == "--still-qm") {
      options.still_qm = true;
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
  if (options.still_ext_partitions) {
    if (!options.m1 && !options.still_gpu) {
      error = "--still-ext-partitions requires --still";
      return ParseStatus::kError;
    }
    if (options.still_gpu) {
      error = "--still-ext-partitions is CPU-only: the CUDA planner has no "
              "extended partition tree";
      return ParseStatus::kError;
    }
    if (options.still_single) {
      error = "--still-ext-partitions cannot be combined with --still-single";
      return ParseStatus::kError;
    }
  }
  if (options.still_sb128 || options.still_sb256) {
    if (options.still_sb128 && options.still_sb256) {
      error = "--still-sb128 and --still-sb256 are mutually exclusive";
      return ParseStatus::kError;
    }
    if (!options.m1 && !options.still_gpu) {
      error = "--still-sb128/--still-sb256 require --still";
      return ParseStatus::kError;
    }
    if (options.still_gpu) {
      error =
          "--still-sb128/--still-sb256 are CPU-only: the CUDA planner accepts "
          "64x64 superblocks only";
      return ParseStatus::kError;
    }
    if (options.still_single) {
      error = "--still-single cannot be combined with a 128x128 superblock";
      return ParseStatus::kError;
    }
    if (options.still_depth == 0) {
      error = "--still-depth 0 cannot be combined with a 128x128 superblock "
              "(the 128 root must split)";
      return ParseStatus::kError;
    }
    if (options.still_cdef) {
      error = "--still-cdef is not wired for 128x128 superblocks";
      return ParseStatus::kError;
    }
    if (options.still_ccso) {
      error = "--still-ccso is not wired for 128x128 superblocks";
      return ParseStatus::kError;
    }
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

#if !AV2ENC_STILL_GPU_CALLABLE
  // Accept the flag in every build but fail before touching the input/output
  // so the caller gets a precise reason instead of a link-time or decode-time
  // surprise.
  if (options.still_gpu) {
#if AV2ENC_HAVE_CUDA
    std::cerr << "av2enc: error: --still-gpu is not available in this build: "
                 "the Still GPU entry point (src/cuda/m1/still_gpu.cu) has not "
                 "landed yet. Use --still for the CPU M1 path.\n";
#else
    std::cerr << "av2enc: error: --still-gpu requires a CUDA build; this is a "
                 "CPU-only reference build. Use --still for the CPU M1 path.\n";
#endif
    return 2;
  }
#endif

  if (options.still_gpu &&
      (options.still_segmentation || options.still_delta_q)) {
    std::cerr << "av2enc: error: --still-segmentation and --still-delta-q are "
                 "CPU-only (--still); the Still GPU path does not signal "
                 "delta-Q/segmentation\n";
    return 2;
  }

  if (options.lossless && options.still_gpu) {
    std::cerr << "av2enc: error: --lossless is implemented by the CPU "
                 "lossless module; it cannot be combined with --still-gpu\n";
    return 2;
  }

  if (options.lossless &&
      (options.still_content_interpretation || options.still_segmentation ||
       options.still_delta_q)) {
    std::cerr << "av2enc: error: --lossless uses its own header writer; "
                 "--still-content-interpretation and the delta-Q/segmentation "
                 "flags are not applied there\n";
    return 2;
  }

  if (options.still_gpu && options.still_content_interpretation) {
    std::cerr << "av2enc: error: --still-content-interpretation is CPU-only "
                 "(--still); the Still GPU path does not emit it\n";
    return 2;
  }

  if (options.still_gpu) {
    const auto reject_cpu_only = [](const char* flag) {
      std::cerr << "av2enc: error: --still-gpu: " << flag
                << " is not ported to the GPU pipeline yet; use --still (CPU)\n";
      return 2;
    };
    if (options.still_ibp) return reject_cpu_only("--still-ibp");
    if (options.still_mrl) return reject_cpu_only("--still-mrl");
    if (options.still_dip) return reject_cpu_only("--still-dip");
    if (options.still_mhccp) return reject_cpu_only("--still-mhccp");
    if (options.still_cfl_explicit) {
      return reject_cpu_only("--still-cfl-explicit");
    }
    if (options.still_cfl_ds != 0) return reject_cpu_only("--still-cfl-ds");
    if (options.still_palette) return reject_cpu_only("--still-palette");
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
  // The CPU M1 path selects the §8.3.1 coefficient-CDF qctx slice from the raw
  // frame base_q_idx and derives the quantiser from qlookup(base_q_idx), so QP
  // 1..255 is supported there.  M0 and the Still GPU path still pin 90.
  const bool still_cpu_qp = options.m1 && !options.still_gpu;
  if (options.qp == 0) {
    if (!options.lossless) {
      std::cerr << "av2enc: error: --qp 0 requires --lossless\n";
      return 1;
    }
  }
  if (!options.lossless && !still_cpu_qp && options.qp != 90) {
    std::cerr << "av2enc: error: "
              << (options.still_gpu ? "Still GPU" : "M0")
              << " pins --qp 90; use --still for QP 1..255\n";
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
  // so on stderr. Explicit --still / --still-gpu are unaffected.
  bool basic_fallback_to_m1 = false;
  if (!options.m1 && !options.still_gpu && !options.lossless) {
    std::string reason;
    if (!av2enc::basic::geometry_supported(frame.width(), frame.height(),
                                        reason)) {
      std::cerr << "av2enc: note: M0 cannot represent " << frame.width() << "x"
                << frame.height() << " (" << reason
                << "); falling back to the M1 still-picture path\n";
      basic_fallback_to_m1 = true;
    }
  }

  if (options.lossless) {
    av2enc::lossless::StillPicture picture;
    picture.width = frame.width();
    picture.height = frame.height();
    picture.y = frame.y();
    picture.u = frame.u();
    picture.v = frame.v();

    av2enc::lossless::Options lossless_options;
    lossless_options.adaptive_cdf = options.adaptive_cdf;
    lossless_options.fsc = options.lossless_force_fsc
                               ? av2enc::lossless::FscMode::kForce
                               : av2enc::lossless::FscMode::kOff;

    av2enc::lossless::EncodeResult encoded;
    if (!av2enc::lossless::encode_still(picture, lossless_options, encoded,
                                        error)) {
      return fail_encode(error);
    }
    if (!encoded.stats.exact) {
      return fail_encode("lossless reconstruction is not exact (max |err| " +
                         std::to_string(encoded.stats.max_abs_recon_error) +
                         ")");
    }
    if (!writer.write_frame(encoded.temporal_unit, 0, error)) {
      return fail_encode(error);
    }
    tile_bytes = encoded.stats.tile_bytes;
    frame_bytes = encoded.stats.frame_bytes;
  } else if (options.m1 || options.still_gpu || basic_fallback_to_m1) {
    av2enc::still::StillPicture picture;
    picture.width = frame.width();
    picture.height = frame.height();
    picture.y = frame.y();
    picture.u = frame.u();
    picture.v = frame.v();

    if (options.still_gpu) {
      if (options.still_qm) {
        return fail_encode(
            "--still-gpu: --still-qm is not ported to the GPU pipeline yet; "
            "use --still for the CPU QM path");
      }
      if (options.still_ist) {
        return fail_encode(
            "--still-gpu: --still-ist is not ported to the GPU pipeline yet; "
            "use --still for the CPU IST path");
      }
      if (options.still_dc_ac_delta_q) {
        return fail_encode(
            "--still-gpu: --still-dc-ac-delta-q is not ported to the GPU pipeline "
            "yet; use --still for the CPU delta-Q path");
      }
#if AV2ENC_STILL_GPU_CALLABLE
      // GpuOptions mirrors StillOptions and adds orchestrator controls; see
      // src/cuda/m1/still_gpu.h and docs/m1-gpu.md.
      av2enc::cuda::still_gpu::GpuOptions gpu_options;
      gpu_options.base_q_idx = static_cast<std::uint8_t>(options.qp);
      gpu_options.enable_cfl = options.still_cfl;
      gpu_options.enable_deblock = options.still_deblock;
      gpu_options.enable_edge_filter = true;
      gpu_options.luma_dc_only = !options.still_full_intra;
      gpu_options.min_block_size = 8;
      if (options.still_depth >= 0) {
        gpu_options.max_partition_depth = options.still_depth;
      }
      gpu_options.adaptive_cdf = options.adaptive_cdf;
      gpu_options.force_single_block = options.still_single;
      gpu_options.force_dc = options.still_dc;
      gpu_options.uv_full = options.still_uv_full;
      gpu_options.enable_cdef = options.still_cdef;
      gpu_options.cdef_num_strengths = options.still_cdef_strengths;
      gpu_options.enable_ccso = options.still_ccso;
      gpu_options.ccso_bands_log2 = options.still_ccso_bands;
      gpu_options.ccso_edge = options.still_ccso_edge;
      gpu_options.enable_wiener = options.still_wiener;
      gpu_options.enable_chroma_lr = options.still_chroma_lr;
      gpu_options.enable_pc_wiener = options.still_pc_wiener;
      gpu_options.enable_gdf = options.still_gdf;
      gpu_options.enable_grain = options.still_grain;
      gpu_options.rdoq = options.still_rdoq;
      gpu_options.rdoq_sse_scale = options.still_rdoq_scale;
      gpu_options.tx_type = options.still_tx_type;
      gpu_options.tx_type_sse_scale = options.still_tx_type_scale;
      gpu_options.rdo_intra = options.still_rdo_intra;
      gpu_options.rdo_sse_scale = options.still_rdo_scale;
      gpu_options.vram_cap_mb = static_cast<std::uint32_t>(
          std::min<std::uint64_t>(options.vram_cap_mib,
                                  std::numeric_limits<std::uint32_t>::max()));

      av2enc::cuda::still_gpu::GpuResult encoded;
      if (!av2enc::cuda::still_gpu::encode_still_gpu(picture, gpu_options, encoded,
                                              error)) {
        return fail_encode(error);
      }
      if (encoded.frame_bytes_from_cpu) {
        std::cerr << "av2enc: note: frame_bytes_from_cpu=1; frame bytes came "
                     "from the CPU M1 reference (linked stage hooks: "
                  << av2enc::cuda::still_gpu::linked_stage_count() << ")\n";
      }
      if (!writer.write_frame(encoded.frame, 0, error)) {
        return fail_encode(error);
      }
      tile_bytes = encoded.tile_bytes;
      frame_bytes = encoded.frame_bytes;
#else
      return fail_encode("--still-gpu is not available in this build");
#endif
    } else {
      av2enc::still::StillOptions still_options;
      still_options.base_q_idx = static_cast<std::uint8_t>(options.qp);
      still_options.enable_cfl = options.still_cfl;
      still_options.enable_deblock = options.still_deblock;
      still_options.adaptive_cdf = options.adaptive_cdf;
      still_options.force_single_block = options.still_single;
      still_options.force_dc = options.still_dc;
      still_options.luma_dc_only =
          !options.still_full_intra && !options.still_rdo_intra &&
          !options.still_mrl;
      still_options.uv_full = options.still_uv_full;
      still_options.enable_mrl = options.still_mrl;
      still_options.cfl_ds_filter = options.still_cfl_ds;
      still_options.cfl_explicit = options.still_cfl_explicit;
      still_options.enable_dip = options.still_dip;
      still_options.palette = options.still_palette;
      still_options.mhccp = options.still_mhccp;
      still_options.rdo_intra = options.still_rdo_intra;
      still_options.rdo_sse_scale = options.still_rdo_scale;
      still_options.rdoq = options.still_rdoq;
      still_options.rdoq_sse_scale = options.still_rdoq_scale;
      still_options.tx_type = options.still_tx_type;
      still_options.tx_type_sse_scale = options.still_tx_type_scale;
      still_options.enable_cdef = options.still_cdef;
      still_options.cdef_num_strengths = options.still_cdef_strengths;
      still_options.enable_ccso = options.still_ccso;
      still_options.ccso_bands_log2 = options.still_ccso_bands;
      still_options.sb128 = options.still_sb128;
      still_options.sb256 = options.still_sb256;
      still_options.ext_partitions = options.still_ext_partitions;
      still_options.enable_segmentation = options.still_segmentation;
      still_options.enable_delta_q = options.still_delta_q;
      still_options.delta_q_res = options.still_delta_q_res;
      still_options.content_interpretation =
          options.still_content_interpretation;
      still_options.qm = options.still_qm;
      still_options.qm_level = options.still_qm_level;
      still_options.ist = options.still_ist;
      still_options.ist_sse_scale = options.still_ist_scale;
      still_options.delta_q = options.still_dc_ac_delta_q;
      still_options.enable_ibp = options.still_ibp;
      still_options.ccso_edge = options.still_ccso_edge;
      still_options.enable_wiener = options.still_wiener;
      still_options.enable_chroma_lr = options.still_chroma_lr;
      still_options.enable_pc_wiener = options.still_pc_wiener;
      still_options.enable_gdf = options.still_gdf;
      still_options.enable_grain = options.still_grain;
      if (options.still_depth >= 0) {
        still_options.max_partition_depth = options.still_depth;
      }

      av2enc::still::StillResult encoded;
      if (!av2enc::still::encode_still_picture(picture, still_options, encoded,
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
    av2enc::basic::StillPicture picture;
    picture.width = frame.width();
    picture.height = frame.height();
    picture.y = frame.y();
    picture.u = frame.u();
    picture.v = frame.v();

    av2enc::basic::StillPictureConfig config;
    config.max_frame_width = static_cast<std::uint16_t>(y4m.width);
    config.max_frame_height = static_cast<std::uint16_t>(y4m.height);
    config.base_q_idx = static_cast<std::uint8_t>(options.qp);
    // Frozen CDFs are the default M0 bitstream. Adaptive coding is an opt-in
    // normative §8.2.6 mode used to test the CDF-update path.
    config.enable_cdf_adaptation = options.adaptive_cdf;

    av2enc::basic::StillEncodeResult encoded;
    if (!av2enc::basic::encode_still_picture(picture, config, encoded, error)) {
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
      options.lossless
          ? "Lossless"
          : (options.still_gpu
                 ? "Still GPU"
                 : (options.m1 ? "M1"
                               : (basic_fallback_to_m1 ? "M1 (M0 fallback)"
                                                       : "M0")));
  std::cout << "av2enc: " << path_name
            << " still-picture encode complete\n"
            << "  input:   " << options.input.string() << " (" << y4m.width
            << "x" << y4m.height << ", " << y4m.colorspace << ", "
            << y4m.fps_num << "/" << y4m.fps_den << " fps)\n"
            << "  output:  " << options.output.string() << " ("
            << writer.bytes_written() << " bytes IVF)\n"
            << "  frames:  1, base_q_idx "
            << (options.lossless ? 0 : options.qp) << ", tile "
            << tile_bytes << " bytes, temporal unit " << frame_bytes
            << " bytes\n";
  return 0;
}

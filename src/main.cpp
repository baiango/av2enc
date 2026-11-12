#include "core/bitstream/basic_headers.h"
#include "codec/entropy/lossless/lossless.h"
#include "formats/format_encode.h"
#include "formats/format_y4m.h"
#include "pipeline/still/basic_encoder.h"
#include "pipeline/still/still_encoder.h"
#include "pipeline/tiles/stream/tiles_stream.h"
#include "pipeline/video/post/rc_adapter.h"
#include "pipeline/dispatch/planner.h"
#include "core/io/ivf.h"
#include "core/io/y4m.h"
#include "pipeline/video/video_frame.h"

#include <charconv>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
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
  // Multi-frame inter (video) path.  Opt-in; every default still/basic stream
  // is untouched.
  bool video = false;
  std::uint64_t p_frames = 0;  // 0 = every frame after the keyframe
  std::uint64_t b_frames = 0;  // reserved; non-zero is rejected until wired
  int video_refs = 1;          // NumTotalRefs (1..3)
  bool video_inter_residual = false;
  bool video_subpel_eighth = false;
  int video_mv_precision = 0;  // 0 = 1/8, 1 = 1/4, 2 = 1/2 pel
  bool video_drl_reorder = false;
  bool video_drl_reorder_always = false;
  bool video_long_term = false;
  bool video_olk = false;
  int video_leading = 0;  // 0 = off
  bool video_sef = false;
  bool video_switch = false;
  bool video_ras = false;
  bool video_force_single = false;
  int video_min_block = 16;
  int video_max_depth = 4;
  int video_me_range = 16;
  bool video_trace = false;
  std::filesystem::path video_recon;
  // Per-frame rate control for --video. Default "cqp" keeps the fixed --qp
  // stream byte-identical; crf/vbr plan a base_q_idx per frame.
  std::string rc_mode = "cqp";
  bool rc_given = false;
  double rc_crf = 32.0;
  bool rc_crf_given = false;
  double rc_bitrate_kbps = 0.0;
  bool rc_bitrate_given = false;
  std::uint64_t vram_cap_mib = 0;  // 0 = largest reservation the device can serve
  int bit_depth = 8;               // 8 (default) or 10; 12-bit is not signalable
  std::string chroma = "420";      // 420 (default), 422, 444, 400
  std::filesystem::path recon;     // optional encoder-reconstruction dump
  // Multi-tile stills. "1x1" (default) keeps the single-tile writer; every
  // other value selects the tiled writer and requires --no-deblock.
  std::string tiles = "1x1";
  // Host-only multi-GPU tile-plan dump. 0 = off (the default); never allocates
  // device memory and never writes a stream.
  int gpu_plan = 0;
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
         "  --bit-depth <8|10>    Sample depth (default 8). 10 requires --still,\n"
         "                       a C420p10 input and base_q_idx 1..90; 12-bit is\n"
         "                       not signalable in AV2 v1.0.0\n"
         "  --chroma <fmt>        Chroma format: 420 (default), 422, 444, 400.\n"
         "                       422/444/400 require --still, an 8-bit input of\n"
         "                       that exact format and --qp 90\n"
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
         "  --tiles <N|CxR>      Still: uniform tile grid, default 1x1. Accepts a\n"
         "                       tile count (2/4/8/16/32) or an explicit grid\n"
         "                       (2x1/2x2/4x2/8x2/16x2). Requires --still and\n"
         "                       --no-deblock; 8-bit 4:2:0 only\n"
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
         "  --video                 Encode a multi-frame P-frame sequence\n"
         "                       (opt-in; default remains one still picture)\n"
         "  --p-frames <N>          Video: number of P frames (default: all\n"
         "                       frames after the keyframe)\n"
         "  --b-frames <N>          Video: B frames (not wired yet; must be 0)\n"
         "  --video-refs <N>        Video: references per P frame, 1..3 (default 1)\n"
         "  --video-subpel-eighth   Video: add the 1/8-pel ME refinement stage\n"
         "  --video-mv-precision P  Video: eighth|quarter|half frame MV precision\n"
         "  --video-inter-residual  Video: code a quantised inter residual\n"
         "  --video-long-term       Video: signal long-term reference ids and tag\n"
         "                       the key frame as LongTermId 0\n"
         "  --video-olk             Video: open random access GOP (CLK, open-loop\n"
         "                       key, P2, P3; needs exactly 4 input frames)\n"
         "  --video-leading <N>     Video: N open-GOP leading pictures (needs at\n"
         "                       least N+2 input frames)\n"
         "  --video-sef             Video: re-show the last picture with a\n"
         "                       show-existing-frame OBU (last picture coded hidden)\n"
         "  --video-switch          Video: CLK + OBU_SWITCH (exactly 2 input frames)\n"
         "  --video-ras             Video: CLK + OBU_SWITCH + OBU_RAS_FRAME with a\n"
         "                       long-term reference (exactly 3 input frames)\n"
         "  --video-drl-reorder     Video: enable normative DRL reordering\n"
         "  --video-drl-reorder-always  Video: DRL_REORDER_ALWAYS (implies reorder)\n"
         "  --rc <mode>          Video: per-frame rate control, cqp|crf|vbr\n"
         "                       (default cqp = the fixed --qp stream)\n"
         "  --crf <V>            Video --rc crf: CRF value 0..63 (default 32;\n"
         "                       linear placeholder mapping to base_q_idx)\n"
         "  --bitrate <kbps>     Video --rc vbr: target bitrate in kbps. VBR runs\n"
         "                       a statistics pass at --qp, then plans one\n"
         "                       base_q_idx per frame\n"         "  --vram-cap <MiB>     Upper bound for the CUDA arena; default 0 takes\n"
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
         "  --still-ist-scale S    lambda sse_scale for --still-ist (0.125)\n"
         "  --gpu-plan <N>         Print the host-only multi-GPU tile plan for N\n"
         "                       GPUs (1..8) and exit: assignment, transfer and\n"
         "                       VRAM estimates; no device memory, no stream\n"
         "  --recon <file>         Dump the encoder reconstruction as Y4M (format\n"
         "                       paths and multi-tile --tiles stills; used by the\n"
         "                       decoder gates)\n"
         "  --video-recon <file>   Video: write the encoder reconstruction Y4M\n"
         "  --video-force-single   Video: one 64x64 leaf per superblock\n"
         "  --video-min-block <N>  Video: minimum partition block size (default 16)\n"
         "  --video-max-depth <D>  Video: partition depth cap [0, 8] (default 4)\n"
         "  --video-me-range <R>   Video: integer ME search radius (default 16)\n"
         "  --video-trace          Video: trace block decisions to stderr\n";
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

av2enc::tiles::TileLayoutPreset tile_preset_from_layout(std::string_view canonical) {
  using av2enc::tiles::TileLayoutPreset;
  if (canonical == "2x1") return TileLayoutPreset::k2x1;
  if (canonical == "2x2") return TileLayoutPreset::k2x2;
  if (canonical == "4x2") return TileLayoutPreset::k4x2;
  if (canonical == "8x2") return TileLayoutPreset::k8x2;
  if (canonical == "16x2") return TileLayoutPreset::k16x2;
  return TileLayoutPreset::k1x1;
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

    if (argument == "--video") {
      options.video = true;
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

    if (argument == "--video-refs") {
      if (i + 1 >= argc) {
        error = "missing value after '--video-refs'";
        return ParseStatus::kError;
      }
      int refs = 0;
      if (!parse_int(argv[++i], refs) || refs < 1 || refs > 3) {
        error = "--video-refs must be an integer in [1, 3]";
        return ParseStatus::kError;
      }
      options.video_refs = refs;
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

    if (argument == "--video-inter-residual") {
      options.video_inter_residual = true;
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
  if (options.output.empty() && options.gpu_plan == 0) {
    error = "no output file specified (use -o <output.ivf>)";
    return ParseStatus::kError;
  }
  if (options.b_frames != 0) {
    error = "--b-frames is not wired yet: the video path currently writes "
            "key + P frames only";
    return ParseStatus::kError;
  }
  if (options.video) {
    if (options.m1 || options.still_gpu || options.lossless) {
      error = "--video cannot be combined with --still, --still-gpu or "
              "--lossless";
      return ParseStatus::kError;
    }
    if (options.adaptive_cdf) {
      error = "--adaptive-cdf is not wired into the video path yet";
      return ParseStatus::kError;
    }
    if (options.video_drl_reorder) {
      error = "--video-drl-reorder is not decoder-gated yet: the pinned "
              "decoder disagrees with the encoder MV-stack order on some "
              "sub-64x64 blocks (docs/negative-results.md)";
      return ParseStatus::kError;
    }
    if (options.b_frames != 0) {
      error = "--b-frames is not wired into the video path yet";
      return ParseStatus::kError;
    }
    if (options.video_olk && options.video_refs != 1) {
      error = "--video-olk needs --video-refs 1";
      return ParseStatus::kError;
    }
    if ((options.video_long_term || options.video_olk) &&
        options.video_mv_precision != 0) {
      error = "--video-long-term/--video-olk need eighth-pel MV precision";
      return ParseStatus::kError;
    }
    if (options.video_leading != 0) {
      if (options.video_long_term || options.video_olk) {
        error = "--video-leading cannot be combined with --video-long-term or "
                "--video-olk";
        return ParseStatus::kError;
      }
      if (options.video_refs != 1 || options.video_mv_precision != 0) {
        error = "--video-leading needs --video-refs 1 and eighth-pel MV "
                "precision";
        return ParseStatus::kError;
      }
    }
    if (options.video_sef &&
        (options.video_long_term || options.video_olk ||
         options.video_leading != 0)) {
      error = "--video-sef cannot be combined with the long-term or leading "
              "frame-type modes";
      return ParseStatus::kError;
    }
    const int frame_modes =
        ((options.video_long_term || options.video_olk) ? 1 : 0) +
        (options.video_leading != 0 ? 1 : 0) + (options.video_sef ? 1 : 0) +
        (options.video_switch ? 1 : 0) + (options.video_ras ? 1 : 0);
    if (frame_modes > 1) {
      error = "select only one frame-type mode: --video-long-term/--video-olk, "
              "--video-leading, --video-sef, --video-switch or --video-ras";
      return ParseStatus::kError;
    }
    if ((options.video_switch || options.video_ras) &&
        (options.video_refs != 1 || options.video_mv_precision != 0)) {
      error = "--video-switch/--video-ras need --video-refs 1 and eighth-pel "
              "MV precision";
      return ParseStatus::kError;
    }
  } else {
    if (options.p_frames != 0 || options.video_refs != 1 ||
        options.video_inter_residual || options.video_subpel_eighth ||
        options.video_mv_precision != 0 || options.video_drl_reorder ||
        options.video_long_term || options.video_olk ||
        options.video_leading != 0 || options.video_sef ||
        options.video_switch || options.video_ras ||
        options.video_force_single || options.video_trace ||
        !options.video_recon.empty()) {
      error = "video options require --video";
      return ParseStatus::kError;
    }
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

struct OwnedVideoFrame {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
};

// One-frame Y4M dump for the tiled still path (8-bit 4:2:0). Comparing this
// with the decoded frame is how the --tiles gate proves sample-exact
// reconstruction; the CLI itself only writes the bitstream.
bool write_still_recon_y4m(const std::filesystem::path& path,
                           const av2enc::Y4mHeader& header, int width, int height,
                           const std::vector<std::uint8_t>& y,
                           const std::vector<std::uint8_t>& u,
                           const std::vector<std::uint8_t>& v, std::string& error) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    error = "cannot open --recon output '" + path.string() + "'";
    return false;
  }
  out << "YUV4MPEG2 W" << width << " H" << height << " F" << header.fps_num << ":"
      << header.fps_den << " Ip A0:0 C420\nFRAME\n";
  out.write(reinterpret_cast<const char*>(y.data()),
            static_cast<std::streamsize>(y.size()));
  out.write(reinterpret_cast<const char*>(u.data()),
            static_cast<std::streamsize>(u.size()));
  out.write(reinterpret_cast<const char*>(v.data()),
            static_cast<std::streamsize>(v.size()));
  if (!out) {
    error = "failed while writing --recon output";
    return false;
  }
  return true;
}


bool write_video_recon_y4m(const std::filesystem::path& path,
                           const av2enc::Y4mHeader& header,
                           const std::vector<av2enc::video::VideoReconFrame>& recon,
                           std::string& error) {  std::ofstream out(path, std::ios::binary);
  if (!out) {
    error = "cannot open --video-recon output '" + path.string() + "'";
    return false;
  }
  out << "YUV4MPEG2 W" << header.width << " H" << header.height << " F"
      << header.fps_num << ":" << header.fps_den << " Ip A1:1 C420\n";
  for (const auto& frame : recon) {
    out << "FRAME\n";
    out.write(reinterpret_cast<const char*>(frame.y.data()),
              static_cast<std::streamsize>(frame.y.size()));
    out.write(reinterpret_cast<const char*>(frame.u.data()),
              static_cast<std::streamsize>(frame.u.size()));
    out.write(reinterpret_cast<const char*>(frame.v.data()),
              static_cast<std::streamsize>(frame.v.size()));
  }
  if (!out) {
    error = "failed while writing --video-recon output";
    return false;
  }
  return true;
}

// Two-pass calibration hook for --rc vbr: scales the built-in placeholder
// bitrate model so that its estimate at the pass-1 QP matches the measured
// pass-1 payload bits. The controller then bisects the calibrated model. This
// corrects the model's absolute scale, which is documented as a placeholder.
struct ScaledBitrateModel {
  double scale = 1.0;
};

double scaled_bitrate_estimate(void* context,
                               const av2enc::rc::FrameComplexity& complexity,
                               int base_q_idx) {
  const auto* model = static_cast<const ScaledBitrateModel*>(context);
  return av2enc::rc::estimate_frame_bits_model(complexity, base_q_idx) *
         model->scale;
}

// Per-frame base_q_idx planning for --video. Empty output means the historical
// fixed --qp stream (the default). CRF and VBR go through the shared rate
// controller and the M2 adapter so the frame header and the coefficient-CDF
// slice always agree. VBR runs a statistics pass at --qp first: those payload
// bits calibrate the placeholder bitrate model, which is this encoder's
// two-pass step.
bool plan_video_rate_control(
    const Options& options, const av2enc::Y4mHeader& y4m,
    const std::vector<av2enc::video::VideoInputFrame>& inputs,
    const std::vector<std::uint32_t>* pass1_payload_bits,
    std::vector<std::uint8_t>* frame_qp, std::string& error) {
  frame_qp->clear();
  if (options.rc_mode == "cqp") {
    return true;
  }
  if (options.rc_mode == "vbr" && (y4m.fps_num == 0 || y4m.fps_den == 0)) {
    error = "--rc vbr needs a non-zero frame rate in the Y4M header";
    return false;
  }

  const int width = static_cast<int>(inputs[0].width);
  const int height = static_cast<int>(inputs[0].height);
  const std::vector<std::uint8_t> zero_plane(
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 0);

  std::vector<av2enc::rc::FrameComplexity> complexity(inputs.size());
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    std::span<const std::uint8_t> predicted(zero_plane);
    if (i > 0) {
      predicted = inputs[i - 1].y;  // previous source = motion-free inter proxy
    }
    complexity[i] = av2enc::rc::complexity_from_u8_planes(
        inputs[i].y, predicted, width, height, width);
    complexity[i].intra_fraction = i == 0 ? 1.0 : 0.0;
  }

  av2enc::video::post::VideoRcConfig rc_config;
  rc_config.depth = av2enc::fmt::BitDepth::k8;
  rc_config.policy.min_base_q_idx = av2enc::rc::kMinQIndex;
  rc_config.policy.max_base_q_idx = av2enc::rc::kMaxQIndex;
  ScaledBitrateModel model_scale;
  if (options.rc_mode == "crf") {
    rc_config.rc.mode = av2enc::rc::RateControlMode::kCrf;
    rc_config.rc.crf.crf = options.rc_crf;
  } else {
    rc_config.rc.mode = av2enc::rc::RateControlMode::kVbr;
    rc_config.rc.vbr.target_bitrate_bps = options.rc_bitrate_kbps * 1000.0;
    rc_config.rc.vbr.frame_rate_num = y4m.fps_num;
    rc_config.rc.vbr.frame_rate_den = y4m.fps_den;
    if (pass1_payload_bits != nullptr) {
      // Calibrate against the pass-1 payload at the fixed --qp.
      double measured = 0.0;
      double modelled = 0.0;
      for (std::size_t i = 0; i < complexity.size(); ++i) {
        measured += static_cast<double>((*pass1_payload_bits)[i]);
        modelled += av2enc::rc::estimate_frame_bits_model(
            complexity[i], static_cast<int>(options.qp));
      }
      if (modelled > 0.0 && std::isfinite(measured)) {
        model_scale.scale =
            std::clamp(measured / modelled, 1.0e-3, 1.0e3);
      }
      rc_config.rc.estimator.context = &model_scale;
      rc_config.rc.estimator.estimate = &scaled_bitrate_estimate;
    }
  }

  av2enc::video::post::VideoRateControl rc;
  if (!rc.configure(rc_config, &error)) {
    return false;
  }

  frame_qp->resize(inputs.size());
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    av2enc::video::post::VideoFrameQp qp;
    if (!rc.begin_frame(complexity[i], qp, &error)) {
      return false;
    }
    (*frame_qp)[i] = static_cast<std::uint8_t>(qp.base_q_idx);
    // Feed the controller's own (calibrated) estimate back into the virtual
    // buffer: the encoding pass is open-loop, so there is no later measurement.
    const std::int64_t actual_bits =
        static_cast<std::int64_t>(std::llround(qp.estimated_bits));
    if (!rc.end_frame(actual_bits, &error)) {
      return false;
    }
  }
  return true;
}

// Host-only multi-GPU planning dump (--gpu-plan). It reads the Y4M header for
// geometry/frame rate, plans the tile layout and the deterministic per-GPU
// assignment, prints the contract and exits. No CUDA context, allocation,
// kernel, stream or device query happens here, and no IVF is written.
int run_gpu_plan(const Options& options) {
  std::string error;
  av2enc::Y4mReader reader(options.input);
  if (!reader.open(error)) {
    std::cerr << "av2enc: " << error << "\n";
    return 1;
  }
  const av2enc::Y4mHeader& y4m = reader.header();
  if (y4m.width % 2U != 0U || y4m.height % 2U != 0U) {
    std::cerr << "av2enc: error: --gpu-plan requires even dimensions, got "
              << y4m.width << "x" << y4m.height << "\n";
    return 1;
  }

  const av2enc::tiles::TileLayoutPreset preset =
      options.tiles != "1x1"
          ? tile_preset_from_layout(options.tiles)
          : av2enc::multigpu::default_layout_for_gpus(options.gpu_plan);
  const av2enc::tiles::TileLayoutDims dims = av2enc::tiles::preset_dims(preset);
  const av2enc::tiles::LayoutResult planned = av2enc::tiles::plan_tile_layout(
      static_cast<int>(y4m.width), static_cast<int>(y4m.height), dims.cols,
      dims.rows, av2enc::tiles::kSuperblockPixels64, 7, 0);
  if (!planned.ok()) {
    std::cerr << "av2enc: error: --gpu-plan: layout "
              << av2enc::tiles::preset_name(preset) << " rejected: "
              << planned.error << "\n";
    return 1;
  }

  av2enc::multigpu::PlannerConfig config;
  config.gpu_count = options.gpu_plan;
  config.bit_depth = 8;
  // The plan has no device query (CPU-only host contract). Use --vram-cap when
  // the caller has one; otherwise the planner's compat8g emulation cap applies.
  // This is an estimate, never an allocation.
  if (options.vram_cap_mib != 0) {
    config.vram_cap_mb = static_cast<double>(options.vram_cap_mib);
  }
  config.target_fps = y4m.fps_den != 0
                          ? static_cast<double>(y4m.fps_num) /
                                static_cast<double>(y4m.fps_den)
                          : 1.0;

  av2enc::multigpu::MultiGpuPlan plan;
  if (!av2enc::multigpu::plan_multi_gpu(planned.layout, config, plan, error)) {
    std::cerr << "av2enc: error: --gpu-plan: " << error << "\n";
    return 1;
  }

  std::cout << "av2enc: host-only multi-GPU tile plan (no device allocation, "
               "no stream)\n"
            << "  input:   " << options.input.string() << " (" << y4m.width
            << "x" << y4m.height << ", " << y4m.fps_num << "/" << y4m.fps_den
            << " fps)\n"
            << "  layout:  " << av2enc::tiles::preset_name(preset) << " ("
            << plan.tile_count << " tiles), sb 64x64\n"
            << "  gpus:    " << config.gpu_count << ", "
            << av2enc::multigpu::assignment_policy_name(config.assignment)
            << "\n";
  for (const av2enc::multigpu::GpuPlan& gpu : plan.gpus) {
    std::cout << "  gpu " << gpu.gpu_index << ":  tiles";
    for (const int tile : gpu.tile_indices) {
      std::cout << " " << tile;
    }
    std::cout << " (" << gpu.tile_count << ", " << gpu.tile_area_pixels
              << " px), vram " << gpu.vram.live_mb << " / " << gpu.vram.cap_mb
              << " MB fits=" << (gpu.vram.fits ? "yes" : "no") << "\n";
  }
  std::cout << "  host:    transfer " << plan.transfer.total_mb << " MB, "
            << plan.transfer.total_ms << " ms ("
            << plan.transfer.pct_of_frame_budget << "% of the frame budget, "
            << plan.transfer.phases << " phases)\n"
            << "  scale:   model only (measured=" << (plan.measured ? 1 : 0)
            << "), estimated speedup " << plan.scaling.speedup
            << "x; not a measurement\n"
            << "  note:    " << plan.disclaimer << "\n";
  return 0;
}

int run_video_encode(const Options& options, av2enc::Y4mReader& reader,
                     av2enc::IvfWriter& writer,
                     const av2enc::Y4mHeader& y4m) {
  std::vector<OwnedVideoFrame> owned;
  std::string error;
  std::size_t limit = 0;
  if (options.max_frames != 0) {
    limit = static_cast<std::size_t>(options.max_frames);
  } else if (options.p_frames != 0) {
    limit = static_cast<std::size_t>(options.p_frames) + 1U;
  }
  av2enc::Y4mFrame frame;
  while (limit == 0 || owned.size() < limit) {
    const av2enc::Y4mReadResult status = reader.read_frame(frame, error);
    if (status == av2enc::Y4mReadResult::kEof) break;
    if (status == av2enc::Y4mReadResult::kError) {
      std::cerr << "av2enc: " << error << "\n";
      return 1;
    }
    OwnedVideoFrame copy;
    copy.width = frame.width();
    copy.height = frame.height();
    copy.y.assign(frame.y().begin(), frame.y().end());
    copy.u.assign(frame.u().begin(), frame.u().end());
    copy.v.assign(frame.v().begin(), frame.v().end());
    owned.push_back(std::move(copy));
  }
  if (owned.empty()) {
    std::cerr << "av2enc: error: input has no frames\n";
    return 1;
  }

  std::vector<av2enc::video::VideoInputFrame> inputs;
  inputs.reserve(owned.size());
  for (const OwnedVideoFrame& f : owned) {
    av2enc::video::VideoInputFrame input;
    input.width = f.width;
    input.height = f.height;
    input.y = std::span<const std::uint8_t>(f.y.data(), f.y.size());
    input.u = std::span<const std::uint8_t>(f.u.data(), f.u.size());
    input.v = std::span<const std::uint8_t>(f.v.data(), f.v.size());
    inputs.push_back(input);
  }

  av2enc::video::VideoOptions video_options;
  video_options.base_q_idx = static_cast<std::uint8_t>(options.qp);
  video_options.num_total_refs = options.video_refs;
  video_options.enable_inter_residual = options.video_inter_residual;
  video_options.me_subpel_eighth = options.video_subpel_eighth;
  video_options.mv_precision = options.video_mv_precision;
  video_options.drl_reorder = options.video_drl_reorder;
  video_options.drl_reorder_always = options.video_drl_reorder_always;
  video_options.long_term = options.video_long_term;
  video_options.olk = options.video_olk;
  video_options.leading_frames = options.video_leading;
  video_options.sef = options.video_sef;
  video_options.ra_switch = options.video_switch;
  video_options.ras = options.video_ras;
  video_options.force_single_block = options.video_force_single;
  video_options.min_block_size = options.video_min_block;
  video_options.max_partition_depth = options.video_max_depth;
  video_options.me_range = options.video_me_range;
  video_options.trace = options.video_trace;

  // Opt-in per-frame rate control. CQP (default) leaves frame_base_q_idx empty,
  // so encode_video_sequence runs the historical fixed-QP path unchanged.
  std::vector<std::uint8_t> frame_qp;
  if (options.rc_mode != "cqp") {
    std::vector<std::uint32_t> pass1_payload_bits;
    if (options.rc_mode == "vbr") {
      // Statistics pass at the fixed --qp: payload bits feed the VBR virtual
      // buffer, the same role as a first pass in a conventional encoder.
      av2enc::video::VideoSequenceResult stats;
      if (!av2enc::video::encode_video_sequence(inputs, video_options, stats,
                                                error)) {
        std::cerr << "av2enc: error: --rc vbr statistics pass: " << error
                  << "\n";
        return 1;
      }
      pass1_payload_bits.reserve(stats.tile_bytes.size());
      for (const std::uint32_t bytes : stats.tile_bytes) {
        pass1_payload_bits.push_back(bytes * 8U);
      }
    }
    if (!plan_video_rate_control(
            options, y4m, inputs,
            options.rc_mode == "vbr" ? &pass1_payload_bits : nullptr, &frame_qp,
            error)) {
      std::cerr << "av2enc: error: --rc " << options.rc_mode << ": " << error
                << "\n";
      return 1;
    }
    video_options.frame_base_q_idx = frame_qp;
    video_options.base_q_idx = frame_qp.front();
  }

  av2enc::video::VideoSequenceResult result;
  if (!av2enc::video::encode_video_sequence(inputs, video_options, result,
                                            error)) {
    std::cerr << "av2enc: error: " << error << "\n";
    return 1;
  }
  for (std::size_t i = 0; i < result.units.size(); ++i) {
    if (!writer.write_frame(result.units[i],
                            static_cast<std::uint64_t>(i), error)) {
      std::cerr << "av2enc: error: " << error << "\n";
      return 1;
    }
  }
  if (!options.video_recon.empty()) {
    if (!write_video_recon_y4m(options.video_recon, y4m, result.recon, error)) {
      std::cerr << "av2enc: error: " << error << "\n";
      return 1;
    }
  }
  if (!writer.close(error)) {
    std::cerr << "av2enc: error: " << error << "\n";
    return 1;
  }

  std::uint64_t total_tile = 0;
  std::uint64_t total_inter = 0;
  std::uint64_t total_intra = 0;
  for (std::size_t i = 0; i < result.tile_bytes.size(); ++i) {
    total_tile += result.tile_bytes[i];
    total_inter += result.inter_blocks[i];
    total_intra += result.intra_blocks[i];
  }
  std::cout << "av2enc: Video P-frame encode complete\n"
            << "  input:   " << options.input.string() << " (" << y4m.width
            << "x" << y4m.height << ", " << y4m.colorspace << ", "
            << y4m.fps_num << "/" << y4m.fps_den << " fps)\n"
            << "  output:  " << options.output.string() << " ("
            << writer.bytes_written() << " bytes IVF)\n"
            << "  frames:  " << result.units.size() << " temporal unit(s), "
            << result.tile_bytes.size() << " coded frame(s), "
            << result.recon.size() << " output frame(s), base_q_idx ";
  if (frame_qp.empty()) {
    std::cout << static_cast<int>(video_options.base_q_idx);
  } else {
    std::cout << "per-frame";
  }
  std::cout << ", tile " << total_tile << " bytes, inter blocks " << total_inter
            << ", intra blocks " << total_intra << "\n";
  if (!frame_qp.empty()) {
    const double duration =
        static_cast<double>(result.units.size()) *
        static_cast<double>(y4m.fps_den) / static_cast<double>(y4m.fps_num);
    const double kbps = duration > 0.0
                            ? static_cast<double>(total_tile) * 8.0 / duration /
                                  1000.0
                            : 0.0;
    std::cout << "  rc:      " << options.rc_mode;
    if (options.rc_mode == "crf") {
      std::cout << " " << options.rc_crf;
    } else if (options.rc_mode == "vbr") {
      std::cout << " target " << options.rc_bitrate_kbps
                << " kbps (two-pass)";
    }
    std::cout << ", achieved " << kbps << " kbps (payload)\n"
              << "  frame QP:";
    for (const std::uint8_t qp : frame_qp) {
      std::cout << " " << static_cast<int>(qp);
    }
    std::cout << "\n  frame bits:";
    for (const std::uint32_t bytes : result.tile_bytes) {
      std::cout << " " << (static_cast<std::uint64_t>(bytes) * 8U);
    }
    std::cout << "\n";
  }
  return 0;
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

  if (options.gpu_plan != 0) {
    if (options.video || options.m1 || options.still_gpu || options.lossless) {
      std::cerr << "av2enc: error: --gpu-plan is a host-planning dry run; do "
                   "not combine it with --video, --still, --still-gpu or "
                   "--lossless\n";
      return 2;
    }
    return run_gpu_plan(options);
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

  // Multi-tile stills (--tiles).  The tiled writer is a distinct composition
  // root: it reuses the M1 reference per tile but pins the tool set to the
  // verified form, so every option that would change the tile payload is
  // rejected by name instead of being ignored.
  const bool multi_tile = options.tiles != "1x1";
  if (multi_tile) {
    if (options.video) {
      std::cerr << "av2enc: error: --tiles is not wired into --video yet; "
                   "multi-tile video needs the per-frame tile splitter\n";
      return 2;
    }
    if (options.still_gpu) {
      std::cerr << "av2enc: error: --tiles is CPU-only (--still); the Still GPU "
                   "pipeline has no tile loop\n";
      return 2;
    }
    if (!options.m1) {
      std::cerr << "av2enc: error: --tiles > 1 requires --still\n";
      return 2;
    }
    if (options.lossless) {
      std::cerr << "av2enc: error: --tiles is not available in the --lossless "
                   "writer\n";
      return 2;
    }
    if (options.bit_depth != 8 || options.chroma != "420") {
      std::cerr << "av2enc: error: the tiled still writer is 8-bit 4:2:0 only; "
                   "--tiles cannot be combined with --bit-depth 10 or --chroma\n";
      return 2;
    }
    if (options.adaptive_cdf) {
      std::cerr << "av2enc: error: --tiles resets frozen CDFs per tile and does "
                   "not signal --adaptive-cdf\n";
      return 2;
    }
    if (options.still_deblock) {
      std::cerr << "av2enc: error: --tiles requires --no-deblock: the pinned "
                   "dav2d filters across interior tile edges, so multi-tile "
                   "reconstruction would not be sample-exact\n";
      return 2;
    }
    const std::pair<bool, const char*> unsupported[] = {
        {options.still_single, "--still-single"},
        {options.still_dc, "--still-dc"},
        {options.still_full_intra, "--still-full-intra"},
        {options.still_uv_full, "--still-uv-full"},
        {options.still_rdo_intra, "--still-rdo-intra"},
        {options.still_rdoq, "--still-rdoq"},
        {options.still_tx_type, "--still-tx-type"},
        {options.still_cdef, "--still-cdef"},
        {options.still_ccso, "--still-ccso"},
        {options.still_sb128, "--still-sb128"},
        {options.still_sb256, "--still-sb256"},
        {options.still_ext_partitions, "--still-ext-partitions"},
        {options.still_segmentation, "--still-segmentation"},
        {options.still_delta_q, "--still-delta-q"},
        {options.still_content_interpretation,
         "--still-content-interpretation"},
        {options.still_qm, "--still-qm"},
        {options.still_ist, "--still-ist"},
        {options.still_dc_ac_delta_q, "--still-dc-ac-delta-q"},
        {options.still_ibp, "--still-ibp"},
        {options.still_mrl, "--still-mrl"},
        {options.still_cfl_ds != 0, "--still-cfl-ds"},
        {options.still_cfl_explicit, "--still-cfl-explicit"},
        {options.still_dip, "--still-dip"},
        {options.still_palette, "--still-palette"},
        {options.still_mhccp, "--still-mhccp"},
        {options.still_ccso_edge, "--still-ccso-edge"},
        {options.still_wiener, "--still-wiener"},
        {options.still_chroma_lr, "--still-chroma-lr"},
        {options.still_pc_wiener, "--still-pc-wiener"},
        {options.still_gdf, "--still-gdf"},
        {options.still_grain, "--still-grain"},
        {options.still_depth >= 0, "--still-depth"},
    };
    for (const auto& [set, flag] : unsupported) {
      if (!set) continue;
      std::cerr << "av2enc: error: --tiles uses the tiled still writer, which "
                   "does not implement "
                << flag << "; drop the flag or use a single-tile --still "
                         "encode\n";
      return 2;
    }
  }

  // Optical-format still paths (10-bit, 4:2:2/4:4:4/4:0:0).  Every flag
  // defaults to today's 8-bit 4:2:0 stream, which never enters this branch.
  const bool format_10bit = options.bit_depth == 10;
  const bool format_chroma = options.chroma != "420";
  if (format_10bit || format_chroma) {
    if (options.still_gpu) {
      std::cerr << "av2enc: error: --bit-depth/--chroma are CPU-only: the "
                   "Still GPU pipeline is 8-bit 4:2:0\n";
      return 2;
    }
    if (!options.m1) {
      std::cerr << "av2enc: error: --bit-depth/--chroma require --still\n";
      return 2;
    }
    if (options.lossless) {
      std::cerr << "av2enc: error: --lossless has its own 8-bit 4:2:0 writer "
                   "and cannot be combined with --bit-depth/--chroma\n";
      return 2;
    }
    if (options.adaptive_cdf) {
      std::cerr << "av2enc: error: --adaptive-cdf is not signalled by the "
                   "format still writers (frozen CDFs)\n";
      return 2;
    }
    if (options.still_single || options.still_dc || options.still_full_intra ||
        options.still_uv_full || options.still_rdo_intra || options.still_rdoq ||
        options.still_tx_type || options.still_cdef || options.still_ccso ||
        options.still_sb128 || options.still_sb256 ||
        options.still_ext_partitions || options.still_segmentation ||
        options.still_delta_q || options.still_content_interpretation ||
        options.still_qm || options.still_ist || options.still_dc_ac_delta_q ||
        options.still_ibp || options.still_mrl || options.still_cfl_ds != 0 ||
        options.still_cfl_explicit || options.still_dip || options.still_palette ||
        options.still_mhccp || options.still_ccso_edge || options.still_wiener ||
        options.still_chroma_lr || options.still_pc_wiener || options.still_gdf ||
        options.still_grain || options.still_depth >= 0) {
      std::cerr << "av2enc: error: the --bit-depth/--chroma format paths use "
                   "their own minimal still writer; M1 tool flags are not "
                   "applied there\n";
      return 2;
    }
    if (format_10bit && (options.qp < 1 || options.qp > 90)) {
      std::cerr << "av2enc: error: 10-bit encoding supports base_q_idx 1..90 "
                   "(coefficient-CDF slice 0); got --qp "
                << options.qp << "\n";
      return 2;
    }
    if (!format_10bit && options.qp != 90) {
      std::cerr << "av2enc: error: 8-bit 4:2:2/4:4:4/4:0:0 pins --qp 90; got "
                << options.qp << "\n";
      return 2;
    }

    av2enc::formats::StillFormatOptions format_options;
    format_options.depth = format_10bit ? av2enc::fmt::BitDepth::k10
                                        : av2enc::fmt::BitDepth::k8;
    if (options.chroma == "422") {
      format_options.chroma = av2enc::fmt::ChromaFormat::k422;
    } else if (options.chroma == "444") {
      format_options.chroma = av2enc::fmt::ChromaFormat::k444;
    } else if (options.chroma == "400") {
      format_options.chroma = av2enc::fmt::ChromaFormat::k400;
    } else {
      format_options.chroma = av2enc::fmt::ChromaFormat::k420;
    }
    format_options.base_q_idx = static_cast<std::uint16_t>(options.qp);
    format_options.recon_out = options.recon;

    av2enc::formats::StillFormatStats stats;
    if (!av2enc::formats::encode_still_format(options.input, options.output,
                                              format_options, &stats, &error)) {
      std::cerr << "av2enc: error: " << error << "\n";
      return 1;
    }
    std::cout << "av2enc: Format still-picture encode complete\n"
              << "  input:   " << options.input.string() << " (" << stats.width
              << "x" << stats.height << ", "
              << av2enc::fmt::y4m_colorspace_tag(stats.chroma, stats.depth)
              << ", " << stats.fps_num << "/" << stats.fps_den << " fps)\n"
              << "  output:  " << options.output.string() << " ("
              << stats.ivf_bytes << " bytes IVF)\n"
              << "  format:  " << stats.width << "x" << stats.height << " "
              << av2enc::fmt::bit_depth_name(stats.depth) << ", "
              << av2enc::fmt::chroma_format_name(stats.chroma) << "\n"
              << "  frames:  1, base_q_idx " << options.qp << ", tile "
              << stats.tile_bytes << " bytes, temporal unit "
              << stats.frame_bytes << " bytes\n";
    if (!options.recon.empty()) {
      std::cout << "  recon:   " << options.recon.string() << "\n";
    }
    return 0;
  }

  if (!options.recon.empty() && !multi_tile) {
    std::cerr << "av2enc: error: --recon is wired for the --bit-depth 10 and "
                 "--chroma format paths and for multi-tile --tiles stills only\n";
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
  if (options.video && options.qp == 0) {
    std::cerr << "av2enc: error: --video does not support lossless QP 0\n";
    return 1;
  }
  if (!options.lossless && !still_cpu_qp && !options.video &&
      options.qp != 90) {
    std::cerr << "av2enc: error: "
              << (options.still_gpu ? "Still GPU" : "M0")
              << " pins --qp 90; use --still for QP 1..255\n";
    return 1;
  }

  if (options.rc_given && !options.video) {
    std::cerr << "av2enc: error: --rc is wired for --video only; the still "
                 "paths keep the fixed --qp stream\n";
    return 2;
  }

  if (options.video) {
    if (options.rc_given && options.rc_mode == "vbr" && !options.rc_bitrate_given) {
      std::cerr << "av2enc: error: --rc vbr requires --bitrate <kbps>\n";
      return 2;
    }
    if (options.rc_crf_given && options.rc_mode != "crf") {
      std::cerr << "av2enc: error: --crf requires --rc crf\n";
      return 2;
    }
    if (options.rc_bitrate_given && options.rc_mode != "vbr") {
      std::cerr << "av2enc: error: --bitrate requires --rc vbr\n";
      return 2;
    }
    if (y4m.width % 8U != 0U || y4m.height % 8U != 0U) {
      std::cerr << "av2enc: error: --video requires dimensions divisible by 8, "
                << "got " << y4m.width << "x" << y4m.height << "\n";
      return 1;
    }
    return run_video_encode(options, reader, writer, y4m);
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
    } else if (multi_tile) {
      // One independent §8.2 entropy stream per tile; every tile is encoded
      // from its own local origin through the M1 reference, so CDFs, above/left
      // contexts and prediction availability reset at the tile boundary
      // (§5.20.2.1, §5.20.9.1).  Deblocking is off (checked above): the pinned
      // decoder filters shared tile edges and would break exactness.
      av2enc::still::tiles_stream::Config tile_config;
      tile_config.width = static_cast<int>(frame.width());
      tile_config.height = static_cast<int>(frame.height());
      tile_config.preset = tile_preset_from_layout(options.tiles);
      tile_config.base_q_idx = static_cast<std::uint8_t>(options.qp);
      tile_config.enable_deblock = false;
      tile_config.disable_loopfilters_across_tiles = true;
      tile_config.seq_level_idx = 7;
      tile_config.enable_cfl_intra = options.still_cfl;
      tile_config.enable_intra_edge_filter = true;
      tile_config.enable_chroma_dctonly = true;

      av2enc::still::tiles_stream::Frame tiled;
      std::string tile_error;
      if (!av2enc::still::tiles_stream::encode(picture, tile_config, &tiled,
                                               &tile_error)) {
        return fail_encode("--tiles " + options.tiles + ": " + tile_error);
      }
      if (!writer.write_frame(tiled.frame, 0, error)) {
        return fail_encode(error);
      }
      std::uint32_t tile_payload_bytes = 0;
      for (const auto& tile : tiled.tiles) {
        tile_payload_bytes += static_cast<std::uint32_t>(tile.payload.size());
      }
      tile_bytes = tile_payload_bytes;
      frame_bytes = static_cast<std::uint32_t>(tiled.frame.size());

      if (!options.recon.empty()) {
        av2enc::still::tiles_stream::Reconstruction recon;
        std::string recon_error;
        if (!av2enc::still::tiles_stream::assemble_reconstruction(
                tiled, &recon, &recon_error)) {
          return fail_encode(recon_error);
        }
        if (!write_still_recon_y4m(options.recon, y4m, recon.width,
                                   recon.height, recon.y, recon.u, recon.v,
                                   error)) {
          return fail_encode(error);
        }
      }
      std::cout << "  tiles:   " << tiled.tiles.size() << " (" << options.tiles
                << "), tile group " << tiled.clk_payload.size() << " bytes ("
                << tiled.tile_size_bytes << "-byte size fields)\n";
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
                 : (options.m1 ? (multi_tile ? "M1 tiled" : "M1")
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

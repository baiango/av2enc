#include "core/bitstream/basic_headers.h"
#include "codec/entropy/lossless/lossless.h"
#include "formats/chroma/chroma_still.h"
#include "formats/format_encode.h"
#include "formats/format_y4m.h"
#include "pipeline/still/basic_encoder.h"
#include "pipeline/still/still_encoder.h"
#include "pipeline/svc/svc_stream.h"
#include "pipeline/tiles/stream/tiles_stream.h"
#include "pipeline/video/post/rc_adapter.h"
#include "pipeline/dispatch/planner.h"
#include "pipeline/planner/effort.h"
#include "core/io/ivf.h"
#include "core/io/y4m.h"
#include "pipeline/video/video_frame.h"
#include "frame/metadata/types/metadata_types.h"

#include <charconv>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
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
// src/cuda/pipeline/still/still_gpu.h lands in the tree.  Field names follow
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
  int ibc_blocks = 0;
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

// Video GPU pipeline (owner: orchestrator, src/cuda/pipeline/video/).  Same
// pattern as the Still path: include the contract header when visible and only
// treat the call as linkable when CMake built the M2 target.
#if __has_include("cuda/pipeline/video/video_gpu.h")
#include "cuda/pipeline/video/video_gpu.h"
#define AV2ENC_VIDEO_GPU_HEADER 1
#else
#define AV2ENC_VIDEO_GPU_HEADER 0
#endif
#if AV2ENC_VIDEO_GPU_HEADER && defined(AV2ENC_HAVE_VIDEO_GPU) && AV2ENC_HAVE_VIDEO_GPU
#define AV2ENC_VIDEO_GPU_CALLABLE 1
#else
#define AV2ENC_VIDEO_GPU_CALLABLE 0
#endif
#else
#define AV2ENC_STILL_GPU_CALLABLE 0
#define AV2ENC_VIDEO_GPU_CALLABLE 0
#endif

namespace {

struct Options {
  std::filesystem::path input;
  std::filesystem::path output;
  std::uint64_t max_frames = 0;  // 0 = all frames
  int qp = 90;
  bool adaptive_cdf = false;
  // §8.2.4/§7.5 average-CDF bookkeeping (opt-in; multi-tile stills now, video
  // once the cross-frame seed export is wired).
  bool avg_cdf = false;
  // §5.18.2 cross-frame CDF carry on --video (requires --avg-cdf).
  bool cdf_continuity = false;
  // §5.4.1/§5.4.12/§5.4.13 timing on --video.
  bool video_timing = false;
  // §5.7 multi-frame header on --video.
  bool video_mfh = false;
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
  bool still_cdef_on_skip_txfm = false;
  int still_cdef_strengths = 4;
  bool still_ccso = false;
  int still_ccso_bands = 6;
  bool still_sb128 = false;
  bool still_sb256 = false;
  bool still_ext_partitions = false;
  bool still_sub8 = false;
  bool still_sdp = false;
  bool still_segmentation = false;
  bool still_delta_q = false;
  int still_delta_q_res = 0;
  // Real per-SB segmentation with SEG_LVL_ALT_Q (opt-in).
  bool still_per_segment_q = false;
  int still_seg_delta = 8;
  bool lossless = false;
  bool lossless_force_fsc = false;
  bool still_content_interpretation = false;
  // Neutral metadata OBUs (§5.17).  Both default off; the payloads are the
  // all-zero/minimal spec-valid forms and do not affect reconstruction.
  bool metadata_timecode = false;
  bool metadata_scan_type = false;
  bool still_qm = false;
  int still_qm_level = 3;
  bool still_user_qm = false;
  bool still_ist = false;
  double still_ist_scale = 0.125;
  bool still_tcq = false;
  double still_tcq_scale = 0.125;
  bool still_cctx = false;
  bool still_tx_partition = false;
  bool still_idtx = false;
  bool still_ddt = false;
  bool still_dc_ac_delta_q = false;
  bool still_ibp = false;
  bool still_mrl = false;
  int still_cfl_ds = 0;
  bool still_cfl_explicit = false;
  bool still_dip = false;
  bool still_palette = false;
  bool still_ibc = false;
  int still_ibc_search_step = 8;
  bool still_mhccp = false;
  bool still_ccso_edge = false;
  bool still_wiener = false;
  bool still_chroma_lr = false;
  bool still_pc_wiener = false;
  bool still_gdf = false;
  bool still_grain = false;
  bool still_grain_adaptive = false;
  int still_depth = -1;  // -1 = component default
  // Multi-frame inter (video) path.  Opt-in; every default still/basic stream
  // is untouched.
  bool video = false;
  // Opt-in CUDA M2 backend for --video (--video-gpu).  Implies --video.
  bool video_gpu = false;
  std::uint64_t p_frames = 0;  // 0 = every frame after the keyframe
  std::uint64_t b_frames = 0;  // B frames per GOP (0 = key + P chain)
  int video_refs = 1;          // NumTotalRefs (1..7)
  // §5.4.6 NumRefFrames (4..8).  0 = the derived historical value (2 for the
  // single-reference chain, 8 for the multi-reference model).
  int video_num_ref_frames = 0;
  // Documented rejected carrier for the §7.8 implicit reference lists.
  bool video_implicit_ref_map = false;
  bool video_compound = false;  // reference_select compound (needs refs 2)
  bool video_inter_residual = false;
  // Opt-in §5.18.8.1 TX_MODE_SELECT luma TX size/type/partition search.
  bool video_tx_search = false;
  double video_tx_lambda = 256.0;
  // Opt-in §7.17 deblocking of every reconstructed video frame.
  bool video_deblock = false;
  bool video_gme = false;
  bool video_warp = false;
  bool video_intra_full = false;  // opt-in 61-candidate luma intra search
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
  int video_tlayers = 0;   // 0 = off; N = one closed hierarchical-P GOP
  int video_tlayer_drop = -1;  // -1 = keep every layer
  bool video_force_single = false;
  int video_min_block = 16;
  int video_max_depth = 4;
  int video_me_range = 16;
  bool video_cdef = false;
  int video_cdef_strength = 1;  // CdefStrengths 1..8
  int video_cdef_damping = 3;   // CdefDamping 3..6
  bool video_ccso = false;
  int video_ccso_bands = 4;  // CcsoMaxBandLog2 (band-offset-only form)
  bool video_wiener = false;
  bool video_chroma_lr = false;
  bool video_gdf = false;
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
  // Measured effort ranking (x264-style preset).  Every field is inert
  // unless --effort or --cpu-used is given; the default stream is
  // byte-identical because the selector is never called.
  bool effort_given = false;
  int effort_budget = 0;
  bool cpu_used_given = false;
  int cpu_used = -1;
  bool effort_plan = false;
  bool effort_active = false;
  std::string effort_report;
  // Option tokens seen on the command line (flags only, not their values).
  // Validation uses this to reject combinations in which a flag would
  // otherwise be accepted and silently ignored.
  std::vector<std::string_view> specified;

  // SVC layer encode (opt-in).  Any --svc-* flag sets svc_given; --oppoint,
  // --alllayers and --spatial-layers select the output subset.
  std::string svc_layers;
  bool svc_given = false;
  int svc_temporal = 1;
  int svc_embedded = 1;
  int svc_lcr_id = 1;
  int svc_oppoint = -1;
  bool svc_all_layers = false;
  int svc_spatial_layers = -1;
  bool help = false;
};

bool flag_given(const Options& options, std::string_view flag) {
  return std::find(options.specified.begin(), options.specified.end(), flag) !=
         options.specified.end();
}

enum class ParseStatus {
  kOk,
  kHelp,
  kError,
};

// Appends the opt-in neutral metadata OBUs as §5.17.2 suffix units at the end
// of one temporal unit.  The payloads are the minimal spec-valid forms (no
// invented timing/hash policy); pinned dav2d ignores OBU types 8/9, so the
// gate is decode invariance.
bool append_neutral_metadata_obus(std::vector<std::uint8_t>& unit,
                                  bool timecode, bool scan_type,
                                  std::string& error) {
  if (!timecode && !scan_type) return true;
  av2enc::hdr::MetadataObuConfig config;
  config.suffix = true;
  config.layer = av2enc::hdr::MetadataLayer::kCurrent;
  config.persistence = av2enc::hdr::MetadataPersistence::kGlobal;
  if (timecode) {
    av2enc::metadata::Timecode value;  // all-zero minimal 23-bit unit
    const auto payload = av2enc::metadata::make_timecode_unit(value, error);
    if (!payload.has_value()) return false;
    const std::vector<std::uint8_t> obu =
        av2enc::metadata::make_short_obu(*payload, config, error);
    if (obu.empty()) return false;
    unit.insert(unit.end(), obu.begin(), obu.end());
  }
  if (scan_type) {
    av2enc::metadata::ScanType value;
    value.mps_pic_struct_type = 0;
    value.mps_source_scan_type_idc = 1;  // Table 6.18: progressive
    value.mps_duplicate_flag = false;
    const auto payload = av2enc::metadata::make_scan_type_unit(value, error);
    if (!payload.has_value()) return false;
    const std::vector<std::uint8_t> obu =
        av2enc::metadata::make_short_obu(*payload, config, error);
    if (obu.empty()) return false;
    unit.insert(unit.end(), obu.begin(), obu.end());
  }
  return true;
}

void print_usage(std::ostream& out) {
  constexpr std::size_t kColumn = 30;
  const auto item = [&out](std::string_view name,
                           std::initializer_list<std::string_view> lines) {
    out << "  " << name;
    if (name.size() < kColumn) {
      out << std::string(kColumn - name.size(), ' ');
    } else {
      out << "\n" << std::string(kColumn + 2, ' ');
    }
    bool first = true;
    for (const std::string_view text : lines) {
      if (!first) {
        out << std::string(kColumn + 2, ' ');
      }
      out << text << "\n";
      first = false;
    }
  };

  out << "Usage: av2enc <input.y4m> -o <output.ivf> [options]\n"
         "       av2enc <input.y4m> --gpu-plan <N> [--tiles <layout>]\n"
         "\n"
         "Modes (default: Basic):\n"
         "  Basic       One key frame: 8-bit 4:2:0, 64x64 superblocks, 32x32\n"
         "              DCT, DC intra only, no filters.\n"
         "  Still       --still: partition tree, full intra modes, variable\n"
         "              transform sizes, CfL and deblock. The CPU reference.\n"
         "  Still GPU   --still-gpu: the same still syntax with the block\n"
         "              pipeline on the GPU. Requires a CUDA build and implies\n"
         "              --still; Still tools without a GPU stage are rejected\n"
         "              by name instead of being ignored.\n"
         "  Video       --video: multi-frame P-frame sequence.\n"
         "  Lossless    --lossless: lossless still picture with its own writer\n"
         "              (DPCM intra, 8x8 blocks, Q=0).\n"
         "Still modes encode exactly one frame; Video encodes --frames frames.\n"
         "Every mode writes an AV2 IVF (fourcc AV02). Usage and option-\n"
         "combination errors exit 2; encoder and I/O failures exit 1.\n"
         "\n"
         "Basic options:\n";
  item("-o, --output <file>", {"Output IVF path (required unless --gpu-plan)"});
  item("--frames <N>", {"Frames to encode; still modes accept only 1"});
  item("--qp <Q>", {"base_q_idx (default 90). Basic pins 90; the CPU Still",
                    "path accepts 1..255 (1..303 for 10-bit); Q=0 requires",
                    "--lossless"});
  item("--adaptive-cdf", {"Enable normative 8.2.6 CDF adaptation"});
  item("--avg-cdf", {"Multi-tile still/video: seed each tile from the frame",
                     "CDFs and average at frame end (§8.2.4/§7.5)"});
  item("--cdf-continuity", {"Video: carry the adapted CDFs across frames by",
                            "loading the primary reference's saved context",
                            "(§5.18.2; requires --avg-cdf --adaptive-cdf)"});
  item("--vram-cap <MiB>", {"CUDA arena upper bound; 0 takes the largest",
                            "reservation the device can serve (default 0).",
                            "CPU-only builds accept it as a documented no-op"});
  item("-h, --help", {"Show this help"});

  out << "\nStill options (--still, --still-gpu):\n";
  item("--no-cfl", {"Disable chroma-from-luma"});
  item("--no-deblock", {"Disable the deblocking filter"});
  item("--still-full-intra", {"Directional/smooth intra search (SSE-only mode",
                              "decision; DC is the default)"});
  item("--still-uv-full", {"Full chroma intra search (DC/CfL is the default;",
                           "CPU-only)"});
  item("--still-rdo-intra", {"Rate-aware luma intra mode decision (CPU-only)"});
  item("--still-rdoq", {"Full-trellis coefficient RDOQ (CPU-only)"});
  item("--still-tx-type", {"Intra luma TX_TYPE search (CPU-only)"});
  item("--still-cdef", {"CDEF, after deblock (CPU; runs on --still-gpu)"});
  item("--still-ccso", {"Band-offset-only CCSO, after CDEF (CPU; runs on",
                       "--still-gpu)"});
  item("--still-ccso-edge", {"Edge-classified CCSO + per-unit skip (CPU; runs",
                             "on --still-gpu)"});
  item("--still-wiener", {"Luma Wiener NS loop restoration (CPU; runs on",
                          "--still-gpu)"});
  item("--still-chroma-lr", {"Chroma Wiener NS loop restoration (CPU; runs on",
                             "--still-gpu)"});
  item("--still-pc-wiener", {"Luma PC-Wiener loop restoration (CPU; runs on",
                             "--still-gpu)"});
  item("--still-gdf", {"Guided detail filter after loop restoration (CPU;",
                       "runs on --still-gpu)"});
  item("--still-grain", {"Signal film grain for decoder synthesis (CPU; runs",
                         "on --still-gpu)"});
  item("--still-grain-adaptive", {"Estimate and signal a film-grain model from",
                                  "the reconstruction; requires --still-grain",
                                  "(CPU-only)"});
  item("--still-cfl-ds <N>", {"cfl_ds_filter_index 0..3 (default 0; runs on",
                              "--still-gpu)"});
  item("--still-cfl-explicit", {"Allow explicit CfL alpha, CFL_EXPLICIT",
                                "(CPU; runs on --still-gpu)"});
  item("--still-ibp", {"Intra bi-prediction, 5.4.5 (CPU; runs on",
                       "--still-gpu)"});
  item("--still-mrl", {"Multiple reference lines for directional luma;",
                       "implies the directional search (CPU; runs on",
                       "--still-gpu)"});
  item("--still-dip", {"DIP as a luma DC alternative (CPU; runs on",
                       "--still-gpu)"});
  item("--still-mhccp", {"MHCCP (CFL_MULTI) chroma prediction (CPU; runs",
                         "on --still-gpu)"});
  item("--still-segmentation", {"Signal a neutral 8-segment table (runs on",
                                "--still-gpu)"});
  item("--still-delta-q", {"Signal neutral per-superblock delta-Q (runs",
                            "on --still-gpu)"});
  item("--still-per-segment-q", {"Real per-SB segmentation with ALT_Q; low-",
                                 "variance SBs use base_q_idx + delta (CPU;",
                                 "runs on --still-gpu)"});
  item("--still-sb128", {"128x128 superblocks (default 64x64; CPU; runs",
                          "on --still-gpu)"});
  item("--still-sb256", {"Signal 256x256 sequence superblocks; intra frames",
                         "still code 128x128 (CPU; runs on --still-gpu)"});
  item("--still-ext-partitions", {"HORZ_3/VERT_3 and uneven 4-way partitions",
                                  "(CPU; runs on --still-gpu)"});
  item("--still-sub8", {"Allow sub-8x8 coding blocks (4xN/Nx4); needs a depth",
                        "of at least 8 (CPU; runs on --still-gpu)"});
  item("--still-sdp", {"Semi-decoupled partition: luma tree then chroma tree",
                       "(forced PARTITION_NONE trees; 64-SB only; CPU; runs on",
                       "--still-gpu)"});
  item("--still-qm", {"Spec-defined quantizer matrix (CPU; runs on",
                      "--still-gpu)"});
  item("--still-user-qm", {"Carry a real user QM OBU (identity matrix; the",
                           "pinned dav2d rejects QM OBUs; CPU-only)"});
  item("--still-ist", {"Intra secondary transform, luma (CPU-only)"});
  item("--still-tcq", {"Trellis-coded quantisation, luma 2-D transforms",
                       "(parity hiding is inferred off in a single picture)"});
  item("--still-cctx", {"Cross-component chroma transform (cctx_type",
                        "rotation)"});
  item("--still-tx-partition", {"TX_MODE_SELECT luma transform partitions",
                                "(NONE/SPLIT/HORZ/VERT)"});
  item("--still-idtx", {"Identity intra transform (enable_idtx_intra +",
                        "fsc_mode)"});
  item("--still-ddt", {"Rejected: DDT/DDTX/FDDT is inter-only"});
  item("--still-dc-ac-delta-q", {"DC/AC delta-Q frame search (CPU; runs on",
                                 "--still-gpu)"});
  item("--still-content-interpretation",
       {"Emit a default content interpretation OBU (CPU-only)"});
  item("--metadata-timecode", {"Emit a neutral metadata_timecode OBU (suffix;",
                               "ignored by the pinned decoder)"});
  item("--metadata-scan-type", {"Emit a neutral metadata_scan_type OBU (suffix;",
                                "ignored by the pinned decoder)"});
  item("--still-palette", {"Luma palette intra (screen content; sets",
                           "allow_screen_content_tools; CPU-only)"});
  item("--still-ibc", {"Intra block copy (screen content; exact copies,",
                       "skip_flag=1, force_integer_mv; CPU-only; needs a frame",
                       "at least 129 luma samples wide to find a source)"});
  item("--still-ibc-search <N>", {"IBC local BV search step in luma samples,",
                                  "1..64 (default 8)"});
  item("--tiles <N|CxR>", {"Uniform tile grid, default 1x1: a count",
                           "2/4/8/16/32 or a grid 2x1/2x2/4x2/8x2/16x2.",
                           "Requires --no-deblock; 8-bit 4:2:0 only"});
  item("--bit-depth <8|10>", {"Sample depth (default 8). 10 requires --still, a",
                              "C420p10/C422p10/C444p10/Cmono10 input and",
                              "base_q_idx 1..303; 12-bit is not signalable in",
                              "AV2 v1.0.0 (CPU-only)"});
  item("--chroma <fmt>", {"420 (default), 422, 444 or 400. The input must",
                          "already have that exact format; 8-bit pins --qp 90,",
                          "10-bit takes base_q_idx 1..303 (CPU-only)"});
  item("--lossless", {"Q=0 lossless still picture with its own writer;",
                      "cannot be combined with --still"});

  out << "\nEffort presets (--still only; opt-in, default off):\n";
  item("--effort <N>", {"Ranked-feature budget, 10..10000 (log scale); enables",
                        "measured features in score order and never exceeds",
                        "N. Implies --still. Default off"});
  item("--cpu-used <0..8>", {"x264-style sugar for --effort: 8 -> 10,",
                             "0 -> 10000, geometric in between"});
  item("--effort-plan", {"Print the effort selection and exit without",
                         "encoding"});

  out << "\nVideo options (--video):\n";
  item("--video", {"Encode a multi-frame P-frame sequence"});
  item("--video-gpu", {"Opt-in CUDA backend for --video (implies --video);",
                       "same decisions/bytes as the CPU reference, no silent",
                       "CPU fallback. CUDA build only"});
  item("--video-timing", {"Signal decoder_model_info and a CI OBU with",
                          "timing_info() from the Y4M frame rate"});
  item("--video-mfh", {"Emit a multi-frame header (§5.7) and reference it",
                       "from every frame header"});
  item("--p-frames <N>", {"Number of P frames (default: all frames after the",
                          "keyframe)"});
  item("--b-frames <N>", {"B frames in one closed IPBB GOP, 0..6 (default 0)"});
  item("--video-tlayers <N>", {"Hierarchical P GOP with N = 4 or 8 temporal",
                              "layers; coding order differs from display order"});
  item("--video-tlayer-drop <L>", {"Encode only temporal layers <= L (dropped-",
                                  "layer substream); implies --video-tlayers"});
  item("--video-compound", {"Compound GLOBAL_GLOBALMV/AVERAGE prediction;",
                            "requires --video-refs >= 2"});
  item("--video-refs <N>", {"References per P frame (NumTotalRefs), 1..7",
                            "(default 1); 2..7 selects per block"});
  item("--video-num-ref-frames <N>", {"§5.4.6 NumRefFrames, 4..8 (default: 2",
                                      "for one reference, 8 for multi-ref)"});
  item("--video-subpel-eighth", {"Add the 1/8-pel ME refinement stage"});
  item("--video-mv-precision P", {"eighth (default), quarter or half frame MV",
                                  "precision"});
  item("--video-inter-residual", {"Code a quantised inter residual"});
  item("--video-tx-search", {"Opt-in §5.18.8 TX_MODE_SELECT luma TX size/type/",
                                "partition search (changes the bitstream)"});
  item("--video-tx-lambda <S>", {"SSE-per-bit weight for --video-tx-search",
                                  "(default 256)"});
  item("--video-deblock", {"Deblock every reconstructed frame (§7.17) before",
                           "it is emitted or stored as a reference; CPU and",
                           "CUDA paths signal §5.18.7.9 and filter pixels"});
  item("--video-gme", {"Estimate and signal §5.18.9 global motion per",
                       "reference; GLOBALMV uses the projected §7.12.2.2 MV"});
  item("--video-warp", {"Predict GLOBALMV blocks with the §7.13.3.19/.20",
                        "block/extended warp when a global model is accepted",
                        "(implies --video-gme)"});
  item("--video-cdef", {"Enable §7.18 CDEF on the reconstructed frame before",
                        "it becomes the output/reference (single strength set)"});
  item("--video-cdef-strength <N>", {"CdefStrengths 1..8 (default 1: no tile",
                                     "symbol); 2..8 codes a per-64x64 index"});
  item("--video-cdef-damping <N>", {"CdefDamping 3..6 (default 3)"});
  item("--video-ccso", {"Enable §7.19 CCSO after CDEF (band-offset only)"});
  item("--video-ccso-bands <N>", {"CcsoMaxBandLog2 0..6 (default 4)"});
  item("--video-wiener", {"Enable §7.20 luma non-separable Wiener restoration"});
  item("--video-chroma-lr", {"Enable §7.20 chroma loop restoration (U/V)"});
  item("--video-gdf", {"Enable §7.20.5 GDF (luma, header-only mode)"});
  item("--video-intra-full", {"Opt-in 61-candidate luma intra search (DC, V, H,",
                              "8 directional modes with angle deltas, SMOOTH,",
                              "SMOOTH_V/H, PAETH); chroma stays DC. Default off",
                              "so the shipped streams stay byte-identical"});
  item("--video-long-term", {"Signal long-term reference ids; tag the key frame",
                             "as LongTermId 0 (at most 16 coded frames)"});
  item("--video-olk", {"Open random access GOP (CLK, open-loop key, P2, P3;",
                       "needs exactly 4 input frames)"});
  item("--video-leading <N>", {"N open-GOP leading pictures, 1..7 (needs at",
                               "least N+2 input frames)"});
  item("--video-sef", {"Re-show the last picture with a show-existing-frame",
                       "OBU (the last picture is coded hidden)"});
  item("--video-switch", {"CLK + OBU_SWITCH (exactly 2 input frames)"});
  item("--video-ras", {"CLK + OBU_SWITCH + OBU_RAS_FRAME with a long-term",
                       "reference (exactly 3 input frames)"});
  item("--video-implicit-ref-map", {"Rejected: §7.8 past/future lists need an",
                                    "implicit map; the M2 writer emits",
                                    "§5.18.2 explicit maps only"});
  item("--video-drl-reorder", {"Rejected: not decoder-gated yet",
                               "(docs/negative-results.md)"});
  item("--video-drl-reorder-always", {"DRL_REORDER_ALWAYS; implies the reorder",
                                      "flag and is rejected for the same reason"});
  item("--rc <mode>", {"Per-frame rate control: cqp|crf|vbr (default cqp, the",
                       "fixed --qp stream)"});
  item("--crf <V>", {"--rc crf quality 0..63 (default 32); owns base_q_idx",
                     "(drop --qp, which only anchors the VBR pass)"});
  item("--bitrate <kbps>", {"--rc vbr target bitrate; VBR runs a statistics",
                            "pass at --qp, then plans one base_q_idx per frame"});
  item("--svc-layers <spec>", {"SVC: spatial layers as x0=WxH[,x1=WxH...]"});
  item("--svc-lcr-id <N>", {"SVC: seq_lcr_id (1..7)"});
  item("--svc-temporal <N>", {"SVC: temporal layers (1 is wired)"});
  item("--svc-embedded <N>", {"SVC: embedded layers (1 is wired)"});
  item("--oppoint <x,y>", {"SVC: emit/select one operating point"});
  item("--alllayers", {"SVC: emit all operating points"});
  item("--spatial-layers <N>", {"SVC: limit the emitted spatial layers"});

  out << "\nDebug options (test-only; not encoding-quality options):\n";
  item("--still-single", {"Force one coding block per superblock"});
  item("--still-dc", {"Force DC-only intra"});
  item("--still-depth <D>", {"Partition depth cap [0, 8]"});
  item("--still-rdo-scale S", {"lambda sse_scale for --still-rdo-intra (0.125)"});
  item("--still-rdoq-scale S", {"lambda sse_scale for --still-rdoq (0.125)"});
  item("--still-tx-type-scale S", {"lambda sse_scale for --still-tx-type (0.125)"});
  item("--still-ist-scale S", {"lambda sse_scale for --still-ist (0.125)"});
  item("--still-tcq-scale S", {"lambda sse_scale for --still-tcq (0.125)"});
  item("--still-cdef-strengths N", {"CDEF strength sets 1..8 (default 4);",
                                    "requires --still-cdef"});
  item("--still-cdef-on-skip-txfm", {"Code the ADAPTIVE frame bit for CDEF on",
                                     "skip-txfm; requires --still-cdef"});
  item("--still-ccso-bands N", {"CCSO max_band_log2 0..6 (default 6 = 64 bands);",
                                "requires --still-ccso"});
  item("--still-delta-q-res N", {"delta_q_res 0..3 (default 0); requires",
                                 "--still-delta-q"});
  item("--still-seg-delta N", {"ALT_Q delta for --still-per-segment-q, 0..64",
                               "(default 8)"});
  item("--still-qm-level L", {"QM default matrix level 0..14 (default 3);",
                              "requires --still-qm"});
  item("--lossless-fsc", {"Force the forward-skip-coding coefficient path"});
  item("--gpu-plan <N>", {"Print the host-only multi-GPU tile plan for N GPUs",
                          "(1..8) and exit: assignment, transfer and VRAM",
                          "estimates; no device memory, no stream. Reads the",
                          "input and --tiles/--vram-cap only"});
  item("--recon <file>", {"Dump the encoder reconstruction as Y4M (format paths",
                          "and multi-tile --tiles stills; the decoder gates",
                          "compare against it)"});
  item("--video-recon <file>", {"Video: write the encoder reconstruction Y4M"});
  item("--video-force-single", {"Video: one 64x64 leaf per superblock"});
  item("--video-min-block <N>",
       {"Video: minimum partition block size, 8..64 (default 16); sub-8 leaves",
        "need shared-chroma prediction, which this path does not implement"});
  item("--video-max-depth <D>", {"Video: partition depth cap [0, 8] (default 4)"});
  item("--video-me-range <R>", {"Video: integer ME search radius (default 16)"});
  item("--video-trace", {"Video: trace block decisions to stderr"});
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

// Expansion of the measured feature ranking into existing CLI options.  Called
// after option parsing and before validation, so the selected flags obey the
// same rejection rules as flags typed by the user.  Only --still (the measured
// CPU reference path) is calibrated; no GPU/video row exists yet.
bool apply_effort_ranking(Options& options, std::string& error) {
  const auto reject = [&](std::string message) {
    error = std::move(message);
    return false;
  };
  if (options.video || options.lossless || options.still_gpu ||
      options.gpu_plan != 0) {
    return reject("--effort/--cpu-used is calibrated for --still only; there "
                  "is no measured video, lossless, GPU or gpu-plan row yet");
  }
  if (options.tiles != "1x1") {
    return reject("--effort/--cpu-used cannot be combined with --tiles: the "
                  "tiled writer pins its tool set");
  }
  if (options.bit_depth != 8 || options.chroma != "420") {
    return reject("--effort/--cpu-used is calibrated on 8-bit 4:2:0 stills; "
                  "the --bit-depth/--chroma format paths pin their tool set");
  }

  const int budget_sources = (options.effort_given ? 1 : 0) +
                             (options.cpu_used_given ? 1 : 0);
  if (budget_sources == 0) {
    return reject("--effort-plan requires --effort or --cpu-used");
  }
  if (budget_sources > 1) {
    return reject("pass only one of --effort or --cpu-used");
  }

  double budget = 0.0;
  std::string source;
  if (options.effort_given) {
    budget = static_cast<double>(options.effort_budget);
    source = "--effort " + std::to_string(options.effort_budget);
  } else {
    budget = av2enc::effort::budget_for_cpu_used(options.cpu_used);
    source = "--cpu-used " + std::to_string(options.cpu_used);
  }

  av2enc::effort::Constraints constraints;
  constraints.cfl = options.still_cfl;
  constraints.sb128 = options.still_sb128;
  constraints.sb256 = options.still_sb256;
  constraints.qm = options.still_qm;
  constraints.dc_ac_delta_q = options.still_dc_ac_delta_q;
  constraints.still_single = options.still_single;
  constraints.mrl = options.still_mrl;
  constraints.wiener = options.still_wiener;
  constraints.chroma_lr = options.still_chroma_lr;
  constraints.pc_wiener = options.still_pc_wiener;
  constraints.gdf = options.still_gdf;

  av2enc::effort::Selection selection;
  if (!av2enc::effort::select(budget, av2enc::effort::Mode::kStill,
                              options.specified, constraints, selection,
                              error)) {
    return false;
  }
  selection.budget_source = source;

  struct FlagMap {
    std::string_view flag;
    bool Options::*member;
  };
  static constexpr FlagMap kFlagMap[] = {
      {"--still-tx-type", &Options::still_tx_type},
      {"--still-ist", &Options::still_ist},
      {"--still-dip", &Options::still_dip},
      {"--still-rdoq", &Options::still_rdoq},
      {"--still-uv-full", &Options::still_uv_full},
      {"--still-cdef", &Options::still_cdef},
      {"--still-ccso", &Options::still_ccso},
      {"--still-mhccp", &Options::still_mhccp},
      {"--still-ext-partitions", &Options::still_ext_partitions},
      {"--still-ibp", &Options::still_ibp},
      {"--still-pc-wiener", &Options::still_pc_wiener},
      {"--still-rdo-intra", &Options::still_rdo_intra},
      {"--still-full-intra", &Options::still_full_intra},
      {"--still-mrl", &Options::still_mrl},
  };
  for (const std::string& flag : selection.flags) {
    bool found = false;
    for (const FlagMap& map : kFlagMap) {
      if (flag == map.flag) {
        options.*(map.member) = true;
        found = true;
        break;
      }
    }
    if (!found) {
      return reject("effort table selected '" + flag +
                    "', which this CLI does not implement");
    }
  }
  options.m1 = true;  // the measured ranking is a Still-path preset
  options.effort_active = true;
  options.effort_report = av2enc::effort::format_selection(selection);
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

    if (argument == "--video-timing") {
      options.video_timing = true;
      continue;
    }

    if (argument == "--video-mfh") {
      options.video_mfh = true;
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

    if (argument == "--video-compound") {
      options.video_compound = true;
      continue;
    }

    if (argument == "--video-inter-residual") {
      options.video_inter_residual = true;
      continue;
    }

    if (argument == "--video-tx-search") {
      options.video_tx_search = true;
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

    if (argument == "--video-gme") {
      options.video_gme = true;
      continue;
    }

    if (argument == "--video-warp") {
      options.video_warp = true;
      options.video_gme = true;
      continue;
    }

    if (argument == "--video-cdef") {
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

    if (argument == "--video-intra-full") {
      options.video_intra_full = true;
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
      if (!parse_int(argv[++i], value) || value < 8 || value > 64) {
        error = "--video-min-block must be an integer in [8, 64]: the video "
                "path has no shared-chroma prediction for sub-8 luma leaves";
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

  if (options.effort_given || options.cpu_used_given || options.effort_plan) {
    if (!apply_effort_ranking(options, error)) {
      return ParseStatus::kError;
    }
  }

  if (options.input.empty() && !options.effort_plan) {
    error = "no input file specified";
    return ParseStatus::kError;
  }
  if (options.output.empty() && options.gpu_plan == 0 &&
      !options.effort_plan) {
    error = "no output file specified (use -o <output.ivf>)";
    return ParseStatus::kError;
  }
  const auto reject = [&](std::string message) {
    error = std::move(message);
    return ParseStatus::kError;
  };

  if (options.gpu_plan != 0) {
    if (options.video || options.m1 || options.still_gpu || options.lossless) {
      error = "--gpu-plan is a host-planning dry run; do not combine it with "
              "--video, --still, --still-gpu or --lossless";
      return ParseStatus::kError;
    }
    // The plan reads the input header, the tile layout and the optional VRAM
    // cap. Every other flag would be accepted without an effect, so name it
    // instead of ignoring it.
    for (const std::string_view flag : options.specified) {
      if (flag == "--gpu-plan" || flag == "--tiles" || flag == "--vram-cap" ||
          flag == "-o" || flag == "--output") {
        continue;
      }
      return reject("--gpu-plan is a host-only dry run; " + std::string(flag) +
                    " has no effect on the plan");
    }
  }
  if (options.video) {
    if (options.m1 || options.still_gpu || options.lossless) {
      error = "--video cannot be combined with --still, --still-gpu or "
              "--lossless";
      return ParseStatus::kError;
    }
    if (options.adaptive_cdf && !options.avg_cdf) {
      error = "--adaptive-cdf on the video path needs --avg-cdf: the "
              "average-CDF bank owns the §8.2.2 tile seed and §7.5 frame-end "
              "bookkeeping";
      return ParseStatus::kError;
    }
    if (options.cdf_continuity &&
        !(options.avg_cdf && options.adaptive_cdf)) {
      error = "--cdf-continuity requires --avg-cdf and --adaptive-cdf";
      return ParseStatus::kError;
    }
    // The encoder side is wired (seed export + §7.4.4 reference load + §7.5
    // frame end), but pinned dav2d f9dafbc rejects a frame whose
    // disable_cross_frame_cdf_init is 0: "Error parsing frame header / Error
    // parsing OBU data".  Reproducer (documented in docs/negative-results.md):
    //   av2enc in.y4m -o out.ivf --video --p-frames 2 --avg-cdf
    //       --adaptive-cdf --cdf-continuity
    // Set the env escape below only for encoder-side experiments; a conforming
    // decoder is required to gate it.
    if ((options.video_timing || options.video_mfh) &&
        (options.video_long_term || options.video_olk ||
         options.video_leading > 0 || options.video_sef ||
         options.video_switch || options.video_ras)) {
      error = "--video-timing/--video-mfh are wired into the plain --video "
              "sequence header only; the frame-type modes (long-term/OLK/"
              "leading/SEF/SWITCH/RAS) build their own headers";
      return ParseStatus::kError;
    }
    if (options.cdf_continuity &&
        std::getenv("AV2ENC_ENABLE_UNGATED_CDF_CONTINUITY") == nullptr) {
      error = "--cdf-continuity is encoder-wired but not decoder-gated: "
              "pinned dav2d f9dafbc rejects disable_cross_frame_cdf_init = 0 "
              "with 'Error parsing frame header' (docs/negative-results.md). "
              "Set AV2ENC_ENABLE_UNGATED_CDF_CONTINUITY=1 only to reproduce";
      return ParseStatus::kError;
    }
    if (options.bit_depth != 8 || options.chroma != "420") {
      error = "--video is 8-bit 4:2:0; --bit-depth/--chroma are still-only";
      return ParseStatus::kError;
    }
    // Rate-control combination errors are usage errors: name them here so the
    // caller exits 2 before the output file is opened (and therefore never
    // truncates an existing -o stream).
    if (options.rc_given && options.rc_mode == "vbr" &&
        !options.rc_bitrate_given) {
      error = "--rc vbr requires --bitrate <kbps>";
      return ParseStatus::kError;
    }
    if (options.rc_crf_given && options.rc_mode != "crf") {
      error = "--crf requires --rc crf";
      return ParseStatus::kError;
    }
    if (options.rc_bitrate_given && options.rc_mode != "vbr") {
      error = "--bitrate requires --rc vbr";
      return ParseStatus::kError;
    }
    if (options.rc_mode != "cqp" &&
        (options.video_olk || options.video_leading != 0 ||
         options.video_ras)) {
      error = "--rc " + options.rc_mode +
              " does not compose with --video-olk, --video-leading or "
              "--video-ras yet: those modes split the sequence into short "
              "sub-encodes whose per-frame QP count differs from the input";
      return ParseStatus::kError;
    }
    if (options.rc_mode != "cqp" &&
        (options.avg_cdf || options.adaptive_cdf ||
         options.cdf_continuity)) {
      error = "--rc " + options.rc_mode +
              " needs a fixed base_q_idx; --avg-cdf/--adaptive-cdf/"
              "--cdf-continuity are not compatible with per-frame QP";
      return ParseStatus::kError;
    }
    // CRF derives base_q_idx from --crf only (qcomp is 0, so the quantiser is
    // constant across frames).  An explicit --qp would be accepted and ignored,
    // so reject it by name; --qp remains the VBR statistics-pass anchor.
    if (options.rc_mode == "crf" && flag_given(options, "--qp")) {
      error = "--rc crf owns the quantizer; drop --qp or use --rc cqp/--rc "
              "vbr (where --qp is the statistics-pass anchor)";
      return ParseStatus::kError;
    }
    if (options.video_drl_reorder) {
      error = "--video-drl-reorder is not decoder-gated yet: the pinned "
              "decoder disagrees with the encoder MV-stack order on some "
              "sub-64x64 blocks (docs/negative-results.md)";
      return ParseStatus::kError;
    }
    if (options.b_frames != 0 && options.b_frames > 6) {
      error = "--b-frames supports 1..6 B frames per GOP";
      return ParseStatus::kError;
    }
    if (options.video_implicit_ref_map) {
      // §7.8 derives the past/future/current lists only for an implicit
      // reference map (explicit_ref_frame_map == 0).  The product encoder
      // writes §5.18.2 explicit ref_frame_idx maps on every path, and the
      // pinned decoder has not been gated on the implicit form
      // (docs/negative-results.md), so the carrier is rejected by name.
      error = "--video-implicit-ref-map is not decoder-gated: the §7.8 "
              "past/future reference lists need explicit_ref_frame_map = 0, "
              "which the M2 writer does not emit; --video and --video-gpu "
              "both reject it";
      return ParseStatus::kError;
    }
    if (options.video_num_ref_frames != 0 &&
        (options.b_frames != 0 || options.video_tlayers != 0 ||
         options.video_long_term || options.video_olk ||
         options.video_leading != 0 || options.video_sef ||
         options.video_switch || options.video_ras)) {
      error = "--video-num-ref-frames applies to the flat P chain only; the "
              "B-frame / temporal-layer / frame-type modes own NumRefFrames";
      return ParseStatus::kError;
    }
    if (options.video_num_ref_frames != 0 &&
        options.video_refs > options.video_num_ref_frames) {
      error = "--video-refs " + std::to_string(options.video_refs) +
              " exceeds --video-num-ref-frames " +
              std::to_string(options.video_num_ref_frames);
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
    if (options.video_deblock && frame_modes != 0) {
      error = "--video-deblock does not compose with --video-long-term/--video-"
              "olk, --video-leading, --video-sef, --video-switch or "
              "--video-ras: those header writers do not signal "
              "deblocking_filter_params()";
      return ParseStatus::kError;
    }
    if (options.video_deblock && options.tiles != "1x1") {
      error = "--video-deblock does not support --tiles > 1x1: the pinned "
              "decoder filters across interior tile edges, so the CPU "
              "reference cannot match it";
      return ParseStatus::kError;
    }
    if (options.video_tx_search && options.tiles != "1x1") {
      error = "--video-tx-search does not support --tiles > 1x1: the tile "
              "writer owns a single tx_mode_select header bit per frame";
      return ParseStatus::kError;
    }
    if (flag_given(options, "--video-tx-lambda") && !options.video_tx_search) {
      error = "--video-tx-lambda requires --video-tx-search";
      return ParseStatus::kError;
    }
    if (options.video_tx_search && options.video_deblock) {
      error = "--video-tx-search does not compose with --video-deblock yet: "
              "the §7.17 grid derives its transform edges from one transform "
              "per leaf and does not model §5.20.6.3 parts";
      return ParseStatus::kError;
    }
    if (options.video_tx_search && (options.avg_cdf || options.adaptive_cdf)) {
      error = "--video-tx-search does not compose with --avg-cdf/--adaptive-"
              "cdf: the §8.2.4 bank does not carry the §5.20.6.3 TX-partition "
              "CDFs";
      return ParseStatus::kError;
    }
    if (options.video_tx_search && frame_modes != 0) {
      error = "--video-tx-search does not compose with the frame-type wrapper "
              "modes (their header writers do not signal tx_mode_select)";
      return ParseStatus::kError;
    }
    if (options.video_tx_search && options.svc_given) {
      error = "--video-tx-search does not compose with --svc-layers";
      return ParseStatus::kError;
    }
    if ((options.video_switch || options.video_ras) &&
        (options.video_refs != 1 || options.video_mv_precision != 0)) {
      error = "--video-switch/--video-ras need --video-refs 1 and eighth-pel "
              "MV precision";
      return ParseStatus::kError;
    }
    if (options.video_gme) {
      // The gated global-motion path is the plain P chain: one model per
      // explicit reference, GLOBALMV projection/warp, no SWITCH/RAS special
      // header form, no per-frame B/temporal plan and no uniform tile grid.
      if (options.b_frames != 0 || options.video_tlayers != 0 ||
          frame_modes != 0 || options.video_compound ||
          options.video_inter_residual || options.video_drl_reorder ||
          options.video_refs > 1) {
        error = "--video-gme/--video-warp are gated on the plain skip-only P "
                "chain: they cannot be combined with --b-frames, "
                "--video-tlayers, a frame-type wrapper, --video-compound, "
                "--video-inter-residual, --video-refs > 1 or "
                "--video-drl-reorder";
        return ParseStatus::kError;
      }
      if (options.tiles != "1x1") {
        error = "--video-gme/--video-warp are gated on a single tile";
        return ParseStatus::kError;
      }
    }
    if (options.video_compound) {
      if (options.b_frames == 0 && options.video_refs != 2) {
        error = "--video-compound needs --video-refs 2 (the (0,1) pair is "
                "implied by NumTotalRefs == 2)";
        return ParseStatus::kError;
      }
      if (options.video_drl_reorder || options.video_long_term ||
          options.video_olk || options.video_leading != 0 || options.video_sef ||
          options.video_switch || options.video_ras ||
          options.video_tlayers != 0) {
        error = "--video-compound cannot be combined with the frame-type or "
                "DRL-reorder modes";
        return ParseStatus::kError;
      }
    }
    if (options.b_frames != 0) {
      if (options.video_tlayers != 0 || frame_modes != 0) {
        error = "--b-frames cannot be combined with the temporal-layer or "
                "frame-type modes";
        return ParseStatus::kError;
      }
      if (options.video_refs != 1) {
        error = "--b-frames owns the per-frame reference list; keep "
                "--video-refs 1";
        return ParseStatus::kError;
      }
      if (options.rc_mode != "cqp") {
        error = "--b-frames does not compose with --rc crf/vbr";
        return ParseStatus::kError;
      }
      if (options.video_drl_reorder) {
        error = "--b-frames cannot be combined with --video-drl-reorder";
        return ParseStatus::kError;
      }
    }
    if (options.video_tlayers != 0) {
      if (options.video_tlayers != 4 && options.video_tlayers != 8) {
        error = "--video-tlayers supports 4 or 8 frames (one closed GOP)";
        return ParseStatus::kError;
      }
      if (frame_modes != 0 || options.video_leading != 0) {
        error = "--video-tlayers cannot be combined with the frame-type modes";
        return ParseStatus::kError;
      }
      if (options.video_refs != 1 || options.video_mv_precision != 0) {
        error = "--video-tlayers needs --video-refs 1 and eighth-pel MV "
                "precision";
        return ParseStatus::kError;
      }
      if (options.rc_mode != "cqp") {
        error = "--video-tlayers does not compose with --rc crf/vbr";
        return ParseStatus::kError;
      }
      if (options.video_tlayer_drop > 3) {
        error = "--video-tlayer-drop is at most 3 (obsolete: max_tlayer_id "
                "is f(2))";
        return ParseStatus::kError;
      }
      // The hierarchical-P plan stops at max_tlayer_id = 1 for N = 4 and 2
      // for N = 8. Reject a higher drop here instead of failing mid-encode.
      const int plan_max_tlayer = options.video_tlayers == 4 ? 1 : 2;
      if (options.video_tlayer_drop > plan_max_tlayer) {
        error = "--video-tlayer-drop " +
                std::to_string(options.video_tlayer_drop) + " is above the " +
                std::to_string(options.video_tlayers) +
                "-frame plan's max_tlayer_id " +
                std::to_string(plan_max_tlayer);
        return ParseStatus::kError;
      }
    } else if (options.video_tlayer_drop >= 0) {
      error = "--video-tlayer-drop requires --video-tlayers";
      return ParseStatus::kError;
    }

    if (options.video_gpu) {
      // The CUDA M2 backend mirrors the flat one-reference P chain.  Every
      // option that builds a different ref/header model is rejected by name
      // instead of being accepted and silently ignored (or falling back to the
      // CPU encoder).  --video without this flag keeps the full CPU surface.
#if !AV2ENC_VIDEO_GPU_CALLABLE
#if AV2ENC_HAVE_CUDA
      error = "--video-gpu is not available in this build: the M2 video GPU "
              "entry point (src/cuda/pipeline/video/) is not linked. Use "
              "--video for the CPU reference path.";
#else
      error = "--video-gpu requires a CUDA build; this is a CPU-only reference "
              "build. Use --video for the CPU reference path.";
#endif
      return ParseStatus::kError;
#else
      const auto reject_gpu = [&](const char* flag) {
        error = std::string("--video-gpu does not implement ") + flag +
                " yet; use --video for the CPU reference path";
        return ParseStatus::kError;
      };
      if (options.cdf_continuity) {
        return reject_gpu("--cdf-continuity (cross-frame CDF carry)");
      }
      if (options.svc_given) return reject_gpu("--svc-layers");
#endif
    }
  } else {
    if (flag_given(options, "--p-frames") ||
        flag_given(options, "--b-frames") ||
        flag_given(options, "--video-refs") ||
        flag_given(options, "--video-num-ref-frames") ||
        flag_given(options, "--video-implicit-ref-map") ||
        flag_given(options, "--video-subpel-eighth") ||
        flag_given(options, "--video-mv-precision") ||
        flag_given(options, "--video-inter-residual") ||
        flag_given(options, "--video-tx-search") ||
        flag_given(options, "--video-tx-lambda") ||
        flag_given(options, "--video-deblock") ||
        flag_given(options, "--video-gme") ||
        flag_given(options, "--video-warp") ||
        flag_given(options, "--video-intra-full") ||
        flag_given(options, "--video-tlayers") ||
        flag_given(options, "--video-tlayer-drop") ||
        flag_given(options, "--video-compound") ||
        flag_given(options, "--video-drl-reorder") ||
        flag_given(options, "--video-drl-reorder-always") ||
        flag_given(options, "--video-long-term") ||
        flag_given(options, "--video-olk") ||
        flag_given(options, "--video-leading") ||
        flag_given(options, "--video-sef") ||
        flag_given(options, "--video-switch") ||
        flag_given(options, "--video-ras") ||
        flag_given(options, "--video-force-single") ||
        flag_given(options, "--video-min-block") ||
        flag_given(options, "--video-max-depth") ||
        flag_given(options, "--video-me-range") ||
        flag_given(options, "--video-cdef") ||
        flag_given(options, "--video-cdef-strength") ||
        flag_given(options, "--video-cdef-damping") ||
        flag_given(options, "--video-ccso") ||
        flag_given(options, "--video-ccso-bands") ||
        flag_given(options, "--video-wiener") ||
        flag_given(options, "--video-chroma-lr") ||
        flag_given(options, "--video-gdf") ||
        flag_given(options, "--video-trace") ||
        flag_given(options, "--video-recon")) {
      error = "video options require --video";
      return ParseStatus::kError;
    }
    if (options.rc_given) {
      error = "--rc is wired for --video only; the still paths keep the fixed "
              "--qp stream";
      return ParseStatus::kError;
    }
    if (options.rc_crf_given) {
      error = "--crf requires --video --rc crf";
      return ParseStatus::kError;
    }
    if (options.rc_bitrate_given) {
      error = "--bitrate requires --video --rc vbr";
      return ParseStatus::kError;
    }
  }
  if (options.lossless) {
    if (options.m1 || options.still_gpu) {
      return reject("--lossless has its own still-picture writer; do not "
                    "combine it with --still or --still-gpu");
    }
    for (const std::string_view flag : options.specified) {
      if ((flag.starts_with("--still-") && flag != "--still-gpu") ||
          flag == "--no-cfl" || flag == "--no-deblock") {
        return reject(std::string(flag) +
                      " is not applied by the --lossless writer");
      }
    }
  }
  // Every other --still tool is a modifier of the Still paths. Outside those
  // paths it would be accepted and ignored, so it is a usage error by name.
  if (!options.m1 && !options.still_gpu) {
    for (const std::string_view flag : options.specified) {
      // Multi-tile stills report their own, more specific mode error below.
      if (flag == "--no-deblock" && options.tiles != "1x1") continue;
      if ((flag.starts_with("--still-") && flag != "--still-gpu") ||
          flag == "--no-cfl" || flag == "--no-deblock") {
        return reject(std::string(flag) + " requires --still or --still-gpu");
      }
    }
  }
  if (!options.video && options.max_frames > 1) {
    return reject(
        "--frames > 1 requires --video; the still modes encode one frame");
  }
  if (options.video && options.qp == 0) {
    return reject("--video does not support lossless QP 0");
  }
  if (!options.video && options.qp == 0 && !options.lossless) {
    return reject("--qp 0 requires --lossless");
  }
  if (options.lossless && flag_given(options, "--qp") && options.qp != 0) {
    return reject("--lossless is Q=0; drop --qp or pass --qp 0");
  }
  // The CPU Still path selects the 8.3.1 coefficient-CDF qctx slice from the
  // raw base_q_idx and derives the quantiser from qlookup(base_q_idx), so QP
  // 1..255 is supported there. The 10-bit format path supports 1..303; 8-bit
  // 4:2:2/4:4:4/4:0:0 pins 90. Basic and the 8-bit 4:2:0 Still GPU path pin 90.
  const bool format_still =
      options.bit_depth != 8 || options.chroma != "420";
  const bool still_cpu_qp =
      (options.m1 && !options.still_gpu) ||
      (options.still_gpu && format_still);
  if (!options.lossless && !options.video && !still_cpu_qp &&
      options.qp != 90) {
    return reject(options.still_gpu
                      ? "--still-gpu pins --qp 90; use --still for QP 1..255"
                      : "Basic pins --qp 90; use --still for QP 1..255");
  }
  // Debug tuning knobs only modify the tool that consumes them.
  if (flag_given(options, "--still-rdo-scale") && !options.still_rdo_intra) {
    return reject("--still-rdo-scale requires --still-rdo-intra");
  }
  if (flag_given(options, "--still-rdoq-scale") && !options.still_rdoq) {
    return reject("--still-rdoq-scale requires --still-rdoq");
  }
  if (flag_given(options, "--still-tx-type-scale") && !options.still_tx_type) {
    return reject("--still-tx-type-scale requires --still-tx-type");
  }
  if (flag_given(options, "--still-ist-scale") && !options.still_ist) {
    return reject("--still-ist-scale requires --still-ist");
  }
  if (flag_given(options, "--still-tcq-scale") && !options.still_tcq) {
    return reject("--still-tcq-scale requires --still-tcq");
  }
  if (flag_given(options, "--still-cdef-strengths") && !options.still_cdef) {
    return reject("--still-cdef-strengths requires --still-cdef");
  }
  if (flag_given(options, "--still-ccso-bands") && !options.still_ccso) {
    return reject("--still-ccso-bands requires --still-ccso");
  }
  if (flag_given(options, "--still-delta-q-res") && !options.still_delta_q) {
    return reject("--still-delta-q-res requires --still-delta-q");
  }
  if (flag_given(options, "--still-qm-level") && !options.still_qm) {
    return reject("--still-qm-level requires --still-qm");
  }
  if (options.lossless_force_fsc && !options.lossless) {
    return reject("--lossless-fsc requires --lossless");
  }
  // The CPU Still writer has its own tool-combination rules. Reject them at
  // the CLI so they exit 2 with the flag names instead of surfacing as an
  // encode failure after the output file has been opened.
  if (options.m1 && !options.still_gpu) {
    if (options.still_qm && options.still_rdoq) {
      return reject("--still-qm is not supported with --still-rdoq yet");
    }
    if (options.still_dc_ac_delta_q && options.still_qm) {
      return reject("--still-dc-ac-delta-q is not supported with --still-qm yet");
    }
    if (options.still_dc_ac_delta_q && options.still_rdoq) {
      return reject(
          "--still-dc-ac-delta-q is not supported with --still-rdoq yet");
    }
    if (options.still_mrl && options.still_ibp) {
      return reject("--still-mrl and --still-ibp cannot be combined");
    }
    if (options.still_mrl && options.still_rdo_intra) {
      return reject("--still-mrl and --still-rdo-intra cannot be combined");
    }
    if (options.still_mhccp && !options.still_cfl) {
      return reject("--still-mhccp requires CfL (do not combine with --no-cfl)");
    }
    // Only the luma restoration type is exclusive: PC-Wiener and Wiener NS
    // both own plane 0. GDF and chroma LR now compose with either.
    if (options.still_pc_wiener && options.still_wiener) {
      return reject("--still-pc-wiener cannot be combined with --still-wiener");
    }
    // The remaining pairs are the CPU Still writer's own rules
    // (src/pipeline/still/still_encoder.cpp). Mirror them here so they exit 2
    // before IvfWriter::open instead of emitting a header-only IVF.
    if (options.still_qm && options.still_tcq) {
      return reject("--still-qm is not supported with --still-tcq yet");
    }
    if (options.still_qm && options.still_cctx) {
      return reject("--still-qm is not supported with --still-cctx yet");
    }
    if (options.still_qm && options.still_tx_partition) {
      return reject(
          "--still-qm is not supported with --still-tx-partition yet");
    }
    if (options.still_qm && options.still_per_segment_q) {
      return reject(
          "--still-qm is not supported with --still-per-segment-q yet");
    }
    if (options.still_cctx && options.still_dc_ac_delta_q) {
      return reject(
          "--still-cctx is not supported with --still-dc-ac-delta-q yet");
    }
    if (options.still_tx_partition && options.still_dc_ac_delta_q) {
      return reject("--still-tx-partition is not supported with "
                    "--still-dc-ac-delta-q yet");
    }
    if (options.still_tx_partition && options.still_dip) {
      return reject(
          "--still-tx-partition is not supported with --still-dip yet");
    }
    if (options.still_tx_partition && options.still_mrl) {
      return reject(
          "--still-tx-partition is not supported with --still-mrl yet");
    }
    if (options.still_tx_partition && options.still_ibp) {
      return reject(
          "--still-tx-partition is not supported with --still-ibp yet");
    }
    if (options.still_tcq && options.still_rdoq) {
      return reject("--still-tcq is not supported with --still-rdoq yet");
    }
    if (options.still_tcq && options.still_dc_ac_delta_q) {
      return reject(
          "--still-tcq is not supported with --still-dc-ac-delta-q yet");
    }
    if (options.still_tcq && options.still_ist) {
      return reject("--still-tcq is not supported with --still-ist yet");
    }
    if (options.still_per_segment_q && options.still_ist) {
      return reject(
          "--still-per-segment-q is not supported with --still-ist yet");
    }
    if (options.still_ddt) {
      return reject("--still-ddt is not signallable in a still picture: DDT/"
                    "DDTX/FDDT is inter-only (use the inter residual path)");
    }
  }
  if (options.still_ext_partitions) {
    if (!options.m1 && !options.still_gpu) {
      error = "--still-ext-partitions requires --still";
      return ParseStatus::kError;
    }

    if (options.still_single) {
      error = "--still-ext-partitions cannot be combined with --still-single";
      return ParseStatus::kError;
    }
  }
  if (options.svc_given) {
    if (!options.video) {
      error = "--svc-* requires --video";
      return ParseStatus::kError;
    }
    if (options.svc_layers.empty()) {
      error = "--svc-* requires --svc-layers x0=WxH[,x1=WxH,...]";
      return ParseStatus::kError;
    }
    if (options.tiles != "1x1") {
      error = "--svc-layers cannot be combined with --tiles: layered tiling is "
              "not gated";
      return ParseStatus::kError;
    }
    if (options.rc_given || options.rc_crf_given ||
        options.rc_bitrate_given) {
      error = "--svc-layers has one fixed quantizer per layer; --rc/--crf/"
              "--bitrate are not wired into the SVC path yet";
      return ParseStatus::kError;
    }
    const int selections = (options.svc_oppoint >= 0 ? 1 : 0) +
                           (options.svc_all_layers ? 1 : 0) +
                           (options.svc_spatial_layers >= 1 ? 1 : 0);
    if (selections > 1) {
      error = "select at most one of --oppoint, --alllayers or "
              "--spatial-layers";
      return ParseStatus::kError;
    }
    if (options.svc_temporal != 1) {
      error = "--svc-temporal > 1 needs the hierarchical GOP scheduler "
              "(one m0/t0 frame per spatial layer is emitted per temporal "
              "unit today)";
      return ParseStatus::kError;
    }
    if (options.svc_embedded != 1) {
      error = "--svc-embedded > 1 needs per-embedded-layer frame generation "
              "that this encoder does not implement";
      return ParseStatus::kError;
    }
    if (options.video_long_term || options.video_olk ||
        options.video_leading != 0 || options.video_sef ||
        options.video_switch || options.video_ras ||
        options.video_drl_reorder) {
      error = "--svc-layers cannot be combined with a frame-type mode";
      return ParseStatus::kError;
    }
    // run_svc_encode() rebuilds svc.video from a hand-picked field list: it
    // hard-codes num_total_refs = 1 and never copies temporal_layers /
    // tlayer_drop. Reject those flags by name instead of accepting them and
    // emitting a byte-identical stream.
    if (flag_given(options, "--video-refs") ||
        flag_given(options, "--video-num-ref-frames") ||
        flag_given(options, "--video-tlayers") ||
        flag_given(options, "--video-tlayer-drop")) {
      error = "--svc-layers codes one reference and one temporal layer per "
              "spatial layer; --video-refs / --video-tlayers / "
              "--video-tlayer-drop are not wired into the SVC path";
      return ParseStatus::kError;
    }
  }
  if (options.still_ext_partitions) {
    if (!options.m1 && !options.still_gpu) {
      error = "--still-ext-partitions requires --still";
      return ParseStatus::kError;
    }

    if (options.still_single) {
      error = "--still-ext-partitions cannot be combined with --still-single";
      return ParseStatus::kError;
    }
  }
  if (options.still_sub8) {
    if (!options.m1 && !options.still_gpu) {
      error = "--still-sub8 requires --still";
      return ParseStatus::kError;
    }
    if (options.still_single) {
      error = "--still-sub8 cannot be combined with --still-single";
      return ParseStatus::kError;
    }
    if (options.still_ext_partitions) {
      error = "--still-sub8 is not wired with --still-ext-partitions yet";
      return ParseStatus::kError;
    }
  }
  if (options.still_sdp) {
    if (!options.m1 && !options.still_gpu) {
      error = "--still-sdp requires --still";
      return ParseStatus::kError;
    }

    if (options.still_single) {
      error = "--still-sdp cannot be combined with --still-single";
      return ParseStatus::kError;
    }
    if (options.still_sb128 || options.still_sb256) {
      error = "--still-sdp is only wired for 64x64 superblocks";
      return ParseStatus::kError;
    }
    if (options.still_ext_partitions) {
      error = "--still-sdp is not wired with --still-ext-partitions yet";
      return ParseStatus::kError;
    }
    if (options.still_sub8) {
      error = "--still-sdp is not wired with --still-sub8 yet";
      return ParseStatus::kError;
    }
    if (options.still_cdef || options.still_ccso) {
      error = "--still-sdp with CDEF/CCSO is not wired yet";
      return ParseStatus::kError;
    }
    if (options.still_segmentation || options.still_delta_q) {
      error = "--still-sdp with segmentation/delta-Q is not wired yet";
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
    if (options.still_single) {
      error = "--still-single cannot be combined with a 128x128 superblock";
      return ParseStatus::kError;
    }
    if (options.still_depth == 0) {
      error = "--still-depth 0 cannot be combined with a 128x128 superblock "
              "(the 128 root must split)";
      return ParseStatus::kError;
    }
  }
  if (options.still_cdef_on_skip_txfm) {
    if (!options.m1 && !options.still_gpu) {
      error = "--still-cdef-on-skip-txfm requires --still";
      return ParseStatus::kError;
    }
    if (!options.still_cdef) {
      error = "--still-cdef-on-skip-txfm requires --still-cdef";
      return ParseStatus::kError;
    }
  }
  if (options.still_grain_adaptive) {
    if (!options.m1 && !options.still_gpu) {
      error = "--still-grain-adaptive requires --still";
      return ParseStatus::kError;
    }
    if (!options.still_grain) {
      error = "--still-grain-adaptive requires --still-grain";
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
// controller and the per-frame RC adapter so the frame header and the coefficient-CDF
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
    return 2;
  }

  // An explicit --tiles layout is a promise: never replace it with a default.
  // Without one, use the GPU-count default and fall back through the verified
  // presets when the preferred grid is not representable at this frame size.
  const bool explicit_tiles = flag_given(options, "--tiles");
  using av2enc::tiles::TileLayoutPreset;
  const TileLayoutPreset preferred =
      explicit_tiles ? tile_preset_from_layout(options.tiles)
                     : av2enc::multigpu::default_layout_for_gpus(
                           options.gpu_plan);
  std::vector<TileLayoutPreset> candidates;
  if (explicit_tiles) {
    candidates.push_back(preferred);
  } else {
    candidates.push_back(preferred);
    static constexpr TileLayoutPreset kFallbacks[] = {
        TileLayoutPreset::k16x2, TileLayoutPreset::k8x2,
        TileLayoutPreset::k4x2,  TileLayoutPreset::k2x2,
        TileLayoutPreset::k2x1,  TileLayoutPreset::k1x1};
    for (const TileLayoutPreset fallback : kFallbacks) {
      if (fallback != preferred) candidates.push_back(fallback);
    }
  }

  TileLayoutPreset preset = preferred;
  av2enc::tiles::LayoutResult planned;
  std::string first_detail;
  bool chosen = false;
  for (const TileLayoutPreset candidate : candidates) {
    const av2enc::tiles::TileLayoutDims dims =
        av2enc::tiles::preset_dims(candidate);
    // Level 5.1 (idx 7) is the M1 default, but Table A.9 MaxTileCols only
    // admits 16 columns from level 6.0. Plan (and report) a level that carries
    // the grid.
    const int plan_level = std::max(
        7, av2enc::tiles::minimum_level_for_grid(dims.cols, dims.rows));
    av2enc::tiles::LayoutResult attempt = av2enc::tiles::plan_tile_layout(
        static_cast<int>(y4m.width), static_cast<int>(y4m.height), dims.cols,
        dims.rows, av2enc::tiles::kSuperblockPixels64, plan_level, 0);
    if (!attempt.ok() || attempt.layout.tile_count() < options.gpu_plan) {
      if (first_detail.empty()) {
        first_detail = attempt.ok() ? "only " +
                                          std::to_string(
                                              attempt.layout.tile_count()) +
                                          " tile(s) for " +
                                          std::to_string(options.gpu_plan) +
                                          " GPU(s)"
                                    : attempt.error;
      }
      continue;
    }
    preset = candidate;
    planned = std::move(attempt);
    chosen = true;
    break;
  }
  if (!chosen) {
    std::cerr << "av2enc: error: --gpu-plan: ";
    if (explicit_tiles) {
      std::cerr << "layout " << av2enc::tiles::preset_name(preferred)
                << " rejected: " << first_detail;
    } else {
      std::cerr << "no tile layout fits " << y4m.width << "x" << y4m.height
                << " for " << options.gpu_plan << " GPU(s); preferred "
                << av2enc::tiles::preset_name(preferred) << ": "
                << first_detail;
    }
    std::cerr << "\n";
    return 2;
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
    return 2;
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
  if (!explicit_tiles && preset != preferred) {
    std::cout << "  note:    preferred layout "
              << av2enc::tiles::preset_name(preferred)
              << " does not fit; using " << av2enc::tiles::preset_name(preset)
              << "\n";
  }
  for (const av2enc::multigpu::GpuPlan& gpu : plan.gpus) {
    std::cout << "  gpu " << gpu.gpu_index << ":  tiles";
    for (const int tile : gpu.tile_indices) {
      std::cout << " " << tile;
    }
    std::cout << " (" << gpu.tile_count << ", " << gpu.tile_area_pixels
              << " px), vram " << gpu.vram.live_mb << " / " << gpu.vram.cap_mb
              << " MB fits=" << (gpu.vram.fits ? "yes" : "no") << "\n";
  }
  std::cout << "  dispatch:";
  for (const av2enc::multigpu::TileDispatchEntry& entry : plan.dispatch) {
    std::cout << " " << entry.raster_slot << ":" << entry.gpu_index;
  }
  std::cout << " (raster merge order; each tile is one work item)\n";
  std::cout << "  host:    transfer " << plan.transfer.total_mb << " MB, "
            << plan.transfer.total_ms << " ms ("
            << plan.transfer.pct_of_frame_budget << "% of the frame budget, "
            << plan.transfer.phases << " phases)\n"
            << "  scale:   model only (measured=" << (plan.measured ? 1 : 0)
            << ", evidence="
            << av2enc::multigpu::evidence_name(plan.evidence)
            << "), estimated speedup " << plan.scaling.speedup
            << "x; not a measurement\n"
            << "  note:    " << plan.disclaimer << "\n";
  return 0;
}

// Frame-count and order-hint preflight for the frame-type modes. The
// sequence encoders discover these after IvfWriter::open, so an invalid
// command used to truncate an existing -o stream and exit 1. Validate once the
// input length is known and before the writer is opened; the caller reports
// the message as a usage error (exit 2).
bool validate_video_sequence_length(const Options& options,
                                    std::size_t frame_count,
                                    std::string& error) {
  if (options.video_olk && frame_count != 4) {
    error = "--video-olk requires exactly 4 input frames (CLK, OLK, P2, P3); "
            "got " + std::to_string(frame_count);
    return false;
  }
  if (options.video_switch && frame_count != 2) {
    error = "--video-switch requires exactly 2 input frames (CLK, SWITCH); "
            "got " + std::to_string(frame_count);
    return false;
  }
  if (options.video_ras && frame_count != 3) {
    error = "--video-ras requires exactly 3 input frames (CLK, SWITCH, RAS); "
            "got " + std::to_string(frame_count);
    return false;
  }
  if (options.video_sef && frame_count < 2) {
    error = "--video-sef needs at least 2 input frames; got " +
            std::to_string(frame_count);
    return false;
  }
  if (options.video_leading != 0) {
    const std::size_t lead = static_cast<std::size_t>(options.video_leading);
    if (frame_count < lead + 2U) {
      error = "--video-leading " + std::to_string(options.video_leading) +
              " needs at least " + std::to_string(lead + 2U) +
              " input frames; got " + std::to_string(frame_count);
      return false;
    }
    const std::size_t regular = frame_count - lead - 2U;
    if (lead + regular > 7U) {
      error = "--video-leading: leading + regular frames exceed the 8 slots "
              "this profile signals (pass at most 9 input frames)";
      return false;
    }
  }
  if (options.video_tlayers != 0 &&
      frame_count != static_cast<std::size_t>(options.video_tlayers)) {
    error = "--video-tlayers codes one closed GOP: pass exactly " +
            std::to_string(options.video_tlayers) + " input frames; got " +
            std::to_string(frame_count);
    return false;
  }
  if (options.b_frames != 0 &&
      frame_count != static_cast<std::size_t>(options.b_frames) + 2U) {
    error = "--b-frames codes one closed GOP: pass exactly " +
            std::to_string(options.b_frames + 2U) +
            " frames (N B frames + key + P anchor); got " +
            std::to_string(frame_count);
    return false;
  }
  if (options.video_long_term && frame_count > 16U) {
    error = "--video-long-term supports at most 16 input frames "
            "(OrderHintBits = 4); got " + std::to_string(frame_count);
    return false;
  }
  return true;
}

int run_video_encode(const Options& options, av2enc::Y4mReader& reader,
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

  if (!validate_video_sequence_length(options, inputs.size(), error)) {
    std::cerr << "av2enc: error: " << error << "\n";
    return 2;
  }

  // The output is opened only after every sequence-level usage check above, so
  // an invalid combination never truncates an existing -o stream.
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

  av2enc::video::VideoOptions video_options;
  video_options.base_q_idx = static_cast<std::uint8_t>(options.qp);
  video_options.avg_cdf = options.avg_cdf;
  video_options.adaptive_cdf = options.adaptive_cdf;
  video_options.cdf_continuity = options.cdf_continuity;
  video_options.mfh = options.video_mfh;
  if (options.video_timing) {
    if (y4m.fps_num == 0 || y4m.fps_den == 0) {
      std::cerr << "av2enc: error: --video-timing needs a non-zero Y4M frame "
                   "rate (num_units_in_display_tick/time_scale)\n";
      return 1;
    }
    video_options.video_timing = true;
    video_options.timing_num_units_in_display_tick = y4m.fps_den;
    video_options.timing_time_scale = y4m.fps_num;
  }
  video_options.num_total_refs = options.video_refs;
  video_options.num_ref_frames = options.video_num_ref_frames;
  video_options.compound = options.video_compound;
  video_options.b_frames = static_cast<int>(options.b_frames);
  video_options.enable_inter_residual = options.video_inter_residual;
  video_options.enable_tx_search = options.video_tx_search;
  video_options.tx_rd_lambda = options.video_tx_lambda;
  video_options.enable_deblock = options.video_deblock;
  video_options.enable_gme = options.video_gme;
  video_options.enable_warp = options.video_warp;
  video_options.enable_intra_full = options.video_intra_full;
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
  video_options.temporal_layers = options.video_tlayers;
  video_options.tlayer_drop = options.video_tlayer_drop;
  video_options.force_single_block = options.video_force_single;
  video_options.min_block_size = options.video_min_block;
  video_options.max_partition_depth = options.video_max_depth;
  video_options.me_range = options.video_me_range;
  video_options.loop_filters.cdef = options.video_cdef;
  video_options.loop_filters.cdef_num_strengths = options.video_cdef_strength;
  video_options.loop_filters.cdef_damping = options.video_cdef_damping;
  video_options.loop_filters.ccso = options.video_ccso;
  video_options.loop_filters.ccso_bands_log2 = options.video_ccso_bands;
  video_options.loop_filters.wiener = options.video_wiener;
  video_options.loop_filters.chroma_lr = options.video_chroma_lr;
  video_options.loop_filters.gdf = options.video_gdf;
  video_options.trace = options.video_trace;
  if (options.tiles != "1x1") {
    const av2enc::tiles::TileLayoutDims dims =
        av2enc::tiles::preset_dims(tile_preset_from_layout(options.tiles));
    video_options.tile_cols = dims.cols;
    video_options.tile_rows = dims.rows;
  }

  // Opt-in per-frame rate control. CQP (default) leaves frame_base_q_idx empty,
  // so encode_video_sequence runs the historical fixed-QP path unchanged.
  // The RC statistics / probe passes must use the same backend as the final
  // encode; --video-gpu never falls back to the CPU encoder.  For CQP the
  // historical single call below is unchanged.
  const auto encode_rc_pass =
      [&options, &error](const std::vector<av2enc::video::VideoInputFrame>& seq,
                         const av2enc::video::VideoOptions& opts,
                         av2enc::video::VideoSequenceResult& out) -> bool {
#if AV2ENC_VIDEO_GPU_CALLABLE
    if (options.video_gpu) {
      return av2enc::cuda::video::encode_video_sequence_gpu(seq, opts, out,
                                                            error);
    }
#endif
    return av2enc::video::encode_video_sequence(seq, opts, out, error);
  };
  std::vector<std::uint8_t> frame_qp;
  // The rate controller's per-frame plan.  VBR keeps this shape and searches a
  // single global offset; it never replaces the plan with one uniform QP.
  std::vector<std::uint8_t> planned_qp;
  if (options.rc_mode != "cqp") {
    std::vector<std::uint32_t> pass1_payload_bits;
    if (options.rc_mode == "vbr") {
      // Statistics pass at the fixed --qp: payload bits feed the VBR virtual
      // buffer, the same role as a first pass in a conventional encoder.
      av2enc::video::VideoSequenceResult stats;
      if (!encode_rc_pass(inputs, video_options, stats)) {
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
    planned_qp = frame_qp;
  }

  av2enc::video::VideoSequenceResult result;
  int vbr_iterations = 0;
  double vbr_target_kbps = 0.0;
  if (options.rc_mode == "vbr") {
    // Closed-loop two-pass VBR.  The statistics pass calibrates the controller
    // and the controller plans a base_q_idx per frame.  Every probe is that
    // plan shifted by one global offset (the frame-to-frame shape is kept, so
    // flat and busy frames do not collapse onto one QP); the measured payload
    // bits drive a bisection over the offset.  The measured payload is in the
    // same units as the target, so the search is deterministic.
    const double duration =
        static_cast<double>(inputs.size()) *
        static_cast<double>(y4m.fps_den) / static_cast<double>(y4m.fps_num);
    vbr_target_kbps = options.rc_bitrate_kbps;
    const double target_bits = vbr_target_kbps * 1000.0 * duration;
    const auto apply_offset = [&](int offset) {
      frame_qp.resize(planned_qp.size());
      for (std::size_t i = 0; i < planned_qp.size(); ++i) {
        frame_qp[i] = static_cast<std::uint8_t>(
            std::clamp(static_cast<int>(planned_qp[i]) + offset, 1, 255));
      }
      video_options.frame_base_q_idx = frame_qp;
      video_options.base_q_idx = frame_qp.front();
    };
    int low = -254;
    int high = 254;
    int offset = 0;
    double best_error = std::numeric_limits<double>::infinity();
    av2enc::video::VideoSequenceResult best_result;
    std::vector<std::uint8_t> best_qp = planned_qp;
    while (low <= high && vbr_iterations < 12) {
      ++vbr_iterations;
      apply_offset(offset);
      av2enc::video::VideoSequenceResult candidate;
      if (!encode_rc_pass(inputs, video_options, candidate)) {
        std::cerr << "av2enc: error: --rc vbr pass " << vbr_iterations << ": "
                  << error << "\n";
        return 1;
      }
      std::uint64_t measured = 0;
      for (const std::uint32_t bytes : candidate.tile_bytes) {
        measured += static_cast<std::uint64_t>(bytes) * 8U;
      }
      const double error_ratio =
          target_bits > 0.0
              ? std::fabs(static_cast<double>(measured) - target_bits) /
                    target_bits
              : 0.0;
      if (error_ratio < best_error) {
        best_error = error_ratio;
        best_result = candidate;
        best_qp = frame_qp;
      }
      if (error_ratio <= 0.03) break;
      if (static_cast<double>(measured) > target_bits) {
        low = offset + 1;  // too many bits: raise every frame's base_q_idx
      } else {
        high = offset - 1;
      }
      offset = low + (high - low) / 2;
    }
    result = std::move(best_result);
    frame_qp = best_qp;
    video_options.frame_base_q_idx = frame_qp;
    video_options.base_q_idx = frame_qp.front();
  } else if (options.video_gpu) {
#if AV2ENC_VIDEO_GPU_CALLABLE
    av2enc::cuda::video::VideoGpuStats gpu_stats;
    if (!av2enc::cuda::video::encode_video_sequence_gpu(
            inputs, video_options, result, error, &gpu_stats)) {
      std::cerr << "av2enc: error: --video-gpu: " << error << "\n";
      return 1;
    }
#if AV2ENC_HAVE_CUDA
    if (options.video_trace) {
      std::cerr << "av2enc: video GPU stats: launches="
                << gpu_stats.total_launches() << " syncs=" << gpu_stats.syncs
                << " uploaded=" << gpu_stats.bytes_uploaded
                << " downloaded=" << gpu_stats.bytes_downloaded
                << " device_ms=" << gpu_stats.device_ms << "\n";
    }
#endif
#else
    std::cerr << "av2enc: error: --video-gpu is not linked in this build\n";
    return 1;
#endif
  } else if (!av2enc::video::encode_video_sequence(inputs, video_options,
                                                   result, error)) {
    std::cerr << "av2enc: error: " << error << "\n";
    return 1;
  }
  if (options.video_gpu) {
#if AV2ENC_VIDEO_GPU_CALLABLE
    // Honest provenance: neither the CQP dispatch nor an --rc crf/vbr probe
    // re-enters the CPU encoder, so frame_bytes_from_cpu stays 0.  Name the
    // linked backend explicitly for every --video-gpu mode.
    std::cerr << "av2enc: --video-gpu backend="
              << av2enc::cuda::video::video_backend_name()
              << " frame_bytes_from_cpu=0\n";
#endif
  }
  for (std::size_t i = 0; i < result.units.size(); ++i) {
    // Neutral metadata is placed once, as suffix units of the first temporal
    // unit.  It is decoder-invariant (§5.17.2 preserves coded output order).
    if (i == 0 &&
        !append_neutral_metadata_obus(result.units[i], options.metadata_timecode,
                                      options.metadata_scan_type, error)) {
      std::cerr << "av2enc: error: " << error << "\n";
      return 1;
    }
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
    // The achieved payload is reported over the displayed duration, which is
    // the same denominator the VBR target used.  --video-sef appends a
    // show-existing-frame unit, so result.units.size() over-counts the coded
    // frames; result.recon.size() is the displayed picture count.
    const std::size_t displayed_frames =
        result.recon.empty() ? result.units.size() : result.recon.size();
    const double duration =
        static_cast<double>(displayed_frames) *
        static_cast<double>(y4m.fps_den) / static_cast<double>(y4m.fps_num);
    const double kbps = duration > 0.0
                            ? static_cast<double>(total_tile) * 8.0 / duration /
                                  1000.0
                            : 0.0;
    std::cout << "  rc:      " << options.rc_mode;
    if (options.rc_mode == "crf") {
      std::cout << " " << options.rc_crf;
    } else if (options.rc_mode == "vbr") {
      const double error_pct =
          vbr_target_kbps > 0.0
              ? (kbps - vbr_target_kbps) / vbr_target_kbps * 100.0
              : 0.0;
      std::cout << " target " << vbr_target_kbps << " kbps (closed-loop "
                << vbr_iterations << " pass(es))";
      std::cout << ", achieved " << kbps << " kbps ("
                << (error_pct >= 0.0 ? "+" : "") << error_pct
                << "% error, payload)\n";
    }
    if (options.rc_mode == "crf") {
      std::cout << ", achieved " << kbps << " kbps (payload)\n";
    }
    std::cout << "  frame QP:";
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

// "x0=WxH,x1=WxH" -> SpatialLayerSpec list.  Ids must start at 0 and be
// contiguous ascending; build_layer_plan() re-validates the same rule.
bool parse_svc_layer_specs(
    std::string_view text,
    std::vector<av2enc::svc::SpatialLayerSpec>& layers, std::string& error) {
  layers.clear();
  std::size_t start = 0;
  while (start <= text.size()) {
    const std::size_t comma = text.find(',', start);
    const std::string_view item =
        text.substr(start, comma == std::string_view::npos
                               ? std::string_view::npos
                               : comma - start);
    if (item.empty()) {
      error = "--svc-layers has an empty layer entry";
      return false;
    }
    const std::size_t eq = item.find('=');
    if (eq == std::string_view::npos) {
      error = "--svc-layers entries must be xID=WxH";
      return false;
    }
    std::string_view id_text = item.substr(0, eq);
    if (!id_text.empty() && (id_text.front() == 'x' || id_text.front() == 'X')) {
      id_text.remove_prefix(1);
    }
    int id = 0;
    auto id_result = std::from_chars(id_text.data(),
                                     id_text.data() + id_text.size(), id);
    if (id_result.ec != std::errc{} ||
        id_result.ptr != id_text.data() + id_text.size() || id < 0 || id > 30) {
      error = "--svc-layers xlayer id must be an integer in [0, 30]";
      return false;
    }
    const std::size_t xpos = item.find('x', eq + 1);
    if (xpos == std::string_view::npos) {
      error = "--svc-layers entries must be xID=WxH";
      return false;
    }
    int width = 0;
    int height = 0;
    auto w_result = std::from_chars(item.data() + eq + 1, item.data() + xpos,
                                   width);
    auto h_result = std::from_chars(item.data() + xpos + 1,
                                    item.data() + item.size(), height);
    if (w_result.ec != std::errc{} ||
        w_result.ptr != item.data() + xpos || h_result.ec != std::errc{} ||
        h_result.ptr != item.data() + item.size() || width <= 0 || height <= 0) {
      error = "--svc-layers WxH must be positive integers";
      return false;
    }
    av2enc::svc::SpatialLayerSpec spec;
    spec.xlayer_id = static_cast<std::uint8_t>(id);
    spec.width = width;
    spec.height = height;
    layers.push_back(spec);
    if (comma == std::string_view::npos) break;
    start = comma + 1;
  }
  return true;
}

int run_svc_encode(const Options& options, av2enc::Y4mReader& reader,
                   const av2enc::Y4mHeader& y4m) {
  std::string error;
  av2enc::svc::SvcEncodeOptions svc;
  if (!parse_svc_layer_specs(options.svc_layers, svc.config.spatial_layers,
                             error)) {
    std::cerr << "av2enc: error: " << error << "\n";
    return 2;
  }
  svc.config.temporal_layers = 1;
  svc.config.embedded_layers = 1;
  svc.config.single_picture_header = false;
  svc.config.still_picture = false;
  svc.config.seq_lcr_id = static_cast<std::uint8_t>(options.svc_lcr_id);
  svc.all_layers = !(options.svc_oppoint >= 0 ||
                     options.svc_spatial_layers >= 1);
  svc.oppoint = options.svc_oppoint;
  svc.spatial_layers = options.svc_spatial_layers;
  svc.video.base_q_idx = static_cast<std::uint8_t>(options.qp);
  svc.video.num_total_refs = 1;
  svc.video.enable_inter_residual = options.video_inter_residual;
  svc.video.me_subpel_eighth = options.video_subpel_eighth;
  svc.video.mv_precision = options.video_mv_precision;
  svc.video.force_single_block = options.video_force_single;
  svc.video.min_block_size = options.video_min_block;
  svc.video.max_partition_depth = options.video_max_depth;
  svc.video.me_range = options.video_me_range;
  svc.video.trace = options.video_trace;

  std::vector<OwnedVideoFrame> owned;
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

  av2enc::svc::SvcStreamResult result;
  if (!av2enc::svc::encode_svc_sequence(inputs, svc, result, error)) {
    std::cerr << "av2enc: error: --svc-layers: " << error << "\n";
    return 1;
  }

  // The IVF header stays the base-layer size (Annex F.3.2 extraction keeps the
  // container header); timestamps are temporal-unit indices.
  av2enc::IvfHeader ivf;
  ivf.width = static_cast<std::uint16_t>(svc.config.spatial_layers.front().width);
  ivf.height =
      static_cast<std::uint16_t>(svc.config.spatial_layers.front().height);
  ivf.fps_num = y4m.fps_num;
  ivf.fps_den = y4m.fps_den;
  av2enc::IvfWriter writer;
  if (!writer.open(options.output, ivf, error)) {
    std::cerr << "av2enc: " << error << "\n";
    return 1;
  }
  for (std::size_t i = 0; i < result.units.size(); ++i) {
    if (!writer.write_frame(result.units[i], static_cast<std::uint64_t>(i),
                            error)) {
      std::cerr << "av2enc: error: " << error << "\n";
      return 1;
    }
  }
  if (!writer.close(error)) {
    std::cerr << "av2enc: error: " << error << "\n";
    return 1;
  }

  std::uint64_t total_payload = 0;
  for (const std::uint32_t bytes : result.unit_payload_bytes) {
    total_payload += bytes;
  }
  std::cout << "av2enc: SVC layered encode complete\n"
            << "  input:   " << options.input.string() << " (" << y4m.width
            << "x" << y4m.height << ", " << y4m.fps_num << "/" << y4m.fps_den
            << " fps)\n"
            << "  output:  " << options.output.string() << " ("
            << writer.bytes_written() << " bytes IVF, base "
            << ivf.width << "x" << ivf.height << ")\n"
            << "  layers:  " << result.layers.size() << "\n";
  for (const av2enc::svc::SvcLayerOutput& layer : result.layers) {
    std::cout << "    x" << layer.xlayer_id << ": " << layer.width << "x"
              << layer.height << ", " << layer.recon.size() << " frame(s)\n";
  }
  std::cout << "  units:   " << result.units.size() << " temporal unit(s), "
            << result.selected_layers << " selected layer(s), "
            << result.selection_label << "\n"
            << "  payload: " << total_payload << " bytes\n";
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

  // The effort report is printed for every --effort/--cpu-used run (the
  // contract asks for the selected flags and the used budget), and
  // --effort-plan stops before the input or output is touched.
  if (options.effort_active) {
    std::cerr << options.effort_report;
  }
  if (options.effort_plan) {
    return 0;
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
    return run_gpu_plan(options);
  }

  if (options.still_ibc && !options.m1 && !options.still_gpu &&
      !options.lossless) {
    std::cerr << "av2enc: error: --still-ibc is an M1 still-picture tool; "
                 "combine it with --still or --still-gpu\n";
    return 2;
  }

  if (options.still_palette && !options.m1 && !options.still_gpu &&
      !options.lossless) {
    std::cerr << "av2enc: error: --still-palette is an M1 still-picture tool; "
                 "combine it with --still or --still-gpu\n";
    return 2;
  }

  if (options.still_ibc && (options.still_sb128 || options.still_sb256)) {
    std::cerr << "av2enc: error: --still-ibc requires 64x64 superblocks "
                 "(drop --still-sb128/--still-sb256)\n";
    return 2;
  }

  if (options.still_ibc_search_step != 8 && !options.still_ibc) {
    std::cerr << "av2enc: error: --still-ibc-search requires --still-ibc\n";
    return 2;
  }

#if !AV2ENC_STILL_GPU_CALLABLE
  // Accept the flag in every build but fail before touching the input/output
  // so the caller gets a precise reason instead of a link-time or decode-time
  // surprise.
  if (options.still_gpu) {
#if AV2ENC_HAVE_CUDA
    std::cerr << "av2enc: error: --still-gpu is not available in this build: "
                 "the Still GPU entry point (src/cuda/pipeline/still/still_gpu.cu) has not "
                 "landed yet. Use --still for the CPU Still path.\n";
#else
    std::cerr << "av2enc: error: --still-gpu requires a CUDA build; this is a "
                 "CPU-only reference build. Use --still for the CPU Still path.\n";
#endif
    return 2;
  }
#endif

  if (options.lossless && options.still_gpu) {
    std::cerr << "av2enc: error: --lossless is implemented by the CPU "
                 "lossless module; it cannot be combined with --still-gpu\n";
    return 2;
  }

  if (options.still_seg_delta != 8 && !options.still_per_segment_q) {
    std::cerr << "av2enc: error: --still-seg-delta requires "
                 "--still-per-segment-q\n";
    return 2;
  }
  if (options.still_per_segment_q) {
    if (!options.m1 && !options.still_gpu) {
      std::cerr << "av2enc: error: --still-per-segment-q requires --still\n";
      return 2;
    }
    if (options.still_segmentation || options.still_delta_q) {
      std::cerr << "av2enc: error: --still-per-segment-q replaces the neutral "
                   "--still-segmentation/--still-delta-q table; use one form\n";
      return 2;
    }
  }

  if (options.lossless &&
      (options.still_content_interpretation || options.still_segmentation ||
       options.still_delta_q || options.still_per_segment_q ||
       options.still_palette || options.still_ibc)) {
    std::cerr << "av2enc: error: --lossless uses its own header writer; "
                 "--still-content-interpretation, the delta-Q/segmentation "
                 "flags and the screen-content tools are not applied there\n";
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
    if (options.still_user_qm) return reject_cpu_only("--still-user-qm");
    if (options.still_tcq) return reject_cpu_only("--still-tcq");
    if (options.still_tx_partition) {
      return reject_cpu_only("--still-tx-partition");
    }
    if (options.still_ddt) return reject_cpu_only("--still-ddt");
    // CDEF, CCSO, loop restoration (Wiener NS, chroma LR, PC-Wiener, GDF) and
    // film-grain signalling now have GPU stages (see
    // gpu_m3_options_supported() in src/cuda/pipeline/still/still_gpu.h); the
    // remaining filters have no stage yet and are rejected by name.
    if (options.still_grain_adaptive) {
      return reject_cpu_only("--still-grain-adaptive");
    }
    if (options.still_rdoq) return reject_cpu_only("--still-rdoq");
    if (options.still_rdo_intra) return reject_cpu_only("--still-rdo-intra");
  }

  // Multi-tile stills (--tiles).  The tiled writer is a distinct composition
  // root: it reuses the Still reference per tile but pins the tool set to the
  // verified form, so every option that would change the tile payload is
  // rejected by name instead of being ignored.
  const bool multi_tile = options.tiles != "1x1";
  if (options.cdf_continuity && !options.video) {
    std::cerr << "av2enc: error: --cdf-continuity is wired for --video only\n";
    return 2;
  }
  if (options.video_timing && !options.video) {
    std::cerr << "av2enc: error: --video-timing needs --video: a "
                 "single-picture still header cannot carry the timing block\n";
    return 2;
  }
  if (options.video_mfh && !options.video) {
    std::cerr << "av2enc: error: --video-mfh needs --video: the pinned "
                 "decoder rejects MFH on the single-picture still path "
                 "(docs/negative-results.md)\n";
    return 2;
  }
  if (options.avg_cdf) {
    if (options.lossless) {
      std::cerr << "av2enc: error: --avg-cdf is not implemented by the "
                   "--lossless writer\n";
      return 2;
    }
    if (!options.video && !multi_tile) {
      std::cerr << "av2enc: error: --avg-cdf needs a multi-tile still "
                   "(--tiles > 1) or --video; a single-tile still already "
                   "infers enable_avg_cdf = 1 and has no frame end to average\n";
      return 2;
    }
  }
  if (multi_tile) {
    if (options.video) {
      // Multi-tile video: the per-frame splitter re-emits the whole-frame
      // block list through post::assemble_video_tiled_frame().  Layouts and
      // header fields are pinned to what that writer signals.
      if (options.tiles == "16x2") {
        std::cerr << "av2enc: error: --tiles 16x2 is not gate-verified for "
                     "--video; use 2x1/2x2/4x2/8x2\n";
        return 2;
      }
      if (options.video_refs != 1) {
        std::cerr << "av2enc: error: multi-tile --video requires --video-refs "
                     "1 (the tiled frame header signals one reference slot)\n";
        return 2;
      }
      if (options.video_mv_precision != 0) {
        std::cerr << "av2enc: error: multi-tile --video requires eighth-pel MV "
                     "precision (the default)\n";
        return 2;
      }
      if (options.video_long_term || options.video_olk ||
          options.video_leading != 0 || options.video_sef ||
          options.video_switch || options.video_ras ||
          options.video_drl_reorder || options.video_drl_reorder_always) {
        std::cerr << "av2enc: error: multi-tile --video cannot be combined with "
                     "a frame-type mode or DRL reordering\n";
        return 2;
      }
      if (options.bit_depth != 8 || options.chroma != "420") {
        std::cerr << "av2enc: error: multi-tile --video is 8-bit 4:2:0 only\n";
        return 2;
      }
    } else {
      if (options.still_gpu) {
        std::cerr << "av2enc: error: --tiles is CPU-only (--still); the Still "
                     "GPU pipeline has no tile loop\n";
        return 2;
      }
      if (!options.m1) {
        std::cerr << "av2enc: error: --tiles > 1 requires --still or --video\n";
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
          {options.still_sub8, "--still-sub8"},
          {options.still_sdp, "--still-sdp"},
          {options.still_segmentation, "--still-segmentation"},
          {options.still_delta_q, "--still-delta-q"},
          {options.still_per_segment_q, "--still-per-segment-q"},
          {options.still_content_interpretation,
           "--still-content-interpretation"},
          {options.still_qm, "--still-qm"},
          {options.still_user_qm, "--still-user-qm"},
          {options.still_ist, "--still-ist"},
          {options.still_tcq, "--still-tcq"},
          {options.still_cctx, "--still-cctx"},
          {options.still_tx_partition, "--still-tx-partition"},
          {options.still_idtx, "--still-idtx"},
          {options.still_ddt, "--still-ddt"},
          {options.still_dc_ac_delta_q, "--still-dc-ac-delta-q"},
          {options.still_ibp, "--still-ibp"},
          {options.still_mrl, "--still-mrl"},
          {options.still_cfl_ds != 0, "--still-cfl-ds"},
          {options.still_cfl_explicit, "--still-cfl-explicit"},
          {options.still_dip, "--still-dip"},
          {options.still_palette, "--still-palette"},
          {options.still_ibc, "--still-ibc"},
          {options.still_mhccp, "--still-mhccp"},
          {options.still_ccso_edge, "--still-ccso-edge"},
          {options.still_wiener, "--still-wiener"},
          {options.still_chroma_lr, "--still-chroma-lr"},
          {options.still_pc_wiener, "--still-pc-wiener"},
          {options.still_gdf, "--still-gdf"},
          {options.still_grain, "--still-grain"},
          {options.still_grain_adaptive, "--still-grain-adaptive"},
          {options.still_cdef_on_skip_txfm, "--still-cdef-on-skip-txfm"},
          {options.metadata_timecode, "--metadata-timecode"},
          {options.metadata_scan_type, "--metadata-scan-type"},
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
  }

  // Optical-format still paths (10-bit, 4:2:2/4:4:4/4:0:0).  Every flag
  // defaults to today's 8-bit 4:2:0 stream, which never enters this branch.
  const bool format_10bit = options.bit_depth == 10;
  const bool format_chroma = options.chroma != "420";
  if ((format_10bit || format_chroma) && options.video) {
    std::cerr << "av2enc: error: --video is 8-bit 4:2:0 only; high-bit-depth / "
                 "4:2:2 / 4:4:4 / 4:0:0 inter is not implemented (the M2 "
                 "video core is uint8)\n";
    return 2;
  }
  if (format_10bit || format_chroma) {
    // Product GPU formats with a parity-gated block stage.  Every product
    // format the CPU format writers support is wired (10-bit
    // 4:2:0/4:2:2/4:4:4/4:0:0 and 8-bit 4:2:2/4:4:4/4:0:0); anything else is
    // rejected by name instead of silently running the CPU encoder under the
    // GPU flag.
    const bool gpu_format_wired =
        format_10bit ||
        (!format_10bit && (options.chroma == "422" ||
                           options.chroma == "444" ||
                           options.chroma == "400"));
    if (options.still_gpu && !gpu_format_wired) {
      std::cerr << "av2enc: error: --still-gpu does not accept this "
                   "--bit-depth/--chroma combination\n";
      return 2;
    }
    if (!options.m1 && !options.still_gpu) {
      std::cerr << "av2enc: error: --bit-depth/--chroma require --still or "
                   "--still-gpu\n";
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
    if (options.metadata_timecode || options.metadata_scan_type) {
      std::cerr << "av2enc: error: --metadata-* is not wired into the "
                   "--bit-depth/--chroma format still writers\n";
      return 2;
    }
    if (options.still_single || options.still_dc || options.still_full_intra ||
        options.still_uv_full || options.still_rdo_intra || options.still_rdoq ||
        options.still_tx_type || options.still_cdef || options.still_ccso ||
        options.still_sb128 || options.still_sb256 ||
        options.still_ext_partitions || options.still_segmentation ||
        options.still_delta_q || options.still_per_segment_q ||
        options.still_content_interpretation ||
        options.still_qm || options.still_user_qm || options.still_ist ||
        options.still_tcq || options.still_cctx ||
        options.still_tx_partition || options.still_idtx || options.still_ddt ||
        options.still_dc_ac_delta_q || options.still_ibp || options.still_mrl ||
        options.still_sub8 || options.still_sdp ||
        flag_given(options, "--still-cfl-ds") ||
        options.still_cfl_explicit || options.still_dip || options.still_palette ||
        options.still_ibc ||
        options.still_mhccp || options.still_ccso_edge || options.still_wiener ||
        options.still_chroma_lr || options.still_pc_wiener || options.still_gdf ||
        options.still_grain || options.still_depth >= 0) {
      std::cerr << "av2enc: error: the --bit-depth/--chroma format paths use "
                   "their own minimal still writer; Still tool flags are not "
                   "applied there\n";
      return 2;
    }
    if (format_10bit && (options.qp < 1 || options.qp > 303)) {
      std::cerr << "av2enc: error: 10-bit encoding supports base_q_idx 1..303 "
                   "(MaxQ at 10-bit); got --qp "
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
    std::unique_ptr<av2enc::chroma::ChromaGpuExecutor> chroma_gpu;
#if AV2ENC_STILL_GPU_CALLABLE
    if (options.still_gpu && format_10bit) {
      format_options.high_bitdepth_gpu =
          &av2enc::cuda::still_gpu::encode_format_gpu;
    } else if (options.still_gpu && !format_10bit) {
      chroma_gpu = av2enc::cuda::still_gpu::make_chroma_gpu_executor();
      format_options.chroma_gpu = chroma_gpu.get();
    }
#endif

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

  if (!options.recon.empty() && !multi_tile && !options.m1) {
    std::cerr << "av2enc: error: --recon is wired for the --bit-depth 10 and "
                 "--chroma format paths, --still, and multi-tile --tiles "
                 "stills only\n";
    return 2;
  }
  if (options.qp > 255) {
    std::cerr << "av2enc: error: --qp " << options.qp
              << " exceeds MaxQ = 255 at 8-bit; base_q_idx 256..303 requires "
                 "--still --bit-depth 10\n";
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
  // Geometry constraints of the selected mode are usage errors: report them
  // before the output file is opened.
  if (options.video) {
    if (y4m.width % 8U != 0U || y4m.height % 8U != 0U) {
      std::cerr << "av2enc: error: --video requires dimensions divisible by 8, "
                << "got " << y4m.width << "x" << y4m.height << "\n";
      return 2;
    }
    const int declared_frames =
        options.max_frames != 0
            ? options.max_frames
            : (options.p_frames != 0 ? options.p_frames + 1 : 0);
    if (options.video_long_term && declared_frames > 16) {
      std::cerr << "av2enc: error: --video-long-term codes OrderHint with "
                   "OrderHintBits = 4, so it supports at most 16 coded frames "
                   "(requested "
                << declared_frames << ")\n";
      return 2;
    }
  } else if (y4m.width % 2U != 0U || y4m.height % 2U != 0U) {
    std::cerr << "av2enc: error: 4:2:0 requires even dimensions, got "
              << y4m.width << "x" << y4m.height << "\n";
    return 2;
  }
  if (options.still_dc_ac_delta_q &&
      (y4m.width % 8U != 0U || y4m.height % 8U != 0U)) {
    std::cerr << "av2enc: error: --still-dc-ac-delta-q requires width and "
                 "height multiples of 8, got "
              << y4m.width << "x" << y4m.height << "\n";
    return 2;
  }

  if (options.svc_given) {
    if (!options.video_recon.empty()) {
      std::cerr << "av2enc: error: --video-recon is not wired for "
                   "--svc-layers (per-layer recon is internal to the gate)\n";
      return 2;
    }
    return run_svc_encode(options, reader, y4m);
  }

  if (options.video) {
    // All video option/sequence validation (including the frame-count and
    // rate-control combinations) happens inside run_video_encode before it
    // opens the output, so an invalid command exits 2 without truncating -o.
    return run_video_encode(options, reader, y4m);
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
  const auto write_unit = [&](std::vector<std::uint8_t>& unit,
                              std::uint64_t timestamp, std::string& err) {
    if (!append_neutral_metadata_obus(unit, options.metadata_timecode,
                                      options.metadata_scan_type, err)) {
      return false;
    }
    return writer.write_frame(unit, timestamp, err);
  };

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
  int ibc_blocks = 0;

  // Basic's edge recursion cannot code blocks below 32x32 in one dimension
  // (it would need 16x16 / 16x8 / 8x16 / 8x32 / 8x8 coders). Rather than emit
  // a desynchronised stream, use the Still path for those geometries and say
  // so on stderr. Explicit --still / --still-gpu are unaffected.
  bool basic_fallback_to_m1 = false;
  if (!options.m1 && !options.still_gpu && !options.lossless) {
    std::string reason;
    if (!av2enc::basic::geometry_supported(frame.width(), frame.height(),
                                        reason)) {
      std::cerr << "av2enc: note: Basic cannot represent " << frame.width()
                << "x" << frame.height() << " (" << reason
                << "); falling back to the Still path\n";
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
    if (!write_unit(encoded.temporal_unit, 0, error)) {
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
#if AV2ENC_STILL_GPU_CALLABLE
      // GpuOptions mirrors StillOptions and adds orchestrator controls; see
      // src/cuda/pipeline/still/still_gpu.h and docs/m1-gpu.md.
      av2enc::cuda::still_gpu::GpuOptions gpu_options;
      gpu_options.base_q_idx = static_cast<std::uint8_t>(options.qp);
      gpu_options.enable_cfl = options.still_cfl;
      gpu_options.cfl_ds_filter = options.still_cfl_ds;
      gpu_options.enable_deblock = options.still_deblock;
      gpu_options.enable_edge_filter = true;
      gpu_options.luma_dc_only = !options.still_full_intra &&
                                 !options.still_rdo_intra && !options.still_mrl;
      gpu_options.min_block_size = 8;
      gpu_options.sb_size = options.still_sb256 ? 256
                            : options.still_sb128 ? 128
                                                  : 64;
      gpu_options.enable_ibp = options.still_ibp;
      gpu_options.enable_mrl = options.still_mrl;
      gpu_options.enable_dip = options.still_dip;
      gpu_options.cfl_explicit = options.still_cfl_explicit;
      gpu_options.mhccp = options.still_mhccp;
      gpu_options.palette = options.still_palette;
      gpu_options.ibc = options.still_ibc;
      gpu_options.ibc_search_step = options.still_ibc_search_step;
      gpu_options.ext_partitions = options.still_ext_partitions;
      gpu_options.sdp = options.still_sdp;
      gpu_options.sub8 = options.still_sub8;
      if (options.still_depth >= 0) {
        gpu_options.max_partition_depth = options.still_depth;
      }
      gpu_options.adaptive_cdf = options.adaptive_cdf;
      gpu_options.force_single_block = options.still_single;
      gpu_options.force_dc = options.still_dc;
      gpu_options.uv_full = options.still_uv_full;
      gpu_options.enable_cdef = options.still_cdef;
      gpu_options.cdef_num_strengths = options.still_cdef_strengths;
      gpu_options.cdef_on_skip_txfm = options.still_cdef_on_skip_txfm;
      gpu_options.enable_ccso = options.still_ccso;
      gpu_options.ccso_bands_log2 = options.still_ccso_bands;
      gpu_options.ccso_edge = options.still_ccso_edge;
      gpu_options.enable_wiener = options.still_wiener;
      gpu_options.enable_chroma_lr = options.still_chroma_lr;
      gpu_options.enable_pc_wiener = options.still_pc_wiener;
      gpu_options.enable_gdf = options.still_gdf;
      gpu_options.enable_grain = options.still_grain;
      gpu_options.delta_q = options.still_dc_ac_delta_q;
      gpu_options.enable_segmentation = options.still_segmentation;
      gpu_options.enable_delta_q = options.still_delta_q;
      gpu_options.delta_q_res = options.still_delta_q_res;
      gpu_options.per_segment_q = options.still_per_segment_q;
      gpu_options.per_segment_q_delta = options.still_seg_delta;
      gpu_options.qm = options.still_qm;
      gpu_options.qm_level = options.still_qm_level;
      gpu_options.rdoq = options.still_rdoq;
      gpu_options.rdoq_sse_scale = options.still_rdoq_scale;
      gpu_options.tx_type = options.still_tx_type;
      gpu_options.tx_type_sse_scale = options.still_tx_type_scale;
      gpu_options.idtx = options.still_idtx;
      gpu_options.ist = options.still_ist;
      gpu_options.ist_sse_scale = options.still_ist_scale;
      gpu_options.tcq = options.still_tcq;
      gpu_options.tcq_sse_scale = options.still_tcq_scale;
      gpu_options.cctx = options.still_cctx;
      gpu_options.tx_partition = options.still_tx_partition;
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
                     "from the CPU Still reference (linked stage hooks: "
                  << av2enc::cuda::still_gpu::linked_stage_count() << ")\n";
      }
      if (!write_unit(encoded.frame, 0, error)) {
        return fail_encode(error);
      }
      tile_bytes = encoded.tile_bytes;
      frame_bytes = encoded.frame_bytes;
#else
      return fail_encode("--still-gpu is not available in this build");
#endif
    } else if (multi_tile) {
      // One independent §8.2 entropy stream per tile; every tile is encoded
      // from its own local origin through the Still reference, so CDFs, above/left
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
      tile_config.adaptive_cdf = options.avg_cdf;

      av2enc::still::tiles_stream::Frame tiled;
      std::string tile_error;
      if (!av2enc::still::tiles_stream::encode(picture, tile_config, &tiled,
                                               &tile_error)) {
        return fail_encode("--tiles " + options.tiles + ": " + tile_error);
      }
      if (!write_unit(tiled.frame, 0, error)) {
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
      still_options.ibc = options.still_ibc;
      still_options.ibc_search_step = options.still_ibc_search_step;
      still_options.mhccp = options.still_mhccp;
      still_options.rdo_intra = options.still_rdo_intra;
      still_options.rdo_sse_scale = options.still_rdo_scale;
      still_options.rdoq = options.still_rdoq;
      still_options.rdoq_sse_scale = options.still_rdoq_scale;
      still_options.tx_type = options.still_tx_type;
      still_options.tx_type_sse_scale = options.still_tx_type_scale;
      still_options.enable_cdef = options.still_cdef;
      still_options.cdef_num_strengths = options.still_cdef_strengths;
      still_options.cdef_on_skip_txfm = options.still_cdef_on_skip_txfm;
      still_options.enable_ccso = options.still_ccso;
      still_options.ccso_bands_log2 = options.still_ccso_bands;
      still_options.sb128 = options.still_sb128;
      still_options.sb256 = options.still_sb256;
      still_options.ext_partitions = options.still_ext_partitions;
      still_options.sub8 = options.still_sub8;
      still_options.sdp = options.still_sdp;
      still_options.enable_segmentation = options.still_segmentation;
      still_options.enable_delta_q = options.still_delta_q;
      still_options.delta_q_res = options.still_delta_q_res;
      still_options.per_segment_q = options.still_per_segment_q;
      still_options.per_segment_q_delta = options.still_seg_delta;
      still_options.content_interpretation =
          options.still_content_interpretation;
      still_options.qm = options.still_qm;
      still_options.qm_level = options.still_qm_level;
      still_options.user_qm = options.still_user_qm;
      still_options.ist = options.still_ist;
      still_options.ist_sse_scale = options.still_ist_scale;
      still_options.tcq = options.still_tcq;
      still_options.tcq_sse_scale = options.still_tcq_scale;
      still_options.cctx = options.still_cctx;
      still_options.tx_partition = options.still_tx_partition;
      still_options.idtx = options.still_idtx;
      still_options.ddt = options.still_ddt;
      still_options.delta_q = options.still_dc_ac_delta_q;
      still_options.enable_ibp = options.still_ibp;
      still_options.ccso_edge = options.still_ccso_edge;
      still_options.enable_wiener = options.still_wiener;
      still_options.enable_chroma_lr = options.still_chroma_lr;
      still_options.enable_pc_wiener = options.still_pc_wiener;
      still_options.enable_gdf = options.still_gdf;
      still_options.enable_grain = options.still_grain;
      still_options.grain_adaptive = options.still_grain_adaptive;
      if (options.still_depth >= 0) {
        still_options.max_partition_depth = options.still_depth;
      }

      av2enc::still::StillResult encoded;
      if (!av2enc::still::encode_still_picture(picture, still_options, encoded,
                                               error)) {
        return fail_encode(error);
      }
      if (!write_unit(encoded.frame, 0, error)) {
        return fail_encode(error);
      }
      if (!options.recon.empty()) {
        if (!write_still_recon_y4m(options.recon, y4m,
                                   static_cast<int>(y4m.width),
                                   static_cast<int>(y4m.height),
                                   encoded.recon_y, encoded.recon_u,
                                   encoded.recon_v, error)) {
          return fail_encode(error);
        }
      }
      tile_bytes = encoded.tile_bytes;
      frame_bytes = encoded.frame_bytes;
      ibc_blocks = encoded.ibc_blocks;
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
    // Frozen CDFs are the default Basic bitstream. Adaptive coding is an opt-in
    // normative §8.2.6 mode used to test the CDF-update path.
    config.enable_cdf_adaptation = options.adaptive_cdf;

    av2enc::basic::StillEncodeResult encoded;
    if (!av2enc::basic::encode_still_picture(picture, config, encoded, error)) {
      return fail_encode(error);
    }
    if (!write_unit(encoded.frame, 0, error)) {
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
                 : (options.m1 ? (multi_tile ? "Still (tiled)" : "Still")
                               : (basic_fallback_to_m1 ? "Still (Basic fallback)"
                                                       : "Basic")));
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
  if (ibc_blocks > 0) {
    std::cout << "  ibc:     " << ibc_blocks << " exact-copy block(s)\n";
  }
  return 0;
}

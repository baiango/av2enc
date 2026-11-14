#include "core/bitstream/basic_headers.h"
#include "codec/entropy/lossless/lossless.h"
#include "codec/entropy/lossless/inter/lossless_inter.h"
#include "formats/chroma/chroma_still.h"
#include "formats/format_y4m.h"
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
#include "codec/inter/bframes/bframes.h"
#include "codec/inter/tlayers/tlayers.h"
#include "codec/rate/rc/hrd/hrd.h"
#include "frame/metadata/cli/metadata_cli.h"
#include "frame/metadata/types/metadata_types.h"

#include <charconv>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <sstream>
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
#if __has_include("cuda/pipeline/still/lossless/still_lossless_gpu.h")
#include "cuda/pipeline/still/lossless/still_lossless_gpu.h"
#define AV2ENC_STILL_LOSSLESS_HEADER 1
#else
#define AV2ENC_STILL_LOSSLESS_HEADER 0
#endif

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
  // §5.4.1 high tier on --video (--video-tier high).
  bool video_tier_high = false;
  // §5.4.1/§5.18.2 multi-sequence bank on --video.
  bool video_multi_sequence = false;
  // §5.8/§5.10 full LCR/OPS carrier on --video.
  bool video_lcr_full = false;
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
  // Neutral metadata OBUs (§5.17).  All default off; the payloads are the
  // all-zero/minimal spec-valid forms and do not affect reconstruction.
  bool metadata_timecode = false;
  bool metadata_scan_type = false;
  // §6.16.5 content light level (`--metadata-cll [max:fall]`).
  bool metadata_cll = false;
  std::uint16_t metadata_max_cll = 0;
  std::uint16_t metadata_max_fall = 0;
  // §6.16.6 mastering display colour volume (`--metadata-mdcv [preset]`).
  bool metadata_mdcv = false;
  av2enc::hdr::MasteringDisplayColorVolume metadata_mdcv_value{};
  // §6.16.9 ICC profile (`--metadata-icc [file]`).
  bool metadata_icc = false;
  std::vector<std::uint8_t> metadata_icc_data;
  // §6.16.11 temporal point info (`--metadata-temporal-point [ticks]`).
  bool metadata_temporal_point = false;
  std::uint32_t metadata_frame_presentation_time = 0;
  // §6.16.13 decoded frame hash (`--metadata-frame-hash [per-plane]`).
  bool metadata_frame_hash = false;
  bool metadata_frame_hash_per_plane = false;
  // §6.16.8 banding hints (`--metadata-banding [neutral|source]`).
  bool metadata_banding = false;
  av2enc::metadata::BandingHints metadata_banding_value{};
  // §6.16.12 user data unregistered (`--metadata-user-data [uuid[:hex]]`).
  bool metadata_user_data = false;
  av2enc::metadata::UserDataUnregistered metadata_user_data_value{};
  // §5.4.13/§6.11 decoder model info and buffer removal timing (--video only
  // for the nested block; BRT is carried per temporal unit).
  bool video_decoder_model = false;
  std::uint32_t decoder_buffer_delay = 70000;
  std::uint32_t encoder_buffer_delay = 20000;
  bool low_delay_mode = false;
  bool video_brt = false;
  std::uint32_t brt_time = 0;
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
  // §7.13.3.27/.28 masked compound, opt-in behind --video-compound.
  bool video_compound_wedge = false;
  bool video_compound_diffwtd = false;
  // §5.4.6 / §5.20.7.6 / §7.13.3.14 compound prediction-surface overlays on the
  // NEAR_NEARMV carrier: non-equal CWP weights, the derived implicit mask blend
  // and the §7.13.3.15 / §7.17 sub-PU deblock syntax.  All off by default.
  bool video_cwp_weighted = false;
  bool video_imp_mask = false;
  bool video_df_sub_pu = false;
  // Rejected carrier: §7.13.3.29-.30 inter-intra is a single-reference motion
  // mode (§5.20.7.14 motion_mode_allowed() requires !isCompound).
  bool video_compound_interintra = false;
  bool video_inter_residual = false;
  // Opt-in measured CPU RD preset (--video-rd-preset <name>).  Empty means no
  // expansion, so the default stream stays byte-identical.  The expansion only
  // sets existing CLI options and then the normal validation runs.
  std::string video_rd_preset;
  // Opt-in §5.18.8.1 TX_MODE_SELECT luma TX size/type/partition search.
  bool video_tx_search = false;
  double video_tx_lambda = 256.0;
  // Opt-in §7.17 deblocking of every reconstructed video frame.
  bool video_deblock = false;
  // Opt-in §5.18.7.1/.8 segmentation + per-superblock delta-Q (CPU --video).
  bool video_segmentation = false;
  bool video_delta_q = false;
  int video_delta_q_res = 0;
  bool video_per_segment_q = false;
  int video_seg_delta = 8;
  // Opt-in §5.4.8/§5.18.6.1 separate UV/DC delta-Q (--video-uv-delta-q D).
  bool video_uv_delta_q = false;
  int video_uv_delta_q_value = 4;
  // Opt-in segmentation extensions (--video-seg-skip/--video-seg-globalmv/
  // --video-seg-temporal/--video-segments N).  All default off.
  bool video_seg_skip = false;
  bool video_seg_globalmv = false;
  bool video_seg_temporal = false;
  int video_segments = 0;  // 0 = historical two-segment planner
  bool video_gme = false;
  bool video_warp = false;
  bool video_bawp = false;  // opt-in §7.13.3.25 block adaptive weighted pred
  // Opt-in §5.4.6 enable_ref_frame_mvs + §7.9/§7.12.2.7/.8 temporal MVs.
  bool video_ref_mvs = false;
  bool video_ref_mvs_stress = false;
  // Opt-in §5.4.6 reference-management tools on the CPU --video writer.
  bool video_short_refresh = false;
  bool video_refmvbank = false;
  bool video_mv_traj = false;
  bool video_same_ref_compound = false;
  // Documented rejected carrier: §7.12.2/§7.22 BRU (pinned dav2d does not
  // parse enable_bru/use_bru).
  bool video_bru = false;
  // Documented rejected carriers.  Both tools decode with the pinned dav2d
  // through their component gates, but neither is reachable from the M2
  // inter product path (see the parse errors).
  bool video_morph = false;
  // §7.13.3.8-.12 / §7.13.3.3-.6 temporal refinement on the NEAR_NEARMV
  // compound carrier (CPU 8-bit 4:2:0; gate_video_opfl_refine.sh).
  bool video_opfl = false;
  bool video_refine_mv = false;
  // Opt-in screen-content tools on the CPU `--video` intra keyframe.  Both
  // default off so every historical stream is byte-identical.
  bool video_palette = false;     // §5.20.8 luma palette selection
  bool video_ibc = false;         // §5.20.5.3/.4 general intra block copy
  // §5.4.7 forced screen-content form on the palette/IBC/morph carriers.
  bool video_scc_force = false;
  // §5.4.7 seq_force_integer_mv (requires --video-scc-force).
  bool video_force_integer_mv = false;
  // §5.4.3 reduce_pb_aspect_ratio with max_pb_aspect_ratio_log2_minus_1.
  bool video_reduce_pb_aspect = false;
  // §5.4.4 seq_seg_info_present_flag (--video-seq-seg-info).  The frozen form
  // codes seq_allow_seg_info_change = 0, so every frame reuses the table with
  // no reuse_seg_info bit.
  bool video_seq_seg_info = false;
  bool video_seq_seg_frozen = false;
  int video_pb_aspect_log2_minus_1 = 1;  // 0 -> 2:1, 1 -> 4:1
  bool video_compound_near_near = false;  // §5.20.7.6 NEAR_NEARMV carrier
  bool video_tip = false;
  // A/B control for the §5.4.6 enable_tip_hole_fill bring-up: keeps the
  // pre-fix sequence header (hole fill off) so a gate can compare both masks.
  bool video_tip_no_hole_fill = false;
  // §5.20.7.10 block-level TIP reference carrier.
  bool video_tip_block_ref = false;
  bool video_intra_full = false;  // opt-in 61-candidate luma intra search
  bool video_cfl = false;         // opt-in §7.13.5 CfL chroma decision
  bool video_mhccp = false;       // opt-in §7.13.6-.8 MHCCP (implies CfL)
  bool video_qm = false;          // opt-in default-matrix QM
  bool video_user_qm = false;     // opt-in user-defined QM OBU (implies QM)
  bool video_tcq = false;         // opt-in §5.20.7.27 trellis TCQ
  bool video_cctx = false;        // opt-in §7.14.6 cross-component transform
  bool video_grain = false;       // opt-in §5.14/§5.18.10 film grain signalling
  bool video_lossless = false;    // opt-in Q=0 lossless P-frame writer
  int video_lossless_intra_period = 0;  // 1/N blocks intra (0 = all inter)
  bool video_idtx = false;        // §5.4.8 enable_idtx_intra (2-D IDTX)
  bool video_mrl = false;         // §5.4.5 enable_mrls (directional luma)
  bool video_ibp = false;         // §5.4.5 enable_ibp (intra bi-prediction)
  bool video_dip = false;         // §5.4.5 enable_dip (luma DC alternative)
  bool video_ist = false;         // §5.4.8 enable_intra_ist (luma intra)
  bool video_idtx_1d = false;     // documented rejected 1-D intra carrier
  bool video_sdp = false;         // §5.4.3 enable_sdp intra luma/chroma trees
  bool video_ddt = false;         // §5.4.8 enable_inter_ddt
  bool video_subpel_eighth = false;
  int video_mv_precision = 0;  // 0 = 1/8, 1 = 1/4, 2 = 1/2 pel
  bool video_adaptive_mvd = false;    // §5.4.6 enable_adaptive_mvd
  bool video_mvd_sign_derive = false;  // §5.4.6 enable_mvd_sign_derive
  bool video_flex_mvres = false;       // §5.4.6 enable_flex_mvres
  bool video_drl_reorder = false;
  bool video_drl_reorder_always = false;
  bool video_long_term = false;
  bool video_olk = false;
  int video_leading = 0;  // 0 = off
  bool video_sef = false;
  bool video_switch = false;
  bool video_ras = false;
  int video_tlayers = 0;   // 0 = off; N = one closed hierarchical-P GOP
  bool video_lossless_b = false;  // --video-lossless: use the B schedule
  int video_tlayer_drop = -1;  // -1 = keep every layer
  int video_gops = 1;      // consecutive closed B-frame GOPs (default 1)
  bool video_force_single = false;
  bool video_sb128 = false;
  bool video_sb256 = false;
  bool video_ext_partitions = false;
  int video_min_block = 16;
  int video_max_depth = 4;
  int video_me_range = 16;
  bool video_cdef = false;
  int video_cdef_strength = 1;  // CdefStrengths 1..8
  int video_cdef_damping = 3;   // CdefDamping 3..6
  bool video_ccso = false;
  int video_ccso_bands = 4;  // CcsoMaxBandLog2 (band-offset-only form)
  // §5.4.5 cfl_ds_filter_index (0..3; 3 aliases 0).  Drives the chroma-from-
  // luma taps of the chroma loop restoration.
  int video_cfl_ds_filter = 0;
  bool video_wiener = false;
  bool video_pc_wiener = false;
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
  // Host CPU workers for the CPU reference video encoder's deterministic
  // parallel-for loops (--cpu-threads).  1 (default) is the serial path;
  // 0 means auto (hardware concurrency).  Ignored by the GPU path.
  unsigned cpu_threads = 1;
  bool cpu_threads_given = false;
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

// Formats the format video core has been verified and gated on.  Everything
// else keeps the historical `--video is 8-bit 4:2:0` rejection, by name.
// Order of landing: 10-bit 4:2:0 first, then the remaining 10-bit chroma
// formats, then the 8-bit chroma formats.
bool video_format_wired(int bit_depth, const std::string& chroma) {
  if (bit_depth != 8 && bit_depth != 10) return false;
  return chroma == "420" || chroma == "422" || chroma == "444" ||
         chroma == "400";
}

// Appends the opt-in metadata OBUs as §5.17.2 suffix units at the end of one
// temporal unit.  The payloads are the minimal spec-valid forms (no invented
// timing/hash policy); pinned dav2d ignores OBU types 8/9, so the gate is
// decode invariance.
bool append_neutral_metadata_obus(
    std::vector<std::uint8_t>& unit,
    const av2enc::metadata::cli::Options& metadata,
    const av2enc::metadata::cli::Options::FramePlanes* planes, bool timecode,
    bool scan_type, std::string& error) {
  if (!metadata.any() && !timecode && !scan_type) return true;
  if (!av2enc::metadata::cli::append_suffix_metadata(unit, metadata, planes,
                                                     error)) {
    return false;
  }
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

// Maps the CLI flags to the metadata carrier options.  Everything stays
// unsignalled unless its flag is set.
av2enc::metadata::cli::Options metadata_cli_options(const Options& options) {
  av2enc::metadata::cli::Options metadata;
  metadata.cll = options.metadata_cll;
  metadata.cll_value.max_cll = options.metadata_max_cll;
  metadata.cll_value.max_fall = options.metadata_max_fall;
  metadata.mdcv = options.metadata_mdcv;
  metadata.mdcv_value = options.metadata_mdcv_value;
  metadata.icc = options.metadata_icc;
  metadata.icc_data = options.metadata_icc_data;
  metadata.temporal_point = options.metadata_temporal_point;
  metadata.frame_presentation_time = options.metadata_frame_presentation_time;
  metadata.frame_hash = options.metadata_frame_hash;
  metadata.frame_hash_per_plane = options.metadata_frame_hash_per_plane;
  metadata.banding = options.metadata_banding;
  metadata.banding_value = options.metadata_banding_value;
  metadata.user_data = options.metadata_user_data;
  metadata.user_data_value = options.metadata_user_data_value;
  return metadata;
}

// 8-bit 4:2:0 reconstruction planes for the §6.16.13 hash.
av2enc::metadata::cli::Options::FramePlanes make_plane_view(
    const std::vector<std::uint8_t>& y, const std::vector<std::uint8_t>& u,
    const std::vector<std::uint8_t>& v, std::uint32_t width,
    std::uint32_t height) {
  av2enc::metadata::cli::Options::FramePlanes planes;
  planes.y = std::span<const std::uint8_t>(y.data(), y.size());
  planes.u = std::span<const std::uint8_t>(u.data(), u.size());
  planes.v = std::span<const std::uint8_t>(v.data(), v.size());
  planes.width = width;
  planes.height = height;
  return planes;
}

av2enc::metadata::cli::Options::FramePlanes make_plane_view(
    std::span<const std::uint8_t> y, std::span<const std::uint8_t> u,
    std::span<const std::uint8_t> v, std::uint32_t width,
    std::uint32_t height) {
  av2enc::metadata::cli::Options::FramePlanes planes;
  planes.y = y;
  planes.u = u;
  planes.v = v;
  planes.width = width;
  planes.height = height;
  return planes;
}

// §6.11 buffer removal timing: one non-dependent BRT OBU with br_time in the
// rg(4) range.  It belongs before the frame OBU inside the coded output frame
// unit (§7.3.3), so it is spliced in rather than appended.
bool make_brt_obu(std::uint32_t br_time, std::vector<std::uint8_t>& obu,
                  std::string& error) {
  av2enc::hrd::BufferRemovalTiming brt;
  brt.br_ops_dependent_flag = false;
  brt.br_time = br_time;
  return av2enc::hrd::build_buffer_removal_timing_obu(brt, obu, &error);
}

// Inserts `obu` immediately before the first frame OBU of a temporal unit.
bool insert_before_first_frame_obu(std::vector<std::uint8_t>& unit,
                                   const std::vector<std::uint8_t>& obu,
                                   std::string& error) {
  std::size_t pos = 0;
  while (pos < unit.size()) {
    std::size_t cursor = pos;
    std::uint64_t size = 0;
    unsigned shift = 0;
    while (true) {
      if (cursor >= unit.size()) {
        error = "temporal unit truncated inside an OBU size";
        return false;
      }
      const std::uint8_t byte = unit[cursor++];
      size |= static_cast<std::uint64_t>(byte & 0x7FU) << shift;
      if ((byte & 0x80U) == 0U) break;
      shift += 7U;
      if (shift > 28U) {
        error = "temporal unit has an oversized OBU size";
        return false;
      }
    }
    if (cursor >= unit.size() || cursor + size > unit.size()) {
      error = "temporal unit truncated inside an OBU payload";
      return false;
    }
    const unsigned type = (unit[cursor] >> 2U) & 0x1FU;
    const bool is_frame =
        type == 4U || type == 5U || type == 6U || type == 7U || type == 10U ||
        type == 11U || type == 12U || type == 13U || type == 14U ||
        type == 19U || type == 21U;
    if (is_frame) {
      unit.insert(unit.begin() + static_cast<std::ptrdiff_t>(pos), obu.begin(),
                  obu.end());
      return true;
    }
    pos = cursor + static_cast<std::size_t>(size);
  }
  error = "temporal unit has no frame OBU";
  return false;
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
  item("--still-deblock", {"Opt in to §7.17 deblocking on the format "
                           "(--bit-depth/--chroma) paths (off by default so "
                           "pinned streams are unchanged)"});
  item("--still-full-intra", {"Directional/smooth intra search (SSE-only mode",
                              "decision; DC is the default)"});
  item("--still-uv-full", {"Full chroma intra search (DC/CfL is the default;",
                           "CPU-only)"});
  item("--still-rdo-intra", {"Rate-aware luma intra mode decision (CPU-only)"});
  item("--still-rdoq", {"Full-trellis coefficient RDOQ (CPU-only)"});
  item("--still-tx-type", {"Intra luma TX_TYPE search (CPU-only)"});
  item("--still-cdef", {"CDEF, after deblock (CPU; runs on --still-gpu; the "
                        "8-bit chroma format path is CPU-only)"});
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
  item("--metadata-cll [max:fall]", {"Emit a §6.16.5 content light level OBU",
                                     "(suffix; default 0:0 = not signalled)"});
  item("--metadata-mdcv [preset|vals]", {"Emit a §6.16.6 mastering display",
                                         "colour volume OBU (suffix; default",
                                         "unknown, or bt2020 / 10 integers)"});
  item("--metadata-icc [file]", {"Emit a §6.16.9 ICC profile OBU (suffix;",
                                 "default 132-byte v4.3 header-only profile)"});
  item("--metadata-temporal-point [t]", {"Emit a §6.16.11 temporal point info",
                                         "short OBU (suffix; default 0 ticks)"});
  item("--metadata-frame-hash [mode]", {"Emit a §6.16.13 MD5 decoded frame hash",
                                        "of the reconstruction (suffix;",
                                        "'per-plane' or one frame digest)"});
  item("--metadata-banding [kind]", {"Emit a neutral §6.16.8 banding hints OBU",
                                     "(suffix; default neither coding nor",
                                     "source banding, or 'source')"});
  item("--metadata-user-data [uuid[:hex]]", {"Emit a §6.16.12 user data",
                                              "unregistered OBU (suffix; default",
                                              "zero UUID, empty payload)"});
  item("--video-decoder-model [dbd:ebd:low]", {"Signal the §5.4.13 nested",
                                               "seq_decoder_model_info block",
                                               "(implies --video-timing)"});
  item("--video-brt [tick]", {"Insert a §6.11 buffer removal timing OBU before",
                              "each frame (br_time 0..511, default 0)"});
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
  item("--cpu-threads <N>", {"CPU --video reference worker count: 1 (default,",
                             "serial), 0 = auto, 2..256. Independent tile",
                             "encoders and the batched NEWMV search run in",
                             "parallel; the bitstream is byte-identical"});
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
  item("--video-tier high", {"Signal §5.4.1 seq_tier = 1 (high tier);",
                             "requires level 4.0+, i.e. seq_level_idx > 3"});
  item("--video-multi-sequence", {"Carry a second §5.4.1 sequence header",
                                  "(seq_header_id 1) and select it with",
                                  "seq_header_id_in_frame_header from the key",
                                  "frame onward"});
  item("--video-lcr-full", {"Activate and emit a full §5.8 global/local LCR",
                            "and §5.10/§5.11 global/local OPS before the key",
                            "frame (--video; default: no LCR/OPS)"});
  item("--p-frames <N>", {"Number of P frames (default: all frames after the",
                          "keyframe)"});
  item("--b-frames <N>", {"B frames per closed IPBB GOP, 0..6 (default 0)"});
  item("--video-gops <N>", {"Consecutive closed B-frame GOPs (default 1);",
                            "each GOP starts with a CLK slot reset"});
  item("--video-tlayers <N>", {"Hierarchical P GOP with N = 4 or 8 temporal",
                              "layers; coding order differs from display order"});
  item("--video-tlayer-drop <L>", {"Encode only temporal layers <= L (dropped-",
                                  "layer substream); implies --video-tlayers"});
  item("--video-compound", {"Compound GLOBAL_GLOBALMV/AVERAGE prediction;",
                            "requires --video-refs >= 2"});
  item("--video-compound-wedge", {"Search §7.13.3.27 wedge masks for compound;",
                                  "requires --video-compound"});
  item("--video-compound-diffwtd", {"Search §7.13.3.28 difference-weight masks;",
                                     "requires --video-compound"});
  item("--video-compound-interintra", {"Single-reference §7.13.3.29/.30",
                                        "inter-intra: codes the §5.20.7.14",
                                        "motion mode/interintra_mode/",
                                        "wedge_interintra and blends the inter",
                                        "prediction with an intra prediction",
                                        "(CPU, --video-refs 1)"});
  item("--video-refs <N>", {"References per P frame (NumTotalRefs), 1..7",
                            "(default 1); 2..7 selects per block. Ignored by",
                            "--video-implicit-ref-map, which derives it"});
  item("--video-num-ref-frames <N>", {"§5.4.6 NumRefFrames, 4..8 (default: 2",
                                      "for one reference, 8 for multi-ref)",
                                      "and 8 for --video-implicit-ref-map"});
  item("--video-subpel-eighth", {"Add the 1/8-pel ME refinement stage"});
  item("--video-mv-precision P", {"eighth (default), quarter or half frame MV",
                                  "precision"});
  item("--video-adaptive-mvd", {"Opt-in §5.4.6 enable_adaptive_mvd: NEWMV",
                                 "blocks may code the §5.20.7.20/.21",
                                 "mv_joint/amvd_index MVD form"});
  item("--video-mvd-sign-derive", {"Opt-in §5.4.6 enable_mvd_sign_derive:",
                                    "eligible NEWMV blocks derive the last",
                                    "MVD sign (§5.20.7.13); needs a frame",
                                    "precision of quarter or half pel"});
  item("--video-flex-mvres", {"Opt-in §5.4.6 enable_flex_mvres: each NEWMV",
                               "block may select a coarser MV precision than",
                               "the frame value (§5.20.7.6)"});
  item("--video-inter-residual", {"Code a quantised inter residual"});
  item("--video-rd-preset <name>", {"Expand a measured CPU RD preset:",
                                     "'quality' = --video-inter-residual",
                                     "--video-deblock --video-ccso",
                                     "--video-tx-search --video-intra-full",
                                     "(default off; the flags stay",
                                     "individually available)"});
  item("--video-tx-search", {"Opt-in §5.18.8 TX_MODE_SELECT luma TX size/type/",
                                "partition search (changes the bitstream)"});
  item("--video-tx-lambda <S>", {"SSE-per-bit weight for --video-tx-search",
                                  "(default 256)"});
  item("--video-sdp", {"Opt-in §5.4.3 enable_sdp: the intra keyframe codes",
                       "per-64x64 luma and chroma partition trees",
                       "(CPU-only, 8-bit 4:2:0, single tile)"});
  item("--video-ddt", {"Opt-in §5.4.8 enable_inter_ddt: inter ADST/FDST",
                       "transforms use the DDTX/FDDT kernels selected per",
                       "block (CPU-only, 8-bit 4:2:0)"});
  item("--video-deblock", {"Deblock every reconstructed frame (§7.17) before",
                           "it is emitted or stored as a reference; CPU and",
                           "CUDA paths signal §5.18.7.9 and filter pixels"});
  item("--video-segmentation", {"Opt-in §5.18.7.1 segmentation: code the",
                                "§5.20.5.7/.8 segment id per leaf from a",
                                "per-superblock luma-variance split (ALT_Q 0)"});
  item("--video-delta-q", {"Opt-in §5.18.7.8 delta-Q: per-superblock",
                           "CurrentQIndex updates move the block quantizer"});
  item("--video-delta-q-res <N>", {"§5.18.7.8 delta_q_res 0..3 (default 0)"});
  item("--video-per-segment-q", {"Opt-in real SEG_LVL_ALT_Q: segment 1 (low",
                                 "variance) uses base + --video-seg-delta in",
                                 "the block quantizer and deblock metadata"});
  item("--video-seg-delta <D>", {"Per-segment qindex delta for",
                                 "--video-per-segment-q and --video-delta-q",
                                 "(default 8)"});
  item("--video-uv-delta-q <D>", {"Opt-in §5.4.8/§5.18.6.1 separate UV/DC",
                                  "delta-Q: U uses +D and V uses -D for both DC",
                                  "and AC (separate_uv_delta_q=1,",
                                  "equal_ac_dc_q=0, diff_uv_delta=1)"});
  item("--video-seg-skip", {"Opt-in SEG_LVL_SKIP on the lowest-variance",
                            "segment (implies --video-segmentation)"});
  item("--video-seg-globalmv", {"Opt-in SEG_LVL_GLOBALMV on a segment",
                                "(implies --video-segmentation)"});
  item("--video-seg-temporal", {"Opt-in segmentation_temporal_update on",
                                "P frames whose primary reference is a P frame"});
  item("--video-segments <N>", {"Number of segments, 1..16 (default 2 when",
                                "segmentation is enabled); N > 8 signals",
                                "enable_ext_seg = 1"});
  item("--video-gme", {"Estimate and signal §5.18.9 global motion per",
                       "reference; GLOBALMV uses the projected §7.12.2.2 MV"});
  item("--video-tip", {"Encode 3..8 frames as key + future anchor P +",
                       "TIP_FRAME_AS_OUTPUT pictures (§7.10, §7.13.3.2);",
                       "hole fill (§7.10.3/.4) on; opt-in, CPU reference"}),
      item("--video-tip-block-ref",
           {"§5.20.7.10 TipFrameMode =", "TIP_FRAME_AS_REF: the TIP-referencing tail",
            "P codes RefFrame[0] == TIP_FRAME blocks", "(needs --video-ref-mvs",
            "--video-force-single)"}),
  item("--video-tip-no-hole-fill", {"Gate control: signal",
                                    "enable_tip_hole_fill = 0 on the TIP",
                                    "sequence (pre-fix A/B stream)"}),
  item("--video-warp", {"Predict GLOBALMV blocks with the §7.13.3.19/.20",
                        "block/extended warp when a global model is accepted",
                        "(implies --video-gme)"});
  item("--video-bawp", {"Opt-in §7.13.3.25 block adaptive weighted prediction:",
                        "eligible inter blocks signal use_bawp and adapt the",
                        "luma prediction from the current/previous template",
                        "(fitted form, no chroma BAWP)"});
  item("--video-morph", {"Opt-in §7.13.3.26 morphological prediction: the",
                         "intra keyframe codes screen-content + IBC mode info",
                         "and morph_pred, and the IBC copy is adapted with the",
                         "§7.13.3.25 BAWP template fit (CPU, single tile)"});
  item("--video-palette", {"Opt-in §5.20.8 luma palette on the intra keyframe:",
                           "signals the per-frame screen-content form and codes",
                           "real palette_mode_info()/tokens for selected blocks",
                           "(CPU, single tile)"});
  item("--video-ibc", {"Opt-in §5.20.5.3/.4 general intra block copy on the",
                       "intra keyframe: use_intrabc + BV stack/DRL search",
                       "(CPU, single tile)"});
  item("--video-scc-force", {"Write the §5.4.7 forced screen-content form",
                             "(seq_force_screen_content_tools = 1) instead of",
                             "the per-frame SELECT form: palette/IBC are",
                             "inferred on for every frame. Requires one of",
                             "--video-palette/--video-ibc/--video-morph"});
  item("--video-force-integer-mv", {"With --video-scc-force: write",
                                    "seq_force_integer_mv = 1 so every frame",
                                    "derives FrameMvPrecision = one pel; the",
                                    "inter MVs are quantised to the integer",
                                    "grid and no frame codes the precision",
                                    "pair"});
  item("--video-reduce-pb-aspect", {"Write §5.4.3 reduce_pb_aspect_ratio = 1",
                                    "with MaxPbAspectRatio = 4 (use",
                                    "--video-pb-aspect-2 for 2:1) and run the",
                                    "partition search under that limit"});
  item("--video-pb-aspect-2", {"With --video-reduce-pb-aspect: select",
                               "max_pb_aspect_ratio_log2_minus_1 = 0",
                               "(MaxPbAspectRatio = 2) instead of 4"});
  item("--video-seq-seg-info", {"Write §5.4.4 seq_seg_info_present_flag = 1:",
                                "the sequence header carries the segment",
                                "table (seq_allow_seg_info_change = 1) and",
                                "each frame codes reuse_seg_info or its own",
                                "seg_info(). Requires --video-segmentation"});
  item("--video-seq-seg-info-frozen", {"With --video-seq-seg-info: write",
                                       "seq_allow_seg_info_change = 0 so every",
                                       "frame reuses the sequence table with",
                                       "no reuse_seg_info bit"});
  item("--video-ref-mvs-stress", {"Gate stress mode for --video-ref-mvs:",
                                  "always prefer a temporal NEARMV candidate",
                                  "when one exists (proves the coded stream",
                                  "selects the §7.12.2.7-.8 entries)"});
  item("--video-bru", {"Rejected: the pinned dav2d does not parse",
                       "enable_bru/use_bru (Error parsing frame header);",
                       "see src/pipeline/video/video_bru_reject_gate.sh"});
  item("--video-ref-mvs", {"Opt-in §5.4.6 enable_ref_frame_mvs: store each",
                           "reference's §7.22 motion field and add the §7.9 /",
                           "§7.12.2.7-.8 temporal MV candidates (P refs >= 2,",
                           "B frames); off keeps the historical bitstream"});
  item("--video-short-refresh", {"Opt-in §5.4.6 enable_short_refresh_frame_flags:",
                                  "every P frame codes has_refresh_frame_flags",
                                  "and one frame_to_refresh index instead of the",
                                  "full refresh mask"});
  item("--video-refmvbank", {"Opt-in §5.4.6 enable_refmvbank: insert the",
                             "§7.12.2.21 reference-MV bank candidates at",
                             "step 19 (flat P chain, explicit ref map)"});
  item("--video-mv-traj", {"Opt-in §5.4.6 enable_mv_traj: run the §7.9.2",
                           "trajectory fill and use TrajMv in the derived",
                           "candidates; requires --video-ref-mvs"});
  item("--video-same-ref-compound", {"Opt-in §5.4.6 num_same_ref_compound = 1:",
                                     "compound blocks may use the same",
                                     "reference in both lists; requires",
                                     "--video-compound and two references"});
  item("--video-compound-near-near", {"Opt-in §5.20.7.6 NEAR_NEARMV compound",
                                      "on a two-reference B frame (shifted MVs",
                                      "from the per-list merge stacks); needs",
                                      "--video-compound and --b-frames"});
  item("--video-cwp-weighted", {"Search non-equal §5.20.7.6 CWP weights",
                                 "(enable_cwp + cwp_idx) for NEAR_NEARMV",
                                 "compound blocks; needs",
                                 "--video-compound-near-near and --b-frames"});
  item("--video-imp-mask", {"Enable §5.4.6 enable_imp_msk_bld and apply the",
                             "derived §7.13.3.14 get_mask() blend on eligible",
                             "NEAR_NEARMV compound blocks; needs",
                             "--video-compound-near-near and --b-frames"});
  item("--video-df-sub-pu", {"Code the §5.18.2 allow_df_sub_pu bit and use",
                              "§7.13.3.15 sub-PU edges in §7.17 deblocking;",
                              "CPU --video only, off by default"});
  item("--video-opfl", {"CPU §7.13.3.8-.12 optical-flow refinement on a",
                        "NEAR_NEARMV compound block (codes use_optflow);",
                        "needs --video-compound-near-near and --b-frames"});
  item("--video-refine-mv", {"CPU §7.13.3.3-.6 refine-MV on a NEAR_NEARMV",
                             "compound block (derived, no bit: §5.20.7.17",
                             "makes it non-switchable); needs",
                             "--video-compound-near-near and --b-frames"});
  item("--video-cdef", {"Enable §7.18 CDEF on the reconstructed frame before",
                        "it becomes the output/reference (single strength set)"});
  item("--video-cdef-strength <N>", {"CdefStrengths 1..8 (default 1: no tile",
                                     "symbol); 2..8 codes a per-64x64 index"});
  item("--video-cdef-damping <N>", {"CdefDamping 3..6 (default 3)"});
  item("--video-ccso", {"Enable §7.19 CCSO after CDEF (band-offset only)"});
  item("--video-ccso-bands <N>", {"CcsoMaxBandLog2 0..6 (default 4)"});
  item("--video-wiener", {"Enable §7.20 luma non-separable Wiener restoration"});
  item("--video-pc-wiener", {"Enable §7.20.4 luma PC-Wiener restoration",
                             "(RESTORE_PC_WIENER per-unit on/off)"});
  item("--video-chroma-lr", {"Enable §7.20 chroma loop restoration (U/V)"});
  item("--video-gdf", {"Enable §7.20.5 GDF (luma, header-only mode)"});
  item("--video-cfl-ds-filter <N>", {"§5.4.5 cfl_ds_filter_index 0..3 for the",
                                     "chroma-LR CfL taps (default 0)"});
  item("--video-intra-full", {"Opt-in 61-candidate luma intra search (DC, V, H,",
                              "8 directional modes with angle deltas, SMOOTH,",
                              "SMOOTH_V/H, PAETH); chroma stays DC. Default off",
                              "so the shipped streams stay byte-identical"});
  item("--video-cfl", {"Opt-in §7.13.5 chroma-from-luma (CfL) decision on intra",
                       "leaves: {DC, CfL} with derived alpha. Default off so the",
                       "shipped streams stay byte-identical"});
  item("--video-mhccp", {"Opt-in §7.13.6-.8 multi-hypothesis CfL (MHCCP): signals",
                         "enable_mhccp and searches the three CFL_MULTI directions.",
                         "Implies --video-cfl; default off"});
  item("--video-qm", {"Opt-in §5.18.6.2 using_qmatrix with the default §9.4",
                      "matrices (level 0, no user QM OBU)"});
  item("--video-user-qm", {"Opt-in §5.13 user-defined QM: a type-22",
                           "OBU_QUANTIZATION_MATRIX with an identity all-32",
                           "matrix for level 0 before the key frame (implies",
                           "--video-qm).  Pinned dav2d rejects QM OBUs; gate",
                           "with avmdec"});
  item("--video-tcq", {"Opt-in §5.4.8/§5.20.7.27 trellis-coded quantisation:",
                       "enable_tcq = 1 with choose_tcq_per_frame = 1, a per-frame",
                       "allow_tcq bit, luma TX_CLASS_2D trellis levels and the",
                       "§7.14.4 extra dequant shift.  Default off"});
  item("--video-cctx", {"Opt-in §5.4.8/§7.14.6 cross-component transform:",
                        "enable_cctx = 1, per-block cctx_type on plane 1 and",
                        "the joint U/V dequant rotation.  Default off"});
  item("--video-grain", {"Opt-in §5.14 film grain signalling: sequence header",
                         "film_grain_params_present = 1, a film grain OBU per",
                         "changed model, and §5.18.10.1 per-frame config",
                         "(CPU, 8-bit 4:2:0)"});
  item("--video-lossless", {"Opt-in Q=0 lossless P-frame sequence (implies",
                            "--video): its own sequence/frame headers,",
                            "CodedLossless quantization, DPCM, reversible WHT",
                            "and FSC coefficient syntax (CPU, 8/10-bit",
                            "4:0:0/4:2:0/4:2:2/4:4:4)"});
  item("--video-lossless-b", {"--video-lossless with the hierarchical-B",
                              "schedule: two references and two-sided",
                              "COMPOUND_AVERAGE prediction on interior",
                              "frames (needs --video-tlayers 4 or 8)"});
  item("--video-lossless-intra-period <N>", {"Force one intra block per N in",
                                             "every P/B frame (0 = all inter)"});
  item("--video-idtx", {"Opt-in §5.4.8 enable_idtx_intra with the 2-D",
                        "identity TX type (9) in the --video-tx-search",
                        "candidate set"});
  item("--video-mrl", {"Opt-in §5.4.5 enable_mrls: directional luma intra",
                       "searches mrl_index 1..3 (and the secondary line) with",
                       "the §7.13.2.1 extra reference rows/columns.  Needs",
                       "--video-intra-full for a non-zero mrl_index; CPU only"});
  item("--video-ibp", {"Opt-in §5.4.5 enable_ibp: the intra keyframe and every",
                       "intra leaf use §7.13.2 intra bi-prediction (modified",
                       "edge filtering, the two-angle blend and the",
                       "§7.13.2.12 DC post-filter).  There is no per-block",
                       "IBP bit; the decoder derives the decision from mode",
                       "and angle_delta.  CPU only; default off"});
  item("--video-dip", {"Opt-in §5.4.5 enable_dip: luma DC_PRED intra",
                       "keyframe blocks code §5.20.7.9 use_dip /",
                       "dip_transpose / dip_mode and use §7.13.2.3 data-driven",
                       "prediction.  CPU only, 8-bit 4:2:0"});
  item("--video-ist", {"Opt-in §5.4.8 enable_intra_ist: luma intra residuals",
                       "code sec_tx_type and the §7.15.3 secondary transform",
                       "(inter IST stays unsignalled).  CPU only"});
  item("--video-idtx-1d", {"Rejected: the one-axis V/H_DCT..V/H_FLIPADST",
                           "types need dc_sign_horz_vert and the",
                           "class-specific scans, and pinned dav2d rejects",
                           "negative H_DCT/V_DCT under --strict 1"});
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
  item("--video-implicit-ref-map", {"§7.7/§7.8 implicit reference map:",
                                    "sequence explicit_ref_frame_map = 0,",
                                    "NumTotalRefs/ref_frame_idx derived from",
                                    "the slot state (flat P, B frames,",
                                    "--video-tlayers)"});
  item("--video-drl-reorder", {"§5.4.6 DRL_REORDER_CONSTRAINT: enable the",
                               "§7.12.2.19 MV-stack reorder (see",
                               "--video-drl-reorder-always)"});
  item("--video-drl-reorder-always", {"DRL_REORDER_ALWAYS; implies the reorder",
                                      "flag and sorts every block with at",
                                      "least two nearest candidates"});
  item("--rc <mode>", {"Per-frame rate control: cqp|crf|vbr (default cqp, the",
                       "fixed --qp stream)"});
  item("--crf <V>", {"--rc crf quality 0..63 (default 32); owns base_q_idx",
                     "(drop --qp, which only anchors the VBR pass)"});
  item("--bitrate <kbps>", {"--rc vbr target bitrate; VBR runs a statistics",
                            "pass at --qp, then plans one base_q_idx per frame"});
  item("--svc-layers <spec>", {"SVC: spatial layers as x0=WxH[,x1=WxH...]"});
  item("--svc-lcr-id <N>", {"SVC: seq_lcr_id (1..7)"});
  item("--svc-temporal <N>", {"SVC: temporal layers 1..4; N > 1 codes one",
                              "hierarchical GOP of 2^N frames"});
  item("--svc-embedded <N>", {"SVC: embedded quality layers 1..3; layer m",
                              "predicts from m-1 of the same spatial layer"});
  item("--svc-inter-layer", {"Rejected: §6.8.2 requires",
                             "lcr_dependent_xlayers_flag = 0"});
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
  item("--video-sb128", {"Video: 128x128 superblocks (§5.4.3",
                         "use_128x128_superblock); leaves stay <= 64x64"});
  item("--video-sb256", {"Video: 256x256 superblocks (§5.4.3",
                         "use_256x256_superblock); leaves stay <= 64x64"});
  item("--video-ext-partitions", {"Video: extended/uneven 4-way partitions",
                                  "(§5.4.3 enable_ext_partitions); leaves",
                                  "stay >= 8x8 and own their chroma"});
  item("--video-min-block <N>",
       {"Video: minimum partition block size, 4..64 (default 16); 4 enables",
        "§5.20.3.1 sub-8 keyframe leaves with shared-chroma ownership"});
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

// Expansion of the measured video RD presets (--video-rd-preset <name>).  The
// preset only sets existing CLI options; the caller then runs the normal
// combination validation, so a preset cannot reach a tool pairing that the
// individual flags reject.  No preset leaves every field untouched, so the
// historical default stream is byte-identical.
//
// The 'quality' row is measured on the CPU --video reference at the 240p30 /
// 100 kbps point (see docs/video-rd-preset.md): a single tile, P-only chain.
bool apply_video_rd_preset(Options& options, std::string& error) {
  if (options.video_rd_preset.empty()) return true;
  if (options.video_rd_preset != "quality") {
    error = "--video-rd-preset expects 'quality' (the only measured row)";
    return false;
  }
  if (!options.video) {
    error = "--video-rd-preset applies to the CPU --video encoder; pass "
            "--video (the --still writer pins its own tool set)";
    return false;
  }
  if (options.video_gpu) {
    error = "--video-rd-preset is the measured CPU tool set; drop --video-gpu "
            "(its decisions are calibrated separately)";
    return false;
  }
  options.video_inter_residual = true;
  options.video_deblock = true;
  options.video_ccso = true;
  options.video_tx_search = true;
  options.video_intra_full = true;
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

    if (argument == "--video-decoder-model") {
      // §5.4.13 nested block; also needs the §5.4.1 timing block.
      options.video_timing = true;
      options.video_decoder_model = true;
      if (i + 1 < argc && argv[i + 1][0] != '-') {
        const std::string_view value = argv[++i];
        const std::size_t first = value.find(':');
        const std::size_t second = first == std::string_view::npos
                                       ? std::string_view::npos
                                       : value.find(':', first + 1);
        std::uint64_t delay = 0;
        std::uint64_t edelay = 0;
        std::uint64_t low = 0;
        if (first == std::string_view::npos ||
            second == std::string_view::npos ||
            !parse_uint64(value.substr(0, first), delay) ||
            !parse_uint64(value.substr(first + 1, second - first - 1),
                          edelay) ||
            !parse_uint64(value.substr(second + 1), low) || low > 1 ||
            delay > 0xFFFFFFFFu || edelay > 0xFFFFFFFFu) {
          error =
              "--video-decoder-model expects decoder_delay:encoder_delay:"
              "low_delay_flag";
          return ParseStatus::kError;
        }
        options.decoder_buffer_delay = static_cast<std::uint32_t>(delay);
        options.encoder_buffer_delay = static_cast<std::uint32_t>(edelay);
        options.low_delay_mode = low != 0;
      }
      continue;
    }

    if (argument == "--video-brt") {
      options.video_brt = true;
      // Optional non-dependent br_time in rg(4) DecCT ticks, 0..511.
      if (i + 1 < argc && argv[i + 1][0] != '-') {
        std::uint64_t tick = 0;
        if (!parse_uint64(argv[++i], tick) || tick > 511) {
          error = "--video-brt expects a br_time in 0..511";
          return ParseStatus::kError;
        }
        options.brt_time = static_cast<std::uint32_t>(tick);
      }
      continue;
    }

    if (argument == "--video-mfh") {
      options.video_mfh = true;
      continue;
    }

    if (argument == "--video-tier") {
      if (i + 1 >= argc) {
        error = "missing value after '--video-tier'";
        return ParseStatus::kError;
      }
      const std::string_view value = argv[++i];
      if (value == "high") {
        options.video_tier_high = true;
      } else if (value == "main" || value == "0") {
        options.video_tier_high = false;
      } else {
        error = "--video-tier expects 'high' or 'main'";
        return ParseStatus::kError;
      }
      continue;
    }

    if (argument == "--video-multi-sequence") {
      options.video_multi_sequence = true;
      continue;
    }

    if (argument == "--video-lcr-full") {
      options.video_lcr_full = true;
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

    if (argument == "--cpu-threads") {
      if (i + 1 >= argc) {
        error = "missing value after '--cpu-threads'";
        return ParseStatus::kError;
      }
      int value = 0;
      if (!parse_int(argv[++i], value) || value < 0 || value > 256) {
        error = "--cpu-threads must be an integer in [0, 256] "
                "(0 = auto, 1 = serial)";
        return ParseStatus::kError;
      }
      options.cpu_threads = static_cast<unsigned>(value);
      options.cpu_threads_given = true;
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
    if (argument == "--still-deblock") {
      options.still_deblock = true;
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

    if (argument == "--metadata-cll") {
      options.metadata_cll = true;
      // Optional value max_cll:max_fall; the no-argument form is the neutral
      // 0:0 ("no upper bound signalled") unit.
      if (i + 1 < argc && argv[i + 1][0] != '-') {
        const std::string_view value = argv[++i];
        const std::size_t colon = value.find(':');
        std::uint64_t max_cll = 0;
        std::uint64_t max_fall = 0;
        if (colon == std::string_view::npos ||
            !parse_uint64(value.substr(0, colon), max_cll) ||
            !parse_uint64(value.substr(colon + 1), max_fall) ||
            max_cll > 65535 || max_fall > 65535) {
          error =
              "--metadata-cll expects max_cll:max_fall with each value in "
              "0..65535";
          return ParseStatus::kError;
        }
        options.metadata_max_cll = static_cast<std::uint16_t>(max_cll);
        options.metadata_max_fall = static_cast<std::uint16_t>(max_fall);
      }
      continue;
    }

    if (argument == "--metadata-mdcv") {
      options.metadata_mdcv = true;
      // No value = all-zero "unknown/unspecified" (every §6.16.6 range is
      // defined to mean that).  `bt2020` is the ST 2086 BT.2020/D65 preset;
      // otherwise ten comma-separated integers
      //   gx,gy,bx,by,rx,ry,wx,wy,luminance_max,luminance_min
      if (i + 1 < argc && argv[i + 1][0] != '-') {
        const std::string_view value = argv[++i];
        if (value == "bt2020") {
          auto& m = options.metadata_mdcv_value;
          m.primary_chromaticity_x = {8500, 6550, 35400};
          m.primary_chromaticity_y = {39850, 2300, 14600};
          m.white_point_chromaticity_x = 15635;
          m.white_point_chromaticity_y = 16450;
          m.luminance_max = 10000000;
          m.luminance_min = 50;
        } else {
          std::uint64_t fields[10] = {};
          std::size_t count = 0;
          std::size_t start = 0;
          while (count < 10) {
            const std::size_t comma = value.find(',', start);
            const std::string_view item =
                value.substr(start, comma == std::string_view::npos
                                        ? std::string_view::npos
                                        : comma - start);
            if (!parse_uint64(item, fields[count])) {
              error =
                  "--metadata-mdcv expects a preset name or ten comma-separated "
                  "integers (gx,gy,bx,by,rx,ry,wx,wy,max,min)";
              return ParseStatus::kError;
            }
            ++count;
            if (comma == std::string_view::npos) break;
            start = comma + 1;
          }
          if (count != 10) {
            error =
                "--metadata-mdcv expects a preset name or ten comma-separated "
                "integers (gx,gy,bx,by,rx,ry,wx,wy,max,min)";
            return ParseStatus::kError;
          }
          for (int k = 0; k < 8; ++k) {
            if (fields[k] > 65535) {
              error = "--metadata-mdcv chromaticities must fit 16 bits";
              return ParseStatus::kError;
            }
          }
          if (fields[8] > 0xFFFFFFFFu || fields[9] > 0xFFFFFFFFu) {
            error = "--metadata-mdcv luminances must fit 32 bits";
            return ParseStatus::kError;
          }
          auto& m = options.metadata_mdcv_value;
          m.primary_chromaticity_x = {static_cast<std::uint16_t>(fields[0]),
                                      static_cast<std::uint16_t>(fields[2]),
                                      static_cast<std::uint16_t>(fields[4])};
          m.primary_chromaticity_y = {static_cast<std::uint16_t>(fields[1]),
                                      static_cast<std::uint16_t>(fields[3]),
                                      static_cast<std::uint16_t>(fields[5])};
          m.white_point_chromaticity_x = static_cast<std::uint16_t>(fields[6]);
          m.white_point_chromaticity_y = static_cast<std::uint16_t>(fields[7]);
          m.luminance_max = static_cast<std::uint32_t>(fields[8]);
          m.luminance_min = static_cast<std::uint32_t>(fields[9]);
        }
      }
      continue;
    }

    if (argument == "--metadata-icc") {
      options.metadata_icc = true;
      options.metadata_icc_data = av2enc::metadata::cli::default_icc_profile();
      // Optional path to a real profile.  A profile header is 128 bytes and
      // avmdec rejects shorter payloads, so that is the minimum accepted here.
      if (i + 1 < argc && argv[i + 1][0] != '-') {
        const std::string path = argv[++i];
        std::ifstream input(path, std::ios::binary);
        if (!input) {
          error = "--metadata-icc cannot open '" + path + "'";
          return ParseStatus::kError;
        }
        input.seekg(0, std::ios::end);
        const std::streamoff size = input.tellg();
        input.seekg(0, std::ios::beg);
        if (size < 128) {
          error = "--metadata-icc profile must be at least 128 bytes";
          return ParseStatus::kError;
        }
        options.metadata_icc_data.assign(static_cast<std::size_t>(size), 0);
        if (!input.read(reinterpret_cast<char*>(options.metadata_icc_data.data()),
                        size)) {
          error = "--metadata-icc cannot read '" + path + "'";
          return ParseStatus::kError;
        }
      }
      continue;
    }

    if (argument == "--metadata-temporal-point") {
      options.metadata_temporal_point = true;
      // Optional uleb128 value 0..2^32-1; no argument emits the neutral 0.
      if (i + 1 < argc && argv[i + 1][0] != '-') {
        std::uint64_t ticks = 0;
        if (!parse_uint64(argv[++i], ticks) || ticks > 0xFFFFFFFFu) {
          error = "--metadata-temporal-point expects a value in 0..2^32-1";
          return ParseStatus::kError;
        }
        options.metadata_frame_presentation_time =
            static_cast<std::uint32_t>(ticks);
      }
      continue;
    }

    if (argument == "--metadata-frame-hash") {
      options.metadata_frame_hash = true;
      // Optional "per-plane" (three plane digests); the default is one MD5
      // over Y+U+V in plane order.
      if (i + 1 < argc && argv[i + 1][0] != '-') {
        const std::string_view value = argv[++i];
        if (value == "per-plane") {
          options.metadata_frame_hash_per_plane = true;
        } else if (value == "frame") {
          options.metadata_frame_hash_per_plane = false;
        } else {
          error = "--metadata-frame-hash expects 'per-plane' or 'frame'";
          return ParseStatus::kError;
        }
      }
      continue;
    }

    if (argument == "--metadata-banding") {
      options.metadata_banding = true;
      // Neutral by default (1-byte unit, both presence flags 0).  `source`
      // sets source_banding_present_flag and still codes no extra syntax.
      if (i + 1 < argc && argv[i + 1][0] != '-') {
        const std::string_view value = argv[++i];
        if (value == "neutral") {
          options.metadata_banding_value = {};
        } else if (value == "source") {
          options.metadata_banding_value = {};
          options.metadata_banding_value.source_banding_present_flag = true;
        } else {
          error = "--metadata-banding expects 'neutral' or 'source'";
          return ParseStatus::kError;
        }
      }
      continue;
    }

    if (argument == "--metadata-user-data") {
      options.metadata_user_data = true;
      options.metadata_user_data_value = {};
      // Optional "uuid[:hex-payload]".  The UUID is 32 hex digits (dashes are
      // ignored); no argument emits the zero UUID with an empty payload.
      if (i + 1 < argc && argv[i + 1][0] != '-') {
        const std::string_view value = argv[++i];
        const auto hex_value = [](char c) -> int {
          if (c >= '0' && c <= '9') return c - '0';
          if (c >= 'a' && c <= 'f') return c - 'a' + 10;
          if (c >= 'A' && c <= 'F') return c - 'A' + 10;
          return -1;
        };
        const std::size_t colon = value.find(':');
        const std::string_view uuid_text = value.substr(0, colon);
        std::string uuid_hex;
        for (const char c : uuid_text) {
          if (c != '-') uuid_hex.push_back(c);
        }
        if (uuid_hex.size() != 32) {
          error = "--metadata-user-data UUID must be 32 hex digits";
          return ParseStatus::kError;
        }
        for (std::size_t k = 0; k < 16; ++k) {
          const int hi = hex_value(uuid_hex[k * 2]);
          const int lo = hex_value(uuid_hex[k * 2 + 1]);
          if (hi < 0 || lo < 0) {
            error = "--metadata-user-data UUID must be hexadecimal";
            return ParseStatus::kError;
          }
          options.metadata_user_data_value.uuid[k] =
              static_cast<std::uint8_t>((hi << 4) | lo);
        }
        if (colon != std::string_view::npos) {
          const std::string_view hex = value.substr(colon + 1);
          if (hex.size() % 2 != 0) {
            error = "--metadata-user-data payload must be byte-aligned hex";
            return ParseStatus::kError;
          }
          for (std::size_t k = 0; k < hex.size(); k += 2) {
            const int hi = hex_value(hex[k]);
            const int lo = hex_value(hex[k + 1]);
            if (hi < 0 || lo < 0) {
              error = "--metadata-user-data payload must be hexadecimal";
              return ParseStatus::kError;
            }
            options.metadata_user_data_value.payload.push_back(
                static_cast<std::uint8_t>((hi << 4) | lo));
          }
        }
      }
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

    if (argument == "--video-gops") {
      if (i + 1 >= argc) {
        error = "missing value after '--video-gops'";
        return ParseStatus::kError;
      }
      int value = 0;
      if (!parse_int(argv[++i], value) || value <= 0) {
        error = "--video-gops must be a positive integer";
        return ParseStatus::kError;
      }
      options.video_gops = value;
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

    if (argument == "--video-adaptive-mvd") {
      options.video_adaptive_mvd = true;
      continue;
    }

    if (argument == "--video-mvd-sign-derive") {
      options.video_mvd_sign_derive = true;
      continue;
    }

    if (argument == "--video-flex-mvres") {
      options.video_flex_mvres = true;
      continue;
    }

    if (argument == "--video-compound") {
      options.video_compound = true;
      continue;
    }

    if (argument == "--video-compound-wedge") {
      options.video_compound_wedge = true;
      continue;
    }

    if (argument == "--video-compound-diffwtd") {
      options.video_compound_diffwtd = true;
      continue;
    }

    if (argument == "--video-cwp-weighted") {
      options.video_cwp_weighted = true;
      continue;
    }

    if (argument == "--video-imp-mask") {
      options.video_imp_mask = true;
      continue;
    }

    if (argument == "--video-df-sub-pu") {
      options.video_df_sub_pu = true;
      continue;
    }

    if (argument == "--video-compound-interintra") {
      options.video_compound_interintra = true;
      continue;
    }

    if (argument == "--video-inter-residual") {
      options.video_inter_residual = true;
      continue;
    }

    if (argument == "--video-rd-preset") {
      if (i + 1 >= argc) {
        error = "missing value after '--video-rd-preset'";
        return ParseStatus::kError;
      }
      options.video_rd_preset = argv[++i];
      continue;
    }

    if (argument == "--video-tx-search") {
      options.video_tx_search = true;
      continue;
    }

    if (argument == "--video-sdp") {
      options.video_sdp = true;
      continue;
    }

    if (argument == "--video-ddt") {
      options.video_ddt = true;
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

    if (argument == "--video-segmentation") {
      options.video_segmentation = true;
      continue;
    }

    if (argument == "--video-delta-q") {
      options.video_delta_q = true;
      continue;
    }

    if (argument == "--video-delta-q-res") {
      if (i + 1 >= argc ||
          !parse_int(argv[++i], options.video_delta_q_res) ||
          options.video_delta_q_res < 0 || options.video_delta_q_res > 3) {
        error = "--video-delta-q-res must be an integer in [0, 3]";
        return ParseStatus::kError;
      }
      continue;
    }

    if (argument == "--video-per-segment-q") {
      options.video_per_segment_q = true;
      continue;
    }

    if (argument == "--video-seg-delta") {
      if (i + 1 >= argc || !parse_int(argv[++i], options.video_seg_delta) ||
          options.video_seg_delta < -351 || options.video_seg_delta > 351) {
        error = "--video-seg-delta must be an integer in [-351, 351]";
        return ParseStatus::kError;
      }
      continue;
    }

    if (argument == "--video-uv-delta-q") {
      if (i + 1 >= argc ||
          !parse_int(argv[++i], options.video_uv_delta_q_value) ||
          options.video_uv_delta_q_value < -63 ||
          options.video_uv_delta_q_value > 63) {
        error = "--video-uv-delta-q must be an integer in [-63, 63]";
        return ParseStatus::kError;
      }
      options.video_uv_delta_q = true;
      continue;
    }

    if (argument == "--video-seg-skip") {
      options.video_seg_skip = true;
      options.video_segmentation = true;
      continue;
    }

    if (argument == "--video-seg-globalmv") {
      options.video_seg_globalmv = true;
      options.video_segmentation = true;
      continue;
    }

    if (argument == "--video-seg-temporal") {
      options.video_seg_temporal = true;
      options.video_segmentation = true;
      continue;
    }

    if (argument == "--video-segments") {
      if (i + 1 >= argc || !parse_int(argv[++i], options.video_segments) ||
          options.video_segments < 1 || options.video_segments > 16) {
        error = "--video-segments must be an integer in [1, 16]";
        return ParseStatus::kError;
      }
      if (options.video_segments > 1) options.video_segmentation = true;
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

    if (argument == "--video-bawp") {
      options.video_bawp = true;
      continue;
    }

    if (argument == "--video-ref-mvs") {
      options.video_ref_mvs = true;
      continue;
    }

    if (argument == "--video-ref-mvs-stress") {
      options.video_ref_mvs = true;
      options.video_ref_mvs_stress = true;
      continue;
    }

    if (argument == "--video-short-refresh") {
      options.video_short_refresh = true;
      continue;
    }

    if (argument == "--video-refmvbank") {
      options.video_refmvbank = true;
      continue;
    }

    if (argument == "--video-mv-traj") {
      options.video_mv_traj = true;
      continue;
    }

    if (argument == "--video-same-ref-compound") {
      options.video_same_ref_compound = true;
      continue;
    }

    if (argument == "--video-bru") {
      options.video_bru = true;
      continue;
    }

    if (argument == "--video-morph") {
      options.video_morph = true;
      continue;
    }

    if (argument == "--video-palette") {
      options.video_palette = true;
      continue;
    }

    if (argument == "--video-ibc") {
      options.video_ibc = true;
      continue;
    }

    if (argument == "--video-scc-force") {
      options.video_scc_force = true;
      continue;
    }

    if (argument == "--video-force-integer-mv") {
      options.video_force_integer_mv = true;
      continue;
    }

    if (argument == "--video-reduce-pb-aspect") {
      options.video_reduce_pb_aspect = true;
      options.video_pb_aspect_log2_minus_1 = 1;  // 4:1
      continue;
    }

    if (argument == "--video-pb-aspect-2") {
      options.video_reduce_pb_aspect = true;
      options.video_pb_aspect_log2_minus_1 = 0;  // 2:1
      continue;
    }

    if (argument == "--video-seq-seg-info") {
      options.video_seq_seg_info = true;
      options.video_seq_seg_frozen = false;
      continue;
    }

    if (argument == "--video-seq-seg-info-frozen") {
      options.video_seq_seg_info = true;
      options.video_seq_seg_frozen = true;
      continue;
    }

    if (argument == "--video-opfl") {
      options.video_opfl = true;
      continue;
    }

    if (argument == "--video-refine-mv") {
      options.video_refine_mv = true;
      continue;
    }

    if (argument == "--video-compound-near-near") {
      options.video_compound_near_near = true;
      continue;
    }

    if (argument == "--video-tip") {
      options.video_tip = true;
      continue;
    }

    if (argument == "--video-tip-no-hole-fill") {
      options.video_tip_no_hole_fill = true;
      continue;
    }

    if (argument == "--video-tip-block-ref") {
      options.video_tip_block_ref = true;
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

    if (argument == "--video-pc-wiener") {
      options.video_pc_wiener = true;
      continue;
    }

    if (argument == "--video-cfl-ds-filter") {
      if (i + 1 >= argc) {
        error = "missing value after '--video-cfl-ds-filter'";
        return ParseStatus::kError;
      }
      int index = 0;
      if (!parse_int(argv[++i], index) || index < 0 || index > 3) {
        error = "--video-cfl-ds-filter must be an integer in [0, 3]";
        return ParseStatus::kError;
      }
      options.video_cfl_ds_filter = index;
      continue;
    }

    if (argument == "--video-cfl") {
      options.video_cfl = true;
      continue;
    }

    if (argument == "--video-qm") {
      options.video_qm = true;
      continue;
    }

    if (argument == "--video-user-qm") {
      options.video_qm = true;
      options.video_user_qm = true;
      continue;
    }

    if (argument == "--video-tcq") {
      options.video_tcq = true;
      continue;
    }

    if (argument == "--video-cctx") {
      options.video_cctx = true;
      continue;
    }

    if (argument == "--video-grain") {
      options.video_grain = true;
      continue;
    }

    if (argument == "--video-lossless") {
      options.video = true;
      options.video_lossless = true;
      continue;
    }

    if (argument == "--video-lossless-b") {
      options.video = true;
      options.video_lossless = true;
      options.video_lossless_b = true;
      continue;
    }

    if (argument == "--video-lossless-intra-period") {
      if (i + 1 >= argc) {
        error = "missing value after '--video-lossless-intra-period'";
        return ParseStatus::kError;
      }
      int value = 0;
      if (!parse_int(argv[++i], value) || value < 0) {
        error = "--video-lossless-intra-period must be a non-negative integer";
        return ParseStatus::kError;
      }
      options.video_lossless_intra_period = value;
      continue;
    }

    if (argument == "--video-idtx") {
      options.video_idtx = true;
      continue;
    }

    if (argument == "--video-mrl") {
      options.video_mrl = true;
      continue;
    }

    if (argument == "--video-ibp") {
      options.video_ibp = true;
      continue;
    }

    if (argument == "--video-dip") {
      options.video_dip = true;
      continue;
    }

    if (argument == "--video-ist") {
      options.video_ist = true;
      continue;
    }

    if (argument == "--video-idtx-1d") {
      options.video_idtx_1d = true;
      continue;
    }

    if (argument == "--video-mhccp") {
      options.video_mhccp = true;
      options.video_cfl = true;
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

    if (argument == "--video-sb128") {
      options.video_sb128 = true;
      continue;
    }

    if (argument == "--video-sb256") {
      options.video_sb256 = true;
      continue;
    }

    if (argument == "--video-ext-partitions") {
      options.video_ext_partitions = true;
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

    if (argument == "--svc-inter-layer") {
      // Named rejection: AV2 v1.0.0 cannot carry inter-layer prediction.
      error =
          "--svc-inter-layer is not a conforming AV2 v1.0.0 feature: §6.8.2 "
          "requires lcr_dependent_xlayers_flag = 0, so no cross-xlayer "
          "reference or resampling can be signalled and there is no decoder "
          "process to gate.  Repro: av2enc in.y4m -o out.ivf --video --frames "
          "3 --svc-layers 0=640x360,1=1280x720 --svc-inter-layer.  Use "
          "--svc-embedded N for cross-mlayer (quality) refinement";
      return ParseStatus::kError;
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

  if (!apply_video_rd_preset(options, error)) {
    return ParseStatus::kError;
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

  if (options.cpu_threads_given) {
    if (!options.video) {
      error = "--cpu-threads applies to the CPU --video reference encoder";
      return ParseStatus::kError;
    }
    if (options.video_gpu) {
      error = "--cpu-threads is a CPU reference knob; the GPU backend "
              "(--video-gpu) has its own scheduling";
      return ParseStatus::kError;
    }
  }

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
    if ((options.bit_depth != 8 || options.chroma != "420") &&
        !video_format_wired(options.bit_depth, options.chroma)) {
      error = "--video supports 8-bit 4:2:0 and 10-bit 4:2:0; the remaining "
              "--bit-depth/--chroma combinations are not verified on the "
              "video path yet";
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
    // --video-drl-reorder is decoder-gated: the sequence header codes
    // disable_drl_reorder = 0 / constrain_drl_reorder = 1.  The pinned dav2d
    // suppresses the §7.12.2.19 sort for a single reference within two POCs
    // (its high-priority-TMVP term is not gated by use_ref_frame_mvs), so the
    // CONSTRAINT mode is bit-exact but inert on the low-delay P surface;
    // --video-drl-reorder-always exercises the transform.  Both are pinned by
    // src/pipeline/video/gate_video_drl_reorder.sh.
    if (options.video_morph) {
      // §5.20.5.4/§7.13.3.26: the morph carrier signals the screen-content +
      // IBC mode-info path on the intra keyframe, so it needs a single-tile
      // keyframe, frozen per-tile CDFs and no loop filters (the IBC copy and
      // morph fit read the unfiltered reconstruction).  Every other composition
      // is rejected by name instead of being silently ignored.
      if (options.video_gpu) {
        error = "--video-morph is CPU-only: the CUDA M2 backend has no IBC "
                "mode-info path";
        return ParseStatus::kError;
      }
      if (options.b_frames != 0 || options.video_tlayers != 0 ||
          options.video_tip || options.video_long_term || options.video_olk ||
          options.video_leading != 0 || options.video_sef ||
          options.video_switch || options.video_ras) {
        error = "--video-morph is gated on the plain key+P chain: it cannot "
                "be combined with --b-frames, --video-tlayers, --video-tip or "
                "the frame-type wrapper modes";
        return ParseStatus::kError;
      }
      if (options.tiles != "1x1") {
        error = "--video-morph is gated on a single tile";
        return ParseStatus::kError;
      }
      if (options.video_deblock || options.video_cdef || options.video_ccso ||
          options.video_wiener || options.video_pc_wiener ||
          options.video_chroma_lr || options.video_gdf) {
        error = "--video-morph does not compose with the loop filters: the "
                "IBC copy/morph template must read the unfiltered "
                "reconstruction";
        return ParseStatus::kError;
      }
      if (options.avg_cdf || options.cdf_continuity) {
        error = "--video-morph uses the per-tile §9.3 IBC/morph CDFs; it is "
                "not wired with --avg-cdf/--cdf-continuity";
        return ParseStatus::kError;
      }
      if (options.video_drl_reorder_always) {
        error = "--video-morph codes use_intrabc and §7.12.2.19 "
                "DRL_REORDER_ALWAYS reorders the decoder's BV stack, which "
                "the IBC stack builder does not implement (CONSTRAINT is "
                "exact)";
        return ParseStatus::kError;
      }
      if (options.video_segmentation || options.video_per_segment_q ||
          options.video_uv_delta_q) {
        error = "--video-morph is not wired with "
                "--video-segmentation/--video-per-segment-q/--video-uv-delta-q";
        return ParseStatus::kError;
      }
    }
    // §5.20.8 palette / §5.20.5.3-.4 general IBC on the intra keyframe.  Both
    // share the single-tile screen-content carrier and are rejected by name on
    // surfaces where their cache/ring or CDF state would not match the
    // decoder.
    if (options.video_palette || options.video_ibc) {
      const char* tool = options.video_palette ? "--video-palette"
                                               : "--video-ibc";
      if (options.video_gpu) {
        error = std::string(tool) +
                " is CPU-only: the CUDA M2 backend has no screen-content "
                "mode-info path";
        return ParseStatus::kError;
      }
      if (options.video_lossless) {
        error = std::string(tool) +
                " is not applied by the --video-lossless writer";
        return ParseStatus::kError;
      }
      if (options.tiles != "1x1") {
        error = std::string(tool) + " is gated on a single tile";
        return ParseStatus::kError;
      }
      if (options.b_frames != 0 || options.video_tlayers != 0 ||
          options.video_tip || options.video_long_term || options.video_olk ||
          options.video_leading != 0 || options.video_sef ||
          options.video_switch || options.video_ras) {
        error = std::string(tool) +
                " is gated on the plain key+P chain: it cannot be combined "
                "with --b-frames, --video-tlayers, --video-tip or the "
                "frame-type wrapper modes";
        return ParseStatus::kError;
      }
      if (options.bit_depth != 8 || options.chroma != "420") {
        error = std::string(tool) +
                " is gated on 8-bit 4:2:0 (the format video core has no "
                "screen-content syntax)";
        return ParseStatus::kError;
      }
      if (options.avg_cdf || options.cdf_continuity) {
        error = std::string(tool) +
                " uses per-frame §9.3 palette/IBC CDFs; it is not wired with "
                "--avg-cdf/--cdf-continuity";
        return ParseStatus::kError;
      }
      if (options.video_palette && options.video_tx_search) {
        error = "--video-palette codes eob=0 palette leaves and is not wired "
                "with --video-tx-search (TX_MODE_SELECT needs a real luma TX "
                "partition per leaf)";
        return ParseStatus::kError;
      }
      if (options.video_ibc && options.video_drl_reorder_always) {
        error = "--video-ibc codes use_intrabc and §7.12.2.19 "
                "DRL_REORDER_ALWAYS also reorders the decoder's BV stack, "
                "which the IBC stack builder does not implement "
                "(--video-drl-reorder CONSTRAINT is exact)";
        return ParseStatus::kError;
      }
      if (options.video_ibc &&
          (options.video_segmentation || options.video_per_segment_q)) {
        error = "--video-ibc is not wired with "
                "--video-segmentation/--video-per-segment-q (the per-SB "
                "delta-Q state and the IBC skip block disagree)";
        return ParseStatus::kError;
      }
    }
    if (options.video_scc_force) {
      // §5.4.7 forced form: only meaningful on the screen-content carriers.
      // The per-tool gates above cover the tile count, frame-type chain,
      // format and CDF surface; here we only require a carrier and CPU.
      if (!(options.video_palette || options.video_ibc ||
            options.video_morph)) {
        error = "--video-scc-force needs one of --video-palette/--video-ibc/"
                "--video-morph: the forced sequence form only changes how "
                "those tools are signalled";
        return ParseStatus::kError;
      }
      if (options.video_gpu) {
        error = "--video-scc-force is CPU-only: the CUDA M2 backend has no "
                "screen-content mode-info path";
        return ParseStatus::kError;
      }
    }
    if (options.video_force_integer_mv) {
      if (!options.video_scc_force) {
        error = "--video-force-integer-mv needs --video-scc-force";
        return ParseStatus::kError;
      }
      // force_integer_mv pins FrameMvPrecision to one pel, so a frame
      // precision flag or a per-block precision tool would be silently
      // ignored (or desynchronise the MVD units).
      if (options.video_mv_precision != 0) {
        error = "--video-force-integer-mv pins FrameMvPrecision to one pel; "
                "drop --video-mv-precision";
        return ParseStatus::kError;
      }
      if (options.video_flex_mvres || options.video_adaptive_mvd ||
          options.video_mvd_sign_derive) {
        error = "--video-force-integer-mv is not wired with "
                "--video-flex-mvres/--video-adaptive-mvd/"
                "--video-mvd-sign-derive (pb_mv_precision and the AMVD/sign "
                "units assume a signalled frame precision)";
        return ParseStatus::kError;
      }
      if (options.video_gme || options.video_warp) {
        error = "--video-force-integer-mv is not wired with "
                "--video-gme/--video-warp (the projected global MVs are not "
                "quantised to the one-pel frame grid)";
        return ParseStatus::kError;
      }
    }
    if (options.video_reduce_pb_aspect) {
      // §5.4.3: the tiled writer has its own minimal sequence header that does
      // not carry the partition config; keep the tiled path byte-identical.
      if (options.tiles != "1x1") {
        error = "--video-reduce-pb-aspect is gated on a single tile (the "
                "tiled sequence-header copy does not carry the partition "
                "config)";
        return ParseStatus::kError;
      }
      if (options.b_frames != 0 || options.video_tlayers != 0 ||
          options.video_tip || options.video_long_term || options.video_olk ||
          options.video_leading != 0 || options.video_sef ||
          options.video_switch || options.video_ras) {
        error = "--video-reduce-pb-aspect is gated on the plain key+P chain: "
                "the B/TIP/frame-type wrappers build their own sequence header "
                "and partition plan";
        return ParseStatus::kError;
      }
    }
    if (options.video_seq_seg_info) {
      if (!options.video_segmentation && !options.video_per_segment_q) {
        error = "--video-seq-seg-info needs --video-segmentation or "
                "--video-per-segment-q";
        return ParseStatus::kError;
      }
      if (options.tiles != "1x1") {
        error = "--video-seq-seg-info is gated on a single tile (the tiled "
                "sequence-header copy does not carry the segment table)";
        return ParseStatus::kError;
      }
      if (options.video_seq_seg_frozen &&
          (options.video_seg_skip || options.video_seg_globalmv)) {
        error = "--video-seq-seg-info-frozen is not wired with "
                "--video-seg-skip/--video-seg-globalmv: those features make "
                "the P-frame table differ from the frozen sequence table";
        return ParseStatus::kError;
      }
    }
    if (options.video_bru) {
      // §7.12.2/§7.22 BRU needs enable_bru/use_bru and the BRU tile mode.
      // The component writer builds a valid stream, but the pinned decoder
      // rejects it at the frame header (repro:
      // src/pipeline/video/video_bru_reject_gate.sh with the pinned dav2d).
      error = "--video-bru is not decoder-gated: the pinned dav2d does not "
              "parse enable_bru/use_bru (Error parsing frame header).  The "
              "BRU component stream and the reproducer are in "
              "src/pipeline/video/video_bru_reject_gate.sh";
      return ParseStatus::kError;
    }
    if (options.video_ref_mvs) {
      if (!options.video) {
        error = "--video-ref-mvs requires --video";
        return ParseStatus::kError;
      }
      if (options.tiles != "1x1") {
        error = "--video-ref-mvs is gated on a single tile: the §7.22 field "
                "is built from one whole-frame MI grid";
        return ParseStatus::kError;
      }
      if (options.video_mv_precision != 0) {
        error = "--video-ref-mvs currently requires eighth-pel MVs";
        return ParseStatus::kError;
      }
      if (options.bit_depth != 8 || options.chroma != "420") {
        error = "--video-ref-mvs is gated on 8-bit 4:2:0";        return ParseStatus::kError;
      }
    }
    if (options.video_compound_near_near) {
      if (!options.video_compound) {
        error = "--video-compound-near-near needs --video-compound";
        return ParseStatus::kError;
      }
      if (options.b_frames == 0) {
        error = "--video-compound-near-near is gated on the two-reference "
                "B-frame path (one past and one future reference): pass "
                "--b-frames";
        return ParseStatus::kError;
      }
      if (options.video_gpu) {
        error = "--video-compound-near-near is CPU-only: the CUDA compound "
                "backend implements the GLOBAL_GLOBALMV subset";        return ParseStatus::kError;
      }
    }
    if (options.video_cwp_weighted || options.video_imp_mask) {
      // Both derive from the NEAR_NEARMV compound carrier: §5.20.7.6 codes
      // cwp_idx only for a NEAR_NEARMV AVERAGE block, and the §7.13.3.14
      // implicit mask excludes GLOBAL_GLOBALMV.
      if (!options.video_compound || !options.video_compound_near_near) {
        error = "--video-cwp-weighted/--video-imp-mask need "
                "--video-compound --video-compound-near-near (the "
                "NEAR_NEARMV AVERAGE carrier)";
        return ParseStatus::kError;
      }
      if (options.video_opfl || options.video_refine_mv) {
        // §5.20.7.6 skips cwp_idx for use_optflow/use_refinemv, and the
        // refinement predictors hard-code CWP_EQUAL.  Keep the overlays apart.
        error = "--video-cwp-weighted/--video-imp-mask do not compose with "
                "--video-opfl/--video-refine-mv";
        return ParseStatus::kError;
      }
      if (options.video_gpu) {
        error = "--video-cwp-weighted/--video-imp-mask are CPU-only";
        return ParseStatus::kError;
      }
      if (options.bit_depth != 8 || options.chroma != "420") {
        error = "--video-cwp-weighted/--video-imp-mask are gated on 8-bit "
                "4:2:0";
        return ParseStatus::kError;
      }
    }
    if (options.video_df_sub_pu) {
      // §5.18.2 allow_df_sub_pu exists only for an inter frame; every other
      // frame type leaves it inferred 0.  The TIP output branch codes its own
      // allow_df_sub_pu/apply_deblocking_filter_tip pair, so keep the carrier
      // on the ordinary P/B tile path.
      if (options.video_gpu) {
        error = "--video-df-sub-pu is CPU-only";
        return ParseStatus::kError;
      }
      if (options.video_tip || options.video_bru) {
        error = "--video-df-sub-pu does not compose with --video-tip/"
                "--video-bru: those TIP output frames code their own "
                "allow_df_sub_pu pair";
        return ParseStatus::kError;
      }
      if (options.video_long_term || options.video_olk ||
          options.video_leading != 0 || options.video_sef ||
          options.video_switch || options.video_ras ||
          options.video_tlayers != 0) {
        error = "--video-df-sub-pu is gated on the plain key+P/B chain: the "
                "frame-type wrappers and --video-tlayers need their own "
                "allow_df_sub_pu handling";
        return ParseStatus::kError;
      }
      if (options.video_lossless) {
        // deblocking_filter_params() returns before allow_df_sub_pu when
        // CodedLossless is set.
        error = "--video-df-sub-pu does not compose with --video-lossless";
        return ParseStatus::kError;
      }
      if (options.bit_depth != 8 || options.chroma != "420") {
        error = "--video-df-sub-pu is gated on 8-bit 4:2:0";
        return ParseStatus::kError;
      }
      if (options.tiles != "1x1") {
        error = "--video-df-sub-pu is gated on a single tile";
        return ParseStatus::kError;
      }
    }
    if ((options.video_opfl || options.video_refine_mv) &&
        !options.video_compound_near_near && !options.video_gpu) {
      // Both tools predict from a non-GLOBALMV compound block
      // (§5.20.7.6 use_optflow, §5.20.7.17 refine-MV), which only the
      // NEAR_NEARMV carrier codes.  --video-gpu reports its own by-name
      // rejection below instead.
      error = "--video-opfl/--video-refine-mv need --video-compound-near-near: "
              "the temporal refinement tools only run on a NEAR_NEARMV "
              "compound block";
      return ParseStatus::kError;
    }
    if (options.b_frames != 0 && options.b_frames > 6) {
      error = "--b-frames supports 1..6 B frames per GOP";
      return ParseStatus::kError;
    }
    if (options.video_implicit_ref_map) {
      // §5.4.6/§7.7 implicit reference maps are wired on the CPU 8-bit 4:2:0
      // flat-P, B-frame and hierarchical-P paths.  Every tool that owns an
      // explicit map (or a separate format core) is rejected by name.
      if (options.video_tip) {
        error = "--video-implicit-ref-map does not compose with --video-tip: "
                "TIP needs the explicit past/future anchor pair";
        return ParseStatus::kError;
      }
      if (options.video_compound || options.video_compound_wedge ||
          options.video_compound_diffwtd) {
        error = "--video-implicit-ref-map does not compose with "
                "--video-compound: reference_select and the compound ref pairs "
                "require an explicit two-reference map";
        return ParseStatus::kError;
      }
      if (options.tiles != "1x1") {
        error = "--video-implicit-ref-map does not compose with --tiles: the "
                "tiled frame header signals a single explicit reference slot";
        return ParseStatus::kError;
      }
      if (options.video_long_term || options.video_olk ||
          options.video_leading != 0 || options.video_sef ||
          options.video_switch || options.video_ras) {
        error = "--video-implicit-ref-map does not compose with the "
                "long-term/OLK/leading/SEF/SWITCH/RAS frame-type wrappers";
        return ParseStatus::kError;
      }
      if (options.bit_depth != 8 || options.chroma != "420") {
        error = "--video-implicit-ref-map is wired for the 8-bit 4:2:0 video "
                "core only; the --bit-depth/--chroma format cores keep the "
                "explicit map";
        return ParseStatus::kError;
      }
      if (options.svc_given) {
        error = "--video-implicit-ref-map does not compose with --svc-layers";
        return ParseStatus::kError;
      }
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
    // §5.4.6 adaptive MVD / MVD sign derivation / flexible MV resolution.
    // The three tools share the §5.20.7.6 block path and are gated only on the
    // flat CPU key+P carrier; every other composition is rejected by name so
    // the sequence header never advertises a tool the tile writer cannot code
    // exactly.
    const bool mv_res_tools = options.video_adaptive_mvd ||
                              options.video_mvd_sign_derive ||
                              options.video_flex_mvres;
    if (mv_res_tools) {
      const char* tool = options.video_adaptive_mvd
                             ? "--video-adaptive-mvd"
                             : (options.video_mvd_sign_derive
                                    ? "--video-mvd-sign-derive"
                                    : "--video-flex-mvres");
      if (!options.video) {
        error = std::string(tool) + " requires --video";
        return ParseStatus::kError;
      }
      if (options.video_gpu) {
        error = std::string(tool) +
                " is CPU-only: the CUDA M2 backend has no per-block MV "
                "precision/AMVD syntax path";
        return ParseStatus::kError;
      }
      if (options.bit_depth != 8 || options.chroma != "420") {
        error = std::string(tool) + " is gated on 8-bit 4:2:0 inter coding";
        return ParseStatus::kError;
      }
      if (options.tiles != "1x1") {
        error = std::string(tool) +
                " is gated on a single tile (the §8.3 MV-precision contexts "
                "are tile-local)";
        return ParseStatus::kError;
      }
      if (options.video_lossless) {
        error = std::string(tool) +
                " is not wired into the --video-lossless writer";
        return ParseStatus::kError;
      }
      if (options.b_frames != 0 || options.video_tlayers != 0 ||
          options.video_tip) {
        error = std::string(tool) +
                " is gated on the flat key+P chain (no --b-frames, "
                "--video-tlayers or --video-tip)";
        return ParseStatus::kError;
      }
      if (frame_modes != 0) {
        error = std::string(tool) +
                " does not compose with the frame-type modes "
                "(--video-long-term/--video-olk, --video-leading, --video-sef, "
                "--video-switch, --video-ras)";
        return ParseStatus::kError;
      }
      if (options.video_compound || options.video_compound_interintra ||
          options.video_bawp || options.video_gme || options.video_warp ||
          options.video_opfl || options.video_refine_mv) {
        // §5.20.7.6 places use_amvd relative to use_bawp/motion_mode/compound
        // modes this writer does not code, so the combined order is not gated.
        error = std::string(tool) +
                " does not compose with --video-compound, "
                "--video-compound-interintra, --video-bawp, --video-gme/"
                "--video-warp or --video-opfl/--video-refine-mv yet "
                "(§5.20.7.6 symbol order)";
        return ParseStatus::kError;
      }
      if (options.video_mvd_sign_derive && options.video_mv_precision == 0) {
        error = "--video-mvd-sign-derive needs --video-mv-precision quarter "
                "or half: §5.20.7.13 derives a sign only when "
                "FrameMvPrecision <= QUARTER and MvPrecision < QUARTER";
        return ParseStatus::kError;
      }
      if (options.video_mvd_sign_derive && options.video_mv_precision == 1 &&
          !options.video_flex_mvres) {
        error = "--video-mvd-sign-derive with quarter-pel frames needs "
                "--video-flex-mvres so a block can select half-pel "
                "(MvPrecision < QUARTER); a quarter-pel block derives no sign";
        return ParseStatus::kError;
      }
    }
    if ((options.video_wiener || options.video_chroma_lr) &&
        options.tiles != "1x1") {
      error = "--video-wiener/--video-chroma-lr do not support --tiles > "
              "1x1: the pinned decoder's loop-restoration unit and window "
              "handling at shared tile edges does not match a frame-level "
              "filter (docs/negative-results.md)";
      return ParseStatus::kError;
    }
    // Every §5.18.7/§5.4.8 segmentation or delta-Q surface shares the same
    // header/decision restrictions (single tile, fixed CQP, no CDF chain,
    // no frame-schedule wrappers, no QM).
    const bool video_qp_tools =
        options.video_segmentation || options.video_delta_q ||
        options.video_per_segment_q || options.video_uv_delta_q ||
        options.video_seg_skip || options.video_seg_globalmv ||
        options.video_seg_temporal || options.video_segments != 0;
    if (flag_given(options, "--video-delta-q-res") && !options.video_delta_q) {
      error = "--video-delta-q-res requires --video-delta-q";
      return ParseStatus::kError;
    }
    if (options.video_uv_delta_q &&
        (options.bit_depth != 8 || options.chroma != "420")) {
      error = "--video-uv-delta-q is only wired on the 8-bit 4:2:0 video core";
      return ParseStatus::kError;
    }
    if (options.video_uv_delta_q && options.video_tcq) {
      error = "--video-uv-delta-q does not compose with --video-tcq: the "
              "per-block plane quantiser differs from the scalar TCQ step";
      return ParseStatus::kError;
    }
    if (options.video_uv_delta_q && options.video_cctx) {
      error = "--video-uv-delta-q does not compose with --video-cctx: the "
              "joint U/V search uses one scalar chroma step";
      return ParseStatus::kError;
    }
    if (options.video_uv_delta_q && options.video_qm) {
      error = "--video-uv-delta-q does not compose with --video-qm: the "
              "per-coefficient QM quantiser is not the scalar DC/AC step";
      return ParseStatus::kError;
    }
    if ((options.video_seg_skip || options.video_seg_globalmv) &&
        options.video_segments == 1) {
      error = "--video-seg-skip/--video-seg-globalmv need at least two "
              "segments";
      return ParseStatus::kError;
    }
    if (options.video_seg_skip && options.video_seg_globalmv &&
        options.video_segments != 0 && options.video_segments < 3) {
      error = "--video-seg-skip and --video-seg-globalmv need --video-segments "
              ">= 3 so the features occupy distinct segment ids";
      return ParseStatus::kError;
    }
    if (options.video_seg_globalmv && options.video_per_segment_q) {
      error = "--video-seg-globalmv does not compose with "
              "--video-per-segment-q: an intra leaf cannot stay in the forced "
              "GLOBALMV segment, and reassigning it would change the "
              "quantizer the block was decided with";
      return ParseStatus::kError;
    }
    if (options.video_seg_temporal && options.b_frames != 0) {
      error = "--video-seg-temporal does not compose with --b-frames";
      return ParseStatus::kError;
    }
    if ((options.video_segmentation || options.video_per_segment_q ||
         options.video_seg_skip || options.video_seg_globalmv ||
         options.video_seg_temporal) &&
        options.video_refs != 1) {
      error = "segmentation needs a single reference: PrevSegmentIds and "
              "SkipSegFrame are replayed from ref 0";
      return ParseStatus::kError;
    }
    if ((options.video_seg_skip || options.video_seg_globalmv) &&
        (options.video_compound || options.video_compound_wedge ||
         options.video_compound_diffwtd || options.video_compound_interintra ||
         options.video_cwp_weighted || options.video_imp_mask)) {
      error = "--video-seg-skip/--video-seg-globalmv do not compose with the "
              "compound prediction surfaces (forced GLOBALMV is single-ref)";
      return ParseStatus::kError;
    }
    if (flag_given(options, "--video-seg-delta") && !options.video_per_segment_q &&
        !options.video_delta_q) {
      error = "--video-seg-delta requires --video-per-segment-q or "
              "--video-delta-q";
      return ParseStatus::kError;
    }
    if (options.video_per_segment_q && options.video_delta_q) {
      error = "--video-per-segment-q does not compose with --video-delta-q";
      return ParseStatus::kError;
    }
    if (options.video_per_segment_q) {
      const int segment_q = options.qp + options.video_seg_delta;
      if (segment_q < 1 || segment_q > 255) {
        error = "per-segment Q: --qp + --video-seg-delta must be in [1, 255] "
                "(the video transform path has no lossless segment)";
        return ParseStatus::kError;
      }
    }
    if (video_qp_tools && options.rc_mode != "cqp") {
      error = "segmentation/delta-Q requires a fixed base_q_idx (--rc cqp); "
              "the CRF/VBR per-frame QP plan is not composed with these tools";
      return ParseStatus::kError;
    }
    if (video_qp_tools && options.video_qm) {
      error = "segmentation/delta-Q does not compose with --video-qm: the "
              "frame-header writer owns the using_qmatrix bit";
      return ParseStatus::kError;
    }
    if (video_qp_tools &&
        (options.avg_cdf || options.adaptive_cdf || options.cdf_continuity)) {
      error = "segmentation/delta-Q does not compose with the CDF chain: the "
              "per-tile delta/segment CDFs are not part of the §8.2.4 image";
      return ParseStatus::kError;
    }
    if (video_qp_tools &&
        (frame_modes != 0 || options.b_frames > 0 ||
         options.video_tlayers > 0 || options.video_tip)) {
      error = "segmentation/delta-Q does not compose with the B/temporal-"
              "layer/random-access/TIP frame schedules";
      return ParseStatus::kError;
    }
    if (video_qp_tools && options.tiles != "1x1") {
      error = "segmentation/delta-Q does not support --tiles > 1x1 yet: the "
              "tiled frame-header writer does not signal §5.18.7";
      return ParseStatus::kError;
    }
    if ((options.video_cdef || options.video_ccso || options.video_wiener ||
         options.video_chroma_lr || options.video_gdf) &&
        frame_modes != 0) {
      error = "video loop filters do not compose with the frame-type wrappers "
              "(their component header writers do not signal the filter rows)";
      return ParseStatus::kError;
    }
    if (flag_given(options, "--video-tx-lambda") && !options.video_tx_search) {
      error = "--video-tx-lambda requires --video-tx-search";
      return ParseStatus::kError;
    }
    if (options.video_mhccp && (options.avg_cdf || options.adaptive_cdf)) {
      error = "--video-mhccp does not compose with --avg-cdf/--adaptive-cdf: "
              "the §8.2.4 bank does not carry the §5.20.7.32 cfl_mhccp/"
              "cfl_mh_dir CDFs";
      return ParseStatus::kError;
    }    if (options.video_tx_search && frame_modes != 0) {
      error = "--video-tx-search does not compose with the frame-type wrapper "
              "modes (their header writers do not signal tx_mode_select)";
      return ParseStatus::kError;
    }
    if (options.video_pc_wiener && options.video_wiener) {
      error = "--video-pc-wiener replaces the luma Wiener NS decision; do not "
              "combine the two luma restoration tools";
      return ParseStatus::kError;
    }
    if (options.video_qm && options.qp != 90) {
      error = "--video-qm is gated at --qp 90: the shared device QM stage "
              "builds q2 from the pinned 528 quantizer step";
      return ParseStatus::kError;
    }
    if (options.video_user_qm && options.video_gpu) {
      error = "--video-user-qm is CPU-only: the CUDA video backend does not "
              "emit the §5.13 QM OBU";
      return ParseStatus::kError;
    }
    if (options.video_grain && options.video_gpu) {
      error = "--video-grain is CPU-only: the CUDA video backend does not "
              "write §5.14 film grain OBUs or frame headers";
      return ParseStatus::kError;
    }
    if (options.video_idtx_1d) {
      error =
          "--video-idtx-1d is rejected: the one-axis V/H_DCT, V/H_ADST and "
          "V/H_FLIPADST intra types need dc_sign_horz_vert (§5.20.7.27) and "
          "the class-specific scans, and pinned dav2d f9dafbc rejects every "
          "negative H_DCT/V_DCT coefficient under --strict 1 (repro: "
          "hv_neg.ivf from src/codec/intra/supplemental/, strict 0 decodes, "
          "strict 1 fails; avmdec fails too).  Use --video-idtx for the 2-D "
          "identity type";
      return ParseStatus::kError;
    }
    if (options.video_idtx) {
      if (!options.video_tx_search) {
        error = "--video-idtx requires --video-tx-search (without the TX "
                "search every intra residual is DCT_DCT)";
        return ParseStatus::kError;
      }
      if (options.video_gpu) {
        error = "--video-idtx is CPU-only: the CUDA TX search pins "
                "allow_idtx = 0";
        return ParseStatus::kError;
      }
    }
    if (options.video_mrl || options.video_ist) {
      if (options.video_gpu) {
        error = "--video-mrl/--video-ist are CPU-only: the CUDA video backend "
                "does not signal the sequence bits or the per-block syntax";
        return ParseStatus::kError;
      }
      if (options.bit_depth != 8 || options.chroma != "420") {
        error = "--video-mrl/--video-ist are wired for 8-bit 4:2:0";
        return ParseStatus::kError;
      }
      if (options.tiles != "1x1") {
        error = "--video-mrl/--video-ist do not compose with --tiles: the "
                "tiled sequence-header writer is a separate copy";
        return ParseStatus::kError;
      }
      if (options.svc_given) {
        error = "--video-mrl/--video-ist do not compose with --svc-layers";
        return ParseStatus::kError;
      }
    }
    if (options.video_ist && options.video_qm) {
      error = "--video-ist does not compose with --video-qm: the §7.15.3 "
              "inverse needs the per-coefficient q2 dequantizer";
      return ParseStatus::kError;
    }
    if (options.video_ibp) {
      // §5.4.5 enable_ibp changes the prediction of every intra leaf, not
      // only the keyframe: the decoder derives applyIbp/useIBP from the
      // sequence flag, YMode and angle_delta.  The CPU writer uses the
      // verified src/codec/intra/ibp component on both the luma search and
      // the DC/chroma paths; combinations that predict through a different
      // path are rejected by name instead of emitting a diverging stream.
      if (options.video_gpu) {
        error = "--video-ibp is CPU-only: the CUDA video backend does not "
                "apply the §7.13.2 IBP predictor";
        return ParseStatus::kError;
      }
      if (options.bit_depth != 8 || options.chroma != "420") {
        error = "--video-ibp is wired for 8-bit 4:2:0";
        return ParseStatus::kError;
      }
      if (options.tiles != "1x1") {
        error = "--video-ibp does not compose with --tiles: the tiled "
                "sequence-header writer is a separate copy";
        return ParseStatus::kError;
      }
      if (options.svc_given) {
        error = "--video-ibp does not compose with --svc-layers";
        return ParseStatus::kError;
      }
      if (options.video_cfl || options.video_mhccp) {
        error = "--video-ibp requires the plain DC chroma path: "
                "--video-cfl/--video-mhccp do not apply the §7.13.2.12 IBP "
                "DC post-filter";
        return ParseStatus::kError;
      }
      if (options.video_mrl) {
        error = "--video-ibp does not compose with --video-mrl: the IBP "
                "component implements MrlIndex = 0 only";
        return ParseStatus::kError;
      }
      if (options.video_palette || options.video_ibc || options.video_morph) {
        const char* tool = options.video_palette
                               ? "--video-palette"
                               : (options.video_ibc ? "--video-ibc"
                                                   : "--video-morph");
        error = std::string(tool) +
                " is not wired with --video-ibp: the screen-content keyframe "
                "mode-info path does not run the §7.13.2 IBP prediction";
        return ParseStatus::kError;
      }
      if (options.video_compound_interintra) {
        error = "--video-ibp does not compose with "
                "--video-compound-interintra: the §7.13.3.29 inter-intra "
                "predictor does not run the IBP edge-filter/blend path";
        return ParseStatus::kError;
      }
      if (options.b_frames != 0 || options.video_tlayers != 0 ||
          options.video_tip || options.video_long_term || options.video_olk ||
          options.video_leading != 0 || options.video_sef ||
          options.video_switch || options.video_ras) {
        error = "--video-ibp is gated on the plain key+P chain: it cannot be "
                "combined with --b-frames, --video-tlayers, --video-tip or "
                "the frame-type wrapper modes";
        return ParseStatus::kError;
      }
    }
    if (options.video_dip) {
      if (options.video_gpu) {
        error = "--video-dip is CPU-only: the CUDA video backend does not "
                "signal enable_dip or the per-block dip_mode_info()";
        return ParseStatus::kError;
      }
      if (options.bit_depth != 8 || options.chroma != "420") {
        error = "--video-dip is wired for 8-bit 4:2:0";
        return ParseStatus::kError;
      }
      if (options.tiles != "1x1") {
        error = "--video-dip does not compose with --tiles: the tiled "
                "sequence-header writer is a separate copy";
        return ParseStatus::kError;
      }
      if (options.b_frames != 0 || options.video_tlayers != 0 ||
          options.video_long_term || options.video_olk ||
          options.video_leading != 0 || options.video_sef ||
          options.video_switch || options.video_ras || options.video_tip ||
          options.svc_given) {
        error = "--video-dip does not compose with the B/temporal-layer/"
                "random-access/TIP/SVC frame schedules";
        return ParseStatus::kError;
      }
      if (options.video_lossless) {
        error = "--video-dip does not compose with --video-lossless: the "
                "lossless writer has its own keyframe mode-info path";
        return ParseStatus::kError;
      }
      if (options.video_sdp) {
        error = "--video-dip does not compose with --video-sdp: the SDP "
                "luma/chroma tree mode-info path is not gated with DIP";        return ParseStatus::kError;
      }
    }
    if (options.video_sdp) {
      if (options.video_gpu) {
        error = "--video-sdp is CPU-only: the CUDA video backend does not "
                "emit the SDP luma/chroma trees";
        return ParseStatus::kError;
      }
      if (options.bit_depth != 8 || options.chroma != "420") {
        error = "--video-sdp is wired for 8-bit 4:2:0";
        return ParseStatus::kError;
      }
      if (options.tiles != "1x1") {
        error = "--video-sdp is wired for a single tile";
        return ParseStatus::kError;
      }
      if (options.b_frames != 0 || options.video_tlayers != 0 ||
          options.video_long_term || options.video_olk ||
          options.video_leading != 0 || options.video_sef ||
          options.video_switch || options.video_ras || options.video_tip ||
          options.svc_given) {
        error = "--video-sdp does not compose with the B/temporal-layer/"
                "random-access/TIP/SVC frame schedules";
        return ParseStatus::kError;
      }
      if (options.video_morph || options.video_cfl || options.video_mhccp) {
        error = "--video-sdp does not compose with --video-morph/--video-cfl/"
                "--video-mhccp: the SDP keyframe chroma tree has its own mode "
                "and CflAllowedInSdp gating";
        return ParseStatus::kError;
      }
      if (options.video_segmentation || options.video_delta_q ||
          options.video_per_segment_q || options.video_uv_delta_q) {
        error = "--video-sdp does not compose with segmentation/delta-Q";
        return ParseStatus::kError;
      }
    }
    if (options.video_ddt) {
      if (options.video_gpu) {
        error = "--video-ddt is CPU-only: the CUDA video backend does not "
                "apply the DDTX/FDDT kernels";
        return ParseStatus::kError;
      }
      if (options.bit_depth != 8 || options.chroma != "420") {
        error = "--video-ddt is wired for 8-bit 4:2:0";
        return ParseStatus::kError;
      }
      if (options.video_lossless) {
        error = "--video-ddt does not compose with --video-lossless: a "
                "lossless block never signals ADST/FDST";
        return ParseStatus::kError;
      }
      if (options.video_tip || options.svc_given) {
        error = "--video-ddt does not compose with --video-tip or "
                "--svc-layers: those header writers do not signal "
                "enable_inter_ddt";
        return ParseStatus::kError;
      }    }
    if (options.video_lossless) {
      if (options.video_gpu) {
        error = "--video-lossless is CPU-only";
        return ParseStatus::kError;
      }
      if (flag_given(options, "--qp") && options.qp != 0) {
        error = "--video-lossless is Q=0; drop --qp or pass --qp 0";
        return ParseStatus::kError;
      }
      if (options.bit_depth != 8 && options.bit_depth != 10) {
        error = "--video-lossless supports 8-bit and 10-bit (12-bit is not "
                "signalable in AV2 v1.0.0)";
        return ParseStatus::kError;
      }
      if (options.video_lossless_b &&
          options.video_tlayers != 4 && options.video_tlayers != 8) {
        error = "--video-lossless-b codes one closed B GOP: pass "
                "--video-tlayers 4 or 8 (GOP size in pictures)";
        return ParseStatus::kError;
      }
      if (flag_given(options, "--video-min-block")) {
        if (options.video_min_block != 8 && options.video_min_block != 16 &&
            options.video_min_block != 32) {
          error = "--video-lossless uses --video-min-block as its coding "
                  "block size; pass 8, 16 or 32";
          return ParseStatus::kError;
        }
      }
      const std::pair<bool, const char*> not_applied[] = {
          {options.video_qm, "--video-qm"},
          {options.video_user_qm, "--video-user-qm"},
          {options.video_tcq, "--video-tcq"},
          {options.video_cctx, "--video-cctx"},
          {options.video_grain, "--video-grain"},
          {options.video_tx_search, "--video-tx-search"},
          {options.video_intra_full, "--video-intra-full"},
          {options.video_cfl, "--video-cfl"},
          {options.video_mhccp, "--video-mhccp"},
          {options.video_deblock, "--video-deblock"},
          {options.video_cdef, "--video-cdef"},
          {options.video_ccso, "--video-ccso"},
          {options.video_wiener, "--video-wiener"},
          {options.video_pc_wiener, "--video-pc-wiener"},
          {options.video_chroma_lr, "--video-chroma-lr"},
          {options.video_gdf, "--video-gdf"},
          {options.video_gme, "--video-gme"},
          {options.video_warp, "--video-warp"},
          {options.video_bawp, "--video-bawp"},
          {options.video_tip, "--video-tip"},
          {options.video_mrl, "--video-mrl"},
          {options.video_ibp, "--video-ibp"},
          {options.video_dip, "--video-dip"},
          {options.video_ist, "--video-ist"},
          {options.video_uv_delta_q, "--video-uv-delta-q"},
          {options.video_segmentation, "--video-segmentation"},
          {options.video_delta_q, "--video-delta-q"},
          {options.video_per_segment_q, "--video-per-segment-q"},
          {options.video_seg_skip, "--video-seg-skip"},
          {options.video_seg_globalmv, "--video-seg-globalmv"},
          {options.video_seg_temporal, "--video-seg-temporal"},
          {options.video_segments != 0, "--video-segments"},
          {options.video_tip_no_hole_fill, "--video-tip-no-hole-fill"},
          {options.video_compound, "--video-compound"},
          {options.video_inter_residual, "--video-inter-residual"},
          {options.video_subpel_eighth, "--video-subpel-eighth"},
          {options.video_refs != 1, "--video-refs"},
          {options.b_frames != 0, "--b-frames"},
          {options.video_long_term, "--video-long-term"},
          {options.video_olk, "--video-olk"},
          {options.video_leading != 0, "--video-leading"},
          {options.video_sef, "--video-sef"},
          {options.video_switch, "--video-switch"},
          {options.video_ras, "--video-ras"},
          {options.tiles != "1x1", "--tiles"},
          {options.rc_mode != "cqp" || options.rc_given, "--rc"},
      };
      for (const auto& [set, flag] : not_applied) {
        if (!set) continue;
        error = std::string(flag) +
                " is not applied by the --video-lossless writer";
        return ParseStatus::kError;
      }
    }
    if (options.video_tcq || options.video_cctx) {
      const char* tool = options.video_tcq ? "--video-tcq" : "--video-cctx";
      if (options.video_gpu) {
        error = std::string(tool) + " is CPU-only: the CUDA video backend has "
                "no TCQ/CCTX entropy path";
        return ParseStatus::kError;
      }
      if (options.bit_depth != 8 || options.chroma != "420") {
        error = std::string(tool) + " is wired for the 8-bit 4:2:0 video core "
                "only; the --bit-depth/--chroma format cores keep the "
                "historical sequence header";
        return ParseStatus::kError;
      }
      if (options.video_qm || options.video_user_qm) {
        error = std::string(tool) + " does not compose with --video-qm: the "
                "per-coefficient QM quantiser is not the scalar step the "
                "TCQ/CCTX search reconstructs with";
        return ParseStatus::kError;
      }
      if (options.video_segmentation || options.video_delta_q ||
          options.video_per_segment_q || options.video_uv_delta_q) {
        error = std::string(tool) + " does not compose with segmentation/"
                "delta-Q: the per-block quantiser differs from the frame "
                "base_q_idx the TCQ/CCTX decision uses";
        return ParseStatus::kError;
      }
      if (options.video_lossless || options.qp == 0) {
        error = std::string(tool) + " does not compose with lossless "
                "quantisation (CodedLossless forces allow_tcq = 0 and "
                "is_cctx_allowed() = 0)";
        return ParseStatus::kError;
      }
      if (options.tiles != "1x1") {
        error = std::string(tool) + " does not compose with --tiles > 1x1: "
                "the tiled sequence/frame header writer does not signal the "
                "TCQ/CCTX flags";
        return ParseStatus::kError;
      }
      if (frame_modes != 0 || options.video_tip) {
        error = std::string(tool) + " does not compose with the frame-type "
                "wrapper modes: their header writers do not signal allow_tcq "
                "or the CCTX sequence bit";
        return ParseStatus::kError;
      }
      if (options.svc_given) {
        error = std::string(tool) + " does not compose with --svc-layers";
        return ParseStatus::kError;
      }
    }
    if (options.video_qm && options.video_tx_search) {
      error = "--video-qm does not compose with --video-tx-search: the device "
              "TX search module has its own scalar quantizer";
      return ParseStatus::kError;
    }
    if (options.video_mhccp && frame_modes != 0) {
      error = "--video-mhccp does not compose with the frame-type wrapper "
              "modes (their sequence headers do not signal enable_mhccp)";
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
          options.video_inter_residual || options.video_drl_reorder) {
        error = "--video-gme/--video-warp are gated on the plain skip-only P "
                "chain: they cannot be combined with --b-frames, "
                "--video-tlayers, a frame-type wrapper, --video-compound, "
                "--video-inter-residual or --video-drl-reorder";
        return ParseStatus::kError;
      }
      if (options.tiles != "1x1") {
        error = "--video-gme/--video-warp are gated on a single tile";
        return ParseStatus::kError;
      }
    }
    if (options.video_bawp) {
      // BAWP is a single-prediction luma tool.  The gated product surface is
      // the plain P chain with no warp/compound/temporal-frame wrapper and a
      // single tile (the fitted template needs whole-frame coordinates).
      if (options.b_frames != 0 || options.video_tlayers != 0 ||
          frame_modes != 0 || options.video_compound ||
          options.video_gme || options.video_warp ||
          options.video_switch || options.video_ras || options.video_olk ||
          options.video_leading != 0 || options.video_sef ||
          options.video_inter_residual) {
        error = "--video-bawp is gated on the plain skip-only "
                "single-prediction P chain: it cannot be combined with "
                "--b-frames, --video-tlayers, a frame-type wrapper, "
                "--video-compound, --video-gme/--video-warp or "
                "--video-inter-residual";
        return ParseStatus::kError;
      }
      if (options.tiles != "1x1") {
        error = "--video-bawp is gated on a single tile";
        return ParseStatus::kError;
      }
    }
    if (options.video_compound_interintra) {
      // §5.20.7.14 motion_mode_allowed() sets AllowedMotionModes[INTERINTRA]
      // only for !isCompound, so the §7.13.3.29/.30 mask blend rides the
      // single-reference blocks.  The flag is kept under its historical
      // --video-compound-interintra name; the syntax is not compound.
      if (options.video_gpu) {
        error = "--video-compound-interintra is CPU-only: the CUDA M2 backend "
                "does not code the §5.20.7.14 motion-mode syntax";
        return ParseStatus::kError;
      }
      if (options.video_compound || options.video_compound_wedge ||
          options.video_compound_diffwtd) {
        error = "--video-compound-interintra is the single-reference "
                "COMPOUND_INTRA path; it cannot be combined with "
                "--video-compound (reference_select makes the block compound, "
                "which forbids INTERINTRA)";
        return ParseStatus::kError;
      }
      if (options.video_refs != 1) {
        error = "--video-compound-interintra requires --video-refs 1: "
                "§5.20.7.14 allows INTERINTRA only for a single-reference "
                "block";
        return ParseStatus::kError;
      }
      if (options.video_bawp || options.video_morph) {
        error = "--video-compound-interintra cannot be combined with "
                "--video-bawp/--video-morph: §5.20.7.14 returns before "
                "AllowedMotionModes[INTERINTRA] when use_bawp is set";
        return ParseStatus::kError;
      }
      if (options.b_frames != 0 || options.video_tlayers != 0 ||
          options.video_tip || options.video_long_term || options.video_olk ||
          options.video_leading != 0 || options.video_sef ||
          options.video_switch || options.video_ras) {
        error = "--video-compound-interintra is gated on the plain key+P "
                "chain: it cannot be combined with --b-frames, "
                "--video-tlayers, --video-tip or the frame-type wrappers";
        return ParseStatus::kError;
      }
      if (options.tiles != "1x1") {
        error = "--video-compound-interintra is gated on a single tile";
        return ParseStatus::kError;
      }
      if (options.avg_cdf || options.cdf_continuity) {
        error = "--video-compound-interintra uses the per-tile §9.3 "
                "inter-intra CDFs; it is not wired with "
                "--avg-cdf/--cdf-continuity";
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
    } else if (options.video_compound_wedge ||
               options.video_compound_diffwtd) {
      error = "--video-compound-wedge/--video-compound-diffwtd need "
              "--video-compound";
      return ParseStatus::kError;
    }
    if (options.video_gops != 1 && options.b_frames == 0) {
      error = "--video-gops only applies to the --b-frames schedule";
      return ParseStatus::kError;
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

    if (options.video_tip_no_hole_fill && !options.video_tip) {
      error = "--video-tip-no-hole-fill is the A/B control for --video-tip";
      return ParseStatus::kError;
    }

    if (options.video_tip_block_ref && !options.video_tip) {
      error =
          "--video-tip-block-ref is the §5.20.7.10 RefFrame[0] == "
          "TIP_FRAME carrier for --video-tip --video-ref-mvs";
      return ParseStatus::kError;
    }
    if (options.video_tip_block_ref && !options.video_ref_mvs) {
      error = "--video-tip-block-ref requires --video-ref-mvs";
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
      if (options.video_tip) {
        return reject_gpu("--video-tip (TIP frame scheduling)");
      }
      if (options.video_ref_mvs) {
        return reject_gpu("--video-ref-mvs (§7.9/§7.22 temporal MVs)");
      }
      if (options.video_bru) {
        return reject_gpu("--video-bru (backwards reference update)");
      }
      if (options.video_opfl) {
        return reject_gpu("--video-opfl (CPU §7.13.3.8-.12 OPFL)");
      }
      if (options.video_refine_mv) {
        return reject_gpu("--video-refine-mv (CPU §7.13.3.3-.6 refine-MV)");
      }
      if (options.cdf_continuity) {
        return reject_gpu("--cdf-continuity (cross-frame CDF carry)");
      }
      if (options.video_segmentation || options.video_delta_q ||
          options.video_per_segment_q || options.video_uv_delta_q ||
          options.video_seg_skip || options.video_seg_globalmv ||
          options.video_seg_temporal || options.video_segments != 0) {
        return reject_gpu("--video-segmentation/--video-delta-q/"
                          "--video-per-segment-q/--video-uv-delta-q "
                          "(CPU reference only)");
      }
      if (options.svc_given) return reject_gpu("--svc-layers");
      if (options.video_implicit_ref_map) {
        return reject_gpu("--video-implicit-ref-map (§7.7 implicit maps)");
      }
      if (options.video_morph) {
        return reject_gpu("--video-morph (IBC keyframe mode info)");
      }
      if (options.video_palette) {
        return reject_gpu("--video-palette (palette keyframe mode info)");
      }
      if (options.video_ibc) {
        return reject_gpu("--video-ibc (IBC keyframe mode info)");
      }
      if (options.video_ibp) {
        return reject_gpu("--video-ibp (§7.13.2 intra bi-prediction)");
      }
      if (options.video_compound_interintra) {
        return reject_gpu("--video-compound-interintra (§5.20.7.14 motion "
                          "mode)");
      }
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
        flag_given(options, "--video-sdp") ||
        flag_given(options, "--video-ddt") ||
        flag_given(options, "--video-tx-lambda") ||
        flag_given(options, "--video-deblock") ||
        flag_given(options, "--video-segmentation") ||
        flag_given(options, "--video-delta-q") ||
        flag_given(options, "--video-delta-q-res") ||
        flag_given(options, "--video-per-segment-q") ||
        flag_given(options, "--video-seg-delta") ||
        flag_given(options, "--video-uv-delta-q") ||
        flag_given(options, "--video-seg-skip") ||
        flag_given(options, "--video-seg-globalmv") ||
        flag_given(options, "--video-seg-temporal") ||
        flag_given(options, "--video-segments") ||
        flag_given(options, "--video-gme") ||
        flag_given(options, "--video-warp") ||
        flag_given(options, "--video-bawp") ||
        flag_given(options, "--video-morph") ||
        flag_given(options, "--video-palette") ||
        flag_given(options, "--video-ibc") ||
        flag_given(options, "--video-opfl") ||
        flag_given(options, "--video-refine-mv") ||
        flag_given(options, "--video-compound-near-near") ||
        flag_given(options, "--video-tip") ||
        flag_given(options, "--video-tip-no-hole-fill") ||
        flag_given(options, "--video-intra-full") ||
        flag_given(options, "--video-cfl") ||
        flag_given(options, "--video-mrl") ||
        flag_given(options, "--video-ibp") ||
        flag_given(options, "--video-dip") ||
        flag_given(options, "--video-ist") ||
        flag_given(options, "--video-mhccp") ||
        flag_given(options, "--video-qm") ||
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
        flag_given(options, "--video-sb128") ||
        flag_given(options, "--video-sb256") ||
        flag_given(options, "--video-ext-partitions") ||
        flag_given(options, "--video-min-block") ||
        flag_given(options, "--video-max-depth") ||
        flag_given(options, "--video-me-range") ||
        flag_given(options, "--video-cdef") ||
        flag_given(options, "--video-cdef-strength") ||
        flag_given(options, "--video-cdef-damping") ||
        flag_given(options, "--video-ccso") ||
        flag_given(options, "--video-ccso-bands") ||
        flag_given(options, "--video-wiener") ||
        flag_given(options, "--video-pc-wiener") ||
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
    if (options.m1) {
      return reject("--lossless has its own still-picture writer; do not "
                    "combine it with --still");
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
  if (options.video && !options.video_lossless && options.qp == 0) {
    return reject("--video does not support lossless QP 0; use "
                  "--video-lossless");
  }
  if (options.video && options.bit_depth == 8 && options.qp > 255) {
    return reject("--qp > 255 requires 10-bit: MaxQ is 255 at 8-bit");
  }
  if (options.video && options.bit_depth == 10 && options.qp > 303) {
    return reject("--qp > 303 exceeds the 10-bit MaxQ");
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
  if (options.lossless_force_fsc && !options.lossless &&
      !options.video_lossless) {
    return reject("--lossless-fsc requires --lossless or --video-lossless");
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

    const int selections = (options.svc_oppoint >= 0 ? 1 : 0) +
                           (options.svc_all_layers ? 1 : 0) +
                           (options.svc_spatial_layers >= 1 ? 1 : 0);
    if (selections > 1) {
      error = "select at most one of --oppoint, --alllayers or "
              "--spatial-layers";
      return ParseStatus::kError;
    }
    if (options.svc_temporal != 1 && options.svc_embedded != 1) {
      error = "--svc-temporal > 1 does not compose with --svc-embedded > 1: "
              "the embedded refinement ladder needs a cross-mlayer reference "
              "schedule inside the hierarchical GOP; encode one of them at a "
              "time";
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
  if (options.video_sb128 || options.video_sb256) {
    if (options.video_sb128 && options.video_sb256) {
      error = "--video-sb128 and --video-sb256 are mutually exclusive";
      return ParseStatus::kError;
    }
    if (options.video_gpu) {
      error = "--video-sb128/--video-sb256 are CPU-only; --video-gpu keeps "
              "64x64 superblocks";
      return ParseStatus::kError;
    }
    if (options.svc_given) {
      error = "--video-sb128/--video-sb256 do not compose with SVC layers";
      return ParseStatus::kError;
    }
  }
  if (options.video_ext_partitions) {
    if (options.video_gpu) {
      error = "--video-ext-partitions is CPU-only; --video-gpu keeps the "
              "base partition set";
      return ParseStatus::kError;
    }
    if (options.svc_given) {
      error = "--video-ext-partitions does not compose with SVC layers";
      return ParseStatus::kError;
    }
    if (options.video_sb256) {
      error = "--video-ext-partitions does not compose with "
              "--video-sb256 (use --video-sb128)";
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

// Multi-frame reconstruction dump for the format video core.  The Y4M body is
// 16-bit little-endian for 10-bit and 8-bit bytes for the 8-bit formats, and
// the C tag matches the signalled format.
bool write_video_recon_y4m_format(
    const std::filesystem::path& path, const av2enc::fmt::Format& format,
    std::uint32_t fps_num, std::uint32_t fps_den,
    const std::vector<av2enc::video::VideoReconFrame>& recon,
    std::string& error) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    error = "cannot open --video-recon output '" + path.string() + "'";
    return false;
  }
  const int bps = format.depth == av2enc::fmt::BitDepth::k8 ? 1 : 2;
  out << "YUV4MPEG2 W" << format.width << " H" << format.height << " F"
      << fps_num << ":" << fps_den << " Ip A1:1 "
      << av2enc::fmt::y4m_colorspace_tag(format.chroma, format.depth) << "\n";
  const int sub_x = av2enc::fmt::subsampling_x(format.chroma);
  const int sub_y = av2enc::fmt::subsampling_y(format.chroma);
  const bool has_chroma = av2enc::fmt::has_chroma(format.chroma);
  const std::uint32_t widths[3] = {format.width, format.width >> sub_x,
                                   format.width >> sub_x};
  const std::uint32_t heights[3] = {format.height, format.height >> sub_y,
                                    format.height >> sub_y};
  const int plane_count = av2enc::fmt::num_planes(format.chroma);
  std::vector<std::uint8_t> row;
  for (const av2enc::video::VideoReconFrame& frame : recon) {
    out << "FRAME\n";
    for (int plane = 0; plane < plane_count; ++plane) {
      const std::vector<std::uint16_t>& src =
          plane == 0 ? frame.y16 : (plane == 1 ? frame.u16 : frame.v16);
      if (src.empty()) {
        error = "reconstruction plane is empty";
        return false;
      }
      row.resize(static_cast<std::size_t>(widths[plane]) *
                 static_cast<std::size_t>(bps));
      for (std::uint32_t y = 0; y < heights[plane]; ++y) {
        for (std::uint32_t x = 0; x < widths[plane]; ++x) {
          const std::uint16_t sample =
              src[static_cast<std::size_t>(y) * widths[plane] + x];
          if (bps == 1) {
            row[x] = static_cast<std::uint8_t>(sample);
          } else {
            row[static_cast<std::size_t>(x) * 2U] =
                static_cast<std::uint8_t>(sample & 0xFFU);
            row[static_cast<std::size_t>(x) * 2U + 1U] =
                static_cast<std::uint8_t>((sample >> 8) & 0x03U);
          }
        }
        out.write(reinterpret_cast<const char*>(row.data()),
                  static_cast<std::streamsize>(row.size()));
      }
    }
  }
  (void)has_chroma;
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

// Encoder coding order for the reordering schedules.  The rate-control loop
// must walk the pictures in the order they are actually coded: the B and
// hierarchical-P schedules emit units in coding order, and the complexity
// proxy for a picture is the source of its primary reference, not the
// previous display frame.  `primary_ref[display]` is -1 for an intra key.
struct VideoCodingSchedule {
  std::vector<int> order;        // display indices in coding order
  std::vector<int> primary_ref;  // primary reference display, -1 for intra
};

bool video_coding_schedule(const Options& options, int frame_count,
                           VideoCodingSchedule& out, std::string& error) {
  out = VideoCodingSchedule{};
  if (options.b_frames != 0) {
    namespace bf = av2enc::inter::bframes;
    const int gop_size = static_cast<int>(options.b_frames) + 2;
    const int num_gops = std::max(1, options.video_gops);
    bf::SequenceConfig sc;
    sc.num_gops = num_gops;
    sc.gop_size = gop_size;
    sc.pattern = bf::GopPattern::kIpbb;
    int order_hint_bits = 4;
    while ((1 << order_hint_bits) < frame_count) ++order_hint_bits;
    sc.order_hint_bits = order_hint_bits;
    sc.num_slots = 16;
    sc.base_q_idx = static_cast<int>(options.qp);
    bf::Plan plan;
    bf::PlanError plan_error = bf::PlanError::kOk;
    if (!bf::build_sequence(sc, plan, &plan_error)) {
      error = std::string("B-frame rate-control schedule: ") +
              bf::plan_error_name(plan_error);
      return false;
    }
    out.order = plan.coding_order;
    out.primary_ref.assign(static_cast<std::size_t>(plan.display_count()), -1);
    for (const bf::FramePlan& fp : plan.frames) {
      if (!fp.is_key) {
        out.primary_ref[static_cast<std::size_t>(fp.display_index)] =
            fp.ref_display[0];
      }
    }
    return true;
  }
  // --svc-temporal N is the same hierarchical-P closed GOP with 1 << N
  // pictures (the SVC writer expands it into temporal units).
  const bool svc_temporal =
      options.svc_given && options.svc_temporal > 1;
  if (options.video_tlayers > 0 || svc_temporal) {
    namespace tl = av2enc::inter::tlayers;
    const int gop = svc_temporal ? (1 << options.svc_temporal)
                                 : options.video_tlayers;
    tl::PlanConfig pc;
    pc.gop_size = gop;
    pc.schedule = tl::Schedule::kHierarchicalP;
    pc.ref_mode = tl::RefMode::kTwo;
    pc.dependency = tl::DependencyMode::kAllLower;
    pc.explicit_dependency_map = false;
    pc.order_hint_bits = gop >= 16 ? 5 : 4;
    pc.num_slots = gop;
    pc.base_q_idx = static_cast<std::uint8_t>(options.qp);
    tl::Plan plan;
    tl::PlanError plan_error = tl::PlanError::kOk;
    if (!tl::build_plan(pc, plan, &plan_error)) {
      error = std::string("temporal-layer rate-control schedule: ") +
              tl::plan_error_name(plan_error);
      return false;
    }
    out.order = plan.coding_order;
    out.primary_ref.assign(static_cast<std::size_t>(plan.display_count()), -1);
    for (const tl::PlannedFrame& pf : plan.frames) {
      if (!pf.is_key) {
        out.primary_ref[static_cast<std::size_t>(pf.display_index)] =
            pf.ref_display[0];
      }
    }
    return true;
  }
  return true;  // flat key+P: display order, previous display frame
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

  // Reordering schedules: plan each picture in coding order using its actual
  // primary reference as the motion-free inter proxy.  The flat key+P chain
  // keeps the historical display-order/previous-frame model.
  VideoCodingSchedule schedule;
  if (!video_coding_schedule(options, static_cast<int>(inputs.size()),
                             schedule, error)) {
    return false;
  }
  if (schedule.order.empty()) {
    schedule.order.resize(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      schedule.order[i] = static_cast<int>(i);
    }
  }
  // `primary_ref` is only populated by video_coding_schedule for the
  // reordering paths; an empty vector means the historical proxy.
  const bool has_primary_ref = !schedule.primary_ref.empty();
  if (has_primary_ref &&
      schedule.primary_ref.size() != inputs.size()) {
    error = "rate-control schedule does not cover every input picture";
    return false;
  }

  std::vector<av2enc::rc::FrameComplexity> complexity(inputs.size());
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    std::span<const std::uint8_t> predicted(zero_plane);
    int ref = i > 0 ? static_cast<int>(i) - 1 : -1;
    if (has_primary_ref) ref = schedule.primary_ref[i];
    if (ref >= 0 && ref < static_cast<int>(inputs.size())) {
      predicted = inputs[static_cast<std::size_t>(ref)].y;
    }
    complexity[i] = av2enc::rc::complexity_from_u8_planes(
        inputs[i].y, predicted, width, height, width);
    complexity[i].intra_fraction = ref < 0 ? 1.0 : 0.0;
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
      // Calibrate against the pass-1 payload at the fixed --qp.  The encoder
      // emitted tile_bytes in coding order, so pair each payload with the
      // complexity of the picture at that coding position.
      double measured = 0.0;
      double modelled = 0.0;
      for (std::size_t c = 0; c < schedule.order.size(); ++c) {
        const std::size_t display =
            static_cast<std::size_t>(schedule.order[c]);
        if (c >= pass1_payload_bits->size()) break;
        measured += static_cast<double>((*pass1_payload_bits)[c]);
        modelled += av2enc::rc::estimate_frame_bits_model(
            complexity[display], static_cast<int>(options.qp));
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
  // The virtual buffer walks the actual coding order.  frame_qp itself stays
  // indexed by display order because that is how the encoder's per-frame plan
  // looks up each picture's base_q_idx.
  for (const int display_index : schedule.order) {
    const std::size_t display = static_cast<std::size_t>(display_index);
    av2enc::video::post::VideoFrameQp qp;
    if (!rc.begin_frame(complexity[display], qp, &error)) {
      return false;
    }
    (*frame_qp)[display] = static_cast<std::uint8_t>(qp.base_q_idx);
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
  if (options.b_frames != 0) {
    const std::size_t per_gop = static_cast<std::size_t>(options.b_frames) + 2U;
    const std::size_t gops = static_cast<std::size_t>(std::max(1, options.video_gops));
    if (frame_count != per_gop * gops) {
      error = "--b-frames --video-gops " + std::to_string(gops) +
              " codes " + std::to_string(gops) + " closed GOP(s): pass exactly " +
              std::to_string(per_gop * gops) +
              " frames (N B frames + key + P anchor per GOP); got " +
              std::to_string(frame_count);
      return false;
    }
  }
  if (options.video_long_term && frame_count > 16U) {
    error = "--video-long-term supports at most 16 input frames "
            "(OrderHintBits = 4); got " + std::to_string(frame_count);
    return false;
  }
  return true;
}

// Video path for the non-default pixel formats (10-bit 4:2:0/4:2:2/4:4:4/4:0:0
// and 8-bit 4:2:2/4:4:4/4:0:0).  The input reader is the strict format Y4M
// reader; the requested --bit-depth/--chroma must match the stream tag exactly
// (no implicit resampling or depth conversion).
int run_video_format_encode(const Options& options) {
  // Tool flags that are 8-bit 4:2:0 only.  Reject by name instead of silently
  // ignoring them on the format path.
  const std::pair<bool, const char*> unsupported[] = {
      {options.tiles != "1x1", "--tiles"},
      {options.video_tx_search, "--video-tx-search"},
      {options.video_tcq, "--video-tcq"},
      {options.video_cctx, "--video-cctx"},
      {options.video_intra_full, "--video-intra-full"},
      {options.video_ibp, "--video-ibp"},
      {options.video_gme, "--video-gme"},
      {options.video_warp, "--video-warp"},
      {options.video_compound, "--video-compound"},
      {options.video_compound_near_near, "--video-compound-near-near"},
      {options.video_opfl, "--video-opfl"},
      {options.video_refine_mv, "--video-refine-mv"},
      {options.video_cwp_weighted, "--video-cwp-weighted"},
      {options.video_imp_mask, "--video-imp-mask"},
      {options.video_df_sub_pu, "--video-df-sub-pu"},
      {options.video_drl_reorder || options.video_drl_reorder_always,
       "--video-drl-reorder"},
      {options.b_frames != 0, "--b-frames"},
      {options.video_tlayers != 0, "--video-tlayers"},
      {options.video_long_term, "--video-long-term"},
      {options.video_olk, "--video-olk"},
      {options.video_leading != 0, "--video-leading"},
      {options.video_sef, "--video-sef"},
      {options.video_switch, "--video-switch"},
      {options.video_ras, "--video-ras"},
      {options.avg_cdf, "--avg-cdf"},
      {options.adaptive_cdf, "--adaptive-cdf"},
      {options.cdf_continuity, "--cdf-continuity"},
      {options.video_mfh, "--video-mfh"},
      {options.video_timing, "--video-timing"},
      {options.video_decoder_model, "--video-decoder-model"},
      {options.video_brt, "--video-brt"},
      {options.metadata_timecode, "--metadata-timecode"},
      {options.metadata_scan_type, "--metadata-scan-type"},
      {options.metadata_cll, "--metadata-cll"},
      {options.metadata_mdcv, "--metadata-mdcv"},
      {options.metadata_icc, "--metadata-icc"},
      {options.metadata_temporal_point, "--metadata-temporal-point"},
      {options.metadata_frame_hash, "--metadata-frame-hash"},
      {options.metadata_banding, "--metadata-banding"},
      {options.metadata_user_data, "--metadata-user-data"},
      {options.rc_mode != "cqp", "--rc"},
  };
  for (const auto& [set, flag] : unsupported) {
    if (set) {
      std::cerr << "av2enc: error: " << flag
                << " is not wired for the --bit-depth/--chroma video formats "
                   "(8-bit 4:2:0 only)\n";
      return 2;
    }
  }

  av2enc::fmt::Format format;
  format.chroma = options.chroma == "422"   ? av2enc::fmt::ChromaFormat::k422
                  : options.chroma == "444" ? av2enc::fmt::ChromaFormat::k444
                  : options.chroma == "400" ? av2enc::fmt::ChromaFormat::k400
                                            : av2enc::fmt::ChromaFormat::k420;
  format.depth = options.bit_depth == 10 ? av2enc::fmt::BitDepth::k10
                                         : av2enc::fmt::BitDepth::k8;

  std::size_t limit = 0;
  if (options.max_frames != 0) {
    limit = static_cast<std::size_t>(options.max_frames);
  } else if (options.p_frames != 0) {
    limit = static_cast<std::size_t>(options.p_frames) + 1U;
  }

  std::vector<av2enc::fmt::Picture> pictures;
  av2enc::fmt::Y4mFormatInfo info;
  std::string error;
  if (!av2enc::fmt::read_y4m_sequence(options.input, &pictures, &info, limit,
                                      &error)) {
    std::cerr << "av2enc: error: " << error << "\n";
    return 1;
  }
  if (info.chroma != format.chroma || info.depth != format.depth) {
    std::cerr << "av2enc: error: --bit-depth/--chroma request "
              << av2enc::fmt::bit_depth_name(format.depth) << " "
              << av2enc::fmt::chroma_format_name(format.chroma)
              << " does not match the input coloursapce "
              << av2enc::fmt::chroma_format_name(info.chroma) << " "
              << av2enc::fmt::bit_depth_name(info.depth)
              << "; the format paths never convert implicitly\n";
    return 2;
  }
  format.width = info.width;
  format.height = info.height;
  // The format core codes the §5.18.4.4 MI-aligned picture (MiCols x MiRows
  // rounded up to 8 pixels) and crops the output back to the real size.  Only
  // the chroma layout constrains the input: 4:2:0 needs even width/height,
  // 4:2:2 needs an even width, and 4:4:4/4:0:0 accept any size.
  if (av2enc::fmt::has_chroma(format.chroma) &&
      ((av2enc::fmt::subsampling_x(format.chroma) != 0 &&
        (format.width & 1U) != 0) ||
       (av2enc::fmt::subsampling_y(format.chroma) != 0 &&
        (format.height & 1U) != 0))) {
    std::cerr << "av2enc: error: "
              << av2enc::fmt::chroma_format_name(format.chroma)
              << " cannot represent " << format.width << "x" << format.height
              << " (chroma subsampling requires even coded dimensions)\n";
    return 2;
  }
  if (!av2enc::fmt::is_signalable(format)) {
    std::cerr << "av2enc: error: the requested format is not signalable in "
                 "AV2 v1.0.0\n";
    return 2;
  }

  const int sub_x = av2enc::fmt::subsampling_x(format.chroma);
  const int sub_y = av2enc::fmt::subsampling_y(format.chroma);
  const bool has_chroma = av2enc::fmt::has_chroma(format.chroma);
  std::vector<std::vector<std::uint16_t>> owned_y;
  std::vector<std::vector<std::uint16_t>> owned_u;
  std::vector<std::vector<std::uint16_t>> owned_v;
  owned_y.reserve(pictures.size());
  owned_u.reserve(pictures.size());
  owned_v.reserve(pictures.size());
  for (const av2enc::fmt::Picture& picture : pictures) {
    const std::uint32_t cw = format.width >> sub_x;
    const std::uint32_t ch = format.height >> sub_y;
    std::vector<std::uint16_t> y(
        static_cast<std::size_t>(format.width) * format.height);
    for (std::uint32_t r = 0; r < format.height; ++r) {
      std::memcpy(y.data() + static_cast<std::size_t>(r) * format.width,
                  picture.planes[0].samples.data() +
                      static_cast<std::size_t>(r) * picture.planes[0].stride,
                  static_cast<std::size_t>(format.width) *
                      sizeof(std::uint16_t));
    }
    owned_y.push_back(std::move(y));
    std::vector<std::uint16_t> u;
    std::vector<std::uint16_t> v;
    if (has_chroma) {
      u.assign(static_cast<std::size_t>(cw) * ch, 0);
      v.assign(static_cast<std::size_t>(cw) * ch, 0);
      for (std::uint32_t r = 0; r < ch; ++r) {
        std::memcpy(u.data() + static_cast<std::size_t>(r) * cw,
                    picture.planes[1].samples.data() +
                        static_cast<std::size_t>(r) * picture.planes[1].stride,
                    static_cast<std::size_t>(cw) * sizeof(std::uint16_t));
        std::memcpy(v.data() + static_cast<std::size_t>(r) * cw,
                    picture.planes[2].samples.data() +
                        static_cast<std::size_t>(r) * picture.planes[2].stride,
                    static_cast<std::size_t>(cw) * sizeof(std::uint16_t));
      }
    }
    owned_u.push_back(std::move(u));
    owned_v.push_back(std::move(v));
  }

  std::vector<av2enc::video::VideoInputFrame> inputs;
  inputs.reserve(pictures.size());
  for (std::size_t i = 0; i < pictures.size(); ++i) {
    av2enc::video::VideoInputFrame input;
    input.width = format.width;
    input.height = format.height;
    input.format = format;
    input.y16 = std::span<const std::uint16_t>(owned_y[i].data(),
                                               owned_y[i].size());
    input.u16 = std::span<const std::uint16_t>(owned_u[i].data(),
                                               owned_u[i].size());
    input.v16 = std::span<const std::uint16_t>(owned_v[i].data(),
                                               owned_v[i].size());
    inputs.push_back(input);
  }

  if (!validate_video_sequence_length(options, inputs.size(), error)) {
    std::cerr << "av2enc: error: " << error << "\n";
    return 2;
  }

  av2enc::IvfHeader ivf;
  ivf.width = static_cast<std::uint16_t>(format.width);
  ivf.height = static_cast<std::uint16_t>(format.height);
  ivf.fps_num = info.fps_num;
  ivf.fps_den = info.fps_den;
  av2enc::IvfWriter writer;
  if (!writer.open(options.output, ivf, error)) {
    std::cerr << "av2enc: error: " << error << "\n";
    return 1;
  }

  av2enc::video::VideoOptions video_options;
  video_options.format = format;
  video_options.base_q_idx = static_cast<std::uint16_t>(options.qp);
  video_options.num_total_refs = options.video_refs;
  video_options.num_ref_frames = options.video_num_ref_frames;
  video_options.implicit_ref_map = options.video_implicit_ref_map;
  video_options.enable_inter_residual = options.video_inter_residual;
  video_options.me_subpel_eighth = options.video_subpel_eighth;
  video_options.mv_precision = options.video_mv_precision;
  video_options.enable_adaptive_mvd = options.video_adaptive_mvd;
  video_options.enable_mvd_sign_derive = options.video_mvd_sign_derive;
  video_options.enable_flex_mvres = options.video_flex_mvres;
  video_options.enable_ref_frame_mvs = options.video_ref_mvs;
  video_options.ref_mvs_stress = options.video_ref_mvs_stress;
  video_options.force_single_block = options.video_force_single;
  video_options.sb128 = options.video_sb128;
  video_options.sb256 = options.video_sb256;
  video_options.ext_partitions = options.video_ext_partitions;
  video_options.min_block_size = options.video_min_block;
  video_options.max_partition_depth = options.video_max_depth;
  if (format.chroma == av2enc::fmt::ChromaFormat::k422 &&
      !flag_given(options, "--video-max-depth")) {
    // Historical 4:2:2 default: 32x32 roots.  16-pixel 4:2:2 leaves are now
    // exact (partition models 4:2:2 chroma), so an explicit
    // --video-max-depth 3/4 is accepted; the default stream stays unchanged.
    video_options.max_partition_depth = 2;
  }
  video_options.me_range = options.video_me_range;
  video_options.trace = options.video_trace;
  video_options.enable_deblock = options.video_deblock;
  video_options.loop_filters.cdef = options.video_cdef;
  video_options.loop_filters.cdef_num_strengths = options.video_cdef_strength;
  video_options.loop_filters.cdef_damping = options.video_cdef_damping;
  video_options.loop_filters.ccso = options.video_ccso;
  video_options.loop_filters.ccso_bands_log2 = options.video_ccso_bands;
  video_options.loop_filters.cfl_ds_filter_index = options.video_cfl_ds_filter;
  video_options.loop_filters.wiener = options.video_wiener;
  video_options.loop_filters.pc_wiener = options.video_pc_wiener;
  video_options.loop_filters.chroma_lr = options.video_chroma_lr;
  video_options.loop_filters.gdf = options.video_gdf;
  video_options.film_grain = options.video_grain;

  if (options.video_gpu &&
      (format.width % 8U != 0U || format.height % 8U != 0U)) {
    std::cerr << "av2enc: error: --video-gpu is CPU-only for dimensions not "
                 "divisible by 8 (the CUDA kernels assume an MI-aligned "
                 "picture); rerun without --video-gpu, got "
              << format.width << "x" << format.height << "\n";
    return 2;
  }
  if (options.video_gpu &&
      (video_options.enable_deblock || video_options.loop_filters.any())) {
    std::cerr << "av2enc: error: the format video loop-filter stage is CPU-only; "
                 "use --video (the 8-bit 4:2:0 --video-gpu path has its own "
                 "device filters)\n";
    return 2;
  }
  av2enc::video::VideoSequenceResult result;
  bool ok = false;
  if (options.video_gpu) {
#if AV2ENC_VIDEO_GPU_CALLABLE
    ok = av2enc::cuda::video::encode_video_sequence_gpu(inputs, video_options,
                                                        result, error);
    if (ok) {
      std::cerr << "av2enc: --video-gpu backend="
                << av2enc::cuda::video::video_backend_name()
                << " frame_bytes_from_cpu=0\n";
    }
#else
    error = "--video-gpu is not linked in this build";
#endif
  } else {
    ok = av2enc::video::encode_video_sequence(inputs, video_options, result,
                                              error);
  }
  if (!ok) {
    std::cerr << "av2enc: error: " << error << "\n";
    return 1;
  }

  for (std::size_t i = 0; i < result.units.size(); ++i) {
    if (!writer.write_frame(result.units[i], static_cast<std::uint64_t>(i),
                            error)) {
      std::cerr << "av2enc: error: " << error << "\n";
      return 1;
    }
  }
  if (!options.video_recon.empty()) {
    if (!write_video_recon_y4m_format(options.video_recon, format,
                                      info.fps_num, info.fps_den, result.recon,
                                      error)) {
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
            << "  input:   " << options.input.string() << " (" << format.width
            << "x" << format.height << ", " << info.colorspace << ", "
            << info.fps_num << "/" << info.fps_den << " fps)\n"
            << "  output:  " << options.output.string() << " ("
            << writer.bytes_written() << " bytes IVF)\n"
            << "  frames:  " << result.units.size() << " temporal unit(s), "
            << result.tile_bytes.size() << " coded frame(s), "
            << result.recon.size() << " output frame(s), base_q_idx "
            << static_cast<int>(video_options.base_q_idx) << ", tile "
            << total_tile << " bytes, inter blocks " << total_inter
            << ", intra blocks " << total_intra << "\n";
  return 0;
}

// --video-lossless: the plain multi-frame sequence is coded by the lossless
// inter writer (`src/codec/entropy/lossless/inter/`).  That module owns its
// sequence/frame headers (base_q_idx = 0, CodedLossless = 1, TxMode =
// ONLY_4X4), the lossless DPCM/WHT/IDTX block syntax and FSC coefficients, so
// the CLI only supplies the pictures and writes the temporal units/recon.
int run_video_lossless_encode(const Options& options,
                              const av2enc::Y4mHeader& y4m,
                              const std::vector<OwnedVideoFrame>& owned) {
  std::string error;
  av2enc::IvfHeader ivf;
  ivf.width = static_cast<std::uint16_t>(y4m.width);
  ivf.height = static_cast<std::uint16_t>(y4m.height);
  ivf.fps_num = y4m.fps_num;
  ivf.fps_den = y4m.fps_den;
  av2enc::IvfWriter writer;
  if (!writer.open(options.output, ivf, error)) {
    std::cerr << "av2enc: error: " << error << "\n";
    return 1;
  }

  std::vector<av2enc::lossless::inter::Picture> pictures;
  pictures.reserve(owned.size());
  for (const OwnedVideoFrame& frame : owned) {
    av2enc::lossless::inter::Picture picture;
    picture.width = static_cast<int>(frame.width);
    picture.height = static_cast<int>(frame.height);
    picture.format = av2enc::lossless::inter::ChromaFormat::k420;
    picture.bit_depth = 8;
    picture.y.assign(frame.y.begin(), frame.y.end());
    picture.u.assign(frame.u.begin(), frame.u.end());
    picture.v.assign(frame.v.begin(), frame.v.end());
    pictures.push_back(std::move(picture));
  }

  av2enc::lossless::inter::Options lossless;
  lossless.bit_depth = 8;
  lossless.temporal_layers = options.video_tlayers;
  lossless.b_schedule = options.video_lossless_b;
  lossless.adaptive_cdf = options.adaptive_cdf;
  lossless.enable_fsc = true;
  lossless.enable_dpcm = true;
  lossless.force_intra_fsc = options.lossless_force_fsc;
  lossless.intra_period = options.video_lossless_intra_period;
  if (flag_given(options, "--video-min-block")) {
    lossless.block_size = options.video_min_block;
  } else {
    lossless.block_size = 8;  // the module's default coding block size
  }
  av2enc::lossless::inter::EncodeResult encoded;
  if (!av2enc::lossless::inter::encode_video(pictures, lossless, &encoded,
                                             &error)) {
    std::cerr << "av2enc: error: --video-lossless: " << error << "\n";
    return 1;
  }
  if (!encoded.exact) {
    std::cerr << "av2enc: error: --video-lossless reconstruction is not exact "
                 "(max |err| "
              << encoded.max_abs_recon_error << ")\n";
    return 1;
  }
  for (std::size_t i = 0; i < encoded.units.size(); ++i) {
    if (!writer.write_frame(encoded.units[i], static_cast<std::uint64_t>(i),
                            error)) {
      std::cerr << "av2enc: error: " << error << "\n";
      return 1;
    }
  }
  if (!options.video_recon.empty()) {
    std::vector<av2enc::video::VideoReconFrame> recon;
    recon.reserve(encoded.recon.size());
    for (const av2enc::lossless::inter::Picture& picture : encoded.recon) {
      av2enc::video::VideoReconFrame frame;
      frame.width = static_cast<std::uint32_t>(picture.width);
      frame.height = static_cast<std::uint32_t>(picture.height);
      frame.y.assign(picture.y.begin(), picture.y.end());
      frame.u.assign(picture.u.begin(), picture.u.end());
      frame.v.assign(picture.v.begin(), picture.v.end());
      recon.push_back(std::move(frame));
    }
    if (!write_video_recon_y4m(options.video_recon, y4m, recon, error)) {
      std::cerr << "av2enc: error: " << error << "\n";
      return 1;
    }
  }
  if (!writer.close(error)) {
    std::cerr << "av2enc: error: " << error << "\n";
    return 1;
  }

  std::uint64_t inter_blocks = 0;
  std::uint64_t intra_blocks = 0;
  for (const av2enc::lossless::inter::FrameStats& stats : encoded.stats) {
    inter_blocks += static_cast<std::uint64_t>(stats.inter_blocks);
    intra_blocks += static_cast<std::uint64_t>(stats.intra_blocks);
  }
  std::cout << "av2enc: Video lossless encode complete\n"
            << "  input:   " << options.input.string() << " ("
            << y4m.width << "x" << y4m.height << ", " << y4m.fps_num << "/"
            << y4m.fps_den << " fps)\n"
            << "  output:  " << options.output.string() << " ("
            << writer.bytes_written() << " bytes IVF)\n"
            << "  frames:  " << encoded.units.size() << " temporal unit(s), "
            << encoded.recon.size() << " output frame(s), base_q_idx 0, "
            << writer.bytes_written() << " IVF bytes, inter blocks "
            << inter_blocks << ", intra blocks " << intra_blocks << "\n";
  return 0;
}

// --video-lossless on the format video cores: 10-bit and 8-bit
// 4:2:2/4:4:4/4:0:0.  The lossless writer is depth/chroma generic; the CLI
// only converts the fmt::Picture planes into its uint16_t pictures.
// Entry point for --video-lossless on a non-4:2:0 or 10-bit input.  It reads
// the input with the format Y4M reader (the 8-bit 4:2:0 path uses the plain
// reader) and hands the pictures to the shared lossless writer.
int run_video_lossless_format_encode(
    const Options& options, const av2enc::fmt::Format& format,
    const av2enc::fmt::Y4mFormatInfo& info,
    const std::vector<av2enc::fmt::Picture>& pictures);

int run_video_lossless_any(const Options& options) {
  av2enc::fmt::Format format;
  format.chroma = options.chroma == "422"   ? av2enc::fmt::ChromaFormat::k422
                  : options.chroma == "444" ? av2enc::fmt::ChromaFormat::k444
                  : options.chroma == "400" ? av2enc::fmt::ChromaFormat::k400
                                            : av2enc::fmt::ChromaFormat::k420;
  format.depth = options.bit_depth == 10 ? av2enc::fmt::BitDepth::k10
                                         : av2enc::fmt::BitDepth::k8;

  std::size_t limit = 0;
  if (options.max_frames != 0) {
    limit = static_cast<std::size_t>(options.max_frames);
  } else if (options.p_frames != 0) {
    limit = static_cast<std::size_t>(options.p_frames) + 1U;
  }
  std::vector<av2enc::fmt::Picture> pictures;
  av2enc::fmt::Y4mFormatInfo info;
  std::string error;
  if (!av2enc::fmt::read_y4m_sequence(options.input, &pictures, &info, limit,
                                      &error)) {
    std::cerr << "av2enc: error: " << error << "\n";
    return 1;
  }
  if (info.chroma != format.chroma || info.depth != format.depth) {
    std::cerr << "av2enc: error: --bit-depth/--chroma request "
              << av2enc::fmt::bit_depth_name(format.depth) << " "
              << av2enc::fmt::chroma_format_name(format.chroma)
              << " does not match the input coloursapce "
              << av2enc::fmt::chroma_format_name(info.chroma) << " "
              << av2enc::fmt::bit_depth_name(info.depth)
              << "; the format paths never convert implicitly\n";
    return 2;
  }
  format.width = info.width;
  format.height = info.height;
  if (format.width % 8U != 0U || format.height % 8U != 0U) {
    std::cerr << "av2enc: error: --video-lossless requires dimensions "
                 "divisible by 8 (the lossless tile writer codes whole "
                 "superblocks), got "
              << format.width << "x" << format.height << "\n";
    return 2;
  }
  if (!av2enc::fmt::is_signalable(format)) {
    std::cerr << "av2enc: error: the requested format is not signalable in "
                 "AV2 v1.0.0\n";
    return 2;
  }
  if (!validate_video_sequence_length(options, pictures.size(), error)) {
    std::cerr << "av2enc: error: " << error << "\n";
    return 2;
  }
  return run_video_lossless_format_encode(options, format, info, pictures);
}

int run_video_lossless_format_encode(
    const Options& options, const av2enc::fmt::Format& format,
    const av2enc::fmt::Y4mFormatInfo& info,
    const std::vector<av2enc::fmt::Picture>& pictures) {
  std::string error;
  const int sub_x = av2enc::fmt::subsampling_x(format.chroma);
  const int sub_y = av2enc::fmt::subsampling_y(format.chroma);
  const bool has_chroma = av2enc::fmt::has_chroma(format.chroma);
  const std::uint32_t cw = has_chroma ? (format.width >> sub_x) : 0;
  const std::uint32_t ch = has_chroma ? (format.height >> sub_y) : 0;

  av2enc::lossless::inter::ChromaFormat chroma =
      av2enc::lossless::inter::ChromaFormat::k420;
  if (format.chroma == av2enc::fmt::ChromaFormat::k422) {
    chroma = av2enc::lossless::inter::ChromaFormat::k422;
  } else if (format.chroma == av2enc::fmt::ChromaFormat::k444) {
    chroma = av2enc::lossless::inter::ChromaFormat::k444;
  } else if (format.chroma == av2enc::fmt::ChromaFormat::k400) {
    chroma = av2enc::lossless::inter::ChromaFormat::k400;
  }
  if (!has_chroma && format.chroma != av2enc::fmt::ChromaFormat::k400) {
    std::cerr << "av2enc: error: --video-lossless: inconsistent monochrome "
                 "format\n";
    return 2;
  }

  std::vector<av2enc::lossless::inter::Picture> lossless_pictures;
  lossless_pictures.reserve(pictures.size());
  for (const av2enc::fmt::Picture& picture : pictures) {
    av2enc::lossless::inter::Picture out;
    out.width = static_cast<int>(format.width);
    out.height = static_cast<int>(format.height);
    out.format = chroma;
    out.bit_depth = format.depth == av2enc::fmt::BitDepth::k10 ? 10 : 8;
    out.y.assign(static_cast<std::size_t>(format.width) * format.height, 0);
    for (std::uint32_t r = 0; r < format.height; ++r) {
      std::memcpy(out.y.data() + static_cast<std::size_t>(r) * format.width,
                  picture.planes[0].samples.data() +
                      static_cast<std::size_t>(r) * picture.planes[0].stride,
                  static_cast<std::size_t>(format.width) *
                      sizeof(std::uint16_t));
    }
    out.u.assign(static_cast<std::size_t>(cw) * ch, 0);
    out.v.assign(static_cast<std::size_t>(cw) * ch, 0);
    if (has_chroma) {
      for (std::uint32_t r = 0; r < ch; ++r) {
        std::memcpy(out.u.data() + static_cast<std::size_t>(r) * cw,
                    picture.planes[1].samples.data() +
                        static_cast<std::size_t>(r) * picture.planes[1].stride,
                    static_cast<std::size_t>(cw) * sizeof(std::uint16_t));
        std::memcpy(out.v.data() + static_cast<std::size_t>(r) * cw,
                    picture.planes[2].samples.data() +
                        static_cast<std::size_t>(r) * picture.planes[2].stride,
                    static_cast<std::size_t>(cw) * sizeof(std::uint16_t));
      }
    }
    lossless_pictures.push_back(std::move(out));
  }

  av2enc::lossless::inter::Options lossless;
  lossless.bit_depth = format.depth == av2enc::fmt::BitDepth::k10 ? 10 : 8;
  lossless.temporal_layers = options.video_tlayers;
  lossless.b_schedule = options.video_lossless_b;
  lossless.adaptive_cdf = options.adaptive_cdf;
  lossless.enable_fsc = true;
  lossless.enable_dpcm = true;
  lossless.force_intra_fsc = options.lossless_force_fsc;
  lossless.intra_period = options.video_lossless_intra_period;
  lossless.block_size =
      flag_given(options, "--video-min-block") ? options.video_min_block : 8;
  av2enc::lossless::inter::EncodeResult encoded;
  if (!av2enc::lossless::inter::encode_video(lossless_pictures, lossless,
                                             &encoded, &error)) {
    std::cerr << "av2enc: error: --video-lossless: " << error << "\n";
    return 1;
  }
  if (!encoded.exact) {
    std::cerr << "av2enc: error: --video-lossless reconstruction is not exact "
                 "(max |err| "
              << encoded.max_abs_recon_error << ")\n";
    return 1;
  }

  av2enc::IvfHeader ivf;
  ivf.width = static_cast<std::uint16_t>(format.width);
  ivf.height = static_cast<std::uint16_t>(format.height);
  ivf.fps_num = info.fps_num;
  ivf.fps_den = info.fps_den;
  av2enc::IvfWriter writer;
  if (!writer.open(options.output, ivf, error)) {
    std::cerr << "av2enc: error: " << error << "\n";
    return 1;
  }
  for (std::size_t i = 0; i < encoded.units.size(); ++i) {
    if (!writer.write_frame(encoded.units[i], static_cast<std::uint64_t>(i),
                            error)) {
      std::cerr << "av2enc: error: " << error << "\n";
      return 1;
    }
  }
  if (!options.video_recon.empty()) {
    std::vector<av2enc::video::VideoReconFrame> recon;
    recon.reserve(encoded.recon.size());
    for (const av2enc::lossless::inter::Picture& picture : encoded.recon) {
      av2enc::video::VideoReconFrame frame;
      frame.width = static_cast<std::uint32_t>(picture.width);
      frame.height = static_cast<std::uint32_t>(picture.height);
      frame.format = format;
      frame.y16 = picture.y;
      frame.u16 = picture.u;
      frame.v16 = picture.v;
      recon.push_back(std::move(frame));
    }
    if (!write_video_recon_y4m_format(options.video_recon, format,
                                      info.fps_num, info.fps_den, recon,
                                      error)) {
      std::cerr << "av2enc: error: " << error << "\n";
      return 1;
    }
  }
  if (!writer.close(error)) {
    std::cerr << "av2enc: error: " << error << "\n";
    return 1;
  }

  std::uint64_t inter_blocks = 0;
  std::uint64_t intra_blocks = 0;
  for (const av2enc::lossless::inter::FrameStats& stats : encoded.stats) {
    inter_blocks += static_cast<std::uint64_t>(stats.inter_blocks);
    intra_blocks += static_cast<std::uint64_t>(stats.intra_blocks);
  }
  std::cout << "av2enc: Video lossless encode complete\n"
            << "  input:   " << options.input.string() << " ("
            << format.width << "x" << format.height << ", "
            << info.colorspace << ", " << info.fps_num << "/" << info.fps_den
            << " fps)\n"
            << "  output:  " << options.output.string() << " ("
            << writer.bytes_written() << " bytes IVF)\n"
            << "  frames:  " << encoded.units.size() << " temporal unit(s), "
            << encoded.recon.size() << " output frame(s), base_q_idx 0, "
            << "inter blocks " << inter_blocks << ", intra blocks "
            << intra_blocks << "\n";
  return 0;
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

  if (options.video_lossless) {
    return run_video_lossless_encode(options, y4m, owned);
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
  video_options.high_tier = options.video_tier_high;
  video_options.multi_sequence = options.video_multi_sequence;
  video_options.lcr_full = options.video_lcr_full;
  if (options.video_timing) {
    if (y4m.fps_num == 0 || y4m.fps_den == 0) {
      std::cerr << "av2enc: error: --video-timing needs a non-zero Y4M frame "
                   "rate (num_units_in_display_tick/time_scale)\n";
      return 1;
    }
    video_options.video_timing = true;
    video_options.video_decoder_model = options.video_decoder_model;
    video_options.decoder_buffer_delay = options.decoder_buffer_delay;
    video_options.encoder_buffer_delay = options.encoder_buffer_delay;
    video_options.low_delay_mode = options.low_delay_mode;
    video_options.timing_num_units_in_display_tick = y4m.fps_den;
    video_options.timing_time_scale = y4m.fps_num;
  }
  video_options.num_total_refs = options.video_refs;
  video_options.num_ref_frames = options.video_num_ref_frames;
  video_options.implicit_ref_map = options.video_implicit_ref_map;
  video_options.compound = options.video_compound;
  video_options.compound_wedge = options.video_compound_wedge;
  video_options.compound_diffwtd = options.video_compound_diffwtd;
  video_options.compound_near_near = options.video_compound_near_near ||
                                     options.video_opfl || options.video_refine_mv;
  video_options.opfl_refine_type = options.video_opfl ? 1 : 0;
  video_options.enable_refinemv = options.video_refine_mv;
  video_options.enable_cwp_weighted = options.video_cwp_weighted;
  video_options.enable_imp_msk_bld = options.video_imp_mask;
  video_options.enable_df_sub_pu = options.video_df_sub_pu;
  video_options.b_frames = static_cast<int>(options.b_frames);
  video_options.video_gops = options.video_gops;
  video_options.enable_inter_residual = options.video_inter_residual;
  video_options.enable_sdp = options.video_sdp;
  video_options.enable_inter_ddt = options.video_ddt;
  video_options.enable_tx_search = options.video_tx_search;
  video_options.tx_rd_lambda = options.video_tx_lambda;
  video_options.enable_deblock = options.video_deblock;
  video_options.enable_segmentation = options.video_segmentation;
  video_options.enable_delta_q = options.video_delta_q;
  video_options.delta_q_res = options.video_delta_q_res;
  video_options.per_segment_q = options.video_per_segment_q;
  video_options.seg_q_delta = options.video_seg_delta;
  video_options.enable_uv_delta_q = options.video_uv_delta_q;
  video_options.uv_delta_q = options.video_uv_delta_q_value;
  if (options.video_uv_delta_q) {
    // §5.4.8: separate U/V deltas with the DC fields enabled.  equal_ac_dc_q=0
    // is required to code base_uv_dc_delta_q / uv_dc_delta_q_enabled.  U uses
    // +D and V uses -D so diff_uv_delta=1 exercises both chroma planes.
    video_options.seq_delta_q = av2enc::dcdeltaq::SequenceDeltaQ{};
    video_options.seq_delta_q.equal_ac_dc_q = false;
    video_options.seq_delta_q.separate_uv_delta_q = true;
    video_options.seq_delta_q.uv_dc_delta_q_enabled = true;
    video_options.seq_delta_q.uv_ac_delta_q_enabled = true;
    video_options.seq_delta_q.base_uv_dc_delta_q = 0;
    video_options.seq_delta_q.base_uv_ac_delta_q = 0;
    video_options.frame_delta_q = av2enc::dcdeltaq::FrameDeltaQ{};
    video_options.frame_delta_q.delta_q_u_dc = options.video_uv_delta_q_value;
    video_options.frame_delta_q.delta_q_v_dc = -options.video_uv_delta_q_value;
    video_options.frame_delta_q.delta_q_u_ac = options.video_uv_delta_q_value;
    video_options.frame_delta_q.delta_q_v_ac = -options.video_uv_delta_q_value;
  }
  video_options.enable_ext_seg =
      options.video_segments > 8;
  video_options.num_segments =
      options.video_segments == 0 ? 2 : options.video_segments;
  video_options.enable_seg_skip = options.video_seg_skip;
  video_options.enable_seg_globalmv = options.video_seg_globalmv;
  video_options.enable_seg_temporal = options.video_seg_temporal;
  video_options.enable_gme = options.video_gme;
  video_options.enable_warp = options.video_warp;
  video_options.enable_bawp = options.video_bawp;
  video_options.enable_morph = options.video_morph;
  video_options.enable_palette = options.video_palette;
  video_options.enable_ibc = options.video_ibc;
  video_options.seq_force_screen_content_tools = options.video_scc_force;
  video_options.seq_force_integer_mv = options.video_force_integer_mv;
  video_options.reduce_pb_aspect_ratio = options.video_reduce_pb_aspect;
  video_options.max_pb_aspect_ratio_log2_minus_1 =
      options.video_pb_aspect_log2_minus_1;
  video_options.seq_seg_info_present = options.video_seq_seg_info;
  video_options.seq_allow_seg_info_change = !options.video_seq_seg_frozen;
  video_options.enable_interintra = options.video_compound_interintra;
  video_options.enable_ref_frame_mvs = options.video_ref_mvs;
  video_options.ref_mvs_stress = options.video_ref_mvs_stress;
  video_options.short_refresh = options.video_short_refresh;
  video_options.refmvbank = options.video_refmvbank;
  video_options.mv_traj = options.video_mv_traj;
  video_options.same_ref_compound = options.video_same_ref_compound;
  video_options.enable_tip = options.video_tip;
  video_options.tip_hole_fill = !options.video_tip_no_hole_fill;
  video_options.tip_block_ref = options.video_tip_block_ref;
  video_options.enable_intra_full = options.video_intra_full;
  video_options.enable_cfl = options.video_cfl;
  video_options.enable_mhccp = options.video_mhccp;
  video_options.enable_qm = options.video_qm;
  video_options.user_qm = options.video_user_qm;
  video_options.enable_tcq = options.video_tcq;
  video_options.enable_cctx = options.video_cctx;
  video_options.film_grain = options.video_grain;
  video_options.enable_idtx_intra = options.video_idtx;
  video_options.enable_mrl = options.video_mrl;
  video_options.enable_ibp = options.video_ibp;
  video_options.enable_dip = options.video_dip;
  video_options.enable_ist = options.video_ist;
  video_options.me_subpel_eighth = options.video_subpel_eighth;
  video_options.mv_precision = options.video_mv_precision;
  video_options.enable_adaptive_mvd = options.video_adaptive_mvd;
  video_options.enable_mvd_sign_derive = options.video_mvd_sign_derive;
  video_options.enable_flex_mvres = options.video_flex_mvres;
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
  video_options.sb128 = options.video_sb128;
  video_options.sb256 = options.video_sb256;
  video_options.ext_partitions = options.video_ext_partitions;
  video_options.min_block_size = options.video_min_block;
  video_options.max_partition_depth = options.video_max_depth;
  video_options.me_range = options.video_me_range;
  video_options.loop_filters.cdef = options.video_cdef;
  video_options.loop_filters.cdef_num_strengths = options.video_cdef_strength;
  video_options.loop_filters.cdef_damping = options.video_cdef_damping;
  video_options.loop_filters.ccso = options.video_ccso;
  video_options.loop_filters.ccso_bands_log2 = options.video_ccso_bands;
  video_options.loop_filters.wiener = options.video_wiener;
  video_options.loop_filters.pc_wiener = options.video_pc_wiener;
  video_options.loop_filters.chroma_lr = options.video_chroma_lr;
  video_options.loop_filters.gdf = options.video_gdf;
  video_options.trace = options.video_trace;
  video_options.cpu_threads = options.cpu_threads;
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
    if (options.video_brt) {
      std::vector<std::uint8_t> brt;
      if (!make_brt_obu(options.brt_time, brt, error) ||
          !insert_before_first_frame_obu(result.units[i], brt, error)) {
        std::cerr << "av2enc: error: " << error << "\n";
        return 1;
      }
    }
    // Neutral metadata is placed once, as suffix units of the first temporal
    // unit.  It is decoder-invariant (§5.17.2 preserves coded output order).
    if (i == 0 &&
        !append_neutral_metadata_obus(result.units[i],
                                      metadata_cli_options(options), nullptr,
                                      options.metadata_timecode,
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
            << ", intra blocks " << total_intra;
  if (result.temporal_candidates != 0) {
    std::cout << ", temporal MV candidates " << result.temporal_candidates;
  }
  if (result.temporal_selected != 0) {
    std::cout << ", temporal-selected leaves " << result.temporal_selected;
  }
  if (result.same_ref_blocks != 0) {
    std::cout << ", same-ref compound blocks " << result.same_ref_blocks;
  }
  if (result.refmvbank_candidates != 0) {
    std::cout << ", refmvbank candidates " << result.refmvbank_candidates;
  }
  if (result.refmvbank_selected != 0) {
    std::cout << ", refmvbank-selected leaves " << result.refmvbank_selected;
  }
  if (result.mv_traj_derived != 0) {
    std::cout << ", mv-traj derived candidates " << result.mv_traj_derived;
  }
  if (result.short_refresh_frames != 0) {
    std::cout << ", short refresh frames " << result.short_refresh_frames;
  }
  if (result.amvd_selected != 0) {
    std::cout << ", AMVD leaves " << result.amvd_selected;
  }
  if (result.flex_precision_selected != 0) {
    std::cout << ", flex-precision leaves " << result.flex_precision_selected;
  }
  if (result.derived_sign_selected != 0) {
    std::cout << ", derived-sign leaves " << result.derived_sign_selected;
  }
  if (result.ref_mvs_warp_cells != 0) {
    std::cout << ", §7.22 warp 8x8 cells " << result.ref_mvs_warp_cells;
  }
  if (result.ref_mvs_wedge_dropped != 0) {
    std::cout << ", §7.22 wedge allow-list cells "
              << result.ref_mvs_wedge_dropped;
  }
  std::cout << "\n";
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
  if (options.metadata_timecode || options.metadata_scan_type ||
      options.metadata_cll || options.metadata_mdcv || options.metadata_icc ||
      options.metadata_temporal_point || options.metadata_frame_hash ||
      options.metadata_banding || options.metadata_user_data ||
      options.video_brt) {
    std::cerr << "av2enc: error: --metadata-* is not wired into the --svc-layers "
                 "layered writer\n";
    return 2;
  }
  av2enc::svc::SvcEncodeOptions svc;
  if (!parse_svc_layer_specs(options.svc_layers, svc.config.spatial_layers,
                             error)) {
    std::cerr << "av2enc: error: " << error << "\n";
    return 2;
  }
  svc.config.temporal_layers = static_cast<std::uint8_t>(options.svc_temporal);
  svc.config.embedded_layers = static_cast<std::uint8_t>(options.svc_embedded);
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

  // Per-frame rate control for the layered writer.  The plan is indexed by
  // display order; svc_stream shifts it by the embedded-layer quality step so
  // every mlayer keeps its own quality rung.  VBR reuses the same two-pass
  // calibration + global-offset bisection as the flat --video path, measuring
  // the summed tile payload of every emitted temporal unit.
  std::vector<std::uint8_t> frame_qp;
  std::vector<std::uint8_t> planned_qp;
  int svc_vbr_iterations = 0;
  double svc_vbr_target_kbps = 0.0;
  if (options.rc_mode != "cqp") {
    std::vector<std::uint32_t> pass1_payload_bits;
    if (options.rc_mode == "vbr") {
      av2enc::svc::SvcEncodeOptions stats_options = svc;
      stats_options.video.base_q_idx = static_cast<std::uint8_t>(options.qp);
      av2enc::svc::SvcStreamResult stats;
      if (!av2enc::svc::encode_svc_sequence(inputs, stats_options, stats,
                                            error)) {
        std::cerr << "av2enc: error: --rc vbr statistics pass: " << error
                  << "\n";
        return 1;
      }
      pass1_payload_bits.reserve(stats.unit_payload_bytes.size());
      for (const std::uint32_t bytes : stats.unit_payload_bytes) {
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
    planned_qp = frame_qp;
    svc.video.frame_base_q_idx = frame_qp;
    svc.video.base_q_idx = frame_qp.front();
  }

  av2enc::svc::SvcStreamResult result;
  if (options.rc_mode == "vbr") {
    const double duration =
        static_cast<double>(inputs.size()) *
        static_cast<double>(y4m.fps_den) / static_cast<double>(y4m.fps_num);
    svc_vbr_target_kbps = options.rc_bitrate_kbps;
    const double target_bits = svc_vbr_target_kbps * 1000.0 * duration;
    const auto apply_offset = [&](int offset) {
      frame_qp.resize(planned_qp.size());
      for (std::size_t i = 0; i < planned_qp.size(); ++i) {
        frame_qp[i] = static_cast<std::uint8_t>(
            std::clamp(static_cast<int>(planned_qp[i]) + offset, 1, 255));
      }
      svc.video.frame_base_q_idx = frame_qp;
      svc.video.base_q_idx = frame_qp.front();
    };
    int low = -254;
    int high = 254;
    int offset = 0;
    double best_error = std::numeric_limits<double>::infinity();
    av2enc::svc::SvcStreamResult best_result;
    std::vector<std::uint8_t> best_qp = planned_qp;
    while (low <= high && svc_vbr_iterations < 12) {
      ++svc_vbr_iterations;
      apply_offset(offset);
      av2enc::svc::SvcStreamResult candidate;
      if (!av2enc::svc::encode_svc_sequence(inputs, svc, candidate, error)) {
        std::cerr << "av2enc: error: --rc vbr pass " << svc_vbr_iterations
                  << ": " << error << "\n";
        return 1;
      }
      std::uint64_t measured = 0;
      for (const std::uint32_t bytes : candidate.unit_payload_bytes) {
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
        low = offset + 1;
      } else {
        high = offset - 1;
      }
      offset = low + (high - low) / 2;
    }
    result = std::move(best_result);
    frame_qp = best_qp;
    svc.video.frame_base_q_idx = frame_qp;
    svc.video.base_q_idx = frame_qp.front();
  } else if (!av2enc::svc::encode_svc_sequence(inputs, svc, result, error)) {
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

  // Per-layer reconstruction dump.  A single xlayer/m0 stream writes exactly
  // --video-recon; a layered stream writes <stem>_x<X>m<M><ext> next to it so
  // the decoder gate can compare every layer/mlayer sample-exactly.
  if (!options.video_recon.empty()) {
    const bool single = result.layers.size() == 1 &&
                        result.layers.front().mlayer_recon.size() == 1;
    for (const av2enc::svc::SvcLayerOutput& layer : result.layers) {
      for (std::size_t m = 0; m < layer.mlayer_recon.size(); ++m) {
        std::filesystem::path path = options.video_recon;
        if (!single) {
          std::ostringstream suffix;
          suffix << "_x" << layer.xlayer_id << "m" << m;
          path = options.video_recon.parent_path() /
                 (options.video_recon.stem().string() + suffix.str() +
                  options.video_recon.extension().string());
        }
        av2enc::Y4mHeader recon_header = y4m;
        recon_header.width = static_cast<std::uint32_t>(layer.width);
        recon_header.height = static_cast<std::uint32_t>(layer.height);
        const std::vector<av2enc::video::VideoReconFrame>& recon =
            layer.mlayer_recon[m].empty() && m + 1 == layer.mlayer_recon.size()
                ? layer.recon
                : layer.mlayer_recon[m];
        if (!write_video_recon_y4m(path, recon_header, recon, error)) {
          std::cerr << "av2enc: error: " << error << "\n";
          return 1;
        }
      }
    }
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
  if (!frame_qp.empty()) {
    const double duration =
        static_cast<double>(inputs.size()) *
        static_cast<double>(y4m.fps_den) / static_cast<double>(y4m.fps_num);
    const double kbps = duration > 0.0
                            ? static_cast<double>(total_payload) * 8.0 /
                                  duration / 1000.0
                            : 0.0;
    std::cout << "  rc:      " << options.rc_mode;
    if (options.rc_mode == "crf") {
      std::cout << " " << options.rc_crf << ", achieved " << kbps
                << " kbps (payload)\n";
    } else if (options.rc_mode == "vbr") {
      const double error_pct =
          svc_vbr_target_kbps > 0.0
              ? (kbps - svc_vbr_target_kbps) / svc_vbr_target_kbps * 100.0
              : 0.0;
      std::cout << " target " << svc_vbr_target_kbps << " kbps (closed-loop "
                << svc_vbr_iterations << " pass(es)), achieved " << kbps
                << " kbps (" << (error_pct >= 0.0 ? "+" : "") << error_pct
                << "% error, payload)\n";
    }
    std::cout << "  frame QP:";
    for (const std::uint8_t qp : frame_qp) {
      std::cout << " " << static_cast<int>(qp);
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

  if (options.still_gpu) {
    const auto reject_cpu_only = [](const char* flag) {
      std::cerr << "av2enc: error: --still-gpu: " << flag
                << " is not ported to the GPU pipeline yet; use --still (CPU)\n";
      return 2;
    };
    const bool still_gpu_format =
        options.bit_depth != 8 || options.chroma != "420";
    if (still_gpu_format && (options.still_tcq || options.still_rdoq)) {
      // The 10-bit / non-4:2:0 product writer runs the `hbd` GPU block stage,
      // which has no §7.16 decision stage; reject instead of dropping the flag.
      std::cerr << "av2enc: error: --still-gpu with --bit-depth 10 / "
                   "--chroma 422|444|400 does not support --still-tcq or "
                   "--still-rdoq yet; use --still (CPU)\n";
      return 2;
    }
    if (options.still_user_qm) return reject_cpu_only("--still-user-qm");
    // --still-tcq now has a GPU stage (device §7.16 rate model + §5.4.8
    // trellis); gpu_m3_options_supported() rejects only the combinations the
    // CPU reference itself refuses.
    if (options.still_tx_partition) {
      return reject_cpu_only("--still-tx-partition");
    }
    if (options.still_ddt) return reject_cpu_only("--still-ddt");
    // CDEF, CCSO, loop restoration (Wiener NS, chroma LR, PC-Wiener, GDF),
    // film-grain signalling (including the adaptive model), content
    // interpretation, TCQ and RDOQ now have GPU stages.  The remaining tools
    // are rejected by name.
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
  if ((options.video_tier_high || options.video_multi_sequence ||
       options.video_lcr_full) &&
      !options.video) {
    std::cerr << "av2enc: error: --video-tier/--video-multi-sequence/"
                 "--video-lcr-full are wired into the non-single-picture "
                 "--video sequence header only\n";
    return 2;
  }
  if (options.video_multi_sequence &&
      (options.video_mfh || options.video_timing)) {
    std::cerr << "av2enc: error: --video-multi-sequence needs the direct "
                 "seq_header_id_in_frame_header field; --video-mfh and "
                 "--video-timing take the single-header form\n";
    return 2;
  }
  if ((options.video_multi_sequence || options.video_lcr_full) &&
      (options.b_frames > 0 || options.video_tlayers > 0 || options.video_tip ||
       options.video_long_term || options.video_olk ||
       options.video_leading > 0 || options.video_sef || options.video_switch ||
       options.video_ras || options.tiles != "1x1" ||
       options.video_lossless || options.bit_depth != 8 ||
       options.chroma != "420")) {
    std::cerr << "av2enc: error: --video-multi-sequence/--video-lcr-full are "
                 "wired into the plain 8-bit 4:2:0 single-tile --video chain; "
                 "the B/temporal/TIP/random-access schedules and the format "
                 "video core write their own headers\n";
    return 2;
  }
  if (options.video_tier_high &&
      (options.b_frames > 0 || options.video_tlayers > 0 || options.video_tip ||
       options.video_long_term || options.video_olk ||
       options.video_leading > 0 || options.video_sef || options.video_switch ||
       options.video_ras || options.tiles != "1x1" ||
       options.video_lossless || options.bit_depth != 8 ||
       options.chroma != "420")) {
    std::cerr << "av2enc: error: --video-tier high is wired into the plain "
                 "8-bit 4:2:0 --video sequence header only; the B/temporal/"
                 "TIP/random-access schedules, --tiles and the format video "
                 "core write their own sequence headers\n";
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
        if (options.qp != 90) {
          std::cerr << "av2enc: error: multi-tile --still-gpu pins base_q_idx "
                       "90 (the device quantiser step); use --qp 90 or --still "
                       "(CPU)\n";
          return 2;
        }
        if (options.metadata_timecode || options.metadata_scan_type ||
            options.metadata_cll || options.metadata_mdcv ||
            options.metadata_icc || options.metadata_temporal_point ||
            options.metadata_frame_hash || options.metadata_banding ||
            options.metadata_user_data) {
          std::cerr << "av2enc: error: multi-tile --still-gpu does not apply "
                       "the neutral metadata OBUs\n";
          return 2;
        }
      }
      if (!options.m1 && !options.still_gpu) {
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
          {options.metadata_cll, "--metadata-cll"},
          {options.metadata_mdcv, "--metadata-mdcv"},
          {options.metadata_icc, "--metadata-icc"},
          {options.metadata_temporal_point, "--metadata-temporal-point"},
          {options.metadata_frame_hash, "--metadata-frame-hash"},
          {options.metadata_banding, "--metadata-banding"},
          {options.metadata_user_data, "--metadata-user-data"},
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
    if (options.video_lossless) {
      return run_video_lossless_any(options);
    }
    if (options.video_user_qm) {
      std::cerr << "av2enc: error: --video-user-qm is wired for the 8-bit "
                   "4:2:0 video core only\n";
      return 2;
    }
    if (options.video_segmentation || options.video_delta_q ||
        options.video_per_segment_q || options.video_uv_delta_q) {
      std::cerr << "av2enc: error: segmentation/delta-Q is only wired on the "
                   "8-bit 4:2:0 video core; use --bit-depth 8 --chroma 420\n";
      return 2;
    }
    if (!video_format_wired(options.bit_depth, options.chroma)) {
      std::cerr << "av2enc: error: --video supports 8-bit 4:2:0 and 10-bit "
                   "4:2:0; the remaining --bit-depth/--chroma combinations "
                   "are not verified on the video path yet\n";
      return 2;
    }
    // The format video core reads its own strict Y4M stream and never falls
    // back to the 8-bit 4:2:0 reader below.
    return run_video_format_encode(options);
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
    if (options.metadata_timecode || options.metadata_scan_type ||
        options.metadata_cll || options.metadata_mdcv ||
        options.metadata_icc || options.metadata_temporal_point ||
        options.metadata_frame_hash || options.metadata_banding ||
        options.metadata_user_data || options.video_brt) {
      std::cerr << "av2enc: error: --metadata-* is not wired into the "
                   "--bit-depth/--chroma format still writers\n";
      return 2;
    }
    // Name the offending flag: the format writer signals none of these tools.
    // 10-bit CDEF/CCSO/LR/GDF are wired; the remaining tools are not.
    const std::pair<bool, const char*> unsupported_by_format[] = {
        {options.still_single, "--still-single"},
        {options.still_dc, "--still-dc"},
        {options.still_full_intra, "--still-full-intra"},
        {options.still_uv_full, "--still-uv-full"},
        {options.still_rdo_intra, "--still-rdo-intra"},
        {options.still_rdoq, "--still-rdoq"},
        {options.still_tx_type, "--still-tx-type"},
        {flag_given(options, "--still-ccso-bands"), "--still-ccso-bands"},
        {options.still_sb128, "--still-sb128"},
        {options.still_sb256, "--still-sb256"},
        {options.still_ext_partitions, "--still-ext-partitions"},
        {options.still_segmentation, "--still-segmentation"},
        {options.still_delta_q, "--still-delta-q"},
        {options.still_per_segment_q, "--still-per-segment-q"},
        {options.still_content_interpretation, "--still-content-interpretation"},
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
        {options.still_sub8, "--still-sub8"},
        {options.still_sdp, "--still-sdp"},
        {flag_given(options, "--still-cfl-ds"), "--still-cfl-ds"},
        {options.still_cfl_explicit, "--still-cfl-explicit"},
        {options.still_dip, "--still-dip"},
        {options.still_palette, "--still-palette"},
        {options.still_ibc, "--still-ibc"},
        {options.still_mhccp, "--still-mhccp"},
        {options.still_grain, "--still-grain"},
        {options.still_depth >= 0, "--still-depth"},
    };
    for (const auto& [set, flag] : unsupported_by_format) {
      if (!set) continue;
      std::cerr << "av2enc: error: " << flag
                << " is not applied by the --bit-depth/--chroma format "
                   "writer: format paths use their own minimal still writer "
                   "(CDEF/CCSO/LR at 10-bit are not implemented)\n";
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
    // §7.17 deblocking is opt-in on this path: existing 10-bit streams signal
    // it off and must stay byte-identical.
    if (flag_given(options, "--still-deblock")) {
      if (flag_given(options, "--no-deblock")) {
        std::cerr << "av2enc: error: --still-deblock and --no-deblock are "
                     "mutually exclusive\n";
        return 2;
      }
      format_options.enable_deblock = true;
    }
    // §7.18 CDEF is wired on both format writers.
    if (options.still_cdef) {
      format_options.enable_cdef = true;
      format_options.cdef_num_strengths = options.still_cdef_strengths;
    }
    // §7.19 CCSO is wired on both format writers (band-offset and
    // edge-classified forms are both searched).
    if (options.still_ccso) {
      format_options.enable_ccso = true;
      format_options.ccso_edge_only = options.still_ccso_edge;
    }
    // §7.20 loop restoration and §7.20.5 GDF on both format writers.
    if (options.still_wiener || options.still_chroma_lr ||
        options.still_pc_wiener || options.still_gdf) {
      if (options.still_pc_wiener && options.still_wiener) {
        std::cerr << "av2enc: error: --still-pc-wiener and --still-wiener "
                     "select two luma restoration filters\n";
        return 2;
      }
      format_options.enable_wiener = options.still_wiener;
      format_options.enable_chroma_lr = options.still_chroma_lr;
      format_options.enable_pc_wiener = options.still_pc_wiener;
      format_options.enable_gdf = options.still_gdf;
    }
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
    // The video path codes the §5.18.4.4 MI-aligned picture and crops the
    // output/reconstruction back to the input size.  This reader supplies
    // 8-bit 4:2:0, so only the chroma layout is a hard dimension constraint.
    if ((y4m.width & 1U) != 0 || (y4m.height & 1U) != 0) {
      std::cerr << "av2enc: error: 4:2:0 video requires even dimensions, got "
                << y4m.width << "x" << y4m.height << "\n";
      return 2;
    }
    if (options.video_gpu &&
        (y4m.width % 8U != 0U || y4m.height % 8U != 0U)) {
      std::cerr << "av2enc: error: --video-gpu is CPU-only for dimensions not "
                   "divisible by 8 (the CUDA kernels assume an MI-aligned "
                   "picture); rerun without --video-gpu, got "
                << y4m.width << "x" << y4m.height << "\n";
      return 2;
    }
    if (options.metadata_frame_hash) {
      std::cerr << "av2enc: error: --metadata-frame-hash is wired into the "
                   "still/basic writers only; the --video frame hash needs a "
                   "per-unit carrier\n";
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
                              std::uint64_t timestamp, std::string& err,
                              const av2enc::metadata::cli::Options::FramePlanes*
                                  planes = nullptr) {
    if (options.video_brt) {
      std::vector<std::uint8_t> brt;
      if (!make_brt_obu(options.brt_time, brt, err) ||
          !insert_before_first_frame_obu(unit, brt, err)) {
        return false;
      }
    }
    if (!append_neutral_metadata_obus(unit, metadata_cli_options(options),
                                      planes, options.metadata_timecode,
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

    // --still-gpu: the DPCM/FSC decision and the reversible transforms run on
    // the device; the CPU lossless module keeps the §5.20 syntax, the §8.2
    // entropy writer and the decoder-side exactness check.
    std::vector<av2enc::lossless::BlockCoding> gpu_blocks;
#if AV2ENC_STILL_GPU_CALLABLE && AV2ENC_STILL_LOSSLESS_HEADER
    if (options.still_gpu) {
      std::string gpu_error;
      if (!av2enc::cuda::still_lossless::compute_block_coding(
              picture, lossless_options, &gpu_blocks, gpu_error)) {
        return fail_encode(gpu_error);
      }
      lossless_options.precomputed = gpu_blocks.data();
    }
#else
    if (options.still_gpu) {
      return fail_encode(
          "--still-gpu: the lossless DPCM/FSC device stage is not built");
    }
#endif

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
    const auto planes = make_plane_view(
        picture.y, picture.u, picture.v, picture.width, picture.height);
    if (!write_unit(encoded.temporal_unit, 0, error, &planes)) {
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

#if AV2ENC_STILL_GPU_CALLABLE
    // GpuOptions mirrors StillOptions and adds orchestrator controls; see
    // src/cuda/pipeline/still/still_gpu.h and docs/m1-gpu.md.  One builder is
    // shared by the single-tile and multi-tile (per-tile) GPU paths so a parsed
    // flag can never reach one branch and not the other.
    const auto build_gpu_options = [&]() {
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
      gpu_options.grain_adaptive = options.still_grain_adaptive;
      gpu_options.content_interpretation =
          options.still_content_interpretation;
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
      return gpu_options;
    };
#endif

    if (options.still_gpu && !multi_tile) {
#if AV2ENC_STILL_GPU_CALLABLE
      av2enc::cuda::still_gpu::GpuOptions gpu_options = build_gpu_options();
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
      const auto planes =
          make_plane_view(encoded.recon_y, encoded.recon_u, encoded.recon_v,
                          picture.width, picture.height);
      if (!write_unit(encoded.frame, 0, error, &planes)) {
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
      bool tiled_ok = false;
#if AV2ENC_STILL_GPU_CALLABLE
      if (options.still_gpu) {
        // Per-tile plan/search/TX/recon on the device.  The CPU tile-group
        // writer still owns the tiled headers, the §8.2.2 seed and the §8.2.4
        // average-CDF bank; this callback only replaces the per-tile encoder.
        const auto gpu_tile_encoder =
            [&](const av2enc::still::StillPicture& tile_picture,
                const av2enc::still::StillOptions& tile_options,
                std::span<const std::uint16_t> cdf_seed,
                av2enc::still::tiles_stream::TileEncodeOutput* out,
                std::string* tile_encode_error) -> bool {
          av2enc::cuda::still_gpu::GpuOptions gpu_options = build_gpu_options();
          gpu_options.base_q_idx = tile_options.base_q_idx;
          gpu_options.enable_cfl = tile_options.enable_cfl;
          gpu_options.enable_deblock = tile_options.enable_deblock;
          gpu_options.adaptive_cdf = tile_options.adaptive_cdf;
          gpu_options.cdf_seed.assign(cdf_seed.begin(), cdf_seed.end());
          gpu_options.min_block_size = tile_options.min_block_size;
          gpu_options.max_partition_depth = tile_options.max_partition_depth;
          gpu_options.luma_dc_only = tile_options.luma_dc_only;
          gpu_options.verify_against_cpu = true;
          av2enc::cuda::still_gpu::GpuResult encoded;
          if (!av2enc::cuda::still_gpu::encode_still_gpu(
                  tile_picture, gpu_options, encoded, *tile_encode_error)) {
            return false;
          }
          out->tile_bytes = encoded.tile_bytes;
          out->frame = std::move(encoded.frame);
          out->recon_y = std::move(encoded.recon_y);
          out->recon_u = std::move(encoded.recon_u);
          out->recon_v = std::move(encoded.recon_v);
          out->cdf_final = std::move(encoded.cdf_final);
          return true;
        };
        tiled_ok = av2enc::still::tiles_stream::encode(
            picture, tile_config, &tiled, &tile_error, gpu_tile_encoder);
      } else
#endif
      {
        tiled_ok = av2enc::still::tiles_stream::encode(
            picture, tile_config, &tiled, &tile_error);
      }
      if (!tiled_ok) {
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
      const auto planes =
          make_plane_view(encoded.recon_y, encoded.recon_u, encoded.recon_v,
                          picture.width, picture.height);
      if (!write_unit(encoded.frame, 0, error, &planes)) {
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
    const auto planes =
        make_plane_view(encoded.recon_y, encoded.recon_u, encoded.recon_v,
                        picture.width, picture.height);
    if (!write_unit(encoded.frame, 0, error, &planes)) {
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

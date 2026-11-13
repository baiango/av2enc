// M2 two-frame P-frame integration on top of the M1 still encoder.
//
// Frame 0 is an intra keyframe. The M2-owned copy of the M1 intra syntax
// writer emits it (the M1 still integration TU was not linkable in this tree
// snapshot); the M1 tables, partition solver, intra prediction, transform and
// entropy coder are reused unchanged. Frame 1..N-1 are P frames that reference
// slot 0 and refresh slot 0.
//
// The P-frame decision path is deliberately small:
//   * partition tree from src/encoder/partition (same rules as M1)
//   * per-leaf inter candidates: GLOBALMV (zero MV), NEARMV (spatial MV stack)
//     and NEWMV (src/encoder/inter/me search quantised to 1/8 pel)
//   * per-leaf intra fallback: DC intra + DCT residual (M1 transform path)
//   * inter leaves default to skip_flag = 1; the gated `enable_inter_residual`
//     option codes a quantised inter residual (feature 1)
//
// This is a CPU correctness reference. The CUDA M2 tree is intentionally
// separate and not required for this gate.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "formats/format.h"
#include "pipeline/video/post/video_loop_filters.h"

namespace av2enc::video {

// §5.18.2 REFS_PER_FRAME: NumTotalRefs is a 3-bit value, so one frame can
// address at most seven explicit reference slots.
inline constexpr int kRefsPerFrame = 7;
// §5.4.6 explicit_num_ref_frames max (4-bit `num_ref_frames_minus_1`).
inline constexpr int kMaxNumRefFrames = 16;
// The largest DPB this encoder model manages (the flat P chain and the frame
// type plans both stop at eight slots).
inline constexpr int kMaxRefSlots = 8;

// Pixel format of one sequence.  The default is the historical 8-bit 4:2:0
// stream, which keeps every existing bitstream byte-identical.  Any other
// format goes through the high-bit-depth / chroma video core
// (`src/pipeline/video/hbd/`).
struct VideoInputFrame {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  // 8-bit packed planes (used when `format.depth == BitDepth::k8`).
  std::span<const std::uint8_t> y;
  std::span<const std::uint8_t> u;
  std::span<const std::uint8_t> v;
  // High-depth packed `uint16_t` planes, LSB-aligned, used for 10-bit.
  std::span<const std::uint16_t> y16;
  std::span<const std::uint16_t> u16;
  std::span<const std::uint16_t> v16;
  fmt::Format format{};
};

struct VideoReconFrame {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
  // High-depth reconstruction planes for the format video core.
  std::vector<std::uint16_t> y16;
  std::vector<std::uint16_t> u16;
  std::vector<std::uint16_t> v16;
  fmt::Format format{};
};

struct VideoOptions {
  // Pixel format.  `{k420, k8}` is the historical M2 core; every other
  // combination is served by the format video core (10-bit / 4:2:2 / 4:4:4 /
  // 4:0:0).  The CLI only sets it for verified formats.
  fmt::Format format{};
  // §5.18.6.1 base_q_idx.  MaxQ is 255 at 8-bit and 303 at 10-bit.
  std::uint16_t base_q_idx = 90;
  // Optional per-frame base_q_idx (one entry per input frame). Empty keeps the
  // fixed `base_q_idx` for every frame, which is the default byte-identical
  // path. Used by the opt-in --rc crf/vbr path; the frame header and the
  // coefficient-CDF qctx slice both follow the per-frame value.
  std::vector<std::uint8_t> frame_base_q_idx;
  // Opt-in §7.17 deblocking of every reconstructed frame (--video-deblock).
  // Off keeps every historical stream byte-identical.  When on, the frame
  // header signals all four apply flags with zero df_delta_q, and the
  // reconstructed frame is deblocked before it is emitted or stored as a
  // reference, so encoder and decoder references match.
  bool enable_deblock = false;
  bool enable_intra_fallback = true;
  // Opt-in full luma intra search (--video-intra-full): the still product's
  // 61-candidate SSE search instead of the DC-only fallback.  Off by default so
  // the shipped streams stay byte-identical.  Chroma stays DC_PRED.
  bool enable_intra_full = false;
  // Opt-in §7.13.5 CfL chroma decision (--video-cfl): the M1 still product's
  // {DC, CfL} candidate set with derived alpha.  Off by default so every
  // historical stream stays byte-identical.
  bool enable_cfl = false;
  // Opt-in §7.13.6–.8 MHCCP (--video-mhccp): signals enable_mhccp and searches
  // the three CFL_MULTI directions after the derived-alpha hypothesis.  Needs
  // enable_cfl; off keeps every historical stream byte-identical.
  bool enable_mhccp = false;
  // Opt-in §5.18.6.2 default-matrix QM (--video-qm): using_qmatrix = 1 with
  // level 0 (the §9.4 default matrices, no user QM OBU).  The transform chain
  // dequantizes with the per-coefficient q2, so reconstruction stays exact.
  bool enable_qm = false;
  bool enable_newmv = true;
  bool enable_nearmv = true;
  bool force_global_mv = false;
  bool force_intra = false;
  bool force_single_block = false;
  // Feature 1: inter residual.  Off by default; the baseline is skip-only.
  // With it on, the skip/residual and intra/inter decisions use a rate-aware
  // RD cost weighted by inter_rd_lambda instead of the historical SSE-only
  // comparison.
  // When true, a non-skip inter leaf codes transform coefficients for all
  // three planes.  force_inter_residual bypasses the rate heuristic and codes
  // residual on every inter leaf (gate stress mode).
  bool enable_inter_residual = false;
  // Distortion-per-bit weight for the opt-in residual RD decision.  Tuned on
  // 360p content: 4 keeps the default-path quality while cutting the stream
  // roughly 3x; the historical implicit value was ~0.5*(qlookup/8)^2 (2178
  // at QP 90), which made every residual look unaffordable.
  double inter_rd_lambda = 4.0;
  bool force_inter_residual = false;
  // Opt-in §5.18.8.1 TX_MODE_SELECT luma TX size/type/partition search
  // (--video-tx-search).  Off keeps every historical stream byte-identical;
  // when on, each coded luma residual evaluates the §5.20.6/§5.20.8 candidate
  // sets and signals the RD-best partition/type through the existing writer.
  bool enable_tx_search = false;
  // SSE-per-bit weight for the TX search (Cost = SSE + lambda * rate_bits).
  // The module's rate is a real bit count, so this is much larger than
  // inter_rd_lambda (which weights the historical estimate_tx_bits heuristic).
  // 256 measured best on the 360p sample at QP 90: with the inter residual
  // tool it lowers rate and raises PSNR slightly; the default skip-only path
  // is rate-neutral because it has no inter residual leaves to search.
  double tx_rd_lambda = 256.0;
  int min_block_size = 16;
  int max_partition_depth = 4;
  // ME integer search radius in luma samples.
  int me_range = 16;
  // Opt-in ME refinement: add the normative 1/8-pel subpel stage.
  bool me_subpel_eighth = false;
  // References per P frame (NumTotalRefs).  1 keeps the historical M2 stream;
  // 2..7 enables per-block single_ref selection over the most recent slots.
  int num_total_refs = 1;
  // §5.4.6 NumRefFrames: the size of the reference-slot pool.  0 derives the
  // historical value (2 for the single-reference chain, 8 for the
  // multi-reference model), so every existing stream stays byte-identical.
  // An explicit 4..8 selects the spec multi-slot surface and must be at least
  // num_total_refs.
  int num_ref_frames = 0;
  // B frames: N consecutive B frames per GOP (0 = the flat key + P chain).
  // Requires num_total_refs == 1 (the bframes plan owns the per-frame refs).
  int b_frames = 0;
  // Compound prediction (reference_select = 1).  Requires num_total_refs == 2;
  // the gated subset is non-joint GLOBAL_GLOBALMV with the AVERAGE blend.
  bool compound = false;
  // Masked compound (§5.20.7.16 / §7.13.3.27-.30), gated behind `compound`.
  // `compound_wedge` searches the 68x2 wedge masks; `compound_diffwtd`
  // searches both difference-weight mask types.  Both default off so the
  // historical AVERAGE-only streams stay byte-identical.
  bool compound_wedge = false;
  bool compound_diffwtd = false;
  // Frame MV precision: 0 = 1/8 pel, 1 = 1/4 pel, 2 = 1/2 pel.
  int mv_precision = 0;
  // Normative DRL reordering (§7.12.2.19).  Off = DRL_REORDER_DISABLED.
  bool drl_reorder = false;
  bool drl_reorder_always = false;
  // Long-term references (§5.4.6 long_term_frame_id_bits = 1, §5.18.2
  // long_term_id_plus_1).  Off keeps the historical header (bits = 0).
  bool long_term = false;
  // Open random access: CLK + open-loop key + long-term reference retained
  // across the RAP (§7.4.4).  Implies long_term; requires exactly 4 frames.
  bool olk = false;
  // Open-GOP leading pictures between the OLK and the first regular frame.
  // 0 = off.  N needs at least N+2 input frames.
  int leading_frames = 0;
  // Re-show the last coded picture with a show-existing-frame OBU: the last
  // picture is coded hidden (immediate_output_frame = 0) and the SEF unit that
  // follows outputs it with derive_sef_order_hint = 1 (§5.18.2, §6.17.2).
  bool sef = false;
  // Random-access switch frame: frame 1 is OBU_SWITCH (§6.2.2 type 10).
  // Requires exactly 2 input frames (CLK + SWITCH).
  bool ra_switch = false;
  // Random access with a long-term reference: CLK + SWITCH + OBU_RAS_FRAME
  // (§6.2.2 type 21, §7.4.5).  Requires exactly 3 input frames.
  bool ras = false;
  // Hierarchical-P temporal layers (§5.4.1 max_tlayer_id, §5.2.2
  // obu_tlayer_id).  0 = off.  N = one closed GOP of N frames (4 or 8).
  int temporal_layers = 0;
  // Emit only the temporal layers <= this id (a decodable substream).  -1
  // keeps every layer.
  int tlayer_drop = -1;
  // Internal: code the last picture hidden.  Only the SEF wrapper in
  // video_frames.cpp sets this; it is not a CLI option.
  bool hide_last_picture = false;
  // Opt-in uniform tile grid (--video --tiles). 0/0 (the default) keeps the
  // single-tile writer and the byte-identical stream.  Every frame is split on
  // superblock boundaries and re-emitted through
  // post::assemble_video_tile_group_from_payloads(); intra prediction is
  // clamped to the containing tile so the reconstruction stays decoder-exact.
  int tile_cols = 0;
  int tile_rows = 0;
  // SVC sequence-header layer limits (§5.4.1).  Defaults reproduce the
  // single-layer header (seq_lcr_id = 0, M = T = 1).
  std::uint8_t seq_lcr_id = 0;
  std::uint8_t max_tlayer_id = 0;
  std::uint8_t max_mlayer_id = 0;
  std::uint8_t seq_max_mlayer_cnt = 1;
  // Opt-in §8.2.4 average-CDF / §7.5 frame-end bookkeeping (--avg-cdf).
  // enable_avg_cdf/avg_cdf_type are then signalled in the sequence header.
  bool avg_cdf = false;
  // Opt-in §8.2.6 adaptation in the video tile writer (--adaptive-cdf with
  // --avg-cdf).  Off keeps disable_cdf_update = 1.
  bool adaptive_cdf = false;
  // Opt-in §5.18.2 cross-frame CDF carry (--cdf-continuity): each P frame
  // installs the primary reference's saved CDFs, signals primary_ref_frame = 0
  // and clears disable_cross_frame_cdf_init, so the decoder loads the same
  // context.
  bool cdf_continuity = false;
  // Opt-in §5.4.1/§5.4.12/§5.4.13 timing: decoder_model_info_present_flag with
  // num_units_in_decoding_tick plus a content-interpretation OBU carrying
  // timing_info().  Pinned dav2d accepts this form (the nested
  // seq_decoder_model_info block and BRT are rejected).
  bool video_timing = false;
  std::uint32_t timing_num_units_in_display_tick = 0;
  std::uint32_t timing_time_scale = 0;
  // Opt-in §5.7 multi-frame header on the non-single-picture path
  // (--video-mfh).  One MFH (id 1) in the key temporal unit; every frame
  // header then sets cur_mfh_id = 1 and omits seq_header_id_in_frame_header.
  bool mfh = false;
  // Global motion (§5.18.9, §7.12.2, §7.13.3.19-.24).  Off by default: the
  // historical stream keeps `enable_global_motion = 0` and every GLOBALMV
  // candidate uses a zero MV.  `enable_gme` estimates one model per explicit
  // reference from a deterministic integer block-MV grid and signals it in the
  // frame header.  `enable_warp` additionally predicts GLOBALMV blocks with the
  // §7.13.3.19/.20 warped predictor when the model is non-identity (otherwise
  // the projected §7.12.2.2 centre MV drives ordinary translational MC).
  bool enable_gme = false;
  bool enable_warp = false;
  // §5.20.7.6 / §7.13.3.25 block adaptive weighted prediction (--video-bawp).
  // Off keeps every historical stream byte-identical; when on, eligible
  // single-prediction luma blocks evaluate the fitted BAWP predictor and code
  // use_bawp.  Chroma BAWP is never signalled (use_bawp_chroma = 0).
  bool enable_bawp = false;
  // Opt-in TIP sequence (--video-tip, §7.10/§7.13.3.2).  Codes one future
  // anchor P and the remaining pictures as TIP_FRAME_AS_OUTPUT frames that
  // reference the anchor plus the previous picture.  Requires 3..8 frames, a
  // single tile and none of the frame-type/loop-filter/CDF-chain modes; off
  // keeps every existing stream byte-identical.
  bool enable_tip = false;
  // Opt-in M2 loop filters (--video-cdef, ...).  All default off, so a default
  // stream is byte-identical.  They run on the reconstructed frame before it
  // becomes the output picture and the reference for later frames.
  post::VideoLoopFilterOptions loop_filters;
  // Debug trace to stderr.
  bool trace = false;
};

struct VideoSequenceResult {
  // One temporal unit per input frame (TD + [SEQ] + frame OBU).
  std::vector<std::vector<std::uint8_t>> units;
  // Decoder output pictures, in display/output order.  Frame types that code
  // non-output pictures (OLK, non-output leading) leave those pictures out, so
  // this can be shorter than the coded frame count.
  std::vector<VideoReconFrame> recon;
  // One entry per **coded** frame unit in decode order (an SEF unit counts as
  // one): index into `recon` when the unit outputs a picture, -1 when it is
  // decoded but never displayed.
  std::vector<std::int32_t> frame_output_index;
  std::vector<std::uint32_t> frame_bytes;
  std::vector<std::uint32_t> tile_bytes;
  std::vector<std::uint32_t> inter_blocks;
  std::vector<std::uint32_t> intra_blocks;
};

// §5.4.6 NumRefFrames actually signalled for these options: the explicit
// `num_ref_frames`, otherwise 8 for the multi-reference model and 2 for the
// historical single-reference chain.
int resolved_num_ref_frames(const VideoOptions& options);

bool encode_video_sequence(std::span<const VideoInputFrame> frames,
                        const VideoOptions& options, VideoSequenceResult& result,
                        std::string& error);

// Format video core dispatch (10-bit 4:2:0/4:2:2/4:4:4/4:0:0 and 8-bit
// 4:2:2/4:4:4/4:0:0).  `encode_video_sequence()` calls this when
// `options.format` is not the historical 8-bit 4:2:0.  The CUDA sibling is
// `encode_video_sequence_gpu()` in the video GPU target.
bool encode_video_sequence_format(std::span<const VideoInputFrame> frames,
                                  const VideoOptions& options,
                                  VideoSequenceResult& result,
                                  std::string& error);

}  // namespace av2enc::video

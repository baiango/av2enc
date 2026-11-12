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

namespace av2enc::video {

struct VideoInputFrame {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::span<const std::uint8_t> y;
  std::span<const std::uint8_t> u;
  std::span<const std::uint8_t> v;
};

struct VideoReconFrame {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
};

struct VideoOptions {
  std::uint8_t base_q_idx = 90;
  // Optional per-frame base_q_idx (one entry per input frame). Empty keeps the
  // fixed `base_q_idx` for every frame, which is the default byte-identical
  // path. Used by the opt-in --rc crf/vbr path; the frame header and the
  // coefficient-CDF qctx slice both follow the per-frame value.
  std::vector<std::uint8_t> frame_base_q_idx;
  bool enable_intra_fallback = true;
  bool enable_newmv = true;
  bool enable_nearmv = true;
  bool force_global_mv = false;
  bool force_intra = false;
  bool force_single_block = false;
  // Feature 1: inter residual.  Off by default; the baseline is skip-only.
  // When true, a non-skip inter leaf codes transform coefficients for all
  // three planes.  force_inter_residual bypasses the rate heuristic and codes
  // residual on every inter leaf (gate stress mode).
  bool enable_inter_residual = false;
  bool force_inter_residual = false;
  int min_block_size = 16;
  int max_partition_depth = 4;
  // ME integer search radius in luma samples.
  int me_range = 16;
  // Opt-in ME refinement: add the normative 1/8-pel subpel stage.
  bool me_subpel_eighth = false;
  // References per P frame (NumTotalRefs).  1 keeps the historical M2 stream;
  // 2..3 enables per-block single_ref selection over the most recent slots.
  int num_total_refs = 1;
  // B frames: N consecutive B frames per GOP (0 = the flat key + P chain).
  // Requires num_total_refs == 1 (the bframes plan owns the per-frame refs).
  int b_frames = 0;
  // Compound prediction (reference_select = 1).  Requires num_total_refs == 2;
  // the gated subset is non-joint GLOBAL_GLOBALMV with the AVERAGE blend.
  bool compound = false;
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

bool encode_video_sequence(std::span<const VideoInputFrame> frames,
                        const VideoOptions& options, VideoSequenceResult& result,
                        std::string& error);

}  // namespace av2enc::video

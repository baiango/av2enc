#pragma once

#include "pipeline/still/still_headers.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace av2enc::still {

struct StillPicture {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::span<const std::uint8_t> y;
  std::span<const std::uint8_t> u;
  std::span<const std::uint8_t> v;
};

struct StillOptions {
  // 1..255.  The coefficient CDFs come from the §8.3.1 COEFF_CDF_Q_CTXS slice
  // selected by this raw value (<=90, <=140, <=190, else), and the quantiser
  // step is qlookup(base_q_idx).  0 (lossless) is not implemented.
  std::uint8_t base_q_idx = 90;
  bool enable_cfl = true;
  bool enable_deblock = true;
  bool enable_intra_edge_filter = true;
  bool adaptive_cdf = false;
  // Opt-in IBP (§5.4.5 enable_ibp, §7.13.2.7-.12).  No per-block syntax: the
  // decoder derives applyIbp/useIBP from the mode and angle delta, so the whole
  // prediction path (DC post-filter, directional edge rules, two-angle blend)
  // must use ibp::predict().  Default off keeps the pinned streams unchanged.
  bool enable_ibp = false;
  // Opt-in MRL (§5.4.5 enable_mrls, §5.20.5.5 mrl_index/mrl_sec_index).
  // Directional luma only; implies the directional luma search because
  // non-directional modes cannot signal mrl_index.  Default off.
  bool enable_mrl = false;
  // §5.4.5 cfl_ds_filter_index f(2): sequence-level CfL luma down-sampling
  // filter (§7.13.5).  0 is the historical filter; 3 aliases 0.
  int cfl_ds_filter = 0;
  // Opt-in explicit CfL alpha (§5.20.7.32 CFL_EXPLICIT).  The per-block
  // decision compares the derived and explicit predictions by SSE and keeps
  // derived on a tie, so cfl_index = 1 stays the stable default.
  bool cfl_explicit = false;
  // Opt-in DIP (§5.4.5 enable_dip, §5.20.7.9 dip_mode_info, §7.13.2.3-.6).
  // Luma DC alternative: the search compares the DIP candidate with ordinary
  // DC by SSE.  Default off.
  bool enable_dip = false;
  // Opt-in palette intra (§5.20.8) for screen content.  Requires the frame
  // allow_screen_content_tools bit.  Luma only; the block codes eob=0
  // (transform-skip) on every plane, so the reconstruction is the palette
  // prediction plus the normal chroma residual.  The decision is SSE-only.
  bool palette = false;
  // Opt-in intra block copy (§5.20.5.3/.4, §7.13.3.18) for screen content.
  // Requires the frame allow_screen_content_tools + allow_intrabc bits and
  // force_integer_mv.  The encoder only selects exact luma copies (skip_flag=1,
  // no residual); `--still-ibc-search-step` bounds the local BV search.
  bool ibc = false;
  int ibc_search_step = 8;
  // Opt-in MHCCP (CFL_MULTI, §7.13.6-.8).  Requires CfL.  The decision compares
  // the derived/explicit hypotheses with the MHCCP direction search by SSE.
  bool mhccp = false;
  // Partition search floor. 8 is the M1 target; 4 exercises the 4x4 path.
  int min_block_size = 8;
  int max_partition_depth = 4;
  // Opt-in sub-8 coding blocks (4x4/4x8/8x4/4x16/16x4/4x32/32x4).  Enables the
  // 4-sample partition floor and depth needed to reach it; chroma ownership is
  // carried by HasChroma/ChromaMi.  CPU-only; default off keeps every pinned
  // stream unchanged.
  bool sub8 = false;
  // Debug/per-tool gates: one PARTITION_NONE superblock and DC intra only.
  bool force_single_block = false;
  bool force_dc = false;
  // Full directional/smooth intra is bit-exact with the decoder. It stays
  // opt-in because the mode search is SSE-only (no rate term yet), which does
  // not reliably beat DC at fixed QP.
  bool luma_dc_only = true;
  // Enable the full chroma intra search (directional/smooth/Paeth) instead of
  // DC/CfL only. Off by default until forced-UV decoder verification passes on
  // the M1 still corpus. CLI: --still-uv-full (to be wired by the orchestrator).
  bool uv_full = false;
  // Rate-aware luma intra mode decision (`src/encoder/rdo/intra_decision.*`).
  // Off by default: the lambda scale is a documented calibration placeholder.
  bool rdo_intra = false;
  double rdo_sse_scale = 0.125;
  // Full-trellis coefficient RDOQ (`src/encoder/rdoq/`).  Same syntax, only the
  // quantised levels change.  Off by default; sse_scale is a calibration
  // placeholder (see the RDOQ README's lambda table).
  bool rdoq = false;
  double rdoq_sse_scale = 0.125;
  // Intra luma TX_TYPE search (`src/encoder/tx_type/`).  Chroma stays DCT_DCT
  // (enable_chroma_dctonly).  Off by default.
  bool tx_type = false;
  double tx_type_sse_scale = 0.125;
  // Opt-in CDEF (§7.18).  The strength search runs after deblock and its
  // filtered planes replace the reconstruction/output.  Default off so the
  // pinned `--still` MD5s do not change.
  bool enable_cdef = false;
  // CdefStrengths: 1 forces one global set (no index symbols); 2..8 signal a
  // per-64x64 index.  4 is the search default ladder size.
  int cdef_num_strengths = 4;
  int cdef_damping = 3;
  bool cdef_on_skip_txfm = false;
  // Opt-in CCSO band-offset-only (§7.19, post-CDEF).  `ccso_bands_log2` sets
  // ccso_max_band_log2 (0..6; 3 = 8 bands).
  bool enable_ccso = false;
  int ccso_bands_log2 = 6;
  // §5.4.3 superblock selection.  Both are opt-in and mutually exclusive; an
  // intra frame uses a 128x128 frame-superblock grid for either flag (§5.18.1),
  // while sb256 signals the 256x256 sequence flag.  Leaves stay <= 64x64.
  bool sb128 = false;
  bool sb256 = false;
  // §5.4.3 extended partitions (HORZ_3/VERT_3 and uneven 4-way).  Opt-in; the
  // search policy only enables families whose children keep their own chroma
  // block.  CPU-only: the CUDA planner has no extended tree.
  bool ext_partitions = false;
  // §5.4.3 enable_sdp: per-64x64 intra luma/chroma partition trees.  Opt-in and
  // CPU-only; the first gated configuration forces PARTITION_NONE on both
  // trees.  Default off keeps every pinned stream unchanged.
  bool sdp = false;
  // Opt-in neutral delta-Q / segmentation syntax (§5.18.7.1/.8).  The signalled
  // segment table has no enabled features and every superblock sends a zero
  // reduced delta, so the reconstruction is unchanged.
  bool enable_segmentation = false;
  bool enable_delta_q = false;
  int delta_q_res = 0;
  // Opt-in real per-superblock segmentation with SEG_LVL_ALT_Q: a variance
  // planner assigns each SB to segment 0 (base qindex) or segment 1
  // (base + per_segment_q_delta).  Forward quantisation, dequantisation and
  // the deblock qindex all use the resulting qindex.  Default off keeps the
  // neutral table and the pinned stream.
  bool per_segment_q = false;
  int per_segment_q_delta = 8;
  // Filled by encode_still_picture when per_segment_q is on; one entry per
  // superblock in raster order.  Not a CLI input.
  std::vector<std::uint8_t> sb_segment;
  // Opt-in content interpretation OBU (§5.15); default parameters.
  bool content_interpretation = false;
  // Opt-in quantizer matrices (§5.18.6.2 + §7.14.4).  Signals a spec-defined
  // default matrix level (0..14) in the frame header and applies the matching
  // q2 = Round2(q * m, 5) in forward quantisation and reconstruction.  Off by
  // default so the pinned `--still` MD5s do not change.  Level 15 is the
  // "no QM" sentinel and is not a selectable set.
  bool qm = false;
  int qm_level = 3;
  // Opt-in user-defined quantizer matrices (§5.4.11 user_defined_qm() +
  // §5.13 quantizer_matrix_obu()).  Signals a real QM OBU with an identity
  // (all-32) user matrix for level 3, so the §7.14.4 q2 equals q.  Implies
  // `qm`.  The pinned dav2d rejects every QM OBU, so the product gate records
  // that rejection; the dequant path itself is the component's tested one.
  bool user_qm = false;
  // Opt-in intra secondary transform (§5.20.8.2, §7.15.3).  Luma-only: the
  // encoder projects the quantised primary coefficients, signals
  // sec_tx_type/most_probable_stx_set, and reconstruction applies the
  // normative inverse before the primary 2-D inverse.  Off by default.
  bool ist = false;
  double ist_sse_scale = 0.125;
  // Opt-in identity intra transform (§5.4.8 enable_idtx_intra, §5.20.8.2).
  // Adds IDTX to the intra luma type candidates (only the sets that can signal
  // it) and codes the §5.20.5.5 fsc_mode = 0 symbol for luma blocks <= 32x32.
  // IDTX is TX_CLASS_2D, so the diagonal scan and 2-D contexts apply; the
  // one-axis identity types (V/H_DCT, ...) stay disabled.  Off by default.
  bool idtx = false;
  // DDT/DDTX/FDDT is inter-only: §5.4.8 omits enable_inter_ddt for a
  // single-picture header and §7.15.4.1 applies the substitution only when
  // is_inter.  Kept as an explicit rejected flag so the CLI is honest.
  bool ddt = false;
  // Opt-in trellis-coded quantisation (§5.4.8 enable_tcq, §5.20.7.27,
  // §7.14.4).  Luma 2-D transform classes only: the coefficient syntax
  // carries the TCQ base level and escape, and §7.14.4 doubles dqDenom for
  // those blocks.  A single-picture header infers choose_tcq_per_frame = 0,
  // so enabling TCQ also forces enable_parity_hiding = 0 (the bit is then
  // absent from the sequence header).  Off by default.
  bool tcq = false;
  double tcq_sse_scale = 0.125;
  // Opt-in cross-component transform (§5.4.8 enable_cctx, §5.20.7.27
  // cctx_type, §7.14.6 joint dequant).  Chroma only: the encoder searches the
  // seven rotation types per chroma block and reconstruction applies the §7.14.6
  // rotation after the per-plane dequant.  Off by default.
  bool cctx = false;
  // Opt-in transform partition signalling (§5.20.6.1-.3 TX_MODE_SELECT).
  // The encoder tries NONE/SPLIT/HORZ/VERT per luma coding block, writes
  // tx_do_partition/tx_partition_type before the coefficients, and runs
  // prediction + reconstruction per transform part.  Chroma is unaffected.
  bool tx_partition = false;
  // Opt-in DC/AC delta-Q (§5.18.6.1).  The encoder runs the component's frame
  // delta search and signals the result; forward quantisation and §7.14.4
  // dequant use the per-plane DC/AC quantizers.  Off by default.
  bool delta_q = false;
  // Opt-in edge-classified CCSO (§7.19.1) with per-unit ccso_blk decisions.
  // The default --still-ccso path stays band-offset-only and all-units-on.
  bool ccso_edge = false;
  // Opt-in non-separable Wiener loop restoration (§7.20.3), applied after
  // CDEF/CCSO.  `enable_wiener` covers luma, `enable_chroma_lr` the shared
  // chroma planes; both use per-unit filters at LoopRestorationSize 64 luma /
  // 32 chroma.  Off by default.
  bool enable_wiener = false;
  bool enable_chroma_lr = false;
  // Opt-in luma PC-Wiener (§7.20.4).  Mutually exclusive with
  // `enable_wiener` (both signal luma loop restoration).
  bool enable_pc_wiener = false;
  // Opt-in GDF (§7.20.5), applied in the loop-restoration stage after CCSO.
  // M1 signals gdf_per_block = 0 (every unit filtered), so no tile symbols.
  bool enable_gdf = false;
  // Opt-in film grain signalling (§5.14/§5.18.10).  Grain is decoder-side
  // output only; the encoder reconstruction and reference stay grain-free.
  bool enable_grain = false;
  // Estimate the grain model with the GrainStreamPlanner on the encoder
  // reconstruction instead of signalling the fixed two-point model.
  bool grain_adaptive = false;
  // §8.2.2 tile CDF seed (packed CDF image from syntax::cdf_image_words()).
  // Empty selects the §9.3 defaults, which is the historical path.  The
  // multi-tile --avg-cdf writer fills this from the average-CDF bank.
  std::vector<std::uint16_t> cdf_seed;
};

struct StillResult {
  std::uint32_t tile_bytes = 0;
  std::uint32_t frame_bytes = 0;
  std::vector<std::uint8_t> frame;
  std::vector<std::uint8_t> recon_y;
  std::vector<std::uint8_t> recon_u;
  std::vector<std::uint8_t> recon_v;
  int leaves = 0;
  // Exact-copy intra-block-copy leaves selected by `--still-ibc`.
  int ibc_blocks = 0;
  // Final §8.2 tile CDFs (packed image) for the average-CDF bank.
  std::vector<std::uint16_t> cdf_final;
  // Exact header/filter configuration the CPU reference emitted.  The GPU
  // orchestrator reuses it for the filter stages so the sequence/frame header
  // bits cannot diverge from the CPU reference (one source of truth for the
  // §5.18.7 restoration parameters).  Fields are default-off.
  StillConfig config;
  // Per-64x64 §5.20.10.1 cdef_idx grid, row-major on the luma 64x64 grid.
  // Empty unless CDEF ran.  The GPU stage uploads it and the syntax bridge
  // attaches the value to the first leaf of each 64x64 block.
  std::vector<std::int8_t> cdef_idx;
  int cdef_n64_cols = 0;
  int cdef_n64_rows = 0;
  // Per-plane §5.20.10.2 ccso_blk grids (one byte per CCSO unit).  Empty when
  // the plane's CCSO is off.  The GPU stage uploads them for §7.19 and the
  // syntax bridge replays them as read_ccso() symbols.
  std::vector<std::uint8_t> ccso_blks[3];
  int ccso_blk_cols[3] = {0, 0, 0};
  int ccso_blk_rows[3] = {0, 0, 0};
};

bool encode_still_picture(const StillPicture& picture, const StillOptions& options,
                             StillResult& result, std::string& error);

}  // namespace av2enc::still

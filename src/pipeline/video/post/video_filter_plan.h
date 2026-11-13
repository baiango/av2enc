// Per-frame loop-filter plan for the M2 video path.
//
// This is the syntax/decision description shared by the frame-header writer
// (`pipeline/video/video_headers.*`), the tile writer
// (`pipeline/video/video_syntax.cpp`) and the pixel stages
// (`pipeline/video/post/video_loop_filters.*`).  It deliberately carries no
// pixel buffers: the same plan drives the CPU filter cores and the CUDA
// kernels, which is what keeps the two streams byte-identical.
//
// Normative order (AV2 v1.0.0 §7.17–§7.20, docs/negative-results.md):
//
//   reconstruct -> deblock -> CDEF -> CCSO -> LR -> GDF -> output/reference
//
// Every tool defaults to disabled, so a default `--video` stream keeps its
// bytes (the header rows are only written when the matching sequence flag is
// set).
#ifndef AV2ENC_PIPELINE_VIDEO_POST_VIDEO_FILTER_PLAN_H_
#define AV2ENC_PIPELINE_VIDEO_POST_VIDEO_FILTER_PLAN_H_

#include <cstdint>
#include <vector>

namespace av2enc::video::post {

// §5.18.7.10 cdef_params() plus the per-64x64 `cdef_idx` grid.
//
// `num_strengths == 1` codes no §5.20.10.1 tile symbol at all (§5.20.10.1
// forces index 0); with 2..8 every 64x64 block codes an index, and index 0
// must be the all-zero strength set so a block can be left unfiltered.
struct VideoCdefPlan {
  bool enabled = false;
  int damping = 3;         // CdefDamping 3..6
  int num_strengths = 1;   // CdefStrengths 1..8
  bool on_skip_txfm = false;
  // Decoded-domain strengths (pri 0..15, sec in {0,1,2,4}).
  struct Strength {
    int y_pri = 0;
    int y_sec = 0;
    int uv_pri = 0;
    int uv_sec = 0;
  };
  Strength strength[8] = {};
  int n64_cols = 0;
  int n64_rows = 0;
  // One entry per 64x64 luma block, raster order, in [0, num_strengths).
  std::vector<std::int8_t> index;
  // §7.18.1 MI metadata.  The M2 baseline codes no skip/lossless segments, so
  // these stay all-zero and are shared by the CPU and CUDA stages.
  std::vector<std::uint8_t> skip;
  std::vector<std::uint8_t> lossless;
  std::vector<std::uint8_t> lossless_chroma;
};

// §5.18.7.12 ccso_params() plus the per-unit `CcsoBlks[]` grids.
struct VideoCcsoPlane {
  bool enabled = false;
  bool bo_only = true;
  int scale_idx = 0;
  int quant_idx = 0;
  int ext_filter = 0;
  int edge_clf = 0;
  int max_band_log2 = 0;
  // ccso_offset_idx values in syntax order (d0, d1, band), each tu(7).
  std::vector<std::uint8_t> offset_idx;
  // CcsoBlks[plane]: one byte per CCSO unit, row-major.
  std::vector<std::uint8_t> blks;
  int blk_cols = 0;
  int blk_rows = 0;
};

struct VideoCcsoPlan {
  // ccso_frame_flag.  False codes no header row and no tile symbol.
  bool enabled = false;
  int ccso_luma_size_log2 = 6;
  VideoCcsoPlane plane[3];
};

// One plane's §7.20 loop restoration (non-separable Wiener) decision.
struct VideoLrPlane {
  // FrameRestorationType[plane] != RESTORE_NONE.
  bool enabled = false;
  // §6.17.7.7 RestorationType value (0 none, 1 PC-Wiener, 2 Wiener NS).
  std::uint8_t restore_type = 0;
  // LoopRestorationSize[plane] in plane samples, and the §5.18.7.11 shift that
  // signals it.
  int unit_size = 64;
  int shift = 3;
  int unit_rows = 0;
  int unit_cols = 0;
  // LrType[plane][unit_row][unit_col] (RESTORE_NONE / RESTORE_WIENER_NONSEP).
  std::vector<std::uint8_t> unit_type;
  // LrWienerNs[plane][unit][coeff], stride kWienerNsCoeffStride.
  std::vector<std::int16_t> unit_coeffs;
};

struct VideoLrPlan {
  bool enabled = false;
  // FrameRestorationType[0] == RESTORE_PC_WIENER (--video-pc-wiener).
  bool pc_wiener = false;
  // §7.20.1 LrTxSkip on the luma MI grid (mi_cols * mi_rows bytes, row-major),
  // exactly the map the §7.20.4 classifier reads.  Empty for a non-PC plan.
  std::vector<std::uint8_t> tx_skip;
  int base_q_idx = 90;
  int tx_skip_cols = 0;
  int tx_skip_rows = 0;
  int luma_shift = 3;
  int chroma_shift = 3;
  VideoLrPlane plane[3];
};

// §5.18.7.9 GDF frame parameters.  The wired mode is header-only
// (gdf_per_block = 0), so no §5.20.10.3 use_gdf symbol is coded.
struct VideoGdfPlan {
  bool enabled = false;
  int qc_idx = 0;
  int scale_idx = 0;
  bool per_block = false;
  // §7.20.1 derivation inputs, so a device stage can rebuild the same indices.
  bool frame_is_intra = true;
  int base_q_idx = 90;
  int num_total_refs = 0;
  int order_hints[2] = {0, 0};
  int frame_distance[2] = {0, 0};
  int restricted_oh = -1;
};

struct VideoFilterPlan {
  VideoCdefPlan cdef;
  VideoCcsoPlan ccso;
  VideoLrPlan lr;
  VideoGdfPlan gdf;

  bool any() const {
    return cdef.enabled || ccso.enabled || lr.enabled || gdf.enabled;
  }
};

}  // namespace av2enc::video::post

#endif  // AV2ENC_PIPELINE_VIDEO_POST_VIDEO_FILTER_PLAN_H_

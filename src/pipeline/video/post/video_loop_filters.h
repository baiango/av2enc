// M2 video loop-filter stage: per-frame decision + pixel application.
//
// The stage runs on the reconstructed frame *after* it is complete and *before*
// it becomes the output picture and the reference for later frames, so the
// encoder and the decoder filter the same samples in the same order.
//
// This file owns only the M2 glue: the decisions and pixel work are delegated
// to the verified components:
//
//   CDEF  §7.18  src/codec/filters/cdef/  (search + filter core) via the
//                src/codec/filters/cdef/syntax/video_cdef.* M2 adapter
//   CCSO  §7.19  src/codec/filters/loop_restoration/
//   LR    §7.20  src/codec/filters/loop_restoration/ (+ pc_wiener/, gdf/)
//
// `run_video_loop_filters()` drives the tool chain in normative order.  With no
// hooks it applies every stage with the CPU component cores; the CUDA pipeline
// passes `VideoLoopFilterHooks` that runs the matching device kernel instead.
// The decisions are shared, which is what keeps the two streams byte-identical.
//
// Scope: 8-bit 4:2:0, dimensions divisible by 8 (the M2 core requires that),
// single tile.  All tools default off.
#ifndef AV2ENC_PIPELINE_VIDEO_POST_VIDEO_LOOP_FILTERS_H_
#define AV2ENC_PIPELINE_VIDEO_POST_VIDEO_LOOP_FILTERS_H_

#include <cstdint>
#include <span>
#include <string>

#include "codec/filters/loop_restoration/restoration_common.h"
#include "pipeline/video/post/video_filter_plan.h"

namespace av2enc::video::post {

// CLI/encoder options for the M2 loop-filter chain.
struct VideoLoopFilterOptions {
  // §5.4.5 cfl_ds_filter_index used by the chroma-from-luma restoration taps
  // (0 default; 1/2 select the other filters, 3 aliases 0).
  int cfl_ds_filter_index = 0;
  bool cdef = false;
  // CdefDamping = cdef_damping_minus_3 + 3.
  int cdef_damping = 3;
  // CdefStrengths.  1 is the header-only mode (no tile symbol); 2..8 codes a
  // per-64x64 `cdef_idx` and includes the all-zero set at index 0.
  int cdef_num_strengths = 1;
  // CdefOnSkipTxfm: 0 DISABLED, 1 ALWAYS_ON, 2 ADAPTIVE.  The M2 baseline
  // codes no skip flag on intra leaves and disables the tool on skipped inter
  // leaves, so DISABLED is the only mode wired here.
  int cdef_on_skip_txfm_mode = 0;

  bool ccso = false;
  // CcsoMaxBandLog2 for the band-offset-only form (0..6).
  int ccso_bands_log2 = 4;

  // §7.20 luma non-separable Wiener and/or chroma loop restoration.
  bool wiener = false;
  bool chroma_lr = false;
  // §7.20.4 luma PC-Wiener (RESTORE_PC_WIENER).  Replaces luma Wiener NS;
  // composes with chroma LR.
  bool pc_wiener = false;

  // §7.20.5 guided detail filter (luma, header-only mode).
  bool gdf = false;

  bool any() const {
    return cdef || ccso || wiener || chroma_lr || pc_wiener || gdf;
  }
};

// One coded leaf, used to build the §7.18.1 Skips[] grid.
struct VideoLoopFilterBlock {
  int mi_row = 0;
  int mi_col = 0;
  int w4 = 1;
  int h4 = 1;
  bool skip = false;
  // Luma eob == 0 (no coefficients).  §7.20.1 LrTxSkip = skip || eob == 0.
  bool luma_eob_zero = false;
};

// MI-aligned reconstruction plus the visible source picture (SSE metric only).
// `source_stride` may be the visible width: the M2 core requires dimensions
// divisible by 8, so the MI-aligned plane equals the visible plane.
struct VideoLoopFilterFrame {
  int width = 0;
  int height = 0;
  int mi_cols = 0;
  int mi_rows = 0;
  std::uint8_t* plane[3] = {nullptr, nullptr, nullptr};
  int stride[3] = {0, 0, 0};
  const std::uint8_t* source[3] = {nullptr, nullptr, nullptr};
  int source_stride[3] = {0, 0, 0};
  bool frame_is_intra = true;
  // Frame base_q_idx (the §7.20 LR search asks for it).
  int base_q_idx = 90;
  // §7.20.1 reference state for the GDF table derivation: NumTotalRefs plus
  // the first Min(NumTotalRefs, 2) reference OrderHints and frame distances.
  int num_total_refs = 0;
  int ref_order_hints[2] = {0, 0};
  int frame_distance[2] = {0, 0};
  // §7.20.1 RESTRICTED_OH: -1 when no reference is restricted (the M2 chain
  // codes no restricted references).
  int restricted_oh = -1;
  // Coded leaves (preorder).  Used to derive §7.18.1 Skips[]: the decoder
  // leaves a fully-skipped 8x8 block unfiltered when CdefOnSkipTxfm is
  // DISABLED, so the encoder must pass the same grid or the reference frames
  // diverge on the next P frame.
  std::span<const VideoLoopFilterBlock> blocks;
  // Optional pre-built Skips[] grid (luma MI, row-major); used when `blocks`
  // is empty.  Null means "no skipped blocks" (the intra case).
  const std::uint8_t* skip = nullptr;
  // Post-deblock / pre-CDEF planes (CurrFrame in §7.19/§7.20).  Filled by
  // run_video_loop_filters() when a later stage needs them; callers leave this
  // null.
  const std::uint8_t* curr[3] = {nullptr, nullptr, nullptr};
  int curr_stride[3] = {0, 0, 0};
  // Post-CCSO / pre-LR planes (CdefFrame in §7.20).  Only GDF reads them.
  const std::uint8_t* cdef[3] = {nullptr, nullptr, nullptr};
  int cdef_stride[3] = {0, 0, 0};
};

enum class VideoFilterStage : std::uint8_t {
  kCdef = 0,
  kCcso = 1,
  kLr = 2,
  kGdf = 3,
};

const char* video_filter_stage_name(VideoFilterStage stage) noexcept;

// Plane bindings handed to a device stage.  `in` is the stage's unfiltered
// input; `out` receives the stage result (same strides); `curr` is the
// post-deblock frame for the stages that sample it (§7.19/§7.20).
struct VideoLoopFilterDeviceFrame {
  const std::uint8_t* in[3] = {nullptr, nullptr, nullptr};
  int in_stride[3] = {0, 0, 0};
  const std::uint8_t* curr[3] = {nullptr, nullptr, nullptr};
  int curr_stride[3] = {0, 0, 0};
  const std::uint8_t* cdef[3] = {nullptr, nullptr, nullptr};
  int cdef_stride[3] = {0, 0, 0};
  std::uint8_t* out[3] = {nullptr, nullptr, nullptr};
  int out_stride[3] = {0, 0, 0};
};

// Device pixel hook.  The CUDA pipeline implements it with the verified filter
// kernels; `plan` already contains the stage's decision.
class VideoLoopFilterHooks {
 public:
  virtual ~VideoLoopFilterHooks() = default;
  virtual bool apply_stage(VideoFilterStage stage,
                           const VideoFilterPlan& plan,
                           const VideoLoopFilterDeviceFrame& frame,
                           std::string* error) = 0;
};

// Runs the configured tools in normative order (CDEF -> CCSO -> LR -> GDF) on
// `frame.plane`.  Fills `plan` with everything the frame header and the tile
// writer need.  `hooks` may be null (CPU component cores); when set, every
// stage's pixels come from the hook instead.
bool run_video_loop_filters(VideoLoopFilterFrame& frame,
                            const VideoLoopFilterOptions& options,
                            VideoFilterPlan* plan, VideoLoopFilterHooks* hooks,
                            std::string* error);

// Applies one stage's pixels with the CPU component cores against the plan's
// decision.  `frame.plane` is filtered in place.  Used by
// run_video_loop_filters() and by the CPU-emulation backend.
bool apply_video_filter_stage(VideoFilterStage stage,
                              VideoLoopFilterFrame& frame,
                              const VideoFilterPlan& plan,
                              std::string* error);

// §7.20 loop-restoration parameters derived from a plan.  `unit_type` /
// `unit_coeffs` point into the plan.
struct LrFilterOptions {
  int frame_type[3] = {0, 0, 0};
  int unit_size[3] = {64, 64, 64};
  int unit_rows[3] = {0, 0, 0};
  int unit_cols[3] = {0, 0, 0};
  const std::uint8_t* unit_type[3] = {nullptr, nullptr, nullptr};
  const std::int16_t* unit_coeffs[3] = {nullptr, nullptr, nullptr};
};
LrFilterOptions lr_filter_options(const VideoLrPlan& plan);

// §7.18 filter parameters derived from a plan.  Exposed so the CUDA backend
// feeds the device kernel exactly the same options the CPU stage uses.
struct CdefFilterOptions {
  int damping = 3;
  int num_strengths = 1;
  bool on_skip_txfm = false;
  struct Strength {
    int y_pri = 0;
    int y_sec = 0;
    int uv_pri = 0;
    int uv_sec = 0;
  };
  Strength strength[8] = {};
};
CdefFilterOptions cdef_filter_options(const VideoCdefPlan& plan);

// §7.19 filter parameters (computed offsets) derived from a plan.  `offsets`
// is the flattened CcsoFilterOffset table the core reads.
struct CcsoFilterOptions {
  struct Plane {
    bool enabled = false;
    bool bo_only = true;
    int scale_idx = 0;
    int quant_idx = 0;
    int ext_filter = 0;
    int edge_clf = 0;
    int max_band_log2 = 0;
    std::int16_t offsets[::av2enc::restoration::kCcsoOffsetEntries] = {};
  };
  Plane plane[3];
  int ccso_luma_size_log2 = 6;
};
CcsoFilterOptions ccso_filter_options(const VideoCcsoPlan& plan);

}  // namespace av2enc::video::post

#endif  // AV2ENC_PIPELINE_VIDEO_POST_VIDEO_LOOP_FILTERS_H_

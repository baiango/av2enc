// Internal device-backend contract for the CUDA M2 P-frame pipeline.
//
// The orchestration in video_pipeline.cpp is the only consumer.  Two
// implementations exist:
//
//   video_backend_cpu.cpp   host emulation; reuses the CPU component references
//   video_backend_cuda.cu   real CUDA backend; reuses src/cuda/inter/{me,mc},
//                        src/cuda/intra_modes and src/cuda/tx_variants
//
// The orchestration owns every decision (partition tree, MV stack, candidate
// comparison, syntax).  The backend only performs pixel/transform work, which
// keeps the CUDA and emulation paths decision-identical by construction.
#ifndef AV2ENC_CUDA_VIDEO_M2_BACKEND_H_
#define AV2ENC_CUDA_VIDEO_M2_BACKEND_H_

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "codec/inter/me/me_core.h"
#include "codec/inter/gme/gme.h"
#include "pipeline/video/post/deblock_grid.h"
#include "pipeline/video/video_frame.h"
#include "pipeline/video/post/video_filter_plan.h"
#include "pipeline/video/video_intra_search.h"
#include "video_gpu.h"

namespace av2enc::cuda {
class Arena;
}

namespace av2enc::cuda::video::detail {

// Device-allocation outcome for the arena-backed backend.  The distinct
// kArenaTooSmall code is what lets a caller grow the arena at a frame
// boundary; `required_bytes` is the capacity to re-init with.
enum class ArenaStatus : int {
  kOk = 0,
  kArenaTooSmall,
  kInvalidArgument,
  kCudaError,
};

struct ArenaAllocation {
  ArenaStatus status = ArenaStatus::kOk;
  std::uint64_t required_bytes = 0;

  bool ok() const { return status == ArenaStatus::kOk; }
};

// One partition-node cost request: visible rectangle in luma samples.
struct VarianceNode {
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
};

// One block in a plane (luma or chroma) and a 1/8-luma-sample MV.  `ref_index`
// selects the entry in the frame's explicit reference list (0 = most recent,
// mirroring `ref_frame_idx[]` in the frame header).
struct InterJob {
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
  int mv_y = 0;
  int mv_x = 0;
  int ref_index = 0;
  // §5.20.7.16 compound blend parameters.  The compound path always predicts
  // both references at `mv_y`/`mv_x` (zero in the gated subset) and blends
  // with this type.  0 = WEDGE, 1 = DIFFWTD, 2 = AVERAGE (default).
  int compound_type = 2;
  int bsize = 0;  // partition::BlockSize, selects the wedge mask geometry
  int wedge_index = 0;
  int wedge_sign = 0;
  int mask_type = 0;
};

// One ME job: luma origin and block size in the current (source) frame.
struct MeRequest {
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
  int ref_index = 0;
};

// ME output in 1/16-luma-sample units, matching me::MeCoreResult.
struct MeOutcome {
  std::int32_t mv_y = 0;
  std::int32_t mv_x = 0;
};

// One GLOBALMV warped-prediction job.  The model is the frame's §5.18.9
// parameter for `ref_index`; the block origin is in frame samples.
struct WarpJob {
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
  int mi_row = 0;
  int mi_col = 0;
  int bw4 = 1;
  int bh4 = 1;
  int ref_index = 0;
  inter::gme::GmParams params{};
};

// Transform result for one plane: quantised levels at the adjusted size in
// natural (not scan) order, and the reconstructed samples at coded size.
struct PlaneTx {
  std::vector<std::int32_t> levels;
  std::vector<std::uint8_t> recon;
};

// One winning transform block of the opt-in luma TX search (--video-tx-search).
// `levels` are in §5.20.7.30 scan order, exactly as the syntax writer reads
// them; `x`/`y` and `mi_row`/`mi_col` are relative to the coding block.
struct TxSearchPart {
  int x = 0;
  int y = 0;
  int coded_w = 0;
  int coded_h = 0;
  int tx_size = 0;
  int tx_type = 0;
  int mi_row = 0;
  int mi_col = 0;
  int eob = 0;
  std::vector<std::int32_t> levels;
};

// Winning luma plan for one residual block.  `recon` is the coded-size
// prediction + residual reconstruction; `rate_bits` is the module's rate the
// orchestration uses in place of the estimate_tx_bits() heuristic.
struct TxSearchPlan {
  int partition = 0;  // av2enc::video::txpartition::TxPartition
  std::vector<TxSearchPart> parts;
  std::vector<std::uint8_t> recon;
  double rate_bits = 0.0;
};

class Backend {
 public:
  virtual ~Backend() = default;

  virtual const char* name() const = 0;

  // Uploads the source planes, binds the explicit reference list (empty for a
  // keyframe, otherwise frame_header ref_frame_idx order), zeroes the
  // reconstructed planes and starts frame timing.  `frame_mi_cols/rows` describe
  // the whole frame (persistent allocation and references); the tile origin and
  // MI size define the coded geometry.  A whole-frame encode passes the frame
  // MI size with tile_x = tile_y = 0.  `base_q_idx` selects the §7.14.2 frame
  // quantizer (qlookup) for the transform chain.
  virtual bool begin_frame(
      const av2enc::video::VideoInputFrame& source, int frame_mi_cols,
      int frame_mi_rows, int tile_x, int tile_y, int mi_cols, int mi_rows,
      std::span<const av2enc::video::VideoReconFrame* const> references,
      int base_q_idx, std::string& error) = 0;

  // Ends frame timing.  The orchestration already holds the host recon.
  virtual bool end_frame(std::string& error) = 0;

  // Partition block-variance costs for every requested visible rectangle.
  virtual bool variance_costs(const std::vector<VarianceNode>& nodes,
                              std::vector<std::int64_t>& costs,
                              std::string& error) = 0;

  // Batched motion estimation against reference slot 0.
  virtual bool me_batch(const std::vector<MeRequest>& jobs,
                        const me::MeCoreOptions& options,
                        std::vector<MeOutcome>& results,
                        std::string& error) = 0;

  // Luma MC + SSE against the current source frame, one per job.
  virtual bool inter_sse_batch(const std::vector<InterJob>& jobs,
                               std::vector<std::int64_t>& sse,
                               std::string& error) = 0;

  // Chosen inter prediction: luma + 4:2:0 chroma.  Writes the prediction into
  // the device reconstructed planes and returns host copies.
  virtual bool mc_420(const InterJob& luma_job, std::vector<std::uint8_t>& y,
                      std::vector<std::uint8_t>& u, std::vector<std::uint8_t>& v,
                      std::string& error) = 0;

  // Compound skip-mode prediction: both references at the same MV, CWP_EQUAL
  // average, no residual.  `compound_sse_batch` scores luma against the source
  // without touching the reconstructed planes; `compound_420` writes the
  // chosen block's prediction into them and returns host copies.
  virtual bool compound_sse_batch(const std::vector<InterJob>& jobs,
                                  std::vector<std::int64_t>& sse,
                                  std::string& error) = 0;
  virtual bool compound_420(const InterJob& luma_job,
                            std::vector<std::uint8_t>& y,
                            std::vector<std::uint8_t>& u,
                            std::vector<std::uint8_t>& v,
                            std::string& error) = 0;

  // §7.13.3.19/.20 GLOBALMV warped prediction.  `warp_sse_batch` scores luma
  // against the current source frame; `warp_420` writes the chosen block's
  // prediction into the device reconstructed planes and returns host copies.
  virtual bool warp_sse_batch(const std::vector<WarpJob>& jobs,
                              std::vector<std::int64_t>& sse,
                              std::string& error) = 0;
  virtual bool warp_420(const WarpJob& luma_job, std::vector<std::uint8_t>& y,
                        std::vector<std::uint8_t>& u,
                        std::vector<std::uint8_t>& v, std::string& error) = 0;

  // §5.18.9 global model fit.  The CPU-emulation backend calls the CPU
  // reference; the CUDA backend runs the device estimator on the same block
  // correspondences, so both paths produce the same quantised model.
  virtual bool gme_estimate(
      const std::vector<inter::gme::GmBlockMv>& blocks,
      const inter::gme::GmOptions& options,
      inter::gme::GmEstimate& estimate, std::string& error) = 0;

  // DC intra candidate for all three planes: prediction is taken from the
  // current device reconstructed planes, the residual is transformed and
  // reconstructed, but nothing is written back to the recon planes.
  virtual bool intra_candidate(int x0, int y0, int w, int h,
                               std::array<PlaneTx, 3>& planes,
                               std::string& error) = 0;

  // Full intra candidate (--video-intra-full): the still product's
  // 61-candidate luma search (DC, V, H, eight directional modes with the six
  // non-zero angle deltas, SMOOTH, SMOOTH_V/H, PAETH) plus DC chroma.  The
  // winning luma mode is returned through `y_mode`/`y_angle_delta` (AV2 YMode
  // values); `above_smooth`/`left_smooth` carry the §7.13.2.15/.16 neighbouring
  // smooth-mode flags computed by the orchestrator from its IntraJointMode
  // grid.  `luma_avail`/`chroma_avail` are the §5.20.9.1 availability counts
  // the orchestrator derives from its §6.19.2.3 BlockDecoded map (the backend
  // cannot see that map).  Nothing is written back to the recon planes.
  virtual bool intra_full_candidate(
      int x0, int y0, int w, int h,
      const av2enc::video::intra_search::Availability& luma_avail,
      const av2enc::video::intra_search::Availability& chroma_avail,
      bool above_smooth, bool left_smooth, std::array<PlaneTx, 3>& planes,
      int& y_mode, int& y_angle_delta, std::string& error) = 0;

  // Writes an already-decided intra reconstruction into the device recon
  // planes (luma + chroma).
  virtual bool write_recon_420(int x0, int y0, int w, int h,
                               const std::vector<std::uint8_t>& y,
                               const std::vector<std::uint8_t>& u,
                               const std::vector<std::uint8_t>& v,
                               std::string& error) = 0;

  // Luma-only reconstruction write.  §7.13.5 CfL reads the reconstructed luma
  // of the current block, so the intra candidate must publish luma before the
  // chroma decision.
  virtual bool write_recon_luma(int x0, int y0, int w, int h,
                                const std::vector<std::uint8_t>& y,
                                std::string& error) = 0;

  // §7.13.5 CfL chroma decision (--video-cfl): the M1 still product's
  // {DC, CfL} candidate set with derived alpha.  The device backend runs
  // `still::chroma_decide()` (bit-exact to the CPU policy) and the CPU
  // emulation runs the shared `video::chroma_search::choose()`.  The luma
  // reconstruction of the current block must already be written.  Returns the
  // winner's U/V transform plan and `is_cfl`/`cfl_index` for the syntax writer.
  // `num4_above_right`/`num4_below_left` are the §7.13.6 MHCCP neighbour
  // availability counts at chroma 4x4 granularity (the orchestrator owns the
  // §6.19.2.3 BlockDecoded map).  `enable_mhccp` also scores CFL_MULTI.
  virtual bool chroma_cfl_candidate(int x0, int y0, int w, int h,
                                    int num4_above_right,
                                    int num4_below_left, bool enable_mhccp,
                                    std::array<PlaneTx, 2>& planes, int& is_cfl,
                                    int& cfl_index, int& cfl_mh_dir,
                                    std::string& error) = 0;

  // Inter residual chain (MC -> residual -> TX/quant/inverse -> recon) for one
  // plane.  Not selected by the M2 baseline decisions; used by the selftest.
  virtual bool inter_residual_candidate(const InterJob& job,
                                        PlaneTx& out, std::string& error) = 0;

  // Inter residual chain for luma + both chroma planes at one luma MV.  The
  // reconstruction is returned but not written to the recon planes; the
  // orchestration writes it only if the residual decision wins.
  virtual bool inter_residual_420(const InterJob& luma_job,
                                  std::array<PlaneTx, 3>& planes,
                                  std::string& error) = 0;

  // Opt-in luma TX size/type/partition search (--video-tx-search).  Evaluates
  // the legal §5.20.6/§5.20.8 candidates for one luma residual block and returns
  // the RD-best plan with scan-order levels and the coded-size reconstruction.
  // The CUDA backend runs the device module under
  // `src/cuda/pipeline/video/tx/`; the CPU-emulation backend runs the shared
  // CPU reference so both paths produce the same plan.
  //
  // `inter` selects the §5.20.8.3 inter vs intra type sets; `job` carries the
  // MC predictor for inter (ignored for intra, which predicts from the current
  // reconstructed plane).  `partition` enables the §5.20.6.3 partition search:
  // it is inter-only because a fixed block predictor is only equivalent to the
  // decoder's per-transform-block prediction when TX covers the coding block.
  virtual bool tx_search_luma(bool inter, const InterJob& job, int x0, int y0,
                              int w, int h, int qindex, double lambda,
                              bool partition, TxSearchPlan& out,
                              std::string& error) = 0;

  // §7.17 deblocking of the finished reconstruction.  `grid` carries the
  // per-plane metadata derived from the real leaf decisions.  The backend
  // filters its device reconstruction (CUDA kernel on the product path) and
  // returns the filtered MI-aligned planes in y/u/v; the CPU-emulation backend
  // runs the shared CPU reference over the supplied planes.  There is no
  // CPU-pixel fallback on the CUDA backend.
  virtual bool deblock_frame(const av2enc::video::post::VideoDeblockGrid& grid,
                             std::vector<std::uint8_t>& y,
                             std::vector<std::uint8_t>& u,
                             std::vector<std::uint8_t>& v,
                             std::string& error) = 0;
  // Opt-in M2 loop filters (--video-cdef, ...).  Applies one stage's planned
  // filter to the device planes with the CUDA kernel and writes the result to
  // `frame.out`.  The decisions and tile syntax come from the shared host
  // stage, so the device and CPU streams are byte-identical.
  virtual bool loop_filter_stage(
      av2enc::video::post::VideoFilterStage stage,
      const av2enc::video::post::VideoFilterPlan& plan,
      const av2enc::video::post::VideoLoopFilterDeviceFrame& frame,
      std::string& error) = 0;
  // §5.18.6.2 default-matrix QM (--video-qm).  Called once before the first
  // frame; the transform chain then dequantizes with the per-coefficient q2.
  virtual void set_qm(bool enabled) = 0;

  virtual const VideoGpuStats& stats() const = 0;
  virtual void reset_stats() = 0;

  // Device bytes the backend needs for the current frame configuration and
  // the size carried by the last kArenaTooSmall failure.  The CPU-emulation
  // backend returns 0 for both.
  virtual std::uint64_t required_device_bytes() const { return 0; }
  virtual std::uint64_t last_arena_required_bytes() const { return 0; }
  // Distinct outcome of the most recent device-allocation attempt.
  virtual ArenaStatus last_arena_status() const { return ArenaStatus::kOk; }

  // Test hook: replaces the current reconstructed plane with `data` so a
  // selftest can exercise intra prediction deterministically.
  virtual bool debug_set_recon(int plane, const std::vector<std::uint8_t>& data,
                               std::string& error) = 0;
};

std::unique_ptr<Backend> make_backend(av2enc::cuda::Arena* arena = nullptr);

// "cuda" or "cpu-emulation", without constructing a backend.
const char* make_backend_name();

}  // namespace av2enc::cuda::video::detail

#endif  // AV2ENC_CUDA_VIDEO_M2_BACKEND_H_

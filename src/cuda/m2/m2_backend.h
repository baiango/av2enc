// Internal device-backend contract for the CUDA M2 P-frame pipeline.
//
// The orchestration in m2_pipeline.cpp is the only consumer.  Two
// implementations exist:
//
//   m2_backend_cpu.cpp   host emulation; reuses the CPU component references
//   m2_backend_cuda.cu   real CUDA backend; reuses src/cuda/inter/{me,mc},
//                        src/cuda/intra_modes and src/cuda/tx_variants
//
// The orchestration owns every decision (partition tree, MV stack, candidate
// comparison, syntax).  The backend only performs pixel/transform work, which
// keeps the CUDA and emulation paths decision-identical by construction.
#ifndef AV2ENC_CUDA_M2_M2_BACKEND_H_
#define AV2ENC_CUDA_M2_M2_BACKEND_H_

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "encoder/inter/me/me_core.h"
#include "encoder/m2/m2_frame.h"

#include "m2_gpu.h"

namespace av2enc::cuda::m2::detail {

// One partition-node cost request: visible rectangle in luma samples.
struct VarianceNode {
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
};

// One block in a plane (luma or chroma) and a 1/8-luma-sample MV.
struct InterJob {
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
  int mv_y = 0;
  int mv_x = 0;
};

// One ME job: luma origin and block size in the current (source) frame.
struct MeRequest {
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
};

// ME output in 1/16-luma-sample units, matching me::MeCoreResult.
struct MeOutcome {
  std::int32_t mv_y = 0;
  std::int32_t mv_x = 0;
};

// Transform result for one plane: quantised levels at the adjusted size in
// natural (not scan) order, and the reconstructed samples at coded size.
struct PlaneTx {
  std::vector<std::int32_t> levels;
  std::vector<std::uint8_t> recon;
};

class Backend {
 public:
  virtual ~Backend() = default;

  virtual const char* name() const = 0;

  // Uploads the source planes, binds the reference (null for a keyframe),
  // zeroes the reconstructed planes and starts frame timing.
  virtual bool begin_frame(const av2enc::m2::M2InputFrame& source, int mi_cols,
                           int mi_rows,
                           const av2enc::m2::M2ReconFrame* reference,
                           std::string& error) = 0;

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

  // DC intra candidate for all three planes: prediction is taken from the
  // current device reconstructed planes, the residual is transformed and
  // reconstructed, but nothing is written back to the recon planes.
  virtual bool intra_candidate(int x0, int y0, int w, int h,
                               std::array<PlaneTx, 3>& planes,
                               std::string& error) = 0;

  // Writes an already-decided intra reconstruction into the device recon
  // planes (luma + chroma).
  virtual bool write_recon_420(int x0, int y0, int w, int h,
                               const std::vector<std::uint8_t>& y,
                               const std::vector<std::uint8_t>& u,
                               const std::vector<std::uint8_t>& v,
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

  virtual const M2GpuStats& stats() const = 0;
  virtual void reset_stats() = 0;

  // Test hook: replaces the current reconstructed plane with `data` so a
  // selftest can exercise intra prediction deterministically.
  virtual bool debug_set_recon(int plane, const std::vector<std::uint8_t>& data,
                               std::string& error) = 0;
};

std::unique_ptr<Backend> make_backend();

// "cuda" or "cpu-emulation", without constructing a backend.
const char* make_backend_name();

}  // namespace av2enc::cuda::m2::detail

#endif  // AV2ENC_CUDA_M2_M2_BACKEND_H_

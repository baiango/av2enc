// Format video core: 10-bit 4:2:0/4:2:2/4:4:4/4:0:0 and 8-bit 4:2:2/4:4:4/4:0:0.
//
// The historical M2 core (`src/pipeline/video/video_frame.cpp`) is 8-bit 4:2:0
// with `uint8_t` planes throughout.  This module is the depth/chroma-generic
// sibling: `uint16_t` planes, depth-aware clipping/prediction, and the chroma
// geometry from `src/formats/`.  It serves `--video` (CPU reference) and
// `--video-gpu` through one orchestration and a pixel backend abstraction, so
// the CUDA path cannot drift from the CPU decisions.
//
// Tool subset: flat key + P chain, 1..7 references, partition depth/min-block/
// force-single, eighth/quarter/half MV precision with subpel ME, DC intra
// fallback, and the opt-in inter residual.  Deblock/CDEF/CCSO/LR/GDF, tiles,
// B frames and the frame-type wrappers stay rejected for these formats (their
// CPU and CUDA implementations are 8-bit 4:2:0 only).
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "codec/inter/mc/mc_types.h"
#include "formats/format.h"
#include "pipeline/video/video_frame.h"

namespace av2enc::video::hbd {

using Mv = inter::mc::Mv;

struct PlaneView {
  const std::uint16_t* data = nullptr;
  int stride = 0;   // samples
  int width = 0;    // visible / coded extent used by prediction
  int height = 0;
};

struct FramePlanes {
  PlaneView plane[3];
};

struct VarianceNode {
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
};

struct MeJob {
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
  int ref_index = 0;
};

// One block prediction/transform job.  `x`/`y`/`w`/`h` are luma coordinates;
// chroma is derived from the sequence format.
struct InterJob {
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
  int mv_y = 0;
  int mv_x = 0;
  int ref_index = 0;
};

// One plane's transform result.  `levels` is in natural (adjusted-size) order;
// `recon` holds the coded-size reconstructed samples.
struct PlaneTx {
  int tx_size = 0;
  int adj_w = 0;
  int adj_h = 0;
  int eob = 0;
  std::vector<std::int32_t> levels;
  std::vector<std::uint16_t> recon;
  bool all_zero() const { return eob == 0; }
};

struct Options {
  fmt::Format format{};
  std::uint16_t base_q_idx = 90;
  std::vector<std::uint8_t> frame_base_q_idx;
  int num_total_refs = 1;
  int num_ref_frames = 0;
  int mv_precision = 0;  // 0 = 1/8, 1 = 1/4, 2 = 1/2 pel
  bool me_subpel_eighth = false;
  int min_block_size = 16;
  int max_partition_depth = 4;
  int me_range = 16;
  bool force_single_block = false;
  bool force_intra = false;
  bool enable_intra_fallback = true;
  bool enable_newmv = true;
  bool enable_nearmv = true;
  bool enable_inter_residual = false;
  bool force_inter_residual = false;
  double inter_rd_lambda = 4.0;
  bool trace = false;
};

struct BeginFrame {
  fmt::Format format{};
  int mi_cols = 0;  // luma
  int mi_rows = 0;
  int width = 0;   // visible luma
  int height = 0;
  int base_q_idx = 90;
  FramePlanes source;  // visible planes, stride = visible width
  FramePlanes recon;   // coded MI-aligned planes, pre-zeroed by the caller
  FramePlanes refs[kMaxRefSlots];
  int num_refs = 0;
};

// Pixel backend.  The CPU implementation is in `video_hbd.cpp`; the CUDA one
// is `src/cuda/pipeline/video/hbd/video_hbd_cuda.cu`.  Every method works on
// `uint16_t` samples and must produce bit-identical results.
class Backend {
 public:
  virtual ~Backend() = default;
  virtual const char* name() const = 0;

  // Installs the sequence options (format, qindex, ME range, precision).
  virtual void set_options(const Options& options) = 0;

  virtual bool begin_frame(const BeginFrame& frame, std::string& error) = 0;
  virtual bool end_frame(std::string& error) = 0;

  virtual bool variance_costs(const std::vector<VarianceNode>& nodes,
                              std::vector<std::int64_t>& costs,
                              std::string& error) = 0;
  virtual bool me_batch(const std::vector<MeJob>& jobs,
                        std::vector<Mv>& mvs, std::string& error) = 0;
  // Luma SSE against the current source for each job (no recon writes).
  virtual bool inter_sse_batch(const std::vector<InterJob>& jobs,
                               std::vector<std::int64_t>& sse,
                               std::string& error) = 0;
  // Chosen inter prediction: luma + chroma; writes the prediction into the
  // device/host reconstructed planes and returns host copies.
  virtual bool mc_420(const InterJob& job, std::vector<std::uint16_t> out[3],
                      std::string& error) = 0;
  // DC intra candidate for all present planes: prediction + forward/
  // quantise/inverse/reconstruct.  Nothing is written to the recon planes.
  virtual bool intra_candidate(int x, int y, int w, int h,
                               std::array<PlaneTx, 3>& planes,
                               std::string& error) = 0;
  // Inter residual chain: MC predictor + residual transform for all planes.
  // The reconstruction is returned but not written.
  virtual bool inter_residual(const InterJob& job,
                              std::array<PlaneTx, 3>& planes,
                              std::string& error) = 0;
  // Writes an already-decided reconstruction into the recon planes.
  virtual bool write_recon(int x, int y, int w, int h,
                           const std::vector<std::uint16_t>& y_plane,
                           const std::vector<std::uint16_t>& u_plane,
                           const std::vector<std::uint16_t>& v_plane,
                           std::string& error) = 0;

  virtual std::uint64_t required_device_bytes() const { return 0; }
  virtual std::uint64_t last_required_device_bytes() const { return 0; }
};

std::unique_ptr<Backend> make_cpu_backend();

// Level/plane count for a format (0 for 4:0:0).
int plane_count(const fmt::Format& format);

// One flat key + P sequence.  The caller has already validated the options.
bool encode_sequence(std::span<const VideoInputFrame> frames,
                     const Options& options, Backend& backend,
                     VideoSequenceResult& result, std::string& error);

// Option validation shared by the CLI and both backends.
bool validate_options(const Options& options, std::size_t frame_count,
                      std::string& error);

}  // namespace av2enc::video::hbd

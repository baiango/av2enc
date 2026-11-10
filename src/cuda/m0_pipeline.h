// CUDA M0 still-picture pipeline (integration layer).
//
// This file owns host orchestration only.  The M0 partition tree and the
// entropy syntax writer stay on the CPU (they are serial and cheap); the
// per-block prediction / residual / forward DCT / quantisation / dequantisation
// / inverse DCT / reconstruction run on the GPU.
//
// The shared component contract is docs/cuda-interface.md:
//   * runtime:    av2enc::cuda runtime/arena layer (src/cuda/context.*, arena.*)
//   * transform:  av2enc::cuda transform layer (src/cuda/transform.*)
//   * intra:      av2enc::cuda intra layer (src/cuda/intra.*)
//   * entropy:    av2enc::cuda entropy layer (src/cuda/entropy.*)
//
// The GPU arithmetic/delegation lives in those components.  This layer only
// supplies the M0 block plan, the host-side §5.20 syntax writer (a port of the
// serial writer in src/encoder/m0_still.cpp), and the runtime orchestration.
// It deliberately does not duplicate DCT, quantisation or DC-prediction code.
//
// What is guaranteed by this layer:
//   * the M0 partition tree, syntax order, CDF adaptation and OBU framing are
//     identical to src/encoder/m0_still.cpp;
//   * the GPU block path is bit-exact with the CPU reference for the M0 target
//     tree (32x32 and 64x64 PARTITION_NONE edge blocks);
//   * the only host round trip is source planes in, coefficients + reconstructed
//     planes out.  Entropy coding then runs on the CPU.
//
// No hidden cudaDeviceSynchronize() is used; the GPU job synchronises once per
// frame.  Device memory comes from the persistent runtime Arena, and host
// staging uses the runtime StagingBuffer (pinned) layer.

#pragma once

#include "encoder/m0_still.h"

#include <cstdint>
#include <string>

namespace av2enc::cuda {

// Host-side timings for one encode.  All values are wall-clock milliseconds
// except `kernel_ms`, which is CUDA-event elapsed time for the block loop.
struct PipelineTimings {
  double upload_ms = 0.0;
  double kernel_ms = 0.0;
  double download_ms = 0.0;
  double entropy_ms = 0.0;
  double total_ms = 0.0;
  int block_count = 0;
  int real_blocks = 0;
  int kernel_launches = 0;
};

struct PipelineOptions {
  // CUDA device ordinal.  The primary target is device 0 (RTX 3060, sm_86).
  int device = 0;
  // Print one line per stage to stderr.  Off by default; the test harness
  // enables it explicitly.
  bool verbose = false;
};

// One coding block in M0 raster/syntax order.  The fields are plain POD so the
// .cu translation unit can consume them without dragging CUDA types into the
// host-only translation units.
namespace detail {

struct BlockDesc {
  // Luma mi position (4x4 units), exactly the CPU TileEncoder coordinates.
  int mi_row = 0;
  int mi_col = 0;

  // Luma coding block size in samples.
  int luma_w = 0;
  int luma_h = 0;

  // Luma transform size.  TX_64X64 is signalled through Adjusted_Tx_Size
  // TX_32X32 in M0, so this is 32x32 for a 64x64 luma block.
  int tx_w = 0;
  int tx_h = 0;
  // 2 for the 64x64 -> adjusted 32x32 residual duplication, else 1.
  int upscale = 1;

  // Chroma transform size (half the luma block size; 32x32 for 64x64).
  int chroma_w = 0;
  int chroma_h = 0;

  // Number of quantised coefficients per plane (raster order, tx_w * tx_h).
  int luma_pels = 0;
  int chroma_pels = 0;

  // tx_sz_ctx (§5.20.7.5) used by the coefficient CDF selection, matching the
  // constants passed by m0_still.cpp's encode_block_* functions.
  int luma_tx_ctx = 0;
  int chroma_tx_ctx = 0;

  // Offset into the frame coefficient buffer: luma, then U, then V.
  int coeff_offset = 0;

  // M0 diagnostic partition modes emit zero residual; those blocks never run
  // on the GPU and recon stays at the 128 initial value (CPU behaviour).
  int real_coeffs = 0;
};

// Host-only description of one GPU job.  Source and destination buffers are
// host pointers; run_gpu_job() owns all device allocation and copies.
struct GpuJob {
  const std::uint8_t* y = nullptr;
  const std::uint8_t* u = nullptr;
  const std::uint8_t* v = nullptr;
  int width = 0;
  int height = 0;
  int chroma_width = 0;
  int chroma_height = 0;

  // Padded reconstruction geometry used by m0_still.cpp:
  //   padded_width  = (width  + 63) & ~63
  //   padded_height = (height + 63) & ~63
  //   chroma padded = (padded_width + 1) / 2 x (padded_height + 1) / 2
  int padded_width = 0;
  int padded_height = 0;
  int chroma_padded_width = 0;
  int chroma_padded_height = 0;

  const BlockDesc* blocks = nullptr;
  int block_count = 0;
  int mi_cols = 0;
  int mi_rows = 0;

  // Host outputs.  recon_* are padded planes (chroma half size), coeffs is
  // laid out per block, preds holds three int32 per block (Y, U, V).
  std::uint8_t* recon_y = nullptr;
  std::uint8_t* recon_u = nullptr;
  std::uint8_t* recon_v = nullptr;
  std::int32_t* coeffs = nullptr;
  std::int32_t* preds = nullptr;
};

struct GpuTimings {
  double upload_ms = 0.0;
  double kernel_ms = 0.0;
  double download_ms = 0.0;
  int launches = 0;
};

// Runs the block loop on `job.device`.  Defined in m0_pipeline.cu.
bool run_gpu_job(const GpuJob& job, int device, std::string& error,
                 GpuTimings* timings);

}  // namespace detail

// Encodes one M0 still picture through the CUDA residual/transform/intra path.
//
// Returns false and fills `error` on invalid input, a CUDA failure, or an
// internal syntax error.  On success `result` exactly matches what
// m0::encode_still_picture() produces for the same picture/config/mode.
//
// `timings` may be null.  The call performs one GPU synchronisation at most
// (inside the block loop); entropy coding and OBU assembly are CPU-only.
bool encode_still_picture_cuda(const m0::StillPicture& picture,
                               const m0::StillPictureConfig& config,
                               m0::StillEncodeResult& result, std::string& error,
                               m0::M0PartitionMode partition_mode,
                               const PipelineOptions& options,
                               PipelineTimings* timings);

}  // namespace av2enc::cuda

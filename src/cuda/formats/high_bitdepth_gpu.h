// GPU block stage for the standalone 10-bit (M10) still picture.
//
// This is a correctness bring-up path, not the product GPU-resident pipeline:
// it keeps the raster dependency order (one 32x32/64x64 leaf at a time), reuses
// the already-parity-tested `cuda::hbd::hbd_tx_batch` transform kernel, and does
// entropy on the CPU.  The product pipeline must batch independent blocks and
// feed the CUDA entropy coder instead.
//
// What runs on the GPU:
//   * §7.13.2.10 DC prediction from the reconstructed plane (bias 512, coded
//     plane clipping)
//   * §7.14/§7.15 forward transform, quantize, dequantize, inverse transform
//     (`hbd_tx_batch`, bit_depth = 10)
//   * reconstruction `clip1(pred + residual, 10)` into the recon planes
//
// CPU reference: `encode_high_bitdepth_cpu()` / `hbd_cpu.cpp`.  The test asserts every
// EOB, every level and every reconstructed coded sample is identical.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "formats/format.h"
#include "pipeline/still/high_bitdepth/high_bitdepth_still.h"

namespace av2enc::cuda {
class Arena;
}

namespace av2enc::cuda::high_bitdepth {

// Device bytes the M10 GPU stage needs.  The mi values are the picture's
// geometry (MiCols/MiRows); width/height are informational.
struct HighBitDepthGpuConfig {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t mi_cols = 0;
  std::uint32_t mi_rows = 0;
};

// Includes arena alignment padding, so `Arena::init(required_device_bytes(c))`
// is large enough for every buffer below.
std::uint64_t required_device_bytes(const HighBitDepthGpuConfig& config);

// Distinct outcome for the arena-backed entry point.  kArenaTooSmall carries
// the number of device bytes the call needs.
struct HighBitDepthGpuStatus {
  enum class Code : int {
    kOk = 0,
    kInvalidArgument,
    kArenaTooSmall,
    kCudaError,
  };

  Code code = Code::kOk;
  std::uint64_t required_bytes = 0;

  bool ok() const { return code == Code::kOk; }
};

// True when the M10 CUDA stage is compiled in.  (The CLI uses this to report
// unavailability rather than silently falling back.)
bool high_bitdepth_gpu_available();

// Arena-backed entry point.  Every device buffer is bump-allocated from
// `arena` once per call and rewound before return; the caller keeps ownership
// of the arena and can reuse it across pictures.
HighBitDepthGpuStatus encode_high_bitdepth_gpu(Arena& arena, const av2enc::fmt::Picture& picture,
                            const av2enc::high_bitdepth::HighBitDepthOptions& options,
                            av2enc::high_bitdepth::HighBitDepthResult& result,
                            std::string* error);

// Convenience wrapper for callers that do not own an arena (e.g. the CLI
// still-picture path).  It creates a short-lived arena sized by
// required_device_bytes(); prefer the Arena overload in a pipelined caller.
bool encode_high_bitdepth_gpu(const av2enc::fmt::Picture& picture,
                    const av2enc::high_bitdepth::HighBitDepthOptions& options,
                    av2enc::high_bitdepth::HighBitDepthResult& result, std::string* error);

}  // namespace av2enc::cuda::high_bitdepth

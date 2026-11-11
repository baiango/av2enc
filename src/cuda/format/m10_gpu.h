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
// CPU reference: `encode_m10_cpu()` / `hbd_cpu.cpp`.  The test asserts every
// EOB, every level and every reconstructed coded sample is identical.
#pragma once

#include <string>

#include "encoder/format/format.h"
#include "encoder/m10/m10_still.h"

namespace av2enc::cuda::m10 {

// True when the M10 CUDA stage is compiled in.  (The CLI uses this to report
// unavailability rather than silently falling back.)
bool m10_gpu_available();

bool encode_m10_gpu(const av2enc::fmt::Picture& picture,
                    const av2enc::m10::M10Options& options,
                    av2enc::m10::M10Result& result, std::string* error);

}  // namespace av2enc::cuda::m10

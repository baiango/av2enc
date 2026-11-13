// CUDA DPCM/FSC block stage for the CPU lossless still writer.
//
// The CPU reference `av2enc::lossless::encode_still()` owns the §5.20 syntax,
// the §8.2 entropy writer and the decoder-side reconstruction check.  This
// module computes the reconstructible per-block part on the device:
//
//   * the §5.20.5.5 DC / V-DPCM / H-DPCM luma and chroma decision (sum of
//     absolute reversible-WHT levels, the CPU's `choose_plan()` order and
//     strict-< tie-break);
//   * the FSC mode decision (`--lossless-fsc` force / auto zero-prefix test);
//   * the reversible §7.15.2.2 WHT and the §7.15.4 DPCM first difference, or
//     the identity levels for an FSC (IDTX) block.
//
// Every block is independent: a lossless reconstruction equals the source, so
// the prediction reference the CPU reads from the reconstruction is byte-equal
// to the source plane this kernel reads.  That is the same assumption the CPU
// cost functions already make.
//
// All device memory comes from one `av2enc::cuda::Arena`; no allocation happens
// in the kernel or per block, and there is a single sync per picture.
#ifndef AV2ENC_CUDA_PIPELINE_STILL_LOSSLESS_STILL_LOSSLESS_GPU_H_
#define AV2ENC_CUDA_PIPELINE_STILL_LOSSLESS_STILL_LOSSLESS_GPU_H_

#include <cstdint>
#include <string>
#include <vector>

#include "codec/entropy/lossless/lossless.h"

namespace av2enc::cuda::still_lossless {

// True when this build has the CUDA block stage compiled in.
bool available() noexcept;

// Fills `blocks` with one `lossless::BlockCoding` per 8x8 luma block in raster
// order (ceil(width/8) columns).  `options.precomputed` is ignored; the
// caller installs the returned array in `options.precomputed` before calling
// `av2enc::lossless::encode_still()`.  Returns false with `error` set on
// geometry, device or arena failure.
bool compute_block_coding(const ::av2enc::lossless::StillPicture& picture,
                          const ::av2enc::lossless::Options& options,
                          std::vector<::av2enc::lossless::BlockCoding>* blocks,
                          std::string& error);

// Device bytes needed for a width x height picture.
std::uint64_t required_device_bytes(int width, int height) noexcept;

}  // namespace av2enc::cuda::still_lossless

#endif  // AV2ENC_CUDA_PIPELINE_STILL_LOSSLESS_STILL_LOSSLESS_GPU_H_

// AV2 §5.20.8 / §7.13.4 palette intra — CUDA device API.
//
// Device twin of the CPU reference in src/encoder/palette/.  The kernels call
// the same host/device core (`palette_core.h`), so the index map, prediction
// and color-context hash are bit-exact with the CPU.
//
// Scope: 8-bit luma palette, one block per launch (64x64 or smaller).  The
// colour *selection* (median cut / exact histogram) runs on the CPU; the GPU
// owns the per-sample histogram, nearest-colour map, prediction and index
// contexts.  No allocations, no hidden synchronisation, no global state.
//
// No chroma palette exists in AV2 v1.0.0; chroma is predicted normally.

#ifndef AV2ENC_CUDA_PALETTE_PALETTE_H_
#define AV2ENC_CUDA_PALETTE_PALETTE_H_

#include <cuda_runtime.h>

#include <cstdint>

namespace av2enc::cuda {

// One palette block.  `src`/`map`/`dst` are device pointers.  The block is
// `block_w x block_h`; only `visible_w x visible_h` samples are read from the
// source, and the map is edge-replicated exactly as §5.20.8.4.
struct PaletteDeviceBlock {
  const std::uint8_t* src = nullptr;
  int src_stride = 0;
  int block_w = 0;
  int block_h = 0;
  int visible_w = 0;
  int visible_h = 0;

  const std::uint8_t* colors = nullptr;  // device array, `size` entries
  int size = 0;

  std::uint8_t* map = nullptr;  // block_w x block_h, index into colors
  int map_stride = 0;
  std::uint8_t* dst = nullptr;  // block_w x block_h prediction
  int dst_stride = 0;
  bool dither = false;
};

// 256-bin luma histogram of the visible area.  `hist` must be zeroed first.
cudaError_t palette_histogram(const std::uint8_t* src, int stride, int w,
                              int h, std::uint32_t* hist,
                              cudaStream_t stream = nullptr);

// Nearest-colour index map (§5.20.8.4), including edge replication.
cudaError_t palette_build_map(const PaletteDeviceBlock& block,
                              cudaStream_t stream = nullptr);

// §7.13.4 prediction: dst[i][j] = colors[map[i][j]].
cudaError_t palette_predict(const PaletteDeviceBlock& block,
                            cudaStream_t stream = nullptr);

// §5.20.8.5 color context for every sample.  `hash` is one byte per sample;
// `order` is `size` bytes per sample (max 8), row-major with `order_stride`.
struct PaletteContextDevice {
  const std::uint8_t* map = nullptr;
  int map_stride = 0;
  int width = 0;
  int height = 0;
  int size = 0;
  std::uint8_t* hash = nullptr;   // width*height
  std::uint8_t* order = nullptr;  // width*height*8
  int order_stride = 8;
};

cudaError_t palette_color_contexts(const PaletteContextDevice& block,
                                   cudaStream_t stream = nullptr);

// Batched forms: one grid.z slice per descriptor.  `blocks` / `contexts` are
// device arrays; the batch entry point assumes the caller validated each
// descriptor (the single-block forms validate).  `palette_apply_batch` runs
// build_map -> predict -> (optional) color contexts for every job on `stream`.
cudaError_t palette_build_map_batch(const PaletteDeviceBlock* blocks, int count,
                                    cudaStream_t stream = nullptr);
cudaError_t palette_predict_batch(const PaletteDeviceBlock* blocks, int count,
                                  cudaStream_t stream = nullptr);
cudaError_t palette_color_contexts_batch(const PaletteContextDevice* blocks,
                                         int count,
                                         cudaStream_t stream = nullptr);
cudaError_t palette_apply_batch(const PaletteDeviceBlock* blocks,
                                const PaletteContextDevice* contexts,
                                int count, cudaStream_t stream = nullptr);

}  // namespace av2enc::cuda

#endif  // AV2ENC_CUDA_PALETTE_PALETTE_H_

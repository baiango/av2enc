// AV2 v1.0.0 §7.20.4 pixel-classified Wiener filter — CUDA launcher.
//
// Device implementation of the CPU reference in
// src/encoder/filters/spec_tables/pc_wiener.{h,cpp}.  Covers:
//
//   * §7.20.4 box features, the 4096-entry LUT -> 64-class map,
//     Pc_Wiener_Sub_Classify and the fixed 25-tap Pc_Wiener_Filters bank,
//   * Pc_Wiener_Sub_Classify2 (frame-class subclass lookup),
//   * skipFilter = 1 (class map only) and skipFilter = 0 (filter blocks),
//   * §7.14.2 get_q / Pc_Wiener_Mode_Weights / Pc_Wiener_Mode_Offsets,
//   * LrTxSkip with the stripe and tile clamps,
//   * 8, 10 and 12 bit luma, arbitrary frame geometry, and every LR unit size
//     (64 / 128 / 256 / 512 luma samples).
//
// Memory: every device byte comes from the caller's persistent
// av2enc::cuda::Arena.  `required_device_bytes()` reports the exact arena
// footprint (uploaded normative tables + class map + scratch).  Creation
// uploads the tables once; launchers never allocate, never cudaMalloc and never
// synchronise.  If the arena is too small, create() returns kArenaExhausted and
// required_bytes() reports the size the caller must provide.
//
// The class map and scratch are single-instance per workspace: serialise calls
// on one workspace (the Arena is single-threaded by contract).
//
// Filter semantics match the specification and the CPU reference:
//   * pc_wiener_filter_grid()     §7.20: LrFrame := CdefFrame, then every
//                                 4x4 MI block is filtered with the class from
//                                 the supplied grid.
//   * pc_wiener_filter_units()    same copy, then only units whose
//                                 UnitType == RESTORE_PC_WIENER (1) filter.
//   * pc_wiener_filter_block()    one block only; no copy.
//   * pc_wiener_filter_class_block() one block with an explicit class (used by
//                                 frame-class mode and by exhaustive tests).

#ifndef AV2ENC_CUDA_PC_WIENER_PC_WIENER_H_
#define AV2ENC_CUDA_PC_WIENER_PC_WIENER_H_

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>

#include "cuda/arena.h"
#include "cuda/error.h"
#include "cuda/pc_wiener/pc_wiener_device.cuh"

namespace av2enc::cuda::pc_wiener {

// Frame geometry plus the restoration unit size.  `mi_cols`/`mi_rows` are
// derived: ceil(width / 4) and ceil(height / 4).
struct PcWienerConfig {
  int width = 0;
  int height = 0;
  int bit_depth = 8;
  int unit_size = 64;  // §5.18.7 LoopRestorationSize; 64/128/256/512
};

// Exact arena bytes needed by PcWienerWorkspace::create().  Returns
// kInvalidArgument for an unsupported configuration (out is set to 0).
Status pc_wiener_required_device_bytes(const PcWienerConfig& config,
                                       std::size_t* out);
// Convenience form; 0 for an invalid configuration.
std::size_t pc_wiener_required_device_bytes(const PcWienerConfig& config);

// Persistent per-frame PC-Wiener workspace.  Holds the uploaded normative
// tables, the class map and the unit-active scratch bitmap.  It does not own
// the arena; reset() only forgets the pointers.
class PcWienerWorkspace {
 public:
  PcWienerWorkspace() = default;
  PcWienerWorkspace(const PcWienerWorkspace&) = delete;
  PcWienerWorkspace& operator=(const PcWienerWorkspace&) = delete;
  ~PcWienerWorkspace() = default;

  // Allocate from `arena` and upload the CPU normative tables.
  //
  // On kArenaExhausted, required_bytes() holds the exact size that a fresh
  // arena must provide; any partial allocation is rewound, so the arena is
  // left exactly as it was on entry.
  Status create(Arena& arena, const PcWienerConfig& config,
                std::string* error = nullptr);

  // Forget all pointers.  Arena memory is not freed (the arena owns it).
  void reset() noexcept;

  bool initialized() const noexcept { return class_grid_ != nullptr; }
  const PcWienerConfig& config() const noexcept { return config_; }
  std::size_t required_bytes() const noexcept { return required_bytes_; }
  std::size_t persistent_bytes() const noexcept { return persistent_bytes_; }
  std::size_t scratch_bytes() const noexcept { return scratch_bytes_; }
  int mi_cols() const noexcept { return mi_cols_; }
  int mi_rows() const noexcept { return mi_rows_; }
  int class_stride() const noexcept { return mi_cols_; }
  int unit_rows() const noexcept { return unit_rows_; }
  int unit_cols() const noexcept { return unit_cols_; }

  // Persistent class map (§7.20.4 FilterClass), mi_rows * mi_cols bytes.
  std::uint8_t* class_grid() noexcept { return class_grid_; }
  const std::uint8_t* class_grid() const noexcept { return class_grid_; }

  // Scratch: one byte per MI block, used by the unit-gated launcher.
  std::uint8_t* unit_active() noexcept { return unit_active_; }

  const PcWienerTables& tables() const noexcept { return tables_; }

 private:
  void clear() noexcept;

  PcWienerConfig config_{};
  std::size_t required_bytes_ = 0;
  std::size_t persistent_bytes_ = 0;
  std::size_t scratch_bytes_ = 0;
  int mi_cols_ = 0;
  int mi_rows_ = 0;
  int unit_rows_ = 0;
  int unit_cols_ = 0;
  PcWienerTables tables_{};
  std::uint8_t* class_grid_ = nullptr;
  std::uint8_t* unit_active_ = nullptr;
  Arena* arena_ = nullptr;
};

// ---------------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------------
//
// `params.lr_tx_skip` must be device memory; the plane pointers in `src` and
// `lr` must be device memory.  A caller-supplied `class_grid` must be a device
// array with row stride ws.mi_cols() and ws.mi_rows() rows.  No function below
// allocates or synchronises.  All return kOk after a successful launch (launch
// errors are sticky and are reported by the next call / at the caller's sync
// point).

// skipFilter = 1 for every MI block: fill ws.class_grid().
Status pc_wiener_classify(PcWienerWorkspace& ws, const PcWienerSource& src,
                          const PcWienerParams& params,
                          cudaStream_t stream = nullptr);

// skipFilter = 1 for one block: write one byte to filter_class[y>>2][x>>2].
Status pc_wiener_classify_block(PcWienerWorkspace& ws, const PcWienerSource& src,
                                const PcWienerParams& params, int x, int y,
                                std::uint8_t* filter_class,
                                int filter_class_stride,
                                cudaStream_t stream = nullptr);

// skipFilter = 0 for every MI block.  Copies CdefFrame to LrFrame first
// (§7.20), then filters using `class_grid` (nullptr = ws.class_grid()).
Status pc_wiener_filter_grid(PcWienerWorkspace& ws, const PcWienerSource& src,
                             const PcWienerParams& params,
                             const std::uint8_t* class_grid,
                             std::uint16_t* lr, int lr_stride,
                             cudaStream_t stream = nullptr);

// skipFilter = 0 for the units with unit_type[unit] == RESTORE_PC_WIENER (1).
// `unit_type` is a device array of ws.unit_rows() * ws.unit_cols() bytes.
Status pc_wiener_filter_units(PcWienerWorkspace& ws, const PcWienerSource& src,
                              const PcWienerParams& params,
                              const std::uint8_t* class_grid,
                              const std::uint8_t* unit_type, std::uint16_t* lr,
                              int lr_stride, cudaStream_t stream = nullptr);

// One block, class derived from (x, y).  Writes only the w x h rectangle.
Status pc_wiener_filter_block(PcWienerWorkspace& ws, const PcWienerSource& src,
                              const PcWienerParams& params, int x, int y, int w,
                              int h, std::uint16_t* lr, int lr_stride,
                              cudaStream_t stream = nullptr);

// One block with an explicit Pc_Wiener class (0..63).
Status pc_wiener_filter_class_block(PcWienerWorkspace& ws,
                                    const PcWienerSource& src,
                                    const PcWienerParams& params, int x, int y,
                                    int w, int h, int cls, std::uint16_t* lr,
                                    int lr_stride,
                                    cudaStream_t stream = nullptr);

// Pc_Wiener_Sub_Classify[filter_set][0..255] -> out[0..255].
Status pc_wiener_class_filter_lut(PcWienerWorkspace& ws, int filter_set,
                                  std::uint8_t* out,
                                  cudaStream_t stream = nullptr);

// Pc_Wiener_Sub_Classify2[filter_set][num_classes_idx][0..255] -> out[0..255].
Status pc_wiener_subclass_lookup(PcWienerWorkspace& ws, int filter_set,
                                 int num_classes_idx, std::uint8_t* out,
                                 cudaStream_t stream = nullptr);

}  // namespace av2enc::cuda::pc_wiener

#endif  // AV2ENC_CUDA_PC_WIENER_PC_WIENER_H_

// AV2 §7.20.5 GDF filter — CUDA launcher API.
//
// The kernel calls the same shared `__host__ __device__` core as the CPU
// reference (gdf_device.cuh + the normative generated tables), so device
// output is bit-exact with src/encoder/filters/spec_tables/gdf.cpp for
// identical planes and parameters.
//
// Scope: luma only (as the specification defines GDF), bit depth 8/10/12,
// unit sizes 32/64/128 and arbitrary even unit extents down to 2x2.  The
// §5.20.10.3 per-unit `use_gdf` decision is made on the CPU and applied on the
// GPU; §7.20.1 table indices are derived on the GPU (gdf_derive_indices).
//
// Memory rules for this component:
//   * every device buffer comes from the caller's persistent Arena;
//   * gdf_buffers_init() is the only entry point that allocates, and it never
//     allocates after the first call for a given configuration;
//   * launchers allocate nothing, copy nothing and never synchronise.
//
// The test drives two launches with Arena::mark()/rewind() and checks that
// used() and peak_used() do not grow.

#ifndef AV2ENC_CUDA_GDF_GDF_H_
#define AV2ENC_CUDA_GDF_GDF_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include "cuda/arena.h"
#include "cuda/gdf/gdf_types.h"

namespace av2enc::cuda::gdf {

// Largest unit capacity accepted by gdf_buffers_init (one descriptor per
// unit; 2^22 units covers an 8K frame at 32x32 units with room to spare).
inline constexpr int kGdfMaxUnits = 1 << 22;

// Sizing request.  `max_units` is a job-slot count: the larger of the batch
// size and gdf_grid_slots(cols, rows) for a unit-grid launch.  A grid launch
// expands every GdfBlks unit into up to kGdfMaxRegionsPerUnit stripe regions,
// so size grids with gdf_grid_slots() (cols*rows*3).
struct GdfConfig {
  int max_units = 1;
};

// Device bytes needed for this configuration: the device-resident §9.8 tables
// (135,680 B: alpha/weight/bias/intra/inter/coords), the persistent job array,
// the persistent use_gdf mask and the grid-expansion scratch job array.  Each
// buffer is 256-byte aligned, matching Arena allocations.  Returns 0 for an
// invalid configuration (max_units <= 0).
std::size_t required_device_bytes(const GdfConfig& config) noexcept;

// Device-resident buffers owned by one GdfConfig.  `jobs` is filled by the
// caller (gdf_upload_jobs or the caller's own copy) for the batch API;
// `use_gdf` is the CPU decision grid for gdf_apply_grid; `grid_jobs` is
// scratch that gdf_apply_grid expands the unit grid into.  None of these may
// alias the LrFrame plane or the source planes.
struct GdfDeviceBuffers {
  GdfUnitJob* jobs = nullptr;
  std::uint8_t* use_gdf = nullptr;
  GdfUnitJob* grid_jobs = nullptr;
  GdfTables tables{};  // device copies of the §9.8 tables, uploaded once
  int max_units = 0;
};

// Allocates the persistent buffers once.  Returns Status::kArenaExhausted and
// leaves the arena's bump pointer untouched when the arena cannot satisfy
// `required_device_bytes(config)`; *required_bytes (when non-null) receives
// the exact requirement so the caller can grow the arena.  No allocation
// happens inside any launcher.
Status gdf_buffers_init(Arena& arena, const GdfConfig& config,
                        GdfDeviceBuffers* out,
                        std::size_t* required_bytes = nullptr,
                        std::string* error = nullptr);

// Setup-time host-to-device copies into the persistent buffers.  These are
// not called by the launchers; a frame loop does one copy per frame at most.
Status gdf_upload_jobs(const GdfDeviceBuffers& buffers,
                       const GdfUnitJob* host_jobs, int job_count,
                       cudaStream_t stream = nullptr,
                       std::string* error = nullptr);

Status gdf_upload_use_gdf(const GdfDeviceBuffers& buffers,
                          const std::uint8_t* host_use_gdf, int unit_count,
                          cudaStream_t stream = nullptr,
                          std::string* error = nullptr);

// §7.20.1 derivation for a batch of descriptions.  `d_params` is device memory
// (an array of descriptors, each with device pointers to its two ref arrays)
// and `d_out` receives one GdfTableIndices per entry.  One thread per entry.
Status gdf_derive_indices(const GdfIndexParams* d_params,
                          GdfTableIndices* d_out, int count,
                          cudaStream_t stream = nullptr,
                          std::string* error = nullptr);

// Filter a batch of units.  `d_jobs` must be device memory; each job's
// width/height must be even and at most `max_unit_size` (32/64/128, or any
// even value in [2, 128] for tests).  A job with a non-negative
// stripe_start_y carries its own §7.20.2 stripe bounds (use this for units
// that span a stripe); otherwise the GdfFrameSource stripe applies.  Units
// with use_gdf == 0 are skipped.
// Host validation covers the source description, pointers, counts and
// max_unit_size; per-job contents are device memory and are defended against
// in the kernel (out-of-range indices or oversize extents make that unit a
// no-op rather than reading out of bounds).
Status gdf_apply_units(const GdfFrameSource& src, const GdfUnitJob* d_jobs,
                       int job_count, int max_unit_size,
                       const GdfTables& tables,
                       cudaStream_t stream = nullptr,
                       std::string* error = nullptr);

// §5.18.7.9 unit-grid path: derive the unit geometry, split every enabled
// GdfBlks unit into its §7.20.2 64-line stripe regions, derive the §7.20.1
// indices on the GPU, expand the regions into `buffers.grid_jobs`, then filter
// them.  Two launches on `stream`, no synchronisation and no allocation.
// `width`/`height` are the padded luma extents and `mi_rows` the padded MiRows
// (used for the last-unit-row clamp).  The buffers must have at least
// gdf_grid_slots(cols, rows) job slots.
Status gdf_apply_grid(const GdfFrameGrid& frame,
                      const GdfDeviceBuffers& buffers,
                      cudaStream_t stream = nullptr,
                      std::string* error = nullptr);

}  // namespace av2enc::cuda::gdf

#endif  // AV2ENC_CUDA_GDF_GDF_H_

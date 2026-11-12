// AV2 §7.20.5 GDF filter — CUDA kernels and launchers.
//
// One CUDA block per unit (grid path: per stripe region); the block
// cooperatively computes the per-2x2-cell gradient sums and direction classes
// into shared memory, then one thread per output sample applies the 22-tap
// filter.  Source samples come from CurrFrame/CdefFrame only and every output
// sample is written once, so no inter-block ordering constraint exists.
//
// The §5.20.10.3 use_gdf decision is host input (a device mask or job field);
// §7.20.1 qpIdx/refDstIdx are derived on the device.  Launchers perform no
// allocation, no copies and no synchronisation.

#include "cuda/codec/filters/gdf/gdf.h"

#include "cuda/codec/filters/gdf/gdf_device.cuh"
#include "codec/filters/tables/gdf_tables.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace av2enc::cuda::gdf {
namespace {

constexpr unsigned kGdfBlockThreads = 256;

namespace tables = ::av2enc::filters::spec_tables;

// Flat table extents (§9.8).  These are device copies uploaded once by
// gdf_buffers_init; device code never references the host constexpr objects.
constexpr std::size_t kAlphaCount = static_cast<std::size_t>(kGdfNumRefDst) *
                                    kGdfNumQpIdx * kGdfNumTaps *
                                    kGdfNumClasses;
constexpr std::size_t kWeightCount = static_cast<std::size_t>(kGdfNumRefDst) *
                                     kGdfNumQpIdx * kGdfNumComponents *
                                     kGdfNumTaps * kGdfNumClasses;
constexpr std::size_t kBiasCount =
    static_cast<std::size_t>(kGdfNumRefDst) * kGdfNumQpIdx * kGdfNumComponents;
constexpr std::size_t kIntraCount =
    static_cast<std::size_t>(kGdfNumQpIdx) * kGdfIntraPositions;
constexpr std::size_t kInterCount =
    static_cast<std::size_t>(5) * kGdfNumQpIdx * kGdfInterPositions;
constexpr std::size_t kCoordsCount = 18;

std::size_t align_up_bytes(std::size_t bytes) noexcept {
  const std::size_t alignment = Arena::kAlignment;
  return (bytes + alignment - 1) & ~(alignment - 1);
}

std::size_t table_device_bytes() noexcept {
  return align_up_bytes(kAlphaCount * sizeof(std::uint16_t)) +
         align_up_bytes(kWeightCount * sizeof(std::int16_t)) +
         align_up_bytes(kBiasCount * sizeof(std::int32_t)) +
         align_up_bytes(kIntraCount * sizeof(std::int16_t)) +
         align_up_bytes(kInterCount * sizeof(std::int16_t)) +
         align_up_bytes(kCoordsCount * sizeof(GdfCoord));
}

void set_error(std::string* error, const char* context,
               const std::string& detail) {
  if (error != nullptr) {
    *error = std::string(context) + ": " + detail;
  }
}

AV2ENC_GDF_HD bool valid_bit_depth(int bit_depth) noexcept {
  return bit_depth == 8 || bit_depth == 10 || bit_depth == 12;
}

bool valid_unit_size(int unit_size) noexcept {
  return unit_size >= kGdfMinUnitSize && unit_size <= kGdfMaxUnitSize &&
         (unit_size & 1) == 0;
}

bool valid_source(const GdfFrameSource& src) noexcept {
  return src.curr != nullptr && src.cdef != nullptr && src.curr_stride > 0 &&
         src.cdef_stride > 0 && src.start_x >= 0 && src.end_x >= src.start_x &&
         src.end_x < src.curr_stride && src.end_x < src.cdef_stride &&
         src.start_y >= 0 && src.end_y >= src.start_y &&
         src.stripe_start_y <= src.stripe_end_y;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__global__ void gdf_derive_indices_kernel(const GdfIndexParams* params,
                                          GdfTableIndices* out, int count) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= count) {
    return;
  }
  out[index] = gdf_derive_table_indices(params[index]);
}

// Expands the GdfBlks grid into per-stripe-region unit jobs, applying the CPU
// use_gdf mask and the §7.20.1 derivation on the device.  One thread per job
// slot; unused slots are marked disabled.
__global__ void gdf_prepare_grid_kernel(GdfFrameGrid frame,
                                        const std::uint8_t* use_gdf,
                                        GdfUnitJob* out, int unit_count,
                                        int slot_count) {
  const int slot = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (slot >= slot_count) {
    return;
  }
  const int unit = slot / kGdfMaxRegionsPerUnit;
  const int region_index = slot - unit * kGdfMaxRegionsPerUnit;

  GdfUnitJob job;
  job.lr = frame.lr;
  job.lr_stride = frame.lr_stride;
  job.pix_scale = frame.pix_scale;
  job.bit_depth = frame.bit_depth;
  job.use_gdf = 0;

  if (unit < unit_count && use_gdf[unit] != 0) {
    const int row = unit / frame.cols;
    const int col = unit - row * frame.cols;
    GdfRegion regions[kGdfMaxRegionsPerUnit];
    const int count =
        gdf_unit_regions(frame.unit_size, row, col, frame.cols, frame.rows,
                         frame.mi_rows, frame.width, frame.height, regions);
    if (region_index < count) {
      const GdfRegion& region = regions[region_index];
      const GdfTableIndices indices = gdf_derive_table_indices(frame.index);
      job.x = region.x;
      job.y = region.y;
      job.w = region.w;
      job.h = region.h;
      job.qp_idx = indices.qp_idx;
      job.ref_dst_idx = indices.ref_dst_idx;
      job.stripe_start_y = region.stripe_start_y;
      job.stripe_end_y = region.stripe_end_y;
      job.use_gdf = (region.w >= kGdfMinUnitSize &&
                     region.h >= kGdfMinUnitSize && (region.w & 1) == 0 &&
                     (region.h & 1) == 0)
                        ? 1
                        : 0;
    }
  }
  out[slot] = job;
}

__global__ void gdf_units_kernel(GdfFrameSource src, GdfTables tables,
                                 const GdfUnitJob* jobs, int count,
                                 int max_unit_size) {
  const int index = static_cast<int>(blockIdx.x);
  if (index >= count) {
    return;
  }
  const GdfUnitJob job = jobs[index];
  if (job.use_gdf == 0 || job.lr == nullptr || job.lr_stride <= 0) {
    return;
  }
  // Per-job fields are device memory; defend against malformed descriptors so
  // a bad job cannot read the tables or shared memory out of bounds.
  if (job.w < kGdfMinUnitSize || job.h < kGdfMinUnitSize ||
      (job.w & 1) != 0 || (job.h & 1) != 0 || job.w > max_unit_size ||
      job.h > max_unit_size || job.qp_idx < 0 ||
      job.qp_idx >= kGdfNumQpIdx || job.ref_dst_idx < 0 ||
      job.ref_dst_idx >= kGdfNumRefDst || !valid_bit_depth(job.bit_depth) ||
      job.pix_scale < 1 || job.pix_scale > 4) {
    return;
  }
  if (job.stripe_start_y >= 0 && job.stripe_end_y < job.stripe_start_y) {
    return;
  }

  // A job may carry its own 64-line stripe bounds (§7.20.2); otherwise the
  // frame-level GdfFrameSource stripe applies.
  GdfFrameSource local = src;
  if (job.stripe_start_y >= 0) {
    local.stripe_start_y = job.stripe_start_y;
    local.stripe_end_y = job.stripe_end_y;
  }

  const int w = job.w;
  const int h = job.h;
  const int cw = w >> 1;
  const int ch = h >> 1;
  const int cells = cw * ch;

  extern __shared__ __align__(16) unsigned char smem[];
  std::uint16_t* taps = reinterpret_cast<std::uint16_t*>(smem);
  std::uint8_t* classes = reinterpret_cast<std::uint8_t*>(taps + cells * 4);

  // Shared statistics: four shifted 4x4 gradient sums and the direction class
  // per 2x2 output cell.
  for (int c = static_cast<int>(threadIdx.x); c < cells;
       c += static_cast<int>(blockDim.x)) {
    const int ci = c / cw;
    const int cj = c - ci * cw;
    int sums[4];
    gdf_cell_sums(local, job.x, job.y, ci, cj, sums);
    std::uint16_t cell_taps[4];
    gdf_taps_from_sums(sums, job.bit_depth, cell_taps);
    for (int d = 0; d < 4; ++d) {
      taps[c * 4 + d] = cell_taps[d];
    }
    classes[c] = static_cast<std::uint8_t>(gdf_class_from_sums(sums));
  }
  __syncthreads();

  const int pixels = w * h;
  for (int p = static_cast<int>(threadIdx.x); p < pixels;
       p += static_cast<int>(blockDim.x)) {
    const int i = p / w;
    const int j = p - i * w;
    const int cell = (i >> 1) * cw + (j >> 1);
    const int delta = gdf_filter_delta(
        local, tables, job.x + j, job.y + i, classes[cell], taps + cell * 4,
        job.qp_idx, job.ref_dst_idx, job.pix_scale, job.bit_depth);
    const std::ptrdiff_t offset =
        static_cast<std::ptrdiff_t>(job.y + i) * job.lr_stride + (job.x + j);
    const int base = static_cast<int>(job.lr[offset]);
    job.lr[offset] = static_cast<std::uint16_t>(
        gdf_clip1(job.bit_depth, base + delta));
  }
}

}  // namespace

std::size_t required_device_bytes(const GdfConfig& config) noexcept {
  if (config.max_units <= 0) {
    return 0;
  }
  const std::size_t units = static_cast<std::size_t>(config.max_units);
  const std::size_t job_bytes = align_up_bytes(units * sizeof(GdfUnitJob));
  const std::size_t mask_bytes = align_up_bytes(units * sizeof(std::uint8_t));
  return table_device_bytes() + job_bytes + mask_bytes +
         job_bytes;  // tables + jobs + use_gdf + grid scratch
}

Status gdf_buffers_init(Arena& arena, const GdfConfig& config,
                        GdfDeviceBuffers* out, std::size_t* required_bytes,
                        std::string* error) {
  if (required_bytes != nullptr) {
    *required_bytes = 0;
  }
  if (out == nullptr) {
    set_error(error, "gdf_buffers_init", "null output");
    return Status::kInvalidArgument;
  }
  *out = GdfDeviceBuffers{};
  if (!arena.initialized()) {
    set_error(error, "gdf_buffers_init", "arena is not initialized");
    return Status::kNotInitialized;
  }
  if (config.max_units <= 0 || config.max_units > kGdfMaxUnits) {
    set_error(error, "gdf_buffers_init",
              "max_units must be in [1, " + std::to_string(kGdfMaxUnits) + "]");
    return Status::kInvalidArgument;
  }

  const std::size_t need = required_device_bytes(config);
  if (required_bytes != nullptr) {
    *required_bytes = need;
  }
  if (need == 0) {
    set_error(error, "gdf_buffers_init", "invalid configuration");
    return Status::kInvalidArgument;
  }

  // Single mark for the whole reservation.  Every buffer is requested with
  // allocate_checked (no pre-check on remaining(): 256-byte alignment can make
  // a request fail even when remaining() >= bytes).  Any failure rewinds to
  // the mark so used() is unchanged and the caller can grow the arena.
  const std::uint64_t entry_remaining = arena.remaining();
  const std::uint64_t mark = arena.mark();
  void* jobs_ptr = nullptr;
  void* mask_ptr = nullptr;
  void* grid_ptr = nullptr;
  void* alpha_ptr = nullptr;
  void* weight_ptr = nullptr;
  void* bias_ptr = nullptr;
  void* intra_ptr = nullptr;
  void* inter_ptr = nullptr;
  void* coords_ptr = nullptr;

  Status status = Status::kOk;
  const auto take = [&arena, &status](std::size_t bytes, void** out) {
    if (is_ok(status)) {
      status = arena.allocate_checked(bytes, out);
    }
  };
  const std::size_t units = static_cast<std::size_t>(config.max_units);
  take(units * sizeof(GdfUnitJob), &jobs_ptr);
  take(units * sizeof(std::uint8_t), &mask_ptr);
  take(units * sizeof(GdfUnitJob), &grid_ptr);
  take(kAlphaCount * sizeof(std::uint16_t), &alpha_ptr);
  take(kWeightCount * sizeof(std::int16_t), &weight_ptr);
  take(kBiasCount * sizeof(std::int32_t), &bias_ptr);
  take(kIntraCount * sizeof(std::int16_t), &intra_ptr);
  take(kInterCount * sizeof(std::int16_t), &inter_ptr);
  take(kCoordsCount * sizeof(GdfCoord), &coords_ptr);
  if (!is_ok(status)) {
    (void)arena.rewind(mark);
    if (status == Status::kArenaExhausted) {
      set_error(error, "gdf_buffers_init",
                "arena exhausted: required " + std::to_string(need) +
                    " bytes, remaining at entry " +
                    std::to_string(entry_remaining) + " bytes");
      return Status::kArenaExhausted;
    }
    set_error(error, "gdf_buffers_init", status_name(status));
    return status;
  }

  GdfUnitJob* jobs = static_cast<GdfUnitJob*>(jobs_ptr);
  std::uint8_t* use_gdf = static_cast<std::uint8_t*>(mask_ptr);
  GdfUnitJob* grid_jobs = static_cast<GdfUnitJob*>(grid_ptr);
  std::uint16_t* alpha = static_cast<std::uint16_t*>(alpha_ptr);
  std::int16_t* weight = static_cast<std::int16_t*>(weight_ptr);
  std::int32_t* bias = static_cast<std::int32_t*>(bias_ptr);
  std::int16_t* intra_error = static_cast<std::int16_t*>(intra_ptr);
  std::int16_t* inter_error = static_cast<std::int16_t*>(inter_ptr);
  GdfCoord* coords = static_cast<GdfCoord*>(coords_ptr);

  // One-time upload of the normative tables into arena device memory.
  GdfCoord host_coords[kCoordsCount];
  for (std::size_t i = 0; i < kCoordsCount; ++i) {
    host_coords[i].dy = tables::kGdfCoords[i].dy;
    host_coords[i].dx = tables::kGdfCoords[i].dx;
  }
  struct Upload {
    const void* host;
    void* device;
    std::size_t bytes;
  };
  const Upload uploads[] = {
      {&tables::kGdfAlpha[0][0][0][0], alpha,
       kAlphaCount * sizeof(std::uint16_t)},
      {&tables::kGdfWeight[0][0][0][0][0], weight,
       kWeightCount * sizeof(std::int16_t)},
      {&tables::kGdfBias[0][0][0], bias, kBiasCount * sizeof(std::int32_t)},
      {&tables::kGdfIntraError[0][0], intra_error,
       kIntraCount * sizeof(std::int16_t)},
      {&tables::kGdfInterError[0][0][0], inter_error,
       kInterCount * sizeof(std::int16_t)},
      {host_coords, coords, kCoordsCount * sizeof(GdfCoord)},
  };
  for (const Upload& upload : uploads) {
    const Status upload_status = status_from_cuda(cudaMemcpy(
        upload.device, upload.host, upload.bytes, cudaMemcpyHostToDevice));
    if (!is_ok(upload_status)) {
      (void)arena.rewind(mark);
      set_error(error, "gdf_buffers_init: table upload",
                status_name(upload_status));
      return upload_status;
    }
  }

  out->jobs = jobs;
  out->use_gdf = use_gdf;
  out->grid_jobs = grid_jobs;
  out->max_units = config.max_units;
  out->tables.alpha = alpha;
  out->tables.weight = weight;
  out->tables.bias = bias;
  out->tables.intra_error = intra_error;
  out->tables.inter_error = inter_error;
  out->tables.coords = coords;
  return Status::kOk;
}

Status gdf_upload_jobs(const GdfDeviceBuffers& buffers,
                       const GdfUnitJob* host_jobs, int job_count,
                       cudaStream_t stream, std::string* error) {
  if (buffers.jobs == nullptr || buffers.max_units <= 0) {
    set_error(error, "gdf_upload_jobs", "buffers are not initialized");
    return Status::kNotInitialized;
  }
  if (job_count < 0 || job_count > buffers.max_units) {
    set_error(error, "gdf_upload_jobs",
              "job_count must be in [0, max_units=" +
                  std::to_string(buffers.max_units) + "]");
    return Status::kInvalidArgument;
  }
  if (job_count == 0) {
    return Status::kOk;
  }
  if (host_jobs == nullptr) {
    set_error(error, "gdf_upload_jobs", "null host jobs");
    return Status::kInvalidArgument;
  }
  const cudaError_t result = cudaMemcpyAsync(
      buffers.jobs, host_jobs,
      static_cast<std::size_t>(job_count) * sizeof(GdfUnitJob),
      cudaMemcpyHostToDevice, stream);
  const Status status = status_from_cuda(result);
  if (!is_ok(status)) {
    set_error(error, "gdf_upload_jobs", status_name(status));
  }
  return status;
}

Status gdf_upload_use_gdf(const GdfDeviceBuffers& buffers,
                          const std::uint8_t* host_use_gdf, int unit_count,
                          cudaStream_t stream, std::string* error) {
  if (buffers.use_gdf == nullptr || buffers.max_units <= 0) {
    set_error(error, "gdf_upload_use_gdf", "buffers are not initialized");
    return Status::kNotInitialized;
  }
  if (unit_count < 0 || unit_count > buffers.max_units) {
    set_error(error, "gdf_upload_use_gdf",
              "unit_count must be in [0, max_units=" +
                  std::to_string(buffers.max_units) + "]");
    return Status::kInvalidArgument;
  }
  if (unit_count == 0) {
    return Status::kOk;
  }
  if (host_use_gdf == nullptr) {
    set_error(error, "gdf_upload_use_gdf", "null host mask");
    return Status::kInvalidArgument;
  }
  const cudaError_t result =
      cudaMemcpyAsync(buffers.use_gdf, host_use_gdf,
                      static_cast<std::size_t>(unit_count) * sizeof(std::uint8_t),
                      cudaMemcpyHostToDevice, stream);
  const Status status = status_from_cuda(result);
  if (!is_ok(status)) {
    set_error(error, "gdf_upload_use_gdf", status_name(status));
  }
  return status;
}

Status gdf_derive_indices(const GdfIndexParams* d_params,
                          GdfTableIndices* d_out, int count,
                          cudaStream_t stream, std::string* error) {
  if (count < 0 || count > kGdfMaxUnits) {
    set_error(error, "gdf_derive_indices", "count out of range");
    return Status::kInvalidArgument;
  }
  if (count == 0) {
    return Status::kOk;
  }
  if (d_params == nullptr || d_out == nullptr) {
    set_error(error, "gdf_derive_indices", "null parameter or output buffer");
    return Status::kInvalidArgument;
  }
  const unsigned count_u = static_cast<unsigned>(count);
  const unsigned blocks = (count_u + kGdfBlockThreads - 1) / kGdfBlockThreads;
  gdf_derive_indices_kernel<<<blocks, kGdfBlockThreads, 0, stream>>>(
      d_params, d_out, count);
  const Status status = status_from_cuda(cudaGetLastError());
  if (!is_ok(status)) {
    set_error(error, "gdf_derive_indices launch", status_name(status));
  }
  return status;
}

Status gdf_apply_units(const GdfFrameSource& src, const GdfUnitJob* d_jobs,
                       int job_count, int max_unit_size,
                       const GdfTables& tables, cudaStream_t stream,
                       std::string* error) {
  if (!valid_source(src)) {
    set_error(error, "gdf_apply_units", "invalid luma source description");
    return Status::kInvalidArgument;
  }
  if (job_count < 0 || job_count > kGdfMaxUnits) {
    set_error(error, "gdf_apply_units", "job_count out of range");
    return Status::kInvalidArgument;
  }
  if (job_count > 0 && d_jobs == nullptr) {
    set_error(error, "gdf_apply_units", "null device job array");
    return Status::kInvalidArgument;
  }
  if (!valid_unit_size(max_unit_size)) {
    set_error(error, "gdf_apply_units",
              "max_unit_size must be even and in [2, 128]");
    return Status::kInvalidArgument;
  }
  if (job_count == 0) {
    return Status::kOk;
  }
  const std::size_t shared_bytes = gdf_unit_shared_bytes(max_unit_size);
  const unsigned job_count_u = static_cast<unsigned>(job_count);
  gdf_units_kernel<<<job_count_u, kGdfBlockThreads, shared_bytes, stream>>>(
      src, tables, d_jobs, job_count, max_unit_size);
  const Status status = status_from_cuda(cudaGetLastError());
  if (!is_ok(status)) {
    set_error(error, "gdf_apply_units launch", status_name(status));
  }
  return status;
}

Status gdf_apply_grid(const GdfFrameGrid& frame,
                      const GdfDeviceBuffers& buffers, cudaStream_t stream,
                      std::string* error) {
  if (!valid_source(frame.src)) {
    set_error(error, "gdf_apply_grid", "invalid luma source description");
    return Status::kInvalidArgument;
  }
  if (frame.lr == nullptr || frame.lr_stride <= 0) {
    set_error(error, "gdf_apply_grid", "null LrFrame plane or bad stride");
    return Status::kInvalidArgument;
  }
  if (!valid_bit_depth(frame.bit_depth)) {
    set_error(error, "gdf_apply_grid", "bit_depth must be 8, 10 or 12");
    return Status::kInvalidArgument;
  }
  if (frame.pix_scale < 1 || frame.pix_scale > 4) {
    set_error(error, "gdf_apply_grid", "pix_scale must be in [1, 4]");
    return Status::kInvalidArgument;
  }
  if (!valid_unit_size(frame.unit_size) || (frame.unit_size & 3) != 0) {
    set_error(error, "gdf_apply_grid",
              "unit_size must be a multiple of 4 in [4, 128]");
    return Status::kInvalidArgument;
  }
  if (frame.cols <= 0 || frame.rows <= 0 || frame.width < 2 ||
      frame.height < 2 || (frame.width & 1) != 0 || (frame.height & 1) != 0 ||
      frame.mi_rows <= 0) {
    set_error(error, "gdf_apply_grid", "invalid unit grid geometry");
    return Status::kInvalidArgument;
  }
  if (frame.index.frame_is_intra == 0 && frame.index.num_total_refs > 0 &&
      (frame.index.order_hints == nullptr ||
       frame.index.frame_distance == nullptr)) {
    set_error(error, "gdf_apply_grid",
              "non-intra derivation requires device ref arrays");
    return Status::kInvalidArgument;
  }

  const std::int64_t units =
      static_cast<std::int64_t>(frame.cols) * frame.rows;
  if (units <= 0 || units > kGdfMaxUnits) {
    set_error(error, "gdf_apply_grid", "unit grid exceeds kGdfMaxUnits");
    return Status::kInvalidArgument;
  }
  const int count = static_cast<int>(units);
  const std::int64_t slots =
      units * static_cast<std::int64_t>(kGdfMaxRegionsPerUnit);
  if (slots > kGdfMaxUnits) {
    set_error(error, "gdf_apply_grid", "unit grid slot count exceeds kGdfMaxUnits");
    return Status::kInvalidArgument;
  }
  const int slot_count = static_cast<int>(slots);
  if (buffers.jobs == nullptr || buffers.use_gdf == nullptr ||
      buffers.grid_jobs == nullptr || buffers.max_units < slot_count) {
    set_error(error, "gdf_apply_grid",
              "GdfDeviceBuffers capacity is smaller than cols*rows*" +
                  std::to_string(kGdfMaxRegionsPerUnit) + " (" +
                  std::to_string(slot_count) + ")");
    return Status::kInvalidArgument;
  }

  const unsigned slot_count_u = static_cast<unsigned>(slot_count);
  const unsigned prepare_blocks =
      (slot_count_u + kGdfBlockThreads - 1) / kGdfBlockThreads;
  gdf_prepare_grid_kernel<<<prepare_blocks, kGdfBlockThreads, 0, stream>>>(
      frame, buffers.use_gdf, buffers.grid_jobs, count, slot_count);
  Status status = status_from_cuda(cudaGetLastError());
  if (!is_ok(status)) {
    set_error(error, "gdf_prepare_grid_kernel launch", status_name(status));
    return status;
  }

  const std::size_t shared_bytes = gdf_unit_shared_bytes(frame.unit_size);
  gdf_units_kernel<<<slot_count_u, kGdfBlockThreads, shared_bytes, stream>>>(
      frame.src, buffers.tables, buffers.grid_jobs, slot_count,
      frame.unit_size);
  status = status_from_cuda(cudaGetLastError());
  if (!is_ok(status)) {
    set_error(error, "gdf_units_kernel launch", status_name(status));
  }
  return status;
}

}  // namespace av2enc::cuda::gdf

// AV2 §7.20.4 pixel-classified Wiener filter — CUDA kernels and launchers.
//
// All device memory is bump-allocated from the caller's persistent
// av2enc::cuda::Arena.  This file contains no cudaMalloc / cudaFree /
// cudaMallocHost and no hidden synchronisation.

#include "pc_wiener.h"

#include <cstddef>
#include <cstdint>
#include <string>

#include "encoder/filters/spec_tables/pc_wiener.h"
#include "encoder/filters/spec_tables/pc_wiener_tables.h"

namespace av2enc::cuda::pc_wiener {
namespace {

using av2enc::filters::spec_tables::kAcQlookup;
using av2enc::filters::spec_tables::kPcWienerFilters;
using av2enc::filters::spec_tables::kPcWienerLutToClass;
using av2enc::filters::spec::kPcWienerModeOffsets;
using av2enc::filters::spec::kPcWienerModeWeights;
using av2enc::filters::spec::kPcWienerNormalizer;
using av2enc::filters::spec_tables::kPcWienerSubClassify;
using av2enc::filters::spec_tables::kPcWienerSubClassify2;
using av2enc::filters::spec_tables::kQlookup;

constexpr int kThreads = 256;

int max_q_host(int bit_depth) {
  return bit_depth == 8 ? 255 : (bit_depth == 10 ? 303 : 351);
}

__host__ __device__ __forceinline__ int min_int(int a, int b) {
  return a < b ? a : b;
}

std::size_t round_up(std::size_t bytes, std::size_t alignment) {
  return (bytes + alignment - 1) & ~(alignment - 1);
}

// One arena allocation per normative table, for a stable layout between
// required_device_bytes() and create().
struct TableBytes {
  std::size_t filters;
  std::size_t lut;
  std::size_t sub_classify;
  std::size_t sub_classify2;
  std::size_t qlookup;
  std::size_t ac_qlookup;
  std::size_t taps;
  std::size_t normalizer;
  std::size_t mode_weights;
  std::size_t mode_offsets;
};

constexpr TableBytes kTableBytes = {
    sizeof(std::int16_t) * kFilterSets * kFiltersPerSet * kCoeffsPerFilter,
    sizeof(std::uint8_t) * kLutEntries,
    sizeof(std::uint8_t) * kFilterSets * kClassEntries,
    sizeof(std::uint8_t) * kFilterSets * kSubClassSets * kSubClassEntries,
    sizeof(std::int32_t) * kQlookupEntries,
    sizeof(std::int32_t) * kAcQlookupEntries,
    sizeof(std::int8_t) * kTaps * 2,
    sizeof(std::int32_t) * 5,
    sizeof(std::int32_t) * 4 * 3,
    sizeof(std::int32_t) * 4,
};

std::size_t aligned_tables_bytes() {
  const std::size_t a = Arena::kAlignment;
  return round_up(kTableBytes.filters, a) + round_up(kTableBytes.lut, a) +
         round_up(kTableBytes.sub_classify, a) +
         round_up(kTableBytes.sub_classify2, a) +
         round_up(kTableBytes.qlookup, a) +
         round_up(kTableBytes.ac_qlookup, a) +
         round_up(kTableBytes.taps, a) +
         round_up(kTableBytes.normalizer, a) +
         round_up(kTableBytes.mode_weights, a) +
         round_up(kTableBytes.mode_offsets, a);
}

bool valid_config(const PcWienerConfig& config) {
  if (config.width <= 0 || config.height <= 0) return false;
  if (config.bit_depth != 8 && config.bit_depth != 10 &&
      config.bit_depth != 12) {
    return false;
  }
  if (config.unit_size != 64 && config.unit_size != 128 &&
      config.unit_size != 256 && config.unit_size != 512) {
    return false;
  }
  return true;
}

Status compute_required(const PcWienerConfig& config, std::size_t* out) {
  if (out == nullptr) return Status::kInvalidArgument;
  *out = 0;
  if (!valid_config(config)) return Status::kInvalidArgument;
  const std::size_t mi_cols =
      (static_cast<std::size_t>(config.width) + kMiSize - 1) / kMiSize;
  const std::size_t mi_rows =
      (static_cast<std::size_t>(config.height) + kMiSize - 1) / kMiSize;
  const std::size_t class_bytes = round_up(mi_cols * mi_rows, Arena::kAlignment);
  const std::size_t scratch_bytes =
      round_up(mi_cols * mi_rows, Arena::kAlignment);
  *out = aligned_tables_bytes() + class_bytes + scratch_bytes;
  return Status::kOk;
}

bool valid_source(const PcWienerWorkspace& ws, const PcWienerSource& src,
                  const PcWienerParams& params, std::string* error) {
  if (src.curr == nullptr || src.cdef == nullptr) {
    if (error != nullptr) *error = "PC-Wiener source planes are null";
    return false;
  }
  if (src.curr_stride <= 0 || src.cdef_stride <= 0) {
    if (error != nullptr) *error = "PC-Wiener source strides are invalid";
    return false;
  }
  if (src.luma_start_x < 0 || src.luma_end_x < src.luma_start_x ||
      src.luma_start_y < 0 || src.luma_end_y < src.luma_start_y) {
    if (error != nullptr) *error = "PC-Wiener luma extents are invalid";
    return false;
  }
  if (src.curr_stride < src.luma_end_x + 1 ||
      src.cdef_stride < src.luma_end_x + 1) {
    if (error != nullptr) *error = "PC-Wiener source stride is too small";
    return false;
  }
  if (src.stripe_start_y > src.stripe_end_y ||
      src.stripe_start_y < src.luma_start_y ||
      src.stripe_end_y > src.luma_end_y) {
    if (error != nullptr) *error = "PC-Wiener stripe bounds are invalid";
    return false;
  }
  if (params.bit_depth != ws.config().bit_depth) {
    if (error != nullptr) *error = "PC-Wiener bit depth does not match config";
    return false;
  }
  if (params.base_q_idx < 0 || params.base_q_idx > max_q_host(params.bit_depth)) {
    if (error != nullptr) *error = "PC-Wiener base_q_idx is out of range";
    return false;
  }
  if (params.mi_col_end < 1 || params.mi_col_end > ws.mi_cols()) {
    if (error != nullptr) *error = "PC-Wiener mi_col_end is out of range";
    return false;
  }
  if (params.mi_row_start < 0 || params.mi_row_end <= params.mi_row_start ||
      params.mi_row_end > ws.mi_rows()) {
    if (error != nullptr) *error = "PC-Wiener tile MI rows are out of range";
    return false;
  }
  if (params.lr_tx_skip == nullptr || params.lr_tx_skip_stride < params.mi_col_end) {
    if (error != nullptr) *error = "PC-Wiener LrTxSkip grid is invalid";
    return false;
  }
  return true;
}

bool valid_output(const PcWienerWorkspace& ws, std::uint16_t* lr,
                  int lr_stride, std::string* error) {
  if (lr == nullptr || lr_stride < ws.config().width) {
    if (error != nullptr) *error = "PC-Wiener LrFrame output is invalid";
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__global__ void classify_grid_kernel(PcWienerTables tables, PcWienerSource src,
                                     PcWienerParams params,
                                     std::uint8_t* filter_class,
                                     int filter_class_stride, int mi_cols,
                                     int height) {
  const int col =
      static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int row = static_cast<int>(blockIdx.y);
  if (col >= mi_cols) return;
  const int x = col * kMiSize;
  const int y = row * kMiSize;
  int stripe_start = 0;
  int stripe_end = 0;
  pcw_stripe_bounds(y, height, &stripe_start, &stripe_end);
  PcWienerSource local = src;
  local.stripe_start_y = stripe_start;
  local.stripe_end_y = stripe_end;
  filter_class[static_cast<std::ptrdiff_t>(row) * filter_class_stride + col] =
      static_cast<std::uint8_t>(pcw_class(tables, local, params, x, y));
}

__global__ void classify_block_kernel(PcWienerTables tables, PcWienerSource src,
                                      PcWienerParams params, int x, int y,
                                      std::uint8_t* filter_class,
                                      int filter_class_stride) {
  pcw_classify_block(tables, src, params, x, y, filter_class,
                     filter_class_stride);
}

__global__ void filter_grid_kernel(PcWienerTables tables, PcWienerSource src,
                                   PcWienerParams params,
                                   const std::uint8_t* class_grid,
                                   int class_stride, std::uint16_t* lr,
                                   int lr_stride, int mi_cols, int width,
                                   int height) {
  const int col =
      static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int row = static_cast<int>(blockIdx.y);
  if (col >= mi_cols) return;
  const int x = col * kMiSize;
  const int y = row * kMiSize;
  const int w = min_int(kMiSize, width - x);
  const int h = min_int(kMiSize, height - y);
  if (w <= 0 || h <= 0) return;
  int stripe_start = 0;
  int stripe_end = 0;
  pcw_stripe_bounds(y, height, &stripe_start, &stripe_end);
  PcWienerSource local = src;
  local.stripe_start_y = stripe_start;
  local.stripe_end_y = stripe_end;
  const int cls =
      class_grid[static_cast<std::ptrdiff_t>(row) * class_stride + col];
  pcw_filter_block_class(tables, local, params, x, y, w, h, cls, lr, lr_stride);
}

__global__ void units_active_kernel(const std::uint8_t* unit_type,
                                    int unit_size, int unit_rows,
                                    int unit_cols, std::uint8_t* active,
                                    int mi_cols) {
  const int col =
      static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int row = static_cast<int>(blockIdx.y);
  if (col >= mi_cols) return;
  const int unit = pcw_unit_index(row, col, unit_size, unit_rows, unit_cols);
  active[static_cast<std::ptrdiff_t>(row) * mi_cols + col] =
      static_cast<std::uint8_t>(unit_type[unit] == 1 ? 1 : 0);
}

__global__ void filter_units_kernel(PcWienerTables tables, PcWienerSource src,
                                    PcWienerParams params,
                                    const std::uint8_t* class_grid,
                                    int class_stride,
                                    const std::uint8_t* active,
                                    std::uint16_t* lr, int lr_stride,
                                    int mi_cols, int width, int height) {
  const int col =
      static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int row = static_cast<int>(blockIdx.y);
  if (col >= mi_cols) return;
  if (active[static_cast<std::ptrdiff_t>(row) * class_stride + col] == 0) return;
  const int x = col * kMiSize;
  const int y = row * kMiSize;
  const int w = min_int(kMiSize, width - x);
  const int h = min_int(kMiSize, height - y);
  if (w <= 0 || h <= 0) return;
  int stripe_start = 0;
  int stripe_end = 0;
  pcw_stripe_bounds(y, height, &stripe_start, &stripe_end);
  PcWienerSource local = src;
  local.stripe_start_y = stripe_start;
  local.stripe_end_y = stripe_end;
  const int cls =
      class_grid[static_cast<std::ptrdiff_t>(row) * class_stride + col];
  pcw_filter_block_class(tables, local, params, x, y, w, h, cls, lr, lr_stride);
}

__global__ void filter_block_kernel(PcWienerTables tables, PcWienerSource src,
                                    PcWienerParams params, int x, int y, int w,
                                    int h, std::uint16_t* lr, int lr_stride) {
  pcw_filter_block(tables, src, params, x, y, w, h, lr, lr_stride);
}

__global__ void filter_class_block_kernel(PcWienerTables tables,
                                          PcWienerSource src,
                                          PcWienerParams params, int x, int y,
                                          int w, int h, int cls,
                                          std::uint16_t* lr, int lr_stride) {
  pcw_filter_block_class(tables, src, params, x, y, w, h, cls, lr, lr_stride);
}

__global__ void class_filter_lut_kernel(PcWienerTables tables, int filter_set,
                                        std::uint8_t* out) {
  const int i =
      static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= kClassEntries) return;
  out[i] = static_cast<std::uint8_t>(
      pcw_filter_index(tables, filter_set, i));
}

__global__ void subclass_lookup_kernel(PcWienerTables tables, int filter_set,
                                       int num_classes_idx, std::uint8_t* out) {
  const int i =
      static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= kSubClassEntries) return;
  out[i] = static_cast<std::uint8_t>(
      pcw_subclass(tables, filter_set, num_classes_idx, i));
}

// ---------------------------------------------------------------------------
// Launch helpers
// ---------------------------------------------------------------------------

Status first_error() { return status_from_cuda(cudaGetLastError()); }

}  // namespace

void PcWienerWorkspace::clear() noexcept {
  tables_ = PcWienerTables{};
  class_grid_ = nullptr;
  unit_active_ = nullptr;
  arena_ = nullptr;
  persistent_bytes_ = 0;
  scratch_bytes_ = 0;
}

// ---------------------------------------------------------------------------
// Sizing
// ---------------------------------------------------------------------------

Status pc_wiener_required_device_bytes(const PcWienerConfig& config,
                                       std::size_t* out) {
  return compute_required(config, out);
}

std::size_t pc_wiener_required_device_bytes(const PcWienerConfig& config) {
  std::size_t bytes = 0;
  return is_ok(compute_required(config, &bytes)) ? bytes : 0;
}

// ---------------------------------------------------------------------------
// Workspace
// ---------------------------------------------------------------------------

Status PcWienerWorkspace::create(Arena& arena, const PcWienerConfig& config,
                                 std::string* error) {
  reset();
  config_ = PcWienerConfig{};
  mi_cols_ = 0;
  mi_rows_ = 0;
  unit_rows_ = 0;
  unit_cols_ = 0;

  std::size_t required = 0;
  Status status = compute_required(config, &required);
  if (!is_ok(status)) {
    if (error != nullptr) *error = "invalid PC-Wiener config";
    required_bytes_ = 0;
    return status;
  }
  required_bytes_ = required;
  if (!arena.initialized()) {
    if (error != nullptr) *error = "PC-Wiener arena is not initialized";
    return Status::kNotInitialized;
  }
  if (arena.remaining() < required_bytes_) {
    if (error != nullptr) {
      *error = "PC-Wiener arena too small: need " +
               std::to_string(required_bytes_) + " bytes, have " +
               std::to_string(arena.remaining());
    }
    return Status::kArenaExhausted;
  }

  const std::uint64_t start = arena.mark();
  const std::uint64_t start_used = arena.used();
  const std::size_t mi_cols =
      (static_cast<std::size_t>(config.width) + kMiSize - 1) / kMiSize;
  const std::size_t mi_rows =
      (static_cast<std::size_t>(config.height) + kMiSize - 1) / kMiSize;

  auto allocate = [&](std::size_t bytes, void** out) -> Status {
    void* pointer = nullptr;
    const Status allocated = arena.allocate_checked(bytes, &pointer);
    if (!is_ok(allocated)) return allocated;
    *out = pointer;
    return Status::kOk;
  };

  void* filters = nullptr;
  void* lut = nullptr;
  void* sub1 = nullptr;
  void* sub2 = nullptr;
  void* qlookup = nullptr;
  void* acq = nullptr;
  void* taps = nullptr;
  void* normalizer = nullptr;
  void* weights = nullptr;
  void* offsets = nullptr;
  void* class_grid = nullptr;
  void* active = nullptr;

  status = allocate(kTableBytes.filters, &filters);
  if (is_ok(status)) status = allocate(kTableBytes.lut, &lut);
  if (is_ok(status)) status = allocate(kTableBytes.sub_classify, &sub1);
  if (is_ok(status)) status = allocate(kTableBytes.sub_classify2, &sub2);
  if (is_ok(status)) status = allocate(kTableBytes.qlookup, &qlookup);
  if (is_ok(status)) status = allocate(kTableBytes.ac_qlookup, &acq);
  if (is_ok(status)) status = allocate(kTableBytes.taps, &taps);
  if (is_ok(status)) status = allocate(kTableBytes.normalizer, &normalizer);
  if (is_ok(status)) status = allocate(kTableBytes.mode_weights, &weights);
  if (is_ok(status)) status = allocate(kTableBytes.mode_offsets, &offsets);
  if (is_ok(status)) status = allocate(mi_cols * mi_rows, &class_grid);
  if (!is_ok(status)) {
    arena.rewind(start);
    clear();
    if (error != nullptr) {
      *error = "PC-Wiener arena allocation failed: " +
               std::string(status_name(status));
    }
    return status;
  }
  persistent_bytes_ = static_cast<std::size_t>(arena.used() - start_used);

  status = allocate(mi_cols * mi_rows, &active);
  if (!is_ok(status)) {
    arena.rewind(start);
    clear();
    if (error != nullptr) {
      *error = "PC-Wiener arena allocation failed: " +
               std::string(status_name(status));
    }
    return status;
  }
  scratch_bytes_ = static_cast<std::size_t>(arena.used() - start_used) -
                   persistent_bytes_;

  std::int8_t tap_data[kTaps * 2];
  for (int i = 0; i < kTaps; ++i) {
    tap_data[i * 2 + 0] = av2enc::filters::spec::kPcWienerConfig[i].dy;
    tap_data[i * 2 + 1] = av2enc::filters::spec::kPcWienerConfig[i].dx;
  }

  struct Upload {
    void* dst;
    const void* src;
    std::size_t bytes;
  };
  const Upload uploads[] = {
      {filters, kPcWienerFilters, kTableBytes.filters},
      {lut, kPcWienerLutToClass, kTableBytes.lut},
      {sub1, kPcWienerSubClassify, kTableBytes.sub_classify},
      {sub2, kPcWienerSubClassify2, kTableBytes.sub_classify2},
      {qlookup, kQlookup, kTableBytes.qlookup},
      {acq, kAcQlookup, kTableBytes.ac_qlookup},
      {taps, tap_data, kTableBytes.taps},
      {normalizer, kPcWienerNormalizer, kTableBytes.normalizer},
      {weights, kPcWienerModeWeights, kTableBytes.mode_weights},
      {offsets, kPcWienerModeOffsets, kTableBytes.mode_offsets},
  };
  for (const Upload& upload : uploads) {
    const cudaError_t err = cudaMemcpy(upload.dst, upload.src, upload.bytes,
                                       cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      arena.rewind(start);
      clear();
      if (error != nullptr) {
        *error = std::string("PC-Wiener table upload failed: ") +
                 cudaGetErrorString(err);
      }
      return status_from_cuda(err);
    }
  }

  config_ = config;
  mi_cols_ = static_cast<int>(mi_cols);
  mi_rows_ = static_cast<int>(mi_rows);
  unit_rows_ = static_cast<int>(
      (static_cast<std::size_t>(config.height) +
       static_cast<std::size_t>(config.unit_size) / 2) /
      static_cast<std::size_t>(config.unit_size));
  if (unit_rows_ < 1) unit_rows_ = 1;
  unit_cols_ = static_cast<int>(
      (static_cast<std::size_t>(config.width) +
       static_cast<std::size_t>(config.unit_size) / 2) /
      static_cast<std::size_t>(config.unit_size));
  if (unit_cols_ < 1) unit_cols_ = 1;

  tables_.filters = static_cast<const std::int16_t*>(filters);
  tables_.lut_to_class = static_cast<const std::uint8_t*>(lut);
  tables_.sub_classify = static_cast<const std::uint8_t*>(sub1);
  tables_.sub_classify2 = static_cast<const std::uint8_t*>(sub2);
  tables_.qlookup = static_cast<const std::int32_t*>(qlookup);
  tables_.ac_qlookup = static_cast<const std::int32_t*>(acq);
  tables_.taps = static_cast<const std::int8_t*>(taps);
  tables_.normalizer = static_cast<const std::int32_t*>(normalizer);
  tables_.mode_weights = static_cast<const std::int32_t*>(weights);
  tables_.mode_offsets = static_cast<const std::int32_t*>(offsets);
  class_grid_ = static_cast<std::uint8_t*>(class_grid);
  unit_active_ = static_cast<std::uint8_t*>(active);
  arena_ = &arena;

  if (error != nullptr) error->clear();
  return Status::kOk;
}

void PcWienerWorkspace::reset() noexcept {
  clear();
  required_bytes_ = 0;
  config_ = PcWienerConfig{};
  mi_cols_ = 0;
  mi_rows_ = 0;
  unit_rows_ = 0;
  unit_cols_ = 0;
}

// ---------------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------------

Status pc_wiener_classify(PcWienerWorkspace& ws, const PcWienerSource& src,
                          const PcWienerParams& params, cudaStream_t stream) {
  if (!ws.initialized()) return Status::kNotInitialized;
  if (!valid_source(ws, src, params, nullptr)) return Status::kInvalidArgument;
  const dim3 block(static_cast<unsigned>(kThreads));
  const dim3 grid(
      static_cast<unsigned>((ws.mi_cols() + kThreads - 1) / kThreads),
      static_cast<unsigned>(ws.mi_rows()));
  classify_grid_kernel<<<grid, block, 0, stream>>>(
      ws.tables(), src, params, ws.class_grid(), ws.class_stride(), ws.mi_cols(),
      ws.config().height);
  return first_error();
}

Status pc_wiener_classify_block(PcWienerWorkspace& ws, const PcWienerSource& src,
                                const PcWienerParams& params, int x, int y,
                                std::uint8_t* filter_class,
                                int filter_class_stride, cudaStream_t stream) {
  if (!ws.initialized()) return Status::kNotInitialized;
  if (!valid_source(ws, src, params, nullptr)) return Status::kInvalidArgument;
  if (filter_class == nullptr || filter_class_stride <= 0 || x < 0 || y < 0 ||
      x >= ws.config().width || y >= ws.config().height) {
    return Status::kInvalidArgument;
  }
  classify_block_kernel<<<1, 1, 0, stream>>>(ws.tables(), src, params, x, y,
                                             filter_class,
                                             filter_class_stride);
  return first_error();
}

Status pc_wiener_filter_grid(PcWienerWorkspace& ws, const PcWienerSource& src,
                             const PcWienerParams& params,
                             const std::uint8_t* class_grid, std::uint16_t* lr,
                             int lr_stride, cudaStream_t stream) {
  if (!ws.initialized()) return Status::kNotInitialized;
  if (!valid_source(ws, src, params, nullptr)) return Status::kInvalidArgument;
  if (!valid_output(ws, lr, lr_stride, nullptr)) {
    return Status::kInvalidArgument;
  }
  const std::uint8_t* classes =
      class_grid != nullptr ? class_grid : ws.class_grid();
  const cudaError_t copy = cudaMemcpy2DAsync(
      lr, static_cast<std::size_t>(lr_stride) * sizeof(std::uint16_t),
      src.cdef, static_cast<std::size_t>(src.cdef_stride) * sizeof(std::uint16_t),
      static_cast<std::size_t>(ws.config().width) * sizeof(std::uint16_t),
      static_cast<std::size_t>(ws.config().height), cudaMemcpyDeviceToDevice,
      stream);
  if (copy != cudaSuccess) return status_from_cuda(copy);

  const dim3 block(static_cast<unsigned>(kThreads));
  const dim3 grid(
      static_cast<unsigned>((ws.mi_cols() + kThreads - 1) / kThreads),
      static_cast<unsigned>(ws.mi_rows()));
  filter_grid_kernel<<<grid, block, 0, stream>>>(
      ws.tables(), src, params, classes, ws.class_stride(), lr, lr_stride,
      ws.mi_cols(), ws.config().width, ws.config().height);
  return first_error();
}

Status pc_wiener_filter_units(PcWienerWorkspace& ws, const PcWienerSource& src,
                              const PcWienerParams& params,
                              const std::uint8_t* class_grid,
                              const std::uint8_t* unit_type,
                              std::uint16_t* lr, int lr_stride,
                              cudaStream_t stream) {
  if (!ws.initialized()) return Status::kNotInitialized;
  if (!valid_source(ws, src, params, nullptr)) return Status::kInvalidArgument;
  if (!valid_output(ws, lr, lr_stride, nullptr)) {
    return Status::kInvalidArgument;
  }
  if (unit_type == nullptr) return Status::kInvalidArgument;

  const std::uint8_t* classes =
      class_grid != nullptr ? class_grid : ws.class_grid();
  const cudaError_t copy = cudaMemcpy2DAsync(
      lr, static_cast<std::size_t>(lr_stride) * sizeof(std::uint16_t),
      src.cdef, static_cast<std::size_t>(src.cdef_stride) * sizeof(std::uint16_t),
      static_cast<std::size_t>(ws.config().width) * sizeof(std::uint16_t),
      static_cast<std::size_t>(ws.config().height), cudaMemcpyDeviceToDevice,
      stream);
  if (copy != cudaSuccess) return status_from_cuda(copy);

  const dim3 block(static_cast<unsigned>(kThreads));
  const dim3 grid(
      static_cast<unsigned>((ws.mi_cols() + kThreads - 1) / kThreads),
      static_cast<unsigned>(ws.mi_rows()));
  units_active_kernel<<<grid, block, 0, stream>>>(
      unit_type, ws.config().unit_size, ws.unit_rows(), ws.unit_cols(),
      ws.unit_active(), ws.class_stride());
  Status status = first_error();
  if (!is_ok(status)) return status;

  filter_units_kernel<<<grid, block, 0, stream>>>(
      ws.tables(), src, params, classes, ws.class_stride(), ws.unit_active(), lr,
      lr_stride, ws.mi_cols(), ws.config().width, ws.config().height);
  return first_error();
}

Status pc_wiener_filter_block(PcWienerWorkspace& ws, const PcWienerSource& src,
                              const PcWienerParams& params, int x, int y, int w,
                              int h, std::uint16_t* lr, int lr_stride,
                              cudaStream_t stream) {
  if (!ws.initialized()) return Status::kNotInitialized;
  if (!valid_source(ws, src, params, nullptr)) return Status::kInvalidArgument;
  if (!valid_output(ws, lr, lr_stride, nullptr)) {
    return Status::kInvalidArgument;
  }
  if (w <= 0 || h <= 0 || x < 0 || y < 0 || x + w > ws.config().width ||
      y + h > ws.config().height) {
    return Status::kInvalidArgument;
  }
  filter_block_kernel<<<1, 1, 0, stream>>>(ws.tables(), src, params, x, y, w, h,
                                           lr, lr_stride);
  return first_error();
}

Status pc_wiener_filter_class_block(PcWienerWorkspace& ws,
                                    const PcWienerSource& src,
                                    const PcWienerParams& params, int x, int y,
                                    int w, int h, int cls, std::uint16_t* lr,
                                    int lr_stride, cudaStream_t stream) {
  if (!ws.initialized()) return Status::kNotInitialized;
  if (!valid_source(ws, src, params, nullptr)) return Status::kInvalidArgument;
  if (!valid_output(ws, lr, lr_stride, nullptr)) {
    return Status::kInvalidArgument;
  }
  if (w <= 0 || h <= 0 || x < 0 || y < 0 || x + w > ws.config().width ||
      y + h > ws.config().height || cls < 0 || cls >= kClassEntries) {
    return Status::kInvalidArgument;
  }
  filter_class_block_kernel<<<1, 1, 0, stream>>>(
      ws.tables(), src, params, x, y, w, h, cls, lr, lr_stride);
  return first_error();
}

Status pc_wiener_class_filter_lut(PcWienerWorkspace& ws, int filter_set,
                                  std::uint8_t* out, cudaStream_t stream) {
  if (!ws.initialized()) return Status::kNotInitialized;
  if (out == nullptr || filter_set < 0 || filter_set >= kFilterSets) {
    return Status::kInvalidArgument;
  }
  class_filter_lut_kernel<<<1, kThreads, 0, stream>>>(ws.tables(), filter_set,
                                                     out);
  return first_error();
}

Status pc_wiener_subclass_lookup(PcWienerWorkspace& ws, int filter_set,
                                 int num_classes_idx, std::uint8_t* out,
                                 cudaStream_t stream) {
  if (!ws.initialized()) return Status::kNotInitialized;
  if (out == nullptr || filter_set < 0 || filter_set >= kFilterSets ||
      num_classes_idx < 0 || num_classes_idx >= kSubClassSets) {
    return Status::kInvalidArgument;
  }
  subclass_lookup_kernel<<<1, kThreads, 0, stream>>>(
      ws.tables(), filter_set, num_classes_idx, out);
  return first_error();
}

}  // namespace av2enc::cuda::pc_wiener

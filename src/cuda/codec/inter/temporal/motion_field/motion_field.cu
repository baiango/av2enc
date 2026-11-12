// AV2 v1.0.0 temporal motion field — CUDA kernels and arena-backed launchers.
//
// Kernel placement and the ordering argument for each stage are documented in
// motion_field_device.cuh.  Launchers never allocate, copy or synchronise; all
// device buffers belong to the single persistent av2enc::cuda::Arena and are
// reserved once by allocate_checked().

#include "cuda/codec/inter/temporal/motion_field/motion_field.h"

#include "cuda/codec/inter/temporal/motion_field/motion_field_device.cuh"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

namespace av2enc::cuda::motion_field {
namespace {

constexpr int kMfThreads = 256;

std::size_t align_up_bytes(std::size_t bytes) noexcept {
  const std::size_t alignment = Arena::kAlignment;
  return (bytes + alignment - 1) & ~(alignment - 1);
}

void set_error(std::string* error, const char* context,
               const std::string& detail) {
  if (error != nullptr) {
    *error = std::string(context) + ": " + detail;
  }
}

MfStatus cuda_status(cudaError_t result, const char* context,
                     std::string* error) {
  const Status status = status_from_cuda(result);
  if (is_ok(status)) {
    return MfStatus::kOk;
  }
  set_error(error, context, status_name(status));
  return MfStatus::kCudaError;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__global__ void mf_zero_kernel(MfField field, MfTraj traj) {
  const std::size_t cells =
      static_cast<std::size_t>(field.w8) * static_cast<std::size_t>(field.h8);
  const std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * static_cast<std::size_t>(blockDim.x);
  for (std::size_t i =
           static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
           static_cast<std::size_t>(threadIdx.x);
       i < cells; i += stride) {
    field.mv[i * 2 + 0] = 0;
    field.mv[i * 2 + 1] = 0;
    field.valid[i] = 0;
    field.offset[i] = 0;
  }
  if (traj.num_refs <= 0 || traj.mv == nullptr) {
    return;
  }
  const std::size_t traj_cells =
      static_cast<std::size_t>(traj.num_refs) * cells;
  for (std::size_t i =
           static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
           static_cast<std::size_t>(threadIdx.x);
       i < traj_cells; i += stride) {
    traj.valid[i] = 0;
    traj.mv[i * 2 + 0] = 0;
    traj.mv[i * 2 + 1] = 0;
  }
  const std::size_t pos_cells =
      static_cast<std::size_t>(3) * static_cast<std::size_t>(traj.num_refs) *
      cells;
  for (std::size_t i =
           static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
           static_cast<std::size_t>(threadIdx.x);
       i < pos_cells; i += stride) {
    traj.pos_valid[i] = 0;
    traj.pos[i * 2 + 0] = 0;
    traj.pos[i * 2 + 1] = 0;
  }
}

__device__ __forceinline__ void mf_atomic_max_int(int* address, int value) {
  atomicMax(address, value);
}

// Pass 1 of §7.22 storage: stamp every covered 8x8 cell with the highest
// descriptor index.  Two 4x4 blocks can share an 8x8 cell (odd origin), so
// this makes the parallel write deterministic and equal to the CPU call order.
__global__ void mf_stamp_blocks_kernel(const MfBlockStoreInput* blocks,
                                       int count, int* stamp) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= count) {
    return;
  }
  const MfBlockStoreInput& in = blocks[index];
  // A disabled block writes nothing in the sequential path, so it must not
  // acquire stamps either (otherwise it would suppress an earlier writer).
  if (in.enable_ref_frame_mvs == 0) {
    return;
  }
  int w8 = 0;
  int h8 = 0;
  int bw4 = 0;
  int bh4 = 0;
  if (!mf_block_geometry(in, &w8, &h8, &bw4, &bh4)) {
    return;
  }
  const int rows = (bh4 + 1) >> 1;
  const int cols = (bw4 + 1) >> 1;
  for (int i8 = 0; i8 < rows; ++i8) {
    const int y8 = (in.r >> 1) + i8;
    if (y8 < 0 || y8 >= h8 || i8 > 7) {
      continue;
    }
    for (int j8 = 0; j8 < cols; ++j8) {
      const int x8 = (in.c >> 1) + j8;
      // Match the entry-helper contract: cells outside the 8x8 sub-block index
      // are not written, so they must not be stamped (last real writer wins).
      if (x8 < 0 || x8 >= w8 || j8 > 7) {
        continue;
      }
      mf_atomic_max_int(&stamp[mf_index(w8, y8, x8)], index);
    }
  }
}

// Pass 2: only the highest-index descriptor covering a cell writes it.
__global__ void mf_store_blocks_kernel(const MfBlockStoreInput* blocks, int count,
                                       MfSavedField saved, const int* stamp) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= count) {
    return;
  }
  mf_store_block(blocks[index], saved.ref, saved.mv, stamp, index);
}

__global__ void mf_project_job_kernel(const MfProjectionParams* jobs,
                                      int index) {
  // One thread per job: the §7.9 replace rule is order-dependent, so the cell
  // sweep must stay in the CPU's row-major source-cell order.  The job array is
  // device memory; the host only advances the pointer between launches.
  mf_project_job(jobs[index]);
}

__global__ void mf_traj_fill_kernel(MfTraj traj, int fill_refs, int sb_64x64,
                                    int proj_step) {
  const int cols = (traj.w8 + 1) >> 1;
  const int rows = (traj.h8 + 1) >> 1;
  const long long total = static_cast<long long>(fill_refs) * rows * cols;
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
       i < total; i += stride) {
    const int anchor = static_cast<int>(i % (static_cast<long long>(rows) * cols));
    const int rf = static_cast<int>(i / (static_cast<long long>(rows) * cols));
    const int x8 = (anchor % cols) * 2;
    const int y8 = (anchor / cols) * 2;
    if (y8 >= traj.h8 || x8 >= traj.w8) {
      continue;
    }
    mf_traj_fill_cell(&traj, rf, y8, x8, traj.w8, traj.h8, sb_64x64, proj_step);
  }
}

__global__ void mf_merge_stacks_kernel(const MfStackBlockInput* blocks,
                                       int count, const RefMvBank* bank,
                                       StackMv* stack_out, int* num_found_out,
                                       int* prune_out) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= count) {
    return;
  }
  StackMv stack[t::kMaxRefMvStackSize];
  int num_found = 0;
  int prune = 0;
  mf_merge_block(blocks[index], bank, stack, &num_found, &prune);
  StackMv* dst =
      stack_out + static_cast<std::size_t>(index) * t::kMaxRefMvStackSize;
  for (int i = 0; i < t::kMaxRefMvStackSize; ++i) {
    dst[i] = stack[i];
  }
  num_found_out[index] = num_found;
  prune_out[index] = prune;
}

__global__ void mf_bank_ops_kernel(RefMvBank* bank, const MfBankOp* ops,
                                   int count, const RefMvBankRowSource* rows) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  for (int i = 0; i < count; ++i) {
    const MfBankOp& op = ops[i];
    if (op.kind == MfBankOp::kReset) {
      mf_reset_ref_mv_bank(bank, op.mi_row, op.mi_col, op.sb_size4,
                           op.frame_is_intra != 0, op.top_row != 0,
                           rows != nullptr ? rows + op.row_offset : nullptr,
                           op.row_mi_cols);
    } else if (op.kind == MfBankOp::kCount) {
      mf_update_ref_mv_count(bank, op.mi_row, op.mi_col, op.bw4, op.bh4,
                             op.sb_size4, op.chroma_part != 0);
    } else {
      (void)mf_update_ref_mv_bank(bank, op.mi_row, op.mi_col, op.bw4, op.bh4,
                                  op.sb_size4, op.chroma_part != 0, op.refs,
                                  op.mv, op.cwp_idx,
                                  op.from_within_sb != 0);
    }
  }
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

bool valid_config(const MfConfig& c) noexcept {
  if (c.mi_cols <= 0 || c.mi_rows <= 0 || c.num_refs < 0 ||
      c.num_refs > t::kRefsPerFrame || c.traj_fill_refs < 0 ||
      c.traj_fill_refs > c.num_refs) {
    return false;
  }
  if (c.proj_step < 1 || c.proj_step > 2) {
    return false;
  }
  if (c.max_blocks <= 0 || c.max_blocks > kMfMaxCount ||
      c.max_stack_blocks <= 0 || c.max_stack_blocks > kMfMaxCount ||
      c.max_projection_jobs <= 0 || c.max_projection_jobs > kMfMaxCount ||
      c.max_bank_ops <= 0 || c.max_bank_ops > kMfMaxCount ||
      c.max_bank_rows <= 0 || c.max_bank_rows > kMfMaxCount) {
    return false;
  }
  return true;
}

const char* first_missing_buffer(const MfDeviceBuffers& b) noexcept {
  if (b.field.mv == nullptr) return "field.mv";
  if (b.field.valid == nullptr) return "field.valid";
  if (b.field.offset == nullptr) return "field.offset";
  if (b.saved.mv == nullptr) return "saved.mv";
  if (b.saved.ref == nullptr) return "saved.ref";
  if (b.saved_stamp == nullptr) return "saved_stamp";
  if (b.blocks == nullptr) return "blocks";
  if (b.stack_blocks == nullptr) return "stack_blocks";
  if (b.stack_out == nullptr) return "stack_out";
  if (b.stack_num_found == nullptr) return "stack_num_found";
  if (b.stack_prune_count == nullptr) return "stack_prune_count";
  if (b.projection_jobs == nullptr) return "projection_jobs";
  if (b.bank == nullptr) return "bank";
  if (b.bank_ops == nullptr) return "bank_ops";
  if (b.bank_rows == nullptr) return "bank_rows";
  return nullptr;
}

bool buffers_ready(const MfDeviceBuffers& b) noexcept {
  return first_missing_buffer(b) == nullptr;
}

}  // namespace

// ---------------------------------------------------------------------------
// Sizing and allocation
// ---------------------------------------------------------------------------

std::size_t required_device_bytes(const MfConfig& config) noexcept {
  if (!valid_config(config)) {
    return 0;
  }
  const std::size_t w8 = static_cast<std::size_t>(config.mi_cols >> 1);
  const std::size_t h8 = static_cast<std::size_t>(config.mi_rows >> 1);
  if (w8 == 0 || h8 == 0) {
    return 0;
  }
  const std::size_t cells = w8 * h8;
  std::size_t total = 0;
  const auto add = [&total](std::size_t bytes) {
    total += align_up_bytes(bytes);
  };

  add(cells * 2 * sizeof(std::int32_t));   // field.mv
  add(cells * sizeof(std::uint8_t));       // field.valid
  add(cells * sizeof(std::int32_t));       // field.offset
  add(cells * 2 * 2 * sizeof(std::int16_t));  // saved.mv
  add(cells * 2 * sizeof(std::int8_t));       // saved.ref
  add(cells * sizeof(std::int32_t));          // saved_stamp

  if (config.num_refs > 0) {
    const std::size_t refs = static_cast<std::size_t>(config.num_refs);
    add(refs * cells * 2 * sizeof(std::int32_t));  // traj.mv
    add(refs * cells * sizeof(std::uint8_t));      // traj.valid
    add(3 * refs * cells * 2 * sizeof(std::int16_t));  // traj.pos
    add(3 * refs * cells * sizeof(std::uint8_t));      // traj.pos_valid
  }

  add(static_cast<std::size_t>(config.max_blocks) *
      sizeof(MfBlockStoreInput));
  add(static_cast<std::size_t>(config.max_stack_blocks) *
      sizeof(MfStackBlockInput));
  add(static_cast<std::size_t>(config.max_stack_blocks) *
      t::kMaxRefMvStackSize * sizeof(StackMv));
  add(static_cast<std::size_t>(config.max_stack_blocks) * sizeof(int) * 2);
  add(static_cast<std::size_t>(config.max_projection_jobs) *
      sizeof(MfProjectionParams));
  add(sizeof(RefMvBank));
  add(static_cast<std::size_t>(config.max_bank_ops) * sizeof(MfBankOp));
  add(static_cast<std::size_t>(config.max_bank_rows) *
      sizeof(RefMvBankRowSource));
  return total;
}

const char* mf_status_name(MfStatus status) noexcept {
  switch (status) {
    case MfStatus::kOk:
      return "ok";
    case MfStatus::kInvalidArgument:
      return "invalid argument";
    case MfStatus::kNotInitialized:
      return "not initialized";
    case MfStatus::kArenaTooSmall:
      return "arena too small";
    case MfStatus::kCudaError:
      return "CUDA error";
  }
  return "unknown";
}

MfStatus allocate_checked(Arena& arena, const MfConfig& config,
                          MfDeviceBuffers* out, std::size_t* required_bytes,
                          std::string* error) {
  if (required_bytes != nullptr) {
    *required_bytes = 0;
  }
  if (out == nullptr) {
    set_error(error, "allocate_checked", "null output");
    return MfStatus::kInvalidArgument;
  }
  *out = MfDeviceBuffers{};
  if (!arena.initialized()) {
    set_error(error, "allocate_checked", "arena is not initialized");
    return MfStatus::kNotInitialized;
  }
  if (!valid_config(config)) {
    set_error(error, "allocate_checked", "invalid MfConfig");
    return MfStatus::kInvalidArgument;
  }
  const std::size_t need = required_device_bytes(config);
  if (required_bytes != nullptr) {
    *required_bytes = need;
  }
  if (need == 0) {
    set_error(error, "allocate_checked", "invalid MfConfig");
    return MfStatus::kInvalidArgument;
  }
  if (arena.remaining() < need) {
    set_error(error, "allocate_checked",
              "arena too small: need " + std::to_string(need) +
                  " bytes, have " + std::to_string(arena.remaining()));
    return MfStatus::kArenaTooSmall;
  }

  const std::uint64_t mark = arena.mark();
  const int w8 = config.mi_cols >> 1;
  const int h8 = config.mi_rows >> 1;
  const std::size_t cells =
      static_cast<std::size_t>(w8) * static_cast<std::size_t>(h8);

  MfDeviceBuffers b;
  b.config = config;
  b.field.w8 = w8;
  b.field.h8 = h8;
  b.field.mv = arena.allocate_array<std::int32_t>(cells * 2);
  b.field.valid = arena.allocate_array<std::uint8_t>(cells);
  b.field.offset = arena.allocate_array<std::int32_t>(cells);
  b.saved.w8 = w8;
  b.saved.h8 = h8;
  b.saved.mv = arena.allocate_array<std::int16_t>(cells * 4);
  b.saved.ref = arena.allocate_array<std::int8_t>(cells * 2);
  b.saved_stamp = arena.allocate_array<int>(cells);
  if (config.num_refs > 0) {
    const std::size_t refs = static_cast<std::size_t>(config.num_refs);
    b.traj.w8 = w8;
    b.traj.h8 = h8;
    b.traj.num_refs = config.num_refs;
    b.traj.mv = arena.allocate_array<std::int32_t>(refs * cells * 2);
    b.traj.valid = arena.allocate_array<std::uint8_t>(refs * cells);
    b.traj.pos = arena.allocate_array<std::int16_t>(3 * refs * cells * 2);
    b.traj.pos_valid = arena.allocate_array<std::uint8_t>(3 * refs * cells);
  }
  b.blocks = arena.allocate_array<MfBlockStoreInput>(
      static_cast<std::size_t>(config.max_blocks));
  b.stack_blocks = arena.allocate_array<MfStackBlockInput>(
      static_cast<std::size_t>(config.max_stack_blocks));
  b.stack_out = arena.allocate_array<StackMv>(
      static_cast<std::size_t>(config.max_stack_blocks) *
      t::kMaxRefMvStackSize);
  b.stack_num_found = arena.allocate_array<int>(
      static_cast<std::size_t>(config.max_stack_blocks));
  b.stack_prune_count = arena.allocate_array<int>(
      static_cast<std::size_t>(config.max_stack_blocks));
  b.projection_jobs = arena.allocate_array<MfProjectionParams>(
      static_cast<std::size_t>(config.max_projection_jobs));
  b.bank = arena.allocate_array<RefMvBank>(1);
  b.bank_ops = arena.allocate_array<MfBankOp>(
      static_cast<std::size_t>(config.max_bank_ops));
  b.bank_rows = arena.allocate_array<RefMvBankRowSource>(
      static_cast<std::size_t>(config.max_bank_rows));

  if (!buffers_ready(b)) {
    (void)arena.rewind(mark);
    set_error(error, "allocate_checked",
              "arena exhausted while reserving " + std::to_string(need) +
                  " bytes");
    return MfStatus::kArenaTooSmall;
  }
  *out = b;
  return MfStatus::kOk;
}

std::uint64_t motion_field_mark(const Arena& arena) noexcept {
  return arena.mark();
}

bool motion_field_rewind(Arena& arena, std::uint64_t mark) noexcept {
  return arena.rewind(mark);
}

// ---------------------------------------------------------------------------
// Setup copies
// ---------------------------------------------------------------------------

MfStatus upload_blocks(const MfDeviceBuffers& buffers,
                       const MfBlockStoreInput* host_blocks, int count,
                       cudaStream_t stream, std::string* error) {
  if (buffers.blocks == nullptr) {
    set_error(error, "upload_blocks", "buffers are not initialized");
    return MfStatus::kNotInitialized;
  }
  if (count < 0 || count > buffers.config.max_blocks) {
    set_error(error, "upload_blocks", "block count out of range");
    return MfStatus::kInvalidArgument;
  }
  if (count == 0) {
    return MfStatus::kOk;
  }
  if (host_blocks == nullptr) {
    set_error(error, "upload_blocks", "null host blocks");
    return MfStatus::kInvalidArgument;
  }
  return cuda_status(
      cudaMemcpyAsync(buffers.blocks, host_blocks,
                      static_cast<std::size_t>(count) *
                          sizeof(MfBlockStoreInput),
                      cudaMemcpyHostToDevice, stream),
      "upload_blocks", error);
}

MfStatus upload_stack_blocks(const MfDeviceBuffers& buffers,
                             const MfStackBlockInput* host_blocks, int count,
                             cudaStream_t stream, std::string* error) {
  if (buffers.stack_blocks == nullptr) {
    set_error(error, "upload_stack_blocks", "buffers are not initialized");
    return MfStatus::kNotInitialized;
  }
  if (count < 0 || count > buffers.config.max_stack_blocks) {
    set_error(error, "upload_stack_blocks", "block count out of range");
    return MfStatus::kInvalidArgument;
  }
  if (count == 0) {
    return MfStatus::kOk;
  }
  if (host_blocks == nullptr) {
    set_error(error, "upload_stack_blocks", "null host blocks");
    return MfStatus::kInvalidArgument;
  }
  return cuda_status(
      cudaMemcpyAsync(buffers.stack_blocks, host_blocks,
                      static_cast<std::size_t>(count) *
                          sizeof(MfStackBlockInput),
                      cudaMemcpyHostToDevice, stream),
      "upload_stack_blocks", error);
}

MfStatus upload_projection_jobs(const MfDeviceBuffers& buffers,
                                const MfProjectionParams* host_jobs, int count,
                                cudaStream_t stream, std::string* error) {
  if (buffers.projection_jobs == nullptr) {
    set_error(error, "upload_projection_jobs", "buffers are not initialized");
    return MfStatus::kNotInitialized;
  }
  if (count < 0 || count > buffers.config.max_projection_jobs) {
    set_error(error, "upload_projection_jobs", "job count out of range");
    return MfStatus::kInvalidArgument;
  }
  if (count == 0) {
    return MfStatus::kOk;
  }
  if (host_jobs == nullptr) {
    set_error(error, "upload_projection_jobs", "null host jobs");
    return MfStatus::kInvalidArgument;
  }
  return cuda_status(
      cudaMemcpyAsync(buffers.projection_jobs, host_jobs,
                      static_cast<std::size_t>(count) *
                          sizeof(MfProjectionParams),
                      cudaMemcpyHostToDevice, stream),
      "upload_projection_jobs", error);
}

MfStatus upload_bank(const MfDeviceBuffers& buffers, const RefMvBank& bank,
                     cudaStream_t stream, std::string* error) {
  if (buffers.bank == nullptr) {
    set_error(error, "upload_bank", "buffers are not initialized");
    return MfStatus::kNotInitialized;
  }
  return cuda_status(
      cudaMemcpyAsync(buffers.bank, &bank, sizeof(RefMvBank),
                      cudaMemcpyHostToDevice, stream),
      "upload_bank", error);
}

MfStatus upload_bank_ops(const MfDeviceBuffers& buffers,
                         const MfBankOp* host_ops, int count,
                         const RefMvBankRowSource* host_rows, int row_count,
                         cudaStream_t stream, std::string* error) {
  if (buffers.bank_ops == nullptr || buffers.bank_rows == nullptr) {
    set_error(error, "upload_bank_ops", "buffers are not initialized");
    return MfStatus::kNotInitialized;
  }
  if (count < 0 || count > buffers.config.max_bank_ops || row_count < 0 ||
      row_count > buffers.config.max_bank_rows) {
    set_error(error, "upload_bank_ops", "count out of range");
    return MfStatus::kInvalidArgument;
  }
  if (count == 0) {
    return MfStatus::kOk;
  }
  if (host_ops == nullptr) {
    set_error(error, "upload_bank_ops", "null host ops");
    return MfStatus::kInvalidArgument;
  }
  MfStatus status = cuda_status(
      cudaMemcpyAsync(buffers.bank_ops, host_ops,
                      static_cast<std::size_t>(count) * sizeof(MfBankOp),
                      cudaMemcpyHostToDevice, stream),
      "upload_bank_ops", error);
  if (status != MfStatus::kOk || row_count == 0) {
    return status;
  }
  if (host_rows == nullptr) {
    set_error(error, "upload_bank_ops", "null host rows");
    return MfStatus::kInvalidArgument;
  }
  return cuda_status(
      cudaMemcpyAsync(buffers.bank_rows, host_rows,
                      static_cast<std::size_t>(row_count) *
                          sizeof(RefMvBankRowSource),
                      cudaMemcpyHostToDevice, stream),
      "upload_bank_ops rows", error);
}

// ---------------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------------

MfStatus zero_field(const MfDeviceBuffers& buffers, cudaStream_t stream,
                    std::string* error) {
  if (!buffers_ready(buffers)) {
    set_error(error, "zero_field", "buffers are not initialized");
    return MfStatus::kNotInitialized;
  }
  const std::size_t cells =
      static_cast<std::size_t>(buffers.field.w8) *
      static_cast<std::size_t>(buffers.field.h8);
  const std::size_t blocks_needed = (cells + kMfThreads - 1) / kMfThreads;
  const int blocks = static_cast<int>(blocks_needed > 65535 ? 65535
                                                            : blocks_needed);
  mf_zero_kernel<<<blocks, kMfThreads, 0, stream>>>(buffers.field,
                                                    buffers.traj);
  return cuda_status(cudaGetLastError(), "zero_field launch", error);
}

MfStatus store_blocks(const MfDeviceBuffers& buffers, int block_count,
                      cudaStream_t stream, std::string* error) {
  if (!buffers_ready(buffers)) {
    set_error(error, "store_blocks", "buffers are not initialized");
    return MfStatus::kNotInitialized;
  }
  if (block_count < 0 || block_count > buffers.config.max_blocks) {
    set_error(error, "store_blocks", "block count out of range");
    return MfStatus::kInvalidArgument;
  }
  if (block_count == 0) {
    return MfStatus::kOk;
  }
  const std::size_t cells =
      static_cast<std::size_t>(buffers.field.w8) *
      static_cast<std::size_t>(buffers.field.h8);
  // Reset the ordering stamps to "uncovered" (-1), then run the two passes.
  MfStatus status = cuda_status(
      cudaMemsetAsync(buffers.saved_stamp, 0xFF, cells * sizeof(int), stream),
      "store_blocks stamp reset", error);
  if (status != MfStatus::kOk) {
    return status;
  }
  const int blocks = (block_count + kMfThreads - 1) / kMfThreads;
  mf_stamp_blocks_kernel<<<blocks, kMfThreads, 0, stream>>>(
      buffers.blocks, block_count, buffers.saved_stamp);
  status = cuda_status(cudaGetLastError(), "store_blocks stamp launch", error);
  if (status != MfStatus::kOk) {
    return status;
  }
  mf_store_blocks_kernel<<<blocks, kMfThreads, 0, stream>>>(
      buffers.blocks, block_count, buffers.saved, buffers.saved_stamp);
  return cuda_status(cudaGetLastError(), "store_blocks launch", error);
}

MfStatus project(const MfDeviceBuffers& buffers, int job_count,
                 cudaStream_t stream, std::string* error) {
  const char* missing = first_missing_buffer(buffers);
  if (missing != nullptr) {
    // A null diagnostic here means a buffer was not reserved (for example a
    // newly added plane missing from allocate_checked).  Keep this check before
    // any pointer arithmetic or launch.
    std::fprintf(stderr,
                 "motion_field::project: missing buffer '%s' "
                 "(jobs=%p field.mv=%p saved_stamp=%p)\n",
                 missing, static_cast<const void*>(buffers.projection_jobs),
                 static_cast<const void*>(buffers.field.mv),
                 static_cast<const void*>(buffers.saved_stamp));
    set_error(error, "project",
              std::string("missing device buffer: ") + missing);
    return MfStatus::kNotInitialized;
  }
  if (job_count < 0 || job_count > buffers.config.max_projection_jobs) {
    std::fprintf(stderr,
                 "motion_field::project: job_count=%d max_projection_jobs=%d "
                 "jobs=%p\n",
                 job_count, buffers.config.max_projection_jobs,
                 static_cast<const void*>(buffers.projection_jobs));
    set_error(error, "project", "job count out of range");
    return MfStatus::kInvalidArgument;
  }
  for (int i = 0; i < job_count; ++i) {
    // The job descriptor lives in device memory; pass the base pointer and the
    // index.  Dereferencing buffers.projection_jobs[i] here would be a host
    // read of device memory and segfaults.
    mf_project_job_kernel<<<1, 1, 0, stream>>>(buffers.projection_jobs, i);
    const MfStatus status =
        cuda_status(cudaGetLastError(), "project launch", error);
    if (status != MfStatus::kOk) {
      return status;
    }
  }
  return MfStatus::kOk;
}

MfStatus traj_fill(const MfDeviceBuffers& buffers, cudaStream_t stream,
                   std::string* error) {
  if (!buffers_ready(buffers)) {
    set_error(error, "traj_fill", "buffers are not initialized");
    return MfStatus::kNotInitialized;
  }
  if (buffers.traj.num_refs <= 0 || buffers.traj.mv == nullptr) {
    return MfStatus::kOk;
  }
  if (buffers.field.w8 <= 0 || buffers.field.h8 <= 0) {
    return MfStatus::kInvalidArgument;
  }
  int fill_refs = buffers.config.traj_fill_refs > 0
                      ? buffers.config.traj_fill_refs
                      : buffers.traj.num_refs;
  if (fill_refs > buffers.traj.num_refs) {
    fill_refs = buffers.traj.num_refs;
  }
  if (fill_refs <= 0) {
    return MfStatus::kOk;
  }
  const int cols = (buffers.traj.w8 + 1) >> 1;
  const int rows = (buffers.traj.h8 + 1) >> 1;
  const long long total = static_cast<long long>(fill_refs) * rows * cols;
  const long long blocks_needed = (total + kMfThreads - 1) / kMfThreads;
  const int blocks = static_cast<int>(blocks_needed > 65535 ? 65535
                                                            : blocks_needed);
  mf_traj_fill_kernel<<<blocks, kMfThreads, 0, stream>>>(
      buffers.traj, fill_refs, buffers.config.sb_64x64,
      buffers.config.proj_step);
  return cuda_status(cudaGetLastError(), "traj_fill launch", error);
}

MfStatus merge_stacks(const MfDeviceBuffers& buffers, int block_count,
                      cudaStream_t stream, std::string* error) {
  if (!buffers_ready(buffers)) {
    set_error(error, "merge_stacks", "buffers are not initialized");
    return MfStatus::kNotInitialized;
  }
  if (block_count < 0 || block_count > buffers.config.max_stack_blocks) {
    set_error(error, "merge_stacks", "block count out of range");
    return MfStatus::kInvalidArgument;
  }
  if (block_count == 0) {
    return MfStatus::kOk;
  }
  const int blocks = (block_count + kMfThreads - 1) / kMfThreads;
  mf_merge_stacks_kernel<<<blocks, kMfThreads, 0, stream>>>(
      buffers.stack_blocks, block_count, buffers.bank, buffers.stack_out,
      buffers.stack_num_found, buffers.stack_prune_count);
  return cuda_status(cudaGetLastError(), "merge_stacks launch", error);
}

MfStatus apply_bank_ops(const MfDeviceBuffers& buffers, int op_count,
                        cudaStream_t stream, std::string* error) {
  if (!buffers_ready(buffers)) {
    set_error(error, "apply_bank_ops", "buffers are not initialized");
    return MfStatus::kNotInitialized;
  }
  if (op_count < 0 || op_count > buffers.config.max_bank_ops) {
    set_error(error, "apply_bank_ops", "op count out of range");
    return MfStatus::kInvalidArgument;
  }
  if (op_count == 0) {
    return MfStatus::kOk;
  }
  mf_bank_ops_kernel<<<1, 1, 0, stream>>>(buffers.bank, buffers.bank_ops,
                                          op_count, buffers.bank_rows);
  return cuda_status(cudaGetLastError(), "apply_bank_ops launch", error);
}

}  // namespace av2enc::cuda::motion_field

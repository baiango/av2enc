// AV2 chroma loop restoration — CUDA kernels and launchers.
//
// No cudaMalloc / cudaFree / cudaMallocHost and no hidden synchronisation in
// this translation unit.  Every persistent byte is bump-allocated from the
// caller's `av2enc::cuda::Arena` by allocate_checked().

#include "lr_chroma.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include "lr_chroma_device.cuh"

namespace av2enc::cuda::lr_chroma {
namespace {

constexpr int kThreads = 256;

std::size_t align_up(std::size_t bytes) {
  const std::size_t a = Arena::kAlignment;
  return (bytes + a - 1) & ~(a - 1);
}

bool checked_mul(std::size_t a, std::size_t b, std::size_t* out) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
    return false;
  }
  *out = a * b;
  return true;
}

struct Layout {
  int unit_cols[2];
  int unit_rows[2];
  int unit_count[2];
  std::size_t units[2];
  std::size_t resolved[2];
  std::size_t active[2];
  std::size_t banks;
  std::size_t total;
};

bool compute_layout(const LrChromaConfig& config, Layout* layout) {
  if (layout == nullptr || !config_is_valid(config)) {
    return false;
  }
  *layout = Layout{};
  const int width = plane_width(config);
  const int height = plane_height(config);
  if (width <= 0 || height <= 0) {
    return false;
  }
  for (int chroma = 0; chroma < 2; ++chroma) {
    const int cols = unit_cols_for(config);
    const int rows = unit_rows_for(config);
    if (cols <= 0 || rows <= 0 ||
        cols > std::numeric_limits<int>::max() / rows) {
      return false;
    }
    const std::size_t count = static_cast<std::size_t>(cols) *
                              static_cast<std::size_t>(rows);
    layout->unit_cols[chroma] = cols;
    layout->unit_rows[chroma] = rows;
    layout->unit_count[chroma] = static_cast<int>(count);

    std::size_t bytes = 0;
    if (!checked_mul(count, sizeof(LrChromaUnit), &bytes)) {
      return false;
    }
    layout->units[chroma] = align_up(bytes);
    if (!checked_mul(count, kCoeffStride * sizeof(std::int16_t), &bytes)) {
      return false;
    }
    layout->resolved[chroma] = align_up(bytes);
    layout->active[chroma] = align_up(count);
  }
  layout->banks = align_up(2 * sizeof(LrChromaBankState));
  std::size_t total = 0;
  for (int chroma = 0; chroma < 2; ++chroma) {
    for (std::size_t bytes :
         {layout->units[chroma], layout->resolved[chroma],
          layout->active[chroma]}) {
      if (total > std::numeric_limits<std::size_t>::max() - bytes) {
        return false;
      }
      total += bytes;
    }
  }
  if (total > std::numeric_limits<std::size_t>::max() - layout->banks) {
    return false;
  }
  layout->total = total + layout->banks;
  return true;
}

bool valid_plane(const LrChromaPlane& plane, int width, int height,
                 bool is_dst) {
  if (plane.curr == nullptr || plane.cdef == nullptr) {
    return false;
  }
  if (is_dst && plane.dst == nullptr) {
    return false;
  }
  if (plane.curr_stride < width || plane.cdef_stride < width) {
    return false;
  }
  if (is_dst && plane.dst_stride < width) {
    return false;
  }
  if (height <= 0 || width <= 0) {
    return false;
  }
  if (is_dst) {
    // The §7.20 copy step and the filter both read CdefFrame/CurrFrame while
    // writing LrFrame; either alias is a data race.
    if (plane.dst == plane.cdef || plane.dst == plane.curr) {
      return false;
    }
  }
  return true;
}

bool valid_frame(const LrChromaWorkspace& ws, const LrChromaFrame& frame,
                 std::string* error) {
  const LrChromaConfig& config = ws.config();
  const int width = plane_width(config);
  const int height = plane_height(config);
  const int luma_width = config.mi_cols * kMiSize;
  auto fail = [&](const char* text) {
    if (error != nullptr) *error = text;
    return false;
  };
  if (frame.mi_cols != config.mi_cols || frame.mi_rows != config.mi_rows ||
      frame.sub_x != config.sub_x || frame.sub_y != config.sub_y ||
      frame.bit_depth != config.bit_depth ||
      frame.cfl_ds_filter_index != config.cfl_ds_filter_index) {
    return fail("lr_chroma: frame does not match the workspace config");
  }
  for (int chroma = 0; chroma < 2; ++chroma) {
    if (!valid_plane(frame.chroma[chroma], width, height, true)) {
      return fail("lr_chroma: invalid chroma plane or safe stride");
    }
  }
  if (frame.luma.curr == nullptr || frame.luma.cdef == nullptr ||
      frame.luma.curr_stride < luma_width ||
      frame.luma.cdef_stride < luma_width) {
    return fail("lr_chroma: invalid luma plane");
  }
  if (frame.disable_loopfilters_across_tiles != 0 &&
      (frame.mi_col_start == nullptr || frame.mi_col_end == nullptr ||
       frame.mi_row_start == nullptr || frame.mi_row_end == nullptr)) {
    return fail("lr_chroma: tile clamp needs all four MI grids");
  }
  return true;
}

LrChromaCore make_core(const LrChromaWorkspace& ws,
                       const LrChromaFrame& frame) {
  const LrChromaConfig& config = ws.config();
  LrChromaCore core{};
  core.chroma[0] = frame.chroma[0];
  core.chroma[1] = frame.chroma[1];
  core.luma = frame.luma;
  core.mi_cols = config.mi_cols;
  core.mi_rows = config.mi_rows;
  core.sub_x = config.sub_x;
  core.sub_y = config.sub_y;
  core.bit_depth = config.bit_depth;
  core.cfl_ds_filter_index = config.cfl_ds_filter_index;
  for (int chroma = 0; chroma < 2; ++chroma) {
    core.unit_size[chroma] = config.unit_size;
    core.unit_cols[chroma] = ws.unit_cols(chroma);
    core.unit_rows[chroma] = ws.unit_rows(chroma);
  }
  core.mi_col_start = frame.mi_col_start;
  core.mi_col_end = frame.mi_col_end;
  core.mi_row_start = frame.mi_row_start;
  core.mi_row_end = frame.mi_row_end;
  core.disable_loopfilters_across_tiles =
      frame.disable_loopfilters_across_tiles;
  return core;
}

__global__ void resolve_kernel(const LrChromaUnit* units0,
                               const LrChromaUnit* units1, int count0,
                               int count1, std::int16_t* resolved0,
                               std::int16_t* resolved1, std::uint8_t* active0,
                               std::uint8_t* active1, LrChromaBankState* banks) {
  const int chroma = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (chroma == 0) {
    lrc_resolve_plane(units0, count0, resolved0, active0, &banks[0]);
  } else if (chroma == 1) {
    lrc_resolve_plane(units1, count1, resolved1, active1, &banks[1]);
  }
}

template <typename Sample>
__global__ void apply_kernel(LrChromaCore core, const std::int16_t* resolved0,
                             const std::int16_t* resolved1,
                             const std::uint8_t* active0,
                             const std::uint8_t* active1) {
  const int per = core.mi_cols * core.mi_rows;
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= 2 * per) {
    return;
  }
  const int chroma = idx / per;
  const int local = idx - chroma * per;
  const int row = local / core.mi_cols;
  const int col = local - row * core.mi_cols;
  if (chroma == 0) {
    lrc_process_block<Sample>(core, 0, row, col, resolved0, active0);
  } else {
    lrc_process_block<Sample>(core, 1, row, col, resolved1, active1);
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Sizing and validation
// ---------------------------------------------------------------------------

int plane_width(const LrChromaConfig& config) noexcept {
  if (config.mi_cols <= 0) return 0;
  return lrc_plane_width(config.mi_cols, config.sub_x);
}

int plane_height(const LrChromaConfig& config) noexcept {
  if (config.mi_rows <= 0) return 0;
  return lrc_plane_height(config.mi_rows, config.sub_y);
}

int unit_cols_for(const LrChromaConfig& config) noexcept {
  const int width = plane_width(config);
  if (width <= 0 || config.unit_size <= 0) return 0;
  return lrc_count_units(config.unit_size, width);
}

int unit_rows_for(const LrChromaConfig& config) noexcept {
  const int height = plane_height(config);
  if (height <= 0 || config.unit_size <= 0) return 0;
  return lrc_count_units(config.unit_size, height);
}

bool config_is_valid(const LrChromaConfig& config) noexcept {
  if (config.mi_cols <= 0 || config.mi_rows <= 0) return false;
  if (config.mi_cols > (1 << 20) || config.mi_rows > (1 << 20)) return false;
  if (config.sub_x < 0 || config.sub_x > 1) return false;
  if (config.sub_y < 0 || config.sub_y > 1) return false;
  if (config.bit_depth != 8 && config.bit_depth != 10) return false;
  if (config.unit_size != 32 && config.unit_size != 64 &&
      config.unit_size != 128 && config.unit_size != 256) {
    return false;
  }
  if (config.cfl_ds_filter_index < 0 || config.cfl_ds_filter_index > 3) {
    return false;
  }
  if (config.sb_shift < 0 || config.sb_shift > 8) return false;
  return true;
}

std::size_t required_device_bytes(const LrChromaConfig& config) noexcept {
  Layout layout{};
  if (!compute_layout(config, &layout)) {
    return 0;
  }
  return layout.total;
}

// ---------------------------------------------------------------------------
// Workspace
// ---------------------------------------------------------------------------

void LrChromaWorkspace::clear() noexcept {
  for (int chroma = 0; chroma < 2; ++chroma) {
    units_[chroma] = nullptr;
    resolved_[chroma] = nullptr;
    active_[chroma] = nullptr;
    unit_cols_[chroma] = 0;
    unit_rows_[chroma] = 0;
    unit_count_[chroma] = 0;
  }
  banks_ = nullptr;
  persistent_bytes_ = 0;
}

void LrChromaWorkspace::reset() noexcept {
  clear();
  required_bytes_ = 0;
  config_ = LrChromaConfig{};
}

Status allocate_checked(Arena& arena, const LrChromaConfig& config,
                        LrChromaWorkspace* ws,
                        std::size_t* required_bytes, std::string* error) {
  if (ws == nullptr) {
    return Status::kInvalidArgument;
  }
  ws->reset();
  if (required_bytes != nullptr) {
    *required_bytes = 0;
  }
  Layout layout{};
  if (!compute_layout(config, &layout)) {
    if (error != nullptr) *error = "lr_chroma: invalid configuration";
    return Status::kInvalidArgument;
  }
  if (required_bytes != nullptr) {
    *required_bytes = layout.total;
  }
  if (!arena.initialized()) {
    if (error != nullptr) *error = "lr_chroma: arena is not initialized";
    return Status::kNotInitialized;
  }

  const std::uint64_t mark = arena.mark();
  const std::uint64_t start_used = arena.used();
  auto allocate = [&](std::size_t bytes, void** out) -> Status {
    void* pointer = nullptr;
    const Status status = arena.allocate_checked(bytes, &pointer);
    if (!is_ok(status)) {
      return status;
    }
    *out = pointer;
    return Status::kOk;
  };

  void* units0 = nullptr;
  void* units1 = nullptr;
  void* resolved0 = nullptr;
  void* resolved1 = nullptr;
  void* active0 = nullptr;
  void* active1 = nullptr;
  void* banks = nullptr;

  Status status = allocate(layout.units[0], &units0);
  if (is_ok(status)) status = allocate(layout.units[1], &units1);
  if (is_ok(status)) status = allocate(layout.resolved[0], &resolved0);
  if (is_ok(status)) status = allocate(layout.resolved[1], &resolved1);
  if (is_ok(status)) status = allocate(layout.active[0], &active0);
  if (is_ok(status)) status = allocate(layout.active[1], &active1);
  if (is_ok(status)) status = allocate(layout.banks, &banks);
  if (!is_ok(status)) {
    arena.rewind(mark);
    ws->reset();
    if (error != nullptr) {
      if (status == Status::kArenaExhausted) {
        *error = "lr_chroma: arena too small: need " +
                 std::to_string(layout.total) + " bytes";
      } else {
        *error = std::string("lr_chroma: arena allocation failed: ") +
                 status_name(status);
      }
    }
    return (status == Status::kArenaExhausted) ? kArenaTooSmall : status;
  }

  ws->config_ = config;
  ws->required_bytes_ = layout.total;
  ws->persistent_bytes_ =
      static_cast<std::size_t>(arena.used() - start_used);
  for (int chroma = 0; chroma < 2; ++chroma) {
    ws->unit_cols_[chroma] = layout.unit_cols[chroma];
    ws->unit_rows_[chroma] = layout.unit_rows[chroma];
    ws->unit_count_[chroma] = layout.unit_count[chroma];
  }
  ws->units_[0] = static_cast<LrChromaUnit*>(units0);
  ws->units_[1] = static_cast<LrChromaUnit*>(units1);
  ws->resolved_[0] = static_cast<std::int16_t*>(resolved0);
  ws->resolved_[1] = static_cast<std::int16_t*>(resolved1);
  ws->active_[0] = static_cast<std::uint8_t*>(active0);
  ws->active_[1] = static_cast<std::uint8_t*>(active1);
  ws->banks_ = static_cast<LrChromaBankState*>(banks);
  if (error != nullptr) error->clear();
  return Status::kOk;
}

Status LrChromaWorkspace::create(Arena& arena, const LrChromaConfig& config,
                                 std::string* error) {
  return lr_chroma::allocate_checked(arena, config, this, nullptr, error);
}

// ---------------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------------

Status upload_units(LrChromaWorkspace& ws, int chroma,
                    const LrChromaUnit* host_units, int count,
                    cudaStream_t stream, std::string* error) {
  if (!ws.initialized()) {
    if (error != nullptr) *error = "lr_chroma: workspace is not initialized";
    return Status::kNotInitialized;
  }
  if (chroma < 0 || chroma > 1) {
    if (error != nullptr) *error = "lr_chroma: chroma plane must be 0 or 1";
    return Status::kInvalidArgument;
  }
  if (count != ws.unit_count(chroma)) {
    if (error != nullptr) {
      *error = "lr_chroma: unit count does not match the workspace grid";
    }
    return Status::kInvalidArgument;
  }
  if (count > 0 && host_units == nullptr) {
    if (error != nullptr) *error = "lr_chroma: null host unit array";
    return Status::kInvalidArgument;
  }
  if (count == 0) {
    return Status::kOk;
  }
  const Status status = status_from_cuda(cudaMemcpyAsync(
      ws.units(chroma), host_units,
      static_cast<std::size_t>(count) * sizeof(LrChromaUnit),
      cudaMemcpyHostToDevice, stream));
  if (!is_ok(status) && error != nullptr) {
    *error = "lr_chroma: unit upload failed";
  }
  return status;
}

Status resolve_units(LrChromaWorkspace& ws, cudaStream_t stream,
                     std::string* error) {
  if (!ws.initialized()) {
    if (error != nullptr) *error = "lr_chroma: workspace is not initialized";
    return Status::kNotInitialized;
  }
  resolve_kernel<<<1, 2, 0, stream>>>(
      ws.units(0), ws.units(1), ws.unit_count(0), ws.unit_count(1),
      ws.resolved(0), ws.resolved(1), ws.active(0), ws.active(1),
      ws.bank_state());
  const Status status = status_from_cuda(cudaGetLastError());
  if (!is_ok(status) && error != nullptr) {
    *error = "lr_chroma: resolve launch failed";
  }
  return status;
}

Status apply(const LrChromaWorkspace& ws, const LrChromaFrame& frame,
             cudaStream_t stream, std::string* error) {
  if (!ws.initialized()) {
    if (error != nullptr) *error = "lr_chroma: workspace is not initialized";
    return Status::kNotInitialized;
  }
  if (!valid_frame(ws, frame, error)) {
    return Status::kInvalidArgument;
  }
  const LrChromaCore core = make_core(ws, frame);
  const long long total =
      2LL * static_cast<long long>(core.mi_cols) * core.mi_rows;
  if (total <= 0 || total > std::numeric_limits<int>::max()) {
    if (error != nullptr) *error = "lr_chroma: empty or oversized frame";
    return Status::kInvalidArgument;
  }
  const int total_int = static_cast<int>(total);
  const int blocks = (total_int + kThreads - 1) / kThreads;
  const std::uint8_t* active0 = ws.active(0);
  const std::uint8_t* active1 = ws.active(1);
  if (core.bit_depth == 8) {
    apply_kernel<std::uint8_t><<<blocks, kThreads, 0, stream>>>(
        core, ws.resolved(0), ws.resolved(1), active0, active1);
  } else {
    apply_kernel<std::uint16_t><<<blocks, kThreads, 0, stream>>>(
        core, ws.resolved(0), ws.resolved(1), active0, active1);
  }
  const Status status = status_from_cuda(cudaGetLastError());
  if (!is_ok(status) && error != nullptr) {
    *error = "lr_chroma: apply launch failed";
  }
  return status;
}

}  // namespace av2enc::cuda::lr_chroma

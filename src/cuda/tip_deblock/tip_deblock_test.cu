// CPU-vs-GPU bit-exactness, arena-contract and validation test for the AV2
// §7.16 TIP deblocking filter.
//
// The kernels and the CPU reference share the §7.16 arithmetic through
// tip_deblock_device.h / the §7.17 core; this test is the regression check for
// device codegen, thread mapping, strides, 8/10-bit dispatch and the arena
// footprint contract.
//
// Usage: ./tip_deblock_test [repo_root]

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "cuda/tip_deblock/tip_deblock.h"

#include "encoder/filters/tip_deblock/tip_deblock.h"

namespace ctd = av2enc::cuda::tip_deblock;
namespace cpu = av2enc::tip_deblock;

namespace {

using av2enc::cuda::Arena;
using av2enc::cuda::Status;
using cpu::InterpFilter;
using cpu::TipDeblockOptions;
using cpu::TipDeblockStatus;
using cpu::TipFrameMode;

int g_checks = 0;
int g_failures = 0;
std::uint64_t g_samples = 0;
std::uint64_t g_changed = 0;

#define CHECK(cond, msg)                                                     \
  do {                                                                       \
    ++g_checks;                                                              \
    if (!(cond)) {                                                           \
      ++g_failures;                                                          \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, (msg));   \
    }                                                                        \
  } while (0)

#define CUDA_CHECK(expr, msg)                                                \
  do {                                                                       \
    const cudaError_t err__ = (expr);                                        \
    ++g_checks;                                                              \
    if (err__ != cudaSuccess) {                                              \
      ++g_failures;                                                          \
      std::fprintf(stderr, "FAIL %s:%d: %s (%s)\n", __FILE__, __LINE__,      \
                   (msg), cudaGetErrorString(err__));                        \
    }                                                                        \
  } while (0)

int random_range(std::mt19937& rng, int low, int high) {
  const std::uint32_t span = static_cast<std::uint32_t>(high - low + 1);
  return low + static_cast<int>(rng() % span);
}

// ---------------------------------------------------------------------------
// Host frame (guarded columns; the device buffer holds the visible extent with
// the same stride, so guard columns are uploaded and compared).
// ---------------------------------------------------------------------------
template <typename Sample>
struct HostFrame {
  int mi_cols = 0;
  int mi_rows = 0;
  int guard_cols = 8;
  int guard_rows = 2;
  int stride[3] = {0, 0, 0};
  int height[3] = {0, 0, 0};
  Sample guard = 0;
  std::vector<Sample> data[3];
  std::vector<std::uint16_t> tile_cols;
  std::vector<std::uint16_t> tile_rows;

  int plane_width(int plane) const {
    const int luma = mi_cols * 4;
    return (plane == 0) ? luma : luma / 2;
  }
  int plane_height(int plane) const {
    const int luma = mi_rows * 4;
    return (plane == 0) ? luma : luma / 2;
  }
  Sample* plane(int plane) { return data[plane].data(); }
  const Sample* plane(int plane) const { return data[plane].data(); }
  std::size_t index(int plane, int y, int x) const {
    return static_cast<std::size_t>(y) *
               static_cast<std::size_t>(stride[plane]) +
           static_cast<std::size_t>(x);
  }
};

template <typename Sample>
HostFrame<Sample> make_host_frame(int mi_cols, int mi_rows, Sample guard) {
  HostFrame<Sample> frame;
  frame.mi_cols = mi_cols;
  frame.mi_rows = mi_rows;
  frame.guard = guard;
  frame.stride[0] = mi_cols * 4 + frame.guard_cols;
  frame.stride[1] = mi_cols * 2 + frame.guard_cols;
  frame.stride[2] = frame.stride[1];
  frame.height[0] = mi_rows * 4 + frame.guard_rows;
  frame.height[1] = mi_rows * 2 + frame.guard_rows;
  frame.height[2] = frame.height[1];
  for (int plane = 0; plane < 3; ++plane) {
    frame.data[plane].assign(
        static_cast<std::size_t>(frame.stride[plane]) *
            static_cast<std::size_t>(frame.height[plane]),
        guard);
  }
  return frame;
}

template <typename Sample, typename Fn>
void fill_plane(HostFrame<Sample>& frame, int plane, Fn fn) {
  const int width = frame.plane_width(plane);
  const int height = frame.plane_height(plane);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      frame.plane(plane)[frame.index(plane, y, x)] =
          static_cast<Sample>(fn(x, y));
    }
  }
}

template <typename Sample>
bool guard_intact(const HostFrame<Sample>& frame) {
  for (int plane = 0; plane < 3; ++plane) {
    const int width = frame.plane_width(plane);
    const int height = frame.plane_height(plane);
    for (int y = 0; y < height + frame.guard_rows; ++y) {
      const int start = (y < height) ? width : 0;
      for (int x = start; x < frame.stride[plane]; ++x) {
        if (frame.plane(plane)[frame.index(plane, y, x)] != frame.guard) {
          return false;
        }
      }
    }
  }
  return true;
}

template <typename Sample>
std::uint64_t count_changed(const HostFrame<Sample>& before,
                            const HostFrame<Sample>& after) {
  std::uint64_t changed = 0;
  for (int plane = 0; plane < 3; ++plane) {
    const int width = before.plane_width(plane);
    const int height = before.plane_height(plane);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        if (before.plane(plane)[before.index(plane, y, x)] !=
            after.plane(plane)[after.index(plane, y, x)]) {
          ++changed;
        }
      }
    }
  }
  return changed;
}

template <typename Sample>
bool same_buffer(const HostFrame<Sample>& a, const HostFrame<Sample>& b,
                 int* plane, int* px, int* py) {
  for (int p = 0; p < 3; ++p) {
    const std::size_t count = a.data[p].size();
    for (std::size_t i = 0; i < count; ++i) {
      ++g_samples;
      if (a.data[p][i] != b.data[p][i]) {
        if (plane != nullptr) {
          *plane = p;
          *px = static_cast<int>(i % static_cast<std::size_t>(a.stride[p]));
          *py = static_cast<int>(i / static_cast<std::size_t>(a.stride[p]));
        }
        return false;
      }
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Options.
// ---------------------------------------------------------------------------
TipDeblockOptions base_options(int bit_depth) {
  TipDeblockOptions options;
  options.tip_frame_mode = TipFrameMode::kAsOutput;
  options.apply_deblocking_filter_tip = true;
  options.enable_df_sub_pu = true;
  options.allow_df_sub_pu = true;
  options.disable_loopfilters_across_tiles = false;
  options.base_q_idx = 255;
  options.delta_q_u_ac = 0;
  options.delta_q_v_ac = 0;
  options.base_uv_ac_delta_q = 0;
  options.bit_depth = bit_depth;
  options.enable_tip_refinemv = false;
  options.tip_interp_filter = InterpFilter::kEighttapSharp;
  return options;
}

TipDeblockOptions tip8_options(int bit_depth) {
  TipDeblockOptions options = base_options(bit_depth);
  options.enable_tip_refinemv = true;
  options.tip_interp_filter = InterpFilter::kEighttapSharp;
  return options;
}

cpu::TipDeblockFrame cpu_view(HostFrame<std::uint8_t>& frame) {
  cpu::TipDeblockFrame view{};
  for (int plane = 0; plane < 3; ++plane) {
    view.planes[plane].data = frame.data[plane].data();
    view.planes[plane].stride = frame.stride[plane];
  }
  view.mi_cols = frame.mi_cols;
  view.mi_rows = frame.mi_rows;
  view.subsampling_x = 1;
  view.subsampling_y = 1;
  view.tile_col_starts =
      frame.tile_cols.empty() ? nullptr : frame.tile_cols.data();
  view.num_tile_cols = static_cast<int>(frame.tile_cols.size());
  view.tile_row_starts =
      frame.tile_rows.empty() ? nullptr : frame.tile_rows.data();
  view.num_tile_rows = static_cast<int>(frame.tile_rows.size());
  return view;
}

template <typename Sample>
ctd::TipDeblockFrameView<Sample> device_view(HostFrame<Sample>& frame) {
  ctd::TipDeblockFrameView<Sample> view{};
  for (int plane = 0; plane < 3; ++plane) {
    view.plane[plane] = frame.data[plane].data();
    view.stride[plane] = frame.stride[plane];
  }
  view.mi_cols = frame.mi_cols;
  view.mi_rows = frame.mi_rows;
  view.sub_x = 1;
  view.sub_y = 1;
  view.tile_col_starts =
      frame.tile_cols.empty() ? nullptr : frame.tile_cols.data();
  view.num_tile_cols = static_cast<int>(frame.tile_cols.size());
  view.tile_row_starts =
      frame.tile_rows.empty() ? nullptr : frame.tile_rows.data();
  view.num_tile_rows = static_cast<int>(frame.tile_rows.size());
  return view;
}

// ---------------------------------------------------------------------------
// Arena-backed device frame.
// ---------------------------------------------------------------------------
template <typename Sample>
struct DeviceFrame {
  Arena arena;
  ctd::TipDeblockArenaFrame handle;
  ctd::TipDeblockConfig config;
  std::size_t required = 0;
  bool ready = false;

  ~DeviceFrame() { arena.release(); }

  bool init(const HostFrame<Sample>& host, int bit_depth,
            std::size_t slack_bytes, const char* what,
            bool expect_exhausted = false, std::size_t* reported = nullptr) {
    config = ctd::TipDeblockConfig{};
    config.mi_cols = host.mi_cols;
    config.mi_rows = host.mi_rows;
    config.subsampling_x = 1;
    config.subsampling_y = 1;
    config.bit_depth = bit_depth;
    for (int plane = 0; plane < 3; ++plane) {
      config.stride[plane] = host.stride[plane];
    }
    config.num_tile_cols = static_cast<int>(host.tile_cols.size());
    config.num_tile_rows = static_cast<int>(host.tile_rows.size());
    required = ctd::tip_deblock_required_device_bytes(config);

    if (expect_exhausted) {
      const std::uint64_t capacity =
          static_cast<std::uint64_t>(required) - 1;
      const Status status = arena.init(capacity);
      CHECK(status == Status::kOk, "short arena init");
      std::size_t reported_size = 0;
      const Status allocation =
          ctd::tip_deblock_arena_frame_init(arena, config, &handle,
                                            &reported_size);
      CHECK(allocation == Status::kArenaExhausted,
            "one-byte-short arena is rejected");
      CHECK(reported_size == required, "short arena reports required bytes");
      if (reported != nullptr) {
        *reported = reported_size;
      }
      return false;
    }

    const std::uint64_t capacity =
        static_cast<std::uint64_t>(required) + static_cast<std::uint64_t>(slack_bytes);
    const Status status = arena.init(capacity);
    CHECK(status == Status::kOk, "arena init");
    if (status != Status::kOk) {
      std::fprintf(stderr, "  arena init failed for %s: %s\n", what,
                   av2enc::cuda::status_name(status));
      return false;
    }
    std::size_t reported_size = 0;
    const Status allocation = ctd::tip_deblock_arena_frame_init(
        arena, config, &handle, &reported_size);
    CHECK(allocation == Status::kOk, "arena frame init");
    CHECK(reported_size == required, "arena frame reports required bytes");
    CHECK(handle.persistent_bytes == required,
          "persistent footprint equals required bytes");
    if (allocation != Status::kOk) {
      std::fprintf(stderr, "  arena frame init failed for %s: %s\n", what,
                   av2enc::cuda::status_name(allocation));
      return false;
    }
    ready = upload(host);
    return ready;
  }

  bool upload(const HostFrame<Sample>& host) {
    for (int plane = 0; plane < 3; ++plane) {
      const std::size_t bytes =
          static_cast<std::size_t>(host.stride[plane]) *
          static_cast<std::size_t>(host.plane_height(plane)) * sizeof(Sample);
      const cudaError_t error = cudaMemcpy(
          handle.frame.plane[plane], host.data[plane].data(), bytes,
          cudaMemcpyHostToDevice);
      if (error != cudaSuccess) {
        CHECK(false, "upload plane");
        std::fprintf(stderr, "  upload: %s\n", cudaGetErrorString(error));
        return false;
      }
    }
    if (!host.tile_cols.empty()) {
      const std::size_t bytes =
          host.tile_cols.size() * sizeof(std::uint16_t);
      const cudaError_t error =
          cudaMemcpy(handle.tile_cols, host.tile_cols.data(), bytes,
                     cudaMemcpyHostToDevice);
      CHECK(error == cudaSuccess, "upload tile cols");
    }
    if (!host.tile_rows.empty()) {
      const std::size_t bytes =
          host.tile_rows.size() * sizeof(std::uint16_t);
      const cudaError_t error =
          cudaMemcpy(handle.tile_rows, host.tile_rows.data(), bytes,
                     cudaMemcpyHostToDevice);
      CHECK(error == cudaSuccess, "upload tile rows");
    }
    return true;
  }

  bool download(HostFrame<Sample>& host) {
    for (int plane = 0; plane < 3; ++plane) {
      const std::size_t bytes =
          static_cast<std::size_t>(host.stride[plane]) *
          static_cast<std::size_t>(host.plane_height(plane)) * sizeof(Sample);
      const cudaError_t error =
          cudaMemcpy(host.data[plane].data(), handle.frame.plane[plane], bytes,
                     cudaMemcpyDeviceToHost);
      if (error != cudaSuccess) {
        CHECK(false, "download plane");
        std::fprintf(stderr, "  download: %s\n", cudaGetErrorString(error));
        return false;
      }
    }
    return true;
  }
};

// ---------------------------------------------------------------------------
// Random content.
// ---------------------------------------------------------------------------
template <typename Sample>
void randomize(HostFrame<Sample>& frame, std::mt19937& rng, int max_value) {
  for (int plane = 0; plane < 3; ++plane) {
    fill_plane(frame, plane, [&rng, max_value](int x, int y) {
      const int noise = random_range(rng, 0, max_value);
      const int texture = ((x * 3) ^ (y * 5)) & 31;
      return std::clamp(noise + texture - 16, 0, max_value);
    });
  }
  if (frame.plane_width(0) > 16) {
    const int split = 8 + 8 * random_range(rng, 0, 1);
    fill_plane(frame, 0, [split, max_value](int x, int) {
      return (x < split) ? 24 : max_value - 24;
    });
  }
  if (frame.plane_height(0) > 16) {
    const int split = 8 + 8 * random_range(rng, 0, 1);
    fill_plane(frame, 1, [split, max_value](int, int y) {
      return (y < split) ? max_value - 24 : 24;
    });
  }
}

template <typename Sample>
HostFrame<Sample> make_random_frame(std::mt19937& rng, int max_value) {
  const int mi_cols = 8 + 2 * random_range(rng, 0, 6);
  const int mi_rows = 8 + 2 * random_range(rng, 0, 6);
  HostFrame<Sample> frame = make_host_frame<Sample>(
      mi_cols, mi_rows, static_cast<Sample>(max_value == 255 ? 0xEE : 0x3EE));
  randomize(frame, rng, max_value);
  frame.tile_cols = {0};
  frame.tile_rows = {0};
  if ((rng() % 3) == 0) {
    frame.tile_cols.push_back(static_cast<std::uint16_t>(
        1 + random_range(rng, 0, mi_cols - 2)));
  }
  if ((rng() % 3) == 0) {
    frame.tile_rows.push_back(static_cast<std::uint16_t>(
        1 + random_range(rng, 0, mi_rows - 2)));
  }
  std::sort(frame.tile_cols.begin(), frame.tile_cols.end());
  std::sort(frame.tile_rows.begin(), frame.tile_rows.end());
  return frame;
}

void report_difference(const char* what, int plane, int x, int y) {
  std::fprintf(stderr, "  first diff: %s plane %d (%d,%d)\n", what, plane, x,
               y);
}

// ---------------------------------------------------------------------------
// 1. Arena contract: required size, no net growth, one-byte-short rejection.
// ---------------------------------------------------------------------------
void test_arena_contract() {
  HostFrame<std::uint8_t> host = make_host_frame<std::uint8_t>(20, 12, 0xEEu);
  std::mt19937 rng(1);
  randomize(host, rng, 255);
  host.tile_cols = {0, 5};
  host.tile_rows = {0, 3};

  DeviceFrame<std::uint8_t> device;
  const bool ok = device.init(host, 8, Arena::kAlignment, "arena contract");
  CHECK(ok, "arena frame ready");
  if (!ok) {
    return;
  }
  const std::size_t required = device.required;
  CHECK(required > 0, "required bytes positive");
  CHECK(device.arena.used() == required, "used equals persistent bytes");
  CHECK(device.arena.peak_used() >= device.arena.used(),
        "peak covers persistent use");
  std::cout << "  arena: required " << required << " bytes, persistent "
            << device.handle.persistent_bytes << ", scratch "
            << ctd::tip_deblock_scratch_bytes(device.config) << "\n";

  TipDeblockOptions options = base_options(8);
  options.base_q_idx = 128;
  const std::uint64_t mark = device.arena.mark();
  for (int iteration = 0; iteration < 2; ++iteration) {
    const Status status =
        ctd::tip_deblock_frame_arena(device.handle, options, device.arena);
    CHECK(status == Status::kOk, "arena launch status");
    CHECK(device.arena.used() == mark, "no net arena growth after launch");
  }
  CHECK(device.arena.used() == required, "persistent footprint unchanged");

  DeviceFrame<std::uint8_t> short_device;
  std::size_t reported = 0;
  (void)short_device.init(host, 8, 0, "short arena", true, &reported);
  CHECK(reported == required, "short arena reported the required size");
  short_device.arena.release();
}

// ---------------------------------------------------------------------------
// 2. All base_q_idx, both tip sizes, full frame, CPU vs GPU (8-bit).
// ---------------------------------------------------------------------------
void test_all_q_8bit() {
  std::mt19937 rng(0x7168u);
  HostFrame<std::uint8_t> host = make_host_frame<std::uint8_t>(10, 10, 0xEEu);
  randomize(host, rng, 255);
  host.tile_cols = {0, 5};
  host.tile_rows = {0, 4};

  DeviceFrame<std::uint8_t> device;
  const bool ready = device.init(host, 8, Arena::kAlignment, "all-q");
  CHECK(ready, "all-q device frame ready");
  if (!ready) {
    return;
  }
  std::uint64_t changed = 0;
  for (int tip8 = 0; tip8 <= 1; ++tip8) {
    for (int q = 0; q <= 255; ++q) {
      TipDeblockOptions options = tip8 ? tip8_options(8) : base_options(8);
      options.base_q_idx = q;
      options.delta_q_u_ac = q % 17 - 8;
      options.delta_q_v_ac = 7 - q % 13;
      options.base_uv_ac_delta_q = q % 5 - 2;
      options.disable_loopfilters_across_tiles = ((q & 1) != 0);

      if (!device.upload(host)) {
        return;
      }
      const Status status =
          ctd::tip_deblock_frame_arena(device.handle, options, device.arena);
      CHECK(status == Status::kOk, "all-q GPU status");
      HostFrame<std::uint8_t> gpu = host;
      if (!device.download(gpu)) {
        return;
      }
      HostFrame<std::uint8_t> reference = host;
      cpu::TipDeblockFrame cpu_ref = cpu_view(reference);
      const TipDeblockStatus cpu_status =
          cpu::tip_deblock_frame(cpu_ref, options);
      CHECK(cpu_status == TipDeblockStatus::kOk, "all-q CPU status");
      int plane = 0;
      int px = 0;
      int py = 0;
      const bool same = same_buffer(gpu, reference, &plane, &px, &py);
      CHECK(same, "all-q GPU matches CPU");
      if (!same) {
        report_difference("all-q", plane, px, py);
      }
      CHECK(guard_intact(gpu), "all-q guard columns intact");
      changed += count_changed(host, reference);
    }
  }
  g_changed += changed;
  std::cout << "  all 256 base_q_idx x 2 tips vs CPU (" << changed
            << " filtered samples)\n";
}

// ---------------------------------------------------------------------------
// 3. Plane/pass granularity.
// ---------------------------------------------------------------------------
void test_plane_passes_8bit() {
  std::mt19937 rng(0x7a55u);
  HostFrame<std::uint8_t> host = make_host_frame<std::uint8_t>(10, 8, 0xEEu);
  randomize(host, rng, 255);
  host.tile_cols = {0, 5};
  host.tile_rows = {0, 3};

  DeviceFrame<std::uint8_t> device;
  const bool ready = device.init(host, 8, Arena::kAlignment, "plane/pass");
  CHECK(ready, "plane/pass device frame ready");
  if (!ready) {
    return;
  }
  for (int tip8 = 0; tip8 <= 1; ++tip8) {
    for (int q = 0; q <= 255; ++q) {
      TipDeblockOptions options = tip8 ? tip8_options(8) : base_options(8);
      options.base_q_idx = q;
      HostFrame<std::uint8_t> cpu_frame = host;
      cpu::TipDeblockFrame cpu_ref = cpu_view(cpu_frame);
      device.upload(host);
      for (int plane = 0; plane < 3; ++plane) {
        for (int pass = 0; pass < 2; ++pass) {
          const TipDeblockStatus status =
              cpu::tip_deblock_plane_pass(cpu_ref, options, plane, pass);
          CHECK(status == TipDeblockStatus::kOk, "CPU plane/pass status");
          const Status gpu_status = ctd::tip_deblock_plane_pass_arena(
              device.handle, options, plane, pass, device.arena);
          CHECK(gpu_status == Status::kOk, "GPU plane/pass status");
          HostFrame<std::uint8_t> gpu = host;
          device.download(gpu);
          // Both sides have applied the same prefix of plane/pass calls.
          int bp = 0;
          int bx = 0;
          int by = 0;
          const bool same = same_buffer(gpu, cpu_frame, &bp, &bx, &by);
          CHECK(same, "plane/pass GPU matches CPU");
          if (!same) {
            report_difference("plane/pass", bp, bx, by);
          }
        }
      }
      CHECK(guard_intact(cpu_frame), "CPU plane/pass guards intact");
    }
  }
  std::cout << "  plane/pass granularity (256 q x 2 tips x 3 planes x 2 passes)\n";
}

// All base_q_idx at 10-bit, both tips: GPU vs the shared-core host emulation
// (the CPU reference is 8-bit only; the emulation test validates that host
// emulation against an independent §7.16 transcription).
void test_all_q_10bit() {
  HostFrame<std::uint16_t> host = make_host_frame<std::uint16_t>(10, 16, 0x3EEu);
  std::mt19937 rng(0x10a11u);
  randomize(host, rng, 1023);
  host.tile_cols = {0, 5};
  host.tile_rows = {0, 4};

  DeviceFrame<std::uint16_t> device;
  if (!device.init(host, 10, Arena::kAlignment, "all-q 10-bit")) {
    return;
  }
  std::uint64_t changed = 0;
  for (int tip8 = 0; tip8 <= 1; ++tip8) {
    for (int q = 0; q <= 255; ++q) {
      TipDeblockOptions options = tip8 ? tip8_options(10) : base_options(10);
      options.base_q_idx = q;
      options.delta_q_u_ac = q % 19 - 9;
      options.delta_q_v_ac = 11 - q % 23;
      options.base_uv_ac_delta_q = q % 7 - 3;
      device.upload(host);
      const Status status =
          ctd::tip_deblock_frame_arena(device.handle, options, device.arena);
      CHECK(status == Status::kOk, "all-q 10-bit GPU status");
      HostFrame<std::uint16_t> gpu = host;
      device.download(gpu);
      HostFrame<std::uint16_t> reference = host;
      ctd::tip_frame_view<std::uint16_t, 10>(
          device_view(reference), ctd::tip_deblock_device_options(options));
      int plane = 0;
      int px = 0;
      int py = 0;
      const bool same = same_buffer(gpu, reference, &plane, &px, &py);
      CHECK(same, "all-q 10-bit GPU matches shared core");
      if (!same) {
        report_difference("all-q 10-bit", plane, px, py);
      }
      CHECK(guard_intact(gpu), "all-q 10-bit guards intact");
      changed += count_changed(host, reference);
    }
  }
  g_changed += changed;
  std::cout << "  all 256 base_q_idx x 2 tips, 10-bit (" << changed
            << " filtered samples)\n";
}

// ---------------------------------------------------------------------------
// 4. Random frames, 8- and 10-bit.
// ---------------------------------------------------------------------------
void test_random_8bit() {
  std::mt19937 rng(0x8b17u);
  std::uint64_t changed = 0;
  for (int iteration = 0; iteration < 60; ++iteration) {
    HostFrame<std::uint8_t> host = make_random_frame<std::uint8_t>(rng, 255);
    TipDeblockOptions options =
        (rng() % 2) != 0 ? tip8_options(8) : base_options(8);
    options.base_q_idx = random_range(rng, 0, 255);
    options.delta_q_u_ac = random_range(rng, -32, 32);
    options.delta_q_v_ac = random_range(rng, -32, 32);
    options.base_uv_ac_delta_q = random_range(rng, -16, 16);
    options.disable_loopfilters_across_tiles = (rng() % 2) != 0;

    DeviceFrame<std::uint8_t> device;
    if (!device.init(host, 8, Arena::kAlignment, "random 8-bit")) {
      return;
    }
    const Status status =
        ctd::tip_deblock_frame_arena(device.handle, options, device.arena);
    CHECK(status == Status::kOk, "random 8-bit GPU status");
    HostFrame<std::uint8_t> gpu = host;
    device.download(gpu);
    HostFrame<std::uint8_t> reference = host;
    cpu::TipDeblockFrame cpu_ref = cpu_view(reference);
    cpu::tip_deblock_frame(cpu_ref, options);
    int plane = 0;
    int px = 0;
    int py = 0;
    const bool same = same_buffer(gpu, reference, &plane, &px, &py);
    CHECK(same, "random 8-bit GPU matches CPU");
    if (!same) {
      report_difference("random 8-bit", plane, px, py);
    }
    CHECK(guard_intact(gpu), "random 8-bit guards intact");
    changed += count_changed(host, reference);
  }
  g_changed += changed;
  std::cout << "  random 8-bit frames (60 vs CPU, " << changed
            << " filtered samples)\n";
}

void test_random_10bit() {
  std::mt19937 rng(0x10b17u);
  std::uint64_t changed = 0;
  for (int iteration = 0; iteration < 30; ++iteration) {
    HostFrame<std::uint16_t> host = make_random_frame<std::uint16_t>(rng, 1023);
    TipDeblockOptions options =
        (rng() % 2) != 0 ? tip8_options(10) : base_options(10);
    options.base_q_idx = random_range(rng, 0, 255);
    options.delta_q_u_ac = random_range(rng, -32, 32);
    options.delta_q_v_ac = random_range(rng, -32, 32);
    options.base_uv_ac_delta_q = random_range(rng, -16, 16);
    options.disable_loopfilters_across_tiles = (rng() % 2) != 0;

    DeviceFrame<std::uint16_t> device;
    if (!device.init(host, 10, Arena::kAlignment, "random 10-bit")) {
      return;
    }
    const Status status =
        ctd::tip_deblock_frame_arena(device.handle, options, device.arena);
    CHECK(status == Status::kOk, "random 10-bit GPU status");
    HostFrame<std::uint16_t> gpu = host;
    device.download(gpu);
    HostFrame<std::uint16_t> reference = host;
    ctd::tip_frame_view<std::uint16_t, 10>(device_view(reference),
                                           ctd::tip_deblock_device_options(options));
    int plane = 0;
    int px = 0;
    int py = 0;
    const bool same = same_buffer(gpu, reference, &plane, &px, &py);
    CHECK(same, "random 10-bit GPU matches shared-core host emulation");
    if (!same) {
      report_difference("random 10-bit", plane, px, py);
    }
    CHECK(guard_intact(gpu), "random 10-bit guards intact");
    bool in_range = true;
    for (int p = 0; p < 3; ++p) {
      for (std::uint16_t value : gpu.data[p]) {
        if (value > 1023) {
          in_range = false;
        }
      }
    }
    CHECK(in_range, "10-bit samples stay in [0,1023]");
    changed += count_changed(host, reference);
  }
  g_changed += changed;
  std::cout << "  random 10-bit frames (30 vs shared core, " << changed
            << " filtered samples)\n";
}

// ---------------------------------------------------------------------------
// 5. Real reconstructed frame.
// ---------------------------------------------------------------------------
bool load_y4m_first_frame(const std::string& path,
                          HostFrame<std::uint8_t>* out) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return false;
  }
  std::string line;
  if (!std::getline(in, line) || line.rfind("YUV4MPEG2", 0) != 0) {
    return false;
  }
  std::istringstream header(line);
  std::string token;
  int width = 0;
  int height = 0;
  bool c420 = false;
  while (header >> token) {
    if (token.size() > 1) {
      const char tag = token[0];
      const std::string value = token.substr(1);
      if (tag == 'W') width = std::stoi(value);
      if (tag == 'H') height = std::stoi(value);
      if (tag == 'C' && value == "420") c420 = true;
    }
  }
  if (width <= 0 || height <= 0 || !c420 || (width % 8) != 0 ||
      (height % 8) != 0) {
    return false;
  }
  if (!std::getline(in, line) || line.rfind("FRAME", 0) != 0) {
    return false;
  }
  HostFrame<std::uint8_t> frame =
      make_host_frame<std::uint8_t>(width / 4, height / 4, 0xEEu);
  std::vector<std::uint8_t> row(static_cast<std::size_t>(width));
  for (int y = 0; y < height; ++y) {
    in.read(reinterpret_cast<char*>(row.data()), width);
    if (!in) {
      return false;
    }
    std::memcpy(frame.plane(0) + frame.index(0, y, 0), row.data(),
                static_cast<std::size_t>(width));
  }
  row.assign(static_cast<std::size_t>(width / 2), 0);
  for (int y = 0; y < height / 2; ++y) {
    in.read(reinterpret_cast<char*>(row.data()), width / 2);
    if (!in) {
      return false;
    }
    std::memcpy(frame.plane(1) + frame.index(1, y, 0), row.data(),
                static_cast<std::size_t>(width / 2));
  }
  for (int y = 0; y < height / 2; ++y) {
    in.read(reinterpret_cast<char*>(row.data()), width / 2);
    if (!in) {
      return false;
    }
    std::memcpy(frame.plane(2) + frame.index(2, y, 0), row.data(),
                static_cast<std::size_t>(width / 2));
  }
  *out = std::move(frame);
  return true;
}

void test_real_frame(const std::string& repo_root) {
  HostFrame<std::uint8_t> host;
  if (!load_y4m_first_frame(repo_root + "/decoded.y4m", &host)) {
    std::cout << "  real frame: skipped (decoded.y4m not usable)\n";
    return;
  }
  for (int tip8 = 0; tip8 <= 1; ++tip8) {
    TipDeblockOptions options = tip8 ? tip8_options(8) : base_options(8);
    options.base_q_idx = 128;
    DeviceFrame<std::uint8_t> device;
    if (!device.init(host, 8, Arena::kAlignment, "real frame")) {
      return;
    }
    const Status status =
        ctd::tip_deblock_frame_arena(device.handle, options, device.arena);
    CHECK(status == Status::kOk, "real frame GPU status");
    HostFrame<std::uint8_t> gpu = host;
    device.download(gpu);
    HostFrame<std::uint8_t> reference = host;
    cpu::TipDeblockFrame cpu_ref = cpu_view(reference);
    cpu::tip_deblock_frame(cpu_ref, options);
    int plane = 0;
    int px = 0;
    int py = 0;
    CHECK(same_buffer(gpu, reference, &plane, &px, &py),
          "real frame GPU matches CPU");
    CHECK(guard_intact(gpu), "real frame guards intact");
    std::cout << "  real frame " << (tip8 ? "8x8" : "16x16") << ": "
              << count_changed(host, reference) << " samples changed\n";
  }
}

// ---------------------------------------------------------------------------
// 6. Validation and inactive-frame behaviour.
// ---------------------------------------------------------------------------
void test_validation() {
  HostFrame<std::uint8_t> host = make_host_frame<std::uint8_t>(8, 8, 0xEEu);
  std::mt19937 rng(7);
  randomize(host, rng, 255);
  DeviceFrame<std::uint8_t> device;
  if (!device.init(host, 8, Arena::kAlignment, "validation")) {
    return;
  }
  TipDeblockOptions options = base_options(8);
  options.base_q_idx = 100;

  ctd::TipDeblockDeviceFrame frame = device.handle.frame;
  ctd::TipDeblockDeviceFrame bad = frame;
  bad.plane[1] = nullptr;
  CHECK(ctd::tip_deblock_frame(bad, options) == cudaErrorInvalidValue,
        "null plane rejected");
  bad = frame;
  bad.stride[0] = 4;
  CHECK(ctd::tip_deblock_frame(bad, options) == cudaErrorInvalidValue,
        "short stride rejected");
  bad = frame;
  bad.mi_cols = 7;
  CHECK(ctd::tip_deblock_frame(bad, options) == cudaErrorInvalidValue,
        "odd MI grid rejected");
  bad = frame;
  bad.subsampling_x = 0;
  bad.subsampling_y = 0;
  CHECK(ctd::tip_deblock_frame(bad, options) == cudaErrorInvalidValue,
        "4:4:4 rejected");
  TipDeblockOptions deep = options;
  deep.bit_depth = 12;
  CHECK(ctd::tip_deblock_frame(frame, deep) == cudaErrorInvalidValue,
        "12-bit rejected");
  TipDeblockOptions unflagged = options;
  unflagged.tip_frame_mode = TipFrameMode::kAsReference;
  CHECK(ctd::tip_deblock_frame(frame, unflagged) == cudaErrorInvalidValue,
        "non-signallable apply flag rejected");

  // Inactive frame: success and no writes.
  HostFrame<std::uint8_t> gpu = host;
  TipDeblockOptions inactive = options;
  inactive.apply_deblocking_filter_tip = false;
  CHECK(ctd::tip_deblock_frame_arena(device.handle, inactive, device.arena) ==
            Status::kOk,
        "inactive frame succeeds");
  device.download(gpu);
  CHECK(same_buffer(gpu, host, nullptr, nullptr, nullptr),
        "inactive frame leaves pixels unchanged");

  // Unsupported plane/pass indices.
  CHECK(ctd::tip_deblock_plane_pass(frame, options, 3, 0) ==
            cudaErrorInvalidValue,
        "plane index validated");
  CHECK(ctd::tip_deblock_plane_pass(frame, options, 0, 2) ==
            cudaErrorInvalidValue,
        "pass index validated");
  std::cout << "  validation cases\n";
}

// ---------------------------------------------------------------------------
// 7. Timing (1080p, 8-bit, in-place, arena-resident).
// ---------------------------------------------------------------------------
void test_timing() {
  HostFrame<std::uint8_t> host = make_host_frame<std::uint8_t>(480, 270, 0xEEu);
  std::mt19937 rng(0x1080u);
  randomize(host, rng, 255);
  DeviceFrame<std::uint8_t> device;
  if (!device.init(host, 8, Arena::kAlignment, "timing")) {
    return;
  }
  TipDeblockOptions options = base_options(8);
  options.base_q_idx = 128;
  constexpr int kIterations = 50;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start), "event create");
  CUDA_CHECK(cudaEventCreate(&stop), "event create");
  CUDA_CHECK(cudaEventRecord(start), "event record");
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    (void)ctd::tip_deblock_frame_arena(device.handle, options, device.arena);
  }
  CUDA_CHECK(cudaEventRecord(stop), "event record");
  CUDA_CHECK(cudaEventSynchronize(stop), "event sync");
  float milliseconds = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop), "event time");
  CUDA_CHECK(cudaEventDestroy(start), "event destroy");
  CUDA_CHECK(cudaEventDestroy(stop), "event destroy");
  std::printf("  1080p 8-bit: %.4f ms/frame (%d iterations)\n",
              static_cast<double>(milliseconds) / kIterations, kIterations);
}

}  // namespace

int main(int argc, char** argv) {
  const std::string repo_root = (argc > 1) ? argv[1] : ".";
  int device_count = 0;
  const cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
  if (cuda_status != cudaSuccess || device_count == 0) {
    std::fprintf(stderr, "no CUDA device: %s\n",
                 cudaGetErrorString(cuda_status));
    return 77;
  }
  std::printf("AV2 §7.16 TIP deblocking CUDA test (%d device(s))\n",
              device_count);

  test_arena_contract();
  test_all_q_8bit();
  test_plane_passes_8bit();
  test_random_8bit();
  test_random_10bit();
  test_all_q_10bit();
  test_real_frame(repo_root);
  test_validation();
  test_timing();

  std::printf("\nchecks: %d  failures: %d  samples compared: %llu  filtered "
              "samples: %llu\n",
              g_checks, g_failures,
              static_cast<unsigned long long>(g_samples),
              static_cast<unsigned long long>(g_changed));
  return (g_failures == 0) ? 0 : 1;
}

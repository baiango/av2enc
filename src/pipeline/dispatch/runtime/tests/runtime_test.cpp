// CPU-only runtime tests. No CUDA, no devices, no CMake.
//
// The mock backend models devices, contexts, streams, events, arenas, pinned
// staging, and host-mediated copies. Every deterministic path the CUDA build
// uses is exercised here: device selection/fallback, per-GPU arena sizing,
// dispatch, raster merge, payload/recon transfer, reference exchange, prefetch,
// and mid-frame abort.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "../dispatch.h"
#include "../reference.h"
#include "../runtime.h"
#include "mock_backend.h"
#include "test_worker.h"

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* expression, const char* file, int line) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    std::cerr << "FAIL " << file << ":" << line << ": " << expression << "\n";
  }
}

#define CHECK(expression) check((expression), #expression, __FILE__, __LINE__)

constexpr std::uint64_t kMib = 1024ull * 1024ull;
constexpr std::uint64_t kGib = 1024ull * kMib;

namespace rt = av2enc::multigpu::runtime;
namespace mg = av2enc::multigpu;
namespace testing = av2enc::multigpu::runtime::testing;

testing::MockBackend make_backend(int count,
                             std::uint64_t total_bytes = 12ull * kGib) {
  std::vector<rt::DeviceProperties> devices;
  for (int i = 0; i < count; ++i) {
    devices.push_back(testing::mock_device(i, total_bytes));
  }
  return testing::MockBackend(std::move(devices));
}

rt::RuntimeConfig make_config(int gpu_count, int width, int height,
                              std::uint32_t stride) {
  rt::RuntimeConfig config;
  for (int i = 0; i < gpu_count; ++i) {
    config.gpu_ordinals.push_back(i);
  }
  config.planner.gpu_count = gpu_count;
  config.width = width;
  config.height = height;
  config.payload_stride_bytes = stride;
  config.min_gpus = 1;
  config.require_homogeneous = true;
  return config;
}

void check_frame_equal(const std::vector<std::uint8_t>& actual,
                       const std::vector<std::uint8_t>& expected,
                       const char* label) {
  ++g_checks;
  if (actual.size() != expected.size()) {
    ++g_failures;
    std::cerr << "FAIL " << label << ": size " << actual.size() << " vs "
              << expected.size() << "\n";
    return;
  }
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] != expected[i]) {
      ++g_failures;
      std::cerr << "FAIL " << label << ": first mismatch at " << i << " ("
                << static_cast<int>(actual[i]) << " vs "
                << static_cast<int>(expected[i]) << ")\n";
      std::size_t shown = 0;
      for (std::size_t j = i; j < actual.size() && shown < 8; ++j) {
        if (actual[j] != expected[j]) {
          std::cerr << "  mismatch[" << j << "]="
                    << static_cast<int>(actual[j]) << " expected "
                    << static_cast<int>(expected[j]) << "\n";
          ++shown;
        }
      }
      return;
    }
  }
}

void test_arena_capacity() {
  rt::DeviceProperties device = testing::mock_device(0, 12ull * kGib);
  device.free_vram_bytes = 11ull * kGib + 512ull * kMib;  // 11.5 GiB free

  rt::ArenaCapacity derived =
      rt::compute_arena_capacity(device, 512ull * kMib, 0);
  CHECK(derived.valid);
  CHECK(derived.capacity_bytes == 11264ull * kMib);
  CHECK(derived.capacity_bytes != 7500ull * kMib);  // no fixed preset
  CHECK(!derived.capped);

  rt::ArenaCapacity capped =
      rt::compute_arena_capacity(device, 512ull * kMib, 7000ull * kMib);
  CHECK(capped.valid);
  CHECK(capped.capacity_bytes == 7000ull * kMib);
  CHECK(capped.capped);

  rt::ArenaCapacity capped_large =
      rt::compute_arena_capacity(device, 512ull * kMib, 64ull * kGib);
  CHECK(capped_large.valid);
  CHECK(capped_large.capacity_bytes == derived.capacity_bytes);
  CHECK(!capped_large.capped);

  rt::DeviceProperties small = testing::mock_device(0, 512ull * kMib);
  small.free_vram_bytes = 512ull * kMib;
  rt::ArenaCapacity no_room =
      rt::compute_arena_capacity(small, 512ull * kMib, 0);
  CHECK(!no_room.valid);

  rt::ArenaCapacity tiny_cap =
      rt::compute_arena_capacity(device, 512ull * kMib, 512ull * 1024ull);
  CHECK(!tiny_cap.valid);

  rt::DeviceProperties ten_gb = testing::mock_device(0, 10ull * kGib);
  rt::ArenaCapacity ten_gb_capacity =
      rt::compute_arena_capacity(ten_gb, 512ull * kMib, 0);
  CHECK(ten_gb_capacity.valid);
  CHECK(ten_gb_capacity.capacity_bytes != 7500ull * kMib);
}

void test_selection_1080p() {
  testing::MockBackend backend = make_backend(2);
  rt::RuntimeConfig config = make_config(2, 1920, 1080, 64);
  rt::Selection selection;
  std::string error;
  const rt::Status status =
      rt::select_devices_and_plan(backend, config, selection, &error);
  CHECK(rt::is_ok(status));
  CHECK(selection.valid);
  CHECK(selection.plan.valid);
  CHECK(selection.slots.size() == 2);
  CHECK(selection.plan.config.gpu_count == 2);
  CHECK(selection.plan.layout.tile_count() == 16);
  CHECK(selection.plan.layout.cols == 8);
  CHECK(selection.plan.layout.rows == 2);
  CHECK(selection.plan.gpus[0].tile_count == 8);
  CHECK(selection.plan.gpus[1].tile_count == 8);
  CHECK(selection.slots[0].arena.valid);
  CHECK(selection.slots[0].arena.capacity_bytes != 7500ull * kMib);
  CHECK(selection.slots[0].required_live_mb > 0.0);
  CHECK(selection.slots[0].capacity_mb > selection.slots[0].required_live_mb);
}

void test_selection_4gpu_16x2() {
  testing::MockBackend backend = make_backend(4);
  rt::RuntimeConfig config = make_config(4, 1920, 1080, 64);
  rt::Selection selection;
  std::string error;
  CHECK(rt::is_ok(
      rt::select_devices_and_plan(backend, config, selection, &error)));
  CHECK(selection.valid);
  CHECK(selection.slots.size() == 4);
  CHECK(selection.plan.config.gpu_count == 4);
  CHECK(selection.plan.layout.cols == 16);
  CHECK(selection.plan.layout.rows == 2);
  CHECK(selection.plan.layout.tile_count() == 32);
  for (const mg::GpuPlan& gpu : selection.plan.gpus) {
    CHECK(gpu.tile_count == 8);
  }
  // 16x2 is a planner default for 4 GPUs but is not decoder-verified; the
  // runtime does not claim otherwise.
  CHECK(!selection.plan.measured);
}

void test_selection_missing_device() {
  testing::MockBackend backend = make_backend(1);
  rt::RuntimeConfig config = make_config(2, 640, 360, 64);
  rt::Selection selection;
  std::string error;
  CHECK(rt::select_devices_and_plan(backend, config, selection, &error) ==
        rt::Status::kDeviceUnavailable);

  config.allow_drop_missing = true;
  const rt::Status status =
      rt::select_devices_and_plan(backend, config, selection, &error);
  CHECK(rt::is_ok(status));
  CHECK(selection.valid);
  CHECK(selection.slots.size() == 1);
  CHECK(selection.slots[0].ordinal == 0);
  CHECK(selection.plan.config.gpu_count == 1);
  CHECK(selection.plan.layout.tile_count() == 8);  // 4x2 for one GPU
  CHECK(selection.fallback_used);
  CHECK(selection.dropped_ordinals.size() == 1);
  CHECK(selection.dropped_ordinals[0] == 1);
}

void test_selection_device_mismatch() {
  std::vector<rt::DeviceProperties> devices;
  devices.push_back(testing::mock_device(0, 12ull * kGib, 8, 6, 28));
  devices.push_back(testing::mock_device(1, 8ull * kGib, 8, 6, 28));
  testing::MockBackend backend(std::move(devices));
  rt::RuntimeConfig config = make_config(2, 640, 360, 64);
  rt::Selection selection;
  std::string error;
  CHECK(rt::select_devices_and_plan(backend, config, selection, &error) ==
        rt::Status::kDeviceMismatch);

  config.allow_drop_failed = true;
  const rt::Status status =
      rt::select_devices_and_plan(backend, config, selection, &error);
  CHECK(rt::is_ok(status));
  CHECK(selection.slots.size() == 1);
  CHECK(selection.slots[0].ordinal == 0);
  CHECK(selection.dropped_ordinals.size() == 1);
  CHECK(selection.dropped_ordinals[0] == 1);

  std::vector<rt::DeviceProperties> arch_mismatch;
  arch_mismatch.push_back(testing::mock_device(0, 12ull * kGib, 8, 6, 28));
  arch_mismatch.push_back(testing::mock_device(1, 12ull * kGib, 8, 9, 28));
  testing::MockBackend arch_backend(std::move(arch_mismatch));
  config.allow_drop_failed = false;
  CHECK(rt::select_devices_and_plan(arch_backend, config, selection, &error) ==
        rt::Status::kDeviceMismatch);
}

void test_dispatch_merge_and_reference() {
  testing::MockBackend backend = make_backend(2);
  rt::RuntimeConfig config = make_config(2, 640, 360, 64);
  rt::Runtime runtime;
  std::string error;
  CHECK(rt::is_ok(runtime.initialize(backend, config, &error)));
  CHECK(runtime.initialized());
  CHECK(runtime.gpu_runtime().size() == 2);
  CHECK(runtime.plan().layout.tile_count() == 16);
  CHECK(runtime.dispatch_table().valid);

  const std::vector<std::uint8_t> source =
      testing::expected_source_frame(640, 360, 0);
  const std::uint64_t frame_bytes = source.size();
  testing::PatternWorker worker(64, true);
  rt::FrameRequest request;
  request.frame_index = 0;
  request.source = source.data();
  request.source_bytes = source.size();
  request.capture_recon = true;

  rt::FrameResult result;
  CHECK(rt::is_ok(runtime.dispatch_frame(request, worker, result, &error)));
  CHECK(result.valid);
  CHECK(result.tile_count == 16);
  CHECK(result.tile_payloads.size() == 16);
  CHECK(result.failed_slot == -1);
  CHECK(worker.visits().size() == 2);
  for (const testing::PatternWorker::Visit& visit : worker.visits()) {
    CHECK(visit.source_ok);
  }

  std::uint64_t payload_bytes = 0;
  for (int tile = 0; tile < 16; ++tile) {
    const std::vector<std::uint8_t> expected =
        testing::expected_payload(tile, 0, 64);
    CHECK(result.tile_payloads[static_cast<std::size_t>(tile)] == expected);
    CHECK(result.tile_sizes[static_cast<std::size_t>(tile)] ==
          expected.size());
    payload_bytes += expected.size();
  }
  CHECK(result.payload_bytes == payload_bytes);
  check_frame_equal(result.recon, testing::expected_recon_frame(640, 360, 0), "frame 0 recon");
  CHECK(result.reference_fill_bytes == frame_bytes);
  CHECK(result.host_tx_bytes == 4 * frame_bytes + payload_bytes);
  CHECK(result.reference_fill_pending);
  CHECK(runtime.reference_fill_pending());
  CHECK(runtime.stats().frames == 1);
  CHECK(runtime.stats().merged_payload_bytes == payload_bytes);
  CHECK(runtime.stats().host_tx_bytes ==
        4 * frame_bytes + payload_bytes);

  // No device-wide sync in the per-frame path.
  CHECK(backend.device_sync_count() == 0);
  CHECK(backend.copy_count(testing::MockCopyKind::kHostToDevice) > 0);
  CHECK(backend.copy_count(testing::MockCopyKind::kDeviceToHost) > 0);
  CHECK(backend.copy_count(testing::MockCopyKind::kDeviceToDevice) > 0);

  // Each GPU uses a compute stream and a per-GPU copy stream.
  for (int ordinal = 0; ordinal < 2; ++ordinal) {
    std::vector<rt::StreamHandle> streams;
    for (const testing::MockCopy& copy : backend.copies()) {
      if (copy.ordinal == ordinal) {
        streams.push_back(copy.stream);
      }
    }
    std::sort(streams.begin(), streams.end());
    streams.erase(std::unique(streams.begin(), streams.end()), streams.end());
    CHECK(streams.size() >= 2);
  }

  CHECK(rt::is_ok(runtime.synchronize(&error)));
  CHECK(!runtime.reference_fill_pending());
  CHECK(runtime.gpu_runtime()[0].reference_fill_pending == false);

  // Frame 1 exercises the previous-frame reference and the rotating slot.
  backend.clear_logs();
  const std::vector<std::uint8_t> source1 =
      testing::expected_source_frame(640, 360, 1);
  testing::PatternWorker worker1(64, true);
  rt::FrameResult result1;
  rt::FrameRequest request1;
  request1.frame_index = 1;
  request1.source = source1.data();
  request1.source_bytes = source1.size();
  request1.capture_recon = true;
  CHECK(rt::is_ok(runtime.dispatch_frame(request1, worker1, result1, &error)));
  CHECK(result1.valid);
  check_frame_equal(result1.recon, testing::expected_recon_frame(640, 360, 1), "frame 1 recon");
  CHECK(worker1.visits().size() == 2);
  for (const testing::PatternWorker::Visit& visit : worker1.visits()) {
    CHECK(visit.reference_ok);
    CHECK(visit.reference_index == 1);
  }
  CHECK(rt::is_ok(runtime.synchronize(&error)));
}

void test_shutdown_releases_resources() {
  testing::MockBackend backend = make_backend(2);
  rt::RuntimeConfig config = make_config(2, 640, 360, 64);
  {
    rt::Runtime runtime;
    std::string error;
    CHECK(rt::is_ok(runtime.initialize(backend, config, &error)));
    std::vector<std::uint8_t> source =
        testing::expected_source_frame(640, 360, 0);
    {
      testing::PatternWorker worker(64, true);
      rt::FrameRequest request;
      request.frame_index = 0;
      request.source = source.data();
      request.source_bytes = source.size();
      rt::FrameResult result;
      CHECK(rt::is_ok(runtime.dispatch_frame(request, worker, result, &error)));
      CHECK(result.valid);
    }
    runtime.shutdown();
    CHECK(!runtime.initialized());
  }
  CHECK(backend.live_arenas() == 0);
  CHECK(backend.live_arena_blocks() == 0);
  CHECK(backend.live_pinned_allocations() == 0);
  CHECK(backend.live_streams() == 0);
  CHECK(backend.live_contexts() == 0);
}

void test_merge_order_is_raster() {
  testing::MockBackend backend = make_backend(2);
  rt::RuntimeConfig config = make_config(2, 640, 360, 64);
  rt::Selection selection;
  std::string error;
  CHECK(rt::is_ok(
      rt::select_devices_and_plan(backend, config, selection, &error)));

  std::vector<std::vector<std::uint8_t>> buffers(16);
  std::vector<rt::TilePayloadView> payloads;
  for (int tile = 15; tile >= 0; --tile) {  // reverse completion order
    buffers[static_cast<std::size_t>(tile)].assign(
        static_cast<std::size_t>(4 + tile),
        static_cast<std::uint8_t>(tile * 3 + 1));
    payloads.push_back(rt::TilePayloadView{
        tile,
        std::span<const std::uint8_t>(
            buffers[static_cast<std::size_t>(tile)].data(),
            buffers[static_cast<std::size_t>(tile)].size())});
  }
  rt::MergedFrame merged;
  CHECK(rt::merge_payloads(selection.plan, payloads, merged, &error));
  CHECK(merged.valid);
  CHECK(merged.tiles.size() == 16);
  for (int tile = 0; tile < 16; ++tile) {
    CHECK(merged.tiles[static_cast<std::size_t>(tile)].size() ==
          static_cast<std::size_t>(4 + tile));
    CHECK(merged.tiles[static_cast<std::size_t>(tile)][0] ==
          static_cast<std::uint8_t>(tile * 3 + 1));
  }

  // Duplicate.
  payloads.push_back(payloads[0]);
  rt::MergedFrame duplicate;
  CHECK(!rt::merge_payloads(selection.plan, payloads, duplicate, &error));
  CHECK(!duplicate.valid);
  CHECK(duplicate.tiles.empty());

  // Missing.
  std::vector<rt::TilePayloadView> missing(payloads.begin(),
                                           payloads.end() - 2);
  rt::MergedFrame incomplete;
  CHECK(!rt::merge_payloads(selection.plan, missing, incomplete, &error));
  CHECK(!incomplete.valid);
  CHECK(incomplete.tiles.empty());
}

void test_prefetch_and_overlap() {
  testing::MockBackend backend = make_backend(2);
  rt::RuntimeConfig config = make_config(2, 640, 360, 64);
  rt::Runtime runtime;
  std::string error;
  CHECK(rt::is_ok(runtime.initialize(backend, config, &error)));

  const std::vector<std::uint8_t> source0 =
      testing::expected_source_frame(640, 360, 0);
  testing::PatternWorker worker0(64, true);
  rt::FrameResult result0;
  rt::FrameRequest request0;
  request0.frame_index = 0;
  request0.source = source0.data();
  request0.source_bytes = source0.size();
  CHECK(rt::is_ok(runtime.dispatch_frame(request0, worker0, result0, &error)));
  const std::uint64_t frame_bytes = source0.size();
  CHECK(runtime.stats().source_upload_bytes == 2 * frame_bytes);
  CHECK(rt::is_ok(runtime.synchronize(&error)));

  const std::vector<std::uint8_t> source1 =
      testing::expected_source_frame(640, 360, 1);
  backend.clear_logs();
  CHECK(rt::is_ok(runtime.prefetch_source(1, source1.data(), source1.size(),
                                          &error)));
  CHECK(runtime.stats().prefetches == 1);
  CHECK(backend.copy_count(testing::MockCopyKind::kHostToDevice) == 2);
  CHECK(runtime.prefetch_source(1, source1.data(), source1.size(), &error) ==
        rt::Status::kAlreadyInitialized);

  testing::PatternWorker worker1(64, true);
  rt::FrameResult result1;
  rt::FrameRequest request1;
  request1.frame_index = 1;
  request1.source = source1.data();
  request1.source_bytes = source1.size();
  request1.capture_recon = true;
  CHECK(rt::is_ok(runtime.dispatch_frame(request1, worker1, result1, &error)));
  CHECK(result1.valid);
  check_frame_equal(result1.recon, testing::expected_recon_frame(640, 360, 1), "frame 1 recon");
  CHECK(worker1.visits().size() == 2);
  for (const testing::PatternWorker::Visit& visit : worker1.visits()) {
    CHECK(visit.source_ok);
    CHECK(visit.reference_ok);
    CHECK(visit.reference_index == 1);
  }
  // The prefetch counted the source upload; dispatch did not upload it again.
  CHECK(runtime.stats().source_upload_bytes == 4 * frame_bytes);
  CHECK(runtime.stats().prefetches == 1);
  // host_tx per frame still accounts for the source.
  CHECK(result1.host_tx_bytes >= 4 * frame_bytes);
  CHECK(rt::is_ok(runtime.synchronize(&error)));
}

void test_worker_failure_aborts_whole_frame() {
  testing::MockBackend backend = make_backend(2);
  rt::RuntimeConfig config = make_config(2, 640, 360, 64);
  rt::Runtime runtime;
  std::string error;
  CHECK(rt::is_ok(runtime.initialize(backend, config, &error)));

  const std::vector<std::uint8_t> source =
      testing::expected_source_frame(640, 360, 0);
  testing::PatternWorker worker(64, true);
  worker.fail_slot_ = 1;
  worker.fail_frame_ = 0;
  rt::FrameResult result;
  rt::FrameRequest request;
  request.frame_index = 0;
  request.source = source.data();
  request.source_bytes = source.size();
  CHECK(runtime.dispatch_frame(request, worker, result, &error) ==
        rt::Status::kDispatchFailed);
  CHECK(!result.valid);
  CHECK(result.failed_slot == 1);
  CHECK(result.tile_payloads.empty());
  CHECK(!result.error.empty());
  // Error path drains streams, not the device.
  CHECK(backend.device_sync_count() == 0);
  CHECK(backend.event_count(testing::MockEventOp::kStreamSynchronize) > 0);

  // The runtime must recover at the next frame boundary, never emit a partial
  // frame, and never silently fall back to one GPU.
  worker.fail_slot_ = -1;
  worker.fail_frame_ = -1;
  rt::FrameResult recovered;
  CHECK(rt::is_ok(runtime.dispatch_frame(request, worker, recovered, &error)));
  CHECK(recovered.valid);
  CHECK(recovered.tile_payloads.size() == 16);
  CHECK(runtime.plan().config.gpu_count == 2);
}

void test_invalid_tile_size_aborts() {
  testing::MockBackend backend = make_backend(2);
  rt::RuntimeConfig config = make_config(2, 640, 360, 64);
  rt::Runtime runtime;
  std::string error;
  CHECK(rt::is_ok(runtime.initialize(backend, config, &error)));

  const std::vector<std::uint8_t> source =
      testing::expected_source_frame(640, 360, 0);
  rt::FrameRequest request;
  request.frame_index = 0;
  request.source = source.data();
  request.source_bytes = source.size();

  testing::PatternWorker zero(64, true);
  zero.bad_tile_size_slot_ = 0;
  zero.bad_tile_size_value_ = 0;
  rt::FrameResult result;
  CHECK(runtime.dispatch_frame(request, zero, result, &error) ==
        rt::Status::kDispatchFailed);
  CHECK(!result.valid);
  CHECK(result.tile_payloads.empty());

  testing::PatternWorker too_large(64, true);
  too_large.bad_tile_size_slot_ = 0;
  too_large.bad_tile_size_value_ = 65;
  rt::FrameResult result2;
  CHECK(runtime.dispatch_frame(request, too_large, result2, &error) ==
        rt::Status::kDispatchFailed);
  CHECK(!result2.valid);

  testing::PatternWorker good(64, true);
  rt::FrameResult result3;
  CHECK(rt::is_ok(runtime.dispatch_frame(request, good, result3, &error)));
  CHECK(result3.valid);
}

void test_copy_failure_aborts_frame() {
  testing::MockOptions options;
  options.fail_copy_after = 40;
  testing::MockBackend backend(
      {testing::mock_device(0, 12ull * kGib),
       testing::mock_device(1, 12ull * kGib)},
      options);
  rt::RuntimeConfig config = make_config(2, 640, 360, 64);
  rt::Runtime runtime;
  std::string error;
  CHECK(rt::is_ok(runtime.initialize(backend, config, &error)));
  CHECK(backend.copy_count(testing::MockCopyKind::kHostToDevice) == 0);

  const std::vector<std::uint8_t> source =
      testing::expected_source_frame(640, 360, 0);
  testing::PatternWorker worker(64, true);
  rt::FrameRequest request;
  request.frame_index = 0;
  request.source = source.data();
  request.source_bytes = source.size();
  rt::FrameResult result;
  const rt::Status status =
      runtime.dispatch_frame(request, worker, result, &error);
  CHECK(!rt::is_ok(status));
  CHECK(!result.valid);
  CHECK(result.tile_payloads.empty());
  CHECK(result.failed_slot >= 0);
  CHECK(!result.error.empty());
  CHECK(backend.device_sync_count() == 0);
  CHECK(runtime.plan().config.gpu_count == 2);  // no mid-stream fallback
}

void test_arena_oom_fallback() {
  testing::MockOptions options;
  options.arena_fail_above[1] = 1ull * kMib;
  std::vector<rt::DeviceProperties> devices;
  devices.push_back(testing::mock_device(0, 12ull * kGib));
  devices.push_back(testing::mock_device(1, 12ull * kGib));
  testing::MockBackend backend(std::move(devices), options);

  rt::RuntimeConfig config = make_config(2, 640, 360, 64);
  config.allow_drop_failed = true;
  config.arena_init_attempts = 1;
  rt::Runtime runtime;
  std::string error;
  CHECK(rt::is_ok(runtime.initialize(backend, config, &error)));
  CHECK(runtime.initialized());
  CHECK(runtime.gpu_runtime().size() == 1);
  CHECK(runtime.gpu_runtime()[0].ordinal == 0);
  CHECK(runtime.plan().config.gpu_count == 1);
  CHECK(runtime.selection().dropped_ordinals.size() == 1);
  CHECK(runtime.selection().dropped_ordinals[0] == 1);
  CHECK(runtime.stats().device_drops == 1);
  CHECK(runtime.stats().fallback_replans >= 1);
  CHECK(backend.live_arenas() == 1);

  const std::vector<std::uint8_t> source =
      testing::expected_source_frame(640, 360, 0);
  testing::PatternWorker worker(64, true);
  rt::FrameRequest request;
  request.frame_index = 0;
  request.source = source.data();
  request.source_bytes = source.size();
  rt::FrameResult result;
  CHECK(rt::is_ok(runtime.dispatch_frame(request, worker, result, &error)));
  CHECK(result.valid);
  CHECK(result.tile_payloads.size() == 8);

  // Strict mode fails with the device ordinal instead of running degraded.
  testing::MockBackend strict_backend(
      {testing::mock_device(0, 12ull * kGib),
       testing::mock_device(1, 12ull * kGib)},
      options);
  rt::RuntimeConfig strict = make_config(2, 640, 360, 64);
  strict.allow_drop_failed = false;
  strict.arena_init_attempts = 1;
  rt::Runtime strict_runtime;
  CHECK(strict_runtime.initialize(strict_backend, strict, &error) ==
        rt::Status::kOutOfMemory);
  CHECK(!strict_runtime.initialized());
  CHECK(strict_backend.live_arenas() == 0);
  CHECK(strict_backend.live_contexts() == 0);
}

void test_allocation_oom_fallback() {
  testing::MockOptions options;
  options.arena_allocate_fail_above[1] = 1024;  // GPU 1 cannot hold a frame
  testing::MockBackend backend(
      {testing::mock_device(0, 12ull * kGib),
       testing::mock_device(1, 12ull * kGib)},
      options);
  rt::RuntimeConfig config = make_config(2, 640, 360, 64);
  config.allow_drop_failed = true;
  config.arena_init_attempts = 1;
  rt::Runtime runtime;
  std::string error;
  CHECK(rt::is_ok(runtime.initialize(backend, config, &error)));
  CHECK(runtime.initialized());
  CHECK(runtime.gpu_runtime().size() == 1);
  CHECK(runtime.gpu_runtime()[0].ordinal == 0);
  CHECK(runtime.plan().config.gpu_count == 1);
  CHECK(runtime.stats().device_drops == 1);
  CHECK(runtime.stats().fallback_replans >= 1);
  CHECK(backend.live_arenas() == 1);

  const std::vector<std::uint8_t> source =
      testing::expected_source_frame(640, 360, 0);
  testing::PatternWorker worker(64, true);
  rt::FrameRequest request;
  request.frame_index = 0;
  request.source = source.data();
  request.source_bytes = source.size();
  rt::FrameResult result;
  CHECK(rt::is_ok(runtime.dispatch_frame(request, worker, result, &error)));
  CHECK(result.valid);
  CHECK(result.tile_payloads.size() == 8);

  testing::MockBackend strict_backend(
      {testing::mock_device(0, 12ull * kGib),
       testing::mock_device(1, 12ull * kGib)},
      options);
  rt::RuntimeConfig strict = make_config(2, 640, 360, 64);
  strict.allow_drop_failed = false;
  rt::Runtime strict_runtime;
  CHECK(!rt::is_ok(strict_runtime.initialize(strict_backend, strict, &error)));
  CHECK(!strict_runtime.initialized());
  CHECK(strict_backend.live_arenas() == 0);
  CHECK(strict_backend.live_contexts() == 0);
}

void test_pinned_allocation_failure() {
  testing::MockOptions options;
  options.fail_pinned_alloc = true;
  testing::MockBackend backend({testing::mock_device(0, 12ull * kGib)}, options);
  rt::RuntimeConfig config = make_config(1, 640, 360, 64);
  rt::Runtime runtime;
  std::string error;
  CHECK(runtime.initialize(backend, config, &error) == rt::Status::kOutOfMemory);
  CHECK(!runtime.initialized());
  CHECK(backend.live_arenas() == 0);
  CHECK(backend.live_streams() == 0);
  CHECK(backend.live_contexts() == 0);
}

void test_arena_halving_retry() {
  testing::MockOptions options;
  options.arena_fail_above[0] = 8ull * kGib;
  testing::MockBackend backend({testing::mock_device(0, 12ull * kGib)}, options);
  rt::RuntimeConfig config = make_config(1, 640, 360, 64);
  config.arena_init_attempts = 3;
  rt::Runtime runtime;
  std::string error;
  CHECK(rt::is_ok(runtime.initialize(backend, config, &error)));
  CHECK(runtime.initialized());
  CHECK(runtime.gpu_runtime().size() == 1);
  const std::uint64_t capacity =
      runtime.gpu_runtime()[0].arena_capacity_bytes;
  CHECK(capacity <= 8ull * kGib);
  CHECK(capacity > 1ull * kGib);
  CHECK(static_cast<double>(capacity) / static_cast<double>(kMib) >=
        runtime.selection().slots[0].required_live_mb);
}

void test_single_gpu_path() {
  testing::MockBackend backend = make_backend(1);
  rt::RuntimeConfig config = make_config(1, 640, 360, 64);
  rt::Runtime runtime;
  std::string error;
  CHECK(rt::is_ok(runtime.initialize(backend, config, &error)));
  CHECK(runtime.plan().layout.tile_count() == 8);
  CHECK(runtime.plan().config.gpu_count == 1);
  CHECK(runtime.gpu_runtime().size() == 1);

  const std::vector<std::uint8_t> source =
      testing::expected_source_frame(640, 360, 0);
  testing::PatternWorker worker(64, true);
  rt::FrameRequest request;
  request.frame_index = 0;
  request.source = source.data();
  request.source_bytes = source.size();
  request.capture_recon = true;
  rt::FrameResult result;
  CHECK(rt::is_ok(runtime.dispatch_frame(request, worker, result, &error)));
  CHECK(result.valid);
  CHECK(result.tile_payloads.size() == 8);
  CHECK(result.reference_fill_bytes == 0);
  CHECK(result.host_tx_bytes ==
        2 * source.size() + result.payload_bytes);
  check_frame_equal(result.recon, testing::expected_recon_frame(640, 360, 0), "frame 0 recon");
  CHECK(rt::is_ok(runtime.synchronize(&error)));
}

void test_region_copy_geometry() {
  testing::MockBackend backend = make_backend(2);
  rt::RuntimeConfig config = make_config(2, 640, 360, 64);
  rt::Selection selection;
  std::string error;
  CHECK(rt::is_ok(
      rt::select_devices_and_plan(backend, config, selection, &error)));
  const auto& layout = selection.plan.layout;
  const std::vector<std::uint8_t> source =
      testing::expected_recon_frame(640, 360, 0);

  std::vector<std::uint8_t> host_assembled(source.size(), 0);
  for (const av2enc::tiles::TileRect& rect : layout.tiles) {
    CHECK(rt::copy_tile_region_host(host_assembled.data(), source.data(),
                                    layout, rect, 8) == rt::Status::kOk);
  }
  check_frame_equal(host_assembled, source, "host region assembly");

  rt::StreamHandle stream = rt::kInvalidStream;
  CHECK(rt::is_ok(backend.stream_create(0, &stream, &error)));
  std::vector<std::uint8_t> device(source.size(), 0);
  for (const av2enc::tiles::TileRect& rect : layout.tiles) {
    CHECK(rt::enqueue_tile_region_copy(backend, rt::CopyDirection::kHostToDevice,
                                       stream, device.data(), source.data(), 0,
                                       layout, rect, 8) == rt::Status::kOk);
  }
  std::vector<std::uint8_t> downloaded(source.size(), 0);
  for (const av2enc::tiles::TileRect& rect : layout.tiles) {
    CHECK(rt::enqueue_tile_region_copy(backend, rt::CopyDirection::kDeviceToHost,
                                       stream, downloaded.data(), device.data(),
                                       0, layout, rect, 8) ==
          rt::Status::kOk);
  }
  check_frame_equal(downloaded, source, "device region round-trip");
  backend.stream_destroy(stream);

  // 1080p 8x2 has 512/568-row tiles; cover the partially filled bottom row.
  rt::RuntimeConfig big_config = make_config(2, 1920, 1080, 64);
  rt::Selection big_selection;
  CHECK(rt::is_ok(rt::select_devices_and_plan(backend, big_config,
                                             big_selection, &error)));
  const std::vector<std::uint8_t> big_source =
      testing::expected_recon_frame(1920, 1080, 0);
  std::vector<std::uint8_t> big_assembled(big_source.size(), 0);
  for (const av2enc::tiles::TileRect& rect : big_selection.plan.layout.tiles) {
    CHECK(rt::copy_tile_region_host(big_assembled.data(), big_source.data(),
                                    big_selection.plan.layout, rect, 8) ==
          rt::Status::kOk);
  }
  check_frame_equal(big_assembled, big_source, "1080p host region assembly");

  // 10-bit frame memory doubles, and the region accounting must match exactly.
  rt::RuntimeConfig hbd_config = make_config(2, 640, 360, 64);
  hbd_config.planner.bit_depth = 10;
  rt::Selection hbd_selection;
  CHECK(rt::is_ok(rt::select_devices_and_plan(backend, hbd_config,
                                             hbd_selection, &error)));
  std::uint64_t hbd_regions = 0;
  for (const av2enc::tiles::TileRect& rect : hbd_selection.plan.layout.tiles) {
    hbd_regions += rt::tile_region_bytes(hbd_selection.plan.layout, rect, 10);
  }
  CHECK(hbd_regions == hbd_selection.plan.memory.frame_bytes);
  CHECK(hbd_selection.plan.memory.frame_bytes == 2 * source.size());
}

void test_reference_schedule() {
  testing::MockBackend backend = make_backend(2);
  rt::RuntimeConfig config = make_config(2, 640, 360, 64);
  rt::Selection selection;
  std::string error;
  CHECK(rt::is_ok(
      rt::select_devices_and_plan(backend, config, selection, &error)));

  const std::uint64_t frame_bytes = selection.plan.memory.frame_bytes;
  std::uint64_t tile_sum = 0;
  for (const av2enc::tiles::TileRect& rect : selection.plan.layout.tiles) {
    tile_sum += rt::tile_region_bytes(selection.plan.layout, rect, 8);
  }
  CHECK(tile_sum == frame_bytes);

  const std::vector<int> ordinals = {0, 1};
  const std::vector<std::uint64_t> payload = {100, 200};
  const rt::ReferenceSchedule schedule =
      rt::plan_reference_schedule(selection.plan, ordinals, payload);
  CHECK(schedule.valid);
  CHECK(schedule.gpu_count == 2);
  CHECK(schedule.tile_count == 16);
  CHECK(schedule.source_upload_bytes == 2 * frame_bytes);
  CHECK(schedule.payload_bytes == 300);
  CHECK(schedule.recon_gather_bytes == frame_bytes);
  CHECK(schedule.reference_fill_bytes == frame_bytes);
  CHECK(schedule.reference_d2d_bytes == frame_bytes);
  CHECK(schedule.host_tx_bytes ==
        2 * frame_bytes + 300 + frame_bytes + frame_bytes);
  CHECK(schedule.estimated_ms > 0.0);

  // Deterministic order: source, payload, raster gather, GPU-major fill.
  std::size_t index = 0;
  for (int slot = 0; slot < 2; ++slot) {
    CHECK(schedule.ops[index].phase == rt::TransferPhase::kSourceUpload);
    CHECK(schedule.ops[index].gpu_slot == slot);
    CHECK(schedule.ops[index].ordinal == ordinals[static_cast<std::size_t>(slot)]);
    ++index;
  }
  for (int slot = 0; slot < 2; ++slot) {
    CHECK(schedule.ops[index].phase == rt::TransferPhase::kPayloadDownload);
    ++index;
  }
  for (int tile = 0; tile < 16; ++tile) {
    CHECK(schedule.ops[index].phase == rt::TransferPhase::kReconGather);
    CHECK(schedule.ops[index].tile_index == tile);
    ++index;
  }
  for (int slot = 0; slot < 2; ++slot) {
    for (int tile = 0; tile < 16; ++tile) {
      CHECK(schedule.ops[index].phase == rt::TransferPhase::kReferenceFill);
      CHECK(schedule.ops[index].tile_index == tile);
      ++index;
    }
  }
  CHECK(index == schedule.ops.size());
}

void test_planner_agrees_with_dispatch() {
  testing::MockBackend backend = make_backend(2);
  rt::RuntimeConfig config = make_config(2, 1920, 1080, 64);
  rt::Selection selection;
  std::string error;
  CHECK(rt::is_ok(
      rt::select_devices_and_plan(backend, config, selection, &error)));
  const std::vector<int> ordinals = {0, 1};
  const rt::DispatchTable table =
      rt::build_dispatch_table(selection.plan, ordinals, &error);
  CHECK(table.valid);
  CHECK(table.tile_count == 16);
  CHECK(table.by_tile.size() == 16);
  CHECK(table.by_gpu.size() == 2);
  for (int tile = 0; tile < 16; ++tile) {
    const rt::DispatchEntry& entry =
        table.by_tile[static_cast<std::size_t>(tile)];
    CHECK(entry.tile_index == tile);
    CHECK(entry.gpu_slot >= 0 && entry.gpu_slot < 2);
    CHECK(table.by_gpu[static_cast<std::size_t>(entry.gpu_slot)]
              [static_cast<std::size_t>(entry.local_index)]
                  .tile_index == tile);
  }
  CHECK(table.by_gpu[0].size() == 8);
  CHECK(table.by_gpu[1].size() == 8);
}

}  // namespace

int main() {
  std::cout << "=== av2enc multi-GPU runtime tests (mock backend) ===\n";
  test_arena_capacity();
  test_selection_1080p();
  test_selection_4gpu_16x2();
  test_selection_missing_device();
  test_selection_device_mismatch();
  test_dispatch_merge_and_reference();
  test_shutdown_releases_resources();
  test_merge_order_is_raster();
  test_prefetch_and_overlap();
  test_worker_failure_aborts_whole_frame();
  test_invalid_tile_size_aborts();
  test_copy_failure_aborts_frame();
  test_arena_oom_fallback();
  test_allocation_oom_fallback();
  test_pinned_allocation_failure();
  test_arena_halving_retry();
  test_single_gpu_path();
  test_region_copy_geometry();
  test_reference_schedule();
  test_planner_agrees_with_dispatch();
  std::cout << "checks: " << g_checks << ", failures: " << g_failures << "\n";
  if (g_failures != 0) {
    std::cout << "RESULT: FAIL\n";
    return 1;
  }
  std::cout << "RESULT: PASS\n";
  return 0;
}

// Real CUDA parity test for AV2 chroma loop restoration.
//
// Build/run on the Linux/CUDA target:
//   ./build_gpu_test.sh
//
// Coverage:
//   * the same 480+ case matrix as the emulation test: subsets 0/1/2,
//     uv_sym on/off, unit sizes 32/64/128/256, 8/10-bit,
//     4:2:0/4:2:2/4:4:4, edge/tiny/odd MI grids;
//   * bank/merged filter expansion on the device against the independent
//     reference;
//   * full-plane bit-exact parity against `lr_chroma::chroma_apply()` for
//     8-bit 4:2:0 and against the bit-depth/subsampling reference otherwise;
//   * one Arena only, exact `required_device_bytes`, one-byte-short
//     `kArenaTooSmall`, two launches with mark()/rewind() and no net growth;
//   * a runtime device-visibility probe kernel.
//
// There is no cudaMalloc/cudaFree/cudaMallocHost in this test.

#define AV2ENC_LRC_TEST_NO_MAIN
#include "lr_chroma_emulation_test.cpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "cuda/arena.h"
#include "cuda/codec/filters/loop_restoration/chroma/lr_chroma.h"

namespace {

using av2enc::cuda::Arena;
using av2enc::cuda::Status;
using av2enc::cuda::is_ok;
namespace lrc2 = av2enc::cuda::lr_chroma;

bool upload_bytes(void* device, const void* host, std::size_t bytes) {
  if (bytes == 0) return true;
  return cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
}

bool download_bytes(void* host, const void* device, std::size_t bytes) {
  if (bytes == 0) return true;
  return cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
}

bool sync_ok() { return cudaStreamSynchronize(nullptr) == cudaSuccess; }

void* alloc_arena(Arena& arena, std::size_t bytes, const char* what) {
  void* p = arena.allocate(bytes);
  check(p != nullptr, what);
  return p;
}

struct DeviceCase {
  void* curr[2] = {nullptr, nullptr};
  void* cdef[2] = {nullptr, nullptr};
  void* dst[2] = {nullptr, nullptr};
  int stride = 0;
};

DeviceCase allocate_device_case(Arena& arena, const TestCase& c) {
  DeviceCase d;
  const int width = case_width(c.config);
  const int height = case_height(c.config);
  d.stride = width;
  const std::size_t samples =
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  const std::size_t sample_bytes = c.config.bit_depth == 8 ? 1 : 2;
  const std::size_t bytes = samples * sample_bytes;
  for (int chroma = 0; chroma < 2; ++chroma) {
    d.curr[chroma] = alloc_arena(arena, bytes, "arena curr");
    d.cdef[chroma] = alloc_arena(arena, bytes, "arena cdef");
    d.dst[chroma] = alloc_arena(arena, bytes, "arena dst");
  }
  check(upload_bytes(d.curr[0], plane_data(c.curr[0], c.config.bit_depth),
                     bytes),
        "upload curr U");
  check(upload_bytes(d.curr[1], plane_data(c.curr[1], c.config.bit_depth),
                     bytes),
        "upload curr V");
  check(upload_bytes(d.cdef[0], plane_data(c.cdef[0], c.config.bit_depth),
                     bytes),
        "upload cdef U");
  check(upload_bytes(d.cdef[1], plane_data(c.cdef[1], c.config.bit_depth),
                     bytes),
        "upload cdef V");
  return d;
}

lrc2::LrChromaFrame make_device_frame(const TestCase& c,
                                      const DeviceCase& d) {
  lrc2::LrChromaFrame f{};
  for (int chroma = 0; chroma < 2; ++chroma) {
    f.chroma[chroma].curr = d.curr[chroma];
    f.chroma[chroma].curr_stride = d.stride;
    f.chroma[chroma].cdef = d.cdef[chroma];
    f.chroma[chroma].cdef_stride = d.stride;
    f.chroma[chroma].dst = d.dst[chroma];
    f.chroma[chroma].dst_stride = d.stride;
  }
  f.luma.curr = d.curr[0];
  f.luma.curr_stride = d.stride;
  f.luma.cdef = d.cdef[0];
  f.luma.cdef_stride = d.stride;
  f.mi_cols = c.config.mi_cols;
  f.mi_rows = c.config.mi_rows;
  f.sub_x = c.config.sub_x;
  f.sub_y = c.config.sub_y;
  f.bit_depth = c.config.bit_depth;
  f.cfl_ds_filter_index = c.config.cfl_ds_filter_index;
  return f;
}

void run_gpu_case(Arena& arena, const TestCase& c) {
  const std::uint64_t mark = arena.mark();
  lrc2::LrChromaWorkspace ws;
  std::size_t required = 0;
  std::string error;
  const Status status =
      lrc2::allocate_checked(arena, c.config, &ws, &required, &error);
  check(is_ok(status), "allocate_checked");
  if (!is_ok(status)) {
    arena.rewind(mark);
    return;
  }
  check(required == lrc2::required_device_bytes(c.config),
        "reported requirement matches required_device_bytes");
  check(ws.persistent_bytes() == required,
        "workspace consumes exactly required_device_bytes");

  for (int chroma = 0; chroma < 2; ++chroma) {
    const int count = case_unit_count(c.config);
    check(is_ok(lrc2::upload_units(ws, chroma, c.units[chroma].data(),
                                   count)),
          "upload_units");
  }
  check(is_ok(lrc2::resolve_units(ws)), "resolve_units");
  check(sync_ok(), "resolve sync");

  // Resolver parity on the device.
  for (int chroma = 0; chroma < 2; ++chroma) {
    const int count = case_unit_count(c.config);
    std::vector<std::int16_t> got(static_cast<std::size_t>(count) * 18, 0);
    std::vector<std::uint8_t> got_active(static_cast<std::size_t>(count), 0);
    check(download_bytes(got.data(), ws.resolved(chroma),
                         got.size() * sizeof(std::int16_t)),
          "download resolved");
    check(download_bytes(got_active.data(), ws.active(chroma),
                         got_active.size()),
          "download active");
    for (int u = 0; u < count; ++u) {
      ++g_resolver_units;
      bool same = got_active[static_cast<std::size_t>(u)] ==
                  c.ref_active[chroma][static_cast<std::size_t>(u)];
      for (int j = 0; j < 18 && same; ++j) {
        same = got[static_cast<std::size_t>(u) * 18 + j] ==
               c.ref_resolved[chroma][static_cast<std::size_t>(u) * 18 + j];
      }
      check(same, "GPU resolver matches reference expansion");
    }
  }

  DeviceCase d = allocate_device_case(arena, c);
  // Luma plane: allocate separately (DeviceCase only covers chroma).
  const int luma_width = c.config.mi_cols * 4;
  const int luma_height = c.config.mi_rows * 4;
  const std::size_t luma_samples =
      static_cast<std::size_t>(luma_width) * static_cast<std::size_t>(luma_height);
  const std::size_t sample_bytes = c.config.bit_depth == 8 ? 1 : 2;
  void* luma_curr = alloc_arena(arena, luma_samples * sample_bytes, "luma curr");
  void* luma_cdef = alloc_arena(arena, luma_samples * sample_bytes, "luma cdef");
  check(upload_bytes(luma_curr, plane_data(c.luma_curr, c.config.bit_depth),
                     luma_samples * sample_bytes),
        "upload luma curr");
  check(upload_bytes(luma_cdef, plane_data(c.luma_cdef, c.config.bit_depth),
                     luma_samples * sample_bytes),
        "upload luma cdef");

  lrc2::LrChromaFrame frame = make_device_frame(c, d);
  frame.luma.curr = luma_curr;
  frame.luma.cdef = luma_cdef;
  frame.luma.curr_stride = luma_width;
  frame.luma.cdef_stride = luma_width;
  if (c.tiled) {
    const std::size_t grid_count =
        static_cast<std::size_t>(c.config.mi_cols) * c.config.mi_rows;
    auto upload_grid = [&](const std::vector<std::uint16_t>& host,
                           const std::uint16_t** out) {
      void* device = alloc_arena(arena, grid_count * sizeof(std::uint16_t),
                                 "tile grid");
      check(upload_bytes(device, host.data(),
                         grid_count * sizeof(std::uint16_t)),
            "upload tile grid");
      *out = static_cast<const std::uint16_t*>(device);
    };
    upload_grid(c.mi_col_start, &frame.mi_col_start);
    upload_grid(c.mi_col_end, &frame.mi_col_end);
    upload_grid(c.mi_row_start, &frame.mi_row_start);
    upload_grid(c.mi_row_end, &frame.mi_row_end);
    frame.disable_loopfilters_across_tiles = 1;
  }

  check(is_ok(lrc2::apply(ws, frame)), "apply");
  check(sync_ok(), "apply sync");

  for (int chroma = 0; chroma < 2; ++chroma) {
    const int width = case_width(c.config);
    const int height = case_height(c.config);
    if (c.config.bit_depth == 8) {
      std::vector<std::uint8_t> got(
          static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
      check(download_bytes(got.data(), d.dst[chroma], got.size()),
            "download dst 8-bit");
      compare_plane(got, c.oracle[chroma].u8,
                    "GPU output matches CPU oracle (8-bit)");
    } else {
      std::vector<std::uint16_t> got(
          static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
      check(download_bytes(got.data(), d.dst[chroma],
                           got.size() * sizeof(std::uint16_t)),
            "download dst 10-bit");
      compare_plane16(got, c.oracle[chroma].u16,
                      "GPU output matches reference (10-bit)");
    }
  }
  arena.rewind(mark);
}

__global__ void visibility_probe_kernel(lrc2::LrChromaCore core,
                                        const lrc2::LrChromaUnit* units,
                                        int count, std::int16_t* resolved,
                                        std::uint8_t* active,
                                        lrc2::LrChromaBankState* bank) {
  lrc2::LrChromaBlock block;
  (void)lrc2::lrc_block_context(core, 0, 0, 0, &block);
  (void)lrc2::lrc_count_units(64, 64);
  (void)lrc2::lrc_plane_width(8, 1);
  (void)lrc2::lrc_plane_height(8, 1);
  (void)lrc2::lrc_present_coeffs(2);
  if (core.mi_cols > 0 && core.mi_rows > 0) {
    (void)lrc2::lrc_source_sample<unsigned char>(core, block, 0, 0, 0);
    (void)lrc2::lrc_source_sample<unsigned short>(core, block, 0, 0, 0);
    (void)lrc2::lrc_luma_sample<unsigned short>(core, block, 0, 0);
  }
  lrc2::lrc_resolve_plane(units, count, resolved, active, bank);
}

void test_required_bytes() {
  const struct {
    const char* name;
    lrc2::LrChromaConfig config;
  } cases[] = {
      {"720p  4:2:0 unit 64",
       lrc2::LrChromaConfig{320, 180, 1, 1, 8, 64, 0, 4}},
      {"1080p 4:2:0 unit 32",
       lrc2::LrChromaConfig{480, 270, 1, 1, 8, 32, 0, 4}},
      {"1080p 4:2:0 unit 64",
       lrc2::LrChromaConfig{480, 270, 1, 1, 8, 64, 0, 4}},
      {"2160p 4:2:0 unit 32",
       lrc2::LrChromaConfig{960, 540, 1, 1, 10, 32, 0, 4}},
      {"2160p 4:2:0 unit 64",
       lrc2::LrChromaConfig{960, 540, 1, 1, 10, 64, 0, 4}},
      {"2160p 4:2:0 unit 128",
       lrc2::LrChromaConfig{960, 540, 1, 1, 10, 128, 0, 4}},
      {"2160p 4:4:4 unit 64",
       lrc2::LrChromaConfig{960, 540, 0, 0, 10, 64, 0, 4}},
      {"1080p 4:2:2 unit 64",
       lrc2::LrChromaConfig{480, 270, 1, 0, 8, 64, 0, 4}},
      {"1080p 4:4:4 unit 32",
       lrc2::LrChromaConfig{480, 270, 0, 0, 8, 32, 0, 4}},
      {"2160p 4:2:2 unit 64",
       lrc2::LrChromaConfig{960, 540, 1, 0, 10, 64, 0, 4}},
      {"2160p 4:4:4 unit 128",
       lrc2::LrChromaConfig{960, 540, 0, 0, 10, 128, 0, 4}},
  };
  for (const auto& c : cases) {
    const std::size_t bytes = lrc2::required_device_bytes(c.config);
    check(bytes > 0, "required_device_bytes positive");
    std::printf("  required_device_bytes(%-24s) = %zu B\n", c.name, bytes);
  }
  lrc2::LrChromaConfig bad{};
  check(lrc2::required_device_bytes(bad) == 0,
        "invalid config reports 0 bytes");
}

TestCase make_tiny_case() {
  return make_case(9, 7, 1, 1, 8, 32, 0, 4, 424242);
}


void test_validation(Arena& arena) {
  const lrc2::LrChromaConfig invalid{};
  check(lrc2::required_device_bytes(invalid) == 0,
        "empty geometry reports 0 required bytes");
  lrc2::LrChromaConfig bad = invalid;
  bad.mi_cols = 10;
  bad.mi_rows = 10;
  bad.bit_depth = 12;
  check(lrc2::required_device_bytes(bad) == 0, "rejects 12-bit config");
  bad.bit_depth = 8;
  bad.unit_size = 512;
  check(lrc2::required_device_bytes(bad) == 0, "rejects unit size 512");
  bad.unit_size = 64;
  bad.sub_x = 2;
  check(lrc2::required_device_bytes(bad) == 0, "rejects sub_x 2");

  lrc2::LrChromaConfig cfg = invalid;
  cfg.mi_cols = 10;
  cfg.mi_rows = 10;
  cfg.unit_size = 32;
  check(lrc2::required_device_bytes(cfg) ==
            lrc2::required_device_bytes(
                lrc2::LrChromaConfig{10, 10, 1, 1, 10, 32, 0, 4}),
        "layout does not depend on bit depth");

  lrc2::LrChromaWorkspace empty;
  check(lrc2::resolve_units(empty) == Status::kNotInitialized,
        "resolve on an empty workspace is kNotInitialized");
  lrc2::LrChromaFrame frame{};
  check(lrc2::apply(empty, frame) == Status::kNotInitialized,
        "apply on an empty workspace is kNotInitialized");
  check(lrc2::upload_units(empty, 0, nullptr, 0) == Status::kNotInitialized,
        "upload on an empty workspace is kNotInitialized");

  const std::uint64_t mark = arena.mark();
  TestCase tiny = make_tiny_case();
  lrc2::LrChromaWorkspace ws;
  check(is_ok(lrc2::allocate_checked(arena, tiny.config, &ws)),
        "validation workspace init");
  const int count = case_unit_count(tiny.config);
  check(lrc2::upload_units(ws, 0, tiny.units[0].data(), count - 1) ==
            Status::kInvalidArgument,
        "upload rejects a short count");
  check(lrc2::upload_units(ws, 0, nullptr, count) == Status::kInvalidArgument,
        "upload rejects null host units");
  check(lrc2::upload_units(ws, 2, tiny.units[0].data(), count) ==
            Status::kInvalidArgument,
        "upload rejects a non-chroma plane");

  DeviceCase d = allocate_device_case(arena, tiny);
  const int luma_width = tiny.config.mi_cols * 4;
  const std::size_t luma_bytes =
      static_cast<std::size_t>(luma_width) * tiny.config.mi_rows * 4;
  void* luma_curr = alloc_arena(arena, luma_bytes, "validation luma curr");
  void* luma_cdef = alloc_arena(arena, luma_bytes, "validation luma cdef");
  std::vector<std::uint8_t> zero_luma(luma_bytes, 0);
  check(upload_bytes(luma_curr, zero_luma.data(), luma_bytes),
        "validation upload luma curr");
  check(upload_bytes(luma_cdef, zero_luma.data(), luma_bytes),
        "validation upload luma cdef");
  lrc2::LrChromaFrame f = make_device_frame(tiny, d);
  f.luma.curr = luma_curr;
  f.luma.cdef = luma_cdef;
  f.luma.curr_stride = luma_width;
  f.luma.cdef_stride = luma_width;
  check(is_ok(lrc2::apply(ws, f)), "validation baseline apply");
  lrc2::LrChromaFrame bad_frame = f;
  bad_frame.bit_depth = 10;
  check(lrc2::apply(ws, bad_frame) == Status::kInvalidArgument,
        "apply rejects a bit-depth mismatch");
  bad_frame = f;
  bad_frame.chroma[0].dst = const_cast<void*>(bad_frame.chroma[0].cdef);
  check(lrc2::apply(ws, bad_frame) == Status::kInvalidArgument,
        "apply rejects dst aliasing cdef");
  bad_frame = f;
  bad_frame.chroma[0].dst_stride = 1;
  check(lrc2::apply(ws, bad_frame) == Status::kInvalidArgument,
        "apply rejects a short destination stride");
  bad_frame = f;
  bad_frame.disable_loopfilters_across_tiles = 1;
  check(lrc2::apply(ws, bad_frame) == Status::kInvalidArgument,
        "apply rejects missing tile grids");
  arena.rewind(mark);
}

void test_arena(Arena& arena, std::uint64_t* peak_before_probe) {
  const std::uint64_t base = arena.mark();

  TestCase tiny = make_tiny_case();
  const std::uint64_t workspace_mark = arena.mark();
  lrc2::LrChromaWorkspace ws;
  std::size_t required = 0;
  check(is_ok(lrc2::allocate_checked(arena, tiny.config, &ws, &required)),
        "arena workspace init");
  for (int chroma = 0; chroma < 2; ++chroma) {
    check(is_ok(lrc2::upload_units(ws, chroma, tiny.units[chroma].data(),
                                   case_unit_count(tiny.config))),
          "arena upload units");
  }
  check(is_ok(lrc2::resolve_units(ws)), "arena resolve");

  DeviceCase d = allocate_device_case(arena, tiny);
  const int luma_width = tiny.config.mi_cols * 4;
  const int luma_height = tiny.config.mi_rows * 4;
  const std::size_t luma_bytes =
      static_cast<std::size_t>(luma_width) * luma_height;
  void* luma_curr = alloc_arena(arena, luma_bytes, "arena luma curr");
  void* luma_cdef = alloc_arena(arena, luma_bytes, "arena luma cdef");
  check(upload_bytes(luma_curr, tiny.luma_curr.u8.data(), luma_bytes),
        "arena upload luma curr");
  check(upload_bytes(luma_cdef, tiny.luma_cdef.u8.data(), luma_bytes),
        "arena upload luma cdef");
  lrc2::LrChromaFrame frame = make_device_frame(tiny, d);
  frame.luma.curr = luma_curr;
  frame.luma.cdef = luma_cdef;
  frame.luma.curr_stride = luma_width;
  frame.luma.cdef_stride = luma_width;

  const std::uint64_t after_setup = arena.used();
  const std::uint64_t launch_mark = arena.mark();
  check(is_ok(lrc2::apply(ws, frame)), "arena launch 1");
  check(is_ok(lrc2::apply(ws, frame)), "arena launch 2");
  check(sync_ok(), "arena sync");
  check(arena.used() == after_setup,
        "launchers do not allocate (used unchanged)");
  check(arena.used() == launch_mark,
        "mark/rewind around two launches has no net growth");
  const std::uint64_t peak = arena.peak_used();
  check(is_ok(lrc2::apply(ws, frame)), "arena launch 3");
  check(sync_ok(), "arena sync 3");
  check(arena.peak_used() == peak, "peak does not grow on repeat launches");

  // Re-init after rewinding past the first workspace reuses the same bump
  // region and addresses.
  arena.rewind(workspace_mark);
  lrc2::LrChromaWorkspace second;
  std::size_t second_required = 0;
  check(is_ok(lrc2::allocate_checked(arena, tiny.config, &second,
                                     &second_required)),
        "arena re-init");
  check(second_required == required, "required bytes are stable");
  check(second.units(0) == ws.units(0),
        "rewind reuses the same persistent addresses");
  check(arena.used() == workspace_mark + required,
        "re-init consumes exactly required_device_bytes");
  arena.rewind(workspace_mark);

  // One byte short.  Fill the arena, then rewind to an exact bump offset that
  // leaves required - 1 bytes (allocations are 256-byte rounded).  Record the
  // peak before the filler so the final line reports the test's real peak.
  *peak_before_probe = arena.peak_used();
  const std::uint64_t short_mark = arena.mark();
  const std::uint64_t entry_remaining = arena.remaining();
  check(entry_remaining >= required, "arena has room for the short test");
  if (entry_remaining > required - 1) {
    void* filler =
        arena.allocate(static_cast<std::size_t>(entry_remaining - (required - 1)));
    check(filler != nullptr, "arena filler allocation");
  }
  const std::uint64_t short_target =
      arena.usable_capacity() - (required - 1);
  check(short_target <= arena.used(), "short setup can rewind back");
  if (short_target <= arena.used()) {
    check(arena.rewind(short_target), "rewind to one byte short");
  }
  check(arena.remaining() == required - 1, "arena prepared one byte short");
  const std::uint64_t used_before_short = arena.used();

  lrc2::LrChromaWorkspace out;
  std::size_t reported = 0;
  std::string error;
  const Status short_status =
      lrc2::allocate_checked(arena, tiny.config, &out, &reported, &error);
  check(short_status == lrc2::kArenaTooSmall,
        "one-byte-short returns kArenaTooSmall");
  check(reported == required, "short status reports required bytes");
  check(!out.initialized(), "short init leaves output empty");
  check(arena.used() == used_before_short,
        "failed init does not consume arena space");
  arena.rewind(short_mark);
  arena.rewind(base);
}

}  // namespace

int main() {
  std::printf("AV2 chroma LR CUDA parity test\n");
  Arena arena;
  const Status init = arena.init_mb(96);
  if (!is_ok(init)) {
    std::printf("FAIL: Arena::init_mb(96): %s\n",
                av2enc::cuda::status_name(init));
    return 1;
  }
  std::printf("  arena capacity: %llu B\n",
              static_cast<unsigned long long>(arena.capacity()));

  test_required_bytes();

  const std::vector<TestCase> cases = build_cases();
  std::printf("  cases: %zu\n", cases.size());
  for (const TestCase& c : cases) {
    run_gpu_case(arena, c);
  }

  test_validation(arena);
  std::uint64_t peak_before_probe = 0;
  test_arena(arena, &peak_before_probe);

  // Runtime device-visibility probe (compiles and launches the shared header
  // entry points; no output is consumed).
  const std::uint64_t mark = arena.mark();
  std::vector<lrc::LrChromaUnit> units(1);
  std::vector<std::int16_t> resolved(18, 0);
  std::vector<std::uint8_t> active(1, 0);
  lrc2::LrChromaUnit* d_units = static_cast<lrc2::LrChromaUnit*>(
      arena.allocate(sizeof(lrc2::LrChromaUnit)));
  std::int16_t* d_resolved =
      static_cast<std::int16_t*>(arena.allocate(18 * sizeof(std::int16_t)));
  std::uint8_t* d_active =
      static_cast<std::uint8_t*>(arena.allocate(1));
  lrc2::LrChromaBankState* d_bank =
      static_cast<lrc2::LrChromaBankState*>(
          arena.allocate(sizeof(lrc2::LrChromaBankState)));
  check(d_units != nullptr && d_resolved != nullptr && d_active != nullptr &&
            d_bank != nullptr,
        "probe allocations");
  check(upload_bytes(d_units, units.data(), sizeof(lrc2::LrChromaUnit)),
        "probe upload units");
  lrc2::LrChromaCore core{};
  visibility_probe_kernel<<<1, 1>>>(core, d_units, 1, d_resolved, d_active,
                                    d_bank);
  check(cudaGetLastError() == cudaSuccess, "visibility probe launch");
  check(sync_ok(), "visibility probe sync");
  check(download_bytes(resolved.data(), d_resolved,
                       resolved.size() * sizeof(std::int16_t)),
        "probe download resolved");
  check(download_bytes(active.data(), d_active, active.size()),
        "probe download active");
  check(active[0] == 0 && resolved[0] == 0, "probe resolver result");
  arena.rewind(mark);

  std::printf(
      "PASS: %d checks, %d failures (%lld samples compared; "
      "chroma_apply oracle cases %d, resolved units %d), "
      "arena peak %llu B (before sizing probe; capacity %llu B)\n",
      g_checks, g_failures, g_samples, g_cpu_oracle_cases, g_resolver_units,
      static_cast<unsigned long long>(peak_before_probe),
      static_cast<unsigned long long>(arena.capacity()));
  return g_failures == 0 ? 0 : 1;
}

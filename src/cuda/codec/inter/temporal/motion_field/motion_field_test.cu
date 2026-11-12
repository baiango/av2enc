// Real-GPU parity test for the AV2 temporal motion-field kernels.
//
// Every launch uses one persistent av2enc::cuda::Arena.  The test:
//   * checks required_device_bytes(), allocate_checked(), mark/rewind;
//   * proves two consecutive launches do not grow used() or peak_used();
//   * proves an arena one byte short returns kArenaTooSmall with the exact
//     requirement and leaves the bump pointer untouched;
//   * compares §7.22 storage, §7.9 projection + trajectory fill, §7.12.2.21
//     bank update/reset and §7.12.2.7/.8/.19 merge-stack generation with the
//     CPU reference over randomized fields, hole patterns, project steps,
//     superblock sizes, reference distances and resolutions from 4x4 MI to
//     1920x1080.
//
// Usage: ./motion_field_test [--quick]

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cuda/arena.h"
#include "cuda/codec/inter/temporal/motion_field/motion_field.h"
#include "cuda/codec/inter/temporal/motion_field/motion_field_cpu_oracle.h"
#include "cuda/codec/inter/temporal/motion_field/motion_field_device.cuh"
#include "codec/inter/temporal/motion_field.h"
#include "codec/inter/temporal/ref_mv_bank.h"
#include "codec/inter/temporal/temporal_math.h"

namespace {

namespace dev = av2enc::cuda::motion_field;
namespace cpu = av2enc::inter::temporal;
namespace av2cuda = av2enc::cuda;
using Arena = av2enc::cuda::Arena;

int g_checks = 0;
int g_failures = 0;
int g_merge_reported = 0;

void Check(bool condition, const char* what) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    if (g_failures <= 30) {
      std::fprintf(stderr, "FAIL: %s\n", what);
    }
  }
}

bool CudaOk(cudaError_t result, const char* what) {
  Check(result == cudaSuccess, what);
  if (result != cudaSuccess) {
    std::fprintf(stderr, "  CUDA: %s\n", cudaGetErrorString(result));
    return false;
  }
  return true;
}

struct Rng {
  std::uint64_t state = 1;
  explicit Rng(std::uint64_t seed) : state(seed == 0 ? 1 : seed) {}
  std::uint32_t next() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return static_cast<std::uint32_t>(state >> 32);
  }
  int range(int low, int high) {
    return low + static_cast<int>(next() %
                                  static_cast<std::uint32_t>(high - low + 1));
  }
  bool coin() { return (next() & 1u) != 0; }
};

template <typename T>
T* Upload(Arena& arena, const std::vector<T>& host, const char* what) {
  if (host.empty()) return nullptr;
  T* ptr = arena.allocate_array<T>(host.size());
  Check(ptr != nullptr, what);
  if (ptr == nullptr) return nullptr;
  CudaOk(cudaMemcpy(ptr, host.data(), host.size() * sizeof(T),
                    cudaMemcpyHostToDevice),
         what);
  return ptr;
}

template <typename T>
bool Download(const T* device, std::vector<T>* host, const char* what) {
  if (host->empty()) return true;
  return CudaOk(cudaMemcpy(host->data(), device, host->size() * sizeof(T),
                           cudaMemcpyDeviceToHost),
                what);
}

struct HostField {
  int w8 = 0;
  int h8 = 0;
  std::vector<std::int32_t> mv;
  std::vector<std::uint8_t> valid;
  std::vector<std::int32_t> offset;
  HostField(int w, int h) : w8(w), h8(h) {
    mv.assign(static_cast<std::size_t>(w) * h * 2, 0);
    valid.assign(static_cast<std::size_t>(w) * h, 0);
    offset.assign(static_cast<std::size_t>(w) * h, 0);
  }
  cpu::MotionField Cpu() {
    cpu::MotionField f;
    f.w8 = w8;
    f.h8 = h8;
    f.mv = mv.data();
    f.valid = valid.data();
    f.offset = offset.data();
    return f;
  }
};

struct HostTraj {
  int w8 = 0;
  int h8 = 0;
  int num_refs = 0;
  std::vector<std::int32_t> mv;
  std::vector<std::uint8_t> valid;
  std::vector<std::int16_t> pos;
  std::vector<std::uint8_t> pos_valid;
  HostTraj(int w, int h, int refs) : w8(w), h8(h), num_refs(refs) {
    const std::size_t cells = static_cast<std::size_t>(w) * h;
    mv.assign(cells * refs * 2, 0);
    valid.assign(cells * refs, 0);
    pos.assign(cells * refs * 3 * 2, 0);
    pos_valid.assign(cells * refs * 3, 0);
  }
  dev::MfTraj Device() {
    dev::MfTraj t;
    t.w8 = w8;
    t.h8 = h8;
    t.num_refs = num_refs;
    t.mv = mv.data();
    t.valid = valid.data();
    t.pos = pos.data();
    t.pos_valid = pos_valid.data();
    return t;
  }
  cpu::TrajectoryField Cpu() {
    cpu::TrajectoryField t;
    t.w8 = w8;
    t.h8 = h8;
    t.num_refs = num_refs;
    t.mv = mv.data();
    t.valid = valid.data();
    t.pos = pos.data();
    t.pos_valid = pos_valid.data();
    return t;
  }
};

struct HostSaved {
  int w8 = 0;
  int h8 = 0;
  std::vector<std::int16_t> mv;
  std::vector<std::int8_t> ref;
  HostSaved(int w, int h) : w8(w), h8(h) {
    mv.assign(static_cast<std::size_t>(w) * h * 4, 0);
    ref.assign(static_cast<std::size_t>(w) * h * 2, cpu::kRefNone);
  }
};

// Fill a saved field with a selectable hole pattern.
void FillSaved(HostSaved* s, Rng* rng, int pattern, int max_dist) {
  const int w8 = s->w8;
  const int h8 = s->h8;
  for (int y = 0; y < h8; ++y) {
    for (int x = 0; x < w8; ++x) {
      const std::size_t pos =
          static_cast<std::size_t>(y) * w8 + static_cast<std::size_t>(x);
      bool hole = false;
      if (pattern == 0) {
        hole = rng->range(0, 4) == 0;
      } else if (pattern == 1) {
        hole = ((x + y) & 1) != 0;
      } else if (pattern == 2) {
        hole = (y % 3) == 0;
      } else {
        hole = ((x / 3 + y / 2) % 4) == 0;
      }
      for (int list = 0; list < 2; ++list) {
        if (hole) {
          s->ref[pos * 2 + list] = cpu::kRefNone;
          s->mv[(pos * 2 + list) * 2 + 0] = 0;
          s->mv[(pos * 2 + list) * 2 + 1] = 0;
        } else {
          s->ref[pos * 2 + list] =
              static_cast<std::int8_t>(rng->range(0, 1));
          const int range = max_dist * 96;
          s->mv[(pos * 2 + list) * 2 + 0] = cpu::compression_mv(
              rng->range(-range, range));
          s->mv[(pos * 2 + list) * 2 + 1] = cpu::compression_mv(
              rng->range(-range, range));
        }
      }
    }
  }
}

struct TestConfig {
  int mi_cols = 0;
  int mi_rows = 0;
  int proj_step = 1;
  int sb_64x64 = 1;
  int max_blocks = 1;
  int max_stack_blocks = 128;
  int max_projection_jobs = 8;
  int max_bank_ops = 128;
  int max_bank_rows = 64;
  int num_refs = 2;
};

dev::MfConfig ToMfConfig(const TestConfig& c) {
  dev::MfConfig m;
  m.mi_cols = c.mi_cols;
  m.mi_rows = c.mi_rows;
  m.num_refs = c.num_refs;
  m.proj_step = c.proj_step;
  m.sb_64x64 = c.sb_64x64;
  m.max_blocks = c.max_blocks;
  m.max_stack_blocks = c.max_stack_blocks;
  m.max_projection_jobs = c.max_projection_jobs;
  m.max_bank_ops = c.max_bank_ops;
  m.max_bank_rows = c.max_bank_rows;
  return m;
}

void ReportOwner(const char* side, int owner,
                 const std::vector<dev::MfBlockStoreInput>& blocks, int count) {
  if (owner < 0 || owner >= count) {
    std::fprintf(stderr, "  %s owner=%d (none)\n", side, owner);
    return;
  }
  const dev::MfBlockStoreInput& b = blocks[static_cast<std::size_t>(owner)];
  std::fprintf(stderr,
               "  %s owner=%d: r=%d c=%d bw4=%d bh4=%d mvMethod=%d wedge=%d "
               "tip=%d enable=%d\n",
               side, owner, b.r, b.c, b.bw4, b.bh4, b.mv_method, b.is_wedge,
               b.tip_pred, b.enable_ref_frame_mvs);
}

void ReportStorageDiff(const char* what, const std::vector<std::int8_t>& want,
                       const std::vector<std::int8_t>& got, int w8,
                       const std::vector<dev::MfBlockStoreInput>& blocks,
                       int count, const std::vector<int>& host_stamp,
                       const std::vector<int>& dev_stamp,
                       const std::vector<std::uint8_t>& cpu_written) {
  for (std::size_t i = 0; i < want.size(); ++i) {
    if (want[i] == got[i]) continue;
    const int cell = static_cast<int>(i / 2);
    const int list = static_cast<int>(i % 2);
    const int y8 = cell / w8;
    const int x8 = cell % w8;
    const int cell_index = static_cast<int>(i / 2);
    std::fprintf(stderr,
                 "  %s first diff: y8=%d x8=%d list=%d cpu_written=%d "
                 "host_stamp=%d dev_stamp=%d expected=%d got=%d\n",
                 what, y8, x8, list,
                 static_cast<int>(cpu_written[static_cast<std::size_t>(
                     cell_index)]),
                 host_stamp[static_cast<std::size_t>(cell_index)],
                 dev_stamp[static_cast<std::size_t>(cell_index)],
                 static_cast<int>(want[i]), static_cast<int>(got[i]));
    ReportOwner("cpu", host_stamp[static_cast<std::size_t>(cell_index)], blocks,
                count);
    ReportOwner("gpu", dev_stamp[static_cast<std::size_t>(cell_index)], blocks,
                count);
    return;
  }
}

void ReportStorageMvDiff(const std::vector<std::int16_t>& want,
                         const std::vector<std::int16_t>& got, int w8,
                         const std::vector<dev::MfBlockStoreInput>& blocks,
                         int count, const std::vector<int>& host_stamp,
                         const std::vector<int>& dev_stamp,
                         const std::vector<std::uint8_t>& cpu_written) {
  for (std::size_t i = 0; i < want.size(); ++i) {
    if (want[i] == got[i]) continue;
    const int cell = static_cast<int>(i / 4);
    const int rem = static_cast<int>(i % 4);
    const int list = rem / 2;
    const int comp = rem % 2;
    const int y8 = cell / w8;
    const int x8 = cell % w8;
    const int cell_index = static_cast<int>(i / 4);
    std::fprintf(stderr,
                 "  storage mv first diff: y8=%d x8=%d list=%d comp=%d "
                 "cpu_written=%d host_stamp=%d dev_stamp=%d expected=%d "
                 "got=%d\n",
                 y8, x8, list, comp,
                 static_cast<int>(cpu_written[static_cast<std::size_t>(
                     cell_index)]),
                 host_stamp[static_cast<std::size_t>(cell_index)],
                 dev_stamp[static_cast<std::size_t>(cell_index)],
                 static_cast<int>(want[i]), static_cast<int>(got[i]));
    ReportOwner("cpu", host_stamp[static_cast<std::size_t>(cell_index)], blocks,
                count);
    ReportOwner("gpu", dev_stamp[static_cast<std::size_t>(cell_index)], blocks,
                count);
    return;
  }
}

// ---------------------------------------------------------------------------
// §7.22 storage parity
// ---------------------------------------------------------------------------

void RunStorage(Arena& arena, const dev::MfDeviceBuffers& buffers,
                const TestConfig& cfg, Rng* rng) {
  const int mi_cols = cfg.mi_cols;
  const int mi_rows = cfg.mi_rows;
  const int w8 = mi_cols >> 1;
  const int h8 = mi_rows >> 1;
  const std::size_t cells = static_cast<std::size_t>(w8) * h8;

  std::vector<std::int32_t> mvs(
      static_cast<std::size_t>(mi_rows) * mi_cols * 4, 0);
  for (auto& v : mvs) v = rng->range(-2300, 2300);
  std::vector<std::int32_t> refine(64 * 64 * 4, 0);
  std::vector<std::int32_t> deltas(64 * 64 * 4, 0);
  for (auto& v : refine) v = rng->range(-1800, 1800);
  for (auto& v : deltas) v = rng->range(-40, 40);
  std::vector<std::uint8_t> mask(64 * 64, 0);
  for (auto& v : mask) v = static_cast<std::uint8_t>(rng->range(0, 255));
  std::vector<int> row_base(static_cast<std::size_t>(mi_rows) * mi_cols, 0);
  std::vector<int> col_base(static_cast<std::size_t>(mi_rows) * mi_cols, 0);
  std::vector<int> sizes16(static_cast<std::size_t>(mi_rows) * mi_cols, 0);
  for (std::size_t i = 0; i < row_base.size(); ++i) {
    row_base[i] = rng->range(0, mi_rows - 1);
    col_base[i] = rng->range(0, mi_cols - 1);
    sizes16[i] = rng->coin() ? 1 : 0;
  }
  std::vector<std::int32_t> field_mv(static_cast<std::size_t>(w8) * h8 * 2, 0);
  std::vector<std::uint8_t> field_valid(static_cast<std::size_t>(w8) * h8, 0);
  std::vector<std::int32_t> field_offset(static_cast<std::size_t>(w8) * h8, 0);
  for (std::size_t i = 0; i < field_valid.size(); ++i) {
    field_valid[i] = rng->range(0, 3) != 0 ? 1 : 0;
    field_mv[i * 2 + 0] = rng->range(-1200, 1200);
    field_mv[i * 2 + 1] = rng->range(-1200, 1200);
    field_offset[i] = rng->range(-8, 8);
  }
  dev::MfField host_field;
  host_field.w8 = w8;
  host_field.h8 = h8;
  host_field.mv = field_mv.data();
  host_field.valid = field_valid.data();
  host_field.offset = field_offset.data();

  // Test scratch from the same arena, then rewind after the comparison.
  const std::uint64_t mark = dev::motion_field_mark(arena);
  std::int32_t* d_mvs = Upload(arena, mvs, "upload mvs");
  std::int32_t* d_refine = Upload(arena, refine, "upload refine");
  std::int32_t* d_deltas = Upload(arena, deltas, "upload deltas");
  std::uint8_t* d_mask = Upload(arena, mask, "upload mask");
  int* d_row_base = Upload(arena, row_base, "upload row base");
  int* d_col_base = Upload(arena, col_base, "upload col base");
  int* d_sizes16 = Upload(arena, sizes16, "upload tip sizes");
  std::int32_t* d_field_mv = Upload(arena, field_mv, "upload field mv");
  std::uint8_t* d_field_valid = Upload(arena, field_valid, "upload field valid");
  std::int32_t* d_field_offset =
      Upload(arena, field_offset, "upload field offset");
  dev::MfField d_field;
  d_field.w8 = w8;
  d_field.h8 = h8;
  d_field.mv = d_field_mv;
  d_field.valid = d_field_valid;
  d_field.offset = d_field_offset;

  // Partition the frame with nominal block sizes; the edge block overhangs.
  const int sizes[] = {1, 2, 4, 8, 16, 32};
  std::vector<dev::MfBlockStoreInput> host_blocks;
  std::vector<dev::MfBlockStoreInput> dev_blocks;
  int mv_method0 = 0;
  int mv_method1 = 0;
  int wedge_blocks = 0;
  int tip_blocks = 0;
  int r = 0;
  while (r < mi_rows) {
    const int bh4 = sizes[rng->range(0, 5)];
    int c = 0;
    while (c < mi_cols) {
      const int bw4 = sizes[rng->range(0, 5)];
      dev::MfBlockStoreInput in;
      in.r = r;
      in.c = c;
      in.bw4 = bw4;
      in.bh4 = bh4;
      in.mi_cols = mi_cols;
      in.mi_rows = mi_rows;
      in.mv_method = rng->coin() ? 0 : 1;
      if (in.mv_method == 0) {
        ++mv_method0;
      } else {
        ++mv_method1;
      }
      const bool tip = rng->range(0, 6) == 0;
      if (tip) ++tip_blocks;
      in.refs[0] = tip ? cpu::kTipFrame : rng->range(-1, 6);
      in.refs[1] = rng->range(-1, 6);
      if (rng->coin() && !tip) in.refs[1] = in.refs[0];
      in.ref_order_hint[0] = rng->range(-1, 10);
      in.ref_order_hint[1] = rng->range(-1, 10);
      in.order_hint = rng->range(0, 10);
      in.is_wedge = rng->coin();
      if (in.is_wedge) ++wedge_blocks;
      in.mask = mask.data();
      in.mask_stride = 64;
      in.src.mvs = mvs.data();
      in.src.mi_cols = mi_cols;
      in.src.mi_rows = mi_rows;
      in.src.delta_stride4 = 2 * bw4;
      in.src.motion_mode = rng->range(0, 3);
      in.src.force_integer_mv = rng->range(0, 3) == 0;
      in.src.is_global_mv_cand = rng->range(0, 3) == 0;
      in.src.use_refinemv = rng->coin();
      in.warp_mv[0] =
          dev::Mv{rng->range(-900, 900), rng->range(-900, 900)};
      in.warp_mv[1] =
          dev::Mv{rng->range(-900, 900), rng->range(-900, 900)};
      in.global_mv[0] =
          dev::Mv{rng->range(-900, 900), rng->range(-900, 900)};
      in.global_mv[1] =
          dev::Mv{rng->range(-900, 900), rng->range(-900, 900)};
      in.src.refine_mvs = refine.data();
      in.src.mv_deltas = deltas.data();
      in.tip_pred = in.refs[0] == cpu::kTipFrame;
      in.tip_frame_as_output = rng->coin();
      in.tip_global_wtd_index = rng->range(0, 7);
      in.closest_past = rng->range(0, 6);
      in.closest_future = rng->range(0, 6);
      in.tip_past_order_hint = rng->range(-1, 10);
      in.tip_future_order_hint = rng->range(-1, 10);
      in.num_past_refs = rng->range(0, 2);
      in.num_future_refs = rng->range(0, 2);
      in.field = host_field;
      in.mi_row_base = row_base.data();
      in.mi_col_base = col_base.data();
      in.tip_sizes16 = sizes16.data();
      host_blocks.push_back(in);

      dev::MfBlockStoreInput d = in;
      d.src.mvs = d_mvs;
      d.src.refine_mvs = d_refine;
      d.src.mv_deltas = d_deltas;
      d.mask = d_mask;
      d.field = d_field;
      d.mi_row_base = d_row_base;
      d.mi_col_base = d_col_base;
      d.tip_sizes16 = d_sizes16;
      dev_blocks.push_back(d);
      c += bw4;
    }
    r += bh4;
  }
  Check(static_cast<int>(host_blocks.size()) <= cfg.max_blocks,
        "block capacity");
  const int count = std::min(static_cast<int>(dev_blocks.size()),
                             cfg.max_blocks);
  // Zero the storage output so untouched cells (a truncated partition) match
  // the CPU oracle's zero-initialised arrays.
  CudaOk(cudaMemset(buffers.saved.ref, 0, cells * 2), "zero saved ref");
  CudaOk(cudaMemset(buffers.saved.mv, 0, cells * 4 * sizeof(std::int16_t)),
         "zero saved mv");
  if (count > 0) {
    Check(dev::upload_blocks(buffers, dev_blocks.data(), count) ==
              dev::MfStatus::kOk,
          "upload blocks");
    Check(dev::store_blocks(buffers, count) == dev::MfStatus::kOk,
          "store blocks launch");
    CudaOk(cudaDeviceSynchronize(), "store blocks sync");
  }

  // Host-side truth for the ordering stamp and for whether the CPU entry
  // predicate writes a cell at all (i8/j8 <= 7 and in-range).
  std::vector<int> host_stamp(cells, -1);
  for (int i = 0; i < count; ++i) {
    dev::mf_mark_block_cells(host_blocks[static_cast<std::size_t>(i)],
                             host_stamp.data(), i);
  }
  std::vector<int> dev_stamp(cells, 0);
  Download(buffers.saved_stamp, &dev_stamp, "download saved stamp");
  for (std::size_t c = 0; c < cells; ++c) {
    if (host_stamp[c] != dev_stamp[c]) {
      const int y8 = static_cast<int>(c) / w8;
      const int x8 = static_cast<int>(c) % w8;
      std::fprintf(stderr,
                   "  §7.22 stamp first diff: y8=%d x8=%d host=%d device=%d\n",
                   y8, x8, host_stamp[c], dev_stamp[c]);
      ReportOwner("host-stamp", host_stamp[c], host_blocks, count);
      ReportOwner("device-stamp", dev_stamp[c], host_blocks, count);
      break;
    }
  }
  Check(host_stamp == dev_stamp, "§7.22 storage stamp parity");

  // Report branch coverage and how many 8x8 cells are written by more than
  // one descriptor (the ordering hazard the stamp pass resolves).
  std::vector<int> coverage(cells, 0);
  std::vector<std::uint8_t> cpu_written(cells, 0);
  for (const dev::MfBlockStoreInput& b : host_blocks) {
    int bw8 = 0;
    int bh8 = 0;
    int cbw4 = 0;
    int cbh4 = 0;
    if (!dev::mf_block_geometry(b, &bw8, &bh8, &cbw4, &cbh4)) continue;
    const int rows = (cbh4 + 1) >> 1;
    const int cols = (cbw4 + 1) >> 1;
    for (int i8 = 0; i8 < rows; ++i8) {
      const int y8 = (b.r >> 1) + i8;
      if (y8 < 0 || y8 >= h8) continue;
      for (int j8 = 0; j8 < cols; ++j8) {
        const int x8 = (b.c >> 1) + j8;
        if (x8 < 0 || x8 >= w8) continue;
        ++coverage[static_cast<std::size_t>(y8) * w8 + x8];
        if (i8 <= 7 && j8 <= 7) {
          cpu_written[static_cast<std::size_t>(y8) * w8 + x8] = 1;
        }
      }
    }
  }
  int shared_cells = 0;
  for (int c : coverage) {
    if (c > 1) ++shared_cells;
  }
  std::printf("  storage: blocks=%zu mvMethod0=%d mvMethod1=%d wedge=%d tip=%d "
              "shared_cells=%d\n",
              host_blocks.size(), mv_method0, mv_method1, wedge_blocks,
              tip_blocks, shared_cells);

  // CPU oracle into host vectors.
  std::vector<std::int8_t> want_ref(cells * 2, 0);
  std::vector<std::int16_t> want_mv(cells * 4, 0);
  for (int i = 0; i < count; ++i) {
    dev::mf_test_cpu_store_block(host_blocks[static_cast<std::size_t>(i)],
                                 want_ref.data(), want_mv.data());
  }
  std::vector<std::int8_t> got_ref(cells * 2, 0);
  std::vector<std::int16_t> got_mv(cells * 4, 0);
  Download(buffers.saved.ref, &got_ref, "download saved ref");
  Download(buffers.saved.mv, &got_mv, "download saved mv");
  if (want_ref != got_ref) {
    ReportStorageDiff("§7.22 storage ref", want_ref, got_ref, w8,
                      host_blocks, count, host_stamp, dev_stamp, cpu_written);
  }
  if (want_mv != got_mv) {
    ReportStorageMvDiff(want_mv, got_mv, w8, host_blocks, count, host_stamp,
                        dev_stamp, cpu_written);
  }
  Check(want_ref == got_ref, "§7.22 storage ref parity");
  Check(want_mv == got_mv, "§7.22 storage mv parity");

  // Two-launch no-growth: measure immediately before the two calls.
  if (count > 0) {
    const std::uint64_t used_before = arena.used();
    const std::uint64_t peak_before = arena.peak_used();
    Check(dev::store_blocks(buffers, count) == dev::MfStatus::kOk,
          "no-growth launch 1");
    Check(dev::store_blocks(buffers, count) == dev::MfStatus::kOk,
          "no-growth launch 2");
    Check(arena.used() == used_before, "two launches do not grow used()");
    Check(arena.peak_used() == peak_before,
          "two launches do not grow peak_used()");
  }

  Check(dev::motion_field_rewind(arena, mark), "storage scratch rewind");
}

// ---------------------------------------------------------------------------
// §7.9 projection + trajectory fill parity
// ---------------------------------------------------------------------------

void RunProjection(Arena& arena, const dev::MfDeviceBuffers& buffers,
                   const TestConfig& cfg, Rng* rng, int pattern, int max_dist,
                   bool use_traj, int proj_step, bool sb64) {
  const int mi_cols = cfg.mi_cols;
  const int mi_rows = cfg.mi_rows;
  const int w8 = mi_cols >> 1;
  const int h8 = mi_rows >> 1;

  // Two past references, no saved refs: the CPU process list is
  // ref1 forward, ref0 forward, ref1 backward.
  cpu::MotionFieldConfig cc;
  cc.mi_cols = mi_cols;
  cc.mi_rows = mi_rows;
  cc.proj_step = proj_step;
  cc.sb_64x64 = sb64;
  cc.enable_tip = false;
  cc.enable_mv_traj = use_traj;
  cc.order_hint = 9;
  cc.num_total_refs = 2;
  cc.num_past_refs = 2;
  cc.num_future_refs = 0;
  cc.frame_width = mi_cols * 4;
  cc.frame_height = mi_rows * 4;
  cc.ref_order_hint[0] = 2;
  cc.ref_order_hint[1] = 7;

  HostSaved saved[2] = {HostSaved(w8, h8), HostSaved(w8, h8)};
  int saved_hints[2][cpu::kRefsPerFrame] = {};
  for (int rf = 0; rf < 2; ++rf) {
    cc.refs[rf].slot = rf;
    cc.refs[rf].type = cpu::FrameType::kInter;
    cc.refs[rf].width = cc.frame_width;
    cc.refs[rf].height = cc.frame_height;
    cc.refs[rf].order_hint = cc.ref_order_hint[rf];
    cc.refs[rf].num_saved_refs = 0;
    for (int k = 0; k < cpu::kRefsPerFrame; ++k) {
      // Exercise distance 0..6 and the restricted hint.
      const int pick = rng->range(0, 7);
      const int hint = pick == 7 ? cpu::kRestrictedOrderHint
                                 : rng->range(0, 9);
      saved_hints[rf][k] = hint;
      cc.refs[rf].saved_order_hint[k] = hint;
    }
    FillSaved(&saved[rf], rng, pattern, max_dist);
    cc.saved[rf].w8 = w8;
    cc.saved[rf].h8 = h8;
    cc.saved[rf].mv = saved[rf].mv.data();
    cc.saved[rf].ref = saved[rf].ref.data();
  }

  HostField cpu_field_buf(w8, h8);
  HostTraj cpu_traj_buf(w8, h8, 2);
  cpu::MotionField cpu_field = cpu_field_buf.Cpu();
  cpu::TrajectoryField cpu_traj = cpu_traj_buf.Cpu();
  cpu::MotionFieldResult result;
  const cpu::TemporalStatus status = cpu::motion_field_estimation(
      cc, &cpu_field, use_traj ? &cpu_traj : nullptr, &result);
  Check(status == cpu::TemporalStatus::kOk, "CPU projection setup");
  if (status != cpu::TemporalStatus::kOk) return;

  // Device saved fields and jobs.  Scratch is rewound afterwards.
  const std::uint64_t mark = dev::motion_field_mark(arena);
  dev::MfSavedField d_saved[2];
  for (int rf = 0; rf < 2; ++rf) {
    d_saved[rf].w8 = w8;
    d_saved[rf].h8 = h8;
    d_saved[rf].mv = Upload(arena, saved[rf].mv, "upload saved mv");
    d_saved[rf].ref = Upload(arena, saved[rf].ref, "upload saved ref");
  }
  Check(dev::zero_field(buffers) == dev::MfStatus::kOk, "zero field");
  CudaOk(cudaDeviceSynchronize(), "zero field sync");

  dev::MfProjectionParams jobs[3];
  const int srcs[3] = {1, 0, 1};
  const int signs[3] = {1, 1, -1};
  const int bwd[3] = {0, 0, 1};
  for (int j = 0; j < 3; ++j) {
    dev::MfProjectionParams& p = jobs[j];
    p.mi_cols = mi_cols;
    p.mi_rows = mi_rows;
    p.proj_step = proj_step;
    p.sb_64x64 = sb64 ? 1 : 0;
    p.enable_mv_traj = use_traj ? 1 : 0;
    p.num_total_refs = 2;
    p.order_hint = cc.order_hint;
    for (int rf = 0; rf < cpu::kRefsPerFrame; ++rf) {
      p.ref_order_hint[rf] = cc.ref_order_hint[rf];
    }
    p.src = srcs[j];
    p.src_order_hint = cc.ref_order_hint[srcs[j]];
    p.num_saved = 0;
    p.dst_sign = signs[j];
    p.is_bwd = bwd[j];
    p.target_frame = cpu::kRefNone;
    p.saved = d_saved[srcs[j]];
    p.field = buffers.field;
    p.traj = buffers.traj;
    for (int k = 0; k < cpu::kRefsPerFrame; ++k) {
      p.saved_order_hint[k] = saved_hints[srcs[j]][k];
    }
  }
  Check(dev::upload_projection_jobs(buffers, jobs, 3) == dev::MfStatus::kOk,
        "upload projection jobs");
  Check(dev::project(buffers, 3) == dev::MfStatus::kOk, "project launch");
  if (use_traj && buffers.traj.num_refs > 0) {
    Check(dev::traj_fill(buffers) == dev::MfStatus::kOk, "traj fill launch");
  }
  CudaOk(cudaDeviceSynchronize(), "projection sync");

  HostField got_field(w8, h8);
  Download(buffers.field.mv, &got_field.mv, "download field mv");
  Download(buffers.field.valid, &got_field.valid, "download field valid");
  Download(buffers.field.offset, &got_field.offset, "download field offset");
  Check(got_field.mv == cpu_field_buf.mv, "§7.9 field mv parity");
  Check(got_field.valid == cpu_field_buf.valid, "§7.9 field valid parity");
  Check(got_field.offset == cpu_field_buf.offset, "§7.9 field offset parity");

  if (use_traj && buffers.traj.num_refs > 0) {
    HostTraj got_traj(w8, h8, 2);
    Download(buffers.traj.mv, &got_traj.mv, "download traj mv");
    Download(buffers.traj.valid, &got_traj.valid, "download traj valid");
    Download(buffers.traj.pos, &got_traj.pos, "download traj pos");
    Download(buffers.traj.pos_valid, &got_traj.pos_valid,
             "download traj pos_valid");
    Check(got_traj.mv == cpu_traj_buf.mv, "§7.9 traj mv parity");
    Check(got_traj.valid == cpu_traj_buf.valid, "§7.9 traj valid parity");
    Check(got_traj.pos == cpu_traj_buf.pos, "§7.9 traj pos parity");
    Check(got_traj.pos_valid == cpu_traj_buf.pos_valid,
          "§7.9 traj pos_valid parity");
  }

  Check(dev::motion_field_rewind(arena, mark), "projection scratch rewind");
}

// ---------------------------------------------------------------------------
// §7.12.2 bank operations parity
// ---------------------------------------------------------------------------

void RunBankOps(Arena& arena, const dev::MfDeviceBuffers& buffers, Rng* rng) {
  (void)arena;
  const int ops = rng->range(1, 32);
  const int rows = rng->range(2, 32);
  std::vector<cpu::RefMvBankRowSource> host_rows(static_cast<std::size_t>(rows));
  for (int i = 0; i < rows; ++i) {
    host_rows[i].is_inter = rng->range(0, 3) != 0 ? 1 : 0;
    host_rows[i].ref[0] = static_cast<std::int8_t>(rng->range(-1, 6));
    host_rows[i].ref[1] = static_cast<std::int8_t>(rng->range(-1, 6));
    host_rows[i].mv[0][0] = rng->range(-300, 300);
    host_rows[i].mv[0][1] = rng->range(-300, 300);
    host_rows[i].mv[1][0] = rng->range(-300, 300);
    host_rows[i].mv[1][1] = rng->range(-300, 300);
    host_rows[i].cwp_idx = rng->range(0, 8);
    host_rows[i].motion_mode = static_cast<std::uint8_t>(rng->range(0, 3));
    host_rows[i].bw4 = rng->range(1, 4);
  }
  std::vector<dev::MfBankOp> host_ops(static_cast<std::size_t>(ops));
  for (int i = 0; i < ops; ++i) {
    dev::MfBankOp& op = host_ops[static_cast<std::size_t>(i)];
    op.kind = rng->range(0, 2);
    op.mi_row = rng->range(0, 32);
    op.mi_col = rng->range(0, 32);
    op.bw4 = rng->range(1, 8);
    op.bh4 = rng->range(1, 8);
    op.sb_size4 = rng->coin() ? 16 : 32;
    op.chroma_part = rng->coin() ? 1 : 0;
    op.frame_is_intra = rng->coin() ? 1 : 0;
    op.top_row = rng->coin() ? 1 : 0;
    op.refs[0] = rng->range(-1, 6);
    op.refs[1] = rng->range(-1, 6);
    op.mv[0] = dev::Mv{rng->range(-400, 400), rng->range(-400, 400)};
    op.mv[1] = dev::Mv{rng->range(-400, 400), rng->range(-400, 400)};
    op.cwp_idx = rng->range(0, 8);
    op.from_within_sb = rng->coin() ? 1 : 0;
    // The row window is [row_offset, rows); using row_mi_cols = rows here
    // would read past the host_rows vector (the old crash).
    op.row_offset = rows > 0 ? rng->range(0, rows - 1) : 0;
    op.row_mi_cols = rows - op.row_offset;
  }
  cpu::RefMvBank want;
  for (int i = 0; i < ops; ++i) {
    const dev::MfBankOp& op = host_ops[static_cast<std::size_t>(i)];
    if (op.kind == dev::MfBankOp::kReset) {
      cpu::reset_ref_mv_bank(&want, op.mi_row, op.mi_col, op.sb_size4,
                             op.frame_is_intra != 0, op.top_row != 0,
                             cpu::RefMvBankRow{host_rows.data() + op.row_offset,
                                               op.row_mi_cols});
    } else if (op.kind == dev::MfBankOp::kCount) {
      cpu::RefMvBankBlockContext ctx;
      ctx.mi_row = op.mi_row;
      ctx.mi_col = op.mi_col;
      ctx.bw4 = op.bw4;
      ctx.bh4 = op.bh4;
      ctx.sb_size4 = op.sb_size4;
      ctx.chroma_part = op.chroma_part != 0;
      cpu::update_ref_mv_count(&want, ctx);
    } else {
      cpu::RefMvBankBlockContext ctx;
      ctx.mi_row = op.mi_row;
      ctx.mi_col = op.mi_col;
      ctx.bw4 = op.bw4;
      ctx.bh4 = op.bh4;
      ctx.sb_size4 = op.sb_size4;
      ctx.chroma_part = op.chroma_part != 0;
      const cpu::Mv mvs[2] = {op.mv[0], op.mv[1]};
      (void)cpu::update_ref_mv_bank(&want, ctx, op.refs, mvs, op.cwp_idx,
                                    op.from_within_sb != 0);
    }
  }

  // Reset device bank, then apply the same sequence.
  cpu::RefMvBank empty;
  Check(dev::upload_bank(buffers, empty) == dev::MfStatus::kOk, "upload bank");
  Check(dev::upload_bank_ops(buffers, host_ops.data(), ops, host_rows.data(),
                             rows) == dev::MfStatus::kOk,
        "upload bank ops");
  Check(dev::apply_bank_ops(buffers, ops) == dev::MfStatus::kOk,
        "apply bank ops");
  CudaOk(cudaDeviceSynchronize(), "bank ops sync");
  cpu::RefMvBank got;
  CudaOk(cudaMemcpy(&got, buffers.bank, sizeof(got), cudaMemcpyDeviceToHost),
         "download bank");
  Check(std::memcmp(&want, &got, sizeof(want)) == 0,
        "§7.12.2.21 bank ops parity");
}

// ---------------------------------------------------------------------------
// §7.12.2 merge-stack parity
// ---------------------------------------------------------------------------

void CpuSortRange(cpu::StackMv* stack, int count, int start, int end) {
  if (start < 0) start = 0;
  if (end > count) end = count;
  if (start + 1 >= end) return;
  int max_weight = stack[start].weight;
  int max_idx = start;
  for (int idx = start + 1; idx < end; ++idx) {
    if (max_weight < stack[idx].weight) {
      max_weight = stack[idx].weight;
      max_idx = idx;
    }
  }
  if (max_idx != start) {
    const cpu::StackMv tmp = stack[start];
    stack[start] = stack[max_idx];
    stack[max_idx] = tmp;
  }
}

bool StackEqual(const dev::StackMv& a, const dev::StackMv& b) {
  return a.mv[0].row == b.mv[0].row && a.mv[0].col == b.mv[0].col &&
         a.mv[1].row == b.mv[1].row && a.mv[1].col == b.mv[1].col &&
         a.weight == b.weight && a.cwp == b.cwp;
}

void ReportMergeMismatch(int seed, int mi_cols, int mi_rows, int block,
                         const dev::MfStackBlockInput& in, int scan_found,
                         int bank_found, int got_found, int got_prune,
                         int want_found, int want_prune,
                         const dev::StackMv* got, const dev::StackMv* want,
                         const cpu::RefMvBank& bank) {
  const int ref_frames[2] = {in.ref0, in.ref1};
  const int ref = dev::mf_get_rmb_list_index(ref_frames);
  const int key =
      in.is_compound ? in.ref0 + (in.ref1 + 1) * dev::t::kBankRefsPerFrame
                     : in.ref0;
  std::fprintf(
      stderr,
      "  merge mismatch: res=%dx%d MI, seed=%d, block=%d rect=(%d,%d) %dx%d, "
      "refs=(%d,%d) oh=(-? cur %d, ref %d,%d) compound=%d traj=%d "
      "temporal=%d bank=%d sort=%d nearest=%d drl=%d\n",
      mi_cols, mi_rows, seed, block, in.mi_col, in.mi_row, in.bw4, in.bh4,
      in.ref0, in.ref1, in.order_hint, in.ref0_order_hint, in.ref1_order_hint,
      in.is_compound, in.use_traj, in.use_temporal, in.use_refmvbank,
      in.reorder, in.num_nearest, in.max_drl_bits_minus_1);
  std::fprintf(stderr,
               "  pre-step: input found=%d prune=%d; after temporal scan=%d; "
               "after bank=%d; bank ref=%d key=%d size=%d start=%d "
               "remain=%d unit=%d hits=%d\n",
               in.num_found, in.prune_count, scan_found, bank_found, ref, key,
               bank.size[ref], bank.start[ref], bank.remain_hits,
               bank.unit_hits, bank.bank_hits);
  for (int i = bank.size[ref] - 1; i >= 0; --i) {
    const int idx = (bank.start[ref] + i) % dev::t::kRefMvBankSize;
    const cpu::RefMvBankEntry& e = bank.bank[ref][idx];
    if (e.p[1] != key) continue;
    std::fprintf(stderr,
                 "  bank row idx=%d key=%d cwp=%d mv0=(%d,%d) mv1=(%d,%d)\n",
                 idx, e.p[1], e.p[0], e.p[2], e.p[3], e.p[4], e.p[5]);
  }
  std::fprintf(stderr,
               "  result: gpu found=%d prune=%d; cpu found=%d prune=%d\n",
               got_found, got_prune, want_found, want_prune);
  for (int k = 0; k < dev::t::kMaxRefMvStackSize; ++k) {
    if (StackEqual(got[k], want[k])) continue;
    std::fprintf(
        stderr,
        "  first differing stack[%d]: gpu mv0=(%d,%d) mv1=(%d,%d) w=%d cwp=%d"
        " | cpu mv0=(%d,%d) mv1=(%d,%d) w=%d cwp=%d\n",
        k, got[k].mv[0].row, got[k].mv[0].col, got[k].mv[1].row,
        got[k].mv[1].col, got[k].weight, got[k].cwp, want[k].mv[0].row,
        want[k].mv[0].col, want[k].mv[1].row, want[k].mv[1].col,
        want[k].weight, want[k].cwp);
    break;
  }
  std::fprintf(stderr,
               "  note: global/warp candidates are caller-owned in this "
               "composition (pre-step stack only).\n");
}

void RunMerge(Arena& arena, const dev::MfDeviceBuffers& buffers,
              const TestConfig& cfg, Rng* rng, int pattern, int seed) {
  const int mi_cols = cfg.mi_cols;
  const int mi_rows = cfg.mi_rows;
  const int w8 = mi_cols >> 1;
  const int h8 = mi_rows >> 1;
  HostField host_field(w8, h8);
  HostTraj host_traj(w8, h8, 2);
  const std::size_t field_cells = static_cast<std::size_t>(w8) * h8;
  for (std::size_t i = 0; i < field_cells; ++i) {
    bool hole = false;
    if (pattern == 1) hole = ((static_cast<int>(i) & 1) != 0);
    else if (pattern == 2) hole = (static_cast<int>(i) % 5) == 0;
    else hole = rng->range(0, 4) == 0;
    host_field.valid[i] = hole ? 0 : 1;
    host_field.mv[i * 2 + 0] = rng->range(-500, 500);
    host_field.mv[i * 2 + 1] = rng->range(-500, 500);
    host_field.offset[i] = rng->range(-6, 6);
  }
  // Pad after filling: the trailing 4x4 row has no field cell, so this row
  // stays zero and both sides reject it.
  host_field.mv.resize(field_cells * 2 + static_cast<std::size_t>(w8) * 2, 0);
  host_field.valid.resize(field_cells + static_cast<std::size_t>(w8), 0);
  host_field.offset.resize(field_cells + static_cast<std::size_t>(w8), 0);
  for (std::size_t i = 0; i < host_traj.valid.size(); ++i) {
    host_traj.valid[i] = rng->range(0, 3) != 0 ? 1 : 0;
    host_traj.mv[i * 2 + 0] = rng->range(-2047, 2047);
    host_traj.mv[i * 2 + 1] = rng->range(-2047, 2047);
  }

  const std::uint64_t mark = dev::motion_field_mark(arena);
  std::int32_t* d_field_mv = Upload(arena, host_field.mv, "merge field mv");
  std::uint8_t* d_field_valid =
      Upload(arena, host_field.valid, "merge field valid");
  std::int32_t* d_field_offset =
      Upload(arena, host_field.offset, "merge field offset");
  std::int32_t* d_traj_mv = Upload(arena, host_traj.mv, "merge traj mv");
  std::uint8_t* d_traj_valid = Upload(arena, host_traj.valid, "merge traj valid");
  std::int16_t* d_traj_pos = Upload(arena, host_traj.pos, "merge traj pos");
  std::uint8_t* d_traj_pos_valid =
      Upload(arena, host_traj.pos_valid, "merge traj pos_valid");
  dev::MfField d_field;
  d_field.w8 = w8;
  d_field.h8 = h8;
  d_field.mv = d_field_mv;
  d_field.valid = d_field_valid;
  d_field.offset = d_field_offset;
  dev::MfTraj d_traj;
  d_traj.w8 = w8;
  d_traj.h8 = h8;
  d_traj.num_refs = 2;
  d_traj.mv = d_traj_mv;
  d_traj.valid = d_traj_valid;
  d_traj.pos = d_traj_pos;
  d_traj.pos_valid = d_traj_pos_valid;

  cpu::RefMvBank bank;
  for (int op = 0; op < 16; ++op) {
    cpu::RefMvBankBlockContext ctx;
    ctx.mi_row = rng->range(0, 31);
    ctx.mi_col = rng->range(0, 31);
    ctx.bw4 = 1 << rng->range(0, 3);
    ctx.bh4 = 1 << rng->range(0, 3);
    ctx.sb_size4 = rng->coin() ? 16 : 32;
    const int refs[2] = {rng->range(-1, 6), rng->range(-1, 6)};
    const cpu::Mv mvs[2] = {
        cpu::Mv{rng->range(-300, 300), rng->range(-300, 300)},
        cpu::Mv{rng->range(-300, 300), rng->range(-300, 300)}};
    (void)cpu::update_ref_mv_bank(&bank, ctx, refs, mvs, rng->range(0, 8),
                                  rng->coin());
  }
  Check(dev::upload_bank(buffers, bank) == dev::MfStatus::kOk,
        "merge upload bank");

  const int count = std::min(64, cfg.max_stack_blocks);
  std::vector<dev::MfStackBlockInput> host_blocks(
      static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    dev::MfStackBlockInput& in = host_blocks[static_cast<std::size_t>(i)];
    in.mi_row = rng->range(0, mi_rows - 1);
    in.mi_col = rng->range(0, mi_cols - 1);
    in.mi_cols = mi_cols;
    in.mi_rows = mi_rows;
    in.bw4 = 1 << rng->range(0, 4);
    in.bh4 = 1 << rng->range(0, 4);
    in.ref0 = rng->range(-1, 7);
    in.ref1 = rng->range(-1, 7);
    in.ref0_order_hint = rng->range(-1, 12);
    in.ref1_order_hint = rng->range(-1, 12);
    in.order_hint = rng->range(-1, 12);
    in.is_compound = rng->coin();
    in.use_traj = rng->coin();
    in.num_refs = 2;
    in.use_temporal = rng->range(0, 4) != 0;
    in.use_refmvbank = rng->coin();
    in.max_drl_bits_minus_1 = rng->range(0, 4);
    in.reorder = rng->coin();
    in.num_nearest = rng->range(0, 5);
    in.num_found = rng->range(0, 4);
    in.prune_count = rng->range(0, 18);
    for (int k = 0; k < dev::t::kMaxRefMvStackSize; ++k) {
      in.stack[k].mv[0] =
          dev::Mv{rng->range(-700, 700), rng->range(-700, 700)};
      in.stack[k].mv[1] =
          dev::Mv{rng->range(-700, 700), rng->range(-700, 700)};
      in.stack[k].weight = rng->range(1, 4);
      in.stack[k].cwp = rng->range(0, 8);
    }
    in.field = d_field;
    in.traj = d_traj;
  }
  Check(dev::upload_stack_blocks(buffers, host_blocks.data(), count) ==
            dev::MfStatus::kOk,
        "upload stack blocks");
  {
    // Echo both inputs back: a stale or truncated upload would explain a
    // device-only divergence.
    cpu::RefMvBank bank_echo{};
    CudaOk(cudaMemcpy(&bank_echo, buffers.bank, sizeof(bank_echo),
                      cudaMemcpyDeviceToHost),
           "echo bank");
    Check(std::memcmp(&bank_echo, &bank, sizeof(bank)) == 0,
          "merge bank upload parity");
    std::vector<dev::MfStackBlockInput> echo(
        static_cast<std::size_t>(count));
    CudaOk(cudaMemcpy(echo.data(), buffers.stack_blocks,
                      echo.size() * sizeof(dev::MfStackBlockInput),
                      cudaMemcpyDeviceToHost),
           "echo stack blocks");
    int echo_bad = 0;
    for (int i = 0; i < count && echo_bad < 3; ++i) {
      const dev::MfStackBlockInput& a = host_blocks[static_cast<std::size_t>(i)];
      const dev::MfStackBlockInput& b = echo[static_cast<std::size_t>(i)];
      if (a.mi_row != b.mi_row || a.mi_col != b.mi_col ||
          a.mi_cols != b.mi_cols || a.mi_rows != b.mi_rows ||
          a.bw4 != b.bw4 || a.bh4 != b.bh4 || a.ref0 != b.ref0 ||
          a.ref1 != b.ref1 || a.use_temporal != b.use_temporal ||
          a.use_refmvbank != b.use_refmvbank || a.reorder != b.reorder ||
          a.num_found != b.num_found || a.prune_count != b.prune_count ||
          a.field.mv != b.field.mv || a.field.valid != b.field.valid ||
          a.traj.mv != b.traj.mv || a.traj.valid != b.traj.valid) {
        std::fprintf(stderr,
                     "  merge upload echo diff block=%d host(rect %d,%d %dx%d "
                     "refs %d,%d found %d) dev(rect %d,%d %dx%d refs %d,%d "
                     "found %d)\n",
                     i, a.mi_col, a.mi_row, a.bw4, a.bh4, a.ref0, a.ref1,
                     a.num_found, b.mi_col, b.mi_row, b.bw4, b.bh4, b.ref0,
                     b.ref1, b.num_found);
        ++echo_bad;
      }
    }
    Check(echo_bad == 0, "merge stack-block upload parity");
  }
  Check(dev::merge_stacks(buffers, count) == dev::MfStatus::kOk,
        "merge stacks launch");
  CudaOk(cudaDeviceSynchronize(), "merge sync");

  std::vector<dev::StackMv> got(static_cast<std::size_t>(count) *
                                dev::t::kMaxRefMvStackSize);
  std::vector<int> got_found(static_cast<std::size_t>(count));
  std::vector<int> got_prune(static_cast<std::size_t>(count));
  CudaOk(cudaMemcpy(got.data(), buffers.stack_out,
                    got.size() * sizeof(dev::StackMv), cudaMemcpyDeviceToHost),
         "download stack");
  CudaOk(cudaMemcpy(got_found.data(), buffers.stack_num_found,
                    got_found.size() * sizeof(int), cudaMemcpyDeviceToHost),
         "download found");
  CudaOk(cudaMemcpy(got_prune.data(), buffers.stack_prune_count,
                    got_prune.size() * sizeof(int), cudaMemcpyDeviceToHost),
         "download prune");

  cpu::MotionField cf = host_field.Cpu();
  cpu::TrajectoryField ct;
  ct.w8 = w8;
  ct.h8 = h8;
  ct.num_refs = 2;
  ct.mv = host_traj.mv.data();
  ct.valid = host_traj.valid.data();
  ct.pos = host_traj.pos.data();
  ct.pos_valid = host_traj.pos_valid.data();

  for (int i = 0; i < count; ++i) {
    const dev::MfStackBlockInput& in = host_blocks[static_cast<std::size_t>(i)];
    cpu::StackMv want[cpu::kMaxRefMvStackSize]{};
    int want_found = in.num_found;
    int want_prune = in.prune_count;
    for (int k = 0; k < cpu::kMaxRefMvStackSize; ++k) {
      want[k].mv[0] = in.stack[k].mv[0];
      want[k].mv[1] = in.stack[k].mv[1];
      want[k].weight = in.stack[k].weight;
      want[k].cwp = in.stack[k].cwp;
    }
    int scan_found = want_found;
    if (in.use_temporal) {
      cpu::TemporalScanInput ci;
      ci.mi_row = in.mi_row;
      ci.mi_col = in.mi_col;
      ci.mi_cols = in.mi_cols;
      ci.mi_rows = in.mi_rows;
      ci.tile_mi_col_start = in.tile_mi_col_start;
      ci.tile_mi_col_end = in.tile_mi_col_end;
      ci.tile_mi_row_start = in.tile_mi_row_start;
      ci.tile_mi_row_end = in.tile_mi_row_end;
      ci.bw4 = in.bw4;
      ci.bh4 = in.bh4;
      ci.ref0 = in.ref0;
      ci.ref1 = in.ref1;
      ci.ref0_order_hint = in.ref0_order_hint;
      ci.ref1_order_hint = in.ref1_order_hint;
      ci.order_hint = in.order_hint;
      ci.is_compound = in.is_compound;
      ci.use_traj = in.use_traj;
      ci.num_refs = in.num_refs;
      ci.field = &cf;
      ci.traj = in.use_traj ? &ct : nullptr;
      cpu::temporal_scan(ci, want, &want_found, &want_prune);
      scan_found = want_found;
    }
    int bank_found = want_found;
    if (in.use_refmvbank) {
      const int refs[2] = {in.ref0, in.ref1};
      cpu::fill_mvp_from_ref_mv_bank(bank, refs, in.is_compound, in.use_intrabc,
                                     in.max_drl_bits_minus_1, in.mi_row,
                                     in.mi_col, in.mi_cols, in.mi_rows,
                                     in.bw4 * 4, in.bh4 * 4,
                                     cpu::kMaxRefMvStackSize, want, &want_found,
                                     &want_prune);
      bank_found = want_found;
    }
    if (in.reorder) {
      CpuSortRange(want, want_found, 0, in.num_nearest);
    }
    const std::size_t base =
        static_cast<std::size_t>(i) * dev::t::kMaxRefMvStackSize;
    bool block_bad =
        got_found[static_cast<std::size_t>(i)] != want_found ||
        got_prune[static_cast<std::size_t>(i)] != want_prune;
    for (int k = 0; k < dev::t::kMaxRefMvStackSize; ++k) {
      if (!StackEqual(got[base + static_cast<std::size_t>(k)], want[k])) {
        block_bad = true;
      }
    }
    if (block_bad && g_merge_reported == 0) {
      ++g_merge_reported;
      ReportMergeMismatch(seed, mi_cols, mi_rows, i, in, scan_found,
                          bank_found, got_found[static_cast<std::size_t>(i)],
                          got_prune[static_cast<std::size_t>(i)], want_found,
                          want_prune, &got[base], want, bank);
    }
    Check(got_found[static_cast<std::size_t>(i)] == want_found,
          "merge num_found parity");
    Check(got_prune[static_cast<std::size_t>(i)] == want_prune,
          "merge prune parity");
    for (int k = 0; k < dev::t::kMaxRefMvStackSize; ++k) {
      const dev::StackMv& g = got[base + static_cast<std::size_t>(k)];
      Check(g.mv[0].row == want[k].mv[0].row &&
                g.mv[0].col == want[k].mv[0].col,
            "merge mv0 parity");
      Check(g.mv[1].row == want[k].mv[1].row &&
                g.mv[1].col == want[k].mv[1].col,
            "merge mv1 parity");
      Check(g.weight == want[k].weight, "merge weight parity");
      Check(g.cwp == want[k].cwp, "merge cwp parity");
    }
  }

  Check(dev::motion_field_rewind(arena, mark), "merge scratch rewind");
}

// ---------------------------------------------------------------------------
// Arena contract
// ---------------------------------------------------------------------------

void TestOneByteShort() {
  dev::MfConfig config;
  config.mi_cols = 64;
  config.mi_rows = 64;
  config.num_refs = 2;
  config.max_blocks = 16;
  config.max_stack_blocks = 4;
  config.max_projection_jobs = 3;
  config.max_bank_ops = 4;
  config.max_bank_rows = 8;
  const std::size_t need = dev::required_device_bytes(config);
  Check(need > 0, "required bytes positive");

  av2cuda::Arena small;
  std::string error;
  Check(av2cuda::is_ok(small.init(need - 1, av2cuda::Arena::kAlignment, &error)),
        error.c_str());
  dev::MfDeviceBuffers buffers;
  std::size_t reported = 0;
  const dev::MfStatus status = dev::allocate_checked(
      small, config, &buffers, &reported, &error);
  Check(status == dev::MfStatus::kArenaTooSmall, "one-byte-short status");
  Check(reported == need, "one-byte-short reports exact need");
  Check(small.used() == 0, "one-byte-short leaves bump pointer untouched");
  Check(dev::motion_field_mark(small) == 0, "mark after failure is zero");
  small.release();
}

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

void RunCase(Arena& arena, const TestConfig& cfg, int seed, bool use_traj,
             int proj_step, bool sb64, int pattern, int max_dist) {
  const std::size_t need =
      dev::required_device_bytes(ToMfConfig(cfg));
  Check(need > 0, "case required bytes");
  Check(need <= arena.remaining(), "arena capacity covers case");

  const std::uint64_t mark = dev::motion_field_mark(arena);
  dev::MfDeviceBuffers buffers;
  std::size_t reported = 0;
  std::string error;
  const dev::MfStatus status = dev::allocate_checked(
      arena, ToMfConfig(cfg), &buffers, &reported, &error);
  Check(status == dev::MfStatus::kOk, error.c_str());
  Check(reported == need, "reported size matches required_device_bytes");
  if (status != dev::MfStatus::kOk) return;

  Rng rng(static_cast<std::uint64_t>(seed) * 6364136223846793005ull + 17);
  Check(dev::zero_field(buffers) == dev::MfStatus::kOk, "initial zero");
  CudaOk(cudaDeviceSynchronize(), "initial zero sync");

  // Fixed-buffer watermark before any test scratch.
  const std::uint64_t fixed_used = arena.used();

  RunStorage(arena, buffers, cfg, &rng);
  RunProjection(arena, buffers, cfg, &rng, pattern, max_dist, use_traj,
                proj_step, sb64);
  RunBankOps(arena, buffers, &rng);
  RunMerge(arena, buffers, cfg, &rng, pattern, seed);

  Check(arena.used() == fixed_used, "case ends at the fixed watermark");
  Check(dev::motion_field_rewind(arena, mark), "case rewind");
  (void)need;
}

}  // namespace

int main(int argc, char** argv) {
  bool quick = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--quick") == 0) quick = true;
  }

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    std::printf("SKIP: no CUDA device\n");
    return 2;
  }
  CudaOk(cudaSetDevice(0), "cudaSetDevice");
  cudaDeviceProp prop{};
  CudaOk(cudaGetDeviceProperties(&prop, 0), "device properties");
  std::printf("device: %s (sm_%d%d), %d MiB\n", prop.name, prop.major,
              prop.minor, static_cast<int>(prop.totalGlobalMem >> 20));

  TestOneByteShort();

  struct Resolution {
    int mi_cols;
    int mi_rows;
  };
  std::vector<Resolution> resolutions = {
      {2, 2}, {8, 6}, {16, 16}, {80, 45}, {160, 90}};
  if (!quick) {
    resolutions.push_back({480, 270});  // 1920x1080
  }

  // Size the single persistent arena for the largest case.  Every allocation
  // still goes through allocate_checked()/mark()/rewind(); the extra headroom
  // is for the test's own upload scratch, not for the module.
  std::size_t max_need = 0;
  for (const Resolution& r : resolutions) {
    TestConfig cfg;
    cfg.mi_cols = r.mi_cols;
    cfg.mi_rows = r.mi_rows;
    cfg.max_blocks = r.mi_cols * r.mi_rows + 8;
    max_need = std::max(max_need, dev::required_device_bytes(ToMfConfig(cfg)));
  }
  const std::size_t arena_bytes =
      max_need + 64ull * 1024ull * 1024ull;  // test upload scratch
  av2cuda::Arena arena;
  std::string error;
  Check(av2cuda::is_ok(
            arena.init(arena_bytes, av2cuda::Arena::kAlignment, &error)),
        error.c_str());
  std::printf("required_device_bytes(max case) = %zu (arena %zu)\n", max_need,
              arena_bytes);

  int case_index = 0;
  for (const Resolution& r : resolutions) {
    for (int seed = 1; seed <= (quick ? 1 : 3); ++seed) {
      TestConfig cfg;
      cfg.mi_cols = r.mi_cols;
      cfg.mi_rows = r.mi_rows;
      cfg.max_blocks = r.mi_cols * r.mi_rows + 8;
      const bool use_traj = (seed % 2) == 0;
      const int proj_step = (seed % 3) == 0 ? 2 : 1;
      const bool sb64 = (seed % 3) != 0;
      const int pattern = (seed + case_index) % 4;
      const int max_dist = 1 << ((seed - 1) % 5);
      cfg.proj_step = proj_step;
      cfg.sb_64x64 = sb64 ? 1 : 0;
      std::printf("case %d: %dx%d MI seed=%d traj=%d step=%d sb64=%d "
                  "pattern=%d dist=%d\n",
                  case_index, r.mi_cols, r.mi_rows, seed, use_traj ? 1 : 0,
                  proj_step, sb64 ? 1 : 0, pattern, max_dist);
      RunCase(arena, cfg, seed * 7919 + case_index, use_traj, proj_step, sb64,
              pattern, max_dist);
      ++case_index;
    }
  }

  Check(arena.used() == 0, "arena returns to zero after every case");
  arena.release();

  if (g_failures == 0) {
    std::printf("PASS: %d checks, 0 failures\n", g_checks);
    return 0;
  }
  std::printf("FAIL: %d checks, %d failures\n", g_checks, g_failures);
  return 1;
}

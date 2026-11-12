// CPU-vs-GPU bit-exactness test for the AV2 §7.20.4 PC-Wiener filter.
//
// Every case builds random luma planes, runs the CPU reference
// (src/encoder/filters/spec_tables/pc_wiener.cpp) and the CUDA launchers on the
// same inputs, and requires byte-for-byte equality.
//
// Coverage:
//   * 8/10/12-bit luma, several base_q_idx values across all four filter sets,
//   * 64-line stripe sample sourcing and LrTxSkip tile clamping,
//   * skipFilter = 1 class maps (and proof that LrFrame is untouched),
//   * skipFilter = 0 full-frame filtered planes,
//   * every class 0..63 through the class-grid filter path (each filter set),
//   * Pc_Wiener_Sub_Classify and Pc_Wiener_Sub_Classify2 devices LUTs,
//   * every LR unit size 64/128/256/512 with random per-unit types,
//   * table uploads are byte-identical to the CPU normative tables,
//   * Arena sizing from required_device_bytes(), a one-byte-short negative
//     test, and two launches with mark()/rewind() and no net growth.
//
// All device memory comes from the persistent av2enc::cuda::Arena; this file
// contains no cudaMalloc / cudaFree / cudaMallocHost.
//
// Build/run: ./build_gpu_test.sh (needs nvcc and a CUDA device).

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "cuda/arena.h"
#include "cuda/pc_wiener/pc_wiener.h"
#include "encoder/filters/spec_tables/pc_wiener.h"
#include "encoder/filters/spec_tables/pc_wiener_tables.h"
#include "encoder/filters/spec_tables/spec_test_common.h"

namespace {

namespace gp = av2enc::cuda::pc_wiener;
using av2enc::cuda::Arena;
using av2enc::cuda::Status;
using av2enc::cuda::is_ok;
using av2enc::cuda::status_name;
using av2enc::filters::spec::kPcWienerConfig;
using av2enc::filters::spec::kPcWienerModeOffsets;
using av2enc::filters::spec::kPcWienerModeWeights;
using av2enc::filters::spec::kPcWienerNormalizer;
using av2enc::filters::spec::kPcWienerTaps;
using av2enc::filters::spec::PcWienerClass;
using gp::kPrecBits;
using av2enc::filters::spec::PcWienerConfig;
using av2enc::filters::spec::PcWienerFilterBlock;
using av2enc::filters::spec::PcWienerFilterSetIndex;
using av2enc::filters::spec::PcWienerFrame;
using av2enc::filters::spec_tables::kPcWienerFilters;
using av2enc::filters::spec_tables::kPcWienerLutToClass;
using av2enc::filters::spec_tables::kPcWienerSubClassify;
using av2enc::filters::spec_tables::kPcWienerSubClassify2;
using av2enc::filters::spec_tables::kQlookup;
using av2enc::filters::spec_tables::kAcQlookup;
using av2enc::filters::spec::test::Image;

int g_checks = 0;
int g_failures = 0;
long g_cases = 0;

#undef CHECK
#define CHECK(cond, msg)                                                     \
  do {                                                                       \
    ++g_checks;                                                              \
    if (!(cond)) {                                                           \
      ++g_failures;                                                          \
      if (g_failures <= 25) {                                                \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, (msg)); \
      }                                                                      \
    }                                                                        \
  } while (0)

#define CUDA_CHECK(expr, msg)                                                \
  do {                                                                       \
    const cudaError_t err__ = (expr);                                        \
    ++g_checks;                                                              \
    if (err__ != cudaSuccess) {                                              \
      ++g_failures;                                                          \
      if (g_failures <= 25) {                                                \
        std::fprintf(stderr, "FAIL %s:%d: %s (%s)\n", __FILE__, __LINE__,    \
                     (msg), cudaGetErrorString(err__));                      \
      }                                                                      \
    }                                                                        \
  } while (0)

std::uint32_t g_rng = 0x9e3779b9u;
std::uint32_t next_random() {
  g_rng ^= g_rng << 13;
  g_rng ^= g_rng >> 17;
  g_rng ^= g_rng << 5;
  return g_rng;
}
int random_range(int low, int high) {
  return low + static_cast<int>(next_random() %
                                static_cast<std::uint32_t>(high - low + 1));
}

int max_q(int bd) { return bd == 8 ? 255 : (bd == 10 ? 303 : 351); }

struct Extents {
  int start_x;
  int end_x;
  int start_y;
  int end_y;
  int stripe_start_y;
  int stripe_end_y;
};

void stripe_bounds(int y, int height, int* start, int* end) {
  const int stripe_num = (y + 8) / 64;
  const int unclipped_start = stripe_num * 64 - 8;
  const int unclipped_end = unclipped_start + 64;
  *start = std::max(0, unclipped_start);
  *end = std::min(height - 1, unclipped_end - 1);
}

Extents extents_for(int width, int height, int y) {
  Extents e{0, width - 1, 0, height - 1, 0, height - 1};
  stripe_bounds(y, height, &e.stripe_start_y, &e.stripe_end_y);
  return e;
}

std::int64_t oracle_round2(std::int64_t x, int n) {
  return n == 0 ? x : (x + (std::int64_t{1} << (n - 1))) >> n;
}

void oracle_filter_class(const Image& curr, const Image& cdef, const Extents& e,
                         int base_q_idx, int cls, int bit_depth, int x, int y,
                         int w, int h, Image* lr) {
  const int index = PcWienerFilterSetIndex(base_q_idx);
  const int filt = kPcWienerSubClassify[index][cls];
  const int maxv = (1 << bit_depth) - 1;
  for (int r2 = 0; r2 < h; ++r2) {
    for (int c2 = 0; c2 < w; ++c2) {
      const int m = av2enc::filters::spec::test::OracleSample(
          curr, cdef, e.start_x, e.end_x, e.start_y, e.end_y, e.stripe_start_y,
          e.stripe_end_y, x + c2, y + r2);
      std::int64_t s = static_cast<std::int64_t>(m) << kPrecBits;
      for (int i = 0; i < kPcWienerTaps; ++i) {
        const std::int64_t coeff = kPcWienerFilters[index][filt][i >> 1];
        const int sample = av2enc::filters::spec::test::OracleSample(
            curr, cdef, e.start_x, e.end_x, e.start_y, e.end_y,
            e.stripe_start_y, e.stripe_end_y, x + c2 + kPcWienerConfig[i].dx,
            y + r2 + kPcWienerConfig[i].dy);
        s += static_cast<std::int64_t>(sample) * coeff;
      }
      const int v = static_cast<int>(oracle_round2(s, kPrecBits));
      lr->At(x + c2, y + r2) =
          static_cast<std::uint16_t>(std::max(0, std::min(maxv, v)));
    }
  }
}

void copy_cdef(const Image& cdef, Image* lr) {
  for (int y = 0; y < cdef.height; ++y) {
    for (int x = 0; x < cdef.width; ++x) lr->At(x, y) = cdef.At(x, y);
  }
}

PcWienerFrame make_cpu_frame(const Image& curr, const Image& cdef, int y,
                             const std::vector<std::uint8_t>& tx_skip, int bd,
                             int q, int mi_cols, int mi_rows, Image* lr) {
  const Extents e = extents_for(curr.width, curr.height, y);
  PcWienerFrame frame{};
  frame.src.curr = curr.px.data();
  frame.src.cdef = cdef.px.data();
  frame.src.curr_stride = curr.stride;
  frame.src.cdef_stride = cdef.stride;
  frame.src.luma_start_x = e.start_x;
  frame.src.luma_end_x = e.end_x;
  frame.src.luma_start_y = e.start_y;
  frame.src.luma_end_y = e.end_y;
  frame.src.luma_stripe_start_y = e.stripe_start_y;
  frame.src.luma_stripe_end_y = e.stripe_end_y;
  frame.lr = lr != nullptr ? lr->px.data() : nullptr;
  frame.lr_stride = lr != nullptr ? lr->stride : 0;
  frame.filter_class = nullptr;
  frame.filter_class_stride = 0;
  (void)tx_skip;
  (void)bd;
  (void)q;
  (void)mi_cols;
  (void)mi_rows;
  return frame;
}

PcWienerConfig make_cpu_config(int q, int bd, int mi_cols, int mi_rows,
                               const std::vector<std::uint8_t>& tx_skip) {
  PcWienerConfig cfg{};
  cfg.base_q_idx = q;
  cfg.bit_depth = bd;
  cfg.mi_col_end = mi_cols;
  cfg.mi_row_start = 0;
  cfg.mi_row_end = mi_rows;
  cfg.lr_tx_skip = tx_skip.data();
  cfg.lr_tx_skip_stride = mi_cols;
  return cfg;
}

void cpu_classify_grid(const Image& curr, const Image& cdef,
                       const std::vector<std::uint8_t>& tx_skip, int bd, int q,
                       int mi_cols, int mi_rows,
                       std::vector<std::uint8_t>* grid) {
  const PcWienerConfig cfg = make_cpu_config(q, bd, mi_cols, mi_rows, tx_skip);
  grid->assign(static_cast<std::size_t>(mi_cols) * mi_rows, 0);
  Image dummy(1, 1);
  for (int row = 0; row < mi_rows; ++row) {
    PcWienerFrame frame = make_cpu_frame(curr, cdef, row * 4, tx_skip, bd, q,
                                         mi_cols, mi_rows, &dummy);
    for (int col = 0; col < mi_cols; ++col) {
      (*grid)[static_cast<std::size_t>(row) * mi_cols + col] =
          static_cast<std::uint8_t>(PcWienerClass(frame, cfg, col * 4, row * 4));
    }
  }
}

void cpu_filter_grid(const Image& curr, const Image& cdef,
                     const std::vector<std::uint8_t>& tx_skip, int bd, int q,
                     int mi_cols, int mi_rows, Image* lr) {
  const PcWienerConfig cfg = make_cpu_config(q, bd, mi_cols, mi_rows, tx_skip);
  copy_cdef(cdef, lr);
  for (int row = 0; row < mi_rows; ++row) {
    PcWienerFrame frame = make_cpu_frame(curr, cdef, row * 4, tx_skip, bd, q,
                                         mi_cols, mi_rows, lr);
    for (int col = 0; col < mi_cols; ++col) {
      PcWienerFilterBlock(frame, cfg, col * 4, row * 4, 4, 4);
    }
  }
}

void cpu_filter_grid_classes(const Image& curr, const Image& cdef,
                             const std::vector<std::uint8_t>& tx_skip, int bd,
                             int q, int mi_cols, int mi_rows,
                             const std::vector<std::uint8_t>& grid,
                             Image* lr) {
  (void)tx_skip;
  copy_cdef(cdef, lr);
  for (int row = 0; row < mi_rows; ++row) {
    const Extents e = extents_for(curr.width, curr.height, row * 4);
    for (int col = 0; col < mi_cols; ++col) {
      const int cls = grid[static_cast<std::size_t>(row) * mi_cols + col];
      oracle_filter_class(curr, cdef, e, q, cls, bd, col * 4, row * 4, 4, 4,
                          lr);
    }
  }
}

int count_units(int unit_size, int frame_size) {
  const int n = (frame_size + (unit_size >> 1)) / unit_size;
  return n > 1 ? n : 1;
}

int unit_index_for_block(int row, int col, int unit_size, int unit_rows,
                         int unit_cols) {
  const int unit_row = std::min(unit_rows - 1, (row * 4 + 8) / unit_size);
  const int unit_col = std::min(unit_cols - 1, (col * 4) / unit_size);
  return unit_row * unit_cols + unit_col;
}

void cpu_filter_units(const Image& curr, const Image& cdef,
                      const std::vector<std::uint8_t>& tx_skip, int bd, int q,
                      int mi_cols, int mi_rows,
                      const std::vector<std::uint8_t>& grid,
                      const std::vector<std::uint8_t>& unit_type, int unit_size,
                      Image* lr) {
  const PcWienerConfig cfg = make_cpu_config(q, bd, mi_cols, mi_rows, tx_skip);
  copy_cdef(cdef, lr);
  const int unit_rows = count_units(unit_size, curr.height);
  const int unit_cols = count_units(unit_size, curr.width);
  for (int row = 0; row < mi_rows; ++row) {
    PcWienerFrame frame = make_cpu_frame(curr, cdef, row * 4, tx_skip, bd, q,
                                         mi_cols, mi_rows, lr);
    for (int col = 0; col < mi_cols; ++col) {
      const int unit =
          unit_index_for_block(row, col, unit_size, unit_rows, unit_cols);
      if (unit_type[static_cast<std::size_t>(unit)] != 1) continue;
      const int cls = grid[static_cast<std::size_t>(row) * mi_cols + col];
      (void)cls;
      PcWienerFilterBlock(frame, cfg, col * 4, row * 4, 4, 4);
    }
  }
}

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

struct DevPlane {
  std::uint16_t* ptr = nullptr;
  int stride = 0;
  int height = 0;
};

bool upload_plane(const Image& src, const DevPlane& dst) {
  const cudaError_t err = cudaMemcpy2D(
      dst.ptr, static_cast<std::size_t>(dst.stride) * sizeof(std::uint16_t),
      src.px.data(), static_cast<std::size_t>(src.stride) * sizeof(std::uint16_t),
      static_cast<std::size_t>(src.width) * sizeof(std::uint16_t),
      static_cast<std::size_t>(src.height), cudaMemcpyHostToDevice);
  return err == cudaSuccess;
}

std::vector<std::uint16_t> download_plane(const DevPlane& src) {
  std::vector<std::uint16_t> out(
      static_cast<std::size_t>(src.stride) * src.height);
  const cudaError_t err = cudaMemcpy2D(
      out.data(), static_cast<std::size_t>(src.stride) * sizeof(std::uint16_t),
      src.ptr, static_cast<std::size_t>(src.stride) * sizeof(std::uint16_t),
      static_cast<std::size_t>(src.stride) * sizeof(std::uint16_t),
      static_cast<std::size_t>(src.height), cudaMemcpyDeviceToHost);
  (void)err;
  return out;
}

bool plane_matches(const Image& want, const DevPlane& got) {
  const std::vector<std::uint16_t> host = download_plane(got);
  for (int y = 0; y < want.height; ++y) {
    for (int x = 0; x < want.width; ++x) {
      if (host[static_cast<std::size_t>(y) * want.stride + x] != want.At(x, y)) {
        if (g_failures < 25) {
          std::fprintf(stderr, "plane mismatch at (%d,%d): got %u want %u\n", x,
                       y, static_cast<unsigned>(
                              host[static_cast<std::size_t>(y) * want.stride + x]),
                       static_cast<unsigned>(want.At(x, y)));
        }
        return false;
      }
    }
  }
  return true;
}

gp::PcWienerSource make_gpu_source(const DevPlane& curr, const DevPlane& cdef,
                                   const Extents& e) {
  gp::PcWienerSource src{};
  src.curr = curr.ptr;
  src.cdef = cdef.ptr;
  src.curr_stride = curr.stride;
  src.cdef_stride = cdef.stride;
  src.luma_start_x = e.start_x;
  src.luma_end_x = e.end_x;
  src.luma_start_y = e.start_y;
  src.luma_end_y = e.end_y;
  src.stripe_start_y = e.stripe_start_y;
  src.stripe_end_y = e.stripe_end_y;
  return src;
}

gp::PcWienerParams make_gpu_params(int q, int bd, int mi_cols, int mi_rows,
                                   const std::uint8_t* tx_skip, int stride) {
  gp::PcWienerParams params{};
  params.base_q_idx = q;
  params.bit_depth = bd;
  params.mi_col_end = mi_cols;
  params.mi_row_start = 0;
  params.mi_row_end = mi_rows;
  params.lr_tx_skip = tx_skip;
  params.lr_tx_skip_stride = stride;
  return params;
}

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

void test_table_upload(gp::PcWienerWorkspace& ws, std::uint8_t* d_lut) {
  const gp::PcWienerTables& tables = ws.tables();

  std::vector<std::int16_t> filters(
      4u * 64u * 13u);
  CUDA_CHECK(cudaMemcpy(filters.data(), tables.filters,
                        filters.size() * sizeof(std::int16_t),
                        cudaMemcpyDeviceToHost),
             "download filters");
  CHECK(std::memcmp(filters.data(), &kPcWienerFilters[0][0][0],
                    filters.size() * sizeof(std::int16_t)) == 0,
        "uploaded Pc_Wiener_Filters");

  std::vector<std::uint8_t> lut(4096);
  CUDA_CHECK(cudaMemcpy(lut.data(), tables.lut_to_class, lut.size(),
                        cudaMemcpyDeviceToHost),
             "download LUT");
  CHECK(std::memcmp(lut.data(), kPcWienerLutToClass, lut.size()) == 0,
        "uploaded Pc_Wiener_LutToClass");

  std::vector<std::uint8_t> sub1(4u * 256u);
  CUDA_CHECK(cudaMemcpy(sub1.data(), tables.sub_classify, sub1.size(),
                        cudaMemcpyDeviceToHost),
             "download Sub_Classify");
  CHECK(std::memcmp(sub1.data(), &kPcWienerSubClassify[0][0], sub1.size()) == 0,
        "uploaded Pc_Wiener_Sub_Classify");

  std::vector<std::uint8_t> sub2(4u * 8u * 256u);
  CUDA_CHECK(cudaMemcpy(sub2.data(), tables.sub_classify2, sub2.size(),
                        cudaMemcpyDeviceToHost),
             "download Sub_Classify2");
  CHECK(std::memcmp(sub2.data(), &kPcWienerSubClassify2[0][0][0],
                    sub2.size()) == 0,
        "uploaded Pc_Wiener_Sub_Classify2");

  std::vector<std::int32_t> qlookup(352);
  CUDA_CHECK(cudaMemcpy(qlookup.data(), tables.qlookup,
                        qlookup.size() * sizeof(std::int32_t),
                        cudaMemcpyDeviceToHost),
             "download qlookup");
  CHECK(std::memcmp(qlookup.data(), kQlookup,
                    qlookup.size() * sizeof(std::int32_t)) == 0,
        "uploaded qlookup");

  std::vector<std::int32_t> acq(25);
  CUDA_CHECK(cudaMemcpy(acq.data(), tables.ac_qlookup,
                        acq.size() * sizeof(std::int32_t),
                        cudaMemcpyDeviceToHost),
             "download Ac_Qlookup");
  CHECK(std::memcmp(acq.data(), kAcQlookup,
                    acq.size() * sizeof(std::int32_t)) == 0,
        "uploaded Ac_Qlookup");

  std::vector<std::int8_t> taps(50);
  CUDA_CHECK(cudaMemcpy(taps.data(), tables.taps, taps.size(),
                        cudaMemcpyDeviceToHost),
             "download taps");
  for (int i = 0; i < kPcWienerTaps; ++i) {
    CHECK(taps[static_cast<std::size_t>(i) * 2 + 0] ==
              kPcWienerConfig[i].dy &&
              taps[static_cast<std::size_t>(i) * 2 + 1] ==
                  kPcWienerConfig[i].dx,
          "uploaded tap config");
  }

  std::vector<std::int32_t> normalizer(5);
  CUDA_CHECK(cudaMemcpy(normalizer.data(), tables.normalizer,
                        normalizer.size() * sizeof(std::int32_t),
                        cudaMemcpyDeviceToHost),
             "download normalizer");
  CHECK(std::memcmp(normalizer.data(), kPcWienerNormalizer,
                    normalizer.size() * sizeof(std::int32_t)) == 0,
        "uploaded Pc_Wiener_Normalizer");

  std::vector<std::int32_t> weights(12);
  CUDA_CHECK(cudaMemcpy(weights.data(), tables.mode_weights,
                        weights.size() * sizeof(std::int32_t),
                        cudaMemcpyDeviceToHost),
             "download mode weights");
  CHECK(std::memcmp(weights.data(), &kPcWienerModeWeights[0][0],
                    weights.size() * sizeof(std::int32_t)) == 0,
        "uploaded Pc_Wiener_Mode_Weights");

  std::vector<std::int32_t> offsets(4);
  CUDA_CHECK(cudaMemcpy(offsets.data(), tables.mode_offsets,
                        offsets.size() * sizeof(std::int32_t),
                        cudaMemcpyDeviceToHost),
             "download mode offsets");
  CHECK(std::memcmp(offsets.data(), kPcWienerModeOffsets,
                    offsets.size() * sizeof(std::int32_t)) == 0,
        "uploaded Pc_Wiener_Mode_Offsets");

  std::vector<std::uint8_t> got_lut(256);
  for (int set = 0; set < 4; ++set) {
    CHECK(is_ok(gp::pc_wiener_class_filter_lut(ws, set, d_lut)),
          "class_filter_lut status");
    CUDA_CHECK(cudaMemcpy(got_lut.data(), d_lut, got_lut.size(),
                          cudaMemcpyDeviceToHost),
               "download class_filter_lut");
    CHECK(std::memcmp(got_lut.data(), kPcWienerSubClassify[set],
                      got_lut.size()) == 0,
          "Pc_Wiener_Sub_Classify kernel");
    for (int idx = 0; idx < 8; ++idx) {
      CHECK(is_ok(gp::pc_wiener_subclass_lookup(ws, set, idx, d_lut)),
            "subclass_lookup status");
      CUDA_CHECK(cudaMemcpy(got_lut.data(), d_lut, got_lut.size(),
                            cudaMemcpyDeviceToHost),
                 "download subclass_lookup");
      CHECK(std::memcmp(got_lut.data(), kPcWienerSubClassify2[set][idx],
                        got_lut.size()) == 0,
            "Pc_Wiener_Sub_Classify2 kernel");
      ++g_cases;
    }
  }
}

struct DeviceSet {
  DevPlane curr;
  DevPlane cdef;
  DevPlane lr;
  std::uint8_t* tx_skip = nullptr;
  std::uint8_t* grid = nullptr;
  std::uint8_t* grid_block = nullptr;
  std::uint8_t* unit_type = nullptr;
  std::uint8_t* lut = nullptr;
  int width = 0;
  int height = 0;
  int mi_cols = 0;
  int mi_rows = 0;
};

void test_skip_and_full_frame(const DeviceSet& d, gp::PcWienerWorkspace& ws,
                              const Image& curr, const Image& cdef,
                              const std::vector<std::uint8_t>& tx_skip,
                              int bd, int q) {
  const int mi_cols = d.mi_cols;
  const int mi_rows = d.mi_rows;

  // skipFilter = 1 must leave LrFrame untouched.
  CUDA_CHECK(cudaMemset(d.lr.ptr, 0x5a, d.lr.stride * d.lr.height *
                                             sizeof(std::uint16_t)),
             "sentinel LrFrame");
  const Extents e0 = extents_for(d.width, d.height, 0);
  const gp::PcWienerSource src0 = make_gpu_source(d.curr, d.cdef, e0);
  gp::PcWienerParams params =
      make_gpu_params(q, bd, mi_cols, mi_rows, d.tx_skip, mi_cols);
  CHECK(is_ok(gp::pc_wiener_classify(ws, src0, params)), "classify status");
  const std::vector<std::uint16_t> sentinel = download_plane(d.lr);
  bool untouched = true;
  for (std::uint16_t v : sentinel) {
    if (v != 0x5a5a) {
      untouched = false;
      break;
    }
  }
  CHECK(untouched, "skipFilter=1 leaves LrFrame untouched");

  std::vector<std::uint8_t> got_grid(
      static_cast<std::size_t>(mi_cols) * mi_rows);
  CUDA_CHECK(cudaMemcpy(got_grid.data(), ws.class_grid(), got_grid.size(),
                        cudaMemcpyDeviceToHost),
             "download class grid");
  std::vector<std::uint8_t> want_grid;
  cpu_classify_grid(curr, cdef, tx_skip, bd, q, mi_cols, mi_rows, &want_grid);
  CHECK(got_grid == want_grid, "skipFilter=1 class map parity");
  ++g_cases;

  // Full-frame skipFilter = 0.
  Image want_lr(d.width, d.height, curr.stride);
  cpu_filter_grid(curr, cdef, tx_skip, bd, q, mi_cols, mi_rows, &want_lr);
  CHECK(is_ok(gp::pc_wiener_filter_grid(ws, src0, params, ws.class_grid(),
                                        d.lr.ptr, d.lr.stride)),
        "filter_grid status");
  CHECK(plane_matches(want_lr, d.lr), "full-frame filtered plane parity");
  ++g_cases;

  // Single-block skipFilter = 1 with a sentinel grid.
  const int bx = 2 * 4;
  const int by = 3 * 4;
  std::vector<std::uint8_t> block_grid(
      static_cast<std::size_t>(mi_cols) * mi_rows, 0xaa);
  CUDA_CHECK(cudaMemcpy(d.grid_block, block_grid.data(), block_grid.size(),
                        cudaMemcpyHostToDevice),
             "upload block grid");
  const Extents eb = extents_for(d.width, d.height, by);
  const gp::PcWienerSource srcb = make_gpu_source(d.curr, d.cdef, eb);
  CHECK(is_ok(gp::pc_wiener_classify_block(ws, srcb, params, bx, by,
                                           d.grid_block, mi_cols)),
        "classify_block status");
  CUDA_CHECK(cudaMemcpy(block_grid.data(), d.grid_block, block_grid.size(),
                        cudaMemcpyDeviceToHost),
             "download block grid");
  const std::size_t bi = static_cast<std::size_t>(by / 4) * mi_cols + bx / 4;
  CHECK(block_grid[bi] == want_grid[bi], "classify_block parity");
  bool block_only = true;
  for (std::size_t i = 0; i < block_grid.size(); ++i) {
    if (i != bi && block_grid[i] != 0xaa) block_only = false;
  }
  CHECK(block_only, "classify_block writes one entry");
  ++g_cases;
}

void test_every_class(const DeviceSet& d, gp::PcWienerWorkspace& ws,
                      const Image& curr, const Image& cdef,
                      const std::vector<std::uint8_t>& tx_skip, int bd) {
  const int mi_cols = d.mi_cols;
  const int mi_rows = d.mi_rows;
  const int q_for_set[4] = {0, 130, 190, 220};
  std::vector<std::uint8_t> grid(static_cast<std::size_t>(mi_cols) * mi_rows);
  for (std::size_t i = 0; i < grid.size(); ++i) {
    grid[i] = static_cast<std::uint8_t>(i % 64);
  }
  bool all_classes = true;
  for (int c = 0; c < 64; ++c) {
    bool seen = false;
    for (std::uint8_t g : grid) {
      if (g == c) seen = true;
    }
    if (!seen) all_classes = false;
  }
  CHECK(all_classes, "class grid covers 0..63");
  CUDA_CHECK(cudaMemcpy(d.grid, grid.data(), grid.size(),
                        cudaMemcpyHostToDevice),
             "upload class grid");

  for (int set = 0; set < 4; ++set) {
    const int q = q_for_set[set];
    if (q > max_q(bd)) continue;
    Image want(d.width, d.height, curr.stride);
    cpu_filter_grid_classes(curr, cdef, tx_skip, bd, q, mi_cols, mi_rows, grid,
                            &want);
    const Extents e = extents_for(d.width, d.height, 0);
    const gp::PcWienerSource src = make_gpu_source(d.curr, d.cdef, e);
    const gp::PcWienerParams params =
        make_gpu_params(q, bd, mi_cols, mi_rows, d.tx_skip, mi_cols);
    CHECK(is_ok(gp::pc_wiener_filter_grid(ws, src, params, d.grid, d.lr.ptr,
                                          d.lr.stride)),
          "all-class filter_grid status");
    CHECK(plane_matches(want, d.lr), "every class x filter set parity");
    ++g_cases;
  }

  // Spot-check the single-block explicit-class entry point.
  const int classes[] = {0, 1, 31, 63};
  for (int cls : classes) {
    const int q = q_for_set[cls % 4];
    if (q > max_q(bd)) continue;
    const Extents e = extents_for(d.width, d.height, 8);
    const gp::PcWienerSource src = make_gpu_source(d.curr, d.cdef, e);
    const gp::PcWienerParams params =
        make_gpu_params(q, bd, mi_cols, mi_rows, d.tx_skip, mi_cols);
    Image want(d.width, d.height, curr.stride);
    want.Fill(0xbeef);
    oracle_filter_class(curr, cdef, e, q, cls, bd, 8, 8, 4, 4, &want);
    CUDA_CHECK(cudaMemset(d.lr.ptr, 0xbe,
                          d.lr.stride * d.lr.height * sizeof(std::uint16_t)),
               "sentinel block plane");
    CHECK(is_ok(gp::pc_wiener_filter_class_block(ws, src, params, 8, 8, 4, 4,
                                                 cls, d.lr.ptr, d.lr.stride)),
          "filter_class_block status");
    const std::vector<std::uint16_t> got = download_plane(d.lr);
    bool ok = true;
    for (int y = 0; y < d.height; ++y) {
      for (int x = 0; x < d.width; ++x) {
        const std::uint16_t v =
            got[static_cast<std::size_t>(y) * d.lr.stride + x];
        const bool inside = x >= 8 && x < 12 && y >= 8 && y < 12;
        const std::uint16_t expect = inside ? want.At(x, y) : 0xbebe;
        if (v != expect) {
          ok = false;
          break;
        }
      }
      if (!ok) break;
    }
    CHECK(ok, "filter_class_block parity");
    ++g_cases;
  }
}

void test_unit_sizes(const DeviceSet& d, Arena& arena,
                     std::uint64_t mark_after_planes, const Image& curr,
                     const Image& cdef, const std::vector<std::uint8_t>& tx_skip,
                     int bd, int unit_size, const gp::PcWienerConfig& config) {
  arena.rewind(mark_after_planes);
  gp::PcWienerWorkspace ws;
  const Status created = ws.create(arena, config);
  CHECK(is_ok(created), "workspace create for unit size");
  if (!is_ok(created)) return;

  const int mi_cols = d.mi_cols;
  const int mi_rows = d.mi_rows;
  const int q = 140;
  const Extents e = extents_for(d.width, d.height, 0);
  const gp::PcWienerSource src = make_gpu_source(d.curr, d.cdef, e);
  const gp::PcWienerParams params =
      make_gpu_params(q, bd, mi_cols, mi_rows, d.tx_skip, mi_cols);
  CHECK(is_ok(gp::pc_wiener_classify(ws, src, params)), "unit classify status");
  std::vector<std::uint8_t> grid(static_cast<std::size_t>(mi_cols) * mi_rows);
  CUDA_CHECK(cudaMemcpy(grid.data(), ws.class_grid(), grid.size(),
                        cudaMemcpyDeviceToHost),
             "download unit class grid");

  const int unit_rows = count_units(unit_size, d.height);
  const int unit_cols = count_units(unit_size, d.width);
  std::vector<std::uint8_t> unit_type(
      static_cast<std::size_t>(unit_rows) * unit_cols);
  for (auto& v : unit_type) {
    v = static_cast<std::uint8_t>(random_range(0, 3));
  }
  CUDA_CHECK(cudaMemcpy(d.unit_type, unit_type.data(), unit_type.size(),
                        cudaMemcpyHostToDevice),
             "upload unit types");

  Image want(d.width, d.height, curr.stride);
  cpu_filter_units(curr, cdef, tx_skip, bd, q, mi_cols, mi_rows, grid,
                   unit_type, unit_size, &want);
  CHECK(is_ok(gp::pc_wiener_filter_units(ws, src, params, ws.class_grid(),
                                         d.unit_type, d.lr.ptr, d.lr.stride)),
        "filter_units status");
  CHECK(plane_matches(want, d.lr), "unit-gated parity");
  ++g_cases;
}

}  // namespace

int main() {
  int device = 0;
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
    std::fprintf(stderr, "no CUDA device\n");
    return 1;
  }
  std::printf("device: %s (sm_%d%d), %d MiB\n", prop.name, prop.major,
              prop.minor, static_cast<int>(prop.totalGlobalMem >> 20));

  constexpr int kWidth = 96;
  constexpr int kHeight = 72;
  constexpr int kStride = kWidth + 8;
  const int mi_cols = kWidth / 4;
  const int mi_rows = kHeight / 4;

  gp::PcWienerConfig config{};
  config.width = kWidth;
  config.height = kHeight;
  config.bit_depth = 8;
  config.unit_size = 64;

  std::size_t required = gp::pc_wiener_required_device_bytes(config);
  CHECK(required > 0, "required_device_bytes");

  // One-byte-short negative test, in its own arena.
  {
    Arena small;
    CHECK(is_ok(small.init(required - 1)), "short arena init");
    gp::PcWienerWorkspace ws;
    const Status status = ws.create(small, config);
    CHECK(status == Status::kArenaExhausted, "short arena rejected");
    CHECK(ws.required_bytes() == required, "short arena reports required size");
    CHECK(small.used() == 0, "short arena rewound");
    small.release();
  }

  const std::size_t plane_bytes =
      static_cast<std::size_t>(kStride) * kHeight * sizeof(std::uint16_t);
  const std::size_t misc_bytes =
      3u * static_cast<std::size_t>(mi_cols) * mi_rows + 4u + 256u;
  const std::uint64_t capacity =
      static_cast<std::uint64_t>(required) + 4 * plane_bytes + misc_bytes +
      64 * 1024;

  Arena arena;
  const Status init = arena.init(capacity);
  CHECK(is_ok(init), "arena init");
  if (!is_ok(init)) {
    std::fprintf(stderr, "arena init failed: %s\n", status_name(init));
    return 1;
  }

  Image curr(kWidth, kHeight, kStride);
  Image cdef(kWidth, kHeight, kStride);
  for (int y = 0; y < kHeight; ++y) {
    for (int x = 0; x < kWidth; ++x) {
      curr.At(x, y) = static_cast<std::uint16_t>(random_range(0, 255));
      cdef.At(x, y) = static_cast<std::uint16_t>(random_range(0, 255));
    }
  }
  std::vector<std::uint8_t> tx_skip(
      static_cast<std::size_t>(mi_cols) * mi_rows);
  for (auto& v : tx_skip) v = static_cast<std::uint8_t>(random_range(0, 1));

  DeviceSet d{};
  d.width = kWidth;
  d.height = kHeight;
  d.mi_cols = mi_cols;
  d.mi_rows = mi_rows;
  d.curr.ptr = arena.allocate_array<std::uint16_t>(
      static_cast<std::size_t>(kStride) * kHeight);
  d.curr.stride = kStride;
  d.curr.height = kHeight;
  d.cdef.ptr = arena.allocate_array<std::uint16_t>(
      static_cast<std::size_t>(kStride) * kHeight);
  d.cdef.stride = kStride;
  d.cdef.height = kHeight;
  d.lr.ptr = arena.allocate_array<std::uint16_t>(
      static_cast<std::size_t>(kStride) * kHeight);
  d.lr.stride = kStride;
  d.lr.height = kHeight;
  d.tx_skip = arena.allocate_array<std::uint8_t>(
      static_cast<std::size_t>(mi_cols) * mi_rows);
  d.grid = arena.allocate_array<std::uint8_t>(
      static_cast<std::size_t>(mi_cols) * mi_rows);
  d.grid_block = arena.allocate_array<std::uint8_t>(
      static_cast<std::size_t>(mi_cols) * mi_rows);
  d.unit_type = arena.allocate_array<std::uint8_t>(
      static_cast<std::size_t>(count_units(64, kHeight)) *
      count_units(64, kWidth));
  d.lut = arena.allocate_array<std::uint8_t>(256);
  CHECK(d.curr.ptr != nullptr && d.cdef.ptr != nullptr && d.lr.ptr != nullptr &&
            d.tx_skip != nullptr && d.grid != nullptr &&
            d.grid_block != nullptr && d.unit_type != nullptr &&
            d.lut != nullptr,
        "arena buffer allocation");

  CHECK(upload_plane(curr, d.curr), "upload curr");
  CHECK(upload_plane(cdef, d.cdef), "upload cdef");
  CUDA_CHECK(cudaMemcpy(d.tx_skip, tx_skip.data(), tx_skip.size(),
                        cudaMemcpyHostToDevice),
             "upload LrTxSkip");
  const std::uint64_t mark_after_planes = arena.mark();

  // Create the primary workspace once, then run two classify launches inside a
  // mark/rewind pair: launchers must not grow the arena.
  gp::PcWienerWorkspace ws;
  const Status created = ws.create(arena, config);
  CHECK(is_ok(created), "workspace create");
  if (!is_ok(created)) {
    std::fprintf(stderr, "create failed: %s\n", status_name(created));
    return 1;
  }
  CHECK(ws.required_bytes() == required, "workspace required size");
  CHECK(ws.persistent_bytes() + ws.scratch_bytes() == required,
        "workspace split");
  const std::size_t persistent_bytes = ws.persistent_bytes();
  const std::size_t scratch_bytes = ws.scratch_bytes();

  const Extents e0 = extents_for(kWidth, kHeight, 0);
  const gp::PcWienerSource src0 = make_gpu_source(d.curr, d.cdef, e0);
  const gp::PcWienerParams params0 =
      make_gpu_params(140, 8, mi_cols, mi_rows, d.tx_skip, mi_cols);
  const std::uint64_t launch_mark = arena.mark();
  CHECK(is_ok(gp::pc_wiener_classify(ws, src0, params0)), "launch 1");
  CHECK(arena.used() == launch_mark, "launch 1 did not grow arena");
  CHECK(is_ok(gp::pc_wiener_classify(ws, src0, params0)), "launch 2");
  CHECK(arena.used() == launch_mark, "launch 2 did not grow arena");
  arena.rewind(launch_mark);
  CHECK(arena.used() == launch_mark, "rewind is a no-op after launches");

  test_table_upload(ws, d.lut);

  // Main bit-depth and qindex sweep, plus skip mode and full-frame parity.
  for (int bd : {8, 10, 12}) {
    const int maxv = (1 << bd) - 1;
    for (int y = 0; y < kHeight; ++y) {
      for (int x = 0; x < kWidth; ++x) {
        curr.At(x, y) = static_cast<std::uint16_t>(random_range(0, maxv));
        cdef.At(x, y) = static_cast<std::uint16_t>(random_range(0, maxv));
      }
    }
    CHECK(upload_plane(curr, d.curr), "upload curr high bit depth");
    CHECK(upload_plane(cdef, d.cdef), "upload cdef high bit depth");
    CUDA_CHECK(cudaMemcpy(d.tx_skip, tx_skip.data(), tx_skip.size(),
                          cudaMemcpyHostToDevice),
               "upload LrTxSkip high bit depth");

    for (int q : {0, 64, 129, 130, 189, 190, 219, 220, 255, 303, 351}) {
      if (q > max_q(bd)) continue;
      arena.rewind(mark_after_planes);
      const gp::PcWienerConfig cfg = [&]() {
        gp::PcWienerConfig c{};
        c.width = kWidth;
        c.height = kHeight;
        c.bit_depth = bd;
        c.unit_size = 64;
        return c;
      }();
      gp::PcWienerWorkspace case_ws;
      const Status st = case_ws.create(arena, cfg);
      CHECK(is_ok(st), "case workspace create");
      if (!is_ok(st)) continue;
      test_skip_and_full_frame(d, case_ws, curr, cdef, tx_skip, bd, q);
    }

    arena.rewind(mark_after_planes);
    gp::PcWienerConfig class_cfg{};
    class_cfg.width = kWidth;
    class_cfg.height = kHeight;
    class_cfg.bit_depth = bd;
    class_cfg.unit_size = 64;
    gp::PcWienerWorkspace class_ws;
    const Status class_status = class_ws.create(arena, class_cfg);
    CHECK(is_ok(class_status), "class workspace create");
    if (is_ok(class_status)) {
      test_every_class(d, class_ws, curr, cdef, tx_skip, bd);
    }
  }

  // Every LR unit size (luma_shift 3..0) with random per-unit decisions.
  for (int bd : {8, 10, 12}) {
    const int maxv = (1 << bd) - 1;
    for (int y = 0; y < kHeight; ++y) {
      for (int x = 0; x < kWidth; ++x) {
        curr.At(x, y) = static_cast<std::uint16_t>(random_range(0, maxv));
        cdef.At(x, y) = static_cast<std::uint16_t>(random_range(0, maxv));
      }
    }
    CHECK(upload_plane(curr, d.curr), "unit upload curr");
    CHECK(upload_plane(cdef, d.cdef), "unit upload cdef");
    for (int unit_size : {64, 128, 256, 512}) {
      gp::PcWienerConfig cfg{};
      cfg.width = kWidth;
      cfg.height = kHeight;
      cfg.bit_depth = bd;
      cfg.unit_size = unit_size;
      test_unit_sizes(d, arena, mark_after_planes, curr, cdef, tx_skip, bd,
                      unit_size, cfg);
    }
  }

  std::printf(
      "pc_wiener_test: %ld cases, %d checks, %d failures\n"
      "  required_device_bytes=%zu persistent=%zu scratch=%zu\n"
      "  arena peak_used=%llu bytes (capacity=%llu)\n",
      g_cases, g_checks, g_failures, required, persistent_bytes,
      scratch_bytes, static_cast<unsigned long long>(arena.peak_used()),
      static_cast<unsigned long long>(arena.capacity()));

  arena.release();
  if (g_failures == 0) {
    std::printf("PC-WIENER CHECKS PASSED\n");
    return 0;
  }
  std::fprintf(stderr, "PC-WIENER: %d/%d checks FAILED\n", g_failures,
               g_checks);
  return 1;
}

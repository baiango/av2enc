// Host emulation of the shared intra-full device bodies.
//
// This TU compiles intra_full_device.cuh with the CUDA keywords stubbed and one
// emulated thread, then compares every result with the real CPU components:
//
//   * av2enc::m1::missing_bits::dc_intra_subsampled  (§7.13.2.11)
//   * av2enc::m1::missing_bits::filter_corner_*      (§7.13.2.14)
//   * av2enc::txv::forward_block / inverse_block     (§7.15, identity types)
//   * av2enc::idtx::derive_chroma_type               (§5.20.7.29)
//
// It also exercises the arena contract with the real Arena implementation
// (built against a host cuda_runtime stub): sizing from
// required_device_bytes(), a two-cycle mark()/rewind() scratch reuse check with
// no net growth, and a one-byte-short negative test.
//
// Build/run: ./build_and_test.sh
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define __device__
#define __host__
#define __forceinline__ inline
#define __shared__ static

struct EmuDim {
  unsigned x = 0;
  unsigned y = 0;
  unsigned z = 0;
};
static EmuDim threadIdx;
[[maybe_unused]] static EmuDim blockIdx;
static EmuDim blockDim;
static inline void __syncthreads() {}
static inline double __dadd_rn(double a, double b) { return a + b; }
static inline double __dmul_rn(double a, double b) { return a * b; }

#include "intra_full_device.cuh"

#include "encoder/intra_modes/intra_modes.h"
#include "encoder/m1/missing_bits/dc_intra_subsampled.h"
#include "encoder/m1/missing_bits/intra_filter_corner.h"
#include "encoder/transform/idtx_intra/idtx_intra.h"
#include "encoder/tx_type/tx_type.h"
#include "encoder/tx_type/tx_type_tables.h"
#include "encoder/tx_variants/tx_variants.h"

namespace {

using av2enc::cuda::intra_full::ChromaQuery;
using av2enc::cuda::intra_full::ChromaResult;
using av2enc::cuda::intra_full::Config;
using av2enc::cuda::intra_full::CornerJob;
using av2enc::cuda::intra_full::DcResult;
using av2enc::cuda::intra_full::DcSubsampledDesc;
using av2enc::cuda::intra_full::TxJob;
using av2enc::cuda::intra_full::Workspace;

long long g_checks = 0;
int g_failures = 0;

void expect(bool condition, const char* what, const char* detail, long long a,
            long long b) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    if (g_failures <= 20) {
      std::printf("FAIL: %s (%s got=%lld want=%lld)\n", what, detail, a, b);
    }
  }
}

#define EXPECT_TRUE(cond, what)       \
  do {                                \
    const bool ok_ = (cond);          \
    expect(ok_, what, "", 0, 0);      \
  } while (0)
#define EXPECT_EQ(a, b, what)                                            \
  do {                                                                   \
    const long long va_ = static_cast<long long>(a);                     \
    const long long vb_ = static_cast<long long>(b);                     \
    expect(va_ == vb_, what, #a " == " #b, va_, vb_);                    \
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

// ---------------------------------------------------------------------------
// Table / mapping parity
// ---------------------------------------------------------------------------

void test_tables() {
  for (int set = 0; set < 7; ++set) {
    for (int type = 0; type < 16; ++type) {
      EXPECT_EQ(av2enc::cuda::intra_full::kTxTypeInSetIntraDev[set][type],
                av2enc::txtype::kTxTypeInSetIntra[set][type],
                "Tx_Type_In_Set_Intra table");
    }
  }
  for (int mode = 0; mode < 14; ++mode) {
    EXPECT_EQ(av2enc::cuda::intra_full::kModeToTxfmDev[mode],
              static_cast<int>(av2enc::idtx::mode_to_txfm(mode)),
              "Mode_To_Txfm table");
  }
  for (int mode = 0; mode < 13; ++mode) {
    EXPECT_EQ(av2enc::cuda::intra_full::kModeToAngleDev[mode],
              av2enc::intra::mode_to_angle(static_cast<av2enc::intra::Mode>(mode)),
              "Mode_To_Angle table");
  }
  for (int i = 0; i < 25; ++i) {
    const int w = av2enc::cuda::intra_full::kTxDimsDev[i][0];
    const int h = av2enc::cuda::intra_full::kTxDimsDev[i][1];
    EXPECT_EQ(av2enc::txv::tx_size_index(w, h), i, "TX size table index");
    for (int t = 0; t < 16; ++t) {
      av2enc::txv::Tx1dType row = av2enc::txv::Tx1dType::kDct;
      av2enc::txv::Tx1dType col = av2enc::txv::Tx1dType::kDct;
      av2enc::txv::tx_1d_types(static_cast<av2enc::txv::TxType>(t), &row, &col);
      int dev_row = -1;
      int dev_col = -1;
      const bool identity =
          av2enc::cuda::intra_full::intra_identity_1d(t, &dev_row, &dev_col);
      EXPECT_EQ(identity ? 1 : 0,
                av2enc::cuda::intra_full::is_intra_identity_type(t) ? 1 : 0,
                "identity predicate");
      EXPECT_EQ(identity, av2enc::idtx::is_identity_type(
                              static_cast<av2enc::txv::TxType>(t)),
                "identity predicate vs planner");
      if (identity) {
        EXPECT_EQ(dev_row, static_cast<int>(row), "identity row type");
        EXPECT_EQ(dev_col, static_cast<int>(col), "identity col type");
      }
    }
  }
}

// ---------------------------------------------------------------------------
// §7.13.2.11 DC subsampled — every extent, availability and bit depth
// ---------------------------------------------------------------------------

void test_dc() {
  threadIdx.x = 0;
  blockDim.x = 1;
  for (int bit_depth = 8; bit_depth <= 12; bit_depth += 2) {
    const int high = (1 << bit_depth) - 1;
    for (int w = 1; w <= 64; ++w) {
      for (int h = 1; h <= 64; ++h) {
        std::vector<std::int16_t> above(static_cast<std::size_t>(w));
        std::vector<std::int16_t> left(static_cast<std::size_t>(h));
        for (int i = 0; i < w; ++i) {
          above[static_cast<std::size_t>(i)] =
              static_cast<std::int16_t>(random_range(0, high));
        }
        for (int i = 0; i < h; ++i) {
          left[static_cast<std::size_t>(i)] =
              static_cast<std::int16_t>(random_range(0, high));
        }
        for (int have_left = 0; have_left <= 1; ++have_left) {
          for (int have_above = 0; have_above <= 1; ++have_above) {
            std::vector<std::uint16_t> want(
                static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
            av2enc::m1::missing_bits::DcSubsampledInput cpu_in;
            cpu_in.above = above.data();
            cpu_in.left = left.data();
            cpu_in.w = w;
            cpu_in.h = h;
            cpu_in.bit_depth = bit_depth;
            cpu_in.have_above = have_above != 0;
            cpu_in.have_left = have_left != 0;
            av2enc::m1::missing_bits::DcSubsampledTrace cpu_trace;
            EXPECT_TRUE(av2enc::m1::missing_bits::dc_intra_subsampled(
                            cpu_in, want.data(), &cpu_trace),
                        "CPU DC subsampled");

            std::vector<std::uint16_t> got(want.size(), 0xFFFFu);
            DcSubsampledDesc dev_in;
            dev_in.above = above.data();
            dev_in.left = left.data();
            dev_in.w = w;
            dev_in.h = h;
            dev_in.bit_depth = bit_depth;
            dev_in.have_above = have_above;
            dev_in.have_left = have_left;
            DcResult dev_trace;
            av2enc::cuda::intra_full::intra_full_dc_subsampled_dev(
                dev_in, got.data(), w, &dev_trace);

            EXPECT_EQ(dev_trace.step_w, cpu_trace.step_w, "DC step_w");
            EXPECT_EQ(dev_trace.step_h, cpu_trace.step_h, "DC step_h");
            EXPECT_EQ(dev_trace.sum, cpu_trace.sum, "DC sum");
            EXPECT_EQ(dev_trace.count, cpu_trace.count, "DC count");
            EXPECT_EQ(dev_trace.avg, cpu_trace.avg, "DC avg");
            for (std::size_t i = 0; i < want.size(); ++i) {
              if (got[i] != want[i]) {
                EXPECT_EQ(got[i], want[i], "DC sample");
                return;  // one report is enough
              }
              ++g_checks;
            }
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// §7.13.2.14 filter corner
// ---------------------------------------------------------------------------

void test_corner() {
  // Guard over every intra extent and a superset of the reachable angles.
  for (int w = 1; w <= 64; ++w) {
    for (int h = 1; h <= 64; ++h) {
      for (int p_angle = 0; p_angle <= 270; ++p_angle) {
        const bool want =
            av2enc::m1::missing_bits::filter_corner_applies(w, h, p_angle);
        const bool got =
            av2enc::cuda::intra_full::intra_full_filter_corner_applies_dev(
                w, h, p_angle) != 0;
        EXPECT_EQ(got, want, "corner guard");
      }
    }
  }

  // 5/6/5 Round2(.,4) value over every 8-bit triple.
  for (int left0 = 0; left0 < 256; ++left0) {
    for (int above_m1 = 0; above_m1 < 256; ++above_m1) {
      for (int above0 = 0; above0 < 256; ++above0) {
        const int want = av2enc::m1::missing_bits::filter_corner_value(
            left0, above_m1, above0);
        const int got =
            av2enc::cuda::intra_full::intra_full_filter_corner_value_dev(
                left0, above_m1, above0);
        if (got != want) {
          EXPECT_EQ(got, want, "corner value");
          return;
        }
        ++g_checks;
      }
    }
  }

  // Application (both corner slots) over sampled guards.
  const int angles[] = {0, 45, 89, 90, 91, 120, 135, 179, 180, 181, 269};
  for (int w = 1; w <= 64; w += 3) {
    for (int h = 1; h <= 64; h += 3) {
      for (int angle : angles) {
        const std::int16_t above_m1 =
            static_cast<std::int16_t>(random_range(0, 255));
        const std::int16_t above0 =
            static_cast<std::int16_t>(random_range(0, 255));
        const std::int16_t left_m1 =
            static_cast<std::int16_t>(random_range(0, 255));
        const std::int16_t left0 =
            static_cast<std::int16_t>(random_range(0, 255));

        av2enc::intra::EdgeRow above_cpu;
        av2enc::intra::EdgeRow left_cpu;
        above_cpu[-1] = above_m1;
        above_cpu[0] = above0;
        left_cpu[-1] = left_m1;
        left_cpu[0] = left0;
        const bool applied_cpu =
            av2enc::m1::missing_bits::filter_corner_if_applicable(
                w, h, angle, above_cpu, left_cpu);

        std::int16_t above_dev[3];
        std::int16_t left_dev[2];
        above_dev[0] = above_m1;
        above_dev[1] = above0;
        left_dev[0] = left_m1;
        left_dev[1] = left0;
        std::int16_t* above_ptr = above_dev + 1;
        std::int16_t* left_ptr = left_dev + 1;
        const bool applied_dev =
            av2enc::cuda::intra_full::intra_full_filter_corner_apply_dev(
                w, h, angle, above_ptr, left_ptr) != 0;
        EXPECT_EQ(applied_dev, applied_cpu, "corner applied flag");
        EXPECT_EQ(above_ptr[-1], above_cpu[-1], "corner above[-1]");
        EXPECT_EQ(left_ptr[-1], left_cpu[-1], "corner left[-1]");
        EXPECT_EQ(above_ptr[0], above_cpu[0], "corner above[0]");
        EXPECT_EQ(left_ptr[0], left_cpu[0], "corner left[0]");
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Identity transforms — 7 TX_TYPEs x 25 sizes
// ---------------------------------------------------------------------------

void test_transforms() {
  threadIdx.x = 0;
  blockDim.x = 1;
  using av2enc::txv::TxSize;
  using av2enc::txv::TxType;

  for (int t = 9; t <= 15; ++t) {
    for (int s = 0; s < 25; ++s) {
      const auto params =
          av2enc::txv::make_tx_params(static_cast<TxSize>(s),
                                      static_cast<TxType>(t));
      if (!params.has_value()) {
        EXPECT_TRUE(false, "make_tx_params for supported size");
        continue;
      }
      const int adj_n = params->w * params->h;
      const int coded_n = params->coded_w * params->coded_h;
      for (int pattern = 0; pattern < 4; ++pattern) {
        std::vector<std::int16_t> residual(static_cast<std::size_t>(adj_n));
        for (int i = 0; i < adj_n; ++i) {
          switch (pattern) {
            case 0:
              residual[static_cast<std::size_t>(i)] =
                  static_cast<std::int16_t>(random_range(-255, 255));
              break;
            case 1:
              residual[static_cast<std::size_t>(i)] =
                  static_cast<std::int16_t>(i == adj_n / 2 ? 255 : 0);
              break;
            case 2:
              residual[static_cast<std::size_t>(i)] =
                  static_cast<std::int16_t>(i % 17 - 8);
              break;
            default:
              residual[static_cast<std::size_t>(i)] =
                  static_cast<std::int16_t>((i % 2) ? -255 : 255);
              break;
          }
        }

        std::vector<double> want_coeff(static_cast<std::size_t>(adj_n));
        std::vector<std::int32_t> levels(static_cast<std::size_t>(adj_n));
        std::vector<std::int32_t> dequant(static_cast<std::size_t>(adj_n));
        std::vector<std::int16_t> want_coded(
            static_cast<std::size_t>(coded_n));
        av2enc::txv::forward_block(residual.data(), *params, want_coeff.data());
        av2enc::txv::quantize(want_coeff.data(), *params, levels.data());
        av2enc::txv::dequantize(levels.data(), *params, dequant.data());
        av2enc::txv::inverse_block(dequant.data(), *params,
                                   want_coded.data());

        std::vector<double> got_coeff(static_cast<std::size_t>(adj_n));
        std::vector<double> tmp(static_cast<std::size_t>(adj_n));
        std::vector<std::int32_t> inter(static_cast<std::size_t>(adj_n));
        std::vector<std::int16_t> got_coded(static_cast<std::size_t>(coded_n));
        av2enc::cuda::intra_full::intra_full_forward_block_dev(
            residual.data(), got_coeff.data(), *params, tmp.data());
        std::memset(inter.data(), 0,
                    inter.size() * sizeof(std::int32_t));
        av2enc::cuda::intra_full::intra_full_inverse_block_dev(
            dequant.data(), got_coded.data(), *params, inter.data());

        for (int i = 0; i < adj_n; ++i) {
          const double a = want_coeff[static_cast<std::size_t>(i)];
          const double b = got_coeff[static_cast<std::size_t>(i)];
          if (std::memcmp(&a, &b, sizeof(double)) != 0) {
            EXPECT_TRUE(false, "forward coefficient bit-exact");
            return;
          }
          ++g_checks;
        }
        for (int i = 0; i < coded_n; ++i) {
          if (got_coded[static_cast<std::size_t>(i)] !=
              want_coded[static_cast<std::size_t>(i)]) {
            EXPECT_EQ(got_coded[static_cast<std::size_t>(i)],
                      want_coded[static_cast<std::size_t>(i)],
                      "inverse residual bit-exact");
            return;
          }
          ++g_checks;
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// §5.20.7.29 chroma derivation
// ---------------------------------------------------------------------------

void test_chroma() {
  for (int plane = 1; plane <= 2; ++plane) {
    for (int dctonly = 0; dctonly <= 1; ++dctonly) {
      for (int uv_mode = 0; uv_mode <= 13; ++uv_mode) {
        for (int delta = -3; delta <= 3; ++delta) {
          for (int s = 0; s < 25; ++s) {
            const int cw = av2enc::cuda::intra_full::kTxDimsDev[s][0];
            const int ch = av2enc::cuda::intra_full::kTxDimsDev[s][1];

            av2enc::idtx::SeqFlags flags;
            flags.enable_chroma_dctonly = dctonly != 0;
            av2enc::txv::TxType want_type = av2enc::txv::TxType::kDctDct;
            av2enc::idtx::Set want_set = av2enc::idtx::Set::kDctOnly;
            EXPECT_TRUE(av2enc::idtx::derive_chroma_type(
                            uv_mode, delta, cw, ch, flags, &want_type,
                            &want_set),
                        "CPU derive_chroma_type");

            ChromaQuery query;
            query.uv_mode = uv_mode;
            query.uv_angle_delta = delta;
            query.coded_w = cw;
            query.coded_h = ch;
            query.plane = plane;
            query.enable_chroma_dctonly = dctonly;
            ChromaResult got;
            const int valid =
                av2enc::cuda::intra_full::intra_full_derive_chroma_type_dev(
                    query, &got);
            EXPECT_EQ(valid, 1, "device derive valid");
            EXPECT_EQ(got.valid, 1, "device result valid");
            EXPECT_EQ(got.tx_type, static_cast<int>(want_type),
                      "chroma derived tx_type");
            EXPECT_EQ(got.set, static_cast<int>(want_set),
                      "chroma derived set");
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Arena contract
// ---------------------------------------------------------------------------

void test_arena() {
  {
    const Config defaults;
    std::printf("  sizing: default config required=%llu (persistent=%llu "
                "scratch=%llu)\n",
                static_cast<unsigned long long>(
                    av2enc::cuda::intra_full::required_device_bytes(defaults)),
                static_cast<unsigned long long>(
                    av2enc::cuda::intra_full::persistent_device_bytes(defaults)),
                static_cast<unsigned long long>(
                    av2enc::cuda::intra_full::scratch_device_bytes(defaults)));
  }
  Config config;
  config.max_dc_jobs = 8;
  config.max_corner_jobs = 512;
  config.max_chroma_jobs = 512;
  config.max_tx_blocks = 3;

  const std::uint64_t required =
      av2enc::cuda::intra_full::required_device_bytes(config);
  const std::uint64_t persistent =
      av2enc::cuda::intra_full::persistent_device_bytes(config);
  const std::uint64_t scratch =
      av2enc::cuda::intra_full::scratch_device_bytes(config);
  EXPECT_TRUE(required > 0, "required_device_bytes > 0");
  EXPECT_EQ(required, persistent + scratch, "required = persistent + scratch");
  EXPECT_EQ(required % av2enc::cuda::Arena::kAlignment, 0,
            "required is arena-aligned");
  EXPECT_EQ(persistent % av2enc::cuda::Arena::kAlignment, 0,
            "persistent is arena-aligned");
  EXPECT_EQ(scratch % av2enc::cuda::Arena::kAlignment, 0,
            "scratch is arena-aligned");

  // Invalid config has no requirement and is rejected distinctly.
  Config bad;
  bad.max_dc_jobs = 0;
  EXPECT_EQ(av2enc::cuda::intra_full::required_device_bytes(bad), 0,
            "invalid config requirement");
  {
    av2enc::cuda::Arena arena;
    EXPECT_TRUE(av2enc::cuda::is_ok(arena.init(required + 8192)),
                "arena init (invalid-config probe)");
    Workspace ws;
    std::uint64_t need = 123;
    EXPECT_EQ(static_cast<int>(ws.init(arena, bad, nullptr, &need)),
              static_cast<int>(av2enc::cuda::Status::kInvalidArgument),
              "invalid config status");
    EXPECT_EQ(need, 0, "invalid config required_out");
  }

  // Exact requirement succeeds; init allocates only the persistent arrays.
  av2enc::cuda::Arena arena;
  EXPECT_TRUE(av2enc::cuda::is_ok(arena.init(required + 4096)),
              "arena init");
  Workspace ws;
  std::string error;
  std::uint64_t need = 0;
  EXPECT_EQ(static_cast<int>(ws.init(arena, config, &error, &need)),
            static_cast<int>(av2enc::cuda::Status::kOk), "workspace init");
  EXPECT_EQ(need, required, "workspace required_out");
  EXPECT_EQ(arena.used(), persistent, "init uses persistent bytes only");
  EXPECT_TRUE(ws.dc_results_device() != nullptr, "dc result buffer");
  EXPECT_TRUE(ws.corner_results_device() != nullptr, "corner result buffer");
  EXPECT_TRUE(ws.chroma_results_device() != nullptr, "chroma result buffer");
  EXPECT_EQ(ws.scratch_bytes(), scratch, "workspace scratch bytes");

  // Two back-to-back "launches" of scratch staging leave used() flat.
  const std::uint64_t mark = arena.mark();
  for (int launch = 0; launch < 2; ++launch) {
    void* p = arena.allocate(scratch);
    EXPECT_TRUE(p != nullptr, "scratch allocation");
    EXPECT_EQ(arena.used(), mark + scratch, "scratch bump");
    EXPECT_TRUE(arena.rewind(mark), "scratch rewind");
    EXPECT_EQ(arena.used(), mark, "no net growth");
  }
  const std::uint64_t peak = arena.peak_used();
  EXPECT_TRUE(peak >= mark + scratch, "peak covers scratch");

  // One-byte-short arena: distinct status and the required size are reported.
  av2enc::cuda::Arena small;
  EXPECT_TRUE(av2enc::cuda::is_ok(small.init(required - 1)),
              "small arena init");
  Workspace ws_small;
  std::uint64_t short_need = 0;
  std::string short_error;
  const av2enc::cuda::Status status =
      ws_small.init(small, config, &short_error, &short_need);
  EXPECT_EQ(static_cast<int>(status),
            static_cast<int>(av2enc::cuda::Status::kArenaExhausted),
            "one-byte-short status");
  EXPECT_EQ(short_need, required, "one-byte-short required_out");
  EXPECT_EQ(small.used(), 0, "failed init leaves the arena untouched");
  EXPECT_TRUE(short_error.find(std::to_string(required)) != std::string::npos,
              "error names the required size");
  EXPECT_TRUE(!ws_small.initialized(), "failed workspace is uninitialized");
}

}  // namespace

int main() {
  std::printf("intra_full host emulation test\n");
  test_tables();
  test_dc();
  test_corner();
  test_transforms();
  test_chroma();
  test_arena();

  std::printf(
      "intra_full_emulation_test: %s (%lld checks, %d failures)\n",
      g_failures == 0 ? "PASS" : "FAIL", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}

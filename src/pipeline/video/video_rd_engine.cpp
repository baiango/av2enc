// RD engine translation unit (docs/rd-engine.md, lane L1).
//
// The engine itself is header-only (`video_rd_engine.h`): the video pipeline's
// source list is an explicit CMake target owned by the integrator, and a lane
// may not edit `CMakeLists.txt`.  This TU is the CMake anchor the integrator
// adds once the lane lands, and it carries the standalone family-split
// self-check:
//
//   c++ -std=c++20 -Isrc -DAV2ENC_RD_ENGINE_SELF_TEST \
//       src/pipeline/video/video_rd_engine.cpp \
//       build-dev-cache/libav2enc_core.a -o /tmp/rd_engine_selftest
//   /tmp/rd_engine_selftest
//
// The check is the property the audit relies on: the per-family split from
// `block_rate()` sums to the §8.3 model total (`Model::scalar_bits()`) for the
// same block, because the split is produced by the model's own SymbolSink and
// never by a second implementation.
#include "pipeline/video/video_rd_engine.h"

namespace av2enc::video::rd_engine {

#ifdef AV2ENC_RD_ENGINE_SELF_TEST

namespace {

struct Case {
  int tx_size;
  int tx_type;
  int plane;
};

// Deterministic sparse levels in scan order: a few low-frequency coefficients
// with magnitudes that exercise base, BR, escape and the DC/non-DC sign paths.
std::vector<std::int32_t> scan_levels(int area, int seed) {
  std::vector<std::int32_t> levels(static_cast<std::size_t>(area), 0);
  std::uint32_t state = static_cast<std::uint32_t>(seed) * 2654435761U + 1U;
  for (int i = 0; i < area; ++i) {
    state = state * 1664525U + 1013904223U;
    const int value = static_cast<int>((state >> 24) & 7U);
    if (i < area / 8 && value > 4) {
      levels[static_cast<std::size_t>(i)] = (value & 1) ? -value : value;
    }
  }
  levels[0] = seed % 2 == 0 ? -3 : 2;
  return levels;
}

double model_total(const BlockPlan& block, int qctx_slice) {
  const auto params_opt = txv::make_tx_params(
      static_cast<txv::TxSize>(block.tx_size),
      static_cast<txv::TxType>(block.tx_type));
  if (!params_opt.has_value()) return -1.0;
  const txv::TxParams& params = *params_opt;
  // `Model::init` takes scan-order levels, exactly what `BlockPlan` carries.
  const std::span<const std::int32_t> levels(block.levels_scan,
                                             static_cast<std::size_t>(block.area));
  av2::BlockInfo info;
  info.plane = block.plane;
  info.params = params;
  info.qctx = qctx_slice;
  av2::Model model;
  std::string error;
  if (!model.init(info, levels, &error)) return -1.0;
  return model.scalar_bits();
}

}  // namespace

int run() {
  const Case cases[] = {
      {0, 0, 0},    // TX_4X4 DCT
      {4, 0, 0},    // TX_8X8
      {5, 5, 0},    // TX_8X16 DCT_ADST
      {9, 0, 0},    // TX_16X16
      {13, 0, 0},   // TX_32X32
      {16, 0, 0},   // TX_64X64 (32x32 adjusted grid)
      {4, 0, 1},    // chroma TX_8X8
  };
  int failures = 0;
  for (const Case& test : cases) {
    const int area = txpart::tx_scan_area(test.tx_size);
    if (area <= 0) {
      std::fprintf(stderr, "selftest: bad tx_size %d\n", test.tx_size);
      ++failures;
      continue;
    }
    for (int seed = 1; seed <= 8; ++seed) {
      const std::vector<std::int32_t> levels = scan_levels(area, seed);
      BlockPlan block;
      block.tx_size = test.tx_size;
      block.tx_type = test.tx_type;
      block.plane = test.plane;
      block.area = area;
      block.levels_scan = levels.data();
      for (int qctx_slice = 0; qctx_slice < 4; ++qctx_slice) {
        RateBreakdown split;
        if (!block_rate(block, /*is_inter=*/false, qctx_slice,
                        /*cdf=*/nullptr, &split)) {
          std::fprintf(stderr, "selftest: block_rate rejected %d/%d\n",
                       test.tx_size, test.tx_type);
          ++failures;
          continue;
        }
        const double expected = model_total(block, qctx_slice);
        const double got = split.total();
        if (expected < 0.0 || std::abs(expected - got) > 1e-6 * (1.0 + expected)) {
          std::fprintf(stderr,
                       "selftest: split != model tx=%d type=%d qctx=%d "
                       "expected=%.6f got=%.6f\n",
                       test.tx_size, test.tx_type, qctx_slice, expected, got);
          ++failures;
        }
        if (split.get(kFamOther) != 0.0) {
          std::fprintf(stderr, "selftest: unclassified family %.6f\n",
                       split.get(kFamOther));
          ++failures;
        }
      }
    }
  }
  std::fprintf(stderr, "rd-engine selftest: %s (%d failures)\n",
               failures == 0 ? "PASS" : "FAIL", failures);
  return failures == 0 ? 0 : 1;
}

#endif  // AV2ENC_RD_ENGINE_SELF_TEST

}  // namespace av2enc::video::rd_engine

#ifdef AV2ENC_RD_ENGINE_SELF_TEST
int main() { return av2enc::video::rd_engine::run(); }
#endif

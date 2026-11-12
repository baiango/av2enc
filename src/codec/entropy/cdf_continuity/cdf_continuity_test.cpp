// Independent tests for the AV2 v1.0.0 tile/frame CDF continuity layer.
//
// Every expected value below is recomputed from the specification text in this
// file; the controller is never used to derive its own expectation.  The
// entropy carry test at the end is in `lockstep_stream_test.cpp`.
//
// Build:
//   ./src/encoder/entropy/continuity/build_and_test.sh
//   GATE=0 ./src/encoder/entropy/continuity/build_and_test.sh   # unit only
//   SANITIZE=1 WERROR=1 ./src/encoder/entropy/continuity/build_and_test.sh

#include "codec/entropy/cdf_continuity/cdf_continuity.h"
#include "codec/entropy/qctx/qctx.h"
#include "core/bitstream/bitwriter.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

namespace cont = av2enc::entropy::continuity;
namespace acdf = av2enc::entropy::avg_cdf;
using av2enc::BitWriter;
using cont::ContinuityStatus;
using cont::FrameCdfContinuity;
using cont::FrameCdfInput;
using cont::FrameCdfPlan;
using cont::FrameCdfSource;
using cont::TileInfoCdfField;

int g_checks = 0;
int g_failures = 0;

#define CHECK(cond)                                                        \
  do {                                                                     \
    ++g_checks;                                                            \
    if (!(cond)) {                                                         \
      ++g_failures;                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    }                                                                      \
  } while (false)

#define CHECK_EQ(a, b)                                                     \
  do {                                                                     \
    ++g_checks;                                                            \
    const auto check_a = (a);                                              \
    const auto check_b = (b);                                              \
    if (!(check_a == check_b)) {                                           \
      ++g_failures;                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s == %s (%lld vs %lld)\n",        \
                   __FILE__, __LINE__, #a, #b,                             \
                   static_cast<long long>(check_a),                        \
                   static_cast<long long>(check_b));                       \
    }                                                                      \
  } while (false)

#define CHECK_STATUS(expr, expected)                                       \
  do {                                                                     \
    ++g_checks;                                                            \
    const ContinuityStatus check_status = (expr);                          \
    if (check_status != (expected)) {                                      \
      ++g_failures;                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s -> %s (expected %s)\n",         \
                   __FILE__, __LINE__, #expr,                              \
                   cont::continuity_status_name(check_status),             \
                   cont::continuity_status_name(expected));                \
    }                                                                      \
  } while (false)

constexpr std::uint16_t kDefaultA[4] = {10000, 20000, 56, 0};
constexpr std::uint16_t kDefaultB[5] = {8000, 16000, 24000, 40, 0};

constexpr std::uint16_t kSpecScale = 1u << 15;

// §8.2.4 avg_cdf(), transcribed from the specification rather than called.
void spec_average_cdf(std::uint16_t* cdf, const std::uint16_t* tile_cdf,
                      std::uint16_t symbols, std::uint32_t num_log2,
                      bool first_tile) {
  const int n = static_cast<int>(symbols);
  if (first_tile) {
    for (int i = 0; i < n - 1; ++i) cdf[i] = 1u << 15;
    cdf[n - 1] = tile_cdf[n - 1];
    cdf[n] = 0;
  }
  for (int i = 0; i < n - 1; ++i) {
    cdf[i] = static_cast<std::uint16_t>(
        cdf[i] - ((kSpecScale - tile_cdf[i]) >> num_log2));
  }
  cdf[n] = static_cast<std::uint16_t>(
      cdf[n] + static_cast<std::uint16_t>(tile_cdf[n] >> num_log2));
}

// §7.4.4 blend_cdf(), transcribed.
void spec_blend_cdf(std::uint16_t* cdf, const std::uint16_t* saved_cdf,
                    std::uint16_t symbols) {
  const int n = static_cast<int>(symbols);
  for (int i = 0; i < n - 1; ++i) {
    const std::uint32_t saved = saved_cdf[i];
    const std::uint32_t current = cdf[i];
    cdf[i] = static_cast<std::uint16_t>(
        kSpecScale - (((kSpecScale - saved) +
                       7u * (kSpecScale - current) + 4u) >> 3));
  }
  cdf[n] = static_cast<std::uint16_t>(
      (saved_cdf[n] + 7u * cdf[n] + 4u) >> 3);
}

// §7.5 frame_end_update_cdf(), transcribed.
void spec_frame_end(std::uint16_t* frame, const std::uint16_t* saved,
                    std::uint16_t symbols) {
  const int n = static_cast<int>(symbols);
  for (int i = 0; i < n + 1; ++i) frame[i] = saved[i];
  frame[n] = static_cast<std::uint16_t>((3u * saved[n]) >> 2);
}

std::uint32_t spec_floor_log2(std::uint32_t v) {
  std::uint32_t r = 0;
  while (v > 1u) {
    v >>= 1u;
    ++r;
  }
  return r;
}

std::uint32_t read_bits_msb(const std::uint8_t* bytes, std::size_t bit_pos,
                            unsigned count) {
  std::uint32_t value = 0;
  for (unsigned i = 0; i < count; ++i) {
    const std::size_t pos = bit_pos + i;
    const unsigned bit =
        (bytes[pos >> 3u] >> (7u - static_cast<unsigned>(pos & 7u))) & 1u;
    value = (value << 1u) | bit;
  }
  return value;
}

// ---------------------------------------------------------------------------
// §5.18.7.2 tile_info() CDF tail
// ---------------------------------------------------------------------------

void test_tile_info_field() {
  {
    const TileInfoCdfField f =
        cont::tile_info_cdf_field(false, false, 1, 1);
    CHECK(!f.cdf_tail_coded);
    CHECK(!f.context_update_present);
    CHECK_EQ(f.context_update_bits, 0U);
    CHECK_EQ(f.tail_bits(), 0U);
  }
  {
    // 2x1 tiles: TileColsLog2=1, TileRowsLog2=0 -> f(1), plus f(2) sizes.
    const TileInfoCdfField f =
        cont::tile_info_cdf_field(false, false, 2, 1);
    CHECK(f.cdf_tail_coded);
    CHECK(f.context_update_present);
    CHECK_EQ(f.tile_cols_log2, 1U);
    CHECK_EQ(f.tile_rows_log2, 0U);
    CHECK_EQ(f.context_update_bits, 1U);
    CHECK_EQ(f.context_update_max, 1U);
    CHECK_EQ(f.tail_bits(), 3U);
  }
  {
    // 2x2 tiles: 2 + 2 -> f(2).
    const TileInfoCdfField f =
        cont::tile_info_cdf_field(false, false, 2, 2);
    CHECK_EQ(f.context_update_bits, 2U);
    CHECK_EQ(f.context_update_max, 3U);
    CHECK_EQ(f.tail_bits(), 4U);
  }
  {
    // 4x2: CeilLog2(4)+CeilLog2(2) = 3.
    const TileInfoCdfField f =
        cont::tile_info_cdf_field(false, false, 4, 2);
    CHECK_EQ(f.tile_cols_log2, 2U);
    CHECK_EQ(f.tile_rows_log2, 1U);
    CHECK_EQ(f.context_update_bits, 3U);
    CHECK_EQ(f.tail_bits(), 5U);
  }
  {
    // 3x3 non-uniform: CeilLog2(3)=2 on each axis -> 4 bits.
    const TileInfoCdfField f =
        cont::tile_info_cdf_field(false, false, 3, 3);
    CHECK_EQ(f.context_update_bits, 4U);
    CHECK_EQ(f.tail_bits(), 6U);
  }
  {
    // enable_avg_cdf && avg_cdf_type: the field disappears (§5.18.7.2), but
    // tile_size_bytes_minus_1 f(2) remains.
    const TileInfoCdfField f =
        cont::tile_info_cdf_field(true, true, 2, 2);
    CHECK(f.cdf_tail_coded);
    CHECK(!f.context_update_present);
    CHECK_EQ(f.tail_bits(), 2U);
  }
  {
    // enable_avg_cdf && !avg_cdf_type keeps the copyCdf field.
    const TileInfoCdfField f =
        cont::tile_info_cdf_field(true, false, 2, 2);
    CHECK(f.context_update_present);
    CHECK_EQ(f.tail_bits(), 4U);
  }
  {
    // Bridge frames and TIP_FRAME_AS_OUTPUT skip the whole tail.
    const TileInfoCdfField bridge =
        cont::tile_info_cdf_field(false, false, 2, 2, true, false);
    CHECK(!bridge.cdf_tail_coded);
    CHECK_EQ(bridge.tail_bits(), 0U);
    const TileInfoCdfField tip =
        cont::tile_info_cdf_field(false, false, 2, 2, false, true);
    CHECK(!tip.cdf_tail_coded);
    CHECK_EQ(tip.tail_bits(), 0U);
  }
  {
    // Endianness/bit-count witness: the field really costs tail_bits().
    BitWriter writer;
    const TileInfoCdfField f =
        cont::tile_info_cdf_field(false, false, 2, 2);
    writer.put_bits(3U, f.context_update_bits);
    writer.put_bits(0U, 2U);  // tile_size_bytes_minus_1
    CHECK_EQ(writer.bit_size(), f.tail_bits());
    CHECK_EQ(read_bits_msb(writer.bytes().data(), 0, 2), 3U);
  }
}

void test_context_update_validation() {
  const TileInfoCdfField two_by_two =
      cont::tile_info_cdf_field(false, false, 2, 2);
  std::string error;
  CHECK(cont::validate_context_update_tile_id(two_by_two, 0, &error));
  CHECK(cont::validate_context_update_tile_id(two_by_two, 3, &error));
  CHECK(!cont::validate_context_update_tile_id(two_by_two, 4, &error));
  CHECK(!error.empty());

  const TileInfoCdfField avg =
      cont::tile_info_cdf_field(true, true, 2, 2);
  CHECK(cont::validate_context_update_tile_id(avg, 0, &error));
  CHECK(!cont::validate_context_update_tile_id(avg, 1, &error));

  const TileInfoCdfField single =
      cont::tile_info_cdf_field(false, false, 1, 1);
  CHECK(cont::validate_context_update_tile_id(single, 0, &error));
  CHECK(!cont::validate_context_update_tile_id(single, 1, &error));
}

// ---------------------------------------------------------------------------
// §8.2.4 copyCdf / avgCdf decision
// ---------------------------------------------------------------------------

void test_tile_exit_plan() {
  for (std::uint32_t tile_count = 1; tile_count <= 64; ++tile_count) {
    const std::uint32_t expect_log2 =
        tile_count <= 1 ? 0
                        : std::min(3u, spec_floor_log2(tile_count));
    const std::uint32_t expect_limit = 1u << expect_log2;

    for (std::uint32_t tile = 0; tile < tile_count; ++tile) {
      const cont::TileExitPlan copy =
          cont::plan_tile_exit(false, false, tile, tile_count, 1, 0);
      CHECK_EQ(copy.num_log2, expect_log2);
      CHECK_EQ(copy.average_limit, expect_limit);
      CHECK(!copy.avg_cdf);
      CHECK_EQ(copy.copy_cdf, tile == 0u);

      const cont::TileExitPlan avg =
          cont::plan_tile_exit(true, true, tile, tile_count, 1, 0);
      CHECK(!avg.copy_cdf);
      CHECK_EQ(avg.avg_cdf, tile < expect_limit);

      // avg_cdf_type=0 still takes the copyCdf branch (§8.2.4 `else`).
      const cont::TileExitPlan copy_type0 =
          cont::plan_tile_exit(true, false, tile, tile_count, 1,
                               tile_count - 1u);
      CHECK(!copy_type0.avg_cdf);
      CHECK_EQ(copy_type0.copy_cdf, tile == tile_count - 1u);
    }
  }

  // 9 tiles: numLog2 = 3, only tiles 0..7 average.
  const cont::TileExitPlan t8 = cont::plan_tile_exit(true, true, 8, 3, 3, 0);
  CHECK_EQ(t8.num_log2, 3U);
  CHECK_EQ(t8.average_limit, 8U);
  CHECK(!t8.avg_cdf);
  CHECK(!t8.copy_cdf);
}

// ---------------------------------------------------------------------------
// §6.17.2 qctx slice
// ---------------------------------------------------------------------------

void test_qctx_slice() {
  struct Case {
    int q;
    int slice;
  };
  const Case cases[] = {
      {-1, 0},  {0, 0},   {89, 0},  {90, 0},  {91, 1},
      {140, 1}, {141, 2}, {190, 2}, {191, 3}, {255, 3}, {303, 3},
  };
  for (const Case& c : cases) {
    CHECK_EQ(cont::qctx_slice_for_base_q_idx(c.q), c.slice);
  }
  // The canonical table owner must agree.
  for (int q = -1; q <= 400; ++q) {
    CHECK_EQ(cont::qctx_slice_for_base_q_idx(q),
             av2enc::still::qctx::slice_for_base_q_idx(q));
  }
  CHECK_EQ(cont::kCoeffCdfSliceCount, av2enc::still::qctx::kSliceCount);
}

// ---------------------------------------------------------------------------
// §5.18.2 / §6.17.2 / §7.4.4 frame installation plan
// ---------------------------------------------------------------------------

void test_frame_plan() {
  // CLK/OLK: FrameIsIntra forces the defaults regardless of other flags.
  {
    FrameCdfInput in;
    in.keyframe = true;
    in.primary_ref_slot = 7;
    const FrameCdfPlan plan = cont::plan_frame_cdfs(in);
    CHECK(plan.reset);
    CHECK_EQ(static_cast<int>(plan.source),
             static_cast<int>(FrameCdfSource::kDefaults));
    CHECK(plan.primary_ref_none);
    CHECK(!plan.blend);
    CHECK_EQ(plan.qctx_slice, 0);
  }
  // Signalled PRIMARY_REF_NONE resets.
  {
    FrameCdfInput in;
    in.primary_ref_none = true;
    in.primary_ref_slot = 3;
    const FrameCdfPlan plan = cont::plan_frame_cdfs(in);
    CHECK(plan.reset);
    CHECK_EQ(static_cast<int>(plan.source),
             static_cast<int>(FrameCdfSource::kDefaults));
  }
  // Derived PRIMARY_REF_NONE also resets and forces the header bit.
  {
    FrameCdfInput in;
    in.derived_primary_ref_none = true;
    const FrameCdfPlan plan = cont::plan_frame_cdfs(in);
    CHECK(plan.reset);
    CHECK(plan.primary_ref_none);
  }
  // disable_cross_frame_cdf_init resets even with a real primary ref.
  {
    FrameCdfInput in;
    in.primary_ref_slot = 2;
    in.disable_cross_frame_cdf_init = true;
    const FrameCdfPlan plan = cont::plan_frame_cdfs(in);
    CHECK(plan.reset);
    CHECK_EQ(static_cast<int>(plan.source),
             static_cast<int>(FrameCdfSource::kDefaults));
  }
  // Carry-over: load_cdfs(ref_frame_idx[primary_ref_frame]).
  {
    FrameCdfInput in;
    in.primary_ref_slot = 5;
    in.raw_base_q_idx = 191;
    const FrameCdfPlan plan = cont::plan_frame_cdfs(in);
    CHECK(!plan.reset);
    CHECK_EQ(static_cast<int>(plan.source),
             static_cast<int>(FrameCdfSource::kPrimaryReference));
    CHECK_EQ(plan.primary_ref_slot, 5U);
    CHECK_EQ(plan.qctx_slice, 3);
    CHECK(!plan.blend);
  }
  // Blend: only enable_avg_cdf && !avg_cdf_type && blend frame && !bru &&
  // !TIP_FRAME_AS_OUTPUT.
  {
    FrameCdfInput in;
    in.primary_ref_slot = 5;
    in.enable_avg_cdf = true;
    in.avg_cdf_type = false;
    in.blend_ref_present = true;
    in.blend_ref_slot = 9;
    const FrameCdfPlan plan = cont::plan_frame_cdfs(in);
    CHECK(!plan.reset);
    CHECK(plan.blend);
    CHECK_EQ(static_cast<int>(plan.source),
             static_cast<int>(FrameCdfSource::kPrimaryReferenceBlended));
    CHECK_EQ(plan.blend_ref_slot, 9U);
  }
  {
    FrameCdfInput in;
    in.primary_ref_slot = 5;
    in.enable_avg_cdf = true;
    in.avg_cdf_type = true;  // avg/type 1: no blend
    in.blend_ref_present = true;
    const FrameCdfPlan plan = cont::plan_frame_cdfs(in);
    CHECK(!plan.blend);
    CHECK_EQ(static_cast<int>(plan.source),
             static_cast<int>(FrameCdfSource::kPrimaryReference));
  }
  {
    FrameCdfInput in;
    in.primary_ref_slot = 5;
    in.enable_avg_cdf = true;
    in.avg_cdf_type = false;
    in.blend_ref_present = true;
    in.bru_inactive = true;
    CHECK(!cont::plan_frame_cdfs(in).blend);
  }
  {
    FrameCdfInput in;
    in.primary_ref_slot = 5;
    in.enable_avg_cdf = true;
    in.avg_cdf_type = false;
    in.blend_ref_present = true;
    in.tip_frame_as_output = true;
    CHECK(!cont::plan_frame_cdfs(in).blend);
  }
  // Reset beats blend.
  {
    FrameCdfInput in;
    in.disable_cross_frame_cdf_init = true;
    in.enable_avg_cdf = true;
    in.avg_cdf_type = false;
    in.blend_ref_present = true;
    const FrameCdfPlan plan = cont::plan_frame_cdfs(in);
    CHECK(plan.reset);
    CHECK(!plan.blend);
  }
  // Validation: out-of-range reference slots and context ids.
  {
    FrameCdfInput in;
    in.primary_ref_slot = cont::kNumRefFrames;
    std::string error;
    CHECK(!cont::validate_frame_cdfs(in, &error));
    CHECK(!error.empty());
  }
  {
    FrameCdfInput in;
    in.tile_cols = 2;
    in.tile_rows = 2;
    in.context_update_tile_id = 4;
    std::string error;
    CHECK(!cont::validate_frame_cdfs(in, &error));
    CHECK(!error.empty());
  }
  {
    FrameCdfInput in;
    in.tile_cols = 2;
    in.tile_rows = 2;
    in.enable_avg_cdf = true;
    in.avg_cdf_type = true;
    in.context_update_tile_id = 1;  // absent field must be 0
    std::string error;
    CHECK(!cont::validate_frame_cdfs(in, &error));
  }
}

// ---------------------------------------------------------------------------
// Controller: copyCdf selection
// ---------------------------------------------------------------------------

void init_controller(FrameCdfContinuity* ctl, std::uint32_t contexts) {
  const acdf::CdfSpec specs[] = {
      acdf::CdfSpec{3, kDefaultA},
      acdf::CdfSpec{4, kDefaultB},
  };
  std::string error;
  CHECK(ctl->init_owned(specs, contexts, &error));
  CHECK(error.empty());
}

void fill_tile(FrameCdfContinuity* ctl, std::uint32_t tag) {
  for (std::size_t t = 0; t < ctl->bank().table_count(); ++t) {
    acdf::CdfTable& table = ctl->bank().table(t);
    const std::size_t entries = static_cast<std::size_t>(table.symbols) + 1u;
    for (std::size_t i = 0; i < entries; ++i) {
      table.tile[i] = static_cast<std::uint16_t>(
          (1000u * (tag + 1u) + 37u * i + t * 101u) & 0x7FFFu);
    }
    table.tile[table.symbols - 1] = 56;
  }
}

void test_controller_copy_cdf() {
  FrameCdfContinuity ctl;
  init_controller(&ctl, cont::kNumRefFrames);
  FrameCdfInput in;
  in.tile_cols = 2;
  in.tile_rows = 2;
  in.context_update_tile_id = 2;
  in.disable_cross_frame_cdf_init = true;
  std::string error;
  CHECK_STATUS(ctl.begin_frame(in, &error), ContinuityStatus::kOk);
  CHECK_EQ(ctl.tile_count(), 4U);

  std::array<std::array<std::uint16_t, 5>, 2> selected_snapshot{};
  for (std::uint32_t tile = 0; tile < 4; ++tile) {
    CHECK_STATUS(ctl.begin_tile(&error), ContinuityStatus::kOk);
    fill_tile(&ctl, tile);
    if (tile == 2u) {
      for (std::size_t t = 0; t < ctl.bank().table_count(); ++t) {
        const acdf::CdfTable& table = ctl.bank().table(t);
        const std::size_t entries =
            static_cast<std::size_t>(table.symbols) + 1u;
        for (std::size_t i = 0; i < entries; ++i) {
          selected_snapshot[t][i] = table.tile[i];
        }
      }
    }
    CHECK_STATUS(ctl.end_tile(tile, &error), ContinuityStatus::kOk);
    // Only tile 2 has replaced Saved*, and it stays installed for tile 3.
    for (std::size_t t = 0; t < ctl.bank().table_count(); ++t) {
      const acdf::CdfTable& table = ctl.bank().table(t);
      const std::size_t entries =
          static_cast<std::size_t>(table.symbols) + 1u;
      if (tile >= 2u) {
        CHECK(std::memcmp(table.saved, selected_snapshot[t].data(),
                          entries * sizeof(std::uint16_t)) == 0);
      } else {
        CHECK(std::memcmp(table.saved, table.defaults,
                          entries * sizeof(std::uint16_t)) == 0);
      }
    }
  }
  CHECK_STATUS(ctl.end_frame(&error), ContinuityStatus::kOk);

  // §7.5: frame = Saved, count = (3 * count) >> 2.
  for (std::size_t t = 0; t < ctl.bank().table_count(); ++t) {
    const acdf::CdfTable& table = ctl.bank().table(t);
    const std::size_t entries = static_cast<std::size_t>(table.symbols) + 1u;
    std::array<std::uint16_t, 9> expect{};
    spec_frame_end(expect.data(), table.saved, table.symbols);
    CHECK(std::memcmp(table.frame, expect.data(),
                      entries * sizeof(std::uint16_t)) == 0);
  }
}

// ---------------------------------------------------------------------------
// Controller: avgCdf accumulation and the 1<<numLog2 cutoff
// ---------------------------------------------------------------------------

void test_controller_avg_cdf() {
  FrameCdfContinuity ctl;
  init_controller(&ctl, cont::kNumRefFrames);
  FrameCdfInput in;
  in.tile_cols = 3;
  in.tile_rows = 3;  // 9 tiles, only the first 8 average
  in.enable_avg_cdf = true;
  in.avg_cdf_type = true;
  in.disable_cross_frame_cdf_init = true;
  std::string error;
  CHECK_STATUS(ctl.begin_frame(in, &error), ContinuityStatus::kOk);

  std::array<std::array<std::uint16_t, 9>, 2> expect{};
  for (std::uint32_t tile = 0; tile < 9; ++tile) {
    CHECK_STATUS(ctl.begin_tile(&error), ContinuityStatus::kOk);
    fill_tile(&ctl, tile);
    for (std::size_t t = 0; t < ctl.bank().table_count(); ++t) {
      const acdf::CdfTable& table = ctl.bank().table(t);
      if (tile < 8u) {
        spec_average_cdf(expect[t].data(), table.tile, table.symbols, 3,
                         tile == 0u);
      }
    }
    CHECK_STATUS(ctl.end_tile(tile, &error), ContinuityStatus::kOk);
    for (std::size_t t = 0; t < ctl.bank().table_count(); ++t) {
      const acdf::CdfTable& table = ctl.bank().table(t);
      const std::size_t entries =
          static_cast<std::size_t>(table.symbols) + 1u;
      CHECK(std::memcmp(table.saved, expect[t].data(),
                        entries * sizeof(std::uint16_t)) == 0);
    }
  }
  CHECK_STATUS(ctl.end_frame(&error), ContinuityStatus::kOk);
}

// ---------------------------------------------------------------------------
// Controller: reference save/load/blend
// ---------------------------------------------------------------------------

void test_controller_reference_blend() {
  FrameCdfContinuity ctl;
  init_controller(&ctl, cont::kNumRefFrames);
  std::string error;

  // Frame 0: defaults, deliberately adapted to a known state.
  FrameCdfInput reset;
  reset.tile_cols = 1;
  reset.tile_rows = 1;
  reset.keyframe = true;
  CHECK_STATUS(ctl.begin_frame(reset, &error), ContinuityStatus::kOk);
  CHECK_STATUS(ctl.begin_tile(&error), ContinuityStatus::kOk);
  fill_tile(&ctl, 1);
  CHECK_STATUS(ctl.end_tile(0, &error), ContinuityStatus::kOk);
  CHECK_STATUS(ctl.end_frame(&error), ContinuityStatus::kOk);
  CHECK_STATUS(ctl.save_reference(5, &error), ContinuityStatus::kOk);

  // Snapshot the post-frame-end frame arrays saved into slot 5.
  std::array<std::array<std::uint16_t, 9>, 2> ref5{};
  for (std::size_t t = 0; t < ctl.bank().table_count(); ++t) {
    const acdf::CdfTable& table = ctl.bank().table(t);
    std::memcpy(ref5[t].data(), table.frame,
                (static_cast<std::size_t>(table.symbols) + 1u) *
                    sizeof(std::uint16_t));
  }

  // A second reference context with different values.
  FrameCdfInput reset2;
  reset2.tile_cols = 1;
  reset2.tile_rows = 1;
  reset2.keyframe = true;
  CHECK_STATUS(ctl.begin_frame(reset2, &error), ContinuityStatus::kOk);
  CHECK_STATUS(ctl.begin_tile(&error), ContinuityStatus::kOk);
  fill_tile(&ctl, 3);
  CHECK_STATUS(ctl.end_tile(0, &error), ContinuityStatus::kOk);
  CHECK_STATUS(ctl.end_frame(&error), ContinuityStatus::kOk);
  CHECK_STATUS(ctl.save_reference(9, &error), ContinuityStatus::kOk);
  std::array<std::array<std::uint16_t, 9>, 2> ref9{};
  for (std::size_t t = 0; t < ctl.bank().table_count(); ++t) {
    const acdf::CdfTable& table = ctl.bank().table(t);
    std::memcpy(ref9[t].data(), table.frame,
                (static_cast<std::size_t>(table.symbols) + 1u) *
                    sizeof(std::uint16_t));
  }

  // Carry-over: frame arrays become the slot-5 contents exactly.
  FrameCdfInput carry;
  carry.tile_cols = 1;
  carry.tile_rows = 1;
  carry.primary_ref_slot = 5;
  CHECK_STATUS(ctl.begin_frame(carry, &error), ContinuityStatus::kOk);
  for (std::size_t t = 0; t < ctl.bank().table_count(); ++t) {
    const acdf::CdfTable& table = ctl.bank().table(t);
    const std::size_t entries = static_cast<std::size_t>(table.symbols) + 1u;
    CHECK(std::memcmp(table.frame, ref5[t].data(),
                      entries * sizeof(std::uint16_t)) == 0);
    CHECK(std::memcmp(table.tile, ref5[t].data(),
                      entries * sizeof(std::uint16_t)) == 0);
  }
  CHECK_STATUS(ctl.begin_tile(&error), ContinuityStatus::kOk);
  CHECK_STATUS(ctl.end_tile(0, &error), ContinuityStatus::kOk);
  CHECK_STATUS(ctl.end_frame(&error), ContinuityStatus::kOk);

  // Blend: load slot 5, then 7/8 current + 1/8 slot 9 per §7.4.4.
  FrameCdfInput blend;
  blend.tile_cols = 1;
  blend.tile_rows = 1;
  blend.primary_ref_slot = 5;
  blend.enable_avg_cdf = true;
  blend.avg_cdf_type = false;
  blend.blend_ref_present = true;
  blend.blend_ref_slot = 9;
  CHECK_STATUS(ctl.begin_frame(blend, &error), ContinuityStatus::kOk);
  for (std::size_t t = 0; t < ctl.bank().table_count(); ++t) {
    acdf::CdfTable& table = ctl.bank().table(t);
    const std::size_t entries = static_cast<std::size_t>(table.symbols) + 1u;
    std::array<std::uint16_t, 9> expect{};
    std::memcpy(expect.data(), ref5[t].data(),
                entries * sizeof(std::uint16_t));
    spec_blend_cdf(expect.data(), ref9[t].data(), table.symbols);
    CHECK(std::memcmp(table.frame, expect.data(),
                      entries * sizeof(std::uint16_t)) == 0);
  }
  CHECK_STATUS(ctl.begin_tile(&error), ContinuityStatus::kOk);
  CHECK_STATUS(ctl.end_tile(0, &error), ContinuityStatus::kOk);
  CHECK_STATUS(ctl.end_frame(&error), ContinuityStatus::kOk);
}

// ---------------------------------------------------------------------------
// Controller: disable_cdf_update path
// ---------------------------------------------------------------------------

void test_controller_frozen_is_reset_equivalent() {
  // With disable_cdf_update = 1 the Tile* arrays never change, so a frozen
  // frame's copyCdf result equals the defaults.  This is exactly why the
  // production writers cannot expose cross-frame CDF state: their only
  // adaptive path is frozen, and the arithmetic below is the whole difference.
  FrameCdfContinuity frozen;
  init_controller(&frozen, cont::kNumRefFrames);
  FrameCdfInput in;
  in.tile_cols = 2;
  in.tile_rows = 2;
  in.context_update_tile_id = 1;
  in.disable_cdf_update = true;
  in.disable_cross_frame_cdf_init = true;
  std::string error;
  CHECK_STATUS(frozen.begin_frame(in, &error), ContinuityStatus::kOk);
  CHECK(frozen.plan().adaptation_enabled() == false);
  for (std::uint32_t tile = 0; tile < 4; ++tile) {
    CHECK_STATUS(frozen.begin_tile(&error), ContinuityStatus::kOk);
    CHECK_STATUS(frozen.end_tile(tile, &error), ContinuityStatus::kOk);
  }
  CHECK_STATUS(frozen.end_frame(&error), ContinuityStatus::kOk);
  for (std::size_t t = 0; t < frozen.bank().table_count(); ++t) {
    const acdf::CdfTable& table = frozen.bank().table(t);
    const std::size_t entries = static_cast<std::size_t>(table.symbols) + 1u;
    CHECK(std::memcmp(table.frame, table.defaults,
                      entries * sizeof(std::uint16_t)) == 0);
  }

  // The header bit is carried in the plan for the writer to emit.
  FrameCdfInput adaptive = in;
  adaptive.disable_cdf_update = false;
  CHECK(cont::plan_frame_cdfs(adaptive).adaptation_enabled());
}

// ---------------------------------------------------------------------------
// Controller: call-order enforcement
// ---------------------------------------------------------------------------

void test_controller_order() {
  FrameCdfContinuity ctl;
  init_controller(&ctl, cont::kNumRefFrames);
  std::string error;
  FrameCdfInput in;
  in.tile_cols = 2;
  in.tile_rows = 2;

  CHECK_STATUS(ctl.begin_tile(&error), ContinuityStatus::kFrameNotOpen);
  CHECK_STATUS(ctl.end_tile(0, &error), ContinuityStatus::kFrameNotOpen);
  CHECK_STATUS(ctl.end_frame(&error), ContinuityStatus::kFrameNotOpen);
  CHECK_STATUS(ctl.begin_frame(in, &error), ContinuityStatus::kOk);
  CHECK_STATUS(ctl.begin_frame(in, &error),
               ContinuityStatus::kFrameAlreadyOpen);
  CHECK_STATUS(ctl.end_tile(0, &error), ContinuityStatus::kTileNotOpen);
  CHECK_STATUS(ctl.end_frame(&error), ContinuityStatus::kFrameIncomplete);
  CHECK_STATUS(ctl.begin_tile(&error), ContinuityStatus::kOk);
  CHECK_STATUS(ctl.begin_tile(&error), ContinuityStatus::kTileAlreadyOpen);
  CHECK_STATUS(ctl.end_tile(1, &error), ContinuityStatus::kTileOutOfRange);
  CHECK_STATUS(ctl.end_tile(0, &error), ContinuityStatus::kOk);
  CHECK_STATUS(ctl.end_frame(&error), ContinuityStatus::kFrameIncomplete);
  CHECK_STATUS(ctl.save_reference(0, &error),
               ContinuityStatus::kFrameNotOpen);
  for (std::uint32_t tile = 1; tile < 4; ++tile) {
    CHECK_STATUS(ctl.begin_tile(&error), ContinuityStatus::kOk);
    CHECK_STATUS(ctl.end_tile(tile, &error), ContinuityStatus::kOk);
  }
  CHECK_STATUS(ctl.end_frame(&error), ContinuityStatus::kOk);
  CHECK_STATUS(ctl.save_reference(cont::kNumRefFrames, &error),
               ContinuityStatus::kContextOutOfRange);
  CHECK_STATUS(ctl.save_reference(0, &error), ContinuityStatus::kOk);
}

// ---------------------------------------------------------------------------
// Controller: 0 reference contexts rejects the carry branch
// ---------------------------------------------------------------------------

void test_controller_no_reference_contexts() {
  FrameCdfContinuity ctl;
  init_controller(&ctl, 0);
  std::string error;
  FrameCdfInput in;
  in.primary_ref_slot = 0;
  CHECK_STATUS(ctl.begin_frame(in, &error),
               ContinuityStatus::kContextOutOfRange);
  CHECK(ctl.bank().table_count() == 2);
}

}  // namespace

int main() {
  test_tile_info_field();
  test_context_update_validation();
  test_tile_exit_plan();
  test_qctx_slice();
  test_frame_plan();
  test_controller_copy_cdf();
  test_controller_avg_cdf();
  test_controller_reference_blend();
  test_controller_frozen_is_reset_equivalent();
  test_controller_order();
  test_controller_no_reference_contexts();

  if (g_failures != 0) {
    std::fprintf(stderr, "FAIL: %d/%d checks failed\n", g_failures, g_checks);
    return 1;
  }
  std::printf("PASS: %d checks\n", g_checks);
  return 0;
}

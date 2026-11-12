// Unit tests for the temporal-layer planner and syntax.
//
// Covers:
//   * hierarchical P/B schedules: coding order, layer ids, references,
//     slots, refresh flags, output flags, order-hint LSBs;
//   * §5.4.1 writer/reader round-trip for max_tlayer_id,
//     tlayer_dependency_present_flag, tlayer_dependency_map and
//     monotonic_output_order_flag;
//   * §5.2.2 obu_tlayer_id placement;
//   * byte-for-byte equivalence with the M2 sequence/frame header writers for
//     the max_tlayer_id = 0 baseline;
//   * pinned golden bytes for 64x64 and 640x360 headers;
//   * validation/mutation failures and the layer-drop invariant.

#include "codec/inter/tlayers/tlayers.h"
#include "codec/inter/tlayers/tlayers_syntax.h"

#include "pipeline/video/video_headers.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

int g_checks = 0;
int g_failures = 0;

#define CHECK(cond)                                                        \
  do {                                                                     \
    ++g_checks;                                                            \
    if (!(cond)) {                                                         \
      ++g_failures;                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    }                                                                      \
  } while (0)

#define CHECK_EQ_INT(a, b)                                                   \
  do {                                                                       \
    ++g_checks;                                                              \
    const long long va = static_cast<long long>(a);                          \
    const long long vb = static_cast<long long>(b);                          \
    if (va != vb) {                                                          \
      ++g_failures;                                                          \
      std::fprintf(stderr, "FAIL %s:%d: %s == %s (%lld != %lld)\n",          \
                   __FILE__, __LINE__, #a, #b, va, vb);                      \
    }                                                                        \
  } while (0)

using namespace av2enc::inter::tlayers;

std::string hex(std::span<const std::uint8_t> bytes) {
  static const char* digits = "0123456789abcdef";
  std::string out;
  out.reserve(bytes.size() * 2);
  for (const std::uint8_t byte : bytes) {
    out.push_back(digits[(byte >> 4U) & 0x0FU]);
    out.push_back(digits[byte & 0x0FU]);
  }
  return out;
}

bool build(PlanConfig config, Plan& plan) {
  PlanError error = PlanError::kOk;
  const bool ok = build_plan(config, plan, &error);
  if (!ok) {
    std::fprintf(stderr, "build_plan failed: %s\n", plan_error_name(error));
  }
  return ok;
}

PlanConfig p8_config() {
  PlanConfig c;
  c.gop_size = 8;
  c.schedule = Schedule::kHierarchicalP;
  c.dependency = DependencyMode::kAllLower;
  c.explicit_dependency_map = false;
  c.order_hint_bits = 4;
  c.num_slots = 8;
  return c;
}

PlanConfig b8_config() {
  PlanConfig c;
  c.gop_size = 8;
  c.schedule = Schedule::kHierarchicalB;
  c.dependency = DependencyMode::kAllLower;
  c.explicit_dependency_map = false;
  c.order_hint_bits = 4;
  c.num_slots = 8;
  return c;
}

// ---------------------------------------------------------------------------

void test_obu_headers() {
  // §5.2.2: (extension_flag << 7) | (obu_type << 2) | obu_tlayer_id.
  CHECK_EQ_INT(obu_header_byte(false, 7, 0), 0x1C);
  CHECK_EQ_INT(obu_header_byte(false, 7, 1), 0x1D);
  CHECK_EQ_INT(obu_header_byte(false, 7, 2), 0x1E);
  CHECK_EQ_INT(obu_header_byte(false, 7, 3), 0x1F);
  CHECK_EQ_INT(obu_header_byte(false, 1, 0), 0x04);
  CHECK_EQ_INT(obu_header_byte(false, 4, 0), 0x10);
  CHECK_EQ_INT(obu_header_byte(false, 2, 0), 0x08);
  CHECK_EQ_INT(obu_header_byte(true, 7, 2), 0x9E);

  // TD OBU is 01 08 (Annex B leb128 length 1, header byte 0x08).
  const std::vector<std::uint8_t> td = make_temporal_delimiter_obu();
  CHECK_EQ_INT(td.size(), 2U);
  CHECK_EQ_INT(td[0], 0x01);
  CHECK_EQ_INT(td[1], 0x08);
  ParsedObuHeader parsed;
  std::string error;
  CHECK(parse_obu(td, 0, parsed, error));
  CHECK_EQ_INT(parsed.obu_type, 2);
  CHECK_EQ_INT(parsed.obu_tlayer_id, 0);
  CHECK_EQ_INT(parsed.header_bytes, 1U);
  CHECK_EQ_INT(parsed.total_bytes, 2U);
}

void test_p8_plan() {
  Plan plan;
  CHECK(build(p8_config(), plan));
  CHECK_EQ_INT(plan.display_count(), 8);
  CHECK_EQ_INT(plan.max_tlayer_id, 2);
  CHECK_EQ_INT(plan.num_layers, 3);
  CHECK_EQ_INT(plan.layer_frame_count[0], 2);
  CHECK_EQ_INT(plan.layer_frame_count[1], 2);
  CHECK_EQ_INT(plan.layer_frame_count[2], 4);
  CHECK_EQ_INT(plan.layer_frame_count[3], 0);
  CHECK_EQ_INT(plan.num_slots, 8);
  CHECK(!plan.tlayer_dependency_present_flag);

  const int expected_coding[8] = {0, 4, 2, 6, 1, 3, 5, 7};
  for (int i = 0; i < 8; ++i) {
    CHECK_EQ_INT(plan.coding_order[static_cast<std::size_t>(i)],
                 expected_coding[i]);
  }
  const int expected_layer[8] = {0, 2, 1, 2, 0, 2, 1, 2};
  for (int d = 0; d < 8; ++d) {
    CHECK_EQ_INT(plan.layer_of(d), expected_layer[d]);
  }

  // Display order hint and signalled LSBs.
  for (int d = 0; d < 8; ++d) {
    const PlannedFrame& f = plan.frames[static_cast<std::size_t>(d)];
    CHECK_EQ_INT(f.display_order_hint, static_cast<std::uint32_t>(d));
    CHECK_EQ_INT(f.order_hint_lsbs, static_cast<std::uint32_t>(d));
    CHECK_EQ_INT(f.immediate_output_frame, d == 0);
    CHECK_EQ_INT(f.implicit_output_frame, d != 0);
    CHECK_EQ_INT(f.obu_tlayer_id(), expected_layer[d]);
    CHECK_EQ_INT(f.slot, f.coding_index);
    if (d == 0) {
      CHECK_EQ_INT(f.refresh_frame_flags, 0xFFU);
      CHECK_EQ_INT(f.num_refs, 0);
    } else {
      CHECK_EQ_INT(f.refresh_frame_flags, 1U << f.slot);
      CHECK_EQ_INT(f.num_refs, 1);
      CHECK_EQ_INT(f.ref_display[0],
                   plan.coding_order[static_cast<std::size_t>(f.coding_index - 1)]);
      CHECK_EQ_INT(f.ref_slot[0],
                   plan.frames[static_cast<std::size_t>(f.ref_display[0])].slot);
      CHECK_EQ_INT(f.ref_order_hint[0], f.ref_display[0]);
      CHECK(f.tlayer_id >=
            plan.layer_of(f.ref_display[0]));
    }
  }

  // TLayerDependencyMap default (all lower layers).
  for (int c = 0; c <= plan.max_tlayer_id; ++c) {
    for (int r = 0; r < kMaxTLayers; ++r) {
      CHECK_EQ_INT(plan.tlayer_dependency[static_cast<std::size_t>(c)]
                                         [static_cast<std::size_t>(r)],
                   r <= c);
    }
  }

  // Drop order.
  const std::vector<int> l0 = plan.coding_order_up_to_layer(0);
  const std::vector<int> l1 = plan.coding_order_up_to_layer(1);
  CHECK_EQ_INT(l0.size(), 2U);
  CHECK_EQ_INT(l0[0], 0);
  CHECK_EQ_INT(l0[1], 4);
  CHECK_EQ_INT(l1.size(), 4U);
  CHECK_EQ_INT(l1[2], 2);
  CHECK_EQ_INT(l1[3], 6);
  PlanError drop_error = PlanError::kOk;
  for (int layer = 0; layer <= plan.max_tlayer_id; ++layer) {
    CHECK(validate_layer_drop(plan, layer, &drop_error));
    CHECK(drop_error == PlanError::kOk);
  }

  // Determinism.
  Plan again;
  CHECK(build(p8_config(), again));
  CHECK_EQ_INT(again.coding_order.size(), plan.coding_order.size());
  for (std::size_t i = 0; i < plan.coding_order.size(); ++i) {
    CHECK_EQ_INT(again.coding_order[i], plan.coding_order[i]);
    CHECK_EQ_INT(again.frames[i].slot, plan.frames[i].slot);
    CHECK_EQ_INT(again.frames[i].order_hint_lsbs, plan.frames[i].order_hint_lsbs);
  }
}

void test_b8_plan() {
  Plan plan;
  CHECK(build(b8_config(), plan));
  CHECK_EQ_INT(plan.max_tlayer_id, 3);
  CHECK_EQ_INT(plan.layer_frame_count[0], 2);
  CHECK_EQ_INT(plan.layer_frame_count[1], 1);
  CHECK_EQ_INT(plan.layer_frame_count[2], 2);
  CHECK_EQ_INT(plan.layer_frame_count[3], 3);
  const int expected_coding[8] = {0, 7, 3, 1, 5, 2, 4, 6};
  for (int i = 0; i < 8; ++i) {
    CHECK_EQ_INT(plan.coding_order[static_cast<std::size_t>(i)],
                 expected_coding[i]);
  }
  const int expected_layer[8] = {0, 2, 3, 1, 3, 2, 3, 0};
  for (int d = 0; d < 8; ++d) {
    CHECK_EQ_INT(plan.layer_of(d), expected_layer[d]);
  }
  // Exact reference maps.
  const int expected_refs0[8] = {-1, 0, 1, 0, 3, 3, 5, 0};
  const int expected_refs1[8] = {-1, 3, 3, 7, 5, 7, 7, -1};
  for (int d = 0; d < 8; ++d) {
    const PlannedFrame& f = plan.frames[static_cast<std::size_t>(d)];
    CHECK_EQ_INT(f.num_refs, d == 0 || d == 7 ? (d == 0 ? 0 : 1) : 2);
    CHECK_EQ_INT(f.ref_display[0], expected_refs0[d]);
    CHECK_EQ_INT(f.ref_display[1], expected_refs1[d]);
    for (int r = 0; r < f.num_refs; ++r) {
      CHECK(plan.layer_of(f.ref_display[r]) <= f.tlayer_id);
    }
  }
  for (int d = 0; d < 8; ++d) {
    CHECK_EQ_INT(plan.frames[static_cast<std::size_t>(d)].order_hint_lsbs,
                 static_cast<std::uint32_t>(d));
  }
  PlanError error = PlanError::kOk;
  CHECK(validate_plan(plan, &error, nullptr, nullptr));
}

void test_all_sizes() {
  for (const Schedule schedule :
       {Schedule::kHierarchicalP, Schedule::kHierarchicalB}) {
    for (int n = 2; n <= 16; n *= 2) {
      PlanConfig config;
      config.gop_size = n;
      config.schedule = schedule;
      config.num_slots = n;
      config.order_hint_bits = (n == 16) ? 5 : 4;
      Plan plan;
      PlanError error = PlanError::kOk;
      const bool ok = build_plan(config, plan, &error);
      if (schedule == Schedule::kHierarchicalB && n == 16) {
        CHECK(!ok);
        CHECK(error == PlanError::kTooManyLayers);
        continue;
      }
      CHECK(ok);
      if (!ok) continue;
      CHECK(error == PlanError::kOk);
      CHECK_EQ_INT(plan.display_count(), n);
      int log2_n = 0;
      for (int v = n; v > 1; v >>= 1) ++log2_n;
      const int expected_max =
          (n == 2) ? 0
                   : (schedule == Schedule::kHierarchicalP ? log2_n - 1
                                                           : log2_n);
      CHECK_EQ_INT(plan.max_tlayer_id, expected_max);
      CHECK(validate_plan(plan, &error, nullptr, nullptr));
      // Every frame's tlayer never regresses along coding order within a layer
      // set and every reference is decoded earlier.
      for (const int d : plan.coding_order) {
        const PlannedFrame& f = plan.frames[static_cast<std::size_t>(d)];
        for (int r = 0; r < f.num_refs; ++r) {
          CHECK(plan.frames[static_cast<std::size_t>(f.ref_display[r])].coding_index <
                f.coding_index);
        }
      }
    }
  }
}

void test_config_errors() {
  Plan plan;
  PlanError error = PlanError::kOk;

  PlanConfig config = p8_config();
  config.gop_size = 3;
  CHECK(!build_plan(config, plan, &error));
  CHECK(error == PlanError::kBadGopSize);

  config = p8_config();
  config.gop_size = 32;
  CHECK(!build_plan(config, plan, &error));
  CHECK(error == PlanError::kBadGopSize);

  config = p8_config();
  config.order_hint_bits = 0;
  CHECK(!build_plan(config, plan, &error));
  CHECK(error == PlanError::kBadOrderHintBits);

  config = p8_config();
  config.num_slots = 0;
  CHECK(!build_plan(config, plan, &error));
  CHECK(error == PlanError::kBadSlotCount);

  config = p8_config();
  config.num_slots = 17;
  CHECK(!build_plan(config, plan, &error));
  CHECK(error == PlanError::kBadSlotCount);

  config = p8_config();
  config.start_order_hint = 40;
  CHECK(!build_plan(config, plan, &error));
  CHECK(error == PlanError::kBadOrderHintBits);

  config = p8_config();
  config.start_order_hint = 40;
  config.order_hint_bits = 6;
  CHECK(build_plan(config, plan, &error));
  CHECK_EQ_INT(plan.frames[0].order_hint_lsbs, 40U);

  // 16-frame P pyramid needs 5 order-hint bits; 4 fails the §5.18 lift.
  config = p8_config();
  config.gop_size = 16;
  config.num_slots = 16;
  config.order_hint_bits = 4;
  CHECK(!build_plan(config, plan, &error));
  CHECK(error == PlanError::kOrderHintReconstruct);
  config.order_hint_bits = 5;
  CHECK(build_plan(config, plan, &error));

  // 16-frame B pyramid needs temporal layer 4, which max_tlayer_id f(2)
  // cannot represent.
  config = b8_config();
  config.gop_size = 16;
  config.num_slots = 16;
  config.order_hint_bits = 5;
  CHECK(!build_plan(config, plan, &error));
  CHECK(error == PlanError::kTooManyLayers);

  // Hierarchical B needs at least four slots at gop_size 8.
  config = b8_config();
  config.num_slots = 3;
  CHECK(!build_plan(config, plan, &error));
  CHECK(error == PlanError::kNotEnoughSlots);
  config.num_slots = 4;
  CHECK(build_plan(config, plan, &error));

  // An explicit-map request on a single-layer plan is a no-op (no
  // tlayer_dependency_present_flag is coded when max_tlayer_id == 0).
  config = p8_config();
  config.gop_size = 2;
  config.num_slots = 2;
  config.explicit_dependency_map = true;
  CHECK(build_plan(config, plan, &error));
  CHECK_EQ_INT(plan.max_tlayer_id, 0);
  CHECK(!plan.tlayer_dependency_present_flag);
}

void test_mutations() {
  Plan base;
  CHECK(build(p8_config(), base));

  // Bad LSBS.
  {
    Plan plan = base;
    plan.frames[2].order_hint_lsbs ^= 1U;
    PlanError error = PlanError::kOk;
    CHECK(!validate_plan(plan, &error, nullptr, nullptr));
    CHECK(error == PlanError::kOrderHintReconstruct);
  }
  // Wrong refresh flags.
  {
    Plan plan = base;
    plan.frames[2].refresh_frame_flags = 1U << 1;
    PlanError error = PlanError::kOk;
    CHECK(!validate_plan(plan, &error, nullptr, nullptr));
    CHECK(error == PlanError::kRefreshFlagsInvalid);
  }
  // Both output flags.
  {
    Plan plan = base;
    plan.frames[2].implicit_output_frame = true;
    plan.frames[2].immediate_output_frame = true;
    PlanError error = PlanError::kOk;
    CHECK(!validate_plan(plan, &error, nullptr, nullptr));
    CHECK(error == PlanError::kOutputFlagsInvalid);
  }
  // Key frame with a reference.
  {
    Plan plan = base;
    plan.frames[0].num_refs = 1;
    plan.frames[0].ref_display[0] = 4;
    PlanError error = PlanError::kOk;
    CHECK(!validate_plan(plan, &error, nullptr, nullptr));
    CHECK(error == PlanError::kKeyFrameHasRefs);
  }
  // Reference to a later-coded frame.
  {
    Plan plan = base;
    plan.frames[2].ref_display[0] = 1;  // display 1 is coded 5th
    PlanError error = PlanError::kOk;
    CHECK(!validate_plan(plan, &error, nullptr, nullptr));
    CHECK(error == PlanError::kReferenceNotDecoded);
  }
  // Reference slot that holds another frame.
  {
    Plan plan = base;
    plan.frames[2].ref_slot[0] = 7;
    PlanError error = PlanError::kOk;
    CHECK(!validate_plan(plan, &error, nullptr, nullptr));
    CHECK(error == PlanError::kReferenceSlotInvalid);
  }
  // Slot index outside NumRefFrames.
  {
    Plan plan = base;
    plan.frames[2].slot = 20;
    plan.frames[2].refresh_frame_flags = 1U << 20;
    PlanError error = PlanError::kOk;
    CHECK(!validate_plan(plan, &error, nullptr, nullptr));
    CHECK(error == PlanError::kRefreshFlagsInvalid);
  }
  // A dependency-map hole (explicit-map plan).
  {
    PlanConfig config = p8_config();
    config.explicit_dependency_map = true;
    config.dependency = DependencyMode::kDirect;
    Plan plan;
    CHECK(build(config, plan));
    plan.tlayer_dependency[2][1] = false;
    PlanError error = PlanError::kOk;
    CHECK(!validate_plan(plan, &error, nullptr, nullptr));
    CHECK(error == PlanError::kDependencyMapIncomplete);
  }
  // Reference to a higher temporal layer.
  {
    Plan plan = base;
    // display 1 has layer 2 and references display 6 (layer 1) at coding 4.
    plan.frames[1].tlayer_id = 0;
    plan.layer_frame_count[0] += 1;
    plan.layer_frame_count[2] -= 1;
    PlanError error = PlanError::kOk;
    CHECK(!validate_plan(plan, &error, nullptr, nullptr));
    CHECK(error == PlanError::kReferenceLayerViolation);
  }
  // Duplicate references (B plan).
  {
    Plan plan;
    CHECK(build(b8_config(), plan));
    plan.frames[1].ref_display[1] = plan.frames[1].ref_display[0];
    PlanError error = PlanError::kOk;
    CHECK(!validate_plan(plan, &error, nullptr, nullptr));
    CHECK(error == PlanError::kDuplicateReference);
  }
  // Coding order is not a permutation.
  {
    Plan plan = base;
    plan.coding_order[1] = plan.coding_order[0];
    PlanError error = PlanError::kOk;
    CHECK(!validate_plan(plan, &error, nullptr, nullptr));
    CHECK(error == PlanError::kBadCodingOrder);
  }
  // Layer-frame-count mismatch is caught by the internal consistency check.
  {
    Plan plan = base;
    plan.layer_frame_count[2] -= 1;
    PlanError error = PlanError::kOk;
    CHECK(!validate_plan(plan, &error, nullptr, nullptr));
    CHECK(error == PlanError::kInternal);
  }
}

void test_sequence_header_roundtrip() {
  for (int width : {64, 640}) {
    for (int height : {64, 360}) {
      for (int max_tlayer = 0; max_tlayer <= kMaxTemporalLayerId; ++max_tlayer) {
        for (const bool explicit_map : {false, true}) {
          if (max_tlayer == 0 && explicit_map) continue;
          for (const bool monotonic : {false, true}) {
            SequenceConfig config;
            config.max_frame_width = static_cast<std::uint16_t>(width);
            config.max_frame_height = static_cast<std::uint16_t>(height);
            config.max_tlayer_id = max_tlayer;
            config.monotonic_output_order_flag = monotonic;
            config.tlayer_dependency_present_flag =
                explicit_map && max_tlayer > 0;
            for (auto& row : config.tlayer_dependency) {
              row.fill(false);
            }
            config.tlayer_dependency[0][0] = true;
            for (int c = 1; c <= max_tlayer; ++c) {
              for (int r = 0; r <= c; ++r) {
                config.tlayer_dependency[static_cast<std::size_t>(c)]
                                        [static_cast<std::size_t>(r)] =
                    (r != 1 || c != 2);  // a non-trivial map
              }
            }
            const std::vector<std::uint8_t> obu =
                make_sequence_header_obu(config);
            CHECK(obu.size() < 128U);
            CHECK_EQ_INT(obu[0], obu.size() - 1U);
            ParsedSequenceHeader parsed;
            std::string error;
            CHECK(parse_sequence_header_obu(obu, parsed, error));
            CHECK(error.empty());
            CHECK_EQ_INT(parsed.max_tlayer_id, max_tlayer);
            CHECK_EQ_INT(parsed.max_mlayer_id, 0);
            CHECK_EQ_INT(parsed.monotonic_output_order_flag, monotonic);
            CHECK_EQ_INT(parsed.tlayer_dependency_present_flag,
                         explicit_map && max_tlayer > 0);
            CHECK_EQ_INT(parsed.multi_tlayer_dependency_map_present_flag, 0);
            CHECK_EQ_INT(parsed.max_frame_width, width);
            CHECK_EQ_INT(parsed.max_frame_height, height);
            CHECK_EQ_INT(parsed.num_ref_frames, config.num_ref_frames);
            CHECK_EQ_INT(parsed.order_hint_bits, config.order_hint_bits);
            for (int c = 0; c <= max_tlayer; ++c) {
              for (int r = 0; r <= max_tlayer; ++r) {
                const bool expected = explicit_map
                                          ? config.tlayer_dependency[static_cast<std::size_t>(c)]
                                                                     [static_cast<std::size_t>(r)]
                                          : (r <= c);
                CHECK_EQ_INT(
                    parsed.tlayer_dependency[static_cast<std::size_t>(c)]
                                            [static_cast<std::size_t>(r)],
                    expected);
              }
            }
          }
        }
      }
    }
  }
}

void test_sequence_level_gating() {
  // §5.4.1: seq_tier is present only when seq_level_idx > 3.
  for (const int level : {3, 5}) {
    SequenceConfig config = baseline_sequence_config(64, 64, 90, 2);
    config.seq_level_idx = static_cast<std::uint8_t>(level);
    const std::vector<std::uint8_t> obu =
        make_sequence_header_obu(config);
    CHECK(!obu.empty());
    ParsedSequenceHeader parsed;
    std::string error;
    CHECK(parse_sequence_header_obu(obu, parsed, error));
    CHECK(error.empty());
    CHECK_EQ_INT(parsed.seq_level_idx, level);
    CHECK_EQ_INT(parsed.max_frame_width, 64);
    CHECK_EQ_INT(parsed.max_frame_height, 64);
  }
}

void test_sequence_header_goldens() {
  // max_tlayer_id = 0 baseline (byte-identical to M2).
  {
    SequenceConfig config = baseline_sequence_config(64, 64, 90, 2);
    const auto obu = make_sequence_header_obu(config);
    CHECK(hex(obu) ==
          "13048075002abffe0028006e23800000101b8082");
  }
  // P8, default dependency map (flag = 0), 64x64, NumRefFrames = 8.
  {
    Plan plan;
    CHECK(build(p8_config(), plan));
    SequenceConfig config = baseline_sequence_config(64, 64, 90, 8);
    config.apply_plan(plan);
    const auto obu = make_sequence_header_obu(config);
    CHECK(hex(obu) == "13048075040abffe0014003771c00000080dc041");
  }
  // P8, explicit all-lower map, 64x64.
  {
    PlanConfig pc = p8_config();
    pc.explicit_dependency_map = true;
    Plan plan;
    CHECK(build(pc, plan));
    SequenceConfig config = baseline_sequence_config(64, 64, 90, 8);
    config.apply_plan(plan);
    const auto obu = make_sequence_header_obu(config);
    CHECK(hex(obu) == "14048075040abffe3f00a001bb8e000000406e0208");
  }
  // P8, explicit direct map, 640x360.
  {
    PlanConfig pc = p8_config();
    pc.explicit_dependency_map = true;
    pc.dependency = DependencyMode::kDirect;
    Plan plan;
    CHECK(build(pc, plan));
    SequenceConfig config = baseline_sequence_config(640, 360, 90, 8);
    config.apply_plan(plan);
    const auto obu = make_sequence_header_obu(config);
    CHECK(hex(obu) == "15048075041313fd9c7c014003771c00000080dc0410");
  }
  // B8, explicit direct map (max_tlayer_id = 3).
  {
    PlanConfig pc = b8_config();
    pc.explicit_dependency_map = true;
    pc.dependency = DependencyMode::kDirect;
    Plan plan;
    CHECK(build(pc, plan));
    SequenceConfig config = baseline_sequence_config(64, 64, 90, 8);
    config.apply_plan(plan);
    const auto obu = make_sequence_header_obu(config);
    CHECK(hex(obu) == "15048075060abffe2b700a001bb8e000000406e02080");
  }
  // P16, default map, OrderHintBits = 5, NumRefFrames = 16.
  {
    PlanConfig pc;
    pc.gop_size = 16;
    pc.schedule = Schedule::kHierarchicalP;
    pc.num_slots = 16;
    pc.order_hint_bits = 5;
    Plan plan;
    CHECK(build(pc, plan));
    SequenceConfig config = baseline_sequence_config(64, 64, 90, 16);
    config.apply_plan(plan);
    const auto obu = make_sequence_header_obu(config);
    CHECK(hex(obu) == "13048075060abffe00140047f1c00000080dc041");
  }
}

void test_video_equivalence() {
  // The M2 baseline sequence header must be byte-identical.
  av2enc::video::VideoConfig video_config;
  video_config.max_frame_width = 64;
  video_config.max_frame_height = 64;
  video_config.base_q_idx = 90;
  video_config.order_hint_bits = 4;
  video_config.num_ref_frames = 2;
  video_config.num_total_refs = 1;
  video_config.max_drl_bits_minus_1 = 4;
  video_config.disable_cdf_update = true;
  video_config.enable_cfl_intra = true;

  SequenceConfig config = baseline_sequence_config(64, 64, 90, 2);
  CHECK(make_sequence_header_obu(config) ==
        av2enc::video::make_sequence_header_obu(video_config));

  const std::vector<std::uint8_t> tile = {1, 2, 3, 4, 5};
  for (const bool key : {false, true}) {
    FrameHeader header;
    header.keyframe = key;
    header.order_hint_lsbs = key ? 0 : 3;
    header.refresh_frame_flags = key ? 3 : 1;
    header.num_total_refs = key ? 0 : 1;
    header.ref_frame_idx[0] = 0;
    header.immediate_output_frame = true;
    header.implicit_output_frame = false;

    av2enc::video::VideoFrameHeader video_header;
    video_header.keyframe = key;
    video_header.order_hint = key ? 0 : 3;
    video_header.ref_slot = 0;
    video_header.refresh_frame_flags = key ? 3 : 1;

    CHECK(make_frame_obu(config, header, 0, tile) ==
          av2enc::video::make_frame_obu(video_config, video_header, tile));
  }
}

void test_frame_header_plan() {
  Plan plan;
  CHECK(build(p8_config(), plan));
  SequenceConfig config = baseline_sequence_config(64, 64, 90, 8);
  config.apply_plan(plan);
  CHECK_EQ_INT(config.max_tlayer_id, 2);
  CHECK(!config.monotonic_output_order_flag);
  CHECK_EQ_INT(config.num_ref_frames, 8);

  for (int d = 0; d < plan.display_count(); ++d) {
    const FrameHeader header = frame_header_for(config, plan, d);
    const PlannedFrame& f = plan.frames[static_cast<std::size_t>(d)];
    CHECK_EQ_INT(header.keyframe, f.is_key);
    CHECK_EQ_INT(header.order_hint_lsbs, f.order_hint_lsbs);
    CHECK_EQ_INT(header.refresh_frame_flags, f.refresh_frame_flags);
    CHECK_EQ_INT(header.num_total_refs, f.num_refs);
    for (int r = 0; r < f.num_refs; ++r) {
      CHECK_EQ_INT(header.ref_frame_idx[r], f.ref_slot[r]);
    }
    CHECK_EQ_INT(header.immediate_output_frame, f.immediate_output_frame);
    CHECK_EQ_INT(header.implicit_output_frame, f.implicit_output_frame);
    CHECK_EQ_INT(header.reference_select, f.num_refs >= 2);
  }

  // Frame OBU framing: leb128 length, header byte, tile-group payload.
  const std::vector<std::uint8_t> tile = {0xAA, 0xBB, 0xCC};
  for (int d = 0; d < plan.display_count(); ++d) {
    const FrameHeader header = frame_header_for(config, plan, d);
    const std::vector<std::uint8_t> obu = make_frame_obu(
        config, header, static_cast<std::uint8_t>(plan.obu_tlayer_id(d)), tile);
    ParsedObuHeader parsed;
    std::string error;
    CHECK(parse_obu(obu, 0, parsed, error));
    CHECK_EQ_INT(parsed.obu_tlayer_id, plan.obu_tlayer_id(d));
    CHECK_EQ_INT(parsed.obu_type, header.keyframe ? 4 : 7);
    if (header.keyframe) {
      CHECK_EQ_INT(parsed.obu_tlayer_id, 0);
    }
    CHECK_EQ_INT(parsed.payload.size(),
                 obu.size() - 2U);  // 1 leb128 + 1 header byte
    CHECK_EQ_INT(parsed.payload[parsed.payload.size() - 3], 0xAA);
    CHECK_EQ_INT(parsed.payload[parsed.payload.size() - 2], 0xBB);
    CHECK_EQ_INT(parsed.payload[parsed.payload.size() - 1], 0xCC);
  }
}

void test_dependency_modes() {
  Plan all_lower;
  CHECK(build(p8_config(), all_lower));
  PlanConfig direct_config = p8_config();
  direct_config.dependency = DependencyMode::kDirect;
  direct_config.explicit_dependency_map = true;
  Plan direct;
  CHECK(build(direct_config, direct));

  CHECK(direct.tlayer_dependency_present_flag);
  // P8 direct: layer 2 references layers 1 and 2 only (frames 1 and 3 ref the
  // previous coding-order frame, both layer 2; display 2 refs layer 0).
  CHECK(direct.tlayer_dependency[0][0]);
  CHECK(direct.tlayer_dependency[1][0]);
  CHECK(direct.tlayer_dependency[1][1]);
  CHECK(!direct.tlayer_dependency[2][0]);
  CHECK(direct.tlayer_dependency[2][1]);
  CHECK(direct.tlayer_dependency[2][2]);
  CHECK(all_lower.tlayer_dependency[2][0]);
  CHECK(!all_lower.tlayer_dependency_present_flag);

  // Both plans remain drop-safe.
  PlanError error = PlanError::kOk;
  for (int layer = 0; layer <= direct.max_tlayer_id; ++layer) {
    CHECK(validate_layer_drop(direct, layer, &error));
  }
  for (int layer = 0; layer <= all_lower.max_tlayer_id; ++layer) {
    CHECK(validate_layer_drop(all_lower, layer, &error));
  }
}

void test_concatenated_units() {
  // A stream is TD + [SEQ] + frame OBU per temporal unit. The parser must walk
  // a concatenation and preserve each OBU's tlayer id.
  Plan plan;
  CHECK(build(p8_config(), plan));
  SequenceConfig config = baseline_sequence_config(64, 64, 90, 8);
  config.apply_plan(plan);
  const std::vector<std::uint8_t> tile = {0x80};
  std::vector<std::uint8_t> stream;
  for (const int d : plan.coding_order) {
    const std::vector<std::uint8_t> td = make_temporal_delimiter_obu();
    stream.insert(stream.end(), td.begin(), td.end());
    if (d == 0) {
      const auto seq = make_sequence_header_obu(config);
      stream.insert(stream.end(), seq.begin(), seq.end());
    }
    const std::vector<std::uint8_t> obu =
        make_frame_obu(config, frame_header_for(config, plan, d),
                       static_cast<std::uint8_t>(plan.obu_tlayer_id(d)), tile);
    stream.insert(stream.end(), obu.begin(), obu.end());
  }
  std::size_t offset = 0;
  int frame_index = 0;
  int frame_obus = 0;
  while (offset < stream.size()) {
    ParsedObuHeader header;
    std::string error;
    CHECK(parse_obu(stream, offset, header, error));
    if (header.obu_type == 4 || header.obu_type == 7) {
      CHECK_EQ_INT(header.obu_tlayer_id,
                   plan.obu_tlayer_id(
                       plan.coding_order[static_cast<std::size_t>(frame_index)]));
      ++frame_obus;
      ++frame_index;
    } else if (header.obu_type == 2) {
      CHECK_EQ_INT(header.obu_tlayer_id, 0);
    } else if (header.obu_type == 1) {
      CHECK_EQ_INT(header.obu_tlayer_id, 0);
    }
    offset += header.total_bytes;
  }
  CHECK_EQ_INT(frame_obus, 8);
  CHECK_EQ_INT(frame_index, 8);
}

void test_single_slot_chain() {
  // One slot is legal for a P chain: the frame reads and refreshes the same
  // slot. This documents that slots alone do not retain implicit-output frames.
  PlanConfig config = p8_config();
  config.num_slots = 1;
  Plan plan;
  CHECK(build(config, plan));
  for (const PlannedFrame& f : plan.frames) {
    CHECK_EQ_INT(f.slot, 0);
  }
}

}  // namespace

int main() {
  test_obu_headers();
  test_p8_plan();
  test_b8_plan();
  test_all_sizes();
  test_config_errors();
  test_mutations();
  test_sequence_header_roundtrip();
  test_sequence_level_gating();
  test_sequence_header_goldens();
  test_video_equivalence();
  test_frame_header_plan();
  test_dependency_modes();
  test_concatenated_units();
  test_single_slot_chain();

  if (g_failures != 0) {
    std::fprintf(stderr, "tlayers_test: %d of %d checks FAILED\n", g_failures,
                 g_checks);
    return 1;
  }
  std::printf("tlayers_test: %d checks passed\n", g_checks);
  return 0;
}

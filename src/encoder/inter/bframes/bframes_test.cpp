// CPU correctness tests for the B-frame hierarchy / reference layer.
//
// The tests cover:
//   * 4/8/16-frame low-delay P, IPBB and dyadic hierarchical GOPs;
//   * exact coding order, temporal layers and explicit references;
//   * order-hint LSB signalling and decoder reconstruction;
//   * slot allocation, key-frame reset and recycling;
//   * leading pictures (display before key, decode after key);
//   * multi-GOP sequences and decoder-visible output order;
//   * malformed plans: out-of-range / forward / duplicate / stale references,
//     bad refresh flags, bad output flags, unreconstructable order hints and
//     output-order violations.
//
// Build with ./build_and_test.sh.

#include "encoder/inter/bframes/bframes.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using namespace av2enc::inter::bframes;

int g_checks = 0;
int g_failures = 0;

void expect(bool condition, const char* what, int line) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    if (g_failures <= 40) {
      std::printf("FAIL line %d: %s\n", line, what);
    }
  }
}

#define EXPECT(cond) expect((cond), #cond, __LINE__)
#define EXPECT_EQ(a, b)                                                                  \
  do {                                                                                   \
    const long long va = static_cast<long long>(a);                                      \
    const long long vb = static_cast<long long>(b);                                      \
    ++g_checks;                                                                          \
    if (va != vb) {                                                                      \
      ++g_failures;                                                                      \
      if (g_failures <= 40) {                                                            \
        std::printf("FAIL line %d: %s (%lld) != %s (%lld)\n", __LINE__, #a, va, #b, vb); \
      }                                                                                  \
    }                                                                                    \
  } while (0)

void expect_error(const Plan& plan, PlanError expected, const char* what, int line) {
  ValidationReport report;
  const bool ok = validate_plan(plan, &report);
  ++g_checks;
  if (ok || report.error != expected) {
    ++g_failures;
    if (g_failures <= 40) {
      std::printf("FAIL line %d: %s: expected %s, got ok=%d error=%s (coding %d display %d)\n",
                  line, what, plan_error_name(expected), ok ? 1 : 0,
                  plan_error_name(report.error), report.error_coding_index,
                  report.error_display_index);
    }
  }
}

#define EXPECT_ERROR(plan, expected) expect_error((plan), (expected), #expected, __LINE__)

bool frame_equal(const FramePlan& a, const FramePlan& b) {
  if (a.display_index != b.display_index || a.display_order_hint != b.display_order_hint ||
      a.coding_index != b.coding_index || a.gop_index != b.gop_index ||
      a.temporal_layer != b.temporal_layer || a.kind != b.kind || a.is_key != b.is_key ||
      a.is_anchor != b.is_anchor || a.is_leading != b.is_leading ||
      a.immediate_output_frame != b.immediate_output_frame ||
      a.implicit_output_frame != b.implicit_output_frame ||
      a.frame_explicit_ref_frame_map != b.frame_explicit_ref_frame_map ||
      a.num_total_refs != b.num_total_refs || a.slot != b.slot ||
      a.refresh_frame_flags != b.refresh_frame_flags ||
      a.order_hint_lsbs != b.order_hint_lsbs) {
    return false;
  }
  for (int r = 0; r < kMaxExplicitRefs; ++r) {
    if (a.ref_display[r] != b.ref_display[r] || a.ref_slot[r] != b.ref_slot[r] ||
        a.ref_order_hint[r] != b.ref_order_hint[r]) {
      return false;
    }
  }
  return true;
}

bool plan_equal(const Plan& a, const Plan& b) {
  if (a.pattern != b.pattern || a.gop_size != b.gop_size ||
      a.start_order_hint != b.start_order_hint || a.order_hint_bits != b.order_hint_bits ||
      a.num_slots != b.num_slots || a.base_q_idx != b.base_q_idx ||
      a.leading_pictures != b.leading_pictures || a.is_sequence != b.is_sequence ||
      a.coding_order != b.coding_order || a.frames.size() != b.frames.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.frames.size(); ++i) {
    if (!frame_equal(a.frames[i], b.frames[i])) {
      return false;
    }
  }
  return true;
}

GopConfig make_gop_config(GopPattern pattern, int gop_size, int num_slots = kMaxRefSlots) {
  GopConfig config;
  config.pattern = pattern;
  config.gop_size = gop_size;
  config.num_slots = num_slots;
  config.order_hint_bits = gop_size == 16 ? 5 : 4;
  return config;
}

int max_temporal_layer(const Plan& plan) {
  int max_layer = -1;
  for (const FramePlan& f : plan.frames) {
    max_layer = std::max(max_layer, f.temporal_layer);
  }
  return max_layer;
}

void check_common_plan(const Plan& plan, const ValidationReport& report) {
  const int count = plan.display_count();
  EXPECT(report.ok);
  EXPECT(report.output_matches_display_order);
  EXPECT_EQ(static_cast<int>(report.output_order.size()), count);
  for (int d = 0; d < count && d < static_cast<int>(report.output_order.size()); ++d) {
    EXPECT_EQ(report.output_order[static_cast<std::size_t>(d)], d);
  }
  EXPECT_EQ(static_cast<int>(report.decode_order.size()), count);
  const std::vector<FrameKind> coding_types = coding_frame_types(plan);
  EXPECT_EQ(static_cast<int>(coding_types.size()), count);
  if (!coding_types.empty()) {
    EXPECT_EQ(coding_types[0], FrameKind::kKey);
  }
  for (int c = 0; c < count && c < static_cast<int>(coding_types.size()); ++c) {
    EXPECT_EQ(coding_types[static_cast<std::size_t>(c)],
              plan.frames[static_cast<std::size_t>(
                              plan.coding_order[static_cast<std::size_t>(c)])]
                  .kind);
  }
  std::vector<char> decoded(static_cast<std::size_t>(count), 0);
  for (int d : report.decode_order) {
    EXPECT(d >= 0 && d < count);
    if (d >= 0 && d < count) {
      EXPECT_EQ(decoded[static_cast<std::size_t>(d)], 0);
      decoded[static_cast<std::size_t>(d)] = 1;
    }
  }
  for (int d = 0; d < count; ++d) {
    EXPECT_EQ(decoded[static_cast<std::size_t>(d)], 1);
  }

  for (int d = 0; d < count; ++d) {
    const FramePlan& f = plan.frames[static_cast<std::size_t>(d)];
    EXPECT_EQ(f.display_index, d);
    EXPECT_EQ(f.display_order_hint, plan.start_order_hint + d);
    EXPECT(f.slot >= 0 && f.slot < plan.num_slots);
    const FrameHeaderFields header = header_fields(f);
    EXPECT_EQ(header.keyframe, f.is_key);
    EXPECT_EQ(header.order_hint_lsbs, static_cast<std::uint32_t>(f.order_hint_lsbs));
    EXPECT_EQ(header.refresh_frame_flags, f.refresh_frame_flags);
    EXPECT_EQ(header.num_total_refs, f.num_total_refs);
    EXPECT_EQ(header.immediate_output_frame, f.immediate_output_frame);
    EXPECT_EQ(header.implicit_output_frame, f.implicit_output_frame);
    EXPECT_EQ(header.reference_select, f.num_total_refs >= 2);
    for (int r = 0; r < f.num_total_refs; ++r) {
      EXPECT_EQ(header.ref_frame_idx[r], f.ref_slot[r]);
      EXPECT_EQ(f.ref_order_hint[r], plan.frames[static_cast<std::size_t>(f.ref_display[r])]
                                          .display_order_hint);
    }
    if (f.is_key) {
      EXPECT_EQ(f.kind, FrameKind::kKey);
      EXPECT_EQ(f.num_total_refs, 0);
      EXPECT_EQ(f.slot, 0);
      EXPECT_EQ(f.refresh_frame_flags,
                (std::uint32_t{1} << static_cast<unsigned>(plan.num_slots)) - 1U);
    } else {
      EXPECT(f.kind == FrameKind::kP || f.kind == FrameKind::kB);
      EXPECT(f.num_total_refs >= 1 && f.num_total_refs <= kMaxExplicitRefs);
      EXPECT_EQ(f.refresh_frame_flags, std::uint32_t{1} << static_cast<unsigned>(f.slot));
      if (f.num_total_refs == 2) {
        EXPECT(f.ref_slot[0] != f.ref_slot[1]);
        EXPECT(f.ref_display[0] != f.ref_display[1]);
      }
    }
  }
}

void check_decode_trace(const Plan& plan) {
  DecodeReport report;
  EXPECT(simulate_decode(plan, &report));
  EXPECT(report.ok);
  EXPECT(report.output_matches_display_order);
  EXPECT_EQ(static_cast<int>(report.frames.size()), plan.display_count());
  EXPECT(report.decode_order == plan.coding_order);
  for (int coding = 0; coding < plan.display_count(); ++coding) {
    const DecodeFrameTrace& trace = report.frames[static_cast<std::size_t>(coding)];
    EXPECT_EQ(trace.coding_index, coding);
    EXPECT_EQ(trace.display_index, plan.coding_order[static_cast<std::size_t>(coding)]);
    EXPECT(trace.ref_slots_held_intended_frames);
    EXPECT(trace.past_future_ok);
    EXPECT(trace.refresh_ok);
    EXPECT_EQ(report.slot_of_frame[static_cast<std::size_t>(trace.display_index)],
              plan.frames[static_cast<std::size_t>(trace.display_index)].slot);
    const FramePlan& f = plan.frames[static_cast<std::size_t>(trace.display_index)];
    if (!f.is_key) {
      EXPECT(trace.auto_num_total_refs >= 1);
      EXPECT(trace.num_past + trace.num_future + trace.num_cur == f.num_total_refs);
    }
  }
}

// ---------------------------------------------------------------------------
// Pattern matrix: every GOP size, every pattern.
// ---------------------------------------------------------------------------

void test_pattern_matrix() {
  for (GopPattern pattern :
       {GopPattern::kLowDelayP, GopPattern::kIpbb, GopPattern::kHierarchicalB}) {
    for (int gop_size : {4, 8, 16}) {
      GopConfig config = make_gop_config(pattern, gop_size);
      Plan plan;
      PlanError error = PlanError::kInternal;
      const bool built = build_gop(config, plan, &error);
      EXPECT(built);
      EXPECT_EQ(error, PlanError::kOk);
      if (!built) {
        continue;
      }
      EXPECT_EQ(plan.display_count(), gop_size);
      EXPECT(plan.frames[0].is_key);
      EXPECT_EQ(plan.frames[0].coding_index, 0);
      EXPECT(plan.frames[static_cast<std::size_t>(gop_size - 1)].is_anchor);
      EXPECT_EQ(plan.frames[static_cast<std::size_t>(gop_size - 1)].kind, FrameKind::kP);

      ValidationReport report;
      EXPECT(validate_plan(plan, &report));
      check_common_plan(plan, report);
      check_decode_trace(plan);

      // Determinism: a second build produces the same schedule bit for bit.
      Plan again;
      PlanError again_error = PlanError::kInternal;
      EXPECT(build_gop(config, again, &again_error));
      if (again_error == PlanError::kOk) {
        EXPECT(plan_equal(plan, again));
      }

      if (pattern == GopPattern::kHierarchicalB) {
        const int expected_max = gop_size == 4 ? 2 : (gop_size == 8 ? 3 : 4);
        EXPECT_EQ(max_temporal_layer(plan), expected_max);
      } else if (pattern == GopPattern::kIpbb) {
        EXPECT_EQ(max_temporal_layer(plan), 1);
      } else {
        EXPECT_EQ(max_temporal_layer(plan), 0);
      }

      for (const FramePlan& f : plan.frames) {
        if (f.is_key) {
          continue;
        }
        if (pattern != GopPattern::kLowDelayP && f.kind == FrameKind::kB) {
          EXPECT(f.ref_display[0] < f.display_index);
          EXPECT(f.ref_display[1] > f.display_index);
          // A child never references a deeper temporal layer.
          for (int r = 0; r < f.num_total_refs; ++r) {
            EXPECT(plan.frames[static_cast<std::size_t>(f.ref_display[r])].temporal_layer <
                   f.temporal_layer);
          }
        }
        if (f.is_anchor) {
          EXPECT_EQ(f.temporal_layer, 0);
          EXPECT_EQ(f.kind, FrameKind::kP);
          EXPECT_EQ(f.num_total_refs, 1);
          EXPECT(f.ref_display[0] < f.display_index);
        }
      }

      // Order-hint LSBs are the low bits of the absolute hint when the unit is
      // self-contained and OrderHintBits is large enough.
      const int mask = (1 << plan.order_hint_bits) - 1;
      for (const FramePlan& f : plan.frames) {
        EXPECT_EQ(f.order_hint_lsbs, f.display_order_hint & mask);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Exact schedules
// ---------------------------------------------------------------------------

void test_ipbb_schedule() {
  GopConfig config = make_gop_config(GopPattern::kIpbb, 4);
  Plan plan;
  PlanError error;
  EXPECT(build_gop(config, plan, &error));
  if (plan.display_count() != 4) {
    return;
  }
  EXPECT(plan.coding_order == std::vector<int>({0, 3, 1, 2}));
  EXPECT_EQ(plan.frames[0].kind, FrameKind::kKey);
  EXPECT_EQ(plan.frames[1].kind, FrameKind::kB);
  EXPECT_EQ(plan.frames[2].kind, FrameKind::kB);
  EXPECT_EQ(plan.frames[3].kind, FrameKind::kP);
  EXPECT_EQ(plan.frames[1].ref_display[0], 0);
  EXPECT_EQ(plan.frames[1].ref_display[1], 3);
  EXPECT_EQ(plan.frames[2].ref_display[0], 0);
  EXPECT_EQ(plan.frames[2].ref_display[1], 3);
  EXPECT_EQ(plan.frames[3].ref_display[0], 0);

  config = make_gop_config(GopPattern::kIpbb, 8);
  EXPECT(build_gop(config, plan, &error));
  if (plan.display_count() != 8) {
    return;
  }
  EXPECT(plan.coding_order == std::vector<int>({0, 7, 1, 2, 3, 4, 5, 6}));
  for (int d = 1; d < 7; ++d) {
    EXPECT_EQ(plan.frames[static_cast<std::size_t>(d)].kind, FrameKind::kB);
    EXPECT_EQ(plan.frames[static_cast<std::size_t>(d)].ref_display[0], 0);
    EXPECT_EQ(plan.frames[static_cast<std::size_t>(d)].ref_display[1], 7);
  }
}

void test_hierarchical_schedule() {
  GopConfig config = make_gop_config(GopPattern::kHierarchicalB, 8);
  Plan plan;
  PlanError error;
  EXPECT(build_gop(config, plan, &error));
  if (plan.display_count() != 8) {
    return;
  }
  EXPECT(plan.coding_order == std::vector<int>({0, 7, 3, 1, 5, 2, 4, 6}));
  const int expected_refs[8][2] = {{-1, -1}, {0, 3}, {1, 3}, {0, 7},
                                   {3, 5},  {3, 7}, {5, 7}, {0, -1}};
  const int expected_layers[8] = {0, 2, 3, 1, 3, 2, 3, 0};
  for (int d = 0; d < 8; ++d) {
    const FramePlan& f = plan.frames[static_cast<std::size_t>(d)];
    EXPECT_EQ(f.temporal_layer, expected_layers[d]);
    if (expected_refs[d][0] >= 0) {
      EXPECT_EQ(f.ref_display[0], expected_refs[d][0]);
    }
    if (expected_refs[d][1] >= 0) {
      EXPECT_EQ(f.ref_display[1], expected_refs[d][1]);
    }
  }

  config = make_gop_config(GopPattern::kHierarchicalB, 16);
  EXPECT(build_gop(config, plan, &error));
  if (plan.display_count() != 16) {
    return;
  }
  EXPECT(plan.coding_order ==
         std::vector<int>({0, 15, 7, 3, 11, 1, 5, 9, 13, 2, 4, 6, 8, 10, 12, 14}));
  EXPECT_EQ(plan.frames[0].temporal_layer, 0);
  EXPECT_EQ(plan.frames[15].temporal_layer, 0);
  EXPECT_EQ(plan.frames[7].temporal_layer, 1);
  EXPECT_EQ(plan.frames[3].temporal_layer, 2);
  EXPECT_EQ(plan.frames[1].temporal_layer, 3);
  EXPECT_EQ(plan.frames[2].temporal_layer, 4);
}

// ---------------------------------------------------------------------------
// Key frame + P combinations and slot recycling
// ---------------------------------------------------------------------------

void test_low_delay_p() {
  GopConfig config = make_gop_config(GopPattern::kLowDelayP, 8, 2);
  Plan plan;
  PlanError error;
  EXPECT(build_gop(config, plan, &error));
  EXPECT_EQ(error, PlanError::kOk);
  if (plan.display_count() != 8) {
    return;
  }
  EXPECT(plan.coding_order == std::vector<int>({0, 1, 2, 3, 4, 5, 6, 7}));
  for (int d = 1; d < 8; ++d) {
    const FramePlan& f = plan.frames[static_cast<std::size_t>(d)];
    EXPECT_EQ(f.kind, FrameKind::kP);
    EXPECT_EQ(f.ref_display[0], d - 1);
    EXPECT_EQ(f.ref_slot[0], plan.frames[static_cast<std::size_t>(d - 1)].slot);
  }
  // Two slots are enough for a display-order P chain. Slot 0 is recycled once
  // the key frame is no longer referenced; slot 1 is used only by P1.
  EXPECT_EQ(plan.frames[0].slot, 0);
  EXPECT_EQ(plan.frames[1].slot, 1);
  for (int d = 2; d < 8; ++d) {
    EXPECT_EQ(plan.frames[static_cast<std::size_t>(d)].slot, 0);
  }

  ValidationReport report;
  EXPECT(validate_plan(plan, &report));
  check_common_plan(plan, report);
  check_decode_trace(plan);

  // The automatic §7.7 ranking must agree with the explicit nearest-past
  // reference for this simple chain.
  DecodeReport decode;
  EXPECT(simulate_decode(plan, &decode));
  for (std::size_t i = 1; i < decode.frames.size(); ++i) {
    const DecodeFrameTrace& trace = decode.frames[i];
    const FramePlan& f = plan.frames[static_cast<std::size_t>(trace.display_index)];
    EXPECT(trace.auto_num_total_refs >= 1);
    EXPECT_EQ(trace.auto_ref_slot[0], f.ref_slot[0]);
    EXPECT_EQ(trace.auto_ref_order_hint[0], f.ref_order_hint[0]);
    EXPECT(trace.auto_scores_distance[0] > 0);
  }
}

// ---------------------------------------------------------------------------
// Leading pictures
// ---------------------------------------------------------------------------

void test_leading_pictures() {
  for (int gop_size : {4, 8}) {
    for (int leading : {1, 2, 3}) {
      GopConfig config = make_gop_config(GopPattern::kHierarchicalB, gop_size);
      config.leading_pictures = leading;
      Plan plan;
      PlanError error;
      const bool built = build_gop(config, plan, &error);
      EXPECT(built);
      EXPECT_EQ(error, PlanError::kOk);
      if (!built) {
        continue;
      }
      const int key_display = leading;
      EXPECT_EQ(plan.display_count(), gop_size + leading);
      EXPECT(plan.frames[static_cast<std::size_t>(key_display)].is_key);
      EXPECT_EQ(plan.coding_order[0], key_display);
      EXPECT(!plan.frames[static_cast<std::size_t>(key_display)].immediate_output_frame);
      EXPECT(plan.frames[static_cast<std::size_t>(key_display)].implicit_output_frame);

      // Leading pictures are decoded latest-display first, immediately after
      // the key frame.
      for (int i = 0; i < leading; ++i) {
        EXPECT_EQ(plan.coding_order[static_cast<std::size_t>(1 + i)], leading - 1 - i);
        EXPECT(plan.frames[static_cast<std::size_t>(leading - 1 - i)].is_leading);
      }

      ValidationReport report;
      EXPECT(validate_plan(plan, &report));
      check_common_plan(plan, report);
      check_decode_trace(plan);

      DecodeReport decode;
      EXPECT(simulate_decode(plan, &decode));
      for (const FramePlan& f : plan.frames) {
        if (f.is_leading) {
          // No past reference can survive the key-frame reset.
          for (int r = 0; r < f.num_total_refs; ++r) {
            EXPECT(f.ref_display[r] > f.display_index);
            EXPECT(f.ref_display[r] <= key_display);
          }
        } else if (!f.is_key) {
          for (int r = 0; r < f.num_total_refs; ++r) {
            EXPECT(f.ref_display[r] >= key_display);
            EXPECT(!plan.frames[static_cast<std::size_t>(f.ref_display[r])].is_leading);
          }
        }
      }
      for (const DecodeFrameTrace& trace : decode.frames) {
        const FramePlan& f = plan.frames[static_cast<std::size_t>(trace.display_index)];
        if (f.is_leading) {
          EXPECT_EQ(trace.num_past, 0);
          EXPECT_EQ(trace.num_future, f.num_total_refs);
          EXPECT_EQ(trace.num_cur, 0);
        }
      }

      // A leading picture may not reference a regular frame displayed after
      // the key frame.
      Plan bad = plan;
      const int regular = key_display + 1;
      FramePlan& b = bad.frames[0];
      b.ref_display[0] = regular;
      b.ref_slot[0] = bad.frames[static_cast<std::size_t>(regular)].slot;
      b.ref_order_hint[0] =
          bad.frames[static_cast<std::size_t>(regular)].display_order_hint;
      EXPECT_ERROR(bad, PlanError::kLeadingPictureRule);

      // A regular frame may not reference a leading picture.
      bad = plan;
      FramePlan& regular_frame = bad.frames[static_cast<std::size_t>(regular)];
      regular_frame.ref_display[0] = 0;
      regular_frame.ref_slot[0] = bad.frames[0].slot;
      regular_frame.ref_order_hint[0] = bad.frames[0].display_order_hint;
      EXPECT_ERROR(bad, PlanError::kLeadingPictureRule);
    }
  }
}

// ---------------------------------------------------------------------------
// Sequences
// ---------------------------------------------------------------------------

void test_sequences() {
  for (int leading : {0, 2}) {
    SequenceConfig config;
    config.num_gops = 3;
    config.gop_size = 8;
    config.pattern = GopPattern::kHierarchicalB;
    config.num_slots = kMaxRefSlots;
    config.order_hint_bits = 4;
    config.leading_pictures = leading;

    Plan plan;
    PlanError error;
    const bool built = build_sequence(config, plan, &error);
    EXPECT(built);
    EXPECT_EQ(error, PlanError::kOk);
    if (!built) {
      continue;
    }
    const int expected_frames = 8 + 2 * (8 + leading);
    EXPECT_EQ(plan.display_count(), expected_frames);
    EXPECT(plan.is_sequence);

    ValidationReport report;
    EXPECT(validate_plan(plan, &report));
    check_common_plan(plan, report);
    check_decode_trace(plan);

    // Key frames at 0, 8 + leading, and 16 + 2*leading (the first GOP has no
    // leading pictures).
    EXPECT(plan.frames[0].is_key);
    EXPECT(plan.frames[static_cast<std::size_t>(8 + leading)].is_key);
    EXPECT(plan.frames[static_cast<std::size_t>(16 + 2 * leading)].is_key);

    // Each GOP starts decoding with its key frame and does not reference a
    // previous GOP.
    std::vector<int> gop_base(3, -1);
    std::vector<int> gop_key(3, -1);
    for (const FramePlan& f : plan.frames) {
      const int g = f.gop_index;
      if (gop_base[static_cast<std::size_t>(g)] < 0) {
        gop_base[static_cast<std::size_t>(g)] = f.display_index;
      }
      if (f.is_key) {
        gop_key[static_cast<std::size_t>(g)] = f.display_index;
      }
    }
    for (int g = 0; g < 3; ++g) {
      EXPECT(gop_base[static_cast<std::size_t>(g)] >= 0);
      EXPECT(gop_key[static_cast<std::size_t>(g)] >= 0);
    }
    for (const FramePlan& f : plan.frames) {
      const int base = gop_base[static_cast<std::size_t>(f.gop_index)];
      for (int r = 0; r < f.num_total_refs; ++r) {
        EXPECT(f.ref_display[r] >= base);
      }
    }

    // Determinism.
    Plan again;
    PlanError again_error;
    EXPECT(build_sequence(config, again, &again_error));
    if (again_error == PlanError::kOk) {
      EXPECT(plan_equal(plan, again));
    }
  }
}

// ---------------------------------------------------------------------------
// Error cases
// ---------------------------------------------------------------------------

void test_sequence_matrix() {
  for (GopPattern pattern :
       {GopPattern::kLowDelayP, GopPattern::kIpbb, GopPattern::kHierarchicalB}) {
    for (int gop_size : {4, 8}) {
      for (int leading : {0, 2}) {
        SequenceConfig config;
        config.num_gops = 2;
        config.gop_size = gop_size;
        config.pattern = pattern;
        config.num_slots = kMaxRefSlots;
        config.order_hint_bits = 4;
        config.leading_pictures = leading;

        Plan plan;
        PlanError error;
        const bool built = build_sequence(config, plan, &error);
        EXPECT(built);
        EXPECT_EQ(error, PlanError::kOk);
        if (!built) {
          continue;
        }
        EXPECT_EQ(plan.display_count(), gop_size + gop_size + leading);
        ValidationReport report;
        EXPECT(validate_plan(plan, &report));
        check_common_plan(plan, report);
        check_decode_trace(plan);

        int key_count = 0;
        for (const FramePlan& f : plan.frames) {
          if (f.is_key) {
            ++key_count;
          }
        }
        EXPECT_EQ(key_count, 2);

        Plan again;
        PlanError again_error;
        EXPECT(build_sequence(config, again, &again_error));
        if (again_error == PlanError::kOk) {
          EXPECT(plan_equal(plan, again));
        }
      }
    }
  }
}

void test_long_sequence() {
  // A long display-order chain exercises slot recycling and the iterative
  // successive-output walk without deep recursion.
  SequenceConfig config;
  config.num_gops = 32;
  config.gop_size = 4;
  config.pattern = GopPattern::kLowDelayP;
  config.num_slots = kMaxRefSlots;
  config.order_hint_bits = 4;
  config.leading_pictures = 0;

  Plan plan;
  PlanError error;
  EXPECT(build_sequence(config, plan, &error));
  EXPECT_EQ(error, PlanError::kOk);
  if (plan.display_count() == 128) {
    ValidationReport report;
    EXPECT(validate_plan(plan, &report));
    EXPECT(report.output_matches_display_order);
    EXPECT_EQ(static_cast<int>(report.output_order.size()), 128);
    for (int d = 0; d < 128 && d < static_cast<int>(report.output_order.size()); ++d) {
      EXPECT_EQ(report.output_order[static_cast<std::size_t>(d)], d);
    }
  }

  config.num_gops = 8;
  config.gop_size = 8;
  config.pattern = GopPattern::kHierarchicalB;
  EXPECT(build_sequence(config, plan, &error));
  EXPECT_EQ(error, PlanError::kOk);
  if (plan.display_count() == 64) {
    ValidationReport report;
    EXPECT(validate_plan(plan, &report));
    EXPECT(report.output_matches_display_order);
  }
}

void test_config_errors() {
  GopConfig config;
  Plan plan;
  PlanError error;

  config = make_gop_config(GopPattern::kHierarchicalB, 4);
  config.gop_size = 1;
  EXPECT(!build_gop(config, plan, &error));
  EXPECT_EQ(error, PlanError::kBadGopSize);
  config.gop_size = 17;
  EXPECT(!build_gop(config, plan, &error));
  EXPECT_EQ(error, PlanError::kBadGopSize);
  config.gop_size = 3;  // hierarchical requires a power of two
  EXPECT(!build_gop(config, plan, &error));
  EXPECT_EQ(error, PlanError::kBadGopSize);
  config = make_gop_config(GopPattern::kIpbb, 4);
  config.pattern = static_cast<GopPattern>(99);
  EXPECT(!build_gop(config, plan, &error));
  EXPECT_EQ(error, PlanError::kBadPattern);
  config = make_gop_config(GopPattern::kIpbb, 4);
  config.order_hint_bits = 0;
  EXPECT(!build_gop(config, plan, &error));
  EXPECT_EQ(error, PlanError::kBadOrderHintBits);
  config.order_hint_bits = 17;
  EXPECT(!build_gop(config, plan, &error));
  EXPECT_EQ(error, PlanError::kBadOrderHintBits);
  config = make_gop_config(GopPattern::kIpbb, 4);
  config.num_slots = 1;
  EXPECT(!build_gop(config, plan, &error));
  EXPECT_EQ(error, PlanError::kBadSlotCount);
  config.num_slots = 17;
  EXPECT(!build_gop(config, plan, &error));
  EXPECT_EQ(error, PlanError::kBadSlotCount);
  config = make_gop_config(GopPattern::kIpbb, 4);
  config.leading_pictures = -1;
  EXPECT(!build_gop(config, plan, &error));
  EXPECT_EQ(error, PlanError::kBadLeadingCount);
  config = make_gop_config(GopPattern::kIpbb, 4);
  config.start_order_hint = -1;
  EXPECT(!build_gop(config, plan, &error));
  EXPECT_EQ(error, PlanError::kBadDisplayOrder);

  SequenceConfig sequence;
  sequence.num_gops = 0;
  EXPECT(!build_sequence(sequence, plan, &error));
  EXPECT_EQ(error, PlanError::kBadGopSize);

  // 16-frame units need OrderHintBits > log2(16) because the future anchor is
  // decoded before the B frames and the §5.18 lift is relative to the largest
  // hint in the reference store.
  config = make_gop_config(GopPattern::kHierarchicalB, 16);
  config.order_hint_bits = 4;
  EXPECT(!build_gop(config, plan, &error));
  EXPECT_EQ(error, PlanError::kOrderHintReconstruct);
  config.order_hint_bits = 5;
  EXPECT(build_gop(config, plan, &error));

  // Slot exhaustion.
  config = make_gop_config(GopPattern::kHierarchicalB, 8, 2);
  EXPECT(!build_gop(config, plan, &error));
  EXPECT_EQ(error, PlanError::kNotEnoughSlots);
  config.num_slots = 3;
  EXPECT(!build_gop(config, plan, &error));
  EXPECT_EQ(error, PlanError::kNotEnoughSlots);
  config.num_slots = 4;
  EXPECT(build_gop(config, plan, &error));
  config = make_gop_config(GopPattern::kIpbb, 4, 2);
  EXPECT(!build_gop(config, plan, &error));
  EXPECT_EQ(error, PlanError::kNotEnoughSlots);

  // A non-zero start hint is legal when OrderHintBits covers it.
  config = make_gop_config(GopPattern::kIpbb, 4);
  config.start_order_hint = 40;
  config.order_hint_bits = 8;
  EXPECT(build_gop(config, plan, &error));
  if (!plan.frames.empty()) {
    EXPECT_EQ(plan.frames[0].display_order_hint, 40);
    ValidationReport report;
    EXPECT(validate_plan(plan, &report));
  }
  config.order_hint_bits = 4;
  EXPECT(!build_gop(config, plan, &error));
  EXPECT_EQ(error, PlanError::kOrderHintReconstruct);

  // Four bits also limits leading pictures: the key hint must be less than
  // half a span (8) ahead of the earliest leading hint.
  config = make_gop_config(GopPattern::kHierarchicalB, 8);
  config.leading_pictures = 7;
  config.order_hint_bits = 4;
  EXPECT(build_gop(config, plan, &error));
  config.leading_pictures = 8;
  EXPECT(!build_gop(config, plan, &error));
  EXPECT_EQ(error, PlanError::kOrderHintReconstruct);
}

void test_mutation_errors() {
  const GopConfig config = make_gop_config(GopPattern::kHierarchicalB, 8);
  Plan base;
  PlanError error;
  EXPECT(build_gop(config, base, &error));
  if (base.display_count() != 8) {
    return;
  }

  // Reference outside the defined display range.
  Plan bad = base;
  bad.frames[3].ref_display[0] = 99;
  EXPECT_ERROR(bad, PlanError::kReferenceOutOfRange);

  // Self reference.
  bad = base;
  bad.frames[3].ref_display[0] = 3;
  bad.frames[3].ref_slot[0] = bad.frames[3].slot;
  bad.frames[3].ref_order_hint[0] = bad.frames[3].display_order_hint;
  EXPECT_ERROR(bad, PlanError::kReferenceOutOfRange);

  // Duplicate display reference.
  bad = base;
  bad.frames[1].ref_display[1] = bad.frames[1].ref_display[0];
  bad.frames[1].ref_order_hint[1] = bad.frames[1].ref_order_hint[0];
  bad.frames[1].ref_slot[1] = bad.frames[1].ref_slot[0];
  EXPECT_ERROR(bad, PlanError::kDuplicateReference);

  // Duplicate slot reference.
  bad = base;
  bad.frames[1].ref_slot[1] = bad.frames[1].ref_slot[0];
  EXPECT_ERROR(bad, PlanError::kDuplicateReference);

  // The saved reference order hint must match the referenced frame.
  bad = base;
  bad.frames[3].ref_order_hint[0] = -1;
  EXPECT_ERROR(bad, PlanError::kReferenceHintMismatch);

  // A reference must already be decoded.
  bad = base;
  bad.frames[1].ref_display[0] = 2;
  bad.frames[1].ref_slot[0] = bad.frames[2].slot;
  bad.frames[1].ref_order_hint[0] = bad.frames[2].display_order_hint;
  EXPECT_ERROR(bad, PlanError::kReferenceNotDecoded);

  // The slot must hold the frame named by ref_display.
  bad = base;
  bad.frames[1].ref_slot[0] = bad.frames[5].slot;
  EXPECT_ERROR(bad, PlanError::kReferenceSlotInvalid);

  // Key frame must refresh allFrames and put the content in the first slot.
  bad = base;
  bad.frames[0].refresh_frame_flags = 1;
  EXPECT_ERROR(bad, PlanError::kRefreshFlagsInvalid);
  bad = base;
  bad.frames[0].slot = 1;
  EXPECT_ERROR(bad, PlanError::kRefreshFlagsInvalid);

  // Non-key refresh flags: exactly one slot.
  bad = base;
  bad.frames[1].refresh_frame_flags |= bad.frames[2].refresh_frame_flags;
  EXPECT_ERROR(bad, PlanError::kRefreshFlagsInvalid);
  bad = base;
  bad.frames[1].refresh_frame_flags = 0;
  EXPECT_ERROR(bad, PlanError::kRefreshFlagsInvalid);

  // Key frames carry no reference map.
  bad = base;
  bad.frames[0].num_total_refs = 1;
  bad.frames[0].ref_display[0] = 0;
  EXPECT_ERROR(bad, PlanError::kKeyFrameHasRefs);

  // Non-key frames need at least one reference; more than two is rejected.
  bad = base;
  bad.frames[1].num_total_refs = 0;
  EXPECT_ERROR(bad, PlanError::kMissingReference);
  bad = base;
  bad.frames[1].num_total_refs = 3;
  EXPECT_ERROR(bad, PlanError::kTooManyRefs);

  // This layer schedules explicit maps only.
  bad = base;
  bad.frames[1].frame_explicit_ref_frame_map = false;
  EXPECT_ERROR(bad, PlanError::kBadPattern);

  // A B frame with two past references violates the one-past/one-future rule.
  bad = base;
  {
    FramePlan& f = bad.frames[2];
    f.ref_display[1] = 0;
    f.ref_slot[1] = bad.frames[0].slot;
    f.ref_order_hint[1] = bad.frames[0].display_order_hint;
  }
  EXPECT_ERROR(bad, PlanError::kBadReferenceSplit);

  // Coding order must be a permutation with matching coding indices.
  bad = base;
  std::swap(bad.coding_order[1], bad.coding_order[2]);
  EXPECT_ERROR(bad, PlanError::kBadCodingOrder);

  // Display hints must be strictly increasing in display order.
  bad = base;
  std::swap(bad.frames[1].display_order_hint, bad.frames[2].display_order_hint);
  EXPECT_ERROR(bad, PlanError::kBadDisplayOrder);

  // Exactly one output flag.
  bad = base;
  bad.frames[1].immediate_output_frame = true;
  bad.frames[1].implicit_output_frame = true;
  EXPECT_ERROR(bad, PlanError::kOutputFlagsInvalid);

  // Unreconstructable order hint.
  bad = base;
  bad.frames[1].order_hint_lsbs = 99;
  EXPECT_ERROR(bad, PlanError::kOrderHintReconstruct);

  // Immediate output of a future anchor breaks decoder-visible order.
  bad = base;
  bad.frames[7].immediate_output_frame = true;
  bad.frames[7].implicit_output_frame = false;
  ValidationReport report;
  const bool ok = validate_plan(bad, &report);
  EXPECT(!ok);
  EXPECT_EQ(report.error, PlanError::kOutputOrderMismatch);
  EXPECT(!report.output_matches_display_order);
  EXPECT(report.output_order.size() >= 2);
  if (report.output_order.size() >= 2) {
    EXPECT_EQ(report.output_order[0], 0);
    EXPECT_EQ(report.output_order[1], 7);
  }

  // A missing key frame is rejected before any decode.
  bad = base;
  bad.frames[0].is_key = false;
  bad.frames[0].kind = FrameKind::kP;
  bad.frames[0].num_total_refs = 1;
  bad.frames[0].ref_display[0] = 1;
  bad.frames[0].ref_slot[0] = bad.frames[1].slot;
  bad.frames[0].ref_order_hint[0] = bad.frames[1].display_order_hint;
  EXPECT_ERROR(bad, PlanError::kMissingKeyFrame);

  // No reference may cross a key-frame reset in a multi-GOP sequence.
  SequenceConfig sequence;
  sequence.num_gops = 2;
  sequence.gop_size = 8;
  sequence.pattern = GopPattern::kHierarchicalB;
  sequence.num_slots = kMaxRefSlots;
  sequence.order_hint_bits = 4;
  Plan sequence_plan;
  EXPECT(build_sequence(sequence, sequence_plan, &error));
  if (sequence_plan.display_count() == 16) {
    bad = sequence_plan;
    FramePlan& f = bad.frames[9];
    f.ref_display[0] = 0;
    f.ref_order_hint[0] = bad.frames[0].display_order_hint;
    EXPECT_ERROR(bad, PlanError::kKeyFrameRefInvalidated);
  }
}

// ---------------------------------------------------------------------------
// Automatic §7.7 ranking cross-check
// ---------------------------------------------------------------------------

void test_auto_ranking_reuse() {
  for (GopPattern pattern : {GopPattern::kIpbb, GopPattern::kHierarchicalB}) {
    GopConfig config = make_gop_config(pattern, 8);
    Plan plan;
    PlanError error;
    EXPECT(build_gop(config, plan, &error));
    if (plan.display_count() != 8) {
      continue;
    }
    DecodeReport report;
    EXPECT(simulate_decode(plan, &report));
    for (const DecodeFrameTrace& trace : report.frames) {
      const FramePlan& f = plan.frames[static_cast<std::size_t>(trace.display_index)];
      if (f.is_key || f.num_total_refs != 2) {
        continue;
      }
      for (int r = 0; r < f.num_total_refs; ++r) {
        bool found = false;
        for (int i = 0; i < trace.auto_num_total_refs; ++i) {
          if (trace.auto_ref_slot[i] == f.ref_slot[r] &&
              trace.auto_ref_order_hint[i] == f.ref_order_hint[r]) {
            found = true;
          }
        }
        EXPECT(found);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Error names and API surface
// ---------------------------------------------------------------------------

void test_names() {
  EXPECT(std::strcmp(plan_error_name(PlanError::kOk), "ok") == 0);
  EXPECT(std::strcmp(plan_error_name(PlanError::kReferenceOutOfRange),
                     "reference_out_of_range") == 0);
  EXPECT(std::strcmp(gop_pattern_name(GopPattern::kIpbb), "ipbb") == 0);
  EXPECT(std::strcmp(frame_kind_name(FrameKind::kB), "b") == 0);
  EXPECT(plan_error_name(static_cast<PlanError>(255)) != nullptr);
}

}  // namespace

int main() {
  test_names();
  test_pattern_matrix();
  test_ipbb_schedule();
  test_hierarchical_schedule();
  test_low_delay_p();
  test_leading_pictures();
  test_sequences();
  test_sequence_matrix();
  test_long_sequence();
  test_config_errors();
  test_mutation_errors();
  test_auto_ranking_reuse();

  if (g_failures == 0) {
    std::printf("bframes_test: %d checks passed\n", g_checks);
    return 0;
  }
  std::printf("bframes_test: %d/%d checks FAILED\n", g_failures, g_checks);
  return 1;
}

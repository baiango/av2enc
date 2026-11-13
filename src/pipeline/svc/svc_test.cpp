// Standalone tests for the SVC layer pipeline.
//
// Covers: 2/3-layer plans, spatial+temporal+embedded configuration,
// deterministic assignment and emission order, operating-point extraction,
// Annex F.3.1 retention, Annex F.3.2 extraction, OBU/LCR/OPS framing, input
// validation, and every rejection path. No CUDA, no encoder core, no decoder
// required; gate_dav2d.sh adds the black-box decoder check.

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "svc_config.h"
#include "svc_framing.h"
#include "svc_pipeline.h"

namespace {

using av2enc::svc::ComposeOptions;
using av2enc::svc::Emission;
using av2enc::svc::EmissionKind;
using av2enc::svc::FrameRequest;
using av2enc::svc::FrameSource;
using av2enc::svc::LayerConfig;
using av2enc::svc::LayerInput;
using av2enc::svc::LayerPlan;
using av2enc::svc::LayerSelection;
using av2enc::svc::LayerSpec;
using av2enc::svc::ObuHeader;
using av2enc::svc::ObuType;
using av2enc::svc::ReferencePolicy;
using av2enc::svc::SelectionMode;
using av2enc::svc::SpatialLayerSpec;
using av2enc::tiles::svc::LayerId;
using av2enc::tiles::svc::RetentionMap;

int g_failures = 0;

void check(bool condition, const char* expression, const char* file, int line) {
  if (!condition) {
    ++g_failures;
    std::cerr << file << ':' << line << ": CHECK failed: " << expression << '\n';
  }
}

#define CHECK(expression) \
  ::check((expression), #expression, __FILE__, __LINE__)

template <typename Fn>
void expect_ok(Fn&& fn, const char* file, int line) {
  std::string error;
  if (!fn(error)) {
    ++g_failures;
    std::cerr << file << ':' << line << ": expected success, got: " << error
              << '\n';
  }
}

template <typename Fn>
void expect_fail(Fn&& fn, const char* substring, const char* file, int line) {
  std::string error;
  if (fn(error)) {
    ++g_failures;
    std::cerr << file << ':' << line << ": expected failure containing '"
              << substring << "', but the call succeeded\n";
    return;
  }
  if (error.find(substring) == std::string::npos) {
    ++g_failures;
    std::cerr << file << ':' << line << ": expected failure containing '"
              << substring << "', got: " << error << '\n';
  }
}

#define EXPECT_OK(fn) ::expect_ok(fn, __FILE__, __LINE__)
#define EXPECT_FAILS(fn, substring) \
  ::expect_fail(fn, substring, __FILE__, __LINE__)

class BitReader {
 public:
  explicit BitReader(std::span<const std::uint8_t> bytes) : bytes_(bytes) {}

  bool ok() const { return ok_; }
  std::size_t byte_position() const { return bit_ / 8; }

  std::uint32_t bits(unsigned count) {
    std::uint32_t value = 0;
    for (unsigned i = 0; i < count; ++i) {
      if (bit_ >= bytes_.size() * 8U) {
        ok_ = false;
        return value;
      }
      const std::size_t byte = bit_ / 8U;
      const unsigned shift = 7U - static_cast<unsigned>(bit_ % 8U);
      value = (value << 1U) | ((bytes_[byte] >> shift) & 1U);
      ++bit_;
    }
    return value;
  }

  bool bit() { return bits(1) != 0; }

  std::uint32_t uvlc() {
    unsigned leading_zeros = 0;
    while (!bit()) {
      ++leading_zeros;
      if (!ok_ || leading_zeros > 31U) {
        ok_ = false;
        return 0;
      }
    }
    if (leading_zeros == 0) {
      return 0;
    }
    return ((1U << leading_zeros) - 1U) + bits(leading_zeros);
  }

  std::uint32_t leb128() {
    std::uint32_t value = 0;
    unsigned shift = 0;
    while (true) {
      const std::uint8_t byte = static_cast<std::uint8_t>(bits(8));
      value |= static_cast<std::uint32_t>(byte & 0x7FU) << shift;
      if ((byte & 0x80U) == 0) {
        break;
      }
      shift += 7U;
      if (shift > 28U) {
        ok_ = false;
        break;
      }
    }
    return value;
  }

  void align() {
    while ((bit_ % 8U) != 0U) {
      (void)bits(1);
    }
  }

 private:
  std::span<const std::uint8_t> bytes_;
  std::size_t bit_ = 0;
  bool ok_ = true;
};

LayerConfig make_config(int spatial_layers, int temporal, int embedded,
                        int base = 64, int step = 64) {
  LayerConfig config;
  for (int x = 0; x < spatial_layers; ++x) {
    SpatialLayerSpec layer;
    layer.xlayer_id = static_cast<std::uint8_t>(x);
    layer.width = base + x * step;
    layer.height = base + x * step;
    config.spatial_layers.push_back(layer);
  }
  config.temporal_layers = static_cast<std::uint8_t>(temporal);
  config.embedded_layers = static_cast<std::uint8_t>(embedded);
  config.seq_lcr_id = 1;
  return config;
}

LayerPlan make_plan(int spatial_layers, int temporal = 1, int embedded = 1) {
  LayerPlan plan;
  std::string error;
  if (!av2enc::svc::build_layer_plan(
          make_config(spatial_layers, temporal, embedded), plan, error)) {
    std::cerr << "test setup failed: " << error << '\n';
    std::abort();
  }
  return plan;
}

LayerId layer(std::uint8_t x, std::uint8_t m = 0, std::uint8_t t = 0) {
  LayerId id;
  id.xlayer_id = x;
  id.mlayer_id = m;
  id.tlayer_id = t;
  return id;
}

// Minimal well-formed §5.4.1 prefix through seq_lcr_id (single_picture_flag =
// 0, level 5.1, 8-bit 4:2:0).  The rest of the payload is zero padding; the
// compose step only rewrites the 3-bit seq_lcr_id field.
std::vector<std::uint8_t> sequence_payload_prefix(std::uint8_t lcr_id,
                                                  bool single_picture = false) {
  std::vector<std::uint8_t> bytes(3, 0);
  std::size_t bit = 0;
  const auto put = [&](std::uint32_t value, unsigned count) {
    for (unsigned i = 0; i < count; ++i) {
      const unsigned shift = count - 1U - i;
      if (((value >> shift) & 1U) != 0U) {
        bytes[(bit + i) / 8U] |= static_cast<std::uint8_t>(
            1U << (7U - ((bit + i) % 8U)));
      }
    }
    bit += count;
  };
  put(1U, 1U);                    // seq_header_id uvlc(0)
  put(0U, 5U);                    // seq_profile_idc
  put(single_picture ? 1U : 0U, 1U);  // single_picture_header_flag
  put(7U, 5U);                    // seq_level_idx = 7 (> 3)
  if (!single_picture) put(0U, 1U);   // seq_tier
  put(1U, 1U);                    // chroma_format_idc uvlc(0)
  put(2U, 3U);                    // bit_depth_idc uvlc(1)
  if (!single_picture) put(lcr_id, 3U);  // seq_lcr_id
  return bytes;
}

std::vector<std::uint8_t> single_layer_unit(
    std::uint8_t marker, std::span<const std::uint8_t> sequence_payload) {  std::vector<std::uint8_t> out;
  std::string error;
  ObuHeader td;
  td.type = ObuType::kTemporalDelimiter;
  td.xlayer_id = 31;
  CHECK(av2enc::svc::frame_obu(td, {}, out, error));

  ObuHeader seq;
  seq.type = ObuType::kSequenceHeader;
  CHECK(av2enc::svc::frame_obu(seq, sequence_payload, out, error));

  ObuHeader frame;
  frame.type = ObuType::kClosedLoopKey;
  const std::uint8_t payload[3] = {marker, 0x5A, 0x33};
  CHECK(av2enc::svc::frame_obu(frame, payload, out, error));
  return out;
}

void test_two_layer_plan() {
  LayerConfig config = make_config(2, 2, 1);
  LayerPlan plan;
  std::string error;
  CHECK(av2enc::svc::build_layer_plan(config, plan, error));

  CHECK(plan.sequences.size() == 2);
  CHECK(plan.sequences[0].max_width == 64 && plan.sequences[0].max_height == 64);
  CHECK(plan.sequences[1].max_width == 128 &&
        plan.sequences[1].max_height == 128);
  CHECK(plan.sequences[0].max_mlayer_id == 0);
  CHECK(plan.sequences[0].max_tlayer_id == 1);

  CHECK(plan.lcr.is_global);
  CHECK(plan.lcr.xlayer_id == 31);
  CHECK(plan.lcr.xlayer_map == 0x3U);
  CHECK(plan.lcr.layers.size() == 2);
  CHECK(plan.lcr.layers[0].mlayer_map == 0x1U);
  CHECK(plan.lcr.layers[0].tlayer_map[0] == 0x3U);
  CHECK(!plan.lcr.layers[0].dependent_xlayers);

  CHECK(plan.ops.points.size() == 2);
  CHECK(plan.ops.points[0].xlayer_map == 0x1U);
  CHECK(plan.ops.points[1].xlayer_map == 0x3U);
  CHECK(plan.operating_points.size() == 2);
  CHECK(plan.operating_points[0].spatial_layers == 1);
  CHECK(plan.operating_points[1].spatial_layers == 2);
  CHECK(plan.operating_points[1].temporal_layers == 2);

  CHECK(plan.layers.size() == 4);
  CHECK(plan.layers[0].id == layer(0, 0, 0));
  CHECK(plan.layers[1].id == layer(0, 0, 1));
  CHECK(plan.layers[2].id == layer(1, 0, 0));
  CHECK(plan.layers[3].id == layer(1, 0, 1));

  EXPECT_OK([&](std::string& e) { return av2enc::svc::validate_layer_plan(plan, e); });
}

void test_three_layer_plan() {
  LayerPlan plan = make_plan(3, 3, 1);
  CHECK(plan.sequences.size() == 3);
  CHECK(plan.sequences[2].max_width == 192);
  CHECK(plan.sequences[2].max_tlayer_id == 2);
  CHECK(plan.layers.size() == 9);
  for (int i = 0; i < 9; ++i) {
    const LayerSpec& spec = plan.layers[static_cast<std::size_t>(i)];
    CHECK(static_cast<int>(spec.id.xlayer_id) == i / 3);
    CHECK(static_cast<int>(spec.id.tlayer_id) == i % 3);
  }
  CHECK(plan.operating_points[2].spatial_layers == 3);
  CHECK(plan.operating_points[2].temporal_layers == 3);
  EXPECT_OK([&](std::string& e) { return av2enc::svc::validate_layer_plan(plan, e); });
}

void test_embedded_layers() {
  LayerPlan plan = make_plan(2, 2, 2);
  CHECK(plan.layers.size() == 8);
  CHECK(plan.lcr.layers[0].mlayer_map == 0x3U);
  CHECK(plan.lcr.layers[0].tlayer_map[1] == 0x3U);
  CHECK(plan.ops.points[0].mlayer_map[0] == 0x3U);
  CHECK(plan.ops.points[0].tlayer_map[0][1] == 0x3U);

  RetentionMap retention;
  LayerSelection selection;
  selection.mode = SelectionMode::kOperatingPoint;
  selection.op_index = 0;
  EXPECT_OK([&](std::string& e) {
    return av2enc::svc::resolve_selection(plan, selection, retention, e);
  });
  CHECK(av2enc::svc::selected_layer_count(retention, plan) == 4);
  CHECK(av2enc::svc::retention_has_layer(retention, layer(0, 1, 1)));
  CHECK(!av2enc::svc::retention_has_layer(retention, layer(1, 0, 0)));
}

void test_operating_point_extraction() {
  LayerPlan plan = make_plan(3, 2, 1);
  RetentionMap retention;

  for (int op = 0; op < 3; ++op) {
    LayerSelection selection;
    selection.mode = SelectionMode::kOperatingPoint;
    selection.op_index = op;
    EXPECT_OK([&](std::string& e) {
      return av2enc::svc::resolve_selection(plan, selection, retention, e);
    });
    CHECK(av2enc::svc::selected_layer_count(retention, plan) == (op + 1) * 2);
    for (int x = 0; x < 3; ++x) {
      CHECK(retention.xlayer_selected[static_cast<std::size_t>(x)] == (x <= op));
      for (int t = 0; t < 2; ++t) {
        CHECK(av2enc::svc::retention_has_layer(
                  retention, layer(static_cast<std::uint8_t>(x), 0,
                                   static_cast<std::uint8_t>(t))) == (x <= op));
      }
    }
    CHECK(retention.xlayer_selected[31]);
    CHECK(retention.retain[31][0][0]);
  }

  LayerSelection all;
  EXPECT_OK([&](std::string& e) {
    return av2enc::svc::resolve_selection(plan, all, retention, e);
  });
  CHECK(av2enc::svc::selected_layer_count(retention, plan) == 6);

  LayerSelection partial;
  partial.mode = SelectionMode::kUpToSpatialLayer;
  partial.max_xlayer = 1;
  EXPECT_OK([&](std::string& e) {
    return av2enc::svc::resolve_selection(plan, partial, retention, e);
  });
  CHECK(av2enc::svc::selected_layer_count(retention, plan) == 4);
  CHECK(!av2enc::svc::retention_has_layer(retention, layer(2, 0, 0)));

  LayerSelection bad_op;
  bad_op.mode = SelectionMode::kOperatingPoint;
  bad_op.op_index = 3;
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::resolve_selection(plan, bad_op, retention, e);
      },
      "outside 0..2");

  LayerSelection bad_max;
  bad_max.mode = SelectionMode::kUpToSpatialLayer;
  bad_max.max_xlayer = 3;
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::resolve_selection(plan, bad_max, retention, e);
      },
      "outside 0..2");
}

void test_retention_closure() {
  LayerPlan plan = make_plan(2, 2, 2);
  RetentionMap retention;
  retention.retain[0][0][0] = true;
  // m1/t1 is retained but its m0/t0 dependency is not: reject.
  retention.retain[0][1][1] = true;
  retention.xlayer_selected[0] = true;
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::validate_retention_closed(plan, retention, e);
      },
      "drops dependency");

  RetentionMap absent;
  absent.retain[2][0][0] = true;
  absent.xlayer_selected[2] = true;
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::validate_retention_closed(plan, absent, e);
      },
      "absent layer");

  RetentionMap empty_selection;
  empty_selection.xlayer_selected[0] = true;
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::validate_retention_closed(plan, empty_selection, e);
      },
      "retains no layer");
}

void test_retention_expansion() {
  LayerPlan plan = make_plan(1, 3, 1);
  av2enc::tiles::svc::OperatingPointSet set = plan.ops;
  // Select only t2; Annex F.3.1 must add t1 and t0 transitively.
  set.points[0].tlayer_map[0][0] = 0x4U;
  av2enc::tiles::svc::RetentionMap retention;
  std::string error;
  CHECK(av2enc::tiles::svc::build_retention_map(
      set, 0, plan.sequences[0].dependencies, retention, error));
  CHECK(retention.retain[0][0][0]);
  CHECK(retention.retain[0][0][1]);
  CHECK(retention.retain[0][0][2]);
  retention.xlayer_selected[0] = true;
  EXPECT_OK([&](std::string& e) {
    return av2enc::svc::validate_retention_closed(plan, retention, e);
  });
  CHECK(av2enc::svc::selected_layer_count(retention, plan) == 3);

  // Selecting t1 only leaves t0 too.
  set.points[0].tlayer_map[0][0] = 0x2U;
  retention = av2enc::tiles::svc::RetentionMap{};
  CHECK(av2enc::tiles::svc::build_retention_map(
      set, 0, plan.sequences[0].dependencies, retention, error));
  CHECK(retention.retain[0][0][0]);
  CHECK(retention.retain[0][0][1]);
  CHECK(!retention.retain[0][0][2]);
}

void test_assignment() {
  LayerPlan plan = make_plan(2, 2, 1);
  std::vector<FrameSource> sources;
  sources.push_back(FrameSource{0, 0, 64, 64, 0});
  sources.push_back(FrameSource{0, 1, 128, 128, 1});
  std::vector<LayerId> assigned;
  EXPECT_OK([&](std::string& e) {
    return av2enc::svc::assign_layers(plan, sources, assigned, e);
  });
  CHECK(assigned.size() == 2);
  CHECK(assigned[0] == layer(0, 0, 0));
  CHECK(assigned[1] == layer(1, 0, 1));

  std::vector<FrameSource> ambiguous;
  LayerConfig same_size = make_config(2, 1, 1);
  same_size.spatial_layers[1].width = same_size.spatial_layers[0].width;
  same_size.spatial_layers[1].height = same_size.spatial_layers[0].height;
  LayerPlan ambiguous_plan;
  std::string plan_error;
  CHECK(av2enc::svc::build_layer_plan(same_size, ambiguous_plan, plan_error));
  ambiguous.push_back(FrameSource{0, 0, 64, 64, 0});
  ambiguous.push_back(FrameSource{0, 0, 64, 64, 1});
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::assign_layers(ambiguous_plan, ambiguous, assigned, e);
      },
      "multiple spatial layers");

  std::vector<FrameSource> none;
  none.push_back(FrameSource{0, 0, 100, 100, 0});
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::assign_layers(plan, none, assigned, e);
      },
      "matches no spatial layer");

  std::vector<FrameSource> bad_t;
  bad_t.push_back(FrameSource{0, 2, 64, 64, 0});
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::assign_layers(plan, bad_t, assigned, e);
      },
      "above max_tlayer_id");

  LayerPlan embedded_plan = make_plan(1, 1, 2);
  std::vector<FrameSource> embedded;
  embedded.push_back(FrameSource{0, 0, 64, 64, 0, 1});
  EXPECT_OK([&](std::string& e) {
    return av2enc::svc::assign_layers(embedded_plan, embedded, assigned, e);
  });
  CHECK(assigned.size() == 1 && assigned[0].mlayer_id == 1);
  embedded[0].mlayer = 2;
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::assign_layers(embedded_plan, embedded, assigned, e);
      },
      "above max_mlayer_id");

  std::vector<FrameSource> duplicate;
  duplicate.push_back(FrameSource{0, 0, 64, 64, 0});
  duplicate.push_back(FrameSource{0, 0, 64, 64, 1});
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::assign_layers(plan, duplicate, assigned, e);
      },
      "has two frames for layer");
}

void test_emission_order() {
  LayerPlan plan = make_plan(2, 2, 1);
  std::vector<Emission> order;
  EXPECT_OK([&](std::string& e) {
    return av2enc::svc::plan_emission_order(plan, 2, order, e);
  });
  CHECK(order.size() == 14);
  CHECK(av2enc::svc::frame_emission_count(order) == 8);
  CHECK(order[0].kind == EmissionKind::kTemporalDelimiter);
  CHECK(order[0].layer.xlayer_id == 31);
  CHECK(order[1].kind == EmissionKind::kLayerConfigRecord);
  CHECK(order[2].kind == EmissionKind::kOperatingPointSet);
  CHECK(order[3].kind == EmissionKind::kSequenceHeader);
  CHECK(order[3].layer == layer(0, 0, 0));
  CHECK(order[4].kind == EmissionKind::kFrame && order[4].layer == layer(0, 0, 0));
  CHECK(order[5].layer == layer(0, 0, 1));
  CHECK(order[6].kind == EmissionKind::kSequenceHeader);
  CHECK(order[6].layer == layer(1, 0, 0));
  CHECK(order[7].layer == layer(1, 0, 0));
  CHECK(order[8].layer == layer(1, 0, 1));
  CHECK(order[9].kind == EmissionKind::kTemporalDelimiter);
  CHECK(order[9].temporal_unit == 1);
  CHECK(order[10].layer == layer(0, 0, 0));
  CHECK(order[13].layer == layer(1, 0, 1));

  std::vector<Emission> repeat;
  EXPECT_OK([&](std::string& e) {
    return av2enc::svc::plan_emission_order(plan, 2, repeat, e);
  });
  CHECK(order.size() == repeat.size());
  for (std::size_t i = 0; i < order.size(); ++i) {
    CHECK(order[i].kind == repeat[i].kind);
    CHECK(order[i].layer == repeat[i].layer);
    CHECK(order[i].temporal_unit == repeat[i].temporal_unit);
  }

  std::vector<Emission> empty;
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::plan_emission_order(plan, 0, empty, e);
      },
      "at least 1");

  std::vector<FrameRequest> requests;
  FrameRequest late;
  late.temporal_unit = 0;
  late.layer = layer(1, 0, 0);
  requests.push_back(late);
  FrameRequest early;
  early.temporal_unit = 0;
  early.layer = layer(0, 0, 0);
  requests.push_back(early);
  FrameRequest non_output;
  non_output.temporal_unit = 0;
  non_output.layer = layer(0, 0, 0);
  non_output.output = false;
  requests.push_back(non_output);
  EXPECT_OK([&](std::string& e) {
    return av2enc::svc::order_frames(plan, requests, order, e);
  });
  CHECK(order.size() == 4);  // TD + 3 frame units
  CHECK(order[0].kind == EmissionKind::kTemporalDelimiter);
  CHECK(order[1].layer == layer(0, 0, 0) && !order[1].output);
  CHECK(order[2].layer == layer(0, 0, 0) && order[2].output);
  CHECK(order[3].layer == layer(1, 0, 0));

  std::vector<FrameRequest> duplicate = requests;
  duplicate.push_back(early);
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::order_frames(plan, duplicate, order, e);
      },
      "repeats layer");

  std::vector<FrameRequest> absent;
  absent.push_back(FrameRequest{0, layer(2, 0, 0), av2enc::svc::FrameRole::kIntra,
                                -1, true});
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::order_frames(plan, absent, order, e);
      },
      "absent layer");

  std::vector<FrameRequest> none;
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::order_frames(plan, none, order, e);
      },
      "empty");
}

void test_reference_policy() {
  LayerPlan plan = make_plan(2, 2, 1);
  ReferencePolicy policy;
  std::string error;

  CHECK(!av2enc::svc::reference_allowed(plan, policy, layer(1, 0, 0),
                                        layer(0, 0, 0), error));
  CHECK(error.find("6.8.2") != std::string::npos);

  // §6.8.2 is a conformance requirement, so even a caller that asks for
  // cross-xlayer prediction gets the same named rejection.
  policy.cross_xlayer_prediction = true;
  error.clear();
  CHECK(!av2enc::svc::reference_allowed(plan, policy, layer(1, 0, 0),
                                        layer(0, 0, 0), error));
  CHECK(error.find("6.8.2") != std::string::npos);

  // The structural validator rejects a plan that declares it at all.
  plan.lcr.layers[1].dependent_xlayers = true;
  plan.lcr.layers[1].dependent_xlayer_ids.push_back(0);
  error.clear();
  CHECK(!av2enc::svc::validate_layer_plan(plan, error));
  CHECK(error.find("6.8.2") != std::string::npos);
  plan.lcr.layers[1].dependent_xlayers = false;
  plan.lcr.layers[1].dependent_xlayer_ids.clear();

  ReferencePolicy defaults;
  CHECK(av2enc::svc::reference_allowed(plan, defaults, layer(0, 0, 1),
                                       layer(0, 0, 0), error));
  error.clear();
  CHECK(!av2enc::svc::reference_allowed(plan, defaults, layer(0, 0, 0),
                                        layer(0, 0, 1), error));
  CHECK(error.find("forward") != std::string::npos);

  // §5.4.1 MLayerDependencyMap: a higher embedded layer may predict from a
  // lower one in the same xlayer (the --svc-embedded path), never the reverse.
  LayerPlan embedded_plan = make_plan(2, 2, 2);
  CHECK(av2enc::svc::reference_allowed(embedded_plan, defaults,
                                       layer(0, 1, 0), layer(0, 0, 0), error));
  error.clear();
  CHECK(!av2enc::svc::reference_allowed(embedded_plan, defaults,
                                        layer(0, 0, 0), layer(0, 1, 0),
                                        error));
  CHECK(error.find("lower mlayer") != std::string::npos);

  ReferencePolicy mlayer;
  mlayer.cross_mlayer_prediction = false;
  error.clear();
  CHECK(!av2enc::svc::reference_allowed(embedded_plan, mlayer, layer(0, 1, 0),
                                        layer(0, 0, 0), error));
  CHECK(error.find("cross-mlayer") != std::string::npos);

  CHECK(!av2enc::svc::reference_allowed(plan, defaults, layer(0, 0, 0),
                                        layer(0, 0, 0), error));
  CHECK(!av2enc::svc::reference_allowed(plan, defaults, layer(0, 0, 0),
                                        layer(2, 0, 0), error));

  std::vector<Emission> order;
  EXPECT_OK([&](std::string& e) {
    return av2enc::svc::plan_emission_order(plan, 2, order, e);
  });
  // Same-xlayer ordering checks: cross-xlayer references cannot exist (§6.8.2),
  // so scheduling is exercised with a lower temporal layer in the same x/m.
  std::vector<av2enc::svc::FrameReference> references;
  references.push_back(av2enc::svc::FrameReference{0, layer(0, 0, 1), 0,
                                                   layer(0, 0, 0)});
  EXPECT_OK([&](std::string& e) {
    return av2enc::svc::validate_frame_references(plan, policy, references,
                                                  order, e);
  });

  std::vector<av2enc::svc::FrameReference> cross_unit;
  cross_unit.push_back(
      av2enc::svc::FrameReference{1, layer(0, 0, 1), 0, layer(0, 0, 0)});
  EXPECT_OK([&](std::string& e) {
    return av2enc::svc::validate_frame_references(plan, policy, cross_unit,
                                                  order, e);
  });

  std::vector<av2enc::svc::FrameReference> future_unit;
  future_unit.push_back(
      av2enc::svc::FrameReference{0, layer(0, 0, 1), 1, layer(0, 0, 0)});
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::validate_frame_references(plan, policy, future_unit,
                                                      order, e);
      },
      "future temporal unit");

  // A schedule that emits the reference after its target must be rejected.
  std::vector<Emission> reversed;
  Emission td;
  td.kind = EmissionKind::kTemporalDelimiter;
  td.layer.xlayer_id = 31;
  td.temporal_unit = 0;
  reversed.push_back(td);
  Emission target;
  target.kind = EmissionKind::kFrame;
  target.layer = layer(0, 0, 1);
  target.temporal_unit = 0;
  reversed.push_back(target);
  Emission reference;
  reference.kind = EmissionKind::kFrame;
  reference.layer = layer(0, 0, 0);
  reference.temporal_unit = 0;
  reversed.push_back(reference);
  std::vector<av2enc::svc::FrameReference> late_ref;
  late_ref.push_back(av2enc::svc::FrameReference{0, layer(0, 0, 1), 0,
                                                 layer(0, 0, 0)});
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::validate_frame_references(plan, policy, late_ref,
                                                      reversed, e);
      },
      "emitted after or with");
}

void test_rejections() {
  LayerPlan plan;
  std::string error;

  LayerConfig empty;
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::build_layer_plan(empty, plan, e);
      },
      "at least one spatial layer");

  LayerConfig not_zero = make_config(2, 1, 1);
  not_zero.spatial_layers[0].xlayer_id = 1;
  not_zero.spatial_layers[1].xlayer_id = 2;
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::build_layer_plan(not_zero, plan, e);
      },
      "contiguous and ascending");

  LayerConfig gap = make_config(2, 1, 1);
  gap.spatial_layers[1].xlayer_id = 2;
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::build_layer_plan(gap, plan, e);
      },
      "must use xlayer_id 1");

  LayerConfig duplicate = make_config(2, 1, 1);
  duplicate.spatial_layers[1].xlayer_id = 0;
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::build_layer_plan(duplicate, plan, e);
      },
      "must use xlayer_id 1");

  LayerConfig shrinks = make_config(2, 1, 1);
  shrinks.spatial_layers[1].width = 32;
  shrinks.spatial_layers[1].height = 32;
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::build_layer_plan(shrinks, plan, e);
      },
      "non-decreasing");

  LayerConfig zero = make_config(1, 1, 1);
  zero.spatial_layers[0].width = 0;
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::build_layer_plan(zero, plan, e);
      },
      "positive coded dimensions");

  LayerConfig no_temporal = make_config(2, 1, 1);
  no_temporal.temporal_layers = 0;
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::build_layer_plan(no_temporal, plan, e);
      },
      "temporal_layers");

  LayerConfig many_temporal = make_config(2, 1, 1);
  many_temporal.temporal_layers = 5;
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::build_layer_plan(many_temporal, plan, e);
      },
      "temporal_layers");

  LayerConfig no_embedded = make_config(2, 1, 1);
  no_embedded.embedded_layers = 0;
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::build_layer_plan(no_embedded, plan, e);
      },
      "embedded_layers");

  LayerConfig many_embedded = make_config(2, 1, 1);
  many_embedded.embedded_layers = 9;
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::build_layer_plan(many_embedded, plan, e);
      },
      "embedded_layers");

  LayerConfig still_multi = make_config(2, 1, 1);
  still_multi.single_picture_header = true;
  still_multi.still_picture = true;
  still_multi.seq_lcr_id = 0;
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::build_layer_plan(still_multi, plan, e);
      },
      "exactly one layer");

  LayerConfig still_lcr = make_config(1, 1, 1);
  still_lcr.single_picture_header = true;
  still_lcr.still_picture = true;
  still_lcr.seq_lcr_id = 1;
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::build_layer_plan(still_lcr, plan, e);
      },
      "seq_lcr_id = 0");

  LayerConfig still_only = make_config(1, 1, 1);
  still_only.still_picture = true;
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::build_layer_plan(still_only, plan, e);
      },
      "without single_picture_header");

  LayerConfig no_lcr = make_config(1, 1, 1);
  no_lcr.seq_lcr_id = 0;
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::build_layer_plan(no_lcr, plan, e);
      },
      "seq_lcr_id must be 1..7");

  LayerConfig many = make_config(8, 1, 1);
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::build_layer_plan(many, plan, e);
      },
      "ops_cnt");

  LayerConfig custom = make_config(2, 2, 1);
  custom.dependencies = av2enc::tiles::svc::default_dependency_maps(0, 1);
  custom.dependencies->tlayer[0][1][0] = false;
  LayerPlan custom_plan;
  EXPECT_OK([&](std::string& e) {
    return av2enc::svc::build_layer_plan(custom, custom_plan, e);
  });
  CHECK(!custom_plan.sequences[0].dependencies.tlayer[0][1][0]);
  CHECK(custom_plan.sequences[1].dependencies.tlayer[0][1][0] == false);

  LayerConfig bad_maps = make_config(2, 2, 1);
  bad_maps.dependencies = av2enc::tiles::svc::default_dependency_maps(0, 1);
  bad_maps.dependencies->mlayer[0][0] = false;
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::build_layer_plan(bad_maps, plan, e);
      },
      "custom dependency maps");

  LayerPlan good = make_plan(2, 2, 1);
  LayerPlan corrupt = good;
  corrupt.lcr.xlayer_map = 0;
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::validate_layer_plan(corrupt, e);
      },
      "does not describe every spatial layer");

  corrupt = good;
  corrupt.sequences[0].max_width = 32;
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::validate_layer_plan(corrupt, e);
      },
      "disagree with the config");

  corrupt = good;
  corrupt.lcr.layers.pop_back();
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::validate_layer_plan(corrupt, e);
      },
      "one entry per spatial layer");

  corrupt = good;
  corrupt.sequences[1].dependencies.tlayer[0][1][0] = false;
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::validate_layer_plan(corrupt, e);
      },
      "disagree on dependency maps");

  corrupt = good;
  corrupt.ops.points[0].xlayer_map = 0x3U;
  corrupt.ops.points[0].mlayer_map[1] = 0x1U;
  corrupt.ops.points[0].tlayer_map[1][0] = 0x1U;
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::validate_layer_plan(corrupt, e);
      },
      "not cumulative");
}

void test_framing() {
  std::vector<std::uint8_t> bytes;
  std::string error;

  ObuHeader td;
  td.type = ObuType::kTemporalDelimiter;
  td.xlayer_id = 31;
  CHECK(av2enc::svc::encode_obu_header(td, bytes, error));
  CHECK(bytes.size() == 1 && bytes[0] == 0x08U);

  ObuHeader sequence;
  sequence.type = ObuType::kSequenceHeader;
  CHECK(av2enc::svc::encode_obu_header(sequence, bytes, error));
  CHECK(bytes.size() == 1 && bytes[0] == 0x04U);

  ObuHeader lcr;
  lcr.type = ObuType::kLayerConfigRecord;
  lcr.xlayer_id = 31;
  CHECK(av2enc::svc::encode_obu_header(lcr, bytes, error));
  CHECK(bytes.size() == 2 && bytes[0] == 0xC0U && bytes[1] == 0x1FU);

  ObuHeader frame;
  frame.type = ObuType::kRegularTileGroup;
  frame.xlayer_id = 2;
  frame.tlayer_id = 1;
  CHECK(av2enc::svc::encode_obu_header(frame, bytes, error));
  CHECK(bytes.size() == 2 && bytes[0] == 0x9DU && bytes[1] == 0x02U);

  ObuHeader decoded;
  std::size_t consumed = 0;
  CHECK(av2enc::svc::decode_obu_header(
      std::span<const std::uint8_t>(bytes), decoded, consumed, error));
  CHECK(consumed == 2);
  CHECK(decoded.type == ObuType::kRegularTileGroup);
  CHECK(decoded.tlayer_id == 1 && decoded.mlayer_id == 0 &&
        decoded.xlayer_id == 2);

  ObuHeader bad_td = td;
  bad_td.xlayer_id = 0;
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::encode_obu_header(bad_td, bytes, e);
      },
      "GLOBAL_XLAYER_ID");
  ObuHeader bad_m = td;
  bad_m.mlayer_id = 1;
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::encode_obu_header(bad_m, bytes, e);
      },
      "obu_mlayer_id = 0");

  const std::uint8_t empty_header[1] = {0};
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::decode_obu_header(
            std::span<const std::uint8_t>(), decoded, consumed, e);
      },
      "truncated OBU header");
  const std::uint8_t truncated[1] = {0x80U};
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::decode_obu_header(
            std::span<const std::uint8_t>(truncated), decoded, consumed, e);
      },
      "truncated OBU header extension");
  (void)empty_header;

  CHECK(av2enc::svc::leb128_size(0) == 1);
  CHECK(av2enc::svc::leb128_size(127) == 1);
  CHECK(av2enc::svc::leb128_size(128) == 2);
  CHECK(av2enc::svc::leb128_size(16383) == 2);
  CHECK(av2enc::svc::leb128_size(16384) == 3);
  std::vector<std::uint8_t> leb;
  av2enc::svc::append_leb128(leb, 300);
  CHECK(leb.size() == 2 && leb[0] == 0xACU && leb[1] == 0x02U);

  std::vector<std::uint8_t> stream;
  const std::uint8_t payload[2] = {0x11, 0x22};
  CHECK(av2enc::svc::frame_obu(frame, payload, stream, error));
  std::vector<av2enc::svc::ParsedObu> parsed;
  CHECK(av2enc::svc::parse_obus(stream, parsed, error));
  CHECK(parsed.size() == 1);
  CHECK(parsed[0].header == frame);
  CHECK(parsed[0].payload.size() == 2 && parsed[0].payload[0] == 0x11);

  const std::uint8_t zero_size[1] = {0x00U};
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::parse_obus(
            std::span<const std::uint8_t>(zero_size), parsed, e);
      },
      "size is zero");
  const std::uint8_t truncated_size[2] = {0x80U, 0x80U};
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::parse_obus(
            std::span<const std::uint8_t>(truncated_size), parsed, e);
      },
      "truncated leb128");
  const std::uint8_t oversized[2] = {0x05U, 0x08U};
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::parse_obus(
            std::span<const std::uint8_t>(oversized), parsed, e);
      },
      "exceeds the remaining");
  const std::uint8_t too_large[5] = {0xFFU, 0xFFU, 0xFFU, 0xFFU, 0x7FU};
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::parse_obus(
            std::span<const std::uint8_t>(too_large), parsed, e);
      },
      "exceeds 32 bits");
}

void expect_xlayer_info(BitReader& reader, std::uint8_t mlayer_map,
                        std::uint8_t tlayer_map) {
  CHECK(reader.bit() == false);  // rep info
  CHECK(reader.bit() == false);  // purpose
  CHECK(reader.bit() == false);  // color
  CHECK(reader.bit() == true);   // embedded info
  reader.align();
  CHECK(reader.bits(8) == mlayer_map);
  for (int m = 0; m < 8; ++m) {
    if ((mlayer_map & (1U << static_cast<unsigned>(m))) == 0) {
      continue;
    }
    CHECK(reader.bits(4) == tlayer_map);
    CHECK(reader.bits(8) == 0);  // TEXTURE_LAYER
    CHECK(reader.bits(8) == 1);  // VIEW_CENTER
    if (m > 0) {
      for (int ref = 0; ref < m; ++ref) {
        CHECK(reader.bit() == true);  // default mlayer dependencies
      }
    }
    CHECK(reader.bit() == true);  // same_sh_max_resolution_flag
    reader.align();
  }
  CHECK(reader.ok());
}

void test_lcr_payload() {
  LayerPlan plan = make_plan(2, 2, 1);
  std::vector<std::uint8_t> payload;
  std::string error;
  CHECK(av2enc::svc::write_layer_config_record_payload(
      plan.lcr, plan.sequences.front().dependencies, payload, error));
  CHECK(!payload.empty());
  CHECK(payload.back() == 0x40U);  // extension flag 0 + trailing bits

  BitReader reader{std::span<const std::uint8_t>(payload)};
  CHECK(reader.bits(3) == 1);       // lcr_global_config_record_id
  CHECK(reader.bits(31) == 0x3U);   // lcr_xlayer_map
  CHECK(reader.bit() == false);     // aggregate info
  CHECK(reader.bit() == false);     // ptl info
  CHECK(reader.bit() == true);      // global payload
  CHECK(reader.bit() == false);     // dependent xlayers
  CHECK(reader.bit() == false);     // atlas id
  CHECK(reader.bits(7) == 0);       // purpose
  // §7.3.7: the 2-temporal-layer plan clears monotonic_output_order_flag, so
  // the global LCR must set lcr_doh_constraint_flag.
  CHECK(reader.bit() == true);      // doh constraint
  CHECK(reader.bit() == false);     // tile alignment
  CHECK(reader.bits(3) == 0);       // reserved
  CHECK(reader.bits(5) == 0);       // reserved

  for (int x = 0; x < 2; ++x) {
    const std::uint32_t size = reader.leb128();
    const std::size_t start = reader.byte_position();
    expect_xlayer_info(reader, 0x1U, 0x3U);
    CHECK(reader.byte_position() - start == size);
  }
  CHECK(reader.ok());

  // Local LCR form.
  av2enc::tiles::svc::LayerConfigRecord local = plan.lcr;
  local.is_global = false;
  local.xlayer_id = 1;
  local.global_config_record_id = 2;
  local.xlayer_map = 0;
  local.layers.clear();
  local.layers.push_back(plan.lcr.layers[1]);
  CHECK(av2enc::svc::write_layer_config_record_payload(
      local, plan.sequences.front().dependencies, payload, error));
  BitReader local_reader{std::span<const std::uint8_t>(payload)};
  CHECK(local_reader.bits(3) == 0);  // lcr_global_id
  CHECK(local_reader.bits(3) == 2);  // lcr_local_id
  CHECK(local_reader.bit() == false);
  CHECK(local_reader.bit() == false);
  CHECK(local_reader.bits(3) == 0);
  CHECK(local_reader.bits(5) == 0);
  expect_xlayer_info(local_reader, 0x1U, 0x3U);
  CHECK(local_reader.ok());

  // A flat (single temporal layer) plan keeps monotonic_output_order_flag = 1
  // and therefore must keep lcr_doh_constraint_flag = 0.
  LayerPlan flat = make_plan(1, 1, 1);
  CHECK(!flat.lcr.doh_constraint);
  CHECK(av2enc::svc::write_layer_config_record_payload(
      flat.lcr, flat.sequences.front().dependencies, payload, error));
  BitReader flat_reader{std::span<const std::uint8_t>(payload)};
  CHECK(flat_reader.bits(3) == 1);
  CHECK(flat_reader.bits(31) == 0x1U);
  CHECK(flat_reader.bit() == false);  // aggregate info
  CHECK(flat_reader.bit() == false);  // ptl info
  CHECK(flat_reader.bit() == true);   // global payload
  CHECK(flat_reader.bit() == false);  // dependent xlayers
  CHECK(flat_reader.bit() == false);  // atlas id
  CHECK(flat_reader.bits(7) == 0);    // purpose
  CHECK(flat_reader.bit() == false);  // doh constraint
  CHECK(flat_reader.ok());

  // The plan validator requires the flag whenever the GOP clears
  // monotonic_output_order_flag.
  LayerPlan broken = make_plan(1, 2, 1);
  broken.lcr.doh_constraint = false;
  error.clear();
  CHECK(!av2enc::svc::validate_layer_plan(broken, error));
  CHECK(error.find("lcr_doh_constraint_flag") != std::string::npos);
}

void test_ops_payload() {
  LayerPlan plan = make_plan(2, 2, 1);
  std::vector<std::uint8_t> payload;
  std::string error;
  CHECK(av2enc::svc::write_operating_point_set_payload(plan.ops, payload, error));
  CHECK(!payload.empty());
  CHECK(payload.back() == 0x40U);

  BitReader reader{std::span<const std::uint8_t>(payload)};
  CHECK(reader.bit() == true);   // ops_reset_flag
  CHECK(reader.bits(4) == 0);    // ops_id
  CHECK(reader.bits(3) == 2);    // ops_cnt
  CHECK(reader.bits(4) == 0);    // priority
  CHECK(reader.bits(7) == 0);    // intent
  CHECK(reader.bit() == false);  // intent present
  CHECK(reader.bit() == false);  // ptl present
  CHECK(reader.bit() == false);  // color present
  CHECK(reader.bits(2) == 1);    // ops_mlayer_info_idc

  for (int op = 0; op < 2; ++op) {
    const std::uint32_t size = reader.leb128();
    const std::size_t start = reader.byte_position();
    CHECK(reader.bit() == false);  // decoder model
    CHECK(reader.bit() == false);  // initial display delay
    CHECK(reader.bits(31) == (op == 0 ? 0x1U : 0x3U));
    for (int x = 0; x <= op; ++x) {
      CHECK(reader.bits(8) == 0x1U);
      CHECK(reader.bits(4) == 0x3U);
    }
    reader.align();
    CHECK(reader.byte_position() - start == size);
  }
  CHECK(reader.ok());

  // Local OPS form.
  av2enc::tiles::svc::OperatingPointSet local;
  local.xlayer_id = 0;
  local.ops_id = 3;
  av2enc::tiles::svc::OperatingPoint point;
  point.op_index = 0;
  point.xlayer_map = 0x1U;
  point.mlayer_map[0] = 0x1U;
  point.tlayer_map[0][0] = 0x1U;
  local.points.push_back(point);
  CHECK(av2enc::svc::write_operating_point_set_payload(local, payload, error));
  BitReader local_reader{std::span<const std::uint8_t>(payload)};
  CHECK(local_reader.bit() == true);
  CHECK(local_reader.bits(4) == 3);
  CHECK(local_reader.bits(3) == 1);
  CHECK(local_reader.bits(4) == 0);
  CHECK(local_reader.bits(7) == 0);
  CHECK(local_reader.bit() == false);
  CHECK(local_reader.bit() == false);
  CHECK(local_reader.bit() == false);
  CHECK(local_reader.bits(2) == 0);
  const std::uint32_t size = local_reader.leb128();
  const std::size_t start = local_reader.byte_position();
  CHECK(local_reader.bit() == false);
  CHECK(local_reader.bit() == false);
  CHECK(local_reader.bits(8) == 0x1U);
  CHECK(local_reader.bits(4) == 0x1U);
  local_reader.align();
  CHECK(local_reader.byte_position() - start == size);
  CHECK(local_reader.ok());
}

void test_compose_and_extract() {
  LayerPlan plan = make_plan(2, 1, 1);
  const std::vector<std::uint8_t> seq0 = sequence_payload_prefix(0);
  const std::vector<std::uint8_t> seq1 = sequence_payload_prefix(0);
  std::vector<LayerInput> inputs;
  inputs.push_back(LayerInput{layer(0, 0, 0), single_layer_unit(0x10, seq0)});
  inputs.push_back(LayerInput{layer(1, 0, 0), single_layer_unit(0x20, seq1)});

  ComposeOptions options;
  std::vector<std::uint8_t> stream;
  std::string error;
  CHECK(av2enc::svc::compose_layered_stream(plan, inputs, options, stream,
                                            error));
  CHECK(!stream.empty());

  std::vector<av2enc::svc::ParsedObu> obus;
  CHECK(av2enc::svc::parse_obus(stream, obus, error));
  CHECK(obus.size() == 7);  // TD + LCR + OPS + 2x(SEQ + CLK)
  CHECK(obus[0].header.type == ObuType::kTemporalDelimiter);
  CHECK(obus[0].header.xlayer_id == 31);
  CHECK(obus[1].header.type == ObuType::kLayerConfigRecord);
  CHECK(obus[2].header.type == ObuType::kOperatingPointSet);
  CHECK(obus[3].header.type == ObuType::kSequenceHeader);
  CHECK(obus[3].header.xlayer_id == 0);
  CHECK(obus[4].header.type == ObuType::kClosedLoopKey);
  CHECK(obus[5].header.type == ObuType::kSequenceHeader);
  CHECK(obus[5].header.xlayer_id == 1);
  CHECK(obus[6].header.type == ObuType::kClosedLoopKey);
  CHECK(obus[6].header.xlayer_id == 1);
  CHECK(obus[4].payload[0] == 0x10U);
  CHECK(obus[6].payload[0] == 0x20U);

  // B7: the composed sequence headers carry the plan's seq_lcr_id (1), so the
  // global LCR association is not dangling.
  const std::vector<std::uint8_t> expected_seq = sequence_payload_prefix(1);
  CHECK(std::vector<std::uint8_t>(obus[3].payload.begin(),
                                  obus[3].payload.end()) == expected_seq);
  CHECK(std::vector<std::uint8_t>(obus[5].payload.begin(),
                                  obus[5].payload.end()) == expected_seq);

  std::vector<std::uint8_t> repeat;
  CHECK(av2enc::svc::compose_layered_stream(plan, inputs, options, repeat,
                                            error));
  CHECK(stream == repeat);

  RetentionMap retention;
  LayerSelection all;
  EXPECT_OK([&](std::string& e) {
    return av2enc::svc::resolve_selection(plan, all, retention, e);
  });
  std::vector<std::uint8_t> extracted;
  CHECK(av2enc::svc::extract_substream(stream, retention, extracted, error));
  CHECK(extracted == stream);

  LayerSelection op0;
  op0.mode = SelectionMode::kOperatingPoint;
  op0.op_index = 0;
  EXPECT_OK([&](std::string& e) {
    return av2enc::svc::resolve_selection(plan, op0, retention, e);
  });
  CHECK(av2enc::svc::extract_substream(stream, retention, extracted, error));
  CHECK(av2enc::svc::parse_obus(extracted, obus, error));
  CHECK(obus.size() == 5);  // TD + LCR + OPS + SEQ0 + CLK0
  for (const auto& obu : obus) {
    CHECK(obu.header.xlayer_id == 31 || obu.header.xlayer_id == 0);
  }

  LayerSelection op1;
  op1.mode = SelectionMode::kOperatingPoint;
  op1.op_index = 1;
  EXPECT_OK([&](std::string& e) {
    return av2enc::svc::resolve_selection(plan, op1, retention, e);
  });
  CHECK(av2enc::svc::extract_substream(stream, retention, extracted, error));
  CHECK(extracted == stream);

  // Keep only x1 (valid: xlayers are independent in the spatial-only path).
  RetentionMap only_x1;
  only_x1.retain[1][0][0] = true;
  only_x1.xlayer_selected[1] = true;
  only_x1.retain[31][0][0] = true;
  only_x1.xlayer_selected[31] = true;
  EXPECT_OK([&](std::string& e) {
    return av2enc::svc::validate_retention_closed(plan, only_x1, e);
  });
  CHECK(av2enc::svc::extract_substream(stream, only_x1, extracted, error));
  CHECK(av2enc::svc::parse_obus(extracted, obus, error));
  CHECK(obus.size() == 5);
  CHECK(obus[3].header.xlayer_id == 1);

  ComposeOptions no_metadata;
  no_metadata.emit_lcr = false;
  no_metadata.emit_ops = false;
  CHECK(av2enc::svc::compose_layered_stream(plan, inputs, no_metadata, stream,
                                            error));
  CHECK(av2enc::svc::parse_obus(stream, obus, error));
  CHECK(obus.size() == 5);
  CHECK(obus[0].header.type == ObuType::kTemporalDelimiter);
  CHECK(obus[1].header.type == ObuType::kSequenceHeader);
  // emit_lcr = false: no association is claimed, so the sequence headers keep
  // their encoded/inferred seq_lcr_id (0).
  const std::vector<std::uint8_t> unassociated_seq = sequence_payload_prefix(0);
  CHECK(std::vector<std::uint8_t>(obus[1].payload.begin(),
                                  obus[1].payload.end()) ==
        unassociated_seq);
  CHECK(std::vector<std::uint8_t>(obus[3].payload.begin(),
                                  obus[3].payload.end()) ==
        unassociated_seq);

  // A sequence header that already carries a different LCR id is rewritten to
  // the plan's association rather than left dangling.
  std::vector<LayerInput> remapped;
  remapped.push_back(
      LayerInput{layer(0), single_layer_unit(0x40, sequence_payload_prefix(3))});
  remapped.push_back(
      LayerInput{layer(1), single_layer_unit(0x50, sequence_payload_prefix(5))});
  CHECK(av2enc::svc::compose_layered_stream(plan, remapped, options, stream,
                                            error));
  CHECK(av2enc::svc::parse_obus(stream, obus, error));
  CHECK(obus.size() == 7);  // TD + LCR + OPS + 2x(SEQ + CLK)
  CHECK(std::vector<std::uint8_t>(obus[3].payload.begin(),
                                  obus[3].payload.end()) == expected_seq);
  CHECK(std::vector<std::uint8_t>(obus[5].payload.begin(),
                                  obus[5].payload.end()) == expected_seq);
}

void test_compose_rejections() {
  LayerPlan plan = make_plan(2, 1, 1);
  ComposeOptions options;
  std::vector<std::uint8_t> stream;
  std::string error;

  const std::vector<std::uint8_t> seq = sequence_payload_prefix(0);
  std::vector<LayerInput> wrong_count;
  wrong_count.push_back(LayerInput{layer(0), single_layer_unit(1, seq)});
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::compose_layered_stream(plan, wrong_count, options,
                                                   stream, e);
      },
      "one single-layer input");

  std::vector<LayerInput> wrong_layer;
  wrong_layer.push_back(LayerInput{layer(1), single_layer_unit(1, seq)});
  wrong_layer.push_back(LayerInput{layer(1), single_layer_unit(2, seq)});
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::compose_layered_stream(plan, wrong_layer, options,
                                                   stream, e);
      },
      "must be layer x0/m0/t0");

  std::vector<std::uint8_t> no_sequence;
  ObuHeader td;
  td.type = ObuType::kTemporalDelimiter;
  td.xlayer_id = 31;
  CHECK(av2enc::svc::frame_obu(td, {}, no_sequence, error));
  ObuHeader frame;
  frame.type = ObuType::kClosedLoopKey;
  CHECK(av2enc::svc::frame_obu(frame, seq, no_sequence, error));
  std::vector<LayerInput> missing_seq;
  missing_seq.push_back(LayerInput{layer(0), no_sequence});
  missing_seq.push_back(LayerInput{layer(1), single_layer_unit(2, seq)});
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::compose_layered_stream(plan, missing_seq, options,
                                                   stream, e);
      },
      "exactly one sequence header");

  std::vector<std::uint8_t> two_frames;
  CHECK(av2enc::svc::frame_obu(td, {}, two_frames, error));
  ObuHeader sequence;
  sequence.type = ObuType::kSequenceHeader;
  CHECK(av2enc::svc::frame_obu(sequence, seq, two_frames, error));
  CHECK(av2enc::svc::frame_obu(frame, seq, two_frames, error));
  CHECK(av2enc::svc::frame_obu(frame, seq, two_frames, error));
  std::vector<LayerInput> multi;
  multi.push_back(LayerInput{layer(0), two_frames});
  multi.push_back(LayerInput{layer(1), single_layer_unit(2, seq)});
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::compose_layered_stream(plan, multi, options, stream,
                                                   e);
      },
      "exactly one coded frame");

  std::vector<std::uint8_t> with_lcr;
  CHECK(av2enc::svc::frame_obu(td, {}, with_lcr, error));
  ObuHeader lcr;
  lcr.type = ObuType::kLayerConfigRecord;
  lcr.xlayer_id = 31;
  CHECK(av2enc::svc::frame_obu(lcr, {}, with_lcr, error));
  CHECK(av2enc::svc::frame_obu(sequence, seq, with_lcr, error));
  CHECK(av2enc::svc::frame_obu(frame, seq, with_lcr, error));
  std::vector<LayerInput> pre_metadata;
  pre_metadata.push_back(LayerInput{layer(0), with_lcr});
  pre_metadata.push_back(LayerInput{layer(1), single_layer_unit(2, seq)});
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::compose_layered_stream(plan, pre_metadata, options,
                                                   stream, e);
      },
      "already carries global layer metadata");

  // B7: a single-picture sequence header infers seq_lcr_id = 0 and cannot be
  // associated with the plan's LCR. Reject instead of emitting it unchanged.
  std::vector<LayerInput> single_picture;
  single_picture.push_back(LayerInput{
      layer(0), single_layer_unit(3, sequence_payload_prefix(0, true))});
  single_picture.push_back(
      LayerInput{layer(1), single_layer_unit(4, sequence_payload_prefix(0))});
  EXPECT_FAILS(
      [&](std::string& e) {
        return av2enc::svc::compose_layered_stream(plan, single_picture,
                                                   options, stream, e);
      },
      "forces seq_lcr_id = 0");
}

void test_three_layer_compose() {
  LayerPlan plan = make_plan(3, 1, 1);
  const std::vector<std::uint8_t> seq = sequence_payload_prefix(0);
  std::vector<LayerInput> inputs;
  inputs.push_back(LayerInput{layer(0), single_layer_unit(0x11, seq)});
  inputs.push_back(LayerInput{layer(1), single_layer_unit(0x22, seq)});
  inputs.push_back(LayerInput{layer(2), single_layer_unit(0x33, seq)});
  ComposeOptions options;
  std::vector<std::uint8_t> stream;
  std::string error;
  CHECK(av2enc::svc::compose_layered_stream(plan, inputs, options, stream,
                                            error));
  std::vector<av2enc::svc::ParsedObu> obus;
  CHECK(av2enc::svc::parse_obus(stream, obus, error));
  CHECK(obus.size() == 9);  // TD+LCR+OPS+3x(SEQ+CLK)
  const std::vector<std::uint8_t> expected_seq = sequence_payload_prefix(1);
  for (const auto& obu : obus) {
    if (obu.header.type == ObuType::kSequenceHeader) {
      CHECK(std::vector<std::uint8_t>(obu.payload.begin(),
                                      obu.payload.end()) == expected_seq);
    }
  }
  int seen_x = -1;
  for (const auto& obu : obus) {
    if (obu.header.type == ObuType::kClosedLoopKey) {
      CHECK(static_cast<int>(obu.header.xlayer_id) == seen_x + 1);
      seen_x = obu.header.xlayer_id;
    }
  }
  CHECK(seen_x == 2);

  RetentionMap retention;
  LayerSelection op1;
  op1.mode = SelectionMode::kOperatingPoint;
  op1.op_index = 1;
  EXPECT_OK([&](std::string& e) {
    return av2enc::svc::resolve_selection(plan, op1, retention, e);
  });
  std::vector<std::uint8_t> extracted;
  CHECK(av2enc::svc::extract_substream(stream, retention, extracted, error));
  CHECK(av2enc::svc::parse_obus(extracted, obus, error));
  CHECK(obus.size() == 7);  // global 3 + x0 pair + x1 pair
  bool saw_x2 = false;
  for (const auto& obu : obus) {
    saw_x2 = saw_x2 || obu.header.xlayer_id == 2;
  }
  CHECK(!saw_x2);
}

void test_single_layer_still_plan() {
  LayerConfig config;
  SpatialLayerSpec layer_spec;
  layer_spec.xlayer_id = 0;
  layer_spec.width = 64;
  layer_spec.height = 64;
  config.spatial_layers.push_back(layer_spec);
  config.single_picture_header = true;
  config.still_picture = true;
  config.temporal_layers = 1;
  config.embedded_layers = 1;
  config.seq_lcr_id = 0;

  LayerPlan plan;
  std::string error;
  CHECK(av2enc::svc::build_layer_plan(config, plan, error));
  CHECK(plan.sequences[0].single_picture_header);
  CHECK(plan.sequences[0].seq_lcr_id == 0);
  CHECK(plan.layers.size() == 1);
}

}  // namespace

int main() {
  test_two_layer_plan();
  test_three_layer_plan();
  test_embedded_layers();
  test_operating_point_extraction();
  test_retention_closure();
  test_retention_expansion();
  test_assignment();
  test_emission_order();
  test_reference_policy();
  test_rejections();
  test_framing();
  test_lcr_payload();
  test_ops_payload();
  test_compose_and_extract();
  test_compose_rejections();
  test_three_layer_compose();
  test_single_layer_still_plan();

  if (g_failures != 0) {
    std::cerr << g_failures << " check(s) failed\n";
    return EXIT_FAILURE;
  }
  std::cout << "svc pipeline tests: all checks passed\n";
  return EXIT_SUCCESS;
}

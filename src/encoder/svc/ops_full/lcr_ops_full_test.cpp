// Unit tests for the field-complete LCR (§5.8) and OPS (§5.10/§5.11) writers,
// the readers, the §6.8/§6.10/§6.11 conformance checks, and the byte-for-byte
// overlap with `svc_framing`.
//
// The pinned goldens below were derived by hand from the PDF syntax (an
// independent Python bit-packer, reproduced in README.md), not by running this
// module. They pin field order, descriptor encodings (f/uvlc/rg/leb128),
// `byte_alignment()` positions and the extensible-OBU tail.
//
// No decoder and no CUDA here; `gate_avmdec.sh` is the black-box gate.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "lcr_ops_full.h"
#include "svc_config.h"
#include "svc_framing.h"

namespace {

using namespace av2enc::svc::ops_full;
namespace legacy = av2enc::svc;
namespace layers = av2enc::tiles::svc;
using av2enc::BitWriter;

int g_failures = 0;

void check(bool condition, const char* expression, const char* file, int line) {
  if (!condition) {
    ++g_failures;
    std::cerr << file << ':' << line << ": CHECK failed: " << expression << '\n';
  }
}

#define CHECK(expression) ::check((expression), #expression, __FILE__, __LINE__)

int hex_digit(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

std::vector<std::uint8_t> unhex(const char* hex) {
  std::vector<std::uint8_t> out;
  int high = -1;
  for (const char* p = hex; *p != '\0'; ++p) {
    if (*p == ' ' || *p == '\n') continue;
    const int digit = hex_digit(*p);
    if (digit < 0) {
      std::cerr << "bad hex digit in golden\n";
      std::exit(2);
    }
    if (high < 0) {
      high = digit;
    } else {
      out.push_back(static_cast<std::uint8_t>((high << 4) | digit));
      high = -1;
    }
  }
  if (high >= 0) {
    std::cerr << "odd hex golden\n";
    std::exit(2);
  }
  return out;
}

std::string to_hex(std::span<const std::uint8_t> bytes) {
  static const char* digits = "0123456789abcdef";
  std::string out;
  for (std::uint8_t byte : bytes) {
    out.push_back(digits[byte >> 4U]);
    out.push_back(digits[byte & 0x0FU]);
  }
  return out;
}

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

void check_report_has(const ValidationReport& report, const char* substring,
                      const char* file, int line) {
  for (const std::string& error : report.errors) {
    if (error.find(substring) != std::string::npos) return;
  }
  ++g_failures;
  std::cerr << file << ':' << line << ": no validation error containing '"
            << substring << "'; got " << report.errors.size()
            << " error(s):\n";
  for (const std::string& error : report.errors) {
    std::cerr << "  " << error << '\n';
  }
}

#define EXPECT_REPORT(report, substring) \
  ::check_report_has((report), (substring), __FILE__, __LINE__)

// ---------------------------------------------------------------------------
// Builders
// ---------------------------------------------------------------------------

OpsProfileTierLevelInfo ops_ptl(std::uint8_t profile, std::uint8_t level,
                                bool tier, std::uint8_t mlayers) {
  OpsProfileTierLevelInfo info;
  info.seq_profile_idc = profile;
  info.level_idx = level;
  info.tier_flag = tier;
  info.mlayer_count = mlayers;
  return info;
}

ProfileTierLevelInfo ptl(std::uint8_t profile, std::uint8_t level, bool tier,
                         std::uint8_t mlayers) {
  ProfileTierLevelInfo info;
  info.seq_profile_idc = profile;
  info.max_level_idx = level;
  info.tier_flag = tier;
  info.max_mlayer_count = mlayers;
  return info;
}

EmbeddedLayerInfo embedded_texture(std::uint8_t mlayer,
                                   std::uint8_t tlayer_map) {
  EmbeddedLayerInfo info;
  info.mlayer_id = mlayer;
  info.tlayer_map = tlayer_map;
  info.layer_type = kLayerTexture;
  info.view_type = kViewCenter;
  info.same_sh_max_resolution = true;
  return info;
}

// The §5.8.2-§5.8.9 example pinned by the "local LCR full" golden.
LayerConfigRecord local_full_record() {
  LayerConfigRecord record;
  record.is_global = false;
  LocalLcrInfo& local = record.local;
  local.xlayer_id = 4;
  local.global_id = 1;
  local.local_id = 3;
  local.profile_tier_level_info_present = true;
  local.ptl = ptl(1, 9, false, 2);
  local.local_atlas_id_present = true;
  local.local_atlas_id = 5;

  XLayerInfo& layer = local.xlayer;
  layer.rep_info_present = true;
  layer.rep.max_pic_width = 64;
  layer.rep.max_pic_height = 48;
  layer.rep.format_info_present = true;
  layer.rep.bit_depth_idc = 1;
  layer.rep.chroma_format_idc = 0;
  layer.rep.cropping_window_present = true;
  layer.rep.crop_left = 2;
  layer.rep.crop_right = 3;
  layer.rep.crop_top = 4;
  layer.rep.crop_bottom = 5;
  layer.purpose_present = true;
  layer.purpose_id = 7;
  layer.color_info_present = true;
  layer.color.description_idc = 1;
  layer.color.full_range = true;
  layer.embedded_layer_info_present = true;
  layer.mlayer_map = 0x03U;

  EmbeddedLayerInfo m0 = embedded_texture(0, 0x03U);
  m0.atlas_segment_id = 9;
  m0.priority_order = 1;
  m0.rendering_method = 2;
  m0.view_type = kViewLeft;

  EmbeddedLayerInfo m1;
  m1.mlayer_id = 1;
  m1.tlayer_map = 0x01U;
  m1.atlas_segment_id = 17;
  m1.priority_order = 2;
  m1.rendering_method = 0;
  m1.layer_type = kLayerAux;
  m1.auxiliary_type = kAuxDepth;
  m1.view_type = kViewExplicit;
  m1.view_id = 6;
  m1.dependent_layer_mask = 1;
  m1.same_sh_max_resolution = false;
  m1.max_expected_width = 32;
  m1.max_expected_height = 24;

  layer.embedded = {m0, m1};
  return record;
}

// The §5.8.1/§5.8.5 example pinned by the "global LCR full" golden.
LayerConfigRecord global_full_record() {
  LayerConfigRecord record;
  record.is_global = true;
  GlobalLcrInfo& global = record.global;
  global.config_record_id = 2;
  global.xlayer_map = 0x05U;  // x0 and x2
  global.aggregate_info_present = true;
  global.aggregate.config_idc = 4;
  global.aggregate.level_idx = 10;
  global.aggregate.max_tier_flag = true;
  global.aggregate.max_interop = 3;
  global.profile_tier_level_info_present = true;
  global.ptl[0] = ptl(0, 10, true, 1);
  global.ptl[2] = ptl(0, 10, false, 1);
  global.global_payload_present = true;
  global.global_purpose_id = 2;
  global.doh_constraint_flag = true;
  global.enforce_tile_alignment_flag = true;
  for (std::uint8_t x : {std::uint8_t{0}, std::uint8_t{2}}) {
    GlobalPayload payload;
    payload.xlayer_id = x;
    payload.info.purpose_present = true;
    payload.info.purpose_id = 3;
    payload.info.color_info_present = true;
    payload.info.color.description_idc = 2;
    payload.info.embedded_layer_info_present = true;
    payload.info.mlayer_map = 0x01U;
    payload.info.embedded.push_back(embedded_texture(0, 0x01U));
    global.payloads.push_back(payload);
  }
  return record;
}

// The §5.10/§5.11 example pinned by the "global OPS full" golden.
OperatingPointSet ops_full_set() {
  OperatingPointSet set;
  set.xlayer_id = kGlobalXLayerId;
  set.ops_id = 0;
  set.reset_flag = true;
  set.priority = 1;
  set.intent = 1;  // OPSI_SCALABILITY
  set.intent_present = true;
  set.ptl_present = true;
  set.color_info_present = true;
  set.mlayer_info_idc = 2;
  for (int i = 0; i < 3; ++i) {
    OpsPoint point;
    point.op_intent = 1;
    point.aggregate.config_idc = 4;
    point.aggregate.level_idx = 10;
    point.aggregate.max_interop = 3;
    point.color.description_idc = 1;
    for (int j = 0; j < kXLayerCount; ++j) {
      point.ptl[static_cast<std::size_t>(j)] = ops_ptl(0, 10, false, 3);
    }
    point.decoder_model_info_present = (i == 0);
    point.decoder_model.decoder_buffer_delay = 80000;
    point.decoder_model.encoder_buffer_delay = 10000;
    point.initial_display_delay_present = true;
    point.initial_display_delay_minus_1 = static_cast<std::uint8_t>(i);
    point.xlayer_map = 0x03U;
    for (int j = 0; j < 2; ++j) {
      OpsLayerMlayerInfo& layer =
          point.layer_info[static_cast<std::size_t>(j)];
      layer.explicit_info = (i == 0 || j == 0);
      layer.mlayer_map = 0x01U;
      layer.tlayer_map[0] = 0x01U;
      layer.embedded_ops_id = 0;
      layer.embedded_op_index = 0;
    }
    set.points.push_back(point);
  }
  return set;
}

XLayerSequenceInfo sequence_info(std::uint8_t xlayer, std::uint32_t width,
                                 std::uint32_t height,
                                 std::uint8_t profile = 0,
                                 std::uint8_t level = 7, bool tier = false,
                                 std::uint8_t max_mlayers = 1) {
  XLayerSequenceInfo info;
  info.xlayer_id = xlayer;
  info.seq_profile_idc = profile;
  info.seq_level_idx = level;
  info.seq_tier = tier;
  info.seq_max_mlayer_cnt = max_mlayers;
  info.max_frame_width = width;
  info.max_frame_height = height;
  info.bit_depth_idc = 1;
  info.chroma_format_idc = 0;
  return info;
}

StreamContext two_layer_context() {
  StreamContext context;
  context.sequences.push_back(sequence_info(0, 64, 64));
  context.sequences.push_back(sequence_info(1, 128, 96));
  context.dependencies = layers::default_dependency_maps(1, 1);
  return context;
}

// ---------------------------------------------------------------------------
// Goldens
// ---------------------------------------------------------------------------

void test_substructure_goldens() {
  {
    BitWriter writer;
    std::string error;
    CHECK(write_lcr_seq_profile_tier_level_info(writer, ptl(2, 7, true, 3),
                                                error));
    CHECK(to_hex(writer.bytes()) == "11ec");
  }
  {
    BitWriter writer;
    std::string error;
    AggregateInfo aggregate;
    aggregate.config_idc = 3;
    aggregate.level_idx = 9;
    aggregate.max_tier_flag = false;
    aggregate.max_interop = 2;
    CHECK(write_lcr_aggregate_info(writer, aggregate, error));
    CHECK(to_hex(writer.bytes()) == "0d22");
  }
  {
    BitWriter writer;
    std::string error;
    ColorDescription color;
    color.description_idc = 1;  // BT.709 SDR
    color.full_range = true;
    CHECK(write_lcr_xlayer_color_info(writer, color, error));
    CHECK(to_hex(writer.bytes()) == "30");
  }
  {
    BitWriter writer;
    std::string error;
    ColorDescription color;
    color.description_idc = 0;
    color.primaries = 9;
    color.transfer = 16;
    color.matrix = 9;
    color.full_range = false;
    CHECK(write_lcr_xlayer_color_info(writer, color, error));
    CHECK(to_hex(writer.bytes()) == "01220120");
  }
  {
    BitWriter writer;
    std::string error;
    EmbeddedLayerInfo embedded;
    embedded.mlayer_id = 1;
    embedded.tlayer_map = 0x01U;
    embedded.atlas_segment_id = 17;
    embedded.priority_order = 2;
    embedded.rendering_method = 0;
    embedded.layer_type = kLayerAux;
    embedded.auxiliary_type = kAuxDepth;
    embedded.view_type = kViewExplicit;
    embedded.view_id = 6;
    embedded.dependent_layer_mask = 1;
    embedded.same_sh_max_resolution = false;
    embedded.max_expected_width = 32;
    embedded.max_expected_height = 24;
    CHECK(write_lcr_embedded_layer_info(writer, embedded, true, error));
    CHECK(to_hex(writer.bytes()) == "1110200010104068108640");
  }
  {
    BitWriter writer;
    std::string error;
    OpsDecoderModelInfo model;
    model.decoder_buffer_delay = 80000;
    model.encoder_buffer_delay = 10000;
    model.low_delay_mode_flag = false;
    CHECK(write_ops_decoder_model_info(writer, model, error));
    // uvlc(80000) = 16 leading zeros + 1 + 16 bits; uvlc(10000);
    // then low_delay_mode_flag = 0.
    CHECK(writer.bit_size() == 17U + 16U + 27U + 1U);
  }
}

void test_local_lcr_golden() {
  const LayerConfigRecord record = local_full_record();
  std::vector<std::uint8_t> payload;
  std::string error;
  CHECK(write_lcr_payload(record, payload, error));
  const std::vector<std::uint8_t> expected =
      unhex("2f0a48a0f020831d590a60e603309010200028111020001010406810864040");
  CHECK(payload == expected);
  CHECK(to_hex(payload) ==
        "2f0a48a0f020831d590a60e603309010200028111020001010406810864040");

  LayerConfigRecord parsed;
  EXPECT_OK([&](std::string& e) {
    return read_lcr_payload(payload, false, parsed, e);
  });
  CHECK(!parsed.is_global);
  CHECK(parsed.local.xlayer_id == 0);  // not coded; caller supplies it
  CHECK(parsed.local.global_id == 1);
  CHECK(parsed.local.local_id == 3);
  CHECK(parsed.local.ptl.seq_profile_idc == 1);
  CHECK(parsed.local.ptl.max_level_idx == 9);
  CHECK(!parsed.local.ptl.tier_flag);
  CHECK(parsed.local.ptl.max_mlayer_count == 2);
  CHECK(parsed.local.local_atlas_id_present);
  CHECK(parsed.local.local_atlas_id == 5);
  CHECK(parsed.local.xlayer.rep_info_present);
  CHECK(parsed.local.xlayer.rep.max_pic_width == 64);
  CHECK(parsed.local.xlayer.rep.max_pic_height == 48);
  CHECK(parsed.local.xlayer.rep.format_info_present);
  CHECK(parsed.local.xlayer.rep.bit_depth_idc == 1);
  CHECK(parsed.local.xlayer.rep.chroma_format_idc == 0);
  CHECK(parsed.local.xlayer.rep.cropping_window_present);
  CHECK(parsed.local.xlayer.rep.crop_left == 2);
  CHECK(parsed.local.xlayer.rep.crop_bottom == 5);
  CHECK(parsed.local.xlayer.purpose_present);
  CHECK(parsed.local.xlayer.purpose_id == 7);
  CHECK(parsed.local.xlayer.color_info_present);
  CHECK(parsed.local.xlayer.color.description_idc == 1);
  CHECK(parsed.local.xlayer.color.full_range);
  CHECK(parsed.local.xlayer.mlayer_map == 0x03U);
  CHECK(parsed.local.xlayer.embedded.size() == 2);
  CHECK(parsed.local.xlayer.embedded[0].atlas_segment_id == 9);
  CHECK(parsed.local.xlayer.embedded[0].priority_order == 1);
  CHECK(parsed.local.xlayer.embedded[0].rendering_method == 2);
  CHECK(parsed.local.xlayer.embedded[0].view_type == kViewLeft);
  CHECK(parsed.local.xlayer.embedded[1].layer_type == kLayerAux);
  CHECK(parsed.local.xlayer.embedded[1].auxiliary_type == kAuxDepth);
  CHECK(parsed.local.xlayer.embedded[1].view_type == kViewExplicit);
  CHECK(parsed.local.xlayer.embedded[1].view_id == 6);
  CHECK(parsed.local.xlayer.embedded[1].dependent_layer_mask == 1);
  CHECK(!parsed.local.xlayer.embedded[1].same_sh_max_resolution);
  CHECK(parsed.local.xlayer.embedded[1].max_expected_width == 32);
  CHECK(parsed.local.xlayer.embedded[1].max_expected_height == 24);

  std::vector<std::uint8_t> rewritten;
  EXPECT_OK([&](std::string& e) {
    return write_lcr_payload(parsed, rewritten, e);
  });
  CHECK(rewritten == payload);
}

void test_global_lcr_golden() {
  const LayerConfigRecord record = global_full_record();
  std::vector<std::uint8_t> payload;
  std::string error;
  CHECK(write_lcr_payload(record, payload, error));
  CHECK(to_hex(payload) ==
        "40000001780b00115302a40284067068011000180670680110001840");

  LayerConfigRecord parsed;
  EXPECT_OK([&](std::string& e) {
    return read_lcr_payload(payload, true, parsed, e);
  });
  CHECK(parsed.is_global);
  CHECK(parsed.global.config_record_id == 2);
  CHECK(parsed.global.xlayer_map == 0x05U);
  CHECK(parsed.global.aggregate_info_present);
  CHECK(parsed.global.aggregate.config_idc == 4);
  CHECK(parsed.global.aggregate.max_tier_flag);
  CHECK(parsed.global.profile_tier_level_info_present);
  CHECK(parsed.global.ptl[0].tier_flag);
  CHECK(!parsed.global.ptl[2].tier_flag);
  CHECK(parsed.global.global_purpose_id == 2);
  CHECK(parsed.global.doh_constraint_flag);
  CHECK(parsed.global.enforce_tile_alignment_flag);
  CHECK(parsed.global.payloads.size() == 2);
  CHECK(parsed.global.payloads[0].xlayer_id == 0);
  CHECK(parsed.global.payloads[1].xlayer_id == 2);
  CHECK(parsed.global.payloads[0].declared_size_bytes == 6);
  CHECK(parsed.global.payloads[0].info.purpose_id == 3);
  CHECK(parsed.global.payloads[0].info.color.description_idc == 2);

  std::vector<std::uint8_t> rewritten;
  EXPECT_OK([&](std::string& e) {
    return write_lcr_payload(parsed, rewritten, e);
  });
  CHECK(rewritten == payload);
}

void test_global_ops_golden() {
  const OperatingPointSet set = ops_full_set();
  std::vector<std::uint8_t> payload;
  std::string error;
  CHECK(write_ops_payload(set, payload, error));
  CHECK(to_hex(payload) ==
        "83103e17022286500009c408002711400000001814640440a320220f02228648800"
        "00003028c80881460000f0222864900000003028c808814600040");

  OperatingPointSet parsed;
  EXPECT_OK([&](std::string& e) {
    return read_ops_payload(payload, kGlobalXLayerId, parsed, e);
  });
  CHECK(parsed.xlayer_id == kGlobalXLayerId);
  CHECK(parsed.reset_flag);
  CHECK(parsed.ops_id == 0);
  CHECK(parsed.priority == 1);
  CHECK(parsed.intent == 1);
  CHECK(parsed.intent_present);
  CHECK(parsed.ptl_present);
  CHECK(parsed.color_info_present);
  CHECK(parsed.mlayer_info_idc == 2);
  CHECK(parsed.points.size() == 3);
  CHECK(parsed.points[0].decoder_model_info_present);
  CHECK(parsed.points[0].decoder_model.decoder_buffer_delay == 80000);
  CHECK(parsed.points[0].decoder_model.encoder_buffer_delay == 10000);
  CHECK(!parsed.points[1].decoder_model_info_present);
  CHECK(parsed.points[2].initial_display_delay_minus_1 == 2);
  CHECK(parsed.points[0].xlayer_map == 0x03U);
  CHECK(parsed.points[0].aggregate.level_idx == 10);
  CHECK(parsed.points[0].color.description_idc == 1);
  CHECK(parsed.points[0].layer_info[1].explicit_info);
  CHECK(!parsed.points[1].layer_info[1].explicit_info);
  CHECK(parsed.points[1].layer_info[1].embedded_ops_id == 0);
  CHECK(parsed.points[1].layer_info[1].embedded_op_index == 0);

  std::vector<std::uint8_t> rewritten;
  EXPECT_OK([&](std::string& e) {
    return write_ops_payload(parsed, rewritten, e);
  });
  CHECK(rewritten == payload);
}

// ---------------------------------------------------------------------------
// Round-trips over the feature matrix
// ---------------------------------------------------------------------------

void test_lcr_roundtrip_matrix() {
  for (int variant = 0; variant < 6; ++variant) {
    LayerConfigRecord record = global_full_record();
    GlobalLcrInfo& global = record.global;
    switch (variant) {
      case 0:
        break;
      case 1:
        global.aggregate_info_present = false;
        break;
      case 2:
        global.profile_tier_level_info_present = false;
        break;
      case 3:
        global.global_atlas_id_present = true;
        global.global_atlas_id = 7;
        global.global_payload_present = false;
        global.payloads.clear();
        global.ptl[0].tier_flag = false;
        global.aggregate.max_tier_flag = false;
        break;
      case 4:
        // Declared lcr_data_size larger than needed exercises
        // lcr_remaining_payload_bit.
        global.payloads[0].declared_size_bytes = 9;
        break;
      case 5:
        record = local_full_record();
        record.local.local_atlas_id_present = false;
        record.local.profile_tier_level_info_present = false;
        record.local.xlayer.color_info_present = false;
        record.local.xlayer.rep_info_present = false;
        break;
      default:
        break;
    }
    std::vector<std::uint8_t> payload;
    EXPECT_OK([&](std::string& e) { return write_lcr_payload(record, payload, e); });
    LayerConfigRecord parsed;
    EXPECT_OK([&](std::string& e) {
      return read_lcr_payload(payload, record.is_global, parsed, e);
    });
    std::vector<std::uint8_t> rewritten;
    EXPECT_OK([&](std::string& e) { return write_lcr_payload(parsed, rewritten, e); });
    CHECK(rewritten == payload);
    if (variant == 4) {
      CHECK(parsed.global.payloads[0].declared_size_bytes == 9);
    }
  }
}

void test_ops_roundtrip_matrix() {
  for (int variant = 0; variant < 7; ++variant) {
    OperatingPointSet set = ops_full_set();
    switch (variant) {
      case 0:
        break;
      case 1:
        set.mlayer_info_idc = 1;
        break;
      case 2:
        set.mlayer_info_idc = 0;
        break;
      case 3:
        set.intent_present = false;
        set.ptl_present = false;
        set.color_info_present = false;
        break;
      case 4:
        // ops_reset_flag = 1, ops_cnt = 0: full reset.
        set.points.clear();
        set.reset_flag = true;
        break;
      case 5:
        // ops_reset_flag = 0, ops_cnt = 0: reset only this OPS id.
        set.points.clear();
        set.reset_flag = false;
        break;
      case 6: {
        // Local OPS.
        set = OperatingPointSet{};
        set.xlayer_id = 3;
        set.ops_id = 5;
        set.reset_flag = true;
        set.priority = 2;
        set.intent = 1;
        set.intent_present = true;
        set.ptl_present = true;
        set.color_info_present = true;
        OpsPoint point;
        point.op_intent = 1;
        point.color.description_idc = 0;
        point.color.primaries = 9;
        point.color.transfer = 16;
        point.color.matrix = 9;
        point.ptl[3] = ops_ptl(0, 9, false, 2);
        point.decoder_model_info_present = true;
        point.decoder_model.low_delay_mode_flag = true;
        point.initial_display_delay_present = true;
        point.initial_display_delay_minus_1 = 4;
        point.layer_info[3].mlayer_map = 0x03U;
        point.layer_info[3].tlayer_map[0] = 0x03U;
        point.layer_info[3].tlayer_map[1] = 0x01U;
        set.points.push_back(point);
        break;
      }
      default:
        break;
    }
    std::vector<std::uint8_t> payload;
    EXPECT_OK([&](std::string& e) { return write_ops_payload(set, payload, e); });
    OperatingPointSet parsed;
    EXPECT_OK([&](std::string& e) {
      return read_ops_payload(payload, set.xlayer_id, parsed, e);
    });
    CHECK(parsed.xlayer_id == set.xlayer_id);
    CHECK(parsed.ops_id == set.ops_id);
    CHECK(parsed.points.size() == set.points.size());
    std::vector<std::uint8_t> rewritten;
    EXPECT_OK([&](std::string& e) { return write_ops_payload(parsed, rewritten, e); });
    CHECK(rewritten == payload);
  }
}

// ---------------------------------------------------------------------------
// Writer rejections
// ---------------------------------------------------------------------------

void test_writer_rejections() {
  {
    LayerConfigRecord record = global_full_record();
    record.global.config_record_id = 0;
    EXPECT_FAILS([&](std::string& e) {
      std::vector<std::uint8_t> out;
      return write_lcr_payload(record, out, e);
    }, "1..7");
  }
  {
    LayerConfigRecord record = global_full_record();
    record.global.xlayer_map = 0;
    EXPECT_FAILS([&](std::string& e) {
      std::vector<std::uint8_t> out;
      return write_lcr_payload(record, out, e);
    }, "lcr_xlayer_map");
  }
  {
    LayerConfigRecord record = global_full_record();
    record.global.dependent_xlayers_flag = true;
    EXPECT_FAILS([&](std::string& e) {
      std::vector<std::uint8_t> out;
      return write_lcr_payload(record, out, e);
    }, "must be 0");
  }
  {
    LayerConfigRecord record = global_full_record();
    record.global.payloads.pop_back();
    EXPECT_FAILS([&](std::string& e) {
      std::vector<std::uint8_t> out;
      return write_lcr_payload(record, out, e);
    }, "one payload per layer");
  }
  {
    LayerConfigRecord record = local_full_record();
    record.local.xlayer.embedded[0].tlayer_map = 0;
    EXPECT_FAILS([&](std::string& e) {
      std::vector<std::uint8_t> out;
      return write_lcr_payload(record, out, e);
    }, "tlayer_map");
  }
  {
    LayerConfigRecord record = local_full_record();
    record.local.xlayer.mlayer_map = 0x07U;  // three bits, two descriptors
    EXPECT_FAILS([&](std::string& e) {
      std::vector<std::uint8_t> out;
      return write_lcr_payload(record, out, e);
    }, "exactly the embedded layers");
  }
  {
    LayerConfigRecord record = local_full_record();
    record.local.xlayer.color.description_idc = 4;  // Table 6.13, not rg(2)
    EXPECT_FAILS([&](std::string& e) {
      std::vector<std::uint8_t> out;
      return write_lcr_payload(record, out, e);
    }, "rg(2)");
  }
  {
    LayerConfigRecord record = local_full_record();
    record.local.xlayer.embedded[0].dependent_layer_mask = 2;  // j = 0
    EXPECT_FAILS([&](std::string& e) {
      std::vector<std::uint8_t> out;
      return write_lcr_payload(record, out, e);
    }, ">= j");
  }
  {
    OperatingPointSet set = ops_full_set();
    set.mlayer_info_idc = 3;
    EXPECT_FAILS([&](std::string& e) {
      std::vector<std::uint8_t> out;
      return write_ops_payload(set, out, e);
    }, "must not be 3");
  }
  {
    OperatingPointSet set = ops_full_set();
    set.points.resize(8);
    EXPECT_FAILS([&](std::string& e) {
      std::vector<std::uint8_t> out;
      return write_ops_payload(set, out, e);
    }, "0..7");
  }
  {
    OperatingPointSet set = ops_full_set();
    set.ops_id = 16;
    EXPECT_FAILS([&](std::string& e) {
      std::vector<std::uint8_t> out;
      return write_ops_payload(set, out, e);
    }, "f(4)");
  }
  {
    OperatingPointSet set = ops_full_set();
    set.points[0].layer_info[0].mlayer_map = 0;
    EXPECT_FAILS([&](std::string& e) {
      std::vector<std::uint8_t> out;
      return write_ops_payload(set, out, e);
    }, "non-empty");
  }
  {
    // Local OPS must carry ops_reserved_2bits = 0.
    OperatingPointSet set;
    set.xlayer_id = 1;
    set.reserved_2bits = 1;
    OpsPoint point;
    point.layer_info[1].mlayer_map = 1;
    point.layer_info[1].tlayer_map[0] = 1;
    set.points.push_back(point);
    EXPECT_FAILS([&](std::string& e) {
      std::vector<std::uint8_t> out;
      return write_ops_payload(set, out, e);
    }, "ops_reserved_2bits");
  }
}

// ---------------------------------------------------------------------------
// Reader rejections
// ---------------------------------------------------------------------------

void test_reader_rejections() {
  std::vector<std::uint8_t> payload;
  std::string error;
  {
    LayerConfigRecord record = global_full_record();
    CHECK(write_lcr_payload(record, payload, error));
  }
  {
    // The extensible-OBU tail is the last thing in the payload: bit 7 of the
    // final byte is obu_extension_flag, bit 6 is the trailing stop bit.
    std::vector<std::uint8_t> bad = payload;
    bad.back() = static_cast<std::uint8_t>(bad.back() | 0x80U);
    EXPECT_FAILS([&](std::string& e) {
      LayerConfigRecord parsed;
      return read_lcr_payload(bad, true, parsed, e);
    }, "obu_extension_flag");
  }
  {
    std::vector<std::uint8_t> bad = payload;
    bad.pop_back();
    EXPECT_FAILS([&](std::string& e) {
      LayerConfigRecord parsed;
      return read_lcr_payload(bad, true, parsed, e);
    }, "truncated");
  }
  {
    // lcr_data_size at byte 13: shrink it below the real payload.
    std::vector<std::uint8_t> bad = payload;
    CHECK(bad.size() > 13);
    bad[13] = 0x20;
    EXPECT_FAILS([&](std::string& e) {
      LayerConfigRecord parsed;
      return read_lcr_payload(bad, true, parsed, e);
    }, "lcr_data_size");
  }
  {
    // Corrupt a trailing padding bit.
    std::vector<std::uint8_t> bad = payload;
    bad.back() = static_cast<std::uint8_t>(bad.back() | 0x01U);
    EXPECT_FAILS([&](std::string& e) {
      LayerConfigRecord parsed;
      return read_lcr_payload(bad, true, parsed, e);
    }, "trailing_bits");
  }

  std::vector<std::uint8_t> ops_payload;
  {
    OperatingPointSet set = ops_full_set();
    CHECK(write_ops_payload(set, ops_payload, error));
  }
  {
    // ops_data_size lives at byte 3 (24-bit header). Grow it so opsBytes no
    // longer equals ops_data_size.
    std::vector<std::uint8_t> bad = ops_payload;
    CHECK(bad.size() > 3);
    bad[3] = static_cast<std::uint8_t>(bad[3] + 1U);
    EXPECT_FAILS([&](std::string& e) {
      OperatingPointSet parsed;
      return read_ops_payload(bad, kGlobalXLayerId, parsed, e);
    }, "opsBytes");
  }
  {
    std::vector<std::uint8_t> bad = ops_payload;
    bad.back() = static_cast<std::uint8_t>(bad.back() | 0x01U);
    EXPECT_FAILS([&](std::string& e) {
      OperatingPointSet parsed;
      return read_ops_payload(bad, kGlobalXLayerId, parsed, e);
    }, "trailing_bits");
  }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

LayerConfigRecord two_layer_global_lcr() {
  LayerConfigRecord record;
  record.is_global = true;
  GlobalLcrInfo& global = record.global;
  global.config_record_id = 1;
  global.xlayer_map = 0x03U;
  global.profile_tier_level_info_present = true;
  global.global_payload_present = true;
  for (std::uint8_t x : {std::uint8_t{0}, std::uint8_t{1}}) {
    global.ptl[x] = ptl(0, 7, false, 1);
    GlobalPayload payload;
    payload.xlayer_id = x;
    payload.info.rep_info_present = true;
    payload.info.rep.max_pic_width = x == 0 ? 64U : 128U;
    payload.info.rep.max_pic_height = x == 0 ? 64U : 96U;
    payload.info.rep.format_info_present = true;
    payload.info.rep.bit_depth_idc = 1;
    payload.info.rep.chroma_format_idc = 0;
    payload.info.embedded_layer_info_present = true;
    payload.info.mlayer_map = 0x01U;
    payload.info.embedded.push_back(embedded_texture(0, 0x01U));
    global.payloads.push_back(payload);
  }
  return record;
}

OperatingPointSet two_layer_ops() {
  OperatingPointSet set;
  set.xlayer_id = kGlobalXLayerId;
  set.ops_id = 0;
  set.mlayer_info_idc = 1;
  set.ptl_present = true;
  for (int i = 0; i < 2; ++i) {
    OpsPoint point;
    point.xlayer_map = static_cast<std::uint32_t>((1U << (i + 1)) - 1U);
    point.aggregate.level_idx = 7;
    for (int x = 0; x <= i; ++x) {
      point.ptl[static_cast<std::size_t>(x)] = ops_ptl(0, 7, false, 1);
      point.layer_info[static_cast<std::size_t>(x)].mlayer_map = 0x01U;
      point.layer_info[static_cast<std::size_t>(x)].tlayer_map[0] = 0x01U;
    }
    set.points.push_back(point);
  }
  return set;
}

void test_validation_lcr() {
  StreamContext context = two_layer_context();
  {
    ValidationReport report;
    CHECK(validate_lcr(two_layer_global_lcr(), context, report));
  }
  {
    LayerConfigRecord record = two_layer_global_lcr();
    record.global.ptl[1].max_level_idx = 5;  // below seq_level_idx 7
    ValidationReport report;
    CHECK(!validate_lcr(record, context, report));
    EXPECT_REPORT(report, "seq_level_idx");
  }
  {
    LayerConfigRecord record = two_layer_global_lcr();
    record.global.ptl[0].seq_profile_idc = 0;
    context.sequences[0].seq_profile_idc = 3;
    ValidationReport report;
    CHECK(!validate_lcr(record, context, report));
    EXPECT_REPORT(report, "seq_profile_idc");
    context.sequences[0].seq_profile_idc = 0;
  }
  {
    LayerConfigRecord record = two_layer_global_lcr();
    record.global.ptl[0].max_mlayer_count = 0;
    ValidationReport report;
    CHECK(!validate_lcr(record, context, report));
    EXPECT_REPORT(report, "seq_max_mlayer_cnt");
  }
  {
    LayerConfigRecord record = two_layer_global_lcr();
    record.global.aggregate_info_present = true;
    record.global.aggregate.level_idx = 6;  // below the per-layer maximum
    ValidationReport report;
    CHECK(!validate_lcr(record, context, report));
    EXPECT_REPORT(report, "lcr_aggregate_level_idx");
  }
  {
    LayerConfigRecord record = two_layer_global_lcr();
    record.global.doh_constraint_flag = false;
    context.sequences[0].monotonic_output_order = false;
    ValidationReport report;
    CHECK(!validate_lcr(record, context, report));
    EXPECT_REPORT(report, "doh_constraint");
    context.sequences[0].monotonic_output_order = true;
  }
  {
    LayerConfigRecord record = two_layer_global_lcr();
    record.global.payloads[0].info.rep.max_pic_width = 65;
    ValidationReport report;
    CHECK(!validate_lcr(record, context, report));
    EXPECT_REPORT(report, "max_frame_width");
  }
  {
    LayerConfigRecord record = two_layer_global_lcr();
    record.global.payloads[0].info.rep.bit_depth_idc = 0;
    ValidationReport report;
    CHECK(!validate_lcr(record, context, report));
    EXPECT_REPORT(report, "bit_depth/chroma");
  }
  {
    LayerConfigRecord record = two_layer_global_lcr();
    context.sequences[0].cropping_window_present = true;
    ValidationReport report;
    CHECK(!validate_lcr(record, context, report));
    EXPECT_REPORT(report, "cropping");
    context.sequences[0].cropping_window_present = false;
  }
  {
    LayerConfigRecord record = two_layer_global_lcr();
    record.global.payloads[0].info.embedded[0].same_sh_max_resolution = false;
    record.global.payloads[0].info.embedded[0].max_expected_width = 128;
    record.global.payloads[0].info.embedded[0].max_expected_height = 64;
    ValidationReport report;
    CHECK(!validate_lcr(record, context, report));
    EXPECT_REPORT(report, "lcr_max_expected_width");
  }
  {
    // xlayer map closure: a selected bit with no payload is reported.
    LayerConfigRecord record = two_layer_global_lcr();
    record.global.xlayer_map |= 1U << 5;
    ValidationReport report;
    CHECK(!validate_lcr(record, context, report));
    EXPECT_REPORT(report, "no payload");
  }
  {
    // Local LCR whose lcr_global_id disagrees with the activated global LCR.
    LayerConfigRecord record = local_full_record();
    StreamContext local = context;
    local.lcr_active = true;
    local.active_global_lcr_id = 2;
    ValidationReport report;
    CHECK(!validate_lcr(record, local, report));
    EXPECT_REPORT(report, "lcr_global_id");
  }
}

void test_validation_ops() {
  StreamContext context = two_layer_context();
  {
    ValidationReport report;
    CHECK(validate_ops(two_layer_ops(), context, report));
  }
  {
    OperatingPointSet set = two_layer_ops();
    set.points[1].ptl[1].level_idx = 6;
    ValidationReport report;
    CHECK(!validate_ops(set, context, report));
    EXPECT_REPORT(report, "ops_seq_profile_tier_level_info x1");
  }
  {
    OperatingPointSet set = two_layer_ops();
    set.points[1].aggregate.level_idx = 6;
    ValidationReport report;
    CHECK(!validate_ops(set, context, report));
    EXPECT_REPORT(report, "ops_aggregate_level_idx");
  }
  {
    OperatingPointSet set = two_layer_ops();
    set.points[1].ptl[1].tier_flag = true;
    set.points[1].aggregate.max_tier_flag = false;
    ValidationReport report;
    CHECK(!validate_ops(set, context, report));
    EXPECT_REPORT(report, "ops_max_tier_flag");
  }
  {
    OperatingPointSet set = two_layer_ops();
    set.points[1].layer_info[1].tlayer_map[0] = 0;
    ValidationReport report;
    CHECK(!validate_ops(set, context, report));
    EXPECT_REPORT(report, "empty");
  }
  {
    OperatingPointSet set = two_layer_ops();
    set.points[1].xlayer_map = 1U << 7;
    ValidationReport report;
    CHECK(!validate_ops(set, context, report));
    EXPECT_REPORT(report, "no activated sequence header");
  }
  {
    // idc 2: inheritance must reference a known OPS.
    OperatingPointSet set = two_layer_ops();
    set.mlayer_info_idc = 2;
    for (OpsPoint& point : set.points) {
      for (int x = 0; x < kXLayerCount; ++x) {
        point.layer_info[static_cast<std::size_t>(x)].explicit_info = true;
      }
    }
    set.points[1].layer_info[1].explicit_info = false;
    set.points[1].layer_info[1].embedded_ops_id = 9;
    StreamContext ops_context = context;
    ops_context.known_ops.push_back(summarize_ops(set));
    ValidationReport report;
    CHECK(!validate_ops(set, ops_context, report));
    EXPECT_REPORT(report, "names no OPS");
  }
  {
    // idc 2: self-referential inheritance must point at an earlier OP.
    OperatingPointSet set = two_layer_ops();
    set.mlayer_info_idc = 2;
    for (OpsPoint& point : set.points) {
      for (int x = 0; x < kXLayerCount; ++x) {
        point.layer_info[static_cast<std::size_t>(x)].explicit_info = true;
      }
    }
    set.points[1].layer_info[1].explicit_info = false;
    set.points[1].layer_info[1].embedded_ops_id = 0;
    set.points[1].layer_info[1].embedded_op_index = 1;
    StreamContext ops_context = context;
    ops_context.known_ops.push_back(summarize_ops(set));
    ValidationReport report;
    CHECK(!validate_ops(set, ops_context, report));
    EXPECT_REPORT(report, "self-referential");
  }
  {
    // Local OPS whose xlayer has no sequence header.
    OperatingPointSet set;
    set.xlayer_id = 9;
    OpsPoint point;
    point.layer_info[9].mlayer_map = 1;
    point.layer_info[9].tlayer_map[0] = 1;
    set.points.push_back(point);
    ValidationReport report;
    CHECK(!validate_ops(set, context, report));
    EXPECT_REPORT(report, "no activated sequence header");
  }
}

void test_validation_closure() {
  layers::LayerDependencyMaps deps = layers::default_dependency_maps(2, 1);
  // m2 depends on m0..m1; t1 depends on t0.
  {
    const std::array<std::uint8_t, 8> tlayers = {0x03, 0x03, 0x03, 0, 0, 0, 0, 0};
    ValidationReport report;
    CHECK(validate_layer_map_closure(0, 0x07U, tlayers, deps, report));
  }
  {
    const std::array<std::uint8_t, 8> tlayers = {0x03, 0x03, 0x03, 0, 0, 0, 0, 0};
    ValidationReport report;
    CHECK(!validate_layer_map_closure(0, 0x04U, tlayers, deps, report));
    EXPECT_REPORT(report, "depends on m0");
  }
  {
    const std::array<std::uint8_t, 8> tlayers = {0x01, 0, 0, 0, 0, 0, 0, 0};
    ValidationReport report;
    CHECK(validate_layer_map_closure(0, 0x01U, tlayers, deps, report));
    // t0 has no dependency, so this is closed.
    CHECK(report.ok());
    const std::array<std::uint8_t, 8> t2 = {0x02, 0, 0, 0, 0, 0, 0, 0};
    ValidationReport report2;
    CHECK(!validate_layer_map_closure(0, 0x01U, t2, deps, report2));
    EXPECT_REPORT(report2, "depends on t0");
  }
  {
    // OPS and LCR share the same closure rule.
    StreamContext context = two_layer_context();
    OperatingPointSet set = two_layer_ops();
    set.points[1].layer_info[1].mlayer_map = 0x02U;  // m1 without m0
    ValidationReport report;
    CHECK(!validate_ops(set, context, report));
    EXPECT_REPORT(report, "depends on m0");
  }
}

// ---------------------------------------------------------------------------
// BRT <-> OPS
// ---------------------------------------------------------------------------

void test_brt() {
  const std::vector<OpsSummary> known = {
      OpsSummary{kGlobalXLayerId, 0, 3},
      OpsSummary{kGlobalXLayerId, 1, 1},
  };
  {
    BrtCorrespondence brt;
    brt.br_ops_dependent_flag = false;
    brt.br_time = 511;
    ValidationReport report;
    CHECK(validate_brt_against_ops(brt, known, report));
  }
  {
    BrtCorrespondence brt;
    brt.br_ops_dependent_flag = false;
    brt.br_time = 512;
    ValidationReport report;
    CHECK(!validate_brt_against_ops(brt, known, report));
    EXPECT_REPORT(report, "rg(4)");
  }
  {
    BrtCorrespondence brt;
    brt.br_ops_dependent_flag = true;
    brt.br_ops_id = 0;
    brt.br_ops_cnt = 3;
    brt.br_time_op = {10, 20, 30};
    ValidationReport report;
    CHECK(validate_brt_against_ops(brt, known, report));
  }
  {
    BrtCorrespondence brt;
    brt.br_ops_dependent_flag = true;
    brt.br_ops_id = 7;
    brt.br_ops_cnt = 1;
    brt.br_time_op = {1};
    ValidationReport report;
    CHECK(!validate_brt_against_ops(brt, known, report));
    EXPECT_REPORT(report, "not an ops_id");
  }
  {
    BrtCorrespondence brt;
    brt.br_ops_dependent_flag = true;
    brt.br_ops_id = 0;
    brt.br_ops_cnt = 2;  // ops_cnt is 3
    brt.br_time_op = {1, 2};
    ValidationReport report;
    CHECK(!validate_brt_against_ops(brt, known, report));
    EXPECT_REPORT(report, "does not equal ops_cnt");
  }
  {
    BrtCorrespondence brt;
    brt.br_ops_dependent_flag = true;
    brt.br_ops_id = 0;
    brt.br_ops_cnt = 3;
    brt.br_time_op = {1, 2};  // count mismatch
    ValidationReport report;
    CHECK(!validate_brt_against_ops(brt, known, report));
    EXPECT_REPORT(report, "br_time_op entries");
  }
  {
    // The same ops_id on two xlayers with different ops_cnt is ambiguous.
    const std::vector<OpsSummary> ambiguous = {
        OpsSummary{kGlobalXLayerId, 4, 2},
        OpsSummary{0, 4, 3},
    };
    BrtCorrespondence brt;
    brt.br_ops_dependent_flag = true;
    brt.br_ops_id = 4;
    brt.br_ops_cnt = 2;
    brt.br_time_op = {1, 2};
    ValidationReport report;
    CHECK(!validate_brt_against_ops(brt, ambiguous, report));
    EXPECT_REPORT(report, "ambiguous");
  }
}

void test_inheritance_resolution() {
  OpsLayerMlayerInfo inherited;
  inherited.explicit_info = false;
  inherited.embedded_ops_id = 2;
  inherited.embedded_op_index = 1;
  std::vector<OpsInheritanceSource> sources(1);
  sources[0].ops_id = 2;
  sources[0].op_index = 1;
  sources[0].mlayer_map = 0x03U;
  sources[0].tlayer_map[0] = 0x03U;
  sources[0].tlayer_map[1] = 0x01U;
  OpsLayerMlayerInfo resolved;
  std::string error;
  CHECK(resolve_ops_inheritance(inherited, sources, resolved, error));
  CHECK(resolved.explicit_info);
  CHECK(resolved.mlayer_map == 0x03U);
  CHECK(resolved.tlayer_map[0] == 0x03U);
  CHECK(resolved.tlayer_map[1] == 0x01U);

  inherited.embedded_ops_id = 5;
  CHECK(!resolve_ops_inheritance(inherited, sources, resolved, error));
  CHECK(error.find("no operating point") != std::string::npos);
}

// ---------------------------------------------------------------------------
// Cross-check against svc_framing (the merge path)
// ---------------------------------------------------------------------------

legacy::LayerPlan build_plan(int spatial_layers, int temporal, int embedded,
                             bool with_lcr) {
  legacy::LayerConfig config;
  const int widths[3] = {64, 128, 256};
  const int heights[3] = {64, 96, 144};
  for (int i = 0; i < spatial_layers; ++i) {
    legacy::SpatialLayerSpec spec;
    spec.xlayer_id = static_cast<std::uint8_t>(i);
    spec.width = widths[i];
    spec.height = heights[i];
    config.spatial_layers.push_back(spec);
  }
  config.temporal_layers = static_cast<std::uint8_t>(temporal);
  config.embedded_layers = static_cast<std::uint8_t>(embedded);
  config.single_picture_header = false;
  config.still_picture = false;
  config.seq_lcr_id = with_lcr ? 1 : 0;
  legacy::LayerPlan plan;
  std::string error;
  CHECK(legacy::build_layer_plan(config, plan, error));
  return plan;
}

void test_cross_check_svc_framing() {
  struct Case {
    int spatial;
    int temporal;
    int embedded;
  };
  const Case cases[] = {{2, 1, 1}, {3, 1, 1}, {2, 2, 2}, {3, 3, 2}};
  for (const Case& c : cases) {
    const legacy::LayerPlan plan =
        build_plan(c.spatial, c.temporal, c.embedded, true);

    std::vector<std::uint8_t> legacy_lcr;
    std::string error;
    CHECK(legacy::write_layer_config_record_payload(
        plan.lcr, plan.sequences.front().dependencies, legacy_lcr, error));

    const LayerConfigRecord converted =
        lcr_from_legacy(plan.lcr, plan.sequences.front().dependencies, false);
    std::vector<std::uint8_t> full_lcr;
    EXPECT_OK([&](std::string& e) { return write_lcr_payload(converted, full_lcr, e); });
    CHECK(full_lcr == legacy_lcr);

    std::vector<std::uint8_t> legacy_ops;
    CHECK(legacy::write_operating_point_set_payload(plan.ops, legacy_ops,
                                                    error));
    const OperatingPointSet converted_ops = ops_from_legacy(plan.ops, false);
    std::vector<std::uint8_t> full_ops;
    EXPECT_OK([&](std::string& e) { return write_ops_payload(converted_ops, full_ops, e); });
    CHECK(full_ops == legacy_ops);

    // The converted skeleton must parse back and validate against sequence
    // headers built from the same plan.
    LayerConfigRecord parsed_lcr;
    EXPECT_OK([&](std::string& e) {
      return read_lcr_payload(legacy_lcr, true, parsed_lcr, e);
    });
    OperatingPointSet parsed_ops;
    EXPECT_OK([&](std::string& e) {
      return read_ops_payload(legacy_ops, kGlobalXLayerId, parsed_ops, e);
    });
    StreamContext context;
    for (const legacy::LayerSpec& layer : plan.layers) {
      if (layer.id.mlayer_id != 0 || layer.id.tlayer_id != 0) continue;
      context.sequences.push_back(sequence_info(
          layer.id.xlayer_id, static_cast<std::uint32_t>(layer.width),
          static_cast<std::uint32_t>(layer.height), 0, 7, false,
          static_cast<std::uint8_t>(c.embedded)));
    }
    context.dependencies = plan.sequences.front().dependencies;
    context.lcr_active = true;
    context.active_global_lcr_id = 1;
    context.known_ops.push_back(
        OpsSummary{kGlobalXLayerId, plan.ops.ops_id,
                   static_cast<std::uint8_t>(plan.ops.points.size())});
    ValidationReport report;
    if (!validate_stream(parsed_lcr, parsed_ops, context, report)) {
      ++g_failures;
      std::cerr << "cross-check validation failed for " << c.spatial << "/"
                << c.temporal << '/' << c.embedded << ":\n";
      for (const std::string& e : report.errors) {
        std::cerr << "  " << e << '\n';
      }
    }
  }
}

void test_cross_check_local_lcr() {
  // svc_framing also writes the local LCR form; the skeleton conversion must
  // match it byte for byte.
  layers::LayerConfigRecord legacy_record;
  legacy_record.is_global = false;
  legacy_record.xlayer_id = 2;
  legacy_record.global_config_record_id = 1;
  layers::ExtendedLayerConfig layer;
  layer.xlayer_id = 2;
  layer.mlayer_map = 0x03U;
  layer.tlayer_map[0] = 0x03U;
  layer.tlayer_map[1] = 0x01U;
  layer.dependent_xlayers = false;
  for (int m = 0; m < 2; ++m) {
    layers::EmbeddedLayerConfig embedded;
    embedded.mlayer_id = static_cast<std::uint8_t>(m);
    embedded.tlayer_map = layer.tlayer_map[static_cast<std::size_t>(m)];
    embedded.same_sh_max_resolution = (m == 0);
    embedded.max_expected_width = (m == 0) ? 0U : 128U;
    embedded.max_expected_height = (m == 0) ? 0U : 96U;
    layer.embedded.push_back(embedded);
  }
  legacy_record.layers.push_back(layer);
  const layers::LayerDependencyMaps deps = layers::default_dependency_maps(1, 1);

  std::vector<std::uint8_t> legacy_payload;
  std::string error;
  CHECK(legacy::write_layer_config_record_payload(legacy_record, deps,
                                                  legacy_payload, error));
  const LayerConfigRecord converted =
      lcr_from_legacy(legacy_record, deps, false);
  std::vector<std::uint8_t> payload;
  EXPECT_OK([&](std::string& e) { return write_lcr_payload(converted, payload, e); });
  CHECK(payload == legacy_payload);

  LayerConfigRecord parsed;
  EXPECT_OK([&](std::string& e) {
    return read_lcr_payload(payload, false, parsed, e);
  });
  CHECK(parsed.local.local_id == 1);
  CHECK(parsed.local.xlayer.embedded.size() == 2);
  CHECK(parsed.local.xlayer.embedded[1].max_expected_width == 128);
}

void test_local_global_lcr_agreement() {
  const LayerConfigRecord global = two_layer_global_lcr();
  LayerConfigRecord local;
  local.is_global = false;
  local.local.xlayer_id = 1;
  local.local.global_id = global.global.config_record_id;
  local.local.local_id = 2;
  local.local.profile_tier_level_info_present = true;
  local.local.ptl = global.global.ptl[1];
  local.local.xlayer = global.global.payloads[1].info;

  ValidationReport report;
  CHECK(validate_local_lcr_matches_global(global, local, report));

  LayerConfigRecord mismatch = local;
  mismatch.local.xlayer.mlayer_map = 0x02U;  // m1 only
  ValidationReport bad;
  CHECK(!validate_local_lcr_matches_global(global, mismatch, bad));
  EXPECT_REPORT(bad, "lcr_mlayer_map");

  LayerConfigRecord colour = local;
  colour.local.xlayer.color_info_present = true;
  colour.local.xlayer.color.description_idc = 0;
  ValidationReport bad2;
  CHECK(!validate_local_lcr_matches_global(global, colour, bad2));
  EXPECT_REPORT(bad2, "presence flags");

  LayerConfigRecord parent = local;
  parent.local.global_id = 3;
  ValidationReport bad3;
  CHECK(!validate_local_lcr_matches_global(global, parent, bad3));
  EXPECT_REPORT(bad3, "lcr_global_id");
}

void test_full_legacy_forms_roundtrip() {
  const legacy::LayerPlan plan = build_plan(2, 2, 2, true);
  const LayerConfigRecord lcr =
      lcr_from_legacy(plan.lcr, plan.sequences.front().dependencies, true);
  const OperatingPointSet ops = ops_from_legacy(plan.ops, true);
  std::vector<std::uint8_t> lcr_payload;
  std::vector<std::uint8_t> ops_payload;
  EXPECT_OK([&](std::string& e) { return write_lcr_payload(lcr, lcr_payload, e); });
  EXPECT_OK([&](std::string& e) { return write_ops_payload(ops, ops_payload, e); });
  CHECK(!lcr_payload.empty());
  CHECK(!ops_payload.empty());

  LayerConfigRecord parsed_lcr;
  EXPECT_OK([&](std::string& e) {
    return read_lcr_payload(lcr_payload, true, parsed_lcr, e);
  });
  OperatingPointSet parsed_ops;
  EXPECT_OK([&](std::string& e) {
    return read_ops_payload(ops_payload, kGlobalXLayerId, parsed_ops, e);
  });
  CHECK(parsed_lcr.global.profile_tier_level_info_present);
  CHECK(parsed_lcr.global.aggregate_info_present);
  CHECK(parsed_lcr.global.payloads[0].info.purpose_present);
  CHECK(parsed_lcr.global.payloads[0].info.color_info_present);
  CHECK(parsed_ops.mlayer_info_idc == 2);
  CHECK(parsed_ops.points[0].decoder_model_info_present);
  CHECK(parsed_ops.points[0].initial_display_delay_present);
}

}  // namespace

int main() {
  test_substructure_goldens();
  test_local_lcr_golden();
  test_global_lcr_golden();
  test_global_ops_golden();
  test_lcr_roundtrip_matrix();
  test_ops_roundtrip_matrix();
  test_writer_rejections();
  test_reader_rejections();
  test_validation_lcr();
  test_validation_ops();
  test_validation_closure();
  test_brt();
  test_inheritance_resolution();
  test_cross_check_svc_framing();
  test_cross_check_local_lcr();
  test_local_global_lcr_agreement();
  test_full_legacy_forms_roundtrip();

  if (g_failures != 0) {
    std::cerr << g_failures << " check(s) failed\n";
    return EXIT_FAILURE;
  }
  std::cout << "LCR/OPS full tests: all checks passed\n";
  return EXIT_SUCCESS;
}

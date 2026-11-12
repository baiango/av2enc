// Complete §5.8 LCR and §5.10/§5.11 OPS syntax, plus the §6.8/§6.10/§6.11
// conformance checks that tie them to sequence headers and to the BRT.
//
// This is the "field-complete" sibling of the skeleton writers in
// `svc_framing.cpp`. It owns only new files; `svc_framing` is unchanged.
// `README.md` in this directory gives the exact merge path.
//
// Covered by this module:
//
//   §5.8.1  lcr_global_info()          all flags, purpose id, atlas id,
//                                      reserved bits, payload loop
//   §5.8.2  lcr_local_info()           global/local ids, local atlas id
//   §5.8.3  lcr_aggregate_info()
//   §5.8.4  lcr_seq_profile_tier_level_info()
//   §5.8.5  lcr_global_payload()       lcr_data_size + remaining bits
//   §5.8.6  lcr_xlayer_info()          rep / purpose / colour / embedded
//   §5.8.7  lcr_rep_info()             size, format, cropping window
//   §5.8.8  lcr_embedded_layer_info()  atlas id, priority, rendering method,
//                                      layer type, aux type, view type/id,
//                                      dependent layer map, resolution
//   §5.8.9  lcr_xlayer_color_info()
//   §5.10   operating_point_set_obu()  including the ops_cnt == 0 reset forms
//   §5.11.1 ops_aggregate_info()
//   §5.11.2 ops_seq_profile_tier_level_info()
//   §5.11.3 ops_decoder_model_info()
//   §5.11.4 ops_color_info()
//   §5.11.5 ops_mlayer_info()
//   §5.11   ops_mlayer_info_idc == 2 inheritance
//           (ops_mlayer_explicit_info_flag / ops_embedded_ops_id /
//            ops_embedded_op_index)
//   §6.11   BRT <-> OPS correspondence
//
// Not covered here (deliberately, out of this task's scope):
//   * OBU framing / leb128 / obu_header (§5.2) — use `svc_framing.h`; the
//     payload writers in this file return the exact bytes that go after the
//     OBU header, including `obu_extension_flag` and trailing bits.
//   * Atlas segment OBUs (§5.9). Only the LCR-side atlas *references* are
//     coded here; there is no `atlas_segment_info_obu()` writer anywhere in
//     the tree.
//
// Bit order, trailing bits and byte alignment follow the PDF v1.0.0 text
// (commit 38f28e7) exactly; every syntax body was read from the PDF, not from
// a README. `lcr_ops_full_test.cpp` pins byte goldens derived independently
// from that text.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "core/bitstream/bitwriter.h"
#include "tiles/svc_layers.h"

namespace av2enc::svc::ops_full {

namespace layers = ::av2enc::tiles::svc;

inline constexpr int kGlobalXLayerId = layers::kGlobalXLayerId;      // 31
inline constexpr int kMaxExtendedLayerId = layers::kMaxExtendedLayerId;
inline constexpr int kXLayerCount = layers::kExtendedLayerCount;     // 32
inline constexpr int kMaxEmbeddedLayers = layers::kEmbeddedLayerCount;  // 8
inline constexpr int kMaxTemporalLayers = layers::kTemporalLayerCount;  // 4
inline constexpr int kMaxOpsId = 15;    // ops_id f(4)
inline constexpr int kMaxOpsCnt = 7;    // ops_cnt f(3), 0 == reset
inline constexpr std::uint32_t kMaxBrTime = 511;  // rg(4)

// Table 6.8 / 6.9 / 6.10 values.
inline constexpr std::uint8_t kLayerTexture = 0;
inline constexpr std::uint8_t kLayerAux = 1;
inline constexpr std::uint8_t kAuxAlpha = 0;
inline constexpr std::uint8_t kAuxDepth = 1;
inline constexpr std::uint8_t kAuxSegmentation = 2;
inline constexpr std::uint8_t kAuxGainMap = 3;
inline constexpr std::uint8_t kViewUnspecified = 0;
inline constexpr std::uint8_t kViewCenter = 1;
inline constexpr std::uint8_t kViewLeft = 2;
inline constexpr std::uint8_t kViewRight = 3;
inline constexpr std::uint8_t kViewExplicit = 4;

// ---------------------------------------------------------------------------
// Shared colour description (§5.8.9 / §5.11.4, Table 6.13)
// ---------------------------------------------------------------------------

struct ColorDescription {
  // layer_color_description_idc / ops_color_description_idc, rg(2).
  // 1..5 imply the primaries/transfer/matrix from Table 6.13; 0 means the
  // three explicit fields are present. 6..127 are reserved by the 8-bit
  // opsin the syntax although only 0..3 fit in rg(2); see README.
  std::uint8_t description_idc = 1;  // BT.709 SDR
  std::uint8_t primaries = 1;
  std::uint8_t transfer = 1;
  std::uint8_t matrix = 1;
  bool full_range = false;
};

// ---------------------------------------------------------------------------
// LCR data model
// ---------------------------------------------------------------------------

// §5.8.4
struct ProfileTierLevelInfo {
  std::uint8_t seq_profile_idc = 0;    // f(5)
  std::uint8_t max_level_idx = 0;      // f(5)
  bool tier_flag = false;              // f(1)
  std::uint8_t max_mlayer_count = 1;   // f(3)
};

// §5.8.3 (identical field layout to §5.11.1)
struct AggregateInfo {
  std::uint8_t config_idc = 0;    // f(6)
  std::uint8_t level_idx = 0;     // f(5)
  bool max_tier_flag = false;     // f(1)
  std::uint8_t max_interop = 0;   // f(4)
};

// §5.8.7
struct RepInfo {
  std::uint32_t max_pic_width = 0;   // uvlc()
  std::uint32_t max_pic_height = 0;  // uvlc()
  bool format_info_present = false;
  std::uint8_t bit_depth_idc = 0;    // uvlc()
  std::uint8_t chroma_format_idc = 0;  // uvlc()
  bool cropping_window_present = false;
  std::uint32_t crop_left = 0;
  std::uint32_t crop_right = 0;
  std::uint32_t crop_top = 0;
  std::uint32_t crop_bottom = 0;
};

// §5.8.8. Atlas fields are present only when the matching atlas-id flag is 1.
struct EmbeddedLayerInfo {
  std::uint8_t mlayer_id = 0;      // index j
  std::uint8_t tlayer_map = 0;     // f(4)
  std::uint8_t atlas_segment_id = 0;  // f(8), atlas-id flag only
  std::uint8_t priority_order = 0;    // f(8), atlas-id flag only
  std::uint8_t rendering_method = 0;  // f(8), atlas-id flag only
  std::uint8_t layer_type = kLayerTexture;  // f(8)
  std::uint8_t auxiliary_type = kAuxAlpha;  // f(8), AUX_LAYER only
  std::uint8_t view_type = kViewCenter;     // f(8)
  std::uint8_t view_id = 0;                 // f(8), VIEW_EXPLICIT only
  // f(j): bit r set means embedded layer j depends on embedded layer r.
  std::uint32_t dependent_layer_mask = 0;
  bool same_sh_max_resolution = true;
  std::uint32_t max_expected_width = 0;   // uvlc(), !same resolution only
  std::uint32_t max_expected_height = 0;
};

// §5.8.6
struct XLayerInfo {
  bool rep_info_present = false;
  RepInfo rep;
  bool purpose_present = false;
  std::uint8_t purpose_id = 0;   // f(7)
  bool color_info_present = false;
  ColorDescription color;
  bool embedded_layer_info_present = true;
  std::uint8_t mlayer_map = 0;   // f(8)
  std::vector<EmbeddedLayerInfo> embedded;
  // Global only, and only when !embedded_layer_info_present and the LCR
  // carries an atlas id.
  std::uint8_t xlayer_atlas_segment_id = 0;
  std::uint8_t xlayer_priority_order = 0;
  std::uint8_t xlayer_rendering_method = 0;
};

// §5.8.5: one lcr_global_payload() entry for one extended layer.
struct GlobalPayload {
  std::uint8_t xlayer_id = 0;
  std::uint32_t dependent_xlayer_mask = 0;
  XLayerInfo info;
  // lcr_data_size in bytes. 0 means "smallest size that fits". A larger value
  // pads with lcr_remaining_payload_bit = 0 and is preserved by the reader so
  // that a re-write is byte-identical.
  std::uint32_t declared_size_bytes = 0;
};

// §5.8.1
struct GlobalLcrInfo {
  std::uint8_t config_record_id = 1;  // f(3), conformance range 1..7
  std::uint32_t xlayer_map = 1;       // f(31), non-zero
  bool aggregate_info_present = false;
  AggregateInfo aggregate;
  bool profile_tier_level_info_present = false;
  std::array<ProfileTierLevelInfo, kXLayerCount> ptl{};
  bool global_payload_present = true;
  bool dependent_xlayers_flag = false;  // conformance: must be 0
  bool global_atlas_id_present = false;
  std::uint8_t global_atlas_id = 0;   // f(3)
  std::uint8_t global_purpose_id = 0; // f(7), Table 6.6
  bool doh_constraint_flag = false;
  bool enforce_tile_alignment_flag = false;
  std::vector<GlobalPayload> payloads;  // ascending xlayer_id
};

// §5.8.2
struct LocalLcrInfo {
  // Not coded in the payload: this is the OBU's obu_xlayer_id. Validation
  // needs it to find the layer's activated sequence header.
  std::uint8_t xlayer_id = 0;
  std::uint8_t global_id = 0;        // f(3), 0 == no global LCR
  std::uint8_t local_id = 1;         // f(3), conformance range 1..7
  bool profile_tier_level_info_present = false;
  ProfileTierLevelInfo ptl;
  bool local_atlas_id_present = false;
  std::uint8_t local_atlas_id = 0;   // f(3)
  XLayerInfo xlayer;
};

struct LayerConfigRecord {
  bool is_global = true;
  GlobalLcrInfo global;
  LocalLcrInfo local;
};

// ---------------------------------------------------------------------------
// OPS data model
// ---------------------------------------------------------------------------

struct OpsAggregateInfo {
  std::uint8_t config_idc = 0;   // f(6)
  std::uint8_t level_idx = 0;    // f(5)
  bool max_tier_flag = false;    // f(1)
  std::uint8_t max_interop = 0;  // f(4)
};

// §5.11.2
struct OpsProfileTierLevelInfo {
  std::uint8_t seq_profile_idc = 0;  // f(5)
  std::uint8_t level_idx = 0;        // f(5)
  bool tier_flag = false;            // f(1)
  std::uint8_t mlayer_count = 1;     // f(3)
};

// §5.11.3
struct OpsDecoderModelInfo {
  std::uint32_t decoder_buffer_delay = 70000;  // uvlc(), 1/90000 s
  std::uint32_t encoder_buffer_delay = 20000;  // uvlc(), 1/90000 s
  bool low_delay_mode_flag = false;
};

// §5.11.5 plus the idc == 2 inheritance fields (§5.11).
struct OpsLayerMlayerInfo {
  bool explicit_info = true;  // ops_mlayer_explicit_info_flag (idc == 2 only)
  std::uint8_t mlayer_map = 0;  // f(8)
  std::array<std::uint8_t, kMaxEmbeddedLayers> tlayer_map{};  // f(4) each
  std::uint8_t embedded_ops_id = 0;    // f(4), inherited only
  std::uint8_t embedded_op_index = 0;  // f(3), inherited only
};

// §5.11 operating_point_payload()
struct OpsPoint {
  bool op_intent_present = false;
  std::uint8_t op_intent = 0;   // f(7), Table 6.12
  OpsAggregateInfo aggregate;   // global OPS only
  // Indexed by obu_xlayer_id (= j in the global loop). For a local OPS only
  // the entry for the OPS's own xlayer_id is used.
  std::array<OpsProfileTierLevelInfo, kXLayerCount> ptl{};
  ColorDescription color;
  bool decoder_model_info_present = false;
  OpsDecoderModelInfo decoder_model;
  bool initial_display_delay_present = false;
  std::uint8_t initial_display_delay_minus_1 = 0;  // f(4)
  std::uint32_t xlayer_map = 0;  // f(31), global OPS only
  std::array<OpsLayerMlayerInfo, kXLayerCount> layer_info{};
};

// §5.10
struct OperatingPointSet {
  std::uint8_t xlayer_id = kGlobalXLayerId;  // 0..31
  std::uint8_t ops_id = 0;                   // f(4)
  bool reset_flag = true;
  std::uint8_t priority = 0;                 // f(4)
  std::uint8_t intent = 0;                   // f(7), Table 6.12
  bool intent_present = false;
  bool ptl_present = false;
  bool color_info_present = false;
  std::uint8_t mlayer_info_idc = 1;  // global only; conformance forbids 3
  std::uint8_t reserved_2bits = 0;   // local only; must be 0
  std::vector<OpsPoint> points;      // ops_cnt entries; empty == reset
};

// ---------------------------------------------------------------------------
// Syntax writers (payload bits + extensible-OBU tail, no OBU header)
// ---------------------------------------------------------------------------

bool write_lcr_payload(const LayerConfigRecord& record,
                       std::vector<std::uint8_t>& out, std::string& error);
bool write_ops_payload(const OperatingPointSet& set,
                       std::vector<std::uint8_t>& out, std::string& error);

// Individual syntax structures, exposed for goldens and for callers that
// already own the surrounding bit writer.
bool write_lcr_aggregate_info(BitWriter& writer, const AggregateInfo& aggregate,
                              std::string& error);
bool write_lcr_seq_profile_tier_level_info(
    BitWriter& writer, const ProfileTierLevelInfo& info, std::string& error);
bool write_lcr_xlayer_color_info(BitWriter& writer,
                                 const ColorDescription& color,
                                 std::string& error);
bool write_lcr_rep_info(BitWriter& writer, const RepInfo& rep,
                        std::string& error);
bool write_lcr_embedded_layer_info(BitWriter& writer,
                                   const EmbeddedLayerInfo& embedded,
                                   bool atlas_segment_present,
                                   std::string& error);
bool write_lcr_xlayer_info(BitWriter& writer, const XLayerInfo& info,
                           bool is_global, bool atlas_id_present,
                           std::string& error);

bool write_ops_aggregate_info(BitWriter& writer,
                              const OpsAggregateInfo& aggregate,
                              std::string& error);
bool write_ops_seq_profile_tier_level_info(
    BitWriter& writer, const OpsProfileTierLevelInfo& info, std::string& error);
bool write_ops_decoder_model_info(BitWriter& writer,
                                  const OpsDecoderModelInfo& info,
                                  std::string& error);
bool write_ops_color_info(BitWriter& writer, const ColorDescription& color,
                          std::string& error);
bool write_ops_mlayer_info(BitWriter& writer, const OpsLayerMlayerInfo& info,
                           std::string& error);

// ---------------------------------------------------------------------------
// Syntax readers (inverse of the writers; validates the extensible tail)
// ---------------------------------------------------------------------------

bool read_lcr_payload(std::span<const std::uint8_t> payload, bool is_global,
                      LayerConfigRecord& record, std::string& error);
bool read_ops_payload(std::span<const std::uint8_t> payload,
                      std::uint8_t xlayer_id, OperatingPointSet& set,
                      std::string& error);

// ---------------------------------------------------------------------------
// Legacy-compatible construction (the svc_framing merge path)
// ---------------------------------------------------------------------------

// Builds the exact skeleton that `svc::write_layer_config_record_payload()`
// emits today: no aggregate/PTL/atlas/purpose/colour, embedded layer maps with
// TEXTURE_LAYER / VIEW_CENTER and same-resolution, atlas ids absent.
LayerConfigRecord lcr_from_legacy(const layers::LayerConfigRecord& record,
                                 const layers::LayerDependencyMaps& deps,
                                 bool include_full_info = false);

// Builds the exact skeleton that `svc::write_operating_point_set_payload()`
// emits today: no intent/PTL/colour/decoder model/delay, idc 1 (global) or
// reserved 0 (local).
OperatingPointSet ops_from_legacy(const layers::OperatingPointSet& set,
                                  bool include_full_info = false);

// ---------------------------------------------------------------------------
// Validation (§6.8 / §6.10 / §6.11)
// ---------------------------------------------------------------------------

// One activated sequence header, reduced to the fields LCR/OPS constrain.
struct XLayerSequenceInfo {
  std::uint8_t xlayer_id = 0;
  std::uint8_t seq_profile_idc = 0;
  std::uint8_t seq_level_idx = 0;
  bool seq_tier = false;
  std::uint8_t seq_max_mlayer_cnt = 1;  // seq_max_mlayer_cnt_minus_1 + 1
  std::uint32_t max_frame_width = 0;    // max_frame_width_minus_1 + 1
  std::uint32_t max_frame_height = 0;
  std::uint8_t bit_depth_idc = 0;
  std::uint8_t chroma_format_idc = 0;
  bool cropping_window_present = false;
  std::uint32_t crop_left = 0;
  std::uint32_t crop_right = 0;
  std::uint32_t crop_top = 0;
  std::uint32_t crop_bottom = 0;
  bool monotonic_output_order = true;
};

struct OpsSummary {
  std::uint8_t xlayer_id = kGlobalXLayerId;
  std::uint8_t ops_id = 0;
  std::uint8_t ops_cnt = 0;
};

struct StreamContext {
  std::vector<XLayerSequenceInfo> sequences;
  // Every OPS already present in the bitstream, including the one being
  // validated (needed for self-referential inheritance).
  std::vector<OpsSummary> known_ops;
  layers::LayerDependencyMaps dependencies =
      layers::default_dependency_maps(0, 0);
  bool dependencies_known = true;
  bool lcr_active = false;
  std::uint8_t active_global_lcr_id = 0;
};

struct ValidationReport {
  std::vector<std::string> errors;

  void add(std::string message) { errors.push_back(std::move(message)); }
  bool ok() const { return errors.empty(); }
};

// Layer-map closure (§6.8.9 / §6.10.7): every layer selected in a map must
// also select the layers it depends on in the activated sequence header.
bool validate_layer_map_closure(std::uint8_t xlayer_id,
                                std::uint32_t mlayer_map,
                                std::span<const std::uint8_t> tlayer_map,
                                const layers::LayerDependencyMaps& dependencies,
                                ValidationReport& report);

// §6.8: mandatory LCR rules, PTL/level consistency against the sequence
// headers, rep-info agreement, aggregate dominance, xlayer-map closure and
// the doh_constraint rule.
bool validate_lcr(const LayerConfigRecord& record,
                  const StreamContext& context, ValidationReport& report);

// §6.10: mandatory OPS rules, PTL/level consistency, aggregate dominance,
// idc-2 inheritance bounds and layer-map closure.
bool validate_ops(const OperatingPointSet& set, const StreamContext& context,
                  ValidationReport& report);

// Both, plus the cross-structure rules (local LCR parent, OPS xlayer subsets).
bool validate_stream(const LayerConfigRecord& record,
                     const OperatingPointSet& set,
                     const StreamContext& context, ValidationReport& report);

// §6.8: a local LCR must describe the same layer information as the activated
// global LCR for its obu_xlayer_id. This is the rule behind avmdec's
// "Local LCR xlayer_info for xlayer_id %d does not match the parent global
// LCR xlayer_info" message.
bool validate_local_lcr_matches_global(const LayerConfigRecord& global,
                                       const LayerConfigRecord& local,
                                       ValidationReport& report);

// §6.11: BRT <-> OPS correspondence. `brt_ops_dependent_flag == 0` has no OPS
// reference and is always accepted.
struct BrtCorrespondence {
  bool br_ops_dependent_flag = false;
  std::uint8_t br_ops_id = 0;   // f(4)
  std::uint8_t br_ops_cnt = 0;  // f(3), must equal ops_cnt
  std::uint32_t br_time = 0;    // non-dependent form, rg(4)
  std::vector<std::uint32_t> br_time_op;  // present BRT times, rg(4)
};

bool validate_brt_against_ops(const BrtCorrespondence& brt,
                              std::span<const OpsSummary> known_ops,
                              ValidationReport& report);

// Resolves idc == 2 inheritance for one (point, xlayer) pair from a list of
// already-received operating points.
struct OpsInheritanceSource {
  std::uint8_t ops_id = 0;
  std::uint8_t op_index = 0;
  std::uint8_t mlayer_map = 0;
  std::array<std::uint8_t, kMaxEmbeddedLayers> tlayer_map{};
};

bool resolve_ops_inheritance(
    const OpsLayerMlayerInfo& inherited,
    std::span<const OpsInheritanceSource> sources,
    OpsLayerMlayerInfo& resolved, std::string& error);

// ---------------------------------------------------------------------------
// Small helpers shared by callers and tests
// ---------------------------------------------------------------------------

std::size_t popcount32(std::uint32_t value);
bool global_xlayer(const OperatingPointSet& set);
bool global_xlayer(const LayerConfigRecord& record);

// (obu_xlayer_id, ops_id, ops_cnt) of a written or parsed OPS, for the BRT
// hook and for cross-checks.
OpsSummary summarize_ops(const OperatingPointSet& set);

}  // namespace av2enc::svc::ops_full

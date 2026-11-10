// SVC / scalable-layer data structures and validation.
//
// Spec-only: AV2 v1.0.0 layer machinery, used as structural metadata for the
// optional M4c layer path:
//   * §5.2.2 / §6.2.2   obu_header(): obu_tlayer_id, obu_mlayer_id, obu_xlayer_id
//   * §5.4.1 / §6.4.1   sequence header layer limits and dependency maps
//   * §5.8 / §6.8       layer config record (LCR)
//   * §5.10 / §5.11     operating point set (OPS) and operating point payload
//   * §7.3.6            coded extended layer unit ordering
//   * Annex F.3.1       operating-point selection / retention map
//
// No SVC stream is emitted here and no SVC quality claim is made. The project
// treats layers as delivery/scalability packaging, not as a compression gain
// (`docs/bitrate-features.md` §7). This module exists so a future layer
// scheduler can reject structurally invalid combinations before writing OBUs.
//
// ID ranges: obu_xlayer_id is 5 bits (0..31), GLOBAL_XLAYER_ID == 31;
// obu_mlayer_id is 3 bits (0..7); obu_tlayer_id is 2 bits (0..3).

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace av2enc::tiles::svc {

inline constexpr int kGlobalXLayerId = 31;
inline constexpr int kMaxExtendedLayerId = 30;
inline constexpr int kExtendedLayerCount = 32;
inline constexpr int kMaxEmbeddedLayerId = 7;
inline constexpr int kEmbeddedLayerCount = 8;
inline constexpr int kMaxTemporalLayerId = 3;
inline constexpr int kTemporalLayerCount = 4;

struct LayerId {
  std::uint8_t xlayer_id = 0;
  std::uint8_t mlayer_id = 0;
  std::uint8_t tlayer_id = 0;
};

bool operator==(const LayerId& a, const LayerId& b);
bool operator<(const LayerId& a, const LayerId& b);
bool validate_layer_id(const LayerId& id, std::string& error);
const char* layer_id_name(const LayerId& id);  // short-lived static buffer

// §5.4.1 dependency arrays. MLayerDependencyMap[curr][ref] and
// TLayerDependencyMap[mLayer][currT][refT]. The derived presence map and the
// encoder's validation use the same closure rule:
//   a selected layer must also select every layer it depends on.
using MLayerDependencyMap = std::array<std::array<bool, kEmbeddedLayerCount>,
                                       kEmbeddedLayerCount>;
using TLayerDependencyMap = std::array<std::array<bool, kTemporalLayerCount>,
                                       kTemporalLayerCount>;

struct LayerDependencyMaps {
  MLayerDependencyMap mlayer{};
  std::array<TLayerDependencyMap, kEmbeddedLayerCount> tlayer{};
};

LayerDependencyMaps default_dependency_maps(int max_mlayer_id,
                                            int max_tlayer_id);
bool validate_dependency_maps(const LayerDependencyMaps& maps,
                              int max_mlayer_id, int max_tlayer_id,
                              std::string& error);

// Returns true when every dependency of every present embedded layer is also
// present. Bits outside 0..7 are ignored but rejected separately by callers.
bool mlayer_dependencies_satisfied(std::uint32_t mlayer_map,
                                   const LayerDependencyMaps& maps,
                                   std::string& error);

bool tlayer_dependencies_satisfied(
    std::uint32_t mlayer_map,
    const std::array<std::uint8_t, kEmbeddedLayerCount>& tlayer_map,
    const LayerDependencyMaps& maps, std::string& error);

struct SequenceLayerConfig {
  bool single_picture_header = false;
  bool still_picture = false;
  std::uint8_t seq_lcr_id = 0;       // f(3)
  std::uint8_t max_tlayer_id = 0;    // f(2)
  std::uint8_t max_mlayer_id = 0;    // f(3)
  std::uint8_t seq_max_mlayer_cnt = 1;  // seq_max_mlayer_cnt_minus_1 + 1
  int max_width = 0;
  int max_height = 0;
  LayerDependencyMaps dependencies;
};

// single_picture_header_flag forces seq_lcr_id = 0, max_tlayer_id = 0,
// max_mlayer_id = 0, SeqMaxMlayerCnt = 1 (§5.4.1). Keep this construction in
// one place so the M0 still path cannot accidentally advertise layers.
SequenceLayerConfig default_sequence_layer_config(int max_mlayer_id,
                                                  int max_tlayer_id,
                                                  bool single_picture_header);

bool validate_sequence_layer_config(const SequenceLayerConfig& config,
                                    std::string& error);

struct AggregateInfo {
  bool present = false;
  std::uint8_t config_idc = 0;   // f(6)
  std::uint8_t level_idx = 0;    // f(5)
  bool max_tier_flag = false;    // f(1)
  std::uint8_t max_interop = 0;  // f(4)
};

struct EmbeddedLayerConfig {
  std::uint8_t mlayer_id = 0;
  std::uint8_t tlayer_map = 0;  // bits 0..3
  bool same_sh_max_resolution = true;
  std::uint32_t max_expected_width = 0;
  std::uint32_t max_expected_height = 0;
  bool has_priority_order = false;
  std::uint32_t priority_order = 0;
};

struct ExtendedLayerConfig {
  std::uint8_t xlayer_id = 0;      // 0..30
  std::uint32_t mlayer_map = 0;    // bits 0..7
  std::array<std::uint8_t, kEmbeddedLayerCount> tlayer_map{};
  bool dependent_xlayers = false;
  std::vector<std::uint8_t> dependent_xlayer_ids;
  std::vector<EmbeddedLayerConfig> embedded;
};

struct LayerConfigRecord {
  bool is_global = true;
  std::uint8_t xlayer_id = kGlobalXLayerId;
  std::uint8_t global_config_record_id = 0;  // f(3)
  AggregateInfo aggregate;
  std::uint32_t xlayer_map = 0;  // global LCR f(31): bits 0..30
  std::vector<ExtendedLayerConfig> layers;
};

bool validate_layer_config_record(const LayerConfigRecord& record,
                                  const SequenceLayerConfig& sequence,
                                  std::string& error);

// One operating point. For a global OPS, `xlayer_map` is ops_xlayer_map f(31)
// and the per-extended-layer maps are indexed by xlayer_id. For a local OPS,
// `xlayer_map` must be exactly 1 << xlayer_id and only that entry is used.
struct OperatingPoint {
  std::uint8_t op_index = 0;  // 0..ops_cnt-1 (f(3) ops_cnt)
  std::uint32_t xlayer_map = 0;
  std::array<std::uint32_t, kExtendedLayerCount> mlayer_map{};
  std::array<std::array<std::uint8_t, kEmbeddedLayerCount>,
             kExtendedLayerCount>
      tlayer_map{};
};

struct OperatingPointSet {
  std::uint8_t xlayer_id = kGlobalXLayerId;  // 0..31
  std::uint8_t ops_id = 0;                   // f(4)
  std::vector<OperatingPoint> points;        // ops_cnt entries
};

bool validate_operating_point_set(const OperatingPointSet& set,
                                  const SequenceLayerConfig& sequence,
                                  const LayerConfigRecord* record,
                                  std::string& error);

// Annex F.3.1 retention map. `retain[xlayer][mlayer][tlayer]` says which coded
// layers an extraction process must keep to decode the selected operating
// point. Dependencies are added transitively.
struct RetentionMap {
  std::array<std::array<std::array<bool, kTemporalLayerCount>,
                        kEmbeddedLayerCount>,
             kExtendedLayerCount>
      retain{};
  std::array<bool, kExtendedLayerCount> xlayer_selected{};
};

bool build_retention_map(const OperatingPointSet& set, int op_index,
                         const LayerDependencyMaps& maps,
                         RetentionMap& retention, std::string& error);

}  // namespace av2enc::tiles::svc

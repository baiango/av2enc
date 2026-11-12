// Embedded-layer (mlayer) planning, dependency maps and stream validation.
//
// Spec-only component for AV2 v1.0.0:
//   §5.4.1 p55-58  seq_max_mlayer_cnt_minus_1, mlayer_dependency_* ,
//                   multi_tlayer_dependency_map_present_flag, tlayer maps,
//                   derived MLayerDependencyMap / MLayerPresenceMap /
//                   TLayerDependencyMap
//   §6.4.1 p261-264 semantics and bitstream-conformance requirements
//   §6.2.2 p257     obu_header() layer-id constraints
//   §7.3.6 p383     coded extended layer unit ordering
//   §7.3.7 p385     temporal-unit extended-layer ordering
//   §7.3.8.6 p387   sequence-header availability
//
// Nothing outside src/encoder/inter/mlayers/ is edited. The byte writers and
// independent readers live in mlayers_syntax.h/.cpp.
//
// ID ranges: obu_mlayer_id f(3) = 0..7, obu_tlayer_id f(2) = 0..3,
// obu_xlayer_id f(5) = 0..31 with 31 = GLOBAL_XLAYER_ID.

#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace av2enc::inter::mlayers {

inline constexpr int kMaxMlayerId = 7;
inline constexpr int kMaxTlayerId = 3;
inline constexpr int kMlayerCount = 8;
inline constexpr int kTlayerCount = 4;
inline constexpr int kGlobalXLayerId = 31;
inline constexpr int kMaxXLayerId = 30;

// MLayerDependencyMap[curr][ref] (§5.4.1 p57).
using MLayerDependencyMap =
    std::array<std::array<bool, kMlayerCount>, kMlayerCount>;
// MLayerPresenceMap[curr][ref] (§5.4.1 p58 derivation).
using MLayerPresenceMap =
    std::array<std::array<bool, kMlayerCount>, kMlayerCount>;
// TLayerDependencyMap[curr][ref] for one embedded layer.
using TLayerDependencyMap =
    std::array<std::array<bool, kTlayerCount>, kTlayerCount>;
// TLayerDependencyMaps[mlayer][curr][ref].
using TLayerDependencyMaps =
    std::array<TLayerDependencyMap, kMlayerCount>;

struct LayerId {
  std::uint8_t xlayer_id = 0;
  std::uint8_t mlayer_id = 0;
  std::uint8_t tlayer_id = 0;
};

bool operator==(const LayerId& a, const LayerId& b);
bool operator<(const LayerId& a, const LayerId& b);
bool validate_layer_id(const LayerId& id, std::string& error);
const char* layer_id_name(const LayerId& id);  // short-lived static buffer

struct LayerDependencyMaps {
  MLayerDependencyMap mlayer{};
  TLayerDependencyMaps tlayer{};
};

// §5.4.1 p57 defaults:
//   MLayerDependencyMap[c][r]          = r <= c && c <= max_mlayer_id
//   TLayerDependencyMap[m][c][r]       = r <= c && c <= max_tlayer_id
//                                        && m <= max_mlayer_id
LayerDependencyMaps default_dependency_maps(int max_mlayer_id,
                                            int max_tlayer_id);

// Exact §5.4.1 p58 MLayerPresenceMap derivation. Entries outside the declared
// layer range are zero because the dependency map is required to be zero there.
MLayerPresenceMap derive_mlayer_presence_map(
    const MLayerDependencyMap& mlayer, int max_mlayer_id);

// Checks range, reflexivity and transitivity. `mlayer` and every `tlayer[m]`
// must be lower-triangular (ref <= curr); every declared layer is reflexive;
// entries for undeclared layers are zero.
bool validate_mlayer_dependency_map(const MLayerDependencyMap& map,
                                    int max_mlayer_id, std::string& error);
bool validate_tlayer_dependency_maps(const TLayerDependencyMaps& maps,
                                     int max_mlayer_id, int max_tlayer_id,
                                     std::string& error);
bool validate_dependency_maps(const LayerDependencyMaps& maps,
                              int max_mlayer_id, int max_tlayer_id,
                              std::string& error);

// A selected set of layers must satisfy every dependency it selects. The maps
// are checked transitively by validate_dependency_maps(), so this is the
// per-selection availability rule used by LCR/OPS and by the stream auditor.
bool mlayer_dependencies_satisfied(std::uint32_t mlayer_map,
                                   const LayerDependencyMaps& maps,
                                   std::string& error);
bool tlayer_dependencies_satisfied(
    std::uint32_t mlayer_map,
    const std::array<std::uint8_t, kMlayerCount>& tlayer_map,
    const LayerDependencyMaps& maps, std::string& error);

// §5.4.1 p56-58 fields that this component serialises. seq_max_mlayer_cnt is
// SeqMaxMlayerCnt (the value plus one of seq_max_mlayer_cnt_minus_1).
struct SequenceLayerConfig {
  int max_mlayer_id = 0;  // f(3)
  int max_tlayer_id = 0;  // f(2)
  int seq_max_mlayer_cnt = 1;  // 1..max_mlayer_id+1
  bool mlayer_dependency_present_flag = false;
  bool tlayer_dependency_present_flag = false;
  bool multi_tlayer_dependency_map_present_flag = false;
  LayerDependencyMaps dependencies;
};

// Defaults from max_mlayer_id / max_tlayer_id and the inferred dependency map.
SequenceLayerConfig default_sequence_layer_config(int max_mlayer_id,
                                                  int max_tlayer_id);

// Checks field ranges, the SeqMaxMlayerCnt <= max_mlayer_id+1 rule (§6.4.1
// p261), flag/map consistency and the map invariants above.
bool validate_sequence_layer_config(const SequenceLayerConfig& config,
                                    std::string& error);

// ---------------------------------------------------------------------------
// §7.3.6 / §7.3.7 emission-order and availability audit
// ---------------------------------------------------------------------------

// One OBU in emission order. The caller fills only the fields that apply:
// sequence headers use seq_header_id/header_*; frame OBUs use the frame fields.
struct LayerObuRecord {
  int obu_type = 0;  // §6.2.2 Table 6.1
  LayerId layer;     // inferred ids after §5.2.2
  bool first_tile_group = true;
  bool frame_unit_start = true;  // false = continuation tile group
  bool coded_output_frame = true;
  bool key_frame = false;         // OBU_CLOSED_LOOP_KEY
  bool open_loop_key = false;     // OBU_OPEN_LOOP_KEY
  bool switch_or_ras = false;     // OBU_SWITCH / OBU_RAS_FRAME
  int order_hint = -1;            // required for output frame units
  int seq_header_id = -1;         // sequence header: this id
  int header_max_mlayer_id = 0;   // sequence header: parsed maxima
  int header_max_tlayer_id = 0;
  int seq_header_ref = -1;        // frame header seq_header_id_in_frame_header
  // Optional inter-layer prediction edge. When enabled, the referenced layer
  // must be present in MLayerPresenceMap[curr] and must already have an output
  // frame unit in the same temporal unit.
  bool has_inter_layer_ref = false;
  LayerId inter_layer_ref;
};

// Incremental validator. Feed every OBU of a bitstream in emission order.
// `observe()` returns false on the first violation; `finish()` closes the last
// temporal unit. The auditor never mutates the caller's config.
class LayerStreamAuditor {
 public:
  explicit LayerStreamAuditor(SequenceLayerConfig config);

  bool observe(const LayerObuRecord& record, std::string& error);
  bool finish(std::string& error);

  // Compatibility streams that cannot signal SeqMaxMlayerCnt (for example the
  // pinned-dav2d dialect) may disable only the §6.4.1 p261 distinct-layer
  // count rule. Every other check remains active. Default: enabled.
  void set_enforce_seq_max_mlayer_cnt(bool enabled) {
    enforce_seq_max_mlayer_cnt_ = enabled;
  }

  const SequenceLayerConfig& config() const { return config_; }

 private:
  struct TuvRecord {
    LayerObuRecord record;
    bool frame_unit_start = false;
  };

  bool finish_temporal_unit(std::string& error);
  bool check_obu_constraints(const LayerObuRecord& record,
                             std::string& error) const;
  bool check_availability(const LayerObuRecord& record,
                          std::string& error);

  SequenceLayerConfig config_;
  MLayerPresenceMap presence_{};
  std::array<TuvRecord, 1024> records_{};
  std::size_t record_count_ = 0;
  bool have_temporal_unit_ = false;
  bool enforce_seq_max_mlayer_cnt_ = true;
  // Active sequence headers, keyed by (xlayer, seq_header_id). A small fixed
  // table is enough for the single-CVS component streams.
  struct ActiveHeader {
    int xlayer_id = 0;
    int seq_header_id = 0;
    int max_mlayer_id = 0;
    int max_tlayer_id = 0;
    bool valid = false;
  };
  std::array<ActiveHeader, 64> headers_{};
};

// Human-readable diagnostic helpers.
const char* obu_type_name(int obu_type);

}  // namespace av2enc::inter::mlayers

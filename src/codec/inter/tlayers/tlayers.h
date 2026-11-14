#ifndef AV2ENC_INTER_TLAYERS_TLAYERS_H_
#define AV2ENC_INTER_TLAYERS_TLAYERS_H_

// Temporal-layer planning for AV2 v1.0.0 hierarchical GOPs.
//
// Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7), §5.4.1 (max_tlayer_id,
// tlayer_dependency_present_flag, tlayer_dependency_map,
// multi_tlayer_dependency_map_present_flag), §5.2.2 (obu_tlayer_id), §7.3.7
// (temporal unit), §5.18.2 (order_hint, get_disp_order_hint), §7.7/§7.23
// (reference selection and refresh).
//
// This directory owns planning and syntax for temporal layers. It does NOT
// edit M1/M2 core files, the existing inter components, src/main.cpp or
// CMakeLists.txt. M2 integration sites are documented in VIDEO_INTEGRATION.md.
//
// Key spec facts pinned by this layer:
//   * obu_tlayer_id is f(2) and is present in every OBU header byte.
//   * max_tlayer_id is f(2), so at most four temporal layers (ids 0..3) are
//     representable in a sequence header.
//   * TLayerDependencyMap[mlayer][curr][ref] is initialised to
//     (ref <= curr && curr <= max_tlayer_id && mlayer <= max_mlayer_id).
//     A frame may only reference a slot whose RefTLayerId is mapped to 1.
//   * A frame's own tlayer id must be <= max_tlayer_id, and a reference from
//     layer C to layer R requires TLayerDependencyMap[0][C][R] == 1.
//   * OBU_SEQUENCE_HEADER and OBU_TEMPORAL_DELIMITER must carry
//     obu_tlayer_id == 0; key/switch/RAS frame OBUs must also carry 0.

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace av2enc::inter::tlayers {

// §5.4.1: max_tlayer_id f(2) => values 0..3. The spec's MAX_NUM_TLAYERS is 4.
inline constexpr int kMaxTLayers = 4;
inline constexpr int kMaxTemporalLayerId = kMaxTLayers - 1;
// NUM_REF_FRAMES.
inline constexpr int kNumRefFrames = 16;
// This layer schedules at most two references per frame.
inline constexpr int kMaxRefs = 2;

enum class Schedule {
  // Dyadic reordered P pyramid. Every frame references the immediately
  // preceding frame in coding order, so a single-reference encoder (the
  // current M2 block layer) can carry it. Example gop_size = 8 (columns are
  // coding order):
  //   coding 0 4 2 6 1 3 5 7
  //   layer  0 0 1 1 2 2 2 2
  //   refs   - 0 4 2 6 1 3 5
  kHierarchicalP,
  // Dyadic B pyramid with two references per interior frame. Not encodable by
  // the current single-reference M2 block layer; planned and validated here so
  // the header/layer model is ready. Example gop_size = 8 (coding order):
  //   coding 0 7 3 1 5 2 4 6
  //   layer  0 0 1 2 2 3 3 3
  //   refs   - 0 0,7 0,3 3,7 1,3 3,5 5,7
  kHierarchicalB,
};

enum class DependencyMode {
  // Code exactly the direct reference relationships of the plan (plus the
  // implied map[0][0] = 1). This produces a non-trivial, minimal map.
  kDirect,
  // Code the transitive closure, which for these schedules is every lower
  // layer: TLayerDependencyMap[c][r] = (r <= c).
  kAllLower,
};

// Hierarchical-P reference mode.
enum class RefMode {
  // Historical carrier: every frame after the anchor references the previous
  // coding-order frame.  §7.9 returns an empty motion field for these frames,
  // so use_ref_frame_mvs is inert but decodable.
  kSingle = 0,
  // Two references per interior frame: the two nearest already-coded pictures
  // whose temporal layer is <= this frame's layer (so the default
  // TLayerDependencyMap[c][r] = (r <= c) admits every reference).  This
  // populates §7.22 fields from two directions so §7.9 can project real
  // candidates.  The first inter picture has only the key available and keeps
  // one reference.
  kTwo = 1,
};

struct PlanConfig {
  // Power of two. kHierarchicalP accepts 2..16; kHierarchicalB accepts 2..8
  // because max_tlayer_id is f(2) and a 16-frame B pyramid needs layer 4.
  int gop_size = 8;
  Schedule schedule = Schedule::kHierarchicalB;
  DependencyMode dependency = DependencyMode::kAllLower;
  // Hierarchical-P reference count.  kTwo leaves the coding order and the
  // layer assignment untouched and only widens the per-frame reference list,
  // so the schedule/payload stays deterministic.
  RefMode ref_mode = RefMode::kSingle;
  // Code the TLayerDependencyMap explicitly. The pinned dav2d nightly
  // (f9dafbc) parses a different, later dependency syntax, so the decoder gate
  // uses the spec's default map (flag = 0) with DependencyMode::kAllLower.
  // Set true only for spec-exact explicit-map streams.
  bool explicit_dependency_map = false;
  // Absolute display order hint of display index 0.
  int start_order_hint = 0;
  // OrderHintBits written in the sequence header (§5.4.6).
  int order_hint_bits = 4;
  // NumRefFrames (§5.4.6), 1..16.
  int num_slots = 8;
  // Stamped into the simulation; must match the frame-header base_q_idx.
  std::uint8_t base_q_idx = 90;
};

struct PlannedFrame {
  int display_index = 0;
  int coding_index = 0;
  int tlayer_id = 0;
  bool is_key = false;
  bool is_anchor = false;  // P anchor (layer 0, no B split below it)

  // Absolute display order hint (start_order_hint + display_index).
  std::uint32_t display_order_hint = 0;
  // Value written with f(OrderHintBits). Derived by replaying
  // get_disp_order_hint() against the live reference store.
  std::uint32_t order_hint_lsbs = 0;

  int num_refs = 0;
  int ref_display[kMaxRefs] = {-1, -1};
  int ref_slot[kMaxRefs] = {-1, -1};
  std::uint32_t ref_order_hint[kMaxRefs] = {0, 0};

  // §7.23 frame store slot refreshed by this frame.
  int slot = 0;
  std::uint32_t refresh_frame_flags = 0;

  // §5.18.2 output flags. With monotonic_output_order_flag == 0 (required for
  // reordered streams) every frame except the first coded frame uses
  // implicit_output_frame.
  bool immediate_output_frame = false;
  bool implicit_output_frame = false;

  // §5.2.2 obu_tlayer_id for the frame OBU that carries this frame.
  int obu_tlayer_id() const { return tlayer_id; }
};

struct Plan {
  PlanConfig config;
  int max_tlayer_id = 0;
  int num_layers = 1;
  int layer_frame_count[kMaxTLayers] = {0, 0, 0, 0};
  int num_slots = 0;

  // frames[display_index]
  std::vector<PlannedFrame> frames;
  // Display indices in encode order.
  std::vector<int> coding_order;

  // TLayerDependencyMap[0][curr][ref]. Entry [0][0] is implied by §5.4.1 and
  // is never coded; the writer/reader still expose it as true.
  std::array<std::array<bool, kMaxTLayers>, kMaxTLayers> tlayer_dependency{};

  bool tlayer_dependency_present_flag = false;
  // max_mlayer_id == 0 in this layer, so §5.4.1 infers this as 0 and codes no
  // bit. Kept explicit for the API and the integration notes.
  bool multi_tlayer_dependency_map_present_flag = false;

  int display_count() const { return static_cast<int>(frames.size()); }
  int frames_in_layer(int layer) const {
    return (layer >= 0 && layer < kMaxTLayers) ? layer_frame_count[layer] : 0;
  }
  int layer_of(int display_index) const {
    return frames[static_cast<std::size_t>(display_index)].tlayer_id;
  }
  // §5.2.2 obu_tlayer_id for the frame OBU that carries this display index.
  // OBU_TEMPORAL_DELIMITER / OBU_SEQUENCE_HEADER / key frame OBUs use 0.
  int obu_tlayer_id(int display_index) const { return layer_of(display_index); }

  // Coding order filtered to obu_tlayer_id <= max_layer. Used by the decoder
  // drop test and by temporal sub-stream extraction.
  std::vector<int> coding_order_up_to_layer(int max_layer) const;
};

enum class PlanError {
  kOk = 0,
  kBadGopSize,
  kBadSchedule,
  kBadOrderHintBits,
  kBadStartOrderHint,
  kTooManyLayers,
  kBadSlotCount,
  kNotEnoughSlots,
  kReferenceOutOfRange,
  kReferenceNotDecoded,
  kReferenceLayerViolation,
  kReferenceSlotInvalid,
  kDuplicateReference,
  kKeyFrameHasRefs,
  kMissingKeyFrame,
  kRefreshFlagsInvalid,
  kOutputFlagsInvalid,
  kOrderHintReconstruct,
  kDependencyMapIncomplete,
  kBadCodingOrder,
  kBadDisplayOrder,
  kInternal,
};

// Builds and self-validates a complete GOP. Returns false and sets *error on
// failure. The explicit gate is validate_plan(); callers must not write bits
// unless it passes.
bool build_plan(const PlanConfig& config, Plan& plan, PlanError* error);

// Structural validation + full decoder-side replay of the coding order:
// reference availability, slot identity, TLayerDependencyMap, §7.23 refresh and
// get_disp_order_hint() reconstruction.
bool validate_plan(const Plan& plan, PlanError* error, int* bad_display_index,
                   int* bad_coding_index);

// Checks that dropping every frame with tlayer_id > max_layer leaves a
// decodable prefix: retained frames may only reference retained frames, and
// each reference satisfies the dependency map. This is the invariant behind
// the decoder drop test.
bool validate_layer_drop(const Plan& plan, int max_layer, PlanError* error);

const char* plan_error_name(PlanError error);
const char* schedule_name(Schedule schedule);
const char* dependency_mode_name(DependencyMode mode);

}  // namespace av2enc::inter::tlayers

#endif  // AV2ENC_INTER_TLAYERS_TLAYERS_H_

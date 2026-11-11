// Encoder-side scalable-layer configuration, operating points, and retention.
//
// This module turns the structural SVC data structures in
// `src/encoder/tiles/svc_layers.h` into a validated encoder plan:
//
//   * spatial layers   -> obu_xlayer_id 0..N-1 (one sequence header each)
//   * embedded layers  -> obu_mlayer_id 0..M-1 (M = 1 by default)
//   * temporal layers  -> obu_tlayer_id 0..T-1
//
// Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
// (commit 38f28e7), §5.2.2, §5.4.1, §5.8, §5.10, §5.11, §7.3.6, Annex F.3.1.
//
// Scope note: AV2 layers are delivery/scalability packaging. This pipeline
// implements the **spatial-only path**: every extended layer is coded
// independently (its own sequence header and frame units) and no sample, MV,
// or coefficient prediction crosses an xlayer boundary. That matches the
// spec's separate per-xlayer decode contexts (`load_xlayer_context()` /
// `save_xlayer_context()`); see `svc_pipeline.h` for the explicit reference
// policy that enforces it.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "tiles/svc_layers.h"

namespace av2enc::svc {

namespace layers = ::av2enc::tiles::svc;

// Non-global xlayer ids are 0..30; GLOBAL_XLAYER_ID == 31.
inline constexpr int kMaxSpatialLayers = 30;
// ops_cnt is f(3) and must be in 1..7.
inline constexpr int kMaxOperatingPoints = 7;
inline constexpr int kMaxEmbeddedLayers = layers::kEmbeddedLayerCount;
inline constexpr int kMaxTemporalLayers = layers::kTemporalLayerCount;

// One spatial layer. Dimensions are the layer's coded (uncropped) size and are
// written into that layer's sequence header maximum frame size.
struct SpatialLayerSpec {
  std::uint8_t xlayer_id = 0;
  int width = 0;
  int height = 0;
};

// Top-level encoder configuration for a layered encode.
//
// `spatial_layers` must start at xlayer 0 and use contiguous ascending ids:
// the assignment is deterministic and not user-orderable. If a caller needs a
// non-contiguous set it must build the plan itself and call
// validate_layer_plan().
struct LayerConfig {
  std::vector<SpatialLayerSpec> spatial_layers;
  std::uint8_t temporal_layers = 1;   // 1..4
  std::uint8_t embedded_layers = 1;   // 1..8
  // A still-picture still stream forces one layer and t/m = 0 (§5.4.1). Real
  // layered video sets both flags false.
  bool single_picture_header = false;
  bool still_picture = false;
  // Associated LCR id (f(3)). 0 means "no LCR association" and is only legal
  // for single-picture stills.
  std::uint8_t seq_lcr_id = 1;
  // Optional dependency-map override applied to every spatial layer. When
  // absent, the default reflexive/transitive chain for max_mlayer_id /
  // max_tlayer_id is used. Custom maps must still be reflexive and transitive
  // (§5.4.1) and must not reference a layer above the configured maximum.
  std::optional<layers::LayerDependencyMaps> dependencies;
};

// One coded layer in deterministic emission order: xlayer, then mlayer, then
// tlayer, each ascending.
struct LayerSpec {
  layers::LayerId id;
  int width = 0;
  int height = 0;
};

// Decoder-facing view of one operating point.
struct OperatingPointInfo {
  std::uint8_t op_index = 0;
  std::uint32_t xlayer_map = 0;
  int spatial_layers = 0;
  int embedded_layers = 0;
  int temporal_layers = 0;
  std::string label;
};

// Fully validated layer plan. `sequences` is indexed by spatial layer position
// (0 == base), not by xlayer id; the two coincide for a valid plan.
struct LayerPlan {
  LayerConfig config;
  std::vector<layers::SequenceLayerConfig> sequences;
  layers::LayerConfigRecord lcr;
  layers::OperatingPointSet ops;
  std::vector<LayerSpec> layers;
  std::vector<OperatingPointInfo> operating_points;
};

// Builds the plan:
//   * one sequence header per spatial layer with that layer's max dimensions;
//   * a global LCR with a complete xlayer/mlayer/tlayer map;
//   * one cumulative operating point per spatial layer
//     (OP i selects xlayers 0..i, all embedded and temporal layers);
//   * the deterministic layer list.
// Rejects every structural inconsistency with a specific message.
bool build_layer_plan(const LayerConfig& config, LayerPlan& plan,
                      std::string& error);

// Revalidates an existing plan (sequence headers, LCR, OPS, layer list) with
// the tiles validators plus plan-level consistency checks.
bool validate_layer_plan(const LayerPlan& plan, std::string& error);

// Deterministic frame-to-layer assignment.
//
// A source frame is assigned to the unique spatial layer whose coded size
// matches (width, height), at the requested embedded/temporal layer. Ambiguous
// sizes (two spatial layers at the same resolution) are rejected rather than
// resolved by order. Embedded-layer ids above the configured maximum are
// rejected.
struct FrameSource {
  int temporal_unit = 0;
  std::uint8_t tlayer = 0;
  int width = 0;
  int height = 0;
  int frame_index = -1;
  std::uint8_t mlayer = 0;
};

bool assign_layers(const LayerPlan& plan,
                   const std::vector<FrameSource>& sources,
                   std::vector<layers::LayerId>& assigned, std::string& error);

// Selection modes mirror the product surface:
//   kAllLayers            --alllayers / full bitstream
//   kOperatingPoint       --oppoint <index>
//   kUpToSpatialLayer     --spatial-layers <count> (cumulative)
enum class SelectionMode {
  kAllLayers,
  kOperatingPoint,
  kUpToSpatialLayer,
};

struct LayerSelection {
  SelectionMode mode = SelectionMode::kAllLayers;
  int op_index = 0;
  int max_xlayer = -1;  // inclusive; only for kUpToSpatialLayer
};

// Fills an Annex F.3.1 retention map. Global OBUs (TD/LCR/OPS at
// GLOBAL_XLAYER_ID, m0/t0) are marked retained whenever the plan carries them,
// matching Annex F.3.1 Step 2. Rejects selections that do not correspond to
// the plan.
bool resolve_selection(const LayerPlan& plan, const LayerSelection& selection,
                       layers::RetentionMap& retention, std::string& error);

bool retention_has_layer(const layers::RetentionMap& retention,
                         const layers::LayerId& id);

// Number of retained (xlayer, mlayer, tlayer) triples, excluding the global
// OBU layer.
int selected_layer_count(const layers::RetentionMap& retention,
                         const LayerPlan& plan);

// Checks that the retention map is dependency-closed (Annex F.3.1) and only
// mentions layers present in the plan. The global OBU entry is exempt.
bool validate_retention_closed(const LayerPlan& plan,
                               const layers::RetentionMap& retention,
                               std::string& error);

// Helpers used by the framing/pipeline layers.
bool find_layer(const LayerPlan& plan, const layers::LayerId& id,
                const LayerSpec** spec);
bool layer_is_present(const LayerPlan& plan, const layers::LayerId& id);

}  // namespace av2enc::svc

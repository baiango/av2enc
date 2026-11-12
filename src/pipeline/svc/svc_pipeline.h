// Layer emission order and inter-layer reference policy.
//
// This is the encoder-side scheduler view of §7.3.6 / §7.3.7:
//
//   temporal unit
//     TD (global)
//     [global LCR, global OPS]              (first temporal unit only)
//     for each xlayer ascending:
//       sequence header                     (start of the layer's CVS)
//       for each mlayer ascending:
//         non-output frame units
//         output frame unit
//       (tlayer ascending inside each mlayer)
//
// Spatial-only path: the plan declares no xlayer dependency, so every layer is
// independently decodable and a reference that crosses xlayers is rejected.
// That is a deliberate product limitation, not an unimplemented edge case: AV2
// specifies separate per-xlayer decode contexts and this encoder does not
// implement cross-layer sample/MV/coefficient prediction.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "svc_config.h"

namespace av2enc::svc {

enum class EmissionKind {
  kTemporalDelimiter,
  kLayerConfigRecord,
  kOperatingPointSet,
  kSequenceHeader,
  kFrame,
};

struct Emission {
  EmissionKind kind = EmissionKind::kFrame;
  layers::LayerId layer;
  int temporal_unit = -1;  // -1 for global configuration OBUs
  int frame_index = -1;
  int operating_point_index = -1;
  bool output = true;  // frames only: output (1) vs non-output (0) unit
};

struct ReferencePolicy {
  // AV2 v1.0.0 layer machinery in this encoder is packaging, not inter-layer
  // prediction. Leave these false unless a future decoder-gated design adds a
  // real cross-layer predictor.
  bool cross_xlayer_prediction = false;
  bool cross_mlayer_prediction = false;
  // Normal temporal prediction across tlayers inside the same x/m is allowed.
  bool cross_tlayer_prediction = true;
};

enum class FrameRole {
  kIntra,
  kInter,
};

struct FrameRequest {
  int temporal_unit = 0;
  layers::LayerId layer;
  FrameRole role = FrameRole::kIntra;
  int frame_index = -1;
  bool output = true;
};

struct FrameReference {
  int temporal_unit = 0;
  layers::LayerId target;
  int reference_temporal_unit = 0;
  layers::LayerId reference;
};

// Default schedule for a video with `temporal_units` identical temporal units:
// every configured layer produces one output frame per unit. Deterministic.
bool plan_emission_order(const LayerPlan& plan, int temporal_units,
                         std::vector<Emission>& order, std::string& error);

// Custom schedule driven by a GOP/temporal-layer scheduler. Emits one global
// TD per temporal unit plus the frame requests, sorted deterministically by
// (temporal unit, xlayer, mlayer, tlayer, non-output before output). Config
// OBUs are not added here; the caller emits them once at the start.
bool order_frames(const LayerPlan& plan, std::vector<FrameRequest> requests,
                  std::vector<Emission>& order, std::string& error);

// True when `reference` may be used to predict `target` under `policy`.
// Rejections explain the concrete rule that failed.
bool reference_allowed(const LayerPlan& plan, const ReferencePolicy& policy,
                       const layers::LayerId& target,
                       const layers::LayerId& reference, std::string& error);

// Checks every explicit reference in a schedule, including that the reference
// is emitted before the target in `order`.
bool validate_frame_references(const LayerPlan& plan,
                               const ReferencePolicy& policy,
                               const std::vector<FrameReference>& references,
                               const std::vector<Emission>& order,
                               std::string& error);

// Number of frame units in a schedule (excludes TD/config OBUs).
std::size_t frame_emission_count(const std::vector<Emission>& order);

}  // namespace av2enc::svc

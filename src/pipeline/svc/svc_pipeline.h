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
// Cross-layer rules: §6.8.2 requires lcr_dependent_xlayers_flag = 0, so a
// cross-xlayer reference can never be signalled and is rejected here
// unconditionally.  Embedded-layer refinement (§5.4.1 MLayerDependencyMap)
// predicts from a lower mlayer of the same xlayer and is allowed.

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
  // AV2 v1.0.0 §6.8.2 makes lcr_dependent_xlayers_flag = 0 a conformance
  // requirement, so cross-xlayer (inter-layer) prediction is unreachable for
  // every conforming stream.  The field is retained for source compatibility;
  // reference_allowed() rejects cross-xlayer references unconditionally.
  bool cross_xlayer_prediction = false;
  // Embedded-layer refinement (--svc-embedded) predicts from a lower mlayer of
  // the same xlayer.  §5.4.1 MLayerDependencyMap / §6.17.2 allow it, and the
  // encoder's refinement sequence writes the matching reference slots and
  // refresh masks.
  bool cross_mlayer_prediction = true;
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

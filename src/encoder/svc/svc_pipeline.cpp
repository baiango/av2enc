#include "svc_pipeline.h"

#include <algorithm>
#include <sstream>
#include <tuple>
#include <utility>

namespace av2enc::svc {
namespace {

bool fail(std::string& error, std::string message) {
  error = std::move(message);
  return false;
}

layers::LayerId global_layer() {
  layers::LayerId id;
  id.xlayer_id = static_cast<std::uint8_t>(layers::kGlobalXLayerId);
  id.mlayer_id = 0;
  id.tlayer_id = 0;
  return id;
}

Emission make_global(EmissionKind kind, int temporal_unit) {
  Emission emission;
  emission.kind = kind;
  emission.layer = global_layer();
  emission.temporal_unit = temporal_unit;
  return emission;
}

bool same_request_key(const FrameRequest& a, const FrameRequest& b) {
  return a.temporal_unit == b.temporal_unit && a.layer == b.layer &&
         a.output == b.output;
}

}  // namespace

bool plan_emission_order(const LayerPlan& plan, int temporal_units,
                         std::vector<Emission>& order, std::string& error) {
  error.clear();
  order.clear();
  if (!validate_layer_plan(plan, error)) {
    return false;
  }
  if (temporal_units < 1) {
    return fail(error, "temporal_units must be at least 1");
  }

  for (int unit = 0; unit < temporal_units; ++unit) {
    order.push_back(make_global(EmissionKind::kTemporalDelimiter, unit));
    if (unit == 0) {
      Emission lcr = make_global(EmissionKind::kLayerConfigRecord, unit);
      order.push_back(std::move(lcr));
      Emission ops = make_global(EmissionKind::kOperatingPointSet, unit);
      ops.operating_point_index = 0;
      order.push_back(std::move(ops));
    }

    // xlayer ascending, then mlayer ascending, then sequence header once per
    // layer at the start of its coded video sequence, then tlayer ascending.
    for (const SpatialLayerSpec& spatial : plan.config.spatial_layers) {
      if (unit == 0) {
        Emission sequence;
        sequence.kind = EmissionKind::kSequenceHeader;
        sequence.layer.xlayer_id = spatial.xlayer_id;
        sequence.layer.mlayer_id = 0;
        sequence.layer.tlayer_id = 0;
        sequence.temporal_unit = unit;
        order.push_back(std::move(sequence));
      }
      for (int m = 0; m < plan.config.embedded_layers; ++m) {
        for (int t = 0; t < plan.config.temporal_layers; ++t) {
          Emission frame;
          frame.kind = EmissionKind::kFrame;
          frame.layer.xlayer_id = spatial.xlayer_id;
          frame.layer.mlayer_id = static_cast<std::uint8_t>(m);
          frame.layer.tlayer_id = static_cast<std::uint8_t>(t);
          frame.temporal_unit = unit;
          frame.output = true;
          order.push_back(std::move(frame));
        }
      }
    }
  }
  return true;
}

bool order_frames(const LayerPlan& plan, std::vector<FrameRequest> requests,
                  std::vector<Emission>& order, std::string& error) {
  error.clear();
  order.clear();
  if (!validate_layer_plan(plan, error)) {
    return false;
  }
  if (requests.empty()) {
    return fail(error, "frame schedule is empty");
  }

  for (std::size_t i = 0; i < requests.size(); ++i) {
    const FrameRequest& request = requests[i];
    if (request.temporal_unit < 0) {
      return fail(error, "frame request has a negative temporal unit");
    }
    if (!layer_is_present(plan, request.layer)) {
      std::ostringstream out;
      out << "frame request " << i << " names absent layer "
          << layers::layer_id_name(request.layer);
      return fail(error, out.str());
    }
    for (std::size_t j = 0; j < i; ++j) {
      if (same_request_key(requests[j], request)) {
        std::ostringstream out;
        out << "temporal unit " << request.temporal_unit
            << " repeats layer " << layers::layer_id_name(request.layer)
            << (request.output ? " (output unit)" : " (non-output unit)");
        return fail(error, out.str());
      }
    }
  }

  std::stable_sort(requests.begin(), requests.end(),
                   [](const FrameRequest& a, const FrameRequest& b) {
                     return std::tie(a.temporal_unit, a.layer.xlayer_id,
                                     a.layer.mlayer_id, a.layer.tlayer_id,
                                     a.output, a.frame_index) <
                            std::tie(b.temporal_unit, b.layer.xlayer_id,
                                     b.layer.mlayer_id, b.layer.tlayer_id,
                                     b.output, b.frame_index);
                   });

  int current_unit = -1;
  for (const FrameRequest& request : requests) {
    if (request.temporal_unit != current_unit) {
      current_unit = request.temporal_unit;
      order.push_back(
          make_global(EmissionKind::kTemporalDelimiter, current_unit));
    }
    Emission frame;
    frame.kind = EmissionKind::kFrame;
    frame.layer = request.layer;
    frame.temporal_unit = request.temporal_unit;
    frame.frame_index = request.frame_index;
    frame.output = request.output;
    order.push_back(std::move(frame));
  }
  return true;
}

bool reference_allowed(const LayerPlan& plan, const ReferencePolicy& policy,
                       const layers::LayerId& target,
                       const layers::LayerId& reference, std::string& error) {
  error.clear();
  if (!layer_is_present(plan, target)) {
    return fail(error,
                "reference target " + std::string(layers::layer_id_name(target)) +
                    " is not present in the layer plan");
  }
  if (!layer_is_present(plan, reference)) {
    return fail(error, "reference " +
                           std::string(layers::layer_id_name(reference)) +
                           " is not present in the layer plan");
  }
  if (target == reference) {
    return fail(error, "a layer cannot reference itself");
  }

  if (reference.xlayer_id != target.xlayer_id) {
    if (!policy.cross_xlayer_prediction) {
      std::ostringstream out;
      out << "cross-xlayer reference "
          << layers::layer_id_name(reference) << " -> "
          << layers::layer_id_name(target)
          << " is rejected: inter-layer prediction is not implemented "
             "(spatial-only path); layers must be independently decodable";
      return fail(error, out.str());
    }
    if (reference.xlayer_id >= target.xlayer_id) {
      return fail(error, "an extended layer may only reference a lower xlayer");
    }
    bool declared = false;
    for (const layers::ExtendedLayerConfig& layer : plan.lcr.layers) {
      if (layer.xlayer_id == target.xlayer_id) {
        for (std::uint8_t dep : layer.dependent_xlayer_ids) {
          if (dep == reference.xlayer_id) {
            declared = true;
          }
        }
      }
    }
    if (!declared) {
      return fail(error,
                  "the LCR declares no inter-layer dependency for this xlayer");
    }
    if (reference.mlayer_id > target.mlayer_id) {
      return fail(error,
                  "cross-xlayer reference uses a higher embedded layer");
    }
    if (reference.tlayer_id > target.tlayer_id) {
      return fail(error,
                  "cross-xlayer reference uses a higher temporal layer");
    }
    return true;
  }

  if (reference.mlayer_id != target.mlayer_id) {
    if (!policy.cross_mlayer_prediction) {
      return fail(error,
                  "cross-mlayer reference is rejected: embedded-layer "
                  "prediction is not implemented");
    }
    if (reference.mlayer_id > target.mlayer_id) {
      return fail(error, "an embedded layer may only reference a lower mlayer");
    }
  }

  if (reference.mlayer_id == target.mlayer_id) {
    if (reference.tlayer_id > target.tlayer_id) {
      return fail(error, "temporal reference points forward in time");
    }
    if (reference.tlayer_id < target.tlayer_id &&
        !policy.cross_tlayer_prediction) {
      return fail(error, "cross-tlayer reference is disabled by policy");
    }
  }
  return true;
}

bool validate_frame_references(const LayerPlan& plan,
                               const ReferencePolicy& policy,
                               const std::vector<FrameReference>& references,
                               const std::vector<Emission>& order,
                               std::string& error) {
  error.clear();
  for (const FrameReference& reference : references) {
    if (!reference_allowed(plan, policy, reference.target, reference.reference,
                           error)) {
      return false;
    }
    auto position_of = [&order](int unit, const layers::LayerId& id) {
      for (std::size_t i = 0; i < order.size(); ++i) {
        const Emission& emission = order[i];
        if (emission.kind == EmissionKind::kFrame &&
            emission.temporal_unit == unit && emission.layer == id) {
          return static_cast<int>(i);
        }
      }
      return -1;
    };
    if (reference.reference_temporal_unit > reference.temporal_unit) {
      return fail(error, "reference points to a future temporal unit");
    }
    const int target_pos = position_of(reference.temporal_unit, reference.target);
    const int ref_pos = position_of(reference.reference_temporal_unit,
                                    reference.reference);
    if (target_pos < 0) {
      return fail(error, "reference target is not in the emission schedule");
    }
    if (ref_pos < 0) {
      std::ostringstream out;
      out << "reference " << layers::layer_id_name(reference.reference)
          << " is not emitted in temporal unit "
          << reference.reference_temporal_unit;
      return fail(error, out.str());
    }
    if (reference.reference_temporal_unit == reference.temporal_unit &&
        ref_pos >= target_pos) {
      std::ostringstream out;
      out << "reference " << layers::layer_id_name(reference.reference)
          << " is emitted after or with its target "
          << layers::layer_id_name(reference.target);
      return fail(error, out.str());
    }
  }
  return true;
}

std::size_t frame_emission_count(const std::vector<Emission>& order) {
  std::size_t count = 0;
  for (const Emission& emission : order) {
    if (emission.kind == EmissionKind::kFrame) {
      ++count;
    }
  }
  return count;
}

}  // namespace av2enc::svc

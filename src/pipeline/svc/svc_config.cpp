#include "svc_config.h"

#include <algorithm>
#include <cstddef>
#include <sstream>
#include <utility>

namespace av2enc::svc {
namespace {

bool fail(std::string& error, std::string message) {
  error = std::move(message);
  return false;
}

int popcount32(std::uint32_t value) {
  int count = 0;
  while (value != 0) {
    count += static_cast<int>(value & 1U);
    value >>= 1U;
  }
  return count;
}

int lowest_set_bit(std::uint32_t value) {
  for (int bit = 0; bit < 32; ++bit) {
    if ((value & (1U << bit)) != 0) {
      return bit;
    }
  }
  return -1;
}

int highest_set_bit(std::uint32_t value) {
  for (int bit = 31; bit >= 0; --bit) {
    if ((value & (1U << bit)) != 0) {
      return bit;
    }
  }
  return -1;
}

std::uint32_t layer_bit(int id) { return 1U << static_cast<unsigned>(id); }

std::string make_label(int op_index, std::uint32_t xlayer_map, int mlayers,
                       int tlayers) {
  std::ostringstream out;
  out << "op" << op_index << ": x" << lowest_set_bit(xlayer_map) << ".."
      << highest_set_bit(xlayer_map) << " m0..m" << (mlayers - 1) << " t0..t"
      << (tlayers - 1);
  return out.str();
}

bool validate_config(const LayerConfig& config, std::string& error) {
  if (config.spatial_layers.empty()) {
    return fail(error, "layer config needs at least one spatial layer");
  }
  if (static_cast<int>(config.spatial_layers.size()) > kMaxSpatialLayers) {
    return fail(error, "layer config has more than 30 spatial layers");
  }
  if (config.temporal_layers < 1 ||
      config.temporal_layers > kMaxTemporalLayers) {
    return fail(error, "temporal_layers is outside 1..4");
  }
  if (config.embedded_layers < 1 ||
      config.embedded_layers > kMaxEmbeddedLayers) {
    return fail(error, "embedded_layers is outside 1..8");
  }
  if (config.seq_lcr_id > 7) {
    return fail(error, "seq_lcr_id is outside f(3)");
  }
  if (config.single_picture_header) {
    if (!config.still_picture) {
      return fail(error,
                  "single_picture_header_flag requires still_picture = 1");
    }
    if (config.seq_lcr_id != 0) {
      return fail(error, "single_picture_header_flag forces seq_lcr_id = 0");
    }
    if (config.spatial_layers.size() != 1 || config.temporal_layers != 1 ||
        config.embedded_layers != 1) {
      return fail(error,
                  "single_picture_header_flag supports exactly one layer with "
                  "one temporal and one embedded layer");
    }
  } else {
    if (config.still_picture) {
      return fail(error,
                  "still_picture without single_picture_header is unsupported");
    }
    if (config.seq_lcr_id == 0) {
      return fail(error,
                  "seq_lcr_id must be 1..7 when an LCR is associated");
    }
  }

  int expected_xlayer = 0;
  int previous_width = 0;
  int previous_height = 0;
  for (std::size_t i = 0; i < config.spatial_layers.size(); ++i) {
    const SpatialLayerSpec& layer = config.spatial_layers[i];
    if (static_cast<int>(layer.xlayer_id) != expected_xlayer) {
      std::ostringstream out;
      out << "spatial layer " << i << " must use xlayer_id " << expected_xlayer
          << " (got " << static_cast<int>(layer.xlayer_id)
          << "); ids must be contiguous and ascending from 0";
      return fail(error, out.str());
    }
    if (layer.width <= 0 || layer.height <= 0) {
      std::ostringstream out;
      out << "spatial layer x" << expected_xlayer
          << " needs positive coded dimensions";
      return fail(error, out.str());
    }
    if (i > 0 && (layer.width < previous_width ||
                  layer.height < previous_height)) {
      std::ostringstream out;
      out << "spatial layer x" << expected_xlayer
          << " is smaller than the layer below it; layers must be "
             "non-decreasing in size";
      return fail(error, out.str());
    }
    previous_width = layer.width;
    previous_height = layer.height;
    ++expected_xlayer;
  }

  if (config.dependencies.has_value()) {
    if (!layers::validate_dependency_maps(
            *config.dependencies,
            static_cast<int>(config.embedded_layers) - 1,
            static_cast<int>(config.temporal_layers) - 1, error)) {
      return fail(error, std::string("custom dependency maps: ") + error);
    }
  }

  if (!config.single_picture_header &&
      static_cast<int>(config.spatial_layers.size()) > kMaxOperatingPoints) {
    std::ostringstream out;
    out << "default operating-point generation needs one point per spatial "
           "layer, but "
        << config.spatial_layers.size() << " layers exceed the f(3) ops_cnt "
        << "limit of " << kMaxOperatingPoints;
    return fail(error, out.str());
  }
  return true;
}

layers::LayerConfigRecord build_lcr(const LayerConfig& config) {
  layers::LayerConfigRecord record;
  record.is_global = true;
  record.xlayer_id = static_cast<std::uint8_t>(layers::kGlobalXLayerId);
  record.global_config_record_id = config.seq_lcr_id;
  record.aggregate.present = false;
  // §7.3.7: the hierarchical temporal-layer GOP codes
  // monotonic_output_order_flag = 0, which requires the activated global LCR
  // to set lcr_doh_constraint_flag = 1.  The flat chain keeps it 0 so the
  // existing streams stay byte-identical.
  record.doh_constraint = config.temporal_layers > 1;

  for (const SpatialLayerSpec& spatial : config.spatial_layers) {
    record.xlayer_map |= layer_bit(spatial.xlayer_id);

    layers::ExtendedLayerConfig extended;
    extended.xlayer_id = spatial.xlayer_id;
    extended.mlayer_map =
        (1U << static_cast<unsigned>(config.embedded_layers)) - 1U;
    for (int m = 0; m < config.embedded_layers; ++m) {
      extended.tlayer_map[static_cast<std::size_t>(m)] =
          static_cast<std::uint8_t>(
              (1U << static_cast<unsigned>(config.temporal_layers)) - 1U);
    }
    // Spatial-only path: no inter-layer prediction dependency is declared.
    extended.dependent_xlayers = false;

    for (int m = 0; m < config.embedded_layers; ++m) {
      layers::EmbeddedLayerConfig embedded;
      embedded.mlayer_id = static_cast<std::uint8_t>(m);
      embedded.tlayer_map = extended.tlayer_map[static_cast<std::size_t>(m)];
      // Each xlayer has its own sequence header sized to that layer, so its
      // embedded descriptors always match "the sequence header" resolution.
      embedded.same_sh_max_resolution = true;
      embedded.max_expected_width = 0;
      embedded.max_expected_height = 0;
      embedded.has_priority_order = false;
      extended.embedded.push_back(embedded);
    }
    record.layers.push_back(std::move(extended));
  }
  return record;
}

layers::OperatingPointSet build_ops(const LayerConfig& config) {
  layers::OperatingPointSet set;
  set.xlayer_id = static_cast<std::uint8_t>(layers::kGlobalXLayerId);
  set.ops_id = 0;

  const int spatial_count = static_cast<int>(config.spatial_layers.size());
  for (int i = 0; i < spatial_count; ++i) {
    layers::OperatingPoint point;
    point.op_index = static_cast<std::uint8_t>(i);
    point.xlayer_map = (1U << static_cast<unsigned>(i + 1)) - 1U;
    for (int x = 0; x <= i; ++x) {
      point.mlayer_map[static_cast<std::size_t>(x)] =
          (1U << static_cast<unsigned>(config.embedded_layers)) - 1U;
      for (int m = 0; m < config.embedded_layers; ++m) {
        point.tlayer_map[static_cast<std::size_t>(x)]
                        [static_cast<std::size_t>(m)] =
            static_cast<std::uint8_t>(
                (1U << static_cast<unsigned>(config.temporal_layers)) - 1U);
      }
    }
    set.points.push_back(point);
  }
  return set;
}

bool selection_global_entry(const LayerPlan& plan) {
  return plan.lcr.is_global || plan.ops.xlayer_id == layers::kGlobalXLayerId;
}

void mark_global_obu(layers::RetentionMap& retention) {
  retention.retain[static_cast<std::size_t>(layers::kGlobalXLayerId)][0][0] =
      true;
  retention.xlayer_selected[static_cast<std::size_t>(
      layers::kGlobalXLayerId)] = true;
}

}  // namespace

bool build_layer_plan(const LayerConfig& config, LayerPlan& plan,
                      std::string& error) {
  error.clear();
  if (!validate_config(config, error)) {
    return false;
  }

  LayerPlan built;
  built.config = config;

  const int embedded_count = config.embedded_layers;
  const int temporal_count = config.temporal_layers;

  // One sequence header per spatial layer (§5.4.1). max_mlayer_id and
  // max_tlayer_id describe the layer's own embedded/temporal range.
  for (const SpatialLayerSpec& spatial : config.spatial_layers) {
    layers::SequenceLayerConfig sequence = layers::default_sequence_layer_config(
        embedded_count - 1, temporal_count - 1, config.single_picture_header);
    sequence.still_picture = config.still_picture;
    sequence.seq_lcr_id =
        config.single_picture_header ? 0 : config.seq_lcr_id;
    if (config.dependencies.has_value()) {
      sequence.dependencies = *config.dependencies;
    }
    sequence.max_width = spatial.width;
    sequence.max_height = spatial.height;
    if (!layers::validate_sequence_layer_config(sequence, error)) {
      std::ostringstream out;
      out << "spatial layer x" << static_cast<int>(spatial.xlayer_id)
          << ": " << error;
      return fail(error, out.str());
    }
    built.sequences.push_back(sequence);
  }

  built.lcr = build_lcr(config);
  built.ops = build_ops(config);

  for (std::size_t i = 0; i < config.spatial_layers.size(); ++i) {
    const SpatialLayerSpec& spatial = config.spatial_layers[i];
    for (int m = 0; m < embedded_count; ++m) {
      for (int t = 0; t < temporal_count; ++t) {
        LayerSpec spec;
        spec.id.xlayer_id = spatial.xlayer_id;
        spec.id.mlayer_id = static_cast<std::uint8_t>(m);
        spec.id.tlayer_id = static_cast<std::uint8_t>(t);
        spec.width = spatial.width;
        spec.height = spatial.height;
        built.layers.push_back(spec);
      }
    }
  }

  for (int i = 0; i < static_cast<int>(config.spatial_layers.size()); ++i) {
    OperatingPointInfo info;
    info.op_index = static_cast<std::uint8_t>(i);
    info.xlayer_map = built.ops.points[static_cast<std::size_t>(i)].xlayer_map;
    info.spatial_layers = popcount32(info.xlayer_map);
    info.embedded_layers = embedded_count;
    info.temporal_layers = temporal_count;
    info.label = make_label(i, info.xlayer_map, embedded_count, temporal_count);
    built.operating_points.push_back(std::move(info));
  }

  plan = std::move(built);
  return validate_layer_plan(plan, error);
}

bool validate_layer_plan(const LayerPlan& plan, std::string& error) {
  error.clear();
  if (!validate_config(plan.config, error)) {
    return false;
  }
  const int spatial_count = static_cast<int>(plan.config.spatial_layers.size());
  if (spatial_count == 0 || static_cast<int>(plan.layers.size()) !=
                                spatial_count * plan.config.embedded_layers *
                                    plan.config.temporal_layers) {
    return fail(error, "layer plan has an inconsistent layer list");
  }
  if (static_cast<int>(plan.sequences.size()) != spatial_count) {
    return fail(error, "layer plan needs one sequence header per spatial layer");
  }
  if (plan.ops.xlayer_id != layers::kGlobalXLayerId) {
    return fail(error, "layer plan expects a global operating point set");
  }
  if (plan.config.temporal_layers > 1 && !plan.lcr.doh_constraint) {
    return fail(error,
                "hierarchical temporal layers require the global LCR "
                "lcr_doh_constraint_flag (§7.3.7)");
  }
  if (static_cast<int>(plan.ops.points.size()) != spatial_count ||
      plan.operating_points.size() != plan.ops.points.size()) {
    return fail(error, "layer plan needs one operating point per spatial layer");
  }
  const std::uint32_t expected_xlayer_map =
      (1U << static_cast<unsigned>(spatial_count)) - 1U;
  if (plan.lcr.xlayer_map != expected_xlayer_map) {
    return fail(error,
                "layer plan LCR xlayer_map does not describe every spatial "
                "layer");
  }
  if (static_cast<int>(plan.lcr.layers.size()) != spatial_count) {
    return fail(error, "layer plan LCR needs one entry per spatial layer");
  }

  for (std::size_t i = 0; i < plan.sequences.size(); ++i) {
    const layers::SequenceLayerConfig& sequence = plan.sequences[i];
    if (!layers::validate_sequence_layer_config(sequence, error)) {
      std::ostringstream out;
      out << "sequence header " << i << ": " << error;
      return fail(error, out.str());
    }
    if (sequence.max_width != plan.config.spatial_layers[i].width ||
        sequence.max_height != plan.config.spatial_layers[i].height) {
      return fail(error, "sequence header dimensions disagree with the config");
    }
  }
  const layers::SequenceLayerConfig& base = plan.sequences.front();
  for (std::size_t i = 1; i < plan.sequences.size(); ++i) {
    const layers::SequenceLayerConfig& candidate = plan.sequences[i];
    if (candidate.max_mlayer_id != base.max_mlayer_id ||
        candidate.max_tlayer_id != base.max_tlayer_id ||
        candidate.dependencies.mlayer != base.dependencies.mlayer ||
        candidate.dependencies.tlayer != base.dependencies.tlayer) {
      return fail(error,
                  "layer plan sequence headers disagree on dependency maps");
    }
  }

  for (std::size_t i = 0; i < plan.lcr.layers.size(); ++i) {
    const layers::ExtendedLayerConfig& lcr_layer = plan.lcr.layers[i];
    if (static_cast<int>(lcr_layer.xlayer_id) != static_cast<int>(i)) {
      return fail(error, "LCR layer entries are not in ascending xlayer order");
    }
    if (lcr_layer.dependent_xlayers ||
        !lcr_layer.dependent_xlayer_ids.empty()) {
      // AV2 v1.0.0 §6.8.2: "It is a requirement of bitstream conformance that
      // the value of lcr_dependent_xlayers_flag is equal to 0."  A conforming
      // stream therefore cannot signal cross-xlayer (inter-layer) prediction,
      // and there is no resampling process to gate.  Embedded (mlayer)
      // refinement is the decodable quality-scalability path (§5.4.1
      // MLayerDependencyMap).
      return fail(
          error,
          "inter-layer prediction is not a conforming AV2 v1.0.0 feature: "
          "§6.8.2 requires lcr_dependent_xlayers_flag = 0, so no cross-xlayer "
          "reference or resampling can be signalled; use embedded (mlayer) "
          "refinement instead");
    }
    const std::uint32_t mlayer_map =
        (1U << static_cast<unsigned>(plan.config.embedded_layers)) - 1U;
    if (lcr_layer.mlayer_map != mlayer_map) {
      return fail(error, "LCR mlayer_map disagrees with the layer config");
    }
    for (int m = 0; m < plan.config.embedded_layers; ++m) {
      const std::uint8_t tlayer_map = static_cast<std::uint8_t>(
          (1U << static_cast<unsigned>(plan.config.temporal_layers)) - 1U);
      if (lcr_layer.tlayer_map[static_cast<std::size_t>(m)] != tlayer_map) {
        return fail(error,
                    "LCR tlayer_map disagrees with the layer config");
      }
    }
  }
  if (!layers::validate_layer_config_record(plan.lcr, base, error)) {
    return fail(error, std::string("LCR: ") + error);
  }
  if (!layers::validate_operating_point_set(plan.ops, base, &plan.lcr, error)) {
    return fail(error, std::string("OPS: ") + error);
  }

  for (int i = 0; i < spatial_count; ++i) {
    const std::uint32_t expected = (1U << static_cast<unsigned>(i + 1)) - 1U;
    if (plan.ops.points[static_cast<std::size_t>(i)].xlayer_map != expected) {
      return fail(error, "operating points are not cumulative over xlayers");
    }
    if (plan.operating_points[static_cast<std::size_t>(i)].op_index !=
        static_cast<std::uint8_t>(i)) {
      return fail(error, "operating point info is out of order");
    }
  }

  int index = 0;
  for (const SpatialLayerSpec& spatial : plan.config.spatial_layers) {
    for (int m = 0; m < plan.config.embedded_layers; ++m) {
      for (int t = 0; t < plan.config.temporal_layers; ++t) {
        const LayerSpec& spec = plan.layers[static_cast<std::size_t>(index)];
        if (spec.id.xlayer_id != spatial.xlayer_id ||
            static_cast<int>(spec.id.mlayer_id) != m ||
            static_cast<int>(spec.id.tlayer_id) != t ||
            spec.width != spatial.width || spec.height != spatial.height) {
          return fail(error, "layer list is not in deterministic x/m/t order");
        }
        ++index;
      }
    }
  }
  return true;
}

bool assign_layers(const LayerPlan& plan,
                   const std::vector<FrameSource>& sources,
                   std::vector<layers::LayerId>& assigned, std::string& error) {
  error.clear();
  assigned.clear();
  assigned.reserve(sources.size());

  std::vector<std::pair<int, layers::LayerId>> seen;
  for (std::size_t i = 0; i < sources.size(); ++i) {
    const FrameSource& source = sources[i];
    if (source.temporal_unit < 0) {
      return fail(error, "frame source has a negative temporal unit");
    }
    if (source.width <= 0 || source.height <= 0) {
      return fail(error, "frame source has non-positive dimensions");
    }
    if (source.tlayer >= plan.config.temporal_layers) {
      std::ostringstream out;
      out << "frame source " << i << " uses tlayer "
          << static_cast<int>(source.tlayer) << " above max_tlayer_id "
          << (plan.config.temporal_layers - 1);
      return fail(error, out.str());
    }
    if (source.mlayer >= plan.config.embedded_layers) {
      std::ostringstream out;
      out << "frame source " << i << " uses mlayer "
          << static_cast<int>(source.mlayer) << " above max_mlayer_id "
          << (plan.config.embedded_layers - 1);
      return fail(error, out.str());
    }
    int matches = 0;
    std::uint8_t match_x = 0;
    for (const SpatialLayerSpec& spatial : plan.config.spatial_layers) {
      if (spatial.width == source.width && spatial.height == source.height) {
        ++matches;
        match_x = spatial.xlayer_id;
      }
    }
    if (matches == 0) {
      std::ostringstream out;
      out << "frame source " << i << " (" << source.width << 'x'
          << source.height << ") matches no spatial layer";
      return fail(error, out.str());
    }
    if (matches > 1) {
      std::ostringstream out;
      out << "frame source " << i << " (" << source.width << 'x'
          << source.height
          << ") matches multiple spatial layers; resolve by distinct coded "
             "sizes, not by submission order";
      return fail(error, out.str());
    }
    layers::LayerId id;
    id.xlayer_id = match_x;
    id.mlayer_id = source.mlayer;
    id.tlayer_id = source.tlayer;
    for (const auto& pair : seen) {
      if (pair.first == source.temporal_unit && pair.second == id) {
        std::ostringstream out;
        out << "temporal unit " << source.temporal_unit
            << " has two frames for layer " << layers::layer_id_name(id);
        return fail(error, out.str());
      }
    }
    seen.emplace_back(source.temporal_unit, id);
    assigned.push_back(id);
  }
  return true;
}

bool resolve_selection(const LayerPlan& plan, const LayerSelection& selection,
                       layers::RetentionMap& retention, std::string& error) {
  error.clear();
  if (!validate_layer_plan(plan, error)) {
    return false;
  }
  retention = layers::RetentionMap{};

  if (selection.mode == SelectionMode::kAllLayers) {
    for (const LayerSpec& spec : plan.layers) {
      retention.retain[static_cast<std::size_t>(spec.id.xlayer_id)]
                      [static_cast<std::size_t>(spec.id.mlayer_id)]
                      [static_cast<std::size_t>(spec.id.tlayer_id)] = true;
    }
  } else if (selection.mode == SelectionMode::kOperatingPoint) {
    const int count = static_cast<int>(plan.ops.points.size());
    if (selection.op_index < 0 || selection.op_index >= count) {
      std::ostringstream out;
      out << "operating point index " << selection.op_index
          << " is outside 0.." << (count - 1);
      return fail(error, out.str());
    }
    if (!layers::build_retention_map(plan.ops, selection.op_index,
                                     plan.sequences.front().dependencies,
                                     retention, error)) {
      return fail(error, std::string("retention: ") + error);
    }
  } else {
    const int count = static_cast<int>(plan.config.spatial_layers.size());
    if (selection.max_xlayer < 0 || selection.max_xlayer >= count) {
      std::ostringstream out;
      out << "spatial layer limit " << selection.max_xlayer
          << " is outside 0.." << (count - 1);
      return fail(error, out.str());
    }
    for (const LayerSpec& spec : plan.layers) {
      if (static_cast<int>(spec.id.xlayer_id) <= selection.max_xlayer) {
        retention.retain[static_cast<std::size_t>(spec.id.xlayer_id)]
                        [static_cast<std::size_t>(spec.id.mlayer_id)]
                        [static_cast<std::size_t>(spec.id.tlayer_id)] = true;
      }
    }
  }

  for (const LayerSpec& spec : plan.layers) {
    if (retention_has_layer(retention, spec.id)) {
      retention.xlayer_selected[static_cast<std::size_t>(spec.id.xlayer_id)] =
          true;
    }
  }
  if (selection_global_entry(plan)) {
    mark_global_obu(retention);
  }
  return validate_retention_closed(plan, retention, error);
}

bool retention_has_layer(const layers::RetentionMap& retention,
                         const layers::LayerId& id) {
  if (id.xlayer_id >= layers::kExtendedLayerCount ||
      id.mlayer_id >= layers::kEmbeddedLayerCount ||
      id.tlayer_id >= layers::kTemporalLayerCount) {
    return false;
  }
  return retention.retain[static_cast<std::size_t>(id.xlayer_id)]
                        [static_cast<std::size_t>(id.mlayer_id)]
                        [static_cast<std::size_t>(id.tlayer_id)];
}

int selected_layer_count(const layers::RetentionMap& retention,
                         const LayerPlan& plan) {
  int count = 0;
  for (const LayerSpec& spec : plan.layers) {
    if (retention_has_layer(retention, spec.id)) {
      ++count;
    }
  }
  return count;
}

bool validate_retention_closed(const LayerPlan& plan,
                               const layers::RetentionMap& retention,
                               std::string& error) {
  error.clear();
  const layers::LayerDependencyMaps& maps =
      plan.sequences.front().dependencies;

  for (const LayerSpec& spec : plan.layers) {
    if (!retention_has_layer(retention, spec.id)) {
      continue;
    }
    const int x = spec.id.xlayer_id;
    const int m = spec.id.mlayer_id;
    const int t = spec.id.tlayer_id;
    for (int ref_m = 0; ref_m < layers::kEmbeddedLayerCount; ++ref_m) {
      if (!maps.mlayer[static_cast<std::size_t>(m)]
                      [static_cast<std::size_t>(ref_m)]) {
        continue;
      }
      for (int ref_t = 0; ref_t < layers::kTemporalLayerCount; ++ref_t) {
        if (!maps.tlayer[static_cast<std::size_t>(m)]
                        [static_cast<std::size_t>(t)]
                        [static_cast<std::size_t>(ref_t)]) {
          continue;
        }
        layers::LayerId ref;
        ref.xlayer_id = static_cast<std::uint8_t>(x);
        ref.mlayer_id = static_cast<std::uint8_t>(ref_m);
        ref.tlayer_id = static_cast<std::uint8_t>(ref_t);
        if (!retention_has_layer(retention, ref)) {
          std::ostringstream out;
          out << "retention keeps " << layers::layer_id_name(spec.id)
              << " but drops dependency " << layers::layer_id_name(ref);
          return fail(error, out.str());
        }
      }
    }
  }

  // Every retained entry must name a layer in the plan, except the global OBU
  // bucket (GLOBAL_XLAYER_ID, m0, t0).
  for (int x = 0; x < layers::kExtendedLayerCount; ++x) {
    for (int m = 0; m < layers::kEmbeddedLayerCount; ++m) {
      for (int t = 0; t < layers::kTemporalLayerCount; ++t) {
        if (!retention.retain[static_cast<std::size_t>(x)]
                             [static_cast<std::size_t>(m)]
                             [static_cast<std::size_t>(t)]) {
          continue;
        }
        if (x == layers::kGlobalXLayerId && m == 0 && t == 0) {
          continue;
        }
        layers::LayerId id;
        id.xlayer_id = static_cast<std::uint8_t>(x);
        id.mlayer_id = static_cast<std::uint8_t>(m);
        id.tlayer_id = static_cast<std::uint8_t>(t);
        if (!layer_is_present(plan, id)) {
          std::ostringstream out;
          out << "retention keeps absent layer " << layers::layer_id_name(id);
          return fail(error, out.str());
        }
      }
    }
  }

  for (int x = 0; x < layers::kExtendedLayerCount; ++x) {
    if (!retention.xlayer_selected[static_cast<std::size_t>(x)]) {
      continue;
    }
    bool has_any = false;
    for (int m = 0; m < layers::kEmbeddedLayerCount && !has_any; ++m) {
      for (int t = 0; t < layers::kTemporalLayerCount; ++t) {
        if (retention.retain[static_cast<std::size_t>(x)]
                            [static_cast<std::size_t>(m)]
                            [static_cast<std::size_t>(t)]) {
          has_any = true;
          break;
        }
      }
    }
    if (!has_any) {
      std::ostringstream out;
      out << "xlayer " << x << " is marked selected but retains no layer";
      return fail(error, out.str());
    }
  }
  return true;
}

bool find_layer(const LayerPlan& plan, const layers::LayerId& id,
                const LayerSpec** spec) {
  for (const LayerSpec& candidate : plan.layers) {
    if (candidate.id == id) {
      if (spec != nullptr) {
        *spec = &candidate;
      }
      return true;
    }
  }
  return false;
}

bool layer_is_present(const LayerPlan& plan, const layers::LayerId& id) {
  return find_layer(plan, id, nullptr);
}

}  // namespace av2enc::svc

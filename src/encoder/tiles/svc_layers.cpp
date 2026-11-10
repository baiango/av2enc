#include "svc_layers.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <string>
#include <utility>

namespace av2enc::tiles::svc {
namespace {

constexpr std::uint32_t kMlayerMask = 0xFFu;         // 8 embedded layers
constexpr std::uint32_t kTlayerMask = 0x0Fu;         // 4 temporal layers
constexpr std::uint32_t kXlayerMapMask = 0x7FFFFFFFu;  // f(31): bits 0..30

bool fail(std::string& error, std::string message) {
  error = std::move(message);
  return false;
}

bool check_mask_range(std::uint32_t bits, std::uint32_t mask,
                      const char* name, std::string& error) {
  if ((bits & ~mask) != 0) {
    std::ostringstream out;
    out << name << " has bits outside the legal range";
    return fail(error, out.str());
  }
  return true;
}

}  // namespace

bool operator==(const LayerId& a, const LayerId& b) {
  return a.xlayer_id == b.xlayer_id && a.mlayer_id == b.mlayer_id &&
         a.tlayer_id == b.tlayer_id;
}

bool operator<(const LayerId& a, const LayerId& b) {
  if (a.xlayer_id != b.xlayer_id) {
    return a.xlayer_id < b.xlayer_id;
  }
  if (a.mlayer_id != b.mlayer_id) {
    return a.mlayer_id < b.mlayer_id;
  }
  return a.tlayer_id < b.tlayer_id;
}

bool validate_layer_id(const LayerId& id, std::string& error) {
  if (id.xlayer_id >= kExtendedLayerCount) {
    return fail(error, "obu_xlayer_id is outside 0..31");
  }
  if (id.mlayer_id >= kEmbeddedLayerCount) {
    return fail(error, "obu_mlayer_id is outside 0..7");
  }
  if (id.tlayer_id >= kTemporalLayerCount) {
    return fail(error, "obu_tlayer_id is outside 0..3");
  }
  return true;
}

const char* layer_id_name(const LayerId& id) {
  static thread_local char buffer[64];
  std::snprintf(buffer, sizeof(buffer), "x%u/m%u/t%u",
                static_cast<unsigned>(id.xlayer_id),
                static_cast<unsigned>(id.mlayer_id),
                static_cast<unsigned>(id.tlayer_id));
  return buffer;
}

LayerDependencyMaps default_dependency_maps(int max_mlayer_id,
                                            int max_tlayer_id) {
  LayerDependencyMaps maps;
  for (int curr = 0; curr < kEmbeddedLayerCount; ++curr) {
    for (int ref = 0; ref < kEmbeddedLayerCount; ++ref) {
      maps.mlayer[static_cast<std::size_t>(curr)][static_cast<std::size_t>(ref)] =
          ref <= curr && curr <= max_mlayer_id;
    }
    for (int curr_t = 0; curr_t < kTemporalLayerCount; ++curr_t) {
      for (int ref_t = 0; ref_t < kTemporalLayerCount; ++ref_t) {
        maps.tlayer[static_cast<std::size_t>(curr)]
                   [static_cast<std::size_t>(curr_t)]
                   [static_cast<std::size_t>(ref_t)] =
            curr <= max_mlayer_id && ref_t <= curr_t &&
            curr_t <= max_tlayer_id;
      }
    }
  }
  return maps;
}

bool validate_dependency_maps(const LayerDependencyMaps& maps,
                              int max_mlayer_id, int max_tlayer_id,
                              std::string& error) {
  if (max_mlayer_id < 0 || max_mlayer_id > kMaxEmbeddedLayerId) {
    return fail(error, "max_mlayer_id is outside 0..7");
  }
  if (max_tlayer_id < 0 || max_tlayer_id > kMaxTemporalLayerId) {
    return fail(error, "max_tlayer_id is outside 0..3");
  }

  for (int curr = 0; curr < kEmbeddedLayerCount; ++curr) {
    for (int ref = 0; ref < kEmbeddedLayerCount; ++ref) {
      const bool dep =
          maps.mlayer[static_cast<std::size_t>(curr)][static_cast<std::size_t>(ref)];
      if (dep && (ref > curr || curr > max_mlayer_id || ref > max_mlayer_id)) {
        return fail(error, "MLayerDependencyMap has an out-of-range dependency");
      }
      if (curr <= max_mlayer_id && ref == curr && !dep) {
        return fail(error, "MLayerDependencyMap is not reflexive");
      }
      if (curr > max_mlayer_id && dep) {
        return fail(error, "MLayerDependencyMap marks a layer above max_mlayer_id");
      }
    }
  }
  // Transitive closure: curr -> mid and mid -> ref implies curr -> ref.
  for (int curr = 0; curr < kEmbeddedLayerCount; ++curr) {
    for (int mid = 0; mid < kEmbeddedLayerCount; ++mid) {
      if (!maps.mlayer[static_cast<std::size_t>(curr)][static_cast<std::size_t>(mid)]) {
        continue;
      }
      for (int ref = 0; ref < kEmbeddedLayerCount; ++ref) {
        if (maps.mlayer[static_cast<std::size_t>(mid)][static_cast<std::size_t>(ref)] &&
            !maps.mlayer[static_cast<std::size_t>(curr)][static_cast<std::size_t>(ref)]) {
          return fail(error, "MLayerDependencyMap is not transitive");
        }
      }
    }
  }

  for (int mlayer = 0; mlayer < kEmbeddedLayerCount; ++mlayer) {
    if (mlayer > max_mlayer_id) {
      continue;  // TLayerDependencyMap is defined for present layers only.
    }
    for (int curr = 0; curr < kTemporalLayerCount; ++curr) {
      for (int ref = 0; ref < kTemporalLayerCount; ++ref) {
        const bool dep = maps.tlayer[static_cast<std::size_t>(mlayer)]
                                        [static_cast<std::size_t>(curr)]
                                        [static_cast<std::size_t>(ref)];
        if (dep && (ref > curr || curr > max_tlayer_id || ref > max_tlayer_id)) {
          return fail(error, "TLayerDependencyMap has an out-of-range dependency");
        }
        if (curr <= max_tlayer_id && ref == curr && !dep) {
          return fail(error, "TLayerDependencyMap is not reflexive");
        }
      }
    }
  }
  for (int mlayer = 0; mlayer < kEmbeddedLayerCount; ++mlayer) {
    if (mlayer > max_mlayer_id) {
      continue;
    }
    for (int curr = 0; curr < kTemporalLayerCount; ++curr) {
      for (int mid = 0; mid < kTemporalLayerCount; ++mid) {
        if (!maps.tlayer[static_cast<std::size_t>(mlayer)]
                         [static_cast<std::size_t>(curr)]
                         [static_cast<std::size_t>(mid)]) {
          continue;
        }
        for (int ref = 0; ref < kTemporalLayerCount; ++ref) {
          if (maps.tlayer[static_cast<std::size_t>(mlayer)]
                              [static_cast<std::size_t>(mid)]
                              [static_cast<std::size_t>(ref)] &&
              !maps.tlayer[static_cast<std::size_t>(mlayer)]
                              [static_cast<std::size_t>(curr)]
                              [static_cast<std::size_t>(ref)]) {
            return fail(error, "TLayerDependencyMap is not transitive");
          }
        }
      }
    }
  }
  return true;
}

bool mlayer_dependencies_satisfied(std::uint32_t mlayer_map,
                                   const LayerDependencyMaps& maps,
                                   std::string& error) {
  if (!check_mask_range(mlayer_map, kMlayerMask, "mlayer_map", error)) {
    return false;
  }
  for (int curr = 0; curr < kEmbeddedLayerCount; ++curr) {
    if ((mlayer_map & (1u << curr)) == 0) {
      continue;
    }
    for (int ref = 0; ref < kEmbeddedLayerCount; ++ref) {
      if (maps.mlayer[static_cast<std::size_t>(curr)][static_cast<std::size_t>(ref)] &&
          (mlayer_map & (1u << ref)) == 0) {
        std::ostringstream out;
        out << "mlayer_map selects layer " << curr << " without dependency "
            << ref;
        return fail(error, out.str());
      }
    }
  }
  return true;
}

bool tlayer_dependencies_satisfied(
    std::uint32_t mlayer_map,
    const std::array<std::uint8_t, kEmbeddedLayerCount>& tlayer_map,
    const LayerDependencyMaps& maps, std::string& error) {
  if (!check_mask_range(mlayer_map, kMlayerMask, "mlayer_map", error)) {
    return false;
  }
  for (int mlayer = 0; mlayer < kEmbeddedLayerCount; ++mlayer) {
    const std::uint32_t bits = tlayer_map[static_cast<std::size_t>(mlayer)];
    if (!check_mask_range(bits, kTlayerMask, "tlayer_map", error)) {
      return false;
    }
    if ((mlayer_map & (1u << mlayer)) == 0) {
      if (bits != 0) {
        return fail(error,
                    "tlayer_map is non-zero for an absent embedded layer");
      }
      continue;
    }
    if (bits == 0) {
      return fail(error, "present embedded layer has an empty tlayer_map");
    }
    for (int curr = 0; curr < kTemporalLayerCount; ++curr) {
      if ((bits & (1u << curr)) == 0) {
        continue;
      }
      for (int ref = 0; ref < kTemporalLayerCount; ++ref) {
        if (maps.tlayer[static_cast<std::size_t>(mlayer)]
                           [static_cast<std::size_t>(curr)]
                           [static_cast<std::size_t>(ref)] &&
            (bits & (1u << ref)) == 0) {
          std::ostringstream out;
          out << "tlayer_map selects m" << mlayer << "/t" << curr
              << " without dependency t" << ref;
          return fail(error, out.str());
        }
      }
    }
  }
  return true;
}

SequenceLayerConfig default_sequence_layer_config(int max_mlayer_id,
                                                  int max_tlayer_id,
                                                  bool single_picture_header) {
  SequenceLayerConfig config;
  config.single_picture_header = single_picture_header;
  config.still_picture = single_picture_header;
  if (single_picture_header) {
    config.seq_lcr_id = 0;
    config.max_tlayer_id = 0;
    config.max_mlayer_id = 0;
    config.seq_max_mlayer_cnt = 1;
  } else {
    config.max_tlayer_id =
        static_cast<std::uint8_t>(max_tlayer_id < 0 ? 0 : max_tlayer_id);
    config.max_mlayer_id =
        static_cast<std::uint8_t>(max_mlayer_id < 0 ? 0 : max_mlayer_id);
    config.seq_max_mlayer_cnt =
        static_cast<std::uint8_t>(config.max_mlayer_id + 1);
  }
  config.dependencies =
      default_dependency_maps(config.max_mlayer_id, config.max_tlayer_id);
  return config;
}

bool validate_sequence_layer_config(const SequenceLayerConfig& config,
                                    std::string& error) {
  if (config.seq_lcr_id > 7) {
    return fail(error, "seq_lcr_id is outside f(3)");
  }
  if (config.max_mlayer_id > kMaxEmbeddedLayerId) {
    return fail(error, "max_mlayer_id is outside 0..7");
  }
  if (config.max_tlayer_id > kMaxTemporalLayerId) {
    return fail(error, "max_tlayer_id is outside 0..3");
  }
  if (config.max_width <= 0 || config.max_height <= 0) {
    return fail(error, "sequence layer config needs positive max dimensions");
  }
  if (config.single_picture_header) {
    if (!config.still_picture) {
      return fail(error,
                  "single_picture_header_flag requires still_picture = 1");
    }
    if (config.seq_lcr_id != 0 || config.max_mlayer_id != 0 ||
        config.max_tlayer_id != 0 || config.seq_max_mlayer_cnt != 1) {
      return fail(error,
                  "single_picture_header_flag forces one base layer, "
                  "seq_lcr_id = 0, and SeqMaxMlayerCnt = 1");
    }
  } else if (config.still_picture) {
    return fail(error, "still_picture without single_picture_header is unsupported");
  }
  if (config.seq_max_mlayer_cnt < 1 ||
      config.seq_max_mlayer_cnt >
          static_cast<int>(config.max_mlayer_id) + 1) {
    return fail(error, "seq_max_mlayer_cnt is outside 1..max_mlayer_id+1");
  }
  return validate_dependency_maps(config.dependencies, config.max_mlayer_id,
                                  config.max_tlayer_id, error);
}

bool validate_layer_config_record(const LayerConfigRecord& record,
                                  const SequenceLayerConfig& sequence,
                                  std::string& error) {
  if (!validate_sequence_layer_config(sequence, error)) {
    return false;
  }
  if (record.global_config_record_id > 7) {
    return fail(error, "lcr_global_config_record_id is outside f(3)");
  }
  if (record.aggregate.present) {
    if (record.aggregate.config_idc > 63 || record.aggregate.level_idx > 31 ||
        record.aggregate.max_interop > 15) {
      return fail(error, "LCR aggregate info field is outside its descriptor");
    }
  }

  if (record.is_global) {
    if (record.xlayer_id != kGlobalXLayerId) {
      return fail(error, "global LCR must use obu_xlayer_id = 31");
    }
    if (!check_mask_range(record.xlayer_map, kXlayerMapMask,
                          "lcr_xlayer_map", error)) {
      return false;
    }
    bool seen[kExtendedLayerCount] = {};
    for (const ExtendedLayerConfig& layer : record.layers) {
      if (layer.xlayer_id > kMaxExtendedLayerId) {
        return fail(error, "global LCR contains the GLOBAL xlayer id");
      }
      if ((record.xlayer_map & (1u << layer.xlayer_id)) == 0) {
        return fail(error, "global LCR layer is absent from lcr_xlayer_map");
      }
      if (seen[layer.xlayer_id]) {
        return fail(error, "global LCR repeats an extended layer");
      }
      seen[layer.xlayer_id] = true;
    }
    // A global LCR may describe only a subset of the map; per-layer LCR OBUs
    // carry the rest. Any listed layer must still be present in the map.
  } else {
    if (record.xlayer_id > kMaxExtendedLayerId) {
      return fail(error, "local LCR must use obu_xlayer_id 0..30");
    }
    if (record.xlayer_map != 0) {
      return fail(error, "local LCR must not carry lcr_xlayer_map");
    }
    if (record.layers.size() != 1) {
      return fail(error, "local LCR must describe exactly its own xlayer");
    }
    if (record.layers.front().xlayer_id != record.xlayer_id) {
      return fail(error, "local LCR xlayer id mismatch");
    }
  }

  for (const ExtendedLayerConfig& layer : record.layers) {
    if (layer.xlayer_id > kMaxExtendedLayerId) {
      return fail(error, "extended layer id is outside 0..30");
    }
    if (!check_mask_range(layer.mlayer_map, kMlayerMask, "mlayer_map", error)) {
      return false;
    }
    if (!tlayer_dependencies_satisfied(layer.mlayer_map, layer.tlayer_map,
                                       sequence.dependencies, error)) {
      return false;
    }
    if (!mlayer_dependencies_satisfied(layer.mlayer_map, sequence.dependencies,
                                       error)) {
      return false;
    }
    for (int m = 0; m < kEmbeddedLayerCount; ++m) {
      if ((layer.mlayer_map & (1u << m)) != 0 &&
          m > static_cast<int>(sequence.max_mlayer_id)) {
        return fail(error, "LCR selects an embedded layer above max_mlayer_id");
      }
      const std::uint32_t bits = layer.tlayer_map[static_cast<std::size_t>(m)];
      if (bits == 0) {
        continue;
      }
      for (int t = 0; t < kTemporalLayerCount; ++t) {
        if ((bits & (1u << t)) != 0 &&
            t > static_cast<int>(sequence.max_tlayer_id)) {
          return fail(error, "LCR selects a temporal layer above max_tlayer_id");
        }
      }
    }
    if (layer.dependent_xlayers) {
      for (std::uint8_t dep : layer.dependent_xlayer_ids) {
        if (dep >= kExtendedLayerCount) {
          return fail(error, "dependent xlayer id is outside 0..31");
        }
        if (dep == layer.xlayer_id) {
          return fail(error, "extended layer depends on itself");
        }
      }
    }

    if (!layer.embedded.empty()) {
      std::uint32_t described = 0;
      for (const EmbeddedLayerConfig& embedded : layer.embedded) {
        if (embedded.mlayer_id > kMaxEmbeddedLayerId ||
            (described & (1u << embedded.mlayer_id)) != 0) {
          return fail(error, "embedded layer id is invalid or repeated");
        }
        described |= 1u << embedded.mlayer_id;
        if (!check_mask_range(embedded.tlayer_map, kTlayerMask,
                              "embedded tlayer_map", error)) {
          return false;
        }
        if (embedded.tlayer_map == 0) {
          return fail(error, "embedded layer has an empty tlayer_map");
        }
        if (embedded.tlayer_map !=
            layer.tlayer_map[static_cast<std::size_t>(embedded.mlayer_id)]) {
          return fail(error, "embedded tlayer_map disagrees with the xlayer map");
        }
        if (embedded.same_sh_max_resolution) {
          if (embedded.max_expected_width != 0 ||
              embedded.max_expected_height != 0) {
            return fail(error,
                        "embedded layer signals dimensions despite "
                        "same_sh_max_resolution_flag = 1");
          }
        } else if (embedded.max_expected_width == 0 ||
                   embedded.max_expected_height == 0) {
          return fail(error,
                      "embedded layer needs dimensions when it differs from the "
                      "sequence header");
        }
        if (embedded.has_priority_order && embedded.priority_order > 255) {
          return fail(error, "lcr_priority_order is outside f(8)");
        }
      }
      if (described != layer.mlayer_map) {
        return fail(error,
                    "embedded layer descriptors disagree with lcr_mlayer_map");
      }
    }
  }
  return true;
}

bool validate_operating_point_set(const OperatingPointSet& set,
                                  const SequenceLayerConfig& sequence,
                                  const LayerConfigRecord* record,
                                  std::string& error) {
  if (!validate_sequence_layer_config(sequence, error)) {
    return false;
  }
  if (set.xlayer_id >= kExtendedLayerCount) {
    return fail(error, "OPS obu_xlayer_id is outside 0..31");
  }
  if (set.ops_id > 15) {
    return fail(error, "ops_id is outside f(4)");
  }
  if (set.points.empty() || set.points.size() > 7) {
    return fail(error, "ops_cnt must be 1..7");
  }
  const bool is_global = set.xlayer_id == kGlobalXLayerId;

  if (record != nullptr) {
    if (!validate_layer_config_record(*record, sequence, error)) {
      return false;
    }
    if (is_global) {
      if (!record->is_global) {
        return fail(error, "global OPS requires a global LCR");
      }
    } else {
      if (record->is_global) {
        return fail(error, "local OPS requires a local LCR");
      }
      if (record->xlayer_id != set.xlayer_id) {
        return fail(error, "local OPS and local LCR xlayer ids differ");
      }
    }
  }

  for (std::size_t i = 0; i < set.points.size(); ++i) {
    const OperatingPoint& point = set.points[i];
    if (point.op_index != i) {
      return fail(error, "operating point op_index is not its OPS position");
    }
    if (!check_mask_range(point.xlayer_map, kXlayerMapMask, "ops_xlayer_map",
                          error)) {
      return false;
    }
    if (is_global) {
      if (point.xlayer_map == 0) {
        return fail(error, "global operating point selects no extended layer");
      }
      if (record != nullptr && record->is_global &&
          (point.xlayer_map & ~record->xlayer_map) != 0) {
        return fail(error,
                    "global operating point selects an xlayer absent from LCR");
      }
    } else {
      const std::uint32_t self = 1u << set.xlayer_id;
      if (point.xlayer_map != self) {
        return fail(error,
                    "local operating point must select exactly its own xlayer");
      }
    }

    for (int x = 0; x < kExtendedLayerCount; ++x) {
      const bool selected = (point.xlayer_map & (1u << x)) != 0;
      if (!selected) {
        if (point.mlayer_map[static_cast<std::size_t>(x)] != 0) {
          return fail(error,
                      "operating point has layer info for an unselected xlayer");
        }
        continue;
      }
      const std::uint32_t mlayer_map =
          point.mlayer_map[static_cast<std::size_t>(x)];
      if (mlayer_map == 0) {
        return fail(error, "selected extended layer has an empty mlayer_map");
      }
      if (!check_mask_range(mlayer_map, kMlayerMask, "ops_mlayer_map", error)) {
        return false;
      }
      for (int m = 0; m < kEmbeddedLayerCount; ++m) {
        if ((mlayer_map & (1u << m)) != 0 &&
            m > static_cast<int>(sequence.max_mlayer_id)) {
          return fail(error,
                      "operating point selects an embedded layer above "
                      "max_mlayer_id");
        }
      }
      if (!mlayer_dependencies_satisfied(mlayer_map, sequence.dependencies,
                                         error)) {
        return false;
      }
      if (!tlayer_dependencies_satisfied(
              mlayer_map, point.tlayer_map[static_cast<std::size_t>(x)],
              sequence.dependencies, error)) {
        return false;
      }
      for (int m = 0; m < kEmbeddedLayerCount; ++m) {
        const std::uint32_t bits =
            point.tlayer_map[static_cast<std::size_t>(x)]
                            [static_cast<std::size_t>(m)];
        for (int t = 0; t < kTemporalLayerCount; ++t) {
          if ((bits & (1u << t)) != 0 &&
              t > static_cast<int>(sequence.max_tlayer_id)) {
            return fail(error,
                        "operating point selects a temporal layer above "
                        "max_tlayer_id");
          }
        }
      }
    }
  }
  return true;
}

bool build_retention_map(const OperatingPointSet& set, int op_index,
                         const LayerDependencyMaps& maps,
                         RetentionMap& retention, std::string& error) {
  if (op_index < 0 || static_cast<std::size_t>(op_index) >= set.points.size()) {
    return fail(error, "operating point index is outside the OPS");
  }
  retention = RetentionMap{};
  const OperatingPoint& point = set.points[static_cast<std::size_t>(op_index)];
  if (!check_mask_range(point.xlayer_map, kXlayerMapMask, "ops_xlayer_map",
                        error)) {
    return false;
  }

  for (int x = 0; x < kExtendedLayerCount; ++x) {
    if ((point.xlayer_map & (1u << x)) == 0) {
      continue;
    }
    retention.xlayer_selected[static_cast<std::size_t>(x)] = true;
    const std::uint32_t mlayer_map =
        point.mlayer_map[static_cast<std::size_t>(x)];
    for (int m = 0; m < kEmbeddedLayerCount; ++m) {
      if ((mlayer_map & (1u << m)) == 0) {
        continue;
      }
      const std::uint32_t tbits =
          point.tlayer_map[static_cast<std::size_t>(x)]
                          [static_cast<std::size_t>(m)];
      for (int t = 0; t < kTemporalLayerCount; ++t) {
        if ((tbits & (1u << t)) != 0) {
          retention.retain[static_cast<std::size_t>(x)]
                          [static_cast<std::size_t>(m)]
                          [static_cast<std::size_t>(t)] = true;
        }
      }
    }
  }

  // Add dependencies transitively until the map is closed.
  bool changed = true;
  while (changed) {
    changed = false;
    for (int x = 0; x < kExtendedLayerCount; ++x) {
      for (int m = 0; m < kEmbeddedLayerCount; ++m) {
        for (int t = 0; t < kTemporalLayerCount; ++t) {
          if (!retention.retain[static_cast<std::size_t>(x)]
                               [static_cast<std::size_t>(m)]
                               [static_cast<std::size_t>(t)]) {
            continue;
          }
          for (int ref_m = 0; ref_m < kEmbeddedLayerCount; ++ref_m) {
            if (!maps.mlayer[static_cast<std::size_t>(m)]
                            [static_cast<std::size_t>(ref_m)]) {
              continue;
            }
            for (int ref_t = 0; ref_t < kTemporalLayerCount; ++ref_t) {
              if (!maps.tlayer[static_cast<std::size_t>(m)]
                              [static_cast<std::size_t>(t)]
                              [static_cast<std::size_t>(ref_t)]) {
                continue;
              }
              bool& slot = retention.retain[static_cast<std::size_t>(x)]
                                           [static_cast<std::size_t>(ref_m)]
                                           [static_cast<std::size_t>(ref_t)];
              if (!slot) {
                slot = true;
                changed = true;
              }
            }
          }
        }
      }
    }
  }
  return true;
}

}  // namespace av2enc::tiles::svc

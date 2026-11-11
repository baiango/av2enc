#include "encoder/inter/mlayers/mlayers.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <sstream>
#include <utility>

namespace av2enc::inter::mlayers {
namespace {

constexpr std::uint32_t kMlayerMask = (1U << kMlayerCount) - 1U;
constexpr std::uint32_t kTlayerMask = (1U << kTlayerCount) - 1U;

bool fail(std::string& error, std::string message) {
  error = std::move(message);
  return false;
}

bool check_mask_range(std::uint32_t bits, std::uint32_t mask,
                      const char* name, std::string& error) {
  if ((bits & ~mask) != 0U) {
    std::ostringstream out;
    out << name << " has bits outside the legal range";
    return fail(error, out.str());
  }
  return true;
}

// OBU types used by the ordering rules (§6.2.2 Table 6.1).
constexpr int kObuSequenceHeader = 1;
constexpr int kObuTemporalDelimiter = 2;
constexpr int kObuMultiFrameHeader = 3;
constexpr int kObuClosedLoopKey = 4;
constexpr int kObuOpenLoopKey = 5;
constexpr int kObuLeadingTileGroup = 6;
constexpr int kObuRegularTileGroup = 7;
constexpr int kObuMetadataShort = 8;
constexpr int kObuMetadataGroup = 9;
constexpr int kObuSwitch = 10;
constexpr int kObuLeadingSef = 11;
constexpr int kObuRegularSef = 12;
constexpr int kObuLeadingTip = 13;
constexpr int kObuRegularTip = 14;
constexpr int kObuBufferRemovalTiming = 15;
constexpr int kObuLayerConfig = 16;
constexpr int kObuAtlasSegment = 17;
constexpr int kObuOperatingPointSet = 18;
constexpr int kObuBridgeFrame = 19;
constexpr int kObuMsdo = 20;
constexpr int kObuRasFrame = 21;
constexpr int kObuQuantizationMatrix = 22;
constexpr int kObuFilmGrain = 23;
constexpr int kObuContentInterpretation = 24;
constexpr int kObuPadding = 25;

bool is_frame_obu(int type) {
  switch (type) {
    case kObuClosedLoopKey:
    case kObuOpenLoopKey:
    case kObuLeadingTileGroup:
    case kObuRegularTileGroup:
    case kObuSwitch:
    case kObuLeadingSef:
    case kObuRegularSef:
    case kObuLeadingTip:
    case kObuRegularTip:
    case kObuBridgeFrame:
    case kObuRasFrame:
      return true;
    default:
      return false;
  }
}

bool is_global_allowed_obu(int type) {
  switch (type) {
    case kObuTemporalDelimiter:
    case kObuMetadataShort:
    case kObuMetadataGroup:
    case kObuBufferRemovalTiming:
    case kObuLayerConfig:
    case kObuAtlasSegment:
    case kObuOperatingPointSet:
    case kObuMsdo:
    case kObuPadding:
      return true;
    default:
      return false;
  }
}

bool is_layer_non_specific_obu(int type) {
  // Table 6.1 "Layer-specific" column: "N" entries.
  switch (type) {
    case kObuSequenceHeader:
    case kObuTemporalDelimiter:
    case kObuLayerConfig:
    case kObuAtlasSegment:
    case kObuOperatingPointSet:
    case kObuMsdo:
      return true;
    default:
      return false;
  }
}

}  // namespace

bool operator==(const LayerId& a, const LayerId& b) {
  return a.xlayer_id == b.xlayer_id && a.mlayer_id == b.mlayer_id &&
         a.tlayer_id == b.tlayer_id;
}

bool operator<(const LayerId& a, const LayerId& b) {
  if (a.xlayer_id != b.xlayer_id) return a.xlayer_id < b.xlayer_id;
  if (a.mlayer_id != b.mlayer_id) return a.mlayer_id < b.mlayer_id;
  return a.tlayer_id < b.tlayer_id;
}

bool validate_layer_id(const LayerId& id, std::string& error) {
  if (id.xlayer_id > kGlobalXLayerId) {
    return fail(error, "obu_xlayer_id is outside 0..31");
  }
  if (id.mlayer_id > kMaxMlayerId) {
    return fail(error, "obu_mlayer_id is outside 0..7");
  }
  if (id.tlayer_id > kMaxTlayerId) {
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
  const int max_m = std::clamp(max_mlayer_id, 0, kMaxMlayerId);
  const int max_t = std::clamp(max_tlayer_id, 0, kMaxTlayerId);
  for (int curr = 0; curr < kMlayerCount; ++curr) {
    for (int ref = 0; ref < kMlayerCount; ++ref) {
      maps.mlayer[static_cast<std::size_t>(curr)][static_cast<std::size_t>(ref)] =
          ref <= curr && curr <= max_m;
    }
    for (int curr_t = 0; curr_t < kTlayerCount; ++curr_t) {
      for (int ref_t = 0; ref_t < kTlayerCount; ++ref_t) {
        maps.tlayer[static_cast<std::size_t>(curr)]
                   [static_cast<std::size_t>(curr_t)]
                   [static_cast<std::size_t>(ref_t)] =
            curr <= max_m && ref_t <= curr_t && curr_t <= max_t;
      }
    }
  }
  return maps;
}

MLayerPresenceMap derive_mlayer_presence_map(
    const MLayerDependencyMap& mlayer, int /*max_mlayer_id*/) {
  MLayerPresenceMap presence{};
  // Exact §5.4.1 p58 loop. Outer mlayerId ascending means the source row
  // referenced for transitive closure has already been computed.
  for (int mlayer_id = 0; mlayer_id < kMlayerCount; ++mlayer_id) {
    for (int ref_mlayer = 0; ref_mlayer < kMlayerCount; ++ref_mlayer) {
      presence[static_cast<std::size_t>(mlayer_id)]
              [static_cast<std::size_t>(ref_mlayer)] = false;
      if (mlayer_id == ref_mlayer ||
          mlayer[static_cast<std::size_t>(mlayer_id)]
                 [static_cast<std::size_t>(ref_mlayer)]) {
        presence[static_cast<std::size_t>(mlayer_id)]
                [static_cast<std::size_t>(ref_mlayer)] = true;
        for (int dep = 0; dep < ref_mlayer; ++dep) {
          presence[static_cast<std::size_t>(mlayer_id)]
                  [static_cast<std::size_t>(dep)] |=
              presence[static_cast<std::size_t>(ref_mlayer)]
                      [static_cast<std::size_t>(dep)];
        }
      }
    }
  }
  return presence;
}

bool validate_mlayer_dependency_map(const MLayerDependencyMap& map,
                                    int max_mlayer_id, std::string& error) {
  if (max_mlayer_id < 0 || max_mlayer_id > kMaxMlayerId) {
    return fail(error, "max_mlayer_id is outside 0..7");
  }
  for (int curr = 0; curr < kMlayerCount; ++curr) {
    for (int ref = 0; ref < kMlayerCount; ++ref) {
      const bool dep =
          map[static_cast<std::size_t>(curr)][static_cast<std::size_t>(ref)];
      if (dep && (ref > curr || curr > max_mlayer_id ||
                  ref > max_mlayer_id)) {
        return fail(error, "MLayerDependencyMap has an out-of-range dependency");
      }
      if (curr <= max_mlayer_id && ref == curr && !dep) {
        return fail(error, "MLayerDependencyMap is not reflexive");
      }
      if (curr > max_mlayer_id && dep) {
        return fail(error,
                    "MLayerDependencyMap marks a layer above max_mlayer_id");
      }
    }
  }
  // Transitive closure: curr -> mid and mid -> ref implies curr -> ref.
  for (int curr = 0; curr < kMlayerCount; ++curr) {
    for (int mid = 0; mid < kMlayerCount; ++mid) {
      if (!map[static_cast<std::size_t>(curr)][static_cast<std::size_t>(mid)]) {
        continue;
      }
      for (int ref = 0; ref < kMlayerCount; ++ref) {
        if (map[static_cast<std::size_t>(mid)][static_cast<std::size_t>(ref)] &&
            !map[static_cast<std::size_t>(curr)]
                [static_cast<std::size_t>(ref)]) {
          return fail(error, "MLayerDependencyMap is not transitive");
        }
      }
    }
  }
  return true;
}

bool validate_tlayer_dependency_maps(const TLayerDependencyMaps& maps,
                                     int max_mlayer_id, int max_tlayer_id,
                                     std::string& error) {
  if (max_mlayer_id < 0 || max_mlayer_id > kMaxMlayerId) {
    return fail(error, "max_mlayer_id is outside 0..7");
  }
  if (max_tlayer_id < 0 || max_tlayer_id > kMaxTlayerId) {
    return fail(error, "max_tlayer_id is outside 0..3");
  }
  for (int m = 0; m < kMlayerCount; ++m) {
    const bool declared = m <= max_mlayer_id;
    for (int curr = 0; curr < kTlayerCount; ++curr) {
      for (int ref = 0; ref < kTlayerCount; ++ref) {
        const bool dep =
            maps[static_cast<std::size_t>(m)][static_cast<std::size_t>(curr)]
                [static_cast<std::size_t>(ref)];
        if (!declared) {
          if (dep) {
            return fail(error,
                        "TLayerDependencyMap is set for an undeclared embedded "
                        "layer");
          }
          continue;
        }
        if (dep && (ref > curr || curr > max_tlayer_id ||
                    ref > max_tlayer_id)) {
          return fail(error,
                      "TLayerDependencyMap has an out-of-range dependency");
        }
        if (curr <= max_tlayer_id && ref == curr && !dep) {
          return fail(error, "TLayerDependencyMap is not reflexive");
        }
      }
    }
    if (!declared) continue;
    for (int curr = 0; curr < kTlayerCount; ++curr) {
      for (int mid = 0; mid < kTlayerCount; ++mid) {
        if (!maps[static_cast<std::size_t>(m)][static_cast<std::size_t>(curr)]
                 [static_cast<std::size_t>(mid)]) {
          continue;
        }
        for (int ref = 0; ref < kTlayerCount; ++ref) {
          if (maps[static_cast<std::size_t>(m)]
                  [static_cast<std::size_t>(mid)]
                  [static_cast<std::size_t>(ref)] &&
              !maps[static_cast<std::size_t>(m)]
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

bool validate_dependency_maps(const LayerDependencyMaps& maps,
                              int max_mlayer_id, int max_tlayer_id,
                              std::string& error) {
  return validate_mlayer_dependency_map(maps.mlayer, max_mlayer_id, error) &&
         validate_tlayer_dependency_maps(maps.tlayer, max_mlayer_id,
                                         max_tlayer_id, error);
}

bool mlayer_dependencies_satisfied(std::uint32_t mlayer_map,
                                   const LayerDependencyMaps& maps,
                                   std::string& error) {
  if (!check_mask_range(mlayer_map, kMlayerMask, "mlayer_map", error)) {
    return false;
  }
  for (int curr = 0; curr < kMlayerCount; ++curr) {
    if ((mlayer_map & (1U << curr)) == 0) continue;
    for (int ref = 0; ref < kMlayerCount; ++ref) {
      if (maps.mlayer[static_cast<std::size_t>(curr)]
                     [static_cast<std::size_t>(ref)] &&
          (mlayer_map & (1U << ref)) == 0) {
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
    const std::array<std::uint8_t, kMlayerCount>& tlayer_map,
    const LayerDependencyMaps& maps, std::string& error) {
  if (!check_mask_range(mlayer_map, kMlayerMask, "mlayer_map", error)) {
    return false;
  }
  for (int m = 0; m < kMlayerCount; ++m) {
    const std::uint32_t bits = tlayer_map[static_cast<std::size_t>(m)];
    if (!check_mask_range(bits, kTlayerMask, "tlayer_map", error)) {
      return false;
    }
    if ((mlayer_map & (1U << m)) == 0) {
      if (bits != 0) {
        return fail(error,
                    "tlayer_map is non-zero for an absent embedded layer");
      }
      continue;
    }
    if (bits == 0) {
      return fail(error, "present embedded layer has an empty tlayer_map");
    }
    for (int curr = 0; curr < kTlayerCount; ++curr) {
      if ((bits & (1U << curr)) == 0) continue;
      for (int ref = 0; ref < kTlayerCount; ++ref) {
        if (maps.tlayer[static_cast<std::size_t>(m)]
                       [static_cast<std::size_t>(curr)]
                       [static_cast<std::size_t>(ref)] &&
            (bits & (1U << ref)) == 0) {
          std::ostringstream out;
          out << "tlayer_map selects m" << m << "/t" << curr
              << " without dependency t" << ref;
          return fail(error, out.str());
        }
      }
    }
  }
  return true;
}

SequenceLayerConfig default_sequence_layer_config(int max_mlayer_id,
                                                  int max_tlayer_id) {
  SequenceLayerConfig config;
  config.max_mlayer_id = std::clamp(max_mlayer_id, 0, kMaxMlayerId);
  config.max_tlayer_id = std::clamp(max_tlayer_id, 0, kMaxTlayerId);
  config.seq_max_mlayer_cnt = config.max_mlayer_id + 1;
  config.mlayer_dependency_present_flag = false;
  config.tlayer_dependency_present_flag = false;
  config.multi_tlayer_dependency_map_present_flag = false;
  config.dependencies =
      default_dependency_maps(config.max_mlayer_id, config.max_tlayer_id);
  return config;
}

bool validate_sequence_layer_config(const SequenceLayerConfig& config,
                                    std::string& error) {
  if (config.max_mlayer_id < 0 || config.max_mlayer_id > kMaxMlayerId) {
    return fail(error, "max_mlayer_id is outside 0..7");
  }
  if (config.max_tlayer_id < 0 || config.max_tlayer_id > kMaxTlayerId) {
    return fail(error, "max_tlayer_id is outside 0..3");
  }
  if (config.seq_max_mlayer_cnt < 1 ||
      config.seq_max_mlayer_cnt > config.max_mlayer_id + 1) {
    return fail(error, "SeqMaxMlayerCnt is outside 1..max_mlayer_id+1");
  }
  if (config.max_mlayer_id == 0 &&
      config.seq_max_mlayer_cnt != 1) {
    return fail(error,
                "max_mlayer_id == 0 infers SeqMaxMlayerCnt = 1 and codes no "
                "seq_max_mlayer_cnt_minus_1");
  }
  if (config.max_mlayer_id == 0 && config.mlayer_dependency_present_flag) {
    return fail(error,
                "mlayer_dependency_present_flag requires max_mlayer_id > 0");
  }
  if (config.max_tlayer_id == 0 &&
      (config.tlayer_dependency_present_flag ||
       config.multi_tlayer_dependency_map_present_flag)) {
    return fail(error,
                "tlayer dependency flags require max_tlayer_id > 0");
  }
  if (!config.tlayer_dependency_present_flag &&
      config.multi_tlayer_dependency_map_present_flag) {
    return fail(error,
                "multi_tlayer_dependency_map_present_flag requires "
                "tlayer_dependency_present_flag = 1");
  }
  if (config.max_mlayer_id == 0 &&
      config.multi_tlayer_dependency_map_present_flag) {
    return fail(error,
                "multi_tlayer_dependency_map_present_flag requires "
                "max_mlayer_id > 0");
  }
  if (!validate_dependency_maps(config.dependencies, config.max_mlayer_id,
                                config.max_tlayer_id, error)) {
    return false;
  }
  // When a flag is zero the spec infers the default map; accepting an
  // explicitly different map here would silently serialise something else.
  const LayerDependencyMaps defaults =
      default_dependency_maps(config.max_mlayer_id, config.max_tlayer_id);
  if (!config.mlayer_dependency_present_flag &&
      config.dependencies.mlayer != defaults.mlayer) {
    return fail(error,
                "MLayerDependencyMap differs from the inferred default but "
                "mlayer_dependency_present_flag = 0");
  }
  if (!config.tlayer_dependency_present_flag) {
    if (config.dependencies.tlayer != defaults.tlayer) {
      return fail(error,
                  "TLayerDependencyMap differs from the inferred default but "
                  "tlayer_dependency_present_flag = 0");
    }
  } else if (!config.multi_tlayer_dependency_map_present_flag) {
    for (int m = 1; m <= config.max_mlayer_id; ++m) {
      if (config.dependencies.tlayer[static_cast<std::size_t>(m)] !=
          config.dependencies.tlayer[0]) {
        return fail(error,
                    "multi_tlayer_dependency_map_present_flag = 0 requires all "
                    "embedded layers to use TLayerDependencyMap[0]");
      }
    }
  }
  return true;
}

const char* obu_type_name(int obu_type) {
  switch (obu_type) {
    case kObuSequenceHeader: return "sequence-header";
    case kObuTemporalDelimiter: return "temporal-delimiter";
    case kObuMultiFrameHeader: return "multi-frame-header";
    case kObuClosedLoopKey: return "closed-loop-key";
    case kObuOpenLoopKey: return "open-loop-key";
    case kObuLeadingTileGroup: return "leading-tile-group";
    case kObuRegularTileGroup: return "regular-tile-group";
    case kObuMetadataShort: return "metadata-short";
    case kObuMetadataGroup: return "metadata-group";
    case kObuSwitch: return "switch";
    case kObuLeadingSef: return "leading-sef";
    case kObuRegularSef: return "regular-sef";
    case kObuLeadingTip: return "leading-tip";
    case kObuRegularTip: return "regular-tip";
    case kObuBufferRemovalTiming: return "buffer-removal-timing";
    case kObuLayerConfig: return "layer-configuration-record";
    case kObuAtlasSegment: return "atlas-segment";
    case kObuOperatingPointSet: return "operating-point-set";
    case kObuBridgeFrame: return "bridge-frame";
    case kObuMsdo: return "msdo";
    case kObuRasFrame: return "ras-frame";
    case kObuQuantizationMatrix: return "quantization-matrix";
    case kObuFilmGrain: return "film-grain";
    case kObuContentInterpretation: return "content-interpretation";
    case kObuPadding: return "padding";
    default: return "reserved";
  }
}

// ---------------------------------------------------------------------------
// LayerStreamAuditor
// ---------------------------------------------------------------------------

LayerStreamAuditor::LayerStreamAuditor(SequenceLayerConfig config)
    : config_(std::move(config)) {
  presence_ = derive_mlayer_presence_map(config_.dependencies.mlayer,
                                         config_.max_mlayer_id);
}

bool LayerStreamAuditor::check_obu_constraints(
    const LayerObuRecord& r, std::string& error) const {
  if (!validate_layer_id(r.layer, error)) return false;
  if (r.obu_type < 0 || r.obu_type > 31) {
    return fail(error, "obu_type is outside f(5)");
  }
  const bool global = r.layer.xlayer_id == kGlobalXLayerId;
  if (global && !is_global_allowed_obu(r.obu_type)) {
    std::ostringstream out;
    out << obu_type_name(r.obu_type)
        << " is not allowed with obu_xlayer_id = GLOBAL_XLAYER_ID";
    return fail(error, out.str());
  }
  if (global && (r.layer.mlayer_id != 0 || r.layer.tlayer_id != 0)) {
    return fail(error,
                "GLOBAL_XLAYER_ID requires obu_mlayer_id = obu_tlayer_id = 0");
  }
  if ((r.obu_type == kObuTemporalDelimiter || r.obu_type == kObuMsdo) &&
      !global) {
    return fail(error,
                obu_type_name(r.obu_type) +
                    std::string(" requires obu_xlayer_id = GLOBAL_XLAYER_ID"));
  }
  if (is_layer_non_specific_obu(r.obu_type) &&
      (r.layer.mlayer_id != 0 || r.layer.tlayer_id != 0)) {
    std::ostringstream out;
    out << obu_type_name(r.obu_type)
        << " requires obu_mlayer_id = obu_tlayer_id = 0 (§6.2.2)";
    return fail(error, out.str());
  }
  if ((r.obu_type == kObuClosedLoopKey || r.obu_type == kObuOpenLoopKey ||
       r.obu_type == kObuSwitch || r.obu_type == kObuRasFrame) &&
      r.layer.tlayer_id != 0) {
    return fail(error,
                "CLK/OLK/SWITCH/RAS require obu_tlayer_id = 0 (§6.2.2)");
  }
  if (r.layer.mlayer_id > config_.max_mlayer_id) {
    std::ostringstream out;
    out << "obu_mlayer_id " << static_cast<int>(r.layer.mlayer_id)
        << " exceeds max_mlayer_id " << config_.max_mlayer_id;
    return fail(error, out.str());
  }
  if (r.layer.tlayer_id > config_.max_tlayer_id) {
    std::ostringstream out;
    out << "obu_tlayer_id " << static_cast<int>(r.layer.tlayer_id)
        << " exceeds max_tlayer_id " << config_.max_tlayer_id;
    return fail(error, out.str());
  }
  if (r.switch_or_ras ||
      r.obu_type == kObuSwitch || r.obu_type == kObuRasFrame) {
    // §6.4.1 p264: for any m != obu_mlayer_id,
    // MLayerDependencyMap[obu_mlayer_id][m] shall be 0.
    for (int m = 0; m < kMlayerCount; ++m) {
      if (m == r.layer.mlayer_id) continue;
      if (config_.dependencies.mlayer[static_cast<std::size_t>(
              r.layer.mlayer_id)][static_cast<std::size_t>(m)]) {
        return fail(
            error,
            "SWITCH/RAS frame selects an embedded layer with a dependency on "
            "another embedded layer");
      }
    }
  }
  if (r.has_inter_layer_ref) {
    if (r.inter_layer_ref.xlayer_id != r.layer.xlayer_id) {
      return fail(error,
                  "embedded-layer prediction cannot cross extended layers");
    }
    if (r.inter_layer_ref.mlayer_id >= r.layer.mlayer_id) {
      return fail(error,
                  "inter-layer prediction must reference a lower embedded "
                  "layer");
    }
    if (!presence_[static_cast<std::size_t>(r.layer.mlayer_id)]
                  [static_cast<std::size_t>(r.inter_layer_ref.mlayer_id)]) {
      return fail(error,
                  "inter-layer reference is absent from MLayerPresenceMap");
    }
  }
  if (is_frame_obu(r.obu_type) && r.seq_header_ref < 0) {
    return fail(error, "frame OBU has no seq_header_id_in_frame_header");
  }
  return true;
}

bool LayerStreamAuditor::check_availability(const LayerObuRecord& r,
                                            std::string& error) {
  if (r.obu_type == kObuSequenceHeader) {
    if (r.seq_header_id < 0) {
      return fail(error, "sequence header has no seq_header_id");
    }
    for (ActiveHeader& h : headers_) {
      if (!h.valid) continue;
      if (h.xlayer_id == r.layer.xlayer_id &&
          h.seq_header_id == r.seq_header_id) {
        if (h.max_mlayer_id != r.header_max_mlayer_id ||
            h.max_tlayer_id != r.header_max_tlayer_id) {
          return fail(error,
                      "repeated sequence header with different maxima in one "
                      "coded video sequence");
        }
        return true;
      }
    }
    for (ActiveHeader& h : headers_) {
      if (h.valid) continue;
      h.valid = true;
      h.xlayer_id = r.layer.xlayer_id;
      h.seq_header_id = r.seq_header_id;
      h.max_mlayer_id = r.header_max_mlayer_id;
      h.max_tlayer_id = r.header_max_tlayer_id;
      return true;
    }
    return fail(error, "too many active sequence headers");
  }

  if (is_frame_obu(r.obu_type)) {
    const ActiveHeader* found = nullptr;
    for (const ActiveHeader& h : headers_) {
      if (h.valid && h.xlayer_id == r.layer.xlayer_id &&
          h.seq_header_id == r.seq_header_ref) {
        found = &h;
        break;
      }
    }
    if (found == nullptr) {
      std::ostringstream out;
      out << "frame references unavailable sequence header " << r.seq_header_ref
          << " in " << layer_id_name(r.layer);
      return fail(error, out.str());
    }
    if (found->max_mlayer_id < r.layer.mlayer_id ||
        found->max_tlayer_id < r.layer.tlayer_id) {
      return fail(error,
                  "frame layer id exceeds the referenced sequence header "
                  "maxima");
    }
  }

  if (r.has_inter_layer_ref) {
    bool available = false;
    for (std::size_t i = 0; i < record_count_; ++i) {
      const LayerObuRecord& prior = records_[i].record;
      if (!is_frame_obu(prior.obu_type) || !prior.coded_output_frame) {
        continue;
      }
      if (prior.layer.xlayer_id == r.inter_layer_ref.xlayer_id &&
          prior.layer.mlayer_id == r.inter_layer_ref.mlayer_id) {
        available = true;
        break;
      }
    }
    if (!available) {
      std::ostringstream out;
      out << "inter-layer reference " << layer_id_name(r.inter_layer_ref)
          << " has no output frame unit earlier in the temporal unit";
      return fail(error, out.str());
    }
  }
  return true;
}

bool LayerStreamAuditor::observe(const LayerObuRecord& record,
                                 std::string& error) {
  if (!check_obu_constraints(record, error)) return false;
  if (record.obu_type == kObuTemporalDelimiter) {
    if (have_temporal_unit_ && !finish_temporal_unit(error)) return false;
    have_temporal_unit_ = true;
    record_count_ = 0;
    return true;
  }
  if (!have_temporal_unit_) {
    return fail(error, "OBU appears before the temporal delimiter");
  }
  if (!check_availability(record, error)) return false;
  if (record_count_ >= records_.size()) {
    return fail(error, "temporal unit has too many OBUs for this auditor");
  }
  records_[record_count_].record = record;
  records_[record_count_].frame_unit_start =
      record.frame_unit_start && is_frame_obu(record.obu_type);
  ++record_count_;
  return true;
}

bool LayerStreamAuditor::finish_temporal_unit(std::string& error) {
  if (!have_temporal_unit_) return true;
  have_temporal_unit_ = false;
  if (record_count_ == 0) {
    return fail(error, "temporal unit contains no coded output frame unit");
  }

  struct MlayerState {
    bool saw_frame_unit = false;
    bool first_is_clk = false;
    bool first_is_olk = false;
    bool saw_output = false;
    bool saw_non_output = false;
    int output_count = 0;
    int order_hint = -1;
    bool order_hint_set = false;
    bool saw_any_frame = false;
    int first_frame_is_clk = 0;
  };
  std::array<MlayerState, kMlayerCount> mlayers{};

  bool in_xlayer = false;
  int xlayer = -1;
  int last_mlayer = -1;
  int first_present_mlayer = -1;
  bool group_has_clk = false;
  bool group_has_olk = false;
  bool group_has_output = false;
  int group_order_hint = -1;
  bool group_order_hint_set = false;
  bool saw_frame_unit_in_group = false;

  auto flush_xlayer = [&]() -> bool {
    if (!in_xlayer) return true;
    if (!group_has_output) {
      std::ostringstream out;
      out << "extended layer x" << xlayer
          << " has no coded output frame unit";
      return fail(error, out.str());
    }
    if (first_present_mlayer >= 0) {
      const MlayerState& first =
          mlayers[static_cast<std::size_t>(first_present_mlayer)];
      if ((group_has_clk || group_has_olk) &&
          first.first_frame_is_clk == 0) {
        std::ostringstream out;
        out << "first coded frame unit of the lowest embedded layer m"
            << first_present_mlayer << " in x" << xlayer
            << " is not a CLK/OLK frame";
        return fail(error, out.str());
      }
    }
    if (group_has_clk && group_has_olk) {
      return fail(error,
                  "coded extended layer unit contains both CLK and OLK frames");
    }
    for (int m = 0; m < kMlayerCount; ++m) {
      const MlayerState& s = mlayers[static_cast<std::size_t>(m)];
      if (!s.saw_any_frame) continue;
      if (s.saw_non_output && !s.saw_output) {
        std::ostringstream out;
        out << "embedded layer m" << m
            << " has a non-output frame unit but no output frame unit";
        return fail(error, out.str());
      }
      if (!s.saw_output) {
        std::ostringstream out;
        out << "embedded layer m" << m << " has no coded output frame unit";
        return fail(error, out.str());
      }
      if (s.output_count > 1) {
        std::ostringstream out;
        out << "embedded layer m" << m
            << " has more than one coded output frame unit";
        return fail(error, out.str());
      }
    }
    return true;
  };

  for (std::size_t i = 0; i < record_count_; ++i) {
    const LayerObuRecord& r = records_[i].record;
    if (r.layer.xlayer_id == kGlobalXLayerId) {
      // Global OBUs are not part of a coded extended layer unit.
      continue;
    }
    if (r.obu_type == kObuSequenceHeader || r.obu_type == kObuLayerConfig ||
        r.obu_type == kObuOperatingPointSet || r.obu_type == kObuAtlasSegment) {
      if (saw_frame_unit_in_group) {
        return fail(error,
                    "sequence/config OBU appears after a frame unit in a coded "
                    "extended layer unit");
      }
      continue;
    }
    if (!is_frame_obu(r.obu_type)) continue;

    if (!in_xlayer || r.layer.xlayer_id != xlayer) {
      if (in_xlayer && !flush_xlayer()) return false;
      if (in_xlayer && r.layer.xlayer_id < xlayer) {
        return fail(error,
                    "coded extended layer units are not in ascending "
                    "obu_xlayer_id order");
      }
      in_xlayer = true;
      xlayer = r.layer.xlayer_id;
      last_mlayer = -1;
      first_present_mlayer = -1;
      group_has_clk = false;
      group_has_olk = false;
      group_has_output = false;
      group_order_hint = -1;
      group_order_hint_set = false;
      saw_frame_unit_in_group = false;
      mlayers = {};
    }

    if (!r.frame_unit_start) {
      // Continuation tile group: must belong to the current mlayer.
      if (r.layer.mlayer_id != last_mlayer) {
        return fail(error,
                    "tile-group continuation changes obu_mlayer_id inside a "
                    "frame unit");
      }
      continue;
    }

    if (r.layer.mlayer_id < last_mlayer) {
      return fail(error,
                  "embedded layers are not in ascending obu_mlayer_id order");
    }
    if (r.layer.mlayer_id != last_mlayer) {
      last_mlayer = r.layer.mlayer_id;
      if (first_present_mlayer < 0) first_present_mlayer = last_mlayer;
    }
    saw_frame_unit_in_group = true;

    MlayerState& s = mlayers[static_cast<std::size_t>(r.layer.mlayer_id)];
    const bool first_in_layer = !s.saw_frame_unit;
    s.saw_frame_unit = true;
    s.saw_any_frame = true;
    if (first_in_layer) {
      s.first_frame_is_clk = r.key_frame ? 1 : 0;
      s.first_is_olk = r.open_loop_key;
    }
    if (r.key_frame) {
      group_has_clk = true;
      if (!first_in_layer) {
        return fail(error,
                    "CLK frame is not the first coded frame unit of its "
                    "embedded layer");
      }
    }
    if (r.open_loop_key) {
      group_has_olk = true;
      if (!first_in_layer) {
        return fail(error,
                    "OLK frame is not the first coded frame unit of its "
                    "embedded layer");
      }
    }

    if (r.coded_output_frame) {
      ++s.output_count;
      s.saw_output = true;
      group_has_output = true;
      if (s.order_hint_set && s.order_hint != r.order_hint) {
        return fail(error,
                    "output frame units in one embedded layer disagree on "
                    "OrderHint");
      }
      s.order_hint = r.order_hint;
      s.order_hint_set = true;
      if (group_order_hint_set && group_order_hint != r.order_hint) {
        return fail(error,
                    "output frame units in one coded extended layer unit "
                    "disagree on OrderHint");
      }
      group_order_hint = r.order_hint;
      group_order_hint_set = true;
    } else {
      s.saw_non_output = true;
    }
  }
  if (!flush_xlayer()) return false;

  // §6.4.1 p261: the number of distinct obu_mlayer_id values present in the
  // coded video sequence is <= SeqMaxMlayerCnt. The note on p261 counts every
  // OBU, including OBU_SEQUENCE_HEADER (forced to embedded layer 0).
  std::array<bool, kMlayerCount> present_layers{};
  present_layers[0] = true;
  for (std::size_t i = 0; i < record_count_; ++i) {
    const LayerObuRecord& r = records_[i].record;
    if (r.layer.xlayer_id == kGlobalXLayerId) continue;
    if (is_frame_obu(r.obu_type) && r.frame_unit_start) {
      present_layers[static_cast<std::size_t>(r.layer.mlayer_id)] = true;
    }
  }
  int present_count = 0;
  for (bool present : present_layers) {
    if (present) ++present_count;
  }
  if (enforce_seq_max_mlayer_cnt_ &&
      present_count > config_.seq_max_mlayer_cnt) {
    std::ostringstream out;
    out << "temporal unit uses " << present_count
        << " distinct embedded layers but SeqMaxMlayerCnt is "
        << config_.seq_max_mlayer_cnt;
    return fail(error, out.str());
  }
  record_count_ = 0;
  return true;
}

bool LayerStreamAuditor::finish(std::string& error) {
  if (!have_temporal_unit_) return true;
  return finish_temporal_unit(error);
}

}  // namespace av2enc::inter::mlayers

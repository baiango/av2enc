#include "pipeline/video/video_lcr_ops.h"

#include "core/bitstream/basic_headers.h"
#include "formats/format.h"
#include "pipeline/svc/ops_full/lcr_ops_full.h"
#include "pipeline/tiles/svc_layers.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace av2enc::video {
namespace {

namespace ops_full = ::av2enc::svc::ops_full;
namespace layers = ::av2enc::tiles::svc;

using av2enc::basic::ObuHeader;
using av2enc::basic::ObuType;

std::uint8_t tlayer_map_for(const VideoConfig& config) {
  if (config.max_tlayer_id == 0U) return 0x01U;
  const unsigned count = static_cast<unsigned>(config.max_tlayer_id) + 1U;
  return static_cast<std::uint8_t>((1U << count) - 1U);
}

std::vector<std::uint8_t> frame_obu(const ObuType type,
                                    std::span<const std::uint8_t> payload,
                                    bool extension_flag,
                                    std::uint8_t xlayer_id) {
  ObuHeader header{type};
  header.extension_flag = extension_flag;
  if (extension_flag) {
    header.mlayer_id = 0;
    header.xlayer_id = xlayer_id;
  }
  return basic::make_obu(header, payload);
}

// Shared per-layer record: rep info, purpose, colour and the embedded-layer
// map for the single mlayer / temporal-layer configuration of this sequence.
void fill_xlayer_info(const VideoConfig& config, int profile,
                      ops_full::XLayerInfo& info) {
  info.rep_info_present = true;
  info.rep.max_pic_width = config.max_frame_width;
  info.rep.max_pic_height = config.max_frame_height;
  info.rep.format_info_present = true;
  info.rep.bit_depth_idc =
      static_cast<std::uint8_t>(fmt::bit_depth_idc(config.depth));
  info.rep.chroma_format_idc =
      static_cast<std::uint8_t>(fmt::chroma_format_idc(config.chroma));
  info.rep.cropping_window_present = false;
  info.purpose_present = true;
  info.purpose_id = 1;  // Table 6.6 (texture / viewport purpose)
  info.color_info_present = true;
  info.color.description_idc = 1;  // Table 6.13: BT.709 SDR
  info.color.full_range = false;
  info.embedded_layer_info_present = true;
  info.mlayer_map = 0x01U;  // embedded layer 0
  info.embedded.clear();
  ops_full::EmbeddedLayerInfo embedded;
  embedded.mlayer_id = 0;
  embedded.tlayer_map = tlayer_map_for(config);
  embedded.atlas_segment_id = 1;
  embedded.priority_order = 1;
  embedded.rendering_method = 0;
  embedded.layer_type = ops_full::kLayerTexture;
  embedded.auxiliary_type = ops_full::kAuxAlpha;
  // Explicit view selection exercises view_type = VIEW_EXPLICIT + view_id.
  embedded.view_type = ops_full::kViewExplicit;
  embedded.view_id = 1;
  embedded.dependent_layer_mask = 0;
  embedded.same_sh_max_resolution = true;
  info.embedded.push_back(embedded);
  (void)profile;
}

ops_full::ProfileTierLevelInfo ptl_for(const VideoConfig& config,
                                       int profile) {
  ops_full::ProfileTierLevelInfo ptl;
  ptl.seq_profile_idc = static_cast<std::uint8_t>(profile);
  ptl.max_level_idx = config.seq_level_idx;
  ptl.tier_flag = config.seq_tier;
  ptl.max_mlayer_count = config.seq_max_mlayer_cnt;
  return ptl;
}

ops_full::LayerConfigRecord make_global_lcr(const VideoConfig& config,
                                            int profile) {
  ops_full::LayerConfigRecord record;
  record.is_global = true;
  ops_full::GlobalLcrInfo& global = record.global;
  global.config_record_id = 1;
  global.xlayer_map = 0x01U;  // xlayer 0
  global.aggregate_info_present = true;
  global.aggregate.config_idc = 0;  // Annex A.3 configuration 0
  global.aggregate.level_idx = config.seq_level_idx;
  global.aggregate.max_tier_flag = config.seq_tier;
  global.aggregate.max_interop = 0;
  global.profile_tier_level_info_present = true;
  global.ptl[0] = ptl_for(config, profile);
  global.global_payload_present = true;
  global.dependent_xlayers_flag = false;
  global.global_atlas_id_present = true;
  global.global_atlas_id = 1;
  global.global_purpose_id = 1;
  global.doh_constraint_flag = true;
  global.enforce_tile_alignment_flag = true;

  ops_full::GlobalPayload payload;
  payload.xlayer_id = 0;
  payload.dependent_xlayer_mask = 0;
  fill_xlayer_info(config, profile, payload.info);
  global.payloads.push_back(std::move(payload));
  return record;
}

ops_full::LayerConfigRecord make_local_lcr(
    const VideoConfig& config, int profile,
    const ops_full::LayerConfigRecord& global) {
  ops_full::LayerConfigRecord record;
  record.is_global = false;
  ops_full::LocalLcrInfo& local = record.local;
  local.xlayer_id = 0;
  local.global_id = global.global.config_record_id;
  local.local_id = 2;
  local.profile_tier_level_info_present = true;
  local.ptl = ptl_for(config, profile);
  local.local_atlas_id_present = true;
  local.local_atlas_id = 1;
  local.xlayer = global.global.payloads[0].info;
  return record;
}

void fill_ops_point(const VideoConfig& config, int profile,
                    ops_full::OpsPoint& point) {
  point.op_intent = 1;  // OPSI_SCALABILITY (Table 6.12)
  point.aggregate.config_idc = 0;
  point.aggregate.level_idx = config.seq_level_idx;
  point.aggregate.max_tier_flag = config.seq_tier;
  point.aggregate.max_interop = 0;
  point.color.description_idc = 1;
  point.color.full_range = false;
  point.decoder_model_info_present = true;
  point.decoder_model.decoder_buffer_delay = 80000;
  point.decoder_model.encoder_buffer_delay = 10000;
  point.decoder_model.low_delay_mode_flag = false;
  point.initial_display_delay_present = true;
  point.initial_display_delay_minus_1 = 0;
  point.xlayer_map = 0x01U;
  point.ptl[0].seq_profile_idc = static_cast<std::uint8_t>(profile);
  point.ptl[0].level_idx = config.seq_level_idx;
  point.ptl[0].tier_flag = config.seq_tier;
  point.ptl[0].mlayer_count = config.seq_max_mlayer_cnt;
  point.layer_info[0].explicit_info = true;
  point.layer_info[0].mlayer_map = 0x01U;
  point.layer_info[0].tlayer_map[0] = tlayer_map_for(config);
}

ops_full::OperatingPointSet make_global_ops(const VideoConfig& config,
                                            int profile) {
  ops_full::OperatingPointSet set;
  set.xlayer_id = static_cast<std::uint8_t>(ops_full::kGlobalXLayerId);
  set.ops_id = 0;
  set.reset_flag = true;
  set.priority = 1;
  set.intent = 1;
  set.intent_present = true;
  set.ptl_present = true;
  set.color_info_present = true;
  set.mlayer_info_idc = 1;
  ops_full::OpsPoint point;
  fill_ops_point(config, profile, point);
  set.points.push_back(point);
  return set;
}

ops_full::OperatingPointSet make_local_ops(const VideoConfig& config,
                                           int profile) {
  ops_full::OperatingPointSet set;
  set.xlayer_id = 0;
  set.ops_id = 1;
  set.reset_flag = true;
  set.priority = 2;
  set.intent = 1;
  set.intent_present = true;
  set.ptl_present = true;
  set.color_info_present = true;
  set.mlayer_info_idc = 1;
  set.reserved_2bits = 0;
  ops_full::OpsPoint point;
  fill_ops_point(config, profile, point);
  // A local OPS does not carry the aggregate or xlayer map.
  point.aggregate = ops_full::OpsAggregateInfo{};
  point.xlayer_map = 0;
  set.points.push_back(point);
  return set;
}

}  // namespace

bool make_full_lcr_ops_obus(const VideoConfig& config,
                            std::vector<LcrOpsObu>& obus, std::string& error) {
  obus.clear();
  error.clear();

  const fmt::Format header_format{config.max_frame_width, config.max_frame_height,
                                  config.chroma, config.depth};
  const int profile = config.seq_profile_override >= 0
                          ? config.seq_profile_override
                          : fmt::profile_idc(header_format);
  if (profile < 0) {
    error = "full LCR/OPS: cannot derive seq_profile_idc from the pixel format";
    return false;
  }
  if (config.seq_level_idx <= 0U || config.seq_level_idx > 31U) {
    error = "full LCR/OPS: seq_level_idx outside 1..31";
    return false;
  }

  const ops_full::LayerConfigRecord global_lcr =
      make_global_lcr(config, profile);
  const ops_full::LayerConfigRecord local_lcr =
      make_local_lcr(config, profile, global_lcr);
  const ops_full::OperatingPointSet global_ops =
      make_global_ops(config, profile);
  const ops_full::OperatingPointSet local_ops =
      make_local_ops(config, profile);

  // §6.8/§6.10/§6.11: validate against the sequence-header values this stream
  // will actually write before returning any bytes.
  ops_full::StreamContext context;
  ops_full::XLayerSequenceInfo sequence;
  sequence.xlayer_id = 0;
  sequence.seq_profile_idc = static_cast<std::uint8_t>(profile);
  sequence.seq_level_idx = config.seq_level_idx;
  sequence.seq_tier = config.seq_tier;
  sequence.seq_max_mlayer_cnt = config.seq_max_mlayer_cnt;
  sequence.max_frame_width = config.max_frame_width;
  sequence.max_frame_height = config.max_frame_height;
  sequence.bit_depth_idc =
      static_cast<std::uint8_t>(fmt::bit_depth_idc(config.depth));
  sequence.chroma_format_idc =
      static_cast<std::uint8_t>(fmt::chroma_format_idc(config.chroma));
  sequence.monotonic_output_order = config.monotonic_output_order_flag;
  context.sequences.push_back(sequence);
  context.dependencies =
      layers::default_dependency_maps(config.max_mlayer_id, config.max_tlayer_id);
  context.dependencies_known = true;
  context.known_ops.push_back(ops_full::summarize_ops(global_ops));
  context.known_ops.push_back(ops_full::summarize_ops(local_ops));
  context.lcr_active = true;
  context.active_global_lcr_id =
      global_lcr.global.config_record_id;

  ops_full::ValidationReport report;
  if (!ops_full::validate_stream(global_lcr, global_ops, context, report) ||
      !ops_full::validate_local_lcr_matches_global(global_lcr, local_lcr,
                                                   report)) {
    error = "full LCR/OPS validation failed: " + report.errors.front();
    return false;
  }

  std::vector<std::uint8_t> payload;
  std::vector<std::uint8_t> bytes;

  if (!ops_full::write_lcr_payload(global_lcr, payload, error)) {
    error = "global LCR payload: " + error;
    return false;
  }
  bytes = frame_obu(ObuType::kLayerConfigurationRecord, payload,
                    /*extension_flag=*/true,
                    static_cast<std::uint8_t>(ops_full::kGlobalXLayerId));
  obus.push_back(LcrOpsObu{std::move(bytes), "global LCR"});

  if (!ops_full::write_lcr_payload(local_lcr, payload, error)) {
    error = "local LCR payload: " + error;
    return false;
  }
  std::vector<std::uint8_t> local_lcr_bytes =
      frame_obu(ObuType::kLayerConfigurationRecord, payload,
                /*extension_flag=*/false, 0);

  if (!ops_full::write_ops_payload(global_ops, payload, error)) {
    error = "global OPS payload: " + error;
    return false;
  }
  bytes = frame_obu(ObuType::kOperatingPointSet, payload,
                    /*extension_flag=*/true,
                    static_cast<std::uint8_t>(ops_full::kGlobalXLayerId));
  obus.push_back(LcrOpsObu{std::move(bytes), "global OPS"});

  // §7.3.6: the global LCR/OPS pair precedes any xlayer-specific record.
  obus.push_back(LcrOpsObu{std::move(local_lcr_bytes), "local LCR"});

  if (!ops_full::write_ops_payload(local_ops, payload, error)) {
    error = "local OPS payload: " + error;
    return false;
  }
  bytes = frame_obu(ObuType::kOperatingPointSet, payload,
                    /*extension_flag=*/false, 0);
  obus.push_back(LcrOpsObu{std::move(bytes), "local OPS"});
  return true;
}

}  // namespace av2enc::video

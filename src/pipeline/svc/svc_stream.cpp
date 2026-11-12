#include "pipeline/svc/svc_stream.h"

#include "pipeline/svc/svc_framing.h"
#include "pipeline/svc/svc_pipeline.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace av2enc::svc {
namespace {

bool fail(std::string& error, const std::string& text) {
  error = text;
  return false;
}

struct OwnedFrame {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
};

// Deterministic integer box filter for an exact integer downscale.  AV2
// layers here are independent encodes of the same source at lower resolution;
// the scaler is encoder-side only and never signalled.
bool downscale_plane(std::span<const std::uint8_t> src, int src_w, int src_h,
                     int dst_w, int dst_h, std::vector<std::uint8_t>& out,
                     std::string& error) {
  if (dst_w <= 0 || dst_h <= 0) {
    return fail(error, "spatial layer has a non-positive plane size");
  }
  if (src_w % dst_w != 0 || src_h % dst_h != 0) {
    std::ostringstream text;
    text << "spatial layer " << dst_w << "x" << dst_h
         << " is not an integer downscale of " << src_w << "x" << src_h;
    return fail(error, text.str());
  }
  const int fx = src_w / dst_w;
  const int fy = src_h / dst_h;
  const int count = fx * fy;
  out.resize(static_cast<std::size_t>(dst_w) * static_cast<std::size_t>(dst_h));
  for (int y = 0; y < dst_h; ++y) {
    for (int x = 0; x < dst_w; ++x) {
      int sum = 0;
      for (int sy = 0; sy < fy; ++sy) {
        const std::size_t row =
            static_cast<std::size_t>(y * fy + sy) *
            static_cast<std::size_t>(src_w);
        for (int sx = 0; sx < fx; ++sx) {
          sum += src[row + static_cast<std::size_t>(x * fx + sx)];
        }
      }
      out[static_cast<std::size_t>(y) * static_cast<std::size_t>(dst_w) +
          static_cast<std::size_t>(x)] =
          static_cast<std::uint8_t>((sum + count / 2) / count);
    }
  }
  return true;
}

bool scale_frame(const video::VideoInputFrame& src, int dst_w, int dst_h,
                 OwnedFrame& out, std::string& error) {
  out.width = static_cast<std::uint32_t>(dst_w);
  out.height = static_cast<std::uint32_t>(dst_h);
  const int src_w = static_cast<int>(src.width);
  const int src_h = static_cast<int>(src.height);
  if (!downscale_plane(src.y, src_w, src_h, dst_w, dst_h, out.y, error) ||
      !downscale_plane(src.u, src_w / 2, src_h / 2, dst_w / 2, dst_h / 2,
                       out.u, error) ||
      !downscale_plane(src.v, src_w / 2, src_h / 2, dst_w / 2, dst_h / 2,
                       out.v, error)) {
    return false;
  }
  return true;
}

struct FrameObu {
  ObuType type = ObuType::kRegularTileGroup;
  std::vector<std::uint8_t> payload;
};

struct LayerEncoded {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> sequence_payload;
  std::vector<FrameObu> frames;
  std::vector<video::VideoReconFrame> recon;
};

void append_obu(std::vector<std::uint8_t>& out,
                const std::vector<std::uint8_t>& obu) {
  out.insert(out.end(), obu.begin(), obu.end());
}

}  // namespace

bool make_layer_config_record_obu(const LayerPlan& plan,
                                  std::vector<std::uint8_t>& out,
                                  std::string& error) {
  out.clear();
  std::vector<std::uint8_t> payload;
  if (plan.sequences.empty()) {
    return fail(error, "LCR: the plan has no sequence headers");
  }
  if (!write_layer_config_record_payload(
          plan.lcr, plan.sequences.front().dependencies, payload, error)) {
    return fail(error, "LCR: " + error);
  }
  ObuHeader header;
  header.type = ObuType::kLayerConfigRecord;
  header.xlayer_id = static_cast<std::uint8_t>(layers::kGlobalXLayerId);
  if (!frame_obu(header, payload, out, error)) {
    return fail(error, "LCR: " + error);
  }
  return true;
}

bool make_operating_point_set_obu(const LayerPlan& plan,
                                  std::vector<std::uint8_t>& out,
                                  std::string& error) {
  out.clear();
  std::vector<std::uint8_t> payload;
  if (!write_operating_point_set_payload(plan.ops, payload, error)) {
    return fail(error, "OPS: " + error);
  }
  ObuHeader header;
  header.type = ObuType::kOperatingPointSet;
  header.xlayer_id = static_cast<std::uint8_t>(layers::kGlobalXLayerId);
  if (!frame_obu(header, payload, out, error)) {
    return fail(error, "OPS: " + error);
  }
  return true;
}

bool encode_svc_sequence(std::span<const video::VideoInputFrame> frames,
                         const SvcEncodeOptions& options,
                         SvcStreamResult& result, std::string& error) {
  error.clear();
  result = SvcStreamResult{};
  if (frames.empty()) return fail(error, "no input frames");

  LayerPlan plan;
  if (!build_layer_plan(options.config, plan, error)) return false;
  if (options.config.embedded_layers != 1 ||
      options.config.temporal_layers != 1) {
    return fail(error,
                "SVC encode currently emits one m0/t0 frame per spatial layer "
                "per temporal unit; --svc-embedded/--svc-temporal > 1 need the "
                "GOP scheduler (INTEGRATION.md step 4, order_frames)");
  }

  const std::size_t layer_count = plan.config.spatial_layers.size();
  std::vector<std::vector<OwnedFrame>> scaled(layer_count);
  std::vector<LayerEncoded> encoded(layer_count);
  for (std::size_t x = 0; x < layer_count; ++x) {
    const SpatialLayerSpec& spec = plan.config.spatial_layers[x];
    if (spec.width % 8 != 0 || spec.height % 8 != 0) {
      std::ostringstream text;
      text << "spatial layer x" << static_cast<int>(spec.xlayer_id) << " "
           << spec.width << "x" << spec.height
           << " must have dimensions divisible by 8";
      return fail(error, text.str());
    }
    scaled[x].resize(frames.size());
    std::vector<video::VideoInputFrame> layer_frames;
    layer_frames.reserve(frames.size());
    for (std::size_t i = 0; i < frames.size(); ++i) {
      if (!scale_frame(frames[i], spec.width, spec.height, scaled[x][i],
                       error)) {
        return false;
      }
      video::VideoInputFrame input;
      input.width = scaled[x][i].width;
      input.height = scaled[x][i].height;
      input.y = std::span<const std::uint8_t>(scaled[x][i].y);
      input.u = std::span<const std::uint8_t>(scaled[x][i].u);
      input.v = std::span<const std::uint8_t>(scaled[x][i].v);
      layer_frames.push_back(input);
    }

    video::VideoOptions layer_options = options.video;
    layer_options.tile_cols = 0;
    layer_options.tile_rows = 0;
    layer_options.seq_lcr_id = plan.sequences[x].seq_lcr_id;
    layer_options.max_tlayer_id = plan.sequences[x].max_tlayer_id;
    layer_options.max_mlayer_id = plan.sequences[x].max_mlayer_id;
    layer_options.seq_max_mlayer_cnt = plan.sequences[x].seq_max_mlayer_cnt;

    video::VideoSequenceResult sequence;
    if (!video::encode_video_sequence(layer_frames, layer_options, sequence,
                                      error)) {
      std::ostringstream text;
      text << "spatial layer x" << static_cast<int>(spec.xlayer_id)
           << " encode failed: " << error;
      return fail(error, text.str());
    }
    if (sequence.units.size() != frames.size()) {
      return fail(error, "spatial layer emitted an unexpected temporal-unit "
                         "count");
    }

    LayerEncoded& layer = encoded[x];
    layer.width = spec.width;
    layer.height = spec.height;
    layer.frames.reserve(sequence.units.size());
    for (std::size_t unit = 0; unit < sequence.units.size(); ++unit) {
      std::vector<ParsedObu> obus;
      if (!parse_obus(sequence.units[unit], obus, error)) {
        return fail(error, "spatial layer " + std::to_string(x) + ": " + error);
      }
      std::size_t frames_found = 0;
      for (const ParsedObu& obu : obus) {
        if (obu.header.type == ObuType::kTemporalDelimiter) continue;
        if (obu.header.type == ObuType::kSequenceHeader) {
          if (unit != 0) {
            return fail(error,
                        "spatial layer repeated its sequence header mid-stream");
          }
          layer.sequence_payload = obu.payload;
          continue;
        }
        if (!is_frame_obu(obu.header.type)) {
          return fail(error, "spatial layer unit carries an unexpected OBU");
        }
        FrameObu frame;
        frame.type = obu.header.type;
        frame.payload = obu.payload;
        layer.frames.push_back(std::move(frame));
        ++frames_found;
      }
      if (frames_found != 1) {
        return fail(error, "spatial layer unit must contain exactly one frame");
      }
      if (unit == 0 && layer.sequence_payload.empty()) {
        return fail(error, "spatial layer emitted no sequence header");
      }
    }
    layer.recon = std::move(sequence.recon);
  }

  // §7.3.6 / §7.3.7 order: TD, [LCR, OPS], then xlayer ascending with the
  // sequence header in the first temporal unit only.
  std::vector<Emission> order;
  if (!plan_emission_order(plan, static_cast<int>(frames.size()), order,
                           error)) {
    return false;
  }
  std::vector<std::uint8_t> lcr_obu;
  std::vector<std::uint8_t> ops_obu;
  if (!make_layer_config_record_obu(plan, lcr_obu, error)) return false;
  if (!make_operating_point_set_obu(plan, ops_obu, error)) return false;

  std::vector<std::vector<std::uint8_t>> units;
  std::vector<std::uint32_t> unit_payload_bytes;
  for (const Emission& emission : order) {
    switch (emission.kind) {
      case EmissionKind::kTemporalDelimiter: {
        units.emplace_back();
        unit_payload_bytes.push_back(0);
        ObuHeader header;
        header.type = ObuType::kTemporalDelimiter;
        header.xlayer_id = static_cast<std::uint8_t>(layers::kGlobalXLayerId);
        std::vector<std::uint8_t> obu;
        if (!frame_obu(header, std::span<const std::uint8_t>{}, obu, error)) {
          return false;
        }
        append_obu(units.back(), obu);
        break;
      }
      case EmissionKind::kLayerConfigRecord:
        append_obu(units.back(), lcr_obu);
        break;
      case EmissionKind::kOperatingPointSet:
        append_obu(units.back(), ops_obu);
        break;
      case EmissionKind::kSequenceHeader: {
        const std::size_t x = emission.layer.xlayer_id;
        if (x >= encoded.size() || encoded[x].sequence_payload.empty()) {
          return fail(error, "emission order names a layer with no sequence "
                             "header");
        }
        ObuHeader header;
        header.type = ObuType::kSequenceHeader;
        header.xlayer_id = static_cast<std::uint8_t>(x);
        header.mlayer_id = 0;
        header.tlayer_id = 0;
        std::vector<std::uint8_t> obu;
        if (!frame_obu(header, encoded[x].sequence_payload, obu, error)) {
          return false;
        }
        append_obu(units.back(), obu);
        break;
      }
      case EmissionKind::kFrame: {
        const std::size_t x = emission.layer.xlayer_id;
        const std::size_t tu = static_cast<std::size_t>(emission.temporal_unit);
        if (x >= encoded.size() || tu >= encoded[x].frames.size()) {
          return fail(error, "emission order names an absent frame");
        }
        const FrameObu& frame = encoded[x].frames[tu];
        ObuHeader header;
        header.type = frame.type;
        header.xlayer_id = static_cast<std::uint8_t>(x);
        header.mlayer_id = emission.layer.mlayer_id;
        header.tlayer_id = emission.layer.tlayer_id;
        std::vector<std::uint8_t> obu;
        if (!frame_obu(header, frame.payload, obu, error)) return false;
        append_obu(units.back(), obu);
        unit_payload_bytes.back() +=
            static_cast<std::uint32_t>(frame.payload.size());
        break;
      }
    }
  }

  // Annex F.3.1/F.3.2 selection.  --alllayers keeps the stream byte-for-byte.
  LayerSelection selection;
  if (options.spatial_layers >= 1) {
    selection.mode = SelectionMode::kUpToSpatialLayer;
    selection.max_xlayer = options.spatial_layers - 1;
    if (selection.max_xlayer >= static_cast<int>(layer_count)) {
      return fail(error, "--spatial-layers exceeds the configured layer count");
    }
    std::ostringstream label;
    label << "spatial-layers " << options.spatial_layers;
    result.selection_label = label.str();
  } else if (options.oppoint >= 0) {
    selection.mode = SelectionMode::kOperatingPoint;
    selection.op_index = options.oppoint;
    std::ostringstream label;
    label << "oppoint " << options.oppoint;
    result.selection_label = label.str();
  } else {
    selection.mode = SelectionMode::kAllLayers;
    result.selection_label = "alllayers";
  }
  layers::RetentionMap retention;
  if (!resolve_selection(plan, selection, retention, error)) return false;
  result.selected_layers = selected_layer_count(retention, plan);

  if (selection.mode == SelectionMode::kAllLayers) {
    result.units = std::move(units);
    result.unit_payload_bytes = std::move(unit_payload_bytes);
  } else {
    result.units.reserve(units.size());
    for (const std::vector<std::uint8_t>& unit : units) {
      std::vector<std::uint8_t> filtered;
      if (!extract_substream(unit, retention, filtered, error)) return false;
      result.units.push_back(std::move(filtered));
    }
    // Per-unit payload accounting is only reported for the full stream.
    result.unit_payload_bytes.clear();
  }

  result.layers.resize(layer_count);
  for (std::size_t x = 0; x < layer_count; ++x) {
    SvcLayerOutput& out = result.layers[x];
    out.xlayer_id = plan.config.spatial_layers[x].xlayer_id;
    out.width = encoded[x].width;
    out.height = encoded[x].height;
    out.recon = std::move(encoded[x].recon);
  }
  return true;
}

}  // namespace av2enc::svc

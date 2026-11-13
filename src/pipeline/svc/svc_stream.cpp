#include "pipeline/svc/svc_stream.h"

#include "pipeline/svc/svc_framing.h"
#include "pipeline/svc/svc_pipeline.h"

#include <algorithm>
#include <array>
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
  std::uint8_t tlayer_id = 0;
  bool keyframe = false;
};

struct LayerEncoded {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> sequence_payload;
  std::vector<FrameObu> frames;  // coding order
  std::vector<video::VideoReconFrame> recon;  // display order
};

void append_obu(std::vector<std::uint8_t>& out,
                const std::vector<std::uint8_t>& obu) {
  out.insert(out.end(), obu.begin(), obu.end());
}

// Parses one single-layer temporal unit (TD, [SEQ], one frame) and stores the
// frame.  `expect_sequence_header` is true for the layer's first unit.
bool parse_layer_unit(const std::vector<std::uint8_t>& unit,
                      bool expect_sequence_header, bool allow_sequence_header,
                      LayerEncoded& layer, std::string& error) {
  std::vector<ParsedObu> obus;
  if (!parse_obus(unit, obus, error)) return false;
  bool saw_sequence = false;
  int frames_found = 0;
  for (const ParsedObu& obu : obus) {
    if (obu.header.type == ObuType::kTemporalDelimiter) continue;
    if (obu.header.type == ObuType::kSequenceHeader) {
      if (!allow_sequence_header) {
        return fail(error, "spatial layer repeated its sequence header "
                           "mid-stream");
      }
      layer.sequence_payload = obu.payload;
      saw_sequence = true;
      continue;
    }
    if (!is_frame_obu(obu.header.type)) {
      return fail(error, "layer unit carries an unexpected OBU");
    }
    FrameObu frame;
    frame.type = obu.header.type;
    frame.payload = obu.payload;
    frame.tlayer_id = obu.header.tlayer_id;
    frame.keyframe = obu.header.type == ObuType::kClosedLoopKey;
    layer.frames.push_back(std::move(frame));
    ++frames_found;
  }
  if (frames_found != 1) {
    return fail(error, "layer unit must contain exactly one frame");
  }
  if (expect_sequence_header && !saw_sequence) {
    return fail(error, "layer emitted no sequence header");
  }
  return true;
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

  const int embedded_count = plan.config.embedded_layers;
  const int temporal_count = plan.config.temporal_layers;
  if (embedded_count > 3) {
    // Annex A caps seq_max_mlayer_cnt per profile: 1 for Main_420_10_IP0,
    // 2 for IP1, 3 for IP2.  The SVC 8-bit 4:2:0 path raises the profile with
    // the ladder, so three embedded layers is the defined maximum.
    return fail(error,
                "--svc-embedded 4..8 exceeds the Annex A profile limit for "
                "8-bit 4:2:0 (Main_420_10_IP2 allows 3 embedded layers)");
  }
  if (embedded_count > 1 && temporal_count > 1) {
    return fail(error,
                "--svc-embedded > 1 does not compose with --svc-temporal > 1: "
                "the refinement sequence would need a cross-mlayer reference "
                "schedule inside the hierarchical GOP; encode the embedded "
                "ladder with --svc-temporal 1");
  }
  // Hierarchical-P closed GOP: T temporal layers are carried by 1 << T
  // pictures (tlayers max_tlayer_id = T-1).  --svc-temporal 1 is the flat
  // key + P chain.
  const std::size_t temporal_units =
      temporal_count == 1 ? frames.size()
                          : (std::size_t{1} << temporal_count);
  if (temporal_count > 1 && frames.size() != temporal_units) {
    std::ostringstream text;
    text << "--svc-temporal " << temporal_count << " codes one closed GOP of "
         << temporal_units << " frames; the input has " << frames.size();
    return fail(error, text.str());
  }

  const std::size_t layer_count = plan.config.spatial_layers.size();
  std::vector<std::vector<OwnedFrame>> scaled(layer_count);
  // encoded[x][m]
  std::vector<std::vector<LayerEncoded>> encoded(
      layer_count, std::vector<LayerEncoded>(embedded_count));
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
    if (temporal_count > 1) {
      // The product hierarchical-P scheduler owns the coding order and the
      // per-OBU tlayer id; SVC only regroups the units into temporal units.
      layer_options.temporal_layers = static_cast<int>(temporal_units);
      layer_options.tlayer_drop = -1;
    }
    if (embedded_count > 1) {
      // One sequence header per xlayer: NumRefFrames must cover the highest
      // slot any embedded layer refreshes or references.
      layer_options.num_ref_frames = std::max(4, embedded_count);
      // Annex A: profile 0 allows one embedded layer, IP1 two, IP2 three.
      layer_options.seq_profile_override = embedded_count - 1;
      // The base chain pins slot 0; refinements own slots 1..M-1.  Without
      // this the rotating base slot collides with a refinement slot and an m0
      // frame would reference a slot holding m1 (§6.17.2 rejection).
      layer_options.fixed_write_slot = 0;
    }

    for (int m = 0; m < embedded_count; ++m) {
      LayerEncoded& layer = encoded[x][static_cast<std::size_t>(m)];
      layer.width = spec.width;
      layer.height = spec.height;

      video::VideoSequenceResult sequence;
      if (m == 0) {
        const int qp = std::min(
            255, static_cast<int>(options.video.base_q_idx) +
                     (embedded_count - 1) * options.embedded_qp_step);
        layer_options.base_q_idx = static_cast<std::uint16_t>(qp);
        if (!video::encode_video_sequence(layer_frames, layer_options,
                                          sequence, error)) {
          std::ostringstream text;
          text << "spatial layer x" << static_cast<int>(spec.xlayer_id)
               << " m0 encode failed: " << error;
          return fail(error, text.str());
        }
      } else {
        // §5.4.1 MLayerDependencyMap / §6.17.2: a higher embedded layer may
        // predict from the lower one.  The refinement sequence references the
        // previous mlayer's decoded picture at the same display index and
        // refreshes its own slot, leaving the lower layer's slot intact.
        const LayerEncoded& below = encoded[x][static_cast<std::size_t>(m - 1)];
        if (below.recon.size() != frames.size()) {
          return fail(error, "refinement source has an unexpected picture "
                             "count");
        }
        if (temporal_count > 1) {
          return fail(error, "embedded refinement needs --svc-temporal 1");
        }
        const int qp = std::min(
            255, static_cast<int>(options.video.base_q_idx) +
                     (embedded_count - 1 - m) * options.embedded_qp_step);
        video::VideoOptions refine_options = layer_options;
        refine_options.base_q_idx = static_cast<std::uint16_t>(qp);
        refine_options.temporal_layers = 0;
        refine_options.tlayer_drop = -1;
        video::RefinementOptions refinement;
        refinement.reference_slot = m - 1;
        refinement.refresh_slot = m;
        refinement.num_ref_frames = std::max(4, embedded_count);
        if (!video::encode_video_refinement_sequence(
                layer_frames, below.recon, refine_options, refinement,
                sequence, error)) {
          std::ostringstream text;
          text << "spatial layer x" << static_cast<int>(spec.xlayer_id)
               << " m" << m << " refinement failed: " << error;
          return fail(error, text.str());
        }
      }
      if (sequence.units.size() != frames.size()) {
        return fail(error, "layer emitted an unexpected temporal-unit count");
      }
      layer.frames.reserve(sequence.units.size());
      for (std::size_t unit = 0; unit < sequence.units.size(); ++unit) {
        if (!parse_layer_unit(sequence.units[unit], unit == 0, unit == 0, layer,
                              error)) {
          return fail(error, "spatial layer " + std::to_string(x) + " m" +
                                 std::to_string(m) + ": " + error);
        }
      }
      layer.recon = std::move(sequence.recon);
      if (temporal_count == 1) {
        // The flat chain is coded in display order, so the refinement encoder
        // can consume these pictures directly.
        if (layer.frames.size() != frames.size() ||
            layer.recon.size() != frames.size()) {
          return fail(error, "flat layer emitted an inconsistent frame count");
        }
      }
    }
  }

  // §7.3.6 / §7.3.7 schedule.  For the flat chain every configured (x, m)
  // layer emits one output unit per temporal unit.  A hierarchical GOP emits
  // one frame per xlayer per coding position, carrying that frame's tlayer id.
  std::vector<FrameRequest> requests;
  requests.reserve(layer_count * static_cast<std::size_t>(embedded_count) *
                   temporal_units);
  ReferencePolicy policy;
  std::vector<FrameReference> reference_checks;
  for (std::size_t unit = 0; unit < temporal_units; ++unit) {
    for (std::size_t x = 0; x < layer_count; ++x) {
      const std::uint8_t xid = plan.config.spatial_layers[x].xlayer_id;
      if (temporal_count > 1) {
        const FrameObu& frame = encoded[x][0].frames[unit];
        FrameRequest request;
        request.temporal_unit = static_cast<int>(unit);
        request.layer.xlayer_id = xid;
        request.layer.mlayer_id = 0;
        request.layer.tlayer_id = frame.tlayer_id;
        request.role = frame.keyframe ? FrameRole::kIntra : FrameRole::kInter;
        request.frame_index = static_cast<int>(unit);
        request.output = true;
        requests.push_back(request);
        continue;
      }
      for (int m = 0; m < embedded_count; ++m) {
        FrameRequest request;
        request.temporal_unit = static_cast<int>(unit);
        request.layer.xlayer_id = xid;
        request.layer.mlayer_id = static_cast<std::uint8_t>(m);
        request.layer.tlayer_id = 0;
        request.role = encoded[x][static_cast<std::size_t>(m)]
                           .frames[unit]
                           .keyframe
                    ? FrameRole::kIntra
                    : FrameRole::kInter;
        request.frame_index = static_cast<int>(unit);
        request.output = true;
        requests.push_back(request);
        if (m > 0) {
          // Same-TU cross-mlayer reference: emitted in ascending mlayer order,
          // so the lower layer's picture is decoded first.
          reference_checks.push_back(FrameReference{
              static_cast<int>(unit), request.layer, static_cast<int>(unit),
              layers::LayerId{xid, static_cast<std::uint8_t>(m - 1), 0}});
        }
      }
    }
  }
  std::vector<Emission> order;
  if (!order_frames(plan, requests, order, error)) return false;
  if (!validate_frame_references(plan, policy, reference_checks, order,
                                 error)) {
    return false;
  }

  std::vector<std::uint8_t> lcr_obu;
  std::vector<std::uint8_t> ops_obu;
  if (!make_layer_config_record_obu(plan, lcr_obu, error)) return false;
  if (!make_operating_point_set_obu(plan, ops_obu, error)) return false;

  // §7.3.6 order: TD, global LCR/OPS, then xlayer ascending.  Each layer's
  // sequence header is the first OBU of its layer unit in the first TU, which
  // is also the CLK's TU (the only CLK we emit per xlayer).
  std::vector<std::vector<std::uint8_t>> units;
  std::vector<std::uint32_t> unit_payload_bytes;
  int current_unit = -1;
  std::array<bool, layers::kExtendedLayerCount> sequence_written{};
  for (const Emission& emission : order) {
    switch (emission.kind) {
      case EmissionKind::kTemporalDelimiter: {
        units.emplace_back();
        unit_payload_bytes.push_back(0);
        current_unit = emission.temporal_unit;
        ObuHeader header;
        header.type = ObuType::kTemporalDelimiter;
        header.xlayer_id = static_cast<std::uint8_t>(layers::kGlobalXLayerId);
        std::vector<std::uint8_t> obu;
        if (!frame_obu(header, std::span<const std::uint8_t>{}, obu, error)) {
          return false;
        }
        append_obu(units.back(), obu);
        if (current_unit == 0) {
          append_obu(units.back(), lcr_obu);
          append_obu(units.back(), ops_obu);
        }
        break;
      }
      case EmissionKind::kLayerConfigRecord:
      case EmissionKind::kOperatingPointSet:
      case EmissionKind::kSequenceHeader:
        return fail(error, "frame schedule must own the global OBU placement");
      case EmissionKind::kFrame: {
        const std::size_t x = emission.layer.xlayer_id;
        const std::size_t m = emission.layer.mlayer_id;
        const std::size_t frame_index =
            static_cast<std::size_t>(emission.frame_index);
        if (x >= encoded.size() || m >= encoded[x].size() ||
            frame_index >= encoded[x][m].frames.size()) {
          return fail(error, "emission order names an absent frame");
        }
        if (current_unit == 0 && !sequence_written[x]) {
          if (encoded[x][m].sequence_payload.empty()) {
            return fail(error,
                        "layer has no sequence header for the first TU");
          }
          ObuHeader sequence_header;
          sequence_header.type = ObuType::kSequenceHeader;
          sequence_header.xlayer_id = static_cast<std::uint8_t>(x);
          sequence_header.mlayer_id = 0;
          sequence_header.tlayer_id = 0;
          std::vector<std::uint8_t> obu;
          if (!frame_obu(sequence_header, encoded[x][m].sequence_payload, obu,
                         error)) {
            return false;
          }
          append_obu(units.back(), obu);
          sequence_written[x] = true;
        }
        const FrameObu& frame = encoded[x][m].frames[frame_index];
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
    out.width = encoded[x][0].width;
    out.height = encoded[x][0].height;
    out.mlayer_recon.resize(embedded_count);
    for (int m = 0; m < embedded_count; ++m) {
      out.mlayer_recon[static_cast<std::size_t>(m)] =
          std::move(encoded[x][static_cast<std::size_t>(m)].recon);
    }
    out.recon = std::move(out.mlayer_recon.back());
  }
  return true;
}

}  // namespace av2enc::svc

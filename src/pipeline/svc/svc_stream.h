// End-to-end SVC layer encode: per-layer encode + §7.3.6 emission + Annex F.3.2
// selection.
//
// This is the core-side composition root described in INTEGRATION.md steps 1-4:
//
//   * every spatial layer is encoded as an independent video sequence with its
//     own sequence header (seq_lcr_id / max_mlayer_id / max_tlayer_id /
//     seq_max_mlayer_cnt from the LayerPlan);
//   * embedded layers (`embedded_layers > 1`) are a quality ladder inside one
//     xlayer: mlayer 0 is the normal CLK + P chain, and each higher mlayer is a
//     refinement sequence whose pictures predict from the lower mlayer's
//     decoded picture at the same display index (§5.4.1 MLayerDependencyMap /
//     §6.17.2).  This is the only decodable cross-layer prediction AV2 v1.0.0
//     defines: §6.8.2 makes `lcr_dependent_xlayers_flag = 0` a conformance
//     requirement, so cross-xlayer prediction/resampling cannot be signalled;
//   * temporal layers (`temporal_layers > 1`) use the hierarchical-P closed GOP
//     (`1 << temporal_layers` frames) per spatial/embedded layer and emit the
//     plan's coding order into temporal units;
//   * the global LCR and OPS are wrapped as OBU types 16/18 in the first
//     temporal unit;
//   * temporal units follow §7.3.6/§7.3.7: TD, [LCR, OPS], then xlayer
//     ascending, with each layer's sequence header in the first TU;
//   * --oppoint / --alllayers / --spatial-layers run resolve_selection() and
//     extract_substream().
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "pipeline/svc/svc_config.h"
#include "pipeline/video/video_frame.h"

namespace av2enc::svc {

struct SvcEncodeOptions {
  LayerConfig config;
  // Base-layer video options.  base_q_idx, ME/partition knobs and the debug
  // trace setting are shared by every layer.  tile_cols/tile_rows and the
  // sequence-layer fields are overwritten by this module.
  video::VideoOptions video;
  // Quality step between embedded layers.  mlayer 0 gets
  // `base_q_idx + (embedded_layers - 1) * embedded_qp_step` and the highest
  // mlayer keeps `base_q_idx`, so each refinement layer predicts from a lower
  // quality picture and improves it (clamped to 1..255).
  int embedded_qp_step = 12;
  // Output selection.  all_layers (the default) keeps the full stream;
  // otherwise exactly one of oppoint (>= 0) or spatial_layers (>= 1) applies.
  bool all_layers = true;
  int oppoint = -1;
  int spatial_layers = -1;
};

struct SvcLayerOutput {
  int xlayer_id = 0;
  int width = 0;
  int height = 0;
  // Highest embedded layer's reconstruction, in display order (one per input
  // frame; the temporal-layer path emits coding-order units but display-order
  // pictures).
  std::vector<video::VideoReconFrame> recon;
  // Per-embedded-layer reconstruction, `mlayer_recon[m]`, same ordering.
  std::vector<std::vector<video::VideoReconFrame>> mlayer_recon;
};

struct SvcStreamResult {
  // One IVF frame payload per temporal unit, in decode order.
  std::vector<std::vector<std::uint8_t>> units;
  std::vector<SvcLayerOutput> layers;
  // Payload (tile-group) bytes per temporal unit, summed over the layers that
  // contribute in that unit.  Sequence/LCR/OPS overhead is excluded.
  std::vector<std::uint32_t> unit_payload_bytes;
  int selected_layers = 0;
  std::string selection_label;
};

// LCR / OPS OBU wrappers (INTEGRATION.md step 2).  Global layer ids; the
// payload writers already include the extensible-OBU tail.
bool make_layer_config_record_obu(const LayerPlan& plan,
                                  std::vector<std::uint8_t>& out,
                                  std::string& error);
bool make_operating_point_set_obu(const LayerPlan& plan,
                                  std::vector<std::uint8_t>& out,
                                  std::string& error);

bool encode_svc_sequence(std::span<const video::VideoInputFrame> frames,
                         const SvcEncodeOptions& options,
                         SvcStreamResult& result, std::string& error);

}  // namespace av2enc::svc

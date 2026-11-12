// End-to-end SVC layer encode: per-layer encode + §7.3.6 emission + Annex F.3.2
// selection.
//
// This is the core-side composition root described in INTEGRATION.md steps 1-4:
//
//   * every spatial layer is encoded as an independent video sequence with its
//     own sequence header (seq_lcr_id / max_mlayer_id / max_tlayer_id /
//     seq_max_mlayer_cnt from the LayerPlan);
//   * the global LCR and OPS are wrapped as OBU types 16/18 in the first
//     temporal unit;
//   * temporal units follow plan_emission_order(): TD, [LCR, OPS], then
//     xlayer ascending with the sequence header in the first TU only;
//   * --oppoint / --alllayers / --spatial-layers run resolve_selection() and
//     extract_substream().
//
// Spatial-only path: layers are independent (no inter-layer prediction), which
// is what the plan's LCR declares.  AV2 layers are delivery packaging, not a
// compression tool; no BD-rate gain is claimed.
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
  std::vector<video::VideoReconFrame> recon;  // one per coded frame
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

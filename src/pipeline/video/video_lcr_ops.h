// §5.8 LCR / §5.10-§5.11 OPS carrier for the plain CPU `--video` writer
// (--video-lcr-full).
//
// The field-complete syntax lives in `pipeline/svc/ops_full`; this module only
// constructs the single-layer records that describe the video sequence header
// and frames the OBUs with the plain bitstream writer.  The historical
// `--video` path writes no LCR/OPS (seq_lcr_id = 0); the flag is opt-in and
// leaves every default stream byte-identical.
#pragma once

#include "pipeline/video/video_headers.h"

#include <cstdint>
#include <string>
#include <vector>

namespace av2enc::video {

// One length-delimited OBU (leb128 + obu_header + payload), ready to append to
// a temporal unit.
struct LcrOpsObu {
  std::vector<std::uint8_t> bytes;
  // Diagnostic label ("global LCR", "local LCR", ...).
  const char* name = "";
};

// Builds the full LCR/OPS carrier for `config`:
//
//   * global LCR  (config_record_id 1, aggregate info, PTL, atlas id,
//                  purpose, doh constraint, tile alignment, rep/colour/
//                  embedded-layer payload for xlayer 0);
//   * local LCR   (xlayer 0, lcr_global_id 1, local id 2, atlas id, same
//                  xlayer info as the parent);
//   * global OPS  (priority, intent, aggregate, PTL, colour, decoder model,
//                  initial display delay, xlayer map, mlayer/tlayer maps);
//   * local OPS   (xlayer 0, own priority/intent/PTL/colour/delay/maps).
//
// The records are validated with `ops_full::validate_stream()` against the
// sequence-header values in `config` before any bytes are returned.  `config`
// must already carry seq_lcr_id = 1 and the final level/tier/layer limits.
bool make_full_lcr_ops_obus(const VideoConfig& config,
                            std::vector<LcrOpsObu>& obus, std::string& error);

}  // namespace av2enc::video

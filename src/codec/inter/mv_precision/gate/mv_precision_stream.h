// Stream assembly for the MV-precision/AMVD decoder gate.
//
// The key and P1 tile payloads come from the M2 core verbatim.  This file
// writes the matching §5.4/§5.18 headers with the tools the M2 baseline
// disables:
//
//   * seq_enabled_motion_modes[INTERINTRA] = 1 and
//     seq_frame_motion_modes_present_flag = 1, so every inter frame codes the
//     §5.18.2 frame_enabled_motion_modes bit;
//   * enable_adaptive_mvd = enable_mvd_sign_derive = enable_flex_mvres = 1;
//   * P2 uses FrameMvPrecision = QUARTER with UsePerBlockMvPrecision = 1 and a
//     switchable interpolation filter, so the tile can exercise AMVD, a
//     coarser per-block precision, and derived mv_sign in one stream.
//
// P1 remains valid under the shared sequence header because its blocks are
// skip/GLOBALMV: no use_amvd, precision, or interp_filter symbol is coded for
// them, and its frame header pins is_filter_switchable = 0.

#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "codec/inter/mv_precision/mv_precision.h"

namespace av2enc::inter::mvprec::gate {

struct StreamConfig {
  int width = 0;
  int height = 0;
  int num_ref_frames = 2;
  int max_drl_bits_minus_1 = 4;
  mvprec::SequenceMotionModeSyntax motion_modes;
  mvprec::SequenceMvResSyntax mv_res;
};

struct StreamInput {
  StreamConfig config;
  std::vector<std::uint8_t> key_tile;
  std::vector<std::uint8_t> p1_tile;
  std::vector<std::uint8_t> p2_tile;
};

struct StreamOutput {
  std::vector<std::vector<std::uint8_t>> units;
  std::size_t sequence_bytes = 0;
};

bool build_stream(const StreamInput& input, StreamOutput* output,
                  std::string* error);

}  // namespace av2enc::inter::mvprec::gate

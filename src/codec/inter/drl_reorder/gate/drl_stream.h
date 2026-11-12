// Stream assembly for the DRL reorder decoder gate.
//
// The key and first P frames come from the M2 core (their tile payloads are
// reused verbatim).  This file writes the matching headers with
// disable_drl_reorder = 0 (and constrain_drl_reorder = 1 for the CONSTRAINT
// mode), then appends the reordering-enabled P2 frame written by drl_tile.cpp.
//
// Header layout follows the M2 writer (§5.4 / §5.5) exactly; the only
// difference is the §5.4.6 DrlReorder derivation and the two-slot explicit
// reference map on P2.

#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace av2enc::inter::drl::gate {

struct StreamInput {
  int width = 0;
  int height = 0;
  bool disable_drl_reorder = false;
  bool constrain_drl_reorder = true;
  int num_total_refs = 2;
  std::vector<std::uint8_t> key_tile;
  std::vector<std::uint8_t> p1_tile;
  std::vector<std::uint8_t> p2_tile;
};

struct StreamOutput {
  std::vector<std::vector<std::uint8_t>> units;  // one temporal unit per frame
  std::uint32_t sequence_bits = 0;
};

bool build_drl_stream(const StreamInput& input, StreamOutput* output,
                      std::string* error);

}  // namespace av2enc::inter::drl::gate

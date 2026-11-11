// Complete minimal AV2 still-picture frames that contain sub-8 coding blocks.
//
// The tile payload comes from sub8_syntax.cpp; the OBU/header wrapping reuses
// the production M1 header writer (src/encoder/m1/m1_headers.cpp).  Nothing in
// the M1 writer itself is bypassed: the frame header, sequence header and CLK
// OBU are the ones every M1 stream uses.  Only the tile payload is built here,
// because the current m1_syntax.cpp writer cannot express a chroma-less
// sub-8 leaf.
#pragma once

#include "encoder/m1/m1_headers.h"
#include "encoder/m1/sub8/sub8.h"
#include "encoder/m1/sub8/sub8_syntax.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace av2enc::m1::sub8 {

// M1Config with every tool that adds tile syntax disabled, matching what
// sub8_syntax.cpp emits (TX_MODE_LARGEST, chroma DCT-only, no CDEF/CCSO).
M1Config minimal_config(const FrameGeometry& geometry,
                        const SyntaxOptions& options);

// Writes one complete temporal unit (TD + sequence header + CLK) whose tile
// payload uses the supplied superblock trees.  Optionally returns the leaves
// parsed back by the independent reader so callers can assert round-trip
// agreement.
bool make_minimal_frame(const FrameGeometry& geometry,
                        std::span<const Node> superblocks,
                        const SyntaxOptions& options,
                        std::vector<std::uint8_t>& frame,
                        std::vector<DecodedLeaf>* round_trip, std::string& error);

}  // namespace av2enc::m1::sub8

// Gate-only hook: routes the M2 residual reconstruction through ddt::
// inverse_block() and lets a build-directory copy of m2_headers.cpp write the
// enable_inter_ddt bit from an environment variable.
//
// This header is *not* part of the M2 core.  See gate/README.md.
#pragma once

#include <cstdint>

#include "encoder/tx_variants/tx_variants.h"

namespace av2enc::m2::ddt::gate {

// True when DDT_GATE_ENABLE=1.  Read once; used only by the patched header
// writer so one binary can produce both the flag-0 and flag-1 stream.
bool enable_inter_ddt();

// Drop-in replacement for txv::inverse_transform() in the gate's build-dir
// copy of inter_residual.cpp.  Always uses the DDT-aware inverse (the header
// flag decides whether a real decoder agrees with it).
void inverse_transform(const std::int32_t* levels, const txv::TxParams& params,
                       std::int32_t* dequant_scratch,
                       std::int16_t* residual_coded);

// Counters printed at process exit, so the gate can assert that DDT actually
// ran on real blocks.
std::uint64_t ddt_blocks();
std::uint64_t ddt_directions();

}  // namespace av2enc::m2::ddt::gate

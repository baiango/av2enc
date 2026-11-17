// Precompiled header for the dev build (AV2ENC_PCH=ON, CMakePresets "dev").
//
// Contains only standard-library headers, deliberately:
//
//  - No project header. Project headers are guarded by AV2ENC_* macros, and a
//    PCH bakes the macro state it was compiled with. av2enc_core defines
//    AV2ENC_PRIMITIVES but the per-component build_and_test.sh gates do not, so
//    a project header here would hand one of them the wrong configuration.
//
//  - No masking of missing includes in the release/CI build: PCH is OFF there,
//    so a TU that relies on this file to compile still fails in the build that
//    produces the pinned streams.
//
// This buys the front-end cost that dominates mid-size TUs (71% of
// partition.cpp's compile time is parsing), without touching code generation.

#ifndef AV2ENC_PCH_H
#define AV2ENC_PCH_H

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

// Project headers that are macro-independent and included by 50-80 TUs each.
// They must not branch on AV2ENC_* (that would bake a target's configuration
// into the PCH); the sanitizer/TSan CI jobs build without PCH as the check that
// a missing include has not been masked here.
#include "core/bitstream/bitwriter.h"
#include "spec_tables/spec_math.h"
#include "codec/entropy/av2/av2_entropy.h"

#endif  // AV2ENC_PCH_H

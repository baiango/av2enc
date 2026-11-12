// Umbrella header for the AV2 §7.13 / §5.20.7 "missing bits" reference set.
//
//   dc_intra_subsampled.h     §7.13.2.11  subsampled DC intra
//   intra_filter_corner.h     §7.13.2.14  filter corner
//   dc_sign_horz_vert.h       §5.20.7.27  dc_sign_horz_vert writer/reader
//   save_subpu.h              §7.13.3.15  save subPU size
//
// Spec-only reference code.  Adds files only; nothing in src/encoder/m1/still_*,
// src/main.cpp or CMakeLists.txt is modified.

#pragma once

#include "codec/intra/supplemental/dc_intra_subsampled.h"
#include "codec/intra/supplemental/dc_sign_horz_vert.h"
#include "codec/intra/supplemental/intra_filter_corner.h"
#include "codec/intra/supplemental/save_subpu.h"

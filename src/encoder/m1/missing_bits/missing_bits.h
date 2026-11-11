// Umbrella header for the AV2 §7.13 / §5.20.7 "missing bits" reference set.
//
//   dc_intra_subsampled.h     §7.13.2.11  subsampled DC intra
//   intra_filter_corner.h     §7.13.2.14  filter corner
//   dc_sign_horz_vert.h       §5.20.7.27  dc_sign_horz_vert writer/reader
//   save_subpu.h              §7.13.3.15  save subPU size
//
// Spec-only reference code.  Adds files only; nothing in src/encoder/m1/m1_*,
// src/main.cpp or CMakeLists.txt is modified.

#pragma once

#include "encoder/m1/missing_bits/dc_intra_subsampled.h"
#include "encoder/m1/missing_bits/dc_sign_horz_vert.h"
#include "encoder/m1/missing_bits/intra_filter_corner.h"
#include "encoder/m1/missing_bits/save_subpu.h"

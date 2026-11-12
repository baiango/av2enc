// Pinned goldens for src/encoder/m1/missing_bits/.
//
// Regenerate with:
//   src/encoder/m1/missing_bits/build_and_test.sh --dump-goldens
//
// The values are normative outputs of the reference processes, not captures
// of the production encoder (except the gate MD5s, which are pinned decoder
// outputs and are listed in README.md).

#pragma once

#include <cstdint>

namespace av2enc::still::missing_bits::goldens {

// §7.13.2.11 DC subsampled: one trace per interesting regime.
struct DcGolden {
  int w;
  int h;
  int bit_depth;
  bool have_above;
  bool have_left;
  int step_w;
  int step_h;
  int sum;
  int count;
  int avg;
};

// Filled from --dump-goldens.  Includes both-active >32, one-active >32,
// both-active <=32 (production-identical), and odd extents.
inline constexpr DcGolden kDcGoldens[] = {
    {64, 64, 8, true, true, 2, 2, 7232, 64, 113},
    {64, 32, 8, true, true, 2, 1, 6544, 64, 102},
    {48, 40, 8, true, true, 2, 2, 4776, 44, 108},
    {33, 7, 8, true, true, 2, 1, 2137, 24, 89},
    {63, 37, 8, true, false, 2, 0, 3712, 32, 116},
    {37, 63, 8, false, true, 0, 2, 3520, 32, 110},
    {32, 32, 8, true, true, 1, 1, 6400, 64, 100},
    {16, 8, 8, true, true, 1, 1, 1116, 24, 46},
    {4, 4, 8, false, false, 0, 0, 0, 0, 128},
    {64, 64, 10, true, true, 2, 2, 12352, 64, 193},
    {64, 64, 12, true, false, 2, 0, 7040, 32, 220},
    {7, 5, 8, true, true, 1, 1, 273, 12, 23},
};

// §7.13.2.14 filter corner: {left0, above_m1, above0, expected}.
inline constexpr int kCornerGoldens[][4] = {
    {0, 0, 0, 0},
    {255, 255, 255, 255},
    {10, 30, 200, 77},
    {127, 128, 129, 128},
    {1, 2, 3, 2},
    {64, 192, 96, 122},
};

// §7.13.3.15 save_subPU: {plane, x, y, w, h, sub, pu_w, pu_h, tx,
//                            luma_row, luma_col, rows, cols}.
struct SubPuGolden {
  int plane;
  int x;
  int y;
  int w;
  int h;
  int sub;
  int pu_w;
  int pu_h;
  int tx;
  int luma_row;
  int luma_col;
  int rows;
  int cols;
};

inline constexpr SubPuGolden kSubPuGoldens[] = {
    // 32x16 luma PU split into two 16x16 sub-PUs.
    {0, 0, 0, 16, 16, 0, 32, 16, 2, 0, 0, 4, 4},
    {0, 16, 0, 16, 16, 0, 32, 16, 2, 0, 4, 4, 4},
    // 16x8 4:2:0 chroma PU split into two 8x8 sub-PUs.
    {1, 0, 0, 8, 8, 1, 16, 8, 1, 0, 0, 4, 4},
    {1, 8, 0, 8, 8, 1, 16, 8, 1, 0, 4, 4, 4},
    {2, 0, 0, 8, 8, 1, 16, 8, 1, 0, 0, 4, 4},
    // 64x64 region inside a 128x128 PuWidth/PuHeight (no early return).
    {0, 0, 0, 64, 64, 0, 128, 128, 4, 0, 0, 16, 16},
};

// `dc_sign_horz_vert` entropy bytes, one sign per symbol:
// signs = {0, 1, 1, 0, 1, 0, 0, 1, 1, 1}, ptype = 0, isHidden = 0, ctx = 0,
// default TileDcSignCdf.  Hex is the §8.2 tile payload including the trailing
// marker bit and byte padding.
inline constexpr const char* kSignFrozenHex = "64200000";
inline constexpr const char* kSignAdaptiveHex = "64700000";

// Minimal gate streams (gate_dc_sign_horz_vert.cpp), pinned 2026-11-12
// against tools/bin/dav2d-macos (f9dafbc):
//   [0] hv_pos   sign 0, normative ctx 0             (strict 1)
//   [1] hv_zero  no dc_sign_horz_vert symbol         (strict 1)
//   [2] hv_ctx1  sign 1 coded with ctx 1             (must be rejected)
//   [3] hv_neg   sign 1, normative ctx 0             (strict 0 only; the
//                pinned dav2d rejects negative 1D coefficients under
//                strict 1 -- a decoder-side check, see README.md)
inline constexpr const char* kGateIvMd5[4] = {
    "ada97e3f12b8582c65e5c2ac0826b0ec",
    "a398d9cfb44d6ed8ceed044121e6a65b",
    "75d363fcdc1ce67f53eae2445675340b",
    "9275c76e44a3492229750c3737c53d6d"};
inline constexpr const char* kGateDecodedMd5[4] = {
    "7a1349d9de949b949a05682fc606af86",
    "54d3959bbd3c80c05af8f900e417a069",
    "rejected",
    "ebfbfaa4b8ecc8eb96547ca22a23c986"};

}  // namespace av2enc::still::missing_bits::goldens

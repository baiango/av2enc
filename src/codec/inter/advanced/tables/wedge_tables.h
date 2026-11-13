// Generated from spec/AV2_Spec_v1.0.0.pdf by
// src/encoder/inter/advanced/tables/extract_spec_tables.py.
// Do not hand-edit; rerun the script instead.
// AV2 v1.0.0 wedge and inter-intra tables.
//
// §5.20.7.18  Wedge_Angle_Dist_2_Index[20][4]
// §7.13.3.27  Wedge_Cos_Lut / Wedge_Sin_Lut / distance weights / codebook
// §7.13.3.29  Ii_Weights_1d[128]
// §9.2        Wedge_Bits[BLOCK_SIZES] (0 = block has no wedge modes)
#ifndef AV2ENC_INTER_ADVANCED_TABLES_WEDGE_TABLES_H_
#define AV2ENC_INTER_ADVANCED_TABLES_WEDGE_TABLES_H_

#include <cstdint>

// Host and device passes need different storage classes for a
// namespace-scope table: CUDA cannot odr-use a host `constexpr` array from
// device code ("identifier is undefined in device code").  Same pattern as
// deblock_core.h / mc_subpel_tables.h.
#if defined(__CUDA_ARCH__)
#define AV2ENC_ADV_TABLE static __device__ constexpr
#else
#define AV2ENC_ADV_TABLE static constexpr
#endif

namespace av2enc::inter::adv::tables {

inline constexpr int kWedgeAngles = 20;
inline constexpr int kWedgeTypes = 68;
inline constexpr int kNumWedgeDist = 4;
inline constexpr int kWedgeBldLutSize = 32;
inline constexpr int kWedgeBoundaryTypes = 2;
inline constexpr int kBlockSizes = 29;

AV2ENC_ADV_TABLE std::int8_t kWedgeCosLut[20] = {
       4,    4,    4,    2,    2,    0,   -2,   -2,   -4,   -4,   -4,   -4,   -4,   -2,   -2,    0,    2,    2,    4,    4,
};

AV2ENC_ADV_TABLE std::int8_t kWedgeSinLut[20] = {
       0,   -1,   -2,   -2,   -4,   -4,   -4,   -2,   -2,   -1,    0,    1,    2,    2,    4,    4,    4,    2,    2,    1,
};

AV2ENC_ADV_TABLE std::int8_t kPosDist2BldWeight[32] = {
       8,    8,    9,    9,   10,   10,   11,   11,   12,   12,   12,   13,   13,   13,   14,   14,
      14,   14,   14,   15,   15,   15,   15,   15,   15,   15,   15,   15,   16,   16,   16,   16,
};

AV2ENC_ADV_TABLE std::int8_t kNegDist2BldWeight[32] = {
       8,    8,    7,    7,    6,    6,    5,    5,    4,    4,    4,    3,    3,    3,    2,    2,
       2,    2,    2,    1,    1,    1,    1,    1,    1,    1,    1,    1,    0,    0,    0,    0,
};

AV2ENC_ADV_TABLE std::int8_t kWedgeAngleDist2Index[20][4] = {
    {   -1,     0,     1,     2},
    {    3,     4,     5,     6},
    {    7,     8,     9,    10},
    {   11,    12,    13,    14},
    {   15,    16,    17,    18},
    {   -1,    19,    20,    21},
    {   22,    23,    24,    25},
    {   26,    27,    28,    29},
    {   30,    31,    32,    33},
    {   34,    35,    36,    37},
    {   -1,    38,    39,    40},
    {   -1,    41,    42,    43},
    {   -1,    44,    45,    46},
    {   -1,    47,    48,    49},
    {   -1,    50,    51,    52},
    {   -1,    53,    54,    55},
    {   -1,    56,    57,    58},
    {   -1,    59,    60,    61},
    {   -1,    62,    63,    64},
    {   -1,    65,    66,    67},
};

AV2ENC_ADV_TABLE std::int8_t kWedgeBits[29] = {
       0,    0,    0,    4,    4,    4,    4,    4,    4,    4,    4,    4,
       4,    0,    0,    0,    0,    0,    0,    0,    0,    4,    4,    4,
       4,    0,    0,    4,    4,
};

AV2ENC_ADV_TABLE std::int16_t kWedgeBlockWidth[29] = {
       4,    4,    8,    8,    8,   16,   16,   16,   32,   32,
      32,   64,   64,   64,  128,  128,  128,  256,  256,    4,
      16,    8,   32,   16,   64,    4,   32,    8,   64,
};

AV2ENC_ADV_TABLE std::int16_t kWedgeBlockHeight[29] = {
       4,    8,    4,    8,   16,    8,   16,   32,   16,   32,
      64,   32,   64,  128,   64,  128,  256,  128,  256,   16,
       4,   32,    8,   64,   16,   32,    4,   64,    8,
};

// §9.2 enum_block_size order (BLOCK_4X4 = 0, BLOCK_8X8 = 3, ...).
inline constexpr int kBlock8x8 = 3;
inline constexpr int kBlock16x16 = 6;
inline constexpr int kBlock64x64 = 12;

AV2ENC_ADV_TABLE std::int8_t kIiWeights1d[128] = {
      60,   58,   56,   54,   52,   50,   48,   47,   45,   44,   42,   41,
      39,   38,   37,   35,   34,   33,   32,   31,   30,   29,   28,   27,
      26,   25,   24,   23,   22,   22,   21,   20,   19,   19,   18,   18,
      17,   16,   16,   15,   15,   14,   14,   13,   13,   12,   12,   12,
      11,   11,   10,   10,   10,    9,    9,    9,    8,    8,    8,    8,
       7,    7,    7,    7,    6,    6,    6,    6,    6,    5,    5,    5,
       5,    5,    4,    4,    4,    4,    4,    4,    4,    4,    3,    3,
       3,    3,    3,    3,    3,    3,    3,    2,    2,    2,    2,    2,
       2,    2,    2,    2,    2,    2,    2,    2,    2,    2,    1,    1,
       1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
       1,    1,    1,    1,    1,    1,    1,    1,
};

// §7.13.3.27 Wedge_Codebook[WEDGE_TYPES][3] = { { angle, xoff, yoff } }.
// `angle` indexes kWedgeCosLut / kWedgeSinLut.
AV2ENC_ADV_TABLE std::int8_t kWedgeCodebook[kWedgeTypes][3] = {
    { 0,  5,  4},
    { 0,  6,  4},
    { 0,  7,  4},
    { 1,  4,  4},
    { 1,  5,  4},
    { 1,  6,  4},
    { 1,  7,  4},
    { 2,  4,  4},
    { 2,  5,  4},
    { 2,  6,  4},
    { 2,  7,  4},
    { 3,  4,  4},
    { 3,  5,  4},
    { 3,  6,  4},
    { 3,  7,  4},
    { 4,  4,  4},
    { 4,  4,  3},
    { 4,  4,  2},
    { 4,  4,  1},
    { 5,  4,  3},
    { 5,  4,  2},
    { 5,  4,  1},
    { 6,  4,  4},
    { 6,  4,  3},
    { 6,  4,  2},
    { 6,  4,  1},
    { 7,  4,  4},
    { 7,  3,  4},
    { 7,  2,  4},
    { 7,  1,  4},
    { 8,  4,  4},
    { 8,  3,  4},
    { 8,  2,  4},
    { 8,  1,  4},
    { 9,  4,  4},
    { 9,  3,  4},
    { 9,  2,  4},
    { 9,  1,  4},
    {10,  3,  4},
    {10,  2,  4},
    {10,  1,  4},
    {11,  3,  4},
    {11,  2,  4},
    {11,  1,  4},
    {12,  3,  4},
    {12,  2,  4},
    {12,  1,  4},
    {13,  3,  4},
    {13,  2,  4},
    {13,  1,  4},
    {14,  4,  5},
    {14,  4,  6},
    {14,  4,  7},
    {15,  4,  5},
    {15,  4,  6},
    {15,  4,  7},
    {16,  4,  5},
    {16,  4,  6},
    {16,  4,  7},
    {17,  5,  4},
    {17,  6,  4},
    {17,  7,  4},
    {18,  5,  4},
    {18,  6,  4},
    {18,  7,  4},
    {19,  5,  4},
    {19,  6,  4},
    {19,  7,  4},
};

}  // namespace av2enc::inter::adv::tables

#undef AV2ENC_ADV_TABLE

#endif  // AV2ENC_INTER_ADVANCED_TABLES_WEDGE_TABLES_H_

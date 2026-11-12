// AV2 IBP normative constant tables.
//
// Single source of truth for the CPU reference (src/encoder/ibp/ibp.cpp) and
// the CUDA device code (src/cuda/ibp/ibp_predict_device.cuh).  The CPU module
// expands these macros into constexpr arrays; the device header expands them
// into __device__ const arrays.  They therefore cannot drift.
//
// Source: AV2 Bitstream & Decoding Process Specification v1.0.0
//   SHA-256 e9916f09...b533f, commit 38f28e7.
//
//   Dr_Intra_Derivative[90]   §7.13.2.8
//   Dr_Interp_Filter[32][4]   §7.13.2.8
//   Div_Lut[129]              §7.13.3.22 (resolve_divisor)
//   Ibp_Weights[5][16]        §7.13.2.12
//   Ii_Weights_1d[128]        §7.13.3.29 (inter-intra mask, NOT IBP)
//
// These are transcribed verbatim.  Do not recompute them from tan() or a
// floating-point reciprocal: the published integer tables are normative.
//
// Note on naming: the user-facing task text mentions "IBP weights
// (Ii_Weights_1d/§7.13.3.29)".  In the v1.0.0 specification those are two
// different tools:
//   * IBP (intra bi-prediction) blends two *directional intra* predictions and
//     uses ibp_weights() (§7.13.2.9) plus Ibp_Weights (§7.13.2.12).
//   * Ii_Weights_1d is the *inter-intra* luma mask (§7.13.3.29).
// Both tables are provided here so the requested data is present, but the IBP
// code path uses only the former two.

#ifndef AV2ENC_ENCODER_IBP_IBP_TABLES_H_
#define AV2ENC_ENCODER_IBP_IBP_TABLES_H_

// §7.13.2.8.  Index = pAngle in degrees (0..89 reachable).
#define AV2ENC_IBP_DR_INTRA_DERIVATIVE_TABLE                            \
  0, 4096, 2048, 1365, 1024, 819, 682, 585, 512, 455, 409, 409, 409,    \
      372, 341, 292, 273, 256, 227, 215, 204, 186, 178, 170, 157, 151,  \
      146, 136, 132, 128, 117, 110, 107, 99, 97, 97, 93, 87, 83, 81,    \
      77, 74, 73, 69, 66, 64, 62, 59, 56, 55, 53, 50, 49, 47, 44, 42,   \
      42, 41, 38, 37, 35, 32, 31, 30, 28, 27, 26, 24, 23, 22, 20, 19,   \
      18, 16, 15, 14, 12, 11, 10, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1

// §7.13.2.8.  Luma IDIF taps, indexed by fractional position 0..31.
#define AV2ENC_IBP_DR_INTERP_FILTER_TABLE                               \
  {0, 128, 0, 0},        {-2, 127, 4, -1},   {-3, 125, 8, -2},        \
      {-5, 123, 13, -3}, {-6, 121, 17, -4},  {-7, 118, 22, -5},        \
      {-9, 116, 27, -6}, {-9, 112, 32, -7},  {-10, 109, 37, -8},       \
      {-11, 106, 41, -8}, {-11, 102, 46, -9}, {-12, 98, 52, -10},      \
      {-12, 94, 56, -10}, {-12, 90, 61, -11}, {-12, 85, 66, -11},      \
      {-12, 81, 71, -12}, {-12, 76, 76, -12}, {-12, 71, 81, -12},      \
      {-11, 66, 85, -12}, {-11, 61, 90, -12}, {-10, 56, 94, -12},      \
      {-10, 52, 98, -12}, {-9, 46, 102, -11}, {-8, 41, 106, -11},      \
      {-8, 37, 109, -10}, {-7, 32, 112, -9},  {-6, 27, 116, -9},       \
      {-5, 22, 118, -7},  {-4, 17, 121, -6},  {-3, 13, 123, -5},       \
      {-2, 8, 125, -3},   {-1, 4, 127, -2}

// §7.13.3.22 Div_Lut[129].
#define AV2ENC_IBP_DIV_LUT_TABLE                                        \
  512, 508, 504, 500, 496, 493, 489, 485, 482, 478, 475, 471, 468, 465, \
      462, 458, 455, 452, 449, 446, 443, 440, 437, 434, 431, 428, 426,  \
      423, 420, 417, 415, 412, 410, 407, 405, 402, 400, 397, 395, 392,  \
      390, 388, 386, 383, 381, 379, 377, 374, 372, 370, 368, 366, 364,  \
      362, 360, 358, 356, 354, 352, 350, 349, 347, 345, 343, 341, 340,  \
      338, 336, 334, 333, 331, 329, 328, 326, 324, 323, 321, 320, 318,  \
      317, 315, 314, 312, 311, 309, 308, 306, 305, 303, 302, 301, 299,  \
      298, 297, 295, 294, 293, 291, 290, 289, 287, 286, 285, 284, 282,  \
      281, 280, 279, 278, 277, 275, 274, 273, 272, 271, 270, 269, 267,  \
      266, 265, 264, 263, 262, 261, 260, 259, 258, 257, 256

// §7.13.2.12 Ibp_Weights[5][16].  Row index is log2(extent) - 2.
#define AV2ENC_IBP_WEIGHTS_TABLE                                        \
  {96, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},                   \
      {86, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},             \
      {77, 90, 102, 115, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},          \
      {71, 78, 86, 92, 100, 107, 114, 121, 0, 0, 0, 0, 0, 0, 0, 0},    \
      {68, 72, 76, 79, 83, 87, 90, 94, 98, 102, 106, 109, 113, 117, 121, \
       124}

// §7.13.3.29 Ii_Weights_1d[128].  Inter-intra luma mask, not IBP.
#define AV2ENC_IBP_II_WEIGHTS_1D_TABLE                                  \
  60, 58, 56, 54, 52, 50, 48, 47, 45, 44, 42, 41, 39, 38, 37, 35, 34,   \
      33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 22, 21, 20, 19,   \
      19, 18, 18, 17, 16, 16, 15, 15, 14, 14, 13, 13, 12, 12, 12, 11,   \
      11, 10, 10, 10, 9, 9, 9, 8, 8, 8, 8, 7, 7, 7, 7, 6, 6, 6, 6, 6,   \
      5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3,    \
      3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1,    \
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1

#endif  // AV2ENC_ENCODER_IBP_IBP_TABLES_H_

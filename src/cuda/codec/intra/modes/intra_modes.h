#ifndef AV2ENC_CUDA_INTRA_MODES_INTRA_MODES_H_
#define AV2ENC_CUDA_INTRA_MODES_INTRA_MODES_H_

// AV2 v1.0.0 full intra prediction (M1): DC / V / H / directional with angle
// deltas / smooth / smooth-V / smooth-H / Paeth, plus the normative intra edge
// filter (corner filter, filter-type selection, edge filtering).
//
// Normative source: AV2 Bitstream & Decoding Process Specification v1.0.0
//   §7.13.2.1   General (AboveRow / LeftCol construction, MAX_X / MAX_Y)
//   §7.13.2.2   Basic intra (Paeth)
//   §7.13.2.7   Directional intra + edge filtering
//   §7.13.2.8   Single directional prediction (IDIF / bilinear)
//   §7.13.2.10  DC intra
//   §7.13.2.13  Smooth intra
//   §7.13.2.14  Filter corner
//   §7.13.2.15-18 Intra filter type / strength / edge filter
//
// Scope of this component (M1):
//   * 8-bit samples, luma (plane 0, IDIF) and chroma (plane > 0, bilinear).
//   * Block dimensions 4, 8, 16, 32, 64 per axis (square or rectangular).
//   * Directional modes with angle_delta in [-MAX_ANGLE_DELTA, +MAX_ANGLE_DELTA]
//     (ANGLE_STEP = 3 degrees).
//   * Wide-angle mapping (WAIP) for rectangular blocks.
//   * MrlIndex = 0, DIP, IBP, CfL, MHCCP, and secondary MRL are out of scope
//     (M4b); they are rejected / not represented.
//
// Bit-exactness: all arithmetic is integer, mirroring the spec's Round2 /
// Clip1 / resolve_divisor formulas. No FP, no FMA contraction sensitivity.
//
// Every function is pure: device pointers + cudaStream_t in, cudaError_t out.
// No global state, no allocations, no device synchronisation.
//
// API summary
// -----------
//   intra_predict(block, d_pred, stream)            one block
//   intra_predict_residual(block, d_pred, d_resid)  one block + residual
//   intra_predict_batch(d_jobs, count, stream)      many independent blocks
//
// The batch entry point takes a device array of IntraJob descriptors. Jobs
// must be mutually independent: no job may read a plane region another job in
// the same launch writes. Prediction input is the *reconstructed* plane.

#include <cstdint>

#include <cuda_runtime.h>

#if defined(__CUDACC__)
#define AV2ENC_INTRA_HD __host__ __device__
#else
#define AV2ENC_INTRA_HD
#endif

namespace av2enc::cuda::intra_modes {

// ---------------------------------------------------------------------------
// Normative constants (AV2 v1.0.0)
// ---------------------------------------------------------------------------

inline constexpr int kMaxBlockDim = 64;    // largest transform extent per axis
inline constexpr int kMaxAngleDelta = 3;   // MAX_ANGLE_DELTA
inline constexpr int kAngleStep = 3;       // ANGLE_STEP, degrees
inline constexpr int kIntraModes = 13;     // INTRA_MODES
inline constexpr int kModeToAngleCount = 13;
inline constexpr int kDrDerivativeCount = 90;
inline constexpr int kDivLutBits = 7;      // DIV_LUT_BITS
inline constexpr int kDivLutPrecBits = 9;  // DIV_LUT_PREC_BITS
inline constexpr int kDivLutNum = 129;     // DIV_LUT_NUM
inline constexpr int kEdgeTaps = 5;        // INTRA_EDGE_TAPS
inline constexpr int kEdgeKernels = 3;     // INTRA_EDGE_KERNELS
inline constexpr int kEdgeMaxPx = 2 * kMaxBlockDim + 1;  // sz <= 129
inline constexpr int kBlendWeightMax = 32;               // BLEND_WEIGHT_MAX

// Intra modes, values match the bitstream YMode/UVMode enums (intra_modes.json).
enum IntraMode : int {
  kDcPred = 0,
  kVpred = 1,
  kHpred = 2,
  kD45Pred = 3,
  kD135Pred = 4,
  kD113Pred = 5,
  kD157Pred = 6,
  kD203Pred = 7,
  kD67Pred = 8,
  kSmoothPred = 9,
  kSmoothVPred = 10,
  kSmoothHPred = 11,
  kPaethPred = 12,
};

AV2ENC_INTRA_HD constexpr bool is_directional_mode(int mode) {
  return mode >= kVpred && mode <= kD67Pred;
}

// ---------------------------------------------------------------------------
// Normative table data, defined once.  The host constexpr accessors below and
// the device table copies in intra_modes.cu both expand these macros, so the
// two copies cannot drift.  Values are transcribed verbatim from the AV2
// v1.0.0 attachments; do not recompute them.
// ---------------------------------------------------------------------------

// Mode_To_Angle[13] (attachment mode_to_angle.h).
#define AV2ENC_INTRA_MODE_TO_ANGLE_TABLE \
    0, 90, 180, 45, 135, 113, 157, 203, 67, 0, 0, 0, 0

// Dr_Intra_Derivative[90] (attachment dr_intra_derivative.h).  Index = pAngle
// in degrees; 0..89 is the reachable range.
#define AV2ENC_INTRA_DR_DERIVATIVE_TABLE \
    0, 4096, 2048, 1365, 1024, 819, 682, 585, 512, 455, 409, 409, \
    409, 372, 341, 292, 273, 256, 227, 215, 204, 186, 178, 170, \
    157, 151, 146, 136, 132, 128, 117, 110, 107, 99, 97, 97, \
    93, 87, 83, 81, 77, 74, 73, 69, 66, 64, 62, 59, \
    56, 55, 53, 50, 49, 47, 44, 42, 42, 41, 38, 37, \
    35, 32, 31, 30, 28, 27, 26, 24, 23, 22, 20, 19, \
    18, 16, 15, 14, 12, 11, 10, 10, 10, 9, 8, 7, \
    6, 5, 4, 3, 2, 1

// Dr_Interp_Filter[32][4] (§7.13.2.8 IDIF taps).
#define AV2ENC_INTRA_DR_INTERP_TABLE \
    {0, 128, 0, 0}, \
    {-2, 127, 4, -1}, \
    {-3, 125, 8, -2}, \
    {-5, 123, 13, -3}, \
    {-6, 121, 17, -4}, \
    {-7, 118, 22, -5}, \
    {-9, 116, 27, -6}, \
    {-9, 112, 32, -7}, \
    {-10, 109, 37, -8}, \
    {-11, 106, 41, -8}, \
    {-11, 102, 46, -9}, \
    {-12, 98, 52, -10}, \
    {-12, 94, 56, -10}, \
    {-12, 90, 61, -11}, \
    {-12, 85, 66, -11}, \
    {-12, 81, 71, -12}, \
    {-12, 76, 76, -12}, \
    {-12, 71, 81, -12}, \
    {-11, 66, 85, -12}, \
    {-11, 61, 90, -12}, \
    {-10, 56, 94, -12}, \
    {-10, 52, 98, -12}, \
    {-9, 46, 102, -11}, \
    {-8, 41, 106, -11}, \
    {-8, 37, 109, -10}, \
    {-7, 32, 112, -9}, \
    {-6, 27, 116, -9}, \
    {-5, 22, 118, -7}, \
    {-4, 17, 121, -6}, \
    {-3, 13, 123, -5}, \
    {-2, 8, 125, -3}, \
    {-1, 4, 127, -2}

// Intra_Edge_Kernel[3][5] (§7.13.2.18).
#define AV2ENC_INTRA_EDGE_KERNEL_TABLE \
    {0, 4, 8, 4, 0}, {0, 5, 6, 5, 0}, {2, 4, 4, 4, 2}

// Div_Lut[129] (§7.13.3.22): round(65536 / (128 + f)).
#define AV2ENC_INTRA_DIV_LUT_TABLE \
    512, 508, 504, 500, 496, 493, 489, 485, 482, 478, 475, 471, \
    468, 465, 462, 458, 455, 452, 449, 446, 443, 440, 437, 434, \
    431, 428, 426, 423, 420, 417, 415, 412, 410, 407, 405, 402, \
    400, 397, 395, 392, 390, 388, 386, 383, 381, 379, 377, 374, \
    372, 370, 368, 366, 364, 362, 360, 358, 356, 354, 352, 350, \
    349, 347, 345, 343, 341, 340, 338, 336, 334, 333, 331, 329, \
    328, 326, 324, 323, 321, 320, 318, 317, 315, 314, 312, 311, \
    309, 308, 306, 305, 303, 302, 301, 299, 298, 297, 295, 294, \
    293, 291, 290, 289, 287, 286, 285, 284, 282, 281, 280, 279, \
    278, 277, 275, 274, 273, 272, 271, 270, 269, 267, 266, 265, \
    264, 263, 262, 261, 260, 259, 258, 257, 256

AV2ENC_INTRA_HD constexpr int mode_to_angle(int mode) {
  constexpr int table[kModeToAngleCount] = {AV2ENC_INTRA_MODE_TO_ANGLE_TABLE};
  return table[mode];
}

AV2ENC_INTRA_HD constexpr int dr_intra_derivative(int angle) {
  constexpr int table[kDrDerivativeCount] = {
      AV2ENC_INTRA_DR_DERIVATIVE_TABLE};
  return table[angle];
}

AV2ENC_INTRA_HD constexpr int dr_interp_filter(int shift, int tap) {
  constexpr int table[32][4] = {AV2ENC_INTRA_DR_INTERP_TABLE};
  return table[shift][tap];
}

AV2ENC_INTRA_HD constexpr int intra_edge_kernel(int strength_index, int tap) {
  constexpr int table[kEdgeKernels][kEdgeTaps] = {
      AV2ENC_INTRA_EDGE_KERNEL_TABLE};
  return table[strength_index][tap];
}

AV2ENC_INTRA_HD constexpr int div_lut(int f) {
  constexpr int table[kDivLutNum] = {AV2ENC_INTRA_DIV_LUT_TABLE};
  return table[f];
}

// WAIP_WH_RATIO_*_THRES (§5.20.7.30).
inline constexpr int kWaipRatio2Thres = 61;
inline constexpr int kWaipRatio4Thres = 73;
inline constexpr int kWaipRatio8Thres = 82;
inline constexpr int kWaipRatio16Thres = 86;

// wide_angle_mapping(mode, w, h, pAngle), reduced to the angle output. The
// returned mode is not used by prediction, so only pAngle is mapped.
AV2ENC_INTRA_HD constexpr int wide_angle_map(int pAngle, int w, int h) {
  if ((h == 2 * w && pAngle < kWaipRatio2Thres) ||
      (h == 4 * w && pAngle < kWaipRatio4Thres) ||
      (h == 8 * w && pAngle < kWaipRatio8Thres) ||
      (h == 16 * w && pAngle < kWaipRatio16Thres)) {
    return 180 + pAngle;
  }
  if ((w == 2 * h && pAngle > 270 - kWaipRatio2Thres) ||
      (w == 4 * h && pAngle > 270 - kWaipRatio4Thres) ||
      (w == 8 * h && pAngle > 270 - kWaipRatio8Thres) ||
      (w == 16 * h && pAngle > 270 - kWaipRatio16Thres)) {
    return pAngle - 180;
  }
  return pAngle;
}

// pAngle for a block, after wide-angle mapping. Non-directional modes return
// Mode_To_Angle (0 for DC/smooth/Paeth) and are not mapped.
AV2ENC_INTRA_HD constexpr int intra_pangle(int mode, int angle_delta,
                                           int width, int height) {
  if (!is_directional_mode(mode)) {
    return mode_to_angle(mode);
  }
  return wide_angle_map(mode_to_angle(mode) + angle_delta * kAngleStep, width,
                        height);
}

// Intra edge filter strength selection (§7.13.2.17). `delta` is the angle
// difference in degrees (angleAbove or angleLeft).
AV2ENC_INTRA_HD constexpr int intra_edge_strength(int width, int height,
                                                  int filter_type, int delta) {
  const int d = delta < 0 ? -delta : delta;
  const int blk = width + height;
  int strength = 0;
  if (filter_type == 0) {
    if (blk <= 8) {
      if (d >= 56) strength = 1;
    } else if (blk <= 12) {
      if (d >= 40) strength = 1;
    } else if (blk <= 16) {
      if (d >= 40) strength = 1;
    } else if (blk <= 24) {
      if (d >= 8) strength = 1;
      if (d >= 16) strength = 2;
      if (d >= 32) strength = 3;
    } else if (blk <= 32) {
      strength = 1;
      if (d >= 4) strength = 2;
      if (d >= 32) strength = 3;
    } else {
      strength = 3;
    }
  } else {
    if (blk <= 8) {
      if (d >= 40) strength = 1;
      if (d >= 64) strength = 2;
    } else if (blk <= 16) {
      if (d >= 20) strength = 1;
      if (d >= 48) strength = 2;
    } else if (blk <= 24) {
      if (d >= 4) strength = 3;
    } else {
      strength = 3;
    }
  }
  return strength;
}

// ---------------------------------------------------------------------------
// Block description
// ---------------------------------------------------------------------------

// One intra prediction block. The predictor reads the *reconstructed* plane at
// x0-1 (left column) and y0-1 (above row), clamped to the visible plane bounds
// [0, max_x] x [0, max_y], exactly like the spec's MAX_X / MAX_Y.
struct IntraBlock {
  const std::uint8_t* recon_plane = nullptr;  // reconstructed plane
  int recon_stride = 0;                       // samples per row
  int x0 = 0;                                 // block origin, plane samples
  int y0 = 0;
  int width = 0;   // 4, 8, 16, 32 or 64
  int height = 0;  // 4, 8, 16, 32 or 64
  int max_x = 0;   // last visible column, inclusive
  int max_y = 0;   // last visible row, inclusive

  int have_left = 0;   // valid samples to the left (block not in column 0)
  int have_above = 0;  // valid samples above (block not in row 0)

  // Valid 4x4 units above-right / below-left of the transform block
  // (num4AboveRight / num4BelowLeft, §7.13.2.1). These bound the extended edge
  // samples used by directional prediction. Caller-supplied; 0 is safe.
  int num4_above_right = 0;
  int num4_below_left = 0;

  // 0 = luma (enableIdif = 1, 4-tap IDIF); > 0 = chroma (bilinear).
  int plane = 0;

  IntraMode mode = kDcPred;
  int angle_delta = 0;  // [-3, 3], only used by directional modes

  // §7.13.2.15/16: 1 if the neighbouring block above / left uses a smooth
  // mode. Both are OR-ed into a single filter type (IBP disabled).
  int filter_type_above = 0;
  int filter_type_left = 0;

  // The normative edge filter is enabled for M1; kept as a switch so tests
  // can isolate it. When 0 the predictor is the unfiltered directional form.
  int enable_edge_filter = 1;

  // Source plane, only used by the residual entry points. Samples are clamped
  // to [0, source_max_x] x [0, source_max_y] (partial edge blocks).
  const std::uint8_t* source_plane = nullptr;
  int source_stride = 0;
  int source_max_x = 0;
  int source_max_y = 0;
};

// Device-resident batch job. `pred` (width*height uint8) is required;
// `residual` (width*height int16) is optional.
struct IntraJob {
  IntraBlock block{};
  std::uint8_t* pred = nullptr;
  std::int16_t* residual = nullptr;
};

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

// Fills pred[0 .. width*height) with the predictor.
cudaError_t intra_predict(const IntraBlock& block, std::uint8_t* pred,
                          cudaStream_t stream = nullptr);

// Predictor + residual = source(clamped) - predictor, one launch.
cudaError_t intra_predict_residual(const IntraBlock& block, std::uint8_t* pred,
                                   std::int16_t* residual,
                                   cudaStream_t stream = nullptr);

// Batch: one CUDA block per job. d_jobs must be device memory. Jobs are
// independent (no job reads what another job in the same launch writes).
// job_count == 0 is a no-op.
cudaError_t intra_predict_batch(const IntraJob* d_jobs, int job_count,
                                cudaStream_t stream = nullptr);

}  // namespace av2enc::cuda::intra_modes

#endif  // AV2ENC_CUDA_INTRA_MODES_INTRA_MODES_H_

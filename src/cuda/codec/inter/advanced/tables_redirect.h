#ifndef AV2ENC_CUDA_INTER_ADVANCED_TABLES_REDIRECT_H_
#define AV2ENC_CUDA_INTER_ADVANCED_TABLES_REDIRECT_H_

// Redirect the shared advanced-inter helpers' table lookups to the
// device-storage copies.  Include this *before* `advanced.h` in every nvcc
// translation unit that compiles the `AV2ENC_ADV_HD` helpers: nvcc compiles
// each `__host__ __device__` body for the device pass even when the host never
// calls it, and a namespace-scope host `constexpr` array is not addressable
// from device code.
//
// The shim keeps host copies with identical values, so the CPU reference in
// the same TU is unchanged.  A host-only static_assert in the shim compares
// every entry with the CPU table.

#include "tables_device.cuh"

#define kBlock16x16 kBlock16x16Dev
#define kBlock64x64 kBlock64x64Dev
#define kBlock8x8 kBlock8x8Dev
#define kBlockSizes kBlockSizesDev
#define kExtWarpedFilterRows kExtWarpedFilterRowsDev
#define kExtWarpedFilterTaps kExtWarpedFilterTapsDev
#define kExtWarpedFilters kExtWarpedFiltersDev
#define kIiWeights1d kIiWeights1dDev
#define kNegDist2BldWeight kNegDist2BldWeightDev
#define kNumWedgeDist kNumWedgeDistDev
#define kPosDist2BldWeight kPosDist2BldWeightDev
#define kWarpedFilterRows kWarpedFilterRowsDev
#define kWarpedFilterTaps kWarpedFilterTapsDev
#define kWarpedFilters kWarpedFiltersDev
#define kWedgeAngleDist2Index kWedgeAngleDist2IndexDev
#define kWedgeAngles kWedgeAnglesDev
#define kWedgeBits kWedgeBitsDev
#define kWedgeBldLutSize kWedgeBldLutSizeDev
#define kWedgeBlockHeight kWedgeBlockHeightDev
#define kWedgeBlockWidth kWedgeBlockWidthDev
#define kWedgeBoundaryTypes kWedgeBoundaryTypesDev
#define kWedgeCodebook kWedgeCodebookDev
#define kWedgeCosLut kWedgeCosLutDev
#define kWedgeSinLut kWedgeSinLutDev
#define kWedgeTypes kWedgeTypesDev

#endif  // AV2ENC_CUDA_INTER_ADVANCED_TABLES_REDIRECT_H_

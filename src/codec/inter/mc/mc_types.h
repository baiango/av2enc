// AV2 §7.13.3 inter motion compensation — shared POD types.
//
// Standalone component for the M2 P-frame baseline (docs/tool-roadmap.md §5):
// translational single-reference MC with the normative 8-tap / 4-tap subpel
// filters, skip/merge candidate selection, and NEWMV/NEARMV motion-vector
// prediction hooks.  8-bit 4:2:0 only.
//
// Motion vectors are expressed in 1/8 luma sample units, matching AV2
// MvPrecision = MV_PRECISION_EIGHTH_PEL for the baseline.  The MC kernel
// converts them to the spec's 1/16-sample filter phase internally.
//
// Reference frame numbering follows the AV2/AV1 convention because the
// integration pass will feed RefFrame[] slots from the frame header:
//   -1 NONE, 0 INTRA_FRAME, 1..7 inter reference slots.
// This component never reads or writes bitstream syntax.

#ifndef AV2ENC_ENCODER_INTER_MC_MC_TYPES_H_
#define AV2ENC_ENCODER_INTER_MC_MC_TYPES_H_

#include <cstdint>

namespace av2enc::inter::mc {

// AV2 §9.2 constants (values from the pinned v1.0.0 spec).
inline constexpr int kMiSize = 4;          // MI_SIZE
inline constexpr int kMvBits = 3;          // 1/8 luma sample
inline constexpr int kSubpelBits = 4;      // SUBPEL_BITS, 1/16 sample
inline constexpr int kSubpelMask = 15;     // SUBPEL_MASK
inline constexpr int kScaleSubpelBits = 10;
inline constexpr int kRefScaleShift = 14;
inline constexpr int kFilterBits = 7;      // FILTER_BITS
inline constexpr int kInterRound0 = 3;
inline constexpr int kInterRound1 = 11;    // single prediction
inline constexpr int kMvBorder = 128;      // MV_BORDER
inline constexpr int kMvLow = -(1 << 16);  // MV_LOW
inline constexpr int kMvUpp = (1 << 16);   // MV_UPP
inline constexpr int kMaxRefMvStackSize = 6;
inline constexpr int kMaxPrNum = 16;
inline constexpr int kMaxBlockDim = 64;    // M2 largest coding block
inline constexpr int kMaxIntermediateRows = 2 * kMaxBlockDim + 8;

// Reference frame slots.  Only the sentinels and "is inter" matter here.
inline constexpr std::int8_t kRefNone = -1;
inline constexpr std::int8_t kRefIntra = 0;
inline constexpr std::int8_t kRefLast = 1;
inline constexpr std::int8_t kRefGolden = 2;
inline constexpr std::int8_t kRefAltRef = 5;

inline constexpr bool is_inter_ref(std::int8_t ref) {
  return ref != kRefNone && ref != kRefIntra;
}

// AV2 InterpFilters (§5.18.5.1).  Four-tap variants are selected by the
// normative width/height rule inside the kernel; the enum values must match
// the Subpel_Filters index.
enum class InterpFilter : std::int32_t {
  kEighttap = 0,
  kEighttapSmooth = 1,
  kEighttapSharp = 2,
  kBilinear = 3,
  kFourtap = 4,        // not signalled; used for w<=4 (regular)
  kFourtapSmooth = 5,  // not signalled; used for w<=4 (smooth)
};

// A motion vector in 1/8 luma sample units.  `y` is the row component,
// `x` the column component (spec order mv[0]/mv[1]).  Components are int32
// because AV2 allows |mv| < 2^16; int16 would truncate large or 8K MVs.
struct Mv {
  std::int32_t y = 0;
  std::int32_t x = 0;
};

inline constexpr bool operator==(Mv a, Mv b) {
  return a.y == b.y && a.x == b.x;
}

// A read-only plane.  `stride` is in samples and must be at least the plane
// width.  The buffer must cover the MI-aligned plane height.
struct PlaneView {
  const std::uint8_t* data = nullptr;
  int stride = 0;
};

// A writable plane (prediction destination).
struct MutablePlane {
  std::uint8_t* data = nullptr;
  int stride = 0;
};

// One reference frame, 8-bit 4:2:0 planar.
//
// `mi_cols`/`mi_rows` are the luma MI dimensions (RefMiCols/RefMiRows).  Planes
// are addressed over the MI-aligned extent: luma mi_cols*4 x mi_rows*4,
// chroma half that.  Reference scaling (a reference whose visible dimensions
// differ from the current frame) is out of M2 scope; every reference passed
// here must have the same dimensions as the frame being encoded.
struct RefFrame {
  PlaneView plane[3];
  int mi_cols = 0;
  int mi_rows = 0;

  int plane_width(int p) const {
    return (p == 0 ? mi_cols * kMiSize : (mi_cols * kMiSize) >> 1);
  }
  int plane_height(int p) const {
    return (p == 0 ? mi_rows * kMiSize : (mi_rows * kMiSize) >> 1);
  }
};

// Luma block rectangle in samples.  For 4:2:0 the chroma rectangle is
// (x>>1, y>>1, w>>1, h>>1); callers must keep x/y/w/h even for the combined
// 4:2:0 entry point.  Partial (edge) blocks simply pass their clipped size.
struct BlockRect {
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
};

// One 4x4 luma MI cell of neighbour mode information, used by the spatial MV
// stack scan (§7.12.2).  The encoder fills this while coding the frame; the
// MC component does not own or allocate it.
//
// `ref`/`mv` are stored per reference list so that the scan can implement
// "RefFrames[mvRow][mvCol][candList] == RefFrame[0]" exactly.  A single-ref
// encoder sets list 1 to kRefNone and is_inter = 1 only for inter blocks.
struct MiInfo {
  std::int8_t ref[2] = {kRefNone, kRefNone};
  std::int32_t mv[2][2] = {{0, 0}, {0, 0}};  // [list][comp], 1/8 pel
  std::int8_t is_inter = 0;
  std::int8_t w4 = 0;             // block extent in MI units
  std::int8_t h4 = 0;
  std::int16_t col_base = 0;      // block origin in MI units
  std::int16_t row_base = 0;
  // §7.22 motion-field-store metadata.  The merge stack never reads these; the
  // temporal field store needs the block-level warp and wedge state that the
  // per-4x4 ref/MV pair cannot express.
  //
  // `warp[list]`: the block is coded GLOBALMV/GLOBAL_GLOBALMV with a
  // non-translation global model, so the decoder's §7.22 store projects the
  // model at every 8x8 centre instead of copying `mv[list]`.
  // `is_wedge`/`wedge_index`/`wedge_sign`/`bsize`: the §7.13.3.27 wedge mask
  // the store samples for its per-8x8 allow-list.  `bsize` is a
  // partition::BlockSize value.
  std::int8_t warp[2] = {0, 0};
  std::int8_t is_wedge = 0;
  std::int8_t wedge_index = 0;
  std::int8_t wedge_sign = 0;
  std::int16_t bsize = 0;
};

}  // namespace av2enc::inter::mc

#endif  // AV2ENC_ENCODER_INTER_MC_MC_TYPES_H_

// §7.20.4 Pixel classified Wiener filter process, using the recovered
// normative tables (pc_wiener_filters.h, pc_wiener_lut_to_class.h,
// pc_wiener_sub_classify.h, pc_wiener_sub_classify2.h).
//
// The tables come from the official attachments; see raw/SHA256SUMS and
// generate_spec_tables.py.  Scalar CPU reference only; not wired into M1/M2.
//
// Scope: luma (the only plane PC-Wiener filters).  BitDepth 8, 10 and 12.

#ifndef AV2ENC_ENCODER_FILTERS_SPEC_TABLES_PC_WIENER_H_
#define AV2ENC_ENCODER_FILTERS_SPEC_TABLES_PC_WIENER_H_

#include <cstdint>

#include "codec/filters/tables/luma_source.h"

namespace av2enc::filters::spec {

// §7.20.4 constants printed in the specification text.
constexpr int kPcWienerTaps = 25;
constexpr int kPcWienerPrecBits = 7;
constexpr int kPcWienerPrecFeature = 14;
constexpr int kPcWienerNumFeatures = 4;
constexpr int kPcWienerLead = 1;
constexpr int kPcWienerLag = 4;

// Pc_Wiener_Config[25][2] = {dy, dx}.  Note the spec's declaration order:
// the first column is dy, the second is dx.
struct PcWienerTap {
  std::int8_t dy;
  std::int8_t dx;
};

inline constexpr PcWienerTap kPcWienerConfig[kPcWienerTaps] = {
    {1, 0},   {-1, 0},  {0, 1},   {0, -1}, {2, 0},   {-2, 0},  {0, 2},
    {0, -2},  {1, 1},   {-1, -1}, {-1, 1}, {1, -1},  {2, 1},   {-2, -1},
    {2, -1},  {-2, 1},  {1, 2},   {-1, -2}, {1, -2}, {-1, 2},  {3, 0},
    {-3, 0},  {0, 3},   {0, -3},  {0, 0},
};

inline constexpr int kPcWienerNormalizer[kPcWienerNumFeatures + 1] = {0, 3739, 3273,
                                                                      3074, 7};

inline constexpr int kPcWienerModeWeights[kPcWienerNumFeatures][3] = {
    {-527, 15325, 321}, {26436, -17705, 17905}, {366, -147, -194}, {202, -267, -179},
};

inline constexpr int kPcWienerModeOffsets[kPcWienerNumFeatures] = {-547, -21565, -573,
                                                                   -680};

// §7.20.4 get_filter_set_index().
constexpr int PcWienerFilterSetIndex(int base_q_idx) {
  if (base_q_idx < 130) return 0;
  if (base_q_idx < 190) return 1;
  if (base_q_idx < 220) return 2;
  return 3;
}

// §7.20.4 per-block inputs.
struct PcWienerConfig {
  int base_q_idx = 0;
  int bit_depth = 8;
  int mi_col_end = 0;   // §7.20.1 MiColEnd for the current block
  int mi_row_start = 0;  // §7.20.1 MiRowStart (tile clamp for LrTxSkip)
  int mi_row_end = 0;    // §7.20.1 MiRowEnd
  const std::uint8_t* lr_tx_skip = nullptr;  // required; luma MI grid
  int lr_tx_skip_stride = 0;
};

// LrFrame plus the §7.20.1 LumaSource, and the §7.20.4 FilterClass grid.
struct PcWienerFrame {
  LumaSource src;
  std::uint16_t* lr = nullptr;
  int lr_stride = 0;
  std::uint8_t* filter_class = nullptr;
  int filter_class_stride = 0;
};

// §7.20.4 class derivation (the part before `if (skipFilter)`).
int PcWienerClass(const PcWienerFrame& frame, const PcWienerConfig& cfg, int x,
                  int y);

// §7.20.4 with skipFilter = 1: writes FilterClass[y>>2][x>>2] only.
void PcWienerClassifyBlock(const PcWienerFrame& frame, const PcWienerConfig& cfg,
                           int x, int y);

// §7.20.4 with skipFilter = 0: filters `w` x `h` luma samples into LrFrame.
//
// The specification invokes this per restore block (in practice 4x4 luma
// samples), and derives one class from the block's (x, y).  Passing a larger
// w/h reuses that one class for the whole rectangle, which only matches the
// decoder when the class is constant across it.
void PcWienerFilterBlock(const PcWienerFrame& frame, const PcWienerConfig& cfg,
                         int x, int y, int w, int h);

// Convenience wrapper matching the spec's single `skipFilter` argument.
void PcWienerProcess(const PcWienerFrame& frame, const PcWienerConfig& cfg, int x,
                     int y, int w, int h, bool skip_filter);

}  // namespace av2enc::filters::spec

#endif  // AV2ENC_ENCODER_FILTERS_SPEC_TABLES_PC_WIENER_H_

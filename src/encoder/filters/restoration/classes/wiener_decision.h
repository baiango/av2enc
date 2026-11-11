// Encoder-side Wiener loop-restoration decisions.
//
// Produces everything the frame header and tile syntax need:
//
//   * FrameRestorationType[plane] (NONE or WIENER_NONSEP),
//   * per-unit vs frame-level filter selection (frame_filters_on),
//   * use_wiener_ns / LrType[plane][unitRow][unitCol],
//   * per-unit coefficients (with LR_BANK_SIZE merging),
//   * frame-level coefficients and class merging,
//   * LoopRestorationSize per plane,
//   * the exact syntax plans consumed by wiener_syntax.cpp.
//
// The RD comparison is SSE + lambda_bits * bits.  SSE is exact integer-filter
// SSE against the encoder's source picture; the bit counts are produced by the
// real syntax writers (so CDF state and bank state are accounted for).  With
// lambda_bits == 0 the search is pure distortion, which is what the tests use;
// a production encoder should set a nonzero lambda.
//
// Filter classes: only NumFilterClasses == 1 is available from the spec text.
// Multi-class frame filters need Pc_Wiener_Lut_To_Class and
// Pc_Wiener_Sub_Classify2, which §9.8 delegates to attachment headers that are
// not part of the specification.  A caller that owns those tables passes them
// in; otherwise multi-class candidates are skipped with a note and the search
// stays conformant at one class.

#ifndef AV2ENC_ENCODER_FILTERS_RESTORATION_CLASSES_WIENER_DECISION_H_
#define AV2ENC_ENCODER_FILTERS_RESTORATION_CLASSES_WIENER_DECISION_H_

#include <cstdint>
#include <string>
#include <vector>

#include "encoder/filters/restoration/classes/wiener_class.h"
#include "encoder/filters/restoration/classes/wiener_fit.h"
#include "encoder/filters/restoration/classes/wiener_syntax.h"
#include "encoder/filters/restoration/restoration.h"

namespace av2enc::restoration::classes {

enum class WienerDecisionStatus {
  kOk = 0,
  kInvalidArgument,
  kMissingNormativeTable,
  kNoCandidates,
  kSyntaxError,
};

const char* wiener_decision_status_string(WienerDecisionStatus status);

struct WienerDecisionInput {
  // Geometry + sample planes.  The decision module overwrites frame_type,
  // unit_size, unit/frame grids on its private copies; curr/cdef/dst pointers
  // and mi_cols/mi_rows/subsampling are taken from here.
  LrCoreFrame core;
  const std::uint8_t* target[3] = {nullptr, nullptr, nullptr};
  int target_stride[3] = {0, 0, 0};
  int base_q_idx = 0;
  // Optional normative PC-Wiener tables (see wiener_class.h).  Needed only
  // when max_num_filter_classes > 1.
  const PcWienerTables* tables = nullptr;
  // LrTxSkip[MI] for §7.20.4 classification (only for multi-class).
  const std::uint8_t* lr_tx_skip = nullptr;
  // Optional pre-computed FilterClass grid (256-wide, luma MI).  If null it is
  // derived when the tables are available.
  const std::uint8_t* filter_class_grid = nullptr;
  const WienerReferenceSet* refs = nullptr;
};

struct WienerDecisionOptions {
  bool allow_luma = true;
  bool allow_chroma = true;
  bool allow_per_unit = true;
  bool allow_frame_filters = true;
  // 1 keeps the search spec-text-only.  Values > 1 are attempted only when the
  // caller supplies the normative class tables.
  int max_num_filter_classes = 1;
  // RD weight in SSE per bit.  0 = pure distortion.
  double lambda_bits = 0.0;
  // Restrict the unit-size search when >= 0.
  int forced_luma_shift = -1;
  int forced_chroma_shift = -1;
};

struct WienerUnitPlan {
  int type = kRestoreNone;
  bool merged = false;
  int use_bank_k = -1;
  int subset = 0;
  bool uv_sym = false;
  std::int16_t coeffs[kWienerNsCoeffStride] = {};
  double sse = 0.0;
  int bits = 0;
};

struct WienerPlaneDecision {
  int mode = 0;  // 0 = none, 1 = per-unit, 2 = frame filters
  int frame_type = kRestoreNone;
  bool frame_filters_on = false;
  int num_filter_classes = 1;
  int num_filter_classes_idx = 0;
  int shift = 3;
  int unit_size = 64;
  int unit_rows = 0;
  int unit_cols = 0;

  std::vector<std::uint8_t> unit_type;
  std::vector<std::int16_t> unit_coeffs;
  std::vector<WienerUnitPlan> units;  // raster order, same length as units

  std::int16_t frame_coeffs[kWienerNsClasses][kWienerNsCoeffStride] = {};
  WienerFrameFilterSyntax frame_filter[kWienerNsClasses];

  // Multi-class luma maps.  Empty when num_filter_classes == 1.
  std::vector<std::uint8_t> filter_class_grid;  // 256-wide FilterClass
  std::vector<std::uint8_t> subclass_grid;      // looked-up subclasses
  std::vector<std::uint8_t> subclass_lookup;    // 256 entries
  int filter_class_count = 0;

  double sse = 0.0;
  int header_bits = 0;
  int tile_bits = 0;
  double cost = 0.0;
};

struct WienerDecision {
  WienerLrHeaderParams params;
  int unit_size[3] = {64, 64, 64};
  int unit_rows[3] = {0, 0, 0};
  int unit_cols[3] = {0, 0, 0};
  std::vector<std::uint8_t> unit_type[3];
  std::vector<std::int16_t> unit_coeffs[3];

  WienerPlaneDecision plane[3];
  // Luma frame-level class maps, ready for LoopRestorationParams.
  std::vector<std::uint8_t> filter_class_grid;
  std::vector<std::uint8_t> subclass_grid;
  std::vector<std::uint8_t> subclass_lookup;
  int filter_class_count = kWienerNsClasses;
  int num_filter_classes = 1;

  double sse = 0.0;
  int header_bits = 0;
  int tile_bits = 0;
  double cost = 0.0;
  std::vector<std::string> notes;
};

WienerDecisionStatus wiener_decide(const WienerDecisionInput& input,
                                   const WienerDecisionOptions& options,
                                   WienerDecision* decision);

// Fills a LoopRestorationParams view.  All pointers refer to storage owned by
// `decision`, which must outlive the call to loop_restoration_frame().
void wiener_decision_to_loop_params(const WienerDecision& decision,
                                    LoopRestorationParams* params);

}  // namespace av2enc::restoration::classes

#endif  // AV2ENC_ENCODER_FILTERS_RESTORATION_CLASSES_WIENER_DECISION_H_

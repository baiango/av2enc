// AV2 quantizer matrices (QM) — CPU reference.
//
// Normative sources in spec/AV2_Spec_v1.0.0.pdf:
//   * §9.4.3     Qm_Offset and Quantizer_Matrix (generated qm_tables.h)
//   * §5.4.11    user_defined_qm()
//   * §5.13      quantizer_matrix_obu()
//   * §5.18.6.2  setup_qm_params()
//   * §7.14.4    dequantization with q2 = Round2(q * m, 5)
//
// Scope: lossy 8-bit 4:2:0 encoder path.  The §7.14.4 dequant clip takes an
// explicit BitDepth (8..12; tested at 8 and 10).  Levels 0..14 index the
// normative matrices; level 15 is the §7.14.4 sentinel that switches QM off
// (lossless segments set SegQMLevel = 15).  User QM data is defined for the
// three fundamental matrices (8x8, 8x4, 4x8) only, matching §5.4.11.
//
// The CUDA half lives in src/cuda/qm/.  Both halves call the same
// `qm_core.h` integer helpers, so the arithmetic is bit-exact.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/bitstream/bitwriter.h"
#include "codec/transform/qm/qm_core.h"
#include "codec/transform/qm/qm_constants.h"

namespace av2enc::qm {

inline constexpr int kMaxQmEntries = 4;  // pic_qm_num_minus_1 is f(2)
inline constexpr int kMaxSegments = 8;   // M1 uses one segment; syntax allows 8
inline constexpr int kUserQmT = 3;       // t = 0 (8x8), 1 (8x4), 2 (4x8)
inline constexpr int kUserQmPlanes = 3;

// UserQm[level][t][plane][i][j] from §5.4.11.  Values are 1..255; 0 is
// forbidden by the conformance requirement.  Presence is tracked by
// QmFrameState::qm_data_present (QmDataPresent).  The struct is exactly 576
// bytes so the [level] stride matches the flattened CUDA layout.
struct UserQmLevel {
  std::uint8_t m[kUserQmT][kUserQmPlanes][8][8] = {};
};
static_assert(sizeof(UserQmLevel) == 576,
              "UserQmLevel must stay 576 bytes for the CUDA [level] stride");

// Frame/sequence state that feeds both setup_qm_params() and dequantization.
// `seg_qm_level` is §5.18.2's SegQMLevel[plane][segment_id]; initialize it to
// kNoQmLevel and set the encoder's chosen level for each segment.
struct QmFrameState {
  bool using_qmatrix = false;
  int pic_qm_num_minus_1 = 0;
  std::uint8_t qm_y[kMaxQmEntries] = {};
  std::uint8_t qm_u[kMaxQmEntries] = {};
  std::uint8_t qm_v[kMaxQmEntries] = {};
  bool separate_uv_delta_q = false;
  bool qm_data_present[kNumQmLevels] = {};
  UserQmLevel user_qm[kNumQmLevels];
  std::uint8_t seg_qm_level[3][kMaxSegments];
  bool lossless_segment[kMaxSegments] = {};

  QmFrameState();
};

// One block's §7.14.4 inputs.  >8x8 blocks take the level from
// state.qm_y/u/v[0]; <=8x8 blocks take SegQMLevel[plane][segment_id].
struct QmDequantParams {
  const QmFrameState* state = nullptr;
  int tx_size = 0;
  int plane = 0;
  int tx_type = 0;
  int segment_id = 0;
  int dc_quant = 0;
  int ac_quant = 0;
  int bit_depth = 8;  // BitDepth for the §7.14.4 Clip3 bound
  bool allow_tcq = false;
  bool lossless = false;
  bool use_fsc = false;
};

// --- Table access -----------------------------------------------------------

bool valid_tx_size(int tx_size);
bool valid_level(int level);
inline int tx_width(int tx_size) { return core::coded_width(tx_size); }
inline int tx_height(int tx_size) { return core::coded_height(tx_size); }
inline int adjusted_width(int tx_size) {
  return core::adjusted_width(tx_size);
}
inline int adjusted_height(int tx_size) {
  return core::adjusted_height(tx_size);
}
std::uint16_t qm_offset(int tx_size);

// Default Quantizer_Matrix entry for an adjusted position.  `plane` 0 is
// luma; any non-zero plane is chroma (the spec has two plane types).
std::uint8_t default_matrix_value(int level, int plane, int tx_size, int i,
                                  int j);

// §7.14.4 segLvl selection: >8x8 uses the frame-level qm_*[0] entries,
// otherwise SegQMLevel[plane][segment_id].
int resolve_seg_level(const QmFrameState& state, int plane, int tx_size,
                      int segment_id);
bool uses_user_qm(const QmFrameState& state, int plane, int tx_size,
                  int seg_level);

// The m coefficient §7.14.4 would select (user matrix or default table).
int matrix_value(const QmFrameState& state, int plane, int tx_size,
                 int seg_level, int i, int j);

// --- Dequantization ---------------------------------------------------------

// Fills `q2` with adjusted_width * adjusted_height entries.
bool build_q2_matrix(const QmDequantParams& params, std::int32_t* q2);

// Dequantizes levels[tw*th] (adjusted-size raster order) into dequant[tw*th].
// Returns false on invalid params (bad tx size/plane/state, null buffers).
bool dequantize_block(const std::int32_t* levels,
                      const QmDequantParams& params, std::int32_t* dequant);

// --- Syntax writing ---------------------------------------------------------

// setup_qm_params() (§5.18.6.2) without the surrounding frame-header bits.
// `num_planes` is NumPlanes from the sequence header (3 for 4:2:0).
// `segmentation_enabled` controls whether pic_qm_num_minus_1 is coded.
bool write_setup_qm_params(BitWriter& writer, const QmFrameState& state,
                           int num_planes, bool segmentation_enabled,
                           std::string* error);

// Fills a UserQmLevel with one constant coefficient (default 32 == identity
// because Round2(q * 32, 5) == q).  Returns false for value 0.
bool fill_user_qm_constant(UserQmLevel* level, std::uint8_t value);

// Builds the payload of quantizer_matrix_obu() (§5.13), including trailing
// bits, without the OBU header.  `bit_map` selects levels; level i uses
// cfg.levels[i].  Returns false when a level is malformed.
struct QmObuLevel {
  bool is_default = true;
  UserQmLevel user;
};
struct QmObuConfig {
  std::uint16_t bit_map = 0;  // f(15); bit i selects level i
  bool chroma_info_present = true;
  QmObuLevel levels[kNumQmLevels];
};
bool write_quantizer_matrix_obu_payload(BitWriter& writer,
                                        const QmObuConfig& config,
                                        std::string* error);

// Complete Annex B.2 OBU (leb128 length + header + payload).
std::vector<std::uint8_t> make_quantizer_matrix_obu(
    const QmObuConfig& config, std::string* error);

// OBU type value for OBU_QUANTIZATION_MATRIX (§6.2.2 Table 6.1).
inline constexpr std::uint8_t kObuTypeQuantizationMatrix = 22;

}  // namespace av2enc::qm

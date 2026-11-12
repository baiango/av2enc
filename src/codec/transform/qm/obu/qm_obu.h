// AV2 v1.0.0 quantizer-matrix OBU state machine, writer and reader.
//
// Normative reference: spec/AV2_Spec_v1.0.0.pdf
//
//   §5.2.1  open_bitstream_unit()      OBU framing (leb128 + obu_header)
//   §5.2.2  obu_header()               obu_type / tlayer / mlayer / xlayer
//   §5.4.11 user_defined_qm()          per-level matrix payload
//   §5.13   quantizer_matrix_obu()     qm_bit_map / qm_is_default_flag
//   §5.18.6.2 setup_qm_params()        frame-header QM fields
//   §6.4.11 User defined QM semantics  quant_delta range, non-zero entries
//   §6.12   Quantizer Matrix OBU semantics
//   §6.5    temporal_delimiter_obu()   resets QmProtected
//
// Scope: this module is the missing *encoder-side* mirror of the decoder state
// variables QmSeen / QmProtected / QmDataPresent / QmMLayerId / QmTLayerId.
// It owns:
//
//   * QmState: one entry per NUM_CUSTOM_QMS level plus the multi-OBU
//     conformance rules ("only the first QM OBU between coded frames may have
//     qm_bit_map == 0"; "a level may not be specified twice in that window").
//   * §5.13 payload writer/reader with the §5.4.11 user-defined matrices.
//   * §5.18.6.2 frame-header writer/reader for using_qmatrix,
//     pic_qm_num_minus_1, qm_y[], qm_uv_same_as_y and qm_u/v[].
//   * Annex B.2 OBU framing + a parser for multi-OBU byte sequences.
//
// It deliberately does not edit or replace src/encoder/qm/qm.{h,cpp}; see
// README.md for the exact merge into qm and m1/video_headers.  The types here are
// plain PODs and the code has no CUDA dependency, so the state machine can be
// used by the CPU reference path and by the header writers.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "core/bitstream/bitwriter.h"
#include "codec/transform/qm/qm.h"  // UserQmLevel, kNumQmLevels

namespace av2enc::qm::obu {

// NUM_CUSTOM_QMS (§6.2.1).  qm_bit_map is f(15), so only levels 0..14 can be
// carried by an OBU.  Level 15 is the §7.14.4 "QM off" sentinel and is valid in
// the frame header's f(4) qm_y/qm_u/qm_v fields.
inline constexpr int kNumCustomQms = kNumQmLevels;  // 15
// pic_qm_num_minus_1 is f(2).
inline constexpr int kMaxFrameQmEntries = 4;
// OBU_QUANTIZATION_MATRIX (§6.2.2 Table 6.1).
inline constexpr std::uint8_t kObuTypeQuantizationMatrix = 22;
inline constexpr std::uint8_t kObuTypeTemporalDelimiter = 2;
inline constexpr std::uint8_t kObuTypeClosedLoopKey = 4;
inline constexpr std::uint8_t kObuTypeOpenLoopKey = 5;
inline constexpr std::uint8_t kObuTypeSwitch = 10;
inline constexpr std::uint8_t kObuTypeRasFrame = 21;
// A non-extension OBU_TEMPORAL_DELIMITER / OBU_MSDO infers GLOBAL_XLAYER_ID.
inline constexpr std::uint8_t kGlobalXLayerId = 31;

// §5.2.2 obu_header() fields.  `extension_flag` is derived: it is present iff
// mlayer_id != 0 or xlayer_id != 0 (for non-global OBU types).
struct ObuHeader {
  std::uint8_t type = kObuTypeQuantizationMatrix;
  std::uint8_t tlayer_id = 0;  // f(2)
  std::uint8_t mlayer_id = 0;  // f(3), extension byte
  std::uint8_t xlayer_id = 0;  // f(5), extension byte
};

// One OBU from an Annex B.2 byte sequence.
struct ParsedObu {
  ObuHeader header;
  std::vector<std::uint8_t> payload;  // bytes after the OBU header
  std::vector<std::uint8_t> raw;      // leb128 prefix + header + payload
};

bool write_obu_header(const ObuHeader& header, std::vector<std::uint8_t>* out,
                      std::string* error);
bool read_obu_header(std::span<const std::uint8_t> bytes, ObuHeader* header,
                     std::size_t* consumed, std::string* error);

// leb128(obu size) | obu_header() | payload  (§5.2.1, Annex B.2).
bool frame_obu(const ObuHeader& header, std::span<const std::uint8_t> payload,
               std::vector<std::uint8_t>* out, std::string* error);

// Parses a whole length-delimited OBU sequence.  Fails on a truncated leb128,
// a zero length, an OBU that overruns the input, or a truncated extension byte.
bool parse_obu_sequence(std::span<const std::uint8_t> bytes,
                        std::vector<ParsedObu>* obus, std::string* error);

// ---------------------------------------------------------------------------
// §5.13 quantizer_matrix_obu() payload
// ---------------------------------------------------------------------------

struct QmLevelData {
  // qm_is_default_flag.  true means "use the normative default matrix"; no
  // user_defined_qm() data follows and QmDataPresent[level] becomes 0.
  bool is_default = true;
  UserQmLevel user;  // valid only when is_default == false
};

struct QuantizerMatrixObu {
  std::uint16_t bit_map = 0;          // f(15)
  bool chroma_info_present = true;    // qm_chroma_info_present_flag
  std::array<QmLevelData, kNumCustomQms> levels{};
};

// Writes the payload only (no obu_header, no leb128, no trailing byte
// padding beyond the §5.2.3 trailing bits).  Rejects bit 15 of bit_map,
// qm_is_default_flag = 0 levels whose matrix contains a zero coefficient
// (§6.4.11), and deltas outside [-128, 127].  The caller supplies a fresh
// BitWriter; the function appends and byte-aligns.
bool write_quantizer_matrix_payload(BitWriter& writer,
                                    const QuantizerMatrixObu& obu,
                                    std::string* error);

// Parses one complete payload (exactly the bytes after the OBU header).
// Enforces the trailing-bit syntax, the svlc() range, and the §6.4.11
// "no UserQm entry is zero" requirement.
bool read_quantizer_matrix_payload(std::span<const std::uint8_t> payload,
                                   QuantizerMatrixObu* obu,
                                   std::string* error);

// Complete Annex B.2 OBU for the given layer ids.
bool make_quantizer_matrix_obu(const QuantizerMatrixObu& obu,
                               const ObuHeader* header,
                               std::vector<std::uint8_t>* out,
                               std::string* error);

// Convenience: one level, user-defined matrix.  Fills `out` with the complete
// OBU.  `value` must be non-zero (a constant matrix, 32 == identity).
bool make_constant_user_qm_obu(int level, std::uint8_t value,
                               bool chroma_info_present,
                               std::vector<std::uint8_t>* out,
                               std::string* error);

// ---------------------------------------------------------------------------
// §5.18.6.2 setup_qm_params() frame-header fields
// ---------------------------------------------------------------------------

struct FrameQm {
  bool using_qmatrix = false;
  int pic_qm_num_minus_1 = 0;
  std::array<std::uint8_t, kMaxFrameQmEntries> qm_y{};
  std::array<std::uint8_t, kMaxFrameQmEntries> qm_u{};
  std::array<std::uint8_t, kMaxFrameQmEntries> qm_v{};
  // qm_uv_same_as_y[i].  When true, qm_u[i] and qm_v[i] are inferred equal to
  // qm_y[i].  Kept explicit so the writer can reproduce a stream bit-exactly.
  std::array<bool, kMaxFrameQmEntries> qm_uv_same_as_y{};
  bool separate_uv_delta_q = false;
};

// Writes using_qmatrix and, when set, the pic_qm_num_minus_1 / qm_* fields.
// `num_planes` is NumPlanes from the sequence header (1..3).
// `segmentation_enabled` controls whether pic_qm_num_minus_1 is coded.
bool write_frame_qm_params(BitWriter& writer, const FrameQm& qm,
                           int num_planes, bool segmentation_enabled,
                           std::string* error);

// Reads the same fields from `bytes` starting at bit 0.  `separate_uv_delta_q`
// comes from the surrounding quantizer_params() syntax, which precedes
// setup_qm_params().  `bits_consumed` is optional and receives the exact
// syntax length.  Levels are validated as f(4) values (0..15).
bool read_frame_qm_params(std::span<const std::uint8_t> bytes, int num_planes,
                          bool segmentation_enabled, bool separate_uv_delta_q,
                          FrameQm* qm, std::size_t* bits_consumed,
                          std::string* error);

// Bits that write_frame_qm_params() emits for `qm`; useful for sizing a splice.
std::size_t frame_qm_params_bits(const FrameQm& qm, int num_planes,
                                 bool segmentation_enabled);

// ---------------------------------------------------------------------------
// §6.12 decoder state variables (encoder-side mirror)
// ---------------------------------------------------------------------------

struct QmLevelState {
  bool qm_seen = false;       // QmSeen[level]
  bool qm_protected = false;  // QmProtected[level]
  bool qm_data_present = false;  // QmDataPresent[level]
  int qm_num_planes = 1;      // QmNumPlanes[level]
  int qm_mlayer_id = -1;      // QmMLayerId[level]
  int qm_tlayer_id = -1;      // QmTLayerId[level]
  bool has_user_qm = false;   // user_defined_qm() data was received
  UserQmLevel user;           // last received matrices for this level
};

class QmState {
 public:
  QmState();

  // Start of bitstream: every QmProtected/QmSeen/QmDataPresent entry is zero
  // and both layer ids are -1 (§6.12).
  void reset();

  // temporal_delimiter_obu() (§5.5): QmProtected[level] = 0 for every level.
  // QmSeen and QmDataPresent are not touched.  The multi-OBU window counter is
  // also not touched: a temporal delimiter is not a coded frame boundary.
  void on_temporal_delimiter();

  // Start of frame_header_info() (§5.18.2): QmSeen[level] = 0 for every level
  // and the "between coded frames" QM-OBU window counter restarts.
  void on_frame_header();

  // reset_qm() from §5.18.2 / §5.18.3.  Only unprotected levels are reset.
  // A protected level keeps its data across a key frame.
  //
  // Single embedded layer: the SWITCH/RAS branch of reset_qm() (needsReset is
  // derived from QmMLayerId and MLayerPresenceMap) is not modelled.  Callers
  // that emit OBU_SWITCH/OBU_RAS_FRAME must not use on_key_frame() for it.
  void reset_qm(int num_planes);

  // Convenience for a key frame: QmSeen reset plus reset_qm() when this is the
  // first picture in the temporal unit (the §5.18.2 condition).
  void on_key_frame(bool first_picture_in_tu, int num_planes);

  // Applies one parsed QM OBU.  Rejects:
  //   * bit_map == 0 when another QM OBU already appeared in this window;
  //   * a level already set in QmSeen in this window;
  //   * bit 15 of bit_map, or a type other than OBU_QUANTIZATION_MATRIX.
  // On success updates QmSeen/QmProtected/QmNumPlanes/QmMLayerId/QmTLayerId/
  // QmDataPresent exactly as §5.13 specifies.
  bool apply_obu(const ObuHeader& header, const QuantizerMatrixObu& obu,
                 std::string* error);

  // Parses one Annex B QM OBU (header + payload) and applies it.
  bool apply_obu_bytes(std::span<const std::uint8_t> bytes,
                       std::string* error);

  const QmLevelState& level(int level) const;
  bool data_present(int level) const;
  bool seen(int level) const;
  bool protected_level(int level) const;
  const UserQmLevel* user_qm(int level) const;
  std::uint16_t bit_map() const;  // union of levels seen in this window
  int num_obus_in_window() const { return obus_in_window_; }

 private:
  std::array<QmLevelState, kNumCustomQms> levels_{};
  int obus_in_window_ = 0;
};

}  // namespace av2enc::qm::obu

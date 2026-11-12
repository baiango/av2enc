// AV2 intra bi-prediction (IBP) — CPU reference implementation.
//
// Normative source: AV2 Bitstream & Decoding Process Specification v1.0.0
// (commit 38f28e7), §7.13.2 Intra prediction process.
//
// Implemented here:
//   §7.13.2.7   IBP-specific reference-sample filtering implications
//               (per-edge filter types, extra right/bottom extent, corner rule)
//   §7.13.2.8   single directional prediction, called twice
//   §7.13.2.9   IBP weights process (ibp_weights)
//   §7.13.2.12  IBP DC process (Ibp_Weights)
//   §7.13.2.14  filter corner
//   §7.13.2.17  intra edge filter strength
//   §7.13.2.18  intra edge filter
//   §5.4.5      sequence_intra_config enable_ibp syntax (one f(1) bit)
//   §7.13.3.29  Ii_Weights_1d inter-intra mask (provided for completeness;
//               this is *not* used by IBP)
//
// IBP is not a separately signalled prediction mode.  The decoder derives it
// from the sequence-level enable_ibp flag and the already-signalled intra mode
// and angle_delta:
//
//   applyIbp = enable_ibp && !(w == 4 && h == 4)
//   useIBP   = applyIbp && (angleDelta is even) && plane == 0 &&
//              (pAngle < 90 || pAngle > 180) && MrlIndex == 0
//
// There is no per-block IBP flag in the bitstream.  When enable_ibp is 1 and
// useIBP is true for the chosen (mode, angle_delta), the decoder *must* apply
// the blend; an encoder therefore cannot skip it for that candidate.  The
// encoder-side decision API below enumerates the candidates that trigger it.
//
// Scope / limitations:
//   * 8-bit samples, transform extents 4..64 per axis, power of two.
//   * MrlIndex is fixed at 0 (the module accepts only MrlIndex == 0, matching
//     the existing intra reference component).  MRL is out of scope.
//   * DIP is out of scope; `use_dip` is only used for the DC applicability
//     predicate.
//   * The caller supplies prepared §7.13.2.1 edges (`intra::prepare_edges`).
//
// All arithmetic is integer.  The CUDA device implementation in
// src/cuda/ibp/ must reproduce these bytes exactly.

#ifndef AV2ENC_ENCODER_IBP_IBP_H_
#define AV2ENC_ENCODER_IBP_IBP_H_

#include "codec/intra/modes/intra_modes.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace av2enc::ibp {

using intra::IntraEdges;
using intra::IntraStatus;
using intra::Mode;
using intra::NeighborRequest;

// ---------------------------------------------------------------------------
// Normative constants (§3)
// ---------------------------------------------------------------------------

inline constexpr int kWeightSize = 16;      // IBP_WEIGHT_SIZE
inline constexpr int kWeightSizeLog2 = 4;   // IBP_WEIGHT_SIZE_LOG2
inline constexpr int kWeightMax = 128;      // IBP_WEIGHT_MAX
inline constexpr int kWeightShift = 7;      // IBP_WEIGHT_SHIFT
inline constexpr int kNumIbpWeightRows = 5; // Ibp_Weights[5][16]
inline constexpr int kAngleStep = 3;        // ANGLE_STEP
inline constexpr int kIiWeights1dSize = 128;  // Ii_Weights_1d[128]

// ---------------------------------------------------------------------------
// §7.13.2.9 IBP weights process
// ---------------------------------------------------------------------------

// weights[r][c], 16x16, each in 0..IBP_WEIGHT_MAX.
struct WeightMatrix {
  std::array<int, kWeightSize * kWeightSize> values{};

  int& at(int r, int c) noexcept {
    return values[static_cast<std::size_t>(r * kWeightSize + c)];
  }
  int at(int r, int c) const noexcept {
    return values[static_cast<std::size_t>(r * kWeightSize + c)];
  }
};

// Computes the §7.13.2.9 blending weights for one directional angle.
// `p_angle` is the spec's pAngle; the process applies Max(39, pAngle) itself,
// and defensively clamps the upper end to 89 (the reachable range).  For
// pAngle > 180 the caller passes 270 - pAngle, as §7.13.2.7 does.
WeightMatrix ibp_weights(int p_angle) noexcept;

// §7.13.2.12 Ibp_Weights[log2_extent - 2][index].  Returns 0 for out-of-range
// rows/indices so callers never read past the normative table.
int ibp_dc_weight(int log2_extent, int index) noexcept;

// ---------------------------------------------------------------------------
// §7.13.3.29 inter-intra mask (NOT IBP)
// ---------------------------------------------------------------------------

enum class InterIntraMode : int {
  kV = 0,       // II_V_PRED
  kH = 1,       // II_H_PRED
  kSmooth = 2,  // II_SMOOTH_PRED
  kDc = 3,      // II_DC_PRED
};

int ii_weights_1d(int index) noexcept;

// Fills a w*h row-major mask (stride w) exactly as §7.13.3.29.
// Returns false if w/h are outside 4..64 or the span is too small.
bool build_inter_intra_mask(int w, int h, InterIntraMode mode,
                            std::span<int> mask) noexcept;

// ---------------------------------------------------------------------------
// Decision API
// ---------------------------------------------------------------------------

struct DecisionParams {
  bool enable_ibp = false;
  bool is_inter = false;  // inter-intra prediction: §7.13.2.7 skips WAIP
  Mode mode = Mode::kDc;
  int angle_delta = 0;  // -3..3, directional modes only
  int w = 0;
  int h = 0;
  int plane = 0;        // 0 = luma
  int mrl_index = 0;    // must be 0 for IBP
};

struct Decision {
  bool valid = false;
  // applyIbp in §7.13.2.7: enable_ibp && not4x4.  Governs the modified edge
  // filtering even when use_ibp is false (e.g. 135-degree predictions).
  bool apply_ibp_edge_rules = false;
  // useIBP: two directional predictions are blended.
  bool use_ibp = false;
  int p_angle = 0;       // after angle_delta and wide-angle mapping
  int second_angle = 0;  // valid only when use_ibp
};

Decision decide(const DecisionParams& params) noexcept;

// §7.13.2.7 conditions for the IBP DC process (§7.13.2.12).
bool ibp_dc_applicable(bool enable_ibp, bool use_dip, Mode mode, int w, int h,
                       int plane, bool uv_is_cfl) noexcept;

// A candidate is a (mode, angle_delta) pair.  For an IBP-enabled sequence,
// every returned pair is guaranteed to derive useIBP = true on the decoder.
struct ModeCandidate {
  Mode mode = Mode::kDc;
  std::int8_t angle_delta = 0;
};

// Enumerates the luma directional pairs that trigger IBP for a w x h block:
// eight base directional modes x angle deltas {-2, 0, +2}, filtered through
// wide-angle mapping and the useIBP predicate.  Returns the number written, or
// -1 if `out` is too small.  `plane != 0` or `mrl_index != 0` yield 0.
int build_ibp_candidates(std::span<ModeCandidate> out, int w, int h,
                         int plane = 0, int mrl_index = 0) noexcept;

// ---------------------------------------------------------------------------
// Prediction API
// ---------------------------------------------------------------------------

struct PredictParams {
  Mode mode = Mode::kDc;
  int angle_delta = 0;
  bool enable_ibp = false;
  bool enable_edge_filter = true;
  // Inter-intra prediction: §7.13.2.7 skips wide-angle mapping.  Intra
  // prediction leaves this false.
  bool is_inter = false;
  // §7.13.2.15/.16 neighbour smooth-mode flags.
  bool above_neighbor_is_smooth = false;
  bool left_neighbor_is_smooth = false;
  // Only consulted by ibp_dc_applicable(); DIP itself is out of scope.
  bool use_dip = false;
  bool uv_is_cfl = false;
};

struct PredictResult {
  bool blended = false;
  bool dc_modified = false;
  int p_angle = 0;
  int second_angle = 0;
};

// Predicts one block for any intra mode, applying every IBP side effect that
// the spec attaches to that mode:
//   * directional modes: IBP-modified edge filtering, and the two-angle blend
//     when useIBP is true;
//   * DC: the §7.13.2.12 post-filter when ibp_dc_applicable() is true;
//   * smooth/Paeth: unchanged (no IBP effect), delegated to the intra module.
//
// `edges.is_luma` selects IDIF (luma) versus bilinear (chroma) interpolation.
// The edges are never modified.
IntraStatus predict(const IntraEdges& edges, const PredictParams& params,
                    std::uint8_t* dst, std::size_t dst_stride,
                    PredictResult* result = nullptr) noexcept;

IntraStatus predict_block(const NeighborRequest& request, int bit_depth,
                          const PredictParams& params, std::uint8_t* dst,
                          std::size_t dst_stride,
                          PredictResult* result = nullptr) noexcept;

// ---------------------------------------------------------------------------
// §5.4.5 sequence_intra_config syntax
// ---------------------------------------------------------------------------
//
// The field order below is normative.  `write_sequence_intra_config()` must be
// called with the BitWriter positioned at the first bit of
// sequence_intra_config(); `read_sequence_intra_config()` likewise.  The
// surrounding sequence_header_obu() owns byte alignment.

struct SequenceIntraConfig {
  bool enable_dip = false;
  bool enable_intra_edge_filter = false;
  bool enable_mrls = false;
  bool enable_cfl_intra = false;
  int cfl_ds_filter_index = 0;  // f(2), omitted when Monochrome
  bool enable_mhccp = false;
  bool enable_ibp = false;
};

class BitWriter {
 public:
  void put_bit(bool value);
  void put_bits(std::uint32_t value, int count);
  const std::vector<std::uint8_t>& bytes() const noexcept { return bytes_; }
  int bit_position() const noexcept { return bit_position_; }

 private:
  std::vector<std::uint8_t> bytes_;
  int bit_position_ = 0;  // number of bits written
};

class BitReader {
 public:
  explicit BitReader(std::span<const std::uint8_t> bytes) : bytes_(bytes) {}
  bool read_bit(bool& value);
  bool read_bits(int count, std::uint32_t& value);
  int bit_position() const noexcept { return bit_position_; }

 private:
  std::span<const std::uint8_t> bytes_;
  int bit_position_ = 0;
};

void write_sequence_intra_config(const SequenceIntraConfig& config,
                                 bool monochrome, BitWriter& writer);
bool read_sequence_intra_config(bool monochrome, BitReader& reader,
                                SequenceIntraConfig& config);

}  // namespace av2enc::ibp

#endif  // AV2ENC_ENCODER_IBP_IBP_H_

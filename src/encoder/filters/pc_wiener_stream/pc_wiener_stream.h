// AV2 v1.0.0 PC-Wiener stream syntax (§5.4.10, §5.18.7.11,
// §5.20.10.5–§5.20.10.6) plus the encoder-side §7.20.4 decision.
//
// This module is deliberately self-contained.  It does not edit or link the
// production restoration core (which rejects PC-Wiener) and it does not write
// any other filter syntax.  It reuses two read-only pieces:
//
//   * encoder/filters/spec_tables/pc_wiener.h   §7.20.4 classifier + 25-tap
//     fixed filter bank, recovered from the normative attachments;
//   * encoder/filters/restoration/wiener_core.h §7.20.2/§7.20.3 sample
//     sourcing and 32-tap Wiener NS filter, used to evaluate the frame-level
//     pixel-classified mode.
//
// Two distinct normative uses of PC-Wiener are covered:
//
// A. FrameRestorationType[0] == RESTORE_PC_WIENER (§5.20.10.5).  The only
//    per-unit syntax element is `use_pc_wiener` (S()); the 25-tap filters are
//    the fixed Pc_Wiener_Filters[index][Pc_Wiener_Sub_Classify[index][cls]]
//    bank.  No coefficients are signalled.
//
// B. frame_filters_on[0] == 1 with NumFilterClasses > 1 (§5.18.7.11,
//    §5.20.10.6).  This is the pixel-classified Wiener mode: the decoder
//    derives FilterClass from §7.20.4, maps it through SubclassLookup =
//    Pc_Wiener_Sub_Classify2[filter_set][num_filter_classes_idx], and uses
//    FrameLrWienerNs[0][subcls].  The PC-Wiener translated-filter group
//    (get_translated_pc_wiener) is addressable only when
//    lr_tools_disable[0][RESTORE_PC_WIENER] == 0; the encoder can select a
//    translated bank entry per class with match_index and merged=1.
//
// §5.20.10.6 also carries the per-unit Wiener NS syntax (merged/bank refs,
// subset and coefficients).  It is implemented here so the mixed switchable
// case and the class-map mode have an exact writer/reader; PC-Wiener itself
// never codes coefficients.
#ifndef AV2ENC_ENCODER_FILTERS_PC_WIENER_STREAM_PC_WIENER_STREAM_H_
#define AV2ENC_ENCODER_FILTERS_PC_WIENER_STREAM_PC_WIENER_STREAM_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "bitstream/bitwriter.h"
#include "entropy/av2/av2_entropy.h"

namespace av2enc::filters::pc_wiener_stream {

// Spec constants used by the LR syntax.
inline constexpr int kRestorationTileSizeMax = 512;
inline constexpr int kMiSize = 4;
inline constexpr int kWienerNsClasses = 16;
inline constexpr int kWienerNsLumaCoeffs = 16;
inline constexpr int kWienerNsChromaCoeffs = 18;
inline constexpr int kWienerNsCoeffStride = 18;
inline constexpr int kWienerNsShortCoeffs = 6;
inline constexpr int kLrBankSize = 4;
inline constexpr int kNumLutClasses = 256;
inline constexpr int kPcWienerFilterSets = 4;
inline constexpr int kPcWienerBank = 64;
inline constexpr int kPcWienerTranslatedCoeffs = 16;

// §6.17.7.7 FrameRestorationType.
enum RestorationType : int {
  kRestoreNone = 0,
  kRestorePcWiener = 1,
  kRestoreWienerNs = 2,
  kRestoreSwitchable = 3,
};

// §5.18.7.11 NumFilterClasses.
inline constexpr std::uint8_t kDecodeNumFilterClasses[8] = {1, 2, 3, 4,
                                                            6, 8, 12, 16};
inline constexpr std::uint8_t kEncodeNumFilterClasses[17] = {
    0, 0, 1, 2, 3, 0, 4, 0, 5, 0, 0, 0, 6, 0, 0, 0, 7};

int FilterSetIndex(int base_q_idx);

// §5.20.10.6 Shuffled_Index[64].
extern const std::uint8_t kShuffledIndex[64];

// get_translated_pc_wiener(m, j): Pc_Wiener_Filters[0][Shuffled_Index[m]]
// rounded to Wiener NS precision and clipped to the luma tap range.
int TranslatedPcWiener(int m, int j);

// Convenience: fill a 16-entry Wiener NS luma coefficient set.
void TranslatedPcWienerSet(int m, std::int16_t out[kWienerNsCoeffStride]);

// SubclassLookup = Pc_Wiener_Sub_Classify2[filter_set][num_classes_idx][cls].
const std::uint8_t* PcWienerSubclassLookup(int filter_set, int num_classes_idx);

// ---------------------------------------------------------------------------
// Minimal independent MSB-first reader (f(n), ns(n)).
// ---------------------------------------------------------------------------
class Reader {
 public:
  Reader(const std::uint8_t* data, std::size_t bit_size)
      : data_(data), bit_size_(bit_size) {}

  bool ReadBit(bool* value);
  bool ReadBits(unsigned count, std::uint32_t* value);
  bool ReadNs(unsigned n, std::uint32_t* value);
  std::size_t bit_pos() const { return bit_pos_; }

 private:
  const std::uint8_t* data_ = nullptr;
  std::size_t bit_size_ = 0;
  std::size_t bit_pos_ = 0;
};

void PutNs(BitWriter& writer, unsigned n, std::uint32_t value);

// ---------------------------------------------------------------------------
// §5.4.10 sequence_filter_config()
// ---------------------------------------------------------------------------
struct SequenceFilterConfig {
  bool disable_loopfilters_across_tiles = false;
  bool enable_cdef = false;
  bool enable_gdf = false;
  bool gdf_unit_matches_sb_size = false;
  bool enable_restoration = false;
  // Index 0 = luma, 1 = chroma.  Index 1 (PC-Wiener) is inferred 1 for chroma
  // and never coded.  When lr_tools_uv_present is false, chroma Wiener NS
  // inherits the luma flag.
  bool lr_tools_disable[2][4] = {
      {false, true, false, false},
      {false, true, false, false},
  };
  bool lr_tools_uv_present = false;
  bool enable_ccso = false;
  bool ccso_unit_matches_sb_size = false;
  bool cdef_on_skip_txfm_always_on = false;
  bool cdef_on_skip_txfm_disabled = false;
  std::uint8_t df_par_bits_minus_2 = 0;
};

// Writes the complete §5.4.10 row after `single_picture_header_flag` has been
// handled by the caller.  `single_picture_header_flag` selects the
// CdefOnSkipTxfm inference.  Returns the exact bit count, or 0 on error.
std::size_t WriteSequenceFilterConfig(BitWriter& writer,
                                      const SequenceFilterConfig& config,
                                      bool single_picture_header_flag,
                                      std::string* error);
bool ReadSequenceFilterConfig(Reader& reader, bool single_picture_header_flag,
                              SequenceFilterConfig* config, std::string* error);
std::size_t SequenceFilterConfigBits(const SequenceFilterConfig& config,
                                     bool single_picture_header_flag,
                                     std::string* error);

// ---------------------------------------------------------------------------
// §5.18.7.11 lr_params()
// ---------------------------------------------------------------------------
struct FrameFilter {
  // 0 = all-zero filter; m < num_classes = FrameLrWienerNs[m - 1];
  // m >= num_classes (luma, PC-Wiener enabled) = translated PC-Wiener bank
  // entry m - num_classes.
  int match_index = 0;
  bool merged = false;
  int subset = 0;
  bool uv_sym = false;
  std::int16_t coeffs[kWienerNsCoeffStride] = {};
};

struct LrParams {
  int frame_type[3] = {kRestoreNone, kRestoreNone, kRestoreNone};
  bool frame_filters_on[3] = {false, false, false};
  bool temporal_pred_flag[3] = {false, false, false};
  int rst_ref_pic_idx[3] = {0, 0, 0};
  int num_filter_classes = 1;
  int num_filter_classes_idx = 0;
  int luma_shift = 3;
  int chroma_shift = 3;

  // Final FrameLrWienerNs[plane][class][coeff].
  std::int16_t frame_coeffs[3][kWienerNsClasses][kWienerNsCoeffStride] = {};
  FrameFilter frame_filter[3][kWienerNsClasses];

  // Derived at the end of lr_params(): SubclassLookup[FilterClass].
  std::uint8_t subclass_lookup[kNumLutClasses] = {};
  int subclass_lookup_count = 0;  // 0 or kNumLutClasses
};

struct LrHeaderContext {
  bool coded_lossless = false;
  bool enable_restoration = false;
  bool single_picture_header_flag = false;
  bool frame_is_intra = true;
  bool is_switch_frame = false;
  int num_total_refs = 0;
  int sb_size = 64;
  int num_planes = 3;
  int sub_x = 1;
  int sub_y = 1;
  int base_q_idx = 90;
  bool lr_tools_disable[2][4] = {
      {false, true, false, false},
      {false, true, false, false},
  };
};

// `params` is in/out: its frame_type/frame_filters_on/num_filter_classes and
// frame_filter decisions are written; frame_coeffs/subclass_lookup are filled
// with what a decoder will reconstruct.  Returns exact bits or 0 on error.
std::size_t WriteLrParams(BitWriter& writer, LrParams* params,
                          const LrHeaderContext& context, std::string* error);
bool ReadLrParams(Reader& reader, const LrHeaderContext& context,
                  LrParams* params, std::string* error);
std::size_t LrParamsBits(LrParams* params, const LrHeaderContext& context,
                         std::string* error);

int CountUnits(int unit_size, int frame_size);

// ---------------------------------------------------------------------------
// §5.20.10.5 / §5.20.10.6 per-unit syntax
// ---------------------------------------------------------------------------
struct UnitDecision {
  int type = kRestoreNone;
  bool merged = false;
  int use_bank_k = -1;  // -1 selects the newest previous bank
  int subset = 0;
  bool uv_sym = false;
  std::int16_t coeffs[kWienerNsCoeffStride] = {};
};

struct TileState {
  LrParams params;
  int mi_cols = 0;
  int mi_rows = 0;
  int sub_x = 1;
  int sub_y = 1;

  int unit_size[3] = {64, 64, 64};
  int unit_rows[3] = {0, 0, 0};
  int unit_cols[3] = {0, 0, 0};

  std::vector<std::uint8_t> unit_type[3];
  std::vector<std::int16_t> unit_coeffs[3];

  std::int16_t ref_bank[3][kWienerNsClasses][kLrBankSize]
                       [kWienerNsCoeffStride] = {};
  int bank_ptr[3][kWienerNsClasses] = {};
  int bank_size[3][kWienerNsClasses] = {};

  std::uint16_t use_wiener_ns[3] = {};
  std::uint16_t use_pc_wiener[3] = {};
  std::uint16_t flex_restoration_type[3][3][3] = {};
  std::uint16_t wiener_ns_length[2][3] = {};
  std::uint16_t wiener_ns_uv_sym[3] = {};
  std::uint16_t wiener_ns_base[5] = {};
};

bool InitTileState(TileState* state, const LrParams& params, int mi_cols,
                   int mi_rows, int sub_x, int sub_y, std::string* error);

bool WriteUnit(av2::Encoder& encoder, TileState& state, int plane,
               int unit_row, int unit_col, const UnitDecision& decision,
               std::string* error);
bool ReadUnit(av2::Decoder& decoder, TileState& state, int plane,
              int unit_row, int unit_col, UnitDecision* decision,
              std::string* error);

struct UnitRange {
  int row_begin = 0;
  int row_end = 0;
  int col_begin = 0;
  int col_end = 0;
};

bool UnitRangeFor(TileState& state, int plane, int mi_row, int mi_col, int w4,
                  int h4, int tile_mi_row_start, int tile_mi_col_start,
                  UnitRange* range, std::string* error);

bool WriteBlock(av2::Encoder& encoder, TileState& state, int mi_row,
                int mi_col, int w4, int h4, int tile_mi_row_start,
                int tile_mi_col_start,
                const std::vector<UnitDecision>& decisions,
                std::string* error);
bool ReadBlock(av2::Decoder& decoder, TileState& state, int mi_row, int mi_col,
               int w4, int h4, int tile_mi_row_start, int tile_mi_col_start,
               std::vector<UnitDecision>* decisions, std::string* error);

// ---------------------------------------------------------------------------
// §7.20.4 encoder decision
// ---------------------------------------------------------------------------
struct DecisionPlanes {
  int width = 0;   // luma samples
  int height = 0;
  int bit_depth = 8;
  int base_q_idx = 90;
  int sub_x = 1;
  int sub_y = 1;
  int luma_shift = 3;
  int chroma_shift = 3;
  // CurrFrame (pre-CDEF) and CdefFrame (post-CDEF/CCSO).  For a gate with all
  // filters off both point at the same reconstruction.
  const std::uint8_t* curr[3] = {nullptr, nullptr, nullptr};
  const std::uint8_t* cdef[3] = {nullptr, nullptr, nullptr};
  int curr_stride[3] = {0, 0, 0};
  int cdef_stride[3] = {0, 0, 0};
  // Optional target for on/off decisions (original source, same layout).
  const std::uint8_t* target[3] = {nullptr, nullptr, nullptr};
  int target_stride[3] = {0, 0, 0};
  // LrTxSkip[mi_row][mi_col] on the luma MI grid; null means all zero.
  const std::uint8_t* lr_tx_skip = nullptr;
  int lr_tx_skip_stride = 0;
};

struct DecisionResult {
  LrParams params;
  std::vector<std::uint8_t> unit_type[3];
  std::vector<std::int16_t> unit_coeffs[3];
  std::vector<std::uint8_t> filter_class;  // mi_rows * mi_cols
  std::uint8_t subclass_lookup[kNumLutClasses] = {};
  int subclass_lookup_count = 0;

  int header_bits = 0;
  int tile_bits = 0;  // exact S() bits including §8.2.4 marker/padding
  int active_units = 0;
  int total_units = 0;
  double identity_sse = 0.0;
  double filtered_sse = 0.0;
  bool used_frame_class_mode = false;
  int num_filter_classes = 1;
};

// Mode A: FrameRestorationType[0] = RESTORE_PC_WIENER; per-unit on/off.
// `lambda` is in SSE units per syntax bit (0 = turn on any SSE reduction).
bool DecidePerUnitPcWiener(const DecisionPlanes& planes, double lambda,
                           DecisionResult* result, std::string* error);

// Mode B: frame_filters_on[0] = 1 and NumFilterClasses = num_classes > 1;
// each class selects a translated PC-Wiener filter with match + merged bits.
// The frame type is RESTORE_WIENER_NONSEP.
bool DecideFrameClassPcWiener(const DecisionPlanes& planes, int num_classes,
                              double lambda, DecisionResult* result,
                              std::string* error);

// Applies the §7.20.4 PC-Wiener filter for the units whose `unit_type` is
// RESTORE_PC_WIENER.  `dst` is a full luma plane; samples outside active units
// are copied from CdefFrame.  Returns the number of filtered 4x4 blocks.
int ApplyPerUnitPcWiener(const DecisionPlanes& planes,
                         const std::vector<std::uint8_t>& unit_type,
                         int unit_size, std::uint8_t* dst, int dst_stride);

// Applies §7.20.3 with FrameLrWienerNs[0] using the §7.20.4 class map and a
// subclass lookup.  Used by tests and the gate to produce the expected
// reconstruction for frame-class mode.
int ApplyFrameClassWiener(const DecisionPlanes& planes,
                          const std::int16_t frame_coeffs[kWienerNsClasses]
                                                    [kWienerNsCoeffStride],
                          const std::uint8_t* filter_class,
                          const std::uint8_t* subclass_lookup,
                          int subclass_count, std::uint8_t* dst,
                          int dst_stride);

// Classifies the luma plane with the recovered §7.20.4 tables.  Output is one
// byte per 4x4 luma block (mi_rows * mi_cols).
void BuildFilterClassGrid(const DecisionPlanes& planes,
                          std::vector<std::uint8_t>* filter_class);

}  // namespace av2enc::filters::pc_wiener_stream

#endif  // AV2ENC_ENCODER_FILTERS_PC_WIENER_STREAM_PC_WIENER_STREAM_H_

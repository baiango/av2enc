// AV2 v1.0.0 film grain bitstream syntax: OBU (§5.14), model (§5.18.10.2),
// per-frame config (§5.18.10.1), and the §7.23.6 grain-model slot state.
//
// Normative sources in spec/AV2_Spec_v1.0.0.pdf (commit 38f28e7):
//   * §5.2.1/§5.2.3  OBU envelope, extensible-OBU tail, trailing_bits()
//   * §5.14 / §6.13  film_grain_obu(): fgm_update_flags, fgm_chroma_idc
//   * §5.18.10.1 / §6.17.10.1  film_grain_config(): apply_grain/fgm_id/seed
//   * §5.18.10.2 / §6.17.10.2  film_grain_model()
//   * §7.3.8.8       film grain OBU availability
//   * §7.21.7        film grain synthesis (implemented by spec_tables)
//
// This module owns the *bitstream* half only.  Synthesis is the existing
// decode-side reference in encoder/filters/spec_tables/film_grain.{h,cpp};
// `ToSynthesisModel()` converts a parsed model into that struct so encoder
// tests and the decoder gate can compare against the normative process.
//
// The chroma idc value space is easy to get wrong: §6.4.1 Table 6.2 assigns
// CHROMA_FORMAT_420 = 0, CHROMA_FORMAT_400 = 1, CHROMA_FORMAT_444 = 2 and
// CHROMA_FORMAT_422 = 3 (it is *not* the usual 0=400, 1=420 ordering).
// GrainChromaIdc uses that numbering because fgm_chroma_idc is compared
// directly against chroma_format_idc in §6.17.10.1.
#ifndef AV2ENC_ENCODER_FILTERS_FILM_GRAIN_STREAM_FILM_GRAIN_SYNTAX_H_
#define AV2ENC_ENCODER_FILTERS_FILM_GRAIN_STREAM_FILM_GRAIN_SYNTAX_H_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "bitstream/bitwriter.h"
#include "encoder/filters/spec_tables/film_grain.h"

namespace av2enc::filters::grain {

// MAX_FILM_GRAIN (§5.4 table of constants, p.35); fgm_update_flags is f(8).
inline constexpr int kNumGrainSlots = 8;
// §6.17.10.2 conformance: num_{y,cb,cr}_points <= 14.
inline constexpr int kMaxGrainPoints = 14;
// numPosLuma(3) + 1 = 2*3*(3+1) + 1; §7.21.7.3 indexes the chroma luma term
// one past the neighbour taps.
inline constexpr int kMaxArPositions = 25;
// OBU type 23 (§6.2.2 Table 6.1).
inline constexpr std::uint8_t kFilmGrainObuType = 23;

// §6.4.1 Table 6.2 numbering (see the file comment).
enum class GrainChromaIdc : int {
  k420 = 0,
  k400 = 1,
  k444 = 2,
  k422 = 3,
};

struct GrainChromaLayout {
  int sub_x = 1;
  int sub_y = 1;
  bool monochrome = false;
};

GrainChromaLayout ChromaLayout(GrainChromaIdc idc);

// GrainChromaIdc values must equal chroma_format_idc (§6.17.10.1), so this is
// a checked cast.
bool ChromaIdcFromSequence(int chroma_format_idc, GrainChromaIdc* idc);

// film_grain_model() syntax elements, in the *writer* representation:
//   * point_*_value[] hold the final (cumulative) x coordinates, 0..255;
//   * ar_coeffs_*[] hold the final signed coefficient (after the syntax
//     parser's sign recentring), not the raw unsigned bit pattern;
//   * bits_per_ar_coeff_*_minus_5 select the coefficient bit width.
// Unused arrays are zero; NormalizeModel() canonicalises this so a
// write/read round trip is comparable field-for-field.
struct GrainModel {
  bool chroma_scaling_from_luma = false;

  int num_y_points = 0;
  int point_y_value[kMaxGrainPoints] = {};
  int point_y_scaling[kMaxGrainPoints] = {};
  // point_value_increment_bits_minus_1 / point_scaling_bits_minus_5 are
  // separate syntax elements per plane and are only present when that
  // plane's num_*_points > 0.  Absent fields are canonicalised to 0.
  int point_y_increment_bits_minus_1 = 7;  // 0..7  -> 1..8 bits
  int point_y_scaling_bits_minus_5 = 3;    // 0..3  -> 5..8 bits

  int num_cb_points = 0;
  int point_cb_value[kMaxGrainPoints] = {};
  int point_cb_scaling[kMaxGrainPoints] = {};
  int point_cb_increment_bits_minus_1 = 7;
  int point_cb_scaling_bits_minus_5 = 3;

  int num_cr_points = 0;
  int point_cr_value[kMaxGrainPoints] = {};
  int point_cr_scaling[kMaxGrainPoints] = {};
  int point_cr_increment_bits_minus_1 = 7;
  int point_cr_scaling_bits_minus_5 = 3;

  int grain_scaling_minus_8 = 0;  // 0..3
  int ar_coeff_lag = 0;           // 0..3; numPosLuma = 2*lag*(lag+1)

  int bits_per_ar_coeff_y_minus_5 = 0;  // 0..3
  int ar_coeffs_y[kMaxArPositions] = {};

  int bits_per_ar_coeff_cb_minus_5 = 0;
  int ar_coeffs_cb[kMaxArPositions] = {};

  int bits_per_ar_coeff_cr_minus_5 = 0;
  int ar_coeffs_cr[kMaxArPositions] = {};

  int ar_coeff_shift_minus_6 = 0;  // 0..3
  int grain_scale_shift = 0;       // 0..3

  int cb_mult = 0;       // 0..255
  int cb_luma_mult = 0;  // 0..255
  int cb_offset = 0;     // 0..511
  int cr_mult = 0;
  int cr_luma_mult = 0;
  int cr_offset = 0;

  bool overlap_flag = false;
  bool clip_to_restricted_range = false;
  bool fg_mc_identity = false;       // inferred 0 when illegal
  bool film_grain_block_size = false;  // false = 16x16, true = 32x32
};

bool operator==(const GrainModel& a, const GrainModel& b);
inline bool operator!=(const GrainModel& a, const GrainModel& b) {
  return !(a == b);
}

// Returns the canonical model for `idc`: fields that the syntax does not
// carry for this layout (e.g. chroma coefficients when num_cb_points == 0)
// are zeroed, and fg_mc_identity is forced to false when
// clip_to_restricted_range is false.
GrainModel NormalizeModel(const GrainModel& model, GrainChromaIdc idc);

// §6.17.10.2 bitstream conformance checks that this writer enforces.  If
// `why` is non-null it receives a short reason on failure.
bool ValidateModel(const GrainModel& model, GrainChromaIdc idc, std::string* why);

// §5.14 payload: fgm_update_flags, fgm_chroma_idc, and one
// film_grain_model() + save_grain_model() per set flag bit.
struct FilmGrainObu {
  std::uint8_t update_flags = 0;  // 1..255 (§6.13)
  GrainChromaIdc chroma_idc = GrainChromaIdc::k420;
  GrainModel models[kNumGrainSlots];  // slot i used iff bit i is set
};

bool ValidateObu(const FilmGrainObu& obu, std::string* why);

// Payload writers/readers.  `WriteObuPayload` writes the §5.14 syntax followed
// by §5.2.3 trailing_bits().  `ReadObuPayload` parses the syntax, verifies the
// trailing bits, and reports how many payload bits were consumed (including
// the trailing one bit, up to but not including zero padding).
bool WriteObuPayload(BitWriter& writer, const FilmGrainObu& obu,
                     std::string* error);
bool ReadObuPayload(std::span<const std::uint8_t> payload, FilmGrainObu* obu,
                    std::size_t* consumed_bits, std::string* error);

// Annex B.2 framing: leb128(size) obu_header() payload.  No extension flag:
// OBU_FILM_GRAIN is not in is_extensible_obu() (§5.2.1 p.54).
std::vector<std::uint8_t> MakeFilmGrainObu(const FilmGrainObu& obu,
                                           std::string* error);

// Parses one complete Annex B OBU starting at `data[0]`.  `consumed` receives
// the number of bytes (leb128 prefix + header + payload).
bool ReadFilmGrainObu(std::span<const std::uint8_t> data, FilmGrainObu* obu,
                      std::size_t* consumed, std::string* error);

// §5.18.10.1 film_grain_config().
//
// The syntax has three inference branches, so the writer needs the frame
// context that a real header writer already has:
//   * film_grain_params_present == 0, or the frame is neither immediately nor
//     implicitly output  -> apply_grain is inferred 0 and nothing is coded;
//   * single_picture_header_flag == 1 -> apply_grain is inferred 1 (no bit),
//     then fgm_id + grain_seed are coded;
//   * otherwise -> apply_grain f(1), then fgm_id + grain_seed if set.
struct FilmGrainConfig {
  bool apply_grain = false;
  int fgm_id = 0;                // 0..7
  std::uint16_t grain_seed = 0;  // f(16)
};

struct FilmGrainConfigContext {
  bool film_grain_params_present = false;
  bool single_picture_header_flag = false;
  bool immediate_output_frame = true;
  bool implicit_output_frame = false;
};

// Returns true when §5.18.10.1 infers apply_grain (no bit is coded) and stores
// the inferred value in `value`.  Returns false when apply_grain f(1) is
// present in the bitstream.
bool ApplyGrainIsInferred(const FilmGrainConfigContext& ctx, bool* value);

// Writes the syntax.  `error` is set for an invalid config (e.g. fgm_id > 7,
// apply_grain required by inference but false).
bool WriteFilmGrainConfig(BitWriter& writer, const FilmGrainConfig& config,
                          const FilmGrainConfigContext& ctx,
                          std::string* error);

// Reads the syntax.  `ctx` supplies the sequence/frame-header state.
bool ReadFilmGrainConfig(std::span<const std::uint8_t> payload,
                         std::size_t* bit_position,
                         const FilmGrainConfigContext& ctx,
                         FilmGrainConfig* config, std::string* error);

// Exact number of bits film_grain_config() occupies for `config`/`ctx`.
// Returns 0 for an invalid config (see WriteFilmGrainConfig) as well as for
// the legitimate zero-bit cases.
std::size_t FilmGrainConfigBits(const FilmGrainConfig& config,
                                const FilmGrainConfigContext& ctx);

// Exact number of bits film_grain_model() occupies (no trailing bits, and not
// counting the 8-bit fgm_update_flags or the fgm_chroma_idc uvlc()).
std::size_t ModelBits(const GrainModel& model, GrainChromaIdc idc);

// Approximate evaluation-time cost of a film grain OBU (payload + OBU header
// + leb128 prefix + trailing bits) in bytes, for rate accounting.
std::size_t FilmGrainObuBytes(const FilmGrainObu& obu);

// §7.23 save_grain_params()/load_grain_model() slot state.  FilmGrainPresent
// is initialised to zero at the start of a bitstream (§6.13) and slot i is
// overwritten by any film grain OBU that sets bit i.
struct GrainSlot {
  bool present = false;
  GrainModel model;
  GrainChromaIdc chroma_idc = GrainChromaIdc::k420;
  int tlayer_id = 0;  // FgmTLayerId[i]
  int mlayer_id = 0;  // FgmMLayerId[i]
};

class GrainSlotState {
 public:
  GrainSlotState() { Reset(); }

  // Initialises every FilmGrainPresent entry to 0 (§6.13).
  void Reset();

  // save_grain_model(i).
  bool SaveModel(int slot, const GrainModel& model, GrainChromaIdc idc,
                 int tlayer_id, int mlayer_id, std::string* error);

  // load_grain_model(idx).
  bool LoadModel(int slot, GrainModel* model, GrainChromaIdc* idc,
                 std::string* error) const;

  const GrainSlot& slot(int i) const { return slots_[i]; }

 private:
  GrainSlot slots_[kNumGrainSlots];
};

// §6.17.10.1 + §7.3.8.8 availability check for one frame header:
//   FilmGrainPresent[fgm_id] == 1,
//   FgmChromaIdc[fgm_id] == chroma_format_idc,
//   TLayerDependencyMap[mlayer][tlayer][FgmTLayerId[fgm_id]] == 1,
//   MLayerDependencyMap[mlayer][FgmMLayerId[fgm_id]] == 1.
bool ValidateConfigAvailability(const GrainSlotState& state,
                                const FilmGrainConfig& config,
                                GrainChromaIdc frame_chroma_idc,
                                int tlayer_id, int mlayer_id,
                                std::string* why);

// Converts to the decode-side synthesis struct (field-for-field; no
// validation).  Callers pass the frame geometry to
// spec::FilmGrainModelValid().
bool ToSynthesisModel(const GrainModel& model,
                      spec::FilmGrainModel* out);

// Minimal MSB-first bit reader used to parse OBU payloads and frame headers.
class GrainBitReader {
 public:
  explicit GrainBitReader(std::span<const std::uint8_t> data,
                          std::size_t start_bit = 0)
      : data_(data), bit_pos_(start_bit) {}

  bool ReadBit(bool* value);
  bool ReadBits(unsigned count, std::uint64_t* value);
  bool ReadUvlc(std::uint32_t* value);
  // Skips to the next byte boundary; the skipped bits must be zero.
  bool ByteAlign();
  std::size_t bit_position() const { return bit_pos_; }
  std::size_t bit_size() const { return data_.size() * 8U; }

 private:
  std::span<const std::uint8_t> data_;
  std::size_t bit_pos_ = 0;
};

}  // namespace av2enc::filters::grain

#endif  // AV2ENC_ENCODER_FILTERS_FILM_GRAIN_STREAM_FILM_GRAIN_SYNTAX_H_

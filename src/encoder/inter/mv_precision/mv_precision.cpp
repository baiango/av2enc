#include "encoder/inter/mv_precision/mv_precision.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace av2enc::inter::mvprec {
namespace {

using av2enc::av2::Status;

constexpr int kMaxFrameDistance = 31;
constexpr std::int32_t kDivMult[32] = {
    0,    16384, 8192, 5461, 4096, 3276, 2730, 2340, 2048, 1820, 1638,
    1489, 1365,  1260, 1170, 1092, 1024, 963,  910,  862,  819,  780,
    744,  712,   682,  655,  630,  606,  585,  564,  546,  528};

int floor_log2(std::int32_t value) {
  int shift = 0;
  while (value > 1) {
    value >>= 1;
    ++shift;
  }
  return shift;
}

int clip3_int(int lo, int hi, int value) {
  return value < lo ? lo : (value > hi ? hi : value);
}

std::int32_t round2_signed_64(std::int64_t value, int shift) {
  if (shift <= 0) return static_cast<std::int32_t>(value);
  const std::int64_t a = value < 0 ? -value : value;
  const std::int64_t half = static_cast<std::int64_t>(1) << (shift - 1);
  const std::int64_t rounded = (a + half) >> shift;
  return static_cast<std::int32_t>(value < 0 ? -rounded : rounded);
}

bool sym(av2::Encoder& enc, std::uint16_t* cdf, int n, int symbol,
         std::string* error) {
  if (enc.encode_symbol(cdf, n, symbol) != Status::kOk) {
    if (error != nullptr) *error = "entropy encode_symbol failed";
    return false;
  }
  return true;
}

bool sym(av2::Decoder& dec, std::uint16_t* cdf, int n, int* symbol,
         std::string* error) {
  if (dec.decode_symbol(cdf, n, symbol) != Status::kOk) {
    if (error != nullptr) *error = "entropy decode_symbol failed";
    return false;
  }
  return true;
}

bool raw_bit(av2::Encoder& enc, bool bit, std::string* error) {
  if (enc.encode_bool(bit) != Status::kOk) {
    if (error != nullptr) *error = "entropy encode_bool failed";
    return false;
  }
  return true;
}

bool raw_bit(av2::Decoder& dec, bool* bit, std::string* error) {
  if (dec.read_bool(bit) != Status::kOk) {
    if (error != nullptr) *error = "entropy read_bool failed";
    return false;
  }
  return true;
}

// §5.10 ns(n): truncated binary code.
bool write_ns(av2::Encoder& enc, std::uint32_t n, std::uint32_t value,
              std::string* error) {
  if (n < 2) {
    if (error != nullptr) *error = "ns(n) requires n >= 2";
    return false;
  }
  unsigned width = 0;
  while ((1U << width) < n) ++width;
  const std::uint32_t m = (1U << width) - n;
  if (value < m) {
    for (int i = static_cast<int>(width) - 2; i >= 0; --i) {
      if (!raw_bit(enc, ((value >> i) & 1U) != 0U, error)) return false;
    }
    return true;
  }
  const std::uint32_t x = value + m;
  for (int i = static_cast<int>(width) - 1; i > 0; --i) {
    if (!raw_bit(enc, ((x >> i) & 1U) != 0U, error)) return false;
  }
  return raw_bit(enc, (x & 1U) != 0U, error);
}

bool read_ns(av2::Decoder& dec, std::uint32_t n, std::uint32_t* value,
             std::string* error) {
  if (n < 2) {
    if (error != nullptr) *error = "ns(n) requires n >= 2";
    return false;
  }
  unsigned width = 0;
  while ((1U << width) < n) ++width;
  const std::uint32_t m = (1U << width) - n;
  std::uint32_t x = 0;
  for (unsigned i = 0; i + 1 < width; ++i) {
    bool bit = false;
    if (!raw_bit(dec, &bit, error)) return false;
    x = (x << 1U) | (bit ? 1U : 0U);
  }
  if (x < m) {
    *value = x;
    return true;
  }
  bool last = false;
  if (!raw_bit(dec, &last, error)) return false;
  *value = ((x << 1U) | (last ? 1U : 0U)) - m;
  return true;
}

int shell_class_symbol_count(int precision, int shell_set) {
  const int base = (11 + precision) >> 1;
  if (shell_set == 0) return base;
  return std::min(11 + precision - base, 8);
}

// Shell MVD in precision units.  `mag[0]` is row, `mag[1]` is column.
bool write_shell(av2::Encoder& enc, MvPrecisionCdfState& cdfs,
                 std::int32_t mag_y, std::int32_t mag_x, int precision,
                 std::string* error) {
  if (!me2::me2_precision_valid(precision) ||
      precision == me2::kMvPrecisionTwoPel) {
    if (error != nullptr) *error = "invalid MvPrecision (no shell CDF)";
    return false;
  }
  const std::int32_t shell = mag_y + mag_x;
  if (shell < 0 || shell >= (1 << 17)) {
    if (error != nullptr) *error = "shell index out of range";
    return false;
  }
  const int cls = shell == 0 ? 0 : floor_log2(shell);
  const int base = (11 + precision) >> 1;
  const bool setsym = cls >= base;
  int class_sym = setsym ? cls - base : cls;
  bool last_two = false;
  if (setsym && precision == me2::kMvPrecisionEighthPel && cls >= 15) {
    class_sym = 7;
    last_two = true;
  }
  if (!sym(enc, &cdfs.shell_set[0], 2, setsym ? 1 : 0, error)) return false;
  const int n = shell_class_symbol_count(precision, setsym ? 1 : 0);
  if (!sym(enc, &cdfs.shell_class[precision][setsym ? 1 : 0][0], n, class_sym,
           error)) {
    return false;
  }
  if (last_two) {
    if (!sym(enc, &cdfs.shell_last_two[0], 2, cls - 15, error)) return false;
  }
  const std::int32_t shell_base = cls == 0 ? 0 : (1 << cls);
  const std::int32_t offset = shell - shell_base;
  if (cls < 2) {
    if (!sym(enc, &cdfs.shell_offset_low[cls][0], 2,
             static_cast<int>(offset), error)) {
      return false;
    }
  } else if (cls == 2) {
    if (offset == 0) {
      if (!sym(enc, &cdfs.shell_offset_class2[0], 2, 0, error)) return false;
    } else {
      if (!sym(enc, &cdfs.shell_offset_class2[0], 2, 1, error)) return false;
      if (offset == 1) {
        if (!raw_bit(enc, false, error)) return false;
      } else {
        if (!raw_bit(enc, true, error)) return false;
        if (!raw_bit(enc, offset == 3, error)) return false;
      }
    }
  } else {
    for (int i = 0; i < cls; ++i) {
      if (!sym(enc, &cdfs.shell_offset_other[i][0], 2,
               static_cast<int>((offset >> i) & 1), error)) {
        return false;
      }
    }
  }

  if (shell > 0) {
    const std::int32_t maximum_pair = shell >> 1;
    bool skip_col_bit = false;
    std::int32_t col = 0;
    if (mag_x == maximum_pair && ((shell & 1) == 0)) {
      skip_col_bit = true;
      col = maximum_pair;
    } else if (mag_x <= maximum_pair) {
      col = mag_x;
    } else {
      col = mag_y;
    }
    if (maximum_pair > 0) {
      const int max_bits = std::min<std::int32_t>(maximum_pair, 2);
      const std::int32_t prefix_col = std::min<std::int32_t>(col, 2);
      for (int i = 0; i < max_bits; ++i) {
        const bool greater = prefix_col > i;
        if (!sym(enc, &cdfs.col_mv_greater[i][0], 2, greater ? 1 : 0, error)) {
          return false;
        }
        if (!greater) break;
      }
      if (maximum_pair > 2 && prefix_col == 2) {
        if (!write_ns(enc, static_cast<std::uint32_t>(maximum_pair - 1),
                      static_cast<std::uint32_t>(col - 2), error)) {
          return false;
        }
      }
    }
    if (!skip_col_bit) {
      const bool index = mag_x > maximum_pair;
      if (!sym(enc, &cdfs.col_mv_index[std::min(cls, 3)][0], 2,
               index ? 1 : 0, error)) {
        return false;
      }
    }
  }
  return true;
}

bool read_shell(av2::Decoder& dec, MvPrecisionCdfState& cdfs, int precision,
                std::int32_t* mag_y, std::int32_t* mag_x, std::string* error) {
  if (!me2::me2_precision_valid(precision) ||
      precision == me2::kMvPrecisionTwoPel) {
    if (error != nullptr) *error = "invalid MvPrecision (no shell CDF)";
    return false;
  }
  int shell_set = 0;
  if (!sym(dec, &cdfs.shell_set[0], 2, &shell_set, error)) return false;
  const int base = (11 + precision) >> 1;
  const int n = shell_class_symbol_count(precision, shell_set);
  int class_sym = 0;
  if (!sym(dec, &cdfs.shell_class[precision][shell_set][0], n, &class_sym,
           error)) {
    return false;
  }
  int shell_class = class_sym;
  if (shell_set != 0) {
    shell_class += base;
    if (precision == me2::kMvPrecisionEighthPel && class_sym == 7) {
      int last_two = 0;
      if (!sym(dec, &cdfs.shell_last_two[0], 2, &last_two, error)) return false;
      shell_class += last_two;
    }
  }
  std::int32_t shell_offset = 0;
  if (shell_class < 2) {
    int value = 0;
    if (!sym(dec, &cdfs.shell_offset_low[shell_class][0], 2, &value, error)) {
      return false;
    }
    shell_offset = value;
  } else if (shell_class == 2) {
    int value = 0;
    if (!sym(dec, &cdfs.shell_offset_class2[0], 2, &value, error)) return false;
    if (value == 0) {
      shell_offset = 0;
    } else {
      bool high = false;
      if (!raw_bit(dec, &high, error)) return false;
      if (!high) {
        shell_offset = 1;
      } else {
        bool top = false;
        if (!raw_bit(dec, &top, error)) return false;
        shell_offset = top ? 3 : 2;
      }
    }
  } else {
    for (int i = 0; i < shell_class; ++i) {
      int bit = 0;
      if (!sym(dec, &cdfs.shell_offset_other[i][0], 2, &bit, error)) {
        return false;
      }
      shell_offset |= static_cast<std::int32_t>(bit) << i;
    }
  }
  const std::int32_t shell_base = shell_class == 0 ? 0 : (1 << shell_class);
  const std::int32_t shell_index = shell_base + shell_offset;
  std::int32_t row = 0;
  std::int32_t col = 0;
  if (shell_index > 0) {
    const std::int32_t maximum_pair = shell_index >> 1;
    if (maximum_pair > 0) {
      const int max_bits = std::min<std::int32_t>(maximum_pair, 2);
      for (int i = 0; i < max_bits; ++i) {
        int greater = 0;
        if (!sym(dec, &cdfs.col_mv_greater[i][0], 2, &greater, error)) {
          return false;
        }
        col = i + greater;
        if (greater == 0) break;
      }
      if (maximum_pair > 2 && col == 2) {
        std::uint32_t remainder = 0;
        if (!read_ns(dec, static_cast<std::uint32_t>(maximum_pair - 1),
                     &remainder, error)) {
          return false;
        }
        col = static_cast<std::int32_t>(remainder + 2);
      }
    }
    const bool skip_col_bit =
        col == maximum_pair && ((shell_index & 1) == 0);
    if (skip_col_bit) {
      col = maximum_pair;
    } else {
      int index = 0;
      if (!sym(dec, &cdfs.col_mv_index[std::min(shell_class, 3)][0], 2, &index,
               error)) {
        return false;
      }
      if (index == 0) {
        // col already holds col_mv_greater's truncated-unary value.
      } else {
        col = shell_index - col;
      }
    }
    row = shell_index - col;
  }
  const int shift = me2::kMvPrecisionEighthPel - precision;
  *mag_y = row << shift;
  *mag_x = col << shift;
  return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Motion modes
// ---------------------------------------------------------------------------

bool write_sequence_motion_modes(BitWriter& writer,
                                 const SequenceMotionModeSyntax& syntax,
                                 std::string* error) {
  bool any = false;
  for (int mode = kMotionInterIntra; mode < kMotionModes; ++mode) {
    writer.put_bit(syntax.seq_enabled.enabled[mode]);
    any = any || syntax.seq_enabled.enabled[mode];
  }
  if (any != syntax.seq_frame_motion_modes_present_flag) {
    if (error != nullptr) {
      *error = "seq_frame_motion_modes_present_flag must equal any(seq modes)";
    }
    return false;
  }
  if (any) writer.put_bit(true);
  if (syntax.seq_enabled.enabled[kMotionDeltaWarp]) {
    writer.put_bit(syntax.enable_six_param_warp_delta);
  } else if (syntax.enable_six_param_warp_delta) {
    if (error != nullptr) {
      *error = "enable_six_param_warp_delta requires DELTAWARP";
    }
    return false;
  }
  return true;
}

bool read_sequence_motion_modes(BitReader& reader,
                                SequenceMotionModeSyntax* syntax,
                                std::string* error) {
  if (syntax == nullptr) {
    if (error != nullptr) *error = "null syntax";
    return false;
  }
  SequenceMotionModeSyntax out;
  bool any = false;
  for (int mode = kMotionInterIntra; mode < kMotionModes; ++mode) {
    bool bit = false;
    if (!reader.read_bit(&bit)) {
      if (error != nullptr) *error = "truncated seq_enabled_motion_modes";
      return false;
    }
    out.seq_enabled.enabled[mode] = bit;
    any = any || bit;
  }
  if (any) {
    bool bit = false;
    if (!reader.read_bit(&bit)) {
      if (error != nullptr) *error = "truncated seq_frame_motion_modes_present";
      return false;
    }
    out.seq_frame_motion_modes_present_flag = bit;
  }
  if (out.seq_enabled.enabled[kMotionDeltaWarp]) {
    bool bit = false;
    if (!reader.read_bit(&bit)) {
      if (error != nullptr) *error = "truncated enable_six_param_warp_delta";
      return false;
    }
    out.enable_six_param_warp_delta = bit;
  }
  *syntax = out;
  return true;
}

bool write_frame_motion_modes(BitWriter& writer,
                              const SequenceMotionModeSyntax& sequence,
                              const MotionModeSet& frame,
                              std::string* error) {
  for (int mode = kMotionInterIntra; mode < kMotionModes; ++mode) {
    if (!sequence.seq_frame_motion_modes_present_flag) {
      if (frame.enabled[mode] != sequence.seq_enabled.enabled[mode]) {
        if (error != nullptr) {
          *error = "frame motion mode must equal the sequence value when "
                   "seq_frame_motion_modes_present_flag == 0";
        }
        return false;
      }
    } else if (sequence.seq_enabled.enabled[mode]) {
      writer.put_bit(frame.enabled[mode]);
    } else if (frame.enabled[mode]) {
      if (error != nullptr) {
        *error = "frame enables a motion mode the sequence disabled";
      }
      return false;
    }
  }
  if (frame.enabled[kMotionSimple]) {
    if (error != nullptr) *error = "frame_enabled_motion_modes[SIMPLE] is not coded";
    return false;
  }
  return true;
}

bool read_frame_motion_modes(BitReader& reader,
                             const SequenceMotionModeSyntax& sequence,
                             MotionModeSet* frame, std::string* error) {
  if (frame == nullptr) {
    if (error != nullptr) *error = "null frame modes";
    return false;
  }
  MotionModeSet out;
  for (int mode = kMotionInterIntra; mode < kMotionModes; ++mode) {
    if (!sequence.seq_frame_motion_modes_present_flag) {
      out.enabled[mode] = sequence.seq_enabled.enabled[mode];
    } else if (sequence.seq_enabled.enabled[mode]) {
      bool bit = false;
      if (!reader.read_bit(&bit)) {
        if (error != nullptr) *error = "truncated frame_enabled_motion_modes";
        return false;
      }
      out.enabled[mode] = bit;
    } else {
      out.enabled[mode] = false;
    }
  }
  *frame = out;
  return true;
}

bool write_sequence_mv_res(BitWriter& writer,
                           const SequenceMvResSyntax& syntax,
                           std::string* error) {
  (void)error;
  writer.put_bit(syntax.enable_adaptive_mvd);
  writer.put_bit(syntax.enable_mvd_sign_derive);
  writer.put_bit(syntax.enable_flex_mvres);
  return true;
}

bool read_sequence_mv_res(BitReader& reader, SequenceMvResSyntax* syntax,
                          std::string* error) {
  if (syntax == nullptr) {
    if (error != nullptr) *error = "null syntax";
    return false;
  }
  SequenceMvResSyntax out;
  if (!reader.read_bit(&out.enable_adaptive_mvd) ||
      !reader.read_bit(&out.enable_mvd_sign_derive) ||
      !reader.read_bit(&out.enable_flex_mvres)) {
    if (error != nullptr) *error = "truncated sequence mv-res flags";
    return false;
  }
  *syntax = out;
  return true;
}

bool write_frame_mv_precision(BitWriter& writer,
                              const me2::FramePrecisionSyntax& syntax,
                              int force_integer_mv, std::string* error) {
  if (syntax.valid == false) {
    if (error != nullptr) *error = "invalid FramePrecisionSyntax";
    return false;
  }
  if (force_integer_mv != 0) {
    if (syntax.frame_precision != me2::kMvPrecisionOnePel ||
        syntax.coded_bits != 0 || syntax.use_per_block_mv_precision != 0) {
      if (error != nullptr) *error = "force_integer_mv pins MV_PRECISION_ONE_PEL";
      return false;
    }
    return true;
  }
  writer.put_bit(syntax.use_qtr_precision_mv != 0);
  if (syntax.use_qtr_precision_mv == 0) {
    writer.put_bit(syntax.allow_high_precision_mv != 0);
  }
  const me2::FramePrecisionSyntax check = me2::me2_frame_precision_from_bits(
      0, syntax.use_qtr_precision_mv, syntax.allow_high_precision_mv,
      syntax.use_per_block_mv_precision);
  if (check.frame_precision != syntax.frame_precision) {
    if (error != nullptr) *error = "frame precision bits disagree with target";
    return false;
  }
  return true;
}

bool read_frame_mv_precision(BitReader& reader, int force_integer_mv,
                             int enable_flex_mvres,
                             me2::FramePrecisionSyntax* syntax,
                             std::string* error) {
  if (syntax == nullptr) {
    if (error != nullptr) *error = "null syntax";
    return false;
  }
  if (force_integer_mv != 0) {
    *syntax = me2::me2_frame_precision_from_bits(1, 0, 0, enable_flex_mvres);
    return true;
  }
  bool use_qtr = false;
  if (!reader.read_bit(&use_qtr)) {
    if (error != nullptr) *error = "truncated use_qtr_precision_mv";
    return false;
  }
  bool allow_high = false;
  if (!use_qtr && !reader.read_bit(&allow_high)) {
    if (error != nullptr) *error = "truncated allow_high_precision_mv";
    return false;
  }
  *syntax = me2::me2_frame_precision_from_bits(0, use_qtr ? 1 : 0,
                                               allow_high ? 1 : 0,
                                               enable_flex_mvres);
  return true;
}

bool write_frame_interp_filter(BitWriter& writer, bool is_switchable,
                               int filter, std::string* error) {
  if (!frame_interp_filter_legal(filter, is_switchable)) {
    if (error != nullptr) *error = "illegal frame interpolation_filter";
    return false;
  }
  writer.put_bit(is_switchable);
  if (!is_switchable) writer.put_bits(static_cast<std::uint64_t>(filter), 2U);
  return true;
}

bool read_frame_interp_filter(BitReader& reader, bool* is_switchable,
                              int* filter, std::string* error) {
  if (is_switchable == nullptr || filter == nullptr) {
    if (error != nullptr) *error = "null output";
    return false;
  }
  bool switchable = false;
  if (!reader.read_bit(&switchable)) {
    if (error != nullptr) *error = "truncated is_filter_switchable";
    return false;
  }
  if (switchable) {
    *is_switchable = true;
    *filter = kInterpSwitchable;
    return true;
  }
  std::uint32_t value = 0;
  if (!reader.read_bits(2, &value)) {
    if (error != nullptr) *error = "truncated interpolation_filter";
    return false;
  }
  *is_switchable = false;
  *filter = static_cast<int>(value);
  return true;
}

// ---------------------------------------------------------------------------
// YMode predicates
// ---------------------------------------------------------------------------

bool allow_amvd_mode(int y_mode) {
  return y_mode == kYModeNewMv || y_mode == kYModeNewNewMv ||
         y_mode == kYModeNearNewMv || y_mode == kYModeNewNearMv ||
         y_mode == kYModeJointNewMv;
}

bool has_newmv(int y_mode) {
  return y_mode == kYModeNewMv || y_mode == kYModeNewNewMv ||
         y_mode == kYModeNearNewMv || y_mode == kYModeNewNearMv ||
         y_mode == kYModeWarpNewMv || y_mode == kYModeJointNewMv;
}

bool has_nearmv(int y_mode) {
  return y_mode == kYModeNearMv || y_mode == kYModeNearNearMv ||
         y_mode == kYModeNearNewMv || y_mode == kYModeNewNearMv;
}

bool is_joint_mvd_coding_mode(int y_mode) {
  return y_mode == kYModeJointNewMv;
}

// ---------------------------------------------------------------------------
// CDF init
// ---------------------------------------------------------------------------

void init_default_mv_precision_cdfs(MvPrecisionCdfState* state) {
  if (state == nullptr) return;
  std::memset(state, 0, sizeof(*state));
  std::memcpy(state->use_amvd, cdf::kUseAmvdCdf, sizeof(state->use_amvd));
  std::memcpy(state->mv_joint, cdf::kMvJointAdaptiveCdf,
              sizeof(state->mv_joint));
  std::memcpy(state->amvd_index, cdf::kAmvdIndicesCdf,
              sizeof(state->amvd_index));
  std::memcpy(state->jmvd_scale_mode, cdf::kJmvdScaleModeCdf,
              sizeof(state->jmvd_scale_mode));
  std::memcpy(state->jmvd_adaptive_scale_mode,
              cdf::kJmvdAdaptiveScaleModeCdf,
              sizeof(state->jmvd_adaptive_scale_mode));
  std::memcpy(state->use_most_probable_precision,
              cdf::kUseMostProbablePrecisionCdf,
              sizeof(state->use_most_probable_precision));
  std::memcpy(state->pb_mv_precision, cdf::kPbMvPrecisionCdf,
              sizeof(state->pb_mv_precision));
  std::memcpy(state->interp_filter, cdf::kInterpFilterCdf,
              sizeof(state->interp_filter));
  std::memcpy(state->shell_set, cdf::kJointShellSetCdf,
              sizeof(state->shell_set));
  std::memcpy(state->shell_last_two, cdf::kJointShellLastTwoClassesCdf,
              sizeof(state->shell_last_two));
  std::memcpy(state->shell_offset_low, cdf::kShellOffsetLowClassCdf,
              sizeof(state->shell_offset_low));
  std::memcpy(state->shell_offset_class2, cdf::kShellOffsetClass2Cdf,
              sizeof(state->shell_offset_class2));
  std::memcpy(state->shell_offset_other, cdf::kShellOffsetOtherClassCdf,
              sizeof(state->shell_offset_other));
  std::memcpy(state->col_mv_greater, cdf::kColMvGreaterCdf,
              sizeof(state->col_mv_greater));
  std::memcpy(state->col_mv_index, cdf::kColMvIndexCdf,
              sizeof(state->col_mv_index));

  const std::uint16_t* class0[7] = {
      cdf::kJointShell0Class0Cdf, cdf::kJointShell1Class0Cdf,
      nullptr,                    cdf::kJointShell3Class0Cdf,
      cdf::kJointShell4Class0Cdf, cdf::kJointShell5Class0Cdf,
      cdf::kJointShell6Class0Cdf};
  const std::uint16_t* class1[7] = {
      cdf::kJointShell0Class1Cdf, cdf::kJointShell1Class1Cdf,
      nullptr,                    cdf::kJointShell3Class1Cdf,
      cdf::kJointShell4Class1Cdf, cdf::kJointShell5Class1Cdf,
      cdf::kJointShell6Class1Cdf};
  for (int precision = 0; precision <= 6; ++precision) {
    for (int shell_set = 0; shell_set < 2; ++shell_set) {
      const std::uint16_t* source =
          shell_set == 0 ? class0[precision] : class1[precision];
      if (source == nullptr) continue;
      const int n = shell_class_symbol_count(precision, shell_set);
      std::memcpy(state->shell_class[precision][shell_set], source,
                  static_cast<std::size_t>(n + 1) * sizeof(std::uint16_t));
    }
  }
}

// ---------------------------------------------------------------------------
// Neighbour contexts
// ---------------------------------------------------------------------------

NeighborSet collect_neighbors(const NeighborMvInfo* mi, int mi_cols,
                              int mi_rows, int mi_row, int mi_col, int bw4,
                              int bh4, int sb_h4) {
  NeighborSet out;
  if (mi == nullptr || mi_cols <= 0 || mi_rows <= 0) return out;
  int sb_log2 = 0;
  int v = sb_h4 > 0 ? sb_h4 : 16;
  while (v > 1) {
    v >>= 1;
    ++sb_log2;
  }
  const struct {
    int row;
    int col;
  } positions[4] = {
      {mi_row + bh4 - 1, mi_col - 1},
      {mi_row - 1, mi_col + bw4 - 1},
      {mi_row, mi_col - 1},
      {mi_row - 1, mi_col},
  };
  for (const auto& position : positions) {
    if (position.row < 0 || position.row >= mi_rows || position.col < 0 ||
        position.col >= mi_cols) {
      continue;
    }
    const NeighborMvInfo& info =
        mi[static_cast<std::size_t>(position.row) * static_cast<std::size_t>(mi_cols) +
           static_cast<std::size_t>(position.col)];
    if (!info.available) continue;
    const bool same_sb_row = (mi_row >> sb_log2) == (position.row >> sb_log2);
    if (same_sb_row && out.num_npos < 2) {
      out.npos[out.num_npos++] = info;
    }
    if (out.num_nbuf < 2) {
      out.nbuf[out.num_nbuf++] = info;
    }
  }
  return out;
}

int use_amvd_cdf_index(int y_mode, bool use_optflow) {
  switch (y_mode) {
    case kYModeNearNewMv:
      return use_optflow ? 2 : 0;
    case kYModeNewNearMv:
      return use_optflow ? 3 : 1;
    case kYModeNewMv:
      return 4;
    case kYModeJointNewMv:
      return use_optflow ? 6 : 5;
    case kYModeNewNewMv:
      return use_optflow ? 8 : 7;
    default:
      return -1;
  }
}

int use_amvd_ctx(const NeighborSet& neighbours, int ref0) {
  int ctx = 0;
  for (int i = 0; i < neighbours.num_nbuf; ++i) {
    const NeighborMvInfo& n = neighbours.nbuf[i];
    if (n.ref[0] == static_cast<std::int8_t>(ref0) && n.uses_amvd) ++ctx;
  }
  return ctx;
}

int use_most_probable_precision_ctx(const NeighborSet& neighbours) {
  int ctx = 0;
  for (int i = 0; i < neighbours.num_npos; ++i) {
    if (neighbours.npos[i].use_most_probable_precision) ++ctx;
  }
  return ctx;
}

int pb_mv_precision_ctx(const NeighborSet& neighbours, int frame_precision) {
  for (int i = 0; i < neighbours.num_npos; ++i) {
    if (neighbours.npos[i].mv_precision < frame_precision) return 1;
  }
  return 0;
}

int interp_filter_ctx(const NeighborSet& neighbours, int ref0,
                      bool is_compound) {
  int ctx = is_compound ? 4 : 0;
  int left_type = 3;
  int above_type = 3;
  const std::int8_t ref = static_cast<std::int8_t>(ref0);
  if (neighbours.num_npos > 0) {
    const NeighborMvInfo& n = neighbours.npos[0];
    if (n.ref[0] == ref || n.ref[1] == ref) left_type = n.interp_filter;
  }
  if (neighbours.num_npos > 1) {
    const NeighborMvInfo& n = neighbours.npos[1];
    if (n.ref[0] == ref || n.ref[1] == ref) above_type = n.interp_filter;
  }
  if (left_type == above_type) {
    ctx += left_type;
  } else if (left_type == 3) {
    ctx += above_type;
  } else if (above_type == 3) {
    ctx += left_type;
  } else {
    ctx += 3;
  }
  return clip3_int(0, 15, ctx);
}

bool needs_interp_filter(bool large, int y_mode, int motion_mode) {
  if (motion_mode >= kMotionLocalWarp) return false;
  if (large && y_mode == kYModeGlobalMv) return false;
  if (large && y_mode == kYModeGlobalGlobalMv) return false;
  return true;
}

bool block_interp_filter_legal(int filter) {
  return filter >= 0 && filter <= 2;
}

bool frame_interp_filter_legal(int filter, bool is_switchable) {
  if (is_switchable) return filter == kInterpSwitchable;
  return filter >= 0 && filter <= 3;
}

// ---------------------------------------------------------------------------
// Precision plan
// ---------------------------------------------------------------------------

bool validate_mv_res_config(const MvResConfig& config, std::string* error) {
  (void)config;
  (void)error;
  // Every combination of the three sequence flags is legal AV2.  The
  // non-obvious consequences are documented and exercised by the tests:
  //   * enable_adaptive_mvd makes IsAdaptiveMvd force MvPrecision =
  //     FrameMvPrecision for AMVD blocks and suppresses per-block precision
  //     symbols and mv-sign derivation.
  //   * enable_flex_mvres only reaches the block through
  //     UsePerBlockMvPrecision, which force_integer_mv forces to 0.
  //   * enable_mvd_sign_derive is inert unless FrameMvPrecision <= QUARTER
  //     and the block MvPrecision < QUARTER.
  return true;
}

bool plan_block_precision(const MvResConfig& config,
                          const me2::FramePrecisionSyntax& frame,
                          int y_mode, bool use_amvd, int target_precision,
                          BlockPrecisionSyntax* plan, std::string* error) {
  if (plan == nullptr || !frame.valid ||
      !me2::me2_precision_valid(target_precision)) {
    if (error != nullptr) *error = "invalid precision plan input";
    return false;
  }
  BlockPrecisionSyntax out;
  out.is_adaptive_mvd = config.enable_adaptive_mvd && use_amvd;
  if (out.is_adaptive_mvd) {
    out.use_most_probable_precision = true;
    out.mv_precision = frame.frame_precision;
    *plan = out;
    return true;
  }
  if (config.enable_flex_mvres && frame.use_per_block_mv_precision != 0 &&
      has_newmv(y_mode)) {
    bool use_most = target_precision == frame.frame_precision;
    int pb = -1;
    if (!use_most) {
      bool found = false;
      // pb_mv_precision is a three-symbol alphabet (0..2) regardless of the
      // me2 search helper's wider inverse loop.
      for (int candidate = 0; candidate <= 2; ++candidate) {
        if (me2::me2_pb_precision_derive(frame.frame_precision, candidate) ==
            target_precision) {
          pb = candidate;
          found = true;
          break;
        }
      }
      if (!found) {
        if (error != nullptr) {
          *error = "target precision is not signallable for this frame/block";
        }
        return false;
      }
    }
    out.code_use_most_probable_precision = true;
    out.use_most_probable_precision = use_most;
    out.mv_precision = use_most ? frame.frame_precision : target_precision;
    if (!use_most) {
      out.code_pb_mv_precision = true;
      out.pb_mv_precision = pb;
      out.pb_cdf_row = me2::me2_pb_precision_cdf_row(frame.frame_precision);
      if (out.pb_cdf_row < 0 || out.pb_cdf_row > 2) {
        if (error != nullptr) *error = "FrameMvPrecision cannot code pb precision";
        return false;
      }
    }
    *plan = out;
    return true;
  }
  if (target_precision != frame.frame_precision) {
    if (error != nullptr) {
      *error = "per-block precision is not signalled for this block";
    }
    return false;
  }
  out.use_most_probable_precision = true;
  out.mv_precision = frame.frame_precision;
  *plan = out;
  return true;
}

bool write_block_precision(av2::Encoder& enc, MvPrecisionCdfState& cdfs,
                           int most_probable_ctx, int pb_ctx,
                           const BlockPrecisionSyntax& plan,
                           std::string* error) {
  if (most_probable_ctx < 0 || most_probable_ctx > 2 || pb_ctx < 0 ||
      pb_ctx > 1) {
    if (error != nullptr) *error = "precision ctx out of range";
    return false;
  }
  if (plan.code_use_most_probable_precision) {
    if (!sym(enc, &cdfs.use_most_probable_precision[most_probable_ctx][0], 2,
             plan.use_most_probable_precision ? 1 : 0, error)) {
      return false;
    }
  }
  if (plan.code_pb_mv_precision) {
    if (plan.pb_mv_precision < 0 || plan.pb_mv_precision > 2 ||
        plan.pb_cdf_row < 0 || plan.pb_cdf_row > 2) {
      if (error != nullptr) *error = "pb_mv_precision symbol out of range";
      return false;
    }
    if (!sym(enc, &cdfs.pb_mv_precision[pb_ctx][plan.pb_cdf_row][0], 3,
             plan.pb_mv_precision, error)) {
      return false;
    }
  }
  return true;
}

bool read_block_precision(av2::Decoder& dec, MvPrecisionCdfState& cdfs,
                          int most_probable_ctx, int pb_ctx,
                          const me2::FramePrecisionSyntax& frame,
                          int y_mode, bool use_amvd,
                          bool enable_adaptive_mvd, bool enable_flex_mvres,
                          BlockPrecisionSyntax* plan, std::string* error) {
  if (plan == nullptr || !frame.valid) {
    if (error != nullptr) *error = "invalid precision reader input";
    return false;
  }
  if (most_probable_ctx < 0 || most_probable_ctx > 2 || pb_ctx < 0 ||
      pb_ctx > 1) {
    if (error != nullptr) *error = "precision ctx out of range";
    return false;
  }
  BlockPrecisionSyntax out;
  out.is_adaptive_mvd = enable_adaptive_mvd && use_amvd;
  out.use_most_probable_precision = true;
  if (out.is_adaptive_mvd) {
    out.mv_precision = frame.frame_precision;
  } else if (enable_flex_mvres && frame.use_per_block_mv_precision != 0 &&
             has_newmv(y_mode)) {
    int use_most = 0;
    if (!sym(dec, &cdfs.use_most_probable_precision[most_probable_ctx][0], 2,
             &use_most, error)) {
      return false;
    }
    out.code_use_most_probable_precision = true;
    out.use_most_probable_precision = use_most != 0;
    if (use_most != 0) {
      out.mv_precision = frame.frame_precision;
    } else {
      const int row = me2::me2_pb_precision_cdf_row(frame.frame_precision);
      if (row < 0 || row > 2) {
        if (error != nullptr) *error = "FrameMvPrecision cannot code pb precision";
        return false;
      }
      int pb = 0;
      if (!sym(dec, &cdfs.pb_mv_precision[pb_ctx][row][0], 3, &pb, error)) {
        return false;
      }
      out.code_pb_mv_precision = true;
      out.pb_mv_precision = pb;
      out.pb_cdf_row = row;
      out.mv_precision = me2::me2_pb_precision_derive(frame.frame_precision, pb);
    }
  } else {
    out.mv_precision = frame.frame_precision;
  }
  *plan = out;
  return true;
}

// ---------------------------------------------------------------------------
// AMVD symbols
// ---------------------------------------------------------------------------

bool write_use_amvd(av2::Encoder& enc, MvPrecisionCdfState& cdfs, int index,
                    int ctx, bool value, std::string* error) {
  if (index < 0 || index > 8 || ctx < 0 || ctx > 2) {
    if (error != nullptr) *error = "use_amvd index/ctx out of range";
    return false;
  }
  return sym(enc, &cdfs.use_amvd[index][ctx][0], 2, value ? 1 : 0, error);
}

bool read_use_amvd(av2::Decoder& dec, MvPrecisionCdfState& cdfs, int index,
                   int ctx, bool* value, std::string* error) {
  if (value == nullptr || index < 0 || index > 8 || ctx < 0 || ctx > 2) {
    if (error != nullptr) *error = "use_amvd index/ctx out of range";
    return false;
  }
  int symbol = 0;
  if (!sym(dec, &cdfs.use_amvd[index][ctx][0], 2, &symbol, error)) return false;
  *value = symbol != 0;
  return true;
}

bool write_mv_joint(av2::Encoder& enc, MvPrecisionCdfState& cdfs, int joint,
                    std::string* error) {
  if (joint < 0 || joint >= kMvJoints) {
    if (error != nullptr) *error = "mv_joint out of range";
    return false;
  }
  return sym(enc, &cdfs.mv_joint[0], kMvJoints, joint, error);
}

bool read_mv_joint(av2::Decoder& dec, MvPrecisionCdfState& cdfs, int* joint,
                   std::string* error) {
  if (joint == nullptr) {
    if (error != nullptr) *error = "null mv_joint";
    return false;
  }
  return sym(dec, &cdfs.mv_joint[0], kMvJoints, joint, error);
}

bool write_amvd_index(av2::Encoder& enc, MvPrecisionCdfState& cdfs, int comp,
                      int index, std::string* error) {
  if (comp < 0 || comp > 1 || index < 0 || index >= kMaxAmvdIndex) {
    if (error != nullptr) *error = "amvd_index out of range";
    return false;
  }
  return sym(enc, &cdfs.amvd_index[comp][0], kMaxAmvdIndex, index, error);
}

bool read_amvd_index(av2::Decoder& dec, MvPrecisionCdfState& cdfs, int comp,
                     int* index, std::string* error) {
  if (index == nullptr || comp < 0 || comp > 1) {
    if (error != nullptr) *error = "amvd_index out of range";
    return false;
  }
  return sym(dec, &cdfs.amvd_index[comp][0], kMaxAmvdIndex, index, error);
}

bool amvd_index_for_magnitude(std::int32_t value, int* index) {
  if (index == nullptr || value < 0) return false;
  for (int i = 0; i < kMaxAmvdIndex; ++i) {
    if (kAmvdIndexToMvd[i] == value) {
      *index = i;
      return true;
    }
  }
  return false;
}

bool amvd_mv_from_mv(mc::Mv mvd, AmvdMvSyntax* syntax, std::string* error) {
  if (syntax == nullptr) {
    if (error != nullptr) *error = "null syntax";
    return false;
  }
  AmvdMvSyntax out;
  const bool row = mvd.y != 0;
  const bool col = mvd.x != 0;
  out.joint = row ? (col ? kMvJointHnzvnz : kMvJointHzvnz)
                  : (col ? kMvJointHnzvz : kMvJointZero);
  const std::int32_t magnitudes[2] = {mvd.y, mvd.x};
  for (int comp = 0; comp < 2; ++comp) {
    const bool nonzero = comp == 0 ? row : col;
    if (!nonzero) {
      out.index[comp] = 0;
      out.magnitude[comp] = 0;
      continue;
    }
    int index = 0;
    if (!amvd_index_for_magnitude(magnitudes[comp], &index)) {
      if (error != nullptr) *error = "MVD is not representable by AMVD";
      return false;
    }
    out.index[comp] = index;
    out.magnitude[comp] = magnitudes[comp];
  }
  *syntax = out;
  return true;
}

bool write_amvd_mv(av2::Encoder& enc, MvPrecisionCdfState& cdfs,
                   const AmvdMvSyntax& syntax, std::string* error) {
  if (!write_mv_joint(enc, cdfs, syntax.joint, error)) return false;
  const bool row = syntax.joint == kMvJointHzvnz ||
                   syntax.joint == kMvJointHnzvnz;
  const bool col = syntax.joint == kMvJointHnzvz ||
                   syntax.joint == kMvJointHnzvnz;
  for (int comp = 0; comp < 2; ++comp) {
    const bool nonzero = comp == 0 ? row : col;
    if (!nonzero) continue;
    int expected = 0;
    if (!amvd_index_for_magnitude(syntax.magnitude[comp], &expected) ||
        expected != syntax.index[comp]) {
      if (error != nullptr) *error = "AMVD magnitude/index mismatch";
      return false;
    }
    if (!write_amvd_index(enc, cdfs, comp, syntax.index[comp], error)) {
      return false;
    }
  }
  return true;
}

bool read_amvd_mv(av2::Decoder& dec, MvPrecisionCdfState& cdfs,
                  AmvdMvSyntax* syntax, std::string* error) {
  if (syntax == nullptr) {
    if (error != nullptr) *error = "null syntax";
    return false;
  }
  AmvdMvSyntax out;
  if (!read_mv_joint(dec, cdfs, &out.joint, error)) return false;
  const bool row = out.joint == kMvJointHzvnz ||
                   out.joint == kMvJointHnzvnz;
  const bool col = out.joint == kMvJointHnzvz ||
                   out.joint == kMvJointHnzvnz;
  for (int comp = 0; comp < 2; ++comp) {
    const bool nonzero = comp == 0 ? row : col;
    if (!nonzero) continue;
    if (!read_amvd_index(dec, cdfs, comp, &out.index[comp], error)) return false;
    if (out.index[comp] < 0 || out.index[comp] >= kMaxAmvdIndex) {
      if (error != nullptr) *error = "amvd_index out of range";
      return false;
    }
    out.magnitude[comp] = kAmvdIndexToMvd[out.index[comp]];
  }
  *syntax = out;
  return true;
}

bool write_jmvd_scale_mode(av2::Encoder& enc, MvPrecisionCdfState& cdfs,
                           bool use_amvd, int mode, std::string* error) {
  if (use_amvd) {
    if (mode < 0 || mode >= kJointAmvdScaleFactorCnt) {
      if (error != nullptr) *error = "jmvd_scale_mode out of range (AMVD)";
      return false;
    }
    return sym(enc, &cdfs.jmvd_adaptive_scale_mode[0],
               kJointAmvdScaleFactorCnt, mode, error);
  }
  if (mode < 0 || mode >= kJointNewMvScaleFactorCnt) {
    if (error != nullptr) *error = "jmvd_scale_mode out of range";
    return false;
  }
  return sym(enc, &cdfs.jmvd_scale_mode[0], kJointNewMvScaleFactorCnt, mode,
             error);
}

bool read_jmvd_scale_mode(av2::Decoder& dec, MvPrecisionCdfState& cdfs,
                          bool use_amvd, int* mode, std::string* error) {
  if (mode == nullptr) {
    if (error != nullptr) *error = "null mode";
    return false;
  }
  return use_amvd
             ? sym(dec, &cdfs.jmvd_adaptive_scale_mode[0],
                   kJointAmvdScaleFactorCnt, mode, error)
             : sym(dec, &cdfs.jmvd_scale_mode[0],
                   kJointNewMvScaleFactorCnt, mode, error);
}

// ---------------------------------------------------------------------------
// Joint-MVD projection
// ---------------------------------------------------------------------------

JointMvdPlan joint_mvd_plan(int distance0, int distance1, bool restricted0,
                            bool restricted1) {
  JointMvdPlan plan;
  int first = distance0 < 0 ? -distance0 : distance0;
  int second = distance1 < 0 ? -distance1 : distance1;
  if (first < second || (!restricted0 && restricted1)) {
    plan.base_list = 1;
    std::swap(first, second);
  }
  const bool same_side = (distance0 < 0 && distance1 < 0) ||
                         (distance0 > 0 && distance1 > 0);
  if (!same_side) second = -second;
  plan.first_distance = first;
  plan.second_distance = second;
  return plan;
}

mc::Mv project_joint_mvd(mc::Mv diff, int second_distance,
                         int first_distance) {
  const int denominator =
      first_distance < 0 ? 0
                         : (first_distance < kMaxFrameDistance
                                ? first_distance
                                : kMaxFrameDistance);
  const int numerator = clip3_int(-kMaxFrameDistance, kMaxFrameDistance,
                                  second_distance);
  const std::int64_t scale = kDivMult[denominator];
  mc::Mv out;
  out.y = static_cast<std::int32_t>(clip3_int(
      me2::kMvLow + 1, me2::kMvUpp - 1,
      round2_signed_64(static_cast<std::int64_t>(diff.y) * numerator * scale,
                       14)));
  out.x = static_cast<std::int32_t>(clip3_int(
      me2::kMvLow + 1, me2::kMvUpp - 1,
      round2_signed_64(static_cast<std::int64_t>(diff.x) * numerator * scale,
                       14)));
  return out;
}

mc::Mv scale_joint_projection(mc::Mv proj, int jmvd_scale_mode,
                              bool use_amvd) {
  if (use_amvd) {
    if (jmvd_scale_mode == 1) {
      proj.y *= 2;
      proj.x *= 2;
    } else if (jmvd_scale_mode == 2) {
      proj.y /= 2;
      proj.x /= 2;
    }
    return proj;
  }
  if (jmvd_scale_mode > 0) {
    const int comp = (jmvd_scale_mode - 1) & 1;
    std::int32_t* value = comp == 0 ? &proj.y : &proj.x;
    if (jmvd_scale_mode <= 2) {
      *value *= 2;
    } else {
      *value /= 2;
    }
  }
  return proj;
}

std::int32_t mv_clamp_to_integer(std::int32_t value) {
  if (value < me2::kMvLow + 1) return me2::kMvLow + 8;
  if (value > me2::kMvUpp - 1) return me2::kMvUpp - 8;
  return value;
}

void assign_joint_mvd(mc::Mv pred[2], mc::Mv diff, const JointMvdPlan& plan,
                      int jmvd_scale_mode, bool use_amvd, mc::Mv out[2]) {
  mc::Mv proj =
      project_joint_mvd(diff, plan.second_distance, plan.first_distance);
  proj = scale_joint_projection(proj, jmvd_scale_mode, use_amvd);
  out[plan.base_list].y = mv_clamp_to_integer(pred[plan.base_list].y + diff.y);
  out[plan.base_list].x = mv_clamp_to_integer(pred[plan.base_list].x + diff.x);
  const int other = 1 - plan.base_list;
  out[other].y = mv_clamp_to_integer(pred[other].y + proj.y);
  out[other].x = mv_clamp_to_integer(pred[other].x + proj.x);
}

// ---------------------------------------------------------------------------
// mv_sign derivation
// ---------------------------------------------------------------------------

bool mvd_sign_derive_allowed(const MvdSignConfig& config) {
  if (config.use_intrabc || !config.enable_mvd_sign_derive ||
      config.motion_mode != kMotionSimple || config.is_adaptive_mvd ||
      config.skip_mode || config.allow_screen_content_tools ||
      config.frame_precision > me2::kMvPrecisionQuarterPel ||
      config.mv_precision >= me2::kMvPrecisionQuarterPel ||
      has_nearmv(config.y_mode)) {
    return false;
  }
  if (config.is_compound) return config.ref_mv_idx == 0;
  return true;
}

namespace {

struct SignScan {
  int last_ref = -1;
  int last_comp = -1;
  std::int32_t last_sign = 0;
  int num_nonzero = 0;
};

SignScan scan_signs(const mc::Mv mvd[2], int lists, int shift) {
  SignScan scan;
  for (int list = 0; list < lists; ++list) {
    const std::int32_t values[2] = {mvd[list].y, mvd[list].x};
    for (int comp = 0; comp < 2; ++comp) {
      const std::int32_t value = values[comp];
      if (value == 0) continue;
      const std::int32_t magnitude = value < 0 ? -value : value;
      scan.last_ref = list;
      scan.last_comp = comp;
      scan.last_sign += magnitude >> shift;
      ++scan.num_nonzero;
    }
  }
  return scan;
}

int sign_threshold(int y_mode) {
  return y_mode == kYModeNewNewMv ? 4 : 1;
}

}  // namespace

int mvd_sign_symbol_count(const MvdSignConfig& config, const mc::Mv mvd[2],
                          int lists) {
  if (lists != 1 && lists != 2) return -1;
  if (!me2::me2_precision_valid(config.mv_precision)) return -1;
  const int shift = me2::kMvPrecisionEighthPel - config.mv_precision;
  const SignScan scan = scan_signs(mvd, lists, shift);
  const bool allowed = mvd_sign_derive_allowed(config) &&
                       scan.num_nonzero >= sign_threshold(config.y_mode);
  return allowed ? scan.num_nonzero - 1 : scan.num_nonzero;
}

bool write_mvd_signs(av2::Encoder& enc, const MvdSignConfig& config,
                     const mc::Mv mvd[2], int lists, std::string* error) {
  if ((lists != 1 && lists != 2) ||
      !me2::me2_precision_valid(config.mv_precision)) {
    if (error != nullptr) *error = "invalid sign writer input";
    return false;
  }
  const int shift = me2::kMvPrecisionEighthPel - config.mv_precision;
  const SignScan scan = scan_signs(mvd, lists, shift);
  const bool allowed = mvd_sign_derive_allowed(config) &&
                       scan.num_nonzero >= sign_threshold(config.y_mode);
  const bool derived_sign = (scan.last_sign & 1) != 0;
  for (int list = 0; list < lists; ++list) {
    const std::int32_t values[2] = {mvd[list].y, mvd[list].x};
    for (int comp = 0; comp < 2; ++comp) {
      if (values[comp] == 0) continue;
      const bool sign = values[comp] < 0;
      if (allowed && list == scan.last_ref && comp == scan.last_comp) {
        if (sign != derived_sign) {
          if (error != nullptr) {
            *error = "target MVD sign contradicts derived last sign";
          }
          return false;
        }
        continue;
      }
      if (!raw_bit(enc, sign, error)) return false;
    }
  }
  return true;
}

bool read_mvd_signs(av2::Decoder& dec, const MvdSignConfig& config,
                    mc::Mv mvd[2], int lists, std::string* error) {
  if ((lists != 1 && lists != 2) ||
      !me2::me2_precision_valid(config.mv_precision)) {
    if (error != nullptr) *error = "invalid sign reader input";
    return false;
  }
  const int shift = me2::kMvPrecisionEighthPel - config.mv_precision;
  for (int list = 0; list < lists; ++list) {
    mvd[list].y = std::abs(mvd[list].y);
    mvd[list].x = std::abs(mvd[list].x);
  }
  const SignScan scan = scan_signs(mvd, lists, shift);
  const bool allowed = mvd_sign_derive_allowed(config) &&
                       scan.num_nonzero >= sign_threshold(config.y_mode);
  const bool derived_sign = (scan.last_sign & 1) != 0;
  for (int list = 0; list < lists; ++list) {
    std::int32_t values[2] = {mvd[list].y, mvd[list].x};
    for (int comp = 0; comp < 2; ++comp) {
      if (values[comp] == 0) continue;
      bool sign = derived_sign;
      if (!(allowed && list == scan.last_ref && comp == scan.last_comp)) {
        if (!raw_bit(dec, &sign, error)) return false;
      }
      values[comp] = sign ? -values[comp] : values[comp];
    }
    mvd[list].y = values[0];
    mvd[list].x = values[1];
  }
  return true;
}

// ---------------------------------------------------------------------------
// Switchable interp_filter
// ---------------------------------------------------------------------------

bool write_block_interp_filter(av2::Encoder& enc, MvPrecisionCdfState& cdfs,
                               int ctx, int filter, std::string* error) {
  if (ctx < 0 || ctx > 15 || !block_interp_filter_legal(filter)) {
    if (error != nullptr) *error = "illegal block interp_filter/ctx";
    return false;
  }
  return sym(enc, &cdfs.interp_filter[ctx][0], 3, filter, error);
}

bool read_block_interp_filter(av2::Decoder& dec, MvPrecisionCdfState& cdfs,
                              int ctx, int* filter, std::string* error) {
  if (filter == nullptr || ctx < 0 || ctx > 15) {
    if (error != nullptr) *error = "illegal block interp_filter/ctx";
    return false;
  }
  return sym(dec, &cdfs.interp_filter[ctx][0], 3, filter, error);
}

// ---------------------------------------------------------------------------
// Generic shell MVD
// ---------------------------------------------------------------------------

bool write_shell_mv(av2::Encoder& enc, MvPrecisionCdfState& cdfs,
                    std::int32_t dy, std::int32_t dx, int precision,
                    std::string* error) {
  if (!me2::me2_precision_valid(precision)) {
    if (error != nullptr) *error = "invalid precision";
    return false;
  }
  const int shift = me2::kMvPrecisionEighthPel - precision;
  const std::int32_t radix = static_cast<std::int32_t>(1) << shift;
  const std::int32_t ay = dy < 0 ? -dy : dy;
  const std::int32_t ax = dx < 0 ? -dx : dx;
  if ((ay % radix) != 0 || (ax % radix) != 0) {
    if (error != nullptr) *error = "MVD is not aligned to MvPrecision";
    return false;
  }
  return write_shell(enc, cdfs, ay / radix, ax / radix, precision, error);
}

bool read_shell_mv(av2::Decoder& dec, MvPrecisionCdfState& cdfs, int precision,
                   mc::Mv* mvd, std::string* error) {
  if (mvd == nullptr) {
    if (error != nullptr) *error = "null mvd";
    return false;
  }
  std::int32_t y = 0;
  std::int32_t x = 0;
  if (!read_shell(dec, cdfs, precision, &y, &x, error)) return false;
  mvd->y = y;
  mvd->x = x;
  return true;
}

}  // namespace av2enc::inter::mvprec

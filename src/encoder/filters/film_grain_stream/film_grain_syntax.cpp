#include "encoder/filters/film_grain_stream/film_grain_syntax.h"

#include "bitstream/m0_headers.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace av2enc::filters::grain {
namespace {

bool Fail(std::string* error, const char* message) {
  if (error != nullptr) *error = message;
  return false;
}

// §4.11.3 uvlc(): leading zero run, a one bit, then `leading_zeros` payload
// bits.  Values above 2^32-2 are not representable.
void PutUvlc(BitWriter& writer, std::uint32_t value) {
  unsigned leading_zeros = 0;
  while (static_cast<std::uint64_t>(value) >=
         ((1ULL << (leading_zeros + 1U)) - 1ULL)) {
    ++leading_zeros;
  }
  writer.put_bits(0U, leading_zeros);
  writer.put_bit(true);
  const std::uint64_t payload =
      static_cast<std::uint64_t>(value) - ((1ULL << leading_zeros) - 1ULL);
  writer.put_bits(payload, leading_zeros);
}

// §5.2.3 trailing_bits(): one set bit then zeros to the byte boundary.
void PutTrailingBits(BitWriter& writer) {
  writer.put_bit(true);
  writer.align_to_byte();
}

int NumPosLuma(int ar_coeff_lag) { return 2 * ar_coeff_lag * (ar_coeff_lag + 1); }

bool WritePoints(BitWriter& writer, int num_points, const int* values,
                 const int* scalings, int increment_bits_minus_1,
                 int scaling_bits_minus_5) {
  if (num_points <= 0) return true;
  const int increment_bits = increment_bits_minus_1 + 1;
  const int scaling_bits = scaling_bits_minus_5 + 5;
  for (int i = 0; i < num_points; ++i) {
    const int encoded = i == 0 ? values[0] : values[i] - values[i - 1];
    writer.put_bits(static_cast<std::uint64_t>(encoded),
                    static_cast<unsigned>(increment_bits));
    writer.put_bits(static_cast<std::uint64_t>(scalings[i]),
                    static_cast<unsigned>(scaling_bits));
  }
  return true;
}

bool WriteCoefficients(BitWriter& writer, int count, const int* coeffs,
                       int bits_minus_5) {
  const int bits = bits_minus_5 + 5;
  const int bias = 1 << (bits - 1);
  for (int i = 0; i < count; ++i) {
    writer.put_bits(static_cast<std::uint64_t>(coeffs[i] + bias),
                    static_cast<unsigned>(bits));
  }
  return true;
}

bool PointRangeOk(int num_points, const int* values, const int* scalings,
                  int increment_bits_minus_1, int scaling_bits_minus_5,
                  const char* label, std::string* why) {
  if (num_points < 0 || num_points > kMaxGrainPoints) {
    if (why != nullptr) *why = std::string(label) + ": num_points > 14";
    return false;
  }
  if (increment_bits_minus_1 < 0 || increment_bits_minus_1 > 7) {
    if (why != nullptr) *why = std::string(label) + ": increment bits > 8";
    return false;
  }
  if (scaling_bits_minus_5 < 0 || scaling_bits_minus_5 > 3) {
    if (why != nullptr) *why = std::string(label) + ": scaling bits > 8";
    return false;
  }
  if (num_points == 0) return true;

  const int increment_bits = increment_bits_minus_1 + 1;
  const int scaling_bits = scaling_bits_minus_5 + 5;
  const int increment_max = (1 << increment_bits) - 1;
  const int scaling_max = (1 << scaling_bits) - 1;
  if (values[0] < 0 || values[0] > increment_max) {
    if (why != nullptr) *why = std::string(label) + ": first value exceeds bits";
    return false;
  }
  for (int i = 0; i < num_points; ++i) {
    if (values[i] < 0 || values[i] > 255) {
      if (why != nullptr) *why = std::string(label) + ": value outside 0..255";
      return false;
    }
    if (i > 0) {
      if (values[i] <= values[i - 1]) {
        if (why != nullptr) {
          *why = std::string(label) + ": values are not strictly increasing";
        }
        return false;
      }
      if (values[i] - values[i - 1] > increment_max) {
        if (why != nullptr) {
          *why = std::string(label) + ": increment exceeds coded bits";
        }
        return false;
      }
    }
    if (scalings[i] < 0 || scalings[i] > scaling_max) {
      if (why != nullptr) *why = std::string(label) + ": scaling exceeds bits";
      return false;
    }
  }
  return true;
}

bool CoefficientRangeOk(int count, const int* coeffs, int bits_minus_5,
                        const char* label, std::string* why) {
  const int bits = bits_minus_5 + 5;
  const int lo = -(1 << (bits - 1));
  const int hi = (1 << (bits - 1)) - 1;
  for (int i = 0; i < count; ++i) {
    if (coeffs[i] < lo || coeffs[i] > hi) {
      if (why != nullptr) *why = std::string(label) + ": coefficient out of range";
      return false;
    }
  }
  return true;
}

// Reads the trailing_bits() tail and verifies it is exactly one 1 bit followed
// by zeros through the end of the payload.
bool ReadTrailingBits(GrainBitReader& reader, std::string* error,
                      std::size_t* consumed) {
  bool one = false;
  if (!reader.ReadBit(&one)) return Fail(error, "truncated trailing bits");
  if (!one) return Fail(error, "film grain payload has no trailing one bit");
  const std::size_t after_one = reader.bit_position();
  while (reader.bit_position() < reader.bit_size()) {
    bool bit = false;
    if (!reader.ReadBit(&bit)) return Fail(error, "truncated trailing zeros");
    if (bit) return Fail(error, "film grain trailing bits are not zero");
  }
  if (consumed != nullptr) *consumed = after_one;
  return true;
}

}  // namespace

GrainChromaLayout ChromaLayout(GrainChromaIdc idc) {
  // §5.14 p.84-85.
  switch (idc) {
    case GrainChromaIdc::k420:
      return GrainChromaLayout{1, 1, false};
    case GrainChromaIdc::k400:
      return GrainChromaLayout{1, 1, true};
    case GrainChromaIdc::k444:
      return GrainChromaLayout{0, 0, false};
    case GrainChromaIdc::k422:
      return GrainChromaLayout{1, 0, false};
  }
  return GrainChromaLayout{1, 1, false};
}

bool ChromaIdcFromSequence(int chroma_format_idc, GrainChromaIdc* idc) {
  if (chroma_format_idc < 0 || chroma_format_idc > 3 || idc == nullptr) {
    return false;
  }
  *idc = static_cast<GrainChromaIdc>(chroma_format_idc);
  return true;
}

bool operator==(const GrainModel& a, const GrainModel& b) {
  if (a.chroma_scaling_from_luma != b.chroma_scaling_from_luma) return false;
  if (a.num_y_points != b.num_y_points || a.num_cb_points != b.num_cb_points ||
      a.num_cr_points != b.num_cr_points) {
    return false;
  }
  if (a.point_y_increment_bits_minus_1 != b.point_y_increment_bits_minus_1 ||
      a.point_y_scaling_bits_minus_5 != b.point_y_scaling_bits_minus_5 ||
      a.point_cb_increment_bits_minus_1 != b.point_cb_increment_bits_minus_1 ||
      a.point_cb_scaling_bits_minus_5 != b.point_cb_scaling_bits_minus_5 ||
      a.point_cr_increment_bits_minus_1 != b.point_cr_increment_bits_minus_1 ||
      a.point_cr_scaling_bits_minus_5 != b.point_cr_scaling_bits_minus_5) {
    return false;
  }
  for (int i = 0; i < kMaxGrainPoints; ++i) {
    if (a.point_y_value[i] != b.point_y_value[i] ||
        a.point_y_scaling[i] != b.point_y_scaling[i] ||
        a.point_cb_value[i] != b.point_cb_value[i] ||
        a.point_cb_scaling[i] != b.point_cb_scaling[i] ||
        a.point_cr_value[i] != b.point_cr_value[i] ||
        a.point_cr_scaling[i] != b.point_cr_scaling[i]) {
      return false;
    }
  }
  if (a.grain_scaling_minus_8 != b.grain_scaling_minus_8 ||
      a.ar_coeff_lag != b.ar_coeff_lag ||
      a.ar_coeff_shift_minus_6 != b.ar_coeff_shift_minus_6 ||
      a.grain_scale_shift != b.grain_scale_shift ||
      a.bits_per_ar_coeff_y_minus_5 != b.bits_per_ar_coeff_y_minus_5 ||
      a.bits_per_ar_coeff_cb_minus_5 != b.bits_per_ar_coeff_cb_minus_5 ||
      a.bits_per_ar_coeff_cr_minus_5 != b.bits_per_ar_coeff_cr_minus_5) {
    return false;
  }
  for (int i = 0; i < kMaxArPositions; ++i) {
    if (a.ar_coeffs_y[i] != b.ar_coeffs_y[i] ||
        a.ar_coeffs_cb[i] != b.ar_coeffs_cb[i] ||
        a.ar_coeffs_cr[i] != b.ar_coeffs_cr[i]) {
      return false;
    }
  }
  if (a.cb_mult != b.cb_mult || a.cb_luma_mult != b.cb_luma_mult ||
      a.cb_offset != b.cb_offset || a.cr_mult != b.cr_mult ||
      a.cr_luma_mult != b.cr_luma_mult || a.cr_offset != b.cr_offset) {
    return false;
  }
  return a.overlap_flag == b.overlap_flag &&
         a.clip_to_restricted_range == b.clip_to_restricted_range &&
         a.fg_mc_identity == b.fg_mc_identity &&
         a.film_grain_block_size == b.film_grain_block_size;
}

GrainModel NormalizeModel(const GrainModel& model, GrainChromaIdc idc) {
  const GrainChromaLayout layout = ChromaLayout(idc);
  GrainModel m = model;
  if (layout.monochrome) {
    m.chroma_scaling_from_luma = false;
    m.num_cb_points = 0;
    m.num_cr_points = 0;
  }
  if (m.num_y_points <= 0) {
    m.num_y_points = 0;
    m.point_y_increment_bits_minus_1 = 0;
    m.point_y_scaling_bits_minus_5 = 0;
    for (int i = 0; i < kMaxGrainPoints; ++i) {
      m.point_y_value[i] = 0;
      m.point_y_scaling[i] = 0;
    }
    m.bits_per_ar_coeff_y_minus_5 = 0;
    for (int i = 0; i < kMaxArPositions; ++i) m.ar_coeffs_y[i] = 0;
  }
  if (m.chroma_scaling_from_luma) {
    m.num_cb_points = 0;
    m.num_cr_points = 0;
  }
  for (int plane = 1; plane <= 2; ++plane) {
    const int num = plane == 1 ? m.num_cb_points : m.num_cr_points;
    int* values = plane == 1 ? m.point_cb_value : m.point_cr_value;
    int* scalings = plane == 1 ? m.point_cb_scaling : m.point_cr_scaling;
    int& increment_bits = plane == 1 ? m.point_cb_increment_bits_minus_1
                                     : m.point_cr_increment_bits_minus_1;
    int& scaling_bits = plane == 1 ? m.point_cb_scaling_bits_minus_5
                                   : m.point_cr_scaling_bits_minus_5;
    int* coeffs = plane == 1 ? m.ar_coeffs_cb : m.ar_coeffs_cr;
    int& coeff_bits = plane == 1 ? m.bits_per_ar_coeff_cb_minus_5
                                 : m.bits_per_ar_coeff_cr_minus_5;
    if (num <= 0) {
      increment_bits = 0;
      scaling_bits = 0;
      for (int i = 0; i < kMaxGrainPoints; ++i) {
        values[i] = 0;
        scalings[i] = 0;
      }
    }
    // Chroma AR coefficients are present only when chroma_scaling_from_luma
    // or that plane's num_points > 0 (§5.18.10.2).
    if (!m.chroma_scaling_from_luma && num == 0) {
      coeff_bits = 0;
      for (int i = 0; i < kMaxArPositions; ++i) coeffs[i] = 0;
    }
  }
  if (m.num_cb_points <= 0) {
    m.cb_mult = 0;
    m.cb_luma_mult = 0;
    m.cb_offset = 0;
  }
  if (m.num_cr_points <= 0) {
    m.cr_mult = 0;
    m.cr_luma_mult = 0;
    m.cr_offset = 0;
  }
  if (!m.clip_to_restricted_range) m.fg_mc_identity = false;
  return m;
}

bool ValidateModel(const GrainModel& raw, GrainChromaIdc idc, std::string* why) {
  const int raw_idc = static_cast<int>(idc);
  if (raw_idc < 0 || raw_idc > 3) return Fail(why, "fgm_chroma_idc exceeds 3");
  const GrainChromaLayout layout = ChromaLayout(idc);
  const GrainModel m = NormalizeModel(raw, idc);

  if (!PointRangeOk(m.num_y_points, m.point_y_value, m.point_y_scaling,
                    m.point_y_increment_bits_minus_1,
                    m.point_y_scaling_bits_minus_5, "point_y", why)) {
    return false;
  }
  if (!PointRangeOk(m.num_cb_points, m.point_cb_value, m.point_cb_scaling,
                    m.point_cb_increment_bits_minus_1,
                    m.point_cb_scaling_bits_minus_5, "point_cb", why)) {
    return false;
  }
  if (!PointRangeOk(m.num_cr_points, m.point_cr_value, m.point_cr_scaling,
                    m.point_cr_increment_bits_minus_1,
                    m.point_cr_scaling_bits_minus_5, "point_cr", why)) {
    return false;
  }

  // §5.18.10.2: the syntax infers num_cb_points = num_cr_points = 0 when
  // chroma_scaling_from_luma is set, so reject a model that tries to signal
  // both.  Check the raw model, not the canonical one.
  if (raw.chroma_scaling_from_luma &&
      (raw.num_cb_points != 0 || raw.num_cr_points != 0)) {
    return Fail(why, "chroma_scaling_from_luma requires zero chroma points");
  }
  if (m.chroma_scaling_from_luma &&
      (m.num_cb_points != 0 || m.num_cr_points != 0)) {
    return Fail(why, "chroma_scaling_from_luma requires zero chroma points");
  }
  // §6.17.10.2: for 4:2:0 both chroma planes or neither.
  if (layout.sub_x == 1 && layout.sub_y == 1 &&
      ((m.num_cb_points == 0) != (m.num_cr_points == 0))) {
    return Fail(why, "4:2:0 requires both or neither chroma point set");
  }

  if (m.grain_scaling_minus_8 < 0 || m.grain_scaling_minus_8 > 3) {
    return Fail(why, "grain_scaling_minus_8 outside 0..3");
  }
  if (m.ar_coeff_lag < 0 || m.ar_coeff_lag > 3) {
    return Fail(why, "ar_coeff_lag outside 0..3");
  }
  if (m.ar_coeff_shift_minus_6 < 0 || m.ar_coeff_shift_minus_6 > 3) {
    return Fail(why, "ar_coeff_shift_minus_6 outside 0..3");
  }
  if (m.grain_scale_shift < 0 || m.grain_scale_shift > 3) {
    return Fail(why, "grain_scale_shift outside 0..3");
  }
  if (m.bits_per_ar_coeff_y_minus_5 < 0 || m.bits_per_ar_coeff_y_minus_5 > 3 ||
      m.bits_per_ar_coeff_cb_minus_5 < 0 || m.bits_per_ar_coeff_cb_minus_5 > 3 ||
      m.bits_per_ar_coeff_cr_minus_5 < 0 || m.bits_per_ar_coeff_cr_minus_5 > 3) {
    return Fail(why, "bits_per_ar_coeff_*_minus_5 outside 0..3");
  }

  const int num_pos_luma = NumPosLuma(m.ar_coeff_lag);
  const int num_pos_chroma = num_pos_luma + (m.num_y_points > 0 ? 1 : 0);
  if (m.num_y_points > 0 &&
      !CoefficientRangeOk(num_pos_luma, m.ar_coeffs_y,
                          m.bits_per_ar_coeff_y_minus_5, "ar_coeffs_y", why)) {
    return false;
  }
  if ((m.chroma_scaling_from_luma || m.num_cb_points > 0) &&
      !CoefficientRangeOk(num_pos_chroma, m.ar_coeffs_cb,
                          m.bits_per_ar_coeff_cb_minus_5, "ar_coeffs_cb", why)) {
    return false;
  }
  if ((m.chroma_scaling_from_luma || m.num_cr_points > 0) &&
      !CoefficientRangeOk(num_pos_chroma, m.ar_coeffs_cr,
                          m.bits_per_ar_coeff_cr_minus_5, "ar_coeffs_cr", why)) {
    return false;
  }

  if (m.num_cb_points > 0) {
    if (m.cb_mult < 0 || m.cb_mult > 255 || m.cb_luma_mult < 0 ||
        m.cb_luma_mult > 255 || m.cb_offset < 0 || m.cb_offset > 511) {
      return Fail(why, "cb_mult/cb_luma_mult/cb_offset outside signalled range");
    }
  }
  if (m.num_cr_points > 0) {
    if (m.cr_mult < 0 || m.cr_mult > 255 || m.cr_luma_mult < 0 ||
        m.cr_luma_mult > 255 || m.cr_offset < 0 || m.cr_offset > 511) {
      return Fail(why, "cr_mult/cr_luma_mult/cr_offset outside signalled range");
    }
  }
  return true;
}

namespace {

bool WriteModel(BitWriter& writer, const GrainModel& raw, GrainChromaIdc idc,
                std::string* error) {
  if (!ValidateModel(raw, idc, error)) return false;
  const GrainModel m = NormalizeModel(raw, idc);
  const GrainChromaLayout layout = ChromaLayout(idc);

  if (!layout.monochrome) {
    writer.put_bit(m.chroma_scaling_from_luma);  // chroma_scaling_from_luma
  }
  writer.put_bits(static_cast<std::uint64_t>(m.num_y_points), 4U);
  if (m.num_y_points > 0) {
    writer.put_bits(
        static_cast<std::uint64_t>(m.point_y_increment_bits_minus_1), 3U);
    writer.put_bits(
        static_cast<std::uint64_t>(m.point_y_scaling_bits_minus_5), 2U);
    WritePoints(writer, m.num_y_points, m.point_y_value, m.point_y_scaling,
                m.point_y_increment_bits_minus_1,
                m.point_y_scaling_bits_minus_5);
  }

  if (layout.monochrome || m.chroma_scaling_from_luma) {
    // num_cb_points = num_cr_points = 0, no bits.
  } else {
    writer.put_bits(static_cast<std::uint64_t>(m.num_cb_points), 4U);
    if (m.num_cb_points > 0) {
      writer.put_bits(
          static_cast<std::uint64_t>(m.point_cb_increment_bits_minus_1), 3U);
      writer.put_bits(
          static_cast<std::uint64_t>(m.point_cb_scaling_bits_minus_5), 2U);
      WritePoints(writer, m.num_cb_points, m.point_cb_value, m.point_cb_scaling,
                  m.point_cb_increment_bits_minus_1,
                  m.point_cb_scaling_bits_minus_5);
    }
    writer.put_bits(static_cast<std::uint64_t>(m.num_cr_points), 4U);
    if (m.num_cr_points > 0) {
      writer.put_bits(
          static_cast<std::uint64_t>(m.point_cr_increment_bits_minus_1), 3U);
      writer.put_bits(
          static_cast<std::uint64_t>(m.point_cr_scaling_bits_minus_5), 2U);
      WritePoints(writer, m.num_cr_points, m.point_cr_value, m.point_cr_scaling,
                  m.point_cr_increment_bits_minus_1,
                  m.point_cr_scaling_bits_minus_5);
    }
  }

  writer.put_bits(static_cast<std::uint64_t>(m.grain_scaling_minus_8), 2U);
  writer.put_bits(static_cast<std::uint64_t>(m.ar_coeff_lag), 2U);

  const int num_pos_luma = NumPosLuma(m.ar_coeff_lag);
  if (m.num_y_points > 0) {
    writer.put_bits(static_cast<std::uint64_t>(m.bits_per_ar_coeff_y_minus_5),
                    2U);
    WriteCoefficients(writer, num_pos_luma, m.ar_coeffs_y,
                      m.bits_per_ar_coeff_y_minus_5);
  }
  const int num_pos_chroma = num_pos_luma + (m.num_y_points > 0 ? 1 : 0);
  if (m.chroma_scaling_from_luma || m.num_cb_points > 0) {
    writer.put_bits(static_cast<std::uint64_t>(m.bits_per_ar_coeff_cb_minus_5),
                    2U);
    WriteCoefficients(writer, num_pos_chroma, m.ar_coeffs_cb,
                      m.bits_per_ar_coeff_cb_minus_5);
  }
  if (m.chroma_scaling_from_luma || m.num_cr_points > 0) {
    writer.put_bits(static_cast<std::uint64_t>(m.bits_per_ar_coeff_cr_minus_5),
                    2U);
    WriteCoefficients(writer, num_pos_chroma, m.ar_coeffs_cr,
                      m.bits_per_ar_coeff_cr_minus_5);
  }

  writer.put_bits(static_cast<std::uint64_t>(m.ar_coeff_shift_minus_6), 2U);
  writer.put_bits(static_cast<std::uint64_t>(m.grain_scale_shift), 2U);

  if (m.num_cb_points > 0) {
    writer.put_bits(static_cast<std::uint64_t>(m.cb_mult), 8U);
    writer.put_bits(static_cast<std::uint64_t>(m.cb_luma_mult), 8U);
    writer.put_bits(static_cast<std::uint64_t>(m.cb_offset), 9U);
  }
  if (m.num_cr_points > 0) {
    writer.put_bits(static_cast<std::uint64_t>(m.cr_mult), 8U);
    writer.put_bits(static_cast<std::uint64_t>(m.cr_luma_mult), 8U);
    writer.put_bits(static_cast<std::uint64_t>(m.cr_offset), 9U);
  }

  writer.put_bit(m.overlap_flag);
  writer.put_bit(m.clip_to_restricted_range);
  if (m.clip_to_restricted_range) writer.put_bit(m.fg_mc_identity);
  writer.put_bit(m.film_grain_block_size);
  return true;
}

bool ReadPoints(GrainBitReader& reader, int* num_points, int* values,
                int* scalings, int* increment_bits_minus_1,
                int* scaling_bits_minus_5, std::string* error) {
  std::uint64_t num = 0;
  if (!reader.ReadBits(4U, &num)) return Fail(error, "truncated num_points");
  if (num > static_cast<std::uint64_t>(kMaxGrainPoints)) {
    return Fail(error, "num_points exceeds 14");
  }
  *num_points = static_cast<int>(num);
  // Absent bit-width fields have no meaning; NormalizeModel() zeroes them.
  *increment_bits_minus_1 = 0;
  *scaling_bits_minus_5 = 0;
  if (*num_points == 0) return true;

  std::uint64_t increment = 0;
  std::uint64_t scaling = 0;
  if (!reader.ReadBits(3U, &increment) || !reader.ReadBits(2U, &scaling)) {
    return Fail(error, "truncated point bit widths");
  }
  *increment_bits_minus_1 = static_cast<int>(increment);
  *scaling_bits_minus_5 = static_cast<int>(scaling);

  const unsigned increment_bits = static_cast<unsigned>(increment) + 1U;
  const unsigned scaling_bits = static_cast<unsigned>(scaling) + 5U;
  int previous = 0;
  for (int i = 0; i < *num_points; ++i) {
    std::uint64_t value = 0;
    std::uint64_t scale = 0;
    if (!reader.ReadBits(increment_bits, &value) ||
        !reader.ReadBits(scaling_bits, &scale)) {
      return Fail(error, "truncated point curve");
    }
    const std::uint64_t cumulative = value + static_cast<std::uint64_t>(previous);
    if (cumulative > 255) return Fail(error, "point value outside 0..255");
    if (i > 0 && cumulative <= static_cast<std::uint64_t>(previous)) {
      return Fail(error, "point values are not strictly increasing");
    }
    values[i] = static_cast<int>(cumulative);
    scalings[i] = static_cast<int>(scale);
    previous = values[i];
  }
  return true;
}

bool ReadCoefficients(GrainBitReader& reader, int count, int* coeffs,
                      int* bits_minus_5, std::string* error) {
  std::uint64_t bits = 0;
  if (!reader.ReadBits(2U, &bits)) return Fail(error, "truncated AR bit width");
  *bits_minus_5 = static_cast<int>(bits);
  const unsigned width = static_cast<unsigned>(bits) + 5U;
  const int bias = 1 << (width - 1U);
  for (int i = 0; i < count; ++i) {
    std::uint64_t raw = 0;
    if (!reader.ReadBits(width, &raw)) return Fail(error, "truncated AR coeffs");
    coeffs[i] = static_cast<int>(raw) - bias;
  }
  return true;
}

bool ReadModel(GrainBitReader& reader, GrainChromaIdc idc, GrainModel* model,
               std::string* error) {
  const GrainChromaLayout layout = ChromaLayout(idc);
  GrainModel m;
  if (!layout.monochrome) {
    bool from_luma = false;
    if (!reader.ReadBit(&from_luma)) return Fail(error, "truncated chroma_scaling");
    m.chroma_scaling_from_luma = from_luma;
  }

  if (!ReadPoints(reader, &m.num_y_points, m.point_y_value, m.point_y_scaling,
                  &m.point_y_increment_bits_minus_1,
                  &m.point_y_scaling_bits_minus_5, error)) {
    return false;
  }
  if (!layout.monochrome && !m.chroma_scaling_from_luma) {
    if (!ReadPoints(reader, &m.num_cb_points, m.point_cb_value,
                    m.point_cb_scaling, &m.point_cb_increment_bits_minus_1,
                    &m.point_cb_scaling_bits_minus_5, error)) {
      return false;
    }
    if (!ReadPoints(reader, &m.num_cr_points, m.point_cr_value,
                    m.point_cr_scaling, &m.point_cr_increment_bits_minus_1,
                    &m.point_cr_scaling_bits_minus_5, error)) {
      return false;
    }
  }

  std::uint64_t value = 0;
  if (!reader.ReadBits(2U, &value)) return Fail(error, "truncated grain_scaling");
  m.grain_scaling_minus_8 = static_cast<int>(value);
  if (!reader.ReadBits(2U, &value)) return Fail(error, "truncated ar_coeff_lag");
  m.ar_coeff_lag = static_cast<int>(value);

  const int num_pos_luma = NumPosLuma(m.ar_coeff_lag);
  if (m.num_y_points > 0) {
    if (!ReadCoefficients(reader, num_pos_luma, m.ar_coeffs_y,
                          &m.bits_per_ar_coeff_y_minus_5, error)) {
      return false;
    }
  }
  const int num_pos_chroma = num_pos_luma + (m.num_y_points > 0 ? 1 : 0);
  if (m.chroma_scaling_from_luma || m.num_cb_points > 0) {
    if (!ReadCoefficients(reader, num_pos_chroma, m.ar_coeffs_cb,
                          &m.bits_per_ar_coeff_cb_minus_5, error)) {
      return false;
    }
  }
  if (m.chroma_scaling_from_luma || m.num_cr_points > 0) {
    if (!ReadCoefficients(reader, num_pos_chroma, m.ar_coeffs_cr,
                          &m.bits_per_ar_coeff_cr_minus_5, error)) {
      return false;
    }
  }

  if (!reader.ReadBits(2U, &value)) return Fail(error, "truncated ar_coeff_shift");
  m.ar_coeff_shift_minus_6 = static_cast<int>(value);
  if (!reader.ReadBits(2U, &value)) return Fail(error, "truncated grain_scale_shift");
  m.grain_scale_shift = static_cast<int>(value);

  if (m.num_cb_points > 0) {
    std::uint64_t mult = 0;
    std::uint64_t luma_mult = 0;
    std::uint64_t offset = 0;
    if (!reader.ReadBits(8U, &mult) || !reader.ReadBits(8U, &luma_mult) ||
        !reader.ReadBits(9U, &offset)) {
      return Fail(error, "truncated cb_* coefficients");
    }
    m.cb_mult = static_cast<int>(mult);
    m.cb_luma_mult = static_cast<int>(luma_mult);
    m.cb_offset = static_cast<int>(offset);
  }
  if (m.num_cr_points > 0) {
    std::uint64_t mult = 0;
    std::uint64_t luma_mult = 0;
    std::uint64_t offset = 0;
    if (!reader.ReadBits(8U, &mult) || !reader.ReadBits(8U, &luma_mult) ||
        !reader.ReadBits(9U, &offset)) {
      return Fail(error, "truncated cr_* coefficients");
    }
    m.cr_mult = static_cast<int>(mult);
    m.cr_luma_mult = static_cast<int>(luma_mult);
    m.cr_offset = static_cast<int>(offset);
  }

  bool overlap = false;
  bool clip = false;
  if (!reader.ReadBit(&overlap) || !reader.ReadBit(&clip)) {
    return Fail(error, "truncated overlap/clip flags");
  }
  m.overlap_flag = overlap;
  m.clip_to_restricted_range = clip;
  if (clip) {
    bool identity = false;
    if (!reader.ReadBit(&identity)) return Fail(error, "truncated fg_mc_identity");
    m.fg_mc_identity = identity;
  }
  bool block = false;
  if (!reader.ReadBit(&block)) return Fail(error, "truncated block size flag");
  m.film_grain_block_size = block;

  *model = NormalizeModel(m, idc);
  return true;
}

}  // namespace

bool ValidateObu(const FilmGrainObu& obu, std::string* why) {
  // §6.13: fgm_update_flags != 0.
  if (obu.update_flags == 0) return Fail(why, "fgm_update_flags is zero");
  const int idc = static_cast<int>(obu.chroma_idc);
  if (idc < 0 || idc > 3) return Fail(why, "fgm_chroma_idc exceeds 3");
  for (int slot = 0; slot < kNumGrainSlots; ++slot) {
    if ((obu.update_flags & (1U << slot)) == 0) continue;
    if (!ValidateModel(obu.models[slot], obu.chroma_idc, why)) return false;
  }
  return true;
}

bool WriteObuPayload(BitWriter& writer, const FilmGrainObu& obu,
                     std::string* error) {
  if (!ValidateObu(obu, error)) return false;
  writer.put_bits(static_cast<std::uint64_t>(obu.update_flags), 8U);
  PutUvlc(writer, static_cast<std::uint32_t>(obu.chroma_idc));
  for (int slot = 0; slot < kNumGrainSlots; ++slot) {
    if ((obu.update_flags & (1U << slot)) == 0) continue;
    const GrainModel m =
        NormalizeModel(obu.models[slot], obu.chroma_idc);
    if (!WriteModel(writer, m, obu.chroma_idc, error)) return false;
  }
  PutTrailingBits(writer);
  return true;
}

bool ReadObuPayload(std::span<const std::uint8_t> payload, FilmGrainObu* obu,
                    std::size_t* consumed_bits, std::string* error) {
  if (obu == nullptr) return Fail(error, "null film grain OBU output");
  if (payload.empty()) return Fail(error, "empty film grain OBU payload");
  GrainBitReader reader(payload);
  FilmGrainObu out;
  std::uint64_t flags = 0;
  if (!reader.ReadBits(8U, &flags)) return Fail(error, "truncated fgm_update_flags");
  out.update_flags = static_cast<std::uint8_t>(flags);
  if (out.update_flags == 0) {
    return Fail(error, "fgm_update_flags == 0 violates §6.13");
  }
  std::uint32_t chroma_idc = 0;
  if (!reader.ReadUvlc(&chroma_idc)) return Fail(error, "truncated fgm_chroma_idc");
  if (chroma_idc > 3) {
    return Fail(error, "fgm_chroma_idc > 3 violates §6.13");
  }
  out.chroma_idc = static_cast<GrainChromaIdc>(chroma_idc);
  for (int slot = 0; slot < kNumGrainSlots; ++slot) {
    if ((out.update_flags & (1U << slot)) == 0) continue;
    if (!ReadModel(reader, out.chroma_idc, &out.models[slot], error)) {
      return false;
    }
  }
  std::size_t consumed = 0;
  if (!ReadTrailingBits(reader, error, &consumed)) return false;
  *obu = out;
  if (consumed_bits != nullptr) *consumed_bits = consumed;
  return true;
}

std::vector<std::uint8_t> MakeFilmGrainObu(const FilmGrainObu& obu,
                                           std::string* error) {
  BitWriter writer;
  if (!WriteObuPayload(writer, obu, error)) return {};
  const std::span<const std::uint8_t> payload = writer.bytes();
  return av2enc::m0::make_obu(
      av2enc::m0::ObuHeader{
          static_cast<av2enc::m0::ObuType>(kFilmGrainObuType)},
      payload);
}

bool ReadFilmGrainObu(std::span<const std::uint8_t> data, FilmGrainObu* obu,
                      std::size_t* consumed, std::string* error) {
  if (data.empty()) return Fail(error, "empty OBU");
  // Annex B.2 leb128 payload size.
  std::size_t pos = 0;
  std::uint64_t obu_size = 0;
  int shift = 0;
  while (true) {
    if (pos >= data.size()) return Fail(error, "truncated leb128");
    const std::uint8_t byte = data[pos++];
    obu_size |= static_cast<std::uint64_t>(byte & 0x7FU) << shift;
    if ((byte & 0x80U) == 0) break;
    shift += 7;
    if (shift > 28) return Fail(error, "leb128 value too large");
  }
  if (obu_size == 0 || pos + obu_size > data.size()) {
    return Fail(error, "truncated OBU payload");
  }
  const std::uint8_t header = data[pos];
  const bool extension_flag = (header & 0x80U) != 0;
  const std::uint8_t type = static_cast<std::uint8_t>((header >> 2U) & 0x1FU);
  if (type != kFilmGrainObuType) {
    return Fail(error, "OBU is not OBU_FILM_GRAIN");
  }
  std::size_t header_bytes = 1;
  if (extension_flag) header_bytes = 2;
  if (obu_size < header_bytes) return Fail(error, "OBU smaller than its header");
  const std::span<const std::uint8_t> payload =
      data.subspan(pos + header_bytes, static_cast<std::size_t>(obu_size) -
                                           header_bytes);
  if (!ReadObuPayload(payload, obu, nullptr, error)) return false;
  if (consumed != nullptr) *consumed = pos + static_cast<std::size_t>(obu_size);
  return true;
}

bool ApplyGrainIsInferred(const FilmGrainConfigContext& ctx, bool* value) {
  if (!ctx.film_grain_params_present ||
      (!ctx.immediate_output_frame && !ctx.implicit_output_frame)) {
    if (value != nullptr) *value = false;
    return true;
  }
  if (ctx.single_picture_header_flag) {
    if (value != nullptr) *value = true;
    return true;
  }
  return false;
}

bool WriteFilmGrainConfig(BitWriter& writer, const FilmGrainConfig& config,
                          const FilmGrainConfigContext& ctx,
                          std::string* error) {
  bool inferred_value = false;
  const bool inferred = ApplyGrainIsInferred(ctx, &inferred_value);
  if (inferred && config.apply_grain != inferred_value) {
    return Fail(error, "apply_grain contradicts the §5.18.10.1 inference");
  }
  if (!inferred) writer.put_bit(config.apply_grain);
  if (!config.apply_grain) return true;

  if (config.fgm_id < 0 || config.fgm_id >= kNumGrainSlots) {
    return Fail(error, "fgm_id outside 0..7");
  }
  writer.put_bits(static_cast<std::uint64_t>(config.fgm_id), 3U);
  writer.put_bits(static_cast<std::uint64_t>(config.grain_seed), 16U);
  return true;
}

bool ReadFilmGrainConfig(std::span<const std::uint8_t> payload,
                         std::size_t* bit_position,
                         const FilmGrainConfigContext& ctx,
                         FilmGrainConfig* config, std::string* error) {
  if (bit_position == nullptr || config == nullptr) {
    return Fail(error, "null film grain config state");
  }
  GrainBitReader reader(payload, *bit_position);
  FilmGrainConfig out;

  bool inferred_value = false;
  if (ApplyGrainIsInferred(ctx, &inferred_value)) {
    out.apply_grain = inferred_value;
  } else {
    bool apply = false;
    if (!reader.ReadBit(&apply)) return Fail(error, "truncated apply_grain");
    out.apply_grain = apply;
  }
  if (out.apply_grain) {
    std::uint64_t id = 0;
    std::uint64_t seed = 0;
    if (!reader.ReadBits(3U, &id)) return Fail(error, "truncated fgm_id");
    if (!reader.ReadBits(16U, &seed)) return Fail(error, "truncated grain_seed");
    out.fgm_id = static_cast<int>(id);
    out.grain_seed = static_cast<std::uint16_t>(seed);
  }
  *bit_position = reader.bit_position();
  *config = out;
  return true;
}

std::size_t FilmGrainConfigBits(const FilmGrainConfig& config,
                                const FilmGrainConfigContext& ctx) {
  BitWriter writer;
  if (!WriteFilmGrainConfig(writer, config, ctx, nullptr)) return 0;
  return writer.bit_size();
}

std::size_t ModelBits(const GrainModel& model, GrainChromaIdc idc) {
  BitWriter writer;
  if (!WriteModel(writer, model, idc, nullptr)) return 0;
  return writer.bit_size();
}

std::size_t FilmGrainObuBytes(const FilmGrainObu& obu) {
  return MakeFilmGrainObu(obu, nullptr).size();
}

void GrainSlotState::Reset() {
  for (int i = 0; i < kNumGrainSlots; ++i) slots_[i] = GrainSlot{};
}

bool GrainSlotState::SaveModel(int slot, const GrainModel& model,
                               GrainChromaIdc idc, int tlayer_id,
                               int mlayer_id, std::string* error) {
  if (slot < 0 || slot >= kNumGrainSlots) {
    return Fail(error, "save_grain_model: slot outside 0..7");
  }
  if (!ValidateModel(model, idc, error)) return false;
  if (tlayer_id < 0 || tlayer_id > 3 || mlayer_id < 0 || mlayer_id > 7) {
    return Fail(error, "save_grain_model: layer id outside range");
  }
  slots_[slot].present = true;
  slots_[slot].model = NormalizeModel(model, idc);
  slots_[slot].chroma_idc = idc;
  slots_[slot].tlayer_id = tlayer_id;
  slots_[slot].mlayer_id = mlayer_id;
  return true;
}

bool GrainSlotState::LoadModel(int slot, GrainModel* model, GrainChromaIdc* idc,
                               std::string* error) const {
  if (slot < 0 || slot >= kNumGrainSlots) {
    return Fail(error, "load_grain_model: slot outside 0..7");
  }
  if (!slots_[slot].present) {
    return Fail(error, "load_grain_model: FilmGrainPresent[slot] == 0");
  }
  if (model != nullptr) *model = slots_[slot].model;
  if (idc != nullptr) *idc = slots_[slot].chroma_idc;
  return true;
}

bool ValidateConfigAvailability(const GrainSlotState& state,
                                const FilmGrainConfig& config,
                                GrainChromaIdc frame_chroma_idc,
                                int tlayer_id, int mlayer_id,
                                std::string* why) {
  if (!config.apply_grain) return true;
  if (config.fgm_id < 0 || config.fgm_id >= kNumGrainSlots) {
    return Fail(why, "fgm_id outside 0..7");
  }
  const GrainSlot& slot = state.slot(config.fgm_id);
  if (!slot.present) {
    return Fail(why, "FilmGrainPresent[fgm_id] == 0 (§7.3.8.8)");
  }
  if (slot.chroma_idc != frame_chroma_idc) {
    return Fail(why, "FgmChromaIdc[fgm_id] != chroma_format_idc");
  }
  if (slot.tlayer_id > tlayer_id || slot.mlayer_id > mlayer_id) {
    return Fail(why, "layer dependency map rejects the grain model");
  }
  return true;
}

bool ToSynthesisModel(const GrainModel& model, spec::FilmGrainModel* out) {
  if (out == nullptr) return false;
  spec::FilmGrainModel m;
  m.chroma_scaling_from_luma = model.chroma_scaling_from_luma;
  m.num_y_points = model.num_y_points;
  for (int i = 0; i < kMaxGrainPoints; ++i) {
    m.point_y_value[i] = model.point_y_value[i];
    m.point_y_scaling[i] = model.point_y_scaling[i];
  }
  m.num_cb_points = model.num_cb_points;
  for (int i = 0; i < kMaxGrainPoints; ++i) {
    m.point_cb_value[i] = model.point_cb_value[i];
    m.point_cb_scaling[i] = model.point_cb_scaling[i];
  }
  m.num_cr_points = model.num_cr_points;
  for (int i = 0; i < kMaxGrainPoints; ++i) {
    m.point_cr_value[i] = model.point_cr_value[i];
    m.point_cr_scaling[i] = model.point_cr_scaling[i];
  }
  m.grain_scaling_minus_8 = model.grain_scaling_minus_8;
  m.ar_coeff_lag = model.ar_coeff_lag;
  for (int i = 0; i < kMaxArPositions; ++i) {
    m.ar_coeffs_y[i] = model.ar_coeffs_y[i];
    m.ar_coeffs_cb[i] = model.ar_coeffs_cb[i];
    m.ar_coeffs_cr[i] = model.ar_coeffs_cr[i];
  }
  m.ar_coeff_shift_minus_6 = model.ar_coeff_shift_minus_6;
  m.grain_scale_shift = model.grain_scale_shift;
  m.cb_mult = model.cb_mult;
  m.cb_luma_mult = model.cb_luma_mult;
  m.cb_offset = model.cb_offset;
  m.cr_mult = model.cr_mult;
  m.cr_luma_mult = model.cr_luma_mult;
  m.cr_offset = model.cr_offset;
  m.overlap_flag = model.overlap_flag;
  m.clip_to_restricted_range = model.clip_to_restricted_range;
  m.fg_mc_identity = model.fg_mc_identity;
  m.film_grain_block_size = model.film_grain_block_size;
  *out = m;
  return true;
}

bool GrainBitReader::ReadBit(bool* value) {
  if (bit_pos_ >= bit_size()) return false;
  const std::size_t byte = bit_pos_ >> 3U;
  const unsigned shift = 7U - static_cast<unsigned>(bit_pos_ & 7U);
  if (value != nullptr) {
    *value = ((data_[byte] >> shift) & 1U) != 0U;
  }
  ++bit_pos_;
  return true;
}

bool GrainBitReader::ReadBits(unsigned count, std::uint64_t* value) {
  if (count > 32U) return false;
  std::uint64_t result = 0;
  for (unsigned i = 0; i < count; ++i) {
    bool bit = false;
    if (!ReadBit(&bit)) return false;
    result = (result << 1U) | (bit ? 1U : 0U);
  }
  if (value != nullptr) *value = result;
  return true;
}

bool GrainBitReader::ReadUvlc(std::uint32_t* value) {
  unsigned leading_zeros = 0;
  while (true) {
    bool one = false;
    if (!ReadBit(&one)) return false;
    if (one) break;
    ++leading_zeros;
    if (leading_zeros >= 32U) return false;
  }
  std::uint64_t payload = 0;
  if (!ReadBits(leading_zeros, &payload)) return false;
  const std::uint64_t result = payload + (1ULL << leading_zeros) - 1ULL;
  if (result > std::numeric_limits<std::uint32_t>::max()) return false;
  if (value != nullptr) *value = static_cast<std::uint32_t>(result);
  return true;
}

bool GrainBitReader::ByteAlign() {
  while ((bit_pos_ & 7U) != 0) {
    bool bit = false;
    if (!ReadBit(&bit)) return false;
    if (bit) return false;
  }
  return true;
}

}  // namespace av2enc::filters::grain

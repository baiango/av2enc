#include "codec/transform/qm/obu/qm_obu.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

namespace av2enc::qm::obu {
namespace {

bool fail(std::string* error, std::string message) {
  if (error != nullptr) *error = std::move(message);
  return false;
}

std::string level_prefix(int level) {
  return "quantizer matrix level " + std::to_string(level) + ": ";
}

// ---------------------------------------------------------------------------
// MSB-first reader with exact bit accounting.
// ---------------------------------------------------------------------------
class BitSpanReader {
 public:
  explicit BitSpanReader(std::span<const std::uint8_t> data) : data_(data) {}

  bool read_bit(bool* value) {
    if (position_ >= data_.size() * 8U) return false;
    const std::uint8_t byte = data_[position_ >> 3U];
    *value = ((byte >> (7U - (position_ & 7U))) & 1U) != 0U;
    ++position_;
    return true;
  }

  bool read_bits(unsigned count, std::uint32_t* value) {
    std::uint32_t out = 0;
    for (unsigned i = 0; i < count; ++i) {
      bool bit = false;
      if (!read_bit(&bit)) return false;
      out = (out << 1U) | (bit ? 1U : 0U);
    }
    *value = out;
    return true;
  }

  bool read_uvlc(std::uint32_t* value) {
    unsigned leading_zeros = 0;
    bool bit = false;
    do {
      if (!read_bit(&bit)) return false;
      if (bit) break;
      ++leading_zeros;
      if (leading_zeros > 31U) return false;
    } while (true);
    if (leading_zeros == 0) {
      *value = 0;
      return true;
    }
    std::uint32_t payload = 0;
    if (!read_bits(leading_zeros, &payload)) return false;
    *value = ((1U << leading_zeros) - 1U) + payload;
    return true;
  }

  bool read_svlc(int* value) {
    std::uint32_t code = 0;
    if (!read_uvlc(&code)) return false;
    if (code == 0) {
      *value = 0;
    } else if ((code & 1U) != 0U) {
      *value = static_cast<int>((code + 1U) / 2U);
    } else {
      *value = -static_cast<int>(code / 2U);
    }
    return *value >= -128 && *value <= 127;
  }

  // §5.2.3 trailing_bits(): one trailing one bit, then zero bits to the end of
  // the payload.
  bool read_trailing_bits(std::string* error) {
    bool one = false;
    if (!read_bit(&one) || !one) {
      return fail(error, "missing trailing_one_bit");
    }
    while (position_ < data_.size() * 8U) {
      bool zero = false;
      if (!read_bit(&zero) || zero) {
        return fail(error, "non-zero trailing bit");
      }
    }
    return true;
  }

  std::size_t position() const { return position_; }

 private:
  std::span<const std::uint8_t> data_;
  std::size_t position_ = 0;
};

void put_uvlc(BitWriter& writer, std::uint32_t value) {
  unsigned leading_zeros = 0;
  while (value >= ((1ULL << (leading_zeros + 1U)) - 1ULL)) {
    ++leading_zeros;
  }
  writer.put_bits(0U, leading_zeros);
  writer.put_bit(true);
  const std::uint64_t payload =
      static_cast<std::uint64_t>(value) - ((1ULL << leading_zeros) - 1ULL);
  writer.put_bits(payload, leading_zeros);
}

// §4.11.4 svlc(); delta is validated to [-128, 127] by the caller.
void put_svlc(BitWriter& writer, int delta) {
  std::uint32_t value = 0;
  if (delta > 0) {
    value = static_cast<std::uint32_t>(2 * delta - 1);
  } else if (delta < 0) {
    value = static_cast<std::uint32_t>(-2 * delta);
  }
  put_uvlc(writer, value);
}

// §5.20.7.30 get_scan(txSz, TX_CLASS_2D) for the three fundamental sizes.
// The diagonal scan index maps c -> pos = row * width + col.
void diagonal_scan(int w, int h, std::uint16_t* scan) {
  int x = 0;
  int y = 0;
  for (int c = 0; c < w * h; ++c) {
    scan[c] = static_cast<std::uint16_t>(y * w + x);
    ++x;
    --y;
    if (y < 0 || x >= w) {
      ++x;
      const int s = std::min(x, h - 1 - y);
      x -= s;
      y += s;
    }
  }
}

void fundamental_dims(int t, int* w, int* h) {
  // Fundamental_Tx_Size[3] = { TX_8X8, TX_8X4, TX_4X8 }.
  if (t == 0) {
    *w = 8;
    *h = 8;
  } else if (t == 1) {
    *w = 8;
    *h = 4;
  } else {
    *w = 4;
    *h = 8;
  }
}

bool matrix_is_symmetric(const std::uint8_t m[8][8]) {
  for (int i = 0; i < 8; ++i) {
    for (int j = i + 1; j < 8; ++j) {
      if (m[i][j] != m[j][i]) return false;
    }
  }
  return true;
}

// t = 2 (4x8) is the transpose of t = 1 (8x4): a[i][j] == b[j][i].
bool matrix_is_transpose(const std::uint8_t a[8][8],
                         const std::uint8_t b[8][8]) {
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 4; ++j) {
      if (a[i][j] != b[j][i]) return false;
    }
  }
  return true;
}

bool write_user_defined_qm(BitWriter& writer, const UserQmLevel& user,
                           int level, int t, int plane, std::string* error) {
  auto set_error = [&](const char* message) {
    return fail(error, level_prefix(level) + "t=" + std::to_string(t) +
                           " plane=" + std::to_string(plane) + ": " +
                           message);
  };

  int w = 0;
  int h = 0;
  fundamental_dims(t, &w, &h);

  if (plane > 0) {
    const bool copy =
        std::memcmp(user.m[t][plane], user.m[t][plane - 1],
                    sizeof(user.m[t][plane])) == 0;
    writer.put_bit(copy);
    if (copy) return true;
  }

  bool symmetric = false;
  if (t == 0) {
    symmetric = matrix_is_symmetric(user.m[0][plane]);
    writer.put_bit(symmetric);
  } else if (t == 2) {
    const bool transpose = matrix_is_transpose(user.m[2][plane],
                                               user.m[1][plane]);
    writer.put_bit(transpose);
    if (transpose) return true;
  }

  std::uint16_t scan[8 * 8];
  diagonal_scan(w, h, scan);
  int quant = 32;
  bool coef_repeat = false;
  for (int c = 0; c < w * h; ++c) {
    const int pos = scan[c];
    const int row = pos / w;
    const int col = pos % w;
    const std::uint8_t value = user.m[t][plane][row][col];
    if (t == 0 && symmetric && col > row) {
      // Inferred from the already-coded transposed entry; no delta is coded.
      const std::uint8_t inferred = user.m[t][plane][col][row];
      if (value != inferred) {
        return set_error("asymmetric inferred entry");
      }
      quant = inferred;
      continue;
    }
    if (coef_repeat) {
      if (value != quant) {
        return set_error("coefficient after a repeat marker differs");
      }
      continue;
    }
    if (value == 0) {
      return set_error("coefficient 0 is forbidden by §6.4.11");
    }
    int delta = (static_cast<int>(value) - quant) & 255;
    if (delta > 127) delta -= 256;
    if (delta < -128 || delta > 127) {
      return set_error("quant_delta is outside [-128, 127]");
    }
    put_svlc(writer, delta);
    const int quant2 = (quant + delta) & 255;
    if (quant2 == 0) {
      coef_repeat = true;
    } else {
      quant = quant2;
    }
    if (quant != value) return set_error("svlc round-trip mismatch");
  }
  return true;
}

bool read_user_defined_qm(BitSpanReader& reader, UserQmLevel* user, int level,
                          int t, int plane, std::string* error) {
  auto set_error = [&](const char* message) {
    return fail(error, level_prefix(level) + "t=" + std::to_string(t) +
                           " plane=" + std::to_string(plane) + ": " +
                           message);
  };

  int w = 0;
  int h = 0;
  fundamental_dims(t, &w, &h);

  if (plane > 0) {
    bool copy = false;
    if (!reader.read_bit(&copy)) return set_error("truncated copy flag");
    if (copy) {
      std::memcpy(user->m[t][plane], user->m[t][plane - 1],
                  sizeof(user->m[t][plane]));
      return true;
    }
  }

  bool symmetric = false;
  if (t == 0) {
    if (!reader.read_bit(&symmetric)) {
      return set_error("truncated qm_8x8_is_symmetric");
    }
  } else if (t == 2) {
    bool transpose = false;
    if (!reader.read_bit(&transpose)) {
      return set_error("truncated qm_4x8_is_transpose_of_8x4");
    }
    if (transpose) {
      for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
          user->m[t][plane][i][j] = user->m[1][plane][j][i];
        }
      }
      return true;
    }
  }

  std::uint16_t scan[8 * 8];
  diagonal_scan(w, h, scan);
  int quant = 32;
  bool coef_repeat = false;
  for (int c = 0; c < w * h; ++c) {
    const int pos = scan[c];
    const int row = pos / w;
    const int col = pos % w;
    if (t == 0 && symmetric && col > row) {
      quant = user->m[t][plane][col][row];
      user->m[t][plane][row][col] = static_cast<std::uint8_t>(quant);
    } else if (coef_repeat) {
      user->m[t][plane][row][col] = static_cast<std::uint8_t>(quant);
    } else {
      int delta = 0;
      if (!reader.read_svlc(&delta)) {
        return set_error("truncated or out-of-range quant_delta");
      }
      const int quant2 = (quant + delta) & 255;
      if (quant2 == 0) {
        coef_repeat = true;
      } else {
        quant = quant2;
      }
      user->m[t][plane][row][col] = static_cast<std::uint8_t>(quant);
    }
  }

  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      if (user->m[t][plane][i][j] == 0) {
        return set_error("decoded coefficient 0 is forbidden by §6.4.11");
      }
    }
  }
  return true;
}

bool is_global_layer_type(std::uint8_t type) {
  return type == kObuTypeTemporalDelimiter ||
         type == static_cast<std::uint8_t>(20);  // OBU_MSDO
}

bool level_in_range(int level) { return level >= 0 && level < kNumCustomQms; }

}  // namespace

// ---------------------------------------------------------------------------
// OBU framing
// ---------------------------------------------------------------------------

bool write_obu_header(const ObuHeader& header, std::vector<std::uint8_t>* out,
                      std::string* error) {
  if (out == nullptr) return fail(error, "null OBU header output");
  out->clear();
  if (header.type > 31U) return fail(error, "obu_type is outside f(5)");
  if (header.tlayer_id > 3U) {
    return fail(error, "obu_tlayer_id is outside f(2)");
  }
  if (header.mlayer_id > 7U) {
    return fail(error, "obu_mlayer_id is outside f(3)");
  }
  if (header.xlayer_id > 31U) {
    return fail(error, "obu_xlayer_id is outside f(5)");
  }
  const bool global = is_global_layer_type(header.type);
  if (global && header.mlayer_id != 0) {
    return fail(error, "GLOBAL_XLAYER_ID requires obu_mlayer_id = 0");
  }
  if (global && header.xlayer_id != kGlobalXLayerId &&
      header.xlayer_id != 0) {
    return fail(error,
                "OBU_TEMPORAL_DELIMITER/OBU_MSDO require "
                "obu_xlayer_id = GLOBAL_XLAYER_ID");
  }
  const bool extension =
      !global && (header.mlayer_id != 0 || header.xlayer_id != 0);

  BitWriter writer;
  writer.put_bit(extension);
  writer.put_bits(header.type, 5U);
  writer.put_bits(header.tlayer_id, 2U);
  if (extension) {
    writer.put_bits(header.mlayer_id, 3U);
    writer.put_bits(header.xlayer_id, 5U);
  }
  const auto bytes = writer.bytes();
  out->assign(bytes.begin(), bytes.end());
  return true;
}

bool read_obu_header(std::span<const std::uint8_t> bytes, ObuHeader* header,
                     std::size_t* consumed, std::string* error) {
  if (header == nullptr || consumed == nullptr) {
    return fail(error, "null OBU header output");
  }
  if (bytes.empty()) return fail(error, "truncated OBU header");
  const std::uint8_t first = bytes[0];
  const bool extension = (first & 0x80U) != 0U;
  header->type = static_cast<std::uint8_t>(
      (static_cast<unsigned>(first) >> 2U) & 0x1FU);
  header->tlayer_id = static_cast<std::uint8_t>(first & 0x03U);
  if (extension) {
    if (bytes.size() < 2) {
      return fail(error, "truncated OBU header extension byte");
    }
    header->mlayer_id = static_cast<std::uint8_t>((bytes[1] >> 5U) & 0x07U);
    header->xlayer_id = static_cast<std::uint8_t>(bytes[1] & 0x1FU);
    *consumed = 2;
  } else {
    header->mlayer_id = 0;
    header->xlayer_id =
        is_global_layer_type(header->type) ? kGlobalXLayerId : 0;
    *consumed = 1;
  }
  return true;
}

bool frame_obu(const ObuHeader& header, std::span<const std::uint8_t> payload,
               std::vector<std::uint8_t>* out, std::string* error) {
  if (out == nullptr) return fail(error, "null OBU output");
  std::vector<std::uint8_t> header_bytes;
  if (!write_obu_header(header, &header_bytes, error)) return false;
  const std::uint64_t size =
      static_cast<std::uint64_t>(header_bytes.size()) + payload.size();
  if (size > std::numeric_limits<std::uint32_t>::max()) {
    return fail(error, "OBU is larger than the leb128 size field");
  }
  std::uint32_t value = static_cast<std::uint32_t>(size);
  do {
    std::uint8_t byte = static_cast<std::uint8_t>(value & 0x7FU);
    value >>= 7U;
    if (value != 0) byte = static_cast<std::uint8_t>(byte | 0x80U);
    out->push_back(byte);
  } while (value != 0);
  out->insert(out->end(), header_bytes.begin(), header_bytes.end());
  out->insert(out->end(), payload.begin(), payload.end());
  return true;
}

bool parse_obu_sequence(std::span<const std::uint8_t> bytes,
                        std::vector<ParsedObu>* obus, std::string* error) {
  if (obus == nullptr) return fail(error, "null OBU sequence output");
  obus->clear();
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    const std::size_t start = offset;
    std::uint64_t size_value = 0;
    unsigned shift = 0;
    unsigned leb_bytes = 0;
    while (true) {
      if (offset >= bytes.size()) {
        return fail(error, "truncated leb128 OBU size");
      }
      const std::uint8_t byte = bytes[offset++];
      if (leb_bytes >= 5U) {
        return fail(error, "leb128 OBU size exceeds 32 bits");
      }
      size_value |= static_cast<std::uint64_t>(byte & 0x7FU) << shift;
      ++leb_bytes;
      if ((byte & 0x80U) == 0U) break;
      shift += 7U;
    }
    if (size_value == 0) return fail(error, "OBU size is zero");
    if (size_value > std::numeric_limits<std::uint32_t>::max()) {
      return fail(error, "leb128 OBU size exceeds 32 bits");
    }
    if (size_value > bytes.size() - offset) {
      return fail(error, "OBU size exceeds the remaining input");
    }
    const auto body = bytes.subspan(offset, static_cast<std::size_t>(size_value));
    ParsedObu obu;
    std::size_t header_bytes = 0;
    if (!read_obu_header(body, &obu.header, &header_bytes, error)) {
      return false;
    }
    obu.payload.assign(
        body.begin() + static_cast<std::ptrdiff_t>(header_bytes), body.end());
    obu.raw.assign(bytes.begin() + static_cast<std::ptrdiff_t>(start),
                   bytes.begin() + static_cast<std::ptrdiff_t>(
                                       offset + static_cast<std::size_t>(
                                                    size_value)));
    offset += static_cast<std::size_t>(size_value);
    obus->push_back(std::move(obu));
  }
  return true;
}

// ---------------------------------------------------------------------------
// §5.13 payload
// ---------------------------------------------------------------------------

bool write_quantizer_matrix_payload(BitWriter& writer,
                                    const QuantizerMatrixObu& obu,
                                    std::string* error) {
  if ((obu.bit_map & 0x8000U) != 0U) {
    return fail(error, "qm_bit_map is f(15); bit 15 is not an OBU level");
  }
  writer.put_bits(obu.bit_map, 15U);
  writer.put_bit(obu.chroma_info_present);
  const int num_planes = obu.chroma_info_present ? 3 : 1;

  if (obu.bit_map == 0) {
    writer.put_bit(true);  // §5.2.3 trailing_one_bit
    writer.align_to_byte();
    return true;
  }

  for (int level = 0; level < kNumCustomQms; ++level) {
    if ((obu.bit_map & (1U << level)) == 0U) continue;
    const QmLevelData& data =
        obu.levels[static_cast<std::size_t>(level)];
    writer.put_bit(data.is_default);
    if (data.is_default) continue;
    for (int t = 0; t < kUserQmT; ++t) {
      for (int plane = 0; plane < num_planes; ++plane) {
        if (!write_user_defined_qm(writer, data.user, level, t, plane,
                                   error)) {
          return false;
        }
      }
    }
  }
  writer.put_bit(true);  // §5.2.3 trailing_one_bit
  writer.align_to_byte();
  return true;
}

bool read_quantizer_matrix_payload(std::span<const std::uint8_t> payload,
                                   QuantizerMatrixObu* obu,
                                   std::string* error) {
  if (obu == nullptr) return fail(error, "null QM OBU output");
  *obu = QuantizerMatrixObu{};
  BitSpanReader reader(payload);
  std::uint32_t bit_map = 0;
  std::uint32_t chroma = 0;
  if (!reader.read_bits(15U, &bit_map) || !reader.read_bits(1U, &chroma)) {
    return fail(error, "truncated qm_bit_map/qm_chroma_info_present_flag");
  }
  obu->bit_map = static_cast<std::uint16_t>(bit_map);
  obu->chroma_info_present = chroma != 0U;
  const int num_planes = obu->chroma_info_present ? 3 : 1;

  if (obu->bit_map != 0U) {
    for (int level = 0; level < kNumCustomQms; ++level) {
      if ((obu->bit_map & (1U << level)) == 0U) continue;
      bool is_default = false;
      if (!reader.read_bit(&is_default)) {
        return fail(error, level_prefix(level) + "truncated qm_is_default_flag");
      }
      QmLevelData& data = obu->levels[static_cast<std::size_t>(level)];
      data.is_default = is_default;
      if (is_default) continue;
      for (int t = 0; t < kUserQmT; ++t) {
        for (int plane = 0; plane < num_planes; ++plane) {
          if (!read_user_defined_qm(reader, &data.user, level, t, plane,
                                    error)) {
            return false;
          }
        }
      }
    }
  }
  if (!reader.read_trailing_bits(error)) return false;
  return true;
}

bool make_quantizer_matrix_obu(const QuantizerMatrixObu& obu,
                               const ObuHeader* header,
                               std::vector<std::uint8_t>* out,
                               std::string* error) {
  if (out == nullptr) return fail(error, "null OBU output");
  out->clear();
  BitWriter writer;
  if (!write_quantizer_matrix_payload(writer, obu, error)) return false;
  const auto payload = writer.bytes();
  ObuHeader actual;
  actual.type = kObuTypeQuantizationMatrix;
  if (header != nullptr) actual = *header;
  if (actual.type != kObuTypeQuantizationMatrix) {
    return fail(error, "OBU header type is not OBU_QUANTIZATION_MATRIX");
  }
  return frame_obu(actual, payload, out, error);
}

bool make_constant_user_qm_obu(int level, std::uint8_t value,
                               bool chroma_info_present,
                               std::vector<std::uint8_t>* out,
                               std::string* error) {
  if (!level_in_range(level)) return fail(error, "level is outside 0..14");
  if (value == 0) return fail(error, "coefficient 0 is forbidden by §6.4.11");
  QuantizerMatrixObu obu;
  obu.bit_map = static_cast<std::uint16_t>(1U << level);
  obu.chroma_info_present = chroma_info_present;
  obu.levels[static_cast<std::size_t>(level)].is_default = false;
  std::memset(&obu.levels[static_cast<std::size_t>(level)].user, value,
              sizeof(obu.levels[static_cast<std::size_t>(level)].user));
  return make_quantizer_matrix_obu(obu, nullptr, out, error);
}

// ---------------------------------------------------------------------------
// §5.18.6.2 frame-header fields
// ---------------------------------------------------------------------------

bool write_frame_qm_params(BitWriter& writer, const FrameQm& qm,
                           int num_planes, bool segmentation_enabled,
                           std::string* error) {
  if (num_planes < 1 || num_planes > 3) {
    return fail(error, "NumPlanes is outside 1..3");
  }
  writer.put_bit(qm.using_qmatrix);
  if (!qm.using_qmatrix) return true;

  if (!segmentation_enabled) {
    if (qm.pic_qm_num_minus_1 != 0) {
      return fail(error, "pic_qm_num_minus_1 must be 0 without segmentation");
    }
  } else if (qm.pic_qm_num_minus_1 < 0 || qm.pic_qm_num_minus_1 > 3) {
    return fail(error, "pic_qm_num_minus_1 is outside f(2)");
  }
  if (segmentation_enabled) {
    writer.put_bits(static_cast<std::uint64_t>(qm.pic_qm_num_minus_1), 2U);
  }

  const int qm_num = qm.pic_qm_num_minus_1 + 1;
  for (int i = 0; i < qm_num; ++i) {
    const unsigned y = qm.qm_y[static_cast<std::size_t>(i)];
    if (y > 15U) return fail(error, "qm_y is outside f(4)");
    writer.put_bits(y, 4U);
    if (num_planes == 1) continue;

    const unsigned u = qm.qm_u[static_cast<std::size_t>(i)];
    const unsigned v = qm.qm_v[static_cast<std::size_t>(i)];
    if (u > 15U || v > 15U) return fail(error, "qm_u/qm_v outside f(4)");
    const bool uv_same = qm.qm_uv_same_as_y[static_cast<std::size_t>(i)];
    if (uv_same && (u != y || v != y)) {
      return fail(error, "qm_uv_same_as_y = 1 requires qm_u = qm_v = qm_y");
    }
    writer.put_bit(uv_same);
    if (uv_same) continue;
    writer.put_bits(u, 4U);
    if (qm.separate_uv_delta_q) {
      writer.put_bits(v, 4U);
    } else if (v != u) {
      return fail(error, "qm_v must equal qm_u when separate_uv_delta_q is 0");
    }
  }
  return true;
}

bool read_frame_qm_params(std::span<const std::uint8_t> bytes, int num_planes,
                          bool segmentation_enabled, bool separate_uv_delta_q,
                          FrameQm* qm, std::size_t* bits_consumed,
                          std::string* error) {
  if (qm == nullptr) return fail(error, "null FrameQm output");
  if (num_planes < 1 || num_planes > 3) {
    return fail(error, "NumPlanes is outside 1..3");
  }
  *qm = FrameQm{};
  qm->separate_uv_delta_q = separate_uv_delta_q;
  BitSpanReader reader(bytes);

  bool using_qmatrix = false;
  if (!reader.read_bit(&using_qmatrix)) {
    return fail(error, "truncated using_qmatrix");
  }
  qm->using_qmatrix = using_qmatrix;
  if (!using_qmatrix) {
    if (bits_consumed != nullptr) *bits_consumed = reader.position();
    return true;
  }

  std::uint32_t value = 0;
  if (segmentation_enabled) {
    if (!reader.read_bits(2U, &value)) {
      return fail(error, "truncated pic_qm_num_minus_1");
    }
    qm->pic_qm_num_minus_1 = static_cast<int>(value);
  } else {
    qm->pic_qm_num_minus_1 = 0;
  }

  const int qm_num = qm->pic_qm_num_minus_1 + 1;
  for (int i = 0; i < qm_num; ++i) {
    if (!reader.read_bits(4U, &value)) return fail(error, "truncated qm_y");
    qm->qm_y[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(value);
    if (num_planes == 1) continue;

    bool uv_same = false;
    if (!reader.read_bit(&uv_same)) {
      return fail(error, "truncated qm_uv_same_as_y");
    }
    qm->qm_uv_same_as_y[static_cast<std::size_t>(i)] = uv_same;
    if (uv_same) {
      qm->qm_u[static_cast<std::size_t>(i)] =
          qm->qm_y[static_cast<std::size_t>(i)];
      qm->qm_v[static_cast<std::size_t>(i)] =
          qm->qm_y[static_cast<std::size_t>(i)];
      continue;
    }
    if (!reader.read_bits(4U, &value)) return fail(error, "truncated qm_u");
    qm->qm_u[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(value);
    if (qm->separate_uv_delta_q) {
      if (!reader.read_bits(4U, &value)) return fail(error, "truncated qm_v");
      qm->qm_v[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(value);
    } else {
      qm->qm_v[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(value);
    }
  }
  if (bits_consumed != nullptr) *bits_consumed = reader.position();
  return true;
}

std::size_t frame_qm_params_bits(const FrameQm& qm, int num_planes,
                                 bool segmentation_enabled) {
  std::size_t bits = 1;  // using_qmatrix
  if (!qm.using_qmatrix) return bits;
  if (qm.pic_qm_num_minus_1 < 0 ||
      qm.pic_qm_num_minus_1 >= kMaxFrameQmEntries) {
    return 0;
  }
  if (segmentation_enabled) bits += 2;
  const int qm_num = qm.pic_qm_num_minus_1 + 1;
  for (int i = 0; i < qm_num; ++i) {
    bits += 4;  // qm_y
    if (num_planes == 1) continue;
    bits += 1;  // qm_uv_same_as_y
    if (qm.qm_uv_same_as_y[static_cast<std::size_t>(i)]) continue;
    bits += 4;  // qm_u
    if (qm.separate_uv_delta_q) bits += 4;
  }
  return bits;
}

// ---------------------------------------------------------------------------
// State machine
// ---------------------------------------------------------------------------

QmState::QmState() { reset(); }

void QmState::reset() {
  for (QmLevelState& level : levels_) {
    level = QmLevelState{};
    level.qm_num_planes = 1;
    level.qm_mlayer_id = -1;
    level.qm_tlayer_id = -1;
  }
  obus_in_window_ = 0;
}

void QmState::on_temporal_delimiter() {
  for (QmLevelState& level : levels_) level.qm_protected = false;
}

void QmState::on_frame_header() {
  for (QmLevelState& level : levels_) level.qm_seen = false;
  obus_in_window_ = 0;
}

void QmState::reset_qm(int num_planes) {
  for (QmLevelState& level : levels_) {
    if (level.qm_protected) continue;
    level.qm_data_present = false;
    level.qm_num_planes = num_planes;
    level.qm_mlayer_id = -1;
    level.qm_tlayer_id = -1;
    level.has_user_qm = false;
  }
}

void QmState::on_key_frame(bool first_picture_in_tu, int num_planes) {
  on_frame_header();
  if (first_picture_in_tu) reset_qm(num_planes);
}

bool QmState::apply_obu(const ObuHeader& header,
                        const QuantizerMatrixObu& obu,
                        std::string* error) {
  if (header.type != kObuTypeQuantizationMatrix) {
    return fail(error, "OBU type is not OBU_QUANTIZATION_MATRIX");
  }
  if ((obu.bit_map & 0x8000U) != 0U) {
    return fail(error, "qm_bit_map is f(15); bit 15 is not an OBU level");
  }
  const int num_planes = obu.chroma_info_present ? 3 : 1;

  if (obu.bit_map == 0) {
    if (obus_in_window_ != 0) {
      return fail(error,
                  "qm_bit_map == 0 is only allowed for the first quantizer "
                  "matrix OBU between coded frames");
    }
    for (QmLevelState& level : levels_) {
      level.qm_protected = true;
      level.qm_num_planes = num_planes;
      level.qm_data_present = false;
      level.qm_mlayer_id = -1;
      level.qm_tlayer_id = -1;
      level.has_user_qm = false;
    }
    ++obus_in_window_;
    return true;
  }

  // Validate the whole bitmap before mutating any state.
  for (int level = 0; level < kNumCustomQms; ++level) {
    if ((obu.bit_map & (1U << level)) == 0U) continue;
    if (levels_[static_cast<std::size_t>(level)].qm_seen) {
      return fail(error,
                  "quantizer matrix level " + std::to_string(level) +
                      " is specified twice between coded frames");
    }
    if (!obu.levels[static_cast<std::size_t>(level)].is_default) {
      const UserQmLevel& user =
          obu.levels[static_cast<std::size_t>(level)].user;
      for (int t = 0; t < kUserQmT; ++t) {
        // Only the fundamental extent is coded: 8x8, 8x4, 4x8.
        const int rows = (t == 1) ? 4 : 8;
        const int cols = (t == 2) ? 4 : 8;
        for (int plane = 0; plane < num_planes; ++plane) {
          for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
              if (user.m[t][plane][i][j] == 0U) {
                return fail(error, level_prefix(level) +
                                       "user-defined coefficient 0 is "
                                       "forbidden by §6.4.11");
              }
            }
          }
        }
      }
    }
  }

  for (int level = 0; level < kNumCustomQms; ++level) {
    if ((obu.bit_map & (1U << level)) == 0U) continue;
    QmLevelState& state = levels_[static_cast<std::size_t>(level)];
    const QmLevelData& data = obu.levels[static_cast<std::size_t>(level)];
    state.qm_seen = true;
    state.qm_protected = true;
    state.qm_num_planes = num_planes;
    state.qm_mlayer_id = header.mlayer_id;
    state.qm_tlayer_id = header.tlayer_id;
    state.qm_data_present = !data.is_default;
    state.has_user_qm = !data.is_default;
    if (!data.is_default) state.user = data.user;
  }
  ++obus_in_window_;
  return true;
}

bool QmState::apply_obu_bytes(std::span<const std::uint8_t> bytes,
                              std::string* error) {
  std::vector<ParsedObu> obus;
  if (!parse_obu_sequence(bytes, &obus, error)) return false;
  if (obus.size() != 1) {
    return fail(error, "expected exactly one OBU");
  }
  QuantizerMatrixObu obu;
  if (!read_quantizer_matrix_payload(obus[0].payload, &obu, error)) {
    return false;
  }
  return apply_obu(obus[0].header, obu, error);
}

const QmLevelState& QmState::level(int level) const {
  static const QmLevelState kEmpty;
  if (level < 0 || level >= kNumCustomQms) return kEmpty;
  return levels_[static_cast<std::size_t>(level)];
}

bool QmState::data_present(int level) const {
  return this->level(level).qm_data_present;
}

bool QmState::seen(int level) const { return this->level(level).qm_seen; }

bool QmState::protected_level(int level) const {
  return this->level(level).qm_protected;
}

const UserQmLevel* QmState::user_qm(int level) const {
  if (level < 0 || level >= kNumCustomQms) return nullptr;
  const QmLevelState& state = levels_[static_cast<std::size_t>(level)];
  return state.has_user_qm ? &state.user : nullptr;
}

std::uint16_t QmState::bit_map() const {
  std::uint16_t map = 0;
  for (int level = 0; level < kNumCustomQms; ++level) {
    if (levels_[static_cast<std::size_t>(level)].qm_seen) {
      map = static_cast<std::uint16_t>(map | (1U << level));
    }
  }
  return map;
}

}  // namespace av2enc::qm::obu

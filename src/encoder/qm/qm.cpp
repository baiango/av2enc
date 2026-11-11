#include "encoder/qm/qm.h"

#include "encoder/qm/qm_tables.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <span>

#include "bitstream/m0_headers.h"

namespace av2enc::qm {
namespace {

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

// §4.11.4 svlc() encoding.  delta must be in [-128, 127].
void put_svlc(BitWriter& writer, int delta) {
  std::uint32_t value;
  if (delta > 0) {
    value = static_cast<std::uint32_t>(2 * delta - 1);
  } else if (delta == 0) {
    value = 0;
  } else {
    value = static_cast<std::uint32_t>(-2 * delta);
  }
  put_uvlc(writer, value);
}

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
  switch (t) {
    case 0:
      *w = 8;
      *h = 8;
      break;
    case 1:
      *w = 8;
      *h = 4;
      break;
    default:
      *w = 4;
      *h = 8;
      break;
  }
}

bool matrix_is_symmetric(const std::uint8_t m[8][8], int w, int h) {
  if (w != h) return false;
  for (int i = 0; i < h; ++i) {
    for (int j = i + 1; j < w; ++j) {
      if (m[i][j] != m[j][i]) return false;
    }
  }
  return true;
}

bool matrix_is_transpose(const std::uint8_t a[8][8], const std::uint8_t b[8][8],
                         int aw, int ah) {
  // a is the 4x8 matrix (w=4,h=8), b is the 8x4 matrix (w=8,h=4):
  // a[i][j] == b[j][i].
  for (int i = 0; i < ah; ++i) {
    for (int j = 0; j < aw; ++j) {
      if (a[i][j] != b[j][i]) return false;
    }
  }
  return true;
}

bool write_user_defined_qm(BitWriter& writer, const UserQmLevel& user,
                           int level, int t, int plane, std::string* error) {
  auto set_error = [&](const char* message) {
    if (error != nullptr) {
      *error = "quantizer matrix level " + std::to_string(level) +
               ", t=" + std::to_string(t) + ", plane=" +
               std::to_string(plane) + ": " + message;
    }
    return false;
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
    symmetric = matrix_is_symmetric(user.m[0][plane], w, h);
    writer.put_bit(symmetric);
  } else if (t == 2) {
    const bool transpose =
        matrix_is_transpose(user.m[2][plane], user.m[1][plane], w, h);
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
      const std::uint8_t inferred = user.m[t][plane][col][row];
      if (value != inferred) return set_error("asymmetric inferred entry");
      quant = inferred;
      continue;
    }
    if (coef_repeat) {
      if (value != quant) return set_error("coefRepeat mismatch");
      continue;
    }
    if (value == 0) {
      return set_error("coefficient 0 is forbidden by §6.4.11");
    }
    int delta = (static_cast<int>(value) - quant) & 255;
    if (delta > 127) delta -= 256;
    put_svlc(writer, delta);
    const int quant2 = (quant + delta) & 255;
    if (quant2 == 0) {
      coef_repeat = true;
    } else {
      quant = quant2;
    }
    if (quant != value) return set_error("delta round-trip mismatch");
  }
  return true;
}

}  // namespace

QmFrameState::QmFrameState() {
  for (int plane = 0; plane < 3; ++plane) {
    for (int segment = 0; segment < kMaxSegments; ++segment) {
      seg_qm_level[plane][segment] =
          static_cast<std::uint8_t>(kNoQmLevel);
    }
  }
}

bool valid_tx_size(int tx_size) { return core::valid_tx_size(tx_size); }
bool valid_level(int level) {
  return level >= 0 && level <= kNoQmLevel;
}

std::uint16_t qm_offset(int tx_size) {
  return valid_tx_size(tx_size) ? kQmOffset[tx_size] : 0;
}

std::uint8_t default_matrix_value(int level, int plane, int tx_size, int i,
                                  int j) {
  if (!valid_tx_size(tx_size) || level < 0 || level >= kNumQmLevels) return 0;
  const int w = adjusted_width(tx_size);
  const int h = adjusted_height(tx_size);
  if (i < 0 || j < 0 || i >= h || j >= w) return 0;
  const int plane_type = plane > 0 ? 1 : 0;
  return kQuantizerMatrix[level][plane_type]
                          [kQmOffset[tx_size] + i * w + j];
}

int resolve_seg_level(const QmFrameState& state, int plane, int tx_size,
                      int segment_id) {
  if (!valid_tx_size(tx_size) || plane < 0 || plane > 2) {
    return kNoQmLevel;
  }
  const int tw = adjusted_width(tx_size);
  const int th = adjusted_height(tx_size);
  if (tw > 8 || th > 8) {
    if (plane == 0) return state.qm_y[0];
    if (plane == 1) return state.qm_u[0];
    return state.qm_v[0];
  }
  if (segment_id < 0 || segment_id >= kMaxSegments) return kNoQmLevel;
  if (state.lossless_segment[segment_id]) return kNoQmLevel;
  return state.seg_qm_level[plane][segment_id];
}

bool uses_user_qm(const QmFrameState& state, int plane, int tx_size,
                  int seg_level) {
  if (!state.using_qmatrix || !valid_tx_size(tx_size)) return false;
  if (seg_level < 0 || seg_level >= kNumQmLevels) return false;
  if (!state.qm_data_present[seg_level]) return false;
  const int tw = adjusted_width(tx_size);
  const int th = adjusted_height(tx_size);
  return tw <= 8 && th <= 8 && plane >= 0 && plane <= 2;
}

int matrix_value(const QmFrameState& state, int plane, int tx_size,
                 int seg_level, int i, int j) {
  if (!valid_tx_size(tx_size)) return 0;
  if (uses_user_qm(state, plane, tx_size, seg_level)) {
    const int tw = adjusted_width(tx_size);
    const int th = adjusted_height(tx_size);
    if (tw < th) return state.user_qm[seg_level].m[2][plane][i][j];
    if (tw > th) return state.user_qm[seg_level].m[1][plane][i][j];
    const int qi = i * 8 / th;
    const int qj = j * 8 / tw;
    return state.user_qm[seg_level].m[0][plane][qi][qj];
  }
  return default_matrix_value(seg_level, plane, tx_size, i, j);
}


namespace {

bool make_core_params(const QmDequantParams& params, core::CoeffParams* out,
                      int* tw, int* th) {
  if (params.state == nullptr) return false;
  if (!valid_tx_size(params.tx_size)) return false;
  if (params.plane < 0 || params.plane > 2) return false;
  if (params.segment_id < 0 || params.segment_id >= kMaxSegments) return false;
  if (params.dc_quant <= 0 || params.ac_quant <= 0) return false;

  const QmFrameState& state = *params.state;
  const int seg_level =
      resolve_seg_level(state, params.plane, params.tx_size,
                        params.segment_id);
  core::CoeffParams core_params;
  core_params.tx_size = params.tx_size;
  core_params.plane = params.plane;
  core_params.tx_type = params.tx_type;
  core_params.seg_qm_level = seg_level;
  if (params.plane == 0) {
    core_params.qm_level = state.qm_y[0];
  } else if (params.plane == 1) {
    core_params.qm_level = state.qm_u[0];
  } else {
    core_params.qm_level = state.qm_v[0];
  }
  core_params.dc_quant = params.dc_quant;
  core_params.ac_quant = params.ac_quant;
  core_params.using_qmatrix = state.using_qmatrix;
  core_params.qm_data_present =
      seg_level >= 0 && seg_level < kNumQmLevels
          ? state.qm_data_present[seg_level]
          : false;
  core_params.allow_tcq = params.allow_tcq;
  core_params.lossless = params.lossless;
  core_params.use_fsc = params.use_fsc;
  core_params.qtable = &kQuantizerMatrix[0][0][0];
  core_params.user_qm =
      seg_level >= 0 && seg_level < kNumQmLevels
          ? &state.user_qm[0].m[0][0][0][0]
          : nullptr;
  *out = core_params;
  *tw = adjusted_width(params.tx_size);
  *th = adjusted_height(params.tx_size);
  return true;
}

}  // namespace

bool build_q2_matrix(const QmDequantParams& params, std::int32_t* q2) {
  core::CoeffParams core_params;
  int tw = 0;
  int th = 0;
  if (q2 == nullptr || !make_core_params(params, &core_params, &tw, &th)) {
    return false;
  }
  (void)tw;
  (void)th;
  core::build_q2_block(core_params, q2);
  return true;
}

bool dequantize_block(const std::int32_t* levels,
                      const QmDequantParams& params, std::int32_t* dequant) {
  core::CoeffParams core_params;
  int tw = 0;
  int th = 0;
  if (levels == nullptr || dequant == nullptr ||
      !make_core_params(params, &core_params, &tw, &th)) {
    return false;
  }
  (void)tw;
  (void)th;
  core::dequantize_block(core_params, levels, dequant);
  return true;
}

bool write_setup_qm_params(BitWriter& writer, const QmFrameState& state,
                           int num_planes, bool segmentation_enabled,
                           std::string* error) {
  auto fail = [&](const char* message) {
    if (error != nullptr) *error = std::string("setup_qm_params: ") + message;
    return false;
  };
  if (num_planes < 1 || num_planes > 3) return fail("NumPlanes out of range");
  writer.put_bit(state.using_qmatrix);
  if (!state.using_qmatrix) {
    // The syntax infers nothing else; qm_y/u/v are unused.
    return true;
  }
  const int qm_num_minus_1 = state.pic_qm_num_minus_1;
  if (!segmentation_enabled) {
    if (qm_num_minus_1 != 0) {
      return fail("pic_qm_num_minus_1 must be 0 without segmentation");
    }
  } else {
    if (qm_num_minus_1 < 0 || qm_num_minus_1 > 3) {
      return fail("pic_qm_num_minus_1 outside f(2)");
    }
    writer.put_bits(static_cast<std::uint64_t>(qm_num_minus_1), 2U);
  }
  const int qm_num = qm_num_minus_1 + 1;
  for (int i = 0; i < qm_num; ++i) {
    if (!valid_level(state.qm_y[i]) || !valid_level(state.qm_u[i]) ||
        !valid_level(state.qm_v[i])) {
      return fail("qm level outside 0..15");
    }
    writer.put_bits(state.qm_y[i], 4U);
    if (num_planes > 1) {
      const bool uv_same =
          state.qm_u[i] == state.qm_y[i] && state.qm_v[i] == state.qm_y[i];
      writer.put_bit(uv_same);
      if (!uv_same) {
        writer.put_bits(state.qm_u[i], 4U);
        if (state.separate_uv_delta_q) {
          writer.put_bits(state.qm_v[i], 4U);
        } else if (state.qm_v[i] != state.qm_u[i]) {
          return fail("qm_v must equal qm_u when separate_uv_delta_q is 0");
        }
      }
    }
  }
  return true;
}

bool fill_user_qm_constant(UserQmLevel* level, std::uint8_t value) {
  if (level == nullptr || value == 0) return false;
  for (int t = 0; t < kUserQmT; ++t) {
    for (int plane = 0; plane < kUserQmPlanes; ++plane) {
      std::memset(level->m[t][plane], value, sizeof(level->m[t][plane]));
    }
  }
  return true;
}

bool write_quantizer_matrix_obu_payload(BitWriter& writer,
                                        const QmObuConfig& config,
                                        std::string* error) {
  auto fail = [&](const char* message) {
    if (error != nullptr) {
      *error = std::string("quantizer_matrix_obu: ") + message;
    }
    return false;
  };
  if ((config.bit_map & 0x8000U) != 0) return fail("qm_bit_map is f(15)");

  writer.put_bits(config.bit_map, 15U);
  writer.put_bit(config.chroma_info_present);
  const int num_planes = config.chroma_info_present ? 3 : 1;
  if (config.bit_map == 0) {
    writer.put_bit(true);  // trailing_one_bit
    writer.align_to_byte();
    return true;
  }

  for (int level = 0; level < kNumQmLevels; ++level) {
    if ((config.bit_map & (1U << level)) == 0) continue;
    const QmObuLevel& entry = config.levels[level];
    writer.put_bit(entry.is_default);
    if (entry.is_default) continue;
    for (int t = 0; t < kUserQmT; ++t) {
      for (int plane = 0; plane < num_planes; ++plane) {
        if (!write_user_defined_qm(writer, entry.user, level, t, plane,
                                   error)) {
          return false;
        }
      }
    }
  }
  writer.put_bit(true);  // trailing_one_bit
  writer.align_to_byte();
  return true;
}

std::vector<std::uint8_t> make_quantizer_matrix_obu(
    const QmObuConfig& config, std::string* error) {
  BitWriter writer;
  if (!write_quantizer_matrix_obu_payload(writer, config, error)) {
    return {};
  }
  const auto payload = writer.bytes();
  return av2enc::m0::make_obu(
      av2enc::m0::ObuHeader{
          static_cast<av2enc::m0::ObuType>(kObuTypeQuantizationMatrix)},
      std::span<const std::uint8_t>(payload.data(), payload.size()));
}

}  // namespace av2enc::qm

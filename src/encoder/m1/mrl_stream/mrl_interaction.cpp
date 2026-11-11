// MrlIndex ↔ edge filter / transform-type interaction.  See mrl_interaction.h.

#include "encoder/m1/mrl_stream/mrl_interaction.h"

#include "encoder/intra_modes/intra_modes.h"

namespace av2enc::m1::mrl_stream {
namespace {

// §5.20.7.29 WAIP_WH_RATIO_*_THRES.
constexpr int kWaipRatio2Thres = 61;
constexpr int kWaipRatio4Thres = 73;
constexpr int kWaipRatio8Thres = 82;
constexpr int kWaipRatio16Thres = 86;

bool valid_mode(int y_mode) noexcept { return y_mode >= 0 && y_mode <= 12; }

// wide_angle_mapping() from §5.20.7.29.  Returns the mapped mode index.
int wide_angle_map(int p_angle, int mode, int w, int h) noexcept {
  if ((h == 2 * w && p_angle < kWaipRatio2Thres) ||
      (h == 4 * w && p_angle < kWaipRatio4Thres) ||
      (h == 8 * w && p_angle < kWaipRatio8Thres) ||
      (h == 16 * w && p_angle < kWaipRatio16Thres)) {
    return static_cast<int>(intra::Mode::kD203);
  }
  if ((w == 2 * h && p_angle > 270 - kWaipRatio2Thres) ||
      (w == 4 * h && p_angle > 270 - kWaipRatio4Thres) ||
      (w == 8 * h && p_angle > 270 - kWaipRatio8Thres) ||
      (w == 16 * h && p_angle > 270 - kWaipRatio16Thres)) {
    return static_cast<int>(intra::Mode::kD45);
  }
  return mode;
}

}  // namespace

int mrl_angle_delta(int mrl_index) noexcept {
  if (mrl_index < 0 || mrl_index > mrl::kMaxMrlIndex) return 0;
  return mrl::mrl_angle_delta(mrl_index);
}

int mrl_p_angle(int y_mode, int angle_delta, int mrl_index) noexcept {
  if (!valid_mode(y_mode) || !intra::is_directional(
                                 static_cast<intra::Mode>(y_mode))) {
    return -1;
  }
  if (angle_delta < -intra::kMaxAngleDelta ||
      angle_delta > intra::kMaxAngleDelta) {
    return -1;
  }
  if (mrl_index < 0 || mrl_index > mrl::kMaxMrlIndex) return -1;
  return intra::mode_to_angle(static_cast<intra::Mode>(y_mode)) +
         angle_delta * intra::kAngleStep + mrl_angle_delta(mrl_index);
}

int mapped_intra_mode_for_mrl(int y_mode, int angle_delta, int mrl_index,
                              int coded_w, int coded_h) noexcept {
  if (!valid_mode(y_mode)) return -1;
  if (!intra::is_directional(static_cast<intra::Mode>(y_mode))) {
    // Non-directional modes ignore the angle and the size; MRL is invalid.
    if (mrl_index != 0) return -1;
    return y_mode;
  }
  if (coded_w <= 0 || coded_h <= 0) return -1;
  const int p_angle = mrl_p_angle(y_mode, angle_delta, mrl_index);
  if (p_angle < 0) return -1;
  return wide_angle_map(p_angle, y_mode, coded_w, coded_h);
}

bool mrl_allows_edge_filter(int mrl_index) noexcept { return mrl_index == 0; }

bool make_predict_params(int y_mode, int angle_delta, int mrl_index,
                         bool mrl_sec_index, bool sequence_enable_edge_filter,
                         bool above_neighbor_is_smooth,
                         bool left_neighbor_is_smooth, mrl::Params* params,
                         int block_w, int block_h) noexcept {
  if (params == nullptr) return false;
  if (!valid_mode(y_mode)) return false;
  const bool directional =
      intra::is_directional(static_cast<intra::Mode>(y_mode));
  if (!directional && (mrl_index != 0 || mrl_sec_index)) return false;
  if (mrl_index < 0 || mrl_index > mrl::kMaxMrlIndex) return false;
  if (mrl_index == 0 && mrl_sec_index) return false;
  if (mrl_sec_index && block_w == 4 && block_h == 4) return false;
  if (angle_delta < -intra::kMaxAngleDelta ||
      angle_delta > intra::kMaxAngleDelta) {
    return false;
  }

  mrl::Params out;
  out.mode = static_cast<intra::Mode>(y_mode);
  out.angle_delta = angle_delta;
  out.mrl_index = mrl_index;
  out.mrl_sec_index = mrl_sec_index;
  // §7.13.2.7: filter only the adjacent reference line.
  out.enable_edge_filter =
      sequence_enable_edge_filter && mrl_allows_edge_filter(mrl_index);
  out.above_neighbor_is_smooth = above_neighbor_is_smooth;
  out.left_neighbor_is_smooth = left_neighbor_is_smooth;
  *params = out;
  return true;
}

bool make_tx_type_query(int y_mode, int angle_delta, int mrl_index,
                        int coded_w, int coded_h, int reduced_tx_set,
                        bool dct_only, txtype::TxTypeQuery* query) noexcept {
  if (query == nullptr) return false;
  if (reduced_tx_set < 0 || reduced_tx_set > 3) return false;
  if (!txv::tx_size_supported(coded_w, coded_h)) return false;
  const int mapped = mapped_intra_mode_for_mrl(y_mode, angle_delta, mrl_index,
                                               coded_w, coded_h);
  if (mapped < 0) return false;
  txtype::TxTypeQuery out;
  out.coded_w = coded_w;
  out.coded_h = coded_h;
  out.mapped_mode = mapped;
  out.reduced_tx_set = reduced_tx_set;
  out.dct_only = dct_only;
  *query = out;
  return true;
}

bool tx_type_symbol_for_mrl(int y_mode, int angle_delta, int mrl_index,
                            int coded_w, int coded_h, int reduced_tx_set,
                            bool dct_only, txtype::TxType type, int* symbol,
                            bool* is_long_side_dct) noexcept {
  txtype::TxTypeQuery query;
  if (!make_tx_type_query(y_mode, angle_delta, mrl_index, coded_w, coded_h,
                          reduced_tx_set, dct_only, &query)) {
    return false;
  }
  return txtype::tx_type_symbol(query, type, symbol, is_long_side_dct);
}

}  // namespace av2enc::m1::mrl_stream

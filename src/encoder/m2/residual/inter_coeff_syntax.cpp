#include "encoder/m2/residual/inter_coeff_syntax.h"

#include "encoder/m0_cdf_tables.h"
#include "encoder/m2/m2_syntax.h"
#include "encoder/m2/residual/inter_tx_type.h"
#include "encoder/tx_variants/tx_variants.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace av2enc::m2::residual {
namespace {

using txv::TxParams;
using txv::TxSize;
using txv::TxType;

constexpr int kNumBaseLevels = 2;    // NUM_BASE_LEVELS
constexpr int kCoeffBaseRange = 3;   // COEFF_BASE_RANGE
constexpr int kLfNumBaseLevels = 4;  // LF_NUM_BASE_LEVELS
constexpr int kMaxBaseBrRange = 6;   // MAX_BASE_BR_RANGE
constexpr int kSigCoefContextsEob = 4;

int floor_log2(int value) {
  int result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}

// §5.20.7.23 get_lf_limits() for TX_CLASS_2D.
bool is_low_frequency(int row, int col, int plane) {
  return (row + col) < (plane == 0 ? 4 : 1);
}

template <typename T, std::size_t N>
void copy_array(T (&dst)[N], const T (&src)[N]) {
  std::memcpy(dst, src, sizeof(dst));
}

// Binds a SymbolSink with error reporting.
class SinkWriter {
 public:
  explicit SinkWriter(const SymbolSink& sink) : sink_(sink) {}

  bool symbol(std::uint16_t* cdf, int n, int value) {
    if (sink_.encode_symbol == nullptr) {
      error_ = "symbol sink is not bound";
      return false;
    }
    if (value < 0 || value >= n) {
      error_ = "symbol " + std::to_string(value) + " outside cdf n=" +
               std::to_string(n);
      return false;
    }
    if (!sink_.encode_symbol(sink_.context, cdf, n, value)) {
      error_ = "symbol sink rejected n=" + std::to_string(n) + " value=" +
               std::to_string(value);
      return false;
    }
    return true;
  }

  bool bit(bool value) {
    if (sink_.encode_bool == nullptr) {
      error_ = "bool sink is not bound";
      return false;
    }
    if (!sink_.encode_bool(sink_.context, value)) {
      error_ = "bool sink rejected a literal";
      return false;
    }
    return true;
  }

  bool literal(int bits, std::uint32_t value) {
    for (int i = bits - 1; i >= 0; --i) {
      if (!bit(((value >> static_cast<unsigned>(i)) & 1U) != 0U)) return false;
    }
    return true;
  }

  const std::string& error() const { return error_; }

 private:
  const SymbolSink& sink_;
  std::string error_;
};

// §5.20.7.27 EOB coding.  `eob_ctx` is 0/1 for luma (intra/inter) and 2 for
// chroma.
bool write_eob(SinkWriter& sink, InterResidualCdfs& cdfs, int eob,
               int eob_ctx, int coded_w, int coded_h) {
  int eob_pt = 1;
  if (eob == 2) {
    eob_pt = 2;
  } else if (eob >= 3) {
    eob_pt = floor_log2(eob - 1) + 2;
  }
  const int multisize = std::min(5, floor_log2(coded_w)) +
                        std::min(5, floor_log2(coded_h)) - 4;
  std::uint16_t* cdf = nullptr;
  int escape_bits = 0;
  int cdf_n = 8;
  switch (multisize) {
    case 0:
      cdf = &cdfs.eob_pt_16[eob_ctx][0];
      cdf_n = 5;
      break;
    case 1:
      cdf = &cdfs.eob_pt_32[eob_ctx][0];
      cdf_n = 6;
      break;
    case 2:
      cdf = &cdfs.eob_pt_64[eob_ctx][0];
      cdf_n = 7;
      break;
    case 3:
      cdf = &cdfs.eob_pt_128[eob_ctx][0];
      break;
    case 4:
      cdf = &cdfs.eob_pt_256[eob_ctx][0];
      escape_bits = 1;
      break;
    case 5:
      cdf = &cdfs.eob_pt_512[eob_ctx][0];
      escape_bits = 2;
      break;
    default:
      cdf = &cdfs.eob_pt_1024[eob_ctx][0];
      escape_bits = 2;
      break;
  }
  int symbol = 0;
  int escape = 0;
  if (eob_pt <= cdf_n) {
    symbol = eob_pt - 1;
  } else {
    symbol = cdf_n - 1;
    escape = eob_pt - cdf_n;
  }
  if (!sink.symbol(cdf, cdf_n, symbol)) return false;
  if (symbol == cdf_n - 1 && escape_bits > 0) {
    if (!sink.literal(escape_bits, static_cast<std::uint32_t>(escape))) {
      return false;
    }
  }
  if (eob_pt >= 3) {
    const int base = (1 << (eob_pt - 2)) + 1;
    const int extra = eob - base;
    const int eob_extra = (extra >> (eob_pt - 3)) & 1;
    if (!sink.symbol(&cdfs.eob_extra[0], 2, eob_extra)) return false;
    for (int i = eob_pt - 4; i >= 0; --i) {
      if (!sink.bit(((extra >> i) & 1) != 0)) return false;
    }
  }
  return true;
}

// §5.20.7.28 read_quant() escape part (allowTcq = 0, isHidden = 0).
bool write_quant_escape(SinkWriter& sink, int x, int& hr_level_avg) {
  const int pred_level = hr_level_avg;
  const int m = std::clamp(pred_level <= 0 ? 0 : floor_log2(pred_level), 1, 6);
  const int k = m + 1;
  const int c_max = std::min(m + 4, 6);
  if (x < (c_max << m)) {
    const int q = x >> m;
    for (int i = 0; i < q; ++i) {
      if (!sink.bit(false)) return false;
    }
    if (!sink.bit(true)) return false;
    if (!sink.literal(m, static_cast<std::uint32_t>(x & ((1 << m) - 1)))) {
      return false;
    }
  } else {
    for (int i = 0; i < c_max; ++i) {
      if (!sink.bit(false)) return false;
    }
    int g = 0;
    while (true) {
      const int length = k + g;
      const int x_base = (c_max << m) + (1 << length) - (1 << k);
      if (x < x_base + (1 << length)) {
        for (int i = 0; i < g; ++i) {
          if (!sink.bit(false)) return false;
        }
        if (!sink.bit(true)) return false;
        if (!sink.literal(length, static_cast<std::uint32_t>(x - x_base))) {
          return false;
        }
        break;
      }
      ++g;
    }
  }
  hr_level_avg = (x + hr_level_avg) >> 1;
  return true;
}

bool tx_params_for(std::uint8_t tx_size, TxType type, TxParams* out,
                   std::string& error) {
  const auto params = txv::make_tx_params(static_cast<TxSize>(tx_size), type);
  if (!params.has_value()) {
    error = "unsupported transform size " + std::to_string(tx_size);
    return false;
  }
  *out = *params;
  return true;
}

}  // namespace

void init_inter_residual_cdfs(InterResidualCdfs& cdfs) {
  copy_array(cdfs.inter_tx_type_set1, kInterTxTypeSet1Cdf);
  copy_array(cdfs.inter_tx_type_index_set1, kInterTxTypeIndexSet1Cdf);
  copy_array(cdfs.inter_tx_type_offset_set1, kInterTxTypeOffsetSet1Cdf);
  copy_array(cdfs.inter_tx_type_set2, kInterTxTypeSet2Cdf);
  copy_array(cdfs.inter_tx_type_index_set2, kInterTxTypeIndexSet2Cdf);
  copy_array(cdfs.inter_tx_type_offset_set2, kInterTxTypeOffsetSet2Cdf);
  copy_array(cdfs.inter_tx_type_set3, kInterTxTypeSet3Cdf);
  copy_array(cdfs.inter_tx_type_set4, kInterTxTypeSet4Cdf);
  copy_array(cdfs.inter_tx_type_long, kInterTxTypeLongCdf);
  copy_array(cdfs.is_long_side_dct, kIsLongSideDctCdf);
  copy_array(cdfs.txb_skip, av2enc::m0::kTxbSkipCdf);
  copy_array(cdfs.v_txb_skip, av2enc::m0::kVTxbSkipCdf);
  copy_array(cdfs.eob_pt_16, kEobPt16Cdf);
  copy_array(cdfs.eob_pt_32, kEobPt32Cdf);
  copy_array(cdfs.eob_pt_64, av2enc::m0::kEobPt64Cdf);
  copy_array(cdfs.eob_pt_128, av2enc::m0::kEobPt128Cdf);
  copy_array(cdfs.eob_pt_256, av2enc::m0::kEobPt256Cdf);
  copy_array(cdfs.eob_pt_512, av2enc::m0::kEobPt512Cdf);
  copy_array(cdfs.eob_pt_1024, av2enc::m0::kEobPt1024Cdf);
  copy_array(cdfs.eob_extra, av2enc::m0::kEobExtraCdf);
  copy_array(cdfs.coeff_base, av2enc::m0::kCoeffBaseCdf);
  copy_array(cdfs.coeff_base_lf, av2enc::m0::kCoeffBaseLfCdf);
  copy_array(cdfs.coeff_base_eob, av2enc::m0::kCoeffBaseEobCdf);
  copy_array(cdfs.coeff_base_lf_eob, av2enc::m0::kCoeffBaseLfEobCdf);
  copy_array(cdfs.coeff_br, av2enc::m0::kCoeffBrCdf);
  copy_array(cdfs.coeff_br_lf, av2enc::m0::kCoeffBrLfCdf);
  copy_array(cdfs.coeff_base_uv, av2enc::m0::kCoeffBaseUvCdf);
  copy_array(cdfs.coeff_base_lf_uv, av2enc::m0::kCoeffBaseLfUvCdf);
  copy_array(cdfs.coeff_base_eob_uv, av2enc::m0::kCoeffBaseEobUvCdf);
  copy_array(cdfs.coeff_base_lf_eob_uv, av2enc::m0::kCoeffBaseLfEobUvCdf);
  copy_array(cdfs.coeff_br_uv, av2enc::m0::kCoeffBrUvCdf);
  copy_array(cdfs.dc_sign, av2enc::m0::kDcSignCdf);
}

bool write_inter_tx_type(const SymbolSink& sink_ref, InterResidualCdfs& cdfs,
                         int tx_size, int tx_type, int eob,
                         std::string& error) {
  error.clear();
  SinkWriter sink(sink_ref);
  if (eob <= 0) {
    error = "inter transform type is only coded for a non-zero EOB";
    return false;
  }
  const auto size = static_cast<TxSize>(tx_size);
  const auto type = static_cast<TxType>(tx_type);
  InterTxCandidate candidate;
  if (!inter_tx_symbol(size, /*reduced_tx_set=*/0, /*enable_inter_ddt=*/false,
                       type, &candidate)) {
    error = "transform type is not codable for this inter TX size";
    return false;
  }
  const InterTxSet set = inter_tx_set(size, /*reduced_tx_set=*/0);
  if (set == InterTxSet::kDctOnly) {
    // §5.20.8.2: TX_SET_DCTONLY codes nothing; the type must be DCT_DCT.
    if (type != TxType::kDctDct) {
      error = "TX_SET_DCTONLY can only signal DCT_DCT";
      return false;
    }
    return true;
  }
  const int ctx = std::clamp(inter_tx_type_context(eob, size), 0, 2);
  const int sqr_sz = std::clamp(tx_size_sqr(size), 0, 3);
  switch (set) {
    case InterTxSet::kInter1:
      if (!sink.symbol(&cdfs.inter_tx_type_set1[ctx][sqr_sz][0], 2,
                       candidate.symbol)) {
        error = sink.error();
        return false;
      }
      if (!sink.symbol(candidate.symbol == 0
                           ? &cdfs.inter_tx_type_index_set1[ctx][0]
                           : &cdfs.inter_tx_type_offset_set1[ctx][0],
                       8, candidate.offset_symbol)) {
        error = sink.error();
        return false;
      }
      return true;
    case InterTxSet::kInter2: {
      if (!sink.symbol(&cdfs.inter_tx_type_set2[ctx][0], 2,
                       candidate.symbol)) {
        error = sink.error();
        return false;
      }
      const int offset_ctx =
          std::clamp(inter_tx_type_offset_context(eob, size), 0, 2);
      if (candidate.symbol == 0) {
        if (!sink.symbol(&cdfs.inter_tx_type_index_set2[offset_ctx][0], 8,
                         candidate.offset_symbol)) {
          error = sink.error();
          return false;
        }
      } else {
        if (!sink.symbol(&cdfs.inter_tx_type_offset_set2[offset_ctx][0], 4,
                         candidate.offset_symbol)) {
          error = sink.error();
          return false;
        }
      }
      return true;
    }
    case InterTxSet::kDctIdtx:
      if (!sink.symbol(&cdfs.inter_tx_type_set3[ctx][sqr_sz][0], 2,
                       candidate.symbol)) {
        error = sink.error();
        return false;
      }
      return true;
    case InterTxSet::kDctIdtxIddct:
      if (!sink.symbol(&cdfs.inter_tx_type_set4[ctx][sqr_sz][0], 4,
                       candidate.symbol)) {
        error = sink.error();
        return false;
      }
      return true;
    case InterTxSet::kWide64:
    case InterTxSet::kHigh64:
    case InterTxSet::kWide32:
    case InterTxSet::kHigh32:
      if (candidate.codes_long_flag) {
        const int flag = candidate.is_long_side_dct ? 1 : 0;
        if (!sink.symbol(&cdfs.is_long_side_dct[flag][0], 2, flag)) {
          error = sink.error();
          return false;
        }
      }
      if (!sink.symbol(&cdfs.inter_tx_type_long[ctx][sqr_sz][0], 4,
                       candidate.symbol)) {
        error = sink.error();
        return false;
      }
      return true;
    case InterTxSet::kDctOnly:
      break;
  }
  error = "unreachable inter transform set";
  return false;
}

int dc_sign_context(const CoeffContextView& contexts, int plane, int x4, int y4,
                    int w4, int h4) {
  const int max_x4 = plane == 0 ? contexts.mi_cols : contexts.mi_cols / 2;
  const int max_y4 = plane == 0 ? contexts.mi_rows : contexts.mi_rows / 2;
  int dc_sign = 0;
  for (int k = 0; k < w4; ++k) {
    if (x4 + k < max_x4) {
      const int sign = contexts.above_dc[plane][x4 + k];
      if (sign == 1) {
        --dc_sign;
      } else if (sign == 2) {
        ++dc_sign;
      }
    }
  }
  for (int k = 0; k < h4; ++k) {
    if (y4 + k < max_y4) {
      const int sign = contexts.left_dc[plane][y4 + k];
      if (sign == 1) {
        --dc_sign;
      } else if (sign == 2) {
        ++dc_sign;
      }
    }
  }
  if (dc_sign < 0) return 1;
  if (dc_sign > 0) return 2;
  return 0;
}

bool build_coeff_plan(int plane, int eob, int adj_w, int adj_h,
                      std::span<const std::int32_t> levels_scan,
                      int dc_sign_ctx, std::vector<CoeffPlanEntry>* plan,
                      std::string* error) {
  if (plan == nullptr) {
    if (error != nullptr) *error = "null plan";
    return false;
  }
  plan->clear();
  const int area = adj_w * adj_h;
  if (eob < 0 || eob > area) {
    if (error != nullptr) *error = "eob outside the adjusted transform area";
    return false;
  }
  if (eob == 0) return true;
  if (levels_scan.size() < static_cast<std::size_t>(eob)) {
    if (error != nullptr) *error = "level array is shorter than the EOB";
    return false;
  }
  std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
  av2enc::m2::syntax::build_scan(adj_w, adj_h, scan);
  // §8.3.2 Sig_Ref_Diff_Offset[TX_CLASS_2D] and
  // Mag_Ref_Offset_With_Tx_Class[TX_CLASS_2D].
  static constexpr int kSigRef[5][2] = {
      {0, 1}, {1, 0}, {1, 1}, {0, 2}, {2, 0}};
  static constexpr int kMagRef[3][2] = {{0, 1}, {1, 0}, {1, 1}};

  std::vector<int> level_pre(static_cast<std::size_t>(area), 0);
  plan->resize(static_cast<std::size_t>(eob));

  // First pass, EOB to DC: base and br symbols build Level[][] (§5.20.7.27).
  for (int c = eob - 1; c >= 0; --c) {
    const int pos = scan[static_cast<std::size_t>(c)];
    const int row = pos / adj_w;
    const int col = pos % adj_w;
    const int level = levels_scan[static_cast<std::size_t>(c)];
    const int abs_value = level < 0 ? -level : level;
    const bool lf = is_low_frequency(row, col, plane);
    const int base_levels = lf ? kLfNumBaseLevels : kNumBaseLevels;
    const int base_max = base_levels + 1;
    const int br_max = (lf && plane > 0) ? 0 : kCoeffBaseRange;
    const int pre_max = base_max + br_max;

    int base_level = 0;
    int br_level = 0;
    if (abs_value <= base_levels) {
      base_level = abs_value;
    } else {
      base_level = std::min(base_levels + 1, base_max);
      br_level = std::min(abs_value, pre_max) - base_level;
    }

    int eob_ctx = 0;
    if (c != 0) {
      if (c <= area / 8) {
        eob_ctx = kSigCoefContextsEob - 3;
      } else if (c <= area / 4) {
        eob_ctx = kSigCoefContextsEob - 2;
      } else {
        eob_ctx = kSigCoefContextsEob - 1;
      }
    }

    int base_ctx = eob_ctx;
    if (c != eob - 1) {
      int mag = 0;
      const int num_ref = plane == 0 ? 5 : 3;
      for (int idx = 0; idx < num_ref; ++idx) {
        const int rr = row + kSigRef[idx][0];
        const int cc = col + kSigRef[idx][1];
        if (rr < adj_h && cc < adj_w) {
          const int limit = lf ? 5 : 3;
          mag += std::min(
              level_pre[static_cast<std::size_t>(rr * adj_w + cc)], limit);
        }
      }
      const int sig_ctx = (mag + 1) >> 1;
      if (plane == 0) {
        if (lf) {
          if (c == 0) {
            base_ctx = std::min(sig_ctx, 8);
          } else if (row + col < 2) {
            base_ctx = std::min(sig_ctx, 6) + 9;
          } else {
            base_ctx = std::min(sig_ctx, 4) + 16;
          }
        } else {
          const int ctx2 = std::min(sig_ctx, 4);
          if (row + col < 6) {
            base_ctx = ctx2;
          } else if (row + col < 8) {
            base_ctx = ctx2 + 5;
          } else {
            base_ctx = ctx2 + 10;
          }
        }
      } else {
        const int ctx2 = std::min(sig_ctx, 3);
        base_ctx = plane == 1 ? ctx2 : ctx2 + 4;
      }
    }

    const int pre_level = base_level + br_level;
    const bool codes_br = pre_level > base_levels && !(lf && plane > 0);
    int br_ctx = 0;
    if (codes_br) {
      int mag = 0;
      for (int idx = 0; idx < 3; ++idx) {
        const int rr = row + kMagRef[idx][0];
        const int cc = col + kMagRef[idx][1];
        if (rr < adj_h && cc < adj_w) {
          mag += std::min(
              level_pre[static_cast<std::size_t>(rr * adj_w + cc)],
              kMaxBaseBrRange - 1);
        }
      }
      mag = std::min((mag + 1) >> 1, kMaxBaseBrRange);
      if (plane == 0) {
        br_ctx = lf ? mag + 7 : mag;
      } else {
        br_ctx = std::min(mag, 3);
      }
    }

    level_pre[static_cast<std::size_t>(pos)] = pre_level;

    CoeffPlanEntry& entry = (*plan)[static_cast<std::size_t>(c)];
    entry.base_level = static_cast<std::uint8_t>(base_level);
    entry.br_level = static_cast<std::uint8_t>(br_level);
    entry.is_eob = c == eob - 1 ? 1 : 0;
    entry.is_lf = lf ? 1 : 0;
    entry.codes_br = codes_br ? 1 : 0;
    entry.base_ctx = static_cast<std::int16_t>(base_ctx);
    entry.br_ctx = static_cast<std::int16_t>(br_ctx);
    entry.eob_ctx = static_cast<std::int16_t>(eob_ctx);
    entry.pos = pos;
    entry.abs_level = abs_value;
    entry.pre_level = pre_level;
    entry.escape_delta = abs_value - pre_level;
  }

  // Second pass, EOB to DC: signs and escapes (§5.20.7.27).
  for (int c = eob - 1; c >= 0; --c) {
    CoeffPlanEntry& entry = (*plan)[static_cast<std::size_t>(c)];
    const int pos = entry.pos;
    const int row = pos / adj_w;
    const int col = pos % adj_w;
    const int level = levels_scan[static_cast<std::size_t>(c)];
    entry.sign = level < 0 ? 1 : 0;
    entry.codes_dc_sign = (row == 0 && col == 0 && plane == 0) ? 1 : 0;
    entry.dc_sign_ctx = static_cast<std::int16_t>(dc_sign_ctx);
    const int max_level =
        entry.is_lf != 0
            ? (plane == 0 ? kLfNumBaseLevels + kCoeffBaseRange + 1
                          : kLfNumBaseLevels + 1)
            : kNumBaseLevels + kCoeffBaseRange + 1;
    entry.codes_escape =
        (entry.abs_level >= max_level && entry.pre_level != 0) ? 1 : 0;
  }
  return true;
}

bool write_inter_residual(const SymbolSink& sink_ref,
                          CoeffContextView& contexts,
                          InterResidualCdfs& cdfs,
                          const InterResidualBlock& block,
                          std::span<const std::int32_t> levels,
                          std::string& error) {
  error.clear();
  SinkWriter sink(sink_ref);
  if (contexts.eob_u == nullptr) {
    error = "coefficient context view has no EobU state";
    return false;
  }
  if (block.luma_w <= 0 || block.luma_h <= 0 || block.chroma_w <= 0 ||
      block.chroma_h <= 0) {
    error = "invalid block geometry";
    return false;
  }
  std::size_t levels_cursor = 0;
  for (int plane = 0; plane < kResidualPlaneCount; ++plane) {
    const InterPlaneResidual& info = block.plane[plane];
    const int coded_w = plane == 0 ? block.luma_w : block.chroma_w;
    const int coded_h = plane == 0 ? block.luma_h : block.chroma_h;
    // §5.20.7.29 compute_tx_type(): M2 sets enable_chroma_dctonly, so chroma is
    // always DCT_DCT and its signalled luma type is ignored.
    const TxType type =
        plane == 0
            ? static_cast<TxType>(info.tx_type)
            : inter_chroma_tx_type(static_cast<TxType>(info.tx_type),
                                   static_cast<TxSize>(info.tx_size),
                                   /*reduced_tx_set=*/0,
                                   /*enable_chroma_dctonly=*/true);
    TxParams params;
    if (!tx_params_for(info.tx_size, type, &params, error)) return false;
    if (params.coded_w != coded_w || params.coded_h != coded_h) {
      error = "plane " + std::to_string(plane) +
              " TX size does not match the TX_MODE_LARGEST block size";
      return false;
    }
    const int adj_w = params.w;
    const int adj_h = params.h;
    const int area = adj_w * adj_h;
    const int eob = static_cast<int>(info.eob);
    if (eob < 0 || eob > area) {
      error = "plane " + std::to_string(plane) + " EOB outside the transform";
      return false;
    }
    if (info.levels_offset != levels_cursor) {
      error = "plane " + std::to_string(plane) +
              " level offset is not contiguous in scan order";
      return false;
    }
    if (levels_cursor + static_cast<std::size_t>(eob) > levels.size()) {
      error = "level array overflow";
      return false;
    }
    const std::span<const std::int32_t> plane_levels =
        levels.subspan(levels_cursor, static_cast<std::size_t>(eob));
    levels_cursor += static_cast<std::size_t>(eob);

    const int w4 = coded_w >> 2;
    const int h4 = coded_h >> 2;
    const int sub_x = plane > 0 ? 1 : 0;
    const int sub_y = plane > 0 ? 1 : 0;
    const int x4 = block.mi_col >> sub_x;
    const int y4 = block.mi_row >> sub_y;
    const int max_x4 = plane == 0 ? contexts.mi_cols : contexts.mi_cols / 2;
    const int max_y4 = plane == 0 ? contexts.mi_rows : contexts.mi_rows / 2;
    const int tx_sz_ctx = (tx_size_sqr(static_cast<TxSize>(info.tx_size)) +
                           tx_size_sqr_up(static_cast<TxSize>(info.tx_size)) +
                           1) >> 1;

    // §8.3.2 all_zero context.  With TX_MODE_LARGEST the plane residual block
    // is the TX block, so the luma context is always 0 (bw == w && bh == h).
    int skip_ctx = 0;
    if (plane == 0) {
      skip_ctx = 0;
    } else {
      int above = 0;
      int left = 0;
      for (int i = 0; i < w4; ++i) {
        if (x4 + i < max_x4) {
          above |= contexts.above_level[plane][x4 + i];
          above |= contexts.above_dc[plane][x4 + i];
        }
      }
      for (int i = 0; i < h4; ++i) {
        if (y4 + i < max_y4) {
          left |= contexts.left_level[plane][y4 + i];
          left |= contexts.left_dc[plane][y4 + i];
        }
      }
      skip_ctx = (above != 0 ? 1 : 0) + (left != 0 ? 1 : 0);
      if (plane == 2) {
        if (coded_w * coded_h > adj_w * adj_h) skip_ctx += 3;
        if (*contexts.eob_u != 0) skip_ctx += 6;
      } else {
        skip_ctx += 6;
      }
    }

    if (plane == 2) {
      if (!sink.symbol(&cdfs.v_txb_skip[skip_ctx][0], 2, eob == 0 ? 1 : 0)) {
        error = sink.error();
        return false;
      }
    } else {
      // is_inter = 1 selects CDF plane 1 (§8.3.2 all_zero).
      if (!sink.symbol(&cdfs.txb_skip[1][tx_sz_ctx][skip_ctx][0], 2,
                       eob == 0 ? 1 : 0)) {
        error = sink.error();
        return false;
      }
    }

    if (eob == 0) {
      for (int i = 0; i < w4; ++i) {
        if (x4 + i < max_x4) {
          contexts.above_level[plane][x4 + i] = 0;
          contexts.above_dc[plane][x4 + i] = 0;
        }
      }
      for (int i = 0; i < h4; ++i) {
        if (y4 + i < max_y4) {
          contexts.left_level[plane][y4 + i] = 0;
          contexts.left_dc[plane][y4 + i] = 0;
        }
      }
      if (plane == 1) *contexts.eob_u = 0;
      continue;
    }

    // Luma EOB uses eobCtx = is_inter = 1; chroma uses 2.
    if (!write_eob(sink, cdfs, eob, plane > 0 ? 2 : 1, coded_w, coded_h)) {
      error = sink.error();
      return false;
    }
    if (plane == 0) {
      if (!write_inter_tx_type(sink_ref, cdfs, info.tx_size, info.tx_type, eob,
                               error)) {
        return false;
      }
    }

    int dc_ctx = 0;
    if (plane == 0) {
      dc_ctx = dc_sign_context(contexts, plane, x4, y4, w4, h4);
    }
    std::vector<CoeffPlanEntry> plan;
    if (!build_coeff_plan(plane, eob, adj_w, adj_h, plane_levels, dc_ctx, &plan,
                          &error)) {
      return false;
    }

    // base / br, scan position eob-1 down to 0.
    for (int c = eob - 1; c >= 0; --c) {
      const CoeffPlanEntry& entry = plan[static_cast<std::size_t>(c)];
      const bool lf = entry.is_lf != 0;
      if (entry.is_eob != 0) {
        if (plane == 0) {
          if (lf) {
            if (!sink.symbol(&cdfs.coeff_base_lf_eob[tx_sz_ctx][entry.eob_ctx]
                                                        [0],
                             5, entry.base_level - 1)) {
              error = sink.error();
              return false;
            }
          } else {
            if (!sink.symbol(&cdfs.coeff_base_eob[tx_sz_ctx][entry.eob_ctx][0],
                             3, entry.base_level - 1)) {
              error = sink.error();
              return false;
            }
          }
        } else {
          if (lf) {
            if (!sink.symbol(&cdfs.coeff_base_lf_eob_uv[entry.eob_ctx][0], 5,
                             entry.base_level - 1)) {
              error = sink.error();
              return false;
            }
          } else {
            if (!sink.symbol(&cdfs.coeff_base_eob_uv[entry.eob_ctx][0], 3,
                             entry.base_level - 1)) {
              error = sink.error();
              return false;
            }
          }
        }
      } else if (plane == 0) {
        if (lf) {
          if (!sink.symbol(
                  &cdfs.coeff_base_lf[tx_sz_ctx][entry.base_ctx][0][0], 6,
                  entry.base_level)) {
            error = sink.error();
            return false;
          }
        } else {
          if (!sink.symbol(&cdfs.coeff_base[tx_sz_ctx][entry.base_ctx][0][0],
                           4, entry.base_level)) {
            error = sink.error();
            return false;
          }
        }
      } else {
        if (lf) {
          if (!sink.symbol(&cdfs.coeff_base_lf_uv[entry.base_ctx][0], 6,
                           entry.base_level)) {
            error = sink.error();
            return false;
          }
        } else {
          if (!sink.symbol(&cdfs.coeff_base_uv[entry.base_ctx][0], 4,
                           entry.base_level)) {
            error = sink.error();
            return false;
          }
        }
      }
      if (entry.codes_br != 0) {
        if (plane == 0) {
          if (lf) {
            if (!sink.symbol(&cdfs.coeff_br_lf[entry.br_ctx][0], 4,
                             entry.br_level)) {
              error = sink.error();
              return false;
            }
          } else {
            if (!sink.symbol(&cdfs.coeff_br[entry.br_ctx][0], 4,
                             entry.br_level)) {
              error = sink.error();
              return false;
            }
          }
        } else {
          if (!sink.symbol(&cdfs.coeff_br_uv[entry.br_ctx][0], 4,
                           entry.br_level)) {
            error = sink.error();
            return false;
          }
        }
      }
    }

    // signs and escapes, scan position eob-1 down to 0.
    int hr_level_avg = 0;
    int cul_level = 0;
    int dc_category = 0;
    for (int c = eob - 1; c >= 0; --c) {
      const CoeffPlanEntry& entry = plan[static_cast<std::size_t>(c)];
      if (entry.pre_level != 0) {
        if (entry.codes_dc_sign != 0) {
          if (!sink.symbol(&cdfs.dc_sign[0][0][entry.dc_sign_ctx][0], 2,
                           entry.sign)) {
            error = sink.error();
            return false;
          }
        } else {
          if (!sink.bit(entry.sign != 0)) {
            error = sink.error();
            return false;
          }
        }
        if (entry.codes_escape != 0) {
          if (!write_quant_escape(sink, entry.escape_delta, hr_level_avg)) {
            error = sink.error();
            return false;
          }
        }
      }
      if (entry.pos == 0 && entry.abs_level > 0) {
        dc_category = entry.sign != 0 ? 1 : 2;
      }
      cul_level = std::min(4, cul_level + entry.abs_level);
    }

    for (int i = 0; i < w4; ++i) {
      if (x4 + i < max_x4) {
        contexts.above_level[plane][x4 + i] =
            static_cast<std::uint8_t>(cul_level);
        contexts.above_dc[plane][x4 + i] =
            static_cast<std::uint8_t>(dc_category);
      }
    }
    for (int i = 0; i < h4; ++i) {
      if (y4 + i < max_y4) {
        contexts.left_level[plane][y4 + i] =
            static_cast<std::uint8_t>(cul_level);
        contexts.left_dc[plane][y4 + i] =
            static_cast<std::uint8_t>(dc_category);
      }
    }
    if (plane == 1) *contexts.eob_u = 1;
  }
  return true;
}

}  // namespace av2enc::m2::residual

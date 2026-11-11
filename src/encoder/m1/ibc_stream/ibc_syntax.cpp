#include "encoder/m1/ibc_stream/ibc_syntax.h"

#include <algorithm>
#include <cstring>

#include "encoder/partition/partition.h"

namespace av2enc::m1::ibc_stream {
namespace {

bool enc_symbol(av2::Encoder& enc, std::uint16_t* cdf, int n, int symbol,
                std::string& error) {
  const av2::Status status = enc.encode_symbol(cdf, n, symbol);
  if (status != av2::Status::kOk) {
    error = std::string("encode_symbol failed: ") + av2::status_name(status);
    return false;
  }
  return true;
}

bool dec_symbol(av2::Decoder& dec, std::uint16_t* cdf, int n, int& symbol,
                std::string& error) {
  const av2::Status status = dec.decode_symbol(cdf, n, &symbol);
  if (status != av2::Status::kOk) {
    error = std::string("decode_symbol failed: ") + av2::status_name(status);
    return false;
  }
  return true;
}

bool enc_bool(av2::Encoder& enc, bool bit, std::string& error) {
  const av2::Status status = enc.encode_bool(bit);
  if (status != av2::Status::kOk) {
    error = std::string("encode_bool failed: ") + av2::status_name(status);
    return false;
  }
  return true;
}

bool dec_bool(av2::Decoder& dec, bool& bit, std::string& error) {
  const av2::Status status = dec.read_bool(&bit);
  if (status != av2::Status::kOk) {
    error = std::string("read_bool failed: ") + av2::status_name(status);
    return false;
  }
  return true;
}

bool enc_bits(av2::Encoder& enc, std::uint32_t value, unsigned n,
              std::string& error) {
  for (unsigned i = 0; i < n; ++i) {
    const bool bit = ((value >> (n - 1U - i)) & 1U) != 0U;
    if (!enc_bool(enc, bit, error)) return false;
  }
  return true;
}

// §4.11.6 ns(n), same bit layout as m1_headers.cpp's put_ns().
bool enc_ns(av2::Encoder& enc, std::uint32_t n, std::uint32_t value,
            std::string& error) {
  if (n < 2 || value >= n) {
    error = "ns value out of range";
    return false;
  }
  unsigned w = 0;
  while ((1U << w) < n) ++w;
  const std::uint32_t m = (1U << w) - n;
  if (value < m) {
    return enc_bits(enc, value, w - 1U, error);
  }
  const std::uint32_t x = value + m;
  if (!enc_bits(enc, x >> 1U, w - 1U, error)) return false;
  return enc_bool(enc, (x & 1U) != 0U, error);
}

bool dec_bits(av2::Decoder& dec, unsigned n, std::uint32_t& value,
              std::string& error) {
  value = 0;
  for (unsigned i = 0; i < n; ++i) {
    bool bit = false;
    if (!dec_bool(dec, bit, error)) return false;
    value = (value << 1U) | (bit ? 1U : 0U);
  }
  return true;
}

bool dec_ns(av2::Decoder& dec, std::uint32_t n, std::uint32_t& value,
            std::string& error) {
  if (n < 2) {
    value = 0;
    return true;
  }
  unsigned w = 0;
  while ((1U << w) < n) ++w;
  std::uint32_t first = 0;
  if (!dec_bits(dec, w - 1U, first, error)) return false;
  const std::uint32_t m = (1U << w) - n;
  if (first < m) {
    value = first;
    return true;
  }
  bool last = false;
  if (!dec_bool(dec, last, error)) return false;
  const std::uint32_t x = (first << 1U) | (last ? 1U : 0U);
  if (x < m || x - m >= n) {
    error = "ns value out of range";
    return false;
  }
  value = x - m;
  return true;
}

int floor_log2_u32(std::uint32_t v) {
  int r = 0;
  while (v > 1U) {
    v >>= 1U;
    ++r;
  }
  return r;
}

int shell_class_len(int p, int q) {
  if (p < 0 || p > 6 || p == 2 || q < 0 || q > 1) return 0;
  if (q == 0) {
    switch (p) {
      case 0: return 6;
      case 1: return 7;
      case 3: return 8;
      case 4: return 8;
      case 5: return 9;
      case 6: return 9;
      default: return 0;
    }
  }
  switch (p) {
    case 0: return 7;
    case 1: return 7;
    case 3: return 8;
    case 4: return 9;
    case 5: return 9;
    case 6: return 9;
    default: return 0;
  }
}

std::uint16_t* shell_class_cdf(MvCdfs& cdfs, int p, int q) {
  if (q == 0) return &cdfs.shell_class0[p][0];
  return &cdfs.shell_class1[p][0];
}

int shell_add(int p) { return (11 + p) >> 1; }

bool enc_shell_index(MvCdfs& cdfs, av2::Encoder& enc, int mv_ctx, int p,
                     int index, int& shell_class, std::string& error) {
  if (index < 0 || index > 32767) {
    error = "shell index out of range";
    return false;
  }
  int cls = 0;
  int base = 0;
  if (index >= 2) {
    cls = floor_log2_u32(static_cast<std::uint32_t>(index));
    base = 1 << cls;
  }
  int q = -1;
  int cls_sym = 0;
  for (int candidate_q = 0; candidate_q < 2; ++candidate_q) {
    const int len = shell_class_len(p, candidate_q);
    if (len == 0) continue;
    const int sym = candidate_q == 0 ? cls : cls - shell_add(p);
    if (sym >= 0 && sym < len - 1) {
      q = candidate_q;
      cls_sym = sym;
      break;
    }
  }
  if (q < 0) {
    error = "shell class cannot represent the MVD magnitude";
    return false;
  }
  if (!enc_symbol(enc, &cdfs.shell_set[mv_ctx][0], 2, q, error)) {
    return false;
  }
  const int len = shell_class_len(p, q);
  if (!enc_symbol(enc, shell_class_cdf(cdfs, p, q), len - 1, cls_sym, error)) {
    return false;
  }
  shell_class = cls_sym;
  if (q != 0) {
    shell_class += shell_add(p);
    if (p == 6 && cls_sym == 7) {
      if (!enc_symbol(enc, &cdfs.shell_last_two_classes[mv_ctx][0], 2, 0,
                      error)) {
        return false;
      }
    }
  }
  const int offset = index - base;
  if (shell_class < 2) {
    return enc_symbol(enc, &cdfs.shell_offset_low_class[mv_ctx][shell_class][0],
                      2, offset & 1, error);
  }
  if (shell_class == 2) {
    if (!enc_symbol(enc, &cdfs.shell_offset_class2[mv_ctx][0], 2,
                    offset == 0 ? 0 : 1, error)) {
      return false;
    }
    if (offset != 0) {
      if (!enc_bool(enc, offset != 1, error)) return false;
      if (offset != 1) {
        if (!enc_bool(enc, offset == 3, error)) return false;
      }
    }
    return true;
  }
  for (int i = 0; i < shell_class; ++i) {
    if (!enc_symbol(enc, &cdfs.shell_offset_other_class[mv_ctx][i][0], 2,
                    (offset >> i) & 1, error)) {
      return false;
    }
  }
  return true;
}

bool dec_shell_index(MvCdfs& cdfs, av2::Decoder& dec, int mv_ctx, int p,
                     int& index, int& shell_class, std::string& error) {
  int q = 0;
  if (!dec_symbol(dec, &cdfs.shell_set[mv_ctx][0], 2, q, error)) {
    return false;
  }
  const int len = shell_class_len(p, q);
  if (len == 0) {
    error = "shell class CDF is unavailable for this MvPrecision";
    return false;
  }
  int cls_sym = 0;
  if (!dec_symbol(dec, shell_class_cdf(cdfs, p, q), len - 1, cls_sym,
                  error)) {
    return false;
  }
  shell_class = cls_sym;
  if (q != 0) {
    shell_class += shell_add(p);
    if (p == 6 && cls_sym == 7) {
      int extra = 0;
      if (!dec_symbol(dec, &cdfs.shell_last_two_classes[mv_ctx][0], 2, extra,
                      error)) {
        return false;
      }
      shell_class += extra;
    }
  }
  int offset = 0;
  if (shell_class < 2) {
    if (!dec_symbol(dec,
                    &cdfs.shell_offset_low_class[mv_ctx][shell_class][0], 2,
                    offset, error)) {
      return false;
    }
  } else if (shell_class == 2) {
    int sym = 0;
    if (!dec_symbol(dec, &cdfs.shell_offset_class2[mv_ctx][0], 2, sym,
                    error)) {
      return false;
    }
    offset = sym;
    if (offset != 0) {
      for (int i = 1; i < 3; ++i) {
        bool bit = false;
        if (!dec_bool(dec, bit, error)) return false;
        offset = (bit ? 1 : 0) + i;
        if (offset == i) break;
      }
    }
  } else {
    for (int i = 0; i < shell_class; ++i) {
      int bit = 0;
      if (!dec_symbol(dec, &cdfs.shell_offset_other_class[mv_ctx][i][0], 2,
                      bit, error)) {
        return false;
      }
      offset |= bit << i;
    }
  }
  index = (shell_class == 0 ? 0 : (1 << shell_class)) + offset;
  return true;
}

bool enc_col(MvCdfs& cdfs, av2::Encoder& enc, int mv_ctx, int shell_class,
             int index, int comp1, std::string& error) {
  if (index == 0) return true;
  const int max_pair = index >> 1;
  const bool balanced = (index & 1) == 0 && comp1 == max_pair;
  int col = comp1;
  int col_index_bit = 0;
  if (!balanced) {
    if (comp1 <= max_pair) {
      col = comp1;
      col_index_bit = 0;
    } else {
      col = index - comp1;
      col_index_bit = 1;
    }
  }
  const int max_idx_bits = std::min(max_pair, 2);
  bool unary_complete = true;
  for (int i = 0; i < max_idx_bits; ++i) {
    const int sym = col > i ? 1 : 0;
    if (!enc_symbol(enc, &cdfs.col_mv_greater[mv_ctx][i][0], 2, sym, error)) {
      return false;
    }
    if (sym == 0) {
      unary_complete = false;
      break;
    }
  }
  if (max_pair > 2 && unary_complete) {
    if (!enc_ns(enc, static_cast<std::uint32_t>(max_pair - 1),
                static_cast<std::uint32_t>(col - 2), error)) {
      return false;
    }
  }
  if (!balanced) {
    const int ctx = std::min(shell_class, 3);
    if (!enc_symbol(enc, &cdfs.col_mv_index[mv_ctx][ctx][0], 2, col_index_bit,
                    error)) {
      return false;
    }
  }
  return true;
}

bool dec_col(MvCdfs& cdfs, av2::Decoder& dec, int mv_ctx, int shell_class,
             int index, int& comp1, std::string& error) {
  comp1 = 0;
  if (index == 0) return true;
  const int max_pair = index >> 1;
  int col = 0;
  const int max_idx_bits = std::min(max_pair, 2);
  for (int i = 0; i < max_idx_bits; ++i) {
    int sym = 0;
    if (!dec_symbol(dec, &cdfs.col_mv_greater[mv_ctx][i][0], 2, sym, error)) {
      return false;
    }
    col = i + sym;
    if (sym == 0) break;
  }
  if (max_pair > 2 && col == 2) {
    std::uint32_t remainder = 0;
    if (!dec_ns(dec, static_cast<std::uint32_t>(max_pair - 1), remainder,
                error)) {
      return false;
    }
    col = static_cast<int>(remainder) + 2;
  }
  const bool balanced = (index & 1) == 0 && col == max_pair;
  if (balanced) {
    comp1 = max_pair;
    return true;
  }
  int col_index = 0;
  const int ctx = std::min(shell_class, 3);
  if (!dec_symbol(dec, &cdfs.col_mv_index[mv_ctx][ctx][0], 2, col_index,
                  error)) {
    return false;
  }
  comp1 = col_index == 0 ? col : index - col;
  return true;
}

}  // namespace

void init_ibc_cdfs(IbcCdfs& cdfs) noexcept {
  std::memcpy(cdfs.intrabc, kIntrabcCdf, sizeof(cdfs.intrabc));
  std::memcpy(cdfs.intrabc_mode, kIntrabcModeCdf, sizeof(cdfs.intrabc_mode));
  std::memcpy(cdfs.intrabc_precision, kIntrabcPrecisionCdf,
              sizeof(cdfs.intrabc_precision));
  std::memcpy(cdfs.skip, kSkipCdf, sizeof(cdfs.skip));
  std::memcpy(cdfs.palette_y_mode, kPaletteYModeCdf,
              sizeof(cdfs.palette_y_mode));
  std::memset(&cdfs.mv, 0, sizeof(cdfs.mv));
  for (int ctx = 0; ctx < 2; ++ctx) {
    std::memcpy(cdfs.mv.shell_set[ctx], kJointShellSetCdf,
                sizeof(cdfs.mv.shell_set[ctx]));
    std::memcpy(cdfs.mv.shell_last_two_classes[ctx],
                kJointShellLastTwoClassesCdf,
                sizeof(cdfs.mv.shell_last_two_classes[ctx]));
    std::memcpy(cdfs.mv.shell_offset_class2[ctx], kShellOffsetClass2Cdf,
                sizeof(cdfs.mv.shell_offset_class2[ctx]));
    for (int cls = 0; cls < 2; ++cls) {
      std::memcpy(cdfs.mv.shell_offset_low_class[ctx][cls],
                  kShellOffsetLowClassCdf[cls],
                  sizeof(cdfs.mv.shell_offset_low_class[ctx][cls]));
    }
    std::memcpy(cdfs.mv.shell_offset_other_class[ctx],
                kShellOffsetOtherClassCdf,
                sizeof(cdfs.mv.shell_offset_other_class[ctx]));
    for (int i = 0; i < 2; ++i) {
      std::memcpy(cdfs.mv.col_mv_greater[ctx][i], kColMvGreaterCdf[i],
                  sizeof(cdfs.mv.col_mv_greater[ctx][i]));
    }
    for (int i = 0; i < 4; ++i) {
      std::memcpy(cdfs.mv.col_mv_index[ctx][i], kColMvIndexCdf[i],
                  sizeof(cdfs.mv.col_mv_index[ctx][i]));
    }
  }
  constexpr int kLen0[7] = {6, 7, 0, 8, 8, 9, 9};
  constexpr int kLen1[7] = {7, 7, 0, 8, 9, 9, 9};
  const std::uint16_t* kC0[7] = {
      kJointShell0Class0Cdf, kJointShell1Class0Cdf, nullptr,
      kJointShell3Class0Cdf, kJointShell4Class0Cdf,
      kJointShell5Class0Cdf, kJointShell6Class0Cdf};
  const std::uint16_t* kC1[7] = {
      kJointShell0Class1Cdf, kJointShell1Class1Cdf, nullptr,
      kJointShell3Class1Cdf, kJointShell4Class1Cdf,
      kJointShell5Class1Cdf, kJointShell6Class1Cdf};
  for (int p = 0; p < 7; ++p) {
    if (p == 2) continue;
    std::memcpy(cdfs.mv.shell_class0[p], kC0[p],
                static_cast<std::size_t>(kLen0[p]) * sizeof(std::uint16_t));
    std::memcpy(cdfs.mv.shell_class1[p], kC1[p],
                static_cast<std::size_t>(kLen1[p]) * sizeof(std::uint16_t));
  }
}

bool intrabc_available(bool allow_intrabc, int bsize,
                       bool tree_is_chroma_part) noexcept {
  const auto size = static_cast<partition::BlockSize>(bsize);
  const int w = partition::block_width(size);
  const int h = partition::block_height(size);
  return allow_intrabc && !tree_is_chroma_part && w > 0 && h > 0 && w <= 64 &&
         h <= 64 && size != partition::BlockSize::k64x64;
}

bool write_use_intrabc(av2::Encoder& enc, IbcCdfs& cdfs, int ctx, bool value,
                       std::string& error) {
  if (ctx < 0 || ctx >= 3) {
    error = "use_intrabc context out of range";
    return false;
  }
  return enc_symbol(enc, &cdfs.intrabc[ctx][0], 2, value ? 1 : 0, error);
}

bool read_use_intrabc(av2::Decoder& dec, IbcCdfs& cdfs, int ctx, bool& value,
                      std::string& error) {
  if (ctx < 0 || ctx >= 3) {
    error = "use_intrabc context out of range";
    return false;
  }
  int symbol = 0;
  if (!dec_symbol(dec, &cdfs.intrabc[ctx][0], 2, symbol, error)) return false;
  value = symbol != 0;
  return true;
}

bool write_skip_flag(av2::Encoder& enc, IbcCdfs& cdfs, int ctx, bool value,
                     std::string& error) {
  if (ctx < 0 || ctx >= 6) {
    error = "skip_flag context out of range";
    return false;
  }
  return enc_symbol(enc, &cdfs.skip[ctx][0], 2, value ? 1 : 0, error);
}

bool read_skip_flag(av2::Decoder& dec, IbcCdfs& cdfs, int ctx, bool& value,
                    std::string& error) {
  if (ctx < 0 || ctx >= 6) {
    error = "skip_flag context out of range";
    return false;
  }
  int symbol = 0;
  if (!dec_symbol(dec, &cdfs.skip[ctx][0], 2, symbol, error)) return false;
  value = symbol != 0;
  return true;
}

BlockVector stack_bv_at(const BvStack& stack, int idx) noexcept {
  if (idx < 0 || idx >= stack.count) return BlockVector{0, 0};
  return stack.entries[static_cast<std::size_t>(idx)].bv;
}

int mv_clamp_to_integer_component(int v) noexcept {
  if (v < kMvLow + 1) return kMvLow + 8;
  if (v > kMvUp - 1) return kMvUp - 8;
  return v;
}

void lower_mv_precision(int mv_precision, BlockVector& bv) noexcept {
  if (mv_precision >= kMvPrecisionHalfPel || mv_precision < 0) return;
  const int bits = kMvPrecisionEighthPel - mv_precision;
  if (bits <= 0 || bits > 6) return;
  const int radix = 1 << bits;
  const int shift = bits - 1;
  const auto lower = [&](int v) {
    const int a = v < 0 ? -v : v;
    int a_int = (a - 1 + (1 << shift)) >> bits;
    if (a_int < 0) a_int = 0;
    int out = v >= 0 ? (a_int << bits) : -((a_int) << bits);
    if ((a_int << bits) != a) {
      out = std::max(kMvLow + radix, std::min(kMvUp - radix, out));
    }
    return out;
  };
  bv.row = lower(bv.row);
  bv.col = lower(bv.col);
}

bool write_mv(av2::Encoder& enc, MvCdfs& cdfs, int mv_ctx, int mv_precision,
              BlockVector diff8, bool derive_sign, std::string& error) {
  if (derive_sign) {
    error = "enable_mvd_sign_derive is not supported by the M1 IBC adapter";
    return false;
  }
  if (mv_ctx < 0 || mv_ctx >= 2) {
    error = "MV context out of range";
    return false;
  }
  if (mv_precision < 0 || mv_precision > 6 || mv_precision == 2) {
    error = "MvPrecision value is not reachable";
    return false;
  }
  const int shift = kMvPrecisionEighthPel - mv_precision;
  const int quantum = 1 << shift;
  if ((diff8.row % quantum) != 0 || (diff8.col % quantum) != 0) {
    error = "MVD is not representable at the selected MvPrecision";
    return false;
  }
  const int ar = (diff8.row < 0 ? -diff8.row : diff8.row) >> shift;
  const int ac = (diff8.col < 0 ? -diff8.col : diff8.col) >> shift;
  const int index = ar + ac;
  if (index > 32767) {
    error = "MVD magnitude exceeds the shell syntax range";
    return false;
  }
  int shell_class = 0;
  if (!enc_shell_index(cdfs, enc, mv_ctx, mv_precision, index, shell_class,
                       error)) {
    return false;
  }
  if (!enc_col(cdfs, enc, mv_ctx, shell_class, index, ac, error)) return false;
  if (diff8.row != 0) {
    if (!enc_bool(enc, diff8.row < 0, error)) return false;
  }
  if (diff8.col != 0) {
    if (!enc_bool(enc, diff8.col < 0, error)) return false;
  }
  return true;
}

bool read_mv(av2::Decoder& dec, MvCdfs& cdfs, int mv_ctx, int mv_precision,
             bool derive_sign, BlockVector& diff8, std::string& error) {
  diff8 = BlockVector{0, 0};
  if (derive_sign) {
    error = "enable_mvd_sign_derive is not supported by the M1 IBC adapter";
    return false;
  }
  if (mv_ctx < 0 || mv_ctx >= 2) {
    error = "MV context out of range";
    return false;
  }
  if (mv_precision < 0 || mv_precision > 6 || mv_precision == 2) {
    error = "MvPrecision value is not reachable";
    return false;
  }
  int index = 0;
  int shell_class = 0;
  if (!dec_shell_index(cdfs, dec, mv_ctx, mv_precision, index, shell_class,
                       error)) {
    return false;
  }
  int comp1 = 0;
  if (!dec_col(cdfs, dec, mv_ctx, shell_class, index, comp1, error)) {
    return false;
  }
  int comp0 = index - comp1;
  bool neg_row = false;
  bool neg_col = false;
  if (comp0 != 0) {
    if (!dec_bool(dec, neg_row, error)) return false;
  }
  if (comp1 != 0) {
    if (!dec_bool(dec, neg_col, error)) return false;
  }
  const int shift = kMvPrecisionEighthPel - mv_precision;
  diff8.row = (neg_row ? -comp0 : comp0) << shift;
  diff8.col = (neg_col ? -comp1 : comp1) << shift;
  return true;
}

bool write_intrabc_info(av2::Encoder& enc, IbcCdfs& cdfs,
                        const ScreenContentConfig& config,
                        const BvStack& stack, const BvSyntax& syntax,
                        BlockVector target_bv, std::string& error) {
  const int mode = syntax.mode == IntrabcMode::kNearMv ? 1 : 0;
  if (!enc_symbol(enc, &cdfs.intrabc_mode[0], 2, mode, error)) return false;
  const int m = config.max_bvp_drl_bits_minus_1 + 1;
  if (syntax.ref_mv_idx < 0 || syntax.ref_mv_idx > m) {
    error = "RefMvIdx is outside the BVP DRL window";
    return false;
  }
  for (int idx = 0; idx < m; ++idx) {
    const bool more = idx < syntax.ref_mv_idx;
    if (!enc_bool(enc, more, error)) return false;
    if (!more) break;
  }
  BlockVector base = stack_bv_at(stack, syntax.ref_mv_idx);
  if (syntax.mode == IntrabcMode::kNearMv) {
    if (target_bv != base) {
      error = "NEARMV target BV does not match the RefStackMv candidate";
      return false;
    }
    base.row = mv_clamp_to_integer_component(base.row);
    base.col = mv_clamp_to_integer_component(base.col);
    if (target_bv != base) {
      error = "NEARMV candidate is outside the integer MV range";
      return false;
    }
    return true;
  }
  int precision = kMvPrecisionOnePel;
  if (!config.force_integer_mv) {
    if (!enc_symbol(enc, &cdfs.intrabc_precision[0], 2,
                    syntax.intrabc_precision_quarter ? 1 : 0, error)) {
      return false;
    }
    precision = syntax.intrabc_precision_quarter ? kMvPrecisionQuarterPel
                                                 : kMvPrecisionOnePel;
  }
  lower_mv_precision(precision, base);
  const BlockVector diff{target_bv.row - base.row, target_bv.col - base.col};
  return write_mv(enc, cdfs.mv, 1, precision, diff,
                  config.enable_mvd_sign_derive, error);
}

bool read_intrabc_info(av2::Decoder& dec, IbcCdfs& cdfs,
                       const ScreenContentConfig& config,
                       const BvStack& stack, BvSyntax& syntax,
                       BlockVector& bv, std::string& error) {
  int mode = 0;
  if (!dec_symbol(dec, &cdfs.intrabc_mode[0], 2, mode, error)) return false;
  syntax.mode = mode == 1 ? IntrabcMode::kNearMv : IntrabcMode::kNewMv;
  const int m = config.max_bvp_drl_bits_minus_1 + 1;
  int ref = 0;
  for (int idx = 0; idx < m; ++idx) {
    bool more = false;
    if (!dec_bool(dec, more, error)) return false;
    if (!more) {
      ref = idx;
      break;
    }
    ref = idx + 1;
  }
  syntax.ref_mv_idx = ref;
  BlockVector base = stack_bv_at(stack, ref);
  if (syntax.mode == IntrabcMode::kNearMv) {
    bv.row = mv_clamp_to_integer_component(base.row);
    bv.col = mv_clamp_to_integer_component(base.col);
    syntax.diff = BlockVector{0, 0};
    return true;
  }
  int precision = kMvPrecisionOnePel;
  if (!config.force_integer_mv) {
    int sym = 0;
    if (!dec_symbol(dec, &cdfs.intrabc_precision[0], 2, sym, error)) {
      return false;
    }
    syntax.intrabc_precision_quarter = sym != 0;
    precision = sym != 0 ? kMvPrecisionQuarterPel : kMvPrecisionOnePel;
  } else {
    syntax.intrabc_precision_quarter = false;
  }
  lower_mv_precision(precision, base);
  BlockVector diff{0, 0};
  if (!read_mv(dec, cdfs.mv, 1, precision, config.enable_mvd_sign_derive, diff,
               error)) {
    return false;
  }
  syntax.diff = diff;
  bv.row = mv_clamp_to_integer_component(base.row + diff.row);
  bv.col = mv_clamp_to_integer_component(base.col + diff.col);
  return true;
}

}  // namespace av2enc::m1::ibc_stream

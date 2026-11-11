#include "syntax.h"

#include <cstddef>

namespace av2enc::cfl::alpha {
namespace {

// A model rate large enough that an unreachable symbol is never selected.
constexpr int kRateCap = 1 << 20;

// log2(x) in Q16 (x >= 1), computed with integer squaring. The fractional
// part is produced bit by bit: y starts in [2^31, 2^32) (Q31 in [1, 2)), and
// each squaring step contributes one bit of log2. No libm, no tables, so the
// RD decision is identical on every host and can be moved to a kernel later.
std::uint32_t log2_q16(std::uint32_t x) {
  int n = 0;
  for (std::uint32_t v = x; v > 1; v >>= 1) {
    ++n;
  }
  std::uint64_t y = static_cast<std::uint64_t>(x) << (31 - n);
  std::uint32_t frac = 0;
  for (int i = 0; i < 31; ++i) {
    y = (y * y) >> 31;  // Q31 value squared, still in [1, 4)
    if (y >= (std::uint64_t{1} << 32)) {
      frac |= std::uint32_t{1} << (30 - i);
      y >>= 1;
    }
  }
  return (static_cast<std::uint32_t>(n) << 16) | (frac >> 15);
}

bool canonical_explicit(const Signalling& s) {
  if (s.cfl_index != kIndexExplicit) {
    return false;
  }
  if (s.alpha_signs < 0 || s.alpha_signs >= kJointSigns) {
    return false;
  }
  const int sign_u = (s.alpha_signs + 1) / 3;
  const int sign_v = (s.alpha_signs + 1) % 3;
  if (sign_u == kSignZero) {
    if (s.alpha_u != 0) {
      return false;
    }
  } else if (s.alpha_u < 0 || s.alpha_u >= kAlphabetSize) {
    return false;
  }
  if (sign_v == kSignZero) {
    if (s.alpha_v != 0) {
      return false;
    }
  } else if (s.alpha_v < 0 || s.alpha_v >= kAlphabetSize) {
    return false;
  }
  return signalable(pair_from_signalling(s));
}

}  // namespace

bool cdfs_valid(const CflCdfs& cdfs) {
  for (int ctx = 0; ctx < kContexts; ++ctx) {
    if (!av2::cdf_is_valid(cdfs.is_cfl[ctx], kIsCflSymbols)) {
      return false;
    }
  }
  if (!av2::cdf_is_valid(cdfs.cfl_index, kCflIndexSymbols)) {
    return false;
  }
  if (!av2::cdf_is_valid(cdfs.cfl_sign, kCflSignSymbols)) {
    return false;
  }
  for (int ctx = 0; ctx < kAlphaContexts; ++ctx) {
    if (!av2::cdf_is_valid(cdfs.cfl_alpha[ctx], kCflAlphaSymbols)) {
      return false;
    }
  }
  return true;
}

int symbol_rate_q8(const std::uint16_t* cdf, int n, int symbol) {
  if (cdf == nullptr || n < 2 || n > av2::kMaxSymbols || symbol < 0 ||
      symbol >= n) {
    return kRateCap;
  }
  av2::Boundaries boundaries{};
  if (av2::compute_boundaries(av2::kSymbolRangeInit, cdf, n, boundaries) !=
      av2::Status::kOk) {
    return kRateCap;
  }
  const std::uint32_t cur = boundaries.cur[symbol];
  const std::uint32_t prev =
      symbol == 0 ? av2::kSymbolRangeInit : boundaries.cur[symbol - 1];
  const std::uint32_t width = prev - cur;
  if (width == 0) {
    return kRateCap;  // zero-probability symbol in the current model
  }
  // bits = -log2(width / 32768) = 15 - log2(width), in 1/256 bit units.
  const int bits_q8 =
      (15 << 8) - static_cast<int>((log2_q16(width) + 128) >> 8);
  return bits_q8 < 0 ? 0 : bits_q8;
}

int alpha_syntax_rate_q8(const Signalling& s, const CflCdfs& cdfs) {
  if (s.cfl_index == kIndexDerivedAlpha) {
    return symbol_rate_q8(cdfs.cfl_index, kCflIndexSymbols, 1);
  }
  if (!canonical_explicit(s)) {
    return kRateCap;
  }
  int total = symbol_rate_q8(cdfs.cfl_index, kCflIndexSymbols, 0);
  total +=
      symbol_rate_q8(cdfs.cfl_sign, kCflSignSymbols, s.alpha_signs);
  const int sign_u = (s.alpha_signs + 1) / 3;
  const int sign_v = (s.alpha_signs + 1) % 3;
  if (sign_u != kSignZero) {
    const int ctx = alpha_u_context(s.alpha_signs);
    total += symbol_rate_q8(cdfs.cfl_alpha[ctx], kCflAlphaSymbols, s.alpha_u);
  }
  if (sign_v != kSignZero) {
    const int ctx = alpha_v_context(s.alpha_signs);
    total += symbol_rate_q8(cdfs.cfl_alpha[ctx], kCflAlphaSymbols, s.alpha_v);
  }
  return total;
}

av2::Status write_is_cfl(av2::Encoder& enc, CflCdfs& cdfs, int context,
                         bool is_cfl) {
  if (context < 0 || context >= kContexts) {
    return av2::Status::kInvalidArgument;
  }
  return enc.encode_symbol(cdfs.is_cfl[context], kIsCflSymbols,
                           is_cfl ? 1 : 0);
}

av2::Status read_is_cfl(av2::Decoder& dec, CflCdfs& cdfs, int context,
                        bool* is_cfl) {
  if (context < 0 || context >= kContexts || is_cfl == nullptr) {
    return av2::Status::kInvalidArgument;
  }
  int symbol = 0;
  const av2::Status status =
      dec.decode_symbol(cdfs.is_cfl[context], kIsCflSymbols, &symbol);
  if (status != av2::Status::kOk) {
    return status;
  }
  *is_cfl = symbol != 0;
  return av2::Status::kOk;
}

av2::Status write_cfl_alphas(av2::Encoder& enc, CflCdfs& cdfs,
                             const Signalling& s) {
  if (s.cfl_index != kIndexExplicit && s.cfl_index != kIndexDerivedAlpha) {
    // CFL_MULTI (MHCCP) needs cfl_mhccp/cfl_mh_dir and is out of scope.
    return av2::Status::kInvalidSymbol;
  }
  if (s.cfl_index == kIndexExplicit && !canonical_explicit(s)) {
    // Validate before touching the coder so a rejected Signalling writes
    // nothing at all.
    return av2::Status::kInvalidSymbol;
  }
  const int index_symbol = s.cfl_index == kIndexExplicit ? 0 : 1;
  av2::Status status =
      enc.encode_symbol(cdfs.cfl_index, kCflIndexSymbols, index_symbol);
  if (status != av2::Status::kOk || s.cfl_index == kIndexDerivedAlpha) {
    return status;
  }
  status = enc.encode_symbol(cdfs.cfl_sign, kCflSignSymbols, s.alpha_signs);
  if (status != av2::Status::kOk) {
    return status;
  }
  const int sign_u = (s.alpha_signs + 1) / 3;
  const int sign_v = (s.alpha_signs + 1) % 3;
  if (sign_u != kSignZero) {
    const int ctx = alpha_u_context(s.alpha_signs);
    status =
        enc.encode_symbol(cdfs.cfl_alpha[ctx], kCflAlphaSymbols, s.alpha_u);
    if (status != av2::Status::kOk) {
      return status;
    }
  }
  if (sign_v != kSignZero) {
    const int ctx = alpha_v_context(s.alpha_signs);
    status =
        enc.encode_symbol(cdfs.cfl_alpha[ctx], kCflAlphaSymbols, s.alpha_v);
    if (status != av2::Status::kOk) {
      return status;
    }
  }
  return av2::Status::kOk;
}

av2::Status read_cfl_alphas(av2::Decoder& dec, CflCdfs& cdfs, Signalling* s) {
  if (s == nullptr) {
    return av2::Status::kInvalidArgument;
  }
  int index_symbol = 0;
  av2::Status status =
      dec.decode_symbol(cdfs.cfl_index, kCflIndexSymbols, &index_symbol);
  if (status != av2::Status::kOk) {
    return status;
  }
  Signalling out{};
  if (index_symbol == 1) {
    out.cfl_index = kIndexDerivedAlpha;
    *s = out;
    return av2::Status::kOk;
  }
  out.cfl_index = kIndexExplicit;
  status = dec.decode_symbol(cdfs.cfl_sign, kCflSignSymbols, &out.alpha_signs);
  if (status != av2::Status::kOk) {
    return status;
  }
  const int sign_u = (out.alpha_signs + 1) / 3;
  const int sign_v = (out.alpha_signs + 1) % 3;
  if (sign_u != kSignZero) {
    const int ctx = alpha_u_context(out.alpha_signs);
    status = dec.decode_symbol(cdfs.cfl_alpha[ctx], kCflAlphaSymbols,
                               &out.alpha_u);
    if (status != av2::Status::kOk) {
      return status;
    }
  }
  if (sign_v != kSignZero) {
    const int ctx = alpha_v_context(out.alpha_signs);
    status = dec.decode_symbol(cdfs.cfl_alpha[ctx], kCflAlphaSymbols,
                               &out.alpha_v);
    if (status != av2::Status::kOk) {
      return status;
    }
  }
  // The decoder cannot produce (zero, zero): cfl_alpha_signs only codes
  // 3*signU + signV - 1 for signU/signV in {0,1,2}, i.e. never both zero.
  if (!canonical_explicit(out)) {
    return av2::Status::kInvalidSymbol;
  }
  *s = out;
  return av2::Status::kOk;
}

av2::Status write_cfl_mode_and_alphas(av2::Encoder& enc, CflCdfs& cdfs,
                                      int context, bool is_cfl,
                                      const Signalling& s) {
  if (context < 0 || context >= kContexts) {
    return av2::Status::kInvalidArgument;
  }
  if (is_cfl) {
    if (s.cfl_index != kIndexExplicit &&
        s.cfl_index != kIndexDerivedAlpha) {
      return av2::Status::kInvalidSymbol;
    }
    if (s.cfl_index == kIndexExplicit && !canonical_explicit(s)) {
      return av2::Status::kInvalidSymbol;
    }
  }
  const av2::Status status = write_is_cfl(enc, cdfs, context, is_cfl);
  if (status != av2::Status::kOk || !is_cfl) {
    return status;
  }
  return write_cfl_alphas(enc, cdfs, s);
}

av2::Status read_cfl_mode_and_alphas(av2::Decoder& dec, CflCdfs& cdfs,
                                     int context, bool* is_cfl, Signalling* s) {
  if (is_cfl == nullptr || s == nullptr) {
    return av2::Status::kInvalidArgument;
  }
  const av2::Status status = read_is_cfl(dec, cdfs, context, is_cfl);
  if (status != av2::Status::kOk) {
    return status;
  }
  if (!*is_cfl) {
    *s = Signalling{};
    s->cfl_index = -1;  // no CfL alphas follow
    return av2::Status::kOk;
  }
  return read_cfl_alphas(dec, cdfs, s);
}

}  // namespace av2enc::cfl::alpha

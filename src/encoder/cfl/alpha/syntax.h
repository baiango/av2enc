#ifndef AV2ENC_ENCODER_CFL_ALPHA_SYNTAX_H_
#define AV2ENC_ENCODER_CFL_ALPHA_SYNTAX_H_

// Exact CfL alpha syntax, §5.20.7.32 read_cfl_alphas(), and the §9.3 default
// CDFs it selects (§8.3, Tables 8.4/8.5).
//
// This is the *bitstream* half of the explicit-alpha path. It is deliberately
// separate from the search/RD half in alpha.h: the search decides what to
// signal, this file is the only place that writes or reads the symbols.
//
// Scope / assumptions:
//   * `enable_cfl_intra` is true and MHCCP is not allowed, so the syntax is
//     `cfl_index S()` (2 symbols: CFL_EXPLICIT=0, CFL_DERIVED_ALPHA=1). The
//     CFL_MULTI branch needs the preceding `cfl_mhccp` flag and is rejected
//     here rather than guessed.
//   * `is_cfl` is written/read by the caller's uv-mode syntax. The helpers in
//     this file cover that symbol too because a CfL block always emits it
//     immediately before cfl_index; uv_mode itself is not in this component.
//
// CDF arrays are caller-owned copies, one per tile, exactly like
// src/entropy/av2. `write_*` and `read_*` mutate them in lockstep through the
// shared §8.2 coder.

#include "../cfl_common.h"

#include "../../../entropy/av2/av2_entropy.h"

#include <cstdint>

namespace av2enc::cfl::alpha {

inline constexpr int kIsCflSymbols = 2;      // is_cfl: 0, 1
inline constexpr int kCflIndexSymbols = 2;   // cfl_index: explicit, derived
inline constexpr int kCflSignSymbols = 8;    // CFL_JOINT_SIGNS
inline constexpr int kCflAlphaSymbols = 8;   // CFL_ALPHABET_SIZE

// One tile's CfL CDF copy. The array shapes are the §9.3 declarations:
//   Default_Is_Cfl_Cdf[3][2+1]
//   Default_Cfl_Index_Cdf[2+1]
//   Default_Cfl_Sign_Cdf[8+1]
//   Default_Cfl_Alpha_Cdf[6][8+1]
struct CflCdfs {
  std::uint16_t is_cfl[kContexts][kIsCflSymbols + 1];
  std::uint16_t cfl_index[kCflIndexSymbols + 1];
  std::uint16_t cfl_sign[kCflSignSymbols + 1];
  std::uint16_t cfl_alpha[kAlphaContexts][kCflAlphaSymbols + 1];
};

// AV2 v1.0.0 §9.3, pages 627 and 647-648. Values are pinned by
// alpha_test.cpp against both the literal spec text and the repo's generated
// src/encoder/m1/m1_cdf_tables.h.
constexpr CflCdfs default_cfl_cdfs() {
  CflCdfs c{};
  c.is_cfl[0][0] = 20441;
  c.is_cfl[0][1] = 30;
  c.is_cfl[0][2] = 0;
  c.is_cfl[1][0] = 11610;
  c.is_cfl[1][1] = 6;
  c.is_cfl[1][2] = 0;
  c.is_cfl[2][0] = 4643;
  c.is_cfl[2][1] = 0;
  c.is_cfl[2][2] = 0;

  c.cfl_index[0] = 12507;
  c.cfl_index[1] = 30;
  c.cfl_index[2] = 0;

  c.cfl_sign[0] = 2421;
  c.cfl_sign[1] = 4332;
  c.cfl_sign[2] = 11256;
  c.cfl_sign[3] = 12766;
  c.cfl_sign[4] = 21386;
  c.cfl_sign[5] = 28725;
  c.cfl_sign[6] = 32087;
  c.cfl_sign[7] = 62;
  c.cfl_sign[8] = 0;

  const std::uint16_t alpha[kAlphaContexts][kCflAlphaSymbols + 1] = {
      {21679, 25305, 30646, 31512, 32537, 32646, 32696, 62, 0},
      {8262, 16302, 24082, 29422, 31398, 32286, 32525, 62, 0},
      {17235, 26166, 30378, 31305, 32373, 32549, 32668, 62, 0},
      {17618, 25732, 27865, 30338, 31125, 31522, 32238, 62, 0},
      {17542, 23066, 27907, 28728, 30702, 31165, 31435, 62, 0},
      {17675, 24802, 30468, 30783, 31841, 32264, 32422, 62, 0},
  };
  for (int i = 0; i < kAlphaContexts; ++i) {
    for (int j = 0; j < kCflAlphaSymbols + 1; ++j) {
      c.cfl_alpha[i][j] = alpha[i][j];
    }
  }
  return c;
}

// Structural check of every array: monotonic inverse CDF, base-rate index in
// range, use count <= 32 (§8.2). Returns false if any row is malformed.
bool cdfs_valid(const CflCdfs& cdfs);

// §8.3 is_cfl context: count of available above/left chroma neighbours whose
// UVCfls flag is set. Availability is derived by the caller at chroma-mi
// granularity (§5.20.4.1).
constexpr int is_cfl_context(bool above_cfl, bool left_cfl) {
  return (above_cfl ? 1 : 0) + (left_cfl ? 1 : 0);
}

// §8.3 Table 8.4: cfl_alpha_u context = cfl_alpha_signs - 2, only meaningful
// when signU != CFL_SIGN_ZERO. Returns -1 otherwise.
constexpr int alpha_u_context(int alpha_signs) {
  if (alpha_signs < 2 || alpha_signs >= kJointSigns) {
    return -1;
  }
  return alpha_signs - 2;
}

// §8.3 Table 8.5: cfl_alpha_v context = (signV - 1) * 3 + signU, only
// meaningful when signV != CFL_SIGN_ZERO. Returns -1 otherwise.
constexpr int alpha_v_context(int alpha_signs) {
  if (alpha_signs < 0 || alpha_signs >= kJointSigns) {
    return -1;
  }
  const int sign_u = (alpha_signs + 1) / 3;
  const int sign_v = (alpha_signs + 1) % 3;
  if (sign_v == kSignZero) {
    return -1;
  }
  return (sign_v - 1) * 3 + sign_u;
}

// Model rate of one symbol under `cdf` (n symbols), in 1/256 bit units, using
// the exact §8.2.6 boundaries at SymbolRange = 1 << 15. Deterministic integer
// arithmetic (no libm), so an RD decision is reproducible across builds.
int symbol_rate_q8(const std::uint16_t* cdf, int n, int symbol);

// Model rate of the whole CfL alpha syntax for `s`, in 1/256 bit units,
// excluding the preceding is_cfl symbol (common to all candidates). Derived
// mode is one cfl_index symbol; explicit mode adds cfl_alpha_signs and one
// cfl_alpha symbol per non-zero sign.
int alpha_syntax_rate_q8(const Signalling& s, const CflCdfs& cdfs);

// ---------------------------------------------------------------------------
// Writers / readers. All functions return av2::Status; on error the coder is
// left in the state the entropy coder documents (no partial symbol).
// ---------------------------------------------------------------------------

// is_cfl S(). `context` is is_cfl_context(). This is the symbol emitted by
// read_intra_uv_mode() immediately before read_cfl_alphas().
av2::Status write_is_cfl(av2::Encoder& enc, CflCdfs& cdfs, int context,
                         bool is_cfl);
av2::Status read_is_cfl(av2::Decoder& dec, CflCdfs& cdfs, int context,
                        bool* is_cfl);

// read_cfl_alphas() without the is_cfl/uv_mode prefix. Writes:
//   cfl_index S()
//   if explicit:
//     cfl_alpha_signs S()
//     cfl_alpha_u S()   when signU != zero
//     cfl_alpha_v S()   when signV != zero
// `s` must be canonical (alpha_u/v are 0 where the sign is zero). Returns
// av2::Status::kInvalidSymbol for CFL_MULTI, a non-canonical Signalling, or a
// (zero, zero) sign pair.
av2::Status write_cfl_alphas(av2::Encoder& enc, CflCdfs& cdfs,
                             const Signalling& s);
av2::Status read_cfl_alphas(av2::Decoder& dec, CflCdfs& cdfs, Signalling* s);

// Convenience for a CfL block: is_cfl=1 followed by read_cfl_alphas(). These
// are the exact call sites in src/encoder/m1/m1_syntax.cpp::code_uv_mode();
// see README.md. When is_cfl is false only the is_cfl symbol is written and
// `s` is unused.
av2::Status write_cfl_mode_and_alphas(av2::Encoder& enc, CflCdfs& cdfs,
                                      int context, bool is_cfl,
                                      const Signalling& s);
av2::Status read_cfl_mode_and_alphas(av2::Decoder& dec, CflCdfs& cdfs,
                                     int context, bool* is_cfl, Signalling* s);

}  // namespace av2enc::cfl::alpha

#endif  // AV2ENC_ENCODER_CFL_ALPHA_SYNTAX_H_

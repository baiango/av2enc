// Inter DDT (Data Dependent Transform) for M2: DDTX / FDDT kernels.
//
// Normative sources (AV2 v1.0.0 PDF):
//   * §5.4.8 sequence_transform_quant_entropy_config(): the only DDT syntax
//     element, `enable_inter_ddt f(1)`, present iff
//     `single_picture_header_flag == 0` (PDF p. 63).
//   * §6.4.8: `enable_inter_ddt = 1` allows DDT for inter blocks (p. 269).
//   * §7.15.2.1 1d inverse transform process: the DDTX / FDDT branches
//     (p. 519) and Table 7.1 (1D transform types 4 = DDTX, 5 = FDDT).
//   * §7.15.4.1 2D matrix transform process: `get_transform_1d_type()`
//     (p. 525):
//
//         useDdt = enable_inter_ddt && !use_intrabc && is_inter
//         t = Transform_1d_Type[PlaneTxType][dir]
//         if (useDdt && (t == ADST || t == FDST) && sz != 4)
//             return (t == ADST) ? DDTX : FDDT
//         return t
//
//   * §9.6 1d transform tables: Ddtx_Kernel8 / Ddtx_Kernel16 (PDF p. 848).
//
// What DDT is NOT:
//   * It is not a new TX_TYPE and it has no per-block or per-frame syntax.
//     The bitstream still signals the ordinary `inter_tx_type` symbol; the
//     decoder derives DDTX/FDDT from that type and the 1D length.
//   * It is not gated by `enable_idtx_intra`, `enable_fsc` or
//     `enable_chroma_dctonly`.  `enable_chroma_dctonly` only constrains the
//     *chroma* TxType to DCT_DCT before §7.15 is reached, so it indirectly
//     disables DDT on chroma.
//   * It does not remove identity transforms from the inter TX sets.  The
//     substitution only rewrites ADST/FDST components (sz != 4).
//
// Block sizes / planes where DDT can change the reconstruction:
//   A 1D direction is rewritten only when its *adjusted* length is 8 or 16
//   (the only Ddtx kernels) and the signalled 1D type is ADST or FDST.
//   Adjusted length 32 is allowed by the predicate but §7.15.2.1's `sz == 32`
//   branch uses Dct_Kernel32, i.e. DDTX/FDDT alias DCT there.  Length 4 is
//   excluded by the explicit `sz != 4` test.
//
//   The legal (tx_size, tx_type) pairs follow from §5.20.8.3 get_tx_set()
//   + Tx_Type_In_Set_Inter; see ddt_constraints.h.  Luma and chroma are both
//   subject; chroma uses the luma TxType when it is in the chroma set.
//
// The inverse below is the normative decoder process.  The AV2 specification
// does not define an encoder forward transform, so `forward_*` is an
// integer-exact transpose of the normative inverse kernel pair (using the
// spec's 2896/4096 rational for the odd-log2 rectangular pre-scale).  It is
// deterministic and produces coefficients in the decoder's dequant scale:
// with a dequant step of 528/8 the matching level is
// `round(coeff * dq_denom / 66)` (see level_from_coeff()).
#pragma once

#include <cstdint>

#include "codec/transform/variants/tx_variants.h"

namespace av2enc::video::ddt {

using txv::Tx1dType;
using txv::TxParams;
using txv::TxSize;
using txv::TxType;

// Table 7.1 1D transform type values.  kDdtx/kFddt are decoder-derived; they
// are never written to the bitstream.
enum class Effective1d : std::int8_t {
  kDct = 0,
  kIdt = 1,
  kAdst = 2,
  kFdst = 3,
  kDdtx = 4,
  kFddt = 5,
};

const char* effective_1d_name(Effective1d type);

// Per-block decoder state needed by get_transform_1d_type().
struct Context {
  bool enable_inter_ddt = false;
  bool is_inter = false;
  bool use_intrabc = false;
  // Not part of get_transform_1d_type(); used to reject lossless/FSC blocks,
  // where compute_tx_type() (§5.20.7.29) can never return ADST/FDST.
  bool lossless = false;
  int bit_depth = 8;
};

// Result of applying get_transform_1d_type() to both directions of a block.
struct Plan {
  Effective1d row = Effective1d::kDct;
  Effective1d col = Effective1d::kDct;
  bool row_ddt = false;
  bool col_ddt = false;

  bool any_ddt() const { return row_ddt || col_ddt; }
  // True when the plan is a valid DDT plan (i.e. at least one kernel changed).
  bool matches(const Plan& other) const;
};

// §7.15.4.1 get_transform_1d_type(dir, sz).  `signaled` is the 1D type from
// Transform_1d_Type[PlaneTxType][dir]; `sz` is the *adjusted* length.
Effective1d effective_1d(Effective1d signaled, int sz, const Context& ctx);

// Applies effective_1d() to the row (width) and column (height) directions of
// `params`.  Returns a plan even when DDT is off.
Plan make_plan(const TxParams& params, const Context& ctx);

// §7.15.2.1 1d inverse transform, including the DDTX/FDDT branches.
//   src[n]  -> out[n]
//   shift   = Transform_Shift[txSz][dir]
//   col_tx  = true for the column pass (changes the clip to +-2^BitDepth)
//   bit_depth is 8 or 10.
// Returns false for an unsupported length/type combination.
bool inverse_1d(const std::int32_t* src, int n, int shift, bool col_tx,
                Effective1d type, int bit_depth, std::int32_t* out);

// Encoder-side forward transform (transpose of inverse_1d).  `inverse_shift`
// is the same Transform_Shift[txSz][dir] the decoder will use.  Output
// coefficients are in the decoder's dequant scale (see the header comment).
bool forward_1d(const std::int32_t* src, int n, int inverse_shift,
                Effective1d type, std::int32_t* out);

// §7.15.4 + §7.15.4.1 inverse on one transform block.  `dequant` is the
// adjusted-size raster array (w*h, the layout txv::dequantize produces);
// `residual_coded` receives coded_w*coded_h samples (adjusted-size sample
// duplication for 64-wide/tall blocks).
bool inverse_block(const std::int32_t* dequant, const TxParams& params,
                   const Context& ctx, std::int16_t* residual_coded);

// Encoder counterpart of inverse_block().  `residual_adjusted` is the
// adjusted-size residual (w*h); `coeff` receives w*h integer coefficients.
bool forward_block(const std::int16_t* residual_adjusted,
                   const TxParams& params, const Context& ctx,
                   std::int32_t* coeff);

// Quantiser helpers for forward_block() output.  The shared dequant process
// (§7.14.4, QUANT_TABLE_BITS = 3) is
//     dequant = Round2(|level| * 528, 3) / dq_denom = |level| * 66 / dq_denom
// so the matching encoder level is round(coeff * dq_denom / 66).
double quant_factor(const TxParams& params);
std::int32_t level_from_coeff(std::int32_t coeff, const TxParams& params);

// ---------------------------------------------------------------------------
// Introspection used by the tests / validation.
// ---------------------------------------------------------------------------

// The exact matrix the transform applies for `type` and length `n`:
//   DDTX: Ddtx_KernelN
//   FDDT: Ddtx_KernelN with the sample (second) index mirrored
//   DCT/ADST/FDST: the txv §9.6 kernel
//   IDT: nullptr (identity)
// Lengths other than 4/8/16/32 return nullptr.
const std::int16_t* effective_kernel(Effective1d type, int n);

// True when DDT changes at least one kernel of this block.  This is stricter
// than plan.any_ddt(): a DDTX/FDDT derived at adjusted length 32 is alias-ed to
// Dct_Kernel32 by §7.15.2.1, so it does not change the reconstruction.
bool plan_has_ddt(const TxParams& params, const Context& ctx);

}  // namespace av2enc::video::ddt

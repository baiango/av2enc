// AV2 v1.0.0 §5.20.7.27 / §8.3 — `dc_sign_horz_vert`.
//
// Normative source: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7):
//
//   §5.20.7.27 coefficients(), p. 228-229 (sign element selection):
//     if (row == 0 && col == 0 && plane == 0)          dc_sign S()
//     else if (txClass == TX_CLASS_HORIZ && col == 0 && plane == 0)
//                                                      dc_sign_horz_vert S()
//     else if (txClass == TX_CLASS_VERT  && row == 0 && plane == 0)
//                                                      dc_sign_horz_vert S()
//     else                                             sign_bit L(1)
//
//   §8.3, p. 603 (CDF):
//     dc_sign:            TileDcSignCdf[ ptype ][ isHidden ][ ctx ]
//     dc_sign_horz_vert:  TileDcSignCdf[ ptype ][ isHidden ][ 0 ]
//
//   §8.3, p. 602 (dc_sign ctx):
//     maxX4 = MiCols; maxY4 = MiRows
//     dcSign = 0
//     for (k = 0; k < w4; k++) if (x4+k < maxX4) { sign = AboveDcContext[plane][x4+k];
//         if (sign == 1) dcSign--; else if (sign == 2) dcSign++; }
//     for (k = 0; k < h4; k++) if (y4+k < maxY4) { sign = LeftDcContext[plane][y4+k];
//         if (sign == 1) dcSign--; else if (sign == 2) dcSign++; }
//     ctx = dcSign < 0 ? 1 : (dcSign > 0 ? 2 : 0)
//
// `dc_sign_horz_vert` therefore has *no* neighbour-derived context: its ctx is
// the constant 0.  `dc_sign` keeps the 3-way neighbour context and is only
// coded for the DC position of a luma block.
//
// The syntax is only reachable with a 1D transform class
// (`H_DCT/H_ADST/H_FLIPADST` or `V_DCT/V_ADST/V_FLIPADST`).  The shipped M1
// writer filters every identity 1D candidate before decision
// (`src/encoder/m1/m1_still.cpp:848-855`), so this element is ABSENT from the
// production path even though the decoder supports it.  `missing_bits`
// supplies the writer/reader and a minimal decoder-gated stream
// (`gate_dc_sign_horz_vert.cpp`).
//
// Production call site for the equivalent sign decision is
// `src/encoder/m1/m1_syntax.cpp:1276-1288`; it only has the `dc_sign` branch
// and falls back to `raw_bit` for every other position, which is exactly the
// §5.20.7.27 `sign_bit L(1)` path.  Adding a 1D transform therefore needs this
// module's `write_sign_element()` at that site.

#pragma once

#include <cstdint>

#include "entropy/av2/av2_entropy.h"

namespace av2enc::m1::missing_bits {

// §5.20.7.29 get_tx_class().
enum class TxClass : std::uint8_t {
  k2D = 0,
  kHoriz = 1,
  kVert = 2,
};

const char* tx_class_name(TxClass tx_class) noexcept;

// Which sign element §5.20.7.27 reads for a non-zero coefficient at
// (row, col) of `plane`.
enum class SignElement : std::uint8_t {
  kDcSign = 0,
  kDcSignHorzVert = 1,
  kSignBit = 2,
};

const char* sign_element_name(SignElement element) noexcept;

SignElement select_sign_element(int plane, int row, int col,
                                TxClass tx_class) noexcept;

// §8.3: dc_sign_horz_vert always uses context 0.
constexpr int dc_sign_horz_vert_context() noexcept { return 0; }

// §8.3 dc_sign context from the AboveDcContext / LeftDcContext arrays.
// `above_dc` has at least x4+w4 entries, `left_dc` at least y4+h4.  Values are
// the dcCategory stored by the coefficient writer (0, 1 or 2).
int dc_sign_context(const std::int8_t* above_dc, const std::int8_t* left_dc,
                    int x4, int y4, int w4, int h4, int mi_cols,
                    int mi_rows) noexcept;

// `TileDcSignCdf[ptype][isHidden][ctx]` is a 2-symbol CDF.  Both helpers
// return the cdf pointer (3 entries: cdf[0], cdf[1], rate/count) that the
// §8.2 coder consumes.  `table` is the `[2][2][3][3]` default table.
using DcSignCdfTable = std::uint16_t[2][2][3][3];

std::uint16_t* dc_sign_cdf_slot(DcSignCdfTable table, int ptype, int is_hidden,
                                int ctx) noexcept;
std::uint16_t* dc_sign_horz_vert_cdf_slot(DcSignCdfTable table, int ptype,
                                          int is_hidden) noexcept;

// Writer / reader for one `dc_sign_horz_vert S()`.  `cdf` must be the
// `TileDcSignCdf[ptype][isHidden][0]` slot.  `sign` is 0 (positive) or 1
// (negative), matching §5.20.7.27's `sign = dc_sign_horz_vert`.
bool write_dc_sign_horz_vert(av2::Encoder& encoder, std::uint16_t* cdf,
                             int sign) noexcept;
bool read_dc_sign_horz_vert(av2::Decoder& decoder, std::uint16_t* cdf,
                            int* sign) noexcept;

bool write_dc_sign(av2::Encoder& encoder, std::uint16_t* cdf, int ctx,
                   int sign) noexcept;
bool read_dc_sign(av2::Decoder& decoder, std::uint16_t* cdf, int ctx,
                  int* sign) noexcept;

// Dispatches the element chosen by select_sign_element().  For kSignBit a raw
// §8.2.3 bit is coded.  `cdf` must be the TileDcSignCdf slot for the selected
// element (ignored for kSignBit); `dc_sign_ctx` is ignored for horz/vert.
bool write_sign_element(av2::Encoder& encoder, SignElement element,
                        std::uint16_t* cdf, int dc_sign_ctx, int sign) noexcept;
bool read_sign_element(av2::Decoder& decoder, SignElement element,
                       std::uint16_t* cdf, int dc_sign_ctx,
                       int* sign) noexcept;

}  // namespace av2enc::m1::missing_bits

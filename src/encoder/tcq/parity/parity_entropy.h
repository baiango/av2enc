// Entropy round-trip for AV2 parity hiding (§8.2 + §5.20.7.27/28).
//
// `write_block()` emits the coefficient payload of one transform block:
//
//     txb_skip, EOB prefix + extra, coeff_base/coeff_base_eob/coeff_base_ph,
//     coeff_br, signs (dc_sign or bypass), read_quant bypass escapes.
//
// `read_block()` is the matching reader.  It is not a second implementation
// of the contexts: it calls the same `coeff_base_cdf_decision()` /
// `coeff_br_cdf_decision()` helpers as the encoder, so a CDF-axis mistake
// shows up as an entropy round-trip failure.
//
// The payload is not a complete AV2 tile.  It is the exact coefficient
// sub-syntax, including the `isHidden`-dependent CDF axes.  Wrapping it in a
// real stream needs the header/obstruction edits listed in README.md.

#pragma once

#include "parity_hiding.h"

#include "entropy/av2/av2_entropy.h"

#include <cstdint>
#include <string>

namespace av2enc::tcq::parity {

// Mutable copies of the coefficient CDFs a tile owns.  The leading
// `COEFF_CDF_Q_CTXS` slice is selected by `qctx`; the arrays match the
// `m1::qctx::CoeffCdfSlice` layouts exactly.
struct CdfStore {
  std::uint16_t txb_skip[2][5][10][3];
  std::uint16_t v_txb_skip[12][3];
  std::uint16_t eob_pt_16[3][6];
  std::uint16_t eob_pt_32[3][7];
  std::uint16_t eob_pt_64[3][8];
  std::uint16_t eob_pt_128[3][9];
  std::uint16_t eob_pt_256[3][9];
  std::uint16_t eob_pt_512[3][9];
  std::uint16_t eob_pt_1024[3][9];
  std::uint16_t eob_extra[3];
  std::uint16_t coeff_base_eob[5][4][4];
  std::uint16_t coeff_base_lf_eob[5][4][6];
  std::uint16_t coeff_base_eob_uv[4][4];
  std::uint16_t coeff_base_lf_eob_uv[4][6];
  std::uint16_t coeff_base[5][20][2][5];
  std::uint16_t coeff_base_lf[5][33][2][7];
  std::uint16_t coeff_base_uv[12][5];
  std::uint16_t coeff_base_lf_uv[12][7];
  std::uint16_t coeff_base_ph[5][5];
  std::uint16_t coeff_br[7][5];
  std::uint16_t coeff_br_lf[14][5];
  std::uint16_t coeff_br_uv[4][5];
  std::uint16_t dc_sign[2][2][3][3];

  explicit CdfStore(std::int32_t qctx = 0);
};

struct EntropyConfig {
  Eligibility eligibility{};
  std::int32_t qctx = 0;
  std::int32_t txb_skip_ctx = 0;  // 0..9 luma/U, 0..11 V
  std::int32_t dc_sign_ctx = 0;   // 0..2
  bool disable_cdf_update = true;
};

bool entropy_config_valid(const EntropyConfig& config) noexcept;

// Writes one block.  `syntax` must come from `derive_syntax()`.
bool write_block(av2::Encoder& encoder, CdfStore& cdfs,
                 const BlockSyntax& syntax, const Geometry& geometry,
                 const EntropyConfig& config, std::string* error = nullptr);

// Reads one block and reconstructs `Quant[]` with `decode_syntax()`.  The
// returned syntax record is identical to the writer's input (including the
// `isHidden` flag, `sumAbs1`, CDF identifiers and contexts).
bool read_block(av2::Decoder& decoder, CdfStore& cdfs,
                const Geometry& geometry, const EntropyConfig& config,
                BlockSyntax* syntax, DecodedBlock* decoded,
                std::string* error = nullptr);

// One `read_quant` escape value.  Exposed for the tests.
bool write_escape_bits(av2::Encoder& encoder, std::int32_t x,
                       std::int32_t predictor, std::string* error = nullptr);
bool read_escape_bits(av2::Decoder& decoder, std::int32_t predictor,
                      std::int32_t* x, std::string* error = nullptr);

}  // namespace av2enc::tcq::parity

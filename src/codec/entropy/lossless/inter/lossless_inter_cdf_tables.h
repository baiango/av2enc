// §9.3 default CDF table needed only by the lossless inter module.
//
// Source: spec/tables/default_cdf.json, AV2 v1.0.0 commit 38f28e7:
//   Default_Lossless_Inter_Tx_Type_Cdf = [16384, 0, 0]  (2 symbols)
//
// `lossless_inter_tx_type S()` (§5.20.8.2) is a symbol-coded element, not a
// raw bit.  Keeping the table here avoids touching the still-picture module's
// generated headers.
#pragma once

#include <cstdint>

namespace av2enc::lossless::inter {

inline constexpr std::uint16_t kLosslessInterTxTypeCdf[3] = {16384, 0, 0};

}  // namespace av2enc::lossless::inter

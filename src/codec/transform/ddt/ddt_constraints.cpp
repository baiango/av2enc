#include "codec/transform/ddt/ddt_constraints.h"

#include <algorithm>

namespace av2enc::video::ddt {
namespace {

// §6.19.8.2 TX_SET_* enum values (Table 6.19: is_inter set -> name/value).
enum class InterSet : std::int8_t {
  kDctOnly = 0,
  kWide64 = 1,
  kHigh64 = 2,
  kWide32 = 3,
  kHigh32 = 4,
  kInter1 = 5,
  kInter2 = 6,
  kDctIdtx = 7,
  kDctIdtxIddct = 8,
};

// §5.20.7.29 Tx_Type_In_Set_Inter[9][16].  Implemented here from the spec
// table rather than borrowed from m2/residual: that sibling's INTER_2 row has
// only the first 8 entries set, while the spec sets types 0..11 (see the
// README "upstream findings").  DDT legality must follow the spec.
constexpr std::uint16_t kInSetInter[9] = {
    0x0001,  // DCTONLY:          DCT_DCT
    0x0813,  // WIDE_64:          0, 1, 4, 11
    0x0425,  // HIGH_64:          0, 2, 5, 10
    0x5e13,  // WIDE_32:          0, 1, 4, 9, 10, 11, 12, 14
    0xae25,  // HIGH_32:          0, 2, 5, 9, 10, 11, 13, 15
    0xffff,  // INTER_1:          all TX_TYPES
    0x0fff,  // INTER_2:          0..11
    0x0201,  // DCT_IDTX:         0, 9
    0x0e01,  // DCT_IDTX_IDDCT:   0, 9, 10, 11
};

InterSet set_for(const txv::TxParams& params, int reduced_tx_set) {
  // §5.20.8.3 get_tx_set(txSz, plane) with is_inter = 1; plane 0 uses
  // reduced_tx_set, chroma passes enable_chroma_dctonly (0 here).  The M2
  // counterpart clamps out-of-range values to 0; do the same.
  if (reduced_tx_set < 0 || reduced_tx_set > 3) reduced_tx_set = 0;
  const int w = params.coded_w;
  const int h = params.coded_h;
  const int sqr = std::min(w, h);
  const int sqr_up = std::max(w, h);
  if (sqr_up > 32) {
    if (sqr >= 32) return InterSet::kDctOnly;
    return w > h ? InterSet::kWide64 : InterSet::kHigh64;
  }
  if (sqr_up == 32 && sqr != 32) {
    return w > h ? InterSet::kWide32 : InterSet::kHigh32;
  }
  if (sqr_up == 32 || reduced_tx_set == 1 || reduced_tx_set == 2) {
    return InterSet::kDctIdtx;
  }
  if (reduced_tx_set == 3) return InterSet::kDctIdtxIddct;
  if (sqr_up == 16 && sqr == 16) return InterSet::kInter2;
  return InterSet::kInter1;
}

bool in_set(InterSet set, TxType type) {
  const int index = static_cast<int>(type);
  if (index < 0 || index >= 16) return false;
  return (kInSetInter[static_cast<int>(set)] &
          static_cast<std::uint16_t>(1U << index)) != 0;
}

bool is_1d_class(TxType type) {
  switch (type) {
    case TxType::kVdct:
    case TxType::kVadst:
    case TxType::kVflipadst:
    case TxType::kHdct:
    case TxType::kHadst:
    case TxType::kHflipadst:
      return true;
    default:
      return false;
  }
}

}  // namespace

const char* status_name(Status status) {
  switch (status) {
    case Status::kOk:
      return "ok";
    case Status::kDisabled:
      return "disabled";
    case Status::kNotInter:
      return "not-inter";
    case Status::kIntrabc:
      return "intrabc";
    case Status::kLossless:
      return "lossless";
    case Status::kUnsupportedSize:
      return "unsupported-size";
    case Status::kTypeNotInSet:
      return "type-not-in-set";
    case Status::kChromaDctOnly:
      return "chroma-dct-only";
    case Status::kNoDdtComponent:
      return "no-ddt-component";
    case Status::kIstConflict:
      return "ist-conflict";
  }
  return "?";
}

bool tx_class_is_2d(TxType type) { return !is_1d_class(type); }

TxType plane_tx_type(const BlockSignal& signal) {
  const auto params = txv::make_tx_params(
      static_cast<txv::TxSize>(signal.tx_size), TxType::kDctDct);
  if (!params.has_value()) return TxType::kDctDct;
  if (signal.plane == 0) return static_cast<TxType>(signal.tx_type);
  if (signal.enable_chroma_dctonly) return TxType::kDctDct;
  // §5.20.7.29: chroma reuses the luma TxType only when it is in the chroma
  // size's set; get_tx_set(txSz, plane) for chroma uses
  // reducedTxSet = enable_chroma_dctonly = 0 here.
  const InterSet set = set_for(*params, /*reduced_tx_set=*/0);
  const TxType luma = static_cast<TxType>(signal.tx_type);
  if (!in_set(set, luma)) return TxType::kDctDct;
  return luma;
}

Validation validate(const BlockSignal& signal) {
  Validation result;
  result.plane_tx_type = plane_tx_type(signal);
  result.tcq_eligible = signal.allow_tcq && signal.plane == 0 &&
                        !signal.lossless && !signal.fsc_mode &&
                        tx_class_is_2d(result.plane_tx_type);
  result.qm_eligible =
      signal.using_qmatrix && result.plane_tx_type < TxType::kIdtx;
  result.ist_eligible = !signal.sec_tx_type;

  if (!signal.enable_inter_ddt) {
    result.status = Status::kDisabled;
    result.reason = "enable_inter_ddt = 0";
    return result;
  }
  if (!signal.is_inter) {
    result.status = Status::kNotInter;
    result.reason = "is_inter = 0";
    return result;
  }
  if (signal.use_intrabc) {
    result.status = Status::kIntrabc;
    result.reason = "use_intrabc = 1";
    return result;
  }
  if (signal.lossless || signal.fsc_mode) {
    result.status = Status::kLossless;
    result.reason = signal.fsc_mode ? "fsc_mode = 1" : "Lossless = 1";
    return result;
  }

  const auto params = txv::make_tx_params(
      static_cast<txv::TxSize>(signal.tx_size), result.plane_tx_type);
  if (!params.has_value()) {
    result.status = Status::kUnsupportedSize;
    result.reason = "not one of the 25 TX sizes";
    return result;
  }

  if (signal.plane == 0) {
    const InterSet set = set_for(*params, signal.reduced_tx_set);
    if (!in_set(set, result.plane_tx_type)) {
      result.status = Status::kTypeNotInSet;
      result.reason =
          "luma TxType is not in Tx_Type_In_Set_Inter for this size/set";
      return result;
    }
  } else if (signal.enable_chroma_dctonly) {
    result.status = Status::kChromaDctOnly;
    result.reason = "enable_chroma_dctonly = 1 forces chroma to DCT_DCT";
    return result;
  }

  Context ctx;
  ctx.enable_inter_ddt = true;
  ctx.is_inter = true;
  ctx.use_intrabc = false;
  ctx.lossless = false;
  ctx.bit_depth = signal.bit_depth;
  result.plan = make_plan(*params, ctx);
  result.ddt_active = plan_has_ddt(*params, ctx);

  if (signal.sec_tx_type && result.ddt_active) {
    result.status = Status::kIstConflict;
    result.reason =
        "inter IST requires TxType == DCT_DCT, which has no DDT component";
    return result;
  }

  if (!result.ddt_active) {
    result.status = Status::kNoDdtComponent;
    result.reason = "no ADST/FDST component at adjusted length 8 or 16";
    return result;
  }

  result.status = Status::kOk;
  return result;
}

}  // namespace av2enc::video::ddt

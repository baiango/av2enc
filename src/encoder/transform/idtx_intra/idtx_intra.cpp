// See idtx_intra.h for the spec contract.
#include "encoder/transform/idtx_intra/idtx_intra.h"

#include "encoder/tx_type/tx_type.h"
#include "encoder/tx_type/tx_type_tables.h"

#include <cstdio>
#include <utility>

namespace av2enc::idtx {
namespace {

using txtype::kIntraTxTypes;
using txtype::kMdIdxToType;
using txtype::kSizeClass;
using txtype::kTxSizeSqr;
using txtype::kTxSizeSqrUp;
using txtype::kTxTypeInSetIntra;
using txtype::kTxTypeInvLong;

// §5.20.7.29 Mode_To_Txfm[UV_INTRA_MODES_CFL_ALLOWED=14].
constexpr TxType kModeToTxfm[14] = {
    TxType::kDctDct,      // DC_PRED
    TxType::kAdstDct,     // V_PRED
    TxType::kDctAdst,     // H_PRED
    TxType::kDctDct,      // D45_PRED
    TxType::kAdstAdst,    // D135_PRED
    TxType::kAdstDct,     // D113_PRED
    TxType::kDctAdst,     // D157_PRED
    TxType::kDctAdst,     // D203_PRED
    TxType::kAdstDct,     // D67_PRED
    TxType::kAdstAdst,    // SMOOTH_PRED
    TxType::kAdstDct,     // SMOOTH_V_PRED
    TxType::kDctAdst,     // SMOOTH_H_PRED
    TxType::kAdstAdst,    // PAETH_PRED
    TxType::kDctDct,      // UV_CFL_PRED
};

bool set_row(Set set, int* row) {
  const int value = static_cast<int>(set);
  if (set == Set::kDctIdtx) {
    *row = -1;  // forced DCT_DCT, never read from the set table
    return true;
  }
  if (value < 0 || value > 6) return false;
  *row = value;
  return true;
}

bool fail(std::string* error, const char* message) {
  if (error != nullptr) *error = message;
  return false;
}

bool fail_with(std::string* error, std::string message) {
  if (error != nullptr) *error = std::move(message);
  return false;
}

int add_candidate(TypeSet* out, TxType type, int symbol, bool flag,
                  bool codes_flag) {
  if (out->count >= 8) return 0;
  Candidate& c = out->candidates[out->count++];
  c.type = type;
  c.symbol = symbol;
  c.is_long_side_dct = flag;
  c.codes_long_flag = codes_flag;
  return 1;
}

bool identity_allowed(const SeqFlags& flags) {
  return derive_sequence_flags(flags).idtx_allowed;
}

bool is_directional(int mode) { return mode >= 1 && mode <= 8; }

}  // namespace

DerivedSeqFlags derive_sequence_flags(const SeqFlags& flags) {
  DerivedSeqFlags out;
  // §5.4.8: `if (enable_fsc) { enable_idtx_intra = 1 } else { ... f(1) }`.
  out.idtx_bit_coded = !flags.enable_fsc;
  out.enable_idtx_intra = flags.enable_fsc || flags.enable_idtx_intra;
  out.idtx_allowed = out.enable_idtx_intra;
  return out;
}

const char* set_name(Set set) {
  switch (set) {
    case Set::kDctOnly:
      return "TX_SET_DCTONLY";
    case Set::kWide64:
      return "TX_SET_WIDE_64";
    case Set::kHigh64:
      return "TX_SET_HIGH_64";
    case Set::kWide32:
      return "TX_SET_WIDE_32";
    case Set::kHigh32:
      return "TX_SET_HIGH_32";
    case Set::kIntra1:
      return "TX_SET_INTRA_1";
    case Set::kIntra2:
      return "TX_SET_INTRA_2";
    case Set::kDctIdtx:
      return "TX_SET_DCT_IDTX";
  }
  return "?";
}

int tx_class(TxType type) {
  // §5.20.7.29 get_tx_class().
  switch (type) {
    case TxType::kVdct:
    case TxType::kVadst:
    case TxType::kVflipadst:
      return 2;  // TX_CLASS_VERT
    case TxType::kHdct:
    case TxType::kHadst:
    case TxType::kHflipadst:
      return 1;  // TX_CLASS_HORIZ
    default:
      return 0;  // TX_CLASS_2D, including IDTX
  }
}

bool intra_set_allows(Set set, TxType type) {
  int row = 0;
  if (!set_row(set, &row)) return false;
  const int value = static_cast<int>(type);
  if (value < 0 || value >= 16) return false;
  if (row < 0) return type == TxType::kDctDct;  // DCT_IDTX, intra override
  return kTxTypeInSetIntra[row][value] != 0;
}

bool get_tx_set(const Query& query, Set* set) {
  if (set == nullptr) return false;
  const int idx = txv::tx_size_index(query.coded_w, query.coded_h);
  if (idx < 0) return false;
  if (query.flags.reduced_tx_set < 0 || query.flags.reduced_tx_set > 3) return false;
  const int sqr = kTxSizeSqr[idx];
  const int sqr_up = kTxSizeSqrUp[idx];

  // §5.20.8.3, TX_32X32 = 3, TX_64X64 = 4.
  if (sqr_up > 3) {
    if (sqr >= 3) {
      *set = Set::kDctOnly;
    } else {
      *set = query.coded_w > query.coded_h ? Set::kWide64 : Set::kHigh64;
    }
    return true;
  }
  if (sqr_up == 3 && sqr != 3) {
    *set = query.coded_w > query.coded_h ? Set::kWide32 : Set::kHigh32;
    return true;
  }
  if (sqr_up == 3) {  // !is_inter
    *set = Set::kDctOnly;
    return true;
  }
  // reducedTxSet = plane == 0 ? reduced_tx_set : enable_chroma_dctonly.
  const int reduced =
      query.plane == 0 ? query.flags.reduced_tx_set
                       : (query.flags.enable_chroma_dctonly ? 1 : 0);
  if (sqr_up == 3 || reduced == 1) {
    *set = Set::kIntra2;
  } else if (reduced == 2) {
    *set = Set::kDctIdtx;
  } else if (reduced == 3) {
    *set = Set::kIntra2;
  } else {
    *set = Set::kIntra1;
  }
  return true;
}

bool enumerate_types(const Query& query, TypeSet* out, std::string* error) {
  if (out == nullptr) return fail(error, "null output set");
  Set set = Set::kDctOnly;
  if (!get_tx_set(query, &set)) {
    return fail(error, "tx size or reduced_tx_set is out of range");
  }
  out->set = set;
  out->count = 0;

  // transform_type(): eob == 1 is DCT_DCT before the set symbol; eob == 0 is
  // txb_skip, so transform_type() is not parsed at all.  `reduced_tx_set == 2`
  // skips the whole type block for an intra block at every size (the
  // `!(reduced_tx_set == 2 && is_inter == 0)` guard), and the DCT_IDTX set is
  // likewise forced to DCT_DCT.
  if (query.eob <= 1 ||
      (query.plane == 0 && query.flags.reduced_tx_set == 2) ||
      set == Set::kDctOnly || set == Set::kDctIdtx) {
    add_candidate(out, TxType::kDctDct, -1, false, false);
    return true;
  }

  const bool idtx = identity_allowed(query.flags);
  const int idx = txv::tx_size_index(query.coded_w, query.coded_h);
  const int sqr = kTxSizeSqr[idx];

  switch (set) {
    case Set::kWide64:
    case Set::kHigh64: {
      const int orientation = set == Set::kWide64 ? 0 : 1;
      // §5.20.8.2: WIDE_64/HIGH_64 imply is_long_side_dct = 1.
      for (int symbol = 0; symbol < 4; ++symbol) {
        const auto type =
            static_cast<TxType>(kTxTypeInvLong[1][orientation][symbol]);
        if (is_identity_type(type) && !idtx) continue;
        add_candidate(out, type, symbol, true, false);
      }
      if (out->count == 0) {
        return fail(error, "empty long-64 set");
      }
      return true;
    }

    case Set::kWide32:
    case Set::kHigh32: {
      const int orientation = set == Set::kWide32 ? 0 : 1;
      // The flag is coded for both 32 sets; list flag 0 then 1 in symbol order.
      for (int flag = 0; flag < 2; ++flag) {
        for (int symbol = 0; symbol < 4; ++symbol) {
          const auto type =
              static_cast<TxType>(kTxTypeInvLong[flag][orientation][symbol]);
          if (is_identity_type(type) && !idtx) continue;
          add_candidate(out, type, symbol, flag == 1, true);
        }
      }
      if (out->count == 0) {
        return fail(error, "empty long-32 set");
      }
      return true;
    }

    case Set::kIntra1:
    case Set::kIntra2: {
      const int n = set == Set::kIntra1 ? kIntraTxTypes : 2;
      const int size_class = kSizeClass[idx];
      if (query.mapped_mode < 0 ||
          query.mapped_mode >= txtype::kNumIntraModes) {
        return fail(error, "mapped intra mode is out of range");
      }
      for (int symbol = 0; symbol < n; ++symbol) {
        const auto type = static_cast<TxType>(
            kMdIdxToType[size_class][query.mapped_mode][symbol]);
        if (is_identity_type(type) && !idtx) continue;
        add_candidate(out, type, symbol, false, false);
      }
      if (out->count == 0) {
        return fail(error, "empty intra set");
      }
      (void)sqr;
      return true;
    }

    case Set::kDctOnly:
    case Set::kDctIdtx:
      break;
  }
  add_candidate(out, TxType::kDctDct, -1, false, false);
  return true;
}

bool plan_type(const Query& query, TxType requested, TypePlan* plan,
               std::string* error) {
  if (plan == nullptr) return fail(error, "null plan");
  if (!enumerate_types(query, &plan->set, error)) return false;
  plan->type_syntax = false;
  plan->symbol = -1;
  plan->is_long_side_dct = false;
  plan->codes_long_flag = false;

  if (query.plane > 0) {
    // Chroma never codes intra_tx_type: compute_tx_type() derives it.
    Set set = Set::kDctOnly;
    TxType derived = TxType::kDctDct;
    if (!derive_chroma_type(query.uv_mode, query.uv_angle_delta,
                            query.coded_w, query.coded_h, query.flags, &derived,
                            &set)) {
      return fail(error, "invalid chroma mode or size");
    }
    plan->type = derived;
    plan->set.set = set;
    if (requested != derived) {
      return fail_with(error,
                       "chroma type is derived as " +
                           std::string(txv::tx_type_name(derived)) +
                           ", cannot signal " +
                           std::string(txv::tx_type_name(requested)));
    }
    return true;
  }

  if (query.eob <= 1) {
    plan->type = TxType::kDctDct;
    if (requested != TxType::kDctDct) {
      return fail_with(
          error, "eob <= 1 forces DCT_DCT; cannot signal " +
                     std::string(txv::tx_type_name(requested)));
    }
    return true;
  }

  for (int i = 0; i < plan->set.count; ++i) {
    const Candidate& c = plan->set.candidates[i];
    if (c.type != requested) continue;
    plan->type = c.type;
    plan->symbol = c.symbol;
    plan->is_long_side_dct = c.is_long_side_dct;
    plan->codes_long_flag = c.codes_long_flag;
    plan->type_syntax = c.symbol >= 0;
    return true;
  }

  if (is_identity_type(requested) && !identity_allowed(query.flags)) {
    return fail_with(error,
                     std::string(txv::tx_type_name(requested)) +
                         " is an intra identity type but enable_idtx_intra == 0"
                         " (and enable_fsc == 0)");
  }
  return fail_with(error,
                   std::string(txv::tx_type_name(requested)) +
                       " is not in " + set_name(plan->set.set) +
                       " for this size/mode/reduced_tx_set");
}

TxType default_type(const Query& query) {
  if (query.plane > 0) {
    TxType type = TxType::kDctDct;
    Set set = Set::kDctOnly;
    if (derive_chroma_type(query.uv_mode, query.uv_angle_delta, query.coded_w,
                           query.coded_h, query.flags, &type, &set)) {
      return type;
    }
    return TxType::kDctDct;
  }
  return TxType::kDctDct;
}

TxType mode_to_txfm(int uv_mode) {
  if (uv_mode < 0 || uv_mode > 13) return TxType::kDctDct;
  return kModeToTxfm[uv_mode];
}

bool derive_chroma_type(int uv_mode, int uv_angle_delta, int coded_w,
                        int coded_h, const SeqFlags& flags, TxType* type,
                        Set* set_out) {
  if (type == nullptr) return false;
  Query query;
  query.coded_w = coded_w;
  query.coded_h = coded_h;
  query.plane = 1;
  query.flags = flags;
  Set set = Set::kDctOnly;
  if (!get_tx_set(query, &set)) return false;
  if (set_out != nullptr) *set_out = set;

  if (flags.enable_chroma_dctonly) {
    *type = TxType::kDctDct;
    return true;
  }
  if (uv_mode < 0 || uv_mode > 13) return false;
  if (uv_angle_delta < -3 || uv_angle_delta > 3) return false;

  int mapped = uv_mode;
  if (is_directional(uv_mode)) {
    mapped = txtype::map_intra_mode(uv_mode, uv_angle_delta, coded_w, coded_h);
    if (mapped < 0) return false;
  }
  TxType derived = kModeToTxfm[mapped];
  // §5.20.7.29: if the derived type is not in the set, use DCT_DCT.
  *type = intra_set_allows(set, derived) ? derived : TxType::kDctDct;
  return true;
}

ToolUse derive_tool_use(const Query& query, TxType plane_type, int seg_level,
                        bool qm_data_present, bool lossless,
                        bool fsc_mode) {
  ToolUse use;
  use.tx_class = tx_class(plane_type);

  const DerivedSeqFlags seq = derive_sequence_flags(query.flags);
  // §5.20.5.5 allow_fsc_intra().
  use.allow_fsc_intra = seq.enable_idtx_intra && query.coded_w <= kFscMax &&
                        query.coded_h <= kFscMax;

  // §7.14.4 useFsc.  This planner is intra-only (is_inter == 0).
  use.use_fsc = query.flags.enable_fsc && plane_type == TxType::kIdtx &&
                query.plane == 0 && fsc_mode;

  // §7.14.4 useQm: PlaneTxType < IDTX excludes every identity type.
  use.use_qm = query.flags.using_qmatrix &&
               static_cast<int>(plane_type) < static_cast<int>(TxType::kIdtx) &&
               seg_level < kNumCustomQms;

  // §5.20.8.2 IST applicability for an intra block.
  const bool large = query.coded_w >= 8 && query.coded_h >= 8;
  if (!large) {
    use.ist_eob_limit = kIst4x4Height;
  } else if ((query.coded_w == 8 && query.coded_h == 8) ||
             plane_type == TxType::kAdstAdst) {
    use.ist_eob_limit = kIst8x8HeightRed;
  } else {
    use.ist_eob_limit = kIst8x8Height;
  }
  use.use_ist = query.flags.enable_intra_ist && !lossless && query.eob != 1 &&
                (plane_type == TxType::kAdstAdst ||
                 plane_type == TxType::kDctDct) &&
                query.y_mode != 12 /* PAETH_PRED */ &&
                query.eob <= use.ist_eob_limit;

  // §7.14.4: TCQ extra shift is keyed on TX_CLASS_2D, which includes IDTX.
  use.tcq_extra_shift = query.flags.allow_tcq && query.plane == 0 &&
                        !lossless && use.tx_class == 0 && !use.use_fsc;

  use.parity_hiding = query.flags.enable_parity_hiding && !lossless &&
                      query.plane == 0 && plane_type != TxType::kIdtx;
  (void)qm_data_present;  // useUserQm, not useQm; see README
  return use;
}

}  // namespace av2enc::idtx

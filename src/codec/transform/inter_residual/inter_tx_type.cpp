#include "codec/transform/inter_residual/inter_tx_type.h"

#include <algorithm>
#include <array>
#include <cstddef>

namespace av2enc::video::residual {
namespace {

using txv::TxType;

constexpr TxType kDctDct = TxType::kDctDct;
constexpr TxType kAdstDct = TxType::kAdstDct;
constexpr TxType kDctAdst = TxType::kDctAdst;
constexpr TxType kAdstAdst = TxType::kAdstAdst;
constexpr TxType kFlipadstDct = TxType::kFlipadstDct;
constexpr TxType kDctFlipadst = TxType::kDctFlipadst;
constexpr TxType kFlipadstFlipadst = TxType::kFlipadstFlipadst;
constexpr TxType kFlipadstAdst = TxType::kFlipadstAdst;
constexpr TxType kAdstFlipadst = TxType::kAdstFlipadst;
constexpr TxType kIdtx = TxType::kIdtx;
constexpr TxType kVdct = TxType::kVdct;
constexpr TxType kHdct = TxType::kHdct;
constexpr TxType kVadst = TxType::kVadst;
constexpr TxType kHadst = TxType::kHadst;
constexpr TxType kVflipadst = TxType::kVflipadst;
constexpr TxType kHflipadst = TxType::kHflipadst;

// Tx_Type_Inter_Inv_Set1[16] (§5.20.8.2).  inter_tx_type * 8 + offset.
constexpr TxType kInterInvSet1[16] = {
    kIdtx,     kVdct,    kHdct,    kVadst,   kHadst,   kVflipadst,
    kHflipadst, kDctDct, kAdstDct, kDctAdst, kFlipadstDct, kDctFlipadst,
    kAdstAdst, kFlipadstFlipadst, kAdstFlipadst, kFlipadstAdst};

// Tx_Type_Inter_Inv_Set2[12] (§5.20.8.2).  inter_tx_type * 8 + offset; the
// second half only has four codable offsets.
constexpr TxType kInterInvSet2[12] = {
    kIdtx,     kVdct,    kHdct,    kDctDct,   kAdstDct,  kDctAdst,
    kFlipadstDct, kDctFlipadst, kAdstAdst, kFlipadstFlipadst, kAdstFlipadst,
    kFlipadstAdst};

constexpr TxType kInterInvSet3[2] = {kIdtx, kDctDct};
constexpr TxType kInterInvSet4[4] = {kDctDct, kVdct, kHdct, kIdtx};

// Tx_Type_Inv_Long[is_long_side_dct][orientation][4] (§5.20.8.2).
constexpr TxType kInvLong[2][2][4] = {
    {{kVdct, kVadst, kVflipadst, kIdtx},
     {kHdct, kHadst, kHflipadst, kIdtx}},
    {{kDctDct, kAdstDct, kFlipadstDct, kHdct},
     {kDctDct, kDctAdst, kDctFlipadst, kVdct}},
};

// Tx_Type_In_Set_Inter[TX_SET_TYPES_INTER = 9][TX_TYPES = 16] (§5.20.7.29).
constexpr std::uint8_t kInSetInter[9][16] = {
    {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
    {1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
    {1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0},
    {1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
    {1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0},
};

int floor_log2(int value) {
  int result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}

// Dimensions of a coded TX size.  txv::make_tx_params() derives the same pair;
// using it here keeps one source of truth for the 25-size table.
bool tx_dims(TxSize tx_size, int* w, int* h) {
  const auto params = txv::make_tx_params(tx_size, TxType::kDctDct);
  if (!params.has_value()) return false;
  *w = params->coded_w;
  *h = params->coded_h;
  return true;
}

int eob_diagonal(int eob, TxSize tx_size) {
  int w = 0;
  int h = 0;
  if (!tx_dims(tx_size, &w, &h)) return 0;
  const int bwl = std::min(floor_log2(w), 5);
  const int eoby = (eob - 1) >> bwl;
  const int eobx = (eob - 1) - (eoby << bwl);
  return eobx + eoby;
}

// tx_variants maps FLIPADST_ADST and ADST_FLIPADST to each other's 1D types
// (`kTransform1dType[7]` / `[8]` in src/encoder/tx_variants/tx_variants.cpp).
// The decoder keys the 1D transforms off the signalled TX_TYPE name, so
// signalling either of them reconstructs with the transposed transform and
// breaks bit-exactness (verified with dav2d: every other inter type decodes
// sample-exactly, these two do not).  They are withheld until the shared
// component swaps the two rows; both are 2D ADST/FLIPADST combinations, so the
// loss is two of the nine INTER_1/INTER_2 candidates.
bool txv_1d_mapping_matches_spec(TxType type) {
  return type != kFlipadstAdst && type != kAdstFlipadst;
}

bool codes_type(InterTxSet set) {
  return set != InterTxSet::kDctOnly;
}

// Fills one candidate for a set that codes a single `inter_tx_type` symbol.
bool make_single_symbol_candidate(TxSize tx_size, InterTxSet set, TxType type,
                                  InterTxCandidate* out) {
  const TxType* table = nullptr;
  int count = 0;
  if (set == InterTxSet::kDctIdtx) {
    table = kInterInvSet3;
    count = 2;
  } else if (set == InterTxSet::kDctIdtxIddct) {
    table = kInterInvSet4;
    count = 4;
  } else {
    return false;
  }
  for (int i = 0; i < count; ++i) {
    if (table[i] != type) continue;
    const auto params = txv::make_tx_params(tx_size, type);
    if (!params.has_value()) return false;
    out->type = type;
    out->is_long_side_dct = true;
    out->codes_long_flag = false;
    out->symbol = static_cast<std::int8_t>(i);
    out->offset_symbol = -1;
    out->params = *params;
    return true;
  }
  return false;
}

bool make_long_candidate(TxSize tx_size, InterTxSet set, bool is_long_side_dct,
                         TxType type, InterTxCandidate* out) {
  int w = 0;
  int h = 0;
  if (!tx_dims(tx_size, &w, &h)) return false;
  const int orientation =
      (set == InterTxSet::kWide64 || set == InterTxSet::kWide32) ? 0 : 1;
  const TxType* row = kInvLong[is_long_side_dct ? 1 : 0][orientation];
  for (int i = 0; i < 4; ++i) {
    if (row[i] != type) continue;
    const auto params = txv::make_tx_params(tx_size, type);
    if (!params.has_value()) return false;
    out->type = type;
    out->is_long_side_dct = is_long_side_dct;
    out->codes_long_flag = set == InterTxSet::kWide32 ||
                           set == InterTxSet::kHigh32;
    out->symbol = static_cast<std::int8_t>(i);
    out->offset_symbol = -1;
    out->params = *params;
    return true;
  }
  return false;
}

}  // namespace

const char* inter_tx_set_name(InterTxSet set) {
  switch (set) {
    case InterTxSet::kDctOnly:
      return "TX_SET_DCTONLY";
    case InterTxSet::kWide64:
      return "TX_SET_WIDE_64";
    case InterTxSet::kHigh64:
      return "TX_SET_HIGH_64";
    case InterTxSet::kWide32:
      return "TX_SET_WIDE_32";
    case InterTxSet::kHigh32:
      return "TX_SET_HIGH_32";
    case InterTxSet::kInter1:
      return "TX_SET_INTER_1";
    case InterTxSet::kInter2:
      return "TX_SET_INTER_2";
    case InterTxSet::kDctIdtx:
      return "TX_SET_DCT_IDTX";
    case InterTxSet::kDctIdtxIddct:
      return "TX_SET_DCT_IDTX_IDDCT";
  }
  return "TX_SET_INVALID";
}

int tx_size_sqr(TxSize tx_size) {
  int w = 0;
  int h = 0;
  if (!tx_dims(tx_size, &w, &h)) return -1;
  return floor_log2(std::min(w, h)) - 2;
}

int tx_size_sqr_up(TxSize tx_size) {
  int w = 0;
  int h = 0;
  if (!tx_dims(tx_size, &w, &h)) return -1;
  return floor_log2(std::max(w, h)) - 2;
}

InterTxSet inter_tx_set(TxSize tx_size, int reduced_tx_set) {
  int w = 0;
  int h = 0;
  if (!tx_dims(tx_size, &w, &h)) return InterTxSet::kDctOnly;
  if (reduced_tx_set < 0 || reduced_tx_set > 3) reduced_tx_set = 0;
  const int sqr = tx_size_sqr(tx_size);
  const int sqr_up = tx_size_sqr_up(tx_size);
  // TX_32X32 = 3, TX_64X64 = 4 in the §6.19.6.1 enum.
  if (sqr_up > 3) {
    if (sqr >= 3) return InterTxSet::kDctOnly;
    return w > h ? InterTxSet::kWide64 : InterTxSet::kHigh64;
  }
  if (sqr_up == 3 && sqr != 3) {
    return w > h ? InterTxSet::kWide32 : InterTxSet::kHigh32;
  }
  // is_inter == 1 from here on, so the `!is_inter && sqrUp == 32` DCTONLY
  // branch never applies.
  const int reduced = reduced_tx_set;
  if (sqr_up == 3 || reduced == 1) return InterTxSet::kDctIdtx;
  if (reduced == 2) return InterTxSet::kDctIdtx;
  if (reduced == 3) return InterTxSet::kDctIdtxIddct;
  return sqr == 2 ? InterTxSet::kInter2 : InterTxSet::kInter1;
}

bool inter_tx_type_is_ddt(TxType type) {
  switch (type) {
    case TxType::kIdtx:
    case TxType::kVdct:
    case TxType::kHdct:
    case TxType::kVadst:
    case TxType::kHadst:
    case TxType::kVflipadst:
    case TxType::kHflipadst:
      return true;
    default:
      return false;
  }
}

bool inter_tx_type_in_set(InterTxSet set, TxType type) {
  const int index = static_cast<int>(type);
  const int row = static_cast<int>(set);
  if (index < 0 || index >= 16 || row < 0 || row >= 9) return false;
  return kInSetInter[row][index] != 0;
}

bool inter_tx_candidates(TxSize tx_size, int reduced_tx_set,
                         bool enable_inter_ddt, InterTxSetInfo* out) {
  if (out == nullptr) return false;
  *out = InterTxSetInfo{};
  const InterTxSet set = inter_tx_set(tx_size, reduced_tx_set);
  out->set = set;
  if (!codes_type(set)) {
    // DCTONLY: the transform type is DCT_DCT and no symbol is coded.
    const auto params = txv::make_tx_params(tx_size, TxType::kDctDct);
    if (!params.has_value()) return false;
    InterTxCandidate& candidate = out->candidates[0];
    candidate.type = TxType::kDctDct;
    candidate.is_long_side_dct = true;
    candidate.codes_long_flag = false;
    candidate.symbol = -1;
    candidate.offset_symbol = -1;
    candidate.params = *params;
    out->count = 1;
    return true;
  }

  if (set == InterTxSet::kInter1 || set == InterTxSet::kInter2) {
    const TxType* table =
        set == InterTxSet::kInter1 ? kInterInvSet1 : kInterInvSet2;
    const int table_len = set == InterTxSet::kInter1 ? 16 : 12;
    for (int type_symbol = 0; type_symbol < 2; ++type_symbol) {
      const int offsets = set == InterTxSet::kInter1 ? 8 : 8;
      for (int offset = 0; offset < offsets; ++offset) {
        const int index = type_symbol * 8 + offset;
        if (index >= table_len) break;
        const TxType type = table[index];
        // §5.20.7.29: the decoder aliases a type outside Tx_Type_In_Set_Inter
        // to DCT_DCT (INTER_2 excludes ADST_FLIPADST), so it is not codable.
        if (!inter_tx_type_in_set(set, type)) continue;
        if (!enable_inter_ddt && inter_tx_type_is_ddt(type)) continue;
        if (!txv_1d_mapping_matches_spec(type)) continue;
        if (out->count >= kMaxInterTxCandidates) break;
        const auto params = txv::make_tx_params(tx_size, type);
        if (!params.has_value()) continue;
        InterTxCandidate& candidate =
            out->candidates[static_cast<std::size_t>(out->count++)];
        candidate.type = type;
        candidate.is_long_side_dct = true;
        candidate.codes_long_flag = false;
        candidate.symbol = static_cast<std::int8_t>(type_symbol);
        candidate.offset_symbol = static_cast<std::int8_t>(offset);
        candidate.params = *params;
      }
    }
    return out->count > 0;
  }

  if (set == InterTxSet::kDctIdtx || set == InterTxSet::kDctIdtxIddct) {
    const TxType* table =
        set == InterTxSet::kDctIdtx ? kInterInvSet3 : kInterInvSet4;
    const int count = set == InterTxSet::kDctIdtx ? 2 : 4;
    for (int i = 0; i < count; ++i) {
      const TxType type = table[i];
      if (!inter_tx_type_in_set(set, type)) continue;
      if (!enable_inter_ddt && inter_tx_type_is_ddt(type)) continue;
      if (!txv_1d_mapping_matches_spec(type)) continue;
      InterTxCandidate candidate;
      if (!make_single_symbol_candidate(tx_size, set, type, &candidate)) {
        continue;
      }
      if (out->count >= kMaxInterTxCandidates) break;
      out->candidates[static_cast<std::size_t>(out->count++)] = candidate;
    }
    return out->count > 0;
  }

  // WIDE_32/HIGH_32 code is_long_side_dct (0 then 1); WIDE_64/HIGH_64 imply 1.
  const bool codes_flag = set == InterTxSet::kWide32 || set == InterTxSet::kHigh32;
  const int long_values = codes_flag ? 2 : 1;
  for (int flag = 0; flag < long_values; ++flag) {
    // The coded flag value 0 is enumerated first; the implied value for the
    // 64-wide sets is 1.
    const bool is_long = codes_flag ? (flag == 1) : true;
    for (int symbol = 0; symbol < 4; ++symbol) {
      if (out->count >= kMaxInterTxCandidates) break;
      const int orientation =
          (set == InterTxSet::kWide64 || set == InterTxSet::kWide32) ? 0 : 1;
      const TxType type = kInvLong[is_long ? 1 : 0][orientation][symbol];
      if (!inter_tx_type_in_set(set, type)) continue;
      if (!enable_inter_ddt && inter_tx_type_is_ddt(type)) continue;
      if (!txv_1d_mapping_matches_spec(type)) continue;
      InterTxCandidate candidate;
      if (!make_long_candidate(tx_size, set, is_long, type, &candidate)) {
        continue;
      }
      out->candidates[static_cast<std::size_t>(out->count++)] = candidate;
    }
  }
  return out->count > 0;
}

bool inter_tx_symbol(TxSize tx_size, int reduced_tx_set, bool enable_inter_ddt,
                     TxType type, InterTxCandidate* out) {
  if (out == nullptr) return false;
  const InterTxSet set = inter_tx_set(tx_size, reduced_tx_set);
  if (!inter_tx_type_in_set(set, type)) return false;
  if (!enable_inter_ddt && inter_tx_type_is_ddt(type)) return false;
  if (!txv_1d_mapping_matches_spec(type)) return false;
  if (!codes_type(set)) {
    if (type != TxType::kDctDct) return false;
    const auto params = txv::make_tx_params(tx_size, type);
    if (!params.has_value()) return false;
    out->type = type;
    out->is_long_side_dct = true;
    out->codes_long_flag = false;
    out->symbol = -1;
    out->offset_symbol = -1;
    out->params = *params;
    return true;
  }
  if (set == InterTxSet::kInter1 || set == InterTxSet::kInter2) {
    const TxType* table =
        set == InterTxSet::kInter1 ? kInterInvSet1 : kInterInvSet2;
    const int table_len = set == InterTxSet::kInter1 ? 16 : 12;
    for (int index = 0; index < table_len; ++index) {
      if (table[index] != type) continue;
      const auto params = txv::make_tx_params(tx_size, type);
      if (!params.has_value()) return false;
      out->type = type;
      out->is_long_side_dct = true;
      out->codes_long_flag = false;
      out->symbol = static_cast<std::int8_t>(index / 8);
      out->offset_symbol = static_cast<std::int8_t>(index % 8);
      out->params = *params;
      return true;
    }
    return false;
  }
  if (set == InterTxSet::kDctIdtx || set == InterTxSet::kDctIdtxIddct) {
    return make_single_symbol_candidate(tx_size, set, type, out);
  }
  // Long sets: only is_long_side_dct = 1 has non-DDT codable types; the
  // decision path never selects the DDT half.
  if (make_long_candidate(tx_size, set, /*is_long_side_dct=*/true, type, out)) {
    return true;
  }
  return make_long_candidate(tx_size, set, /*is_long_side_dct=*/false, type,
                             out);
}

int inter_tx_type_context(int eob, TxSize tx_size) {
  if (eob <= 0) return 0;
  int w = 0;
  int h = 0;
  if (!tx_dims(tx_size, &w, &h)) return 0;
  const int diag = eob_diagonal(eob, tx_size);
  if (diag < 2) return 1;
  const int limit = std::min(w, 32) + std::min(h, 32) - 4;
  if (diag > limit) return 2;
  return 0;
}

int inter_tx_type_offset_context(int eob, TxSize tx_size) {
  if (eob <= 0) return 0;
  int w = 0;
  int h = 0;
  if (!tx_dims(tx_size, &w, &h)) return 0;
  const int diag = eob_diagonal(eob, tx_size);
  if (diag < 2) return 1;
  // §8.3.2: the offset context threshold is NOT clamped to 32.
  if (diag > (w + h - 4)) return 2;
  return 0;
}

TxType inter_chroma_tx_type(TxType luma_tx_type, TxSize chroma_tx_size,
                            int reduced_tx_set, bool enable_chroma_dctonly) {
  if (enable_chroma_dctonly) return TxType::kDctDct;
  const InterTxSet set = inter_tx_set(chroma_tx_size, reduced_tx_set);
  if (set == InterTxSet::kDctOnly) return TxType::kDctDct;
  if (!inter_tx_type_in_set(set, luma_tx_type)) return TxType::kDctDct;
  return luma_tx_type;
}

}  // namespace av2enc::video::residual

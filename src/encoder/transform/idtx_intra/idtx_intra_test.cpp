// Tests for the intra identity-transform planner, the §5.20.8.2 type-syntax
// codec and the §5.4.8 enable_idtx_intra conditional.
//
// Coverage:
//   1. §5.4.8 enable_fsc -> enable_idtx_intra inference and bit presence.
//   2. Exhaustive type x size x plane legality for every reduced_tx_set,
//      enable_idtx_intra and enable_chroma_dctonly combination.
//   3. Chroma non-DCT derivation (Mode_To_Txfm + set filter) for every size,
//      UV mode and delta.
//   4. QM (`tx_type < IDTX`), IST (DCT_DCT / ADST_ADST) and TCQ-class
//      interactions, including the `TX_CLASS_2D` identity case.
//   5. Identity reconstruction parity against `txv::inverse_transform` for all
//      seven identity types at all 25 sizes (independent §7.15.4.1 process).
//   6. §5.20.8.2 writer/reader round trip with CDF adaptation + goldens.
#include "encoder/transform/idtx_intra/idtx_intra.h"
#include "encoder/transform/idtx_intra/seq_bits.h"
#include "encoder/transform/idtx_intra/type_syntax.h"

#include "encoder/m1/m1_headers.h"
#include "encoder/tx_type/tx_type_tables.h"
#include "encoder/tx_variants/tx_variants.h"
#include "idtx_goldens.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <string>
#include <vector>

namespace {

namespace idtx = av2enc::idtx;
using av2enc::txv::TxParams;
using av2enc::txv::TxType;

int g_checks = 0;
int g_failures = 0;

#define CHECK(cond)                                                       \
  do {                                                                    \
    ++g_checks;                                                           \
    if (!(cond)) {                                                        \
      ++g_failures;                                                       \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);         \
    }                                                                     \
  } while (0)

#define CHECK_MSG(cond, message)                                          \
  do {                                                                    \
    ++g_checks;                                                           \
    if (!(cond)) {                                                        \
      ++g_failures;                                                       \
      std::printf("FAIL %s:%d  %s (%s)\n", __FILE__, __LINE__, #cond,     \
                  std::string(message).c_str());                          \
    }                                                                     \
  } while (0)

std::uint64_t fnv1a(std::uint64_t hash, const std::uint8_t* data,
                    std::size_t size) {
  for (std::size_t i = 0; i < size; ++i) {
    hash ^= data[i];
    hash *= 1099511628211ULL;
  }
  return hash;
}

// ---------------------------------------------------------------------------
// 1. §5.4.8 conditional bit
// ---------------------------------------------------------------------------
void test_sequence_flags() {
  idtx::SeqFlags off;
  CHECK(!idtx::derive_sequence_flags(off).enable_idtx_intra);
  CHECK(idtx::derive_sequence_flags(off).idtx_bit_coded);

  idtx::SeqFlags explicit_on = off;
  explicit_on.enable_idtx_intra = true;
  CHECK(idtx::derive_sequence_flags(explicit_on).enable_idtx_intra);
  CHECK(idtx::derive_sequence_flags(explicit_on).idtx_bit_coded);

  idtx::SeqFlags fsc = off;
  fsc.enable_fsc = true;  // raw idtx bit is 0 but must be inferred to 1
  const idtx::DerivedSeqFlags fsc_derived = idtx::derive_sequence_flags(fsc);
  CHECK(fsc_derived.enable_idtx_intra);
  CHECK(!fsc_derived.idtx_bit_coded);

  idtx::TransformSeqConfig base;
  base.enable_chroma_dctonly = true;
  base.enable_intra_ist = false;

  idtx::TransformSeqBits info0;
  const std::vector<std::uint8_t> bits0 =
      idtx::write_transform_quant_entropy_config(base, &info0);
  CHECK(info0.idtx_bit_coded);
  CHECK(!info0.enable_idtx_intra);

  idtx::TransformSeqConfig on = base;
  on.enable_idtx_intra = true;
  idtx::TransformSeqBits info1;
  const std::vector<std::uint8_t> bits1 =
      idtx::write_transform_quant_entropy_config(on, &info1);
  CHECK(info1.bit_size == info0.bit_size);
  CHECK(info0.bit_size == 17);  // see golden below
  CHECK(bits0 != bits1);

  idtx::TransformSeqConfig inferred = base;
  inferred.enable_fsc = true;
  idtx::TransformSeqBits info2;
  const std::vector<std::uint8_t> bits2 =
      idtx::write_transform_quant_entropy_config(inferred, &info2);
  CHECK(!info2.idtx_bit_coded);
  CHECK(info2.enable_idtx_intra);
  CHECK(info2.bit_size + 1 == info0.bit_size);

  // Round trip for all combinations of the conditional and its neighbours.
  for (int mask = 0; mask < 32; ++mask) {
    idtx::TransformSeqConfig config;
    config.enable_fsc = (mask & 1) != 0;
    config.enable_idtx_intra = (mask & 2) != 0;
    config.enable_intra_ist = (mask & 4) != 0;
    config.enable_inter_ist = (mask & 8) != 0;
    config.enable_chroma_dctonly = (mask & 16) != 0;
    const std::vector<std::uint8_t> bytes =
        idtx::write_transform_quant_entropy_config(config);
    idtx::TransformSeqConfig read_back;
    read_back.monochrome = config.monochrome;
    read_back.single_picture_header = config.single_picture_header;
    std::size_t consumed = 0;
    CHECK(idtx::read_transform_quant_entropy_config(
        bytes.data(), bytes.size(), 0, &read_back, &consumed));
    CHECK(consumed <= bytes.size() * 8);
    CHECK(read_back.enable_fsc == config.enable_fsc);
    if (!config.enable_fsc) {
      CHECK(read_back.enable_idtx_intra == config.enable_idtx_intra);
    }
    CHECK(read_back.enable_intra_ist == config.enable_intra_ist);
    CHECK(read_back.enable_chroma_dctonly == config.enable_chroma_dctonly);
    CHECK(idtx::transform_seq_flags(read_back).enable_idtx_intra ==
          (config.enable_fsc || config.enable_idtx_intra));
  }

  // Golden: the §5.4.8 block bits for the M1-shaped all-off configuration.
  CHECK(info0.bit_size == kSeqBitsAllOffBits);
  CHECK(bits0 == std::vector<std::uint8_t>(std::begin(kSeqBitsAllOff),
                                           std::end(kSeqBitsAllOff)));

  // The incumbent M1 sequence-header writer codes exactly one bit for
  // enable_idtx_intra when enable_fsc is 0: the two headers differ in one bit
  // and have the same length.
  av2enc::m1::M1Config header_off;
  header_off.enable_idtx_intra = false;
  av2enc::m1::M1Config header_on = header_off;
  header_on.enable_idtx_intra = true;
  const std::vector<std::uint8_t> h0 =
      av2enc::m1::make_sequence_header_obu(header_off);
  const std::vector<std::uint8_t> h1 =
      av2enc::m1::make_sequence_header_obu(header_on);
  CHECK(h0.size() == h1.size());
  int differing_bits = 0;
  for (std::size_t i = 0; i < h0.size() && i < h1.size(); ++i) {
    for (int bit = 0; bit < 8; ++bit) {
      const int a = (h0[i] >> (7 - bit)) & 1;
      const int b = (h1[i] >> (7 - bit)) & 1;
      if (a != b) ++differing_bits;
    }
  }
  CHECK(differing_bits == 1);
}

// ---------------------------------------------------------------------------
// 2. Exhaustive type x size x plane legality
// ---------------------------------------------------------------------------
void test_exhaustive_legality() {
  const int eob_values[5] = {0, 1, 2, 7, 1024};
  std::uint64_t legality_hash = 1469598103934665603ULL;
  long long enumerated = 0;

  for (int size = 0; size < 25; ++size) {
    const auto base = av2enc::txv::make_tx_params(
        static_cast<av2enc::txv::TxSize>(size));
    CHECK(base.has_value());
    if (!base.has_value()) continue;
    const int w = base->coded_w;
    const int h = base->coded_h;

    for (int plane = 0; plane < 3; ++plane) {
      for (int reduced = 0; reduced < 4; ++reduced) {
        for (int idtx = 0; idtx < 2; ++idtx) {
          for (int dctonly = 0; dctonly < 2; ++dctonly) {
            for (int mode = 0; mode < 13; ++mode) {
              for (int uv = 0; uv < 14; ++uv) {
                for (int eob_index = 0; eob_index < 5; ++eob_index) {
                  const int eob = eob_values[eob_index];

                  idtx::Query query;
                  query.coded_w = w;
                  query.coded_h = h;
                  query.plane = plane;
                  query.mapped_mode = mode;
                  query.y_mode = mode;
                  query.uv_mode = uv;
                  query.uv_angle_delta = 0;
                  query.eob = eob;
                  query.flags.enable_idtx_intra = idtx != 0;
                  query.flags.enable_chroma_dctonly = dctonly != 0;
                  query.flags.reduced_tx_set = reduced;

                  idtx::Set set = idtx::Set::kDctOnly;
                  CHECK(idtx::get_tx_set(query, &set));

                  idtx::TypeSet types;
                  std::string error;
                  CHECK_MSG(idtx::enumerate_types(query, &types, &error), error);
                  ++enumerated;

                  // DCT_DCT is always representable.
                  bool have_dct = false;
                  for (int i = 0; i < types.count; ++i) {
                    if (types.candidates[i].type == TxType::kDctDct) {
                      have_dct = true;
                    }
                  }
                  CHECK(have_dct);

                  // No duplicate (type, flag) pair.
                  for (int i = 0; i < types.count; ++i) {
                    for (int j = i + 1; j < types.count; ++j) {
                      CHECK(!(types.candidates[i].type ==
                                  types.candidates[j].type &&
                              types.candidates[i].is_long_side_dct ==
                                  types.candidates[j].is_long_side_dct));
                    }
                  }

                  const bool effective_idtx =
                      query.flags.enable_fsc ||
                      query.flags.enable_idtx_intra;
                  const bool forced_dct =
                      eob <= 1 ||
                      (plane == 0 && reduced == 2) ||
                      set == idtx::Set::kDctOnly ||
                      set == idtx::Set::kDctIdtx;

                  for (int i = 0; i < types.count; ++i) {
                    const idtx::Candidate& c = types.candidates[i];
                    ++legality_hash;
                    legality_hash = fnv1a(legality_hash,
                                          reinterpret_cast<const std::uint8_t*>(
                                              &c.symbol),
                                          sizeof(c.symbol));
                    if (plane == 0 && eob >= 2 && !forced_dct) {
                      // Every coded type must be legal in Tx_Type_In_Set_Intra.
                      CHECK(idtx::intra_set_allows(types.set, c.type));
                      // Identity types only with the effective sequence flag.
                      if (idtx::is_identity_type(c.type)) {
                        CHECK(effective_idtx);
                      }
                      CHECK(c.symbol >= 0);
                    } else if (plane == 0) {
                      CHECK(c.type == TxType::kDctDct);
                      CHECK(c.symbol == -1);
                    } else {
                      // Chroma never codes a type; the enumeration is the set
                      // capability, so every entry must still be set-legal.
                      CHECK(idtx::intra_set_allows(types.set, c.type));
                    }

                    idtx::TypePlan plan;
                    std::string plan_error;
                    if (plane == 0) {
                      CHECK_MSG(idtx::plan_type(query, c.type, &plan,
                                                &plan_error),
                                plan_error);
                      CHECK(plan.type == c.type);
                      CHECK(plan.symbol == c.symbol);
                      CHECK(plan.type_syntax == (c.symbol >= 0));
                      CHECK(plan.is_long_side_dct == c.is_long_side_dct);
                      CHECK(plan.codes_long_flag == c.codes_long_flag);
                    }
                  }

                  // Types outside the set must be rejected.
                  for (int type = 0; type < 16; ++type) {
                    const TxType candidate = static_cast<TxType>(type);
                    bool in_set = false;
                    for (int i = 0; i < types.count; ++i) {
                      if (types.candidates[i].type == candidate) in_set = true;
                    }
                    idtx::TypePlan plan;
                    std::string plan_error;
                    if (plane == 0) {
                      const bool accepted = idtx::plan_type(
                          query, candidate, &plan, &plan_error);
                      CHECK(accepted == in_set);
                      if (!accepted) CHECK(!plan_error.empty());
                    } else {
                      TxType derived = TxType::kDctDct;
                      idtx::Set cset = idtx::Set::kDctOnly;
                      CHECK(idtx::derive_chroma_type(
                          uv, 0, w, h, query.flags, &derived, &cset));
                      const bool accepted = idtx::plan_type(
                          query, candidate, &plan, &plan_error);
                      CHECK(accepted == (candidate == derived));
                      CHECK(plan.type_syntax == false);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  std::printf("legality: %lld enumerations, hash %016llx\n", enumerated,
              static_cast<unsigned long long>(legality_hash));
  CHECK(enumerated == 25LL * 3 * 4 * 2 * 2 * 13 * 14 * 5);
  CHECK(legality_hash == kLegalityHash);
}

// ---------------------------------------------------------------------------
// 3. Chroma derivation
// ---------------------------------------------------------------------------
void test_chroma_derivation() {
  // Mode_To_Txfm golden row.
  const TxType expected[14] = {
      TxType::kDctDct,  TxType::kAdstDct, TxType::kDctAdst, TxType::kDctDct,
      TxType::kAdstAdst, TxType::kAdstDct, TxType::kDctAdst, TxType::kDctAdst,
      TxType::kAdstDct, TxType::kAdstAdst, TxType::kAdstDct,
      TxType::kDctAdst, TxType::kAdstAdst, TxType::kDctDct};
  for (int mode = 0; mode < 14; ++mode) {
    CHECK(idtx::mode_to_txfm(mode) == expected[mode]);
  }

  idtx::SeqFlags flags;
  flags.enable_chroma_dctonly = false;

  // 8x8 (no wide-angle remap): V_PRED -> ADST_DCT, H_PRED -> DCT_ADST.
  TxType type = TxType::kDctDct;
  CHECK(idtx::derive_chroma_type(1, 0, 8, 8, flags, &type));
  CHECK(type == TxType::kAdstDct);
  // H_PRED derives DCT_ADST.
  CHECK(idtx::derive_chroma_type(2, 0, 8, 8, flags, &type));
  CHECK(type == TxType::kDctAdst);
  // D45 -> DCT_DCT; D135 -> ADST_ADST; PAETH -> ADST_ADST.
  CHECK(idtx::derive_chroma_type(3, 0, 8, 8, flags, &type));
  CHECK(type == TxType::kDctDct);
  CHECK(idtx::derive_chroma_type(4, 0, 8, 8, flags, &type));
  CHECK(type == TxType::kAdstAdst);
  CHECK(idtx::derive_chroma_type(12, 0, 8, 8, flags, &type));
  CHECK(type == TxType::kAdstAdst);
  // UV_CFL_PRED has no mode derivation.
  CHECK(idtx::derive_chroma_type(13, 0, 8, 8, flags, &type));
  CHECK(type == TxType::kDctDct);
  // 8x16 is the gate's chroma size: V_PRED stays vertical (no WAIP remap at
  // pAngle 90), D45 remaps to D203 and therefore derives DCT_ADST.
  CHECK(idtx::derive_chroma_type(1, 0, 8, 16, flags, &type));
  CHECK(type == TxType::kAdstDct);
  CHECK(idtx::derive_chroma_type(3, 0, 8, 16, flags, &type));
  CHECK(type == TxType::kDctAdst);

  // enable_chroma_dctonly = 1 forces DCT_DCT for every mode and size.
  idtx::SeqFlags dctonly;
  dctonly.enable_chroma_dctonly = true;
  for (int size = 0; size < 25; ++size) {
    const auto params = av2enc::txv::make_tx_params(
        static_cast<av2enc::txv::TxSize>(size));
    if (!params.has_value()) continue;
    for (int uv = 0; uv < 14; ++uv) {
      CHECK(idtx::derive_chroma_type(uv, 0, params->coded_w, params->coded_h,
                                     dctonly, &type));
      CHECK(type == TxType::kDctDct);
    }
  }

  // Chroma 32x32 is DCT-only (§5.20.8.3: !is_inter && sqrUp == 32x32) even
  // with enable_chroma_dctonly = 0.
  CHECK(idtx::derive_chroma_type(1, 0, 32, 32, flags, &type));
  CHECK(type == TxType::kDctDct);

  // The derived type is always in the set the syntax reports, and never an
  // identity type.
  for (int size = 0; size < 25; ++size) {
    const auto params = av2enc::txv::make_tx_params(
        static_cast<av2enc::txv::TxSize>(size));
    if (!params.has_value()) continue;
    for (int uv = 0; uv < 14; ++uv) {
      for (int delta = -3; delta <= 3; ++delta) {
        idtx::Set set = idtx::Set::kDctOnly;
        CHECK(idtx::derive_chroma_type(uv, delta, params->coded_w,
                                       params->coded_h, flags, &type, &set));
        CHECK(!idtx::is_identity_type(type));
        CHECK(idtx::intra_set_allows(set, type));
        if (uv == 13) CHECK(type == TxType::kDctDct);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// 4. reduced_tx_set 1..3
// ---------------------------------------------------------------------------
void test_reduced_sets() {
  idtx::Query query;
  query.coded_w = 16;
  query.coded_h = 16;
  query.plane = 0;
  query.mapped_mode = 0;
  query.eob = 8;
  query.flags.enable_idtx_intra = true;

  struct Expect {
    int reduced;
    idtx::Set set;
    int count;
    bool codes_syntax;
  };
  // 16x16: sqrUp == 2, so reduced 0 -> INTRA_1, 1/3 -> INTRA_2, 2 -> DCT.
  const Expect expects[4] = {
      {0, idtx::Set::kIntra1, 7, true},
      {1, idtx::Set::kIntra2, 2, true},
      {2, idtx::Set::kDctIdtx, 1, false},
      {3, idtx::Set::kIntra2, 2, true},
  };
  for (const Expect& e : expects) {
    query.flags.reduced_tx_set = e.reduced;
    idtx::TypeSet types;
    std::string error;
    CHECK_MSG(idtx::enumerate_types(query, &types, &error), error);
    CHECK(types.set == e.set);
    CHECK(types.count == e.count);
    CHECK(types.candidates[0].type == TxType::kDctDct);
    CHECK((types.candidates[0].symbol >= 0) == e.codes_syntax);
    if (e.codes_syntax) {
      CHECK(types.candidates[1].type == TxType::kAdstAdst);
    }
  }

  // reduced_tx_set == 2 removes the symbol block even in a long-set size: the
  // get_tx_set result is still HIGH_32, but transform_type() skips the symbol.
  query.coded_w = 16;
  query.coded_h = 32;
  query.flags.reduced_tx_set = 2;
  idtx::TypeSet types;
  CHECK(idtx::enumerate_types(query, &types));
  CHECK(types.set == idtx::Set::kHigh32);
  CHECK(types.count == 1);
  CHECK(types.candidates[0].type == TxType::kDctDct);
  CHECK(types.candidates[0].symbol == -1);

  // LONG-32 with identity off drops exactly the identity entries.
  query.flags.reduced_tx_set = 0;
  query.flags.enable_idtx_intra = true;
  CHECK(idtx::enumerate_types(query, &types));
  int with_idtx = types.count;
  query.flags.enable_idtx_intra = false;
  CHECK(idtx::enumerate_types(query, &types));
  int without_idtx = types.count;
  CHECK(with_idtx == 8);
  CHECK(without_idtx == 3);
  for (int i = 0; i < types.count; ++i) {
    CHECK(!idtx::is_identity_type(types.candidates[i].type));
  }

  // 16x64 is HIGH_64: width < height, so it implies is_long_side_dct = 1 and
  // lists V_DCT.
  query.coded_w = 16;
  query.coded_h = 64;
  query.flags.enable_idtx_intra = true;
  CHECK(idtx::enumerate_types(query, &types));
  CHECK(types.set == idtx::Set::kHigh64);
  CHECK(types.count == 4);
  for (int i = 0; i < types.count; ++i) {
    CHECK(types.candidates[i].is_long_side_dct);
    CHECK(!types.candidates[i].codes_long_flag);
  }
  CHECK(types.candidates[3].type == TxType::kVdct);
}

// ---------------------------------------------------------------------------
// 5. QM / IST / TCQ interaction
// ---------------------------------------------------------------------------
void test_tool_interactions() {
  idtx::Query query;
  query.coded_w = 16;
  query.coded_h = 16;
  query.plane = 0;
  query.mapped_mode = 0;
  query.y_mode = 0;
  query.eob = 4;
  query.flags.enable_idtx_intra = true;
  query.flags.enable_intra_ist = true;
  query.flags.using_qmatrix = true;
  query.flags.allow_tcq = true;
  query.flags.enable_parity_hiding = true;

  for (int type = 0; type < 16; ++type) {
    const TxType t = static_cast<TxType>(type);
    const idtx::ToolUse use = idtx::derive_tool_use(query, t, 5, true, false,
                                                    false);
    // §7.14.4 useQm requires PlaneTxType < IDTX.
    CHECK(use.use_qm == (type < static_cast<int>(TxType::kIdtx)));
    // IST only for DCT_DCT / ADST_ADST, and never for an identity type.
    const bool ist_type = t == TxType::kDctDct || t == TxType::kAdstAdst;
    CHECK(use.use_ist == ist_type);
    // §7.14.4 TCQ extra shift is keyed on TX_CLASS_2D, which includes IDTX.
    CHECK(use.tcq_extra_shift == (idtx::tx_class(t) == 0));
    if (t == TxType::kIdtx) CHECK(use.tcq_extra_shift);
    CHECK(use.parity_hiding == (t != TxType::kIdtx));
    CHECK(use.tx_class == idtx::tx_class(t));
  }

  // segLvl out of range disables QM; identically for a user QM bit.
  CHECK(!idtx::derive_tool_use(query, TxType::kDctDct, 15, false, false, false)
             .use_qm);
  CHECK(idtx::derive_tool_use(query, TxType::kDctDct, 14, true, false, false)
            .use_qm);

  // IST eob limits and the PAETH exclusion.
  idtx::Query small = query;
  small.coded_w = 4;
  small.coded_h = 4;
  CHECK(idtx::derive_tool_use(small, TxType::kDctDct, 0, false, false, false)
            .use_ist);
  small.eob = 9;
  CHECK(!idtx::derive_tool_use(small, TxType::kDctDct, 0, false, false, false)
             .use_ist);
  small.eob = 8;
  CHECK(idtx::derive_tool_use(small, TxType::kDctDct, 0, false, false, false)
            .use_ist);
  idtx::Query paeth = query;
  paeth.y_mode = 12;
  CHECK(!idtx::derive_tool_use(paeth, TxType::kDctDct, 0, false, false, false)
             .use_ist);
  // ADST_ADST at 8x8 uses the reduced IST height.
  idtx::Query eight = query;
  eight.coded_w = 8;
  eight.coded_h = 8;
  CHECK(idtx::derive_tool_use(eight, TxType::kAdstAdst, 0, false, false, false)
            .ist_eob_limit == idtx::kIst8x8HeightRed);

  // Lossless disables IST and the TCQ shift.
  CHECK(!idtx::derive_tool_use(query, TxType::kDctDct, 0, false, true, false)
             .use_ist);
  CHECK(!idtx::derive_tool_use(query, TxType::kIdtx, 0, false, true, false)
             .tcq_extra_shift);

  // FSC: identity + enable_fsc + fsc_mode -> useFsc; that removes the TCQ
  // shift; allow_fsc_intra is bounded by FSC_MAX.
  idtx::Query fsc = query;
  fsc.flags.enable_fsc = true;
  fsc.flags.enable_idtx_intra = false;
  CHECK(idtx::derive_sequence_flags(fsc.flags).enable_idtx_intra);
  const idtx::ToolUse fsc_use =
      idtx::derive_tool_use(fsc, TxType::kIdtx, 0, false, false, true);
  CHECK(fsc_use.use_fsc);
  CHECK(!fsc_use.tcq_extra_shift);
  CHECK(fsc_use.allow_fsc_intra);
  idtx::Query big = query;
  big.coded_w = 32;
  big.coded_h = 64;
  CHECK(!idtx::derive_tool_use(big, TxType::kIdtx, 0, false, false, false)
             .allow_fsc_intra);
}

// ---------------------------------------------------------------------------
// 6. Identity reconstruction parity against txv
// ---------------------------------------------------------------------------
int identity_scale(int log2_size) {
  // §7.15.4.1 get_identity_scale().
  switch (log2_size) {
    case 2:
      return 128;
    case 3:
      return 181;
    case 4:
      return 256;
    default:
      return 362;
  }
}

std::int32_t round2_ref(std::int32_t value, int n) {
  if (n <= 0) return value;
  return static_cast<std::int32_t>(
      (static_cast<std::int64_t>(value) + (1LL << (n - 1))) >> n);
}

int clip3_ref(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

// Independent §7.15.4.1 process: row pass, column pass, identity scaling,
// clips and adjusted-size duplication.  The kernels are the §9.6 tables
// (shared with txv); everything else is written from the spec text.
void reference_inverse(const std::int32_t* dequant, const TxParams& p,
                       std::int16_t* out) {
  const int w = p.w;
  const int h = p.h;
  const av2enc::txv::Tx1dType row_type =
      static_cast<av2enc::txv::Tx1dType>(p.row_type);
  const av2enc::txv::Tx1dType col_type =
      static_cast<av2enc::txv::Tx1dType>(p.col_type);
  const std::int16_t* kw = av2enc::txv::tx_kernel(row_type, w);
  const std::int16_t* kh = av2enc::txv::tx_kernel(col_type, h);
  const bool odd = ((p.log2_w ^ p.log2_h) & 1) != 0;

  std::vector<std::int32_t> row(static_cast<std::size_t>(w));
  std::vector<std::int32_t> mid(static_cast<std::size_t>(w * h));

  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      const std::int32_t d = dequant[i * w + j];
      row[static_cast<std::size_t>(j)] =
          odd ? round2_ref(static_cast<std::int32_t>(
                               static_cast<std::int64_t>(d) * 2896),
                           12)
              : d;
    }
    for (int o = 0; o < w; ++o) {
      std::int32_t value = 0;
      if (kw == nullptr) {
        const std::int64_t s = static_cast<std::int64_t>(row[static_cast<std::size_t>(o)]) *
                               identity_scale(p.log2_w);
        value = round2_ref(static_cast<std::int32_t>(s), p.row_shift);
      } else {
        std::int64_t sum = 0;
        for (int j = 0; j < w; ++j) {
          sum += static_cast<std::int64_t>(kw[j * w + o]) * row[static_cast<std::size_t>(j)];
        }
        value = round2_ref(static_cast<std::int32_t>(sum), p.row_shift);
      }
      mid[static_cast<std::size_t>(i * w + o)] =
          clip3_ref(-(1 << 15), (1 << 15) - 1, value);
    }
  }

  for (int oi = 0; oi < h; ++oi) {
    for (int j = 0; j < w; ++j) {
      std::int32_t value = 0;
      if (kh == nullptr) {
        const std::int64_t s =
            static_cast<std::int64_t>(mid[static_cast<std::size_t>(oi * w + j)]) *
            identity_scale(p.log2_h);
        value = round2_ref(static_cast<std::int32_t>(s), p.col_shift);
      } else {
        std::int64_t sum = 0;
        for (int i = 0; i < h; ++i) {
          sum += static_cast<std::int64_t>(kh[i * h + oi]) *
                 mid[static_cast<std::size_t>(i * w + j)];
        }
        value = round2_ref(static_cast<std::int32_t>(sum), p.col_shift);
      }
      out[oi * w + j] = static_cast<std::int16_t>(
          clip3_ref(-256, 255, value));
    }
  }

  if (w != p.coded_w || h != p.coded_h) {
    const int sx = p.coded_w / w;
    const int sy = p.coded_h / h;
    for (int y = p.coded_h - 1; y >= 0; --y) {
      for (int x = p.coded_w - 1; x >= 0; --x) {
        out[y * p.coded_w + x] = out[(y / sy) * w + (x / sx)];
      }
    }
  }
}

void test_identity_parity() {
  std::uint64_t hash = 1469598103934665603ULL;
  long long checks = 0;
  const TxType identity_types[7] = {
      TxType::kIdtx,      TxType::kVdct,      TxType::kHdct,
      TxType::kVadst,     TxType::kHadst,     TxType::kVflipadst,
      TxType::kHflipadst};

  for (int size = 0; size < 25; ++size) {
    for (TxType type : identity_types) {
      const auto params = av2enc::txv::make_tx_params(
          static_cast<av2enc::txv::TxSize>(size), type);
      CHECK(params.has_value());
      if (!params.has_value()) continue;
      const int area = params->w * params->h;
      std::vector<std::int32_t> levels(static_cast<std::size_t>(area));
      std::uint32_t state = 0x12345678u + static_cast<std::uint32_t>(size) * 37u +
                            static_cast<std::uint32_t>(type) * 101u;
      for (int i = 0; i < area; ++i) {
        state = state * 1664525u + 1013904223u;
        const int magnitude = static_cast<int>((state >> 16) % 97) - 48;
        levels[static_cast<std::size_t>(i)] = magnitude;
      }
      std::vector<std::int32_t> dequant(static_cast<std::size_t>(area));
      av2enc::txv::dequantize(levels.data(), *params, dequant.data());
      std::vector<std::int16_t> reference(
          static_cast<std::size_t>(params->coded_w * params->coded_h));
      std::vector<std::int16_t> actual(reference.size());
      reference_inverse(dequant.data(), *params, reference.data());
      av2enc::txv::inverse_block(dequant.data(), *params, actual.data());
      for (std::size_t i = 0; i < reference.size(); ++i) {
        ++checks;
        hash = fnv1a(hash, reinterpret_cast<const std::uint8_t*>(&actual[i]),
                     sizeof(actual[i]));
        CHECK(actual[i] == reference[i]);
      }
    }
  }
  std::printf("identity parity: %lld samples, hash %016llx\n", checks,
              static_cast<unsigned long long>(hash));
  CHECK(hash == kIdentityParityHash);
}

// ---------------------------------------------------------------------------
// 7. §5.20.8.2 type-syntax writer/reader
// ---------------------------------------------------------------------------
void test_type_syntax() {
  struct Case {
    idtx::TypeSet types;
    int tx_size;
    TxType type;
    int symbol;
    bool flag;
  };

  // A deterministic multi-set stream that exercises adaptation, the coded
  // long-32 flag and the implied long-64 flag.
  std::vector<Case> cases;
  for (int i = 0; i < 64; ++i) {
    idtx::Query query;
    query.plane = 0;
    query.eob = 8;
    query.mapped_mode = i % 13;
    query.coded_w = (i % 2) == 0 ? 16 : 8;
    query.coded_h = (i % 2) == 0 ? 16 : 32;
    query.flags.enable_idtx_intra = true;
    query.flags.reduced_tx_set = (i % 5 == 0) ? 1 : 0;
    idtx::TypeSet types;
    CHECK(idtx::enumerate_types(query, &types));
    const idtx::Candidate& c =
        types.candidates[static_cast<std::size_t>(i) %
            static_cast<std::size_t>(types.count)];
    cases.push_back(Case{types,
                         av2enc::txv::tx_size_index(query.coded_w, query.coded_h),
                         c.type, c.symbol, c.is_long_side_dct});
  }
  // Long-64 (implied flag 1) and long-32 with flag 1, plus INTRA_2.
  {
    idtx::Query query;
    query.coded_w = 32;
    query.coded_h = 16;
    query.plane = 0;
    query.mapped_mode = 0;
    query.eob = 8;
    query.flags.enable_idtx_intra = true;
    idtx::TypeSet types;
    CHECK(idtx::enumerate_types(query, &types));
    cases.push_back(
        Case{types, av2enc::txv::tx_size_index(32, 16), TxType::kHdct, 3, true});
  }
  {
    idtx::Query query;
    query.coded_w = 8;
    query.coded_h = 8;
    query.plane = 0;
    query.mapped_mode = 0;
    query.eob = 8;
    query.flags.reduced_tx_set = 1;
    idtx::TypeSet types;
    CHECK(idtx::enumerate_types(query, &types));
    cases.push_back(
        Case{types, av2enc::txv::tx_size_index(8, 8), TxType::kAdstAdst, 1, false});
  }
  {
    idtx::Query query;
    query.coded_w = 16;
    query.coded_h = 64;
    query.plane = 0;
    query.mapped_mode = 0;
    query.eob = 8;
    query.flags.enable_idtx_intra = true;
    idtx::TypeSet types;
    CHECK(idtx::enumerate_types(query, &types));
    cases.push_back(
        Case{types, av2enc::txv::tx_size_index(16, 64), TxType::kVdct, 3, true});
  }

  idtx::TypeSyntaxWriter writer;
  std::string error;
  for (const Case& c : cases) {
    CHECK_MSG(writer.write(c.types.set, c.tx_size, c.type, c.symbol, c.flag,
                           true, &error),
              error);
  }
  std::vector<std::uint8_t> bytes;
  CHECK_MSG(writer.finish(&bytes, &error), error);
  CHECK(!bytes.empty());
  const std::uint64_t byte_hash =
      fnv1a(1469598103934665603ULL, bytes.data(), bytes.size());
  std::printf("type syntax: %zu bytes, hash %016llx\n", bytes.size(),
              static_cast<unsigned long long>(byte_hash));
  CHECK(bytes.size() == kTypeSyntaxBytes);
  CHECK(byte_hash == kTypeSyntaxHash);

  idtx::TypeSyntaxReader reader;
  CHECK_MSG(reader.init(bytes, bytes.size(), &error), error);
  int case_index = 0;
  for (const Case& c : cases) {
    idtx::TypeSymbols out;
    CHECK_MSG(reader.read(c.types, c.tx_size, true, &out, &error), error);
    if (out.type != c.type) {
      std::printf("type mismatch at %d: set %s size %d expected %s got %s "
                  "(symbol %d flag %d)\n",
                  case_index, idtx::set_name(c.types.set), c.tx_size,
                  av2enc::txv::tx_type_name(c.type),
                  av2enc::txv::tx_type_name(out.type), out.symbol, out.flag);
    }
    CHECK(out.type == c.type);
    CHECK(out.symbol == c.symbol);
    CHECK(out.codes_flag == (c.types.set == idtx::Set::kWide32 ||
                             c.types.set == idtx::Set::kHigh32));
    CHECK(out.flag == c.flag);
  }
  CHECK_MSG(reader.finish(&error), error);

  // No-syntax sets consume no bits and write no bytes.
  idtx::TypeSyntaxWriter empty;
  CHECK(empty.write(idtx::Set::kDctOnly, 1, TxType::kDctDct, -1, false));
  std::vector<std::uint8_t> empty_bytes;
  CHECK(empty.finish(&empty_bytes, &error));
  CHECK(empty_bytes.empty());
}

}  // namespace

int main() {
  test_sequence_flags();
  test_exhaustive_legality();
  test_chroma_derivation();
  test_reduced_sets();
  test_tool_interactions();
  test_identity_parity();
  test_type_syntax();

  std::printf("idtx_intra checks: %d, failures: %d\n", g_checks, g_failures);
  if (g_failures != 0) return 1;
  std::printf("IDTX_INTRA TESTS PASS\n");
  return 0;
}

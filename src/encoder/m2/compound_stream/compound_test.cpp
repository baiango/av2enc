// Unit tests for the compound / skip-mode inter mode-info syntax.
//
// Covers every grammar branch this component can emit:
//   * single_ref for both reference lists of a 2-reference frame
//   * comp_mode + read_compound_ref for a distinct pair
//   * compound_mode_same_refs (same-reference compound)
//   * skip_mode (derived refs, no comp_mode/ref symbols, SkipDrlMode CDF)
//   * use_optflow S() with REFINE_SWITCHABLE (one past + one future ref)
//   * enable_masked_compound comp_group_idx/compound_type/wedge/mask_type
//   * cwp_idx for NEAR_NEARMV + AVERAGE
//   * all-disabled byte identity with the M2 core single-ref writer
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "encoder/m2/compound_stream/compound_stream.h"
#include "encoder/m2/compound_stream/compound_syntax.h"

namespace {

int g_checks = 0;
int g_failures = 0;

#define CHECK(cond, what)                                              \
  do {                                                                 \
    ++g_checks;                                                        \
    if (!(cond)) {                                                     \
      ++g_failures;                                                    \
      std::printf("FAIL: %s (%s:%d)\n", (what), __FILE__, __LINE__);    \
    }                                                                  \
  } while (0)

using av2enc::m2::compound::CompoundBlockPlan;
using av2enc::m2::compound::CompoundBlockResult;
using av2enc::m2::compound::CompoundFrameConfig;
using av2enc::m2::multiref::CompoundModeParams;
using av2enc::m2::multiref::RefFrameMap;

CompoundFrameConfig make_config(bool features) {
  CompoundFrameConfig cfg;
  RefFrameMap& map = cfg.ref_map;
  map.num_total_refs = 2;
  map.ref_frame_idx[0] = 0;
  map.ref_frame_idx[1] = 1;
  map.order_hint = 2;
  map.order_hints[0] = 0;
  map.order_hints[1] = 4;
  map.frame_distance[0] = 2;
  map.frame_distance[1] = -2;
  map.reference_select = features;
  map.num_same_ref_compound = features ? 1 : 0;
  cfg.skip_mode_present = features;
  cfg.skip_mode_frame[0] = 0;
  cfg.skip_mode_frame[1] = 1;
  cfg.opfl_refine_type =
      features ? av2enc::m2::compound::kRefineSwitchable
               : av2enc::m2::compound::kRefineNone;
  cfg.enable_masked_compound = features;
  cfg.enable_cwp = features;
  cfg.enable_imp_msk_bld = false;
  return cfg;
}

CompoundBlockPlan base_plan(int sb) {
  CompoundBlockPlan plan;
  plan.mi_row = (sb / 4) * 16;
  plan.mi_col = (sb % 4) * 16;
  plan.w4 = 16;
  plan.h4 = 16;
  plan.skip_flag = true;
  plan.ref_frame[0] = 0;
  plan.ref_frame[1] = av2enc::m2::compound::kNoneRef;
  plan.y_mode = av2enc::m2::compound::kGlobalMv;
  return plan;
}

void test_disabled_round_trip() {
  const CompoundFrameConfig cfg = make_config(false);
  std::vector<CompoundBlockPlan> plans;
  for (int sb = 0; sb < 4; ++sb) plans.push_back(base_plan(sb));
  std::vector<std::uint8_t> payload;
  std::string error;
  CHECK(av2enc::m2::compound::write_compound_tile(plans, 256, 64, cfg, payload,
                                                  error),
        error.c_str());
  std::vector<CompoundBlockResult> decoded;
  CHECK(av2enc::m2::compound::read_compound_tile(payload, 256, 64, cfg, decoded,
                                                 error),
        error.c_str());
  CHECK(decoded.size() == plans.size(), "decoded block count");
  for (std::size_t i = 0; i < decoded.size() && i < plans.size(); ++i) {
    CHECK(decoded[i].ref_frame[0] == 0, "single ref 0");
    CHECK(decoded[i].ref_frame[1] == av2enc::m2::compound::kNoneRef,
          "no second ref");
    CHECK(decoded[i].y_mode == av2enc::m2::compound::kGlobalMv, "GLOBALMV");
    CHECK(decoded[i].skip_flag, "skip flag");
    CHECK(!decoded[i].skip_mode, "skip mode off");
    CHECK(!decoded[i].use_optflow, "optflow off");
  }

  bool identical = false;
  std::size_t bytes = 0;
  CHECK(av2enc::m2::compound::check_disabled_byte_identity(256, 64, 90,
                                                           &identical, &bytes,
                                                           error),
        error.c_str());
  CHECK(identical, "disabled tile must be byte-identical to the M2 core");
}

void test_full_round_trip() {
  const CompoundFrameConfig cfg = make_config(true);
  std::vector<CompoundBlockPlan> plans;
  for (int sb = 0; sb < 8; ++sb) plans.push_back(base_plan(sb));

  // 0: distinct-pair AVERAGE, GLOBAL_GLOBALMV.
  plans[0].ref_frame[1] = 1;
  plans[0].y_mode = av2enc::m2::compound::kGlobalGlobalMv;

  // 1: skip_mode with derived refs (0,1), YMode NEAR_NEARMV.
  plans[1].skip_mode = true;
  plans[1].y_mode = av2enc::m2::compound::kNearNearMv;
  plans[1].ref_frame[1] = 1;
  plans[1].compound.cwp_weight = av2enc::m2::compound::adv::kCwpEqual;

  // 2: same-reference compound pair (0,0), GLOBAL_GLOBALMV.
  plans[2].ref_frame[0] = 0;
  plans[2].ref_frame[1] = 0;
  plans[2].y_mode = av2enc::m2::compound::kGlobalGlobalMv;

  // 3: use_optflow with a past and a future reference.
  plans[3].ref_frame[1] = 1;
  plans[3].y_mode = av2enc::m2::compound::kNearNearMv;
  plans[3].use_optflow = true;

  // 4/5: single_ref for the first and the last reference.
  plans[4].ref_frame[0] = 0;
  plans[4].y_mode = av2enc::m2::compound::kGlobalMv;
  plans[5].ref_frame[0] = 1;
  plans[5].y_mode = av2enc::m2::compound::kGlobalMv;

  // 6: wedge.
  plans[6].ref_frame[1] = 1;
  plans[6].y_mode = av2enc::m2::compound::kGlobalGlobalMv;
  plans[6].compound.type = av2enc::m2::compound::adv::CompoundType::kWedge;
  plans[6].compound.wedge_index = 22;
  plans[6].compound.wedge_sign = 0;

  // 7: diffwtd.
  plans[7].ref_frame[1] = 1;
  plans[7].y_mode = av2enc::m2::compound::kGlobalGlobalMv;
  plans[7].compound.type = av2enc::m2::compound::adv::CompoundType::kDiffWtd;
  plans[7].compound.mask_type = 1;

  std::vector<std::uint8_t> payload;
  std::string error;
  CHECK(av2enc::m2::compound::write_compound_tile(plans, 256, 128, cfg, payload,
                                                  error),
        error.c_str());
  std::vector<CompoundBlockResult> decoded;
  CHECK(av2enc::m2::compound::read_compound_tile(payload, 256, 128, cfg,
                                                 decoded, error),
        error.c_str());
  CHECK(decoded.size() == plans.size(), "full decoded block count");
  if (decoded.size() != plans.size()) return;

  for (std::size_t i = 0; i < decoded.size(); ++i) {
    const CompoundBlockPlan& p = plans[i];
    const CompoundBlockResult& d = decoded[i];
    CHECK(d.skip_mode == p.skip_mode, "skip_mode round trip");
    CHECK(d.skip_flag == p.skip_flag, "skip_flag round trip");
    CHECK(d.y_mode == p.y_mode, "YMode round trip");
    CHECK(d.ref_frame[0] == p.ref_frame[0], "RefFrame[0] round trip");
    CHECK(d.use_optflow == p.use_optflow, "use_optflow round trip");
  }
  CHECK(decoded[0].is_compound, "block 0 compound");
  CHECK(decoded[0].ref_frame[1] == 1, "block 0 ref pair");
  CHECK(decoded[1].skip_mode, "block 1 skip mode");
  CHECK(decoded[1].ref_frame[0] == 0 && decoded[1].ref_frame[1] == 1,
        "skip mode derived refs");
  CHECK(decoded[2].ref_frame[0] == 0 && decoded[2].ref_frame[1] == 0,
        "same-reference pair");
  CHECK(decoded[3].use_optflow, "block 3 optflow");
  CHECK(decoded[4].ref_frame[0] == 0 && !decoded[4].is_compound,
        "single ref first");
  CHECK(decoded[5].ref_frame[0] == 1 && !decoded[5].is_compound,
        "single ref last");
  CHECK(decoded[6].compound.type ==
            av2enc::m2::compound::adv::CompoundType::kWedge,
        "wedge round trip");
  CHECK(decoded[6].compound.wedge_index == 22, "wedge index round trip");
  CHECK(decoded[7].compound.type ==
            av2enc::m2::compound::adv::CompoundType::kDiffWtd,
        "diffwtd round trip");
  CHECK(decoded[7].compound.mask_type == 1, "mask_type round trip");
}

void test_negative_cases() {
  std::string error;
  // Masked compound is not signalable with enable_masked_compound = 0.
  {
    const CompoundFrameConfig cfg = make_config(false);
    std::vector<CompoundBlockPlan> plans = {base_plan(0)};
    plans[0].ref_frame[1] = 1;
    plans[0].y_mode = av2enc::m2::compound::kGlobalGlobalMv;
    plans[0].compound.type =
        av2enc::m2::compound::adv::CompoundType::kWedge;
    std::vector<std::uint8_t> payload;
    CHECK(!av2enc::m2::compound::write_compound_tile(plans, 64, 64, cfg,
                                                     payload, error),
          "masked compound must be refused when disabled");
  }
  // skip_mode cannot be coded when skip_mode_present = 0.
  {
    const CompoundFrameConfig cfg = make_config(false);
    std::vector<CompoundBlockPlan> plans = {base_plan(0)};
    plans[0].skip_mode = true;
    plans[0].ref_frame[1] = 1;
    plans[0].y_mode = av2enc::m2::compound::kNearNearMv;
    std::vector<std::uint8_t> payload;
    CHECK(!av2enc::m2::compound::write_compound_tile(plans, 64, 64, cfg,
                                                     payload, error),
          "skip_mode must be refused when not present");
  }
  // Same-reference compound needs NumSameRefCompound > 0.
  {
    CompoundFrameConfig cfg = make_config(true);
    cfg.ref_map.num_same_ref_compound = 0;
    std::vector<CompoundBlockPlan> plans = {base_plan(0)};
    plans[0].ref_frame[0] = 0;
    plans[0].ref_frame[1] = 0;
    plans[0].y_mode = av2enc::m2::compound::kGlobalGlobalMv;
    std::vector<std::uint8_t> payload;
    CHECK(!av2enc::m2::compound::write_compound_tile(plans, 64, 64, cfg,
                                                     payload, error),
          "same-ref compound must be refused when not signalled");
  }
  // use_optflow requires one past and one future reference.
  {
    CompoundFrameConfig cfg = make_config(true);
    cfg.ref_map.frame_distance[1] = 1;  // both past
    cfg.ref_map.order_hints[1] = 1;
    std::vector<CompoundBlockPlan> plans = {base_plan(0)};
    plans[0].ref_frame[1] = 1;
    plans[0].y_mode = av2enc::m2::compound::kNearNearMv;
    plans[0].use_optflow = true;
    std::vector<std::uint8_t> payload;
    CHECK(!av2enc::m2::compound::write_compound_tile(plans, 64, 64, cfg,
                                                     payload, error),
          "optflow must be refused for same-side refs");
  }
}

void test_refine_all_inference() {
  CompoundFrameConfig cfg = make_config(true);
  cfg.opfl_refine_type = av2enc::m2::compound::kRefineAll;
  std::vector<CompoundBlockPlan> plans = {base_plan(0), base_plan(1)};
  // REFINE_ALL average block: use_optflow is inferred, no symbol.
  plans[0].ref_frame[1] = 1;
  plans[0].y_mode = av2enc::m2::compound::kNearNearMv;
  plans[0].use_optflow = true;
  // A masked block must not derive use_optflow.
  plans[1].ref_frame[1] = 1;
  plans[1].y_mode = av2enc::m2::compound::kGlobalGlobalMv;
  plans[1].compound.type = av2enc::m2::compound::adv::CompoundType::kWedge;
  plans[1].compound.wedge_index = 22;
  std::vector<std::uint8_t> payload;
  std::string error;
  CHECK(av2enc::m2::compound::write_compound_tile(plans, 128, 64, cfg, payload,
                                                  error),
        error.c_str());
  std::vector<CompoundBlockResult> decoded;
  CHECK(av2enc::m2::compound::read_compound_tile(payload, 128, 64, cfg, decoded,
                                                 error),
        error.c_str());
  if (decoded.size() != 2) return;
  CHECK(decoded[0].use_optflow, "REFINE_ALL inferred use_optflow");
  CHECK(!decoded[1].use_optflow, "REFINE_ALL masked block stays unrefined");

  // Requesting use_optflow on a masked block is not representable.
  plans[1].use_optflow = true;
  std::vector<std::uint8_t> bad;
  CHECK(!av2enc::m2::compound::write_compound_tile(plans, 128, 64, cfg, bad,
                                                   error),
        "REFINE_ALL masked+optflow must be refused");
}

void test_three_references() {
  // NumTotalRefs > 2 exercises the full read_single_ref() and
  // read_compound_ref() loops, not just the two-reference shortcut.
  CompoundFrameConfig cfg;
  cfg.ref_map.num_total_refs = 3;
  cfg.ref_map.ref_frame_idx[0] = 0;
  cfg.ref_map.ref_frame_idx[1] = 1;
  cfg.ref_map.ref_frame_idx[2] = 2;
  cfg.ref_map.order_hint = 2;
  cfg.ref_map.order_hints[0] = 0;
  cfg.ref_map.order_hints[1] = 4;
  cfg.ref_map.order_hints[2] = 8;
  cfg.ref_map.frame_distance[0] = 2;
  cfg.ref_map.frame_distance[1] = -2;
  cfg.ref_map.frame_distance[2] = -6;
  cfg.ref_map.reference_select = true;
  cfg.ref_map.num_same_ref_compound = 1;
  cfg.skip_mode_present = false;
  cfg.opfl_refine_type = av2enc::m2::compound::kRefineNone;

  std::vector<CompoundBlockPlan> plans;
  for (int sb = 0; sb < 8; ++sb) plans.push_back(base_plan(sb));
  plans[0].ref_frame[0] = 0;
  plans[1].ref_frame[0] = 1;
  plans[2].ref_frame[0] = 2;
  const int pairs[5][2] = {{0, 1}, {0, 2}, {1, 2}, {0, 0}, {1, 2}};
  for (int i = 0; i < 5; ++i) {
    plans[3 + i].ref_frame[0] = pairs[i][0];
    plans[3 + i].ref_frame[1] = pairs[i][1];
    plans[3 + i].y_mode = av2enc::m2::compound::kGlobalGlobalMv;
  }
  std::vector<std::uint8_t> payload;
  std::string error;
  CHECK(av2enc::m2::compound::write_compound_tile(plans, 256, 128, cfg, payload,
                                                  error),
        error.c_str());
  std::vector<CompoundBlockResult> decoded;
  CHECK(av2enc::m2::compound::read_compound_tile(payload, 256, 128, cfg,
                                                 decoded, error),
        error.c_str());
  CHECK(decoded.size() == plans.size(), "three-ref block count");
  if (decoded.size() != plans.size()) return;
  for (int i = 0; i < 3; ++i) {
    CHECK(decoded[i].ref_frame[0] == i, "single_ref reaches every index");
    CHECK(!decoded[i].is_compound, "single_ref stays single");
  }
  for (int i = 0; i < 5; ++i) {
    CHECK(decoded[3 + i].ref_frame[0] == pairs[i][0],
          "compound ref0 round trip");
    CHECK(decoded[3 + i].ref_frame[1] == pairs[i][1],
          "compound ref1 round trip");
  }
}

}  // namespace

int main() {
  std::printf("== compound_stream unit tests ==\n");
  test_disabled_round_trip();
  test_full_round_trip();
  test_negative_cases();
  test_refine_all_inference();
  test_three_references();
  std::printf("== %d checks, %d failure(s) ==\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}

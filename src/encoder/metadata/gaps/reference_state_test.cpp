// Unit tests for the §7.23 reference-state save/load container in
// src/encoder/metadata/gaps/reference_state.h.
//
// Covers every saved object and every index domain:
//   SavedMvs, SavedRefFrames, SavedGmParams, SavedSegmentIds (slots 0..15);
//   save_grain_params/load_grain_params (slots 0..NUM_REF_FRAMES inclusive);
//   save_ccso_params/load_ccso_params (slots 0..15 x planes 0..2).
//
// Each round trip fills every element with a distinct value, saves, mutates
// the source to prove the store made a deep copy, reloads and compares.

#include "encoder/metadata/gaps/reference_state.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace av2enc::metadata::gaps;

int g_failures = 0;

#define CHECK(expression)                                                     \
  do {                                                                        \
    if (!(expression)) {                                                      \
      ++g_failures;                                                           \
      std::cerr << __FILE__ << ':' << __LINE__                                \
                << ": CHECK failed: " #expression "\n";                       \
    }                                                                         \
  } while (false)

#define REQUIRE(expression)                                                   \
  do {                                                                        \
    if (!(expression)) {                                                      \
      ++g_failures;                                                           \
      std::cerr << __FILE__ << ':' << __LINE__                                \
                << ": REQUIRE failed: " #expression "\n";                     \
      return;                                                                 \
    }                                                                         \
  } while (false)

// ---------------------------------------------------------------------------
// Snapshot builders
// ---------------------------------------------------------------------------

ReferenceSlot make_slot(int seed) {
  ReferenceSlot slot;
  slot.mvs.rows = 3;
  slot.mvs.cols = 5;
  slot.mvs.units.resize(slot.mvs.rows * slot.mvs.cols);
  slot.ref_frames.rows = slot.mvs.rows;
  slot.ref_frames.cols = slot.mvs.cols;
  slot.ref_frames.units.resize(slot.mvs.units.size());

  int counter = 0;
  for (std::size_t y = 0; y < slot.mvs.rows; ++y) {
    for (std::size_t x = 0; x < slot.mvs.cols; ++x) {
      const std::size_t index = y * slot.mvs.cols + x;
      for (std::size_t list = 0; list < 2; ++list) {
        const int row_value = (seed * 97 + counter++) % 20000 - 10000;
        const int col_value = (seed * 89 + counter++) % 20000 - 10000;
        slot.mvs.units[index].list[list].row =
            static_cast<std::int16_t>(row_value);
        slot.mvs.units[index].list[list].col =
            static_cast<std::int16_t>(col_value);
        slot.ref_frames.units[index].list[list] =
            static_cast<std::int16_t>((seed * 7 + static_cast<int>(list) +
                                       static_cast<int>(index)) %
                                      16);
      }
    }
  }

  for (std::size_t ref = 0; ref < kRefsPerFrame; ++ref) {
    for (std::size_t j = 0; j < 6; ++j) {
      slot.gm_params.refs[ref].params[j] =
          static_cast<std::int32_t>(seed * 100000 + static_cast<int>(ref) * 100 +
                                    static_cast<int>(j)) -
          50000;
    }
  }

  slot.segment_ids.rows = slot.mvs.rows * 2;
  slot.segment_ids.cols = slot.mvs.cols * 2;
  slot.segment_ids.ids.resize(slot.segment_ids.rows * slot.segment_ids.cols);
  for (std::size_t i = 0; i < slot.segment_ids.ids.size(); ++i) {
    slot.segment_ids.ids[i] =
        static_cast<std::uint8_t>((seed + static_cast<int>(i)) % 16);
  }
  return slot;
}

GrainParams make_grain(int seed, bool chroma_from_luma = false,
                       bool minimal = false) {
  GrainParams grain;
  grain.fgm_chroma_idc = static_cast<std::uint8_t>(seed % 4);
  grain.config.apply_grain = true;
  grain.config.fgm_id = static_cast<std::uint8_t>(seed % 8);
  grain.config.grain_seed = static_cast<std::uint16_t>(seed * 1111);

  FilmGrainModel& model = grain.model;
  model.chroma_scaling_from_luma = chroma_from_luma;
  model.grain_scaling_minus_8 = static_cast<std::uint8_t>(seed % 4);
  model.ar_coeff_lag = minimal ? 0 : static_cast<std::uint8_t>(seed % 4);
  model.ar_coeff_shift_minus_6 = static_cast<std::uint8_t>(seed % 4);
  model.grain_scale_shift = static_cast<std::uint8_t>((seed + 1) % 4);
  model.overlap_flag = (seed & 1) != 0;
  model.clip_to_restricted_range = (seed & 2) != 0;
  model.fg_mc_identity = (seed & 4) != 0 && model.clip_to_restricted_range;
  model.film_grain_block_size = (seed & 1) != 0;

  if (!minimal) {
    model.num_y_points = 2;
    model.point_value_increment_bits_minus_1 = 1;
    model.point_scaling_bits_minus_5 = 1;
    model.y_points = {{10, 16}, {40, 20}};
  }

  if (!minimal && !chroma_from_luma) {
    model.num_cb_points = 1;
    model.cb_point_value_increment_bits_minus_1 = 1;
    model.cb_point_scaling_bits_minus_5 = 0;
    model.cb_points = {{8, 7}};
    model.num_cr_points = 1;
    model.cr_point_value_increment_bits_minus_1 = 1;
    model.cr_point_scaling_bits_minus_5 = 0;
    model.cr_points = {{9, 6}};
  }

  const std::size_t num_pos_luma =
      2U * model.ar_coeff_lag * (model.ar_coeff_lag + 1U);
  const std::size_t num_pos_chroma =
      num_pos_luma + (model.num_y_points != 0U ? 1U : 0U);
  if (model.num_y_points != 0U) {
    model.bits_per_ar_coeff_y_minus_5 = 1;  // 6 signed bits
    model.ar_coeffs_y.resize(num_pos_luma);
    for (std::size_t i = 0; i < num_pos_luma; ++i) {
      model.ar_coeffs_y[i] =
          static_cast<std::int16_t>((static_cast<int>(i) + seed) % 31 - 15);
    }
  }
  const bool cb_coded = chroma_from_luma || model.num_cb_points != 0U;
  if (cb_coded) {
    model.bits_per_ar_coeff_cb_minus_5 = 1;
    model.ar_coeffs_cb.resize(num_pos_chroma);
    for (std::size_t i = 0; i < num_pos_chroma; ++i) {
      model.ar_coeffs_cb[i] =
          static_cast<std::int16_t>(-(static_cast<int>(i) + seed) % 31 + 1);
    }
  }
  const bool cr_coded = chroma_from_luma || model.num_cr_points != 0U;
  if (cr_coded) {
    model.bits_per_ar_coeff_cr_minus_5 = 1;
    model.ar_coeffs_cr.resize(num_pos_chroma);
    for (std::size_t i = 0; i < num_pos_chroma; ++i) {
      model.ar_coeffs_cr[i] =
          static_cast<std::int16_t>((static_cast<int>(i) * 3 + seed) % 31 - 15);
    }
  }

  if (model.num_cb_points != 0U) {
    model.cb_mult = static_cast<std::uint8_t>(seed * 3 + 1);
    model.cb_luma_mult = static_cast<std::uint8_t>(seed * 5 + 1);
    model.cb_offset = static_cast<std::uint16_t>(seed * 17 + 1);
  }
  if (model.num_cr_points != 0U) {
    model.cr_mult = static_cast<std::uint8_t>(seed * 7 + 1);
    model.cr_luma_mult = static_cast<std::uint8_t>(seed * 11 + 1);
    model.cr_offset = static_cast<std::uint16_t>(seed * 19 + 1);
  }
  return grain;
}

CcsoPlaneParams make_ccso_plane(int seed, bool enabled = true,
                                bool bo_only = false) {
  CcsoPlaneParams plane;
  plane.enabled = enabled;
  if (!enabled) {
    return plane;
  }
  plane.bo_only = bo_only;
  plane.scale_idx = static_cast<std::uint8_t>(seed % 4);
  plane.quant_idx =
      static_cast<std::uint8_t>(bo_only ? 0 : seed % 4);
  plane.ext_filter =
      static_cast<std::uint8_t>(bo_only ? 0 : seed % 8);
  plane.edge_clf = static_cast<std::uint8_t>(bo_only ? 0 : seed % 2);
  plane.max_band_log2 =
      static_cast<std::uint8_t>(bo_only ? seed % 8 : seed % 4);
  for (std::size_t i = 0; i < kCcsoOffsetEntries; ++i) {
    plane.filter_offset[i] =
        static_cast<std::int16_t>((seed * 1000 + static_cast<int>(i)) % 30000 -
                                  15000);
  }
  plane.blk_rows = 2;
  plane.blk_cols = 3;
  plane.blks = {0, 1, 0, 1, 1, 0};
  return plane;
}

CcsoParams make_ccso_frame(int seed) {
  CcsoParams params;
  params.luma_size_log2 = static_cast<std::uint8_t>(6 + seed % 3);
  params.planes[0] = make_ccso_plane(seed);
  params.planes[1] = make_ccso_plane(seed + 1, /*enabled=*/true,
                                     /*bo_only=*/(seed % 2) != 0);
  params.planes[2] = make_ccso_plane(seed + 2, /*enabled=*/false);
  for (auto& plane : params.planes) {
    plane.luma_size_log2 = params.luma_size_log2;
  }
  return params;
}

// ---------------------------------------------------------------------------
// Round trips
// ---------------------------------------------------------------------------

void test_saved_mvs_round_trip() {
  ReferenceStateStore store;
  for (std::size_t slot = 0; slot < kNumRefFrames; ++slot) {
    const SavedMvs source = make_slot(static_cast<int>(slot)).mvs;
    std::string error;
    REQUIRE(store.save_mvs(slot, source, error));
    REQUIRE(store.has_reference(slot));
    SavedMvs loaded;
    REQUIRE(store.load_mvs(slot, loaded));
    CHECK(loaded == source);
  }
  std::cout << "PASS saved_mvs_round_trip (" << kNumRefFrames << " slots)\n";
}

void test_saved_ref_frames_round_trip() {
  ReferenceStateStore store;
  for (std::size_t slot = 0; slot < kNumRefFrames; ++slot) {
    const SavedRefFrames source = make_slot(static_cast<int>(slot + 10))
                                      .ref_frames;
    std::string error;
    REQUIRE(store.save_ref_frames(slot, source, error));
    SavedRefFrames loaded;
    REQUIRE(store.load_ref_frames(slot, loaded));
    CHECK(loaded == source);
  }
  std::cout << "PASS saved_ref_frames_round_trip (" << kNumRefFrames
            << " slots)\n";
}

void test_saved_gm_params_round_trip() {
  ReferenceStateStore store;
  for (std::size_t slot = 0; slot < kNumRefFrames; ++slot) {
    const SavedGmParams source =
        make_slot(static_cast<int>(slot + 20)).gm_params;
    std::string error;
    REQUIRE(store.save_gm_params(slot, source, error));
    SavedGmParams loaded;
    REQUIRE(store.load_gm_params(slot, loaded));
    CHECK(loaded == source);
  }
  std::cout << "PASS saved_gm_params_round_trip (" << kNumRefFrames
            << " slots)\n";
}

void test_saved_segment_ids_round_trip() {
  ReferenceStateStore store;
  for (std::size_t slot = 0; slot < kNumRefFrames; ++slot) {
    const SavedSegmentIds source =
        make_slot(static_cast<int>(slot + 30)).segment_ids;
    std::string error;
    REQUIRE(store.save_segment_ids(slot, source, error));
    SavedSegmentIds loaded;
    REQUIRE(store.load_segment_ids(slot, loaded));
    CHECK(loaded == source);
  }
  std::cout << "PASS saved_segment_ids_round_trip (" << kNumRefFrames
            << " slots)\n";
}

void test_reference_slot_round_trip() {
  std::array<ReferenceSlot, kNumRefFrames> sources;
  ReferenceStateStore store;
  std::string error;
  for (std::size_t slot = 0; slot < kNumRefFrames; ++slot) {
    sources[slot] = make_slot(static_cast<int>(slot));
    REQUIRE(store.save_reference(slot, sources[slot], error));
  }
  // Mutating the caller's snapshots must not change the store.
  for (auto& source : sources) {
    source.mvs.units.assign(source.mvs.units.size(), MvPair{});
    source.ref_frames.units.assign(source.ref_frames.units.size(),
                                   RefFramePair{});
    source.gm_params = SavedGmParams{};
    source.segment_ids.ids.assign(source.segment_ids.ids.size(), 0);
  }
  for (std::size_t slot = 0; slot < kNumRefFrames; ++slot) {
    ReferenceSlot loaded;
    REQUIRE(store.load_reference(slot, loaded));
    CHECK(loaded == make_slot(static_cast<int>(slot)));
  }
  std::cout << "PASS reference_slot_round_trip (" << kNumRefFrames
            << " slots, deep copy)\n";
}

void test_individual_saves_coexist() {
  ReferenceStateStore store;
  const ReferenceSlot source = make_slot(3);
  std::string error;
  REQUIRE(store.save_mvs(1, source.mvs, error));
  REQUIRE(store.save_gm_params(1, source.gm_params, error));
  REQUIRE(store.save_segment_ids(1, source.segment_ids, error));

  SavedMvs mvs;
  SavedGmParams gm;
  SavedSegmentIds segments;
  REQUIRE(store.load_mvs(1, mvs));
  REQUIRE(store.load_gm_params(1, gm));
  REQUIRE(store.load_segment_ids(1, segments));
  CHECK(mvs == source.mvs);
  CHECK(gm == source.gm_params);
  CHECK(segments == source.segment_ids);

  // SavedRefFrames was never written for slot 1 and defaults to empty.
  SavedRefFrames refs;
  REQUIRE(store.load_ref_frames(1, refs));
  CHECK(refs.units.empty());

  // A partially saved slot is not a complete §7.23 snapshot.
  ReferenceSlot partial;
  CHECK(!store.load_reference(1, partial));

  // Saving the full slot makes load_reference() succeed.
  CHECK(store.save_reference(1, source, error));
  ReferenceSlot complete;
  REQUIRE(store.load_reference(1, complete));
  CHECK(complete == source);
  std::cout << "PASS individual_saves_coexist\n";
}

void test_grain_round_trip() {
  ReferenceStateStore store;
  std::string error;
  std::vector<GrainParams> sources;
  for (std::size_t index = 0; index <= kNumRefFrames; ++index) {
    const bool chroma_from_luma = (index % 3U) == 1U;
    const bool minimal = (index % 5U) == 4U;
    const GrainParams source = make_grain(static_cast<int>(index),
                                          chroma_from_luma, minimal);
    REQUIRE(store.save_grain_params(index, source, error));
    REQUIRE(store.has_grain_params(index));
    sources.push_back(source);
  }
  for (std::size_t index = 0; index <= kNumRefFrames; ++index) {
    GrainParams loaded;
    REQUIRE(store.load_grain_params(index, loaded));
    CHECK(loaded == sources[index]);
  }
  // Mutate one input and re-check the stored copy is unaffected.
  GrainParams mutated = sources[0];
  mutated.model.y_points.clear();
  GrainParams loaded;
  REQUIRE(store.load_grain_params(0, loaded));
  CHECK(loaded != mutated);
  std::cout << "PASS grain_round_trip (" << (kNumRefFrames + 1)
            << " indices, incl. mixed-output slot)\n";
}

void test_ccso_round_trip() {
  ReferenceStateStore store;
  std::string error;
  for (std::size_t slot = 0; slot < kNumRefFrames; ++slot) {
    for (std::size_t plane = 0; plane < kCcsoPlanes; ++plane) {
      const CcsoPlaneParams source = make_ccso_plane(
          static_cast<int>(slot * 3U + plane),
          /*enabled=*/(plane != 2U), /*bo_only=*/(slot % 2U) != 0U);
      REQUIRE(store.save_ccso_params(slot, plane, source, error));
      CHECK(store.has_ccso_params(slot, plane));
      CcsoPlaneParams loaded;
      REQUIRE(store.load_ccso_params(slot, plane, loaded));
      CHECK(loaded == source);
    }
  }
  std::cout << "PASS ccso_round_trip (" << kNumRefFrames << " slots x "
            << kCcsoPlanes << " planes)\n";
}

void test_ccso_frame_round_trip() {
  ReferenceStateStore store;
  std::string error;
  std::array<CcsoParams, kNumRefFrames> sources;
  for (std::size_t slot = 0; slot < kNumRefFrames; ++slot) {
    sources[slot] = make_ccso_frame(static_cast<int>(slot));
    REQUIRE(store.save_ccso_frame(slot, sources[slot], error));
  }
  for (std::size_t slot = 0; slot < kNumRefFrames; ++slot) {
    CcsoParams loaded;
    REQUIRE(store.load_ccso_frame(slot, loaded));
    CHECK(loaded == sources[slot]);
  }
  std::cout << "PASS ccso_frame_round_trip (" << kNumRefFrames << " slots)\n";
}

void test_overwrite_and_clear() {
  ReferenceStateStore store;
  std::string error;
  const ReferenceSlot first = make_slot(1);
  const ReferenceSlot second = make_slot(2);
  REQUIRE(store.save_reference(3, first, error));
  REQUIRE(store.save_reference(3, second, error));
  ReferenceSlot loaded;
  REQUIRE(store.load_reference(3, loaded));
  CHECK(loaded == second);

  REQUIRE(store.save_grain_params(4, make_grain(4), error));
  REQUIRE(store.save_grain_params(4, make_grain(5), error));
  GrainParams grain;
  REQUIRE(store.load_grain_params(4, grain));
  CHECK(grain == make_grain(5));

  store.clear();
  CHECK(!store.has_reference(3));
  CHECK(!store.has_grain_params(4));
  CHECK(!store.has_ccso_params(0, 0));
  CHECK(!store.load_reference(3, loaded));
  std::cout << "PASS overwrite_and_clear\n";
}

// ---------------------------------------------------------------------------
// Invalid input
// ---------------------------------------------------------------------------

void test_invalid_saved_arrays() {
  ReferenceStateStore store;
  std::string error;

  SavedMvs bad_mvs;
  bad_mvs.rows = 3;
  bad_mvs.cols = 5;
  bad_mvs.units.resize(4);
  CHECK(!validate_saved_mvs(bad_mvs, error));

  SavedRefFrames bad_refs;
  bad_refs.rows = 2;
  bad_refs.cols = 2;
  bad_refs.units.resize(3);
  CHECK(!validate_saved_ref_frames(bad_refs, error));

  SavedSegmentIds bad_segments;
  bad_segments.rows = 4;
  bad_segments.cols = 4;
  bad_segments.ids.assign(15, 0);
  CHECK(!validate_saved_segment_ids(bad_segments, error));
  bad_segments.ids.assign(16, 0);
  bad_segments.ids[7] = 16;
  CHECK(!validate_saved_segment_ids(bad_segments, error));

  // The SavedMvs grid must be exactly MiRows>>1 x MiCols>>1.
  ReferenceSlot bad_slot = make_slot(0);
  bad_slot.mvs.rows = 4;
  bad_slot.mvs.units.resize(bad_slot.mvs.rows * bad_slot.mvs.cols);
  CHECK(!validate_reference_slot(bad_slot, error));

  // Failed saves must leave an existing stored value untouched.
  const ReferenceSlot good = make_slot(6);
  CHECK(store.save_reference(0, good, error));
  CHECK(!store.save_reference(0, bad_slot, error));
  ReferenceSlot loaded;
  REQUIRE(store.load_reference(0, loaded));
  CHECK(loaded == good);

  CHECK(!store.save_reference(kNumRefFrames, good, error));
  SavedMvs dummy;
  CHECK(!store.load_mvs(kNumRefFrames, dummy));
  std::cout << "PASS invalid_saved_arrays\n";
}

void test_invalid_grain() {
  std::string error;
  GrainParams good = make_grain(1);
  CHECK(validate_grain_params(good, error));

  GrainParams bad = good;
  bad.config.fgm_id = 8;
  CHECK(!validate_grain_params(bad, error));

  bad = good;
  bad.model.num_y_points = 16;
  CHECK(!validate_grain_params(bad, error));

  bad = good;
  bad.model.y_points.pop_back();
  CHECK(!validate_grain_params(bad, error));

  bad = good;
  bad.model.chroma_scaling_from_luma = true;
  CHECK(!validate_grain_params(bad, error));

  bad = good;
  bad.model.ar_coeffs_y.pop_back();
  CHECK(!validate_grain_params(bad, error));

  bad = good;
  bad.model.bits_per_ar_coeff_y_minus_5 = 0;  // 5 bits: -16..15
  bad.model.ar_coeffs_y[0] = 16;
  CHECK(!validate_grain_params(bad, error));

  bad = good;
  bad.model.clip_to_restricted_range = false;
  bad.model.fg_mc_identity = true;
  CHECK(!validate_grain_params(bad, error));

  bad = good;
  bad.model.num_y_points = 0;
  bad.model.y_points.clear();
  bad.model.ar_coeffs_y.clear();
  bad.model.point_value_increment_bits_minus_1 = 1;
  CHECK(!validate_grain_params(bad, error));

  bad = good;
  bad.fgm_chroma_idc = 4;
  CHECK(!validate_grain_params(bad, error));

  ReferenceStateStore store;
  CHECK(!store.save_grain_params(kNumRefFrames + 1, good, error));
  CHECK(!store.save_grain_params(0, bad, error));
  GrainParams loaded;
  CHECK(!store.load_grain_params(0, loaded));
  std::cout << "PASS invalid_grain\n";
}

void test_invalid_ccso() {
  std::string error;
  CcsoPlaneParams good = make_ccso_plane(3);
  CHECK(validate_ccso_plane(good, error));

  CcsoPlaneParams bad = good;
  bad.scale_idx = 4;
  CHECK(!validate_ccso_plane(bad, error));

  bad = good;
  bad.quant_idx = 4;
  CHECK(!validate_ccso_plane(bad, error));

  bad = good;
  bad.ext_filter = 8;
  CHECK(!validate_ccso_plane(bad, error));

  bad = good;
  bad.edge_clf = 2;
  CHECK(!validate_ccso_plane(bad, error));

  bad = good;
  bad.max_band_log2 = 4;  // non-bo_only codes f(2): 0..3
  CHECK(!validate_ccso_plane(bad, error));

  bad = good;
  bad.bo_only = true;
  bad.quant_idx = 1;
  CHECK(!validate_ccso_plane(bad, error));

  bad = good;
  bad.blks.pop_back();
  CHECK(!validate_ccso_plane(bad, error));

  bad = good;
  bad.blks[0] = 2;
  CHECK(!validate_ccso_plane(bad, error));

  bad = CcsoPlaneParams{};
  bad.scale_idx = 1;  // disabled planes keep every field at its default
  CHECK(!validate_ccso_plane(bad, error));

  CcsoParams frame = make_ccso_frame(1);
  frame.luma_size_log2 = 5;
  CHECK(!validate_ccso_params(frame, error));
  frame = make_ccso_frame(1);
  frame.planes[1].luma_size_log2 = 8;
  CHECK(!validate_ccso_params(frame, error));

  ReferenceStateStore store;
  CHECK(!store.save_ccso_params(kNumRefFrames, 0, good, error));
  CHECK(!store.save_ccso_params(0, kCcsoPlanes, good, error));
  CHECK(!store.save_ccso_frame(kNumRefFrames, make_ccso_frame(1), error));
  CcsoPlaneParams loaded_plane;
  CHECK(!store.load_ccso_params(0, 0, loaded_plane));
  CcsoParams loaded_frame;
  CHECK(!store.load_ccso_frame(0, loaded_frame));
  std::cout << "PASS invalid_ccso\n";
}

}  // namespace

int main() {
  test_saved_mvs_round_trip();
  test_saved_ref_frames_round_trip();
  test_saved_gm_params_round_trip();
  test_saved_segment_ids_round_trip();
  test_reference_slot_round_trip();
  test_individual_saves_coexist();
  test_grain_round_trip();
  test_ccso_round_trip();
  test_ccso_frame_round_trip();
  test_overwrite_and_clear();
  test_invalid_saved_arrays();
  test_invalid_grain();
  test_invalid_ccso();

  if (g_failures != 0) {
    std::cerr << g_failures << " failure(s)\n";
    return 1;
  }
  std::cout << "reference_state_test: all checks passed\n";
  return 0;
}

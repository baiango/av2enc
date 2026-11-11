// Unit tests for the temporal CCSO reuse module.
//
// Three independent oracles are used:
//   1. pinned bit strings derived field-by-field from §5.18.7.12 (Python
//      reference, not this writer),
//   2. the existing non-reuse writer in
//      src/encoder/m1/filters_stream/filters_stream.cpp for streams that code
//      no reuse bits,
//   3. the normative §8.2 entropy decoder for the ccso_blk tile symbols.
#include "encoder/filters/ccso_reuse/ccso_reuse.h"

#include "encoder/m1/filters_stream/filters_stream.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

namespace {

namespace cr = av2enc::ccso_reuse;
namespace fs = av2enc::m1::filters_stream;
namespace av2 = av2enc::av2;
using av2enc::BitWriter;

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const std::string& what) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  }
}

std::string hex_string(std::span<const std::uint8_t> bytes) {
  static constexpr char kDigits[] = "0123456789abcdef";
  std::string out;
  for (std::uint8_t b : bytes) {
    out.push_back(kDigits[b >> 4U]);
    out.push_back(kDigits[b & 0x0fU]);
  }
  return out;
}

cr::PlaneParams bo_plane(int scale, int band,
                         std::vector<std::uint8_t> offsets) {
  cr::PlaneParams p;
  p.enabled = true;
  p.bo_only = true;
  p.scale_idx = scale;
  p.max_band_log2 = band;
  p.offset_idx = std::move(offsets);
  return p;
}

cr::PlaneParams normal_plane(int scale, int quant, int ext, int edge, int band,
                             std::vector<std::uint8_t> offsets) {
  cr::PlaneParams p;
  p.enabled = true;
  p.bo_only = false;
  p.scale_idx = scale;
  p.quant_idx = quant;
  p.ext_filter = ext;
  p.edge_clf = edge;
  p.max_band_log2 = band;
  p.offset_idx = std::move(offsets);
  return p;
}

cr::HeaderContext context_64(bool intra, int nrefs = 1) {
  cr::HeaderContext c;
  c.coded_lossless = false;
  c.enable_ccso = true;
  c.single_picture_header_flag = false;
  c.frame_is_intra = intra;
  c.num_planes = 3;
  c.sb_size = 64;
  c.ccso_unit_matches_sb_size = true;  // CcsoLumaSizeLog2 = 6
  c.tile_start_or = 0;
  c.mi_cols = 64;
  c.mi_rows = 48;
  c.sub_x = 1;
  c.sub_y = 1;
  c.num_total_refs = nrefs;
  return c;
}

std::vector<std::uint8_t> make_blks(int cols, int rows, int value) {
  return std::vector<std::uint8_t>(
      static_cast<std::size_t>(cols) * static_cast<std::size_t>(rows),
      static_cast<std::uint8_t>(value));
}

void save_plane_frame(cr::SlotStore* store, int slot,
                      const cr::FrameParams& params, int luma_size_log2,
                      int mi_cols, int mi_rows, int blk_value,
                      int order_hint = 0, bool valid = true) {
  std::string error;
  std::vector<std::uint8_t> blks[3];
  int cols[3] = {0, 0, 0};
  int rows[3] = {0, 0, 0};
  for (int plane = 0; plane < 3; ++plane) {
    const int width = (mi_cols * 4) >> (plane == 0 ? 0 : 1);
    const int height = (mi_rows * 4) >> (plane == 0 ? 0 : 1);
    cr::ccso_blk_grid_size(width, height, luma_size_log2,
                           plane == 0 ? 0 : 1, plane == 0 ? 0 : 1,
                           &cols[plane], &rows[plane]);
    blks[plane] = make_blks(cols[plane], rows[plane], blk_value);
  }
  cr::SavedFrame frame;
  frame.ccso_luma_size_log2 = luma_size_log2;
  frame.order_hint = order_hint;
  frame.mi_cols = mi_cols;
  frame.mi_rows = mi_rows;
  frame.params = &params;
  frame.blks = blks;
  frame.blk_cols = cols;
  frame.blk_rows = rows;
  check(cr::save_ccso_params(*store, slot, frame, &error),
        "save_ccso_params: " + error);
  store->slot(slot).valid = valid;
}

struct Written {
  std::string hex;
  std::size_t bits = 0;
  std::vector<std::uint8_t> bytes;
};

Written write_row(cr::FrameParams params, const cr::HeaderContext& context) {
  BitWriter writer;
  std::string error;
  const std::size_t bits = cr::write_ccso_params(writer, params, context,
                                                 &error);
  check(error.empty(), "write_ccso_params: " + error);
  Written out;
  out.bits = bits;
  std::span<const std::uint8_t> bytes = writer.bytes();
  out.bytes.assign(bytes.begin(), bytes.end());
  out.hex = hex_string(bytes);
  return out;
}

cr::FrameParams read_row(const Written& written,
                         const cr::HeaderContext& context) {
  cr::BitReader reader(std::span<const std::uint8_t>(written.bytes),
                       written.bits);
  cr::FrameParams params;
  std::string error;
  check(cr::read_ccso_params(reader, context, &params, &error),
        "read_ccso_params: " + error);
  check(reader.bits_left() == 0U, "reader leaves bits unread");
  return params;
}

// ---------------------------------------------------------------------------
// Pinned §5.18.7.12 bit strings
// ---------------------------------------------------------------------------

void test_goldens() {
  // V1: single-picture intra, three bo_only planes, luma size 6.
  {
    cr::HeaderContext c = context_64(true);
    c.single_picture_header_flag = true;
    cr::FrameParams p;
    p.frame_flag = true;
    for (int plane = 0; plane < 3; ++plane) {
      p.plane[plane] = bo_plane(1, 2, {0, 1, 2, 3});
    }
    const Written w = write_row(p, c);
    check(w.hex == "d4b76a5bb52dc0", "V1 golden hex");
    check(w.bits == 51U, "V1 golden bit count");

    cr::FrameParams back = read_row(w, c);
    check(back.frame_flag, "V1 frame flag");
    for (int plane = 0; plane < 3; ++plane) {
      check(back.plane[plane] == p.plane[plane], "V1 plane round-trip");
      check(!back.reuse_ccso[plane] && !back.sb_reuse_ccso[plane],
            "V1 intra reuse bits absent");
    }
  }

  // V2: inter, fresh params, NumTotalRefs = 1 => ccso_ref_idx is zero-width.
  {
    cr::HeaderContext c = context_64(false);
    const int refs[1] = {0};
    c.ref_frame_idx = refs;
    cr::FrameParams p;
    p.frame_flag = true;
    p.plane[0] = bo_plane(2, 1, {0, 1});
    const Written w = write_row(p, c);
    check(w.hex == "cc50", "V2 golden hex");
    check(w.bits == 15U, "V2 golden bit count");
    cr::FrameParams back = read_row(w, c);
    check(back.plane[0] == p.plane[0], "V2 plane round-trip");
    check(!back.plane[1].enabled && !back.plane[2].enabled,
          "V2 disabled planes");
    check(!back.reuse_ccso[0] && !back.sb_reuse_ccso[0],
          "V2 fresh reuse bits");
  }

  // V3: inter, both reuse bits set, NumTotalRefs = 1 (no ref index bits).
  {
    cr::HeaderContext c = context_64(false);
    c.ccso_unit_matches_sb_size = false;  // sb reuse requires luma size 8
    const int refs[1] = {6};
    c.ref_frame_idx = refs;
    cr::SlotStore store;
    cr::FrameParams source;
    source.frame_flag = true;
    source.plane[0] = bo_plane(3, 2, {7, 6, 5, 4});
    save_plane_frame(&store, 6, source, 8, c.mi_cols, c.mi_rows, 1);
    c.slots = &store;

    cr::FrameParams p;
    p.frame_flag = true;
    p.plane[0] = source.plane[0];
    p.reuse_ccso[0] = true;
    p.sb_reuse_ccso[0] = true;
    const Written w = write_row(p, c);
    check(w.hex == "f0", "V3 golden hex");
    check(w.bits == 6U, "V3 golden bit count");
    cr::FrameParams back = read_row(w, c);
    check(back.plane[0] == source.plane[0], "V3 loaded params");
    check(back.reuse_ccso[0] && back.sb_reuse_ccso[0], "V3 reuse bits");
  }

  // V4: ccso_ref_idx f(2) = 3 with NumTotalRefs = 4.
  {
    cr::HeaderContext c = context_64(false, 4);
    c.ccso_unit_matches_sb_size = false;  // sb reuse requires luma size 8
    const int refs[4] = {10, 11, 12, 13};
    c.ref_frame_idx = refs;
    cr::SlotStore store;
    cr::FrameParams source;
    source.frame_flag = true;
    source.plane[1] = bo_plane(0, 0, {5});
    save_plane_frame(&store, 13, source, 8, c.mi_cols, c.mi_rows, 1);
    c.slots = &store;

    cr::FrameParams p;
    p.frame_flag = true;
    p.plane[1] = source.plane[1];
    p.reuse_ccso[1] = true;
    p.sb_reuse_ccso[1] = true;
    p.ccso_ref_idx[1] = 3;
    const Written w = write_row(p, c);
    check(w.hex == "be", "V4 golden hex");
    check(w.bits == 8U, "V4 golden bit count");
    cr::FrameParams back = read_row(w, c);
    check(back.ccso_ref_idx[1] == 3, "V4 ref index");
    check(back.plane[1] == source.plane[1], "V4 loaded params");
  }

  // V5: sb_reuse only, CcsoLumaSizeLog2 = 8, fresh params, ref idx f(1) = 1.
  {
    cr::HeaderContext c = context_64(false, 2);
    c.ccso_unit_matches_sb_size = false;  // single tile => luma size 8
    const int refs[2] = {4, 5};
    c.ref_frame_idx = refs;
    cr::SlotStore store;
    cr::FrameParams source;
    source.frame_flag = true;
    source.plane[0] = bo_plane(0, 0, {0});
    save_plane_frame(&store, 5, source, 8, c.mi_cols, c.mi_rows, 1);
    c.slots = &store;

    cr::FrameParams p;
    p.frame_flag = true;
    p.plane[0] = bo_plane(3, 0, {5});
    p.sb_reuse_ccso[0] = true;
    p.ccso_ref_idx[0] = 1;
    const Written w = write_row(p, c);
    check(w.hex == "df1f00", "V5 golden hex");
    check(w.bits == 19U, "V5 golden bit count");
    cr::FrameParams back = read_row(w, c);
    check(back.plane[0] == p.plane[0], "V5 fresh params kept");
    check(back.sb_reuse_ccso[0], "V5 sb reuse bit");
  }

  // V6: disabled plane, fresh non-band plane, parameter-only reuse.
  {
    cr::HeaderContext c = context_64(false, 2);
    const int refs[2] = {7, 9};
    c.ref_frame_idx = refs;
    cr::SlotStore store;
    cr::FrameParams source;
    source.frame_flag = true;
    source.plane[2] = normal_plane(
        3, 1, 2, 1, 2,
        {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7});
    save_plane_frame(&store, 7, source, 6, c.mi_cols, c.mi_rows, 0);
    c.slots = &store;

    cr::FrameParams p;
    p.frame_flag = true;
    p.plane[1] = normal_plane(1, 2, 3, 1, 1, {0, 1, 2, 3, 4, 5, 6, 7});
    p.plane[2] = source.plane[2];
    p.reuse_ccso[2] = true;
    p.ccso_ref_idx[2] = 0;
    const Written w = write_row(p, c);
    check(w.hex == "a19d5bbdf7eff8", "V6 golden hex");
    check(w.bits == 55U, "V6 golden bit count");
    cr::FrameParams back = read_row(w, c);
    check(!back.plane[0].enabled, "V6 plane0 disabled");
    check(back.plane[1] == p.plane[1], "V6 fresh plane");
    check(back.plane[2] == source.plane[2], "V6 reused plane");
    check(back.reuse_ccso[2] && !back.sb_reuse_ccso[2],
          "V6 parameter-only reuse");
  }
}

// ---------------------------------------------------------------------------
// Default streams: CCSO disabled writes and reads nothing
// ---------------------------------------------------------------------------

void test_default_off() {
  cr::HeaderContext c = context_64(false);
  c.enable_ccso = false;
  cr::FrameParams p;
  p.frame_flag = true;
  p.plane[0] = bo_plane(1, 1, {1, 1});
  BitWriter writer;
  std::string error;
  check(cr::write_ccso_params(writer, p, c, &error) == 0U,
        "disabled CCSO writes no bits");
  check(writer.bit_size() == 0U, "disabled CCSO writer empty");
  check(!p.frame_flag, "disabled CCSO clears frame flag");
  check(!p.plane[0].enabled, "disabled CCSO clears planes");
  check(error.empty(), "disabled CCSO no error");

  cr::BitReader reader(std::span<const std::uint8_t>(), 0);
  cr::FrameParams back;
  check(cr::read_ccso_params(reader, c, &back, &error),
        "disabled CCSO reader");
  check(!back.frame_flag, "disabled CCSO reader frame flag");
  check(reader.bit_pos() == 0U, "disabled CCSO reader consumes nothing");
}

// ---------------------------------------------------------------------------
// Byte equality with the existing non-reuse writer
// ---------------------------------------------------------------------------

void test_existing_writer_equivalence() {
  cr::HeaderContext cc = context_64(false);
  cr::FrameParams p;
  p.frame_flag = true;
  p.plane[0] = bo_plane(2, 1, {0, 1});
  const Written w = write_row(p, cc);

  fs::CcsoHeaderContext fc;
  fc.coded_lossless = false;
  fc.enable_ccso = true;
  fc.single_picture_header_flag = false;
  fc.frame_is_intra = false;
  fc.num_planes = 3;
  fc.sb_size = 64;
  fc.ccso_unit_matches_sb_size = true;
  fc.tile_start_or = 0;
  fs::CcsoParams fp;
  fp.frame_flag = true;
  fs::CcsoPlaneParams& f0 = fp.plane[0];
  f0.enabled = true;
  f0.bo_only = true;
  f0.scale_idx = 2;
  f0.max_band_log2 = 1;
  f0.offset_idx = {0, 1};
  BitWriter writer;
  std::string error;
  const std::size_t bits = fs::write_ccso_params(writer, fp, fc, &error);
  check(error.empty(), "filters_stream writer: " + error);
  check(bits == w.bits, "fresh path bit count matches filters_stream");
  const std::span<const std::uint8_t> bytes = writer.bytes();
  check(hex_string(bytes) == w.hex,
        "fresh path bytes match filters_stream");

  // Single-picture intra form (M1): ccso_frame_flag is inferred 1 and the
  // reuse bits are absent.
  cr::HeaderContext ci = context_64(true);
  ci.single_picture_header_flag = true;
  cr::FrameParams pi;
  pi.frame_flag = true;
  for (int plane = 0; plane < 3; ++plane) {
    pi.plane[plane] = bo_plane(1, 2, {0, 1, 2, 3});
  }
  const Written wi = write_row(pi, ci);
  fs::CcsoHeaderContext fi;
  fi.coded_lossless = false;
  fi.enable_ccso = true;
  fi.single_picture_header_flag = true;
  fi.frame_is_intra = true;
  fi.num_planes = 3;
  fi.sb_size = 64;
  fi.ccso_unit_matches_sb_size = true;
  fi.tile_start_or = 0;
  fs::CcsoParams fpi;
  fpi.frame_flag = true;
  for (int plane = 0; plane < 3; ++plane) {
    fs::CcsoPlaneParams& dst = fpi.plane[plane];
    dst.enabled = true;
    dst.bo_only = true;
    dst.scale_idx = 1;
    dst.max_band_log2 = 2;
    dst.offset_idx = {0, 1, 2, 3};
  }
  BitWriter writer_i;
  error.clear();
  const std::size_t bits_i = fs::write_ccso_params(writer_i, fpi, fi, &error);
  check(error.empty(), "filters_stream intra writer: " + error);
  check(bits_i == wi.bits, "intra bit count matches filters_stream");
  const std::span<const std::uint8_t> bytes_i = writer_i.bytes();
  check(hex_string(bytes_i) == wi.hex,
        "intra bytes match filters_stream");
}

// ---------------------------------------------------------------------------
// §7.23 save / restore and slot lifetimes
// ---------------------------------------------------------------------------

void test_save_restore() {
  const int mi_cols = 64;
  const int mi_rows = 48;
  cr::FrameParams params;
  params.frame_flag = true;
  params.ccso_luma_size_log2 = 8;
  params.plane[0] = bo_plane(2, 3, {0, 1, 2, 3, 4, 5, 6, 7});
  params.plane[1] = normal_plane(1, 2, 3, 1, 1,
                                 {0, 1, 2, 3, 4, 5, 6, 7});
  // plane 2 disabled => SavedCcsoPlanes[slot][2] == 0.

  std::vector<std::uint8_t> blks[3];
  int cols[3] = {0, 0, 0};
  int rows[3] = {0, 0, 0};
  for (int plane = 0; plane < 3; ++plane) {
    const int width = (mi_cols * 4) >> (plane == 0 ? 0 : 1);
    const int height = (mi_rows * 4) >> (plane == 0 ? 0 : 1);
    cr::ccso_blk_grid_size(width, height, 8, plane == 0 ? 0 : 1,
                           plane == 0 ? 0 : 1, &cols[plane], &rows[plane]);
    blks[plane] = make_blks(cols[plane], rows[plane], plane + 1);
  }
  cr::SavedFrame frame;
  frame.ccso_luma_size_log2 = 8;
  frame.order_hint = 7;
  frame.mi_cols = mi_cols;
  frame.mi_rows = mi_rows;
  frame.params = &params;
  frame.blks = blks;
  frame.blk_cols = cols;
  frame.blk_rows = rows;

  cr::SlotStore store;
  std::string error;
  check(cr::save_ccso_params(store, 4, frame, &error),
        "save slot: " + error);
  check(store.slot(4).valid, "slot valid");
  check(store.slot(4).order_hint == 7, "slot order hint");
  check(store.slot(4).plane[0].saved, "plane0 SavedCcsoPlanes");
  check(store.slot(4).plane[1].saved, "plane1 SavedCcsoPlanes");
  check(!store.slot(4).plane[2].saved, "plane2 SavedCcsoPlanes == 0");
  check(store.slot(4).plane[0].blks == blks[0], "plane0 blks saved");
  check(store.slot(4).plane[1].blks == blks[1], "plane1 blks saved");

  // resolve() reproduces the effective frame, including CcsoBlks.
  cr::HeaderContext c = context_64(false);
  c.ccso_unit_matches_sb_size = false;
  c.mi_cols = mi_cols;
  c.mi_rows = mi_rows;
  const int refs[1] = {4};
  c.ref_frame_idx = refs;
  c.slots = &store;
  cr::FrameParams syntax;
  syntax.frame_flag = true;
  syntax.ccso_luma_size_log2 = 8;
  syntax.plane[0].enabled = true;
  syntax.plane[1].enabled = true;
  syntax.reuse_ccso[0] = true;
  syntax.sb_reuse_ccso[0] = true;
  syntax.reuse_ccso[1] = true;
  syntax.sb_reuse_ccso[1] = true;
  cr::EffectiveFrame effective;
  check(cr::resolve_ccso_params(c, syntax, store, &effective, &error),
        "resolve saved slot: " + error);
  check(effective.plane[0].params == params.plane[0], "resolve params p0");
  check(effective.plane[1].params == params.plane[1], "resolve params p1");
  check(effective.plane[0].blks == blks[0], "resolve blks p0");
  check(effective.plane[1].blks == blks[1], "resolve blks p1");
  check(effective.plane[0].ref_slot == 4, "resolve ref slot");
  check(effective.plane[2].params.enabled == false, "resolve disabled p2");

  // Slot lifetime: a later save to the same slot replaces the old state.
  cr::FrameParams newer = params;
  newer.plane[0] = bo_plane(0, 0, {7});
  cr::SavedFrame frame2 = frame;
  frame2.params = &newer;
  std::vector<std::uint8_t> blks2[3] = {make_blks(cols[0], rows[0], 0),
                                        blks[1], blks[2]};
  frame2.blks = blks2;
  check(cr::save_ccso_params(store, 4, frame2, &error),
        "overwrite slot: " + error);
  cr::EffectiveFrame newer_effective;
  check(cr::resolve_ccso_params(c, syntax, store, &newer_effective, &error),
        "resolve overwritten slot: " + error);
  check(newer_effective.plane[0].params == newer.plane[0],
        "slot lifetime returns newest params");
  check(newer_effective.plane[0].blks == blks2[0],
        "slot lifetime returns newest blks");

  store.reset();
  check(!store.slot(4).valid, "reset clears slot validity");
}

// ---------------------------------------------------------------------------
// use_ccso() reference selection
// ---------------------------------------------------------------------------

void test_use_ccso() {
  cr::HeaderContext c = context_64(false);
  c.ccso_unit_matches_sb_size = false;
  c.mi_cols = 64;
  c.mi_rows = 48;
  cr::SlotStore store;

  cr::PlaneParams a = bo_plane(1, 1, {4, 5});
  cr::PlaneParams b = bo_plane(2, 2, {0, 1, 2, 3});
  cr::PlaneParams unknown = bo_plane(0, 0, {3});

  cr::FrameParams params_a;
  params_a.frame_flag = true;
  params_a.plane[0] = a;
  cr::FrameParams params_b;
  params_b.frame_flag = true;
  params_b.plane[0] = b;
  save_plane_frame(&store, 2, params_a, 8, c.mi_cols, c.mi_rows, 0);
  save_plane_frame(&store, 3, params_b, 8, c.mi_cols, c.mi_rows, 1);
  save_plane_frame(&store, 5, params_b, 8, c.mi_cols, c.mi_rows, 0);
  save_plane_frame(&store, 6, params_a, 8, c.mi_cols, c.mi_rows, 1);
  save_plane_frame(&store, 7, params_a, 8, c.mi_cols, c.mi_rows, 0);

  std::vector<std::uint8_t> ones = make_blks(1, 1, 1);
  std::string error;

  // Both parameter and block reuse; index 0 does not match, index 1 does.
  {
    const int refs[2] = {2, 3};
    c.ref_frame_idx = refs;
    c.num_total_refs = 2;
    cr::ReuseChoice choice;
    check(cr::use_ccso(b, ones, 1, 1, 0, c, store, &choice, &error),
          "use_ccso both: " + error);
    check(choice.reuse_ccso && choice.sb_reuse_ccso,
          "use_ccso both reuse bits");
    check(choice.ccso_ref_idx == 1 && choice.ref_slot == 3,
          "use_ccso selects matching reference index");
  }

  // Parameters match (slot 5), block flags differ => parameter-only reuse.
  {
    const int refs[1] = {5};
    c.ref_frame_idx = refs;
    c.num_total_refs = 1;
    cr::ReuseChoice choice;
    check(cr::use_ccso(b, ones, 1, 1, 0, c, store, &choice, &error),
          "use_ccso params only: " + error);
    check(choice.reuse_ccso && !choice.sb_reuse_ccso,
          "use_ccso parameter-only choice");
    check(choice.ccso_ref_idx == 0 && choice.ref_slot == 5,
          "use_ccso parameter-only ref");
  }

  // Block flags match (slot 6), parameters differ => sb-only reuse.
  {
    const int refs[1] = {6};
    c.ref_frame_idx = refs;
    c.num_total_refs = 1;
    cr::ReuseChoice choice;
    check(cr::use_ccso(b, ones, 1, 1, 0, c, store, &choice, &error),
          "use_ccso sb only: " + error);
    check(!choice.reuse_ccso && choice.sb_reuse_ccso,
          "use_ccso sb-only choice");
    check(choice.ccso_ref_idx == 0 && choice.ref_slot == 6,
          "use_ccso sb-only ref");
  }

  // Neither matches (slot 7) => no reuse.
  {
    const int refs[1] = {7};
    c.ref_frame_idx = refs;
    c.num_total_refs = 1;
    cr::ReuseChoice choice;
    check(cr::use_ccso(unknown, ones, 1, 1, 0, c, store, &choice, &error),
          "use_ccso no match: " + error);
    check(!choice.reuse_ccso && !choice.sb_reuse_ccso, "use_ccso no reuse");
  }

  // Intra frames never reuse.
  {
    cr::HeaderContext intra = c;
    intra.frame_is_intra = true;
    cr::ReuseChoice choice;
    check(cr::use_ccso(b, ones, 1, 1, 0, intra, store, &choice, &error),
          "use_ccso intra: " + error);
    check(!choice.reuse_ccso && !choice.sb_reuse_ccso,
          "intra infers reuse bits 0");
  }
}

// ---------------------------------------------------------------------------
// Validation: every §6.17.7.8 requirement fails loudly
// ---------------------------------------------------------------------------

void expect_write_fails(const cr::FrameParams& params,
                        const cr::HeaderContext& context,
                        const std::string& what) {
  cr::FrameParams copy = params;
  BitWriter writer;
  std::string error;
  const std::size_t bits = cr::write_ccso_params(writer, copy, context,
                                                 &error);
  check(bits == 0U && !error.empty(), what + " (write)");
}

void test_validation() {
  // Invalid fresh plane parameters.
  {
    cr::HeaderContext c = context_64(false);
    cr::FrameParams p;
    p.frame_flag = true;
    cr::PlaneParams bad = bo_plane(1, 1, {0, 1});
    bad.quant_idx = 2;
    p.plane[0] = bad;
    expect_write_fails(p, c, "bo_only with non-zero quant");

    p.plane[0] = bo_plane(1, 1, {0, 1});
    p.plane[0].ext_filter = 7;
    expect_write_fails(p, c, "ext_filter == 7");

    p.plane[0] = normal_plane(1, 2, 2, 1, 4, {});
    expect_write_fails(p, c, "max_band_log2 4 with bo_only = 0");

    p.plane[0] = bo_plane(1, 1, {0});
    expect_write_fails(p, c, "offset count mismatch");
  }

  // ccso_ref_idx outside NumTotalRefs.
  {
    cr::HeaderContext c = context_64(false, 2);
    const int refs[2] = {0, 1};
    c.ref_frame_idx = refs;
    cr::SlotStore store;
    cr::FrameParams source;
    source.frame_flag = true;
    source.plane[0] = bo_plane(0, 0, {0});
    save_plane_frame(&store, 0, source, 8, c.mi_cols, c.mi_rows, 1);
    c.slots = &store;
    cr::FrameParams p;
    p.frame_flag = true;
    p.plane[0] = source.plane[0];
    p.reuse_ccso[0] = true;
    p.ccso_ref_idx[0] = 2;
    expect_write_fails(p, c, "ccso_ref_idx >= NumTotalRefs");
  }

  // Reference slot unavailable / not a valid reference.
  {
    cr::HeaderContext c = context_64(false);
    const int refs[1] = {9};
    c.ref_frame_idx = refs;
    cr::SlotStore store;
    cr::FrameParams source;
    source.frame_flag = true;
    source.plane[0] = bo_plane(0, 0, {0});
    save_plane_frame(&store, 9, source, 8, c.mi_cols, c.mi_rows, 1,
                     /*order_hint=*/0, /*valid=*/false);
    c.slots = &store;
    cr::FrameParams p;
    p.frame_flag = true;
    p.plane[0] = source.plane[0];
    p.reuse_ccso[0] = true;
    expect_write_fails(p, c, "reference slot not RefValid");
  }

  // SavedCcsoPlanes[idx][plane] == 0.
  {
    cr::HeaderContext c = context_64(false);
    const int refs[1] = {1};
    c.ref_frame_idx = refs;
    cr::SlotStore store;
    cr::FrameParams source;  // plane disabled => SavedCcsoPlanes == 0
    source.frame_flag = true;
    save_plane_frame(&store, 1, source, 8, c.mi_cols, c.mi_rows, 1);
    c.slots = &store;
    cr::FrameParams p;
    p.frame_flag = true;
    p.plane[0] = bo_plane(0, 0, {0});
    p.reuse_ccso[0] = true;
    expect_write_fails(p, c, "SavedCcsoPlanes == 0");
  }

  // RESTRICTED_OH reference.
  {
    cr::HeaderContext c = context_64(false);
    const int refs[1] = {0};
    c.ref_frame_idx = refs;
    cr::SlotStore store;
    cr::FrameParams source;
    source.frame_flag = true;
    source.plane[0] = bo_plane(0, 0, {0});
    save_plane_frame(&store, 0, source, 8, c.mi_cols, c.mi_rows, 1,
                     cr::kRestrictedOrderHint);
    c.slots = &store;
    cr::FrameParams p;
    p.frame_flag = true;
    p.plane[0] = source.plane[0];
    p.reuse_ccso[0] = true;
    expect_write_fails(p, c, "RESTRICTED_OH reference");
  }

  // sb_reuse requires CcsoLumaSizeLog2 == 8 (64x64 matching SB gives 6).
  {
    cr::HeaderContext c = context_64(false);  // luma size 6
    const int refs[1] = {0};
    c.ref_frame_idx = refs;
    cr::SlotStore store;
    cr::FrameParams source;
    source.frame_flag = true;
    source.plane[0] = bo_plane(0, 0, {0});
    save_plane_frame(&store, 0, source, 6, c.mi_cols, c.mi_rows, 1);
    c.slots = &store;
    cr::FrameParams p;
    p.frame_flag = true;
    p.plane[0] = source.plane[0];
    p.sb_reuse_ccso[0] = true;
    expect_write_fails(p, c, "sb_reuse with CcsoLumaSizeLog2 != 8");
  }

  // sb_reuse geometry mismatch.
  {
    cr::HeaderContext c = context_64(false);
    c.ccso_unit_matches_sb_size = false;
    c.mi_cols = 64;
    c.mi_rows = 48;
    const int refs[1] = {0};
    c.ref_frame_idx = refs;
    cr::SlotStore store;
    cr::FrameParams source;
    source.frame_flag = true;
    source.plane[0] = bo_plane(0, 0, {0});
    save_plane_frame(&store, 0, source, 8, 64, 32, 1);  // different MiRows
    c.slots = &store;
    cr::FrameParams p;
    p.frame_flag = true;
    p.plane[0] = source.plane[0];
    p.sb_reuse_ccso[0] = true;
    expect_write_fails(p, c, "sb_reuse MiRows mismatch");
  }

  // Parameter reuse values must match the saved slot.
  {
    cr::HeaderContext c = context_64(false);
    const int refs[1] = {0};
    c.ref_frame_idx = refs;
    cr::SlotStore store;
    cr::FrameParams source;
    source.frame_flag = true;
    source.plane[0] = bo_plane(0, 0, {0});
    save_plane_frame(&store, 0, source, 6, c.mi_cols, c.mi_rows, 1);
    c.slots = &store;
    cr::FrameParams p;
    p.frame_flag = true;
    p.plane[0] = bo_plane(3, 3, {7});
    p.reuse_ccso[0] = true;
    expect_write_fails(p, c, "reused parameters differ from saved slot");
  }

  // Reader side: reuse without a slot store is rejected, not guessed.
  {
    cr::HeaderContext c = context_64(false);
    const int refs[1] = {0};
    c.ref_frame_idx = refs;
    c.slots = nullptr;
    const std::vector<std::uint8_t> bytes = {0xf0};
    cr::BitReader reader(std::span<const std::uint8_t>(bytes), 4);
    cr::FrameParams back;
    std::string error;
    check(!cr::read_ccso_params(reader, c, &back, &error),
          "reader requires saved slots for reuse");
    check(!error.empty(), "reader reports missing slots");
  }
}

// ---------------------------------------------------------------------------
// §5.20.10.2 ccso_blk symbols
// ---------------------------------------------------------------------------

Written encode_tile(cr::TileState* state, std::vector<std::uint8_t>& buf,
                    std::size_t* payload, std::string* error) {
  av2::Encoder encoder;
  check(encoder.init_symbol(buf.data(), buf.size()) == av2::Status::kOk,
        "encoder init");
  encoder.disable_cdf_update = 1;
  const int mi_cols = state->mi_cols();
  const int mi_rows = state->mi_rows();
  const int step = 1 << (state->ccso_luma_size_log2() - 2);
  for (int r = 0; r < mi_rows; r += step) {
    for (int c = 0; c < mi_cols; c += step) {
      check(state->write_block(encoder, r, c, error),
            "write_block: " + *error);
    }
  }
  check(encoder.exit_symbol(payload) == av2::Status::kOk, "exit_symbol");
  Written out;
  out.bits = *payload * 8U;
  out.bytes.assign(buf.begin(), buf.begin() +
                                  static_cast<std::ptrdiff_t>(*payload));
  out.hex = hex_string(out.bytes);
  return out;
}

void test_tile_symbols() {
  std::string error;
  cr::HeaderContext c = context_64(false);
  c.ccso_unit_matches_sb_size = false;
  c.mi_cols = 64;  // 256 luma samples
  c.mi_rows = 48;  // 192 luma samples
  const int refs[1] = {0};
  c.ref_frame_idx = refs;
  cr::SlotStore store;
  cr::FrameParams source;
  source.frame_flag = true;
  source.plane[0] = bo_plane(0, 0, {0});
  source.plane[1] = bo_plane(0, 0, {0});
  source.plane[2] = bo_plane(0, 0, {0});
  save_plane_frame(&store, 0, source, 8, c.mi_cols, c.mi_rows, 1);
  c.slots = &store;

  // Fresh frame: all three planes enabled, all unit flags set.
  cr::FrameParams fresh = source;
  fresh.frame_flag = true;
  cr::EffectiveFrame effective;
  check(cr::resolve_ccso_params(c, fresh, store, &effective, &error),
        "tile resolve fresh: " + error);
  cr::TileState enc_state;
  check(enc_state.init(effective, c.mi_cols, c.mi_rows, 1, 1, &error),
        "tile init fresh: " + error);
  for (int plane = 0; plane < 3; ++plane) {
    check(enc_state.codes_block_flags(plane), "fresh codes block flags");
    const std::vector<std::uint8_t> want =
        make_blks(enc_state.blk_cols(plane), enc_state.blk_rows(plane), 1);
    check(enc_state.set_block_flags(plane, want.data(),
                                    enc_state.blk_cols(plane),
                                    enc_state.blk_rows(plane), &error),
          "set_block_flags: " + error);
  }
  std::vector<std::uint8_t> buffer(4096);
  std::size_t payload = 0;
  const Written tile = encode_tile(&enc_state, buffer, &payload, &error);
  check(!tile.bytes.empty(), "tile payload non-empty");
  check(tile.hex == "fc0000", "tile payload regression pin");

  cr::TileState dec_state;
  check(dec_state.init(effective, c.mi_cols, c.mi_rows, 1, 1, &error),
        "tile init decode: " + error);
  av2::Decoder decoder;
  check(decoder.init_symbol(buffer.data(), buffer.size(), payload) ==
            av2::Status::kOk,
        "decoder init");
  decoder.disable_cdf_update = 1;
  const int step = 1 << (dec_state.ccso_luma_size_log2() - 2);
  for (int r = 0; r < c.mi_rows; r += step) {
    for (int col = 0; col < c.mi_cols; col += step) {
      int blk[3] = {0, 0, 0};
      check(dec_state.read_block(decoder, r, col, blk, &error),
            "read_block: " + error);
      check(blk[0] == 1 && blk[1] == 1 && blk[2] == 1,
            "decoded ccso_blk symbols");
    }
  }
  check(decoder.exit_symbol() == av2::Status::kOk, "decoder exit");
  for (int plane = 0; plane < 3; ++plane) {
    check(dec_state.blks(plane) == enc_state.blks(plane),
          "tile round-trip CcsoBlks");
  }

  // sb_reuse frame: no ccso_blk symbols at all; saved flags are inherited.
  cr::FrameParams reuse = fresh;
  for (int plane = 0; plane < 3; ++plane) {
    reuse.reuse_ccso[plane] = true;
    reuse.sb_reuse_ccso[plane] = true;
  }
  cr::EffectiveFrame reused;
  check(cr::resolve_ccso_params(c, reuse, store, &reused, &error),
        "tile resolve reuse: " + error);
  cr::TileState reuse_state;
  check(reuse_state.init(reused, c.mi_cols, c.mi_rows, 1, 1, &error),
        "tile init reuse: " + error);
  for (int plane = 0; plane < 3; ++plane) {
    check(!reuse_state.codes_block_flags(plane),
          "sb_reuse codes no block flags");
    check(reuse_state.blks(plane) ==
              make_blks(reuse_state.blk_cols(plane),
                        reuse_state.blk_rows(plane), 1),
          "sb_reuse inherits saved flags");
  }
  std::size_t reuse_payload = 0;
  const Written reuse_tile =
      encode_tile(&reuse_state, buffer, &reuse_payload, &error);
  check(reuse_payload == 2U, "sb_reuse emits only tile termination bytes");
  check(reuse_tile.hex == "8000", "sb_reuse tile payload pin");
}

}  // namespace

int main() {
  test_goldens();
  test_default_off();
  test_existing_writer_equivalence();
  test_save_restore();
  test_use_ccso();
  test_validation();
  test_tile_symbols();
  std::printf("ccso_reuse_test: %d checks, %d failures\n", g_checks,
              g_failures);
  return g_failures == 0 ? 0 : 1;
}

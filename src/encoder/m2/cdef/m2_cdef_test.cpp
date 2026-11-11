// M2 CDEF adapter tests (host, no CUDA).
//
// Covers:
//   * geometry / metadata validation;
//   * single-strength planning: zero index bits and every cdef_idx == 0;
//   * multi-strength planning: legal index grid, exact §5.20.10.1 symbol
//     round-trip (including the neighbour context and the no-op repeat call);
//   * §5.18.7.10 header bits: the writer's bit count matches the reported
//     count and the §5.18.7.10 reader recovers the same parameters;
//   * the adapter's filter output equals cdef_frame() exactly;
//   * skip / lossless / chroma-lossless copy behaviour;
//   * tile clipping;
//   * deterministic planning on real reconstructed content.
//
// Build/run: ./build_and_test.sh in this directory.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "bitstream/bitwriter.h"
#include "cuda/m1/cdef/m1_cdef_index_syntax.h"
#include "encoder/filters/cdef/cdef.h"
#include "encoder/filters/cdef/cdef_search.h"
#include "encoder/filters/cdef/cdef_test_common.h"
#include "encoder/m1/filters_stream/filters_stream.h"
#include "encoder/m2/cdef/m2_cdef.h"

namespace {

using av2enc::BitWriter;
using av2enc::av2::Decoder;
using av2enc::av2::Encoder;
using av2enc::av2::Status;
using av2enc::cdef::CdefStrength;
using av2enc::cdef::test::Pattern;
using av2enc::cdef::test::TestFrame;
using av2enc::m2::cdef_adapter::FrameGrids;
using av2enc::m2::cdef_adapter::FrameView;
using av2enc::m2::cdef_adapter::Options;
using av2enc::m2::cdef_adapter::Plan;

int g_checks = 0;
int g_failures = 0;

void check(bool ok, const char* what) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::fprintf(stderr, "FAIL: %s\n", what);
  }
}

struct Fixture {
  TestFrame recon;
  TestFrame ref;
  FrameGrids grids;
  FrameView view;

  void build(int mi_cols, int mi_rows, unsigned seed, bool noise) {
    ref = av2enc::cdef::test::make_frame(mi_cols, mi_rows);
    av2enc::cdef::test::fill_pattern(ref, Pattern::kRipple);
    recon = av2enc::cdef::test::make_frame(mi_cols, mi_rows);
    std::mt19937 rng(seed);
    for (int plane = 0; plane < 3; ++plane) {
      const int w = recon.plane_width(plane);
      const int h = recon.plane_height(plane);
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          int value = ref.src[plane][av2enc::cdef::test::pix_index(
              y, ref.stride[plane], x)];
          if (noise) {
            value += static_cast<int>(rng() % 13) - 6;
          }
          value = value < 0 ? 0 : (value > 255 ? 255 : value);
          recon.src[plane][av2enc::cdef::test::pix_index(
              y, recon.stride[plane], x)] = static_cast<std::uint8_t>(value);
        }
      }
    }
    grids.zero(mi_cols, mi_rows);
    view = FrameView{};
    for (int plane = 0; plane < 3; ++plane) {
      view.recon[plane].data = recon.src[plane].data();
      view.recon[plane].stride = recon.stride[plane];
      view.source[plane].data = ref.src[plane].data();
      view.source[plane].stride = ref.stride[plane];
    }
    view.width = mi_cols * 4;
    view.height = mi_rows * 4;
    view.mi_cols = mi_cols;
    view.mi_rows = mi_rows;
    grids.bind(&view);
  }

  Plan manual(const CdefStrength& strength) const {
    Plan plan;
    plan.num_strengths = 1;
    plan.mi_cols = recon.mi_cols;
    plan.mi_rows = recon.mi_rows;
    plan.n64_cols = recon.n64_cols;
    plan.n64_rows = recon.n64_rows;
    plan.options.num_strengths = 1;
    plan.options.strengths[0] = strength;
    plan.options.damping = 3;
    plan.cdef_idx.assign(static_cast<std::size_t>(recon.n64_cols) *
                             static_cast<std::size_t>(recon.n64_rows),
                         0);
    return plan;
  }
};

// SSE of the filtered destination against a reference source.
std::int64_t filtered_sse(const TestFrame& filtered, int plane,
                          const TestFrame& ref) {
  const int w = filtered.plane_width(plane);
  const int h = filtered.plane_height(plane);
  std::int64_t sse = 0;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const int va = filtered.dst[plane][av2enc::cdef::test::pix_index(
          y, filtered.stride[plane], x)];
      const int vb = ref.src[plane][av2enc::cdef::test::pix_index(
          y, ref.stride[plane], x)];
      const int d = va - vb;
      sse += static_cast<std::int64_t>(d) * d;
    }
  }
  return sse;
}

// Compares two filtered frames (both destinations).
bool filtered_equal(const TestFrame& a, const TestFrame& b, int plane) {
  const int w = a.plane_width(plane);
  const int h = a.plane_height(plane);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      if (a.dst[plane][av2enc::cdef::test::pix_index(y, a.stride[plane], x)] !=
          b.dst[plane][av2enc::cdef::test::pix_index(y, b.stride[plane], x)]) {
        return false;
      }
    }
  }
  return true;
}

// Compares a filtered destination against an unfiltered source.
bool filtered_equals_source(const TestFrame& filtered, int plane,
                            const TestFrame& src) {
  const int w = filtered.plane_width(plane);
  const int h = filtered.plane_height(plane);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      if (filtered.dst[plane][av2enc::cdef::test::pix_index(
              y, filtered.stride[plane], x)] !=
          src.src[plane][av2enc::cdef::test::pix_index(y, src.stride[plane],
                                                       x)]) {
        return false;
      }
    }
  }
  return true;
}

void test_validation() {
  Fixture fx;
  fx.build(20, 12, 7, true);
  Plan plan;
  std::string error;

  Options options;
  options.num_strengths = 1;
  check(av2enc::m2::cdef_adapter::plan(fx.view, options, &plan, &error),
        "plan builds");

  FrameView bad = fx.view;
  bad.skip = nullptr;
  check(!av2enc::m2::cdef_adapter::plan(bad, options, &plan, &error),
        "missing skip grid rejected");
  bad = fx.view;
  bad.width = 81;
  check(!av2enc::m2::cdef_adapter::plan(bad, options, &plan, &error),
        "non-multiple-of-8 width rejected");
  bad = fx.view;
  bad.recon[1].stride = 4;
  check(!av2enc::m2::cdef_adapter::plan(bad, options, &plan, &error),
        "short recon stride rejected");

  Options illegal = options;
  illegal.num_strengths = 9;
  check(!av2enc::m2::cdef_adapter::plan(fx.view, illegal, &plan, &error),
        "nine strengths rejected");
  illegal = options;
  illegal.cdef_on_skip_txfm = 0;
  illegal.on_skip_txfm = true;
  check(!av2enc::m2::cdef_adapter::plan(fx.view, illegal, &plan, &error),
        "DISABLED with on_skip_txfm rejected");
  illegal = options;
  illegal.cdef_on_skip_txfm = 1;
  illegal.on_skip_txfm = false;
  check(!av2enc::m2::cdef_adapter::plan(fx.view, illegal, &plan, &error),
        "ALWAYS_ON without on_skip_txfm rejected");

  // filter: null/aliasing destinations.
  TestFrame out = av2enc::cdef::test::make_frame(fx.recon.mi_cols,
                                                 fx.recon.mi_rows);
  std::uint8_t* dst[3] = {out.dst[0].data(), out.dst[1].data(),
                          out.dst[2].data()};
  int stride[3] = {out.stride[0], out.stride[1], out.stride[2]};
  check(!av2enc::m2::cdef_adapter::filter(fx.view, plan, nullptr, stride, &error),
        "null destination rejected");
  std::uint8_t* alias[3] = {const_cast<std::uint8_t*>(fx.view.recon[0].data),
                            dst[1], dst[2]};
  check(!av2enc::m2::cdef_adapter::filter(fx.view, plan, alias, stride, &error),
        "aliasing destination rejected");
  TestFrame wide = av2enc::cdef::test::make_frame(fx.recon.mi_cols,
                                                  fx.recon.mi_rows, 16, 2);
  std::uint8_t* wide_dst[3] = {wide.dst[0].data(), wide.dst[1].data(),
                               wide.dst[2].data()};
  int wide_stride[3] = {wide.stride[0], wide.stride[1], wide.stride[2]};
  check(!av2enc::m2::cdef_adapter::filter(fx.view, plan, wide_dst, wide_stride,
                                          &error),
        "mismatched source/destination stride rejected");
}

void test_single_strength() {
  Fixture fx;
  fx.build(20, 12, 11, true);
  Options options;
  options.num_strengths = 1;
  Plan plan;
  std::string error;
  check(av2enc::m2::cdef_adapter::plan(fx.view, options, &plan, &error),
        "single-strength plan");
  check(plan.num_strengths == 1, "one strength");
  check(plan.forced_single_strength, "search flags a forced single strength");
  check(plan.header_only_emittable, "single strength is header-only emittable");
  check(plan.index_bits == 0, "single strength costs no index bits");
  check(std::all_of(plan.cdef_idx.begin(), plan.cdef_idx.end(),
                    [](std::int8_t v) { return v == 0; }),
        "single strength forces index 0");
  check(plan.params.frame_enable, "header frame_enable set");
  check(plan.params.damping_minus_3 == 0, "damping 3");
  check(plan.params.num_strengths == 1, "one header strength");
  check(plan.params.strengths_minus_1 == 0, "strengths_minus_1 = 0");

  BitWriter writer;
  const std::size_t bits = av2enc::m2::cdef_adapter::write_frame_header_cdef(
      writer, plan, /*single_picture_header=*/false,
      /*cdef_on_skip_txfm=*/0, &error);
  check(error.empty(), "header write succeeds");
  check(static_cast<int>(bits) == plan.header_bits,
        "reported header bits match the writer");
  int expected_bits = 6;  // frame_enable + damping(2) + strengths(3), mode 0
  for (int i = 0; i < plan.num_strengths; ++i) {
    const auto& set = plan.params.set[i];
    expected_bits += 1 + (set.y_pri != 0 ? 4 : 0) + 2;
    expected_bits += 1 + (set.uv_pri != 0 ? 4 : 0) + 2;
  }
  check(static_cast<int>(bits) == expected_bits,
        "M2 single-strength header bit count");
  const std::vector<std::uint8_t> bytes = writer.take_bytes();
  check(bytes.size() == (bits + 7) / 8, "header bytes cover the bit count");

  // Read it back with the §5.18.7.10 mirror.
  av2enc::m1::filters_stream::BitReader reader(bytes, bits);
  av2enc::m1::filters_stream::CdefHeaderContext context;
  context.coded_lossless = false;
  context.enable_cdef = true;
  context.single_picture_header_flag = false;
  context.cdef_on_skip_txfm = 0;
  context.num_planes = 3;
  av2enc::m1::filters_stream::CdefParams decoded;
  check(av2enc::m1::filters_stream::read_cdef_params(reader, context, &decoded,
                                                     &error),
        "header reads back");
  check(decoded.frame_enable && decoded.damping_minus_3 == 0 &&
            decoded.strengths_minus_1 == 0,
        "decoded header matches");
  bool sets_equal = true;
  for (int i = 0; i < plan.num_strengths; ++i) {
    sets_equal = sets_equal && decoded.set[i].y_pri == plan.params.set[i].y_pri &&
                 decoded.set[i].y_sec == plan.params.set[i].y_sec &&
                 decoded.set[i].uv_pri == plan.params.set[i].uv_pri &&
                 decoded.set[i].uv_sec == plan.params.set[i].uv_sec;
  }
  check(sets_equal, "decoded strength sets match");

  // The single-picture (M1) form adds no frame_enable bit but does code the
  // ADAPTIVE skip flag, so it is one bit longer here.
  BitWriter m1_writer;
  const std::size_t m1_bits = av2enc::m2::cdef_adapter::write_frame_header_cdef(
      m1_writer, plan, /*single_picture_header=*/true, /*cdef_on_skip_txfm=*/2,
      &error);
  // Single-picture form: no frame_enable bit, one ADAPTIVE skip bit, so for
  // the default (skip disabled) configuration the count is identical.
  check(m1_bits == bits, "single-picture header swaps the enable/skip bits");

  // SSE accounting: filter output must be the samples the plan costed.
  TestFrame out = av2enc::cdef::test::make_frame(fx.recon.mi_cols,
                                                 fx.recon.mi_rows);
  std::uint8_t* dst[3] = {out.dst[0].data(), out.dst[1].data(),
                          out.dst[2].data()};
  int stride[3] = {out.stride[0], out.stride[1], out.stride[2]};
  check(av2enc::m2::cdef_adapter::filter(fx.view, plan, dst, stride, &error),
        "single-strength filter");
  std::int64_t sse = 0;
  for (int plane = 0; plane < 3; ++plane) {
    sse += filtered_sse(out, plane, fx.ref);
  }
  check(sse == plan.sse_after, "plan SSE equals the filtered samples");
  check(plan.sse_after <= plan.sse_before, "filtering never hurts");
}

void test_multi_strength_index() {
  for (int strengths : {2, 4, 8}) {
    Fixture fx;
    fx.build(50, 34, static_cast<unsigned>(strengths) * 13 + 1, true);
    Options options;
    options.num_strengths = strengths;
    Plan plan;
    std::string error;
    check(av2enc::m2::cdef_adapter::plan(fx.view, options, &plan, &error),
          "multi-strength plan");
    check(plan.num_strengths >= 1 && plan.num_strengths <= strengths,
          "strength count within the request");
    check(!plan.header_only_emittable || plan.num_strengths == 1,
          "header-only flag matches the strength count");

    for (std::int8_t idx : plan.cdef_idx) {
      check(idx >= 0 && idx < plan.num_strengths, "index in range");
    }
    if (plan.num_strengths == 1) continue;

    // Every coded set is used at least once (the search prunes).
    std::vector<int> used(static_cast<std::size_t>(plan.num_strengths), 0);
    for (std::int8_t idx : plan.cdef_idx) used[static_cast<std::size_t>(idx)]++;
    for (int i = 0; i < plan.num_strengths; ++i) {
      check(used[static_cast<std::size_t>(i)] > 0, "coded set is used");
    }

    std::vector<std::uint8_t> payload;
    check(av2enc::m2::cdef_adapter::write_index_symbols(plan, &payload, &error),
          "index symbols write");
    check(!payload.empty(), "index payload non-empty");
    check(static_cast<int>(payload.size() * 8U) == plan.index_bits,
          "reported index bits match the payload");
    std::vector<std::int8_t> decoded;
    check(av2enc::m2::cdef_adapter::read_index_symbols(payload, plan, &decoded, &error),
          "index symbols read");
    check(decoded == plan.cdef_idx, "decoded index grid matches the plan");

    // The tile writer must be a no-op on the second call for a 64x64.
    std::vector<std::uint8_t> buffer(1U << 16, 0);
    Encoder encoder;
    check(encoder.init_symbol(buffer.data(), buffer.size()) == Status::kOk,
          "encoder init");
    av2enc::m2::cdef_adapter::TileIndexWriter tile;
    check(tile.init(fx.recon.mi_cols, fx.recon.mi_rows, plan.num_strengths, 4,
                    &error),
          "tile writer init");
    bool ok = true;
    for (int br = 0; br < plan.n64_rows && ok; ++br) {
      for (int bc = 0; bc < plan.n64_cols && ok; ++bc) {
        tile.clear_superblock(br * 16, bc * 16);
        const std::size_t at = static_cast<std::size_t>(br) *
                                   static_cast<std::size_t>(plan.n64_cols) +
                               static_cast<std::size_t>(bc);
        ok = tile.write(encoder, br * 16, bc * 16,
                        static_cast<int>(plan.cdef_idx[at]), &error);
        // Repeat call inside the same 64x64: must code nothing.
        ok = ok && tile.write(encoder, br * 16, bc * 16,
                              static_cast<int>(plan.cdef_idx[at]), &error);
      }
    }
    check(ok, "tile writer codes every 64x64 once");
    std::size_t bytes = 0;
    check(encoder.exit_symbol(&bytes) == Status::kOk && bytes > 0,
          "tile encoder exit");
    Decoder decoder;
    check(decoder.init_symbol(buffer.data(), buffer.size(), bytes) ==
              Status::kOk,
          "tile decoder init");
    av2enc::m2::cdef_adapter::TileIndexWriter tile_read;
    check(tile_read.init(fx.recon.mi_cols, fx.recon.mi_rows, plan.num_strengths,
                         4, &error),
          "tile reader init");
    bool grid_ok = true;
    for (int br = 0; br < plan.n64_rows && grid_ok; ++br) {
      for (int bc = 0; bc < plan.n64_cols && grid_ok; ++bc) {
        tile_read.clear_superblock(br * 16, bc * 16);
        int idx = -1;
        grid_ok = av2enc::m2::cdef_adapter::read_tile_index(decoder, &tile_read,
                                                    br * 16, bc * 16, &idx,
                                                    &error);
        const std::size_t at = static_cast<std::size_t>(br) *
                                   static_cast<std::size_t>(plan.n64_cols) +
                               static_cast<std::size_t>(bc);
        grid_ok = grid_ok && idx == plan.cdef_idx[at];
      }
    }
    check(grid_ok, "tile round-trip matches the plan grid");
    check(decoder.exit_symbol() == Status::kOk,
          "tile decoder sees clean trailing bits");

    // Determinism.
    Plan again;
    check(av2enc::m2::cdef_adapter::plan(fx.view, options, &again, &error),
          "repeat plan");
    check(again.cdef_idx == plan.cdef_idx && again.sse_after == plan.sse_after,
          "plan is deterministic");
  }
}

void test_filter_matches_reference() {
  Fixture fx;
  fx.build(20, 12, 21, true);
  Options options;
  options.num_strengths = 4;
  Plan plan;
  std::string error;
  check(av2enc::m2::cdef_adapter::plan(fx.view, options, &plan, &error),
        "plan for filter comparison");

  TestFrame out = av2enc::cdef::test::make_frame(fx.recon.mi_cols,
                                                 fx.recon.mi_rows);
  std::uint8_t* dst[3] = {out.dst[0].data(), out.dst[1].data(),
                          out.dst[2].data()};
  int stride[3] = {out.stride[0], out.stride[1], out.stride[2]};
  check(av2enc::m2::cdef_adapter::filter(fx.view, plan, dst, stride, &error),
        "adapter filter");

  TestFrame reference = av2enc::cdef::test::make_frame(fx.recon.mi_cols,
                                                       fx.recon.mi_rows);
  av2enc::cdef::CdefFrame cf = reference.view();
  for (int plane = 0; plane < 3; ++plane) {
    cf.planes[plane].src = fx.recon.src[plane].data();
    cf.planes[plane].dst = reference.dst[plane].data();
  }
  cf.cdef_idx = plan.cdef_idx.data();
  cf.n64_cols = plan.n64_cols;
  cf.n64_rows = plan.n64_rows;
  check(av2enc::cdef::cdef_frame(cf, plan.options) ==
            av2enc::cdef::CdefStatus::kOk,
        "reference filter");
  for (int plane = 0; plane < 3; ++plane) {
    check(filtered_equal(out, reference, plane),
          "adapter filter equals cdef_frame");
  }
}

void test_skip_and_lossless() {
  Fixture fx;
  fx.build(20, 12, 31, true);
  const CdefStrength strong{8, 4, 8, 4};
  const Plan plan = fx.manual(strong);
  std::string error;
  TestFrame out = av2enc::cdef::test::make_frame(fx.recon.mi_cols,
                                                 fx.recon.mi_rows);
  std::uint8_t* dst[3] = {out.dst[0].data(), out.dst[1].data(),
                          out.dst[2].data()};
  int stride[3] = {out.stride[0], out.stride[1], out.stride[2]};

  std::fill(fx.grids.skip.begin(), fx.grids.skip.end(),
            static_cast<std::uint8_t>(1));
  check(av2enc::m2::cdef_adapter::filter(fx.view, plan, dst, stride, &error),
        "skipped frame filters");
  for (int plane = 0; plane < 3; ++plane) {
    check(filtered_equals_source(out, plane, fx.recon), "skipped frame is copied");
  }

  // Luma lossless (skip regardless of on_skip_txfm).
  std::fill(fx.grids.skip.begin(), fx.grids.skip.end(),
            static_cast<std::uint8_t>(0));
  std::fill(fx.grids.lossless.begin(), fx.grids.lossless.end(),
            static_cast<std::uint8_t>(1));
  TestFrame out2 = av2enc::cdef::test::make_frame(fx.recon.mi_cols,
                                                  fx.recon.mi_rows);
  std::uint8_t* dst2[3] = {out2.dst[0].data(), out2.dst[1].data(),
                           out2.dst[2].data()};
  int stride2[3] = {out2.stride[0], out2.stride[1], out2.stride[2]};
  check(av2enc::m2::cdef_adapter::filter(fx.view, plan, dst2, stride2, &error),
        "lossless frame filters");
  check(filtered_equals_source(out2, 0, fx.recon), "luma lossless copies luma");

  // Chroma lossless: luma is filtered, chroma is copied.
  std::fill(fx.grids.lossless.begin(), fx.grids.lossless.end(),
            static_cast<std::uint8_t>(0));
  std::fill(fx.grids.lossless_chroma.begin(), fx.grids.lossless_chroma.end(),
            static_cast<std::uint8_t>(1));
  TestFrame out3 = av2enc::cdef::test::make_frame(fx.recon.mi_cols,
                                                  fx.recon.mi_rows);
  std::uint8_t* dst3[3] = {out3.dst[0].data(), out3.dst[1].data(),
                           out3.dst[2].data()};
  int stride3[3] = {out3.stride[0], out3.stride[1], out3.stride[2]};
  check(av2enc::m2::cdef_adapter::filter(fx.view, plan, dst3, stride3, &error),
        "chroma-lossless frame filters");
  check(!filtered_equals_source(out3, 0, fx.recon), "luma still filtered");
  check(filtered_equals_source(out3, 1, fx.recon) &&
            filtered_equals_source(out3, 2, fx.recon),
        "chroma copied when lossless");

  // on_skip_txfm ignores the skip grid.
  std::fill(fx.grids.lossless_chroma.begin(), fx.grids.lossless_chroma.end(),
            static_cast<std::uint8_t>(0));
  std::fill(fx.grids.skip.begin(), fx.grids.skip.end(),
            static_cast<std::uint8_t>(1));
  Plan forced = plan;
  forced.options.on_skip_txfm = true;
  TestFrame out4 = av2enc::cdef::test::make_frame(fx.recon.mi_cols,
                                                  fx.recon.mi_rows);
  std::uint8_t* dst4[3] = {out4.dst[0].data(), out4.dst[1].data(),
                           out4.dst[2].data()};
  int stride4[3] = {out4.stride[0], out4.stride[1], out4.stride[2]};
  check(av2enc::m2::cdef_adapter::filter(fx.view, forced, dst4, stride4, &error),
        "on_skip_txfm frame filters");
  check(!filtered_equals_source(out4, 0, fx.recon),
        "on_skip_txfm filters a skipped block");
}

void test_tile_clipping() {
  Fixture fx;
  fx.build(48, 16, 41, true);
  // Two tiles in x at 32 MI (128 luma samples).
  std::vector<std::uint16_t> cols = {0, 32};
  std::vector<std::uint16_t> rows = {0};
  fx.view.tile_col_starts = cols.data();
  fx.view.num_tile_cols = 2;
  fx.view.tile_row_starts = rows.data();
  fx.view.num_tile_rows = 1;
  const CdefStrength strong{12, 4, 12, 4};
  Plan plan = fx.manual(strong);
  plan.options.disable_loopfilters_across_tiles = false;

  TestFrame across = av2enc::cdef::test::make_frame(fx.recon.mi_cols,
                                                   fx.recon.mi_rows);
  std::uint8_t* dst_a[3] = {across.dst[0].data(), across.dst[1].data(),
                            across.dst[2].data()};
  int stride[3] = {across.stride[0], across.stride[1], across.stride[2]};
  std::string error;
  check(av2enc::m2::cdef_adapter::filter(fx.view, plan, dst_a, stride, &error),
        "frame-wide filter");

  Plan clipped = fx.manual(strong);
  clipped.options.disable_loopfilters_across_tiles = true;
  TestFrame inside = av2enc::cdef::test::make_frame(fx.recon.mi_cols,
                                                    fx.recon.mi_rows);
  std::uint8_t* dst_i[3] = {inside.dst[0].data(), inside.dst[1].data(),
                            inside.dst[2].data()};
  check(av2enc::m2::cdef_adapter::filter(fx.view, clipped, dst_i, stride, &error),
        "tile-clipped filter");

  int near_diff = 0;
  int far_diff = 0;
  const int boundary = 128;
  for (int y = 0; y < across.plane_height(0); ++y) {
    for (int x = 0; x < across.plane_width(0); ++x) {
      const std::uint8_t a = across.dst[0][av2enc::cdef::test::pix_index(
          y, across.stride[0], x)];
      const std::uint8_t c = inside.dst[0][av2enc::cdef::test::pix_index(
          y, inside.stride[0], x)];
      if (a == c) continue;
      if (x >= boundary - 8 && x < boundary + 8) {
        ++near_diff;
      } else if (x < boundary - 16 || x >= boundary + 16) {
        ++far_diff;
      }
    }
  }
  check(near_diff > 0, "tile clipping changes samples at the boundary");
  check(far_diff == 0, "tile clipping leaves distant samples untouched");
}

void test_sequence_filter_config() {
  BitWriter writer;
  std::string error;
  const std::size_t bits = av2enc::m2::cdef_adapter::write_sequence_filter_config(
      writer, /*enable_cdef=*/true, /*on_skip_always_on=*/false,
      /*on_skip_disabled=*/true, &error);
  check(error.empty() && bits > 0, "sequence filter row writes");
  const std::vector<std::uint8_t> bytes = writer.take_bytes();
  av2enc::m1::filters_stream::BitReader reader(bytes, bits);
  av2enc::m1::filters_stream::SequenceFilterConfig config;
  check(av2enc::m1::filters_stream::read_sequence_filter_config(
            reader, /*single_picture_header_flag=*/false, /*sb_64x64=*/true,
            &config, &error),
        "sequence filter row reads back");
  check(config.enable_cdef, "enable_cdef round-trips");
  check(config.cdef_on_skip_txfm_disabled && !config.cdef_on_skip_txfm_always_on,
        "CDEF_ON_SKIP_TXFM_DISABLED round-trips");
  check(!config.enable_gdf && !config.enable_restoration && !config.enable_ccso,
        "other tools stay off");
}

void test_real_frame(const std::string& path) {
  TestFrame real;
  if (!av2enc::cdef::test::load_y4m_first_frame(path, &real)) {
    std::fprintf(stderr, "skip: cannot load %s\n", path.c_str());
    return;
  }
  if ((real.mi_cols & 1) != 0 || (real.mi_rows & 1) != 0) {
    std::fprintf(stderr, "skip: odd MI grid in %s\n", path.c_str());
    return;
  }
  Fixture fx;
  fx.recon = real;
  fx.ref = av2enc::cdef::test::make_frame(real.mi_cols, real.mi_rows);
  for (int plane = 0; plane < 3; ++plane) {
    const int w = real.plane_width(plane);
    const int h = real.plane_height(plane);
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const int base = real.src[plane][av2enc::cdef::test::pix_index(
            y, real.stride[plane], x)];
        const unsigned n = static_cast<unsigned>(x * 37 + y * 91);
        int value = base + static_cast<int>(n % 7) - 3;
        value = value < 0 ? 0 : (value > 255 ? 255 : value);
        fx.ref.src[plane][av2enc::cdef::test::pix_index(
            y, fx.ref.stride[plane], x)] = static_cast<std::uint8_t>(value);
      }
    }
  }
  fx.grids.zero(real.mi_cols, real.mi_rows);
  fx.view = FrameView{};
  for (int plane = 0; plane < 3; ++plane) {
    fx.view.recon[plane].data = fx.recon.src[plane].data();
    fx.view.recon[plane].stride = fx.recon.stride[plane];
    fx.view.source[plane].data = fx.ref.src[plane].data();
    fx.view.source[plane].stride = fx.ref.stride[plane];
  }
  fx.view.width = real.mi_cols * 4;
  fx.view.height = real.mi_rows * 4;
  fx.view.mi_cols = real.mi_cols;
  fx.view.mi_rows = real.mi_rows;
  fx.grids.bind(&fx.view);

  for (int strengths : {1, 4}) {
    Options options;
    options.num_strengths = strengths;
    Plan plan;
    std::string error;
    check(av2enc::m2::cdef_adapter::plan(fx.view, options, &plan, &error),
          "real-frame plan");
    TestFrame out = av2enc::cdef::test::make_frame(real.mi_cols, real.mi_rows);
    std::uint8_t* dst[3] = {out.dst[0].data(), out.dst[1].data(),
                            out.dst[2].data()};
    int stride[3] = {out.stride[0], out.stride[1], out.stride[2]};
    check(av2enc::m2::cdef_adapter::filter(fx.view, plan, dst, stride, &error),
          "real-frame filter");
    std::int64_t sse = 0;
    for (int plane = 0; plane < 3; ++plane) {
      sse += filtered_sse(out, plane, fx.ref);
    }
    check(sse == plan.sse_after, "real-frame SSE matches the filter");
    std::printf(
        "real %dx%d: %d strength(s), blocks %dx%d, sse %lld -> %lld, "
        "header %d bits, index %d bits\n",
        fx.view.width, fx.view.height, plan.num_strengths, plan.n64_cols,
        plan.n64_rows, static_cast<long long>(plan.sse_before),
        static_cast<long long>(plan.sse_after), plan.header_bits,
        plan.index_bits);
  }
}

}  // namespace

int main(int argc, char** argv) {
  test_validation();
  test_single_strength();
  test_multi_strength_index();
  test_filter_matches_reference();
  test_skip_and_lossless();
  test_tile_clipping();
  test_sequence_filter_config();
  test_real_frame(argc > 1 ? argv[1] : "decoded.y4m");

  std::printf("m2_cdef_test: %d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}

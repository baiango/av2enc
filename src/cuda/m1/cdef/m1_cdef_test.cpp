// M1 GPU path — CDEF stage tests (host, no CUDA).
//
// Covers the pieces of the stage that the orchestrator and the M1 syntax owner
// consume:
//   * §5.18.7.10 header packing, including the exact bit string the frame
//     header must emit (the 20-bit single-strength case was verified against
//     dav2d f9dafbc via the stream gate);
//   * MI grid materialisation (skip / LosslessArray[SegmentIds]);
//   * the stage plan: geometry, index signalability, determinism, and the
//     objective matching cdef_frame() exactly;
//   * §5.20.10.1 cdef_idx symbol round-trip through the normative §8.2 coder for
//     every CdefStrengths value, including the cdef_index0 neighbour context.
//
// Build/run: ./build_and_test.sh (this directory).  Link line includes
// src/encoder/filters/cdef/cdef.cpp and src/entropy/av2/av2_entropy.cpp.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "cuda/m1/cdef/m1_cdef.h"
#include "cuda/m1/cdef/m1_cdef_cdf.h"
#include "cuda/m1/cdef/m1_cdef_index_syntax.h"
#include "encoder/filters/cdef/cdef.h"
#include "encoder/filters/cdef/cdef_test_common.h"
#include "entropy/av2/av2_entropy.h"

namespace {

using av2enc::cdef::CdefOptions;
using av2enc::cdef::CdefStatus;
using av2enc::cdef::CdefStrength;
using av2enc::cuda::m1::CdefHeaderParams;
using av2enc::cuda::m1::CdefIndexCdfs;
using av2enc::cuda::m1::CdefIndexGrid;
using av2enc::cuda::m1::CdefMiGrids;
using av2enc::cuda::m1::CdefMiInputs;
using av2enc::cuda::m1::CdefStageFrame;
using av2enc::cuda::m1::CdefStageOptions;
using av2enc::cuda::m1::CdefStagePlan;

int g_checks = 0;
int g_failures = 0;

void check(bool ok, const char* what) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::fprintf(stderr, "FAIL: %s\n", what);
  }
}

// Minimal MSB-first bit writer used to pin the header bit string independently
// of the stage's own bookkeeping.
struct BitString {
  std::string bits;
  void put(int value, int n) {
    for (int i = n - 1; i >= 0; --i) {
      bits.push_back(((value >> i) & 1) ? '1' : '0');
    }
  }
  void put_bit(bool b) { bits.push_back(b ? '1' : '0'); }
};

void test_header_bits() {
  CdefOptions options;
  options.damping = 3;
  options.num_strengths = 1;
  options.strengths[0] = CdefStrength{4, 2, 4, 2};
  CdefHeaderParams header;
  std::string error;
  check(av2enc::cuda::m1::m1_cdef_build_header(options, true, &header, &error),
        "header builds");
  check(header.frame_enable, "single picture header enables CDEF");
  check(header.damping_minus_3 == 0, "damping 3 -> minus_3 = 0");
  check(header.strengths_minus_1 == 0, "one strength -> minus_1 = 0");
  check(!header.on_skip_txfm_frame_enable, "skip flag off");
  check(header.num_strengths == 1, "one strength set");
  check(header.set[0].y_pri_strength == 4 && !header.set[0].y_pri_zero,
        "y pri 4 is coded");
  check(header.set[0].y_sec_strength == 2, "y sec 2 is coded as 2");
  check(header.set[0].uv_pri_strength == 4 && !header.set[0].uv_pri_zero,
        "uv pri 4 is coded");
  check(header.set[0].uv_sec_strength == 2, "uv sec 2 is coded as 2");
  check(header.bits == 20, "single strength costs 20 bits");

  BitString bits;
  bits.put(header.damping_minus_3, 2);
  bits.put(header.strengths_minus_1, 3);
  bits.put_bit(header.on_skip_txfm_frame_enable);
  bits.put_bit(header.set[0].y_pri_zero);
  if (!header.set[0].y_pri_zero) bits.put(header.set[0].y_pri_strength, 4);
  bits.put(header.set[0].y_sec_strength, 2);
  bits.put_bit(header.set[0].uv_pri_zero);
  if (!header.set[0].uv_pri_zero) bits.put(header.set[0].uv_pri_strength, 4);
  bits.put(header.set[0].uv_sec_strength, 2);
  check(bits.bits == "00000000100100010010",
        "single strength cdef_params bit string matches the dav2d-verified "
        "stream");
  check(static_cast<int>(bits.bits.size()) == header.bits,
        "reported bit count matches the written bits");

  // Secondary strength 4 maps to syntax 3.
  options.strengths[0] = CdefStrength{0, 4, 0, 4};
  check(av2enc::cuda::m1::m1_cdef_build_header(options, true, &header, &error),
        "header builds for zero primary");
  check(header.set[0].y_pri_zero && header.set[0].uv_pri_zero,
        "zero primary uses the 1-bit zero form");
  check(header.set[0].y_sec_strength == 3 && header.set[0].y_sec_value == 4,
        "secondary 4 is coded as syntax 3");
  check(header.bits == 2 + 3 + 1 + (1 + 2) + (1 + 2), "zero primary costs 9 bits");

  // Multi-strength header.
  options.num_strengths = 3;
  options.strengths[0] = CdefStrength{4, 2, 4, 2};
  options.strengths[1] = CdefStrength{8, 4, 8, 4};
  options.strengths[2] = CdefStrength{0, 0, 0, 0};
  check(av2enc::cuda::m1::m1_cdef_build_header(options, true, &header, &error),
        "3-strength header builds");
  check(header.strengths_minus_1 == 2, "strengths_minus_1 = 2");
  check(header.bits == 2 + 3 + 1 + (1 + 4 + 2 + 1 + 4 + 2) * 2 +
                          (1 + 2 + 1 + 2),
        "3-strength bit count");

  // Non-single-picture adds the frame_enable bit.
  check(av2enc::cuda::m1::m1_cdef_build_header(options, false, &header, &error),
        "non-single-picture header builds");
  check(header.bits == 2 + 3 + 1 + 1 + (1 + 4 + 2 + 1 + 4 + 2) * 2 +
                          (1 + 2 + 1 + 2),
        "frame_enable costs one bit outside a single picture header");

  // Invalid configurations.
  options.num_strengths = 1;
  options.strengths[0] = CdefStrength{16, 0, 0, 0};
  check(!av2enc::cuda::m1::m1_cdef_build_header(options, true, &header, &error),
        "pri 16 rejected");
  options.strengths[0] = CdefStrength{0, 3, 0, 0};
  check(!av2enc::cuda::m1::m1_cdef_build_header(options, true, &header, &error),
        "sec 3 rejected (must be written as 4)");
  options.strengths[0] = CdefStrength{0, 0, 0, 0};
  options.damping = 7;
  check(!av2enc::cuda::m1::m1_cdef_build_header(options, true, &header, &error),
        "damping 7 rejected");
}

void test_mi_grids() {
  CdefMiInputs inputs;
  inputs.mi_cols = 6;
  inputs.mi_rows = 4;
  CdefMiGrids grids;
  std::string error;
  check(av2enc::cuda::m1::m1_cdef_build_mi_grids(inputs, &grids, &error),
        "zero MI grids build");
  check(grids.skip.size() == 24 && grids.lossless.size() == 24 &&
            grids.lossless_chroma.size() == 24,
        "MI grid sizes");
  check(std::all_of(grids.skip.begin(), grids.skip.end(),
                    [](std::uint8_t v) { return v == 0; }),
        "M1 default skip grid is zero");
  check(std::all_of(grids.lossless.begin(), grids.lossless.end(),
                    [](std::uint8_t v) { return v == 0; }),
        "M1 default lossless grid is zero");

  std::vector<std::uint8_t> skip(24, 1);
  std::vector<std::uint8_t> segments(24, 0);
  segments[5] = 1;
  std::vector<std::uint8_t> lossless_array = {0, 1};
  inputs.skip = skip.data();
  inputs.segment_id = segments.data();
  inputs.chroma_segment_id = segments.data();
  inputs.lossless_array = lossless_array.data();
  inputs.lossless_array_size = 2;
  check(av2enc::cuda::m1::m1_cdef_build_mi_grids(inputs, &grids, &error),
        "segment MI grids build");
  check(grids.skip[0] == 1, "skip copied");
  check(grids.lossless[0] == 0 && grids.lossless[5] == 1,
        "lossless follows the segment id");
  check(grids.lossless_chroma[5] == 1, "chroma lossless follows the segment id");
  segments[7] = 4;
  check(!av2enc::cuda::m1::m1_cdef_build_mi_grids(inputs, &grids, &error),
        "segment outside the lossless array rejected");
}

struct StageFixture {
  av2enc::cdef::test::TestFrame recon;
  av2enc::cdef::test::TestFrame ref;
  std::vector<std::uint8_t> skip;
  std::vector<std::uint8_t> lossless;
  std::vector<std::uint8_t> lossless_chroma;
  CdefStageFrame view;

  void build(int mi_cols, int mi_rows, unsigned seed) {
    // Reference is a smooth, filter-friendly texture; the "reconstruction" is
    // the reference plus deterministic noise, so CDEF has something to remove.
    ref = av2enc::cdef::test::make_frame(mi_cols, mi_rows);
    av2enc::cdef::test::fill_pattern(ref,
                                     av2enc::cdef::test::Pattern::kRipple);
    recon = av2enc::cdef::test::make_frame(mi_cols, mi_rows);
    std::mt19937 rng(seed);
    for (int plane = 0; plane < 3; ++plane) {
      const int w = recon.plane_width(plane);
      const int h = recon.plane_height(plane);
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          const int base =
              ref.src[plane][av2enc::cdef::test::pix_index(
                  y, ref.stride[plane], x)];
          int value = base + static_cast<int>(rng() % 13) - 6;
          value = value < 0 ? 0 : (value > 255 ? 255 : value);
          recon.src[plane][av2enc::cdef::test::pix_index(
              y, recon.stride[plane], x)] = static_cast<std::uint8_t>(value);
        }
      }
    }
    skip.assign(static_cast<std::size_t>(mi_cols) *
                    static_cast<std::size_t>(mi_rows), 0);
    lossless.assign(skip.size(), 0);
    lossless_chroma.assign(skip.size(), 0);

    view = CdefStageFrame{};
    for (int plane = 0; plane < 3; ++plane) {
      view.plane[plane].src = recon.src[plane].data();
      view.plane[plane].dst = recon.dst[plane].data();
      view.plane[plane].stride = recon.stride[plane];
      view.plane[plane].height = recon.height[plane];
      view.ref[plane] = ref.src[plane].data();
      view.ref_stride[plane] = ref.stride[plane];
    }
    view.width = mi_cols * 4;
    view.height = mi_rows * 4;
    view.mi_cols = mi_cols;
    view.mi_rows = mi_rows;
    view.skip = skip.data();
    view.lossless = lossless.data();
    view.lossless_chroma = lossless_chroma.data();
  }

  std::int64_t filtered_sse(const CdefStagePlan& plan) {
    av2enc::cdef::test::reset_dst(recon);
    for (std::size_t i = 0; i < recon.idx.size(); ++i) {
      recon.idx[i] = plan.cdef_idx[i];
    }
    const av2enc::cdef::CdefFrame frame = recon.view();
    const CdefStatus status =
        av2enc::cdef::cdef_frame(frame, plan.search.options);
    if (status != CdefStatus::kOk) {
      return -1;
    }
    std::int64_t sse = 0;
    for (int plane = 0; plane < 3; ++plane) {
      const int w = recon.plane_width(plane);
      const int h = recon.plane_height(plane);
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          const int a =
              recon.dst[plane][av2enc::cdef::test::pix_index(
                  y, recon.stride[plane], x)];
          const int b = ref.src[plane][av2enc::cdef::test::pix_index(
              y, ref.stride[plane], x)];
          const int d = a - b;
          sse += static_cast<std::int64_t>(d) * d;
        }
      }
    }
    return sse;
  }
};

void test_plan() {
  StageFixture fx;
  fx.build(20, 12, 5);
  CdefStageOptions options;
  options.damping = 3;
  options.num_strengths = 4;
  CdefStagePlan plan;
  std::string error;
  check(av2enc::cuda::m1::m1_cdef_plan(fx.view, options, &plan, &error),
        "stage plan succeeds");
  check(plan.num_strengths >= 1 && plan.num_strengths <= 4,
        "plan respects num_strengths");
  check(plan.n64_cols == 2 && plan.n64_rows == 1, "64x64 grid geometry");
  check(plan.header_bits == plan.header.bits, "header bits reported");
  // Pruning invariant: every coded strength set is used by at least one block.
  for (int i = 0; i < plan.num_strengths; ++i) {
    int count = 0;
    for (std::int8_t choice : plan.search.block_choice) {
      count += choice == i ? 1 : 0;
    }
    check(count > 0, "every coded strength set is used");
  }
  if (plan.num_strengths == 1) {
    check(plan.index_bits_estimate == 0,
          "one used set => no cdef_idx syntax");
    check(plan.forced_single_strength, "one used set is forced");
  } else {
    check(plan.index_bits_estimate > 0, "index syntax has a cost estimate");
  }
  check(fx.filtered_sse(plan) == plan.sse_total,
        "plan SSE equals the filtered samples");
  check(plan.sse_total <= plan.sse_off_total, "filtering never hurts");

  CdefStagePlan again;
  check(av2enc::cuda::m1::m1_cdef_plan(fx.view, options, &again, &error),
        "stage plan is repeatable");
  check(again.cdef_idx == plan.cdef_idx && again.sse_total == plan.sse_total,
        "stage plan is deterministic");

  // Single strength from the caller: no index syntax, all indices zero.
  CdefStageOptions single = options;
  single.num_strengths = 1;
  CdefStagePlan one;
  check(av2enc::cuda::m1::m1_cdef_plan(fx.view, single, &one, &error),
        "single-strength plan succeeds");
  check(one.num_strengths == 1 && one.forced_single_strength,
        "single strength is forced");
  check(one.index_bits_estimate == 0, "no index bits with one strength");
  check(std::all_of(one.cdef_idx.begin(), one.cdef_idx.end(),
                    [](std::int8_t v) { return v == 0; }),
        "one strength forces cdef_idx = 0");
  check(fx.filtered_sse(one) == one.sse_total,
        "single-strength SSE matches the filter");

  // Every index is signallable.
  for (std::int8_t idx : plan.cdef_idx) {
    check(idx >= 0 && idx < plan.num_strengths, "cdef_idx in range");
  }

  // Invalid geometry and aliasing are rejected.
  CdefStageFrame bad = fx.view;
  bad.plane[0].dst = const_cast<std::uint8_t*>(bad.plane[0].src);
  check(!av2enc::cuda::m1::m1_cdef_plan(bad, options, &plan, &error),
        "aliasing source/destination rejected");
  bad = fx.view;
  bad.ref[1] = nullptr;
  check(!av2enc::cuda::m1::m1_cdef_plan(bad, options, &plan, &error),
        "missing reference plane rejected");
  CdefStageOptions illegal = options;
  CdefStrength bad_strength{0, 3, 0, 0};
  illegal.candidates = &bad_strength;
  illegal.candidate_count = 1;
  check(!av2enc::cuda::m1::m1_cdef_plan(fx.view, illegal, &plan, &error),
        "illegal candidate rejected");
}

// Independent transcription of the §8.3.1 cdef_index0 context pseudocode.
// Kept separate from m1_cdef_index_syntax.h on purpose: it must catch
// transcription errors there, not mirror them.
int spec_index0_context(const CdefIndexGrid& grid, int sb_width_log2,
                        int mi_col_start, int mi_row_start, int mi_row,
                        int mi_col) {
  const int cdef_size4 = 16;
  const int cdef_mask4 = ~(cdef_size4 - 1);
  int ctx = 0;
  int cnt = 0;
  const int left_col = (mi_col - cdef_size4) & cdef_mask4;
  const int left_row = mi_row & cdef_mask4;
  if (left_col >= mi_col_start) {
    ctx += grid.at(left_row, left_col) == 0 ? 1 : 0;
    cnt += 1;
  }
  const int above_col = mi_col & cdef_mask4;
  const int above_row = (mi_row - cdef_size4) & cdef_mask4;
  const int cur_sb_row = mi_row >> sb_width_log2;
  const int above_sb_row = above_row >> sb_width_log2;
  if (above_row >= mi_row_start && above_sb_row == cur_sb_row) {
    ctx += grid.at(above_row, above_col) == 0 ? 1 : 0;
    cnt += 1;
  }
  if (ctx != 0 && cnt == ctx) {
    ctx += 1;
  }
  return ctx;
}

void test_index_context_matches_spec() {
  // 4x3 64x64 blocks: enough to exercise left and above neighbours, the
  // superblock-row rule and the cnt == ctx bump.
  const int mi_cols = 4 * 16;
  const int mi_rows = 3 * 16;
  CdefIndexGrid grid;
  grid.reset(mi_cols, mi_rows);
  std::mt19937 rng(99);
  for (int br = 0; br < 3; ++br) {
    for (int bc = 0; bc < 4; ++bc) {
      grid.set(br * 16, bc * 16, static_cast<std::int8_t>(rng() % 9));
    }
  }
  int mismatches = 0;
  for (int br = 0; br < 3; ++br) {
    for (int bc = 0; bc < 4; ++bc) {
      const int got = av2enc::cuda::m1::cdef_index0_context(
          grid, /*sb_width_log2=*/4, /*mi_col_start=*/0, /*mi_row_start=*/0,
          br * 16, bc * 16);
      const int want = spec_index0_context(grid, 4, 0, 0, br * 16, bc * 16);
      if (got != want) ++mismatches;
      check(got >= 0 && got < 4, "context within CDEF_STRENGTH_INDEX0_CTX");
    }
  }
  check(mismatches == 0, "cdef_index0 context matches the §8.3.1 pseudocode");

  // Tile-scoped left neighbour: with MiColStart at the block's own column the
  // left neighbour is unavailable, so the context must differ from the
  // frame-scoped derivation whenever the left block was coded as strength 0.
  grid.reset(mi_cols, mi_rows);
  grid.set(0, 0, 0);
  const int frame_ctx = av2enc::cuda::m1::cdef_index0_context(
      grid, 4, 0, 0, 0, 16);
  const int tile_ctx = av2enc::cuda::m1::cdef_index0_context(
      grid, 4, /*mi_col_start=*/16, 0, 0, 16);
  check(frame_ctx != tile_ctx, "tile start removes the left neighbour");
}

// §5.20.10.1 round-trip through the normative §8.2 coder.
void test_index_syntax_roundtrip() {
  for (int strengths = 1; strengths <= 8; ++strengths) {
    for (int blocks : {1, 2, 12}) {
      const int n64_cols = blocks > 1 ? 3 : 1;
      const int n64_rows = blocks > 1 ? 4 : 1;
      const int mi_cols = n64_cols * 16;
      const int mi_rows = n64_rows * 16;
      std::mt19937 rng(static_cast<unsigned>(strengths * 31 + blocks));
      std::vector<int> expected(
          static_cast<std::size_t>(n64_cols) *
              static_cast<std::size_t>(n64_rows));
      for (int& v : expected) {
        v = strengths == 1
                ? 0
                : static_cast<int>(rng() % static_cast<unsigned>(strengths));
      }

      std::vector<std::uint8_t> buffer(1U << 16, 0);
      std::uint16_t index0[4][3];
      std::memcpy(index0, av2enc::cuda::m1::kCdefIndex0Cdf,
                  sizeof(index0));
      std::uint16_t minus1[8];
      std::memcpy(minus1, av2enc::cuda::m1::kCdefIndexMinus1With8Cdf,
                  sizeof(minus1));
      const std::uint16_t* source_minus1 =
          av2enc::cuda::m1::cdef_index_minus1_default(strengths);
      if (source_minus1 != nullptr) {
        std::memcpy(minus1, source_minus1,
                    sizeof(std::uint16_t) * static_cast<std::size_t>(strengths));
      }
      CdefIndexCdfs cdfs;
      cdfs.index0 = &index0[0][0];
      cdfs.minus1 = minus1;

      av2enc::av2::Encoder encoder;
      check(encoder.init_symbol(buffer.data(), buffer.size()) ==
                av2enc::av2::Status::kOk,
            "encoder init");
      CdefIndexGrid grid;
      grid.reset(mi_cols, mi_rows);
      bool encode_ok = true;
      for (int br = 0; br < n64_rows && encode_ok; ++br) {
        for (int bc = 0; bc < n64_cols && encode_ok; ++bc) {
          const int idx = expected[static_cast<std::size_t>(br) *
                                       static_cast<std::size_t>(n64_cols) +
                                   static_cast<std::size_t>(bc)];
          encode_ok = av2enc::cuda::m1::cdef_encode_block_index(
              &grid, strengths, /*sb_size4=*/4, /*tile col=*/0, /*row=*/0,
              br * 16, bc * 16, idx, cdfs,
              [&encoder](std::uint16_t* cdf, int n, int symbol) {
                return encoder.encode_symbol(cdf, n, symbol) ==
                       av2enc::av2::Status::kOk;
              });
        }
      }
      check(encode_ok, "all indices encode");
      std::size_t payload_bytes = 0;
      check(encoder.exit_symbol(&payload_bytes) == av2enc::av2::Status::kOk,
            "encoder exit");
      check(payload_bytes > 0, "payload written");

      // Fresh CDFs for the decode side.
      std::memcpy(index0, av2enc::cuda::m1::kCdefIndex0Cdf, sizeof(index0));
      std::memcpy(minus1, av2enc::cuda::m1::kCdefIndexMinus1With8Cdf,
                  sizeof(minus1));
      if (source_minus1 != nullptr) {
        std::memcpy(minus1, source_minus1,
                    sizeof(std::uint16_t) * static_cast<std::size_t>(strengths));
      }
      av2enc::av2::Decoder decoder;
      check(decoder.init_symbol(buffer.data(), payload_bytes, payload_bytes) ==
                av2enc::av2::Status::kOk,
            "decoder init");
      CdefIndexGrid decoded;
      decoded.reset(mi_cols, mi_rows);
      bool decode_ok = true;
      for (int br = 0; br < n64_rows && decode_ok; ++br) {
        for (int bc = 0; bc < n64_cols && decode_ok; ++bc) {
          int idx = -1;
          decode_ok = av2enc::cuda::m1::cdef_decode_block_index(
              &decoded, strengths, 4, 0, 0, br * 16, bc * 16, cdfs,
              [&decoder](std::uint16_t* cdf, int n) -> int {
                int symbol = -1;
                if (decoder.decode_symbol(cdf, n, &symbol) !=
                    av2enc::av2::Status::kOk) {
                  return -1;
                }
                return symbol;
              },
              &idx);
          if (decode_ok) {
            const int want = expected[static_cast<std::size_t>(br) *
                                          static_cast<std::size_t>(n64_cols) +
                                      static_cast<std::size_t>(bc)];
            check(idx == want, "decoded index matches");
          }
        }
      }
      check(decode_ok, "all indices decode");
      check(decoder.exit_symbol() == av2enc::av2::Status::kOk,
            "decoder sees clean trailing bits");
      check(decoded.origins(n64_cols, n64_rows) ==
                grid.origins(n64_cols, n64_rows),
            "decoded grid matches the encoder grid");

      // The context derivation must see the already-coded neighbours: coding
      // the same index with a different left neighbour must not change the
      // decoded value but must select a valid context.
      CdefIndexGrid probe;
      probe.reset(mi_cols, mi_rows);
      probe.set(0, 0, 0);
      const int ctx_off = av2enc::cuda::m1::cdef_index0_context(probe, 4, 0, 0,
                                                               0, 16);
      probe.set(0, 0, 1);
      const int ctx_on = av2enc::cuda::m1::cdef_index0_context(probe, 4, 0, 0,
                                                              0, 16);
      check(ctx_on != ctx_off, "left-neighbour strength selects a context");
      check(ctx_off >= 0 && ctx_off < 4 && ctx_on >= 0 && ctx_on < 4,
            "contexts stay in CDEF_STRENGTH_INDEX0_CTX");
    }
  }
}

void test_index_syntax_edge_cases() {
  std::uint16_t index0[4][3];
  std::memcpy(index0, av2enc::cuda::m1::kCdefIndex0Cdf, sizeof(index0));
  std::uint16_t minus1[8] = {0};
  CdefIndexCdfs cdfs;
  cdfs.index0 = &index0[0][0];
  cdfs.minus1 = minus1;
  CdefIndexGrid grid;
  grid.reset(32, 32);
  int calls = 0;
  auto emit = [&calls](std::uint16_t*, int, int) {
    ++calls;
    return true;
  };
  // strengths == 1 codes nothing.
  check(av2enc::cuda::m1::cdef_encode_block_index(&grid, 1, 4, 0, 0, 0, 0, 0,
                                                  cdfs, emit),
        "single strength encodes");
  check(calls == 0, "single strength writes no symbol");
  CdefIndexGrid fresh;
  fresh.reset(32, 32);
  check(av2enc::cuda::m1::cdef_encode_block_index(&fresh, 1, 4, 0, 0, 0, 0, 1,
                                                  cdfs, emit) == false,
        "single strength rejects a non-zero index");
  // A second call for the same 64x64 is a no-op.
  grid.reset(32, 32);
  check(av2enc::cuda::m1::cdef_encode_block_index(&grid, 4, 4, 0, 0, 0, 0, 2,
                                                  cdfs, emit),
        "multi strength encodes");
  const int before = calls;
  check(av2enc::cuda::m1::cdef_encode_block_index(&grid, 4, 4, 0, 0, 4, 4, 1,
                                                  cdfs, emit),
        "second call for the same 64x64 is a no-op");
  check(calls == before, "no extra symbol for the same 64x64");
  // A different block does code.
  check(av2enc::cuda::m1::cdef_encode_block_index(&grid, 4, 4, 0, 0, 0, 16, 3,
                                                  cdfs, emit),
        "neighbouring 64x64 encodes");
  check(calls == before + 2, "index0 + minus1 for 4 strengths");
}

void print_plan_summary(const char* label, const CdefStagePlan& plan) {
  std::printf(
      "%s: strengths %d, blocks %dx%d, sse %lld -> %lld, header %d bits, "
      "index ~%d bits\n",
      label, plan.num_strengths, plan.n64_cols, plan.n64_rows,
      static_cast<long long>(plan.sse_off_total),
      static_cast<long long>(plan.sse_total), plan.header_bits,
      plan.index_bits_estimate);
}

}  // namespace

int main(int argc, char** argv) {
  test_header_bits();
  test_mi_grids();
  test_plan();
  test_index_context_matches_spec();
  test_index_syntax_roundtrip();
  test_index_syntax_edge_cases();

  // Real reconstructed 640x360 content, if available.
  const std::string real = argc > 1 ? argv[1] : "decoded.y4m";
  av2enc::cdef::test::TestFrame recon;
  if (av2enc::cdef::test::load_y4m_first_frame(real, &recon)) {
    StageFixture fx;
    fx.recon = recon;
    fx.ref = av2enc::cdef::test::make_frame(recon.mi_cols, recon.mi_rows);
    // The stream gate measures the real quality delta.  Here the reference is
    // the reconstruction plus deterministic +-3 noise, so the search has a
    // non-trivial decision to make on real content while staying reproducible.
    for (int plane = 0; plane < 3; ++plane) {
      const int w = recon.plane_width(plane);
      const int h = recon.plane_height(plane);
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          const int base = recon.src[plane][av2enc::cdef::test::pix_index(
              y, recon.stride[plane], x)];
          const unsigned n = static_cast<unsigned>(x * 37 + y * 91);
          int value = base + static_cast<int>(n % 7) - 3;
          value = value < 0 ? 0 : (value > 255 ? 255 : value);
          fx.ref.src[plane][av2enc::cdef::test::pix_index(
              y, fx.ref.stride[plane], x)] = static_cast<std::uint8_t>(value);
        }
      }
    }
    fx.skip.assign(static_cast<std::size_t>(recon.mi_cols) *
                       static_cast<std::size_t>(recon.mi_rows), 0);
    fx.lossless.assign(fx.skip.size(), 0);
    fx.lossless_chroma.assign(fx.skip.size(), 0);
    fx.view = CdefStageFrame{};
    for (int plane = 0; plane < 3; ++plane) {
      fx.view.plane[plane].src = fx.recon.src[plane].data();
      fx.view.plane[plane].dst = fx.recon.dst[plane].data();
      fx.view.plane[plane].stride = fx.recon.stride[plane];
      fx.view.plane[plane].height = fx.recon.height[plane];
      fx.view.ref[plane] = fx.ref.src[plane].data();
      fx.view.ref_stride[plane] = fx.ref.stride[plane];
    }
    fx.view.width = recon.mi_cols * 4;
    fx.view.height = recon.mi_rows * 4;
    fx.view.mi_cols = recon.mi_cols;
    fx.view.mi_rows = recon.mi_rows;
    fx.view.skip = fx.skip.data();
    fx.view.lossless = fx.lossless.data();
    fx.view.lossless_chroma = fx.lossless_chroma.data();

    CdefStageOptions options;
    options.num_strengths = 4;
    CdefStagePlan plan;
    std::string error;
    check(av2enc::cuda::m1::m1_cdef_plan(fx.view, options, &plan, &error),
          "real-frame plan succeeds");
    check(fx.filtered_sse(plan) == plan.sse_total,
          "real-frame SSE matches the filter");
    print_plan_summary("real 640x360", plan);
  } else {
    std::fprintf(stderr, "skip: cannot load %s\n", real.c_str());
  }

  std::printf("m1_cdef_test: %d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}

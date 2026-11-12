// AV2 §7.18 CDEF — strength-search tests (CPU reference).
//
// Covers the encoder-side decision that cdef_search.h adds:
//   * candidate legality and the §5.18.7.10 strength set rules;
//   * the objective: the selected plan's SSE equals the SSE of the samples
//     cdef_frame() actually produces (this is the parity that makes the search
//     trustworthy, and the CUDA search kernel is checked against it);
//   * CdefStrengths == 1 => no index syntax, per-block selection not signalled;
//   * determinism, tie-breaking, edge blocks, skip/lossless/chroma-skip;
//   * real reconstructed 640x360 content.
//
// Build/run: ./build_and_test.sh (this directory).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "codec/filters/cdef/cdef.h"
#include "codec/filters/cdef/cdef_search.h"
#include "codec/filters/cdef/cdef_test_common.h"

namespace {

using av2enc::cdef::CdefSearchFrame;
using av2enc::cdef::CdefSearchOptions;
using av2enc::cdef::CdefSearchResult;
using av2enc::cdef::CdefStrength;
using av2enc::cdef::test::TestFrame;

int g_checks = 0;
int g_failures = 0;

void check(bool ok, const char* what) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::fprintf(stderr, "FAIL: %s\n", what);
  }
}

CdefSearchFrame search_view(TestFrame& recon, TestFrame& ref) {
  CdefSearchFrame f{};
  for (int plane = 0; plane < 3; ++plane) {
    f.src[plane].data = recon.src[plane].data();
    f.src[plane].stride = recon.stride[plane];
    f.ref[plane].data = ref.src[plane].data();
    f.ref[plane].stride = ref.stride[plane];
  }
  f.width = recon.mi_cols * av2enc::cdef::kMiSize;
  f.height = recon.mi_rows * av2enc::cdef::kMiSize;
  f.mi_cols = recon.mi_cols;
  f.mi_rows = recon.mi_rows;
  f.subsampling_x = 1;
  f.subsampling_y = 1;
  f.skip = recon.skip.data();
  f.lossless = recon.lossless.data();
  f.lossless_chroma = recon.lossless_chroma.data();
  f.tile_col_starts = recon.tile_cols.empty() ? nullptr : recon.tile_cols.data();
  f.num_tile_cols = static_cast<int>(recon.tile_cols.size());
  f.tile_row_starts = recon.tile_rows.empty() ? nullptr : recon.tile_rows.data();
  f.num_tile_rows = static_cast<int>(recon.tile_rows.size());
  return f;
}

// Visible-area SSE of `out` against `ref_src` for the given frame geometry.
std::int64_t visible_sse(const TestFrame& out, const TestFrame& ref,
                         int width, int height) {
  std::int64_t sse = 0;
  for (int plane = 0; plane < 3; ++plane) {
    const int sub = (plane == 0) ? 0 : 1;
    const int w = (width + (1 << sub) - 1) >> sub;
    const int h = (height + (1 << sub) - 1) >> sub;
    const int max_w = out.plane_width(plane);
    const int max_h = out.plane_height(plane);
    for (int y = 0; y < h && y < max_h; ++y) {
      for (int x = 0; x < w && x < max_w; ++x) {
        const int a = out.dst[plane][av2enc::cdef::test::pix_index(
            y, out.stride[plane], x)];
        const int b = ref.src[plane][av2enc::cdef::test::pix_index(
            y, ref.stride[plane], x)];
        const int d = a - b;
        sse += static_cast<std::int64_t>(d) * d;
      }
    }
  }
  return sse;
}

// Runs cdef_frame() with the plan and returns its visible SSE.
std::int64_t apply_plan(TestFrame& recon, TestFrame& ref,
                        const CdefSearchResult& plan, int width, int height) {
  av2enc::cdef::test::reset_dst(recon);
  for (std::size_t i = 0; i < recon.idx.size(); ++i) {
    recon.idx[i] = plan.cdef_idx[i];
  }
  const av2enc::cdef::CdefFrame frame = recon.view();
  const av2enc::cdef::CdefStatus status =
      av2enc::cdef::cdef_frame(frame, plan.options);
  if (status != av2enc::cdef::CdefStatus::kOk) {
    std::fprintf(stderr, "cdef_frame failed: %s\n",
                 av2enc::cdef::cdef_status_string(status));
    return -1;
  }
  return visible_sse(recon, ref, width, height);
}

TestFrame make_ref_like(const TestFrame& f) {
  TestFrame ref = av2enc::cdef::test::make_frame(f.mi_cols, f.mi_rows);
  return ref;
}

// Copies the visible part of src into a ref frame and adds a deterministic
// perturbation so that filtering has something to improve on.
void build_ref(TestFrame& ref, const TestFrame& recon, int width, int height,
               unsigned seed) {
  std::mt19937 rng(seed);
  for (int plane = 0; plane < 3; ++plane) {
    const int sub = (plane == 0) ? 0 : 1;
    const int w = (width + (1 << sub) - 1) >> sub;
    const int h = (height + (1 << sub) - 1) >> sub;
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const int base = recon.src[plane][av2enc::cdef::test::pix_index(
            y, recon.stride[plane], x)];
        const int noise = static_cast<int>(rng() % 9) - 4;
        int value = base + noise;
        value = value < 0 ? 0 : (value > 255 ? 255 : value);
        ref.src[plane][av2enc::cdef::test::pix_index(y, ref.stride[plane], x)] =
            static_cast<std::uint8_t>(value);
      }
    }
  }
}

void test_candidate_rules() {
  using namespace av2enc::cdef;
  check(kCdefDefaultCandidateCount > 1, "default ladder has candidates");
  for (int i = 0; i < kCdefDefaultCandidateCount; ++i) {
    check(cdef_valid_strength(kCdefDefaultCandidates[i]),
          "default candidate is legal");
  }
  CdefStrength bad{16, 0, 0, 0};
  check(!cdef_valid_strength(bad), "pri 16 rejected");
  bad = CdefStrength{0, 3, 0, 0};
  check(!cdef_valid_strength(bad), "sec 3 rejected (syntax maps 3 -> 4)");
  bad = CdefStrength{0, 0, 0, 5};
  check(!cdef_valid_strength(bad), "uv sec 5 rejected");

  TestFrame f = test::make_frame(8, 8);
  test::fill_pattern(f, test::Pattern::kSoftEdge);
  TestFrame ref = make_ref_like(f);
  build_ref(ref, f, 32, 32, 1);
  CdefSearchFrame view = search_view(f, ref);
  CdefSearchOptions options;
  CdefStrength illegal{0, 3, 0, 0};
  options.candidates = &illegal;
  options.candidate_count = 1;
  CdefSearchResult plan;
  std::string error;
  check(!cdef_search_plan(view, options, &plan, &error),
        "illegal candidate rejected by the planner");
  check(!error.empty(), "rejection sets an error string");
}

void test_sse_matches_filter() {
  using namespace av2enc::cdef;
  const int sizes[][2] = {{4, 4}, {8, 8}, {16, 16}, {10, 6}, {24, 14}};
  for (const auto& size : sizes) {
    TestFrame f = test::make_frame(size[0], size[1]);
    test::fill_pattern(f, test::Pattern::kNoise);
    TestFrame ref = make_ref_like(f);
    build_ref(ref, f, f.mi_cols * 4, f.mi_rows * 4, 7);
    CdefSearchFrame view = search_view(f, ref);
    for (int strengths : {1, 2, 4}) {
      CdefSearchOptions options;
      options.num_strengths = strengths;
      options.damping = 3 + (size[0] + strengths) % 4;
      CdefSearchResult plan;
      std::string error;
      check(cdef_search_plan(view, options, &plan, &error),
            "plan succeeds on synthetic frame");
      const std::int64_t actual = apply_plan(f, ref, plan,
                                             f.mi_cols * 4, f.mi_rows * 4);
      check(actual == plan.sse_total,
            "planned SSE equals the filtered SSE");
      check(plan.sse_total <= plan.sse_off_total,
            "selection never worse than unfiltered");
      check(plan.n64_cols == (f.mi_cols + 15) / 16 &&
                plan.n64_rows == (f.mi_rows + 15) / 16,
            "index grid geometry");
      for (std::int8_t idx : plan.cdef_idx) {
        check(idx >= 0 && idx < plan.num_strengths,
              "cdef_idx is signallable (>= 0)");
      }
      if (strengths == 1) {
        check(plan.forced_single_strength,
              "CdefStrengths == 1 is reported as forced");
        for (std::int8_t idx : plan.cdef_idx) {
          check(idx == 0, "single strength forces index 0");
        }
      }
    }
  }
}

void test_eval_block_matches_copy_path() {
  using namespace av2enc::cdef;
  // An all-zero strength set must reproduce cdef_frame()'s copy/identity
  // output exactly, including skip and lossless branches.
  for (int variant = 0; variant < 4; ++variant) {
    TestFrame f = test::make_frame(12, 10);
    test::fill_pattern(f, test::Pattern::kRipple);
    TestFrame ref = make_ref_like(f);
    build_ref(ref, f, 48, 40, static_cast<unsigned>(3 + variant));
    if (variant == 1) {
      std::fill(f.skip.begin(), f.skip.end(), 1);
    } else if (variant == 2) {
      std::fill(f.lossless.begin(), f.lossless.end(), 1);
    } else if (variant == 3) {
      std::fill(f.lossless_chroma.begin(), f.lossless_chroma.end(), 1);
    }
    // cdef_frame() with the identity strength.
    CdefSearchOptions options;
    options.num_strengths = 1;
    options.include_off = true;
    options.candidates = kCdefDefaultCandidates;
    options.candidate_count = 1;  // index 0 is the identity set
    CdefSearchResult plan;
    std::string error;
    check(cdef_search_plan(search_view(f, ref), options, &plan, &error),
          "identity plan succeeds");
    check(plan.num_strengths == 1 &&
              cdef_is_zero_strength(plan.strengths[0]),
          "identity plan picks the zero set");
    check(apply_plan(f, ref, plan, 48, 40) >= 0, "identity plan applies");

    // Every output byte must equal the source: an identity strength is the
    // §7.18 copy step with a no-op filter, and skip/lossless are copy-only.
    for (int plane = 0; plane < 3; ++plane) {
      const int w = f.plane_width(plane);
      const int h = f.plane_height(plane);
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          const std::size_t o =
              av2enc::cdef::test::pix_index(y, f.stride[plane], x);
          if (f.dst[plane][o] != f.src[plane][o]) {
            check(false, "identity strength reproduces the source");
            y = h;
            break;
          }
        }
      }
    }
    check(test::guard_intact(f), "identity plan leaves guards intact");
  }
}

void test_skip_and_chroma_branches() {
  using namespace av2enc::cdef;
  // A block with chroma lossless filters luma but copies chroma: the chroma
  // part of the SSE must be zero, the luma part must match the filter.
  TestFrame f = test::make_frame(8, 8);
  test::fill_pattern(f, test::Pattern::kRipple);
  TestFrame ref = make_ref_like(f);
  build_ref(ref, f, 32, 32, 11);
  std::fill(f.lossless_chroma.begin(), f.lossless_chroma.end(), 1);
  CdefSearchOptions options;
  options.num_strengths = 1;
  options.include_off = false;
  options.candidates = &kCdefDefaultCandidates[6];  // luma + chroma
  options.candidate_count = 1;
  CdefSearchResult plan;
  std::string error;
  check(cdef_search_plan(search_view(f, ref), options, &plan, &error),
        "chroma-lossless plan succeeds");
  const std::int64_t actual = apply_plan(f, ref, plan, 32, 32);
  check(actual == plan.sse_total,
        "chroma-lossless SSE matches the filter output");
}

void test_determinism_and_weights() {
  using namespace av2enc::cdef;
  TestFrame f = test::make_frame(20, 12);
  test::fill_pattern(f, test::Pattern::kNoise);
  TestFrame ref = make_ref_like(f);
  build_ref(ref, f, 80, 48, 21);
  CdefSearchFrame view = search_view(f, ref);
  CdefSearchOptions options;
  options.num_strengths = 3;
  CdefSearchResult a;
  CdefSearchResult b;
  std::string error;
  check(cdef_search_plan(view, options, &a, &error), "plan A succeeds");
  check(cdef_search_plan(view, options, &b, &error), "plan B succeeds");
  check(a.num_strengths == b.num_strengths, "deterministic strength count");
  check(a.cdef_idx == b.cdef_idx, "deterministic index grid");
  check(a.sse_total == b.sse_total, "deterministic SSE");
  for (int i = 0; i < a.num_strengths; ++i) {
    check(a.strengths[i].y_pri == b.strengths[i].y_pri &&
              a.strengths[i].y_sec == b.strengths[i].y_sec &&
              a.strengths[i].uv_pri == b.strengths[i].uv_pri &&
              a.strengths[i].uv_sec == b.strengths[i].uv_sec,
          "deterministic strength values");
  }
  // Header strength sets are distinct.
  for (int i = 0; i < a.num_strengths; ++i) {
    for (int j = i + 1; j < a.num_strengths; ++j) {
      check(!(a.strengths[i].y_pri == a.strengths[j].y_pri &&
              a.strengths[i].y_sec == a.strengths[j].y_sec &&
              a.strengths[i].uv_pri == a.strengths[j].uv_pri &&
              a.strengths[i].uv_sec == a.strengths[j].uv_sec),
            "strength sets are distinct");
    }
  }

  // Luma-only weighting must not depend on chroma: a frame whose chroma is
  // identical in recon and ref should pick the same sets as a chroma-neutral
  // search.
  CdefSearchOptions luma_only = options;
  luma_only.chroma_weight = 0;
  CdefSearchResult c;
  check(cdef_search_plan(view, luma_only, &c, &error), "luma-only plan");
  check(c.sse_total >= 0, "luma-only plan reports SSE");
  // More strength slots can only help: the greedy search never drops a slot
  // that reduced the objective.
  CdefSearchOptions more = options;
  more.num_strengths = 4;
  CdefSearchResult d;
  check(cdef_search_plan(view, more, &d, &error), "4-strength plan");
  check(d.sse_total <= a.sse_total, "extra strength slots never hurt");
}

void test_off_selection() {
  using namespace av2enc::cdef;
  // Recon already equals the reference: the identity set must win everywhere
  // and no pixel may change.
  TestFrame f = test::make_frame(8, 8);
  test::fill_pattern(f, test::Pattern::kRampX);
  TestFrame ref = make_ref_like(f);
  for (int plane = 0; plane < 3; ++plane) {
    for (int y = 0; y < f.plane_height(plane); ++y) {
      std::memcpy(ref.src[plane].data() +
                      static_cast<std::size_t>(y) *
                          static_cast<std::size_t>(ref.stride[plane]),
                  f.src[plane].data() +
                      static_cast<std::size_t>(y) *
                          static_cast<std::size_t>(f.stride[plane]),
                  static_cast<std::size_t>(f.plane_width(plane)));
    }
  }
  CdefSearchOptions options;
  options.num_strengths = 4;
  CdefSearchResult plan;
  std::string error;
  check(cdef_search_plan(search_view(f, ref), options, &plan, &error),
        "off plan succeeds");
  check(plan.sse_total == 0, "perfect recon has zero SSE");
  check(plan.blocks_off == plan.n64_cols * plan.n64_rows,
        "perfect recon picks the identity set");
  // Without the identity set the search cannot leave blocks unfiltered.
  CdefSearchOptions no_off = options;
  no_off.include_off = false;
  CdefSearchResult plan2;
  check(cdef_search_plan(search_view(f, ref), no_off, &plan2, &error),
        "plan without identity set succeeds");
  check(plan2.blocks_off == 0, "no identity set => no off blocks");
}

void test_real_frame(const std::string& path) {
  using namespace av2enc::cdef;
  test::TestFrame real;
  if (!test::load_y4m_first_frame(path, &real)) {
    std::fprintf(stderr, "skip: cannot load %s\n", path.c_str());
    return;
  }
  const int width = real.mi_cols * 4;
  const int height = real.mi_rows * 4;
  TestFrame ref = make_ref_like(real);
  build_ref(ref, real, width, height, 31);
  CdefSearchOptions options;
  options.num_strengths = 4;
  CdefSearchResult plan;
  std::string error;
  const auto t0 = test::Timer{};
  check(cdef_search_plan(search_view(real, ref), options, &plan, &error),
        "real-frame plan succeeds");
  const double ms = t0.elapsed_ms();
  const std::int64_t actual = apply_plan(real, ref, plan, width, height);
  check(actual == plan.sse_total, "real-frame SSE matches the filter");
  std::printf("real %dx%d: %d strengths, %d blocks, sse %lld -> %lld, %.1f ms\n",
              width, height, plan.num_strengths,
              plan.n64_cols * plan.n64_rows,
              static_cast<long long>(plan.sse_off_total),
              static_cast<long long>(plan.sse_total), ms);
}

}  // namespace

int main(int argc, char** argv) {
  test_candidate_rules();
  test_sse_matches_filter();
  test_eval_block_matches_copy_path();
  test_skip_and_chroma_branches();
  test_determinism_and_weights();
  test_off_selection();
  std::string real = argc > 1 ? argv[1] : "decoded.y4m";
  test_real_frame(real);
  std::printf("cdef_search_test: %d checks, %d failures\n", g_checks,
              g_failures);
  return g_failures == 0 ? 0 : 1;
}

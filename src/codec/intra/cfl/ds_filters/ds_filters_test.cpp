// Tests for src/encoder/cfl/ds_filters/.
//
// Covers:
//   * every normative Cfl_Filters_420 / Cfl_Filters_422 entry and index 3
//     aliasing to 0;
//   * constant-luma invariance and the fact that filters 0/1/2 actually
//     produce different predictions on real, non-uniform luma;
//   * byte equality with the existing cfl.cpp reference for synthetic edges,
//     partial blocks and real 4:2:0 blocks;
//   * deterministic encoder index selection (SSE and RD, tie to index 0);
//   * pinned prediction goldens for every filter on synthetic edges and real
//     frame blocks.
//
// Usage:
//   ds_filters_test
//   ds_filters_test --print-goldens     (emit C++ initialisers to paste into
//                                        goldens.h after an intentional change)

#include "ds_filters.h"

#include "goldens.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

namespace {

using av2enc::cfl::Alpha;
using av2enc::cfl::AlphaPair;
using av2enc::cfl::Block;
using av2enc::cfl::SearchMode;
using av2enc::cfl::SourcePlane;
using av2enc::cfl::ds::test::Golden;
using av2enc::cfl::ds::FilterCost;
using av2enc::cfl::ds::FilterChoice;
using av2enc::cfl::ds::FilterEvaluation;
using av2enc::cfl::ds::LumaScratch;
using av2enc::cfl::ds::SearchOptions;
using av2enc::cfl::ds::SequenceFilterSearch;

int g_checks = 0;
int g_failures = 0;

#define CHECK(cond)                                                       \
  do {                                                                    \
    ++g_checks;                                                           \
    if (!(cond)) {                                                        \
      ++g_failures;                                                       \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    }                                                                     \
  } while (0)

#define CHECK_EQ(a, b)                                                    \
  do {                                                                    \
    ++g_checks;                                                           \
    const auto va = (a);                                                  \
    const auto vb = (b);                                                  \
    if (!(va == vb)) {                                                    \
      ++g_failures;                                                       \
      std::fprintf(stderr, "FAIL %s:%d: %s == %s (%lld vs %lld)\n",       \
                   __FILE__, __LINE__, #a, #b,                           \
                   static_cast<long long>(va),                            \
                   static_cast<long long>(vb));                           \
    }                                                                     \
  } while (0)

std::uint64_t fnv1a(const void* data, std::size_t size,
                    std::uint64_t seed = 1469598103934665603ULL) {
  const auto* p = static_cast<const unsigned char*>(data);
  std::uint64_t h = seed;
  for (std::size_t i = 0; i < size; ++i) {
    h ^= p[i];
    h *= 1099511628211ULL;
  }
  return h;
}

std::uint64_t hash_values(const int* values, std::size_t count) {
  std::uint64_t h = 1469598103934665603ULL;
  for (std::size_t i = 0; i < count; ++i) {
    std::uint32_t v = static_cast<std::uint32_t>(values[i]);
    h = fnv1a(&v, sizeof(v), h);
  }
  return h;
}

std::vector<std::uint8_t> make_plane(int width, int height,
                                     const std::function<int(int, int)>& fn) {
  std::vector<std::uint8_t> out(static_cast<std::size_t>(width) *
                                static_cast<std::size_t>(height));
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      out[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
          static_cast<std::size_t>(x)] =
          static_cast<std::uint8_t>(std::clamp(fn(x, y), 0, 255));
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Y4M reader: first frame, 8-bit 4:2:0.
// ---------------------------------------------------------------------------

struct Frame420 {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
  int chroma_width() const { return (width + 1) / 2; }
  int chroma_height() const { return (height + 1) / 2; }
};

bool read_first_frame(const char* path, Frame420& frame) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return false;
  }
  std::string header;
  std::getline(in, header);
  if (header.rfind("YUV4MPEG2", 0) != 0) {
    return false;
  }
  int width = 0;
  int height = 0;
  std::size_t pos = 0;
  while (pos < header.size()) {
    const std::size_t next = header.find(' ', pos);
    const std::string token =
        header.substr(pos, next == std::string::npos ? std::string::npos
                                                     : next - pos);
    if (token.size() > 1 && token[0] == 'W') {
      width = std::atoi(token.c_str() + 1);
    } else if (token.size() > 1 && token[0] == 'H') {
      height = std::atoi(token.c_str() + 1);
    }
    if (next == std::string::npos) {
      break;
    }
    pos = next + 1;
  }
  if (width <= 0 || height <= 0 || (width & 1) != 0 || (height & 1) != 0) {
    return false;
  }
  std::string frame_header;
  std::getline(in, frame_header);
  if (frame_header.rfind("FRAME", 0) != 0) {
    return false;
  }
  const std::size_t luma_size = static_cast<std::size_t>(width) *
                                static_cast<std::size_t>(height);
  const std::size_t chroma_size =
      static_cast<std::size_t>((width + 1) / 2) *
      static_cast<std::size_t>((height + 1) / 2);
  frame.width = width;
  frame.height = height;
  frame.y.resize(luma_size);
  frame.u.resize(chroma_size);
  frame.v.resize(chroma_size);
  in.read(reinterpret_cast<char*>(frame.y.data()),
          static_cast<std::streamsize>(luma_size));
  in.read(reinterpret_cast<char*>(frame.u.data()),
          static_cast<std::streamsize>(chroma_size));
  in.read(reinterpret_cast<char*>(frame.v.data()),
          static_cast<std::streamsize>(chroma_size));
  return static_cast<std::size_t>(in.gcount()) > 0;
}

// ---------------------------------------------------------------------------
// Table tests
// ---------------------------------------------------------------------------

void test_tables() {
  using namespace av2enc::cfl::ds;
  // Literal spec tables.
  constexpr int k420[3][3][3] = {{{0, 0, 0}, {0, 2, 2}, {0, 2, 2}},
                                 {{0, 0, 0}, {1, 2, 1}, {1, 2, 1}},
                                 {{0, 1, 0}, {1, 4, 1}, {0, 1, 0}}};
  constexpr int k422[3][3] = {{0, 4, 4}, {2, 4, 2}, {0, 8, 0}};
  for (int f = 0; f < 3; ++f) {
    const Taps420 t = taps_420(f);
    int sum = 0;
    for (int dy = 0; dy < 3; ++dy) {
      for (int dx = 0; dx < 3; ++dx) {
        CHECK_EQ(t.w[dy][dx], k420[f][dy][dx]);
        sum += t.w[dy][dx];
      }
    }
    CHECK_EQ(sum, 8);
    const Taps422 t422 = taps_422(f);
    int sum422 = 0;
    for (int dx = 0; dx < 3; ++dx) {
      CHECK_EQ(t422.w[dx], k422[f][dx]);
      sum422 += t422.w[dx];
    }
    CHECK_EQ(sum422, 8);
  }
  // Index 3 is normalised to 0 everywhere.
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      CHECK_EQ(weight_420(3, dy, dx), weight_420(0, dy, dx));
      CHECK_EQ(weight(3, 1, 1, dy, dx), weight(0, 1, 1, dy, dx));
    }
  }
  for (int dx = -1; dx <= 1; ++dx) {
    CHECK_EQ(weight_422(3, dx), weight_422(0, dx));
  }
  CHECK_EQ(normalise_index(3), 0);
  CHECK_EQ(normalise_index(2), 2);
  CHECK(valid_index(3));
  CHECK(!valid_index(4));
  CHECK(!valid_index(-1));
  CHECK_EQ(clamp_sequence_index(3), 0);
  CHECK_EQ(clamp_sequence_index(2), 2);
  CHECK_EQ(clamp_sequence_index(9), 0);
  CHECK_EQ(kSyntaxBits, 2);
  // 4:4:4 / 4:4:0 fallback: exactly 8 * v.
  CHECK_EQ(weight(0, 0, 0, 0, 0), 8);
  CHECK_EQ(weight(2, 0, 0, 0, 0), 8);
  CHECK_EQ(weight(1, 1, 1, 0, 0), 2);
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

struct Case {
  const char* name;
  int w;
  int h;
  int x;
  int y;
  int coded_w;
  int coded_h;
  int sb_top_y;
  int have_above;
  int have_left;
  int luma_w;
  int luma_h;
  std::vector<std::uint8_t> luma;
  std::vector<std::uint8_t> chroma_u;
  std::vector<std::uint8_t> chroma_v;
  int chroma_stride;
};

Case synthetic_case(const char* name, int w, int h, int x, int y, int coded_w,
                    int coded_h, int luma_w, int luma_h, int sb_top_y,
                    int pattern) {
  Case c{};
  c.name = name;
  c.w = w;
  c.h = h;
  c.x = x;
  c.y = y;
  c.coded_w = coded_w;
  c.coded_h = coded_h;
  c.sb_top_y = sb_top_y;
  c.have_above = av2enc::cfl::cfl_avail_above(y, 1) ? 1 : 0;
  c.have_left = av2enc::cfl::cfl_avail_left(x, 1) ? 1 : 0;
  c.luma_w = luma_w;
  c.luma_h = luma_h;
  c.luma = make_plane(luma_w, luma_h, [pattern](int px, int py) {
    switch (pattern) {
      case 0:
        return 128;
      case 1:
        return (px * 3 + py * 5 + ((px ^ py) & 7)) & 255;
      case 2:
        return ((px / 2) * 7 + (py / 3) * 11 + 32) & 255;
      default:
        return (px * 13 + py * 17) & 255;
    }
  });
  // The coded grid may be smaller than the block extent; the decoder's
  // CurrFrame is padded, so allocate a plane that can hold the full block
  // while coded_w/coded_h still drive the luma-average replication.
  c.chroma_stride = std::max(coded_w, x + w);
  const int chroma_h = std::max(coded_h, y + h);
  c.chroma_u = make_plane(c.chroma_stride, chroma_h,
                          [pattern](int px, int py) {
                            return (96 + ((px * 5 + py * 3) & 63) + pattern * 7) &
                                   255;
                          });
  c.chroma_v = make_plane(c.chroma_stride, chroma_h,
                          [pattern](int px, int py) {
                            return (144 - ((px * 3 + py * 5) & 31) + pattern * 3) &
                                   255;
                          });
  return c;
}

Block make_block(const Case& c, int filter, std::vector<std::uint8_t>& chroma) {
  Block b{};
  b.luma = c.luma.data();
  b.luma_stride = c.luma_w;
  b.luma_max_x = c.luma_w - 1;
  b.luma_max_y = c.luma_h - 1;
  b.sb_top_y = c.sb_top_y;
  b.chroma = chroma.data();
  b.chroma_stride = c.chroma_stride;
  b.x = c.x;
  b.y = c.y;
  b.w = c.w;
  b.h = c.h;
  b.coded_width = c.coded_w;
  b.coded_height = c.coded_h;
  b.have_above = c.have_above;
  b.have_left = c.have_left;
  b.bit_depth = 8;
  b.ds_filter = filter;
  b.sub_x = 1;
  b.sub_y = 1;
  return b;
}

SourcePlane source_from(const std::vector<std::uint8_t>& plane, int stride,
                        int max_x, int max_y) {
  SourcePlane s{};
  s.plane = plane.data();
  s.stride = stride;
  s.max_x = max_x;
  s.max_y = max_y;
  return s;
}

struct GoldenResult {
  std::uint64_t l_crc = 0;
  int derived_alpha = 0;
  std::uint64_t pred_fixed_crc = 0;
  std::uint64_t pred_derived_crc = 0;
};

constexpr int kGoldenAlphaQ8 = 64;  // alpha = +2 eighths

GoldenResult compute_golden(const Case& c, int filter) {
  std::vector<std::uint8_t> chroma = c.chroma_u;
  Block b = make_block(c, filter, chroma);
  LumaScratch scratch;
  av2enc::cfl::ds::compute_luma(b, filter, scratch);
  GoldenResult r{};
  r.l_crc = hash_values(scratch.l, static_cast<std::size_t>(c.w) *
                                        static_cast<std::size_t>(c.h));
  r.derived_alpha = av2enc::cfl::ds::derive_implicit_alpha(b, scratch);
  av2enc::cfl::ds::apply_prediction(b, kGoldenAlphaQ8, scratch);
  r.pred_fixed_crc = fnv1a(b.chroma, chroma.size());
  std::vector<std::uint8_t> chroma2 = c.chroma_u;
  Block b2 = make_block(c, filter, chroma2);
  av2enc::cfl::ds::apply_prediction(b2, r.derived_alpha, scratch);
  r.pred_derived_crc = fnv1a(b2.chroma, chroma2.size());
  return r;
}

std::vector<Case> synthetic_cases() {
  std::vector<Case> cases;
  cases.push_back(synthetic_case("synth_uniform_8x8", 8, 8, 8, 8, 32, 32, 64,
                                 64, 0, 0));
  cases.push_back(synthetic_case("synth_no_neighbours_4x4", 4, 4, 0, 0, 32, 32,
                                 64, 64, 0, 1));
  cases.push_back(synthetic_case("synth_gradient_left_edge_4x8", 4, 8, 4, 4, 32,
                                 32, 64, 64, 0, 1));
  cases.push_back(synthetic_case("synth_sb_top_16x8", 16, 8, 32, 32, 64, 64,
                                 128, 128, 64, 2));
  cases.push_back(synthetic_case("synth_partial_right_bottom_8x8", 8, 8, 28,
                                 28, 30, 30, 64, 64, 0, 3));
  return cases;
}

std::vector<Case> real_cases() {
  std::vector<Case> cases;
  Frame420 frame;
  const char* path = AV2ENC_TESTDATA;
  if (!read_first_frame(path, frame)) {
    std::fprintf(stderr, "warning: cannot read %s; skipping real-frame cases\n",
                 path);
    return cases;
  }
  struct Shape {
    const char* name;
    int w;
    int h;
    int x;
    int y;
  };
  const Shape shapes[] = {
      {"real_64x64_at_100_80", 64, 64, 100, 80},
      {"real_32x16_at_300_150", 32, 16, 300, 150},
      {"real_16x8_at_320_180", 16, 8, 320, 180},
      {"real_8x8_at_400_200", 8, 8, 400, 200},
      {"real_4x4_at_500_250", 4, 4, 500, 250},
  };
  const int cw = frame.chroma_width();
  const int ch = frame.chroma_height();
  for (const Shape& s : shapes) {
    if (s.x + s.w > cw || s.y + s.h > ch) {
      continue;
    }
    Case c{};
    c.name = s.name;
    c.w = s.w;
    c.h = s.h;
    c.x = s.x;
    c.y = s.y;
    c.coded_w = cw;
    c.coded_h = ch;
    c.sb_top_y = ((s.y * 2) / 64) * 64;
    c.have_above = av2enc::cfl::cfl_avail_above(s.y, 1) ? 1 : 0;
    c.have_left = av2enc::cfl::cfl_avail_left(s.x, 1) ? 1 : 0;
    c.luma_w = frame.width;
    c.luma_h = frame.height;
    c.luma = frame.y;
    c.chroma_stride = cw;
    c.chroma_u = frame.u;
    c.chroma_v = frame.v;
    cases.push_back(std::move(c));
  }
  return cases;
}

// ---------------------------------------------------------------------------
// Cross-checks against the existing CPU reference
// ---------------------------------------------------------------------------

void cross_check_case(const Case& c, int filter) {
  std::vector<std::uint8_t> ours_chroma = c.chroma_u;
  std::vector<std::uint8_t> core_chroma = c.chroma_u;
  Block ours = make_block(c, filter, ours_chroma);
  Block core = make_block(c, filter, core_chroma);

  LumaScratch scratch;
  av2enc::cfl::ds::compute_luma(ours, filter, scratch);
  av2enc::cfl::Scratch core_scratch;
  av2enc::cfl::compute_luma(core, core_scratch);

  CHECK_EQ(scratch.luma_avg, core_scratch.luma_avg);
  for (int i = 0; i < c.w; ++i) {
    CHECK_EQ(scratch.above[i], core_scratch.luma_above[i]);
  }
  for (int j = 0; j < c.h; ++j) {
    CHECK_EQ(scratch.left[j], core_scratch.luma_left[j]);
  }
  for (int i = 0; i < c.w * c.h; ++i) {
    CHECK_EQ(scratch.l[i], core_scratch.luma[i]);
  }

  const int derived = av2enc::cfl::ds::derive_implicit_alpha(ours, scratch);
  CHECK_EQ(derived, av2enc::cfl::derive_implicit_alpha(core, core_scratch));

  av2enc::cfl::ds::apply_prediction(ours, derived, scratch);
  av2enc::cfl::apply_prediction(core, derived, core_scratch);
  CHECK(std::memcmp(ours_chroma.data(), core_chroma.data(),
                    ours_chroma.size()) == 0);

  av2enc::cfl::ds::apply_prediction(ours, kGoldenAlphaQ8, scratch);
  av2enc::cfl::apply_prediction(core, kGoldenAlphaQ8, core_scratch);
  CHECK(std::memcmp(ours_chroma.data(), core_chroma.data(),
                    ours_chroma.size()) == 0);

  // SSE cross-check on the U source.
  const SourcePlane source = source_from(c.chroma_u, c.chroma_stride,
                                         c.coded_w - 1, c.coded_h - 1);
  CHECK_EQ(av2enc::cfl::ds::prediction_sse(ours, derived, scratch, source),
           av2enc::cfl::prediction_sse(core, derived, core_scratch, source));
}

void test_cross_checks() {
  for (const Case& c : synthetic_cases()) {
    for (int f = 0; f < 4; ++f) {
      cross_check_case(c, f);
    }
  }
  for (const Case& c : real_cases()) {
    for (int f = 0; f < 4; ++f) {
      cross_check_case(c, f);
    }
  }
}

void test_filters_differ() {
  // Non-uniform luma: at least one pair of predictions must differ.  This is
  // the property the encoder search exploits; if all filters aliased there
  // would be no reason to signal a non-zero index.
  const Case c = synthetic_case("diff_check", 16, 16, 16, 16, 64, 64, 128, 128,
                                0, 1);
  std::uint64_t crc[3] = {};
  for (int f = 0; f < 3; ++f) {
    std::vector<std::uint8_t> chroma = c.chroma_u;
    Block b = make_block(c, f, chroma);
    LumaScratch scratch;
    av2enc::cfl::ds::compute_luma(b, f, scratch);
    av2enc::cfl::ds::apply_prediction(b, kGoldenAlphaQ8, scratch);
    crc[f] = fnv1a(b.chroma, chroma.size());
  }
  CHECK(crc[0] != crc[1] || crc[0] != crc[2] || crc[1] != crc[2]);

  // Constant luma: lumaAvg is exact and L is constant, so CfL prediction is
  // the same for every filter.
  const Case uniform = synthetic_case("uniform_alias", 16, 16, 16, 16, 64, 64,
                                      128, 128, 0, 0);
  std::uint64_t u_crc[3] = {};
  for (int f = 0; f < 3; ++f) {
    std::vector<std::uint8_t> chroma = uniform.chroma_u;
    Block b = make_block(uniform, f, chroma);
    LumaScratch scratch;
    av2enc::cfl::ds::compute_luma(b, f, scratch);
    CHECK_EQ(scratch.luma_avg, 128 * 8);
    for (int i = 0; i < uniform.w * uniform.h; ++i) {
      CHECK_EQ(scratch.l[i], 128 * 8);
    }
    av2enc::cfl::ds::apply_prediction(b, kGoldenAlphaQ8, scratch);
    u_crc[f] = fnv1a(b.chroma, chroma.size());
  }
  CHECK_EQ(u_crc[0], u_crc[1]);
  CHECK_EQ(u_crc[0], u_crc[2]);
}

// ---------------------------------------------------------------------------
// Encoder selection
// ---------------------------------------------------------------------------

void test_selection() {
  const Case c = synthetic_case("select", 16, 16, 16, 16, 64, 64, 128, 128, 0,
                                1);
  std::vector<std::uint8_t> chroma_u = c.chroma_u;
  std::vector<std::uint8_t> chroma_v = c.chroma_v;
  Block u = make_block(c, 0, chroma_u);
  Block v = make_block(c, 0, chroma_v);
  v.chroma = chroma_v.data();
  const SourcePlane su = source_from(c.chroma_u, c.chroma_stride, c.coded_w - 1,
                                     c.coded_h - 1);
  const SourcePlane sv = source_from(c.chroma_v, c.chroma_stride, c.coded_w - 1,
                                     c.coded_h - 1);

  av2enc::cfl::Scratch scratch;
  const SearchOptions options{};
  const std::optional<FilterChoice> choice =
      av2enc::cfl::ds::search_filter(u, v, su, sv, options,
                                     SearchMode::kExhaustiveSse, scratch);
  CHECK(choice.has_value());
  if (!choice.has_value()) {
    return;
  }
  // Manual minimum over the three explicit evaluations, ties to the lowest
  // index.  This is the deterministic contract of search_filter.
  int expected = 0;
  std::int64_t best = 0;
  bool have = false;
  for (int f = 0; f < 3; ++f) {
    av2enc::cfl::Scratch local;
    const auto eval = av2enc::cfl::ds::evaluate_filter(
        u, v, su, sv, f, SearchMode::kExhaustiveSse, local);
    CHECK(eval.has_value());
    if (!eval.has_value()) {
      continue;
    }
    if (!have || eval->total_sse() < best) {
      have = true;
      best = eval->total_sse();
      expected = f;
    }
  }
  CHECK_EQ(choice->index, expected);
  CHECK_EQ(choice->costs[0].total_sse() >= 0, true);

  // Repeat: same answer, same costs.
  av2enc::cfl::Scratch scratch2;
  const auto again = av2enc::cfl::ds::search_filter(
      u, v, su, sv, options, SearchMode::kExhaustiveSse, scratch2);
  CHECK(again.has_value());
  CHECK_EQ(again->index, choice->index);
  for (int f = 0; f < 3; ++f) {
    CHECK_EQ(again->costs[static_cast<std::size_t>(f)].total_sse(),
             choice->costs[static_cast<std::size_t>(f)].total_sse());
  }

  // RD mode with a large rate penalty on non-zero indices must choose 0 even
  // if 1 or 2 has lower SSE.
  SearchOptions rd = options;
  rd.rate_distortion = true;
  rd.lambda = 1.0;
  rd.rate_bits[0] = 0.0;
  rd.rate_bits[1] = 1.0e9;
  rd.rate_bits[2] = 1.0e9;
  av2enc::cfl::Scratch scratch3;
  const auto rd_choice = av2enc::cfl::ds::search_filter(
      u, v, su, sv, rd, SearchMode::kExhaustiveSse, scratch3);
  CHECK(rd_choice.has_value());
  CHECK_EQ(rd_choice->index, 0);

  // Sequence accumulator over one block must agree with the per-block search.
  SequenceFilterSearch seq;
  av2enc::cfl::Scratch scratch4;
  CHECK(seq.add_block(u, v, su, sv, SearchMode::kExhaustiveSse, scratch4));
  CHECK_EQ(seq.blocks(), 1);
  CHECK_EQ(seq.choose(SearchOptions{}), choice->index);
  const FilterCost cost1 = seq.cost(1, SearchOptions{});
  CHECK_EQ(cost1.total_sse(), choice->costs[1].total_sse());

  // Uniform block: every filter ties, deterministic tie-break picks 0.
  const Case uniform = synthetic_case("uniform_select", 8, 8, 8, 8, 64, 64, 128,
                                      128, 0, 0);
  std::vector<std::uint8_t> cu = uniform.chroma_u;
  std::vector<std::uint8_t> cv = uniform.chroma_v;
  Block uu = make_block(uniform, 0, cu);
  Block vv = make_block(uniform, 0, cv);
  const SourcePlane suu = source_from(uniform.chroma_u, uniform.chroma_stride,
                                      uniform.coded_w - 1, uniform.coded_h - 1);
  const SourcePlane svv = source_from(uniform.chroma_v, uniform.chroma_stride,
                                      uniform.coded_w - 1, uniform.coded_h - 1);
  av2enc::cfl::Scratch scratch5;
  const auto tie = av2enc::cfl::ds::search_filter(
      uu, vv, suu, svv, SearchOptions{}, SearchMode::kExhaustiveSse, scratch5);
  CHECK(tie.has_value());
  CHECK_EQ(tie->index, 0);
}

// ---------------------------------------------------------------------------
// Goldens
// ---------------------------------------------------------------------------

void print_golden(const char* group, const char* name, int filter,
                  const GoldenResult& g) {
  std::printf(
      "  {\"%s\", \"%s\", %d, 0x%016llxULL, %d, 0x%016llxULL, "
      "0x%016llxULL},\n",
      group, name, filter, static_cast<unsigned long long>(g.l_crc),
      g.derived_alpha, static_cast<unsigned long long>(g.pred_fixed_crc),
      static_cast<unsigned long long>(g.pred_derived_crc));
}

void print_goldens() {
  std::printf("// Generated by ds_filters_test --print-goldens\n");
  std::printf("inline constexpr Golden kSyntheticGoldens[] = {\n");
  for (const Case& c : synthetic_cases()) {
    for (int f = 0; f < 4; ++f) {
      print_golden("synthetic", c.name, f, compute_golden(c, f));
    }
  }
  std::printf("};\ninline constexpr Golden kRealGoldens[] = {\n");
  for (const Case& c : real_cases()) {
    for (int f = 0; f < 4; ++f) {
      print_golden("real", c.name, f, compute_golden(c, f));
    }
  }
  std::printf("};\n");
}

void check_goldens(const char* group, const std::vector<Case>& cases,
                   const Golden* goldens, std::size_t count) {
  std::size_t seen = 0;
  for (const Case& c : cases) {
    for (int f = 0; f < 4; ++f) {
      const Golden* found = nullptr;
      for (std::size_t i = 0; i < count; ++i) {
        if (std::strcmp(goldens[i].group, group) == 0 &&
            std::strcmp(goldens[i].name, c.name) == 0 &&
            goldens[i].filter == f) {
          found = &goldens[i];
          break;
        }
      }
      CHECK(found != nullptr);
      if (found == nullptr) {
        continue;
      }
      ++seen;
      const GoldenResult got = compute_golden(c, f);
      CHECK_EQ(got.l_crc, found->l_crc);
      CHECK_EQ(got.derived_alpha, found->derived_alpha);
      CHECK_EQ(got.pred_fixed_crc, found->pred_fixed_crc);
      CHECK_EQ(got.pred_derived_crc, found->pred_derived_crc);
    }
  }
  CHECK_EQ(seen, count);
}

void test_goldens() {
  const std::vector<Case> synthetic = synthetic_cases();
  check_goldens("synthetic", synthetic, av2enc::cfl::ds::test::kSyntheticGoldens,
                sizeof(av2enc::cfl::ds::test::kSyntheticGoldens) /
                    sizeof(av2enc::cfl::ds::test::kSyntheticGoldens[0]));
  const std::vector<Case> real = real_cases();
  if (!real.empty()) {
    check_goldens("real", real, av2enc::cfl::ds::test::kRealGoldens,
                  sizeof(av2enc::cfl::ds::test::kRealGoldens) /
                      sizeof(av2enc::cfl::ds::test::kRealGoldens[0]));
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::strcmp(argv[1], "--print-goldens") == 0) {
    print_goldens();
    return 0;
  }
  test_tables();
  test_cross_checks();
  test_filters_differ();
  test_selection();
  test_goldens();
  std::printf("ds_filters_test: %d checks, %d failures\n", g_checks,
              g_failures);
  return g_failures == 0 ? 0 : 1;
}

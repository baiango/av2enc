// Tests for the morphological prediction syntax and §7.13.3.26 process.
//
//   morph_test                  run all checks
//   morph_test --emit-goldens   print the pinned golden initializer
//   morph_test --out-prefix P   also write the dav2d gate stream variants

#include "encoder/inter/morph/morph_common.h"
#include "encoder/inter/morph/morph_stream.h"
#include "encoder/inter/morph/morph_syntax.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "encoder/partition/partition.h"
#include "morph_test_goldens.h"

namespace av2 = av2enc::av2;
using namespace av2enc::inter::morph;

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  }
}

std::uint32_t fnv1a(const std::uint8_t* data, std::size_t n) {
  std::uint32_t h = 2166136261u;
  for (std::size_t i = 0; i < n; ++i) {
    h ^= data[i];
    h *= 16777619u;
  }
  return h;
}

struct FakeFit {
  int alpha = 256;
  int beta = -128;
  int count = 0;
};

// Independent transcription of §7.13.3.25 with morphPred=1, using 64-bit
// accumulation and the pinned Div_Lut formula. This is the oracle the module
// output is compared against in every case.
FakeFit literal_fit(const std::uint8_t* plane, int stride, int mi_col,
                    int mi_row, int mi_cols, int mi_rows, int x, int y, int w,
                    int h, int mv_row, int mv_col, bool avail_u, bool avail_l,
                    int* ref_x_out, int* ref_y_out) {
  FakeFit fit{};
  const int plane_w = mi_cols * 4;
  const int plane_h = mi_rows * 4;
  const int dy = to_fullmv(mv_row);
  const int dx = to_fullmv(mv_col);
  const int ref_y = mi_row * 4 + dy;
  const int ref_x = mi_col * 4 + dx;
  *ref_x_out = ref_x;
  *ref_y_out = ref_y;
  int bw = plane_w - x;
  if (w < bw) bw = w;
  int bh = plane_h - y;
  if (h < bh) bh = h;
  const int bw2 = bw < 16 ? bw : 16;
  const int bh2 = bh < 16 ? bh : 16;
  const int width = bw2 == 12 ? 8 : bw2;
  const int height = bh2 == 12 ? 8 : bh2;
  int num_up = 0;
  int num_left = 0;
  if (avail_u && avail_l) {
    if (width == 16 && height == 16) {
      num_up = 16;
      num_left = 16;
    } else if (width > 4 && height > 4) {
      num_up = 8;
      num_left = 8;
    } else if (width < 16 && height < 16) {
      num_up = 4;
      num_left = 4;
    } else if (width == 16) {
      num_up = 16;
    } else {
      num_left = 16;
    }
  } else if (avail_u) {
    num_up = width;
  } else if (avail_l) {
    num_left = height;
  }
  long long sum_x = 0;
  long long sum_y = 0;
  long long sum_xx = 0;
  long long sum_xy = 0;
  int count = 0;
  if (num_up > 0) {
    const int step = width / num_up;
    for (int i = step >> 1; i < width; i += step) {
      const int recon = plane[(y - 1) * stride + x + i];
      const int ref = plane[(ref_y - 1) * stride + ref_x + i];
      sum_x += ref;
      sum_y += recon;
      sum_xy += static_cast<long long>(ref) * recon;
      sum_xx += static_cast<long long>(ref) * ref;
    }
    count += num_up;
  }
  if (num_left > 0) {
    const int step = height / num_left;
    for (int i = step >> 1; i < height; i += step) {
      const int recon = plane[(y + i) * stride + x - 1];
      const int ref = plane[(ref_y + i) * stride + ref_x - 1];
      sum_x += ref;
      sum_y += recon;
      sum_xy += static_cast<long long>(ref) * recon;
      sum_xx += static_cast<long long>(ref) * ref;
    }
    count += num_left;
  }
  if (count > 0) {
    const int nor = static_cast<int>(sum_xy - sum_x * sum_y / count);
    const int der = static_cast<int>(sum_xx - sum_x * sum_x / count);
    if (der != 0 && nor != 0) {
      fit.alpha = resolve_division(nor, der, 8);
      if (fit.alpha == 0) fit.alpha = 256;
    } else {
      fit.alpha = 256;
    }
    fit.beta = static_cast<int>(((sum_y << 8) - sum_x * fit.alpha) / count);
  }
  fit.count = count;
  return fit;
}

std::vector<std::uint8_t> make_plane(int w, int h) {
  std::vector<std::uint8_t> p(static_cast<std::size_t>(w) * h);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int v = 37 + 7 * x + 13 * y + 3 * (x / 4) + 5 * (y / 4) + (x * y) / 17;
      p[static_cast<std::size_t>(y) * w + x] =
          static_cast<std::uint8_t>(v & 255);
    }
  }
  return p;
}

struct SizeCase {
  int bsize;
  int w;
  int h;
};

// Every IBC-legal block size: width/height <= 64 and MiSize != 64x64.
constexpr std::array<SizeCase, 22> kIbcSizes{{
    {0, 4, 4},    {1, 4, 8},    {2, 8, 4},    {3, 8, 8},    {4, 8, 16},
    {5, 16, 8},   {6, 16, 16},  {7, 16, 32},  {8, 32, 16},  {9, 32, 32},
    {10, 32, 64}, {11, 64, 32}, {19, 4, 16},  {20, 16, 4},  {21, 8, 32},
    {22, 32, 8},  {23, 16, 64}, {24, 64, 16}, {25, 4, 32},  {26, 32, 4},
    {27, 8, 64},  {28, 64, 8},
}};

void test_math() {
  check(to_fullmv(0) == 0, "to_fullmv(0)");
  check(to_fullmv(8) == 1, "to_fullmv(8)");
  check(to_fullmv(-8) == -1, "to_fullmv(-8)");
  check(to_fullmv(1) == 0, "to_fullmv(1) rounds down");
  check(to_fullmv(-1) == 0, "to_fullmv(-1)");
  check(to_fullmv(4) == 1, "to_fullmv(4)");
  check(to_fullmv(-4) == -1, "to_fullmv(-4)");
  check(div_lut_entry(0) == 512, "Div_Lut[0]");
  check(div_lut_entry(128) == 256, "Div_Lut[128]");
  check(resolve_division(1, 1, 8) == 256, "resolve_division unit");
  check(resolve_division(-1, 1, 8) == -256, "resolve_division negative");
}

void test_template_plan() {
  // Every legal size, every availability combination, against the literal
  // plan above.
  for (const SizeCase& s : kIbcSizes) {
    for (int mask = 0; mask < 4; ++mask) {
      const bool au = (mask & 1) != 0;
      const bool al = (mask & 2) != 0;
      const MorphTemplatePlan plan = morph_template_plan(s.w, s.h, au, al);
      int bw = s.w;
      int bh = s.h;
      const int bw2 = bw < 16 ? bw : 16;
      const int bh2 = bh < 16 ? bh : 16;
      const int width = bw2 == 12 ? 8 : bw2;
      const int height = bh2 == 12 ? 8 : bh2;
      int num_up = 0;
      int num_left = 0;
      if (au && al) {
        if (width == 16 && height == 16) {
          num_up = 16;
          num_left = 16;
        } else if (width > 4 && height > 4) {
          num_up = 8;
          num_left = 8;
        } else if (width < 16 && height < 16) {
          num_up = 4;
          num_left = 4;
        } else if (width == 16) {
          num_up = 16;
        } else {
          num_left = 16;
        }
      } else if (au) {
        num_up = width;
      } else if (al) {
        num_left = height;
      }
      check(plan.width == width && plan.height == height &&
                plan.num_up == num_up && plan.num_left == num_left,
            "template plan size " + std::to_string(s.w) + "x" +
                std::to_string(s.h) + " avail " + std::to_string(mask));
      if (num_up > 0) {
        check(plan.up_step == width / num_up, "up step");
      }
      if (num_left > 0) {
        check(plan.left_step == height / num_left, "left step");
      }
    }
  }
}

void run_prediction_case(const SizeCase& s, int mask, int* alpha, int* beta,
                         int* count, std::uint32_t* hash,
                         std::vector<std::uint8_t>* out_plane) {
  const int pw = 256;
  const int ph = 256;
  std::vector<std::uint8_t> plane = make_plane(pw, ph);
  const int x = 96;
  const int y = 96;
  const int ref_x = 32;
  const int ref_y = 32;
  const int mv_row = (ref_y - y) * 8;
  const int mv_col = (ref_x - x) * 8;
  // IBC copy: prediction is the source block.
  for (int i = 0; i < s.h; ++i) {
    for (int j = 0; j < s.w; ++j) {
      plane[static_cast<std::size_t>(y + i) * pw + x + j] =
          plane[static_cast<std::size_t>(ref_y + i) * pw + ref_x + j];
    }
  }
  const bool au = (mask & 1) != 0;
  const bool al = (mask & 2) != 0;
  MorphContext ctx;
  ctx.mi_col = x / 4;
  ctx.mi_row = y / 4;
  ctx.mi_cols = pw / 4;
  ctx.mi_rows = ph / 4;
  ctx.avail_u = au;
  ctx.avail_l = al;
  MorphPlane mp{plane.data(), pw};
  MorphRect rect{x, y, s.w, s.h};
  MorphFit fit{};
  check(morph_predict(ctx, mp, rect, mv_row, mv_col, &fit),
        "morph_predict succeeds " + std::to_string(s.w) + "x" +
            std::to_string(s.h));
  int rx = 0;
  int ry = 0;
  const FakeFit want = literal_fit(plane.data(), pw, ctx.mi_col, ctx.mi_row,
                                   ctx.mi_cols, ctx.mi_rows, x, y, s.w, s.h,
                                   mv_row, mv_col, au, al, &rx, &ry);
  // literal_fit above ran after morph_predict changed the block, but it only
  // reads the template (above/left) and the reference block, both outside the
  // adapted block, so the fit is unchanged.
  check(fit.alpha == want.alpha,
        "alpha golden " + std::to_string(s.w) + "x" + std::to_string(s.h));
  check(fit.beta == want.beta, "beta golden");
  check(fit.count == want.count, "count golden");
  check(fit.ref_x == rx && fit.ref_y == ry, "reference origin");
  for (int i = 0; i < s.h; ++i) {
    for (int j = 0; j < s.w; ++j) {
      const int orig =
          plane[static_cast<std::size_t>(ref_y + i) * pw + ref_x + j];
      const int expected = clip1((orig * want.alpha + want.beta) >> 8, 8);
      check(plane[static_cast<std::size_t>(y + i) * pw + x + j] == expected,
            "sample golden " + std::to_string(s.w) + "x" +
                std::to_string(s.h));
    }
  }
  *alpha = fit.alpha;
  *beta = fit.beta;
  *count = fit.count;
  *hash = fnv1a(plane.data() + static_cast<std::size_t>(y) * pw + x,
                static_cast<std::size_t>(s.w) * s.h);
  *out_plane = std::move(plane);
}

void emit_goldens() {
  std::printf("// Generated by: morph_test --emit-goldens\n");
  std::printf("inline constexpr GoldenRow kGoldens[] = {\n");
  for (const SizeCase& s : kIbcSizes) {
    for (int mask = 0; mask < 4; ++mask) {
      int alpha = 0;
      int beta = 0;
      int count = 0;
      std::uint32_t hash = 0;
      std::vector<std::uint8_t> plane;
      run_prediction_case(s, mask, &alpha, &beta, &count, &hash, &plane);
      std::printf("    {%d, %d, %d, %d, %d, 0x%08xu},\n", s.bsize, mask,
                  alpha, beta, count, hash);
    }
  }
  std::printf("};\n");
}

void test_prediction_goldens() {
  check(sizeof(kGoldens) / sizeof(kGoldens[0]) == kIbcSizes.size() * 4,
        "golden table size");
  std::size_t index = 0;
  for (const SizeCase& s : kIbcSizes) {
    for (int mask = 0; mask < 4; ++mask, ++index) {
      int alpha = 0;
      int beta = 0;
      int count = 0;
      std::uint32_t hash = 0;
      std::vector<std::uint8_t> plane;
      run_prediction_case(s, mask, &alpha, &beta, &count, &hash, &plane);
      const GoldenRow& g = kGoldens[index];
      check(g.bsize == s.bsize && g.avail == mask, "golden key");
      check(alpha == g.alpha, "golden alpha pin " + std::to_string(index));
      check(beta == g.beta, "golden beta pin " + std::to_string(index));
      check(count == g.count, "golden count pin " + std::to_string(index));
      check(hash == g.hash, "golden hash pin " + std::to_string(index));
    }
  }
}

void test_bounds() {
  const int pw = 128;
  const int ph = 128;
  std::vector<std::uint8_t> plane = make_plane(pw, ph);
  MorphPlane mp{plane.data(), pw};
  MorphContext ctx;
  ctx.mi_col = 24;
  ctx.mi_row = 24;
  ctx.mi_cols = pw / 4;
  ctx.mi_rows = ph / 4;
  ctx.avail_u = true;
  ctx.avail_l = true;
  MorphRect rect{96, 96, 16, 16};
  MorphFit fit{};
  // ref at x=0 fails the spec's refX >= 1.
  check(!morph_predict(ctx, mp, rect, 0, (0 - 96) * 8, &fit),
        "refX < 1 rejected");
  // ref at y=0 fails.
  check(!morph_predict(ctx, mp, rect, (0 - 96) * 8, 0, &fit),
        "refY < 1 rejected");
  // ref block crossing the plane fails.
  check(!morph_predict(ctx, mp, rect, 0,
                       (pw - 8 - 96) * 8, &fit),
        "refX+bw > planeWidth rejected");
  // avail_u with y == 0 is inconsistent.
  MorphContext top = ctx;
  top.mi_row = 0;
  check(!morph_predict(top, mp, MorphRect{96, 0, 16, 16}, 0, 0, &fit),
        "avail_u at y=0 rejected");
  // A valid zero-motion reference inside the block is legal for the predictor
  // (the syntax validator is responsible for IBC overlap rules).
  check(morph_predict(ctx, mp, rect, 0, 0, &fit), "zero BV geometry accepted");
  check(fit.count > 0, "zero BV template present");
  // Zero template path: no above/left -> alpha 256, beta -128.
  MorphContext none = ctx;
  none.avail_u = false;
  none.avail_l = false;
  std::vector<std::uint8_t> p2 = make_plane(pw, ph);
  MorphPlane mp2{p2.data(), pw};
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      p2[static_cast<std::size_t>(96 + i) * pw + 96 + j] = 100;
    }
  }
  check(morph_predict(none, mp2, rect, 0, 0, &fit), "zero template accepted");
  check(fit.alpha == 256 && fit.beta == -128 && fit.count == 0,
        "zero template defaults");
  check(p2[static_cast<std::size_t>(96) * pw + 96] == 99,
        "zero template subtracts one");
}

void test_syntax() {
  MorphCdfs cdfs{};
  init_morph_cdfs(cdfs);
  check(cdfs.morph_pred[0][0] == 31715 && cdfs.morph_pred[0][1] == 55 &&
            cdfs.morph_pred[0][2] == 0,
        "morph CDF ctx0");
  check(cdfs.morph_pred[1][0] == 19667 && cdfs.morph_pred[1][1] == 1,
        "morph CDF ctx1");
  check(cdfs.morph_pred[2][0] == 10555 && cdfs.morph_pred[2][1] == 91,
        "morph CDF ctx2");

  // NPos matrix: block at SB row start has no above (different SB row), a
  // non-start block sees above, and a wide-left block can fill both NPos slots.
  MorphNeighbours n = morph_neighbours(16, 16, 16, 8, 128, 64, 64);
  check(n.count == 2, "NPos count at SB row start");
  check(n.row[0] == 23 && n.col[0] == 15, "NPos bottom-left first");
  check(n.row[1] == 16 && n.col[1] == 15, "NPos left second");
  n = morph_neighbours(20, 32, 4, 4, 128, 64, 64);
  check(n.count == 2 && n.row[0] == 23 && n.col[0] == 31,
        "NPos bottom-left");
  check(n.row[1] == 19 && n.col[1] == 35, "NPos above");
  n = morph_neighbours(0, 0, 16, 8, 128, 64, 64);
  check(n.count == 0, "NPos empty at top-left");

  // Exhaustive NPos comparison against a literal §5.20.4.1 add_neighbor()
  // implementation for every IBC-legal size at representative positions.
  const auto literal_npos = [](int mi_row, int mi_col, int bw4, int bh4,
                               int mi_cols, int mi_rows, int* rows,
                               int* cols) {
    int count = 0;
    const auto inside = [&](int r, int c) {
      return r >= 0 && c >= 0 && r < mi_rows && c < mi_cols;
    };
    const auto add = [&](int r, int c) {
      if (count >= 2 || !inside(r, c)) return;
      if ((mi_row >> 4) != (r >> 4)) return;
      rows[count] = r;
      cols[count] = c;
      ++count;
    };
    add(mi_row + bh4 - 1, mi_col - 1);
    add(mi_row - 1, mi_col + bw4 - 1);
    add(mi_row, mi_col - 1);
    add(mi_row - 1, mi_col);
    return count;
  };
  for (const SizeCase& s : kIbcSizes) {
    const int bw4 = s.w / 4;
    const int bh4 = s.h / 4;
    for (int mr : {0, 1, 7, 8, 15, 16, 24}) {
      for (int mc : {0, 1, 15, 16, 32}) {
        int er[2] = {0, 0};
        int ec[2] = {0, 0};
        const int ecnt =
            literal_npos(mr, mc, bw4, bh4, 128, 64, er, ec);
        const MorphNeighbours got =
            morph_neighbours(mr, mc, bw4, bh4, 128, 64, 64);
        check(got.count == ecnt, "NPos count exhaustive");
        for (int i = 0; i < ecnt; ++i) {
          check(got.row[i] == er[i] && got.col[i] == ec[i],
                "NPos position exhaustive");
        }
      }
    }
  }

  std::vector<std::uint8_t> grid(static_cast<std::size_t>(128) * 64, 0);
  grid[23 * 128 + 15] = 1;
  grid[16 * 128 + 15] = 1;
  check(morph_symbol_context(grid.data(), 128, 64, 16, 16, 16, 8, 64) == 2,
        "morph context 2");
  grid[16 * 128 + 15] = 0;
  check(morph_symbol_context(grid.data(), 128, 64, 16, 16, 16, 8, 64) == 1,
        "morph context 1");
  grid[23 * 128 + 15] = 0;
  check(morph_symbol_context(grid.data(), 128, 64, 16, 16, 16, 8, 64) == 0,
        "morph context 0");

  // Writer/reader round-trip over every ctx and value, adaptive and frozen.
  for (int freeze = 0; freeze < 2; ++freeze) {
    MorphCdfs enc_cdfs{};
    MorphCdfs dec_cdfs{};
    init_morph_cdfs(enc_cdfs);
    init_morph_cdfs(dec_cdfs);
    std::vector<std::uint8_t> payload(1U << 16);
    av2::Encoder enc;
    enc.init_symbol(payload.data(), payload.size());
    enc.disable_cdf_update = static_cast<std::uint8_t>(freeze);
    std::vector<std::pair<int, bool>> expected;
    std::string error;
    for (int k = 0; k < 60; ++k) {
      const int ctx = k % 3;
      const bool value = ((k / 3) & 1) != 0;
      check(write_morph_pred(enc, enc_cdfs, ctx, value, error),
            "write_morph_pred: " + error);
      expected.emplace_back(ctx, value);
    }
    std::size_t sz = 0;
    check(enc.exit_symbol(&sz) == av2::Status::kOk, "morph exit_symbol");
    av2::Decoder dec;
    check(dec.init_symbol(payload.data(), payload.size(), sz) ==
              av2::Status::kOk,
          "morph decoder init");
    dec.disable_cdf_update = static_cast<std::uint8_t>(freeze);
    for (const auto& e : expected) {
      bool value = false;
      check(read_morph_pred(dec, dec_cdfs, e.first, value, error),
            "read_morph_pred: " + error);
      check(value == e.second, "morph round-trip value");
    }
    check(dec.exit_symbol() == av2::Status::kOk, "morph decoder exit");
  }
  MorphCdfs c{};
  init_morph_cdfs(c);
  std::vector<std::uint8_t> payload(64);
  av2::Encoder enc;
  enc.init_symbol(payload.data(), payload.size());
  std::string error;
  check(!write_morph_pred(enc, c, 3, true, error), "ctx 3 rejected");
  check(!write_morph_pred(enc, c, -1, true, error), "ctx -1 rejected");
}

void test_gate() {
  MorphPredGate g;
  g.enable_bawp = true;
  g.allow_screen_content_tools = true;
  g.frame_is_intra = true;
  g.use_intrabc = true;
  check(morph_pred_present(g), "morph present when all three bits set");
  check(morph_pred_allowed(g, true), "morph allowed");
  std::string why;
  check(!morph_pred_allowed(g, false, &why), "offset invalid rejected");
  check(why.find("is_offset_mv_valid") != std::string::npos,
        "offset rejection reason");
  g.enable_bawp = false;
  check(!morph_pred_present(g), "enable_bawp=0 suppresses syntax");
  check(!morph_pred_allowed(g, true), "enable_bawp=0 rejected");
  g.enable_bawp = true;
  g.allow_screen_content_tools = false;
  check(!morph_pred_present(g), "screen tools 0 suppresses syntax");
  g.allow_screen_content_tools = true;
  g.frame_is_intra = false;
  check(!morph_pred_present(g), "inter frame suppresses syntax");
  g.frame_is_intra = true;
  g.use_intrabc = false;
  check(!morph_pred_allowed(g, true), "non-IBC rejected");
  g.use_intrabc = true;
  // enable_imp_msk_bld has no normative effect on morph_pred.
  g.enable_imp_msk_bld = false;
  const bool a = morph_pred_allowed(g, true);
  g.enable_imp_msk_bld = true;
  const bool b = morph_pred_allowed(g, true);
  check(a && b, "enable_imp_msk_bld does not gate morph_pred");

  // Exhaustive truth table for the three syntax bits and the two conformance
  // inputs.
  for (int bits = 0; bits < 16; ++bits) {
    MorphPredGate e;
    e.frame_is_intra = (bits & 1) != 0;
    e.allow_screen_content_tools = (bits & 2) != 0;
    e.enable_bawp = (bits & 4) != 0;
    e.use_intrabc = (bits & 8) != 0;
    const bool present = morph_pred_present(e);
    check(present == ((bits & 7) == 7), "present truth table");
    for (int off = 0; off < 2; ++off) {
      const bool allowed = morph_pred_allowed(e, off != 0);
      check(allowed == (present && e.use_intrabc && off != 0),
            "allowed truth table");
    }
  }
}

MorphStreamPlan make_gate_plan(bool morph_enabled) {
  MorphStreamPlan plan;
  plan.config.width = 512;
  plan.config.height = 128;
  const int sb_cols = 8;
  const int sb_rows = 2;
  for (int r = 0; r < sb_rows; ++r) {
    for (int c = 0; c < sb_cols; ++c) {
      MorphStreamSb sb;
      sb.partition = MorphStreamSb::Partition::kHorz;
      for (int half = 0; half < 2; ++half) {
        MorphStreamBlock b;
        b.mi_row = r * 16 + half * 8;
        b.mi_col = c * 16;
        b.bsize = 11;  // 64x32
        b.skip = true;
        // Row 0 is SMOOTH intra: with no residual, this is the only way to
        // make the reconstructed source region non-flat so the row-1 morph
        // fits are non-trivial. SMOOTH is reordered mode index 1 and keeps
        // the y_mode_index context at 0 (non-directional joint mode).
        if (r == 0) b.y_mode = 9;
        if (r == 1 && c >= 2) {
          b.ibc = true;
          const int src_x = c * 64;
          const int src_y = 8;
          const int x = b.mi_col * 4;
          const int y = b.mi_row * 4;
          b.bv.row = (src_y - y) * 8;
          b.bv.col = (src_x - x) * 8;
          b.syntax.mode = av2enc::m1::ibc_stream::IntrabcMode::kNewMv;
          b.syntax.intrabc_precision_quarter = false;
          // Exercise both symbol values and all three contexts. With
          // 2*left + above, the checkerboard below yields ctx 0/1/2.
          b.morph_pred = morph_enabled && (((c + half) % 3) != 0);
        }
        sb.leaves.push_back(b);
      }
      plan.sbs.push_back(std::move(sb));
    }
  }
  return plan;
}

void test_stream(const std::string& prefix) {
  MorphStreamResult r;
  std::string error;
  const MorphStreamPlan plan = make_gate_plan(true);
  check(encode_morph_stream(plan, r, error), "encode morph stream: " + error);
  check(r.ibc_blocks == 12, "gate IBC block count");
  check(r.morph_symbols == 12, "gate morph symbol count");
  check(r.morph_blocks > 0 && r.morph_blocks < r.morph_symbols,
        "gate mixes morph 0/1");
  check(r.morph_ctx[0] > 0 && r.morph_ctx[1] > 0 && r.morph_ctx[2] > 0,
        "gate exercises all three contexts");
  check(!r.frame.empty(), "gate frame non-empty");
  int n127 = 0;
  int n128 = 0;
  for (std::uint8_t v : r.recon_y) {
    n127 += v == 127 ? 1 : 0;
    n128 += v == 128 ? 1 : 0;
  }
  // Ideal flag rate from the frozen §9.3 CDFs. The arithmetic coder's final
  // alignment makes the byte delta (4 bytes) coarser than this.
  double flag_bits = 0.0;
  static constexpr double kCdf[3][2] = {
      {31715.0 / 32768.0, 55.0 / 32768.0},
      {19667.0 / 32768.0, 1.0 / 32768.0},
      {10555.0 / 32768.0, 91.0 / 32768.0},
  };
  double flag_bits_off = 0.0;
  for (int c = 0; c < 3; ++c) {
    const int total = r.morph_ctx_value[c][0] + r.morph_ctx_value[c][1];
    for (int v = 0; v < 2; ++v) {
      for (int k = 0; k < r.morph_ctx_value[c][v]; ++k) {
        flag_bits -= std::log2(kCdf[c][v]);
      }
    }
    flag_bits_off -= total * std::log2(kCdf[c][0]);
  }
  const double flag_delta = flag_bits - flag_bits_off;
  std::printf("gate recon: morph_blocks=%d fitted=%d alpha=%d..%d 127=%d "
              "128=%d flag=%.1f bits (all-zero %.1f, delta %+.1f) "
              "ctx0/1/2=%d/%d/%d\n",
              r.morph_blocks, r.morph_fitted, r.morph_alpha_min,
              r.morph_alpha_max, n127, n128, flag_bits, flag_bits_off,
              flag_delta, r.morph_ctx[0], r.morph_ctx[1], r.morph_ctx[2]);

  // The same plan with every morph_pred = 0 must be valid and must decode
  // through the same writer.
  MorphStreamResult off;
  const MorphStreamPlan plan_off = make_gate_plan(false);
  check(encode_morph_stream(plan_off, off, error),
        "encode morph-off stream: " + error);
  check(off.morph_symbols == 12 && off.morph_blocks == 0,
        "off stream still codes symbols");
  check(off.frame.size() != r.frame.size() ||
            std::memcmp(off.frame.data(), r.frame.data(), r.frame.size()) != 0,
        "morph changes the bitstream");

  if (!prefix.empty()) {
    check(write_morph_ivf(r, plan.config, prefix + ".ivf", error),
          "write gate IVF: " + error);
    check(write_morph_y4m(r, plan.config, prefix + ".y4m", error),
          "write gate Y4M: " + error);
    check(write_morph_ivf(off, plan.config, prefix + ".off.ivf", error),
          "write off IVF: " + error);
    check(write_morph_y4m(off, plan.config, prefix + ".off.y4m", error),
          "write off Y4M: " + error);
  }

  // enable_bawp=0 suppresses the symbol entirely; a morph_pred=1 plan must be
  // rejected rather than silently coding no flag.
  MorphStreamResult no_bawp;
  MorphStreamPlan no_bawp_plan = make_gate_plan(false);
  no_bawp_plan.config.enable_bawp = false;
  check(encode_morph_stream(no_bawp_plan, no_bawp, error),
        "enable_bawp=0 stream: " + error);
  check(no_bawp.morph_symbols == 0 && no_bawp.morph_blocks == 0,
        "enable_bawp=0 emits no morph_pred symbol");
  MorphStreamPlan bad_bawp = make_gate_plan(true);
  bad_bawp.config.enable_bawp = false;
  MorphStreamResult bad_bawp_out;
  check(!encode_morph_stream(bad_bawp, bad_bawp_out, error),
        "morph_pred=1 rejected when enable_bawp=0");

  // A morph_pred=1 block whose -1,-1 border is outside the tile must be
  // rejected before a bit is emitted.
  MorphStreamPlan bad = make_gate_plan(true);
  for (MorphStreamSb& sb : bad.sbs) {
    for (MorphStreamBlock& b : sb.leaves) {
      if (b.ibc) {
        b.bv = av2enc::ibc::BlockVector{0, 0};  // source == current block
        b.morph_pred = true;
        break;
      }
    }
  }
  MorphStreamResult bad_out;
  check(!encode_morph_stream(bad, bad_out, error),
        "invalid morph reference rejected");

  // A base BV that is legal for IBC but whose -1,-1 border leaves the tile:
  // rejected by the morph conformance check even with strict_border off.
  MorphStreamPlan border = make_gate_plan(true);
  border.config.strict_border = false;
  for (MorphStreamSb& sb : border.sbs) {
    for (MorphStreamBlock& b : sb.leaves) {
      if (b.ibc) {
        b.bv.col = -b.mi_col * 4 * 8;  // source x = 0
        b.bv.row = (8 - b.mi_row * 4) * 8;
        b.morph_pred = true;
        break;
      }
    }
  }
  MorphStreamResult border_out;
  check(!encode_morph_stream(border, border_out, error),
        "morph_pred rejected when is_offset_mv_valid(-1,-1) is false");
}

int load_y4m_luma(const char* path, int* w, int* h,
                  std::vector<std::uint8_t>* y) {
  std::FILE* f = std::fopen(path, "rb");
  if (f == nullptr) return 0;
  char line[256];
  if (std::fgets(line, sizeof(line), f) == nullptr) {
    std::fclose(f);
    return 0;
  }
  int W = 0;
  int H = 0;
  if (std::sscanf(line, "YUV4MPEG2 W%d H%d", &W, &H) != 2) {
    std::fclose(f);
    return 0;
  }
  if (std::fgets(line, sizeof(line), f) == nullptr ||
      std::strncmp(line, "FRAME", 5) != 0) {
    std::fclose(f);
    return 0;
  }
  y->assign(static_cast<std::size_t>(W) * H, 0);
  const std::size_t got = std::fread(y->data(), 1, y->size(), f);
  std::fclose(f);
  if (got != y->size()) return 0;
  *w = W;
  *h = H;
  return 1;
}

// Prediction-only mode-selection impact. The reconstructed source frame is
// used as the surrounding context, so this isolates the brightness fit from
// residual coding. Every IBC-legal 64x32 block copies a nearby earlier region.
void benchmark_mode_selection() {
  int w = 0;
  int h = 0;
  std::vector<std::uint8_t> src;
  const char* candidates[] = {
      "../../../../testdata/shine_256x256_1f.y4m",
      "../../../testdata/shine_256x256_1f.y4m",
      "testdata/shine_256x256_1f.y4m",
  };
  bool loaded = false;
  for (const char* p : candidates) {
    if (load_y4m_luma(p, &w, &h, &src)) {
      loaded = true;
      break;
    }
  }
  if (!loaded) {
    w = 256;
    h = 256;
    src = make_plane(w, h);
  }
  if (w < 128 || h < 128) return;
  long long sse_copy = 0;
  long long sse_morph = 0;
  int blocks = 0;
  int selected = 0;
  int skipped = 0;
  int ctx[3] = {0, 0, 0};
  int ctx_value[3][2] = {};
  static constexpr double kCdf[3][2] = {
      {31715.0 / 32768.0, 55.0 / 32768.0},
      {19667.0 / 32768.0, 1.0 / 32768.0},
      {10555.0 / 32768.0, 91.0 / 32768.0},
  };
  const int mi_cols = 2 * ((w + 7) / 8);
  const int mi_rows = 2 * ((h + 7) / 8);
  std::vector<std::uint8_t> morph_grid(
      static_cast<std::size_t>(mi_cols) * static_cast<std::size_t>(mi_rows),
      0);
  for (int by = 0; by + 32 <= h; by += 32) {
    for (int bx = 0; bx + 64 <= w; bx += 64) {
      if (by == 0 && bx == 0) {
        ++skipped;
        continue;
      }
      // Raster IBC: use the block one 64-wide column to the left, or the last
      // block of the previous row for the first column.
      int ref_x = 0;
      int ref_y = 0;
      if (bx >= 64) {
        ref_x = bx - 64;
        ref_y = by;
      } else {
        ref_x = ((w - 64) / 64) * 64;
        ref_y = by - 32;
      }
      if (ref_x < 1 || ref_y < 1) {
        ++skipped;
        continue;
      }
      std::vector<std::uint8_t> pred(src);
      for (int i = 0; i < 32; ++i) {
        std::memcpy(pred.data() + static_cast<std::size_t>(by + i) * w + bx,
                    src.data() + static_cast<std::size_t>(ref_y + i) * w +
                        ref_x,
                    64);
      }
      const int mi_col = bx / 4;
      const int mi_row = by / 4;
      const int bw4 = 16;
      const int bh4 = 8;
      const int c = morph_symbol_context(morph_grid.data(), mi_cols, mi_rows,
                                         mi_row, mi_col, bw4, bh4, 64);
      MorphContext ctxm;
      ctxm.mi_col = mi_col;
      ctxm.mi_row = mi_row;
      ctxm.mi_cols = mi_cols;
      ctxm.mi_rows = mi_rows;
      ctxm.avail_u = by > 0;
      ctxm.avail_l = bx > 0;
      MorphPlane mp{pred.data(), w};
      MorphRect rect{bx, by, 64, 32};
      MorphFit fit{};
      const bool ok = morph_predict(ctxm, mp, rect, (ref_y - by) * 8,
                                    (ref_x - bx) * 8, &fit);
      long long d0 = 0;
      long long d1 = 0;
      for (int i = 0; i < 32; ++i) {
        for (int j = 0; j < 64; ++j) {
          const int s = src[static_cast<std::size_t>(by + i) * w + bx + j];
          const int p0 = src[static_cast<std::size_t>(ref_y + i) * w + ref_x + j];
          const int p1 = pred[static_cast<std::size_t>(by + i) * w + bx + j];
          d0 += (s - p0) * (s - p0);
          d1 += (s - p1) * (s - p1);
        }
      }
      ++blocks;
      sse_copy += d0;
      if (ok && d1 < d0) {
        ++selected;
        sse_morph += d1;
        ctx[c] += 1;
        ++ctx_value[c][1];
        for (int yy = 0; yy < 8; ++yy) {
          for (int xx = 0; xx < 16; ++xx) {
            morph_grid[static_cast<std::size_t>(mi_row + yy) * mi_cols +
                       mi_col + xx] = 1;
          }
        }
      } else {
        sse_morph += d0;
        ++ctx_value[c][0];
      }
    }
  }
  if (blocks == 0) return;
  const double n = static_cast<double>(blocks) * 64.0 * 32.0;
  const double mse0 = static_cast<double>(sse_copy) / n;
  const double mse1 = static_cast<double>(sse_morph) / n;
  const double psnr0 =
      mse0 == 0.0 ? 99.0 : 10.0 * std::log10(255.0 * 255.0 / mse0);
  const double psnr1 =
      mse1 == 0.0 ? 99.0 : 10.0 * std::log10(255.0 * 255.0 / mse1);
  double flag_bits = 0.0;
  double flag_bits_off = 0.0;
  for (int c = 0; c < 3; ++c) {
    const int total = ctx_value[c][0] + ctx_value[c][1];
    for (int v = 0; v < 2; ++v) {
      for (int k = 0; k < ctx_value[c][v]; ++k) {
        flag_bits -= std::log2(kCdf[c][v]);
      }
    }
    flag_bits_off -= total * std::log2(kCdf[c][0]);
  }
  std::printf(
      "mode-selection: blocks=%d skipped=%d selected=%d (%.1f%%) "
      "PSNR copy=%.3f dB morph=%.3f dB delta=%+.3f dB "
      "flag=%+.1f bits ctx0/1/2=%d/%d/%d\n",
      blocks, skipped, selected, 100.0 * selected / blocks, psnr0, psnr1,
      psnr1 - psnr0, flag_bits - flag_bits_off, ctx[0], ctx[1], ctx[2]);
}

}  // namespace

int main(int argc, char** argv) {
  bool emit = false;
  std::string prefix;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--emit-goldens") == 0) {
      emit = true;
    } else if (std::strcmp(argv[i], "--out-prefix") == 0 && i + 1 < argc) {
      prefix = argv[++i];
    }
  }
  if (emit) {
    emit_goldens();
    return 0;
  }

  test_math();
  test_template_plan();
  test_prediction_goldens();
  test_bounds();
  test_syntax();
  test_gate();
  test_stream(prefix);
  benchmark_mode_selection();

  if (g_failures != 0) {
    std::fprintf(stderr, "morph_test: %d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("morph_test: %d checks passed\n", g_checks);
  return 0;
}

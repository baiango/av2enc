// Tests for the encoder-side Wiener search and decisions.
//
// Coverage:
//   * least-squares fit on synthetic and real (Y4M) reconstruction pairs,
//   * tap-range clipping, subset selection, chroma uv_sym pairs,
//   * deterministic repeats,
//   * missing normative class tables are reported, never invented,
//   * per-unit and frame-level decisions,
//   * decision SSE matches the §7.20 restoration core exactly,
//   * header + tile syntax round-trip reproduces the same recon,
//   * real recon from tests/data/basic_64x64_1f.y4m.
//
// Build/run: ./build_and_test.sh

#include "codec/filters/loop_restoration/classes/wiener_class.h"
#include "codec/filters/loop_restoration/classes/wiener_decision.h"
#include "codec/filters/loop_restoration/classes/wiener_fit.h"
#include "codec/filters/loop_restoration/classes/wiener_syntax.h"
#include "codec/filters/loop_restoration/restoration.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

using namespace av2enc::restoration;
using namespace av2enc::restoration::classes;
using av2enc::BitWriter;
namespace av2 = av2enc::av2;

int g_checks = 0;
int g_failures = 0;

#define CHECK(cond, msg)                                                   \
  do {                                                                     \
    ++g_checks;                                                            \
    if (!(cond)) {                                                         \
      ++g_failures;                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, \
                 std::string(msg).c_str()); \
    }                                                                      \
  } while (0)

constexpr int kGuard = 8;

int clip3i(int lo, int hi, int v) {
  return v < lo ? lo : (v > hi ? hi : v);
}

struct TestImage {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> plane[3];

  int plane_width(int p) const { return width >> (p == 0 ? 0 : 1); }
  int plane_height(int p) const { return height >> (p == 0 ? 0 : 1); }
  std::uint8_t& at(int p, int y, int x) {
    return plane[p][static_cast<std::size_t>(y) *
                        static_cast<std::size_t>(plane_width(p)) +
                    static_cast<std::size_t>(x)];
  }
  std::uint8_t at(int p, int y, int x) const {
    return plane[p][static_cast<std::size_t>(y) *
                        static_cast<std::size_t>(plane_width(p)) +
                    static_cast<std::size_t>(x)];
  }
};

TestImage read_y4m(const char* path) {
  TestImage img;
  std::ifstream in(path, std::ios::binary);
  if (!in.good()) {
    return img;
  }
  std::string header;
  std::getline(in, header);
  const std::size_t wpos = header.find(" W");
  const std::size_t hpos = header.find(" H");
  if (wpos == std::string::npos || hpos == std::string::npos) {
    CHECK(false, "parse Y4M dimensions");
    return img;
  }
  img.width = std::stoi(header.substr(wpos + 2));
  img.height = std::stoi(header.substr(hpos + 2));
  std::string frame;
  std::getline(in, frame);
  CHECK(frame.rfind("FRAME", 0) == 0, "Y4M FRAME marker");
  for (int p = 0; p < 3; ++p) {
    const int w = img.plane_width(p);
    const int h = img.plane_height(p);
    img.plane[p].resize(static_cast<std::size_t>(w) *
                        static_cast<std::size_t>(h));
    in.read(reinterpret_cast<char*>(img.plane[p].data()),
            static_cast<std::streamsize>(img.plane[p].size()));
  }
  CHECK(in.gcount() > 0 || in.eof(), "read Y4M planes");
  return img;
}

TestImage make_synthetic(int width, int height) {
  TestImage img;
  img.width = width;
  img.height = height;
  for (int p = 0; p < 3; ++p) {
    const int w = img.plane_width(p);
    const int h = img.plane_height(p);
    img.plane[p].resize(static_cast<std::size_t>(w) *
                        static_cast<std::size_t>(h));
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const int v = 24 + ((x * 5 + y * 3 + p * 37) & 127) +
                      ((x > w / 2) ? 64 : 0);
        img.at(p, y, x) = static_cast<std::uint8_t>(clip3i(0, 255, v));
      }
    }
  }
  return img;
}

// A deterministic degradation that a linear filter can partially undo.
TestImage blur(const TestImage& src) {
  TestImage out = src;
  for (int p = 0; p < 3; ++p) {
    const int w = src.plane_width(p);
    const int h = src.plane_height(p);
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const int c = src.at(p, y, x);
        const int l = src.at(p, y, std::max(0, x - 1));
        const int r = src.at(p, y, std::min(w - 1, x + 1));
        const int u = src.at(p, std::max(0, y - 1), x);
        const int d = src.at(p, std::min(h - 1, y + 1), x);
        out.at(p, y, x) =
            static_cast<std::uint8_t>((c * 4 + l + r + u + d + 4) / 8);
      }
    }
  }
  return out;
}

struct TestCore {
  LrCoreFrame core;
  std::vector<std::uint8_t> curr[3];
  std::vector<std::uint8_t> cdef[3];
  std::vector<std::uint8_t> lr[3];
  int stride[3] = {0, 0, 0};
  int width[3] = {0, 0, 0};
  int height[3] = {0, 0, 0};
};

TestCore make_core(const TestImage& recon) {
  TestCore t;
  t.core.mi_cols = recon.width / 4;
  t.core.mi_rows = recon.height / 4;
  t.core.sub_x = 1;
  t.core.sub_y = 1;
  t.core.bit_depth = 8;
  t.core.sb_shift = 4;
  t.core.cfl_ds_filter_index = 0;
  for (int p = 0; p < 3; ++p) {
    const int w = recon.plane_width(p);
    const int h = recon.plane_height(p);
    t.width[p] = w;
    t.height[p] = h;
    t.stride[p] = w + kGuard;
    const std::size_t bytes =
        static_cast<std::size_t>(t.stride[p]) *
                       static_cast<std::size_t>(h + kGuard);
    t.curr[p].assign(bytes, 0);
    t.cdef[p].assign(bytes, 0);
    t.lr[p].assign(bytes, 0);
    for (int y = 0; y < h; ++y) {
      std::memcpy(&t.curr[p][static_cast<std::size_t>(y) *
                       static_cast<std::size_t>(t.stride[p])],
                  &recon.plane[p][static_cast<std::size_t>(y) *
                                static_cast<std::size_t>(w)],
                  static_cast<std::size_t>(w));
      std::memcpy(&t.cdef[p][static_cast<std::size_t>(y) *
                       static_cast<std::size_t>(t.stride[p])],
                  &recon.plane[p][static_cast<std::size_t>(y) *
                                static_cast<std::size_t>(w)],
                  static_cast<std::size_t>(w));
    }
    t.core.plane[p].curr = t.curr[p].data();
    t.core.plane[p].curr_stride = t.stride[p];
    t.core.plane[p].cdef = t.cdef[p].data();
    t.core.plane[p].cdef_stride = t.stride[p];
    t.core.plane[p].dst = t.lr[p].data();
    t.core.plane[p].dst_stride = t.stride[p];
  }
  return t;
}

double plane_sse(const std::vector<std::uint8_t>& dst, int stride,
                 const TestImage& target, int plane) {
  double sse = 0.0;
  for (int y = 0; y < target.plane_height(plane); ++y) {
    for (int x = 0; x < target.plane_width(plane); ++x) {
      const int a = dst[static_cast<std::size_t>(y) *
                             static_cast<std::size_t>(stride) +
                         static_cast<std::size_t>(x)];
      const int b = target.at(plane, y, x);
      sse += static_cast<double>(a - b) * (a - b);
    }
  }
  return sse;
}

double total_sse(const TestCore& core, const TestImage& target) {
  double sse = 0.0;
  for (int p = 0; p < 3; ++p) {
    sse += plane_sse(core.lr[p], core.stride[p], target, p);
  }
  return sse;
}

void apply_decision(const WienerDecision& d, TestCore* core) {
  RestorationFrame frame;
  for (int p = 0; p < 3; ++p) {
    frame.curr[p].data = core->curr[p].data();
    frame.curr[p].stride = core->stride[p];
    frame.cdef[p].data = core->cdef[p].data();
    frame.cdef[p].stride = core->stride[p];
    frame.lr[p].data = core->lr[p].data();
    frame.lr[p].stride = core->stride[p];
  }
  frame.mi_cols = core->core.mi_cols;
  frame.mi_rows = core->core.mi_rows;
  frame.subsampling_x = core->core.sub_x;
  frame.subsampling_y = core->core.sub_y;
  frame.bit_depth = 8;
  frame.cfl_ds_filter_index = 0;

  LoopRestorationParams lp;
  wiener_decision_to_loop_params(d, &lp);
  const RestorationStatus st = loop_restoration_frame(frame, lp);
  CHECK(st == RestorationStatus::kOk, "loop_restoration_frame accepts decision");
}

// ---------------------------------------------------------------------------
// Fit tests
// ---------------------------------------------------------------------------

void test_fit_identity_and_blur() {
  const TestImage src = make_synthetic(64, 64);
  const TestImage recon = blur(src);
  TestCore core = make_core(recon);
  core.core.frame_type[0] = kRestoreWienerNs;
  core.core.unit_size[0] = 64;
  core.core.plane[0].unit_rows = 1;
  core.core.plane[0].unit_cols = 1;
  std::vector<std::uint8_t> types(1, kRestoreWienerNs);
  std::vector<std::int16_t> coeffs(18, 0);
  core.core.plane[0].unit_type = types.data();
  core.core.plane[0].unit_coeffs = coeffs.data();

  WienerFitOptions opts;
  WienerFitResult fit;
  const WienerFitStatus st = wiener_fit_candidates(
      core.core, 0, 0, 0, -1, nullptr, src.plane[0].data(),
      src.plane_width(0), opts, &fit);
  CHECK(st == WienerFitStatus::kOk, "luma fit succeeds");
  CHECK(!fit.candidates.empty(), "luma fit returns candidates");
  const WienerFitCandidate* best = wiener_best_fit_candidate(fit);
  CHECK(best != nullptr, "luma best candidate");
  if (best != nullptr) {
    CHECK(best->sse < fit.identity_sse,
          "blur fit improves on the unfiltered reconstruction");
    for (int j = 0; j < kWienerNsLumaCoeffs; ++j) {
      CHECK(best->coeffs[j] >= wiener_ns_coeff_min(false, j) &&
                best->coeffs[j] <= wiener_ns_coeff_max(false, j),
            "luma coefficient inside Wiener_Ns_Taps range");
      if (!wiener_tap_present(0, best->subset, j)) {
        CHECK(best->coeffs[j] == 0, "absent luma tap is zero");
      }
    }
  }

  // Determinism.
  WienerFitResult again;
  CHECK(wiener_fit_candidates(core.core, 0, 0, 0, -1, nullptr,
                              src.plane[0].data(), src.plane_width(0), opts,
                              &again) == WienerFitStatus::kOk,
        "second luma fit succeeds");
  CHECK(again.candidates.size() == fit.candidates.size(),
        "deterministic candidate count");
  if (!again.candidates.empty() && !fit.candidates.empty()) {
    const WienerFitCandidate* b2 = wiener_best_fit_candidate(again);
    CHECK(b2 != nullptr && b2->sse == best->sse &&
              std::memcmp(b2->coeffs, best->coeffs, sizeof(best->coeffs)) == 0,
          "deterministic fitted coefficients");
  }

  // Identity target: the fit must never be worse than no filtering.
  WienerFitResult ident;
  core.core.frame_type[0] = kRestoreWienerNs;
  CHECK(wiener_fit_candidates(core.core, 0, 0, 0, -1, nullptr,
                              recon.plane[0].data(), recon.plane_width(0),
                              opts, &ident) == WienerFitStatus::kOk,
        "identity fit succeeds");
  const WienerFitCandidate* ib = wiener_best_fit_candidate(ident);
  CHECK(ib != nullptr && ib->sse <= ident.identity_sse + 1e-9,
        "identity fit is not worse than the identity filter");
}

void test_fit_chroma_uv_sym() {
  const TestImage src = make_synthetic(64, 64);
  const TestImage recon = blur(src);
  TestCore core = make_core(recon);
  core.core.frame_type[1] = kRestoreWienerNs;
  core.core.unit_size[1] = 32;
  core.core.plane[1].unit_rows = 1;
  core.core.plane[1].unit_cols = 1;
  std::vector<std::uint8_t> types(1, kRestoreWienerNs);
  std::vector<std::int16_t> coeffs(18, 0);
  core.core.plane[1].unit_type = types.data();
  core.core.plane[1].unit_coeffs = coeffs.data();

  WienerFitOptions opts;
  WienerFitResult fit;
  const WienerFitStatus st = wiener_fit_candidates(
      core.core, 1, 0, 0, -1, nullptr, src.plane[1].data(),
      src.plane_width(1), opts, &fit);
  CHECK(st == WienerFitStatus::kOk, "chroma fit succeeds");
  bool saw_sym = false;
  for (const WienerFitCandidate& c : fit.candidates) {
    if (!c.uv_sym) {
      continue;
    }
    saw_sym = true;
    for (int j = kWienerNsShortCoeffs; j + 1 < kWienerNsChromaCoeffs;
         j += 2) {
      CHECK(c.coeffs[j] == c.coeffs[j + 1], "uv_sym pair is equal");
    }
  }
  CHECK(saw_sym, "chroma fit offers a uv_sym candidate");
}

// ---------------------------------------------------------------------------
// Class tables
// ---------------------------------------------------------------------------

void test_missing_class_tables() {
  TestCore holder = make_core(make_synthetic(64, 64));
  LrCoreFrame& core = holder.core;
  LrBlock b;
  // Populate a minimal block context through the core.
  core.frame_type[0] = kRestoreWienerNs;
  core.unit_size[0] = 64;
  std::vector<std::uint8_t> types(1, kRestoreWienerNs);
  std::vector<std::int16_t> coeffs(18, 0);
  core.plane[0].unit_rows = 1;
  core.plane[0].unit_cols = 1;
  core.plane[0].unit_type = types.data();
  core.plane[0].unit_coeffs = coeffs.data();
  CHECK(lr_block_context(core, 0, 0, 0, &b), "block context for classifier");

  PcWienerTables none;
  WienerClassifierContext ctx;
  ctx.core = &core;
  ctx.mi_row_end = core.mi_rows;
  ctx.mi_col_end = core.mi_cols;
  ctx.base_q_idx = 90;
  ctx.bit_depth = 8;
  std::uint8_t cls = 0;
  CHECK(wiener_derive_filter_class(core, b, none, ctx, &cls) ==
            WienerTableStatus::kMissingNormativeTable,
        "classifier refuses to invent Pc_Wiener_Lut_To_Class");

  std::vector<std::uint8_t> grid(4, 0);
  std::vector<std::uint8_t> sub;
  CHECK(wiener_build_subclass_grid(grid.data(), 2, 2, 1, 2, 1, none, &sub) ==
            WienerTableStatus::kMissingNormativeTable,
        "subclass grid refuses to invent Pc_Wiener_Sub_Classify2");
}

void test_class_derivation_synthetic_tables() {
  const TestImage img = make_synthetic(64, 64);
  TestCore t = make_core(img);
  LrCoreFrame& core = t.core;
  core.frame_type[0] = kRestoreWienerNs;
  core.unit_size[0] = 64;
  std::vector<std::uint8_t> types(1, kRestoreWienerNs);
  std::vector<std::int16_t> coeffs(18, 0);
  core.plane[0].unit_rows = 1;
  core.plane[0].unit_cols = 1;
  core.plane[0].unit_type = types.data();
  core.plane[0].unit_coeffs = coeffs.data();

  // Synthetic tables exercise the mechanics only; they are not normative.
  std::vector<std::uint8_t> lut(4096);
  for (int i = 0; i < 4096; ++i) {
    lut[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>((i * 7) & 255);
  }
  std::vector<std::uint8_t> sub2(4 * 8 * 256);
  for (int set = 0; set < 4; ++set) {
    for (int idx = 0; idx < 8; ++idx) {
      for (int cls = 0; cls < 256; ++cls) {
        sub2[static_cast<std::size_t>((set * 8 + idx) * 256 + cls)] =
            static_cast<std::uint8_t>(cls % decode_num_filter_classes(idx));
      }
    }
  }
  PcWienerTables tables;
  tables.lut_to_class = lut.data();
  tables.sub_classify2 = sub2.data();

  WienerClassifierContext ctx;
  ctx.core = &core;
  ctx.mi_row_end = core.mi_rows;
  ctx.mi_col_end = core.mi_cols;
  ctx.base_q_idx = 90;
  ctx.bit_depth = 8;
  std::vector<std::uint8_t> classes;
  CHECK(wiener_derive_filter_class_grid(core, tables, ctx, &classes) ==
            WienerTableStatus::kOk,
        "synthetic classifier produces a grid");
  CHECK(classes.size() ==
            static_cast<std::size_t>(core.mi_cols) *
                static_cast<std::size_t>(core.mi_rows),
        "class grid size");

  std::vector<std::uint8_t> sub;
  CHECK(wiener_build_subclass_grid(classes.data(), core.mi_cols, core.mi_rows,
                                   1, 4, 3, tables, &sub) ==
            WienerTableStatus::kOk,
        "synthetic subclass grid");
  for (std::size_t i = 0; i < sub.size(); ++i) {
    CHECK(sub[i] < 4, "subclass in range");
    CHECK(sub[i] == classes[i] % 4, "subclass mapping");
  }
}

// ---------------------------------------------------------------------------
// Decisions
// ---------------------------------------------------------------------------

struct DecisionFixture {
  TestImage src;
  TestImage recon;
  TestCore core;
};

DecisionFixture make_fixture(const TestImage& src) {
  DecisionFixture f;
  f.src = src;
  f.recon = blur(src);
  f.core = make_core(f.recon);
  return f;
}

WienerDecisionInput make_input(const DecisionFixture& f) {
  WienerDecisionInput in;
  in.core = f.core.core;
  for (int p = 0; p < 3; ++p) {
    in.target[p] = f.src.plane[p].data();
    in.target_stride[p] = f.src.plane_width(p);
  }
  in.base_q_idx = 90;
  return in;
}

void check_decision_roundtrip(DecisionFixture* f, const WienerDecision& d,
                              const char* label) {
  apply_decision(d, &f->core);
  const double got = total_sse(f->core, f->src);
  CHECK(std::fabs(got - d.sse) < 1e-6,
        std::string(label) + ": decision SSE matches restoration core");

  // Header round-trip.
  WienerLrHeaderConfig config;
  config.enable_restoration = true;
  config.num_planes = 3;
  config.sb_size = 64;
  config.sub_x = 1;
  config.sub_y = 1;
  config.base_q_idx = 90;
  config.lr_tools_disable[0][kRestorePcWiener] = true;
  config.lr_tools_disable[1][kRestorePcWiener] = true;
  config.lr_tools_disable[0][kRestoreWienerNs] = false;
  config.lr_tools_disable[1][kRestoreWienerNs] = false;
  WienerReferenceSet refs;
  BitWriter writer;
  WienerLrHeaderParams params = d.params;
  std::string error;
  CHECK(wiener_write_lr_params(writer, config, &params, refs, &error),
        std::string(label) + ": lr_params write");
  CHECK(error.empty(), std::string(label) + ": lr_params no error");
  WienerLrHeaderParams back;
  WienerBitReader reader(writer.bytes().data(), writer.bit_size());
  CHECK(wiener_read_lr_params(reader, config, &back, refs, &error),
        std::string(label) + ": lr_params read");
  CHECK(error.empty(), std::string(label) + ": lr_params read no error");
  CHECK(reader.bits_left() == 0, std::string(label) + ": header fully consumed");
  for (int p = 0; p < 3; ++p) {
    CHECK(back.frame_type[p] == params.frame_type[p],
          std::string(label) + ": frame_type round-trip");
    CHECK(back.frame_filters_on[p] == params.frame_filters_on[p],
          std::string(label) + ": frame_filters_on round-trip");
    if (back.frame_filters_on[p]) {
      for (int c = 0; c < back.num_filter_classes; ++c) {
        for (int j = 0; j < kWienerNsCoeffStride; ++j) {
          CHECK(back.frame_coeffs[p][c][j] == params.frame_coeffs[p][c][j],
                std::string(label) + ": frame coefficient round-trip");
        }
      }
    }
  }

  // Tile round-trip.
  WienerTileState write_state;
  CHECK(wiener_tile_init(&write_state, back, f->core.core.mi_cols,
                         f->core.core.mi_rows, 1, 1, &error),
        std::string(label) + ": tile init");
  std::vector<WienerUnitSyntax> plans[3];
  for (int p = 0; p < 3; ++p) {
    plans[p].assign(d.plane[p].units.size(), WienerUnitSyntax{});
    for (std::size_t i = 0; i < plans[p].size(); ++i) {
      const WienerUnitPlan& u = d.plane[p].units[i];
      plans[p][i].type = u.type;
      plans[p][i].merged = u.merged;
      plans[p][i].use_bank_k = u.use_bank_k;
      plans[p][i].subset = u.subset;
      plans[p][i].uv_sym = u.uv_sym;
      for (int j = 0; j < kWienerNsCoeffStride; ++j) {
        plans[p][i].coeffs[j] = u.coeffs[j];
      }
    }
  }
  std::vector<std::uint8_t> buffer(1 << 22);
  av2::Encoder encoder;
  CHECK(encoder.init_symbol(buffer.data(), buffer.size()) == av2::Status::kOk,
        std::string(label) + ": tile encoder init");
  CHECK(wiener_write_tile(encoder, write_state, plans, 64, 0, 0, &error),
        std::string(label) + ": tile write");
  CHECK(error.empty(), std::string(label) + ": tile write no error");

  WienerTileState read_state;
  CHECK(wiener_tile_init(&read_state, back, f->core.core.mi_cols,
                         f->core.core.mi_rows, 1, 1, &error),
        std::string(label) + ": read tile init");
  av2::Decoder decoder;
  std::size_t payload = 0;
  CHECK(encoder.exit_symbol(&payload) == av2::Status::kOk,
        std::string(label) + ": tile encoder exit");
  CHECK(decoder.init_symbol(buffer.data(), buffer.size(), payload) ==
            av2::Status::kOk,
        std::string(label) + ": tile decoder init");
  std::vector<WienerUnitSyntax> got_plans[3];
  CHECK(wiener_read_tile(decoder, read_state, got_plans, 64, 0, 0, &error),
        std::string(label) + ": tile read");
  CHECK(error.empty(), std::string(label) + ": tile read no error");
  CHECK(decoder.exit_symbol() == av2::Status::kOk,
        std::string(label) + ": tile decoder exit");
  for (int p = 0; p < 3; ++p) {
    CHECK(read_state.unit_type[p] == d.unit_type[p],
          std::string(label) + ": unit type grid round-trip");
    CHECK(read_state.unit_coeffs[p] == d.unit_coeffs[p],
          std::string(label) + ": unit coefficient grid round-trip");
  }
}

void test_decision_synthetic() {
  DecisionFixture f = make_fixture(make_synthetic(64, 64));
  WienerDecisionInput in = make_input(f);
  WienerDecisionOptions opts;
  opts.max_num_filter_classes = 1;
  opts.lambda_bits = 0.0;
  WienerDecision d;
  const WienerDecisionStatus st = wiener_decide(in, opts, &d);
  CHECK(st == WienerDecisionStatus::kOk, "synthetic decision succeeds");

  // Identity baseline.
  std::int16_t zero[18] = {};
  double identity = 0.0;
  for (int p = 0; p < 3; ++p) {
    identity += plane_sse(f.core.cdef[p], f.core.stride[p], f.src, p);
  }
  (void)zero;
  CHECK(d.sse <= identity + 1e-9,
        "decision is not worse than no loop restoration");
  CHECK(d.sse < identity, "decision finds a filter for a blurred recon");

  // Determinism.
  WienerDecision d2;
  CHECK(wiener_decide(in, opts, &d2) == WienerDecisionStatus::kOk,
        "second decision succeeds");
  CHECK(d2.sse == d.sse, "deterministic SSE");
  CHECK(d2.header_bits == d.header_bits, "deterministic header bits");
  CHECK(d2.tile_bits == d.tile_bits, "deterministic tile bits");
  for (int p = 0; p < 3; ++p) {
    CHECK(d2.params.frame_type[p] == d.params.frame_type[p],
          "deterministic frame type");
    CHECK(d2.unit_type[p] == d.unit_type[p], "deterministic unit types");
    CHECK(d2.unit_coeffs[p] == d.unit_coeffs[p],
          "deterministic unit coefficients");
  }

  check_decision_roundtrip(&f, d, "synthetic");
}

void test_decision_frame_filters() {
  DecisionFixture f = make_fixture(make_synthetic(64, 64));
  WienerDecisionInput in = make_input(f);
  WienerDecisionOptions opts;
  opts.allow_per_unit = false;
  opts.allow_frame_filters = true;
  opts.max_num_filter_classes = 1;
  opts.lambda_bits = 0.0;
  WienerDecision d;
  CHECK(wiener_decide(in, opts, &d) == WienerDecisionStatus::kOk,
        "frame-filter decision succeeds");
  CHECK(d.params.frame_type[0] == kRestoreWienerNs,
        "frame-filter decision enables luma LR");
  CHECK(d.params.frame_filters_on[0],
        "frame-filter decision uses frame-level luma filters");
  check_decision_roundtrip(&f, d, "frame");
}

void test_decision_real_recon() {
  TestImage src = read_y4m("tests/data/basic_64x64_1f.y4m");
  if (src.width == 0) {
    // Fall back when run from the component directory.
    src = read_y4m("../../../../../tests/data/basic_64x64_1f.y4m");
  }
  CHECK(src.width == 64 && src.height == 64, "real Y4M loaded");
  if (src.width == 0) {
    return;
  }
  DecisionFixture f = make_fixture(src);
  WienerDecisionInput in = make_input(f);
  WienerDecisionOptions opts;
  opts.max_num_filter_classes = 1;
  opts.lambda_bits = 0.0;
  WienerDecision d;
  CHECK(wiener_decide(in, opts, &d) == WienerDecisionStatus::kOk,
        "real-recon decision succeeds");
  double identity = 0.0;
  for (int p = 0; p < 3; ++p) {
    identity += plane_sse(f.core.cdef[p], f.core.stride[p], f.src, p);
  }
  CHECK(d.sse <= identity + 1e-9, "real decision not worse than identity");
  check_decision_roundtrip(&f, d, "real");
}

void test_class_grid_256_extension() {
  TestImage img = make_synthetic(64, 64);
  TestCore t = make_core(img);
  RestorationFrame frame;
  for (int p = 0; p < 3; ++p) {
    frame.curr[p].data = t.curr[p].data();
    frame.curr[p].stride = t.stride[p];
    frame.cdef[p].data = t.cdef[p].data();
    frame.cdef[p].stride = t.stride[p];
    frame.lr[p].data = t.lr[p].data();
    frame.lr[p].stride = t.stride[p];
  }
  frame.mi_cols = t.core.mi_cols;
  frame.mi_rows = t.core.mi_rows;
  frame.subsampling_x = 1;
  frame.subsampling_y = 1;
  frame.bit_depth = 8;

  LoopRestorationParams lp;
  lp.frame_type[0] = kRestoreWienerNs;
  lp.frame_type[1] = kRestoreNone;
  lp.frame_type[2] = kRestoreNone;
  lp.unit_size[0] = 64;
  lp.unit_size[1] = 32;
  lp.unit_size[2] = 32;
  lp.unit_rows[0] = 1;
  lp.unit_cols[0] = 1;
  std::vector<std::uint8_t> type(1, kRestoreWienerNs);
  std::vector<std::int16_t> unit_coeffs(18, 0);
  lp.unit_type[0] = type.data();
  lp.unit_coeffs[0] = unit_coeffs.data();

  const int cells = frame.mi_cols * frame.mi_rows;
  std::vector<std::uint8_t> filter_class(static_cast<std::size_t>(cells));
  std::vector<std::uint8_t> subclass(static_cast<std::size_t>(cells));
  std::vector<std::uint8_t> lookup(256);
  for (int i = 0; i < cells; ++i) {
    filter_class[static_cast<std::size_t>(i)] =
        static_cast<std::uint8_t>((i * 37) & 255);
    subclass[static_cast<std::size_t>(i)] =
        static_cast<std::uint8_t>(i & 1);
  }
  for (int i = 0; i < 256; ++i) {
    lookup[static_cast<std::size_t>(i)] =
        static_cast<std::uint8_t>(i & 1);
  }
  std::int16_t frame_coeffs[kWienerNsClasses][kWienerNsCoeffStride] = {};
  frame_coeffs[0][0] = 0;
  frame_coeffs[1][0] = 16;  // tap (1,0), a strong vertical high-pass
  lp.frame_filters_on[0] = 1;
  lp.num_filter_classes = 2;
  lp.filter_class_count = 256;
  lp.filter_class_grid = filter_class.data();
  lp.subclass_lookup = lookup.data();
  lp.frame_coeffs[0] = &frame_coeffs[0][0];

  const RestorationStatus st = loop_restoration_frame(frame, lp);
  CHECK(st == RestorationStatus::kOk, "256-class frame filter accepted");
  int changed = 0;
  int unchanged_class0 = 0;
  for (int row = 0; row < frame.mi_rows; ++row) {
    for (int col = 0; col < frame.mi_cols; ++col) {
      const int idx = row * frame.mi_cols + col;
      const int cls = subclass[static_cast<std::size_t>(idx)];
      for (int dy = 0; dy < 4; ++dy) {
        for (int dx = 0; dx < 4; ++dx) {
          const int y = row * 4 + dy;
          const int x = col * 4 + dx;
          const int a = t.lr[0][static_cast<std::size_t>(y) *
                              static_cast<std::size_t>(t.stride[0]) +
                          static_cast<std::size_t>(x)];
          const int b = t.cdef[0][static_cast<std::size_t>(y) *
                                static_cast<std::size_t>(t.stride[0]) +
                            static_cast<std::size_t>(x)];
          if (cls == 0) {
            if (a == b) {
              ++unchanged_class0;
            }
          } else if (a != b) {
            ++changed;
          }
        }
      }
    }
  }
  CHECK(changed > 0, "class 1 uses its own filter");
  CHECK(unchanged_class0 == (cells / 2) * 16,
        "class 0 uses the zero/identity filter");
}

void test_decision_mixed_unit_sizes() {
  DecisionFixture f = make_fixture(make_synthetic(64, 64));
  WienerDecisionInput in = make_input(f);
  WienerDecisionOptions opts;
  opts.max_num_filter_classes = 1;
  opts.lambda_bits = 0.0;
  opts.forced_luma_shift = 0;    // 512-sample luma units
  opts.forced_chroma_shift = 1;  // 128-sample chroma units
  WienerDecision d;
  const WienerDecisionStatus st = wiener_decide(in, opts, &d);
  CHECK(st == WienerDecisionStatus::kOk, "mixed-size decision succeeds");
  CHECK(d.unit_size[0] == 512, "forced luma unit size");
  CHECK(d.unit_size[1] == 128, "forced chroma unit size");
  CHECK(d.unit_size[2] == d.unit_size[1], "LoopRestorationSize[2] == [1]");
  check_decision_roundtrip(&f, d, "mixed");

  // All chroma planes off, luma at a non-default shift.
  DecisionFixture f2 = make_fixture(make_synthetic(64, 64));
  WienerDecisionInput in2 = make_input(f2);
  WienerDecisionOptions opts2 = opts;
  opts2.allow_chroma = false;
  opts2.forced_luma_shift = 2;
  WienerDecision d2;
  CHECK(wiener_decide(in2, opts2, &d2) == WienerDecisionStatus::kOk,
        "luma-only decision succeeds");
  CHECK(d2.params.frame_type[1] == kRestoreNone, "chroma 1 forced off");
  CHECK(d2.params.frame_type[2] == kRestoreNone, "chroma 2 forced off");
  check_decision_roundtrip(&f2, d2, "luma-only");
}

void test_decision_multi_class_synthetic() {
  DecisionFixture f = make_fixture(make_synthetic(64, 64));
  WienerDecisionInput in = make_input(f);
  std::vector<std::uint8_t> lut(4096);
  for (int i = 0; i < 4096; ++i) {
    lut[static_cast<std::size_t>(i)] =
        static_cast<std::uint8_t>((i * 7) & 255);
  }
  std::vector<std::uint8_t> sub2(4 * 8 * 256);
  for (int set = 0; set < 4; ++set) {
    for (int idx = 0; idx < 8; ++idx) {
      for (int cls = 0; cls < 256; ++cls) {
        sub2[static_cast<std::size_t>((set * 8 + idx) * 256 + cls)] =
            static_cast<std::uint8_t>(cls % decode_num_filter_classes(idx));
      }
    }
  }
  PcWienerTables tables;
  tables.lut_to_class = lut.data();
  tables.sub_classify2 = sub2.data();
  std::vector<std::uint8_t> tx_skip(
      static_cast<std::size_t>(f.core.core.mi_cols) *
          static_cast<std::size_t>(f.core.core.mi_rows),
      0);
  in.tables = &tables;
  in.lr_tx_skip = tx_skip.data();

  WienerDecisionOptions opts;
  opts.allow_per_unit = false;
  opts.allow_frame_filters = true;
  opts.max_num_filter_classes = 4;
  opts.lambda_bits = 0.0;
  WienerDecision d;
  const WienerDecisionStatus st = wiener_decide(in, opts, &d);
  CHECK(st == WienerDecisionStatus::kOk, "multi-class decision succeeds");
  CHECK(d.num_filter_classes >= 1, "multi-class decision class count");
  if (d.num_filter_classes > 1) {
    CHECK(d.filter_class_count == 256, "256-entry FilterClass space");
    CHECK(d.subclass_lookup.size() == 256, "256-entry subclass lookup");
    CHECK(d.subclass_grid.size() ==
              static_cast<std::size_t>(f.core.core.mi_cols) *
                  static_cast<std::size_t>(f.core.core.mi_rows),
          "subclass grid covers the luma MI grid");
  }
  check_decision_roundtrip(&f, d, "multi-class");
}

void test_decision_missing_multi_class() {
  DecisionFixture f = make_fixture(make_synthetic(64, 64));
  WienerDecisionInput in = make_input(f);
  WienerDecisionOptions opts;
  opts.max_num_filter_classes = 4;
  opts.lambda_bits = 0.0;
  WienerDecision d;
  const WienerDecisionStatus st = wiener_decide(in, opts, &d);
  CHECK(st == WienerDecisionStatus::kOk,
        "multi-class request degrades to a conformant single-class decision");
  CHECK(d.num_filter_classes == 1,
        "no multi-class filter without Pc_Wiener_* tables");
  bool noted = false;
  for (const std::string& note : d.notes) {
    if (note.find("Pc_Wiener") != std::string::npos) {
      noted = true;
    }
  }
  CHECK(noted, "missing-table reason is recorded");
}

}  // namespace

int main() {
  test_fit_identity_and_blur();
  test_fit_chroma_uv_sym();
  test_missing_class_tables();
  test_class_derivation_synthetic_tables();
  test_class_grid_256_extension();
  test_decision_synthetic();
  test_decision_frame_filters();
  test_decision_mixed_unit_sizes();
  test_decision_multi_class_synthetic();
  test_decision_real_recon();
  test_decision_missing_multi_class();
  std::printf("wiener_classes_test: %d checks, %d failures\n", g_checks,
              g_failures);
  return g_failures == 0 ? 0 : 1;
}

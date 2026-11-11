// Round-trip and decision tests for the M2 advanced-inter syntax layer.
//
// Every symbol test encodes with av2::Encoder and decodes with av2::Decoder —
// the normative §8.2 state machine used by the decoder — so a passing test
// proves the writer is the exact inverse of the syntax, including CDF
// adaptation.  No private coder is involved.

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "advanced_decision.h"
#include "advanced_syntax.h"
#include "entropy/av2/av2_entropy.h"
#include "encoder/inter/advanced/tables/wedge_tables.h"

namespace {

namespace av2 = av2enc::av2;
using namespace av2enc::m2::adv;

int g_checks = 0;
int g_failures = 0;

void expect(bool condition, const char* what, int line) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    if (g_failures <= 40) std::printf("FAIL line %d: %s\n", line, what);
  }
}

#define EXPECT(cond) expect((cond), #cond, __LINE__)
#define EXPECT_EQ(a, b)                                                              \
  do {                                                                               \
    const long long va = static_cast<long long>(a);                                  \
    const long long vb = static_cast<long long>(b);                                  \
    ++g_checks;                                                                      \
    if (va != vb) {                                                                  \
      ++g_failures;                                                                  \
      if (g_failures <= 40)                                                          \
        std::printf("FAIL line %d: %s (%lld) != %s (%lld)\n", __LINE__, #a, va, #b,  \
                    vb);                                                             \
    }                                                                                \
  } while (0)

std::uint32_t g_rng = 0x2545f491u;
std::uint32_t next_random() {
  g_rng ^= g_rng << 13;
  g_rng ^= g_rng >> 17;
  g_rng ^= g_rng << 5;
  return g_rng;
}
int random_range(int low, int high) {
  return low + static_cast<int>(next_random() %
                                static_cast<std::uint32_t>(high - low + 1));
}

bool cdfs_equal(const AdvancedCdfState& a, const AdvancedCdfState& b) {
  return std::memcmp(&a, &b, sizeof(a)) == 0;
}

// Encode `write(enc, cdfs, err)`, then decode `read(dec, cdfs, err)` with a
// fresh CDF state.  CDF adaptation stays enabled so the final states must also
// match.  Returns the payload size.
template <typename WriteFn, typename ReadFn>
bool round_trip(WriteFn write, ReadFn read, std::string* error,
                std::size_t* payload_bytes = nullptr) {
  std::vector<std::uint8_t> buffer(1u << 20);
  av2::Encoder encoder;
  av2::Encoder* enc = &encoder;
  if (enc->init_symbol(buffer.data(), buffer.size()) != av2::Status::kOk) {
    *error = "encoder init failed";
    return false;
  }
  AdvancedCdfState enc_cdfs;
  enc_cdfs.reset();
  if (!write(*enc, enc_cdfs, error)) return false;
  std::size_t bytes = 0;
  if (enc->exit_symbol(&bytes) != av2::Status::kOk) {
    *error = "encoder exit failed";
    return false;
  }

  av2::Decoder decoder;
  av2::Decoder* dec = &decoder;
  if (dec->init_symbol(buffer.data(), buffer.size(), bytes) !=
      av2::Status::kOk) {
    *error = "decoder init failed";
    return false;
  }
  AdvancedCdfState dec_cdfs;
  dec_cdfs.reset();
  if (!read(*dec, dec_cdfs, error)) return false;
  if (dec->exit_symbol() != av2::Status::kOk) {
    *error = "decoder exit_symbol rejected the trailing bits";
    return false;
  }
  if (!cdfs_equal(enc_cdfs, dec_cdfs)) {
    *error = "encoder/decoder CDF state diverged";
    return false;
  }
  if (payload_bytes != nullptr) *payload_bytes = bytes;
  return true;
}

// ---------------------------------------------------------------------------
// CDF tables
// ---------------------------------------------------------------------------

void test_cdf_defaults() {
  AdvancedCdfState a;
  AdvancedCdfState b;
  a.reset();
  b.reset();
  EXPECT(cdfs_equal(a, b));

  // Spot-check the generated §9.3 values against the spec table.
  EXPECT_EQ(cdf::kIsWarpCdf[0][0], 25999);
  EXPECT_EQ(cdf::kWarpPrecisionCdf[28][0], 18893);
  EXPECT_EQ(cdf::kWarpDeltaParamLowCdf[1][8], 0);
  EXPECT_EQ(cdf::kInterIntraModeCdf[3][0], 3564);
  EXPECT_EQ(cdf::kWedgeAngleCdf[2][1], 22099);
  EXPECT_EQ(cdf::kUseBawpCdf[0], 26456);
  EXPECT_EQ(cdf::kExplicitBawpCdf[1][0], 15275);
  EXPECT_EQ(cdf::kSizeGroup[12], 3);
  EXPECT_EQ(cdf::kSizeGroup[23], 2);
}

// ---------------------------------------------------------------------------
// Wedge mode mapping and weights
// ---------------------------------------------------------------------------

void test_wedge_mapping() {
  for (int index = 0; index < wt::kWedgeTypes; ++index) {
    WedgeModeSymbols s{};
    EXPECT(wedge_mode_symbols(index, &s));
    int decoded = -1;
    EXPECT(wedge_index_from_symbols(s, &decoded));
    EXPECT_EQ(decoded, index);
    const int angle = s.wedge_quad * 5 + s.wedge_angle;
    const bool expect_dist2 =
        (angle >= 10) || (angle == 5) || (angle == 0);
    EXPECT_EQ(s.use_dist2, expect_dist2);
    if (s.use_dist2) {
      EXPECT(s.wedge_dist2 >= 0 && s.wedge_dist2 <= 2);
    } else {
      EXPECT(s.wedge_dist1 >= 0 && s.wedge_dist1 <= 3);
    }
  }
  WedgeModeSymbols unused{};
  EXPECT(!wedge_mode_symbols(-1, &unused));
  EXPECT(!wedge_mode_symbols(68, &unused));
  EXPECT(!wedge_mode_symbols(127, &unused));

  // Round-trip every index through the normative entropy coder.
  for (int index = 0; index < wt::kWedgeTypes; ++index) {
    for (int sign = 0; sign < 2; ++sign) {
      WedgeModeSymbols in{};
      EXPECT(wedge_mode_symbols(index, &in));
      std::string error;
      auto write = [&](av2::Encoder& enc, AdvancedCdfState& cdfs,
                       std::string* err) {
        return write_wedge_mode(enc, cdfs, in, err) &&
               write_wedge_sign(enc, sign != 0, err);
      };
      auto read = [&](av2::Decoder& dec, AdvancedCdfState& cdfs,
                      std::string* err) {
        WedgeModeSymbols out{};
        int decoded_sign = -1;
        return read_wedge_mode(dec, cdfs, &out, err) &&
               read_wedge_sign(dec, &decoded_sign, err) &&
               (out.wedge_index == index) && (decoded_sign == sign);
      };
      EXPECT(round_trip(write, read, &error));
    }
  }
}

void test_wedge_weights() {
  const int blocks[] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 21, 22, 23, 24, 25, 26, 27, 28};
  for (int bsize : blocks) {
    if (!ia::bsize_has_wedge(bsize)) continue;
    const int w = av2enc::partition::block_width(
        static_cast<av2enc::partition::BlockSize>(bsize));
    const int h = av2enc::partition::block_height(
        static_cast<av2enc::partition::BlockSize>(bsize));
    std::vector<std::uint8_t> mask(static_cast<std::size_t>(w) * h);
    for (int index = 0; index < wt::kWedgeTypes; ++index) {
      for (int sign = 0; sign < 2; ++sign) {
        EXPECT(build_wedge_weights(bsize, sign, index, mask.data(), w));
        for (int i = 0; i < h; ++i) {
          for (int j = 0; j < w; ++j) {
            const int expected =
                ia::wedge_mask_value(bsize, sign, index, i, j);
            EXPECT_EQ(mask[static_cast<std::size_t>(i) * w + j], expected);
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Warp symbols
// ---------------------------------------------------------------------------

void test_warp_symbols() {
  std::string error;
  auto write = [&](av2::Encoder& enc, AdvancedCdfState& cdfs, std::string* err) {
    if (!write_is_warp(enc, cdfs, 0, false, err)) return false;
    if (!write_is_warp(enc, cdfs, 4, true, err)) return false;
    if (!write_warp_mv(enc, cdfs, true, err)) return false;
    if (!write_warp_idx(enc, cdfs, 0, err)) return false;
    if (!write_warp_idx(enc, cdfs, 2, err)) return false;
    if (!write_warp_idx(enc, cdfs, 3, err)) return false;
    if (!write_warpmv_with_mvd(enc, cdfs, true, err)) return false;
    return true;
  };
  auto read = [&](av2::Decoder& dec, AdvancedCdfState& cdfs, std::string* err) {
    bool warp = true;
    bool warp_mv = false;
    bool with_mvd = false;
    int idx = -1;
    return read_is_warp(dec, cdfs, 0, &warp, err) && !warp &&
           read_is_warp(dec, cdfs, 4, &warp, err) && warp &&
           read_warp_mv(dec, cdfs, &warp_mv, err) && warp_mv &&
           read_warp_idx(dec, cdfs, &idx, err) && idx == 0 &&
           read_warp_idx(dec, cdfs, &idx, err) && idx == 2 &&
           read_warp_idx(dec, cdfs, &idx, err) && idx == 3 &&
           read_warpmv_with_mvd(dec, cdfs, &with_mvd, err) && with_mvd;
  };
  EXPECT(round_trip(write, read, &error));

  // Context bounds.
  av2::Encoder enc;
  std::vector<std::uint8_t> buffer(1024);
  EXPECT(enc.init_symbol(buffer.data(), buffer.size()) == av2::Status::kOk);
  AdvancedCdfState cdfs;
  cdfs.reset();
  EXPECT(!write_is_warp(enc, cdfs, 5, true, &error));
  EXPECT(!write_warp_idx(enc, cdfs, 4, &error));
}

void test_warp_delta_symbols() {
  for (int high = 0; high <= 1; ++high) {
    for (int six = 0; six <= 1; ++six) {
      WarpDeltaSymbols symbols{};
      symbols.high_precision = high != 0;
      symbols.magnitude[0] = high ? 14 : 7;
      symbols.negative[0] = true;
      symbols.magnitude[1] = 1;
      symbols.magnitude[2] = six ? (high ? 13 : 7) : 0;
      symbols.negative[2] = six != 0;
      symbols.magnitude[3] = six ? 2 : 0;

      std::string error;
      auto write = [&](av2::Encoder& enc, AdvancedCdfState& cdfs,
                       std::string* err) {
        return write_warp_delta(enc, cdfs, 3, symbols, six != 0, err);
      };
      auto read = [&](av2::Decoder& dec, AdvancedCdfState& cdfs,
                      std::string* err) {
        WarpDeltaSymbols out{};
        if (!read_warp_delta(dec, cdfs, 3, six != 0, &out, err)) return false;
        if (out.high_precision != (high != 0)) return false;
        const int count = six ? 4 : 2;
        for (int i = 0; i < count; ++i) {
          if (out.magnitude[i] != symbols.magnitude[i]) return false;
          if (out.negative[i] != symbols.negative[i]) return false;
        }
        return true;
      };
      EXPECT(round_trip(write, read, &error));
    }
  }
}

void test_warp_delta_quantization() {
  ia::WarpParams base{};
  base.p[2] = 65536 + 1024;
  base.p[3] = 0;
  base.p[4] = 0;
  base.p[5] = 65536 + 1024;

  for (int six = 0; six <= 1; ++six) {
    ia::WarpParams target = base;
    target.p[2] += six ? 4096 : 2048;
    target.p[3] -= 2048;
    if (six) {
      target.p[4] += 1024;
      target.p[5] -= 2048;
    } else {
      target.p[4] = -target.p[3];
      target.p[5] = target.p[2];
    }
    for (int high = 0; high <= 1; ++high) {
      const int mv[2] = {0, 0};
      const WarpDeltaQuantization q =
          quantize_warp_delta(base, target, high != 0, six != 0, 32, 32, mv);
      const ia::WarpParams decoded =
          apply_warp_delta(base, q.symbols, six != 0, 32, 32, mv);
      for (int i = 0; i < 6; ++i) EXPECT_EQ(decoded.p[i], q.model.p[i]);
    }

    const int mv[2] = {0, 0};
    WarpDeltaChoice choice =
        choose_warp_delta_precision(base, target, six != 0, 32, 32, mv);
    // A 1024-unit delta needs high precision; 2048 is exact in standard.
    EXPECT_EQ(choice.high_precision, six != 0);
    EXPECT(choice.quantization.param_error == 0);
  }

  // Non-representable target: the quantiser still produces a legal symbol set.
  ia::WarpParams far = base;
  far.p[2] += 100000;
  const int mv[2] = {1, -1};
  const WarpDeltaQuantization q =
      quantize_warp_delta(base, far, true, false, 0, 0, mv);
  EXPECT(q.symbols.magnitude[0] <= 14);
  EXPECT(q.symbols.high_precision);
}

void test_warp_decoding_reconstruction() {
  // Encoder quantisation + decoder reconstruction must be bit-exact through
  // the syntax symbols.
  ia::WarpParams base{};
  base.p[2] = 65536;
  base.p[3] = 0;
  base.p[4] = 0;
  base.p[5] = 65536;
  for (int n = 0; n < 24; ++n) {
    ia::WarpParams target = base;
    target.p[2] += random_range(-3000, 3000);
    target.p[3] += random_range(-3000, 3000);
    target.p[4] = -target.p[3];
    target.p[5] = target.p[2];
    const int mv[2] = {random_range(-8, 8), random_range(-8, 8)};
    const int mid_x = random_range(0, 64);
    const int mid_y = random_range(0, 64);
    const WarpDeltaChoice choice =
        choose_warp_delta_precision(base, target, false, mid_x, mid_y, mv);

    std::string error;
    auto write = [&](av2::Encoder& enc, AdvancedCdfState& cdfs,
                     std::string* err) {
      return write_warp_delta(enc, cdfs, 12, choice.quantization.symbols,
                              false, err);
    };
    WarpDeltaSymbols decoded{};
    auto read = [&](av2::Decoder& dec, AdvancedCdfState& cdfs,
                    std::string* err) {
      return read_warp_delta(dec, cdfs, 12, false, &decoded, err);
    };
    EXPECT(round_trip(write, read, &error));
    const ia::WarpParams model =
        apply_warp_delta(base, decoded, false, mid_x, mid_y, mv);
    for (int i = 0; i < 6; ++i) EXPECT_EQ(model.p[i], choice.quantization.model.p[i]);
  }
}

// ---------------------------------------------------------------------------
// §7.11.2 neighbour contexts
// ---------------------------------------------------------------------------

ModeContext literal_context(const NeighborGrid& g, int mi_row, int mi_col,
                            int bsize, int sb, int ref0, int ref1) {
  ModeContext out{};
  const int bw4 = av2enc::partition::num_4x4_blocks_wide(
      static_cast<av2enc::partition::BlockSize>(bsize));
  const int bh4 = av2enc::partition::num_4x4_blocks_high(
      static_cast<av2enc::partition::BlockSize>(bsize));
  const int is_sb_border = (mi_row & (sb - 1)) == 0 ? 1 : 0;
  const int dr[4] = {bh4 - 1, -1, 0, -1};
  const int dc[4] = {-1, is_sb_border ? std::max(0, bw4 - 2) : bw4 - 1, -1, 0};
  const int count = bw4 >= (is_sb_border ? 4 : 2) ? 4 : 3;
  for (int n = 0; n < count; ++n) {
    int delta_row = dr[n];
    int delta_col = dc[n];
    if (delta_row < 0 && is_sb_border) delta_col -= mi_col & 1;
    const int r = mi_row + delta_row;
    const int c = mi_col + delta_col;
    if (!g.inside(r, c)) continue;
    const NeighborMi& mi = g.at(r, c);
    if (!mi.written || !mi.is_inter) continue;
    if (mi.ref[0] == ref0 || mi.ref[1] == ref0) {
      if (!out.warp_sample_found[0]) {
        out.extend_delta_row = delta_row;
        out.extend_delta_col = delta_col;
      }
      out.warp_sample_found[0] = true;
      if (mi.motion_mode >= static_cast<int>(ia::MotionMode::kLocalWarp)) {
        ++out.warp_mv_count;
      }
    }
    if (mi.ref[0] == ref1 || mi.ref[1] == ref1) out.warp_sample_found[1] = true;
  }

  const int nr[4] = {mi_row + bh4 - 1, mi_row - 1, mi_row, mi_row - 1};
  const int nc[4] = {mi_col - 1, mi_col + bw4 - 1, mi_col - 1, mi_col};
  for (int n = 0; n < 4; ++n) {
    if (out.npos_count >= 2) break;
    if (!g.inside(nr[n], nc[n])) continue;
    if ((mi_row / sb) != (nr[n] / sb)) continue;
    out.npos[out.npos_count][0] = nr[n];
    out.npos[out.npos_count][1] = nc[n];
    ++out.npos_count;
  }
  int has_warp = 0;
  for (int n = 0; n < out.npos_count; ++n) {
    const int mode = g.at(out.npos[n][0], out.npos[n][1]).motion_mode;
    if (mode >= static_cast<int>(ia::MotionMode::kLocalWarp)) {
      has_warp = 1;
      ++out.extend_warp_ctx;
    }
    if (mode == static_cast<int>(ia::MotionMode::kLocalWarp)) ++out.local_warp_ctx;
  }
  out.local_warp_ctx += has_warp;
  out.local_warp_ctx_npos = out.local_warp_ctx;
  out.extend_warp_ctx_npos = out.extend_warp_ctx;
  return out;
}

void test_warp_contexts() {
  const int cols = 32;
  const int rows = 32;
  std::vector<NeighborMi> cells(static_cast<std::size_t>(cols) * rows);
  NeighborGrid grid{cols, rows, cells.data()};
  const int bsize = static_cast<int>(av2enc::partition::BlockSize::k16x16);
  for (int trial = 0; trial < 200; ++trial) {
    for (auto& cell : cells) {
      cell = NeighborMi{};
      cell.written = (next_random() & 3) != 0;
      cell.is_inter = (next_random() & 1) != 0;
      cell.ref[0] = random_range(-1, 3);
      cell.ref[1] = (next_random() & 1) ? random_range(-1, 3) : -1;
      cell.motion_mode = random_range(0, 4);
      cell.y_mode = random_range(0, 24);
    }
    const int mi_row = random_range(0, rows - 1);
    const int mi_col = random_range(0, cols - 1);
    const int ref0 = random_range(0, 3);
    const int ref1 = (next_random() & 1) ? random_range(0, 3) : -1;
    const ModeContext got =
        compute_mode_context(grid, mi_row, mi_col, bsize, 16, ref0, ref1);
    const ModeContext want =
        literal_context(grid, mi_row, mi_col, bsize, 16, ref0, ref1);
    EXPECT_EQ(got.warp_mv_count, want.warp_mv_count);
    EXPECT_EQ(got.warp_sample_found[0], want.warp_sample_found[0]);
    EXPECT_EQ(got.warp_sample_found[1], want.warp_sample_found[1]);
    EXPECT_EQ(got.extend_delta_row, want.extend_delta_row);
    EXPECT_EQ(got.extend_delta_col, want.extend_delta_col);
    EXPECT_EQ(got.npos_count, want.npos_count);
    EXPECT_EQ(got.local_warp_ctx, want.local_warp_ctx);
    EXPECT_EQ(got.extend_warp_ctx, want.extend_warp_ctx);
  }

  // A concrete case: left A has a matching local warp, above-right has ref 1.
  for (auto& cell : cells) {
    cell = NeighborMi{};
    cell.written = true;
    cell.is_inter = true;
  }
  const int mi_row = 8;
  const int mi_col = 8;
  NeighborMi& left = cells[static_cast<std::size_t>(mi_row + 3) * cols + (mi_col - 1)];
  left.ref[0] = 0;
  left.motion_mode = static_cast<int>(ia::MotionMode::kLocalWarp);
  NeighborMi& above_right = cells[static_cast<std::size_t>(mi_row - 1) * cols + (mi_col + 3)];
  above_right.ref[0] = 1;
  above_right.motion_mode = static_cast<int>(ia::MotionMode::kSimple);
  const ModeContext ctx =
      compute_mode_context(grid, mi_row, mi_col, bsize, 16, 0, 1);
  EXPECT_EQ(ctx.warp_mv_count, 1);
  EXPECT(ctx.warp_sample_found[0]);
  EXPECT(ctx.warp_sample_found[1]);
  EXPECT_EQ(ctx.npos_count, 2);
  EXPECT_EQ(ctx.npos[0][0], mi_row + 3);
  EXPECT_EQ(ctx.npos[0][1], mi_col - 1);
  EXPECT_EQ(ctx.npos[1][0], mi_row - 1);
  EXPECT_EQ(ctx.npos[1][1], mi_col + 3);
  EXPECT_EQ(ctx.local_warp_ctx, 2);  // one LOCALWARP + hasWarp
  EXPECT_EQ(ctx.extend_warp_ctx, 1);
}

// ---------------------------------------------------------------------------
// Warp decision
// ---------------------------------------------------------------------------

ia::WarpBlockGeometry make_geom(int x, int y, int w, int h) {
  ia::WarpBlockGeometry geom{};
  geom.plane = 0;
  geom.rect = ia::BlockRect{x, y, w, h};
  geom.dst_x = 0;
  geom.dst_y = 0;
  geom.mi_col = x / 4;
  geom.mi_row = y / 4;
  geom.bw4 = std::max(1, w / 4);
  geom.bh4 = std::max(1, h / 4);
  geom.mi_cols = 64;
  geom.mi_rows = 64;
  geom.ref_w = 256;
  geom.ref_h = 256;
  geom.frame_w = 256;
  geom.frame_h = 256;
  geom.bit_depth = 8;
  return geom;
}

ia::PlaneView make_plane(std::vector<std::uint8_t>& data, int w, int h,
                         int stride) {
  data.assign(static_cast<std::size_t>(stride) * h, 0);
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      data[static_cast<std::size_t>(i) * stride + j] =
          static_cast<std::uint8_t>((i * 7 + j * 11 + 13) & 255);
    }
  }
  return ia::PlaneView{data.data(), stride, w - 1, h - 1};
}

void test_warp_decision() {
  const int w = 16;
  const int h = 16;
  std::vector<std::uint8_t> ref_storage;
  ia::PlaneView ref = make_plane(ref_storage, 128, 128, 128);

  ia::WarpParams identity{};
  ia::WarpParams affine{};
  affine.p[2] = 65536 + 1024;
  affine.p[5] = 65536 + 1024;
  const int zero_mv[2] = {0, 0};
  ia::get_warp_translation(affine, 16 + w / 2, 16 + h / 2, zero_mv);

  const ia::WarpBlockGeometry geom = make_geom(16, 16, w, h);
  std::int16_t scratch[64 * 64];
  std::vector<std::uint8_t> source(static_cast<std::size_t>(w) * h);

  // Source = the affine prediction, so the affine model is the best fit.
  EXPECT(ia::warp_predict(affine, ref, geom, false, scratch, w));
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      source[static_cast<std::size_t>(i) * w + j] =
          static_cast<std::uint8_t>(scratch[i * w + j]);
    }
  }

  const ia::WarpParams candidates[2] = {identity, affine};
  const int mv[2] = {0, 0};
  WarpSearchInput in{};
  in.candidates = candidates;
  in.num_candidates = 2;
  in.ref = ref;
  in.geom = geom;
  in.source = source.data();
  in.source_stride = w;
  in.scratch = scratch;
  in.scratch_stride = w;
  in.mid_x = 16 + w / 2;
  in.mid_y = 16 + h / 2;
  in.block_mv = mv;
  in.enable_six_param = true;

  WarpCodingDecision direct = select_warp_coding(in);
  EXPECT(direct.use_warp);
  EXPECT(direct.warpmv);
  EXPECT_EQ(direct.warp_idx, 1);
  EXPECT(direct.sse < 8);
  const ia::WarpParams local =
      warpmv_local_model(affine, in.mid_x, in.mid_y, mv);
  for (int i = 0; i < 6; ++i) EXPECT_EQ(direct.model.p[i], local.p[i]);
  EXPECT_EQ(default_wm_params().p[2], 65536);
  EXPECT_EQ(reduce_wm_params(affine).p[2], affine.p[2]);

  // Remove the affine candidate: the encoder must code a delta from identity.
  in.num_candidates = 1;
  in.target = &affine;
  WarpCodingDecision delta = select_warp_coding(in);
  EXPECT(delta.use_warp);
  EXPECT(!delta.warpmv);
  EXPECT_EQ(delta.warp_idx, 0);
  EXPECT(!delta.six_param);
  EXPECT(delta.high_precision);  // 1024 needs the high-precision step
  EXPECT(delta.sse < 8);
  const ia::WarpParams rebuilt =
      apply_warp_delta(identity, delta.delta, delta.six_param, in.mid_x, in.mid_y,
                       mv);
  for (int i = 0; i < 6; ++i) EXPECT_EQ(rebuilt.p[i], delta.model.p[i]);
}

// ---------------------------------------------------------------------------
// Inter-intra
// ---------------------------------------------------------------------------

void test_inter_intra_symbols() {
  for (int is_warp = 0; is_warp <= 1; ++is_warp) {
    for (int mi_size = 0; mi_size < kBlockSizes; ++mi_size) {
      const int w = av2enc::partition::block_width(
          static_cast<av2enc::partition::BlockSize>(mi_size));
      const int h = av2enc::partition::block_height(
          static_cast<av2enc::partition::BlockSize>(mi_size));
      if (is_warp && (w > 64 || h > 64)) continue;
      for (int mode = 0; mode < 4; ++mode) {
        for (int wedge = 0; wedge <= 1; ++wedge) {
          if (wedge && kWedgeBits[mi_size] == 0) continue;
          MotionModeSymbols symbols{};
          symbols.inter_intra_allowed = true;
          symbols.inter_intra = true;
          symbols.interintra_mode = mode;
          symbols.wedge_interintra = wedge != 0;
          symbols.wedge_index = wedge ? (mi_size * 7 + mode) % 68 : 0;
          std::string error;
          auto write = [&](av2::Encoder& enc, AdvancedCdfState& cdfs,
                           std::string* err) {
            return write_inter_intra(enc, cdfs, is_warp != 0, mi_size, symbols,
                                     err);
          };
          auto read = [&](av2::Decoder& dec, AdvancedCdfState& cdfs,
                          std::string* err) {
            bool enabled = false;
            bool decoded_wedge = false;
            int decoded_mode = -1;
            int decoded_index = -1;
            if (!read_inter_intra(dec, cdfs, is_warp != 0, mi_size, true,
                                  &enabled, &decoded_mode, &decoded_wedge,
                                  &decoded_index, err)) {
              return false;
            }
            return enabled && decoded_mode == mode &&
                   decoded_wedge == (wedge != 0) &&
                   (!wedge || decoded_index == symbols.wedge_index);
          };
          EXPECT(round_trip(write, read, &error));
        }
      }
    }
  }
}

void test_motion_mode_symbols() {
  std::string error;
  MotionModeSymbols s{};
  s.is_warp_newmv = true;
  s.extend_allowed = true;
  s.local_allowed = true;
  s.use_extend_warp = false;
  s.use_local_warp = true;
  s.extend_warp_ctx = 2;
  s.local_warp_ctx = 3;
  auto write = [&](av2::Encoder& enc, AdvancedCdfState& cdfs, std::string* err) {
    return write_motion_mode(enc, cdfs, 12, s, err);
  };
  auto read = [&](av2::Decoder& dec, AdvancedCdfState& cdfs, std::string* err) {
    MotionModeSymbols out{};
    if (!read_motion_mode(dec, cdfs, kSizeGroup[12], true, true, true, 2, 3,
                          false, false, &out, err)) {
      return false;
    }
    return out.is_warp_newmv && !out.use_extend_warp && out.use_local_warp;
  };
  EXPECT(round_trip(write, read, &error));

  s = MotionModeSymbols{};
  s.inter_intra_allowed = true;
  s.inter_intra = true;
  s.interintra_mode = 3;
  s.wedge_allowed = true;
  s.wedge_interintra = true;
  s.wedge_index = 12;
  s.local_allowed = true;
  s.use_local_warp = false;
  auto write2 = [&](av2::Encoder& enc, AdvancedCdfState& cdfs, std::string* err) {
    return write_motion_mode(enc, cdfs, 12, s, err);
  };
  auto read2 = [&](av2::Decoder& dec, AdvancedCdfState& cdfs, std::string* err) {
    MotionModeSymbols out{};
    return read_motion_mode(dec, cdfs, kSizeGroup[12], false, false, true, 0, 0,
                            true, true, &out, err) &&
           out.inter_intra && out.interintra_mode == 3 &&
           out.wedge_interintra && out.wedge_index == 12 && !out.use_local_warp;
  };
  EXPECT(round_trip(write2, read2, &error));
}

void test_inter_intra_decision_and_weights() {
  const int w = 16;
  const int h = 16;
  std::vector<std::uint8_t> source(static_cast<std::size_t>(w) * h);
  std::vector<std::uint8_t> inter(static_cast<std::size_t>(w) * h);
  std::vector<std::uint8_t> intra[4];
  for (int m = 0; m < 4; ++m) intra[m].resize(static_cast<std::size_t>(w) * h);
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      const std::size_t p = static_cast<std::size_t>(i) * w + j;
      source[p] = static_cast<std::uint8_t>((i * 3 + j * 5) & 255);
      inter[p] = static_cast<std::uint8_t>((source[p] + 16) & 255);
      for (int m = 0; m < 4; ++m) {
        intra[m][p] = m == 3 ? source[p]
                             : static_cast<std::uint8_t>((source[p] + m) & 255);
      }
    }
  }
  std::vector<std::uint8_t> scratch(inter);

  ia::InterIntraSearchInput in{};
  in.bsize = static_cast<int>(av2enc::partition::BlockSize::k16x16);
  in.rect = ia::BlockRect{0, 0, w, h};
  for (int m = 0; m < 4; ++m) in.intra[m] = intra[m].data();
  in.intra_stride = w;
  in.scratch = scratch.data();
  in.scratch_stride = w;
  in.source = source.data();
  in.source_stride = w;
  in.inter_pred = inter.data();
  in.inter_pred_stride = w;
  in.enable_wedge_interintra = false;
  in.wedge_sign = 0;

  InterIntraCodingInput ci{};
  ci.search = in;
  ci.inter_sse = ia::block_sse(source.data(), w, 0, 0, inter.data(), w, 0, 0, w,
                               h);
  const InterIntraCodingDecision decision = decide_inter_intra(ci);
  const auto expected = ia::search_inter_intra(in);
  EXPECT(expected.has_value());
  EXPECT(decision.enabled);
  EXPECT_EQ(decision.interintra_mode, static_cast<int>(expected->mode));
  EXPECT_EQ(decision.wedge_interintra, expected->wedge_interintra);
  EXPECT_EQ(decision.wedge_index, expected->wedge_index);
  EXPECT_EQ(decision.wedge_sign, 0);
  EXPECT_EQ(decision.sse, expected->sse);

  MotionModeSymbols symbols{};
  fill_inter_intra_symbols(decision, &symbols);
  EXPECT(symbols.inter_intra);
  EXPECT_EQ(symbols.interintra_mode, decision.interintra_mode);
  EXPECT_EQ(symbols.wedge_interintra, decision.wedge_interintra);
  EXPECT_EQ(symbols.wedge_index, decision.wedge_index);

  // Weights match §7.13.3.29 for all four modes.
  std::vector<std::uint8_t> weights(static_cast<std::size_t>(w) * h);
  for (int m = 0; m < 4; ++m) {
    build_inter_intra_weights(m, w, h, weights.data(), w);
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        EXPECT_EQ(weights[static_cast<std::size_t>(i) * w + j],
                  ia::intra_mode_mask_value(static_cast<ia::InterIntraMode>(m),
                                            w, h, i, j));
      }
    }
  }
}

// ---------------------------------------------------------------------------
// BAWP
// ---------------------------------------------------------------------------

void test_bawp_symbols() {
  for (int use = 0; use <= 1; ++use) {
    for (int explicit_bawp = 0; explicit_bawp <= 1; ++explicit_bawp) {
      for (int scale = 0; scale <= 1; ++scale) {
        for (int chroma = 0; chroma <= 1; ++chroma) {
          if (!use && (explicit_bawp || scale || chroma)) continue;
          if (!explicit_bawp && scale) continue;
          BawpSymbols symbols{};
          symbols.use_bawp = use != 0;
          symbols.explicit_bawp = explicit_bawp != 0;
          symbols.explicit_bawp_scale = scale != 0;
          symbols.use_bawp_chroma = chroma != 0;
          symbols.explicit_ctx = 2;
          std::string error;
          auto write = [&](av2::Encoder& enc, AdvancedCdfState& cdfs,
                           std::string* err) {
            return write_bawp_symbols(enc, cdfs, symbols, true, err);
          };
          auto read = [&](av2::Decoder& dec, AdvancedCdfState& cdfs,
                          std::string* err) {
            BawpSymbols out{};
            return read_bawp_symbols(dec, cdfs, 2, true, &out, err) &&
                   out.use_bawp == symbols.use_bawp &&
                   out.explicit_bawp == symbols.explicit_bawp &&
                   out.explicit_bawp_scale == symbols.explicit_bawp_scale &&
                   out.use_bawp_chroma == symbols.use_bawp_chroma;
          };
          EXPECT(round_trip(write, read, &error));
        }
      }
    }
  }
}

void test_bawp_alpha_rules() {
  for (int list = 0; list < 3; ++list) {
    for (int dist = 0; dist <= 7; ++dist) {
      for (int sign = 0; sign <= 1; ++sign) {
        int scale = list + 1;
        if (dist > 4) ++scale;
        if (!sign) scale = -scale;
        EXPECT_EQ(explicit_bawp_alpha(list, dist, sign != 0), 256 + 16 * scale);
      }
    }
  }
  EXPECT_EQ(bawp_alpha_for_plane(0, 300, 0), 300);
  EXPECT_EQ(bawp_alpha_for_plane(1, 300, 0), 256);
  EXPECT_EQ(bawp_alpha_for_plane(1, 300, 8), 300);
  EXPECT_EQ(bawp_alpha_for_plane(2, 77, 4), 77);
}

void test_bawp_decision() {
  const int plane_w = 64;
  const int plane_h = 64;
  const int alpha = 288;  // Q8
  auto scaled = [&](int v) { return std::min(255, (v * alpha) >> 8); };

  std::vector<std::uint8_t> prev_storage(
      static_cast<std::size_t>(plane_w) * plane_h);
  for (int i = 0; i < plane_h; ++i) {
    for (int j = 0; j < plane_w; ++j) {
      prev_storage[static_cast<std::size_t>(i) * plane_w + j] =
          static_cast<std::uint8_t>((i * 5 + j * 3 + 17) & 255);
    }
  }
  const ia::BlockRect rect{16, 16, 16, 16};
  // CurrFrame initially holds the unscaled inter prediction.
  std::vector<std::uint8_t> cur_storage = prev_storage;
  // Reconstructed above/left neighbours follow the scaled relation.
  std::vector<std::uint8_t> source(static_cast<std::size_t>(rect.w) * rect.h);
  for (int j = 0; j < rect.w; ++j) {
    const std::size_t p = static_cast<std::size_t>(rect.y - 1) * plane_w +
                          rect.x + j;
    cur_storage[p] = static_cast<std::uint8_t>(
        scaled(prev_storage[p]));
  }
  for (int i = 0; i < rect.h; ++i) {
    const std::size_t p = static_cast<std::size_t>(rect.y + i) * plane_w +
                          rect.x - 1;
    cur_storage[p] = static_cast<std::uint8_t>(
        scaled(prev_storage[p]));
  }
  for (int i = 0; i < rect.h; ++i) {
    for (int j = 0; j < rect.w; ++j) {
      const std::size_t p = static_cast<std::size_t>(rect.y + i) * plane_w +
                            rect.x + j;
      source[static_cast<std::size_t>(i) * rect.w + j] =
          static_cast<std::uint8_t>(scaled(prev_storage[p]));
    }
  }

  ia::BawpJob job{};
  job.ctx = ia::BawpContext{};
  job.ctx.plane = 0;
  job.ctx.mi_col = rect.x / 4;
  job.ctx.mi_row = rect.y / 4;
  job.ctx.mi_cols = plane_w / 4;
  job.ctx.mi_rows = plane_h / 4;
  job.ctx.avail_u = true;
  job.ctx.avail_l = true;
  job.ctx.bit_depth = 8;
  job.rect = rect;
  job.mv[0] = 0;
  job.mv[1] = 0;
  job.cur = ia::PlaneView{cur_storage.data(), plane_w, plane_w - 1, plane_h - 1};
  job.prev = ia::PlaneView{prev_storage.data(), plane_w, plane_w - 1, plane_h - 1};
  job.dst = ia::MutablePlaneView{cur_storage.data(), plane_w, plane_w - 1, plane_h - 1};

  std::vector<std::uint8_t> scratch(cur_storage.size());
  BawpDecisionInput in{};
  in.job = job;
  in.source = source.data();
  in.source_stride = rect.w;
  in.scratch = scratch.data();
  in.scratch_stride = plane_w;
  in.allow_bawp = true;
  in.ref_scaled = false;
  in.y_mode = static_cast<int>(YMode::kNearMv);
  in.order_hint = 1;
  in.ref_order_hint = 0;

  BawpDecision decision = decide_bawp(in);
  EXPECT(decision.use_bawp);
  EXPECT(decision.sse_best <= decision.sse_off);
  EXPECT(decision.sse_best < decision.sse_off);
  EXPECT(decision.luma_alpha > 256);
  EXPECT_EQ(decision.explicit_ctx, 0);
  EXPECT_EQ(decision.list_index, 0);
  EXPECT_EQ(decision.first_ref_dist, 1);

  const BawpSymbols symbols = bawp_symbols_from_decision(decision);
  EXPECT_EQ(symbols.use_bawp, decision.use_bawp);
  EXPECT_EQ(symbols.explicit_bawp, decision.explicit_bawp);
  EXPECT_EQ(symbols.explicit_bawp_scale, decision.explicit_bawp_scale);
  EXPECT_EQ(symbols.use_bawp_chroma, decision.use_bawp_chroma);
  EXPECT_EQ(symbols.explicit_ctx, decision.explicit_ctx);
  ia::BawpJob configured = job;
  apply_bawp_decision_to_job(decision, false, &configured);
  EXPECT_EQ(configured.explicit_bawp, decision.explicit_bawp);
  EXPECT_EQ(configured.list_index, decision.list_index);
  EXPECT_EQ(configured.first_ref_dist, decision.first_ref_dist);
  ia::BawpJob chroma_configured = job;
  apply_bawp_decision_to_job(decision, true, &chroma_configured);
  EXPECT_EQ(chroma_configured.bawp_alpha, decision.luma_alpha);
  EXPECT(!chroma_configured.explicit_bawp);

  // Conditions that disable BAWP.
  in.y_mode = static_cast<int>(YMode::kGlobalMv);
  EXPECT(!decide_bawp(in).use_bawp);
  in.y_mode = static_cast<int>(YMode::kNearMv);
  in.ref_scaled = true;
  EXPECT(!decide_bawp(in).use_bawp);
  in.ref_scaled = false;
  in.allow_bawp = false;
  EXPECT(!decide_bawp(in).use_bawp);

  // The explicit-alpha path must be selectable when the template is absent.
  ia::BawpJob no_template = job;
  no_template.ctx.avail_u = false;
  no_template.ctx.avail_l = false;
  in.allow_bawp = true;
  in.job = no_template;
  BawpDecision explicit_decision = decide_bawp(in);
  EXPECT(explicit_decision.use_bawp);
  EXPECT(explicit_decision.explicit_bawp);
}

void test_bawp_chroma_decision() {
  const int lw = 32;
  const int lh = 32;
  const int luma_alpha = 288;
  auto scale8 = [&](int v, int a) { return std::min(255, (v * a) >> 8); };

  std::vector<std::uint8_t> lprev(static_cast<std::size_t>(lw) * lh);
  for (int i = 0; i < lh; ++i) {
    for (int j = 0; j < lw; ++j) {
      lprev[static_cast<std::size_t>(i) * lw + j] =
          static_cast<std::uint8_t>((i * 5 + j * 3 + 17) & 255);
    }
  }
  const ia::BlockRect lrect{8, 8, 8, 8};
  std::vector<std::uint8_t> lcur = lprev;
  std::vector<std::uint8_t> lsource(static_cast<std::size_t>(lrect.w) * lrect.h);
  for (int j = 0; j < lrect.w; ++j) {
    const std::size_t p = static_cast<std::size_t>(lrect.y - 1) * lw + lrect.x + j;
    lcur[p] = static_cast<std::uint8_t>(scale8(lprev[p], luma_alpha));
  }
  for (int i = 0; i < lrect.h; ++i) {
    const std::size_t p = static_cast<std::size_t>(lrect.y + i) * lw + lrect.x - 1;
    lcur[p] = static_cast<std::uint8_t>(scale8(lprev[p], luma_alpha));
  }
  for (int i = 0; i < lrect.h; ++i) {
    for (int j = 0; j < lrect.w; ++j) {
      const std::size_t p = static_cast<std::size_t>(lrect.y + i) * lw + lrect.x + j;
      lsource[static_cast<std::size_t>(i) * lrect.w + j] =
          static_cast<std::uint8_t>(scale8(lprev[p], luma_alpha));
    }
  }

  ia::BawpJob luma{};
  luma.ctx.plane = 0;
  luma.ctx.mi_col = lrect.x / 4;
  luma.ctx.mi_row = lrect.y / 4;
  luma.ctx.mi_cols = lw / 4;
  luma.ctx.mi_rows = lh / 4;
  luma.ctx.avail_u = true;
  luma.ctx.avail_l = true;
  luma.rect = lrect;
  luma.mv[0] = 0;
  luma.mv[1] = 0;
  luma.cur = ia::PlaneView{lcur.data(), lw, lw - 1, lh - 1};
  luma.prev = ia::PlaneView{lprev.data(), lw, lw - 1, lh - 1};
  luma.dst = ia::MutablePlaneView{lcur.data(), lw, lw - 1, lh - 1};

  const int cw = 16;
  const int ch = 16;
  const int chroma_alpha = 300;
  std::vector<std::uint8_t> cprev(static_cast<std::size_t>(cw) * ch);
  for (int i = 0; i < ch; ++i) {
    for (int j = 0; j < cw; ++j) {
      cprev[static_cast<std::size_t>(i) * cw + j] =
          static_cast<std::uint8_t>((i * 9 + j * 7 + 3) & 255);
    }
  }
  const ia::BlockRect crect{4, 4, 4, 4};
  std::vector<std::uint8_t> ccur = cprev;
  std::vector<std::uint8_t> csource(static_cast<std::size_t>(crect.w) * crect.h);
  for (int j = 0; j < crect.w; ++j) {
    const std::size_t p = static_cast<std::size_t>(crect.y - 1) * cw + crect.x + j;
    ccur[p] = static_cast<std::uint8_t>(scale8(cprev[p], chroma_alpha));
  }
  for (int i = 0; i < crect.h; ++i) {
    const std::size_t p = static_cast<std::size_t>(crect.y + i) * cw + crect.x - 1;
    ccur[p] = static_cast<std::uint8_t>(scale8(cprev[p], chroma_alpha));
  }
  for (int i = 0; i < crect.h; ++i) {
    for (int j = 0; j < crect.w; ++j) {
      const std::size_t p = static_cast<std::size_t>(crect.y + i) * cw + crect.x + j;
      csource[static_cast<std::size_t>(i) * crect.w + j] =
          static_cast<std::uint8_t>(scale8(cprev[p], chroma_alpha));
    }
  }

  ia::BawpJob chroma{};
  chroma.ctx.plane = 1;
  chroma.ctx.sub_x = 1;
  chroma.ctx.sub_y = 1;
  chroma.ctx.mi_col = crect.x / 2;
  chroma.ctx.mi_row = crect.y / 2;
  chroma.ctx.mi_cols = cw / 2;
  chroma.ctx.mi_rows = ch / 2;
  chroma.ctx.avail_u = true;
  chroma.ctx.avail_l = true;
  chroma.rect = crect;
  chroma.mv[0] = 0;
  chroma.mv[1] = 0;
  chroma.cur = ia::PlaneView{ccur.data(), cw, cw - 1, ch - 1};
  chroma.prev = ia::PlaneView{cprev.data(), cw, cw - 1, ch - 1};
  chroma.dst = ia::MutablePlaneView{ccur.data(), cw, cw - 1, ch - 1};

  std::vector<std::uint8_t> scratch(lcur.size());
  std::vector<std::uint8_t> cscratch(ccur.size());
  BawpDecisionInput in{};
  in.job = luma;
  in.source = lsource.data();
  in.source_stride = lrect.w;
  in.scratch = scratch.data();
  in.scratch_stride = lw;
  in.chroma_job = &chroma;
  in.chroma_source = csource.data();
  in.chroma_source_stride = crect.w;
  in.chroma_scratch = cscratch.data();
  in.chroma_scratch_stride = cw;
  in.has_chroma = true;
  in.y_mode = static_cast<int>(YMode::kNearMv);

  const BawpDecision decision = decide_bawp(in);
  EXPECT(decision.use_bawp);
  EXPECT(decision.use_bawp_chroma);
  EXPECT_EQ(decision.chroma_alpha, decision.luma_alpha);
}

// ---------------------------------------------------------------------------
// Full ordered block writer/reader (advanced symbols + raw baseline)
// ---------------------------------------------------------------------------

struct RawBaseline {
  av2::Encoder* enc = nullptr;
  av2::Decoder* dec = nullptr;
};

bool raw_put(av2::Encoder* enc, std::uint32_t value, int bits) {
  for (int i = bits - 1; i >= 0; --i) {
    if (enc->encode_bool(((value >> i) & 1u) != 0) != av2::Status::kOk) {
      return false;
    }
  }
  return true;
}

bool raw_get(av2::Decoder* dec, std::uint32_t* value, int bits) {
  std::uint32_t v = 0;
  for (int i = 0; i < bits; ++i) {
    bool bit = false;
    if (dec->read_bool(&bit) != av2::Status::kOk) return false;
    v = (v << 1) | (bit ? 1u : 0u);
  }
  *value = v;
  return true;
}

bool raw_single_write(void* ctx, int, int symbol) {
  auto* b = static_cast<RawBaseline*>(ctx);
  return raw_put(b->enc, static_cast<std::uint32_t>(symbol), 2);
}
bool raw_single_read(void* ctx, int, int* symbol) {
  auto* b = static_cast<RawBaseline*>(ctx);
  std::uint32_t v = 0;
  if (!raw_get(b->dec, &v, 2)) return false;
  *symbol = static_cast<int>(v);
  return true;
}
bool raw_drl_write(void* ctx, int, int index) {
  auto* b = static_cast<RawBaseline*>(ctx);
  return raw_put(b->enc, static_cast<std::uint32_t>(index), 3);
}
bool raw_drl_read(void* ctx, int, int* index) {
  auto* b = static_cast<RawBaseline*>(ctx);
  std::uint32_t v = 0;
  if (!raw_get(b->dec, &v, 3)) return false;
  *index = static_cast<int>(v);
  return true;
}
bool raw_mvd_write(void* ctx, int dy, int dx) {
  auto* b = static_cast<RawBaseline*>(ctx);
  return raw_put(b->enc, static_cast<std::uint32_t>(dy & 255), 8) &&
         raw_put(b->enc, static_cast<std::uint32_t>(dx & 255), 8);
}
bool raw_mvd_read(void* ctx, int* dy, int* dx) {
  auto* b = static_cast<RawBaseline*>(ctx);
  std::uint32_t y = 0;
  std::uint32_t x = 0;
  if (!raw_get(b->dec, &y, 8) || !raw_get(b->dec, &x, 8)) return false;
  *dy = static_cast<int>(static_cast<std::int8_t>(y));
  *dx = static_cast<int>(static_cast<std::int8_t>(x));
  return true;
}

void test_ordered_block_round_trip() {
  std::vector<AdvancedBlockCoding> plans;
  auto base_plan = [] {
    AdvancedBlockCoding p{};
    p.mi_size = static_cast<int>(av2enc::partition::BlockSize::k16x16);
    p.new_mv_ctx = 2;
    p.allow_warpmv_mode = true;
    p.allow_bawp = true;
    p.has_chroma = true;
    p.motion.wedge_allowed = true;
    p.motion.inter_intra_allowed = true;
    p.motion.local_allowed = true;
    p.motion.extend_allowed = true;
    p.motion.extend_warp_ctx = 1;
    p.motion.local_warp_ctx = 2;
    return p;
  };

  {
    AdvancedBlockCoding p = base_plan();
    p.single_mode = 0;  // NEARMV
    p.bawp.use_bawp = true;
    p.bawp.use_bawp_chroma = false;
    p.drl_index = 1;
    plans.push_back(p);
  }
  {
    AdvancedBlockCoding p = base_plan();
    p.single_mode = 2;  // NEWMV
    p.bawp.use_bawp = false;
    p.motion.inter_intra = true;
    p.motion.interintra_mode = 2;
    p.motion.wedge_interintra = true;
    p.motion.wedge_index = 7;
    p.drl_index = 0;
    p.mvd_y = 3;
    p.mvd_x = -5;
    plans.push_back(p);
  }
  {
    AdvancedBlockCoding p = base_plan();
    p.is_warp = true;
    p.warp_mv = false;  // WARP_NEWMV
    p.motion.use_extend_warp = true;
    p.warp_idx = 2;
    p.drl_index = 2;
    p.mvd_y = -7;
    p.mvd_x = 4;
    plans.push_back(p);
  }
  {
    AdvancedBlockCoding p = base_plan();
    p.is_warp = true;
    p.warp_mv = false;  // WARP_NEWMV + DELTAWARP
    p.warp_idx = 1;
    p.six_param = true;
    p.warp_delta.high_precision = true;
    p.warp_delta.magnitude[0] = 9;
    p.warp_delta.negative[0] = true;
    p.warp_delta.magnitude[1] = 2;
    p.warp_delta.magnitude[2] = 3;
    p.warp_delta.magnitude[3] = 1;
    p.warp_delta.negative[3] = true;
    p.drl_index = 0;
    p.mvd_y = 1;
    p.mvd_x = 2;
    plans.push_back(p);
  }
  {
    AdvancedBlockCoding p = base_plan();
    p.is_warp = true;
    p.warp_mv = true;  // WARPMV
    p.warp_idx = 0;
    p.warpmv_with_mvd = true;
    p.mvd_y = 2;
    p.mvd_x = -3;
    p.motion.inter_intra = true;
    p.motion.interintra_mode = 1;
    p.motion.wedge_interintra = false;
    plans.push_back(p);
  }
  {
    AdvancedBlockCoding p = base_plan();
    p.is_warp = true;
    p.warp_mv = true;
    p.force_integer_mv = true;
    p.warp_idx = 3;
    p.motion.inter_intra = false;
    plans.push_back(p);
  }
  {
    AdvancedBlockCoding p = base_plan();
    p.single_mode = 1;  // GLOBALMV
    p.bawp.use_bawp = false;
    plans.push_back(p);
  }
  {
    AdvancedBlockCoding p = base_plan();
    p.single_mode = 0;  // NEARMV + explicit BAWP + chroma
    p.bawp.use_bawp = true;
    p.bawp.explicit_bawp = true;
    p.bawp.explicit_bawp_scale = true;
    p.bawp.use_bawp_chroma = true;
    p.bawp.explicit_ctx = 0;
    p.drl_index = 0;
    plans.push_back(p);
  }
  {
    AdvancedBlockCoding p = base_plan();
    p.single_mode = 2;  // NEWMV + LOCALWARP
    p.motion.use_local_warp = true;
    p.drl_index = 1;
    p.mvd_y = 0;
    p.mvd_x = 0;
    plans.push_back(p);
  }
  {
    AdvancedBlockCoding p = base_plan();
    p.is_warp = true;
    p.warp_mv = false;  // WARP_NEWMV + LOCALWARP
    p.motion.use_local_warp = true;
    p.drl_index = 0;
    p.mvd_y = 2;
    p.mvd_x = 3;
    plans.push_back(p);
  }
  {
    AdvancedBlockCoding p = base_plan();
    p.is_warp = true;
    p.warp_mv = false;  // WARP_NEWMV + DELTAWARP, 4-param, no candidate
    p.warp_idx = 3;
    p.warp_delta.high_precision = false;
    p.warp_delta.magnitude[0] = 3;
    p.warp_delta.negative[0] = true;
    p.warp_delta.magnitude[1] = 2;
    p.drl_index = 0;
    p.mvd_y = -1;
    p.mvd_x = 1;
    plans.push_back(p);
  }
  {
    AdvancedBlockCoding p = base_plan();
    p.is_warp = true;
    p.warp_mv = true;  // WARPMV + wedge inter-intra
    p.warp_idx = 1;
    p.warpmv_with_mvd = false;
    p.motion.inter_intra = true;
    p.motion.interintra_mode = 0;
    p.motion.wedge_interintra = true;
    p.motion.wedge_index = 12;
    plans.push_back(p);
  }

  for (const AdvancedBlockCoding& plan : plans) {
    std::vector<std::uint8_t> buffer(1u << 16);
    av2::Encoder enc;
    EXPECT(enc.init_symbol(buffer.data(), buffer.size()) == av2::Status::kOk);
    AdvancedCdfState enc_cdfs;
    enc_cdfs.reset();
    RawBaseline raw{&enc, nullptr};
    BaselineInterWriter writer{&raw, raw_single_write, raw_drl_write,
                               raw_mvd_write};
    std::string error;
    EXPECT(write_advanced_inter_block(enc, enc_cdfs, plan, writer, &error));
    std::size_t bytes = 0;
    EXPECT(enc.exit_symbol(&bytes) == av2::Status::kOk);

    av2::Decoder dec;
    EXPECT(dec.init_symbol(buffer.data(), buffer.size(), bytes) ==
           av2::Status::kOk);
    AdvancedCdfState dec_cdfs;
    dec_cdfs.reset();
    RawBaseline raw_dec{nullptr, &dec};
    BaselineInterReader reader{&raw_dec, raw_single_read, raw_drl_read,
                               raw_mvd_read};
    AdvancedBlockCoding gates = plan;
    AdvancedBlockCoding decoded{};
    EXPECT(read_advanced_inter_block(dec, dec_cdfs, gates, reader, &decoded,
                                     &error));
    EXPECT(dec.exit_symbol() == av2::Status::kOk);
    EXPECT(cdfs_equal(enc_cdfs, dec_cdfs));
    EXPECT_EQ(decoded.is_warp, plan.is_warp);
    if (plan.is_warp) EXPECT_EQ(decoded.warp_mv, plan.warp_mv);
    if (!plan.is_warp) EXPECT_EQ(decoded.single_mode, plan.single_mode);
    const bool warp_idx_coded =
        (plan.is_warp && plan.warp_mv) ||
        (plan.is_warp && !plan.motion.use_extend_warp &&
         !plan.motion.use_local_warp);
    if (warp_idx_coded) EXPECT_EQ(decoded.warp_idx, plan.warp_idx);
    if (plan.warp_idx < 2 && plan.is_warp && plan.warp_mv) {
      EXPECT_EQ(decoded.warpmv_with_mvd, plan.warpmv_with_mvd);
    }
    if (!plan.is_warp) {
      EXPECT_EQ(decoded.bawp.use_bawp, plan.bawp.use_bawp);
      EXPECT_EQ(decoded.bawp.explicit_bawp, plan.bawp.explicit_bawp);
      EXPECT_EQ(decoded.bawp.use_bawp_chroma, plan.bawp.use_bawp_chroma);
    }
    EXPECT_EQ(decoded.motion.inter_intra, plan.motion.inter_intra);
    EXPECT_EQ(decoded.motion.interintra_mode, plan.motion.interintra_mode);
    EXPECT_EQ(decoded.motion.wedge_interintra, plan.motion.wedge_interintra);
    EXPECT_EQ(decoded.motion.wedge_index, plan.motion.wedge_index);
    EXPECT_EQ(decoded.motion.use_extend_warp, plan.motion.use_extend_warp);
    EXPECT_EQ(decoded.motion.use_local_warp, plan.motion.use_local_warp);
    EXPECT_EQ(decoded.drl_index, plan.drl_index);
    EXPECT_EQ(decoded.mvd_y, plan.mvd_y);
    EXPECT_EQ(decoded.mvd_x, plan.mvd_x);
    if (plan.six_param && plan.warp_idx == 1) {
      EXPECT_EQ(decoded.six_param, plan.six_param);
      EXPECT(decoded.warp_delta.high_precision ==
             plan.warp_delta.high_precision);
      for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(decoded.warp_delta.magnitude[i], plan.warp_delta.magnitude[i]);
        EXPECT_EQ(decoded.warp_delta.negative[i], plan.warp_delta.negative[i]);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Sequence / frame bits
// ---------------------------------------------------------------------------

struct BitReader {
  const std::uint8_t* data = nullptr;
  std::size_t bits = 0;
  std::size_t pos = 0;

  bool get(int n, std::uint32_t* value) {
    if (n < 0 || pos + static_cast<std::size_t>(n) > bits) return false;
    std::uint32_t v = 0;
    for (int i = 0; i < n; ++i) {
      const std::size_t p = pos++;
      v = (v << 1) | ((data[p >> 3] >> (7 - (p & 7))) & 1u);
    }
    *value = v;
    return true;
  }
};

void test_sequence_frame_bits() {
  SequenceAdvancedConfig seq{};
  seq.seq_enabled_motion_modes[static_cast<int>(ia::MotionMode::kInterIntra)] = true;
  seq.seq_enabled_motion_modes[static_cast<int>(ia::MotionMode::kLocalWarp)] = true;
  seq.seq_enabled_motion_modes[static_cast<int>(ia::MotionMode::kDeltaWarp)] = true;
  seq.seq_enabled_motion_modes[static_cast<int>(ia::MotionMode::kExtendWarp)] = true;
  seq.seq_frame_motion_modes_present_flag = true;
  seq.enable_six_param_warp_delta = true;
  seq.enable_masked_compound = true;
  seq.enable_bawp = true;
  seq.enable_cwp = true;
  seq.enable_imp_msk_bld = true;

  av2enc::BitWriter writer;
  std::string error;
  EXPECT(write_sequence_motion_mode_config(&writer, seq, &error));
  EXPECT(write_sequence_bawp_cwp_bits(&writer, seq, &error));
  EXPECT(write_sequence_global_motion_bit(&writer, seq, &error));
  const std::size_t bit_count = writer.bit_size();
  EXPECT_EQ(bit_count, 4 + 1 + 1 + 1 + 3 + 1);

  BitReader reader{writer.bytes().data(), writer.bit_size(), 0};
  std::uint32_t bit = 0;
  EXPECT(reader.get(1, &bit) && bit == 1);  // INTERINTRA
  EXPECT(reader.get(1, &bit) && bit == 1);  // LOCALWARP
  EXPECT(reader.get(1, &bit) && bit == 1);  // DELTAWARP
  EXPECT(reader.get(1, &bit) && bit == 1);  // EXTENDWARP
  EXPECT(reader.get(1, &bit) && bit == 1);  // seq_frame_motion_modes_present
  EXPECT(reader.get(1, &bit) && bit == 1);  // enable_six_param_warp_delta
  EXPECT(reader.get(1, &bit) && bit == 1);  // enable_masked_compound
  EXPECT(reader.get(1, &bit) && bit == 1);  // enable_bawp
  EXPECT(reader.get(1, &bit) && bit == 1);  // enable_cwp
  EXPECT(reader.get(1, &bit) && bit == 1);  // enable_imp_msk_bld
  EXPECT(reader.get(1, &bit) && bit == 0);  // enable_global_motion

  FrameAdvancedConfig frame{};
  frame.frame_is_intra = false;
  frame.seq_frame_motion_modes_present_flag = true;
  for (int mode = 1; mode < kMotionModes; ++mode) {
    frame.seq_enabled_motion_modes[mode] =
        seq.seq_enabled_motion_modes[mode];
    frame.frame_enabled_motion_modes[mode] =
        seq.seq_enabled_motion_modes[mode];
  }
  frame.frame_enabled_motion_modes[static_cast<int>(ia::MotionMode::kLocalWarp)] = false;
  frame.enable_bawp = true;
  frame.allow_bawp = true;
  frame.allow_warpmv_mode = true;

  av2enc::BitWriter fw;
  EXPECT(write_frame_motion_mode_bits(&fw, frame, &error));
  EXPECT(write_frame_bawp_warp_bits(&fw, frame, &error));
  EXPECT_EQ(fw.bit_size(), 4 + 2);
  BitReader fr{fw.bytes().data(), fw.bit_size(), 0};
  EXPECT(fr.get(1, &bit) && bit == 1);  // INTERINTRA frame-enabled
  EXPECT(fr.get(1, &bit) && bit == 0);  // LOCALWARP frame-disabled
  EXPECT(fr.get(1, &bit) && bit == 1);  // DELTAWARP
  EXPECT(fr.get(1, &bit) && bit == 1);  // EXTENDWARP
}

void test_invalid_frame_bits() {
  std::string error;
  SequenceAdvancedConfig seq{};
  seq.seq_frame_motion_modes_present_flag = true;
  av2enc::BitWriter writer;
  EXPECT(!write_sequence_motion_mode_config(&writer, seq, &error));

  seq = SequenceAdvancedConfig{};
  seq.seq_enabled_motion_modes[static_cast<int>(ia::MotionMode::kInterIntra)] = true;
  seq.enable_six_param_warp_delta = true;
  av2enc::BitWriter writer2;
  EXPECT(!write_sequence_motion_mode_config(&writer2, seq, &error));

  FrameAdvancedConfig frame{};
  frame.frame_is_intra = true;
  frame.allow_bawp = true;
  av2enc::BitWriter writer3;
  EXPECT(!write_frame_bawp_warp_bits(&writer3, frame, &error));

  frame = FrameAdvancedConfig{};
  frame.seq_enabled_motion_modes[static_cast<int>(ia::MotionMode::kDeltaWarp)] = true;
  frame.frame_enabled_motion_modes[static_cast<int>(ia::MotionMode::kDeltaWarp)] = false;
  av2enc::BitWriter writer4;
  // No seq_frame_motion_modes_present_flag: the frame value must match.
  EXPECT(!write_frame_motion_mode_bits(&writer4, frame, &error));
}

}  // namespace

int main() {
  test_cdf_defaults();
  test_wedge_mapping();
  test_wedge_weights();
  test_warp_symbols();
  test_warp_delta_symbols();
  test_warp_delta_quantization();
  test_warp_decoding_reconstruction();
  test_warp_contexts();
  test_warp_decision();
  test_inter_intra_symbols();
  test_motion_mode_symbols();
  test_inter_intra_decision_and_weights();
  test_bawp_symbols();
  test_bawp_alpha_rules();
  test_bawp_decision();
  test_bawp_chroma_decision();
  test_ordered_block_round_trip();
  test_sequence_frame_bits();
  test_invalid_frame_bits();

  if (g_failures == 0) {
    std::printf("advanced_test: %d checks passed\n", g_checks);
    return 0;
  }
  std::printf("advanced_test: %d / %d checks FAILED\n", g_failures, g_checks);
  return 1;
}

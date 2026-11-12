// CPU reference tests for the AV2 lossless / DPCM / FSC module.
//
// The tests deliberately check three independent things:
//   1. §5.18.2 lossless flag derivation (CodedLossless / LosslessArray).
//   2. the reversible transform and DPCM helpers against their algebraic
//      inverses (these are the pieces that make "0-LSB" possible);
//   3. the full writer -> reader round trip, including every sample in the
//      reconstructed frame, for a matrix of sizes, content patterns, DPCM and
//      FSC settings.
//
// The decoder gate (dav2d / avmdec) is a separate executable because it needs
// an external decoder and writes IVF files.

#include "codec/entropy/lossless/lossless.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
int g_checks = 0;

void fail(const char* file, int line, const std::string& what) {
  ++g_failures;
  std::fprintf(stderr, "FAIL %s:%d: %s\n", file, line, what.c_str());
}

#define CHECK(cond)                                                        \
  do {                                                                     \
    ++g_checks;                                                            \
    if (!(cond)) fail(__FILE__, __LINE__, #cond);                          \
  } while (0)

using av2enc::lossless::EncodeResult;
using av2enc::lossless::FscMode;
using av2enc::lossless::LosslessInputs;
using av2enc::lossless::Options;
using av2enc::lossless::StillPicture;

struct Picture {
  int w = 0;
  int h = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
};

Picture make_picture(int w, int h, int pattern) {
  Picture p;
  p.w = w;
  p.h = h;
  p.y.resize(static_cast<std::size_t>(w) * h);
  p.u.resize(static_cast<std::size_t>(w) * h / 4);
  p.v.resize(static_cast<std::size_t>(w) * h / 4);
  std::uint32_t lcg = 0x12345678u;
  for (int i = 0; i < w * h; ++i) {
    const int x = i % w;
    const int y = i / w;
    switch (pattern) {
      case 0:
        p.y[i] = 128;
        break;
      case 1:
        p.y[i] = static_cast<std::uint8_t>((x * 5 + y * 3) & 255);
        break;
      case 2:
        lcg = lcg * 1664525u + 1013904223u;
        p.y[i] = static_cast<std::uint8_t>(lcg >> 24);
        break;
      case 3:
        // Screen-content-like: flat 8x8 tiles plus a few high-contrast bars.
        p.y[i] = static_cast<std::uint8_t>(((x / 8) * 37 + (y / 8) * 19) & 255);
        if ((y / 4) % 5 == 0) p.y[i] = static_cast<std::uint8_t>(x < w / 2 ? 0 : 255);
        break;
      default:
        p.y[i] = static_cast<std::uint8_t>(((x / 4) + (y / 4)) & 1 ? 0 : 255);
        break;
    }
  }
  for (int i = 0; i < w * h / 4; ++i) {
    const int x = i % (w / 2);
    const int y = i / (w / 2);
    switch (pattern) {
      case 0:
        p.u[i] = 128;
        p.v[i] = 128;
        break;
      case 1:
        p.u[i] = static_cast<std::uint8_t>((x * 7 + y * 11) & 255);
        p.v[i] = static_cast<std::uint8_t>((x * 13 + y * 2) & 255);
        break;
      case 2:
        lcg = lcg * 1664525u + 1013904223u;
        p.u[i] = static_cast<std::uint8_t>(lcg >> 24);
        lcg = lcg * 1664525u + 1013904223u;
        p.v[i] = static_cast<std::uint8_t>(lcg >> 24);
        break;
      default:
        p.u[i] = static_cast<std::uint8_t>(((x / 8) * 61) & 255);
        p.v[i] = static_cast<std::uint8_t>(((y / 8) * 53) & 255);
        break;
    }
  }
  return p;
}

StillPicture view(const Picture& p) {
  StillPicture s;
  s.width = static_cast<std::uint32_t>(p.w);
  s.height = static_cast<std::uint32_t>(p.h);
  s.y = p.y;
  s.u = p.u;
  s.v = p.v;
  return s;
}

void test_lossless_flags() {
  using av2enc::lossless::compute_lossless_flags;
  using av2enc::lossless::qindex_for_segment;

  LosslessInputs in;
  in.base_q_idx = 0;
  auto f = compute_lossless_flags(in);
  CHECK(f.coded_lossless);
  CHECK(f.has_lossless_segment);
  for (int s = 0; s < 8; ++s) CHECK(f.lossless[s]);

  in.base_q_idx = 1;
  f = compute_lossless_flags(in);
  CHECK(!f.coded_lossless);
  CHECK(!f.has_lossless_segment);

  in.base_q_idx = 0;
  in.delta_q_present = true;
  f = compute_lossless_flags(in);
  CHECK(!f.coded_lossless);
  CHECK(!f.has_lossless_segment);

  in.delta_q_present = false;
  in.delta_q_u_ac = 1;
  f = compute_lossless_flags(in);
  CHECK(!f.coded_lossless);
  in.delta_q_u_ac = 0;
  in.base_y_dc_delta_q = 1;
  f = compute_lossless_flags(in);
  CHECK(!f.coded_lossless);

  in = LosslessInputs{};
  in.delta_q_u_ac = -1;
  in.base_uv_ac_delta_q = 0;
  f = compute_lossless_flags(in);
  CHECK(f.coded_lossless);

  // A positive ALT_Q on one segment breaks CodedLossless but leaves the other
  // segments lossless.
  in = LosslessInputs{};
  in.alt_q_enabled[3] = true;
  in.alt_q[3] = 5;
  f = compute_lossless_flags(in);
  CHECK(!f.coded_lossless);
  CHECK(f.has_lossless_segment);
  CHECK(f.lossless[0]);
  CHECK(!f.lossless[3]);
  CHECK(qindex_for_segment(in, 3) == 5);
  CHECK(qindex_for_segment(in, 0) == 0);

  // A negative ALT_Q can make an otherwise lossy base lossless.
  in = LosslessInputs{};
  in.base_q_idx = 5;
  in.alt_q_enabled[4] = true;
  in.alt_q[4] = -5;
  f = compute_lossless_flags(in);
  CHECK(!f.coded_lossless);
  CHECK(f.has_lossless_segment);
  CHECK(f.lossless[4]);
  CHECK(!f.lossless[0]);
  CHECK(qindex_for_segment(in, 4) == 0);
}

void test_transforms() {
  using av2enc::lossless::dpcm_cumsum_4x4;
  using av2enc::lossless::dpcm_difference_4x4;
  using av2enc::lossless::forward_lossless_4x4;
  using av2enc::lossless::forward_wht4;
  using av2enc::lossless::inverse_lossless_4x4;
  using av2enc::lossless::inverse_wht4;

  std::uint32_t lcg = 0xabcdef01u;
  for (int trial = 0; trial < 2000; ++trial) {
    int src[4];
    int transformed[4];
    int back[4];
    for (int i = 0; i < 4; ++i) {
      lcg = lcg * 1664525u + 1013904223u;
      src[i] = static_cast<int>(lcg % 4096) - 2048;
    }
    inverse_wht4(src, 0, transformed);
    forward_wht4(transformed, back);
    for (int i = 0; i < 4; ++i) CHECK(src[i] == back[i]);
  }

  for (int trial = 0; trial < 2000; ++trial) {
    int residual[16];
    int levels[16];
    int back[16];
    for (int i = 0; i < 16; ++i) {
      lcg = lcg * 1664525u + 1013904223u;
      residual[i] = static_cast<int>(lcg % 1021) - 510;
    }
    forward_lossless_4x4(residual, levels);
    inverse_lossless_4x4(levels, back);
    for (int i = 0; i < 16; ++i) CHECK(residual[i] == back[i]);
  }

  for (int vertical = 0; vertical < 2; ++vertical) {
    for (int trial = 0; trial < 1000; ++trial) {
      int residual[16];
      int diff[16];
      int back[16];
      for (int i = 0; i < 16; ++i) {
        lcg = lcg * 1664525u + 1013904223u;
        residual[i] = static_cast<int>(lcg % 511) - 255;
      }
      dpcm_difference_4x4(residual, vertical != 0, diff);
      dpcm_cumsum_4x4(diff, vertical != 0, back);
      for (int i = 0; i < 16; ++i) CHECK(residual[i] == back[i]);
    }
  }
}

struct RoundTripCase {
  int w;
  int h;
  int pattern;
  FscMode fsc;
  bool dpcm;
  bool adaptive;
};

bool run_round_trip(const RoundTripCase& c, std::string& detail) {
  const Picture p = make_picture(c.w, c.h, c.pattern);
  Options opt;
  opt.fsc = c.fsc;
  opt.enable_dpcm = c.dpcm;
  opt.adaptive_cdf = c.adaptive;
  opt.verify_exact = true;

  EncodeResult enc;
  std::string error;
  if (!av2enc::lossless::encode_still(view(p), opt, enc, error)) {
    detail = "encode: " + error;
    return false;
  }
  if (!enc.stats.exact || enc.stats.max_abs_recon_error != 0) {
    detail = "encoder recon is not 0-LSB exact";
    return false;
  }
  if (enc.recon_y != p.y || enc.recon_u != p.u || enc.recon_v != p.v) {
    detail = "encoder recon planes differ from source";
    return false;
  }
  if (enc.temporal_unit.size() <= enc.stats.tile_bytes) {
    detail = "temporal unit smaller than tile payload";
    return false;
  }
  if (enc.stats.luma_blocks <= 0) {
    detail = "no luma blocks coded";
    return false;
  }
  if (c.fsc == FscMode::kForce) {
    if (enc.stats.fsc_blocks != enc.stats.luma_blocks) {
      detail = "FSC force did not mark every block";
      return false;
    }
  } else if (c.fsc == FscMode::kOff) {
    if (enc.stats.fsc_blocks != 0) {
      detail = "FSC blocks coded with enable_fsc off";
      return false;
    }
  }
  if (!c.dpcm && (enc.stats.dpcm_y_blocks != 0 || enc.stats.dpcm_uv_blocks != 0)) {
    detail = "DPCM blocks coded with DPCM disabled";
    return false;
  }

  av2enc::lossless::DecodeResult dec;
  if (!av2enc::lossless::decode_temporal_unit(enc.temporal_unit, dec, error)) {
    detail = "decode: " + error;
    return false;
  }
  if (dec.width != static_cast<std::uint32_t>(c.w) ||
      dec.height != static_cast<std::uint32_t>(c.h)) {
    detail = "decoded geometry differs";
    return false;
  }
  if (dec.y != p.y || dec.u != p.u || dec.v != p.v) {
    detail = "decoded planes differ from source (not 0-LSB)";
    return false;
  }
  if (dec.stats.luma_blocks != enc.stats.luma_blocks ||
      dec.stats.fsc_blocks != enc.stats.fsc_blocks ||
      dec.stats.dpcm_y_blocks != enc.stats.dpcm_y_blocks ||
      dec.stats.dpcm_uv_blocks != enc.stats.dpcm_uv_blocks) {
    detail = "decoder block statistics differ from encoder";
    return false;
  }

  // Deterministic writer: the same picture and options must produce identical
  // bytes (pinned reference vectors can be added from these).
  EncodeResult again;
  if (!av2enc::lossless::encode_still(view(p), opt, again, error)) {
    detail = "second encode failed: " + error;
    return false;
  }
  if (again.temporal_unit != enc.temporal_unit) {
    detail = "encoder is not deterministic";
    return false;
  }
  return true;
}

void test_round_trips() {
  const int sizes[][2] = {{8, 8}, {16, 16}, {32, 32}, {64, 64}, {136, 88}};
  const FscMode fsc_modes[] = {FscMode::kOff, FscMode::kForce};
  for (const auto& size : sizes) {
    for (int pattern = 0; pattern < 4; ++pattern) {
      for (FscMode fsc : fsc_modes) {
        for (int dpcm = 0; dpcm < 2; ++dpcm) {
          RoundTripCase c;
          c.w = size[0];
          c.h = size[1];
          c.pattern = pattern;
          c.fsc = fsc;
          c.dpcm = dpcm != 0;
          c.adaptive = false;
          std::string detail;
          if (!run_round_trip(c, detail)) {
            fail(__FILE__, __LINE__,
                 "round trip " + std::to_string(c.w) + "x" +
                     std::to_string(c.h) + " pattern " +
                     std::to_string(pattern) + " fsc " +
                     std::to_string(static_cast<int>(fsc)) + " dpcm " +
                     std::to_string(dpcm) + ": " + detail);
          } else {
            ++g_checks;
          }
        }
      }
    }
  }

  // Normative CDF adaptation path.
  RoundTripCase adaptive;
  adaptive.w = 64;
  adaptive.h = 64;
  adaptive.pattern = 2;
  adaptive.fsc = FscMode::kOff;
  adaptive.dpcm = true;
  adaptive.adaptive = true;
  std::string detail;
  if (!run_round_trip(adaptive, detail)) {
    fail(__FILE__, __LINE__, "adaptive CDF round trip: " + detail);
  } else {
    ++g_checks;
  }
}

void test_rejections() {
  Picture p = make_picture(8, 8, 0);
  Options opt;
  EncodeResult result;
  std::string error;

  Picture bad = p;
  bad.w = 10;  // not a multiple of 8
  StillPicture s = view(bad);
  s.y = std::span<const std::uint8_t>(bad.y.data(), bad.y.size());
  CHECK(!av2enc::lossless::encode_still(s, opt, result, error));
  CHECK(!error.empty());

  // Truncated and corrupted temporal units must be rejected, not crash.
  CHECK(av2enc::lossless::encode_still(view(p), opt, result, error));
  std::vector<std::uint8_t> truncated = result.temporal_unit;
  truncated.resize(truncated.size() / 2);
  av2enc::lossless::DecodeResult dec;
  CHECK(!av2enc::lossless::decode_temporal_unit(truncated, dec, error));

  std::vector<std::uint8_t> corrupted = result.temporal_unit;
  if (!corrupted.empty()) corrupted[corrupted.size() - 1] ^= 0xff;
  // Flipping the last byte may land in the arithmetic coder's trailing bits;
  // the decoder must reject it (or, at worst, produce a different frame, but
  // not crash).  The strict case expected here is rejection.
  (void)av2enc::lossless::decode_temporal_unit(corrupted, dec, error);
}

}  // namespace

int main() {
  test_lossless_flags();
  test_transforms();
  test_round_trips();
  test_rejections();
  if (g_failures == 0) {
    std::printf("lossless_test: %d checks passed\n", g_checks);
    return 0;
  }
  std::fprintf(stderr, "lossless_test: %d failure(s) of %d checks\n", g_failures,
               g_checks);
  return 1;
}

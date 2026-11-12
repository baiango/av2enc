// M1 MRL syntax / interaction / minimal-stream tests.
//
// Everything here is deliberately independent of the implementation it tests
// where it matters:
//   * NPos / context derivation has a literal §5.20.4.1 transcription;
//   * TxType mode mapping has a literal §5.20.7.29 transcription;
//   * writer/reader are checked against each other and against pinned bytes;
//   * the sequence-header copy is checked byte-for-byte against the M1 writer.

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "pipeline/still/still_headers.h"
#include "codec/intra/mrl/stream/mrl_interaction.h"
#include "codec/intra/mrl/stream/mrl_stream.h"
#include "codec/intra/mrl/stream/mrl_syntax.h"
#include "codec/transform/tx_type/tx_type.h"
#include "codec/entropy/av2/av2_entropy.h"

namespace {

namespace ms = av2enc::still::mrl_stream;
namespace av2 = av2enc::av2;

int g_checks = 0;
int g_failures = 0;

void fail(const char* what, int detail = 0) {
  ++g_checks;
  ++g_failures;
  if (g_failures <= 30) {
    std::printf("FAIL: %s (%d)\n", what, detail);
  }
}

void expect(bool condition, const char* what) {
  ++g_checks;
  if (!condition) fail(what);
}

void expect_eq_int(int got, int want, const char* what) {
  ++g_checks;
  if (got != want) {
    ++g_failures;
    if (g_failures <= 30) {
      std::printf("FAIL: %s (got %d, want %d)\n", what, got, want);
    }
  }
}

std::string hex(const std::vector<std::uint8_t>& bytes) {
  static const char* kDigits = "0123456789abcdef";
  std::string out;
  out.reserve(bytes.size() * 2);
  for (std::uint8_t b : bytes) {
    out.push_back(kDigits[b >> 4]);
    out.push_back(kDigits[b & 0x0F]);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Literal spec model
// ---------------------------------------------------------------------------

struct BlockSpec {
  int r = 0;
  int c = 0;
  int w4 = 0;
  int h4 = 0;
  int mode = 1;
  int mrl = 0;
  bool sec = false;
};

int log2_int(int v) {
  int out = 0;
  while (v > 1) {
    v >>= 1;
    ++out;
  }
  return out;
}

// §5.20.4.1 add_neighbor() over the four probes, written independently.
int naive_npos(const ms::MrlBlockSyntax& b, int mi_rows, int mi_cols,
               int sb_size_mi, int out[2][2]) {
  const int shift = log2_int(sb_size_mi);
  const int probes[4][2] = {
      {b.mi_row + b.bh4 - 1, b.mi_col - 1},
      {b.mi_row - 1, b.mi_col + b.bw4 - 1},
      {b.mi_row, b.mi_col - 1},
      {b.mi_row - 1, b.mi_col},
  };
  int count = 0;
  for (int i = 0; i < 4 && count < 2; ++i) {
    const int nr = probes[i][0];
    const int nc = probes[i][1];
    if (nr < 0 || nc < 0 || nr >= mi_rows || nc >= mi_cols) continue;
    if ((b.mi_row >> shift) != (nr >> shift)) continue;
    out[count][0] = nr;
    out[count][1] = nc;
    ++count;
  }
  return count;
}

int naive_index_context(const ms::MrlContext& ctx, const ms::MrlBlockSyntax& b,
                        int sb_size_mi) {
  int npos[2][2];
  const int num = naive_npos(b, ctx.mi_rows(), ctx.mi_cols(), sb_size_mi, npos);
  int sum = 0;
  for (int i = 0; i < num; ++i) {
    if (ctx.uses_at(npos[i][0], npos[i][1]) > 0) ++sum;
  }
  return sum;
}

int naive_sec_context(const ms::MrlContext& ctx, const ms::MrlBlockSyntax& b,
                      int sb_size_mi) {
  int npos[2][2];
  const int num = naive_npos(b, ctx.mi_rows(), ctx.mi_cols(), sb_size_mi, npos);
  int sum = 0;
  for (int i = 0; i < num; ++i) {
    if (ctx.uses_at(npos[i][0], npos[i][1]) == 2) ++sum;
  }
  return sum;
}

// Literal §5.20.7.29 wide_angle_mapping.
int naive_wide(int p, int mode, int w, int h) {
  if ((h == 2 * w && p < 61) || (h == 4 * w && p < 73) ||
      (h == 8 * w && p < 82) || (h == 16 * w && p < 86)) {
    return 7;  // D203_PRED
  }
  if ((w == 2 * h && p > 270 - 61) || (w == 4 * h && p > 270 - 73) ||
      (w == 8 * h && p > 270 - 82) || (w == 16 * h && p > 270 - 86)) {
    return 3;  // D45_PRED
  }
  return mode;
}

int naive_mode_angle(int mode) {
  static constexpr int kTable[13] = {0, 90, 180, 45, 135, 113, 157,
                                     203, 67, 0, 0, 0, 0};
  return kTable[mode];
}

int naive_mrl_delta(int mrl) {
  static constexpr int kTable[4] = {0, 1, -1, 0};
  return kTable[mrl];
}

// Reference values are written out again so a typo in the header is caught.
constexpr std::uint16_t kMrlIndexCdf_ref[3][5] = {
    {29573, 31193, 32023, 78, 0},
    {21812, 27066, 30279, 75, 0},
    {16076, 23806, 28762, 1, 0},
};
constexpr std::uint16_t kMrlSecIndexCdf_ref[3][3] = {
    {19678, 6, 0},
    {12287, 6, 0},
    {9574, 6, 0},
};

// Pinned writer bytes for golden_blocks() (see --dump-goldens).  These are
// filled once and frozen: they cover ctx 0/1/2, the secondary symbol, an SB-row
// boundary, D67 and a 4x4 leaf.
const std::string kGoldenFrozen = "dbdc16500000";
const std::string kGoldenAdaptive = "dbdceb500000";

// ---------------------------------------------------------------------------
// 1. Default CDFs (§9.3)
// ---------------------------------------------------------------------------
void test_default_cdfs() {
  for (int ctx = 0; ctx < ms::kMrlIndexContexts; ++ctx) {
    for (int i = 0; i < ms::kMrlIndexSymbols + 1; ++i) {
      if (kMrlIndexCdf_ref[ctx][i] != ms::kDefaultMrlIndexCdf[ctx][i]) {
        fail("Default_Mrl_Index_Cdf mismatch");
      }
    }
    for (int i = 0; i < 3; ++i) {
      if (kMrlSecIndexCdf_ref[ctx][i] != ms::kDefaultMrlSecIndexCdf[ctx][i]) {
        fail("Default_Mrl_Sec_Index_Cdf mismatch");
      }
    }
  }
  ++g_checks;

  ms::MrlCdfs cdfs;
  ms::init_mrl_cdfs(cdfs);
  expect(cdfs.index[0][0] == 29573, "init_mrl_cdfs index[0][0]");
  expect(cdfs.index[2][2] == 28762, "init_mrl_cdfs index[2][2]");
  expect(cdfs.index[2][3] == 1, "init_mrl_cdfs index[2][3]");
  expect(cdfs.sec[1][0] == 12287, "init_mrl_cdfs sec[1][0]");
  expect(cdfs.index[0][4] == 0 && cdfs.index[0][3] == 78,
         "init_mrl_cdfs rate/count tail");
}

// ---------------------------------------------------------------------------
// 2. NPos / contexts
// ---------------------------------------------------------------------------
void test_npos_and_contexts() {
  const int frames[][2] = {{64, 64}, {128, 128}, {192, 128}};
  const int sbs[] = {16, 32};
  const int dims4[] = {1, 2, 4, 8, 16};
  for (const auto& frame : frames) {
    const int mi_rows = frame[0] / 4;
    const int mi_cols = frame[1] / 4;
    for (int sb : sbs) {
      ms::MrlContext ctx(mi_rows, mi_cols, sb);
      for (int r = 0; r < mi_rows; r += 2) {
        for (int c = 0; c < mi_cols; c += 2) {
          for (int w4 : dims4) {
            for (int h4 : dims4) {
              if (r + h4 > mi_rows || c + w4 > mi_cols) continue;
              ms::MrlBlockSyntax b{r, c, w4, h4};
              ms::NeighborPos got[2];
              const int got_num = ms::collect_npos(b, mi_rows, mi_cols, sb, got);
              int want[2][2];
              const int want_num = naive_npos(b, mi_rows, mi_cols, sb, want);
              if (got_num != want_num) {
                fail("NNum mismatch");
                continue;
              }
              for (int i = 0; i < got_num; ++i) {
                if (got[i].row != want[i][0] || got[i].col != want[i][1]) {
                  fail("NPos order mismatch");
                }
              }
              expect_eq_int(ctx.index_context(b),
                            naive_index_context(ctx, b, sb), "mrl ctx");
              expect_eq_int(ctx.sec_context(b),
                            naive_sec_context(ctx, b, sb), "mrl sec ctx");
            }
          }
        }
      }
    }
  }

  // SB-boundary rule: a block on the first row of an SB has no above/above-
  // right neighbour for context purposes, so it can only see left/below-left.
  ms::MrlContext ctx(32, 32, 16);  // 128x128 frame
  ms::MrlBlockSyntax row16{16, 8, 4, 4};
  ms::NeighborPos npos[2];
  expect_eq_int(ms::collect_npos(row16, 32, 32, 16, npos), 2,
                "SB-row block sees below-left + left");
  ctx.update(ms::MrlBlockSyntax{16, 7, 1, 1}, 1, false);   // same-row left
  ctx.update(ms::MrlBlockSyntax{15, 7, 1, 1}, 2, true);    // above: excluded
  expect_eq_int(ctx.index_context(row16), 1, "SB row ignores above MRL");
  expect_eq_int(ctx.sec_context(row16), 0, "SB row ignores above secondary");

  // Directional-only availability is enforced by the writer/reader, not by
  // the context grid.
  expect(!ms::is_directional_y_mode(0), "DC is non-directional");
  expect(ms::is_directional_y_mode(1), "V is directional");
  expect(ms::is_directional_y_mode(8), "D67 is directional");
  expect(!ms::is_directional_y_mode(9), "SMOOTH is non-directional");
  expect(!ms::is_directional_y_mode(12), "PAETH is non-directional");
}

// ---------------------------------------------------------------------------
// 3. Writer / reader round-trip + validity
// ---------------------------------------------------------------------------
std::vector<BlockSpec> roundtrip_blocks() {
  std::vector<BlockSpec> blocks;
  const int modes[] = {1, 2, 3, 4, 5, 6, 7, 8, 0, 9, 10, 11, 12};
  int index = 0;
  for (int mode : modes) {
    for (int mrl = 0; mrl <= 3; ++mrl) {
      for (int sec = 0; sec <= 1; ++sec) {
        BlockSpec b;
        b.r = (index % 4) * 4;
        b.c = ((index / 4) % 4) * 4;
        b.w4 = 4;  // 16x16 (secondary is legal)
        b.h4 = 4;
        b.mode = mode;
        b.mrl = ms::is_directional_y_mode(mode) ? mrl : 0;
        b.sec = ms::is_directional_y_mode(mode) && b.mrl > 0 && sec != 0;
        blocks.push_back(b);
        ++index;
      }
    }
  }
  // 4x4 blocks never signal the secondary bit.
  for (int mrl = 1; mrl <= 3; ++mrl) {
    BlockSpec b;
    b.r = index % 12;
    b.c = (index * 5) % 12;
    b.w4 = 1;
    b.h4 = 1;
    b.mode = 8;
    b.mrl = mrl;
    b.sec = false;
    blocks.push_back(b);
    ++index;
  }
  // Every power-of-two luma size 4..64, square and rectangular.
  const int sizes[] = {1, 2, 4, 8, 16};
  for (int w4 : sizes) {
    for (int h4 : sizes) {
      BlockSpec b;
      b.r = 0;
      b.c = 0;
      b.w4 = w4;
      b.h4 = h4;
      b.mode = ((w4 * h4) & 1) != 0 ? 3 : 1;  // D45 or V
      b.mrl = static_cast<int>(index % 4);
      b.sec = b.mrl > 0 && !(w4 == 1 && h4 == 1) && (index & 1) != 0;
      blocks.push_back(b);
      ++index;
    }
  }
  return blocks;
}

struct RoundTrip {
  std::vector<std::uint8_t> bytes;
  std::vector<ms::MrlDecision> decisions;
  bool ok = false;
  std::string error;
};

RoundTrip roundtrip(const std::vector<BlockSpec>& blocks, bool adaptive,
                    int mi_rows, int mi_cols, int sb_mi) {
  RoundTrip out;
  ms::MrlCdfs cdfs;
  ms::init_mrl_cdfs(cdfs);
  ms::MrlContext context(mi_rows, mi_cols, sb_mi);
  std::vector<std::uint8_t> buffer(1U << 20);
  av2::Encoder encoder;
  if (encoder.init_symbol(buffer.data(), buffer.size()) != av2::Status::kOk) {
    out.error = "encoder init";
    return out;
  }
  encoder.disable_cdf_update = adaptive ? 0 : 1;
  for (const BlockSpec& b : blocks) {
    ms::MrlBlockSyntax block{b.r, b.c, b.w4, b.h4};
    const ms::MrlDecision decision{b.mrl, b.sec};
    const ms::MrlSyntaxStatus status =
        ms::write_mrl_syntax(encoder, cdfs, context, block, b.mode, decision,
                             &out.error);
    if (status != ms::MrlSyntaxStatus::kOk) {
      out.error = "write: " + out.error;
      return out;
    }
  }
  std::size_t payload_bytes = 0;
  if (encoder.exit_symbol(&payload_bytes) != av2::Status::kOk) {
    out.error = "encoder exit";
    return out;
  }
  out.bytes.assign(buffer.begin(),
                   buffer.begin() + static_cast<std::ptrdiff_t>(payload_bytes));

  ms::MrlCdfs read_cdfs;
  ms::init_mrl_cdfs(read_cdfs);
  ms::MrlContext read_context(mi_rows, mi_cols, sb_mi);
  av2::Decoder decoder;
  if (decoder.init_symbol(out.bytes.data(), out.bytes.size(), out.bytes.size()) !=
      av2::Status::kOk) {
    out.error = "decoder init";
    return out;
  }
  decoder.disable_cdf_update = adaptive ? 0 : 1;
  for (const BlockSpec& b : blocks) {
    ms::MrlBlockSyntax block{b.r, b.c, b.w4, b.h4};
    ms::MrlDecision decision;
    const ms::MrlSyntaxStatus status = ms::read_mrl_syntax(
        decoder, read_cdfs, read_context, block, b.mode, &decision, &out.error);
    if (status != ms::MrlSyntaxStatus::kOk) {
      out.error = "read: " + out.error;
      return out;
    }
    out.decisions.push_back(decision);
  }
  if (decoder.exit_symbol() != av2::Status::kOk) {
    out.error = "decoder exit";
    return out;
  }
  out.ok = true;
  return out;
}

void test_roundtrip() {
  const std::vector<BlockSpec> blocks = roundtrip_blocks();
  for (bool adaptive : {false, true}) {
    const RoundTrip rt = roundtrip(blocks, adaptive, 16, 16, 16);
    expect(rt.ok, "round-trip");
    if (!rt.ok) {
      std::printf("  %s\n", rt.error.c_str());
      continue;
    }
    expect_eq_int(static_cast<int>(rt.decisions.size()),
                  static_cast<int>(blocks.size()), "round-trip symbol count");
    for (std::size_t i = 0; i < blocks.size() && i < rt.decisions.size(); ++i) {
      expect_eq_int(rt.decisions[i].mrl_index, blocks[i].mrl, "rt mrl_index");
      expect(rt.decisions[i].mrl_sec_index == blocks[i].sec,
             "rt mrl_sec_index");
    }
  }

  // Validity rules: nothing is written before a rejected decision.
  ms::MrlCdfs cdfs;
  ms::init_mrl_cdfs(cdfs);
  ms::MrlContext context(16, 16, 16);
  std::vector<std::uint8_t> buffer(1U << 16);
  av2::Encoder encoder;
  encoder.init_symbol(buffer.data(), buffer.size());
  std::string error;
  ms::MrlBlockSyntax block{0, 0, 1, 1};  // 4x4
  expect(ms::write_mrl_syntax(encoder, cdfs, context, block, 0,
                              ms::MrlDecision{1, false}, &error) ==
             ms::MrlSyntaxStatus::kInvalidMode,
         "non-directional rejects MRL");
  expect(ms::write_mrl_syntax(encoder, cdfs, context, block, 1,
                              ms::MrlDecision{4, false}, &error) ==
             ms::MrlSyntaxStatus::kInvalidIndex,
         "mrl_index 4 rejected");
  expect(ms::write_mrl_syntax(encoder, cdfs, context, block, 1,
                              ms::MrlDecision{0, true}, &error) ==
             ms::MrlSyntaxStatus::kInvalidSecondary,
         "secondary with mrl_index 0 rejected");
  expect(ms::write_mrl_syntax(encoder, cdfs, context, block, 1,
                              ms::MrlDecision{2, true}, &error) ==
             ms::MrlSyntaxStatus::kInvalidSecondary,
         "secondary at 4x4 rejected by the writer");
  expect_eq_int(static_cast<int>(encoder.symbol_count), 0,
                "rejections wrote no symbols");
  expect(!ms::mrl_secondary_applies(4, 4, 2, true), "4x4 secondary ignored");
  expect(!ms::mrl_secondary_applies(8, 8, 0, true), "mrl0 secondary ignored");
  expect(ms::mrl_secondary_applies(8, 8, 1, true), "secondary applies");
  expect(ms::mrl_secondary_applies(4, 8, 1, true), "4x8 secondary applies");

  // The reader must accept a syntactically present 4x4 secondary bit (the
  // decoder ignores its effect); only the writer rejects it.
  {
    ms::MrlCdfs write_cdfs;
    ms::init_mrl_cdfs(write_cdfs);
    ms::MrlContext write_ctx(16, 16, 16);
    std::vector<std::uint8_t> buffer2(1U << 12);
    av2::Encoder encoder2;
    encoder2.init_symbol(buffer2.data(), buffer2.size());
    encoder2.disable_cdf_update = 1;
    expect(encoder2.encode_symbol(write_cdfs.index[0], 4, 2) ==
               av2::Status::kOk,
           "manual 4x4 index");
    expect(encoder2.encode_symbol(write_cdfs.sec[0], 2, 1) ==
               av2::Status::kOk,
           "manual 4x4 secondary");
    std::size_t bytes2 = 0;
    expect(encoder2.exit_symbol(&bytes2) == av2::Status::kOk,
           "manual 4x4 exit");
    std::vector<std::uint8_t> payload2(
        buffer2.begin(), buffer2.begin() + static_cast<std::ptrdiff_t>(bytes2));
    ms::MrlCdfs read_cdfs;
    ms::init_mrl_cdfs(read_cdfs);
    ms::MrlContext read_ctx(16, 16, 16);
    av2::Decoder decoder2;
    expect(decoder2.init_symbol(payload2.data(), payload2.size(),
                                payload2.size()) == av2::Status::kOk,
           "manual 4x4 decoder init");
    decoder2.disable_cdf_update = 1;
    ms::MrlDecision decoded;
    std::string read_error;
    expect(ms::read_mrl_syntax(decoder2, read_cdfs, read_ctx,
                               ms::MrlBlockSyntax{0, 0, 1, 1}, 1, &decoded,
                               &read_error) == ms::MrlSyntaxStatus::kOk,
           "reader accepts 4x4 secondary");
    expect_eq_int(decoded.mrl_index, 2, "4x4 secondary index");
    expect(decoded.mrl_sec_index, "4x4 secondary bit preserved");
    expect(!ms::mrl_secondary_applies(4, 4, decoded.mrl_index,
                                      decoded.mrl_sec_index),
           "4x4 decoded secondary has no effect");
  }
}

// ---------------------------------------------------------------------------
// 4. Pinned bit goldens (fixed context sequence)
// ---------------------------------------------------------------------------
std::vector<BlockSpec> golden_blocks() {
  return {
      {0, 0, 8, 8, 1, 0, false},    // 32x32 at origin, ctx 0
      {0, 8, 8, 8, 1, 1, true},     // top-right, ctx 1
      {8, 0, 8, 8, 1, 2, false},    // bottom-left, ctx 1 (above + left)
      {8, 8, 8, 8, 1, 3, true},     // bottom-right, ctx 2
      {16, 0, 8, 8, 1, 1, false},   // next SB row: above excluded
      {24, 24, 4, 4, 8, 2, false},  // D67 16x16
      {4, 4, 1, 1, 3, 0, false},    // 4x4 D45, no MRL
  };
}

void test_goldens() {
  const std::vector<BlockSpec> blocks = golden_blocks();
  for (bool adaptive : {false, true}) {
    const RoundTrip rt = roundtrip(blocks, adaptive, 32, 32, 16);
    expect(rt.ok, "golden round-trip");
    if (!rt.ok) continue;
    const std::string got = hex(rt.bytes);
    const std::string want =
        adaptive ? kGoldenAdaptive : kGoldenFrozen;
    if (got != want) {
      std::printf("  golden %s: got %s\n          want %s\n",
                  adaptive ? "adaptive" : "frozen", got.c_str(), want.c_str());
      fail("pinned MRL bit golden");
    } else {
      ++g_checks;
    }
  }
}

// ---------------------------------------------------------------------------
// 5. Edge-filter / TX-type interaction
// ---------------------------------------------------------------------------
void test_interaction() {
  const int sizes[][2] = {{4, 4},   {4, 8},   {8, 4},   {8, 8},   {8, 16},
                          {16, 8},  {16, 16}, {16, 32}, {32, 16}, {32, 32},
                          {16, 64}, {64, 16}, {32, 64}, {64, 32}, {64, 64}};
  for (int mrl = 0; mrl <= 3; ++mrl) {
    expect_eq_int(ms::mrl_angle_delta(mrl), naive_mrl_delta(mrl),
                  "Mrl_Index_To_Delta");
  }
  expect(ms::mrl_allows_edge_filter(0), "edge filter on line 0");
  expect(!ms::mrl_allows_edge_filter(1), "edge filter off line 1");
  expect(!ms::mrl_allows_edge_filter(3), "edge filter off line 3");

  for (int mode = 0; mode < 13; ++mode) {
    for (int delta = -3; delta <= 3; ++delta) {
      for (const auto& size : sizes) {
        const int w = size[0];
        const int h = size[1];
        const bool directional = mode >= 1 && mode <= 8;
        if (!directional) {
          expect_eq_int(ms::mapped_intra_mode_for_mrl(mode, delta, 0, w, h),
                        av2enc::txtype::map_intra_mode(mode, delta, w, h),
                        "non-directional map");
          expect_eq_int(ms::mapped_intra_mode_for_mrl(mode, delta, 1, w, h), -1,
                        "non-directional with MRL rejected");
          continue;
        }
        for (int mrl = 0; mrl <= 3; ++mrl) {
          const int p = naive_mode_angle(mode) + delta * 3 +
                        naive_mrl_delta(mrl);
          expect_eq_int(ms::mapped_intra_mode_for_mrl(mode, delta, mrl, w, h),
                        naive_wide(p, mode, w, h), "MRL-aware mode map");
        }
        expect_eq_int(ms::mapped_intra_mode_for_mrl(mode, delta, 0, w, h),
                      av2enc::txtype::map_intra_mode(mode, delta, w, h),
                      "mrl 0 == txtype::map_intra_mode");
      }
    }
  }

  // Edge filter must be off for every mrl_index > 0 regardless of the
  // sequence flag; line 0 keeps the sequence setting.
  for (int mrl = 0; mrl <= 3; ++mrl) {
    av2enc::mrl::Params params;
    expect(ms::make_predict_params(1, 0, mrl, false, true, false, false,
                                   &params, 16, 16),
           "make_predict_params");
    expect_eq_int(params.enable_edge_filter ? 1 : 0, mrl == 0 ? 1 : 0,
                  "edge filter follows MrlIndex");
  }
  av2enc::mrl::Params params;
  expect(ms::make_predict_params(1, 0, 0, false, false, false, false, &params),
         "make_predict_params seq off");
  expect(!params.enable_edge_filter, "sequence flag disables edge filter");

  // TX type query + symbol path with MRL.
  for (int mode = 1; mode <= 8; ++mode) {
    for (int mrl = 0; mrl <= 3; ++mrl) {
      av2enc::txtype::TxTypeQuery query;
      expect(ms::make_tx_type_query(mode, 0, mrl, 8, 8, 0, false, &query),
             "make_tx_type_query");
      expect_eq_int(query.mapped_mode,
                    ms::mapped_intra_mode_for_mrl(mode, 0, mrl, 8, 8),
                    "query mapped mode");
      int symbol = -1;
      bool long_side = false;
      expect(ms::tx_type_symbol_for_mrl(mode, 0, mrl, 8, 8, 0, false,
                                        av2enc::txtype::TxType::kDctDct,
                                        &symbol, &long_side),
             "DCT_DCT is codable with MRL");
      expect(symbol >= 0, "DCT_DCT has a symbol in INTRA_1");
    }
  }
  // A directional narrow block stays in the same set at mrl 0..3; a 32x8
  // rectangular block exercises wide-angle remapping.
  for (int mrl = 0; mrl <= 3; ++mrl) {
    av2enc::txtype::TxTypeQuery query;
    expect(ms::make_tx_type_query(1, 0, mrl, 32, 8, 0, false, &query),
           "32x8 query");
    expect_eq_int(query.mapped_mode, naive_wide(90 + naive_mrl_delta(mrl), 1,
                                                32, 8),
                  "32x8 mapped mode");
  }
}

// ---------------------------------------------------------------------------
// 6. Sequence header copy + minimal stream shape
// ---------------------------------------------------------------------------
void test_sequence_header() {
  for (int w : {64, 128, 640}) {
    av2enc::still::StillConfig config;
    config.max_frame_width = static_cast<std::uint16_t>(w);
    config.max_frame_height = static_cast<std::uint16_t>(w);
    const std::vector<std::uint8_t> core =
        av2enc::still::make_sequence_header_obu(config);
    const std::vector<std::uint8_t> off =
        ms::make_sequence_header_obu(config, false);
    expect(core == off, "MRL header copy == still_headers with enable_mrls=0");

    const std::vector<std::uint8_t> on =
        ms::make_sequence_header_obu(config, true);
    expect(on.size() == off.size(), "enable_mrls does not change length");
    int differing_bits = 0;
    for (std::size_t i = 0; i < on.size() && i < off.size(); ++i) {
      std::uint8_t diff = static_cast<std::uint8_t>(on[i] ^ off[i]);
      while (diff != 0) {
        differing_bits += diff & 1U;
        diff = static_cast<std::uint8_t>(diff >> 1U);
      }
    }
    expect_eq_int(differing_bits, 1, "enable_mrls flips exactly one bit");
  }
}

void test_minimal_stream() {
  for (int target : {4, 8, 16, 32, 64}) {
    ms::MinimalMrlConfig config;
    config.width = 64;
    config.height = 64;
    config.target_block_size = target;
    ms::MinimalMrlFrame frame;
    const bool ok = ms::write_minimal_mrl_frame(config, frame);
    if (!ok) {
      std::printf("  target %d: %s\n", target, frame.error.c_str());
    }
    expect(ok, "minimal 64x64 stream");
    if (!ok) continue;
    expect_eq_int(frame.blocks, (64 / target) * (64 / target),
                  "minimal leaf count");
    expect(!frame.temporal_unit.empty(), "temporal unit present");
    expect(!frame.tile_payload.empty(), "tile payload present");
    expect(frame.recon_y.size() == 64U * 64U, "expected luma size");
    if (target == 64) {
      expect_eq_int(frame.mrl_blocks, 0, "origin-only frame defaults to mrl 0");
    } else {
      expect(frame.mrl_blocks > 0, "some MRL blocks");
    }
    if (target == 4 || target == 64) {
      expect_eq_int(frame.secondary_blocks, 0,
                    "single/4x4 blocks never signal secondary");
    } else {
      expect(frame.secondary_blocks > 0, "secondary exercised");
    }
  }

  // SB-boundary coverage: 128x128 with 64x64 SBs, 32x32 leaves.
  ms::MinimalMrlConfig config;
  config.width = 128;
  config.height = 128;
  config.target_block_size = 32;
  ms::MinimalMrlFrame frame;
  expect(ms::write_minimal_mrl_frame(config, frame), "minimal 128x128 stream");
  bool saw_sb_row = false;
  for (const ms::MinimalMrlLeafRecord& leaf : frame.leaves) {
    if (leaf.y == 64 && leaf.leaf.mrl_index > 0) saw_sb_row = true;
  }
  expect(saw_sb_row, "MRL used at an SB-row boundary");

  // Explicit decisions: DC + MRL is rejected; the count must match.
  ms::MinimalMrlConfig bad;
  bad.width = 64;
  bad.height = 64;
  bad.target_block_size = 32;
  bad.leaves.resize(4);
  bad.leaves[0].y_mode = 0;   // DC_PRED
  bad.leaves[0].mrl_index = 1;
  ms::MinimalMrlFrame bad_frame;
  expect(!ms::write_minimal_mrl_frame(bad, bad_frame),
         "stream rejects non-directional MRL");

  ms::MinimalMrlConfig short_list;
  short_list.width = 64;
  short_list.height = 64;
  short_list.target_block_size = 32;
  short_list.leaves.resize(3);
  ms::MinimalMrlFrame short_frame;
  expect(!ms::write_minimal_mrl_frame(short_list, short_frame),
         "explicit decision count is validated");

  // A DC-only frame is valid with enable_mrls = 1 and codes no MRL symbols.
  ms::MinimalMrlConfig dc;
  dc.width = 64;
  dc.height = 64;
  dc.target_block_size = 64;
  dc.leaves.resize(1);
  dc.leaves[0].y_mode = 0;
  dc.leaves[0].mrl_index = 0;
  ms::MinimalMrlFrame dc_frame;
  expect(ms::write_minimal_mrl_frame(dc, dc_frame), "DC-only MRL-enabled stream");
  expect_eq_int(dc_frame.mrl_blocks, 0, "DC-only has no MRL blocks");

  // A single 64x64 leaf at the frame origin with MRL > 0 exercises the
  // unavailable-neighbour fillers on both the primary and secondary lines.
  ms::MinimalMrlConfig origin;
  origin.width = 64;
  origin.height = 64;
  origin.target_block_size = 64;
  origin.leaves.resize(1);
  origin.leaves[0].y_mode = 1;
  origin.leaves[0].mrl_index = 3;
  origin.leaves[0].mrl_sec_index = true;
  ms::MinimalMrlFrame origin_frame;
  expect(ms::write_minimal_mrl_frame(origin, origin_frame),
         "origin MRL + secondary stream");
  expect_eq_int(origin_frame.mrl_blocks, 1, "origin MRL block counted");
  expect_eq_int(origin_frame.secondary_blocks, 1, "origin secondary counted");
}

// ---------------------------------------------------------------------------
// 7. Optional golden dump
// ---------------------------------------------------------------------------
void dump_goldens() {
  const std::vector<BlockSpec> blocks = golden_blocks();
  for (bool adaptive : {false, true}) {
    const RoundTrip rt = roundtrip(blocks, adaptive, 32, 32, 16);
    if (!rt.ok) {
      std::printf("dump failed: %s\n", rt.error.c_str());
      continue;
    }
    std::printf("%s=%s\n", adaptive ? "kGoldenAdaptive" : "kGoldenFrozen",
                hex(rt.bytes).c_str());
  }
}

}  // namespace

int main(int argc, char** argv) {
  bool dump = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--dump-goldens") == 0) dump = true;
  }
  if (dump) {
    dump_goldens();
    return 0;
  }

  test_default_cdfs();
  test_npos_and_contexts();
  test_roundtrip();
  test_goldens();
  test_interaction();
  test_sequence_header();
  test_minimal_stream();

  if (g_failures == 0) {
    std::printf("mrl_stream_test: PASS (%d checks)\n", g_checks);
    return 0;
  }
  std::printf("mrl_stream_test: FAIL (%d failures / %d checks)\n", g_failures,
              g_checks);
  return 1;
}

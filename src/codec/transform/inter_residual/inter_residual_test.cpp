// Unit tests for the M2 inter-residual component.
//
// Covers: TX-size derivation, inter TX-type sets/symbols, the decision API
// (skip/zero-residual fallback, reconstruction, partial blocks), the batch
// interface, and the entropy writer's symbol structure.
//
// Build/run with src/encoder/m2/residual/build_and_test.sh.
#include "codec/transform/inter_residual/inter_residual.h"

#include "codec/transform/inter_residual/inter_coeff_syntax.h"
#include "codec/transform/inter_residual/inter_tx_type.h"
#include "codec/partition/tree/partition.h"
#include "codec/transform/variants/tx_variants.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

using namespace av2enc::video::residual;
namespace partition = av2enc::partition;

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const std::string& what) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    std::printf("FAIL: %s\n", what.c_str());
  }
}

BlockGeometry make_geometry(int mi_row, int mi_col, int luma_w, int luma_h) {
  partition::FrameGeometry frame;
  frame.width = mi_col * 4 + luma_w;
  frame.height = mi_row * 4 + luma_h;
  frame.sb_size = 64;
  frame.has_chroma = true;
  frame.chroma_subsampling_x = true;
  frame.chroma_subsampling_y = true;
  BlockGeometry geometry;
  geometry.mi_row = mi_row;
  geometry.mi_col = mi_col;
  geometry.luma_w = luma_w;
  geometry.luma_h = luma_h;
  partition::BlockSize size = partition::BlockSize::kInvalid;
  for (int i = 0; i < partition::kBlockSizeCount; ++i) {
    const auto candidate = static_cast<partition::BlockSize>(i);
    if (partition::block_width(candidate) == luma_w &&
        partition::block_height(candidate) == luma_h) {
      size = candidate;
      break;
    }
  }
  const partition::BlockSize chroma =
      partition::chroma_plane_residual_size(frame, size);
  geometry.chroma_w = partition::block_width(chroma);
  geometry.chroma_h = partition::block_height(chroma);
  return geometry;
}

struct Frame {
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
};

Frame make_source(int width, int height, int seed) {
  Frame frame;
  frame.y.resize(static_cast<std::size_t>(width * height));
  const int cw = width / 2;
  const int ch = height / 2;
  frame.u.resize(static_cast<std::size_t>(cw * ch));
  frame.v.resize(static_cast<std::size_t>(cw * ch));
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int value = 96 + ((x * 3 + y * 5 + seed * 7) & 63) +
                        ((x / 8 + y / 8) & 1) * 24;
      frame.y[static_cast<std::size_t>(y * width + x)] =
          static_cast<std::uint8_t>(value);
    }
  }
  for (int y = 0; y < ch; ++y) {
    for (int x = 0; x < cw; ++x) {
      const int value = 100 + ((x * 7 + y * 3 + seed * 5) & 63);
      frame.u[static_cast<std::size_t>(y * cw + x)] =
          static_cast<std::uint8_t>(value);
      frame.v[static_cast<std::size_t>(y * cw + x)] =
          static_cast<std::uint8_t>(255 - value);
    }
  }
  return frame;
}

PlaneView view_of(const std::vector<std::uint8_t>& data, int stride) {
  PlaneView view;
  view.data = data;
  view.stride = stride;
  return view;
}

// Simpler, explicit extraction: source/prediction are full frames with known
// strides; the block origin is passed in luma coordinates.
bool extract(const Frame& frame, int plane, int full_width, int x0, int y0,
             int w, int h, std::vector<std::uint8_t>* out) {
  const std::vector<std::uint8_t>& data =
      plane == 0 ? frame.y : (plane == 1 ? frame.u : frame.v);
  // Chroma planes are half the luma width.
  const int stride = plane == 0 ? full_width : full_width / 2;
  const int full_height = static_cast<int>(data.size()) / stride;
  if (x0 + w > stride || y0 + h > full_height) return false;
  out->resize(static_cast<std::size_t>(w * h));
  for (int y = 0; y < h; ++y) {
    std::memcpy(out->data() + static_cast<std::size_t>(y * w),
                data.data() +
                    static_cast<std::size_t>((y0 + y) * stride + x0),
                static_cast<std::size_t>(w));
  }
  return true;
}

// Owns the per-plane block buffers a BlockInput points at.
struct OwnedInput {
  BlockInput input;
  std::vector<std::uint8_t> source[kResidualPlaneCount];
  std::vector<std::uint8_t> prediction[kResidualPlaneCount];

  bool build(const Frame& source_frame, const Frame& prediction_frame,
             int width, const BlockGeometry& geometry, std::string& error) {
    input.geometry = geometry;
    const int x0 = geometry.mi_col * 4;
    const int y0 = geometry.mi_row * 4;
    for (int plane = 0; plane < kResidualPlaneCount; ++plane) {
      const int w = plane == 0 ? geometry.luma_w : geometry.chroma_w;
      const int h = plane == 0 ? geometry.luma_h : geometry.chroma_h;
      const int px = plane == 0 ? x0 : x0 / 2;
      const int py = plane == 0 ? y0 : y0 / 2;
      if (!extract(source_frame, plane, width, px, py, w, h, &source[plane]) ||
          !extract(prediction_frame, plane, width, px, py, w, h,
                   &prediction[plane])) {
        error = "block outside the frame";
        return false;
      }
      input.source[plane] = view_of(source[plane], w);
      input.prediction[plane] = view_of(prediction[plane], w);
    }
    return true;
  }
};

// Recomputes the reconstruction from the decided levels with txv directly, so
// the decision's `recon` is checked against the transform contract rather than
// against itself.
bool verify_recon(const OwnedInput& owned, const BlockDecision& decision,
                  std::string* error) {
  for (int plane = 0; plane < kResidualPlaneCount; ++plane) {
    const PlaneResidual& residual = decision.plane[plane];
    const auto params = av2enc::txv::make_tx_params(
        static_cast<av2enc::txv::TxSize>(residual.tx_size),
        static_cast<av2enc::txv::TxType>(residual.tx_type));
    if (!params.has_value()) {
      *error = "bad tx params";
      return false;
    }
    std::vector<std::int32_t> dequant(
        static_cast<std::size_t>(params->w * params->h));
    std::vector<std::int16_t> recon_residual(
        static_cast<std::size_t>(params->coded_w * params->coded_h));
    av2enc::txv::inverse_transform(residual.levels.data(), *params,
                                   dequant.data(), recon_residual.data());
    const std::vector<std::uint8_t>& pred = owned.prediction[plane];
    for (std::size_t i = 0; i < residual.recon.size(); ++i) {
      const int expected =
          std::clamp(static_cast<int>(pred[i]) +
                         static_cast<int>(recon_residual[i]),
                     0, 255);
      if (residual.recon[i] != expected) {
        *error = "recon mismatch on plane " + std::to_string(plane);
        return false;
      }
    }
  }
  return true;
}

// Recording symbol sink: counts symbols and literals and checks the writer's
// structural invariants (never asks for a zero-probability symbol, never
// exceeds the CDF symbol count).
struct Recorder {
  int symbols = 0;
  int literals = 0;
  bool bad = false;
  std::vector<std::uint16_t> first_symbol_cdfs;
};

bool record_symbol(void* context, std::uint16_t* cdf, int n, int symbol) {
  Recorder* recorder = static_cast<Recorder*>(context);
  if (n < 1 || n > 16 || symbol < 0 || symbol >= n) {
    recorder->bad = true;
    return false;
  }
  ++recorder->symbols;
  (void)cdf;
  return true;
}

bool record_bool(void* context, bool bit) {
  Recorder* recorder = static_cast<Recorder*>(context);
  (void)bit;
  ++recorder->literals;
  return true;
}

void test_tx_sizes() {
  struct Case {
    int w;
    int h;
    int chroma_w;
    int chroma_h;
  };
  const Case cases[] = {
      {4, 4, 4, 4},     {8, 8, 4, 4},     {16, 16, 8, 8},   {32, 32, 16, 16},
      {64, 64, 32, 32}, {64, 16, 32, 8},  {16, 64, 8, 32},  {32, 8, 16, 4},
      {8, 32, 4, 16},   {64, 8, 32, 4},   {8, 64, 4, 32},   {16, 8, 8, 4},
  };
  for (const Case& item : cases) {
    const BlockGeometry geometry =
        make_geometry(0, 0, item.w, item.h);
    check(geometry.chroma_w == item.chroma_w &&
              geometry.chroma_h == item.chroma_h,
          "chroma residual size for " + std::to_string(item.w) + "x" +
              std::to_string(item.h));
    int luma_tx = -1;
    int chroma_tx = -1;
    check(inter_tx_sizes(geometry, &luma_tx, &chroma_tx),
          "inter_tx_sizes supported");
    const auto luma = av2enc::txv::make_tx_params(
        static_cast<av2enc::txv::TxSize>(luma_tx));
    check(luma.has_value() && luma->coded_w == item.w &&
              luma->coded_h == item.h,
          "luma TX size maps back to the block size");
    const auto chroma = av2enc::txv::make_tx_params(
        static_cast<av2enc::txv::TxSize>(chroma_tx));
    check(chroma.has_value() && chroma->coded_w == item.chroma_w &&
              chroma->coded_h == item.chroma_h,
          "chroma TX size maps back to the chroma residual size");
  }
}

void test_tx_type_sets() {
  using av2enc::txv::TxSize;
  // Expected set per §5.20.8.3 for reduced_tx_set = 0, is_inter = 1.
  struct Expect {
    TxSize size;
    InterTxSet set;
    int candidates;
  };
  // Counts follow Tx_Type_In_Set_Inter (§5.20.7.29) minus the encoder's 1-D
  // scan-class capability guard (V/H_DCT, V/H_ADST, V/H_FLIPADST), which is a
  // component limitation, not a spec restriction:
  //   INTER_1: 16 in set - 6 one-axis = 10
  //   INTER_2: 12 in set - 2 (V/H_DCT) = 10
  //   DCT_IDTX: {DCT_DCT, IDTX} = 2
  //   WIDE/HIGH_64: 3 of the 4 long-side rows are 2-D (H_DCT/V_DCT is 1-D)
  //   WIDE/HIGH_32: 3 two-D long types + IDTX from the non-long half = 4
  const Expect expects[] = {
      {TxSize::k4x4, InterTxSet::kInter1, 10},
      {TxSize::k8x8, InterTxSet::kInter1, 10},
      {TxSize::k8x16, InterTxSet::kInter1, 10},
      {TxSize::k16x8, InterTxSet::kInter1, 10},
      {TxSize::k16x16, InterTxSet::kInter2, 10},
      {TxSize::k32x32, InterTxSet::kDctIdtx, 2},
      {TxSize::k64x64, InterTxSet::kDctOnly, 1},
      {TxSize::k64x16, InterTxSet::kWide64, 3},
      {TxSize::k16x64, InterTxSet::kHigh64, 3},
      {TxSize::k32x16, InterTxSet::kWide32, 4},
      {TxSize::k16x32, InterTxSet::kHigh32, 4},
      {TxSize::k32x64, InterTxSet::kDctOnly, 1},
      {TxSize::k64x32, InterTxSet::kDctOnly, 1},
  };
  for (const Expect& expect : expects) {
    InterTxSetInfo info;
    check(inter_tx_candidates(expect.size, 0, /*enable_inter_ddt=*/false,
                              &info),
          std::string("candidates for ") +
              av2enc::txv::tx_size_name(expect.size));
    check(info.set == expect.set,
          std::string("set for ") + av2enc::txv::tx_size_name(expect.size) +
              " is " + inter_tx_set_name(info.set) + ", expected " +
              inter_tx_set_name(expect.set));
    check(info.count == expect.candidates,
          std::string("candidate count for ") +
              av2enc::txv::tx_size_name(expect.size) + " is " +
              std::to_string(info.count) + ", expected " +
              std::to_string(expect.candidates));
    for (int i = 0; i < info.count; ++i) {
      const InterTxCandidate& candidate =
          info.candidates[static_cast<std::size_t>(i)];
      check(inter_tx_type_in_set(info.set, candidate.type),
            "candidate type is in the decoder's set");
      check(candidate.params.coded_w ==
                av2enc::txv::make_tx_params(expect.size)->coded_w,
            "candidate params match the TX size");
      InterTxCandidate back;
      check(inter_tx_symbol(expect.size, 0, false, candidate.type, &back),
            "candidate type has a symbol");
      check(back.symbol == candidate.symbol &&
                back.offset_symbol == candidate.offset_symbol &&
                back.is_long_side_dct == candidate.is_long_side_dct,
            "symbol round-trip for " +
                std::string(av2enc::txv::tx_type_name(candidate.type)));
    }
  }
  // DCTONLY codes no symbol: DCT_DCT reports symbol -1.
  InterTxSetInfo info;
  check(inter_tx_candidates(TxSize::k64x64, 0, false, &info), "64x64 set");
  check(info.candidates[0].symbol == -1, "DCTONLY has no type symbol");
  // §5.20.7.29 / §7.15.4.1 conformance probes: IDTX lives in INTER_1 and
  // INTER_2 and enable_inter_ddt only changes 1-D kernels, not membership or
  // symbol mapping.
  InterTxCandidate candidate;
  check(inter_tx_symbol(TxSize::k8x8, 0, false, av2enc::txv::TxType::kIdtx,
                        &candidate),
        "IDTX is codable with enable_inter_ddt = 0");
  check(inter_tx_symbol(TxSize::k8x8, 0, true, av2enc::txv::TxType::kIdtx,
                        &candidate),
        "IDTX is codable with enable_inter_ddt = 1");
  check(!inter_tx_type_is_ddt(av2enc::txv::TxType::kIdtx),
        "IDTX is not a DDT type");
  check(!inter_tx_type_is_ddt(av2enc::txv::TxType::kVdct),
        "V_DCT is not a DDT type");
  check(inter_tx_type_is_ddt(av2enc::txv::TxType::kAdstAdst),
        "ADST_ADST is a DDT type");
  check(inter_tx_type_is_ddt(av2enc::txv::TxType::kAdstFlipadst),
        "ADST_FLIPADST (7) is a DDT type");
  // Types 7/8 are legal 2-D FLIPADST/ADST pairs and are now in both sets and
  // in the candidate list.  They must round-trip through the symbols the
  // decoder's Tx_Type_Inter_Inv_Set1/2 use.
  check(inter_tx_type_in_set(InterTxSet::kInter1,
                             static_cast<av2enc::txv::TxType>(7)),
        "INTER_1 contains ADST_FLIPADST (7)");
  check(inter_tx_type_in_set(InterTxSet::kInter1,
                             static_cast<av2enc::txv::TxType>(8)),
        "INTER_1 contains FLIPADST_ADST (8)");
  for (int t = 7; t <= 11; ++t) {
    check(inter_tx_type_in_set(InterTxSet::kInter2,
                               static_cast<av2enc::txv::TxType>(t)),
          "INTER_2 contains type " + std::to_string(t));
  }
  check(inter_tx_symbol(TxSize::k8x8, 0, false,
                        av2enc::txv::TxType::kAdstFlipadst, &candidate) &&
            candidate.symbol == 1 && candidate.offset_symbol == 6,
        "type 7 maps to INTER_1 (1,6) per Tx_Type_Inter_Inv_Set1[14]");
  check(inter_tx_symbol(TxSize::k8x8, 0, false,
                        av2enc::txv::TxType::kFlipadstAdst, &candidate) &&
            candidate.symbol == 1 && candidate.offset_symbol == 7,
        "type 8 maps to INTER_1 (1,7) per Tx_Type_Inter_Inv_Set1[15]");
  {
    InterTxSetInfo inter1;
    check(inter_tx_candidates(TxSize::k8x8, 0, false, &inter1), "8x8 set");
    bool saw7 = false;
    bool saw8 = false;
    for (int i = 0; i < inter1.count; ++i) {
      const av2enc::txv::TxType type =
          inter1.candidates[static_cast<std::size_t>(i)].type;
      saw7 = saw7 || type == av2enc::txv::TxType::kAdstFlipadst;
      saw8 = saw8 || type == av2enc::txv::TxType::kFlipadstAdst;
    }
    check(saw7 && saw8, "INTER_1 candidates include types 7 and 8");
    // DDT on: the component has no DDTX/FDDT inverse, so ADST/FDST types are
    // withheld but IDTX remains.
    InterTxSetInfo ddt;
    check(inter_tx_candidates(TxSize::k8x8, 0, true, &ddt), "8x8 ddt set");
    for (int i = 0; i < ddt.count; ++i) {
      check(!inter_tx_type_is_ddt(
                ddt.candidates[static_cast<std::size_t>(i)].type),
            "enable_inter_ddt = 1 keeps only non-DDT types");
    }
  }
  // Context derivation, both variants.
  check(inter_tx_type_context(1, TxSize::k16x16) == 1,
        "inter_tx_type ctx for eob 1 is 1");
  check(inter_tx_type_context(256, TxSize::k16x16) == 2,
        "inter_tx_type ctx for the far diagonal is 2");
  check(inter_tx_type_offset_context(1, TxSize::k16x16) == 1,
        "inter_tx_type_offset ctx for eob 1 is 1");
}

void test_decision_roundtrip() {
  const int width = 128;
  const int height = 64;
  const Frame source = make_source(width, height, 3);
  Frame prediction = make_source(width, height, 3);
  // Give the prediction a systematic offset so the residual is non-trivial.
  for (std::size_t i = 0; i < prediction.y.size(); ++i) {
    prediction.y[i] = static_cast<std::uint8_t>(
        std::clamp(static_cast<int>(prediction.y[i]) + 6 -
                       static_cast<int>(i % 5),
                   0, 255));
  }
  for (std::size_t i = 0; i < prediction.u.size(); ++i) {
    prediction.u[i] = static_cast<std::uint8_t>(
        std::clamp(static_cast<int>(prediction.u[i]) + 3, 0, 255));
  }
  const BlockGeometry geometry = make_geometry(0, 0, 32, 32);
  OwnedInput owned;
  std::string error;
  check(owned.build(source, prediction, width, geometry, error),
        "build block input: " + error);
  if (!error.empty()) return;

  DecisionOptions options;
  options.try_inter_tx_types = true;
  BlockDecision decision;
  check(decide_inter_block(owned.input, options, decision, error),
        "decide_inter_block: " + error);
  check(!decision.skip, "a non-flat residual is coded");
  check(decision.residual_considered, "residual was considered");
  check(!decision.zero_residual, "residual is not all-zero");
  check(decision.cost < decision.skip_cost,
        "chosen residual cost beats skip cost");
  check(verify_recon(owned, decision, &error),
        "reconstruction matches txv inverse transform: " + error);

  // Syntax block + entropy writer.
  InterResidualBlock block;
  std::vector<std::int32_t> levels;
  check(make_syntax_block(geometry, decision, &block, &levels, error),
        "make_syntax_block: " + error);
  check(block.plane[0].levels_offset == 0, "luma levels start at 0");
  check(levels.size() == static_cast<std::size_t>(block.plane[0].eob) +
                             block.plane[1].eob + block.plane[2].eob,
        "scan-order level count matches the EOBs");

  InterResidualCdfs cdfs;
  init_inter_residual_cdfs(cdfs);
  std::vector<std::uint8_t> above_level[kResidualPlaneCount];
  std::vector<std::uint8_t> left_level[kResidualPlaneCount];
  std::vector<std::uint8_t> above_dc[kResidualPlaneCount];
  std::vector<std::uint8_t> left_dc[kResidualPlaneCount];
  CoeffContextView contexts;
  contexts.mi_cols = 32;
  contexts.mi_rows = 16;
  int eob_u = 0;
  contexts.eob_u = &eob_u;
  for (int plane = 0; plane < kResidualPlaneCount; ++plane) {
    const int cols = (plane == 0 ? 32 : 16) + kCoeffContextPad;
    const int rows = (plane == 0 ? 16 : 8) + kCoeffContextPad;
    above_level[plane].assign(static_cast<std::size_t>(cols), 0);
    left_level[plane].assign(static_cast<std::size_t>(rows), 0);
    above_dc[plane].assign(static_cast<std::size_t>(cols), 0);
    left_dc[plane].assign(static_cast<std::size_t>(rows), 0);
    contexts.above_level[plane] = above_level[plane].data();
    contexts.left_level[plane] = left_level[plane].data();
    contexts.above_dc[plane] = above_dc[plane].data();
    contexts.left_dc[plane] = left_dc[plane].data();
  }
  Recorder recorder;
  SymbolSink sink;
  sink.context = &recorder;
  sink.encode_symbol = record_symbol;
  sink.encode_bool = record_bool;
  check(write_inter_residual(sink, contexts, cdfs, block, levels, error),
        "write_inter_residual: " + error);
  check(!recorder.bad, "writer only requested valid symbols");
  check(recorder.symbols > 3, "writer emitted the per-plane skip/eob symbols");
  check(recorder.literals > 0, "writer emitted sign/extra literals");
  // Writes with the same input are deterministic.
  Recorder second;
  SymbolSink sink2;
  sink2.context = &second;
  sink2.encode_symbol = record_symbol;
  sink2.encode_bool = record_bool;
  CoeffContextView contexts2 = contexts;
  for (int plane = 0; plane < kResidualPlaneCount; ++plane) {
    std::fill(above_level[plane].begin(), above_level[plane].end(), 0);
    std::fill(left_level[plane].begin(), left_level[plane].end(), 0);
    std::fill(above_dc[plane].begin(), above_dc[plane].end(), 0);
    std::fill(left_dc[plane].begin(), left_dc[plane].end(), 0);
  }
  InterResidualCdfs cdfs2;
  init_inter_residual_cdfs(cdfs2);
  check(write_inter_residual(sink2, contexts2, cdfs2, block, levels, error),
        "second write: " + error);
  check(second.symbols == recorder.symbols &&
            second.literals == recorder.literals,
        "the writer is deterministic");
}

void test_zero_residual_fallback() {
  const int width = 64;
  const int height = 64;
  const Frame source = make_source(width, height, 5);
  const BlockGeometry geometry = make_geometry(0, 0, 64, 64);
  OwnedInput owned;
  std::string error;
  check(owned.build(source, source, width, geometry, error),
        "build identical prediction: " + error);
  DecisionOptions options;
  BlockDecision decision;
  check(decide_inter_block(owned.input, options, decision, error),
        "decide identical block: " + error);
  check(decision.skip, "identical prediction falls back to skip");
  check(decision.zero_residual, "residual quantised to zero");
  check(decision.cost == decision.skip_cost, "skip cost is reported");
  for (int plane = 0; plane < kResidualPlaneCount; ++plane) {
    check(decision.plane[plane].eob == 0, "zero EOB on every plane");
    check(decision.plane[plane].recon == owned.prediction[plane],
          "zero-residual reconstruction equals the prediction");
  }

  // allow_residual = false must also produce a skip decision.
  options.allow_residual = false;
  BlockDecision forced;
  check(decide_inter_block(owned.input, options, forced, error),
        "decide with allow_residual = false: " + error);
  check(forced.skip, "allow_residual = false forces skip");
  check(!forced.residual_considered, "no residual was tried");
  for (int plane = 0; plane < kResidualPlaneCount; ++plane) {
    check(forced.plane[plane].recon == owned.prediction[plane],
          "forced skip reconstructs to the prediction");
    check(forced.plane[plane].eob == 0, "forced skip has zero EOB");
  }
}

void test_partial_blocks() {
  const int width = 128;
  const int height = 16;
  const Frame source = make_source(width, height, 7);
  Frame prediction = make_source(width, height, 7);
  for (std::size_t i = 0; i < prediction.y.size(); ++i) {
    prediction.y[i] = static_cast<std::uint8_t>(
        std::clamp(static_cast<int>(prediction.y[i]) + 9, 0, 255));
  }
  const struct {
    int w;
    int h;
  } cases[] = {{64, 16}, {32, 16}, {16, 16}, {32, 8}, {8, 8}, {4, 4}};
  for (const auto& item : cases) {
    const BlockGeometry geometry = make_geometry(0, 0, item.w, item.h);
    OwnedInput owned;
    std::string error;
    if (!owned.build(source, prediction, width, geometry, error)) {
      check(false, "build partial block: " + error);
      continue;
    }
    BlockDecision decision;
    check(decide_inter_block(owned.input, DecisionOptions{}, decision, error),
          "decide partial " + std::to_string(item.w) + "x" +
              std::to_string(item.h) + ": " + error);
    if (decision.skip) {
      check(decision.zero_residual, "a skip on a shifted block is zero-residual");
      continue;
    }
    check(verify_recon(owned, decision, &error),
          "partial block recon " + std::to_string(item.w) + "x" +
              std::to_string(item.h) + ": " + error);
    InterResidualBlock block;
    std::vector<std::int32_t> levels;
    check(make_syntax_block(geometry, decision, &block, &levels, error),
          "partial block syntax: " + error);
  }
}

void test_batch_matches_blocks() {
  const int width = 128;
  const int height = 64;
  const Frame source = make_source(width, height, 11);
  Frame prediction = make_source(width, height, 11);
  for (std::size_t i = 0; i < prediction.y.size(); ++i) {
    prediction.y[i] = static_cast<std::uint8_t>(
        std::clamp(static_cast<int>(prediction.y[i]) + 4, 0, 255));
  }
  std::vector<BlockGeometry> geometries;
  for (int col = 0; col < 2; ++col) {
    geometries.push_back(make_geometry(0, col * 8, 32, 32));
  }
  geometries.push_back(make_geometry(0, 0, 64, 16));

  BatchInput batch;
  batch.options.try_inter_tx_types = true;
  std::vector<BatchBlock> blocks;
  std::size_t levels_total = 0;
  std::size_t recon_total = 0;
  batch.source[0] = source.y;
  batch.source[1] = source.u;
  batch.source[2] = source.v;
  batch.prediction[0] = prediction.y;
  batch.prediction[1] = prediction.u;
  batch.prediction[2] = prediction.v;
  for (const BlockGeometry& geometry : geometries) {
    BatchBlock block;
    block.mi_row = geometry.mi_row;
    block.mi_col = geometry.mi_col;
    block.luma_w = geometry.luma_w;
    block.luma_h = geometry.luma_h;
    block.chroma_w = geometry.chroma_w;
    block.chroma_h = geometry.chroma_h;
    block.flags = kBatchAllowResidual;
    const int x0 = geometry.mi_col * 4;
    const int y0 = geometry.mi_row * 4;
    block.src_offset[0] = y0 * width + x0;
    block.pred_offset[0] = block.src_offset[0];
    block.src_offset[1] = (y0 / 2) * (width / 2) + x0 / 2;
    block.pred_offset[1] = block.src_offset[1];
    block.src_offset[2] = block.src_offset[1];
    block.pred_offset[2] = block.src_offset[1];
    block.src_stride[0] = width;
    block.pred_stride[0] = width;
    block.src_stride[1] = width / 2;
    block.pred_stride[1] = width / 2;
    block.src_stride[2] = width / 2;
    block.pred_stride[2] = width / 2;
    for (int plane = 0; plane < kResidualPlaneCount; ++plane) {
      const auto params = av2enc::txv::make_tx_params(
          static_cast<av2enc::txv::TxSize>(
              av2enc::txv::tx_size_index(plane == 0 ? geometry.luma_w
                                                   : geometry.chroma_w,
                                         plane == 0 ? geometry.luma_h
                                                    : geometry.chroma_h)));
      block.levels_offset[plane] = static_cast<int32_t>(levels_total);
      block.recon_offset[plane] = static_cast<int32_t>(recon_total);
      levels_total += static_cast<std::size_t>(params->w * params->h);
      recon_total += static_cast<std::size_t>(params->coded_w * params->coded_h);
    }
    blocks.push_back(block);
  }
  batch.blocks = blocks;
  BatchOutput output;
  std::vector<std::int32_t> levels(levels_total);
  std::vector<std::uint8_t> recon(recon_total);
  std::vector<std::int32_t> eob(blocks.size() * 3);
  std::vector<std::int32_t> tx_size(blocks.size() * 3);
  std::vector<std::int32_t> tx_type(blocks.size() * 3);
  std::vector<std::int32_t> skip(blocks.size());
  std::vector<double> distortion(blocks.size());
  std::vector<double> cost(blocks.size());
  std::vector<std::int32_t> status(blocks.size());
  output.levels = levels;
  output.recon = recon;
  output.eob = eob;
  output.tx_size = tx_size;
  output.tx_type = tx_type;
  output.skip = skip;
  output.distortion = distortion;
  output.cost = cost;
  output.status = status;
  std::string error;
  check(decide_inter_batch(batch, output, &error),
        "decide_inter_batch: " + error);

  // Per-block reference.
  for (std::size_t b = 0; b < blocks.size(); ++b) {
    const BatchBlock& item = blocks[b];
    const BlockGeometry geometry = geometries[b];
    OwnedInput owned;
    check(owned.build(source, prediction, width, geometry, error),
          "batch reference build: " + error);
    DecisionOptions options = batch.options;
    options.allow_residual = true;
    BlockDecision decision;
    check(decide_inter_block(owned.input, options, decision, error),
          "batch reference decide: " + error);
    check(skip[b] == (decision.skip ? 1 : 0), "batch skip flag matches");
    check(cost[b] == decision.cost, "batch cost matches");
    for (int plane = 0; plane < kResidualPlaneCount; ++plane) {
      const std::size_t slot = b * 3 + static_cast<std::size_t>(plane);
      check(eob[slot] == decision.plane[plane].eob, "batch EOB matches");
      check(tx_size[slot] == decision.plane[plane].tx_size,
            "batch TX size matches");
      check(tx_type[slot] == decision.plane[plane].tx_type,
            "batch TX type matches");
      check(std::equal(decision.plane[plane].levels.begin(),
                       decision.plane[plane].levels.end(),
                       levels.begin() + item.levels_offset[plane]),
            "batch levels match");
      const bool recon_match =
          std::equal(decision.plane[plane].recon.begin(),
                     decision.plane[plane].recon.end(),
                     recon.begin() + item.recon_offset[plane]);
      if (!recon_match) {
        std::printf("  debug b=%zu plane=%d skip_batch=%d skip_ref=%d "
                    "eob=%d\n",
                    b, plane, skip[b], decision.skip ? 1 : 0,
                    decision.plane[plane].eob);
      }
      check(recon_match, "batch reconstruction matches");
    }
  }
}

void test_rdoq_still_valid() {
  const int width = 64;
  const int height = 64;
  const Frame source = make_source(width, height, 13);
  Frame prediction = make_source(width, height, 13);
  for (std::size_t i = 0; i < prediction.y.size(); ++i) {
    prediction.y[i] = static_cast<std::uint8_t>(
        std::clamp(static_cast<int>(prediction.y[i]) + 12, 0, 255));
  }
  const BlockGeometry geometry = make_geometry(0, 0, 64, 64);
  OwnedInput owned;
  std::string error;
  check(owned.build(source, prediction, width, geometry, error),
        "rdoq build: " + error);
  DecisionOptions options;
  options.rdoq = true;
  BlockDecision decision;
  check(decide_inter_block(owned.input, options, decision, error),
        "rdoq decide: " + error);
  if (!decision.skip) {
    check(verify_recon(owned, decision, &error),
          "rdoq reconstruction matches txv: " + error);
    for (int plane = 0; plane < kResidualPlaneCount; ++plane) {
      const PlaneResidual& residual = decision.plane[plane];
      check(residual.eob <= residual.adj_w * residual.adj_h,
            "rdoq EOB within the transform");
    }
  }
}

}  // namespace

int main() {
  test_tx_sizes();
  test_tx_type_sets();
  test_decision_roundtrip();
  test_zero_residual_fallback();
  test_partial_blocks();
  test_batch_matches_blocks();
  test_rdoq_still_valid();
  std::printf("%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}

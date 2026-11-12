// M2 loop-restoration adapter gates.
//
// Covers the deliverables that can be verified without the M2 core tile
// writer:
//
//   1. §5.4.10 sequence filter row: write -> read round trip.
//   2. §5.18.7.11 lr_params(): frame filters and per-unit modes.
//   3. §5.18.7.12 ccso_params(): band-offset-only and edge-classified rows.
//   4. §5.20.10.2 read_ccso() tile symbols: encode -> decode grid equality.
//   5. §5.20.10.4–§5.20.10.6 read_lr()/read_lr_unit()/read_wienerns_filter():
//      encode -> decode unit type and coefficient equality.
//   6. §7.20.4 class-map gates: single-class accepted, multi-class without a
//      map rejected, explicit map accepted, PC-Wiener rejected.
//   7. Exact tool bit cost (header + standalone symbol stream).
//   8. CPU adapter parity: adapter-driven CCSO + Wiener is byte-identical to a
//      direct rest::ccso_frame()/loop_restoration_frame() call.
//   9. Two-frame (key + P) syntax gate: every row can be produced for both
//      frame types and read back.
//
// Build/run: ./build_and_test.sh

#include "codec/filters/loop_restoration/video/video_restoration.h"

#include "codec/filters/loop_restoration/restoration_test_common.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace rest = av2enc::restoration;
namespace {

using namespace av2enc;
using av2enc::video::restoration::VideoRestorationCost;
using av2enc::video::restoration::VideoRestorationTools;
namespace fs = av2enc::still::filters_stream;
using rest::kRestoreNone;
using rest::kRestorePcWiener;
using rest::kRestoreWienerNs;
using rest::test::TestFrame;
using rest::test::Rng;

int g_checks = 0;
int g_failures = 0;

#define CHECK(cond, msg)                                                   \
  do {                                                                     \
    ++g_checks;                                                            \
    if (!(cond)) {                                                         \
      ++g_failures;                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, (msg)); \
    }                                                                      \
  } while (0)

bool contains(const std::string& haystack, const char* needle) {
  return haystack.find(needle) != std::string::npos;
}

fs::BitReader reader_for(const BitWriter& writer) {
  return fs::BitReader(
      std::span<const std::uint8_t>(writer.bytes().data(),
                                    writer.bytes().size()));
}

std::int16_t coeff_at(bool chroma, int j, int bias) {
  const int min = rest::kWienerNsTapsMin[chroma ? 1 : 0][j];
  const int k = rest::kWienerNsTapsK[chroma ? 1 : 0][j];
  return static_cast<std::int16_t>(min + ((1 << k) / 2) + bias);
}

// ---------------------------------------------------------------------------
// 1. Sequence filter row
// ---------------------------------------------------------------------------

void test_sequence_row() {
  VideoRestorationTools tools;
  tools.ccso = true;
  tools.wiener_ns = true;
  tools.frame_filters = true;
  tools.num_filter_classes = 1;
  tools.chroma_lr = true;
  std::string error;
  CHECK(av2enc::video::restoration::validate_tools(tools, &error), "tools validate");
  CHECK(error.empty(), "tools validate error free");

  BitWriter writer;
  const std::size_t bits =
      av2enc::video::restoration::write_sequence_filter_config(writer, tools, &error);
  CHECK(error.empty(), "sequence write");
  CHECK(bits > 0, "sequence bits");
  fs::BitReader reader = reader_for(writer);
  VideoRestorationTools back;
  CHECK(av2enc::video::restoration::read_sequence_filter_config(reader, &back, &error),
        "sequence read");
  CHECK(error.empty(), "sequence read error free");
  CHECK(back.ccso, "ccso flag");
  CHECK(back.ccso_unit_matches_sb_size, "ccso unit size flag");
  CHECK(back.wiener_ns, "restoration flag");
  CHECK(back.chroma_lr, "chroma lr flag");

  // Off row is shorter and still round-trips.
  VideoRestorationTools off;
  BitWriter off_writer;
  const std::size_t off_bits =
      av2enc::video::restoration::write_sequence_filter_config(off_writer, off, &error);
  CHECK(error.empty(), "off row");
  CHECK(bits > off_bits, "enabled row costs more bits");
}

// ---------------------------------------------------------------------------
// 2/3. Frame headers
// ---------------------------------------------------------------------------

fs::CcsoParams make_ccso(bool bo_only, int bands_log2) {
  fs::CcsoParams p;
  p.frame_flag = true;
  p.ccso_luma_size_log2 = 6;
  for (int plane = 0; plane < 3; ++plane) {
    fs::CcsoPlaneParams& q = p.plane[plane];
    q.enabled = true;
    q.bo_only = bo_only;
    q.scale_idx = bo_only ? 0 : 1;
    q.quant_idx = bo_only ? 0 : 2;
    q.ext_filter = bo_only ? 0 : (plane + 2);
    q.edge_clf = bo_only ? 0 : 1;
    q.max_band_log2 = bands_log2;
    const int interval =
        bo_only ? 1 : (rest::kCcsoInputInterval - q.edge_clf);
    const int count = (1 << bands_log2) * interval * interval;
    q.offset_idx.resize(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
      q.offset_idx[static_cast<std::size_t>(i)] =
          static_cast<std::uint8_t>((i * 5 + plane) & 7);
    }
  }
  return p;
}

void test_ccso_header() {
  for (int bo = 0; bo <= 1; ++bo) {
    const int bands_log2 = bo ? 3 : 2;
    fs::CcsoParams params = make_ccso(bo != 0, bands_log2);
    VideoRestorationTools tools;
    tools.ccso = true;
    std::string error;
    BitWriter writer;
    const fs::CcsoHeaderContext context =
        av2enc::video::restoration::make_ccso_header_context(tools, 64, true, 0);
    const std::size_t bits =
        av2enc::video::restoration::write_ccso_params(writer, params, context, &error);
    CHECK(error.empty(), "ccso header write");
    CHECK(bits > 0, "ccso header bits");
    CHECK(params.ccso_luma_size_log2 == 6, "CcsoLumaSizeLog2 derived");

    fs::BitReader reader = reader_for(writer);
    fs::CcsoParams back;
    CHECK(av2enc::video::restoration::read_ccso_params(reader, context, &back, &error),
          "ccso header read");
    CHECK(error.empty(), "ccso header read error free");
    CHECK(back.frame_flag, "ccso frame flag");
    CHECK(back.ccso_luma_size_log2 == 6, "ccso luma size round trip");
    for (int plane = 0; plane < 3; ++plane) {
      CHECK(back.plane[plane].enabled, "ccso plane enabled");
      CHECK(back.plane[plane].bo_only == params.plane[plane].bo_only,
            "ccso bo_only round trip");
      CHECK(back.plane[plane].max_band_log2 == bands_log2,
            "ccso band count round trip");
      CHECK(back.plane[plane].offset_idx == params.plane[plane].offset_idx,
            "ccso offset indices round trip");
    }
  }
}

fs::LrParams make_lr_frame_filters(int classes, int mi_cols, int mi_rows) {
  fs::LrParams p;
  for (int plane = 0; plane < 3; ++plane) {
    p.frame_type[plane] = kRestoreWienerNs;
    p.frame_filters_on[plane] = true;
  }
  p.num_filter_classes = classes;
  for (int plane = 0; plane < 3; ++plane) {
    const int n = (plane == 0) ? 16 : 18;
    const int ccount = (plane == 0) ? classes : 1;
    for (int c = 0; c < ccount; ++c) {
      p.frame_filter[plane][c].match_index = c;
      p.frame_filter[plane][c].merged = false;
      p.frame_filter[plane][c].subset = (plane == 0) ? 3 : 2;
      for (int j = 0; j < n; ++j) {
        p.frame_filter[plane][c].coeffs[j] = coeff_at(plane > 0, j, c);
      }
      for (int j = n; j < rest::kWienerNsCoeffStride; ++j) {
        p.frame_filter[plane][c].coeffs[j] = 0;
      }
    }
  }
  if (classes > 1) {
    p.class_map.num_classes = classes;
    p.class_map.has_subclass_lookup = true;
    p.class_map.subclass_lookup.fill(0);
    p.class_map.filter_class_grid.assign(
        static_cast<std::size_t>(mi_cols) * static_cast<std::size_t>(mi_rows),
        0);
  }
  return p;
}

void test_lr_header() {
  VideoRestorationTools tools;
  tools.wiener_ns = true;
  tools.frame_filters = true;
  tools.num_filter_classes = 1;
  tools.chroma_lr = true;
  std::string error;
  fs::LrParams params = make_lr_frame_filters(1, 16, 16);
  BitWriter writer;
  const fs::LrHeaderContext context =
      av2enc::video::restoration::make_lr_header_context(tools, 64, 90, true);
  const std::size_t bits =
      av2enc::video::restoration::write_lr_params(writer, params, context, &error);
  CHECK(error.empty(), "lr header write");
  CHECK(bits > 0, "lr header bits");
  fs::BitReader reader = reader_for(writer);
  fs::LrParams back;
  CHECK(av2enc::video::restoration::read_lr_params(reader, context, &back, &error),
        "lr header read");
  CHECK(error.empty(), "lr header read error free");
  for (int plane = 0; plane < 3; ++plane) {
    CHECK(back.frame_type[plane] == kRestoreWienerNs, "lr type round trip");
    CHECK(back.frame_filters_on[plane], "frame filters round trip");
  }
  CHECK(back.num_filter_classes == 1, "classes round trip");
  for (int plane = 0; plane < 3; ++plane) {
    const int n = (plane == 0) ? 16 : 18;
    for (int j = 0; j < n; ++j) {
      CHECK(back.frame_filter[plane][0].coeffs[j] ==
                params.frame_filter[plane][0].coeffs[j],
            "frame coeff round trip");
    }
  }

  // Per-unit mode: frame_filters_on = 0, coefficients live in read_lr().
  tools.frame_filters = false;
  fs::LrParams per_unit = make_lr_frame_filters(1, 16, 16);
  for (int plane = 0; plane < 3; ++plane) {
    per_unit.frame_filters_on[plane] = false;
  }
  const fs::LrHeaderContext pu_context =
      av2enc::video::restoration::make_lr_header_context(tools, 64, 90, true);
  BitWriter pu_writer;
  const std::size_t pu_bits =
      av2enc::video::restoration::write_lr_params(pu_writer, per_unit, pu_context, &error);
  CHECK(error.empty(), "per-unit lr header");
  CHECK(pu_bits > 0 && pu_bits < bits, "per-unit header is shorter");
}

// ---------------------------------------------------------------------------
// 4/5. Tile symbols
// ---------------------------------------------------------------------------

std::vector<std::uint8_t> encode_ccso(const fs::CcsoParams& params,
                                       int mi_cols, int mi_rows,
                                       fs::CcsoTileState* state) {
  std::string error;
  std::vector<std::uint8_t> buffer(1U << 16);
  av2::Encoder encoder;
  CHECK(encoder.init_symbol(buffer.data(), buffer.size()) == av2::Status::kOk,
        "ccso encoder init");
  encoder.disable_cdf_update = 1;
  const int unit = 1 << (params.ccso_luma_size_log2 - 2);
  int n = 0;
  for (int row = 0; row < mi_rows; row += unit) {
    for (int col = 0; col < mi_cols; col += unit) {
      const int blk[3] = {(n % 2) ? 0 : 1, (n % 3) ? 1 : 0, 1};
      CHECK(av2enc::video::restoration::write_read_ccso(encoder, *state, row, col, blk,
                                             &error),
            "ccso tile write");
      ++n;
    }
  }
  std::size_t payload = 0;
  CHECK(encoder.exit_symbol(&payload) == av2::Status::kOk, "ccso exit_symbol");
  buffer.resize(payload);
  return buffer;
}

void test_ccso_tile() {
  const int mi_cols = 32;
  const int mi_rows = 20;  // 128x80, non-multiple of 64 in luma rows
  fs::CcsoParams params = make_ccso(false, 3);
  fs::CcsoTileState state;
  std::string error;
  CHECK(av2enc::video::restoration::init_ccso_tile(&state, params, mi_cols, mi_rows, 1, 1,
                                        &error),
        "ccso tile init");
  CHECK(error.empty(), "ccso tile init error free");
  std::vector<std::uint8_t> payload = encode_ccso(params, mi_cols, mi_rows,
                                                   &state);
  CHECK(!payload.empty(), "ccso tile payload");

  fs::CcsoTileState dec;
  CHECK(av2enc::video::restoration::init_ccso_tile(&dec, params, mi_cols, mi_rows, 1, 1,
                                        &error),
        "ccso dec init");
  av2::Decoder decoder;
  CHECK(decoder.init_symbol(payload.data(), payload.size(), payload.size()) ==
            av2::Status::kOk,
        "ccso decoder init");
  decoder.disable_cdf_update = 1;
  const int unit = 1 << (params.ccso_luma_size_log2 - 2);
  int n = 0;
  for (int row = 0; row < mi_rows; row += unit) {
    for (int col = 0; col < mi_cols; col += unit) {
      int blk[3] = {0, 0, 0};
      CHECK(av2enc::video::restoration::read_ccso(decoder, dec, row, col, blk, &error),
            "ccso tile read");
      const int expect0 = (n % 2) ? 0 : 1;
      const int expect1 = (n % 3) ? 1 : 0;
      CHECK(blk[0] == expect0 && blk[1] == expect1 && blk[2] == 1,
            "ccso block decision round trip");
      ++n;
    }
  }
  CHECK(decoder.exit_symbol() == av2::Status::kOk, "ccso decoder final");
  for (int p = 0; p < 3; ++p) {
    CHECK(dec.blks[p] == state.blks[p], "ccso decoded grid equality");
  }

  // §7.19 application through the M2 adapter on a synthetic frame.
  TestFrame f;
  f.mi_cols = mi_cols;
  f.mi_rows = mi_rows;
  f.allocate();
  Rng rng(7);
  rest::test::fill_random(f, rng);
  f.cdef[0] = f.curr[0];
  f.cdef[1] = f.curr[1];
  f.cdef[2] = f.curr[2];
  // Snapshot the pre-CCSO buffers before the adapter modifies `f`.
  TestFrame direct;
  direct.mi_cols = mi_cols;
  direct.mi_rows = mi_rows;
  direct.allocate();
  for (int p = 0; p < 3; ++p) {
    direct.curr[p] = f.curr[p];
    direct.cdef[p] = f.cdef[p];
  }

  rest::RestorationFrame view = f.view();
  const rest::RestorationStatus st =
      av2enc::video::restoration::apply_ccso_frame(view, params, dec);
  CHECK(st == rest::RestorationStatus::kOk, "adapter ccso apply ok");

  rest::RestorationFrame direct_view = direct.view();
  rest::CcsoFrameParams frame_params;
  CHECK(av2enc::video::restoration::make_ccso_frame_params(params, &dec, mi_cols, mi_rows,
                                                1, 1, &frame_params, &error),
        "make ccso frame params");
  CHECK(rest::ccso_frame(direct_view, frame_params) ==
            rest::RestorationStatus::kOk,
        "direct ccso apply");
  for (int p = 0; p < 3; ++p) {
    CHECK(f.cdef[p] == direct.cdef[p], "adapter ccso parity");
  }
}

fs::LrParams make_lr_per_unit(int mi_cols, int mi_rows) {
  fs::LrParams p;
  for (int plane = 0; plane < 3; ++plane) {
    p.frame_type[plane] = kRestoreWienerNs;
    p.frame_filters_on[plane] = false;
    p.luma_shift = 3;
    p.chroma_shift = 3;
  }
  (void)mi_cols;
  (void)mi_rows;
  return p;
}

void fill_unit_types(fs::LrState* state) {
  for (int plane = 0; plane < 3; ++plane) {
    for (std::size_t i = 0; i < state->unit_type[plane].size(); ++i) {
      state->unit_type[plane][i] =
          static_cast<std::uint8_t>((i % 3 == 0) ? kRestoreNone
                                                : kRestoreWienerNs);
      const int n = (plane == 0) ? 16 : 18;
      for (int j = 0; j < n; ++j) {
        state->unit_coeffs[plane][i * rest::kWienerNsCoeffStride +
                                  static_cast<std::size_t>(j)] =
            coeff_at(plane > 0, j, static_cast<int>(i % 3));
      }
    }
  }
}

std::vector<std::uint8_t> encode_lr(fs::LrState* state, int mi_cols,
                                     int mi_rows, int sb_size) {
  std::string error;
  std::vector<std::uint8_t> buffer(1U << 18);
  av2::Encoder encoder;
  CHECK(encoder.init_symbol(buffer.data(), buffer.size()) == av2::Status::kOk,
        "lr encoder init");
  encoder.disable_cdf_update = 1;
  const int sb_mi = sb_size / 4;
  for (int row = 0; row < mi_rows; row += sb_mi) {
    for (int col = 0; col < mi_cols; col += sb_mi) {
      std::vector<fs::LrUnitDecision> decisions;
      CHECK(av2enc::video::restoration::build_lr_decisions(*state, row, col, sb_mi,
                                                sb_mi, 0, 0, &decisions,
                                                &error),
            "build lr decisions");
      CHECK(av2enc::video::restoration::write_read_lr(encoder, *state, row, col, sb_mi,
                                           sb_mi, 0, 0, decisions, &error),
            "lr tile write");
    }
  }
  std::size_t payload = 0;
  CHECK(encoder.exit_symbol(&payload) == av2::Status::kOk, "lr exit_symbol");
  buffer.resize(payload);
  return buffer;
}

void test_lr_tile() {
  const int mi_cols = 32;
  const int mi_rows = 20;
  fs::LrParams params = make_lr_per_unit(mi_cols, mi_rows);
  fs::LrState enc;
  std::string error;
  CHECK(av2enc::video::restoration::init_lr_tile(&enc, params, mi_cols, mi_rows, 1, 1,
                                      64, &error),
        "lr tile init");
  fill_unit_types(&enc);
  std::vector<std::uint8_t> payload = encode_lr(&enc, mi_cols, mi_rows, 64);
  CHECK(!payload.empty(), "lr tile payload");

  fs::LrState dec;
  CHECK(av2enc::video::restoration::init_lr_tile(&dec, params, mi_cols, mi_rows, 1, 1,
                                      64, &error),
        "lr dec init");
  av2::Decoder decoder;
  CHECK(decoder.init_symbol(payload.data(), payload.size(), payload.size()) ==
            av2::Status::kOk,
        "lr decoder init");
  decoder.disable_cdf_update = 1;
  const int sb_mi = 16;
  for (int row = 0; row < mi_rows; row += sb_mi) {
    for (int col = 0; col < mi_cols; col += sb_mi) {
      std::vector<fs::LrUnitDecision> decisions;
      CHECK(av2enc::video::restoration::read_lr(decoder, dec, row, col, sb_mi, sb_mi, 0,
                                     0, &decisions, &error),
            "lr tile read");
    }
  }
  CHECK(decoder.exit_symbol() == av2::Status::kOk, "lr decoder final");
  for (int p = 0; p < 3; ++p) {
    CHECK(dec.unit_type[p] == enc.unit_type[p], "lr unit type equality");
    bool coeffs_match = true;
    for (std::size_t u = 0; u < enc.unit_type[p].size(); ++u) {
      if (enc.unit_type[p][u] != kRestoreWienerNs) continue;
      for (int j = 0; j < rest::kWienerNsCoeffStride; ++j) {
        const std::size_t k =
            u * rest::kWienerNsCoeffStride + static_cast<std::size_t>(j);
        if (dec.unit_coeffs[p][k] != enc.unit_coeffs[p][k]) {
          coeffs_match = false;
        }
      }
    }
    CHECK(coeffs_match, "lr unit coefficient equality");
  }

  // Direct unit entry points (read_lr_unit / read_wienerns_filter aliases).
  std::vector<std::uint8_t> unit_buffer(1U << 12);
  av2::Encoder unit_enc;
  CHECK(unit_enc.init_symbol(unit_buffer.data(), unit_buffer.size()) ==
            av2::Status::kOk,
        "unit encoder init");
  unit_enc.disable_cdf_update = 1;
  fs::LrState unit_state;
  CHECK(av2enc::video::restoration::init_lr_tile(&unit_state, params, mi_cols, mi_rows, 1,
                                      1, 64, &error),
        "unit state init");
  fs::LrUnitDecision unit_decision;
  unit_decision.type = kRestoreWienerNs;
  unit_decision.subset = 3;
  for (int j = 0; j < 16; ++j) {
    unit_decision.coeffs[j] = coeff_at(false, j, 1);
  }
  CHECK(av2enc::video::restoration::write_read_wienerns_filter(
            unit_enc, unit_state, 0, 0, 0, unit_decision, &error),
        "wienerns filter write");
  std::size_t unit_payload = 0;
  CHECK(unit_enc.exit_symbol(&unit_payload) == av2::Status::kOk,
        "unit exit_symbol");
  fs::LrState unit_dec;
  CHECK(av2enc::video::restoration::init_lr_tile(&unit_dec, params, mi_cols, mi_rows, 1,
                                      1, 64, &error),
        "unit dec state");
  av2::Decoder unit_reader;
  CHECK(unit_reader.init_symbol(unit_buffer.data(), unit_buffer.size(),
                                unit_payload) == av2::Status::kOk,
        "unit decoder init");
  unit_reader.disable_cdf_update = 1;
  fs::LrUnitDecision back;
  CHECK(av2enc::video::restoration::read_wienerns_filter(unit_reader, unit_dec, 0, 0, 0,
                                              &back, &error),
        "wienerns filter read");
  CHECK(back.type == kRestoreWienerNs, "unit type read");
  for (int j = 0; j < 16; ++j) {
    CHECK(back.coeffs[j] == unit_decision.coeffs[j], "unit coeff read");
  }
  CHECK(unit_reader.exit_symbol() == av2::Status::kOk, "unit decoder final");
}

void test_lr_frame_filter_tile() {
  const int mi_cols = 32;
  const int mi_rows = 20;
  VideoRestorationTools tools;
  tools.wiener_ns = true;
  tools.frame_filters = true;
  tools.num_filter_classes = 1;
  tools.chroma_lr = true;
  std::string error;
  fs::LrParams params = make_lr_frame_filters(1, mi_cols, mi_rows);
  CHECK(av2enc::video::restoration::validate_class_map(params, mi_cols, mi_rows, &error),
        "single-class map gate");
  fs::LrState enc;
  CHECK(av2enc::video::restoration::init_lr_tile(&enc, params, mi_cols, mi_rows, 1, 1,
                                      64, &error),
        "frame filter state init");
  // init_lr_tile fills every unit with WIENER_NS.
  std::vector<std::uint8_t> payload = encode_lr(&enc, mi_cols, mi_rows, 64);
  CHECK(!payload.empty(), "frame filter payload");
  fs::LrState dec;
  CHECK(av2enc::video::restoration::init_lr_tile(&dec, params, mi_cols, mi_rows, 1, 1,
                                      64, &error),
        "frame filter dec init");
  av2::Decoder decoder;
  CHECK(decoder.init_symbol(payload.data(), payload.size(), payload.size()) ==
            av2::Status::kOk,
        "frame filter decoder init");
  decoder.disable_cdf_update = 1;
  for (int row = 0; row < mi_rows; row += 16) {
    for (int col = 0; col < mi_cols; col += 16) {
      std::vector<fs::LrUnitDecision> decisions;
      CHECK(av2enc::video::restoration::read_lr(decoder, dec, row, col, 16, 16, 0, 0,
                                     &decisions, &error),
            "frame filter read_lr");
    }
  }
  CHECK(decoder.exit_symbol() == av2::Status::kOk, "frame filter final");
  for (int p = 0; p < 3; ++p) {
    CHECK(dec.unit_type[p] == enc.unit_type[p],
          "frame filter unit type equality");
  }
}

// ---------------------------------------------------------------------------
// 6. Class map / PC-Wiener gates
// ---------------------------------------------------------------------------

void test_class_map_gates() {
  const int mi_cols = 16;
  const int mi_rows = 8;
  std::string error;

  fs::LrParams single = make_lr_frame_filters(1, mi_cols, mi_rows);
  CHECK(av2enc::video::restoration::validate_class_map(single, mi_cols, mi_rows, &error),
        "single class accepted");
  CHECK(error.empty(), "single class no error");

  fs::LrParams multi = make_lr_frame_filters(2, mi_cols, mi_rows);
  multi.class_map = fs::WienerNsClassMap{};
  error.clear();
  CHECK(!av2enc::video::restoration::validate_class_map(multi, mi_cols, mi_rows, &error),
        "multi class without map rejected");
  CHECK(contains(error, "class lookup") || contains(error, "class map"),
        "missing class map error is explicit");

  multi = make_lr_frame_filters(2, mi_cols, mi_rows);
  error.clear();
  CHECK(av2enc::video::restoration::validate_class_map(multi, mi_cols, mi_rows, &error),
        "multi class with map accepted");

  multi = make_lr_frame_filters(2, mi_cols, mi_rows);
  multi.class_map.filter_class_grid.pop_back();
  error.clear();
  CHECK(!av2enc::video::restoration::validate_class_map(multi, mi_cols, mi_rows, &error),
        "bad class grid size rejected");

  fs::LrParams pc = make_lr_frame_filters(1, mi_cols, mi_rows);
  pc.frame_type[1] = kRestorePcWiener;
  error.clear();
  CHECK(!av2enc::video::restoration::validate_no_pc_wiener(pc, mi_cols, mi_rows, &error),
        "PC-Wiener rejected");
  CHECK(contains(error, "PC-Wiener"), "PC-Wiener blocker named");

  fs::LrParams layout = make_lr_frame_filters(2, mi_cols, mi_rows);
  layout.frame_filter[0][0].match_index = 1;
  error.clear();
  CHECK(!av2enc::video::restoration::validate_frame_filter_layout(layout, &error),
        "first class match != 0 rejected");

  layout = make_lr_frame_filters(2, mi_cols, mi_rows);
  error.clear();
  CHECK(av2enc::video::restoration::validate_frame_filter_layout(layout, &error),
        "deterministic group-0 layout accepted");

  fs::WienerNsClassMap map;
  CHECK(av2enc::video::restoration::make_single_class_map(&map, mi_cols, mi_rows, &error),
        "make single class map");
  CHECK(map.num_classes == 1, "single class count");
  CHECK(map.filter_class_grid.size() ==
            static_cast<std::size_t>(mi_cols * mi_rows),
        "single class grid size");
}

// ---------------------------------------------------------------------------
// 6b. Gated single-class / single-unit path (64x64: 1 unit per plane)
// ---------------------------------------------------------------------------

void test_single_class_single_unit() {
  const int mi_cols = 16;  // 64x64
  const int mi_rows = 16;
  std::string error;
  VideoRestorationTools tools;
  tools.wiener_ns = true;
  tools.frame_filters = true;
  tools.num_filter_classes = 1;
  tools.chroma_lr = true;

  fs::LrParams params = make_lr_frame_filters(1, mi_cols, mi_rows);
  CHECK(av2enc::video::restoration::validate_class_map(params, mi_cols, mi_rows, &error),
        "single-unit single-class map gate");
  fs::LrState state;
  CHECK(av2enc::video::restoration::init_lr_tile(&state, params, mi_cols, mi_rows, 1, 1,
                                      64, &error),
        "single-unit state");
  for (int p = 0; p < 3; ++p) {
    CHECK(state.unit_cols[p] == 1 && state.unit_rows[p] == 1,
          "one restoration unit per plane");
  }
  CHECK(av2enc::video::restoration::lr_block_unit_count(state, 0, 0, 16, 16, 0, 0) == 3,
        "read_lr visits one unit per plane");
  CHECK(av2enc::video::restoration::validate_no_pc_wiener(params, mi_cols, mi_rows,
                                               &error),
        "single-unit no PC-Wiener");
  CHECK(av2enc::video::restoration::validate_frame_filter_layout(params, &error),
        "single-unit group-0 layout");

  BitWriter header;
  const fs::LrHeaderContext context =
      av2enc::video::restoration::make_lr_header_context(tools, 64, 90, true);
  const std::size_t bits =
      av2enc::video::restoration::write_lr_params(header, params, context, &error);
  CHECK(error.empty() && bits > 0, "single-unit lr header");

  std::vector<std::uint8_t> payload = encode_lr(&state, mi_cols, mi_rows, 64);
  CHECK(!payload.empty(), "single-unit payload");
  fs::LrState dec;
  CHECK(av2enc::video::restoration::init_lr_tile(&dec, params, mi_cols, mi_rows, 1, 1,
                                      64, &error),
        "single-unit dec state");
  av2::Decoder decoder;
  CHECK(decoder.init_symbol(payload.data(), payload.size(), payload.size()) ==
            av2::Status::kOk,
        "single-unit decoder");
  decoder.disable_cdf_update = 1;
  std::vector<fs::LrUnitDecision> decisions;
  CHECK(av2enc::video::restoration::read_lr(decoder, dec, 0, 0, 16, 16, 0, 0, &decisions,
                                 &error),
        "single-unit read_lr");
  CHECK(decisions.size() == 3, "single-unit decision count");
  CHECK(decoder.exit_symbol() == av2::Status::kOk, "single-unit final");
}

// ---------------------------------------------------------------------------
// 7. Cost
// ---------------------------------------------------------------------------

void test_cost() {
  const int mi_cols = 40;  // 160x90 -> 10x6 SBs, the 640x360 M1 geometry
  const int mi_rows = 90;
  std::string error;
  VideoRestorationTools tools;
  tools.ccso = true;
  tools.wiener_ns = true;
  tools.frame_filters = true;
  tools.num_filter_classes = 1;
  tools.chroma_lr = true;

  fs::CcsoParams ccso = make_ccso(false, 3);
  fs::CcsoTileState ccso_state;
  CHECK(av2enc::video::restoration::init_ccso_tile(&ccso_state, ccso, mi_cols, mi_rows, 1,
                                        1, &error),
        "cost ccso init");
  fs::LrParams lr_params = make_lr_frame_filters(1, mi_cols, mi_rows);
  fs::LrState lr_state;
  CHECK(av2enc::video::restoration::init_lr_tile(&lr_state, lr_params, mi_cols, mi_rows,
                                      1, 1, 64, &error),
        "cost lr init");
  VideoRestorationCost cost;
  CHECK(av2enc::video::restoration::measure_restoration_cost(
            tools, mi_cols, mi_rows, 64, 90, true, 0, &ccso, &ccso_state,
            &lr_state, &cost, &error),
        "measure cost");
  CHECK(error.empty(), "measure cost error free");
  CHECK(cost.sequence_bits > 0, "sequence cost");
  CHECK(cost.ccso_header_bits > 0, "ccso header cost");
  CHECK(cost.lr_header_bits > 0, "lr header cost");
  CHECK(cost.ccso_symbol_bytes > 0, "ccso symbol cost");
  CHECK(cost.lr_symbol_bytes > 0, "lr symbol cost");
  CHECK(cost.total_bits() > 8 * (cost.ccso_symbol_bytes +
                                 cost.lr_symbol_bytes),
        "header contributes to total");

  std::printf(
      "  cost %dx%d: seq %zu b, ccso hdr %zu b, lr hdr %zu b, "
      "ccso sym %zu B, lr sym %zu B, total %zu B\n",
      mi_cols * 4, mi_rows * 4, cost.sequence_bits, cost.ccso_header_bits,
      cost.lr_header_bits, cost.ccso_symbol_bytes, cost.lr_symbol_bytes,
      (cost.total_bits() + 7) / 8);
}

// ---------------------------------------------------------------------------
// 8. CPU adapter parity for the full CCSO -> Wiener chain
// ---------------------------------------------------------------------------

void test_cpu_chain_parity() {
  const int mi_cols = 24;
  const int mi_rows = 14;
  TestFrame a;
  a.mi_cols = mi_cols;
  a.mi_rows = mi_rows;
  a.allocate();
  Rng rng(99);
  rest::test::fill_random(a, rng);

  TestFrame b = a;

  std::string error;
  fs::CcsoParams ccso = make_ccso(false, 2);
  fs::CcsoTileState ccso_state;
  CHECK(av2enc::video::restoration::init_ccso_tile(&ccso_state, ccso, mi_cols, mi_rows, 1,
                                        1, &error),
        "chain ccso init");
  fs::LrParams lr_params = make_lr_per_unit(mi_cols, mi_rows);
  fs::LrState lr_state;
  CHECK(av2enc::video::restoration::init_lr_tile(&lr_state, lr_params, mi_cols, mi_rows,
                                      1, 1, 64, &error),
        "chain lr init");
  fill_unit_types(&lr_state);

  rest::RestorationFrame a_view = a.view();
  CHECK(av2enc::video::restoration::apply_restoration_frame(a_view, &ccso_state,
                                                 &lr_state) ==
            rest::RestorationStatus::kOk,
        "adapter chain apply");

  // Direct reference: same parameters through the CPU API.
  rest::RestorationFrame b_view = b.view();
  rest::CcsoFrameParams frame_params;
  CHECK(av2enc::video::restoration::make_ccso_frame_params(ccso, &ccso_state, mi_cols,
                                                mi_rows, 1, 1, &frame_params,
                                                &error),
        "chain direct params");
  CHECK(rest::ccso_frame(b_view, frame_params) ==
            rest::RestorationStatus::kOk,
        "chain direct ccso");
  CHECK(av2enc::video::restoration::apply_lr_frame(b_view, lr_state) ==
            rest::RestorationStatus::kOk,
        "chain direct lr");
  for (int p = 0; p < 3; ++p) {
    CHECK(a.cdef[p] == b.cdef[p], "chain cdef parity");
    CHECK(a.lr[p] == b.lr[p], "chain lr parity");
  }
}

// ---------------------------------------------------------------------------
// 9. Two-frame syntax gate (key + P)
// ---------------------------------------------------------------------------

void test_two_frame_gate() {
  const int mi_cols = 32;
  const int mi_rows = 18;
  std::string error;
  VideoRestorationTools tools;
  tools.ccso = true;
  tools.wiener_ns = true;
  tools.frame_filters = true;
  tools.num_filter_classes = 1;
  tools.chroma_lr = true;

  for (int frame = 0; frame < 2; ++frame) {
    const bool intra = frame == 0;
    fs::CcsoParams ccso = make_ccso(false, 2);
    fs::CcsoTileState ccso_state;
    CHECK(av2enc::video::restoration::init_ccso_tile(&ccso_state, ccso, mi_cols, mi_rows,
                                          1, 1, &error),
          "2f ccso init");
    fs::LrParams lr_params = intra
                                 ? make_lr_frame_filters(1, mi_cols, mi_rows)
                                 : make_lr_per_unit(mi_cols, mi_rows);
    fs::LrState lr_state;
    CHECK(av2enc::video::restoration::init_lr_tile(&lr_state, lr_params, mi_cols, mi_rows,
                                        1, 1, 64, &error),
          "2f lr init");

    BitWriter seq;
    const std::size_t seq_bits =
        av2enc::video::restoration::write_sequence_filter_config(seq, tools, &error);
    CHECK(error.empty() && seq_bits > 0, "2f sequence row");

    BitWriter ccso_bits;
    const fs::CcsoHeaderContext ccso_context =
        av2enc::video::restoration::make_ccso_header_context(tools, 64, intra, 0);
    const std::size_t ccso_header =
        av2enc::video::restoration::write_ccso_params(ccso_bits, ccso, ccso_context,
                                           &error);
    CHECK(error.empty() && ccso_header > 0, "2f ccso header");

    BitWriter lr_bits;
    const fs::LrHeaderContext lr_context =
        av2enc::video::restoration::make_lr_header_context(tools, 64, 90, intra);
    const std::size_t lr_header =
        av2enc::video::restoration::write_lr_params(lr_bits, lr_params, lr_context,
                                         &error);
    CHECK(error.empty() && lr_header > 0, "2f lr header");

    // The P-frame CCSO row codes reuse_ccso/sb_reuse_ccso = 0 for each plane,
    // so it is longer than the intra row (which infers both zero).
    if (frame == 1) {
      fs::CcsoParams intra_copy = ccso;
      BitWriter intra_bits;
      const fs::CcsoHeaderContext intra_context =
          av2enc::video::restoration::make_ccso_header_context(tools, 64, true, 0);
      const std::size_t intra_header = av2enc::video::restoration::write_ccso_params(
          intra_bits, intra_copy, intra_context, &error);
      CHECK(ccso_header > intra_header,
            "P-frame CCSO row codes reuse flags");
    }

    std::printf("  frame %d (%s): seq %zu b, ccso hdr %zu b, lr hdr %zu b\n",
                frame, intra ? "key" : "P", seq_bits, ccso_header, lr_header);
  }
}

}  // namespace

int main() {
  std::printf("AV2 M2 restoration adapters — component gates\n");
  test_sequence_row();
  test_ccso_header();
  test_lr_header();
  test_ccso_tile();
  test_lr_tile();
  test_lr_frame_filter_tile();
  test_class_map_gates();
  test_single_class_single_unit();
  test_cost();
  test_cpu_chain_parity();
  test_two_frame_gate();
  std::printf("%s: %d checks, %d failures\n",
              g_failures == 0 ? "PASS" : "FAIL", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}

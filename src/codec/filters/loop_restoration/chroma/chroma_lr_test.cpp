// Chroma loop-restoration unit tests: unit sizing, filter rules, exact syntax
// goldens, syntax round-trips against the shared luma implementation, the
// encoder decision on real reconstruction, and lr_tools_uv_present semantics.

#include "codec/filters/loop_restoration/chroma/chroma_lr.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "core/bitstream/bitwriter.h"
#include "codec/filters/loop_restoration/classes/wiener_fit.h"
#include "codec/filters/stream/filters_stream.h"

namespace {

int g_checks = 0;
int g_failures = 0;

#define CHECK(cond, what)                                                   \
  do {                                                                      \
    ++g_checks;                                                             \
    if (!(cond)) {                                                          \
      ++g_failures;                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, (what)); \
    }                                                                       \
  } while (0)

std::string hex(const std::vector<std::uint8_t>& bytes) {
  static const char* kHex = "0123456789abcdef";
  std::string out;
  for (std::uint8_t b : bytes) {
    out.push_back(kHex[b >> 4]);
    out.push_back(kHex[b & 0xf]);
  }
  return out;
}

av2enc::lr_chroma::ChromaLrInput make_checker_input(int width, int height,
                                                    bool /*identical_uv*/) {
  using namespace av2enc::lr_chroma;
  ChromaLrInput in;
  in.plane_width[0] = in.plane_width[1] = width;
  in.plane_height[0] = in.plane_height[1] = height;
  in.mi_cols = (width * 2 + 3) / 4;
  in.mi_rows = (height * 2 + 3) / 4;
  in.sub_x = 1;
  in.sub_y = 1;
  in.sb_size = 64;
  return in;
}

struct CheckFrame {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> src_u, src_v, rec_u, rec_v, luma;
};

CheckFrame make_check_frame(int width, int height, bool identical_uv) {
  CheckFrame f;
  f.width = width;
  f.height = height;
  const std::size_t n =
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  f.src_u.resize(n);
  f.src_v.resize(n);
  f.rec_u.resize(n);
  f.rec_v.resize(n);
  f.luma.resize(static_cast<std::size_t>(width) * 2 *
                static_cast<std::size_t>(height) * 2);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const std::size_t i =
          static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
          static_cast<std::size_t>(x);
      const int u = 128 + static_cast<int>(24.0 * std::sin(x * 0.20) +
                                           18.0 * std::cos(y * 0.15));
      const int v = identical_uv
                        ? u
                        : 128 + static_cast<int>(20.0 * std::cos((x + y) * 0.11));
      f.src_u[i] = static_cast<std::uint8_t>(u);
      f.src_v[i] = static_cast<std::uint8_t>(v);
      // Quantisation-like noise: the Wiener filter should smooth it.
      const int nu = ((x * 13 + y * 7 + 5) % 25) - 12;
      const int nv =
          identical_uv ? nu : (((x * 5 + y * 17 + 9) % 25) - 12);
      f.rec_u[i] = static_cast<std::uint8_t>(
          std::max(0, std::min(255, static_cast<int>(f.src_u[i]) + nu)));
      f.rec_v[i] = static_cast<std::uint8_t>(
          std::max(0, std::min(255, static_cast<int>(f.src_v[i]) + nv)));
    }
  }
  const int lw = width * 2;
  const int lh = height * 2;
  for (int y = 0; y < lh; ++y) {
    for (int x = 0; x < lw; ++x) {
      f.luma[static_cast<std::size_t>(y) * static_cast<std::size_t>(lw) +
             static_cast<std::size_t>(x)] =
          static_cast<std::uint8_t>(128 + static_cast<int>(30.0 *
                                                           std::sin(x * 0.07)));
    }
  }
  return f;
}

void bind(const CheckFrame& f, av2enc::lr_chroma::ChromaLrInput* in) {
  in->curr[0] = in->cdef[0] = f.rec_u.data();
  in->curr_stride[0] = in->cdef_stride[0] = f.width;
  in->curr[1] = in->cdef[1] = f.rec_v.data();
  in->curr_stride[1] = in->cdef_stride[1] = f.width;
  in->luma_curr = in->luma_cdef = f.luma.data();
  in->luma_curr_stride = in->luma_cdef_stride = f.width * 2;
  in->luma_width = f.width * 2;
  in->luma_height = f.height * 2;
  in->source[0] = f.src_u.data();
  in->source[1] = f.src_v.data();
  in->source_stride[0] = in->source_stride[1] = f.width;
}

av2enc::lr_chroma::ChromaDecision empty_decision(int width, int height,
                                                 int shift) {
  using namespace av2enc::lr_chroma;
  ChromaDecision d;
  d.shift = shift;
  d.unit_size = (512 >> 1) >> shift;
  d.sb_size = 64;
  std::string error;
  for (int c = 0; c < 2; ++c) {
    CHECK(chroma_make_unit_grid(width, height, d.unit_size, &d.grid[c],
                                &error),
          "grid");
  }
  return d;
}

void test_unit_sizing() {
  using namespace av2enc::lr_chroma;
  std::string error;
  struct Case {
    int desired;
    int shift;
  };
  const Case cases64[] = {{128, 1}, {256, 0}, {64, 2}, {32, 3}};
  for (const Case& c : cases64) {
    ChromaUnitSize size;
    CHECK(chroma_select_unit_size(c.desired, 64, 1, 1, &size, &error),
          "select 64");
    CHECK(size.shift == c.shift, "shift 64");
    CHECK(size.size == c.desired, "size 64");
    av2enc::BitWriter writer;
    CHECK(chroma_write_unit_size(writer, size, &error), "write size");
    const std::size_t written = writer.bit_size();
    const auto bytes = writer.take_bytes();
    av2enc::restoration::classes::WienerBitReader reader(bytes.data(), written);
    ChromaUnitSize back;
    CHECK(chroma_read_unit_size(reader, 64, 1, 1, &back, &error),
          "read size");
    CHECK(back.shift == c.shift, "read shift");
    CHECK(back.size == c.desired, "read size");
    CHECK(reader.bit_pos() == written, "size bits consumed");
  }
  ChromaUnitSize size;
  CHECK(!chroma_select_unit_size(16, 64, 1, 1, &size, &error),
        "16 unsupported at 4:2:0");
  const Case cases128[] = {{256, 0}, {128, 1}, {64, 2}};
  for (const Case& c : cases128) {
    ChromaUnitSize s;
    CHECK(chroma_select_unit_size(c.desired, 128, 1, 1, &s, &error),
          "select 128");
    CHECK(s.shift == c.shift, "shift 128");
  }
  CHECK(!chroma_select_unit_size(32, 128, 1, 1, &size, &error),
        "32 unsupported at 128");
  const Case cases256[] = {{256, 0}, {128, 1}};
  for (const Case& c : cases256) {
    ChromaUnitSize s;
    CHECK(chroma_select_unit_size(c.desired, 256, 1, 1, &s, &error),
          "select 256");
    CHECK(s.shift == c.shift, "shift 256");
  }
  CHECK(!chroma_select_unit_size(64, 256, 1, 1, &size, &error),
        "64 unsupported at 256");

  // count_units_in_frame matches §5.20.10.4.
  CHECK(chroma_count_units(64, 320) == 5, "units 320/64");
  CHECK(chroma_count_units(64, 180) == 3, "units 180/64");
  CHECK(chroma_count_units(128, 96) == 1, "units 96/128");
}

void test_filter_model() {
  using namespace av2enc::lr_chroma;
  CHECK(chroma_present_coeffs(0) == 6, "present subset0");
  CHECK(chroma_present_coeffs(1) == 10, "present subset1");
  CHECK(chroma_present_coeffs(2) == 18, "present subset2");
  CHECK(chroma_coded_coeffs(0, false) == 6, "coded subset0");
  CHECK(chroma_coded_coeffs(1, false) == 10, "coded subset1 asym");
  CHECK(chroma_coded_coeffs(2, false) == 18, "coded subset2 asym");
  CHECK(chroma_coded_coeffs(1, true) == 8, "coded subset1 sym");
  CHECK(chroma_coded_coeffs(2, true) == 12, "coded subset2 sym");

  ChromaFilter f;
  f.coeffs[0] = 1;
  CHECK(chroma_derive_subset(f) == 0, "derive subset0");
  f.coeffs[8] = -2;
  CHECK(chroma_derive_subset(f) == 1, "derive subset1");
  f.coeffs[12] = 3;
  CHECK(chroma_derive_subset(f) == 2, "derive subset2");
  CHECK(!chroma_filter_fits_subset(f, 1), "subset1 rejects tap 12");

  ChromaFilter sym;
  sym.subset = 2;
  sym.coeffs[6] = 5;
  sym.coeffs[7] = 1;
  sym.coeffs[12] = -3;
  sym.coeffs[13] = 1;
  std::string error;
  CHECK(chroma_mirror_filter(&sym, &error), "mirror");
  CHECK(sym.uv_sym, "mirror sets uv_sym");
  CHECK(sym.coeffs[6] == sym.coeffs[7] && sym.coeffs[6] == 3, "pair 6/7");
  CHECK(sym.coeffs[12] == sym.coeffs[13] && sym.coeffs[12] == -1, "pair 12/13");
  CHECK(chroma_filter_validate(sym, &error), "symmetric validates");

  ChromaFilter bad = sym;
  bad.coeffs[8] = 2;
  bad.coeffs[9] = 3;
  CHECK(!chroma_filter_validate(bad, &error), "non-mirrored pair rejected");

  ChromaFilter range;
  range.subset = 0;
  range.coeffs[0] = 40;  // chroma range is [-24, 39]
  CHECK(!chroma_filter_validate(range, &error), "range rejected");

  ChromaFilter zero;
  zero.subset = 0;
  zero.coeffs[0] = 39;
  CHECK(chroma_filter_validate(zero, &error), "max accepted");
}

void test_syntax_goldens() {
  using namespace av2enc::lr_chroma;
  std::string error;
  const CheckFrame frame = make_check_frame(128, 128, false);
  ChromaLrInput in = make_checker_input(128, 128, false);
  bind(frame, &in);
  ChromaDecision d = empty_decision(128, 128, 2);
  fs::LrState state;
  CHECK(chroma_init_tile_state(d, in, &state, &error), "state");

  ChromaFilter filter;
  std::vector<std::uint8_t> bytes;
  CHECK(chroma_measure_unit_bytes(state, 1, 0, 0, filter, false, &bytes,
                                  &error),
        "off bytes");
  CHECK(hex(bytes) == "200000", "off golden");

  filter = ChromaFilter{};
  filter.subset = 0;
  filter.coeffs[0] = 32;
  CHECK(chroma_measure_unit_bytes(state, 1, 0, 0, filter, true, &bytes,
                                  &error),
        "subset0 bytes");
  CHECK(hex(bytes) == "572312de0000", "subset0 golden");

  filter = ChromaFilter{};
  filter.subset = 2;
  filter.coeffs[0] = 32;
  filter.coeffs[1] = 32;
  filter.coeffs[2] = -3;
  filter.coeffs[12] = 1;
  CHECK(chroma_measure_unit_bytes(state, 1, 0, 0, filter, true, &bytes,
                                  &error),
        "subset2 asym bytes");
  CHECK(hex(bytes) == "9a47642034002dd50000", "subset2 asym golden");

  filter.uv_sym = true;
  for (int j = 6; j + 1 < 18; j += 2) filter.coeffs[j + 1] = filter.coeffs[j];
  CHECK(chroma_measure_unit_bytes(state, 1, 0, 0, filter, true, &bytes,
                                  &error),
        "subset2 sym bytes");
  CHECK(hex(bytes) == "a2f599220ac5480000", "subset2 sym golden");

  filter = ChromaFilter{};
  filter.subset = 1;
  filter.uv_sym = true;
  filter.coeffs[6] = 4;
  filter.coeffs[7] = 4;
  filter.coeffs[8] = -2;
  filter.coeffs[9] = -2;
  CHECK(chroma_measure_unit_bytes(state, 1, 0, 0, filter, true, &bytes,
                                  &error),
        "subset1 sym bytes");
  CHECK(hex(bytes) == "65c513fde30000", "subset1 sym golden");

  // Read back each golden from a fresh state and check the reconstructed
  // filter, not just that it parses.
  const std::string goldens[] = {"200000", "572312de0000",
                                 "9a47642034002dd50000",
                                 "a2f599220ac5480000", "65c513fde30000"};
  ChromaFilter expected[5];
  expected[1].subset = 0;
  expected[1].coeffs[0] = 32;
  expected[2].subset = 2;
  expected[2].coeffs[0] = 32;
  expected[2].coeffs[1] = 32;
  expected[2].coeffs[2] = -3;
  expected[2].coeffs[12] = 1;
  expected[3] = expected[2];
  expected[3].uv_sym = true;
  expected[3].coeffs[13] = 1;
  expected[4].subset = 1;
  expected[4].uv_sym = true;
  expected[4].coeffs[6] = 4;
  expected[4].coeffs[7] = 4;
  expected[4].coeffs[8] = -2;
  expected[4].coeffs[9] = -2;
  for (int i = 0; i < 5; ++i) {
    std::vector<std::uint8_t> raw;
    raw.reserve(goldens[i].size() / 2);
    for (std::size_t c = 0; c < goldens[i].size(); c += 2) {
      raw.push_back(static_cast<std::uint8_t>(
          std::strtoul(goldens[i].substr(c, 2).c_str(), nullptr, 16)));
    }
    fs::LrState read_state;
    CHECK(chroma_init_tile_state(d, in, &read_state, &error), "read state");
    av2enc::av2::Decoder decoder;
    CHECK(decoder.init_symbol(raw.data(), raw.size(), raw.size()) ==
              av2enc::av2::Status::kOk,
          "decoder init");
    decoder.disable_cdf_update = 1;
    ChromaUnitPlan plan;
    CHECK(chroma_read_unit(decoder, read_state, 1, 0, 0, &plan, &error),
          "read unit");
    CHECK(plan.on == (i != 0), "read on");
    if (i != 0) {
      CHECK(chroma_filter_validate(plan.filter, &error), "read valid");
      CHECK(plan.filter.subset == expected[i].subset, "read subset");
      CHECK(plan.filter.uv_sym == expected[i].uv_sym, "read uv_sym");
      for (int j = 0; j < 18; ++j) {
        CHECK(plan.filter.coeffs[j] == expected[i].coeffs[j],
              "read coefficient");
      }
    }
  }

  // The shared filters_stream writer is the reference for the merged/bank and
  // coefficient order.  For a symmetric subset-2 filter it must produce the
  // same bytes as this chroma-specific writer.
  {
    ChromaFilter f = ChromaFilter{};
    f.subset = 2;
    f.uv_sym = true;
    f.coeffs[0] = 30;
    f.coeffs[1] = 30;
    f.coeffs[2] = -5;
    f.coeffs[3] = -5;
    f.coeffs[6] = 3;
    f.coeffs[7] = 3;
    f.coeffs[12] = 2;
    f.coeffs[13] = 2;
    std::vector<std::uint8_t> ours;
    CHECK(chroma_measure_unit_bytes(state, 1, 0, 0, f, true, &ours, &error),
          "ours bytes");
    fs::LrState ref_state;
    CHECK(chroma_init_tile_state(d, in, &ref_state, &error), "ref state");
    std::vector<std::uint8_t> buffer(1 << 16);
    av2enc::av2::Encoder encoder;
    CHECK(encoder.init_symbol(buffer.data(), buffer.size()) ==
              av2enc::av2::Status::kOk,
          "ref encoder");
    encoder.disable_cdf_update = 1;
    fs::LrUnitDecision decision;
    decision.type = av2enc::restoration::kRestoreWienerNs;
    decision.subset = f.subset;
    decision.uv_sym = f.uv_sym;
    for (int j = 0; j < 18; ++j) decision.coeffs[j] = f.coeffs[j];
    CHECK(fs::lr_write_unit(encoder, ref_state, 1, 0, 0, decision, &error),
          "ref write");
    std::size_t size = 0;
    CHECK(encoder.exit_symbol(&size) == av2enc::av2::Status::kOk, "ref exit");
    std::vector<std::uint8_t> ref(buffer.begin(),
                                  buffer.begin() +
                                      static_cast<std::ptrdiff_t>(size));
    CHECK(hex(ours) == hex(ref), "bytes match shared luma syntax");
  }

  // Merge: the second unit reuses the first bank entry, and the reader
  // reconstructs the same coefficients.
  {
    ChromaFilter f = ChromaFilter{};
    f.subset = 0;
    f.coeffs[2] = -4;
    f.coeffs[3] = 7;
    fs::LrState write_state;
    CHECK(chroma_init_tile_state(d, in, &write_state, &error), "merge state");
    std::vector<std::uint8_t> buffer(1 << 16);
    av2enc::av2::Encoder encoder;
    CHECK(encoder.init_symbol(buffer.data(), buffer.size()) ==
              av2enc::av2::Status::kOk,
          "merge encoder");
    CHECK(chroma_write_unit_decision(encoder, write_state, 1, 0, 0, f, true,
                                     &error),
          "merge first");
    ChromaFilter merged = f;
    merged.merged = true;
    merged.use_bank_k = 0;
    CHECK(chroma_write_unit_decision(encoder, write_state, 1, 1, 0, merged,
                                     true, &error),
          "merge second");
    std::size_t size = 0;
    CHECK(encoder.exit_symbol(&size) == av2enc::av2::Status::kOk, "merge exit");
    fs::LrState read_state;
    CHECK(chroma_init_tile_state(d, in, &read_state, &error), "merge read");
    av2enc::av2::Decoder decoder;
    CHECK(decoder.init_symbol(buffer.data(), size, size) ==
              av2enc::av2::Status::kOk,
          "merge decoder");
    decoder.disable_cdf_update = 1;
    ChromaUnitPlan p0;
    ChromaUnitPlan p1;
    CHECK(chroma_read_unit(decoder, read_state, 1, 0, 0, &p0, &error),
          "read first");
    CHECK(chroma_read_unit(decoder, read_state, 1, 1, 0, &p1, &error),
          "read second");
    CHECK(p1.filter.merged, "read merged");
    for (int j = 0; j < 18; ++j) {
      CHECK(p0.filter.coeffs[j] == p1.filter.coeffs[j], "merged coeffs");
    }
  }
}

void test_header() {
  using namespace av2enc::lr_chroma;
  std::string error;
  ChromaDecision d;
  d.shift = 2;
  d.unit_size = 64;
  int bits = 0;
  CHECK(chroma_measure_header_bits(d, 64, 90, &bits, &error),
        "header bits");
  CHECK(bits == 8, "header bit golden");

  classes::WienerLrHeaderConfig config;
  config.coded_lossless = false;
  config.enable_restoration = true;
  config.frame_is_intra = true;
  config.num_planes = 3;
  config.sb_size = 64;
  config.sub_x = 1;
  config.sub_y = 1;
  config.base_q_idx = 90;
  config.lr_tools_disable[0][rest::kRestorePcWiener] = true;
  config.lr_tools_disable[1][rest::kRestorePcWiener] = true;
  classes::WienerLrHeaderParams params;
  params.frame_type[0] = rest::kRestoreNone;
  params.frame_type[1] = rest::kRestoreWienerNs;
  params.frame_type[2] = rest::kRestoreWienerNs;
  params.chroma_shift = 2;
  classes::WienerReferenceSet refs;
  av2enc::BitWriter writer;
  CHECK(classes::wiener_write_lr_params(writer, config, &params, refs, &error),
        "header write");
  const std::size_t total = writer.bit_size();
  const auto bytes = writer.take_bytes();
  classes::WienerBitReader reader(bytes.data(), total);
  classes::WienerLrHeaderParams back;
  CHECK(classes::wiener_read_lr_params(reader, config, &back, refs, &error),
        "header read");
  CHECK(back.frame_type[0] == rest::kRestoreNone, "header luma type");
  CHECK(back.frame_type[1] == rest::kRestoreWienerNs, "header U type");
  CHECK(back.frame_type[2] == rest::kRestoreWienerNs, "header V type");
  CHECK(back.chroma_shift == 2, "header chroma shift");
  CHECK(!back.frame_filters_on[1] && !back.frame_filters_on[2],
        "header per-unit");
}

void test_uv_present() {
  using namespace av2enc::still::filters_stream;
  SequenceFilterConfig config;
  config.enable_restoration = true;
  config.lr_disable_luma_pc_wiener = true;
  config.lr_disable_luma_wiener_ns = false;
  config.lr_uv_present = true;
  config.lr_disable_chroma_wiener_ns = false;
  config.cdef_on_skip_txfm_disabled = true;
  av2enc::BitWriter writer;
  std::string error;
  const std::size_t bits =
      write_sequence_filter_config(writer, config, false, &error);
  CHECK(bits > 0, "sequence row bits");
  const auto bytes = writer.take_bytes();
  BitReader reader(std::span<const std::uint8_t>(bytes.data(), bytes.size()));
  SequenceFilterConfig back;
  CHECK(read_sequence_filter_config(reader, false, true, &back, &error),
        "sequence row read");
  CHECK(back.enable_restoration, "restoration enabled");
  CHECK(back.lr_uv_present, "uv present coded");
  CHECK(!back.lr_disable_chroma_wiener_ns, "chroma wiener enabled");

  // With lr_tools_uv_present = 0 the chroma flag is inferred from luma.
  SequenceFilterConfig inferred;
  inferred.enable_restoration = true;
  inferred.lr_disable_luma_pc_wiener = true;
  inferred.lr_disable_luma_wiener_ns = false;
  inferred.lr_uv_present = false;
  inferred.cdef_on_skip_txfm_disabled = true;
  av2enc::BitWriter w2;
  CHECK(write_sequence_filter_config(w2, inferred, false, &error) > 0,
        "inferred row");
  const auto b2 = w2.take_bytes();
  BitReader r2(std::span<const std::uint8_t>(b2.data(), b2.size()));
  SequenceFilterConfig back2;
  CHECK(read_sequence_filter_config(r2, false, true, &back2, &error),
        "inferred read");
  CHECK(!back2.lr_uv_present, "uv absent");
  CHECK(back2.lr_disable_chroma_wiener_ns ==
            back2.lr_disable_luma_wiener_ns,
        "chroma inferred from luma");
}

void test_decision() {
  using namespace av2enc::lr_chroma;
  const int width = 160;
  const int height = 96;
  const CheckFrame frame = make_check_frame(width, height, false);
  ChromaLrInput in = make_checker_input(width, height, false);
  bind(frame, &in);
  ChromaLrOptions options;
  options.forced_shift = 2;
  ChromaDecision decision;
  std::string error;
  CHECK(chroma_decide(in, options, &decision, &error), "decide");
  CHECK(decision.sse < decision.identity_sse, "decision improves SSE");
  CHECK(decision.sse < 0.985 * decision.identity_sse, "decision margin");

  int active = 0;
  int sym = 0;
  int nonzero = 0;
  for (const ChromaUnitPlan& plan : decision.units) {
    if (plan.on) ++active;
    if (plan.on && plan.filter.uv_sym) ++sym;
    if (plan.on && chroma_filter_validate(plan.filter, &error)) {
      for (int j = 0; j < rest::kWienerNsChromaCoeffs; ++j) {
        if (plan.filter.coeffs[j] != 0) {
          ++nonzero;
          break;
        }
      }
    }
  }
  CHECK(active > 0, "at least one unit on");
  CHECK(nonzero > 0, "at least one non-zero filter");
  (void)sym;
  CHECK(decision.tile_bits > 0, "tile bits measured");
  CHECK(decision.header_bits == 8, "header bits in decision");

  // The reference §7.20 core must reproduce the decision's filtering and
  // improve the real chroma planes.
  std::vector<std::uint8_t> out_u(static_cast<std::size_t>(width) * height);
  std::vector<std::uint8_t> out_v(static_cast<std::size_t>(width) * height);
  std::uint8_t* dst[2] = {out_u.data(), out_v.data()};
  int stride[2] = {width, width};
  CHECK(chroma_apply(in, decision, dst, stride) ==
            rest::RestorationStatus::kOk,
        "apply");
  double actual = 0.0;
  double identity = 0.0;
  for (std::size_t i = 0; i < out_u.size(); ++i) {
    const double du = static_cast<double>(out_u[i]) - frame.src_u[i];
    const double dv = static_cast<double>(out_v[i]) - frame.src_v[i];
    const double iu = static_cast<double>(frame.rec_u[i]) - frame.src_u[i];
    const double iv = static_cast<double>(frame.rec_v[i]) - frame.src_v[i];
    actual += du * du + dv * dv;
    identity += iu * iu + iv * iv;
  }
  CHECK(actual < identity, "apply improves SSE");
  CHECK(actual < 0.995 * identity, "apply margin");
  CHECK(std::fabs(actual - decision.sse) < 0.15 * identity,
        "decision SSE is the same order as the reference core");
}

void test_shared_uv() {
  using namespace av2enc::lr_chroma;
  const int width = 128;
  const int height = 96;
  const CheckFrame frame = make_check_frame(width, height, true);
  ChromaLrInput in = make_checker_input(width, height, true);
  bind(frame, &in);
  ChromaLrOptions options;
  options.forced_shift = 2;
  options.allow_shared_uv = true;
  ChromaDecision decision;
  std::string error;
  CHECK(chroma_decide(in, options, &decision, &error), "shared decide");
  CHECK(decision.shared_uv, "shared U/V selected for identical planes");
  for (std::size_t i = 0; i < decision.units.size(); ++i) {
    for (std::size_t j = i + 1; j < decision.units.size(); ++j) {
      const ChromaUnitPlan& a = decision.units[i];
      const ChromaUnitPlan& b = decision.units[j];
      if (a.plane == 1 && b.plane == 2 && a.unit_row == b.unit_row &&
          a.unit_col == b.unit_col) {
        CHECK(a.on == b.on, "shared on");
        CHECK(a.filter.subset == b.filter.subset, "shared subset");
        CHECK(a.filter.uv_sym == b.filter.uv_sym, "shared uv_sym");
        CHECK(std::memcmp(a.filter.coeffs, b.filter.coeffs,
                          sizeof(a.filter.coeffs)) == 0,
              "shared coefficients");
      }
    }
  }
}

void test_apply_off_is_identity() {
  using namespace av2enc::lr_chroma;
  const int width = 64;
  const int height = 64;
  const CheckFrame frame = make_check_frame(width, height, false);
  ChromaLrInput in = make_checker_input(width, height, false);
  bind(frame, &in);
  ChromaDecision decision = empty_decision(width, height, 2);
  // All units stay RESTORE_NONE.
  std::vector<std::uint8_t> out_u(static_cast<std::size_t>(width) * height);
  std::vector<std::uint8_t> out_v(static_cast<std::size_t>(width) * height);
  std::uint8_t* dst[2] = {out_u.data(), out_v.data()};
  int stride[2] = {width, width};
  CHECK(chroma_apply(in, decision, dst, stride) ==
            rest::RestorationStatus::kOk,
        "apply off");
  CHECK(out_u == frame.rec_u, "off U identity");
  CHECK(out_v == frame.rec_v, "off V identity");
}

}  // namespace

int main() {
  test_unit_sizing();
  test_filter_model();
  test_syntax_goldens();
  test_header();
  test_uv_present();
  test_decision();
  test_shared_uv();
  test_apply_off_is_identity();
  if (g_failures != 0) {
    std::fprintf(stderr, "chroma_lr_test: %d checks, %d failures\n", g_checks,
                 g_failures);
    return 1;
  }
  std::printf("PASS: chroma_lr_test %d checks\n", g_checks);
  return 0;
}

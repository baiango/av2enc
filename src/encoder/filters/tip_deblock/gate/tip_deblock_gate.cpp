// §7.16 TIP deblocking gate.
//
// Two modes:
//
// 1. Default — synthetic TIP-sized grid.
//    There is no TIP stream in this tree yet: M2 writes enable_tip = 0 and no
//    OBU_REGULAR_TIP / TIP_FRAME_AS_OUTPUT writer exists.  This mode therefore
//    builds a TIP-shaped reconstruction (block-structured, 8x8 and 16x16
//    discontinuities), runs §7.16 for both tipSize values, and checks the
//    structural invariants: only TIP-grid edges are touched, guards are
//    clean, output is deterministic, flat content is unchanged.
//
// 2. `--tip-ivf <stream.ivf> --recon-y4m <encoder-recon.y4m>`
//    The real `dav2d --strict 1 --threads 1` gate.  It is ready for the TIP
//    stream agent's output: the stream must contain an OBU_REGULAR_TIP
//    (type 14, or OBU_LEADING_TIP type 13) with TipFrameMode ==
//    TIP_FRAME_AS_OUTPUT and apply_deblocking_filter_tip == 1, and the Y4M
//    must be the encoder's reconstruction *after* §7.16.  The gate decodes the
//    stream, requires at least one TIP OBU, and requires zero Y/U/V sample
//    differences.
//
// SEAM (closing the decoder gate):
//   * enable_tip = 1, disable_tip_output = 0 and (for the flag) enable_df_sub_pu = 1
//     in the sequence header;
//   * TipFrameMode == TIP_FRAME_AS_OUTPUT, allow_df_sub_pu = 1 and
//     apply_deblocking_filter_tip = 1 in the TIP frame header;
//   * the encoder builds the §7.10.6 TIP frame, calls
//     tip_deblock_frame() exactly once (and does not call §7.17), and dumps
//     that frame as the Y4M recon.  No CDEF/CCSO/LR are applied.
//   The TIP stream agent owns the first two items and the §7.10.6 frame; this
//   gate owns the §7.16 comparison.
//
// Exit: 0 pass, 1 fail, 77 dav2d unavailable.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "encoder/filters/tip_deblock/tip_deblock.h"
#include "encoder/filters/tip_deblock/tip_deblock_test_common.h"

namespace {

using av2enc::tip_deblock::TipDeblockFrame;
using av2enc::tip_deblock::TipDeblockOptions;
using av2enc::tip_deblock::TipDeblockStatus;
using av2enc::tip_deblock::TipFrameMode;
using namespace av2enc::tip_deblock::test;

int g_failures = 0;

#define GATE_CHECK(cond, msg)                                              \
  do {                                                                     \
    if (!(cond)) {                                                         \
      ++g_failures;                                                        \
      std::fprintf(stderr, "GATE FAIL %s:%d: %s\n", __FILE__, __LINE__,    \
                   (msg));                                                 \
    }                                                                      \
  } while (0)

TipDeblockOptions gate_options(bool tip8) {
  TipDeblockOptions options;
  options.tip_frame_mode = TipFrameMode::kAsOutput;
  options.apply_deblocking_filter_tip = true;
  options.enable_df_sub_pu = true;
  options.allow_df_sub_pu = true;
  options.disable_loopfilters_across_tiles = false;
  options.base_q_idx = 200;
  options.bit_depth = 8;
  options.enable_tip_refinemv = tip8;
  options.tip_interp_filter = av2enc::tip_deblock::InterpFilter::kEighttapSharp;
  return options;
}

// Block-structured frame reminiscent of a §7.10.6 TIP output: constant-ish
// 8x8 luma blocks with alternating levels, so both the 8x8 and the 16x16 TIP
// edge lattices contain real discontinuities.
TipTestFrame make_synthetic_tip_grid(int mi_cols, int mi_rows) {
  TipTestFrame frame = make_frame(mi_cols, mi_rows);
  for (int plane = 0; plane < 3; ++plane) {
    const int sub = (plane == 0) ? 0 : 1;
    const int block = 8 >> sub;
    fill_plane(frame, plane, [block](int x, int y) {
      const int bx = x / block;
      const int by = y / block;
      int value = ((bx + by) & 1) ? 168 : 88;
      value += ((x * 3 + y * 5) & 7) - 4;  // break exact ties
      return std::max(0, std::min(255, value));
    });
  }
  return frame;
}

void report_plane_changes(const TipTestFrame& before, const TipTestFrame& after,
                          const char* label) {
  const std::size_t y = count_changed_in_plane(before, after, 0);
  const std::size_t u = count_changed_in_plane(before, after, 1);
  const std::size_t v = count_changed_in_plane(before, after, 2);
  std::printf("  %s: Y %zu, U %zu, V %zu samples changed (max |d| %d)\n",
              label, y, u, v, max_abs_delta(before, after));
}

int run_synthetic_gate() {
  const int mi_cols = 80;  // 320 luma
  const int mi_rows = 48;  // 192 luma
  std::printf("synthetic TIP grid: %dx%d luma (mi %dx%d), q=200\n", mi_cols * 4,
              mi_rows * 4, mi_cols, mi_rows);

  for (int tip8 = 0; tip8 <= 1; ++tip8) {
    TipTestFrame frame = make_synthetic_tip_grid(mi_cols, mi_rows);
    const TipTestFrame before = frame;
    const TipDeblockOptions options = gate_options(tip8 != 0);

    WriteMask mask;
    build_write_mask(frame, options, &mask);

    TipDeblockFrame view = frame.view();
    const TipDeblockStatus status =
        av2enc::tip_deblock::tip_deblock_frame(view, options);
    GATE_CHECK(status == TipDeblockStatus::kOk, "synthetic §7.16 run");
    GATE_CHECK(guard_intact(frame), "synthetic guard samples intact");
    GATE_CHECK(count_changed_in_plane(before, frame, 0) > 0,
               "synthetic luma is filtered");
    GATE_CHECK(count_changed_in_plane(before, frame, 1) > 0,
               "synthetic U is filtered");
    GATE_CHECK(count_changed_in_plane(before, frame, 2) > 0,
               "synthetic V is filtered");
    int plane = 0;
    int px = 0;
    int py = 0;
    GATE_CHECK(!changed_pixels_outside_mask(before, frame, mask, &plane, &px,
                                            &py),
               "only TIP-grid edges are modified");
    GATE_CHECK(checksum(frame) != checksum(before), "synthetic frame changes");

    TipTestFrame rerun = before;
    TipDeblockFrame rerun_view = rerun.view();
    GATE_CHECK(av2enc::tip_deblock::tip_deblock_frame(rerun_view, options) ==
                   TipDeblockStatus::kOk,
               "synthetic rerun");
    GATE_CHECK(checksum(rerun) == checksum(frame),
               "synthetic §7.16 is deterministic");
    report_plane_changes(before, frame, tip8 ? "tipSize 8x8 " : "tipSize 16x16");
  }

  // Flat content must be untouched: a TIP frame with no block discontinuity
  // has no §7.16 work to do.
  {
    TipTestFrame flat = make_frame(16, 16);
    fill_flat(flat, 128);
    const TipTestFrame before = flat;
    TipDeblockOptions options = gate_options(false);
    TipDeblockFrame view = flat.view();
    GATE_CHECK(av2enc::tip_deblock::tip_deblock_frame(view, options) ==
                   TipDeblockStatus::kOk,
               "flat synthetic run");
    GATE_CHECK(count_changed(before, flat) == 0, "flat synthetic unchanged");
    std::printf("  flat content: unchanged\n");
  }

  if (g_failures != 0) {
    std::fprintf(stderr, "SYNTHETIC GATE FAILED (%d)\n", g_failures);
    return 1;
  }
  std::printf("PASS: synthetic §7.16 TIP grid\n");
  return 0;
}

// ---------------------------------------------------------------------------
// Real dav2d gate support.
// ---------------------------------------------------------------------------

struct Y4mFrame {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
};

bool read_y4m(const std::string& path, std::vector<Y4mFrame>* frames,
              std::string* error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    *error = "cannot open " + path;
    return false;
  }
  std::string line;
  if (!std::getline(in, line) || line.rfind("YUV4MPEG2", 0) != 0) {
    *error = "not a Y4M file: " + path;
    return false;
  }
  int width = 0;
  int height = 0;
  bool c420 = false;
  std::istringstream header(line);
  std::string token;
  while (header >> token) {
    if (token.size() <= 1) continue;
    const std::string value = token.substr(1);
    if (token[0] == 'W') width = std::stoi(value);
    if (token[0] == 'H') height = std::stoi(value);
    if (token[0] == 'C' &&
        (value == "420" || value == "420jpeg" || value == "420mpeg2")) {
      c420 = true;
    }
  }
  if (width <= 0 || height <= 0 || !c420) {
    *error = "only 8-bit 4:2:0 Y4M is supported";
    return false;
  }
  const int cw = width / 2;
  const int ch = height / 2;
  while (std::getline(in, line)) {
    if (line.rfind("FRAME", 0) != 0) {
      break;
    }
    Y4mFrame frame;
    frame.width = width;
    frame.height = height;
    frame.y.resize(static_cast<std::size_t>(width) *
                   static_cast<std::size_t>(height));
    frame.u.resize(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch));
    frame.v.resize(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch));
    in.read(reinterpret_cast<char*>(frame.y.data()),
            static_cast<std::streamsize>(frame.y.size()));
    in.read(reinterpret_cast<char*>(frame.u.data()),
            static_cast<std::streamsize>(frame.u.size()));
    in.read(reinterpret_cast<char*>(frame.v.data()),
            static_cast<std::streamsize>(frame.v.size()));
    if (!in) {
      *error = "truncated frame in " + path;
      return false;
    }
    frames->push_back(std::move(frame));
  }
  if (frames->empty()) {
    *error = "no frames in " + path;
    return false;
  }
  return true;
}

std::uint64_t leb128(const std::vector<std::uint8_t>& data, std::size_t* pos) {
  std::uint64_t value = 0;
  int shift = 0;
  while (*pos < data.size()) {
    const std::uint8_t byte = data[(*pos)++];
    value |= static_cast<std::uint64_t>(byte & 0x7F) << shift;
    if ((byte & 0x80) == 0) {
      return value;
    }
    shift += 7;
  }
  return 0;
}

// Counts TIP OBUs (OBU_LEADING_TIP = 13, OBU_REGULAR_TIP = 14) in an IVF.
bool count_tip_obus(const std::string& path, int* tip_obus, int* units,
                    std::string* error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    *error = "cannot open " + path;
    return false;
  }
  char header[32] = {};
  in.read(header, 32);
  if (!in || std::memcmp(header, "DKIF", 4) != 0) {
    *error = "not an IVF file: " + path;
    return false;
  }
  *tip_obus = 0;
  *units = 0;
  while (true) {
    std::uint32_t size = 0;
    in.read(reinterpret_cast<char*>(&size), 4);
    if (!in) break;
    char timestamp[8] = {};
    in.read(timestamp, 8);
    if (!in) break;
    std::vector<std::uint8_t> payload(size);
    in.read(reinterpret_cast<char*>(payload.data()),
            static_cast<std::streamsize>(size));
    if (!in) {
      *error = "truncated IVF unit";
      return false;
    }
    ++*units;
    std::size_t pos = 0;
    while (pos < payload.size()) {
      const std::uint64_t obu_size = leb128(payload, &pos);
      if (obu_size == 0 || pos >= payload.size()) break;
      if (pos + obu_size > payload.size()) break;
      const int type = (payload[pos] >> 2) & 0x1F;
      if (type == 13 || type == 14) {
        ++*tip_obus;
      }
      pos += static_cast<std::size_t>(obu_size);
    }
  }
  return true;
}

std::string find_dav2d(const std::string& repo_root,
                       const std::string& explicit_path) {
  if (!explicit_path.empty()) {
    return explicit_path;
  }
  if (const char* env = std::getenv("DAV2D")) {
    if (env[0] != '\0') {
      return env;
    }
  }
  const char* names[] = {"dav2d", "dav2d-macos"};
  for (const char* name : names) {
    const std::string candidate = repo_root + "/tools/bin/" + name;
    std::ifstream probe(candidate, std::ios::binary);
    if (probe.good()) {
      return candidate;
    }
  }
  return {};
}

int run_dav2d_gate(const std::string& ivf, const std::string& recon_y4m,
                   const std::string& dav2d_path,
                   const std::string& repo_root) {
  std::string error;
  int tip_obus = 0;
  int units = 0;
  if (!count_tip_obus(ivf, &tip_obus, &units, &error)) {
    std::fprintf(stderr, "dav2d gate: %s\n", error.c_str());
    return 1;
  }
  if (tip_obus == 0) {
    std::fprintf(stderr,
                 "dav2d gate: %s has no OBU_REGULAR_TIP/LEADING_TIP OBU; "
                 "this gate requires a TIP_FRAME_AS_OUTPUT stream\n",
                 ivf.c_str());
    return 1;
  }

  const std::string decoder = find_dav2d(repo_root, dav2d_path);
  if (decoder.empty()) {
    std::fprintf(stderr,
                 "dav2d gate: decoder not found (use --dav2d or DAV2D)\n");
    return 77;
  }

  const std::string tmp_dir =
      std::getenv("TMPDIR") != nullptr ? std::getenv("TMPDIR") : "/tmp";
  const std::string decoded_path =
      tmp_dir + "/tip_deblock_gate_decoded.y4m";
  const std::string command = "\"" + decoder + "\" -q -i \"" + ivf +
                              "\" -o \"" + decoded_path +
                              "\" --strict 1 --threads 1";
  std::printf("dav2d: %s\n", command.c_str());
  const int status = std::system(command.c_str());
  if (status != 0) {
    std::fprintf(stderr, "dav2d --strict 1 failed on %s\n", ivf.c_str());
    return 1;
  }

  std::vector<Y4mFrame> decoded;
  std::vector<Y4mFrame> recon;
  if (!read_y4m(decoded_path, &decoded, &error) ||
      !read_y4m(recon_y4m, &recon, &error)) {
    std::fprintf(stderr, "dav2d gate: %s\n", error.c_str());
    std::remove(decoded_path.c_str());
    return 1;
  }
  std::remove(decoded_path.c_str());

  if (decoded.size() != recon.size()) {
    std::fprintf(stderr,
                 "dav2d gate: decoder produced %zu frames, recon has %zu\n",
                 decoded.size(), recon.size());
    return 1;
  }

  std::size_t diffs = 0;
  for (std::size_t i = 0; i < decoded.size(); ++i) {
    const Y4mFrame& a = decoded[i];
    const Y4mFrame& b = recon[i];
    if (a.width != b.width || a.height != b.height) {
      std::fprintf(stderr, "dav2d gate: frame %zu dimension mismatch\n", i);
      return 1;
    }
    for (std::size_t k = 0; k < a.y.size(); ++k) diffs += a.y[k] != b.y[k];
    for (std::size_t k = 0; k < a.u.size(); ++k) diffs += a.u[k] != b.u[k];
    for (std::size_t k = 0; k < a.v.size(); ++k) diffs += a.v[k] != b.v[k];
  }
  std::printf("dav2d --strict 1 --threads 1: %zu frame(s), %d TIP OBU(s), "
              "%zu sample diffs vs %s\n",
              decoded.size(), tip_obus, diffs, recon_y4m.c_str());
  if (diffs != 0) {
    std::fprintf(stderr, "dav2d gate: reconstruction mismatch\n");
    return 1;
  }
  std::printf("PASS: dav2d §7.16 TIP deblock gate\n");
  return 0;
}

void print_seam() {
  std::printf(
      "\nSEAM: §7.16 is gated only structurally.  No encoder TIP stream exists\n"
      "yet:\n"
      "  * src/encoder/m2/m2_headers.cpp writes enable_tip = 0;\n"
      "  * nothing emits OBU_REGULAR_TIP / TipFrameMode == TIP_FRAME_AS_OUTPUT;\n"
      "  * the M2 production path writes enable_df_sub_pu = 0, so\n"
      "    apply_deblocking_filter_tip cannot be coded.\n"
      "A decoder test vector with TIP OBUs does exist\n"
      "(testdata/sample_480x270_130f.ivf: OBU type 14, 70 IVF units, 130 output\n"
      "frames, `dav2d --strict 1 --threads 1` clean).  It cannot close this\n"
      "gate by itself: dav2d exposes only whole-chain --inloopfilters toggles,\n"
      "so `none` vs `deblock` also changes every reference frame; isolating\n"
      "§7.16 needs the encoder's pre-filter TIP frame (or a decoder hook).\n"
      "When the TIP stream agent lands, run the decoder gate with:\n"
      "  tip_deblock_gate --tip-ivf <stream.ivf> --recon-y4m <recon.y4m>\n"
      "The recon must be the encoder's output after §7.16, and the frame must\n"
      "not be passed through §7.17/CDEF/CCSO/LR.\n");
}

}  // namespace

int main(int argc, char** argv) {
  std::string tip_ivf;
  std::string recon_y4m;
  std::string dav2d;
  std::string repo_root = "../../../..";

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--tip-ivf" && i + 1 < argc) {
      tip_ivf = argv[++i];
    } else if (arg == "--recon-y4m" && i + 1 < argc) {
      recon_y4m = argv[++i];
    } else if (arg == "--dav2d" && i + 1 < argc) {
      dav2d = argv[++i];
    } else if (arg == "--repo-root" && i + 1 < argc) {
      repo_root = argv[++i];
    } else if (arg == "-h" || arg == "--help") {
      std::printf(
          "usage: tip_deblock_gate [--tip-ivf stream.ivf --recon-y4m recon.y4m]\n"
          "                       [--dav2d path] [--repo-root path]\n");
      return 0;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      return 2;
    }
  }

  const int synthetic = run_synthetic_gate();
  if (synthetic != 0) {
    return synthetic;
  }

  if (tip_ivf.empty()) {
    print_seam();
    return 0;
  }
  if (recon_y4m.empty()) {
    std::fprintf(stderr, "--tip-ivf also requires --recon-y4m\n");
    return 2;
  }
  return run_dav2d_gate(tip_ivf, recon_y4m, dav2d, repo_root);
}

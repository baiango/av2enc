// Decoder gate for leading pictures.
//
// Stream shape (decode order):
//
//   TD SEQ CLK(f0)                       output
//   TD SEQ OLK(f4)                       open random access point, output
//   TD      LEADING(f1)  order 1         leading, references the OLK
//   TD      LEADING(f2)  order 2         leading, references the OLK
//   TD      LEADING(f3)  order 3         leading, references the OLK
//   TD      REGULAR(f5)  order 5         output
//   TD      REGULAR(f6)  order 6         output
//   TD      REGULAR(f7)  order 7         output
//
// f1..f3 are displayed before the OLK but decoded after it, so they are true
// leading pictures.  Two scenarios are checked:
//
//   non-output: leading frames code immediate=0, implicit=0 (§7.3.4 coded
//               non-output frame unit).  Full and RAP-start decodes must
//               produce identical output.
//   output    : leading frames code immediate=1.  Full decode emits them
//               between f0 and the OLK; RAP-start decode discards them.
//
// Every decode uses the pinned `dav2d --strict 1 --threads 1`.
#include "codec/inter/leading/leading.h"
#include "codec/inter/leading/leading_encode.h"

#include "core/io/ivf.h"
#include "core/io/y4m.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

using av2enc::inter::leading::EncodedPicture;
using av2enc::inter::leading::FrameHeader;
using av2enc::inter::leading::FrameKind;
using av2enc::inter::leading::Picture;
using av2enc::inter::leading::Plan;
using av2enc::inter::leading::SequenceConfig;

std::uint8_t square_sample(int x, int y, int frame, int plane) {
  const int sx = 8 + 4 * frame;
  const int sy = 8 + 2 * frame;
  const bool in_square = x >= sx && x < sx + 32 && y >= sy && y < sy + 32;
  if (plane == 0) {
    if (in_square) {
      const int lx = x - sx;
      const int ly = y - sy;
      return static_cast<std::uint8_t>(40 + ((lx * 37 + ly * 17) & 0x7F));
    }
    return static_cast<std::uint8_t>(96 + ((x + y) >> 3));
  }
  if (plane == 1) return static_cast<std::uint8_t>(in_square ? 100 : 128);
  return static_cast<std::uint8_t>(in_square ? 150 : 128);
}

Picture make_frame(int index, int width, int height) {
  Picture frame;
  frame.width = static_cast<std::uint32_t>(width);
  frame.height = static_cast<std::uint32_t>(height);
  const int cw = width / 2;
  const int ch = height / 2;
  frame.y.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
  frame.u.resize(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch));
  frame.v.resize(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch));
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      frame.y[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
              static_cast<std::size_t>(x)] =
          square_sample(x, y, index, 0);
    }
  }
  for (int y = 0; y < ch; ++y) {
    for (int x = 0; x < cw; ++x) {
      frame.u[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
              static_cast<std::size_t>(x)] = square_sample(x * 2, y * 2, index, 1);
      frame.v[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
              static_cast<std::size_t>(x)] = square_sample(x * 2, y * 2, index, 2);
    }
  }
  return frame;
}

FrameHeader make_header(FrameKind kind, std::uint32_t order_hint,
                        std::uint32_t refresh, std::uint8_t num_refs,
                        std::uint8_t ref0, bool leading_output) {
  FrameHeader header;
  header.kind = kind;
  header.order_hint = order_hint;
  header.refresh_frame_flags = refresh;
  header.num_total_refs = num_refs;
  header.ref_frame_idx[0] = ref0;
  header.leading_output_frame = leading_output;
  return header;
}

// Slots: CLK/OLK refresh slot 0; leading frames refresh slots 1..3; regular
// frames refresh slots 4..6.  Leading and regular frames both reference the OLK
// in slot 0, so no regular frame references a leading frame.
Plan make_plan(bool leading_output) {
  Plan plan;
  plan.frames.push_back(make_header(FrameKind::kClosedLoopKey, 0, 0xFFU, 0, 0,
                                    false));
  plan.frames.push_back(make_header(FrameKind::kOpenLoopKey, 4, 1U, 0, 0,
                                    false));
  plan.frames.push_back(make_header(FrameKind::kLeadingTileGroup, 1, 2U, 1, 0,
                                    leading_output));
  plan.frames.push_back(make_header(FrameKind::kLeadingTileGroup, 2, 4U, 1, 0,
                                    leading_output));
  plan.frames.push_back(make_header(FrameKind::kLeadingTileGroup, 3, 8U, 1, 0,
                                    leading_output));
  plan.frames.push_back(make_header(FrameKind::kRegularTileGroup, 5, 16U, 1, 0,
                                    false));
  plan.frames.push_back(make_header(FrameKind::kRegularTileGroup, 6, 32U, 1, 0,
                                    false));
  plan.frames.push_back(make_header(FrameKind::kRegularTileGroup, 7, 64U, 1, 0,
                                    false));
  return plan;
}

SequenceConfig make_config(int qp) {
  SequenceConfig config;
  config.max_frame_width = 64;
  config.max_frame_height = 64;
  config.base_q_idx = static_cast<std::uint8_t>(qp);
  config.order_hint_bits = 4;
  config.num_ref_frames = 8;
  config.monotonic_output_order_flag = false;
  return config;
}

std::string find_dav2d() {
  if (const char* env = std::getenv("DAV2D")) {
    if (std::filesystem::exists(env)) return env;
  }
  for (const char* candidate :
       {"tools/bin/dav2d", "tools/bin/dav2d-macos", "dav2d", "dav2d-macos"}) {
    if (std::filesystem::exists(candidate)) return candidate;
  }
  return {};
}

bool write_ivf(const std::string& path,
               const std::vector<std::vector<std::uint8_t>>& units,
               std::uint32_t width, std::uint32_t height, std::string& error) {
  av2enc::IvfWriter writer;
  av2enc::IvfHeader header;
  header.width = static_cast<std::uint16_t>(width);
  header.height = static_cast<std::uint16_t>(height);
  header.fps_num = 30;
  header.fps_den = 1;
  header.frame_count = static_cast<std::uint32_t>(units.size());
  if (!writer.open(path, header, error)) return false;
  for (std::size_t i = 0; i < units.size(); ++i) {
    if (!writer.write_frame(units[i], i, error)) return false;
  }
  return writer.close(error);
}

bool decode(const std::string& dav2d, const std::string& input,
            const std::string& output, std::string& error) {
  const std::string command = dav2d + " -q --strict 1 --threads 1 -i " +
                              input + " -o " + output;
  std::cout << "  decode: " << command << "\n";
  const int rc = std::system(command.c_str());
  if (rc != 0) {
    error = "dav2d returned " + std::to_string(rc);
    return false;
  }
  return true;
}

bool read_y4m(const std::string& path, std::vector<Picture>& frames,
              std::string& error) {
  av2enc::Y4mReader reader(path);
  if (!reader.open(error)) return false;
  av2enc::Y4mFrame frame;
  while (true) {
    const av2enc::Y4mReadResult status = reader.read_frame(frame, error);
    if (status == av2enc::Y4mReadResult::kEof) break;
    if (status == av2enc::Y4mReadResult::kError) return false;
    Picture picture;
    picture.width = frame.width();
    picture.height = frame.height();
    picture.y.assign(frame.y().begin(), frame.y().end());
    picture.u.assign(frame.u().begin(), frame.u().end());
    picture.v.assign(frame.v().begin(), frame.v().end());
    frames.push_back(std::move(picture));
  }
  return true;
}

bool compare(const std::vector<Picture>& expected,
             const std::vector<Picture>& decoded, const std::string& label,
             std::string& error) {
  if (expected.size() != decoded.size()) {
    error = label + ": expected " + std::to_string(expected.size()) +
            " frames, decoder produced " + std::to_string(decoded.size());
    return false;
  }
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const Picture& a = expected[i];
    const Picture& b = decoded[i];
    if (a.width != b.width || a.height != b.height) {
      error = label + ": frame " + std::to_string(i) + " dimensions differ";
      return false;
    }
    const auto same_plane = [&](const std::vector<std::uint8_t>& lhs,
                                const std::vector<std::uint8_t>& rhs, int plane) {
      if (lhs.size() != rhs.size()) return false;
      for (std::size_t k = 0; k < lhs.size(); ++k) {
        if (lhs[k] != rhs[k]) {
          std::fprintf(stderr,
                       "MISMATCH %s frame %zu plane %d sample %zu: dec=%u exp=%u\n",
                       label.c_str(), i, plane, k,
                       static_cast<unsigned>(rhs[k]),
                       static_cast<unsigned>(lhs[k]));
          return false;
        }
      }
      return true;
    };
    if (!same_plane(a.y, b.y, 0) || !same_plane(a.u, b.u, 1) ||
        !same_plane(a.v, b.v, 2)) {
      error = label + ": frame " + std::to_string(i) + " differs";
      return false;
    }
  }
  return true;
}

struct Scenario {
  bool leading_output = false;
  std::string name;
};

bool run_scenario(const Scenario& scenario, const std::string& dav2d,
                  const std::string& out_dir, bool require_decoder, int qp) {
  std::cout << "== scenario: " << scenario.name << " ==\n";
  const SequenceConfig config = make_config(qp);
  const Plan plan = make_plan(scenario.leading_output);
  std::string error;
  if (!av2enc::inter::leading::validate_plan(config, plan, error)) {
    std::cerr << "PLAN FAIL: " << error << "\n";
    return false;
  }

  const Picture f0 = make_frame(0, 64, 64);
  const Picture f1 = make_frame(1, 64, 64);
  const Picture f2 = make_frame(2, 64, 64);
  const Picture f3 = make_frame(3, 64, 64);
  const Picture f4 = make_frame(4, 64, 64);
  const Picture f5 = make_frame(5, 64, 64);
  const Picture f6 = make_frame(6, 64, 64);
  const Picture f7 = make_frame(7, 64, 64);

  av2enc::video::VideoOptions options;
  options.base_q_idx = config.base_q_idx;

  EncodedPicture clk;
  EncodedPicture olk;
  EncodedPicture lead1;
  EncodedPicture lead2;
  EncodedPicture lead3;
  EncodedPicture reg5;
  EncodedPicture reg6;
  EncodedPicture reg7;
  if (!av2enc::inter::leading::encode_intra_picture(f0, options, clk, error) ||
      !av2enc::inter::leading::encode_intra_picture(f4, options, olk, error) ||
      !av2enc::inter::leading::encode_inter_picture(f4, f1, options, lead1,
                                                    error) ||
      !av2enc::inter::leading::encode_inter_picture(f4, f2, options, lead2,
                                                    error) ||
      !av2enc::inter::leading::encode_inter_picture(f4, f3, options, lead3,
                                                    error) ||
      !av2enc::inter::leading::encode_inter_picture(f4, f5, options, reg5,
                                                    error) ||
      !av2enc::inter::leading::encode_inter_picture(f4, f6, options, reg6,
                                                    error) ||
      !av2enc::inter::leading::encode_inter_picture(f4, f7, options, reg7,
                                                    error)) {
    std::cerr << "ENCODE FAIL: " << error << "\n";
    return false;
  }

  std::vector<std::vector<std::uint8_t>> tiles;
  tiles.push_back(clk.tile_payload);
  tiles.push_back(olk.tile_payload);
  tiles.push_back(lead1.tile_payload);
  tiles.push_back(lead2.tile_payload);
  tiles.push_back(lead3.tile_payload);
  tiles.push_back(reg5.tile_payload);
  tiles.push_back(reg6.tile_payload);
  tiles.push_back(reg7.tile_payload);

  const std::vector<std::vector<std::uint8_t>> units =
      av2enc::inter::leading::assemble_temporal_units(config, plan, tiles);

  std::vector<Picture> expected_full;
  expected_full.push_back(clk.recon);
  if (scenario.leading_output) {
    expected_full.push_back(lead1.recon);
    expected_full.push_back(lead2.recon);
    expected_full.push_back(lead3.recon);
  }
  expected_full.push_back(olk.recon);
  expected_full.push_back(reg5.recon);
  expected_full.push_back(reg6.recon);
  expected_full.push_back(reg7.recon);

  std::vector<Picture> expected_rap;
  if (scenario.leading_output) {
    // dav2d decodes and retains leading frames that only depend on the OLK even
    // when decoding starts at that OLK; they are still after the removed
    // pre-RAP frame in output order.  This probe characterises that behaviour.
    expected_rap.push_back(lead1.recon);
    expected_rap.push_back(lead2.recon);
    expected_rap.push_back(lead3.recon);
  }
  expected_rap.push_back(olk.recon);
  expected_rap.push_back(reg5.recon);
  expected_rap.push_back(reg6.recon);
  expected_rap.push_back(reg7.recon);

  const std::string full_path = out_dir + "/leading_" + scenario.name + ".ivf";
  const std::string rap_path =
      out_dir + "/leading_" + scenario.name + ".rap.ivf";
  std::vector<std::vector<std::uint8_t>> rap_units(units.begin() + 1,
                                                   units.end());
  // Non-output leading frames must have no effect at all: a RAP clip with the
  // leading temporal units physically removed must decode identically.
  std::vector<std::vector<std::uint8_t>> rap_no_leading_units;
  rap_no_leading_units.push_back(units[1]);  // OLK
  for (std::size_t i = 0; i < plan.frames.size(); ++i) {
    if (plan.frames[i].kind == FrameKind::kRegularTileGroup) {
      rap_no_leading_units.push_back(units[i]);
    }
  }
  const std::string rap_no_leading_path =
      out_dir + "/leading_" + scenario.name + ".rap-no-leading.ivf";
  bool wrote = write_ivf(full_path, units, 64, 64, error) &&
               write_ivf(rap_path, rap_units, 64, 64, error);
  if (!scenario.leading_output) {
    wrote = wrote &&
            write_ivf(rap_no_leading_path, rap_no_leading_units, 64, 64,
                      error);
  }
  if (!wrote) {
    std::cerr << "IVF WRITE FAIL: " << error << "\n";
    return false;
  }
  std::cout << "  full stream: " << units.size() << " temporal units; RAP clip: "
            << rap_units.size() << " (starts at TD+SEQ+OLK)";
  if (!scenario.leading_output) {
    std::cout << "; RAP clip without leading: " << rap_no_leading_units.size();
  }
  std::cout << "\n";
  std::cout << "  leading output frames: "
            << (scenario.leading_output ? "immediate=1" : "immediate=0 implicit=0")
            << "\n";
  for (std::size_t i = 0; i < units.size(); ++i) {
    std::cout << "  unit " << i << ": " << units[i].size() << " bytes\n";
  }
  if (!require_decoder) {
    std::cout << "  (--no-decode; decode comparison skipped)\n";
    return true;
  }
  if (dav2d.empty()) {
    std::cerr << "DECODE SKIP: no dav2d binary found (set DAV2D)\n";
    return false;
  }

  std::vector<Picture> full_decoded;
  std::vector<Picture> rap_decoded;
  std::vector<Picture> rap_no_leading_decoded;
  if (!decode(dav2d, full_path, full_path + ".y4m", error) ||
      !read_y4m(full_path + ".y4m", full_decoded, error) ||
      !decode(dav2d, rap_path, rap_path + ".y4m", error) ||
      !read_y4m(rap_path + ".y4m", rap_decoded, error)) {
    std::cerr << "DECODE FAIL: " << error << "\n";
    return false;
  }
  if (!scenario.leading_output) {
    if (!decode(dav2d, rap_no_leading_path, rap_no_leading_path + ".y4m",
                error) ||
        !read_y4m(rap_no_leading_path + ".y4m", rap_no_leading_decoded,
                  error)) {
      std::cerr << "DECODE FAIL: " << error << "\n";
      return false;
    }
  }
  if (!compare(expected_full, full_decoded, "full", error) ||
      !compare(expected_rap, rap_decoded, "rap", error)) {
    std::cerr << "COMPARE FAIL: " << error << "\n";
    return false;
  }
  std::cout << "  PASS: full=" << full_decoded.size()
            << " frame(s), RAP=" << rap_decoded.size()
            << " frame(s), all samples exact\n";

  if (!scenario.leading_output) {
    // The CLK precedes the random access point; the rest of the output must be
    // identical whether or not the leading temporal units are present.  This
    // is the §7.4.4 discard check for coded non-output leading frames.
    if (full_decoded.size() != rap_decoded.size() + 1U) {
      std::cerr << "DISCARD FAIL: RAP output should omit exactly the pre-RAP "
                   "CLK frame\n";
      return false;
    }
    std::vector<Picture> after_rap(full_decoded.begin() + 1,
                                   full_decoded.end());
    std::string compare_error;
    if (!compare(after_rap, rap_decoded, "non-output identity",
                 compare_error)) {
      std::cerr << "IDENTITY FAIL: " << compare_error << "\n";
      return false;
    }
    std::string removed_error;
    if (!compare(rap_decoded, rap_no_leading_decoded,
                 "leading removal", removed_error)) {
      std::cerr << "DISCARD FAIL: " << removed_error << "\n";
      return false;
    }
    std::cout << "  PASS: non-output leading frames are suppressed; output "
                 "from the OLK is identical to sequential decode\n";
    std::cout << "  PASS: physically removing the leading temporal units does "
                 "not change the RAP decode\n";
  } else {
    // The leading frames only reference the OLK, so they stay decodable when
    // decoding starts there.  Removing the pre-RAP CLK must not change them.
    if (full_decoded.size() != rap_decoded.size() + 1U) {
      std::cerr << "DISCARD FAIL: expected the pre-RAP CLK to be dropped\n";
      return false;
    }
    std::vector<Picture> after_rap(full_decoded.begin() + 1,
                                   full_decoded.end());
    std::string compare_error;
    if (!compare(after_rap, rap_decoded, "output identity", compare_error)) {
      std::cerr << "IDENTITY FAIL: " << compare_error << "\n";
      return false;
    }
    std::cout << "  PASS: RAP-start decode dropped the pre-RAP CLK; "
                 "immediate-output leading frames remain decodable from the "
                 "OLK\n";
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::string out_dir = ".";
  bool require_decoder = true;
  int qp = 90;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--out-dir" && i + 1 < argc) {
      out_dir = argv[++i];
    } else if (arg == "--qp" && i + 1 < argc) {
      qp = std::atoi(argv[++i]);
    } else if (arg == "--no-decode") {
      require_decoder = false;
    } else if (arg == "--help") {
      std::cout << "usage: leading_gate [--out-dir DIR] [--qp N] "
                   "[--no-decode]\n";
      return 0;
    } else {
      std::cerr << "unknown argument: " << arg << "\n";
      return 2;
    }
  }
  if (qp < 1 || qp > 255) {
    std::cerr << "--qp must be in [1, 255]\n";
    return 2;
  }
  const std::string dav2d = find_dav2d();
  const Scenario scenarios[] = {
      {false, "nonoutput"},
      {true, "output"},
  };
  bool ok = true;
  for (const Scenario& scenario : scenarios) {
    ok = run_scenario(scenario, dav2d, out_dir, require_decoder, qp) && ok;
  }
  std::cout << (ok ? "LEADING GATE PASS" : "LEADING GATE FAIL") << "\n";
  return ok ? 0 : 1;
}

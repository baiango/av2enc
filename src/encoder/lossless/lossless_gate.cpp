// External decoder gate for the lossless still encoder.
//
// This executable is deliberately independent of the repo CLI: it builds a
// lossless still in memory, writes an Annex-B/IVF temporal unit, invokes a
// black-box AV2 decoder, and compares every decoded I420 sample against the
// source.  It is the authority for "the syntax is not just self-consistent".
//
// Usage:
//   lossless_gate --dav2d /path/to/dav2d [--avmdec /path/to/avmdec]
//                 [--tmpdir DIR] [--keep]
//
// Exit codes: 0 pass, 1 failure, 77 skip (no decoder supplied/found).

#include "encoder/lossless/lossless.h"
#include "io/ivf.h"
#include "io/y4m.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

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
  std::uint32_t lcg = 0x9e3779b9u;
  for (int i = 0; i < w * h; ++i) {
    const int x = i % w;
    const int y = i / w;
    switch (pattern) {
      case 0:
        p.y[i] = static_cast<std::uint8_t>(((x / 8) * 37 + (y / 8) * 19) & 255);
        if ((y / 4) % 5 == 0) {
          p.y[i] = static_cast<std::uint8_t>(x < w / 2 ? 0 : 255);
        }
        break;
      case 1:
        p.y[i] = static_cast<std::uint8_t>((x * 5 + y * 3) & 255);
        break;
      default:
        lcg = lcg * 1664525u + 1013904223u;
        p.y[i] = static_cast<std::uint8_t>(lcg >> 24);
        break;
    }
  }
  for (int i = 0; i < w * h / 4; ++i) {
    const int x = i % (w / 2);
    const int y = i / (w / 2);
    if (pattern == 0) {
      p.u[i] = static_cast<std::uint8_t>(((x / 8) * 61) & 255);
      p.v[i] = static_cast<std::uint8_t>(((y / 8) * 53) & 255);
    } else if (pattern == 1) {
      p.u[i] = static_cast<std::uint8_t>((x * 7 + y * 11) & 255);
      p.v[i] = static_cast<std::uint8_t>((x * 13 + y * 2) & 255);
    } else {
      lcg = lcg * 1664525u + 1013904223u;
      p.u[i] = static_cast<std::uint8_t>(lcg >> 24);
      lcg = lcg * 1664525u + 1013904223u;
      p.v[i] = static_cast<std::uint8_t>(lcg >> 24);
    }
  }
  return p;
}

std::vector<std::uint8_t> source_yuv(const Picture& p) {
  std::vector<std::uint8_t> out;
  out.reserve(p.y.size() + p.u.size() + p.v.size());
  out.insert(out.end(), p.y.begin(), p.y.end());
  out.insert(out.end(), p.u.begin(), p.u.end());
  out.insert(out.end(), p.v.begin(), p.v.end());
  return out;
}

bool load_y4m_picture(const std::string& path, Picture& out, std::string& error) {
  av2enc::Y4mReader reader(path);
  if (!reader.open(error)) return false;
  av2enc::Y4mFrame frame;
  const av2enc::Y4mReadResult result = reader.read_frame(frame, error);
  if (result != av2enc::Y4mReadResult::kFrame) {
    if (error.empty()) error = "no frame in " + path;
    return false;
  }
  out.w = static_cast<int>(frame.width());
  out.h = static_cast<int>(frame.height());
  out.y.assign(frame.y().begin(), frame.y().end());
  out.u.assign(frame.u().begin(), frame.u().end());
  out.v.assign(frame.v().begin(), frame.v().end());
  return true;
}

std::string shell_quote(const std::string& s) {
  std::string out = "'";
  for (char c : s) {
    if (c == '\'') out += "'\\''";
    else out += c;
  }
  out += "'";
  return out;
}

bool read_file(const std::filesystem::path& path, std::vector<std::uint8_t>& out,
               std::string& error) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    error = "cannot open " + path.string();
    return false;
  }
  file.seekg(0, std::ios::end);
  const std::streamoff size = file.tellg();
  file.seekg(0, std::ios::beg);
  if (size < 0) {
    error = "cannot size " + path.string();
    return false;
  }
  out.resize(static_cast<std::size_t>(size));
  file.read(reinterpret_cast<char*>(out.data()), size);
  return file.good() || file.eof();
}

bool write_file(const std::filesystem::path& path,
                const std::vector<std::uint8_t>& data, std::string& error) {
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    error = "cannot create " + path.string();
    return false;
  }
  file.write(reinterpret_cast<const char*>(data.data()),
             static_cast<std::streamsize>(data.size()));
  return file.good();
}

struct GateCase {
  int w;
  int h;
  int pattern;
  av2enc::lossless::FscMode fsc;
  bool adaptive;
};

bool run_dav2d(const std::string& decoder, const std::filesystem::path& ivf,
               const std::filesystem::path& decoded, std::string& error) {
  const std::string cmd = shell_quote(decoder) + " -i " + shell_quote(ivf.string()) +
                          " -o " + shell_quote(decoded.string()) +
                          " --muxer yuv --strict 1";
  const int status = std::system(cmd.c_str());
  if (status != 0) {
    error = "dav2d failed (" + std::to_string(status) + "): " + cmd;
    return false;
  }
  return true;
}

bool run_avmdec(const std::string& decoder, const std::filesystem::path& ivf,
                const std::filesystem::path& decoded, std::string& error) {
  const std::string cmd = shell_quote(decoder) + " --i420 -o " +
                          shell_quote(decoded.string()) + " " +
                          shell_quote(ivf.string());
  const int status = std::system(cmd.c_str());
  if (status != 0) {
    error = "avmdec failed (" + std::to_string(status) + "): " + cmd;
    return false;
  }
  return true;
}

bool check_case(const Picture& p, av2enc::lossless::FscMode fsc, bool adaptive,
                const std::string& stem, const std::string& dav2d,
                const std::string& avmdec, const std::filesystem::path& work,
                bool keep, int* checks, std::string* error) {
  av2enc::lossless::StillPicture pic;
  pic.width = static_cast<std::uint32_t>(p.w);
  pic.height = static_cast<std::uint32_t>(p.h);
  pic.y = p.y;
  pic.u = p.u;
  pic.v = p.v;
  av2enc::lossless::Options opt;
  opt.fsc = fsc;
  opt.adaptive_cdf = adaptive;
  av2enc::lossless::EncodeResult result;
  if (!av2enc::lossless::encode_still(pic, opt, result, *error)) {
    *error = "encode: " + *error;
    return false;
  }

  const std::filesystem::path ivf = work / (stem + ".ivf");
  const std::filesystem::path src = work / (stem + ".src.yuv");
  av2enc::IvfWriter writer;
  av2enc::IvfHeader header;
  header.width = static_cast<std::uint16_t>(p.w);
  header.height = static_cast<std::uint16_t>(p.h);
  header.fps_num = 30;
  header.fps_den = 1;
  header.frame_count = 1;
  if (!writer.open(ivf, header, *error) ||
      !writer.write_frame(result.temporal_unit, 0, *error) ||
      !writer.close(*error)) {
    *error = "IVF: " + *error;
    return false;
  }
  const std::vector<std::uint8_t> expected = source_yuv(p);
  if (!write_file(src, expected, *error)) return false;

  std::error_code ec;
  bool ok = true;
  if (!dav2d.empty()) {
    const std::filesystem::path decoded = work / (stem + ".dav2d.yuv");
    if (!run_dav2d(dav2d, ivf, decoded, *error)) {
      ok = false;
    } else {
      std::vector<std::uint8_t> actual;
      if (!read_file(decoded, actual, *error)) {
        ok = false;
      } else if (actual != expected) {
        std::size_t diff = 0;
        for (std::size_t i = 0; i < std::min(actual.size(), expected.size());
             ++i) {
          if (actual[i] != expected[i]) ++diff;
        }
        *error = "dav2d output differs (" + std::to_string(diff) + "/" +
                 std::to_string(expected.size()) + " samples)";
        ok = false;
      } else {
        ++*checks;
        std::printf("ok   %s (dav2d, %u tile bytes)\n", stem.c_str(),
                    result.stats.tile_bytes);
      }
    }
    if (!keep) std::filesystem::remove(decoded, ec);
  }

  if (!avmdec.empty()) {
    const std::filesystem::path decoded = work / (stem + ".avmdec.yuv");
    if (!run_avmdec(avmdec, ivf, decoded, *error)) {
      ok = false;
    } else {
      std::vector<std::uint8_t> actual;
      if (!read_file(decoded, actual, *error)) {
        ok = false;
      } else if (actual != expected) {
        std::size_t diff = 0;
        for (std::size_t i = 0; i < std::min(actual.size(), expected.size());
             ++i) {
          if (actual[i] != expected[i]) ++diff;
        }
        *error = "avmdec output differs (" + std::to_string(diff) + "/" +
                 std::to_string(expected.size()) + " samples)";
        ok = false;
      } else {
        ++*checks;
        std::printf("ok   %s (avmdec)\n", stem.c_str());
      }
    }
    if (!keep) std::filesystem::remove(decoded, ec);
  }

  if (!keep) {
    std::filesystem::remove(ivf, ec);
    std::filesystem::remove(src, ec);
  }
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  std::string dav2d;
  std::string avmdec;
  std::string y4m_path;
  std::filesystem::path tmpdir = std::filesystem::temp_directory_path();
  bool keep = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--dav2d" && i + 1 < argc) {
      dav2d = argv[++i];
    } else if (arg == "--avmdec" && i + 1 < argc) {
      avmdec = argv[++i];
    } else if (arg == "--y4m" && i + 1 < argc) {
      y4m_path = argv[++i];
    } else if (arg == "--tmpdir" && i + 1 < argc) {
      tmpdir = argv[++i];
    } else if (arg == "--keep") {
      keep = true;
    } else {
      std::fprintf(stderr,
                   "usage: %s --dav2d PATH [--avmdec PATH] [--y4m FILE] "
                   "[--tmpdir DIR] [--keep]\n",
                   argv[0]);
      return 1;
    }
  }

  // If no decoder was passed, try the repository's pinned binaries relative
  // to the current directory.
  auto find = [](const std::filesystem::path& p) {
    return std::filesystem::exists(p) ? p.string() : std::string();
  };
  if (dav2d.empty()) {
    dav2d = find("tools/bin/dav2d-macos");
    if (dav2d.empty()) dav2d = find("tools/bin/dav2d");
  }
  if (avmdec.empty()) {
    avmdec = find("tools/bin/avm-macos/avmdec");
  }
  if (dav2d.empty() && avmdec.empty()) {
    std::fprintf(stderr,
                 "SKIP: no external decoder (pass --dav2d or --avmdec)\n");
    return 77;
  }

  const std::filesystem::path work = tmpdir / "av2enc-lossless-gate";
  std::error_code ec;
  std::filesystem::create_directories(work, ec);

  const GateCase cases[] = {
      {64, 64, 0, av2enc::lossless::FscMode::kOff, false},
      {64, 64, 0, av2enc::lossless::FscMode::kForce, false},
      {136, 88, 1, av2enc::lossless::FscMode::kOff, false},
      {136, 88, 1, av2enc::lossless::FscMode::kForce, false},
      {136, 88, 2, av2enc::lossless::FscMode::kForce, false},
      {64, 64, 3, av2enc::lossless::FscMode::kAuto, false},
      {640, 360, 1, av2enc::lossless::FscMode::kOff, false},
      {640, 360, 2, av2enc::lossless::FscMode::kForce, false},
      {64, 64, 2, av2enc::lossless::FscMode::kOff, true},
  };

  int failures = 0;
  int checks = 0;
  for (const GateCase& c : cases) {
    const Picture p = make_picture(c.w, c.h, c.pattern);
    const std::string stem = std::to_string(c.w) + "x" + std::to_string(c.h) +
                             "_p" + std::to_string(c.pattern) + "_f" +
                             std::to_string(static_cast<int>(c.fsc)) +
                             (c.adaptive ? "_ad" : "");
    std::string error;
    if (!check_case(p, c.fsc, c.adaptive, stem, dav2d, avmdec, work, keep,
                    &checks, &error)) {
      std::fprintf(stderr, "FAIL %s: %s\n", stem.c_str(), error.c_str());
      ++failures;
    }
  }

  // Optional real-content still.  The repo's 64x64 M0 frame is a convenient
  // black-box gate because it is not synthetic screen content.
  if (y4m_path.empty()) {
    const std::filesystem::path candidate = "tests/data/m0_64x64_1f.y4m";
    if (std::filesystem::exists(candidate)) y4m_path = candidate.string();
  }
  if (!y4m_path.empty()) {
    Picture p;
    std::string error;
    if (!load_y4m_picture(y4m_path, p, error)) {
      std::fprintf(stderr, "FAIL y4m %s: %s\n", y4m_path.c_str(),
                   error.c_str());
      ++failures;
    } else {
      for (int force = 0; force < 2; ++force) {
        const auto fsc = force ? av2enc::lossless::FscMode::kForce
                               : av2enc::lossless::FscMode::kOff;
        const std::string stem = "y4m_" + std::to_string(p.w) + "x" +
                                 std::to_string(p.h) + "_f" +
                                 std::to_string(static_cast<int>(fsc));
        if (!check_case(p, fsc, false, stem, dav2d, avmdec, work, keep, &checks,
                        &error)) {
          std::fprintf(stderr, "FAIL %s: %s\n", stem.c_str(), error.c_str());
          ++failures;
        }
      }
    }
  }

  if (!keep) std::filesystem::remove_all(work, ec);
  if (failures == 0) {
    std::printf("lossless_gate: %d decoder checks passed\n", checks);
    return 0;
  }
  std::fprintf(stderr, "lossless_gate: %d failure(s)\n", failures);
  return 1;
}

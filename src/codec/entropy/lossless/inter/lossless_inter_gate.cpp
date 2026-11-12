// External decoder gate for the AV2 lossless inter module.
//
// Each case encodes a 2-frame sequence (key + P), writes both temporal units to
// an IVF, runs the pinned `dav2d --strict 1 --threads 1` decoder, and requires
// the decoded I420 samples of both frames to equal the source exactly.  The
// module's independent reader is also required to reproduce both frames.
//
// Usage: lossless_inter_gate [--dav2d PATH] [--tmpdir DIR] [--keep]
// Exit codes: 0 pass, 1 failure, 77 skip (no decoder).

#include "codec/entropy/lossless/inter/lossless_inter.h"
#include "core/io/ivf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

using av2enc::lossless::inter::ChromaFormat;
using av2enc::lossless::inter::Picture;

std::string shell_quote(const std::string& s) {
  std::string out = "'";
  for (char c : s) {
    if (c == '\'') out += "'\\''";
    else out += c;
  }
  out += "'";
  return out;
}

Picture make_picture(int w, int h, ChromaFormat format, int frame) {
  Picture p;
  p.width = w;
  p.height = h;
  p.format = format;
  const int sub_x = format == ChromaFormat::k444 ? 0 : 1;
  const int sub_y = format == ChromaFormat::k420 ? 1 : 0;
  const int cw = w >> sub_x;
  const int ch = h >> sub_y;
  p.y.resize(static_cast<std::size_t>(w) * h);
  p.u.resize(static_cast<std::size_t>(cw) * ch);
  p.v.resize(static_cast<std::size_t>(cw) * ch);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int v = 0;
      if (frame == 0) {
        v = ((x / 8) * 37 + (y / 8) * 19 + x * 3 + y * 5) & 255;
        if ((y / 4) % 7 == 0) v ^= (x < w / 2) ? 0x33 : 0xcc;
      } else {
        v = ((x / 4) * 17 + (y / 4) * 29 + x * 7 + y * 11 + 13) & 255;
        if ((x % 37) == 0 || (y % 41) == 0) v ^= 0x55;
      }
      p.y[static_cast<std::size_t>(y) * w + x] = static_cast<std::uint8_t>(v);
    }
  }
  for (int y = 0; y < ch; ++y) {
    for (int x = 0; x < cw; ++x) {
      p.u[static_cast<std::size_t>(y) * cw + x] = static_cast<std::uint8_t>(
          (x * 5 + y * 3 + frame * 19 + ((x * y) % 13)) & 255);
      p.v[static_cast<std::size_t>(y) * cw + x] = static_cast<std::uint8_t>(
          (x * 11 + y * 7 + frame * 23 + ((x + y) % 17)) & 255);
    }
  }
  return p;
}

void append_plane(std::vector<std::uint8_t>& out,
                  const std::vector<std::uint8_t>& plane) {
  out.insert(out.end(), plane.begin(), plane.end());
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

bool run_dav2d(const std::string& decoder, const std::filesystem::path& ivf,
               const std::filesystem::path& decoded, std::string& error) {
  const std::string cmd = shell_quote(decoder) + " --strict 1 --threads 1 -i " +
                          shell_quote(ivf.string()) + " -o " +
                          shell_quote(decoded.string()) + " --muxer yuv";
  const int status = std::system(cmd.c_str());
  if (status != 0) {
    error = "dav2d failed (" + std::to_string(status) + "): " + cmd;
    return false;
  }
  return true;
}

struct GateCase {
  int w;
  int h;
  ChromaFormat format;
  int block_size;
  int intra_period;
  bool fsc;
  const char* name;
};

bool check_case(const GateCase& c, const std::string& dav2d,
                const std::filesystem::path& work, bool keep, int* checks,
                std::string* error) {
  std::vector<Picture> frames{
      make_picture(c.w, c.h, c.format, 0),
      make_picture(c.w, c.h, c.format, 1)};
  av2enc::lossless::inter::Options opt;
  opt.block_size = c.block_size;
  opt.intra_period = c.intra_period;
  opt.enable_fsc = c.fsc;
  av2enc::lossless::inter::EncodeResult enc;
  if (!av2enc::lossless::inter::encode_video(frames, opt, &enc, error)) {
    *error = "encode: " + *error;
    return false;
  }
  if (!enc.exact || enc.units.size() != 2) {
    *error = "encoder did not produce two exact frames";
    return false;
  }

  // The module's own reader must reproduce both frames.
  av2enc::lossless::inter::DecodeResult dec;
  if (!av2enc::lossless::inter::decode_video(enc.units, &dec, error)) {
    *error = "reader: " + *error;
    return false;
  }
  if (dec.frames.size() != 2) {
    *error = "reader produced the wrong frame count";
    return false;
  }
  for (int i = 0; i < 2; ++i) {
    if (dec.frames[i].y != frames[i].y || dec.frames[i].u != frames[i].u ||
        dec.frames[i].v != frames[i].v) {
      *error = "reader reconstruction differs at frame " + std::to_string(i);
      return false;
    }
  }
  ++*checks;

  const std::filesystem::path ivf = work / (std::string(c.name) + ".ivf");
  av2enc::IvfWriter writer;
  av2enc::IvfHeader header;
  header.width = static_cast<std::uint16_t>(c.w);
  header.height = static_cast<std::uint16_t>(c.h);
  header.fps_num = 30;
  header.fps_den = 1;
  header.frame_count = 2;
  if (!writer.open(ivf, header, *error)) return false;
  for (std::size_t i = 0; i < enc.units.size(); ++i) {
    if (!writer.write_frame(enc.units[i], static_cast<std::uint32_t>(i),
                            *error)) {
      return false;
    }
  }
  if (!writer.close(*error)) return false;

  std::vector<std::uint8_t> expected;
  for (const Picture& p : frames) {
    append_plane(expected, p.y);
    append_plane(expected, p.u);
    append_plane(expected, p.v);
  }

  const std::filesystem::path decoded = work / (std::string(c.name) + ".yuv");
  if (!run_dav2d(dav2d, ivf, decoded, *error)) return false;
  std::vector<std::uint8_t> actual;
  if (!read_file(decoded, actual, *error)) return false;
  if (actual.size() != expected.size()) {
    *error = "dav2d output size " + std::to_string(actual.size()) +
             " != expected " + std::to_string(expected.size());
    return false;
  }
  if (actual != expected) {
    std::size_t diff = 0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
      if (actual[i] != expected[i]) ++diff;
    }
    *error = "dav2d output differs (" + std::to_string(diff) + "/" +
             std::to_string(expected.size()) + " samples)";
    return false;
  }
  ++*checks;
  std::printf("ok   %-24s (dav2d 2 frames, %zu tile bytes)\n", c.name,
              enc.units[1].size());
  if (!keep) {
    std::error_code ec;
    std::filesystem::remove(ivf, ec);
    std::filesystem::remove(decoded, ec);
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::string dav2d;
  std::filesystem::path tmpdir = std::filesystem::temp_directory_path();
  bool keep = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--dav2d" && i + 1 < argc) {
      dav2d = argv[++i];
    } else if (arg == "--tmpdir" && i + 1 < argc) {
      tmpdir = argv[++i];
    } else if (arg == "--keep") {
      keep = true;
    } else {
      std::fprintf(stderr,
                   "usage: %s [--dav2d PATH] [--tmpdir DIR] [--keep]\n",
                   argv[0]);
      return 1;
    }
  }
  if (dav2d.empty()) {
    for (const char* candidate : {"tools/bin/dav2d-macos", "tools/bin/dav2d"}) {
      if (std::filesystem::exists(candidate)) {
        dav2d = candidate;
        break;
      }
    }
  }
  if (dav2d.empty()) {
    std::fprintf(stderr, "SKIP: no dav2d decoder (pass --dav2d)\n");
    return 77;
  }

  const std::filesystem::path work = tmpdir / "av2enc-lossless-inter-gate";
  std::error_code ec;
  std::filesystem::create_directories(work, ec);

  const GateCase cases[] = {
      {64, 64, ChromaFormat::k420, 8, 0, true, "64x64_420_b8_p0"},
      {64, 64, ChromaFormat::k420, 16, 4, true, "64x64_420_b16_p4"},
      {64, 64, ChromaFormat::k420, 32, 4, true, "64x64_420_b32_p4"},
      {136, 88, ChromaFormat::k420, 8, 0, true, "136x88_420_b8_p0"},
      {136, 88, ChromaFormat::k420, 16, 4, true, "136x88_420_b16_p4"},
      {640, 360, ChromaFormat::k420, 16, 4, true, "640x360_420_b16_p4"},
      {64, 64, ChromaFormat::k444, 16, 4, true, "64x64_444_b16_p4"},
      {64, 64, ChromaFormat::k422, 16, 4, true, "64x64_422_b16_p4"},
  };

  int checks = 0;
  int failures = 0;
  for (const GateCase& c : cases) {
    std::string error;
    if (!check_case(c, dav2d, work, keep, &checks, &error)) {
      std::fprintf(stderr, "FAIL %s: %s\n", c.name, error.c_str());
      ++failures;
    }
  }
  if (!keep) std::filesystem::remove_all(work, ec);
  if (failures == 0) {
    std::printf("lossless_inter_gate: %d checks passed\n", checks);
    return 0;
  }
  std::fprintf(stderr, "lossless_inter_gate: %d failure(s)\n", failures);
  return 1;
}

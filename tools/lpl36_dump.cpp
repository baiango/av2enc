// Test harness for the lpl36 port: score two Y4M files frame by frame on their
// luma planes.  It links only the core Y4M reader and the header-only metric,
// so the parity gate can compare C++ against the Python reference on identical
// Y planes without going through the encoder.
//
// Usage:
//   lpl36_dump --ref ref.y4m --test test.y4m [--frames N]
//
// Prints one "frame=<i> lpl_error=<x> lpl_score=<y>" line per frame and a
// final "frames=<n> width=<w> height=<h> mean_lpl_error=<x>" summary.
// Exit: 0 pass, 1 malformed/mismatched input, 2 usage.
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include "core/io/y4m.h"
#include "pipeline/video/video_lpl36.h"

namespace {

void usage() {
  std::cerr << "usage: lpl36_dump --ref ref.y4m --test test.y4m "
               "[--frames N]\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string ref_path;
  std::string test_path;
  int max_frames = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--ref" && i + 1 < argc) {
      ref_path = argv[++i];
    } else if (arg == "--test" && i + 1 < argc) {
      test_path = argv[++i];
    } else if (arg == "--frames" && i + 1 < argc) {
      char* end = nullptr;
      const long value = std::strtol(argv[++i], &end, 10);
      if (end == argv[i] || *end != '\0' || value < 0) {
        std::cerr << "lpl36_dump: invalid --frames value\n";
        return 2;
      }
      max_frames = static_cast<int>(value);
    } else if (arg == "-h" || arg == "--help") {
      usage();
      return 0;
    } else {
      std::cerr << "lpl36_dump: unknown argument " << arg << "\n";
      usage();
      return 2;
    }
  }
  if (ref_path.empty() || test_path.empty()) {
    usage();
    return 2;
  }

  av2enc::Y4mReader ref_reader(ref_path);
  av2enc::Y4mReader test_reader(test_path);
  std::string error;
  if (!ref_reader.open(error)) {
    std::cerr << "lpl36_dump: " << ref_path << ": " << error << "\n";
    return 1;
  }
  if (!test_reader.open(error)) {
    std::cerr << "lpl36_dump: " << test_path << ": " << error << "\n";
    return 1;
  }
  const av2enc::Y4mHeader& ref_header = ref_reader.header();
  const av2enc::Y4mHeader& test_header = test_reader.header();
  if (ref_header.width != test_header.width || ref_header.height != test_header.height) {
    std::cerr << "lpl36_dump: dimension mismatch: " << ref_header.width << "x" << ref_header.height
              << " vs " << test_header.width << "x" << test_header.height << "\n";
    return 1;
  }
  const int width = static_cast<int>(ref_header.width);
  const int height = static_cast<int>(ref_header.height);
  av2enc::video::lpl36::Metric metric;
  metric.Init(width, height);

  av2enc::Y4mFrame ref_frame;
  av2enc::Y4mFrame test_frame;
  int frames = 0;
  double sum = 0.0;
  while (max_frames == 0 || frames < max_frames) {
    const av2enc::Y4mReadResult ref_result = ref_reader.read_frame(ref_frame, error);
    if (ref_result == av2enc::Y4mReadResult::kEof) break;
    if (ref_result == av2enc::Y4mReadResult::kError) {
      std::cerr << "lpl36_dump: " << ref_path << ": " << error << "\n";
      return 1;
    }
    const av2enc::Y4mReadResult test_result = test_reader.read_frame(test_frame, error);
    if (test_result == av2enc::Y4mReadResult::kEof) {
      std::cerr << "lpl36_dump: " << test_path << " ended before the reference stream\n";
      return 1;
    }
    if (test_result == av2enc::Y4mReadResult::kError) {
      std::cerr << "lpl36_dump: " << test_path << ": " << error << "\n";
      return 1;
    }
    if (ref_frame.width() != ref_header.width || ref_frame.height() != ref_header.height ||
        test_frame.width() != ref_header.width || test_frame.height() != ref_header.height) {
      std::cerr << "lpl36_dump: frame " << frames << " changed dimensions mid-stream\n";
      return 1;
    }
    const double value = metric.Error(ref_frame.y().data(), width, test_frame.y().data(), width);
    std::printf("frame=%d lpl_error=%.9f lpl_score=%.9f\n", frames, value, -value);
    sum += value;
    ++frames;
  }
  if (frames == 0) {
    std::cerr << "lpl36_dump: no frames scored\n";
    return 1;
  }
  const double mean = sum / static_cast<double>(frames);
  std::printf(
      "frames=%d width=%d height=%d mean_lpl_error=%.9f "
      "mean_lpl_score=%.9f\n",
      frames, width, height, mean, -mean);
  return 0;
}

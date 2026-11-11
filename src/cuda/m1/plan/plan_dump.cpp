// Standalone dump for the M1 host FramePlan.
//
//   plan_dump -i frame.y4m --tsv > plan.tsv
//   plan_dump -i frame.y4m --json -o plan.json
//   plan_dump --synthetic 640x360 --tsv
//
// The block order matches the CPU AV2ENC_M1_TRACE=1 trace; use
// compare_trace.py to check the (mi_row, mi_col, size) sequence.

#include "cuda/m1/plan/plan.h"
#include "io/y4m.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace m1p = av2enc::m1::plan;

namespace {

void usage(std::ostream& out) {
  out << "Usage: plan_dump (-i <input.y4m> | --synthetic WxH) [options]\n"
         "\n"
         "Options:\n"
         "  -i, --input <file>      first frame of a Y4M (8-bit 4:2:0)\n"
         "  --synthetic WxH         deterministic synthetic luma plane\n"
         "  -o, --output <file>     write the dump here instead of stdout\n"
         "  --tsv                   tab-separated dump (default)\n"
         "  --json                  one JSON object (blocks and tree)\n"
         "  --tree                  preorder partition-tree TSV instead of blocks\n"
         "  --print-options         print the resolved PlanOptions and exit\n"
         "  --depth N               max_partition_depth (default 4)\n"
         "  --min-block-size N      partition floor in pixels (default 8)\n"
         "  --qp N                  base_q_idx used by the signal cost (default 90)\n"
         "  --single                force one PARTITION_NONE superblock\n"
         "  --square-only           apply the AV2ENC_M1_SQUARE_ONLY gate\n"
         "  --leaf32                apply the AV2ENC_M1_LEAF32 gate\n"
         "  --no-env                ignore the AV2ENC_M1_* debug environment\n"
         "  -h, --help\n";
}

bool parse_int(std::string_view text, int& value) {
  if (text.empty()) return false;
  try {
    std::size_t used = 0;
    const int parsed = std::stoi(std::string(text), &used);
    if (used != text.size()) return false;
    value = parsed;
    return true;
  } catch (...) {
    return false;
  }
}

bool parse_size(std::string_view text, int& width, int& height) {
  const std::size_t x = text.find('x');
  if (x == std::string_view::npos) return false;
  return parse_int(text.substr(0, x), width) &&
         parse_int(text.substr(x + 1), height);
}

std::vector<std::uint8_t> synthetic_luma(int width, int height) {
  std::vector<std::uint8_t> luma(
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 0);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int block = ((x / 8) ^ (y / 8)) * 17;
      const int value = (x * 3 + y * 5 + block + ((x * y) >> 3)) & 255;
      luma[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
           static_cast<std::size_t>(x)] = static_cast<std::uint8_t>(value);
    }
  }
  return luma;
}

bool read_first_luma(const std::filesystem::path& path, int& width, int& height,
                     std::vector<std::uint8_t>& luma, std::string& error) {
  av2enc::Y4mReader reader(path);
  if (!reader.open(error)) return false;
  av2enc::Y4mFrame frame;
  const av2enc::Y4mReadResult result = reader.read_frame(frame, error);
  if (result != av2enc::Y4mReadResult::kFrame) {
    if (error.empty()) error = "input has no frames";
    return false;
  }
  width = static_cast<int>(frame.width());
  height = static_cast<int>(frame.height());
  const std::span<const std::uint8_t> y = frame.y();
  luma.assign(y.begin(), y.end());
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path input;
  std::filesystem::path output;
  bool have_synthetic = false;
  int synthetic_w = 0;
  int synthetic_h = 0;
  bool json = false;
  bool tree = false;
  bool print_options = false;
  bool use_env = true;
  m1p::PlanOptions options;

  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    const auto next = [&](const char* what) -> const char* {
      if (i + 1 >= argc) {
        std::cerr << "plan_dump: missing value after " << what << '\n';
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "-i" || arg == "--input") {
      input = next("--input");
    } else if (arg == "--synthetic") {
      if (!parse_size(next("--synthetic"), synthetic_w, synthetic_h)) {
        std::cerr << "plan_dump: --synthetic expects WxH\n";
        return 2;
      }
      have_synthetic = true;
    } else if (arg == "-o" || arg == "--output") {
      output = next("--output");
    } else if (arg == "--tsv") {
      json = false;
    } else if (arg == "--json") {
      json = true;
    } else if (arg == "--tree") {
      tree = true;
    } else if (arg == "--print-options") {
      print_options = true;
    } else if (arg == "--depth") {
      if (!parse_int(next("--depth"), options.max_partition_depth)) return 2;
    } else if (arg == "--min-block-size") {
      if (!parse_int(next("--min-block-size"), options.min_block_size)) {
        return 2;
      }
    } else if (arg == "--qp") {
      int qp = 90;
      if (!parse_int(next("--qp"), qp)) return 2;
      options.base_q_idx = static_cast<std::uint8_t>(qp);
    } else if (arg == "--single") {
      options.force_single_block = true;
    } else if (arg == "--square-only") {
      options.square_only = true;
    } else if (arg == "--leaf32") {
      options.leaf32 = true;
    } else if (arg == "--no-env") {
      use_env = false;
    } else if (arg == "-h" || arg == "--help") {
      usage(std::cout);
      return 0;
    } else {
      std::cerr << "plan_dump: unknown argument " << arg << '\n';
      usage(std::cerr);
      return 2;
    }
  }

  if (use_env) m1p::apply_debug_env(options);

  if (print_options) {
    std::cout << "{\"base_q_idx\":" << static_cast<int>(options.base_q_idx)
              << ",\"min_block_size\":" << options.min_block_size
              << ",\"max_partition_depth\":" << options.max_partition_depth
              << ",\"force_single_block\":"
              << (options.force_single_block ? "true" : "false")
              << ",\"square_only\":"
              << (options.square_only ? "true" : "false")
              << ",\"leaf32\":" << (options.leaf32 ? "true" : "false")
              << "}\n";
    return 0;
  }

  if (input.empty() && !have_synthetic) {
    std::cerr << "plan_dump: pass exactly one of -i <y4m> or --synthetic WxH\n";
    return 2;
  }
  if (!input.empty() && have_synthetic) {
    std::cerr << "plan_dump: pass exactly one of -i <y4m> or --synthetic WxH\n";
    return 2;
  }

  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> luma;
  std::string error;
  if (have_synthetic) {
    width = synthetic_w;
    height = synthetic_h;
    if (width <= 0 || height <= 0) {
      std::cerr << "plan_dump: synthetic dimensions must be positive\n";
      return 2;
    }
    luma = synthetic_luma(width, height);
  } else if (!read_first_luma(input, width, height, luma, error)) {
    std::cerr << "plan_dump: " << error << '\n';
    return 1;
  }

  m1p::FramePlan plan;
  if (!m1p::build_frame_plan(width, height, luma, options, plan, error)) {
    std::cerr << "plan_dump: " << error << '\n';
    return 1;
  }
  const std::string dump =
      json ? m1p::plan_to_json(plan)
           : (tree ? m1p::plan_tree_to_tsv(plan) : m1p::plan_to_tsv(plan));

  if (output.empty()) {
    std::cout << dump;
  } else {
    std::ofstream file(output, std::ios::binary);
    if (!file) {
      std::cerr << "plan_dump: cannot open " << output << '\n';
      return 1;
    }
    file << dump;
    if (!file) {
      std::cerr << "plan_dump: write failed for " << output << '\n';
      return 1;
    }
  }
  return 0;
}

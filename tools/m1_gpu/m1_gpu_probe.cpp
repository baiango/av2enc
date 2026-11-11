// m1_gpu_probe: encode the first Y4M frame through the GPU M1 orchestrator
// (`av2enc::cuda::m1gpu::encode_m1_gpu`) and write:
//
//   * an `M1DUMP1` reconstruction dump (same format as tools/m1_verify's
//     m1_dump), for bit-exact recon parity against the CPU M1Result;
//   * a stats JSON (tools/m1_gpu/README.md contract) describing which stages
//     ran on the GPU, launches, arena high-water mark and device info.
//
// Why a probe: the CLI writes only the bitstream, and `GpuResult::recon_y/u/v`
// is not exposed there. This links the same core library as the CLI, so the
// recon planes and stage states are the orchestrator's own.
//
// IMPORTANT: pass the same options as the CLI encode being verified. A depth /
// deblock / CfL mismatch reconstructs a different partition tree and looks like
// a transform bug (docs/negative-results.md).
//
// Build on a Linux + CUDA host (not part of the encoder build):
//   tools/m1_gpu/build_probe.sh
#include "cuda/m1/m1_gpu.h"
#include "io/y4m.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

void usage(std::ostream& out) {
  out << "Usage: m1_gpu_probe -i <input.y4m> -o <recon.dump> [options]\n"
         "\n"
         "Runs av2enc::cuda::m1gpu::encode_m1_gpu on the first Y4M frame and\n"
         "writes an M1DUMP1 recon dump plus an optional stats JSON.\n"
         "Defaults mirror GpuOptions{} and M1Options{}.\n"
         "\n"
         "Options (must match the CLI encode):\n"
         "  -i, --input <file>      input Y4M (required)\n"
         "  -o, --output <file>     recon dump path (required)\n"
         "  --stats <file>          write the stats JSON here\n"
         "  --qp N                  base_q_idx (default 90)\n"
         "  --depth N               max_partition_depth 0..8 (default 4)\n"
         "  --min-block-size N      partition floor (default 8)\n"
         "  --no-deblock            --no-cfl            --no-edge-filter\n"
         "  --full-intra            --m1-single         --m1-dc\n"
         "  --adaptive-cdf          --vram-cap MiB\n"
         "  --allow-cpu-fallback    allow the staged CPU reference (default: fail)\n"
         "  --no-verify-cpu         skip the orchestrator's internal CPU parity\n"
         "  --verbose               one stderr line per stage\n"
         "  --print-json            print the stats JSON to stdout\n"
         "  -h, --help\n";
}

std::string json_escape(std::string_view in) {
  std::string out;
  out.reserve(in.size() + 8);
  for (const char c : in) {
    switch (c) {
      case '\\': out += "\\\\"; break;
      case '"': out += "\\\""; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[7];
          std::snprintf(buf, sizeof(buf), "\\u%04x",
                        static_cast<unsigned>(static_cast<unsigned char>(c)));
          out += buf;
        } else {
          out += c;
        }
    }
  }
  return out;
}

bool parse_int(std::string_view text, int& value) {
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

struct Cli {
  std::filesystem::path input;
  std::filesystem::path output;
  std::filesystem::path stats;
  av2enc::cuda::m1gpu::GpuOptions options;
  bool print_json = false;
  bool help = false;
  Cli() { options.allow_cpu_fallback = false; }
};

int parse_args(int argc, char** argv, Cli& cli) {
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    const auto next = [&](const char* what, std::string& out) {
      if (i + 1 >= argc) {
        std::cerr << "m1_gpu_probe: missing value after " << what << "\n";
        return false;
      }
      out = argv[++i];
      return true;
    };
    std::string value;
    if (arg == "-i" || arg == "--input") {
      if (!next("--input", value)) return 2;
      cli.input = value;
    } else if (arg == "-o" || arg == "--output") {
      if (!next("--output", value)) return 2;
      cli.output = value;
    } else if (arg == "--stats") {
      if (!next("--stats", value)) return 2;
      cli.stats = value;
    } else if (arg == "--qp") {
      if (!next("--qp", value)) return 2;
      int qp = 0;
      if (!parse_int(value, qp) || qp < 0 || qp > 255) {
        std::cerr << "m1_gpu_probe: --qp must be in [0, 255]\n";
        return 2;
      }
      cli.options.base_q_idx = static_cast<std::uint8_t>(qp);
    } else if (arg == "--depth" || arg == "--m1-depth") {
      if (!next("--depth", value)) return 2;
      int depth = 0;
      if (!parse_int(value, depth) || depth < 0 || depth > 8) {
        std::cerr << "m1_gpu_probe: --depth must be in [0, 8]\n";
        return 2;
      }
      cli.options.max_partition_depth = depth;
    } else if (arg == "--min-block-size") {
      if (!next("--min-block-size", value)) return 2;
      int size = 0;
      if (!parse_int(value, size) || size < 4 || size > 64) {
        std::cerr << "m1_gpu_probe: --min-block-size must be in [4, 64]\n";
        return 2;
      }
      cli.options.min_block_size = size;
    } else if (arg == "--vram-cap") {
      if (!next("--vram-cap", value)) return 2;
      int mib = 0;
      if (!parse_int(value, mib) || mib < 0) {
        std::cerr << "m1_gpu_probe: --vram-cap must be >= 0\n";
        return 2;
      }
      cli.options.vram_cap_mb = static_cast<std::uint32_t>(mib);
    } else if (arg == "--no-deblock") {
      cli.options.enable_deblock = false;
    } else if (arg == "--no-cfl") {
      cli.options.enable_cfl = false;
    } else if (arg == "--no-edge-filter") {
      cli.options.enable_edge_filter = false;
    } else if (arg == "--full-intra") {
      cli.options.luma_dc_only = false;
    } else if (arg == "--adaptive-cdf") {
      cli.options.adaptive_cdf = true;
    } else if (arg == "--m1-single") {
      cli.options.force_single_block = true;
    } else if (arg == "--m1-dc") {
      cli.options.force_dc = true;
    } else if (arg == "--allow-cpu-fallback") {
      cli.options.allow_cpu_fallback = true;
    } else if (arg == "--no-verify-cpu") {
      cli.options.verify_against_cpu = false;
    } else if (arg == "--verbose") {
      cli.options.verbose = true;
    } else if (arg == "--print-json") {
      cli.print_json = true;
    } else if (arg == "-h" || arg == "--help") {
      cli.help = true;
      return 0;
    } else {
      std::cerr << "m1_gpu_probe: unknown argument: " << arg << "\n";
      return 2;
    }
  }
  if (cli.help) return 0;
  if (cli.input.empty()) {
    std::cerr << "m1_gpu_probe: missing -i <input.y4m>\n";
    return 2;
  }
  if (cli.output.empty()) {
    std::cerr << "m1_gpu_probe: missing -o <recon.dump>\n";
    return 2;
  }
  return -1;
}

const char* state_name(av2enc::cuda::m1gpu::StageState state) {
  return av2enc::cuda::m1gpu::stage_state_name(state);
}

bool any_gpu_stage(const av2enc::cuda::m1gpu::GpuStageStatus& stages) {
  using av2enc::cuda::m1gpu::StageState;
  return stages.plan == StageState::kGpu || stages.search == StageState::kGpu ||
         stages.tx == StageState::kGpu || stages.recon == StageState::kGpu ||
         stages.chroma == StageState::kGpu || stages.cfl == StageState::kGpu ||
         stages.deblock == StageState::kGpu || stages.sched == StageState::kGpu;
}

std::string stats_json(const Cli& cli, std::uint32_t width,
                       std::uint32_t height,
                       const av2enc::cuda::m1gpu::GpuResult& result,
                       double wall_ms) {
  using av2enc::cuda::m1gpu::StageState;
  const bool gpu_bytes = any_gpu_stage(result.stages) &&
                         !result.frame_bytes_from_cpu;
  const std::string backend = gpu_bytes ? "gpu" : "cpu-reference";
  const auto& s = result.stats;
  auto flag = [](bool value) { return value ? "true" : "false"; };
  std::string j = "{";
  j += "\"backend\":\"" + backend + "\"";
  j += ",\"frame_bytes_from_cpu\":";
  j += flag(result.frame_bytes_from_cpu);
  j += ",\"width\":" + std::to_string(width);
  j += ",\"height\":" + std::to_string(height);
  j += ",\"launches\":" + std::to_string(s.kernel_launches);
  j += ",\"sync_count\":" + std::to_string(s.sync_count);
  j += ",\"cuda_events\":false";
  j += ",\"encode_ns\":";
  j += std::to_string(static_cast<long long>(wall_ms * 1e6));
  j += ",\"orchestrator_ms\":" + std::to_string(s.total_ms);
  j += ",\"arena_capacity_bytes\":";
  j += std::to_string(s.arena_capacity_bytes);
  j += ",\"arena_peak_bytes\":" + std::to_string(s.arena_used_bytes);
  j += ",\"device_total_bytes\":" + std::to_string(s.device_total_bytes);
  j += ",\"device_free_bytes\":" + std::to_string(s.device_free_bytes);
  j += ",\"block_count\":" + std::to_string(s.block_count);
  j += ",\"leaf_count\":" + std::to_string(s.leaf_count);
  j += ",\"host_upload_bytes\":" + std::to_string(s.host_upload_bytes);
  j += ",\"host_download_bytes\":" + std::to_string(s.host_download_bytes);
  j += ",\"gpu_runtime_ready\":";
  j += flag(s.gpu_runtime_ready);
  j += ",\"device_name\":\"" + json_escape(s.device_name) + "\"";
  j += ",\"stages\":{";
  j += "\"plan\":\"" + std::string(state_name(result.stages.plan)) + "\"";
  j += ",\"search\":\"" + std::string(state_name(result.stages.search)) + "\"";
  j += ",\"tx\":\"" + std::string(state_name(result.stages.tx)) + "\"";
  j += ",\"recon\":\"" + std::string(state_name(result.stages.recon)) + "\"";
  j += ",\"chroma\":\"" + std::string(state_name(result.stages.chroma)) + "\"";
  j += ",\"cfl\":\"" + std::string(state_name(result.stages.cfl)) + "\"";
  j += ",\"deblock\":\"" + std::string(state_name(result.stages.deblock)) + "\"";
  j += ",\"sched\":\"" + std::string(state_name(result.stages.sched)) + "\"";
  j += "}";
  j += ",\"options\":{\"qp\":" + std::to_string(cli.options.base_q_idx) +
       ",\"depth\":" + std::to_string(cli.options.max_partition_depth) +
       ",\"deblock\":" + flag(cli.options.enable_deblock) +
       ",\"cfl\":" + flag(cli.options.enable_cfl) +
       ",\"edge_filter\":" + flag(cli.options.enable_edge_filter) +
       ",\"full_intra\":" + flag(!cli.options.luma_dc_only) +
       ",\"adaptive_cdf\":" + flag(cli.options.adaptive_cdf) +
       ",\"single\":" + flag(cli.options.force_single_block) +
       ",\"dc\":" + flag(cli.options.force_dc) +
       ",\"vram_cap_mb\":" + std::to_string(cli.options.vram_cap_mb) + "}";
  j += "}";
  return j;
}

std::string dump_header(std::uint32_t width, std::uint32_t height,
                        const Cli& cli, const av2enc::cuda::m1gpu::GpuResult& r) {
  const std::uint32_t cw = (width + 1U) / 2U;
  const std::uint32_t ch = (height + 1U) / 2U;
  auto flag = [](bool value) { return value ? "true" : "false"; };
  std::string j = "{";
  j += "\"width\":" + std::to_string(width);
  j += ",\"height\":" + std::to_string(height);
  j += ",\"chroma_width\":" + std::to_string(cw);
  j += ",\"chroma_height\":" + std::to_string(ch);
  j += ",\"base_q_idx\":" + std::to_string(cli.options.base_q_idx);
  j += ",\"max_partition_depth\":";
  j += std::to_string(cli.options.max_partition_depth);
  j += ",\"min_block_size\":" + std::to_string(cli.options.min_block_size);
  j += ",\"enable_deblock\":";
  j += flag(cli.options.enable_deblock);
  j += ",\"enable_cfl\":";
  j += flag(cli.options.enable_cfl);
  j += ",\"enable_intra_edge_filter\":";
  j += flag(cli.options.enable_edge_filter);
  j += ",\"luma_dc_only\":";
  j += flag(cli.options.luma_dc_only);
  j += ",\"adaptive_cdf\":";
  j += flag(cli.options.adaptive_cdf);
  j += ",\"force_single_block\":";
  j += flag(cli.options.force_single_block);
  j += ",\"force_dc\":";
  j += flag(cli.options.force_dc);
  j += ",\"tile_bytes\":" + std::to_string(r.tile_bytes);
  j += ",\"frame_bytes\":" + std::to_string(r.frame_bytes);
  j += ",\"frame_bytes_from_cpu\":";
  j += flag(r.frame_bytes_from_cpu);
  j += ",\"source\":\"m1_gpu_probe\"}";
  return j;
}

bool write_dump(const std::filesystem::path& path, const std::string& header,
                const std::vector<std::uint8_t>& y,
                const std::vector<std::uint8_t>& u,
                const std::vector<std::uint8_t>& v) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) return false;
  const std::string magic = "M1DUMP1\n" + header + "\n";
  out.write(magic.data(), static_cast<std::streamsize>(magic.size()));
  const auto plane = [&out](const std::vector<std::uint8_t>& data) {
    if (data.empty()) return true;
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(data.size()));
    return static_cast<bool>(out);
  };
  return plane(y) && plane(u) && plane(v) && static_cast<bool>(out);
}

}  // namespace

int main(int argc, char** argv) {
  Cli cli;
  const int parse_status = parse_args(argc, argv, cli);
  if (parse_status == 0) {
    usage(std::cout);
    return 0;
  }
  if (parse_status > 0) return parse_status;

  av2enc::Y4mReader reader(cli.input);
  std::string error;
  if (!reader.open(error)) {
    std::cerr << "m1_gpu_probe: " << error << "\n";
    return 1;
  }
  av2enc::Y4mFrame frame;
  const av2enc::Y4mReadResult read = reader.read_frame(frame, error);
  if (read != av2enc::Y4mReadResult::kFrame) {
    std::cerr << "m1_gpu_probe: "
              << (read == av2enc::Y4mReadResult::kEof ? "input has no frames"
                                                      : error)
              << "\n";
    return 1;
  }

  av2enc::m1::StillPicture picture;
  picture.width = frame.width();
  picture.height = frame.height();
  picture.y = frame.y();
  picture.u = frame.u();
  picture.v = frame.v();

  av2enc::cuda::m1gpu::GpuResult result;
  const auto start = std::chrono::steady_clock::now();
  const bool ok = av2enc::cuda::m1gpu::encode_m1_gpu(picture, cli.options,
                                                     result, error);
  const auto stop = std::chrono::steady_clock::now();
  if (!ok) {
    std::cerr << "m1_gpu_probe: " << error << "\n";
    return 1;
  }
  const double wall_ms =
      std::chrono::duration<double, std::milli>(stop - start).count();

  const std::string header = dump_header(picture.width, picture.height, cli,
                                         result);
  if (!write_dump(cli.output, header, result.recon_y, result.recon_u,
                  result.recon_v)) {
    std::cerr << "m1_gpu_probe: failed to write " << cli.output << "\n";
    return 1;
  }

  const std::string stats = stats_json(cli, picture.width, picture.height,
                                       result, wall_ms);
  if (!cli.stats.empty()) {
    std::ofstream out(cli.stats, std::ios::binary | std::ios::trunc);
    if (!out || !(out << stats << "\n")) {
      std::cerr << "m1_gpu_probe: failed to write " << cli.stats << "\n";
      return 1;
    }
  }
  if (cli.print_json) {
    std::cout << stats << "\n";
  }

  const bool gpu_bytes = any_gpu_stage(result.stages) &&
                         !result.frame_bytes_from_cpu;
  std::cout << "m1_gpu_probe: " << picture.width << "x" << picture.height
            << " backend=" << (gpu_bytes ? "gpu" : "cpu-reference")
            << " launches=" << result.stats.kernel_launches
            << " arena_peak=" << result.stats.arena_used_bytes << "B"
            << " wall=" << wall_ms << "ms"
            << " tile=" << result.tile_bytes << "B"
            << " frame=" << result.frame_bytes << "B"
            << " -> " << cli.output << "\n";
  if (!gpu_bytes) {
    std::cout << "m1_gpu_probe: note: frame bytes came from the CPU reference "
                 "or no stage reported kGpu\n";
  }
  return 0;
}

// Precompute-aggressiveness / VRAM estimator for av2enc.
//
// This is a planning tool, not an allocator. It computes the frame, persistent
// state, scratch, RDO candidate-cache and headroom requirements for a
// configuration, selects a profile from the detected VRAM and memory
// bandwidth, and fails cleanly if the chosen profile does not fit the cap.
//
// The byte model is deliberately conservative and is documented in
// docs/vram.md. It does not touch GPU memory beyond cudaMemGetInfo.

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

constexpr uint64_t kMiB = 1024ull * 1024ull;
constexpr uint64_t kHeadroomMb = 512;
constexpr uint64_t kStateBytesPerBlock = 64;   // per 16x16 block
constexpr uint64_t kScratchBytesPerPixel = 24;  // search/transform workspace
constexpr uint64_t kScratchFixedPerTile = 2 * kMiB;

struct Config {
  int width = 3840;
  int height = 2160;
  int bit_depth = 10;
  std::string chroma = "420";
  int refs = -1;
  int lookahead = -1;
  int tile_cols = -1;
  int tile_rows = -1;
  int rdo_mult = -1;
  int search_radius = -1;
  int cap_mb = -1;
  int total_mb = -1;  // simulation override (--total-vram-mb)
  double bandwidth_gbps = -1.0;
  bool bandwidth_unknown = false;
  int cc_major = -1;
  int cc_minor = -1;
  std::string profile = "auto";
};

struct Profile {
  const char* name;
  uint64_t cap_mb;
  int refs;
  int lookahead;
  int tile_cols;
  int tile_cols_large;  // for frames above large_at_pixels
  uint64_t large_at_pixels;
  int tile_rows;
  int rdo_mult;
  int search_radius;
  uint64_t min_total_mb;
  double min_bandwidth_gbps;
  const char* recommended;
};

constexpr Profile kProfiles[] = {
    {"compat4g", 3500, 2, 1, 2, 2, 0, 2, 1, 32, 0, 0.0, "1080p"},
    {"compat6g", 5500, 3, 2, 2, 4, 1920ull * 1080ull, 2, 1, 48, 5000, 0.0,
     "1080p / light 1440p"},
    {"compat8g", 7500, 4, 4, 4, 4, 0, 2, 1, 64, 7000, 0.0, "1080p / 4K light"},
    {"auto12g", 11000, 8, 12, 8, 8, 0, 2, 2, 96, 11000, 200.0, "4K"},
    {"auto16g", 15000, 12, 16, 8, 16, 3840ull * 2160ull, 2, 3, 112, 16000,
     250.0, "4K / 8K light"},
    {"auto24g", 20000, 16, 24, 8, 16, 3840ull * 2160ull, 2, 4, 128, 24000,
     300.0, "8K"},
};

struct Estimate {
  uint64_t pixels = 0;
  uint64_t samples = 0;
  uint64_t frame_bytes = 0;
  uint64_t state_bytes = 0;
  uint64_t refs_bytes = 0;
  uint64_t lookahead_bytes = 0;
  uint64_t current_bytes = 0;
  uint64_t persistent_bytes = 0;
  uint64_t tiles = 0;
  uint64_t tile_pixels = 0;
  uint64_t scratch_per_tile_bytes = 0;
  uint64_t scratch_bytes = 0;
  uint64_t candidate_bytes = 0;
  uint64_t headroom_bytes = kHeadroomMb * kMiB;
  uint64_t total_bytes = 0;
  int refs = 0;
  int lookahead = 0;
  int rdo_mult = 0;
  int search_radius = 0;
  int tile_cols = 0;
  int tile_rows = 0;
  int bit_depth = 0;
  std::string chroma;
};

double mib(uint64_t bytes) { return static_cast<double>(bytes) / static_cast<double>(kMiB); }

void usage(const char* argv0) {
  std::printf(
      "usage: %s [options]\n"
      "  --profile NAME        auto|compat4g|compat6g|compat8g|auto12g|auto16g|auto24g\n"
      "                        (default auto)\n"
      "  --width N             frame width (default 3840)\n"
      "  --height N            frame height (default 2160)\n"
      "  --bit-depth N        8 or 10 (default 10)\n"
      "  --chroma FMT          420|422|444 (default 420)\n"
      "  --refs N              override reference count\n"
      "  --lookahead N         override lookahead depth\n"
      "  --tiles CxR           override tile grid, e.g. 4x2\n"
      "  --rdo-mult N          override RDO candidate-cache multiplier\n"
      "  --search-radius N     override motion-search starting radius (pixels)\n"
      "  --cap-mb N            hard VRAM cap in MiB (default: profile cap)\n"
      "  --total-vram-mb N     simulate a device with N MiB total VRAM\n"
      "  --bandwidth-gbps N    simulate memory bandwidth (default: query)\n"
      "  --bandwidth-unknown   force capacity-only auto selection\n"
      "  --cc MAJOR.MINOR      simulate compute capability (e.g. 7.5)\n"
      "  --help                show this help\n",
      argv0);
}

bool parse_int(const char* text, int* out) {
  errno = 0;
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || value <= 0 || value > 1000000) {
    return false;
  }
  *out = static_cast<int>(value);
  return true;
}

std::string normalize_chroma(const char* text) {
  std::string s(text);
  s.erase(std::remove(s.begin(), s.end(), ':'), s.end());
  return s;
}

const Profile* find_profile(const std::string& name) {
  for (const Profile& p : kProfiles) {
    if (name == p.name) {
      return &p;
    }
  }
  return nullptr;
}

Estimate estimate_config(const Config& cfg, const Profile& profile) {
  Estimate e;
  e.refs = (cfg.refs >= 0) ? cfg.refs : profile.refs;
  e.lookahead = (cfg.lookahead >= 0) ? cfg.lookahead : profile.lookahead;
  e.rdo_mult = (cfg.rdo_mult >= 0) ? cfg.rdo_mult : profile.rdo_mult;
  e.search_radius =
      (cfg.search_radius >= 0) ? cfg.search_radius : profile.search_radius;
  e.tile_cols = (cfg.tile_cols >= 0) ? cfg.tile_cols : profile.tile_cols;
  e.tile_rows = (cfg.tile_rows >= 0) ? cfg.tile_rows : profile.tile_rows;
  e.bit_depth = cfg.bit_depth;
  e.chroma = cfg.chroma;

  e.pixels = static_cast<uint64_t>(cfg.width) * static_cast<uint64_t>(cfg.height);
  // The 2x2/4x2 and 8x2/16x2 presets widen only above their documented
  // resolution threshold.
  if (cfg.tile_cols < 0 && profile.tile_cols_large > profile.tile_cols &&
      e.pixels > profile.large_at_pixels) {
    e.tile_cols = profile.tile_cols_large;
  }
  const uint64_t w = static_cast<uint64_t>(cfg.width);
  const uint64_t h = static_cast<uint64_t>(cfg.height);
  if (cfg.chroma == "420") {
    e.samples = e.pixels + 2 * ((w + 1) / 2) * ((h + 1) / 2);
  } else if (cfg.chroma == "422") {
    e.samples = e.pixels + 2 * ((w + 1) / 2) * h;
  } else {
    e.samples = 3 * e.pixels;
  }
  const uint64_t bytes_per_sample = (cfg.bit_depth <= 8) ? 1 : 2;
  e.frame_bytes = e.samples * bytes_per_sample;

  e.state_bytes = std::max<uint64_t>(
      1 * kMiB, ((e.pixels + 255) / 256) * kStateBytesPerBlock);
  e.refs_bytes = static_cast<uint64_t>(e.refs) * e.frame_bytes;
  e.lookahead_bytes = static_cast<uint64_t>(e.lookahead) * e.frame_bytes;
  e.current_bytes = 2 * e.frame_bytes;  // current source + reconstruction
  e.persistent_bytes =
      e.refs_bytes + e.lookahead_bytes + e.current_bytes + e.state_bytes;

  e.tiles = static_cast<uint64_t>(e.tile_cols) * static_cast<uint64_t>(e.tile_rows);
  e.tile_pixels = (e.pixels + e.tiles - 1) / e.tiles;
  e.scratch_per_tile_bytes =
      e.tile_pixels * kScratchBytesPerPixel + kScratchFixedPerTile;
  e.scratch_bytes = e.scratch_per_tile_bytes * e.tiles;

  e.candidate_bytes = e.frame_bytes * static_cast<uint64_t>(e.rdo_mult);
  e.total_bytes =
      e.persistent_bytes + e.scratch_bytes + e.candidate_bytes + e.headroom_bytes;
  return e;
}

void print_estimate(const Estimate& e) {
  std::printf("\nbyte breakdown (estimate):\n");
  std::printf("  frame: %.2f MiB (%llu B), %d-bit chroma %s, %llu samples\n",
              mib(e.frame_bytes), static_cast<unsigned long long>(e.frame_bytes),
              e.bit_depth, e.chroma.c_str(),
              static_cast<unsigned long long>(e.samples));
  std::printf("  persistent: %.2f MiB\n", mib(e.persistent_bytes));
  std::printf("    refs:      %d x %.2f = %.2f MiB\n", e.refs, mib(e.frame_bytes),
              mib(e.refs_bytes));
  std::printf("    lookahead: %d x %.2f = %.2f MiB\n", e.lookahead,
              mib(e.frame_bytes), mib(e.lookahead_bytes));
  std::printf("    current+recon: 2 x %.2f = %.2f MiB\n", mib(e.frame_bytes),
              mib(e.current_bytes));
  std::printf("    state:     %.2f MiB (64 B / 16x16 block, min 1 MiB)\n",
              mib(e.state_bytes));
  std::printf("  scratch: %.2f MiB\n", mib(e.scratch_bytes));
  std::printf("    %llu tiles, %.2f MiB/tile (%llu tile pixels x 24 B + 2 MiB)\n",
              static_cast<unsigned long long>(e.tiles),
              mib(e.scratch_per_tile_bytes),
              static_cast<unsigned long long>(e.tile_pixels));
  std::printf("  candidate cache: %dx frame = %.2f MiB\n", e.rdo_mult,
              mib(e.candidate_bytes));
  std::printf("  search radius: +/-%d pixels (tunable, not in byte model)\n",
              e.search_radius);
  std::printf("  headroom: %.2f MiB\n", mib(e.headroom_bytes));
  std::printf("  estimate total: %.2f MiB (%llu B)\n", mib(e.total_bytes),
              static_cast<unsigned long long>(e.total_bytes));
}

}  // namespace

int main(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    const char* arg = argv[i];
    const bool has_value = (i + 1 < argc);
    if (std::strcmp(arg, "--help") == 0) {
      usage(argv[0]);
      return EXIT_SUCCESS;
    } else if (std::strcmp(arg, "--profile") == 0 && has_value) {
      cfg.profile = argv[++i];
    } else if (std::strcmp(arg, "--width") == 0 && has_value) {
      if (!parse_int(argv[++i], &cfg.width)) {
        std::fprintf(stderr, "invalid --width\n");
        return EXIT_FAILURE;
      }
    } else if (std::strcmp(arg, "--height") == 0 && has_value) {
      if (!parse_int(argv[++i], &cfg.height)) {
        std::fprintf(stderr, "invalid --height\n");
        return EXIT_FAILURE;
      }
    } else if (std::strcmp(arg, "--bit-depth") == 0 && has_value) {
      if (!parse_int(argv[++i], &cfg.bit_depth)) {
        std::fprintf(stderr, "invalid --bit-depth\n");
        return EXIT_FAILURE;
      }
    } else if (std::strcmp(arg, "--chroma") == 0 && has_value) {
      cfg.chroma = normalize_chroma(argv[++i]);
    } else if (std::strcmp(arg, "--refs") == 0 && has_value) {
      if (!parse_int(argv[++i], &cfg.refs)) {
        std::fprintf(stderr, "invalid --refs\n");
        return EXIT_FAILURE;
      }
    } else if (std::strcmp(arg, "--lookahead") == 0 && has_value) {
      if (!parse_int(argv[++i], &cfg.lookahead)) {
        std::fprintf(stderr, "invalid --lookahead\n");
        return EXIT_FAILURE;
      }
    } else if (std::strcmp(arg, "--rdo-mult") == 0 && has_value) {
      if (!parse_int(argv[++i], &cfg.rdo_mult)) {
        std::fprintf(stderr, "invalid --rdo-mult\n");
        return EXIT_FAILURE;
      }
    } else if (std::strcmp(arg, "--search-radius") == 0 && has_value) {
      if (!parse_int(argv[++i], &cfg.search_radius)) {
        std::fprintf(stderr, "invalid --search-radius\n");
        return EXIT_FAILURE;
      }
    } else if (std::strcmp(arg, "--cap-mb") == 0 && has_value) {
      if (!parse_int(argv[++i], &cfg.cap_mb)) {
        std::fprintf(stderr, "invalid --cap-mb\n");
        return EXIT_FAILURE;
      }
    } else if ((std::strcmp(arg, "--total-vram-mb") == 0 ||
                std::strcmp(arg, "--total-mb") == 0) &&
               has_value) {
      if (!parse_int(argv[++i], &cfg.total_mb)) {
        std::fprintf(stderr, "invalid --total-vram-mb\n");
        return EXIT_FAILURE;
      }
    } else if (std::strcmp(arg, "--bandwidth-unknown") == 0) {
      cfg.bandwidth_unknown = true;
    } else if (std::strcmp(arg, "--bandwidth-gbps") == 0 && has_value) {
      errno = 0;
      char* end = nullptr;
      const double value = std::strtod(argv[++i], &end);
      if (errno != 0 || end == argv[i] || *end != '\0' || value <= 0.0) {
        std::fprintf(stderr, "invalid --bandwidth-gbps\n");
        return EXIT_FAILURE;
      }
      cfg.bandwidth_gbps = value;
    } else if (std::strcmp(arg, "--cc") == 0 && has_value) {
      int major = -1;
      int minor = -1;
      if (std::sscanf(argv[++i], "%d.%d", &major, &minor) != 2 || major < 0 ||
          minor < 0) {
        std::fprintf(stderr, "invalid --cc (expected MAJOR.MINOR, e.g. 7.5)\n");
        return EXIT_FAILURE;
      }
      cfg.cc_major = major;
      cfg.cc_minor = minor;
    } else if (std::strcmp(arg, "--tiles") == 0 && has_value) {
      int cols = 0;
      int rows = 0;
      if (std::sscanf(argv[++i], "%dx%d", &cols, &rows) != 2 || cols <= 0 ||
          rows <= 0) {
        std::fprintf(stderr, "invalid --tiles (expected CxR)\n");
        return EXIT_FAILURE;
      }
      cfg.tile_cols = cols;
      cfg.tile_rows = rows;
    } else {
      std::fprintf(stderr, "unknown or incomplete argument: %s\n", arg);
      usage(argv[0]);
      return EXIT_FAILURE;
    }
  }

  if (cfg.bit_depth != 8 && cfg.bit_depth != 10) {
    std::fprintf(stderr, "unsupported bit depth %d (supported: 8, 10)\n",
                 cfg.bit_depth);
    return EXIT_FAILURE;
  }
  if (cfg.chroma != "420" && cfg.chroma != "422" && cfg.chroma != "444") {
    std::fprintf(stderr, "unsupported chroma '%s' (supported: 420, 422, 444)\n",
                 cfg.chroma.c_str());
    return EXIT_FAILURE;
  }
  if (cfg.chroma != "444" && ((cfg.width % 2) != 0 || (cfg.height % 2) != 0)) {
    std::fprintf(stderr, "subsampled chroma requires even width/height\n");
    return EXIT_FAILURE;
  }
  if (cfg.width <= 0 || cfg.height <= 0 || cfg.width > 65536 || cfg.height > 65536) {
    std::fprintf(stderr, "unsupported frame size %dx%d\n", cfg.width, cfg.height);
    return EXIT_FAILURE;
  }

  cudaError_t status = cudaFree(nullptr);
  if (status != cudaSuccess) {
    std::fprintf(stderr, "CUDA init failed: %s\n", cudaGetErrorString(status));
    return EXIT_FAILURE;
  }

  cudaDeviceProp prop{};
  status = cudaGetDeviceProperties(&prop, 0);
  if (status != cudaSuccess) {
    std::fprintf(stderr, "cudaGetDeviceProperties failed: %s\n",
                 cudaGetErrorString(status));
    return EXIT_FAILURE;
  }
  const int cc_major = (cfg.cc_major >= 0) ? cfg.cc_major : prop.major;
  const int cc_minor = (cfg.cc_minor >= 0) ? cfg.cc_minor : prop.minor;
  const bool cc_simulated = (cfg.cc_major >= 0);
  const bool cc_supported =
      (cc_major == 8 && cc_minor >= 6) || (cc_major == 12 && cc_minor == 0);

  size_t free_size = 0;
  size_t total_size = 0;
  status = cudaMemGetInfo(&free_size, &total_size);
  if (status != cudaSuccess) {
    std::fprintf(stderr, "cudaMemGetInfo failed: %s\n", cudaGetErrorString(status));
    return EXIT_FAILURE;
  }

  const uint64_t detected_total_mb = static_cast<uint64_t>(total_size) / kMiB;
  const uint64_t detected_free_mb = static_cast<uint64_t>(free_size) / kMiB;
  const uint64_t detected_used_mb = detected_total_mb - detected_free_mb;
  const double detected_bw_gbps =
      2.0 * static_cast<double>(prop.memoryClockRate) * 1e3 *
      (static_cast<double>(prop.memoryBusWidth) / 8.0) / 1e9;
  const bool bandwidth_known =
      !cfg.bandwidth_unknown &&
      (cfg.bandwidth_gbps > 0.0 ||
       (prop.memoryClockRate > 0 && prop.memoryBusWidth > 0));

  const bool simulated = (cfg.total_mb > 0 || cfg.bandwidth_gbps > 0.0);
  const uint64_t total_mb = (cfg.total_mb > 0) ? static_cast<uint64_t>(cfg.total_mb)
                                               : detected_total_mb;
  const uint64_t effective_free_mb =
      (cfg.total_mb > 0)
          ? (total_mb > detected_used_mb ? total_mb - detected_used_mb : 0)
          : detected_free_mb;
  const uint64_t effective_free_bytes = effective_free_mb * kMiB;
  const double bandwidth_gbps =
      (cfg.bandwidth_gbps > 0.0) ? cfg.bandwidth_gbps : detected_bw_gbps;

  std::printf("=== av2enc VRAM config estimator ===\n");
  std::printf("device: %s (detected sm_%d%d)\n", prop.name, prop.major, prop.minor);
  std::printf("compute capability: %d.%d %s; current fatbin: sm_86;sm_120\n",
              cc_major, cc_minor, cc_simulated ? "(SIMULATED)" : "(detected)");
  if (!cc_supported) {
    std::fprintf(stderr,
                 "\nRESULT: FAIL - compute capability %d.%d (sm_%d%d) is not "
                 "supported by the current fatbin (sm_86;sm_120).\n",
                 cc_major, cc_minor, cc_major, cc_minor);
    if (cc_major == 7 && cc_minor == 5) {
      std::fprintf(stderr,
                   "sm_75 (GTX 1650/1660, RTX 2060-class) requires a separate "
                   "sm_75 build and a no-cp.async fallback; the estimator "
                   "refuses to select a precompute profile.\n");
    }
    std::fprintf(stderr,
                 "Add the architecture to the fatbin and to the portability "
                 "matrix before selecting a profile.\n");
    return 2;
  }
  std::printf("detected: total=%llu MiB, free=%llu MiB, bandwidth=",
              static_cast<unsigned long long>(detected_total_mb),
              static_cast<unsigned long long>(detected_free_mb));
  if (bandwidth_known) {
    std::printf("%.1f GB/s\n", detected_bw_gbps);
  } else {
    std::printf("unknown\n");
  }
  if (simulated) {
    std::printf("SIMULATED: total=%llu MiB, free=%llu MiB, bandwidth=",
                static_cast<unsigned long long>(total_mb),
                static_cast<unsigned long long>(effective_free_mb));
    if (bandwidth_known) {
      std::printf("%.1f GB/s\n", bandwidth_gbps);
    } else {
      std::printf("unknown\n");
    }
  } else {
    std::printf("using:    total=%llu MiB, free=%llu MiB, bandwidth=",
                static_cast<unsigned long long>(total_mb),
                static_cast<unsigned long long>(effective_free_mb));
    if (bandwidth_known) {
      std::printf("%.1f GB/s\n", bandwidth_gbps);
    } else {
      std::printf("unknown (capacity-only selection)\n");
    }
  }
  std::printf("input: %dx%d, %d-bit, chroma=%s, profile=%s\n", cfg.width,
              cfg.height, cfg.bit_depth, cfg.chroma.c_str(), cfg.profile.c_str());

  if (cfg.cap_mb > 0 && static_cast<uint64_t>(cfg.cap_mb) > total_mb) {
    std::fprintf(stderr,
                 "\nRESULT: FAIL - cap %d MiB exceeds device total %llu MiB; "
                 "refusing an impossible budget\n",
                 cfg.cap_mb, static_cast<unsigned long long>(total_mb));
    return 2;
  }

  const Profile* chosen = nullptr;
  if (cfg.profile == "auto") {
    std::printf("\nauto selection (capacity class, bandwidth floor):\n");
    std::printf(
        "  floors: 6g >=5000 MiB; 8g >=7000 MiB; 12g >=11000 MiB & "
        ">=200 GB/s; 16g >=16000 MiB & >=250 GB/s; 24g >=24000 MiB & "
        ">=300 GB/s; 4g fallback\n");
    if (!bandwidth_known) {
      std::printf("  bandwidth unknown -> capacity-only selection\n");
    }
    const Profile* const order[] = {&kProfiles[5], &kProfiles[4],
                                    &kProfiles[3], &kProfiles[2],
                                    &kProfiles[1], &kProfiles[0]};
    for (const Profile* pp : order) {
      const Profile& p = *pp;
      const uint64_t effective_cap =
          (cfg.cap_mb > 0) ? static_cast<uint64_t>(cfg.cap_mb) : p.cap_mb;
      const Estimate e = estimate_config(cfg, p);
      const char* reason = nullptr;
      if (total_mb < p.min_total_mb) {
        reason = "device total below profile minimum";
      } else if (bandwidth_known && bandwidth_gbps < p.min_bandwidth_gbps) {
        reason = "bandwidth below profile minimum";
      } else if (effective_cap > total_mb) {
        reason = "cap exceeds device total";
      } else if (p.cap_mb > effective_cap) {
        reason = "nominal profile cap exceeds effective cap";
      } else if (e.total_bytes > effective_cap * kMiB) {
        reason = "estimate exceeds cap";
      } else if (e.total_bytes > effective_free_bytes) {
        reason = "estimate exceeds simulated/current free VRAM";
      }
      if (reason == nullptr) {
        chosen = pp;
        std::printf("  %-8s SELECTED (cap=%llu MiB)\n", p.name,
                    static_cast<unsigned long long>(effective_cap));
        break;
      }
      std::printf("  %-8s rejected: %s\n", p.name, reason);
    }
    if (chosen == nullptr) {
      std::fprintf(stderr,
                   "\nRESULT: FAIL - no profile fits total=%llu MiB, "
                   "bandwidth=%s, cap=%d MiB\n",
                   static_cast<unsigned long long>(total_mb),
                   bandwidth_known ? "known" : "unknown", cfg.cap_mb);
      return 2;
    }
  } else {
    chosen = find_profile(cfg.profile);
    if (chosen == nullptr) {
      std::fprintf(stderr, "unknown profile '%s'\n", cfg.profile.c_str());
      return EXIT_FAILURE;
    }
  }

  const uint64_t cap_mb =
      (cfg.cap_mb > 0) ? static_cast<uint64_t>(cfg.cap_mb) : chosen->cap_mb;
  if (cap_mb > total_mb) {
    std::fprintf(stderr,
                 "\nRESULT: FAIL - cap %llu MiB exceeds device total %llu MiB; "
                 "refusing an impossible budget\n",
                 static_cast<unsigned long long>(cap_mb),
                 static_cast<unsigned long long>(total_mb));
    return 2;
  }

  const Estimate e = estimate_config(cfg, *chosen);
  std::printf("\nprofile: %s (cap=%llu MiB, refs=%d, lookahead=%d, "
              "tiles=%dx%d (%llu total), rdo=%dx, search=+/-%d)\n",
              chosen->name, static_cast<unsigned long long>(cap_mb), e.refs,
              e.lookahead, e.tile_cols, e.tile_rows,
              static_cast<unsigned long long>(e.tiles), e.rdo_mult,
              e.search_radius);
  std::printf("recommendation: %s (guidance; the hard check is the cap)\n",
              chosen->recommended);
  print_estimate(e);

  const uint64_t cap_bytes = cap_mb * kMiB;
  const bool fits_cap = (e.total_bytes <= cap_bytes);
  const bool fits_physical = (e.total_bytes <= total_mb * kMiB);
  const bool fits_free = (e.total_bytes <= effective_free_bytes);
  std::printf("\nchecks:\n");
  std::printf("  cap: %llu MiB -> %s", static_cast<unsigned long long>(cap_mb),
              fits_cap ? "fits" : "EXCEEDS");
  if (fits_cap) {
    std::printf(" (%.2f MiB spare)\n", mib(cap_bytes - e.total_bytes));
  } else {
    std::printf(" by %.2f MiB\n", mib(e.total_bytes - cap_bytes));
  }
  std::printf("  physical total: %llu MiB -> %s\n",
              static_cast<unsigned long long>(total_mb),
              fits_physical ? "fits" : "EXCEEDS");
  std::printf("  available: %llu MiB -> %s\n",
              static_cast<unsigned long long>(effective_free_mb),
              fits_free ? "fits" : "EXCEEDS (close other GPU work)");

  if (!fits_cap || !fits_physical || !fits_free) {
    std::fprintf(stderr, "\nRESULT: FAIL - profile '%s' does not fit; "
                         "reduce refs/lookahead/tile batch/rdo cache or raise the cap\n",
                 chosen->name);
    return 2;
  }

  std::printf("\nRESULT: OK - profile '%s' with cap %llu MiB\n", chosen->name,
              static_cast<unsigned long long>(cap_mb));
  return EXIT_SUCCESS;
}

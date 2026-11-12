// Host-side GpuOptions plumbing test (no CUDA required).
//
// The GPU gate (`tools/m3_gpu/verify_m3_gpu.sh`) classifies a feature as
// SKIP-REJECTED only when the encoder says the flag is unsupported.  A flag
// that is parsed by the CLI, dropped from `GpuOptions`, or omitted from the
// internal CPU parity encode is a correctness bug: it either silently changes
// the GPU/CPU comparison or crashes later.
//
// This test pins both directions:
//   * `gpu_options_to_m1()` maps every field the CLI can set;
//   * `gpu_m3_options_supported()` rejects every M3 tool by name until a GPU
//     stage exists, and accepts the default (no M3 tool) configuration.
//
// Build/run: src/cuda/m1/build_options_test.sh
#include "cuda/pipeline/still/still_gpu.h"

#include <cstdio>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const char* what) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

void check_equal(int got, int want, const char* what) {
  if (got != want) {
    std::fprintf(stderr, "FAIL: %s (got %d, want %d)\n", what, got, want);
    ++failures;
  }
}

void check_close(double got, double want, const char* what) {
  if (got != want) {
    std::fprintf(stderr, "FAIL: %s (got %.17g, want %.17g)\n", what, got,
                 want);
    ++failures;
  }
}

void test_mapping_matches_cli() {
  using av2enc::cuda::still_gpu::GpuOptions;
  using av2enc::cuda::still_gpu::gpu_options_to_m1;

  GpuOptions gpu;
  gpu.base_q_idx = 90;  // GPU pins 90 today
  gpu.enable_cfl = false;
  gpu.enable_deblock = false;
  gpu.enable_edge_filter = false;
  gpu.luma_dc_only = false;
  gpu.min_block_size = 16;
  gpu.max_partition_depth = 2;
  gpu.adaptive_cdf = true;
  gpu.force_single_block = true;
  gpu.force_dc = true;
  gpu.uv_full = true;
  gpu.enable_cdef = true;
  gpu.cdef_num_strengths = 2;
  gpu.cdef_damping = 5;
  gpu.cdef_on_skip_txfm = true;
  gpu.enable_ccso = true;
  gpu.ccso_bands_log2 = 3;
  gpu.rdoq = true;
  gpu.rdoq_sse_scale = 0.25;
  gpu.tx_type = true;
  gpu.tx_type_sse_scale = 0.5;
  gpu.rdo_intra = true;
  gpu.rdo_sse_scale = 1.0;

  const av2enc::still::StillOptions cpu = gpu_options_to_m1(gpu);
  check_equal(cpu.base_q_idx, 90, "base_q_idx");
  check(cpu.enable_cfl == false, "enable_cfl");
  check(cpu.enable_deblock == false, "enable_deblock");
  check(cpu.enable_intra_edge_filter == false, "enable_intra_edge_filter");
  check(cpu.luma_dc_only == false, "luma_dc_only");
  check_equal(cpu.min_block_size, 16, "min_block_size");
  check_equal(cpu.max_partition_depth, 2, "max_partition_depth");
  check(cpu.adaptive_cdf == true, "adaptive_cdf");
  check(cpu.force_single_block == true, "force_single_block");
  check(cpu.force_dc == true, "force_dc");
  check(cpu.uv_full == true, "uv_full");
  check(cpu.enable_cdef == true, "enable_cdef");
  check_equal(cpu.cdef_num_strengths, 2, "cdef_num_strengths");
  check_equal(cpu.cdef_damping, 5, "cdef_damping");
  check(cpu.cdef_on_skip_txfm == true, "cdef_on_skip_txfm");
  check(cpu.enable_ccso == true, "enable_ccso");
  check_equal(cpu.ccso_bands_log2, 3, "ccso_bands_log2");
  check(cpu.rdoq == true, "rdoq");
  check_close(cpu.rdoq_sse_scale, 0.25, "rdoq_sse_scale");
  check(cpu.tx_type == true, "tx_type");
  check_close(cpu.tx_type_sse_scale, 0.5, "tx_type_sse_scale");
  check(cpu.rdo_intra == true, "rdo_intra");
  check_close(cpu.rdo_sse_scale, 1.0, "rdo_sse_scale");
}

// The default GpuOptions must be supported: only M3 tool flags are gated.
void test_default_supported() {
  av2enc::cuda::still_gpu::GpuOptions gpu;
  std::string error;
  check(av2enc::cuda::still_gpu::gpu_m3_options_supported(gpu, &error),
        "default options are supported");
  check(error.empty(), "default options report no error");
}

void expect_rejected(const av2enc::cuda::still_gpu::GpuOptions& gpu,
                     const char* flag, const char* what) {
  std::string error;
  const bool ok = av2enc::cuda::still_gpu::gpu_m3_options_supported(gpu, &error);
  if (ok) {
    std::fprintf(stderr, "FAIL: %s was accepted\n", what);
    ++failures;
    return;
  }
  if (error.find(flag) == std::string::npos) {
    std::fprintf(stderr, "FAIL: %s error does not name %s: %s\n", what, flag,
                 error.c_str());
    ++failures;
  }
  if (error.find("not ported") == std::string::npos) {
    std::fprintf(stderr,
                 "FAIL: %s error is not classified as a rejection: %s\n",
                 what, error.c_str());
    ++failures;
  }
}

void test_every_m3_flag_rejected_by_name() {
  using av2enc::cuda::still_gpu::GpuOptions;

  GpuOptions gpu;
  gpu.enable_cdef = true;
  expect_rejected(gpu, "--still-cdef", "--still-cdef");

  gpu = GpuOptions{};
  gpu.enable_ccso = true;
  expect_rejected(gpu, "--still-ccso", "--still-ccso");

  gpu = GpuOptions{};
  gpu.rdoq = true;
  expect_rejected(gpu, "--still-rdoq", "--still-rdoq");

  gpu = GpuOptions{};
  gpu.tx_type = true;
  expect_rejected(gpu, "--still-tx-type", "--still-tx-type");

  gpu = GpuOptions{};
  gpu.rdo_intra = true;
  expect_rejected(gpu, "--still-rdo-intra", "--still-rdo-intra");

  gpu = GpuOptions{};
  gpu.uv_full = true;
  expect_rejected(gpu, "--still-uv-full", "--still-uv-full");
}

}  // namespace

int main() {
  test_mapping_matches_cli();
  test_default_supported();
  test_every_m3_flag_rejected_by_name();
  if (failures != 0) {
    std::fprintf(stderr, "still_gpu_options_test: %d failure(s)\n", failures);
    return 1;
  }
  std::printf("still_gpu_options_test: PASS\n");
  return 0;
}

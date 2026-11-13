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
// Build/run: src/cuda/pipeline/still/build_options_test.sh
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
  gpu.cfl_ds_filter = 2;
  gpu.enable_deblock = false;
  gpu.enable_edge_filter = false;
  gpu.luma_dc_only = false;
  gpu.min_block_size = 16;
  gpu.max_partition_depth = 2;
  gpu.sb_size = 128;
  gpu.adaptive_cdf = true;
  gpu.force_single_block = true;
  gpu.force_dc = true;
  gpu.uv_full = true;
  gpu.enable_ibp = true;
  gpu.enable_mrl = true;
  gpu.enable_dip = true;
  gpu.cfl_explicit = true;
  gpu.mhccp = true;
  gpu.palette = true;
  gpu.ibc = true;
  gpu.ibc_search_step = 4;
  gpu.enable_cdef = true;
  gpu.cdef_num_strengths = 2;
  gpu.cdef_damping = 5;
  gpu.cdef_on_skip_txfm = true;
  gpu.enable_ccso = true;
  gpu.ccso_bands_log2 = 3;
  gpu.ccso_edge = true;
  gpu.enable_wiener = true;
  gpu.enable_chroma_lr = true;
  gpu.enable_pc_wiener = true;
  gpu.enable_gdf = true;
  gpu.enable_grain = true;
  gpu.delta_q = true;
  gpu.enable_segmentation = true;
  gpu.enable_delta_q = true;
  gpu.delta_q_res = 2;
  gpu.per_segment_q = true;
  gpu.per_segment_q_delta = 12;
  gpu.qm = true;
  gpu.qm_level = 2;
  gpu.rdoq = true;
  gpu.rdoq_sse_scale = 0.25;
  gpu.tx_type = true;
  gpu.tx_type_sse_scale = 0.5;
  gpu.rdo_intra = true;
  gpu.rdo_sse_scale = 1.0;
  gpu.ist = true;
  gpu.ist_sse_scale = 0.75;
  gpu.idtx = true;
  gpu.tcq = true;
  gpu.tcq_sse_scale = 0.25;
  gpu.cctx = true;
  gpu.tx_partition = true;

  const av2enc::still::StillOptions cpu = gpu_options_to_m1(gpu);
  check_equal(cpu.base_q_idx, 90, "base_q_idx");
  check(cpu.enable_cfl == false, "enable_cfl");
  check_equal(cpu.cfl_ds_filter, 2, "cfl_ds_filter");
  check(cpu.enable_deblock == false, "enable_deblock");
  check(cpu.enable_intra_edge_filter == false, "enable_intra_edge_filter");
  check(cpu.luma_dc_only == false, "luma_dc_only");
  check_equal(cpu.min_block_size, 16, "min_block_size");
  check_equal(cpu.max_partition_depth, 2, "max_partition_depth");
  check(cpu.sb128 == true, "sb128");
  check(cpu.sb256 == false, "sb256");
  check(cpu.adaptive_cdf == true, "adaptive_cdf");
  check(cpu.force_single_block == true, "force_single_block");
  check(cpu.force_dc == true, "force_dc");
  check(cpu.uv_full == true, "uv_full");
  check(cpu.enable_ibp == true, "enable_ibp");
  check(cpu.enable_mrl == true, "enable_mrl");
  check(cpu.enable_dip == true, "enable_dip");
  check(cpu.cfl_explicit == true, "cfl_explicit");
  check(cpu.mhccp == true, "mhccp");
  check(cpu.palette == true, "palette");
  check(cpu.ibc == true, "ibc");
  check_equal(cpu.ibc_search_step, 4, "ibc_search_step");
  check(cpu.enable_cdef == true, "enable_cdef");
  check_equal(cpu.cdef_num_strengths, 2, "cdef_num_strengths");
  check_equal(cpu.cdef_damping, 5, "cdef_damping");
  check(cpu.cdef_on_skip_txfm == true, "cdef_on_skip_txfm");
  check(cpu.enable_ccso == true, "enable_ccso");
  check_equal(cpu.ccso_bands_log2, 3, "ccso_bands_log2");
  check(cpu.ccso_edge == true, "ccso_edge");
  check(cpu.enable_wiener == true, "enable_wiener");
  check(cpu.enable_chroma_lr == true, "enable_chroma_lr");
  check(cpu.enable_pc_wiener == true, "enable_pc_wiener");
  check(cpu.enable_gdf == true, "enable_gdf");
  check(cpu.enable_grain == true, "enable_grain");
  check(cpu.delta_q == true, "delta_q");
  check(cpu.enable_segmentation == true, "enable_segmentation");
  check(cpu.enable_delta_q == true, "enable_delta_q");
  check_equal(cpu.delta_q_res, 2, "delta_q_res");
  check(cpu.per_segment_q == true, "per_segment_q");
  check_equal(cpu.per_segment_q_delta, 12, "per_segment_q_delta");
  check(cpu.qm == true, "qm");
  check_equal(cpu.qm_level, 2, "qm_level");
  check(cpu.rdoq == true, "rdoq");
  check_close(cpu.rdoq_sse_scale, 0.25, "rdoq_sse_scale");
  check(cpu.tx_type == true, "tx_type");
  check_close(cpu.tx_type_sse_scale, 0.5, "tx_type_sse_scale");
  check(cpu.rdo_intra == true, "rdo_intra");
  check_close(cpu.rdo_sse_scale, 1.0, "rdo_sse_scale");
  check(cpu.ist == true, "ist");
  check_close(cpu.ist_sse_scale, 0.75, "ist_sse_scale");
  check(cpu.idtx == true, "idtx");
  check(cpu.tcq == true, "tcq");
  check_close(cpu.tcq_sse_scale, 0.25, "tcq_sse_scale");
  check(cpu.cctx == true, "cctx");
  check(cpu.tx_partition == true, "tx_partition");
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

// Filters that have a wired GPU stage must be accepted (the CDEF stage applies
// the §7.18 kernel on device; the CPU reference only supplies the planned
// per-64x64 index grid).
void test_wired_filters_accepted() {
  using av2enc::cuda::still_gpu::GpuOptions;
  GpuOptions gpu;
  gpu.enable_cdef = true;
  std::string error;
  check(av2enc::cuda::still_gpu::gpu_m3_options_supported(gpu, &error),
        "--still-cdef is wired into the GPU pipeline");
  check(error.empty(), "--still-cdef reports no rejection");

  // Film grain is decoder-side synthesis; the encoder only emits the model
  // OBU and the frame-header config, so it is wired on the GPU path.
  gpu = GpuOptions{};
  gpu.enable_grain = true;
  error.clear();
  check(av2enc::cuda::still_gpu::gpu_m3_options_supported(gpu, &error),
        "--still-grain is supported on the GPU path (metadata only)");
  check(error.empty(), "--still-grain reports no rejection");

  // CCSO (both the band-offset-only and the edge-classified form) applies the
  // §7.19 kernel on device.
  gpu = GpuOptions{};
  gpu.enable_ccso = true;
  error.clear();
  check(av2enc::cuda::still_gpu::gpu_m3_options_supported(gpu, &error),
        "--still-ccso is wired into the GPU pipeline");
  check(error.empty(), "--still-ccso reports no rejection");

  gpu = GpuOptions{};
  gpu.enable_ccso = true;
  gpu.ccso_edge = true;
  error.clear();
  check(av2enc::cuda::still_gpu::gpu_m3_options_supported(gpu, &error),
        "--still-ccso-edge is wired into the GPU pipeline");
  check(error.empty(), "--still-ccso-edge reports no rejection");

  // §7.20 loop restoration: Wiener NS luma and shared chroma LR both apply
  // their device kernels after CDEF/CCSO.
  gpu = GpuOptions{};
  gpu.enable_wiener = true;
  error.clear();
  check(av2enc::cuda::still_gpu::gpu_m3_options_supported(gpu, &error),
        "--still-wiener is wired into the GPU pipeline");
  check(error.empty(), "--still-wiener reports no rejection");

  gpu = GpuOptions{};
  gpu.enable_chroma_lr = true;
  error.clear();
  check(av2enc::cuda::still_gpu::gpu_m3_options_supported(gpu, &error),
        "--still-chroma-lr is wired into the GPU pipeline");
  check(error.empty(), "--still-chroma-lr reports no rejection");

  // §7.20.4 PC-Wiener classifies and filters luma on the device.
  gpu = GpuOptions{};
  gpu.enable_pc_wiener = true;
  error.clear();
  check(av2enc::cuda::still_gpu::gpu_m3_options_supported(gpu, &error),
        "--still-pc-wiener is wired into the GPU pipeline");
  check(error.empty(), "--still-pc-wiener reports no rejection");

  // §7.20.5 GDF runs last, on the post-restoration LrFrame.
  gpu = GpuOptions{};
  gpu.enable_gdf = true;
  error.clear();
  check(av2enc::cuda::still_gpu::gpu_m3_options_supported(gpu, &error),
        "--still-gdf is wired into the GPU pipeline");
  check(error.empty(), "--still-gdf reports no rejection");

  // §5.18.6.2 QM changes the per-coefficient q2 and rides the shared device
  // delta-Q quantizer.
  gpu = GpuOptions{};
  gpu.qm = true;
  error.clear();
  check(av2enc::cuda::still_gpu::gpu_m3_options_supported(gpu, &error),
        "--still-qm is wired into the GPU pipeline");
  check(error.empty(), "--still-qm reports no rejection");

  // §5.18.7.1/.8 segmentation and delta-Q syntax reaches the CPU writer from
  // the GPU plan; the neutral table does not change the reconstruction.
  gpu = GpuOptions{};
  gpu.enable_segmentation = true;
  error.clear();
  check(av2enc::cuda::still_gpu::gpu_m3_options_supported(gpu, &error),
        "--still-segmentation is wired into the GPU pipeline");
  check(error.empty(), "--still-segmentation reports no rejection");

  gpu = GpuOptions{};
  gpu.enable_delta_q = true;
  gpu.delta_q_res = 1;
  error.clear();
  check(av2enc::cuda::still_gpu::gpu_m3_options_supported(gpu, &error),
        "--still-delta-q is wired into the GPU pipeline");
  check(error.empty(), "--still-delta-q reports no rejection");

  // §5.4.9 per-superblock SEG_LVL_ALT_Q segmentation derives the segment map
  // and per-SB qindex on the device.
  gpu = GpuOptions{};
  gpu.per_segment_q = true;
  gpu.per_segment_q_delta = 8;
  error.clear();
  check(av2enc::cuda::still_gpu::gpu_m3_options_supported(gpu, &error),
        "--still-per-segment-q is wired into the GPU pipeline");
  check(error.empty(), "--still-per-segment-q reports no rejection");
  // §7.13.5 full-UV chroma search: the device candidate loop consumes the
  // luma-mode/AngleDeltaY and UVSmooth inputs the orchestrator now fills.
  gpu = GpuOptions{};
  gpu.uv_full = true;
  error.clear();
  check(av2enc::cuda::still_gpu::gpu_m3_options_supported(gpu, &error),
        "--still-uv-full is wired into the GPU pipeline");
  check(error.empty(), "--still-uv-full reports no rejection");

  // §5.20.8 palette and §5.20.5.3-.4 IBC: the CPU reference plans the colour
  // set / block vector and the GPU applies the palette and copy kernels.
  gpu = GpuOptions{};
  gpu.palette = true;
  error.clear();
  check(av2enc::cuda::still_gpu::gpu_m3_options_supported(gpu, &error),
        "--still-palette is wired into the GPU pipeline");
  check(error.empty(), "--still-palette reports no rejection");

  gpu = GpuOptions{};
  gpu.ibc = true;
  error.clear();
  check(av2enc::cuda::still_gpu::gpu_m3_options_supported(gpu, &error),
        "--still-ibc is wired into the GPU pipeline");
  check(error.empty(), "--still-ibc reports no rejection");}

void test_every_m3_flag_rejected_by_name() {
  using av2enc::cuda::still_gpu::GpuOptions;

  GpuOptions gpu;
  std::string error;
  gpu = GpuOptions{};
  gpu.rdoq = true;
  expect_rejected(gpu, "--still-rdoq", "--still-rdoq");

  gpu = GpuOptions{};
  gpu.tx_type = true;
  check(av2enc::cuda::still_gpu::gpu_m3_options_supported(gpu, &error),
        "--still-tx-type is wired into the GPU pipeline");

  gpu = GpuOptions{};
  gpu.idtx = true;
  check(av2enc::cuda::still_gpu::gpu_m3_options_supported(gpu, &error),
        "--still-idtx is wired into the GPU pipeline");

  gpu = GpuOptions{};
  gpu.ist = true;
  check(av2enc::cuda::still_gpu::gpu_m3_options_supported(gpu, &error),
        "--still-ist is wired into the GPU pipeline");

  gpu = GpuOptions{};
  gpu.tcq = true;
  expect_rejected(gpu, "--still-tcq", "--still-tcq");

  gpu = GpuOptions{};
  gpu.cctx = true;
  check(av2enc::cuda::still_gpu::gpu_m3_options_supported(gpu, &error),
        "--still-cctx is wired into the GPU pipeline");

  gpu = GpuOptions{};
  gpu.tx_partition = true;
  expect_rejected(gpu, "--still-tx-partition", "--still-tx-partition");

  gpu = GpuOptions{};
  gpu.rdo_intra = true;
  expect_rejected(gpu, "--still-rdo-intra", "--still-rdo-intra");

  // The per-SB segmentation path does not carry the separate DC/AC delta-Q
  // plane quantizers yet.
  gpu = GpuOptions{};
  gpu.per_segment_q = true;
  gpu.delta_q = true;
  expect_rejected(gpu, "--still-per-segment-q",
                  "--still-per-segment-q with --still-dc-ac-delta-q");
}

}  // namespace

int main() {
  test_mapping_matches_cli();
  test_default_supported();
  test_wired_filters_accepted();
  test_every_m3_flag_rejected_by_name();
  if (failures != 0) {
    std::fprintf(stderr, "still_gpu_options_test: %d failure(s)\n", failures);
    return 1;
  }
  std::printf("still_gpu_options_test: PASS\n");
  return 0;
}

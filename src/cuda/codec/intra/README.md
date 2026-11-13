# CUDA intra tools

Device implementations and CPU-vs-GPU parity tests for the AV2 intra tools,
plus the arena-backed dispatcher the still orchestrator can call.

| Directory | Tool | Entry point |
|---|---|---|
| `modes/` | §7.13.2 DC/V/H/directional/smooth/Paeth, edge filter, WAIP | `intra_modes::intra_predict_batch` |
| `mrl/` | §7.13.2.1/.7/.8 MRL 0..3 + secondary line | `mrl::mrl_predict_batch` |
| `ibp/` | §7.13.2.7/.9/.12 intra bi-prediction | `ibp::ibp_predict_batch` |
| `dip/` | §7.13.2.3-.6 data-driven intra | `dip::dip_predict_batch` |
| `cfl/` | §7.13.5 CfL explicit + derived alpha, four downsampling filters | `cfl_predict_batch`, `cfl_predict_derived_batch` |
| `palette/` | §5.20.8/§7.13.4 palette map/predict/contexts | `palette_apply_batch` |
| `ibc/` | Intra block copy copy/SAD/validate + arena search | `ibc_copy_block`, `ibc_sad_batch` |
| `supplemental/` | DC-subsampled, filter corner, IDTX/V/H_DCT/ADST/FLIPADST, chroma TX derivation, `dc_sign` | `intra_full::Workspace` |
| `workspace/` | one Arena, one staging region, all of the above | `intra_workspace::Workspace` |

MHCCP's device path lives beside its CPU reference in
`src/codec/intra/cfl/mhccp/` (`mhccp_gpu.cu`, `mhccp_gpu_test.cu`); it is
verified with the same box run and is reached through the CPU-side descriptor,
not through `intra_workspace` (whose `Config` has no MHCCP job yet).

## Evidence (RTX 3060, sm_86, CUDA 13.0)

| Suite | Result |
|---|---|
| `modes/intra_modes_test` | 101,060,224 samples, 0 mismatches |
| `mrl/mrl_test` | PASS 22,512 checks |
| `ibp/build_cuda_test.sh` | PASS 147,814,518 checks |
| `dip/build_gpu_test.sh` | 22,771 checks, 0 failures |
| `cfl/build_cfl_test.sh` | 1,022,200 samples, 412 derived alphas, exact |
| `palette/build_and_test.sh` | 200 checks, 0 failures |
| `ibc/build_and_test.sh` | 36 checks, 0 failures |
| `supplemental/build_gpu_test.sh` | 70,920,528 checks, 0 failures |
| `src/codec/intra/cfl/mhccp/build_gpu_test.sh` | 542,560 samples, 0 mismatches |
| `src/codec/intra/cfl/ds_filters/build_cuda_test.sh` | 597 checks, 0 failures |
| `src/codec/intra/cfl/alpha` (manual sm_86 build) | 110 cases, 3,702,784 samples, exact |
| `workspace/build_intra_workspace_test.sh` | PASS 4,379 checks (1,727,696 CPU-exact predictor samples) |

`tools/check_arena.sh` passes for the tree: no `cudaMalloc`/`cudaFree` outside
`arena.cpp`/`context.cpp`; the workspace, supplemental and IBC search allocate
from the caller's `Arena` only.

## Build

Each module has a self-contained script that does not touch the root CMake
build:

```sh
src/cuda/codec/intra/modes/build_intra_modes_test.sh
src/cuda/codec/intra/mrl/build_mrl_test.sh
src/cuda/codec/intra/ibp/build_cuda_test.sh
src/cuda/codec/intra/dip/build_gpu_test.sh
src/cuda/codec/intra/cfl/build_cfl_test.sh
src/cuda/codec/intra/palette/build_and_test.sh
src/cuda/codec/intra/ibc/build_and_test.sh
src/cuda/codec/intra/supplemental/build_gpu_test.sh
src/cuda/codec/intra/workspace/build_intra_workspace_test.sh
```

## Product use

`--still-gpu` calls `modes`, `mrl`, `dip`, `ibp`, `cfl` and `chroma` decision
stages directly.  Palette and IBC are used as **pixel** stages: the CPU
reference plans the palette colour set/direction (§5.20.8 analysis) and the IBC
block vector (§7.12.2 stack + encodability + exact-copy check), and the GPU
applies `palette_build_map`/`palette_predict` and `ibc_copy_block` for the
reconstruction.  `supplemental` and `workspace` remain component-only (no
product bitstream claim yet).

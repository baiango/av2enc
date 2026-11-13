# CUDA advanced-inter index

Entry point for the product video orchestrator (`src/cuda/pipeline/video/`).
Every module here is component code: it does not edit the pipeline, the root
CMake build, or another agent's files. All pointers in the public job structs
are device pointers and all device memory comes from one `av2enc::cuda::Arena`.

| Area | Module | Main entry points | Real sm_86 parity |
|---|---|---|---|
| Compound / CWP / masks, inter-intra, OBMC | `advanced/` | `compound_batch`, `compound_one`, `warp_batch`, `ext_warp_batch`, `bawp_batch`, `obmc_batch` | 1,319 checks |
| Compound (10-bit), OPFL, refine-MV, skip, §7.13.3.4 ref area | `compound/` | `blend_batch`, `skip_batch`, `opfl_batch`, `refine_search_batch`, `refine_predict_batch` | 17,263 checks |
| Block / extended warp, corner/warp-sample/least-squares models (8- and 10-bit) | `warp_stack/` | `Workspace::run_predict`, `predict_launch`, `corner_launch`, `sample_launch`, `estimate_launch` | 210 checks |
| Global motion estimation (§5.18.9, §7.12.2.2) | `gme/` | `estimate_blocks`, `estimate_warps`, `estimate_points`, `gme_global_mv_for_block` | 81,245 checks |
| Motion estimation (SAD/SATD, subpel, multi-ref) | `me/` | `me_search_batch`, `me_cost_batch`, `me_interp_batch` | 485 checks |
| Subpel MC (single block, 4:2:0, batch) | `mc/` | `mc_predict_plane`, `mc_predict_block_420`, `mc_predict_batch` | 1,069 checks |
| BAWP / morphological prediction (§7.13.3.25/.26) | `morph/` | `morph_predict_batch`, `morph_run_batch` | 36,940 checks |
| TIP motion field, plan and prediction (§7.10, §7.13.3.2) | `tip/` | `tip_setup_motion_field`, `tip_build_plan`, `tip_predict_plan`, `tip_predict_batch` | 5,759 checks |
| Temporal motion field, projection, ref-MV bank, §7.22 storage | `temporal/motion_field/` | `project`, `store_blocks`, `traj_fill`, `merge_stacks`, `apply_bank_ops` | 31,780 checks |
| Spatial MV stack, multi-reference pair stacks, merge candidates, DRL reorder | `mv_stack/` | `stack_build_batch`, `pair_build_batch`, `merge_set_batch`, `Workspace` | 4,150 checks |

## Areas with no pixel kernel (by design)

* **B-frames / temporal layers.** Scheduling, reference maps, slot refresh,
  order hints and the decoder-output simulation are frame-level serial work in
  `src/codec/inter/bframes/` (`build_sequence`, `validate_plan`) and
  `src/codec/inter/tlayers/` (`build_plan`). They are already linked into
  `av2enc_core`; a kernel would add latency without changing a sample. The B
  frame's pixel path is `mc_predict_batch` for each reference list plus the
  compound blend. `pair_build_batch` consumes the plan's past/future reference
  pair.
* **`save_subpu` (§7.13.3.15).** The kernel lives in
  `src/cuda/codec/filters/deblock/subpu/` (filter-module ownership, 77,986
  checks on sm_86). This directory consumes the resulting `MiInfo` grid rather
  than duplicating it.
* **Entropy, headers, syntax, CDFs, metadata.** CPU by spec; no kernel.

## Build fragments for the video target

The root `CMakeLists.txt` is not touched here. When the video target is wired,
add these kernel translation units (plus the existing `src/cuda/arena.cpp`) and
keep `-I src` on the include path:

```text
src/cuda/codec/inter/advanced/advanced.cu
src/cuda/codec/inter/compound/compound_full.cu
src/cuda/codec/inter/gme/gme.cu            src/cuda/codec/inter/gme/gme_plan.cpp
src/cuda/codec/inter/mc/mc.cu
src/cuda/codec/inter/me/me.cu
src/cuda/codec/inter/morph/morph.cu         src/cuda/codec/inter/morph/morph_workspace.cpp
src/cuda/codec/inter/tip/tip.cu
src/cuda/codec/inter/temporal/motion_field/motion_field.cu
src/cuda/codec/inter/mv_stack/mv_stack.cu
src/cuda/codec/inter/warp_stack/warp_stack.cu
```

`advanced.cu` and `advanced_test.cu` include `tables_redirect.h` before
`advanced.h`; that is what gives the shared §7.13.3.27/§9.5 tables device
storage. Do not include `advanced.h` from another nvcc TU without it.

## Verification commands

```sh
# per-module (run on the sm_86 box)
src/cuda/codec/inter/advanced/build_advanced_test.sh
src/cuda/codec/inter/compound/build_and_test.sh
src/cuda/codec/inter/gme/build_gpu_test.sh
src/cuda/codec/inter/mc/build_and_test.sh
src/cuda/codec/inter/me/build_and_test.sh
src/cuda/codec/inter/morph/build_gpu_test.sh
src/cuda/codec/inter/tip/build_gpu_test.sh
src/cuda/codec/inter/temporal/motion_field/build_gpu_test.sh
src/cuda/codec/inter/mv_stack/build_gpu_test.sh
src/cuda/codec/inter/warp_stack/build_gpu_test.sh
```

Host code is a correctness oracle only; `morph`, `tip`, `gme`, `mv_stack` also
ship no-GPU emulation tests.

# Pixel-kernel inventory — SAD / SAD+SSE / SATD / variance / block copy / intra edge filter

Base commit: `d5aeafe` (after the Phase A host migrations; branch
`exp/rf4-gaps`).  Line numbers move; re-grep before trusting one.

Scope: every host file+function that inlines one of the operations the layer
exposes, plus the CUDA/emulation mirrors that must be kept equal.  Phase A
moved the first host call sites onto the seam; this branch closes the four
remaining gaps:

1. `sad_sse` fuses the two-pass SAD+SSE in `intra::prediction_cost` and
   `dip::prediction_cost8` back into one pass (no value changes).
2. `sad_multi` is deleted: it never had a production caller (ME evaluates
   candidates sequentially and the centre moves between iterations).
3. The 16-bit format core's clipped SAD and variance become `sad_u16_clamped`
   and `variance_u16`.
4. The even-row SAD is a call-site migration (no slot); the fractional subpel
   SAD gets its own slot (`subpel_sad_rows`, §6.1) because the profile put it
   at 46-67% of a single-thread 240p encode.

Every slot in `KernelTable` has a test.  `sad_sse` is the exception on the
caller side: `intra::prediction_cost`/`predict_and_cost` are exercised by the
component test but have no production caller (the video intra search has its own
SSE loop), and `dip::prediction_cost8` only runs under `--video-dip`.  That is
why the NEON `sad_sse` pilot measured flat; see `docs/simd-plan.md`.

## The seam

`src/primitives/kernels.h` / `kernels.cpp`:

| Kernel | Scalar reference transcribed from | Notes |
|---|---|---|
| `sad` | `me_sad_rows` (integer path), `ibc::sad_block`, `rdo::sad_u8` | All three are byte-identical today; any of them is a valid oracle. |
| `sad_sse` | `intra::prediction_cost` + `dip::prediction_cost8` (the SAD kernel plus their inline SSE loop) | One pass, integer only. `int` product is exact for 8-bit (`|d| <= 255`), so the intra (`int`) and dip (`int64`) forms give the same values. The intra path has no production caller today; DIP is opt-in (see the note above). |
| `satd(..., SatdTile::k8x8)` | `me_hadamard8`/`me_satd_unit`, summed by `me_cost_plane(kMeSatd)` | 8x8 Hadamard, `(sum + 8) >> 4` per tile, zero-padded edge tiles. |
| `satd(..., SatdTile::k4x4)` | `rdo::hadamard4`/`satd_u8` | 4x4 Hadamard, unscaled, zero-padded edge tiles. **Not interchangeable** with the 8x8 variant. |
| `variance` | `block_variance_cost_at` (`video_frame.cpp`, `still_encoder.cpp`) | int64 sum -> binary64 mean -> binary64 SSE -> int64 truncation. Requires `-ffp-contract=off`. |
| `sad_u16_clamped` | `hbd::hbd_sad_at` (`pipeline/video/hbd/video_hbd_ops.h`) | 16-bit container, per-sample reference clip to `[0, ref_w/ref_h)`. |
| `variance_u16` | `hbd::hbd_variance_cost` (same header) | Same binary64 sequence as `variance`, `uint16_t` container. Requires `-ffp-contract=off`. |
| `block_copy` | `ibc::copy_plane` | Row-wise, forward; no overlap support. |
| `edge_filter` | `intra::intra_edge_filter` | §7.13.2.18, strengths 1..3, snapshot-then-write. `edge[0] == AboveRow[-1]`. |
| `subpel_sad_rows` | `me_sad_subpel_rows` -> `me_sample_at`/`me_interp_sample` | The fractional-MV ME cost: two-pass 8-tap filter, Round0/Round1, clip, then `|cur - pred|`. Keeps `r0/r1/rstep` for the CUDA warp shape. See §6.1. |
| `intra_edge_filter_strength` | `intra::edge_filter_strength` | §7.13.2.17; a branch table, not a dispatch slot. |

Dispatch: `kernels()` reads `AV2ENC_SIMD` (`auto` / `scalar` / `neon`; default
`auto`) once at start-up.  With no SIMD table built, all requests resolve to
`scalar`.  `select_kernels()` is the pure resolver used by tests; `intro()` is
the `--simd`-style line.  The NEON table landed with `sad_sse` as its first
(and, at the time of writing, only) overridden slot — see "Pilot:
`sad_sse` landed, measured flat" in `docs/simd-plan.md` for its evidence and
for why the measurement says not to continue in slot order.

Call sites key off `AV2ENC_PRIMITIVES`, which the product `av2enc_core` target
defines.  Component `build_and_test.sh` gates compile a single TU without
`kernels.cpp`, so every migrated site keeps its original inline loop under
`#else`.  That applies to the files this branch touched (intra modes, DIP,
IBC, format-video hbd header) and to the Phase A sites in `me.cpp`, `mc.cpp`,
`rdo.cpp` and `video_frame.cpp`.

## 1. SAD

| File:line | Function | Should map to | Migration note |
|---|---|---|---|
| `src/codec/inter/me/me_core.h:396` | `me::me_sad_rows` | `sad` | Generalised by `r0/r1/rstep` for the CUDA warp loop (`r0=lane, rstep=32`). The host ME batch (`me.cpp`) uses `sad`; the header keeps this function for the device mirror. A host call that still uses it passes `r0=0, r1=h, rstep=1`. |
| `src/codec/inter/me/me_core.h:420` | `me::me_sad_subpel_rows` | `subpel_sad_rows` | **Migrated** behind `AV2ENC_PRIMITIVES && !__CUDA_ARCH__`; the `#else` branch is the untouched per-sample oracle. Design history in §6.1. |
| `src/codec/inter/me/me_core.h:510` | `me::me_cost_plane` | `sad` + `satd` | Dispatcher: integer-MV SAD, subpel SAD, or per-8x8-tile SATD. |
| `src/codec/rate/rdo/rdo.cpp:179` | `rdo::sad_u8` | `sad` | Public `std::span` API; add the span checks at the wrapper, not in the kernel. |
| `src/codec/rate/rdo/rdo.cpp:201` | `rdo::satd_u8` | `satd(SatdTile::k4x4)` | 4x4, unscaled. |
| `src/codec/intra/ibc/core/ibc_core.h:186` | `ibc::sad_block` | `sad` | Used by IBC search; kept as the transcription source/oracle for the host/device header. |
| `src/codec/intra/ibc/core/ibc.cpp:373`, `:390` | `ibc` stack/full search | `sad` | Both call sites **migrated**; `ibc_gate` compiles the TU without `kernels.cpp`, so the `sad_block` fallback is retained. There is no `sad_multi` caller. |
| `src/codec/intra/modes/intra_modes.cpp:681` | `intra::prediction_cost` | `sad_sse` | **Migrated**: one fused pass replaces `sad()` + the inline SSE loop. Values unchanged (proven below). |
| `src/codec/intra/dip/dip.cpp:382` | `dip::prediction_cost8` | `sad_sse` | **Migrated**: same fused pass; its old SSE used `int64` products, value-identical for 8-bit. |
| `src/codec/inter/mc/mc.cpp:516` | `mc::sad_block` | `sad` | Merge-candidate scoring (Phase A batch). |
| `src/codec/inter/temporal/temporal_predict.h:190` | `temporal::refine_sad` | **call-site migration, no slot** | Exactly `sad(p0, 2*p0_stride, p1, 2*p1_stride, w, (h + 1) / 2)`. Verified on 289 shapes: the `(h+1)/2` form matches all 289; `h/2` mismatches all 153 odd-height cases (the loop visits rows `0,2,...` so the count is `ceil(h/2)`). Real leaf heights are even (4/8/16/32/64), where both agree, but the call site must use `(h+1)/2`. |
| `src/pipeline/video/hbd/video_hbd_ops.h:228` | `hbd::hbd_sad_at` | `sad_u16_clamped` | **Migrated** behind `AV2ENC_PRIMITIVES`; the clamp parameters `(x, y, mv_y, mv_x, ref_w, ref_h)` are part of the kernel signature. The header keeps the inline loop for CUDA device passes and component gates. |
| `src/pipeline/video/hbd/video_hbd_ops.h:hbd_me_search` | `hbd::hbd_me_search` | `sad_u16_clamped` via `hbd_sad_at` | The diamond search inherits the migration; its fractional stage is a separate SSE loop (not in this layer). |
| `src/cuda/codec/inter/me/me.cu:48`, `:51`, `:61` | `me_device_cost` | device mirror of `sad`/`satd` | Instantiates `me_core.h`; changes with the shared header, not with the host table. |
| `src/cuda/codec/inter/me_precision/me2_gpu_cost.cuh:48` | ME2 warp cost | device mirror of `satd` | Same shared 8x8 tiles. |

## 2. SATD

| File:line | Function | Should map to | Migration note |
|---|---|---|---|
| `src/codec/inter/me/me_core.h:440` | `me::me_hadamard8` | internal to `satd(k8x8)` | 1-D 8-point transform. |
| `src/codec/inter/me/me_core.h:453` | `me::me_hadamard8_2d` | internal to `satd(k8x8)` | Rows then columns. |
| `src/codec/inter/me/me_core.h:475` | `me::me_satd_unit` | `satd(k8x8)` | Per-tile `(sum + 8) >> 4`; edge tiles zero-padded. |
| `src/codec/inter/me/me_core.h:506` | `me::me_satd_unit_count` | not a kernel | Tile count only. |
| `src/codec/rate/rdo/rdo.cpp:30` | `rdo::hadamard4` | internal to `satd(k4x4)` | 1-D 4-point transform. |
| `src/codec/rate/rdo/rdo.cpp:201` | `rdo::satd_u8` | `satd(k4x4)` | Unscaled 4x4 path used by RDO/`DistortionMetric::kSatd`. |
| `src/codec/rate/rdo/rdo.cpp:256` | `rdo::compute_distortion` | `sad`/`satd` dispatch | `kSsd` is not in this layer; `kSad` and `kSatd` are. |
| `src/cuda/codec/inter/me/me.cu:61` | `me_device_cost` | device mirror of `satd(k8x8)` | |

Do not merge the two SATD definitions. The ME path's `(sum+8)>>4` rounding is
what keeps SATD in the SAD range; RDO's 4x4 path is intentionally unscaled.

## 3. Variance (partition leaf cost)

| File:line | Function | Should map to | Migration note |
|---|---|---|---|
| `src/pipeline/video/video_frame.cpp:2143` | `VideoFrame::block_variance_cost_at` | `variance` | Private member; the public kernel takes the visible rectangle's origin directly. Fused into `leaf` costs at `:2680`, `:3004`. |
| `src/pipeline/still/still_encoder.cpp:1517` | `StillEncoder::block_variance_cost_at` | `variance` | Same formula, `source_sample`-clipped tile coordinates. Call site `:470`. |
| `src/codec/transform/inter_residual/gate/video_residual_gate.cpp:264` | `block_variance_cost_at` | `variance` | Standalone-gate copy of the same loop. Migrating it is optional but keeps the gate honest. |
| `src/pipeline/video/hbd/video_hbd_ops.h:358` | `hbd::hbd_variance_cost` | `variance_u16` | **Migrated** behind `AV2ENC_PRIMITIVES`; the wrapper keeps its `(x, y)` signature and adds `y*stride + x` before calling the kernel. The inline body stays for CUDA device passes and component gates. |
| `src/pipeline/video/hbd/video_hbd.cpp:706` | `block_variance_cost` (hbd) | `variance_u16` via `hbd_variance_cost` | Thin wrapper; inherits the migration. |
| `src/cuda/pipeline/video/video_device.cuh:60` | `device::block_variance_cost` | `variance` | Host-side CUDA-backend mirror of `video_frame.cpp`; runs on the CPU in `video_backend_cpu.cpp:310`. This is the easiest CUDA-side consumer to migrate. |
| `src/cuda/pipeline/still/plan/plan.cpp:57` | `block_variance_cost_at` (plan) | `variance` | Host-side GPU-planner mirror; see also `plan_cost.h:10`. |
| `src/cuda/pipeline/still/plan/plan_cost.cu:13` | `variance_kernel` | device mirror of `variance` | One thread per rectangle, deliberately same order (`-fmad=false`). |

The variance contract is floating-point sequence, not just the formula. Any
migration must keep `-ffp-contract=off` on the translation unit, and a SIMD
implementation must reproduce the per-block double sum order or it will not be
bit-exact. `primitives_test` asserts both the transcribed oracle and
`hbd_variance_cost`.

## 4. Block copy

| File:line | Function | Should map to | Migration note |
|---|---|---|---|
| `src/codec/intra/ibc/core/ibc_core.h:173` | `ibc::copy_plane` | `block_copy` | The transcription source. |
| `src/codec/intra/ibc/core/ibc.cpp:203` | `ibc::copy_block` | `block_copy` | Whole block: luma plus two chroma planes, each via `copy_plane` (`:228`, `:249`). Migrate the per-plane loop, not the plane selection. |
| `src/pipeline/video/post/video_loop_filters.cpp:97` | `copy_plane` | `block_copy` | Post-filter scratch/plane copies (`:363`, `:1375`, `:1610`, `:1733`, `:1780`). Uses `std::copy_n` per row; same result. |
| `src/codec/inter/advanced/video/advanced_decision.cpp:14` | `adv::copy_rect` | `block_copy` | Adds `x/y` to both pointers first, then copies `w*h`. Call sites `:341`, `:404`. |
| `src/codec/filters/cdef/cdef.cpp:83` | `cdef::copy_block` | `block_copy` | Copies the block back from the CDEF scratch frame; call site `:197`. |
| `src/codec/filters/loop_restoration/wiener_core.h:503` | `lr_copy_plane_t` | `block_copy` | Plane-sized copy, used by `restoration.cpp:718`, `:738`, `:886`. |
| `src/cuda/codec/intra/ibc/ibc.cu:245` | `cuda::ibc_copy_block` | device mirror of `block_copy` | |
| `src/cuda/codec/filters/loop_restoration/chroma/lr_chroma_device.cuh:276` | `lrc_copy_block` | device mirror of `block_copy` | |
| `src/cuda/formats/chroma/chroma_formats.cu:466` | `Workspace::copy_planes` | device mirror of `block_copy` | Batch plane copies for 4:2:2 / 4:4:4 / 4:0:0. |

## 5. Intra edge filter / strength (§7.13.2.17–.18)

| File:line | Function | Should map to | Migration note |
|---|---|---|---|
| `src/codec/intra/modes/intra_modes.cpp:336` | `intra::edge_filter_strength` | `intra_edge_filter_strength` | §7.13.2.17. `delta` is `p_angle - 90` / `p_angle - 180`; `smooth_neighbors` is the OR of the existing-neighbour smooth flags. |
| `src/codec/intra/modes/intra_modes.cpp:401` | `intra::intra_edge_filter` | `edge_filter` | §7.13.2.18. Snapshot `edge[0..size-1]`, write `edge[1..size-1]`. |
| `src/codec/intra/modes/intra_modes.cpp:714` | `intra::predict_intra` (directional branch) | consume both above | Computes the corner, `num_px`, strengths (`:727`, `:734`) and calls the filter (`:730`, `:737`). This is the first host call site to migrate. |
| `src/codec/intra/modes/intra_modes.cpp:795` | `intra::predict_and_cost` | `sad` + edge filter | Predicts then calls `prediction_cost` (`:802`). |
| `src/cuda/codec/intra/modes/intra_modes.h:234` | `cuda::intra_modes::intra_edge_strength` | `intra_edge_filter_strength` | Constexpr device mirror of §7.13.2.17. |
| `src/cuda/codec/intra/modes/intra_predict_device.cuh:209` | `im_compute_filter_params` | strength + call-site orchestration | Device mirror of `intra_modes.cpp:714`'s strength/corner/numPx logic. |
| `src/cuda/codec/intra/modes/intra_predict_device.cuh:257` | `im_edge_filter_pass` | `edge_filter` | Device mirror of §7.13.2.18, one filtered position per thread. |
| `src/cuda/codec/intra/modes/intra_predict_device.cuh:460` | `intra_run_block_device` | calls `im_edge_filter_pass` | Applies above then left, same order as the CPU path. |
| `src/cuda/codec/intra/mrl/mrl_device.cuh:284`, `:291` | MRL prediction | `intra_edge_filter_strength` | Second device consumer of the strength table. |
| `src/cuda/codec/intra/ibp/ibp_predict_device.cuh:264`, `:273` | IBP prediction | `intra_edge_filter_strength` | Third device consumer. |

The "4-tap smoothing" wording in the task refers to the directional intra
smoothing stage: the five-tap kernels are `{0,4,8,4,0}`, `{0,5,6,5,0}` and
`{2,4,4,4,2}` for strengths 1..3 (strength 1/2 have a zero outer tap). The
kernel therefore transcribes the normative five-tap loop and the
strength-selection table exactly; nothing was simplified to four taps.

## 6. Not in this layer (deliberately)

| Operation | Where | Why not now |
|---|---|---|
| Standalone SSE / SSD sums | `rdo::ssd_u8`, CDEF/PC-Wiener searches | A separate kernel; only the SAD+SSE pair that two call sites already computed together is fused. |
| Subpel interpolation | `me_core.h:me_sample_at`, `me_interp_sample`, `mc_core.h` | Normative two-pass filter; not a sum kernel. See §6.1 for the SAD design note. |
| 16-bit SATD / block copy | `pipeline/video/hbd/video_hbd_ops.h` | Only the clipped SAD and variance have live 16-bit consumers (format video core, 10-bit path). Add the rest when a caller exists. |
| Warp-cooperative SAD | `me_core.h:me_sad_rows` (`rstep=32`) | NEON/AVX2 host tables do not cover the CUDA warp shape. |
| Warp/morph/compound SSE | `advanced_decision.cpp`, `compound_*`, `morph_*` | Decision costs, not one of the layer's kernels. |

### 6.1 Subpel SAD: the first kernel driven by the profile

`me_core.h:me_sad_subpel_rows` (the cost used by `me_cost_plane` for fractional
MVs) interpolates every reference sample through `me_sample_at`, then
accumulates `|cur - pred|`.  `/usr/bin/sample` put that one function at 46-67%
of a single-thread 240p encode and 81.6% of a 720p encode, while the whole
`KernelTable` was under 1%, so this was the first slot driven by a profile
rather than by slot order (`docs/simd-plan.md`).

What landed is option (b) of the earlier design note, in two steps:

1. **Scalar block kernel, migrated.** `subpel_sad_rows(cur, ..., ref_w, ref_h,
   x, y, w, mv, filter, r0, r1, rstep)` is a verbatim transcription of
   `me_sad_subpel_rows` -> `me_sample_at` -> `me_interp_sample`, including
   the integer-MV shortcut (which reads the reference *without* clamping), the
   per-tap clamp in both axes, `(sum + 4) >> 3`, `(sum + 1024) >> 11` and the
   final clip to `[0, 255]`.  `r0/r1/rstep` are kept because the CUDA warp cost
   owns a row subset; the CPU path uses `r0 = 0, rstep = 1`.
2. **NEON table override.** See "Subpel SAD on NEON" in `docs/simd-plan.md` for
   the row-reuse design, the interior/border split and the measured result.

The oracle contract is unchanged: the header keeps the per-sample loop under
`#else`, and `tests/primitives_test.cpp` compares the kernel against it over
every filter, every fractional phase (both signs of the integer part), every
block size in the encoder's set, positions that clamp on all four edges,
padded/offset planes, and `rstep` 1, 2 and 32.

The `codec -> primitives` direction is unchanged: the primitive layer does not
include `me_core.h`.  The 6x16x8 tap table is copied into
`src/primitives/subpel_filters.h`, and the test asserts all 768 entries against
`me::kMeSubpelFilters` before comparing any output.

## 7. Migration status

1. Host intra path: **done** (edge filter/strength, `prediction_cost`, DIP,
   IBC). Phase A left the SAD and SSE as two passes; `sad_sse` restores one.
2. Host ME: **done** (`me_cost_plane`, `me.cpp`), including the fractional
   subpel SAD (`subpel_sad_rows`, §6.1).
3. Host RDO: **done** (`rdo::sad_u8` / `satd_u8`).
4. Partition variance: host 8-bit copies are done; the hbd mirror is done via
   `variance_u16`. The CUDA-side mirrors are unchanged.
5. Block copy: IBC done; the loop-filter/CDEF/restoration call sites remain
   for the NEON batch.
6. NEON v1 **started**: `sad_sse` is overridden in the NEON table (see
   `docs/simd-plan.md` for the pilot evidence and its flat measurement).
   `primitives_test` is the acceptance gate: it compares every kernel against
   the inlined implementation, exactly, and runs the whole sweep once per
   compiled-in table.  A run now covers 17,988 individual comparisons: 1,602
   fused SAD+SSE blocks over both tables (six oracle comparisons each plus 801
   direct scalar-vs-NEON pairs), 512 end-to-end
   `prediction_cost`/`prediction_cost8` call-site blocks, 3,920 clipped 16-bit
   SAD blocks (four MVs per shape plus the fully-clamped corner case) over both
   tables, 784 16-bit variance rectangles over both tables, and the 8-bit
   SAD/SATD/variance/copy/edge cases over both tables.

## 8. Gates

```
cmake --build build-t -j1
ctest --test-dir build-t -R gate_primitives --output-on-failure
ctest --test-dir build-t -LE '^gate$'          # stays 10/10
tools/refactor/verify_baseline.sh --check      # additive gate_ test tolerated
tools/gate_video_nightly.sh
src/pipeline/video/gate_video_qp_lambda.sh
src/formats/format_gate.sh                     # 10-bit 4:2:0 still path
```

Pinned streams must remain byte-identical; the fused `sad_sse` and the 16-bit
kernels are integer transcriptions, so that is expected, not incidental.  Any
move means a transcription bug (the test pins both outputs against the old
loops).

Component `build_and_test.sh` gates compile the migrated TUs without
`kernels.cpp` and without `AV2ENC_PRIMITIVES`; every migrated site keeps the
original loop in the `#else` branch, and those were checked to compile and
link standalone (`intra_modes.cpp`, `dip.cpp`, `ibc.cpp`, a hbd TU calling both
16-bit wrappers).

## 9. Not a `KernelTable` slot: the RDOQ model's all-zero table

`rdoq::av2::Model::init` (`src/codec/rate/rdoq/rdoq_av2_model.cpp`) was the
single largest function in the `--nightly-b` profile (14–19 % of work). 93 % of
its calls are on an all-zero block, whose per-position context table is a pure
function of `(w, h, plane)`; it is now built once per geometry and shared
lock-free. See "RDOQ `Model::init`: the all-zero table" in `docs/simd-plan.md`
for the measurement, the −15.7 % result and the exactness evidence.

It is deliberately **not** a `KernelTable` slot: nothing is vectorised, the win
is eliminating a recomputation, and the remaining per-block gather is ~1.2 % of
work. The row is recorded here so the next profile pass does not re-open it as
an unclaimed pixel kernel.

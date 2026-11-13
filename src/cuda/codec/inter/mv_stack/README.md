# MV stack / merge / DRL — CUDA kernels

CUDA side of the AV2 v1.0.0 inter block-mode stack:

| Clause | Process | Entry point |
|---|---|---|
| §7.12.2 (steps 6–22) | spatial MV stack scan, weights, derived candidates, global-MV candidate, §7.12.2.20 MVP combinations, MV clamps | `stack_build_batch` |
| §7.12.2.16 | single add derived process (non-temporal) | inside `stack_build_batch` |
| §7.12.2.14 | compound add derived process (non-temporal) | inside `stack_build_batch` |
| §7.12.2.19 | DRL reordering (one move-to-front swap per step) | inside `stack_build_batch` (`reorder` field) |
| multi-reference | two single-list stacks + the two-list `skip_mode` stack | `pair_build_batch` |
| multi-reference merge | deterministic compound merge candidate set (single-index + cross-list pairs, de-duplicated) | `merge_set_batch` |

The CPU reference is `src/codec/inter/merge/` and
`src/codec/inter/multiref/mvstack_multi.cpp`. This module is the device
counterpart the video orchestrator needs when it moves the block mode decision
off the host: one launch resolves the stack for every block in a frame.

**Not in this module** (by design):

* B-frame / temporal-layer scheduling and reference-slot bookkeeping is serial
  frame-level work (`src/codec/inter/bframes/`, `.../tlayers/`,
  `.../random_access/`). The pixel side of a B frame is `mc` + compound, both
  already device-resident. There is no per-pixel kernel to add.
* `save_subpu` (§7.13.3.15) lives in `src/cuda/codec/filters/deblock/subpu/`
  and is owned by the filter module; this component consumes the resulting MI
  grid (`MiInfo`) instead of duplicating it.
* Temporal candidates, the ref-MV bank and §7.22 motion-field storage are in
  `src/cuda/codec/inter/temporal/motion_field/`.
* TIP stacks are in `src/cuda/codec/inter/tip/`.

## Files

```text
src/cuda/codec/inter/mv_stack/
  mv_stack.h                     public API, POD jobs/results, Arena workspace
  mv_stack_device.cuh            shared host/device transcription of the stack core
  mv_stack.cu                    kernels + launchers + workspace
  mv_stack_emulation_test.cpp    host parity test (no GPU/nvcc)
  mv_stack_test.cu               real sm_86 parity + arena test
  build_and_test.sh              GPU when nvcc is present, else emulation
  build_emulation_test.sh
  build_gpu_test.sh
```

`mv_stack_device.cuh` is the single transcription. nvcc compiles it into the
kernels; the host compiler compiles it into the emulation test. Every helper is
`AV2ENC_MVS_HD`, `§7.9.4 Div_Mult` has device storage, and the CPU types
(`merge::StackParams`, `merge::MvStack`, `merge::StackCandidate`,
`multiref::RefStackPair`) are reused directly so there is no layout mirror to
drift.

## API

```cpp
namespace av2enc::cuda::mvstack {

using StackJob = video::merge::StackParams;   // device mi / frame_distance

struct StackResult { video::merge::MvStack stack; int status; };  // status = merge::Status
using PairJob = MvsPairJob;                    // params + ref0/ref1
using PairStacks = MvsPairStacks;              // list[2] + compound + flags
using MergeSet = MvsMergeSet;                  // <= 40 fixed candidates

cudaError_t stack_build_batch(const StackJob*, StackResult*, int count, cudaStream_t = nullptr);
cudaError_t pair_build_batch(const PairJob*, PairStacks*, int count, cudaStream_t = nullptr);
cudaError_t merge_set_batch(const PairStacks*, MergeSet*, int count, cudaStream_t = nullptr);

std::size_t required_device_bytes(const MvStackConfig&);

class Workspace {                              // persistent arena tables
  Status create(Arena&, const MvStackConfig&, std::string* = nullptr,
                std::size_t* out_required = nullptr);
  Status upload_stack_jobs(const StackJob*, int, cudaStream_t = nullptr);
  Status run_stacks(int, cudaStream_t = nullptr);
  Status download_stacks(StackResult*, int, cudaStream_t = nullptr);
  // pair_jobs / run_pairs / download_pairs, run_merge_sets / download_merge_sets
};
}
```

* One thread per job (`kThreads = 128`, 1D grid); no shared memory, no
  barriers, no per-job allocation.
* `StackJob.params.mi` and `frame_distance` must be **device** pointers. The
  `StackResult.status` is `merge::Status` as an int; an invalid job writes a
  non-zero status and a zeroed stack.
* Launchers are asynchronous. Only `download_*` synchronizes the stream (it must
  return host data).
* `pair_build_batch` builds list 0, list 1, then the compound stack exactly like
  `multiref::build_pair_stacks`; `merge_set_batch` reproduces
  `build_compound_merge_candidates` including the insertion order and the
  MV-pair de-duplication.

## Arena contract

Every device byte comes from the caller's single `av2enc::cuda::Arena`. There is
no `cudaMalloc`, `cudaFree` or `cudaMallocHost` in the component.

`required_device_bytes({max_jobs = N})` is the exact persistent size:
`N * (StackJob + StackResult + PairJob + PairStacks + MergeSet)`, each array
rounded to the arena's 256-byte alignment. Measured:

| `max_jobs` | bytes |
|---:|---:|
| 8 | 18,944 |
| 48 | 111,104 |
| 64 | 147,456 |

`Workspace::create()` bump-allocates once. A short arena returns
`Status::kArenaExhausted`, writes the exact requirement to `out_required`, and
leaves the bump pointer untouched. Launchers never grow the arena; the GPU test
asserts that the watermark and `peak_used()` stay flat across repeated runs.

## Tests

### Host emulation (`./build_emulation_test.sh`, no GPU)

Compiles the shared device header with the host compiler and links the real
`merge.cpp` / `mvstack_multi.cpp`. Coverage:

* 1,500 randomized grids × single/compound jobs × all three `DrlReorder`
  modes × 1–4 references × `max_drl_bits_minus_1` 0–4; every `MvStack` field
  compared;
* 10 invalid parameter classes compared by status (null grid, bad block size,
  non-power-of-two `sb_h4`, out-of-range MI origin, Intrabc, bad DRL cap, TIP
  reference, compound without list 1, single with list 1);
* 800 multi-reference pair-stack + merge-set cases;
* DRL sort observability (a reordered stack is actually produced).

Result on the development host:

```
mv_stack_emulation_test: 13022 checks, 0 failures
```

### Real GPU (`./build_gpu_test.sh`, sm_86)

One `Arena`; random MI grids uploaded once per case. 400 raw `stack_build_batch`
cases, 48 persistent-workspace cases, 48 pair/merge cases, 4 invalid jobs, and
the one-byte-short arena check. Every stack field, status, merge candidate and
arena invariant is asserted.

```
device: NVIDIA GeForce RTX 3060 (sm_86, 28 SMs)
required_device_bytes(48) = 111104
mv_stack_cuda_test: 4150 checks, 0 failures
arena: used=222208 peak=579840 capacity=67256576
```

`compute-sanitizer --tool memcheck` reports `0 errors`.

## Integration notes for the video orchestrator

1. Upload the frame's `MiInfo` grid once (the same grid the encoder already
   fills); it is read-only during the launch.
2. Fill `StackJob`/`PairJob` descriptors for the frame's blocks, upload once,
   call `run_stacks`/`run_pairs` once, and download only if the decision stays
   on the host. A future all-device decision consumes `StackResult` in place.
3. `merge_set_batch` chains directly off the `pair_results()` device buffer, so
   multi-reference candidate scoring needs no host round trip.
4. The two reference lists of a B frame use `pair_build_batch`; the bi-predicted
   prediction itself is `mc_predict_batch` (two lists) + the compound blend.
5. Keep `max_jobs` at the frame's block count capacity; growing means
   re-initialising the arena at a frame boundary.

## Rules followed

* Integer-only arithmetic; no floating point and no `-fmad` sensitivity.
* One transcription shared by device and emulation; CPU reference is the oracle.
* Arena-only allocation; no direct allocator in the component.
* No Blackwell-only instructions (no TMA, clusters, `wgmma`/`tcgen05`, FP4).
* No root `CMakeLists.txt`, `src/main.cpp`, pipeline or other-agent changes.

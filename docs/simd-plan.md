# SIMD plan — NEON on macOS as a first-class target

Status: **binding decision (2026-11-16).** NEON on Apple silicon is a real
build target for this repo, not a portability chore.

## Decision and why

The production encoder is GPU-resident, and `AGENTS.md` still describes the CPU
path as a correctness oracle. That remains true for *shipping*. But development
happens on an M4 Mac, where every gate run, every QP sweep and every RD
experiment is CPU-bound. A fast local CPU path is therefore a **development
velocity** feature: it makes the measurement loop shorter, not the product
faster.

Consequences:

- CPU encode time is a target on this host, but the KPI stays **quality at
  matched rate**. Speed never justifies a stream change.
- The 47-gate suite plus the three pinned streams become the SIMD correctness
  harness for free — that is the main reason the kernel layer was extracted
  first.
- `AVX2` slots stay documented but unimplemented: there is no x86 development
  host here. `NEON` is implemented first.

## The seam (already merged)

`src/primitives/kernels.h` exposes one `KernelTable` selected once at start-up:

| Slot | Role |
|---|---|
| `sad` | integer-pel SAD, full block |
| `sad_sse` | fused SAD + SSE, one pass (intra RDO) |
| `satd` | Hadamard SATD, `k4x4` (RDO) and `k8x8` (ME) |
| `variance` | partition block variance |
| `variance_u16` | 10-bit partition block variance |
| `sad_u16_clamped` | 10-bit ME SAD with reference border clamp |
| `block_copy` | IBC plane copy |
| `edge_filter` | §7.13.2.18 intra edge filter |

Selection: `AV2ENC_SIMD=auto|scalar|neon`. Default `auto`; `neon` is only
accepted when the binary contains a NEON table (`__ARM_NEON`), otherwise it
falls back to `scalar` with a one-line notice. `scalar` must remain byte-
identical to today's pins forever — it is the oracle.

## Exactness contract

Every kernel is **integer**. Integer arithmetic is associative, so any
vectorised order returns the identical value; that is why this layer can be
optimised without re-deriving decisions.

One explicit exception to vectorise carefully:

- `variance` returns an `int64_t` computed as `int64 sum` → `double mean` →
  `double` SSE → cast. The integer sum may be vectorised (exact); the
  `double` tail must stay scalar, in the same order of operations, on the same
  values. Changing the reduction order of the `double` tail changes the
  returned integer for a non-negligible fraction of blocks and therefore moves
  a pin. `-ffp-contract=off` already applies to `av2enc_core`; do not add
  fast-math to any TU in this layer.

## Order of work

1. **`sad_sse`** — the hottest intra cost path (two cost functions per
   prediction, every mode, every block). Highest value per line.
2. **`sad`** — ME inner loop and merge-candidate scoring.
3. **`satd`** (`k4x4`, `k8x8`) — ME metric and RDO tie-breaking input.
4. **`variance`** — partition recursion at every node.
5. **`edge_filter`** — directional intra; 129-entry snapshot, mode-dependent
   strengths.
6. **`variance_u16` / `sad_u16_clamped`** — 10-bit path; lower priority, higher
   arithmetic complexity (border clamp must be reproduced exactly).
7. **`block_copy`** — IBC only; low value, do last or skip.

Each kernel lands as its own commit with its own test evidence. No kernel lands
with a pin change.

**Measured 2026-11-16:** the ordering below is not what the profile says.  See
"Pilot: `sad_sse` landed, measured flat" at the end of this document before
starting slot 2.

## Profile: the table is not where the time is (2026-11-16)

`/usr/bin/sample`, 10 s, 240p QP106 nightly, `--cpu-threads 1`, 7,676 leaf
samples.  An independent 1 ms-interval run by the pilot agent put the same
function at ~46%; treat it as **46–67% depending on attribution**, still
dominant by an order of magnitude:

| Share | Function | In the table? |
|---:|---|---|
| **66.8%** | `me::me_interp_sample` — 8-tap separable subpel filter, per sample | **no** |
| 6.4% | `txv::forward_block` (`double*`) | no |
| 4.7% | `txv::inverse_block` | no |
| 3.5% | `cdef::cdef_filter_sample_t` | no |
| 2.9% | `intra::predict_directional` | no |
| 1.9% | `rdoq::av2::Model::init` | no |
| 1.8% | `me::me_search_batch` | no (calls the table) |
| 1.5% | `mc::predict_plane` | no |
| ~0.3% | `primitives::scalar_sad` (all table slots) | yes |

Three consequences, all measured rather than argued:

1. **`sad_sse` has no live caller.** A counter in the scalar kernel recorded
   **0 calls** over a full 60-frame encode. The live intra cost is
   `video_intra_search.h::detail::sse_against_source` (SSE-only, no SAD);
   `intra::prediction_cost`/`predict_and_cost` are test-only.
2. **The whole table is under 1%.** Even a free table would not move the
   encode. Slots 2–8 of the original order are therefore **not** the next
   work.
3. **`AV2ENC_NATIVE=ON` is a no-op here** (17.12 s vs 17.10 s single-thread):
   Apple clang already targets the host CPU on arm64, so there is no free
   compiler lever.

### Revised target: subpel ME, not the table

The hot loop is the ME interpolation: an 8-tap horizontal pass, `Round2`, an
8-tap vertical pass, `Round2`, clip to 8 bits — evaluated per sample, per
candidate MV, with per-tap border clamping. It must become a **block** kernel:

1. scalar `subpel_sad_rows` transcribed verbatim from
   `me_sad_subpel_rows`/`me_interp_sample` (keeping `r0/r1/rstep`, since the
   CUDA warp path owns a row subset), tests against the existing per-sample
   oracle over random blocks, all filters, all 16 fractional phases and
   clamped edges; **done**
2. migrate the CPU path to it, byte-identical; **done**
3. NEON: horizontal filter 8–16 samples at a time into an `int32` row, then the
   vertical pass + SAD; borders either scalar or a small clamped-load path;
   **done**
4. measure against the 17.1 s single-thread baseline. **done: 17.4 s -> 6.11 s
   (2.9x) on the same binary pair; see the last section.**

A 4× kernel win here is worth roughly a third of total encode time — two
orders of magnitude more than finishing the table.

The table work is **not** wasted: `sad_sse` is the tested template for dispatch,
partial tables and the dual-table harness, and `sad`/`variance` remain useful if
profiling ever puts decision cost on top. They are just not the bottleneck now.

## Result: subpel ME block kernel (landed 2026-11-16)

Two commits, both bit-exact:

| Stage | Change | Same-binary effect |
|---|---|---|
| 1 | scalar `subpel_sad_rows` block kernel, CPU path migrated, inline oracle kept for device/no-archive builds | 1.28× on its own — the 8-tap body no longer inlines into the search loop |
| 2 | NEON: 8 taps × 8 samples horizontally with a row ring (each reference row filtered once), fused vertical + round + clip + SAD | **2.20×** at 240p 1T, 1.41× at 8T, **2.35×** at 720p 8T |

Merged, against the baseline recorded before the NEON phase (threading arrived
later; the 8-thread figures here improved again with phases B and C, see
`docs/wavefront-plan.md` §10):

| Workload | Before | After | Speedup |
|---|---:|---:|---:|
| 240p QP106, 1 thread | 16.55 s | **6.03 s** | **2.74×** |
| 240p QP106, 8 threads | 7.97 s | 5.22 s | 1.53× |
| 720p QP217, 8 threads | 148.0 s | 45.15 s | 3.3× |

Profile after the change: the subpel kernel drops from **55.9% to 8.1%** of the
single-thread encode; nothing else is above ~6% (transform, intra, RDOQ, CDEF
each at a few percent). No second hotspot exists.

Correctness: 10,647 cases per table, 3.1 M samples per table, 21,294 oracle
comparisons, exact; a deliberate 1-LSB rounding change fails 5,751 checks, so
the suite is sensitive to the thing it is meant to catch. `AV2ENC_SIMD=scalar`
vs `=neon` outputs are `cmp`-identical at 240p and 720p, and the 720p stream is
byte-identical to the pre-refactor encode (`c4ba2813…`, 36,572 B).

Dev loop: the 49-gate suite drops from **1,358 s to 873 s of compute (−36%)**.
It does not scale like the encoder because the suite is dominated by process
startup, decoding and non-encode checks (`gate_video_tcq` 146 s,
`gate_loop_restoration` 134 s).

The suite is also process-parallel: `tools/run_gates.sh --gate-jobs 6` runs six
gates at a time (each has its own `mktemp` workspace and no shared paths) and
takes **273 s wall** instead of ~900 s serial, with the same 49/49. The default
stays 1 so timing runs are not contaminated; the encoder itself is untouched,
so this cannot move a pin.

**Recommendation: stop SIMD work here.** The gain is 2.7× on the dev loop, the
hotspot moved, and further slots would optimise a few percent. The remaining
quality problem is algorithmic (the 720p RD gap), not arithmetic throughput.

## Implementation notes

- AArch64 NEON is baseline; no `-mfpu`/`-march` flags are needed for intrinsics.
  Keep `AV2ENC_NATIVE=OFF` as the default so binaries stay portable.
- Do not use `dotprod`/`i8mm` initially. `SAD` is `|a-b|` accumulation, not a
  multiply-accumulate, and `SATD` is adds/shifts — baseline NEON covers them.
  Adding an `i8mm` path later is a separate measured commit with runtime
  detection.
- One new TU, `src/primitives/kernels_neon.cpp`, guarded by `__ARM_NEON &&
  __aarch64__`, filling a second `KernelTable`. No `#ifdef` sprinkling in the
  scalar file, no per-call-site architecture checks.
- The scalar table stays compiled on every platform: it is the fallback, the
  oracle and the only implementation the tests can compare against.

## Validation (per kernel, non-negotiable)

1. `gate_primitives`: exact equality against the scalar table over the existing
   shape/stride sweep (256 shapes × 2 passes, unaligned pointers, odd widths).
2. Pins byte-identical: default `25,009 a1d0dba1`, quality `25,141 3984677b`,
   nightly `23,690 1b99127a`, explicit-λ control `24,113 7f03eb6e`.
3. Full suite: 49/49 gates, including `gate_tile_flush`, `gate_video_ext_nightly`,
   `gate_build_seam` and `gate_primitives`.
4. Decode check: `dav2d --strict 1` picture counts unchanged.

Any pin movement means the kernel is wrong; revert the kernel, never re-pin.

## Measuring the gain (so it is not folklore)

Encode wall time only; quality is already fixed by byte-identical output.

| Workload | Command shape | Why |
|---|---|---|
| 240p 60f | `--qp 106 --fast-decode --cpu-threads 1` and `8` | the RD search loop, single vs parallel |
| 720p 60f | `--qp 217 --fast-decode --video-key-qp-offset -57 --video-leaf-rd-lambda 250 --video-ext-partitions` | real detail, filter-heavy |
| gate suite | `tools/run_gates.sh` | the actual dev loop |

Protocol: 3 runs each, median, `AV2ENC_SIMD=scalar` vs `neon`, same binary.
Report `md5` of the output beside the timing — a speedup with a different md5
is a bug, not a result.

Scalar baseline, 2026-11-16, M4 (idle machine, medians of 3):

| Workload | Time | Output |
|---|---:|---|
| 240p QP106 nightly, `--cpu-threads 1` | 16.55 s | 25,008 B `0639b436…` |
| 240p QP106 nightly, `--cpu-threads 8` | 7.97 s | 25,008 B `0639b436…` |
| gate suite (49 gates) | 1,358 s compute | 49/49 green |

The single-thread 240p number is the honest NEON signal: at `-j8` the encode is
already parallel, so kernel speedups partly disappear into scheduling.

## Sequencing against the refactor

NEON now comes **before** Phase B (monster-file splits), because it is the
development-velocity lever while the splits are hygiene:

1. A.5 (running): fused `sad_sse`, 16-bit slots, build seam, dead-code removal.
2. **NEON v1**: the eight slots above, in order, each independently gated.
3. Re-measure the dev loop; publish the speedup with md5s attached.
4. Phase B (split `video_frame.cpp`, `main.cpp`, `video_syntax.cpp`,
   `still_encoder.cpp`) — now on a base where the hot loop boundary is already
   the kernel seam, so the splits have a natural cut line.

## What this does not fix

NEON shortens the loop; it does not improve rate-distortion. The 240p gap to
SVT preset 0 (0.097 JOD / 6.7 K points) and the 720p gap (~0.9 JOD) remain
algorithmic — compound/multi-reference prediction in the 720p case — and are
tracked separately.

## Pilot: `sad_sse` landed, measured flat (2026-11-16)

The first slot is implemented and proven bit-identical, but its effect on the
240p loop is **zero within noise**. That is not a kernel problem: the slot has
no production caller on the default video path.

### What landed

- `src/primitives/kernels_neon.cpp`: `neon_sad_sse`, 16 samples per iteration
  (`vabdq_u8`, then `vpaddlq_u8` -> `vpadalq_u16` for SAD and `vmulq_u16` ->
  `vpadalq_u16` for SSE, both into 4x uint32 lanes).  The `w % 16` remainder and
  `w < 16` take a scalar tail and never over-read a row.
- The NEON table is a copy of `scalar_kernels()` with only `sad_sse` overridden,
  so every unimplemented slot stays scalar rather than null.
- `AV2ENC_PRIMITIVES_HAVE_NEON` comes from `__ARM_NEON && __aarch64__` in
  `kernels.h` (compiler property, no build flag), matching the guard on the TU.
- `AV2ENC_SIMD=neon` on a build without the table resolves to scalar with one
  stderr notice from the start-up snapshot; `scalar` always wins.

### Correctness

`gate_primitives` runs the whole sweep twice, once per compiled-in table, and
compares both `sad_sse` outputs directly against scalar: **17,988
comparisons, 801 scalar-vs-NEON pairs, 1,602 `sad_sse` blocks**, all exact.
Shapes cover `w,h = 1..17` (every `w % 16` remainder and every scalar-tail
width), unaligned pointer offsets, padded strides, `w = h = 1`, and the
null/zero-size degenerate cases.  Pins byte-identical: 25,009 `a1d0dba1`,
25,141 `3984677b`, 23,690 `1b99127a`, 24,113 `7f03eb6e`; the
`AV2ENC_SIMD=scalar` and `=neon` IVFs are `cmp`-identical, not just same-md5.

### Encode time (M4, idle, medians of 3)

`nasa_480x240_30fps_60f.y4m --video --frames 60 --qp 106 --fast-decode`:

| Threads | scalar | neon | delta | Output |
|---|---:|---:|---:|---|
| 1 | 17.10 s | 17.13 s | -0.2% | 25,008 B `0639b436…` |
| 8 | 8.19 s | 7.89 s | +3.7%* | 25,008 B `0639b436…` |

\* Noise, not a speedup: the 8-thread ranges overlap (scalar 7.73-8.63, neon
7.72-8.65), and CPU-time medians are unchanged (1T 15.39 vs 15.56 s user, 8T
21.34 vs 20.75 s user).  The scalar md5 and byte count reproduced exactly;
wall time on this host today is ~3% above the 16.55 s recorded above, which is
why A/B runs are interleaved rather than compared to a day-old number.

### Why it is flat: the slot is not called

A temporary counter in the scalar kernel recorded **0 `sad_sse` calls** for the
whole 60-frame 240p encode (and for a still encode).  `intra::prediction_cost`
is only reached from `predict_and_cost`, which the video path does not use —
`video_intra_search.h:detail::sse_against_source` is the live intra cost loop.
`dip::prediction_cost8` runs only under `--video-dip`; that workload also
measured flat (16.91 s scalar vs 16.85 s neon, identical md5).

`sample` (1 ms) on the 1-thread encode puts ~46% of the time in
`me::me_interp_sample` (subpel ME interpolation, deliberately not a slot), and
under 1% in *all* table functions together.

### Per-call kernel speed (L1-resident microbenchmark, medians of 3)

| Block | 16x16 | 16x8 | 17x17 | 32x16 | 32x32 | 48x48 | 64x64 | 8x16 | 8x8 | 4x4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Speedup | 6.2x | 5.5x | 5.0x | 3.5x | 3.1x | 2.0x | 1.5x | 1.0x | 0.91x | 0.97x |

The scalar loop is already auto-vectorised by clang, so the win is concentrated
where its per-row prologue dominates (16..32-wide).  8-wide widths never enter
the 16-wide loop and 64x64 is only 1.5x because the handwritten loop is not
unrolled; an 8-wide step and 2x unrolling are cheap follow-ups if those shapes
ever matter.

### Verdict for the rest of the phase

1. Do not expect the remaining seven slots to move the 240p encode: every table
   function together is under 1% of its samples.
2. The lever the profile actually shows is subpel ME interpolation
   (`me_interp_sample`, ~46%), which §6.1 of the inventory deliberately left
   out.  If NEON effort is to buy loop time, start there, not with slots 2-8.
3. `sad_sse` is the wrong shape for the live intra cost anyway: that loop
   computes SSE only (clamped source), so a dedicated row-wise SSE kernel would
   be the drop-in, and wiring it is a call-site change with its own proof.

## Subpel SAD on NEON: landed, 2.2x single-thread (2026-11-16)

The revised target from the pilot is done.  `subpel_sad_rows` is a `KernelTable`
slot (scalar reference + NEON override) and the host ME cost path calls it:
`me_sad_subpel_rows` takes the kernel under
`AV2ENC_PRIMITIVES && !__CUDA_ARCH__`, the component gates keep the inline
per-sample loop, and the CUDA device pass is unchanged.

### What the NEON kernel does differently

The scalar oracle filters every sample with 8 horizontal taps, rounds, 8
vertical taps, rounds, clips, and accumulates `|cur - pred|` — twice 8 multiply
accumulates per sample, per candidate.  `neon_subpel_sad_rows` keeps the same
arithmetic but changes the shape:

1. **Horizontal pass, once per reference row.**  A row of `w` samples is
   filtered into an `int16` row: 8 taps broadcast as `int16`, `vmlal_s16` into
   `int32` (8 samples per iteration, 16-bit coefficients), then `+4 >> 3`.  The
   `int16` row is exact: the largest tap sum is `(240 * 255 + 4) >> 3 = 7650`.
2. **Row reuse.**  Consecutive output rows need windows of 8 reference rows that
   differ by one row, so an 8-entry ring keyed by the *clamped* row index filters
   each row once instead of up to eight times.  This is the bigger part of the
   win: it is pure work elimination, and it cannot change a value because the
   filtered row is a function of (row, `fx`, filter) alone.
3. **Vertical pass fused with clip and SAD.**  For 8 output samples:
   `vmlal_s16` over the 8 `int16` rows, `+1024 >> 11`, clamp with
   `vmax/vmin`, narrow to `u8`, `vabd_u8` against the current block,
   `vpaddl_u8` into a `uint32` accumulator.  The prediction block is never
   written to memory.
4. **Interior/border split, exact.**  The horizontal vector loop runs only where
   every tap of the 8-sample group is inside `[0, ref_w-1]`, so it never
   over-reads a row; columns outside that range (and `w % 8`) take the scalar
   tap loop.  Rows that clamp at the top/bottom of the reference are filtered by
   the same code and cached like any other row.
5. **Fallbacks.**  `rstep != 1` calls the scalar kernel: the row ring assumes
   consecutive rows and the CPU cost path always passes `rstep = 1`.  The CUDA
   warp shape (`r0 = lane`, `rstep = 32`) therefore stays on the scalar/device
   code and is only tested for equality through the fallback.  Widths above the
   encoder's 128-sample maximum also fall back rather than sizing a stack buffer
   from an argument.  The integer-MV shortcut (`fx | fy == 0`) has its own
   unclamped 16-wide `vabdq_u8` path, because that shortcut reads the reference
   without border clamp.

### Correctness

`gate_primitives` runs the whole sweep against the untouched per-sample oracle
in `me_core.h`, once per table: **21,294 kernel-vs-oracle comparisons,
10,647 direct scalar-vs-NEON pairs, 6,175,416 samples, 768 tap-table entries**,
all exact, covering every filter, all 16x16 phases both signs, `w,h` 1..64 at
the left/top edge, middle, flush right/bottom and on the last row/column,
padded/offset planes, and `rstep` 1, 2, 32.  As a negative control, changing the
NEON vertical rounding bias by 1 LSB fails 5,751 checks.

* pins byte-identical under **both** tables (`gate_video_nightly`,
  `gate_video_qp_lambda` run once with `AV2ENC_SIMD=scalar` and once with the
  default `auto` → neon): 25,009 `a1d0dba1`, 25,141 `3984677b`, 23,690
  `1b99127a`, 24,113 `7f03eb6e`;
* `AV2ENC_SIMD=scalar` vs `=neon` IVFs are `cmp`-identical (not just same
  md5) for the 240p 1T/8T and 720p workloads;
* the primitives test also builds and passes under ASan, so the interior vector
  loads and the row ring do not read past the reference or the ring.

### Encode time (M4, idle, same binary, interleaved A/B)

`nasa_480x240_30fps_60f.y4m --video --frames 60 --qp 106 --fast-decode`:

| Threads | scalar (same binary) | neon | speedup | Output |
|---|---:|---:|---:|---|
| 1 | 13.47 s | 6.11 s | **2.20x** | 25,008 B `0639b436…` |
| 8 | 7.46 s | 5.30 s | **1.41x** | 25,008 B `0639b436…` |

`space8k_1440x720_60f.y4m --video --frames 60 --qp 217 --fast-decode
--video-key-qp-offset -57 --video-leaf-rd-lambda 250 --video-ext-partitions`,
`--cpu-threads 8` (medians of 2, ~2 min/run):

| | scalar (same binary) | neon | speedup | Output |
|---|---:|---:|---:|---|
| 720p 8T | 113.65 s | 48.28 s | **2.35x** | 36,572 B `c4ba2813…` |

Against the **pre-change binary** (`b1bccaa`, same host, same session): 240p 1T
17.4-17.8 s → 6.11 s (**2.9x**), 240p 8T 8.51 s → 5.30 s (**1.6x**), 720p 8T
148.0 s → 48.3 s (**3.1x**).

One honest detail: the *migration alone* (scalar table, no NEON) is already
1.28x at 1T (17.4 → 13.6 s).  Pulling the 8-tap body out of the header stops it
being inlined into `me_search_core`'s per-candidate call site, which shrinks the
search loop.  So of the 2.9x end-to-end, ~1.3x is code shape and ~2.2x is the
NEON kernel, not 2.9x of vectorisation.

### Where the hotspot went

`/usr/bin/sample`, 1 ms, 240p QP106 nightly, `--cpu-threads 1`, 180 frames, same
binary:

| Function | before (scalar table) | after (neon table) |
|---|---:|---:|
| subpel SAD kernel | 4,332 / 7,751 = **55.9%** | 197 / 2,429 = **8.1%** |

The sample counts differ between the two runs (the sampler's effective rate is
not stable), so treat the share as indicative; the wall-clock A/B above is the
primary number.  The subpel cost is no longer the dominant term: the profile is
now transform (`txv::forward_block`/`inverse_block`), intra prediction, RDOQ and
CDEF, each small.

### Verdict

1. Subpel ME was the right target: it is now 8% of the loop and the single-
   thread encode is 2.2x faster than the same binary's scalar table.
2. The interior/border split was worth it: the border path is a scalar tap loop
   that runs only within 4 samples of a reference edge (or for `w % 8`), and the
   exactness tests cover both halves.  The row-ring is where most of the win
   is: it removes seven of every eight horizontal passes, which no amount of
   vectorisation of the per-sample form can do.
3. Keep the table work stopped.  The remaining hotspots are spread across
   transform, intra and RDOQ at 3-7% each; there is no second 50% function to
   attack.  If more loop time is wanted, the next lever is algorithmic (the 720p
   RD gap), not another kernel slot.

## Inverse-transform row pass on NEON (2026-11-24)

The `--nightly-b` profile (`docs/profiles/README.md`) put
`txv::inverse_block` and `txv::forward_block` at **10.7 % and 10.4 %** of work
— together 21.1 %, the largest single subsystem after the RDOQ trellis. This
adds the inverse row pass as a kernel slot.

### Why it vectorises cleanly

Measured on a harness that mirrors the row pass (int16 kernel x int32
coefficient, int64 accumulation, Round2, clip):

| size | full | MAC only | round/clip only | MAC share |
|---|---:|---:|---:|---:|
| 4x4 | 98 ns | 99 ns | 11 ns | 101 % |
| 8x8 | 420 | 344 | 28 | 82 % |
| 16x16 | 950 | 840 | 30 | 88 % |
| 32x32 | 7 819 | 7 514 | 74 | 96 % |

The epilogue is 1-4 %; the loop is multiply-accumulate, which is what NEON is
for.

### Exactness

The scalar contract is int64 accumulation. `vmlaq_s32` on int16 taps widened
to int32 reproduces it as long as the partial sums stay in int32, which holds
for every legal transform (the §9.6 kernels are bounded by 64). Verified over
**320 000 randomised blocks** with coefficients up to 2^20 — 0 mismatches — and
then end-to-end: all five wavefront pins byte-identical at `--cpu-threads 1`
and `8` with matching Stage-0 block dumps, `ctest` 12/12, `gate_primitives`,
`gate_video_nightly`.

### Cost of one detail

The normative kernel is column-major (`k[j*n + o]`); the seam walks taps
contiguously, so it needs the transpose. That is built once per `(type, size)`
and cached — not per block.

### Speed

| arm | median of 3, 16 f QP116 `--nightly-b`, `--cpu-threads 1` |
|---|---:|
| `AV2ENC_SIMD=scalar` | **115.4 s** |
| `AV2ENC_SIMD=auto` (NEON) | **101.5 s** |

**12.1 % off single-thread CPU time**, consistent across reps. Quote the
scalar-vs-auto comparison, not a wall-clock A/B against an older binary: an
earlier attempt to measure it that way read 2.4 % because the "baseline" build
still carried the dispatch slot.

### What is left

The forward transform is the mirror image and has the same structure, so it is
the obvious next slot (~10 % of work, same MAC-bound shape). The row pass here
was the larger half only because the inverse runs on more candidates.

## Forward-transform row pass on NEON (2026-11-24)

Same shape as the inverse pass above, and the other half of the 21.1 % that
`txv::forward_block` + `txv::inverse_block` take in the `--nightly-b` profile.

### Why reordering the double sum is safe

The scalar loop accumulates in binary64: `sum += (double)k[j] * (double)r[j]`
over int16 taps and int16 residual samples. Every product is an **exact
integer**, and the total is bounded by `n * 64 * 32768` — 67,108,864 at n = 32,
against 2^53 ≈ 9.0e15. So the sum is exact in binary64 and *any* summation
order gives the identical double: a vector reduction is bit-identical, not an
approximation. Verified over 800,000 randomised rows against a reversed sum, a
4-lane reduction and an int64 accumulate — 0 mismatches — and then end-to-end.

### The bug this caught

The first attempt passed `tx_kernel_transposed()` to the forward slot as well,
copying the inverse pass. That is wrong: the forward row pass indexes
`kw[v * w + j]` (output then tap, already contiguous) while the inverse indexes
`kw[j * w + oj]` (column-major). Only the inverse needs a transpose. The 720p
and explambda pins failed loudly (36,916 vs 36,572 B) and the fix was to pass
the kernel untouched. Worth knowing: the two passes do **not** share a layout.

### Speed

| arm | median of 3, 16 f QP116 `--nightly-b`, `--cpu-threads 1` |
|---|---:|
| `AV2ENC_SIMD=scalar` | **108.7 s** |
| `AV2ENC_SIMD=auto` (NEON) | **91.7 s** |

**15.6 % off single-thread CPU time** (reps 108.52-108.76 vs 91.35-91.77), on
top of the 12.1 % from the inverse pass. Together the two transform slots
account for the bulk of the 21.1 % the transform subsystem held in the profile.

## RDOQ `Model::init`: the all-zero table (2026-11-24)

The `--nightly-b` profile puts the RDOQ model + trellis at ~33 % of work, and
inside it `rdoq::Model::init` is the single largest function (~14–19 % of all
work here). Instrumenting `init` on the 16-frame QP116 recipe showed why:

| `Model::init` case | calls | time |
|---|---:|---:|
| all-zero block | 18,246,495 (**93 %**) | **13.80 s** |
| any non-zero level | 1,397,514 | 1.85 s |

For an all-zero block every pre-level is 0, so the whole position table
(contexts and escape predictors) is a **pure function of `(w, h, plane)`** —
yet it was rebuilt, for every TX candidate, eighteen million times.

### What landed

`Model::init` takes a shared immutable table for an all-zero block, built once
per `(w, h, plane)` and published lock-free (one `atomic<const vector*>`
compare-exchange per slot; a racing duplicate computes the identical table and
is simply dropped). The context builder is now one static function,
`Model::build_positions()`, used by both the per-block path and the cache, so
the shared and generic tables cannot drift. The model reaches either table
through `positions()`; a copied model reads its own buffers (tested).

No arithmetic moved: this is work elimination, not vectorisation. That is
deliberate — the remaining per-block gather is ~1.8 s and a 2× vectorised
version would buy ~1.2 % of the encode, while the zero table was 14 %.

### Correctness

- `rdoq_test` `test_zero_table_matches_generic`: for 14 geometries × 3 planes ×
  {intra, inter} it builds the shared and the generic table and compares EOB,
  total bits and the per-position symbol rate for magnitudes 0..9, both signs,
  EOB and non-EOB — **2.29 M checks, 0 failures** (and clean under
  ASan/UBSan).
- All five pinned streams **and** their Stage-0 dumps byte-identical at 1 and 8
  threads (`check_wavefront_pins`): default `25,009 a1d0dba1`, quality
  `25,141 3984677b`, nightly `23,690 1b99127a`, explambda `24,113 7f03eb6e`,
  720p `36,572 c4ba2813`. `gate_video_nightly` OK; `ctest` 14/14.

### Speed

16 f QP116 `--nightly-b`, `--cpu-threads 1`, 7 interleaved reps, median of 3
nearby reps each side:

| arm | median | best |
|---|---:|---:|
| pre-change binary | 94.67 s | 84.26 s |
| all-zero table | **79.78 s** | **68.06 s** |

**−15.7 % median (−19.2 % best-case) of single-thread CPU time**, every stream
`cmp`-identical. Re-profiling confirms the move: `Model::init` drops from
**#1 at 13.9 %** to 1.7 % of work; the new #1 is the trellis itself
(`rdoq_quantize`, 10.9 %) with the model pricing next (`cost_symbol` 4.1 %,
`coefficient_bits` 4.0 %).

### What is left

The non-zero gather is now ~1.2 % of work. The large remaining blocks are the
trellis DP and the per-symbol `-log2 p` pricing, which are double-precision
reductions where a reordered sum would move a pin, plus `level_rates.assign`'s
per-position fill. Those need a different shape of proof than a pixel kernel;
do not reach for a `KernelTable` slot without measuring which of them actually
dominates after this change.

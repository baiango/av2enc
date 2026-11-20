# Candidate-level parallel RDO (phase B)

Status: **landed on `exp/b-candidates`** (2026-11-16). Measured, bit-exact.

## Why a pool first

`parallel::parallel_for` created and joined `std::thread`s **per call**. That is
the right shape for whole-frame batches — motion search precompute, partition
probe leaves, per-tile encoders — which run once or a few times per frame.

A per-block RDO loop is the opposite: the TX-type loop runs once per coded
leaf (31,364 times in a 60-frame 240p encode) and the intra mode search once per
intra leaf. Spawning threads there costs tens of microseconds per call, more
than the candidate work itself, so the first step was a persistent pool.

## The pool

`src/core/parallel/parallel_for.h` (header-only, no new TU):

- `Pool(workers)` owns `workers - 1` long-lived threads and counts the
  submitting thread as the last worker. `Pool(1)` owns none.
- `run(count, body)` is fork-join: it blocks until every index has been visited
  exactly once. The body is type-erased with a context pointer plus a function
  pointer, so a call allocates nothing (no `std::function`, no heap job).
- Chunks are drained from an atomic counter, four chunks per participant. That
  keeps the tail short for a 16-candidate loop without paying an atomic per
  index.
- Each worker sleeps on its own condition variable; only the workers that are
  wanted are woken. No spinning when idle.
- `PoolScope` installs the pool for the current thread (`thread_local`), so an
  encode never shares its pool with an unrelated thread that calls
  `parallel_for` concurrently. A single-worker pool is not installed, which
  keeps `--cpu-threads 1` on the historical serial path.
- `parallel_for` keeps its signature and its per-call thread path when no pool
  is installed. Nested calls still run serially through
  `in_parallel_region()`: a pool worker, or the submitting thread while it
  drains its own chunks, never opens a nested region.
- `parallel_for_pooled(count, min_count, body)` is the per-block helper: it
  uses an installed pool only, and otherwise runs the serial loop. A per-block
  site therefore can never fall back to per-call thread creation.

Lifecycle: **one pool per encode**, created at the composition root
(`encode_video_sequence` in `src/pipeline/video/video_frame.cpp`) and sized
from `--cpu-threads`. It is not per frame and not per block.

Determinism: chunk assignment follows timing, but every index runs exactly once
and every caller merges per-index slots in index order. The pool is a
scheduling change, not an arithmetic one.

## A candidate loop is a different contract from a batch loop

Both RDO loops share three properties that make them safe to parallelise:

1. **Candidates are independent.** A TX candidate owns every buffer it writes
   (`coded`, `adj`, `coeff`, `levels`, `scan`, `recon` are locals inside
   `evaluate_candidate`); the §8.3 models are constructed per call. An intra
   candidate writes only its own predictor buffer and reads a `const`
   `IntraEdges` prepared once. There is no mutable static or scratch state on
   either path (checked across the transform, DDT, TCQ, RDOQ, scan, inter
   residual, intra and IBP modules).
2. **The candidate set is fixed.** No candidate is pruned or early-exited based
   on an earlier candidate's cost.
3. **Selection is a sequential pass in the original order.** This is the part
   that must not be turned into a reduction: the TX comparator tie-breaks on
   `(cost, sse, tx_type)` and the intra search keeps the *first* strict minimum,
   so the earlier candidate wins a full tie. Both loops therefore evaluate into
   a preallocated vector and then run the original loop body, unchanged, over
   the slots in index order.

Nothing sums a candidate's cost across threads. TX costs are `double` (SSE plus
`lambda * bits`), and reordering a sum would change rounding and therefore
decisions; each candidate keeps its own sequential sum.

## Landed targets

| Target | Site | Gate | Measured (720p QP217 8T) |
|---|---|---|---|
| TX type candidates | `video_tx_search.cpp`, `evaluate_plan`'s `types` loop | ≥ 4 candidates | **−3.8 s** |
| Intra mode candidates | `video_intra_search.h`, `search_luma` | leaf ≥ 256 samples | **−3.3 s** |
| TX partition plans | `evaluate_plan` call sites | — | **not kept** (−0.39 s, i.e. it loses) |

The plan target was implemented and measured first (task order) and then
dropped: see the negative-result row below.

## Thresholds, measured

Thread hand-off is not free (a fork-join round trip wakes up to 7 threads
through a condition variable). Both gates are set from 240p QP106
`--cpu-threads 8` medians, interleaved:

| TX-type minimum | median | Intra minimum (px) | median |
|---|---:|---|---:|
| 2 | 4.84 s | 64 (8x8) | 4.49 s |
| **4** | **4.47 s** | **256 (16x16)** | **4.46 s** |
| 8 | 4.66 s | 1024 (32x32) | 4.66 s |
| 16 | 4.64 s | | |

At 240p the TX-type loop averages 3.3 candidates per leaf (103,913 evaluations
over 31,364 calls), and only ~40% of leaves reach 4. Below that, the serial
loop wins. The intra search always has 61 candidates, so its gate is leaf area
(the concurrent path needs one `w*h` buffer per candidate).

## Results

Interleaved A/B on the same host, medians; both binaries built from this tree at
`-j1`, `Release`, CPU-only.

| Workload | main HEAD | branch | change |
|---|---:|---:|---:|
| 240p QP106 `--cpu-threads 1` | 5.81 s | 5.80 s | 0% (no 1T regression) |
| 240p QP106 `--cpu-threads 8` | 5.04 s | **4.46 s** | **1.13x** |
| 720p QP217 `--cpu-threads 8` | 45.20 s | **40.83 s** | **1.11x** |
| 720p QP217 `--cpu-threads 1` | 66.67 s | 66.88 s | 0% |

Implied parallel fraction (Amdahl, `p = (1 - 1/S8) / (1 - 1/8)`):

| Workload | main HEAD | branch |
|---|---:|---:|
| 240p QP106 | 15% | **26%** |
| 720p QP217 | 37% | **44%** |

Every stream is byte-identical at 1 and 8 threads, and byte-identical to the
main-HEAD build (`0639b436…` at 240p QP106, `c4ba2813…` / 36,572 B at 720p
QP217). The host was not fully idle during the 240p runs (Spotlight indexing,
load average 5-10), so the absolute times are ~5% higher than the idle numbers
in `docs/simd-plan.md`; the A/B is interleaved and the medians are stable to
±0.05 s.

## Determinism guard

`tools/refactor/check_thread_determinism.sh` encodes both measured points with
`--cpu-threads 1` and `--cpu-threads 8` and requires `cmp`-identical IVFs. It
also checks the 720p pin (36,572 B, `c4ba2813…`) so that two matching *wrong*
streams cannot pass. It is not registered in `tools/run_gates.sh` yet.

## What was rejected

**TX-partition plan candidates** (task target 2). Parallelising the plan loop
is the wrong axis: `--fast-decode` selects about 2 plans per inter leaf, and the
plans are dominated by the TX-type loop inside them (up to 16 candidates per
leaf). With plans concurrent, each plan's type loop becomes a nested serial
region, so the coarse axis *suppresses* the fine one. Measured 8T medians of 3:

| Workload | plans concurrent | plans serial | verdict |
|---|---:|---:|---|
| 240p QP106 | 4.12 s | 4.10 s | wash |
| 720p QP217 | 36.90 s | 36.52 s | concurrent slightly loses |

Both are within run-to-run noise, and the 720p number is on the wrong side, so
the parallel path was not kept. The type loop covers the same candidates at a
better granularity.

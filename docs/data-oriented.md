# Data-oriented TX-type search (S1) and dispatch sweep (S2)

Branch: `exp/d-txsearch` (from `main` at `54f6a8c`, 2026-11-17).
Scope: `src/pipeline/video/video_tx_search.cpp` plus the new header-only
`src/pipeline/video/video_tx_scratch.h`. No other file, and no arithmetic.

The refactor removes the per-candidate heap work from `evaluate_candidate()`.
Byte-identical streams, measured allocation drop, small CPU-time win. The
dispatch threshold (S2) does **not** have a measurable winner and stays at 4.

> **Timing caveat.** The measurements below were taken on the shared Apple M4
> dev host while four other agents were running encode/CVVDP jobs (load average
> 9–11 on 10 cores). Allocation counts and `cmp` identity are unaffected by
> load; the wall-clock tables are contended and flagged for re-measurement on
> an idle machine. The control row in S2 shows the noise floor: two binaries
> that execute *identical* code paths differ by 3.1% wall / 4.2% user on that
> host.

## What changed

`evaluate_candidate()` used to build ~10 `std::vector`s per TX-type candidate
and to call `txpart::get_scan()` per candidate even though the scan is a pure
function of `(tx_size, class)`.

* **Per-worker scratch** (`TxScratch`, `thread_local`, 100,704 bytes):
  `coded`, `recon`, `adj`, `coeff`, `coeff_i`, `levels`, `dequant`,
  `dequant_levels`, `aux` (TCQ reference / RDOQ scalar seed), `syntax_scan`,
  the TCQ and RDOQ `Coefficient` inputs, and the two §8.3 model objects.
  The model objects are reused, so their position tables keep their capacity
  instead of reallocating on every symbol pass.
* **Hoisted scan cache**: four direct-mapped entries keyed by
  `(tx_size, TxScanClass)`. A leaf uses at most three of them (2D, HORIZ,
  VERT), so each `(tx_size, class)` is generated once per worker per leaf
  instead of once per candidate. Values are the same `get_scan()` output.
* **Hot/cold split**: scoring writes a `CandidateSlot` holding only
  `(tx_type, cost, sse, rate_bits, eob, nonzero, valid)` plus spans into a
  per-caller payload ring. Losing candidates never own heap storage. The
  winner is copied into the existing `LeafCandidate` by
  `materialize_candidate()` — the same bytes the old `std::move(cand)`
  produced, without a per-candidate vector.
* **Payload ring sizing**: the ring holds one slot per candidate of the leaf
  in the pooled path and one slot in the serial path. Slot *i* is written only
  by candidate *i*, so the pooled body needs no locking beyond the disjoint
  slots. Worst case is 16 candidates at 64x64 = 192 KiB of ring (16 x (1024
  int32 levels + 4096 int16 residual)); a typical 8x8 leaf is 6 KiB. It is
  allocated once per caller thread and grows to the high-water mark.

Preserved exactly (the code was moved, not rewritten): candidate evaluation
order, the `consider()` comparator and its `(cost, sse, tx_type)` tie-break,
every double's accumulation order, and every error string. There is no new
fast-math and no reordering. `assert`s guard the slot sizes (the adjusted
`params.w * params.h` grid equals `tx_scan_area(tx_size)` for all 25 sizes);
they compile out under `NDEBUG`.

## S1 allocations

Local `DYLD_INSERT_LIBRARIES` shim (not part of the repo), 60-frame 240p
`--nightly` at `--qp 106`:

```sh
DYLD_INSERT_LIBRARIES=/tmp/allocprof/allocprof.dylib build-dev/av2enc \
  testdata/nasa_480x240_30fps_60f.y4m -o /tmp/a.ivf --video --frames 60 \
  --cpu-threads 1 --qp 106 --nightly
grep -m1 ALLOC /tmp/a.err
```

| Run | Before | After | Change |
|---|---:|---:|---:|
| 240p QP106 `--nightly`, 1T | 1,886,163 calls / 6,627.1 MB | **807,716 / 2,432.5 MB** | **−57.2% calls / −63.3% bytes** |
| same, 8T | 1,903,053 / 6,684.5 MB | 814,861 / 2,487.6 MB | −57.2% / −62.8% |

Derived rates (1T, 31,364 `search()` calls and 103,913 candidate evaluations
for this encode):

| Metric | Before | After |
|---|---:|---:|
| allocations per coded leaf (`search()` call) | 60.1 | 25.8 |
| allocations per frame | 31,436 | 13,462 |
| allocations removed per candidate evaluation | — | 10.4 |

The removed 1.08M calls are exactly the per-candidate vectors:
`coded`, `adj`, `coeff`/`coeff_i`, `levels`, `scan`, `dequant`,
`dequant_levels`, `recon`, the TCQ/RDOQ inputs and reference, and the §8.3
model inputs. What remains in the 807,716 is (a) per-leaf containers in this
TU (`types`, `order`, the `best` payload vectors, the final `Out` leaf copies),
(b) the plan vector, (c) the `tcq::Result` / `rdoq::RdoqResult` vectors:
both trellis entry points do `result = Result{}` and therefore free/allocate
three vectors per call — that code is in `codec/transform/tcq` and
`codec/rate/rdoq`, outside this task's ownership, and (d) the caller's
frame/block bookkeeping, which the shim counts process-wide.

## Byte-identity evidence

Pre-change (`54f6a8c`) and post-change binaries were built from the same
presets (`dev` and `release-timing`) and the streams compared with `cmp` (not
md5 alone), at 1 and 8 threads.

| Recipe | Bytes | md5 |
|---|---:|---|
| default `--video --qp 189` | 25,009 | `a1d0dba1259b20a4b628d4cd1e14c33c` |
| quality `--video --qp 190 --video-rd-preset quality` | 25,141 | `3984677b9b665952984b07b7cf75c6ea` |
| nightly `--video --qp 110 --nightly` | 23,690 | `1b99127a4cd1d386fa5600fb63c57a83` |
| explicit-lambda (the 136/125/0.1 recipe) | 24,113 | `7f03eb6e82d6ba4cf6666c78285f945b` |
| 720p `--qp 217 --nightly --video-key-qp-offset -57 ...` | 36,572 | `c4ba28138d8f6569d5e08721a66d755c` |

All ten checks (5 recipes x {1T, 8T}) are `cmp`-identical on both presets, for
the 240p fixture `nasa_480x240_30fps_60f.y4m` and the 720p fixture
`space8k_1440x720_60f.y4m`. The allocation-run stream
(`qp 106 --nightly`) is also identical to the pre-change binary's
(`0639b43630abb454f7be5ac494d2efb4`, 25,008 B).

Every stream produced by the timing runs below (78 files: 60 measured plus
probes) was `cmp`-identical and decoded with the pinned `dav2d`
(`tools/validate_dav2d.sh`, 78/78). No stream in this experiment is new.

Verification: `tools/test.sh --build build-dev` 10/10;
`tools/run_gates.sh --bin build-dev/av2enc --no-ctest` 51/51, including
`gate_video_nightly` and `gate_video_qp_lambda`; `gate_wavefront_pins` OK.

## S2 — dispatch threshold sweep

The type loop still calls `parallel::parallel_for_pooled()` when
`order.size() >= kTypeCandidateParallelMin` (4). Four `release-timing`
binaries were built with the constant set to 4, 8, 16 and 1000 (the last is
"serial": the gate can never fire, so the loop body is the serial loop). Runs
were interleaved config-by-config inside each rep; output streams were checked
before the times were trusted.

`release-timing`, medians of interleaved reps (wall seconds; user CPU seconds;
brackets are min..max wall):

| Workload | Threshold 4 | 8 | 16 | Serial (1000) |
|---|---:|---:|---:|---:|
| 240p QP106 1T (control) | 8.07 [7.84..8.11] / 6.75 | — | — | 7.83 [7.42..9.04] / 6.48 |
| 240p QP106 8T | 4.52 [4.36..5.29] / 7.57 | 4.54 [4.32..4.97] / 7.75 | 4.76 [3.94..5.51] / 7.75 | 4.46 [4.15..5.20] / 7.59 |
| 720p QP217 8T | 30.44 [28.44..33.94] / 96.68 | 31.51 [29.91..33.76] / 98.65 | 29.81 [27.76..34.27] / 97.48 | 31.51 [30.27..33.37] / 97.83 |

Reps: 3 (720p, 240p 1T), 4 (240p 8T).

At 1T no pool is installed, so every binary executes exactly the same serial
path; that row is a pure noise control. The two binaries measured there differ
by 3.1% wall / 4.2% user, which is larger than every difference in the 8T rows
(≤2.4% user CPU). Per-config wall spreads are 10–40%, so no setting is
separable from the others.

**Verdict: leave the threshold at 4.** There is no measured winner, and the
`docs/video-candidate-parallel.md` record already established 4 as the
measured optimum on an idle host. Nothing is changed speculatively.

## S1 timing on `release-timing`

Pre-change (`54f6a8c`) vs post-change with the same threshold 4, interleaved,
medians:

| Workload | Before wall / user | After wall / user | Wall spread (before / after) |
|---|---:|---:|---:|
| 240p QP106 1T | 6.27 / 5.59 | 6.35 / 5.56 | 36.5% / 36.8% |
| 240p QP106 8T | 3.51 / 6.96 | 3.45 / 6.83 | 3.5% / 15.4% |
| 720p QP217 8T | 28.73 / 96.37 | 30.05 / 95.69 | 7.6% / 12.0% |

Reps: 5 (240p), 3 (720p). User CPU moves by −0.5% / −1.9% / −0.7%; the wall
numbers are inside the host's noise (one earlier 3-rep block, taken before the
load rose, had 240p 8T 4.37 → 4.18 and 720p 8T 32.20 → 31.80, the same
direction as the user-time numbers).

This is the expected size: ~1.08M allocations at roughly 50–100 ns each is
0.05–0.1 s of a 3.5–6 s encode, i.e. 1–3%, and the allocator's cost is partly
hidden by the surrounding transform work. The refactor is a memory/allocation
change, not a throughput change: the remaining time is the forward/inverse
transform, the §8.3 model and the decisions themselves.

**Marked for re-measurement on an idle host** before any timing claim is
quoted. The allocation counts and the identity evidence are final.

## Follow-ups (not done here)

1. `tcq::Result` / `rdoq::RdoqResult` free their vectors on every call
   (`result = Result{}`). Moving the buffers to a caller-provided span would
   remove another ~3 allocations per trellis call; those files are owned by
   the trellis/rate owners.
2. The per-leaf containers (`types`, `order`, the plan vector) still allocate
   ~10 times per `search()` call. A small per-thread container arena would
   take the 25.8/leaf down further, at the cost of more state in this TU.
3. Re-run S2 on an idle host. Locally the dispatch axis is bounded by the
   noise; the doc's earlier threshold table remains the best available data.

# CPU encode speed on the 720p still: two exact hot-path cuts

Status: **landed, byte-identical, measured.** On `~/Downloads/src720.png`
(1280x720, converted to `src720_1f.y4m`, md5 `8070833614…`) the `--nightly-b`
key at QP 177 gets **1.069x** faster at 1 thread and **1.053x** at 8 with the
**same** stream (38,089 B, md5 `01e931f8c868eca71fe18da12b6d8db8` in every
timed run). Nothing here changes a decision: both changes remove work that
provably cannot affect a value, and the proofs are in the code and covered by a
new 2.3M-check unit test.  The full 57-gate suite is green and all five 240p
README pins still reproduce byte-exactly.

Provenance: `src720_1f.y4m` made with `ffmpeg -i src720.png -pix_fmt yuv420p
-color_range tv -colorspace bt709 -color_primaries bt709 -color_trc bt709`
(md5 `80708336148219e2d20c502a43675f94`, identical to
`docs/k-720p-levers.md`). Encoder `build-release/av2enc`, Apple M4, one frame,
`--video --frames 1 --nightly-b --video-gdf --video-uv-qp-offset -23` with
`AV2ENC_RD_DRIVER_MIN=16`.

```sh
AV2ENC_RD_DRIVER_MIN=16 build-release/av2enc src720_1f.y4m -o out.ivf \
  --video --frames 1 --qp 177 --cpu-threads 1 --nightly-b \
  --video-gdf --video-uv-qp-offset -23
```

## 0. Where the time goes (1 thread, `sample` at 1 ms)

`--nightly-b` on a lone key is a **serial** key walk: threading does not help it
(1T 20.1 s, 2T 23.4 s, 4T 25.1 s, 8T 25.2 s at `MIN=16` — the pool only collects
leaves that do not exist on a key picture). Single-thread speed is therefore the
only lever, and the profile put the cost in a few places:

| share | where |
|---:|---|
| 41.6 % | RDOQ: model (`coefficient_bits`, `build_positions`, `init`) + trellis (`rdoq_quantize`) |
| 16.9 % | transform pair (`txv::forward_block` / `inverse_block`) |
| 12.4 % | TX search driver |
| 14.3 % | `_platform_memmove` — dominated by the key prune's full-recon snapshot |

Ablations (Release build, `AV2ENC_RDOQ_*` guards, since removed) bound the
pieces: with the DP loop skipped the encode is **14.6 s**, with
`rdoq_quantize()` a total no-op **12.0 s**, against **20.5 s** baseline. So the
trellis is ~5.9 s and the model ~2.6 s.

## 1. The AV2 model's rate is state-free, so the trellis runs once

`RateModel::coefficient_rate_ignores_state` already told the trellis that its
model prices a symbol from `(scan_pos, level, is_eob)` only. The DP did not use
that: it searched the same state-free problem once per arrival state. Every
winner is bit-identical across states — the candidate costs read only the
distortion, the (cached) rate and the shared `dp/dd/dr_next`, and the all-zero
seed is the same for every state — so `rdoq_quantize()` now runs the search
once and copies the winner into each state (`src/codec/rate/rdoq/rdoq.cpp`).
A state-reading model keeps the historical per-state loop untouched.

`src/codec/rate/rdoq/rdoq_test.cpp::test_state_free_trellis_matches_per_state`
proves it: 280 random blocks over 7 shapes and 4 state caps, the AV2 model's
state-free path against the same model with the flag off, **bit-identical**
`cost`, `distortion`, `rate_bits`, levels, EOB, search range and truncation flag
(exact `==`, not a tolerance), with a state-reading control model that must keep
the slow path.

## 2. Removed two per-call allocation sets on the RDOQ path

- `rdoq_quantize()` built ten small `std::vector`s per call (~280k calls on this
  still). They now come from one `thread_local` block
  (`TrellisScratch`), fully assigned before use, so no value changes.
- `Model::init()` built a pre-level plane per call. It uses a `thread_local`
  scratch too; the shared zero-table build now publishes under its own lock
  instead of sharing that scratch unsynchronised, and re-checks the cache after
  taking the lock so the serial fast path is a single acquire load.

## 3. The key prune copied the whole reconstruction per tree node

`seq_driver_decide_sb()` snapshots the frame at every node so a losing trial can
be undone. It copied **all three recon planes, the MV grid and the JOINT/UV
planner grids** per node — the largest `memmove` in the key profile. This walk
runs only with every inter tool off, and a trial leaf never writes the MV/JOINT/
UV grids, so the snapshot now keeps only the intra joint-mode map, the
superblock state and the block count. The syntax buffers (`levels`, `palettes`,
`leaf_costs`, `leaf_dump`, `sub_pus`) are append-only inside a trial and every
read compares against `sink.blocks.size()` or the index the leaf itself pushed,
so they need no rewind; `intra_joint_` is copied in place.

## 4. Measurements

Same binary pair, interleaved runs (5 reps each, arms alternating every run) on
an otherwise idle Mac M4 after the full gate suite finished, so no background
load contaminates either arm:

```sh
tools/bench_720p_still.sh --a build-base/av2enc --b build-release/av2enc \
  --runs 5 --threads 1,8
```

| Arm | Threads | median wall | speedup | IVF | md5 |
|---|---:|---:|---:|---:|---|
| baseline `6bc50337` | 1 | 18.06 s | — | 38,089 B | `01e931f8…` |
| this change | 1 | **16.89 s** | **1.069x** | 38,089 B | `01e931f8…` |
| baseline `6bc50337` | 8 | 18.63 s | — | 38,089 B | `01e931f8…` |
| this change | 8 | **17.69 s** | **1.053x** | 38,089 B | `01e931f8…` |

Re-measured on the final tree (after the concurrently landed
`AV2ENC_TX_LIVE_CTX` work merged); an earlier run of the same pair on the
same host gave 19.49 -> 18.19 s (1.071x) at 1 thread and 20.65 -> 19.24 s
(1.073x) at 8, i.e. the gain is stable across sessions while the absolute
times drift with host load.  Baseline medians on this host move between
sessions by ~8 %, which is why every number here is an interleaved A/B.

All 20 encodes of each run produced one md5 —
`01e931f8c868eca71fe18da12b6d8db8` — so the speedup is on the pinned stream,
not a different one.  Threading is *negative* on this path (8T is slower than
1T for both arms: the lone-key walk is serial and the pool pays for itself only
when the leaf/candidate loops have work to hand out), which is why the work
here is single-thread code.

Byte-identity also holds at a second operating point:

| point | args | bytes | md5 |
|---|---|---:|---|
| 720p still QP 177 | §0 recipe | 38,089 | `01e931f8c868eca71fe18da12b6d8db8` |
| 720p still QP 229 | `--qp 229` | 6,252 (6,079 B tile) | `618619337947136bc8c0b264b3d38cd8` |

### 4.1 Gates

`tools/run_gates.sh --bin build-release/av2enc --gate-jobs 4` after the change:
**56 passed, 1 skipped** (`gate_lpl36_parity`, prerequisites missing), **0
failed**, ctest 14/14.  That includes the gates that pin the RDOQ and key
streams: `gate_video_rdoq`, `gate_video_nightly`, `gate_video_ext_nightly`,
`gate_video_qp_lambda`, `gate_wavefront_pins`, `gate_video_rd_preset`,
`gate_build_seam`.

The component test (`src/codec/rate/rdoq/build_and_test.sh`): **2,296,120
checks, 0 failures**, including the new state-free-vs-per-state equality sweep.

### 4.2 The 240p ladder still reproduces every README pin

The change is code-path independent of resolution, so the five 480x240 pins
were re-encoded on the final binary (`--cpu-threads 8`):

| pin | bytes | md5 |
|---|---:|---|
| default QP189 | 25,009 | `a1d0dba1259b20a4b628d4cd1e14c33c` |
| `--video-rd-preset quality` QP190 | 25,141 | `3984677b9b665952984b07b7cf75c6ea` |
| `--fast-decode` QP110 | 23,690 | `1b99127a4cd1d386fa5600fb63c57a83` |
| `--fast-decode` QP106 | 25,008 | `0639b43630abb454f7be5ac494d2efb4` |
| `--nightly-b` 16 f QP116 (gops 2) | 17,222 | `2a7fbe710542a89c3a9d2c93fab94101` |

All five equal the README table exactly.

## 5. Where the next exact win is, and where it is not

The profile after this change still spends, on the 1-thread still:

| share | where | can it be cut exactly? |
|---:|---|---|
| ~9 % | `txv::forward_block` + `inverse_block` | partially: the NEON row kernels exist, but the column passes and the sample-duplication tail are still scalar |
| ~10 % | TX search driver (`evaluate_candidate`, `TxScratch`, `get_scan`) | `get_scan` is regenerated per candidate for the same `(tx_size, class)` in `uv_rd::trellis_plan`; caching it is pure work elimination |
| ~14 % | `memmove`, now mostly the *frame-start* restore per superblock | not exactly removable: the raster walk is defined to leave the frame untouched between superblocks, so that restore is real work.  The per-node copies are gone |
| ~35 % | RDOQ model + trellis | the model is down to its CDF lookups; further cuts mean changing the rate model, which changes decisions |

`AV2ENC_RDOQ_PREV_CAP=0` (states = 1) measures the same stream here, i.e. the
state dimension is inert on this picture — another reason the trellis change is
exact but not the biggest lever.  `AV2ENC_SIMD=scalar` vs `auto` (NEON) was also
byte-identical, so the NEON kernels are doing what the pins say.

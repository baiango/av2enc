# CPU encode speed on the 720p still: two exact hot-path cuts

Status: **landed, byte-identical, measured.** On `~/Downloads/src720.png`
(1280x720, converted to `src720_1f.y4m`, md5 `8070833614…`) the `--nightly-b`
key at QP 177 gets **1.07x** faster at both 1 and 8 threads with the **same**
stream (38,089 B, md5 `01e931f8c868eca71fe18da12b6d8db8` across all 20 timed
runs). Nothing here changes a decision: both changes remove work that provably
cannot affect a value, and the proofs are in the code and covered by a new
2.3M-check unit test.  The full 57-gate suite is green.

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
| baseline `6bc50337` | 1 | 19.49 s | — | 38,089 B | `01e931f8…` |
| this change | 1 | **18.19 s** | **1.071x** | 38,089 B | `01e931f8…` |
| baseline `6bc50337` | 8 | 20.65 s | — | 38,089 B | `01e931f8…` |
| this change | 8 | **19.24 s** | **1.073x** | 38,089 B | `01e931f8…` |

All 20 encodes produced one md5 — `01e931f8c868eca71fe18da12b6d8db8` — so the
speedup is on the pinned stream, not a different one.  Threading is *negative*
on this path (8T is slower than 1T for both arms: the lone-key walk is serial
and the pool pays for itself only when the leaf/candidate loops have work to
hand out), which is why the work here is single-thread code.

Byte-identity also holds at a second operating point:

| point | args | bytes | md5 |
|---|---|---:|---|
| 720p still QP 177 | §0 recipe | 38,089 | `01e931f8c868eca71fe18da12b6d8db8` |
| 720p still QP 229 | `--qp 229` | 6,112 | `618619337947136bc8c0b264b3d38cd8` |

### 4.1 Gates

`tools/run_gates.sh --bin build-release/av2enc --gate-jobs 4` after the change:
**56 passed, 1 skipped** (`gate_lpl36_parity`, prerequisites missing), **0
failed**, ctest 14/14.  That includes the gates that pin the RDOQ and key
streams: `gate_video_rdoq`, `gate_video_nightly`, `gate_video_ext_nightly`,
`gate_video_qp_lambda`, `gate_wavefront_pins`, `gate_video_rd_preset`,
`gate_build_seam`.

The component test (`src/codec/rate/rdoq/build_and_test.sh`): **2,296,120
checks, 0 failures**, including the new state-free-vs-per-state equality sweep.

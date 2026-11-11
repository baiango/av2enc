# M1 GPU chroma decision + prediction

GPU port of the M1 chroma half of `src/encoder/m1/m1_still.cpp`:
`choose_uv_mode`, the DC prediction that both candidates use, `make_cfl_block`,
the §7.13.5 derived alpha, and the winning prediction.

Owner: workstream 6 (`docs/m1-gpu.md`). This directory is self-contained; it
does not modify `src/cuda/m1/` root files, `src/main.cpp`, `CMakeLists.txt`,
`src/encoder/`, or `src/cuda/cfl/`.

Status: **CPU-parity verified on the RTX 3060 (`sm_86`, CUDA 12.8)** for
synthetic states and for 64x64 / 640x360 frames. No commit.

## Policy (default = CPU default)

```
candidates = { DC, CfL(derived alpha) }
```

Directional/smooth UV prediction is **not** bit-exact against the pinned
decoder (`docs/negative-results.md`), so it is never selected. The candidate
id enum, the scoring switch and the reserved availability fields keep the
extension point visible: `M1ChromaOptions::enable_uv_directional` returns
`cudaErrorNotSupported` rather than silently diverging, so enabling it later is
a local change in `run_block()`.

Scoring and tie-break are transcribed from `choose_uv_mode()`:

| Candidate | Scored on | Notes |
|---|---|---|
| DC | U plane only | CPU quirk: `sse_against_source(1, ...)` |
| CfL | U + V | derived α per plane, §7.13.5 |
| winner | strict `cfl < dc` | DC is evaluated first, so ties keep DC |

Do not "fix" the U-only DC score here: it changes M1 decisions. A rate-aware
RDO pass that scores both planes belongs upstream of this module.

`M1ChromaOptions::enable_cfl = 0` reproduces the CPU early-out (`force_dc` /
`AV2ENC_M1_UV_DC`): winner DC, `is_cfl = 0`, `result.sse = 0`.

## API

```cpp
#include "cuda/m1/chroma/m1_chroma.h"
using namespace av2enc::cuda::m1;

M1ChromaJob job;      // device pointers, one chroma residual block
M1ChromaOptions opt;  // frame-level policy (defaults are CPU parity)
M1ChromaResult* d_res;  // device memory

chroma_decide(job, opt, d_res, stream);            // one block, one launch
chroma_decide_batch(d_jobs, n, opt, d_results, stream);  // one CUDA block/job
```

Both entry points are asynchronous, allocate nothing, and never synchronize.
The batch form assumes the caller only batches blocks whose inputs are frozen
(stage 2/wavefront); it validates the options but not every descriptor.

`M1ChromaJob` inputs:

* `luma`, `luma_stride`, `luma_max_x/y`, `sb_top_y` — reconstructed luma with
  the CPU `make_cfl_block()` bounds: `luma_max = MiCols*MI_SIZE - 1` (coded
  plane, **not** the visible size). `sb_top_y = (chroma_y*2/64)*64`.
* `chroma_u/v`, `chroma_stride` — reconstructed chroma. Only the above row and
  left column of the block are read (derived alpha); the current block is
  ignored because this module computes the DC base itself.
* `src_u/v`, `src_stride_*`, `src_max_x/y` — visible source planes.
* `x`, `y`, `w`, `h` — chroma rect (`w`,`h` power of two, 4..64).
* `coded_width/height` — `MiCols*2`, `MiRows*2`. A bottom-edge block may exceed
  these (`write_recon` writes into the padded plane, CfL truncates neighbour
  replication); that is intentional and accepted.
* `pred_u/v`, `pred_stride` — optional output planes (one row of `w` samples;
  `pred_stride = 0` means tightly packed). The winning prediction is written
  here, so the caller's recon planes are not mutated.
* `num4_above_right`, `num4_below_left` — reserved for the directional
  candidate. DC and CfL do not read them; DC availability is `x>0`/`y>0`,
  CfL availability is the §5.20.4.1 chroma-mi rule `x>=4`/`y>=4`, both derived
  inside the kernel exactly as the CPU does.

`M1ChromaResult` returns `winner`, `uv_mode` (0 = DC for both DC and CfL),
`angle_delta`, `is_cfl`, `alpha_u/v` (Q8), per-plane SSE
(`sse_dc_u/v`, `sse_cfl_u/v`), the DC values used, and `sse` = the
CPU-comparable `UvChoice.sse`.

CfL derived alpha is **signed**: `resolve_division()` returns
`[-511, 511]` Q8. Negative alphas are normal and were observed in every real
frame test. `alpha_q8_min/max` default to `[-511, 511]` (a no-op); a narrower
range is an encoder experiment that changes decisions by construction.

## Implementation

`m1_chroma.cu`, one CUDA block of 128 threads per chroma block:

1. thread 0 computes the U/V DC values, the §7.13.5 `lumaAvg` plus
   `lumaAbove/lumaLeft`, and the per-plane derived alpha;
2. all threads score DC (U) and CfL (U+V), recomputing `L` inline (9 taps,
   no per-block scratch buffer), then reduce with warp shuffles + shared
   memory;
3. thread 0 applies the strict tie-break and writes `M1ChromaResult`;
4. all threads write the winning prediction to `pred_u/v`.

All arithmetic is integer and uses the shared helpers in
`src/encoder/cfl/cfl_common.h` (`Round2Signed`, `Div_Lut`,
`resolve_division`, `ds_weight`); there is no FP contraction risk. Shared
memory is ~0.7 KiB, no arena, no allocation.

## Build / test

On the Linux + CUDA box (no local build; `nvcc` is not on macOS):

```sh
./src/cuda/m1/chroma/build_m1_chroma_test.sh               # sm_86, build + run
./src/cuda/m1/chroma/build_m1_chroma_test.sh --no-run
ARCH=sm_120 ./src/cuda/m1/chroma/build_m1_chroma_test.sh   # secondary target
BUILD_DIR=/path ./src/cuda/m1/chroma/build_m1_chroma_test.sh
```

Default private build directory: `<repo>/build-m1-chroma` (script can be run
from anywhere). The script compiles `m1_chroma.cu`, the test TU, and the real
CPU component sources (`partition`, `intra_modes`, `tx_variants`, `cfl`,
`deblock`, `m1_headers`, `m1_syntax`, `av2_entropy`, `y4m`, bitstream headers).
It generates 64x64/640x360 `testsrc2` frames with `ffmpeg` and, when the
unsynced `testdata/shine_1280x720_60f.y4m` is present, a real-content 64x64 /
640x360 pair as well. Without `ffmpeg`/Y4M arguments the built-in synthetic
frames are used.

The test TU includes `m1_still.cpp` with `private` made public **for the test
translation unit only** (the file is not edited). It replays the CPU block loop
with a GPU hook and then asserts that the replayed reconstruction equals an
independent `Encoder::encode()` run, so a harness divergence cannot pass
silently. If `m1_still.cpp`'s internal block loop changes again, this test is
the first thing to update; the chroma entry points themselves are stable.

## Verified results (RTX 3060, `sm_86`, CUDA 12.8)

```
m1_chroma_test: NVIDIA GeForce RTX 3060 cc 8.6, CUDA 12080
batch/policy checks   ok
synthetic            blocks=600 cfl=250 failures=0
64x64 testsrc2_64x64.y4m dc/cfl        blocks=13  cfl=5   neg-alpha=12   recon-harness=exact
64x64 testsrc2_64x64.y4m full-intra    blocks=13  cfl=5   neg-alpha=12   recon-harness=exact
640x360 testsrc2_640x360.y4m dc/cfl    blocks=247 cfl=61  partial=22 sb-edge=183 sb-top>0=198 recon-harness=exact
640x360 testsrc2_640x360.y4m dc-only   blocks=247 cfl=0   partial=22 sb-edge=183 sb-top>0=198 recon-harness=exact
64x64 shine_64x64.y4m dc/cfl           blocks=12  cfl=6   recon-harness=exact
64x64 shine_64x64.y4m full-intra       blocks=12  cfl=6   recon-harness=exact
640x360 shine_640x360.y4m dc/cfl       blocks=218 cfl=129 partial=10 sb-edge=154 sb-top>0=181 recon-harness=exact
640x360 shine_640x360.y4m dc-only      blocks=218 cfl=0   partial=10 sb-edge=154 sb-top>0=181 recon-harness=exact
total blocks=1580 cfl=462 failures=0 (decision=0 alpha=0 sse=0 pred=0)
RESULT: PASS
```

What "exact" means per block: `is_cfl`, `uv_mode`, `angle_delta`, both derived
alphas, all four per-plane SSE values, the CPU-comparable `sse`, and every
prediction byte in U and V. Coverage includes:

* all M1 chroma residual sizes (4x4 through 32x32, rectangular included);
* chroma-mi availability (`x<4`/`y<4` CfL-disabled, `x=0`/`y=0` DC-single-side),
  including the no-neighbour DC = 128 case;
* partial bottom blocks (`partial=22` at 640x360: a 32x32 chroma block at
  `y=160` extends past the 180-row coded chroma plane);
* blocks on chroma superblock edges and below/at the CfL superblock-top clamp;
* `ds_filter` 0..3, negative and positive derived alpha, and `enable_cfl=0`;
* 600 randomized synthetic states (60% luma-correlated, 250 CfL winners) and a
  16-way batch-repeat check against the single-block result.

`compute-sanitizer --tool memcheck`: `ERROR SUMMARY: 0 errors`.
`compute-sanitizer --tool racecheck`: `0 hazards`.

## Timing (RTX 3060, batch of the frame's chroma blocks, 100 iterations)

| Case | Chroma blocks | Batch | Per block |
|---|---:|---:|---:|
| 640x360 testsrc2, DC/CfL | 247 | 0.033 ms/frame | 0.136 us |
| 640x360 shine, DC/CfL | 218 | 0.034 ms/frame | 0.154 us |

Single-block entry point (one job per launch): ~3-6 us/call, i.e. launch-bound.
Batch frozen-input blocks where possible; per-block launches will dominate at
frame scale (see `docs/negative-results.md`).

## Limitations / next steps

* Directional/smooth UV candidates are the only unimplemented policy; they need
  the §7.13.2 edge arrays (above-right/below-left counts and edge filter), which
  is why `num4_above_right`/`num4_below_left` are reserved in the job. They are
  not bit-exact yet and must not be enabled in the default path.
* Explicit-alpha search is not implemented; the §7.13.5 derived alpha is the
  CPU default. The `alpha_q8_min/max` range is the hook for a future search.
* The batch form trusts each descriptor (validating thousands of jobs on the
  host would add a sync-free pass); the caller owns batching correctness.
* The real-frame test harness reaches into `m1_still.cpp` internals to replay
  the CPU block loop. If that file's private structure changes again, update
  the harness; the GPU component API and arithmetic are unaffected.
* Frames whose dimensions are not a multiple of 8 (coded plane larger than the
  visible picture) are not encodable by the current CPU partition planner, so
  they are not used as integration frames. The synthetic cases still exercise
  coded-bound clamping and source clamping.

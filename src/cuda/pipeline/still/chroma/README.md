# M1 GPU chroma decision + prediction

GPU port of the M1 chroma half of `src/encoder/m1/still_encoder.cpp`:
`choose_uv_mode`, the DC prediction that both candidates use, `make_cfl_block`,
the §7.13.5 derived alpha, the winning prediction, and the optional full-UV
directional/smooth/Paeth search (`StillOptions::uv_full`).

This directory is self-contained; it does not modify `src/cuda/m1/` root
files, `src/main.cpp`, `CMakeLists.txt`, or `src/encoder/`.

Status:

* Default policy `{DC, CfL}`: CPU-parity verified on an RTX 3060 (`sm_86`,
  CUDA 12.8) for synthetic states and 64x64 / 640x360 frames.
* Full-UV policy (`enable_uv_full` / `enable_uv_directional`): wired and
  parity-tested against the CPU `uv_full` path by a **host emulation** of the
  exact device search. 4000 randomized blocks match on decision, mode/delta,
  alphas, all SSE diagnostics, and every prediction byte; the real-frame
  harness matches `choose_uv_mode()` on 64x64, 640x360 and 1920x1080
  (648 / 1254 / 4803 checked blocks, 0 failures). The on-device test now runs
  the same cases; run it on the Linux + CUDA box for the on-GPU result line.

## Policy

```
default  candidates = { DC (U-only score), CfL (U+V score) }
uv_full  candidates = 13 base modes (U+V score), then CfL (U+V score)
```

The full search is a transcription of `choose_uv_mode()` when
`options_.uv_full` is true:

* every mode is scored on **U+V** (DC included);
* `AngleDeltaUV` is derived from `AngleDeltaY` only when the candidate mode
  equals a directional `YMode` (§6.19.5.6); there is no UV angle-delta syntax;
* `filter_type_above/left` are the §7.13.2.15/.16 smooth-neighbour flags from
  the chroma `uv_smooth_` grid; edge filtering uses the sequence flag;
* DC is evaluated first, CfL last, strict `<` tie-break, so ties keep the
  earlier candidate;
* `enable_cfl = 0` with `uv_full` still runs the 13-mode search (CPU parity;
  it is not the old DC early-out);
* `enable_uv_directional` is a legacy alias for the same full search.

The default policy keeps the CPU quirks exactly: DC is scored on U only, CfL
on U+V, and a tie keeps DC. Do not "fix" that here; it changes M1 decisions.

## API

```cpp
#include "cuda/m1/chroma/still_chroma.h"
using namespace av2enc::cuda::still;

StillChromaJob job;        // device pointers + block geometry
StillChromaOptions opt;    // frame policy
StillChromaResult* d_res;  // device memory

chroma_decide(job, opt, d_res, stream);                  // one block
chroma_decide_batch(d_jobs, n, opt, d_results, stream);  // one CUDA block/job
```

Both entry points are asynchronous, allocate nothing, and never synchronize.
The batch form validates the options but not every descriptor; the caller owns
batching correctness.

`StillChromaJob` inputs:

* `luma`, `luma_stride`, `luma_max_x/y`, `sb_top_y` — reconstructed luma with
  the CPU `make_cfl_block()` bounds: `luma_max = MiCols*MI_SIZE - 1` (coded
  plane, not visible), `sb_top_y = (chroma_y*2/64)*64`.
* `chroma_u/v`, `chroma_stride` — reconstructed chroma. Only the above row and
  left column are read; the current block is ignored.
* `src_u/v`, `src_stride_*`, `src_max_x/y` — visible source planes.
* `x`, `y`, `w`, `h` — chroma rect (`w`,`h` power of two, 4..64).
* `coded_width/height` — `MiCols*2`, `MiRows*2`.
* `pred_u/v`, `pred_stride` — optional prediction planes.
* `num4_above_right`, `num4_below_left` — §7.13.2.1 availability in chroma 4x4
  units. Required only for the full-UV search; the device applies the chroma
  >32 rule itself.
* `y_mode`, `y_angle_delta`, `filter_type_above`, `filter_type_left`,
  `enable_edge_filter` — full-UV inputs. `filter_type_*` are 0/1 smooth flags,
  `enable_edge_filter` mirrors the sequence flag. Malformed values make the
  entry point return `cudaErrorInvalidValue` instead of silently diverging.

`StillChromaResult` returns `winner` (`DC`, `CfL`, or `UvDirectional`), `uv_mode`,
`angle_delta`, `is_cfl`, Q8 `alpha_u/v`, DC values, per-plane SSE
(`sse_dc_u/v`, `sse_cfl_u/v`), and `sse`. For full UV, `sse` is the winning
candidate's U+V SSE (including a DC or directional winner), matching
`UvChoice.sse`.

CfL derived alpha is signed (`[-511, 511]` Q8). Negative alphas are normal and
were observed in every real frame test.

## Implementation

`still_chroma_device.cuh` contains the device body (DC, CfL arithmetic, full-UV
search) so the kernel and the host emulation test run the same code.

`still_chroma.cu` launches one CUDA block of 128 threads per chroma block:

1. thread 0 computes the U/V DC values, §7.13.5 `lumaAvg`/lines and the
   per-plane derived alpha;
2. the default path scores DC (U) and CfL (U+V) and reduces with warp
   shuffles; the full path predicts all 13 candidates with the shared
   `src/cuda/intra_modes/intra_predict_device.cuh` predictor, scores U+V, then
   CfL, and re-predicts the winner into `pred_u/v`;
3. thread 0 applies the strict tie-break and writes `StillChromaResult`;
4. all threads write the winning prediction.

All arithmetic is integer; the CfL helpers in `src/encoder/cfl/cfl_common.h`
are shared with the CPU reference. No FP contraction risk.

## Build / test

On the Linux + CUDA box:

```sh
./src/cuda/m1/chroma/build_still_chroma_test.sh        # sm_86, build + run
./src/cuda/m1/chroma/build_still_chroma_test.sh --no-run
ARCH=sm_120 ./src/cuda/m1/chroma/build_still_chroma_test.sh
```

The test script generates 64x64, 640x360 and 1920x1080 frames (`testsrc2`, and
real content when `testdata/shine_1280x720_60f.y4m` is present), then runs the
DC/CfL, full-UV, DC-only, and (for 64x64) full-intra + full-UV frame cases.
The frame harness replays `still_encoder.cpp`'s block loop and compares against an
independent `encode()` run.

Without CUDA, the exact device search can be checked against the CPU
components:

```sh
./src/cuda/m1/chroma/build_uv_full_emulation_test.sh
# chroma_uv_emulation_test: PASS (4000 checks)

./src/cuda/m1/chroma/build_frame_emulation_test.sh \
    --y4m frame_64x64.y4m --y4m frame_640x360.y4m --y4m frame_1920x1080.y4m
# total blocks=... failures=0 (decision=0 alpha=0 sse=0 pred=0)
# RESULT: PASS
```

The first test compiles `still_chroma_device.cuh` with CUDA keywords stubbed and
one emulated thread, builds the CPU expectation from the real `intra` and
`cfl` components, and compares decision/SSE/prediction. The second compiles
the real `still_chroma_test.cu` as C++, replays the actual `still_encoder.cpp` block
loop through a shim that calls the same device body, and covers all frame
cases (DC/CfL, full UV, DC-only, and full-intra + full UV).

Measured emulation totals:

| Frame | blocks | CfL | uv-full blocks | failures |
|---|---:|---:|---:|---:|
| 64x64 | 648 | 274 | 12 | 0 |
| 640x360 | 1254 | 499 | 218 | 0 |
| 1920x1080 | 4803 | 1927 | 1401 | 0 |

## Verified results (RTX 3060, `sm_86`, CUDA 12.8) — default policy

```
batch/policy checks   ok
synthetic            blocks=600 cfl=250 failures=0
64x64 testsrc2_64x64.y4m dc/cfl        blocks=13  cfl=5   neg-alpha=12   recon-harness=exact
640x360 testsrc2_640x360.y4m dc/cfl    blocks=247 cfl=61  partial=22 sb-edge=183 sb-top>0=198 recon-harness=exact
640x360 testsrc2_640x360.y4m dc-only   blocks=247 cfl=0   partial=22 sb-edge=183 sb-top>0=198 recon-harness=exact
total blocks=1580 cfl=462 failures=0
RESULT: PASS
```

`compute-sanitizer --tool memcheck`: 0 errors; `racecheck`: 0 hazards.

## Full-UV quality at QP 90 (CPU reference, dav2d-decoded)

`shine_1280x720` frame scaled to the test size, luma DC only, `--still --qp 90`:

| Size | tile bytes base → uv_full | chroma SSE base → uv_full | PSNR U/V | overall PSNR |
|---|---:|---:|---:|---:|
| 64x64 | 674 → 669 | 3086 → 3045 | 47.38/45.52 → 47.50/45.53 | 44.615 → 44.628 (+0.013 dB) |
| 640x360 | 13193 → 13225 | 55552 → 56118 | 52.22/50.54 → 52.15/50.52 | 46.389 → 46.384 (-0.005 dB) |
| 1920x1080 | 51460 → 51448 | 376889 → 376980 | 51.89/53.27 → 51.87/53.29 | 48.7589 → 48.7588 (-0.0001 dB) |

Verdict: roughly rate-neutral and not a reliable net win. 64x64 gains a little
quality at a slightly smaller tile; 640x360 loses ~0.1 dB U/V for +32 bytes;
1080p is within noise (+91 chroma SSE, -12 bytes). Greedy per-block U+V SSE
with changed reconstructed neighbours can cascade worse than the default, so
"full search" is not monotone in frame SSE. Default off is the right policy.

## Limitations

* Full-UV is SSE-only (no rate term), like the CPU `uv_full` path.
* Explicit-alpha search is not implemented; the derived §7.13.5 alpha is used.
* `force_dc` is not a separate job flag. Callers that want the CPU
  `force_dc` early-out must keep `enable_uv_full = 0`; combining `force_dc`
  with `uv_full` has no GPU equivalent in this ABI.
* The batch form trusts each descriptor.
* Frames whose dimensions are not a multiple of 8 are not encodable by the
  current CPU partition planner and are not used as integration frames.

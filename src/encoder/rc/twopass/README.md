# Two-pass, lookahead and VBV/HRD rate control

Status: **policy component only.** This directory is not wired into the root
`CMakeLists.txt`, `src/main.cpp`, the M1/M2 frame loop, or the bitstream
writer. It produces decisions, not bits. Nothing here changes a reference
stream or the AV2 syntax.

It is a layer on top of `src/encoder/rc/` (`rc::RateController`,
`estimate_frame_bits()`, `plan_delta_q()`, `effective_qindex()`). It adds the
pieces that the single-pass controller deliberately leaves out:

- first-pass complexity analysis and two-pass bit allocation;
- a bounded lookahead window with online re-planning;
- a leaky-bucket VBV model with underflow prevention and deterministic
  reallocation;
- CRF/VBR target tracking with per-frame, per-tile and per-superblock QP
  assignment;
- correct interaction with delta-Q, segmentation and per-tile
  `CurrentQIndex` resets.

Normative references: AV2 Bitstream & Decoding Process Specification v1.0.0
§5.18.6.1 (`base_q_idx`), §5.18.7.1 (`segmentation_params`), §5.18.7.8
(`delta_q_params`), §5.20.1 (tile reset), §5.20.2.1 (ReadDeltas),
§5.20.5.11 (`read_delta_qindex`), §6.17.2 (coefficient-CDF slice) and
§7.14.2 (`get_qindex`/`qlookup`).

## Files

| File | Purpose |
|---|---|
| `twopass.h` / `twopass.cpp` | Public API and implementation |
| `first_pass_luma()` | Deterministic block-DC first pass over one luma plane |
| `twopass_test.cpp` | Synthetic trace, buffer, allocation, tile and determinism tests |
| `twopass_real_test.cpp` | Real 2-frame and 8-frame 720p M2 encodes |
| `build_and_test.sh` | Standalone build + test driver |

## Build and test

```sh
cd src/encoder/rc/twopass
./build_and_test.sh                  # synthetic + real 720p (when testdata exists)
TWOPASS_REAL=0 ./build_and_test.sh   # synthetic only
TWOPASS_SANITIZE=1 ./build_and_test.sh
TWOPASS_WERROR=1 ./build_and_test.sh
CXX=g++-16 ./build_and_test.sh
```

The synthetic suite links only `twopass.cpp` and `../rc.cpp`. The real suite
links the unmodified M2 CPU reference frame loop (`src/encoder/m2/`) and reads
`testdata/shine_1280x720_60f.y4m`. It is skipped when the clip is absent.
`DAV2D` is auto-detected from `tools/bin/` for a decode check.

## API overview

```cpp
using namespace av2enc::rc;
using namespace av2enc::rc::twopass;

// First pass, one record per frame. Complexity is the residual of a cheap
// prediction (first_pass_luma) or a probe encode; tiles may be empty for a
// single tile covering the frame.
std::vector<FrameStats> pass1 = ...;

TwoPassConfig config;
config.rc.mode = RateControlMode::kVbr;
config.rc.vbr.target_bitrate_bps = 500000.0;
config.rc.vbr.frame_rate_num = 60;
config.rc.vbr.frame_rate_den = 1;
config.rc.estimator = calibrated_estimator;   // optional
config.vbv.enabled = true;
config.vbv.buffer_frames = 4.0;
config.enable_delta_q = true;
config.delta_q_res = 1;
config.spatial_qp_strength = 8;

TwoPassController rc;
std::string error;
if (!rc.configure(config, &error)) { /* invalid config */ }
if (!rc.analyze(pass1, &error)) { /* invalid stats */ }

for (std::size_t i = 0; i < pass1.size(); ++i) {
  FramePlan plan;
  if (!rc.begin_frame(plan, &error)) { /* window exhausted / invalid */ }
  // ... encode frame i using plan.base_q_idx and plan.tiles ...
  if (!rc.end_frame(actual_payload_bits, &error)) { /* ... */ }
}
```

`actual_payload_bits` is the coded frame payload (header + tile data, not
container bytes). Feeding bytes here is a 8x rate error.

Streaming use replaces `analyze()` with `set_lookahead(window, &error)`.
`window[0]` must be the next uncoded frame (index `rc.frame_count()`); call it
again whenever the visible window changes. The window is copied and may be
larger than `config.lookahead_frames`, which caps how many frames are planned.

## First pass and allocation

`FrameStats` carries:

- `complexity`: whole-frame residual MAD/MSE (`FrameComplexity`);
- `first_pass_bits`: optional measured probe payload size;
- `sb_size`: superblock size (64 for M1/M2; 128/256 accepted);
- `tiles`: per-tile extent, per-tile complexity and optional per-SB weights.

If `first_pass_bits > 0` the frame allocation weight is the probe size. This is
the preferred mode: it captures inter-frame cost, which a source-only spatial
measure cannot. Otherwise the weight is

```text
weight = pixels * max(sigma, 1e-6)^complexity_exponent
```

with `sigma = frame_complexity_sigma()`. Keyframes multiply by
`keyframe_weight`.

`allocate_frames()` then:

1. normalizes the tile map (full cover, no overlap, no out-of-frame SB);
2. water-fills `total_bits` proportional to weights, with per-frame
   `[min_frame_bits_ratio, max_frame_bits_ratio] * average` clamps;
3. runs a forward VBV pass over the window and caps frames that would
   underflow;
4. redistributes the freed budget to later frames with headroom (never to an
   earlier frame whose bitstream may already be on the wire);
5. reports `underflow_prevented`, `underflow_feasible`, `overflow_feasible`,
   `reallocations`, `dropped_bits` and the fill trajectory.

In `analyze()` the window is the whole clip. In `set_lookahead()` the window is
the supplied prefix (capped by `lookahead_frames`) and the budget is the
nominal rate times that window, plus a bounded buffer credit (or debit) for the
current fill relative to `initial_fill`. That makes an underspent streaming
window recover in the next one instead of persisting. `analyze()` stays
available for offline two-pass; both paths are deterministic.

## VBV / HRD model

This is an **internal encoder-side leaky bucket**, not a claim of decoder HRD
conformance. There is no CPB/DPB syntax, no padding frame and no frame drop.

```text
size   = buffer_size_bits > 0 ? buffer_size_bits : buffer_frames * avg_frame_bits
drain  = max_rate_bps > 0 ? max_rate_bps * frame_duration : avg_frame_bits
low    = underflow_margin * size
high   = size - overflow_margin * size
fill_0 = clamp(initial_fill * size, low, high)

After frame i:
  fill' = fill + drain - bits_i

Underflow prevention: bits_i <= fill + drain - low
Overflow diagnostic:  fill' <= high (cannot be fixed without padding/drop)
```

`vbv_apply()` counts underflow/overflow events and clamps the fill back into
`[low, high]` so planning continues from a defined state. The controller uses
the modelled estimate for planning and the real payload bits after encoding;
an estimator error can still produce an event, which is why the rate-feedback
EMA and the deterministic reallocation of the remaining window exist.

## CRF and VBR target tracking

- **VBR**: the per-frame budget is the allocation target, capped by
  `vbv_max_bits()`. The controller bisects `estimate_frame_bits()` for the
  smallest qindex whose corrected estimate fits. `target_reachable` is false
  when even `max_qindex` misses. A slew limit
  (`max_qindex_step`) smooths the trajectory; an underflow cap may override
  the slew limit but never the qindex range.
- **CRF**: the qindex comes from the existing `rc::RateController` CRF curve
  (including `qcomp`). The two-pass layer only overrides it upward when the
  VBV cap requires it, reporting `vbv_underflow_prevented`. The nominal
  `target_bitrate_bps` is the buffer clock/allocation budget in CRF mode; it
  does not otherwise change the CRF mapping.
- **CQP**: the configured qindex is used, with the same VBV rescue.
- **Rate feedback**: `actual / estimated` is clamped and folded into an EMA
  (`rate_feedback_weight`, `rate_feedback_clamp`). It scales every estimate
  used for QP selection and buffer checks. Setting the weight to 0 disables it.

Keyframes reset the slew trajectory; they are independent of the previous
frame.

## Per-frame, per-tile and per-superblock QP

`FramePlan` contains the frame header decision and a `TilePlan` per tile:

```cpp
struct FramePlan {
  int base_q_idx;                 // frame header
  int coeff_cdf_qctx;             // always from base_q_idx
  double lambda_sse;
  double allocation_bits, target_bits, estimated_bits, rate_feedback;
  DeltaQParams delta_q;
  SegmentationParams segmentation;
  bool enable_ext_seg;
  SegmentationMap segmentation_map;   // 4x4 MI grid, when segmentation is on
  std::vector<TilePlan> tiles;
  // diagnostics: vbv_fill_before/after_estimate, underflow_prevented,
  // overflow_risk, target_reachable, clamped_by_slew/range, spatial_clamped
};

struct TilePlan {
  int sb_cols, sb_rows, sb_col_start, sb_row_start;
  int target_qindex;                    // desired tile representative
  std::vector<int> sb_target_qindex;    // desired effective qindex per SB
  std::vector<int> sb_achieved_qindex;  // after delta-Q/segmentation clip
  std::vector<std::uint8_t> sb_segment_id;
  DeltaQPlan delta_q;                   // per tile, starts at base_q_idx
};
```

Base qindex selection uses the **sum of per-tile estimates** at that qindex,
not one frame-average sigma. This keeps the frame budget consistent with a
spatially varying QP plan: a non-uniform frame really does cost more than its
average-sigma estimate.

Spatial assignment is a bounded, deterministic offset:

```text
tile_offset = -strength * clamp(log2(tile_weight / mean_tile_weight) / 2, -1, 1)
sb_offset   = -strength/2 * clamp(log2(sb_weight / mean_sb_weight), -1, 1)
target_q    = clamp(base_q + tile_offset + sb_offset, min_qindex, max_qindex)
```

Higher activity gets a *finer* quantiser. The pixel-weighted mean tile offset
is removed so the base qindex is not shifted. If the summed spatial estimate
exceeds 102% of the base cost (or the hard VBV cap), the offsets are halved
deterministically until they fit; `spatial_clamped` reports it.

### Delta-Q

With `enable_delta_q`, each tile is planned independently with
`rc::plan_delta_q()` and `delta_q_res`. Every plan's first step has
`qindex_before == base_q_idx`: the decoder resets `CurrentQIndex` at each tile
(§5.20.1), and the controller never carries state across a tile. With
`delta_q_res > 0`, achieved qindex can differ from the target by at most
`2^res - 1`.

### Segmentation

With `enable_segmentation`, the controller assigns each SB to a class and emits
`SEG_LVL_ALT_Q` values on a symmetric ladder:

```text
segment 0: alt 0 (feature disabled)
segment k (1..L):       -step * k
segment L+k (1..L):     +step * k
```

`segment_levels` is `L`; the segment count is `2L+1`, limited to 8 without
`enable_ext_seg` and 16 with it. `update_map = true`, `temporal_update = false`.
`segmentation_map` is materialized on the frame's 4x4 MI grid (16 MI per 64x64
SB), so the block loop can use it directly; `TilePlan::sb_segment_id` is the
same information on the SB grid.

When **both** delta-Q and segmentation are enabled, §7.14.2 makes ALT_Q
relative to `CurrentQIndex`:

```text
effective = clamp(CurrentQIndex + ALT_Q, 0, 255)
```

The controller therefore plans `CurrentQIndex = target_effective - ALT_Q`.
The returned delta plan is already compensated; do not add ALT_Q again in the
encoder. `TilePlan::sb_achieved_qindex` is the predicted effective value, and
the tests verify it against `rc::effective_qindex()` for every SB, including
`ignoreDeltaQ = true`.

## M2 frame-loop integration

The exact second-pass call sequence M2 should use:

```cpp
#include "encoder/rc/twopass/twopass.h"

// 1. Configure once.
av2enc::rc::twopass::TwoPassConfig cfg;
cfg.rc.mode = av2enc::rc::RateControlMode::kVbr;
cfg.rc.vbr.target_bitrate_bps = target_bps;
cfg.rc.vbr.frame_rate_num = 60;
cfg.rc.vbr.frame_rate_den = 1;
cfg.rc.estimator = estimator;      // calibrated, or default model
cfg.vbv.enabled = true;
cfg.vbv.buffer_frames = 4.0;
cfg.enable_delta_q = true;
cfg.delta_q_res = 1;
cfg.spatial_qp_strength = 8;

av2enc::rc::twopass::TwoPassController rc;
if (!rc.configure(cfg, &error)) return fail(error);
if (!rc.analyze(pass1_stats, &error)) return fail(error);
// or, for streaming:
//   rc.set_lookahead(next_frames_stats, &error);

// 2. Per frame, after the first-pass data for this frame is known.
av2enc::rc::twopass::FramePlan plan;
if (!rc.begin_frame(plan, &error)) return fail(error);

// 3. Use the decision.
M2Options options = base_options;
options.base_q_idx = static_cast<std::uint8_t>(plan.base_q_idx);
M2Config config = base_config;
config.base_q_idx = plan.base_q_idx;

m2::deltaq::FrameQpPlan qp;
qp.base_q_idx = plan.base_q_idx;
qp.delta_q = plan.delta_q;
qp.segmentation = plan.segmentation;
qp.enable_ext_seg = plan.enable_ext_seg;

// Frame header: write segmentation_params() / delta_q_params() from
// plan.segmentation and plan.delta_q (m2::deltaq::write_frame_qp_syntax).
// Coefficient CDFs: init from plan.coeff_cdf_qctx (raw base_q_idx).
// RDO: plan.lambda_sse.
// Tile: m2::deltaq::begin_frame(ctx, qp, header, mi_cols, mi_rows, ...);
//       m2::deltaq::begin_tile() inside M2 must start at plan.base_q_idx.
// Per SB delta: m2::deltaq::write_superblock_delta_q(ctx, encoder,
//                 plan.tiles[t].delta_q.steps[sb], first_block, skip_flag).
// Per block segment id: plan.tiles[t].sb_segment_id[sb] (or the frame-wide
//       plan.segmentation_map on the 4x4 MI grid).

// 4. Encode the frame, then report the real coded payload bits.
if (!rc.end_frame(actual_payload_bits, &error)) return fail(error);
```

Notes:

1. **One tile today.** M2 emits one tile per frame, so `plan.tiles` has one
   entry. The API supports multiple tiles; call `begin_tile` per tile and use
   `plan.tiles[t]` for tile `t`. Delta-Q plans are per tile and all start at
   `plan.base_q_idx`.
2. **`M2Options` has no per-frame QP field.** The loop must construct the frame
   encoder (and `M2Config`) for that frame with `plan.base_q_idx` instead of a
   sequence-wide value. The keyframe uses `plan.base_q_idx` too.
3. **Coefficient CDFs** come from `plan.coeff_cdf_qctx`, which is derived from
   the raw `base_q_idx`. Never derive it from a delta-Q or segment qindex.
4. **`end_frame()`** takes coded payload bits. Use tile payload + frame header
   (or the frame OBU payload), not the IVF record length in bytes.
5. **Ordering.** Exactly one `begin_frame()` and one `end_frame()` per frame.
   `set_lookahead()` and `analyze()` cannot run while a frame is open.
6. **First pass.** `first_pass_luma()` is a spatial fallback. For clip-level
   allocation prefer a probe encode: set `FrameStats::first_pass_bits` to the
   probe payload size and fill per-tile `FrameComplexity` for spatial QP.
7. **No syntax.** This component does not write segmentation, delta-Q, frame
   headers or CDFs. The existing `m2::deltaq` adapter owns those bits.

## Determinism

All decisions are pure arithmetic over the config, the first-pass vector and
the actual-bit history. There are no timers, RNGs, threads or hidden caches.
The estimator hook must be pure for a given `(context, complexity, qindex)`.
Build with `-ffp-contract=off` when comparing across hosts; libm `log2`/`sqrt`
can differ in the last ulp. The test suite compares every decision field
(including doubles) across repeated runs and finds bit-identical plans.

## Tests

`twopass_test.cpp` covers:

- VBV: size/drain/margins, max/min spend, underflow/overflow events, disabled
  model, invalid inputs, nominal frame duration/bits.
- Allocation: water-filling, monotone weights, flat clips, min/max clamps,
  VBV caps, later-frame redistribution, dropped budget, infeasible windows,
  bit-identical repeat runs.
- Lookahead: windowed versus offline allocation, horizon cap, window
  exhaustion, re-`set_lookahead`, sequence errors, reset, and the streaming
  buffer credit/debit.
- VBR/CRF: built-in and fitted estimators, target tracking, CRF endpoints and
  `qcomp`, VBV override, slew limit, rate-feedback EMA compensation.
- Per-tile: delta-Q per tile with the base reset, `delta_q_res` error bound,
  segmentation-only ALT_Q, both mechanisms verified through
  `rc::effective_qindex()` (`ignoreDeltaQ` on and off), per-SB activity
  direction, zero-strength uniformity, invalid tile geometry.
- First pass: synthetic smooth/checkerboard plane, tile moments, stride
  padding, determinism, invalid geometry.
- Config validation, a 10-frame full-plan determinism run, and estimator
  failure recovery (a failed plan rolls the inner CRF frame back).

Local result: **2,586 checks, 0 failures** with Apple clang 21 and
Homebrew g++ 16, `TWOPASS_WERROR=1`, and ASan + UBSan clean.

`twopass_real_test.cpp` performs real M2 encodes of
`testdata/shine_1280x720_60f.y4m`:

- 8 frames at QP 60/120/180/220 and 2 frames at QP 90/180/220;
- a monotone rate curve built from the real payload sizes, used as the
  calibrated estimator and the per-frame "actual" at the chosen qindex;
- QP 120 is the probe first pass (`first_pass_bits`);
- 8-frame VBR with a 6-frame VBV, 2x2 delta-Q tiles and segmentation:
  target 94,782 bits, actual 101,267 bits (+6.8%), no underflow;
- 2-frame VBR: target 283,680 bits, actual 274,977 bits, no underflow;
- CRF 8-frame run (`crf=40`, `qcomp=0.5`): q range [162, 162] on this spatially
  uniform clip, deterministic;
- a tight-buffer CRF run forces underflow prevention;
- one 2-frame stream is decoded by `dav2d` (exit 0).

Local result: **192 checks, 0 failures**, ~34 s without sanitizers (~3:15 with
ASan). The real test's per-frame "actual" is the log-linear interpolation of
the measured anchor points at the planned qindex; the anchors themselves are
real encodes, and the decoded stream proves the M2 path is producing a valid
bitstream. Do not read the interpolation as a quality result.

## Limitations and next steps

- No motion-compensated first pass. Use a probe encode (`first_pass_bits`) for
  clip-level allocation; `first_pass_luma()` only measures spatial activity.
- The VBV is an internal model, not HRD conformance. No padding, dropping,
  CPB/DPB timing or buffering-period syntax.
- No GOP/pyramid/reorder support: the window is assumed to be encoded in
  display order. Add per-frame dependency metadata before B-pyramids.
- Segmentation is a scalar ALT_Q class ladder; no temporal segment prediction
  (`temporal_update=0`) and no SKIP/GLOBALMV features.
- Per-SB achieved qindex is quantized by `delta_q_res` (or the segment ladder).
  `sb_achieved_qindex` reports the predicted result; the block encoder must use
  the same value.
- The rate model is still the `rc::` placeholder unless an estimator is
  installed. No BD-rate or quality claim follows from this component.
- Multi-tile plans are implemented and tested, but M2 currently emits one tile
  per frame. Wire the per-tile calls when multi-tile M2 lands.

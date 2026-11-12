# Rate control: CQP / CRF / VBR, QP slices, segmentation and delta-Q planning

Status: **component only.** This directory is not wired into the root
`CMakeLists.txt`, `src/main.cpp`, `src/encoder/basic_encoder.cpp`, the entropy coder,
or the bitstream writer. It produces no bits and cannot change a reference
stream. The integration pass owns wiring these decisions into the frame header
and block syntax.

Scope:

- CQP, CRF and VBR per-frame rate-control models.
- `base_q_idx` / QP selection and per-frame updates.
- Coefficient-CDF quantizer-context slices ("QP slices") from `base_q_idx`.
- Segmentation and delta-Q **planning data structures**, validated against the
  AV2 v1.0.0 semantics, with no bitstream syntax implementation.
- Bitrate and frame-complexity estimation hooks with a deterministic default
  model.
- Deterministic tests on synthetic frame complexity and target bitrates.

Not in scope: entropy coding, OBU/frame-header writing, CUDA kernels, VBV/HRD,
lookahead, two-pass, per-block QP search, or any quality/BD-rate claim.

Per `docs/architecture.md`, rate control is **CPU-owned per-frame policy**, not a
per-block GPU stage. There is therefore no `src/cuda/rc/` in this change. The
controller is pure arithmetic and allocates nothing per block; a GPU port would
add launch overhead without moving pixel work.

Files:

| File | Purpose |
|---|---|
| `rc.h` / `rc.cpp` | Public API and implementation |
| `rc_tables.h` | Generated quantizer tables (do not hand-edit) |
| `generate_rc_tables.py` | Regenerates `rc_tables.h` from `spec/tables/quant.json` |
| `rc_test.cpp` | Standalone deterministic tests |
| `build_and_test.sh` | Standalone build + test driver |

## Build and test

```sh
cd src/encoder/rc
./build_and_test.sh
RC_SANITIZE=1 ./build_and_test.sh   # ASan + UBSan
RC_WERROR=1 ./build_and_test.sh     # warnings are errors
```

No CMake, no external dependencies. The script uses `CXX` if set, otherwise
`c++`, and compiles with `-Wall -Wextra -Wpedantic -Wconversion -Wshadow
-ffp-contract=off`.

Regenerate the tables after regenerating the spec tables:

```sh
python3 src/encoder/rc/generate_rc_tables.py
```

`rc_tables.h` is generated from `spec/tables/quant.json` (§7.14.2). Do not
hand-type quantizer tables.

## Units and conventions

- **`base_q_idx` / qindex:** integer `0..255` for 8-bit streams. `0` is lossless
  and outside the rate-control domain; `QpDecision` clamps to `1..255`.
- **`qp`:** a continuous quantizer level used internally by CRF/VBR. It is
  rounded to `base_q_idx` with deterministic round-half-up.
- **Rate:** bits per second for targets; bits per frame for estimates.
- **Lambda:** squared-error units per bit. `lambda_sse_from_qindex()` uses the
  normative `qlookup` step and the same `sse_scale` placeholder as
  `src/encoder/rdo/`.

## Quantizer domain and QP slices

```cpp
double step = quant_step_from_qindex(90);        // normative qlookup -> 528
int    q    = qindex_from_qp(90.5);              // 91 (round half up)
int    base = clamp_base_qindex(0);              // 1

int ctx = coeff_cdf_qctx(90);                    // 0
const QpSlice& slice = coeff_cdf_slice(181);     // index 2, [141, 190]
```

`kCoeffCdfQpSlices` covers `0..255` contiguously with the §6.17.2 slice
boundaries inferred in `docs/open-questions.md` B1:

| Slice | `base_q_idx` | Coefficient-CDF `qctx` |
|---:|---|---:|
| 0 | 0–90 | 0 |
| 1 | 91–140 | 1 |
| 2 | 141–190 | 2 |
| 3 | 191–255 | 3 |

**Trap:** the coefficient-CDF slice is selected from the frame **`base_q_idx`**,
not from the per-block effective qindex. Delta-Q and segmentation change the
quantizer used for coefficients but do **not** change which coefficient CDF
slice was initialized for the frame. Keep `QpDecision::coeff_cdf_qctx` and
`QpDecision::qp_slice` tied to `base_q_idx`.

## Complexity and bitrate estimation

```cpp
FrameComplexity c =
    complexity_from_u8_planes(original, predicted, width, height, stride);
// c.residual_mad, c.residual_mse are in 8-bit sample units; padding is ignored.
double sigma = frame_complexity_sigma(c);  // model/CRF residual scale

BitrateEstimator estimator;   // null callback = built-in placeholder model
double bits = 0.0;
std::string error;
if (!estimate_frame_bits(estimator, c, base_q_idx, bits, &error)) {
  // invalid complexity, invalid params, or an invalid hook result
}
```

The built-in model is a **calibration placeholder**, not a normative AV2 model:

```text
sigma = sqrt(residual_mse)              (or mad_to_sigma * residual_mad when mse == 0)
bits  = header_bits
      + pixels * min(max_bits_per_pixel,
                     coefficient_bits_per_sample * log2(1 + sigma / step))
step  = qlookup(clamp(base_q_idx, 1, 255))
```

It is monotone non-increasing in qindex by construction, so VBR can bisect it.
Calibrate `header_bits`, `coefficient_bits_per_sample`, and
`max_bits_per_pixel` before using the estimates for quality decisions.

An integration may replace the model with a measured or lookahead-driven
estimator:

```cpp
struct MyModel { /* ... */ };

double my_estimate(void* ctx, const FrameComplexity& c, int base_q_idx) {
  // Must be finite, non-negative, and at most 1e12 bits.
  return calibrated_bits(c, base_q_idx);
}

BitrateEstimator estimator;
estimator.context = &my_model;
estimator.estimate = &my_estimate;
```

Callbacks must not throw. A null `estimate` selects the built-in model.

## CQP

```cpp
RateControlConfig config;
config.mode = RateControlMode::kCqp;
config.cqp.base_q_idx = 90;

RateController rc;
if (!rc.configure(config, &error)) { /* invalid config */ }

QpDecision d;
if (!rc.begin_frame(complexity, d, &error)) { /* ... */ }
// d.base_q_idx, d.coeff_cdf_qctx, d.estimated_bits, d.lambda_sse
if (!rc.end_frame(actual_payload_bits, &error)) { /* ... */ }
```

CQP returns the configured `base_q_idx` every frame. `end_frame()` is still
required so frame bookkeeping and the rate EMA stay consistent.

## CRF

```cpp
config.mode = RateControlMode::kCrf;
config.crf.crf = 32.0;          // [0, 63], lower = better quality
config.crf.qindex_at_crf0 = 1.0;
config.crf.qindex_at_crf63 = 255.0;
config.crf.qcomp = 0.0;         // default: constant qindex
```

The CRF-to-qindex mapping is a **linear calibration placeholder** with explicit
endpoints, not a standard CRF curve. `qcomp` is off by default, so CRF is
constant quantizer and the bitrate varies with content. Enabling `qcomp`
requires `complexity_reference > 0` and applies a bounded per-frame adjustment:

```text
adjust = clamp(qcomp * 24 * log2(sigma / complexity_reference),
               -max_qindex_adjust, +max_qindex_adjust)
```

This is a knob for future calibration, not a claim of perceptual constant
quality.

## VBR

```cpp
config.mode = RateControlMode::kVbr;
config.vbr.target_bitrate_bps = 500000.0;
config.vbr.frame_rate_num = 30;
config.vbr.frame_rate_den = 1;
config.vbr.min_qindex = 1;
config.vbr.max_qindex = 255;
config.vbr.max_qindex_step = 8;   // per-frame slew limit
config.vbr.buffer_frames = 30.0;
config.vbr.buffer_gain = 0.5;
```

Per frame:

1. `target_bits = target_bitrate_bps * frame_rate_den / frame_rate_num`.
2. `adjusted = clamp(target_bits + buffer_gain * virtual_buffer,
                     0.25 * target_bits, 2.0 * target_bits)`.
3. Bisect the estimator for the smallest `base_q_idx` whose estimate fits
   `adjusted`; flag `target_reachable = false` when even `max_qindex` misses it.
4. Clamp the move to `max_qindex_step` from the previous frame; set
   `clamped_by_step`.
5. `end_frame(actual_bits)` accumulates
   `virtual_buffer += target_bits - actual_bits`, clamped to
   `+/- buffer_frames * target_bits`.

The virtual buffer is a deterministic proportional correction, **not** an
HRD/VBV model. VBV/HRD is deferred. There is no wall-clock input, no RNG, and no
lookahead in this component.

`begin_frame()` must be followed by exactly one `end_frame()`; dropping frames is
not supported. A failed `end_frame()` (for example negative `actual_bits`) leaves
the frame open so the caller can report a valid value. `configure()` validates
and resets all controller state (including any open frame); call it at
initialization or a deliberate mode change.

## Segmentation

`SegmentationParams` mirrors §5.18.7.1/§6.17.7.1: up to 16 segments, three
features (`ALT_Q`, `SKIP`, `GLOBALMV`), and the `update_map` / `temporal_update`
flags. `ALT_Q` is signed and clipped to `+/-351`; `SKIP` and `GLOBALMV` are
zero-bit flags and must carry data `0`.

```cpp
SegmentationParams seg = disabled_segmentation();
set_segment_alt_q(seg, /*segment_id=*/2, /*delta_q=*/-18);
if (!segmentation_params_valid(seg, &error)) { /* ... */ }
```

`SegmentationMap` is a row-major `segment_id` array on the 4x4 luma MI grid
(`mi_cols * mi_rows` entries, each `< 16`). The component validates the map but
does not code `segment_id` symbols or the segmentation map syntax.

## Effective qindex and delta-Q

`effective_qindex()` mirrors the decoder's `get_qindex(ignoreDeltaQ,
segmentId)` (§7.14.2):

```cpp
QpContext ctx;
ctx.base_q_idx = 90;
ctx.current_q_index = 70;          // CurrentQIndex for delta-Q
ctx.delta_q = {true, /*res=*/1};
ctx.segmentation = seg;

int qindex = 0;
effective_qindex(ctx, 2, /*ignore_delta_q=*/false, qindex, &error);
```

Semantics:

1. If `ALT_Q` is active:
   `q = CurrentQIndex + data` when delta-Q is present and not ignored;
   otherwise `q = base_q_idx + data`.
2. Else if delta-Q is present and not ignored: `q = CurrentQIndex`.
3. Else: `q = base_q_idx`.
4. Clamp to `[0, 255]`.

`delta_q_reset(base_q_idx, state)` sets `CurrentQIndex = base_q_idx`.
`delta_q_apply(params, reduced_delta, state)` performs the normative
`Clip3(1, 255, CurrentQIndex + (reduced_delta << res))` update from
`read_delta_qindex()`.

### Delta-Q planning

```cpp
DeltaQPlanConfig plan_config;
plan_config.base_q_idx = 90;
plan_config.delta_q_res = 1;
plan_config.cols = 4;
plan_config.rows = 2;

std::vector<int> targets = /* row-major, cols * rows, values in [1, 255] */;
DeltaQPlan plan;
plan_delta_q(plan_config, targets, plan, &error);
```

Each `DeltaQPlanStep` records the superblock position, coded `reduced_delta`,
qindex before/after clipping, and the plan reports `max_abs_reduced_delta` and
`final_qindex`. The planner picks the nearest reachable qindex for the given
`delta_q_res`.

Assumptions and limits:

- **Per tile.** The decoder sets `CurrentQIndex = base_q_idx` at every tile
  (§5.20.2.1). Plan each tile independently; never carry state across a tile
  start.
- **Superblock root.** The plan assumes the delta is coded at the superblock
  root, i.e. the block is not a fully skipped `MiSize == SbSize` block, where
  `read_delta_qindex()` is not called.
- **Lossless.** Target `0` is rejected: delta-Q clips to `1..255`. Express
  lossless segments with `ALT_Q = -base_q_idx` (or another value that drives
  `get_qindex()` to 0) through segmentation instead.
- **No syntax.** The plan does not encode `delta_q_abs`, the escape path, or
  sign bits; the integration pass owns the actual symbol coding.

## Integration notes

1. **Frame header.** Use `QpDecision::base_q_idx` for `base_q_idx` (§5.18.6.1).
   It is always in `[1, 255]`. `base_q_idx = 0` means lossless and must be
   requested through a lossless segment, not this controller. The M0
   regression stream pins `base_q_idx = 90`; any mode that moves it changes the
   pinned MD5 and needs its own decoder/reconstruction gate.
2. **Coefficient CDFs.** Call `init_coeff_cdfs()` with
   `QpDecision::coeff_cdf_qctx` from `base_q_idx`. Do not derive it from a
   delta-Q or segment qindex. **Important:** the current
   `src/encoder/basic_cdf_tables.h` is generated for `base_q_idx = 90` (qctx 0,
   see `src/encoder/generate_basic_cdf.py`). Until tables for qctx 1–3 exist, an
   M0 integration must keep `base_q_idx <= 90`; otherwise the entropy coder
   would use the wrong CDF slice.
3. **RDO.** Pass `QpDecision::lambda_sse` (or the qindex) to the RDO layer. Its
   step is normative; `sse_scale` remains the documented calibration
   placeholder. Keep the lambda domain consistent with the distortion metric.
4. **Segmentation.** `SegmentationParams` maps directly to the §5.18.7.1 fields;
   the integration pass still has to write `reuse_seg_info`, the feature
   enables/data, and the segment-id map syntax.
5. **Delta-Q.** If `delta_q_present = 1`, use `DeltaQPlan` per tile and emit
   `read_delta_qindex()` syntax in the planned order. `delta_q_res` is the
   config's `res`; it is not a qindex delta.
6. **VBR accounting.** Pass actual **payload bits** (not bytes, not OBU
   overhead) to `end_frame()`. The virtual buffer is internal; do not feed it
   into an HRD model.
7. **Determinism.** No timers, RNG, or hidden host state. Decisions are a pure
   function of config, complexity, and the actual-bits history. Build with
   `-ffp-contract=off` when comparing against a second host; `log2`/`sqrt` may
   differ in the last ulp across libm implementations.
8. **No quality claim.** Nothing here has been integrated into a stream, so no
   BD-rate or CVVDP result may be attributed to this component. The rate-control
   gate is a later step: decoded bitrate within ±5% of target, no decode
   failures, and no buffer underflow/overflow in whatever buffer model is
   actually implemented.

## Tests

`rc_test.cpp` covers:

- `qlookup` values, clamping, monotone step, `qp` rounding, lambda.
- All four coefficient-CDF slices and their boundaries/contiguity.
- Complexity from 8-bit planes, stride padding, invalid spans/dimensions.
- Model monotonicity in qindex and complexity, zero-residual header-only case.
- Custom estimator hooks, invalid (NaN/negative) hook results, invalid params,
  and a controller run driven by a custom hook.
- CQP decisions, end/begin ordering errors, negative actual bits, reconfigure.
- CRF endpoints, monotonicity across `crf`, `qcomp` on/off and its clamp.
- VBR target/quality direction, unreachable target, slew limit, virtual-buffer
  correction and clamp, determinism across two controllers.
- Segmentation validation, `ALT_Q` clipping, `SKIP`/`GLOBALMV` zero-data rule.
- `get_qindex()` semantics with and without delta-Q, `ignoreDeltaQ`, and clipping.
- Delta-Q state reset/apply/clipping and planner quantization, ordering,
  same-seed determinism, size/target validation, per-tile reset.
- Segmentation map validation.

Local result: **1479 checks, 0 failures**, with `RC_WERROR=1` under Apple
clang 21 and Homebrew g++ 16; ASan + UBSan clean. This is a CPU correctness
result on the macOS dev host, not a product speed or quality result.

## Known limitations / next steps

- The bitrate model and CRF mapping are calibration placeholders. Measure on a
  real encode before trusting them.
- No VBV/HRD, lookahead, two-pass, or per-frame complexity from motion/partition
  statistics; only the supplied `FrameComplexity` moments and optional hook.
- No segmentation-map generation heuristics; only validation.
- No delta-Q bitstream writing and no `reuse_seg_info` / sequence- or
  multi-frame-header segmentation reuse logic.
- VBR is tested against the synthetic model only. It is not an integration or
  conformance result.

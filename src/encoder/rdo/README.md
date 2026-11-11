# RDO / RDOQ framework (CPU reference)

Status: **component only.** This directory is not wired into the root
`CMakeLists.txt`, `src/main.cpp`, `src/encoder/m0_still.cpp`, the entropy coder or
the bitstream writer. It produces no bits and cannot change a reference stream.
The integration pass owns wiring it into the encoder.

Scope:

- Lagrangian cost (`distortion + lambda * rate`).
- Distortion hooks for SSD, SAD and SATD, plus coefficient-domain SSE.
- A deterministic candidate/mode selector for partition, intra, inter and
  transform decisions.
- A concrete rate-aware intra mode decision for the M1 full-intra search.
- Lambda selection from `base_q_idx` or a quantiser step.
- A limited RDOQ trellis for coefficient levels with a pluggable rate model.

Files:

| File | Purpose |
|---|---|
| `rdo.h` / `rdo.cpp` | Framework implementation |
| `intra_decision.h` / `intra_decision.cpp` | Rate-aware M1 intra mode decision |
| `rdo_test.cpp` | Standalone tests |
| `build_and_test.sh` | Standalone build + test driver |
| `experiment/` | Patch-neutral CPU experiment against the real M1 encoder |

## Build and test

```sh
cd src/encoder/rdo
./build_and_test.sh
RDO_SANITIZE=1 ./build_and_test.sh   # ASan + UBSan
```

No CMake, no external dependencies. The script uses `CXX` if set, otherwise
`c++`. It compiles with the project warning set
(`-Wall -Wextra -Wpedantic -Wconversion -Wshadow`) and `-ffp-contract=off`.

## Units and conventions

- **Distortion:** SSD/SAD/SATD sample-domain sums, or coefficient-domain SSE.
- **Rate:** bits, not bytes. A caller with byte costs must multiply by 8.
- **Lambda:** distortion units per bit.
- The lambda domain must match the distortion metric. Use
  `lambda_from_qp(q, metric)` or `make_rdo_config(q, metric)` and pass that same
  metric to `compute_distortion()`.

## Distortion

```cpp
std::uint64_t ssd_u8(a, b, width, height, stride);
std::uint64_t sad_u8(a, b, width, height, stride);
std::uint64_t satd_u8(a, b, width, height, stride);
double compute_distortion(metric, a, b, width, height, stride);
double coefficient_sse(original, reconstructed);
```

`a` and `b` are 8-bit planes with a row stride in bytes. Only the visible
`width x height` region is read. Partial 4x4 blocks at the right/bottom edge in
`satd_u8` are zero-padded. These are cost hooks, not normative transforms.

## Lambda

```cpp
LambdaParams params;                 // step_at_qp0, qp_per_octave, sse_scale
double step   = quant_step_from_qp(base_q_idx, params);
double lambda = lambda_sse_from_quant_step(step, params.sse_scale);
double lambda = lambda_from_qp(base_q_idx, DistortionMetric::kSsd, params);
```

`base_q_idx` is clamped to `[0, 255]`. The default model is a **calibration
placeholder**, not a normative AV2 quantiser table:

```text
step(q)   = step_at_qp0 * 2^(q / qp_per_octave)
lambda_sse = sse_scale * step^2
lambda_sad = lambda_satd = sqrt(lambda_sse)
```

For the M0/M1 CPU reference path, prefer passing the real quantiser step from
the transform/quant stage to `lambda_sse_from_quant_step()` rather than relying
on the QP model. `make_rdo_config(q, metric, tie_break, params)` accepts custom
`LambdaParams`; calibrate `sse_scale` for the actual distortion units before
using the numbers to claim a BD-rate change.

## Candidate selection

One candidate is already-costed data supplied by the calling tool:

```cpp
CandidateCost candidate = make_intra_candidate(mode, distortion, rate_bits);
// kind = kIntra, id = mode, aux = 0

CandidateCost inter =
    make_inter_candidate(mode, reference_index, distortion, rate_bits);
CandidateCost tx =
    make_transform_candidate(tx_type, distortion, rate_bits);
CandidateCost part =
    make_partition_candidate(partition_type, distortion, rate_bits);
```

Prefer the streaming selector in decisions that enumerate many modes. It keeps
only the best and second-best candidates and rejects invalid input (negative or
non-finite distortion/rate):

```cpp
RdoConfig config = make_rdo_config(base_q_idx, DistortionMetric::kSsd);
CandidateSelector selector(config);
for (const auto& mode : modes) {
  std::string error;
  if (!selector.add(make_intra_candidate(mode.id, mode.distortion,
                                         mode.rate_bits), &error)) {
    // invalid candidate; selector counted it in rejected_count
  }
}
const RdoDecision& decision = selector.decision();
if (decision.found) {
  use(decision.best.candidate.id);
}
```

`select_best()` is the non-streaming convenience wrapper:

```cpp
RdoDecision decision = select_best(candidates, config, &error);
```

### Tie-breaking

`candidate_better()` defines a total order:

1. lower total cost (`distortion + lambda * rate_bits`)
2. lower distortion, then lower rate (`TieBreak::kDistortionThenRate`, default)
   or lower rate, then lower distortion (`kRateThenDistortion`)
3. lower `CandidateKind` (`kPartition < kIntra < kInter < kTransform < kRdoq`)
4. lower `id`
5. lower `aux`
6. lower insertion index

Comparisons are exact; there is no epsilon. That is deterministic for a given
build. Build with `-ffp-contract=off` when comparing CPU and GPU decisions.
`RdoDecision::margin()` reports the best-to-second cost gap; a small margin is a
useful signal for deferred refinement. Transcendental functions in the lambda
model and the placeholder RDOQ rate model (`exp2`, `log2`) can differ across
libm implementations; pass explicit lambda/rate values when bit-exact
cross-host parity is required.

### Integration points

**Intra:** one candidate per mode, `id = intra mode`.
**Inter:** one candidate per mode/reference, `id = inter mode`, `aux = reference`.
**Transform:** one candidate per transform type/size, `id = tx type`,
`aux = tx size`.
**Partition:** recurse first, then aggregate the best child costs:

```cpp
CandidateCost none =
    make_partition_candidate(PARTITION_NONE, none_distortion, none_rate);
CandidateCost split = make_partition_split_candidate(
    PARTITION_SPLIT, child_best_costs, split_signaling_bits);

CandidateCost choices[] = {none, split};
RdoDecision d = select_best(choices, config);
```

`make_partition_split_candidate()` sums child distortion and rate and adds the
split signaling bits. The caller is responsible for including every syntax cost
that differs between the two choices.

## Intra mode decision (M1 full intra)

`intra_decision.h` is the concrete rate-aware intra selector. It exists because
the M1 full search is SSE-only: a mode that predicts better can still lose once
its mode/angle and coefficient bits are counted.

```cpp
#include "encoder/rdo/intra_decision.h"

struct IntraCandidateCost {
  std::int32_t mode;                 // AV2 YMode 0..12
  std::int32_t angle_delta;          // -3..3
  double sse;
  double syntax_rate_bits;           // mode + angle signalling
  double coefficient_rate_bits;      // residual coefficient delta
};

IntraRdoConfig config = make_intra_rdo_config(
    static_cast<double>(txv::kQuantStep),  // 528 in the current M1 path
    0.125,                                 // placeholder sse_scale
    true);                                 // rate_aware

IntraModeChoice choice = select_intra_mode(candidates, config);
```

`select_intra_mode()` is O(1) memory and deterministic:

1. lowest cost (`sse + lambda * (syntax + coefficient)` when rate-aware;
   `sse` alone when `rate_aware = false`)
2. lowest SSE
3. lowest rate (when rate-aware and `prefer_lower_rate_on_tie`)
4. fixed AV2 candidate order: all 13 base modes first, then the six non-zero
   angle deltas `-3,-2,-1,1,2,3` for each directional mode
5. insertion index (CPU only; `intra_candidate_better()` omits it for a stable
   GPU reduction)

The input is a plain POD array, so a GPU kernel can compute candidate SSE/rate
values and reduce them with the same comparator. No pointers, allocation, or
entropy state cross the API.

### Rate inputs and calibration

`estimate_intra_rates()` is the pluggable rate hook:

```cpp
IntraRateRequest request;
request.mode = av2_mode;
request.angle_delta = angle_delta;
request.block_w = w;
request.block_h = h;
request.base_q_idx = base_q_idx;
request.sse = candidate_sse;
request.quant_step = static_cast<double>(txv::kQuantStep);

IntraRateEstimate estimate;
estimate_intra_rates(entropy_hook, entropy_context, request, estimate, &error);
candidate.syntax_rate_bits = estimate.syntax_rate_bits;
candidate.coefficient_rate_bits = estimate.coefficient_rate_bits;
```

The default helpers are documented placeholders:

- `default_intra_syntax_rate_bits()` — rough mode/angle bits.
- `default_intra_coefficient_rate_bits()` — SSE-based placeholder.
- `intra_coefficient_rate_bits_from_levels()` — the experiment's proxy over the
  candidate's real quantised levels; it reuses the RDOQ bring-up model.

The entropy owner must replace these with exact AV2 symbol costs before the
decision becomes a default policy. Lambda comes from
`lambda_sse_from_quant_step(quant_step, sse_scale)`; `0.125` is a placeholder
scale and the measured results are sensitive to it.

### M1 integration sketch

Inside `choose_luma_mode()`, after predicting each candidate:

```cpp
av2enc::rdo::IntraCandidateCost candidate;
candidate.mode = av2_y_mode(params.mode);
candidate.angle_delta = params.angle_delta;
candidate.sse = static_cast<double>(sse);
candidate.syntax_rate_bits = /* estimator or default */;
candidate.coefficient_rate_bits = /* estimator or level-based proxy */;
candidates.push_back(candidate);
```

Then replace M1's SSE-only winner:

```cpp
const auto config = av2enc::rdo::make_intra_rdo_config(
    static_cast<double>(txv::kQuantStep));
const auto choice = av2enc::rdo::select_intra_mode(candidates, config);
if (choice.found) {
  best.mode = static_cast<intra::Mode>(choice.mode);
  best.angle_delta = choice.angle_delta;
  best.av2_mode = choice.mode;
  best.sse = static_cast<std::int64_t>(choice.sse);
}
```

`rate_aware = false` reproduces the current SSE-only decision. The production
denominator is still the same: full intra remains opt-in until the integration
owner wires this and the decoder/quality gate passes.

### Experiment

`experiment/run_m1_intra_rdo.sh` builds an experiment-only patched M1 copy (the
repository file is untouched), compares DC / SSE-only / rate-aware streams with
`tools/quality.py`, and verifies the rate-aware reconstruction bit-exactly with
`tools/m1_verify`. See `experiment/README.md` for measured QP-90 results.

## RDOQ trellis

`rdoq_quantize()` optimises one transform block in scan order. Inputs are the
forward coefficients, the scalar-quantised levels, a per-coefficient context and
a dequant step. Outputs are signed levels in scan order, EOB, distortion, rate
and total cost.

```cpp
RdoqConfig rdoq;
rdoq.lambda = lambda_sse;
rdoq.quant_step = dequant_step;
rdoq.max_level_delta = 1;

RdoqRateModel rate_model = default_rdoq_rate_model();  // placeholder
// Production: set rate_model.context/coefficient_rate/eob_rate to an
// entropy-cost estimate. This framework never calls into the entropy coder.

RdoqResult result;
std::string error;
if (!rdoq_quantize(coefficients, rdoq, rate_model, result, &error)) {
  // invalid config, invalid rate from the hook, or too many coefficients
}
```

Semantics:

- `eob` is the number of coded coefficients. Positions `>= eob` are zero;
  `eob == 0` means the whole block is zero.
- Candidates per coefficient: zero, the initial level, and levels within
  `max_level_delta` above/below it. `allow_level_decrease`,
  `allow_level_increase` and `allow_zero_to_one` control the neighbourhood.
- Trellis state is the previous magnitude clamped to `{0, 1, 2}`. The DP also
  considers terminating the block at position 0 or immediately after a non-zero
  coefficient, so trailing zeros are not coded and EOB is canonical.
- Cost is `distortion + lambda * rate_bits`, where distortion is
  `(original - level * quant_step)^2`.
- DP is `O(N * 3 * L)` with `L <= 2 + 2 * max_level_delta`.

### Rate hook contract

```cpp
double coefficient_rate(void* context,
                        int32_t scan_pos,
                        int32_t prev_abs_level,  // clamped to {0,1,2}
                        int32_t abs_level,
                        int32_t level_context);

double eob_rate(void* context, int32_t eob, int32_t num_coefficients);
```

Both must return finite, non-negative bit counts. A null callback falls back to
the documented placeholder model. `level_context` is copied from
`RdoqCoefficient::context`, so the integration pass can pass a band/position
context without changing this API.

The `default_rdoq_*` functions are **not** entropy estimates. They exist for
tests and bring-up. Real decisions need a rate model derived from the AV2
coefficient coding contexts (or measured tables). That model belongs to the
integration pass; do not edit `src/entropy/` from this component.

### Limitations

This is deliberately a limited path, not a full AV2 RDOQ:

- The context is a three-state previous-magnitude model, not the full
  coefficient-context derivation.
- The search is local around the scalar-quantised level.
- The default rate model is a placeholder.
- It is a CPU reference. A GPU port must match this code bit-for-bit before it
  replaces it, and must not add per-block host round trips.
- `max_level_delta` is capped at 1024 and `max_coefficients` defaults to 4096 to
  bound the reference path.

## Tests

`rdo_test.cpp` covers:

- Lagrangian arithmetic and lambda monotonicity/clamping.
- SSD/SAD/SATD known values, edge padding and stride handling.
- Coefficient SSE.
- Candidate orderings under low/medium/high lambda.
- Both tie-break policies and deterministic reordering.
- Invalid candidate rejection and selector bookkeeping.
- Partition child-cost aggregation.
- RDOQ exact-level recovery, heavy-lambda zeroing, level-increase/zero-to-one
  switches, EOB termination and determinism.
- An exhaustive 4-coefficient brute-force comparison that validates the trellis
  DP, EOB path and final cost recomputation together.
- Invalid RDOQ configs and invalid rate hooks.
- Intra selection: SSE-only vs rate-aware ordering, tie-breaks, invalid
  candidate rejection, lambda calibration, default rate helpers and the
  pluggable rate hook.
- The experiment runner's patch-neutrality gate (DC/SSE streams byte-identical
  to the unpatched CLI) and bit-exact `dav2d` reconstruction for rate-aware
  streams.

## Integration checklist

1. Pass a `RdoConfig` whose lambda matches the distortion metric.
2. Supply real rate estimates in bits. Do not mix byte and bit costs.
3. Use the same tie-break policy and coefficient generation order everywhere so
   decisions are reproducible.
4. For partition, include all child costs and split signaling bits.
5. For RDOQ, replace the placeholder rate model with an entropy-derived one
   before measuring quality.
6. For intra, keep `rate_aware = false` until the entropy owner supplies the
   rate hook and the target-corpus lambda scale is calibrated.
7. Re-run the decoder gate after wiring: the output must still pass
   `tools/validate_dav2d.sh`; CPU/GPU decision parity is checked before any
   BD-rate claim.
8. Keep CPU-only results labelled as reference results, not product speed.

No bitstream, entropy, M0 encoder or root build file edits are made or required
by this component.

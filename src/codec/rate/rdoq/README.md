# Full-trellis RDOQ (`src/encoder/rdoq/`)

Status: **component only.** It is not wired into `CMakeLists.txt`,
`src/main.cpp`, `src/encoder/m1/`, the entropy coder or the bitstream writer.
It produces no bytes by itself and cannot change a reference stream. The
integration pass owns wiring it into the TX stage.

This is the encoder-side, coefficient-level RDOQ requested for the M1 wave. It
replaces the *limited* local search in `src/encoder/rdo/` with a full trellis
over levels, an explicit EOB termination model, previous-level state, and a
pluggable entropy-rate callback.

## Files

| File | Purpose |
|---|---|
| `rdoq.h` / `rdoq.cpp` | Core DP, scalar baseline, batch interface |
| `rdoq_av2_model.h` / `.cpp` | §9.3/§8.3 CDF-derived rate model (example) |
| `rdoq_cdf_tables.h` | Generated qctx-0 CDF tables |
| `generate_rdoq_cdf.py` | Regenerates the tables from `spec/tables/default_cdf.json` |
| `rdoq_test.cpp` | Exhaustive + randomized CPU tests |
| `still_rdoq_quality.cpp` | Real-M1 coefficient replay + scalar-vs-RDOQ measurement |
| `build_and_test.sh` | Standalone build/test/measurement |

## Build and test

```sh
cd src/encoder/rdoq
./build_and_test.sh
RDOQ_SANITIZE=1 ./build_and_test.sh   # ASan + UBSan (core and harness)
RDOQ_SKIP_M1=1 ./build_and_test.sh    # core test only
```

No CMake. The quality harness compiles the existing CPU M1 components
read-only and writes nothing into the repository.

## API

```cpp
namespace av2enc::rdoq {

struct Coefficient {
  double original;          // unquantised forward coefficient, scan order
  std::int32_t level_context;
  std::int32_t scalar_level; // seed; kAutoScalarLevel derives it
};

struct Config {
  double lambda;            // from rdo::lambda_sse_from_quant_step()
  double quant_step;        // reconstruction step in the forward domain
  std::int32_t prev_level_cap; // state = min(|prev|, cap); default 2
  std::int32_t level_cap;   // 0 = derived bound + 64-level margin
  bool allow_zero_to_one;
  std::int32_t max_coefficients;
};

using CoefficientRateFn = double (*)(void* context, int scan_pos,
                                     int prev_abs_state, int signed_level,
                                     int level_context, int is_eob);
using EobRateFn = double (*)(void* context, int eob, int num_coefficients);

struct RateModel { void* context; CoefficientRateFn coefficient_rate;
                   EobRateFn eob_rate; };

struct Result {
  std::vector<std::int32_t> levels; // signed, scan order
  std::int32_t eob;
  double distortion, rate_bits, cost;
  std::int32_t max_level_searched;
  bool level_range_truncated;
};

bool rdoq_quantize(std::span<const Coefficient>, const Config&,
                   const RateModel&, Result&, std::string* error = nullptr);

// Pointer/offset batch form (same POD can be handed to a device kernel).
bool rdoq_quantize_batch(const BatchInput&, const BatchConfig&,
                         const RateModel&, BatchOutput&, std::string* error);

// Largest magnitude the DP searches for one coefficient (GPU/table builders).
std::int32_t rdoq_level_bound(const Coefficient&, const Config&) noexcept;
}
```

`quant_step` is the reconstruction step in the **forward coefficient**
domain: the scalar quantiser is `level = lround(original / quant_step)`. For
`tx_variants`, `quant_step = 1.0 / TxParams::quant_factor` (for an 8x8 block
that is 8.25; for 32x32 it is 8.25 with `dq_denom = 2`, i.e. the coded step is
still 8.25 in forward units). Do not pass the decoder dequant value
(`66 / dq_denom`) directly; it is in a different scale.

Lambda comes from the RDO framework:

```cpp
const double step = 1.0 / params.quant_factor;
const double lambda = av2enc::rdo::lambda_sse_from_quant_step(step, sse_scale);
```

`sse_scale = 0.125` is the RDO framework default and is documented there as a
calibration placeholder. The harness exposes `--sse-scale` so the number can be
re-calibrated without changing this component.

## Trellis semantics

The DP runs in reverse scan order. `dp(i, prev)` is the minimum cost to code
coefficients `i..N-1`, given the previous coded magnitude clamped to
`prev_level_cap`, with the constraint that the block has **not** terminated
yet. At each position:

* continue with any magnitude `0..bound` (state becomes `min(m, cap)`);
  coefficient rate uses `is_eob = 0`;
* terminate with a non-zero magnitude: coefficient rate uses `is_eob = 1`
  (the `coeff_base_eob` syntax), then the EOB rate for `eob = i+1`;
* at position 0 only, terminate with an all-zero block and the EOB/all-zero
  rate.

This makes EOB canonical: trailing zeros are never coded, and the last coded
coefficient is the unique EOB coefficient. There is no separate "terminate
after a zero" transition, so the DP cannot produce a non-canonical EOB.

**Level range.** The search does not assume a monotone rate. It evaluates every
magnitude up to the distortion minimum, then keeps going while the local
distortion alone is below the best total cost found for that state. Once the
local distortion exceeds that cost, every higher magnitude has at least that
distortion and cannot win (rate and future cost are non-negative). The loop cap
is the classic `floor(2|c|/s)+1` bound plus a 64-level margin, because the AV2
CDF symbol costs can dip by a fraction of a bit as the base/BR symbol changes.
`Config::level_cap` can clip the cap for pathological inputs; reaching it
without the stop rule sets `Result::level_range_truncated`.
`rdoq_level_bound()` exposes the capped range to GPU/table builders.

The brute-force tests deliberately use a monotone test model and set
`level_cap` equal to the enumerated range, so they compare exactly the same
level set as the DP. The AV2 model's small non-monotone dips are covered by the
margin and the stop rule.

**Tie-break.** `cost`, then total distortion, then total rate, then smaller
magnitude, then terminal before non-terminal, then signed level. Comparisons
are exact binary64. The order is a total order over the suffix chosen from each
state, so the output is deterministic for a given build and callback. Build
with `-ffp-contract=off` (CPU) and `-fmad=false` (GPU) before comparing.

## Rate model

`rdoq_av2_model.*` implements the coefficient syntax costs from the normative
default CDFs (qctx 0, base_q_idx <= 90) for the M1 writer subset:

* txb_skip / all-zero flag, EOB prefix + extra bits;
* `coeff_base` / `coeff_base_eob`, LF variants and chroma variants;
* `coeff_br` and the §5.20.7.28 escape prefix/Golomb/literal;
* sign bits, including the luma DC-sign CDF.

The cost is the ideal code length `-log2(p)` under the static CDF. M1 disables
CDF adaptation, so this is close to the real coder cost (the range coder adds a
fraction of a bit per symbol plus one trailing marker).

**Documented approximation.** The AV2 base/BR contexts depend on already-coded
neighbouring levels. A trellis that carries only the previous level cannot
recompute them for every candidate. The model freezes them from the scalar
quantisation:

* neighbour sums for `coeff_base` / `coeff_br` come from the scalar pre-levels
  at the below/right positions;
* the escape `hr_level_avg` predictor is the scalar reverse-scan state;
* the txb_skip and DC-sign contexts are supplied by the caller, which knows the
  above/left cumulative levels.

The model covers the coefficient syntax, the all-zero/txb_skip flag and EOB. It
does **not** include the luma transform-type symbol, whose presence depends on
`eob` (the M1 writer emits none for `eob <= 1`). The M1 writer also maps EOB
multisize 1 (4x8/8x4) to `eob_pt_64`, while the model uses the spec's
`eob_pt_32`; that changes only the estimate for those small rectangles. The
actual tile re-encode in the measurement uses the writer, so it is the final
count.

This is the standard RDOQ approximation and keeps the DP additive. It is why
the measured actual tile saving is a little smaller than the model's estimate.
The entropy owner can replace the callback with exact costs without changing
the core API.

`default_rate_model()` is a **bring-up** model for unit tests; do not use it for
quality decisions. The AV2 model above is the example used for the M1
measurement.

## Tests

`rdoq_test.cpp` covers:

* config validation and invalid/non-finite rates;
* `lambda = 0` recovers the scalar levels and EOB exactly;
* deterministic repeats;
* **exhaustive brute force on 4x4** (16 coefficients, binary levels: all 65,536
  canonical paths) and **4x4 with 8 active coefficients and levels 0..3**
  (~65k paths);
* **exhaustive brute force on 8x8** with a 10-coefficient active prefix and
  levels 0..2 (~59k paths, canonical EOB and state included);
* randomized 32/64/128/1024-coefficient blocks against an independent
  **forward** DP over exactly the same per-coefficient range, plus
  scalar/all-zero bounds. The test rate model has a deliberate non-monotone
  dip, so this exercises the safe distortion stop rule;
* the pointer/offset batch interface against per-block calls;
* AV2 model scan permutation, a DC monotonicity spot check (not required by
  the trellis), and a like-for-like scalar-vs-RDOQ cost check.

Current result:

```
av2enc RDOQ trellis test
5260 checks, 0 failures
RESULT: PASS
```

The same test is clean under `-fsanitize=address,undefined`.

## Real-M1 measurement (QP 90)

`still_rdoq_quality` replays the M1 encoder without editing it:

* partition tree from `src/cuda/m1/plan/` (M1-identical search);
* luma = M1 default DC; chroma = M1 default DC/CfL with the M1 U-only DC score
  and U+V CfL score;
* `tx_variants` residual/transform/quantise/inverse, `-ffp-contract=off`;
* txb_skip, DC-sign and cumulative-level contexts tracked exactly like the §8.3
  writer;
* the replay is compared to a real `av2enc::still::encode_still_picture` run.

On both pinned fixtures the replay is **byte-exact for all three planes** and
the reconstructed scalar syntax stream has the **same tile byte count** as the
reference encoder:

```
64x64:    Y 0/4096, U 0/1024, V 0/1024 differ; tile 870 B MATCH
640x360:  Y 0/230400, U 0/57600, V 0/57600 differ; tile 13288 B MATCH
```

The measurement re-encodes the same partition tree, luma DC mode and TX sizes
with RDOQ levels and measures the actual §8.2 tile payload, so the bit delta is
real (not just a model estimate). The RDOQ pass is a second replay that uses
its own reconstruction, so later predictions are consistent with the coded
levels; the chroma DC/CfL decision is repeated against that new luma
reconstruction, exactly as an integrated encoder must do. Both streams were
decoded with the pinned `dav2d` and the decoded PSNR matches the reported block
SSE.

### 64x64 (`tests/data/basic_64x64_1f.y4m`)

| | scalar | RDOQ |
|---|---:|---:|
| non-zero levels | 1511 | 1226 |
| sum of EOBs | 2974 | 2297 |
| coefficient SSE | 25,075.8 | 31,786.3 |
| pixel SSE (visible) | 18,855 | 24,080 |
| PSNR | 43.26 dB | 42.20 dB |
| model coefficient bits | 6,869.7 | 5,579.5 (−18.8%) |
| **actual tile bytes** | **870 B** | **728 B (−16.32%)** |
| model RD cost | 106,079 | 98,058 (−7.56%) |
| actual-tile RD (weighted λ) | 100,923 | 92,753 (−8.10%) |

### 640x360 (`tests/data/basic_640x360_1f.y4m`)

| | scalar | RDOQ |
|---|---:|---:|
| non-zero levels | 22,217 | 18,578 |
| sum of EOBs | 41,785 | 32,976 |
| coefficient SSE | 321,392 | 402,739 |
| pixel SSE (visible) | 519,989 | 587,607 |
| PSNR | 46.36 dB | 45.83 dB |
| model coefficient bits | 105,017 | 87,737 (−16.5%) |
| **actual tile bytes** | **13,288 B** | **11,420 B (−14.06%)** |
| model RD cost | 1,347,583 | 1,251,155 (−7.16%) |
| actual-tile RD (weighted λ) | 1,558,756 | 1,480,346 (−5.03%) |

Per plane at 640x360: Y bits −14.68%, PSNR 45.08→44.60 dB; U bits −39.46%,
PSNR 52.42→51.95; V bits −33.87%, PSNR 50.48→49.28. Chroma gains are large
because the M1 chroma quantiser is very fine and many chroma coefficients are
near the zero decision.

The RDOQ stream is produced by a second replay that uses its own
reconstruction (`use_rdoq_recon`), so later predictions and CfL are consistent
with the coded levels. Both streams were decoded with the pinned `dav2d`
(f9dafbc); the decoded PSNR matches the block-local measurement exactly
(640x360: Y 44.600, U 51.954, V 49.275; average 45.826 dB). The scalar stream
also passes `validate_dav2d.sh` and its tile bytes match the M1 reference
byte-for-byte.

**Verdict.** With a fixed partition tree, luma DC mode and TX sizes (chroma
repeats the DC/CfL decision), the full trellis is a real **RD-cost win**:
−14.06% actual tile payload for −0.53 dB PSNR at QP 90 on 640x360, with the
measured Lagrangian cost improving 5.03% using lambda from the RDO framework.
The stream decodes. This is **not** a full-encoder BD-rate claim: partition and
luma mode decisions were held at the scalar choices, and a real encoder would
re-decide them after quantization. Treat the number as a
quantisation-only integration result; re-run the mode/partition search and the
`dav2d` gate before quoting BD-rate.

### Lambda sensitivity (640x360, all planes)

The RDO framework's `sse_scale` is a calibration placeholder. Measured with all
three planes on the same real tile re-encode:

| `sse_scale` | weighted λ | actual tile bits | tile saving | PSNR | actual-tile RD |
|---:|---:|---:|---:|---:|---:|
| 0.03125 | 2.44 | 106,304 → 101,376 | 4.64% | 46.36 → 46.31 dB | −0.76% |
| 0.0625 | 4.89 | 106,304 → 97,520 | 8.26% | 46.36 → 46.17 dB | −1.99% |
| **0.125** | **9.77** | **106,304 → 91,360** | **14.06%** | **46.36 → 45.83 dB** | **−5.03%** |
| 0.25 | 19.54 | 106,304 → 82,152 | 22.72% | 46.36 → 44.93 dB | −10.39% |
| 0.5 | 39.09 | 106,304 → 69,568 | 34.56% | 46.36 → 43.14 dB | −18.50% |

The default 0.125 is conservative; the RD cost improves monotonically with λ
because the same λ is used on both sides of the measured comparison. Pick the
scale from the final rate-control operating point, not from this table.


## TX-stage integration notes

The integration pass should:

1. Feed the forward transform coefficients in **scan order** (the writer's §8.3
   order), one `Coefficient` per position. Set `scalar_level` from the existing
   `txv::quantize()` result when exact scalar parity matters; the trellis itself
   only needs `original`.
2. Pass `quant_step = 1.0 / TxParams::quant_factor` and
   `lambda = rdo::lambda_sse_from_quant_step(quant_step, sse_scale)`.
   Re-calibrate `sse_scale`; 0.125 is the framework placeholder.
3. Build the block rate model from the scalar levels, the caller-known txb_skip
   / DC-sign contexts, and either the AV2 CDF model or an exact entropy-cost
   hook. The hook contract is:
   `coefficient_rate(context, scan_pos, prev_abs_state, signed_level,
   level_context, is_eob)` and `eob_rate(context, eob, N)`; both return bits.
4. Use the returned `levels` up to `eob` as the scan-order levels and set the
   block EOB. The §8.3 writer already consumes scan-order levels; no syntax
   changes are needed.
5. Keep the same tie-break/build flags in CPU and GPU; compare decisions before
   changing any reference MD5.
6. Re-run `tools/validate_dav2d.sh` and the quality gate after wiring. The
   component itself writes no bits.

GPU: `src/cuda/rdoq/` provides a pointer/offset batch kernel that consumes a
precomputed per-coefficient rate table and mirrors this DP. It has a
CPU-vs-GPU parity test; see that directory's README.

## Limitations

* No adaptive CDF: the model is the static qctx-0 set (base_q_idx <= 90).
* Contexts are frozen from the scalar quantisation (see above).
* The safe stop rule is exact for any finite non-negative rate, but the loop
  still has a hard cap (derived bound + 64 levels, or `level_cap`). A callback
  with a large non-monotone dip beyond that cap could be clipped; check
  `Result::level_range_truncated`, raise `level_cap`, or supply a monotone
  model.
* M1 default modes only in the harness (luma DC, chroma DC/CfL). Directional
  luma/chroma (`--still-full-intra`, `--still-uv-full`) is not part of the replay.
* No deblocking in the replay. It does not affect coefficients, but the
  measurement's PSNR is pre-deblock.
* The batch interface is allocation-free and SoA; it is intended for a device
  port, but the CPU implementation still calls the host callback per candidate.

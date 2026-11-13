# AV2 TCQ / CCTX (`src/codec/transform/tcq/`)

Status: **component + integrated still path.** The CPU `--still-tcq` path
(`src/pipeline/still/still_encoder.cpp` + `still_syntax.cpp` +
`still_headers.cpp`) now writes `enable_tcq = 1`, omits the
`enable_parity_hiding` bit (single-picture inference), selects
`coeff_base`/`coeff_base_lf` with `(tcqState >> 1) & 1`, uses the
`read_quant` escape for the syntax level, and doubles `dqDenom` for luma
2-D blocks. The rest of this document describes the standalone component,
which remains the CPU oracle.

Video integration status (`src/pipeline/video/video_tcq_gate.sh`): the CPU
`--video-tcq` writer enables §5.4.8 TCQ with a non-single-picture sequence
(`enable_tcq = 1`, `choose_tcq_per_frame = 1`, `enable_parity_hiding = 0`) and
codes `allow_tcq = 1` in every frame header.  The luma quantiser runs the same
trellis in both video paths: `compute_tx()` for the single-DCT path and the
`--video-tx-search` candidate evaluator.  The coefficient writer uses the
`(tcqState >> 1) & 1` CDF selectors, `syntax_level_for()`, the `read_quant`
escape and the doubled `dqDenom` reconstruction.  The gate decodes 64x64 and
360p streams at QP 60/90/120 with `dav2d --strict 1` and pins the encoder recon
exactly.  TCQ is rejected by name with `--video-qm`, segmentation/delta-Q,
lossless, `--tiles > 1x1`, the frame-type wrappers, `--svc-layers` and the GPU
backend.

Video integration status for CCTX (`src/pipeline/video/video_cctx_gate.sh`):
the CPU `--video-cctx` writer sets §5.4.8 `enable_cctx = 1`, quantises the
chroma pair jointly with the seven-type search, codes `cctx_type` on plane 1
after the U EOB (including the inter `eob == 1` form), and reconstructs with
the §7.14.6 joint dequant rotation.  It composes with `--video-tcq` and
`--video-tx-search`; the gate decodes 64x64 and 360p streams at QP 60/90/120
with `dav2d --strict 1`, pins the encoder recon exactly, and asserts at least
one non-zero rotation is coded.  CCTX is rejected by name with `--video-qm`,
segmentation/delta-Q, lossless, `--tiles > 1x1`, the frame-type wrappers,
`--svc-layers` and the GPU backend.

Integration status (`src/codec/transform/tcq/still_tcq_gate.sh`):

* `--still-tcq` is default-off; the pinned 64x64 and 640x360 `--still`
  streams are byte-identical. TCQ is enabled per sequence and applies to
  luma `TX_CLASS_2D` blocks only, exactly as §5.20.7.27 defines `useTcq`.
* The decoder gate decodes `--still-tcq` at QP 60/90/120 with
  `dav2d --strict 1` and compares every plane against the encoder
  reconstruction (0 differing samples). The pinned IVF MD5s are in the gate.
* Parity hiding cannot be signalled in a single-picture stream:
  `choose_tcq_per_frame` is absent and inferred 0, so `enable_tcq = 1`
  infers `enable_parity_hiding = 0`. The encoder writes no parity bit in
  that case and never codes the PH coefficient symbols.
* `--still-tcq` composes with TX_TYPE, RDO-intra and CDEF. It is rejected by
  name with RDOQ, QM, DC/AC delta-Q and IST (each changes the quantiser or
  needs the Quant domain the TCQ trellis is not run in).
* CCTX (`enable_cctx`, `cctx_type`, §7.14.6) is integrated separately in
  `--still-cctx`; see the `still_cctx_gate.sh` note below.

Component status (below): **component only** for the standalone build.
It is not wired into `CMakeLists.txt`'s standalone harness, `src/main.cpp`'s
default path, the entropy writer or the bitstream writer by itself.
The component implements the AV2 quantizer tools that sit beyond scalar and
RDOQ quantisation:

* **TCQ** — `Tcq_Next_State`, the two-pass state sequence, the
  `Quant = 2 * Quant - q0` level mapping, and the `shift += 1` dequant
  interaction (§5.20.7.27, §5.20.7.28, §7.14.4);
* **CCTX** — `cctx_type`, the six rotations, encoder search, and the §7.14.6
  joint dequant;
* **header/entropy implications** — `enable_tcq`, `choose_tcq_per_frame`,
  `enable_parity_hiding`, `allow_tcq`, `allow_parity_hiding`, `enable_cctx`,
  the `(tcqState >> 1) & 1` `coeff_base` CDF selector, and the TCQ
  `read_quant` escape.

No file outside `src/encoder/tcq/` is modified. Nothing is committed.

## Normative vs encoder-side

The AV2 spec defines the **decoder** mapping only. It does not prescribe an
encoder trellis. `tcq.cpp` contains:

* the normative tables and mapping, exposed as `next_state()`,
  `state_q0()`, `decode_levels()` and `dequant_shift_extra()`;
* an encoder DP over the eight states, which is **not normative** but is
  constrained to produce exactly what `decode_levels()` and §5.20.7.28 can
  represent.

The DP output has two level arrays:

* `base_levels[c]` — signed full coefficient level, the value the §8.3 writer
  consumes (`coeff_base`/`coeff_br` plus the `read_quant` escape);
* `quant_levels[c]` — signed `Quant[]` after the TCQ mapping, the value the
  §7.14.4 dequant consumes with the extra shift.

For a coded full level `q` and state `s`:

```
q0   = (s >> 1) & 1
m    = 2*q - q0            // final Quant magnitude
l    = syntax_level_for(q, pre_max)
next = Tcq_Next_State[s][l & 1]
```

`pre_max` is 6 for a non-low-frequency 2D luma position and 8 for
`(row + col) < 4`.  With `allowTcq = 1`, `read_quant` reads the escape when
the syntax level is at least `maxLevel - 1 = pre_max - 1`, and adds
`x << 1`.  The writer therefore uses:

```
l = q                                    if q <  pre_max - 1
l = pre_max       when (q ^ pre_max) is even
l = pre_max - 1   when (q ^ pre_max) is odd
x = (q - l) / 2
```

This covers both parities; no full level is rejected.  The first-pass state
uses the syntax level `l` parity (equal to `q & 1`), exactly as §5.20.7.27
does; the second pass uses the `read_quant` output parity, which is the same.

## Files

| File | Purpose |
|---|---|
| `tcq.h` / `tcq.cpp` | state table, dequant shift, encoder DP, normative decoder mapping |
| `tcq_flags.h` / `.cpp` | sequence/frame header flag derivation and bit counts |
| `cctx.h` / `.cpp` | CCTX types, §7.14.6 dequant, inverse forward rotation, seven-type search |
| `tcq_av2_model.h` / `.cpp` | static §9.3 rate model with the exact TCQ CDF selector |
| `tcq_cdf_tables.h` | generated `Default_Cctx_Type_Cdf` |
| `generate_tcq_cdf.py` | regenerates the CCTX table from `spec/tables/default_cdf.json` |
| `tcq_test.cpp` | exhaustive/round-trip/header/CCTX CPU tests |
| `tcq_model_test.cpp` | rate-model, state-selector and rejection tests |
| `tcq_rd_compare.cpp` | real-coefficient scalar vs RDOQ vs TCQ and CCTX RD harness |
| `build_and_test.sh` | standalone build/test/measurement |

## Build and test

```sh
cd src/encoder/tcq
./build_and_test.sh
TCQ_SANITIZE=1 ./build_and_test.sh      # ASan + UBSan, unit/model tests
TCQ_SKIP_RD=1 ./build_and_test.sh       # unit/model tests only
TCQ_QPS="60 90 120" ./build_and_test.sh # RD sweep
```

No CMake. The harness compiles `tx_variants`, `rdoq`, `rdo` and `y4m`
read-only as the reference. It writes only to `${TMPDIR:-/tmp}`.

Current result:

```
core TCQ / CCTX / header tests: 1459 checks, 0 failures
TCQ rate-model tests:            151 checks, 0 failures
```

Both are clean under `-fsanitize=address,undefined`.

## API

```cpp
namespace av2enc::tcq {

// State machine (§5.20.7.27).
constexpr int next_state(int state, int parity);
constexpr int state_q0(int state);        // (state >> 1) & 1
constexpr int state_cdf_index(int state); // same bit
constexpr int syntax_level_for(int full_level, int pre_max);
constexpr int syntax_escape_for(int full_level, int pre_max); // x

// §7.14.4.
constexpr int dequant_shift_extra(bool allow_tcq, int plane, bool lossless,
                                  bool tx_class_2d, bool use_fsc);
constexpr int dequant_shift(int coded_pels, bool allow_tcq, int plane,
                            bool lossless, bool tx_class_2d, bool use_fsc);

struct Coefficient {
  double original;
  std::int32_t level_context;
  std::int32_t scalar_level; // kAutoScalarLevel to derive
  std::int32_t pre_max;      // 6 HF, 8 for (row + col) < 4
};

struct Config {
  double lambda;
  double quant_step;         // scalar forward step S; TCQ reconstructs at S/2
  std::int32_t level_cap;
  std::int32_t max_coefficients;
  bool allow_zero_to_one;
};

using CoefficientRateFn = double (*)(void* context, std::int32_t scan_pos,
                                     std::int32_t state,
                                     std::int32_t base_level,
                                     std::int32_t next_state,
                                     std::int32_t signed_quant,
                                     std::int32_t level_context,
                                     std::int32_t is_eob);
using EobRateFn = double (*)(void* context, std::int32_t eob,
                             std::int32_t num_coefficients);
struct RateModel { void* context; CoefficientRateFn coefficient_rate;
                   EobRateFn eob_rate; };

struct Result {
  std::vector<std::int32_t> base_levels;  // writer levels, scan order
  std::vector<std::int32_t> quant_levels; // dequant Quant, signed
  std::vector<std::int32_t> states;       // state entering each position
  std::int32_t eob;
  double distortion, rate_bits, cost;
  std::int32_t max_level_searched;
  bool level_range_truncated;
};

bool quantize(std::span<const Coefficient>, const Config&,
              const RateModel&, Result&, std::string* error = nullptr);

// Normative decoder mapping, independent of the DP.
bool decode_levels(std::span<const std::int32_t> full_levels_scan,
                   bool allow_tcq, bool lossless, DecodeResult*,
                   std::string* error = nullptr,
                   std::span<const std::int32_t> pre_max_scan = {});
}
```

CCTX:

```cpp
namespace av2enc::tcq {

enum class CctxType : std::int8_t { kNone = 0, k45, k30, k60,
                                    kMinus45, kMinus30, kMinus60 };

bool cctx_allowed(bool enable_cctx, bool lossless, bool is420,
                  int chroma_w, int chroma_h);
CctxPair cctx_dequant_pair(std::int32_t u, std::int32_t v, CctxType, int bd);
double cctx_type_bits(CctxType);  // Default_Cctx_Type_Cdf

struct CctxConfig { bool enabled; double quant_step_u, quant_step_v;
                    int bitdepth; };
struct CctxResult { CctxType type; std::vector<std::int32_t> u_levels, v_levels;
                    std::int32_t eob_u, eob_v; double distortion; };

bool cctx_enumerate_scalar(std::span<const double> u,
                           std::span<const double> v, int w, int h,
                           const CctxConfig&, std::vector<CctxResult>*,
                           std::string* error = nullptr);
bool cctx_quantize_scalar(...); // minimum-distortion convenience
}
```

`cctx_enumerate_scalar()` returns all seven candidates. An RD encoder scores
each with its own U/V coefficient model plus `cctx_type_bits(type)` when the
frame signals `cctx_type`. When `enabled = false` only `CCTX_NONE` is returned.

## Header flags

`tcq_flags.h` derives the normative interactions and the exact bit count:

* `choose_tcq_per_frame` exists only when `enable_tcq && !single_picture_header_flag`;
* `enable_parity_hiding` is forced to 0 when `enable_tcq && !choose_tcq_per_frame`;
* `allow_tcq` is 0 in `CodedLossless`, otherwise the frame bit when
  `choose_tcq_per_frame`, otherwise `enable_tcq`;
* `allow_parity_hiding` is forced to 0 when `CodedLossless || !enable_parity_hiding || allow_tcq`.

Consequently, enabling TCQ in a single-picture stream (M1/M2) always disables
parity hiding. The sequence header must also **omit** the
`enable_parity_hiding` bit in that case; the current core writers always emit
it.

## Entropy implications

1. `coeff_base` and `coeff_base_lf` select their CDF with
   `(tcqState >> 1) & 1`. `coeff_base_eob` has no TCQ axis.
2. The first pass over coded levels updates the state with the parity of the
   level the syntax reconstructs (`min(full_level, pre_max)`); the state is
   reset before the second pass.
3. `read_quant` is called with `allowTcq = useTcq`, so the escape threshold is
   `maxLevel - 1 = pre_max - 1` and the increment is `x << 1`. The writer must
   first map the full level to `l = syntax_level_for(full, pre_max)`, decompose
   `l` (not the full level) into `coeff_base` + `coeff_br`, and then send
   `x = (full - l) / 2`. The escape condition also moves from
   `full >= pre_max` to `l >= pre_max - 1`; levels `pre_max - 1` and `pre_max`
   themselves pay an escape with `x = 0`.
4. The dequant denominator is doubled for luma 2D lossy TCQ:
   `shift += 1` (§7.14.4). The inverse transform therefore consumes
   `quant_levels`, not `base_levels`.
5. TCQ is luma-only (`plane == 0`), 2D-class only, and disabled for lossless
   and FSC.

CCTX is independent: it adds a `cctx_type` symbol only for plane 1 when
`(is_inter || eob != 1) && is_cctx_allowed()`, and the decoder applies the
rotation to the saved U/V dequant blocks before the inverse transforms.

## Real-coefficient RD comparison

`tcq_rd_compare` uses the real Y4M fixtures, DC-predicted 8x8 residual blocks,
the `tx_variants` forward transform, the same lambda
(`rdo::lambda_sse_from_quant_step(1 / quant_factor, 0.125)`) for every
candidate, and the static §9.3 coefficient model. These are model bits, not a
tile re-encode.

640x360, `tests/data/basic_640x360_1f.y4m`, 3600 luma blocks:

| QP | scalar cost | RDOQ vs scalar | TCQ vs scalar | **TCQ vs RDOQ** | TCQ bits vs RDOQ |
|---:|---:|---:|---:|---:|---:|
| 60  | 288,400 | −2.25% | −1.57% | **+0.69%** | +3.61% |
| 90  | 1,123,049 | −4.89% | −6.40% | **−1.59%** | +2.75% |
| 120 | 4,027,380 | −8.46% | −8.91% | **−0.49%** | +3.30% |

At QP 90 TCQ lowers coefficient SSE from 287,042 (RDOQ) to 248,647 while
spending 2.75% more model bits; the Lagrangian cost improves 1.59%. At the
finest QP in the sweep (60) the extra rate makes TCQ slightly worse than RDOQ.
This is the honest result for this model: TCQ is a win at the M1 operating
point, not an unconditional win.

Chroma CCTX, same fixture, tool-off NONE vs tool-on RD-selected:

| QP | cost NONE | cost CCTX | delta | non-NONE |
|---:|---:|---:|---:|---:|
| 60  | 117,649 | 104,497 | **−11.18%** | 674/3600 |
| 90  | 394,045 | 357,441 | **−9.29%** | 613/3600 |
| 120 | 1,246,234 | 1,325,132 | **+6.33%** | 386/3600 |

CCTX pays a `cctx_type` symbol on every eligible block when the tool is on, so
it hurts at the coarsest QP. A frame-level `enable_cctx` RD decision should be
part of integration; per-block search alone cannot remove the symbol overhead.

64x64 QP 90 for reference: TCQ vs RDOQ cost −2.70% (SSE 14,970 vs 17,640; bits
+2.07%); CCTX vs NONE −11.48%.

## Decoder gate

**No real stream gate is possible without core edits.** The current core
writers hardcode both flags to 0 and the entropy writer ignores the TCQ state,
the doubled dequant shift, TCQ escape halving and CCTX.

* `src/encoder/m1/still_headers.cpp:136-138` (and `video_headers.cpp:197-199`,
  `src/bitstream/basic_headers.cpp:243-245`) write
  `enable_cctx = 0`, `enable_tcq = 0`, `enable_parity_hiding = 0`. For a
  single-picture TCQ stream the correct sequence is `enable_tcq = 1` and **no**
  `enable_parity_hiding` bit (the `choose_tcq_per_frame` bit is absent).
* `src/encoder/m1/still_encoder.cpp:882-950` (`run_quant`) quantises scalar/RDOQ
  only. TCQ must replace the levels in `plan.levels` with
  `tcq::Result::base_levels`, while reconstruction uses
  `tcq::Result::quant_levels`.
* `src/encoder/m1/still_encoder.cpp:765-784` (`dequant_inverse_qindex`) and
  `:770` (`txv::inverse_transform`) use `params.dq_denom`. TCQ luma 2D needs
  one extra dequant shift; either pass a `TxParams` copy with
  `dq_denom * 2` or implement the custom path with `dq_denom << 1`.
* `src/encoder/m1/still_syntax.cpp:1204` and `:1215` index
  `coeff_base_lf[tsc][lf_ctx][0][0]` / `coeff_base[tsc][hf_ctx][0][0]`. The
  third index must become `(tcqState >> 1) & 1`, with `tcqState` computed by
  the first-pass loop at `:1127-1267`.
* `src/encoder/m1/still_syntax.cpp:1132-1145` derives `base_level`/`br_level`
  from the full level. Under TCQ it must derive them from
  `syntax_level_for(abs_value, pre_max)` instead. `:1292` must call
  `write_quant_escape((abs_value - syntax_level) / 2, ...)` whenever
  `syntax_level >= pre_max - 1` (not `abs_value >= pre_max`), so the
  `pre_max - 1` / `pre_max` levels are also escaped with `x = 0`.
* CCTX: `src/encoder/m1/still_syntax.cpp:1059` writes EOB and then the luma TX
  type. The `cctx_type` symbol belongs between those for plane 1, using
  `Default_Cctx_Type_Cdf` (add it to `CdfState`, near `:180-199`). CCTX
  reconstruction needs a joint step between the U/V dequant and
  `txv::inverse_block`: `src/encoder/m1/still_encoder.cpp:1126-1129`
  (`compute_tx(1/2)` and `write_recon(1/2)`) currently reconstructs each plane
  independently.
* `src/encoder/tx_variants/tx_variants.cpp` (`dequantize`) has no TCQ shift
  parameter; it is another workstream's file. Integration can call a
  `TxParams` copy with the doubled `dq_denom` rather than editing it.

For a non-single-picture stream, the frame-level `allow_tcq` bit belongs after
the segmentation/QM block and before `deblocking_filter_params()` in
`write_frame_header_syntax` (same file as the sequence flags).

Once those call sites exist, the gate is
`tools/validate_dav2d.sh <tcq.ivf> tests/data/...` with `enable_tcq = 1` and
`enable_cctx` tested separately and together. This component's own
`decode_levels()` + `dequant_shift_extra()` is the CPU oracle for that pass.

## Limitations

* The TCQ rate model freezes neighbour/escape contexts from a reference
  base-level block (the scalar half-step approximation in the harness), the
  same standard RDOQ approximation.  The TCQ state itself is exact along the
  DP path.
* The RD harness is quantisation-only: fixed DC prediction, 8x8 blocks, no
  partition/mode re-decision and no real tile re-encode.  Model bits are static
  `-log2(p)`; no CDF adaptation is modelled.
* The CCTX rate in the harness uses zero `txb_skip`/DC contexts and does not
  model the plane-2 `EobU` context.  The CCTX reconstruction and symbol table
  are exact; the rate is the usual static-model estimate.
* The encoder trellis is our design.  Only the decoder mapping and syntax are
  normative.
* No decoder gate was run.  See the blocked-integration list below.

## Spec tables / missing data

Everything needed is present in the committed artefacts:

* `Tcq_Next_State[8][2]` — spec HTML cache §5.20.7.27;
* `Cctx_Mtx[6][2]`, `CCTX_PREC_BITS = 8`, `CCTX_TYPES = 7` — spec and
  `spec/tables/constants.json`;
* `Default_Cctx_Type_Cdf` — `spec/tables/default_cdf.json`;
* `Default_Coeff_Base_Cdf` / `Default_Coeff_Base_Lf_Cdf` with their TCQ axis —
  `src/encoder/m1/qctx/qctx_cdf_tables.h`.

The normative PDF is not redistributed, but the extracted `spec.html` cache and
the committed tables contain all TCQ/CCTX constants and syntax. The only thing
the spec does **not** define is an encoder trellis; the DP in `tcq.cpp` is an
encoder design, not a claimed normative algorithm.

# M1 RDO decision layer (`src/encoder/m1/rdo_decide/`)

Status: **component + CPU experiment.** It is not wired into `CMakeLists.txt`,
`src/main.cpp`, `src/encoder/m1/m1_still.cpp`, the entropy coder, or the
bitstream writer. It owns no encoder state, writes no bits, and changes no
reference stream. The integration pass owns enabling it by default.

This directory adds one rate-aware decision layer for the four places the M1
reference still decides on distortion alone:

| Decision | API | Experiment hook |
|---|---|---|
| luma intra mode + angle | `intra_rdo.h` | `AV2ENC_RDO_DECIDE=1` |
| partition NONE / split / rect | `partition_rdo.h` | `AV2ENC_RDO_NO_PART=1` disables |
| TX type + RDOQ | `tx_rdo.h` | `AV2ENC_RDO_TXTYPE=1`, `AV2ENC_RDO_RDOQ=1` |
| chroma DC / CfL (U+V score) | `chroma_rdo.h` | `AV2ENC_RDO_NO_CHROMA=1` disables |

The shared rate model for a quantised transform stream is
`transform_stream.h`.

## Files

| File | Purpose |
|---|---|
| `config.h/.cpp` | lambda calibration and shared cost convention |
| `transform_stream.h/.cpp` | allocation-free AV2 static-CDF rate for a level stream, with a pluggable override |
| `intra_rdo.h/.cpp` | luma mode/angle wrapper over `src/encoder/rdo/intra_decision.*` |
| `chroma_rdo.h/.cpp` | DC/CfL selector on `u_sse + v_sse + lambda * bits` |
| `partition_rdo.h/.cpp` | raw `(distortion, bits)` → `partition::NodeCosts` + frame/superblock solver |
| `tx_rdo.h/.cpp` | TX-type selection and an allocation-free coefficient trellis (RDOQ) |
| `rdo_decide.h` | umbrella include |
| `rdo_decide_test.cpp` | standalone self-tests |
| `build_and_test.sh` | host-only build + test |
| `experiment/` | patch-neutral M1 harness, results, reproduction |

## Cost model and lambda calibration

Every decision uses the same convention:

```text
cost = SSE + lambda * rate_bits
lambda = sse_scale * quant_step^2
```

`lambda` comes from `av2enc::rdo::lambda_sse_from_quant_step()`.

**The step matters.** M1 pins `base_q_idx = 90`. At that QP the dequantiser is

```text
dequant = Round2(level * 528, 3) / dq_denom
```

so the reconstruction step in sample-domain distortion units is **66**, not
528. `txv::kQuantStep = 528` is the forward quantiser-table entry; feeding it to
`lambda_sse_from_quant_step()` inflates lambda by `8^2 = 64x`. This layer uses
66 by default (`kM1Qp90QuantStep`) and exposes `kCalibratedSseScale = 0.5`.

`0.5` is the measured compromise on the two QP-90 fixtures (see the experiment
below): lambda = 2178 at QP 90. It is not a target-corpus guarantee. The sweep
is non-monotonic because decisions are discrete; calibrate on the target
content before changing a default.

RDOQ is in the **forward-coefficient** domain, where the step is
`1 / TxParams::quant_factor` (8.25 for M1) and the equivalent pixel-domain
lambda is `64x` larger. `rdoq_lambda_from_pixel_lambda()` performs that
conversion. The repository RDOQ bring-up uses `sse_scale = 0.125` in that
domain; the experiment keeps that separate from the pixel-domain decision
scale.

## Rate callbacks

- Intra: `av2enc::rdo::IntraRateEstimator` (existing framework hook).
- Chroma: `ChromaRateEstimator`.
- Partition: the caller returns rate bits directly in `PartitionCostInputs`;
  there is no hidden model.
- TX type / RDOQ: `TransformStreamRate` owns the default AV2 static-CDF model
  and exposes `set_rate_override(coefficient_fn, eob_fn, context)`. The
  integrator can replace it with exact §8.2 costs or a measured table without
  changing the decision code.

The default stream model is the same §9.3/qctx-0 model as
`src/encoder/rdoq/rdoq_av2_model.cpp`, but reimplemented with fixed 1024-entry
arrays. It is derived from the normative tables, not from another encoder.

### Determinism and allocation

- All selectors compare exact binary64 values: cost, then distortion, then
  rate, then a fixed candidate order. No epsilon and no hash iteration.
- Intra follows the framework order: all 13 base modes, then angle deltas
  `-3,-2,-1,1,2,3`.
- Chroma: cost, total SSE, rate, non-CfL before CfL, mode, angle, input index.
- Partition keeps `NONE, SPLIT, HORZ, VERT` and spec-forced/implied behavior.
- TX type: cost, SSE, numeric `TxType` (DCT_DCT first), and it filters
  identity-based types because M1 signals `enable_idtx_intra = 0`.
- RDOQ: cost, distortion, rate, smaller magnitude, terminal before
  non-terminal, smaller signed level.
- The hot decisions allocate nothing: candidates and scratch are caller-owned
  spans, the stream adapter and RDOQ workspace are fixed-size objects, and the
  RDOQ trellis runs in `RdoqWorkspace`. The partition solver is a frame-level
  tree operation and uses `partition::decide_frame()`'s tree storage by design.

Build with `-ffp-contract=off` when comparing CPU and GPU decisions.

## Exact API the integrator should call

`#include "encoder/m1/rdo_decide/rdo_decide.h"`.

### Luma intra

```cpp
using namespace av2enc::m1::rdo_decide;

LambdaConfig config;          // defaults: step 66, sse_scale 0.5
const double lambda = lambda_sse(config);

std::array<av2enc::rdo::IntraCandidateCost, 61> candidates;
std::array<av2enc::rdo::IntraCandidateCost, 61> scratch;
int count = 0;
TransformStreamRate stream;
for (each predicted candidate) {
  TransformStreamBlock block;
  block.plane = 0;
  block.tx_size = plan.tx_size;
  block.w = plan.adj_w;
  block.h = plan.adj_h;
  block.levels_scan = scan_order_levels;   // plan.adj_w * plan.adj_h
  block.eob = plan.eob;
  stream.init(block);

  auto& c = candidates[count++];
  c.mode = av2_y_mode(params.mode);
  c.angle_delta = params.angle_delta;
  c.sse = static_cast<double>(sse);
  c.syntax_rate_bits = intra_syntax_bits(c.mode, c.angle_delta);
  c.coefficient_rate_bits = intra_coefficient_bits(stream);
}
const auto choice = decide_luma_intra(
    std::span<const av2enc::rdo::IntraCandidateCost>(candidates.data(), count),
    {}, config);
// choice.found / choice.mode / choice.angle_delta
```

### Partition

```cpp
PartitionRawCostFunction raw = [&](const partition::PartitionNodeInfo& info) {
  PartitionCostInputs c;
  c.leaf_distortion = variance_sse(info);          // caller's distortion
  c.leaf_rate_bits = estimate_leaf_rate_bits(c.leaf_distortion, w, h, 66.0);
  c.none_signal_bits = 0.0;
  c.split_signal_bits = 2.0;
  c.horz_signal_bits = 2.0;
  c.vert_signal_bits = 2.0;
  return c;
};

PartitionDecideConfig pconfig;
pconfig.lambda = lambda_sse(config);
pconfig.rate_aware = config.rate_aware;
pconfig.sse_only_split_cost = 2.0 * pconfig.lambda;  // control mode only

std::string error;
auto decision = decide_frame_rdo(geometry, raw, partition_options, pconfig,
                                 &error);
// decision.status == partition::DecisionStatus::kOk
// decision.frame is the same FrameDecision the current M1 code consumes.
```

`estimate_leaf_rate_bits()` is a variance-based ranking aid, not an entropy
estimate: the partition search runs before prediction/transform because the
tree fixes reconstruction order. Replace it with a per-node cost when the
integration has one.

### TX type

```cpp
TxTypeDecisionConfig tx_config;
tx_config.lambda = lambda_sse(config);   // pixel-domain SSE from txtype
tx_config.rate_aware = config.rate_aware;
tx_config.enable_idtx_intra = false;     // M1 sequence header

std::array<std::int32_t, 1024> level_scratch, levels;
std::array<std::int16_t, 1024> recon_scratch, recon;
TransformStreamRate stream;
TxTypeDecision tx_decision;
decide_tx_type(residual_coded, coded_w, coded_h, mapped_mode, 0,
               /*dct_only=*/false, tx_config, level_scratch, recon_scratch,
               levels, recon, stream, tx_decision, &error);
// tx_decision.candidate.type / .eob / .sse / .rate_bits / .cost
```

### RDOQ

```cpp
RdoqConfig rq;
rq.quant_step = 1.0 / params.quant_factor;
rq.lambda = lambda_sse_from_quant_step(rq.quant_step, 0.125);  // coeff domain
rq.prev_level_cap = 2;

RdoqWorkspace workspace;               // ~40 KiB, one per encoder/thread
TransformStreamRate stream;            // ~28 KiB, one per encoder/thread
RdoqDecision rq_decision;
decide_rdoq(original_scan, scalar_levels_scan, plane, tx_size, w, h,
            txb_skip_context, dc_sign_context, rq, workspace, stream,
            levels_out, rq_decision, &error);
// levels_out[0..rq_decision.eob), then zero
```

The experiment reproduces the repository's full-trellis RDOQ byte-for-byte
(`806` bytes / `11294` bytes on the two fixtures), so this path is a verified
allocation-free replacement for `av2enc::rdoq::rdoq_quantize()`.

### Chroma

```cpp
ChromaCandidateCost c;
c.is_cfl = is_cfl;
c.mode = av2_y_mode(mode);
c.angle_delta = derived_angle_delta;
c.block_w = cw;
c.block_h = ch;
c.sse_u = sse_u;
c.sse_v = sse_v;
c.syntax_rate_bits = default_chroma_syntax_bits(c.is_cfl, c.mode, c.angle_delta);
c.coefficient_rate_bits = stream_bits_u + stream_bits_v;

ChromaChoice choice = decide_chroma(candidates, {}, config);
// choice.is_cfl / choice.mode / choice.sse_total / choice.cost
```

A U-only SSE score can pick a mode that is worse for V; the selector always
uses `u_sse + v_sse`.

## CPU experiment (QP 90, one frame, one tile)

Harness: `experiment/run_m1_rdo_decide.sh`. It generates a patched copy of
`m1_still.cpp` under `build-rdo-decide/` (the repository file is untouched),
checks that the `dc` and `sse` cases are byte-identical to the unpatched CLI,
decodes every stream with `dav2d`, scores with `tools/quality.py` (PSNR,
SSIM, CVVDP), and verifies every RDO stream per sample with
`tools/m1_verify/verify.py`. All shown RDO streams are **bit-exact**.

```sh
cd src/encoder/m1/rdo_decide/experiment
./run_m1_rdo_decide.sh --scales 0.25,0.5,1.0
```

### 64x64, `testdata/m0_64x64_1f.y4m`

| Case | Bytes | PSNR-Y | PSNR-avg | CVVDP |
|---|---:|---:|---:|---:|
| DC (default) | 936 | 42.5016 | 43.2809 | 9.9009 |
| SSE-only full intra | 979 | 42.5048 | 43.2627 | 9.9035 |
| RDO partition | 988 | 42.5141 | 43.1764 | 9.8931 |
| RDO chroma | 973 | 42.5048 | 43.2885 | 9.8960 |
| RDO intra+chroma | 943 | 42.4704 | 43.2613 | 9.9175 |
| TX rdo-decide (isolated) | 975 | 42.5301 | 43.2477 | 9.9024 |
| RDOQ rdo-decide (isolated) | 806 | 41.5128 | 42.2818 | 9.9003 |
| RDO full @0.25 | 954 | 42.4666 | 43.3018 | 9.9075 |
| **RDO full @0.5** | **955** | 42.4526 | 43.2232 | **9.9151** |
| RDO full @1.0 | 947 | 42.4822 | 43.2670 | 9.8925 |

### 640x360, `tests/data/m0_640x360_1f.y4m`

| Case | Bytes | PSNR-Y | PSNR-avg | CVVDP |
|---|---:|---:|---:|---:|
| DC (default) | 13,355 | 45.0892 | 46.3683 | 9.8867 |
| SSE-only full intra | 13,199 | 45.0227 | 46.3016 | 9.8843 |
| RDO partition | 14,100 | 47.5997 | 48.5749 | 9.9413 |
| RDO chroma | 13,153 | 45.0227 | 46.3069 | 9.8845 |
| RDO intra+chroma | 12,971 | 45.0622 | 46.3484 | 9.8863 |
| TX rdo-decide (isolated) | 13,135 | 45.0358 | 46.3136 | 9.8833 |
| RDOQ rdo-decide (isolated) | 11,294 | 44.5722 | 45.8045 | 9.8597 |
| RDO full @0.25 | 13,848 | 47.8425 | 48.7908 | 9.9451 |
| **RDO full @0.5** | **13,770** | 47.6242 | 48.6085 | **9.9415** |
| RDO full @1.0 | 13,724 | 47.6083 | 48.5983 | 9.9403 |

Budgets versus the DC baseline at the calibrated scale (0.5):

| Fixture | Bytes | PSNR-Y | CVVDP |
|---|---:|---:|---:|
| 64x64 | +19 (+2.0%) | -0.049 dB | **+0.0142 JOD** |
| 640x360 | +415 (+3.1%) | **+2.535 dB** | **+0.0548 JOD** |

### Controlled partition comparison (same lambda, no intra/chroma RDO)

This isolates the leaf-rate term from the lower split penalty. Both rows use
`lambda` from `sse_scale = 0.5`; the SSE-only control charges the same two
split bits but no leaf rate.

| Fixture | Partition mode | Bytes | PSNR-Y | CVVDP |
|---|---|---:|---:|---:|
| 64x64 | SSE-only | 993 | 42.4243 | 9.8941 |
| 64x64 | rate-aware | 988 | 42.5141 | 9.8931 |
| 640x360 | SSE-only | 13,924 | 46.9455 | 9.9280 |
| 640x360 | rate-aware | 14,100 | 47.5997 | 9.9413 |

## Do full intra / partition splits become net wins?

- **Full intra rate-aware is a rate fix, not a universal quality win.** With
  no partition change (`RDO intra+chroma`), 640x360 saves 228 bytes versus the
  SSE-only full search at essentially the same PSNR/CVVDP; 64x64 is small
  (+7 B versus DC, +0.017 JOD). It repairs the SSE-only full-intra regression
  (`--m1-full-intra` was 156 B *larger* and slightly worse than DC here).
- **Partition splits are a net RD win on 640x360, not on 64x64.** At the same
  lambda, the leaf-rate term buys +0.65 dB / +0.013 JOD for +1.3% bytes on
  640x360, and is neutral on 64x64 (-5 B, +0.09 dB, -0.001 JOD). The full
  rate-aware stack at 640x360 spends +3.1% bytes for +2.54 dB / +0.055 JOD;
  that is a genuine RD improvement, not just a byte reduction.
- **The gain is content- and scale-dependent.** At `sse_scale = 2.0` the
  640x360 stack collapses to 42.7 dB, and at `4.0` to 37.1 dB: the mode search
  starts choosing low-rate DC-ish modes. The calibration cliff is real; sweep
  before changing the constant.
- **RDOQ is parity with the repository path** (identical byte counts and
  metrics), which validates the allocation-free trellis and stream adapter.
- **TX type is a small trade.** The isolated layer saves 64 bytes on 640x360
  with -0.001 JOD, while the existing per-nonzero selector buys +0.003 JOD for
  +57 bytes. Both are bit-exact. The stream-rate selector is more
  rate-oriented; the choice of objective should be calibrated like lambda.

## Integration notes

1. Add the layer's `.cpp` files to `av2enc_core` (owned by the build agent):
   `config.cpp`, `transform_stream.cpp`, `intra_rdo.cpp`, `chroma_rdo.cpp`,
   `partition_rdo.cpp`, `tx_rdo.cpp`. No CUDA headers, no globals.
2. Keep `LambdaConfig` in one place per frame. M1 pins QP 90, so
   `calibrated_m1_config()` is the default; pass the real dequant step if M1
   ever leaves QP 90.
3. Replace the SSE-only winner in `choose_luma_mode()` with
   `decide_luma_intra()`. The current `--m1-rdo-intra` path is the same idea but
   uses the placeholder coefficient model; this layer's stream model is closer
   to the writer.
4. Replace the partition cost lambda in `Encoder::encode()` with
   `decide_frame_rdo()`. Keep `partition::validate_frame()` in the path: the
   solver still owns spec legality. Do not add `PARTITION_SPLIT` below 128x128;
   the module reports it as not allowed.
5. `compute_tx()` can call `decide_tx_type()` before its quantise path and
   `decide_rdoq()` inside it. The writer already supports non-DCT TX types and
   arbitrary levels, so neither changes syntax. `enable_idtx_intra = 0` must be
   honoured (the layer does; the raw `txtype` candidate list does not).
6. Chroma: use `decide_chroma()` with `u_sse + v_sse`; the current default
   path scores DC on U only.
7. Re-run `tools/validate_dav2d.sh`, `tools/m1_verify/run_matrix.sh`, and the
   pinned MD5s after wiring. The experiment's patch-neutrality check proves the
   layer is inert while disabled, but the integration still owns the default
   policy change.
8. Build with `-ffp-contract=off` for CPU/GPU decision parity. The GPU port
   must reproduce the same comparator orders and avoid per-block allocation.

## Non-claims

- CPU-only, one-frame, correctness/experiment results. No speed claim.
- No bitstream syntax change and no reference MD5 change: the layer emits no
  bits and the experiment only changes mode/partition/level decisions, then
  gates every stream through `dav2d`.
- The static-CDF rate model and the partition leaf-rate proxy are not exact
  §8.2 costs. Replace them through the documented callbacks before quoting a
  BD-rate number.

## Build and test

```sh
cd src/encoder/m1/rdo_decide
./build_and_test.sh
SANITIZE=1 ./build_and_test.sh
WERROR=1 ./build_and_test.sh
```

The experiment is described in `experiment/README.md` and runs from
`experiment/run_m1_rdo_decide.sh`.

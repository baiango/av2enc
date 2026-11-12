# M2 P-frame rate-aware decision layer (`src/encoder/m2/rdo/`)

Status: **component + CPU experiment.** It is not wired into `CMakeLists.txt`,
`src/main.cpp`, or `src/encoder/m2/video_frame.cpp`. It owns decisions, not
syntax: it writes no bits and does not change a reference stream. The
integration pass owns enabling it by default.

This directory adds one rate-aware decision layer for the M2 P-frame path:

| Decision | API | Experiment hook |
|---|---|---|
| skip / merge / NEWMV / intra fallback | `inter_decision.h` | `AV2ENC_VIDEO_RDO=1` |
| residual `skip_flag` | `residual_decision.h` | `AV2ENC_VIDEO_RDO_NO_RESIDUAL=1` disables |
| partition NONE / HORZ / VERT | `partition_decision.h` | `AV2ENC_VIDEO_RDO_NO_PART=1` disables |
| syntax rate model | `rate_model.h` | pluggable callbacks |

The shared pieces stay shared:

- lambda calibration: `src/encoder/m1/rdo_decide/config.h`
- deterministic candidate selection: `src/encoder/rdo/rdo.h`
- MV-stack / skip / merge resolution: `src/encoder/m2/merge/`
- residual transform, quantise, EOB, recon and symbol-cost measurement:
  `src/encoder/m2/residual/`
- intra coefficient rate: `src/encoder/m1/rdo_decide/transform_stream.h`
- partition geometry and legality: `src/encoder/partition/`

This layer does not reimplement any of that math.

## Files

| File | Purpose |
|---|---|
| `config.h/.cpp` | M1 QP-90 lambda (step 66), metric/tie-break config |
| `context.h/.cpp` | `RateContext` adapters over `m2/merge` (skip / new-MV ctx) |
| `rate_model.h/.cpp` | §9.3 CDF mode/MV/ref bits + pluggable callbacks |
| `residual_decision.h/.cpp` | `skip_flag` decision on top of `m2/residual` |
| `inter_decision.h/.cpp` | candidate cost, scoring, deterministic selection |
| `intra_fallback.h/.cpp` | M1 `TransformStreamRate` for the DC fallback |
| `partition_decision.h/.cpp` | M1 rate-aware partition solver wrapper |
| `rdo.h` | umbrella include |
| `video_rdo_test.cpp` | standalone self-tests |
| `build_and_test.sh` | host-only build + tests |
| `experiment/` | patch-neutral M2 harness, results, reproduction |

## Cost model and lambda

Every decision uses the same convention:

```text
cost = distortion + lambda * rate_bits
```

Lambda comes from the M1 QP-90 calibration, not a second copy of the
calibration math:

```text
lambda_sse  = sse_scale * 66^2          # 66 = QP-90 reconstruction step
lambda_sad  = sqrt(lambda_sse)          # L1-like distortion
lambda_satd = sqrt(lambda_sse)
```

`calibrated_video_config()` returns the M1 profile: step **66**, `sse_scale`
**0.5**, so `lambda_sse = 2178`. The experiment runs the M2 default
`base_q_idx = 90`, which is the condition under which that calibration is
valid. The core now accepts other QPs, but the current `m2/residual` decision
is QP-90-only: if the integration leaves QP 90, set `LambdaConfig::quant_step`
to the real dequant step (`qlookup(base_q_idx)/8`) and extend the residual
path with the frame quant step first.

The square-root relation is the one documented by `src/encoder/rdo/rdo.h`
("SAD/SATD costs are L1-like, so lambda is the square root of the SSE
lambda"). Distortion units and lambda units must match; callers use
`lambda_for_metric()` and `selector_config()` rather than forming lambda.

The integrated experiment defaults to SSE for the final selection because the
residual decision reconstructs and scores SSE. SAD/SATD pre-selection is wired
through `InterCandidateSpec::has_distortion_override` (the gate hook is
`AV2ENC_VIDEO_RDO_METRIC=1|2`) and is unit-tested; mixing a SAD distortion with an
SSE lambda is not allowed by the API.

## Candidate families

The M2 header has `NumTotalRefs = 1`, `reference_select = 0`, so true
compound `skip_mode` is not signalable yet (see `src/encoder/m2/README.md`).
The names here mean:

- **skip GLOBALMV** — `skip_flag = 1` with the zero MV;
- **merge NEARMV** — `skip_flag = 1/0` with the spatial MV-stack candidate
  (`src/encoder/m2/merge/`), DRL index selectable;
- **NEWMV** — the ME result plus a signalled shell-coded MVD;
- **intra fallback** — the P-frame DC intra block.

A residual variant of any inter family keeps the kind and sets
`skip_flag = false`. The layer is ready for true `skip_mode` candidates when
the multi-reference header lands; they become another `InterCandidate` with
two references and the compound mode bits.

## Rate model

`InterRateModel` charges:

- `is_inter` (§5.20.7.3), `skip_flag` (§5.20.5.10), `single_mode`
  (§5.20.7.6), `drl_idx` (§5.20.7.8);
- the shell-coded MVD for NEWMV (§5.20.7.20), transcribed from the writer's
  `write_mv()`;
- reference syntax: **0 bits** with the M2 profile (`NumTotalRefs = 1`); a
  `ref_bits` callback is available for multi-reference work.

The MVD shell model is exact for a given MVD: the shell CDFs do not depend on
neighbours and M2 sets `disable_cdf_update = 1`. `is_inter`, `skip_flag` and
`single_mode` use neutral contexts by default (0); they are 1-2 bits each and
are fully replaceable through the callbacks. The writer's exact neighbouring
contexts are the production hook.

Coefficient rate is **not** reimplemented here:

- inter residual bits come from `residual::decide_inter_block()`, which
  measures the writer's own symbol sequence (TX type, EOB, base/BR codes,
  signs, escapes), including `skip_flag`;
- intra fallback coefficient bits come from the M1 `TransformStreamRate`
  adapter in `intra_fallback.cpp`.

### Pluggable callbacks

`RateCallbacks` exposes:

```cpp
double (*mode_bits)(void*, const ModeRateInput&);
double (*residual_bits)(void*, const residual::BlockDecision&);
double (*intra_syntax_bits)(void*, int y_mode, int y_angle_delta,
                            int uv_mode, int is_inter_ctx);
double (*ref_bits)(void*, int ref);
```

A null callback selects the default backend. The integrator can replace any
subset with exact §8.2 costs or a measured table without changing the
decision code.

`context.h` fills `RateContext` from the neighbor grids by delegating to
`m2/merge` (`skip_flag_context`, `derive_new_mv_context`); it does not
reimplement the §7.11.2 / §5.20.5.10 math. The experiment uses neutral
contexts because the core does not track a neighbour skip/`YMode` grid.

## Deterministic tie-breaks

`decide_inter_candidates()` feeds the shared `rdo::CandidateSelector`, so the
order is the documented one:

1. lower total cost;
2. lower distortion or lower rate (per `RdoConfig::tie_break`;
   default distortion first);
3. lower `CandidateKind`;
4. lower `id` (mode family: `kSkipGlobal` < `kMerge` < `kNewMv` < `kIntra`);
5. lower `aux` (`drl_index * 2 + (skip ? 0 : 1)`, so skip precedes residual);
6. lower generation index.

Comparisons are exact; no epsilon and no hash iteration. Candidate generation
is fixed: GLOBALMV, then NEARMV in ascending DRL order, then NEWMV, then intra.

## API the integrator should call

`#include "encoder/m2/rdo/rdo.h"`.

### One inter candidate

```cpp
using namespace av2enc::video::rdo;

RdoConfig config = calibrated_video_config();
const InterRateModel model;
RateContext ctx;  // neutral 0 contexts; derive exact neighbours when available

InterCandidateSpec spec;
spec.kind = InterCandidateKind::kNewMv;
spec.inter_mode = syntax::kInterNewMv;  // 0 GLOBAL, 1 NEAR, 2 NEW
spec.drl_index = 0;
spec.mv = resolved.mv[0];
spec.mvd = resolved.mvd[0];
spec.ref = 0;

ScoredInterCandidate scored;
std::string error;
if (!score_inter_candidate(spec, block_input, /*evaluate_residual=*/true,
                           ctx, config, model, scored, &error)) { ... }
// scored.candidate.distortion / .mode_bits / .residual_bits / .rate_bits
// scored.residual.skip / .plane[].recon
```

`block_input` is a `residual::BlockInput` with block-local source and
prediction plane views. `evaluate_residual = false` still measures the skip
distortion (total Y+U+V SSE) and `skip_flag` cost through the shared residual
path, so every candidate uses consistent distortion units.

The M2 P-frame path requires a codable chroma residual size. With the default
`min_block_size = 16` the chroma block is at least 8x8; 4x4 luma leaves would
produce a 2x2 chroma residual, which `txv` does not support. Keep the layer on
the M2 default tree policy (or the `--single-block` stress mode).

### Final inter vs intra selection

```cpp
std::array<InterCandidate, 5> candidates;
// ... push the scored inter candidates ...
candidates[n++] = make_intra_candidate(
    intra_sse_total, intra_coeff_bits, 0, 0, 0, ctx, config, model);
const InterDecision decision =
    decide_inter_candidates(std::span(candidates.data(), n), config, &error);
// decision.best_index indexes the input span; decision.margin is the
// best-to-second cost gap.
```

Render the chosen candidate:

- `skip_flag = decision.candidate.skip_flag`;
- if skip, reconstruct with the candidate prediction;
- otherwise use `scored.residual.plane[plane].recon` and
  `residual::append_scan_levels(scored.residual.plane[plane], &levels)`.

### Residual only

```cpp
residual::DecisionOptions options = make_residual_options(config);
residual::BlockDecision residual;
if (!decide_residual(block_input, config, /*allow_residual=*/true,
                     /*force_residual=*/false, residual, &error)) { ... }
// residual.skip, .plane[].recon, .rate_bits, .distortion
```

`make_residual_options()` fixes `base_q_idx = 90`, `lambda = 2178`,
DCT_DCT only, TX_MODE_LARGEST, no RDOQ and no inter DDT, matching the M2 tile
writer.

### Partition

```cpp
PartitionRdoConfig pconfig;
pconfig.rdo = calibrated_video_config();
LeafCostFunction leaf = [&](const partition::PartitionNodeInfo& info) {
  LeafCost cost;
  cost.distortion = /* real inter leaf distortion */;
  cost.rate_bits = /* real inter leaf rate bits */;
  return cost;
};
partition::FrameDecision frame =
    decide_partition_rdo(geometry, options, leaf, pconfig, &error);
// Keep partition::validate_frame() in the path: the solver owns spec legality.
```

`variance_leaf_rate_bits()` is the documented M1 variance proxy used by the
experiment. It is a ranking aid, not a real inter cost. Production should
return the actual per-node inter cost from the callback; the experiment shows
the proxy is not a reliable partition win.

## Exact integration call sites

Build wiring (owned by the build agent): add `config.cpp`, `context.cpp`,
`rate_model.cpp`, `residual_decision.cpp`, `inter_decision.cpp`,
`intra_fallback.cpp` and `partition_decision.cpp` to `av2enc_core`. The layer
has no CUDA headers and no globals; it links against the existing M1 RDO,
`encoder/rdo`, `m2/merge` and `m2/residual` sources.

Line numbers are from the working tree at experiment time. The anchors
(function names) are the stable references; the core file moved during this
work (QP helper landed concurrently), so re-check anchors before editing.

1. **Partition decision** — `InterFrameEncoder::encode()`,
   `src/encoder/m2/video_frame.cpp:359-375`.
   Replace the `lambda = 0.85 * 2^((qp-12)/6)` / `signal_cost` /
   `cost_fn` / `partition::decide_frame` block with
   `rdo::decide_partition_rdo()`. Guard the P-frame path:
   `if (!keyframe_)`. Keep `partition::validate_frame()` immediately after
   the decision (lines 376-378). The keyframe must keep its existing
   intra partition decision.

2. **Inter candidate scoring** — `InterFrameEncoder::encode_leaf()`,
   `src/encoder/m2/video_frame.cpp:579` (`consider` lambda).
   Replace the SSE-only `cand.sse < best.sse` comparison. The three candidate
   producers are:
   - GLOBALMV: line 592 (`cand global mv=(0,0)`),
   - NEARMV: lines 611-644 (`cand near drl=...`),
   - NEWMV: lines 641-677 (`cand new me=...`).
   Build an `InterCandidateSpec` at each producer, call
   `score_inter_candidate()` with the candidate's `predict_all()` planes, and
   collect the results. Select with `decide_inter_candidates()`.

3. **Residual `skip_flag`** — `encode_leaf()` lines 709-765.
   Replace `compute_tx()` + `estimate_tx_bits()` + the `want` comparison with
   `rdo::decide_residual()`:
   - `allow_residual = options_.enable_inter_residual ||
     options_.force_inter_residual`;
   - `force_residual = options_.force_inter_residual`;
   - `skip_flag = residual.skip ? 1 : 0`;
   - if not skip, write `residual.plane[plane].recon` and append levels with
     `residual::append_scan_levels()`.
   This removes the 3-bit + 2.5-bits/nonzero + 0.5-bits/EOB proxy at lines
   251-260 and uses the writer's own symbol cost.

4. **Intra fallback comparison** — `encode_leaf()` lines 774-776
   (`use_intra = ... intra_sse < inter_luma_sse`).
   Replace the luma-only SSE comparison with:
   - `intra_coefficient_bits()` over the three `TxPlan`s;
   - `make_intra_candidate(intra_total_sse, coeff_bits, 0, 0, 0, ...)`;
   - include it in the `decide_inter_candidates()` list.
   The current comparison ignores the intra residual's rate, which is the
   main reason the baseline picks expensive intra blocks on P frames.

5. **Keyframe**: do not call the P-frame layer. `collect_node()` must dispatch
   `encode_leaf_rdo()` only when `!keyframe_` (the experiment hook is at
   `src/encoder/m2/video_frame.cpp:439-441`).

## CPU experiment (1280x720, QP 90)

Harness: `experiment/run_video_rdo_experiment.sh`. It:

1. builds the baseline gate from the repository `video_frame.cpp`;
2. builds an RDO gate from a generated copy (the repository file is untouched);
3. checks that the RDO binary with the hook **off** is byte-identical to the
   baseline binary;
4. decodes every stream through `experiment/dav2d_strict.sh`, which forces
   `dav2d --strict 1 -q -i <stream> -o <decoded.y4m> --threads 0`;
5. compares decoded pictures against the encoder reconstruction sample by
   sample (0 diffs required);
6. runs `tools/quality.py` for PSNR/CVVDP.

The fixture is `testdata/shine_1280x720_60f.y4m`. Results (bytes are the
gate's `total_bytes`; CVVDP is JOD):

### 2-frame, single 64x64 leaf per superblock

| variant | bytes | Δbytes | PSNR-Y | ΔPSNR-Y | PSNR-avg | CVVDP | ΔCVVDP |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 38,469 | +0.00% | 34.887 | +0.000 | 36.569 | 9.3991 | +0.0000 |
| baseline `--inter-residual` | 38,469 | +0.00% | 34.887 | +0.000 | 36.569 | 9.3991 | +0.0000 |
| rate-aware full | **36,323** | **-5.58%** | 34.864 | -0.024 | 36.546 | 9.3990 | -0.0001 |
| rate-aware, baseline partition | 36,323 | -5.58% | 34.864 | -0.024 | 36.546 | 9.3990 | -0.0001 |
| rate-aware, no residual coding | 36,323 | -5.58% | 34.864 | -0.024 | 36.546 | 9.3990 | -0.0001 |
| rate-aware, distortion-only selection | 39,075 | +1.58% | 34.887 | +0.000 | 36.569 | 9.3991 | +0.0000 |

### 2-frame, full variance-driven tree

| variant | bytes | Δbytes | PSNR-Y | ΔPSNR-Y | PSNR-avg | CVVDP | ΔCVVDP |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 74,512 | +0.00% | 45.331 | +0.000 | 46.392 | 9.8043 | +0.0000 |
| baseline `--inter-residual` | 74,512 | +0.00% | 45.331 | +0.000 | 46.392 | 9.8043 | +0.0000 |
| rate-aware full | **44,106** | **-40.81%** | 45.002 | -0.329 | 46.091 | 9.8037 | -0.0006 |
| rate-aware, baseline partition | 43,975 | -40.98% | 45.002 | -0.329 | 46.091 | 9.8037 | -0.0006 |
| rate-aware, no residual coding | 44,106 | -40.81% | 45.002 | -0.329 | 46.091 | 9.8037 | -0.0006 |
| rate-aware, distortion-only selection | 81,590 | +9.50% | 46.439 | +1.108 | 47.323 | 9.8062 | +0.0019 |

### 8-frame, single 64x64 leaf per superblock

| variant | bytes | Δbytes | PSNR-Y | ΔPSNR-Y | PSNR-avg | CVVDP | ΔCVVDP |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 41,447 | +0.00% | 35.250 | +0.000 | 36.919 | 9.4077 | +0.0000 |
| baseline `--inter-residual` | 41,447 | +0.00% | 35.250 | +0.000 | 36.919 | 9.4077 | +0.0000 |
| rate-aware full | **36,831** | **-11.14%** | 35.237 | -0.013 | 36.906 | 9.4065 | -0.0012 |
| rate-aware, baseline partition | 36,831 | -11.14% | 35.237 | -0.013 | 36.906 | 9.4065 | -0.0012 |
| rate-aware, no residual coding | 36,831 | -11.14% | 35.237 | -0.013 | 36.906 | 9.4065 | -0.0012 |
| rate-aware, distortion-only selection | 46,493 | +12.17% | 35.241 | -0.009 | 36.911 | 9.4078 | +0.0001 |

### 8-frame, full variance-driven tree

| variant | bytes | Δbytes | PSNR-Y | ΔPSNR-Y | PSNR-avg | CVVDP | ΔCVVDP |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 245,609 | +0.00% | 45.497 | +0.000 | 46.516 | 9.8126 | +0.0000 |
| baseline `--inter-residual` | 245,609 | +0.00% | 45.497 | +0.000 | 46.516 | 9.8126 | +0.0000 |
| rate-aware full | **46,460** | **-81.08%** | 43.343 | -2.154 | 44.547 | 9.8037 | -0.0089 |
| rate-aware, baseline partition | 45,544 | **-81.46%** | 43.827 | -1.670 | 44.990 | 9.8043 | -0.0083 |
| rate-aware, no residual coding | 46,460 | -81.08% | 43.343 | -2.154 | 44.547 | 9.8037 | -0.0089 |
| rate-aware, distortion-only selection | 289,866 | +18.02% | 47.614 | +2.118 | 48.242 | 9.8292 | +0.0166 |

### Matched-quality lambda sweep

The scale is not a rate-control knob; it is the `sse_scale` of the same
decision layer. Lower scale buys quality with more M2 decision rate. On the
8-frame full tree:

| `AV2ENC_VIDEO_RDO_SCALE` | bytes | PSNR-Y | PSNR-avg | CVVDP |
|---|---:|---:|---:|---:|
| baseline (no RDO) | 245,609 | 45.497 | 46.516 | 9.8126 |
| 0.05 | **46,907** | 45.208 | 46.204 | 9.8079 |
| 0.10 | 46,660 | 44.511 | 45.608 | 9.8049 |
| 0.20 | 46,545 | 44.208 | 45.335 | 9.8048 |
| 0.25 | 46,497 | 43.989 | 45.137 | 9.8047 |
| 0.50 (default) | 46,460 | 43.343 | 44.547 | 9.8037 |

At scale 0.05 the rate-aware path is within **0.29 dB** of the baseline PSNR
while using **80.9% fewer bytes**, and CVVDP differs by **-0.0047 JOD**. The
marginal PSNR above that point is very expensive for the baseline.

SAD/SATD pre-selection is gate-tested as a cross-check. On the 2-frame
single-leaf case (`AV2ENC_VIDEO_RDO_METRIC=1` and `=2`) the P frame is 93 bytes
and PSNR-Y is 34.877 / 34.826 dB, versus 84 bytes / 34.864 dB for SSE. The
metric changes which candidate wins, and every stream still passes
`dav2d --strict 1` with 0 recon diffs. SSE remains the default because it is
the distortion the residual decision actually minimises.

## Does the video path improve?

**Bottom line:** yes for rate at matched perceptual quality. The current
P-frame heuristic compares luma SSE only and spends bytes on intra refresh;
the rate-aware layer charges the bits it is about to code. On the 8-frame full
tree the default scale saves **81%** of the bytes for **-0.009 JOD**; the
matched-quality point (scale 0.05) is within **0.29 dB** PSNR-Y of the
baseline for **80.9% fewer bytes**. The PSNR at the default scale does drop
2.15 dB because the rate-aware P frames do not refresh; that is a different
point on the rate curve, not a loss at matched quality. Partition RDO with the
variance proxy is neutral/negative and should wait for a real inter leaf cost;
the residual decision is neutral on this low-motion fixture.

**Mode/intra selection: yes, strongly.** The baseline compares luma SSE only:
`intra_sse < inter_luma_sse` at `video_frame.cpp:772-774`. It never charges the
intra residual's rate, so P frames choose hundreds of intra blocks with full
coefficients. The rate-aware layer charges mode/MV/residual bits and keeps
those blocks inter-skip. On this fixture that is:

- **-5.58%** bytes (2f, single leaf), CVVDP -0.0001;
- **-40.81%** bytes (2f, full tree), CVVDP -0.0006;
- **-11.14%** bytes (8f, single leaf), CVVDP -0.0012;
- **-81.08%** bytes (8f, full tree), CVVDP -0.0089.

The 8-frame full-tree PSNR drop is -2.15 dB because the baseline spends bytes
on intra refresh in every P frame. Per-frame luma PSNR (decoded Y4M):

| frame | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 45.295 | 45.366 | 45.495 | 45.577 | 45.594 | 45.571 | 45.565 | 45.521 |
| rate-aware | 45.295 | 44.727 | 43.839 | 43.407 | 43.195 | 42.585 | 42.387 | 42.249 |

The keyframe is identical; the rate-aware P frames drift because they do not
refresh. At the matched-quality scale (0.05) the whole 8-frame stream stays at
45.208 dB Y, so the low-scale point is the fair "same quality, much lower
rate" comparison. A proper BD-rate claim still needs a rate-control sweep and
a second content fixture; this report does not make one.

**Residual `skip_flag`: implemented, neutral on this clip.** The rate-aware
residual decision declines every residual on this low-motion fixture; the
baseline `--inter-residual` also declines every residual here (consistent with
`src/encoder/m2/README.md`). `rdo_full` and `rdo_no_residual` are byte-identical
for that reason. The unit tests exercise the skip/residual decision with a
constant-offset block. The RDO forced-residual path was also gate-tested:
`--force-inter-residual` with `AV2ENC_VIDEO_RDO=1` on the 2-frame single-leaf
case produced a 10,292-byte P frame that decoded with `--strict 1` and matched
the encoder reconstruction sample by sample.

**Partition split: neutral to slightly negative with the variance proxy.** The
M2 partition search runs before prediction, so the experiment uses the M1
`estimate_leaf_rate_bits()` variance proxy. At the same mode decisions,
`rdo_nopart` is 131 bytes smaller in the 2-frame tree and 916 bytes smaller
with +0.48 dB in the 8-frame tree. The proxy is not a real inter cost; the
production integration should return the actual per-node inter rate from
`LeafCostFunction` before enabling partition RDO by default. The solver and
legality checks are shared and ready for that.

**Distortion-only control: yes, the rate term is the active ingredient.**
Disabling the candidate-selection rate term but keeping the same API,
partition and residual path (`AV2ENC_VIDEO_RDO_SSE_ONLY=1`) costs **+1.58%**,
**+9.50%**, **+12.17%** and **+18.02%** bytes versus the baseline in the four
cases. The residual decision keeps its own SSE lambda in this control; only
the candidate selector and partition leaf-rate term are zeroed. It sometimes
buys PSNR/CVVDP (8f tree: +2.12 dB, +0.0166 JOD for +18% bytes); that is a
different point on the curve, not a rate win at matched quality.

## Non-claims

- CPU-only, one 1280x720 fixture, QP 90. No speed claim and no BD-rate claim.
- The default rate model is not exact §8.2: `is_inter`, `skip_flag` and
  `single_mode` use neutral contexts. Replace them through `RateCallbacks`
  before quoting production numbers.
- The partition leaf rate is the documented M1 variance proxy, not a real
  inter leaf cost. The experiment measures it as neutral/negative.
- The residual decision is neutral on this low-motion clip; it is exercised by
  unit tests and the forced-residual stress mode, not by a second corpus.
- Every shown stream was decoded with `dav2d --strict 1` and matched the
  encoder reconstruction sample by sample (0 diffs). That is a correctness
  gate, not a quality claim.
- The core `video_frame.cpp` moved during this work (a QP helper landed
  concurrently). Line numbers above are a snapshot; use the function anchors
  and re-run the gate after integration.

## Build and test

```sh
cd src/encoder/m2/rdo
./build_and_test.sh          # layer self-tests
WERROR=1 ./build_and_test.sh
SANITIZE=1 ./build_and_test.sh
./experiment/run_video_rdo_experiment.sh   # baseline vs rate-aware, strict decode
./experiment/run_video_rdo_scale_sweep.sh  # optional lambda sweep (reuses the gate)
```

The experiment writes `results.md` and `results.json` under its `--out`
directory (`${TMPDIR:-/tmp}/video_rdo_experiment` by default). Useful options:

```sh
./experiment/run_video_rdo_experiment.sh --cases 2f_sb,8f_tree
./experiment/run_video_rdo_experiment.sh --no-quality
./experiment/run_video_rdo_experiment.sh --input clip.y4m --out /tmp/m2-rdo
```

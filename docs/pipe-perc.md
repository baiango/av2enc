# Perceptual RD pipeline (`AV2ENC_PIPE_PERC`)

Status: **complete, env-gated, default off; no CLI flag.** Branch
`exp/pipe-perc`, worktree `pipe-perc`. Gate on:
`AV2ENC_PIPE_PERC=1`. The tuned point is λ-scale `0.2`, top-N `2`, detail
strength `1.0`, on the measured drift recipe. Gate off is byte-identical to
the three pinned 240p streams and the 720p pin.

## Why a pipeline, not a term

The earlier perceptual attempts changed one term inside an SSE/λ pipeline
calibrated around it: the detail-weighted TX lambda (`docs/perceptual-rd.md`),
AQ, QM dead zones, perceptual CDEF. Each was absorbed or lost because D, R and
λ did not move together. This branch is one decision path:

```text
source frame
  │
  ├─ 8x8 source-gradient weights (integer Q8, detail up only) ──┐
  │                                                             │
  ▼                                                             ▼
cheap inter candidates ── top-N ── predict ── TX search ── quantise ── recon
  │                              (context §8.3.2)      │
  │                                                   ├─ D = Σ w·e²
  │                                                   ├─ R = exact syntax
  │                                                   │    + context coeffs
  │                                                   └─ J = D + λ·R
  │                                                             │
  ├─ intra candidate ────────────────────────────────────────────┤
  │                                                             ▼
  └──────────── skip / residual / mode / TX all chosen under J ──┘
                                                                 │
                                              existing writer + entropy
                                                                 │
                                          measured drift refresh (frame)
```

The **writer, transform bitstream, quantiser and entropy coder are the
existing ones**. The gate changes only which candidates the decision selects.
The frame-level drift refresh is unchanged and remains part of the recipe.

## K oracle (required before tuning)

Definition from `docs/k-oracle.md`: `K*_P = sup_θ K_P(θ)` at 240p/100 kbps.
The wall is **K 83.80** (tuned drift, reproduced locally as 83.80 from
QP110 100.584 kbps K83.94 and QP111 99.800 kbps K83.75).

**Method.** The continuous knob is λ scale; the ablation knobs are detail
strength and top-N. Each arm was encoded at two or three QPs that bracket
100.0 kbps, decoded with `dav2d --strict 1`, and scored with
`tools/quality.py` (CVVDP 0.5.7, `standard_fhd`, MPS),
`tools/extra_metrics.py` (Butteraugli, SSIMULACRA2) and `tools/kpi.py`. K is
linearly interpolated in rate to 100.0 kbps. The sweep is
λ ∈ {0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1, 2}, detail ∈ {0, 0.5, 1, 2}, and
top-N ∈ {2, 4}, on the drift recipe (threshold 1.5 dB, budget 3250 B).

**Achievable envelope (primary, a lower bound on K*):**

| Arm | Best K at 100.0 kbps | Rate | Δ vs wall 83.80 |
|---|---:|---:|---:|
| full knob sweep, best point | **84.91** | 100.07 | **+1.11** |
| detail-aware only (detail > 0) | **84.87** | 100.26 | **+1.07** |
| shipped point (detail 1.0, λ 0.2, N=2) | **84.85** | 99.98 | **+1.05** |

The envelope **exceeds the wall**, so tuning is allowed. The best measured
setting is the plain-SSE ablation of D (`detail=0`), i.e. the gain is the
joint R+λ refit, not the D shape; the best detail-aware setting is within
0.04 K. No allocation or decision bound was computed: the pipeline has no
frame-level reallocation knob (the drift threshold/budget is the measured
1.5/3250 recipe), and an exact-bit decision oracle (R05) was not built. The
envelope is therefore the reported oracle; the true `K*` is at least 84.91.

## D: local, source-driven distortion

Luma distortion is weighted by 8x8 source cells:

```text
detail  = mean(|dx| + |dy|) over the 8x8 cell, edge-clamped (0..510)
weightQ8 = 256 + 384 * detail / (detail + 24)     // 1.0 .. 4.0, integer
D        = Σ weightQ8 * (src - recon)^2 / 256
```

Flat cells stay at 1.0. Textured/edge cells are up-weighted; there is **no
smooth-cell down-weighting** (the measured negative direction). The map is
built once per frame from the source, is deterministic integer arithmetic,
and is handed to the TX candidate search as the per-pixel distortion map, so
TX size/type/partition is selected against the same D. Chroma keeps plain
SSE.

The D-shape ablation is in the results: at matched rate it is neutral. It is
kept because it is the pipeline's local detail term, not because it is the
source of the K gain.

## R: exact syntax + context-coded coefficients

The leaf rate is the sum of two measured models:

* **Syntax:** the R09 live-CDF probe (`docs/rate-hybrid-s.md`,
  `AV2ENC_RATE_HYBRID_S`). It seeds the frame's §8.2 CDF image, mirrors the
  writer's symbol sequence (`is_inter`, `skip_flag`, `comp_mode`,
  `single_ref`, `single_mode`, `drl_idx`, the eighth-pel raw-shell MVD and
  BAWP), advances in coded order and prices `-log2(p)`. R09 measured
  predicted 1.559 vs actual 1.577 bits/leaf, bias −0.02, median +0.07.
* **Coefficients:** the static §8.3.2 coefficient syntax from the §9.3 CDF
  slice (`txb_skip`, EOB prefix/extra, `coeff_base` / `coeff_base_eob`,
  `coeff_br`, the `read_quant` escape and sign), the same model the
  `--video-tx-context-rate` path uses. This is a context-coded model, not an
  iid Laplacian: R09 measured the iid Laplacian at +16.28 bits/leaf on
  residual leaves, worse than the tuned average; the static context model is
  the R10/TX-context model and is reused here, not re-derived.

The inter luma TX search runs with `use_context_rate=true`, so the TX
candidate's `rate_bits` already carries the coefficient syntax plus the TX
partition/type symbols. Intra luma and chroma plans are priced from their
scan-order levels with the same static model. A leaf the probe cannot model
(compound / TIP / AMVD / flex precision) falls back to the historical
constants and is counted; the measured nightly point has `fallbacks=0`.

## λ: joint refit

One weight prices both the leaf and the TX candidate:

```text
lambda = leaf_rd_lambda(qindex, options) * AV2ENC_PIPE_PERC_SCALE
J      = D + lambda * R
```

The historical leaf and TX lambdas are different because their D/R units were
different. Here D is the weighted reconstruction error and R is (near) true
bits, so the same λ is valid at both sites. The default scale is the measured
**0.2**; `AV2ENC_PIPE_PERC_SCALE` reproduces the sweep.

## Search

The cheap inter candidates keep their existing SSE+static-syntax ranking, but
the proxy winner plus the next cheapest distinct candidates (top-N, default
2) are re-scored through the real path: predict → BAWP/inter-intra →
`search_luma_tx` → chroma TX → reconstruct → rate. Inside every candidate,
**skip and residual are both priced with the same J**, and the TX
size/type/partition search uses the same weighted D and context R. The intra
candidate enters the same final comparison with its own D and context-coded
plan; there is no separate sequential intra/inter heuristic on this path.

The exact-syntax probe advances one shared CDF/grid state in coded order, so
the superblock wavefront is disabled while the gate is on. That is the main
encode-cost item.

## Frame level

`--video-drift-refresh` is unchanged. The measured tuned recipe is
`--video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250` and
is the recipe used for the drift bar. The pipeline does not add another
frame-level allocation change; it only changes which leaves/frames trigger
the existing refresh.

## Results

Host: Apple M4, `AV2ENC_CPU_FAST=OFF` dev build (`-O1`), `--cpu-threads 8`.
Clip: `testdata/nasa_480x240_30fps_60f.y4m`, 60 frames, 8-bit 4:2:0. Every
measured stream decoded 60/60 under `dav2d --strict 1`; the shipped point's
`--video-recon` payload equals the decoded payload byte for byte
(10,368,000 B).

### Baseline (this worktree)

| Point | QP | Bytes | kbps | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|---:|---:|
| plain nightly pin | 106 | 25,008 | 100.032 | 9.7334 | 3.4202 | 80.837 | 82.15 |
| tuned drift | 110 | 25,146 | 100.584 | 9.7605 | 2.6562 | 81.250 | 83.94 |
| tuned drift | 111 | 24,950 | 99.800 | 9.7513 | 2.7310 | 81.422 | 83.75 |
| tuned drift @100.0 | — | — | 100.000 | — | — | — | **83.80** |

### Tuned pipeline (λ 0.2, detail 1.0, N=2)

| Recipe | QP | Bytes | kbps | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|---:|---:|
| plain | 118 | 25,058 | 100.232 | 9.7623 | 2.3801 | 82.270 | 84.80 |
| plain | 119 | 24,641 | 98.564 | 9.7451 | 2.4024 | 81.851 | 84.58 |
| **plain @100.0** | — | — | 100.000 | — | — | — | **84.77** |
| drift | 118 | 25,530 | 102.120 | 9.7655 | 2.3570 | 82.285 | 84.88 |
| drift | **119** | **24,994** | **99.976** | **9.7584** | **2.3425** | **82.106** | **84.85** |
| drift | 120 | 24,677 | 98.708 | 9.7467 | 2.3156 | 81.502 | 84.76 |
| **drift @100.0** | — | — | 100.000 | — | — | — | **84.85** |

At the shipped point the decisions change materially: 145 inter-candidate
winner changes, 573 skip/residual flips and 2 intra flips over 2,695 leaves
(5,390 re-scored candidate evaluations); 1,232 leaves emit a residual. The
rate is the same 100 kbps but the bit distribution moves to the detail band:
Butteraugli 3.42 → 2.34, SSIMULACRA2 80.84 → 82.11, CVVDP 9.733 → 9.758.

### K vs λ scale at 100.0 kbps (drift, detail 1.0, N=2)

| λ scale | QP bracket (bytes) | K @ 100.0 | Δ vs 83.80 |
|---:|---|---:|---:|
| 0.15 | 120/121 (25,343 / 25,035) | 83.93 | +0.13 |
| **0.2** | **118/119/120 (25,530 / 24,994 / 24,677)** | **84.85** | **+1.05** |
| 0.25 | 118/119 (24,992 / 24,394) | 84.74 | +0.94 |
| 0.35 | 115/116/117 (25,755 / 24,971 / 24,663) | 84.42 | +0.62 |
| 0.5 | 114/115/116 (25,730 / 24,976 / 24,516) | 84.11 | +0.31 |
| 0.75 | 114/115 (25,107 / 24,534) | 83.96 | +0.16 |
| 1.0 | 112/113/114 (25,634 / 26,270 / 24,727) | 83.28 | −0.52 |
| 2.0 | 111/113 (25,271 / 24,688) | 82.84 | −0.96 |

The curve peaks at 0.2 and falls on both sides. The 0.15 point is in the
large adjacent-QP ripple (its QP120 neighbour is 84.71 at 101.37 kbps), so
the 0.2 plateau is the robust choice, not a single-QP fit.

### Knob ablations at the tuned point

| Knob | Point | kbps | JOD | BA | SS2 | K |
|---|---|---|---:|---:|---:|---:|---:|
| detail 0.0 (plain SSE D) | QP117 | 100.072 | 9.7683 | 2.3482 | 82.244 | 84.91 |
| detail 0.5 | QP118 | 100.256 | 9.7667 | 2.3604 | 82.271 | 84.87 |
| **detail 1.0 (shipped)** | **QP119** | **99.976** | **9.7584** | **2.3425** | **82.106** | **84.85** |
| detail 2.0 | QP120 | 101.584 | 9.7377 | 2.3486 | 81.431 | 84.62 |
| top-N 4 | QP119 | 99.672 | 9.7566 | 2.3456 | 82.052 | 84.82 |

The D shape is neutral at matched rate: the local detail term moves rate at a
fixed QP (detail 0 is 23,639 B vs 24,994 B at QP119) but the K at 100 kbps is
flat from detail 0 to 1. The gain is the R+λ refit. Top-N 4 does not improve
on N=2 and costs about twice the candidate evaluations, so N=2 ships.

### Plain recipe at the same scale

| λ scale | QP bracket (bytes) | K @ 100.0 |
|---:|---|---:|
| 0.2 | 118/119 (25,058 / 24,641) | **84.77** |
| 0.5 | 112/113 (25,201 / 24,879) | 83.97 |
| 1.0 | 110 (25,090) | 82.56 |

At scale 0.35 the plain QP114 point is 100.944 kbps / K 84.79 (a one-sided
bracket; the QP115 neighbour is 106.08 kbps / K 85.42), so the plain curve
has the same peak region as drift. Against the plain pin **82.15**, the tuned
pipeline is **+2.62 K** at 100.0 kbps. Against the drift wall **83.80**, the
tuned drift point is **+1.05 K**.

## Encode cost

60-frame 240p, dev `-O1` build, 8 threads, two runs each, same QP/recipe:

| Arm | wall | user CPU |
|---|---:|---:|
| baseline nightly QP106 | 5.77 / 6.71 s | 10.24 / 10.60 s |
| baseline drift QP119 | 7.12 / 7.61 s | 10.86 / 11.42 s |
| pipeline drift QP119 | 20.72 / 22.98 s | 17.10 / 17.74 s |

The pipeline is ~3.0× wall and ~1.6× user CPU at its own operating point.
The wavefront is disabled because the exact-syntax probe advances in coded
order; the top-2 re-scoring adds the second full TX/chroma/recon pass. This is
a research gate, not a production speed claim.

## 720p pin check

Gate off reproduces the 720p pin exactly: 36,572 B, MD5
`c4ba28138d8f6569d5e08721a66d755c` at QP217. The winner does **not** transfer
to the pin recipe. A 12-frame JOD-first direction check at the same key-offset
recipe:

| 12-frame arm | QP | Bytes | CVVDP | PSNR-Y |
|---|---:|---:|---:|---:|
| baseline pin recipe | 217 | 21,922 | 7.6676 | 33.27 dB |
| pipeline, same recipe | 222 | 21,254 | 7.4415 | 32.92 dB |

The pipeline is 0.226 JOD worse at 3% lower rate, so it loses at iso-rate even
before the 60-frame pin. QP228 is 18,411 B (further below the baseline rate)
and was not scored. No 720p JOD or K result is claimed against the 36,572 B /
146.288 kbps pin; a 720p pipeline would need its own λ refit.

## Verification

* Gate off: default QP189 25,009 B `a1d0dba1…`, quality QP190 25,141 B
  `3984677b…`, nightly QP110 23,690 B `1b99127a…`, 720p pin 36,572 B
  `c4ba2813…` — all byte-identical.
* `AV2ENC_PIPE_PERC=0` is byte-identical to unset.
* Gate on, shipped point: 24,994 B `650f2e3514978257c78f373f7bf99a50`,
  `dav2d --strict 1` 60/60, encoder `--video-recon` payload equals the
  decoded payload byte for byte (10,368,000 B).
* `src/pipeline/video/gate_video_pipe_perc.sh --bin build-dev/av2enc`: OK
  (pins, gate-off identity, tuned point, 60/60 decode, recon, counters,
  env-only). Registered as `gate_video_pipe_perc` in `tools/run_gates.sh`.
* `tools/test.sh --build build-dev`: 10/10.
* `tools/gate_video_nightly.sh`: OK.
* `src/pipeline/video/gate_video_qp_lambda.sh`: OK.
* Re-encoding the shipped point reproduces the MD5.

## Reproduce

```sh
REF=testdata/nasa_480x240_30fps_60f.y4m
BIN=build-dev/av2enc

# shipped point (defaults: scale 0.2, detail 1, top 2)
AV2ENC_PIPE_PERC=1 "$BIN" "$REF" -o /tmp/pp.ivf --video --frames 60 \
    --cpu-threads 8 --qp 119 --fast-decode \
    --video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250

# lambda sweep
for s in 0.15 0.2 0.25 0.35 0.5 0.75 1 2; do
  AV2ENC_PIPE_PERC=1 AV2ENC_PIPE_PERC_SCALE=$s "$BIN" "$REF" ...
done

# D ablation
AV2ENC_PIPE_PERC=1 AV2ENC_PIPE_PERC_DETAIL=0 "$BIN" "$REF" ...

# score
DAV2D=tools/bin/dav2d-macos python3 tools/quality.py --ref "$REF" \
    --ivf /tmp/pp.ivf --json /tmp/q.json
tools/bin/dav2d-macos --strict 1 -i /tmp/pp.ivf -o /tmp/pp.y4m
python3 tools/extra_metrics.py --ref "$REF" --test /tmp/pp.y4m --json /tmp/e.json
python3 tools/kpi.py --quality-json /tmp/q.json --extra-json /tmp/e.json
```

## Env knobs (no CLI flags)

| Variable | Default | Meaning |
|---|---:|---|
| `AV2ENC_PIPE_PERC` | off | master gate; any non-zero value enables |
| `AV2ENC_PIPE_PERC_SCALE` | 0.2 | λ multiplier (K-vs-λ sweep) |
| `AV2ENC_PIPE_PERC_TOP` | 2 | cheap inter candidates re-scored (1..8) |
| `AV2ENC_PIPE_PERC_DETAIL` | 1.0 | detail gain multiplier; 0 is the SSE ablation |

## Limits

* Measured on the CPU single-tile, low-delay flat P chain (the `--fast-decode`
  surface). Tiles/B frames/TIP/compound/AMVD/flex-precision leaves fall back
  to the historical constants (the probe report counts them; the measured
  point has zero fallbacks). The CUDA device path is not wired.
* The syntax probe is exact against the live CDF but the coefficient term is
  the static §9.3 model, not the live adaptive coefficient CDF. It is the
  context-coded model, not an iid Laplacian; the residual gap to the live
  coder remains.
* The drift refresh is the existing mechanism, but its trigger count is
  content/QP dependent: the shipped point refreshes 20 frames (3,448 B). The
  measured drift bar and the pipeline are compared at the same 100 kbps
  budget, not the same refresh schedule.
* The D shape is neutral at matched rate; do not attribute the gain to the
  weighted distortion. The positive move is the joint R+λ refit with the
  measured drift allocation.

# RD merge — key-intra TX plan × chroma coefficient levels

Status: **KEEP.** The two measured positive RD mechanisms compose. On the
merged tree the four-arm interleaved ladder (QP 116–122, base 109.0 → 96.0 kbps)
gives matched K @100 kbps **86.013 vs base 85.634 (+0.379)** for
`AV2ENC_RD_KEY_TX=1 AV2ENC_RD_UVLEVEL=1`, against the programme bar 85.784
(+0.15). The full-ladder line fit is also positive and additive:
**+0.131** for the composition versus **+0.063** (L3 alone) and **+0.061**
(L5 alone); the two fits sum to +0.124. Every stream decodes 60/60 and the
encoder recon equals the decoded payload.

Branch `exp/rd-merge` (from `315a7fbb`), worktree `/Users/arin/work/rd-merge`.
This report is the only new document; both mechanisms keep their own lane docs
(`docs/rd-lane3.md`, `docs/rd-lane5.md`).

| arm | gates | matched K@100 | Δ vs base | full-ladder fit K@100 | fit Δ |
|---|---|---:|---:|---:|---:|
| base | — | 85.634 | — | 85.450 | — |
| +L3 | `AV2ENC_RD_KEY_TX=1` | 85.942 | **+0.308** | 85.513 | +0.063 |
| +L5 | `AV2ENC_RD_UVLEVEL=1` | 85.698 | **+0.064** | 85.511 | +0.061 |
| **+L3+L5** | both | **86.013** | **+0.379** | **85.580** | **+0.131** |

## 1. Merge

Cherry-picked, in order:

| commit | lane | subject |
|---|---|---|
| `eb9e3443` | L3 | key-intra TX plan and coefficient level RD under one J |
| `82571fe0` | L3 | docs: cross-build determinism check |
| `bb9651bf` | L3 | docs: two 7-point ladders and TX-weight sweep |
| `4e10ec91` | L5 | chroma coefficient-level RD under the perceptual J |

All four applied with **no textual conflict**; git auto-merged the expected
files (`video_frame_intra.cpp`, `video_frame_decision.cpp`,
`video_frame_internal.h`, `video_tx_search.*`, `video_uv_search.h`). The
overlap is only at the call surface, and the merged tree was checked for the
failure modes a clean auto-merge can hide:

* **The gates are independent and default off.** L3 is read only in
  `encode_intra_luma_tx_partition`; L5 only in `encode_leaf` (intra chroma) and
  the inter chroma residual, and only when `AV2ENC_RD_UVFULL=1` and
  `level_variants > 0`. Neither reads the other's env var.
* **M4/L5 reuse path stays off with L5 unset.** `uv_search::search` is a
  wrapper over `search_with_levels` with a `level()` that returns false; the
  `have_plans` reuse in `encode_leaf` is guarded by
  `level_variants > 0 && uv_result.have_plans`.
* **L3's TX search defaults are the historical ones.** `intra_rdoq = false`
  and `model_rate_scale = 0.4` keep every inter `search_luma_tx` call exactly
  as before; only the exact-stage key leaves pass `intra_rdoq = true`,
  `model_rate_scale = 1.0`.
* **Per-arm stream identity.** At overlapping QPs each arm alone reproduces
  the source lane's own measured stream byte for byte, and the base reproduces
  the programme base:

| point | merged stream | source |
|---|---|---|
| base QP119 | 25,330 B | lane-3 ladder base 25,330 |
| base QP120 | 24,760 B / `49228567054c354f2ebd573f9845198f` | lane-3/lane-5 base |
| +L3 QP119 | 25,148 B / `1354bc5929f1c8552a112977c03c09a6` | lane-3 arm (QP119) |
| +L3 QP120 | 24,612 B / `42b4811fda3f0921cccdb22e64c0ec1a` | lane-3 arm |
| +L5 QP118 | 25,732 B | lane-5 candidate 25,732 |
| +L5 QP119 | 25,205 B | lane-5 candidate 25,205 |
| +L5 QP120 | 25,005 B | lane-5 candidate 25,005 |

No lane-2 or lane-4 code is present. No CLI, CMake, `main.cpp`, options/parse,
gate or pin file was edited; the only addition besides this doc is the ladder
driver `tools/lane-merge/ladder.sh`.

## 2. Build and gate-off pins

`cmake --preset dev-cache` and `cmake --preset release` (warnings + `-Werror`)
both build clean; `ctest` on the release build is 12/12. There were no warnings
to fix after the cherry-picks.

Gate-off pins, plain `--nightly`, release build:

| QP | bytes | MD5 |
|---|---:|---|
| 106 | 25,008 | `0639b43630abb454f7be5ac494d2efb4` |
| 110 | 23,690 | `1b99127a4cd1d386fa5600fb63c57a83` |

Both match the recorded pins. Re-checked after the final rebuild that produced
the reported streams.

## 3. Measurement recipe

```sh
REF=testdata/nasa_480x240_30fps_60f.y4m
BASE_ENV="AV2ENC_EFF_PERC=1 AV2ENC_RD_UVFULL=1 \
          AV2ENC_TX_DETAIL_SEARCH_TAIL=1 AV2ENC_TX_DETAIL_MIN=0.50"
COMMON="--video --frames 60 --cpu-threads 8 --nightly \
        --video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250"

env $BASE_ENV build-release/av2enc "$REF" -o out.ivf --qp Q $COMMON
# arms: add AV2ENC_RD_KEY_TX=1 and/or AV2ENC_RD_UVLEVEL=1
```

Seven interleaved QPs (116–122), arm order rotated at every QP, driven by
`tools/lane-merge/ladder.sh` under bash. Arm knobs are the lanes' measured
defaults, all env-only:

* L3: `AV2ENC_RD_KEY_TX_TOP=4`, `AV2ENC_RD_KEY_TX_SCALE=0.25`,
  `AV2ENC_RD_KEY_LEVEL=1`, `AV2ENC_RD_KEY_LEVEL_SCALE=auto` (0.25× the RDOQ
  scale), `AV2ENC_RD_KEY_TX_D=1`.
* L5: `AV2ENC_RD_UVLAMBDA=1.0`, `AV2ENC_RD_UVLEVEL_LAMBDA=0.125`,
  `AV2ENC_RD_UVLEVEL_INTER=on`, `AV2ENC_RD_UVLEVEL_TRIES=3`,
  `AV2ENC_RD_UVLEVEL_TOPN=3`.

Matched-rate K is
`tools/oracle.py --target-kbps 100 --no-segments` over each arm's own ladder;
it interpolates linearly in rate between the two adjacent points that straddle
100 kbps. The full-ladder fit is an ordinary least-squares line of KPI against
kbps over all seven points, evaluated at 100 kbps; it is reported because the
100 kbps bracket is only one pair and the L3 key RD is known to be
ripple-prone.

## 4. Ladder

Base rate falls 109.02 → 95.99 kbps; the +L3+L5 arm 107.54 → 95.90 kbps.
`keyB` is the first IVF packet (key picture), `Y-PSNR` its luma PSNR.

| QP | arm | bytes | kbps | KPI | BA | SS2 | keyB | Y-PSNR |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 116 | base | 27,254 | 109.02 | 85.880 | 2.139 | 83.18 | 16,799 | 46.673 |
| 116 | +L3 | 27,104 | 108.42 | 86.050 | 2.092 | 83.42 | 16,594 | 46.684 |
| 116 | +L5 | 26,942 | 107.77 | 85.927 | 2.148 | 83.51 | 16,623 | 46.673 |
| 116 | +L3+L5 | 26,884 | 107.54 | 86.184 | 2.063 | 83.56 | 16,411 | 46.684 |
| 117 | base | 26,513 | 106.05 | 85.859 | 2.138 | 83.16 | 16,528 | 46.474 |
| 117 | +L3 | 26,518 | 106.07 | 85.746 | 2.176 | 83.31 | 16,322 | 46.457 |
| 117 | +L5 | 26,294 | 105.18 | 85.846 | 2.174 | 83.52 | 16,389 | 46.474 |
| 117 | +L3+L5 | 26,616 | 106.46 | 85.917 | 2.108 | 83.21 | 16,182 | 46.457 |
| 118 | base | 26,165 | 104.66 | 85.933 | 2.103 | 83.10 | 16,236 | 46.301 |
| 118 | +L3 | 25,805 | 103.22 | 85.573 | 2.228 | 83.16 | 16,115 | 46.253 |
| 118 | +L5 | 25,732 | 102.93 | 86.012 | 2.107 | 83.39 | 16,088 | 46.301 |
| 118 | +L3+L5 | 25,578 | 102.31 | 85.802 | 2.170 | 83.24 | 15,970 | 46.253 |
| 119 | base | 25,330 | 101.32 | 85.837 | 2.126 | 83.04 | 15,984 | 46.048 |
| 119 | +L3 | 25,148 | 100.59 | 86.077 | 2.038 | 82.89 | 15,805 | 46.019 |
| 119 | +L5 | 25,205 | 100.82 | 85.789 | 2.144 | 82.87 | 15,849 | 46.048 |
| 119 | +L3+L5 | 24,984 | 99.94 | 86.019 | 2.065 | 82.86 | 15,687 | 46.019 |
| 120 | base | 24,760 | 99.04 | 85.487 | 2.183 | 82.45 | 15,743 | 45.862 |
| 120 | +L3 | 24,612 | 98.45 | 85.588 | 2.162 | 82.51 | 15,621 | 45.826 |
| 120 | +L5 | 25,005 | 100.02 | 85.704 | 2.135 | 82.60 | 15,606 | 45.862 |
| 120 | +L3+L5 | 24,616 | 98.46 | 85.674 | 2.145 | 82.61 | 15,490 | 45.826 |
| 121 | base | 24,471 | 97.88 | 85.115 | 2.313 | 82.25 | 15,476 | 45.643 |
| 121 | +L3 | 24,268 | 97.07 | 85.121 | 2.320 | 82.34 | 15,308 | 45.651 |
| 121 | +L5 | 24,537 | 98.15 | 85.103 | 2.307 | 82.22 | 15,373 | 45.643 |
| 121 | +L3+L5 | 23,953 | 95.81 | 84.896 | 2.394 | 82.28 | 15,202 | 45.651 |
| 122 | base | 23,997 | 95.99 | 85.024 | 2.311 | 82.06 | 15,265 | 45.460 |
| 122 | +L3 | 24,041 | 96.16 | 85.074 | 2.310 | 82.07 | 15,291 | 45.505 |
| 122 | +L5 | 23,641 | 94.56 | 84.956 | 2.348 | 82.04 | 15,147 | 45.460 |
| 122 | +L3+L5 | 23,975 | 95.90 | 85.112 | 2.295 | 82.06 | 15,196 | 45.505 |

Matched K and fit (the base row reproduces the programme base 85.634 exactly;
the single-arm rows reproduce `docs/rd-lane3.md` 85.942 and
`docs/rd-lane5.md` 85.698):

| arm | 100 kbps bracket | K@100 | Δ | fit K@100 | fit Δ | slope | fit R² |
|---|---|---|---:|---:|---:|---:|---:|
| base | QP119/120 | 85.634 | — | 85.450 | — | +0.0706 | 0.755 |
| +L3 | QP119/120 | 85.942 | +0.308 | 85.513 | +0.063 | +0.0637 | 0.549 |
| +L5 | QP120/121 | 85.698 | +0.064 | 85.511 | +0.061 | +0.0807 | 0.729 |
| +L3+L5 | QP118/119 | **86.013** | **+0.379** | **85.580** | **+0.131** | +0.0842 | 0.696 |

The composition's bracket is QP118/119 because its QP119 point is already
below 100 kbps (99.94); the L3 and L5 brackets are unchanged from their lanes.
The fit delta is the conservative estimate and it is almost exactly additive
(+0.063 + 0.061 = +0.124 vs +0.131): the chroma mechanism keeps contributing
its ~+0.06 on top of the key-TX mechanism, and the composition has the
steepest KPI/rate slope of the four arms.

### Per-QP KPI deltas vs base

| QP | +L3 | +L5 | +L3+L5 |
|---:|---:|---:|---:|
| 116 | +0.170 | +0.047 | +0.304 |
| 117 | −0.113 | −0.013 | +0.058 |
| 118 | −0.360 | +0.079 | −0.131 |
| 119 | +0.241 | −0.048 | +0.182 |
| 120 | +0.102 | +0.217 | +0.187 |
| 121 | +0.005 | −0.012 | −0.219 |
| 122 | +0.050 | −0.067 | +0.089 |

The composition is positive at 5 of 7 QPs. The two negatives are the same
Butteraugli ripple points the single lanes already show (QP118: BA 2.170 vs
2.103; QP121: BA 2.394 vs 2.313; CVVDP and SS2 are flat-to-better at both).
The QP121 drop also comes with the arm spending 2.1 kbps fewer than base.

### Equal-rate pairs

Closest ladder point per base point, and the arm KPI interpolated at the base
rate (the rate-adjusted view):

| base QP / kbps | closest +L3 | Δkbps / dKPI | closest +L5 | Δkbps / dKPI | closest +L3+L5 | Δkbps / dKPI | interp dKPI at base rate |
|---|---|---|---:|---:|---:|---:|---:|
| 116 / 109.02 | q116 108.42 | −0.60 / +0.170 | q116 107.77 | −1.25 / +0.047 | q116 107.54 | −1.48 / +0.304 | n/a (above range) |
| 117 / 106.05 | q117 106.07 | +0.02 / −0.113 | q117 105.18 | −0.88 / −0.013 | q117 106.46 | +0.41 / +0.058 | +0.046 |
| 118 / 104.66 | q117 106.07 | +1.41 / −0.187 | q117 105.18 | +0.52 / −0.086 | q117 106.46 | +1.80 / −0.016 | −0.066 |
| 119 / 101.32 | q119 100.59 | −0.73 / +0.241 | q119 100.82 | −0.50 / −0.048 | q118 102.31 | +0.99 / −0.035 | +0.056 |
| 120 / 99.04 | q120 98.45 | −0.59 / +0.102 | q121 98.15 | −0.89 / −0.384 | q120 98.46 | −0.58 / +0.187 | +0.322 |
| 121 / 97.88 | q120 98.45 | +0.56 / +0.473 | q121 98.15 | +0.26 / −0.012 | q120 98.46 | +0.58 / +0.558 | +0.431 |
| 122 / 95.99 | q122 96.16 | +0.18 / +0.050 | q122 94.56 | −1.42 / −0.067 | q122 95.90 | −0.09 / +0.089 | +0.108 |

At every base rate from 101.3 kbps down, the composition's rate-adjusted KPI
is positive (+0.06 … +0.43). The only slightly negative rate-adjusted point is
104.7 kbps (−0.07), again the QP118 BA ripple.

## 5. Key/chroma composition

Key picture bytes (first IVF packet), every QP:

| QP | base | +L3 | +L5 | +L3+L5 | Δ L3 | Δ L5 | Δ both |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 116 | 16,799 | 16,594 | 16,623 | 16,411 | −205 | −176 | −388 |
| 117 | 16,528 | 16,322 | 16,389 | 16,182 | −206 | −139 | −346 |
| 118 | 16,236 | 16,115 | 16,088 | 15,970 | −121 | −148 | −266 |
| 119 | 15,984 | 15,805 | 15,849 | 15,687 | −179 | −135 | −297 |
| 120 | 15,743 | 15,621 | 15,606 | 15,490 | −122 | −137 | −253 |
| 121 | 15,476 | 15,308 | 15,373 | 15,202 | −168 | −103 | −274 |
| 122 | 15,265 | 15,291 | 15,147 | 15,196 | +26 | −118 | −69 |

The key deltas add (−179 −135 = −314 expected, −297 actual at QP119; 95 % of
the sum). Key luma PSNR moves only with L3 (−0.048 … +0.045 dB); L5 is
luma-transparent (0.000 dB at every QP) because it only changes chroma levels.

### Key picture by family (QP119, `AV2ENC_BIT_CENSUS`)

Information bits, first picture:

| family | base | +L3 | +L5 | +L3+L5 | Δ both |
|---|---:|---:|---:|---:|---:|
| payload (B) | 15,944 | 15,765 | 15,809 | 15,647 | −297 |
| partition | 247.4 | 247.4 | 247.4 | 247.4 | 0 |
| intra_mode | 3,248.2 | 3,622.6 | 3,242.4 | 3,627.0 | +378.8 |
| residual | 81,515.9 | 79,724.4 | 80,770.4 | 79,013.0 | −2,502.9 |
| ccso_blk | 28.0 | 32.0 | 29.0 | 32.0 | +4 |
| txb_skip | 577.1 | 598.0 | 601.9 | 613.6 | +36.5 |
| eob | 5,098.7 | 5,561.1 | 4,959.1 | 5,418.8 | +320.1 |
| coeff_base | 63,912.4 | 62,401.2 | 63,169.1 | 61,716.1 | −2,196.3 |
| coeff_br | 13,854.8 | 13,339.0 | 13,868.4 | 13,347.3 | −507.5 |
| sign | 537.4 | 640.7 | 537.4 | 640.7 | +103.3 |
| tx_partition | 453.5 | 350.5 | 453.5 | 350.5 | −103.0 |
| coeff_mag (escape) | 39,031.0 | 38,827.0 | 38,819.0 | 38,683.0 | −348.0 |

`residual` is additive to 99 %: L3 −1,791.5, L5 −745.5, sum −2,537.0 vs actual
−2,502.9. The composition keeps L3's trade (fewer `coeff_base`/`coeff_br`,
more `eob` and `sign`, SPLIT-heavy TX plan) and adds L5's chroma-level saving.

### Key luma vs chroma information bits

Measured with a local plane-tagged build of the existing census (diagnostic
only, not committed; the key packets of that build are byte-identical to the
ladder streams). Write-time information bits over `code_tx_view`, i.e. the
coefficients plus that plane's TX symbols, key picture QP119:

| plane | base | +L3 | +L5 | +L3+L5 | Δ both |
|---|---:|---:|---:|---:|---:|
| Y | 75,973.8 | 74,712.1 | 75,976.1 | 74,709.7 | −1,264.1 |
| U | 3,769.7 | 3,732.6 | 3,337.7 | 3,333.7 | −436.0 |
| V | 5,259.1 | 5,140.8 | 4,937.6 | 4,835.2 | −423.9 |
| **U+V** | **9,028.9** | **8,873.4** | **8,275.3** | **8,168.8** | **−860.0 (−9.53 %)** |

Chroma deltas: L3 −155.4 bits (−1.72 %) from the changed luma recon feeding
chroma prediction; L5 −753.5 bits (−8.35 %, the lane's own measured ~12 % cut
in *static* coefficient bits); both −860.0 bits with essentially no overlap
(sum −909, 95 % realised). L3's gain is essentially luma-only (−1,264 bits; the
−155 chroma bits are a prediction-follow-on) and L5's is chroma-only (−754
bits): the two mechanisms act on different planes.

## 6. Per-frame PSNR and skip share

QP119 per-frame PSNR-Y is stable; no P-tail collapse. Mean over 60 frames:
base 44.934, L3 44.893 (−0.041), L5 44.933 (−0.017), L3+L5 44.923 (−0.011).
Selected frames (base → +L3+L5):

| frame | base | +L3+L5 | Δ |
|---:|---:|---:|---:|
| 0 (key) | 46.048 | 46.019 | −0.029 |
| 1 | 46.049 | 46.021 | −0.028 |
| 10 | 45.081 | 45.046 | −0.035 |
| 20 | 45.138 | 45.039 | −0.099 |
| 30 | 44.536 | 44.550 | +0.014 |
| 33 | 43.872 | 43.865 | −0.007 |
| 36 | 43.963 | 44.002 | +0.039 |
| 40 | 44.376 | 44.360 | −0.016 |
| 48 | 44.813 | 44.991 | +0.178 |
| 59 | 46.718 | 46.772 | +0.054 |

Per-frame range vs base at QP119: L3 −0.254 … +0.108 dB, L5 −0.452 … +0.143,
both −0.213 … +0.178; only 3/60 frames move by more than 0.15 dB.

Skip share (share of inter leaves with `skip = 1`, `AV2ENC_DECISION_DUMP`,
QP119, whole stream): base **48.74 %**, +L3 **49.26 %**, +L5 **49.15 %**,
+L3+L5 **51.19 %** (+2.45 pp). The composition skips more blocks, which is
consistent with the smaller residual.

## 7. Conformance

* 28/28 ladder streams: `tools/validate_dav2d.sh` OK, `dav2d --strict 1`
  decodes 60/60, IVF header 60 frames.
* 28/28: encoder `--video-recon` payload equals the strict-decoded payload
  byte for byte.
* Gate-off pins match (§2); final release build reproduces the +L3+L5 QP119
  stream exactly (24,984 B / `f778b2a5eb3e7227e6913494387a1434`).
* `ctest` 12/12; dev-cache and release (warnings, `-Werror`) clean.
* No gate, pin, CLI, options/parse, `main.cpp` or CMake change.

## 8. Verdict

**KEEP.** The composition is **86.013** (base 85.634), clearing the 85.784 bar
by 0.229 (the same as the +0.15-over-base form), with a positive full-ladder
fit (+0.131) and a positive rate-adjusted KPI at every base rate at or below
101.3 kbps. It is also the best of the four arms and the only one whose
full-ladder fit is roughly the sum of the two single-mechanism fits, so the
composition is not a new ripple: L5 keeps its ~+0.06 fit gain on top of L3, and
the key RD changes are plane-disjoint (L3 luma, L5 chroma) and additive to
95–99 %.

Caveats for the integrator:

* The +0.379 bracket is the favourable QP118/119 pair; the conservative
  estimate is the fit delta **+0.131**. If the KEEP rule is read as requiring
  the *fit* delta ≥ +0.15, this is a high LEAD rather than a KEEP. The matched
  K clears the stated 85.784 bar either way.
* Two same-QP points are negative (QP118 −0.131, QP121 −0.219), both driven by
  Butteraugli (2.170 vs 2.103 and 2.394 vs 2.313); CVVDP and SS2 are
  flat-to-better there. This is the same BA ripple the single lanes report.
* The l3l5 QP121→122 rate step is slightly non-monotone (+22 B), the
  documented L3 key flat spot at QP122 (key +26 B there).
* Both gates stay default off; nothing changes for any existing gate-off pin
  or M4 UVFULL stream.

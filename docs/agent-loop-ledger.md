# Agent loop ledger

| # | target | patch | probe | K @100 | verdict |
|---:|---|---|---|---:|---|
| 1 | ME/allocation: zero-MV skip under an illumination ramp | `AV2ENC_SKIP_DRIFT_ILLUM` frame-level source-statistic latch (`AV2ENC_SKIP_DRIFT_ILLUM_T`, default 0.35) | `fade` **87.226** (84.128 fade gate, 82.701 nightly) | **84.062** (82.149) | **kept** |
| 2 | ME/allocation: reference-deficit condition for the zero-MV skip | `AV2ENC_SKIP_DRIFT_REFDEF` measured primary-reference coding loss vs the key anchor (`AV2ENC_SKIP_DRIFT_REFDEF_T`, default 1.5 dB) | `static_repeat` `595974f6`, `near_static` `da71dbca`, `fade` and `me_half` byte-identical | **84.508** (84.062 stack; 84.496 standalone, 82.149 gate off) | **kept** |

Baseline (gate off, this worktree `exp/agent-loop` @ `89a0b4f0`): nightly
QP104/106/108 = 26,158 / 25,008 / 24,291 B, K 80.734 / 82.155 / 81.604,
matched **82.149**. Reproduces `docs/tool-improvement-queue-2.md` exactly.

## 1. Attempt 1 — frame-level illumination latch for the zero-MV skip (kept)

**Why.** Queue 2 attempt 1 measured the full zero-MV charge
(`AV2ENC_SKIP_DRIFT_ZEROMV=1`) at natural 84.403 (+2.25) but rejected it:
`static_repeat` loses 1.01 K because the surcharge codes residual on exact
static content. The note left open was "the zero-MV surcharge needs a
reference-deficit condition". Attempt 7's per-leaf DC-bias test recovered part
of it (+1.32 natural, +1.43 `fade`) but fires only on a leaf whose *mean* error
is >= 1 LSB. A fade leaf whose local mean sits near the fade pivot has no DC
bias yet its texture is rescaled every frame, so it stays skip-only. Measured
attribution at QP112 (`--decoded`/`--segment-from`): the fade gate improves
f0-32 +0.28 / f33-43 +0.30 / f44-59 +1.36 K; the full zero-MV charge improves
+0.53 / +0.45 / +2.24. The extra is mostly still in the fade tail, so the
missing condition is frame-level illumination evidence, not a leaf DC bias.

**Patch.** `AV2ENC_SKIP_DRIFT_ILLUM` (used with `AV2ENC_SKIP_DRIFT=1`, and
independently of `AV2ENC_SKIP_DRIFT_FADE`). The frame loop measures the source
luma mean and standard deviation and forms `delta = |d mean| + |d std|` against
the previous source frame. Every zero-MV skip in a frame with
`delta >= AV2ENC_SKIP_DRIFT_ILLUM_T` (default 0.35 LSB) pays the same
future-error surcharge a subpel skip already pays. Measured deltas: natural head
0.03-0.05, natural fade tail 1.8-1.9, `fade` probe ~0.5, `static_repeat` 0.0,
`near_static` 0.01 (sensor noise averages out of the frame statistics). No
syntax, no preset/CLI default, no pin change; gate off is byte-identical.

**Natural 240p** (`AV2ENC_SKIP_DRIFT=1 AV2ENC_SKIP_DRIFT_FADE=1
AV2ENC_SKIP_DRIFT_ILLUM=1`), 60 frames, 4:2:0, AVM oracle:

| QP | bytes | kbps | K |
|---:|---:|---:|---:|
| 104 | 30,998 | 123.992 | 85.661 |
| 106 | 29,625 | 118.500 | 85.154 |
| 108 | 28,152 | 112.608 | 84.975 |
| 112 | 26,270 | 105.080 | 84.317 |
| 114 | 25,438 | 101.752 | 83.937 |
| 116 | 24,545 | 98.180 | 84.192 |
| 118 | 23,627 | 94.508 | 83.783 |

Matched **84.062 (+1.913)** by interpolation over QP116/114. Every bracket
point is above the baseline at its own iso-rate (e.g. QP114 +2.5, QP116 +2.3).
The gate alone (without the fade gate) brackets at **83.969 (+1.820)**;
the fade gate alone in this build is 83.473. So the two overlap heavily and
both give ~+1.8-1.9 over baseline.

**Probe.** `fade` (the target probe; 82.70 nightly, 84.128 fade-gate):

| arm | QP | bytes | kbps | K |
|---|---:|---:|---:|---:|
| fade gate | 119 | 25,436 | 101.744 | 84.219 |
| | 121 | 24,552 | 98.208 | 84.035 |
| | 123 | 23,934 | 95.736 | 83.257 |
| + illum | 121 | 27,616 | 110.464 | 87.871 |
| | 123 | 26,389 | 105.556 | 87.436 |
| | 125 | 25,526 | 102.104 | 87.410 |
| | 127 | 24,606 | 98.424 | 87.089 |

Matched **87.226 vs 84.128 (+3.10)**, closing most of the 6.55 K nightly gap to
SVT (89.25) on this probe. The mechanism is the P chain, not a metric trick:
BA 2.14 -> 1.51-1.56 and SS2 75.2 -> 79.2-80.8 at matched rate, PSNR-Y -0.5 dB
(the peak-perceptual trade the K composite rewards).

**Static/skip gates.** `static_repeat` QP132 MD5 `595974f6...` and
`near_static` QP139 MD5 `da71dbca...` are byte-identical to baseline (and to
queue 2's records). `me_half` QP150 is byte-identical with and without the new
gate (the translating texture has constant frame statistics), so the ME probe
keeps the skip-drift result unchanged.

**Threshold.** `AV2ENC_SKIP_DRIFT_ILLUM_T=0.5` and `=1.0` give identical
streams (25,615 / 24,665 B at QP114/116, matched 84.110); the 0.35 default is
0.05 below that, inside the ladder ripple. 1.0 is not usable as the default:
the `fade` probe moves ~0.5 per frame, so it would not fire there at all.
0.35 sits two orders of magnitude above the static/near-static movement and
below the fade probe.

**Verification.** Gate off reproduces all five 240p/720p pins
(25,009 `a1d0dba1`, 25,141 `3984677b`, 23,690 `1b99127a`, 24,113 `7f03eb6e`,
720p 36,572 `c4ba2813`) plus the block-dump pins and `--threads 1 == 8`;
`tools/test.sh --build build-release` 10/10; no build warnings.

**Blockers / next.** The arm spends ~18% more bytes at a fixed QP, so any
future composition must be re-bracketed (QP106 is already 118.5 kbps). The
latch is a frame-level gate: a hard scene cut also moves the source statistics,
and a fade that holds mean and contrast constant (cross-fade of two textures)
is invisible to it. Only the zero-MV skip is addressed; the remaining P-tail
gap is still the residual allocation the luma audit describes.

## 2. Attempt 2 — reference-deficit latch for the zero-MV skip (kept)

**Why.** Attempt 1's illumination latch covers the fade probe and the natural
tail, but the 32-frame static head still copies chain-added error: its source
statistics move only 0.03-0.05 LSB per frame (below the 0.35 latch), while the
reference has already lost 1.1-2.2 dB against the key by f6-f31
(`docs/luma-audit.md` §4.5). Queue 2's rejected full zero-MV charge
(`AV2ENC_SKIP_DRIFT_ZEROMV=1`, +2.25 natural but -1.01 on `static_repeat`)
left the missing condition open: **a measured reference deficit**. That
condition can be written directly instead of inferred from source statistics.
For each inter frame, compare the primary reference's stored reconstruction
against the source picture it was coded from (the same currency
`--video-drift-refresh` uses) and subtract the key's own measured coding PSNR
(the anchor). The deficit is zero on an exact repeat, is the source-noise floor
on `near_static`, and grows only when the P chain itself has added error beyond
the key's deliberate QP choice.

Measured deficit (dB below the key anchor; selected frames, from
`--video-recon` and the stack's decisions):

| content | anchor | f1 | f5 | f6 | f10 | f20 | f25 | f33 | f44 | f59 | frames >= 1.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| natural 240p | 46.65 | 0.00 | 1.63 | 1.63 | 1.12 | 1.18 | 2.21 | 2.97 | 3.34 | 1.91 | 40/59 |
| `static_repeat` | 39.60 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0/59 |
| `near_static` | 37.44 | 1.31 | 1.28 | 1.28 | 1.28 | 1.29 | 1.30 | 1.31 | 1.28 | 1.30 | 0/59 |
| `me_half` | 39.06 | 0.20 | 0.31 | 0.36 | 0.57 | 1.00 | 1.15 | 1.29 | 1.47 | 1.73 | 15/59 |
| `fade` | 42.34 | 0.18 | 0.09 | 0.11 | 0.00 | -0.39 | -0.55 | -0.97 | -1.57 | -2.56 | 0/59 |

**Patch.** `AV2ENC_SKIP_DRIFT_REFDEF` (used with `AV2ENC_SKIP_DRIFT=1`).
`video_frame.cpp` captures the key anchor once, while slot 0 still holds the
key, then per inter frame measures the primary reference's PSNR against its
own source and passes `deficit = anchor - ref_psnr` into the frame encoder.
`video_frame_decision.cpp` arms the existing future-error surcharge for every
zero-MV skip when `deficit >= AV2ENC_SKIP_DRIFT_REFDEF_T` (default 1.5 dB).
No syntax, no preset/CLI default, no pin, no decoder path. Gate off is
byte-identical (the pins below).

**Natural 240p** (stack = `AV2ENC_SKIP_DRIFT=1 AV2ENC_SKIP_DRIFT_FADE=1
AV2ENC_SKIP_DRIFT_ILLUM=1`, plus `AV2ENC_SKIP_DRIFT_REFDEF=1`), 60 frames,
4:2:0, AVM oracle:

| QP | stack bytes | stack kbps | stack K | +refdef bytes | +refdef kbps | +refdef K |
|---:|---:|---:|---:|---:|---:|---:|
| 112 | 26,270 | 105.080 | 84.317 | 27,239 | 108.956 | 84.600 |
| 114 | 25,438 | 101.752 | 83.937 | 26,132 | 104.528 | 84.556 |
| 116 | 24,545 | 98.180 | 84.192 | 25,066 | 100.264 | **84.535** |
| 118 | 23,627 | 94.508 | 83.783 | 24,012 | 96.048 | 84.106 |
| 120 | 23,126 | 92.504 | 83.458 | 23,341 | 93.364 | 83.849 |
| 122 | — | — | — | 22,640 | 90.560 | 83.487 |

Matched **84.508 vs 84.062 (+0.446)** by interpolation over QP118/116. The
iso-rate deltas at the arm's bracket points are **+0.15** (96.048), **+0.30**
(104.528) and **+0.49** (100.264) against the stack's own curve, so the sign is
consistent. Standalone (`SKIP_DRIFT=1` + `REFDEF=1`, no fade/illum gate) the
same ladder is QP104/106/108 = 32,608 / 30,948 / 29,895 B at 130.4 / 123.8 /
119.6 kbps, K 85.546 / 85.374 / 85.222, matched **84.496 (+2.347 vs 82.149)**.
At QP112/114/116 the stack+refdef streams are byte-identical to refdef alone,
so on natural content the measured reference deficit subsumes the illumination
latch; the illumination latch remains necessary on `fade`, where the deficit is
negative and refdef is inert.

**Static / probe gates.** With the stack, adding the gate leaves
`static_repeat` QP132 (24,993 B, MD5 `595974f6...`), `near_static` QP139
(22,578 B, `da71dbca...`), `fade` QP121/123 (24,247 / 23,507 B, identical) and
`me_half` QP150/152 (25,381 / 24,939 B) **byte-identical**.
`me_half` has essentially no zero-MV skips on its translating texture, so the
probe keeps attempt 1's 76.105 matched K. Every scored stream decoded with
`dav2d --strict 1` (60/60 pictures).

**Threshold.** At the QP116 point (`T` = `AV2ENC_SKIP_DRIFT_REFDEF_T`):
`1.3` 100.572 kbps / 84.309; `1.4` 100.772 / 84.486; **`1.5` 100.264 /
84.535**; `1.6` 101.216 / 84.363; `1.75` 100.080 / 84.356; `2.0` 99.224 /
84.376. The local differences are inside the QP ripple; 1.5 is best-or-tied
and is the only value with a principled margin on both sides: above the
near_static noise floor (1.27-1.32 dB) and below the natural head's first real
drift (1.63 dB at f5). 1.3 would fire on `near_static` and is both worse here
and a static-gate violation.

**Verification.** Gate off reproduces all five pins (25,009 `a1d0dba1`, 25,141
`3984677b`, 23,690 `1b99127a`, 24,113 `7f03eb6e`, 720p 36,572 `c4ba2813`);
clean `Release` build with 0 warnings; `tools/test.sh --build build-release`
10/10; quick product gates 9 passed / 1 skipped (0 failed);
`tools/gate_video_nightly.sh` OK.

**Blockers / next.** The latch is frame-level and one frame causal: it charges
every zero-MV skip in a frame whose primary reference is deficient, including
leaves whose own prediction is still accurate. A per-superblock or per-leaf
reference-deficit metric (low-band MSE against the source) is the natural next
refinement and should recover part of the fixed-QP cost. The standalone arm
spends ~24% more bytes at a fixed QP (QP106 30,948 vs 25,008 B), so any
composition must be re-bracketed; at matched rate the gate is the QP move, not
a free saving. The gate chooses *where* residual is coded, not how efficiently:
the remaining P-tail gap is still the residual allocation and coefficient
decision, and the coefficient lanes are closed.

# Frame interpolation filter: implemented, measured, no envelope gain

Status: **tool implemented (default off), measured negative on the 240p/100 kbps
target.** 2026-11-19. This is the first implementation of the §5.18.5.1
frame-level `interpolation_filter` choice in the video encoder; the historical
stream hard-coded `EIGHTTAP` and never exercised the other three kernels.

The point of the experiment was the measured 1-4 px P-frame band deficit: the
subpel kernel is the only tool in that band whose implementation was missing.
It is now measured and closed for the frame-global form.

## Implementation

- `src/pipeline/video/video_interp_filter.h`: env gate `AV2ENC_INTERP_FILTER`
  (0 EIGHTTAP, 1 EIGHTTAP_SMOOTH, 2 EIGHTTAP_SHARP, 3 BILINEAR; unset = 0).
- `video_headers.cpp` / `post/tiled_frame.cpp`: the two existing header bits
  (`is_filter_switchable = 0`, `interpolation_filter`) now carry the gate.
- `video_frame_me.cpp`: the two block-prediction choke points (`predict_luma`,
  `predict_all`) and the compound carrier use the frame filter.
- `video_frame_decision.cpp`: the gated compound predictor uses the frame filter.

No new syntax, no new tables, no preset change. The CUDA backend keeps
EIGHTTAP; this gate is a CPU experiment.

## Gate

| check | result |
|---|---|
| default (`unset`) vs `AV2ENC_INTERP_FILTER=0` | byte-identical |
| QP106 nightly pin | 25,008 B / `0639b43630abb454f7be5ac494d2efb4` (matches) |
| eff-perc QP119 pin | 25,230 B / K **85.099** (matches) |
| filter 0/1/2/3, 8 frames, `--video` + `--video-recon` | dav2d `--strict 1` decodes 8/8; decoded payload == encoder reconstruction byte for byte (4/4) |
| MC component self-test | 3,532 checks, 0 failures |

So the kernels are not the problem: every filter is decodable and the encoder
mirror is exact.

## Measurement

Nightly, 240p, 60 frames, AVM oracle, 3-point ladders (105/106/107; bilinear
107/108), matched to 100.000 kbps.

| filter | matched K @100 | delta vs EIGHTTAP |
|---|---:|---:|
| EIGHTTAP (default) | **82.048** | — |
| EIGHTTAP_SMOOTH | 77.738 | **-4.310** |
| EIGHTTAP_SHARP | 81.349 | **-0.699** |
| BILINEAR | 79.293 | **-2.755** |

eff-perc recipe, 2-point 119/120:

| filter | matched K | note |
|---|---:|---|
| EIGHTTAP | **85.099** | bracket |
| EIGHTTAP_SHARP | 84.804 | 99.748 kbps (nearest, not a bracket) |

The envelope over the new knob is the default value: **82.048 nightly / 85.099
eff-perc**, i.e. no gain. This is a closed candidate, not an oracle gain.

## Why sharp is not the answer here

At a *fixed* QP the sharp filter is slightly better in PSNR: QP105 mean luma
PSNR 45.129 dB vs 44.992 dB (+0.137 dB), better on 43/60 frames (24/33 head,
19/27 tail). But at matched rate it is 0.70 K worse. The sharp predictor buys
PSNR in a band the composite metric charges for: the perceptual score falls
even though PSNR rises. That is the same pattern as the coefficient-domain
dead-zone experiments in `docs/negative-results.md`.

## Per-block switchable form

`is_filter_switchable = 1` selects `TileInterpFilterCdf` per eligible block
(the syntax, context and CDF already exist in
`src/codec/inter/mv_precision/`). That variant has a strictly higher ideal
bound than the frame-global form, but its measured inputs are poor:

- the frame-global sharp choice loses 0.70 K while gaining 0.14 dB PSNR, so a
  per-block PSNR/SSE rule would select the wrong filter;
- the block symbol costs 2-3 bits plus a context dependency, and smooth loses
  4.3 K, so only a narrow sharp subset could pay.

It stays **unmeasured** rather than claimed. Do not implement it without a
perceptual per-block RD rule and a decoder-exact switchable gate.

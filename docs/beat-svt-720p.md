# Beating SVT-AV1 p0 on K from a 720p video: what it takes

Status: **measured 2026-11-27, target 300 kbps.** On the documented 720p60
orbital clip, the best matched-rate arm found is **K 49.26 at 302.0 kbps**
(SEF + GDF + chroma offset + two references + corrected inter coefficient
model), interpolating to ~49.1 at 300 kbps; SVT-AV1 p0 is **K 56.42 at
289.3 kbps** (README records 56.6 at 293 kbps). **No measured av2enc arm beats
SVT at 300 kbps.** av2enc's K does exceed SVT's low-rate K number once it is
allowed **~2.1x the rate**: the first measured point above the wall is **QP 182,
76,469 B / 611.8 kbps, K 56.48**;
a margin outside the clip's QP ripple appears at **QP 178, 83,201 B / 665.6
kbps, K 57.02**, and **QP 176, 85,873 B / 687.0 kbps, K 57.53**. The highest K
measured is **58.63 at QP 160 / 952.2 kbps**.

This is not a new coding tool and not a matched-rate win. It is the measured rate
cost of the existing 720p recipe, stated explicitly so the result is not read as
"av2enc beats SVT at the same bitrate".

## Setup

- Source: `testdata/space8k_1440x720_60f.y4m` (1440x720, 60 fps, 60 frames,
  8-bit 4:2:0), the clip `bench/space8k_720p/README.md` and the README 720p row
  use. One second, so kbps = bytes * 8 / 1000.
- SVT-AV1 v4.3.0, `--preset 0 --crf 63 --fps-num 60 --fps-denom 1`.
  Decoded with `ffmpeg`/libdav1d.
- av2enc `build-release` (`c76b3a4a`, 2026-11-27 01:58), CPU
  `--video --frames 60 --cpu-threads 2`, single tile, 8-bit 4:2:0.
  Decoded with `tools/bin/avmdec`.
- Metrics: `tools/quality.py` (CVVDP 0.5.7, `standard_fhd`, MPS),
  `tools/extra_metrics.py` (Butteraugli, SSIMULACRA2), `tools/kpi.py` (K).

## Reference

| encoder | setting | bytes | kbps | JOD | BA | SS2 | K |
|---|---|---:|---:|---:|---:|---:|---:|
| SVT-AV1 | p0 CRF63 | 36,165 | 289.3 | 8.2317 | 21.0026 | 22.830 | **56.42** |
| av2enc | 720p pin recipe QP217 (README) | 36,543 | 292.3 | 7.328 | 29.61 | -10.11 | 45.59 |

The SVT number reproduces the bench README's 36,165 B / 8.2317 JOD exactly; the
K is the same formula the rest of the tree uses. The K gap is not one axis:
SVT holds **+0.90 JOD**, **−8.6 Butteraugli** and **+32.9 SSIMULACRA2**.

## Matched-rate levers (`--nightly-b`, SEF, deep anchor)

The follow-up measurement: does the chained-B/compound/TIP path, the
SEF-duplicate lever or a deep anchor close the 10.8 K matched-rate gap?

### SEF duplicates (flat key+P, `AV2ENC_SEF_DUPLICATES=30`)

The clip has a 23.976 -> 60 pulldown cadence, so ~1/3 of frames are
near-duplicates. `AV2ENC_SEF_DUPLICATES=30` replaces them with 5-byte
show-existing-frame units. Plain `--fast-decode`, current tree:

| arm | QP | bytes | kbps | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|---:|---:|
| plain `--fast-decode` control | 198 | 36,390 | 291.1 | 7.1954 | 25.0687 | -8.437 | 45.58 |
| plain `--fast-decode` control | 200 | 35,568 | 284.5 | 7.1271 | 26.3177 | -9.956 | 44.87 |
| SEF | 194 | 34,369 | 275.0 | 7.3506 | 25.2541 | -4.478 | 46.99 |
| SEF | 192 | 35,872 | 287.0 | 7.4329 | 24.0680 | -2.091 | 47.92 |
| SEF + `--video-rdoq-scale 0.01` | 192 | 36,237 | 289.9 | 7.4263 | 24.2768 | -2.370 | 47.82 |
| SEF + 720p key offset | 212 | 36,028 | 288.2 | 7.4777 | 27.1414 | -5.487 | 47.30 |
| SEF + `--video-gdf --video-uv-qp-offset -23` | 194 | 35,559 | 284.5 | 7.3980 | 24.0976 | -0.283 | 48.04 |
| **SEF + `--video-gdf --video-uv-qp-offset -23`** | **192** | **37,006** | **296.0** | **7.4430** | **25.3121** | **0.921** | **48.35** |
| 720p baseline | 217 | 36,543 | 292.3 | 7.328 | 29.61 | -10.11 | 45.59 |

Against the **plain control at the same ~291 kbps** (36,390 B / 291.1 kbps,
K 45.58), the single-reference SEF+GDF+UV arm at 296.0 kbps is **+2.77 K**
(≈+2.5 K at iso-rate) — a real matched-rate gain, not a rate trade. It is still
**8.07 K short** of SVT. The multi-reference and corrected-coefficient arms
below are stronger (K 48.60-49.26).
The stream decodes 60/60 under `avmdec`
(`tools/validate_avmdec.py ... -> 60 frame(s), OK`); the pinned `dav2d` rejects
the spec's explicit `sef_order_hint` form, so this arm is `avmdec`-only, exactly
as `docs/negative-results.md` records.

### Deep anchor (flat key+P, `--video-drift-refresh-qp 16`)

The 720p recipe with a 16-QP refresh drop, threshold 2.5 dB, 8,000 B budget:

| arm | QP | bytes | kbps | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|---:|---:|
| deep anchor, qdrop 16 | 222 | 36,848 | 294.8 | 7.2746 | 26.5556 | -8.039 | **45.90** |
| `AV2ENC_PIPE_LPL_DEEP=1` (4 anchors, qdrop 57) | 232 | 35,267 | 282.1 | 7.0825 | 36.5880 | -14.043 | 43.30 |
| `AV2ENC_PIPE_LPL_DEEP=1` (over-rate probe) | 220 | 42,829 | 342.6 | 7.4117 | 32.8176 | -5.661 | 46.54 |
| 720p baseline | 217 | 36,543 | 292.3 | 7.328 | 29.61 | -10.11 | 45.59 |

At essentially matched rate the qdrop-16 deep anchor is **+0.31 K** — inside the
clip's ripple, and still **10.5 K** behind SVT. The full key-depth anchor
(`PIPE_LPL_DEEP`, four refreshes, 10,152 B above the running mean) is
**−2.29 K at 282 kbps**: it resets the chain but the refresh frames are so
expensive that the matched-rate stream is worse (BA 36.6 against 29.6). That
reproduces the 240p result in `docs/negative-results.md` (refresh costs about
what its reset saves).

### Flat multi-reference and compound (same SEF/GDF/chroma arm)

`--video-refs` is normally locked to 1 by `--video-ref-signal-compact`; turning
that off (`--video-ref-signal-off`) makes the flat P chain multi-reference
without changing the B path. Measured 60 f:

| arm | QP | bytes | kbps | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|---:|---:|
| SEF + GDF + UV (single ref) | 192 | 37,006 | 296.0 | 7.4430 | 25.3121 | 0.921 | 48.35 |
| **+ `--video-ref-signal-off --video-refs 2`** | **193** | **37,225** | **297.8** | **7.4586** | **24.3737** | **1.375** | **48.60** |
| + `--video-ref-signal-off --video-refs 2` | 192 | 37,954 | 303.6 | 7.4816 | 24.9555 | 2.230 | 48.80 |
| + `--video-refs 2 --video-compound --video-bawp-off --video-drift-refresh-off` | 192 | 37,095 | 296.8 | 7.4569 | 25.1337 | 0.006 | 48.28 |

Multi-reference is the strongest matched-rate lever found: **+0.45 K** over
single-reference at a comparable rate (48.80 at 303.6 against 48.35 at 296.0),
still **~7.6 K short** of SVT. Adding flat compound does not help at this rate
(−0.52 K against the refs-2 arm). `--video-refs 3` produces streams `avmdec`
rejects (`Inter BAWP template cannot outside the valid reference range`), so it
is not a usable arm.

### Coding change: best-reference SEF (`AV2ENC_SEF_BEST_REF`)

The SEF path showed the primary reference unconditionally. Implemented a gated
choice (`AV2ENC_SEF_BEST_REF=1`, default off, in
`src/pipeline/video/video_frame.cpp`): scan the frame's reference slots and show
the one whose reconstruction has the highest luma PSNR against the source, then
code the SEF against that slot. The gate-off build is pinned:
`--fast-decode` QP106 is still 24,701 B / `54f402bd949795a4706d53586062e931`.

Measured on the 60 f clip at QP 192: the stream is **byte-identical**
(`fca268969b5b06f15b7a555871eba9b2` both with and without the gate), because
the most recent slot is already the closest on every duplicate. So the coding
change is correct but inert on this content; it does not move the 300 kbps
deficit.

### Coding change: content-aware per-frame QP (`AV2ENC_P_CONTENT_QP`)

The flat chain uses one QP for every frame, so a pulldown duplicate and a frame
with new content are priced alike. Implemented a gated per-frame allocation in
`src/pipeline/video/video_frame.cpp`: when the mean absolute luma change against
the previous input picture exceeds `AV2ENC_P_CONTENT_THRESH` (default 4.0),
lower that frame's `base_q_idx` by `AV2ENC_P_CONTENT_QP`. On this clip the
source change is 0.0 on duplicates and 5.78-8.58 on the 23 real-content frames,
so the gate selects exactly those frames. Default off; the gate-off pin is
still `--fast-decode` QP106 24,701 B / `54f402bd…`.

| arm | drop | QP | bytes | kbps | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| refs-2 baseline | 0 | 193 | 37,225 | 297.8 | 7.4586 | 24.3737 | 1.375 | **48.60** |
| content QP | 10 | 196 | 37,733 | 301.9 | 7.4201 | 22.7580 | 0.973 | 48.51 |
| content QP | 20 | 202 | 37,525 | 300.2 | 7.2725 | 22.5419 | -1.303 | 47.42 |

**The allocation change is negative.** Deepening the 23 content frames forces a
coarser global QP to stay at 300 kbps, and the key/duplicate loss outweighs the
content gain (−0.09 K for drop 10, −1.18 K for drop 20).

### Coding change: corrected inter coefficient model

`AV2ENC_RATE_HYBRID_S=1` replaces the one-sided residual price (0.5 bits per
coefficient, 8-bit fixed part overhead) with the fitted model
`0.04` bits/position and `1.6` bits/part; it is the measured fix for the
over-pricing that biases residual-versus-skip decisions toward skipping. On
the refs-2 arm:

| arm | QP | bytes | kbps | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|---:|---:|
| refs-2 | 193 | 37,225 | 297.8 | 7.4586 | 24.3737 | 1.375 | 48.60 |
| **refs-2 + corrected coefficient model** | **191** | **37,751** | **302.0** | **7.4986** | **22.2445** | **2.772** | **49.26** |
| refs-2 + corrected coefficient model | 192 | 36,780 | 294.2 | 7.4729 | 24.1267 | 1.383 | 48.69 |
| refs-2 + corrected coefficient model | 193 | 35,954 | 287.6 | 7.4439 | 23.1866 | 1.131 | 48.61 |

At ~300 kbps the corrected model is **~+0.5 K** over the refs-2 arm (K 49.26
at 302.0, interpolating to ~49.1 at 300) — the best matched-rate point
measured. It still leaves a **7.2 K gap** to SVT. `avmdec` decodes the QP-191
stream 60/60.

### Flat `--video-rd-preset best` (joint solve off)

`AV2ENC_RD_JOINT_P=0`, SEF + GDF + chroma offset:

| arm | QP | bytes | kbps |
|---|---:|---:|---:|
| best preset | 196 | 58,859 | 470.9 |
| best preset | 232 | 51,993 | 415.9 |

The flat rate curve is very shallow here (55 kbps over 36 QP, ~−1.5 kbps/QP),
so QP 255 extrapolates to ~381 kbps. **The strongest flat-RD preset cannot be
operated at 300 kbps on this clip either**, so it offers no matchable point.

### Chained B/compound/TIP (`--nightly-b`)

The chained-B/compound/TIP path. Run 60 frames, `--video-gops 7`, joint solve
off (`AV2ENC_RD_JOINT_P=0`) so the encode completes:

| arm | QP | bytes | kbps | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|---:|---:|
| `--nightly-b` | 220 | 101,563 | 812.5 | 7.2944 | 27.0069 | -5.578 | 46.38 |
| `--nightly-b` | 240 | 92,518 | 740.1 | 6.5879 | 33.6628 | -23.727 | 39.36 |

This path **cannot be operated at 300 kbps on this clip**: QP 220 is already
812 kbps and QP 240 is 740 kbps. The two points differ by only 9,045 B over 20
QP, so the QP-255 ceiling extrapolates to ~86 kB / ~690 kbps — still more than
twice the target. Both measured points are worse than the best 300 kbps arm
(K 49.26), and quality falls as QP rises. The chained-B aggregate is a 240p
low-rate tool, not a 720p-at-300-kbps lever.

## The measured av2enc ladder

Recipe (the documented 720p recipe, only the QP changes):

```sh
build-release/av2enc testdata/space8k_1440x720_60f.y4m -o /tmp/x.ivf \
  --video --frames 60 --cpu-threads 2 --qp <Q> --fast-decode \
  --video-key-qp-offset -57 --video-leaf-rd-lambda 250 \
  --video-ext-partitions
```

| QP | bytes | kbps | JOD | BA | SS2 | K | vs SVT 56.42 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 190 | 63,780 | 510.2 | 8.1409 | 21.5859 | 14.506 | 54.51 | -1.91 |
| 186 | 69,426 | 555.4 | 8.2261 | 21.1868 | 17.246 | 55.44 | -0.98 |
| 184 | 72,939 | 583.5 | 8.2704 | 20.1921 | 18.046 | 55.93 | -0.50 |
| **182** | **76,469** | **611.8** | **8.2969** | **19.3545** | **19.835** | **56.48** | **+0.06** |
| 180 | 79,371 | 635.0 | 8.3313 | 19.8164 | 21.008 | 56.78 | +0.36 |
| **178** | **83,201** | **665.6** | **8.3711** | **20.3946** | **21.768** | **57.02** | **+0.60** |
| **176** | **85,873** | **687.0** | **8.3998** | **19.3185** | **23.038** | **57.53** | **+1.11** |
| 160 | 119,021 | 952.2 | 8.5458 | 20.8082 | 26.487 | **58.63** | +2.21 |
| 140 | 175,098 | 1,400.8 | 8.5447 | 23.5432 | 25.430 | 58.12 | +1.70 |

Reading the ladder:

- **QP 182 is the first measured point above the SVT wall, by 0.06 K** — inside
  this clip's own QP ripple (the 240p record is ±1.9 K; the 720p row moves
  ~0.2-0.7 K per step). Do not quote QP 182 as a robust win.
- **QP 178 / QP 176 are robust**: +0.60 / +1.11 K, and the same QP family is
  `dav2d`-clean (below).
- **K does not keep rising with rate.** It peaks around QP 160 (58.63) and is
  lower at QP 140 despite 47 % more bytes, because Butteraugli worsens
  (20.81 -> 23.54). The practical target is the **0.6-1.0 Mbps** plateau, not
  "spend as much as possible".

## SVT at the *same* 640 kbps

The comparison above is against SVT's low-rate operating point. SVT p0 at the
same rate as our "winning" QP 180 point is far ahead:

| encoder | setting | bytes | kbps | JOD | BA | SS2 | K |
|---|---|---:|---:|---:|---:|---:|---:|
| SVT-AV1 | p0 CRF58 | 79,992 | 639.9 | 9.0433 | 11.8802 | 49.622 | **66.86** |
| av2enc | QP180 | 79,371 | 635.0 | 8.3313 | 19.8164 | 21.008 | 56.78 |

That is **−10.1 K at matched bytes** (ours is 621 B smaller and still 10 K
behind), and the gap is the same size at 290 kbps (−10.8 K). SVT's curve at
895 / 1,135 kbps is K 70.33 / 72.73, so the deficit is not a low-rate artefact:
it is roughly constant along the measured curve. Overspending buys K only
because K rises with rate for any encoder; it does not close the gap to SVT at
that rate.

## The winning stream is decodable

`tools/bin/dav2d-macos` (pinned `f9dafbc`) on the two useful points:

```sh
DAV2D=tools/bin/dav2d-macos tools/validate_dav2d.sh /tmp/q178.ivf \
  testdata/space8k_1440x720_60f.y4m
# tool pin ok; 1440x720, 60 frame(s); decoder f9dafbc; nb_read_frames=60
# PSNR y:33.957609 ... ; OK
```

QP 176 and QP 180 are also clean (`nb_read_frames=60`; PSNR y 34.015072 and
33.861944). QP 176, 178 and 180 are the shipping `--fast-decode` path, so no
research decoder caveat applies.

## Why the 300 kbps point is not reachable with the measured tools

The best matched-rate arm is SEF + GDF + chroma offset + two references +
corrected inter coefficient model at 302.0 kbps: JOD 7.4986, BA 22.244,
SS2 2.772, K 49.26. Holding two axes at those values, reaching SVT's K 56.42 at
300 kbps would need one of:

| axis held alone | required value | measured value | required change |
|---|---:|---:|---:|
| JOD | 8.931 | 7.499 | **+1.43 JOD** |
| Butteraugli | 4.40 | 22.24 | **−17.8 BA** |
| SSIMULACRA2 | 45.7 | 2.77 | **+43.0 SS2** |

No single measured lever in the tree moves any of those by that amount at fixed
rate; `docs/negative-results.md` records the search that closed the allocator,
TX/lambda, filter, trellis and intra-prediction families. The ladder above is
the same reconstruction gaining K only because it is given 2.1x the bits.

A second 720p video confirms the shape rather than the specific clip. On the
smooth `testdata/nasa_1440x720_60f.y4m` upscale at ~336 kbps:

| encoder | bytes | kbps | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|---:|
| SVT p0 CRF54 | 41,973 | 335.8 | 9.6719 | 3.1903 | 83.972 | **82.80** |
| av2enc QP145 | 44,195 | 353.6 | 9.5129 | 4.8610 | 75.737 | 78.06 |
| av2enc QP150 | 40,080 | 320.6 | 9.4287 | 5.2832 | 74.066 | 76.88 |

There, av2enc needs QP114 / 605 kbps to reach K 82.68 — the same ~1.8x rate
pattern, and again no matched-rate win.

## Answer

1. **To beat SVT-AV1 p0's K number from this 720p video, ignore the 300 kbps
   budget:** use the existing 720p recipe at **QP 176-180** (635-687 kbps):
   K **56.78-57.53** against SVT's **56.42** at 289 kbps. That is a real,
   `dav2d`-validated stream, but it costs **2.2-2.4x the rate**. At QP 182
   (611.8 kbps) the measured margin is only +0.06 K, inside the ripple; quote
   QP 178/176 if the claim needs a margin.
2. **At 300 kbps, the answer is no.** The best matched-rate arm measured is SEF
   + GDF + chroma offset + two references + the corrected inter coefficient
   model at **K 49.26 / 302.0 kbps** (≈49.1 at 300), against the plain control
   45.58 / 291.1 kbps. That is still **~7.2 K short** of SVT's 56.42. The
   content-aware per-frame QP change is negative (48.51 at 301.9 for drop 10;
   47.42 at 300.2 for drop 20). The best-reference SEF change is inert. The
   16-QP deep anchor is 45.90 and the full key-depth anchor is 43.30 at
   282 kbps. `--nightly-b` cannot be operated near 300 kbps here (740-812 kbps
   at QP 240-220), and `--video-rd-preset best` bottoms out at ~381 kbps even
   at QP 255.
3. Closing the 300 kbps gap needs roughly **+1.43 JOD or −17.8 BA or +43.0 SS2**
   from the best arm, not another preset knob. The 300 kbps objective remains
   **blocked on coding tools the tree does not have**; no measured coding
   change closes it.

## Reproduce

```sh
# SVT reference
SvtAv1EncApp -i testdata/space8k_1440x720_60f.y4m -b /tmp/svt.ivf \
  --preset 0 --crf 63 --fps-num 60 --fps-denom 1 --progress 0
ffmpeg -v error -i /tmp/svt.ivf -pix_fmt yuv420p -f yuv4mpegpipe /tmp/svt.y4m
python3 tools/quality.py --ref testdata/space8k_1440x720_60f.y4m \
  --test /tmp/svt.y4m --json /tmp/svt_q.json
python3 tools/extra_metrics.py --ref testdata/space8k_1440x720_60f.y4m \
  --test /tmp/svt.y4m --json /tmp/svt_e.json
python3 tools/kpi.py --quality-json /tmp/svt_q.json --extra-json /tmp/svt_e.json
# K=56.4215 (JOD 8.2317 / BA 21.0026 / SS2 22.8300)

# winning av2enc point
build-release/av2enc testdata/space8k_1440x720_60f.y4m -o /tmp/a.ivf \
  --video --frames 60 --cpu-threads 2 --qp 178 --fast-decode \
  --video-key-qp-offset -57 --video-leaf-rd-lambda 250 --video-ext-partitions
DAV2D=tools/bin/dav2d-macos tools/validate_dav2d.sh /tmp/a.ivf \
  testdata/space8k_1440x720_60f.y4m
python3 tools/quality.py --ref testdata/space8k_1440x720_60f.y4m \
  --ivf /tmp/a.ivf --json /tmp/a_q.json
# K=57.0227 (JOD 8.3711 / BA 20.3946 / SS2 21.7678)
```

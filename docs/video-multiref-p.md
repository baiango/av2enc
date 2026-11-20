# Multi-reference P: per-block reference selection at 240p30 / 100 kbps

Status: **negative (measured with the AVM oracle).** Branch `exp/luma-multiref`,
base `d0f03222`. The candidate is a plain single-prediction low-delay P chain
with `--video-refs 2` and `--video-refs 3`: each P frame lists the N most recent
past slots and every inter leaf picks its reference per block. There are no B
frames, no reordering and no future anchor. It loses to the deployable
one-reference nightly pin and to the eff-perc drift recipe; against an
explicit-map refs-1 control the nightly arms are only a tie inside a 1.65 K QP
ripple (one arm +0.16 K, one −0.03 K). Either way the candidate is 7.4-7.6 K
below the 88.8 wall, so no refs-4/lambda sweep or segment rows were run and
no syntax was added.

Measurement build: `cmake --preset release`, Apple M4, `--cpu-threads 8`,
`testdata/nasa_480x240_30fps_60f.y4m` (60 frames, 240p30). Scoring is
`tools/oracle.py` (AVM decoder path through `tools/quality.py`; CVVDP + per-frame
Butteraugli/SSIMULACRA2, pooled by `tools/kpi.py`). One CVVDP job at a time.

## 1. What was tested

`--video-refs N` (1..7) rotates one refresh slot per frame, lists the N most
recent valid slots newest-first in the explicit §5.18.2 reference map, and the
inter leaf loop in `video_frame_decision.cpp` evaluates every listed
`ref_index` (GLOBALMV / NEARMV / NEWMV, then the calibrated leaf RD cost). The
block codes `read_single_ref()`, so the choice is per block. Every listed
reference is a past picture; this is not `--video-compound`, not the closed-GOP
B path and not a hidden future anchor.

There is no separate per-block ref-selection knob. The count is the knob; the
search already picks the block's best listed reference. `--video-refmvbank` is
the only search-side reference extension and it cannot ride the nightly
carried-CDF chain (see §2).

## 2. Legal control and flag conflicts

`--fast-decode` expands `--video-ref-signal-compact`, which pins `NumRefFrames = 1`
and therefore requires `--video-refs 1`:

```text
av2enc: --video-ref-signal-compact needs --video-refs 1: the §7.7 derived map must hold exactly the one live slot
```

**The legal switch is `--video-ref-signal-off`.** It suppresses only that
nightly expansion element and falls back to the explicit §5.18.2 map. The map
cost is 60-65 B / 60 frames depending on byte alignment (60 B at QP106,
64 B at QP108, 61-65 B on the eff-perc bracket; 0.24-0.26% of the stream).
The map changes no decoded picture: JOD, BA, SS2 and K are identical at the
same QP. The arms are compared two ways: against a refs-1 control with the
same explicit map (isolates the reference count) and against the plain nightly
pin (the deployable one-reference recipe, which charges the map to the arm).
Both are reported because 60 B is enough to move the narrow interpolation.

Other conflicts checked:

| flag | composes with multi-ref nightly? | evidence |
|---|---|---|
| `--video-drift-refresh` | **yes** | the only named rejection is `--video-compound`; the eff-perc drift arms with refs 2/3 encode and decode 60/60 |
| `--video-bawp` / `--video-bawp-carrier` | **yes** | no reference-count gate; the carrier tracks `ref_index` |
| `--video-refmvbank` | **no** | `error: reference-management tools are not wired with the average-CDF/CDF-continuity chain`; dropping the chain costs ≈4.5 K by itself (`docs/reference-structure.md` §3.3) |
| `--video-ref-mvs` | no (nightly shape) | requires `--video-edge-inter` off, which costs 4.7 K before the tool is switched on; with two refs it also measured −2.21 K in `docs/reference-structure.md` §3.4 |
| `--video-compound` | no (and out of scope) | BAWP rejects compound; this is the already-measured two-hypothesis path |

Legal controls used here:

```sh
# nightly control (reference count 1, same explicit map as the arms)
av2enc REF -o ctrl.ivf --video --frames 60 --cpu-threads 8 --qp Q \
    --fast-decode --video-ref-signal-off --video-refs 1

# multi-reference arms
av2enc REF -o ref2.ivf --video --frames 60 --cpu-threads 8 --qp Q \
    --fast-decode --video-ref-signal-off --video-refs 2

# eff-perc drift composition (control takes --video-refs 1)
AV2ENC_EFF_PERC=1 av2enc REF -o eff-ref3.ivf --video --frames 60 \
    --cpu-threads 8 --qp Q --fast-decode --video-ref-signal-off --video-refs 3 \
    --video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250
```

`--video-refs 4` was not screened: the refs 2/3 arms are 7.4-7.6 K below the
wall and at best a tie inside the QP ripple, so the decision rule says stop.
The earlier non-oracle screen measured refs4 at **80.978** at 100 kbps
(`docs/reference-structure.md` §3.1), below refs2/3; the count trend is not
toward a win.

## 3. Cheap screen: nightly QP104/106/108 (plus QP107)

The nightly pin is reproduced exactly at QP106 (25,008 B, MD5
`0639b43630abb454f7be5ac494d2efb4`, K 82.155). `base` is the plain nightly
stream with its compact map; `ctrl` is the legal explicit-map control. Their K
is identical at every QP; only the 60-65 B reference map differs.

The prescribed 104/106/108 bracket skipped QP107, where the K surface sits
1.65 K below QP106 at almost the same rate. QP107 was therefore added to
resolve the oracle ripple (the oracle document's own "three points are better
than two" rule); it is not a tuning point. The refs2/refs3 QP107 sizes and K reproduce
`docs/reference-structure.md` exactly (24,804 B / 80.697 and 24,998 B /
81.190), so the added point is independent corroboration, not a new stream.

| candidate | QP | bytes | kbps | PSNR-Y | SSIM | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 104 | 26,158 | 104.632 | 44.493 | 0.9941 | 9.7185 | 4.1720 | 80.108 | 80.734 |
| base | 106 | 25,008 | 100.032 | 44.668 | 0.9942 | 9.7334 | 3.4202 | 80.837 | 82.155 |
| base | 107 | 24,885 | 99.540 | 44.545 | 0.9940 | 9.7113 | 4.3229 | 80.256 | 80.508 |
| base | 108 | 24,291 | 97.164 | 44.598 | 0.9938 | 9.7216 | 3.6824 | 80.633 | 81.604 |
| ctrl refs1 | 104 | 26,219 | 104.876 | 44.493 | 0.9941 | 9.7185 | 4.1720 | 80.108 | 80.734 |
| ctrl refs1 | 106 | 25,068 | 100.272 | 44.668 | 0.9942 | 9.7334 | 3.4202 | 80.837 | 82.155 |
| ctrl refs1 | 107 | 24,948 | 99.792 | 44.545 | 0.9940 | 9.7113 | 4.3229 | 80.256 | 80.508 |
| ctrl refs1 | 108 | 24,355 | 97.420 | 44.598 | 0.9938 | 9.7216 | 3.6824 | 80.633 | 81.604 |
| refs2 | 104 | 26,175 | 104.700 | 44.704 | 0.9942 | 9.6948 | 3.7655 | 80.494 | 81.309 |
| refs2 | 106 | 25,182 | 100.728 | 44.627 | 0.9941 | 9.7323 | 3.4630 | 80.454 | 82.008 |
| refs2 | 107 | 24,804 | 99.216 | 44.541 | 0.9939 | 9.7080 | 4.1866 | 80.323 | 80.697 |
| refs2 | 108 | 24,586 | 98.344 | 44.728 | 0.9939 | 9.7239 | 3.7016 | 80.721 | 81.598 |
| refs3 | 104 | 26,138 | 104.552 | 44.867 | 0.9944 | 9.7239 | 3.7195 | 81.083 | 81.628 |
| refs3 | 106 | 25,407 | 101.628 | 44.600 | 0.9940 | 9.7062 | 3.7770 | 80.512 | 81.350 |
| refs3 | 107 | 24,998 | 99.992 | 44.572 | 0.9939 | 9.7158 | 3.9004 | 80.451 | 81.190 |
| refs3 | 108 | 24,715 | 98.860 | 44.624 | 0.9939 | 9.7307 | 3.4566 | 80.560 | 82.029 |

Matched rate (linear interpolation of K to 100.000 kbps across each candidate's
own 4-point ladder). The 100 kbps target falls in the narrow QP107..QP106
interval for every candidate:

| candidate | matched K @100 | vs nightly pin | vs legal ctrl | vs 88.8 | method / bracket |
|---|---:|---:|---:|---:|---|
| nightly base (compact) | **82.048** | — | — | −6.75 | linear (base@107..base@106) |
| nightly ctrl refs1 | **81.222** | — | — | −7.58 | linear (ctrl@107..ctrl@106) |
| refs2 | **81.377** | **−0.671** | +0.155 | −7.42 | linear (ref2@107..ref2@106) |
| refs3 | **81.191** | **−0.857** | −0.031 | −7.61 | linear (ref3@107..ref3@106) |

The matched-rate point estimate is not stable at this bracket width. The K
surface jumps 1.65 K from QP107 (80.508) to QP106 (82.155) over 0.5-1.6 kbps,
and the 60 B map moves the control's QP107 point from 99.540 to 99.792 kbps,
which changes the interpolation fraction from 0.94 to 0.43 and drops the
control's interpolated K by 0.83 K without changing a decoded picture. The
refs2 +0.155 against that control is the same artefact: at the two QPs that
bracket 100 kbps the same-QP deltas are **refs2 +0.189 at QP107 and −0.147 at
QP106**, and **refs3 +0.682 at QP107 and −0.805 at QP106**. Both arms change
sign across the bracket, so neither meets the "≥ +0.15 K over two QPs" gate
in a reproducible way. Against the nightly pin, which is the deployable
one-reference recipe, refs2 loses 0.671 K and refs3 loses 0.857 K.

The wider QP104/108 points show the same shifted surface, not a trend: at
QP104 refs2 is +0.575 and refs3 is +0.894 at equal or slightly lower rate; at
QP108 refs2 is −0.006 and refs3 is +0.425. QP104 is 4.7% above the deployment
rate. A single favourable QP is not an oracle.

## 4. Eff-perc drift composition

The current best internal recipe is `AV2ENC_EFF_PERC=1` on the tuned drift
point. It composes with multi-reference (the drift guard only rejects
`--video-compound`). Legal control: the same recipe with `--video-refs 1` and
the explicit map.

| candidate | QP | bytes | kbps | PSNR-Y | SSIM | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eff-ctrl refs1 | 119 | 25,291 | 101.164 | 44.463 | 0.9929 | 9.7713 | 2.2630 | 82.441 | 85.180 |
| eff-ctrl refs1 | 120 | 24,739 | 98.956 | 44.341 | 0.9927 | 9.7625 | 2.2874 | 81.924 | 84.985 |
| eff refs2 | 119 | 25,316 | 101.264 | 44.470 | 0.9929 | 9.7706 | 2.3370 | 82.432 | 84.978 |
| eff refs2 | 120 | 24,833 | 99.332 | 44.381 | 0.9927 | 9.7630 | 2.3943 | 82.076 | 84.731 |
| eff refs3 | 119 | 25,337 | 101.348 | 44.487 | 0.9928 | 9.7696 | 2.2945 | 82.443 | 85.087 |
| eff refs3 | 120 | 24,874 | 99.496 | 44.383 | 0.9927 | 9.7621 | 2.4335 | 82.141 | 84.637 |

| candidate | matched K @100 | vs eff-ctrl | vs 85.10 best internal | vs 88.8 |
|---|---:|---:|---:|---:|
| eff-ctrl refs1 | **85.077** | — | −0.02 | −3.72 |
| eff refs2 | **84.816** | **−0.261** | −0.28 | −3.98 |
| eff refs3 | **84.760** | **−0.317** | −0.34 | −4.04 |

The unmodified eff-perc baseline reproduces the published 85.10 here
(85.077 at 100.0 with the explicit-map control; 85.099 with the shipped
compact map). Multi-reference does not add to it; the two arms are 0.26 and
0.32 K below the control, and BA rises at both QPs (2.263/2.287 → 2.337/2.394
for refs2, 2.295/2.434 for refs3).

## 5. Oracle vs the walls

| candidate | matched K @100 | vs SVT 88.8 | vs AVM 90.2 |
|---|---:|---:|---:|
| nightly base (1 ref, compact) | 82.048 | −6.75 | −8.15 |
| nightly ctrl (1 ref, explicit map) | 81.222 | −7.58 | −8.98 |
| multi-ref P, refs 2 | 81.377 | −7.42 | −8.82 |
| multi-ref P, refs 3 | 81.191 | −7.61 | −9.01 |
| eff-perc control (1 ref) | 85.077 | −3.72 | −5.12 |
| eff-perc + refs 2 | 84.816 | −3.98 | −5.38 |
| eff-perc + refs 3 | 84.760 | −4.04 | −5.44 |

**No multi-reference P arm is a path.** The best measured point is the
unmodified eff-perc control at 85.08-85.10, still 3.7 K below SVT and 5.1 K
below AVM. Adding references moves it the wrong way on both the nightly and
eff-perc recipes; the nightly refs2 arm is inside the QP ripple of the
explicit-map control but 0.67 K below the deployable nightly pin.

## 6. Why it loses

This is the same conclusion as `docs/reference-structure.md` §3.1/§3.5, now
measured with the AVM oracle and with a legal explicit-map control:

* Reference *count* is not the missing lever. The leaf loop already picks the
  best listed reference; adding older pictures changes which leaf skips and
  which leaf spends residual, but does not allocate bits to the tail. The
  P-tail remains an allocation problem (`docs/luma-audit.md`).
* The ref count itself is inside the QP ripple. Against the explicit-map
  refs-1 control the matched deltas are +0.16/−0.03 K, but the sign flips at
  the two bracketing QPs (+0.19/−0.15 and +0.68/−0.81) and the control's own
  interpolated K moves 0.83 K from a 60 B map change with identical decoded
  pictures. Against the deployable nightly pin the arms lose 0.67/0.86 K.
* The refs3 QP104 local win (+0.89), QP106 loss (−0.81), QP107 win (+0.68)
  and QP108 win (+0.43) are the same non-monotone surface documented for refs4
  in `docs/reference-structure.md`. A single QP point is not an oracle.
* The ref-MV bank is not a way around this: it cannot be enabled with the
  nightly carried-CDF chain, and the chain is worth ≈4.5 K.

No segment or source-replacement ceiling was measured. The cheap screen is
negative, so per the go/no-go rule there is no tuning below 88.8 and no reason
to build the larger sweep.

## 7. Reproduction

```sh
REF=testdata/nasa_480x240_30fps_60f.y4m
BIN=build-release/av2enc

# nightly base/control and the two arms at each QP. QP107 is required: the
# QP106/107 step is a 1.65 K K-jump, so a 106/108 interpolation is not usable.
for QP in 104 106 107 108; do
  "$BIN" "$REF" -o /tmp/base-$QP.ivf --video --frames 60 --cpu-threads 8 \
      --qp "$QP" --fast-decode
  "$BIN" "$REF" -o /tmp/ctrl-$QP.ivf --video --frames 60 --cpu-threads 8 \
      --qp "$QP" --fast-decode --video-ref-signal-off --video-refs 1
  for R in 2 3; do
    "$BIN" "$REF" -o /tmp/ref$R-$QP.ivf --video --frames 60 --cpu-threads 8 \
        --qp "$QP" --fast-decode --video-ref-signal-off --video-refs "$R"
  done
done

python3 tools/oracle.py --ref "$REF" --no-segments \
  --ivf base@104=/tmp/base-104.ivf --ivf base@106=/tmp/base-106.ivf \
  --ivf base@107=/tmp/base-107.ivf --ivf base@108=/tmp/base-108.ivf \
  --ivf ctrl@104=/tmp/ctrl-104.ivf --ivf ctrl@106=/tmp/ctrl-106.ivf \
  --ivf ctrl@107=/tmp/ctrl-107.ivf --ivf ctrl@108=/tmp/ctrl-108.ivf \
  --ivf ref2@104=/tmp/ref2-104.ivf --ivf ref2@106=/tmp/ref2-106.ivf \
  --ivf ref2@107=/tmp/ref2-107.ivf --ivf ref2@108=/tmp/ref2-108.ivf \
  --ivf ref3@104=/tmp/ref3-104.ivf --ivf ref3@106=/tmp/ref3-106.ivf \
  --ivf ref3@107=/tmp/ref3-107.ivf --ivf ref3@108=/tmp/ref3-108.ivf \
  --target-kbps 100 --json /tmp/multiref.oracle.json

# eff-perc composition
for QP in 119 120; do
  AV2ENC_EFF_PERC=1 "$BIN" "$REF" -o /tmp/eff-ctrl-$QP.ivf --video \
      --frames 60 --cpu-threads 8 --qp "$QP" --fast-decode \
      --video-ref-signal-off --video-refs 1 \
      --video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250
  for R in 2 3; do
    AV2ENC_EFF_PERC=1 "$BIN" "$REF" -o /tmp/eff-ref$R-$QP.ivf --video \
        --frames 60 --cpu-threads 8 --qp "$QP" --fast-decode \
        --video-ref-signal-off --video-refs "$R" \
        --video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250
  done
done
```

Hand-off artifacts (schema `luma-oracle/1`): the nightly 16-point ladder is
`/tmp/multiref/screen4.json` (ctrl/refs2/refs3 × QP104/106/107/108) plus the
base rows in `/tmp/multiref/screen.json` and `/tmp/multiref/qp107.json`; the
eff-perc rows are `/tmp/multiref/eff-screen.json` (6 points) plus the two
`eff-ctrl` rows in `/tmp/multiref/controls.json`. The tables above are copied
from those files; the matched K is linear interpolation over each ladder.

## 8. Validation and caveats

* Representative streams decode 60/60 with `tools/validate_dav2d.sh`
  (`refs2` QP106/QP108, `refs3` QP106/QP108, eff-refs2 QP119/QP120, eff-refs3
  QP119/QP120); the oracle itself decodes with `avmdec`.
* `build-release` is warning-free (`AV2ENC_WARNINGS=ON`, `AV2ENC_WERROR=ON`),
  all five pins are byte-identical (25,009 `a1d0dba1`, 25,141 `3984677b`,
  23,690 `1b99127a`, 24,113 `7f03eb6e`, 720p 36,572 `c4ba2813`), and
  `tools/test.sh --build build-release` is 10/10.
* One clip, one resolution, one operating point. The QP ripple on this clip is
  ±1 K (1.65 K at the nightly QP106/107 step). The nightly comparison against
  the explicit-map control is a tie inside that ripple (+0.16/−0.03 K); against
  the deployable nightly pin it is −0.67/−0.86 K. The eff-perc bracket is
  adjacent-QP and smooth, and is unambiguously negative (−0.26/−0.32 K). The
  conclusion is the wall gap and the eff-perc result, not a single QP point.
* The 60-65 B map cost is measured, not estimated: base and ctrl differ by
  60-65 B and have identical decoded pictures and K at the same QP.
* Encode cost was not re-timed. `docs/reference-structure.md` measured
  +13-15% for refs2/3/4 and +38-72% with drift on the same host; nothing here
  changes that.

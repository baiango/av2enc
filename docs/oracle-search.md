# Oracle search: automatic candidate scoring

Status: **automation.** Branch `exp/luma-oracle`, base `a0aa1097`. This change
adds a scoring harness and this document; it does not change encoder code.

The harness exists because the K oracle was a manual one-off after every
experiment. A candidate now hands over its IVF ladder and gets the same
matched-rate K, the same segment ceilings, and the same wall verdict as every
other candidate. No candidate is compared at its own favourite bitrate.

## 1. Definition

For a candidate `P` with knobs `θ`:

```
K*_P(100) = best K_P(θ) at 100.000 kbps
```

`K` is `tools/kpi.py` over the three published components:
CVVDP/JOD (50%), Butteraugli (25%) and SSIMULACRA2 (25%). CVVDP is scored over
the whole sequence; BA and SS2 are per-frame arithmetic means. This is the same
pooling as `docs/luma-audit.md`.

The harness reports three different bounds:

| Bound | Question it answers | How it is measured |
|---|---|---|
| **Matched-rate K** | Is the candidate competitive at the deployment point? | Linear interpolation of K across the candidate's own QP ladder to 100.000 kbps. **Only this number decides go/no-go.** |
| **Rate-free ceiling** | Could the mechanism reach the wall if rate were free? | The best measured point of the same mechanism with the rate constraint removed (e.g. all frames at the key QP). A lower bound on the mechanism, not an encodable 100 kbps stream. |
| **Segment ceiling** | Which part of the sequence owns the gap? | Replace one frame range of the decoded stream with source frames (or another decoder's frames), re-score all 60 frames, report ΔK. A pooling bound, not a stream. |

A segment ceiling above the wall is **not** a claim. Infinite bits on 33 frames
can exceed 90 K without saying anything about an encoder. The only number that
can open a path is the matched-rate K.

## 2. Run it

### Requirements

- `tools/bin/avmdec` (or `AVMDEC=/path/to/avmdec`); the harness imports
  `tools/quality.py` and uses its `find_avmdec`/`decode_avm`. There is no
  second decoder.
- `ffmpeg`, `butteraugli_main`, `ssimulacra2`, and `cvvdp` on `PATH`
  (`CVVDP_CMD` also works). `python3` needs no extra packages.
- An 8-bit 4:2:0 Y4M reference. The AVM path rejects 10-bit and 4:2:2/4:4:4.
- Candidate streams must decode to the reference geometry and frame count.
- The segment oracles are defined for the 60-frame fixture; on other lengths
  they are skipped and the JSON says why.

### One command per candidate

Single point:

```sh
REF=testdata/nasa_480x240_30fps_60f.y4m
python3 tools/oracle.py --ref "$REF" \
  --ivf nightly=/tmp/nightly.ivf \
  --target-kbps 100 --json /tmp/nightly.oracle.json
```

A QP ladder is a repeated `--ivf` with `candidate@point` labels. The labels
group the streams; the point name is yours (`119`, `q119`, etc.):

```sh
python3 tools/oracle.py --ref "$REF" \
  --ivf eff@118=/tmp/eff118.ivf \
  --ivf eff@119=/tmp/eff119.ivf \
  --ivf eff@120=/tmp/eff120.ivf \
  --target-kbps 100 --json /tmp/eff.oracle.json
```

Rules for the ladder:

- Change only QP between points. Every other knob is fixed.
- Bracket 100 kbps. A ladder that stops at 98 or starts at 103 is reported as
  `nearest` and cannot be a matched-rate oracle.
- Three points are better than two: the K surface has a QP ripple, and a single
  bracket can land on the wrong side of it.
- The tool labels a one-point candidate `single-point`; that is a measurement,
  not an oracle.

### Segment ceilings

Source replacement is always reported:

```sh
python3 tools/oracle.py --ref "$REF" \
  --ivf nightly=/tmp/nightly.ivf \
  --target-kbps 100 --json /tmp/nightly.oracle.json
```

The audit's cross-candidate replacement is reproduced by decoding the other
stream once and passing it with `--decoded`. `avmdec` only decodes AV02, so the
SVT AV1 stream is decoded outside the harness (the audit used the same decoded
frames). A `--decoded` label is scored as a segment replacement automatically;
`--segment-from` is only needed to select a subset:

```sh
SvtAv1EncApp -i "$REF" -b /tmp/svt.ivf --preset 0 --crf 36.5 \
    --fps-num 30 --fps-denom 1 --progress 0
ffmpeg -v error -i /tmp/svt.ivf -pix_fmt yuv420p -f yuv4mpegpipe /tmp/svt.dec.y4m

python3 tools/oracle.py --ref "$REF" \
  --ivf nightly=/tmp/nightly.ivf \
  --ivf eff@119=/tmp/eff119.ivf --ivf eff@120=/tmp/eff120.ivf \
  --decoded svt=/tmp/svt.dec.y4m --segment-from svt \
  --target-kbps 100 --json /tmp/audit.oracle.json
```

| flag | effect |
|---|---|
| `--target-kbps` | matched-rate target, default 100 |
| `--ivf LABEL=PATH` | candidate stream; repeat for a ladder. `name@point` groups points |
| `--decoded LABEL=PATH` | pre-decoded Y4M used only for segment replacement; scored automatically unless `--segment-from` selects a subset |
| `--segment-from LABEL` | also score that replacement (`source`, a scored point, or a `--decoded` label); source rows are always included |
| `--no-segments` | matched-rate only; fast |
| `--json PATH` | the hand-off artifact |
| `--keep` | keep decoded/replaced Y4M files for inspection |

Output is a Markdown report on stdout and the full JSON at `--json`. Progress
goes to stderr.

### JSON

The schema is `luma-oracle/1`. It carries every stream (bytes, kbps, PSNR,
SSIM, JOD, BA, SS2, K and K components), the interpolated matched K with its
bracket and method, the segment rows with ΔK, and the wall verdict. Attach it
to the candidate result; do not re-type numbers.

## 3. Validation against the pinned streams

The harness was validated on the 480x240 / 60-frame fixture with the exact
pins from `docs/luma-audit.md` and `docs/eff-perc.md`.

### Produce the validation streams

Build the encoder from the exact base commit (`a0aa1097`); a shared `build-dev`
can be older than the eff-perc gate. On this host a clean build reproduced
25,008 / 25,230 / 24,674 B; a stale `build-dev` produced 22,446 B for the
eff-perc point and would have silently invalidated the comparison.

```sh
BIN=/path/to/clean/build/av2enc
REF=testdata/nasa_480x240_30fps_60f.y4m

"$BIN" "$REF" -o /tmp/nightly.ivf --video --frames 60 --cpu-threads 8 \
    --qp 106 --nightly
AV2ENC_EFF_PERC=1 "$BIN" "$REF" -o /tmp/eff119.ivf --video --frames 60 \
    --cpu-threads 8 --qp 119 --nightly \
    --video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250
AV2ENC_EFF_PERC=1 "$BIN" "$REF" -o /tmp/eff120.ivf --video --frames 60 \
    --cpu-threads 8 --qp 120 --nightly \
    --video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250
```

### Streams

| stream | bytes | kbps | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|---:|
| nightly QP106 | 25,008 | 100.032 | 9.7334 | 3.4202 | 80.837 | **82.155** |
| eff-perc QP119 | 25,230 | 100.920 | 9.7713 | 2.2630 | 82.441 | **85.180** |
| eff-perc QP120 | 24,674 | 98.696 | 9.7625 | 2.2874 | 81.924 | 84.985 |
| eff-perc matched at 100.000 | — | 100.000 | — | — | — | **85.099** |

The QP119/QP120 bracket gives 85.099 at 100.000 kbps, i.e. the documented
85.10. The nightly single point is 82.155 at 100.032 kbps.

### Segment oracles

The audit's section-6 numbers are **cross-candidate replacements** with SVT's
decoded frames, not source replacements. The harness reports both. With
`--segment-from svt` it reproduces the audit:

| base | replacement | key f0 | f0-32 | f33-43 | f44-59 |
|---|---|---:|---:|---:|---:|
| nightly 82.155 | SVT | −0.003 | +0.903 | +1.102 | +3.758 |
| eff-perc 85.180 | SVT | +0.008 | +1.244 | +0.717 | +1.456 |

Rounded: nightly +0.00 / +0.90 / +1.10 / +3.76 and eff-perc +0.01 / +1.24 /
+0.72 / +1.46, matching the audit's +0.91 / +1.11 / +3.76 and +1.24 / +0.72 /
+1.46 within the documented ±0.1 K mixing ripple.

The stronger source-replacement ceiling on the same streams:

| base | key f0 | f0-32 | f33-43 | f44-59 |
|---|---:|---:|---:|---:|
| nightly 82.155 | +0.084 (82.239) | +3.700 (85.855) | +2.212 (84.367) | +5.911 (88.066) |
| eff-perc 85.180 | +0.125 (85.305) | +5.017 (90.197) | +2.119 (87.300) | +3.815 (88.995) |

Read the source table as a bound only. It says the early chain and the tail
each have enough room to matter; it does not say any encoder can buy that room
at 100 kbps. The audit's SVT replacement is the smaller, achievable-reference
version of the same probe. **The default source rows are therefore not expected
to equal the audit's +0.91/+1.11/+3.76 numbers**; they are a stronger bound.
Pass the SVT frames with `--decoded` to reproduce the audit exactly.

## 4. Standing candidate queue

A queue entry without a matched-rate oracle is a question, not a candidate.
The owner is the branch or workstream that can answer it. Results below are
matched K at 100 kbps unless marked as a ceiling.

| candidate | status | owner | result | next |
|---|---|---|---|---|
| Hidden future anchor / alt-ref | **running** | inter / alt-ref workstream | pending; no committed oracle | The one untested prediction class (audit §9): a one-key hidden-ARF pyramid, not the closed-GOP B family already measured negative. Intermediate gate K > 86.0 to continue; the standing wall is 88.8. Run the ladder and post the JSON. |
| Multi-reference P (2-3 past refs, per-block selection) | closed (measured) | `exp/luma-multiref` | **81.38 / 81.19** (refs 2/3 @100) | Plain single-prediction P chain over the N most recent past slots; every leaf picks its reference. Requires `--video-ref-signal-off` (the compact map needs one ref). Against the nightly pin (82.05) the arms lose 0.67/0.86 K; the explicit-map refs-1 control is a tie inside a 1.65 K QP ripple (+0.16/−0.03). On the eff-perc drift recipe 84.82/84.76 against 85.08. The ref-MV bank cannot ride the carried-CDF chain. Details in `docs/video-multiref-p.md`. Do not queue. |
| Hierarchical B/P | closed (measured) | `exp/eff-hierb` | **84.40** | Two-sided interiors + BA allocation; BAWP composes (+0.33 same-QP, +0.50 iso), but the anchor cost leaves it 0.45 below the P chain. Reopen only with a free anchor / hidden-ARF budget. |
| Two-sided + anchor | closed (measured) | `exp/eff-twosided` | **84.49** | Predictor wins at the same QP (83.29 → 84.65, BA 2.455); the reordered-anchor penalty (+6.7% stream) eats it. +0.50 iso with BAWP. Combine only with a cost-free anchor. |
| Learned / tool-class | closed (bound) | metrics workstream | **0 at 100 kbps** | Encoder-side learned predictors need coded residual; the normative learned filters (Wiener/PC-Wiener/GDF) are negative at 240p. The 95 wall is a new-codec program. Do not queue for 100 kbps. |
| Tiles | closed (240p quality) | parallel workstream | **n/a** | Tiles are entropy-independent parallelism, not an RD tool. 8x2 costs ≈5% bytes on one 720p frame and there is no 240p quality lever. Keep them for GPU throughput. |
| Coefficient entropy | closed (bounded) | entropy workstream | **≤0.3 K** | The coder is 1.015× its own per-context bound; pooled contexts cost 39% more. The bound is below the 88.8 wall. Do not spend. |

The audit's rank-1 frequency-weighted trellis was tested after the audit and is
also closed: the guarded inter half is −0.143 K at g=0.2 and the most favourable
unguarded point is +0.094 K with sign flips; the eff-perc point is byte-identical
(`docs/video-rdoq-hf.md`). Do not queue it as an open path.

## 5. Current measured oracles

All at 240p / 100 kbps unless noted. "Ceiling" is rate-free or allocation, not
an encodable 100 kbps stream.

| candidate | recipe / gate | ladder | matched K @100 | ceiling | vs 88.8 wall |
|---|---|---|---:|---:|---:|
| nightly | `--qp 106 --nightly` | 1 point | **82.155** | — | −6.65 |
| drift | nightly + threshold 1.5 / budget 3250 | 2 points | **83.80** (83.94 @100.58) | — | −5.00 |
| pipe-perc | perceptual RD pipeline | 2 points | **84.85** | — | −3.95 |
| eff-hierb | hierarchical B + two-sided + BAWP | envelope | **84.40** | — | −4.40 |
| eff-twosided | two-sided prediction + anchor | envelope | **84.49** | — | −4.31 |
| eff-joint | affordable joint search | envelope | **84.59** | 84.65 exhaustive | −4.21 |
| **eff-perc** | error-CSF + interval rate + λ refit | 2 points | **85.10** | — | −3.70 |
| multi-ref P refs 2 | nightly + `--video-ref-signal-off --video-refs 2` | 4 points | **81.38** | — | −7.42 |
| multi-ref P refs 3 | nightly + `--video-ref-signal-off --video-refs 3` | 4 points | **81.19** | — | −7.61 |
| eff-perc + refs 2 | eff-perc + `--video-ref-signal-off --video-refs 2` | 2 points | **84.82** | — | −3.98 |
| eff-perc + refs 3 | eff-perc + `--video-ref-signal-off --video-refs 3` | 2 points | **84.76** | — | −4.04 |
| deep inter | anchor depth envelope | — | 84.02 | **87.69** allocation (2.4× rate) | −4.78 (ceiling −1.11) |
| SVT-AV1 4.3.0 p0 | CRF 36.5 | 2 points | **88.8** | — | wall |
| AVM e6f99d3 | QP 124 | 2 points | **90.2** | — | target |

No internal candidate is above the wall. The best internal oracle is
**85.10** (eff-perc), 3.70 K short of SVT and 5.10 K short of AVM. That is the
honest starting point for the queue. The decoded hybrid attributes 3.32 K of
the 3.93 K gap to luma and 0.61 K to chroma; it is an attribution tool, not an
encodable candidate, so it is not in the oracle table.

## 6. Go/no-go rule

1. **No tuning below SVT 88.8.** If the matched-rate oracle of an architecture
   is below 88.8, stop. Report the envelope and the ceiling; do not sweep λ,
   QP offsets, or tool flags inside it. A local win over 82.15 is not a path.
2. **An architecture must clear 88.8 to be pursued.** Above 88.8 it is
   competitive with the strongest non-AVM encoder and worth implementation
   effort.
3. **90.2 is the AVM wall.** Clearing it beats the reference encoder at the
   same 100 kbps point. Nothing in the current queue does.
4. **Do not claim a path without an oracle above the wall.** A segment ceiling,
   a rate-free ceiling, a single QP point, or a trend line is not an oracle.
   Only the matched-rate K from a bracketing ladder is.

SVT's measured point is 24,844 B / 99.376 kbps / K 88.641, which interpolates
to 88.8 at 100.0. AVM is 24,836 B / 99.344 kbps / K 90.192, which rounds to
90.2 at 100.0. The walls are quoted at the deployment point, not at each
encoder's incidental rate.

## 7. Caveats

- **QP ripple.** The same recipe can move 0.3-1.5 K between adjacent QPs. That
  is why the ladder must bracket the target and why a single fixed-QP "win" is
  not an oracle.
- **Mixing ripple.** Segment replacement re-pools a sequence that was never
  encoded as one; the audit records ±0.1 K. Do not read a ±0.1 ΔK as a result.
- **CVVDP saturation.** At JOD 9.7-9.9 the CVVDP component is near its ceiling;
  BA and SS2 carry most of the K difference. A candidate that only moves JOD is
  suspect.
- **Source replacement is infinite bits.** It is a ceiling on the value of a
  segment, not a rate-feasible mechanism.
- **Segment definitions.** `key f0` is frame 0 alone; `f0-32` includes frames
  0-32 (key plus static head); `f33-43` is the fade onset; `f44-59` is the tail.
  The rows are independent; they do not sum to the whole-sequence gap.
- **One fixture.** This document's numbers are the 480x240 / 60-frame fixture.
  The harness works on any 8-bit 4:2:0 Y4M, but the segment rows and the walls
  are pinned to this clip until the queue is ported to the 720p point.

## 8. Candidate hand-off

1. Freeze every knob except QP. Encode at least two QPs that bracket 100 kbps;
   three is better.
2. Run one `tools/oracle.py` command per candidate with `--json`.
3. Attach the JSON and put the matched-rate K in the queue's result column.
4. If the matched K is below 88.8, stop and report the envelope and ceiling.
   Do not tune inside the candidate.
5. If the matched K is above 88.8, state the headroom to 90.2 and the segment
   row that owns it. A segment ceiling alone is not a claim.

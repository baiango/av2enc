# Oracle search: automatic candidate scoring

Status: **automation + self-driving architecture loop.** Branch
`exp/luma-search`, base `0bf1d001`. This change adds a scoring loop and this
document; it does not change encoder code. The candidate unit is an
**architecture** (source + flags); a parameter grid is only that
architecture's inner sweep.

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

> **Scheduling is now registry-driven.** This section is the measured result
> table and the historical queue. `docs/oracle-candidates.json` +
> `tools/oracle_search.py` decide what runs next (section 9).

A queue entry without a matched-rate oracle is a question, not a candidate.
The owner is the branch or workstream that can answer it. Results below are
matched K at 100 kbps unless marked as a ceiling.

| candidate | status | owner | result | next |
|---|---|---|---|---|
| Hidden future anchor / alt-ref | **closed (measured)** | `exp/luma-structure` | **78.34 @ spacing 4**; legal full-tool subset **76.76 vs flat control 79.36** (topology **-2.6 K**); perfect spacing 4 = 87.2; only perfect spacing 2 = 89.8 (not decodable, below AVM) | Full-nightly-tool alt-ref is illegal: key-offset, drift, tile-group-compact, ref-signal-compact, BAWP and partition-RD are rejected. The ~2.8-3.2 K gap to the nightly pin is composition; the topology term is negative. Do not tune spacing/QP/lambda. Details in `docs/hidden-anchor-oracle.md`/`docs/video-open-gop.md`. |
| TIP long chain | **closed (measured)** | `exp/luma-tip` | **50.50 @ G=4**; ideal spacing 4 = 87.2 | `TIP_FRAME_AS_OUTPUT` cannot spend rate on displayed pictures; block-mode TIP is bounded by 87.2 at spacing 4. Details in `docs/tip-long-chain.md`. |
| Multi-reference P (2-3 past refs, per-block selection) | closed (measured) | `exp/luma-multiref` | **81.38 / 81.19** (refs 2/3 @100) | Plain single-prediction P chain over the N most recent past slots; every leaf picks its reference. Requires `--video-ref-signal-off` (the compact map needs one ref). Against the nightly pin (82.05) the arms lose 0.67/0.86 K; the explicit-map refs-1 control is a tie inside a 1.65 K QP ripple (+0.16/−0.03). On the eff-perc drift recipe 84.82/84.76 against 85.08. The ref-MV bank cannot ride the carried-CDF chain. Details in `docs/video-multiref-p.md`. Do not queue. |
| Prediction tool combinations | **closed (measured)** | `exp/combo-pred` | **82.331** best; source-replacement ceiling **88.555** | 22 legal 60-frame arms. Best is BAWP-off (known marginal, inside ripple); best novel combo loses to its own control. Every arm's rate-free ceiling is below 88.8. GME/ref-MV/compound/OPFL/morph/palette are illegal or decode-invalid on the nightly carrier. Details in `docs/combo-pred.md`. |
| Hierarchical B/P | closed (measured) | `exp/eff-hierb` | **84.40** | Two-sided interiors + BA allocation; BAWP composes (+0.33 same-QP, +0.50 iso), but the anchor cost leaves it 0.45 below the P chain. Reopen only with a free anchor / hidden-ARF budget. |
| Two-sided + anchor | closed (measured) | `exp/eff-twosided` | **84.49** | Predictor wins at the same QP (83.29 → 84.65, BA 2.455); the reordered-anchor penalty (+6.7% stream) eats it. +0.50 iso with BAWP. Combine only with a cost-free anchor. |
| Learned / tool-class | closed (bound) | metrics workstream | **0 at 100 kbps** | Encoder-side learned predictors need coded residual; the normative learned filters (Wiener/PC-Wiener/GDF) are negative at 240p. The 95 wall is a new-codec program. Do not queue for 100 kbps. |
| Tiles | closed (240p quality) | parallel workstream | **n/a** | Tiles are entropy-independent parallelism, not an RD tool. 8x2 costs ≈5% bytes on one 720p frame and there is no 240p quality lever. Keep them for GPU throughput. |
| Coefficient entropy | closed (bounded) | entropy workstream | **≤0.3 K** | The coder is 1.015× its own per-context bound; pooled contexts cost 39% more. The bound is below the 88.8 wall. Do not spend. |

The audit's rank-1 frequency-weighted trellis was tested after the audit and is
also closed: the guarded inter half is −0.143 K at g=0.2 and the most favourable
unguarded point is +0.094 K with sign flips; the eff-perc point is byte-identical
(`docs/video-rdoq-hf.md`). Do not queue it as an open path.

## 5. Composition graph

The per-tool matrix is `docs/tool-matrix.md` (flag, family, preset state,
carrier, named rejection, measured result, legal carriers, untapped
potential). This section condenses it into what can coexist, what needs a
structural carrier, and what is dead as a single toggle but untested in
combination. **This is a graph, not a cap.** A rejected composition is a
carrier mismatch. An untested pair is a budget item, not a closed path.

### 5.1 Families and carriers

`Y` = legal and gated; `C` = legal only after the named ablation; `N` =
rejected by name or decoder; `—` = not applicable.

| family | FP | B | TIP | TL | T | SVC | W | SC | LL |
|---|---|---|---|---|---|---|---|---|---|
| Flat-P decisions (edge path, key QP offset, drift refresh, partition probe, RDOQ, BAWP, CDF carry) | Y | N | N | N | C | N | N | N | N |
| Intra prediction add-ons (CFL, MRL, IBP, DIP, IST, IDTX) | Y | Y | C | Y | C | N | N | N | N |
| Transform/quant add-ons (TX search, key intra TX part, TCQ, CCTX, DDT) | Y | Y | C | Y | C | N | N | N | N |
| QM (`--video-qm/--video-user-qm`) | C | C | N | C | N | N | N | N | N |
| Loop filters (deblock, CDEF, CCSO, GDF) | Y | Y | C | Y | Y1 | N | N | N | N |
| Loop restoration (Wiener, PC-Wiener, chroma LR) | Y | Y | C | Y | N | N | N | N | N |
| Delta-Q / AQ / segmentation | C2 | N | N | N | N | N | N | N | N |
| Reference structure (B, TL, TIP, refs >1, compound) | C3 | Y | N | Y | C | Y4 | N | N | N |
| Screen content (palette, IBC, morph) | C5 | N | N | N | N | N | N | Y | N |
| Tiles | C6 | C6 | C6 | C6 | Y | N | N | N | N |
| SVC layers | C7 | N | N | N | N | Y | N | N | N |
| Header/overhead levers (tile flush, tile-group compact, ref compact, CCSO header RD) | Y | C | C | C | C | N | C | N | N |

1. CDEF/CCSO/GDF only. Deblock, Wiener, PC-Wiener and chroma LR reject
   `--tiles > 1x1` (`validate.cpp` and `docs/negative-results.md`).
2. Only with `--video-cdf-per-frame-qp`; the carried image does not include the
   §5.18.7 delta/segment CDFs. Segmentation and per-segment Q additionally need
   `--video-rdoq-off`; delta-Q and AQ compose with RDOQ.
3. `--video-refs > 1` needs `--video-ref-signal-off`; compound also needs
   `--video-bawp-off --video-drift-refresh-off --video-cdf-chain-off` (the
   carried CDF image desynchronises the pinned decoder with compound syntax:
   3/60 pictures).
4. Single-layer SVC only; `--video-refs/--video-tlayers` are not wired into the
   SVC path.
5. Needs the CDF chain off, loop filters off, TX search off (palette) and
   single tile; morph also rejects DRL_ALWAYS and segmentation.
6. `--tiles` needs the edge pair, deblock, drift, probe and RDOQ off. CDEF,
   CCSO and GDF compose with tiles.
7. SVC rejects key-QP offset, drift refresh, TX search, partition probe, RDOQ,
   refs, tlayers and tiles.

### 5.2 Coexistence rules

* **The nightly stack is one locked flat-P object.** It simultaneously owns
  the bottom-edge path, the key/first-P QP offset, the drift refresh, the
  partition probe, RDOQ, BAWP, the compact reference/tile/CCSO headers and the
  carried CDF image. A tool that changes the reference structure (B, TL, TIP,
  W), the writer (lossless), the layer model (SVC) or the tile grid must drop
  most of that stack. The measured value of the dropped elements is 13.4 K of
  marginal (`docs/nightly-ablation.md`), which is why none of those carriers is
  a cheap composition.
* **Entropy is the most exclusive family.** `--cdf-continuity` cannot ride
  delta-Q, AQ, segmentation, MHCCP, reference-management tools, palette/IBC or
  compound. The reset form `--video-cdf-per-frame-qp` costs 2.9 % rate but makes
  delta-Q/AQ/segmentation expressible. `--video-tile-group-compact`,
  `--video-ref-signal-compact` and `--video-tile-flush-min` are flat-P,
  single-tile header levers and are cheap to drop (0.04–0.11 K each).
* **Prediction locks.** BAWP excludes compound, GME, inter-intra and adaptive
  MVD/flex precision. Compound excludes BAWP, drift refresh, compact reference
  signalling and the CDF carry. Inter-intra excludes BAWP and morph. IBP
  excludes CfL, MHCCP, MRL, screen content and inter-intra. GME excludes
  inter residual, B/TL/W, compound and DRL.
* **Transform locks.** RDOQ excludes TCQ, QM, segmentation, UV delta-Q and
  lossless. TX search excludes QM, frame-type wrappers and SVC. TCQ and CCTX
  compose with each other and with TX search; DDT is independent but measured
  negative alone.
* **Filter locks.** Deblock/CDEF/CCSO/GDF compose on FP, B and TL. Frame-type
  wrappers reject every filter row because their component headers do not
  write it. Tiles keep CDEF/CCSO/GDF but reject deblock and loop restoration.
  Wiener and PC-Wiener are mutually exclusive.
* **Allocation locks.** Key-QP offset and drift refresh are flat-P cqp only.
  The partition probe is flat-P, single-tile, 8-bit 4:2:0 and rejects B/TL/TIP/W,
  SVC, tiles and the 128/256 trees. AQ/segmentation/delta-Q need the CDF reset
  and reject frame schedules. `--video-qp-lambda` needs cqp; `--rc crf/vbr` is
  therefore hard-blocked inside `--nightly` (nightly always sets qp-lambda and
  has no off switch).
* **Structural carriers.** Compound overlays need B (or refs 2 on P);
  OPFL/refine-MV/CWP/IMP need B + `--video-compound-near-near`; TIP needs its
  own 3–8-frame key/anchor/TIP schedule and rejects loop filters and TX search;
  palette/IBC/morph need the single-tile screen-content keyframe; SVC needs
  `--svc-layers`; the frame-type wrappers own their §5.18.2 header writers.

### 5.3 Dead as a single toggle, untested in combination

| tool | single-toggle result | untested combination |
|---|---|---|
| `--video-tcq` | +0.01 JOD for ~60x encode on the quality preset (`video-rd-preset.md`) | TCQ + CCTX on `--nightly --video-rdoq-off` |
| `--video-cctx` | neutral on the quality preset | same |
| `--video-ddt` | −1.22 K matched rate (`partition-tools.md`) | SDP+DDT already measured below its control; no open pair |
| `--video-mrl` | selected 64/189 key leaves without a measured win (`negative-results.md`) | MRL on the nightly aggregate, with the keyframe TX part |
| `--video-aq` | +0.05 K inside ripple; −0.47 K under nightly (`video-aq.md`) | AQ with the drift/partition relaxations at a different strength — measured null, not a path |
| `--video-delta-q` | 6.161 JOD vs 6.302 preset (`video-rd-preset.md`) | delta-Q with the probe + drift at `--video-drift-refresh-qp 0`; the zero-plan stream is recon-identical |
| `--video-adaptive-mvd` | neutral on the quality preset | adaptive MVD + flex precision + sign derivation on `--nightly --video-bawp-off` |
| `--video-ext-partitions` | −0.47 K on tuned recipe | ext + `--video-min-block 4` together (both compose with nightly) |
| `--video-min-block 4` | −0.74 K on tuned recipe | same pair |
| `--video-idtx`, `--video-ist`, `--video-ibp`, `--video-dip`, `--video-cfl` | no composite K | each on the full nightly aggregate |
| `--video-cdef-strength 4` | header cost only (~6.4 B/frame) | multi-strength CDEF on nightly |
| `--video-cdef-rd-lambda` | changes padded reconstruction; no iso-rate sweep | priced CDEF-row drop on nightly |
| `--video-mhccp` | no K; blocked by the CDF bank | MHCCP + `--video-cdf-chain-off` |
| `--video-palette/--video-ibc/--video-morph` | no K; screen-content only | stripped nightly + screen carrier, ≥2 SB rows |

### 5.4 Composition candidates (ranked, not a cap)

Ranked by expected oracle headroom. Full flag lists and carriers are in
`docs/tool-matrix.md` §10.

| rank | candidate | carrier | status | next |
|---:|---|---|---|---|
| 1 | TIP future anchor on the stripped flat-P carrier | TIP (3–8 pictures) | untested; decoder-gated only | intermediate gate K > 86.0, then a matched ladder |
| 2 | CDEF per-block strength (`--nightly --video-cdef-strength 4`) | FP | untested on K; CDEF is 60-frame gated | one ladder; stop if below the local control |
| 3 | CfL on the nightly tree (`--nightly --video-cfl`) | FP | accepted in a 2-frame probe; 60-frame composition untested | one ladder; chroma owns 0.61 K of the gap |
| 4 | Adaptive MVD + flex precision + sign derivation (`--nightly --video-bawp-off ...`) | FP | gated alone, untested together | one ladder; costs 0.36 K to make legal |
| 5 | IBP on every intra leaf (`--nightly --video-ibp`) | FP | gated alone; 60-frame composition untested | one ladder; keyframe-weighted |

Every measured dead combination (B, compound, multi-ref, GME, SDP, AQ,
delta-Q, ext partitions, min-block 4, skip collapse, Wiener/PC-Wiener/
chroma-LR/GDF, TX context rate, sb128/256) stays closed at 240p/100 kbps.
None of the five candidates above is claimed to clear the 88.8 wall; they are
the remaining legal selections that have not had an oracle ladder.

## 6. Current measured oracles

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

## 7. Go/no-go rule

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

## 8. Caveats

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

## 9. Candidate hand-off

1. Freeze every knob except QP. Encode at least two QPs that bracket 100 kbps;
   three is better.
2. Run one `tools/oracle.py` command per candidate with `--json`.
3. Attach the JSON and put the matched-rate K in the queue's result column.
4. If the matched K is below 88.8, stop and report the envelope and ceiling.
   Do not tune inside the candidate.
5. If the matched K is above 88.8, state the headroom to 90.2 and the segment
   row that owns it. A segment ceiling alone is not a claim.

## 9. Self-driving loop

`tools/oracle_search.py` runs the architecture oracle without a person choosing
the next encode. It reads `docs/oracle-candidates.json`, probes ceilings, gates
compute, screens inner knobs, promotes finalists, and writes the decision log.
The oracle itself is architecture-agnostic: any candidate that can emit an IVF
ladder goes through the same `tools/oracle.py` call.

### 9.1 Candidate = architecture, not a parameter point

A registry entry is one architecture. Its knobs are an inner sweep of that
architecture, never a comparison across architectures. A 0.18 lambda point and
a 0.20 lambda point are the same architecture; a hierarchical-B branch and a
flat-P branch are different architectures.

```json
{
  "name": "eff-perc",
  "status": "open",
  "source": {
    "kind": "binary",
    "note": "main build-release"
  },
  "build": null,
  "flags": ["{bin}", "{ref}", "-o", "{out}", "--video", "--frames",
            "{frames}", "--cpu-threads", "{threads}", "--qp", "{qp}",
            "--nightly", "--video-drift-refresh-threshold", "1.5",
            "--video-drift-refresh-budget", "3250"],
  "env": {
    "AV2ENC_EFF_PERC": "1",
    "AV2ENC_EFF_PERC_SCALE": "{lambda}"
  },
  "qp_ladder": [118, 119, 120, 121],
  "knobs": {
    "lambda": {"values": [0.15, 0.18, 0.20, 0.25, 0.30], "default": 0.18}
  },
  "ceiling": {"kind": "source-segment"}
}
```

Fields:

| field | meaning |
|---|---|
| `name` | stable candidate id; also the oracle label prefix |
| `status` | `open`, `measured`, `closed` or `blocked`; the default run selects `open` |
| `source.kind` | `binary` (use `--bin`), `worktree` (read-only prebuilt binary), `commit`/`branch` (build a commit or branch), or `patch` (build a base commit + patch) |
| `source.binary` | binary path for `worktree`; optional override for `binary` |
| `source.worktree` | existing tree for `worktree`; the tool never builds in or modifies it |
| `source.commit` / `source.branch` / `source.base_commit` | tree to build for `commit`/`branch`/`patch` sources |
| `source.patch` | patch file for `patch`; applied to a `git archive` extraction in the cache, not to another worktree |
| `build.command` | build argv for `commit`/`patch`; placeholders `{src}`, `{build}`, `{bin}`, `{jobs}`, `{root}` |
| `build.binary` | binary path template produced by the build |
| `flags` | argv template; `{bin}`, `{ref}`, `{out}`, `{frames}`, `{threads}`, `{qp}` plus knob names |
| `env` | environment template; all `AV2ENC_*` variables are cleared before this map is applied, so a candidate cannot inherit another candidate's gate |
| `qp_ladder` | sorted QPs available to the architecture; the screen takes the middle `--screen-qps` and extends only when 100 kbps is not bracketed |
| `knobs` | inner sweep; each knob is a list or `{values, default}`; the cross product is capped at 64 arms |
| `ceiling` | `source-segment` (default), `decoded` (use `--replacement`), or `declared` with a measured `k` |
| `validation` | optional known-best / expected-gate / reference-K record |

A non-scalar knob value is not supported. Use a separate knob for each
independent flag. The QP is the rate axis, not a knob.

### 9.2 Architecture oracle: three numbers

For each architecture the loop reports:

1. **Ideal/rate-free ceiling.** The default is the source-segment replacement
   from `tools/oracle.py`: replace one frame range with the source and re-pool
   the 60-frame sequence. The reported ceiling is the matched-rate K plus the
   largest source `delta_k`. This is the optimistic budget gate. It is a
   pooling bound, not an encodable stream. A decoded-domain replacement (for
   example SVT frames standing in for an ideal future reference) enters with
   `--replacement LABEL=Y4M`; a previously measured allocation or rate-free
   bound enters as `"ceiling": {"kind": "declared", "k": 87.69}`. The
   replacement is a run-level flag; use `--only` when one architecture needs a
   distinct ideal-future or decoded-domain reference.
2. **Matched-rate envelope over the inner knobs.** The best matched-rate K
   across the architecture's arms, at 100.000 kbps. Screens use two QPs and a
   short ladder extension; finalists use the full `qp_ladder`.
3. **Gap to the walls.** `envelope - 88.8` and `envelope - 90.2`. A ceiling's
   gap is reported separately and is never mixed into the result column.

The three are deliberately different. A ceiling can clear 88.8 while the
envelope does not; that is a statement about where the mechanism could go, not
a result. Only the matched-rate envelope decides go/no-go.

### 9.3 Selection policy

Deterministic, in registry order:

1. **Gate probe.** Probe the default arm and the far end of the knob grid at
   two QPs, with source segments. Compute the ceiling. A probe whose ladder
   does not bracket 100 kbps is not an oracle and is excluded.
2. **Hard gate.** If no valid probe ceiling reaches 88.8, close the
   architecture. Do not screen the remaining knobs, do not run a full ladder,
   do not sweep. The decision log records the ceiling and the missing margin.
3. **Screen.** If the gate opens, screen the remaining arms at two QPs. Extend
   the QP set only when the target rate is not bracketed. The default screen is
   `--screen-qps 2`; `--max-screen-qps` caps extension.
4. **Promote.** Rank the valid screens. Keep arms within
   `--promotion-margin` of the best and cap at `--max-finalists`. This is
   successive halving over the knob grid: a wide grid is cut to a small
   finalist set before any full ladder.
5. **Full ladder.** Run the complete QP ladder only for finalists, with source
   segments. The full matched K is the envelope. If it is below 88.8, stop; the
   decision log says `no full sweep` and the remaining arms stay unmeasured.
6. **Above-wall sweep.** Only a confirmed envelope at or above 88.8 unlocks
   the remaining arms and full ladders. This is the only path to an exhaustive
   inner sweep. `--no-expand-above-wall` disables even that.

The surrogate is a small pure-Python GP with an RBF kernel, standardized
inputs and EI or UCB acquisition (`--acquisition ei|ucb`). It orders a large
grid before screening and logs its mean/sigma/acquisition for every arm. It
does **not** replace a measurement: the envelope is always the measured
matched-rate K. With fewer than two measured arms it falls back to registry
order.

### 9.4 Wall rules

1. **88.8 is the SVT wall; 90.2 is the AVM wall.** Both are quoted at
   100.000 kbps.
2. **A ceiling is a compute gate, not a claim.** It answers "could this
   architecture be worth a full ladder?" It never answers "does it clear
   88.8?".
3. **No full sweep below 88.8.** A candidate whose screen/ceiling cannot clear
   the wall gets no exhaustive knob sweep. A finalist full ladder is still
   allowed because it is the measurement that decides go/no-go.
4. **No tuning inside a closed architecture.** If the full envelope is below
   88.8, the loop stops and reports the envelope and ceiling. It does not try
   another lambda to find a local win.
5. **No claim without a bracketing ladder.** A single point, a nearest-rate
   point, a segment ceiling, or a trend line is not an oracle.
6. **State the plain result.** When the run finishes below the wall, the
   Markdown and JSON say `No candidate clears 88.8.` There is no rounding up.

### 9.5 Proposal hook: how code enters

A proposal is a new architecture, not a new knob. The loop accepts it through
the same oracle:

1. Put the implementation in a patch file (or a worktree with a prebuilt
   binary). Do not edit the search worktree's encoder source.
2. Add a registry entry with `status: "open"`, the source (`patch` with
   `base_commit` + `patch`, or `worktree` with `binary`), the build command for
   a patch, the flag template, env, QP ladder, knobs, and a ceiling policy:

   ```json
   {
     "name": "hidden-arf",
     "status": "open",
     "source": {
       "kind": "patch",
       "base_commit": "<main commit>",
       "patch": "../proposals/hidden-arf.patch"
     },
     "build": {
       "command": ["cmake", "-S", "{src}", "-B", "{build}",
                   "-DCMAKE_BUILD_TYPE=Release"],
       "binary": "{build}/av2enc"
     },
     "flags": ["{bin}", "{ref}", "-o", "{out}", "--video", "--frames",
               "{frames}", "--cpu-threads", "{threads}", "--qp", "{qp}",
               "--nightly", "<proposal flags>"],
     "env": {"AV2ENC_HIDDEN_ARF": "1"},
     "qp_ladder": [118, 119, 120, 121],
     "knobs": {},
     "ceiling": {"kind": "source-segment"}
   }
   ```

3. Run `tools/oracle_search.py --only <name> --bin <clean-av2enc>`.
4. The proposal gets the same gate probe and ceiling. If the ceiling cannot
   clear 88.8, it is closed before a full ladder. If it can, finalists get the
   full ladder; below-wall envelopes stop the sweep.
5. Attach `docs/oracle-results.json` and the per-candidate decision log.

Patch builds are extracted with `git archive` into `.oracle-cache/builds/` and
built there. The tool never applies a patch to, or builds in, another agent's
worktree. A patch that fails to build is a registry error, not a result.

The oracle is architecture-agnostic: any encoder that can write an IVF ladder
can be added. It does not need to know what the code does.

### 9.6 Determinism, resume and outputs

- Arms are generated in registry order in the cross-product order; ties break
  on the canonical arm JSON. No random seed is used.
- Scores cache by `(candidate, qp, md5)`; complete ladders cache by the sorted
  `(qp, md5)` list, target, oracle hash, segment mode and replacements. Encode
  blobs live in `.oracle-cache/blobs/` and are reused only when the md5 matches.
- A rerun of a completed registry is a cache read: on this host the three
  validation candidates replayed in 0.26 s with 15 ladder-cache hits and zero
  oracle calls.
- Outputs: `docs/oracle-results.json` (schema `oracle-search/1`),
  `docs/oracle-search.md` (the marked generated section), and
  `docs/oracle-decisions/<name>.md` per candidate.
- `--dry-run` prints the resolved arms without running. `--only` and
  `--status` select candidates. `--keep-work` keeps encoder scratch dirs.

### 9.7 Validation run

Command used on this host:

```sh
BIN=${AV2ENC_BIN:-../av2enc/build-release/av2enc}
ORACLE=${AV2ENC_ORACLE:-../av2enc/tools/oracle.py}
python3 tools/oracle_search.py \
  --only nightly,eff-perc,eff-hierb \
  --bin "$BIN" --oracle "$ORACLE"
```

The fixture is `testdata/nasa_480x240_30fps_60f.y4m`, 60 frames, 8-bit 4:2:0,
target 100.000 kbps, 8 threads. The results below are measured, not copied from
the earlier tables.

| candidate | architecture | gate | gate ceiling K | matched envelope K | gap SVT | full ladders | result |
|---|---|---|---:|---:|---:|---:|---|
| nightly | flat key+P baseline | closed | 87.959 | 82.048 (screen) | −6.752 | 0/1 | gate closed; no full ladder |
| eff-perc | perceptual RD core + drift recipe, lambda inner sweep | open | 90.116 | **85.099** | −3.701 | 2/5 | lambda 0.18 is the best full arm; below wall, stop |
| eff-hierb | hierarchical B, BA refresh, two-sided compound | open | 89.201 | **84.412** | −4.388 | 1/1 | reproduces the 84.4 reference; below wall, stop |

The selector found the known eff-perc best point with **two** full ladders out
of five arms: it screened all five under the two-QP policy, extending the QP
set only where 100 kbps was not bracketed, promoted lambda 0.18 and 0.25, and
the full ladders confirmed lambda 0.18 at 85.099. Brute force would have run
all five full ladders. The lambda 0.30 gate probe did not bracket 100 kbps at
QPs 118-121 (its nearest matched rate was 99.056), so it was excluded from the
gate and from promotion; it was not rounded into a result.

Nightly's source ceiling is **87.959**, just below the wall, so the gate closed
before any full ladder. The earlier manual source-replacement table gives
88.066 for the same mechanism (82.155 + 5.911) at an incidental rate; the
matched-rate probe here is 82.048 and its ceiling is 87.959. Both are below
88.8. This is the expected near-miss, not a new result.

The eff-hierb run used the existing `../eff-hierb/build-release/av2enc` binary
read-only and reproduced the recorded **84.4** envelope at **84.412**
(Δ +0.012). No encoder source was modified. Hidden ARF is not available as a
code candidate on this tree, and `--video-tip` codes only 3-8 pictures, so it
cannot produce the 60-frame matched-rate ladder; eff-hierb is the available
architecture candidate. A hidden-ARF patch can enter through section 9.5 when
one exists.

**No candidate clears 88.8.** The best measured envelope is eff-perc at
**85.099**, 3.701 K short of SVT and 5.101 K short of AVM. Two ceilings clear
88.8 (eff-perc 90.116, eff-hierb 89.201), which is exactly why the gate spent
finalist ladders on them; the envelopes did not, so neither is a path.

### 9.8 Limitation

The loop allocates experiments, screens candidates, tunes inner
parameters/structures, and updates its beliefs from measured results. It
**cannot invent a new coding tool**. A registry entry without an
implementation is a question, not a candidate; the oracle can rank a
hierarchical-B branch, a patch, or a new predictor only after a proposal agent
writes the code. The next layer is that proposal/implementation agent: it
generates patches, the same gate probe and full ladder rank them, and no patch
gets a different oracle or a lower wall. Until then, the honest statement is
the one in the generated table: no candidate clears 88.8.

<!-- oracle-search:generated:begin -->
## 10. Generated architecture search

Fixture: `testdata/nasa_480x240_30fps_60f.y4m`, 60 frames, target 100.000 kbps. Walls: SVT 88.8, AVM 90.2. Oracle: `../av2enc/tools/oracle.py`.

| candidate | gate | gate ceiling | screen envelope | full envelope | gap SVT | gap AVM | full ladders | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **nightly** | closed | 87.959 | 82.048 | — | -6.752 | -8.152 | 0/1 | gate closed below SVT 88.8 |
| **eff-perc** | open | 90.116 | 85.099 | 85.099 | -3.701 | -5.101 | 2/5 | NO-GO: below SVT 88.8 |
| **eff-hierb** | open | 89.201 | 84.412 | 84.412 | -4.388 | -5.788 | 1/1 | NO-GO: below SVT 88.8 |

**No candidate clears 88.8.**

A ceiling above the wall is not a claim: only the matched-rate K from a bracketing ladder is. The gate uses the ceiling to decide whether an architecture is worth a full ladder; it never promotes a ceiling row into the result column.

### Validation

| candidate | check | observed | result |
|---|---|---|---|
| nightly | gate expected closed | closed | pass |
| eff-perc | known best `{"lambda":0.18}` | `{"lambda":0.18}` | pass |
| eff-hierb | reference K 84.40 | 84.412 | pass |

- nightly: 0 full ladder(s) vs 1 for brute force; saved 1.
- eff-perc: 2 full ladder(s) vs 5 for brute force; saved 3.
- eff-hierb: 1 full ladder(s) vs 1 for brute force; saved 0.

### Decision logs

Per-candidate logs: [`nightly.md`](oracle-decisions/nightly.md), [`eff-perc.md`](oracle-decisions/eff-perc.md), [`eff-hierb.md`](oracle-decisions/eff-hierb.md).

<!-- oracle-search:generated:end -->

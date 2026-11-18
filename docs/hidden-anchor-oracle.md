# Hidden future anchors / alt-ref: go/no-go against SVT 88.8 and AVM 90.2

Status: **NO-GO.** A one-key open GOP with periodic hidden future anchors
(alt-ref) does not clear either wall in any encodable form measured here.

| bound | best value | vs SVT 88.8 | vs AVM 90.2 |
|---|---:|---:|---:|
| matched-rate envelope, real schedule (harness) | **78.34** (spacing 4) | −10.46 | −11.86 |
| rate-free ceiling, same mechanism (spacing 4, 314.8 kbps) | **79.50** | −9.30 | −10.70 |
| ideal perfect-anchor oracle, spacing 8 | **83.9** | −4.9 | −6.3 |
| ideal perfect-anchor oracle, spacing 4 | **87.2** | −1.6 | −3.0 |
| ideal perfect-neighbour oracle, spacing 2 | **89.8** | +1.0 | −0.4 |

For scale, the same-tool flat key+P control is **78.98** at 100.73 kbps
(harness, nearest) and the nightly pin is **82.155** at 100.032 kbps. The
hidden-anchor schedule is below the flat control at essentially the same
rate, and its rate-free ceiling is only 0.5 K above that flat point at 3.1x
the bitrate.

Only the spacing-2 perfect-neighbour oracle clears SVT, and it misses AVM by
0.4 K. Spacing 2 is not a hidden-anchor schedule: every frame is both a
reference and a displayed picture, so the oracle's "perfect reference" would
require coding every frame near-losslessly. The rate-free ceiling of the real
mechanism (314.8 kbps, 3.1x the target) is **79.50**, so that oracle is not
approachable by this architecture.

Branch `exp/luma-structure`, based on `main` at `a1159bb8`; the oracle harness
is the copy of `main`'s `tools/oracle.py` / `docs/oracle-search.md` at
`7c8ba8d5`. 240p fixture `testdata/nasa_480x240_30fps_60f.y4m`, 60 frames,
`tools/quality.py` with the AVM `avmdec` decoder. No second decoder.

## 1. What was implemented

`AV2ENC_OPEN_GOP_ALTREF=1` (default off) selects a new two-sided schedule in
the open-GOP planner:

* one key at display 0, then units of `AV2ENC_OPEN_GOP` frames (4/8/16);
* the next unit's anchor is coded **first** as a hidden picture
  (`immediate_output_frame = 0`, `implicit_output_frame = 0`);
* every visible frame in the unit is coded from the two endpoints of its
  interval `{past anchor, future anchor}` (then `{0,4}`, `{4,8}`, …), so the
  block layer can choose either side per block;
* after the unit's last visible frame, an `OBU_REGULAR_SEF` shows the hidden
  anchor (`derive_sef_order_hint = 1`); its §7.21.6 output pass emits the
  lower-hint leaves first, then the anchor. Output is display order and the
  decoded picture count is 60;
* only the key is visible by coding position; no extra keyframes.

The hidden anchors are coded at the base QP by default; two experiment-only
knobs allocate to them: `AV2ENC_OPEN_GOP_KEY_QP_OFFSET` and
`AV2ENC_OPEN_GOP_ANCHOR_QP_OFFSET`. The best envelope applies `-35` to the key
and `0` to the anchors.

Verified: `dav2d --strict 1` and `avmdec` both decode 60 output pictures for
spacing 4/8/16; `--video-recon` equals the decoded Y4M byte for byte on the
spacing-4 matched-rate stream; the encoder is warning-free and the gate-off
pins are unchanged (§5).

## 2. Harness table

Command (the `altref4-free` point is the rate-free ceiling of the same
mechanism):

```sh
python3 tools/oracle.py --ref testdata/nasa_480x240_30fps_60f.y4m \
  --ivf altref4@129=/tmp/arK4_129.ivf --ivf altref4@130=/tmp/arK4_130.ivf \
  --ivf altref4@131=/tmp/arK4_131.ivf \
  --ivf altref8@134=/tmp/arK8_134.ivf --ivf altref8@136=/tmp/arK8_136.ivf \
  --ivf altref8@138=/tmp/arK8_138.ivf \
  --ivf altref16@148=/tmp/arK16_148.ivf --ivf altref16@150=/tmp/arK16_150.ivf \
  --ivf altref16@152=/tmp/arK16_152.ivf \
  --ivf altref4-free@60=/tmp/arK4_60.ivf \
  --target-kbps 100 --json /tmp/hidden.oracle.json
```

All ladders change only QP. The spacing-4 ladder uses key offset −35 and
`--video-rdoq-scale 0.2`; spacing 8/16 the same; the tool set is the
`--nightly`-compatible CPU set minus the tools that reject the temporal
schedule (`--video-edge-inter --video-edge-mask --video-deblock --video-cdef
--video-cdef-rd --video-tx-search --video-intra-full --video-rdoq
--video-leaf-rd --video-key-intra-txpart --video-subpel-eighth
--video-qp-lambda --avg-cdf --adaptive-cdf --cdf-continuity
--video-ccso*`).

### Streams

| candidate | QP | bytes | kbps | PSNR-Y | SSIM | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| altref4 | 129 | 25,413 | 101.652 | 41.794 | 0.9898 | 9.5561 | 5.163 | 75.750 | 77.927 |
| altref4 | 130 | 24,930 | 99.720 | 41.972 | 0.9899 | 9.5676 | 4.809 | 75.778 | 78.406 |
| altref4 | 131 | 24,783 | 99.132 | 41.839 | 0.9895 | 9.5646 | 4.668 | 75.266 | 78.482 |
| altref8 | 134 | 25,177 | 100.708 | 39.917 | 0.9861 | 9.4273 | 6.770 | 69.637 | 74.752 |
| altref8 | 136 | 24,939 | 99.756 | 39.598 | 0.9856 | 9.3644 | 7.445 | 68.964 | 73.830 |
| altref8 | 138 | 24,659 | 98.636 | 39.353 | 0.9847 | 9.3390 | 7.217 | 67.816 | 73.672 |
| altref16 | 148 | 25,049 | 100.196 | 37.819 | 0.9788 | 9.1808 | 7.836 | 64.020 | 71.829 |
| altref16 | 150 | 24,689 | 98.756 | 37.516 | 0.9772 | 9.1446 | 8.089 | 62.420 | 71.223 |
| altref16 | 152 | 23,495 | 93.980 | 36.986 | 0.9753 | 9.0901 | 7.267 | 61.421 | 71.325 |
| altref4-free | 60 | 78,706 | 314.824 | 44.987 | 0.9964 | 9.6457 | 4.641 | 78.759 | 79.504 |

### Matched rate

| candidate | ladder | matched kbps | K @ 100 | method | verdict |
|---|---:|---:|---:|---|---|
| altref4 | 3 | 100.000 | **78.336** | linear (130..129) | NO-GO |
| altref8 | 3 | 100.000 | **74.066** | linear (136..134) | NO-GO |
| altref16 | 3 | 100.000 | **71.746** | linear (150..148) | NO-GO |
| altref4-free | 1 | 314.824 | 79.504 | single-point | NO-GO |

### Segment ceilings (source replacement)

The harness replaces one frame range with the source and re-scores the full
60-frame sequence. These are pooling bounds, not streams.

| candidate | base kbps | segment | K | dK |
|---|---:|---|---:|---:|
| altref4 | 99.720 | key f0 | 78.511 | +0.105 |
| altref4 | 99.720 | f0-32 | 83.035 | +4.629 |
| altref4 | 99.720 | f33-43 | 81.447 | +3.042 |
| altref4 | 99.720 | f44-59 | 83.606 | +5.201 |
| altref8 | 99.756 | key f0 | 73.992 | +0.162 |
| altref8 | 99.756 | f0-32 | 80.946 | +7.116 |
| altref8 | 99.756 | f33-43 | 75.976 | +2.146 |
| altref8 | 99.756 | f44-59 | 79.041 | +5.211 |
| altref16 | 100.196 | key f0 | 72.029 | +0.200 |
| altref16 | 100.196 | f0-32 | 79.379 | +7.550 |
| altref16 | 100.196 | f33-43 | 74.922 | +3.094 |
| altref16 | 100.196 | f44-59 | 76.895 | +5.066 |
| altref4-free | 314.824 | key f0 | 79.526 | +0.022 |
| altref4-free | 314.824 | f0-32 | 82.925 | +3.421 |
| altref4-free | 314.824 | f33-43 | 82.589 | +3.085 |
| altref4-free | 314.824 | f44-59 | 85.083 | +5.579 |

No single segment replacement reaches 88.8 even at the rate-free point. The
key is worth +0.02..+0.2 K, so the gap is the inter pictures, spread across
the sequence; the largest single segment (f0-32, 33 frames) is +4.6..+7.6 K.

## 3. Ideal-reference oracle (manual)

This is the first task's bound, not a stream: every inter picture is coded
against the **source** of its past/next anchor (perfect, rate-free reference
quality), while all 60 frames are still coded and counted. The QP ladder is
swept to 100 kbps; the `--video-recon` Y4M is scored directly (the IVF is not
decoder-conformant because the perfect reference is not in the DPB). This
bounds what perfect future anchors could buy. It is deliberately stronger
than a real schedule.

| spacing | key offset | bracket (kbps / K) | K@100 | vs 88.8 | vs 90.2 |
|---:|---:|---|---:|---:|---:|
| 2 | −27 | 99.8 / 89.81 · 103.3 / 89.75 | **89.8** | +1.0 | −0.4 |
| 4 | −15 | 99.2 / 87.14 · 97.6 / 87.27 | **87.2** | −1.6 | −3.0 |
| 8 | 0 | 102.1 / 83.82 · 97.6 / 83.95 | **83.9** | −4.9 | −6.3 |
| 16 | −27 | 113.4 / 76.85 · 87.8 / 75.34 | **76.1** | −12.7 | −14.1 |
| 32 | −27 | 113.7 / 76.83 · 86.4 / 75.08 | **76.0** | −12.8 | −14.2 |

Only spacing 2 clears SVT, and it is the dense-neighbour limit (every frame
references its two perfect neighbours), not a periodic hidden-anchor
schedule. Spacing 4 is already 1.6 K short with perfect references; spacing
8/16/32 are 4.9-12.8 K short. The requested hidden-anchor spacings therefore
cannot clear 88.8 even if the anchors were free and perfect.

The real schedule is 8-9 K below its perfect-reference oracle at spacing 4
(78.3 vs 87.2) and 10 K below at spacing 8 (74.1 vs 83.9): the decoded
anchor's quality, not the schedule's syntax, is the binding constraint.

## 4. Why the rate-free ceiling is so low

The real spacing-4 stream at QP60 is 78,706 B / 314.8 kbps / K 79.50. That is
2.4x the rate of the flat key+P chain at QP60 (32,231 B / 128.9 kbps / K
79.67) for the same K. The alt-ref structure is not rate-starved at 100 kbps:
even with 3x the budget its K stays at the flat chain's plateau (~79-80),
because the anchors and leaves are still coded at a finite QP and the clip's
B/SS2 ceiling is set by per-picture coding efficiency, not by the reference
topology.

At 100 kbps the stream is a 41.8 dB PSNR-Y picture with BA 4.8 and SS2 75.8;
SVT's 88.8 point is 9.83 JOD / 1.51 BA / 88.30 SS2. The gap is the
per-picture fidelity (BA and SS2), not the frame schedule: the segment table
shows only +4.6..+7.6 K from making a third of the sequence lossless.

## 5. Confound check: nightly tool gating

The same-tool flat control in §2 (78.98) is below the nightly pin (82.155), so
the topology comparison had to be repeated under the **identical** nightly
tool subset. The alt-ref carrier rejects six nightly elements by name:

| nightly flag | rejection |
|---|---|
| `--video-key-qp-offset` (nightly sets −27) | `--video-key-qp-offset is wired for the flat key+P chain; drop the B-frame / reorder / temporal-layer / frame-type wrapper mode` |
| `--video-drift-refresh` | `--video-drift-refresh is wired for the flat key+P chain; drop the B-frame / reorder / temporal-layer / frame-type wrapper mode` |
| `--video-tile-group-compact` | `--video-tile-group-compact is gated on the plain key+P chain: the B/TIP/frame-type wrappers build their own sequence header` |
| `--video-ref-signal-compact` | `--video-ref-signal-compact is gated on the plain key+P chain` |
| `--video-bawp` + `--video-bawp-carrier` | `--video-bawp is gated on the plain single-prediction P chain: it cannot be combined with --b-frames, --video-tlayers, a frame-type wrapper, --video-compound or --video-gme/--video-warp (and --video-inter-residual needs --video-bawp-carrier)` |
| `--video-partition-rd` | `--video-partition-rd does not compose with the B/temporal-layer/random-access/TIP frame schedules` |

`--video-edge-inter` is also rejected without the open gate
(`--video-edge-inter is measured on the low-delay P chain; ...`); this branch
allows it when `AV2ENC_OPEN_GOP` is set. Everything else in `--nightly` (RDOQ,
CDEF/CDEF-RD, CCSO/CCSO-header-RD, deblock, TX search, key intra TX-part,
leaf RD, 1/8-pel, QP-scaled λ, the CDF chain, edge mask) encodes on the
alt-ref path.

A full-nightly-tool alt-ref is therefore **not legal**. The maximal legal
subset is `--nightly` with those six elements off (key QP −27 is applied to
both arms through the experiment env on the alt-ref side):

| arm (identical legal nightly subset) | QP | bytes | kbps | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|---:|---:|
| flat key+P | 109 | 25,049 | 100.196 | 9.6627 | 4.859 | 79.520 | **79.444** |
| alt-ref spacing 4 | 126 | 24,897 | 99.588 | 9.5190 | 5.591 | 72.785 | **76.789** |

Matched to 100.000 kbps the alt-ref is **76.764** and the same-tool flat is
**79.36**; the topology delta is **−2.68 K**. The full nightly pin is 82.155,
so the remaining **~2.8 K is composition**: the six rejected tools are worth
that much on the flat chain, and none of them is alt-ref-specific (drift is
single-reference, ref-signal-compact needs one slot, tile-group-compact is a
header packing, BAWP/partition are flat-chain-gated). A tool that helps only
the alt-ref by >2.7 K would be needed to reverse the sign; none is known.

This closes the topology question: under matched tools the hidden-anchor
schedule loses. The perfect-anchor oracle remains only a bound (spacing 4:
87.2, still below 88.8); it is not an arm.

## 6. Verdict and blockers

**This architecture does not clear SVT 88.8 or AVM 90.2.** The matched-rate
envelope is 78.34 (spacing 4), the rate-free ceiling of the same mechanism is
79.50, and the perfect-anchor bound at the requested spacings 8/16/32 is
83.9/76.1/76.0. Only the unattainable spacing-2 perfect-neighbour bound
(89.8) clears SVT, and it misses AVM.

Do not tune this architecture. The next lever is per-picture coding
efficiency (keyframe and inter residual/BA/SS2 at a fixed 25 kB), as
`docs/oracle-search.md`'s candidate queue already says; the reference
structure is a dead end for this clip.

Implementation blockers for the record (all after the negative result):

* hidden pictures need the SEF path; the existing tlayers planner assumed
  every coded picture is output and `validate_plan` rejected both output
  flags clear (now allowed only for the alt-ref builder);
* the SEF must run after the unit leaves and does not carry tile data, so the
  unit/stat accounting is one extra `frame_output_index = anchor` row per SEF;
* a 16-frame spacing needs 16 slots (15 leaves + 2 anchors, with the last
  leaf reusing the past-anchor slot); 32 does not fit and is rejected;
* `--rc vbr` still cannot allocate this schedule
  (`rate-control schedule does not cover every input picture`), so the QP
  offsets are global, not per-frame;
* the `--nightly` tool aggregate cannot compose with the temporal schedule
  (`--video-edge-inter`, `--video-rdoq`, `--video-partition-rd`,
  `--video-key-qp-offset`, BAWP and the drift refresh all reject it), so the
  comparison is against the same-tool flat control (79.0/79.6), not the
  nightly pin.

## 7. Acceptance

Gate off, release build (`cmake --preset release`, warnings + `-Werror`):

| pin | bytes | md5 |
|---|---:|---|
| default `--video --frames 60 --qp 189` | 25,009 | `a1d0dba1259b20a4b628d4cd1e14c33c` |
| quality `--qp 190 --video-rd-preset quality` | 25,141 | `3984677b9b665952984b07b7cf75c6ea` |
| nightly `--qp 110 --nightly` | 23,690 | `1b99127a4cd1d386fa5600fb63c57a83` |
| explicit-λ `--nightly --video-qp-lambda …` | 24,113 | `7f03eb6e82d6ba4cf6666c78285f945b` |
| 720p `space8k` QP217 nightly recipe | 36,572 | `c4ba28138d8f6569d5e08721a66d755c` |

`tools/test.sh --build build-release` is 10/10; the alt-ref streams decode
60/60 with `dav2d --strict 1` and `avmdec`, and the spacing-4 stream is
recon-exact. The gate is environment-only and default off.

## 8. Reproduction

```sh
REF=testdata/nasa_480x240_30fps_60f.y4m
BIN=build-dev/av2enc
TT="--video-inter-residual --video-ccso --video-ccso-rd --video-ccso-bands 6
  --video-ccso-lambda 22 --video-ccso-refit 2 --video-ccso-unit-bits 2
  --video-ccso-header-rd --video-tx-search --video-intra-full --video-rdoq
  --video-leaf-rd --video-key-intra-txpart --video-subpel-eighth
  --video-qp-lambda --avg-cdf --adaptive-cdf --cdf-continuity
  --video-edge-inter --video-edge-mask --video-deblock --video-cdef
  --video-cdef-rd --video-rdoq-scale 0.2"

# real hidden-anchor stream (spacing 4, matched point)
AV2ENC_OPEN_GOP=4 AV2ENC_OPEN_GOP_ALTREF=1 \
AV2ENC_OPEN_GOP_KEY_QP_OFFSET=35 "$BIN" "$REF" -o /tmp/ar4.ivf \
  --video --frames 60 --cpu-threads 8 --qp 130 --video-tlayers 8 $TT

# perfect-future-reference oracle (not decodable; score --video-recon)
AV2ENC_IDEAL_FUTURE_REF=8 "$BIN" "$REF" -o /tmp/oracle8.ivf \
  --video --frames 60 --cpu-threads 8 --qp 122 --video-refs 2 \
  --video-recon /tmp/oracle8.y4m $TT
python3 tools/quality.py --ref "$REF" --test /tmp/oracle8.y4m --json /tmp/q.json

# harness
python3 tools/oracle.py --ref "$REF" \
  --ivf altref4@129=/tmp/arK4_129.ivf --ivf altref4@130=/tmp/arK4_130.ivf \
  --ivf altref4@131=/tmp/arK4_131.ivf --target-kbps 100 \
  --json /tmp/hidden.oracle.json
```

The `AV2ENC_IDEAL_FUTURE_REF` oracle is compiled into the flat P path and is
off by default; it must never be used to produce a stream that is claimed
decodable.

## 9. Limits

* One clip, one rate, one host, one CPU build.
* The oracle's perfect reference is the *source* past/next anchor, so it
  overstates reference quality twice: no coding loss and no rate. It is the
  intended upper bound, not a candidate.
* The rate-free ceiling is the same mechanism at QP60 (314.8 kbps); a lower
  QP would cost more and moves K only inside the clip's ~80 plateau.
* The segment rows use the harness's fixed 60-frame ranges; a different
  segmentation would move individual dK values, not the verdict.

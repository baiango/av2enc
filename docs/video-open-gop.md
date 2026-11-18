# Structural alternatives at 240p30 / 100 kbps: tiles and a one-key open GOP

Status: **negative on both axes.** Multi-tile coding costs quality at matched
rate and does not buy speed on the CPU path. A one-key open-GOP hierarchical-P
tail — the structure `docs/bframes.md` listed as the missing test — is
**−3.1 to −7.5 K** against the flat key+P chain with the same tool set, and
**−5.7 K** against the nightly pin even in its best measured form. The flat
low-delay P chain is not what limits the encoder on this clip: it is the
cheapest structure per unit of quality because the clip's inter frames are
nearly redundant, and every reference schedule that spends rate on the tail
instead of the key loses.

Branch `exp/luma-structure`, from `main` at `a1159bb8`. Measurement build
`cmake --preset dev` (`-O1`, bitstreams identical to release). Host Apple M4,
`--cpu-threads 8`, `testdata/nasa_480x240_30fps_60f.y4m` (60 frames) unless
noted. Quality is `tools/quality.py` with the AVM `avmdec` oracle, plus
`tools/extra_metrics.py`; `K` is `tools/kpi.py`. One CVVDP job at a time.

## Verdict

| question | answer |
|---|---|
| do tiles improve quality at matched rate? | no. 2x1 −1.6 K, 2x2 −3.3 K, 4x2 −3.6 K at 240p; 2x1 −2.0 K at 720p. |
| do tiles buy speed? | no on this CPU path: 31.7 fps (1x1) → 24.5 (2x1) → 21.7 (2x2) → 21.1 (4x2) at the same QP, dev build, 60 frames, 8 threads. |
| can the current CLI express a one-key 60-frame open GOP? | no. `--video-tlayers` codes exactly one 4/8-frame closed GOP; `--b-frames/--video-gops` codes M closed GOPs; `--video-tip` is 3..8 frames; `--video-long-term` caps at 16. |
| does a one-key open-GOP hierarchical tail lift K? | no. Best measured 76.5 K vs 79.6 for the flat control with the same tools and the same key-QP offset; 82.16 is the nightly pin. |
| what is worth pursuing? | keyframe coding efficiency / allocation, not reference structure. In the flat arm the key carries 76-80% of the payload and the 59 inter frames average 82-97 B. |

## 1. Tile sweep

`--tiles` requires `--no-deblock`, rejects `--video-edge-inter`,
`--video-rdoq`, `--video-cdf-continuity`, `--video-ref-mvs`,
`--video-implicit-ref-map`, the segmentation tools and the frame-type wrappers.
The measured control is therefore a *tile-legal* tool set, run identically on
1x1 and on the tiled arms:

```sh
TT="--video-inter-residual --video-ccso --video-ccso-rd
  --video-ccso-bands 6 --video-ccso-lambda 22 --video-ccso-refit 2
  --video-ccso-unit-bits 2 --video-ccso-header-rd
  --video-tx-search --video-intra-full --video-leaf-rd
  --video-key-intra-txpart --video-subpel-eighth --video-qp-lambda
  --avg-cdf --adaptive-cdf --video-cdef --video-cdef-rd"
```

(`--video-deblock` is off by construction; `--cdf-continuity` is off because
video tiles + the carried CDF chain decode only 59/60 on dav2d and fail
outright on the 2x2/4x2 arms — a separate latent bug, not measured here.)

`4x1` is **not a legal layout**. The CLI accepts
`1x1/2x1/2x2/4x2/8x2/16x2` (or a total count), so the 4-column arm is 4x2
(eight tiles), not 4x1.

240p arms are interpolated linearly to 100.0 kbps; bracket points are listed
so the ripple can be checked.

| arm | QP: kbps / K | K@100 | Δ vs 1x1 | encode fps | decode fps |
|---|---|---|---|---:|---:|---:|
| 1x1 control | 187: 101.06 / 42.29 · 188: 97.98 / 41.69 | **42.08** | — | 31.7 | 189 |
| 2x1 | 189: 102.48 / 40.99 · 190: 99.51 / 40.41 | **40.51** | **−1.57** | 24.5 | 211 |
| 2x2 | 193: 100.62 / 38.86 · 194: 98.52 / 38.64 | **38.79** | **−3.29** | 21.7 | 202 |
| 4x2 | 193: 103.50 / 39.04 · 194: 100.17 / 38.48 | **38.51** | **−3.57** | 21.1 | 148 |

Raw quality points (JOD is the CVVDP value):

| arm | QP | bytes | kbps | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1x1 | 187 | 25,265 | 101.06 | 6.2819 | 20.17 | −4.16 | 42.29 |
| 1x1 | 188 | 24,495 | 97.98 | 6.2281 | 21.29 | −5.25 | 41.69 |
| 2x1 | 189 | 25,621 | 102.48 | 6.1430 | 21.51 | −6.69 | 40.99 |
| 2x1 | 190 | 24,877 | 99.51 | 6.0816 | 22.64 | −7.51 | 40.41 |
| 2x2 | 193 | 25,156 | 100.62 | 5.9077 | 22.43 | −11.75 | 38.86 |
| 2x2 | 194 | 24,630 | 98.52 | 5.8688 | 22.04 | −12.21 | 38.64 |
| 4x2 | 193 | 25,875 | 103.50 | 5.9112 | 21.02 | −11.82 | 39.04 |
| 4x2 | 194 | 25,043 | 100.17 | 5.8387 | 21.10 | −12.98 | 38.48 |

**Rate cost at the same QP** (QP 190, before rate matching): 1x1 23,273 B;
2x1 +6.9% (24,877); 2x2 +12.5% (26,168); 4x2 +17.7% (27,397). The matched-rate
2x2/4x2 arms run 3-4 QP higher, which is where the 3.3-3.6 K comes from.

720p (`space8k_1440x720_60f.y4m`, same TT set):

| arm | QP: kbps / K | K@100 | Δ |
|---|---|---|---:|
| 1x1 | 204: 101.5 / 28.55 · 205: 99.6 / 28.34 | **28.38** | — |
| 2x1 | 209: 101.0 / 26.72 · 210: 99.9 / 26.40 | **26.43** | **−1.95** |

The absolute 720p K is low because the tile-legal set has no edge inter,
rdoq or deblock at an operating point where the bottom superblock row is a
large fraction of 60 frames; the 2x1 delta is the measurement.

**Conclusion:** tiles are independent partitions with a per-tile CDF reset and
no shared prediction/loop-filter context. On this CPU path they also encode
slower. They are a parallelism/robustness feature, not an RD lever at 240p/
100 kbps or 720p.

## 2. What the CLI can express at 60 frames

Verified on this build with `--video --frames 60` (gate off):

| flag | 60-frame result |
|---|---|
| `--video-tlayers 4/8` | rejected: `codes one closed GOP: pass exactly N input frames; got 60` |
| `--video-tlayers 4/8` + exact N frames | works, N/ N decoded; only one closed GOP |
| `--video-tip` | rejected: `supports 3..8 input frames (got 60)` |
| `--video-gops` without `--b-frames` | rejected: `only applies to the --b-frames schedule` |
| `--b-frames 1/2/3/4 --video-gops 20/15/12/10` | legal; 20/15/12/10 **closed** GOPs, i.e. 20/15/12/10 key pictures |
| `--b-frames 6 --video-gops 7` | rejected at 60: `pass exactly 56 frames`; N=5 does not divide 60 |
| `--video-sef` | accepts 60 but codes 61 units / 60 output pictures; it is a leading frame-type wrapper, not a tail schedule |
| `--video-switch` | exactly 2 frames |
| `--video-ras` | exactly 3 frames |
| `--video-leading N` | at most 9 input frames |
| `--video-long-term` | `OrderHintBits = 4`, at most 16 coded frames |
| `--video-open-gop` | **does not exist** (new experimental env gate below) |

So the 60-frame structures the CLI cannot express today are:

* one key + 59 reordered inter pictures (the open-GOP / alt-ref tail);
* a hierarchical or temporal-layer tail longer than one 4/8-frame closed GOP;
* a periodic-intra-refresh or alt-ref/hidden-picture tail independent of the
  B pattern (`--video-lossless-intra-period` is block-level and lossless only);
* a per-frame QP/allocation plan for the temporal schedule: `--rc vbr` works
  on the closed 8-frame `--video-tlayers` stream but the open 60-frame stream
  fails with `--rc vbr: rate-control schedule does not cover every input
  picture`;
* long-term/OLK beyond 16 frames.

The only 60-frame "one key + long tail" the CLI could express before this
branch is the **flat P chain** itself — which is what the nightly already is.

## 3. One-key open-GOP prototype

### Design

`AV2ENC_OPEN_GOP=<unit>` (default off; `1` selects 8) turns the existing
`--video-tlayers 4/8` path into one key plus a long dyadic hierarchical-P
tail. With unit 8, a 60-frame sequence is seven 8-frame units plus one 4-frame
unit:

```
display  0 | 1 2 3 4 5 6 7 | 8 9 .. 15 | ... | 56 57 58 59
coding   0 | 4 2 6 1 3 5 7 | 8 12 10 14 9 11 13 15 | ...
layer    0 | 0 1 1 2 2 2 2 | 0 0  1  1 2  2  2  2 | ...
```

* only display 0 is a key (`refresh_frame_flags = 0xff`);
* each unit's first frame is a P anchor that references the previous unit's
  anchor; the rest of the unit is the same dyadic split as the closed
  hierarchical-P GOP, and every non-anchor references the previous
  coding-order frame (so the first leaves reference a future picture);
* unit anchors are `immediate_output_frame = 1`; that runs the §7.21.6
  `frame_buffers(-1, …)` flush of the previous unit's lower-hint implicit
  leaves before the refresh. This is the same mechanism the multi-GOP B
  schedule uses, and it makes an 8-slot store enough for 60 frames;
* the fixed slot map is anchor → slot 0, the unit's leaves → slots 1..7, so a
  leaf is never displaced before its unit anchor flushes it;
* `OrderHintBits` is the smallest width that spans the sequence (6 for 60
  frames); `NumRefFrames` is 8 for a 4/8-frame unit and 16 for a 16-frame unit.

An optional second reference policy (`AV2ENC_OPEN_GOP_REF=1`) makes each leaf
reference the nearest already-coded lower-display picture instead of the
previous coding-order frame. It measured the same (below), so the future
reference is not the issue; the tail's residual is.

`AV2ENC_OPEN_GOP_KEY_QP_OFFSET=<d>` applies the flat chain's key-QP offset to
the single key (an experiment-only allocation knob; `--video-key-qp-offset`
rejects this schedule).

### A real bug found on the way

The temporal-layer path stored each reference with `to_recon_frame()`, which
crops to the visible height. With `--video-edge-inter` the encoder's reference
plane must include the SB-aligned rows below the coded MI edge; the short
plane made the bottom superblock row predict from out-of-bounds samples. On a
4-frame unit at QP186 that turned 59/45-byte inter frames into 449/440-byte
frames. The path now stores the padded reference (rows below `MiRows*4`
replicated from the MI edge, exactly like the flat path's `pad_recon_bottom()`)
and crops only the output picture. The existing `--video-tlayers` gate still
passes 6/6 and its streams are unchanged for /8 heights.

`--video-edge-inter` is still rejected for `--video-tlayers` unless the open
gate is set; the permission is part of the experimental gate, not a product
change.

### Results at 240p / 100 kbps

Same `TT`-plus-network tool set for every arm (`TT` above plus
`--video-rdoq --video-deblock --video-cdf-continuity --video-edge-inter
--video-edge-mask`), 60 frames. `flat` is the existing one-key + 59-P chain.
`key −27` is `--video-key-qp-offset -27` on the flat arm and
`AV2ENC_OPEN_GOP_KEY_QP_OFFSET=27` on the open arms.

| arm | QP: kbps / K | K@100 | key bytes | inter mean | vs flat | vs 82.16 |
|---|---|---|---:|---:|---:|---:|
| flat | 85: 100.7 / 79.04 · 86: 96.5 / 79.03 | **79.04** | 18,686 | 97 B | — | −3.12 |
| flat, key −27 | 109: 100.6 / 79.44 · 110: 98.1 / 80.05 | **79.6** | 19,572 | 82 B | — | −2.6 |
| open unit 8 | 129: 100.4 / 71.32 · 130: 99.5 / 71.78 | **71.5** | 9,142 | 258 B | **−7.5** | −10.7 |
| open unit 8, key −27 | 139: 102.6 / 73.63 · 140: 99.2 / 74.15 | **74.0** | 12,327 | 199 B | **−5.6** | −8.2 |
| open unit 4 | 118: 99.2 / 74.36 · 119: 97.3 / 74.43 | **74.4** | 11,529 | 212 B | **−4.7** | −7.8 |
| open unit 4, key −27 | 129: 101.0 / 76.53 · 130: 97.1 / 76.49 | **76.5** | 14,473 | 170 B | **−3.1** | −5.7 |

Raw bracket points:

| arm | QP | bytes | kbps | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|---:|---:|
| flat | 85 | 25,178 | 100.70 | 9.6506 | 4.877 | 77.60 | 79.04 |
| flat | 86 | 24,121 | 96.50 | 9.6557 | 4.909 | 77.59 | 79.03 |
| flat key −27 | 108 | 25,621 | 102.48 | 9.6735 | 4.289 | 79.73 | 80.28 |
| flat key −27 | 109 | 25,155 | 100.62 | 9.6627 | 4.859 | 79.52 | 79.44 |
| flat key −27 | 110 | 24,523 | 98.09 | 9.6789 | 4.478 | 79.75 | 80.05 |
| flat key −27 | 111 | 24,689 | 98.76 | 9.6732 | 4.658 | 79.37 | 79.72 |
| open 8 | 128 | 25,678 | 102.71 | 9.1280 | 6.620 | 62.23 | 72.14 |
| open 8 | 129 | 25,090 | 100.36 | 9.0933 | 7.218 | 61.07 | 71.32 |
| open 8 | 130 | 24,883 | 99.53 | 9.0910 | 6.648 | 61.29 | 71.78 |
| open 8 | 131 | 25,175 | 100.70 | 9.0272 | 6.687 | 60.91 | 71.36 |
| open 8 key −27 | 139 | 25,641 | 102.56 | 9.3864 | 7.496 | 67.32 | 73.63 |
| open 8 key −27 | 140 | 24,791 | 99.16 | 9.3909 | 6.837 | 67.42 | 74.15 |
| open 8 key −27 | 141 | 24,177 | 96.71 | 9.3797 | 7.417 | 66.68 | 73.54 |
| open 4 | 118 | 24,800 | 99.20 | 9.3347 | 6.264 | 67.55 | 74.36 |
| open 4 | 119 | 24,337 | 97.35 | 9.3204 | 6.055 | 67.27 | 74.43 |
| open 4 key −27 | 129 | 25,245 | 100.98 | 9.5131 | 5.740 | 72.32 | 76.53 |
| open 4 key −27 | 130 | 24,278 | 97.11 | 9.5013 | 5.648 | 71.86 | 76.49 |
| open 4 key −27 | 131 | 24,140 | 96.56 | 9.4906 | 5.449 | 71.35 | 76.55 |

All bracket points decoded **60/60** with `dav2d --strict 1` and with the AVM
oracle; `--video-recon` equals the decoded Y4M byte for byte on the open
unit-4 + key-offset QP129 stream. `OG 4/8` names the unit size.

### Where the bytes go

At ~100 kbps the flat chain puts 76-80% of the payload in the key and codes
the 59 inter frames in 82-97 bytes each. The open schedules shift 7-10 kB
from the key into the tail, and lose 3-8 K doing so:

| arm | key share | inter mean | interpretation |
|---|---:|---:|---|
| flat | 76.5% | 97 B | key carries the picture; tail is skip |
| flat key −27 | 80.2% | 82 B | even more key-centric |
| open 8 | 37.6% | 258 B | tail residual, worse key at matched rate |
| open 8 key −27 | 51.3% | 199 B | better key, still 12.3 kB |
| open 4 | 47.9% | 212 B | shorter gaps, less residual |
| open 4 key −27 | 59.1% | 170 B | best open arm |

Encode cost (dev `-O1`, median of 3, 60 frames, 8 threads; compare only with
each other):

| arm | median | fps |
|---|---:|---:|
| flat QP85 | 5.36 s | 11.2 |
| open unit 4 QP118 | 3.88 s | 15.5 |
| open unit 8 QP129 | 3.63 s | 16.5 |

The open arms are faster because they run at a much higher QP; that is not a
recommendation.

### Why it loses

The clip's consecutive frames are nearly identical (a slow fade over a static
NASA image). A frame-to-frame P with skip costs 1 bit per block. A hierarchical
leaf that references a picture 2-8 frames away must code the fade difference,
and at 100 kbps the budget is 25 kB for two seconds; moving bytes from the key
into those residuals lowers JOD/BA/SS2 together (open 8: JOD 9.09, BA 7.2,
SS2 61 vs flat: JOD 9.65, BA 4.88, SS2 77.6).

The result is not caused by the future reference: the nearest-past-reference
policy (`AV2ENC_OPEN_GOP_REF=1`) measured the same K as the coding-order
reference on the broken-padding build (74.3 vs 74.4 at ~100 kbps, before the
padding fix), and after the fix both are dominated by the same gap cost.

## 4. Oracle bound vs the 90 wall

The strongest open-GOP point measured is **K 76.5** at 99-101 kbps. The flat
control with the same tools is **79.6**, the nightly pin is **82.16**, and the
wall is **90.0**. The structure is 13.5 K below the wall before any tuning,
and the comparison that isolates the structure (same tools, same key offset)
is −3.1 K. Extending the hierarchy (unit 16) is not available through the CLI
gate and would increase the gap: unit 8 is already 2.5 K below unit 4 with the
same allocation.

`docs/k-oracle.md`'s deep-anchor allocation ceiling (87.69) and the
perceptual-RD envelope (85.18) both assume the flat chain. Nothing in this
experiment suggests a reference schedule can raise those bounds; the open-GOP
envelope is below the flat control's achieved point, so there is no headroom
to chase.

## 5. Acceptance

Gate off, release build (`cmake --preset release`, warnings + `-Werror`):

| pin | bytes | md5 |
|---|---:|---|
| default `--video --frames 60 --qp 189` | 25,009 | `a1d0dba1259b20a4b628d4cd1e14c33c` |
| quality `--qp 190 --video-rd-preset quality` | 25,141 | `3984677b9b665952984b07b7cf75c6ea` |
| nightly `--qp 110 --nightly` | 23,690 | `1b99127a4cd1d386fa5600fb63c57a83` |
| explicit-λ `--nightly --video-qp-lambda …` | 24,113 | `7f03eb6e82d6ba4cf6666c78285f945b` |
| 720p `space8k` QP217 nightly recipe | 36,572 | `c4ba28138d8f6569d5e08721a66d755c` |

`tools/test.sh --build build-release` is 10/10. `gate_video_tlayer_refs.sh`
6/6 and `gate_avg_cdf.sh` pass. The open gate is env-only and default off; the
flat, B, TIP, SEF, long-term and SVC paths are untouched.

## 6. Reproduction

```sh
REF=testdata/nasa_480x240_30fps_60f.y4m
BIN=build-dev/av2enc
TT="--video-inter-residual --video-ccso --video-ccso-rd --video-ccso-bands 6
  --video-ccso-lambda 22 --video-ccso-refit 2 --video-ccso-unit-bits 2
  --video-ccso-header-rd --video-tx-search --video-intra-full --video-leaf-rd
  --video-key-intra-txpart --video-subpel-eighth --video-qp-lambda
  --avg-cdf --adaptive-cdf --cdf-continuity --video-edge-inter
  --video-edge-mask --video-deblock --video-cdef --video-cdef-rd"

# flat control
$BIN $REF -o /tmp/flat.ivf --video --frames 60 --cpu-threads 8 --qp 85 $TT
# one-key open GOP, 4-frame units, key QP -27
AV2ENC_OPEN_GOP=4 AV2ENC_OPEN_GOP_KEY_QP_OFFSET=27 \
  $BIN $REF -o /tmp/open4.ivf --video --frames 60 --cpu-threads 8 --qp 129 \
  --video-tlayers 8 $TT
# score
tools/bin/dav2d-macos --strict 1 -i /tmp/open4.ivf -o /tmp/dec.y4m \
  --muxer yuv4mpeg2
AVMDEC=tools/bin/avmdec python3 tools/quality.py --ref $REF --ivf /tmp/open4.ivf \
  --json /tmp/q.json
python3 tools/extra_metrics.py --ref $REF --test /tmp/dec.y4m --json /tmp/e.json
python3 tools/kpi.py --quality-json /tmp/q.json --extra-json /tmp/e.json
```

Tiles:

```sh
TT2="--video-inter-residual --video-ccso --video-ccso-rd --video-ccso-bands 6
  --video-ccso-lambda 22 --video-ccso-refit 2 --video-ccso-unit-bits 2
  --video-ccso-header-rd --video-tx-search --video-intra-full --video-leaf-rd
  --video-key-intra-txpart --video-subpel-eighth --video-qp-lambda
  --avg-cdf --adaptive-cdf --video-cdef --video-cdef-rd"
$BIN $REF -o /tmp/t1.ivf --video --frames 60 --cpu-threads 8 --qp 187 \
  --tiles 1x1 $TT2
$BIN $REF -o /tmp/t2.ivf --video --frames 60 --cpu-threads 8 --qp 189 \
  --tiles 2x1 $TT2
```

## 7. Limits

* One clip, one rate, one host; the NASA clip's fade is unusually friendly to
  frame-to-frame skip. A busier clip would make the flat chain's P frames more
  expensive, which narrows the absolute gap, but the direction in every
  measured arm is set by moving bits from the key to the tail.
* The open gate is experimental and env-only; no `--video-open-gop` flag, no
  per-frame allocator, and unit 16 is not reachable through the CLI check
  (`unit exceeds --video-tlayers`). Unit 4 and 8 are measured.
* The tilerate/tile-quality comparison is under the tile-legal tool set; the
  absolute K (42 at 240p) is far below nightly because edge inter, rdoq,
  deblock and the carried CDF chain are all unavailable with `--tiles`.
* The video-tile + `--cdf-continuity` failure (59/60 at 2x1, encoder error at
  2x2/4x2) is real and undocumented before this run; it is isolated, not
  fixed here.

## 8. Hidden future anchors (follow-up)

The open-GOP prototype was extended to a one-key hidden-future-anchor /
alt-ref schedule (`AV2ENC_OPEN_GOP_ALTREF=1`) with SEF-shown anchors and
two-sided past/future references. The go/no-go result is in
`docs/hidden-anchor-oracle.md`: matched-rate envelope **78.34** (spacing 4),
rate-free ceiling **79.50**, perfect-anchor oracle **83.9** (spacing 8) /
**87.2** (spacing 4), all below SVT 88.8 and AVM 90.2. Do not pursue this
architecture either.

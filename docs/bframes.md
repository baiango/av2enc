# B frames and closed-GOP structures at 240p30 / 100 kbps

`--fast-decode` is a flat key+P chain, so no B-frame tool had ever been scored at
the 240p / 100 kbps point. `--video-opfl`, `--video-refine-mv` and
`--video-compound-near-near` need the two-reference B carrier, `--video-compound`
needs two references that the one-reference nightly chain does not have, and
`--b-frames` itself is rejected by `--video-edge-inter`, which `--fast-decode`
enables. This doc builds a P control under the *same* tool set B can use,
rate-matches four closed-GOP shapes and the three refinement tools to 100 kbps,
and reports the gap to the 82.15 nightly pin and the 83.80 tuned-drift point.

**Verdict: clean negative.** The best B arm (N=4, 10 closed GOPs) interpolates
to **K 36.85** at 100.0 kbps: **−45.30 vs the 82.15 pin, −46.95 vs the 83.80
current best**, and **−7.83 against its own P control** (same flags minus
`--b-frames`/`--video-gops`, K 44.68). The structure spends 42-62% of the
payload on closed-GOP key pictures, so rate-matching it at 240p forces
QP197-204 and a 25-26 dB stream. `--video-compound`, `--video-opfl` and
`--video-refine-mv` are inert (+0.04 K), and `--video-compound` is not even
recon-exact on this clip (see below).

## The schedule the CLI can express

`--b-frames N --video-gops M` codes `M` **closed** GOPs of `N` B pictures plus a
key and a P anchor, so the frame count must be exactly `M*(N+2)`. Every GOP
starts with a CLK that resets all slots; a later CLK has to be
`immediate_output_frame = 1` because the pinned decoder does not flush slots the
refresh overwrites (`docs/negative-results.md`, "RESOLVED 2026-11-14"). The
result is not one sequence with periodic intra refresh: it is `M` independent
closed GOPs, i.e. `M` keyframes.

At 60 frames the legal shapes are:

| arm | `--b-frames` | `--video-gops` | GOP length | key pictures |
|---|---:|---:|---:|---:|
| b1 | 1 | 20 | 3 | 20 |
| b2 | 2 | 15 | 4 | 15 |
| b3 | 3 | 12 | 5 | 12 |
| b4 | 4 | 10 | 6 | 10 |

N=6 is the CLI maximum (gop 8) and needs 56 or 64 frames, N=5 (gop 7) does not
divide 60, so **N=4 / M=10 is the fewest-key 60-frame shape the CLI can
produce** and is included as the "best B arm" the refinement tools are added
to. `--video-refs` stays at its default 1: the B path owns the per-frame
reference list.

`--video-tlayers 4/8` cannot cover this clip: it codes exactly one closed GOP
and rejects any other input length
(`--video-tlayers codes one closed GOP: pass exactly 8 input frames; got 60`).
For the record, 4 and 8 input frames do encode and decode 4/4 and 8/8, but a
4- or 8-frame stream is not a 60-frame RD measurement, so it is not forced.

## Baseline construction log

`--fast-decode` cannot be combined with B frames directly, and neither can six of
the decisions it turns on. All of these are named rejections, verified on this
build:

| flag | rejection when combined with `--b-frames` |
|---|---|
| `--video-edge-inter` (nightly) | `--video-edge-inter is measured on the low-delay P chain; drop the B-frame / reorder / temporal-layer mode` |
| `--video-edge-mask` (nightly) | needs `--video-edge-inter`: `--video-edge-mask needs --video-edge-inter` |
| `--video-partition-rd` (nightly) | `--video-partition-rd does not compose with the B/temporal-layer/random-access/TIP frame schedules` |
| `--video-key-qp-offset -27` (nightly) | `--video-key-qp-offset is wired for the flat key+P chain; drop the B-frame / ... mode` |
| `--video-bawp --video-bawp-carrier` (nightly) | `--video-bawp is gated on the plain single-prediction P chain: ... cannot be combined with --b-frames ...` |
| `--video-drift-refresh` (nightly) | `--video-drift-refresh is wired for the flat key+P chain; drop the B-frame / ... mode` |
| `--video-ref-signal-compact` (nightly default) | `--video-ref-signal-compact is gated on the plain key+P chain` |
| `--video-tile-group-compact` (nightly default) | `--video-tile-group-compact is gated on the plain key+P chain: the B/TIP/frame-type wrappers build their own sequence header` |

The remaining nightly decisions are legal, and that set is the fair tool set
(`B-LEGAL` below). Note that `--video-rd-preset quality` expands to
`--video-inter-residual --video-deblock --video-ccso --video-tx-search
--video-intra-full`; dropping BAWP does not drop the inter residual, because
plain `--video-inter-residual` without the carrier is a different (legal) path.

```sh
REF=testdata/nasa_480x240_30fps_60f.y4m
BIN=build-dev/av2enc
B_LEGAL="--video-rd-preset quality
  --video-ccso-rd --video-ccso-bands 6 --video-ccso-lambda 22
  --video-ccso-refit 2 --video-ccso-unit-bits 2
  --video-rdoq --video-cdef --video-cdef-rd
  --video-leaf-rd --video-subpel-eighth --video-key-intra-txpart
  --video-qp-lambda
  --avg-cdf --adaptive-cdf --cdf-continuity
  --video-tile-flush-min --video-ccso-header-rd"
"$BIN" "$REF" -o /tmp/x.ivf --video --frames 60 --cpu-threads 8 --qp Q $B_LEGAL
# B arms append:  --b-frames N --video-gops M
```

The control ladder (the two nightly rows are absolute references, not
`B_LEGAL`-based; the other three are `B_LEGAL` plus the named flags):

| control | composition | QP | kbps | JOD | BA | SS2 | K |
|---|---|---:|---:|---:|---:|---:|---:|
| nightly pin (verified) | the real `--fast-decode` | 106 | 100.032 | 9.7334 | 3.420 | 80.84 | **82.15** |
| current best (verified) | `--fast-decode` + drift threshold 1.5 / budget 3250 | 110/111 | 100.584 / 99.800 | 9.7536 | 2.712 | 81.38 | **83.80** |
| **edge control** | `B_LEGAL` + `--video-edge-inter --video-edge-mask` | 86/87 | 100.916 / 98.700 | 9.6542 | 4.886 | 77.57 | **79.04** |
| three-lever control | `B_LEGAL` + `--video-key-qp-offset -27 --video-bawp --video-bawp-carrier` + tuned drift | 187/188/189 | 103.544 / 100.488 / 98.080 | 7.0450 | 19.073 | 15.11 | **49.48** |
| **fair P control** | `B_LEGAL` alone | 183/184/185 | 103.392 / 100.080 / 97.316 | 6.5122 | 20.359 | 3.45 | **44.68** |

Both pins reproduce exactly on this build, so the pipeline and the quality
tools agree with the earlier measurements.

The ladder shows where the 45 K gap comes from. The bottom-edge path alone is
worth **+34.36 K** at iso-rate under this tool set (`44.68 → 79.04`), and the
remaining B-incompatible levers add **+3.11 K** to reach the nightly pin. The
three per-frame levers that B also cannot use are worth +4.80 without the edge
path (`44.68 → 49.48`), which is why the ladder is not additive: the key QP
offset and the refresh mostly repay drift that the edge-path chain does not
accumulate in the first place.

**Why the no-edge controls sit at QP 183-190.** Without `--video-edge-inter`,
the bottom 64x64 superblock row of a 240-line picture has no inter path (the
coded extent runs past `MiRows*4`), so its nodes are intra-coded:
`--video-trace` shows all eight bottom-row superblocks as 64x64 intra leaves at
QP184, and a 64x64/64x32 intra mix at QP118, on a picture whose remaining
payload is inter. That bill grows as QP falls, which is why the rate-vs-QP curve
of every no-edge control is non-monotone (e.g. 57.4 kB at QP106, 79.3 kB at
QP142, 20.8 kB at QP190) and why it is the documented bottom-row failure mode
(`docs/negative-results.md`: the variance proxy rewards splits, "rate explodes
to 120.4 kbps for +0.02 JOD"). Both the P control and the B arms pay it
identically, so the comparison is apples to apples, but the absolute K of the
whole B-legal tool set is 37 K below the nightly pin before a single B picture
is coded: the same tool set *with* the edge path reaches 43.53 dB PSNR-Y at
100 kbps (K 79.04) while the no-edge control needs QP184 and lands at 28.04 dB
(K 44.68). This is a property of the tool-set tunnel B is forced into, not a
B-frame result.

## Arm table

Interpolated to 100.0 kbps (linear in kbps between the bracket points) with the
same quality pipeline as every other doc (`tools/quality.py` CVVDP 0.5.7
`standard_fhd` MPS, `tools/extra_metrics.py`, `tools/kpi.py`):

| arm | QP bracket | kbps | JOD | BA | SS2 | K | vs 82.15 | vs 83.80 | vs P control |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| **fair P control** | 183/184/185 | 103.4/100.1/97.3 | 6.5122 | 20.359 | 3.45 | **44.68** | −37.47 | −39.12 | — |
| b1 N=1 × 20 GOPs | 203/204 | 101.3/98.3 | 5.2833 | 25.458 | −28.36 | **32.66** | −49.49 | −51.14 | −12.02 |
| b2 N=2 × 15 GOPs | 200/201 | 104.4/98.3 | 5.4602 | 22.600 | −23.73 | **34.61** | −47.54 | −49.19 | −10.07 |
| b3 N=3 × 12 GOPs | 198/199 | 101.3/98.5 | 5.5898 | 21.272 | −20.70 | **35.92** | −46.23 | −47.88 | −8.76 |
| b4 N=4 × 10 GOPs | 197/198 | 102.3/97.0 | 5.6888 | 21.378 | −18.03 | **36.85** | −45.30 | −46.95 | −7.83 |
| b4 + `--video-compound` | 197/198 | 102.4/97.0 | 5.6858 | 21.339 | −17.69 | **36.89** | −45.26 | −46.91 | −7.79 |
| b4 + compound + near-near | 197/198 | 102.4/96.9 | 5.6870 | 21.388 | −17.70 | **36.89** | −45.26 | −46.91 | −7.79 |
| b4 + compound + OPFL | 197/198 | 102.4/97.1 | 5.6881 | 21.452 | −17.69 | **36.89** | −45.26 | −46.91 | −7.79 |
| b4 + compound + refine-MV | 197/198 | 102.4/96.9 | 5.6870 | 21.388 | −17.70 | **36.89** | −45.26 | −46.91 | −7.79 |
| b4 + compound + OPFL + refine-MV | 197/198 | 102.4/97.1 | 5.6881 | 21.452 | −17.69 | **36.89** | −45.26 | −46.91 | −7.79 |

Raw bracket points, so the interpolation can be re-checked (the P curve's
±1.6 K ripple is the reason every arm is bracketed rather than pinned at one
QP):

| arm | QP | bytes | kbps | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|---:|---:|
| P control | 183 | 25,848 | 103.392 | 6.5948 | 22.045 | 5.15 | 45.16 |
| P control | 184 | 25,020 | 100.080 | 6.5129 | 20.317 | 3.52 | 44.70 |
| P control | 185 | 24,329 | 97.316 | 6.4875 | 21.772 | 1.19 | 44.00 |
| b1 | 203 | 25,316 | 101.264 | 5.3044 | 25.283 | −27.81 | 32.87 |
| b1 | 204 | 24,570 | 98.280 | 5.2545 | 25.695 | −29.09 | 32.37 |
| b2 | 200 | 26,108 | 104.432 | 5.5265 | 21.885 | −21.76 | 35.35 |
| b2 | 201 | 24,572 | 98.288 | 5.4346 | 22.876 | −24.49 | 34.32 |
| b3 | 198 | 25,325 | 101.300 | 5.6147 | 21.030 | −20.30 | 36.14 |
| b3 | 199 | 24,622 | 98.488 | 5.5608 | 21.553 | −21.16 | 35.66 |
| b4 | 197 | 25,567 | 102.268 | 5.7399 | 21.189 | −16.92 | 37.31 |
| b4 | 198 | 24,249 | 96.996 | 5.6211 | 21.628 | −19.51 | 36.23 |
| b4 + compound | 197 | 25,600 | 102.400 | 5.7385 | 21.232 | −16.46 | 37.38 |
| b4 + compound | 198 | 24,250 | 97.000 | 5.6199 | 21.472 | −19.24 | 36.29 |
| b4 + compound + OPFL | 197 | 25,588 | 102.352 | 5.7417 | 21.360 | −16.45 | 37.38 |
| b4 + compound + OPFL | 198 | 24,274 | 97.096 | 5.6219 | 21.567 | −19.21 | 36.29 |

The bracket spans 2-5 kbps and K moves 0.5-1.1 K across it (slope
0.14-0.21 K/kbps). With two points per arm the ±1.6 K ripple of the P curve is
not resolved, so an interpolated B value could be off by up to ~1 K — still an
order of magnitude below the 7.8 K gap to the P control.

More B pictures per GOP helps only because it removes key pictures, not because
the B pictures get better: K rises monotonically 32.66 → 36.85 as the key count
falls 20 → 10. The direction of the whole table is set by key overhead.

## Where the bytes go

Per-unit payload sizes from the IVF records (coding order; the first unit of
each closed GOP is its key):

| arm | units | payload | keys | key share | inter unit (mean) |
|---|---:|---:|---:|---:|---:|
| b1 (20 keys) | 60 | 24,564 B | 15,121 B | 61.6% | 236 B |
| b2 (15 keys) | 60 | 25,356 B | 13,662 B | 53.9% | 260 B |
| b3 (12 keys) | 60 | 24,573 B | 11,464 B | 46.7% | 273 B |
| b4 (10 keys) | 60 | 24,815 B | 10,456 B | 42.1% | 287 B |
| P control (1 key) | 60 | 24,268 B | 2,045 B | 8.4% | 376 B |

At 100 kbps there are 25,000 bytes for two seconds. Ten keyframes take 10.5 kB
of that before a single B picture is coded, so every QP has to go up until the
key itself is a 25 dB picture. The stream PSNR-Y is **26.04 dB** for b4
(mean-of-frame PSNR 26.27 dB) against **28.04 dB** for the P control and
**44.67 dB** at the nightly pin. The first six b4 keys are 25.18-25.23 dB; the
fading second half's keys reach 28.69 dB as they also get cheaper, and the B
pictures track their GOP's key because they predict from it. There is no
allocation trick inside the schedule that fixes this: the key's quality is the
ceiling of its GOP.

Rate-matching also forces a QP the RD stack was never calibrated at: the B arms
live at QP197-203, where `--video-leaf-rd` and the QP-scaled lambdas are far
outside their measured band. That is not a separate excuse — it *is* the cost
of the structure at this rate.

## Compound / OPFL / refine-MV

All three land on the b4 arm, which is the best B arm. They are enumerated in
the table above; the result is that they do nothing at this operating point:

* payload moves by **+33 B** (`--video-compound`), **+30 B** (compound +
  near-near, and compound + near-near + refine-MV; refine-MV itself changes no
  byte count) and **+21 B** (compound + near-near + OPFL, with or without
  refine-MV) on a 25.6 kB stream;
* K moves by **+0.04** (36.85 → 36.89), i.e. inside the bracket's rounding;
* refine-MV produces the same byte count as the near-near stream at both QPs
  (same size, different symbols), so it truly never fires.

The tools do not become useful at a better operating point either. At
QP130/150/170 the same flags move the payload by only −0.24%, −0.09% and
−0.08% (compound + near-near; adding OPFL/refine-MV moves a further ≤0.03%),
so the B pictures are single-reference-dominated on this content at every QP.
Two-sided prediction is a syntax the clip does not ask for.

**`--video-compound` is not recon-exact on this clip.** Encoding with
`--video-recon` and comparing to `dav2d --strict 1` gives a mismatch for every
B schedule and QP tested (N=4/QP130, N=4/QP150, N=4/QP197, N=3/QP198,
N=2/QP200). At N=4/QP197 four pictures differ (display 31, 32, 37, 49 — all B
pictures), 1,571-2,205 luma samples each, max 8-14 LSB, first mismatch at luma
row 62 of display 31. The non-compound arms (b1-b4) are byte-exact in the same
check, so this is the compound path, not the B path. The scored numbers above
come from the decoder output (the correct side to score), but until this is
diagnosed the compound arms cannot be recommended regardless of RD, and the
`gate_video_opfl_refine.sh` pan fixture does not cover the case.

Reproducer:

```sh
$BIN $REF -o /tmp/c.ivf --video --frames 60 --cpu-threads 8 --qp 197 \
    $B_LEGAL --b-frames 4 --video-gops 10 --video-compound \
    --video-recon /tmp/c.y4m
tools/bin/dav2d-macos --strict 1 -i /tmp/c.ivf -o /tmp/c.dec.y4m \
    --muxer yuv4mpeg2
cmp /tmp/c.y4m /tmp/c.dec.y4m      # differs at byte 5,387,244
```

RESOLVED 2026-11-17 (`fix/compound-recon`, BUG-INTER-003). Both mismatches in
the compound carrier were encoder-side:

* **Recon drift with `--video-inter-residual`.** The leaf decision ran the
  inter-residual search on a compound winner and replaced `inter_y` with
  `pred + dequantised residual`, but §5.20.7.16 `write_compound_mode_info()`
  codes every compound leaf with `skip_flag = 1` (a non-skip compound leaf is
  rejected as "compound inter residual is not wired"). The decoder therefore
  reconstructs the plain blend while `--video-recon` carried the residual.
  Four B pictures drifted 1,571-2,205 luma samples, max 8-14 LSB, and P-frame
  carriers propagated the drift into the next picture. The residual search is
  now gated on `!best.compound`.
* **`--cdf-continuity` desync.** The fourteen compound CDF rows
  (`comp_mode`, `comp_ref0/1`, `is_joint`, `compound_mode_non_joint/same_refs`,
  `use_optflow`, `comp_group_idx`, `compound_type`, `cwp_idx`, the four wedge
  rows) adapted in the tile but were missing from the §8.2 packed image, so
  the encoder restarted at the §9.3 defaults while the decoder loaded the
  reference frame's adapted rows: 3/60 pictures decoded. They are now
  `cdf_arrays()` descriptors.

The non-compound B arms were byte-exact because they never select a compound
leaf. Post-fix, `--video-recon` equals the `dav2d --strict 1` payload for the
documented reproducer (QP 197, 25,592 B) and for every compound type on the
240p carrier; see `gate_video_near_near.sh` (residual arm) and
`gate_video_cdf_carry.sh` (compound CDF arm).

## Is the QP bracket unfair to B?

A global QP spends the same quantiser on a key and on a B picture. The obvious
counter-argument is that a per-frame allocator would spend less on keys and get
a better stream out of the same 25 kB. The encoder's allocator
(`--rc vbr --bitrate 100`) cannot be combined with `--video-qp-lambda`
(`--video-qp-lambda needs a fixed --qp stream`), and it requires
`--video-cdf-per-frame-qp`, which drops the carried CDF chain (2.9% rate) and
changes the tool set. Measured anyway, at anchor QP190:

| arm | bytes | kbps | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|---:|
| b4, `--rc vbr --bitrate 85` (100.2 kbps file rate) | 25,051 | 100.204 | 3.4612 | 36.29 | −56.38 | 19.21 |
| P control, `--rc vbr --bitrate 85` (98.9 kbps) | 24,722 | 98.888 | 5.3328 | 35.09 | −23.20 | 33.10 |

The allocator makes *both* schedules worse than the fixed-QP bracket (P control
44.68 → 33.10) and hurts the B arm far more (−17.64 K against its own 36.85 CQP
point), because its per-frame model is calibrated on the flat chain: it starves
the pictures that the closed GOP makes most expensive. This is a diagnostic,
not a fair arm — it is listed with its own CDF/λ caveat — but it says the B
result is not an artefact of one global QP.

## Encode cost

Median of three wall-clock encodes, 60 frames, `--cpu-threads 8`, Apple M4,
**dev preset (-O1)**, so compare the rows only with each other (the release
build is the one to quote in absolute terms):

| configuration | median | fps | vs fair P |
|---|---:|---:|---:|
| nightly QP106 | 8.49 s | 7.1 | — |
| fair P control QP184 | 6.15 s | 9.8 | 1.00× |
| b4 QP197 | 7.20 s | 8.3 | 1.17× |
| b4 + compound + near-near + OPFL + refine-MV QP197 | 7.16 s | 8.4 | 1.16× |

The B schedule costs ~17% more than the same tool set on a P chain even though
it runs 13 QP higher (less residual work). The refinement tools are free at
this operating point because they never fire.

## Composition with the tuned drift refresh

It does not compose, and not because of a measurement subtlety: the two modes
are mutually exclusive by name.

```text
$ av2enc … --b-frames 4 --video-gops 10 --video-drift-refresh
av2enc: --video-drift-refresh is wired for the flat key+P chain; drop the
        B-frame / reorder / temporal-layer / TIP / frame-type wrapper mode
```

`--video-compound` is rejected by the refresh too
(`--video-drift-refresh is wired for single-reference inter prediction`), so
even the compound carrier cannot carry it. The reason is structural: the
refresh is a frame-level trigger that measures the *primary* reference's
reconstruction and forces the inter residual on that frame. A B schedule has
two references per picture, codes in a different order than it displays, and
owns its own per-frame QP plan through the schedule writer; "the reference"
that the trigger needs does not exist. The tuned operating point (threshold 1.5,
budget 3250) is therefore unavailable to every arm in this doc, and the 83.80
point is not a baseline that B can be compared against on equal terms — only
the 82.15 pin and the B-legal controls are.

## Verdict

* **Against the nightly pin (82.15):** best B arm 36.85, **−45.30 K**.
* **Against the current best (83.80):** **−46.95 K**.
* **Against the fair P control under identical flags (44.68):** **−7.83 K**.
  This is the number that isolates B frames: same tools, same rate, same clip;
  the only difference is the closed-GOP schedule. It loses.
* The rest of the gap is the tool-set tunnel: `--video-edge-inter` (+34.36 K)
  and the other B-incompatible nightly decisions (+3.11 K) are not available
  to any B stream, so a B arm starts 37 K behind before it codes a B picture.
* `--video-compound`, `--video-compound-near-near`, `--video-opfl` and
  `--video-refine-mv` are inert here (+0.04 K, ≤33 B) and compound is not
  recon-exact, so there is no partial credit to bank.
* The structure cannot be rate-matched honestly at this point: 42-62% of a
  25 kB budget goes to 10-20 key pictures, which forces QP197-204 and a 25-26 dB
  stream. This confirms the existing `docs/negative-results.md` row ("Expect a
  non-causal reference structure to beat the flat P chain") at the 240p /
  100 kbps operating point, with the fair control it was missing.

What would change the answer is not a B tool but a different *structure*: an
open GOP with one key plus a long intra-refresh tail, or an intra period the
CLI can set independently of the B pattern. `--b-frames --video-gops` can
express neither, and adding a per-frame allocator to the schedule would have to
come before any of these tools are worth re-measuring.

## Caveats

* One clip (`nasa_480x240_30fps_60f.y4m`), one host (Apple M4), CPU-only, one
  operating point. The clip's second half fades and flattens, which is why the
  last keys of each schedule are 3-6x cheaper than the first; a busier clip
  would make the key overhead worse, not better.
* The 60-frame/closed-GOP constraint is a CLI property, not a codec limit. The
  measured K is the K of *this schedule shape*, not of B prediction in
  general.
* The P control is itself handicapped by the missing bottom-edge path; its
  QP-rate curve is non-monotone for that reason. B is compared against it
  because both sides pay the same bill — the absolute 44.68 is not a fair
  statement of what a P chain can do at 240p.
* The QP bracket only has two points per B arm, so the ±1.6 K ripple of the P
  curve is not resolved for them; the interpolation is good to ~1 K (the
  measured K change across each bracket), far below the 7.8 K gap to the P
  control.
* Encode times are from the dev (-O1) build and are relative.

## Verification

* `tools/validate_dav2d.sh` passes on the b4 and compound streams (60/60
  decoded, PSNR-Y 26.0347 and 26.0459 matching the measurement).
* Every point decoded **60/60** with `tools/bin/dav2d-macos --strict 1`; the
  B streams are written in coding order and emitted in display order
  (`monotonic_output_order_flag = 0`, implicit output), which the smooth
  per-frame PSNR confirms.
* Non-compound recon: `--video-recon` equals the decoded Y4M byte for byte for
  b1 (QP203), b2 (QP200), b3 (QP198) and b4 (QP197).
* Compound recon: **differs** at N=2/3/4 and QP130/150/197/198/200 — see the
  reproducer above.
* The 82.15 pin (25,008 B / 100.032 kbps / 9.7334 JOD / 3.4202 BA / 80.837 SS2)
  and the tuned-drift points (25,146 B / 83.94 K at QP110, 24,950 B / 83.75 K
  at QP111) reproduce exactly, so the metric pipeline matches the earlier
  docs.
* Key counts were verified from the per-unit IVF sizes
  (20/15/12/10 units of ~0.9-1.2 kB at the head of each GOP), not assumed from
  the flag values.

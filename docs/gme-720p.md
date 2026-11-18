# Global motion and warped prediction at 720p (`--video-gme`, `--video-warp`)

**Verdict: negative. Stop before phase 2.**

`--video-gme` is the only untried prediction tool aimed at the 720p failure mode,
where the composite K is JOD-dominated (36.6 of 45.6 at the pin). It does not
survive phase 1 for three independent reasons:

1. **Its warp path is not decoder-exact at this operating point.** `dav2d
   --strict 1` decodes 60/60 pictures, but its reconstruction diverges from the
   encoder's from frame 4 onward (49.2M differing bytes of a 93.3M payload, 56
   of 60 frames). The stream below that point measures **5.50 JOD** at the
   pinned 36,572 B, i.e. −1.77 JOD vs the fair baseline and −1.82 JOD vs the
   pin. This is the **escape-hatch** path (`AV2ENC_GME_UNPROVEN_WARP=1`); the
   default is translation/identity-only and does not take the warped predictor
   (see "Fail-safe default" below).
2. **Even the encoder's own reconstruction is not a win worth a source
   project.** It measures 7.3185 JOD at 36,572 B: +0.043 JOD / +0.51 K over the
   non-GME baseline the lock forces, but −0.005 JOD / −0.75 K against the
   existing 720p pin it would have to beat.
3. **The 240p check is an exact no-op.** The fit degrades to IDENTITY on
   `nasa_480x240`; the decoded pictures are MD5-identical to the baseline at
   both measured QPs, and the stream grows by 1–4 header bytes
   (QP107/109/101/102 all checked).

## Fail-safe default (translation/identity only)

The warp path is no longer reachable by default. `make_gm_frame_plan()`
(`src/pipeline/video/video_gme.cpp`) inspects every accepted model and, unless
the escape hatch is set, replaces a model whose linear part is not the identity
(zoom, rotation, shear, general affine) with its **translation-only
projection**: `p[0]/p[1]` from the fit and an identity linear part, signalled as
ROTZOOM. The header therefore never carries a model that applies warp geometry.
GLOBALMV uses the projected §7.12.2.2 centre MV, which is constant for a
translation model, and both sides take the same rounding path the existing pan
fixture already gates. A purely translational fit is emitted unchanged; a fit
that collapses to zero translation becomes IDENTITY. `use_global_motion` is set
only from the post-fallback model, so an affine-only fit with zero translation
degrades to the historical no-GME header.

`AV2ENC_GME_UNPROVEN_WARP=1` restores the previous behaviour exactly: the
fitted model (including its linear part) is signalled and the §7.13.3.19–.24
predictor runs. That path remains **conformance-unproven**; every measurement
in this document that uses a non-identity model was taken with it.

Regression gate: `src/pipeline/video/video_ref_mvs_gate.sh` generates a
4-picture 640×360 pure-zoom clip (1.01× per frame, no external media), encodes
it with `--video-gme --video-warp`, decodes with `dav2d --strict 1`, and
compares the decoded payload with `--video-recon` byte for byte on every frame.
On the pre-fix tree that stream differed in **120,823 of 1,382,400 payload
bytes** across frames 1–3 (max 35); with the fallback it is **0**. The gate also
encodes the same clip with `AV2ENC_GME_UNPROVEN_WARP=1` and requires a different
stream and at least one warped candidate, so the default arm cannot pass on a
clip whose fit went purely translational. The existing translation-model GME
cases (`gme64`, `gme360`, `gme64_refs3`, `gme360_refs3`, `warp360`) are
byte-identical to the pre-fix tree.

The real 720p fair-baseline configuration (8-frame prefix, QP215) shows the same
repair at scale: the pre-fix warp stream differed in **1,235,050 of 12,441,600
payload bytes** (frames 4–7); the default fallback differs in **0**. The full
60-frame 49.2M-byte measurement below was taken on the escape-hatch path before
this change.

## The 720p point

Pin (reproduced byte-exactly in this worktree):

```sh
build/av2enc testdata/space8k_1440x720_60f.y4m -o /tmp/pin.ivf \
  --video --frames 60 --cpu-threads 8 --qp 217 --nightly \
  --video-key-qp-offset -57 --video-leaf-rd-lambda 250 \
  --video-ext-partitions
```

36,572 B, MD5 `c4ba2813…`, JOD 7.3236, BA 28.8713, SS2 −10.146, K 45.61
(JOD 36.62 + BA 2.35 + SS2 6.64). K is JOD-dominated as expected; every
judgement below is made on JOD first.

## Baseline construction (the lock)

`--video-gme`/`--video-warp` are rejected with `--video-inter-residual`
(`src/cli/validate.cpp:1712-1726`) and `--video-bawp` rejects GME
(`:1743-1771`). `--nightly` enables inter-residual through the `quality` preset
(`src/cli/nightly.cpp:32` → `apply_video_rd_preset`) and BAWP/BAWP-carrier
directly (`:169-170`). `--video-rdoq` in turn requires `--video-inter-residual`
(`src/cli/validate.cpp:1657-1662`). The minimum set that unblocks GME is
therefore four flags:

```text
--video-inter-residual  --video-rdoq  --video-bawp  --video-bawp-carrier
```

Everything else in the pin stays — including `--video-edge-inter`,
`--video-edge-mask`, `--video-partition-rd` (with the QP-scaled probe lambda
nightly resolves), `--video-key-intra-txpart`, the CDF carry and the overhead
wave. The fair baseline is:

```sh
build/av2enc testdata/space8k_1440x720_60f.y4m -o /tmp/base.ivf \
  --video --frames 60 --cpu-threads 8 --qp 217 \
  --video-ccso-rd --video-edge-inter --video-partition-rd \
  --video-key-qp-offset -57 --video-edge-mask \
  --video-cdef --video-cdef-rd --video-qp-lambda \
  --video-tile-flush-min --video-ref-signal-compact \
  --video-tile-group-compact --video-ccso-header-rd \
  --video-ccso-bands 6 --video-ccso-lambda 22 \
  --video-ccso-refit 2 --video-ccso-unit-bits 2 \
  --video-leaf-rd --video-subpel-eighth --video-key-intra-txpart \
  --avg-cdf --adaptive-cdf --cdf-continuity \
  --video-drift-refresh \
  --video-leaf-rd-lambda 250 --video-ext-partitions \
  --video-deblock --video-ccso --video-tx-search --video-intra-full
```

Control: adding the four removed flags back reproduces the pin byte for byte
(36,572 B, MD5 `c4ba2813…`), so no other flag drifted in the reconstruction.

`--video-drift-refresh` stays for flag parity but is inert without the residual
path: the baseline with and without it is byte-identical (`df5e5a68…`), although
the encoder log still reports "drift refresh frames".

## 720p arms

Host: Apple M4, `-j2` dev build, `--cpu-threads 8`. Encode 50–55 s per 60-frame
stream, CVVDP ~20 s, Butteraugli/SSIMULACRA2 ~60 s. `--video-gme`,
`--video-warp` and `--video-gme --video-warp` are byte-identical at every QP
tested (MD5 `ba14a33d…` at QP215): `--video-warp` is an alias, and
`global_mv_for_block()` already forces the warped predictor from the model, so
the flag adds nothing. The three requested arms are one bitstream.

Two QPs per arm bracket the 36,572 B target:

| Arm | QP | Bytes | JOD | BA | SS2 | K | Note |
|---|---:|---:|---:|---:|---:|---:|---|
| pin (reference) | 217 | 36,572 | 7.3236 | 28.8713 | −10.146 | 45.61 | `c4ba2813…` |
| baseline | 217 | 35,803 | 7.2379 | 33.4267 | −15.484 | 44.00 | conformant |
| baseline | 216 | 36,729 | 7.2830 | 34.6064 | −14.016 | 44.41 | conformant |
| **baseline @36,572** | — | **36,572** | **7.275** | ~34.41 | ~−14.26 | **44.34** | log-rate interpolation |
| GME = warp = both | 217 | 35,346 | — | — | — | — | encoded, not scored |
| GME = warp = both | 216 | 36,369 | 6.0385 | — | — | — | decoded |
| GME = warp = both | 215 | 36,587 | 5.4613 | 79.9165 | −47.902 | 28.56 | decoded |
| GME, encoder recon | 216 | 36,369 | 7.3102 | 33.1913 | −13.058 | 44.78 | diagnostic |
| GME, encoder recon | 215 | 36,587 | 7.3191 | 32.8349 | −12.993 | 44.86 | diagnostic |
| **GME @36,572, encoder recon** | — | **36,572** | **7.3185** | ~32.86 | ~−13.00 | **44.85** | best case, not decodable |

At a fixed QP GME is ~1% smaller (e.g. 36,587 vs 36,942 at QP215), worth about
one QP of rate: to match 36,572 B the GME arm runs at QP215 against the
baseline's QP216.

Matched-rate deltas:

| Comparison | ΔJOD | ΔK |
|---|---:|---:|
| GME (encoder recon) vs fair baseline | **+0.043** | +0.51 |
| GME (encoder recon) vs pin | −0.005 | −0.75 |
| GME (as decoded) vs fair baseline | **−1.77** | −15.6 |

## The decisive finding: the GME warp stream is not decoder-exact

Every row in this section is the **escape-hatch** path
(`AV2ENC_GME_UNPROVEN_WARP=1`); it is why the default no longer emits a warp
model. Every GME stream was decoded with `dav2d --strict 1` and compared with
the encoder's `--video-recon` payload:

| QP | Bytes | dav2d strict | Recon vs decoded | Frames affected |
|---:|---:|---|---|---:|
| 215 | 36,587 | 60/60 | 49,180,899 differing bytes | 4–59 (56/60) |
| 216 | 36,369 | 60/60 | 47,937,443 differing bytes | 4–59 (56/60) |
| baseline 216 | 36,729 | 60/60 | 0 differing bytes | — |

The frames before the mismatch are the four QP160 reference-intro pictures;
the divergence starts at the first normal-QP P frame and compounds because the
decoder uses its own reconstruction as the next reference. That is pure drift,
not a rate trade: the decoded JOD is non-monotonic in QP (6.04 at QP216 vs 5.46
at QP215 — the higher-rate arm drifts further because more `GLOBALMV` leaves
survive).

`tools/validate_dav2d.sh` passes on the GME stream (`nb_read_frames=60`,
`OK: dav2d decoded`): the stream is *syntactically* valid, and only the
recon-payload comparison exposes the predictor disagreement. Do not treat a
green decode as conformance for this path.

`--video-trace` shows why the path is live: in the failing configuration the
projected global MVs are non-zero with a linear gradient across the frame
(`(-3,30), (-4,27), … (-9,12)`), and 298 block candidates evaluate with
`warp=1`. The fitted model therefore has a real linear part and the encoder
takes the §7.13.3.19/.20 warped predictor on both sides — the disagreement is
inside warped prediction/geometry, not a missing header bit or a translational
centre-MV rounding difference. This is the documented open issue
(`docs/negative-results.md:151`: any model that survives the 1/64 linear
quantisation mismatches; `--video-gme` warp prediction is not
conformance-proven), now confirmed on real 720p content at the operating point.

The failure has a second, narrower manifestation. With only the filter group
and the overhead wave enabled (`--video-ccso-rd --video-partition-rd
--video-cdef --video-cdef-rd --video-deblock --video-ccso --video-tx-search
--video-intra-full --video-tile-flush-min --video-ref-signal-compact
--video-tile-group-compact --video-ccso-header-rd` plus the CCSO knobs, no
reference-intro offset, no QP-scaled lambdas, no leaf RD, no extended tree) the
disagreement is confined to the bottom edge: 3,003 bytes, luma rows 635–719
only (6-frame check). Adding the QP160 reference-intro offset and the measured
partition/leaf RD widens it to the whole frame. `--video-edge-inter` is
therefore implicated in the narrow form but is not required: removing it, or
removing CDEF, still leaves the whole-frame variant. A conformant configuration
exists only far from the pin's tool set
(`--video-key-qp-offset -57 --video-qp-lambda --video-leaf-rd --video-deblock
--video-ccso --video-tx-search --video-intra-full`, 6-frame check: 0 differing
bytes), which is too weak a baseline to extrapolate from.

Because the decoder is the conformance oracle everywhere else in this project,
the as-decoded numbers are the only valid 720p JOD measurement. They are a large
loss, and they do not represent GME's coding efficiency — they represent a
predictor the decoder does not reproduce. The encoder-recon row is quoted only
to bound the ceiling.

## 240p check

`testdata/nasa_480x240_30fps_60f.y4m`, 60 frames, fair baseline flag set with
the 240p key offset (−27) and no `--video-ext-partitions`:

| Arm | QP | Bytes | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|---:|
| nightly pin | 109 | 23,997 | 9.7270 | 3.3547 | 80.593 | 82.20 |
| fair baseline | 109 | 21,718 | 9.6752 | 4.6046 | 79.250 | 79.78 |
| **GME** | 109 | 21,720 | 9.6752 | 4.6046 | 79.250 | 79.78 |
| fair baseline | 101 | 24,192 | 9.6769 | 4.7050 | 79.635 | 79.72 |
| **GME** | 101 | 24,196 | 9.6769 | 4.7050 | 79.635 | 79.72 |

QP101 matches the pin's 24.0k. The GME decoded picture is MD5-identical to the
baseline at both QPs (`e956a585…` at QP109, `d53b5856…` at QP101): on this
slow orbital clip the fit degrades to IDENTITY (every `GLOBALMV` candidate in
the trace has MV (0,0); no `warp=1` candidate), so the only difference is the
`enable_global_motion` header bits. GME is inert here, not negative — and the
clip cannot exercise the tool.

For reference, at the same ~24.0k the nightly pin is +0.05 JOD and +2.5 K over
the residual-free baseline (the BA/SS2 components, 3.35 vs 4.71 and 80.59 vs
79.64, are what the inter-residual/rdoq/BAWP chain buys at 240p). GME does not
recover any of that.

## Why no phase 2

Phase 2 was gated on a phase-1 JOD gain. The gate is not met:

- The only decoder-valid 720p measurement is **−1.77 JOD** at matched rate.
- The best case the encoder can even aim at is **+0.043 JOD / +0.51 K** over a
  baseline that exists only because the lock removed three measured tools, and
  **−0.005 JOD / −0.75 K** against the pin that would actually be shipped.
- The 240p content does not fit a non-identity model, so there is no second
  data point to suggest the gain grows elsewhere.

A phase-2 project would have to fix warped-prediction conformance first (open,
unscoped, and currently documented as out of scope) and then wire the warped
predictor into the residual/decision path, to chase at most ~0.04 JOD that the
existing pin already achieves. That is not a defensible source project.

## Reproduction

Artifacts (dev build, 8 threads). Reproduce every GME/warp round trip below
with `AV2ENC_GME_UNPROVEN_WARP=1`; without it the encoder emits the
translation-only projection and the stream is decoder-exact but is no longer
the measured warp arm.

| File | Bytes | MD5 |
|---|---:|---|
| pin QP217 | 36,572 | `c4ba28138d8f6569d5e08721a66d755c` |
| baseline QP215 / 216 / 217 / 218 | 36,942 / 36,729 / 35,803 / 35,147 | `df665359…` / `e6276aa9…` / `df5e5a68…` / `dd895dd4…` |
| GME = warp = both QP215 | 36,587 | `ba14a33db89c149a6a3660108ac6cba5` |
| GME QP216 / 217 / 218 | 36,369 / 35,346 / 34,872 | `3367ef80…` / `8b924a9b…` / `f8d62030…` |
| 240p pin QP109 | 23,997 | `ade89c68…` |
| 240p baseline / GME QP101 | 24,192 / 24,196 | `5787fbf7…` / `7ecb1bb6…` |
| 240p baseline / GME QP109 | 21,718 / 21,720 | `98bffb1c…` / `2b05f4b6…` |

Scoring per stream:

```sh
tools/bin/dav2d-macos --strict 1 -i x.ivf -o x.y4m --muxer yuv4mpeg2
DAV2D=tools/bin/dav2d-macos python3 tools/quality.py \
    --ref testdata/space8k_1440x720_60f.y4m --ivf x.ivf --json q.json
python3 tools/extra_metrics.py --ref testdata/space8k_1440x720_60f.y4m \
    --test x.y4m --json e.json
python3 tools/kpi.py --quality-json q.json --extra-json e.json
```

The encoder-recon diagnostic scores the `--video-recon` Y4M directly with
`tools/quality.py --test x.recon.y4m`. Conformance is the payload comparison in
`tools/gate_video_aq.sh` (`decode_and_compare`): strip the Y4M header and
`FRAME\n` markers from both files and compare bytes.

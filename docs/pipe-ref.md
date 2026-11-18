# Pipeline reference allocation (`AV2ENC_PIPE_REF`)

Status: **negative / ceiling record. Do not merge.** The corrected allocator's
best matched-rate point is **K 83.83**, inside the ±0.03–0.1 ripple of the old
drift wall (83.80) and **6.17 K below the new target of 90** (SVT-AV1 p0
88.83, project best 84.81). The branch closes as a measured ceiling, not a
candidate.

The design was the missing half of the reference experiments in
`docs/reference-structure.md`: those tested extra references and compound as
per-block add-ons inside a flat single-reference pipeline and lost. This branch
builds the frame-level allocation and the per-block prediction together, with
the second reference priced rather than free.

Gate: `AV2ENC_PIPE_REF=1` (env only, default off, no CLI flag). With the gate
unset every pinned stream is byte-identical.

## K oracle

Per `docs/k-oracle.md` (protocol on `exp/rate-composite`), a pipeline must
announce its bound before reporting a tuned result.

| Bound | Value at 240p / 100 kbps | Method |
|---|---:|---|
| **Achievable envelope** | **K 83.83** | best linear interpolation to 100.0 kbps over the coarse allocator-λ × recent-primary sweep below |
| Allocation bound | **unwritten** | would need per-frame JOD/BA/SS2 curves + knapsack; see below |
| Wall (old drift recipe) | 83.795 | QP109–112 interpolation in `docs/reference-structure.md`, reproduced here |
| New target | 90 | branch instruction 2026-11-18 |

**Can the bound exceed the old wall?** The measured envelope is +0.038 K above
83.795, i.e. a tie inside the QP ripple (±0.03–0.1 K), and 6.17 K below 90.
Per the oracle rule this is a **below-target ceiling**: no further tuning was
spent after the corrected envelope was known.

**Why the allocation bound is unwritten.** `tools/quality.py` reports only the
pooled CVVDP for the whole sequence; `tools/extra_metrics.py` averages
Butteraugli/SSIMULACRA2. There is no per-frame JOD/BA/SS2 producer in the tree,
so a per-frame convex-hull/knapsack bound requires a new harness (one CVVDP job
per frame per QP plus per-frame BA/SS2 on at least three QPs, then an assembled
stream re-measure). That is not cheap, and a bound can only tighten a ceiling
that is already 6.17 K under the target. PSNR alone is not a K bound and was not
substituted. The envelope is the number this branch stands on.

## What is measured

Host: Apple M4, CPU reference build (`cmake --preset dev`, `build-dev/av2enc`,
8 threads), `testdata/nasa_480x240_30fps_60f.y4m`, 60 frames, 240p30.
Quality: `tools/quality.py` (CVVDP 0.5.7, `standard_fhd`, MPS) on the decoded
stream, `tools/extra_metrics.py` (Butteraugli, SSIMULACRA2), `tools/kpi.py`.
`kbps = IVF bytes * 8 / 2 s / 1000`. Every quality point decoded 60/60 under
`dav2d --strict 1`.

The drift recipe is the old wall's:
`--video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250`.

### Corrected free allocator and λ sweep

`MINREFS` defaults to 1 (the first sweep accidentally defaulted it to
`MAXREFS`; those forced-3 points are discarded). `λ` is
`AV2ENC_PIPE_REF_LAMBDA` times the leaf SSE-per-bit lambda at the frame
qindex. `recent-primary` keeps the newest decoded slot as the primary and
allocates only additional references around it.

| Arm | QP | Bytes | kbps | JOD | BA | SS2 | K | K@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| λ0.5 | 111 | 25,365 | 101.460 | 9.7080 | 2.6472 | 81.319 | 83.71 | **83.58** |
| λ0.5 | 112 | 24,978 | 99.912 | 9.7225 | 2.7460 | 81.452 | 83.57 | |
| **λ1 (free)** | 111 | 25,239 | 100.956 | 9.7309 | 2.6494 | 81.432 | 83.84 | **83.76** |
| λ1 (free) | 112 | 24,892 | 99.568 | 9.7266 | 2.6757 | 81.307 | 83.73 | |
| λ2 | 111 | 25,221 | 100.884 | 9.7283 | 2.7044 | 81.437 | 83.70 | **83.75** |
| λ2 | 112 | 24,933 | 99.732 | 9.7210 | 2.6468 | 81.264 | 83.76 | |
| **λ1 + recent primary** | 111 | 25,147 | 100.588 | 9.7119 | 2.6878 | 81.313 | 83.63 | **83.83** |
| λ1 + recent primary | 112 | 24,915 | 99.660 | 9.7287 | 2.5981 | 81.429 | 83.95 | |

K-vs-λ at matched rate for the unconstrained primary:

| λ | K@100 |
|---:|---:|
| 0.5 | 83.58 |
| 1.0 | 83.76 |
| 2.0 | 83.75 |

The curve is flat within the ripple. The recent-primary ablation is the best
point (83.83): it keeps the drift baseline's primary prediction and spends the
allocator only on extra references, which recovers the wall while adding a few
compound leaves. The free allocator instead sometimes moves the primary to an
older slot and gives back ~0.07 K.

### Free-allocator trace, λ1 QP111 (59 P frames)

`AV2ENC_PIPE_REF_TRACE` writes one CSV row per P frame. Corrected free run:

- reference counts: **46 frames with 1 ref, 9 with 2, 4 with 3**
- primary slot distribution (slots 0..3): **15 / 14 / 17 / 13**
- **29 compound leaves**, **46 leaves selecting a non-primary reference**
- **23 drift-refresh frames**

Chosen ordered ref sets (slot numbers; frame 0 is the key and is not traced):

```text
f1:[0] f2:[1] f3:[2] f4:[2] f5:[0] f6:[1] f7:[2] f8:[3]
f9:[3] f10:[1] f11:[2] f12:[3] f13:[0] f14:[1] f15:[2] f16:[3]
f17:[0] f18:[1] f19:[2] f20:[3] f21:[0] f22:[0] f23:[2] f24:[3]
f25:[0] f26:[1] f27:[2] f28:[3] f29:[0] f30:[1] f31:[2] f32:[2]
f33:[0,2] f34:[1] f35:[2] f36:[3] f37:[0,3] f38:[1] f39:[2] f40:[3]
f41:[0] f42:[1,0] f43:[2] f44:[3,2,1] f45:[0] f46:[1,0] f47:[2,1] f48:[3]
f49:[0,3,2] f50:[1] f51:[2,1] f52:[3,2] f53:[0,3] f54:[1] f55:[2] f56:[3,2]
f57:[0,3,2] f58:[1,0,3] f59:[2]
```

At QP112 the same arm chose 43 / 15 / 1 frames with 1 / 2 / 3 refs, 27
compound leaves, 49 extra-reference leaves and 23 refreshes. The allocation is
content-dependent and sparse: most frames are best served by one reference;
the allocator only adds the second/third on the drift-heavy tail.

The recent-primary λ1 run chose 47 / 9 / 3 frames at QP111 and 47 / 10 / 2 at
QP112, with 20/16 compound leaves; its extra references are used more
(48/59 leaves).

### Compound

Compound never dominates this clip. The free allocator coded 29 compound
leaves (of ~2,900 inter leaves) at QP111; the recent-primary arm coded 20.
Masked compound is available (`--video-compound-wedge --video-compound-diffwtd`
are set by the gate) and priced with comp_group_idx + compound_type + mask
bits, but the AVERAGE blend wins most local comparisons. With the gate's
forced stress mode, 10 frames code **367 compound leaves** at three references
and decode 10/10; forcing DIFFWTD also decodes 10/10. That proves the
three-reference compound syntax is correct; it is not a rate result.

## What was built

### Frame-level allocator

`src/pipeline/video/video_pipe_ref.{h,cpp}`:

- for each P frame, scores the newest decoded slots on a 16x16 luma grid;
- cheap integer ME per reference (5x5 coarse at radius 8, ±2 refine,
  edge-clamped) so an older reference is not penalised for a global shift the
  real leaf search would find;
- evaluates ordered reference sets of 1..3 slots (all permutations of the top
  candidates, or the newest primary first with `RECENT_PRIMARY`);
- per 16x16 block, chooses the cheapest single reference or AVERAGE compound
  pair using **measured syntax prices**, not a free reference:
  - reference index: 0.55 / 2.20 / 3.20 / 4.00 … bits (R12 probe means);
  - compound: comp_mode + is_joint + compound_mode_non_joint + comp_ref
    (three references);
  - masked: comp_group_idx + compound_type + wedge/diffwtd mask;
- cost is `D + λ·R`; λ is the leaf SSE-per-bit lambda at the frame qindex times
  `AV2ENC_PIPE_REF_LAMBDA`;
- the chosen ordered list becomes the explicit §5.18.2 map; the existing
  writer codes it. No new syntax.

`num_ref_frames = 4` and an explicit map are used. The map costs
`3 + count·2` bits/frame, plus the `f(4)` refresh field instead of the compact
`f(1)` form. That is the price of allocation and is included in every number
above.

### Per-block prediction

`video_frame_decision.cpp` prices each inter candidate with:

```
D + λ * (inter_syntax_bits + ref_index_bits / compound bits + mask bits)
```

The compound candidates are the existing AVERAGE + WEDGE + DIFFWTD search over
references `(0,1)`. With three references the third is a single-prediction
candidate; the allocator's ordering decides which pair is compound-capable.
`video_syntax_inter.cpp` extends the writer's compound path to
`NumTotalRefs == 3` for pair `(0,1)`, including masked compound. This is
existing §5.20.7.6/.10/.11/.16 syntax, not a new tool.

### Drift refresh as an allocation option

`--video-drift-refresh` remains enabled under the gate
(`AV2ENC_PIPE_REF_DRIFT=0` removes it). The allocator chooses the primary; the
drift trigger then decides whether that frame is coded as a refresh. At λ1/QP111
23 of 59 P frames refreshed. This is the only reason the pipeline gets near the
old wall; without the tuned drift recipe it sits around K 83.1.

### Verification

- Gate off pins byte-identical:
  - default 480x240 QP189: 25,009 B `a1d0dba1…`
  - quality QP190: 25,141 B `3984677b…`
  - nightly QP110: 23,690 B `1b99127a…`
  - explicit-λ: 24,113 B `7f03eb6e…`
  - 720p: 36,572 B `c4ba2813…`
- `tools/test.sh --build build-dev`: 10/10.
- `tools/gate_video_nightly.sh`: OK.
- `src/pipeline/video/gate_video_qp_lambda.sh`: OK.
- `src/pipeline/video/video_pipe_ref_gate.sh`: pins, 60/60 decode, recon
  `cmp`, force-compound/mask, trace shape.
- Pipeline `--video-recon` equals the decoded Y4M byte for byte.

## Verdict

| Comparison | Value |
|---|---:|
| Best pipeline K@100 (recent-primary λ1) | **83.83** |
| Free allocator λ1 K@100 | 83.76 |
| Old drift wall at 100 kbps | 83.795 |
| Delta vs old wall | **+0.04** (tie inside ripple) |
| New target | **90.00** |
| Delta vs new target | **−6.17** |

The pipeline is a **ceiling record**, not a candidate. Reference allocation and
compound prediction can recover the old drift point and add a few compound
leaves, but they cannot close a 6 K gap. The remaining gap is per-frame coding
efficiency / a different prediction surface, not reference structure. Do not
merge; do not retune this branch against 90.

## Reproduction

```sh
REF=testdata/nasa_480x240_30fps_60f.y4m
BIN=build-dev/av2enc

# corrected free allocator, λ1, QP111
AV2ENC_PIPE_REF=1 AV2ENC_PIPE_REF_TRACE=/tmp/pipe.csv \
  "$BIN" "$REF" -o /tmp/p.ivf --video --frames 60 --cpu-threads 8 \
  --qp 111 --nightly \
  --video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250

# recent-primary ablation
AV2ENC_PIPE_REF=1 AV2ENC_PIPE_REF_RECENT_PRIMARY=1 \
  "$BIN" "$REF" -o /tmp/rp.ivf ... --qp 111 ...

# decode and score
tools/bin/dav2d-macos --strict 1 -i /tmp/p.ivf -o /tmp/p.y4m --muxer yuv4mpeg2
DAV2D=tools/bin/dav2d-macos python3 tools/quality.py --ref "$REF" --ivf /tmp/p.ivf --json /tmp/q.json
python3 tools/extra_metrics.py --ref "$REF" --test /tmp/p.y4m --json /tmp/e.json
python3 tools/kpi.py --quality-json /tmp/q.json --extra-json /tmp/e.json
```

Forced compound syntax check (gate stress only):

```sh
AV2ENC_PIPE_REF=1 AV2ENC_PIPE_REF_MINREFS=2 AV2ENC_PIPE_REF_FORCE_COMPOUND=1 \
  "$BIN" "$REF" -o /tmp/c.ivf --video --frames 10 --cpu-threads 8 --qp 106 --nightly
AV2ENC_PIPE_REF=1 AV2ENC_PIPE_REF_MINREFS=2 AV2ENC_PIPE_REF_FORCE_COMPOUND=1 \
  AV2ENC_PIPE_REF_FORCE_MASK=1 \
  "$BIN" "$REF" -o /tmp/m.ivf --video --frames 10 --cpu-threads 8 --qp 106 --nightly
```

# `--video-opfl` / `--video-refine-mv` diagnosis

Both temporal refinement tools are implemented on the
`--video-compound-near-near` NEAR_NEARMV B-frame carrier
(`src/pipeline/video/video_frame.cpp`: `predict_compound_optflow`,
`predict_compound_refinemv`, `compound_plain`).  Neither was bit-exact under
the pinned decoder (`dav2d-macos`, AVM `cd7d8bc`), so both were rejected by
name in `src/main.cpp`; the two root causes below were found with an
instrumented decoder build and are fixed.  Both flags are now on by default on
the CPU path and gated by `src/pipeline/video/gate_video_opfl_refine.sh`.

Verification (both flags, six streams: pan3 depth 3, pan5 depth 3, and pan3
`--video-max-depth 5 --video-min-block 8` for the non-16-aligned sub-block
split): `dav2d --strict 1 --threads 1` emits the expected picture count and
`--video-recon` is byte-identical to the decoded Y4M, with pinned IVF MD5s in
the gate.  `--video-gpu` still rejects both flags.

Reproducer (the same deterministic pan fixture the gates use — a smooth
sinusoid panned by `(3,1)` per frame, 3 frames, one B picture):

```sh
# generate pan3.y4m exactly as gate_video_opfl_refine.sh does, then
build/av2enc pan3.y4m -o opfl.ivf --video --b-frames 1 --video-compound \
    --video-opfl --video-recon opfl.y4m
tools/bin/dav2d-macos --strict 1 --threads 1 -i opfl.ivf -o opfl.dec.y4m
cmp opfl.y4m opfl.dec.y4m           # before the fix: 4 bytes differed

build/av2enc pan3.y4m -o rmv.ivf --video --b-frames 1 --video-compound \
    --video-refine-mv --video-recon rmv.y4m
tools/bin/dav2d-macos --strict 1 --threads 1 -i rmv.ivf -o rmv.dec.y4m
cmp rmv.y4m rmv.dec.y4m             # before the fix: 48,417 bytes differed
```

Measured on the B picture only (the key and P pictures were already exact):

| mode | Y | U | V | max |
|---|---|---|---|---|
| `--video-opfl` | 0 | 2 | 2 | 1 |
| `--video-refine-mv` | 16,149 | 2,791 | 29,477 | 114 |

A decoder built from the pinned source was instrumented to dump the per-sub-block
motion vectors, §7.13.3.4 ref areas and sub-block geometry, which identified
both causes.

## Cause 1 — OPFL must not take the §7.13.3.8 `sad` shortcut

§7.13.3.8 terminates the optical-flow block process when
`sad = get_sad(w, h, 0) < sadThresh` (6 outside `TIP_FRAME_AS_OUTPUT`) and
predicts the block with the coded MVs instead.  The encoder did that; the pinned
decoder only implements the check for TIP blocks
(`skip_opfl_refine_with_tip()`); for an ordinary `use_optflow` block it always
runs the refinement.  This is observable on the pan fixture:

* block `(560,64) 16x32` has coded MVs `(8,24)/(-8,-24)` (1 and 3 luma samples,
  i.e. integer in luma) and `sad = 3`;
* luma is identical either way (integer MVs make the filter irrelevant), which
  is why luma was already exact;
* chroma sits at half-pel (phase 8/16) in both directions, where the decoder's
  refined 4x4 sub-block prediction (EIGHTTAP_SHARP, downgraded to the four-tap
  table, clipped to the §7.13.3.4 ref area) differs from the plain EIGHTTAP
  block prediction by 1 LSB.

The differing samples were `U(286,40)`, `U(287,44)`, `V(289,45)`, `V(292,44)`
of the B picture.  Fix: always refine (the product path has no TIP), with a
comment recording why the shortcut must stay out.

## Cause 2 — refine-MV is a per-16x16 sub-block process

§7.13.3.1 predicts a refine-MV block in parts:

```
tw = Min(w, 16 >> subX)
th = Min(h, 16 >> subY)
for i = 0 .. h step th:
  for j = 0 .. w step tw:
    predict_refinemv(plane, x + j, y + i, j, i, tw, th, candMvs, ...)
```

so the §7.13.3.6 24-neighbour search, the `th = (w*h) << 1` early-exit
threshold, the §7.13.3.4 ref area and the final EIGHTTAP_SHARP prediction are
all sized by the **sub-block**, not by the coding block.  The encoder searched
and predicted the whole coding block at once, so a 16x32/32x16 leaf used a
different SAD region (and a different early-exit threshold) and a different ref
area than the decoder.

Chroma follows the same split: `twc = Min(cw, 8)`, `thc = Min(ch, 8)` for 4:2:0,
and §7.13.3.3 takes the *luma* refined MV for the colocated sub-block from
`RefineMvs` (`offsetMvs = RefineMvs[i >> (2 - subY)][j >> (2 - subX)]`) instead
of searching again; only the ref area is re-derived at chroma resolution.

Fix: split both planes into `Min(., 16)` (luma) / `Min(., 8)` (chroma)
sub-blocks, search per luma sub-block, keep the resulting offset MVs in a small
grid and reuse them for the colocated chroma sub-blocks.

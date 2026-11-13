# TIP decoder gate and `--video-tip` product stream

Status: **decoder support proven; CPU reference wired; GPU path, TIP-as-ref
tile symbols and the §7.22 temporal merge feed remain open.**

Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
(commit `38f28e7`), §5.4.6, §5.18.2, §6.20, §7.9, §7.10, §7.13.3.2, §7.22.

Pinned decoder: `dav2d f9dafbc` (macOS arm64 `tools/bin/dav2d-macos`, Linux
`/workspace/av2enc/tools/bin/dav2d`), always with `--strict 1 --threads 1`.

## 1. Decoder verdict

`dav2d` decodes TIP (OBU_REGULAR_TIP, type 14) in both TIP_FRAME_AS_OUTPUT
framing and with `enable_tip = 0` control streams.  The component writer
(`src/codec/inter/tip/`) emits a 4-unit IPTT stream whose two TIP OBUs have no
tile payload; the pinned decoder outputs all four pictures and the
reconstruction is sample-exact.

### Component gate repro

```sh
cd src/codec/inter/tip
DAV2D=$PWD/../../../tools/bin/dav2d-macos ./build_and_test.sh --allow-control-drift
```

Observed (macOS arm64, `f9dafbc`):

```text
tip_gate 64x64: units=4 bytes=1540 control_bytes=1519
  tip1: blocks=16 field_valid=0/64 holes_filled=0 stored=8x8
  tip2: blocks=16 field_valid(est)=49 field_valid(setup)=64 holes=15
  decode: key_diff=0(max 0) p_diff=0(max 0) tip1_diff=0(max 0) tip2_diff=0(max 0)
tip_gate 640x360: units=4 bytes=31689 control_bytes=31667
  tip1: blocks=920 field_valid=0/3600 holes_filled=0 stored=80x45
  tip2: blocks=920 field_valid(est)=3081 field_valid(setup)=3600 holes=519
  decode: key_diff=0(max 0) p_diff=0(max 0) tip1_diff=0(max 0) tip2_diff=0(max 0)
```

The same two IVF streams decode on the Linux 3060 box with the pinned
`dav2d`:

```text
dav2d -i tip_gate_64.ivf  -o out.y4m --strict 1 --threads 1   # 4/4 frames, exit 0
dav2d -i tip_gate_360.ivf -o out.y4m --strict 1 --threads 1   # 4/4 frames, exit 0
```

### `--allow-control-drift`

The gate also re-emits frames 0 and 1 with `enable_tip = 0` and asserts they
are **byte-identical** to `encode_video_sequence()`.  That assertion is now
stale: the current M2 writer differs from the gate's hand-written P header in
one bit (unit 1 byte 6, `0x56` vs `0x5a`; size is unchanged).  It is a
control-stream drift, not a TIP or decode failure, so the diagnostic flag
prints the difference and continues to the decode check.  The default
(no flag) still fails on the drift.  Fixing the gate's header copy is tracked
as follow-up work; the decode assertions above run with the flag.

## 2. Product `--video-tip` (CPU reference)

`--video --video-tip` codes 3..8 input pictures as:

| coded unit | frame | hint | slot | refs |
|---|---|---|---|---|
| 0 | key | 0 | 0 | — |
| 1 | future anchor P | N-1 | 1 | key |
| 2..N-1 | `TIP_FRAME_AS_OUTPUT` | i | i+1 | past = key or previous TIP, future = anchor |

Every reference frame's §7.22 field is stored from its **real coded MV grid**
(`store_motion_field_entry()` per 8x8), and every TIP frame stores its own
field with `store_tip_frame_motion_field()` so the next TIP in the chain can
project it.  `enable_ref_frame_mvs = 1` and `use_ref_frame_mvs = 1` on the TIP
frames; the anchor P keeps `use_ref_frame_mvs = 0` because its only reference
is the intra key (an empty field).  `Tip_Weighting_Factor` is chosen from the
temporal position, never index 2 (the single-list weight 16).

**Hole fill is off** (`enable_tip_hole_fill = 0`).  The encoder's §7.10.3 hole
fill is spec-exact on the component gate's field, but on a coded reference
field the pinned decoder's post-estimation validity mask differs by the cells
our §7.22 store marks NONE: the encoder fills them and the decoder keeps the
hole (MV 0), producing sparse luma diffs (max 4) on TIP frames.  With hole
fill off both sides treat holes as MV 0 and the stream is sample-exact.  See
`docs/negative-results.md`.

### Gate result (640x360, `testdata/shine_1280x720_60f.y4m` cropped)

`dav2d -q -i <ivf> -o <dec.y4m> --strict 1 --threads 1`, then byte-compare the
decoded and encoder-recon Y4M (the Y4M MD5s are equal, so the raw frames are
equal):

| stream | frames | IVF bytes | IVF MD5 | decoded Y4M MD5 = recon MD5 |
|---|---|---:|---|---|
| `--video-tip` 4 pictures | 4 | 16385 | `3b4a123e8a83fdf45f3cd9338c501a13` | `6d2ba7193cb85e04c5ed88eff3d19092` |
| `--video-tip` 8 pictures | 8 | 17957 | `3c7170d2bcefe7390b2916a1fc582454` | `aea77d1c2366d78ccaf94ca782073b8d` |

Raw-plane MD5s (without the Y4M header): 4 pictures
`cfbee4f61c3d352c1479050ed3fe73b3`, 8 pictures
`969955f00770978faf9e4e8d05df1b8f`.

The in-tree gate case is `src/pipeline/video/build_and_test.sh`, which runs
`video_gate --tip` at 4 and 8 pictures (synthetic 64x64 clip by default) and
requires `dav2d --strict 1`, exact recon and exact display order.  The same
phases at 640x360 pass with the real clip.

Default streams are unchanged: `--video-tip` is off, and the sequence/frame
header writers only add fields inside `if (config.enable_tip)` /
`if (config.enable_ref_frame_mvs)` branches.

## 3. Remaining gaps

1. **GPU path.** `--video-gpu --video-tip` is rejected by name in
   `src/main.cpp` (`video_tip` in the GPU reject list).  Wiring it needs the
   frame-level schedule in `src/cuda/pipeline/video/video_pipeline.cpp` plus
   the arena-backed `src/cuda/codec/inter/tip/` kernels
   (`tip_setup_motion_field`, `tip_build_plan`, `tip_predict_plan`) and the
   CUDA §7.9 motion-field estimation.  CPU↔GPU parity is not claimed.
2. **TIP_FRAME_AS_REF tile blocks.** §5.20.7.10 `tip_mode`, `tip_drl_mode`
   and `tip_pred_mode` symbols are not written, so `RefFrame[0] == TIP_FRAME`
   cannot appear in a tile.  The §9.3 CDFs are now installed in the M2 CDF
   state (`kTipModeCdf` / `kTipDrlModeCdf` / `kTipPredModeCdf` in
   `video_cdf_tables.h`) and the §7.12.2.15/.17/.18 candidate helpers exist in
   `src/codec/inter/tip/tip_candidates.cpp`.
3. **§7.22 feed into the merge stack.** Wired for the ordinary P and B chains
   (`--video-ref-mvs`, `docs/ref-frame-mvs.md`): every reference slot stores its
   §7.22 field, §7.9 estimation runs at frame start and the §7.12.2.7/.8 scan is
   spliced at steps 5/12 in both the decision and syntax stacks.  What remains
   open is **TIP as a reference inside a tile-coded frame**: `RefFrame[0] ==
   TIP_FRAME` needs the `TIP_FRAME_AS_REF` tile symbols and a following
   tile-coded frame, which the product TIP schedule does not build.
4. **TIP hole fill** as described above.
5. **OPFL/refine-MV inside TIP** (`enable_tip_refinemv`,
   `enable_opfl_refine`) stays `REFINE_NONE`; the component supports it but
   the product path does not build those statistics.

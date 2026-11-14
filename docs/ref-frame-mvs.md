# `--video-ref-mvs` — §7.22 field storage and §7.9/§7.12.2.7 temporal MVs

Status: **CPU product path wired and decoder-gated** on the ordinary P and B
chains, on GME/warp and on wedge-compound frames.  Default streams are
byte-identical when the flag is off.  GPU and multi-reference temporal layers
remain open (see §5).

Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
(commit `38f28e7`), §5.4.6 (`enable_ref_frame_mvs`), §5.18.2
(`use_ref_frame_mvs`, `fill_tpl_mvs_sample_gap`), §5.20.6.3 (`find_tx_size`),
§7.9, §7.12.2.7/.8, §7.13.3.14/.15, §7.22, §7.23.

Pinned decoder: `dav2d f9dafbc`, always `--strict 1 --threads 1`.

## 1. What is signalled

| field | where | value with `--video-ref-mvs` |
|---|---|---|
| `enable_ref_frame_mvs` | sequence header §5.4.6 | 1 |
| `use_ref_frame_mvs` | every non-intra frame header §5.18.2 | 1 |
| `enable_tip` | unchanged | 0 (TIP keeps its own path) |

`reduced_ref_frame_mvs_mode` stays 0 so §7.9 projects every reference, and
`tmvp_sample_step_minus_1` is not coded because the product superblock is
64×64 (ProjStep = 1; §5.18.2 codes the bit only for a non-64 superblock with
more than one reference).

## 2. Encoder model

For every coded inter frame with `enable_ref_frame_mvs = 1`:

1. **§7.22 storage.** `store_slot_field()` walks the frame's per-4×4 coded
   mode/MV grid and calls `inter::temporal::store_motion_field_entry()` for each
   8×8 entry: reference-index mapping, `REFMVS_LIMIT` rejection, `compression_mv`
   and the past/future list reordering.  Each slot also keeps
   `num_saved_refs`/`saved_order_hint[]`, mirroring §7.23
   (`RefNumTotalRefs[i] = NumTotalRefs`, `SavedOrderHints[i][j] = OrderHints[j]`).
   The intra key starts with an empty field.
   The grid cell also carries the block-level state §7.22 needs but the
   per-4×4 `ref`/`mv` pair cannot express (`MiInfo::warp`, `is_wedge`,
   `wedge_index`, `wedge_sign`, `bsize`):
   * a `GLOBALMV`/`GLOBAL_GLOBALMV` block whose §5.18.9 model derives a
     `TransformationType > TRANSLATION` and passes §7.13.3.21 shear validation
     stores the model projected at **every 8×8 centre**
     (`gm_sub_block_warp_mv()`, WARPEDMODEL_PREC_BITS → 1/8 pel, half away from
     zero), not the block-centre MV;
   * a §7.13.3.27 wedge compound block stores the block-local luma mask
     (`wedge_mask_value()`), and `store_motion_field_entry()` applies the
     `>60`/`<4` per-8×8 allow-list, which can drop one list.
   The last two are the decoder's `av2_copy_frame_mvs()` warp/wedge branches;
   validation is `video_gate --ref-mvs-field-selftest` for the projection
   arithmetic and the gate statistics `§7.22 warp 8x8 cells` /
   `§7.22 wedge allow-list cells` for the integration.
2. **§7.9 estimation.** At the start of an inter frame,
   `estimate_motion_field()` fills `MotionFieldConfig` from the current
   explicit reference map (slot, type, order hint, saved field) and calls
   `motion_field_estimation()` plus `fill_motion_field_sample_gap()`
   (a no-op at ProjStep = 1, as in the decoder).  §7.8's past/future split
   supplies `NumPastRefs`/`NumFutureRefs` and `OrigClosestPast/Future`.
3. **§7.12.2.7/.8 candidates.** The merge stack builder runs the temporal scan
   at the normative points: step 5 when `useTemporalFirst`, otherwise step 12
   (between the step 6–11 spatial scans and step 13's `(-1,-1-isSbBorder)`).
   `PruneCount` is the shared §7.12.2 budget; new entries keep
   `RefStackRowOffset/ColOffset = 0` and `CWP_EQUAL`, and single-prediction
   entries leave list 1 at its initialised zero.  The same callback is attached
   to the syntax writer's stack, so the writer's NEARMV/NEWMV validation uses
   the identical candidate list.

`merge` has no link dependency on `motion_field.cpp`: `StackParams` carries a
`TemporalScanFn` callback and `pipeline/video/video_temporal.h` is the only
adapter (used by `video_frame.cpp`).  `video_syntax.cpp` copies the function
pointer through `FramePlan::temporal`, so standalone gates keep linking.

### §7.13.3.15 save_subpu

`video::post::video_save_subpu()` implements `find_tx_size(w,h)` and the
sub-PU grid update (`SubPuSize`/`SubPuRowBase`/`SubPuColBase`, chroma
subsampled origin/stride).  `add_video_deblock_blocks()` applies it to an
optional per-frame sub-PU list.  Ordinary P/B leaves predict the whole PU, so
the process takes its normative early return (`w == PuWidth && h == PuHeight`)
and §6.19.3's `Max_Tx_Size_Rect[MiSize]` remains in the deblock grid — which is
what the grid already stored.  The self-test
(`video_gate --subpu-selftest`) pins the table lookup, the early return and the
luma/chroma sub-PU rectangles.

## 3. Gate

```sh
# CPU build, full writer (B / temporal layers need the CMake build)
OUT=/tmp/rg src/pipeline/video/video_ref_mvs_gate.sh \
  --bin build/av2enc --input clip.y4m --dav2d /path/to/dav2d \
  --base-bin /path/to/pre-change/av2enc
```

The script is registered in `tools/run_gates.sh` as `gate_video_ref_mvs`.

Every case encodes, decodes with `--strict 1 --threads 1`, checks the decoded
picture count, compares the decoded Y4M MD5 with `--video-recon`, and requires
temporal candidates (and, in the stress cases, at least one **selected**
temporal candidate).  With `--base-bin`, the default invocations must be
byte-identical.

Measured on the RTX 3060 host, 8-bit 4:2:0, QP 90, `f9dafbc`:

| case (360p, `shine360.y4m`) | pictures | IVF MD5 | recon = decoded Y4M MD5 | temporal candidates |
|---|---:|---|---|---:|
| default (flag off) | 8 | `1d053fe962098fb76139dfdfe340715a` | `3b5ee1369221c6ac2cba680ef0aa524f` | 0 |
| `--video-refs 2 --video-ref-mvs` | 8 | `f37eea404b4163bf91bafafa2eb28d98` | `3b5ee1369221c6ac2cba680ef0aa524f` | 564 |
| `--video-refs 3 --video-ref-mvs` | 8 | `48df657674764cae3b79f34bf5e6adb5` | `3b5ee1369221c6ac2cba680ef0aa524f` | 799 |
| `--video-refs 4 --video-ref-mvs` | 16 | `41eb82f9b0f7e41e91ea4e4ae6e5ed47` | `dbcf18457914210043b790e17c5f5f80` | 2567 |
| `--b-frames 2 --video-ref-mvs` | 4 | `cf69ce85780589b5ded0a3f3c8cbba48` | `702fae96d994e80ba48fed9a4ca3b549` | 96 |
| `--b-frames 4 --video-ref-mvs` | 6 | `694133a824ae83e1c31459f2c142a08d` | `f4b507fd446fb9313a8682afcd84759b` | 192 |
| `--video-tlayers 4 --video-ref-mvs` | 4 | `0a879da764e4fde23b2fec59cd2b56d9` | `702fae96d994e80ba48fed9a4ca3b549` | 0 (single reference) |

| case (720p, `shine_1280x720_60f.y4m`) | pictures | IVF MD5 | recon = decoded Y4M MD5 | candidates / selected leaves |
|---|---:|---|---|---:|
| `--video-refs 2 --video-ref-mvs` | 8 | `18e894f29093b15586c98930aac6aab9` | `14177f971462d911400089df98519fd6` | 4678 / 535 |

The stress mode (`--video-ref-mvs-stress`, gate only) always prefers a
NEARMV candidate that the temporal scan inserted: 282 of 564 candidates are
selected on the 360p P chain and 40 of 42 on the 4-picture B stream, still
recon-exact.  That proves the coded DRL indices name the decoder's temporal
candidates, not just that the entries exist.

Byte identity with the flag off was checked against a pristine `eb19656`
build for `--frames 8` (P), `--video-refs 2`, `--b-frames 2` and
`--video-tlayers 4` at 360p, and for the 8-picture 720p `--video-refs 2`
stream (`c090fc079bef453957536c01117c7e60`): all `cmp`-identical.

### Composition cases: GME/warp and wedge compound

The gate generates four deterministic fixtures (`fixture_pan64/pan360`,
`fixture_wedge64/wedge360`) so the tool is actually selected; each case is
encoded through the CLI, decoded by `dav2d --strict 1`, and compared
sample-for-sample.  Measured on macOS arm64 with `f9dafbc` (2026-11-14):

| case | pictures | recon = decoded Y4M MD5 | temporal candidates | §7.22 evidence |
|---|---:|---|---:|---|
| 64×64 pan, `--video-gme --video-refs 2 --video-ref-mvs` | 8 | `ace351ef470c22b63099fffd3641a284` | 42 | field stored from the GME grid |
| 64×64 pan, `--video-gme --video-refs 3 --video-ref-mvs` | 8 | `ace351ef470c22b63099fffd3641a284` | 59 | idem |
| 360p pan, `--video-gme --video-refs 2 --video-ref-mvs` | 8 | `051a7807ca210b4023b47b39861c5b04` | 7336 (2557 selected) | idem |
| 360p pan, `--video-gme --video-refs 3 --video-ref-mvs` | 8 | `135503a867b689fffadba987ecb05553` | 10182 (2338 selected) | idem |
| 360p pan, `--video-warp --video-refs 2 --video-ref-mvs` | 8 | `051a7807ca210b4023b47b39861c5b04` | 7336 | `--video-warp` is the GME alias |
| 64×64 wedge blend, `--video-refs 2 --video-compound --video-compound-wedge --video-ref-mvs` | 8 | `73bfea61be447e97ed535d30a5420b99` | 40 | 60 8×8 cells lost a list |
| 360p wedge blend, same flags | 8 | `55ee53f43120ac805f87ff2e12825423` | 2426 (60 selected) | 3050 8×8 cells lost a list |

The wedge allow-list counters are per-sequence and appear in the summary as
`§7.22 wedge allow-list cells N`; the gate fails if a wedge case reports 0.
On these fixtures the §5.18.9 fit is translation-only, so the decoder also
takes the non-warp store branch: `§7.22 warp 8x8 cells` stays 0 and the
per-8×8 projection is covered by `video_gate --ref-mvs-field-selftest`
(zoom, rotation, translation, rounding and the identity/translation model
classification).  A genuinely affine `--video-gme` model is not
reconstruction-exact in this tree for an unrelated warp-predictor reason
(`docs/negative-results.md`, "Gate `--video-gme` on genuinely affine
content"), so the composition is gated on content where the carrier itself is
exact.

Also fixed while gating: an 8-picture wedged P chain hit
`masked type without comp_group_idx` whenever the §7.13.3.27 search picked
AVERAGE in a stream with `enable_masked_compound = 1`.  The leaf now stores
`compound_type = 0` (the no-symbol sentinel) when `comp_group_idx` codes 0.

### TIP display frame as a reference

`--video-tip --video-ref-mvs` (4..8 pictures) codes the final input picture as
a tile-coded P whose references are the final `TIP_FRAME_AS_OUTPUT` picture and
the future anchor.  The TIP slot's §7.22 field (stored by
`store_tip_frame_motion_field()` when that TIP frame was built) is the §7.9
input, and every reported temporal candidate comes from that P.  Gate
(`video_tip_gate.sh`, synthetic 64×64 `video_gate --tip --ref-mvs`):

| stream | pictures | IVF MD5 | recon MD5 | candidates / selected |
|---|---:|---|---|---:|
| 360p `--video-tip --video-ref-mvs` | 4 | `12d333a4a77f83480bff7e4488e20e45` | `226fb5a076af193a4a5bf50bfe91a61a` | 255 / 19 |
| 720p `--video-tip --video-ref-mvs` | 8 | `ac84848c6e63c863770e699e8f3dff08` | `e14d6be1bc188aae01fa35fd192deb66` | 852 / 50 |

The plain `--video-tip` streams keep the pinned MD5s in
`docs/tip-decoder-gate.md`.  See that file's §2b for the schedule and the
`TipInterpFilter` syntax fix this needed.

## 4. Rate/quality position

This is a correctness feature, not an RD claim.  On the 360p single-GOP P
chain the SSE decision never selects a temporal candidate in normal mode
(insertion changes stack weights only; `temporal-selected leaves 0`), so the
tile bytes are unchanged and only the sequence/frame header bits differ.  At
720p the same settings select 535 temporal candidates; the stream size
changes from the flag-off baseline but no BD-rate measurement is claimed.
The gate therefore asserts insertion and, through the stress mode, selection.

## 5. Remaining gaps

1. **GPU.** `--video-gpu --video-ref-mvs` is rejected by name.  The CUDA M2
   path has no §7.9 estimator, no §7.22 store and no temporal stack hook.
2. **TIP block form (`RefFrame[0] == TIP_FRAME`).** The TIP *display frame* is
   now a valid reference slot for a tile-coded P (`--video-tip
   --video-ref-mvs`, `docs/tip-decoder-gate.md` §2b), with its §7.22 field
   feeding §7.9.  The §5.20.7.10 `tip_mode`/`tip_drl_mode`/`tip_pred_mode`
   block symbols are still not written, so a block cannot predict from the
   current frame's own §7.10 field.
3. **Temporal layers.** `--video-tlayers` is single-reference by construction
   (`plan.num_refs = 1`), and §7.9 returns an empty field for one reference.
   The flag is accepted, `use_ref_frame_mvs = 1` is signalled, both sides
   insert nothing and the stream decodes exactly — but no candidate exists.  A
   multi-reference hierarchical plan would be needed to exercise it.
4. **Real-affine `--video-gme` content.** The §7.22 per-8×8 warp store is
   implemented and unit-checked, but the product `--video-gme` warp predictor
   itself disagrees with the decoder once the fitted model has a non-identity
   linear part, independent of `--video-ref-mvs` (repro in
   `docs/negative-results.md`).  The composition is gated on translation-model
   content, where the decoder's store also takes the block-MV branch.
5. **TIP hole fill** is on in the product TIP stream (§7.10.3/.4 with the
   per-frame `allow_tip_hole_fill` bit); the encoder's and the pinned
   decoder's validity masks are cell-identical, see
   `docs/tip-decoder-gate.md`.

## 6. BRU (feature 3)

`--video-bru` is rejected by name.  The component writer builds a valid BRU
stream, but the pinned decoder does not parse `enable_bru`/`use_bru`:

```text
Error decoding frame: Invalid argument
Error parsing frame header
Error parsing OBU data
```

Reproducer (2026-11-13, Linux `f9dafbc`):

```sh
src/pipeline/video/video_bru_reject_gate.sh --bin build/av2enc \
    --core build/libav2enc_core.a --dav2d /path/to/dav2d
```

The script links `src/codec/inter/bru/bru_gate.cpp` against the built core
(the component build script's source list is stale), asserts the pinned
decoder stops at the BRU frame header, and asserts the CLI rejects
`--video-bru` with exit 2 and a message naming the flag.

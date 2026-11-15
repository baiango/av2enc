# Cross-frame CDF carry (`--cdf-continuity`)

`--cdf-continuity` makes a P frame start from the primary reference's adapted
CDF image instead of re-installing the §9.3 defaults: the encoder signals
`primary_ref_frame = 0` with `disable_cross_frame_cdf_init = 0`, the decoder
runs §7.4.4 `load_cdfs(ref_frame_idx[0])`, and both sides derive the next
frame's image with §8.2.4 `exit_symbol()` + §7.5 `frame_end_update_cdf()`.

It is CPU-only, single tile, 8-bit 4:2:0, and wired for the flat key+P chain
(including `--video-drift-refresh` and the whole `--nightly` aggregate). Gated
by `src/pipeline/video/gate_video_cdf_carry.sh`.

## What the encoder has to mirror

The decoder never reads our CDF image, so every frame's seed must be reproduced
exactly from the bitstream:

| Step | Decoder | Encoder |
|---|---|---|
| §5.18.2 | picks `primary_ref_frame` / `disable_cross_frame_cdf_init` | must reach the same decision (`CdfBank::begin_frame`) |
| §6.17.2 | reset branch installs `init_non_coeff_cdfs()` + `init_coeff_cdfs()` from the frame's own `base_q_idx` | `syntax::cdf_image_defaults_for_base_q_idx(frame_qp(i))` |
| §7.4.4 | `load_cdfs(ctx)` = the reference slot's saved context | `cdf_slots[ref_frame_idx[0]]` |
| §8.2.4 | tile CDFs → `Saved*` (here: `avgCdf`, `numLog2 = 0` for one tile, i.e. identity) | `CdfBank::save_tile_cdfs` |
| §7.5 | `frame = saved`, `count = (3*count)>>2` | `CdfBank::average_at_frame_end` |
| §7.23 | `save_cdfs(slot)` for `refresh_frame_flags` | `cdf_slots[write_slot] = cdf_frame` |

## Root cause: §5.18.2 forces a reset on the first inter frame

`choose_primary_secondary_ref_frame()` (§5.18.2) only lets a reference compete
for `primary_ref_frame` when `RefFrameType[idx] == INTER_FRAME`. For the first
P frame of a key+P chain the only reference is the keyframe, so it returns
`PRIMARY_REF_NONE`, and the frame header then forces

```
if ( DerivedPrimaryRefFrame == PRIMARY_REF_NONE ||
     primary_ref_frame == PRIMARY_REF_NONE ) {
    primary_ref_frame = PRIMARY_REF_NONE
    DerivedPrimaryRefFrame = PRIMARY_REF_NONE
    disable_cross_frame_cdf_init = 1
}
```

The signalled `primary_ref_frame = 0` is discarded: the decoder installs the
§9.3 image for that frame (`dav2d`: `p_ref_idx == DAV2D_PRIMARY_REF_NONE` →
`dav2d_cdf_thread_init_static`), while the encoder was seeding it from the
keyframe's §7.4.4 context. Every later frame then inherited the desync, so a
60-picture clip decoded 1/60 pictures (`Error decoding frame: Invalid
argument`, which `dav2d` reports while still exiting 0).

Function and line: `encode_video_sequence()`, the P-frame CDF-seed block in
`src/pipeline/video/video_frame.cpp` (`derived_primary_none` at the
`ref_count` derivation, used at the seed and at the frame-header bit). This is
also why the failure looked like "any frame that codes a real tile": an
all-skip frame codes few enough symbols that the wrong seed survives, and a
static clip decodes.

## The fix

1. **Mirror the §5.18.2 reset.** `derived_primary_none` is true when no
   reference slot holds an inter frame (`!slots[slot].is_key`). Such a frame
   re-derives the §9.3 image (`seed_frame_cdfs(i)`) and codes
   `disable_cross_frame_cdf_init = 1`, which is the value the decoder uses.
   Every later frame loads the carried context.
2. **The image was incomplete.** CDEF (`cdef_index0`, `cdef_idx_minus_1` for
   each `CdefStrengths`) and CCSO (`ccso_blk`) tile symbols adapt too, but
   their CDFs were not in the `§8.2` packed image: the nightly aggregate
   (`--video-cdef --video-ccso-rd`) desynchronised at the first carried frame
   that coded a CCSO symbol. They are now part of `CdfState`
   (`src/pipeline/video/video_syntax.cpp`: the new members, `init_cdfs()`, the
   `cdf_arrays()` descriptors, and the per-tile import/export around
   `cdef_tile_init`/`ccso_tile_init`).
3. **Composition.** `--cdf-continuity` no longer needs the reset-per-frame
   opt-in, so `--nightly` stops implying `--video-cdf-per-frame-qp` when the
   carry is on, and the drift-refresh prohibition is gone (`--video-drift-refresh`
   is allowed: the carried image follows the coded symbols, so a refresh
   frame's extra residual needs no §9.3 re-derivation). The CLI gate is gone:
   pinned `dav2d` f9dafbc decodes the carried form.

## Measurements (240p30 clip, 60 pictures, `--nightly --qp 110`, 8 threads)

| Stream | Bytes | kbps | CVVDP (JOD) | PSNR-Y |
|---|---:|---:|---:|---:|
| `--nightly` (reset-per-frame pin) | 24,904 | 99.616 | 9.7069 | 43.8667 |
| `+ --cdf-continuity` | 24,214 | 96.856 | 9.7069 | 43.8667 |
| `+ --cdf-continuity --video-key-qp-offset -27` | 24,885 | 99.540 | 9.7177 | 43.9358 |
| `+ --video-cdef-rd` (nightly pin) | 24,822 | 99.288 | 9.7177 | 43.9358 |

The first two rows have **identical decisions** (same `--video-recon`
MD5 `9fc52a211517548c08adac095c056a12`, `dav2d --strict 1` decodes 60/60
pictures and its payload equals the recon byte for byte), so carry is a pure
entropy win: **−2.77% rate**. Spend it on the reference intro and the iso-rate
point (99.616 kbps / 24,904 B) moves to ≈**9.714 JOD**: interpolating the
QP 108/109 carry sweep (25,054 B / 9.7043, 24,631 B / 9.7309) gives 9.7137, and
the key −27 anchor gives 9.7180. Read it as **+0.007 to +0.011 JOD** with
**+0.07 dB** PSNR-Y, inside the ±0.03 JOD QP ripple — the same "spend it on
the keyframe" shape as the rest of the CDF work.

The carry stream at QP 110 is 24,214 B, MD5
`a05e83ff6f99df506b7d64dddb5bb671`.

### Static-clip floor

A 60-picture clip whose every picture is the same frame isolates the per-frame
reseed cost (all P frames are skip-only):

| Image | IVF bytes | Bytes per P frame |
|---|---:|---:|
| reset per frame (`--video-cdf-per-frame-qp`) | 20,440 | 31.0 |
| carried (`--cdf-continuity`) | 19,818 | 20.5 |

−34% on the floor: the reset form re-codes the adapted image's cost every
frame, the carry form pays only the skip symbol.

## Scope

Wired and measured for: CPU, one tile, 8-bit 4:2:0, flat key+P chain with the
nightly aggregate (edge/partition RD, CDEF, CCSO, BAWP, RDOQ, TX search,
drift refresh). Rejected (by name) with `--video-cdf-per-frame-qp`,
segmentation/delta-Q, and a per-frame `--rc crf/vbr` plan. B/temporal-layer/
random-access/TIP schedules, multiple tiles and the 10-bit/4:2:2/4:4:4 cores do
not carry yet — each needs its own `dav2d --strict 1` + recon gate.

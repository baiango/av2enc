# Format video core (10-bit and 8-bit chroma formats)

This directory extends the M2 `--video` reference to the pixel formats dav2d
already decodes:

| Format | Status | Notes |
|---|---|---|
| 10-bit 4:2:0 | gated | `--bit-depth 10 --chroma 420` |
| 10-bit 4:2:2 | gated | `--video-max-depth <= 2` (32x32 roots) |
| 10-bit 4:4:4 | gated | |
| 10-bit 4:0:0 | gated | |
| 8-bit 4:2:2 | gated | `--video-max-depth <= 2` |
| 8-bit 4:4:4 | gated | |
| 8-bit 4:0:0 | gated | |

The default 8-bit 4:2:0 `--video` stream is **not** routed here; it keeps the
historical `video_frame.cpp` path byte-for-byte (`c6d588647aacb2275dfaea41a00fb501`
for 6 frames of `shine_1280x720_60f.y4m` with `--video-max-depth 3`).

## Layout

| File | Purpose |
|---|---|
| `video_hbd.h` | `Options`, `BeginFrame`, `PlaneTx`, the pixel `Backend` interface and the entry points |
| `video_hbd.cpp` | shared orchestration (partition tree, MV stack, mode decision, syntax assembly) + CPU pixel backend |
| `video_hbd_ops.h` | host/device inline arithmetic: DC prediction, subpel MC (with the §7.13.3.18 four-tap downgrade), SAD/SSE/variance, diamond + subpel ME |
| `src/cuda/pipeline/video/hbd/` | CUDA backend (same orchestration) |

The orchestration owns every decision and the bitstream; the backend performs
pixels.  `video_hbd_ops.h` is compiled for both host and device, so CPU and CUDA
cannot drift.

## Tool subset

Wired: flat key + P chain, `--video-refs 1..7` / `--video-num-ref-frames 4..8`,
partition depth/min-block/force-single, eighth/quarter/half MV precision,
subpel ME, DC intra fallback (intra keyframe and per-leaf fallback), the
opt-in inter residual, the §7.17-§7.20.5 in-loop filters, and `--video-grain`
(§5.14 model OBU + §5.18.10.1 config; see `video_grain_gate.sh`).

Rejected by name on these formats (all 8-bit 4:2:0 only today): tiles,
B frames, temporal layers, the long-term/OLK/leading/SEF/SWITCH/RAS wrappers,
`--video-tx-search`, `--video-intra-full`, `--video-gme`/`--video-warp`,
compound, DRL reorder, average/adaptive CDF, MFH, timing and CRF/VBR rate
control.  12-bit is not signalable in AV2 v1.0.0 and is rejected.

### 4:2:2 depth

`partition::chroma_plane_residual_size()` models 4:2:2 (`Subsampled_Size[.][1][0]`),
so the tree is no longer luma-only.  `--video-max-depth 3/4` produces 16-pixel
4:2:2 leaves (16x32 / 32x16) that decode exactly with the pinned `dav2d
--strict 1`; `gate_video_422_depth.sh` covers 8/10-bit at 64x64 and 640x360.
The default stays `--video-max-depth 2` for historical 4:2:2 streams; the
rejection of deeper trees was removed.

## Gate

`src/cuda/pipeline/video/gate_video_formats.sh` generates a deterministic
640x360 5-frame source in every format and, for each case, requires:

* CPU `--video` and CUDA `--video-gpu` byte-identical temporal units,
* byte-identical `--video-recon`,
* `dav2d --strict 1` decode with the expected picture count (4), and
* decoded output identical to the encoder reconstruction.

Cases: baseline, `--video-refs 2`, `--video-inter-residual`,
`--video-mv-precision quarter`, `--video-force-single`, and
`--video-min-block 8` (except 4:2:2).  Measured on the RTX 3060 (sm_86),
CUDA 13.0:

| Format | baseline MD5 (4 frames) |
|---|---|
| 420p10 | `927d08881841eecd27276f490c116bd9` |
| 422p10 | `0a781a50ff58307ee5b065245597cab3` |
| 444p10 | `ad4e395659c47762fd357d90287bb5a5` |
| mono10 | `c8aa8261c28747026ef23ff1d025bfff` |
| 422 (8-bit) | `52aa1f607847ebcede33782ea170eb2b` |
| 444 (8-bit) | `6f996d6054a8187111073518dc795028` |
| mono (8-bit) | `3bf37235e534044d792c291c9f3dd3e0` |

## Padding / edge rules

* Coded planes are `MiCols*4` by `MiRows*4` luma (chroma shifted), not
  64-aligned: the buffers match the decoder's `CurrFrame` allocation.
* DC prediction reads above/left samples clamped to `maxX`/`maxY`, exactly like
  §7.13.2.1 and the M1 intra component.  A partial bottom block must not read
  past `MiRows*4`.
* Subpel MC clips taps to the reference coded extent and applies the
  §7.13.3.18 four-tap downgrade when a pass is at most four samples wide/high
  (this is what makes 8x4 / 16x4 chroma blocks exact).
* ME returns MVs in 1/8 luma samples (an integer-pel offset is 8 units); the
  mode decision and the syntax writer share the same 1/8-pel values.

## In-loop filters (§7.17-§7.20.5)

`--video-deblock`, `--video-cdef`, `--video-ccso`, `--video-wiener`,
`--video-chroma-lr`, `--video-pc-wiener` and `--video-gdf` are wired on every
format served here (10-bit 4:2:0/4:2:2/4:4:4/4:0:0 and 8-bit 4:2:2/4:4:4/4:0:0).
Every flag defaults off, so every existing stream is byte-identical.

`video_hbd_filters.{h,cpp}` builds the frame-level `post::VideoFilterPlan` that
the shared M2 syntax writer already consumes and applies the same
depth-parameterised cores as the still writers:

| Filter | Applies to |
|---|---|
| §7.17 deblock | every frame (the metadata grid carries each leaf's skip flag) |
| §7.18 CDEF | intra key frame |
| §7.19 CCSO | intra key frame |
| §7.20 Wiener NS / PC-Wiener | intra key frame |
| §7.20.5 GDF | intra key frame |

Every frame is reconstructed in full before its tile payload is assembled, so
the key-frame plan is made from the completed reconstruction exactly like the
still writer.  Inter frames code `cdef_frame_enable = 0` / all
`FrameRestorationType = NONE`, which is why their skipped leaves need no CDEF or
LR skip grid.  The filtered key frame is also the reference for the following P
frames, so encoder and decoder references stay identical.

Scope limits (named):
* 4:2:2 / 4:4:4 chroma LR + deblock is rejected for the same pinned-decoder
  lossless-mask aliasing reason as the still writer.
* 8-bit 4:0:0 PC-Wiener is rejected with more than one frame: the key frame
  decodes bit-exactly, but the pinned decoder rejects the second frame's
  header.  10-bit 4:0:0 and 8-bit 4:2:2/4:4:4 PC-Wiener are gated exact.
* The CPU path only; the `--video-gpu` format dispatch rejects filters by name.

Gate: `src/pipeline/video/gate_video_filters.sh` encodes a deterministic
5-frame clip in every format, for each filter and the full chain, requires
`dav2d --strict 1` and a 0-diff reconstruction, and checks the
chroma-LR+deblock / 8-bit mono PC-Wiener rejections.

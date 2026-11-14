# CPU video RD preset (`--video-rd-preset quality`)

`--video-rd-preset quality` expands to the opt-in CPU tool set plus the
measured algorithmic partition decision that measured best on the project's CPU
quality gate (480x240, 30 fps, 60 frames, ~100 kbps). It is **off by default**:
without the flag no field is touched and every pinned stream keeps its bytes.
The preset is a pure CLI expansion, so each tool stays individually available
and the normal combination validation still runs.

```sh
av2enc testdata/nasa_480x240_30fps_60f.y4m -o /tmp/x.ivf \
    --video --frames 60 --qp 165 --cpu-threads 8 --video-rd-preset quality
```

Expansion (exactly this list; `tools/gate_video_rd_preset.sh` asserts that the
preset stream is byte-identical to typing the flags out):

| Setting | What it adds |
|---|---|
| `--video-inter-residual` | codes the quantised inter residual instead of skip-only leaves |
| `--video-deblock` | §7.17 deblocking on every reconstructed frame (output + reference) |
| `--video-ccso` | §7.19 CCSO after CDEF (band offsets) |
| `--video-tx-search` | §5.18.8 TX_MODE_SELECT luma TX size/type/partition search |
| `--video-intra-full` | 61-candidate luma intra search for the keyframe and intra leaves |
| `--video-partition-rd` | rate-aware partition tree: SSE-domain leaf cost + per-leaf syntax bits |

`--video-partition-rd` is not a tool toggle: it replaces the partition decision
cost. The historical `decide()` charged a source-variance leaf (SSE units)
against a SAD-domain signal cost, so at QP 190 the signal cost (`1.44e9`)
exceeded the largest possible 64x64 variance reduction (`6.66e7`) and every
optional split was unaffordable. The new cost puts both terms in the SSE
domain and charges every leaf its estimated mode/MV/skip syntax. See
`docs/video-partition-rd.md` for the derivation and the measured table.

It is CPU `--video` only (single tile, P-only chain); `--still` and
`--video-gpu` reject it. It does not change `--qp`, the TX lambda (256) or any
default.

## Measured RD

The stable `quality` rows below use the historical TX-search rate. The
newer algorithmic changes are **not** in this preset; they are the CPU-only
`--nightly` aggregate (`docs/video-nightly.md`), which adds the §8.3.2 TX
context rate, rate-aware CCSO and bottom-edge inter. `--video-tx-context-rate`
turns on the TX context model explicitly.

Host: Apple M4 (10 cores), `AV2ENC_CPU_FAST=ON`, `--cpu-threads 8`. Clip:
`testdata/nasa_480x240_30fps_60f.y4m`, 60 frames. Quality: `tools/quality.py`
(CVVDP 0.5.7, `standard_fhd`, MPS) against the pinned `dav2d-macos` decoder.

### Current preset (with `--video-partition-rd`)

| QP | kbps | CVVDP JOD | PSNR-Y |
|---:|---:|---:|---:|
| 164 | 102.132 | 8.3782 | 33.802 |
| 165 | **100.252** | **8.3438** | **33.547** |
| 166 | 99.280 | 8.3337 | 33.463 |
| 167 | 98.572 | 8.2810 | 33.363 |
| 168 | 98.324 | 8.2533 | 33.218 |
| 169 | 96.976 | 8.2026 | 33.036 |
| 170 | 96.812 | 8.2058 | 32.956 |
| 189 | 81.376 | 7.3391 | 30.015 |
| 190 | 81.920 | 7.3181 | 29.884 |
| 191 | 82.776 | 7.2597 | 29.792 |
| 192 | 81.980 | 7.2329 | 29.686 |
| 205 | 78.316 | 6.4597 | 27.551 |
| 208 | 75.108 | 6.3098 | 27.074 |
| 210 | 75.080 | 6.2725 | 26.913 |

At the original operating point:

| Stream | Rate | CVVDP | PSNR-Y |
|---|---:|---:|---:|
| old preset, QP 190 | 100.564 kbps | 6.3021 | 27.5816 |
| new preset, QP 165 (nearest measured) | 100.252 kbps (−0.31%) | **8.3438** | **33.547** |
| new preset, interpolated to 100.564 kbps (QP 164/165) | 100.564 kbps | **8.3495** | **33.589** |
| **delta (interpolated)** | matched | **+2.047 JOD** | **+6.008 dB** |

At iso-CVVDP (6.3021 JOD) the new curve reaches 75.1 kbps (interpolated QP
208/210), **−25.3% rate** versus 100.564 kbps.

### Pre-change preset (kept for comparison)

The preset before the partition change (25,141 B at QP 190):

| QP | default kbps | default JOD | default PSNR-Y | preset kbps | preset JOD | preset PSNR-Y |
|---:|---:|---:|---:|---:|---:|---:|
| 185 | 116.40 | 4.9989 | 24.72 | 118.88 | 6.5770 | 28.28 |
| 187 | 108.16 | 4.8875 | 24.58 | 111.06 | 6.4665 | 27.95 |
| 189 | **100.04** | **4.8021** | **24.47** | 103.70 | 6.3657 | 27.70 |
| 190 | — | — | — | **100.56** | **6.3021** | **27.58** |
| 191 | 95.04 | 4.7050 | 24.35 | 101.40 | 6.2466 | 27.44 |
| 192 | — | — | — | 98.42 | 6.1832 | 27.31 |
| 193 | 89.88 | 4.5918 | 24.24 | 96.16 | 6.1116 | 27.18 |
| 195 | 82.59 | 4.4809 | 24.10 | 86.91 | 5.9878 | 26.85 |

### Stage-by-stage at ~100 kbps (pre-change tools)

Linear interpolation between the two bracketing measured QPs, same clip:

| Tool set (incremental) | kbps | JOD | PSNR-Y | encode time |
|---|---:|---:|---:|---:|
| default | 100.04 | 4.802 | 24.47 | 1.0 s |
| `--video-inter-residual` (measured at 98.9) | 98.91 | 4.887 | 24.54 | 1.1 s |
| + `--video-deblock --video-ccso` | 100.04 | 5.909 | 25.37 | 1.5 s |
| + `--video-cdef --video-gdf` | 100.04 | 5.963 | 25.40 | 14.4 s |
| + `--video-tx-search` (on deblock+ccso) | 100.04 | 6.163 | 27.38 | 1.9 s |
| + `--video-intra-full` | 100.04 | 6.273¹ | 27.51¹ | 2.6 s |
| + `--video-partition-rd` | 100.56 | **8.3495** | **33.589** | 11.9 s (this host) |

¹ measured QP 190 (100.56 kbps) / interpolated.

Encode time (interleaved, best of 3, 60 frames, pre-change tools): default
0.97 s, preset 2.57 s (2.6x). On the current host the same binaries take
10.7 s (pre-change preset, QP 190) and 11.9 s (new preset, QP 165); the
partition change is ~1.1x, not a multiple.

## Verification

```sh
BIN=build-algo/av2enc
DAV2D=/Users/arin/work/av2enc/tools/bin/dav2d-macos
$BIN testdata/nasa_480x240_30fps_60f.y4m -o /tmp/p.ivf --video --frames 60 \
     --qp 165 --cpu-threads 8 --video-rd-preset quality --video-recon /tmp/p.y4m
tools/validate_dav2d.sh /tmp/p.ivf                 # OK, 60 frames
$DAV2D -q --strict 1 -i /tmp/p.ivf -o /tmp/p_dec.y4m --threads 1
# encoder recon payload == decoder payload, 10,368,000 B, byte for byte
tools/quality.py --ref testdata/nasa_480x240_30fps_60f.y4m --ivf /tmp/p.ivf \
     --json /tmp/p.json                            # 8.3438 JOD, 33.5471 dB
```

`tools/gate_video_rd_preset.sh` re-checks the expansion, the flag-less 480x240
default stream (25,009 B / `a1d0dba1...`), `--strict 1` decode, and the
whole-frame recon comparison. `docs/video-partition-rd.md` records the cost
model and the full RD table.

## Bottom-edge inter (`--video-edge-inter`)

The partition rule only requires a leaf's midpoint to be inside the picture,
so the bottom superblock row codes 64x64 leaves whose coded extent runs past
`MiRows*4` (480x240: leaves at `y=192` cover rows 192..255, but the MI extent is
240). The decoder reconstructs the full block into its SB-aligned picture
buffer; the visible rows are output and the reference is edge-extended from the
last visible row before the next frame reads it. The encoder used to decline
that emulation: `inter_fits` required `y0 + h <= mi_rows_*4`, so all eight
bottom-row leaves per P frame fell back to intra. They are 20% of the pixels
and, on a static clip, most of the steady P-frame bytes.

`--video-edge-inter` (CPU single-tile P chain only) makes the encoder match the
decoder:

* the per-frame reconstruction planes grow to the next superblock row
  (`round_up(MiRows*4, SbSize)`); the full coded block is reconstructed and
  neighbouring intra leaves still read the true hidden samples, while
  §7.13.2 availability (`maxX`/`maxY`) stays clamped to `MiRows*4`/`MiCols*4`;
* the frame buffer and every reference slot use the same padded extent, and
  after the in-loop filters the hidden rows are filled by replicating the last
  visible row — exactly the reference the decoder pads;
* `inter_fits` keeps the full-width test but accepts a bottom leaf whose visible
  part is inside the picture. Non-fitting right-edge leaves still go intra.

It is deliberately **not** in `--video-rd-preset quality`: the preset stream is
pinned, and this changes the block decisions. The flag is a separate opt-in, so
the default (25,009 B / `a1d0dba1...`) and preset (25,141 B) streams stay
byte-identical.

Measured on the same host/clip, `tools/quality.py`, `--video-rd-preset quality`
plus the flag:

| QP | kbps | JOD | PSNR-Y | SSIM-Y | bottom-row inter/intra leaves |
|---:|---:|---:|---:|---:|---:|
| 143 | 100.89 | 8.6892 | 35.2239 | 0.9592 | 472 / 8 (key only) |
| 144 | 98.18 | 8.6735 | 35.1301 | 0.9588 | 472 / 8 |
| 150 | 86.28 | 8.4906 | 34.3742 | 0.9544 | 472 / 8 |
| 165 | 64.34 | 7.9335 | 32.1921 | 0.9369 | 472 / 8 |
| 186 | 40.34 | 6.8644 | 28.9539 | 0.8983 | 472 / 8 |

At the preset's QP 190 point (100.56 kbps, 6.3021 JOD, 27.5816 dB) the
iso-rate edge point is QP 143: **+2.39 JOD and +7.64 dB PSNR-Y for the same
rate**. The flag also costs little time on this clip: 5.1 s vs 4.6 s for the
same 60-frame preset encode.

Verification for the new path:

```sh
BIN=build-g/av2enc
DAV2D=/Users/arin/work/av2enc/tools/bin/dav2d-macos
$BIN testdata/nasa_480x240_30fps_60f.y4m -o /tmp/e.ivf --video --frames 60 \
     --qp 142 --cpu-threads 8 --video-rd-preset quality --video-edge-inter \
     --video-recon /tmp/e.y4m --video-trace 2>/tmp/e.trace
# bottom-row inter leaves: 472, intra: 8 (the intra keyframe)
$DAV2D -q --strict 1 -i /tmp/e.ivf -o /tmp/e_dec.y4m --threads 1
# encoder recon payload == decoder payload for all visible samples
tools/gate_video_edge_inter.sh --bin "$BIN" --dav2d "$DAV2D"
```

`tools/gate_video_edge_inter.sh` checks the trace counts, the same-QP size
drop, `--strict 1` decode, the visible-sample recon equality, and the named
`--video-gpu` / `--tiles` / `--still` rejections. `--video-edge-inter` rejects
the B-frame/reorder/temporal-layer modes and `--video-ref-frame-mvs` because
their motion fields have no coded MI rows below `MiRows*4`.

Earlier partial design (rejected): copy the reconstructed hidden rows straight
into the reference. It decodes differently in the bottom four visible rows,
because the decoder discards them when it extends the reference. The
`maxY = plane.height-1` variant was also wrong for intra leaves: it let DC/left
neighbours read the padded rows (the same class of bug as the
`docs/negative-results.md` partial-bottom-block entry). Both are why the
reference pad is an explicit post-filter edge replication and the intra clamp
is stated in coded MI units.

## Rejected variants (same clip, same host)

| Variant | Result | Why it is not in the preset |
|---|---|---|
| partition domain fix only (`price * 2` signal bits, no leaf term), price 4,000 | 8.247 JOD at 99.68 kbps | slightly below the 8.344 JOD/100.25 kbps point; keeps leaves unpriced |
| leaf-term model with price 667 | 8.266 JOD at 99.11 kbps | same family as the chosen model; 700 measured better |
| `price` scaled by `step²` (calibrated at QP 190) | 8.08 JOD at 100.56 kbps (QP 172/174 interpolation) | over-splits at lower QP; the variance proxy is QP-independent, so a QP-scaled Lagrangian under-prices splits at high quality |
| `price` scaled linearly with `step` | 8.12 JOD at 100.56 kbps (QP 170/172 interpolation) | same over-split |
| domain-fix `price` 16,000 (less splitting) | 8.09 JOD at 97.1 kbps (QP 170); below the chosen curve at iso-rate | loses ~0.2 JOD at 100 kbps |
| `--video-cdef --video-gdf` | +0.05 JOD | 10x encode time (14.4 s vs 1.5 s) for a small filter gain |
| `--video-tcq` | 6.223 JOD at 98.1 kbps | 145 s encode (60x) for ~+0.01 JOD |
| `--video-tx-lambda 128/512/1024` | 6.18/6.17/6.18 | default 256 is already on the plateau |
| `--video-subpel-eighth` | +0.02–0.03 JOD | real but marginal; leave it as an explicit add-on |
| `--video-per-segment-q` | ~+0.02 JOD | marginal, adds a rate knob (`--video-seg-delta`) |
| `--video-delta-q` | 6.161 | worse at matched rate and 13.7 s encode |
| `--video-cctx` | 6.216 | neutral |
| `--video-uv-delta-q {1,2,4,6,10,15,20}` | 6.187 at 2; higher deltas raised rate | worse at matched rate |
| `--video-refs 2 --video-compound` | 6.138 | worse at matched rate |
| `--video-adaptive-mvd` | 6.160 | neutral |
| `--video-ccso-bands 5/6` | neutral / worse | no gain |
| `--video-pc-wiener`, `--video-wiener`, `--video-chroma-lr` | small / neutral / rate explosion | not worth the symbol cost at 100 kbps |
| `--video-gme/--video-warp`, `--video-bawp` | rejected by the CLI | gated on the plain skip-only P chain, incompatible with `--video-inter-residual` |
| `--video-opfl`, `--video-compound` | rejected / needs B frames | requires `--b-frames`, which this 60-frame P chain does not use |
| `--video-qm` | rejected | gated at QP 90 |

## Caveats

This is one clip at one quality point (240p30, ~100 kbps), measured on one
host. The preset is an RD preset, not a speed preset, and the per-tool gains do
not transfer as a fixed JOD number to other content or rates. The partition
price is calibrated on this clip; it should be re-measured before being used as
a fixed constant on very different content. The default stream, the GPU
calibration stream and every pinned baseline remain byte-identical; the flag is
the only way to opt in.

# entropy bug scout

Lane: coefficient coding / EOB / sign / level contexts, CDF selection and
adaptation, `qctx` slices, lossless, `txb` context resets, tile/frame CDF init,
and average-CDF. Build: `build/av2enc` (main `527738a`); decoder
`tools/bin/dav2d-macos` (`f9dafbc`).

| id | severity | class | title | repro | evidence | files | status |
|---|---|---|---|---|---|---|---|
| SCOUT-ENTROPY-01 | med | confirmed | `validate_dav2d.sh` passes a multi-frame stream whose later frames failed to decode | `head -c $((sz-50)) v.ivf > vcut.ivf; tools/validate_dav2d.sh vcut.ivf` | 4-frame IVF: `nb_read_frames=3`, dav2d rc 0, validator rc 0 and prints `OK`; same for a 4-frame CDF-continuity stream that decodes 1/4 | `tools/validate_dav2d.sh:55,85-92,139` | new |
| SCOUT-ENTROPY-02 | med | suspected | packed CDF image / average-CDF bank silently omits adapted CDF arrays (CCTX, TX-partition, CDEF, palette, IBC, ext-partition, compound) | static: compare `cdf_arrays()` with the adapted members | 43 (still) / 76 (video) descriptor entries packed; `cctx_type`, `tx_partition`, `fsc_mode`, `cdef_*`, `palette_cdfs_`, `ext_cdfs_`, `ibc_cdfs_`, `lr_state_` (still) and `comp_mode`/`is_joint`/`compound_mode_non_joint` (video) are absent. Current streams stay consistent only because `--tiles` rejects those tools and every video frame resets | `src/pipeline/still/still_syntax.cpp:253-258,345-411,517,2888-2913`; `src/pipeline/video/video_syntax.cpp:275-277,379-490` | new |
| SCOUT-ENTROPY-03 | high | duplicate | `--avg-cdf --adaptive-cdf` + B-frames / temporal layers / long-term / OLK / leading / SWITCH / RAS writes adapted tiles under headers that signal `disable_cdf_update=1` | `build/av2enc testdata/shine_1280x720_60f.y4m -o /tmp/b.ivf --video --frames 4 --b-frames 2 --avg-cdf --adaptive-cdf` | encode rc 0; dav2d rc 1, `No data decoded`; `--b-frames 2 --avg-cdf` is byte-identical to no `--avg-cdf` | `src/pipeline/video/video_frame.cpp:470-472,1234,1434`; `src/pipeline/video/video_frames.cpp:84,335,435` | duplicate — `SCOUT-FRAME-01` (`frame-types.md`), `SCOUT-INTER-ADV-01` (`inter-advanced.md`) |

## Details

### SCOUT-ENTROPY-01 — validator never checks the decoded frame count

Repro:
```sh
cd /Users/arin/work/av2enc
./build/av2enc testdata/shine_1280x720_60f.y4m -o /tmp/v.ivf --video --frames 4
sz=$(stat -f%z /tmp/v.ivf); head -c $((sz-50)) /tmp/v.ivf > /tmp/vcut.ivf
./tools/bin/dav2d-macos -q -i /tmp/vcut.ivf -o /tmp/vcut.y4m --threads 1; echo dav2d=$?
ffprobe -v error -f yuv4mpegpipe -i /tmp/vcut.y4m -count_frames \
    -show_entries stream=nb_read_frames -of default=nw=1:nk=1      # 3
tools/validate_dav2d.sh /tmp/vcut.ivf; echo validate=$?           # 0
```
Observed: `dav2d=0`, `nb_read_frames=3`, and the validator prints
`ivf: 1280x720, 4 frame(s)`, `nb_read_frames=3`, `OK: dav2d decoded`, exit 0.
The same happens for the documented entropy-only reproducer
`AV2ENC_ENABLE_UNGATED_CDF_CONTINUITY=1 ./build/av2enc testdata/shine_1280x720_60f.y4m
-o /tmp/gh.ivf --video --frames 4 --avg-cdf --adaptive-cdf --cdf-continuity`
(4-frame IVF, dav2d decodes 1 frame with 5 `Error` lines, exit 0, validator
reports OK).

Expected: fail when the decoded frame count is below the IVF header count
(read at `validate_dav2d.sh:55`) or the source count (line 111).

Likely cause: the IVF `count` is only printed, `FRAMES` is used only to gate
the PSNR comparison, and the only decode-failure check is "output file is
empty" (`:80-83,139`). dav2d returns 0 after dropping unparseable later
frames, so every multi-frame entropy gate built on this script can pass with
frame 0 only.

### SCOUT-ENTROPY-02 — CDF image/bank omits adapted arrays

Repro (static; grep the packed descriptor list and the adapted members):
```sh
cd /Users/arin/work/av2enc
sed -n '345,411p' src/pipeline/still/still_syntax.cpp   # cdf_arrays()
sed -n '251,258p;2888,2913p' src/pipeline/still/still_syntax.cpp
sed -n '373,490p' src/pipeline/video/video_syntax.cpp   # cdf_arrays()
```
Observed: `cdf_arrays()` is a hand-maintained list and does not cover every
array that `init_cdfs()`/the writers adapt:
* still `CdfState`: `cctx_type` (coded at `still_syntax.cpp:1769`),
  `tx_partition` (`:2657-2668`), `fsc_mode` (`:2600`).
* still writer members outside `CdfState`: `cdef_index0_`/`cdef_minus1_`
  (`:2911-2912`, coded at `:2561`), `palette_cdfs_` (`:2895`, used at
  `:2193`), `ext_cdfs_` (`:517`, used by the partition writer),
  `ibc_cdfs_`, `lr_state_`.
* video `CdfState`: `comp_mode`, `is_joint`, `compound_mode_non_joint`
  (`video_syntax.cpp:275-277`, coded at `:1935-1963`).

Consequence: `export_cdf_image()`/`import_cdf_image()` and
`cdf_image_tables()` (`still_syntax.cpp:426-446,2974-3007`) cannot seed or
average those arrays. A tiled still that enabled CCTX/TX-partition would start
every tile from the §9.3 defaults instead of the frame CDF; a cross-frame CDF
carry would lose them and desync at the first block that codes one.

Status: suspected. Repro is blocked today because `--tiles` rejects every one
of these tools by name (`main.cpp:3208-3260`) and the video chain resets to
defaults on every frame (`video_frame.cpp:2102-2112`), which keeps the omitted
arrays at defaults on both sides. It becomes a real desync as soon as
`--tiles` admits those tools or `--cdf-continuity` is enabled.

Expected: pack every adapted array (including arrays outside `CdfState`), or
assert/reject when a non-packed array is enabled.

### SCOUT-ENTROPY-03 — duplicate: CDF flags lost in frame-type wrappers

Repro:
```sh
cd /Users/arin/work/av2enc
./build/av2enc testdata/shine_1280x720_60f.y4m -o /tmp/b.ivf \
    --video --frames 4 --b-frames 2 --avg-cdf --adaptive-cdf
./tools/bin/dav2d-macos -q -i /tmp/b.ivf -o /tmp/b.y4m --threads 1; echo $?  # 1
./build/av2enc testdata/shine_1280x720_60f.y4m -o /tmp/b2.ivf \
    --video --frames 4 --b-frames 2 --avg-cdf   # byte-identical to no --avg-cdf
```
Observed: encode exits 0; dav2d exits 1 with `Error parsing OBU data` /
`No data decoded`. The per-frame `InterFrameEncoder` derives
`disable_cdf_update = !options_.adaptive_cdf` and `enable_avg_cdf =
options_.avg_cdf` (`video_frame.cpp:470-472`), but `encode_b_sequence`
(`:1234`), `encode_temporal_sequence` (`:1434`) and the long-term/leading/
random-access `SequenceConfig`s (`video_frames.cpp:84,335,435`) hard-code
`disable_cdf_update = true` and leave `enable_avg_cdf = 0`
(`video_headers.h:55,59`). Same root for `--video-tlayers`,
`--video-long-term`, `--video-olk`, `--video-leading`, `--video-switch`,
`--video-ras`; `--avg-cdf` alone is silently dropped (identical stream).
`docs/negative-results.md:78` still claims the gated `--avg-cdf
--adaptive-cdf` form "decodes exactly", which is false for these wrappers.

Already reported: `SCOUT-FRAME-01` (`frame-types.md`) and
`SCOUT-INTER-ADV-01` (`inter-advanced.md`). Kept here only to close the lane.

## Coverage that found no defect

Everything below was encoded and compared byte-exactly against the dav2d
decode (raw-frame MD5), with dav2d's default strict conformance checking
(`--strict 1`):

* still `--qp` 1/45/89/90/91/115/140/141/165/190/191/255 (all four `qctx`
  slices) with and without `--adaptive-cdf`, plus `testsrc2`/noise/flat/black
  content at qp 1/45/90/191/255;
* `--still-tcq`, `--still-idtx`, `--still-cctx`, `--still-ist`,
  `--still-tx-partition`, `--still-rdoq`, `--still-tx-type`, `--still-mrl`,
  `--still-dip`, `--still-mhccp`, `--still-palette`, `--still-ibc`,
  `--still-sub8`, `--still-ext-partitions`, `--still-sb128/256`,
  `--still-per-segment-q`, `--still-delta-q`, and TCQ/QP combinations
  (128x128 and 256x256 fixtures; the pre-existing ext-partitions/SDP edge
  failures on 100x70 are `SCOUT-PARTITION-01/07`, not entropy);
* tiled still 2x1/2x2/4x2/16 with `--avg-cdf`, qp 45/91/141/191/255, on
  256x256 and 640x360 (plus plain `--still` on 100x70);
* `--lossless` (4x4, FSC 8x8, `--adaptive-cdf`) on 8x8..256x256, odd
  multiples of 8 (72x72/136x88/120x120), and screen-content fixtures —
  always sample-exact;
* video: plain, `--tiles 2x2/4x2/8x2`, `--video-compound`, `--video-refs 2`,
  `--video-inter-residual`, `--video-timing`, `--video-mfh`, each with and
  without `--avg-cdf --adaptive-cdf`; 8/64-frame runs (`--video-recon`
  compared to dav2d) exact. B-frames/temporal-layers/rate control are exact
  in their supported combinations; B-frames/tlayers + `--avg-cdf
  --adaptive-cdf` is SCOUT-ENTROPY-03.

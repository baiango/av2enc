# Chroma-format encoder component (`src/encoder/format/chroma/`)

Status: **encoder-side format support for 4:2:2, 4:4:4 and 4:0:0**, with the
§7.17-§7.20.5 in-loop filter stage.  Standalone component and gate streams; the
M1/M2 cores are intentionally untouched. The three extra formats are encoded, decoded by `dav2d
--strict 1`, and compared bit-exactly against the encoder reconstruction at
64x64 and 640x360.

Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
(commit `38f28e7`). `dav2d` pin: `f9dafbc` (`tools/bin/dav2d*`).

## Product wiring (2026-11-12)

`av2enc --still --chroma 422|444|400` routes to
`chroma::encode_chroma_still_picture` through `src/formats/format_encode.*`
and the strict Y4M reader `src/formats/format_y4m.*`. The input must be an
8-bit Y4M in the requested format (`C422`, `C444` or `Cmono`); the product path
does not resample. `--qp` pins 90 (the writer's qlookup(90) step) and
`--recon <file>` dumps the reconstruction. `src/formats/format_gate.sh` pins
both 64x64 and 640x360 IVF/decoded MD5s and requires a 0-diff recon match.
`--chroma 420` remains the default 8-bit 4:2:0 M1 stream.

Related existing components that were reused, not modified:

| Component | Reuse |
|---|---|
| `src/encoder/format/` | `chroma_format_idc`, `Subsampled_Size`, 4:2:0→4:2:2/4:4:4/4:0:0 sample conversion |
| `src/bitstream/` | `BitWriter`, OBU framing, temporal delimiter, M0 frame header |
| `src/entropy/av2/` | §8.2 symbol coder; `src/encoder/basic_cdf_tables.h` CDFs (qctx 0) |
| `src/encoder/basic_tables.h` | DCT kernels and quantiser step `528` |
| `src/io/` | Y4M input and IVF output |

## Spec findings that constrain the work

1. **`chroma_format_idc` is not the enum order** (§6.4.1 Table 6.2):
   `4:2:0 = 0`, `4:0:0 = 1`, `4:4:4 = 2`, `4:2:2 = 3`.
2. **`Monochrome` is inferred**, not signalled:
   `Monochrome = (chroma_format_idc == CHROMA_FORMAT_400)`,
   `NumPlanes = Monochrome ? 1 : 3`.
   `SubsamplingX/Y` come from the same table; 4:0:0 uses `(1,1)` even though
   there is no chroma plane.
3. **Monochrome removes syntax**, it does not merely zero it (§5.4.3, §5.4.5,
   §5.4.8):
   * `enable_sdp` is not coded (inferred 0);
   * `cfl_ds_filter_index` is not coded;
   * `enable_chroma_dctonly`, `enable_cctx`, `separate_uv_delta_q` are not
     coded;
   * the whole UV delta-q block (`base_uv_ac_delta_q`,
     `uv_ac_delta_q_enabled`) is absent.
   A writer that emits those bits for 4:0:0 produces a desynchronised header.
4. **Chroma tools have no plane in Monochrome**: CfL, MHCCP, chroma CCSO,
   CDEF UV strengths, loop-restoration UV tools and U/V deblock planes are
   unavailable. `read_intra_uv_mode()` and the chroma residual are absent
   because `HasChroma = NumPlanes > 1 && TreeType != LUMA_PART` (§5.20.5.3).
5. **`ChromaMiSize`** (§5.20.3.2) is the luma block size whose residual a
   chroma block shares. With SDP disabled it equals `MiSize` for every leaf
   whose chroma residual is a legal `Subsampled_Size` entry; it differs only
   when `is_chroma_offset_for_subsize()` would shrink a subsampled dimension
   below 4 samples. Our minimal writer uses `BLOCK_64X64` leaves only, so
   `ChromaMiSize == BLOCK_64X64`.
6. **Chroma residual sizes** (§5.20.7.26):

   | Luma | 4:2:0 | 4:2:2 | 4:4:4 | 4:0:0 |
   |---|---|---|---|---|
   | 64x64 | 32x32 | 32x64 | 64x64 | — |
   | 32x32 | 16x16 | 16x32 | 32x32 | — |
   | 32x16 | 16x8 | 16x16 | 32x16 | — |
   | 32x8 | 16x4 | 16x8 | 32x8 | — |

   `Subsampled_Size` has real `BLOCK_INVALID` entries (for example 4x8 luma in
   4:2:2). `chroma_mi_size()` returns `nullopt` instead of guessing.
7. **Chroma transform size** (§5.20.7.25): `get_tx_size(plane>0)` is
   `Max_Tx_Size_Rect[get_plane_residual_size(ChromaMiSize, plane)]`.
   `TX_32X64` and `TX_64X64` map through `Adjusted_Tx_Size` to a 32x32
   coefficient grid, then §7.15.4 expands the residual by sample duplication.
   **The row/column shifts and the rectangular pre-scale use the nominal
   `txSz`, not the adjusted size.** For `TX_32X64` the shifts are `(6,12)`;
   using the adjusted 32x32 square would wrongly give `(6,13)` and the recon
   would differ from the decoder. This was the one non-obvious bug in the
   bring-up (`chroma_residual.*` now takes both sizes).
8. **Profiles** (Annex A.2): 4:0:0/4:2:0 → profile 0, 4:2:2 → profile 3,
   4:4:4 → profile 4. 8-bit only; 10-bit is owned by
   `src/encoder/format/` + `src/cuda/m1/hbd/`.

## Format / tool matrix

| Format | idc | SubX | SubY | Monochrome | Planes | Profile | `enable_sdp` | CfL | MHCCP | chroma CCSO | `chroma_dctonly` | `separate_uv_delta_q` | UV mode |
|---|---:|---:|---:|:---:|---:|---:|---|---|---|---|---|---|---|
| 4:2:0 | 0 | 1 | 1 | 0 | 3 | 0 | coded, 0 | no | no | possible | coded | coded | coded |
| 4:0:0 | 1 | 1 | 1 | 1 | 1 | 0 | **force 0, bit absent** | **off** | **off** | **off** | **force 0, bit absent** | **force 0, bit absent** | **absent** |
| 4:4:4 | 2 | 0 | 0 | 0 | 3 | 4 | coded, 0 | no | no | possible | coded | coded | coded |
| 4:2:2 | 3 | 1 | 0 | 0 | 3 | 3 | coded, 0 | no | no | possible | coded | coded | coded |

`ToolLegality` and `tool_legality_report()` expose this as data; tests pin the
Monochrome-forced entries.

## API map

| Area | Header | Functions |
|---|---|---|
| Table 6.2 | `chroma_format.h` | `chroma_format_info`, `chroma_format_from_idc`, `chroma_format_idc`, `subsampling_x/y`, `is_monochrome`, `num_planes` |
| Validation | `chroma_format.h` | `validate_encoder_format` |
| `ChromaMiSize` | `chroma_format.h` | `chroma_mi_size`, `is_chroma_offset_for_subsize` |
| Geometry | `chroma_format.h` | `make_frame_shape`, `make_frame_shape_checked`, `supports_64x64_none`, `PlaneShape`, `FrameShape` |
| Tool legality | `chroma_format.h` | `tool_legality`, `tool_legality_report`, `chroma_checklist` |
| Header writer | `chroma_headers.h` | `validate_chroma_seq_config`, `write_sequence_header_syntax`, `make_sequence_header_obu`, `assemble_chroma_still` |
| Header reader | `chroma_headers.h` | `read_sequence_header_syntax`, `parse_sequence_header_obu`, `find_sequence_header`, `SequenceHeaderInfo` |
| Residual shape | `chroma_residual.h` | `make_tx_shape`, `transform_shifts`, `forward_dct`, `inverse_dct`, `quantize_coefficients`, `dequantize_block`, `build_diagonal_scan` |
| Still writer | `chroma_still.h` | `make_chroma_picture`, `encode_chroma_still_picture`, `copy_visible_plane`, `write_recon_y4m` |
| In-loop filters | `chroma_filters.h` | `ChromaFilterOptions`, `ChromaFilterPlan`, `ChromaFilterPlanes`, `run_chroma_filters` |
| Gate tool | `chroma_still_main.cpp` | `chroma_still` CLI |

`SequenceHeaderInfo` exposes the inferred values (`monochrome`,
`subsampling_x/y`, `num_planes`, `bit_depth`, `max_quantizer`) next to the
coded fields, so a reader cannot accidentally treat `Monochrome` as an
independent bit.

## Minimal stream design

One key frame, one tile, one `PARTITION_NONE` 64x64 superblock at a time:
DC_PRED intra, `TX_MODE_LARGEST`, DCT_DCT, optional §7.17-§7.20.5 filters
(off by default), frozen CDFs,
`base_q_idx = 90` (qctx slice 0, `qlookup(90) = 528`). Every superblock of
64x64 and 640x360 satisfies §5.20.3.2's `has_rows`/`has_cols` half-block test,
so no implied partition is needed. `supports_64x64_none()` rejects other
geometries with a precise reason rather than writing an invalid tree.

* Luma residual: `BLOCK_64X64` → `TX_64X64` → `Adjusted_Tx_Size` 32x32,
  duplicate 2x2.
* 4:2:0 chroma: `BLOCK_32X32` → `TX_32X32`.
* 4:2:2 chroma: `BLOCK_32X64` → `TX_32X64` → adjusted 32x32, duplicate
  vertically only; nominal shifts `(6,12)`.
* 4:4:4 chroma: `BLOCK_64X64` → `TX_64X64` → adjusted 32x32, duplicate 2x2.
* 4:0:0: no chroma plane, no `read_intra_uv_mode()`, no chroma residual.

The encoder reconstruction is produced from the *same* quantised levels and
the normative dequant/inverse transform, so the gate can require 0 diffs
rather than PSNR.

## Gate

```sh
./src/encoder/format/chroma/build_and_test.sh   # unit test + CLI
./src/encoder/format/chroma/gate.sh             # dav2d --strict 1 + recon 0 diffs

# Single stream, with the reader's view of the sequence header:
"${TMPDIR:-/tmp}/av2enc_chroma_build/chroma_still" \
    tests/data/basic_64x64_1f.y4m -o out422.ivf --format 422 --inspect
```

The gate writes one stream per format/size, runs
`tools/validate_dav2d.sh`, decodes again with `--strict 1`, compares every
plane sample with `compare_y4m.py`, and checks the pinned goldens. Current
output (dav2d `f9dafbc`, macOS arm64 host):

| Format | Visible | TD+SEQ+CLK bytes | Tile bytes | IVF MD5 | Decoded MD5 | Recon |
|---|---:|---:|---:|---|---|---|
| 4:2:2 | 64x64 | 798 | 776 | `18d2d0bfc237eb41d9e13eb02bd744af` | `45209387c0d522fc24beaca3ad6d6c8d` | 0 diffs |
| 4:2:2 | 640x360 | 13227 | 13204 | `575b7dca4d546664cd7ba3db23275e97` | `032962273f91ed4fa69060f5a054e8ad` | 0 diffs |
| 4:4:4 | 64x64 | 912 | 891 | `f2daaaa1cee5588ec8b26b061be683c8` | `f625ca5025f64eedbd7a31f4f33874d3` | 0 diffs |
| 4:4:4 | 640x360 | 14775 | 14752 | `d8ed7538c85d39de1ae9d499c950ef60` | `15a98e3b0b3dada22a2011f5a25f8f9e` | 0 diffs |
| 4:0:0 | 64x64 | 465 | 445 | `d76fec5d9902caa1da7a8486bf6b37a2` | `bc6a0f75fad798268cac4a84f7c796b3` | 0 diffs |
| 4:0:0 | 640x360 | 8282 | 8260 | `2e0289b56af5aa6ae3addab7bdb91be6` | `0fb733b7f390c56f8aa991e46b61d11d` | 0 diffs |

4:2:0 is included in the golden file as a sanity case (`9b3cfdc6…`,
`78c7d6d4…`). `goldens/chroma_goldens.txt` is the machine-readable form;
regenerate with `CHROMA_UPDATE_GOLDENS=1 ./src/encoder/format/chroma/gate.sh`.

The gate prints `PASS: chroma format gate (4:2:2, 4:4:4, 4:0:0; 64x64 and
640x360)` on success. Test media is local (`tests/data/`); a missing file
returns exit 77, not a false pass. The same writer was also checked manually at
100x100 (partial right edge) and 101x99 (odd width and height) for all four
formats: `dav2d --strict 1` decoded every stream and the reconstruction was
bit-exact. Edge prediction clamps the above/left reference reads to the coded
plane (`MiCols*4 >> subX`, `MiRows*4 >> subY`), not the 64x64-padded buffer;
using the padded extent changes the DC average of a right/bottom edge block by
1 LSB.

## Exact M1/M2 core call sites (not edited)

These are the places a future integration must change. Line numbers are for
the tree at the time of writing; the function names are the stable anchor.

### M1

| Site | Current code | Required change |
|---|---|---|
| `src/encoder/m1/still_encoder.h:10-16` | `StillPicture { y,u,v }` is 8-bit 4:2:0 | add a `chroma::ChromaFormat`/`FrameShape`, or accept `chroma::ChromaPicture` |
| `src/encoder/m1/still_encoder.h:18-67` | `StillOptions` has no chroma field | carry `ChromaFormat`; default 4:2:0 |
| `src/encoder/m1/still_encoder.cpp:204-206` | `geometry_.has_chroma = true; geometry_.chroma_subsampling_x/y = true;` | `FrameShape shape = chroma::make_frame_shape(width,height,fmt); geometry_.has_chroma = !shape.monochrome; geometry_.chroma_subsampling_x = shape.info.subsampling_x; …` |
| `src/encoder/m1/still_encoder.cpp:1456-1465` | `encode_still_picture` rejects odd dimensions with “M1 4:2:0 requires even dimensions” | call `chroma::validate_encoder_format()`; odd dimensions are legal for 4:4:4/4:0:0 and only chroma-ceil at the edge |
| `src/encoder/m1/still_encoder.cpp:1503-1504` | deblock/filter frame `subsampling_x/y = 1` | use `shape.info.subsampling_x/y` |
| `src/encoder/m1/still_encoder.cpp:1642-1643`, `1785-1786` | filter frame `subsampling_x/y = 1` | same |
| `src/encoder/m1/still_syntax.cpp:310-312` | `Writer` sets `has_chroma=true`, `chroma_subsampling_x/y=true` | take the shape from `FramePlan` (add `chroma`, `subsampling_x/y`) |
| `src/encoder/m1/still_syntax.cpp:1555` | `assemble_still_frame` rejects non-even dimensions | same validation as above |
| `src/encoder/m1/still_headers.cpp:84` | `put_uvlc(writer, 0); // chroma_format_idc = 4:2:0` | `put_uvlc(writer, chroma::chroma_format_idc(config.chroma));` |
| `src/encoder/m1/still_headers.cpp:103-120` | writes `enable_sdp`, `cfl_ds_filter_index` unconditionally | `enable_sdp` and `cfl_ds_filter_index` are absent for Monochrome (§5.4.3/§5.4.5) |
| `src/encoder/m1/still_headers.cpp:135-142` | writes `enable_chroma_dctonly`, `enable_cctx`, `separate_uv_delta_q`, `base_uv_ac_delta_q`, `uv_ac_delta_q_enabled` unconditionally | all five are absent for Monochrome (§5.4.8) |
| `src/encoder/m1/still_headers.cpp:145-154` | `enable_ccso` / `ccso_unit_matches_sb_size` | keep the sequence bit; for Monochrome the per-plane chroma CCSO parameters are absent |
| `src/encoder/m1/still_headers.cpp:180-188` | `have_chroma_deblock = apply_deblock_y_vert || apply_deblock_y_horz` | also require `NumPlanes > 1`: §5.18.5.2 does not code `apply_deblocking_filter[2..3]` for Monochrome |
| `src/encoder/m1/still_headers.h:58-86` | `StillConfig` has no chroma field | add `ChromaFormat chroma` and `ChromaFormatInfo`; derive the `Monochrome` conditionals from `chroma::tool_legality()` |
| `src/encoder/m1/still_encoder.cpp:218-219` | `chroma_stride_ = luma_stride_/2; chroma_padded_height_ = luma_height_/2` | `>> shape.info.subsampling_x/y` |
| `src/encoder/m1/still_encoder.cpp:226-227` | `recon_[1/2].resize(chroma_stride_, chroma_padded_height_, 0)` | use `FrameShape::planes[1/2].padded_width/height` |
| `src/encoder/m1/still_encoder.cpp:1058` | `partition::chroma_plane_residual_size(geometry_, node.size)` | use `chroma::chroma_mi_size()`; `partition` returns invalid for 4:2:2 |
| `src/encoder/m1/still_encoder.cpp:1174-1181` | `chroma_mi_rows = mi_rows_/2; chroma_mi_cols = mi_cols_/2` | `>> sub_y` / `>> sub_x` (uvCfL grid) |
| `src/encoder/m1/still_encoder.cpp:1465` / `src/main.cpp:525-529` | CLI even-dimension guard | replace with `chroma::validate_encoder_format` + `chroma::supports_…` |
| `src/main.cpp:573-665` | `av2enc::still::StillPicture` built from `frame.y/u/v`; `encode_still_picture` | fill the new format field from the Y4M/CLI; conversion can reuse `chroma::make_chroma_picture` |
| `src/encoder/m1/still_encoder.cpp` block coders | chroma TX sizes are the 4:2:0 values (e.g. 16x16 for 32x32 luma) | use `chroma::chroma_mi_size(luma_bsize, fmt)`; `chroma_residual::make_tx_shape()` for 32x64/64x64 |
| `src/encoder/m1/still_syntax.cpp` chroma contexts | `mi_cols_/2`, `mi_rows_/2`, `>>1` shifts | `mi_cols_ >> sub_x`, `mi_rows_ >> sub_y`, `plane_origin_x/y()` |

### M2

| Site | Current code | Required change |
|---|---|---|
| `src/encoder/m2/video_frame.h:19-30` | `VideoInputFrame` is 8-bit 4:2:0 | add format/shape, or accept `chroma::ChromaPicture` |
| `src/encoder/m2/video_frame.h:76` | `encode_video_sequence` has no chroma option | add it to `VideoOptions`; default 4:2:0 |
| `src/encoder/m2/video_frame.cpp:285-293` | `chroma_width_ = width_/2; chroma_height_ = height_/2; geometry_.chroma_subsampling_x/y = true;` | `chroma::make_frame_shape()`; 4:2:2 keeps full height, 4:4:4 full width/height |
| `src/encoder/m2/video_frame.cpp:297-299` | `recon_[1/2].resize(mi_cols_*2, mi_rows_*2, 0)` | size chroma recon from `FrameShape::planes[1/2].padded_width/height` |
| `src/encoder/m2/video_frame.cpp:508` | `inter::mc::predict_block_420(...)` | select the predictor from the format; 4:2:2/4:4:4 need the chroma MC shapes |
| `src/encoder/m2/video_frame.cpp:519-522` | ME planes use `chroma_width_/chroma_height_` | use `FrameShape::planes[1/2]` coded/padded dims |
| `src/encoder/m2/video_syntax.cpp:404-406` | `geometry_.has_chroma = true; chroma_subsampling_x/y = true` | take from `FramePlan` |
| `src/encoder/m2/video_syntax.cpp:1964-1976` | `cr = r/2; cc0 = c/2; chroma_mi_rows = mi_rows_/2; chroma_mi_cols = mi_cols_/2` | `>> sub_y` / `>> sub_x` throughout the chroma context grid |
| `src/encoder/m2/video_syntax.cpp:2058` | “M2 4:2:0 requires even dimensions” | `chroma::validate_encoder_format` |
| `src/encoder/m2/video_headers.cpp:101` | `put_uvlc(writer, 0); // chroma_format_idc = 4:2:0` | `chroma::chroma_format_idc(config.chroma)` |
| `src/encoder/m2/video_headers.cpp:140-147` | hard-wires `enable_intra_edge_filter = 1`, `enable_cfl_intra = 1`, `cfl_ds_filter_index` | Monochrome must clear CfL and omit `cfl_ds_filter_index`; derive from `tool_legality()` |
| `src/encoder/m2/video_headers.cpp:194` | `enable_chroma_dctonly = 1` unconditionally | bit is absent for Monochrome |
| `src/encoder/m2/video_headers.h:21-40` | `VideoConfig` has no chroma field | add `ChromaFormat`; apply the Monochrome conditionals |
| `src/encoder/m2/video_gate.cpp:293` | `encode_video_sequence(inputs, options, …)` | pass the new format option |
| `src/encoder/partition/partition.h:145-147` | comment: “4:2:2 is not modelled” | replace with `chroma::FrameShape` or extend `chroma_plane_residual_size` |
| `src/encoder/partition/partition.cpp:944-956` | `chroma_plane_residual_size()` returns `kInvalid` for 4:2:2 | use `fmt::chroma_residual_size()` / `chroma::chroma_mi_size()` |
| `src/io/y4m.h:13-20` | input reader accepts only 8-bit 4:2:0 | the gate converts with `chroma::make_chroma_picture`; a native 4:2:2/4:4:4 reader is future work |

`src/encoder/format/format.*` already has the correct `chroma_format_idc`,
`Subsampled_Size`, `num_planes` and 4:2:0↔4:2:2/4:4:4/4:0:0 conversion; the M1/M2
cores should call it rather than re-deriving these values.

## Files

| File | Purpose |
|---|---|
| `chroma_format.h/.cpp` | Table 6.2, Monochrome/subsampling inference, `ChromaMiSize`, geometry, tool legality, checklist |
| `chroma_headers.h/.cpp` | §5.4.1 sequence-header writer/reader, OBU/TD/SEQ/CLK assembly, Monochrome conditionals |
| `chroma_residual.h/.cpp` | `TxShape`, `Adjusted_Tx_Size`, nominal `Transform_Shift`, forward/inverse DCT, quant/dequant |
| `chroma_still.h/.cpp` | minimal 64x64-`PARTITION_NONE` still writer, 4:2:0→format conversion, recon dump |
| `chroma_still_main.cpp` | `chroma_still` CLI used by the gate |
| `chroma_test.cpp` | 2,531-check CPU unit test |
| `compare_y4m.py` | exact per-plane recon-vs-decoder comparator |
| `build_and_test.sh` | builds test + CLI in `${TMPDIR}/av2enc_chroma_build` |
| `gate.sh` | dav2d `--strict 1` + 0-diff + golden gate |
| `goldens/chroma_goldens.txt` | pinned IVF/decoded MD5s |

## In-loop filters (§7.17-§7.20.5)

The writer now carries the full filter stage: `read_deblocking_filter()`,
`gdf_params()`, `cdef_params()`, `lr_params()`, `ccso_params()` in the frame
header, plus the per-superblock `read_lr()`/`read_cdef()`/`read_ccso()` tile
symbols.  `--still-deblock`, `--still-cdef`, `--still-ccso`, `--still-wiener`,
`--still-chroma-lr`, `--still-pc-wiener` and `--still-gdf` are accepted on
`--still --chroma 422|444|400`; every default (all filters off) stream is
byte-identical to the previous writer.

`chroma_filters.{h,cpp}` owns the plan/application.  The 8-bit samples are
carried in the uint16_t filter container with `BitDepth = 8` and run through
the same depth-parameterised cores as the 10-bit writer:

| Filter | Core | Plan/reuse |
|---|---|---|
| §7.17 deblock | `deblock_core.h` (`deblock_frame16`, BitDepth 8) | `ChromaFilterBlock` tx sizes |
| §7.18 CDEF | `cdef_core.h` (`cdef_frame16`, BitDepth 8) | `hbd::plan_cdef` |
| §7.19 CCSO | `ccso_core.h` (`ccso_frame16`, BitDepth 8) | `hbd::plan_ccso` |
| §7.20 Wiener NS | `wiener_core.h` | `hbd::plan_lr` + `apply_wiener16` |
| §7.20.4 PC-Wiener | `tables/pc_wiener.*` | `hbd::plan_pc_wiener16` + `apply_pc_wiener16` |
| §7.20.5 GDF | `tables/gdf.*` | `hbd::plan_gdf` + `apply_gdf16` |

The shared "16" entries accept `BitDepth = 8` for the non-4:2:0 geometry
(the uint8_t entries stay scoped to the 4:2:0 M1/M2 paths).  The container only
stores clipped samples, so the arithmetic is identical to an 8-bit buffer — the
dav2d gate is the evidence.

The writer runs two passes when filters are enabled: the first reconstructs
and collects the per-leaf `luma_eob_zero`/tx-size metadata, `run_chroma_filters`
plans and applies the chain in normative order (deblock -> CDEF -> CCSO -> LR ->
GDF), and the second re-codes the same coefficients with the filter syntax at
the §5.20.10 hook points.  Filters require `disable_cdf_update = 1`; the filter
symbol CDFs are frozen (default).  Filters are not wired to the 8-bit chroma GPU
block stage and are rejected there by name.

`--still-chroma-lr --still-deblock` is rejected by name at 4:2:2/4:4:4 for the
same pinned-decoder chroma-lossless-mask aliasing reason as the 10-bit writer.

Gate: `src/formats/format_gate.sh` runs each filter alone plus the full chain at
64x64 and 640x360 for 4:2:2/4:4:4/4:0:0 with `dav2d --strict 1` and a 0-diff
reconstruction comparison.  Two textured-source cases make the Wiener/LR search
win units so the §5.18.7.11 row and §5.20.10.4 symbols are exercised.  At the
pinned `--qp 90` the natural fixtures leave every Wiener/chroma-LR unit
unfiltered (the plan then signals no restoration at all), so those cases are
byte-identical to the default stream; the 10-bit gate at q303 covers the same
plan/apply code at a coarser quantiser.

Residual gaps (still gated):

* 8-bit chroma LR finds no improving unit at `--qp 90` on the gate media; its
  header/tile rows are shared with the exercised luma LR path.
* The 8-bit writer keeps the §8.3.1 slice-0 coefficient CDFs, so `--qp` stays
  pinned at 90; a qindex above 90 would need slices 1..3.
* 4:2:2/4:4:4 chroma LR uses the chroma-only Wiener subset (the CfL luma taps
  are not decoder-exact with the pinned decoder), exactly like the 10-bit
  writer.

## Limitations

* Streams are minimal format/gate streams, not a new production mode. They use
  one 64x64 `PARTITION_NONE` block per superblock and DC-only intra, so they do
  not exercise 4:2:2/4:4:4 partition, directional intra, CfL, filter RDO or
  motion.  The in-loop filters are driven by one fixed per-frame plan.
* `supports_64x64_none()` intentionally rejects geometries where the half-block
  test fails (for example 640x322); use the M1 partition tree for those once it
  is chroma-aware.
* The minimal still writer's EOB switch covers `eob_multisize` 2..6, i.e.
  transform shapes 16x4 and larger. 4x4/4x8 chroma residuals are reachable in
  the format layer and the transform/dequant functions, but not emitted by
  this writer; the M1 CDF set already has the `eob_pt_16`/`eob_pt_32` tables
  for a future integration.
* 8-bit only. 10-bit already has a separate path; 12-bit is not signalable in
  AV2 v1.0.0.
* No packed 10-bit input, no chroma interpolation, no colour conversion.

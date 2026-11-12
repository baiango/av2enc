# Palette intra — AV2 §5.20.8 / §7.13.4

Wave-3 screen-content tool. Spec-only implementation of AV2 v1.0.0 palette
intra for **8-bit 4:2:0**, with a CPU reference, a CUDA twin, and a
bit-exact `dav2d` gate. No AVM/SVT/clave/av2-turbo code.

Status: **`dav2d` f9dafbc decodes the gate stream at `--strict 1` and the
decoded Y4M is byte-identical to the encoder reconstruction.** Palette 3
colours, two 64x32 leaves, 112-byte tile / 132-byte frame.

## Spec map

| Clause | Used for |
|---|---|
| §5.20.8.1 | `palette_mode_info`: `has_palette_y`, size, colour cache/literal/delta |
| §5.20.8.2 | `get_palette_cache` (above/left MI palettes, interleaved) |
| §5.20.8.4 | `palette_tokens`: direction, identity rows, colour indices, onscreen clamping and edge replication |
| §5.20.8.5 | `get_palette_color_context` (ColorOrder + ColorContextHash) |
| §6.19.8.1/.3 | semantics: identity rows 0/1/2, sorted palette, cache reuse |
| §7.13.4 | palette prediction (`ColorMapY[y*4+i][x*4+j]`) |
| §8.3.2 | palette CDF selection (`Table 8.1`, `Palette_Color_Context`) |
| §9.3 | default CDFs (generated from `spec/tables/default_cdf.json`) |

### Luma-only (important)

AV2 v1.0.0 `palette_mode_info()` is gated on `PlaneStart == 0` and contains no
U/V palette size, colour or token syntax. **There is no chroma palette.** Chroma
uses normal intra prediction (DC/CfL) after the palette-predicted luma is
reconstructed. `chroma_rule()` records this; the tests assert it. This is the
main difference from AV1-style palette implementations.

### Transform-skip residual

A palette block always codes a residual (intra `skip_flag` is only signalled for
IBC blocks). "Palette-only" therefore means `eob = 0` on all three planes, i.e.
transform-skip; the reconstruction equals the palette prediction. The CPU API
also exposes `palette_residual()` for a caller that wants to code a residual on
top of the palette.

## Files

```text
src/encoder/palette/
  palette_core.h     host+device normative core: constants, §5.20.8.5 context,
                     ColorOrder inverse, §7.13.4 prediction, identity rows,
                     nearest colour, edge replication helper
  palette.h/.cpp     CPU reference: colour selection, cache, map, exact §8.2
                     encode/decode round-trip, bit estimation
  scc_cdf.h          generated palette/IBC CDFs (gen_scc_cdf.py)
  gen_scc_cdf.py     generator from spec/tables/default_cdf.json
  scc_stream.h/.cpp  shared minimal stream writer (palette + IBC gates)
  palette_test.cpp   3506 checks: context, cache, direction, partial blocks,
                     delta limits, cache round-trip, encode/decode parity
  palette_gate.cpp   minimal 64x64 palette-only IVF + reference Y4M
  build_and_test.sh  standalone build, tests, dav2d gate
src/cuda/palette/
  palette.h/.cu      histogram, index map, prediction, colour contexts
  palette_test.cu    CPU/GPU parity on 8x8..64x64, partial, dither
  build_and_test.sh  nvcc sm_86 build/run
```

## API (CPU)

```cpp
using namespace av2enc::palette;

PaletteState state;                 // per-MI PaletteSizes/PaletteColors
state.reset(mi_cols, mi_rows);

std::uint8_t cache[16];
int n = get_palette_cache(state, mi_row, mi_col, avail_l, cache);

PaletteParams params;               // size 2..8, exact, dither, direction
PaletteResult pr;
analyze_palette(src, stride, block_w, block_h, visible_w, visible_h,
                cache, n, params, pr, error);

PaletteCdfState cdfs; init_palette_cdfs(cdfs);
encode_palette_y(encoder, cdfs, pr, block_w, block_h, visible_w, visible_h,
                 cache, n, error);           // real §8.2 encoder
decode_palette_y(decoder, cdfs, ...);        // local decoder for tests/parity

predict_palette_block(pr, block_w, block_h, dst, dst_stride);
palette_residual(src, src_stride, pr, block_w, block_h, residual, res_stride);
```

Palette sizes and colours are sorted ascending, matching the decoder's
`sort(palette_colors_y, 0, PaletteSizeY-1)`. The encoder picks the smallest
`palette_num_extra_bits_y` that makes the delta run representable; if a colour
set is not representable (e.g. 126/255 with the dynamic `CeilLog2(range)`
shrink), the analysis merges colours until it is.

Direction is only signalled when `block_w < 64 && block_h < 64`; 64-wide/tall
blocks are pinned to rows. Identity rows 0/1/2 are chosen exactly as §6.19.8.3
(2 = same as the previous line, only legal for the second line onward).

Partial blocks: only the visible area is read/analysed; the colour map is
replicated to the full coded block with §5.20.8.4's column-then-row rule.

## Gate

```sh
./src/encoder/palette/build_and_test.sh
```

The script emits two 64x64 streams, each with a reference Y4M holding the
writer's reconstruction, runs `dav2d`, and requires the decoded Y4M to be
identical:

* `--single`: one PARTITION_NONE 64x64 palette block (direction is not
  signalled at 64x64), 6 colours, 134-byte tile / 155-byte frame.
* default: HORZ with two 64x32 palette leaves; the second leaf exercises the
  above-palette cache, 112-byte tile / 132-byte frame.

```
leaf 0: palette size=3 colours=3 bits=448
leaf 1: palette size=3 colours=3 bits=448
palette gate: recon diff=0, tile=112, frame=132 bytes
single 64x64: palette size=6 colours=6 bits=1064
palette gate: recon diff=0, tile=134, frame=155 bytes
OK: dav2d decoded both palette gates bit-exactly (f9dafbc)
```

Decoded MD5 for both streams (same source frame):
`e4d958b01d1bd8c566540a71536c93e0`.

## Quality on synthetic screen content

The gate content is a 64x64 screen-content frame (flat fills, vertical stripes,
icon squares) with 3 distinct luma values per 64x32 leaf. The palette
prediction is exact, so the decoded luma has 0 error and chroma is flat DC.
The frame is 132 bytes (112-byte tile).

The repo's M1 encoder (DC/partition/CfL, QP 90, deblock) on the same content
produces a 266-byte temporal unit at 52.96 dB Y PSNR (54.72 dB average,
`ffmpeg psnr` against the exact source). This is a one-frame synthetic still,
not a natural-video BD-rate claim.

## CUDA

`src/cuda/palette/` has kernels for the per-sample 256-bin histogram, nearest
index map + edge replication, §7.13.4 prediction, and §5.20.8.5 colour
contexts. They call `palette_core.h`, so the output is bit-exact with the CPU.
Colour selection (median cut/exact) stays on the CPU because it is per-block
control work, not pixel work. Test on the GPU box:

```sh
GPU_ARCH=sm_86 ./src/cuda/palette/build_and_test.sh
```

## Integration notes (not wired)

`src/encoder/m1/still_syntax.cpp` and `src/main.cpp` were not modified. To wire
palette into M1:

1. In `still_headers.cpp`, allow `allow_screen_content_tools=1` in the frame
   header (the sequence header already has `single_picture_header_flag=1`, so
   §5.4.7 infers `seq_force_screen_content_tools=SELECT_SCREEN_CONTENT_TOOLS`
   and the frame codes `allow_screen_content_tools` + `force_integer_mv`).
2. In `mode_info`/`intra_frame_mode_info`, after `read_intra_uv_mode`, code
   `palette_mode_info()` when `MiSize >= BLOCK_8X8`, `w,h <= 64`, and
   `allow_screen_content_tools`.
3. After the mode info, code `palette_tokens()` before `read_block_tx_size`.
4. Maintain a `PaletteState` (per-MI sizes/colours) in the writer and read the
   cache exactly as `get_palette_cache` does; do not skip the cache bits when
   the decoder has neighbours with palettes.
5. Palette uses the normal 3-plane residual path with `eob=0` for palette-only
   blocks. There is no chroma palette branch.

No `CMakeLists.txt` changes are needed for the component tests; the root build
does not link these files.

## Limits

* 8-bit 4:2:0 only.
* Palette size 2..8 (the syntax limit).
* 128x128 superblocks and SDP/chroma-partition trees are not handled by the
  gate writer (64x64 SB, NONE/HORZ/VERT only).
* The CUDA test has not been run in this macOS workspace (no `nvcc`); the
  script is ready for the sm_86 box.

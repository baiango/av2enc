# Intra block copy (IBC) — AV2 §5.20.5.4 / §6.19.7.12

Wave-3 screen-content tool. Spec-only implementation of AV2 v1.0.0 intra block
copy for **8-bit 4:2:0**, 64x64 superblocks, integer block vectors. CPU
reference + CUDA twin + bit-exact `dav2d` gate. No AVM/SVT/clave/
av2-turbo code.

Status: **`dav2d` f9dafbc decodes both gate streams at `--strict 1`.** The
plain gate is IBC + DC only (53-byte frame); the pattern gate uses a palette
source so the copied 64x32 region is byte-identical to the source region
(97-byte tile / 119-byte frame).

## Spec map

| Clause | Used for |
|---|---|
| §5.20.5.3/.4 | `intra_frame_mode_info`: `use_intrabc`, `read_skip`, `intrabc_mode`, `intrabc_drl_mode`, `morph_pred` |
| §6.19.5.4 | `intrabc_mode=1` = NEARMV (no MVD); `morph_pred` scope |
| §6.19.7.12 | `is_mv_valid`: tile bounds, local buffer, `INTRABC_DELAY_SB64`, wavefront |
| §6.17.3.3 | `allow_global_intrabc` / `allow_local_intrabc` semantics |
| §5.20.5.1 | `update_ibc_buffers`, `ibc_buffer_index`, `IBCCoded` |
| §7.13.3.17 | motion-vector scaling (luma/chroma plane offset) |
| §7.13.3.18 | integer-pel block copy |
| §8.3.2 | `use_intrabc`/`skip_flag`/`intrabc_mode` CDF selection |

## Block vectors and validity

A BV is in **1/8 luma samples** (`BlockVector{row, col}`), as in the bitstream.
`plane_offset()` implements §7.13.3.17: for 4:2:0 the chroma offset is the luma
offset divided by two with the spec's rounding (`-2561/8` luma -> `-321`
samples, chroma -> `-161`).

`is_bv_valid` implements §6.19.7.12 exactly:

1. source block (including the fractional bottom/right border) must be inside
   the tile;
2. if `allow_local`: `check_valid_local_ibc` at the top-left and bottom-right
   against the four 64x64 ring buffers and the `IBCCoded` MI grid (a valid
   local source returns immediately);
3. if `!allow_global`: reject;
4. global delay: the source SB64 index must be at least
   `INTRABC_DELAY_SB64` behind the active one;
5. global wavefront: `srcSb64Col < activeSb64Col - 4 + 5*(activeRow-srcRow)`.

`IbcBufferState` owns the ring entries, the current buffer position and the
`IBCCoded` grid; call `update_buffers()` at the start of each block and
`mark_coded()` after reconstruction, exactly like §5.20.5.1 / `decode_block`.

The decoder's `is_mv_valid` also checks the ±1 reference border
(`is_offset_mv_valid`) used by morph prediction. This module does not implement
morph prediction, but the gate keeps the source block **interior** so the
pinned decoder's strict check passes; a source touching the frame edge is
rejected by `dav2d --strict 1` even for integer BVs.

## Candidate stack (NEARMV)

`build_stack()` mirrors the §7.12.2.20 IBC candidates:

```
(0, -SbHeight), (-SbWidth - 256, 0), (0, -BlockHeight), (-BlockWidth, 0)
```

in that order, followed by already-IBC neighbour BVs (above, left, below-left,
above-right, deduplicated). The stack is capped at
`max_bvp_drl_bits_minus_1 + 2`; the gate uses the sequence-header value 0, so
there are two entries and `intrabc_drl_mode` chooses index 0 or 1. The defaults
are appended **without** pruning, like `add_to_ref_bv()`.

`intrabc_mode=1` (NEARMV) copies `RefStackMv[RefMvIdx]` with no MVD. `NEWMV` /
`read_mv()` and `morph_pred` are out of scope for this module.

## Files

```text
src/encoder/ibc/
  ibc_core.h     host+device core: plane offset, BV geometry/global validity,
                 local buffer check, copy, SAD
  ibc.h/.cpp     CPU reference: buffer state, full is_mv_valid, candidate stack,
                 search, copy, residual, synthetic screen generator
  ibc_test.cpp   36 checks: offsets, stack order, delay/wavefront/bounds,
                 local ring eviction, search, copy, chroma, lossless PSNR
  ibc_gate.cpp   768x128 gate: palette source + IBC copy; IVF + reference Y4M
  build_and_test.sh  standalone build, tests, dav2d gate
src/cuda/ibc/
  ibc.h/.cu      copy, SAD batch, global validity, brute-force search
  ibc_test.cu    CPU/GPU parity for copy, SAD, validity and search
  build_and_test.sh  nvcc sm_86 build/run
```

## API (CPU)

```cpp
using namespace av2enc::ibc;

IbcConfig config;                 // sb_size, global/local, DRL bits
IbcBufferState state; state.reset(mi_cols, mi_rows);
FrameView frame = screen.view();  // reconstructed planes

bool ok = is_bv_valid(frame, config, state, mi_row, mi_col, bsize, bv, &why);

BvStack stack = build_stack(frame, config, state, mi_row, mi_col, bsize,
                            neighbours);
Match m = search_block(frame, config, state, mi_row, mi_col, bsize, options);
// options.stack_only = true -> NEARMV only (no MVD syntax)
// options.step = 4          -> full raster search

predict_copy(frame, mi_row, mi_col, bsize, m.bv, w, h, copy);
copy_block(...); copy_then_residual(...);
```

`search_block` only proposes BVs that pass `is_bv_valid`, so every match is
decodable under the same frame configuration. Transform-skip is `skip_flag=1`:
no residual is coded and the copy is exact; `copy_then_residual` is provided
for a future non-skip residual path.

## Gate

```sh
./src/encoder/ibc/build_and_test.sh
```

The gate emits a 768x128 frame (12x2 64x64 superblocks, HORZ leaves). The
64x32 block at (704,64) is IBC with BV `(-320,0)` (default candidate index 1),
pointing at (384,64). The script runs two variants:

* `--plain`: the source is flat DC, so the stream contains **only IBC and DC
  intra** (53-byte frame). This is the tool-isolation gate.
* default (`pattern`): the source is a palette-coded checkerboard, so the
  copied region is observable (97-byte tile / 119-byte frame) and is compared
  byte-for-byte with the source.

The source is interior (not on any frame edge) because `dav2d --strict 1`
enforces the §6.19.7.12 reference-pixel border even for integer BVs.

```
mode=plain   source palette=0 colours, copy diff=0, tile=31, frame=53 bytes
mode=pattern source palette=3 colours, copy diff=0, tile=97, frame=119 bytes
OK: dav2d decoded both IBC gates bit-exactly (f9dafbc)
```

Decoded MD5: plain `2e2237db5ce70eb5f7d2c4d96212c73b`, pattern
`4bb2166a705196d103418f8f6db673c4`.

The build script compares dav2d's Y4M with the writer's reconstruction
byte-for-byte (all three planes).

## CUDA

`src/cuda/ibc/` has kernels for block copy (luma + 4:2:0 chroma), batched SAD,
global BV validation and a brute-force search (one thread per candidate with
`atomicMin` packing). The device code calls `ibc_core.h`, so offsets, copy and
validity arithmetic are identical to the CPU. The local four-buffer ring stays
on the CPU (control work); the GPU copy only needs an already-validated BV.
Test on the GPU box:

```sh
GPU_ARCH=sm_86 ./src/cuda/ibc/build_and_test.sh
```

## Quality on synthetic screen content

The gate content is synthetic (repeated 64x32 tiles, flat fills, text-like
stripes). Both variants are byte-exact against `dav2d`, i.e. PSNR is infinite
on the luma region the tool reconstructs.

For a still-frame rate reference, the repo's M1 encoder (DC/partition/CfL,
QP 90, deblock) on the same 768x128 content produces a 1,674-byte temporal
unit at 57.66 dB Y PSNR (`ffmpeg psnr` vs the exact source), versus 119 bytes
and exact reconstruction here. This is one synthetic still, not a BD-rate
claim.

## Integration notes (not wired)

`src/encoder/m1/m1_syntax.cpp` and `src/main.cpp` were not modified. To wire
IBC into M1:

1. Frame header: `allow_screen_content_tools=1`; `intrabc_params()` with
   `allow_intrabc=1`, `allow_global_intrabc` / `allow_local_intrabc` as chosen,
   and `force_integer_mv=1`. Sequence header value
   `seq_max_bvp_drl_bits_minus_1` is already 0 in the M1 writer.
2. In `intra_frame_mode_info` (non-IBC branch first): code `use_intrabc` when
   `allow_intrabc && TreeType != CHROMA_PART && w,h <= 64 && MiSize != 64x64`.
   If 1, code `read_skip`, then `read_intrabc_info` (`intrabc_mode=1`,
   `intrabc_drl_mode` bits); do not code Y/UV mode or palette mode info.
3. Maintain the `IbcBufferState` and the candidate stack in the writer; the
   decoder derives both from the decoded stream.
4. `skip_flag=1` means no residual symbols. `skip_flag=0` uses the normal
   residual path with the IBC prediction in `CurrFrame`.
5. `allow_local_intrabc=1` requires the four-buffer ring and `IBCCoded`; keep
   it off until the writer models them exactly. The global path is arithmetic
   only and is what the gate exercises.

No `CMakeLists.txt` changes are needed for the component tests.

## Limits

* 8-bit 4:2:0, 64x64 SBs, integer BVs (`force_integer_mv=1`).
* `NEWMV`/MVD syntax, `morph_pred`, 128x128 SBs, 4:2:2/4:4:4 are not
  implemented.
* The CUDA test has not been run in this macOS workspace (no `nvcc`); the
  script is ready for the sm_86 box.

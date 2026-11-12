# Chroma (U/V) loop restoration — AV2 §5.18.7.11 / §5.20.10.5–.6 / §7.20.3

This directory owns the **chroma-specific** loop-restoration path that the
luma Wiener work in `../restoration/classes/` does not cover:

* §5.18.7.11 `LoopRestorationSize[1]` sizing and the `lr_chroma_use_*` flags;
* per-unit chroma type selection (`RESTORE_NONE` / `RESTORE_WIENER_NONSEP`);
* the 18-coefficient chroma layout (12 chroma taps + 12 luma taps sharing the
  last 12 coefficient slots), `Wiener_Ns_Taps_Present`, and subset selection;
* `wiener_ns_uv_sym` mirroring (coefficients `j` and `j+1` coded once for
  `j >= WIENER_NS_SHORT_COEFFS`);
* shared-U/V filters (one coefficient set used by both planes) and the
  per-plane mirrored forms;
* exact `read_lr_unit()` / `read_wienerns_filter()` writer and reader for
  planes 1 and 2, with pinned byte goldens;
* an encoder decision on **real reconstruction vs the source picture** on U
  and V, and the §7.20 apply path;
* a 2-frame M2 decoder gate and a 640x360 / 720p quality measurement.

PC-Wiener is luma-only.  §5.4.10 forces
`lr_tools_disable[1][RESTORE_PC_WIENER] = 1`; chroma units are therefore
`RESTORE_NONE` or `RESTORE_WIENER_NONSEP` only.  `gdf`/film grain are out of
scope.

## Why `lr_tools_uv_present` matters

`sequence_filter_config()` writes:

```text
lr_tools_disable[0][RESTORE_PC_WIENER]      f(1)
lr_tools_disable[0][RESTORE_WIENER_NONSEP]  f(1)
lr_tools_disable[1][RESTORE_PC_WIENER]      = 1
lr_tools_uv_present                         f(1)
if (lr_tools_uv_present)
  lr_tools_disable[1][RESTORE_WIENER_NONSEP] f(1)
else
  lr_tools_disable[1][RESTORE_WIENER_NONSEP] =
      lr_tools_disable[0][RESTORE_WIENER_NONSEP]
```

With `lr_tools_uv_present = 0`, chroma Wiener is *inferred* from luma.  A
stream that enables luma Wiener therefore also enables chroma Wiener even
though no chroma flag is coded.  The M2 `VideoRestorationTools::chroma_lr` flag
already maps to `lr_uv_present`; the gate writes it explicitly (`1`) so the
chroma tool set is independent of luma.  `test_uv_present()` checks both the
explicit and inferred forms.

## Unit sizing

`chroma_select_unit_size(desired, sb_size, sub_x, sub_y)` derives the exact
`lr_chroma_use_half_size` / `_max_size` / `_quarter_size` flags.  For 8-bit
4:2:0:

| `SbSize` | legal chroma `LoopRestorationSize[1]` (shift) |
|---|---|
| 64x64 | 256 (0), 128 (1), 64 (2), 32 (3) |
| 128x128 | 256 (0), 128 (1), 64 (2) |
| 256x256 | 256 (0), 128 (1) |

32 is the §6.17.7.7 minimum for 4:2:0 (`64 >> SubsamplingY`); 16 is rejected.
`LoopRestorationSize[2] = LoopRestorationSize[1]`.  Unit origins include the
+8 luma-line stripe offset used by §7.20.1, so a unit's sample region is not
the `unit_size` rectangle in plane coordinates; the decision uses the same
`lr_block_context()` unit assignment as the decoder.

## Filter layout and `uv_sym`

```text
j =  0..5   chroma tap groups (Wiener_Ns_Config_Uv[][2])
j =  6..17  chroma-from-luma taps: coefficient i+6 belongs to config tap i
```

Subsets (`wiener_ns_length`, chroma has `numSubsets = 3`):

| subset | present coefficients | coded values without `uv_sym` | with `uv_sym` |
|---|---|---:|---:|
| 0 | 0..5 | 6 | 6 (uv_sym is not coded) |
| 1 | 0..9 | 10 | 8 |
| 2 | 0..17 | 18 | 12 |

`uv_sym` mirrors pairs `(6,7), (8,9), ...` within the present set.  The
decoder copies the first value of each pair; the encoder codes it once.
`chroma_mirror_filter()` applies round-half-away-from-zero to the pair, and
`chroma_filter_validate()` rejects non-mirrored pairs when `uv_sym` is set,
out-of-range taps, and non-zero taps outside the subset.

The shared `filters_stream` writer requires all subset>0 pairs to be
symmetric even when `uv_sym == 0`; the spec does not.  This module therefore
carries its own §5.20.10.6 chroma writer/reader, and the tests pin its bytes
against `filters_stream::lr_write_unit()` for the symmetric case (which is
the intersection of the two).

## API

```cpp
#include "encoder/filters/lr_chroma/chroma_lr.h"
namespace lr = av2enc::lr_chroma;

// Header row (§5.18.7.11).
lr::ChromaUnitSize size;
lr::chroma_select_unit_size(64, 64, 1, 1, &size, &error);
lr::chroma_write_unit_size(writer, size, &error);

// Decision on the real reconstruction (U, V) vs source.
lr::ChromaLrInput in;            // curr/cdef + luma curr/cdef + source
in.plane_width[0] = in.plane_width[1] = cw;
in.plane_height[0] = in.plane_height[1] = ch;
in.curr[0] = in.cdef[0] = recon_u;   in.curr_stride[0] = cw;
in.curr[1] = in.cdef[1] = recon_v;   in.curr_stride[1] = cw;
in.luma_curr = in.luma_cdef = recon_y; // CfL taps read luma
in.luma_curr_stride = in.luma_cdef_stride = luma_w;
in.source[0] = src_u; in.source[1] = src_v;
in.source_stride[0] = in.source_stride[1] = cw;
in.mi_cols = (luma_w + 3) / 4; in.mi_rows = (luma_h + 3) / 4;
in.sub_x = in.sub_y = 1; in.sb_size = 64;

lr::ChromaLrOptions options;
options.lambda_bits = rd_lambda;   // 0 = pure distortion
lr::ChromaDecision decision;
lr::chroma_decide(in, options, &decision, &error);

// Header + tile state.
fs::LrParams lr_params;
lr::chroma_make_lr_params(decision, &lr_params, &error);
fs::LrState state;
lr::chroma_init_tile_state(decision, in, &state, &error);

// Per superblock, before read_partition():
lr::chroma_write_read_lr(encoder, state, mi_row, mi_col, sb_mi, sb_mi,
                         tile_mi_row_start, tile_mi_col_start, decision,
                         &error);

// Apply with the reference §7.20 core.
std::uint8_t* dst[2] = {out_u, out_v};
int stride[2] = {cw, cw};
lr::chroma_apply(in, decision, dst, stride);
```

`chroma_measure_header_bits()` returns the exact `lr_params()` bit count of
the chroma-only configuration (8 bits for a 64x64 SB with both chroma planes
at shift 2).  `chroma_measure_unit_bytes()` serialises one unit through a
fresh §8.2 coder (used for RD and the goldens).

## Exact M1/M2 call sites

The module is self-contained; the core files below are **not** edited by this
workstream.  These are the exact insertion points.

1. **Sequence header** — `src/encoder/m2/restoration/video_restoration.cpp`,
   `make_sequence_filter_config()`:
   `config.lr_uv_present = tools.chroma_lr;` and
   `config.lr_disable_chroma_wiener_ns = false;` (already present), then
   `fs::write_sequence_filter_config()`.
   The M1 equivalent is `src/encoder/m1/filters_stream/filters_stream.cpp`,
   `write_sequence_filter_config()`.

2. **Frame header** — `src/encoder/m2/video_headers.cpp`,
   `write_frame_header_syntax()`, after the deblocking/delta-q row and before
   `tx_mode_select` (the `lr_params()` slot).  Use
   `av2enc::video::restoration::write_lr_params()` with
   `lr_tools_disable[1][RESTORE_WIENER_NONSEP] = false`, then
   `chroma_make_lr_params(decision, &lr_params)` supplies the chroma shift
   and frame types.  M1's equivalent is
   `src/encoder/m1/filters_stream/filters_stream.cpp::write_lr_params()`.

3. **Tile, per superblock** — `src/encoder/m2/video_syntax.cpp`,
   `Writer::run()`, immediately before `write_node()` (before
   `read_partition()`).  Write luma units first if luma restoration is
   enabled, then `chroma_write_read_lr(enc_, state, node.mi_row,
   node.mi_col, sb_mi, sb_mi, tile_mi_row_start, tile_mi_col_start,
   decision, &error)`.  For M1 the same call goes in
   `src/encoder/m1/still_syntax.cpp::Writer::run()` before `write_node()`.
   `read_lr()` is **once per superblock**, not once per coding block.

4. **Apply** — `src/encoder/m2/restoration/video_restoration.cpp`
   `apply_lr_frame()` / `src/encoder/m2/post/loop_filters.cpp`: feed the
   decision's unit types/coefficients into `LoopRestorationParams` and call
   `loop_restoration_frame()`, or use `chroma_apply()` directly with the
   pre-CDEF `curr` and post-CDEF/CCSO `cdef` planes.

5. **Decision** — run `chroma_decide()` on the completed reconstruction
   (post-deblock `curr`, post-CDEF/CCSO `cdef`) against the source picture,
   then serialise the header and tile.  `decision.header_bits` +
   `decision.tile_bits` are the exact syntax cost; the alternative
   `sse`/`cost` fields drive the on/off and unit-size search.

The gate's `gate/vendor_video_syntax.py` performs exactly the per-superblock
insertion in a build-directory copy of `video_syntax.cpp` for the decoder gate.
The source SHA-256 is pinned, so a changed M2 writer fails the gate instead
of silently producing a different stream.

## Tests

```sh
./build_and_test.sh                 # 271 checks
SANITIZE=1 ./build_and_test.sh      # ASan + UBSan
WERROR=1 ./build_and_test.sh
```

Coverage:

* unit-size flags for 64/128/256 SBs, write/read round-trip, 4:2:0 minimum;
* present/coded coefficient counts, subset derivation, mirroring, range and
  `uv_sym` validation;
* pinned byte goldens for off, subset 0, subset 2 asymmetric, subset 2
  `uv_sym`, subset 1 `uv_sym`, plus read-back of every golden;
* byte equality with `filters_stream::lr_write_unit()` for the shared
  symmetric form;
* bank merge/`merged` round-trip;
* header bit-count golden and `wiener_write_lr_params()` round-trip;
* `lr_tools_uv_present` explicit and inferred sequence rows;
* decision on real reconstruction vs source (subset choice, non-zero filters,
  account of `sse`, header/tile bits), whole-plane `chroma_apply()` parity,
  shared-U/V selection for identical planes, and all-off identity.

The decision's `sse` is the exact §7.20 unit-region SSE: it equals the
whole-plane `chroma_apply()` SSE (the test asserts this).  Earlier versions
used the `unit_size` rectangle and over-reported by 3–6% because §7.20 unit
origins are offset by +8 luma lines; that is fixed.

## Decoder gate

```sh
DAV2D=tools/bin/dav2d-macos ./gate/build_and_test.sh
```

The gate builds a 2-frame M2 stream (key + P) at 256x192 and decodes it with
the pinned `dav2d` (`f9dafbc`, `--strict 1 --threads 1`).  `--width`/
`--height` accept any 8-aligned 4:2:0 size; 640x360 also passes.

1. all restoration types NONE, decode with the pinned dav2d → base recon;
2. `chroma_decide()` on the base recon vs a noisy target;
3. the same tiles with chroma `RESTORE_WIENER_NONSEP` units;
4. decode with `dav2d --strict 1 --threads 1` and compare every Y/U/V sample
   with `chroma_apply()`.

Current result on this host:

```text
baseline: 256x192, 2 frames decoded
frame0: shift=3 unit=32 units=24 active=13 uv_sym=0 sse 245167 -> 244787
frame1: shift=3 unit=32 units=24 active=18 uv_sym=5 sse 346507 -> 345954
decoded frame0 vs encoder §7.20: 0 diffs
decoded frame1 vs encoder §7.20: 0 diffs
filter active: frame0 U and V both changed=1
bit cost: baseline 220 bytes, chroma LR 468 bytes; measured syntax
          8 header + 760 tile bits (frame0) / 8 + 1256 (frame1)
PASS: chroma restoration decoded bit-exactly
```

The relative +112% is a property of the 220-byte synthetic fixture, not of
real content; see the quality table below for realistic rates.

## Quality (640x360 and 720p)

```sh
./gate/run_quality.sh
```

This encodes the fixture with the shipped M1 CPU reference at `base_q_idx=90`
with M1 defaults, decodes with the pinned dav2d (`f9dafbc`), and runs
`chroma_decide()` on the decoded reconstruction against the source.  The target is the source picture (oracle measurement); the
reported bits are the exact `lr_params()` + unit syntax cost.  It is not a
full rate-distortion integration.

| Input | M1 IVF | U PSNR | V PSNR | chroma SSE | syntax cost | CVVDP recon → restored |
|---|---:|---|---|---:|---:|---|
| 640x360 | 13,355 B | 52.452 → **53.734** dB (+1.282) | 50.530 → **52.175** dB (+1.645) | −29.19% | 438 B/frame (3,504 bits) | 9.9567 → 9.9573 JOD |
| 1280x720 | 25,349 B | 50.920 → **54.676** dB (+3.757) | 52.699 → **55.545** dB (+2.846) | −53.98% | 1,404 B/frame (11,232 bits) | 9.9696 → 9.9704 JOD |

At 30 fps the syntax cost is **105 kbps** at 640x360 (3.3% of the 13,355-byte
M1 frame) and **337 kbps** at 720p (5.5% of the 25,349-byte M1 frame).  The
720p fixture is a 1280x720 upscale of the 640x360 still.  CVVDP is
luma-dominated at this quality, so the small JOD delta is expected; the
chroma PSNR and bit cost are the useful numbers.  Filtered Y4M files are left
in the quality output directory for CVVDP inspection.

## Files

```text
src/encoder/filters/lr_chroma/
  chroma_lr.h              public API
  chroma_lr.cpp            unit sizing, filter model, fit/decision, syntax, apply
  chroma_lr_test.cpp       unit tests, pinned goldens, decision/golden coverage
  build_and_test.sh        standalone CPU build/test
  gate/
    vendor_video_syntax.py    vendored M2 writer with one read_lr() hook
    lr_chroma_gate.cpp     2-frame decoder gate
    build_and_test.sh      gate build/run
    lr_chroma_quality.cpp  PSNR/bit-cost/CVVDP probe
    run_quality.sh         640x360 + 720p measurement
  README.md                this file
```

## Limitations

* 8-bit 4:2:0 only; other bit depths/subsamplings are rejected.
* `RESTORE_SWITCHABLE` is not emitted (it needs PC-Wiener enabled to be a
  signallable tool).  Chroma plans use `RESTORE_NONE` / `RESTORE_WIENER_NONSEP`.
* Frame-level chroma filters (`frame_filters_on = 1`) are not part of the
  decision; the API writes/reads per-unit filters.  The §7.20 core still
  supports frame-level chroma coefficients if a caller supplies them.
* The quality probe uses the decoded output as both `curr` and `cdef` (the M1
  CLI does not retain a pre-CDEF plane).  On streams without CDEF this is
  exact; with CDEF it only affects taps that cross a stripe boundary.
* No CMake/root build edits.  The gate and quality tools are standalone.

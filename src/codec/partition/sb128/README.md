# encoder/sb128 — 128x128 superblock reference

CPU correctness oracle and minimal stream writer for the AV2 v1.0.0
128x128-superblock path. **Additive:** no M1/M2 core file, partition module,
`src/main.cpp`, or `CMakeLists.txt` is modified.

- Spec: AV2 Bitstream & Decoding Process Specification v1.0.0 (2026-05-28,
  commit `38f28e7`), §5.4.3, §5.18.4.4, §5.18.7.6, §5.20.3.1/2, §8.3.2,
  §9.2, §9.3.
- No AVM, SVT-AV1/AV2, clave, or av2-turbo encoder code was consulted.
- Verified decoder: `tools/bin/dav2d-macos`, build `f9dafbc` (the repo pin).

## Status

| Check | Result |
|---|---|
| Unit tests (ASan/UBSan, `-Werror`) | pass |
| Cross-check against `src/encoder/partition/` | pass |
| Minimal 128-SB stream, 128x128 / 256x256 / 640x360 / 1920x1080 | `dav2d` decodes every byte; all planes 128 |
| Determinism | two builds byte-identical in-process and across runs |
| Negative control | same tile with `use_128x128_superblock=0` is rejected by `dav2d` |

```sh
./src/encoder/sb128/build_and_test.sh
WERROR=1 SANITIZE=1 ./src/encoder/sb128/build_and_test.sh
SKIP_GATE=1 ./src/encoder/sb128/build_and_test.sh       # no dav2d
DAV2D=/path/to/dav2d ./src/encoder/sb128/build_and_test.sh
```

The script compiles `partition.cpp`, `bitwriter.cpp`, `basic_headers.cpp` and
`av2_entropy.cpp` directly. It does not touch the root CMake build.

## Files

| File | Purpose |
|---|---|
| `sb128.h` | public reference API: geometry, tables, boundary/implied, allowed set, contexts, CDF rows, reference tree |
| `sb128.cpp` | implementation; tables copied from §9.2 and independently cross-checked |
| `sb128_stream.h` | minimal 128-SB stream API |
| `sb128_stream.cpp` | sequence/frame headers, `use_128x128_superblock`, tile syntax, DC/no-residual leaf coder |
| `sb128_test.cpp` | unit tests, partition-module cross-checks, header/stream/determinism tests |
| `sb128_gate.cpp` | writes the four gate IVFs |
| `patch_use128_flag.py` | negative control: clears the SB flag, leaving the tile unchanged |
| `build_and_test.sh` | standalone build, tests, gate |

## API

All types are in `av2enc::sb128`.

### Geometry and §9.2 tables

```cpp
struct Geometry {
  int width, height;          // visible luma samples
  int sb_size = 128;          // 128; 64 accepted for cross-checks
  bool has_chroma = true;
  bool chroma_subsampling_x = true, chroma_subsampling_y = true;
  int max_pb_aspect_ratio = 8;
};
int mi_cols(const Geometry&);   // 2 * ((width + 7) >> 3)
int mi_rows(const Geometry&);
int sb_size_mi(const Geometry&);  // 32 for a 128 SB
int sb_cols(const Geometry&);
int sb_rows(const Geometry&);

BlockSize partition_subsize(PartitionType, BlockSize);
// SPLIT(128x128) == 64x64; SPLIT below 128 == BLOCK_INVALID.
// HORZ(128x128)=128x64, VERT(128x128)=64x128.
// HORZ(128x64) is invalid; VERT(128x64)=64x64.
```

`partition_implied_at_boundary()` and `partition_implied()` implement the
half-block tests and the `BLOCK_4X4` / thin-block `PARTITION_NONE` rule.
`init_allowed_partitions()` applies `rect_type_implied_by_bsize()`, the chroma
residual check, aspect-ratio limits, and the `block_coded()` chroma-offset
rules, then forces `PARTITION_NONE` if nothing is allowed.

### Block / chroma geometry and edges

```cpp
BlockRect block_rect(...);  // coded rect plus visible_width/visible_height
BlockSize chroma_plane_residual_size(...);   // 4:2:0 or 4:4:4
ChromaState resolve_chroma(...);             // ChromaMiRow/Col/Size
```

A 64x64 leaf maps to a 32x32 chroma residual; a 128x128 block to 64x64. A
partial superblock keeps the coded block size; only the visible rectangle is
clipped. This is why 640x360 and 1080p decode correctly at the bottom edge.

### §8.3.2 contexts and CDF rows

```cpp
struct Contexts { int do_split, do_square_split, rect_type; };
Contexts partition_contexts(geometry, r, c, size, MiSizeGrid&);
const std::uint16_t* do_split_cdf_row(int plane, int context);
const std::uint16_t* do_square_split_cdf_row(int plane, int context);
const std::uint16_t* rect_type_cdf_row(int plane, int context);
```

The `[2]` dimension of `TileDoSplitCdf`, `TileDoSquareSplitCdf` and
`TileRectTypeCdf` is **PlaneStart**, not the superblock flag. SHARED_PART uses
row 0. The 128-SB flag changes the context derivation:

| node | do_split ctx | do_square_split ctx | rect_type ctx |
|---|---:|---:|---:|
| 128x128 root (no neighbours) | 24 | 0 | 36 |
| 64x64 root (no neighbours) | 12 | 0 | 24 |
| 128x128, left 64x64 | 26 | 2 | 38 |
| 128x128, above 64x64 | 25 | 1 | 37 |
| 128x128, both | 27 | 3 | 39 |

`Partition_Size_Adjust[128x128]` is 6, so the 128 root does not share the
64x64/32x32 context rows. `do_square_split` is a separate table because
`read_partition()` codes `do_split` and then `do_square_split`; a
`PARTITION_SPLIT` at 128x128 is `do_split=1, do_square_split=1`.

Geometry-dependent context helpers (unit-tested, not yet exercised by the
minimal stream because CDEF/CCSO are off there):

```cpp
int cdef_sb_width_log2(int sb_size);   // 4 for 64, 5 for 128
int ccso_luma_size_log2(int sb_size);  // 6 for 64, 7 for 128
int ccso_unit_mi_shift(int sb_size);   // 4 for 64, 5 for 128
```

### Reference tree and stream

```cpp
bool build_reference_frame(const Geometry&, FrameTree&, std::string&);
bool validate_frame(const Geometry&, const FrameTree&, std::string&);
bool make_sb128_stream(const StreamConfig&, StreamResult&, std::string&);
```

`build_reference_frame()` is deterministic and uses only 64x64 leaves:

- full 128x128 root: `PARTITION_SPLIT`;
- boundary-forced root: implied `HORZ`/`VERT`;
- 128x64: `VERT`; 64x128: `HORZ`;
- 64x64: `PARTITION_NONE` where legal.

`make_sb128_stream()` writes a one-frame IVF payload (TD + SEQ + CLK) with
`use_128x128_superblock=1`, one tile, frozen default CDFs, DC_PRED luma/chroma,
and no residual. `stream_geometry_supported()` is a restriction of the minimal
writer, not of the geometry reference: width must be a multiple of 128, and the
MiRows remainder modulo 32 must be 0, 9..16, or 25..32. The excluded remainders
force the boundary recursion below 64x64, which the reference leaf coder does
not support.

Gate outputs:

| Stream | IVF | tile | leaves | partition symbols |
|---|---:|---:|---:|---:|
| 128x128 | 25 B | 5 B | 4 | 6 |
| 256x256 | 34 B | 12 B | 16 | 24 |
| 640x360 | 61 B | 39 B | 60 | 90 |
| 1920x1080 | 335 B | 312 B | 510 | 765 |

## dav2d capabilities verified

Decoding the four streams, then checking every luma/chroma byte equals 128:

1. `use_128x128_superblock` is parsed from the sequence header. The negative
   control (same tile, flag cleared) fails with `Invalid argument`.
2. `MiCols/MiRows` rounding and the 128-SB grid: 1x1, 2x2, 5x3 and 15x9
   superblock layouts decode at the requested dimensions.
3. The 128x128 root partition syntax: `do_split` first, then
   `do_square_split` on the separate CDF table, with the
   `Partition_Size_Adjust` row 6 context. Omitting `do_square_split` or using
   the 64x64 context row desynchronises immediately.
4. Boundary/implied recursion:
   - 640x360 last SB row starts at mi 64; `PARTITION_SPLIT` with clipped
     bottom 64x64 leaves;
   - 1920x1080 last SB row starts at mi 256 and is too short for the half-block
     test, so `partition_implied()` forces `HORZ` with no syntax, then
     128x64 -> implied/allowed `VERT` -> two 64x64 leaves.
5. Block/chroma mapping for all 64x64 leaves (32x32 chroma residual) and the
   visible-edge clipping on the bottom row.
6. Determinism: the gate builds each stream twice and compares bytes; the
   decoded output is identical across runs.

Not exercised by the stream (unit-tested only, or out of scope): CDEF/CCSO
symbol rows at 128 SB, residual coefficient contexts, SDP, extended partitions,
256x256 superblocks, inter frames, multi-tile layouts.

## Integration call sites (no core edits made)

These are the exact places a future M1 integration has to change. Line numbers
are from the current tree.

| File:line | Current behavior | Required for 128 SB |
|---|---|---|
| `src/encoder/m1/still_headers.cpp:104` | writes `use_128x128_superblock = 0` | take the flag from config/plan |
| `src/encoder/m1/still_headers.cpp:64-75` | `write_tile_info()` hardcodes `(mi_cols+15)>>4`, i.e. 64 SB | use `(mi_cols + sb_mi - 1) / sb_mi` with `sb_mi = 32` |
| `src/encoder/m1/still_headers.cpp:152` | `ccso_unit_matches_sb_size` assumes 64x64 units | derive the CCSO unit size from the SB size (6/7) |
| `src/encoder/m1/still_syntax.cpp:516-532` | codes only `do_split`; no `do_square_split` | emit `do_split` then `do_square_split`; use `TileDoSquareSplitCdf[PlaneStart][ctx]` |
| `src/encoder/m1/still_syntax.cpp:522` | always indexes `cdfs_.do_split[0]` | keep PlaneStart 0 for SHARED_PART but derive the 128x128 context row (24..27) |
| `src/encoder/m1/still_syntax.cpp:1416` | CDEF `sb_width_log2` is `4` | `5` for 128 SB (`cdef_sb_width_log2`) |
| `src/encoder/m1/still_syntax.cpp:1324` | CCSO `kCcsoShift` is `4` | `5` for 128 SB (`ccso_unit_mi_shift`) |
| `src/encoder/m1/still_encoder.cpp:203,1842` | geometry/plan forced to 64 | plumb `sb_size` through the still encoder |
| `src/encoder/m1/stream/still_stream_bridge.cpp:28` | default `sb_size = 64` | carry the plan's SB size |
| `src/encoder/m2/post/tiled_frame.cpp:141` | writes `use_128x128_superblock = 0` | config/plan-driven flag |
| `src/encoder/pipeline/tiled_still_gate.cpp:156` | writes `use_128x128_superblock = 0` | config/plan-driven flag |
| `src/bitstream/basic_headers.cpp:208` | M0 fixed 64 SB | leave M0 pinned; M1 needs its own header writer (this module) |

`src/encoder/m1/still_headers.h` also has no `use_128x128_superblock` field in
`StillConfig`; one is needed before the line-104 change can be made.

## Notes / non-goals

- The reference tree deliberately stops at 64x64 leaves. Coding a 128x128
  coding block needs `residual()`'s 64x64 chunking and multiple transform
  blocks; that is M1/M2 work, not SB-syntax work.
- `stream_geometry_supported()` is a restriction of the *minimal writer*, not
  of the geometry reference. The tree/context code handles arbitrary sizes;
  the leaf coder only handles 64x64.
- CDF row 0 is used everywhere because `enable_sdp = 0` (SHARED_PART). The
  `[2]` rows are retained in the API so a future CHROMA_PART consumer can use
  row 1 without changing the table layout.

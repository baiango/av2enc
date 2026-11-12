# encoder/sb256 — 256x256 superblock reference and `use_256x256_superblock` gate

CPU correctness oracle, 256-root partition model, and minimal intra stream
writer for the AV2 v1.0.0 `use_256x256_superblock` path. **Additive:** no M1/M2
core file, partition module, `src/encoder/sb128/`, `src/main.cpp`, or
`CMakeLists.txt` is modified.

- Spec: AV2 Bitstream & Decoding Process Specification v1.0.0 (2026-05-28,
  commit `38f28e7`), §5.4.3, §5.18.1, §5.18.4.4, §5.18.7.2/6, §5.20.3.1/2,
  §5.20.9.1, §5.20.10.1/2, §8.3.2, §9.2.
- No AVM, SVT-AV1/AV2, clave, or av2-turbo encoder code was consulted.
- Verified decoder: `tools/bin/dav2d-macos`, build `f9dafbc` (the repo pin).

## The spec fact this module is built around

`use_256x256_superblock` is **not** a request for a 256x256 intra superblock:

```
§5.4.3  use_256x256_superblock f(1)
        if (!use_256x256_superblock) use_128x128_superblock f(1)
§5.18.1 if (use_256x256_superblock)
          SbSize = FrameIsIntra ? BLOCK_128X128 : BLOCK_256X256
§5.18.7.6 get_seq_sb_size() = BLOCK_256X256 when the flag is 1
```

So there are two distinct values:

| value | 256 flag = 1, intra | 256 flag = 1, inter |
|---|---:|---:|
| `get_seq_sb_size()` (sequence level) | 256x256 | 256x256 |
| `SbSize` (frame header) | **128x128** | 256x256 |

A still-picture stream can therefore gate the sequence flag (and the
sequence-level sizing rules), but its partition tree is a 128x128 tree.
`make_sb256_stream()` writes exactly that; the full 256x256 recursion is
modelled and unit-tested for the future inter path, where it is actually used.
`sb256.h` exposes both `get_seq_sb_size()` and `frame_sb_size()` so callers
cannot accidentally conflate them.

## Status

| Check | Result |
|---|---|
| Unit tests (`-Werror`, ASan/UBSan) | pass |
| Cross-check against `src/encoder/partition/` | pass |
| Cross-check against `src/encoder/sb128/` | pass |
| `dav2d --strict 1 --threads 1`, 256x256 / 640x360 / 1920x1080 / 520x300 | decodes every byte; recon 0 diffs |
| `use_128x128_superblock` positive control | decodes; byte-identical tile |
| Negative control (clear `use_256x256_superblock`) | rejected by `dav2d` |
| Determinism | two in-process builds byte-identical; pinned SHA-256 goldens |

```sh
./src/encoder/sb256/build_and_test.sh
WERROR=1 SANITIZE=1 ./src/encoder/sb256/build_and_test.sh
SKIP_GATE=1 ./src/encoder/sb256/build_and_test.sh       # no dav2d
DAV2D=/path/to/dav2d ./src/encoder/sb256/build_and_test.sh
```

The script compiles `sb256.cpp`, `sb256_stream.cpp`, `partition.cpp`,
`sb128.cpp`, `sb128_stream.cpp`, `bitwriter.cpp`, `basic_headers.cpp` and
`av2_entropy.cpp` directly. It does not touch the root CMake build.

## Files

| File | Purpose |
|---|---|
| `sb256.h` | public reference API: SbSize/get_seq_sb_size, geometry, tables, boundary/implied, allowed set, contexts, CDF rows, CDEF/CCSO/entropy sizing, syntax plan, reference tree |
| `sb256.cpp` | implementation; tables copied from §9.2 and cross-checked against `sb128` and `partition` |
| `sb256_stream.h` | minimal intra stream API (`use_256=1` and the `use_128=1` control) |
| `sb256_stream.cpp` | sequence/frame headers, conditional `use_128` syntax, tile syntax, DC/no-residual square-leaf coder |
| `sb256_test.cpp` | unit tests, exhaustive cross-checks, header/stream/golden/determinism tests |
| `sb256_gate.cpp` | writes the four gate IVFs plus the 128-SB control |
| `patch_use256_flag.py` | negative control: clears `use_256x256_superblock`, leaves the tile unchanged |
| `build_and_test.sh` | standalone build, tests, dav2d gate, negative control |

## API

All types are in `av2enc::sb256`.

### Superblock configuration

```cpp
struct SbConfig {
  bool use_256x256_superblock = false;
  bool use_128x128_superblock = false;  // absent when use_256 is 1
  bool frame_is_intra = true;
};
bool validate_sb_config(const SbConfig&, std::string&);
BlockSize get_seq_sb_size(bool use_256, bool use_128);  // 256 / 128 / 64
BlockSize frame_sb_size(const SbConfig&);               // intra clamps to 128
int frame_sb_size_pixels(const SbConfig&);
```

`validate_sb_config()` rejects `use_256 && use_128` (the second syntax element
is not present at all when the first is 1).

### Geometry and §9.2 tables

```cpp
struct Geometry { int width, height; int sb_size = 256; /* 64/128/256 */
                  bool has_chroma, chroma_subsampling_x/y; int max_pb_aspect_ratio; };
int mi_cols(const Geometry&);   // 2 * ((width + 7) >> 3)
int mi_rows(const Geometry&);
int sb_size_mi(const Geometry&);  // 64 for a 256 SB
int sb_cols(const Geometry&);
int sb_rows(const Geometry&);
int entropy_left_context_rows(const Geometry&);   // sb_rows * sb_size_mi
int entropy_above_context_cols(const Geometry&);  // sb_cols * sb_size_mi

BlockSize partition_subsize(PartitionType, BlockSize);
```

`superblock_size(256) == BLOCK_256X256`. The relevant §9.2 rows are:

| bSize | `Partition_Size_Adjust` | `_Rect_Type` | HORZ | VERT | SPLIT |
|---|---:|---:|---|---|---|
| 256x256 | 9 | 12 | 256x128 | 128x256 | **128x128** |
| 256x128 | 8 | 11 | invalid | 128x128 | invalid |
| 128x256 | 7 | 10 | 128x128 | invalid | invalid |
| 128x128 | 6 | 9 | 128x64 | 64x128 | 64x64 |

`rect_type_implied_by_bsize()` returns VERT for 256x128 and HORZ for 128x256,
so the invalid direction can never be signalled.

### Boundary / implied / allowed set

```cpp
Boundary partition_implied_at_boundary(...);  // half-block test
Boundary partition_implied(...);              // adds BLOCK_4X4 / thin-block NONE
bool is_partition_allowed(...);
PartitionSet init_allowed_partitions(...);    // forces NONE when nothing is legal
```

A 1080p 256 root at mi row 256 has `MiRows = 270`, so the half-block test
(`256 + 32 < 270`) fails and `partition_implied()` forces HORZ with no syntax.
A 520x300 256 root at mi col 128 has only 2 mi columns left and forces VERT in
the top row. Partial roots keep their coded size; only the visible rectangle is
clipped.

### §8.3.2 contexts and CDF rows

```cpp
struct Contexts { int do_split, do_square_split, rect_type; };
Contexts partition_contexts(geometry, r, c, size, MiSizeGrid&);
const std::uint16_t* do_split_cdf_row(int plane, int context);
const std::uint16_t* do_square_split_cdf_row(int plane, int context);
const std::uint16_t* rect_type_cdf_row(int plane, int context);
```

The `[2]` dimension is PlaneStart (SHARED_PART uses row 0), not the superblock
flag. At a 256x256 root:

| node | do_split | do_square_split | rect_type |
|---|---:|---:|---:|
| no decided neighbours | **36** | **4** | **48** |
| left 64x64 | 38 | 6 | 50 |
| above 64x64 | 37 | 5 | 49 |
| both | 39 | 7 | 51 |
| 256x128, no neighbours | 32 | 0 | 44 |
| 128x256, no neighbours | 28 | 0 | 40 |

`TileDoSplitCdf[0][36..39]` and `TileRectTypeCdf[0][48..51]` are valid rows;
`do_square_split` lives on the separate `TileDoSquareSplitCdf` table and its
root row is 4..7 (the +4 is `bSize == BLOCK_256X256`). A `PARTITION_SPLIT` at
256x256 codes `do_split = 1` then `do_square_split = 1`, and lowers to four
128x128 children.

### SB-level syntax plan

```cpp
enum class PartitionSymbolKind { kDoSplit, kDoSquareSplit, kRectType };
struct PartitionSymbol { PartitionSymbolKind kind; int value; int context; };
bool plan_partition_syntax(const Node&, const Contexts&,
                           std::vector<PartitionSymbol>&, std::string&);
```

For a full 256 root the plan is exactly two symbols:
`{kDoSplit, 1, 36}`, `{kDoSquareSplit, 1, 4}`. Forced nodes return an empty
plan. This is the API a future M1/M2 integration should call instead of
re-deriving the order.

### SbSize-dependent unit sizing

```cpp
int cdef_unit_size_pixels();          // 64, always (cdefSize4)
int cdef_units_per_sb(sb);            // 1 / 2 / 4
int cdef_sb_width_log2(sb);           // 4 / 5 / 6 (cdef_index0 above-row test)

int ccso_luma_size_log2_matching_sb(sb);      // 6 / 7 / 8
int ccso_luma_size_log2_from_tile_alignment(unsigned tile_start_or);
int ccso_luma_size_log2(sb, bool matches, unsigned tile_start_or);
int ccso_unit_mi_shift(sb);                   // 4 / 5 / 6
```

CDEF remains a 64x64 unit even for a 256 SB: `clear_cdef()` clears 4x4 units
and `cdef_index0` uses `Mi_Width_Log2[SbSize] = 6` for the above-superblock-row
test. CCSO with `ccso_unit_matches_sb_size = 1` is 256x256
(`CcsoLumaSizeLog2 = 8`, MI shift 6); otherwise the tile-start alignment picks
6/7/8 exactly as §5.18.7.12 says.

### Chroma, tree and stream

```cpp
BlockSize chroma_plane_residual_size(geometry, luma_size);
ChromaState resolve_chroma(...);            // ChromaMiRow/Col/Size
bool build_reference_frame(const Geometry&, FrameTree&, std::string&);
bool validate_frame(const Geometry&, const FrameTree&, std::string&);

bool stream_geometry_supported(const Geometry&, std::string&);
bool make_sb256_stream(const StreamConfig&, StreamResult&, std::string&);
bool make_use128_control_stream(const StreamConfig&, StreamResult&, std::string&);
std::size_t write_sequence_header(BitWriter&, const StreamConfig&, const SbFlags&);
std::size_t write_frame_header(BitWriter&, const StreamConfig&);
```

4:2:0 chroma mapping:

| luma | chroma residual |
|---|---|
| 256x256 | 128x128 |
| 256x128 | 128x64 |
| 128x256 | 64x128 |
| 128x128 | 64x64 |
| 64x64 | 32x32 |
| 8x8 | 4x4 |

A partial superblock keeps the coded block size; `block_rect()` exposes the
visible clipping separately.

## Reference tree

`build_reference_frame()` is deterministic. Full 256 roots use
`PARTITION_SPLIT` to 128x128 and then to 64x64; full 128x128 roots split to
64x64; boundary-forced roots use the implied HORZ/VERT; rectangular blocks
split toward a square; the resulting leaves are square and 8x8..64x64 for the
gate geometries. For 520x300 the 8 px right edge is coded with 8x8 leaves, which
is why this module can write a size the 64x64-leaf-only sb128 writer rejects.

| geometry | SBs | nodes | leaves | tree FNV-1a-64 |
|---|---:|---:|---:|---|
| 256x256 @ 256 | 1 | 21 | 16 | `b1c19dcf73682ffc` |
| 640x360 @ 256 | 6 | 85 | 60 | `39966745e16f3603` |
| 1920x1080 @ 256 | 40 | 712 | 510 | `cc96e8f191c1e408` |
| 520x300 @ 256 | 6 | 178 | 78 | `ffef91337db99a97` |
| 520x300 @ 128 | 15 | 168 | 78 | `62676a57cd225e4d` |
| 128x128 @ 128 | 1 | 5 | 4 | `ee86f38467a69566` |

`validate_frame()` re-derives every boundary/allowed/chroma value, checks child
order and sizes, and requires every in-frame cell in each superblock to be
covered exactly once.

## Stream and gate

`make_sb256_stream()` writes a one-frame IVF payload (TD + SEQ + CLK) with:

- `use_256x256_superblock = 1`, and **no** `use_128x128_superblock` bit;
- one tile, `seq_tile_info_present_flag = 0`, frozen default CDFs;
- `SbSize = 128x128` for the intra frame (§5.18.1);
- DC_PRED luma/chroma, no residual, no CDEF/CCSO/deblock/restoration.

`make_use128_control_stream()` writes the same tile under
`use_256=0, use_128=1`; the tile bytes are identical, which is the positive
control that the 256 flag does not change the intra partition syntax.

Pinned goldens (`result.frame` is the TD+SEQ+CLK OBU payload; `tile` is the
range-coded tile):

| stream | frame B | tile B | SBs | leaves | symbols | frame SHA-256 |
|---|---:|---:|---:|---:|---:|---|
| 256x256 | 33 | 12 | 4 | 16 | 24 | `b058de2377804e5b643866bda5e3b16cfd4b435f9c3fafab1ccf19a8924886ee` |
| 640x360 | 61 | 39 | 15 | 60 | 90 | `fd7da565aa788d56dd6406bea5e84c03586ba92ab2b333b18c5002a73ddc7f03` |
| 1920x1080 | 335 | 312 | 135 | 510 | 765 | `71894e605cbb7fc54eca6cc0dee062112f4012046101b6830089962ffb153878` |
| 520x300 | 89 | 67 | 15 | 78 | 145 | `6cd70e53f53124a9cd2a4734c5a204374e6ed00657b171b54da61243ffc5979d` |
| 128x128 `use_256` | 25 | 5 | 1 | 4 | 6 | `6bc442569621e7db23911485a55904f7683b3c9cd238bf9acaaa95e803971380` |
| 128x128 `use_128` control | 25 | 5 | 1 | 4 | 6 | `6f75573d9a5ee7050d07af82527ab98fe42081248677d14a109df20e6361c3ea` |

Gate command (run by `build_and_test.sh`):

```sh
dav2d -q -i sb256_<case>.ivf -o sb256_<case>.y4m --strict 1 --threads 1
```

Observed output:

```
sb256_256x256.ivf: 33 B, 12 tile B, 16 leaves, 4 superblocks, 24 partition symbols
sb256_640x360.ivf: 61 B, 39 tile B, 60 leaves, 15 superblocks, 90 partition symbols
sb256_1920x1080.ivf: 335 B, 312 tile B, 510 leaves, 135 superblocks, 765 partition symbols
sb256_520x300.ivf: 89 B, 67 tile B, 78 leaves, 15 superblocks, 145 partition symbols
sb256_control_use128_128x128.ivf: 25 B, 5 tile B, 4 leaves, 1 superblocks, 6 partition symbols
sb256 gate: two independent runs are byte-identical
sb256 gate: negative control sb256_256x256 rejected (use_256x256_superblock is parsed)
...
sb256_256x256: dav2d OK, 98304 bytes, recon 0 diffs (all planes 128)
sb256_640x360: dav2d OK, 345600 bytes, recon 0 diffs (all planes 128)
sb256_1920x1080: dav2d OK, 3110400 bytes, recon 0 diffs (all planes 128)
sb256_520x300: dav2d OK, 234000 bytes, recon 0 diffs (all planes 128)
sb256_control_use128_128x128: dav2d OK, 24576 bytes, recon 0 diffs (all planes 128)
sb256 gate: PASS
```

“Recon 0 diffs” is checked against the flat-128 source: exact expected plane
size and zero bytes other than 128.

### Negative control

`patch_use256_flag.py` clears `use_256x256_superblock` in the sequence header
and leaves the tile unchanged. The decoder then reads the next bit (our
`enable_sdp = 0`) as `use_128x128_superblock = 0`, expects a 64x64-superblock
tile, and rejects the stream (“Error parsing sequence header” / “Invalid
argument”). All four gate streams are tested. This proves the flag is parsed,
not ignored.

## Tests and cross-checks

`sb256_test` covers:

- §9.2 tables for all 29 block sizes against **both** `sb128` and `partition`;
- `Partition_Size_Adjust` / `_Rect_Type` rows 7..9 and the 256 split chain;
- `SbConfig` / `get_seq_sb_size` / the intra 128 clamp / both-flags error;
- MiCols/MiRows and the 64/128/256 superblock grids over 20 widths × 18
  heights, including 520x300 and 1080p;
- boundary/implied and allowed sets against `sb128` (and `partition` on a
  subset): a broad sweep over 20 widths × 18 heights with sampling for large
  frames, plus an exhaustive all-mi-position, all-29-size check for 520x300,
  300x520, 256x256, 256x300, 300x256, 200x136, 136x200 and 128x128;
- §8.3.2 contexts against `sb128` and `partition`, including 256 roots,
  do_square_split 4..7, and 256x128/128x256;
- partition syntax plans, CDF row pointers, chroma mapping, CDEF/CCSO/entropy
  sizing;
- reference-tree counts, digests, validation and determinism;
- header bit parsing (the 256 header is exactly one bit shorter than the 128
  control), pinned stream SHA-256 goldens, tile equality with the control and
  with `sb128` where `sb128` supports the geometry;
- `520x300`: supported here, rejected by `make_sb128_stream` (its leaf coder is
  64x64-only), and decoded by `dav2d`.

## Which tools are NOT gated with 256 SBs

The gate proves the **sequence flag, intra SbSize clamp, MiCols/MiRows,
superblock geometry, partition/tree/contexts for 256 roots, chroma mapping,
CDEF/CCSO unit arithmetic, and the intra stream**. These pieces are **not**
gated at 256, and the listed call sites are where a future integration has to
change:

| Tool / path | Status at 256 SB | Exact call sites |
|---|---|---|
| CDEF (M1 symbol writer) | **wired:** `sb_width_log2 = 5` for 128 SB (`--still-cdef`); still a 64x64 filter unit | `src/pipeline/still/still_syntax.cpp` (`cdef_encode_block_index`) |
| CDEF (M2 index writer) | `TileIndexWriter::init` accepts 5/6, but the writer hardcodes 4 | `src/encoder/m2/cdef/video_cdef.cpp:373`, `:380` (“M2's superblock is 64x64”), `:418`; `src/encoder/m2/cdef/video_cdef.h:180` |
| CDEF (stream/`filters_stream`) | `cdef_tile_init` accepts 5/6; `cdef_clear_superblock` still clears one 16-mi 64x64 origin | `src/encoder/m1/filters_stream/filters_stream.cpp:560`, `:581-586`; `filters_stream.h:140` |
| CDEF (CUDA) | index-context helper takes `sb_width_log2`, integration does not | `src/cuda/m1/cdef/still_cdef_index_syntax.h:97-118`; `src/cuda/m1/cdef/run_stream_gate.cpp:136-137` writes both flags 0 |
| CCSO header/filter | `CcsoLumaSizeLog2` 6/7/8 supported in the generic filter and header core, but no 256-SB stream gate | `src/encoder/m1/filters_stream/filters_stream.cpp:702-717`; `src/encoder/m2/restoration/video_restoration.cpp:534-596`; `src/cuda/restoration/ccso.cu:48` |
| CCSO (M1 writer) | **wired:** `CcsoLumaSizeLog2 = 7` (128x128 unit) with `ccso_unit_matches_sb_size = 1` (`--still-ccso`) | `src/pipeline/still/still_syntax.cpp` (`write_ccso`) and `still_encoder.cpp` (unit grid) |
| CCSO (M2 post) | grid built with `CcsoLumaSizeLog2 = 6` | `src/encoder/m2/post/loop_filters.cpp:130` |
| Residual / coefficient syntax | tests and gates pin `sb_size = 64`; no 256-root tile is coded | `src/encoder/m2/residual/inter_residual_test.cpp:43`; `src/encoder/m2/residual/gate/video_residual_gate.cpp:204`, `:523`; `src/encoder/m2/video_frame.cpp:290`, `:393` |
| Inter frames (the only place a 256 root is legal) | every inter header writer forces `use_256 = 0` | `src/encoder/m2/video_headers.cpp:130-131`; `src/encoder/inter/leading/leading.cpp:118-119`; `src/encoder/inter/random_access/random_access.cpp:304-305`; `src/encoder/inter/longterm/longterm_headers.cpp:235-236`; `src/encoder/inter/tlayers/tlayers_syntax.cpp:157-158`; `src/encoder/inter/gme/gme_gate.cpp:208-209` |
| M1 core / partition engine | rejects `sb_size == 256` and caps at 128 | `src/encoder/partition/partition.h:58`; `src/encoder/partition/partition.cpp:765`; `src/encoder/m1/still_syntax.cpp:1558`; `src/encoder/m1/still_encoder.cpp:203`, `:1842`; `src/encoder/m1/still_headers.cpp:65-66` |
| M1 partition-stream bridge | uses `partition::superblock_size()`, which has no 256 case | `src/encoder/m1/partition_stream/partition_stream.cpp:611`; `src/encoder/m1/stream/still_stream_bridge.cpp:28` |
| M1 headers | writes both superblock flags 0 and a 64-SB tile grid | `src/encoder/m1/still_headers.cpp:65-66`, `:103-104` |
| M2 tiled post / tiles | tile planner rejects 256 | `src/encoder/m2/post/tiled_frame.cpp:140-141`, `:282-288`, `:495-501`; `src/encoder/tiles/tile_layout.cpp:240-242`, `:450-451` |
| CUDA M1 plan | rejects any SB size other than 64 | `src/cuda/m1/plan/plan.cpp:239-240`, `:566`; `src/cuda/m1/still_gpu.h:179`; `src/cuda/m1/plan/plan_gpu_bridge.cpp:116` |
| Pipeline still gate | writes both flags 0 | `src/encoder/pipeline/tiled_still_gate.cpp:155-156` |
| M0 | intentionally pinned to 64 | `src/bitstream/basic_headers.cpp:207-208` |
| Generic header helpers | 256 syntax is representable, but no mutual-exclusion/intra clamp | `src/encoder/format/chroma/chroma_headers.cpp:331-333` (writer), `:571-574` (reader); `src/encoder/syntax/small_header/small_header.cpp:975-979` |

In short: the **SB-syntax layer** is gated at 256; CDEF/CCSO symbol emission,
residual coding and inter frames are not. The generic CDEF/CCSO header/filter
cores have partial 5/6-aware code, but their M1/M2 integrations still assume a
64x64 superblock.

## Integrated product path (`--still-sb256`)

`--still-sb256` signals the 256x256 sequence flag, and an intra frame codes a
128x128 `SbSize` grid (§5.18.1).  The M1 writer now supports `--still-cdef` and
`--still-ccso` on that grid: CDEF uses `sb_width_log2 = 5` and CCSO uses
`CcsoLumaSizeLog2 = 7` (`ccso_unit_matches_sb_size = 1`).  The product gate is
shared with the 128-SB module:

```sh
src/codec/partition/sb128/product_gate.sh build-int
```

It covers `--still-sb256` at 640x360 (edges) and 256x256 with CDEF, CCSO, both,
8 CDEF strengths and edge CCSO; all decode with the pinned `dav2d` and match the
encoder reconstruction exactly.  A 256x256 *coding* block is still not emitted
(an intra frame cannot use one); the flag only changes the sequence-level size
plus the tile grid.

## Notes / non-goals

- The reference tree deliberately emits DC_PRED / no-residual square leaves so
  the superblock syntax can be gated in isolation.
- An intra frame cannot exercise a 256 root because of the §5.18.1 clamp. Doing
  that needs an inter frame with a real reference setup, which is M2 work.
- `stream_geometry_supported()` is a restriction of the minimal leaf coder
  (square 8x8..64x64), not of the geometry reference.
- CDF row 0 is used everywhere because `enable_sdp = 0` (SHARED_PART). The `[2]`
  rows remain in the API for a future CHROMA_PART consumer.

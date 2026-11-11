# M2 multi-tile-group framing

Status: **spec-only component, no M2 core/CMake/main edits, no shipped path
uses it yet.** This directory owns the §5.19 framing for a frame that is split
across two or more tile-group OBUs. It writes and independently reads:

- `is_first_tile_group` / the `SeenFrameHeader` rule;
- `frame_header_present_flag`;
- `frame_header_copy()` / `header_bit[i]` (§5.18.1, §6.17.1);
- `tile_start_and_end_present_flag`, `tg_start`, `tg_end`;
- `bru_tile_active` (§5.19, §5.20.2.1);
- the `bru_inactive` header-only early return;
- `tile_size_minus_1` interleaving and the implied last-tile size;
- every §6.18 conformance constraint (contiguity, no gap, no overlap, first
  group starts at `TileNum 0`, last group ends at `NumTiles - 1`, one
  `is_first_tile_group`, bit-exact header copies).

Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
(2026-05-28, commit `38f28e7`). Related specs read for this component:
`docs/spec-audit/03-frame-header.md`, `docs/spec-audit/04-tile-block.md`,
`docs/spec-audit/05-decode-ordering.md`, `docs/tiles.md`.

## Files

| File | Purpose |
|---|---|
| `tile_groups.h` / `.cpp` | framing writer, independent reader, §6.18 validation |
| `tile_groups_test.cpp` | unit tests, invalid-sequence rejection, pinned goldens |
| `tile_groups_gate.cpp` | 4 stream `dav2d --strict 1 --threads 1` gate |
| `goldens/*.hex` | pinned group payload bytes |
| `build_and_test.sh` | standalone build/run (no CMake changes) |

Build and run:

```sh
./src/encoder/m2/tile_groups/build_and_test.sh
SANITIZE=1 ./src/encoder/m2/tile_groups/build_and_test.sh
WERROR=1   ./src/encoder/m2/tile_groups/build_and_test.sh
GATE=0     ./src/encoder/m2/tile_groups/build_and_test.sh   # unit tests only
DAV2D=/path/to/dav2d ./src/encoder/m2/tile_groups/build_and_test.sh
```

Current result on the macOS dev host:

```text
PASS: tile group framing writer/reader, goldens, and invalid-sequence rejection
  640x360 4x1 tiles, groups=2+2 shared_header key_tile_bytes=13/13/9/9 tile_size_bytes=1
  640x360 4x1 tiles, groups=1+2+1 header_copy key_tile_bytes=13/13/9/9 tile_size_bytes=1
  1920x1080 4x1 tiles, groups=2+2 shared_header key_tile_bytes=82/82/72/72 tile_size_bytes=1
  1920x1080 4x1 tiles, groups=1+2+1 header_copy key_tile_bytes=82/82/72/72 tile_size_bytes=1
PASS: 4 multi-tile-group streams decoded (dav2d --strict 1 --threads 1),
      decoded samples match the constant DC/GLOBALMV reconstruction exactly
```

ASan/UBSan clean, including the decoder gate. This is a CPU correctness result,
not a speed or quality result.

## API

```cpp
#include "encoder/m2/tile_groups/tile_groups.h"
namespace tg = av2enc::m2::tile_groups;

// 1. Frame-header bits are produced once by the M2 header writer. The header
//    is not re-serialised per group; a copy re-emits these exact bits.
BitWriter header_writer;
std::size_t header_bits = 0;
post::write_m2_frame_header_syntax(header_writer, config, sequence, frame_header,
                                   layout, frame.tile_size_bytes, &header_bits,
                                   &error);
tg::FrameHeaderBits header;
tg::capture_frame_header_bits(header_writer.bytes(), header_bits, &header);

// 2. Geometry comes from the validated tile layout + TileSizeBytes from the
//    frame header. TileSizeBytes covers the largest tile in the whole frame.
tg::TileGroupGeometry geometry =
    tg::geometry_from_layout(layout, frame.tile_size_bytes,
                             /*use_bru=*/false, /*bru_inactive=*/false);

// 3. Contiguous groups, e.g. 2+2 or 1+2+1 tiles.
std::vector<tg::TileGroupSpan> groups;
tg::plan_contiguous_tile_groups(geometry, std::vector<int>{2, 2},
                                /*copy_frame_header=*/false,
                                /*bru_active_default=*/false, &groups, &error);

tg::FrameTileGroupPlan plan;
plan.geometry = geometry;
plan.frame_header = header;
plan.copy_frame_header = false;   // later groups use frame_header_present_flag = 0
plan.groups = groups;

// 4. `tile_payloads` is the whole frame in raster TileNum order, as returned by
//    post::assemble_m2_tiled_frame(). One payload per group comes back.
std::vector<std::vector<std::uint8_t>> group_payloads;
tg::write_frame_tile_groups(plan, frame.tile_payloads, &group_payloads, &error);

// 5. Independent reader. `header.bit_count` is §5.18.1 NumFrameHeaderBits,
//    saved by the frame-header parser; grouping fields are re-read from bytes.
tg::TileGroupSequence parsed;
tg::read_tile_group_sequence(group_payloads, geometry, header.bit_count,
                             &parsed, &error);
```

Group OBU payloads are wrapped with the frame's own OBU type
(`OBU_CLOSED_LOOP_KEY` for a keyframe, `OBU_REGULAR_TILE_GROUP` for a P frame).
All groups of one frame are concatenated into the same temporal unit; only the
first group carries the frame header (or a bit-exact copy when
`copy_frame_header = true`).

### Writer rules

For group `g` of `NumTiles`:

```text
is_first_tile_group                          f(1)
if (!is_first) frame_header_present_flag     f(1)
if (frame_header_present) frame_header() / frame_header_copy()
if (bru_inactive) { trailing_bits(...); return; }   // header-only OBU
if (NumTiles > 1) tile_start_and_end_present_flag f(1)
if (tile_start_and_end_present_flag) {
  tg_start f(TileColsLog2 + TileRowsLog2)
  tg_end   f(TileColsLog2 + TileRowsLog2)
}
if (use_bru && NumTiles > 1)
  bru_tile_active f(1) per tile in [tg_start, tg_end]
byte_alignment()                             zero bits
for TileNum = tg_start .. tg_end:
  if (TileNum != tg_end) le(TileSizeBytes) tile_size_minus_1
  tile payload
```

`tile_start_and_end_present_flag` is written as `0` when the group covers the
whole frame (matching the existing single-group writers) and `1` otherwise.
`tg_end >= tg_start` and `tg_end < NumTiles` are checked before writing.

### Reader rules

The reader does not trust any writer state. It re-reads the flags, the
`tg_start`/`tg_end` fields, the BRU bits, the zero `byte_alignment()` padding,
the little-endian size fields, and the payload extent. It rejects:

| Invalid stream | Rejection |
|---|---|
| first group does not start at tile 0 | `§6.18` first-group check |
| tile gap (`0..1`, `3..3`) | `tg_start != expected` (gap) |
| overlap/reordering (`0..2`, `1..3`) | `tg_start != expected` (overlap/reordering) |
| out-of-order groups (`2..3`, `0..1`) | first group starts at 2 |
| missing final group | last `tg_end != NumTiles - 1` |
| duplicate `is_first_tile_group = 1` | `SeenFrameHeader` mismatch |
| `frame_header_copy` differs from the first header | §6.17.1 bit comparison |
| non-zero `byte_alignment()` padding | padding check |
| `tile_size_minus_1` past the OBU payload | size/extent check |
| tile payload shorter than 2 bytes | `init_symbol()` minimum |
| truncated header/copy/`bru_tile_active` | bounds-checked bit reader |
| `bru_inactive` without `trailing_bits()` one bit | early-return check |

## Pinned goldens

`goldens/tile_groups_2group.hex` and `goldens/tile_groups_3group.hex` pin the
complete group OBU payloads (OBU header excluded) for:

- a 2×2-tile frame, `TileSizeBytes = 2`, `use_bru = 1`, groups `2+2`, second
  group shared-header (`frame_header_present_flag = 0`), mixed per-tile
  `bru_tile_active` maps `{1,0}` / `{0,1}`;
- the same 2×2-tile frame, `TileSizeBytes = 1`, groups `1+2+1`,
  `frame_header_copy()` on both later groups.

The test also re-parses every golden and requires the expected header-copy /
shared-header counts. Regenerate only after an intentional syntax change:

```sh
"$BUILD/tile_groups_test" --emit-goldens src/encoder/m2/tile_groups/goldens
```

## dav2d gate

`tile_groups_gate` encodes four keyframe + P-frame streams at 640×360 and
1920×1080 with 4×1 tiles:

| stream | groups | later groups |
|---|---|---|
| 640×360 | `2+2` | `frame_header_present_flag = 0` |
| 640×360 | `1+2+1` | `frame_header_copy()` |
| 1920×1080 | `2+2` | `frame_header_present_flag = 0` |
| 1920×1080 | `1+2+1` | `frame_header_copy()` |

Every stream is decoded with `dav2d -q -i … -o … --strict 1 --threads 1`. The
block plan is one 64×64 intra leaf with `eob = 0` per superblock for the
keyframe, and GLOBALMV + `skip_flag = 1` for the P frame. The keyframe
reconstruction is therefore provably the DC predictor (`128`) everywhere, the
P-frame reconstruction is a zero-MV copy of it, and the gate requires **every**
decoded Y/U/V sample to equal `128` in both frames. All loop filters are off,
so no tile-edge filtering can perturb the comparison
(`docs/negative-results.md`: pinned `dav2d` filters across tile edges even with
`disable_loopfilters_across_tiles = 1`).

Before decoding, the gate also:

- re-reads the group bytes with the independent reader and compares every
  `tg_start`/`tg_end`, tile size and tile payload byte against the encoded
  payloads;
- builds a one-group plan from the same header bits and tile payloads and
  requires it to be **byte-identical** to
  `post::assemble_m2_tile_group_from_payloads()`'s OBU. This is the
  compatibility contract with the existing single-group writer.

Exit code 77 means no decoder was found (CTest-style skip).

## M2 call sites

Nothing outside this directory is modified. The sites that must call this
component when a frame is split:

1. **`src/encoder/m2/post/tiled_frame.cpp:642`
   `assemble_m2_tile_group_from_payloads()`** — currently the only way to turn
   raster tile payloads into an OBU. It hardcodes
   `is_first_tile_group = 1` (`:680`) and
   `tile_start_and_end_present_flag = 0` (`:688`) and therefore cannot emit a
   second group. Replace the tail of that function with
   `write_frame_tile_groups()` when the caller asks for more than one group;
   keep the existing path (or the one-group plan, which is byte-identical) for
   a single group.
2. **`src/encoder/m2/post/tiled_frame.cpp:592`
   `assemble_m2_tiled_frame()`** — already produces raster
   `M2TiledFrame::tile_payloads`, `tile_bytes` and the global
   `tile_size_bytes`, which is exactly this module's input. A split-aware
   variant only has to call the writer from (1) with the plan.
3. **`src/encoder/m2/post/tiled_frame.cpp:371`
   `write_m2_frame_header_syntax()`** — supplies the one header bit string.
   Do not re-serialise the header per group; pass its output through
   `capture_frame_header_bits()` once, then share or copy it.
4. **`src/encoder/m1/tiles_stream/tiles_stream.cpp:428-431`** and
   **`src/encoder/m2/m2_headers.cpp:321`** — the M1/M2 single-tile writers
   have the same hardcoded `is_first_tile_group = 1`; they stay as-is for the
   one-group path.
5. **`src/encoder/tiles/tile_group.cpp:49`** — `plan_single_tile_group()`
   explicitly rejects a non-first group with a frame header
   (`"partial non-first tile groups with a frame header are not modelled"`).
   That is the `frame_header_copy()` gap this module closes; the planner can
   delegate to `validate_frame_tile_groups()` instead of rejecting.
6. **`src/encoder/m2/m2_frame.cpp` / `m2_gate.cpp`** — one tile, one group;
   unaffected.

When wiring, the per-frame loop must also respect §5.20.1/§7.5: run
`frame_end_update_cdf()` and `decode_frame_wrapup()` only after the group whose
`tg_end == NumTiles - 1`, and clear `SeenFrameHeader` there. With the current
M2 configuration (`disable_cdf_update = 1`, `enable_avg_cdf = 0`, frozen
per-frame CDFs) both processes are inert, but a split-aware adaptive-CDF path
must not update CDFs at an intermediate `tg_end`.

## Interaction with `bru_tile_active`

`bru_tile_active` is only coded when **all** of these hold:

- `enable_bru = 1` in the sequence header and
  `use_bru = 1` in the frame header (§5.18.2, inter frame only);
- the frame is not `bru_inactive`;
- `NumTiles > 1`.

Per §5.19 the bit is written for every tile in `tg_start .. tg_end`; the
expression `BruTileActives[tileRow][tileCol]` is frame state, so the writer and
reader keep the flag with the *tile*, not with the group. A tile appears in
exactly one group because §6.18 forbids overlap, so the whole-frame activity map
is unambiguous. For `NumTiles == 1` the spec infers
`BruTileActives[0][0] = 1` and codes no bit.

The decoder consumes the map in §5.20.2.1:

```text
BruTileActive = use_bru ? BruTileActives[tileRow][tileCol] : 0
...
if (BruTileActive) bru_mode S()
else bru_mode = use_bru ? BRU_INACTIVE : BRU_ACTIVE
```

so an inactive tile codes no `bru_mode` per superblock and uses
`bru_mode_info()` (zero-MV copy from `bru_ref`), while an active tile codes one
`bru_mode` symbol per superblock. This component only frames the bits; the
`use_bru` / `bru_ref` / `bru_inactive` frame-header fields and the per-SB
`bru_mode` symbols are outside it. The current M2 sequence header writes
`enable_bru = 0`, so the `use_bru = 1` path is unit-tested and pinned in the
goldens but not decoded by `dav2d` yet.

`bru_inactive = 1` (frame-level) is different from `bru_tile_active = 0`
(tile-level): the former makes the whole frame a header-only tile-group OBU
(§5.19 early return with `trailing_bits()`), the latter only disables the BRU
update inside one tile while the tile payload is still coded.

## Interaction with `disable_loopfilters_across_tiles`

`disable_loopfilters_across_tiles` is a **sequence-header** bit (§5.4.10,
semantics §6.4.10); it is not part of the tile-group framing and does not change
when a frame is split into groups. What matters:

- **Group boundaries are tile boundaries.** §6.18 coverage is a contiguous tile
  range, so a new group never cuts a tile and never creates a new filter
  boundary. Deblocking, CDEF, CCSO, GDF and loop restoration see the same tile
  geometry regardless of the number of groups.
- **The flag controls cross-tile access**, not cross-group access. With
  `disable_loopfilters_across_tiles = 1`, filters clamp at
  `MiColStart/MiColEnd/MiRowStart/MiRowEnd`; this is what makes per-tile
  encoding/decoding (and therefore per-group parallelism) independent. The M2
  post layer (`src/encoder/m2/post/loop_filters.cpp`) and the filter cores
  already model the tile starts.
- **Set it to 1 for multi-tile streams with filters on.** The single-tile/
  still writers currently write 0. This module's gate writes 1 with all
  filters off (inert, but signals the multi-tile-safe value) to prove the
  header path and the multi-group framing through `dav2d --strict 1`.
- **Pinned-decoder caveat.** `docs/negative-results.md` records that the pinned
  `dav2d` still filters across a shared tile edge even with the flag set, which
  is why the gate keeps every filter off and uses an analytically constant
  reconstruction. Do not turn filters on in a sample-exact multi-tile gate
  without re-checking that note.

## Limitations

- No CMake entry; the standalone script builds the same source set as
  `src/encoder/m2/post/build_and_test.sh` plus `tile_groups.cpp`.
- `use_bru = 1` is framing-only: the sequence/frame header flags and per-SB
  `bru_mode` symbols do not exist in the M2 writer yet.
- Adaptive CDFs with multiple tiles/groups remain unsupported by the M2 tile
  writer (`disable_cdf_update = 1`).
- The gate's source is a synthetic constant block plan, so it validates framing
  and sample-exact reconstruction, not rate-distortion quality.
- The reader needs `NumFrameHeaderBits` and `TileSizeBytes` from the parsed
  frame header, as any §5.18.1-conforming framing reader does.

# Tiles, tile groups, and SVC layer validation

Status: **component only, spec-only, no bitstream.** Nothing here is wired into
`src/main.cpp`, the root `CMakeLists.txt`, `src/entropy/av2/`, or
`src/bitstream/`. The module plans geometry and rejects invalid layer
combinations; it never writes an OBU, range-codes a tile, or allocates device
memory.

Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
(2026-05-28, commit `38f28e7`), §5.18.7.2–§5.18.7.7, §5.19, §5.20.1, §5.8,
§5.10, §5.11, §7.3.6, §9.2, and Annex F.3.1. No AVM, SVT, clave, or av2-turbo
encoder code was consulted.

## Files

| File | Purpose |
|---|---|
| `tile_layout.h` / `.cpp` | Uniform-spacing geometry, boundary flags, level checks |
| `tile_group.h` / `.cpp` | Per-tile size fields, payload offsets, OBU payload estimate |
| `svc_layers.h` / `.cpp` | Layer IDs, dependency maps, LCR/OPS validation, retention map |
| `tiles_test.cpp` | Geometry, rejection, tile-group, SVC, determinism tests |
| `build_and_test.sh` | Standalone C++20 build and test driver |

## Build and test

```sh
cd src/encoder/tiles
./build_and_test.sh
SANITIZE=1 ./build_and_test.sh   # ASan + UBSan
WERROR=1 ./build_and_test.sh
```

No CMake, no CUDA, no external dependencies. The script uses `CXX` if set,
otherwise `c++`, and the project warning set
(`-Wall -Wextra -Wpedantic -Wconversion -Wshadow`).

## Tile geometry

```cpp
auto result = av2enc::tiles::plan_tile_layout(width, height, cols, rows,
                                              /*sb_size=*/64,
                                              /*seq_level_idx=*/31,
                                              /*seq_tier=*/0);
if (!result.ok()) {
  // result.status is kInvalidGeometry / kInvalidLayout /
  // kLevelConstraint / kNotRepresentable; result.error explains it.
}
const av2enc::tiles::TileLayout& layout = result.layout;
```

The planner reproduces `uniform_spacing()` exactly. Extra superblocks go to the
**first** tiles, and the final tile extends to the end of the superblock grid.
Measured geometry from the planner (not from a bitstream):

| Frame | Layout | Tile column widths (px) | Tile row heights (px) |
|---|---:|---|---|
| 1920x1080 | 4x2 | 512, 512, 448, 448 | 512, 568 |
| 1920x1080 | 8x2 | 256x6, 192x2 | 512, 568 |
| 1920x1080 | 16x2 | 128x14, 64x2 | 512, 568 |
| 3840x2160 | 4x2 | 960x4 | 1088, 1072 |
| 3840x2160 | 8x2 | 512x4, 448x4 | 1088, 1072 |
| 3840x2160 | 16x2 | 256x12, 192x4 | 1088, 1072 |

Each `TileRect` carries visible pixel bounds, superblock origin/span, and a
`TileBoundaries` record with frame-vs-interior edges plus partial-superblock
flags. The bottom 1080p row is the partial one (568 px, ending in a 56 px strip),
not the top row.

Validation is explicit and never falls back to another layout:

- width/height positive; superblock size 64 or 128;
- columns and rows are powers of two and at most 64x64;
- the requested grid is within `maxLog2TileCols/Rows` and the level's
  `minLog2TileCols` / `minLog2TileRows`;
- the tile count respects the level's `minLog2Tiles` from `MAX_TILE_AREA`;
- `uniform_spacing()` must produce exactly the requested tile count (a 64x64
  frame cannot be 2x1);
- each tile is at least one superblock and within the level width/area bound;
- tiles cover the visible frame exactly once and the superblock grid exactly
  once.

Level checks model the video conformance bounds in §9.2 (`still_picture = 0`).
Pass `seq_level_idx = 31` for an unconstrained still-picture plan; the default
does that.

`validate_tile_layout()` rechecks a layout structurally, including raster order,
start monotonicity, boundary flags, and coverage.

## Tile-group payload plan

`plan_single_tile_group()` consumes raster-order tile byte counts and returns:

- `tile_size_bytes` (1..4) for `tile_size_bytes_minus_1`;
- `size_field_count == NumTiles - 1` (the last size is implied by the OBU
  length);
- `tile_size_minus_1` for tiles `0..N-2`;
- byte offsets of each payload inside the payload region;
- header bits/bytes, payload bytes, and the estimated OBU payload size.

Rules from §5.19/§5.20.1 that the planner enforces:

- raster `TileNum` order, never completion order;
- every tile at least 2 bytes (`init_symbol()` reads up to 15 bits);
- one contiguous byte range per tile, not split across OBUs;
- `tile_start_and_end_present_flag = 0` for the single-group packing;
- the last tile's size is not signalled;
- sizes that do not fit a 4-byte field are rejected.

The bit accounting is an estimate: `frame_header_bits` is supplied by the
integration pass because this component does not own the frame header.

## SVC / scalable layers

The SVC module is a validator, not an encoder. It models:

- `LayerId` (`xlayer_id` 0..31 with 31 = `GLOBAL_XLAYER_ID`, `mlayer_id` 0..7,
  `tlayer_id` 0..3);
- sequence layer limits and `MLayerDependencyMap` / `TLayerDependencyMap`,
  including reflexivity and transitivity;
- LCR global/local records, `lcr_xlayer_map`, per-extended-layer
  `mlayer_map`/`tlayer_map`, embedded-layer descriptors;
- operating-point sets and points, including `ops_xlayer_map` for a global OPS;
- Annex F.3.1 retention-map expansion with transitive dependencies.

Key conformance checks:

- a selected embedded/temporal layer must also select every dependency;
- LCR maps must agree with the activated sequence-header dependency maps;
- a global OPS may only select extended layers present in the LCR;
- a local OPS selects exactly its own extended layer and must be checked
  against a local LCR for that same xlayer;
- a global LCR's listed extended layers must all appear in `xlayer_map`, with
  no duplicate entries;
- a local LCR describes exactly one xlayer;
- `single_picture_header_flag` forces `seq_lcr_id = 0`, `max_mlayer_id = 0`,
  `max_tlayer_id = 0`, and `SeqMaxMlayerCnt = 1`.

**No compression gain is claimed for SVC.** AV2 layers are delivery/scalability
packaging (`docs/bitrate-features.md` §7). Do not use this module to justify a
BD-rate number.

## Tests

`tiles_test.cpp` covers:

- the 1080p/4K 4x2, 8x2, and 16x2 geometries above, including the 56 px 1080p
  bottom strip;
- exact frame and superblock coverage;
- rejection of non-power-of-two grids, oversized grids, 64x64 2x1, 1080p 64x2,
  and an 8K level-5.1 1x1 layout;
- structural validation after deliberately corrupting a rectangle, boundary
  flag, and raster index;
- tile-size field widths (256 -> 1 byte, 257 -> 2, 65536 -> 2, 65537 -> 3),
  payload offsets, last-size-implied behaviour, and rejection of 1-byte tiles;
- SVC sequence limits, dependency-map transitivity, LCR map agreement,
  local/global OPS constraints, and retention-map expansion;
- deterministic repeat plans.

All tests pass under ASan+UBSan. They are CPU-only correctness tests; no
encoder stream is produced.

## Integration plan

1. **Coordinator geometry.** The integration pass calls `plan_tile_layout()`
   once per resolution/profile, then uses `TileRect` for per-tile source
   windows, prediction availability, and per-GPU work. Do not recompute tile
   starts in another module; a mismatch is a decoder error, not a tuning choice.
2. **Tile-aware coding.** Each tile resets CDFs (`init_symbol`), above/left
   contexts, and neighbour availability at tile boundaries exactly as §5.20.2.1
   and §5.20.9.1 describe. Set `disable_loopfilters_across_tiles = 1` when tiles
   and filters coexist, or implement the explicit cross-tile filter pass.
3. **Payload assembly.** Encode tiles into independent buffers, record sizes,
   call `plan_single_tile_group()`, and write size fields plus payloads in raster
   order. Never append in GPU completion order. If any payload is short or
   missing, abort the frame; do not emit a partial frame.
4. **Decoder gate.** Test every layout with
   `tools/validate_dav2d.sh out.ivf` before relying on it. 1x1, 2x1, 2x2, 4x2,
   and 8x2 are reported to decode at `f9dafbc`; **16x2 is not yet verified.**
   Gate 16x2 before the 4-GPU default.
5. **SVC (optional).** Only after the single-layer path is stable. Validate
   sequence/LCR/OPS structures with this module, then produce a minimal
   two-layer stream and run the same decoder gate. Treat SVC as `U2` decoder
   risk.

## Blockers and open items

- **No bitstream integration.** This is a planner/validator; the integration
  pass owns OBU writing and the frame header.
- **16x2 is unverified with `dav2d`.** Do not use it in 4-GPU tile mode until a
  minimal 16x2 stream decodes.
- **SVC is unverified (`U2`).** No layered stream exists; no `--oppoint` /
  `--alllayers` extraction has been tested.
- **Tile size fields.** The planner rejects >4-byte sizes and <2-byte payloads;
  very high bitrates may force smaller tiles.
- **Adaptive CDF averaging.** The planner does not model the first-8-tile
  average or cross-GPU CDF handoff; the default `disable_cdf_update=1` path has
  no such dependency.
- **No multi-GPU validation.** See `src/multi_gpu/README.md`.

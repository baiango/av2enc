# M1 tiled still writer (`src/encoder/m1/tiles_stream/`)

Status: **reusable component + decoder gate**. It is not wired into
`src/main.cpp`, the root `CMakeLists.txt`, the M1/M2 core, or the pipeline.
Nothing outside this directory was modified.

This module turns the verified
[`src/encoder/pipeline/tiled_still_gate.cpp`](../pipeline/tiled_still_gate.cpp)
experiment into a callable writer for M1 still pictures:

- uniform tile layout through `av2enc::tiles::plan_tile_layout()`;
- tiled sequence/frame headers, including `tile_info()` and
  `tile_size_bytes_minus_1` (§5.18.7.2/§5.18.7.3). The sequence header keeps
  `seq_tile_info_present_flag = 0` (the verified form); `tile_info()` is coded
  in the frame header;
- one tile group with `tile_start_and_end_present_flag = 0` and **interleaved**
  `le(TileSizeBytes) tile_size_minus_1` / payload pairs in raster `TileNum`
  order (§5.19, §5.20.1); the last tile size is implied by the OBU length;
- per-tile cropping, encoding, CDF/context reset, and tile-local superblock
  origins;
- per-tile visible reconstructions plus a full-frame paste helper;
- a `dav2d --strict 1` gate with a byte-exact (0-difference) reconstruction
  requirement.

Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
(2026-05-28, commit `38f28e7`). No AVM, SVT-AV1/AV2, clave, or av2-turbo
encoder code was consulted.

## Files

| File | Purpose |
|---|---|
| `tiles_stream.h` | Public API: `Config`, `Tile`, `Frame`, `encode()`, low-level header/tile-group writers |
| `tiles_stream.cpp` | Layout, headers, payload extraction, tile encode loop, assembly |
| `tiles_stream_test.cpp` | Geometry, bit syntax, interleaving, reset, parity, partial tiles |
| `build_and_test.sh` | Standalone C++20 test build (no CMake, no CUDA) |
| `gate_dav2d_strict.cpp` | `dav2d --strict 1` decode gate and deblock diagnostic |
| `gate_dav2d_strict.sh` | Builds and runs the gate; defaults to 640x360 + 1080p |

## API

```cpp
#include "encoder/m1/tiles_stream/tiles_stream.h"

using namespace av2enc::m1;
using namespace av2enc::m1::tiles_stream;

Config config;
config.width = 1920;
config.height = 1080;
config.preset = av2enc::tiles::TileLayoutPreset::k4x2;
config.base_q_idx = 90;
config.enable_deblock = false;  // exact multi-tile recon (see below)

StillPicture picture;  // full-frame 4:2:0 planes
Frame frame;
std::string error;
if (!encode(picture, config, &frame, &error)) { /* error */ }

// frame.frame        ready-to-mux TD + sequence-header + CLK OBUs
// frame.clk_payload  §5.20.1 tile-group payload
// frame.tiles[i]     payload + visible recon + frame/local SB geometry
// frame.tile_size_bytes

Reconstruction recon;
assemble_reconstruction(frame, &recon, &error);  // expected decoder output
```

Lower-level pieces, exported for tests and future integrations:

- `build_layout(width, height, preset, sb_size, seq_level_idx, ...)`
- `select_tile_size_bytes(sizes, ...)`
- `write_tile_info(writer, layout, tile_size_bytes)`
- `write_frame_header(writer, config, layout, tile_size_bytes)`
- `make_sequence_header_obu(config)`
- `make_tile_group_payload(layout, config, payloads, ...)`
- `core_header_parity(config, ...)`

`encode()` runs `core_header_parity()` before doing any work. It requires the
1x1 frame header to be byte-identical to
`av2enc::m1::write_frame_header()` and the
`disable_loopfilters_across_tiles = 0` sequence header to be byte-identical to
`av2enc::m1::make_sequence_header_obu()`. A core header change therefore fails
loudly instead of producing a stream the decoder disagrees with.

### Layout support

| Frame | Supported presets | Tile counts |
|---|---|---|
| 640x360 (10 SB columns) | 1x1, 2x1, 2x2, 4x2, 8x2 | 1, 2, 4, 8, 16 |
| 1920x1080 (30 SB columns) | 1x1, 2x1, 2x2, 4x2, 8x2, 16x2 | 1, 2, 4, 8, 16, 32 |

`16x2` at 640x360 is rejected by the uniform-spacing planner (10 superblock
columns cannot yield 16 tiles). The gate reports it as `UNSUPPORTED`; it never
falls back to a different layout. Partial tiles (non-uniform visible widths and
the final partial superblock row, e.g. 1080p `512 / 568`) are encoded and gated.

### Per-tile reset and tile-local origins

Each tile is cropped and encoded as a complete still picture through
`encode_m1_still_picture()`. Consequences, all required by §5.20.2.1:

- the entropy coder starts at `init_symbol()` with a fresh CDF copy;
- the superblock walk starts at tile-local origin `(0, 0)`;
- above/left contexts are clear at the tile start and at every superblock row
  inside the tile;
- no prediction or context state crosses a tile boundary.

`Tile` records both the frame-global superblock rectangle
(`sb_col_start/sb_row_start/sb_cols/sb_rows`) and the local geometry
(`local_sb_cols/local_sb_rows`, pixel size). The writer validates that the two
agree before encoding. `tiles_stream_test.cpp` also proves the reset with a
64x64-periodic source in a 2x2 layout: all four payloads and all four
reconstructions must be byte-identical. A carried CDF or above/left context
would break that.

## Build and test

```sh
cd src/encoder/m1/tiles_stream
./build_and_test.sh
WERROR=1 ./build_and_test.sh
SANITIZE=1 ./build_and_test.sh
```

The script compiles this directory's sources with the project warning set
(`-Wall -Wextra -Wpedantic -Wconversion -Wshadow`, optional `-Werror`), and the
pre-existing core translation units with `-w`. Local results: PASS with and
without ASan/UBSan.

Covered without a decoder:

- layout counts and coverage for all supported presets;
- rejection of non-representable layouts;
- `tile_info()` bit layout and exact bit count for 1/2/4/8/16/32 tiles;
- `tile_size_bytes_for()` boundaries (1..4 bytes, <2-byte rejection);
- interleaved payload order, `le` widths, implied last size, error paths;
- 1x1 frame/sequence header parity and full OBU parity with
  `assemble_m1_still_picture()`;
- per-tile CDF/context reset via identical periodic tiles;
- partial-tile geometry with a partial superblock row;
- determinism and invalid-config rejection.

## dav2d `--strict 1` gate

```sh
cd src/encoder/m1/tiles_stream
./gate_dav2d_strict.sh
DAV2D=/path/to/dav2d ./gate_dav2d_strict.sh
./gate_dav2d_strict.sh testdata/shine_1920x1080_10f.y4m --layout 4x2
./gate_dav2d_strict.sh --deblock-diagnostic-only
```

The gate uses `dav2d --strict 1 --threads 1`, writes one IVF per case, and
requires the decoded Y/U/V planes to be **byte-identical** (0 differences) to
the pasted per-tile reconstructions. A missing pinned decoder exits 127; a
decode or conformance failure fails the gate.

Local result with `tools/bin/dav2d-macos` (`f9dafbc`), QP 90, deblock off:

| Source | Layout | Tiles | Result |
|---|---|---:|---|
| `tests/data/m0_640x360_1f.y4m` | 1x1 | 1 | PASS, 0 diffs |
| `tests/data/m0_640x360_1f.y4m` | 2x1 | 2 | PASS, 0 diffs |
| `tests/data/m0_640x360_1f.y4m` | 2x2 | 4 | PASS, 0 diffs |
| `tests/data/m0_640x360_1f.y4m` | 4x2 | 8 | PASS, 0 diffs |
| `tests/data/m0_640x360_1f.y4m` | 8x2 | 16 | PASS, 0 diffs |
| `tests/data/m0_640x360_1f.y4m` | 16x2 | — | UNSUPPORTED (not representable) |
| `testdata/shine_1920x1080_10f.y4m` | 1x1 | 1 | PASS, 0 diffs |
| `testdata/shine_1920x1080_10f.y4m` | 2x1 | 2 | PASS, 0 diffs |
| `testdata/shine_1920x1080_10f.y4m` | 2x2 | 4 | PASS, 0 diffs |
| `testdata/shine_1920x1080_10f.y4m` | 4x2 | 8 | PASS, 0 diffs |
| `testdata/shine_1920x1080_10f.y4m` | 8x2 | 16 | PASS, 0 diffs |
| `testdata/shine_1920x1080_10f.y4m` | 16x2 | 32 | PASS, 0 diffs |

That covers **1/2/4/8/16 tiles at both 640x360 and 1080p** (and the verified
16x2/32-tile extra at 1080p).

## Deblock-across-tile finding

Exact multi-tile reconstruction requires `enable_deblock = false`.

With deblocking on, the CPU reference deblocks each cropped tile as if every
tile edge were a frame edge. The pinned dav2d build still filters the **shared**
tile edge even when the sequence header sets
`disable_loopfilters_across_tiles = 1`. The gate's 2x2 diagnostic (per-tile
reconstruction vs decoder output, tolerance 16 luma samples for the longest
deblock taps plus chroma phase) reports:

| Source | Deblock | Y (total/interior) | U (total/interior) | V (total/interior) |
|---|---|:--:|:--:|:--:|
| 640x360 2x2 | on | 88 / 0 | 10 / 0 | 28 / 0 |
| 1080p 2x2 | on | 114 / 0 | 50 / 0 | 178 / 0 |

Every difference is within 16 luma samples of an interior tile edge and **zero**
are in the tile interior. With deblocking off, all layouts are byte-identical.
A 6-luma window was too small at 1080p: two V samples at luma x=952 (8 samples
from the x=960 edge) were misclassified; the longest AV2 deblock taps reach
further than 6.

Do not ship a filtered multi-tile stream and claim exact reconstruction across
tile edges until the coordinator cross-tile filter pass from
`docs/multi-gpu.md §3.2` exists (or tile mode keeps deblocking off). The
diagnostic is informational; the exactness matrix is deblock-off.

## Exact M1 core call sites

The module edits none of these; it calls the public entry points:

| Call site | Purpose |
|---|---|
| `av2enc::m1::encode_m1_still_picture(picture, options, result, error)` — declared `src/encoder/m1/m1_still.h:79`, defined `src/encoder/m1/m1_still.cpp:1456`; called from `tiles_stream.cpp:600` (`encode()` tile loop) | One fresh encoder per cropped tile: tile-local prediction, CDF/context reset, deblock (when enabled), syntax emission |
| `M1Options` / `M1Result` — `src/encoder/m1/m1_still.h` | Tool set pinned to the verified M1 reference; `result.tile_bytes` isolates the entropy payload and `result.recon_*` are the visible tile reconstruction |
| `av2enc::m1::write_frame_header()` — `src/encoder/m1/m1_headers.h:99`, `m1_headers.cpp:271`; parity at `tiles_stream.cpp:464` | Only in `core_header_parity()`: 1x1 header must match the tiled writer byte-for-byte |
| `av2enc::m1::make_sequence_header_obu()` — `src/encoder/m1/m1_headers.h:88`, `m1_headers.cpp:275`; parity at `tiles_stream.cpp:476` | Only in `core_header_parity()`: sequence header with `disable_loopfilters_across_tiles = 0` must match |
| `av2enc::tiles::plan_tile_layout()` / `preset_dims()` / `preset_name()` — `src/encoder/tiles/tile_layout.h:197/56/55` | Uniform-spacing geometry, level bounds, partial tiles, explicit rejection |
| `av2enc::tiles::tile_size_bytes_for()` — `src/encoder/tiles/tile_group.h:62` | Minimal 1..4-byte `tile_size_minus_1` field width |
| `av2enc::tiles::sb_cols()` / `sb_rows()` — `tile_layout.h:75-76` | Validate tile-local SB geometry against the plan |
| `av2enc::m0::make_obu()` / `ObuHeader` / `ObuType` — `src/bitstream/m0_headers.h` | TD / sequence-header / CLK OBU framing |

The actual tiled frame header and sequence header are written by this module
(`tiles_stream.cpp`) because the M1 core writers hardcode a 1x1 `tile_info()`
and `disable_loopfilters_across_tiles = 0`. The parity check keeps the copied
syntax honest.

## Limitations / next integration steps

- One tile group, one still picture; no partial tile groups, no multi-frame.
- 64x64 superblocks only (the M1 core does not implement 128x128).
- CDEF/CCSO/restoration are disabled, matching the verified reference; enabling
  them needs per-tile header/params work and a new gate.
- CDFs are frozen and reset per tile (`disable_cdf_update = 1`). Adaptive CDFs
  across tiles would need the coordinator's first-eight-tile average policy.
- Deblocking across tile edges is not exact with the pinned decoder (finding
  above).
- `src/main.cpp`, `CMakeLists.txt`, `tests/CMakeLists.txt`, and the M1/M2 core
  are intentionally untouched. The orchestrator can add
  `tiles_stream.cpp` to `av2enc_core`, and the two scripts as CTest entries
  with `SKIP_RETURN_CODE 77` for the decoder gate.

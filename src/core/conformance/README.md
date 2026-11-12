# Encoder-side Annex A conformance validator

Spec-only AV2 v1.0.0 (commit 38f28e7) Annex A.2/A.3/A.4/A.5 check for the
**configuration the encoder is about to signal** plus the **measured stream
statistics**. It writes no bits, allocates no CUDA memory, and changes no
encoder behaviour.

Scope: the 33 `ABSENT` rows of `docs/spec-audit/14-conformance.md`, with the
`PARTIAL` level rows (tile width/area, reserved levels) covered as well. The
result carries a stable check id, the §, and the spec page for every finding.

Status of this tree: built and tested standalone. It is **not wired into any
encoder path**; the exact call sites are listed below and left for the owner of
each file.

```
src/encoder/conformance/
  conformance_tables.h/.cpp   Annex A tables (profiles, levels, tile scaling,
                              multi-sequence configs, interoperability points)
  conformance.h/.cpp          TileGrid, Config, StreamStats, validate()
  known_plans.h/.cpp          the documented 1080p/4K matrix + DCI-4K
  conformance_test.cpp        level-boundary tests (3.1/4.1/5.1/6.1)
  plan_check.cpp              standalone printer for the plan matrix
  README.md                   this file
```

## API

```cpp
#include "encoder/conformance/conformance.h"

av2enc::conformance::Config config;
config.width = 3840;
config.height = 2160;
config.seq_profile_idc = 0;
config.seq_level_idx = 7;   // 5.1
config.seq_tier = 0;
config.chroma_format_idc = 0;
config.bit_depth_idc = 1;   // 8-bit
config.explicit_num_ref_frames = true;
config.num_ref_frames = 4;
config.tiles = av2enc::conformance::TileGrid::uniform(3840, 2160, 8, 2);
config.ci_timing_info_present = false;
config.decoder_model_info_present = false;

av2enc::conformance::StreamStats stats;
stats.fps = 60.0;
stats.max_frame_headers_per_second = 60;
stats.num_ref_frames = 4;
stats.max_bitrate_bps_over_1s = 38.0e6;
// Optional per-frame budget:
//   stats.frames[i].luma_sample_count
//   stats.frames[i].frame_obu_bytes
//   stats.frames[i].symbol_count
//   stats.frames[i].frame_parsing_time_seconds
//   stats.frames[i].decode_count_2 / closed_loop_key

const auto result = av2enc::conformance::validate(config, stats);
for (const auto& v : result.violations) {
  // v.check, v.section, v.page, v.severity, v.detail
}
if (!result.conformant()) { /* at least one kError */ }
```

Properties:

- **Returns all violations**, not the first one. Tile failures are aggregated
  per constraint with the count and the first failing tile index.
- `Severity::kError` = a normative "shall" is violated.
  `Severity::kWarning` = the level claim cannot be supported (E.4.4) or the
  profile factors are undefined (profile 31).
- `Result::not_evaluated` lists checks that the supplied statistics cannot
  decide (no BRT, no measured bytes, no bitrate). Missing data is never treated
  as a pass.
- `TileGrid::uniform()` reproduces §5.18.7.5 `uniform_spacing()`;
  `TileGrid` also accepts explicit per-tile widths so a future non-uniform
  writer is covered. TileWidth/TileHeight are §5.18.7.3 MI-based extents.
- `resolve_level()` and the `max_*` helpers expose the same limits the
  validator uses, for gates and tests.

### Per-frame inputs

`StreamStats::frames` is optional. Without it the geometry, display-rate,
header-rate, reference-frame, bitrate and buffer checks still run; the
per-frame decode budget (`LumaSampleCount`, `NumTiles <= ...`,
`CompressedSize`, `FrameSymbolCount`) is reported as not evaluated.

`FrameParsingTime` comes from `Removal[i+1] - Removal[i]` divided by
`DecodeCount`. Because no encoder path emits BRT, `FrameSample::
frame_parsing_time_seconds` can be supplied directly, or the validator falls
back to `1/fps` (halved for `DecodeCount == 2`). That fallback is a modelling
choice, not a measurement; supply the real removal interval when timing lands.

## Tables

### A.2 profile scaling (p1102)

| `seq_profile_idc` | `ProfileScalingFactor` | `PicSizeProfileFactor` | `BitrateProfileFactor` |
|---|---|---|---|
| 0, 1, 2 | 0 | 15 | 1.0 |
| 3 | 1 | 20 | 1.667 |
| 4 | 2 | 30 | 2.5 |
| 31 | — | — | — |

`PicSizeProfileFactor` is applied in `CompressedSize` and `FrameSymbolCount`;
`BitrateProfileFactor` multiplies `MaxBitrate`/`MaxBufferSize` (p1111, p1125).
`ProfileScalingFactor` is transcribed but the A.4/A.5 text never references it;
do not silently apply it. Profile 31 has no factors, so those checks are
reported as not evaluable.

A.1 chroma sets (numeric `chroma_format_idc`: 0 = 4:2:0, 1 = 4:0:0,
2 = 4:4:4, 3 = 4:2:2):

| profile | allowed chroma | interop point |
|---|---|---|
| 0 | 4:0:0, 4:2:0 | 0 |
| 1 | 4:0:0, 4:2:0 | 1 |
| 2 | 4:0:0, 4:2:0 | 2 |
| 3 | 4:0:0, 4:2:0, 4:2:2 | 1 |
| 4 | 4:0:0, 4:2:0, 4:4:4 | 1 |
| 5–30 | Reserved | — |
| 31 | 4:0:0, 4:2:0, 4:2:2, 4:4:4 | from MSDO/LCR |

### A.8/A.9 level boundaries used by the tests

Full 22-row transcription is in `conformance_tables.cpp`; these are the four
boundary levels.

| idx | level | MaxPicSize | MaxH/VSize | MaxDisplayRate | MaxDecodeRate | MaxHeaderRate | Main/High Mbps | Main/High CR | MaxTiles | MaxTileCols |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 3.1 | 1 065 024 | 1 720 | 31 950 720 | 39 938 400 | 150 | 10 / — | 2 / — | 16 | 6 |
| 5 | 4.1 | 2 359 296 | 2 560 | 141 557 760 | 155 713 536 | 300 | 20 / 50 | 4 / 4 | 32 | 8 |
| 7 | 5.1 | 8 912 896 | 4 975 | 534 773 760 | 547 430 400 | 300 | 40 / 160 | 8 / 4 | 64 | 8 |
| 11 | 6.1 | 35 651 584 | 9 951 | 2 139 095 040 | 2 189 721 600 | 300 | 100 / 480 | 8 / 4 | 128 | 16 |

`MaxNumFrameHeadersPerSec = MaxHeaderRate * (1 + (seq_tier << 1))` (p1109).

### §9.2 tile scaling factors (p622)

| level idx | tier 0 width / area | tier 1 width / area |
|---|---|---|
| 0–13 | 4 / 4 | 4 / 4 |
| 14–17 | 8 / 8 | 8 / 16 |
| 18–21 | 8 / 16 | 16 / 32 |
| 22–30 | reserved | reserved |

Tile limits: `TileWidth <= WSF * 4096 / 4`,
`TileWidth * TileHeight <= ASF * 4096 * 2304 / 4`;
`seq_tier = 1` only for level ≥ 4.0 (p1106).

`MaxLevelRefFrames` (p1109) is
`Min(8 * MaxPicSize / frame_pixels [-1 for DecodeCount == 2],
8 << explicit_num_ref_frames)`, where the `-1` applies when a
`DecodeCount == 2` frame exists and either `max_mlayer_id != 0` or one such
frame is not `OBU_CLOSED_LOOP_KEY`.

## Tests

`conformance_test.cpp` covers:

- **Tables**: 32 profile rows, 22 level rows, 5 interoperability rows, 4
  multi-sequence rows, and the exact §9.2 arrays; spot values for every
  boundary level (3.1/4.1/5.1/6.1).
- **Geometry boundaries**: exactly `MaxPicSize` passes and `+1` sample fails at
  each of the four levels (1 032², 2048×1152, 4096×2176, 8192×4352);
  `MaxHSize`/`MaxVSize`; 16 sample minimum; `MaxTiles` (level 3.1: 32 > 16);
  `MaxTileCols` (3.1: 8 > 6, 5.1: 16 > 8, 6.1: 32 > 16); tile width floor for
  non-rightmost tiles; tile width ceiling; tile area ceiling.
- **Rates**: 4K60 passes / 4K120 fails at 5.1; 1080p60/120 at 4.1;
  8K30/60/120 at 6.1; header rate 300/301 (tier 0) and 900/901 (tier 1).
- **Reference frames**: 4K 5.1 limit 8, 1080p 5.1 limit 16, the `DecodeCount
  = 2` reserve (7 at 4K), and `explicit_num_ref_frames == 0` inferring 8.
- **Per-frame budget**: exact `LumaSampleCount`, `NumTiles` parallelism,
  `CompressedSize` (2 138 400 B at 4K60/profile 0) and `FrameSymbolCount`
  (18 057 600 at 4K60/profile 0) boundaries.
- **Bitrate/buffer**: 40 Mbit/s at 5.1/profile 0; profile 3's ×1.667 ceiling;
  `MaxBufferSize = MaxBitrate × 1 s`.
- **Legality**: `seq_tier = 1` rejected at 3.1, accepted at 4.0/4.1; reserved
  levels 22–30; level 31 has no numeric level constraints; `seq_tier` cannot be
  expressed under `single_picture_header_flag`.
- **Profiles/interop/LCR**: profile/chroma mismatches, reserved profile,
  reserved bit depth, A.4 MSDO/LCR rules for IOP 0/1/2, Table A.6
  multi-sequence configs, reserved config ids.
- **Coverage**: exactly 33 audit rows; 22 implemented, 7 N/A (Annex D, no
  emission path), 4 timing-only.
- **Documented plans**: the 1080p/4K matrix, the level-6.0 witness and DCI-4K.

Build and run standalone (no CUDA, no CMake change):

```sh
clang++ -std=c++20 -I src -O2 \
  src/encoder/conformance/conformance_tables.cpp \
  src/encoder/conformance/conformance.cpp \
  src/encoder/conformance/known_plans.cpp \
  src/encoder/conformance/conformance_test.cpp -o conformance_test
./conformance_test    # -> conformance_test: PASS
```

CMake wiring (add to `CMakeLists.txt`'s `av2enc_core` sources and the test
list; this is the exact two-line shape, not applied here):

```cmake
target_sources(av2enc_core PRIVATE
  src/encoder/conformance/conformance_tables.cpp
  src/encoder/conformance/conformance.cpp
  src/encoder/conformance/known_plans.cpp)

add_executable(conformance_test src/encoder/conformance/conformance_test.cpp)
target_link_libraries(conformance_test PRIVATE av2enc_core)
av2enc_apply_warnings(conformance_test)
add_test(NAME conformance_test COMMAND conformance_test)
```

## Findings on the documented plans

`audit_documented_plans(30.0)` (30 fps isolates the geometry findings; the
bench matrix does not document a frame rate):

| plan | result |
|---|---|
| 1080p, 1/2 GPUs, 4x2/8x2, 5.1 | pass |
| **1080p, 4 GPUs, 16x2, 5.1** | **FAIL `A.4.MaxTileCols`: TileCols = 16 > MaxTileCols 8** |
| 4K, 1/2 GPUs, 4x2/8x2, 5.1 | pass |
| **4K, 4 GPUs, 16x2, 5.1** | **FAIL `A.4.MaxTileCols`: TileCols = 16 > MaxTileCols 8** |
| 4K, 4 GPUs, 16x2, **6.0** | pass (MaxTileCols = 16 at level 6.0) |
| **DCI-4K 4096×2304, 4x2, 5.1** | **FAIL `A.4.MaxPicSize`: 9 437 184 > 8 912 896** |

Both known violations are reproduced exactly:

1. **`16x2` multi-GPU default vs level 5.1 `MaxTileCols`.** 16 columns exceeds
   `MaxTileCols = 8` at 1080p and 4K. `NumTiles` is fine (32 ≤ 64) and tile
   width/area are fine, so this is the only geometry error. It affects the
   bench-documented 4-GPU default and the planner whenever `16x2` is enabled
   (`allow_unverified`, or `decoder_verified[5] = true`). Remediation: level
   6.0 (MaxTileCols = 16) or an 8x2 preset. This does not edit the planner.
2. **DCI-4K `4096×2304` vs `MaxPicSize`.** 9 437 184 > 8 912 896 at level
   5.1, independent of tiling. At 60 fps it *also* exceeds `MaxDisplayRate`
   (566 231 040 > 534 773 760), so raising `MaxPicSize` alone is not enough;
   at 30 fps the `MaxPicSize` error is the only one.

Additional findings the validator surfaces today:

- A **single-tile** 1080p stream at 300 frame headers/s violates
  `MaxTileSizeInLumaSamples * NumFrameHeadersPerSec` (2 073 600 × 300 >
  547 430 400). The spec NOTE says this constraint "has no effect on levels
  5.1 and below" because a coherent stream hits `MaxDisplayRate` first; the
  check still catches the inequality when header rate and display rate are
  supplied independently.
- A **single-tile 8K** still/frame at level 6.1 violates
  `TileWidth * TileHeight <= 9 437 184` (35 651 584) — the M0/M1 still writers
  emit exactly that when the planner is bypassed.
- An **8K** frame at level 5.1 fails `MaxPicSize`, `MaxHSize` and the tile area
  bound; 5K fails `MaxPicSize` and `MaxHSize` (4976 > 4975).
- **16 references at 4K/5.1** fails `MaxLevelRefFrames` (limit 8).
- Every non-timed video stream gets
  `E.4.4.LevelConformanceUnverifiable` (warning): no
  `ci_timing_info_present_flag`/`decoder_model_info_present_flag` means the
  level claim is not verifiable by either operating mode. This matches the
  audit's systemic finding and does not pretend the stream is verified.

## Coverage of the 33 ABSENT audit rows

`absent_constraint_coverage()` returns the machine-readable mapping. Summary:

| audit rows | area | disposition |
|---|---|---|
| 1–3 | A.2/A.3 profiles, interop, Configurable | `check:` / `table:` |
| 4–19 | A.4 levels | `check:` |
| 20–26 | Annex D alpha/atlas/resampling | `N/A:` — no emission path |
| 27–33 | Annex E timing/decoder model | 3 `check:`, 4 `timing:` |

Precisely: 21 rows are fully checked, 1 row (30, `PresentationInterval`) is
partially checked because only `MaxDisplayRate` is evaluable without
presentation timestamps, 7 are N/A and 4 are timing-only. The Annex D rows are
not silently "checked": the encoder writes no `ats_*`, background colour, atlas
segment or alpha plane, so there is nothing to validate. The Annex E
`timing:` rows need `Removal[]`/`PresentationTime[]` that no writer emits;
`E.4.4` makes that absence a reported warning instead.

## Exact call sites (not applied)

Each snippet constructs the config from values already in scope. `validate()`
is side-effect free, so the call can sit immediately before the writer's first
`put_bits()` or in the caller's pre-flight.

1. **M0 sequence header** — `src/bitstream/basic_headers.cpp:167`
   `write_sequence_header()`, after the `assert` block at lines 171–174:

   ```cpp
   av2enc::conformance::Config cc;
   cc.width = config.max_frame_width;
   cc.height = config.max_frame_height;
   cc.seq_profile_idc = 0;          // writer hardcodes 0
   cc.seq_level_idx = config.seq_level_idx;
   cc.chroma_format_idc = 0;        // writer hardcodes CHROMA_FORMAT_420
   cc.bit_depth_idc = 1;            // writer hardcodes 8-bit
   cc.still_picture = true;
   cc.single_picture_header_flag = true;
   cc.tiles = av2enc::conformance::TileGrid::uniform(
       cc.width, cc.height, 1, 1, 64);
   const auto result = av2enc::conformance::validate(cc);
   // Consumer: log result.violations (check § page detail) before emitting.
   ```

2. **M1 sequence header** — `src/encoder/m1/still_headers.cpp:77`
   `write_sequence_header()`, after `const std::size_t start`:

   ```cpp
   av2enc::conformance::Config cc;
   cc.width = c.max_frame_width;
   cc.height = c.max_frame_height;
   cc.seq_profile_idc = 0;          // still_headers.cpp:81
   cc.seq_level_idx = c.seq_level_idx;
   cc.chroma_format_idc = 0;
   cc.bit_depth_idc = 1;
   cc.still_picture = true;
   cc.single_picture_header_flag = true;
   cc.tiles = av2enc::conformance::TileGrid::uniform(
       cc.width, cc.height, 1, 1, 64);
   const auto result = av2enc::conformance::validate(cc);
   ```

3. **M2 sequence header** — `src/encoder/m2/video_headers.cpp:90`
   `write_sequence_header()`, after the asserts:

   ```cpp
   av2enc::conformance::Config cc;
   cc.width = c.max_frame_width;
   cc.height = c.max_frame_height;
   cc.seq_profile_idc = 0;          // video_headers.cpp:97
   cc.seq_level_idx = c.seq_level_idx;
   cc.seq_tier = 0;                 // video_headers.cpp:100
   cc.chroma_format_idc = 0;
   cc.bit_depth_idc = 1;
   cc.still_picture = false;
   cc.single_picture_header_flag = false;
   cc.explicit_num_ref_frames = true;   // video_headers.cpp:161
   cc.num_ref_frames = c.num_ref_frames;
   cc.tiles = av2enc::conformance::TileGrid::uniform(
       cc.width, cc.height, 1, 1, 64);
   const auto result = av2enc::conformance::validate(cc);
   ```

4. **Pipeline planner** — `src/encoder/pipeline/pipeline.cpp:488`
   `PipelinePlanner::configure()`, immediately after the successful
   `select_layout()` at line 522 (the layout and preset are final there):

   ```cpp
   av2enc::conformance::Config cc;
   cc.width = stored.width;
   cc.height = stored.height;
   cc.sb_size = stored.sb_size;
   cc.seq_profile_idc = 0;
   cc.seq_level_idx = stored.seq_level_idx;
   cc.seq_tier = stored.seq_tier;
   cc.chroma_format_idc = 0;
   cc.bit_depth_idc = 1;
   cc.still_picture = false;
   cc.single_picture_header_flag = false;
   cc.explicit_num_ref_frames = true;
   cc.num_ref_frames = stored.gpu.reference_frames;  // field name per planner
   cc.tiles.cols = layout.cols;
   cc.tiles.rows = layout.rows;
   for (const auto& tile : layout.tiles) {
     cc.tiles.tile_widths.push_back(tile.sb_cols * layout.sb_size);
     cc.tiles.tile_heights.push_back(tile.sb_rows * layout.sb_size);
   }
   const auto result = av2enc::conformance::validate(cc);
   // Return false with the first violation in `error` for a hard gate.
   ```

5. **CLI pre-flight** — `src/main.cpp:525`, after the even-dimension check
   and before constructing the writer. This is the highest-value site for the
   size envelope (it sees the input before any frame is encoded):

   ```cpp
   av2enc::conformance::Config cc;
   cc.width = static_cast<int>(y4m.width);
   cc.height = static_cast<int>(y4m.height);
   cc.seq_profile_idc = 0;
   cc.seq_level_idx = 7;            // all writers default to 5.1
   cc.chroma_format_idc = 0;
   cc.bit_depth_idc = 1;
   cc.still_picture = options.m1;   // M0/M1 still, M2 video
   cc.single_picture_header_flag = options.m1;
   cc.tiles = av2enc::conformance::TileGrid::uniform(
       cc.width, cc.height, 1, 1, 64);  // single-tile writers
   const auto result = av2enc::conformance::validate(cc);
   if (!result.conformant()) {
     // print result.violations and return 1 (or warn, for non-still plans)
   }
   ```

6. **M1 tiled-stream validation** — `src/encoder/m1/tiles_stream/
   tiles_stream.cpp:285` `validate_config()`, or after `build_layout()`
   succeeds at line 504. Build `cc.tiles` from `layout->tiles` exactly as in
   site 4; this is where `16x2` at level 5.1 becomes visible before entropy
   coding.

7. **M0 still entry** — `src/encoder/basic_encoder.cpp:2069`
   `encode_still_picture()`, after the frame-size equality check at line 2085,
   using the same `cc` as site 1. It catches a caller that bypassed
   `write_sequence_header()`.

`resolve_level()` and the `max_*` helpers are also usable as a cheap gate in
`PipelineConfig` validation (`pipeline.cpp:338`), but the tile grid does not
exist until `select_layout()`, so site 4 is the correct place for the full
check.

## Non-goals

- Not a decoder-model simulation. Annex E conformance needs BRT/timing, which
  is not emitted; only the two level-side consequences
  (`MaxLevelRefFrames`, `MaxDisplayRate`) and the E.4.4 verifiability warning
  are modelled.
- Per-frame `frame_size_override` tile grids are not modelled: `Config::tiles`
  is the sequence-maximum geometry, and every `FrameSample` uses it for
  `NumTiles`. Today's writers emit one grid per stream.
- Not an encode-path change: no writer, CMake target or manifest is modified
  by this directory.
- Not a substitute for `tools/validate_dav2d.sh`. A stream can be level
  conformant and still fail to decode; the two checks are orthogonal.

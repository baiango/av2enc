# M2 post-processing / rate-control layer

Status: **integration-ready layers, no M2 core edits.** This directory owns
per-frame rate control for M2, multi-tile frame headers/tile groups, the
post-reconstruction deblock → CDEF → CCSO chain, and the 10-bit
geometry/header checklist. It produces real bytes and has a `dav2d` gate for a
tiled two-frame stream, but it does not change `src/encoder/m2/`,
`src/main.cpp` or `CMakeLists.txt`.

Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
(2026-05-28, commit `38f28e7`). The existing components it reuses are the
source of truth for their own clauses:

| Owner | Reused for |
|---|---|
| `src/encoder/rc/` | CQP/CRF/VBR, `QpDecision`, coefficient-CDF slices |
| `src/encoder/tiles/` | uniform tile geometry, `TileSizeBytes`, tile-group sizing |
| `src/encoder/pipeline/` | verified layout policy, per-tile QP, arena/GPU plan |
| `src/encoder/filters/{deblock,cdef,restoration}/` | §7.17/§7.18/§7.19 pixel cores |
| `src/encoder/m1/filters_stream/` | §5.4.10/§5.18.7.10/§5.18.7.12 header syntax |
| `src/encoder/m2/m2_syntax.*` | per-tile partition/mode/residual payload writer |
| `src/encoder/format/` | bit-depth idc, MaxQ, base_q_idx width |

## Files

| File | Purpose |
|---|---|
| `m2_post.h` | umbrella header |
| `rc_adapter.h` / `.cpp` | M2-constrained per-frame QP decisions |
| `tiled_frame.h` / `.cpp` | multi-tile frame header, tile group, per-tile assembly |
| `loop_filters.h` / `.cpp` | post-recon deblock → CDEF → CCSO + header bits |
| `bit_depth_checklist.h` / `.cpp` | 10-bit geometry/header checklist, no 12-bit |
| `m2_post_test.cpp` | deterministic unit tests |
| `tiled_gate.cpp` | 2-frame 2×1-tile `dav2d` gate |
| `build_and_test.sh` | standalone build/run (no CMake changes) |

Build and run:

```sh
./src/encoder/m2/post/build_and_test.sh
SANITIZE=1 ./src/encoder/m2/post/build_and_test.sh
WERROR=1   ./src/encoder/m2/post/build_and_test.sh
GATE=0     ./src/encoder/m2/post/build_and_test.sh   # unit tests only
```

Current result on the macOS dev host:

```text
m2_post_test: 211 checks, 0 failures
  128x64 2x1 tiles: key_tile_bytes=3/3 tile_size_bytes=1 inter_tile_bytes=3/3
  256x64 2x1 tiles: key_tile_bytes=4/4 tile_size_bytes=1 inter_tile_bytes=4/4
  256x128 2x2 tiles: key_tile_bytes=4/4/4/4 tile_size_bytes=1 inter_tile_bytes=4/4/4/4
PASS: 3 tiled 2-frame streams decoded, all samples are the expected DC/GLOBALMV constant
```

ASan/UBSan clean. This is a CPU correctness result, not a speed or quality
result.

## API

```cpp
#include "encoder/m2/post/m2_post.h"
namespace post = av2enc::m2::post;

// 1. Rate control (one controller per encode).
post::M2RcConfig rc_cfg;
rc_cfg.rc.mode = av2enc::rc::RateControlMode::kVbr;
rc_cfg.rc.vbr.target_bitrate_bps = 500000.0;
rc_cfg.policy.max_base_q_idx = 90;     // shipped coefficient-CDF slice 0
post::M2RateControl rc;
std::string error;
if (!rc.configure(rc_cfg, &error)) { /* ... */ }

post::M2FrameQp qp;
if (!rc.begin_frame(complexity, qp, &error)) { /* ... */ }

// 2. Tile geometry.  If a pipeline::FramePipelinePlan exists, adopt its
//    already-validated layout and frame qindex instead of re-planning.
post::M2TilePlan tiles = post::plan_m2_tiles(1280, 720, 2, 1);
if (!tiles.ok) { /* tiles.error */ }
// or:
// post::M2TiledFrameHeader fh_from_pipeline;
// if (!post::adopt_pipeline_frame_plan(pipeline_plan, &fh_from_pipeline,
//                                      &tiles, &error)) { /* ... */ }

// 3. Sequence header (once).
post::M2PostSequenceOptions sequence;   // all filters off by default
auto seq_obu = post::make_m2_sequence_header_obu(config, sequence, &error);

// 4. One frame.
post::M2TiledFrameHeader fh;
fh.keyframe = false;
fh.order_hint = 1;
fh.ref_slot = 0;
fh.refresh_frame_flags = 1;
fh.base_q_idx = qp.base_q_idx;          // from rate control

post::M2TiledFrame frame;
if (!post::assemble_m2_tiled_frame(config, sequence, fh, tiles.layout,
                                   blocks, levels, &frame, &error)) { /* ... */ }
// frame.obu is one OBU_CLOSED_LOOP_KEY / OBU_REGULAR_TILE_GROUP.
if (!rc.end_frame(payload_bits, &error)) { /* ... */ }
```

`blocks` is the whole-frame preorder `m2::syntax::BlockSyntax` list (the same
shape `m2_frame.cpp` already produces), and `levels` is its coefficient array.
The layer splits it on superblock boundaries, shifts coordinates to each tile
origin, and re-enters `m2::syntax::assemble_m2_frame()` per tile.

### Why re-entering the syntax writer per tile is legal

AV2 tile boundaries are superblock-aligned. Every partition subtree is
contained in its superblock, so no subtree crosses a tile boundary (the splitter
rejects one that does). Each tile is an independent §8.2 entropy stream:

- `init_symbol(tileSize)` starts a fresh coder;
- CDFs are initialised for the tile;
- above/left partition and coefficient contexts are cleared;
- `CurrentQIndex = base_q_idx` at every tile (`§5.20.1`);
- `MiColStarts/MiRowStarts/MiColEnd/MiRowEnd` bound prediction availability.

The per-tile writer therefore starts from cleared arrays and the §9.3 defaults,
which is exactly what the decoder does **when `disable_cdf_update = 1`**. With
adaptive CDFs the decoder carries the CDF state from tile to tile and uses
`context_update_tile_id`; that is not modelled, so
`assemble_m2_tiled_frame()` rejects `NumTiles > 1` with
`disable_cdf_update = 0` rather than emitting a desynchronised stream.

## Exact frame-header fields

`write_m2_frame_header_syntax()` writes, in this order (no `is_first_tile_group`
bit; the tile-group wrapper adds that):

```text
cur_mfh_id
seq_header_id_in_frame_header
frame_is_inter                       inter only
immediate_output_frame
frame_size_override_flag = 0
order_hint                           f(OrderHintBits)
  inter: signal_primary_ref_frame, disable_cross_frame_cdf_init,
         refresh_frame_flags, frame_explicit_ref_frame_map,
         num_total_refs, ref_frame_idx,
         allow_intrabc, use_qtr_precision_mv, allow_high_precision_mv,
         is_filter_switchable, interpolation_filter
  key:   allow_intrabc
disable_cdf_update
tile_info()
    uniform_tile_spacing_flag = 1
    increment_tile_cols_log2      unary: N ones then a zero, capped by
    increment_tile_rows_log2      maxLog2TileRows
    if NumTiles > 1:
      context_update_tile_id      f(TileColsLog2 + TileRowsLog2)
      tile_size_bytes_minus_1     f(2); TileSizeBytes = value + 1
base_q_idx                            f(8) at 8-bit, f(9) at 10-bit
segmentation_enabled
using_qmatrix
delta_q_present
deblocking_filter_params()
    apply_deblocking_filter[0..1]
    apply_deblocking_filter[2..3]      only if a luma direction is enabled
    df_delta_q_present                 one f(1) per enabled plane
cdef_params()                          only when the sequence enables CDEF
lr_params()                            not emitted; restoration is rejected
ccso_params()                          only when the sequence enables CCSO
tx_mode_select
reference_select / skip_mode_present   inter only
reduced_tx_set                         f(2)
```

`context_update_tile_id` and `tile_size_bytes_minus_1` are inside `tile_info()`
and are easy to miss: the one-tile M2 core writer emits neither. They must
appear before `base_q_idx`, otherwise the decoder desynchronises at the first
tile. `TileSizeBytes` is derived from the largest assembled tile payload
(`tiles::tile_size_bytes_for()`), which is why tile payloads are encoded before
the frame header is written.

`tile_group_obu()` is then:

```text
is_first_tile_group = 1
frame_header()
if NumTiles > 1: tile_start_and_end_present_flag = 0
byte_alignment()
for TileNum in 0 .. NumTiles-1:
    if TileNum != NumTiles-1:
        le(TileSizeBytes) tile_size_minus_1
    tile payload
```

The last tile's size is implied. The size field for each tile is immediately
before that tile's payload, not a block of size fields followed by payloads.
With two tiles there is only one size field, so that mistake still decodes; it
desynchronises at NumTiles >= 3. Payloads are always raster `TileNum` order,
never GPU completion order.

## Rate control for M2

`M2RateControl` wraps exactly one `rc::RateController` and adds the M2 limits:

- `base_q_idx` is taken from `begin_frame()` and clamped (or rejected) to the
  configured policy. Defaults: `[1, 90]`.
- `90` is not arbitrary: the shipped M2 coefficient-CDF tables
  (`m2_cdf_tables.h`, `m0_cdf_tables.h`, `m1_cdf_tables.h`) are §8.3.1
  quantizer-context **slice 0** (`base_q_idx <= 90`). A decision above 90 would
  need slices 1–3 tables before it could be coded.
- `coeff_cdf_qctx == qp_slice` is derived from the raw `base_q_idx`, never from
  a tile/delta-Q/segment qindex (`docs/negative-results.md`).
- `base_q_idx_bits` is 8 at 8-bit and 9 at 10-bit; `max_quantizer` is 255/303.
- CRF and VBR are the existing placeholder models. No quality claim is made.
- `end_frame(actual_payload_bits)` must receive payload bits, not bytes and not
  OBU overhead.

QP slices (coefficient-CDF contexts) are unchanged from `rc::`:

| Slice / `qctx` | `base_q_idx` | M2 tables shipped |
|---:|---:|---|
| 0 | 0–90 | yes |
| 1 | 91–140 | no |
| 2 | 141–190 | no |
| 3 | 191–255 | no |

Set `policy.require_m2_coeff_cdf_slice = false` only after generating slice 1–3
tables and teaching the tile writer to initialize them.

## Loop filters after reconstruction

Normative order, enforced by `run_m2_post_filters()`:

```text
reconstruct -> deblock -> CDEF -> CCSO -> (loop restoration, if any)
```

- **Deblock** (§7.17) filters the reconstruction in place. It needs the
  partition/transform metadata (`DeblockMi`), qindex, skip/lossless flags and
  tile starts. `M2PostFilterFrame::init_from_recon()` installs a conservative
  default (each 4×4 cell is its own intra block); a real encode must call
  `set_deblock_block()` from its partition tree.
- **CDEF** (§7.18) reads the post-deblock samples and writes a **separate**
  destination plane, so a block never observes another block's output.
- **CCSO** (§7.19) reads the post-deblock buffer (`curr`) and the post-CDEF
  buffer (`cdef`) and filters `cdef` in place.
- Loop restoration (Wiener) would follow CCSO; this layer does not emit its
  tile syntax.

Header flags:

| Tool | Sequence flag | Frame header | Tile symbols |
|---|---|---|---|
| deblock | none | `apply_deblocking_filter[0..3]` + `df_delta_q_present` | **none** |
| CDEF | `enable_cdef` + `cdef_on_skip_txfm_*` | `cdef_params()` | `read_cdef()` per 64×64 block |
| CCSO | `enable_ccso` + `ccso_unit_matches_sb_size` | `ccso_params()` | `read_ccso()` per CCSO unit origin |
| LR | `enable_restoration` + `lr_tools_*` | `lr_params()` | `read_lr()` once per SB, before `read_partition` |

**Critical integration constraint:** the current M2 tile writer has no
`read_cdef()` / `read_ccso()` / `read_lr()` hooks. Enabling CDEF or CCSO in the
sequence/frame headers without inserting those symbols desynchronises the
stream. Deblocking needs no tile symbols and can be enabled with only the frame
header flags. The header writers here are ready; the tile hooks are integration
work in `m2_syntax.cpp` (outside this directory). Keep `enable_cdef = 0` and
`enable_ccso = 0` for streams produced today.

When the hooks are added, the call sites are fixed by the spec (see
`src/encoder/m1/filters_stream/README.md`):

- `read_cdef()` / `read_ccso()` in `inter_frame_mode_info()` /
  `intra_frame_mode_info()`, after segment/skip syntax and before
  `read_delta_qindex()` and mode syntax;
- `read_lr()` once per superblock at `bSize == SbSize`, **before** the
  `do_split` symbol;
- one `CdefTileState` / `CcsoTileState` / `LrState` per tile, never shared
  across tiles or frames.

## 10-bit checklist (no 12-bit)

`make_m2_bit_depth_checklist()` reports each item as PASS / FAIL /
NOT IMPLEMENTED.

| Item | 8-bit | 10-bit |
|---|---|---|
| `bit_depth_idc` | 1 | **0** |
| `base_q_idx` descriptor | f(8) | **f(9)** |
| MaxQ | 255 | **303** |
| chroma | 4:2:0 | 4:2:0 |
| MI grid | `2*((dim+7)>>3)` | same |
| M2 tile syntax / transform / recon | ready | **not implemented** |
| deblock / CDEF / CCSO | 8-bit ready | **not implemented (8-bit only)** |
| 12-bit | — | **reserved, always rejected** |

AV2 v1.0.0 `bit_depth_idc` is one bit: 0 = 10-bit, 1 = 8-bit. Values above 1 are
reserved, so no encoder path may emit 12-bit (`fmt::BitDepth::k12` is internal
buffer bookkeeping only). The verified 10-bit still-picture path is
`src/encoder/m10/`; it does not make M2 10-bit.

For a 10-bit M2 stream the remaining work is exactly what the checklist lists:
hbd transform/quant/recon, uint16 planes with `Clip1 = 0..1023`, DC bias 512,
the §7.14.3/§7.15.2.1 clamps, and 10-bit gates for each loop filter. Until then
keep every loop-filter flag off.

## Integration notes

1. **Do not edit `src/encoder/m2/`.** This layer calls its public
   `m2_headers.h` / `m2_syntax.h` APIs. The one-tile parity test
   (`m2_post_test`) compares the duplicated sequence header and frame OBU
   byte-for-byte against `m2::make_sequence_header_obu()` and
   `m2::make_frame_obu()`. If the core header changes, that test fails and tells
   you to re-sync.
2. **CMake (owner: integrator).** Add to `av2enc_core`:
   `src/encoder/m2/post/rc_adapter.cpp`,
   `src/encoder/m2/post/bit_depth_checklist.cpp`,
   `src/encoder/m2/post/loop_filters.cpp`,
   `src/encoder/m2/post/tiled_frame.cpp`.
   It also needs `src/encoder/rc/rc.cpp`, `src/encoder/tiles/tile_layout.cpp`,
   `src/encoder/tiles/tile_group.cpp`,
   `src/encoder/m1/filters_stream/filters_stream.cpp`,
   `src/encoder/m2/merge/merge.cpp` (the current M2 syntax writer validates
   inter modes through `merge::`), `src/encoder/pipeline/pipeline.cpp`,
   `src/multi_gpu/planner.cpp`, `src/cuda/pipeline/pipeline_gpu.cpp`,
   `src/encoder/filters/{deblock,cdef,restoration}/*.cpp`,
   `src/encoder/format/format.cpp`, which the standalone script already links.
   No CUDA symbols are required.
3. **Frame loop.** Per frame: `begin_frame` → pick layout (pipeline plan or
   `plan_m2_tiles`) → write frame header after encoding all tile payloads →
   `end_frame(payload_bits)`. Exactly one `end_frame` per `begin_frame`.
4. **CDF/context resets.** One fresh entropy state per tile. With
   `disable_cdf_update = 1` the §9.3 defaults are frozen and per-tile
   re-initialisation is correct. Do not carry `LrState`/CDEF/CCSO state between
   tiles; their §8.3 adaptation is tile-local. `CurrentQIndex` resets to
   `base_q_idx` at every tile, so per-tile QP plans never cross a tile.
5. **Tile geometry.** Never recompute tile starts. Use
   `pipeline::FramePipelinePlan::layout` or `plan_m2_tiles()` (which calls
   `tiles::plan_tile_layout()`). Plan with the same `seq_level_idx` as the
   sequence header: `plan_m2_tiles()` defaults to 7 (level 5.1), and
   `assemble_m2_tiled_frame()` rejects a layout whose level-derived
   `minLog2TileCols`/`minLog2Tiles` limits differ from `config.seq_level_idx`.
   The verified presets are 1×1, 2×1, 2×2, 4×2 and 8×2; 16×2 stays off until a
   committed gate says otherwise.
6. **Determinism.** Decisions are a pure function of the config, frame
   complexity and the actual-bits history. No clock, RNG or device query. The
   build uses `-ffp-contract=off`; `log2`/`sqrt` can differ in the last ulp
   across libm implementations.
7. **Deblock metadata.** `init_from_recon()` defaults are a safe placeholder,
   not the real partition tree. Feed `set_deblock_block()` from each decoded
   leaf's `mi_row/mi_col`, `w4/h4`, skip, region-intra and qindex, or the filter
   will use wrong edge strengths.
8. **Loop-filter headers vs. symbols.** See the table above. Deblock is safe to
   enable immediately; CDEF/CCSO are header-only until the M2 tile writer gains
   their per-block symbols.

## Tests

`m2_post_test.cpp` covers:

- **QP selection**: CQP, CRF monotonicity, VBR target direction, VBR
  determinism across controllers, M2 slice clamping and the strict no-clamp
  error, 8-bit vs 10-bit `base_q_idx` width/MaxQ, coefficient-CDF slice
  boundaries, `begin/end` ordering.
- **Tile geometry**: 2×1 and 2×2 layouts, exact tile rectangles, rejection of
  too-small/non-power-of-two/non-multiple-of-8 layouts, superblock splitting
  with coordinate shift, multi-tile header assembly, multi-tile + adaptive CDF
  rejection, rejection of a layout whose level-derived limits differ from the
  sequence level, and end-to-end adoption of a real `pipeline::PipelinePlanner`
  2×2 plan.
- **Header parity**: byte-identical sequence header and frame OBU against the
  untouched M2 core for the one-tile, filters-off case.
- **Filter ordering**: report order is deblock → CDEF → CCSO; a manual
  deblock-only → CDEF+CCSO chain matches the all-three call; the wrong order
  differs; no-filter output is a copy; 10-bit rejection; tile clipping
  metadata; CDEF/CCSO header flags and the sequence-vs-frame mismatch error;
  deblock `df_delta_q_present` bit counts.
- **10-bit checklist**: 8/10-bit pass for headers, 10-bit blocked for the tile
  path, 12-bit rejected, MaxQ overflow rejected, geometry rejection.

`tiled_gate.cpp` encodes three keyframe + P-frame streams (128×64 2×1,
256×64 2×1 with two superblocks per tile, and 256×128 2×2), wraps each in IVF,
decodes with the pinned `dav2d`, and requires two frames whose Y/U/V samples are
exactly the expected DC/GLOBALMV constant (128). It exits 77 (skip) when no
decoder is found. The 2×2 case is what caught the tile-size-field ordering bug;
two tiles alone are not enough to test §5.20.1.

The gate also enables deblocking in every frame header and enables CDEF/CCSO at
sequence level while disabling them per frame. That exercises the deblock
`apply`/`df_delta_q_present` bits, the full §5.4.10 sequence row, and the
`cdef_params()` / `ccso_params()` frame rows in front of a real decoder without
needing per-block symbols. This is the tiled stream gate; the sample-exact M2
core gate remains `src/encoder/m2/build_and_test.sh`.

## Known limitations

- CDEF/CCSO header flags are emitted and tested, but their tile symbols are not
  injected into the M2 tile writer. Do not enable them in a real M2 stream yet.
- Loop restoration (Wiener) is not emitted; `enable_restoration` is rejected by
  the frame-header writer.
- Adaptive CDFs with multiple tiles (per-tile CDF carry + context tile) are
  rejected.
- 10-bit M2 is a checklist, not an implementation.
- The rate model and CRF curve are the existing calibration placeholders. No
  BD-rate or quality claim is made.
- The tiled gate uses a constant synthetic source; it validates tile framing and
  the multi-tile header, not rate-distortion quality.

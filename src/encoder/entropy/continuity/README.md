# Tile/frame CDF continuity (AV2 v1.0.0 §5.18.2 / §5.18.7.2 / §6.17.2 / §7.4.4 / §7.5 / §8.2.2 / §8.2.4)

Spec-only module. It owns the *decisions and ordering* of AV2's tile/frame
CDF state machine; the integer arithmetic stays in
`src/encoder/entropy/avg_cdf/`, which is consumed read-only. No file outside
`src/encoder/entropy/continuity/` is modified.

The audit (`docs/spec-audit/09-entropy.md`, rows for `copyCdf`,
`context_update_tile_id`, `load_cdfs`/`blend_cdfs`, tile/frame init) found the
rules present but unwired. This module makes the missing surface explicit,
testable and callable from the M1/M2 writers.

## Normative map

| Clause | Rule | API |
|---|---|---|
| §5.18.7.2 | `context_update_tile_id` presence and `f(TileRowsLog2+TileColsLog2)` cost in `tile_info()`; `tile_size_bytes_minus_1 f(2)` follows | `tile_info_cdf_field()`, `TileInfoCdfField::tail_bits()`, `validate_context_update_tile_id()` |
| §6.17.7.3 | conformance: `context_update_tile_id < TileCols * TileRows` | `validate_context_update_tile_id()` |
| §8.2.4 | per tile `copyCdf = TileNum == context_update_tile_id`, `avgCdf = TileNum < 1<<numLog2` when `enable_avg_cdf && avg_cdf_type` | `plan_tile_exit()`, `TileExitPlan` |
| §8.2.2 | frame CDFs -> `Tile*` at every `init_symbol()` | `FrameCdfContinuity::begin_tile()` |
| §8.2.4 | `Saved*` copy/average at `exit_symbol()` | `FrameCdfContinuity::end_tile()` |
| §6.17.2 | `init_coeff_cdfs()` qctx slice from the raw frame `base_q_idx` | `qctx_slice_for_base_q_idx()` |
| §5.18.2 | `primary_ref_frame == PRIMARY_REF_NONE` / `disable_cross_frame_cdf_init` -> defaults; CLK/OLK is `FrameIsIntra` | `plan_frame_cdfs()` |
| §7.4.4 | `load_cdfs(ref_frame_idx[primary_ref_frame])`; conditional `blend_cdfs(blendFrame)` | `plan_frame_cdfs()`, `FrameCdfContinuity::begin_frame()` |
| §7.5 | `frame_end_update_cdf()`: `frame = Saved`, `frame[N] = (3*Saved[N])>>2` | `FrameCdfContinuity::end_frame()` |
| §7.23 | `save_cdfs(i)` for each refreshed slot | `FrameCdfContinuity::save_reference()` |

## Exact `tile_info()` tail

From §5.18.7.2 (PDF p.122):

```
if ((TileCols > 1 || TileRows > 1) && !IsBridge &&
    TipFrameMode != TIP_FRAME_AS_OUTPUT) {
  if (!enable_avg_cdf || !avg_cdf_type) {
    n = TileRowsLog2 + TileColsLog2
    context_update_tile_id f(n)
  }
  tile_size_bytes_minus_1 f(2)
  TileSizeBytes = tile_size_bytes_minus_1 + 1
} else {
  context_update_tile_id = 0
}
```

`TileInfoCdfField` reproduces this exactly. `tail_bits()` is the total after
the two `increment_tile_*_log2` unary fields:

| tiles | `enable_avg_cdf && avg_cdf_type` | context bits | tail bits |
|---|---|---:|---:|
| 1x1 | either | 0 (field absent) | 0 |
| 2x1 | 0 or type 0 | 1 | 3 (1 + 2) |
| 2x2 | 0 or type 0 | 2 | 4 |
| 3x3 (non-uniform) | 0 or type 0 | 4 (`CeilLog2(3)` twice) | 6 |
| 2x2 | 1 and type 1 | 0 (absent) | 2 (only `TileSizeBytes`) |
| any | any + Bridge/TIP-output | 0 (whole tail absent) | 0 |

The unit test writes the field with `BitWriter` and checks the resulting bit
count and value.

## Exact `copyCdf` / `avgCdf` choice

From §8.2.4 (PDF p.586):

```
copyCdf = 0; avgCdf = 0;
if (enable_avg_cdf && avg_cdf_type) {
  avgCdf = TileNum < 1 << numLog2;      // numLog2 = Min(3, FloorLog2(TileCols*TileRows))
} else {
  copyCdf = (TileNum == context_update_tile_id);
}
```

The controller implements the `Saved*` writes with the avg_cdf free functions
(`average_cdf`, `blend_cdf`) and `memcpy` for the copy branch. Tiles at or above
`1<<numLog2` run neither branch. `end_frame()` then applies §7.5.

## Frame installation truth table (`plan_frame_cdfs`)

Let `resolvedNone = keyframe || primary_ref_none || derived_primary_ref_none`
(§5.18.2 `choose_primary_secondary_ref_frame` returns NONE/NONE for
`FrameIsIntra` and SWITCH frames; any NONE forces
`disable_cross_frame_cdf_init = 1`).

| condition | source | entropy bytes |
|---|---|---|
| `resolvedNone` or `disable_cross_frame_cdf_init` | `init_non_coeff_cdfs()` + `init_coeff_cdfs()` (defaults, qctx slice) | reset |
| otherwise | `load_cdfs(ref_frame_idx[primary_ref_frame])` | reference |
| the above **and** `enable_avg_cdf && !avg_cdf_type && blend_ref_present && !bru_inactive && !TipFrameMode==AS_OUTPUT` | additionally `blend_cdfs(ref_frame_idx[blendFrame])` with 7/8 + 1/8 | reference blended |

CLK and OLK are `KEY_FRAME`s, so they always take the reset row.

`qctx_slice_for_base_q_idx()` uses the §6.17.2 boundaries **90 / 140 / 190**
on the raw frame `base_q_idx`, not on a tile, segment or delta-Q qindex. The
unit test cross-checks every value in `[-1, 400]` against the canonical
`src/encoder/m1/qctx/qctx.h`.

## `disable_cdf_update`

The flag is a frame-header bit that gates the coder's `adapt_cdf` call; it is
not part of the CDF bank. The controller carries it in
`FrameCdfPlan::disable_cdf_update` so a writer can assert that the bit it emits
equals the flag it gives `av2::Encoder`/`av2::Decoder`.

Important: `copyCdf`/`avgCdf` and §7.5 still run when
`disable_cdf_update == 1`; the `Tile*` arrays simply never changed. That is why
a frozen stream is bit-identical whether it resets or carries CDFs — the
production situation documented below. The unit test
`test_controller_frozen_is_reset_equivalent` pins this.

## API

```cpp
#include "encoder/entropy/continuity/cdf_continuity.h"

using namespace av2enc::entropy::continuity;

// Pure helpers.
TileInfoCdfField tile_info_cdf_field(bool enable_avg_cdf, bool avg_cdf_type,
                                     uint32_t tile_cols, uint32_t tile_rows,
                                     bool is_bridge = false,
                                     bool tip_frame_as_output = false);
bool validate_context_update_tile_id(const TileInfoCdfField&, uint32_t id,
                                     std::string* error = nullptr);
TileExitPlan plan_tile_exit(bool enable_avg_cdf, bool avg_cdf_type,
                            uint32_t tile_num, uint32_t tile_cols,
                            uint32_t tile_rows, uint32_t context_update_id);
int qctx_slice_for_base_q_idx(int raw_base_q_idx);
FrameCdfPlan plan_frame_cdfs(const FrameCdfInput&);
bool validate_frame_cdfs(const FrameCdfInput&, std::string* error = nullptr);

// Ordered bookkeeping (one per entropy context).
FrameCdfContinuity ctl;
ctl.init_owned(specs, /*reference_contexts=*/kNumRefFrames);
ctl.begin_frame(input);          // §5.18.2 set_primary_ref_frame_and_ctx(1)
for (tile = 0; tile < tile_count; ++tile) {
  ctl.begin_tile();              // immediately before init_symbol()
  // code symbols through ctl.bank().tile_cdf(i)
  ctl.end_tile(tile);            // immediately after exit_symbol()
}
ctl.end_frame();                 // tg_end == NumTiles - 1 (§7.5)
ctl.save_reference(slot);        // §7.23 save_cdfs(slot)
```

`init()` accepts existing `avg_cdf::CdfTable` views (the writers' own
`CdfState` sub-arrays); `init_owned()` allocates. `bank()` exposes
`frame_cdf(i)`, `tile_cdf(i)`, `saved_cdf(i)` and the reference contexts.

Misordered calls are rejected: `begin_tile` before `begin_frame`, double
`begin_tile`, `end_tile` out of order, `end_frame` before the last tile,
`save_reference` while a frame is open, out-of-range slots and invalid
`context_update_tile_id`.

## Exact integration seam for the M1/M2 writers

Current line numbers (verify before editing; docs go stale):

1. **`src/encoder/m2/post/tiled_frame.cpp:328-367` (`write_m2_tile_info`)**
   replaces the hard-coded `writer.put_bits(0U, n)` with
   `tile_info_cdf_field(...)` + `writer.put_bits(context_update_tile_id,
   field.context_update_bits)`. The `tile_size_bytes_minus_1 f(2)` field follows
   in both branches. The guard `:379` that rejects multi-tile adaptive CDFs is
   the policy that this API removes.
2. **`src/encoder/m2/post/tiled_frame.cpp:436`**
   `writer.put_bit(config.disable_cdf_update)` must match
   `FrameCdfPlan::disable_cdf_update`.
3. **`src/encoder/m2/post/tiled_frame.cpp:610-631`** — per-tile payload loop.
   Add `begin_frame` before the loop; around each `assemble_m2_frame` call do
   `begin_tile()` / seed the writer's `CdfState` from `bank().tile_cdf(i)` /
   `end_tile(tile_index)`; after the loop `end_frame()`. The same change at
   `:650` removes the other adaptive guard.
4. **`src/encoder/m2/m2_syntax.cpp:410`** — `init_cdfs(cdfs_, qctx_slice)`
   inside `Writer` must become "copy the tile seed" when the caller supplies
   one. `:489` is `exit_symbol`; the final `cdfs_` must be copied back into
   `bank().tile` before `end_tile`. Member at `:1992`.
5. **`src/encoder/m1/m1_syntax.cpp:316`** — same change for M1; `:404` is
   `exit_symbol`, member `:1492`.
6. **`src/encoder/m1/tiles_stream/tiles_stream.cpp:188`** — the hard-coded
   `disable_cdf_update = 1` and the `adaptive_cdf = false` policy at `:527`
   must follow the frame plan. `write_tile_info` at `:355-381` has no
   `context_update_tile_id` at all and must add it. The per-tile loop is
   `:542-634`; `encode_m1_still_picture` is called at `:600`, and
   `extract_clk_payload` at `:610`.
7. **Sequence flags**: `src/encoder/m2/m2_headers.cpp:200` and
   `src/encoder/m2/post/tiled_frame.cpp:205` hard-code `enable_avg_cdf = 0`.
   Turn this into the sequence-level flag; when 1, write `avg_cdf_type`
   immediately after it (§5.4.8).
8. **Frame flags**: every inter writer must expose
   `signal_primary_ref_frame` / `disable_cross_frame_cdf_init` (M2 currently
   always writes `false` / `true`, `m2_headers.cpp:254-255`) and pass the
   resolved `primary_ref_slot` from `ref_frame_idx[primary_ref_frame]`.

The seam order is fixed: **`tile_info()` -> quantization -> 
`set_primary_ref_frame_and_ctx(1)` (`begin_frame`) -> `init_symbol`
(`begin_tile`) -> `exit_symbol` (`end_tile`) -> `frame_end_update_cdf`
(`end_frame`) -> reference update (`save_reference`)**. `tile_info()` itself
must be parsed before `begin_frame` because `context_update_tile_id` is used by
`end_tile`.

## Why production can never exercise the carry/load/blend branches today

This is structural, not a missing test:

* `m1::syntax::Writer` (`m1_syntax.cpp:316`) and `m2::syntax::Writer`
  (`m2_syntax.cpp:410`) construct a private `CdfState` from the §9.3 defaults
  inside their constructors. Neither accepts a seed and neither exports the
  final arrays (`m1_syntax.cpp:1492`, `m2_syntax.cpp:1992`).
* `M2Config.disable_cdf_update` is `true` at both M2 frame constructors
  (`m2_frame.cpp:403,996`), so no M2 symbol ever adapts.
* Tiled M1 forces `options.adaptive_cdf = false` (`tiles_stream.cpp:527`) and
  writes `disable_cdf_update = 1` (`:188`).
* Tiled M2 rejects multi-tile adaptive CDFs (`tiled_frame.cpp:379,650`).
* Every M2 inter frame writes `disable_cross_frame_cdf_init = 1`
  (`m2_headers.cpp:255`) and every M0/M1 stream is
  `single_picture_header_flag = 1`. Both force the §5.18.2 reset row, so
  `load_cdfs`/`blend_cdfs` are never reached.
* `enable_avg_cdf = 0` at the sequence level (`m2_headers.cpp:200`,
  `tiled_frame.cpp:205`), and single-picture headers infer `1/1` but have no
  next frame.
* Consequently no emitted stream can have a frame whose first tile starts from
  a non-default CDF. Even `--adaptive-cdf` only adapts within one still
  picture.

A second, subtler point: the M2 post README
(`src/encoder/m2/post/README.md:120-127`) says the decoder "carries CDF state
from tile to tile". It does not. §8.2.2 copies the **frame** CDFs into `Tile*`
at every `init_symbol()`, and `Frame*` changes only at §7.5. So encoding each
tile independently from the frame CDFs is correct *for a reset frame*; what
requires continuity is the **frame** boundary. This module and the lockstep
test model the correct rule.

## Tests

`./src/encoder/entropy/continuity/build_and_test.sh` (`GATE=0`, `SANITIZE=1`,
`WERROR=1`, `DAV2D=...` supported).

Current results on this host:
`cdf_continuity_test` **17,261 checks**, `lockstep_stream_test` **7,979
checks**, gate PASS on 2 streams.

### `cdf_continuity_test.cpp` — independent §8.2.4 / §6.17.2 arithmetic

Every expected value is recomputed inside the test from the specification
text (`spec_average_cdf`, `spec_blend_cdf`, `spec_frame_end`), not obtained
from the controller.

* `tile_info()` bit cost: 1x1, 2x1, 2x2, 4x2, 3x3, avg/type 1, avg/type 0,
  Bridge, TIP-output; a `BitWriter` witness of the exact tail bits.
* `context_update_tile_id` conformance (`< TileCols*TileRows`) and the
  absent-field inference to 0.
* `copyCdf`/`avgCdf` choices for every tile count 1..64 and every tile index,
  including the 9-tile `1<<3` cutoff and `avg_cdf_type=0` (copy branch).
* qctx boundaries -1, 0, 89, 90, 91, 140, 141, 190, 191, 255, 303 and a full
  `[-1, 400]` cross-check against `m1::qctx`.
* Frame plan: CLK/OLK, signalled/derived PRIMARY_REF_NONE,
  `disable_cross_frame_cdf_init`, carry-over, and every blend condition
  (`enable_avg_cdf`, `avg_cdf_type`, polarities of `blendFrame`, `bru_inactive`,
  TIP output, reset precedence).
* Controller: copyCdf selects exactly tile 2 of 4; avgCdf accumulates exactly
  `1<<numLog2` tiles; §7.5 count scaling; `save_cdfs`/`load_cdfs` exactness;
  §7.4.4 blend exactness; frozen-frame reset equivalence; call-order errors;
  zero-reference-context rejection.

### `lockstep_stream_test.cpp` — a stream that carries CDFs across a frame

Three frames x 2x2 tiles, two CDF arrays (N=3 and N=4), 64 symbols per tile and
array. The encoder and decoder are independent `FrameCdfContinuity` instances
driving `av2::Encoder` / `av2::Decoder`:

| frame | CDF source | copyCdf id | qctx slice |
|---|---|---:|---:|
| 0 | CLK reset | 2 | 0 (`base_q_idx=90`) |
| 1 | `load_cdfs(slot 0)` | 1 | 1 (`120`) |
| 2 | `load_cdfs(slot 0)` + `blend_cdfs(slot 1)` | 3 | 3 (`191`) |

Frames 1 and 2 are entropy-coded against the CDFs installed by
`frame_end_update_cdf()` in the previous frame. The decoder starts each frame
from the same saved reference context and reproduces every symbol and every
`frame`/`tile`/`saved` array byte-for-byte. The test also re-encodes the same
symbols against (a) a reset frame and (b) no blend:

```
carried frame1 = 128 B, reset = 132 B
carried frame2 = 128 B, reset = 132 B
carried frame2 = 128 B, no-blend = 128 B but different bytes
carried final CDF digest 284315bbaeffd9bf
reset   final CDF digest f23bf65846acac07
no-blend final CDF digest f0ce524c864c882f
```

`--out stream.bin` writes the carried stream as a small container
(`"CDFC"`, frame count, per-frame tile count/sizes/payloads). It is raw §8.2
tile payload data, not an OBU stream.

### `gate_dav2d_continuity.cpp` — `dav2d --strict 1 --threads 1`

Two 3-frame, 256x128, 2x2-tile streams produced with the existing M2 tile
payload writer at `disable_cdf_update = 0` and a gate-local header writer:

* `copyCdf`: `enable_avg_cdf=0`, `avg_cdf_type` absent,
  `context_update_tile_id = 2` (`f(2)` bits); key + two P frames, each
  re-initialising at §5.18.2.
* `avgCdf`: `enable_avg_cdf=1`, `avg_cdf_type=1`, context id absent.

Both decode with zero reconstruction differences (all samples 128; the key
frame is DC intra and the P frames are GLOBALMV skip, all filters off):

```
frame 0 (CLK) tile_bytes=3/3/3/3 TileSizeBytes=1
frame 1 (P)   tile_bytes=4/4/4/4 TileSizeBytes=1
frame 2 (P)   tile_bytes=4/4/4/4 TileSizeBytes=1
PASS: 2 three-frame 2x2-tile adaptive-CDF streams decoded with
      --strict 1 --threads 1, all samples 128 (recon 0 diffs)
```

**Gate limits.** It proves the syntax branch, the exact `f(n)` cost, and
adaptive per-tile coding across three frames. It does **not** carry CDFs across
a frame boundary, because the core M2 writer cannot be seeded; that is
`lockstep_stream_test.cpp`'s job. A dav2d gate that carries CDFs needs the
writer change in items 1-5 above (a seed parameter and a final-CDF export), and
then a frame whose header sets `disable_cross_frame_cdf_init = 0` with a real
`primary_ref_frame`.

## Files

| File | Purpose |
|---|---|
| `cdf_continuity.h` | public API, plan structs, controller |
| `cdf_continuity.cpp` | exact decisions and ordered bank calls |
| `cdf_continuity_test.cpp` | independent §8.2.4 / §6.17.2 arithmetic |
| `lockstep_stream_test.cpp` | multi-frame multi-tile carry, encoder/decoder pair |
| `gate_dav2d_continuity.cpp` | OBU-wrapped 3-frame 2x2 dav2d gate |
| `build_and_test.sh` | host-only CPU build and gates |

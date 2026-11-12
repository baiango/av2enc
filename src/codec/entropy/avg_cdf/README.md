# Average-CDF bank (AV2 v1.0.0 §5.4.8 / §7.5 / §8.2.4)

Spec-only module. It owns the *CDF bookkeeping* for `enable_avg_cdf = 1`,
`avg_cdf_type = 1`; it does not own the entropy coder or the M1/M2 syntax
writers. No file outside `src/encoder/entropy/avg_cdf/` is modified.

Normative sources:

| Clause | What it fixes |
|---|---|
| §5.4.8 | `enable_avg_cdf`, `avg_cdf_type`; `single_picture_header_flag` infers `1, 1` |
| §5.18.7.2 | `context_update_tile_id` is absent when `enable_avg_cdf && avg_cdf_type` |
| §8.2.2 | at tile start, frame CDFs are copied to `Tile*`; only the copy adapts |
| §8.2.4 | `numLog2 = Min(3, FloorLog2(TileCols*TileRows))`; `avgCdf = TileNum < 1<<numLog2`; exact `avg_cdf()` |
| §7.5 | `FrameEndUpdateCdf()`: `Saved* -> frame*`, then `count = (3*count)>>2` |
| §7.4.4 | `save_cdfs`/`load_cdfs`/`blend_cdfs` reference contexts (type 0 blend) |

## Exact arithmetic

Each CDF array has `N+1` entries: inverse-CDF values `[0..N-2]`, the rate
index `[N-1]`, and the activation count `[N]`.

`avg_cdf(cdf, tilecdf, N)` (`tile_num == 0` first):

```
for i in 0..N-2:  cdf[i] = 1 << 15
cdf[N-1] = tilecdf[N-1]          // rate comes from tile 0
cdf[N]   = 0

for i in 0..N-2:  cdf[i] -= (32768 - tilecdf[i]) >> numLog2
cdf[N] += tilecdf[N] >> numLog2
```

`numLog2 = Min(3, FloorLog2(tile_count))`; only tiles `0 .. (1<<numLog2)-1`
run this. With 2 tiles all 2 are averaged, with 4 all 4, with 8 all 8, with 9
only 0..7. The rate entry is copied from tile 0; the count is summed after the
per-tile shift.

`average_at_frame_end()` is §7.5:

```
frame[i] = saved[i]              for all N+1 entries
frame[N] = (3 * saved[N]) >> 2
```

`reset()` writes the frozen default array into `frame`, `tile` and `saved`.
`load_tile_cdfs()` copies `frame -> tile` at `init_symbol()`.
`begin_frame(false)` copies `frame -> tile` but keeps the frame state (after
`load_reference(ctx)`); `begin_frame(true)` is `reset()`.

## API

`avg_cdf_bank.h`:

```cpp
struct CdfSpec { uint16_t symbols; const uint16_t* defaults; };
struct CdfTable { uint16_t symbols; uint16_t *frame, *tile, *saved;
                  const uint16_t* defaults; /* optional reference contexts */ };
struct CdfBankConfig {
  bool enable_avg_cdf = true;
  bool avg_cdf_type = true;
  uint32_t context_update_tile_id = 0;
  uint32_t reference_contexts = 0;  // 16 for NUM_REF_FRAMES if needed
};

class CdfBank {
  bool init_owned(span<const CdfSpec>, const CdfBankConfig&, string* error);
  bool init(span<const CdfTable>, const CdfBankConfig&, string* error);
  void reset();
  void begin_frame(bool reset_to_defaults);
  void load_tile_cdfs();
  bool save_tile_cdfs(uint32_t tile_num, uint32_t tile_count);
  void average_at_frame_end();
  bool save_reference(uint32_t ctx);
  bool load_reference(uint32_t ctx);
  bool blend_reference(uint32_t ctx);   // avg_cdf_type = 0 path
};
```

Free helpers `average_log2()`, `average_tile_limit()`, `average_cdf()` and
`blend_cdf()` expose the exact integer rules to tests and to a future CUDA
implementation.

Usage order for one frame:

```
// frame header: CLK/OLK or primary_ref_frame==PRIMARY_REF_NONE/disable_cross...
bank.begin_frame(/*reset_to_defaults=*/true);
// or, after load_cdfs(ref):  bank.load_reference(slot); bank.begin_frame(false);

for (tile_num = 0; tile_num < tile_count; ++tile_num) {
  bank.load_tile_cdfs();             // init_symbol()
  ... encode/decode with bank.tile_cdf(i) ...
  bank.save_tile_cdfs(tile_num, tile_count);  // exit_symbol()
}
bank.average_at_frame_end();          // frame_end_update_cdf(), §7.5
bank.save_reference(ctx);             // save_cdfs(ctx), §7.4.4
```

### Reset points

CLK (`OBU_CLOSED_LOOP_KEY`) and OLK (`OBU_OPEN_LOOP_KEY`) are `KEY_FRAME`s
(§5.18.2), so `FrameIsIntra = 1`, `primary_ref_frame = PRIMARY_REF_NONE`, and
both `init_non_coeff_cdfs()` and `init_coeff_cdfs()` run. Use
`begin_frame(true)` there. The same reset applies to any regular/intra-only
frame with `primary_ref_frame == PRIMARY_REF_NONE` or
`disable_cross_frame_cdf_init == 1`. Otherwise the frame loads a reference
context with `load_reference(ctx)` and calls `begin_frame(false)`.

## Bitstream evidence

`test_syntax_evidence()` reads and writes the exact fields:

| case | entropy config bits | tile_info tail (2 tiles, `TileSizeBytes=2`) |
|---|---|---|
| single-picture | none; `enable_avg_cdf=1`, `avg_cdf_type=1` inferred | `01` (no context id) |
| non-single, avg off | `0` | `0` + `01` (1-bit context id) |
| non-single, avg on/type 1 | `11` | `01` (context id absent) |
| non-single, avg on/type 0 | `10` | `0` + `01` |

So enabling average CDFs changes exactly two things at the syntax layer: one
extra sequence-header bit (`avg_cdf_type`), and removal of the
`log2(TileCols*TileRows)`-bit `context_update_tile_id` from `tile_info()`.
Nothing inside a tile payload changes on frame 0 because every tile starts
from the same frozen/averaged frame CDFs.

The writer/reader entropy test uses two tiles with different symbol
distributions. With a reset frame, `avg on` and `avg off` emit identical tile
bytes (8+8 bytes). After `frame_end_update_cdf()` the next frame starts from
different CDFs, and the same symbol streams emit 8 bytes (avg) vs 9 bytes
(copy). The decoder path uses the same bank order and reproduces the encoder's
tile and frame CDF state exactly.

## Tests

`./src/encoder/entropy/avg_cdf/build_and_test.sh` (add `GATE=0` for unit tests
only, `SANITIZE=1`, `WERROR=1`).

Current result: **555 checks pass**.

- exact `avg_cdf` values and cross-check against `av2::average_cdf`;
- `numLog2` / `1<<numLog2` cutoffs at 1, 2, 3, 4, 8, 9, 64 tiles;
- 2 tiles with different distributions, average vs copy, frame-0 equality and
  frame-1 byte change;
- 4-frame / 2-tile adaptive sequence with a CLK-style reset in the middle,
  encoder/decoder CDF lockstep, pinned final frame CDF `15193, 19866, 56, 15`
  and hash `0x6043017ccc2461f9`;
- `reset()` / `begin_frame()` / reference `save_cdfs` / `load_cdfs` /
  `blend_cdf` exact values;
- owning and non-owning banks;
- bit-level sequence-config and tile-info reader/writer evidence above.

## dav2d gate

`gate_dav2d_avg_cdf.cpp` writes a 2-tile / 2-frame M2 stream with
`enable_avg_cdf=1`, `avg_cdf_type=1` and no `context_update_tile_id`, using the
existing M2 tile-payload writer. Frame 0 is a DC-intra key frame; frame 1 is a
GLOBALMV skip. With all filters off the expected reconstruction is constant
128. Decoder invocation is exactly:

```
dav2d --strict 1 --threads 1 -q -i avg_cdf_gate.ivf.2x1.ivf -o decoded.y4m
```

Measured result (pinned `tools/bin/dav2d-macos`):

```
128x64  2x1 tiles  key 3/3 B   inter 3/3 B
256x64  2x1 tiles  key 4/4 B   inter 4/4 B
256x128 2x2 tiles  key 4/4/4/4 B  inter 4/4/4/4 B
PASS: 3 tiled 2-frame streams ... recon 0 diffs
```

The gate deliberately keeps `disable_cdf_update=1` and
`disable_cross_frame_cdf_init=1`, because the existing M2 syntax writer
initializes its private CDF state per tile. That makes the gate a pure header
test: it proves the `enable_avg_cdf=1` / `avg_cdf_type=1` syntax path is
accepted by `dav2d` with zero reconstruction changes. The actual averaged-CDF
entropy change is covered by the bank/round-trip tests above. A gate that
exercises cross-frame averaged probabilities end-to-end needs the M1/M2
syntax writer to accept a seed CDF state (integration item below).

## Integration call sites (not edited)

The requested integration API is ready; these are the exact places the tile
writer must call it. Line numbers are current at the time of writing.

1. **`src/encoder/m1/tiles_stream/tiles_stream.cpp:542-634`** — per-tile loop
   around `encode_still_picture()`. Before the call:
   `bank.load_tile_cdfs()` and seed the M1 writer with `bank.tile_cdf(i)`.
   After the call (before `extract_clk_payload`), copy the final CDFs back into
   `bank.tile` and call `bank.save_tile_cdfs(rect.index, layout.tile_count())`.
   At `:644` (after the loop) call `bank.average_at_frame_end()`.
   The current writer has `options.adaptive_cdf = false` (`:527`) and
   `writer.put_bit(true) // disable_cdf_update` (`:188`); both must follow the
   frame-level policy once seeding exists.

2. **`src/encoder/m1/still_syntax.cpp:316`** — `init_cdfs(cdfs_, qctx_slice)`.
   Replace with "copy bank tile CDF into `cdfs_`" when the bank supplies a
   non-default seed; `:404` (`enc_.exit_symbol`) is where the final `cdfs_`
   must be copied back into `bank.tile`. The private member is `:1492`.

3. **`src/encoder/m2/video_syntax.cpp:410`** — same `init_cdfs(cdfs_, ...)` in
   `Writer`; `:489` is `exit_symbol`; member `:1992`. The M2 syntax writer
   initializes per tile today, so this is the change that makes cross-frame
   `avg_cdf_type=1` observable.

4. **`src/encoder/m2/post/tiled_frame.cpp:608-631`** — per-tile payload loop
   (`split_video_blocks_for_tile` + `syntax::assemble_video_frame`). Add
   `begin_frame` before the loop, `load_tile_cdfs`/`save_tile_cdfs` per tile,
   and `average_at_frame_end` after the loop.

5. **`src/encoder/m2/video_headers.cpp:200`** and
   **`src/encoder/m2/post/tiled_frame.cpp:205`** — hard-coded
   `put_bit(false) // enable_avg_cdf`. Make this the sequence-level
   `enable_avg_cdf`; when it is 1 also write `avg_cdf_type` (§5.4.8).

6. **`src/encoder/m2/post/tiled_frame.cpp:350-365`** — `write_video_tile_info`
   unconditionally writes `context_update_tile_id`. It must be written only
   when `!enable_avg_cdf || !avg_cdf_type` (§5.18.7.2). With avg/type 1 the
   field disappears and `tile_size_bytes_minus_1` follows immediately.

7. **`src/encoder/m2/post/tiled_frame.cpp:379-383`** and
   **`:650-653`** — the current guards reject multi-tile adaptive CDFs. Once
   the bank is wired they become the integration point rather than an error.

8. **`src/bitstream/basic_headers.cpp:246`** — single-picture M1 already leaves
   `enable_avg_cdf`/`avg_cdf_type` inferred as 1/1; no bit change is needed
   there.

### Expected entropy change

On the first frame after a reset (or after any frame that starts from the same
CDFs), tile payloads are **byte-identical** with avg on or off: the flag only
affects the state installed at `frame_end_update_cdf()`. On later frames the
frame CDF is the element-wise §8.2.4 average of the first `1<<numLog2` tile
CDFs instead of a copy of `context_update_tile_id`'s tile. For an ideal symbol
with averaged probability `p_s` and copied probability `q_s`, the per-symbol
rate change is approximately

```
Δbits(s) = log2(p_s / q_s)
```

and in the test sequence it moves the second frame from 9 bytes (copy policy)
to 8 bytes (average policy). Real savings depend on how differently the tiles
adapt; the bank does not renormalize beyond the specification's shifts, so the
emitted bytes are exactly the spec's integer result.

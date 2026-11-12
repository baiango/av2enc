# Film grain stream (encoder-side bitstream half)

AV2 v1.0.0 film grain **signalling**, model estimation and slot selection.
This directory adds the encoder side of `OBU_FILM_GRAIN` and
`film_grain_config()`.

**Integration status (2026-11-12).**  The still composition root wires this
module behind `--still-grain`: `film_grain_params_present = 1`, one
`film_grain_obu()` before the coded-frame OBU, and `film_grain_config()` as the
last frame-header field.  `--still-grain` keeps the fixed two-point 4:2:0 model;
`--still-grain-adaptive` additionally runs `GrainStreamPlanner::PlanFrame()` on
the reconstruction (single frame, seed `GrainSeedForFrame(0)`) and signals the
estimated model/slot instead.  Grain output is decoder-side, so the encoder
recon stays unfiltered.  Gated by
`tools/still_verify/loop_restoration_gate.sh` (`--filmgrain 0` recon exactness
plus a `--filmgrain 1` synthesis check) at 64x64, 640x360 and 1080p for both
models.  The stale `pipeline/grain/stream/` include paths were corrected while
wiring.

Normative sources: `spec/AV2_Spec_v1.0.0.pdf` (commit `38f28e7`).

| Clause | What | File |
|---|---|---|
| §5.14 p.84 | `film_grain_obu()`, `fgm_update_flags`, `fgm_chroma_idc` | `film_grain_syntax.cpp` |
| §5.18.10.1 p.141 | `film_grain_config()` (`apply_grain`, `fgm_id`, `grain_seed`) | `film_grain_syntax.cpp` |
| §5.18.10.2 p.141-144 | `film_grain_model()` | `film_grain_syntax.cpp` |
| §6.13 p.302, §6.17.10 p.346 | conformance requirements used by the validators | `film_grain_syntax.cpp` |
| §7.3.8.8 p.387 | OBU availability and layer-dependency rules | `grain_stream.cpp` |
| §7.3.3/.4 p.381-382 | film grain OBU position inside a frame unit | `grain_stream.cpp` |
| §7.21.7.3/.4/.5 | synthesis (reused oracle) | `../spec_tables/film_grain.cpp` |
| §7.23.6 p.577 | `save_grain_model()` / `load_grain_model()` slots | `film_grain_syntax.cpp` |

## Chroma idc numbering trap

`fgm_chroma_idc` uses the §6.4.1 Table 6.2 value space, which is **not** the
usual ordering:

```
0 = CHROMA_FORMAT_420   (subX 1, subY 1)
1 = CHROMA_FORMAT_400   (subX 1, subY 1, monochrome)
2 = CHROMA_FORMAT_444   (subX 0, subY 0)
3 = CHROMA_FORMAT_422   (subX 1, subY 0)
```

§6.17.10.1 compares `FgmChromaIdc[fgm_id]` directly against
`chroma_format_idc`, so `GrainChromaIdc` keeps this numbering. 4:2:0 is `0`,
not `1`.

## API

### Syntax (`film_grain_syntax.h`)

```cpp
// §5.18.10.2 model, writer representation (cumulative point values, signed
// AR taps, explicit bit widths so every legal bitstream round-trips).
struct GrainModel;
bool ValidateModel(const GrainModel&, GrainChromaIdc, std::string* why);
GrainModel NormalizeModel(const GrainModel&, GrainChromaIdc);
std::size_t ModelBits(const GrainModel&, GrainChromaIdc);  // exact syntax bits

// §5.14 OBU payload and Annex B framing.
struct FilmGrainObu { std::uint8_t update_flags; GrainChromaIdc chroma_idc;
                      GrainModel models[8]; };
bool WriteObuPayload(BitWriter&, const FilmGrainObu&, std::string*);
bool ReadObuPayload(std::span<const uint8_t>, FilmGrainObu*, size_t*, std::string*);
std::vector<uint8_t> MakeFilmGrainObu(const FilmGrainObu&, std::string*);
bool ReadFilmGrainObu(std::span<const uint8_t>, FilmGrainObu*, size_t*, std::string*);
std::size_t FilmGrainObuBytes(const FilmGrainObu&);       // rate accounting

// §5.18.10.1 per-frame config. The three inference branches need the frame
// context, exactly like the syntax does.
struct FilmGrainConfig { bool apply_grain; int fgm_id; uint16_t grain_seed; };
struct FilmGrainConfigContext { bool film_grain_params_present;
                                bool single_picture_header_flag;
                                bool immediate_output_frame;
                                bool implicit_output_frame; };
bool ApplyGrainIsInferred(const FilmGrainConfigContext&, bool* value);
bool WriteFilmGrainConfig(BitWriter&, const FilmGrainConfig&,
                          const FilmGrainConfigContext&, std::string*);
bool ReadFilmGrainConfig(span, size_t* bit_position, ctx, FilmGrainConfig*, std::string*);
std::size_t FilmGrainConfigBits(cfg, ctx);

// §7.23 slot state (FilmGrainPresent[] + saved models).
class GrainSlotState { void Reset(); bool SaveModel(...); bool LoadModel(...); };

// §7.3.8.8 + §6.17.10.1 availability for one frame header.
bool ValidateConfigAvailability(const GrainSlotState&, const FilmGrainConfig&,
                                GrainChromaIdc frame_chroma_idc,
                                int tlayer_id, int mlayer_id, std::string* why);
```

`WriteObuPayload` appends `trailing_bits()`; `MakeFilmGrainObu` wraps it as
`leb128(size) obu_header() payload` with `obu_type = 23`. `OBU_FILM_GRAIN` is
not in `is_extensible_obu()`, so there is no `obu_extension_flag` and no
`byte_alignment()` in the payload.

### Estimation and slot planning (`grain_stream.h`)

```cpp
struct GrainFrameView;         // 8-bit planar y/u/v + strides
struct GrainEstimateOptions;   // lag, gain, activity gate, overlap, block size
struct GrainEstimate { bool apply; double activity; GrainModel model; };
GrainEstimate EstimateGrainModel(view, options);

class GrainStreamPlanner {
  bool PlanFrame(view, frame_index, tlayer, mlayer, GrainFramePlan*, std::string*);
  GrainFramePlan CancelFrame(frame_index) const;
  const GrainSlotState& slots() const;
};
```

* **Intensity** — mean `|4-neighbour Laplacian|` over interior luma. Below
  `min_activity` the frame is cancelled (`apply_grain = 0`).
* **Scaling** — nine luma buckets (`0,32,…,224,255`), each scaling value is
  `round(mean_activity_in_bucket * scaling_gain)` clipped to `0..255`.
  Consecutive equal scalings are collapsed, so a flat curve costs one point.
* **AR taps** — regularised least squares (ridge `1e-4 * trace/N`) of the
  high-pass residual against the §7.21.7.3 raster template, quantised to
  `bits_per_ar_coeff_*_minus_5` with `ar_coeff_shift_minus_6`. Chroma reuses
  the luma spatial taps and leaves the extra luma-grain tap at zero.
* **Chroma** — `chroma_scaling_from_luma = 1` by default (chroma point curves
  are not estimated); 4:2:0 requires both or neither chroma curve, which this
  trivially satisfies.
* **fgm_id** — 8-slot LRU over exactly-equal models. A repeated model reuses
  the slot and emits no OBU; a new model overwrites the least-recently-used
  slot and returns a one-slot `FilmGrainObu` in the plan. `emit_model` tells
  the caller whether it must transmit.
* **grain_seed** — `(0x9E37 * (frame_index + 1) + 0x1D2F) & 0xffff`.
* **Cancellation** — `GrainPlanOptions::disable` or a below-threshold estimate
  returns `apply_grain = 0` and leaves slot state untouched; models are only
  written when `emit_model` is true.

The estimator is deterministic for a given build (fixed iteration order, no
randomness). Goldens are pinned only for the syntax layer, whose output is
platform-independent; the estimator is checked for repeatability, validity and
the intensity/curve properties.

### Stream assembly

```cpp
bool ParseTemporalUnit(unit, std::vector<GrainObuView>*, std::string*);
bool InsertFilmGrainObu(unit, obu_bytes, out, std::string*);  // before frame OBU
bool ReadSequenceHeaderFilmGrainPresent(unit, bool*, std::string*);
bool PatchSequenceHeaderFilmGrainPresent(unit, bool, std::string*);
```

`InsertFilmGrainObu` inserts immediately before the coded-frame OBU
(CLK/OLK/tile-group/SEF/TIP/bridge/RAS), matching §7.3.3/.4, and rejects a
second film grain OBU in the same unit. `PatchSequenceHeaderFilmGrainPresent`
flips the bit in place; the repo's sequence writers end with
`seq_tile_info_present_flag`, `film_grain_params_present`,
`obu_extension_flag`, `trailing_bits()`, so the target bit is the one two
positions before the last set bit. The helper checks the extension bit is 0.

## Integration sites (when the core files are in scope)

Nothing here is called by the shipped `av2enc` CLI. To wire it:

1. **`src/bitstream/basic_headers.h`** — add `kFilmGrain = 23` to `av2enc::basic::ObuType`
   (the module currently uses a checked cast, so no core edit is required for
   the gate).
2. **Sequence writers** (`src/bitstream/basic_headers.cpp:268`,
   `src/encoder/m2/video_headers.cpp:219`, `m1/still_headers.cpp:159`,
   `hbd/hbd_headers.cpp:171`, `sb128/sb128_stream.cpp:506`,
   `sb256/sb256_stream.cpp:471`, `inter/leading/leading.cpp:208`,
   `inter/random_access/random_access.cpp:376`,
   `inter/tlayers/tlayers_syntax.cpp:247`,
   `inter/longterm/longterm_headers.cpp:363`,
   `pipeline/tiled_still_gate.cpp:200`) — replace the hard-coded
   `put_bit(false)` with the writer's `film_grain_params_present`.
3. **Frame headers** — append `grain::WriteFilmGrainConfig(writer, cfg, ctx,
   &error)` at the *end* of `frame_header_info()`, i.e. after
   `global_motion_params()`/`reduced_tx_set` and before `byte_alignment()`.
   `av2enc::video::write_frame_header_syntax()` is the first call site; the M1 still
   writer is next.
4. **Per frame** — `GrainStreamPlanner::PlanFrame()` on the *reconstruction*
   luma/chroma, then `MakeFilmGrainObu(plan.obu)` before the frame OBU
   (`video_frame.cpp` builds its unit from TD + [SEQ] + frame) and pass
   `plan.fgm_id`/`plan.grain_seed` to step 3.

### Keeping the current `film_grain_params_present = 0` default byte-identical

* The sequence bit is already written unconditionally. Keep the config
  default `false`; the emitted bit stays 0 and every existing golden is
  unchanged.
* `WriteFilmGrainConfig` writes **zero bits** when
  `film_grain_params_present == 0` (or when the frame is neither immediately
  nor implicitly output), so appending the call to a frame header is a no-op
  for every current stream. The unit test pins this (`silent_writer.bit_size()
  == 0`), and the gate checks the base stream is identical with
  `--filmgrain 0` and `--filmgrain 1`.
* The planner is opt-in: unless `PlanFrame` returns `emit_model`, no OBU is
  produced, and `apply_grain` stays 0.

## Tests

`film_grain_stream_test` (206 checks, warning-clean with `-Werror`):

* §6.4.1 chroma-idc mapping and reverse.
* Hand-derived **byte goldens** for two complete OBU payloads (4:2:0 with
  `chroma_scaling_from_luma`, lag 2, restricted range; 4:4:4 with chroma point
  curves, cb/cr mults, lag 1) and the full Annex B framing
  (`30 5c …`, `1f 5c …`), plus exact `ModelBits` counts.
* Round trips for 4:2:0 / 4:4:4 / 4:0:0 / lag 0 and lag 3 / `-128..127` taps.
* `film_grain_params_present` inference: 0-bit, 1-bit and 19-bit
  single-picture forms with byte goldens (`db ee f0`, `b7 dd e0`); rejection
  of contradictions (`apply_grain` forced by inference, `fgm_id = 8`).
* Invalid-range rejection: 15 points, non-increasing points, value 256,
  chroma points with `chroma_scaling_from_luma`, 4:2:0 cb/cr mismatch,
  `grain_scaling_minus_8`/`ar_coeff_lag`/shift > 3, coefficient outside its
  coded width, `cb_offset = 512`, increments/scalings wider than their bits,
  `fgm_update_flags == 0`, chroma idc 4.
* Slot state: `save_grain_model` overwrite, `load_grain_model` on an empty
  slot fails, `Reset()` clears `FilmGrainPresent`, availability rejects an
  unknown slot, a chroma mismatch and a future temporal layer.
* Planner: emit/reuse/cancel, seed sequence, slot exhaustion (LRU overwrite),
  flat-frame cancel.
* Estimator: repeatability, validity, AR bounds, flat-frame cancel, reference
  synthesis conversion and reproducibility.
* Stream assembly: OBU order (TD, SEQ, grain, frame), one model OBU per unit,
  sequence-bit patch idempotence with exactly one changed byte.

## Decoder gate

`film_grain_stream_gate` builds its own M2 streams with the existing encoder
and runs `dav2d --strict 1 --threads 1`:

1. **Base** — decoder output equals the M2 reconstruction; `--filmgrain 0`
   equals `--filmgrain 1` (no grain OBU present).
2. **Phase A** — `film_grain_params_present = 1`, a §5.14 OBU before the frame
   OBU, and `film_grain_config()` appended to every frame header. Decoded
   output equals `spec_tables` synthesis applied to the reconstruction, for
   every frame and every plane (exact equality, raster order, grain on each of
   the estimator's three models).
3. **Phase B** — `apply_grain = 0` with `film_grain_params_present = 1` and no
   OBU: output is byte-identical to the base stream.
4. **Phase C** — slot 0 holds model A (frame 0), frame 1 reuses it with no new
   OBU, then slot 0 is overwritten with model B for frame 2. All three frames
   match the reference synthesis; A and B produce different grain.

Measured on the default 96x96 3-frame clip at qp 90:

```
base stream                    2098 B
film grain OBUs (3 models)      173 B  (8.25% of base)
film_grain_config()              60 bits (20 bits/frame with grain)
total grain syntax              181 B
per-model syntax           427-443 bits (lag 2, 12 taps, 9-point curve)
```

On `testdata/shine_1280x720_60f.y4m` (3 frames, qp 90): base 103,140 B,
film grain OBUs 189 B (**0.18%**), config 60 bits, models 475 bits each.  The
percentage is content- and resolution-dependent; the OBU cost is fixed by the
model syntax (point curves + AR taps), not by the picture size.

For a stream that reuses one model, the steady-state cost is
`apply_grain f(1) + fgm_id f(3) + grain_seed f(16) = 20 bits/frame`, with no
OBU. A single-picture stream costs 19 bits (apply_grain is inferred).

Run:

```sh
./src/encoder/filters/film_grain_stream/build_and_test.sh
DAV2D=/path/to/dav2d ./src/encoder/filters/film_grain_stream/build_and_test.sh
./src/encoder/filters/film_grain_stream/build_and_test.sh --test-only
SANITIZE=1 ./src/encoder/filters/film_grain_stream/build_and_test.sh --test-only
```

`WERROR=1` applies to the unit-test build. The gate build drops `-Werror`
because `src/encoder/m2/video_syntax.cpp` has pre-existing
`-Wsign-conversion` warnings.

## Limitations / follow-ups

* The gate rewrites an existing M2 stream (sequence bit + frame-header tail +
  inserted OBU) instead of driving `video_headers.cpp` directly, because core
  files are out of scope for this stream. The rewrite is byte-for-byte the
  same syntax the integration recipe above produces.
* SEF re-showing (§5.18.10.1's `save_grain_params(frame_to_show_map_idx)`
  after `film_grain_config()`) is not modelled. The planner only plans coded
  output frames.
* Multi-layer dependency maps are assumed to be the §6.4.2 defaults
  (`refTLayer <= currTLayer`, `refLayer <= currLayer`); the repo's streams
  have `max_tlayer_id = max_mlayer_id = 0`, so every slot satisfies them.
* `fgm_chroma_idc` is validated against the frame's chroma format, but only
  4:2:0 is exercised end-to-end; 4:0:0 / 4:2:2 / 4:4:4 models are covered by
  syntax round-trip tests.


# M2 integration — exact call sites

Status: **the leading layer is ready; `src/encoder/m2/` is not edited.**
This is the patch plan. Line numbers are from the current snapshot; search by
symbol if another agent's edits move them.

Two integration levels are possible:

1. **External framing (no M2 edits).** Encode key/P tile payloads with the
   existing M2 encoder, then wrap them with
   `leading::assemble_temporal_units()`. This is what `leading_gate.cpp` does.
2. **Native M2 path.** Add a leading schedule to `encode_video_sequence()`. The
   exact sites are below.

Both need the same header/plan fields, so the invariants are listed first.

---

## 1. Invariants the M2 patch must preserve

Sequence level (`VideoConfig`):

* `monotonic_output_order_flag = 0` for every leading stream. An OLK has
  `immediate_output_frame` inferred 0 and this profile writes no second output
  frame in the OLK temporal unit, so monotonic order 0 is the only way to output
  the OLK via `implicit_output_frame = 1`.
* `num_ref_frames` must be the plan's slot count (the gate uses 8).
* `order_hint_bits` must cover every order hint.
* `explicit_ref_frame_map = 1`, `explicit_num_ref_frames = 1` already hold.
* `long_term_frame_id_bits = 0` (current value). Leading frames cannot reference
  pre-OLK normal frames in this profile; the OLK key frame invalidates them.

Per-frame header (`leading::FrameHeader`):

| Frame | `order_hint` | `refresh_frame_flags` | refs | output |
|---|---|---|---|---|
| CLK | 0 | all slots | none | immediate = 1 |
| OLK | `L` (the leading count) | slot 0 | none | implicit = 1 |
| leading `j` (0..L-1) | `1 + j` | slot `1 + j` | slot 0 (OLK) | non-output |
| regular | `L + 1 + j` | slot `L + 1 + j` | slot 0 (OLK) | immediate = 1 |

`L = 3` is what the gate uses. Slots 0..6 are enough; `num_ref_frames = 8`.
`validate_plan()` rejects any deviation from the leading refresh and regular
reference rules.

---

## 2. `src/encoder/m2/video_headers.h`

### `VideoConfig` (`video_headers.h:21`)

Add:

```cpp
// §5.4.1 monotonic_output_order_flag.  Keep true as the default so existing
// M2 streams stay byte-identical.  Leading/OLK streams set it to false.
bool monotonic_output_order_flag = true;
```

Keep `num_ref_frames` (line 27), `order_hint_bits` (line 22) and
`num_total_refs` (line 30) as they are; the leading plan supplies per-frame
`num_total_refs` and `ref_frame_idx`.

### `VideoFrameHeader` (`video_headers.h:42`)

Extend it, or replace the header writer call for leading streams with
`leading::make_frame_obu()`. The fields the writer needs are:

```cpp
#include "encoder/inter/leading/leading.h"

struct VideoFrameHeader {
  bool keyframe = false;
  leading::FrameKind kind = leading::FrameKind::kRegularTileGroup;
  std::uint32_t order_hint = 0;          // code LSBs, not absolute hint
  std::uint32_t refresh_frame_flags = 1;
  std::uint8_t num_total_refs = 1;
  std::array<std::uint8_t, 8> ref_frame_idx{};
  bool leading_output_frame = false;     // leading only; false = non-output
};
```

`ref_slot` can remain as a compatibility alias for `ref_frame_idx[0]`, but do
not write it twice.

---

## 3. `src/encoder/m2/video_headers.cpp`

| Site | Current | Required |
|---|---|---|
| `write_sequence_header`, line 110 | `writer.put_bit(true); // monotonic_output_order_flag` | `writer.put_bit(c.monotonic_output_order_flag);` |
| `write_frame_header_syntax`, line 240 | `writer.put_bit(true); // immediate_output_frame` | Derive from `h.kind` / `h.leading_output_frame` per §5.18.2; write `implicit_output_frame` only when `!immediate && !c.monotonic_output_order_flag` |
| line 238 | no refresh for key | For `kOpenLoopKey`, write `refresh_frame_flags f(NumRefFrames)` after `order_hint` (CLK still infers `allFrames`) |
| line 265 | `writer.put_bits(c.num_total_refs, 3U);` | `writer.put_bits(h.num_total_refs, 3U);` |
| lines 266–267 | one `h.ref_slot` | loop `h.num_total_refs`, each `ref_frame_idx[r]` with `CeilLog2(NumRefFrames)` bits |
| `make_frame_obu`, line 328 | type is `keyframe ? CLK : REGULAR` | select `CLK`, `OLK`, `LEADING_TILE_GROUP`, `REGULAR_TILE_GROUP` from `h.kind`; use leading::ObuType values 4/5/6/7 |
| `make_sequence_header_obu`, line 304 | unchanged | automatically honours the new `VideoConfig` bit |

The cheapest correct implementation is to delegate the whole frame OBU to
`leading::make_frame_obu(config, header, tile)` and the sequence header to
`leading::make_sequence_header_obu(config)`. That writer has the same fixed
tool set as M2, and `leading_test.cpp` already proves byte equality with the M2
writer when `monotonic_output_order_flag = 1`.

Do **not** change the ordering of `refresh_frame_flags` for inter frames: it
comes after `signal_primary_ref_frame`/`disable_cross_frame_cdf_init` (§5.18.2).
For key frames it comes after `order_hint`, before `frame_size()`.

---

## 4. `src/encoder/m2/video_frame.h`

### `VideoOptions`

Add:

```cpp
// Open-GOP leading pictures between the OLK and the first regular frame.
int leading_frames = 0;
// false = coded non-output frame units (recommended); true = immediate output.
bool leading_output_frames = false;
```

### `VideoSequenceResult`

`recon` is currently indexed by display order. Leading decode order differs, so
add a display mapping or make the arrays explicitly display-indexed:

```cpp
std::vector<std::uint32_t> coded_display_index;  // decode order -> display
std::vector<std::uint32_t> display_coded_index;  // display -> decode
```

Keep `units` in **decode order** (the order they are written to IVF). The gate
must compare decoder output to `recon[display]`, not `recon[decode]`.

---

## 5. `src/encoder/m2/video_frame.cpp`

### Entry point (`encode_video_sequence`, line 929)

After the existing dimension and plane checks (lines 941–967), build and
validate the leading plan before encoding any picture:

```cpp
leading::SequenceConfig lc;
lc.max_frame_width  = static_cast<std::uint16_t>(width);
lc.max_frame_height = static_cast<std::uint16_t>(height);
lc.base_q_idx = options.base_q_idx;
lc.order_hint_bits = 4;              // 8-frame GOP
lc.num_ref_frames = 8;
lc.monotonic_output_order_flag = false;

leading::Plan plan = leading::make_plan(options);  // new helper, see below
std::string plan_error;
if (!leading::validate_plan(lc, plan, plan_error)) {
  error = "leading plan: " + plan_error;
  return false;
}
```

For `leading_frames == 0` keep the current path unchanged so the pinned M2
goldens do not move.

### Key/OLK picture (line 974)

The OLK is the key picture of the open GOP, not display frame 0:

```cpp
const int olk_display = options.leading_frames;  // L
InterFrameEncoder olk_encoder(frames[olk_display], empty_reference, options,
                              options.trace, /*keyframe=*/true);
```

Use its tile for the `kOpenLoopKey` header at plan index 1. The CLK at plan
index 0 remains a separate `keyframe=true` encode of display frame 0 with
`kClosedLoopKey`.

### Leading pictures (between the key block and the P loop, line 1026)

Leading frames are decoded after the OLK but displayed before it. Encode them
in plan order against the OLK reconstruction:

```cpp
const VideoReconFrame& olk_recon = result.recon[olk_display];
for (int j = 0; j < options.leading_frames; ++j) {
  const int display = 1 + j;
  InterFrameEncoder enc(frames[display], olk_recon, options, options.trace,
                        /*keyframe=*/false);
  std::vector<std::uint8_t> tile;
  std::vector<syntax::BlockSyntax> blocks;
  std::vector<std::int32_t> levels;
  if (!enc.encode(tile, blocks, levels, error)) { ... }

  leading::FrameHeader h;
  h.kind = leading::FrameKind::kLeadingTileGroup;
  h.order_hint = static_cast<std::uint32_t>(display);
  h.refresh_frame_flags = 1U << (1 + j);
  h.num_total_refs = 1;
  h.ref_frame_idx[0] = 0;
  h.leading_output_frame = options.leading_output_frames;
  // make_frame_obu(lc, h, tile) -> unit
}
```

### P loop (line 1026 onward)

* Replace `const VideoReconFrame& reference = result.recon.back();` (line 995) with
  the plan's reference slot. For the gate plan every regular frame references
  the OLK: `const VideoReconFrame& reference = ref_slots[plan.frames[i].ref_frame_idx[0]];`
* Replace `header.order_hint = i` (line 1039) with the plan header's hint.
* Replace `header.refresh_frame_flags = 1` (line 1041) with the plan header's
  flags.
* The tile is unchanged by the header fields, so the existing
  `InterFrameEncoder` output can be reused.

### Unit assembly (lines 1007–1016 and 1044–1050)

Use `leading::assemble_temporal_units(lc, plan, tiles)` instead of hand-building
TD/SEQ/frame OBUs. It already:

* emits one TD per frame;
* repeats SEQ at the CLK and at the OLK (random access availability, §7.3.8);
* emits no SEQ at leading/regular units.

If the native path keeps the current assembly, the three changes are:
`make_sequence_header_obu(config)` at line 1008 must use the leading config,
`make_frame_obu(config, header, tile)` at lines 1010/1045 must use the
leading OBU type, and `result.units.push_back` must occur in decode order.

---

## 6. `src/encoder/m2/video_gate.cpp`

| Site | Current | Required |
|---|---|---|
| line 273 `run_phase` | compares decoder frames to `sequence.recon` in input order | compare to display-indexed `recon`; leading non-output frames must be absent from decoder output |
| line 333 decode command | `--threads 0` | `--strict 1 --threads 1` for the leading gate |
| line 439 phase list | two-frame P / short clip | add a leading phase that writes the full stream and a RAP clip (drop the CLK IVF frame) and compares both |

Expected output counts for the gate clip (8 coded frames, 3 non-output leading):

* full stream: 5 output frames (`CLK, OLK, REG5, REG6, REG7`);
* RAP clip starting at the OLK: 4 output frames (`OLK, REG5, REG6, REG7`);
* the two must be identical after dropping the full stream's first frame.

---

## 7. Checklist

- [ ] `VideoConfig::monotonic_output_order_flag` added and written at
      `video_headers.cpp:110`; default `true` preserves old streams.
- [ ] `VideoFrameHeader` carries `kind`, `num_total_refs`, `ref_frame_idx[]`,
      `leading_output_frame`.
- [ ] Key branch writes OLK refresh flags; CLK still infers `allFrames`.
- [ ] `immediate_output_frame`/`implicit_output_frame` derivations match
      `leading::output_flags()`.
- [ ] `make_frame_obu` selects OBU type 4/5/6/7 from `kind`.
- [ ] `encode_video_sequence()` builds and validates `leading::Plan` before
      encoding.
- [ ] OLK is the key picture at display index `L`; leading frames are encoded
      against the OLK and decoded after it.
- [ ] Regular frames reference only the OLK slot; never a leading slot.
- [ ] `units` stay in decode order; `recon`/stat arrays are display-indexed.
- [ ] `video_gate` compares display-order output and checks the RAP clip.
- [ ] `leading_frames == 0` leaves the current M2 bitstream path untouched.

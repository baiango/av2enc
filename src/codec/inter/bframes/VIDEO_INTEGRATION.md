# M2 integration notes — exact call sites

Status: **the reference/hierarchy layer is ready; the M2 core is not edited.**
These notes are the patch plan for `src/encoder/m2/` and are deliberately
specific about fields and call sites. Line numbers are from this snapshot
(`video_headers.cpp` / `video_frame.cpp`); search by symbol if another agent's edits
move them.

The layer itself is `src/encoder/inter/bframes/bframes.h`. It writes no bits. The
M2 stream is only correct if the fields below are written exactly as the plan
says **and** `bframes::validate_plan()` passes.

---

## 1. Build the plan before encoding

`encode_video_sequence()` starts at `src/encoder/m2/video_frame.cpp:895`. Add the
include near the top of the file:

```cpp
#include "encoder/inter/bframes/bframes.h"
```

Before the keyframe encoder is constructed (`video_frame.cpp:941`), build and
validate the schedule:

```cpp
using namespace av2enc::inter::bframes;

// The M2 gate currently feeds 2, 8, ... frames. Choose a gop_size that divides
// the input; this layer builds complete GOP units only.
const int gop_size = 8;                 // 4/8/16 supported; 2 for a 2-frame clip
SequenceConfig sc;
sc.num_gops = static_cast<int>(frames.size()) / gop_size;
sc.gop_size = gop_size;
sc.pattern = GopPattern::kHierarchicalB;  // or kIpbb / kLowDelayP
sc.order_hint_bits = (gop_size == 16) ? 5 : 4;
sc.num_slots = 16;                        // must cover the live frames; see §5
sc.base_q_idx = options.base_q_idx;
sc.leading_pictures = 0;                  // M2 writes no leading OBUs yet

Plan plan;
PlanError plan_error = PlanError::kOk;
if (!build_sequence(sc, plan, &plan_error)) {
  error = std::string("B-frame schedule: ") + plan_error_name(plan_error);
  return false;
}
if (plan.display_count() != static_cast<int>(frames.size())) {
  error = "input frame count is not a whole number of GOP units";
  return false;
}

ValidationReport validation;
if (!validate_plan(plan, &validation)) {
  error = std::string("B-frame plan invalid: ") + plan_error_name(validation.error) +
          " (coding " + std::to_string(validation.error_coding_index) +
          ", display " + std::to_string(validation.error_display_index) + ")";
  return false;
}
```

`build_*` already self-checks; the explicit `validate_plan()` is the gate and
must stay. Do not write a sequence-header OBU or any frame OBU until it passes.

If the input count is not a multiple of `gop_size`, either pick a divisor, or
extend the builder for a truncated final unit (not implemented here).

---

## 2. Sequence header (`VideoConfig`) fields

`VideoConfig` is in `src/encoder/m2/video_headers.h:21`.

| Field | Current M2 value | B-frame value | Writer |
|---|---|---|---|
| `order_hint_bits` | 4 (`video_frame.cpp:960`) | `plan.order_hint_bits`; **5 for a 16-frame GOP**, 4 for ≤8 | `video_headers.cpp:157` `order_hint_bits_minus_1` |
| `num_ref_frames` | 2 (`video_frame.cpp:961`) | `plan.num_slots` (≤16). A hierarchical-8 plan needs at least 4 safe slots; use 8/16 for margin | `video_headers.cpp:166` `num_ref_frames_minus_1` |
| `num_total_refs` | 1 (`video_frame.cpp:962`) | **remove/ignore**; `num_total_refs` is per frame now | `video_headers.cpp:265` |
| `drl_reorder` | from options | unchanged | — |
| new `monotonic_output_order_flag` | implicitly `true` (`video_headers.cpp:110`) | **`false`** for any stream with `implicit_output_frame = 1` (all B schedules) | `video_headers.cpp:110` |

The sequence header already has:

* `explicit_ref_frame_map = 1` (`video_headers.cpp:164`),
* `explicit_num_ref_frames = 1` (`:165`),
* `num_ref_frames_minus_1` (`:166`),
* `enable_short_refresh_frame_flags = 0` (later in the inter config), so every
  non-key frame codes `f(NumRefFrames)` refresh bits.

### `monotonic_output_order_flag` is the critical one

`video_headers.cpp:110` currently writes `true`. In `frame_header_info()` the
`implicit_output_frame` syntax bit is only present when

```text
if (IsBridge || immediate_output_frame || monotonic_output_order_flag)
    implicit_output_frame = 0
else
    f(1) implicit_output_frame
```

With `monotonic_output_order_flag = 1`, a frame with
`immediate_output_frame = 0` has `implicit_output_frame = 0` and is **never
output** by the decoder. B-frame output reordering needs
`monotonic_output_order_flag = 0` and `implicit_output_frame = 1` on every
reordered frame. Add a `bool monotonic_output_order_flag = false;` to
`VideoConfig`, write it at `video_headers.cpp:110`, and use it in the frame-header
branch below.

---

## 3. Frame header (`VideoFrameHeader`) fields

Current header (`video_headers.h:45`):

```cpp
struct VideoFrameHeader {
  bool keyframe = false;
  std::uint32_t order_hint = 0;
  std::uint8_t ref_slot = 0;             // single reference
  std::uint32_t refresh_frame_flags = 1;
};
```

Replace/extend it with the plan's per-frame fields:

```cpp
struct VideoFrameHeader {
  bool keyframe = false;
  // Coded order_hint LSBs, NOT the absolute hint.
  std::uint32_t order_hint_lsbs = 0;
  std::uint32_t refresh_frame_flags = 0;
  bool frame_explicit_ref_frame_map = true;
  std::uint8_t num_total_refs = 0;       // 0 for key, 1 for P, 2 for B
  std::uint8_t ref_frame_idx[2] = {0, 0};
  bool immediate_output_frame = true;
  bool implicit_output_frame = false;
  bool reference_select = false;         // 1 for a two-reference B frame
};
```

Conversion from a plan entry:

```cpp
const FrameHeaderFields h = header_fields(plan.frames[display]);
header.keyframe = h.keyframe;
header.order_hint_lsbs = h.order_hint_lsbs;
header.refresh_frame_flags = h.refresh_frame_flags;
header.frame_explicit_ref_frame_map = h.frame_explicit_ref_frame_map;
header.num_total_refs = h.num_total_refs;
for (int r = 0; r < h.num_total_refs; ++r) {
  header.ref_frame_idx[r] = h.ref_frame_idx[r];
}
header.immediate_output_frame = h.immediate_output_frame;
header.implicit_output_frame = h.implicit_output_frame;
header.reference_select = h.reference_select;
```

### Exact writer edits in `video_headers.cpp`

| Current line | Current code | Required code |
|---|---|---|
| 110 | `writer.put_bit(true); // monotonic_output_order_flag` | `writer.put_bit(c.monotonic_output_order_flag);` |
| 240 | `writer.put_bit(true); // immediate_output_frame` | `writer.put_bit(h.immediate_output_frame);` |
| 241 | comment only | `if (!h.immediate_output_frame && !c.monotonic_output_order_flag) { writer.put_bit(h.implicit_output_frame); }` (otherwise inferred 0) |
| 249 | `writer.put_bits(h.order_hint, c.order_hint_bits);` | `writer.put_bits(h.order_hint_lsbs, c.order_hint_bits);` |
| 262 | `writer.put_bits(h.refresh_frame_flags, c.num_ref_frames);` | unchanged, but the value must be the plan's flags (key: `allFrames`; non-key: one bit) |
| 264 | `writer.put_bit(true); // frame_explicit_ref_frame_map` | `writer.put_bit(h.frame_explicit_ref_frame_map);` (still 1 for this layer) |
| 265 | `writer.put_bits(c.num_total_refs, 3U);` | `writer.put_bits(h.num_total_refs, 3U);` |
| 266–267 | one `h.ref_slot` with `ceil_log2(num_ref_frames)` bits | loop `for (int r = 0; r < h.num_total_refs; ++r) writer.put_bits(h.ref_frame_idx[r], ceil_log2(c.num_ref_frames));` |
| 295 | `writer.put_bit(false); // reference_select` | `writer.put_bit(h.reference_select);` |

The key-frame branch (`video_headers.cpp:243`) writes no refresh bits; the CLK
inference is `allFrames`, which matches `FramePlan::refresh_frame_flags` for a
key frame. The key-frame OBU branch also has no `reference_select`. Both are
correct.

`num_total_refs` is `f(3)` (0..7); this layer uses 0/1/2. Each
`ref_frame_idx[r]` is `f(CeilLog2(NumRefFrames))`.

---

## 4. Encoding loop: coding order, recon slots, references

Current code:

* key frame encoded from `frames[0]` at `video_frame.cpp:941`;
* P-frame loop at `video_frame.cpp:995`;
* one `reference = result.recon.back()` (`:996`);
* header order hint = input index (`:1009`), one ref slot 0 (`:1010`), refresh
  slot 0 (`:1011`).

Replace that with a coding-order loop. Sketch:

```cpp
std::vector<VideoReconFrame> ref_frames(plan.num_slots);  // §7.23 frame store
std::vector<VideoReconFrame> display_recon(plan.display_count());
std::vector<std::vector<std::uint8_t>> display_units(plan.display_count());
std::vector<std::uint32_t> display_frame_bytes(plan.display_count());
std::vector<std::uint32_t> display_tile_bytes(plan.display_count());
std::vector<std::uint32_t> display_inter(plan.display_count());
std::vector<std::uint32_t> display_intra(plan.display_count());

for (int coding = 0; coding < plan.display_count(); ++coding) {
  const int display = plan.coding_order[coding];
  const FramePlan& f = plan.frames[display];

  if (f.is_key) {
    // Key must be the first frame of its GOP. Encode frames[display], not
    // frames[0], so a leading-picture plan would also work.
    VideoReconFrame empty;
    InterFrameEncoder enc(frames[display], empty, options, options.trace, true);
    ...
  } else {
    // Reference list in plan order. The block layer currently accepts one
    // reference; see §7 for the required extension.
    std::vector<VideoReconFrame> refs;
    for (int r = 0; r < f.num_total_refs; ++r) {
      refs.push_back(ref_frames[f.ref_slot[r]]);
    }
    InterFrameEncoder enc(frames[display], refs, options, options.trace, false);
    ...
  }

  const FrameHeaderFields hf = header_fields(f);
  VideoFrameHeader header;
  header.keyframe = hf.keyframe;
  header.order_hint_lsbs = hf.order_hint_lsbs;
  header.refresh_frame_flags = hf.refresh_frame_flags;
  header.frame_explicit_ref_frame_map = hf.frame_explicit_ref_frame_map;
  header.num_total_refs = hf.num_total_refs;
  for (int r = 0; r < hf.num_total_refs; ++r) header.ref_frame_idx[r] = hf.ref_frame_idx[r];
  header.immediate_output_frame = hf.immediate_output_frame;
  header.implicit_output_frame = hf.implicit_output_frame;
  header.reference_select = hf.reference_select;

  display_units[display] = /* TD + [SEQ only for display 0] + frame OBU */;
  display_recon[display] = to_recon_frame(...);
  display_frame_bytes[display] = ...;
  display_tile_bytes[display] = ...;
  display_inter[display] = enc.inter_blocks();
  display_intra[display] = enc.intra_blocks();

  // §7.23: store this frame in every refreshed slot. This layer always sets
  // exactly one bit for non-key frames and allFrames for a key.
  for (int s = 0; s < plan.num_slots; ++s) {
    if ((f.refresh_frame_flags >> s) & 1U) {
      if (s == 0 || !f.is_key) {
        ref_frames[s] = display_recon[display];  // key: only first slot is valid
      }
    }
  }
  if (f.is_key) {
    // allFrames refresh invalidates slots 1..NumRefFrames-1.
    for (int s = 1; s < plan.num_slots; ++s) ref_frames[s] = VideoReconFrame{};
  }
}
```

Notes:

* `result.units` must be pushed in **coding order** (the loop order), not input
  order. The sequence header OBU is emitted with the first coded unit only.
* `result.recon`, `frame_bytes`, `tile_bytes`, `inter_blocks`, `intra_blocks`
  should be filled by **display index** (`display_recon[display]`, etc.) so the
  gate can compare decoder output frames in display order.
* The key frame is `plan.coding_order[0]`; with `leading_pictures = 0` it is
  display 0, matching today's M2 assumption. With leading pictures it is display
  `L`, so do not index the input as `frames[0]` for the key.
* A key frame stores its recon only in slot 0 and invalidates every other slot
  (§7.23 `RefValid[i] = first`). `adv::refresh_reference_slots` in the plan
  simulation is the reference for this rule.

---

## 5. Slot/refresh rules used by the plan

* Key: `refresh_frame_flags = (1 << num_slots) - 1`, slot 0 valid, all other
  refreshed slots invalid.
* Non-key: `refresh_frame_flags = 1 << slot`, exactly one bit.
* The allocator prefers an empty slot, then the lowest slot whose frame is no
  longer referenced by any later-coded frame. It never overwrites a slot still
  needed by a future reference.
* Every frame is stored in one slot because §7.21 implicit output finds it
  there. A unit normally needs `gop_size + leading` slots; the allocator can
  recycle and the tests cover `num_slots = 2` (low-delay P), `3` (IPBB-4), `4`
  (hierarchical-8), `8` (hierarchical-16). Set `VideoConfig::num_ref_frames` to
  `plan.num_slots` (≤16).
* If the builder returns `kNotEnoughSlots`, increase `num_slots`; do not relax
  the allocator.

---

## 6. Decoder/gate: output order

* `plan.frames[d]` is display index `d`; `plan.coding_order[c]` is display index
  `c` in encode order.
* The plan sets `immediate_output_frame` only for display 0 and
  `implicit_output_frame` for every other frame. With
  `monotonic_output_order_flag = 0`, `dav2d` outputs frames in display order,
  even though the units are written in coding order.
* The gate must therefore collect decoder output frames in output order and
  compare them to `display_recon[0..N-1]`, not to `result.recon.back()` or to the
  coding order.
* `bframes::simulate_decode()` already proves this relative order in the unit
  test; the stream gate repeats it through `dav2d`.

---

## 7. Block-level gap: two references are not yet usable by M2

The header/ref-list layer supports `num_total_refs = 2` and
`reference_select = 1`, but the current M2 block encoder is single-reference:

* `InterFrameEncoder` constructor takes one `const VideoReconFrame&`
  (`video_frame.cpp:250`);
* it stores one `inter::mc::RefFrame ref_frame_` (`video_frame.cpp:859`);
* inter blocks use `inter::mc::kRefLast` / `kRefLastAv2 = 0`
  (`video_frame.cpp:34`, `:436`, `:614`, `:832`);
* `video_syntax.cpp` does not code a per-block reference selection or compound
  mode.

Two safe integration stages:

1. **Header-only / schedule validation stage.** Set `num_total_refs = 1` in the
   plan's header fields for each non-key frame (keep the future reference only
   in the plan, or point ref 0 at the past ref). This produces a valid stream
   and exercises coding order, order hints, slot refresh and output reordering,
   but B frames predict from one reference only. Do not set
   `reference_select = 1` until the block syntax codes it.
2. **Full B stage.** Extend `InterFrameEncoder` to hold
   `inter::mc::RefFrame ref_frames[2]` (or a span) and pass the plan's
   `ref_slot[0..num_total_refs-1]` through `VideoReconFrame` slots. Extend
   `inter_frame_mode_info()` in `video_syntax.cpp` to code the single/compound
   reference selection required by `reference_select = 1`, and extend the MC
   path to predict from both refs. Then set `num_total_refs = 2` and
   `reference_select = 1` for B frames.

`header_fields()` sets `reference_select = (num_total_refs >= 2)`. If stage 1
keeps `num_total_refs = 1`, `reference_select` stays 0 and the existing block
syntax remains consistent.

The plan's `temporal_layer` is encoder scheduling metadata. M2 currently writes
`max_tlayer_id = 0` (`video_headers.cpp:107`) and always uses
`obu_tlayer_id = 0`, so do not signal temporal-layer IDs until the layer
dependency table is modelled. Keep `max_tlayer_id = 0` for the first B
integration.

---

## 8. Leading pictures in M2

`leading_pictures > 0` needs OBU-level support that M2 does not have:

* an open random access point (`OBU_OPEN_LOOP_KEY`), and
* `OBU_LEADING_TILE_GROUP` for the leading frames.

M2 currently writes only `OBU_CLOSED_LOOP_KEY` (key) and
`OBU_REGULAR_TILE_GROUP` (inter) via `make_frame_obu()`
(`video_headers.cpp`, `make_frame_obu`). A CLK resets all reference slots, so
leading frames cannot reference pre-key frames. Keep
`SequenceConfig::leading_pictures = 0` in M2; the schedule and validator are
ready for a later leading-OBU patch.

---

## 9. Checklist

- [ ] Include `encoder/inter/bframes/bframes.h`.
- [ ] Build `Plan` from `SequenceConfig` and call `validate_plan()`.
- [ ] Set `monotonic_output_order_flag = 0`; code `implicit_output_frame`.
- [ ] Set `num_ref_frames = plan.num_slots`; `order_hint_bits` per plan.
- [ ] Extend `VideoFrameHeader` with per-frame `num_total_refs`,
      `ref_frame_idx[]`, output flags, `reference_select`.
- [ ] Write `order_hint_lsbs`, not the absolute hint.
- [ ] Loop in `plan.coding_order`; push units in coding order.
- [ ] Keep a `ref_frames[16]` recon store; refresh according to
      `FramePlan::refresh_frame_flags`; key invalidates all but slot 0.
- [ ] Fill `result.recon`/stat arrays by display index.
- [ ] Gate compares decoder output (display order) to `display_recon`.
- [ ] Keep `leading_pictures = 0` until leading OBUs exist.
- [ ] Keep two-reference block coding off until `video_syntax.cpp`/MC support it.

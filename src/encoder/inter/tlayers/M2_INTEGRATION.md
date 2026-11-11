# M2 integration notes — exact call sites

Status: **the planner and syntax writer are ready; the M2 core is not edited.**
Line numbers are from this snapshot (`src/encoder/m2/m2_headers.cpp`,
`m2_headers.h`, `m2_frame.cpp`); search by symbol if other edits move them.

The layer itself is `src/encoder/inter/tlayers/tlayers.h` +
`tlayers_syntax.h`. The stream is only correct if

1. `build_plan()` and `validate_plan()` pass;
2. the header fields below are written exactly as the plan says;
3. every OBU carries the plan's `obu_tlayer_id`;
4. the decoder gate in `build_and_test.sh` still passes.

Two integration stages:

* **Stage 1 (recommended first):** hierarchical P, one reference per frame,
  `tlayer_dependency_present_flag = 0`, M2 block layer unchanged. This is the
  configuration exercised by `tlayers_gate.cpp` end to end.
* **Stage 2:** two-reference B frames. Requires compound block syntax in
  `m2_syntax.cpp` and two `inter::mc::RefFrame`s in `InterFrameEncoder`; out of
  scope for this component (see §7 of `inter/bframes/M2_INTEGRATION.md`).

---

## 1. Build the plan before encoding

Add near the top of `src/encoder/m2/m2_frame.cpp`:

```cpp
#include "encoder/inter/tlayers/tlayers.h"
#include "encoder/inter/tlayers/tlayers_syntax.h"
```

Before the keyframe encoder is constructed (`m2_frame.cpp:973`):

```cpp
using namespace av2enc::inter::tlayers;

// Stage 1: hierarchical P, 2/4/8-frame GOPs. Input size must be a multiple of
// gop_size; the builder emits complete GOPs only.
const int gop_size = 8;
PlanConfig pc;
pc.gop_size = gop_size;
pc.schedule = Schedule::kHierarchicalP;
pc.dependency = DependencyMode::kAllLower;
pc.explicit_dependency_map = false;   // pinned dav2d parses the default map only
pc.order_hint_bits = (gop_size == 16) ? 5 : 4;
pc.num_slots = gop_size;              // retain every implicit-output frame
pc.base_q_idx = options.base_q_idx;

Plan plan;
PlanError plan_error = PlanError::kOk;
if (!build_plan(pc, plan, &plan_error)) {
  error = std::string("temporal-layer plan: ") +
          plan_error_name(plan_error);
  return false;
}
if (plan.display_count() != static_cast<int>(frames.size())) {
  error = "input frame count is not a whole number of GOP units";
  return false;
}
PlanError validation_error = PlanError::kOk;
int bad_display = -1;
int bad_coding = -1;
if (!validate_plan(plan, &validation_error, &bad_display, &bad_coding)) {
  error = std::string("temporal-layer plan invalid: ") +
          plan_error_name(validation_error);
  return false;
}
```

`build_plan()` self-checks, but `validate_plan()` is the explicit gate and must
stay. Do not construct a sequence header or any frame OBU before it passes.

---

## 2. Sequence header (`M2Config`, `m2_headers.h:21`)

| Field | Current | Required |
|---|---|---|
| `max_tlayer_id` | hardcoded `0` at `m2_headers.cpp:107` | `plan.max_tlayer_id` (`f(2)`, 0..3) |
| `monotonic_output_order_flag` | hardcoded `true` at `:110` | `false` for any reordered stream |
| `tlayer_dependency_present_flag` | absent | `plan.tlayer_dependency_present_flag`; insert the block after the decoder-model flags |
| `tlayer_dependency_map` | absent | `plan.tlayer_dependency[0][curr][ref]` |
| `multi_tlayer_dependency_map_present_flag` | absent | never coded (`max_mlayer_id == 0`) |
| `order_hint_bits` | `4` | `plan.config.order_hint_bits` (5 for a 16-frame P GOP) |
| `num_ref_frames` | `2` | `plan.num_slots` |
| `num_total_refs` | `1` | move to the per-frame header (see §3) |

### Exact writer edits (`m2_headers.cpp`)

```cpp
// :107
writer.put_bits(static_cast<std::uint64_t>(c.max_tlayer_id), 2U);

// :110
writer.put_bit(c.monotonic_output_order_flag);

// After the decoder_model_info_present_flag block (currently :115), before
// sequence_partition_config():
if (c.max_tlayer_id > 0) {
  writer.put_bit(c.tlayer_dependency_present_flag);
  if (c.tlayer_dependency_present_flag) {
    // max_mlayer_id == 0 => multi_tlayer_dependency_map_present_flag = 0,
    // no bit coded.
    for (int curr = 1; curr <= c.max_tlayer_id; ++curr) {
      for (int ref = curr; ref >= 0; --ref) {
        writer.put_bit(c.tlayer_dependency[curr][ref]);
      }
    }
  }
}
```

This is exactly `tlayers::write_sequence_header()` in
`tlayers_syntax.cpp`. With `max_tlayer_id = 0` and
`monotonic_output_order_flag = 1` the OBU is byte-identical to the current M2
header (pinned by `tlayers_test.cpp::test_m2_equivalence`). The recommended
integration is to call `tlayers::make_sequence_header_obu()` directly and avoid
duplicating these fields in M2.

The current values `explicit_ref_frame_map = 1`,
`explicit_num_ref_frames = 1`, `num_ref_frames_minus_1`, and
`enable_short_refresh_frame_flags = 0` are already what the plan needs.

---

## 3. Frame header (`M2FrameHeader`, `m2_headers.h:41`)

Replace the single-reference struct with:

```cpp
struct M2FrameHeader {
  bool keyframe = false;
  std::uint32_t order_hint_lsbs = 0;   // coded value, not the absolute hint
  std::uint32_t refresh_frame_flags = 0;
  bool frame_explicit_ref_frame_map = true;
  std::uint8_t num_total_refs = 0;     // 0 key, 1 P, 2 B
  std::uint8_t ref_frame_idx[2] = {0, 0};
  bool immediate_output_frame = true;
  bool implicit_output_frame = false;
  bool reference_select = false;
};
```

Conversion:

```cpp
const FrameHeader h = frame_header_for(sequence_config, plan, display);
M2FrameHeader header;
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

### Exact writer edits (`m2_headers.cpp`, `write_frame_header_syntax`)

| Current line | Current code | Required |
|---|---|---|
| `:236` | `writer.put_bit(true);` `immediate_output_frame` | `writer.put_bit(h.immediate_output_frame);` |
| `:237` | comment only | `if (!h.immediate_output_frame && !c.monotonic_output_order_flag) writer.put_bit(h.implicit_output_frame);` |
| `:245` | `writer.put_bits(h.order_hint, ...)` | `writer.put_bits(h.order_hint_lsbs, c.order_hint_bits);` |
| `:260` | `writer.put_bit(true);` explicit map | `writer.put_bit(h.frame_explicit_ref_frame_map);` |
| `:261` | `writer.put_bits(c.num_total_refs, 3U)` | `writer.put_bits(h.num_total_refs, 3U);` |
| `:262` | one `h.ref_slot` | loop `for (r = 0; r < h.num_total_refs; ++r) writer.put_bits(h.ref_frame_idx[r], ceil_log2(c.num_ref_frames));` |
| `:291` | `writer.put_bit(false);` `reference_select` | `writer.put_bit(h.reference_select);` |

The key branch already writes no refresh bits (CLK + `max_mlayer_id == 0`
infers `allFrames`), which matches `FrameRefreshFrameFlags` for a key.

---

## 4. OBU header: `obu_tlayer_id` per OBU

`m0::ObuHeader` already has `tlayer_id` (`src/bitstream/m0_headers.h`), and
`m0::write_obu_header()` writes it in the two LSBs. M2 currently constructs a
default `ObuHeader` in `make_sequence_header_obu()` and `make_frame_obu()`, so
every OBU gets 0.

Change `make_frame_obu()` (`m2_headers.cpp:328`) to take the layer id:

```cpp
std::vector<std::uint8_t> make_frame_obu(const M2Config& config,
                                         const M2FrameHeader& header,
                                         std::uint8_t obu_tlayer_id,
                                         std::span<const std::uint8_t> tile_payload) {
  ...
  ObuHeader obu;
  obu.type = header.keyframe ? ObuType::kClosedLoopKey
                             : ObuType::kRegularTileGroup;
  obu.tlayer_id = obu_tlayer_id;   // §5.2.2
  return av2enc::m0::make_obu(obu, payload);
}
```

Rules enforced by `tlayers::make_frame_obu()` / the gate:

* frame OBU: `obu_tlayer_id = plan.obu_tlayer_id(display)`;
* `OBU_TEMPORAL_DELIMITER`: 0 (`§6.2.2`);
* `OBU_SEQUENCE_HEADER`: 0 (`§6.2.2`);
* closed-loop key: 0 (`§6.2.2`; a key cannot be in an enhancement temporal
  layer).

---

## 5. Encoding loop

Current code encodes the key from `frames[0]` (`m2_frame.cpp:973`), then walks
`i = 1..N-1` referencing `result.recon.back()` (`:1026-1027`), writing
`order_hint = i`, `ref_slot = 0`, `refresh = 1` (`:1038-1042`).

Replace with a coding-order loop:

```cpp
std::vector<M2ReconFrame> ref_frames(plan.num_slots);   // §7.23 store
std::vector<M2ReconFrame> display_recon(plan.display_count());
std::vector<std::vector<std::uint8_t>> display_units(plan.display_count());

for (int coding = 0; coding < plan.display_count(); ++coding) {
  const int display = plan.coding_order[coding];
  const PlannedFrame& f = plan.frames[display];

  M2ReconFrame reference;                 // zero for the key
  if (!f.is_key) {
    reference = ref_frames[f.ref_slot[0]]; // Stage 1: one reference
  }
  InterFrameEncoder enc(frames[display], reference, options, options.trace,
                        f.is_key);
  std::vector<std::uint8_t> tile;
  ...
  const FrameHeader hf = frame_header_for(sequence_config, plan, display);
  std::vector<std::uint8_t> obu =
      make_frame_obu(config, header, plan.obu_tlayer_id(display), tile);

  // TD always layers 0; SEQ only in the first coding unit.
  display_units[display] = {td, seq?, obu};
  display_recon[display] = ...;

  // §7.23: store in the plan's refresh slot. Key invalidates all but slot 0.
  if (f.is_key) {
    ref_frames.assign(plan.num_slots, M2ReconFrame{});
    ref_frames[0] = display_recon[display];
  } else {
    ref_frames[f.slot] = display_recon[display];
  }
}
```

`result.units` must be pushed in **coding order**; `result.recon`,
`frame_bytes`, `tile_bytes`, `inter_blocks`, `intra_blocks` should be filled by
**display index** so the gate compares decoder output in display order.

Notes:

* The key is `plan.coding_order[0]` (display 0 for these schedules); index the
  input by `display`, not `frames[0]`.
* A key only stores its recon in slot 0 and invalidates every other slot.
* Use one slot per frame (`num_slots = gop_size`) so `implicit_output_frame`
  can find every frame in the store at flush time. The gate proves this: with a
  single refreshed slot only the last frame survives for output and `dav2d`
  reports 2 of 8 frames.
* M2's tile payload is independent of the header's slot index; only the
  decoder-side reference lookup uses `ref_frame_idx`, so Stage 1 can reuse the
  existing single-reference tile syntax.

---

## 6. Decoder gate and drop test

* Set `monotonic_output_order_flag = 0`; code `implicit_output_frame` on every
  frame except the first. Decoder output order is then display order even though
  units are written in coding order.
* Compare decoder output frames in output (display) order with `display_recon`.
* Drop test: rebuild the IVF with only units whose `obu_tlayer_id <= L` for
  `L = 0, 1`. `plan.coding_order_up_to_layer(L)` gives the retained coding order
  and `validate_layer_drop(plan, L)` proves no retained frame references a
  dropped frame. `tlayers_gate.cpp` does this through `dav2d --strict 1
  --threads 1` and requires 0 diffs for both sub-streams.

For an M2 CLI experiment, add a `--max-tlayer <N>` that filters coded units by
`f.tlayer_id` after planning; do not renumber order hints or layer ids.

---

## 7. Checklist

- [ ] Include `tlayers.h` / `tlayers_syntax.h` in `m2_frame.cpp`.
- [ ] Build a `Plan` and call `validate_plan()`; stop on failure.
- [ ] Sequence header: `max_tlayer_id`, `monotonic_output_order_flag = 0`,
      dependency flag/map (or call `tlayers::make_sequence_header_obu()`).
- [ ] `num_ref_frames = plan.num_slots`; `order_hint_bits` from the plan.
- [ ] Extend `M2FrameHeader`; write `order_hint_lsbs`, per-frame refresh,
      explicit ref indices, output flags, `reference_select`.
- [ ] Pass `plan.obu_tlayer_id(display)` into `make_frame_obu()`.
- [ ] Loop in `plan.coding_order`; push units in coding order.
- [ ] Keep a `M2ReconFrame ref_frames[16]` store; refresh `f.slot`; key resets.
- [ ] Fill result arrays by display index.
- [ ] Gate: full stream 0 diffs; layers `<= 0` and `<= 1` 0 diffs.
- [ ] Keep `explicit_dependency_map = false` and
      `DependencyMode::kAllLower` until the pinned decoder parses the v1.0.0
      explicit map.
- [ ] Keep `Schedule::kHierarchicalP` until compound block syntax exists.

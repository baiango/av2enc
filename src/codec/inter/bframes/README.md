# B-frame hierarchy / reference layer

Encoder-side scheduling, reference lists, slot refresh rules, order hints and
decoder-visible output-order simulation for AV2 v1.0.0 B-frames.

**Component only.** This directory writes no bits, edits no M2/M1 core files and
does not touch `src/main.cpp` or the root `CMakeLists.txt`. It reuses the
decoder-exact bookkeeping in `src/encoder/inter/advanced/b_frame.*`; the M2
integration is documented in [`VIDEO_INTEGRATION.md`](VIDEO_INTEGRATION.md).

## What it adds over `advanced/b_frame.*`

`advanced/b_frame.*` already provides the §7.7 ranking, §7.8
past/future/current classification, §7.23 slot refresh, §5.18 order-hint lift
and a dyadic GOP builder. This layer wraps those primitives with:

* absolute display order hints plus the exact `order_hint` LSB value to signal;
* explicit `frame_explicit_ref_frame_map = 1` reference maps (up to two refs);
* deterministic coding order and deterministic slot allocation/recycling;
* key-frame reset and leading-picture schedules;
* key-frame + P / IPBB / hierarchical B pattern selection;
* validation that references are in range, already decoded, distinct, held by
  the named slot, not stale after a key-frame reset, and inside the leading /
  regular reference rules;
* a §7.21.4/.5/.6 output simulation that checks the decoder-visible output
  order against the intended display order.

The §7.7 ranking code is **not** duplicated: `simulate_decode()` calls
`adv::build_ref_frame_list()` on the live decoder state for every frame and
exposes the result in `DecodeFrameTrace::auto_*`. The plan's explicit map is
allowed to differ, but for the simple cases the tests assert the automatic
ranking agrees with the explicit nearest reference.

## Build and test

```sh
./src/encoder/inter/bframes/build_and_test.sh
WERROR=1 ./src/encoder/inter/bframes/build_and_test.sh
SANITIZE=1 ./src/encoder/inter/bframes/build_and_test.sh
```

Expected:

```text
bframes_test: 11893 checks passed
```

The build script compiles `src/encoder/inter/advanced/b_frame.cpp` and this
directory's `bframes.cpp` with `-std=c++20 -Wall -Wextra -Wpedantic -Wconversion
-Wshadow`. No CMake change is needed.

## API

Header: `src/encoder/inter/bframes/bframes.h`

| Symbol | Purpose |
|---|---|
| `GopConfig` | one closed GOP unit: `gop_size`, `pattern`, `start_order_hint`, `order_hint_bits`, `num_slots`, `leading_pictures` |
| `SequenceConfig` | `num_gops` units; later units may carry leading pictures |
| `GopPattern` | `kLowDelayP`, `kIpbb`, `kHierarchicalB` |
| `FrameKind` | `kKey`, `kP`, `kB` (encoder labels; AV2 itself is intra/inter) |
| `FramePlan` | display index/hint, coding index, temporal layer, kind, refs, slots, refresh flags, output flags, `order_hint_lsbs` |
| `Plan` | frames in display order + `coding_order` (display indices in encode order) |
| `FrameHeaderFields` | exact fields the M2 frame-header writer needs |
| `header_fields(const FramePlan&)` | converts a plan entry to those fields |
| `coding_frame_types(plan)` | frame kinds in decoder/encode order |
| `build_gop(config, plan, &error)` | builds and self-validates one GOP unit |
| `build_sequence(config, plan, &error)` | builds and self-validates `num_gops` units |
| `validate_plan(plan, &report)` | structural checks + full decoder simulation |
| `simulate_decode(plan, &report)` | walks coding order, applies §7.23 and the output processes |
| `plan_error_name`, `gop_pattern_name`, `frame_kind_name` | diagnostics |

`build_gop()` and `build_sequence()` return `false` on invalid input or when the
resulting plan would not pass `validate_plan()`. The integration must still call
`validate_plan()` (it is the explicit gate) and must not write bits on failure.

## Schedule patterns

All patterns start with a key frame and end with a P anchor at the last display
index of the unit.

### `kLowDelayP`

```text
display:  0   1   2   3
coding:   0   1   2   3
type:     I   P   P   P
refs:     -   0   1   2
```

Display order equals coding order. This is the only pattern that works with
`num_slots = 2`; it is also the compatibility pattern for the current M2
P-frame path.

### `kIpbb`

One P anchor plus flat B frames. Coding frame types are `I P B B ...`.

```text
gop_size = 4
display:  0   1   2   3
coding:   0   3   1   2
type:     I   B   B   P
refs:     -  0,3 0,3 0
```

```text
gop_size = 8
coding:   0   7   1   2   3   4   5   6
type:     I   P   B   B   B   B   B   B
refs:     -   0  0,7 0,7 0,7 0,7 0,7 0,7
```

### `kHierarchicalB`

Dyadic pyramid, breadth-first left-to-right coding order. A B frame brackets
its display position with the nearest already-coded past and future frames.
Temporal layers increase by one per split; a child never references a deeper
layer.

```text
gop_size = 4
display:  0   1   2   3
coding:   0   3   1   2
type:     I   B   B   P
layer:    0   1   2   0
refs:     -  0,3 1,3 0
```

```text
gop_size = 8
display:  0   1   2   3   4   5   6   7
coding:   0   7   3   1   5   2   4   6
type:     I   B   B   B   B   B   B   P
layer:    0   2   3   1   3   2   3   0
refs:     -  0,3 1,3 0,7 3,5 3,7 5,7 0
```

```text
gop_size = 16
display:  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
coding:   0  15   7   3  11   1   5   9  13   2   4   6   8  10  12  14
layer:    0   3   4   2   4   3   4   1   4   3   4   2   4   3   4   0
```

`gop_size` must be a power of two for `kHierarchicalB`; all patterns accept
2..16. The same schedule works for a truncated unit (e.g. an interval of 6
frames) because the splitter is not restricted to powers of two, but the public
builder only exposes complete units.

## Leading pictures

`GopConfig::leading_pictures = L` attaches `L` frames displayed before the key
frame but decoded after it:

```text
gop_size = 4, leading = 2

display:  0   1   2   3   4   5
coding:   2   1   0   4   5   3
type:     B   B   I   B   B   P
refs:    1,2  2  -  2,5 3,5  2
         ^ leading pictures ^   ^ regular GOP ^
```

The key frame is at display `L` and is decoded first. Leading pictures are
decoded latest-display first (`L-1, L-2, ..., 0`). A key-frame reset invalidates
every reference slot except the key's first refreshed slot, so a leading picture
has **no valid past reference**: it may only reference the key frame or another
already-decoded leading picture (both are future references). The validator
rejects a leading picture that references a regular frame, and rejects a regular
frame that references a leading picture.

At the OBU level, a real AV2 stream needs `OBU_OPEN_LOOP_KEY` /
`OBU_LEADING_TILE_GROUP` for these frames. The M2 path currently writes only
`OBU_CLOSED_LOOP_KEY` and `OBU_REGULAR_TILE_GROUP`, so keep
`leading_pictures = 0` in M2 until leading OBU support exists. This layer
provides and validates the schedule so that work can land later without changing
the reference model.

## Order hints and `ref_order_hint`

* `FramePlan::display_order_hint` is the absolute hint:
  `start_order_hint + display_index`.
* `FramePlan::order_hint_lsbs` is the value to write with `f(OrderHintBits)`.
  It is derived by `adv::order_hint_lsbs_for()` from the live decoder slot
  state, and `adv::display_order_hint()` must reconstruct the absolute hint.
* `FramePlan::ref_order_hint[]` is the saved hint of the referenced slot
  (`RefOrderHint[slot]`), i.e. `frames[ref_display].display_order_hint`.
  It is **not** a frame-header syntax element and is not written.
* `OrderHintBits` must be large enough for the §5.18 lift. A 16-frame unit with
  a future anchor decoded before its B frames needs 5 bits; 4 bits fails with
  `kOrderHintReconstruct`. Four bits is sufficient for 4/8-frame units and for
  up to 7 leading pictures (the earliest leading hint must be less than half a
  span behind the key hint).
* A stream starting at a non-zero hint needs enough bits: `start_order_hint =
  40` works with 8 bits, not with 4 (the first key frame is reconstructed
  relative to a zero-initialised store).

## Slot refresh rules

* Key frame: `refresh_frame_flags = allFrames = (1 << NumRefFrames) - 1`.
  §7.23 stores it in the first refreshed slot (slot 0) and **invalidates** every
  other refreshed slot, so pre-key frames cannot be referenced.
* Non-key frame: exactly one slot, `refresh_frame_flags = 1 << slot`.
  Every frame is stored, because §7.21 implicit output finds the frame in a
  reference slot.
* Allocation is deterministic: prefer the lowest empty slot; otherwise the
  lowest slot whose frame is no longer referenced by any later-coded frame.
  A slot is only recycled after its last consumer has been coded.
* Slot count: a closed unit with `gop_size + leading` frames can usually use
  that many slots, but the allocator recycles safely. Tests cover
  `num_slots = 2` for low-delay P, `3` for IPBB-4, `4` for hierarchical-8 and
  `8` for hierarchical-16. If no safe slot exists the builder returns
  `kNotEnoughSlots`.

## Output reordering / decoder-visible order

The plan sets:

* `immediate_output_frame = 1` only for display index 0;
* `implicit_output_frame = 1` for every other frame.

The simulator transcribes the relative-order parts of:

* §7.21.3 output successive frames;
* §7.21.4 output implicit output frame;
* §7.21.5 flush implicit output frames;
* §7.21.6 output frame buffers, including outputting an eligible frame before
  its slot is overwritten by §7.23.

It then requires `output_order == 0,1,...,N-1`. A malformed plan that outputs a
future anchor too early fails with `kOutputOrderMismatch`.

**Important:** `implicit_output_frame` is only coded when
`monotonic_output_order_flag == 0`. The current M2 sequence header sets
`monotonic_output_order_flag = 1`, which forces `implicit_output_frame = 0`
(i.e. the frame is not output at all unless `immediate_output_frame = 1`). The
M2 integration must set the flag to 0 for reordered B-frame streams.

## Validation / error codes

`validate_plan()` reports:

| Error | Meaning |
|---|---|
| `kBadGopSize` / `kBadPattern` / `kBadSlotCount` / `kBadOrderHintBits` / `kBadLeadingCount` | bad configuration |
| `kNotEnoughSlots` | no safe reference slot for a frame |
| `kBadDisplayOrder` / `kBadCodingOrder` | non-monotonic hints, wrong display indices, or `coding_order` not a permutation |
| `kMissingKeyFrame` / `kKeyFrameHasRefs` | GOP does not start with a key, or a key carries refs |
| `kTooManyRefs` / `kMissingReference` | >2 refs, or a non-key frame with no ref |
| `kReferenceOutOfRange` / `kReferenceNotDecoded` | reference outside `[0,N)` or later in coding order |
| `kReferenceSlotInvalid` | slot invalid or holds a different frame |
| `kReferenceHintMismatch` | saved ref hint disagrees with the referenced frame |
| `kKeyFrameRefInvalidated` | reference crosses a key-frame reset / GOP |
| `kLeadingPictureRule` | leading/regular reference direction violated |
| `kDuplicateReference` | same display or slot referenced twice |
| `kBadReferenceSplit` | B/P reference is not one past + one future (leading pictures: all future) |
| `kRefreshFlagsInvalid` | key isn't `allFrames`, non-key isn't exactly one slot |
| `kOutputFlagsInvalid` | both or neither output flag set |
| `kOrderHintReconstruct` | signalled LSBs do not reconstruct the absolute hint |
| `kOutputOrderMismatch` / `kOutputMissingFrame` | decoder-visible order wrong / frame never output |
| `kInternal` | builder invariant broken |

## Tests

`bframes_test.cpp` runs 11,893 checks:

* All patterns × GOP sizes 4/8/16: build, validate, decode-simulate, exact
  display/hint/coding order, one key per unit, P anchor, temporal layers,
  one-past/one-future B refs, distinct slots, refresh flags, order-hint LSBs,
  deterministic second build, decoder output order.
* Exact schedules for IPBB-4/8 and hierarchical-4/8/16.
* Low-delay P with two slots and slot recycling; §7.7 automatic ranking agrees
  with the explicit nearest past reference.
* Leading pictures `L = 1,2,3` for 4/8-frame units: key reset, all-future refs,
  reverse display coding, output order, and two leading/regular rule errors.
* Three-GOP sequences with and without leading pictures: key positions,
  no cross-GOP references, output order, determinism.
* Long sequences (128 low-delay P frames, 64 hierarchical B frames): slot
  recycling and iterative output ordering.
* Configuration errors (size/pattern/bits/slots/leading/start hint), 16-frame
  `OrderHintBits = 4` failure, slot exhaustion, non-zero start hint.
* Mutation errors for every validation class: out-of-range, self, forward,
  stale slot, duplicate display/slot, hint mismatch, bad refresh flags, key
  refs, missing/too many refs, bad split, bad coding/display order, bad output
  flags, bad LSBs, early output, missing key, cross-key reference, leading rule.
* ASan/UBSan clean (`SANITIZE=1`).

## Limitations / traps

1. At most **two** explicit references per frame. AV2 allows seven; the B/P
   schedules here need two and the validator rejects a third.
2. One embedded layer (`max_mlayer_id = 0`), no multi-layer output ordering.
3. No `show_existing_frame`, switch frames, bridge frames, long-term references,
   hidden/reference-only frames, or cross-GOP references.
4. Complete GOP units only. `SequenceConfig::num_gops` is not a frame count; the
   last unit is not truncated by the builder. M2 must choose a `gop_size` that
   divides its input, or extend the builder for truncated units.
5. Every frame is shown and stored. `immediate_output_frame ==
   implicit_output_frame` is rejected rather than treating the frame as hidden.
6. Leading pictures are a schedule/model feature; OBU-level
   `OBU_LEADING_TILE_GROUP` / `OBU_OPEN_LOOP_KEY` support is not implemented in
   M2.
7. The output simulation models relative order, not decoder timing. It assumes
   `max_mlayer_id = 0`; the `output_ordering()` layer term is therefore the
   order hint.
8. `advanced/b_frame.*` is owned by the advanced inter component. This
   directory only includes it; it never edits it.

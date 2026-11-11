# Temporal layers (`tlayers`)

Encoder-side temporal-layer planning and exact AV2 v1.0.0 syntax for a
hierarchical GOP: layer count, per-frame layer assignment, `TLayerDependencyMap`,
per-OBU `obu_tlayer_id`, order-hint assignment, and encoder-side reference
restriction per layer.

Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
(2026-05-28, commit `38f28e7`), §5.2.2, §5.4.1, §5.18.2, §7.3.7, §7.7, §7.23.

**Component only.** Nothing outside `src/encoder/inter/tlayers/` is edited; the
M2 integration is documented in [`M2_INTEGRATION.md`](M2_INTEGRATION.md).
Tile syntax, entropy coding and RDO are not touched.

## Why this exists

The M2 sequence header currently writes `max_tlayer_id = 0`
(`m2_headers.cpp:107`), so every OBU carries `obu_tlayer_id = 0` and the stream
has one temporal layer. This component supplies:

* a validated layer plan (coding order, layer ids, references, slots, output
  flags, order hints);
* the §5.4.1 header fields that make layers signalable;
* the §5.2.2 `obu_tlayer_id` assignment for every OBU in a temporal unit;
* a gate that proves the resulting stream decodes with the pinned `dav2d`,
  including dropping upper layers.

## Spec facts pinned by this layer

| Fact | Value / consequence |
|---|---|
| `obu_tlayer_id` | `f(2)`, present in every OBU header byte (`header = ext<<7 | type<<2 | tlayer`) |
| `max_tlayer_id` | `f(2)` => **at most four layers (ids 0..3)** |
| Default map | `TLayerDependencyMap[m][c][r] = r <= c && c <= max_tlayer_id && m <= max_mlayer_id` |
| Explicit map | `tlayer_dependency_present_flag f(1)`, then for `mLayer = 0..max_mlayer_id`, `curr = 1..max_tlayer_id`, `ref = curr..0`: one `tlayer_dependency_map f(1)` |
| `multi_tlayer_dependency_map_present_flag` | only coded when `max_mlayer_id > 0`; inferred 0 here |
| Reference legality | `RefTLayerId[ref]` must satisfy `TLayerDependencyMap[0][obu_tlayer_id][RefTLayerId] == 1`; the encoder must never emit a reference from layer C to layer R > C |
| Forced tlayer 0 | `OBU_SEQUENCE_HEADER`, `OBU_TEMPORAL_DELIMITER`, and key/switch/RAS frame OBUs (`§6.2.2`) |
| Order hint | `order_hint f(OrderHintBits)`, lifted by `get_disp_order_hint()` using the maximum hint of showable, dependency-mapped reference slots |
| Output | reordered streams need `monotonic_output_order_flag = 0`; frames other than the first use `implicit_output_frame = 1` |

Consequences:

* A 16-frame hierarchical **B** pyramid needs temporal layer 4 and cannot be
  signalled. `kHierarchicalB` accepts GOP sizes 2..8.
* A 16-frame hierarchical **P** pyramid uses layers 0..3 (OK) but needs
  `OrderHintBits = 5`; 4 bits fails the §5.18 lift.
* `max_mlayer_id = 0` throughout, so there is one embedded layer and no
  multi-layer dependency table.

## API

Header: `src/encoder/inter/tlayers/tlayers.h`

| Symbol | Purpose |
|---|---|
| `PlanConfig` | `gop_size`, `schedule`, `dependency`, `explicit_dependency_map`, `start_order_hint`, `order_hint_bits`, `num_slots`, `base_q_idx` |
| `Schedule` | `kHierarchicalP` (single-reference carrier, M2-encodable), `kHierarchicalB` (two references) |
| `DependencyMode` | `kDirect` (code only direct reference edges), `kAllLower` (transitive closure = every lower layer) |
| `PlannedFrame` | display/coding index, `tlayer_id`, absolute and signalled order hint, refs (display, slot, saved hint), refresh slot/flags, output flags, `obu_tlayer_id()` |
| `Plan` | display-ordered frames, `coding_order`, `max_tlayer_id`, `layer_frame_count`, `tlayer_dependency[4][4]`, `tlayer_dependency_present_flag`, `coding_order_up_to_layer()` |
| `build_plan(config, plan, &error)` | builds and self-validates one closed GOP |
| `validate_plan(plan, &error, &display, &coding)` | structural checks + full §7.23 / §5.18 replay |
| `validate_layer_drop(plan, layer, &error)` | proves a retained prefix is independently decodable |
| `plan_error_name`, `schedule_name`, `dependency_mode_name` | diagnostics |

Syntax header: `src/encoder/inter/tlayers/tlayers_syntax.h`

| Symbol | Purpose |
|---|---|
| `SequenceConfig` | §5.4.1 fields, including `max_tlayer_id`, `monotonic_output_order_flag`, dependency flag/map; `apply_plan(plan)` |
| `make_sequence_header_obu(config)` | Annex B.2 OBU; byte-identical to M2 for `max_tlayer_id = 0` |
| `FrameHeader` | `order_hint_lsbs`, refresh flags, explicit ref map, ref indices, output flags, `reference_select` |
| `frame_header_for(config, plan, display)` | converts a plan entry |
| `make_frame_obu(config, header, obu_tlayer_id, tile)` | §5.19 tile-group header + payload, wrapped with the right `obu_tlayer_id` |
| `make_temporal_delimiter_obu()` | TD with `obu_tlayer_id = 0` |
| `parse_obu`, `parse_sequence_header_obu` | reader used by goldens and the gate |
| `obu_header_byte(ext, type, tlayer)` | §5.2.2 header byte |

## Schedules

### `kHierarchicalP` (gate carrier)

Every frame references the immediately preceding frame in coding order. This is
a dyadic reordered P pyramid, so the current single-reference M2 block layer can
encode it. `gop_size = 8`:

```text
display:  0   1   2   3   4   5   6   7
coding:   0   4   2   6   1   3   5   7
layer:    0   2   1   2   0   2   1   2
refs:     -   6   4   1   0   3   2   5      (display indices)
hints:    0   1   2   3   4   5   6   7
```

`gop_size = 4`: coding `0 2 1 3`, layers `0 1 1 0`. `gop_size = 16`: coding
`0 8 4 12 2 6 10 14 1 3 5 7 9 11 13 15`, layers 0..3, `OrderHintBits = 5`.

### `kHierarchicalB`

Dyadic B pyramid, two references per interior frame. Encoder-side planning and
validation only; the M2 block layer is single-reference. `gop_size = 8`:

```text
display:  0   1   2   3   4   5   6   7
coding:   0   7   3   1   5   2   4   6
layer:    0   2   3   1   3   2   3   0
refs:     -  0,3 1,3 0,7 3,5 3,7 5,7 0
hints:    0   1   2   3   4   5   6   7
```

Every reference satisfies `layer(ref) <= layer(frame)`; the validator rejects a
plan otherwise. "Forward reference" in this component means a reference to a
**higher temporal layer**, not a later display index: the P schedule
intentionally references future-display frames that were coded earlier (for
example display 2 references display 4), and those are legal because layer 0
was decoded first.

## Dependency maps

`kAllLower` sets every `r <= c` entry for `c <= max_tlayer_id`. `kDirect` codes
only the actual reference edges (plus the implied `[0][0] = 1`), then
`validate_plan()` checks that every frame's references are covered.

`PlanConfig::explicit_dependency_map = false` (default) writes
`tlayer_dependency_present_flag = 0`; the decoder then uses the §5.4.1 default
map, which is exactly `kAllLower`. `explicit_dependency_map = true` codes the
map bits and requires `Plan` to be built with the matching mode.

### Pinned `dav2d` caveat (f9dafbc)

The pinned decoder nightly parses the dependency table with a **different syntax**
than the v1.0.0 PDF:

```c
/* dav2d f9dafbc src/obu.c */
if (max_tlayer_id) {
  tlayer_dependency_present = get_bit();
  if (tlayer_dependency_present)
    for (n = 1; n < max_tlayer_id; n++) deps[n] = get_bits(n);
}
if (max_mlayer_id) { ... }
```

The PDF codes `curr = 1..max_tlayer_id`, `ref = curr..0` (including self) and
orders the mlayer block before the tlayer block. With `max_mlayer_id = 0` the
block *positions* agree, so a stream with `tlayer_dependency_present_flag = 0`
is parsed identically by both: both apply the default `ref <= curr` map. The
decoder gate therefore uses the default map. The spec-exact explicit-map writer
and reader are unit-tested and pinned; use them only once the decoder matches
v1.0.0. This is also why M2 integration keeps
`explicit_dependency_map = false`.

## Order hints and output

* Absolute hint = `start_order_hint + display_index`.
* `order_hint_lsbs` is derived by replaying `get_disp_order_hint()` against the
  live slot store; `validate_plan()` rejects any LSBs that do not reconstruct.
* First coded frame: `immediate_output_frame = 1`.
  Every later frame: `immediate_output_frame = 0`,
  `implicit_output_frame = 1` (requires `monotonic_output_order_flag = 0`).
* Implicit output finds frames in the reference store, so a gate/integration
  stream must retain them: use `num_slots >= display_count()` unless an
  output-timing model is added. One slot is legal for a P *reference* chain, but
  then only the last frame remains for §7.21 output.

## Slot allocation and refresh

* Key: `refresh_frame_flags = (1 << num_slots) - 1`; §7.23 keeps slot 0 valid
  and invalidates the rest.
* Non-key: exactly one bit, `1 << slot`.
* Allocation prefers an empty slot, then the lowest slot whose frame is no
  longer referenced by any later-coded frame. A frame may reference and refresh
  the same slot.

## Build and test

```sh
./src/encoder/inter/tlayers/build_and_test.sh
./src/encoder/inter/tlayers/build_and_test.sh --no-gate
WERROR=1 ./src/encoder/inter/tlayers/build_and_test.sh --no-gate
SANITIZE=1 ./src/encoder/inter/tlayers/build_and_test.sh --no-gate
```

Expected:

```text
tlayers_test: 1787 checks passed
tlayers gate: 64x64, 8 frame(s)
  recon: 8 frame(s), 0 diffs
...
TLayers gate PASS
tlayers gate: 640x360, 8 frame(s)
...
TLayers gate PASS
```

No CMake change is required. The unit test links `m2_headers.cpp` only to prove
byte equivalence for the `max_tlayer_id = 0` baseline.

## Decoder gate

`tlayers_gate.cpp`:

1. builds an 8-frame hierarchical-P plan (`max_tlayer_id = 2`);
2. synthesises a moving-square clip, reorders the frames into `coding_order`,
   and encodes them with `m2::encode_m2_sequence()` as a single-reference P
   chain;
3. extracts each M2 tile payload, re-wraps it with the plan's sequence header
   (`max_tlayer_id`, `monotonic_output_order_flag = 0`), frame header
   (`order_hint_lsbs`, refresh/ref map, implicit output) and per-OBU
   `obu_tlayer_id`;
4. audits every OBU: TD/SEQ/key are layer 0, frame OBUs match the plan, and the
   parsed sequence header reproduces the plan's map;
5. decodes with `dav2d --strict 1 --threads 1` and requires **0 recon diffs**;
6. rebuilds the stream with only layers `<= 0` and `<= 1` and requires both
   sub-streams to decode with 0 diffs (2 and 4 frames).

The gate runs at 64x64 and 640x360. `--no-drop` skips step 6; `--out PREFIX`
keeps the IVFs.

Goldens exercised by the unit test:

| Object | Pinned value |
|---|---|
| TD OBU | `0108` |
| OBU header bytes, type 7, tlayer 0..3 | `1c 1d 1e 1f` |
| OBU header bytes, types SEQ/CLK/TD, tlayer 0 | `04 / 10 / 08` |
| 64x64, `max_tlayer_id = 0`, NumRefFrames 2, monotonic 1 | `13048075002abffe0028006e23800000101b8082` |
| 64x64, P8, default map, NumRefFrames 8 | `13048075040abffe0014003771c00000080dc041` |
| 64x64, P8, explicit all-lower map | `14048075040abffe3f00a001bb8e000000406e0208` |
| 640x360, P8, explicit direct map | `15048075041313fd9c7c014003771c00000080dc0410` |
| 64x64, B8, explicit direct map, `max_tlayer_id = 3` | `15048075060abffe2b700a001bb8e000000406e02080` |

The `max_tlayer_id = 0` sequence header and the key/P frame headers are also
compared byte-for-byte against `m2::make_sequence_header_obu()` /
`m2::make_frame_obu()`.

## Limitations

1. One GOP. `PlanConfig::gop_size` frames, one key, no cross-GOP references.
2. At most two references per frame; the B schedule is plan-only until the M2
   block layer supports compound references.
3. `max_mlayer_id = 0`; no embedded/multi-layer dependency table.
4. No leading pictures, switch/bridge/SEF frames, or long-term references.
5. Implicit-output retention is not modelled; use `num_slots >= display_count()`.
6. Explicit dependency maps are spec-exact but the pinned decoder nightly parses
   a later syntax; see the caveat above.
7. `validate_plan()` checks relative bitstream validity, not decoder timing.

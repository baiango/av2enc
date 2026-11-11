# Long-term references, open-loop keys and random access

Spec-only implementation of the AV2 v1.0.0 long-term-reference layer:
sequence-level `long_term_frame_id_bits`, frame-level `ref_long_term_id` /
`num_key_ref_frames`, `OBU_OPEN_LOOP_KEY` frame encoding, the reference-pool
invalidation rules and the closed/open random-access conformance rules.

Normative reference: `spec/AV2_Spec_v1.0.0.pdf` (2026-05-28, commit `38f28e7`).

| Section | What this module takes from it |
|---|---|
| §5.4.1/§5.4.6 | `sequence_header_obu()` tail: `long_term_frame_id_bits f(3)`, `num_ref_frames_minus_1`, `enable_short_refresh_frame_flags` |
| §5.18.2 | `frame_header_info()`: `long_term_id_plus_1`, `num_key_ref_frames`, `ref_long_term_id[i]`, `immediate_output_frame`, `refresh_frame_flags`, `num_total_refs`, `ref_frame_idx[i]`, plus the first-Regular-frame invalidation loop |
| §6.4.6/§6.17 | field semantics and the bitstream-conformance requirements on those fields |
| §7.3.9 | availability of long-term reference frames (same buffer slot under sequential and random-access decoding) |
| §7.4.2 | random access with (`long_term_frame_id_bits != 0`) and without (`== 0`) long-term references |
| §7.4.3 | closed random access: a CLK invalidates every reference buffer |
| §7.4.4 | open random access: only `OlkRefresh` and the named long-term references survive |
| §7.23 | `RefValid` / `RefLongTermId` / `RefOrderHint` update, including the `first` rule for KEY_FRAME |

Nothing here is copied from an encoder implementation; the bit layout is read
from the spec syntax tables and the reader is an independent parser.

## Files

```text
src/encoder/inter/longterm/
  longterm.h            public API (config, frame header, pool, planners)
  longterm_headers.cpp  §5.4.6 / §5.18.2 writer and reader, OBU assembly
  longterm_pool.cpp     §5.18.2 + §7.23 state machine, validators, planners
  longterm_stream.h/.cpp temporal-unit assembly (M2 tile payload splice)
  longterm_test.cpp     327 unit checks (goldens, round-trips, pool, rules)
  longterm_gate.cpp     3-phase dav2d decoder gate
  build_and_test.sh     standalone build (no CMake changes)
  README.md             this file
```

## API

### Sequence and frame headers

```cpp
struct SequenceConfig {
  std::uint16_t max_frame_width, max_frame_height;
  std::uint8_t  seq_level_idx, base_q_idx;
  std::uint8_t  order_hint_bits;            // OrderHintBits (1..16)
  std::uint8_t  num_ref_frames;             // NumRefFrames (1..16)
  std::uint8_t  long_term_frame_id_bits;    // f(3); 0 disables LTRs
  std::uint8_t  max_drl_bits_minus_1;
  bool disable_cdf_update, enable_cfl_intra;
  LongTermListSyntax olk_long_term_list;    // see "pinned dav2d" below
};

enum class FrameRole : std::uint8_t {       // values are §6.2.2 obu_type
  kClosedLoopKey = 4, kOpenLoopKey = 5, kLeading = 6, kRegular = 7,
};

struct FrameHeader {
  FrameRole role;
  bool frame_is_inter;                      // INTER_FRAME / INTRA_ONLY_FRAME
  std::uint32_t order_hint;
  std::int32_t  long_term_id;               // -1 = not a long-term reference
  std::uint32_t refresh_frame_flags;
  std::uint8_t  num_total_refs;
  std::vector<std::uint8_t> ref_frame_idx;
  std::vector<std::uint8_t> ref_long_term_id;  // OLK only
  bool immediate_output_frame;              // inferred 0 for an OLK
};

std::size_t write_sequence_header(BitWriter&, const SequenceConfig&);
std::vector<std::uint8_t> make_sequence_header_obu(const SequenceConfig&);
std::size_t write_frame_header(BitWriter&, const SequenceConfig&, const FrameHeader&);
std::size_t write_tile_group_header(BitWriter&, const SequenceConfig&, const FrameHeader&);
std::vector<std::uint8_t> make_frame_obu(const SequenceConfig&, const FrameHeader&,
                                         std::span<const std::uint8_t> tile);
std::vector<std::uint8_t> make_temporal_delimiter_obu();
```

Reader side:

```cpp
bool parse_obu(std::span<const std::uint8_t>, ObuView*, std::size_t* consumed, std::string*);
bool read_tile_group_header(std::span<const std::uint8_t> payload,
                            const SequenceConfig&, ParsedFrameHeader*, std::string*);
bool read_sequence_header(std::span<const std::uint8_t> payload,
                          const SequenceConfig&, std::size_t* bits, std::string*);
```

`ParsedFrameHeader` returns the decoded header, the exact header bit count and
the byte offset of the tile payload, so a caller can verify that the payload
behind the header is untouched. The reader needs the `FrameRole` seeded on
`out->header.role` because the OBU type is not repeated inside the tile-group
payload.

### Reference pool

```cpp
struct PoolState {                       // the normative decoder variables
  std::array<SlotState, 16> slots;       // RefValid, RefLongTermId,
                                         // RefOrderHint, RefFrameType
  bool olk_encountered;                  // OlkEncountered
  std::uint32_t olk_refresh;             // OlkRefresh[obu_mlayer_id]
  std::uint32_t olk_order_hint;          // OlkTUOrderHint
  std::int32_t pending_ltr_ids[7];       // ids the OLK promised to keep
  int pending_ltr_count;
  ...
};

void reset_pool(const SequenceConfig&, PoolState*);
bool apply_frame(const SequenceConfig&, const FrameHeader&, bool first_picture_in_tu,
                 PoolState*, PoolEvents*, std::string* error);
```

`apply_frame()` runs the normative order exactly:

1. `startCVS` (CLK + FirstPictureInTU) invalidates every slot.
2. The first-Regular-frame-after-OLK invalidation: a slot is dropped when it is
   not in `OlkRefresh` **and** its `RefLongTermId == -1`. Every long-term id the
   OLK promised is checked to still be held here.
3. Leading-frame rules (`refresh & OlkRefresh == 0`, no overwriting a promised
   long-term slot).
4. `ref_frame_idx[i]` must be `RefValid`.
5. §7.23 update: for each refreshed slot
   `RefValid = (KEY_FRAME || SWITCH_FRAME) ? first : 1`, `RefLongTermId =
   LongTermId`, `RefOrderHint = OrderHint`, then `first = 0`. For a KEY_FRAME
   that means **only the lowest refreshed slot becomes valid**.
6. `OlkEncountered`/`OlkRefresh` bookkeeping.

The pool is copied before each frame, so a rejected header leaves it
unchanged (the tests rely on this).

### Validation

```cpp
bool validate_frame_header(const SequenceConfig&, const FrameHeader&, std::string*);
bool validate_temporal_unit(const SequenceConfig&, std::span<const FrameHeader>, std::string*);
bool validate_closed_random_access(const SequenceConfig&, const FrameHeader&, std::string*);
bool validate_open_random_access(const SequenceConfig&, const FrameHeader&,
                                 const PoolState&, std::span<const std::int32_t> required,
                                 std::string*);
bool is_leading_candidate(const PoolState&, std::uint32_t order_hint);
```

Static rules (§5.18.2/§6.4.6/§6.17) that are enforced:

* `order_hint` fits `OrderHintBits`; `refresh_frame_flags` fits `NumRefFrames`;
* a CLK with `max_mlayer_id == 0` infers `allFrames` (the field must match);
* `immediate_output_frame == 0` requires `refresh_frame_flags != 0`;
* only a KEY_FRAME codes `long_term_id_plus_1`; the id must be in
  `[-1, (1 << long_term_frame_id_bits) - 2]`;
* `num_key_ref_frames`/`ref_long_term_id` are OLK-only, `<= 7` entries, and a
  `ref_long_term_id[i] == (1 << long_term_frame_id_bits) - 1` is reserved;
* both are absent when `long_term_frame_id_bits == 0`;
* an OLK infers `immediate_output_frame = 0`;
* `num_total_refs <= min(REFS_PER_FRAME, NumRefFrames)`, `ref_frame_idx < NumRefFrames`;
* an `INTRA_ONLY_FRAME` cannot refresh every slot (`NumRefFrames > 1`).

Temporal-unit rules (§7.3.6/§7.3.7):

* a CLK/OLK may only be the first coded frame unit; CLK and OLK never mix;
* a TU with a leading frame contains only leading frames;
* at least one coded output frame unit per TU (**this is why an OLK must share
  its temporal unit with an output Regular frame** — an OLK is never output);
* all coded output frame units in a TU share one `OrderHint`.

`validate_open_random_access()` additionally enforces the §7.4.4 long-term
condition, which with the §5.18.2 invalidation code is exactly:

* every named id is held in a valid slot when the OLK is encountered;
* the OLK does not refresh a slot holding a named id (the reference would not
  be held at the first Regular frame);
* every live long-term reference the OLK does not refresh is named (otherwise
  the invalidation would drop a reference the spec says survives).

### Planners

```cpp
bool plan_closed_loop_key(const SequenceConfig&, std::uint32_t order_hint,
                          std::int32_t long_term_id, FrameHeader*, std::string*);
bool plan_open_loop_key(const SequenceConfig&, const PoolState&, std::uint32_t order_hint,
                        std::uint8_t olk_slot, std::int32_t olk_long_term_id,
                        std::span<const std::int32_t> retain_long_term_ids,
                        FrameHeader*, std::string*);
bool plan_inter_frame(const SequenceConfig&, FrameRole, std::uint32_t order_hint,
                      std::span<const std::uint8_t> ref_slots, std::uint8_t refresh_slot,
                      bool immediate_output_frame, FrameHeader*, std::string*);
```

`plan_open_loop_key()` puts the OLK picture in `olk_slot` (only that slot is
refreshed, so every retained long-term slot survives) and names
`retain_long_term_ids` in `ref_long_term_id`.

### Stream assembly

```cpp
struct PlannedFrame { FrameHeader header; std::vector<std::uint8_t> tile_payload; };
struct TemporalUnitPlan { std::vector<PlannedFrame> frames; };

bool extract_m2_tiles(std::span<const std::vector<std::uint8_t>> units,
                      std::span<const std::uint32_t> tile_bytes,
                      std::vector<std::vector<std::uint8_t>>* tiles, std::string*);
bool assemble_sequence(const SequenceConfig&, const std::vector<TemporalUnitPlan>&,
                       AssembledSequence*, std::string*);
```

`assemble_sequence()` validates every TU, walks the pool, emits `TD` per TU
plus the sequence header in unit 0, and returns one IVF unit per temporal unit
together with the per-frame pool events (refreshed / invalidated / valid mask).

## Exact bits

`long_term_frame_id_bits` is `f(3)` after `num_ref_frames_minus_1` in
§5.4.6. For a key frame (§5.18.2):

```text
cur_mfh_id                        uvlc()
seq_header_id_in_frame_header     uvlc()
[inter only] frame_is_inter       f(1)
long_term_id_plus_1               f(long_term_frame_id_bits)   # 0 = no id
[OLK, bits != 0, v1.0.0] num_key_ref_frames f(3)
[OLK, bits != 0, v1.0.0] ref_long_term_id[i] f(long_term_frame_id_bits)
immediate_output_frame            f(1)   # inferred 0 (no bits) for an OLK
frame_size_override_flag          f(1)
order_hint                        f(OrderHintBits)
refresh_frame_flags               f(NumRefFrames)  # CLK infers allFrames
```

Goldens in `longterm_test.cpp` (num_ref_frames = 4, order_hint_bits = 4,
long_term_frame_id_bits = 2), exact header bytes:

| Frame | `frame_header_info()` bytes | bits |
|---|---|---|
| CLK, order 0, id 0 | `d81ad000` | 28 |
| OLK, order 1, id 1, `ref_long_term_id = {0}`, slot 1 | `e2049ad000` | 37 |
| P, order 2, ref slot 1, refresh slot 2 | `f129291ad000` | 47 |
| Leading, order 3, ref slot 0, refresh slot 0 | `e1a3211ad000` | 47 |
| OLK, pinned-dav2d layout (no list), slot 1 | `f049ad0000` | 33 |

Plus the tile-group form of the CLK header (`is_first_tile_group` +
`byte_alignment()`): `ec0d6800`.

Byte equivalence with the untouched M2 writer is asserted for
`long_term_frame_id_bits == 0` (`make_sequence_header_obu`, CLK OBU, P OBU and
the temporal delimiter are byte-identical), so the two writers cannot drift.

## Pinned dav2d does not read the OLK long-term list

`dav2d` f9dafbc (nightly 2026-11-10) is the validator. Probing its parser with
an otherwise byte-identical OLK header gives:

| OLK header after `order_hint` | result |
|---|---|
| no extra field | decodes |
| `num_key_ref_frames f(3)=1` + `ref_long_term_id f(2)=0` (v1.0.0) | `Error parsing frame header` |
| `uvlc` count, `f(2)` count, 1-bit flag — at any position inside the header | `Error parsing frame header` |
| bits added only inside the byte-alignment padding | decodes (padding is skipped) |

So that build does not implement the v1.0.0 OLK list. The module therefore has
an explicit `LongTermListSyntax`:

* `kSpecV1` (default) — the §5.18.2 bits above, full §7.4.4 validation;
* `kOmitPinnedDav2d` — no list for an OLK. Retention still happens because
  §5.18.2 only invalidates slots whose `RefLongTermId == -1`, and the "must
  name every live reference" direction is not expressible, so it is not
  enforced. Every other rule is unchanged.

The gate uses `kOmitPinnedDav2d` for its open-random-access phase; the v1.0.0
layout is covered by the goldens, round-trips and pool tests. When a decoder
that implements §5.18.2 is pinned, flip `SequenceConfig::olk_long_term_list`
back to `kSpecV1` and pass the retained ids to `plan_open_loop_key()`; nothing
else changes.

## Decoder gate

`./build_and_test.sh` builds the unit test, then runs three IVF streams
through the pinned decoder with `--strict 1 --threads 1` and requires 0 sample
differences on Y, U and V:

```text
== phase closed: CLK + 2 P frames, long_term_frame_id_bits = 0 ==
  CLK      order  0  obu    566 B  refreshed 0xf  invalidated 0x0  valid 0x1
  REGULAR  order  1  obu    562 B  refreshed 0x1  invalidated 0x0  valid 0x1
  REGULAR  order  2  obu    553 B  refreshed 0x1  invalidated 0x0  valid 0x1
  3 temporal unit(s), 1707 bytes total
  3/3 picture(s) decoded, 0 sample diffs, max 0
  PASS: closed recon exact
== phase open-ltr: CLK, OLK (pinned-dav2d list layout), P frames, long-term retention across the RAP ==
  CLK      order  0  obu    566 B  refreshed 0xf  invalidated 0x0  valid 0x1
  OLK      order  1  obu    560 B  refreshed 0x2  invalidated 0x0  valid 0x3
  REGULAR  order  2  obu    553 B  refreshed 0x4  invalidated 0x0  valid 0x7
  REGULAR  order  3  obu    890 B  refreshed 0x4  invalidated 0x4  valid 0x7
  REGULAR  order  4  obu    920 B  refreshed 0x4  invalidated 0x0  valid 0x7
  4 temporal unit(s), 3517 bytes total
  4/4 picture(s) decoded, 0 sample diffs, max 0
  PASS: open-ltr recon exact
== phase open-no-ltr: open random access, long_term_frame_id_bits = 0 ==
  CLK      order  0  obu    566 B  refreshed 0xf  invalidated 0x0  valid 0x1
  OLK      order  1  obu    559 B  refreshed 0x2  invalidated 0x0  valid 0x3
  REGULAR  order  2  obu    553 B  refreshed 0x4  invalidated 0x0  valid 0x7
  REGULAR  order  3  obu    890 B  refreshed 0x4  invalidated 0x5  valid 0x6
  REGULAR  order  4  obu    920 B  refreshed 0x4  invalidated 0x0  valid 0x6
  4 temporal unit(s), 3516 bytes total
  4/4 picture(s) decoded, 0 sample diffs, max 0
  PASS: open-no-ltr recon exact
```

The 720p gate with `INPUT=testdata/shine_1280x720_60f.y4m` gives the same
result (43.7 kB CLK, 170 kB for the open-ltr stream). The OLK picture is not
in the decoded output (it is a non-output key frame), so each phase compares
the pictures the decoder must output, in coding order.

Nothing is benchmarked: the CPU path is a correctness oracle.

## Integration sites in `src/encoder/m2/`

This module does not edit the M2 core. To wire it in, the following are the
exact touch points.

| Site | Current code | Change |
|---|---|---|
| `m2_headers.h:21-40` (`M2Config`) | no `long_term_frame_id_bits`, `num_ref_frames` fixed by the caller | add `std::uint8_t long_term_frame_id_bits = 0;` and an `olk_long_term_list` field, or replace `M2Config` with `longterm::SequenceConfig` |
| `m2_headers.cpp:163` | `writer.put_bits(0U, 3U); // long_term_frame_id_bits` | write `c.long_term_frame_id_bits` |
| `m2_headers.cpp:182` | `enable_short_refresh_frame_flags = 0` | keep; a per-frame `frame_to_refresh` is not needed for the LTR path |
| `m2_headers.cpp:235-237` | comment: `long_term_id_plus_1` absent; `immediate_output_frame` always coded | key frames write `long_term_id_plus_1 f(bits)`; an OLK codes no `immediate_output_frame` |
| `m2_headers.cpp:239-243` | CLK refresh inferred, nothing else | add the OLK branch: `num_key_ref_frames`/`ref_long_term_id` (v1.0.0), `refresh_frame_flags f(NumRefFrames)`, `OlkEncountered`/`OlkRefresh` bookkeeping |
| `m2_headers.cpp:255-264` | `refresh_frame_flags`, `num_total_refs`, single `ref_slot` | `num_total_refs`/`ref_frame_idx[]` become vectors so a frame can name an LTR slot |
| `m2_headers.cpp:328-338` (`make_frame_obu`) | `keyframe ? kClosedLoopKey : kRegularTileGroup` | select the OBU type from `FrameRole` (`4/5/6/7`) |
| `m2_headers.h:42-50` (`M2FrameHeader`) | `keyframe`, `order_hint`, `ref_slot`, `refresh_frame_flags` | replace with `longterm::FrameHeader` |
| `m2_frame.h:44-64` (`M2Options`) | no random-access policy | add `enable_open_loop_key`, `long_term_frame_id_bits`, `olk_slot`, retained ids, `leading` flags |
| `m2_frame.cpp:988-1003` (sequence + key header) | seq config + `key_header` | take the sequence config from the LTR policy; use `plan_closed_loop_key()` for frame 0 |
| `m2_frame.cpp:1010` (key unit assembly) | `make_frame_obu(config, key_header, key_tile)` | unchanged call, new writer |
| `m2_frame.cpp:1038-1042` (P header) | `ref_slot = 0`, `refresh_frame_flags = 1`, `order_hint = i` | ask the pool for `ref_frame_idx`/`refresh_frame_flags`; code `plan_inter_frame()` |
| `m2_frame.cpp:1044-1050` (P unit assembly) | one TD + one OBU per frame | group the OLK with an output Regular frame in one TU (`longterm::assemble_sequence()` does this) |
| `m2_frame.cpp:1052-1054` (recon plumbing) | each P frame predicts from `result.recon.back()` | predict from the slot the header names, and keep the LTR picture alive when it is used |
| `m2_syntax.cpp` | tile syntax only; no reference-slot dependence | no change: changing `ref_frame_idx` is header-only, which is what makes the M2 tile splice valid |

`m2_frame.cpp` keeps every reconstructed picture, so the "predict from the slot
the header names" step needs a slot→recon map instead of `recon.back()`; that
is the one real behavioural change. Everything else is header plumbing.

## Build and test

```sh
./src/encoder/inter/longterm/build_and_test.sh              # unit tests + gate
./src/encoder/inter/longterm/build_and_test.sh --no-gate    # unit tests only
SANITIZE=1 ./src/encoder/inter/longterm/build_and_test.sh   # ASan+UBSan
WERROR=1   ./src/encoder/inter/longterm/build_and_test.sh
INPUT=testdata/shine_1280x720_60f.y4m ./src/encoder/inter/longterm/build_and_test.sh
```

No CMake change is needed; the script compiles the module plus the untouched
M2 core (the same source list `src/encoder/m2/build_and_test.sh` uses), like
the CDEF/bframes sub-modules do.

## Not implemented

* `OBU_RAS_FRAME` (§7.4.5). `FrameRole` has no RAS value; the pool and the
  writers would need `restricted_prediction_switch` and the `switch` path.
* `OBU_LEADING_SEF` / `OBU_LEADING_TIP` and SEF output (`show_existing_frame`).
  `kLeading` is `OBU_LEADING_TILE_GROUP` only.
* `implicit_output_frame` coding. `monotonic_output_order_flag` is always 1, so
  the field is never present; a `SequenceConfig` with the flag off would need
  that bit (the writer has the branch, the pool does not model output queues).
* Output/picture ordering: the gate relies on `monotonic_output_order_flag = 1`
  (output order == decoding order) so the decoded Y4M order matches coding
  order.
* `immediate_output_frame = 0` on a CLK (a hidden CLK long-term anchor). The
  writer supports it; no gate stream uses it because a CLK refreshes every
  slot, so it cannot leave a second valid slot.
* Rate control interaction. Retaining an LTR slot costs a slot, not bits.

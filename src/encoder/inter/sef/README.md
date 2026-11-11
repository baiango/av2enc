# Show existing frame (SEF)

Encoder-side construction, validation and decoder gate for AV2 v1.0.0
show-existing-frame OBUs.

Normative reference: **AV2 Bitstream & Decoding Process Specification v1.0.0**
(2026-05-28, commit `38f28e7`), §5.18.1/§5.18.2, §6.2.2 Table 6.1, §6.17.2,
§7.2, §7.21.1-§7.21.6 and §7.23.

This directory owns syntax and bookkeeping only. It never codes tile data and
does not modify `src/encoder/m2`, `src/encoder/m1`, any other
`src/encoder/inter/*` directory, `src/main.cpp` or `CMakeLists.txt`.

## OBU forms

| `obu_type` | name | value |
|---|---|---|
| 11 | `OBU_LEADING_SEF` | 11 |
| 12 | `OBU_REGULAR_SEF` | 12 |

`is_sef()` (§5.2.1) is `obu_type == OBU_LEADING_SEF || obu_type ==
OBU_REGULAR_SEF`. Table 6.1 marks both as layer-specific.

An SEF OBU payload is **only** `frame_header(1)` (§5.2.1); the SEF path of
`frame_header_info()` (§5.18.2) returns before frame type, references, tile
info and tile groups are coded:

```
obu_header()                                  §5.2.2
cur_mfh_id                    uvlc()
seq_header_id_in_frame_header uvlc()
ShowExistingFrame = is_sef()                  // 1, implied by obu_type
frame_to_show_map_idx         f(CeilLog2(NumRefFrames))
derive_sef_order_hint         f(1)
if (!derive_sef_order_hint)
  sef_order_hint              f(OrderHintBits)
// return: refresh_frame_flags = 0, immediate_output_frame = 1
trailing_bits()                               §5.2.3
```

There is no `is_first_tile_group` bit and no tile payload. §7.3.3 makes an SEF
its own *coded output frame unit*, so one temporal unit contains exactly one SEF
OBU (plus the temporal delimiter). `make_sef_temporal_unit()` returns that
`TD + SEF` unit, ready for one IVF frame.

## Leading vs regular

The emitted payload bits are identical; only `obu_type` differs.

* `OBU_REGULAR_SEF` (`IsRegular == 1`) is the correct choice for the M2
  closed-GOP stream. M2 writes `OBU_CLOSED_LOOP_KEY` and
  `OBU_REGULAR_TILE_GROUP` only.
* `OBU_LEADING_SEF` marks a leading frame: a frame displayed before the open
  loop key it follows. Leading frames are discarded when decoding starts at
  that OLK without them, and §6.17.2 forbids a leading frame's
  `refresh_frame_flags` from intersecting `OlkRefresh`. On the SEF path
  `IsRegular` only affects `OlkEncountered` / `OlkTUOrderHint`; it does not
  change any bits written here.
* Leading SEFs are supported for syntax completeness and for future OLK
  streams. They must not be used to label a normal M2 frame.

## Output order and buffer retention

SEF sets `refresh_frame_flags = 0`; `decode_frame_wrapup()` (§7.2) skips loop
filtering and §7.23, then calls `output_frame_buffers()` (§7.21.6) with
`derive_sef_order_hint ? frame_to_show_map_idx : -1`.

* **derive = 1** — the referenced frame is output at its own `OrderHint`, then
  §7.21.3 outputs successive implicit frames.
* **derive = 0** — `output_process(-1)` takes the `mixedOutput` branch
  (§7.21.1) and outputs the referenced `FrameStore` as if it had the signalled
  `sef_order_hint`, then outputs successive frames after that hint.
* An SEF never refreshes or invalidates a slot. It only marks the referenced
  frame as already output; a later derive = 1 SEF may no longer use it.
* §6.17.2 requires, for derive = 1, that the reference slot is valid, the frame
  has not already been shown, and `RefImplicitOutputFrame[slot] == 0` and
  `RefImmediateOutputFrame[slot] == 0`. **derive = 1 is therefore only legal
  for a hidden frame.** An already-output frame must use derive = 0.
* A frame can only be shown while it is still retained in a slot. In the
  current M2 stream a key frame refreshes `allFrames`, but §7.23 marks only the
  *first* refreshed slot valid (`RefValid[i] = first`), and each P frame
  refreshes slot 0. So after any P frame only slot 0 is valid and it holds the
  newest frame; older frames cannot be re-shown unless the encoder widens
  `refresh_frame_flags`.
* `monotonic_output_order_flag` is 1 in M2: output order equals decoding order.
  An SEF temporal unit is therefore output at the end of the stream even when
  its `OrderHint` is older. With derive = 1 this is consistent (the hidden
  frame has never been output); with derive = 0 it is the explicit re-show
  case described by the §6.17.2 note.

## Pinned dav2d behaviour

The pinned decoder (`dav2d f9dafbc`) adds two behaviours that matter for the
gate:

1. `src/obu.c` rejects derive = 0 unless the explicit hint reconstructs to the
   referenced frame's `frame_offset` (`if (cpoc != poc) goto error`). The spec
   allows a different hint, but this writer always signals the reference's own
   hint (via `make_show_header`), so the stream is accepted.
2. `dav2d_queue_output()` drops an output whose POC equals `dpb_poc` (the last
   queued output). Re-showing an already-output frame with derive = 0 hits this
   path. `--outputinvisible 1` bypasses the dedup. This is why the pure-append
   gate phase uses that flag; the hidden-frame derive = 1 phase does not.

`dav2d_needs_output_invisible(slot)` exposes rule 2.

## API (`sef.h`)

```cpp
namespace av2enc::inter::sef {

enum class SefKind : uint8_t { kLeading = 11, kRegular = 12 };

struct SefHeader {
  SefKind kind = SefKind::kRegular;
  uint32_t cur_mfh_id = 0;        // M2 codes 0
  uint32_t seq_header_id = 0;     // M2 codes 0
  uint8_t frame_to_show_map_idx = 0;
  bool derive_sef_order_hint = true;
  uint32_t sef_order_hint = 0;    // only when derive == false
};

struct SefParams {
  uint8_t num_ref_frames = 2;     // M2
  uint8_t order_hint_bits = 4;    // M2
  bool film_grain_params_present = false;  // non-zero is rejected
};

size_t sef_frame_header_bits(const SefParams&, const SefHeader&);
size_t write_sef_frame_header(BitWriter&, const SefParams&, const SefHeader&);
std::vector<uint8_t> make_sef_obu(const SefParams&, const SefHeader&);
std::vector<uint8_t> make_sef_temporal_unit(const SefParams&, const SefHeader&);

bool parse_sef_obu(span<const uint8_t> bytes, const SefParams&,
                   SefHeader&, size_t& bytes_consumed, SefError*);
bool parse_sef_frame_header(span<const uint8_t> payload, const SefParams&,
                            SefHeader&, size_t& bits_consumed, SefError*);

struct RefSlotState {
  bool valid; uint32_t order_hint; uint32_t frame_store_id;
  bool immediate_output_frame; bool implicit_output_frame; bool shown;
};
struct SlotMap { uint8_t num_ref_frames; std::array<RefSlotState,16> slots; ... };
struct RefreshFrame { bool key_frame; uint32_t order_hint;
                      uint32_t frame_store_id; bool immediate_output_frame;
                      bool implicit_output_frame; uint32_t refresh_frame_flags; };

void apply_refresh(SlotMap&, const RefreshFrame&);  // §7.23 first-refreshed rule
void mark_shown(SlotMap&, uint8_t index);           // §7.21.1 bookkeeping

bool validate_sef_reference(const SlotMap&, const SefParams&,
                            const SefHeader&, SefError*);   // §6.17.2
uint32_t disp_order_hint(uint32_t lsbs, uint32_t max_disp, uint8_t bits);
bool sef_order_hint_lsbs(uint32_t target, uint32_t max_disp, uint8_t bits,
                         uint32_t& lsbs);
bool make_show_header(const SlotMap&, const SefParams&, uint8_t index,
                      SefKind, SefHeader&, SefError*);

bool validate_dav2d_compat(const SlotMap&, const SefParams&,
                           const SefHeader&, SefError*);
bool dav2d_needs_output_invisible(const RefSlotState&);
}
```

`make_show_header` chooses derive = 1 when the slot is hidden and unshown,
otherwise derive = 0 with the reference's own order hint. `SefError` has a
`sef_error_name()` string for every failure.

## Bit goldens

With `num_ref_frames = 2`, `order_hint_bits = 4`, ids 0 (the M2 configuration):

| kind | idx | derive | hint | payload | OBU bytes |
|---|---|---|---|---|---|
| regular | 0 | 1 | — | `d8` | `02 30 d8` |
| leading | 0 | 1 | — | `d8` | `02 2c d8` |
| regular | 0 | 0 | 1 | `c1 80` | `03 30 c1 80` |
| leading | 0 | 0 | 1 | `c1 80` | `03 2c c1 80` |
| regular | 1 | 1 | — | `f8` | `02 30 f8` |
| regular | 1 | 0 | 0 | `e0 80` | `03 30 e0 80` |
| regular (4 refs) | 2 | 0 | 5 | `e2 c0` | `03 30 e2 c0` |
| regular (16 refs) | 15 | 1 | — | `ff` | `02 30 ff` |

`TD + regular(0, derive=0, hint=1)` is `01 08 03 30 c1 80`.

Hand derivation: `cur_mfh_id uvlc(0) = 1`, `seq_header_id uvlc(0) = 1`,
`frame_to_show_map_idx f(1)`, `derive f(1)`, optional `sef_order_hint f(4)`,
then `trailing_one_bit = 1` and zero padding.

## Tests

```sh
./src/encoder/inter/sef/build_and_test.sh            # syntax tests + decoder gate
./src/encoder/inter/sef/build_and_test.sh --test-only
./src/encoder/inter/sef/build_and_test.sh --gate-only
SANITIZE=1 ./src/encoder/inter/sef/build_and_test.sh
WERROR=1   ./src/encoder/inter/sef/build_and_test.sh
DAV2D=/path/to/dav2d ./src/encoder/inter/sef/build_and_test.sh

# Run the gate on an existing verified 2- or 3-frame M2 IVF:
"$TMPDIR/sef_build/sef_gate" --dav2d ./tools/bin/dav2d-macos \
  --input-ivf m2_gate.ivf --work /tmp/sef_existing_2f
"$TMPDIR/sef_build/sef_gate" --dav2d ./tools/bin/dav2d-macos \
  --input-ivf m2_gate.ivf.clip.ivf --work /tmp/sef_existing_3f
```

`sef_test` covers:

* exact OBU bit/byte goldens for both kinds, both derive modes, 1/2/4/16
  reference slots and several hint widths;
* writer/reader round-trips over `num_ref_frames ∈ {1,2,3,4,8,16}`,
  `order_hint_bits ∈ {1,2,4,8,16}`, every index, both derive modes;
* truncation, non-SEF OBU, and bad trailing-bit rejection;
* §7.23 key-frame retention (`RefValid = first`) and P-frame overwrite;
* §6.17.2 derive = 1 rejection for shown / non-hidden frames;
* explicit-hint reconstruction (`disp_order_hint`) including the window wrap,
  and the pinned-dav2d hint-equality rule;
* naming helpers.

`build_and_test.sh` builds the syntax test with `-Werror` when `WERROR=1`.
The M2 core is not `-Werror`-clean (pre-existing sign-conversion warnings), so
the gate link line drops only `-Werror`; all SEF checks stay enabled.

## Decoder gate

The gate builds a 2-frame and a 3-frame 64x64 M2 stream with
`av2enc::m2::encode_m2_sequence`, verifies the baseline decode against the
encoder reconstruction, then appends SEF OBUs. Every decode is:

```
dav2d -q -i in.ivf -o out.y4m --threads 1 --strict 1 [--outputinvisible 1]
```

The gate also accepts `--input-ivf` for an already verified stream. Both
`m2_gate.ivf` (2 frames, 1280x720) and `m2_gate.ivf.clip.ivf` (3 frames,
1280x720) pass all three phases with the default `--threads 1 --strict 1`
decode line.

Observed output for the generated streams (`SEF GATE PASS`):

```
== two-frame M2 + SEF: 2 frame(s) 64x64 ==
  baseline: 2 frame(s), recon exact
  A: OBU_REGULAR_SEF idx=0 derive=0 hint=1 obu=[03 30 c1 80]
  A default: duplicate POC dropped (outputs=2)
  A pass: extra output == baseline frame 1
  B pass: OBU_REGULAR_SEF + OBU_LEADING_SEF => both extras == baseline frame 1
  C pass: hidden frame + derive=1 SEF => output 1 == baseline frame 1 (no --outputinvisible)
== three-frame M2 + SEF: 3 frame(s) 64x64 ==
  baseline: 3 frame(s), recon exact
  A: OBU_REGULAR_SEF idx=0 derive=0 hint=2 obu=[03 30 c2 80]
  A default: duplicate POC dropped (outputs=3)
  A pass: extra output == baseline frame 2
  B pass: OBU_REGULAR_SEF + OBU_LEADING_SEF => both extras == baseline frame 2
  C pass: hidden frame + derive=1 SEF => output 2 == baseline frame 2 (no --outputinvisible)
SEF GATE PASS
```

Phase meaning:

* **A** — pure append to the unmodified stream. The SEF uses derive = 0 with
  the last frame's order hint. The pinned decoder drops the duplicate POC by
  default; with `--outputinvisible 1` exactly one extra frame appears and is
  byte-identical to the referenced frame.
* **B** — two appended SEFs (regular + leading) each produce a byte-identical
  extra output, proving multiple SEFs and both OBU kinds.
* **C** — normative hidden path. The last M2 frame's `immediate_output_frame`
  bit is cleared in place (tile payload untouched), then a derive = 1 SEF
  re-shows it. No `--outputinvisible` is needed and the output sequence is
  baseline frames `0..N-2` followed by the SEF copy of frame `N-1`.

## M2 integration sites

`sef_gate.cpp` needs no M2 edits. When M2 itself starts emitting SEF, these are
the exact places to touch:

| file | site | change |
|---|---|---|
| `src/encoder/m2/m2_frame.h:66` | `struct M2SequenceResult` | add a per-unit output descriptor (`kCodedFrame` vs `kShowExisting`) next to `units`, or a parallel `output_order` vector |
| `src/encoder/m2/m2_headers.h:42` | `struct M2FrameHeader` | add `bool immediate_output_frame = true;` and write it in `m2_headers.cpp` |
| `src/encoder/m2/m2_headers.cpp:236` | `write_frame_header_syntax()` | replace the hard-coded `writer.put_bit(true);  // immediate_output_frame` with the header field; `implicit` stays inferred 0 because M2 sets `monotonic_output_order_flag = 1` |
| `src/encoder/m2/m2_frame.cpp:1001` | key-frame `M2FrameHeader` | set `immediate_output_frame = false` for a key frame reserved for a later SEF |
| `src/encoder/m2/m2_frame.cpp:1038-1043` | P-frame `M2FrameHeader` loop | set the visibility flag; `order_hint = i` still comes from the coding order |
| `src/encoder/m2/m2_frame.cpp:1042` | `header.refresh_frame_flags = 1;` | widen the refresh mask if an older frame must stay retained for a later SEF (with 2 slots, `0b01` only retains the newest frame; `0b11` would retain the previous frame in slot 1) |
| `src/encoder/m2/m2_frame.cpp:1050` | `result.units.push_back(std::move(unit));` | after the coded unit, push `sef::make_sef_temporal_unit(params, header)` when re-showing a retained slot; do not push `recon`/tile/block stats for it |
| `src/encoder/m2/m2_gate.cpp:257-270` | `write_ivf()` | no change: one `sequence.units[i]` is one IVF frame / temporal unit, which is exactly the SEF unit shape |
| `src/io/ivf.h` | `IvfWriter` | no change |

Ordering rules for that integration:

1. Write the coded unit first; the SEF can only reference a frame already
   decoded and stored by §7.23.
2. SEF input and P-frame input are separate paths; never append an SEF OBU to a
   tile-group temporal unit (§7.3.3).
3. For an SEF, keep `frame_header_info`'s SEF path out of
   `write_frame_header_syntax()`; `make_sef_obu` is a standalone writer.
4. If the referenced frame was output, use derive = 0 and signal its own
   order hint; if it was coded hidden (`immediate_output_frame = 0`,
   `implicit_output_frame = 0`), use derive = 1.

## Limitations

* Only one embedded layer (`max_mlayer_id = 0`), one extended layer, and
  `cur_mfh_id = 0`. Non-zero ids are rejected.
* Film grain parameters are rejected rather than guessed; M2 sets
  `film_grain_params_present = 0`.
* No switch/bridge/TIP frames, no multistream, no long-term references.
* The reader validates the SEF header/trailing bits only; it does not parse
  reference frame state from a full stream.

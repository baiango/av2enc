# Multi-frame header (MFH) — §5.7 writer, reader, semantics, M2 gate

Status: **spec-complete writer/reader + inherited-field resolver + a real
two-and-more-frame M2 decoder gate.** No M1/M2 core edits, no `src/main.cpp`,
no `CMakeLists.txt`. The gate re-emits the untouched M2 tile payloads with an
MFH-aware copy of the frame-header writer, so the only stream difference is the
MFH itself and the inherited bits the frames omit.

Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
(2026-05-28, commit `38f28e7`), `spec/AV2_Spec_v1.0.0.pdf`.

```text
src/encoder/frame/mfh/
  mfh.h                 §5.7 syntax structs, writer/reader, inheritance resolver
  mfh.cpp               implementation; bit reader; §5.18 precedence rules
  mfh_m2_adapter.h/.cpp the exact M2 frame-header mirror used by the gate
  mfh_test.cpp          216 checks: goldens, round-trips, errors, precedence
  mfh_gate.cpp          M2 stream + one MFH + pinned dav2d gate
  build_and_test.sh     standalone build/run (no CMake changes)
  README.md             this file
```

## Spec map

| Syntax / rule | Section |
|---|---|
| `multi_frame_header_obu()` | §5.7 |
| MFH field semantics | §6.7 (the task brief calls it §6.6; this PDF revision numbers it §6.7) |
| `OBU_MULTI_FRAME_HEADER = 3`, layer-specific | §6.2.2 Table 6.1 |
| `obu_extension_flag` must be 0 | §6.2.1 |
| `open_bitstream_unit()` extensible-OBU trailer | §5.2.1 |
| `is_extensible_obu()` includes `OBU_MULTI_FRAME_HEADER` | §5.2.1 |
| `trailing_bits( nbBits )` | §5.2.3 |
| `MAX_MFH_NUM = 16`, `MAX_SEQ_NUM = 16` | §6.2.1 constants |
| `cur_mfh_id`, `seq_header_id_in_frame_header` | §5.18.2, §6.17.2 |
| `load_sequence_header( MfhSeqHeaderId[ cur_mfh_id ] )` | §5.18.2 |
| `frame_size()` uses `mfh_frame_width/height_minus_1` | §5.18.4.1 |
| `deblocking_filter_params()` uses the MFH apply flags | §5.18.5.2 |
| `segmentation_params()` reuse / `mfh_allow_seg_info_change` | §5.18.7.1 |
| `seg_info( numSegments )`, `SEG_LVL_MAX = 3` | §5.4.9 |
| `Segmentation_Feature_Bits/Signed/Max` | §5.18.7.2 |
| MFH availability before `cur_mfh_id` | §7.3.8.7 |
| Sequence header availability | §7.3.8.6 |
| MFH placement (zero or more before the frame) | §7.3.3 / §7.3.4 |
| `uvlc()`, `f(n)`, `su(n)`, `trailing_bits()` | §4.11.3, §4.11.2, §4.11.7, §5.2.3 |

## What an MFH carries

```
multi_frame_header_obu() {
  uvlc()  mfh_seq_header_id
  uvlc()  mfh_id_minus_1                 // mfhId = mfh_id_minus_1 + 1
  f(1)    mfh_frame_size_present_flag
  if (present) {
    f(4)  mfh_frame_width_bits_minus_1   // n = value + 1, 1..16
    f(4)  mfh_frame_height_bits_minus_1
    f(n)  mfh_frame_width_minus_1
    f(n)  mfh_frame_height_minus_1
  }
  f(1)    mfh_deblocking_filter_update
  if (update) {
    f(1)  mfh_apply_deblocking_filter[4] // Y-vert, Y-horiz, U, V
  }
  f(1)    mfh_seg_info_present_flag
  if (present) {
    f(1)  mfh_ext_seg_flag               // 0 = 8 segments, 1 = 16
    f(1)  mfh_allow_seg_info_change
    seg_info( ext ? 16 : 8 )             // §5.4.9
  }
}
// §5.2.1 open_bitstream_unit(), because is_extensible_obu() contains type 3:
f(1)      obu_extension_flag             // payload trailer, must be 0
trailing_bits( remainingPayloadBits - 1 )
```

There are **two** different extension flags:

* `obu_header_extension_flag` — bit 7 of the OBU header byte. The MFH OBU
  header is `0x0C` for `obu_tlayer_id = 0` (header flag 0, type 3, tlayer 0).
* `obu_extension_flag` — the §5.2.1 extensible-payload trailer bit, read after
  the §5.7 syntax and before `trailing_bits()`. §6.2.1 requires it to be 0.

Annex B.2 framing is `leb128(header + payload) header payload`.
`mfh_seq_header_id` must be `< MAX_SEQ_NUM`; `mfh_id` must be in
`[1, MAX_MFH_NUM)`. `obu_mlayer_id` is inferred 0 because conforming streams
must have `obu_header_extension_flag == 0` as well.

## Pinned bit goldens

These are recomputed from the writer and asserted verbatim in `mfh_test.cpp`.

| Case | Complete OBU |
|---|---|
| minimal: seq 0, id 1, nothing else | `02 0c c2` |
| 64×64, size present, deblocking update with `{0,0,0,0}`, 8-seg table all disabled | `09 0c ea bf ff 08 00 00 00 80` |
| 1920×1080, size present, deblocking `{1,0,1,0}`, no segments | `07 0c f5 5d fe 1b e8 80` |

Hand derivation of the minimal payload: `uvlc(0)=1`, `uvlc(0)=1`,
`size_present=0`, `update=0`, `seg_present=0` → `1 1 0 0 0`; then
`obu_extension_flag = 0` at bit 5, the trailing one bit at bit 6 and a zero pad
bit at bit 7 → `11000010` = `0xc2`. The 64×64 case has 55 syntax bits, then the
flag at bit 55, the trailing one at bit 56 and zero padding to 64 bits
(`... 00 00 00 80`). The 1080p case has 39 syntax bits and a 6-byte payload.
An audit that looks only at the OBU header will miss the payload flag; both are
required.

## Inheritance semantics and precedence

`FrameHeaderFields.cur_mfh_id` selects the MFH. `cur_mfh_id == 0` is the
no-MFH path; `cur_mfh_id > 0` requires an available MFH with that id.

| Field | `cur_mfh_id == 0` | MFH referenced | Explicit frame-header value wins? |
|---|---|---|---|
| sequence header | `seq_header_id_in_frame_header` is coded | `load_sequence_header( MfhSeqHeaderId[id] )`; the field is **absent** | no |
| frame size | sequence `max_frame_width/height` | `mfh_frame_width/height_minus_1 + 1` when `mfh_frame_size_present_flag` | **yes** — `frame_size_override_flag = 1` replaces it |
| deblocking apply flags | two (or four) bits in the frame header | MFH flags; the apply bits are **absent** | no — the MFH wins |
| segment table | sequence `seq_seg_info_present_flag` path | MFH `seg_info` when `mfh_ext_seg_flag == enable_ext_seg` | **conditionally** — `mfh_allow_seg_info_change` codes `reuse_seg_info` |

Exact deblocking rules (§5.18.5.2):

1. `CodedLossless == 1` forces all flags to 0 and returns early, before either
   the MFH or explicit bits. Our resolver reports `FieldSource::kForced`.
2. Otherwise, when `mfh_deblocking_filter_update[cur_mfh_id] == 1`, the frame
   header codes **no** `apply_deblocking_filter` bits. Chroma flags are forced
   to 0 when both luma flags are 0; otherwise they come from the MFH.
3. `df_delta_q_present` / `df_delta_q` are still coded for every plane whose
   resolved apply flag is 1, even when the flags came from the MFH. The MFH
   does not inherit the delta-q fields.
4. Only when the MFH has no update flag does the frame header code the apply
   bits. The resolver reports `deblocking_bits_present = true` so an encoder
   knows which bits to write.

Exact segmentation rules (§5.18.7.1):

```
if (segmentation_enabled == 0) -> all features disabled
else if (cur_mfh_id > 0 && mfh_seg_info_present) {
    haveSegParams = (mfh_ext_seg_flag == sequence.enable_ext_seg)
    allowChange   = haveSegParams && mfh_allow_seg_info_change
    source        = MFH
} else if (seq_seg_info_present) {
    haveSegParams = 1
    allowChange   = seq_allow_seg_info_change
    source        = sequence
} else {
    haveSegParams = 0
    allowChange   = 0
}
if (allowChange) reuse_seg_info = coded bit
else             reuse_seg_info = haveSegParams
if (reuse_seg_info) copy source features
else                parse a new seg_info( MaxSegments )
```

Two consequences worth pinning:

* An `mfh_ext_seg_flag` mismatch does **not** fall through to the sequence
  table. The MFH branch was taken, `haveSegParams` is false, and the frame must
  code a fresh table (`mfh_test.cpp` asserts this).
* `MaxSegments` comes from the **sequence** `enable_ext_seg`, not from the MFH
  flag. A matching MFH ext flag is required to reuse the MFH table.

## Conformance / validation

The writer is strict and refuses values that the syntax would silently clip or
ignore; the reader enforces bitstream requirements.

| Check | Writer | Reader |
|---|---|---|
| `mfh_seq_header_id < 16` | reject | reject |
| `mfh_id` in `[1, 16)` | reject | reject |
| `obu_tlayer_id <= 3` | reject | reject |
| payload `obu_extension_flag == 0` (§5.2.1 trailer) | writer always clears | reject |
| `obu_header_extension_flag == 0` (OBU header byte) | writer always clears | reject |
| frame size present but width/height 0 | reject | reject (0 underflows to 0 after `+1`) |
| `mfh_frame_*_bits` outside `[1, 16]` | reject | encoded field is 4 bits + 1 |
| size minus 1 does not fit the coded width | reject | cannot occur from a read |
| size fields non-zero without the present flag | reject | not coded |
| apply flags without `mfh_deblocking_filter_update` | reject | not coded |
| `mfh_allow_seg_info_change` without seg info | reject | not coded |
| segment features without seg info | reject | not coded |
| `ALT_Q` outside `[-351, 351]` (`MAXQ_BITS`) | reject | **clipped**, as the spec requires |
| `SEG_LVL_SKIP` / `SEG_LVL_GLOBALMV` value ≠ 0 | reject | always 0 (zero coded bits) |
| feature row ≥ segment count | reject | rows are not coded |
| trailing bits (`1` then zeros) | writer emits | reject a missing 1 or nonzero padding |
| referenced MFH missing | — | reject |
| `mfh_frame_*` > sequence maximum | — | reject |
| referenced sequence header unavailable | — | reject |
| `TLayerDependencyMap` / `MLayerDependencyMap` | — | checked when a `LayerDependencyMap` is supplied |

`seg_info()` feature geometry (§5.4.9 / §5.18.7.2):

| `j` | Name | Bits | Signed | Max |
|---|---|---|---|---|
| 0 | `SEG_LVL_ALT_Q` | 9 | yes (`su(10)`) | `MAXQ_BITS = 351` |
| 1 | `SEG_LVL_SKIP` | 0 | no | 0 |
| 2 | `SEG_LVL_GLOBALMV` | 0 | no | 0 |

The `feature_enabled` bit is still present for `j = 1, 2`; enabling them codes
zero value bits and a zero value.

## API

```cpp
namespace av2enc::frame::mfh {

struct SegmentFeature { bool enabled; int32_t value; };
struct SegInfo { bool ext_seg; std::array<std::array<SegmentFeature,3>,16> feature; };
struct MultiFrameHeader {
  uint32_t seq_header_id; uint32_t mfh_id; uint8_t tlayer_id;
  bool frame_size_present; uint8_t frame_width_bits, frame_height_bits;
  uint32_t frame_width, frame_height;
  bool deblocking_filter_update; std::array<bool,4> apply_deblocking_filter;
  bool seg_info_present; bool allow_seg_info_change; SegInfo seg_info;
};

bool validate_multi_frame_header(const MultiFrameHeader&, std::string*);
size_t write_multi_frame_header_syntax(BitWriter&, const MultiFrameHeader&,
                                       std::string*);
std::vector<uint8_t> make_multi_frame_header_obu(const MultiFrameHeader&,
                                                 std::string*);

struct ObuView { uint8_t obu_type, tlayer_id, mlayer_id, xlayer_id;
                 bool extension_flag; size_t offset, payload_offset,
                 payload_size, total_size; };
bool parse_obu(span<const uint8_t>, size_t offset, ObuView*, std::string*);
bool parse_multi_frame_header_payload(span<const uint8_t>, MultiFrameHeader*,
                                      std::string*);
bool parse_multi_frame_header_obu(span<const uint8_t>, size_t offset,
                                  MultiFrameHeader*, ObuView*, std::string*);

class MfhStore {           // decoder-side MfhSeqHeaderId[]/MfhTLayerId[] state
  void reset();
  bool insert(const MultiFrameHeader&, std::string*);
  const MultiFrameHeader* find(uint32_t mfh_id) const;
  bool parse_and_insert(span<const uint8_t>, size_t offset,
                        MultiFrameHeader*, std::string*);
  size_t size() const;
};
bool scan_multi_frame_headers(span<const uint8_t>, MfhStore*, size_t*, std::string*);

enum class FieldSource { kAbsent, kSequence, kMfh, kExplicit, kForced };
struct SequenceFrameFields { uint16_t max_frame_width, max_frame_height;
                             bool monochrome, enable_ext_seg;
                             bool seq_seg_info_present, seq_allow_seg_info_change;
                             SegInfo seq_seg_info;
                             std::array<bool,16> sequence_headers; };
struct FrameHeaderFields { uint32_t cur_mfh_id, seq_header_id;
                           uint8_t obu_mlayer_id, obu_tlayer_id;
                           bool frame_size_override; uint32_t frame_width, frame_height;
                           bool coded_lossless;
                           std::array<bool,4> apply_deblocking_filter;
                           bool segmentation_enabled, reuse_seg_info;
                           SegInfo seg_info; };
struct ResolvedFrameFields { uint32_t seq_header_id, frame_width, frame_height;
                             FieldSource frame_size_source;
                             std::array<bool,4> apply_deblocking_filter;
                             FieldSource deblocking_source;
                             bool deblocking_bits_present;
                             bool deblocking_delta_q_bits_present[4];
                             bool segmentation_enabled; int max_segments;
                             bool reuse_seg_info_bit_present, reuse_seg_info;
                             FieldSource seg_info_source; SegInfo seg_info; };
bool resolve_frame_fields(const MfhStore&, const SequenceFrameFields&,
                          const FrameHeaderFields&,
                          const LayerDependencyMap*, ResolvedFrameFields*,
                          std::string*);
}
```

`mfh_m2_adapter.h` mirrors `m2::write_frame_header_syntax()` for the M2 subset:

```cpp
struct M2MfhConfig { uint16_t max_frame_width, max_frame_height;
                     uint8_t base_q_idx, order_hint_bits, num_ref_frames,
                             num_total_refs;
                     bool disable_cdf_update, enable_ext_seg; };
struct M2MfhFrameHeader { bool keyframe, frame_is_inter; uint32_t order_hint,
                          refresh_frame_flags; uint8_t ref_slot;
                          bool allow_intrabc, frame_size_override;
                          uint32_t frame_width, frame_height;
                          bool segmentation_enabled, reuse_seg_info;
                          SegInfo seg_info; };
size_t write_m2_frame_header_with_mfh(BitWriter&, const M2MfhConfig&,
                                      const M2MfhFrameHeader&,
                                      const MultiFrameHeader*, std::string*);
size_t write_m2_tile_group_header_with_mfh(...);
std::vector<uint8_t> make_m2_frame_obu_with_mfh(..., span<const uint8_t> tile,
                                                std::string*);
```

With `mfh == nullptr` the adapter is **bit-exact** with the untouched M2
writer; `mfh_test.cpp` compares the complete tile-group header bytes for key
and inter frames.

## Tests

```sh
src/encoder/frame/mfh/build_and_test.sh --no-gate
# or with the decoder:
src/encoder/frame/mfh/build_and_test.sh
SANITIZE=1 src/encoder/frame/mfh/build_and_test.sh
```

216 checks cover:

* the three pinned goldens above;
* full-field round-trips for 8- and 16-segment tables, explicit size bit
  widths, all deblocking flags, ALT_Q at both limits;
* scanning several MFHs in one temporal unit, `MfhStore` insert/overwrite/reset;
* the §5.2.1 extensible-OBU trailer: an explicit bit-position assertion that
  `obu_extension_flag` follows the §5.7 syntax and precedes `trailing_bits()`,
  plus a reader rejection for `obu_extension_flag == 1` with no extension data;
* every writer rejection listed above, plus payload truncation, wrong OBU type,
  `obu_header_extension_flag == 1`, a corrupted trailing bit and an empty
  payload;
* frame-size precedence (sequence → MFH → explicit override), missing MFH,
  missing sequence header, oversized MFH and oversized explicit size;
* deblocking precedence (MFH, explicit, CodedLossless, chroma-forced-off,
  monochrome, `df_delta_q` bits);
* segmentation precedence (MFH reuse without a bit, allow-change bit in both
  states, ext-segment mismatch, sequence fallback, 16-segment rows, disabled);
* layer dependency maps (pass, temporal violation, embedded-layer violation,
  missing map for a multi-layer reference);
* M2 adapter bit-exactness and the 1-bit net saving of the default MFH header.

The suite is clean under `-fsanitize=address,undefined` and `WERROR=1`.

## Decoder gate

```sh
src/encoder/frame/mfh/build_and_test.sh
src/encoder/frame/mfh/build_and_test.sh --frames 4
INPUT=testdata/shine_1280x720_60f.y4m src/encoder/frame/mfh/build_and_test.sh --frames 4
```

The gate:

1. encodes a short M2 sequence with the untouched encoder (synthetic 64×64 by
   default, or `INPUT=...`);
2. writes the baseline IVF;
3. re-emits every unit through `mfh_m2_adapter` with `mfh == nullptr` and
   requires the pinned decoder to produce the same samples (proves the adapter
   and the OBU/tile split are correct);
4. writes a second stream with **one** MFH OBU after the sequence header in the
   first temporal unit and every frame header using `cur_mfh_id = 1`. The MFH
   has the real frame size, `mfh_deblocking_filter_update = 1` with all four
   flags 0, and an 8-segment all-disabled `seg_info`;
5. decodes every stream with `dav2d -q --threads 1 --strict 1` and compares Y,
   U and V against `M2SequenceResult::recon`, requiring **0 differing samples**;
6. probes an MFH whose `mfh_seq_header_id` is 1 while only sequence header 0 is
   present, and requires `--strict 1` to reject the stream.

Example output (8-frame synthetic 64×64 clip, pinned `dav2d` `f9dafbc`):

```
mfh_test: 216 checks, 0 failures
mfh_gate: decoder tools/bin/dav2d-macos
mfh_gate: 8 frame(s) 64x64, base q 90
  baseline payload 7021 bytes; reheader 7021; MFH 7031 (+10 MFH)
  baseline 8 frame(s), 0 sample diff(s)
  reheader 8 frame(s), 0 sample diff(s)
  mfh      8 frame(s), 0 sample diff(s)
mfh_gate: 6 checks, 0 failure(s)
MFH GATE PASS
```

The 1280×720 4-frame run also passes (`baseline 130513`, `MFH 130524`, 0 diffs;
the 720p MFH stays 11 bytes because the extra flag bit fits in existing byte
padding).

What the gate proves with this pinned decoder:

* `OBU_MULTI_FRAME_HEADER` (type 3) is parsed and accepted;
* `cur_mfh_id = 1` loads the sequence header through `MfhSeqHeaderId[1]` — an
  unavailable id is rejected;
* the frame headers really omit the two `apply_deblocking_filter` bits: if the
  decoder did not take them from the MFH, the range-coded tile would desync and
  the reconstruction would not be exact;
* the MFH `seg_info()` bits are parsed (the following frame OBU must still be
  found at the right offset).

Pinned-decoder limitations observed while building the gate:

* **MFH frame size is not applied.** MFH sizes 32×32, 48×32, 64×64 and 128×64
  (with sequence maximum 64×64) all decode as 64×64 and are accepted. The
  stream stays conformant; a spec-conformant decoder uses
  `mfh_frame_width_minus_1` / `mfh_frame_height_minus_1` per §5.18.4.1. The gate
  therefore proves the *syntax* and
  acceptance, not the decoded dimensions. Unit tests pin the resolver.
* **MFH trailing bits are not validated.** Clearing the trailing one bit of the
  MFH payload is accepted; the same decoder rejects a corrupted HDR metadata
  trailing byte. Our reader rejects it (`mfh_test.cpp`). Do not rely on `dav2d`
  to catch malformed MFH padding.

These are validator limitations, not reasons to change the writer.

## Exact M2 integration sites

No core file was modified. To ship MFH in the encoder:

1. `src/encoder/m2/m2_headers.h`
   * add `uint32_t mfh_id = 0;` (or an owning `MultiFrameHeader`) to `M2Config`;
   * add `uint32_t cur_mfh_id = 0;` to `M2FrameHeader`.

2. `src/encoder/m2/m2_headers.cpp:228-229` — replace

   ```cpp
   put_uvlc(writer, 0);  // cur_mfh_id
   put_uvlc(writer, 0);  // seq_header_id_in_frame_header
   ```

   with

   ```cpp
   put_uvlc(writer, c.mfh_id);              // cur_mfh_id
   if (c.mfh_id == 0) put_uvlc(writer, 0);  // seq_header_id_in_frame_header
   ```

   (the M2 stream uses sequence header id 0; when `mfh_id != 0` the sequence is
   loaded through `MfhSeqHeaderId`).

3. `src/encoder/m2/m2_headers.cpp:283-288` — the current unconditional
   segmentation/deblocking bits. Gate them:

   ```cpp
   const bool lossless = false;  // M2 rejects q == 0
   const bool mfh_deblock = c.mfh_id != 0 && mfh.deblocking_filter_update;
   writer.put_bit(segmentation_enabled);
   writer.put_bit(false);  // using_qmatrix
   writer.put_bit(false);  // delta_q_present
   if (!mfh_deblock) {
     writer.put_bit(apply[0]);
     writer.put_bit(apply[1]);
   } else {
     for (i = 0; i < 4; ++i)
       if (mfh.apply[i]) writer.put_bit(false);  // df_delta_q_present
   }
   ```

   A real segmentation implementation must also mirror the §5.18.7.1 control
   flow from `mfh::resolve_frame_fields()` (reuse bit, ext-seg match, new
   `seg_info()`), because `enable_segmentation` changes the tile syntax.

4. `src/encoder/m2/m2_frame.cpp:999-1003` — build the key `M2FrameHeader` with
   `cur_mfh_id = 1` when MFH is enabled.

5. `src/encoder/m2/m2_frame.cpp:1005-1015` — insert the MFH OBU between the
   sequence header and `OBU_CLOSED_LOOP_KEY`:

   ```cpp
   const auto mfh_obu = mfh::make_multi_frame_header_obu(mfh, &error);
   key_unit.insert(key_unit.end(), mfh_obu.begin(), mfh_obu.end());
   ```

   §7.3.3 permits zero or more MFH OBUs before the coded frame.

6. `src/encoder/m2/m2_frame.cpp:1038-1050` — set `cur_mfh_id = 1` on every P
   frame. Do **not** repeat the MFH OBU: `Mfh*` state persists, and §7.3.8.7
   only requires that the MFH be available before the first reference. To place
   a fresh MFH in a later temporal unit, insert it before that unit's frame OBU
   using the same helper.

7. A reader front end (not present in this tree) should:
   * call `mfh::scan_multi_frame_headers()` (or `parse_multi_frame_header_obu()`
     + `MfhStore::insert()`) for every type-3 OBU;
   * parse `cur_mfh_id` first in `frame_header_info()`;
   * call `resolve_frame_fields()` with the sequence-level `max_frame_*`,
     `enable_ext_seg`, `seq_seg_info` and the set of available sequence headers;
   * use `deblocking_bits_present` / `deblocking_delta_q_bits_present` to know
     which bits `deblocking_filter_params()` consumes;
   * use `reuse_seg_info_bit_present` to know whether `reuse_seg_info` is coded.

8. Build wiring (not applied; M2 itself is not in the root `CMakeLists.txt`):

   ```cmake
   target_sources(av2enc_core PRIVATE
     src/encoder/frame/mfh/mfh.cpp
     src/encoder/frame/mfh/mfh_m2_adapter.cpp)
   # add_executable(mfh_test src/encoder/frame/mfh/mfh_test.cpp)
   # add_test(NAME mfh_feature_gate COMMAND bash
   #   src/encoder/frame/mfh/build_and_test.sh)
   ```

## Cost note

For the gate's MFH (id + frame size + deblocking update + 8-segment disabled
table) the OBU is **10 bytes = 80 bits**. Each frame saves the two deblocking
apply bits and pays one extra `cur_mfh_id` uvlc bit, i.e. **1 bit/frame**.
Break-even is therefore ~80 frames for this particular field mix. MFH is not a
win for a 2-frame GOP; it pays off with long sequences, repeated explicit frame
sizes / segment tables, or 16-segment tables with several enabled features
(8×3 or 16×3 feature bits plus 10-bit ALT_Q values per enabled entry). The cost
is independent of picture size; the per-frame savings are not. This is a
tooling decision for `src/main.cpp`, not a conformance requirement.

## Known limits

* MFH is not enabled in the M2 core; the gate re-headers M2 tile payloads. The
  exact replacement sites are above.
* The M2 adapter supports `frame_size_override` only for intra frames. An inter
  frame needs the `frame_size_with_refs()` `found_ref` bits first; the adapter
  rejects it rather than emitting a wrong header. The real M2 writer would add
  those bits in its existing `frame_size` slot.
* `obu_header_extension_flag` writers are not supported: §6.2.1 requires the
  OBU header flag to be 0, so `obu_mlayer_id` is always 0. The mandatory payload
  `obu_extension_flag` trailer (§5.2.1) is always written as 0. Multi-layer MFH
  references can be *validated* with an explicit `LayerDependencyMap` but
  cannot be *written* with a non-zero mlayer id.
* The gate cannot prove segment-table inheritance because enabling
  `segmentation_enabled` changes the M2 tile syntax (`segment_id` symbols). The
  resolver and bit writer/reader are covered by unit tests; the gate proves
  that the MFH `seg_info()` payload is parsed without desynchronising.
* The gate cannot prove MFH frame-size application because the pinned decoder
  ignores it (see above).
* `seg_info()` values are clipped per the spec on read and rejected outside the
  range on write. No other feature levels exist in v1.0.0 (`SEG_LVL_MAX == 3`).

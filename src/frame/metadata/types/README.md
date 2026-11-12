# Remaining AV2 metadata unit types (`src/encoder/metadata/types/`)

AV2 v1.0.0 metadata units other than HDR CLL/MDCV: ITU-T T.35, timecode,
banding hints, ICC profile, scan type, temporal point info, decoded frame hash
and user data unregistered. Carried in `OBU_METADATA_SHORT` (type 8) or
`OBU_METADATA_GROUP` (type 9).

This directory is self-contained. It does **not** edit the HDR module, the
M1/M2 core, `src/main.cpp` or `CMakeLists.txt`. OBU framing is reused from
`av2enc::hdr` by including its public header, not by copying its implementation.

Normative reference: `spec/AV2_Spec_v1.0.0.pdf` (v1.0.0, commit `38f28e7`).

| Syntax | Section |
|---|---|
| `metadata_unit(metadataPayloadSize)`, `metadata_unit_remaining_bit` | §5.17.1 |
| `metadata_short_obu(obuPayloadSize)` | §5.17.2 |
| `metadata_group_obu()` | §5.17.3 |
| `metadata_itut_t35(metadataPayloadSize)` | §5.17.4 |
| `metadata_timecode()` | §5.17.7 |
| `metadata_banding_hints()` | §5.17.8 |
| `metadata_icc_profile(metadataPayloadSize)` | §5.17.9 |
| `metadata_scan_type()` | §5.17.10 |
| `metadata_temporal_point_info()` | §5.17.11 |
| `metadata_decoded_frame_hash()` | §5.17.12 |
| `metadata_user_data_unregistered(metadataPayloadSize)` | §5.17.13 |
| ITU-T T.35 semantics | §6.16.4 |
| timecode semantics | §6.16.7 |
| banding hints semantics | §6.16.8 |
| ICC profile semantics | §6.16.9 |
| scan type semantics (Table 6.18) | §6.16.10 |
| temporal point info (short OBU only) | §6.16.11 |
| user data unregistered semantics | §6.16.12 |
| decoded frame hash semantics (MD5) | §6.16.13 |
| `metadata_type` values (Table 6.17) | §6.16.3 |
| prefix/suffix placement | §7.3.3 |

## Files

| File | Purpose |
|---|---|
| `metadata_types.h` / `metadata_types.cpp` | values, validation, serializers, decoders, short/group framing, scanner |
| `metadata_types_test.cpp` | 17 test groups: pinned goldens, round-trips, framing matrix, malformed inputs |
| `append_metadata_types.cpp` | IVF tool: append every type (short/group, prefix/suffix, cancel) or scan |
| `build.sh` | standalone build (no CMake target) |
| `run_metadata_gate.sh` | M2 encode + pinned `dav2d --strict 1` decoder gate |

## Exact payload layouts

All fields are MSB-first `f(n)` unless noted. Writers emit the **canonical**
byte count: `ceil(syntax_bits / 8)`, with the final partial byte zero-padded by
`metadata_unit_remaining_bit` (§5.17.1). Readers accept a longer payload and
ignore the extra bits/bytes, but the framing writers reject non-canonical
padding.

### `metadata_itut_t35` — type 3

```
f(8)  itu_t_t35_country_code
      if (country_code == 0xFF)
f(8)    itu_t_t35_country_code_extension_byte
le(n) itu_t_t35_payload_bytes            // raw pass-through
```

Unit size = `1 + (country_code == 0xFF) + payload.size()`. Minimum 1 byte
structurally; §6.16.4 requires a terminal provider code in the first one or
more payload bytes, so the writer rejects an empty payload. The payload is
opaque (dynamic HDR metadata such as HDR10+ rides here).

### `metadata_timecode` — type 4

```
f(5)  counting_type              // 0..6 defined, 7..31 reserved
f(1)  full_timestamp_flag
f(1)  discontinuity_flag
f(1)  cnt_dropped_flag
f(9)  n_frames
if (full_timestamp_flag) {
f(6)    seconds_value            // 0..59
f(6)    minutes_value            // 0..59
f(5)    hours_value              // 0..23
} else {
f(1)    seconds_flag
        if (seconds_flag) {
f(6)      seconds_value
f(1)      minutes_flag
          if (minutes_flag) {
f(6)        minutes_value
f(1)        hours_flag
            if (hours_flag) {
f(5)          hours_value
            }
          }
        }
}
f(5)  time_offset_length         // 0..31
if (time_offset_length > 0)
f(L)    time_offset_value
```

Canonical size: 3 bytes minimum (no timestamp, zero offset), up to 10 bytes
(seconds + minutes + hours with a 31-bit offset). `decode_timecode()` returns
`seconds_flag/minutes_flag/hours_flag = true` for the full-timestamp form
because those flags are not coded there. The writer rejects a value that does
not fit the selected `time_offset_length`, out-of-range clock fields,
`n_frames > 511`, and flags that cannot be represented (e.g. `minutes_flag`
without `seconds_flag`).

### `metadata_banding_hints` — type 6

```
f(1) coding_banding_present_flag
f(1) source_banding_present_flag
if (coding_banding_present_flag) {
f(1)   banding_hints_flag
       if (banding_hints_flag) {
f(1)     three_color_components_flag
         numComponents = three_color_components_flag ? 3 : 1
         for (plane = 0; plane < numComponents; plane++) {
f(1)       banding_in_component_present_flag
           if (banding_in_component_present_flag) {
f(6)         max_band_width_minus_4
f(4)         max_band_step_minus_1
           }
         }
f(1)     band_units_information_present_flag
         if (band_units_information_present_flag) {
f(5)       num_band_units_rows_minus_1      // 0..31 => 1..32 rows
f(5)       num_band_units_cols_minus_1      // 0..31 => 1..32 cols
f(1)       varying_size_band_units_flag
           if (varying_size_band_units_flag) {
f(3)         band_block_in_luma_samples     // block size = 16 << value
             for (r = 0; r <= rows_minus_1; r++)
f(5)           vert_size_in_band_blocks_minus_1[r]
             for (c = 0; c <= cols_minus_1; c++)
f(5)           horz_size_in_band_blocks_minus_1[c]
           }
           for (r ...) for (c ...)
f(1)         banding_in_band_unit_present[r][c]
         }
       }
}
```

Canonical size 1 byte minimum (two flags) up to 175 bytes (1396 syntax bits:
3 components + full 32x32 varying-size grid). The writer requires absent syntax
to stay at its default value, so a value -> unit -> value round-trip is
lossless; it also checks array lengths against the signalled grid and that
matrix entries are 0/1.

### `metadata_icc_profile` — type 7

```
le(metadataPayloadSize) icc_profile_data_payload_bytes
```

The unit is exactly the ICC profile bytes, pass-through. §6.16.9 reads
`ICCmajorVer = bytes[8]` and `ICCminorVer = bytes[9] >> 4`;
`icc_profile_version()` implements that and returns `nullopt` for fewer than 10
bytes. Profiles shorter than 10 bytes warn (or error under `strict`).

### `metadata_scan_type` — type 8

```
f(5) mps_pic_struct_type          // 0..12 defined, 13..31 reserved
f(2) mps_source_scan_type_idc     // same semantics as ci_scan_type_idc
f(1) mps_duplicate_flag
```

Exactly 1 byte. `scan_type_required_ci_scan_type_idc()` maps Table 6.18
(picture structure -> required `ci_scan_type_idc`: progressive 1, field 2,
complementary field pair 3, frame doubling/tripling 1).

### `metadata_temporal_point_info` — type 9

```
leb128() frame_presentation_time    // 32-bit value
```

1..5 bytes. **Short-OBU only** (§6.16.11): `make_group_obu()` refuses it and
`scan_metadata()` counts a group-carried unit as invalid.

### `metadata_decoded_frame_hash` — type 5

```
f(4)  hash_type           // 0 = MD5, 1..15 reserved
f(1)  per_plane
f(1)  has_grain
f(1)  is_monochrome
f(1)  reserved            // must be 0
if (per_plane) {
  numPlanes = is_monochrome ? 1 : 3
  for (i = 0; i < numPlanes; i++)
le(16)  plane_hash[i]
} else {
le(16)  frame_hash
}
```

17 bytes (whole frame or per-plane monochrome) or 49 bytes (per-plane Y/U/V).
The reader rejects a set `reserved` bit. The writer rejects non-zero hashes in
fields that are not coded (e.g. `frame_hash` when `per_plane == 1`).

### `metadata_user_data_unregistered` — type 10

```
f(128) uuid_iso_iec_11578
for (i = 16; i < metadataPayloadSize; i++)
f(8)   user_data_payload_byte
```

Minimum 16 bytes. UUID and payload are opaque; pass-through.

## OBU framing and flags

The module reuses `hdr::make_metadata_short_obu()`,
`hdr::make_metadata_group_obu()`, `hdr::parse_obu()`,
`hdr::parse_metadata_short()`, `hdr::parse_metadata_group()` and
`hdr::insert_metadata_obus()`.

### Short form (§5.17.2)

```
f(1) metadata_is_suffix
f(3) muh_layer_idc            // must be < 3
f(1) muh_cancel_flag
f(3) muh_persistence_idc
leb128() metadata_type
metadata_unit(obuPayloadSize - 2 - Leb128Bytes)
```

`metadata_necessity_idc`, `metadata_application_id` and `muh_priority` are
implicitly 0. `MetadataObuConfig::necessity/application` are therefore ignored
by `make_short_obu()` (they are group-only), while `suffix`, `layer`,
`persistence` and `cancel` apply. `cancel` must use
`make_short_cancel_obu()`; the value writer rejects `config.cancel == true`.

### Group form (§5.17.3)

```
f(1) metadata_is_suffix
f(2) metadata_necessity_idc
f(5) metadata_application_id
leb128() metadata_unit_cnt_minus_1
for each unit:
  leb128() metadata_type
  f(7) muh_header_size
  f(1) muh_cancel_flag
  if (!cancel) {
    leb128() muh_payload_size
    f(3) muh_layer_idc
    f(3) muh_persistence_idc
    f(8) muh_priority
    f(2) muh_reserved_zero_2bits     // 0
  }
  muh_header_extension_byte[]
  if (!cancel) metadata_unit(muh_payload_size)
```

`GroupUnit` carries per-unit `layer`, `persistence`, `priority` and `cancel`.
`make_group_obu()` rejects `LAYER_VALUES` (no explicit layer maps in this
writer), reserved persistence 4..7, reserved application ids 6..15, empty
groups, non-canonical payloads, and group-carried temporal point info. Cancel
entries must not carry payload bytes.

The encoder emits `muh_header_size = 2 + leb128_size(payload)` for a normal
unit and `0` for a cancel unit, matching the HDR writer's pinned group goldens.

## Pinned bit goldens

Golden values (identical to `metadata_types_test.cpp` and the gate tool):

| Type | Value | Canonical unit bytes |
|---|---|---|
| ITU-T T.35 | cc `0xB5`, payload `00 3C 00 01 04` | `b5 00 3c 00 01 04` |
| ITU-T T.35 (extension) | cc `0xFF`, ext `00`, payload `31 32` | `ff 00 31 32` |
| timecode | counting 0, full, `n_frames=42`, 01:03:07, offset 0 | `04 15 0e 18 40` |
| banding hints | 1 component, 2x3 varying grid | `a9 87 08 a4 01 00 45 58` |
| ICC profile | `ADBE` header, version 4.3 | `00 00 01 20 41 44 42 45 04 30 00 00 6d 6e 74 72` |
| scan type | top field, source `idc=2`, no duplicate | `0c` |
| temporal point info | `frame_presentation_time = 1000` | `e8 07` |
| decoded frame hash | MD5 header, per-plane, hashes `00..0f/10..1f/20..2f` | `08` + 48 bytes |
| user data unregistered | UUID `00..0f`, payload `de ad be ef` | `00..0f de ad be ef` |

Complete short OBUs (prefix, `muh_layer_idc = LAYER_CURRENT`,
`muh_persistence_idc = GLOBAL_PERSISTENCE`, type 8, Annex B.2 leb128 size).
`..` abbreviates the hash/payload runs shown in the table above; the exact
strings are asserted in `metadata_types_test.cpp`.

```
itut_t35   0a 20 20 03 b5 00 3c 00 01 04 80
timecode   09 20 20 04 04 15 0e 18 40 80
frame_hash 35 20 20 05 08 00 01 .. 2f 80
banding    0c 20 20 06 a9 87 08 a4 01 00 45 58 80
icc        14 20 20 07 00 00 01 20 41 44 42 45 04 30 00 00 6d 6e 74 72 80
scan_type  05 20 20 08 0c 80
temporal   06 20 20 09 e8 07 80
user_data  18 20 20 0a 00 01 .. 0f de ad be ef 80
```

Suffix ITU-T T.35 (`metadata_is_suffix = 1`): `0a 20 a0 03 b5 00 3c 00 01 04 80`.
Cancel ITU-T T.35 short OBU: `04 20 28 03 80`.

One group OBU with the seven group-eligible types (advisory necessity,
application 0, layer current, persistence global, hash priority 1) is 146 bytes
(`..` abbreviates the hash/payload runs; the exact string is asserted in the
test):

```
90 01 24 40 06
  03 06 06 40 00 b5 00 3c 00 01 04
  04 06 05 40 00 04 15 0e 18 40
  05 06 31 40 04 08 00 01 .. 2f
  06 06 08 40 00 a9 87 08 a4 01 00 45 58
  07 06 10 40 00 00 00 01 20 41 44 42 45 04 30 00 00 6d 6e 74 72
  08 06 01 40 00 0c
  0a 06 14 40 00 00 01 .. 0f de ad be ef
80
```

Suffix flips the group header byte `0x40` to `0xc0`. All-cancel group:
`12 24 40 06 03 01 04 01 05 01 06 01 07 01 08 01 0a 01 80`.

## API

```cpp
namespace av2enc::metadata {

using Type = hdr::MetadataType;   // CLL=1, MDCV=2, ..., UDU=10

struct ItutT35 { uint8_t country_code, country_code_extension_byte;
                 std::vector<uint8_t> payload; };
struct Timecode { /* counting/full/disc/drop/n_frames, s/m/h flags+values,
                     time_offset_length/value */ };
struct BandingHints { /* flags, 3 component hints, band-unit grid */ };
struct IccProfile { std::vector<uint8_t> data; };
struct ScanType { uint8_t mps_pic_struct_type, mps_source_scan_type_idc;
                  bool mps_duplicate_flag; };
struct TemporalPointInfo { uint32_t frame_presentation_time; };
struct DecodedFrameHash { uint8_t hash_type; bool per_plane, has_grain,
                          is_monochrome; array<FrameHash,3> plane_hash;
                          FrameHash frame_hash; };
struct UserDataUnregistered { array<uint8_t,16> uuid;
                              std::vector<uint8_t> payload; };

using Value = std::variant<hdr::ContentLightLevel,
                           hdr::MasteringDisplayColorVolume,
                           ItutT35, Timecode, BandingHints, IccProfile,
                           ScanType, TemporalPointInfo, DecodedFrameHash,
                           UserDataUnregistered>;

// Validation (reuses hdr::ValidationResult).
hdr::ValidationResult validate_<type>(const T&, bool strict = false);

// Exact sizes.
size_t itut_t35_unit_size(const ItutT35&);
size_t timecode_syntax_bits(const Timecode&);
size_t timecode_unit_size(const Timecode&);
size_t banding_hints_syntax_bits(const BandingHints&);
size_t banding_hints_unit_size(const BandingHints&);
size_t icc_profile_unit_size(const IccProfile&);
size_t temporal_point_info_unit_size(const TemporalPointInfo&);
size_t decoded_frame_hash_unit_size(const DecodedFrameHash&);
size_t user_data_unregistered_unit_size(const UserDataUnregistered&);

// Canonical serialization.
std::vector<uint8_t> serialize_<type>_unit(const T&);

// Validated value -> {type, canonical bytes}.
struct MetadataUnitPayload { Type type; std::vector<uint8_t> bytes; };
std::optional<MetadataUnitPayload> make_<type>_unit(const T&,
                                                    std::string& error,
                                                    bool strict = false);

// Framing (config: suffix/layer/persistence/strict_ranges; group also
// necessity/application).
std::vector<uint8_t> make_short_obu(const MetadataUnitPayload&,
                                    const hdr::MetadataObuConfig&,
                                    std::string& error);
std::vector<uint8_t> make_short_cancel_obu(Type,
                                           const hdr::MetadataObuConfig&,
                                           std::string& error);
struct GroupUnit { MetadataUnitPayload payload;
                   hdr::MetadataLayer layer;
                   hdr::MetadataPersistence persistence;
                   uint8_t priority; bool cancel; };
std::vector<uint8_t> make_group_obu(std::span<const GroupUnit>,
                                    const hdr::MetadataObuConfig&,
                                    std::string& error);

// Readers. Extra trailing bytes are accepted as §5.17.1 padding.
bool decode_<type>(std::span<const uint8_t>, T&, std::string& error);
bool decode_unit(Type, std::span<const uint8_t>, Value&, std::string& error);

// Whole temporal unit: records + counts of supported/unsupported/invalid/
// cancelled units. Supports CLL/MDCV through the HDR module.
bool scan_metadata(std::span<const uint8_t>, ScanResult&, std::string& error);

// Helpers.
bool is_known_type(Type);
bool is_group_allowed(Type);
const char* type_name(Type);
std::optional<IccProfileVersion> icc_profile_version(const IccProfile&);
std::optional<uint8_t> scan_type_required_ci_scan_type_idc(uint8_t);
}
```

## Validation summary

| Check | Severity |
|---|---|
| ITU-T T.35 empty payload (§6.16.4) | always error |
| ITU-T T.35 extension byte with `country_code != 0xFF` | always error |
| timecode clock ranges (`seconds/minutes <= 59`, `hours <= 23`) | always error |
| `n_frames > 511`, `time_offset_length > 31`, offset does not fit | always error |
| timecode presence flags that cannot be coded | always error |
| timecode reserved `counting_type` 7..31 | warning / error under strict |
| banding fields set while their presence flag is 0 | always error |
| banding array length / matrix length mismatch, matrix entry > 1 | always error |
| banding 5-bit/6-bit/4-bit/3-bit field overflow | always error |
| ICC profile shorter than 10 bytes | warning / error under strict |
| scan type `mps_pic_struct_type > 12` (§6.16.10) | always error |
| decoded frame hash `reserved != 0` | always error (reader too) |
| decoded frame hash `hash_type != 0` | warning / error under strict |
| decoded frame hash non-zero uncoded hash fields | always error |
| group-carried temporal point info (§6.16.11) | always error |

`MetadataObuConfig::strict_ranges` is honoured by both `make_*_unit()` (when
passed as `strict`) and the framing writers (which re-validate the payload).

## Tests

```sh
src/encoder/metadata/types/build.sh
./build-metadata-types/metadata_types_test
```

17 test groups cover:

- pinned unit and OBU goldens for all eight types, suffix/cancel/group goldens;
- exact payload sizes, including the 23/73-bit timecode and 2/1396-bit banding
  extremes and every `leb128` boundary;
- value -> unit -> value and unit -> value -> unit round-trips;
- a 8 type x 3 layer x 4 persistence x 2 placement framing matrix, plus group
  layer/persistence/priority 0/128/255, all necessity ids and application ids;
- `metadata_unit_remaining_bit` acceptance with rejection at the writer;
- malformed input: empty/truncated units at every prefix, missing T.35
  extension byte, truncated/overflowing leb128, hash reserved bit, group-carried
  temporal point info;
- every validation rule above, strict vs. non-strict;
- group framing with HDR CLL/MDCV in the same OBU;
- scanner ordering, placement, values, cancel records, unsupported and invalid
  counts.

Clean under `-fsanitize=address,undefined`.

## Decoder gate

```sh
src/encoder/metadata/types/run_metadata_gate.sh
src/encoder/metadata/types/run_metadata_gate.sh --ivf /tmp/existing_m2.ivf
```

The gate:

1. builds the module, unit test and tool;
2. builds a valid 64x64 two-frame M2 stream with
   `src/encoder/m2/build_and_test.sh` (or uses `--ivf`);
3. appends each type separately as prefix metadata (frame 0) and suffix
   metadata (frame 1), rescanning to check type, placement, flags and decoded
   values;
4. appends all eight types at once (prefix and suffix), a group OBU (prefix and
   suffix), a global/basic-persistence variant, cancel short units and a
   cancel group;
5. verifies the IVF file header and all frame timestamps are unchanged and only
   the selected frame grew;
6. requires `dav2d --strict 1 --threads 1` to decode every variant to the
   **same Y4M MD5** as the baseline;
7. checks that a group-carried temporal point info unit is refused, and that
   flipping a metadata unfinished trailing-bits byte makes the self-parser
   reject the stream.

Exit codes: 0 pass, 1 fail, 2 usage, 77 skip (no decoder / no M2 stream).

### Pinned decoder behaviour (f9dafbc, 2026-11-10)

The pinned `dav2d` does not implement §5.17.2/§5.17.3. It parses OBU type 8 as
a pre-1.0 AV1-style metadata OBU, reading the first byte after the OBU header as
`metadata_type`. For a spec short OBU that first byte is the AV2 options byte
(`0x20` for prefix/current/global), so the decoder logs messages such as:

```
Unknown Metadata OBU type 32
Unknown Metadata OBU type 416     (suffix options byte 0xA0 continues into type)
```

and ignores the OBU. It has no case for OBU type 9 and logs:

```
Unknown OBU type 9 of size 143
```

The same is true for group and cancel variants. Decoding exits 0 and every
variant produces the baseline Y4M MD5
`2da6c81f1b4a0333fdd742d7291af895` for the synthetic M2 clip. The gate
therefore passes on **decode-invariance plus self-parse**, exactly as required.
Do not change the writers to match this decoder.

One interaction worth knowing: because the pinned decoder treats the options
byte as `metadata_type`, `muh_layer_idc = LAYER_UNSPECIFIED` with
`muh_persistence_idc = BASIC_PERSISTENCE` yields options byte `0x01`, which the
legacy parser mistakes for CLL. The gate uses `LAYER_CURRENT`/`LAYER_GLOBAL`, so
that collision does not occur. Conforming AV2 decoders read the byte as options.

## Integration sites

The module is not wired into the CLI. To emit metadata from the encoder:

- **M2 key frame** — `src/encoder/m2/video_frame.cpp`,
  `encode_video_sequence()`: build prefix OBUs with `make_short_obu()` /
  `make_group_obu()` and insert them between `seq` and `obu`, and suffix OBUs
  after `obu`, in the `key_unit` block (currently lines ~1005-1016). The P-frame
  loop (lines ~1035-1050) has no sequence header in its temporal unit, so only
  suffix or `GLOBAL_XLAYER_ID` (§7.3.7) prefix metadata applies there.
- **M1 still** — `src/encoder/m1/still_headers.cpp`,
  `assemble_still_picture()`: insert prefix metadata between `seq` and `clk`
  (lines ~294-303), suffix after `clk`.
- **Placement helper** — for existing serialized units use
  `hdr::insert_metadata_obus(..., MetadataPlacement::kBeforeFirstFrame)` or
  `kEndOfTemporalUnit`; it validates `metadata_is_suffix` and requires a
  sequence header before the frame for prefix insertion.
- **Container** — metadata is part of the temporal unit, so the IVF frame
  payload size changes; `IvfWriter::write_frame()` already takes a full payload.
- **CLI flags** (suggested, not implemented): `--metadata-timecode`,
  `--metadata-scan-type`, `--metadata-icc <file>`, `--metadata-hash`,
  `--metadata-user-data <uuid:hex>`, `--metadata-group`, `--metadata-suffix`.
  Metadata must not affect rate control or reconstruction.
- **CMake snippet** (not applied; owner of `CMakeLists.txt`):

  ```cmake
  target_sources(av2enc_core PRIVATE
    src/encoder/metadata/types/metadata_types.cpp)
  # tests:
  # add_executable(metadata_types_test
  #   src/encoder/metadata/types/metadata_types_test.cpp)
  # target_link_libraries(metadata_types_test PRIVATE av2enc_core)
  # add_test(NAME metadata_types_test COMMAND metadata_types_test)
  # add_test(NAME metadata_feature_gate COMMAND bash
  #   src/encoder/metadata/types/run_metadata_gate.sh)
  ```

## Known limits

- No CMake target; `build.sh` is the build entry point until the owner wires it
  up. `build.sh` compiles `hdr_metadata.cpp` too, because it reuses the HDR OBU
  layer.
- `LAYER_VALUES` group writers are not implemented (the reader and HDR parser
  accept them for stream analysis).
- Group-carried temporal point info cannot be written (normative) and is
  reported as invalid by the scanner.
- The decoded-frame-hash unit only carries the 16-byte MD5 digests; computing
  them from output frames is the caller's job (§6.16.13 describes the byte
  conversion).
- The pinned `dav2d` does not parse any of these units; the gate relies on
  decode-invariance and the module's own scanner.
- ICC version interpretation is informational only; the profile bytes are
  passed through unchanged.

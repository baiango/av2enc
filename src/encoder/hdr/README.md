# HDR metadata OBUs (`src/encoder/hdr/`)

AV2 v1.0.0 high dynamic range metadata: **content light level** (CLL,
§5.17.5) and **mastering display colour volume** (MDCV, §5.17.6), carried in
`OBU_METADATA_SHORT` (type 8) or `OBU_METADATA_GROUP` (type 9).

This directory is self-contained. It does **not** edit the M1/M2 core,
`src/main.cpp`, or `CMakeLists.txt`.

Normative reference: `spec/AV2_Spec_v1.0.0.pdf`.

| Syntax | Section |
|---|---|
| `obu_header()` (extension flag, type, temporal layer) | §5.2.2 |
| `metadata_unit(metadataPayloadSize)` | §5.17.1 |
| `metadata_short_obu(obuPayloadSize)` | §5.17.2 |
| `metadata_group_obu()` | §5.17.3 |
| `metadata_hdr_cll()` | §5.17.5 |
| `metadata_hdr_mdcv()` | §5.17.6 |
| OBU type numbers 8 / 9 | §6.2.2 Table 6.1 |
| `metadata_type` 1 / 2 | §6.16.3 Table 6.17 |
| CLL / MDCV semantics and value ranges | §6.16.5 / §6.16.6 |
| Coded output frame unit ordering (prefix/suffix metadata) | §7.3.3 |
| Temporal unit ordering | §7.3.7 |
| Coded extended layer unit (sequence header before frame) | §7.3.6 |

CLL and MDCV are the only HDR metadata units in v1.0.0. Colour description
(primaries / transfer / matrix) is not a `metadata_unit()`; it lives in the
LCR/OPS colour info and is out of scope. Dynamic HDR (HDR10+) would ride in
`METADATA_TYPE_ITUT_T35` (type 3) and is not implemented.

## Files

| File | Purpose |
|---|---|
| `hdr_metadata.h` / `hdr_metadata.cpp` | writers, readers, validation, ordering |
| `hdr_metadata_test.cpp` | unit tests + pinned bit goldens |
| `append_hdr_metadata.cpp` | IVF tool: insert metadata / scan a stream |
| `build.sh` | standalone build (no CMake target) |
| `run_hdr_gate.sh` | encoder + dav2d decoder gate |

## Wiring (OBU syntax)

Annex B.2 framing is used, same as the M1 temporal unit:

```
leb128(obu_size) obu_header() payload
```

`obu_size` counts the OBU header plus payload, not the leb128 prefix. The module
reuses `av2enc::m0::make_obu()` / `write_obu_header()` so there is one AV2 OBU
header implementation.

### CLL (`metadata_hdr_cll`)

```
f(16) max_cll
f(16) max_fall
```

4 bytes. `max_cll` is the upper bound on maximum sample light level, `max_fall`
on the frame average, both in cd/m² in the 4:4:4 linear-light representation
(§6.16.5). `0` means "not signalled"; every other 16-bit value is valid.

### MDCV (`metadata_hdr_mdcv`)

```
for i = 0..2:
  f(16) primary_chromaticity_x[i]
  f(16) primary_chromaticity_y[i]
f(16) white_point_chromaticity_x
f(16) white_point_chromaticity_y
f(32) luminance_max
f(32) luminance_min
```

24 bytes, big-endian. Units are SMPTE ST 2086 integers: chromaticity in
`0.00002`, luminance in `0.0001 cd/m²`. Index order is suggested as
green / blue / red (§6.16.6), **not** R/G/B.

### `metadata_short_obu` (§5.17.2)

```
f(1) metadata_is_suffix        // 0 = prefix, 1 = suffix
f(3) muh_layer_idc             // must be < 3
f(1) muh_cancel_flag
f(3) muh_persistence_idc
leb128() metadata_type
metadata_unit(obuPayloadSize - 2 - Leb128Bytes)
byte 0x80                      // trailing_bits
```

`metadata_necessity_idc`, `metadata_application_id` and `muh_priority` are
implicitly 0 in the short form.

### `metadata_group_obu` (§5.17.3)

```
f(1) metadata_is_suffix
f(2) metadata_necessity_idc
f(5) metadata_application_id
leb128() metadata_unit_cnt_minus_1       // < 16383
for each unit:
  leb128() metadata_type
  f(7) muh_header_size                   // excludes the cancel flag
  f(1) muh_cancel_flag
  if (!cancel) {
    leb128() muh_payload_size
    f(3) muh_layer_idc
    f(3) muh_persistence_idc
    f(8) muh_priority
    f(2) muh_reserved_zero_2bits         // must be 0
    if (muh_layer_idc == 3) { ... layer maps ... }
  }
  muh_header_extension_byte[]            // header_size remainder
  if (!cancel) metadata_unit(muh_payload_size)
byte 0x80                                // trailing_bits
```

The writer emits `muh_header_size = 2 + leb128_size(payload_size)` for a normal
unit and `0` for a cancel unit. It rejects `muh_layer_idc == LAYER_VALUES`
because explicit layer maps are not implemented.

## Pinned bit goldens

Sample: BT.2020 + D65, 1000 cd/m², 0.005 cd/m²:
`max_cll=1000`, `max_fall=400`, primaries `G(8500,39850) B(6550,2300)
R(35400,14600)`, white `(15635,16450)`, `luminance_max=10000000`,
`luminance_min=50`.

CLL short OBU (prefix, layer `LAYER_CURRENT`, persistence `GLOBAL`):

```
08 20 20 01 03 e8 01 90 80
```

MDCV short OBU:

```
1c 20 20 02
   21 34 9b aa  19 96 08 fc  8a 48 39 08       # G, B, R x/y
   3d 13 40 42                                  # white x/y
   00 98 96 80                                  # luminance_max
   00 00 00 32                                  # luminance_min
   80
```

Group OBU carrying both units (advisory, application 0, layer current,
persistence global, priority 0):

```
2a 24 40 01
   01 06 04 40 00 03 e8 01 90
   02 06 18 40 00 21 34 9b aa 19 96 08 fc 8a 48 39 08
                  3d 13 40 42 00 98 96 80 00 00 00 32
   80
```

Suffix CLL (flip `metadata_is_suffix`): `08 20 a0 01 03 e8 01 90 80`.
Cancel MDCV short OBU: `04 20 28 02 80`.

These strings are asserted in `hdr_metadata_test.cpp`.

## API

```cpp
namespace av2enc::hdr {

struct ContentLightLevel { uint16_t max_cll, max_fall; };
struct MasteringDisplayColorVolume { /* 3x x/y, white x/y, max, min */ };
struct HdrMetadata { bool has_cll; ContentLightLevel cll;
                     bool has_mdcv; MasteringDisplayColorVolume mdcv; };

// Serialize the canonical §5.17.5/§5.17.6 unit payloads (4 / 24 bytes).
std::vector<uint8_t> serialize_cll_unit(const ContentLightLevel&);
std::vector<uint8_t> serialize_mdcv_unit(const MasteringDisplayColorVolume&);

// Writers (return a complete leb128-prefixed OBU).
std::vector<uint8_t> make_cll_short_obu(...);
std::vector<uint8_t> make_mdcv_short_obu(...);
std::vector<uint8_t> make_cancel_short_obu(...);
std::vector<uint8_t> make_metadata_group_obu(...);
std::vector<uint8_t> make_hdr_metadata_obus(const HdrMetadata&,
                                            HdrObuPacking,
                                            const MetadataObuConfig&,
                                            std::string& error);

// Readers.
bool parse_obu(...);              // one Annex B.2 OBU
bool parse_obu_sequence(...);     // a whole temporal unit
bool parse_metadata_short(...);
bool parse_metadata_group(...);
bool decode_cll(...);
bool decode_mdcv(...);

// Stream analysis: every CLL/MDCV unit, plus counts of unknown/invalid units.
bool scan_hdr_metadata(std::span<const uint8_t> temporal_unit,
                       HdrScanResult&, std::string& error);

// Ordering: insert before the first frame OBU (prefix) or at the end (suffix).
bool insert_metadata_obus(std::span<const uint8_t> temporal_unit,
                          std::span<const uint8_t> metadata_obus,
                          MetadataPlacement, std::vector<uint8_t>& out,
                          std::string& error, size_t* inserted_at = nullptr);
}
```

`MetadataObuConfig` controls `suffix`, `layer`, `persistence`, `cancel`,
`strict_ranges`, and the group-only `necessity` / `application`.

## Validation ranges

`validate_mdcv()` implements §6.16.6:

| Field | Valid range |
|---|---|
| `primary_chromaticity_x[i]`, `white_point_chromaticity_x` | 5..37000 |
| `primary_chromaticity_y[i]`, `white_point_chromaticity_y` | 5..42000 |
| `luminance_max` | 50000..100000000 |
| `luminance_min` | 1..50000 |

Values outside a range are **not** conformance errors: §6.16.6 defines them as
"unknown / unspecified". They are spec-legal and the writer encodes them
unchanged (it never clamps). By default they produce warnings; with
`strict_ranges = true` they become errors so an encoder can refuse to ship a
placeholder. The one normative constraint — `luminance_max == 50000` requires
`luminance_min != 50000` — is always rejected.

CLL has no reserved values (`validate_cll()` never fails). `0` is the "not
signalled" value.

## Ordering

`insert_metadata_obus()` places prefix metadata **immediately before the first
frame OBU**. For the M1 stream (TD, SEQ, CLK) that is after the sequence header
and before `OBU_CLOSED_LOOP_KEY`, exactly the §7.3.3 position inside the coded
output frame unit. It also works when a multi-frame header / content
interpretation OBU precedes the frame. Suffix metadata is appended after the
last OBU (§7.3.3 suffix position). The helper rejects a `metadata_is_suffix`
value that contradicts the requested placement, and refuses a prefix insert
when no sequence header precedes the frame.

`OBU_METADATA_*` with `obu_xlayer_id == GLOBAL_XLAYER_ID` (extension flag set,
xlayer 31) may also sit in the temporal unit ahead of the coded extended layer
unit (§7.3.7). That form is **not** emitted here; with the extension flag clear
the inferred xlayer is 0, so the metadata belongs inside the coded frame unit.

## Tests

```sh
src/encoder/hdr/build.sh
./build-hdr/hdr_metadata_test
```

Covers:

- pinned bit goldens for CLL, MDCV, group, suffix and cancel OBUs;
- writer → stream → scanner round-trips, including extreme values;
- every §6.16.6 range, strict vs. non-strict severity, and the normative
  `luminance_max == 50000` constraint;
- multiple OBUs in one temporal unit (two short OBUs, group with CLL + unknown
  user data, cancel units);
- ordering and mismatch rejection (prefix/suffix, missing sequence header,
  non-metadata insertion);
- truncated / overflowing leb128, bad OBU sizes, bad trailing bits, reserved
  bits, short CLL/MDCV units;
- metadata payload independence from bit depth (see below).

The suite is clean under `-fsanitize=address,undefined`.

### 8/10-bit note

AV2 `bit_depth_idc` is inverted relative to intuition: `0` = 10-bit, `1` =
8-bit (§6.4.1 Table 6.3). CLL/MDCV contain only fixed integer fields and no BitDepth-
dependent syntax, so the same units produce byte-identical OBUs for 8- and
10-bit streams. The M1 encoder is 8-bit only today; the test asserts the module
is format-agnostic by inserting the same metadata into two streams whose
sequence-header payloads differ.

## Decoder gate

```sh
src/encoder/hdr/run_hdr_gate.sh --bin build-m1/av2enc
src/encoder/hdr/run_hdr_gate.sh --bin build-m1/av2enc \
    --fixture tests/data/m0_640x360_1f.y4m
```

The gate:

1. builds the HDR unit test and tool;
2. encodes a baseline M1 still picture and records the decoded Y4M MD5;
3. appends CLL+MDCV as two short OBUs, as one group OBU, as suffix metadata,
   and MDCV-only; rescans each stream and checks the exact metadata counts,
   values and placement;
4. requires `dav2d --strict 1` to decode every variant to the **same MD5** as
   the baseline, with the IVF file header and frame timestamp unchanged;
5. checks that out-of-range ("unknown") MDCV values decode and are reported as
   zero, and that `--strict-ranges` rejects them at write time;
6. probes the decoder with the legacy layout and requires a corrupted
   `trailing_bits` byte to be rejected.

Exit codes: 0 pass, 1 fail, 2 usage, 77 skip (no decoder / fixture).

### Pinned dav2d compatibility finding (f9dafbc, 2026-11-10)

The pinned decoder parses `obu_type == 8` as a **pre-1.0 AV1-style** metadata
OBU: `leb128(metadata_type)` immediately after the OBU header, with no
`metadata_is_suffix` / `muh_layer_idc` / `muh_cancel_flag` /
`muh_persistence_idc` byte. It has no case for `obu_type == 9`.

Consequences observed by the gate:

| Stream | dav2d result |
|---|---|
| spec short CLL/MDCV | decodes identically; logs `Unknown Metadata OBU type 32` and ignores the OBUs |
| spec group (type 9) | decodes identically; logs `Unknown OBU type 9` |
| spec suffix | decodes identically; ignores the OBUs |
| legacy AV1-style layout | decodes identically, no warning (metadata parsed) |
| legacy with corrupted trailing bits | rejected under `--strict 1` |

So the spec-conformant metadata is safely **ignored** by this decoder snapshot,
which is the required gate outcome: appending it does not change or break
decode. The legacy helpers exist only to prove the validator actively parses
HDR metadata when it understands the layout; they are **not** AV2 v1.0.0 and
must never be shipped.

Do not "fix" the spec writer to match the pinned decoder. When dav2d picks up
§5.17.2/§5.17.3, the same streams will be parsed and reported with no encoder
change.

## CLI / integration notes

- The tool does not encode. To emit metadata from the encoder, call
  `make_hdr_metadata_obus()` while assembling the temporal unit and insert the
  result with `insert_metadata_obus(..., MetadataPlacement::kBeforeFirstFrame)`
  before the CLK OBU. In `m1::assemble_m1_still_picture()` that is: build
  `td`, `seq`, `metadata`, then `clk` (owner of `m1_headers.cpp` must apply
  this; the file was not modified here).
- Container: the metadata is part of the temporal unit, so IVF frame payload
  size grows and must be written accordingly (`IvfWriter::write_frame` already
  takes a full payload).
- Suggested flags for `src/main.cpp` (not implemented here): `--hdr-cll
  max:fall`, `--hdr-mdcv <preset|spec>`, `--hdr-group`, `--hdr-suffix`.
  Metadata should not change rate control or reconstruction; the gate asserts
  bit-identical decoded output.
- CMake snippet for the owner of `CMakeLists.txt` (not applied):

  ```cmake
  target_sources(av2enc_core PRIVATE
    src/encoder/hdr/hdr_metadata.cpp)
  # tests: add_executable(hdr_metadata_test
  #   src/encoder/hdr/hdr_metadata_test.cpp) + link av2enc_core
  # add_test(NAME hdr_feature_gate COMMAND bash
  #   src/encoder/hdr/run_hdr_gate.sh --bin $<TARGET_FILE:av2enc>)
  ```

- Constants worth reusing: `kObuTypeMetadataShort` (8),
  `kObuTypeMetadataGroup` (9), `kMetadataTypeHdrCll`-equivalent enum values
  `MetadataType::kHdrCll` (1) / `kHdrMdcv` (2), `kCllUnitBytes` (4),
  `kMdcvUnitBytes` (24).

## Known limits

- `LAYER_VALUES` writers are not implemented (readers parse them for stream
  analysis). `muh_layer_idc` for writing is restricted to
  unspecified/global/current.
- Only CLL and MDCV are implemented; other metadata types can be carried as
  opaque `MetadataGroupEntry` payloads and are skipped by the scanner.
- The module has no CMake target and is not compiled into `av2enc` until its
  owner wires it up.
- `bitstream/obu.h::parse_header()` is still a stub. This module implements its
  own Annex B.2 OBU reader (`parse_obu`) from §5.2.2; it can be promoted to
  `bitstream/obu.*` later, but that file belongs to another owner and was not
  modified.

# Content interpretation OBU (`src/encoder/metadata/content_interpretation/`)

AV2 v1.0.0 content interpretation OBU (`OBU_CONTENT_INTERPRETATION`, type 24).
It carries the per-embedded-layer interpretation of the decoded video: scan
type (progressive / interlace), colour description, chroma sample positions,
sample aspect ratio and frame timing.

This directory is self-contained. It does **not** edit the M1/M2 core,
`src/encoder/hdr/`, `src/main.cpp`, or `CMakeLists.txt`.

Normative reference: `spec/AV2_Spec_v1.0.0.pdf` (commit `38f28e7`).

| Syntax / rule | Section |
|---|---|
| `open_bitstream_unit()`: extensible OBU payload, `obu_extension_flag`, `trailing_bits` | §5.2.1 |
| `obu_header()` | §5.2.2 |
| `trailing_bits()` | §5.2.3 |
| `timing_info()` syntax | §5.4.12 |
| `content_interpretation_obu()` + `Aspect_Ratio_Width/Height[17]` | §5.15 |
| `uvlc()` | §4.11.3 |
| `rg(n)` (used by `ci_color_description_idc`) | §4.11.10 |
| OBU type 24 | §6.2.2 Table 6.1 |
| `timing_info()` value ranges | §6.4.12 |
| CI semantics: scan type, colour, chroma positions, aspect ratio, per-CVS rule | §6.14 |
| Coded output frame unit: CI is the **first** OBU of the unit | §7.3.3 |
| Coded extended layer unit: sequence header precedes the frame unit | §7.3.6 |
| Temporal unit order | §7.3.7 |
| CI OBU availability (presence in the first TU of a CVS) | §7.3.8.10 |
| Content interpretation parameters initialization | §7.3.8.11 |

## Files

| File | Purpose |
|---|---|
| `content_interpretation.h` / `.cpp` | writer, reader, validation, ordering, per-CVS tracker |
| `content_interpretation_test.cpp` | unit tests + pinned bit goldens + invalid inputs |
| `append_content_interpretation.cpp` | IVF tool: insert a CI OBU at the §7.3.3 position or scan a stream |
| `build.sh` | standalone build (no CMake target) |
| `run_ci_gate.sh` | unit tests + encoder + pinned `dav2d` decoder gate |
| `README.md` | this document |

## Syntax implemented

```
content_interpretation_obu() {
  f(2)  ci_scan_type_idc                       // 0..3
  f(1)  ci_color_description_present_flag
  f(1)  ci_chroma_sample_position_present_flag
  f(1)  ci_aspect_ratio_info_present_flag
  f(1)  ci_timing_info_present_flag
  f(2)  ci_reserved_2bit                       // writer emits 0
  if (ci_color_description_present_flag) {
    rg(2) ci_color_description_idc             // 0..127
    if (ci_color_description_idc == 0) {
      f(8) ci_color_primaries
      f(8) ci_transfer_characteristics
      f(8) ci_matrix_coefficients
    }
    f(1) ci_full_range_flag
  }
  if (ci_chroma_sample_position_present_flag) {
    uvlc() ci_chroma_sample_position_top
    if (ci_scan_type_idc != 1) {
      uvlc() ci_chroma_sample_position_bottom
    } else {
      ci_chroma_sample_position_bottom = ci_chroma_sample_position_top
    }
  } else {
    top = bottom = CSP_UNSPECIFIED
  }
  if (ci_aspect_ratio_info_present_flag) {
    f(8) ci_aspect_ratio_idc                   // <= 16 or 255
    if (ci_aspect_ratio_idc == 255) {
      uvlc() ci_sar_width
      uvlc() ci_sar_height
    } else {
      ci_sar_width  = Aspect_Ratio_Width[idc]
      ci_sar_height = Aspect_Ratio_Height[idc]
    }
  }
  if (ci_timing_info_present_flag) {
    timing_info()
  }
}
```

`timing_info()` is §5.4.12: `f(32) num_units_in_display_tick`,
`f(32) time_scale`, `f(1) equal_picture_interval`, and, when equal,
`uvlc() num_ticks_per_picture_minus_1`.

`ci_reserved_2bit` is written as 0. §6.14 says the value "shall be ignored by
a decoder", so the reader records it but does not reject non-zero values.

The `Aspect_Ratio_*` tables are pinned in the header:

```
Aspect_Ratio_Width [17] = { 0, 1, 12, 10, 16, 40, 24, 20, 32, 80, 18, 15, 64, 160, 4, 3, 2 }
Aspect_Ratio_Height[17] = { 0, 1, 11, 11, 11, 33, 11, 11, 11, 33, 11, 11, 33, 99, 3, 2, 1 }
```

## API

```cpp
namespace av2enc::ci {

enum class ScanType { kUnspecified, kProgressive, kInterlaceField,
                      kInterlaceComplementaryPair };
enum class ChromaSamplePosition { kLeft, kCenter, kTopLeft, kTop,
                                  kBottomLeft, kBottom, kUnspecified };
enum class ChromaFormat { k420, k422 };

struct ChromaSampleOffset { bool applicable; double horizontal, vertical; };
ChromaSampleOffset chroma_sample_offset(ChromaSamplePosition, ChromaFormat);

struct ColorDescription { std::uint8_t idc, primaries, transfer, matrix;
                          bool full_range; };
ColorDescription derive_color_description(std::uint8_t idc);  // Table 6.13

struct AspectRatio { std::uint8_t idc; std::uint32_t sar_width, sar_height;
                     resolved_width(); resolved_height(); resolved(); };

struct TimingInfo { std::uint32_t num_units_in_display_tick, time_scale;
                    bool equal_picture_interval;
                    std::uint32_t num_ticks_per_picture_minus_1; };

struct ContentInterpretation { ScanType scan_type;
                               bool has_color_description; ColorDescription color;
                               bool has_chroma_sample_position;
                               ChromaSamplePosition chroma_top, chroma_bottom;
                               bool has_aspect_ratio; AspectRatio aspect;
                               bool has_timing_info; TimingInfo timing; };

ContentInterpretation default_content_interpretation();  // §7.3.8.11

ValidationResult validate_content_interpretation(const ContentInterpretation&,
                                                 ChromaFormat = k420,
                                                 bool strict = false);

std::vector<std::uint8_t> serialize_content_interpretation_payload(...);
std::vector<std::uint8_t> make_content_interpretation_obu(...);

struct ObuView { /* type, tlayer, extension, leb128 offsets, payload */ };
bool parse_obu(...);
bool parse_obu_sequence(...);
struct ContentInterpretationView { ContentInterpretation params;
                                   std::uint8_t reserved_bits;
                                   std::size_t syntax_bits;
                                   bool chroma_bottom_signalled; };
bool parse_content_interpretation(std::span<const std::uint8_t> payload,
                                  ContentInterpretationView&, std::string&,
                                  bool strict = true,
                                  ChromaFormat = k420);

struct ContentInterpretationScanResult { /* records, ci_obus, invalid_ci_obus,
                                             first_frame_offset, ... */ };
bool scan_content_interpretation(std::span<const std::uint8_t> temporal_unit,
                                 ContentInterpretationScanResult&, ...);

bool insert_content_interpretation(std::span<const std::uint8_t> temporal_unit,
                                   std::span<const std::uint8_t> ci_obu,
                                   std::vector<std::uint8_t>& out,
                                   std::string& error,
                                   std::size_t* inserted_at = nullptr);

class CvsConstancyTracker { bool observe(const CiObservation&, std::string&); ... };
ValidationResult check_cvs_constancy(std::span<const CiObservation>);

}
```

### Writer guarantees

* Validates before emitting. Hard conformance violations (chroma position 6
  when present, aspect `idc` in 17..254, zero timing values, ticks >
  2^32 − 2) are errors. Reserved codepoints and "field is present but ignored"
  mismatches are warnings unless `strict = true`.
* Canonical framing: `obu_extension_flag = 0`, `obu_tlayer_id = 0`, Annex B.2
  `leb128` size prefix, §5.2.1 extensible-OBU trailing bits.
* Progressive scan codes only `ci_chroma_sample_position_top`; a supplied
  bottom of `CSP_UNSPECIFIED` is derived from top, a conflicting bottom is an
  error.

### Reader guarantees

* Rejects truncated payloads, non-zero `obu_extension_flag`, missing
  trailing 1 bit, and non-zero padding bits.
* `strict = true` also enforces the hard §6.14 / §6.4.12 limits. Reserved
  colour codepoints stay warnings because §6.14 requires decoders to ignore
  them.
* `scan_content_interpretation()` reports every CI OBU (valid or not) with its
  byte position and whether it precedes the first frame OBU.

## Validation rules

| Rule | Section | Severity |
|---|---|---|
| `ci_scan_type_idc` in 0..3 | §5.15 `f(2)` | always error if out of range |
| colour `idc` in 0..127 | §5.15 `rg(2)` | always error if out of range |
| colour `idc` 6..127 reserved for AOMedia | Table 6.13 | warning (error with `strict`) |
| explicit primaries/transfer/matrix reserved codepoints | Tables 6.14/6.15 | warning (error with `strict`) |
| chroma position <= 5 when present | §6.14 | always error |
| 4:2:2 cannot use CSP_TOPLEFT/CSP_TOP/CSP_BOTTOMLEFT/CSP_BOTTOM | §6.14 table | warning (error with `strict`) |
| progressive codes one chroma position; bottom must equal top | §5.15 | error if conflicting |
| aspect `idc` <= 16 or 255 | §6.14 | always error |
| explicit SAR 0:0 or one zero dimension | §6.14 | warning (ratio unspecified) |
| table value used when `idc != 255` | §5.15 | warning if explicit values disagree |
| `num_units_in_display_tick > 0`, `time_scale > 0` | §6.4.12 | always error |
| `num_ticks_per_picture_minus_1 <= 2^32 - 2` | §6.4.12 | always error |
| ticks coded only when `equal_picture_interval` | §5.4.12 | warning when non-zero and absent |

### Per-CVS constancy (§6.14, §7.3.8.10, §7.3.8.11)

`CvsConstancyTracker` (and the bulk `check_cvs_constancy`) enforces:

1. all CI instances for an embedded layer in a CVS carry identical
   parameters;
2. if a CI OBU appears in any temporal unit, one also appears in the first
   temporal unit of the CVS for that layer, with identical contents;
3. a CI OBU in a non-random-access temporal unit matches the parameters
   established at the most recent CLK/OLK;
4. `timing_info()` is identical across all embedded layers in the CVS;
5. different layers may otherwise carry different parameters (e.g. different
   scan types).

Layer state is keyed by `mlayer`; the tracker is reset at each new CVS.

## Pinned bit goldens

The unit test pins nine payloads/OBUs computed independently from the syntax.
Examples:

| Case | Payload | OBU (`leb128 header payload`) |
|---|---|---|
| all presence flags 0 | `00 40` | `03 60 00 40` |
| progressive / BT.709 / chroma center / 1:1 / 1-30 timing | `7c 24 02 00 00 00 02 00 00 00 3d a0` | `0d 60 7c 24 02 00 00 00 02 00 00 00 3d a0` |
| interlace field / explicit BT.2020 PQ / chroma 0,5 / SAR 16:11 | `b8 01 22 01 39 bf c2 23 10` | `0a 60 b8 01 22 01 39 bf c2 23 10` |
| complementary pair / chroma TOPLEFT,TOP / 4:3 | `d8 64 0e 40` | `05 60 d8 64 0e 40` |
| timing only (1001/30000) | `04 00 00 03 e9 00 00 75 30 20` | `0b 60 04 00 00 03 e9 00 00 75 30 20` |
| sYCC (`rg(2)` q=1) / SAR 160:99 / ticks 1 | `7c 92 ff 01 42 06 40 00 00 00 10 00 00 03 ca 40` | `11 60 7c 92 ff 01 42 06 40 00 00 00 10 00 00 03 ca 40` |

The full table is in `kGoldens` in `content_interpretation_test.cpp`.

## Build and test

```sh
src/encoder/metadata/content_interpretation/build.sh
./build-ci/content_interpretation_test
```

## Decoder gate

```sh
src/encoder/metadata/content_interpretation/run_ci_gate.sh --bin build-m1/av2enc
# or gate on an existing verified M2 stream:
src/encoder/metadata/content_interpretation/run_ci_gate.sh \
    --stream build-m2-cdef/m2_cdef.ivf
```

The gate inserts five spec-conformant variants, requires a byte-exact
self-round-trip and before-first-frame ordering from the scanner, then decodes
the unmodified and patched streams with `dav2d --strict 1 --threads 1` and
requires byte-identical Y4M output.

Result with the pinned decoder `f9dafbc` (macOS arm64):

```
ok:   content_interpretation_test
ok:   unmodified stream decodes with --strict 1
ok:   rich (interlace-field, BT.2100 PQ, chroma 0/5, explicit 16:11 SAR, 1001/30000 timing), 936 -> 952 bytes
ok:   progressive (progressive, BT.709 SDR, 4:2:0 CSP_CENTER, 1:1 SAR, no timing), 936 -> 942 bytes
ok:   explicit_color (complementary field pair, explicit primaries/transfer/matrix, 4:3 SAR), 936 -> 945 bytes
ok:   minimal (all presence flags clear), 936 -> 940 bytes
ok:   four_two_two (progressive 4:2:2 table, 160:99 SAR), 936 -> 945 bytes
note: dav2d parses content_interpretation_obu() (truncated payload reports
      'Overrun in OBU bit buffer' while an unknown OBU type reports
      'Unknown OBU type')
note: dav2d does not reject ci_chroma_sample_position_top = 6 under --strict 1;
      CI conformance checking is not implemented. Decode output is unchanged.
GATE PASS
```

M2: `build-m2-cdef/m2_cdef.ivf` (640×360, 2 temporal units) inserts at byte 23
of frame 0 and decodes byte-identically. `GATE PASS`.

### Decoder behaviour, stated plainly

* `dav2d` f9dafbc **parses** the content interpretation OBU grammar: an empty
  or truncated CI payload fails with `Overrun in OBU bit buffer`, whereas an
  unknown OBU type (26/30) is reported as `Unknown OBU type` and ignored.
* `dav2d` f9dafbc **does not enforce** the §6.14 conformance limit
  `ci_chroma_sample_position_* <= 5`: a probe with position 6 decodes
  unchanged under `--strict 1`.
* Therefore the gate's hard requirements are (a) the CI OBU is accepted at the
  §7.3.3 position without a decoder error, (b) decoded pixels are unchanged,
  and (c) the scanner's byte-exact self-round-trip passes. The grammar-parse
  probe is reported, not required, so a future decoder that ignores the OBU
  still passes on invariance.

## Exact M1/M2 integration sites

This module is a leaf; the owning agents must wire it in. The insertion point
is always *between the sequence header and the first frame OBU* of the first
temporal unit of a coded video sequence.

### M1 (single still picture)

`src/encoder/m1/m1_headers.cpp`, `assemble_m1_still_picture()` currently:

```cpp
out.insert(out.end(), td.begin(), td.end());      // line 300
out.insert(out.end(), seq.begin(), seq.end());    // line 301
out.insert(out.end(), clk.begin(), clk.end());    // line 302
```

Insert the CI OBU between the `seq` and `clk` inserts:

```cpp
const auto ci = av2enc::ci::make_content_interpretation_obu(params, error);
std::vector<std::uint8_t> with_ci;
av2enc::ci::insert_content_interpretation(/* TD+SEQ+CLK */ unit, ci, with_ci, error);
```

The CLI path (`--m1`) assembles the TU in
`src/encoder/m1/m1_still.cpp:1849` via `syntax::assemble_m1_frame()` into
`M1Result::frame`, and writes it in `src/main.cpp:659`. If editing the M1 core
is out of scope, wrap `encoded.frame` immediately before that
`writer.write_frame(encoded.frame, 0, error)` call.

### M2 (keyframe + P frames)

`src/encoder/m2/m2_frame.cpp`, keyframe unit assembly (lines 1007-1016):

```cpp
const std::vector<std::uint8_t> td = make_temporal_delimiter_obu();
const std::vector<std::uint8_t> seq = make_sequence_header_obu(config);
const std::vector<std::uint8_t> obu = make_frame_obu(config, key_header, key_tile);
key_unit.insert(key_unit.end(), td.begin(), td.end());
key_unit.insert(key_unit.end(), seq.begin(), seq.end());
key_unit.insert(key_unit.end(), obu.begin(), obu.end());   // <- insert CI before this
```

Only `frame 0` (the first coded extended layer unit) gets the CI OBU. P-frame
units (lines 1044-1050) must not carry one. Ditto `src/encoder/m2/m2_gate.cpp`
if it writes units itself.

### Encoder-side constancy check

Feed one `CiObservation` per temporal unit into `CvsConstancyTracker` while
building a sequence and fail the encode on any error. `is_first_tu_of_cvs` is
true for the keyframe TU, `has_random_access` is true for CLK/OLK TUs, and
`params` is `nullptr` for TUs without a CI OBU.

## Notes and limitations

* **Not the same as `metadata_scan_type`.** `metadata_scan_type()` (§5.17.10,
  `METADATA_TYPE_SCAN_TYPE` = 8 in `OBU_METADATA_SHORT`) is a per-picture
  metadata unit. The content interpretation OBU is a per-layer, per-CVS HLS
  OBU. They are independent syntaxes and do not override each other.
* **Single coded extended layer / single embedded layer.** The insertion
  helper rejects multiple sequence headers with different `obu_xlayer_id` and
  global (xlayer 31) metadata before the coded extended layer unit, because a
  single insertion offset is then ambiguous. MSDO / LCR scalably layered
  streams are out of scope.
* **No `obu_extension_data()`.** The reader rejects `obu_extension_flag = 1`;
  the writer never emits it. §5.15 has no extension syntax, so there is
  nothing to parse.
* **§7.3.8.11 inconsistency.** The initialization list mentions
  `ci_extension_present_flag = 0`, but §5.15 has no such syntax element. The
  module follows §5.15 and §5.2.1 and does not invent one.
* `ci_color_description_idc != 0` makes the explicit primaries / transfer /
  matrix syntax elements absent; `derive_color_description()` returns the
  Table 6.13 triple for callers that want the effective values.
* `ci_aspect_ratio_idc == 0` selects table row 0 (`0:0`), which §6.14 defines
  as "unspecified"; use `AspectRatio::resolved()` to test that.
* The probe writers (`make_nonconforming_chroma_probe_obu`,
  `--probe-invalid-chroma`, `--probe-truncated`) exist only for the decoder
  gate and must never be used in a shipped stream.

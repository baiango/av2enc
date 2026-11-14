# Leading pictures (AV2 v1.0.0)

Status: **framing, validation, reader/writer, unit tests and the pinned `dav2d`
gate are implemented. M2 core is not edited.** Integration sites are listed in
`VIDEO_INTEGRATION.md`.

Normative reference: `spec/AV2_Spec_v1.0.0.pdf`, AV2 Bitstream & Decoding
Process Specification v1.0.0 (2026-05-28, commit `38f28e7`).

| Topic | Section |
|---|---|
| OBU types 6 / 11 / 13, Table 6.1 | §6.2.2 |
| `IsRegular`, leading vs regular frame | §2, §5.18.2 |
| `immediate_output_frame`, `implicit_output_frame` | §5.18.2, §6.17.2 |
| Coded output / coded non-output frame unit | §7.3.3, §7.3.4 |
| Coded extended layer unit, temporal unit | §7.3.6, §7.3.7 |
| Closed random access | §7.4.3 |
| Open random access, leading frames discarded | §7.4.4 |
| Output frame buffers | §7.21.5, §7.21.6 |
| Leading refresh restriction | §6.17.2 |
| `film_grain_params_present`, `film_grain_config()` (non-output leading pictures code nothing) | §5.4.2, §5.18.10.1 |

## Files

| File | Contents |
|---|---|
| `leading.h` | `SequenceConfig`, `FrameHeader`, `Plan`, output-flag derivation, plan validation, OBU/header writers, temporal-unit assembler |
| `leading.cpp` | Exact §5.4.1 sequence header, §5.18.2 frame header, §5.19 tile-group header and §5.2.2 OBU writer |
| `leading_reader.h/.cpp` | Bit reader, Annex B.2 OBU parser, sequence-header parser, frame-header parser |
| `leading_encode.h/.cpp` | Thin adapter that drives the existing M2 intra/inter picture encoder and returns the §5.20 tile payload + reconstruction |
| `leading_test.cpp` | 108 checks: bit goldens, M2 writer equivalence, round trips, output suppression, plan validation, parse errors |
| `leading_gate.cpp` | CLK + OLK + 3 leading + 3 regular stream, full decode and RAP-start decode with `dav2d --strict 1 --threads 1` |
| `build_and_test.sh` | Builds and runs both. `SANITIZE=1`, `WERROR=1`, `--no-decode`, `--no-run` |

No CMake edit is required. The script builds the module plus the existing
M2 host sources directly.

## API

```cpp
namespace av2enc::inter::leading {

struct SequenceConfig { /* 64x64 defaults, num_ref_frames=8, monotonic=0 */ };
struct FrameHeader {
  FrameKind kind;                 // CLK / OLK / leading tile group / regular
  uint32_t order_hint;
  uint32_t refresh_frame_flags;
  uint8_t num_total_refs;
  std::array<uint8_t, 8> ref_frame_idx;
  bool leading_output_frame;      // leading only
};
struct Plan { std::vector<FrameHeader> frames; };

OutputFlags output_flags(const SequenceConfig&, const FrameHeader&);
bool validate_plan(const SequenceConfig&, const Plan&, std::string& error);

size_t write_frame_header(BitWriter&, const SequenceConfig&, const FrameHeader&);
size_t write_tile_group_header(BitWriter&, const SequenceConfig&, const FrameHeader&);
std::vector<uint8_t> make_frame_obu(const SequenceConfig&, const FrameHeader&,
                                    std::span<const uint8_t> tile_payload);
std::vector<uint8_t> make_sequence_header_obu(const SequenceConfig&);
std::vector<uint8_t> make_temporal_delimiter_obu();
std::vector<std::vector<uint8_t>> assemble_temporal_units(
    const SequenceConfig&, const Plan&,
    std::span<const std::vector<uint8_t>> tile_payloads);

// leading_reader.h
bool parse_obu(std::span<const uint8_t>, ParsedObu&, std::string&);
bool parse_obu_sequence(std::span<const uint8_t>, std::vector<ParsedObu>&, std::string&);
bool parse_frame_header(std::span<const uint8_t> payload, ObuType,
                        const SequenceConfig&, ParsedFrameHeader&, std::string&);
bool parse_sequence_header(std::span<const uint8_t>, SequenceConfig&, std::string&);

}
```

`assemble_temporal_units()` writes one temporal unit per plan frame in decode
order:

```
TD (+ SEQ for the CLK) (+ frame OBU)
TD (+ SEQ for the OLK) (+ leading/regular frame OBU)
...
```

The sequence header is repeated in the OLK temporal unit. §7.3.8 requires every
HLS OBU referenced by a random access point to be available there.

## Output suppression

§5.18.2 gives the exact bit presence rules. `output_flags()` implements them:

| Frame kind | `immediate_output_frame` | `implicit_output_frame` |
|---|---|---|
| CLK | 1 (coded) | absent, inferred 0 |
| OLK | inferred 0 | coded when `monotonic_output_order_flag == 0`; otherwise inferred 0 |
| leading, `leading_output_frame = false` | 0 (coded) | 0 (coded when monotonic = 0, otherwise absent/inferred 0) |
| leading, `leading_output_frame = true` | 1 (coded) | absent, inferred 0 |
| regular | 1 (coded) | absent, inferred 0 |

The default leading mode is the **coded non-output frame unit** of §7.3.4:
`immediate_output_frame = 0`, `implicit_output_frame = 0`. Such a frame is
decoded as a reference but never queued for output. That is what makes the
output of a RAP-start decode identical to sequential decode.

`implicit_output_frame = 1` is only possible when
`monotonic_output_order_flag == 0`. An OLK always has
`immediate_output_frame = 0`, so the encoder must use monotonic order 0 for the
OLK to be output. `validate_plan()` rejects a plan with an OLK and
`monotonic_output_order_flag = 1`.

## Reference restrictions (§6.17.2, §7.4.4)

`validate_plan()` enforces the constraints that matter for this one-frame-per-TU
profile:

1. `frames[0]` must be the CLK, followed by exactly one OLK.
2. Leading frames must follow the OLK and precede the first regular frame.
3. For every leading frame,
   `refresh_frame_flags & OlkRefresh[i] == 0` for all embedded layers.
   This module has one embedded layer, so `OlkRefresh[0]` is the OLK refresh
   mask. Overlap is rejected.
4. A regular frame that follows the OLK may only reference slots refreshed by
   the OLK. References to pre-OLK slots, and references to slots refreshed only
   by leading frames, are rejected.
5. Any frame with `immediate_output_frame == 0` and
   `implicit_output_frame == 0` must refresh at least one slot (§6.17.2).
6. `refresh_frame_flags` and every `ref_frame_idx` must fit `NumRefFrames`, and
   `order_hint` must fit `OrderHintBits`.

Rule 4 is the §7.4.4 requirement that regular frames after an OLK must not
reference leading frames or pre-OLK state. Rule 3 is the matching §6.17.2
requirement that leading frames must not overwrite a slot the OLK made valid
for regular frames.

## Random access interaction (CLK / OLK / RAS)

* **CLK** (§7.4.3) starts a new coded video sequence. It invalidates every
  reference buffer, so no leading picture can precede it and none may follow it
  as part of the same open GOP. The gate starts with a CLK only to make the
  file a complete sequence; the random access point under test is the OLK.
* **OLK** (§7.4.4) is the open random access point. During sequential decoding
  it does not start a new coded video sequence, so leading frames can follow it
  and use the OLK or state the OLK kept valid. When decoding is initiated at the
  OLK, all buffers not refreshed by the OLK (or listed as long-term references)
  are invalidated and the leading frames are discarded. Regular frames after
  the OLK must not reference leading frames or pre-OLK state; rule 4 above is
  the encoder-side check.
* **RAS** (§7.4.5) is inter-predicted random access using long-term reference
  frames. This profile sets `long_term_frame_id_bits = 0`, so RAS and long-term
  leading references are out of scope.
* `OBU_LEADING_SEF` (11) and `OBU_LEADING_TIP` (13) are classified as leading by
  `is_leading_obu()` and parsed as OBUs, but this module does not emit them.
  SEF frames are output frames by construction and TIP requires sequence tools
  this encoder disables.

## OBU / header layout

* OBU header (§5.2.2): `obu_header_extension_flag f(1)`, `obu_type f(5)`,
  `obu_tlayer_id f(2)`. Extension data is not written.
* OBU framing: Annex B.2 `leb128(size)` where `size` counts the header and
  payload, not the leb128 prefix.
* Tile groups are `usedArith` in §5.2.1: there is **no** OBU extension flag and
  **no** OBU trailing-bits byte after the tile payload. The §8.2.4
  `exit_symbol()` inside the tile payload supplies the trailing bits.
* Sequence and temporal-delimiter OBUs are extensible/plain respectively and use
  the §5.2.1 trailing-bits rule.

64x64, `num_ref_frames=8`, `base_q_idx=90`, `order_hint_bits=4`, monotonic = 0,
tile payload `aa55`:

```
seq  : 13048075000abffe0028006ee3800000101b8082
CLK  : 0710f035a000aa55                 order 0, refresh all
OLK  : 0814f200b5a000aa55               order 4, refresh slot 0
LEAD : 0a18f028148235a000aa55           order 1, refresh slot 1, ref slot 0,
                                        immediate=0 implicit=0
REG  : 0a1cf95109046b4000aa55           order 5, refresh slot 4, ref slot 0
```

These are asserted in `leading_test.cpp::test_bit_goldens()`.

The unit test also asserts, byte for byte, that:

* with `monotonic_output_order_flag = 1`, our sequence header equals
  `av2enc::video::make_sequence_header_obu()`;
* our CLK and regular-P frame OBUs equal `av2enc::video::make_frame_obu()` for the same
  header fields.

This keeps the new writer in lock-step with M2 while allowing the extra leading
fields M2 cannot write today.

## Tests

```sh
./src/encoder/inter/leading/build_and_test.sh
```

```
compiler: Apple clang version 21.0.0 (clang-2100.1.1.101)
108 checks, 0 failure(s)
...
LEADING GATE PASS
```

Unit coverage:

* bit goldens above;
* M2 sequence/frame writer equivalence;
* OBU and frame-header round trips, including the non-output and
  immediate-output leading variants;
* sequence-header round trip;
* `output_flags()` suppression table;
* plan validation: OLK refresh overlap, leading before OLK, leading after the
  first regular frame, regular referencing a leading slot, regular referencing
  a pre-OLK slot, OLK with monotonic = 1, non-output frame with refresh = 0;
* temporal-unit assembly (SEQ present at CLK and OLK, absent at leading/regular);
* truncated/zero-size OBU rejection.

## Decoder gate

The gate uses a 64x64 synthetic 8-frame clip and the existing M2 picture
encoder. Decode order:

```
0 CLK(f0)  order 0
1 OLK(f4)  order 4   <- open random access point, SEQ repeated
2 LEAD(f1) order 1   <- displayed before the OLK
3 LEAD(f2) order 2
4 LEAD(f3) order 3
5 REG(f5)  order 5
6 REG(f6)  order 6
7 REG(f7)  order 7
```

Leading frames reference the OLK. They cannot reference pre-OLK normal frames:
an OLK is a key frame, and §7.23 leaves only its refreshed slot valid. Regular
frames reference the OLK only, so they never reference a leading frame.

Both decodes use the pinned decoder:

```
dav2d -q --strict 1 --threads 1 -i <stream>.ivf -o <stream>.y4m
```

The gate writes two streams per scenario:

* full: CLK + OLK + leading + regular;
* RAP: the same IVF with the CLK temporal unit removed, so it starts at
  `TD + SEQ + OLK`.

Observed PASS (macOS pinned `tools/bin/dav2d-macos`, strict 1, threads 1):

```
== scenario: nonoutput ==
  full stream: 8 temporal units; RAP clip: 7 (starts at TD+SEQ+OLK)
  leading output frames: immediate=0 implicit=0
  PASS: full=5 frame(s), RAP=4 frame(s), all samples exact
  PASS: non-output leading frames are suppressed; output from the OLK is
        identical to sequential decode
  PASS: physically removing the leading temporal units does not change the RAP
        decode
== scenario: output ==
  full stream: 8 temporal units; RAP clip: 7 (starts at TD+SEQ+OLK)
  leading output frames: immediate=1
  PASS: full=8 frame(s), RAP=7 frame(s), all samples exact
  PASS: RAP-start decode dropped the pre-RAP CLK; immediate-output leading
        frames remain decodable from the OLK
LEADING GATE PASS
```

In the non-output case the full stream has 8 coded frames but only 5 output
frames: the three leading frames are suppressed. Starting at the OLK yields 4
frames, exactly the full output minus the pre-RAP CLK, sample for sample.

### Observed decoder behaviour

`dav2d` drops the pre-RAP CLK when decoding starts at the OLK. It does **not**
drop a leading frame that is still decodable from the OLK: with
`immediate_output_frame = 1` it decodes and outputs those frames even in the
RAP-start decode. This is why the interoperable profile in this module is the
§7.3.4 coded **non-output** leading frame. Such frames never enter the output
pipeline, so sequential and RAP-start decodes are identical by construction.

The spec's decoder-model conformance requirement is weaker than "the decoder
must throw the frames away": §E.7 requires that a bitstream formed by removing
the leading frames is still conformant. The gate verifies the observable
contract we need for seeking: after the pre-RAP CLK is removed, output from the
OLK is unchanged and leading frames are absent from output.

## Limitations

* Only leading **tile group** frames are written. `OBU_LEADING_SEF` and
  `OBU_LEADING_TIP` are recognised by the parser/classifier but not emitted.
* One tile per frame, one embedded layer, no long-term reference frames, and
  the same sequence-level tool set as M2. `long_term_frame_id_bits = 0`, so
  leading frames cannot reference pre-OLK normal frames (the OLK invalidates
  them); this is documented rather than silently allowed.
* Picture coding is delegated to M1/M2. This module owns framing, output
  suppression, reference restrictions and the gate.

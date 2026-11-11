# Random-access switch / RAS frames (AV2 v1.0.0)

Self-contained component for `OBU_SWITCH` (obu_type 10) and `OBU_RAS_FRAME`
(obu_type 21): frame-type derivation, allowed reference sets, conformance
checks, header writer/reader with bit goldens, an encoder-side reuse/re-encode
planner, and a three-frame `dav2d --strict 1 --threads 1` decoder gate.

**Component only.** Nothing outside `src/encoder/inter/random_access/` is
edited. There is no BD-rate claim and no runtime M2 integration.

Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
(commit `38f28e7`):

- §6.2.2 Table 6.1 — `OBU_SWITCH = 10`, `OBU_RAS_FRAME = 21`, both
  layer-specific, both require `obu_tlayer_id == 0`.
- §5.18.2 `frame_header_info()` — SWITCH/RAS branch:
  `restricted_prediction_switch`, `FrameType = SWITCH_FRAME`,
  `long_term_id_plus_1`, `num_key_ref_frames` / `ref_long_term_id`,
  forced `frame_size_override_flag = 1`, forced `explicitRefFrameMap = 1`,
  derived RAS `refresh_frame_flags`, and the omitted
  `signal_primary_ref_frame` / `disable_cross_frame_cdf_init` /
  `skip_mode_present` fields.
- §6.17 frame-header semantics — `restricted_prediction_switch`,
  `ref_long_term_id`, and the RAS conformance requirement
  `long_term_id_in_use(RefLongTermId[ref_frame_idx[i]]) == 1`.
- §7.4.5 Random Access Switch — RAS is inter-predicted but may only reference
  long-term frames named in `ref_long_term_id`; it refreshes every other slot
  and does not start a new CVS.
- §5.4.6 — `long_term_frame_id_bits` (3-bit sequence field).

## Status

| Item | State |
|---|---|
| SWITCH/RAS frame-type derivation (Table 6.1, §5.18.2) | complete |
| Allowed reference masks (SWITCH: valid slots; RAS: named long-term slots) | complete |
| §5.18.2 RAS refresh derivation | complete |
| Conformance validation (tlayer, long-term ids, refs, dimensions, order hint) | complete |
| Sequence header writer with `long_term_frame_id_bits` | complete |
| Frame header writer/reader (CLK / SWITCH / RAS) | complete |
| Bit goldens (sequence + key + switch + RAS, normative and pinned-decoder form) | complete |
| Encoder strategy: reuse a P payload / re-encode | complete |
| Unit tests | 142 checks, pass |
| `dav2d --strict 1 --threads 1` gate, synthetic 128x128 | pass, 0 recon diffs |
| `dav2d --strict 1 --threads 1` gate, real 720p | pass, 0 recon diffs |
| M2 runtime wiring | not wired (sites in `M2_INTEGRATION.md`) |

## Files

```text
src/encoder/inter/random_access/
  random_access.h          public API and spec notes
  random_access.cpp        derivation, refs, validation, writer/reader, planner
  random_access_test.cpp   142 unit checks + bit goldens
  ras_gate.cpp             three-frame SWITCH+RAS decoder gate
  build_and_test.sh        builds, runs tests, runs the gate
  M2_INTEGRATION.md        exact M2 call sites and splice recipe
  README.md
```

## API

All names are in `av2enc::inter::random_access`.

### Derivation

```cpp
FrameTypeInfo derive_frame_type(ObuType, bool restricted_prediction_switch);
```

`ObuType` is the raw Table 6.1 value (`kSwitch = 10`, `kRas = 21`, plus
`kClosedLoopKey = 4` and `kRegularTileGroup = 7`). For both SWITCH and RAS the
result is:

```text
FrameType              = SWITCH_FRAME (3)
FrameIsIntra           = 0
restricted bit         = present
explicitRefFrameMap    = 1 (forced)
frame_size_override    = 1 (forced)
primary_ref_frame      = PRIMARY_REF_NONE (no signal bits)
signal_primary_ref / disable_cross_frame_cdf_init  absent
skip_mode_present      absent
```

RAS additionally reads `num_key_ref_frames` and derives
`refresh_frame_flags`; SWITCH reads `refresh_frame_flags` explicitly. SWITCH
with `restricted_prediction_switch = 1` (and every RAS) resets QM state.

### Reference sets

```cpp
bool long_term_id_in_use(int32_t id, span<const uint8_t> ref_long_term_ids);
uint16_t allowed_reference_mask(ObuType, const RefState&, span<const uint8_t>);
uint16_t derive_ras_refresh_flags(const RefState&, span<const uint8_t>,
                                  uint8_t num_ref_frames = 16);
```

- SWITCH may reference any valid slot.
- RAS may reference only valid slots whose `RefLongTermId` appears in
  `ref_long_term_id`.
- RAS refresh:
  `refresh_frame_flags |= 1 << i` for every `i < NumRefFrames` with
  `!RefValid[i] || !long_term_id_in_use(RefLongTermId[i])`.

`update_reference_state()` applies the §7.23 update used by the gate and tests,
including marking existing slots restricted for a restricted SWITCH before the
new frame is stored.

### Conformance

```cpp
ValidationResult validate_header(const SequenceConfig&, const FrameHeader&,
                                 const RefState&, uint8_t tlayer_id = 0);
```

Checks include: Table 6.1 `obu_tlayer_id == 0`; `long_term_frame_id_bits`
fits 3 bits; `long_term_id_plus_1` fits the field and is not the reserved
all-ones value; `ref_long_term_id[i] != (1 << bits) - 1`; every named
long-term id is held in a slot (§7.4.5); a referencing RAS needs
`num_key_ref_frames > 0`; RAS reference slots must be long-term and named;
`num_total_refs <= REFS_PER_FRAME`; `ref_frame_idx < NumRefFrames`; reference
slots valid; §6.17.4.3 dimension ratio (1:2 .. 16:1); unrestricted RAS
`OrderHint < (1 << OrderHintBits)`; SWITCH refresh bits within `NumRefFrames`;
and the derived RAS refresh mask.

### Writer / reader

```cpp
std::vector<uint8_t> make_sequence_header_obu(const SequenceConfig&);
std::vector<uint8_t> make_frame_obu(const SequenceConfig&, const FrameHeader&,
                                    span<const uint8_t> tile_payload);
std::size_t write_frame_header_info(BitWriter&, const SequenceConfig&,
                                    const FrameHeader&);
std::size_t write_tile_group_header(BitWriter&, const SequenceConfig&,
                                    const FrameHeader&);

bool parse_obu(span<const uint8_t>, ObuView&, size_t& consumed, string& error);
bool read_sequence_header(span<const uint8_t>, SequenceHeaderView&, string&);
bool read_frame_header(span<const uint8_t>, ObuType, const SequenceHeaderView&,
                       FrameHeader&, size_t& bits_consumed, string&,
                       bool ras_codes_refresh_frame_flags = false);
```

With `long_term_frame_id_bits = 0` the sequence header is **byte-identical** to
`av2enc::m2::make_sequence_header_obu()` for the same 64x64 profile; the unit
test asserts this.

### Encoder strategy

```cpp
PFrameCandidate { reference_slot, refresh_frame_flags, order_hint,
                  has_inter_blocks };
PayloadPlan plan_switch_frame(config, refs, candidate, restricted);
PayloadPlan plan_ras_frame(config, refs, candidate, ras_long_term_ids);
```

`PayloadDecision::kReusePFrame` means only the frame header and OBU type need
to change; the M2 range-coded tile payload is copied verbatim. The M2 tile
payload depends on:

- the frame being inter (`inter_frame_mode_info`): both SWITCH and RAS have
  `FrameIsIntra = 0`;
- the CDF state: M2 sets `disable_cross_frame_cdf_init = 1` and
  `disable_cdf_update = 1`, and SWITCH/RAS use `PRIMARY_REF_NONE`, so both
  paths initialise the same default CDFs;
- the reference slot: header `ref_frame_idx[0]` must select the same slot the
  payload was encoded against;
- the frame dimensions and quantiser, which are copied from the M2 config.

SWITCH: the candidate's slot must be valid; the header keeps the payload's
reference slot and (typically) refreshes a *different* slot so the long-term
slot survives.

RAS: the candidate's slot must hold the long-term frame named in
`ref_long_term_id`. M2 chains `P_i` against `recon[i-1]`, so a P payload
encoded against the preceding SWITCH cannot be reused; encode a separate
two-frame sequence `[frame0, frame2]` and use its P payload, which predicts
from frame 0. An all-intra P payload is reusable too, but the RAS still
declares `num_total_refs = 1` (the pinned decoder rejects 0).

When no long-term slot exists (`long_term_frame_id_bits == 0`, or the payload
predicts from a short-term slot), the plan returns
`kReencodeRequired` with the reason. The caller must either encode a P frame
against a long-term slot or keep the frame as a regular P frame.

### Pinned-decoder compatibility

The pinned dav2d build (`f9dafbc`, 2026-11-10) does not implement the §5.18.2
RAS refresh derivation. Its `parse_frame_header()` falls through to the generic
`SWITCH_FRAME` branch and reads `NumRefFrames` explicit refresh bits for
`OBU_RAS_FRAME` (the source has a `FIXME special cases for bridge and ras
frames` comment). It also rejects `n_ref_frames == 0`.

`SequenceConfig::ras_codes_refresh_frame_flags` selects that form. The default
is the normative behaviour. The gate sets it to `true` and writes exactly the
mask `derive_ras_refresh_flags()` would produce, so the stream is consistent
with §5.18.2 apart from the redundant bits the pinned decoder needs.

## Bit goldens

64x64, `order_hint_bits = 4`, `num_ref_frames = 2`, q = 90,
`long_term_frame_id_bits = 1`:

```text
sequence header  13 04 80 75 00 2a bf fe 00 28 00 6e 27 80 00 00 10 1b 80 82
key frame OBU    05 10 f8 1a d0 00
                   is_first=1, restricted absent, long_term_id_plus_1=1,
                   order_hint=0, refresh=allFrames
switch frame OBU 08 28 f8 c5 ff e4 6b 40 00
                   restricted=1, order_hint=1, refresh=0x2, n_refs=1, ref=0
RAS (normative)  08 54 e2 91 7f f9 1a d0 00
                   restricted=0, n_key_ref=1, ref_lt_id=0, refresh derived,
                   n_refs=1, ref=0
RAS (pinned)     08 54 e2 94 5f fe 46 b4 00
                   same plus 2 explicit refresh bits (mask 0x2)
```

128x128 `long_term_frame_id_bits = 1` sequence header:

```text
14 04 80 75 00 2c df ff 80 0a 00 1b 89 e0 00 00 04 06 e0 20 80
```

## Gate

```sh
./src/encoder/inter/random_access/build_and_test.sh
```

Default output (synthetic 128x128, then real 720p when
`testdata/shine_1280x720_60f.y4m` is present):

```text
== unit tests ==
142 checks, 0 failure(s)
RANDOM ACCESS TEST PASS
== gate: synthetic 128x128, 3 frames ==
  frame 0: OBU_CLOSED_LOOP_KEY  order=0 long_term_id=0 tile=1281 bytes
  frame 1: OBU_SWITCH           order=1 restricted=1 ref_slot=0 refresh=0x2 inter_blocks=7 tile=737 bytes
  frame 2: OBU_RAS_FRAME        order=2 num_long_term=1 num_total_refs=1 ref_long_term_id={0} ref_slot=0 derived_refresh=0x2 inter_blocks=7 tile=688 bytes
  decode: tools/bin/dav2d-macos --strict 1 --threads 1 -q -i .../ra_gate.ivf -o .../ra_gate.ivf.decoded.y4m
  PASS: 3 frame(s) decoded with --strict 1 --threads 1, recon exact (0 diffs)
== gate: real 720p, 3 frames ==
  frame 0: OBU_CLOSED_LOOP_KEY  order=0 long_term_id=0 tile=43694 bytes
  frame 1: OBU_SWITCH           order=1 restricted=1 ref_slot=0 refresh=0x2 inter_blocks=315 tile=30775 bytes
  frame 2: OBU_RAS_FRAME        order=2 num_long_term=1 num_total_refs=1 ref_long_term_id={0} ref_slot=0 derived_refresh=0x2 inter_blocks=325 tile=27416 bytes
  PASS: 3 frame(s) decoded with --strict 1 --threads 1, recon exact (0 diffs)
RANDOM ACCESS GATE PASS
```

The gate splices the M2 tile payloads rather than synthesising headers around a
dummy payload:

1. M2 three-frame encode: frame 0 CLK, frame 1 P payload (for SWITCH).
2. M2 two-frame encode `[frame0, frame2]`: its P payload predicts from frame 0
   (for RAS).
3. Strip each M2 tile-group header (size from `m2::write_tile_group_header()`),
   keep the range-coded tile bytes.
4. Rebuild the units with this component's sequence/key/switch/RAS headers.
5. Decode with `dav2d --strict 1 --threads 1`; compare every Y/U/V sample with
   the encoder reconstruction for frames `{seq3.recon[0], seq3.recon[1],
   seq2.recon[1]}`.

## Limitations

- CPU reference only; no CUDA kernel.
- Reader supports the fixed M2 sequence profile (64x64 superblocks, frozen
  CDFs, no CDEF/GDF/LR/CCSO, no TIP/BRU/BAWP/OPFL/global motion). A leading
  tool switch that differs sets `supported_profile = false`.
- SWITCH/RAS payloads must use the single-reference `LAST` mode the M2 tile
  writer emits (`ref_frame_idx[0]`); compound, TIP and bridge syntax are out of
  scope.
- The pinned decoder requires the compatibility refresh bits described above;
  normative streams (no RAS refresh bits) should be re-tested after a dav2d
  update.

# `ops_full` — field-complete LCR (§5.8) and OPS (§5.10/§5.11)

Status: **component, spec-only.** No `av2enc` CLI flag reaches this code and no
existing `src/encoder/svc/` file was modified. Everything below is gated by
`build_and_test.sh` (unit tests + goldens) and `gate_avmdec.sh` (black-box
`avmdec` at 2 and 3 layers).

Normative source: `spec/AV2_Spec_v1.0.0.pdf` (v1.0.0, commit `38f28e7`).
Syntax bodies were extracted from the PDF pages 71–75 (§5.8) and 80–83
(§5.10/§5.11); semantics from pages 276–287 (§6.8), 293–300 (§6.10) and 301
(§6.11). Bit order is MSB-first; `uvlc()`/`rg(n)`/`leb128()` follow §4.11 and
Annex B.2.

This module exists because `svc_framing.cpp` emits only the reset/explicit-map
skeleton (audit rows "LCR seq profile tier level info: ABSENT", "LCR xlayer
colour info: ABSENT", "OPS aggregate info: ABSENT", "OPS idc-2 inheritance:
ABSENT", …). The skeleton is correct, but every optional record — PTL, colour,
purpose, atlas reference, embedded-layer type/view/priority/rendering, decoder
model, initial display delay, inheritance — was hardcoded absent.

## Files

| File | Purpose |
|---|---|
| `lcr_ops_full.h` | data model, writers, readers, validation API |
| `lcr_ops_full.cpp` | all syntax, conformance checks, legacy conversion |
| `lcr_ops_full_test.cpp` | goldens, round-trips, rejections, validation, `svc_framing` cross-check |
| `lcr_ops_mux.cpp` | gate-only tool: splices this module's LCR/OPS into a composed stream |
| `build_and_test.sh` | builds and runs the unit suite |
| `gate_avmdec.sh` | 2/3-layer `avmdec` gate, `--select-ops` per operating point |
| `README.md` | this file |

## API

Namespace `av2enc::svc::ops_full`. `layers::` is
`av2enc::tiles::svc`.

### Payload writers

```cpp
bool write_lcr_payload(const LayerConfigRecord&, std::vector<uint8_t>& out,
                       std::string& error);
bool write_ops_payload(const OperatingPointSet&, std::vector<uint8_t>& out,
                       std::string& error);
```

Both return the bytes that follow the OBU header, including the
extensible-OBU tail (`obu_extension_flag = 0` + `trailing_bits()`), exactly as
`layer_config_record_obu()` / `operating_point_set_obu()` require
(§5.2.1, §5.8, §5.10). Frame them with `svc::frame_obu()` /
`svc::encode_obu_header()`.

Sub-structure writers are public for goldens and for callers that already own
the surrounding `BitWriter`:

```
write_lcr_aggregate_info                  write_ops_aggregate_info
write_lcr_seq_profile_tier_level_info     write_ops_seq_profile_tier_level_info
write_lcr_xlayer_color_info               write_ops_decoder_model_info
write_lcr_rep_info                        write_ops_color_info
write_lcr_embedded_layer_info             write_ops_mlayer_info
write_lcr_xlayer_info
```

### Readers

```cpp
bool read_lcr_payload(std::span<const uint8_t> payload, bool is_global,
                      LayerConfigRecord&, std::string& error);
bool read_ops_payload(std::span<const uint8_t> payload, uint8_t xlayer_id,
                      OperatingPointSet&, std::string& error);
```

`is_global` / `xlayer_id` come from `obu_xlayer_id` (they are not coded in the
payload). The readers are the exact inverse of the writers: they validate
`obu_extension_flag == 0`, the trailing-bits pattern, every mandatory `0`
reserved field, `lcr_data_size` (including
`lcr_remaining_payload_bit` padding) and `opsBytes == ops_data_size`
(§6.10.2). A re-write of a parsed payload is byte-identical.

### Data model

* `LayerConfigRecord { bool is_global; GlobalLcrInfo global; LocalLcrInfo local; }`
* `GlobalLcrInfo`: `config_record_id`, `xlayer_map`, `aggregate_info_present`
  + `AggregateInfo`, `profile_tier_level_info_present` + `ptl[32]`,
  `global_payload_present`, `dependent_xlayers_flag`, `global_atlas_id_present`
  + `global_atlas_id`, `global_purpose_id`, `doh_constraint_flag`,
  `enforce_tile_alignment_flag`, `payloads` (`GlobalPayload` each carrying
  `xlayer_id`, `dependent_xlayer_mask`, `XLayerInfo`,
  `declared_size_bytes`).
* `LocalLcrInfo`: `xlayer_id` (from the OBU header), `global_id`, `local_id`,
  PTL flag + `ptl`, `local_atlas_id_present` + `local_atlas_id`, `xlayer`.
* `XLayerInfo`: `rep_info_present` + `RepInfo`, `purpose_present` +
  `purpose_id`, `color_info_present` + `ColorDescription`,
  `embedded_layer_info_present`, `mlayer_map`, `embedded[]`
  (`EmbeddedLayerInfo`: atlas/priority/rendering, layer/auxiliary type,
  view type/id, dependent layer mask, resolution), and the global-only
  xlayer atlas triple used when embedded info is absent.
* `OperatingPointSet`: `xlayer_id`, `ops_id`, `reset_flag`, `priority`,
  `intent`, `intent_present`, `ptl_present`, `color_info_present`,
  `mlayer_info_idc`, `reserved_2bits`, `points[]`.
* `OpsPoint`: op intent, `aggregate`, `ptl[32]`, colour, decoder-model
  presence + `OpsDecoderModelInfo`, initial display delay, `xlayer_map`,
  `layer_info[32]` (`OpsLayerMlayerInfo`: `explicit_info`, `mlayer_map`,
  `tlayer_map[8]`, `embedded_ops_id`, `embedded_op_index`).
* `ColorDescription` is shared by LCR and OPS: `description_idc` (rg(2)),
  explicit primaries/transfer/matrix (present only for idc 0), `full_range`.

`ops_cnt == 0` is supported in both forms (`reset_flag = 1` = full reset,
`reset_flag = 0` = reset only `ops_id`), which the old writer rejected.

### Validation

```cpp
struct XLayerSequenceInfo { /* reduced activated sequence header */ };
struct OpsSummary { uint8_t xlayer_id, ops_id, ops_cnt; };
struct StreamContext { sequences, known_ops, dependencies,
                       dependencies_known, lcr_active,
                       active_global_lcr_id };
struct ValidationReport { std::vector<std::string> errors; bool ok() const; };

bool validate_layer_map_closure(xlayer_id, mlayer_map, tlayer_map,
                                dependencies, report);
bool validate_lcr(record, context, report);
bool validate_ops(set, context, report);
bool validate_stream(lcr, ops, context, report);
bool validate_local_lcr_matches_global(global, local, report);
bool validate_brt_against_ops(brt, known_ops, report);
bool resolve_ops_inheritance(inherited, sources, resolved, error);
OpsSummary summarize_ops(const OperatingPointSet&);
```

What is enforced (all "shall"/"requirement of bitstream conformance" text):

| Rule | § | Where |
|---|---|---|
| `lcr_global_config_record_id` 1..7, `lcr_xlayer_map != 0` | 6.8.2 | writer + `validate_lcr` |
| `lcr_dependent_xlayers_flag == 0` | 6.8.2 | writer + `validate_lcr` |
| all extended layers reference one global LCR id | 6.8.2 | `validate_lcr` (`lcr_active`) |
| `lcr_local_id` 1..7, `lcr_global_id` points at the active global LCR | 6.8.3 | writer + `validate_lcr` |
| PTL: `seq_profile_idc <= lcr_seq_profile_idc`, `seq_level_idx <= lcr_max_level_idx`, `seq_tier <= lcr_tier_flag`, `seq_max_mlayer_cnt <= lcr_max_mlayer_count` | 6.8.5 | `validate_lcr` |
| aggregate dominates every per-layer PTL (`level`, `tier`) | 6.8.4 | `validate_lcr` / `validate_ops` |
| `lcr_max_pic_width/height`, bit depth, chroma format, crop flag and offsets equal the activated sequence header | 6.8.7/6.8.8 | `validate_lcr` |
| `lcr_max_expected_width/height <= max_frame_width/height` | 6.8.9 | `validate_lcr` |
| `lcr_doh_constraint_flag == 1` when any sequence header has `monotonic_output_order_flag == 0` | 6.8.2 | `validate_lcr` |
| mlayer/tlayer map closure against `MLayerDependencyMap` / `TLayerDependencyMap` | 6.8.9, 6.10.7 | `validate_layer_map_closure` |
| local xlayer info == global xlayer info (embedded maps, types, views, colour, rep, PTL) | 6.8 | `validate_local_lcr_matches_global` |
| `ops_mlayer_info_idc != 3`, global header fields present only when `ops_cnt > 0`, local `ops_reserved_2bits == 0` | 6.10.2 | writer + `validate_ops` |
| `ops_embedded_op_index < ops_cnt[refID]`, and `< j` when `refID == opsID` | 6.10.2 | `validate_ops` |
| OPS PTL bounds and `seq_max_mlayer_cnt <= ops_mlayer_count` | 6.10.4 | `validate_ops` |
| OPS selects only layers described by `lcr_xlayer_map` | 6.8/6.10 | `validate_stream` |
| `br_ops_id` names a present `ops_id`; `br_ops_cnt == ops_cnt`; tick ranges | 6.11 | `validate_brt_against_ops` |

Not enforced, deliberately (documented limits):

* Table 6.13 `ops_color_description_idc` 4/5 and Table 6.6 purpose ids > 7 are
  legal values that this syntax version cannot encode: the descriptor is
  `rg(2)` (0..3) and `lcr_global_purpose_id` is `f(7)`. The writer rejects
  `description_idc > 3` with an explicit message instead of truncating.
  `lcr_layer_type`/`view_type`/`auxiliary_type`/`rendering_method` reserved
  values are written verbatim and *not* rejected: the spec says decoders
  ignore them, so a rejection would be stricter than the bitstream.
* `ops_decoder_buffer_delay + ops_encoder_buffer_delay` constancy across a
  random-access sequence (§6.10.5) is a sequence-wide property; the hook needs
  all OPS OBUs of a CVS and is left to the caller (see "call sites").
* No atlas segment OBU (§5.9) exists anywhere in the tree; only the LCR-side
  atlas *references* are coded. `validate_lcr` therefore cannot check
  `lcr_xlayer_atlas_segment_id` against a real atlas.

## Pinned goldens

The five sub-structure goldens and the three payload goldens were derived with
an independent bit-packer written from the PDF text (not by this C++). The
derivation, in full:

```python
class BW:                      # MSB-first packer
    def __init__(self): self.bits = []
    def put(self, v, n): self.bits += [(v >> i) & 1 for i in range(n - 1, -1, -1)]
    def bit(self, b): self.put(1 if b else 0, 1)
    def uvlc(self, v):
        lz = 0
        while v >= (1 << (lz + 1)) - 1: lz += 1
        self.put(0, lz); self.bit(1); self.put(v - ((1 << lz) - 1), lz)
    def rg(self, v, n):        # q one bits, zero, n-bit remainder
        for _ in range(v >> n): self.bit(1)
        self.bit(0); self.put(v & ((1 << n) - 1), n)
    def align(self):
        while len(self.bits) % 8: self.bits.append(0)
    def hex(self):
        self.align()
        return bytes(int(''.join(map(str, self.bits[i:i+8])), 2)
                     for i in range(0, len(self.bits), 8)).hex()
```

| Structure | Inputs | Golden |
|---|---|---|
| `lcr_seq_profile_tier_level_info` | profile 2, level 7, tier 1, max_mlayer 3 | `11ec` |
| `lcr_aggregate_info` | config 3, level 9, main, interop 2 | `0d22` |
| `lcr_xlayer_color_info` | idc 1, full range | `30` |
| `lcr_xlayer_color_info` | idc 0, 9/16/9, studio | `01220120` |
| `lcr_embedded_layer_info` | m1, t=0b0001, aux depth, VIEW_EXPLICIT id 6, atlas 17/prio 2/render 0, dep m0, 32×24 | `1110200010104068108640` |
| local LCR payload (§5.8.2–§5.8.9) | see `local_full_record()` | `2f0a48a0f020831d590a60e603309010200028111020001010406810864040` |
| global LCR payload (§5.8.1/§5.8.5) | x0+x2, aggregate+PTL+atlas-free payloads | `40000001780b00115302a40284067068011000180670680110001840` |
| global OPS payload (§5.10–§5.11.5) | 3 points, idc 2, decoder model on op0, delay per op | `83103e17022286500009c408002711400000001814640440a320220f0222864880000003028c80881460000f0222864900000003028c808814600040` |

The `lcr_data_size` byte of the global LCR golden is `0x06` (offset 13) and the
first `ops_data_size` byte of the OPS golden is `0x17` (offset 3); the reader
tests mutate exactly those bytes to prove the size checks fire.

## Tests

`./build_and_test.sh` (add `SANITIZE=1`, `WERROR=1`, `CXX=…`):

```
building: lcr_ops_full_test
building: lcr_ops_mux (compile-only)
running: lcr_ops_full_test
LCR/OPS full tests: all checks passed
```

Coverage:

1. **Goldens** — the eight pinned records above, plus
   `ops_decoder_model_info` bit-count.
2. **Round-trip matrix** — 6 LCR variants (aggregate on/off, PTL on/off,
   global atlas + no payload, padded `lcr_data_size`, local LCR) and 7 OPS
   variants (idc 1/2/0, no intent/PTL/colour, both `ops_cnt == 0` reset forms,
   local OPS) all re-encode byte-identically after parsing.
3. **Writer rejections** — config id 0, empty `xlayer_map`, dependent-xlayer
   flag, missing payload, empty `tlayer_map`, `mlayer_map`/descriptor
   mismatch, `rg(2)` overflow, dependent map ≥ j, idc 3, ops_cnt 8, ops_id 16,
   empty `ops_mlayer_map`, local reserved bits.
4. **Reader rejections** — `obu_extension_flag = 1`, truncation,
   `lcr_data_size` overflow, bad `lcr_data_size`, non-zero padding bit,
   `opsBytes != ops_data_size`, bad trailing bit.
5. **Validation** — PTL profile/level/mlayer-count violations, aggregate
   dominance, DOH constraint, rep size/format/crop mismatch, expected-size
   overflow, xlayer-map/payload closure, local-parent id, OPS PTL/aggregate,
   empty tlayer map, unknown xlayer, inheritance (unknown OPS, self-reference),
   mlayer/tlayer closure, all BRT cases (ok, missing OPS, cnt mismatch,
   entry-count mismatch, ambiguous cross-xlayer ops_id, `rg(4)` range).
6. **`svc_framing` cross-check** — for 2-layer/3-layer plans with
   temporal 1..3 and embedded 1..2, the skeleton built by
   `lcr_from_legacy()`/`ops_from_legacy()` is **byte-identical** to
   `svc::write_layer_config_record_payload()` /
   `svc::write_operating_point_set_payload()`, for the global *and* the local
   LCR form. The converted skeletons also parse back and pass
   `validate_stream` against sequence headers built from the same plan.
7. **Full legacy form** — `include_full_info = true` round-trips aggregate,
   PTL, purpose, colour, idc 2, decoder model and delay.

## Decoder gate

`./gate_avmdec.sh` (needs `avmdec`; `dav2d` optional; `KEEP=1` keeps streams):

```
ok: avmdec 2-layer --all-layers
ok: avmdec 2-layer --select-ops=0,0 / 0,1
ok: avmdec 2-layer --select-local-ops=0,1,0 / 1,1,0
ok: dav2d 2-layer --alllayers 1
ok: avmdec 3-layer --all-layers
ok: avmdec 3-layer --select-ops=0,0 / 0,1 / 0,2
ok: avmdec 3-layer --select-local-ops=0..2
ok: dav2d 3-layer --alllayers 1
ok: local LCR mismatch rejected before the decoder saw it
ok: avmdec 2-layer basic LCR/OPS
PASS: LCR/OPS full writers gated by avmdec at 2 and 3 layers
```

How it works and why it is not `svc/gate_dav2d.sh`:

* The existing SVC gate composes `av2enc` **still** streams, whose sequence
  headers are `single_picture_header_flag = 1`. That header has no
  `seq_lcr_id` field at all, so `compose_layered_stream()` refuses to associate
  an LCR and the gate exits 77 (`SKIP`). It has therefore never exercised
  LCR/OPS with a decoder.
* This gate instead uses `video_gate --force-intra` streams. They are intra
  pictures with a `single_picture_header_flag = 0` header, so
  `seq_lcr_id = 1` is present and the LCR is **activated**.
* `svc_mux --no-lcr --no-ops` composes the layer OBUs (this part is the
  unchanged `svc_framing` path); `lcr_ops_mux` inserts the `ops_full` bytes in
  the §7.3.3 config position. `lcr_ops_mux` validates the stream before writing
  it, so a gate failure is a writer/validator defect, not a malformed stream.
* Per operating point: OP *i* selects xlayers 0..*i*. `avmdec --select-ops=0,i`
  must decode exactly the concatenation of the independently decoded layers,
  which proves the `ops_xlayer_map` of each point, the OPS parser and the
  decoder's OP selection all agree.
* The 3-layer stream additionally carries `ops_mlayer_info_idc = 2` with
  `ops_mlayer_explicit_info_flag = 0` inheritance for the newest layer of
  OP 1/2, so the inheritance fields are decoder-gated, not just unit-tested.
* `--atlas` sets `lcr_global_atlas_id_present` and the per-embedded-layer atlas
  id/priority/rendering fields; `avmdec` accepts them. (There is still no atlas
  segment OBU to reference; the ids are structural.)
* `--local-lcr` emits a local LCR per extended layer with
  `lcr_global_id = 1`; `avmdec` parses it. The negative control
  (`--local-lcr-mismatch`) makes the local record disagree with the global one
  and must be rejected by `validate_local_lcr_matches_global()` — this is the
  check behind avmdec's "Local LCR xlayer_info … does not match" message.
* `--features basic` reproduces the `svc_framing` skeleton byte-for-byte and
  must still decode; it is the regression guard for the merge.
* `dav2d` ignores types 16/18 (it prints "Unknown OBU type 16/18") and decodes
  the same pictures, so the added config OBUs are picture-neutral.

## Exact merge path into `svc_framing`

The merge is mechanical because `lcr_from_legacy()` / `ops_from_legacy()` with
`include_full_info = false` already reproduce today's bytes exactly (test 6).

1. Add `lcr_ops_full.cpp` to the SVC build (`src/encoder/svc/build_and_test.sh`,
   and the `SVC_CORE` arrays in the gates). No CMake change: the SVC module is
   not in the root `CMakeLists.txt`.
2. In `svc_framing.cpp`, delete the static writer bodies
   `write_xlayer_info()`, `write_mlayer_info()`,
   `write_layer_config_record_payload()` and
   `write_operating_point_set_payload()` (and the now-unused helpers
   `sorted_layers`, `find_embedded`, `xlayer_dependency_mask`,
   `validate_ops_maps`, `append_leb128_bytes`), and keep the public signatures
   as thin forwarders:

   ```cpp
   bool write_layer_config_record_payload(
       const layers::LayerConfigRecord& record,
       const layers::LayerDependencyMaps& dependencies,
       std::vector<std::uint8_t>& out, std::string& error) {
     return ops_full::write_lcr_payload(
         ops_full::lcr_from_legacy(record, dependencies), out, error);
   }

   bool write_operating_point_set_payload(const layers::OperatingPointSet& set,
                                          std::vector<std::uint8_t>& out,
                                          std::string& error) {
     return ops_full::write_ops_payload(ops_full::ops_from_legacy(set), out,
                                        error);
   }
   ```

   After this step the unit suite and the existing `svc_test` must still pass
   unchanged: the bytes are identical.
3. To *enable* the new fields, extend `svc_config` (`build_lcr` / `build_ops`)
   with the optional records and pass `include_full_info = true`, or build the
   `ops_full` structs directly. Two preconditions:
   * global LCR: `config.seq_lcr_id` is already constrained to 1..7 for
     non-single-picture plans (`svc_config.cpp`, `validate_config`), which is
     exactly the `lcr_global_config_record_id` range the new writer enforces;
   * local LCR: `svc_framing` has no separate local-id field (it reuses
     `global_config_record_id`). Set `LocalLcrInfo::local_id` explicitly to
     1..7 before calling `write_lcr_payload`; `lcr_from_legacy` keeps the old
     mapping so the skeleton stays byte-identical.
4. Insert validation in `compose_layered_stream()` right before the LCR/OPS
   OBUs are framed:

   ```cpp
   ops_full::StreamContext context;
   // one XLayerSequenceInfo per spatial layer, from plan.sequences[i]
   context.sequences = ...;
   context.dependencies = plan.sequences.front().dependencies;
   context.known_ops.push_back(ops_full::summarize_ops(plan.ops));
   context.lcr_active = true;
   context.active_global_lcr_id = plan.lcr.global_config_record_id;
   ops_full::ValidationReport report;
   if (!ops_full::validate_stream(plan.lcr, plan.ops, context, report)) {
     return fail(error, report.errors.front());   // or join them
   }
   ```

   `XLayerSequenceInfo` is filled from `layers::SequenceLayerConfig`
   (`seq_profile_idc`/`seq_level_idx`/`seq_tier` come from the sequence header
   writer that produced the layer input; `max_frame_width/height` from
   `SpatialLayerSpec`; `bit_depth_idc`/`chroma_format_idc`/crop fields from the
   same header). Passing only the plan is not enough — the checker must see
   the values actually written.
5. Keep `rewrite_sequence_lcr_id()` as is: it performs the `seq_lcr_id`
   association that activates the LCR, and it rejects the single-picture case
   that this module cannot fix.

## BRT validation hook

`§6.11` requires, for `br_ops_dependent_flag == 1`:

* `br_ops_id` equals an `ops_id` present in the bitstream; and
* `br_ops_cnt[br_ops_id]` equals `ops_cnt[obu_xlayer_id][br_ops_id]`, with the
  i-th BRT entry corresponding to the i-th operating point.

`validate_brt_against_ops()` implements this for the whole temporal unit. Wire
it where the BRT OBU is emitted, i.e. **before** framing the type-15 OBU (the
`svc` component has no BRT writer yet; `src/encoder/rc/hrd/hrd.cpp` does):

```cpp
#include "rc/hrd/hrd.h"          // BufferRemovalTiming
#include "svc/ops_full/lcr_ops_full.h"

ops_full::BrtCorrespondence brt;
brt.br_ops_dependent_flag = timing.br_ops_dependent_flag;
brt.br_ops_id   = timing.br_ops_id;
brt.br_ops_cnt  = static_cast<std::uint8_t>(timing.op_times.size());
brt.br_time     = timing.br_time;
for (const hrd::BrOpTime& entry : timing.op_times) {
  if (entry.br_decoder_model_present_op_flag) brt.br_time_op.push_back(entry.br_time);
}

ops_full::ValidationReport report;
if (!ops_full::validate_brt_against_ops(brt, known_ops, report)) {
  // refuse to emit the BRT OBU
}
```

`known_ops` is every OPS already retained for the temporal unit
(`ops_full::summarize_ops()` of each `OperatingPointSet`, or the parsed
`svc_framing` OPS). Two notes:

* the BRT OBU does not carry `obu_xlayer_id`, so an `ops_id` that appears in
  more than one xlayer with **different** `ops_cnt` is unresolvable; the
  validator rejects it with an "ambiguous" error rather than guessing. Emit one
  global OPS (`GLOBAL_XLAYER_ID`) to keep the correspondence unambiguous;
* `src/encoder/rc/hrd/hrd.cpp` already has
  `hrd::validate_brt_against_ops(brt, brt_xlayer_id, ops, error)` for the
  single-xlayer case. Use the `ops_full` version when the stream is layered
  (it takes the whole `known_ops` span and enforces the BRT entry count and
  `rg(4)` range). A stream that satisfies the `ops_full` check also satisfies
  the `hrd` one for every candidate xlayer; the reverse is not true when the
  stream has several xlayers.

## Call sites

Today (spec-only, no production wiring):

| Call site | What it uses |
|---|---|
| `lcr_ops_full_test.cpp` | every writer, reader, validator, both conversion functions, `svc_framing` cross-check |
| `lcr_ops_mux.cpp` (gate tool) | `write_lcr_payload`, `write_ops_payload`, `validate_stream`, `validate_lcr`, `validate_local_lcr_matches_global`, `summarize_ops` |
| `gate_avmdec.sh` | builds/runs the tool against `svc_mux` + `video_gate`, drives `avmdec` |

Planned (the merge above): `svc_framing::compose_layered_stream()` becomes the
first production caller; `svc_mux` inherits it through `compose_layered_stream`;
the BRT hook lands in whatever emits `OBU_BUFFER_REMOVAL_TIMING`.

## Audit rows this closes

| Audit row (`docs/spec-audit/01-obus.md` §4.3/§4.5) | Before | Now |
|---|---|---|
| `lcr_seq_profile_tier_level_info()` | ABSENT | writer + reader + goldens + validator |
| `lcr_xlayer_color_info()` | ABSENT | writer + reader + goldens + validation vs sequence header |
| `lcr_xlayer_purpose_id` | PARTIAL (0 only) | writer/reader, 0..127, Table 6.6 values |
| `lcr_aggregate_info()` | PARTIAL (present only) | full writer/reader + aggregate-dominance check |
| atlas ids (`global`/`local`/`xlayer`/`embedded`) | PARTIAL/ABSENT | all five coded; `lcr_global_atlas_id`/`lcr_local_atlas_id` f(3) range-checked |
| embedded-layer type/aux/view/priority/rendering | PARTIAL | AUX_LAYER, aux type, all 5 view types, explicit view id, atlas priority/rendering, per-j dep map |
| reserved bits / `lcr_remaining_payload_bit` | PARTIAL | emitted as 0, parsed as padding, preserved on rewrite, size-checked |
| `ops_aggregate_info()` | ABSENT | writer/reader + dominance check |
| `ops_seq_profile_tier_level_info()` | ABSENT | writer/reader + PTL bounds |
| `ops_decoder_model_info_for_this_op_present_flag` + info | PARTIAL (0 only) | per-point presence + delays + low-delay flag |
| idc-2 inheritance | ABSENT | idc 2 writer/reader, conformance bounds, resolver, decoder-gated |
| `ops_color_info()` | ABSENT | writer/reader, Table 6.13 idc 0..3 |
| `ops_initial_display_delay_minus_1` | PARTIAL (0 only) | per-point f(4) |
| `ops_cnt == 0` reset forms | ABSENT | both forms writer/reader |
| BRT ↔ OPS correspondence | COMPONENT (`hrd` single-xlayer) | layered checker + hook |
| PTL/level consistency, layer-map closure | not checked | `validate_lcr`/`validate_ops`/`validate_stream` |

Still open after this work (out of scope here): no atlas segment OBU, no MSDO,
no film grain OBU, `seq_lcr_id` association only for non-single-picture
headers, and no CLI surface.

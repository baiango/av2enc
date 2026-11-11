# `src/encoder/metadata/gaps/` — metadata header gaps and §7.23 reference state

Spec-only module. It fills the metadata-OBU header gaps left by
`src/encoder/metadata/types/` and adds the §7.23 reference-state save/restore
pairs that the encoder does not model anywhere.

Normative reference: `spec/AV2_Spec_v1.0.0.pdf` (v1.0.0, commit `38f28e7`).

| File | Contents |
|---|---|
| `metadata_gaps.h/.cpp` | §5.17.2/§5.17.3 writers + reader + temporal-unit scanner for cancel, `LAYER_VALUES` maps and header-extension bytes |
| `reference_state.h/.cpp` | §7.23 `SavedMvs`, `SavedRefFrames`, `SavedGmParams`, `SavedSegmentIds`, `save/load_grain_params`, `save/load_ccso_params` |
| `metadata_gaps_test.cpp` | pinned goldens, round-trips and invalid-input rejection for the metadata gap surface |
| `reference_state_test.cpp` | exhaustive save/load for every slot, plane and array |
| `gap_stream_gate.cpp` | inserts gap metadata into a valid M2 IVF and self-parses each variant |
| `run_gap_gate.sh` | builds everything, obtains an M2 stream, runs both tests, the tool and the pinned dav2d decode-invariance check |
| `build.sh` | standalone build (there is deliberately no CMake target; this agent owns only this directory) |

```sh
# build + unit tests
src/encoder/metadata/gaps/build.sh /tmp/gaps
/tmp/gaps/metadata_gaps_test
/tmp/gaps/reference_state_test

# full gate (needs tools/bin/dav2d*)
src/encoder/metadata/gaps/run_gap_gate.sh
```

## 1. Metadata header gaps

### 1.1 What the shared module does not do

`src/encoder/metadata/types/` writes canonical units and the short/group OBU
shells, but:

* `make_group_obu` rejects `muh_layer_idc == LAYER_VALUES`;
* no writer emits `muh_header_extension_byte`;
* `hdr::parse_metadata_group` parses the layer maps and then discards them
  (`MetadataUnitView` has no map fields);
* `hdr::MetadataObuConfig` has no OBU header-extension control, so
  `obu_xlayer_id == GLOBAL_XLAYER_ID` (31) — required for the `f(32)` map form —
  cannot be produced.

This module owns exactly those gaps. It reuses `hdr::parse_obu`,
`hdr::insert_metadata_obus` and the unit codecs; for a group unit with no maps
and no extension bytes its writer is byte-identical to
`hdr::make_metadata_group_obu` (pinned by `plain_group_matches_hdr_writer`).

### 1.2 Writers

```cpp
struct LayerValueMaps {
  std::uint32_t xlayer_map = 0;          // §5.17.3 muh_xlayer_map f(32)
  std::vector<std::uint8_t> mlayer_map;  // f(8) per set xlayer bit
};

struct MetadataUnitHeader {
  Type type;
  bool cancel;                            // muh_cancel_flag
  hdr::MetadataLayer layer;               // muh_layer_idc
  hdr::MetadataPersistence persistence;   // muh_persistence_idc
  std::uint8_t priority;                  // muh_priority f(8)
  LayerValueMaps layer_values;            // LAYER_VALUES only
  std::vector<std::uint8_t> header_extension;  // muh_header_extension_byte
};

struct MetadataGapUnit { MetadataUnitHeader header; std::vector<uint8_t> payload; };

std::vector<std::uint8_t> make_short_obu(const MetadataGapUnit&, const GapObuConfig&, std::string&);
std::vector<std::uint8_t> make_short_cancel_obu(Type, const GapObuConfig&, std::string&);
std::vector<std::uint8_t> make_group_obu(std::span<const MetadataGapUnit>, const GapObuConfig&, std::string&);
```

`GapObuConfig` carries only OBU-level fields: `suffix`, `necessity`,
`application`, `tlayer_id`, `mlayer_id`, `xlayer_id`. Layer/persistence/priority
are per metadata unit because §5.17.3 codes them per unit. The OBU header
extension byte is emitted iff `mlayer_id != 0 || xlayer_id != 0` (§5.2.2).

Cancel semantics:

* short form: `muh_layer_idc`, `muh_cancel_flag` and `muh_persistence_idc` are
  in the options byte; the `metadata_type` is still coded; no body follows
  (§5.17.2, §6.16.2). `make_short_cancel_obu()` is the convenience form.
* group form: `muh_cancel_flag == 1` skips the payload size, layer fields and
  layer maps; only `muh_header_extension_byte` may follow (§5.17.3). Those
  non-coded fields are reported at their defaults by the reader.

`muh_layer_idc == LAYER_VALUES`:

* `obu_xlayer_id == 31` (`GLOBAL_XLAYER_ID`): `muh_xlayer_map f(32)`, then one
  `muh_mlayer_map f(8)` for every set bit 0..30, in bit order. Bit 31 must be
  zero.
* otherwise: exactly one `muh_mlayer_map f(8)` for the current xlayer, and no
  `muh_xlayer_map`.
* every map clears the bits below `obu_mlayer_id` (§6.16.3).
* `LAYER_VALUES` is not legal in `OBU_METADATA_SHORT` (`muh_layer_idc < 3`).

`muh_header_extension_byte`: opaque bytes after the layer maps, counted by
`muh_header_size` (`f(7)`, so the total header is ≤ 127 bytes). For a cancel
unit they are the only header payload.

### 1.3 Readers

`parse_short_obu()` / `parse_group_obu()` consume a complete Annex B.2 OBU
(leb128 size prefix + `obu_header()` + payload), verify the OBU type and
trailing bits, and expose:

* `ParsedUnit::header` — type, cancel, layer, persistence, priority,
  `layer_values` maps and `header_extension` bytes;
* `ParsedUnit::payload` — a span into the input (empty when cancelled);
* `ParsedGroupObu` — suffix/necessity/application and tlayer/mlayer/xlayer.

`scan_gap_metadata()` walks a whole temporal unit and returns one `GapRecord`
per unit (including cancels) with the decoded value when the type is known.

### 1.4 Pinned goldens

Hand-derived, field by field, in `metadata_gaps_test.cpp`:

| Case | Bytes |
|---|---|
| short cancel, TIMEcode, current layer, prefix | `0420280480` |
| short cancel, HDR_MDCV, global layer, no persistence, suffix | `04209a0280` |
| group `LAYER_VALUES` local xlayer, CLL, priority 0x5A | `0e24230001080465680203e8019080` |
| group `LAYER_VALUES` `GLOBAL_XLAYER_ID`, map 0b101 | `14a41f2300011204656800000005010403e8019080` |
| group header extension `AA BB` | `10242300010c04656802aabb03e8019080` |
| group cancel with extension `11 22` | `082423000405112280` |
| short OBU header extension (xlayer 5, mlayer 1) | `09a025200103e8019080` |

`0x0e2423...` decodes as: OBU size 14, header `0x24`
(`obu_type = OBU_METADATA_GROUP`), options `0x23` (NECESSARY/TV), one unit,
type 1 (CLL), `muh_header_size` 4, payload 4, layer/persistence/priority bytes
`65 68`, `muh_mlayer_map 0x02`, CLL `03 e8 01 90`, trailing `80`.

### 1.5 Invalid input rejected

Writer-side (unit tests):

* short OBU with `LAYER_VALUES`, layer maps or header-extension bytes;
* `LAYER_VALUES` without a map byte, with the wrong number of map bytes for the
  global form, or with a non-zero `muh_xlayer_map` outside the global form;
* `muh_xlayer_map` bit 31 set; a map bit below `obu_mlayer_id`;
* a cancel unit with a payload;
* unknown/reserved `metadata_type`, non-canonical or empty unit payloads;
* reserved `muh_layer_idc`/`muh_persistence_idc`, reserved application ids 6..15;
* `muh_header_size > 127`; OBU header fields that do not fit their widths;
* empty groups.

Reader-side (crafted bytes):

* wrong OBU type, truncated OBU, bytes after the OBU;
* corrupted trailing bits (both forms);
* short-form `muh_layer_idc == 3`;
* group `muh_reserved_zero_2bits != 0`;
* `muh_header_size` smaller than the fixed fields or larger than the payload;
* truncated `muh_mlayer_map`; `muh_xlayer_map` bit 31 set;
* reserved persistence values.

### 1.6 Gate

`run_gap_gate.sh`:

1. builds the module and runs both unit binaries;
2. obtains a valid 2-frame M2 stream (`src/encoder/m2/build_and_test.sh`) unless
   `--ivf` is given, and decodes it with pinned dav2d `--strict 1`;
3. asks `gap_stream_gate` to insert:
   * a short cancel + a group cancel with an extension byte (prefix, frame 0);
   * a local-xlayer `LAYER_VALUES` unit (prefix, frame 0);
   * a `GLOBAL_XLAYER_ID` `LAYER_VALUES` unit with a 32-bit map (prefix, frame 0);
   * a suffix group with a cancel and a `LAYER_VALUES` unit plus extensions
     (frame 1);
   and to self-parse every variant, including one deliberately corrupted
   trailing-bits variant that the reader must reject;
4. checks that only the intended IVF frame payload grew (header, timestamps and
   other frames unchanged) and that every variant decodes to exactly the
   baseline Y4M md5.

The pinned dav2d (`f9dafbc`) does not implement AV2 §5.17.2/§5.17.3; it logs
`Unknown Metadata OBU type ...` / `Unknown OBU type 9 of size ...` and skips the
OBU by length. The gate therefore proves decode-invariance plus self-parse, not
decoder-side metadata validation. This is the same contract as the existing
`metadata/types/run_metadata_gate.sh`.

Measured on the 64x64 M2 clip:

```
cancel_prefix         frame 0 573 -> 586 bytes
layer_local_prefix    frame 0 573 -> 588 bytes
layer_global_prefix   frame 0 573 -> 594 bytes
layer_values_suffix   frame 1 555 -> 574 bytes
baseline and all variants: md5 2da6c81f1b4a0333fdd742d7291af895
```

## 2. §7.23 reference-state save/restore

`ReferenceStateStore` is a pure state container. It does not write syntax and
does not change a decision; it gives the reference-update path a tested place
to perform the §7.23 saves and later readers (`§7.22`, §5.18.7.12 reuse) the
matching loads.

### 2.1 Shapes and index domains

| Spec object | C++ type | Shape |
|---|---|---|
| `SavedMvs[i][y8][x8][list][comp]` | `SavedMvs` | `rows = MiRows>>1`, `cols = MiCols>>1`, `units[y8*cols+x8].list[list]` is a `MotionVector{row,col}` |
| `SavedRefFrames[i][y8][x8][list]` | `SavedRefFrames` | same 8×8 grid, `units[...].list[list]` is an `int16` ref index |
| `SavedGmParams[i][ref][j]` | `SavedGmParams` | fixed `REFS_PER_FRAME (7) × 6` `int32` params |
| `SavedSegmentIds[i][row][col]` | `SavedSegmentIds` | luma MI grid `MiRows × MiCols`, ids ≤ 15 |
| `save_grain_params(i)` | `GrainParams` | per-reference `FilmGrainConfig` + `FilmGrainModel` preserving every §5.18.10 syntax element (including point-curve widths and AR coefficient widths) |
| `save_ccso_params(i, plane)` | `CcsoPlaneParams` | `CcsoLumaSizeLog2`, the six scalars, the 576-entry `CcsoFilterOffset` table and the `CcsoBlks` grid, per `(i, plane)` |

Constants are the spec's §3 values: `NUM_REF_FRAMES = 16`,
`REFS_PER_FRAME = 7`, `MAX_FILM_GRAIN = 8`.

The grain store has `NUM_REF_FRAMES + 1` entries because §7.21.2 calls
`load_grain_params(NUM_REF_FRAMES)` for mixed output. The other stores are
`NUM_REF_FRAMES` entries.

Saves are deep copies and validate first. In particular:

* `SavedMvs`/`SavedRefFrames` must have `rows*cols` entries and share a grid;
* `SavedSegmentIds` entries must be ≤ 15;
* in a `ReferenceSlot`, the motion grid must be exactly
  `(MiRows>>1, MiCols>>1)` of the segment grid;
* grain point/AR-coefficient counts must match `num_*_points` and
  `ar_coeff_lag`, absent syntax must stay at its default, and every coded field
  must fit its `f(n)` width;
* a disabled CCSO plane keeps every coded field at its default; an enabled
  plane must fit the §5.18.7.12 widths (`edge_clf` is 1 bit, `ext_filter` 3,
  `quant_idx`/`scale_idx` 2, `max_band_log2` 2 or 3, luma size log2 6..8).

A failed save leaves any previously stored value untouched. Loading an index
that was never saved returns `false`.

Related existing components (kept separate on purpose):

* `src/encoder/filters/ccso_reuse/` already implements the §5.18.7.12
  syntax-level CCSO store and the `reuse_ccso`/`sb_reuse_ccso` decision. The
  gaps store is the uniform §7.23 snapshot; when wiring, keep its
  `(i, plane)` entries consistent with `ccso_reuse::SlotStore` (same
  `CcsoLumaSizeLog2`, scalars, offsets and block grid). `resolve_ccso_params()`
  is the function to call after `read_ccso_params()`.
* `src/encoder/filters/spec_tables/film_grain.h` has the synthesis-side
  `FilmGrainModel`; it drops the point/coefficient width fields that
  `save_grain_params` must preserve. The gaps `FilmGrainModel` is the lossless
  store view.

### 2.2 Call sites in the reference update path

§7.23 (p576–577) runs once per decoded frame, at the end of the frame. For each
slot `i` with bit `i` of `refresh_frame_flags` set, after
`RefValid`…`RefNumTotalRefs`, `SavedOrderHints` and `FrameStore` are updated and
**before** `save_cdfs(i)`:

1. copy `MfRefFrames`/`MfMvs` into `SavedMvs`/`SavedRefFrames`
   (`save_reference`, or the individual `save_mvs`/`save_ref_frames`);
2. copy `gm_params` into `SavedGmParams` (`save_gm_params`);
3. copy `SegmentIds` into `SavedSegmentIds` (`save_segment_ids`);
4. if `film_grain_params_present == 1`:
   a. `load_grain_params(NUM_REF_FRAMES)` — the step before saving;
   b. `save_grain_params(i)` (`save_grain_params`);
   c. `save_ccso_params(i, plane)` for plane 0..2 (`save_ccso_params`).

Concrete encoder locations:

| Location | Role | Insertion point |
|---|---|---|
| `adv::refresh_reference_slots()` — `src/encoder/inter/advanced/b_frame.cpp:61-89` | §7.23 slot loop used by the b-frame planner | after the per-slot fields are updated, once per refreshed `i`. Motion field / GM / segment arrays come from the decision stage. |
| `bframes::simulate_decode()` — `src/encoder/inter/bframes/bframes.cpp:737-766` | model of the full §7.23 order, including output-before-overwrite | already calls `refresh_reference_slots`; add the saves beside it, after the output-eligibility step and before the new frame is stored. `output.frame_buffers` is the §7.21.6 load-side consumer. |
| `m2::encode_m2_sequence()` — `src/encoder/m2/m2_frame.cpp:999-1055` | production (trivial) update: key refreshes all slots, P refreshes slot 0 | production has no motion field, GM or segment maps, so a save would pass empty/zero snapshots. Do **not** call the grain/CCSO saves until the sequence/frame writers signal `film_grain_params_present` or `enable_ccso`; the default M2 stream sets both off. |
| `sef::apply_refresh()` — `src/encoder/inter/sef/sef.cpp:347-371` | SEF slot refresh | same per-slot insertion; the SEF writer emits `refresh_frame_flags = 0` in its current form, so no save fires. |
| `inter/longterm/longterm_pool.cpp` | long-term retention / OLK invalidation | whenever a slot is invalidated (`RefValid[i] = 0`), drop the matching `ReferenceSlot`; a slot that is re-used must not expose a stale motion field or film-grain model. `ReferenceStateStore::clear()` / per-index overwrite is the mechanism. |
| §7.21.2 intermediate output preparation — p561-562 | output path | when `RefFilmGrainPresent[frameToShowMapIdx] == 1`, call `load_grain_params(mixedOutput ? NUM_REF_FRAMES : frameToShowMapIdx)` before applying grain. |
| §5.18.7.12 `ccso_params()` — p134 | reuse path | when `reuse_ccso[plane] || sb_reuse_ccso[plane]`, resolve `ccso_ref_idx[plane]` to a slot and call `load_ccso_params(idx, plane)`. `ccso_reuse::resolve_ccso_params()` implements the same step. |
| §7.22 `store_motion_field_entry()` / projection — `src/encoder/inter/temporal/motion_field.cpp` | motion-field consumer | `SavedMvs`/`SavedRefFrames` are the storage the projection reads once `enable_ref_frame_mvs` can be non-zero. |

Ordering note: the spec saves the four `Saved*` arrays for every refreshed slot
before the conditional grain/CCSO block. Keep that order so a failure cannot
leave grain/CCSO state for a slot whose motion field was never saved.

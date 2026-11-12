# Spec audit 05 — §7.1–§7.8: decoding, OBU ordering, random access, CDF, ref lists

Scope: AV2 v1.0.0 §7.1–§7.8 (spec pages **375–401**): general decoding, frame
wrapup, OBU ordering and availability (§7.3.x), random access (CLK / OLK / RAS /
multistream), frame-end CDF update, extended-layer context, and reference-frame
selection (§7.7) plus past/future/current lists (§7.8).

Normative text read from `spec/AV2_Spec_v1.0.0.pdf` (v1.0.0, 2026-05-28, commit
`38f28e7`), pypdf, page index = spec page − 1. Code read at commit `1fdf58b`
(2026-11-12). Read-only audit; no source edits.

## Status vocabulary

| Status | Meaning |
|---|---|
| **WIRED** | Emitted/guaranteed by the encoder core streams (M0/M1 still path that `src/main.cpp` drives, or the M2 frame writer that produces the gate streams). The rule is satisfied by construction, even when only one frame lives in a temporal unit. |
| **COMPONENT** | A standalone module implements and tests the rule (usually with its own `dav2d` gate), but no M0/M1/M2 core code calls it and the relevant sequence/frame flag is hard-coded off. |
| **PARTIAL** | Some of the rule is encoded or checked; a branch that the syntax can reach (or is one header flag away from reachable) is missing. |
| **ABSENT** | No encoder-side implementation anywhere in `src/`. |
| **DECODER-ONLY** | The clause constrains the decoder/reference output only; for the syntax this encoder emits there is no bitstream obligation. |

## Baseline facts that decide most statuses

* The shipped CLI encodes **one still picture**: `src/main.cpp:122,135` accepts
  `--frames N` but clamps it to 1. M0 (`assemble_basic_encoder_picture`) and M1
  (`assemble_still_picture`) emit exactly `TD + SEQ + CLK`
  (`src/bitstream/basic_headers.cpp:356-369`, `src/encoder/m1/still_syntax.cpp:284-304`).
* The multi-frame stream is `av2enc::video::encode_video_sequence()`
  (`src/encoder/m2/video_frame.cpp:929`). It is **not** in `CMakeLists.txt` and is
  not reachable from the CLI; it is used by gates. It emits one temporal unit
  per picture: `TD + [SEQ on frame 0] + frame OBU`
  (`video_frame.cpp:1005-1015,1044-1050`).
* M2 is IPPP, `num_ref_frames = 2`, `order_hint = frame index + 0`,
  `monotonic_output_order_flag = 1`, `long_term_frame_id_bits = 0`,
  `enable_avg_cdf = 0`, `disable_cdf_update = 1`,
  `disable_cross_frame_cdf_init = 1`, one tile, one tile group per frame
  (`src/encoder/m2/video_headers.cpp:110,163,200,224-315,321`).
* Every tile-group writer emits exactly one tile group with
  `is_first_tile_group = 1`; the tile-group planner explicitly does not model
  partial groups (`src/encoder/tiles/tile_group.h:70-76`). `is_first_tile_group
  = 0` is never produced.
* All inter/RA machinery outside `m2/` is self-described as **component only**
  (`inter/longterm`, `inter/random_access`, `inter/leading`, `inter/sef`,
  `inter/tlayers`, `inter/bframes`, `frame/mfh`, `svc`, `metadata/content_interpretation`,
  `rc/hrd`, `entropy/avg_cdf`). Their `VIDEO_INTEGRATION.md` files list the wiring
  that has not happened.

---

## 1. §7.1 General decoding and §7.2 Decode frame wrapup

| Rule | §/page | Status | Encoder obligation | Evidence | Gap |
|---|---|---|---|---|---|
| `R1` Output frames identical in all respects and in the same order as the reference process; with `film_grain_params_present = 1`, one of the two §7.1 conformance options (exact grain synthesis, or intermediate frames exact + perceptually equivalent grain) | §7.1 p375 | **DECODER-ONLY** | None on the entropy stream; the encoder only has to make the decoder's job possible (crop, grain OBU, flags). | M0/M1/M2 write `seq_cropping_window_present_flag = 0` and `film_grain_params_present = 0` (`video_headers.cpp:125,219`; `basic_headers.cpp:202,268`; `still_headers.cpp:100,159`). | No crop-window and no grain signalling; only the trivial "output = frame, no grain" case is expressible. |
| `R2` General process: parse every OBU in turn, then `flush_implicit_output_frames(0)` at the end | §7.1 p376 | **DECODER-ONLY** | None. | n/a | — |
| `R3` A singlestream can be decoded directly by the general process | §7.1 p376 | **WIRED** | Every emitted stream must be a valid singlestream. | M0/M1 gates decode the still stream with `dav2d`; `src/encoder/m2/build_and_test.sh` decodes 2- and 8-frame 720p; all compare recon exactly. | — |
| `R4` A multistream can be decoded per stream by concatenating `obu_xlayer_id == GLOBAL_XLAYER_ID` OBUs (except MSDO) with the stream's own xlayer OBUs | §7.1 p376 | **COMPONENT** | If xlayers are used, every OBU must be attributable to exactly one stream + the global bucket. | `svc::compose_layered_stream()` re-headers per xlayer and `obu_retained()` implements the Annex F.3.2 split (`src/encoder/svc/svc_framing.cpp:608-745,747-790`); `gate_dav2d.sh` decodes composed 2/3-layer streams and `avmdec --select-ops` extracts. | Not wired into the encoder core; no MSDO form. |
| `R5` `TipFrameMode == TIP_FRAME_AS_OUTPUT` wrapup: deblock-for-TIP, `LrFrame = CurrFrame`, `bru_inactive` copies `FrameStore[ref_frame_idx[bru_ref]]`, `MfRefFrames/MfMvs` filled with `bru_ref`/0 | §7.2 p376-377 | **DECODER-ONLY** | None: `enable_tip = 0`, `enable_bru = 0`, `enable_df_sub_pu = 0` in every sequence header we write (`video_headers.cpp:169-180`; the still header omits the whole inter config, `still_headers.cpp:122-124`). | Branch unreachable. | — |
| `R6` Save the film-grain model of every frame into the `NUM_REF_FRAMES` slot (needed because reference reload can re-apply grain) | §7.2 p377 | **DECODER-ONLY** | Would be an encoder obligation only with `apply_grain = 1`. | No film-grain OBU writer in `src/`; only the synthesis model in `filters/spec_tables/film_grain.cpp`. | Grain is unreachable; §7.3.8.8 also unmet (see `R47`). |
| `R7` Invoke the §7.23 reference update and decide which frames to output | §7.2 p377 | **DECODER-ONLY** | The encoder must mirror the slot state to choose valid references. | M2 always refreshes/reads slot 0 (`video_frame.cpp:1041-1042`); `longterm::apply_frame()` replicates §5.18.2 + §7.23 including the `first` rule (`longterm_pool.cpp:244-400`). | Slot mirroring exists only in components; M2 has one slot. |
| `R8` Post-processing order: deblock → CDEF → CCSO → loop restoration; segmentation-map reuse | §7.2 p377 | **WIRED** | The encoder's reconstruction must use the same filters, or its prediction drifts from the decoder. | M1 applies deblock `still_encoder.cpp:1510`, CDEF `:1651`, CCSO `:1788`; `enable_restoration = 0` (`still_headers.cpp:148`), which makes the LR step a no-op. | Loop restoration is not in the M1 pipeline; deblock/CDEF/CCSO are gated bit-exact. |
| `R9` `segmentation_enabled == 1 && segmentation_update_map == 0` ⇒ `SegmentIds = PrevSegmentIds` | §7.2 p377 | **DECODER-ONLY** | None: every writer codes `segmentation_enabled = 0` (`basic_headers.cpp:309`, `still_headers.cpp:176`, `video_headers.cpp:283`). | — | Segmentation is never signalled. |
| `R10` `use_bru == 1` ⇒ `bru_region_valid() == 1` (no inactive block inside a merged active rectangle, and each rectangle border is off-screen or `BRU_SUPPORT`) | §7.2 p377-378 | **ABSENT** | If BRU is enabled, the map must be valid. | `use_bru` is only coded when `enable_bru == 1` (p102); M2 hard-codes `enable_bru = 0` (`video_headers.cpp:177`) and the still header forces the whole inter config (`still_headers.cpp:122-124`). Only filter-side BRU *consumers* exist (`filters/restoration/*`, `inter/temporal/motion_field.cpp`). | No `bru_mode`/`bru_ref`/`bru_inactive` writer and no region check; `bru_region_valid()` is not implemented. |
| `R11` `ShowExistingFrame == 1` ⇒ skip filtering/§7.23, call `output_frame_buffers(derive_sef_order_hint ? frame_to_show_map_idx : -1)`; SEF sets `refresh_frame_flags = 0`, `immediate_output_frame = 1`, `OrderHint` from the map index or `sef_order_hint` | §7.2 p377 | **COMPONENT** | SEF syntax + output-order bookkeeping. | `src/encoder/inter/sef/sef.cpp:190-233` (`make_sef_obu`, `make_sef_temporal_unit`), `sef.h:33-47`; `dav2d --strict 1` gate in `sef_gate.cpp`. | No M2 core SEF path; no leading/regular SEF mixed with real frames. |

---

## 2. §7.3.1–§7.3.2 General ordering and multistream boundaries

| Rule | §/page | Status | Encoder obligation | Evidence | Gap |
|---|---|---|---|---|---|
| `R12` A bitstream is ≥1 coded video sequence; a temporal unit contains at least one coded output frame unit belonging to one coded extended layer unit | §7.3.1 p379 | **WIRED** | Every TU must carry an output frame. | M2 writes one output P/CLK per TU (`video_frame.cpp:1044-1050`); `longterm::validate_temporal_unit()` rejects a TU with zero output units (`longterm_pool.cpp:449-457`). | — |
| `R13` A coded multistream video sequence has OBUs with ≥2 distinct non-global `obu_xlayer_id` values | §7.3.1 p379 | **COMPONENT** | Multistream needs ≥2 layers. | `svc::compose_layered_stream()` takes one single-layer input per spatial layer and the gate composes 2- and 3-layer streams (`svc/gate_dav2d.sh`); `build_layer_plan()` accepts 1..30 spatial layers (`svc_config.cpp:56-58`). | The ≥2-distinct-xlayer rule itself is not validated (a one-layer plan degenerates to a legal singlestream); core streams are single-xlayer. |
| `R14` An MSDO OBU or an activated global LCR is present; when MSDO is present it is present in **each** TU that contains a random access point; a non-RAP MSDO is identical to the previous MSDO | §7.3.1 p379, §7.3.8.2 p386 | **PARTIAL** | MSDO writer + per-RAP repetition + identity across non-RAP TUs. | `svc_framing.h:69` only names MSDO in the OBU-type enum; `encode_obu_header()` supports it, but nothing ever emits or validates one. `svc` writes a global LCR (`svc_framing.cpp:444-470`). | No MSDO OBU syntax/fields, no RAP-presence check, no "identical to previous" check. |
| `R15` Every OBU with `obu_xlayer_id != GLOBAL_XLAYER_ID` must name a value in the preceding MSDO `sub_xlayer_id` or in the activated LCR `LcrXLayerID` | §7.3.1 p379 | **COMPONENT** | Layer ids in the stream must be declared by MSDO/LCR. | `svc::compose_layered_stream()` assigns xlayer ids from the plan; `validate_layer_plan()` requires `lcr.xlayer_map == (1 << spatial_count) - 1` with one LCR layer entry per spatial layer (`svc_config.cpp:315-324`), and the LCR writer rejects a map entry with no matching layer (`svc_framing.cpp:430-444`). | Only checked inside `svc`; no check of an arbitrary input bitstream. |
| `R16` All extended layers within a temporal unit share the same output time | §7.3.1 p379 | **ABSENT** | Same presentation time per TU. | `svc::compose_layered_stream()` merges whatever layer units the caller passes; it never compares timing. No per-TU output-time field exists in the layer plan or in `compose_layered_stream()`; HRD timing is a separate component (`rc/hrd`). | No output-time model at all. |
| `R17` Coded extended layer units from different extended layers within a TU appear in ascending `obu_xlayer_id` order | §7.3.1 p379, §7.3.7 p385 | **COMPONENT** | Emit global prefixes, then xlayer units sorted ascending. | `svc::compose_layered_stream()` iterates inputs in `spatial_layers` order (`svc_framing.cpp:729-747`); `plan_emission_order()` sorts `(TU, xlayer, mlayer, tlayer, non-output before output)` (`svc_pipeline.cpp:39-87`). | Single-xlayer core streams are trivially ordered; no core check. |
| `R18` The extracted bitstream for each individual stream forms a valid bitstream | §7.3.1 p379 | **COMPONENT** | Each xlayer must decode independently (no illegal cross-layer prediction). | `svc::ReferencePolicy` rejects cross-xlayer references; `gate_dav2d.sh` extracts with `avmdec --select-ops` and decodes. | Not part of the core encode path; cross-xlayer prediction is not implemented. |
| `R19` Coded multistream video sequence boundaries: begin at a TU with CLK for ≥1 extended layer plus MSDO/global-LCR condition; end at a new-MS sequence TU, a CVS-start TU without MSDO/global LCR, or end of bitstream; at the end all frames are output and all reference buffers invalidated; MSDO-derived and LCR-derived boundary sets must be identical | §7.3.2 p380 | **ABSENT** | Validate/emit multistream boundaries. | No boundary state machine anywhere; `svc` treats its input list as the whole sequence. | Even the presence of an MSDO/global LCR is not tied to a boundary. |

---

## 3. §7.3.3–§7.3.5 Coded frame units

| Rule | §/page | Status | Encoder obligation | Evidence | Gap |
|---|---|---|---|---|---|
| `R20` Coded output frame unit presence order: ≤1 `OBU_CONTENT_INTERPRETATION`, ≥0 `OBU_MULTI_FRAME_HEADER`, then in any order `OBU_BUFFER_REMOVAL_TIMING` / `OBU_QUANTIZATION_MATRIX` / `OBU_FILM_GRAIN` / prefix `OBU_METADATA_*`, then the coded-frame OBUs, then suffix metadata; padding anywhere | §7.3.3 p380-382 | **PARTIAL** | The optional OBUs must be emitted in that order, not just individually correct. | Core emits only the frame OBU (a valid subsequence). Inserters exist for each slot: CI at the §7.3.3 position (`content_interpretation.h:345-360`), MFH before the frame (`mfh_gate.cpp:399-409`), QM before the frame (`qm_gate.cpp:351-352`), BRT immediately before the frame (`hrd_stream_gate.cpp:373-381`), prefix/suffix metadata (`hdr_metadata.h:382-391`, `insert_metadata_obus()`). `hrd::validate_temporal_unit_order()` checks TD/SEQ/CI/BRT position (`hrd.cpp:843-935`). | No single assembler composes CI + MFH + QM + FG + metadata + frame, so the relative order of two optional OBUs is never checked. `validate_temporal_unit_order()` rejects anything except the M2 subset, so it cannot validate a full frame unit. |
| `R21` For frame OBUs of types CLK/OLK/LEADING_TILE_GROUP/REGULAR_TILE_GROUP/SWITCH/LEADING_TIP/REGULAR_TIP/RAS, the first encountered OBU has `is_first_tile_group = 1` and all remaining same-type OBUs have 0 | §7.3.3 p381 | **WIRED** | First tile group flag. | Every writer hard-codes `true` (`video_headers.cpp:321`, `still_syntax.cpp:287`, `basic_headers.cpp:332`, `longterm_headers.cpp:451`, `random_access.cpp:1334`, `leading.cpp:514`); one tile group per frame. `tiles::plan_single_tile_group()` does not model partial groups (`tile_group.h:70-76`). | Partial/extra tile groups are unsupported, so the `= 0` branch is never exercised. |
| `R22` A `LEADING_SEF` or `REGULAR_SEF` is one OBU and is itself a coded output frame unit, associated with an `OrderHint` | §7.3.3 p381 | **COMPONENT** | SEF TU = TD + SEF only. | `make_sef_temporal_unit()` (`sef.cpp:220-233`); payload is only `frame_header(1)` and the gate decodes it. | No SEF in the M2 core; M2 only writes CLK/REGULAR_TILE_GROUP (`video_headers.cpp:328-339`). |
| `R23` `OBU_PADDING` may appear at any position within a coded output frame unit | §7.3.3 p382 | **ABSENT** | Padding is optional; only placement matters. | No writer emits `OBU_PADDING`. | No padding support at all (not a conformance failure for current streams). |
| `R24` Coded non-output frame unit: same presence order for frames with `immediate_output_frame == 0` and `implicit_output_frame == 0`; frame types include `OBU_BRIDGE_FRAME`; a non-output frame must refresh ≥1 slot | §7.3.4 p382-383, §6.17.2 p323 | **PARTIAL** | Non-output frames must be grouped/ordered like output frames and refresh a slot. | `longterm` emits an OLK as a non-output frame sharing a TU with an output P frame (`longterm_stream.h:1-30`); `validate_frame_header()` enforces `!immediate ⇒ refresh != 0` (`longterm_pool.cpp:158-165`); `leading` enforces the same (`leading.cpp:419-427`). | No `OBU_BRIDGE_FRAME` writer (type 19 only appears in parsers/enums: `svc_framing.cpp:220`, `hdr_metadata.cpp:813`); no general non-output assembly outside `longterm`. |

§7.3.5 is definitional (a coded frame unit is either a coded output frame unit
or a coded non-output frame unit); it has no separate obligation and is covered
by `R20` (output form) and `R24` (non-output form).

---

## 4. §7.3.6 Coded extended layer unit and §7.3.7 Temporal unit

| Rule | §/page | Status | Encoder obligation | Evidence | Gap |
|---|---|---|---|---|---|
| `R25` Per extended layer unit: ≥0 LCR, ≥0 OPS, ≥0 atlas, ≥0 sequence header, then **for each embedded layer in ascending `obu_mlayer_id`** ≥0 coded non-output frame units followed by **≤1** coded output frame unit; at least one output unit must exist | §7.3.6 p383 | **PARTIAL** | Group frame units by mlayer and order hidden-before-output; exactly one output unit per layer. | `assemble_sequence()` writes LCR/OPS/atlas/SEQ nowhere except `TD + [SEQ] + frames` and puts frames in caller order (`longterm_stream.cpp:83-125`); `validate_temporal_unit()` checks only "≥1 output" and equal OrderHint (`longterm_pool.cpp:419-458`). | Two output frame units in one TU are accepted (spec allows ≤1 per embedded layer). Non-output units are not required to precede the output unit. `obu_mlayer_id` grouping/ascending order is not modelled (`max_mlayer_id = 0` everywhere). |
| `R26` If a non-output frame unit is present in an embedded layer, an output frame unit must also be present in the same layer | §7.3.6 p383 | **WIRED** | Hidden frame must share its TU with a shown frame. | `validate_temporal_unit()` rejects a TU with only non-output frames (`longterm_pool.cpp:449-457`); the gate puts the OLK with a P frame (`longterm_stream.h:10-13`). | Only one embedded layer is expressible, so "same layer" is vacuously the TU. |
| `R27` All coded output frame units in a coded extended layer unit share one `OrderHint` | §7.3.6 p384 | **WIRED** | Output frames in a TU need the same hint. | `validate_temporal_unit()` compares every output unit's hint (`longterm_pool.cpp:428-439`); `validate_sequence` tests cover the reject case. | M2 has one output unit per TU, so the check never fires in the core. |
| `R28` If the unit contains a CLK: only the first coded frame unit in each embedded layer may be CLK, and the first frame unit of the **lowest embedded layer present** must be a CLK | §7.3.6 p383-384 | **PARTIAL** | CLK only first; and a CLK in the lowest layer. | `validate_temporal_unit()` rejects any CLK/OLK at index > 0 (`longterm_pool.cpp:421-424`); `assemble_sequence()` additionally rejects a non-CLK first picture (`longterm_stream.cpp:76-81`). | The "lowest embedded layer" rule is not modelled; it is enforced TU-wide, which is stricter for multi-layer but cannot express the §7.3.9 exception. |
| `R29` If the unit contains an OLK: same first-frame-unit rule, lowest layer must be OLK, and CLK and OLK cannot coexist in one unit | §7.3.6 p384 | **PARTIAL** | OLK first, no CLK+OLK mix. | `validate_temporal_unit()` rejects mix (`:441-443`) and non-first RA (`:421-424`); `leading` has a single-OLK plan with the OLK before leading frames (`leading.cpp:445-465`). | Same lowest-layer limitation; `leading` forbids `monotonic_output_order_flag = 1` with an OLK (`leading.cpp:453-458`), which is stricter than §5.18.2 (an OLK with `implicit_output_frame = 0` is a legal non-output frame). |
| `R30` If a leading frame is present, all coded frame units in the extended layer unit are leading frames | §7.3.6 p384 | **WIRED** | No regular frame may share a TU with a leading frame. | `validate_temporal_unit()` checks `leading > 0 ⇒ leading == frames.size()` (`longterm_pool.cpp:444-448`); `leading::validate_plan()` enforces leading-before-regular in coding order (`leading.cpp:459-470`). | Vacuous for M2 (one frame per TU); the general multi-frame check exists only in `longterm`/`leading`. |
| `R31` CI OBU only in the first coded frame unit of each embedded layer; if present in any unit of a CVS it is also in the first unit of the CVS; contents identical | §7.3.6 p384 | **COMPONENT** | CI placement and constancy. | `insert_content_interpretation()` writes it at the §7.3.3 position; `CvsConstancyTracker::observe()` enforces per-layer constancy and first-TU presence (`content_interpretation.h:388-418`). | Placement is per temporal unit, not per embedded-layer frame unit; no core path emits CI. |
| `R32` If `monotonic_output_order_flag == 0`: no coded output frame unit `Y` later in output order may have `OrderHint(Y) <= OrderHint(X)` unless a SWITCH frame with `restricted_prediction_switch == 1` appears between `X` and `Y` in coding order | §7.3.6 p384 | **ABSENT** | Validate output-order/OrderHint monotonicity for reordered streams. | `tlayers` and `bframes` support `monotonic_output_order_flag = 0` plans (`tlayers_syntax.cpp:427`, `bframes.cpp:681`) but neither checks this relation; M2 always writes 1 (`video_headers.cpp:110`). | A reordered plan that reuses/does not increase hints passes the planners. |
| `R33` When a coded extended layer unit contains a CLK/OLK **and** long-term reference OBUs are needed at a RAS/OLK random access point, the LTR OBUs may break the §7.3.6 order: required CLKs first, then required OLKs, with `immediate_output_frame = 0` and `implicit_output_frame = 0` | §7.3.9 p389 (NOTE on §7.3.6) | **ABSENT** | Emit LTR anchor OBUs in the same unit as the RAP when `long_term_frame_id_bits != 0`. | `validate_temporal_unit()` rejects any RA role at index > 0 (`longterm_pool.cpp:421-424`) and `assemble_sequence()` never emits more than one key OBU per TU. `longterm_headers.cpp:365` comments `RAS, not modelled`. | The one arrangement the spec created for LTR-at-RAP is unreachable. |
| `R34` A new coded video sequence of an extended layer starts at each TU with a CLK; redundant activated sequence headers must be bit-identical; only one sequence header is active until the next CLK; changing parameters requires a new CVS | §7.3.6 p385 | **PARTIAL** | Resend the same SEQ at a mid-stream CLK; never change parameters without a CLK. | `leading::assemble_temporal_units()` repeats the sequence header at CLK and OLK (`leading.cpp:553-560`); `longterm` and `random_access` emit it only in unit 0 (`longterm_stream.cpp:84-93`; `random_access/VIDEO_INTEGRATION.md:151-163`). | A mid-stream CLK (the M2 writer only makes frame 0 a keyframe) or an OLK TU has no SEQ in the bitstream: acceptable only if HLS is supplied "through external means" (§7.3.8.1). No active-SH tracking, no parameter-change check. |
| `R35` `OrderHint` is reset at the start of a new CVS and at a switch frame with `restricted_prediction_switch == 1` | §7.3.6 p384 NOTE | **ABSENT** | Restart the hint counter after those events. | M2 sets `order_hint = i` monotonically and has one CVS; `tlayers` has a per-GOP `start_order_hint` (`tlayers.h:83`) but no reset event. | No reset model; concatenating GOPs reuses increasing hints only because the caller chooses so. |
| `R36` Temporal unit order: exactly one `OBU_TEMPORAL_DELIMITER` (`GLOBAL_XLAYER_ID`), ≤1 MSDO, ≥0 global LCR/OPS/atlas, ≥0 global prefix metadata, then coded extended layer units in ascending xlayer; padding anywhere with `GLOBAL_XLAYER_ID` outside layer units | §7.3.7 p385 | **PARTIAL** | Emit the prefix OBUs in this order. | M2: `TD + [SEQ] + frame` (SEQ is inside the layer unit, correct). `svc`: `TD, LCR, OPS`, then ascending xlayers (`svc_framing.cpp:700-747`). `hrd::validate_temporal_unit_order()` checks TD first and prefix-before-frame (`hrd.cpp:843-935`). | No MSDO, atlas, or metadata in the composed order; no padding. The `svc` order puts LCR/OPS every TU (legal, but the ordering validator does not cover it). |
| `R37` All frame units within a temporal unit use the same `OrderHintBits` | §7.3.7 p385 | **PARTIAL** | One `OrderHintBits` across layers. | Single-layer M2: vacuously true. `svc` builds per-layer sequence headers from one base config (`svc_config.cpp:240-360`) so they should agree. | No cross-layer check; `svc` never compares the per-layer `order_hint_bits_minus_1` fields. |
| `R38` Coded output frame units in multiple coded extended layer units within one TU share the same `OrderHint` | §7.3.7 p385 | **ABSENT** | Cross-layer hint equality. | No code compares per-layer hints; `plan_emission_order()` does not carry an OrderHint at all (`svc_pipeline.cpp:39-87`). | Unchecked (only required when the doh constraints of `R39` are on). |
| `R39` When `lcr_doh_constraint_flag == 1` or `multistream_doh_constraint_flag == 1`, the same `OrderHintBits` and `OrderHint` conditions additionally hold for each TU | §7.3.7 p385, §7.4.6 p395 | **ABSENT** | If the encoder sets a doh constraint flag, it must enforce the matching hints. | `svc_framing.cpp:454` writes `lcr_doh_constraint_flag = 0` unconditionally; there is no MSDO so no `multistream_doh_constraint_flag`. | The constraint cannot be requested and is never enforced. |

---

## 5. §7.3.8 Availability of high-level-syntax OBUs

| Rule | §/page | Status | Encoder obligation | Evidence | Gap |
|---|---|---|---|---|---|
| `R40` HLS OBUs must be available before they are referenced, including when decoding starts at any random access point and drops leading-frame TUs (bitstream inclusion or external means) | §7.3.8.1 p386 | **PARTIAL** | Re-emit every HLS OBU a RAP needs in that RAP's TU. | Only `leading` re-emits the sequence header at CLK/OLK (`leading.cpp:553-560`); `longterm` re-emits it only in unit 0 (`longterm_stream.cpp:84-93`); `random_access` emits it once (`VIDEO_INTEGRATION.md:162`). No pass over the assembled stream checks availability. | No global availability check; mid-stream CLK/OLK/RAS TUs can lack SEQ, LCR, MFH, QM, FG. Today that is only "conforming" if HLS is externally provided, which no stream signals. |
| `R41` MSDO available at each RAP and identical when not at a RAP | §7.3.8.2 p386 | **ABSENT** | MSDO writer + availability. | `svc_framing.h:69` enum only. | See `R14`. |
| `R42` A global LCR must be available before a local LCR's `lcr_global_id` or a sequence header's `seq_lcr_id` references it | §7.3.8.3 p386 | **COMPONENT** | LCR emitted before the referencing SEQ. | M0/M1/M2 write `seq_lcr_id = 0` (`video_headers.cpp:104`), so no reference. `svc` writes a global LCR before the per-layer sequence headers (`svc_framing.cpp:700-747`). | Local LCRs are not written. |
| `R43` Atlas segment availability before an LCR references it | §7.3.8.4 p386-387 | **ABSENT** | Atlas OBU writer. | `kAtlasSegment` exists only in enums (`svc_framing.h:52`). | No atlas syntax or writer. |
| `R44` OPS availability before being referenced | §7.3.8.5 p387 | **COMPONENT** | OPS before use. | `write_operating_point_set_payload()` (`svc_framing.cpp:705-718`) writes a complete global OPS (cumulative per-layer maps) before the layer units. | OPS is emitted every TU in `svc`, which is legal; there is no `ops_id` reference check. |
| `R45` A sequence header must be available before a frame header's `seq_header_id_in_frame_header` or an MFH's `mfh_seq_header_id` references it | §7.3.8.6 p387 | **PARTIAL** | SEQ before first referencing frame. | M0/M1/M2 always put SEQ in the same TU before the CLK (`basic_headers.cpp:356-369`; `still_syntax.cpp:295-303`; `video_frame.cpp:1008-1014`). `mfh.cpp:774-830` checks both ids against a caller-supplied `MfhAvailability`. | No assembler constructs that availability state; later frames reference `seq_header_id = 0` with no in-bitstream header in their TU (sequential decode only, not RAP start). |
| `R46` MFH must be available before a frame header's `cur_mfh_id` references it; TLayer/MLayer dependency maps of the referenced MFH must be satisfied | §7.3.8.7 p387 | **PARTIAL** | MFH before referencing frame + dependency maps. | `mfh.cpp` implements the writer/reader, inheritance resolver and the §7.3.8.7 layer-dependency check (`mfh.cpp:752-830`; `LayerDependencyMap` at `mfh.h:196-214`); the gate places one MFH in TU 0 (`mfh_gate.cpp:399-409`). | MFH is not re-emitted at a later RAP, and the gate never exercises a second CVS; no stream in the tree combines MFH with an OLK/CLK RAP. |
| `R47` When `apply_grain == 1`, the referenced film-grain OBU with `FilmGrainPresent[fgm_id] == 1` must be available; its dependency maps must be satisfied | §7.3.8.8 p387 | **ABSENT** | Film-grain OBU writer + per-frame reference. | Only the synthesis model exists (`filters/spec_tables/film_grain.{h,cpp}`); `film_grain_params_present = 0` everywhere (`video_headers.cpp:209`). | No `film_grain_config()` syntax writer and no `fgm_id` availability check. |
| `R48` When `using_qmatrix == 1`, referenced QM levels must be available; QM levels from previous TUs are reset at the first CLK/OLK/SWITCH/RAS in a TU (SWITCH only if `restricted_prediction_switch == 1`), protected by `QmProtected` | §7.3.8.9 p388 | **PARTIAL** | Emit QM before the frame and re-emit after each reset point. | QM OBU writer + frame-header writer exist (`src/encoder/qm/qm.cpp:400-430`); `qm_gate.cpp:315-352` splices the OBU at the §7.3.3 position and sets `using_qmatrix=1`. | No `QmProtected`/reset model; every core writer sets `using_qmatrix = 0` (`video_headers.cpp:284`); no multi-frame QM stream. |
| `R49` CI must be available from the first coded extended layer unit of the CVS in which it is present; all instances for a layer in a CVS are identical; CI only in the first coded frame unit of each layer within a TU; if present in any TU it is present in the first TU of the CVS | §7.3.8.10 p388 | **COMPONENT** | CI presence and constancy. | `CvsConstancyTracker::observe()` enforces the constancy and first-TU rules (`content_interpretation.h:388-418`); `scan_content_interpretation()` reports CI after the first frame as invalid (`content_interpretation.h:318-326`). | Not enforced by M2/M1; no core CI emission. |
| `R50` CI parameters reset to defaults at decoder start and at every RAP (CLK/OLK); at a RAP, use CI if present, else inherit from the highest dependent mlayer `k < m` with established parameters; a CI in a non-RAP TU (without a same-layer or dependent-layer CLK/OLK) must match the parameters established at the most recent RAP | §7.3.8.11 p388-389 | **PARTIAL** | If CI is emitted, match the RAP defaults/inheritance. | `CvsConstancyTracker::observe()` re-initialises at a new CVS and at a RAP, falls back to `default_content_interpretation()` when a RAP has no CI, and applies the non-RAP equality rule (`content_interpretation.cpp:1145-1190`; `content_interpretation.h:372-386`). | The MLayerPresenceMap inheritance branch (highest dependent `k < m`) is not implemented; the RAP default fallback is explicitly single-embedded-layer only. |

---

## 6. §7.3.9 Availability of long-term reference frames

| Rule | §/page | Status | Encoder obligation | Evidence | Gap |
|---|---|---|---|---|---|
| `R51` Every LTR frame must be available before being referenced and must be held in the same reference-buffer slot it would occupy under sequential decoding | §7.3.9.1 p389 | **COMPONENT** | Keep the slot identity stable across RAP and sequential decode. | `longterm::apply_frame()` models `RefValid/RefLongTermId/RefOrderHint` and the §5.18.2 first-Regular invalidation (`longterm_pool.cpp:244-400`); `validate_open_random_access()` requires every promised id to still be held (`:476-540`). | Only single-digit frame counts are gated; no core wiring. The gate uses the pinned-dav2d layout that omits `ref_long_term_id` (`longterm/README.md` "Pinned dav2d does not read the OLK long-term list"), so the §7.4.4 condition is not exercised in the gate. |
| `R52` At a RAP with RAS, or with OLK when `long_term_frame_id_bits != 0`, LTR OBUs may appear in the same coded extended layer unit: required CLKs first, then OLKs, all with `immediate_output_frame = 0` and `implicit_output_frame = 0` | §7.3.9.1 p389 | **ABSENT** | Co-locate LTR anchors with the RAP frame. | `validate_temporal_unit()` forbids any RA role at index > 0 (`longterm_pool.cpp:421-424`); `random_access` emits RAS alone in its TU (`VIDEO_INTEGRATION.md:151-160`). | The `random_access` RAS gate names `ref_long_term_id = {0}` whose CLK is in an earlier TU; `longterm`'s OLK gate omits the list only because of the pinned decoder. Neither puts the LTR OBU in the RAP TU, so neither stream is self-contained at that RAP. |

---

## 7. §7.4.1–§7.4.3 Random access: general, LTR use, CLK

| Rule | §/page | Status | Encoder obligation | Evidence | Gap |
|---|---|---|---|---|---|
| `R53` A TU with any CLK/OLK/RAS OBU is a random access point; the process order is: set `isRandomAccessPoint`, derive `MultiStreamDecoderMode`, then apply the per-extended-layer random access process | §7.4.1 p390 | **DECODER-ONLY** | Only the presence/type of the RAP OBU. | CLK: M0/M1/M2 (`basic_headers.cpp:341-356`; `video_headers.cpp:328-339`); OLK: `longterm_headers.cpp:346-380`; RAS/SWITCH: `random_access.cpp:1189-1290`. | — |
| `R54` `MultiStreamDecoderMode = 1` iff the TU contains an MSDO OBU; it persists until the next RAP | §7.4.1 p390 | **ABSENT** | Emit MSDO to select multistream mode. | No MSDO writer. | — |
| `R55` With `long_term_frame_id_bits != 0`, open/switch random access may rely on prior CLK/OLK frame data, which may have to be provided | §7.4.2.1 p391 | **PARTIAL** | Provide or co-locate the LTR data. | `longterm` names retained ids in the OLK planner and models the pool (`longterm.h:321-328`; `longterm_pool.cpp:476-540`); `random_access` validates that a named id is in use (`random_access.cpp:1009-1180`). | The data is never co-located (see `R52`); the encoder can only rely on external provision. |
| `R56` With `long_term_frame_id_bits == 0`, open/switch random access must not use any previous CLK/OLK frame data | §7.4.2.2 p391 | **WIRED** | Do not reference pre-RAP slots when LTRs are off. | M2 writes `long_term_frame_id_bits = 0` (`video_headers.cpp:163`) and every P frame references only slot 0 in the same CVS (`video_frame.cpp:1041`). `leading` restricts regular refs to OLK-refreshed slots (`leading.cpp:477-492`). | — |
| `R57` Closed random access: invalidate every reference buffer, flush pending implicit outputs, activate the referenced sequence header, re-establish CI for every embedded layer, reset QM levels | §7.4.3 p391 | **PARTIAL** | A CLK must carry/reset everything it needs. | M2 frame 0 is the only CLK and is preceded by SEQ; the pool clears all slots (`longterm_pool.cpp:265-274`). | A mid-stream CLK would not be preceded by SEQ/CI, and QM reset is not modelled. `flush_implicit_output_frames` is a decoder operation; with `monotonic_output_order_flag = 1` nothing is pending. |

---

## 8. §7.4.4 Open random access (OLK)

| Rule | §/page | Status | Encoder obligation | Evidence | Gap |
|---|---|---|---|---|---|
| `R58` When decoding starts at an OLK it is treated as the start of a new CVS for that layer: all buffers not refreshed by the OLK are invalidated except the named long-term references, leading frames are discarded, and the referenced sequence header is activated | §7.4.4 p391-392 | **COMPONENT** | OLK refresh mask + named LTRs + SEQ activation. | `plan_open_loop_key()` refreshes only the OLK slot (`longterm.h:321-328`); `validate_open_random_access()` and `apply_frame()` implement the invalidation (`longterm_pool.cpp:476-540,276-315`). | Gate uses `kOmitPinnedDav2d`, so the named-LTR path is unit-tested but not decoder-gated. |
| `R59` At an OLK, the available HLS set must include the referenced sequence header, its LCR, atlas, MFH (`cur_mfh_id > 0`), film grain (`apply_grain = 1`), QM (non-default levels), and MSDO/global LCR in a multistream | §7.4.4 p392 | **PARTIAL** | Repeat HLS in the OLK TU. | `leading` repeats SEQ at the OLK (`leading.cpp:556-560`); `longterm` does not (`longterm_stream.cpp:84-93`). | No LCR/atlas/MFH/FG/QM re-emission at an OLK anywhere; `longterm` cannot produce a self-contained open RAP. |
| `R60` The long-term reference condition: (`long_term_frame_id_bits == 0`) OR (`num_key_ref_frames == 0`) OR (every `ref_long_term_id` frame is available, retained from the previous CVS) | §7.4.4 p392 | **COMPONENT** | Name exactly the LTRs that survive. | `validate_open_random_access()` + `validate_olk_long_term_list()` (`longterm_pool.cpp:68-103,476-540`). | Pinned decoder accepts the OLK only without the list, so the gate stream does not signal it. |
| `R61` Regular frames after an OLK must not reference any frame or information preceding the OLK TU except what the OLK refreshed or the named LTRs; regular frames after leading frames must not reference leading frames or HLS OBUs indicated in leading TUs | §7.4.4 p392-393 | **PARTIAL** | Enforce the reference-set restriction. | `leading::validate_plan()` rejects a regular frame that references a slot not refreshed by the OLK or a leading slot (`leading.cpp:477-492`). | The named-LTR exception is not modelled (the check is stricter than the spec); "HLS indicated in leading TUs" is not tracked at all. |
| `R62` An LTR must be included in an OLK's `ref_long_term_id` iff it is held at the OLK, held at the first Regular frame after the OLK, and in the same or a dependent embedded layer; when `long_term_frame_id_bits > 0`, OLK `OrderHint < (1 << OrderHintBits)` | §7.4.4 p393 | **COMPONENT** | Exact LTR list + hint range. | `validate_olk_long_term_list()` implements the iff in both directions (`longterm_pool.cpp:68-103`); `validate_frame_header()` rejects `order_hint >= (1 << OrderHintBits)` (`:141-143`). | The `m`-layer dependency term is absent (only mlayer 0 exists). |

---

## 9. §7.4.5 Random access switch (RAS)

| Rule | §/page | Status | Encoder obligation | Evidence | Gap |
|---|---|---|---|---|---|
| `R63` Correct initiation at a RAS requires either `num_key_ref_frames == 0` or the named long-term references to be available (§7.3.9) | §7.4.5 p394 | **COMPONENT** | RAS reference set. | `validate_header()` checks every named id is in use for a referencing RAS (`random_access.cpp:1009-1180`); `plan_ras_frame()` rejects an unreachable candidate (`:1223-1290`). | Availability at the RAP is not actually satisfied by the gate stream (`R52`). |
| `R64` RAS refreshes every reference buffer that does not hold a named LTR; the active sequence header and LCR remain in effect; QM levels are reset | §7.4.5 p394 | **COMPONENT** | Derive the refresh mask; QM reset. | `derive_ras_refresh_flags()` (`random_access.cpp:947-962`); `update_reference_state()` (`:963-1008`). | QM reset is not modelled; the active SH is assumed, not re-emitted. |
| `R65` If a LTR is in the RAS `ref_long_term_id` list, it must be held in a reference buffer when the RAS is encountered under sequential decoding | §7.4.5 p394 | **COMPONENT** | Validate LTR liveness. | `validate_header()` requires `long_term_id_in_use(RefLongTermId[ref_frame_idx[i]])` (`random_access.cpp:1009-1180`); the gate decodes the three-frame stream sequentially. | No RAP-start decode of the RAS stream is gated. |
| `R66` When `long_term_frame_id_bits > 0`, the `OrderHint` of a RAS with `restricted_prediction_switch == 0` is `< (1 << OrderHintBits)` | §7.4.5 p394 | **COMPONENT** | Hint range check. | Explicit check in `validate_header()` (`random_access.cpp:1009-1180`). | — |

---

## 10. §7.4.6 Multistream random access

| Rule | §/page | Status | Encoder obligation | Evidence | Gap |
|---|---|---|---|---|---|
| `R67` Random access points need not align across extended layers; a TU may be a RAP for some layers only; a decoder must not decode an extended layer until a RAP for that layer is encountered | §7.4.6 p394-395 | **ABSENT** | Every layer's first TU must start at a CLK (or the layer must be omitted until then). | `svc` composes whatever per-layer units the caller supplies; there is no "first frame is CLK" check and no per-layer RAP schedule. `plan_emission_order()` marks every frame as output with no type (`svc_pipeline.cpp:39-87`). | A layer can begin with a P frame; no layer-start gating. |
| `R68` When `MultiStreamDecoderMode == 1` and `multistream_doh_constraint_flag == 1`, or when a global LCR with `lcr_doh_constraint_flag == 1` is active, all coded output frame units in a TU share `OrderHintBits` and `OrderHint` | §7.4.6 p395 | **ABSENT** | Enforce the constraint if the flag is set. | `lcr_doh_constraint_flag = 0` is hard-coded (`svc_framing.cpp:454`); no MSDO. | See `R38`/`R39`. |
| `R69` An MSDO OBU is parsed before any coded extended layer unit in the TU (`MultiStreamDecoderMode` and `sub_xlayer_id` are established before the per-layer RA processes) | §7.4.6 p395, §7.3.7 p385 | **ABSENT** | MSDO first after TD. | `compose_layered_stream()` emits TD, LCR, OPS, then layers (`svc_framing.cpp:700-747`); no MSDO. | No MSDO syntax at all. |

---

## 11. §7.5 Frame end update CDF process

| Rule | §/page | Status | Encoder obligation | Evidence | Gap |
|---|---|---|---|---|---|
| `R70` `frame_end_update_cdf()`: every CDF array named by `init_coeff_cdfs`/`init_non_coeff_cdfs` is copied from its `Saved*` array into the frame array, then the last entry (use count) becomes `(3 * count) >> 2` | §7.5 p395 | **PARTIAL** | If the frame CDF state is carried to the next frame, the encoder must install exactly the same post-frame arrays. | `CdfBank::average_at_frame_end()` implements the copy + 3/4 count (`src/encoder/entropy/avg_cdf/avg_cdf_bank.cpp:265-277`); the whole bank is a standalone component (only `avg_cdf_test.cpp` and `gate_dav2d_avg_cdf.cpp` use it). | Not wired. It does not matter for current streams because CDFs are per-frame frozen (`video_headers.cpp:200,255,278`; `video_syntax.cpp:414`), and the still path has no next frame. |
| `R71` The process runs after `tg_end == NumTiles - 1` of the frame (from the tile-group payload syntax), not at arbitrary points | §5.20.1 p146, §7.5 p395 | **PARTIAL** | Trigger point and per-frame ordering. | One tile group per frame (`tile_group.h:70-76`), so `tg_end == NumTiles - 1` is always the single group; the encoder never models the trigger. | No frame-end CDF installation in the core: the next frame resets to defaults instead of loading the §7.5 result. Any future adaptive multi-frame path must add this first. |

**CDF reset points (asked explicitly).** The only CDF-state reset the encoder
ever relies on today is the frame-header reset: `primary_ref_frame ==
PRIMARY_REF_NONE || disable_cross_frame_cdf_init == 1` ⇒ `init_non_coeff_cdfs()`
plus `init_coeff_cdfs()` (§5.18.2 p107, §6.17.2 p328). M2 writes
`disable_cross_frame_cdf_init = 1` on **every** inter frame
(`video_headers.cpp:255`) and `disable_cdf_update = 1` (`:278`), and the sequence
header sets `enable_avg_cdf = 0` (`:200`). M0/M1 (single-picture,
`single_picture_header_flag = 1`) force `enable_avg_cdf = 1` and
`avg_cdf_type = 1` by inference (`basic_headers.cpp:246`); with one tile and one
frame, §8.2.4 `avg_cdf()` and §7.5 both run only after all symbols are coded, so
they cannot affect bits. Consequence: **the encoder never implements a
"carry CDF state across frames" mode, so it also never needs §7.5 today — but
`--adaptive-cdf` (`src/main.cpp:269`, `basic_encoder.cpp:498`, `still_encoder.cpp:1810`)
only exercises within-frame adaptation on a single picture.**

---

## 12. §7.6 Extended layer context management

| Rule | §/page | Status | Encoder obligation | Evidence | Gap |
|---|---|---|---|---|---|
| `R72` `save_xlayer_context(obu_xlayer_id)` / `load_xlayer_context(obu_xlayer_id)` save/load all decoder state per extended layer, with `streamID = i` from `sub_xlayer_id` when `MultiStreamDecoderMode`, else `streamID = obu_xlayer_id`; `GLOBAL_XLAYER_ID` returns immediately | §7.6 p395-396 | **ABSENT** | Maintain one CDF/reference/segmentation context per xlayer if the layers are interleaved. | `svc` explicitly relies on "a separate decode context per extended layer" only in prose (`svc/README.md:91-92`; `svc_config.h:17-18`). No save/load state machine exists. | Current multistream streams have no cross-layer prediction and freeze CDFs per frame, so the bits are unaffected; any adaptive or cross-layer mode would diverge. |

---

## 13. §7.7 Get ref frames and §7.8 Get past/future/cur ref lists

| Rule | §/page | Status | Encoder obligation | Evidence | Gap |
|---|---|---|---|---|---|
| `R73` `get_ref_frames`: build `mapOrderHint[]` from `first_slot_with_ref()`, `RefValid`, `RefOrderHint != RESTRICTED_OH`, IsBridge/`bridge_frame_ref_idx`, `AllowedFrames`, `TLayerDependencyMap`/`MLayerDependencyMap`, and `valid_ref_frame_size()` (2:1 .. 16:1 in each dimension) | §7.7 p397 | **COMPONENT** | Only when `frame_explicit_ref_frame_map == 0`; otherwise the encoder supplies `ref_frame_idx` directly. | `adv::build_ref_frame_list()` (`src/encoder/inter/advanced/b_frame.cpp:87-230`) implements the map, `first_slot_with_ref`, size check and layer maps; exercised by `bframes` (`bframes.cpp:681`) and `tlayers`. | M2 writes `explicit_ref_frame_map = 1` at sequence and frame level (`video_headers.cpp:160,260`), so `get_ref_frames` never runs; there is no implicit-mode writer. |
| `R74` Score each candidate (`tDist << DIST_WEIGHT_BITS + q`, or `Dist_Score_Lookup` decay form; subtract `FloorLog2(ref area) << 5`), deduplicate on `(OrderHint, score, mlayer)`, and when `NRanked > REFS_PER_FRAME` drop one reference using the q-threshold + past/future-imbalance rule | §7.7 p397-399 | **COMPONENT** | Exact ranking if implicit mapping is used. | `b_frame.cpp:87-230` implements `kDistScoreLookup`, `kDecayDistCap`, dedup and `get_unmapped_ref` logic; `bframes` tests cover it. | Not called by any bitstream writer. |
| `R75` Bubble-sort by score, take `ref_frame_idx[i] = ScoresIndex[i]` for `i < Min(NRanked, ActiveNumRefFrames)`, then append remaining restricted frames at the end when `checkRes && !IsBridge` and slots remain | §7.7 p399-400 | **COMPONENT** | Exact order and restricted append. | `b_frame.cpp:200-230` (sort + `num_total_refs` + restricted append). | No encoder path uses it. |
| `R76` Conformance of the used indices: `RefValid[ref_frame_idx[i]] == 1`, same bit depth/profile/chroma/colour space, `MLayerDependencyMap`/`TLayerDependencyMap` satisfied, and for RAS `long_term_id_in_use(RefLongTermId[ref_frame_idx[i]]) == 1` (values need not be distinct) | §6.17.2 p324 | **PARTIAL** | Validate the emitted `ref_frame_idx` against the live slot state. | M2 writes a single `ref_slot = 0` with no validity/format check (`video_headers.cpp:259-263`); `longterm` checks `ref_frame_idx < NumRefFrames` and `RefValid` in `apply_frame()`; `random_access` checks RAS long-term ids. `bframes` additionally requires distinct refs (`bframes.cpp:443-495`), which is stricter than the spec. | No profile/chroma/bit-depth/colour-space check; no dependency-map check in any header writer; distinctness is over-enforced in the B-frame planner. |
| `R77` When `explicit_ref_frame_map == 0`, the computed mapping (and the display-order-hint computation) must be identical for every operating point that contains the frame; with an explicit map, a buffer index must denote the same frame for every operating point | §6.4.1 p263 | **ABSENT** | Deterministic ref mapping across operating points. | `svc` builds operating points but `order_frames()`/`compose_layered_stream()` never compare per-OP mappings or hints. | Any multistream with OP-dependent reference state would be non-conforming, unchecked. |
| `R78` `get_past_future_cur_ref_lists`: classify each selected ref by `ScoresDistance` sign; track `NumPastRefs`, `NumFutureRefs`, `numCurRefs`, `ClosestPast`, `ClosestFuture`, `FurthestFuture`; `SkipSegFrame = numCurRefs > 0 ? curRefs[0] : ClosestPast`, else 0 | §7.8 p400-401 | **COMPONENT** | The decoder derives these lists from `ref_frame_idx`; the encoder must not assume different lists. | `adv::get_past_future_cur_ref_lists()` (`b_frame.cpp:234-300`); `bframes` calls it per frame in `simulate_decode()` (`bframes.cpp:432`). | M2 disables `skip_mode_present`, `use_ref_frame_mvs` and TIP, so the lists have no bitstream effect; no core code models them. |
| `R79` Edge tracking: `OrigClosestFuture = ClosestFuture`, `OrigClosestPast = ClosestPast` before motion-field projection | §7.8 p400-401 | **COMPONENT** | Needed only for `use_ref_frame_mvs`/TIP. | The B-frame simulation keeps the classification; `inter/temporal` consumes it separately. | No core path. |

---

## Cross-references

* Audit 11 (`11-output-grain.md`) covers §7.21–§7.23 output/slot state in
  detail; this audit stops at the §7.2/§7.23 boundary.
* Audit 06 (`06-motion.md`) covers §7.9–§7.12; the `get_past_future_cur_ref_lists`
  output (`SkipSegFrame`, `ClosestPast/Future`, `FurthestFuture`) is consumed
  there.
* Audit 15 (`15-cli-reachability.md`) records that the M2 writer and every
  component above are outside the shipped CLI; that is the reason so many rows
  are **COMPONENT** rather than **WIRED**.

---

## Counts

| Status | Rows |
|---|---:|
| WIRED | 8 |
| COMPONENT | 24 |
| PARTIAL | 21 |
| ABSENT | 19 |
| DECODER-ONLY | 7 |
| **Total** | **79** |

By section:

| Section | Rows | W | C | P | A | D |
|---|---:|---:|---:|---:|---:|---:|
| §7.1–§7.2 | 11 | 2 | 2 | 0 | 1 | 6 |
| §7.3.1–§7.3.2 | 8 | 1 | 4 | 1 | 2 | 0 |
| §7.3.3–§7.3.5 | 5 | 1 | 1 | 2 | 1 | 0 |
| §7.3.6–§7.3.7 | 15 | 3 | 1 | 6 | 5 | 0 |
| §7.3.8 | 11 | 0 | 3 | 5 | 3 | 0 |
| §7.3.9 | 2 | 0 | 1 | 0 | 1 | 0 |
| §7.4.1–§7.4.3 | 5 | 1 | 0 | 2 | 1 | 1 |
| §7.4.4 | 5 | 0 | 3 | 2 | 0 | 0 |
| §7.4.5 | 4 | 0 | 4 | 0 | 0 | 0 |
| §7.4.6 | 3 | 0 | 0 | 0 | 3 | 0 |
| §7.5 | 2 | 0 | 0 | 2 | 0 | 0 |
| §7.6 | 1 | 0 | 0 | 0 | 1 | 0 |
| §7.7–§7.8 | 7 | 0 | 5 | 1 | 1 | 0 |

*(W/C/P/A/D = WIRED/COMPONENT/PARTIAL/ABSENT/DECODER-ONLY.)*

---

## Ordering rules the encoder does not currently enforce

This is the actionable list: every §7.3/§7.4 ordering or availability rule with
no enforcement in any code path, followed by the ones enforced only inside a
component. "Unenforced" means a planner or caller can produce a non-conforming
byte sequence and receive no error.

**No enforcement anywhere (core or component):**

1. §7.3.1 p379 — all extended layers in a TU share one output time (`R16`).
2. §7.3.2 p380 — coded multistream video sequence boundary rules and the
   requirement that the MSDO-derived and LCR-derived boundaries agree (`R19`).
3. §7.3.6 p383 — at most **one** coded output frame unit per embedded layer in
   a coded extended layer unit; and all coded non-output frame units must
   precede it. `longterm::validate_temporal_unit()` accepts two shown frames in
   one TU and any order (`R25`).
4. §7.3.6 p383-384 — for CLK/OLK, the *first frame unit of the lowest embedded
   layer present* must be the key frame. Only the TU-wide "index 0" rule is
   enforced (`R28`, `R29`).
5. §7.3.6 p384 — `monotonic_output_order_flag == 0` OrderHint monotonicity
   against output order, with the restricted-SWITCH exception (`R32`).
6. §7.3.6/§7.3.9 p384,389 — the LTR-at-RAP exception: CLK/OLK long-term
   reference OBUs placed before the RAS/OLK in the same coded extended layer
   unit, with both output flags zero (`R33`, `R52`).
7. §7.3.6 p384 — `OrderHint` counter reset at a new CVS and at a restricted
   switch frame (`R35`).
8. §7.3.7 p385 — coded output frame units in different xlayer units of one TU
   share `OrderHint` (`R38`).
9. §7.3.7 p385 — the `lcr_doh_constraint_flag` / `multistream_doh_constraint_flag`
   variants of the hint equality (`R39`, `R68`).
10. §7.3.7 p385 — MSDO after TD and global atlas/metadata placement in the TU
    order; no MSDO or atlas is ever emitted (`R36`, `R41`, `R43`, `R69`).
11. §7.3.8.2 p386 — MSDO availability at every RAP and bit-identity of non-RAP
    MSDOs (`R41`).
12. §7.3.8.8 p387 — film-grain OBU availability and its dependency maps
    (`R47`).
13. §7.4.1 p390 / §7.4.6 p395 — `MultiStreamDecoderMode` and the per-layer
    "do not decode an extended layer before its RAP" rule; no check that each
    layer starts with a CLK (`R54`, `R67`).
14. §7.3.8.9 p388 — QM level reset at the first CLK/OLK/SWITCH/RAS of a TU
    (and the SWITCH `restricted_prediction_switch` condition) (`R48`, `R64`).
15. §6.4.3 p263 — explicit/implicit reference mapping and the display-order-hint
    computation identical across operating points (`R77`).

**Enforced only by an unwired component (the M2 core has no check):**

16. §7.3.6 p383 — mlayer-ascending grouping and non-output-before-output order
    inside a layer (`R25`, `R36`); `longterm` only checks ≥1 output.
17. §7.3.8.1 p386 — HLS availability at every RAP. Only `leading` repeats the
    sequence header at its CLK/OLK; `longterm` and `random_access` emit it once
    in unit 0, so their OLK/RAS TUs are self-contained only if HLS is supplied
    externally (`R34`, `R40`, `R45`, `R59`).
18. §7.3.8.10 p388 — CI only in the first coded frame unit of each embedded
    layer (the tracker works per TU, not per layer frame unit) (`R49`).
19. §7.3.8.11 p388-389 — MLayerPresenceMap inheritance of CI parameters at a
    RAP (`R50`).
20. §7.3.9.1 p389 — an LTR must occupy the same buffer slot under sequential
    and random-access decode. The pool models it, but the LTR list is not
    signalled in the decoder-gated stream (`kOmitPinnedDav2d`) (`R51`).
21. §7.4.5 p394 — QM reset during the RAS process; the RAS gate assumes the
    active sequence header instead of re-emitting it (`R64`).
22. §7.5 p395 — frame-end CDF installation and the `(3*count)>>2` count scale;
    implemented only in `entropy/avg_cdf`, which no frame writer calls, while
    the core resets CDFs per frame instead (`R70`, `R71`).

**Conforming only because the tool is disabled** (should be re-audited if the
flag is ever turned on): BRU region validity (`R10`), TIP wrapup (`R5`), film
grain (`R6`, `R47`), segmentation-map reuse (`R9`), bridge frames (`R24`).

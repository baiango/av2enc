# Spec audit 03 — frame header (§5.18, §6.17)

Read-only audit of AV2 v1.0.0 `frame_header()` coverage against
`spec/AV2_Spec_v1.0.0.pdf`. Every status below was checked against the
implementation, not against `docs/`. Line numbers are 1-based.

- Syntax: §5.18, PDF pages 93–144.
- Semantics: §6.17, PDF pages 319–349.
- Page index used for extraction: spec page − 1 (pypdf).

## Status vocabulary

| Status | Meaning |
|---|---|
| **WIRED** | Emitted by the shipped `av2enc` binary (`src/main.cpp`) and at least one CLI option or input can change the value. |
| **PARTIAL** | Emitted by the shipped binary, but every reachable path pins one value, or only one legal variant of the syntax is reachable (e.g. CCSO band-offset-only). |
| **COMPONENT** | A conforming writer exists in the tree, but only behind a gate executable, a standalone test harness, or an unused helper. The shipped binary cannot produce it. |
| **ABSENT** | Nothing in the tree writes the element. The value is fixed by a disabled sequence-level tool or not implemented. |
| **DECODER-ONLY** | Only a reader/derived state exists. (No row landed here; see note under counts.) |
| **OUT-OF-SCOPE** | Bridge frame, atlas, MSDO, padding. |

### Reachability of the shipped binary

`src/main.cpp` is the only encoder entry point. It dispatches to exactly two
paths and encodes **one still picture** (`single_picture_header_flag = 1`):

- `av2enc::still::encode_still_picture` — `src/main.cpp:656`
- `av2enc::basic::encode_still_picture` — `src/main.cpp:683`

Consequences that apply to almost every row below:

- `single_picture_header_flag = 1` statutorily removes ShowExistingFrame,
  `frame_is_inter`, `refresh_frame_flags` coding, `order_hint` bits, primary-ref
  signalling, and all inter-only tools (`src/bitstream/basic_headers.cpp:294-300`,
  `src/encoder/m1/still_headers.cpp:168-172`).
- Every multi-frame capability (M2 P frames, MFH, long-term refs, SEF, RAS,
  SWITCH, leading pictures, TIP, global motion, delta-Q/segmentation, QM,
  multi-tile, SCC/IBC, lossless, 10-bit) is only reachable through the gate
  executables listed in the evidence column.
- `enable_avg_cdf = 1` and `avg_cdf_type = 1` are forced in the still-picture
  sequence header, which is why no still-picture path codes
  `context_update_tile_id` (§5.18.7.2).

Field order was checked end-to-end against §5.18.2 for the enabled tool subsets
(M0, M1, M2, tiled M2). No ordering violation was found in those writers.

---

## 1. Frame-header envelope and frame types

| Field | §/page | Current | Status | Evidence file:line | Gap |
|---|---|---|---|---|---|
| `is_first_tile_group` | §5.19 / 144; §6.18 / 349 | `writer.put_bit(true);  // is_first_tile_group` | PARTIAL | `src/encoder/m1/still_headers.cpp:287`; `src/bitstream/basic_headers.cpp:332` | Always 1. No path emits a second tile group, so `frame_header_present_flag` is unreachable. Legal but means tile groups are never split. |
| `frame_header_present_flag` | §5.19 / 144 | never written | ABSENT | `src/encoder/tiles/tile_group.cpp:49-53` rejects a non-first group with a frame header | Only one tile group per frame. §5.19 allows partial tile groups; the encoder cannot produce them. |
| `frame_header_copy` / `header_bit[i]` | §5.18.1 / 93; §6.17.1 / 319 | never written | ABSENT | — | Bit-exact duplicate of the first frame header is not emitted because there is never a later tile group. |
| `cur_mfh_id` | §5.18.2 / 95; §6.17.2 / 319 | `put_uvlc(writer, 0);  // cur_mfh_id` | PARTIAL | `src/encoder/m1/still_headers.cpp:168`; `src/bitstream/basic_headers.cpp:297`; `src/encoder/m2/video_headers.cpp:228` | CLI always 0. The MFH adapter can write a real id (`src/encoder/frame/mfh/mfh_video_adapter.cpp:145`), the SEF writer can copy one (`src/encoder/inter/sef/sef.cpp:190`), but both are gate-only. |
| `seq_header_id_in_frame_header` | §5.18.2 / 95; §6.17.2 / 319 | `put_uvlc(writer, 0);  // seq_header_id_in_frame_header` | PARTIAL | `src/encoder/m1/still_headers.cpp:169`; `src/encoder/m2/video_headers.cpp:229` | Always 0. No path encodes a second sequence header, so `load_sequence_header(seq_header_id)` is never exercised. The MFH adapter correctly omits the field when `cur_mfh_id != 0` (`mfh_video_adapter.cpp:146-148`). |
| FrameType = KEY_FRAME (CLK) | §5.18.2 / 96-97; §6.17.2 / 320 | OBU type `OBU_CLOSED_LOOP_KEY`, no `frame_is_inter` bit | WIRED | `src/encoder/m1/still_headers.cpp:294`; `src/bitstream/basic_headers.cpp:348` | Only reachable frame type in the product. |
| `frame_is_inter` → INTER_FRAME / INTRA_ONLY_FRAME | §5.18.2 / 97; §6.17.2 / 321 | `writer.put_bit(true);  // frame_is_inter = 1 -> INTER_FRAME` | COMPONENT | `src/encoder/m2/video_headers.cpp:233`; `src/encoder/inter/leading/leading.cpp:247` | `frame_is_inter = 0` (INTRA_ONLY) is written by no non-key frame. `longterm_headers.cpp:356` accepts the variable, but the only false values come from the key planners, where the bit is not coded (`longterm_pool.cpp:512,531`). |
| FrameType = SWITCH_FRAME (`restricted_prediction_switch`) | §5.18.2 / 97; §6.17.2 / 320 | `writer.put_bit(h.restricted_prediction_switch);` | COMPONENT | `src/encoder/inter/random_access/random_access.cpp:718-719` | Gate only (`ras_gate.cpp`). Product has no switch-frame path. |
| FrameType = SWITCH_FRAME (OBU_RAS_FRAME) | §5.18.2 / 97; §6.17.2 / 320 | same derivation for `ObuType::kRas` | COMPONENT | `src/encoder/inter/random_access/random_access.cpp:892-901` | Gate only. RAS also carries the refresh-flags conformance hack (see violations V1). |
| OBU_OPEN_LOOP_KEY (OLK) | §5.18.2 / 95-100; §6.17.2 / 320 | `case FrameKind::kOpenLoopKey: return ObuType::kOpenLoopKey;` | COMPONENT | `src/encoder/inter/leading/leading.cpp:332-333`; `src/encoder/inter/longterm/longterm.h:124-126` | Gate only; OLK requires `monotonic_output_order_flag = 0` for output, which the product never sets. |
| OBU_LEADING_SEF / OBU_REGULAR_SEF | §5.18.2 / 96; §6.17.2 / 320 | `writer.put_bits(header.frame_to_show_map_idx, index_bits);` | COMPONENT | `src/encoder/inter/sef/sef.cpp:190-199`, OBU at `214-217` | Gate only. Film grain on a SEF is rejected (`sef.cpp:188`), so `film_grain_config()` is never exercised there. |
| OBU_LEADING_TIP / OBU_REGULAR_TIP (TipFrameMode) | §5.18.2 / 103-104; §6.17.2 / 326 | `write_frame_temporal_syntax()` exists | COMPONENT | `src/encoder/inter/temporal/temporal_syntax.cpp:208-282` | Helper has **no caller** outside tests; `enable_tip` is written 0 in every sequence header (`video_headers.cpp:169`). The whole TIP frame-header block is dead code. |
| OBU_LEADING_TILE_GROUP | §5.18.2 / 100; §6.17.2 / 323 | `case FrameKind::kLeadingTileGroup: return ObuType::kLeadingTileGroup;` | COMPONENT | `src/encoder/inter/leading/leading.cpp:334-335` | Gate only; leading refresh/ordering rules are validated in `leading.cpp:457-493`. |
| OBU_REGULAR_TILE_GROUP | §5.18.2 / 100; §6.17.2 / 323 | `type = header.keyframe ? kClosedLoopKey : kRegularTileGroup` | COMPONENT | `src/encoder/m2/video_headers.cpp:335-337` | M2 P frames are gate-only (`video_gate.cpp`, and every inter gate through `av2enc::video::encode_video_sequence`). |
| OBU_BRIDGE_FRAME | §5.18.2 / 95-99; §6.17.4.2 / 336 | no writer | OUT-OF-SCOPE | no `bridge_frame_*` references outside docs | Per audit scope. |

## 2. Reference state, output flags, order hints

| Field | §/page | Current | Status | Evidence file:line | Gap |
|---|---|---|---|---|---|
| `long_term_id_plus_1` / LongTermId | §5.18.2 / 97; §6.17.2 / 321 | `writer.put_bits(plus_one, c.long_term_frame_id_bits);` | COMPONENT | `src/encoder/inter/longterm/longterm_headers.cpp:359-364`; `src/encoder/inter/random_access/random_access.cpp:724-727` | CLI writes a zero-width field (`long_term_frame_id_bits = 0`). Gate-only writer. |
| `num_key_ref_frames` | §5.18.2 / 97; §6.17.2 / 321 | `writer.put_bits(h.ref_long_term_id.size(), 3U);` | COMPONENT | `src/encoder/inter/longterm/longterm_headers.cpp:366-368`; `random_access.cpp:729-731` | Gate-only. The long-term gate can also omit it under the pinned-dav2d switch (violation V2). |
| `ref_long_term_id[i]` | §5.18.2 / 97; §6.17.2 / 321 | `writer.put_bits(id, c.long_term_frame_id_bits);` | COMPONENT | `src/encoder/inter/longterm/longterm_headers.cpp:369-371` | Reserved value `(1 << bits) - 1` is rejected only by the pool (`longterm_pool.cpp:203-207`) and by `random_access.cpp:1095-1097`; the raw writer does not check. |
| `immediate_output_frame` | §5.18.2 / 98; §6.17.2 / 321 | M2: `writer.put_bit(true);` | COMPONENT | `src/encoder/m2/video_headers.cpp:236`; `src/encoder/inter/leading/leading.cpp:266`; `longterm_headers.cpp:374-378` | CLI infers 1 via `single_picture_header_flag`. Every M2/regular frame is forced to output immediately, so decode order == output order and B-frame reordering is impossible (`src/encoder/inter/advanced/b_frame.cpp` is not wired to the header). |
| `implicit_output_frame` | §5.18.2 / 98; §6.17.2 / 321 | `writer.put_bit(false);  // implicit_output_frame` | COMPONENT | `src/encoder/inter/longterm/longterm_headers.cpp:380-382`; `leading.cpp:258` (OLK writes 1) | No CLI path. `leading.cpp` writes 1 only for an OLK; `longterm_headers.cpp` writes 0, so a long-term OLK is never output unless a SEF is added. |
| `frame_size_override_flag` | §5.18.2 / 98; §6.17.2 / 322 | `writer.put_bit(false);  // frame_size_override_flag = 0` | COMPONENT | `src/encoder/m2/video_headers.cpp:244`; `longterm_headers.cpp:384`; `mfh_video_adapter.cpp:158` | CLI: inferred 0 (single-picture). M2: always 0. Only the MFH adapter can code 1 (key frames), and it rejects `frame_size_override` for inter frames (`mfh_video_adapter.cpp:140-146`). |
| `order_hint` | §5.18.2 / 98; §6.17.2 / 322 | `writer.put_bits(h.order_hint, c.order_hint_bits);` | COMPONENT | `src/encoder/m2/video_headers.cpp:245`; `longterm_headers.cpp:385`; `leading.cpp:275`; `random_access.cpp:748` | CLI: zero-width (OrderHintBits = 0). M2 increments by one, so order hint == decode index; no reorder/GOP structure. |
| `signal_primary_ref_frame` | §5.18.2 / 99; §6.17.2 / 322 | `writer.put_bit(false);  // signal_primary_ref_frame` | COMPONENT | `src/encoder/m2/video_headers.cpp:254`; `leading.cpp:288`; `longterm_headers.cpp:388` | Always 0. The encoder never explicitly selects a primary reference; `primary_ref_frame` stays `PRIMARY_REF_CHOOSE` and is derived by the decoder. |
| `disable_cross_frame_cdf_init` | §5.18.2 / 99; §6.17.2 / 322 | `writer.put_bit(true);  // disable_cross_frame_cdf_init` | COMPONENT | `src/encoder/m2/video_headers.cpp:255`; `leading.cpp:289` | Always 1. Cross-frame CDF inheritance is never used, even when `primary_ref_frame` resolves to a valid reference. |
| `primary_ref_frame` | §5.18.2 / 99; §6.17.2 / 322 | never coded (signal bit is 0) | ABSENT | `src/encoder/m2/video_headers.cpp:254` | The explicit f(3) form is unreachable; primary reference is always derivable-only. |
| `max_drl_bits_minus_1` / `change_drl` | §5.18.2 / 104; §6.17.2 / 323 | never coded | ABSENT | every sequence header writes `allow_frame_max_drl_bits = false` (e.g. `video_headers.cpp:165`) | Per-frame DRL depth cannot be changed. |
| `bridge_frame_overwrite_flag` | §5.18.2 / 99; §6.17.2 / 323 | no writer | OUT-OF-SCOPE | — | Bridge only. |
| `refresh_frame_flags` | §5.18.2 / 99-101; §6.17.2 / 323-324 | M2: `writer.put_bits(h.refresh_frame_flags, c.num_ref_frames);` | COMPONENT | `src/encoder/m2/video_headers.cpp:258`; `leading.cpp:290-291`; `longterm_headers.cpp:392-402` | CLI infers `allFrames` for a CLK (`basic_headers.cpp:300`). Inter refresh masks are gate-only. RAS writes the field only through the non-conforming compat switch (V1). |
| `has_refresh_frame_flags` / `frame_to_refresh` | §5.18.2 / 100; §6.17.2 / 323 | never coded | ABSENT | all sequences write `enable_short_refresh_frame_flags = 0` (`longterm_headers.cpp:287`) | Single-slot refresh is unreachable. `longterm_pool.cpp:137-140` rejects enabling a short refresh, and `longterm_headers.cpp:397` asserts it is off. |
| `frame_explicit_ref_frame_map` | §5.18.2 / 101; §6.17.2 / 323 | `writer.put_bit(true);  // frame_explicit_ref_frame_map` | COMPONENT | `src/encoder/m2/video_headers.cpp:260`; `leading.cpp:292`; `longterm_headers.cpp:409` | Always 1 in the inter writers. The implicit `get_ref_frames()` default map is not used by any path. |
| `num_total_refs` | §5.18.2 / 101; §6.17.2 / 324 | `writer.put_bits(c.num_total_refs, 3U);` | COMPONENT | `src/encoder/m2/video_headers.cpp:261`; `mfh_video_adapter.cpp:167` | M2 pins 1 (`VideoConfig::num_total_refs = 1`). Multi-reference is reachable only through compound gates that rewrite the header. |
| `ref_frame_idx[i]` | §5.18.2 / 101; §6.17.2 / 324 | `writer.put_bits(h.ref_slot, ceil_log2(c.num_ref_frames));` | COMPONENT | `src/encoder/m2/video_headers.cpp:262-263`; `longterm_headers.cpp:412-414` | M2 writes one slot for all blocks. The generic per-reference array is only used by the long-term/leading/tlayers gates. |
| `screen_content_params()` / `allow_screen_content_tools` | §5.18.3.3 / 113-114; §6.17.3.2 / 335 | `writer.put_bit(false);  // allow_screen_content_tools` | PARTIAL | `src/encoder/m1/still_headers.cpp:170`; `src/bitstream/basic_headers.cpp:301` | CLI always 0, so palette and intra BC are impossible. The SCC/IBC components write 1 (`palette/scc_stream.cpp:59`; `m1/ibc_stream/still_ibc_stream.cpp:54`), but both are gate-only. |
| `force_integer_mv` | §5.18.3.3 / 114; §6.17.3.2 / 335 | `writer.put_bit(true);  // force_integer_mv (IBC/CRC use integer vectors)` | COMPONENT | `src/encoder/palette/scc_stream.cpp:60-63`; `m1/ibc_stream/still_ibc_stream.cpp:55-57` | Only coded when `allow_screen_content_tools = 1`; unreachable from the CLI. |
| `allow_intrabc` (`intrabc_params()`) | §5.18.3.4 / 114; §6.17.3.3 / 335 | `writer.put_bit(false);  // allow_intrabc` | PARTIAL | `src/encoder/m1/still_headers.cpp:171`; `src/encoder/m2/video_headers.cpp:269` | CLI always 0. `palette_stream.cpp:52-57` sets it but rejects IBC in the palette-only stream; `still_ibc_stream.cpp:59-64` writes it for the IBC gate. |
| `allow_global_intrabc` / `allow_local_intrabc` | §5.18.3.4 / 114; §6.17.3.3 / 335 | `writer.put_bit(c.allow_global_intrabc); ...` | COMPONENT | `src/encoder/m1/ibc_stream/still_ibc_stream.cpp:60-65`; `scc_stream.cpp:66-70` | Gate-only. The main M1 writer does not implement the intra-BC allow branches. |
| `change_bvp_drl` / `max_bvp_drl_bits_minus_1` | §5.18.3.4 / 114; §6.17.3.3 / 335 | never coded | ABSENT | `allow_frame_max_bvp_drl_bits = false` in every sequence (e.g. `still_headers.cpp:126`) | Per-frame IBC DRL depth cannot change. |
| `use_qtr_precision_mv` / `allow_high_precision_mv` (FrameMvPrecision) | §5.18.2 / 105; §6.17.2 / 325 | `writer.put_bit(false); ... writer.put_bit(true);` | PARTIAL | `src/encoder/m2/video_headers.cpp:271-272`; `leading.cpp:300-301`; `longterm_headers.cpp:419-420` | Always EIGHTH_PEL. Quarter-pel and half-pel are never signalled. `enable_flex_mvres = false` (`video_headers.cpp:180`) makes `use_per_block_mv_precision = 0`. |
| `read_interpolation_filter()` / `is_filter_switchable` + `interpolation_filter` | §5.18.5.1 / 116; §6.17.5.1 / 337 | `writer.put_bit(false); writer.put_bits(0U, 2U);  // EIGHTTAP` | PARTIAL | `src/encoder/m2/video_headers.cpp:273-274`; `leading.cpp:302-303` | Always frame-level EIGHTTAP. Switchable, smooth, sharp and bilinear are unreachable. |
| `frame_enabled_motion_modes` | §5.18.2 / 105; §6.17.2 / 325 | never coded | ABSENT | `seq_frame_motion_modes_present_flag = 0` because all `seq_enabled_motion_modes` are 0 (`video_headers.cpp:150-153`) | Inter-intra, OBMC, warp, compound etc. have no per-frame enable. |
| `disable_cdf_update` | §5.18.2 / 106; §6.17.2 / 326 | `writer.put_bit(c.disable_cdf_update);` | WIRED | `src/encoder/m1/still_headers.cpp:172`; `src/bitstream/basic_headers.cpp:303` | M0 `--adaptive-cdf` and M1 vary it (`basic_headers.h:81-88`; `still_encoder.cpp:1810`). M1 still defaults to frozen CDFs. |
| `use_bru` / `bru_ref` / `bru_inactive` | §5.18.2 / 102; §6.17.2 / 324 | never coded | ABSENT | all sequences write `enable_bru = false` (e.g. `video_headers.cpp:177`) | Backwards reference update is not implemented. |
| `use_ref_frame_mvs` / `tmvp_sample_step_minus_1` | §5.18.2 / 102; §6.17.2 / 325 | `writer.put_bit(...)` inside temporal helper | COMPONENT | `src/encoder/inter/temporal/temporal_syntax.cpp:208-245` | Helper only, no caller; `enable_ref_frame_mvs = false` in every sequence header (`video_headers.cpp:155`). TMVP projection step is unreachable. |
| TIP block (`tip_frame_mode`, `frame_opfl_refine_type`, `allow_tip_hole_fill`, `tip_global_wtd_index`, `tip_mv_*`, `tip_sharp`, `tip_regular`) | §5.18.2 / 103-104; §6.17.2 / 326 | `write_frame_temporal_syntax()` / `write_opfl_refine_type()` | COMPONENT | `src/encoder/inter/temporal/temporal_syntax.cpp:153-282` | Dead helper: nothing calls it. `enable_tip = false` everywhere. |
| `allow_tcq` / `allow_parity_hiding` | §5.18.2 / 109; §6.17.2 / 327 | never coded | ABSENT | sequences write `enable_tcq = false` (`video_headers.cpp:198`, `still_headers.cpp:137`) | TCQ and parity hiding are unavailable. |

## 3. Frame size and cropping

| Field | §/page | Current | Status | Evidence file:line | Gap |
|---|---|---|---|---|---|
| `frame_size()` `frame_width_minus_1` / `frame_height_minus_1` | §5.18.4.1 / 115; §6.17.4.1 / 335 | never coded on the CLI (`frame_size_override_flag = 0`, size inherited from `max_frame_*`) | COMPONENT | `src/encoder/m2/video_headers.cpp:244-245`; `mfh_video_adapter.cpp:160-166`; `random_access.cpp:783-789` | Sizes are always derived from `mfh_frame_width_minus_1[cur_mfh_id]` (which equals `max_frame_*` for id 0). Explicit smaller/larger frame size inside a sequence is unreachable; only MFH-adapter key frames and SWITCH/RAS gate frames code the fields. |
| `frame_size_with_refs()` / `found_ref` | §5.18.4.3 / 115-116; §6.17.4.3 / 336 | never coded | ABSENT | `mfh_video_adapter.cpp:140-146` explicitly rejects the inter case | The "inherit size from reference i" form is not implemented. Every inter frame is forced to the sequence/MFH size. |
| `frame_size_with_bridge()` / `bridge_frame_width_minus_1` | §5.18.4.2 / 115; §6.17.4.2 / 336 | no writer | OUT-OF-SCOPE | — | Bridge only. |
| `compute_image_size()` / cropping (`CropLeft`, `CropTop`, `CropWidth`, `CropHeight`) | §5.18.4.4 / 116; §6.17.4.4 / 336 | every sequence writes `writer.put_bit(false);  // seq_cropping_window_present_flag` | ABSENT | `src/bitstream/basic_headers.cpp:202`; `src/encoder/m1/still_headers.cpp:100`; `src/encoder/m2/video_headers.cpp:125` | No cropping window can ever be signalled. Inputs must be exact multiples of 8 for `MiCols/MiRows` not to imply padding. Odd/edge sizes are handled by padding the codable frame, not by cropping. |
| MFH `mfh_frame_width_minus_1` / `mfh_frame_height_minus_1` inheritance | §5.18.2 / 95; §6.17.2 / 319 | `writer.put_bits(header.frame_width - 1U, width_bits)` | COMPONENT | `src/encoder/frame/mfh/mfh.cpp:395-411`; `mfh_video_adapter.cpp:159-166` | MFH OBU writer is correct, but only `mfh_gate` emits it. CLI has no MFH. |

## 4. Quantization, QM, segmentation, delta-Q

| Field | §/page | Current | Status | Evidence file:line | Gap |
|---|---|---|---|---|---|
| `base_q_idx` | §5.18.6.1 / 118; §6.17.6.1 / 337 | `writer.put_bits(c.base_q_idx, 8U);  // base_q_idx` | WIRED | `src/encoder/m1/still_headers.cpp:175`; `high_bitdepth_headers.cpp:190` (f(9)) | M1 varies 1–255 via `--qp` (`still_encoder.cpp:1809`). M0/M10/HBD pin 90/90. 10-bit `f(9)` is implemented but not reachable from the CLI. |
| `DeltaQYDc` (`read_delta_q`) | §5.18.6.1 / 118; §6.17.6.1 / 337 | never coded | ABSENT | sequences write `equal_ac_dc_q = true` and `y_dc_delta_q_enabled = 0` (`still_headers.cpp:140-142`) | Y DC offset cannot be signalled. |
| `diff_uv_delta` | §5.18.6.1 / 118; §6.17.6.1 / 337 | never coded | ABSENT | `separate_uv_delta_q = false` in every sequence (`still_headers.cpp:139`) | U and V always share deltas. |
| `DeltaQUDc` / `DeltaQUAc` | §5.18.6.1 / 118; §6.17.6.1 / 337-338 | never coded | ABSENT | `uv_dc_delta_q_enabled = 0`, `uv_ac_delta_q_enabled = false` (`still_headers.cpp:141-142`) | Chroma quantizers are fixed relative to `base_q_idx`. |
| `DeltaQVDc` / `DeltaQVAc` | §5.18.6.1 / 118-119; §6.17.6.1 / 338 | never coded | ABSENT | same as above | Same. |
| `using_qmatrix` | §5.18.6.2 / 119; §6.17.6.2 / 338 | `writer.put_bit(false);  // using_qmatrix` | PARTIAL | `src/encoder/m1/still_headers.cpp:177`; `src/encoder/m2/video_headers.cpp:284` | Always 0. `src/encoder/m1/deltaq/deltaq.cpp:439-441` hard-rejects `using_qmatrix = 1`; the m3 tool matrix lists QM as BLOCKED. |
| `pic_qm_num_minus_1`, `qm_y[i]`, `qm_uv_same_as_y`, `qm_u[i]`, `qm_v[i]` | §5.18.6.2 / 119; §6.17.6.2 / 338 | never coded | ABSENT | no writer; `fixed_qm`/`qm.cpp` implement the tables/quantizer only | No QM payload syntax path exists. |
| `segmentation_enabled` | §5.18.7.1 / 120; §6.17.7.1 / 339 | `writer.put_bit(false);  // segmentation_enabled` | PARTIAL | `src/encoder/m1/still_headers.cpp:176`; `video_headers.cpp:283` | CLI always 0. The delta-Q gate can write 1 by splicing, so the syntax exists but not in the product. |
| `reuse_seg_info` | §5.18.7.1 / 120; §6.17.7.1 / 339 | `writer.put_bit(header.reuse_seg_info);` | COMPONENT | `src/encoder/frame/mfh/mfh_video_adapter.cpp:104-120` | Gate-only, and only reachable when an MFH or sequence seg-info source exists (none by default). |
| `seg_info()` `feature_enabled` / `feature_value` (ALT_Q `su(10)`) | §5.18.7.1 / 120; §5.4.9 / 66; §6.17.7.1 / 339 | `writer.put_bit(feature.enabled); ... put_su(...)` | COMPONENT | `src/encoder/frame/mfh/mfh.cpp:205-219`; `mfh_video_adapter.cpp:72-88`; `deltaq.cpp:109-160` | Gate-only. Values are validated to `[-MAXQ_BITS, MAXQ_BITS]` (= ±351) in `mfh.cpp:182-190` (`kSegAltQMax` in `mfh.h:59-60`). |
| `segmentation_update_map` | §5.18.7.1 / 121; §6.17.7.1 / 339 | `writer.put_bit(params.update_map);` | COMPONENT | `src/encoder/m1/deltaq/deltaq.cpp:398` | Gate-only. Key frames infer 1, inter frames code the bit. |
| `segmentation_temporal_update` | §5.18.7.1 / 121; §6.17.7.1 / 339 | `writer.put_bit(...)` guarded by `FrameType != SWITCH_FRAME` | COMPONENT | `src/encoder/m1/deltaq/deltaq.cpp:399-402` | Gate-only; M2's `FrameType` is never SWITCH (`video_deltaq.cpp:38`). |
| `delta_q_present` | §5.18.7.8 / 127; §6.17.7.4 / 341 | `writer.put_bit(false);  // delta_q_present` | PARTIAL | `src/encoder/m1/still_headers.cpp:178`; `video_headers.cpp:285` | CLI always 0. The delta-Q gate splices `write_frame_qp_syntax()` into the M2 header (`src/encoder/m2/deltaq/deltaq_gate.cpp:233`), so the syntax exists but only in the gate. |
| `delta_q_res` | §5.18.7.8 / 127; §6.17.7.4 / 341 | `writer.put_bits(params.res, 2U)` | COMPONENT | `src/encoder/m1/deltaq/deltaq.cpp:430-433` | Gate-only. `pipeline.h:136` exposes `delta_q_res` but the CLI never sets it. |

## 5. Loop filters

| Field | §/page | Current | Status | Evidence file:line | Gap |
|---|---|---|---|---|---|
| `apply_deblocking_filter[0..3]` | §5.18.5.2 / 116-117; §6.17.5.2 / 337 | `writer.put_bit(c.apply_deblock_y_vert);` etc. | WIRED | `src/encoder/m1/still_headers.cpp:182-189` | M1 varies all four via `--no-deblock` (`still_encoder.cpp:1819-1822`). M0 and the plain M2 writer pin the luma pair to 0 (`basic_headers.cpp:312-313`; `video_headers.cpp:286-287`). |
| `df_delta_q_present[i]` | §5.18.5.2 / 117; §6.17.5.2 / 337 | `writer.put_bit(false)` per enabled plane | PARTIAL | `src/encoder/m1/still_headers.cpp:194-197`; `src/encoder/m2/post/loop_filters.cpp:459-468` | Always 0. `write_video_deblock_params()` can accept `df_delta_q_present = true`, but then it omits the following `df_delta_q` (V4). |
| `df_delta_q[i]` | §5.18.5.2 / 117; §6.17.5.2 / 337 | never coded | ABSENT | `src/encoder/m2/post/loop_filters.cpp:459-470` writes only the present bit | Missing field when present = 1 (conformance bug, V4). No path writes it. |
| `gdf_frame_enable`, `gdf_per_block`, `gdf_pic_qc_idx`, `gdf_pic_scale_idx` | §5.18.7.9 / 127-128; §6.17.7.5 / 341 | never coded | ABSENT | `enable_gdf = false` in every sequence (e.g. `still_headers.cpp:147`); `video_restoration.cpp:499` forces `gdf_frame_enable = 0` | GDF is entirely absent from the frame header. The GDF filter itself exists (`src/encoder/filters/spec_tables/gdf.cpp`) but cannot be signalled. |
| `cdef_frame_enable` | §5.18.7.10 / 128; §6.17.7.6 / 341 | inferred 1 for single-picture; `writer.put_bit(...)` only in tiled M2 | PARTIAL | `src/encoder/m1/still_headers.cpp:199-201` (comment: no bit); `src/encoder/m2/post/tiled_frame.cpp:465-482` | CLI never codes the flag. Tiled M2 codes it. |
| `cdef_damping_minus_3`, `cdef_strengths_minus_1`, `cdef_on_skip_txfm_frame_enable`, per-strength Y/UV pri/sec | §5.18.7.10 / 128-129; §6.17.7.6 / 341 | `writer.put_bits(c.cdef.damping - 3, 2U); ...` | WIRED | `src/encoder/m1/still_headers.cpp:201-227`; `filters_stream.cpp:431-493` | M1 varies the strength sets via `--still-cdef` and a CDEF search (`still_cdef.cpp`). Values are validated in `still_headers.cpp:202-212`. |
| `FrameRestorationType` (`tool_index`) | §5.18.7.11 / 129-130; §6.17.7.7 / 341 | `put_ns(writer, n, tool_index);` | COMPONENT | `src/encoder/m1/filters_stream/filters_stream.cpp:1434-1508`; `wiener_syntax.cpp:934` | Sequence `enable_restoration = false` everywhere (`still_headers.cpp:148`), so the field never appears on the CLI. Gate-only (`lr_chroma_gate.cpp`). |
| `frame_filters_on[plane]` | §5.18.7.11 / 130; §6.17.7.7 / 342 | `writer.put_bit(params.frame_filters_on[plane]);` | COMPONENT | `filters_stream.cpp:1500-1515` | Gate-only; M3 lists Wiener NS as BLOCKED. |
| `temporal_pred_flag[plane]` / `rst_ref_pic_idx` | §5.18.7.11 / 130; §6.17.7.7 / 342 | never coded (rejected) | COMPONENT | `filters_stream.cpp:1503-1507` returns "temporal frame-filter prediction is not supported" | Temporal LR prediction is unreachable. |
| `num_filter_classes_idx` | §5.18.7.11 / 131; §6.17.7.7 / 342 | `writer.put_bits(params.num_filter_classes_idx, 3U)` | COMPONENT | `filters_stream.cpp:1509-1514` | Gate-only; only non-temporal (intra) frame filter sets. |
| `lr_luma_use_half_size` / `lr_luma_use_max_size` / `lr_luma_use_quarter_size` (and chroma equivalents) | §5.18.7.11 / 131-132; §6.17.7.7 / 342 | `writer.put_bit(shift == 1)` / `put_bit(shift == 0)` / `put_bit(shift == 2)` | COMPONENT | `filters_stream.cpp:1087-1113` | Gate-only. |
| `ccso_frame_flag` | §5.18.7.12 / 134; §6.17.7.8 / 343 | inferred 1 for single-picture; `writer.put_bit(...)` in tiled M2 | PARTIAL | `src/encoder/m1/still_headers.cpp:229-232` (no bit); `src/encoder/m2/post/tiled_frame.cpp:487-510` | CLI never codes it. |
| `ccso_planes[plane]` | §5.18.7.12 / 134; §6.17.7.8 / 343 | `writer.put_bit(p.enabled);` | WIRED | `src/encoder/m1/still_headers.cpp:233-236` | M1 varies per plane via `--still-ccso` (`still_encoder.cpp:1794-1803`). |
| `reuse_ccso[plane]` / `sb_reuse_ccso[plane]` / `ccso_ref_idx[plane]` | §5.18.7.12 / 134; §6.17.7.8 / 343 | never coded | ABSENT | `still_headers.cpp:233-256` writes the fresh form only; `filters_stream.cpp:755-820` supports it but no path sets reuse | Cross-frame CCSO reuse is unreachable. |
| `ccso_bo_only` | §5.18.7.12 / 134; §6.17.7.8 / 343 | `assert(p.bo_only); writer.put_bit(p.bo_only);` | PARTIAL | `src/encoder/m1/still_headers.cpp:237-240` | Always 1. The full (non-band-offset) CCSO form is not emitted. |
| `ccso_quant_idx`, `ccso_ext_filter`, `ccso_edge_clf`, `ccso_max_band_log2`, `ccso_offset_idx` | §5.18.7.12 / 134-135; §6.17.7.8 / 343-344 | `writer.put_bits(scale_idx, 2U); ... put_bits(max_band_log2, band_bits);` offsets only | PARTIAL | `src/encoder/m1/still_headers.cpp:241-254` | `bo_only = 1` makes quant/ext/edge inferred 0; only `max_band_log2` and the offset indices vary. `ccso_ext_filter == 7` is not representable because the field is never coded. |
| `read_tx_mode()` / `tx_mode_select` | §5.18.8.1 / 135; §6.17.8.1 / 344 | `writer.put_bit(c.tx_mode_select);  // 0 = TX_MODE_LARGEST` | PARTIAL | `src/encoder/m1/still_headers.cpp:258`; `video_headers.cpp:289` | Every CLI path sets `tx_mode_select = false` (`still_encoder.cpp:1817`). `TX_MODE_SELECT` is implemented in tile syntax (`still_syntax.cpp`) but cannot be signalled. `CodedLossless` forces ONLY_4X4 with no bit (`lossless.cpp:2880-2883`). |
| `frame_reference_mode()` / `reference_select` | §5.18.8.3 / 136; §6.17.8.3 / 344 | `writer.put_bit(false);  // reference_select` | PARTIAL | `src/encoder/m2/video_headers.cpp:291`; `m2/post/tiled_frame.cpp:514` | Always 0. Compound prediction signalling exists in gates (`tlayers_syntax.cpp:325`, `drl_stream.cpp:238`) but not in the product. |
| `skip_mode_params()` / `skip_mode_present` | §5.18.8.2 / 136; §6.17.8.2 / 344 | `writer.put_bit(false);  // skip_mode_present` | PARTIAL | `src/encoder/m2/video_headers.cpp:292`; `m2/post/tiled_frame.cpp:515` | Always 0. Skip mode is never enabled. |

## 6. Global motion, film grain, tile info and tile group

| Field | §/page | Current | Status | Evidence file:line | Gap |
|---|---|---|---|---|---|
| `reduced_tx_set` | §5.18.2 / 110; §6.17.2 / 327 | `writer.put_bits(0U, 2U);  // reduced_tx_set` | PARTIAL | `src/encoder/m1/still_headers.cpp:259`; `video_headers.cpp:295`; `basic_headers.cpp:316` | Always 0 on the CLI. The M1 config field exists (`still_headers.h:76`) but `still_encoder.cpp:1818` pins 0; HBD/IBC expose it but are gate-only. |
| `allow_bawp` | §5.18.2 / 109; §6.17.2 / 327 | never coded | ABSENT | `enable_bawp = false` in every sequence (`video_headers.cpp:171`) | BAWP cannot be signalled. The BAWP implementation exists in `advanced/` but has no header path. |
| `allow_warpmv_mode` | §5.18.2 / 110; §6.17.2 / 327 | never coded | ABSENT | `seq_enabled_motion_modes[DELTAWARP] = 0` (`video_headers.cpp:150-152`) | Warp-MV mode is unreachable. |
| `use_global_motion` | §5.18.9.1 / 137; §6.17.9.1 / 345 | `writer.put_bit(context.use_global_motion);` | COMPONENT | `src/encoder/inter/gme/gme_syntax.cpp:292-295` | Gate-only: `gme_gate.cpp:717`. `enable_global_motion = false` in every sequence header (`video_headers.cpp:181`), so the CLI cannot use GM. |
| `our_ref` / `their_ref` | §5.18.9.1 / 137; §6.17.9.1 / 345 | `put_ns(writer, num_total_refs + 1, our_ref);` and `put_ns(writer, n, their_ref);` | COMPONENT | `src/encoder/inter/gme/gme_syntax.cpp:305-338` | Gate-only. Differential base-warp selection is implemented and round-tripped (`gme_syntax.cpp:384-440`). |
| `is_global` / `is_rot_zoom` | §5.18.9.1 / 138; §6.17.9.1 / 345 | `writer.put_bit(true); ... writer.put_bit(rot_zoom);` | COMPONENT | `gme_syntax.cpp:353-367` | Gate-only. Identity refs code `is_global = 0`. |
| `read_global_param()` / `decode_signed_subexp_with_ref` | §5.18.9.2-5 / 139-140; §6.17.9.2-5 / 345 | `write_signed_subexp_with_ref(...)` | COMPONENT | `gme_syntax.cpp:232-256`, `117-143` | Gate-only. Encoder validates the syntax quantisation grid (`gme_syntax.cpp:244-250`). |
| `film_grain_config()` / `apply_grain` | §5.18.10.1 / 141; §6.17.10.1 / 346 | never coded | ABSENT | every sequence writes `film_grain_params_present = false` (`still_headers.cpp:159`, `video_headers.cpp:219`); `sef.cpp:188` asserts no grain | Film grain cannot be applied to an encoded frame. Only the decoder-side synthesis model exists (`src/encoder/filters/spec_tables/film_grain.cpp`). |
| `fgm_id` / `grain_seed` | §5.18.10.1 / 141; §6.17.10.1 / 346 | never coded | ABSENT | no writer; `FG` model is decode-side only | No film-grain OBU writer either (`svc/svc_framing.h:53` only names the OBU type). |
| `film_grain_model()` (`chroma_scaling_from_luma`, `num_y_points`, point tables, AR coefficients, `grain_scaling_minus_8`, `cb/cr_*`, `overlap_flag`, `clip_to_restricted_range`, `fg_mc_identity`, `film_grain_block_size`) | §5.18.10.2 / 141-144; §6.17.10.2 / 346-348 | never coded | ABSENT | `src/encoder/filters/spec_tables/film_grain.h:22-56` is a model struct, not a writer | The full model syntax is decode/synthesis only. |
| `uniform_tile_spacing_flag` | §5.18.7.2-3 / 122-124; §6.17.7.2-3 / 339-340 | `writer.put_bit(true);  // uniform_tile_spacing_flag` | PARTIAL | `src/encoder/m1/still_headers.cpp:68`; `basic_headers.cpp:90`; `video_headers.cpp:81` | Always 1. The non-uniform form (`width_in_sbs_minus_1` / `height_in_sbs_minus_1`) is never written. |
| `increment_tile_cols_log2` | §5.18.7.3 / 124; §6.17.7.3 / 340 | `writer.put_bit(false);  // increment_tile_cols_log2 = 0` | PARTIAL | `src/encoder/m1/still_headers.cpp:69-71` | CLI always 0 (one tile). `tiles_stream.cpp:355-381` walks the real loop, and `m2/post/tiled_frame.cpp:328-355` also does, but both are gate-only. |
| `increment_tile_rows_log2` | §5.18.7.3 / 124; §6.17.7.3 / 340 | `writer.put_bit(false);  // increment_tile_rows_log2 = 0` | PARTIAL | `src/encoder/m1/still_headers.cpp:72-74` | Same. |
| `width_in_sbs_minus_1` / `height_in_sbs_minus_1` | §5.18.7.3 / 125; §6.17.7.3 / 340 | never coded | ABSENT | no writer emits a non-uniform tile grid | Explicit tile widths/heights are unreachable. |
| `context_update_tile_id` | §5.18.7.2 / 123; §6.17.7.2 / 340 | `writer.put_bits(0U, n);` | COMPONENT | `src/encoder/m2/post/tiled_frame.cpp:360-364` | Written only by the tiled M2 gate, and only with `disable_cdf_update = 1` (the code rejects adaptive CDF multi-tile at `tiled_frame.cpp:377-382`). No still path codes it because `avg_cdf_type = 1`. |
| `tile_size_bytes_minus_1` / `TileSizeBytes` | §5.18.7.2 / 123; §6.17.7.2 / 340 | `writer.put_bits(tile_size_bytes - 1, 2U);` | COMPONENT | `src/encoder/m1/tiles_stream/tiles_stream.cpp:376-380`; `m2/post/tiled_frame.cpp:365-366` | Gate-only; the CLI is one tile so the field is absent. `select_tile_size_bytes()` validates 1..4 bytes (`tiles_stream.cpp:333-353`). |
| `tile_start_and_end_present_flag` | §5.19 / 144; §6.18 / 349 | `writer.put_bit(false);  // tile_start_and_end_present_flag = 0` when `tile_count > 1` | COMPONENT | `src/encoder/m1/tiles_stream/tiles_stream.cpp:430-432`; `m2/post/tiled_frame.cpp` | Gate-only. Every emitted tile group covers the whole frame. |
| `tg_start` / `tg_end` | §5.19 / 144; §6.18 / 349 | never coded | ABSENT | `tile_group.cpp:77-91` models only the flag = 0 path | Partial tile groups (a tile subset in one OBU) cannot be expressed. |
| `tile_size_minus_1` / `le(TileSizeBytes)` | §5.20.1 / 349; §6.19 / 349 | `append_le(payload, sizes[i] - 1U, bytes);` | COMPONENT | `src/encoder/m1/tiles_stream/tiles_stream.cpp:436-444` | Gate-only; correct handling of the "last tile has no size field" rule. |

---

## Counts

| Status | Rows |
|---|---:|
| WIRED | 6 |
| PARTIAL | 22 |
| COMPONENT | 45 |
| ABSENT | 26 |
| DECODER-ONLY | 0 |
| OUT-OF-SCOPE | 3 |
| **Total** | **102** |

Notes on the counts:

- WIRED rows: CLK frame type, `base_q_idx`, `disable_cdf_update`,
  deblocking apply flags, CDEF parameters, CCSO plane enables. These are the
  only §5.18 elements the shipped binary can vary.
- The 45 COMPONENT rows are structurally explained by one fact: only
  still-picture intra frames are wired into `src/main.cpp`. The entire inter
  frame-header engine (M2, MFH, long-term, SEF, RAS, leading, TIP, GM) lives
  in gate executables.
- No row is DECODER-ONLY. Every element that has a reader also has at least a
  syntax helper on the writer side, except the bridge/atlas/MSDO group.
- The 26 ABSENT rows fall into four mechanisms: (a) a sequence-level tool is
  written 0 in every header (GDF, QM, GM, film grain, BAWP, TCQ, BRU, TIP,
  short refresh, frame-max DRL); (b) a syntactic form is not implemented
  (`found_ref`, non-uniform tiles, `tg_start/tg_end`, cropping, `df_delta_q`);
  (c) an explicitly rejected case (`frame_size_with_refs` for inter frames,
  temporal LR prediction, QM); (d) `primary_ref_frame` explicit signalling.

## Conformance rules currently violated

### V1 — RAS writes `refresh_frame_flags` that §5.18.2 derives

§5.18.2 (p.100) says for `obu_type == OBU_RAS_FRAME && max_mlayer_id == 0`:

```
refresh_frame_flags = 0
for ( i = 0; i < NumRefFrames; i++ ) { ... refresh_frame_flags |= (1 << i) ... }
```

No bits are coded. The RAS gate instead codes the generic SWITCH/regular field:

```c
// src/encoder/inter/random_access/random_access.cpp:759-765
} else if (info.derives_refresh_frame_flags) {
  // Normatively RAS derives refresh_frame_flags (§5.18.2) and codes no bits.
  if (c.ras_codes_refresh_frame_flags) {
    writer.put_bits(h.refresh_frame_flags, c.num_ref_frames);
  }
```

enabled by `config.ras_codes_refresh_frame_flags = true;` in
`src/encoder/inter/random_access/ras_gate.cpp:357`. The comment there says the
pinned `dav2d` build reads the generic field. This is a deliberate deviation
from AV2 v1.0.0 and is only tolerated because the gate streams are not
product output. Any future integration must remove it.

### V2 — OLK long-term list may be omitted

§5.18.2 (p.97) makes `num_key_ref_frames` and `ref_long_term_id[]` present for
an OLK when `long_term_frame_id_bits != 0`. The long-term component provides a
switch that suppresses them for the pinned decoder:

```c
// src/encoder/inter/longterm/longterm.h:58-61
enum class LongTermListSyntax : std::uint8_t {
  kSpecV1 = 0,           // AV2 v1.0.0 §5.18.2
  kOmitPinnedDav2d = 1,  // no num_key_ref_frames/ref_long_term_id for an OLK
};
```

`longterm_headers.cpp:366-372` skips both fields when the switch is
`kOmitPinnedDav2d`. The default is spec-conforming, but a reachable gate mode
emits a non-conforming OLK header.

### V3 — level and tile limits are not enforced on the still-picture paths

Annex A.4 (p.1110) makes these bitstream conformance requirements for the
signalled `seq_level_idx`:

```
FrameWidth * FrameHeight <= MaxPicSize
FrameWidth  <= MaxHSize
FrameHeight <= MaxVSize
TileWidth <= Tile_Width_Scaling_Factor[seq_tier][LevelIdx] * MAX_TILE_WIDTH / 4
TileWidth * TileHeight <= Tile_Area_Scaling_Factor[...] * 4096 * 2304 / 4
```

The CLI pins `seq_level_idx = 7` (level 5.1: MaxPicSize = 8,912,896,
MaxHSize = MaxVSize = 4975) but accepts any Y4M up to 65535×65535
(`src/main.cpp:499-501`) and passes it to M1/M0 unchanged. M1's tile writer
only asserts `sb_cols >= 1` and writes a single terminating increment:

```c
// src/encoder/m1/still_headers.cpp:64-75
const std::uint32_t sb_cols = (mi_cols(c) + 15U) >> 4U;
const std::uint32_t sb_rows = (mi_rows(c) + 15U) >> 4U;
assert(sb_cols >= 1U && sb_rows >= 1U);
writer.put_bit(true);  // uniform_tile_spacing_flag
if (sb_cols > 1U) writer.put_bit(false);  // increment_tile_cols_log2 = 0
```

M0's only guard is an `assert` (`src/bitstream/basic_headers.cpp:86-88`), which is
compiled out in release builds; its area bound (2304 superblocks = 9,437,184
px) is also looser than the level's `MaxPicSize` (8,912,896 px). For a frame
above the level limits the emitted header is non-conforming; for a frame wide
enough that `minLog2TileCols > 0` (above 4096 px at level 5.1) the single
zero increment is also syntactically wrong because the spec requires the
`true` increments up to `minLog2TileCols`. `m2/post/tiled_frame.cpp:328-355`
and `m1/tiles_stream/tiles_stream.cpp:355-374` do implement the real loop and
the level bounds, so this is a gap between the shipping writers and the
already-correct gate writer. Adding the `tiles::plan_tile_layout()` check to
the CLI paths would close it.

### V4 — `df_delta_q` is missing when `df_delta_q_present` is 1

§5.18.5.2 (p.117):

```
for ( i = 0; i < 4; i++ ) {
  if ( apply_deblocking_filter[ i ] ) {
    df_delta_q_present[ i ] f(1)
    if ( df_delta_q_present[ i ] ) {
      dfParBits = df_par_bits_minus_2 + 2
      df_delta_q[ i ] f(dfParBits)
      ...
```

`write_video_deblock_params()` writes the `df_delta_q_present` bit and then stops:

```c
// src/encoder/m2/post/loop_filters.cpp:459-468
for (int plane = 0; plane < 4; ++plane) {
  const bool enabled = ...;
  if (enabled) {
    writer.put_bit(df_delta_q_present != nullptr
                       ? df_delta_q_present[plane]
                       : false);
  }
}
```

No reader/writer pair sets it to 1 today, so no emitted stream is affected
yet, but the function signature invites it and the resulting header would
desynchronise the decoder. Either write `df_delta_q f(dfParBits)` after a 1
bit, or reject `df_delta_q_present[i] == true` explicitly.

### V5 (latent, not a current violation) — M1 relies on `single_picture_header_flag` for absent fields

Not a violation today, but worth recording: `still_headers.cpp` relies on
`single_picture_header_flag = 1` for the absence of `order_hint`,
`refresh_frame_flags`, `frame_size_override_flag` and the output flags. Any
future attempt to reuse this writer for a non-single-picture sequence would
silently drop those fields. `longterm_headers.cpp` and `video_headers.cpp` are the
correct models for the general case.

## Elements that are correct and should be treated as reference

- Field order for every writer matches §5.18.2 for the enabled subset:
  `cur_mfh_id`, `seq_header_id`, frame-type bit, output flags, size override,
  `order_hint`, primary-ref block, refresh, reference map, size, screen
  content, intra BC, MV precision, interpolation filter, `disable_cdf_update`,
  `tile_info`, `base_q_idx`, segmentation/QM/delta-Q, deblock/GDF/CDEF/LR/CCSO,
  TX mode, reference mode, skip mode, BAWP/WarpMV, `reduced_tx_set`, GM, film
  grain.
- M1's deblocking conditional is exact: chroma flags and
  `df_delta_q_present` bits appear only when the corresponding plane is
  enabled (`still_headers.cpp:180-197`).
- M1's CDEF writer implements the `4 → 3` secondary-strength mapping and the
  `y_pri_zero` / `uv_pri_zero` shortcut (`still_headers.cpp:213-225`).
- M1's CCSO writer implements the `bo_only` shortcut, `f(2 + bo_only)` band
  bits, and `tu(7)` offset indices (`still_headers.cpp:241-254`).
- `m2/post/tiled_frame.cpp:328-369` implements the full
  `increment_tile_cols_log2` / `increment_tile_rows_log2` loops and
  `context_update_tile_id`, and validates the level tile bounds.
- `mfh_video_adapter.cpp` correctly omits `seq_header_id_in_frame_header` when
  `cur_mfh_id != 0` and inherits `mfh_frame_*` when the MFH has a size.
- `gme_syntax.cpp` round-trips `global_motion_params()` including
  `scale_warp_model`, `our_ref`/`their_ref`, and the subexp coding.
- `longterm_headers.cpp` / `longterm_pool.cpp` enforce the long-term id
  range, the reserved `(1 << bits) - 1` value, and the "non-output requires a
  non-zero refresh" rule.

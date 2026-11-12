# Spec audit 01 — §5.2–§5.17 OBU syntax and §6.2–§6.16 OBU semantics

Scope: AV2 v1.0.0 PDF pages **52–93** (`§5.2`–`§5.17`, every OBU type, OBU
header/extension, trailing bits, sequence-header config groups, metadata,
layer config, operating points, BRT, QM, film grain, content interpretation,
padding, atlas, MSDO) and pages **255–319** (`§6.2`–`§6.16`, the matching
semantics). Read-only audit of `/Users/arin/work/av2enc`.

Spec identity: `spec/AV2_Spec_v1.0.0.pdf`, AV2 Bitstream & Decoding Process
Specification v1.0.0, 2026-05-28, commit `38f28e7`. Page numbers are physical
PDF pages (`pypdf` index = page − 1). Section text was extracted with `pypdf`
from the PDF; syntax was never guessed from READMEs.

Status legend:

| Status | Meaning |
|---|---|
| **WIRED** | The shipped `av2enc` CLI (`src/main.cpp` → M0/M1 still path) emits the syntax. |
| **COMPONENT** | A writer exists, is unit-tested and (usually) decoder-gated, but no `av2enc` CLI flag reaches it. |
| **PARTIAL** | A writer/codec exists but required sub-fields, byte(s) or value branches are missing; often the bit is hardcoded off or fixed to one value. |
| **ABSENT** | No encoder-side syntax writer. |
| **DECODER-ONLY** | No encoder obligation; notes whether the encoder can produce a stream that exercises it. |
| **OUT-OF-SCOPE** | Atlas, MSDO, bridge, padding, inter-layer prediction only (per audit brief); existence of code is still noted. |

Production-path fact used throughout: `av2enc` encodes **one still picture**
with M0 (`TD + SEQ + CLK`) or the M1 still path
(`src/main.cpp:121-160`, `src/main.cpp:573-691`). The M2 IPPP writer
(`src/encoder/m2/video_headers.cpp`) and every gate-specific writer are not
reachable from the CLI. `src/bitstream/obu.{h,cpp}` is a stub
(`src/bitstream/obu.cpp:18-36`); the real OBU writer is
`src/bitstream/basic_headers.cpp`.

Evidence abbreviations:
`m0` = `src/bitstream/basic_headers.cpp`;
`m1` = `src/encoder/m1/still_headers.cpp`;
`m2` = `src/encoder/m2/video_headers.cpp`;
`small` = `src/encoder/syntax/small_header/small_header.cpp`;
`filters` = `src/encoder/m1/filters_stream/filters_stream.cpp`;
`dq` = `src/encoder/quant/dcdeltaq/syntax.cpp`;
`qm` = `src/encoder/qm/qm.cpp`;
`lcr/ops` = `src/encoder/svc/svc_framing.cpp`;
`md` = `src/encoder/hdr/hdr_metadata.cpp`;
`mdt` = `src/encoder/metadata/types/metadata_types.cpp`;
`ci` = `src/encoder/metadata/content_interpretation/content_interpretation.cpp`;
`mfh` = `src/encoder/frame/mfh/mfh.cpp`;
`brt` = `src/encoder/rc/hrd/hrd.cpp`;
`fg` = `src/encoder/filters/spec_tables/film_grain.cpp`.

Baseline tests run for this audit (all pass): `metadata_types_test`
(PASS), `hdr_metadata_test` (PASS), `content_interpretation_test` (all
checks), `film_grain_test` (611 checks), `qm_test` (258,446), `mfh_test`
(201), `mfh_gate` (6, dav2d 0 diffs), `hrd_test` (510) + `hrd_stream_gate`
(PASS), `svc_test` (PASS), `tlayers` gate (PASS), `dcdeltaq_test` (10.9M),
`longterm_test` (327), `small_header` gate (PASS), `sdp_test` (PASS).

---

## 1. §5.2 / §6.2 — OBU envelope, header and framing

| Item | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| `obu_header()` bit layout (`obu_header_extension_flag f(1)`, `obu_type f(5)`, `obu_tlayer_id f(2)`) | §5.2.2 p54; §6.2.2 p255-257 | **WIRED** | `m0:106-123`; test `tests/basic_headers_test.cpp:208-231` | Only `obu_header_extension_flag = 0` is emitted by M0/M1. `src/bitstream/obu.cpp` is a non-functional stub, so there is no shared OBU header parser/writer outside `basic_headers`. |
| `obu_header_extension_flag = 1` path: `obu_mlayer_id f(3)`, `obu_xlayer_id f(5)` | §5.2.2 p54; §6.2.2 p257-258 | **COMPONENT** | `src/encoder/svc/svc_framing.cpp:247-303` (encode/decode); `src/encoder/svc/svc_test.cpp` round-trips | No CLI path; M0/M1 never set the flag. `GLOBAL_XLAYER_ID` inference for TD/MSDO is implemented (`svc_framing.cpp:196-205,296-300`). |
| `obu_extension_data(sz)` for extensible OBUs | §5.2.1 p53-54; §6.2.1 p255 | **DECODER-ONLY** | No writer sets `obu_extension_flag = 1`; the CI payload reader explicitly rejects extension data (`ci.cpp:903-910`) | §6.2.1 requires `obu_extension_flag == 0` in conforming bitstreams, so the encoder must *not* emit extension data. The encoder cannot produce a stream exercising it; correct. |
| `obu_type` value space and `reserved_obu()` | §5.2.2 p54; §6.2.2 Table 6.1 p256; §6.3 p258 | **WIRED** (only 1/2/4 reachable) | `m0:26-31`; `m0:106-123` | Reserved OBU types 0 and 26–31 have no writer (correct: reserved for AOMedia). Other types are component-only; see §2. |
| `obu_tlayer_id` | §5.2.2 p54; §6.2.2 p257 | **WIRED** (fixed 0) | `m0:114`; `m1:80-159` never sets a non-zero tlayer | Multi-temporal-layer OBU headers exist only in components (`tlayers` gate writes `obu_tlayer_id` 0..3: `src/encoder/inter/tlayers/tlayers_syntax.cpp:117`, gate PASS). Conformance checks (`tlayer <= max_tlayer_id`, key/RAS/SWITCH forced 0) exist only in component validators (`random_access.cpp:1020-1035`, `leading.cpp:343-358`). |
| Annex B.2 length prefix `leb128(size)` / `obuPayloadSize` | §5.2.1 p52; Annex B.2 | **WIRED** | `m0:125-165`; `tests/basic_headers_test.cpp` bit goldens | No `obu_has_size_field` bit (correct for AV2). `leb128` writer caps at 32 bits. |
| `trailing_bits()` (single 1 then zeros, byte alignment) | §5.2.3 p54; §6.2.3 p258 | **WIRED** | `m0:54-57,283`; `m1:52-55,279`; tests `basic_headers_test.cpp:289-310` | Tile-group OBUs correctly have no OBU trailing bits (`usedArith`, `m0:341-349`). |
| `byte_alignment()` (zero bits to the next byte) | §5.2.4 p55; §6.2.4 p258 | **WIRED** | `src/bitstream/bitwriter.cpp:30-34`; `m0:335` | — |
| `usedArith = is_tile_group()` payload boundary | §5.2.1 p53 | **WIRED** | `m0:148-153,341-349`; `m2:328-338` | Tile payloads are out of this audit's scope. |
| Extensible-OBU tail selection (`is_extensible_obu()` ⇒ `obu_extension_flag` before `trailing_bits`) | §5.2.1 p53-54 | **PARTIAL** | SEQ/CI/LCR/OPS write `obu_extension_flag = 0`: `m0:282`, `ci.cpp:669-672`, `lcr/ops:508`, `lcr/ops:601` | **MFH omits the bit**: `mfh:427-443` writes MFH syntax then `trailing_one_bit`, although p54 lists `OBU_MULTI_FRAME_HEADER` in `is_extensible_obu()`. A conforming decoder reads the trailing `1` as `obu_extension_flag = 1`; §6.2.1 requires it to be 0. The pinned dav2d tolerates this (gate PASS), so the gate cannot catch it. |
| `load_xlayer_context()` / `save_xlayer_context()` per OBU | §5.2.1 p52-53 | **DECODER-ONLY** | No writer-side equivalent | Encoder emits a single xlayer per still; no per-OBU context stack needed. |

## 2. §5.2.2 / §6.2.2 — every OBU type (Table 6.1 p256)

| Item | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| 0 — Reserved | p256 | **DECODER-ONLY** | no writer | Correct (must not be emitted). |
| 1 — `OBU_SEQUENCE_HEADER` | p256 | **WIRED** | `m0:273-287`; `m1:275-282` | Only the single-picture / layer-0 forms; see §3. |
| 2 — `OBU_TEMPORAL_DELIMITER` | p256 | **WIRED** | `m0:351-355`; `m1:264-267` | Empty payload, no trailing bits (correct, §5.5 note p69). |
| 3 — `OBU_MULTI_FRAME_HEADER` | p256 | **COMPONENT** | `mfh:384-443`; `mfh_test` 201 checks; `mfh_gate` 6 checks, dav2d 0 diffs | Missing extensible-OBU flag (row above). No CLI flag; MFH inheritance resolver exists but no production frame header reads it. |
| 4 — `OBU_CLOSED_LOOP_KEY` | p256 | **WIRED** | `m0:341-349`; `m1:284-304`; `m2:328-338` | — |
| 5 — `OBU_OPEN_LOOP_KEY` | p256 | **COMPONENT** | `src/encoder/inter/leading/leading.cpp:328-339` (`kOpenLoopKey`); `leading_gate` | No CLI path. Long-term `num_key_ref_frames` also written (`longterm_headers.cpp:365-371`). |
| 6 — `OBU_LEADING_TILE_GROUP` | p256 | **COMPONENT** | `leading.cpp:328-339,443-465`; `leading_gate` | Leading output ordering required; no production path. |
| 7 — `OBU_REGULAR_TILE_GROUP` | p256 | **COMPONENT** | `m2:335-337`; `drl_stream.cpp:256-257`; `gme_gate.cpp:726` | The CLI never emits type 7 (single still = CLK); M2/PP gates do. |
| 8 — `OBU_METADATA_SHORT` | p256 | **COMPONENT** | `md:264-293`; `mdt:361-375`; `run_metadata_gate.sh`; `hdr_metadata_test` | Not reachable from `av2enc`; only from `append_hdr_metadata` / `append_metadata_types`. Pinned dav2d does not implement §5.17.2/3, so the gate proves decode-invariance + self-parse only. |
| 9 — `OBU_METADATA_GROUP` | p256 | **COMPONENT** | `md:318-397`; `mdt:387-395`; `run_metadata_gate.sh` | Same as above; `LAYER_VALUES` writer absent (see §5.17). |
| 10 — `OBU_SWITCH` | p256 | **COMPONENT** | `src/encoder/inter/random_access/random_access.cpp:1193,1314-1321`; `random_access` README: three-frame dav2d gate | No CLI path; no M2 integration. |
| 11 — `OBU_LEADING_SEF` | p256 | **COMPONENT** | `src/encoder/inter/sef/sef.cpp:215-217,308-316`; `sef_gate` | No CLI path. |
| 12 — `OBU_REGULAR_SEF` | p256 | **COMPONENT** | `sef.cpp:215-217`; `sef_gate` | No CLI path. |
| 13 — `OBU_LEADING_TIP` | p256 | **ABSENT** | TIP decision/syntax exists (`inter/temporal/tip.cpp`, `temporal_syntax.cpp`) but nothing wraps a TIP frame in OBU type 13 | No OBU writer for TIP frames; `is_tip_frame()` only recognized in readers (`content_interpretation.cpp:943-946`). |
| 14 — `OBU_REGULAR_TIP` | p256 | **ABSENT** | same | same. |
| 15 — `OBU_BUFFER_REMOVAL_TIMING` | p256 | **COMPONENT** | `brt:473-489,941-949`; `hrd_test`; `hrd_stream_gate` PASS (self-parse; pinned dav2d rejects type 15) | No CLI path; decoder never consumes it. |
| 16 — `OBU_LAYER_CONFIGURATION_RECORD` | p256 | **COMPONENT** | `lcr/ops:393-511`; `svc_test`; `svc/gate_dav2d.sh` (dav2d ignores LCR; avmdec validates OPS selection) | Writer is field-incomplete (§5.8 below). |
| 17 — `OBU_ATLAS_SEGMENT` | p256 | **OUT-OF-SCOPE** | No writer; only enum + ordering knowledge (`svc_framing.h:47`, `svc_framing.cpp:228-231`) | No `atlas_segment_info_obu()` implementation anywhere. |
| 18 — `OBU_OPERATING_POINT_SET` | p256 | **COMPONENT** | `lcr/ops:516-604`; `svc_test`; `svc/gate_dav2d.sh` (avmdec `--select-ops`) | Field-incomplete (§5.10/§5.11 below). |
| 19 — `OBU_BRIDGE_FRAME` | p256 | **OUT-OF-SCOPE** | `random_access` derivations mention bridge only; no OBU writer | No bridge frame header, no `bridge_frame_ref_idx`. |
| 20 — `OBU_MSDO` | p256 | **OUT-OF-SCOPE** | No writer; enum + `GLOBAL_XLAYER_ID` special case only (`svc_framing.cpp:196-205`) | No `multistream_decoder_operation_obu()`. |
| 21 — `OBU_RAS_FRAME` | p256 | **COMPONENT** | `random_access.cpp:1227,1314-1321`; `random_access` gate | No CLI path; RAS requires `long_term_frame_id_bits > 0` from a component header. |
| 22 — `OBU_QUANTIZATION_MATRIX` | p256 | **COMPONENT** | `qm:399-430`; `qm_test` 258k checks; `qm_gate.sh` | No CLI path. Pinned dav2d rejects all QM OBUs (`qm_gate.sh:9-10`), so only `using_qmatrix` with *default* matrices is decoder-gated; user QM data is never decoder-validated. |
| 23 — `OBU_FILM_GRAIN` | p256 | **ABSENT** | No `film_grain_obu()` writer; enum value only (`svc_framing.h:53`) | `fgm_update_flags`, `fgm_chroma_idc`, `film_grain_model()` and slot bookkeeping are unimplemented. Film-grain *synthesis* exists (`fg:57-560`, `film_grain_test` 611 checks) but is output-side only. |
| 24 — `OBU_CONTENT_INTERPRETATION` | p256 | **COMPONENT** | `ci:583-672`; `ci` unit tests; `run_ci_gate.sh`; also a second writer in `brt:491-520,951-962` | No CLI path. Two independent implementations (CI module and HRD module) risk drift. |
| 25 — `OBU_PADDING` | p256 | **OUT-OF-SCOPE** | No writer | No `padding_obu()`; no in-place padding conversion helper. |
| 26–31 — Reserved | p256 | **DECODER-ONLY** | no writer | Correct. |

## 3. §5.4.1–§5.4.13 / §6.4 — sequence header and config groups

### 3.1 §5.4.1 general sequence header (p55–58 / semantics p258–264)

| Item | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| `seq_header_id uvlc()` | §5.4.1 p55; §6.4.1 p258 | **PARTIAL** | `m0:176` (`0`); `m1:80`; `m2:96`; `chroma_headers.cpp:336` | Only id 0. Reader rejects anything else (`chroma_headers.cpp:503-508`); no `MAX_SEQ_NUM`-wide header bank, so `seq_header_id_in_frame_header` can only select 0. |
| `seq_profile_idc f(5)` | p55; p258 | **COMPONENT** | CLI writes 0 (`m0:177`); profiles 3/4 via `chroma_format.cpp:44-60`; `chroma/gate.sh` decodes 4:2:2/4:4:4 | CLI is profile 0 only. Profile 1/2 never emitted. |
| `single_picture_header_flag f(1)` | p55; p258-259 | **WIRED** | `m0:178` (1); `m2:98` (0) | Non-single-picture form exists only in components (M2/prediction gates). |
| `seq_level_idx f(5)` | p55; p258 | **WIRED** | `m0:179` (config, default 7 = level 5.1); `m1:83` | Levels other than 7 not exposed by CLI. |
| `seq_tier f(1)` (only when `seq_level_idx > 3 && !single_picture`) | p55; p258 | **PARTIAL** | `m2:100`, `tlayers_syntax.cpp:110`, `random_access.cpp:270` all write `0` | High tier (1) is never emitted; only main-tier streams exist although tile-limit code accepts tier 1 (`tiles/tile_layout.cpp:100-120`). |
| `chroma_format_idc uvlc()` | p55; p259 Table 6.2 | **COMPONENT** | CLI 0 (`m0:181`); 0/1/2/3 writer + decoder gate `chroma_headers.cpp:329-376`, `chroma_format.cpp:35-65`, `chroma/gate.sh` | CLI cannot encode 4:0:0/4:2:2/4:4:4. |
| `bit_depth_idc uvlc()` | p55; p259-260 Table 6.3 | **COMPONENT** | CLI 1 = 8-bit (`m0:182`); 0 = 10-bit in `hbd_headers.cpp:99`, `high_bitdepth_headers.cpp:98`, gates `build-hdr`/`build-m10` | Spec allows only 0/1 (10/8-bit); 12-bit is not representable by the syntax and is not emitted. |
| `set_chroma_format_and_bit_depth()` (SubsamplingX/Y, Monochrome, NumPlanes, BitDepth, MaxQ) | §6.4.1 p259-260 | **DECODER-ONLY** | Encoder mirror: `chroma_format.cpp:35-65`, `format/format.cpp:330-350` | Encoder can produce all four chroma idcs and both bit depths (components), so the inference is exercisable. |
| `seq_lcr_id f(3)` | p56; p260-261 | **PARTIAL** | `m2:105` writes 0; `svc_mux.cpp:225` sets plan id 1 but never rewrites the sequence header | No stream association: composed SVC streams keep `seq_lcr_id = 0`, so the emitted LCR is not referenced by any sequence header. §6.4.1's LCR-association rules cannot be satisfied by an emitted stream. |
| `still_picture f(1)` | p56; p260 | **WIRED** | M0/M1 single-picture infers 1; `m2:106` writes 0 | — |
| `max_tlayer_id f(2)` | p56; p261 | **COMPONENT** | single-picture infers 0; `tlayers_syntax.cpp:117` writes 1..3; TLayers gate decodes hierarchical GOP | No CLI path. `max_tlayer_id > 0` requires non-single-picture headers. |
| `max_mlayer_id f(3)` | p56; p261 | **PARTIAL** | `m2:108` writes 0; `tlayers_syntax.cpp:118` writes 0 | No writer emits >0; inter-layer prediction/embedded layers are out of scope, but then the syntax branch is dead. |
| `seq_max_mlayer_cnt_minus_1 f(n)` | p56; p261 | **ABSENT** | no writer | Only reachable with `max_mlayer_id > 0`; no `CeilLog2` field writer, no `SeqMaxMlayerCnt > 1`. |
| `monotonic_output_order_flag f(1)` | p56; p258-259 | **COMPONENT** | `m2:110` (1); `tlayers_syntax.cpp:120`; `leading.cpp:102` can write 0 | CLI single-picture infers 1; no CLI path to reordering. |
| `frame_width_bits_minus_1 f(4)`, `max_frame_width_minus_1 f(n)` | p56; p261 | **WIRED** | `m0:186-199`; all writers compute minimal width bits | Width up to 65535 (16-bit field cap, `m0:194`). |
| `frame_height_bits_minus_1 f(4)`, `max_frame_height_minus_1 f(n)` | p56; p261 | **WIRED** | `m0:190-199` | same. |
| `seq_cropping_window_present_flag f(1)` + four `uvlc()` offsets | p56; p261 | **COMPONENT** | `small:402-470`, `chroma_headers.cpp:366-373`; `small_header` gate `still_360s/crop` decoded 0 diffs | CLI always writes 0 (`m0:202`); no cropping/display-size CLI option. Range checks exist (`small_header.cpp` validation; `chroma_headers.cpp:292-300`). |
| `seq_initial_display_delay_present_flag f(1)`, `seq_initial_display_delay_minus_1 f(4)` | p56; p262 | **COMPONENT** | `brt:198-205` writer, `307-310` splice; `hrd_test`; `hrd_stream_gate` | CLI infers/omits the block. Pinned dav2d accepts the delay fields (gate PASS) but not the nested decoder-model block. |
| `decoder_model_info_present_flag f(1)`, `num_units_in_decoding_tick f(32)` | p56-57; p262 | **COMPONENT** | `brt:203-208`, `307-316`; `hrd_stream_gate` decodes the "decoder-compatible" stream | CLI always 0 (`m0:203`). |
| `seq_decoder_model_info_present_flag f(1)` + `seq_decoder_model_info()` | §5.4.13 p69; p262 | **COMPONENT** | `brt:198-202`, `316`; `hrd_test` | Pinned dav2d does **not** parse the spec `seq_decoder_model_info()` block; `hrd_stream_gate` reports this and gates the full form by self-parse only. |
| Derived `TLayerDependencyMap` / `MLayerDependencyMap` initialization | p57; §6.4.2 p264 | **PARTIAL** | `tiles/svc_layers.h:48-78` models both maps; `tlayers` validates tlayer maps; validation tests | MLayer map is validation-only; no sequence writer emits a >1 embedded-layer map. Presence-map closure is not written or checked at the bitstream level. |
| `mlayer_dependency_present_flag f(1)` + `mlayer_dependency_map f(1)` | p57; p264 | **ABSENT** | no writer (grep: only `svc_layers.h` data model) | Dead branch because `max_mlayer_id` is never >0. |
| `tlayer_dependency_present_flag f(1)` + `tlayer_dependency_map f(1)` | p57-58; p264 | **COMPONENT** | `tlayers_syntax.cpp:143-154`; TLayers gate PASS | Gate uses the spec-default map (flag 0) because the pinned dav2d expects a different later syntax (`tlayers.h:70-78`). Explicit-map streams are spec-only, not decoder-verified. |
| `multi_tlayer_dependency_map_present_flag f(1)` | p57; p264 | **ABSENT** | no writer | Requires `max_mlayer_id > 0`. |
| `save_sequence_header()` / header bank | p58; p264 | **PARTIAL** | Writers serialize a header; no id-indexed store (only id 0) | Multi-sequence streams impossible. |

### 3.2 §5.4.2 sequence tile config (p59; §6.4.2 p264)

| Item | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| `seq_tile_info_present_flag f(1)` | p59; p264 | **COMPONENT** | `small:1359-1395`; gate `still_360s/tile` 0 diffs. M0/M1/M2 write 0 (`m0:265`, `m1:157`, `m2:217`) | CLI always 0, so tiles are always frame-level. |
| `allow_tile_info_change f(1)` | p59; p264 | **COMPONENT** | `small:1359-1380`; gate `allow_change = true` (`small_header/gate.cpp:656`) | Not reachable from CLI. |
| `tile_params()` / `get_seq_sb_size()` at sequence level (`SeqSbRowStarts`, `SeqTileRowsLog2`, etc.) | p59; p264 | **COMPONENT** | `small:1014-1370`; gate tile stream decodes 0 diffs; frame-level multi-tile writer `m1/tiles_stream/tiles_stream.cpp:356-361` | Sequence-level form is exercised by the `still_360s/tile` gate (0 diffs). CLI emits one tile only. |

### 3.3 §5.4.3 sequence partition config (p59–60; §6.4.3 p264–265)

| Item | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| `use_256x256_superblock f(1)` | p59; p264 | **COMPONENT** | `sb256_stream.cpp:415-418`; `sb256` gate | CLI is 64x64 SB only. |
| `use_128x128_superblock f(1)` | p59; p265 | **COMPONENT** | `sb128_stream.cpp:455-457`; `sb128` gate | same. |
| `enable_sdp f(1)` | p59; p265 | **PARTIAL** | Tool + tree + syntax model: `src/encoder/sdp/sdp.cpp`, `sdp_syntax.cpp`, `sdp_test` PASS; all header writers hardcode 0 (`m0:209`, `m1:105`, `m2:132`) | No writer sets it; no decoder-gated SDP stream (README: header-only patch is insufficient, `sdp/README.md:320-332`). |
| `enable_extended_sdp f(1)` | p59; p265 | **ABSENT** | `sdp.h:76-85` validates the flag but no writer emits it | Only meaningful for inter frames; production path is single-picture. |
| `enable_ext_partitions f(1)` | p59; p265 | **PARTIAL** | Syntax/decision model + gate: `partition_ext/partition_ext.cpp`, `ext_syntax.cpp`; `gate_dav2d.py:13-14` rewrites the header bit to 1 | Production writers force 0 (`m1:106`); only the Python gate writes the bit. |
| `enable_uneven_4way_partitions f(1)` | p59; p265 | **PARTIAL** | Same gate rewrites bit to 1; `partition_ext.cpp:706-709` validates dependency | Same: no C++ header writer sets it. |
| `reduce_pb_aspect_ratio f(1)` | p59; p265 | **COMPONENT** | `small:514-580`; gate `still_360s/aspect` 0 diffs + negative control | CLI always 0. |
| `max_pb_aspect_ratio_log2_minus_1 f(1)` | p59; p265 | **COMPONENT** | `small:514-540`; gate negative control proves the cap parses | same. |

### 3.4 §5.4.4 sequence segment config (p60; §6.4.4 p265–266)

| Item | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| `enable_ext_seg f(1)` | p60; p265 | **COMPONENT** | `small:755-762`; gate `f.segment.enable_ext_seg = true` (`small_header/gate.cpp:629,651`) | CLI writes 0 (`m0:214`); 16-segment path only in gate. |
| `seq_seg_info_present_flag f(1)` | p60; p266 | **COMPONENT** | `small:755-762`; gate `present = true` | same. |
| `seq_allow_seg_info_change f(1)` | p60; p266 | **COMPONENT** | `small:755-762`; `validate_sequence_segment_config` (`small:740-754`) | Writer supports; gate uses both values. |

### 3.5 §5.4.5 sequence intra config (p60; §6.4.5 p266)

| Item | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| `enable_dip f(1)` | p60; p266 | **PARTIAL** | DIP prediction + block syntax: `encoder/dip/dip.cpp`, `dip_syntax.cpp`, `dip_test`; every sequence writer hardcodes 0 (`m0:218`, `m1:114`, `m2:141`) | No header writer sets it and no decoder-gated DIP stream exists. |
| `enable_intra_edge_filter f(1)` | p60; p266 | **WIRED** | `m1:115` writes config, M1 default true (`still_headers.h:64`); enabled by `src/encoder/m1/still_encoder.cpp:1813` unless `AV2ENC_STILL_NO_EDGE`; M2 writes 1 (`m2:142`) | Fixed-on in M1; only the env var can disable it. |
| `enable_mrls f(1)` | p60; p266 | **COMPONENT** | `m1/mrl_stream/mrl_stream.cpp:539-573,863` writes 1; `mrl_gate` | CLI emits 0 (`m1:116`). |
| `enable_cfl_intra f(1)` | p60; p266 | **WIRED** | `m1:117`; CLI `--no-cfl` (`main.cpp:283-286,634`) | — |
| `cfl_ds_filter_index f(2)` | p60; p266 | **PARTIAL** | CLI writes 0 (`m1:118`); all header writers 0; `cfl/ds_filters/ds_filters.cpp` implements filters 0..2 with gate | No header writer can emit 1/2/3; spec value 3 behaves as 0. |
| `enable_mhccp f(1)` | p60; p266 | **PARTIAL** | MHCCP prediction + syntax: `cfl/mhccp/mhccp.cpp`, `mhccp_syntax.cpp`, tests; header writers force 0 (`m1:119`) | `mhccp/README.md:158-191` explicitly lists the missing M1 header edit; no decoder-gated stream. |
| `enable_ibp f(1)` | p60; p266 | **PARTIAL** | `ibp/ibp.cpp`, syntax group writer `ibp.cpp:714-742`, `ibp_test`; header writers force 0 (`m1:120`) | `ibp/README.md:250-252`: no stream gate, M1/M2 integration pending. |

### 3.6 §5.4.6 sequence inter config (p60–63; §6.4.6 p266–269)

| Item | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| `seq_enabled_motion_modes[i] f(1)` (INTERINTRA..EXTENDWARP) | p61; p266 | **COMPONENT** | `m2/advanced/advanced_syntax.cpp:1138-1160`; `advanced_test.cpp:1348-1390`; production writers zero all modes (`m2:150-152`) | No stream enables a warp/motion mode except default. |
| `seq_frame_motion_modes_present_flag f(1)` | p61; p266 | **COMPONENT** | `advanced_syntax.cpp:1138-1160`; advanced test | Not reachable. |
| `enable_six_param_warp_delta f(1)` | p61; p267 | **COMPONENT** | `advanced_syntax.cpp:1138-1160`; `advanced_test` | Not reachable. |
| `enable_masked_compound f(1)` | p61; p267 | **COMPONENT** | `advanced_syntax.cpp:1157`; decoder-gated stream `m2/multiref/gate/compound_stream.cpp:132` writes 1 | Not reachable from CLI. |
| `enable_ref_frame_mvs f(1)` / `reduced_ref_frame_mvs_mode f(1)` | p61; p267 | **COMPONENT** | `inter/temporal/temporal_syntax.cpp:61-67`; `temporal_test` round-trip; motion-field implementation `temporal/motion_field.cpp` | No full-stream writer sets 1; no decoder gate. |
| `order_hint_bits_minus_1 f(4)` | p61-62; p267 | **COMPONENT** | `m2:157`; many gates (M2, compound, avg_cdf) | CLI single-picture forces `OrderHintBits = 0`. |
| `enable_refmvbank f(1)` | p62; p267 | **COMPONENT** | `temporal_syntax.cpp:68`; `ref_mv_bank.cpp`; `temporal_test` | No stream writes 1. |
| `disable_drl_reorder f(1)`, `constrain_drl_reorder f(1)` | p62; p267 Table 6.4 | **COMPONENT** | `m2:159` (disabled); `drl_reorder/gate/drl_stream.cpp:136-138`; DRL gate decoder-verified | CLI only disabled. |
| `explicit_ref_frame_map f(1)` | p62; p267 | **COMPONENT** | `m2:160`; compound/avg_cdf/drl gates | — |
| `explicit_num_ref_frames f(1)`, `num_ref_frames_minus_1 f(4)` | p62; p267 | **COMPONENT** | `m2:161-162`; gates use 2/4 refs | CLI single-picture infers 2. |
| `long_term_frame_id_bits f(3)` | p62; p267 | **COMPONENT** | `longterm_headers.cpp:268`; `longterm_gate` decodes CLK/OLK + LTR streams | CLI emits 0. |
| `seq_max_drl_bits_minus_1 ns(n)` | p62; p267 | **COMPONENT** | `m2:164`; drl gate | Value variable in M2 config only. |
| `allow_frame_max_drl_bits f(1)` | p62; p268 | **ABSENT** | all writers hardcode 0; frame-level `change_drl` is never written | Tool decision exists (`drl_reorder.cpp:45`) but no syntax path. |
| `seq_max_bvp_drl_bits_minus_1 ns(n)` | p62; p268 | **WIRED** | `m0:230`, `m1:125`, `m2:166` write value 0 | Only value 0; IBC's BVP stack is not configurable at the header. |
| `allow_frame_max_bvp_drl_bits f(1)` | p62; p268 | **WIRED** (fixed 0) | `m0:231`; `m1:126` | Frame-level `change_bvp_drl` never written. |
| `num_same_ref_compound f(2)` | p62; p268 | **PARTIAL** | all writers write 0 (`m2:168`); `m2/multiref/ref_map.cpp:43-55` accepts non-zero input | No header writer emits >0; `random_access.cpp:566` rejects non-zero. |
| `enable_tip f(1)`, `disable_tip_output f(1)`, `enable_tip_hole_fill f(1)` | p62; p268 | **COMPONENT** | `temporal_syntax.cpp:70-78`; TIP decision/fill `temporal/tip.cpp`, `motion_field.cpp`; `temporal_test` | No stream sets `enable_tip = 1`; no OBU type 13/14 output. |
| `enable_mv_traj f(1)` | p62; p268 | **COMPONENT** | `temporal_syntax.cpp:79`; trajectory in `motion_field.cpp:345-386,466,661`; tests | No stream sets 1. |
| `enable_bawp f(1)` | p62; p268 | **COMPONENT** | `m2/advanced/advanced_syntax.cpp:1161-1169`; `advanced_test`; BAWP decision `inter/advanced/bawp.cpp` | No stream sets 1; no decoder gate. |
| `enable_cwp f(1)` | p62; p268 | **COMPONENT** | `advanced_syntax.cpp`; compound gate writes 1 (`compound_stream.cpp:149`) | Not reachable. |
| `enable_imp_msk_bld f(1)` | p62; p268 | **COMPONENT** | `advanced_syntax.cpp:1167`; compound gate writes 1 (`compound_stream.cpp:150`); `compound_predict.cpp:240` | Not reachable. |
| `enable_df_sub_pu f(1)` | p62; p268 | **PARTIAL** | `temporal_syntax.cpp:73` writes the bit; no frame-level `allow_df_sub_pu` / DF-sub-PU process | Syntax only; enabling it would desynchronise the frame header. |
| `enable_tip_explicit_qp f(1)` | p63; p268 | **PARTIAL** | `temporal_syntax.cpp:80-83` writes the conditional bit | TIP QP signalling not implemented; no stream. |
| `enable_opfl_refine f(2)` | p63; p269 Table 6.5 | **COMPONENT** | `temporal_syntax.cpp:84`; `temporal/opfl.cpp`; tests | No stream sets non-zero. |
| `enable_refinemv f(1)` | p63; p269 | **COMPONENT** | `temporal_syntax.cpp:85`; `temporal/refinemv.cpp`; tests | No stream sets 1. |
| `enable_tip_refinemv f(1)` | p63; p269 | **COMPONENT** | `temporal_syntax.cpp:86-89`; `tip.h:19-43` gating; tests | No stream. |
| `enable_bru f(1)` | p63; p269 | **ABSENT** | every writer writes 0; no encoder `use_bru` producer (only restoration read-side skip `restoration.cpp:346`) | No BRU update tool; bit can never be 1. |
| `enable_adaptive_mvd f(1)` | p63; p269 | **ABSENT** | every writer 0; no implementation anywhere | — |
| `enable_mvd_sign_derive f(1)` | p63; p269 | **PARTIAL** | IBC block syntax supports the flag (`m1/ibc_stream/ibc_syntax.cpp:632,674`); sequence writers force 0 | No header can advertise it; IBC adapter explicitly rejects `true` (`ibc_syntax.cpp:511`). |
| `enable_flex_mvres f(1)` | p63; p269 | **PARTIAL** | `inter/me2/me2.cpp:69` and tests implement per-block precision; gme test writes the bit (`gme_test.cpp:178`) | Production/gate sequence headers write 0; no full stream. |
| `enable_global_motion f(1)` | p63; p269 | **COMPONENT** | `gme_gate.cpp:249` writes `enable_gm`; gme gate decoder-verified; `gme.cpp` | Not reachable. |
| `enable_short_refresh_frame_flags f(1)` | p63; p269 | **PARTIAL** | `longterm_headers.cpp:287` can write the bit; `longterm_pool.cpp:137-140` rejects 1 | Tool not modelled; only 0 legal in practice. |

### 3.7 §5.4.7 sequence SCC config (p63–64; §6.4.7 p270)

| Item | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| `seq_choose_screen_content_tools f(1)` | p63; p270 | **PARTIAL** | only non-single writers code it, always 0 (`m2:186`); single-picture infers SELECT | No writer emits 1. |
| `seq_force_screen_content_tools f(1)` | p63; p270 | **PARTIAL** | `m2:187` writes 0; single-picture infers SELECT | "Force off" only; `SELECT` is reached via the single-picture inference, so IBC works, but no force-on stream. |
| `seq_choose_integer_mv f(1)` | p63; p270 | **ABSENT** | branch never entered (requires force > 0) | — |
| `seq_force_integer_mv f(1)` | p63-64; p270 | **ABSENT** | branch never entered; inferred SELECT otherwise | — |

### 3.8 §5.4.8 sequence transform/quant/entropy config (p64–66; §6.4.8 p270–271)

| Item | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| `enable_fsc f(1)` | p64; p270 | **COMPONENT** | `lossless.cpp:2837-2839`; `lossless_gate` decodes FSC streams | CLI emits 0; FSC not reachable. |
| `enable_idtx_intra f(1)` | p64; p270 | **PARTIAL** | `m1:131` writes config (default false); lossless writes 0 when FSC off | IDTX transform implemented (`tx_variants`) but the sequence bit can only be 0 in reachable paths; no IDTX stream gate. |
| `enable_intra_ist f(1)` | p64; p270 | **COMPONENT** | `tx_secondary.cpp:458`; `dq:288`; `tx_secondary_test` | No stream sets 1. |
| `enable_inter_ist f(1)` | p64; p270 | **COMPONENT** | `tx_secondary.cpp:459`; `dq:289` | No inter stream sets 1. |
| `enable_chroma_dctonly f(1)` | p64; p270 | **WIRED** | `m1:134`, M1 default true (`still_headers.h:68`) | Fixed-on in M1. |
| `enable_inter_ddt f(1)` | p64; p270 | **COMPONENT** | `m2/ddt/ddt_syntax.cpp:5-20`; header patched to 1 by `ddt/gate/patch_video_headers.py`; DDT gate | CLI single-picture omits the bit. |
| `reduced_tx_part_set f(1)` | p64; p270 | **COMPONENT** | `dq:294`; tx-part tooling `tx_variants` | No stream sets 1. |
| `enable_cctx f(1)` | p64; p270 | **PARTIAL** | `tcq/cctx.cpp`, `tcq_flags.cpp:12`; header writers force 0 (`m2:197`) | `tcq/README.md` status "component only"; no CCTX stream or decoder gate. |
| `enable_tcq f(1)` | p64; p271 | **PARTIAL** | `tcq/tcq.cpp`, `tcq_flags.cpp:13`; writers force 0 | No TCQ stream; `tcq/build_and_test.sh:104` says a real dav2d TCQ gate is blocked. |
| `choose_tcq_per_frame f(1)` | p64; p271 | **PARTIAL** | `tcq_flags.cpp:14-17`; writers force 0 | Only in unit tests. |
| `enable_parity_hiding f(1)` | p64-65; p271 | **PARTIAL** | `tcq/parity/parity_hiding.cpp`; `tcq_flags.cpp:18-21`; parity entropy round-trip gate | No bitstream stream; `parity/build_and_test.sh:100` says the dav2d gate is blocked. |
| `enable_avg_cdf f(1)` / `avg_cdf_type f(1)` | p65; p271 | **COMPONENT** | `entropy/avg_cdf/avg_cdf_bank.cpp`; gate writes 1/1 (`gate_dav2d_avg_cdf.cpp:205-206`) and decodes with dav2d; `avg_cdf_test` | CLI single-picture infers 1/1, but that is a one-tile no-op; the two-tile averaging stream is component-only. |
| `separate_uv_delta_q f(1)` | p65; p271 | **COMPONENT** | `dq:296-310`; `dcdeltaq_test` 10.9M checks; `chroma_headers.cpp:296-300` rejects true | No full stream sets 1 (frame header would need `diff_uv_delta`). |
| `equal_ac_dc_q f(1)` | p65; p271 | **WIRED** (value 1) | `m0:248`, `m1:140`, `m2:202` write 1 | `equal_ac_dc_q = 0` only in `dq` component; it adds Y/UV DC base syntax that production writers cannot emit. |
| `base_y_dc_delta_q f(5)`, `y_dc_delta_q_enabled f(1)` | p65; p271 | **COMPONENT** | `dq:161-200`; `dcdeltaq_test`; `qindex`/dequant tests | Not reachable; `equal_ac_dc_q = 1` in every production header removes the branch. |
| `base_uv_dc_delta_q f(5)`, `uv_dc_delta_q_enabled f(1)` | p65; p271 | **COMPONENT** | `dq:180-210`; `dcdeltaq_test` | same. |
| `base_uv_ac_delta_q f(5)` | p65; p271 | **WIRED** (value 23 ⇒ 0) | `m0:251`, `m1:141`, `m2:203`; `dq:210` supports the full range | CLI cannot signal a non-zero base UV AC delta. |
| `uv_ac_delta_q_enabled f(1)` | p65; p271 | **WIRED** (value 0) | `m0:252`, `m1:142`, `m2:204` | CLI cannot enable the frame-level UV AC delta. |

### 3.9 §5.4.9 seg_info (p66; §6.4.9 p271)

| Item | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| `seg_info(numSegments)` loop, `feature_enabled f(1)`, `feature_value su/f` | p66; p271 | **COMPONENT** | `small:678-755`; 8/16 segments, 3 feature levels (`small:661-700`); gate `still_360s/segment` 0 diffs; duplicate implementation `mfh:205-240` + `mfh_test` | No CLI path (all production headers write `segmentation_enabled = 0`, e.g. `m0:309`). |

### 3.10 §5.4.10 sequence filter config (p66–67; §6.4.10 p271–272)

| Item | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| `disable_loopfilters_across_tiles f(1)` | p66; p271 | **COMPONENT** | `filters:321`; `restoration.cpp:218-219,344-346` model the tile clipping; `ccso_test.cpp:416-434` | Only 0 emitted in streams; no gate sets 1. |
| `enable_cdef f(1)` | p67; p272 | **WIRED** | `m1:146`; CLI `--still-cdef` (`main.cpp:358-361,647`); CDEF search + decoder-verified M1 gate | Fixed off unless flag. |
| `enable_gdf f(1)` | p66; p272 | **PARTIAL** | GDF filter + weights implemented (`filters/spec_tables/gdf.cpp`, `gdf_tables.h`, `gdf_test`); `filters:317-320` refuses `enable_gdf = 1` | No writer emits 1 and no GDF stream exists. |
| `gdf_unit_matches_sb_size f(1)` | p66; p272 | **ABSENT** | no writer | Only reachable with GDF, which is unwritable. |
| `enable_restoration f(1)` | p66; p272 | **COMPONENT** | `filters:313-347` full group writer; `wiener` + `lr_chroma` gates; `restoration.cpp` | CLI emits 0. |
| `lr_tools_disable[0][RESTORE_PC_WIENER] f(1)` | p66; p272 | **COMPONENT** | `filters:325-330`; `wiener_syntax.h:133-140` (encoder requires PC-Wiener disabled) | Only component streams. |
| `lr_tools_disable[0][RESTORE_WIENER_NONSEP] f(1)` | p67; p272 | **COMPONENT** | `filters:326`; `wiener_syntax.cpp` NS coefficients | same. |
| `lr_tools_disable[1][RESTORE_PC_WIENER] = 1` (inferred) | p67; p272 | **COMPONENT** | `filters:327-329` (no bit, as spec requires) | same. |
| `lr_tools_uv_present f(1)` + `lr_tools_disable[1][RESTORE_WIENER_NONSEP]` | p67; p272 | **COMPONENT** | `filters:328-334`; `lr_chroma` gate exercises U/V present | same. |
| `enable_ccso f(1)` | p67; p272 | **WIRED** | `m1:150-153`; CLI `--still-ccso` (`main.cpp:363-366,649`); CCSO params writer `m1:229-256` | Fixed off unless flag. |
| `ccso_unit_matches_sb_size f(1)` | p67; p272 | **WIRED** | `m1:151-153` writes 1 when CCSO on | Only 64x64 units. |
| `cdef_on_skip_txfm_always_on f(1)` / `cdef_on_skip_txfm_disabled f(1)` | p67; p272 | **COMPONENT** | `m2:212-213`; `filters:336-342`; avg_cdf gate uses `disabled = 1` and decodes | CLI single-picture infers ADAPTIVE, so no bits. |
| `df_par_bits_minus_2 f(2)` | p67; p272 | **COMPONENT** | `filters:343`; `small:582-620`; gate `still_360s/dfpar` 0 diffs | CLI writes 0 (`m0:261`). |

### 3.11 §5.4.11 user_defined_qm (p67–69; §6.4.11 p273)

| Item | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| `qm_copy_from_previous_plane f(1)` | p67; p273 | **COMPONENT** | `qm:107-115`; `qm_test` | QM OBU not reachable and not decoder-validated. |
| `qm_8x8_is_symmetric f(1)` | p68; p273 | **COMPONENT** | `qm:118-122,137-143`; `qm_test` | same. |
| `qm_4x8_is_transpose_of_8x4 f(1)` | p68; p273 | **COMPONENT** | `qm:123-130`; `qm_test` | same. |
| `quant_delta svlc()` scan-order coding, `coefRepeat`, 8×8/8×4/4×8 order | p68-69; p273 | **COMPONENT** | `qm:131-171` (diagonal scan, `& 255` delta, repeat, range check); `qm_test` | same. |

### 3.12 §5.4.12 timing_info (p69; §6.4.12 p273–274)

| Item | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| `num_units_in_display_tick f(32)`, `time_scale f(32)` | p69; p273 | **COMPONENT** | `brt:198-205`, `491-520`; range checks `brt:60-90`; `hrd_test` | Carried only inside CI; no CLI flag. |
| `equal_picture_interval f(1)`, `num_ticks_per_picture_minus_1 uvlc()` | p69; p273-274 | **COMPONENT** | `brt:198-205`; `hrd_test` pins `uvlc` and the ≤ 2³²−2 range | same. |

### 3.13 §5.4.13 seq_decoder_model_info (p69; §6.4.13 p274)

| Item | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| `decoder_buffer_delay uvlc()`, `encoder_buffer_delay uvlc()`, `low_delay_mode_flag f(1)` | p69; p274 | **COMPONENT** | `brt:198-202`, `230-250`; `hrd_test` | Pinned dav2d rejects the nested block (gate note), so only self-parse. |

## 4. §5.5–§5.17 OBU payloads

### 4.1 §5.5 TD, §5.6 MSDO (p69–70 / p274–275)

| Item | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| `temporal_delimiter_obu()` (empty payload; resets `SeenFrameHeader`, `QmProtected`) | §5.5 p69; §6.5 p274 | **WIRED** | `m0:351-355`; `m1:264-267` | Decoder-side reset not implemented (no decoder in tree); encoder obligation is just the empty OBU. |
| `multistream_decoder_operation_obu()`: `num_streams_minus_2`, profile/level/tier, even allocation, per-substream caps, DOH flag | §5.6 p69–70; §6.6 p274–275 | **OUT-OF-SCOPE** | No writer; enum `kMsdo` only (`svc_framing.h:50`), `GLOBAL_XLAYER_ID` inference for TD/MSDO (`svc_framing.cpp:196-205`) | No MSDO stream; multistream is packaging-only. |

### 4.2 §5.7 MFH (p70–71 / §6.7 p276)

| Item | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| `mfh_seq_header_id uvlc()` | p70; p276 | **COMPONENT** | `mfh:395`; range `< MAX_SEQ_NUM` (`mfh.cpp:120-160`); `mfh_test` | — |
| `mfh_id_minus_1 uvlc()` | p70; p276 | **COMPONENT** | `mfh:396`; `< MAX_MFH_NUM` validation | — |
| `mfh_frame_size_present_flag` + width/height bits | p70; p276 | **COMPONENT** | `mfh:397-411`; `mfh_test` | — |
| `mfh_deblocking_filter_update f(1)` + 4 apply flags | p70; p276 | **COMPONENT** | `mfh:412-417`; inheritance resolver `mfh.cpp:749-1000` | — |
| `mfh_seg_info_present_flag`, `mfh_ext_seg_flag`, `mfh_allow_seg_info_change`, `seg_info()` | p70–71; p276 | **COMPONENT** | `mfh:418-423`; `mfh_test`; `mfh_gate` | 8/16 segments supported. |
| Public OBU framing (type 3, `obu_tlayer_id`) | §5.2.1 p53–54 | **PARTIAL** | `mfh:427-443` | Missing `obu_extension_flag = 0` before trailing bits (see §1). |

### 4.3 §5.8 LCR (p71–75 / §6.8 p276–287)

| Item | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| `lcr_global_config_record_id f(3)` | p71; p278 | **COMPONENT** | `lcr/ops:449,399-403`; `svc_test` | Range 1..7 not enforced for global (0 allowed by writer); spec requires 1..7. |
| `lcr_xlayer_map f(31)` | p71; p278 | **COMPONENT** | `lcr/ops:450,408-447`; writer requires every set bit described | — |
| `lcr_aggregate_info_present_flag f(1)` + `lcr_aggregate_info()` (`config_idc`, `level_idx`, `max_tier`, `max_interop`) | p71–72; p278,281 | **COMPONENT** | `lcr/ops:451,459-467`; `svc_test` | — |
| `lcr_seq_profile_tier_level_info_present_flag f(1)` + `lcr_seq_profile_tier_level_info()` | p72; p278,281–282 | **ABSENT** | writer hardcodes 0 (`lcr/ops:452`) | No per-xlayer PTL record; `lcr_seq_profile_idc`/level/tier/mlayer-count never emitted. |
| `lcr_global_payload_present_flag f(1)`, `lcr_data_size leb128()`, `lcr_global_payload()` | p71–73; p280,282 | **COMPONENT** | `lcr/ops:453,469-480`; payload size is byte-exact by construction | `lcr_remaining_payload_bit` is zero-padded by `take_aligned`, not explicitly emitted. |
| `lcr_dependent_xlayers_flag f(1)` + `lcr_num_dependent_xlayer_map` | p71; p278 | **COMPONENT** | `lcr/ops:454,471-474` | Spec requires the flag to be 0; writer can set it if a layer declares deps. `svc_config` rejects cross-xlayer deps, so streams remain conformant, but the guard is indirect. |
| `lcr_global_atlas_id_present_flag f(1)` + id / reserved | p71; p279 | **PARTIAL** | `lcr/ops:455-457` writes flag 0 + 3 reserved zeros | Atlas association never signalled. |
| `lcr_global_purpose_id f(7)` | p71; p278-279 Table 6.6 | **PARTIAL** | `lcr/ops:453` writes 0 | Only "Unspecified". |
| `lcr_doh_constraint_flag f(1)` | p71; p279 | **PARTIAL** | `lcr/ops:456` writes 0 | Cannot satisfy the "monotonic = 0 ⇒ DOH constraint = 1" rule if a non-monotonic multistream were built. |
| `lcr_enforce_tile_alignment_flag f(1)` | p71; p279 | **PARTIAL** | `lcr/ops:457` writes 0 | No tile-alignment signalling. |
| `lcr_local_info()`: `lcr_global_id f(3)`, `lcr_local_id f(3)`, PTL flag, atlas flag, reserved bits | p72; p280–281 | **PARTIAL** | `lcr/ops:488-504`; `lcr_global_id` always 0 | Local LCR can never point at a global LCR; no local atlas id. |
| `lcr_xlayer_info()`: rep/purpose/color/embedded flags | p73–74; p282–287 | **PARTIAL** | `lcr/ops:74-92` writes rep=0, purpose=0, color=0, embedded=1 | No `lcr_rep_info` (max picture size/crop/format), no purpose id, no colour info. |
| `lcr_embedded_layer_info()`: `lcr_mlayer_map`, `lcr_tlayer_map`, atlas/priority/render/layer type/aux/view/dependent map/resolution | p74–75; p285–287 | **PARTIAL** | `lcr/ops:94-142` writes mlayer/tlayer maps, `layer_type = TEXTURE_LAYER`, `view_type = VIEW_CENTER`, dependent map, resolution | Atlas id, priority order, rendering method, `AUX_LAYER`, `VIEW_EXPLICIT`, view id, auxiliary type never emitted. |
| `lcr_xlayer_color_info()` (`layer_color_description_idc rg(2)`, primaries/transfer/matrix/full range) | p75; p287 | **ABSENT** | writer sets color-present flag 0 (`lcr/ops:82`) | — |
| LCR OBU framing, extensible tail | p71; p277 | **COMPONENT** | `lcr/ops:508-510` writes `obu_extension_flag = 0` + trailing bits | — |

### 4.4 §5.9 atlas segment (p75–79 / §6.9 p288–293)

| Item | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| `atlas_segment_info_obu()`: mode idc, enhanced/basic/single/multistream/multistream-alpha info, nominal size, region grid, segment mapping, label ids | §5.9 p75–79; §6.9 p288–293 | **OUT-OF-SCOPE** | No writer. Only enum value 17 and config-OBU ordering (`svc_framing.h:47`, `svc_framing.cpp:228-231`) | Entire atlas syntax absent. `lcr_layer_atlas_segment_id`, `ats_*` never emitted. |

### 4.5 §5.10 OPS, §5.11 operating point payload (p80–83 / §6.10 p293–300)

| Item | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| `ops_reset_flag f(1)` | p80; p294 | **COMPONENT** | `lcr/ops:532` (always 1); `svc_test` | Reset-only form; `ops_cnt = 0` reset variant not emitted (writer requires 1..7, `lcr/ops:526-530`). |
| `ops_id f(4)` | p80; p294 | **COMPONENT** | `lcr/ops:533` | — |
| `ops_cnt f(3)` | p80; p294 | **COMPONENT** | `lcr/ops:534`; 1..7 enforced | — |
| `ops_priority f(4)` | p80; p294 | **PARTIAL** | `lcr/ops:535` writes 0 | — |
| `ops_intent f(7)` | p80; p294-295 Table 6.12 | **PARTIAL** | `lcr/ops:536` writes 0 | No intent filtering (scalability/stereo/alpha/depth/gain/multiview). |
| `ops_intent_present_flag f(1)` | p80; p295 | **PARTIAL** | `lcr/ops:537` writes 0 | `ops_op_intent` never emitted. |
| `ops_ptl_present_flag f(1)` | p80; p295 | **PARTIAL** | `lcr/ops:538` writes 0 | No operating-point PTL. |
| `ops_color_info_present_flag f(1)` | p80; p295 | **PARTIAL** | `lcr/ops:539` writes 0 | No OPS colour info. |
| `ops_mlayer_info_idc f(2)` (global) / `ops_reserved_2bits` (local) | p80; p295 | **COMPONENT** | `lcr/ops:540-544` (global 1, local 0) | idc 2 (inherit) not emitted; idc 3 correctly rejected. |
| `ops_data_size leb128()`, `opsBytes` consistency | p80–81; p295-296 | **COMPONENT** | `lcr/ops:590-596` writes size after byte-align; self-parser checks it (`svc_test`) | — |
| `ops_aggregate_info()` | §5.11.1 p82; p297 | **ABSENT** | writer hardcodes `ops_ptl_present_flag = 0` | No config_idc/level/tier/interop. |
| `ops_seq_profile_tier_level_info()` | §5.11.2 p82; p297 | **ABSENT** | same | — |
| `ops_decoder_model_info_for_this_op_present_flag` + `ops_decoder_model_info()` | §5.11.3 p82; p297-298 | **PARTIAL** | flag written 0 (`lcr/ops:578`) | Buffer-delay/low-delay per OP absent. |
| `ops_initial_display_delay_present_flag` + `_minus_1` | p81; p296 | **PARTIAL** | flag written 0 (`lcr/ops:579`) | — |
| `ops_xlayer_map f(31)` (global) | p81; p296 | **COMPONENT** | `lcr/ops:580-584` | — |
| `ops_mlayer_explicit_info_flag`, `ops_embedded_ops_id`, `ops_embedded_op_index` (idc 2 inheritance) | p81; p296 | **ABSENT** | idc 2 never emitted | No embedded-OP inheritance. |
| `ops_mlayer_info()` (`ops_mlayer_map f(8)`, `ops_tlayer_map f(4)` per set bit) | §5.11.5 p82–83; p300 | **COMPONENT** | `lcr/ops:163-178,586-596`; `svc_test`; retention maps `tiles/svc_layers.cpp` | — |
| `byte_alignment()` inside `operating_point_payload` | p81; p296 | **COMPONENT** | `take_aligned` (`lcr/ops:40-43,589`) | — |
| OPS OBU framing, extensible tail | p80; p293 | **COMPONENT** | `lcr/ops:601-604` | — |

### 4.6 §5.12 BRT (p83 / §6.11 p301)

| Item | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| `br_ops_dependent_flag f(1)` | p83; p301 | **COMPONENT** | `brt:477-488`; both forms in `hrd_test` | No CLI path; pinned dav2d does not implement OBU type 15 (gate note). |
| `br_ops_id f(4)` | p83; p301 | **COMPONENT** | `brt:481`; correspondence validation against OPS (`brt:184-192`) | Never verified against a real OPS in a stream because no production OPS+BRT composition exists. |
| `br_ops_cnt f(3)` | p83; p301 | **COMPONENT** | `brt:482`; `ops_cnt` equality check | — |
| `br_decoder_model_present_op_flag f(1)` | p83; p301 | **COMPONENT** | `brt:484` | — |
| `br_time_op rg(4)` | p83; p301 | **COMPONENT** | `brt:486`; `put_rg` + 0..511 range (`hrd.h:189-192`) | Decoder does not consume it. |
| `br_time rg(4)` (dependent-flag = 0) | p83; p301 | **COMPONENT** | `brt:478-480` | same. |
| Decoder-model BRT planner (RC → removal time) | §6.11 p301 | **COMPONENT** | `rc/hrd/hrd_buffer.cpp`; `hrd_stream_gate` | Planner not integrated into production rate control. |

### 4.7 §5.13 QM OBU (p83–84 / §6.12 p301–302)

| Item | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| `qm_bit_map f(15)` | p83; p301 | **COMPONENT** | `qm:399-403`; `qm_test` | No CLI path; pinned dav2d rejects QM OBUs. |
| `qm_chroma_info_present_flag f(1)` | p83; p302 | **COMPONENT** | `qm:403-405` | — |
| `qm_bit_map == 0` reset branch | p83; p302 | **COMPONENT** | `qm:406-410` writes flag + trailing bits | Decoder bookkeeping (`QmProtected = 1`, `QmDataPresent = 0`) is not mirrored in encoder state. |
| `qm_is_default_flag f(1)` | p84; p302 | **COMPONENT** | `qm:414-417`; `qm_test` | — |
| `user_defined_qm()` for t = 0,1,2 × plane | p83–84; p302 | **COMPONENT** | `qm:414-421`; full scan/symmetry/transpose writer `qm:101-171` | Never decoder-validated. |
| `QmSeen` / `QmProtected` / `QmDataPresent` / `QmMLayerId` / `QmTLayerId` state | §6.12 p302 | **PARTIAL** | only `QmDataPresent` exists (`qm.h:54`, `qm.cpp:217`); `QmSeen`/`QmProtected`/layer ids absent | "Same level not specified twice", "only first OBU may have bit_map 0", and CLK/OLK reset rules are not enforced by the writer. |
| `reset_qm()` at CLK/OLK and `QmProtected` reset at TD | p84,302 | **ABSENT** | no encoder-side state machine | No violation is produced today only because no QM OBU is ever written into a real stream. |

### 4.8 §5.14 film grain OBU (p84–85 / §6.13 p302)

| Item | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| `fgm_update_flags f(8)` | p84; p302 | **ABSENT** | no writer; enum `kFilmGrain` only (`svc_framing.h:53`) | No OBU type 23 anywhere. |
| `fgm_chroma_idc uvlc()` and subX/subY derivation | p84–85; p302 | **ABSENT** | — | — |
| `film_grain_model()` (`num_y_points`, point arrays, AR coefficients, scaling, overlap, clip, `fg_mc_identity`, block size) | p85; p302 | **ABSENT** | Model struct only: `fg:16-52` (`FilmGrainModel`); no bitstream parser/writer | Synthesis reference exists (`fg:57-560`) and validates conformance (`FilmGrainModelValid`), but cannot be driven from a bitstream. |
| `save_grain_model()` / `FilmGrainPresent` slot bookkeeping | p85; p302 | **ABSENT** | — | — |
| `film_grain_params_present f(1)` | p58; p264 | **PARTIAL** | `chroma_headers.cpp:476` can write config; every other writer 0 (`m0:268`). No frame-level `film_grain_config()` exists (`chroma_still.cpp` has none) | Setting it true would desynchronise the frame header. CLI cannot enable grain models. |

### 4.9 §5.15 content interpretation (p85–86 / §6.14 p303–304)

| Item | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| `ci_scan_type_idc f(2)` | p85; p303 | **COMPONENT** | `ci:591-597`; `ci` tests | No CLI path; `brt` has a second writer (`brt:495`). |
| `ci_color_description_present_flag f(1)`, `ci_color_description_idc rg(2)`, explicit primaries/transfer/matrix, `ci_full_range_flag` | p85; p303 | **COMPONENT** | `ci:598-615`; Table 6.13 derivation `ci.cpp:120-180`; tests | — |
| `ci_chroma_sample_position_present_flag f(1)`, `ci_chroma_sample_position_top/bottom uvlc()` | p85–86; p304 | **COMPONENT** | `ci:616-628`; 4:2:0/4:2:2 tables (`ci.cpp:180-260`); tests | Pinned dav2d parses but does not enforce `<= 5`; probe OBU documents this (`ci.h:350-366`). |
| `ci_aspect_ratio_info_present_flag f(1)`, `ci_aspect_ratio_idc f(8)`, `ci_sar_width/height uvlc()` | p86; p304 | **COMPONENT** | `ci:630-646`; `Aspect_Ratio_*[17]` table `ci.h:65-68`; tests | — |
| `ci_timing_info_present_flag f(1)` + `timing_info()` | p86; p304 | **COMPONENT** | `ci:648-661`; `brt:491-520`; dav2d gate shows frame times change | — |
| `ci_reserved_2bit` | p85; p303 | **COMPONENT** | `ci:597` writes 0; reader accepts non-zero and records it | — |
| Extensible-OBU tail (`obu_extension_flag = 0` + trailing bits) | p85; p303 | **COMPONENT** | `ci:668-671`; `brt:957-961` | — |
| CI per-CVS constancy / layer inheritance / timing constancy | §6.14 p303 | **COMPONENT** | `CvsConstancyTracker`, `check_cvs_constancy` (`ci.h:372-424`); `ci` tests | No production stream with multiple layers/CVS. |

### 4.10 §5.16 padding (p86 / §6.15 p305)

| Item | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| `padding_obu()` (`obu_padding_byte f(8)`, `obu_padding_length` derived) | §5.16 p86; §6.15 p305 | **OUT-OF-SCOPE** | No writer | No padding OBU, no in-place OBU→padding conversion. |

### 4.11 §5.17 metadata (p87–93 / §6.16 p305–319)

| Item | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| `metadata_unit()` dispatch (`metadata_type`) | §5.17.1 p87–88; §6.16.1 p305 | **COMPONENT** | `md:264-293`; `mdt:424-425` typed decode; all 10 types unit-tested | Not reachable from `av2enc`; pinned dav2d does not implement AV2 metadata OBUs. |
| `metadata_unit_remaining_bit` (`remainingMuPayloadBits`) | §5.17.1 p88; p305 | **COMPONENT** | canonical-size padding: `mdt.cpp` serializers + `canonical_unit_size`; readers ignore extra bytes | Writer only emits canonical/byte-aligned payloads, never arbitrary remaining bits. |
| `metadata_short_obu()` header: `metadata_is_suffix`, necessity=0, application=0, `muh_layer_idc f(3)`, `muh_cancel_flag`, `muh_persistence_idc`, `metadata_type leb128()` | §5.17.2 p88; §6.16.2 p305–306 | **COMPONENT** | `md:264-293` options byte; `check_short_config` rejects layer ≥ 3; `hdr_metadata_test:720-790` | Short form cannot carry necessity/application by spec; correct. |
| `metadataPayloadSize = obuPayloadSize - 2 - Leb128Bytes` derivation | §5.17.2 p88; p306 | **COMPONENT** | writer emits options + leb128(type) + unit + 0x80; reader recomputes identically (`md:580-625`) | — |
| `metadata_cancel_flag = 1` (no body, type signalled) | §5.17.2 p88; p307 | **COMPONENT** | `md:238-244`; `mdt:372-375`; tests | `metadata_unit` committed with `muh_cancel_flag`; body absent correctly. |
| `metadata_group_obu()` header: `metadata_is_suffix`, `metadata_necessity_idc f(2)`, `metadata_application_id f(5)`, `metadata_unit_cnt_minus_1 leb128()` | §5.17.3 p88–89; p306–307 | **COMPONENT** | `md:318-345`; Table 6.16 enum (`hdr_metadata.h:84-91`); tests | Application ids 6..15 rejected (reserved); 16..31 externally-defined values also rejected. |
| per-unit `metadata_type leb128()`, `muh_header_size f(7)`, `muh_cancel_flag`, `muh_payload_size leb128()`, `muh_layer_idc f(3)`, `muh_persistence_idc f(3)`, `muh_priority f(8)`, `muh_reserved_zero_2bits` | §5.17.3 p89; p307–308 | **COMPONENT** | `md:346-392`; byte layout tested (`hdr_metadata_test:809-830`) | `muh_header_size` computed as leb128(payload)+2; no extension bytes. |
| `muh_layer_idc == LAYER_VALUES`: `muh_xlayer_map f(32)` + `muh_mlayer_map f(8)` | §5.17.3 p89; p307–308 | **ABSENT** (writer) | Writer explicitly rejects LAYER_VALUES (`md:116-121,346-356`); reader implements maps (`md:700-760`) and a test parses a hand-built LAYER_VALUES OBU (`hdr_metadata_test:809-820`) | Encoder cannot target a specific set of layers with metadata. |
| `muh_header_extension_byte f(8)` | §5.17.3 p89; p309 | **ABSENT** | writer always `headerRemainingBytes = 0` | No metadata-header extension bytes. |
| `metadata_itut_t35()` | §5.17.4 p90; p309 | **COMPONENT** | `mdt:723-750,915` (`serialize_itut_t35_unit` / `make_itut_t35_unit`), country-code/extension handling; tests | Payload is opaque bytes; no T.35 terminal-provider validation. |
| `metadata_hdr_cll()` (`max_cll f(16)`, `max_fall f(16)`) | §5.17.5 p90; p309–310 | **COMPONENT** | `md:238-262`; `hdr_metadata_test`; `run_hdr_gate.sh` | Pinned dav2d uses the AV1 metadata layout, so the gate proves decode-invariance, not AV2 parsing. |
| `metadata_hdr_mdcv()` (3× x/y, white point, min/max luminance) | §5.17.6 p90; p310–311 | **COMPONENT** | `md:244-262`; range validator `md:186-230`; tests | Same decoder caveat. |
| `metadata_timecode()` (`counting_type`, timestamp flags/values, `time_offset_length/value`) | §5.17.7 p90–91; p312–313 | **COMPONENT** | `mdt.cpp` timecode serializer/decoder; `metadata_types_test` | — |
| `metadata_banding_hints()` (component flags, band-unit grid, varying sizes, per-unit flags) | §5.17.8 p91–92; p313–315 | **COMPONENT** | `mdt.cpp` banding serializer/decoder; 32×32 max grid; tests | — |
| `metadata_icc_profile()` (`icc_profile_data_payload_bytes`) | §5.17.9 p92; p315 | **COMPONENT** | `mdt.cpp` opaque payload + ICC version derivation (`icc_profile_version`); tests | — |
| `metadata_scan_type()` (`mps_pic_struct_type`, `mps_source_scan_type_idc`, `mps_duplicate_flag`) | §5.17.10 p92; p315–316 | **COMPONENT** | `mdt.cpp` scan-type serializer + Table 6.18 validation; tests | — |
| `metadata_temporal_point_info()` (`frame_presentation_time leb128()`) | §5.17.11 p92; p316 | **COMPONENT** | `mdt.cpp` temporal-point serializer; group writer rejects it (`mdt.h:240`); tests | Correctly short-OBU-only. |
| `metadata_decoded_frame_hash()` (`hash_type`, per_plane, has_grain, is_monochrome, reserved, plane/frame hash) | §5.17.12 p93; p317–318 | **COMPONENT** | `mdt.cpp` serializer/decoder; MD5 byte-layout tests; `validate_decoded_frame_hash` | Encoder cannot compute the hash from reconstructed output (no output process); only caller-supplied values. |
| `metadata_user_data_unregistered()` (128-bit UUID + bytes) | §5.17.13 p93; p317 | **COMPONENT** | `mdt.cpp` serializer/decoder; tests | — |

## 5. §6.2–§6.16 semantics cross-reference

Rows here cover the semantics-level obligations not already captured by a field
above. Page references are the §6 body.

| Item | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| General OBU semantics: trailing bits present unless header-only; extension flag must be 0; tile groups consume trailing bits via `exit_symbol` | §6.2.1 p255 | **PARTIAL** | `m0:54-57,341-349`; `mfh` exception noted in §1 | MFH violates `obu_extension_flag == 0`; everything else conforms. |
| Table 6.1 layer-specific status and tlayer/mlayer constraints (`obu_tlayer_id <= max_tlayer_id`, key/SWITCH/RAS tlayer 0, GLOBAL_XLAYER types, MSDO/TD global) | §6.2.2 p256–257 | **PARTIAL** | Header writer enforces field widths only (`m0:106-123`); component validators enforce some rules (`svc_framing.cpp:236-275`, `random_access.cpp:1020-1035`, `leading.cpp:343-358`, `tlayers.cpp`) | No CLI-reachable validators; GLOBAL_XLAYER type whitelist (p257) is not checked centrally. |
| Trailing-bits / byte-alignment semantics | §6.2.3–§6.2.4 p258 | **WIRED** | same as §1 rows | — |
| Reserved OBU semantics | §6.3 p258 | **DECODER-ONLY** | no writer | Correct. |
| Sequence header semantics: profiles, chroma/bit-depth tables, crop ranges, initial delay, decoder-model, dependency maps, tile/partition/segment/intra/inter/SCC/TQ/filter config, film grain flag | §6.4.1–§6.4.13 p258–274 | **PARTIAL** | field rows in §3; value validators in `dq`, `small`, `brt`, `tlayers`, `svc_layers` | Several "shall" rules are not enforced by production writers: `seq_tier` high tier, `num_ref_frames_minus_1` upper bound 16 (m2 asserts 1..16), `seq_max_mlayer_cnt_minus_1 <= max_mlayer_id` (dead), `MaxPbAspectRatio` (component only), crop offsets (component only). |
| TD semantics (`SeenFrameHeader`, `QmProtected` reset) | §6.5 p274 | **WIRED** | empty OBU `m0:351-355` | No decoder state machine in tree, so the reset effect is not testable here. |
| MSDO semantics | §6.6 p274–275 | **OUT-OF-SCOPE** | no writer | — |
| MFH semantics (inheritance of size/deblock/segments, layer-dependency availability) | §6.7 p276 | **COMPONENT** | `mfh.cpp:749-1000` resolver; `mfh_test`; `mfh_gate` (8 frames, 0 diffs) | Not reachable; MFH extensible-flag bug above. |
| LCR semantics (global/local scope, aggregate, PTL, rep/format/crop, embedded layer metadata, colour) | §6.8 p276–287 | **PARTIAL** | see §4.3; `svc_layers.cpp` validation; `svc_test` | Missing PTL/rep/purpose/colour/atlas/rendering metadata; local LCR cannot reference a global one. |
| Atlas semantics (modes, region grid, segment IDs, multistream composition) | §6.9 p288–293 | **OUT-OF-SCOPE** | no code | — |
| OPS semantics (reset/update table, intent, PTL, colour, decoder model, layer maps, inheritance) | §6.10 p293–300 | **PARTIAL** | see §4.5; `svc_layers.cpp` validation | Only reset+explicit maps form; no intent/PTL/colour/decoder-model/inheritance. |
| BRT semantics (`br_ops_id`/`br_ops_cnt` correspondence, tick origin) | §6.11 p301 | **COMPONENT** | `brt:184-192` validation; `hrd_test`; decoder ignores type 15 | No stream where a decoder consumes BRT. |
| QM semantics (QmProtected/QmSeen/QmDataPresent, one bit_map-0 OBU per frame, no duplicate level, reset at CLK/OLK) | §6.12 p301–302 | **PARTIAL** | `qm.h:47-60` models only `QmDataPresent`; `qm_test`; `qm_gate.sh` (default matrices only) | No protection/seen state, no duplicate-level guard, no CLK/OLK reset. |
| Film-grain OBU semantics (`fgm_update_flags != 0`, `fgm_chroma_idc <= 3`, slot uniqueness, `FilmGrainPresent`) | §6.13 p302 | **ABSENT** | no OBU/syntax writer; synthesis model only (`fg`) | Decoder/output-side synthesis is implemented as a reference (`film_grain_test` 611 checks) but no stream can carry a model. |
| Content interpretation semantics (scan type, colour equivalence to OPS, chroma positions ≤ 5, aspect idc ≤ 16 or 255, per-CVS constancy/inheritance) | §6.14 p303–304 | **COMPONENT** | `ci.cpp` validators; `CvsConstancyTracker`; tests; `run_ci_gate.sh` | Pinned decoder does not enforce the chroma-position rule; layer inheritance only simulated. |
| Padding semantics | §6.15 p305 | **OUT-OF-SCOPE** | no writer | — |
| Metadata semantics (unit types, layer-specific table, suffix/prefix, necessity/application, header size, layer maps, persistence, priority, all 10 unit payloads, sentinel `remaining_bit`) | §6.16 p305–319 | **COMPONENT** | `md`, `mdt`, tests, `run_metadata_gate.sh`, `run_hdr_gate.sh` | Pinned dav2d does not implement AV2 §5.17.2/3; layer-set (`LAYER_VALUES`) targeting and header-extension bytes absent; no encoder-side persistence/priority policy beyond caller input. |
| Inter-layer prediction / MLayer dependency-only signalling | §5.4.1 p57; §6.4.2 p264; §6.8 p285 | **OUT-OF-SCOPE** | `svc_pipeline.h:20-28` sets `cross_xlayer_prediction = cross_mlayer_prediction = false`; `svc_layers` validates dependency maps only | No inter-layer sample/MV/coefficient prediction; dependency maps exist only as delivery metadata. |

---

## 6. Counts

Status counts over the rows in §1–§5:

| Status | Count |
|---|---:|
| WIRED | 29 |
| COMPONENT | 144 |
| PARTIAL | 51 |
| ABSENT | 26 |
| DECODER-ONLY | 6 |
| OUT-OF-SCOPE | 11 |
| **Total rows** | **267** |

Derived observations:

- **Reachable OBU types from `av2enc`:** 1, 2 and 4 (plus 7 only through the M2
  component). The CLI cannot emit any of 3, 5, 6, 8–25.
- **Fully implemented OBU payload groups:** metadata short/group (all 10
  unit types, minus layer-set maps), content interpretation, BRT, QM
  matrices, MFH, TD, and the M0/M1 sequence header subset.
- **Highest-risk gaps found by inspection:**
  1. `OBU_MULTI_FRAME_HEADER` omits the `obu_extension_flag f(1)` that
     `is_extensible_obu()` requires (§5.2.1 p53–54). The pinned dav2d gate
     passes, so this is invisible to current CI.
  2. Film-grain OBU type 23 is entirely absent; `film_grain_params_present`
     can be set in one writer without a matching frame-header
     `film_grain_config()`.
  3. No writer can emit `max_mlayer_id > 0`, `seq_max_mlayer_cnt_minus_1`,
     `mlayer_dependency_present_flag`, or `multi_tlayer_dependency_map_present_flag`
     (§5.4.1 p56–58). All embedded-layer signalling is dead code except
     packaging/validation.
  4. LCR/OPS writers emit only the reset/explicit-map skeleton; PTL, colour,
     purpose, atlas, decoder-model, intent and inheritance fields are missing
     (`§5.8`/`§5.10`), and composed SVC streams never set `seq_lcr_id`, so the
     LCR is not actually referenced.
  5. QM state (`QmSeen`, `QmProtected`, layer ids, reset) is not modelled, and
     because the pinned decoder rejects QM OBUs, no user-defined QM has ever
     been decoded.
  6. Several tools have a sequence-header bit that production writers hardcode
     to 0 and no gate sets to 1: SDP, DIP, MHCCP, IBP, GDF, TCQ/CCTX/parity
     hiding, IDTX, extended partitions. Their algorithms are tested in
     isolation but no conforming stream exercises them.

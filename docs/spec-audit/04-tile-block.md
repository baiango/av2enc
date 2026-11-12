# Spec audit 04 — tile-group, partition, block, mode, MV/prediction, coefficient and filter syntax

**Spec:** `spec/AV2_Spec_v1.0.0.pdf`, AV2 Bitstream & Decoding Process
Specification v1.0.0 (2026-05-28, commit `38f28e7`, SHA-256
`e9916f091e4e83446aad6b4601641c5b292e569c144c4163b26a4497573b533f`).
Page numbers below are *spec* pages (PDF index = page − 1).

**Scope:** §5.19–§5.20 (pp. 144–255), §6.18–§6.19 (pp. 349–375): tile-group
header/payload, partition tree, block decoding, mode information (intra,
inter, palette, IBC, CfL, MHCCP, MRL, DIP, IBP), transform/quantisation
parameters, MV/prediction structures, coding tools, coefficient syntax and
filter syntax.

**Method:** read the PDF syntax and semantics, then inspected the encoder
code and tests. There are no mock/implementation-status inputs in this table —
each status is backed by a writer/reader call site or a test/gate. The repo's
docs are stale in places (for example `README.md` calls M1 "variable
transforms", but `src/encoder/m1/still_syntax.cpp:1389-1400` rejects any TX size
that is not the block size, and `still_headers.cpp:258` writes
`tx_mode_select = 0`). Where docs and code disagree, the code is used.

## Classification key

| Status | Meaning |
|---|---|
| **WIRED** | An encoder path in this tree emits the element and the output has a decoder gate (M1 CLI `--still` / `--still-cdef` / `--still-ccso`, or the gated M2 P-frame pipeline). "WIRED" does not imply every legal value is exercised. |
| **COMPONENT** | A standalone writer (and usually a reader/test, sometimes a `dav2d` gate) exists under `src/encoder/...`, but it is not wired into a shipped encoder path. The directory is named in the evidence. |
| **PARTIAL** | Some sub-elements/values are emitted or some conformance checks exist, but the syntax family cannot be fully produced by the encoder. |
| **ABSENT** | No encoder-side writing. A decoder model or rate estimate may exist; it cannot put the element in a stream. |
| **DECODER-ONLY** | The element/process is parsed or modelled on the decode side only; no writer exists and none is needed for the current single-frame outputs. |

"Component-gated" means the component's `README`/gate script records a pinned
`dav2d` pass; "unit tests only" means no end-to-end decoder gate is recorded.

---

## 1. §5.19 Tile group OBU syntax (pp. 144–145)

| Item | §/page | Status | Evidence file:line or test | Gap |
|---|---|---|---|---|
| `is_first_tile_group` | §5.19 p144 | WIRED | `src/encoder/m1/still_headers.cpp:287`; `src/encoder/m1/tiles_stream/tiles_stream.cpp:428`; `src/encoder/m2/post/tiled_frame.cpp:678` | Only `= 1` is ever written. No second/later tile group. |
| `frame_header_present_flag` | §5.19 p144 | ABSENT | No `put_bit(false)` path; every group writer hardcodes `is_first_tile_group = 1` (`tiles_stream.cpp:428`) | Cannot emit a non-first group (`frame_header_present_flag = 0`). |
| `bru_inactive` / `use_bru` branch | §5.19 p144 | ABSENT | No `use_bru` field in any writer; `src/encoder/sdp/sdp.h:87` only models the inferred value | BRU (background replacement) is not supported. |
| `tile_start_and_end_present_flag` | §5.19 p144 | PARTIAL | `tiles_stream.cpp:431`; `src/encoder/m2/post/tiled_frame.cpp:686` | Always `0`. Writer rejects partial groups (`src/encoder/tiles/tile_group.cpp:49`). |
| `tg_start` | §5.19 p144–145 | ABSENT | `src/encoder/tiles/tile_group.h:18` models `all_tiles` only; no writer call | Cannot address a tile subset. |
| `tg_end` | §5.19 p145 | ABSENT | Same as `tg_start` | Cannot address a tile subset. |
| `bru_tile_active` | §5.19 p145 | ABSENT | No occurrence in `src/encoder` | No BRU tile activity map. |
| `byte_alignment()` | §5.19 p145 | WIRED | `src/encoder/m1/still_headers.cpp:289` (`align_to_byte`); `src/encoder/pipeline/tiled_still_gate.cpp:503` | Correct zero-bit alignment is emitted. |
| `tile_group_payload()` loop | §5.20.1 p145 | COMPONENT | `src/encoder/m1/tiles_stream/tiles_stream.cpp:398-447`; `src/encoder/tiles/tile_group.cpp:78-110` | Main M1 path is 1×1 tiles; multi-tile is component-gated. |
| `tile_size_minus_1` (`le(TileSizeBytes)`) | §5.20.1 p145 | COMPONENT | `tiles_stream.cpp:379`; `src/encoder/m2/post/tiled_frame.cpp:366`; `src/encoder/tiles/tiles_test.cpp` | Only emitted for `NumTiles > 1`. Last tile size implied, as required. |
| `init_symbol(tileSize)` / `exit_symbol()` / `CurrentQIndex = base_q_idx` | §5.20.1 p146 | WIRED | `src/encoder/m1/still_syntax.cpp:319`, `:404`; `src/encoder/m2/video_syntax.cpp:413`, `:489` | Single tile per frame; no per-tile QIndex delta. |
| `frame_end_update_cdf()` | §5.20.1 p146 | DECODER-ONLY | `src/encoder/entropy/avg_cdf/avg_cdf_bank.h:174`; M2 explicitly writes `enable_avg_cdf = 0` (`video_headers.cpp:200`) | No next-frame CDF handoff is emitted; only the average-CDF component models it. |
| `decode_frame_wrapup()` | §5.20.1 p146 | DECODER-ONLY | No encoder call; frame output is handled by IVF assembly | Not a bitstream element for a still/short GOP encoder. |
| `IsBridge` path | §5.20.1 p145–146 | ABSENT | No bridge writer in `src/encoder` | Bridge/embedded-layer frames unsupported. |

## 2. §5.20.2 Tile-level structures (pp. 146–148)

| Item | §/page | Status | Evidence file:line or test | Gap |
|---|---|---|---|---|
| `decode_tile()` superblock raster + per-SB context reset | §5.20.2.1 p146–147 | WIRED | `src/encoder/m1/still_syntax.cpp:338-394`; `src/encoder/m1/still_encoder.cpp:312` | Single-tile M1; multi-tile reset exists only in `m1/tiles_stream`. |
| Tile-relative `is_inside()` / neighbour availability | §5.20.2.1 p146; §5.20.9.1 p243 | PARTIAL | `src/encoder/m1/still_syntax.cpp:453-460`; `src/encoder/m1/still_encoder.cpp:157`, `:434-490` | M1 writer uses frame bounds (valid only for one tile); `tiles_stream` re-bases per tile in the component. |
| `clear_above_context()` | §5.20.2.1 p146 | WIRED | `still_syntax.cpp:338`, `:453`; `video_syntax.cpp:544` | — |
| `clear_left_context()` | §5.20.2.1 p147 | WIRED | `still_syntax.cpp:391`, `:460`; `video_syntax.cpp:551` | — |
| `reset_refmv_bank()` | §5.20.2.2 p148 | COMPONENT | `src/encoder/inter/temporal/ref_mv_bank.cpp`; `src/encoder/inter/temporal/temporal_test.cpp` | Prediction/state model only; M2 core does not enable the reference-MV bank. |
| `clear_block_decoded_flags()` | §5.20.2.3 p148 | WIRED | `src/encoder/m1/still_encoder.cpp:312`, `:434-490` | Encoder mirrors the border presets for top-right/bottom-left availability. |
| `bru_mode S()` | §5.20.2.1 p147 | ABSENT | No writer; only filter code consumes a caller-provided `bru_modes` array (`restoration.h:74`) | No BRU syntax. |
| `RegionType` / `TreeType` (SDP) | §5.20.2.1 p147; §6.19.2 p351 | COMPONENT | `src/encoder/sdp/sdp_syntax.cpp:379-383`; `src/encoder/sdp/sdp_test.cpp` | Symbol plans + tree model; no block stream and no `dav2d` gate. |
| `IBCCoded` / `update_ibc_buffers()` | §5.20.2.1 p147; §6.19.4 p355 | COMPONENT | `src/encoder/m1/ibc_stream/ibc_syntax.cpp:590-700`; `ibc_stream_test.cpp` | IBC is gated in its own 64×32 stream; not in M1 CLI. |
| `clear_cdef()` | §5.20.9.6 p244 | WIRED | `src/encoder/m1/still_syntax.cpp:1415`; `src/cuda/m1/cdef/still_cdef_index_syntax.h:19-24` | Per-64×64 grid reset. |
| `CurrentQIndex = base_q_idx` per tile | §5.20.1 p146 | WIRED | `still_syntax.cpp:317` | No delta-Q path in the wired encoder. |

## 3. §5.20.3 Partition structures (pp. 148–161)

| Item | §/page | Status | Evidence file:line or test | Gap |
|---|---|---|---|---|
| `PARTITION_NONE` / `HORZ` / `VERT` | §5.20.3.1 p150–155 | WIRED | `src/encoder/m1/still_syntax.cpp:538-658`; `src/encoder/partition/partition.cpp:702+`; `src/encoder/partition/partition_test.cpp` | Tree down to 8×8; 4×N rejected by the M1 writer. |
| `do_split S()` | §5.20.3.2 p156 | WIRED | `still_syntax.cpp:516`, `:574` | Only `NONE`/non-`NONE` distinction. |
| `do_square_split S()` | §5.20.3.2 p156 | COMPONENT | `src/encoder/sb128/sb128_stream.cpp:198`, `:319`; `src/encoder/sb256/sb256_stream.cpp:247`; `sb128_test.cpp`, `sb256_test.cpp` | Only legal at 128/256 SB roots; never coded by M1 (64×64 SB). Both components are `dav2d`-gated. |
| `rect_type S()` | §5.20.3.2 p156 | WIRED | `still_syntax.cpp:526`, `:610` | `HORZ`/`VERT` only. |
| `do_ext_partition S()` | §5.20.3.2 p157 | COMPONENT | `src/encoder/m1/partition_stream/partition_stream.h:279`; `partition_stream.cpp:439`; `gate_minimal.cpp` | Adapter tests pass and block-level `dav2d` gate passes for `HORZ_3`; not wired into `still_syntax`. |
| `do_uneven_4way_partition S()` | §5.20.3.2 p157 | COMPONENT | Same as above; `partition_ext/ext_syntax.cpp` | Not wired. |
| `uneven_4way_partition_type L(1)` | §5.20.3.2 p157 | COMPONENT | `partition_stream.cpp:129`; `partition_stream_test.cpp` | Not wired. |
| `region_type S()` | §5.20.3.1 p150 | COMPONENT | `src/encoder/sdp/sdp_syntax.cpp:379`; `sdp_test.cpp` | SDP model + symbol planner, no stream, not gated. |
| `partition_implied()` / boundary recursion | §5.20.3.2 p159–160 | WIRED | `src/encoder/partition/partition.cpp:852`, `:897`; `partition_test.cpp` edge cases | Correct implied partition at partial SB edges. |
| `init_allowed_partitions()` / `is_partition_allowed()` / chroma offset rules | §5.20.3.2 p157–159 | WIRED | `partition.cpp:768-820`, `:921-945`; `partition_test.cpp` | Covers square/rect only. |
| Extended `subSize` / `Rect_Part_Table` / `H_Partition_Midsize` | §5.20.3.2 p155–157; §6.19.3.1 p352–353 | COMPONENT | `src/encoder/partition_ext/ext_syntax.cpp`, `:ext_tables_generated.h`; `partition_ext_test.cpp` | Tables/model are complete; only the adapter gate exercises streams. |
| 128×128 and 256×256 superblock roots | §5.20.3.1 p148; §6.19.3 p351 | COMPONENT | `src/encoder/sb128/sb128_stream.cpp`; `src/encoder/sb256/sb256_stream.cpp`; gates in both READMEs | Gated DC streams; not wired into the main partition solver. |
| `ChromaFollowsLuma` / `ChromaPartitionKnown` state | §6.19.3.1 p353–354 | COMPONENT | `src/encoder/sdp/sdp.cpp`; `sdp_validate.cpp`; `sdp_test.cpp` | SDP only. |

## 4. §5.20.4 Block decoding structures (pp. 161–169)

| Item | §/page | Status | Evidence file:line or test | Gap |
|---|---|---|---|---|
| `mode_info()` dispatch (BRU/intra/inter) | §5.20.4.1 p161; §5.20.5.1 p169 | PARTIAL | `src/encoder/m1/still_syntax.cpp:1372-1445` (intra); `src/encoder/m2/video_syntax.cpp:1881-1935` (intra/inter) | No BRU or SDP dispatch; no palette/IBC calls in the core. |
| `palette_tokens()` call after mode info | §5.20.4.1 p162 | COMPONENT | `src/encoder/m1/palette_stream/palette_stream.cpp:214`; `palette_stream_test.cpp`; four-mode `dav2d` gate | Not called by `still_syntax.cpp`. |
| `read_block_tx_size()` | §5.20.4.1 p162; §5.20.6.2 p183 | ABSENT | `still_syntax.cpp:1389-1400` validates "TX size == block size"; `still_headers.cpp:258` writes `tx_mode_select = 0` | No TX-partition or TX-size syntax; always `TX_MODE_LARGEST`. |
| `reset_block_context()` when `skip_flag` | §5.20.4.1 p164 | WIRED | `still_syntax.cpp:453-460`; `video_syntax.cpp:544-560` | — |
| `compute_prediction()` / `residual()` ordering | §5.20.4.1 p163; §5.20.7.22–.23 p214–216 | WIRED | `src/encoder/m1/still_encoder.cpp:1193-1320`; `src/encoder/m2/video_frame.cpp:407+`; reconstruction gates | — |

## 5. §5.20.5 Mode information (pp. 169–182)

| Item | §/page | Status | Evidence file:line or test | Gap |
|---|---|---|---|---|
| `skip_flag` in intra frame (`skip_flag = 0` inference) | §5.20.5.3 p171; §5.20.5.10 p181 | ABSENT | No symbol in `still_syntax.cpp` intra leaf; `skip_flag` stays 0 | Not a loss while `SegIdPreSkip = 0`; needed for segmentation one-pass. |
| `skip_flag S()` in inter frame | §5.20.5.10 p181; §5.20.7.1 p187 | WIRED | `video_syntax.cpp:1514`, `:1909` | Only `skip_mode = 0` context. |
| `skip_mode S()` | §5.20.5.9 p180 | ABSENT | `video_headers.cpp` writes `skip_mode_present = 0`; no writer | No B/compound skip mode. |
| `use_intrabc S()` | §5.20.5.3 p171; §5.20.7.3 p188 | COMPONENT | `src/encoder/m1/ibc_stream/ibc_syntax.cpp:79-86`, `:590`; `ibc_stream_test.cpp` | Four gated IBC streams, not in M1 CLI. |
| `intrabc_mode S()` | §5.20.5.4 p173 | COMPONENT | `ibc_syntax.cpp:595` | Gate covers NEWMV/NEARMV. |
| `intrabc_drl_mode L(1)` loop | §5.20.5.4 p173 | COMPONENT | `ibc_syntax.cpp:590-620`; `ibc_stream_test.cpp` | Gate covers default/neighbour candidate. |
| `intrabc_precision S()` | §5.20.5.4 p173 | COMPONENT | `ibc_syntax.cpp:622` | Quarter-pel path tested; gate uses integer. |
| `morph_pred S()` | §5.20.5.4 p173 | ABSENT | No writer; `src/encoder/ibc/ibc.h:22` lists it as out of scope | No BAWP/morph IBC. |
| `intra_segment_id()` / `seg_id_ext_flag` / `segment_id` | §5.20.5.7–.8 p178–179 | COMPONENT | `src/encoder/m1/deltaq/deltaq.cpp:214-290`, `:960`; `deltaq_test.cpp` | Unit tests; header-level `dav2d` gate only (`m2/deltaq/build_gate.sh`). |
| `inter_segment_id()` / `seg_id_predicted` | §5.20.7.2 p187 | COMPONENT | `deltaq.cpp:987-1044` | Not wired into M2 tile writer; no block-level gate. |
| `delta_q_abs` / `delta_q_rem_bits` / `delta_q_abs_bits` / `delta_q_sign_bit` | §5.20.5.11 p181 | COMPONENT | `deltaq.cpp:696`, `:766`; `deltaq_test.cpp` | Header gate proves the frame fields; block symbols are unit-tested only. |
| `use_dpcm_y S()` / `dpcm_mode_y S()` | §5.20.5.5 p174 | COMPONENT | `src/encoder/lossless/lossless.cpp:2144`; `lossless.h`; `lossless_gate.cpp` (22 decoder checks) | Lossless component only; not wired. |
| `fsc_mode S()` | §5.20.5.5 p174–175 | COMPONENT | `lossless.cpp:1167`, `:2257`; `lossless_gate.cpp` | FSC/IDTX coefficient path is gated in the lossless component. |
| `y_mode_set S()` / `y_mode_index S()` / `y_mode_offset S()` / `y_second_mode L(4)` | §5.20.5.5 p174 | WIRED | `still_syntax.cpp:747-766` (`write_literal(4, second)`) | All 61 luma modes/angles are representable; search is DC-only unless `--still-full-intra`. |
| `mrl_index S()` / `mrl_sec_index S()` | §5.20.5.5 p175 | COMPONENT | `src/encoder/m1/mrl_stream/mrl_syntax.cpp:152`; `mrl_stream_test.cpp`; `gate_dav2d_mrl.py` (8 cases) | M1 sequence writes `enable_mrls = 0` (`still_headers.cpp:116`). |
| `use_dpcm_uv S()` / `dpcm_mode_uv S()` | §5.20.5.6 p177 | COMPONENT | `lossless.cpp`; `lossless_gate.cpp` | Lossless only. |
| `is_cfl S()` | §5.20.5.6 p177 | WIRED | `still_syntax.cpp:790-800`; CfL gate cases in `src/encoder/m1/rdo_decide/chroma_rdo.cpp` and `src/encoder/cfl/cfl_test.cpp` | Only `CFL_DERIVED_ALPHA` in the core. |
| `uv_mode S()` / `uv_mode_idx L(3)` | §5.20.5.6 p177 | WIRED | `still_syntax.cpp:800-807` | Full UV mode values; directional search is opt-in (`--still-uv-full`). |
| `cfl_index S()` | §5.20.7.32 p235 | PARTIAL | `still_syntax.cpp:795` always writes `CFL_DERIVED_ALPHA`; explicit writer in `src/encoder/cfl/alpha/syntax.cpp:150` | `CFL_EXPLICIT`/`CFL_MULTI` not reachable from M1. |
| `cfl_mhccp S()` / `cfl_mh_dir S()` | §5.20.7.32 p235 | COMPONENT | `src/encoder/cfl/mhccp/mhccp_syntax.cpp:131`; `mhccp_test.cpp` | No `dav2d` gate (core edits listed in its README). |
| `cfl_alpha_signs S()` / `cfl_alpha_u S()` / `cfl_alpha_v S()` | §5.20.7.32 p235 | COMPONENT | `src/encoder/cfl/alpha/syntax.cpp:150`; `alpha_test.cpp` | Pinned payload round-trips; no frame gate. |
| `has_palette_y` / `palette_size_y_minus_2` / cache / literal / extra-bits / delta | §5.20.8.1 p235–236 | COMPONENT | `src/encoder/palette/palette.cpp:636-712`; `src/encoder/m1/palette_stream/palette_stream.cpp:214`; four-mode `dav2d` gate | Not in M1 CLI; `allow_screen_content_tools = 0` (`still_headers.cpp:170`). |
| `palette_direction` / `identity_row_y` / `color_index_map_y NS` / `palette_color_idx_y` | §5.20.8.4 p240–241 | COMPONENT | `palette.cpp:572-635`; `palette_stream_test.cpp`; gate | Same as above. |
| `transform_type()` intra | §5.20.8.2 p237–239 | WIRED | `src/encoder/m1/still_syntax.cpp:1064-1108`; `--still-tx-type` gate | `reduced_tx_set = 0`, `enable_idtx_intra = 0`; identity types excluded. |

## 6. §5.20.6 Transform and quantisation structures (pp. 182–186)

| Item | §/page | Status | Evidence file:line or test | Gap |
|---|---|---|---|---|
| `lossless_tx_size S()` | §5.20.6.1 p182 | COMPONENT | `src/encoder/lossless/lossless.cpp:2144-2148`; `lossless_gate.cpp` | Lossless component only. |
| `tx_do_partition S()` | §5.20.6.3 p183 | ABSENT | Only comments reference it (`src/encoder/m1/sub8/sub8.cpp:413`, `src/encoder/m2/residual/inter_residual.h:22`) | No transform-partition writer. |
| `tx_partition_type S()` | §5.20.6.3 p183 | ABSENT | No writer | No 2-way/split TX partition. |
| `tx_2or3_partition_type S()` | §5.20.6.3 p183 | ABSENT | No writer | No HORZ4/VERT4 TX partition. |
| `read_tx_size()` / `TxSize` signalling | §5.20.6.1 p182 | PARTIAL | `still_syntax.cpp:1389-1400` accepts only the block-size transform; `still_headers.cpp:258` | `TX_MODE_LARGEST` only; no per-block TX-size bits. |
| `reduced_tx_set f(2)` | §5.20.6.1 p182; header | WIRED | `still_headers.cpp:259`; `still_syntax.cpp:1072` | Always 0 in M1; transform_stream supports 0..3 but is not wired. |
| `delta_q_present` / `delta_q_res` / `CurrentQIndex` update | §5.18.7.8; §5.20.5.11 p181 | COMPONENT | `src/encoder/m1/deltaq/deltaq.cpp:433-460`; `m2/deltaq/build_gate.sh` | Frame header gate only; block delta-Q not wired. |
| `using_qmatrix` / `setup_qm_params()` / QM dequant | §5.18.6.2; §6.17.6 | COMPONENT | `src/encoder/qm/qm.cpp`; `src/encoder/qm/qm_gate.cpp`; `qm_test.cpp` | Gate splices a QM OBU into an M1 stream; not a runtime option. |
| `base_q_idx` / `qlookup` / dequant | §5.18.6.1; §7.14 | WIRED | `still_encoder.cpp:196`; `still_headers.cpp:175`; QP 1..255 gate | Correct quantiser, no per-block delta. |

## 7. §5.20.7 Motion vector and prediction structures (pp. 186–235)

| Item | §/page | Status | Evidence file:line or test | Gap |
|---|---|---|---|---|
| `is_inter S()` | §5.20.7.3 p188 | WIRED | `video_syntax.cpp:1510`, `:1904`; `video_gate.cpp` | Single-reference P only. |
| `comp_mode S()` | §5.20.7.10 p199 | COMPONENT | `src/encoder/m2/multiref/gate/compound_tile.cpp:436`; compound `dav2d` gate | Compound only in the minimal gate writer; M2 core writes `reference_select = 0`. |
| `comp_ref S()` loop | §5.20.7.11 p200–201 | COMPONENT | `compound_tile.cpp:332-399`; compound gate | `NumTotalRefs > 1` only in the gate writer. |
| `single_ref S()` loop | §5.20.7.12 p201 | ABSENT | `video_headers.h:31` (`num_total_refs = 1`); no writer | With one total ref the loop has zero symbols. Multi-ref selection missing. |
| `tip_mode S()` | §5.20.7.10 p199 | COMPONENT | `src/encoder/inter/temporal/temporal_syntax.cpp:396`; `temporal_test.cpp` | No decoder gate; no M2 integration. |
| `tip_pred_mode S()` | §5.20.7.6 p191–192 | COMPONENT | `temporal_syntax.cpp:426` | No decoder gate. |
| `compound_mode_same_refs S()` | §5.20.7.6 p191 | ABSENT | `compound_tile.cpp:462` explicitly rejects same-reference compound | CDF exists, writer refuses. |
| `is_joint S()` | §5.20.7.6 p191 | COMPONENT | `compound_tile.cpp:447`; compound gate | Gate picks non-joint. |
| `compound_mode_non_joint S()` | §5.20.7.6 p191 | COMPONENT | `compound_tile.cpp:452`; compound gate | Gate covers AVERAGE/WEDGE/DIFFWTD types. |
| `use_optflow S()` | §5.20.7.6 p191 | ABSENT | No per-block writer; `temporal_syntax.h:64` only writes `opfl_refine_type` header bits | OPFL prediction exists in `inter/temporal/opfl.cpp`, but its block symbol cannot be emitted. |
| `use_amvd S()` | §5.20.7.6 p191 | ABSENT | No writer; only decision helper `advanced_decision.cpp:320` | Adaptive MVD not signallable. |
| `is_warp S()` / `warp_mv S()` | §5.20.7.6 p192 | COMPONENT | `src/encoder/m2/advanced/advanced_syntax.cpp:409`; `advanced_test.cpp` | Writer/reader + decision; no frame gate. |
| `single_mode S()` | §5.20.7.6 p192 | WIRED | `video_syntax.cpp:1518`, `:1690` | NEARMV/GLOBALMV/NEWMV only. |
| `use_bawp S()` / `explicit_bawp S()` / `explicit_bawp_scale S()` / `use_bawp_chroma S()` | §5.20.7.6 p192–193 | COMPONENT | `advanced_syntax.cpp:282-350`; `advanced_test.cpp` | No decoder gate; M2 sequence has `enable_bawp = 0` (`video_headers.cpp:171`). |
| `warp_idx S()` | §5.20.7.6 p193 | COMPONENT | `advanced_syntax.cpp:455` (`write_warp_idx`) | No decoder gate. |
| `warpmv_with_mvd S()` | §5.20.7.6 p194 | COMPONENT | `advanced_syntax.cpp` staged writer; `advanced_test.cpp` | No decoder gate. |
| `jmvd_scale_mode S()` | §5.20.7.6 p194 | ABSENT | No writer; `JOINT_NEWMV` never produced | Joint MVD unsupported. |
| `drl_mode S()` | §5.20.7.8 p198 | WIRED | `video_syntax.cpp:1522`, `:1705`; `m2/merge` stack | DRL reorder disabled (`video_syntax.cpp:1660`). |
| `use_most_probable_precision S()` / `pb_mv_precision S()` | §5.20.7.6 p194 | ABSENT | `src/encoder/inter/me2/me2_precision.h` maps the values, but README says "No bitstream syntax is written here" | Frame precision fixed to 1/8 pel. |
| `warp_delta_precision S()` / `warp_delta_param_low S()` / `_high S()` / `_sign S()` | §5.20.7.7 p197–198 | COMPONENT | `advanced_syntax.cpp:477-520`; `advanced_test.cpp` | No decoder gate. |
| `use_dip S()` / `dip_transpose L(1)` / `dip_mode S()` | §5.20.7.9 p199 | COMPONENT | `src/encoder/dip/dip_syntax.cpp:32`; `dip_test.cpp` | No decoder gate. |
| `use_extend_warp S()` / `use_local_warp S()` | §5.20.7.14 p205–206 | COMPONENT | `advanced_syntax.cpp:409-450`; `advanced_test.cpp` | No decoder gate. |
| `inter_intra S()` / `warp_inter_intra S()` / `interintra_mode S()` / `wedge_interintra S()` | §5.20.7.15 p207–208 | COMPONENT | `advanced_syntax.cpp:352-410`; `advanced_test.cpp`; prediction in `inter/advanced/compound.cpp` | No decoder gate. |
| `comp_group_idx S()` / `compound_type S()` | §5.20.7.16 p208–209 | COMPONENT | `compound_tile.cpp:475-490`; compound gate | Only in the gate writer. |
| `wedge_quad S()` / `wedge_angle S()` / `wedge_dist1 S()` / `wedge_dist2 S()` / `wedge_sign L(1)` / `mask_type L(1)` | §5.20.7.18 p210; §5.20.7.16 p209 | COMPONENT | `compound_tile.cpp:496-526`; `advanced_syntax.cpp:269`; wedge tables | Gate covers wedge/diffwtd on 64×64. |
| `use_refinemv S()` | §5.20.7.17 p209 | COMPONENT | `temporal_syntax.cpp:476`; `temporal_test.cpp`; prediction in `refinemv.cpp` | No decoder gate. |
| `interp_filter S()` | §5.20.7.6 p196 | ABSENT | `video_headers.cpp:273-274` fixes `is_filter_switchable = 0` and `EIGHTTAP`; no block writer | No switchable per-block filter. |
| `mv_joint S()` | §5.20.7.20 p212 | ABSENT | Adaptive MVD disabled; no writer | Shell coding only. |
| `shell_set S()` / `shell_class S()` / `joint_shell_last_two_classes S()` | §5.20.7.20 p212 | WIRED | `video_syntax.cpp:1548-1566` | Shell classes 0..17+; no adaptive-MVD alternative. |
| `shell_offset_low_class S()` / `shell_offset_class2 S()` / `shell_offset_class2_high L(1)` / `shell_offset_other_class S()` | §5.20.7.20 p212–213 | WIRED | `video_syntax.cpp:1568-1589` | — |
| `col_mv_greater S()` / `col_remainder NS()` / `col_mv_index S()` | §5.20.7.20 p213 | WIRED | `video_syntax.cpp:1609-1622` (`write_ns_bits` at `:1535`) | — |
| `mv_sign L(1)` | §5.20.7.13 p203 | WIRED | `video_syntax.cpp:1628`; `video_headers.cpp:179` disables MVD-sign derivation | Both components raw; correct for this config. |
| `amvd_index S()` | §5.20.7.21 p214 | ABSENT | No writer | Adaptive MVD unsupported. |
| `assign_mv()` joint projection (`projMv`) | §5.20.7.13 p203–204 | ABSENT | `JOINT_NEWMV` never emitted by `video_syntax.cpp` | No joint-MVD block. |

## 8. §5.20.7.23–.32 Coefficient syntax (pp. 216–235)

| Item | §/page | Status | Evidence file:line or test | Gap |
|---|---|---|---|---|
| `all_zero S()` (txb skip) | §5.20.7.27 p224 | WIRED | `still_syntax.cpp:1043`; `video_syntax.cpp:1177` | Intra/inter CDF axes wired; qctx slices gated. |
| `eob_pt_16` / `_32` / `_64` / `_128` / `_256` / `_512` / `_1024` | §5.20.7.27 p225 | WIRED | `still_syntax.cpp:813-868`; `video_syntax.cpp:904-960` | All 7 multisizes coded. |
| `eob_pt_256_extra L(1)` / `_512_extra L(2)` / `_1024_extra L(2)` | §5.20.7.27 p225 | WIRED | `still_syntax.cpp:859-870` | Escape bits emitted. |
| `eob_extra S()` / `eob_extra_bit L(1)` | §5.20.7.27 p225–226 | WIRED | `still_syntax.cpp:877-884` | — |
| `cctx_type S()` | §5.20.7.27 p226; §6.19.7.23 p369 | PARTIAL | `src/encoder/tcq/cctx.h:74` cost only; `cctx.cpp:35`; no `encode_symbol` | Decision/rate model, no bit writer, no gate. |
| `coeff_base_eob S()` | §5.20.7.27 p227–228 | WIRED | `still_syntax.cpp:1270-1290`; `video_syntax.cpp:1274-1289` | LF/HF and UV variants. |
| `coeff_base S()` | §5.20.7.27 p228 | WIRED | `still_syntax.cpp:1292-1330`; `video_syntax.cpp:1316-1340` | — |
| `coeff_br S()` | §5.20.7.27 p228 | WIRED | `still_syntax.cpp:1345-1370`; `video_syntax.cpp:1362-1370` | — |
| `coeff_base_bob S()` | §5.20.7.27 p226 | COMPONENT | `src/encoder/lossless/lossless.cpp:1167-1200`; `lossless_gate.cpp` | FSC lossless only. |
| `coeff_base_idtx S()` | §5.20.7.27 p226 | COMPONENT | `lossless.cpp`; `lossless_gate.cpp` | FSC lossless only. |
| `coeff_br_idtx S()` | §5.20.7.27 p226 | COMPONENT | `lossless.cpp`; `lossless_gate.cpp` | FSC lossless only. |
| `idtx_sign S()` | §5.20.7.27 p226 | COMPONENT | `lossless.cpp`; `lossless_gate.cpp` | FSC lossless only. |
| `dc_sign S()` | §5.20.7.27 p228 | WIRED | `still_syntax.cpp:1389-1395`; `video_syntax.cpp:1395` | Luma DC only. |
| `dc_sign_horz_vert S()` | §5.20.7.27 p228 | ABSENT | No writer; M1/M2 only use `TX_CLASS_2D` transforms | Needs IDTX/1D transform classes. |
| `sign_bit L(1)` | §5.20.7.27 p229 | WIRED | `still_syntax.cpp:1398`; `video_syntax.cpp:1628` | Raw AC/other-component sign. |
| `q_length_bit L(1)` / `golomb_length_bit L(1)` / `coeff_rem L(length)` | §5.20.7.28 p231 | WIRED | `still_syntax.cpp:889-930` (`write_quant_escape`) | `allowTcq = 0`; the TCQ-doubling branch is not emitted. |
| TCQ state / parity hiding (`isHidden`, `sumAbs1`, coefficient doubling) | §5.20.7.27 p226–230 | COMPONENT | `src/encoder/tcq/parity/parity_entropy.h:70` (`write_block`); `parity_test.cpp` | Coefficient-payload round-trip only; no tile/OBU or `dav2d` gate. |
| `get_scan()` / `get_tx_row_col()` / `get_lf_limits()` | §5.20.7.30 p233; p230 | WIRED | `still_syntax.cpp:1595+` (`build_scan`); `still_syntax.cpp:1120-1145` | 2D/HORIZ/VERT scan classes; only 2D used. |
| `compute_tx_type()` / `is_cctx_allowed()` | §5.20.7.29 p231; p230 | PARTIAL | `still_syntax.cpp:1043-1108`; `tcq/cctx.h:69` predicate | Intra + DCT-only inter; no CCTX, no IDTX, no DDT. |

## 9. §5.20.8 Coding tools (pp. 235–243)

| Item | §/page | Status | Evidence file:line or test | Gap |
|---|---|---|---|---|
| `palette_size_y_minus_2 S()` | §5.20.8.1 p236 | COMPONENT | `src/encoder/palette/palette.cpp:649`; `palette_stream` gate | Not in M1 CLI. |
| `use_palette_color_cache_y L(1)` | §5.20.8.1 p236 | COMPONENT | `palette.cpp:668-680` | Not in M1 CLI. |
| `palette_colors_y L(BitDepth)` | §5.20.8.1 p236 | COMPONENT | `palette.cpp:682` | 8-bit literal only in the core; gate 8-bit. |
| `palette_num_extra_bits_y L(2)` | §5.20.8.1 p236 | COMPONENT | `palette.cpp:694`; `palette_extra_bits_for` `:140` | Not in M1 CLI. |
| `palette_delta_y L(paletteBits)` | §5.20.8.1 p236 | COMPONENT | `palette.cpp:705-718` | Not in M1 CLI. |
| `palette_direction L(1)` | §5.20.8.4 p241 | COMPONENT | `palette.cpp:700-720`; `palette_stream_test.cpp` | 64-wide/tall blocks pin rows per spec. |
| `identity_row_y S()` | §5.20.8.4 p241 | COMPONENT | `palette.cpp:588-600` | Identity 0/1/2 exercised in gate. |
| `color_index_map_y NS(PaletteSizeY)` | §5.20.8.4 p241 | COMPONENT | `palette.cpp:600-615` | First sample only, per spec. |
| `palette_color_idx_y S()` | §5.20.8.4 p241 | COMPONENT | `palette.cpp:611-628`; `get_palette_color_context` | Not in M1 CLI. |
| `is_long_side_dct S()` | §5.20.8.2 p238 | WIRED | `still_syntax.cpp:1089-1102`; `video_syntax.cpp:1068` | Intra WIDE_32/HIGH_32; inter path only DCT. |
| `inter_tx_type S()` | §5.20.8.2 p238 | WIRED | `video_syntax.cpp:1056-1083`; `video_gate.cpp` | Only `DCT_DCT` is selected. |
| `inter_tx_type_offset S()` | §5.20.8.2 p238 | WIRED | `video_syntax.cpp:1071-1080` | Only the DCT offset is emitted. |
| `lossless_inter_tx_type S()` | §5.20.8.2 p237–238 | ABSENT | No writer; lossless component is intra-only | Lossless inter unsupported. |
| `intra_tx_type S()` | §5.20.8.2 p238 | WIRED | `still_syntax.cpp:1086`; `--still-tx-type` gate | Identity types excluded (`enable_idtx_intra = 0`). |
| `sec_tx_type S()` / `most_probable_stx_set S()` | §5.20.8.2 p239 | PARTIAL | `src/encoder/tx_secondary/tx_secondary.cpp:215-264`; no entropy writer | Kernel/legality only; `enable_intra_ist = 0`. |

## 10. §5.20.10 Filtering structures (pp. 244–254)

| Item | §/page | Status | Evidence file:line or test | Gap |
|---|---|---|---|---|
| `cdef_index0 S()` | §5.20.10.1 p244 | WIRED | `still_syntax.cpp:1415`; `src/cuda/m1/cdef/still_cdef_index_syntax.h:19-24`; `--still-cdef` gate | Opt-in; `CdefStrengths == 1` writes no index. |
| `cdef_index_minus_1 S()` | §5.20.10.1 p244 | WIRED | Same; `still_cdef_index_syntax.h:132-220` | Opt-in. |
| `ccso_blk S()` | §5.20.10.2 p245 | WIRED | `still_syntax.cpp:1322-1360`; `--still-ccso` gate | Band-offset-only form; `bo_only = 1`. |
| `use_gdf S()` | §5.20.10.3 p245 | ABSENT | No syntax writer; `src/encoder/filters/spec_tables/gdf.h:20` only has filter/side-info; `still_headers.cpp:147` writes `enable_gdf = 0` | GDF filter process exists, block symbol does not. |
| `use_wiener_ns S()` | §5.20.10.5 p247 | COMPONENT | `src/encoder/filters/restoration/classes/wiener_syntax.cpp:1613`; `wiener_syntax_test.cpp` | No M1/M2 integration; no luma `dav2d` stream. Chroma path has a gate (`filters/lr_chroma/gate`). |
| `use_pc_wiener S()` | §5.20.10.5 p247 | ABSENT | `wiener_syntax.cpp:950-953` refuses to write; reader also errors (`:1696-1697`) | PC-Wiener tables not in the PDF; deliberately unsupported. |
| `flex_restoration_type S()` | §5.20.10.5 p247 | COMPONENT | `wiener_syntax.cpp:1644-1655` | No integration. |
| `use_alt_group f(1)` | §5.20.10.6 p248 | COMPONENT | `wiener_syntax.cpp:638` | Frame-filter group coding only. |
| `group_bit f(1)` | §5.20.10.6 p248 | COMPONENT | `wiener_syntax.cpp:646-658` | — |
| `merged_param f(1)/L(1)` | §5.20.10.6 p249 | COMPONENT | `wiener_syntax.cpp:857` | — |
| `use_bank L(1)` | §5.20.10.6 p249 | COMPONENT | `wiener_syntax.cpp:1365-1371` (`use_bank_k`) | — |
| `wiener_ns_length f(1)/S()` | §5.20.10.6 p250 | COMPONENT | `wiener_syntax.cpp:1415` | — |
| `wiener_ns_uv_sym f(1)/S()` | §5.20.10.6 p250 | COMPONENT | `wiener_syntax.cpp:1428` | Chroma symmetry pairs. |
| `wiener_ns_base S()` / `wiener_ns_rem L(bits)` | §5.20.10.6 p251 | COMPONENT | `wiener_syntax.cpp` `encode_signed_4part`; `:1515` | — |
| Deblocking parameters (`apply_deblocking_filter[]`, `df_delta_q_present`) | §5.18.5.2 (not §5.20) | WIRED | `still_headers.cpp:191-200`; `still_encoder.cpp:1494-1508`; `filters/deblock/deblock_test.cpp` | Frame-header syntax only; no block-level deblock symbols. |

## 11. §6.18–§6.19 Semantics / conformance constraints (pp. 349–375)

| Item | §/page | Status | Evidence file:line or test | Gap |
|---|---|---|---|---|
| `SeenFrameHeader != is_first_tile_group` | §6.18 p349 | WIRED | `still_headers.cpp:287`; `tiles_stream.cpp:428`; `m2/post/tiled_frame.cpp:678` | Single group; constraint holds trivially. |
| `tg_start == TileNum` / `tg_end == NumTiles - 1` | §6.18 p349 | ABSENT | No `tg_start`/`tg_end` writer | Partial/multi-group streams impossible. |
| `tileSize` includes padding, excludes size field; last tile includes OBU trailing bits | §6.19.1 p349 | PARTIAL | `src/encoder/tiles/tile_group.cpp:11-120`; `tiles_stream.cpp:398-447` | Multi-tile component models it; no partial groups or multi-group framing. |
| `ChromaFollowsLuma` / `ChromaPartitionKnown` / `TreeType` semantics | §6.19.3.1 p353–354 | COMPONENT | `src/encoder/sdp/sdp.cpp`, `sdp_validate.cpp`, `sdp_test.cpp` | No decodable stream. |
| `MaxPbAspectRatio` block conformance | §6.19.4.1 p354 | WIRED | `src/encoder/partition/partition.cpp:985-990`; `partition_test.cpp` | 8:1 limit enforced. |
| `Quant` magnitude `< 1 << 20` | §6.19.7.23 p369 | PARTIAL | No bound check in `still_syntax.cpp` / `video_syntax.cpp`; levels are `int32_t` | Encoder could emit a non-conforming magnitude for pathological input. |
| `is_mv_valid()` / `check_valid_local_ibc()` for IBC | §6.19.7.12 p362–364 | COMPONENT | `src/encoder/m1/ibc_stream/ibc_stack.h`; `ibc_stream_test.cpp`; IBC `dav2d --strict 1` gate | Not wired into M1. |
| `RefFrame`, `YMode`, `TxSize`, `cdef_idx` enum semantics | §6.19.5–.9 p355–373 | PARTIAL | `video_syntax.h` enums; `still_syntax.cpp` block enums; `partition.h` block sizes | Baseline values only; no B/compound/TIP/warp values in a wired stream. |

---

## 12. Derived-only helpers checked (no syntax element)

These functions are in scope but emit no bits; they are listed so the audit is
not read as "not inspected". They are **not** part of the 167 syntax rows/counts.

| Helper | §/page | Status | Evidence |
|---|---|---|---|
| `neg_deinterleave()` / `get_segment_id()` | §5.20.3.2 p159; §5.20.7.4 p189 | COMPONENT | `src/encoder/m1/deltaq/deltaq.cpp` (`neg_interleave`, `segment_id_context`) |
| `get_mode()` (YMode → component mode) | §5.20.7.19 p211 | WIRED | `m2/merge` stack; `video_syntax.cpp:1633+` |
| `get_tx_size()` / `get_plane_residual_size()` | §5.20.7.25–.26 p223 | WIRED | `src/encoder/partition/partition.cpp`; `still_syntax.cpp` block-size validation |
| `compute_tx_type()` / `get_tx_set()` | §5.20.7.29 p231; §5.20.8.3 p239 | WIRED | `still_syntax.cpp:1043-1108`; `src/encoder/tx_type/tx_type.cpp` |
| `get_scan()` / `get_tx_row_col()` / `get_lf_limits()` | §5.20.7.30 p233; §5.20.7.27 p230 | WIRED | `still_syntax.cpp:1595+`; `still_syntax.cpp:1120-1145` |
| `is_directional_mode()` | §5.20.7.31 p234 | WIRED | `still_syntax.cpp:89`; `still_syntax.cpp:127` |
| `get_palette_cache()` / `get_palette_color_context()` | §5.20.8.1 p236; §5.20.8.5 p242 | COMPONENT | `src/encoder/palette/palette.cpp`; `m1/palette_stream` |
| `is_inside()` / `is_inside_frame()` / `is_inside_filter_region()` | §5.20.9.1–.3 p243 | PARTIAL / COMPONENT | M1 uses frame bounds (single tile); tile-relative form in `m1/tiles_stream`; filter-region form in `src/encoder/filters/restoration/` |
| `clamp_mv_row()` / `clamp_mv_col()` | §5.20.9.4–.5 p243 | WIRED / COMPONENT | MC/clamping in `src/encoder/inter/mc`; M2 writer validates MVs via `merge` |
| `clear_cdef()` | §5.20.9.6 p244 | WIRED | `still_syntax.cpp:1415`; `still_cdef_index_syntax.h` |
| `decode_signed_4part()` / `decode_unsigned_4part()` / `decode_4part()` | §5.20.10.6 p251 | COMPONENT | `src/encoder/filters/restoration/classes/wiener_syntax.cpp`; `wiener_syntax_test.cpp` |
| `search_frame_filters()` / `fill_first_slot_of_bank_with_filter_match()` / `predict_group()` | §5.20.10.6 p248–252 | COMPONENT | `wiener_syntax.cpp`; `wiener_decision.cpp`; `wiener_syntax_test.cpp` |

---

## Tally method and caveats

- Counted *table rows*, not distinct syntax-token names. Closely related fields
  that share one writer/reader decision are grouped (for example all seven
  `eob_pt_*` CDFs, or the `shell_*` family). The total is therefore a
  coverage-row count, not the number of identifiers in the PDF.
- `WIRED` includes opt-in M1 tools (`--still-tx-type`, `--still-cdef`, `--still-ccso`)
  because they are exposed by `src/main.cpp` and have recorded `dav2d` gates.
  It also includes the M2 P-frame pipeline, which is a gated encoder but not
  exposed by `src/main.cpp`; see `src/encoder/m2/README.md`.
- `COMPONENT` rows may still have a passing `dav2d` gate (palette, IBC, MRL,
  sub-8, extended partitions, sb128/sb256, compound). "Component" means *not
  reachable from the shipped encode path*, not "unverified".
- M1's main stream is one tile and one frame. Therefore elements that are
  decoder-side bookkeeping for multi-tile/multi-frame operation
  (`frame_end_update_cdf`, `decode_frame_wrapup`, CDF averaging) are marked
  DECODER-ONLY rather than missing compression tools.
- The M1 writer currently emits `TX_MODE_LARGEST`, contrary to the phrase
  "variable transforms" in `README.md`; this audit follows
  `still_syntax.cpp:1389-1400` and `still_headers.cpp:258`.

---

## Counts

<!-- COUNT_TABLE -->
| Status | Count |
|---|---:|
| WIRED | 51 |
| COMPONENT | 74 |
| PARTIAL | 11 |
| ABSENT | 29 |
| DECODER-ONLY | 2 |
| **Total** | **167** |

Per-section distribution (same 167 rows):

| Section | WIRED | COMPONENT | PARTIAL | ABSENT | DECODER-ONLY | Rows |
|---|---:|---:|---:|---:|---:|---:|
| §5.19 tile-group OBU | 3 | 2 | 1 | 6 | 2 | 14 |
| §5.20.2 tile-level structures | 6 | 3 | 1 | 1 | 0 | 11 |
| §5.20.3 partition structures | 5 | 8 | 0 | 0 | 0 | 13 |
| §5.20.4 block decoding | 2 | 1 | 1 | 1 | 0 | 5 |
| §5.20.5 mode information | 5 | 15 | 1 | 3 | 0 | 24 |
| §5.20.6 TX / quantisation | 2 | 3 | 1 | 3 | 0 | 9 |
| §5.20.7 MV / prediction | 7 | 17 | 0 | 10 | 0 | 34 |
| §5.20.7.23–.32 coefficients | 11 | 5 | 2 | 1 | 0 | 19 |
| §5.20.8 coding tools | 4 | 9 | 1 | 1 | 0 | 15 |
| §5.20.10 filters | 4 | 9 | 0 | 2 | 0 | 15 |
| §6.18–§6.19 conformance | 2 | 2 | 3 | 1 | 0 | 8 |

*Counts are regenerated from the rows above; see “Tally method and caveats” above.*

## The 10 most impactful missing block-level elements

Ranked by expected rate/quality or throughput impact on the shipped encoder,
not by implementation cost. "Missing" means not reachable from a shipped
encoder path (`src/main.cpp` M1 or the gated M2 P-frame pipeline), even when an
isolated component exists.

1. **Transform partition + `TX_MODE_SELECT`** — `tx_do_partition`,
   `tx_partition_type`, `tx_2or3_partition_type`, `read_tx_size`
   (§5.20.6.1–.3, pp. 182–186). Entirely ABSENT. M1/M2 are locked to one
   transform per plane (`still_headers.cpp:258`), which is the largest single
   intra/inter RD loss in the current encoder. This is the top gap for both
   stills and video.
2. **Segmentation and per-block delta-Q** — `segment_id`, `seg_id_ext_flag`,
   `seg_id_predicted`, `delta_q_abs`/`_rem_bits`/`_abs_bits`/`_sign_bit`
   (§5.20.5.7–.11, pp. 178–181). A complete component exists
   (`src/encoder/m1/deltaq/`) but only the frame-header part is gated; the
   block symbols are not wired. Adaptive quantisation is the biggest
   rate-control win available without new prediction tools.
3. **Compound and B-frame inter syntax** — `comp_mode`, `comp_ref`,
   `compound_mode_*`, `comp_group_idx`, `compound_type`, `cwp_idx`, plus
   `single_ref` (§5.20.7.6–.16, pp. 190–209). The compound gate
   (`m2/multiref/gate/`) decodes but the M2 core writes
   `num_total_refs = 1` and `reference_select = 0`; `single_ref` and
   `compound_mode_same_refs` have no writer at all.
4. **Warped motion, BAWP, inter-intra, TIP, refine-MV, OPFL in the wired path** —
   `is_warp`, `warp_mv`, `warp_idx`, `warpmv_with_mvd`, `warp_delta_*`,
   `use_local_warp`, `use_extend_warp`, `use_bawp`, `explicit_bawp`,
   `use_bawp_chroma`, `inter_intra`, `interintra_mode`, `wedge_interintra`,
   `tip_mode`, `tip_pred_mode`, `use_refinemv`, `use_optflow`
   (§5.20.7.6–.17, pp. 190–209). Writers exist in `m2/advanced/` and
   `inter/temporal/`, but neither is integrated or gated; `use_optflow` has no
   block writer anywhere.
5. **MRL, DIP, MHCCP and IBP intra tools** — `mrl_index`, `mrl_sec_index`,
   `use_dip`/`dip_transpose`/`dip_mode`, `cfl_mhccp`/`cfl_mh_dir`,
   and the sequence-derived IBP application (§5.20.5.5–.6, §5.20.7.9/32,
   pp. 175–177/199/235). Each has a tested component, but the M1 sequence
   header hardcodes the enabling flags to 0 (`still_headers.cpp:114-120`) and
   `still_syntax` has no corresponding leaf fields.
6. **Palette and IntraBC in the shipped intra path** — all
   `palette_mode_info`/`palette_tokens` fields and
   `use_intrabc`/`intrabc_mode`/`intrabc_drl_mode`/`intrabc_precision`
   (§5.20.5.4, §5.20.8.1/.4, pp. 171–173/235–241). Both are decoder-gated in
   isolated streams (`.ivf` 101–132 bytes) but `allow_screen_content_tools = 0`
   in M1, so screen content gets none of the largest gains available to it.
7. **Coefficient tools: CCTX, TCQ, parity hiding, FSC/IDTX and
   `dc_sign_horz_vert`** — `cctx_type`, `coeff_base_bob`, `coeff_base_idtx`,
   `coeff_br_idtx`, `idtx_sign`, TCQ state/parity (§5.20.7.27–.28,
   pp. 224–231; §5.20.8.2, p238). TCQ/parity have a coefficient-payload
   round-trip but no tile gate; CCTX has only a rate estimate; the IDTX path is
   blocked because M1 sets `enable_idtx_intra = 0`, so `dc_sign_horz_vert`
   is unreachable too.
8. **Extended partitions and SDP** — `do_ext_partition`,
   `do_uneven_4way_partition`, `uneven_4way_partition_type`, `region_type`
   (§5.20.3.1–.2, pp. 150–157). The `m1/partition_stream` adapter passes a
   block-level `dav2d` gate for HORZ_3/4-way, but is not merged into
   `still_syntax`; SDP has only a symbol planner and no stream. This limits
   partition shapes exactly where 64×64 roots and 4:2:0 chroma constraints
   matter.
9. **Loop restoration and the non-band CCSO/GDF forms** — `use_wiener_ns`,
   `flex_restoration_type`, `use_alt_group`, `group_bit`, `merged_param`,
   `use_bank`, `wiener_ns_length`, `wiener_ns_uv_sym`, `wiener_ns_base/rem`,
   `use_gdf`, `use_pc_wiener`, plus edge-classified `ccso_blk`
   (§5.20.10.1–.6, pp. 244–254). Wiener NS syntax is complete only as a
   component; GDF and PC-Wiener have no block writer at all, and M1 only
   signals band-offset CCSO. These are the post-CDEF quality tools.
10. **Multi-tile-group framing** — `frame_header_present_flag`, `tg_start`,
    `tg_end`, `bru_tile_active` (§5.19, pp. 144–145). Every writer emits one
    all-tiles group; partial groups, non-first groups and BRU tile activity are
    absent. This does not change RD, but it is the blocker for independent
    per-tile decode/GPU tile parallelism and for any future tile-group
    robustness feature.

**Honourable mentions:** switchable per-block `interp_filter` (fixed
`EIGHTTAP`), adaptive MVD (`use_amvd`/`amvd_index`/`mv_joint`/`jmvd_scale_mode`),
per-block MV precision (`use_most_probable_precision`/`pb_mv_precision`), and
`lossless_inter_tx_type` — all ABSENT and each cheap to add relative to the ten
above, but lower expected gain.

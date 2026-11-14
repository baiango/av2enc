# AV2 v1.0.0 sequence header audit (§5.4.1–§5.4.13, §6.4)

Audit date: 2026-11-12. Read-only audit; no encoder code was changed.

Normative source: `spec/AV2_Spec_v1.0.0.pdf` (v1.0.0, 2026-05-28,
commit `38f28e7`, 1169 pages). Page numbers below are the **printed spec
pages** (pypdf list index = printed page − 1): §5.4.1 starts p55, §5.4.13 ends
p69; §6.4 runs p258–p274.

Method: every claim was checked against a bit writer in `src/`, not against
`docs/`. As a cross-check, 64×64 M0 and M1 streams were encoded with the
existing binaries and the sequence-header payload was bit-parsed independently:
both are exactly **83 syntax bits** (`src/bitstream/basic_headers.h:96`) and match
the table values below. M0/M1 write `seq_tier` as absent, `base_uv_ac_delta_q`
as 23 (→ `BaseUVAcDeltaQ = 0`), `enable_avg_cdf = 1`, `avg_cdf_type = 1`, and
the M1-only bits `enable_intra_edge_filter = 1`, `enable_cfl_intra = 1`,
`enable_chroma_dctonly = 1`.

## Status vocabulary

| Status | Meaning in this audit |
|---|---|
| **WIRED** | At least one complete, decoder-gated stream path sets the field to a meaningful value (production CLI, standalone feature encoder, or feature gate that owns its sequence header). |
| **PARTIAL** | The field is coded, but (a) only via a header rewrite/bit-splice helper rather than an encoder that owns the stream, or (b) only a subset of legal values is reachable, or (c) the writer is malformed for part of the legal range. |
| **COMPONENT** | A syntax writer exists and is unit-tested, but no complete stream in the tree exercises a non-default value. |
| **ABSENT** | No stream-producing path can set a non-trivial value; the corresponding coding tool is unreachable. |
| **DECODER-ONLY** | Only a reader exists. |
| **OUT-OF-SCOPE** | Not a sequence-header field (called from another OBU). |

“Current value” is the integrated **M1** writer
(`src/encoder/m1/still_headers.cpp`) unless stated. M0
(`src/bitstream/basic_headers.cpp`) is the CLI default and differs only where
noted. Fields that `single_picture_header_flag = 1` deletes are marked
“absent (inferred)” and the forcing line is quoted.

## Path inventory (why “another value” is or is not reachable)

| Path | Kind | Sequence-header writer | Notes |
|---|---|---|---|
| `av2enc` default (M0) | production CLI | `src/bitstream/basic_headers.cpp:167` | still picture, all tools off |
| `av2enc --still` (M1) | production CLI | `src/encoder/m1/still_headers.cpp:77` | still picture; edge filter, CfL, chroma-DCT-only, opt-in CDEF/CCSO |
| lossless reference | standalone encoder | `src/encoder/lossless/lossless.cpp:2798` | 8-bit 4:2:0, q=0, FSC option |
| M2 | standalone inter encoder | `src/encoder/m2/video_headers.cpp:90` | first non-single-picture header in the tree |
| M10 / HBD | standalone 10-bit encoders | `src/encoder/m10/high_bitdepth_headers.cpp:85`, `src/encoder/hbd/hbd_headers.cpp:85` | `bit_depth_idc = 0` |
| 4:0:0 / 4:2:2 / 4:4:4 | standalone encoder | `src/encoder/format/chroma/chroma_headers.cpp:329` | full Monochrome conditionals |
| sb128 / sb256 | standalone encoders | `src/encoder/sb128/sb128_stream.cpp:432`, `src/encoder/sb256/sb256_stream.cpp:394` | `use_128x128` / `use_256x256` |
| MRL | standalone encoder | `src/encoder/m1/mrl_stream/mrl_stream.cpp:539` | `enable_mrls = true` |
| tiles | standalone encoder | `src/encoder/m1/tiles_stream/tiles_stream.cpp:110` | frame-level multi-tile |
| tlayers / longterm / random-access / leading / GME / DRL / compound | standalone inter components with gates | see per-field evidence | narrow tool sets |
| small_header | header rewrite gate | `src/encoder/syntax/small_header/gate.cpp:209` | crop, PB aspect, `df_par`, seq tiles, seq segments |
| QM | OBU/header splice gate | `src/encoder/qm/qm.cpp:95` | QM OBU + `using_qmatrix` |
| dcdeltaq | header splice gate | `src/encoder/quant/dcdeltaq/syntax.cpp:273` | full §5.4.8 delta-Q |
| HRD/timing | stream rewrite gate | `src/encoder/rc/hrd/hrd.cpp:427` | display delay, decoder model, CI timing |

---

## §5.4.1 General sequence header (p55–p58)

| Field | §/page | Current value | Status | Evidence file:line | Gap |
|---|---|---|---|---|---|
| `seq_header_id` | §5.4.1 p55 | `put_uvlc(writer, 0)` (uvlc 0); `config.seq_header_id` | WIRED (CPU `--video`, opt-in) | `video_headers.cpp` (sequence header) | Default id 0. `--video-multi-sequence` carries a second header with id 1; the decoder-gated stream selects it from every frame header. |
| `seq_profile_idc` | §5.4.1 p55 / §A.2 p1101 | `put_bits(0U, 5U)` — profile 0 (`Main_420_10_IP0`) | WIRED | `src/encoder/m1/still_headers.cpp:81` | Nonzero profiles only in the chroma component: 4:2:2 → 3, 4:4:4 → 4 (`src/encoder/format/chroma/chroma_format.cpp:49-57`). CLI has no profile selection. |
| `single_picture_header_flag` | §5.4.1 p55 | `put_bit(true)` | WIRED | `src/encoder/m1/still_headers.cpp:82` | Both values exist: M2/leading/tlayers/longterm/random-access write 0 (`src/encoder/m2/video_headers.cpp:98`). This one bit deletes every non-single field below. |
| `seq_level_idx` | §5.4.1 p55 / §A.4 p1104 | `put_bits(c.seq_level_idx, 5U)`, default 7 = level 5.1 | PARTIAL | `src/encoder/m1/still_headers.cpp:83` | No CLI/API to choose a level. Lossless is the only other validated value (31 = “Maximum parameters”, written at `src/encoder/lossless/lossless.cpp:2804`; default at `src/encoder/lossless/lossless.h:152`). The integrated writers assume level-5.1 tile limits: M0 asserts them (`src/bitstream/basic_headers.cpp:80-88`), and the tiles encoder derives limits from `seq_level_idx`; other levels are not validated. |
| `seq_tier` | §5.4.1 p55 / §A.4 p1106 | absent for single picture; M2 writes 0; `config.seq_tier` | WIRED (CPU `--video`, opt-in) | `video_headers.cpp` | `--video-tier high` writes 1 for `seq_level_idx > 3`; the bit is omitted for levels 0–3. Decoder-gated by `gate_video_headers.sh`. |
| `chroma_format_idc` | §5.4.1 p55 / §6.4.1 p259 | `put_uvlc(writer, 0)` = 4:2:0 | WIRED | `src/encoder/m1/still_headers.cpp:84` | 4:0:0/4:2:2/4:4:4 only in the standalone chroma encoder (`src/encoder/format/chroma/chroma_headers.cpp:341`), not in the CLI. |
| `bit_depth_idc` | §5.4.1 p55 / §6.4.1 p260 | `put_uvlc(writer, 1)` = 8-bit | WIRED | `src/encoder/m1/still_headers.cpp:85` | 10-bit is a separate standalone encoder (`put_uvlc(writer, 0)`, `src/encoder/m10/high_bitdepth_headers.cpp:98`, `src/encoder/hbd/hbd_headers.cpp:99`). 12-bit is reserved by Table 6.3, correctly never emitted. |
| `set_chroma_format_and_bit_depth()`: `Monochrome` | §6.4.1 p260 | `chroma_format_idc == 400` → 0 in all CLI paths | WIRED | `src/encoder/format/chroma/chroma_format.cpp:43` | Only the chroma component encodes Monochrome; the Monochrome syntax deletions are implemented there (`src/encoder/format/chroma/chroma_headers.cpp:377-380`, `398-402`, `422-447`). |
| `set_chroma_format_and_bit_depth()`: `BitDepth`, `MaxQ` | §6.4.1 p260, Table 6.3 | 8 / 255 | WIRED | `src/encoder/format/format.cpp:97-127` | 10 / 303 in M10/HBD; 12 / 351 is internal-only and cannot be signalled. |
| `NumPlanes`, `SubsamplingX/Y` | §6.4.1 p259-260, Table 6.2 | 3 / 1,1 | WIRED | `src/encoder/format/chroma/chroma_format.cpp:39-44` | Derived correctly for all four formats in the chroma encoder. |
| `seq_lcr_id` | §5.4.1 p56 | absent for single (inferred 0); M2 writes 0; `--video-lcr-full` writes 1 | WIRED (CPU `--video`, opt-in) | `video_headers.cpp`; `pipeline/svc/ops_full` | Emits a field-complete global/local LCR and OPS before the sequence header; avmdec parses and validates them (`gate_video_headers.sh`). |
| `still_picture` | §5.4.1 p56 | absent for single (inferred 1); M2 `put_bit(false)` | WIRED | `src/encoder/m2/video_headers.cpp:106` | Both values reachable. |
| `max_tlayer_id` | §5.4.1 p56 | absent for single (inferred 0); M2 `put_bits(0U, 2U)` | WIRED | `src/encoder/m2/video_headers.cpp:107` | `tlayers` writes 0–3 and the dependency block (`src/encoder/inter/tlayers/tlayers_syntax.cpp:117`, `143-154`). `longterm` writes the field but omits the required `tlayer_dependency_present_flag` when `> 0` (`src/encoder/inter/longterm/longterm_headers.cpp:213-217`), so its >0 path is malformed. |
| `max_mlayer_id` | §5.4.1 p56 | absent for single; M2 `put_bits(0U, 3U)` | PARTIAL | `src/encoder/m2/video_headers.cpp:108` | Only `longterm` writes a var (`:214`); it then omits both `seq_max_mlayer_cnt_minus_1` and `mlayer_dependency_present_flag`, so any value >0 is malformed. SVC plans embedded layers but no writer emits them. |
| `seq_max_mlayer_cnt_minus_1` | §5.4.1 p56 | never coded (`SeqMaxMlayerCnt = 1`) | ABSENT | `src/encoder/inter/tlayers/tlayers_syntax.cpp:119` (comment: absent) | No writer anywhere emits it. Multi-embedded-layer streams cannot be produced. |
| `monotonic_output_order_flag` | §5.4.1 p56 | absent for single (inferred 1); M2 `put_bit(true)` | WIRED | `src/encoder/m2/video_headers.cpp:110` | Leading-picture encoder writes 0 and validates it (`src/encoder/inter/leading/leading.cpp:102`). |
| `frame_width_bits_minus_1` | §5.4.1 p56 | `put_bits(width_bits - 1U, 4U)` (5 for 64×64) | WIRED | `src/encoder/m1/still_headers.cpp:93` | Computed from actual max width; no bug found. It is not minimal (n may exceed the minimum needed), which is legal. |
| `frame_height_bits_minus_1` | §5.4.1 p56 | `put_bits(height_bits - 1U, 4U)` | WIRED | `src/encoder/m1/still_headers.cpp:94` | Same. |
| `max_frame_width_minus_1` | §5.4.1 p56 | `put_bits(max_frame_width - 1U, width_bits)` | WIRED | `src/encoder/m1/still_headers.cpp:95-96` | Equal to the visible width; no coded-size/padded-size separation (that would require cropping). |
| `max_frame_height_minus_1` | §5.4.1 p56 | `put_bits(max_frame_height - 1U, height_bits)` | WIRED | `src/encoder/m1/still_headers.cpp:97-98` | Same. |
| `seq_cropping_window_present_flag` | §5.4.1 p56 / §6.4.1 p261 | `put_bit(false)` | PARTIAL | `src/encoder/m1/still_headers.cpp:100` | Implemented by small_header rewrite (`src/encoder/syntax/small_header/gate.cpp:229`) and accepted by the chroma writer (`src/encoder/format/chroma/chroma_headers.cpp:361`); no CLI/encoder path produces a cropped frame. Pinned `dav2d` does not crop its Y4M output. |
| `seq_cropping_win_{left,right,top,bottom}_offset` | §5.4.1 p56 | absent (inferred 0) | PARTIAL | `src/encoder/syntax/small_header/small_header.cpp:402-410` | Written only by the small_header rewrite; all CLI streams infer 0. |
| `seq_initial_display_delay_present_flag` | §5.4.1 p56 / §6.4.1 p262 | absent for single; M2 `put_bit(false)` | PARTIAL | `src/encoder/rc/hrd/hrd.cpp:455` | HRD stream rewrite can set 1; no encoder-owned header does. |
| `seq_initial_display_delay_minus_1` | §5.4.1 p56 | absent (inferred `NumRefFrames + 1`) | PARTIAL | `src/encoder/rc/hrd/hrd.cpp:457` | Same; only HRD splice. |
| `decoder_model_info_present_flag` | §5.4.1 p56 / §6.4.1 p262 | absent for single; M2 `put_bit(false)` | PARTIAL | `src/encoder/rc/hrd/hrd.cpp:459` | Only HRD splice. No production encoder signals a decoder model. |
| `num_units_in_decoding_tick` | §5.4.1 p57 / §6.4.1 p262 | absent | PARTIAL | `src/encoder/rc/hrd/hrd.cpp:461` | HRD splice only. |
| `seq_decoder_model_info_present_flag` | §5.4.1 p57 | absent | PARTIAL | `src/encoder/rc/hrd/hrd.cpp:462` | HRD can write it, but pinned `dav2d f9dafbc` rejects the spec v1.0.0 `seq_decoder_model_info()` layout; the gate deliberately falls back to 0 (`src/encoder/rc/hrd/hrd_stream_gate.cpp:411-417`, `725-733`). Open decoder-vs-spec question. |
| `mlayer_dependency_present_flag` | §5.4.1 p57 / §6.4.1 p264 | never coded | ABSENT | `src/encoder/inter/tlayers/tlayers_syntax.cpp:139-142` (max_mlayer = 0) | No writer emits the block. |
| `mlayer_dependency_map` | §5.4.1 p57 | never coded | ABSENT | `src/encoder/tiles/svc_layers.cpp:107-126` (validation only) | Arrays are validated in SVC plans but never serialized. |
| `tlayer_dependency_present_flag` | §5.4.1 p57 | never coded for single; M2 max_tlayer=0 | WIRED | `src/encoder/m2/video_headers.cpp:107` | tlayers writes it and custom maps when `max_tlayer_id > 0` (`src/encoder/inter/tlayers/tlayers_syntax.cpp:143-154`). |
| `multi_tlayer_dependency_map_present_flag` | §5.4.1 p57 | never coded (inferred 0) | ABSENT | `src/encoder/inter/tlayers/tlayers_syntax.cpp:139-142` | Requires `max_mlayer_id > 0`; unreachable. |
| `tlayer_dependency_map` | §5.4.1 p58 | default `ref <= curr` | WIRED | `src/encoder/m2/video_headers.cpp:107` | Custom maps written by tlayers only (`src/encoder/inter/tlayers/tlayers_syntax.cpp:146-152`). |
| `film_grain_params_present` | §5.4.1 p58 / §6.4.1 p264 | `put_bit(false)` | PARTIAL | `src/encoder/m1/still_headers.cpp:159` | Chroma writer can set 1 (`src/encoder/format/chroma/chroma_headers.cpp:476`), but there is no film-grain OBU writer anywhere (`OBU_FILM_GRAIN` appears only as an enum in `src/encoder/svc/svc_framing.h:53`). Film grain synthesis is unreachable. |
| `save_sequence_header()` | §5.4.1 p58 | one id-0 header per stream | PARTIAL | `src/encoder/m1/still_headers.cpp:275-282` | No encoder-side sequence-header store; MFH has the presence map but no sequence writer (`src/encoder/frame/mfh/mfh.h:193`). |

## §5.4.2 Sequence tile config (p59) and §6.4.2 (p264)

| Field | §/page | Current value | Status | Evidence file:line | Gap |
|---|---|---|---|---|---|
| `seq_tile_info_present_flag` | §5.4.2 p59 / §6.4.2 p264 | `put_bit(false)` | PARTIAL | `src/encoder/m1/still_headers.cpp:157` | Only small_header rewrite sets 1 (`src/encoder/syntax/small_header/gate.cpp:277`). Every encoder describes tiles at frame level. `chroma_headers` writes the flag from config but **does not write `allow_tile_info_change`/`tile_params`** when 1 (`src/encoder/format/chroma/chroma_headers.cpp:473`), a latent malformed-header bug (validator does not reject it). |
| `allow_tile_info_change` | §5.4.2 p59 / §6.4.2 p264 | absent | PARTIAL | `src/encoder/syntax/small_header/small_header.cpp:1359-1377` | Written only when the small_header rewrite sets `present`; no encoder path. |
| `tile_params()`: `uniform_tile_spacing_flag` | §5.4.2 p59 / §5.18.7.3 p123 | frame-level `put_bit(true)` (all streams) | PARTIAL | `src/encoder/m1/still_headers.cpp:68` (frame header) | All streams are uniform. Non-uniform spacing is implemented only by small_header (`src/encoder/syntax/small_header/small_header.cpp:1014`), never used by a stream. |
| `tile_params()`: `increment_tile_cols_log2` / `increment_tile_rows_log2` | §5.4.2 p59 / §5.18.7.3 p123 | frame-level 0 for single tile; tiles_stream writes nonzero for presets | PARTIAL | `src/encoder/m1/tiles_stream/tiles_stream.cpp:355-371` | Multi-tile exists only in the standalone tiles encoder; CLI is one tile. |
| `tile_params()`: explicit `tile_width/height` lists | §5.18.7.3 p123 | absent | COMPONENT | `src/encoder/syntax/small_header/small_header.cpp:1014` | Non-uniform starts are unit-tested but no gate stream uses them. |
| `SeqSbSize` / `sbShift` | §5.4.2 p59 / §5.18.7.6 | 64×64 / 4 | WIRED | `src/encoder/sb256/sb256.cpp:570` | 128/256 superblocks in sb128/sb256 (`src/encoder/sb256/sb256_stream.cpp:418-421`). |
| `SeqSbRows`, `SeqSbCols` | §5.4.2 p59 / §5.18.7.3 | derived from frame size (one grid) | WIRED | `src/encoder/tiles/tile_layout.cpp:147-160`, `:257-258` | Correct for one tile; sequence-level grid only via small_header. |
| `SeqTileRows/Log2`, `SeqTileCols/Log2` | §5.4.2 p59 / §5.18.7.3 | 1 / 0 | PARTIAL | `src/encoder/m1/tiles_stream/tiles_stream.cpp:355-371` | Nonzero only in the standalone tiles encoder. |
| `SeqUniformTileSpacingFlag` | §5.4.2 p59 | 1 | PARTIAL | `src/encoder/syntax/small_header/small_header.cpp:1014` | 0 never used by a stream. |

## §5.4.3 Sequence partition config (p59–p60) and §6.4.3 (p264–p265)

| Field | §/page | Current value | Status | Evidence file:line | Gap |
|---|---|---|---|---|---|
| `use_256x256_superblock` | §5.4.3 p59 / §6.4.3 p264 | `put_bit(false)` | WIRED | `src/encoder/m1/still_headers.cpp:103` | 1 only in the standalone sb256 encoder (`src/encoder/sb256/sb256_stream.cpp:418`, `make_stream` with `use_256 = true`). CLI is 64×64 SB only. |
| `use_128x128_superblock` | §5.4.3 p59 / §6.4.3 p265 | `put_bit(false)` | WIRED | `src/encoder/m1/still_headers.cpp:104` | 1 in sb128 (`src/encoder/sb128/sb128_stream.cpp:456`) and in sb256's 128 control stream (`src/encoder/sb256/sb256_stream.cpp:420`). CLI rejects 128 (`src/encoder/m1/tiles_stream/tiles_stream.h:79`). |
| `enable_sdp` | §5.4.3 p59 / §6.4.3 p265 | `put_bit(false)` | ABSENT | `src/encoder/m1/still_headers.cpp:105` | Every stream-producing path forces 0; the chroma writer exposes the bit but its validator rejects 1 (`src/encoder/format/chroma/chroma_headers.cpp:242-245`). The SDP component exists (`src/encoder/sdp/`) but no sequence writer can enable it. Semi-decoupled chroma partitions are unreachable. |
| `enable_extended_sdp` | §5.4.3 p59 / §6.4.3 p265 | absent (inferred 0) | ABSENT | `src/encoder/m2/video_headers.cpp:132` (enable_sdp = 0) | Requires `enable_sdp = 1` and non-single; both unreachable. |
| `enable_ext_partitions` | §5.4.3 p59 / §6.4.3 p265 | `put_bit(false)` | PARTIAL | `src/encoder/m1/still_headers.cpp:106` | No C++ writer sets 1. The partition_stream gate writes extended partitions in tile syntax and a Python step patches the two header bits (`src/encoder/m1/partition_stream/gate_minimal.cpp:10-16`). HORZ_3/HORZ_4A/B and VERT_3/4 shapes are therefore not producible by an encoder-owned header. |
| `enable_uneven_4way_partitions` | §5.4.3 p59 / §6.4.3 p265 | absent (inferred 0) | PARTIAL | `src/encoder/m1/partition_stream/gate_minimal.cpp:328` (tools only) | Same Python header patch; no header writer. |
| `reduce_pb_aspect_ratio` | §5.4.3 p59 / §6.4.3 p265 | `put_bit(false)` | PARTIAL | `src/encoder/m1/still_headers.cpp:107` | Only small_header rewrite sets 1 (`src/encoder/syntax/small_header/gate.cpp:235`). sb128/sb256 hardcode `max_pb_aspect_ratio = 8` (`src/encoder/sb128/sb128_stream.cpp:77`), never signalling a reduced value. |
| `max_pb_aspect_ratio_log2_minus_1` | §5.4.3 p59 / §6.4.3 p265 | absent (inferred `MaxPbAspectRatio = 8`) | PARTIAL | `src/encoder/syntax/small_header/small_header.cpp:514-530` | 0/1 (→ 2/4) only in the rewrite; no encoder path. |

## §5.4.4 Sequence segment config (p60) and §5.4.9 `seg_info()` (p66)

| Field | §/page | Current value | Status | Evidence file:line | Gap |
|---|---|---|---|---|---|
| `enable_ext_seg` | §5.4.4 p60 / §6.4.4 p265 | `put_bit(false)` | PARTIAL | `src/encoder/m1/still_headers.cpp:110` | Only small_header rewrite sets 1 (`src/encoder/syntax/small_header/small_header.cpp:755`); `MaxSegments` is 8 in every encoder. |
| `MaxSegments` | §5.4.4 p60 / §6.4.4 p265 | 8 | PARTIAL | `src/encoder/m1/still_headers.cpp:110` | 16 only through the small_header rewrite. |
| `seq_seg_info_present_flag` | §5.4.4 p60 / §6.4.4 p266 | `put_bit(false)` | PARTIAL | `src/encoder/m1/still_headers.cpp:111` | Only small_header rewrite. Chroma writer accepts the flag but its validator rejects 1 (`src/encoder/format/chroma/chroma_headers.cpp:257-260`). |
| `seq_allow_seg_info_change` | §5.4.4 p60 / §6.4.4 p266 | absent | PARTIAL | `src/encoder/syntax/small_header/small_header.cpp:1359` (segment writer), `:773` | Only small_header rewrite. |
| `seg_info()`: `feature_enabled` (3 features × 8/16 segments) | §5.4.9 p66 / §6.4.9 p271 | absent | PARTIAL | `src/encoder/syntax/small_header/small_header.cpp:678-696` | Header-only gate enables `SEG_LVL_ALT_Q` but leaves `segmentation_enabled = 0` in the frame (`src/encoder/syntax/small_header/gate.cpp:310`); no block segmentation coding. |
| `seg_info()`: `feature_value` (`SEG_LVL_ALT_Q` su(10), skip/globalmv f(0)) | §5.4.9 p66 | absent | PARTIAL | `src/encoder/syntax/small_header/small_header.cpp:689-695` | No per-segment QP/skip/globalmv in any encoder tile path. |

## §5.4.5 Sequence intra config (p60) and §6.4.5 (p266)

| Field | §/page | Current value | Status | Evidence file:line | Gap |
|---|---|---|---|---|---|
| `enable_dip` | §5.4.5 p60 / §6.4.5 p266 | `put_bit(false)` | COMPONENT | `src/encoder/ibp/ibp.cpp:714-726` (full writer), `src/encoder/dip/dip_syntax.cpp` | DIP syntax/prediction exists, but no encoder header sets 1. `use_dip` is never signalled in a stream. |
| `enable_intra_edge_filter` | §5.4.5 p60 / §6.4.5 p266 | `put_bit(c.enable_intra_edge_filter)` = 1 (M1), 0 (M0) | WIRED | `src/encoder/m1/still_headers.cpp:115` | M1 default on; can be disabled by config/env (`src/encoder/m1/still_encoder.cpp:1811-1814`). |
| `enable_mrls` | §5.4.5 p60 / §6.4.5 p266 | `put_bit(false)` | WIRED | `src/encoder/m1/still_headers.cpp:116` | MRL only in the standalone MRL encoder/gate (`src/encoder/m1/mrl_stream/mrl_stream.cpp:572`); CLI header says 0. |
| `enable_cfl_intra` | §5.4.5 p60 / §6.4.5 p266 | `put_bit(c.enable_cfl_intra)` = 1 (M1), 0 (M0) | WIRED | `src/encoder/m1/still_headers.cpp:117` | CLI `--no-cfl` disables; M0 has no CfL. |
| `cfl_ds_filter_index` | §5.4.5 p60 / §6.4.5 p266 | `put_bits(0U, 2U)` | PARTIAL | `src/encoder/m1/still_headers.cpp:118` | Nonzero values only via the CFL down-sampler gate, which copies and patches the M1 writer to read `AV2ENC_CFL_DS_FILTER` (`src/encoder/cfl/ds_filters/gate/build_gate.sh:61-68`). CLI and all encoders pin 0; the 1/2/3 down-sampling filters are unreachable in production. |
| `enable_mhccp` | §5.4.5 p60 / §6.4.5 p266 | `put_bit(false)` | COMPONENT | `src/encoder/ibp/ibp.cpp:714-726` | Writer exists; no stream. MHCCP coding unreachable. |
| `enable_ibp` | §5.4.5 p60 / §6.4.5 p266 | `put_bit(false)` | COMPONENT | `src/encoder/ibp/ibp.cpp:714-726` | Writer exists; no stream. IBP coding unreachable. |

## §5.4.6 Sequence inter config (p60–p63) and §6.4.6 (p266–p269)

Single-picture branch (M0/M1/M10/HBD/lossless/sb128/sb256/chroma/mrl/tiles)
codes only the five bits below; everything else is inferred and therefore
unreachable in the CLI.

| Field | §/page | Current value | Status | Evidence file:line | Gap |
|---|---|---|---|---|---|
| `seq_enabled_motion_modes[INTERINTRA..EXTENDWARP]` | §5.4.6 p61 / §6.4.6 p266 | absent for single (all 0); M2 writes four `put_bit(false)` | COMPONENT | `src/encoder/m2/video_headers.cpp:150-152` | The advanced syntax writer can enable modes (`src/encoder/m2/advanced/advanced_syntax.cpp:1138-1159`), but no complete stream sets any. INTERINTRA, LOCALWARP, DELTAWARP, EXTENDWARP are unreachable. |
| `seq_frame_motion_modes_present_flag` | §5.4.6 p61 / §6.4.6 p267 | absent (0) | COMPONENT | `src/encoder/m2/advanced/advanced_syntax.cpp:1138-1159` | Depends on a motion mode being enabled; none is. |
| `enable_six_param_warp_delta` | §5.4.6 p61 / §6.4.6 p267 | absent (0) | COMPONENT | `src/encoder/m2/advanced/advanced_syntax.cpp:1138-1159` | Requires DELTAWARP; unreachable. |
| `enable_masked_compound` | §5.4.6 p61 / §6.4.6 p267 | `put_bit(false)` | WIRED | `src/encoder/m2/video_headers.cpp:154` | 1 only in the compound gate (`src/encoder/m2/multiref/gate/compound_stream.cpp:132`); M2/CLI write 0. |
| `enable_ref_frame_mvs` | §5.4.6 p61 / §6.4.6 p267 | `put_bit(false)` | COMPONENT | `src/encoder/m2/video_headers.cpp:155` | Temporal helper writes it (`src/encoder/inter/temporal/temporal_syntax.cpp:65-68`); no complete stream sets 1. TMVP is unreachable. |
| `reduced_ref_frame_mvs_mode` | §5.4.6 p61 / §6.4.6 p267 | absent (0) | COMPONENT | `src/encoder/inter/temporal/temporal_syntax.cpp:66-68` | Requires `enable_ref_frame_mvs = 1`; unreachable. |
| `order_hint_bits_minus_1` | §5.4.6 p61 / §6.4.6 p267 | absent for single (`OrderHintBits = 0`); M2 `put_bits(c.order_hint_bits - 1, 4)` = 3 | WIRED | `src/encoder/m2/video_headers.cpp:157` | 0 only for still picture (`still_headers.cpp:82`); non-single paths write 4..16-bit values. |
| `enable_refmvbank` | §5.4.6 p62 / §6.4.6 p267 | `put_bit(false)` | COMPONENT | `src/encoder/m2/video_headers.cpp:158` | Ref-MV bank prediction is implemented (`src/encoder/inter/temporal/ref_mv_bank.*`) and the temporal helper writes the bit (`temporal_syntax.cpp:69`), but no stream enables it. |
| `disable_drl_reorder` | §5.4.6 p62 / §6.4.6 p267 | `put_bit(true)` → `DRL_REORDER_DISABLED` | WIRED | `src/encoder/m2/video_headers.cpp:159` | DRL gate writes 0 and the constrain bit (`src/encoder/inter/drl_reorder/gate/drl_stream.cpp:136-139`); all three `DrlReorder` values reachable only there. Production is disabled. |
| `constrain_drl_reorder` | §5.4.6 p62 / §6.4.6 p267 | absent (disable = 1) | WIRED | `src/encoder/m2/video_headers.cpp:159` | Only the DRL gate (`src/encoder/inter/drl_reorder/gate/drl_stream.cpp:137-139`); `DRL_REORDER_CONSTRAINT`/`ALWAYS` unreachable elsewhere. |
| `seq_max_bvp_drl_bits_minus_1` | §5.4.6 p62 / §6.4.6 p268 | `put_ns(writer, 3U, 0U)` | ABSENT | `src/encoder/m1/still_headers.cpp:125` | Every writer pins 0 (1 BVP DRL index). |
| `allow_frame_max_bvp_drl_bits` | §5.4.6 p62 / §6.4.6 p268 | `put_bit(false)` | ABSENT | `src/encoder/m1/still_headers.cpp:126` | Never 1; frame-level `change_bvp_drl` unreachable. |
| `enable_bawp` | §5.4.6 p62 / §6.4.6 p268 | `put_bit(false)` | COMPONENT | `src/encoder/m2/video_headers.cpp:171` | BAWP is implemented (`advanced_syntax`, temporal) but no stream sets 1; the temporal helper writes it at `temporal_syntax.cpp:70`. |
| `enable_mv_traj` | §5.4.6 p62 / §6.4.6 p268 | absent (0) | COMPONENT | `src/encoder/m2/video_headers.cpp:170` | MV trajectory analysis is implemented but no stream enables it; the temporal helper writes it at `temporal_syntax.cpp:79`. |
| `enable_cwp` | §5.4.6 p62 / §6.4.6 p268 | `put_bit(false)` | WIRED | `src/encoder/m2/video_headers.cpp:172` | 1 only in the compound gate (`src/encoder/m2/multiref/gate/compound_stream.cpp:149`). |
| `enable_imp_msk_bld` | §5.4.6 p62 / §6.4.6 p268 | `put_bit(false)` | WIRED | `src/encoder/m2/video_headers.cpp:173` | 1 only in the compound gate (`src/encoder/m2/multiref/gate/compound_stream.cpp:150`). |
| `enable_df_sub_pu` | §5.4.6 p62 / §6.4.6 p268 | `put_bit(false)` | COMPONENT | `src/encoder/m2/video_headers.cpp:174` | No stream; the temporal helper writes it at `temporal_syntax.cpp:73`. |
| `enable_tip_explicit_qp` | §5.4.6 p63 / §6.4.6 p268 | absent (0) | COMPONENT | `src/encoder/inter/temporal/temporal_syntax.cpp:80-83` | Requires TIP output + df_sub_pu; unreachable. |
| `enable_opfl_refine` | §5.4.6 p63 / §6.4.6 p269 | `put_bits(0U, 2U)` = `REFINE_NONE` | WIRED | `src/encoder/m2/video_headers.cpp:175` | The M2 core pins `REFINE_NONE`; the CPU video path codes `REFINE_SWITCHABLE` for `--video-opfl` (G-OPFL-RMV), and `temporal_syntax.cpp:84` writes the field. |
| `enable_refinemv` | §5.4.6 p63 / §6.4.6 p269 | `put_bit(false)` | WIRED | `src/encoder/m2/video_headers.cpp:176` | The M2 core pins 0; the CPU video path sets 1 for `--video-refine-mv` (G-OPFL-RMV); `temporal_syntax.cpp:85` writes it. |
| `enable_tip_refinemv` | §5.4.6 p63 / §6.4.6 p269 | absent (0) | COMPONENT | `src/encoder/inter/temporal/temporal_syntax.cpp:86-89` | Requires TIP + OPFL/refine-MV; unreachable. |
| `enable_bru` | §5.4.6 p63 / §6.4.6 p269 | `put_bit(false)` | ABSENT | `src/encoder/m2/video_headers.cpp:177` | Every writer pins 0; no writer sets 1. Backwards reference update unreachable. |
| `enable_adaptive_mvd` | §5.4.6 p63 / §6.4.6 p269 | `put_bit(false)` | ABSENT | `src/encoder/m2/video_headers.cpp:178` | No writer sets 1. |
| `enable_mvd_sign_derive` | §5.4.6 p63 / §6.4.6 p269 | `put_bit(false)` | ABSENT | `src/encoder/m2/video_headers.cpp:179` | No writer sets 1; the IBC adapter explicitly refuses it (`src/encoder/m1/ibc_stream/ibc_syntax.cpp:511`). |
| `enable_flex_mvres` | §5.4.6 p63 / §6.4.6 p269 | `put_bit(false)` | ABSENT | `src/encoder/m2/video_headers.cpp:180` | Only a raw unit-test bit in `gme_test.cpp:178`; no stream. |
| `enable_global_motion` | §5.4.6 p63 / §6.4.6 p269 | `put_bit(false)` | WIRED | `src/encoder/m2/video_headers.cpp:181` | 1 only in the GME gate (`src/encoder/inter/gme/gme_gate.cpp:249`). |
| `enable_short_refresh_frame_flags` | §5.4.6 p63 / §6.4.6 p269 | `put_bit(false)` | PARTIAL | `src/encoder/m2/video_headers.cpp:182` | The longterm writer accepts 1 (`longterm_headers.cpp:287`), but the pool rejects it (`src/encoder/inter/longterm/longterm_pool.cpp:137-139`) and the frame-header path asserts 0 (`longterm_headers.cpp:396-397`). Compact refresh signalling is unreachable. |
| `explicit_ref_frame_map` | §5.4.6 p62 / §6.4.6 p267 | absent for single; M2 `put_bit(true)` | WIRED | `src/pipeline/video/video_headers.cpp` | The CPU `--video-implicit-ref-map` path (flat P, B frames, `--video-tlayers`) sets it to 0 and codes no map bits; the GPU core, the frame-type wrappers and the format cores keep 1. `longterm` also exposes 0. Gate: `src/pipeline/video/gate_implicit_ref_map.sh`. |
| `explicit_num_ref_frames` | §5.4.6 p62 / §6.4.6 p267 | absent for single; M2 `put_bit(true)` | PARTIAL | `src/encoder/inter/longterm/longterm_headers.cpp:264` | longterm can write 0 (`NumRefFrames = 8`), but M2/leading/tlayers/random-access pin 1. |
| `num_ref_frames_minus_1` | §5.4.6 p62 | absent for single (`NumRefFrames = 2`); M2 `put_bits(1, 4)` (2 refs) | WIRED | `src/encoder/m2/video_headers.cpp:162` | Configurable in M2/leading/longterm/random-access/tlayers. |
| `ActiveNumRefFrames` | §6.4.6 p267 | `Min(7, NumRefFrames) = 2` | WIRED | `src/encoder/m2/video_headers.cpp:162` | Correctly bounded by `NumRefFrames`; no writer emits >7. |
| `long_term_frame_id_bits` | §5.4.6 p62 / §6.4.6 p267 | absent for single (0); M2 `put_bits(0U, 3U)` | WIRED | `src/encoder/m2/video_headers.cpp:163` | Nonzero only in longterm (`src/encoder/inter/longterm/longterm_headers.cpp:270`) and random-access (`src/encoder/inter/random_access/random_access.cpp:321`). |
| `seq_max_drl_bits_minus_1` | §5.4.6 p62 / §6.4.6 p267 | absent for single; M2 `put_ns(5, 4)` | WIRED | `src/encoder/m2/video_headers.cpp:164` | Configurable in M2/leading/tlayers/longterm/random-access. |
| `allow_frame_max_drl_bits` | §5.4.6 p62 / §6.4.6 p268 | `put_bit(false)` | ABSENT | `src/encoder/m2/video_headers.cpp:165` | Never 1; frame-level `change_drl` unreachable. |
| `num_same_ref_compound` | §5.4.6 p62 / §6.4.6 p268 | `put_bits(0U, 2U)` | ABSENT | `src/encoder/m2/video_headers.cpp:168` | Ref-map logic models it (`src/encoder/m2/multiref/ref_map.cpp:43-59`) but no header writer emits >0. Same-reference compound prediction unreachable. |
| `enable_tip` | §5.4.6 p62 / §6.4.6 p268 | `put_bit(false)` | COMPONENT | `src/encoder/m2/video_headers.cpp:169` | TIP prediction is implemented but no complete stream enables it; the temporal helper writes it at `temporal_syntax.cpp:74`. |
| `disable_tip_output` / `EnableTipOutput` | §5.4.6 p62 / §6.4.6 p268 | absent (`EnableTipOutput = 0`) | COMPONENT | `src/encoder/inter/temporal/temporal_syntax.cpp:75-78` | Requires `enable_tip = 1`; unreachable. |
| `enable_tip_hole_fill` | §5.4.6 p62 / §6.4.6 p268 | absent | COMPONENT | `src/encoder/inter/temporal/temporal_syntax.cpp:75-78` | Requires `enable_tip = 1`. |

## §5.4.7 Sequence screen content config (p63–p64) and §6.4.7 (p270)

`single_picture_header_flag = 1` infers `seq_force_screen_content_tools =
SELECT_SCREEN_CONTENT_TOOLS` and `seq_force_integer_mv = SELECT_INTEGER_MV`, so
the M1/palette/IBC headers code **zero bits** here. The palette/IBC frame
headers then code `allow_screen_content_tools`/`force_integer_mv`.

| Field | §/page | Current value | Status | Evidence file:line | Gap |
|---|---|---|---|---|---|
| `seq_choose_screen_content_tools` | §5.4.7 p63 / §6.4.7 p270 | absent for single; M2 `put_bit(false)` | PARTIAL | `src/encoder/m2/video_headers.cpp:186` | All non-single writers pin 0; the palette writer's 1 path (`src/encoder/m1/palette_stream/palette_stream.cpp:80-89`) is never used by a stream. |
| `seq_force_screen_content_tools` | §5.4.7 p63 / §6.4.7 p270 | absent; inferred `SELECT_SCREEN_CONTENT_TOOLS` (2) | PARTIAL | `src/encoder/m1/palette_stream/palette_stream.cpp:75-78` | Under single picture the explicit field returns 0 bits (`:75-78`); non-single writers code 0. Only the inferred SELECT value is used; explicit 1 is unreachable. |
| `seq_choose_integer_mv` | §5.4.7 p63 / §6.4.7 p270 | absent for single | COMPONENT | `src/encoder/m1/palette_stream/palette_stream.cpp:86-89` | Writer exists only for a non-single palette stream that is never produced. |
| `seq_force_integer_mv` | §5.4.7 p64 / §6.4.7 p270 | absent; inferred `SELECT_INTEGER_MV` (2) | PARTIAL | `src/encoder/m1/palette_stream/palette_stream.cpp:75-78` | Only inferred SELECT; explicit 0/1 unreachable. |

## §5.4.8 Sequence transform/quant/entropy config (p64–p66) and §6.4.8 (p270–p271)

| Field | §/page | Current value | Status | Evidence file:line | Gap |
|---|---|---|---|---|---|
| `enable_fsc` | §5.4.8 p64 / §6.4.8 p270 | `put_bit(false)` | WIRED | `src/encoder/m1/still_headers.cpp:130` | Lossless reference encoder writes 1 for `FscMode::kAuto/kForce` (`src/encoder/lossless/lossless.cpp:2837-2838`); no lossy path uses FSC. |
| `enable_idtx_intra` | §5.4.8 p64 / §6.4.8 p270 | `put_bit(c.enable_idtx_intra)` = 0 (M1) | PARTIAL | `src/encoder/m1/still_headers.cpp:131` | Explicit 1 is never written. IDTX is reachable only by inference when `enable_fsc = 1` (`lossless.cpp:2838-2839`); M1 filters identity candidates out (`src/encoder/m1/still_encoder.cpp:831-848`). |
| `enable_intra_ist` | §5.4.8 p64 / §6.4.8 p270 | `put_bit(false)` | COMPONENT | `src/encoder/m1/still_headers.cpp:132` | IST writer exists (`src/encoder/tx_secondary/tx_secondary.cpp:457-459`); no stream. |
| `enable_inter_ist` | §5.4.8 p64 / §6.4.8 p270 | `put_bit(false)` | COMPONENT | `src/encoder/m1/still_headers.cpp:133` | Same. |
| `enable_chroma_dctonly` | §5.4.8 p64 / §6.4.8 p270 | `put_bit(c.enable_chroma_dctonly)` = 1 (M1/M2), 0 (M0/lossless) | WIRED | `src/encoder/m1/still_headers.cpp:134` | Both values reachable. |
| `enable_inter_ddt` | §5.4.8 p64 / §6.4.8 p270 | absent for single; M2 `put_bit(false)` | COMPONENT | `src/encoder/m2/video_headers.cpp:195` | DDT writer exists (`src/encoder/quant/dcdeltaq/syntax.cpp:292`); no stream sets 1. |
| `reduced_tx_part_set` | §5.4.8 p64 / §6.4.8 p270 | `put_bit(false)` = full set | WIRED | `src/encoder/m1/still_headers.cpp:135` | Full set is used; the reduced set (`1`) is untested/unreachable. |
| `enable_cctx` | §5.4.8 p64 / §6.4.8 p270 | `put_bit(false)` | COMPONENT | `src/encoder/m1/still_headers.cpp:136` | CCTX implementation exists (`src/encoder/tcq/cctx.cpp`); no writer sets 1. The component writer is at `src/encoder/quant/dcdeltaq/syntax.cpp:295`. |
| `enable_tcq` | §5.4.8 p65 / §6.4.8 p271 | `put_bit(false)` | COMPONENT | `src/encoder/m1/still_headers.cpp:137` | TCQ encoder + RD exists (`src/encoder/tcq/`) but no stream enables it; the component writer is at `src/encoder/quant/dcdeltaq/syntax.cpp:296`. |
| `choose_tcq_per_frame` | §5.4.8 p65 / §6.4.8 p271 | absent (0) | COMPONENT | `src/encoder/quant/dcdeltaq/syntax.cpp:297-299` | Requires `enable_tcq` + non-single; unreachable. |
| `enable_parity_hiding` | §5.4.8 p65 / §6.4.8 p271 | `put_bit(false)` | COMPONENT | `src/encoder/m1/still_headers.cpp:138` | Parity hiding is implemented (`src/encoder/tcq/parity/`) but no stream enables it; the component writer is at `src/encoder/quant/dcdeltaq/syntax.cpp:300-302`. |
| `enable_avg_cdf` | §5.4.8 p65 / §6.4.8 p271 | absent for single (inferred 1); M2 `put_bit(false)` | WIRED | `src/encoder/m2/video_headers.cpp:200` | Both values reachable; the non-single 1/1 gate exists (`src/encoder/entropy/avg_cdf/gate_dav2d_avg_cdf.cpp:202`). |
| `avg_cdf_type` | §5.4.8 p65 / §6.4.8 p271 | absent for single (inferred 1); M2 0 | PARTIAL | `src/encoder/m2/video_headers.cpp:200` | Only type 1 (“average across tiles”) is ever coded (`src/encoder/entropy/avg_cdf/gate_dav2d_avg_cdf.cpp:203`); type 0 (reference-CDF blending) is unreachable. |
| `separate_uv_delta_q` | §5.4.8 p65 / §6.4.8 p271 | `put_bit(false)` | PARTIAL | `src/encoder/quant/dcdeltaq/syntax.cpp:189-191` | dcdeltaq gate can splice 1 into a real stream (`src/encoder/quant/dcdeltaq/dcdeltaq_gate.cpp:28-50`); no encoder-owned header. |
| `equal_ac_dc_q` | §5.4.8 p65 / §6.4.8 p271 | `put_bit(true)` | PARTIAL | `src/encoder/quant/dcdeltaq/syntax.cpp:193` | dcdeltaq gate can rewrite to 0; all encoder headers pin 1, which forces `Base{YDc,UVDc} = 0` and omits both DC base fields. |
| `base_y_dc_delta_q` | §5.4.8 p65 / §6.4.8 p271 | absent (`equal_ac_dc_q = 1`) | PARTIAL | `src/encoder/quant/dcdeltaq/syntax.cpp:194-198` | Only dcdeltaq splice; no encoder can signal a nonzero Y-DC base delta. |
| `y_dc_delta_q_enabled` | §5.4.8 p65 / §6.4.8 p271 | absent (0) | PARTIAL | `src/encoder/quant/dcdeltaq/syntax.cpp:198` | Same. |
| `base_uv_dc_delta_q` | §5.4.8 p65 / §6.4.8 p271 | absent (`equal_ac_dc_q = 1`) | PARTIAL | `src/encoder/quant/dcdeltaq/syntax.cpp:200-205` | Same. |
| `uv_dc_delta_q_enabled` | §5.4.8 p65 / §6.4.8 p271 | absent (0) | PARTIAL | `src/encoder/quant/dcdeltaq/syntax.cpp:205` | Same. |
| `base_uv_ac_delta_q` | §5.4.8 p65 / §6.4.8 p271 | `put_bits(23U, 5U)` → `BaseUVAcDeltaQ = 0` | PARTIAL | `src/encoder/m1/still_headers.cpp:141` | All current streams pin 23 (the chroma writer exposes the 5-bit config but its CLI never changes the 23 default); dcdeltaq splice can set other values. |
| `uv_ac_delta_q_enabled` | §5.4.8 p65 / §6.4.8 p271 | `put_bit(false)` | PARTIAL | `src/encoder/m1/still_headers.cpp:142` | Only dcdeltaq splice; frame-level UV AC deltas otherwise impossible. |

## §5.4.10 Sequence filter config (p66–p67) and §6.4.10 (p271–p272)

| Field | §/page | Current value | Status | Evidence file:line | Gap |
|---|---|---|---|---|---|
| `disable_loopfilters_across_tiles` | §5.4.10 p66 / §6.4.10 p271 | `put_bit(false)` | WIRED | `src/encoder/m1/still_headers.cpp:145` | The tiles encoder writes 1 by default (`src/encoder/m1/tiles_stream/tiles_stream.cpp:167`); CLI one-tile writes 0. |
| `enable_cdef` | §5.4.10 p66 / §6.4.10 p272 | `put_bit(c.cdef.enable)` | WIRED | `src/encoder/m1/still_headers.cpp:146` | CLI `--still-cdef` enables; frame `cdef_params()` is written. |
| `enable_gdf` | §5.4.10 p66 / §6.4.10 p272 | `put_bit(false)` | ABSENT | `src/encoder/m1/filters_stream/filters_stream.cpp:317-319` | The only filter-config writer explicitly rejects GDF; no GDF implementation exists. |
| `gdf_unit_matches_sb_size` | §5.4.10 p66 / §6.4.10 p272 | absent | ABSENT | `src/encoder/m1/filters_stream/filters_stream.cpp:317-319` | Requires `enable_gdf = 1`; unreachable. |
| `enable_restoration` | §5.4.10 p66 / §6.4.10 p272 | `put_bit(false)` | WIRED | `src/encoder/m1/still_headers.cpp:148` | Only the LR-chroma gate enables it (`src/encoder/filters/lr_chroma/gate/lr_chroma_gate.cpp:484`); CLI/M1 always 0. |
| `lr_tools_disable[0][RESTORE_PC_WIENER]` | §5.4.10 p66 / §6.4.10 p272 | absent (restoration off) | WIRED | `src/encoder/filters/lr_chroma/gate/lr_chroma_gate.cpp:486` | Written only with restoration on; the gate pins 1. |
| `lr_tools_disable[0][RESTORE_WIENER_NONSEP]` | §5.4.10 p67 / §6.4.10 p272 | absent | WIRED | `src/encoder/filters/lr_chroma/gate/lr_chroma_gate.cpp:487` | Gate pins 0 (Wiener-NS enabled). |
| `lr_tools_disable[1][RESTORE_PC_WIENER]` | §5.4.10 p67 | absent; inferred 1 | WIRED | `src/encoder/m1/filters_stream/filters_stream.cpp:334` | Spec fixes this to 1; no bit is coded. |
| `lr_tools_uv_present` | §5.4.10 p67 / §6.4.10 p272 | absent | WIRED | `src/encoder/filters/lr_chroma/gate/lr_chroma_gate.cpp:485` | Gate writes 1. |
| `lr_tools_disable[1][RESTORE_WIENER_NONSEP]` | §5.4.10 p67 / §6.4.10 p272 | absent | WIRED | `src/encoder/filters/lr_chroma/gate/lr_chroma_gate.cpp:488` | Gate writes 0; chroma Wiener-NS enabled. |
| `enable_ccso` | §5.4.10 p67 / §6.4.10 p272 | `put_bit(c.ccso.enable)` | WIRED | `src/encoder/m1/still_headers.cpp:150` | CLI `--still-ccso` enables; frame `ccso_params()` is written. |
| `ccso_unit_matches_sb_size` | §5.4.10 p67 / §6.4.10 p272 | `put_bit(true)` when CCSO on | PARTIAL | `src/encoder/m1/still_headers.cpp:151-153` | 0 (tile-alignment-based size) never written; M1 only supports 64×64 units. |
| `cdef_on_skip_txfm_always_on` | §5.4.10 p67 / §6.4.10 p272 | absent for single; M2 `put_bit(false)` | COMPONENT | `src/encoder/m2/video_headers.cpp:212` | The filters component can write it (`src/encoder/m1/filters_stream/filters_stream.cpp:344-349`); no stream selects ALWAYS_ON. |
| `cdef_on_skip_txfm_disabled` | §5.4.10 p67 / §6.4.10 p272 | absent for single (`CdefOnSkipTxfm = ADAPTIVE`); M2 `put_bit(true)` | PARTIAL | `src/encoder/m2/video_headers.cpp:212-213` | Single-picture ADAPTIVE and M2 DISABLED are reachable; ALWAYS_ON is not. |
| `df_par_bits_minus_2` | §5.4.10 p67 / §6.4.10 p272 | `put_bits(0U, 2U)` | PARTIAL | `src/encoder/m1/still_headers.cpp:154` | Nonzero only via small_header rewrite (`src/encoder/syntax/small_header/small_header.cpp:582`). |

## §5.4.11 `user_defined_qm()` (p67–p69) and §6.4.11 (p273)

This syntax is **not called from `sequence_header_obu()`**; its only caller is
`quantizer_matrix_obu()` (§5.13). It is included because §5.4.11 is in the
requested page range.

| Field | §/page | Current value | Status | Evidence file:line | Gap |
|---|---|---|---|---|---|
| `qm_copy_from_previous_plane` | §5.4.11 p67 / §6.4.11 p273 | not emitted | PARTIAL | `src/encoder/qm/qm.cpp:110-114` | QM OBU writer exists; only the QM splice gate uses it. No encoder emits a QM OBU or `using_qmatrix = 1` (all writers pin `put_bit(false)`). |
| `qm_8x8_is_symmetric` | §5.4.11 p68 / §6.4.11 p273 | not emitted | PARTIAL | `src/encoder/qm/qm.cpp:118-121` | Same. |
| `qm_4x8_is_transpose_of_8x4` | §5.4.11 p68 / §6.4.11 p273 | not emitted | PARTIAL | `src/encoder/qm/qm.cpp:122-127` | Same. |
| `quant_delta` / `UserQm` coefficients | §5.4.11 p68 / §6.4.11 p273 | not emitted | PARTIAL | `src/encoder/qm/qm.cpp:95-190`, `:376-407` | QM gate bit-splices a QM OBU and the `using_qmatrix` bit into an M1 stream (`src/encoder/qm/qm_gate.cpp:208-226`, `:316-323`); pinned `dav2d` rejects user QMs, so the gate uses `avmdec`. |

## §5.4.12 `timing_info()` (p69) and §6.4.12 (p273–p274)

`timing_info()` is not called by `sequence_header_obu()` at all. Its only
caller is `content_interpretation_obu()` (§5.15, p85–p86), which is where the
HRD component writes it. The sequence-header-adjacent display-delay fields are
in §5.4.1 above.

| Field | §/page | Current value | Status | Evidence file:line | Gap |
|---|---|---|---|---|---|
| `num_units_in_display_tick` | §5.4.12 p69 / §6.4.12 p273 | not in any sequence header | PARTIAL | `src/encoder/rc/hrd/hrd.cpp:430` | HRD writes it into a content-interpretation OBU; no encoder-owned header or CI OBU. |
| `time_scale` | §5.4.12 p69 / §6.4.12 p273 | not written | PARTIAL | `src/encoder/rc/hrd/hrd.cpp:431` | Same. |
| `equal_picture_interval` | §5.4.12 p69 / §6.4.12 p273 | not written | PARTIAL | `src/encoder/rc/hrd/hrd.cpp:432` | Same. |
| `num_ticks_per_picture_minus_1` | §5.4.12 p69 / §6.4.12 p273 | not written | PARTIAL | `src/encoder/rc/hrd/hrd.cpp:434-436` | Same. |
| `DecCT = num_units_in_decoding_tick / time_scale` | §6.4.1 p262 | undefined (no timing) | PARTIAL | `src/encoder/rc/hrd/hrd_buffer.cpp:87-133` | Derived only in the HRD component. |
| `DispCT = num_units_in_display_tick / time_scale` | §6.4.12 p273 | undefined | PARTIAL | `src/encoder/rc/hrd/hrd.cpp:430-431` | Same. |

## §5.4.13 `seq_decoder_model_info()` (p69) and §6.4.13 (p274)

| Field | §/page | Current value | Status | Evidence file:line | Gap |
|---|---|---|---|---|---|
| `decoder_buffer_delay` | §5.4.13 p69 / §6.4.13 p274 | absent | COMPONENT | `src/encoder/rc/hrd/hrd.cpp:441-447` | Written only by HRD; not integrated, and the pinned decoder rejects the block. |
| `encoder_buffer_delay` | §5.4.13 p69 / §6.4.13 p274 | absent | COMPONENT | `src/encoder/rc/hrd/hrd.cpp:445-446` | Same. |
| `low_delay_mode_flag` | §5.4.13 p69 / §6.4.13 p274 | absent | COMPONENT | `src/encoder/rc/hrd/hrd.cpp:447` | Same. |

## Derived variables (§6.4)

| Variable | §/page | Current value | Status | Evidence file:line | Gap |
|---|---|---|---|---|---|
| `Monochrome` | §6.4.1 p260 | 0 (4:2:0) | WIRED | `src/encoder/format/chroma/chroma_format.cpp:43` | 1 only in the chroma encoder (4:0:0). |
| `BitDepth` / `MaxQ` | §6.4.1 p260 | 8 / 255 | WIRED | `src/encoder/format/format.cpp:97-127` | 10 / 303 in M10/HBD; 12 cannot be signalled. |
| `NumPlanes`, `SubsamplingX/Y` | §6.4.1 p259-260 | 3 / 1,1 | WIRED | `src/encoder/format/chroma/chroma_format.cpp:39-44` | All four formats supported only in the chroma encoder. |
| `SeqMaxMlayerCnt` | §6.4.1 p261 | 1 | PARTIAL | `src/encoder/inter/tlayers/tlayers_syntax.cpp:119` | Always 1; no writer can signal >1. |
| `OrderHintBits` | §6.4.6 p267 | 0 (single) / 4 (M2) | WIRED | `src/encoder/m2/video_headers.cpp:157` | Configurable 4..16 in non-single paths. |
| `NumRefFrames` / `ActiveNumRefFrames` | §6.4.6 p267 | 2 / 2 (single); 2 (M2) | WIRED | `src/encoder/m2/video_headers.cpp:162` | `NumRefFrames` configurable in M2/leading/longterm/random-access/tlayers; `Active = Min(7, NumRefFrames)`. |
| `MaxPbAspectRatio` | §6.4.3 p265 | 8 | PARTIAL | `src/encoder/m1/still_headers.cpp:107` | 2/4 only via small_header rewrite (`src/encoder/syntax/small_header/small_header.cpp:514`). |
| `MaxSegments` | §6.4.4 p265 | 8 | PARTIAL | `src/encoder/m1/still_headers.cpp:110` | 16 only via small_header rewrite. |
| `EnableTipOutput` | §6.4.6 p268 | 0 | ABSENT | `src/encoder/m2/video_headers.cpp:169` (`enable_tip = 0`) | TIP output unreachable; temporal helper can write it but no stream. |
| `CdefOnSkipTxfm` | §6.4.10 p272 | ADAPTIVE (single) / DISABLED (M2) | PARTIAL | `src/encoder/m2/video_headers.cpp:212-213` | ALWAYS_ON reachable only in the filters component (`src/encoder/m1/filters_stream/filters_stream.cpp:344-349`). |
| `DrlReorder` | §6.4.6 p267 | `DRL_REORDER_DISABLED` | WIRED | `src/encoder/inter/drl_reorder/gate/drl_stream.cpp:136-139` | All three values reachable only in the DRL gate. |
| `BaseYDcDeltaQ`, `BaseUVDcDeltaQ`, `BaseUVAcDeltaQ` | §6.4.8 p271 | 0, 0, 0 | PARTIAL | `src/encoder/m1/still_headers.cpp:140-142` | Nonzero only via dcdeltaq splice; `equal_ac_dc_q = 1` forces the DC bases to 0. |
| `y_dc_delta_q_enabled`, `uv_dc_delta_q_enabled`, `uv_ac_delta_q_enabled` | §6.4.8 p271 | 0, 0, 0 | PARTIAL | `src/encoder/m1/still_headers.cpp:142` | Only dcdeltaq splice. |
| `SeqSbRows/Cols`, `SeqTileRows/Cols`, `SeqTileRowsLog2/ColsLog2`, `SeqUniformTileSpacingFlag`, `sbShift` | §6.4.2 p264 / §5.18.7.3 | one tile, uniform, `sbShift = 4` | PARTIAL | `src/encoder/m1/still_headers.cpp:66-73`, `:157` | Multi-tile only in tiles encoder; non-uniform only at frame level in small_header; sequence-level tile config only rewrite. |
| `MLayerPresenceMap` | §5.4.1 p58 / §6.4.1 p264 | diagonal only | PARTIAL | `src/encoder/tiles/svc_layers.cpp:107-126` | No writer serializes embedded-layer dependencies; SVC only validates plans. |
| `TLayerDependencyMap` | §5.4.1 p57 / §6.4.1 p264 | default lower-triangular | WIRED | `src/encoder/inter/tlayers/tlayers_syntax.cpp:146-152` | Custom map only in tlayers. |
| `MLayerDependencyMap` | §5.4.1 p57 / §6.4.1 p264 | default lower-triangular | ABSENT | `src/encoder/inter/longterm/longterm_headers.cpp:214-217` | Never serialized; no writer emits the mlayer block. |
| `lr_tools_disable[1][RESTORE_PC_WIENER] = 1` | §5.4.10 p67 | 1 | WIRED | `src/encoder/m1/filters_stream/filters_stream.cpp:334` | Spec-fixed inference; no bit is coded. |

---

## Conformance bugs found during the audit (fix before enabling the field)

1. **`chroma_headers` can write `seq_tile_info_present_flag = 1` without
   `allow_tile_info_change`/`tile_params`.** `src/encoder/format/chroma/chroma_headers.cpp:473`
   writes only the flag; the validator does not reject it. Any config that sets
   the flag produces a malformed sequence header.
2. **`longterm` serializes `max_tlayer_id`/`max_mlayer_id` without the
   required dependent syntax.** `src/encoder/inter/longterm/longterm_headers.cpp:213-217`
   emits neither `seq_max_mlayer_cnt_minus_1`, `mlayer_dependency_present_flag`,
   nor `tlayer_dependency_present_flag` when those maxima are >0.
3. **Most non-single writers emit `seq_tier` unconditionally.** M2, leading,
   tlayers, and longterm write the bit even when `seq_level_idx <= 3`, where
   §5.4.1 omits it. `random_access.cpp:262` is the only correct guard.
4. **SVC LCR association is dangling.** `svc_config` plans `seq_lcr_id = 1` and
   an LCR whose `global_config_record_id = 1`
   (`src/encoder/svc/svc_config.cpp:153`, `:245`), but `svc_framing` re-wraps
   the original sequence payloads unchanged
   (`src/encoder/svc/svc_framing.cpp:730-742`), so the sequence headers still
   carry `seq_lcr_id = 0`.
5. **`seq_decoder_model_info()` is not accepted by the pinned decoder.** The
   HRD gate documents the fallback: `decoder_model_info_present_flag = 1` and
   `num_units_in_decoding_tick` are accepted, the nested
   `seq_decoder_model_info()` is not
   (`src/encoder/rc/hrd/hrd_stream_gate.cpp:411-417`, `725-733`). Either the
   decoder or the spec reading needs resolution before this syntax is claimed.
6. **`seq_tile_info_present_flag` in chroma is not validated.** The writer
   accepts `true` but cannot serialize the sub-structure; the check is missing
   in `validate_chroma_seq_config()`.


## Honorable mentions

Outside the top 10 by coding impact, but real coverage holes:
`seq_lcr_id`/`max_mlayer_id`/`seq_max_mlayer_cnt_minus_1` are never serialized,
so SVC's LCR/embedded-layer plan is not actually referenced by a sequence
header; `film_grain_params_present` can be set but no film-grain OBU exists;
`explicit_ref_frame_map` is 1 in every exercised stream (only longterm exposes
0, and it is not gate-exercised), so implicit mapping is effectively
unreachable; `num_same_ref_compound` is always 0; `enable_bru`,
`enable_adaptive_mvd`, `enable_mvd_sign_derive`, and `enable_flex_mvres` have
no writer at all.

## Counts

Status counts over the 173 audited sequence-header/derived rows:

| Status | Count |
|---|---|
| WIRED | 55 |
| PARTIAL | 67 |
| COMPONENT | 32 |
| ABSENT | 19 |
| DECODER-ONLY | 0 |
| OUT-OF-SCOPE | 0 |
| **Total** | **173** |

Breakdown by clause:

| Clause | Rows | WIRED | PARTIAL | COMPONENT | ABSENT |
|---|---|---|---|---|---|
| §5.4.1 general | 34 | 16 | 13 | 0 | 5 |
| §5.4.2 tiles | 9 | 2 | 6 | 1 | 0 |
| §5.4.3 partition | 8 | 2 | 4 | 0 | 2 |
| §5.4.4/§5.4.9 segments | 6 | 0 | 6 | 0 | 0 |
| §5.4.5 intra | 7 | 3 | 1 | 3 | 0 |
| §5.4.6 inter | 38 | 11 | 3 | 16 | 8 |
| §5.4.7 SCC | 4 | 0 | 3 | 1 | 0 |
| §5.4.8 TQE | 21 | 4 | 10 | 7 | 0 |
| §5.4.10 filters | 15 | 9 | 3 | 1 | 2 |
| §5.4.11 QM | 4 | 0 | 4 | 0 | 0 |
| §5.4.12 timing | 6 | 0 | 6 | 0 | 0 |
| §5.4.13 decoder model | 3 | 0 | 0 | 3 | 0 |
| §6.4 derived | 18 | 8 | 8 | 0 | 2 |

Note: counts are by table row, not by syntax element; e.g. the three
`seg_info()` features are covered by two rows and the four crop offsets by one
row.

## Top 10 gaps by coding impact

1. **All temporal/inter coding tools are off in the production CLI.**
   `seq_enabled_motion_modes = 0`, `enable_ref_frame_mvs = 0`,
   `enable_tip = 0`, `enable_bawp = 0`, `enable_opfl_refine = REFINE_NONE`,
   `enable_refmvbank = 0`, `enable_mv_traj = 0`. The CLI is intra-only; the
   M2-family standalone gates (M2, leading, random-access, longterm, tlayers)
   have P/B frames but disable every advanced inter tool. TIP, OBMC/warp
   modes, TMVP, ref-MV bank, BAWP, OPFL, and refine-MV are unreachable in any
   encoder-owned stream; DRL reordering, masked compound, CWP, implicit mask
   blending, and GME are reachable only in narrow standalone gates. This is the
   single largest rate/quality gap.
2. **SDP and extended partitions are unreachable.**
   `enable_sdp = 0` in every stream-producing path; `enable_extended_sdp` therefore absent;
   `enable_ext_partitions = 0` and `enable_uneven_4way_partitions` inferred 0
   everywhere. The SDP and partition_ext components exist, but only a Python
   gate patch can turn the header bits on
   (`src/encoder/m1/partition_stream/gate_minimal.cpp:10-16`). Semi-decoupled
   chroma trees, HORZ_3/VERT_3, and uneven 4-way splits cannot be produced.
3. **The transform toolset is DCT-centric.**
   `enable_intra_ist = 0`, `enable_inter_ist = 0`, `enable_inter_ddt = 0`,
   `enable_cctx = 0`, `enable_tcq = 0`, `enable_parity_hiding = 0`,
   `enable_idtx_intra = 0` (explicit). IST, DDT, CCTX, TCQ, and parity hiding
   are implemented as components but no stream enables them; explicit identity
   transforms are reachable only through FSC in the lossless encoder. Only
   DCT/DCT-only plus the non-identity TX-type search is usable in the main
   encoder.
4. **Intra prediction is missing DIP, MHCCP, IBP, and non-default CfL
   down-sampling.** `enable_dip/mhccp/ibp = 0` everywhere; `cfl_ds_filter_index`
   is 0 in every encoder (nonzero only via the CFL gate env). MRLS is reachable
   only in the standalone MRL encoder. This leaves DC/directional/smooth/CfL as
   the only production intra modes.
5. **Sequence-level segmentation is header-only.**
   `enable_ext_seg = 0`, `seq_seg_info_present_flag = 0`,
   `seq_allow_seg_info_change` absent in every encoder. The small_header gate
   writes `seg_info()` but leaves the frame's `segmentation_enabled = 0`, so
   segment-adaptive QP, skip, and global-MV coding are not actually exercised.
6. **Per-plane delta-Q cannot be signalled.**
   Every encoder pins `equal_ac_dc_q = 1`, `separate_uv_delta_q = 0`,
   `base_uv_ac_delta_q = 23` (→ 0), and both UV enables 0. Y-DC/UV-DC base
   deltas and `diff_uv_delta` are impossible without the dcdeltaq header
   splice. This blocks chroma-specific rate/quality tuning and QP-0 lossless
   interactions.
7. **Tiles are frame-level, uniform, and mostly single-tile.**
   `seq_tile_info_present_flag = 0`, `allow_tile_info_change` absent, and
   `uniform_tile_spacing_flag = 1`. Multi-tile exists only in the standalone
   tiles encoder; non-uniform starts only in small_header; the CLI is one tile.
   This blocks tile-parallel encoding/decoding and tile-size control.
8. **Loop-filter coverage is incomplete.**
   `enable_gdf = 0` with no implementation; `gdf_unit_matches_sb_size` absent;
   `enable_restoration` only in the LR-chroma gate; `cdef_on_skip_txfm_always_on`
   never selected; `ccso_unit_matches_sb_size` fixed 1. CDEF and CCSO are
   available in the CLI, but GDF, production restoration, and the ALWAYS_ON
   CDEF-skip mode are not.
9. **Screen-content and QM signalling are inferred/spliced.**
   The SCC sequence fields are never coded with a tool-enabling value (single
   picture infers SELECT; non-single M2 pins 0), and `using_qmatrix = 0` in
   every encoder; user QMs require a QM-OBU bit-splice. Palette/IBC streams
   work, but a non-single screen-content sequence cannot signal its tool
   choices, and perceptual quantization cannot be produced by the encoder.
10. **Timing/decoder-model and level/tier conformance are thin.**
    No production header writes `timing_info()`, `seq_initial_display_delay`,
    or decoder-model info; the HRD component does, but pinned `dav2d f9dafbc`
    rejects the spec-v1.0.0 `seq_decoder_model_info()` layout. Level is pinned
    to 7 (or 31 in lossless), `seq_tier` is always 0 (high tier unreachable),
    and M0's tile assertions are hardcoded to level-5.1 constants. This prevents
    conformant HRD/bitrate signalling and validated operation at other levels.


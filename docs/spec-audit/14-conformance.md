# Spec audit 14 — decoder conformance, alpha/resampling, timing/DFG, decoder model, levels

Scope: AV2 v1.0.0 Annex A.2–A.5, Annex B, Annex C, Annex D, Annex E
(spec pages **1101–1138**). Annex F (p1139+) is informative extraction and is
mentioned only where it constrains an encoder. Normative text read from
`spec/AV2_Spec_v1.0.0.pdf` (page index = spec page − 1, pypdf).
Code read from `src/bitstream/**`, `src/encoder/m2/**`, `src/encoder/m1/**`,
`src/encoder/tiles/**`, `src/encoder/format/**`, `src/encoder/rc/hrd/**`,
`src/encoder/rc/twopass/**`, `src/encoder/svc/**`, `src/main.cpp`.

Question answered per row: **does the encoder enforce this constraint on the
bytes it emits?** Not "is there a helper somewhere".

## Legend

| Status | Meaning |
|---|---|
| **WIRED** | The check runs on a path whose output is written to the bitstream (or the constraint is satisfied structurally by the syntax the writer emits). |
| **PARTIAL** | A check or implementation exists, but it is incomplete, advisory, only in a debug `assert` (compiled out by `-DNDEBUG`), or **not on the emission path** (standalone component/gate). |
| **ABSENT** | No check or no implementation anywhere in `src/`. |

"The encoder" is not one path. Four emit bytes today, and they have different
guards — this decides most statuses:

| Path | Header writer | Level guards |
|---|---|---|
| `av2enc` CLI, M0 still | `src/bitstream/basic_headers.cpp` | none (one `assert` on tile area) |
| `av2enc --still`, M1 still (also the M0 geometry fallback) | `src/encoder/m1/still_headers.cpp` | none |
| M1 tiled gate / pipeline planner | `src/encoder/m1/tiles_stream/*` | tile geometry only, via `tiles::plan_tile_layout()` |
| M2/M3 multi-frame gates | `src/encoder/m2/video_headers.cpp` | none |

Default level everywhere is `seq_level_idx = 7` (level 5.1):
`src/bitstream/basic_headers.h:74`, `src/encoder/m1/still_headers.h:61`,
`src/encoder/m2/video_headers.h:24`, `src/encoder/pipeline/pipeline.h:173`,
`src/encoder/m1/tiles_stream/tiles_stream.h:87`. The only exception is the
lossless still reference, which defaults to 31 (`src/encoder/lossless/lossless.h:152`).

Level 5.1 (Table A.8/A.9, p1105–1106): `MaxPicSize = 8 912 896`,
`MaxHSize = MaxVSize = 4 975`, `MaxDisplayRate = 534 773 760`,
`MaxDecodeRate = 547 430 400`, `MaxTiles = 64`, `MaxTileCols = 8`,
`MaxHeaderRate = 300`, `MainMbps = 40`, `HighMbps = 160`, tier 0.
`MaxBitrate = MaxMbps × 1 000 000` and `MaxBufferSize = MaxBitrate × 1 s`
(p1111), i.e. 40 Mbit Main / 160 Mbit High at level 5.1.

---

## 1. Profiles (Annex A.2/A.3, p1101–1104)

| Constraint | §/page | Status | Evidence | Consequence of violation |
|---|---|---|---|---|
| `seq_profile_idc` must match `chroma_format_idc`: 0/1/2 ⇒ 4:0:0 or 4:2:0; 3 ⇒ +4:2:2; 4 ⇒ +4:4:4 (Table A.1) | A.2 / p1101–1102 | **WIRED** | `src/encoder/format/format.cpp:330-344` derives 0/3/4 from the format; `src/encoder/format/chroma/chroma_format.cpp:47-58` repeats it per `ChromaFormatInfo`; the writer emits the derived value (`chroma_headers.cpp:337`) rather than a caller field. M0/M1/M2 hardcode `0` with `chroma_format_idc = 0` (`basic_headers.cpp:177-181`, `video_headers.cpp:97-102`). | Decoder rejects the sequence header; the profile claim is false (A.2 "shall"). |
| `bit_depth_idc` ∈ {0,1} (10-bit / 8-bit); values > 1 are reserved | A.2 Table A.1, §6.4.1 / p1101–1102 | **WIRED** | `format.cpp:322-326` returns `kReservedBitDepth` for 12-bit ("bit_depth_idc > 1 is reserved"); `chroma_headers.cpp:547-548` rejects readers with idc > 1; 10-bit writers assert `idc == 0` (`hbd_headers.cpp:89`, `high_bitdepth_headers.cpp:89`). | Reserved header value; decoder desync. |
| `ProfileScalingFactor`, `PicSizeProfileFactor`, `BitrateProfileFactor` (Table A.2) must multiply the level and decoder-model numbers | A.2 / p1102, used at p1108–1111, p1125 | **ABSENT** | No factor table exists in `src/`; grep for `PicSizeProfileFactor`/`BitrateProfileFactor` returns nothing. `BufferSize` is taken from RC/VBV state, not `MaxBufferSize × BitrateProfileFactor` (`src/encoder/rc/hrd/hrd_buffer.h:84-86`). | Level limits are applied with profile factor 1.0 even for profiles 3/4, under-counting by 1.667×/2.5×. |
| Interoperability-point layer counts (Table A.3) and their mandatory MSDO/LCR mix (Table A.4) | A.3 / p1102–1103 | **ABSENT** | `svc` writes LCR/OPS (`svc_framing.cpp:444-470`) but `validate_layer_config_record()` (`tiles/svc_layers.h:137`) checks structure only; grep for `Interop`, `IP0`, `IP1`, `max_interop` shows a 4-bit descriptor range check (`svc_framing.cpp:460`) and no Table A.3/A.4 rule. | A 2-embedded-layer stream claiming IOP 0 (profile 0) is non-conforming; missing MSDO where IOP 2 requires it. |
| Multi-sequence configuration id (`lcr_config_idc`) must map to Table A.5/A.6 allowed syntax values | A.3 / p1104 | **PARTIAL** | `svc_framing.cpp:460-464` bounds `config_idc ≤ 63`, `level_idx ≤ 31`, `max_interop ≤ 15`; nothing ties a config id to the chroma/bit-depth values of the layers it describes. | Decoder derives aggregate constraints the bitstream does not actually meet. |
| Configurable profile (31): constraints come from `chroma_format_idc`, `bit_depth_idc`, `SeqMaxMlayerCnt` | A.2 / p1102 | **ABSENT** | No path emits profile 31 except lossless level 31 (`lossless.cpp:2802` writes profile 0); no profile-31 constraint derivation exists. | Only reachable if a caller hand-writes a header; no guard. |

---

## 2. Levels (Annex A.4/A.5, p1104–1111)

These are "requirement of bitstream conformance" (p1110). The **only** level
table in the tree is the tile scaling factor pair
(`src/encoder/tiles/tile_layout.cpp:100-127`); there is no
`MaxPicSize`/rate/bitrate/header-rate table anywhere.

| Constraint | §/page | Status | Evidence | Consequence of violation |
|---|---|---|---|---|
| `FrameWidth * FrameHeight <= MaxPicSize` | A.4 / p1110 | **ABSENT** | grep `MaxPicSize` over `src/`, `tools/`, `spec/tables/` → no hit. `main.cpp:496` accepts up to 65535×65535 (only even dimensions are checked, `main.cpp:525`). | Any frame above the envelope is non-conforming while still claiming level 5.1. DCI-4K `4096x2304 = 9 437 184 > 8 912 896` already violates; so do 5K and 8K. |
| `FrameWidth <= MaxHSize`, `FrameHeight <= MaxVSize` | A.4 / p1110 | **ABSENT** | Same. No width/height cap other than the `uint16` IVF limit. | `5120 > 4975` at level 5.1. |
| `FrameWidth >= 16`, `FrameHeight >= 16` | A.4 / p1110 | **ABSENT** | `pipeline_config_valid()` checks only `> 0` and evenness (`pipeline.cpp:338-357`); `main.cpp:496-530` checks only evenness. | A sub-16-luma-sample frame passes every encoder-side check; `FrameWidth >= 16` is a level requirement. |
| `NumTiles <= MaxTiles` (64 at 5.1) | A.4 / p1110 | **ABSENT** | `tile_layout.cpp:295-298` caps the grid at `MAX_TILE_COLS/ROWS = 64` (syntax cap, `tile_layout.h:34-36`), never at the level's `MaxTiles`. Verification: presets top out at 16×2 = 32. | 64×64 grid (4096 tiles) plans successfully at level 5.1. |
| `TileCols <= MaxTileCols` (8 at 5.1) | A.4 / p1110 | **ABSENT** | `max_log2_tile_cols = tile_log2(1, min(sb_cols, kMaxTileColumns))` with `kMaxTileColumns = 64` (`tile_layout.cpp:295-296`). **Product default for ≥3 GPUs is `k16x2`** (`pipeline.cpp:297-302`), i.e. 16 columns; `pipeline_test.cpp:277-290` accepts it at level 5.1 with 32 tiles. | First 4-GPU 1080p/4K stream violates level 5.1 `MaxTileCols = 8`; needs level 6.0 (16) or a different preset. |
| `TileWidth <= Tile_Width_Scaling_Factor[seq_tier][level] * MAX_TILE_WIDTH / 4` | A.4 / p1110 | **PARTIAL** | Enforced in the planner: `tile_layout.cpp:282-283` (`max_tile_width_sb`) and `:397-400`. The **M0/M1 single-tile writers never call the planner** (`still_headers.cpp:64-75` writes one tile with no bound; `basic_headers.cpp:79-95`). | A single-tile 8K M1 stream has `TileWidth = 7680 > 4096` and is accepted. |
| `TileWidth >= 64` when `RightMostTile == 0` | A.4 / p1110 | **ABSENT** | No lower bound on tile width in `plan_tile_layout()` or `validate_tile_layout()` (`tile_layout.cpp:227-440` planner, `:444-619` validator). | Many narrow columns on a small frame (e.g. 16 columns over 640 px ⇒ 40 px) pass. |
| `TileWidth * TileHeight <= Tile_Area_Scaling_Factor * 4096 * 2304 / 4` | A.4 / p1110 | **PARTIAL** | Planner: `tile_layout.cpp:284-286`, `:401-405`. Still writers: `assert(...)` only — `basic_headers.cpp:88`, `high_bitdepth_headers.cpp:74`; **no check at all in the M1 writer**. Release build is `-DNDEBUG` (`build/CMakeCache.txt:53`), so the assert is dead in the shipping binary. | 8000×8000 single tile = 15 625 SB > 2304 SB; emitted silently. |
| `TotalDisplayLumaSampleRate <= MaxDisplayRate` | A.4 / p1110 | **ABSENT** | No `TotalDisplayLumaSampleRate` computation; no per-temporal-unit accounting in any writer. | 4K120 at `995 328 000 > 534 773 760` still claims 5.1. |
| `NumFrameHeadersPerSec <= MaxNumFrameHeadersPerSec` (`MaxHeaderRate*(1+tier*2)` = 300 at 5.1/tier0) | A.4 / p1110 (derivation p1109) | **ABSENT** | No header-rate model. `CountFrameHeaderForLevelConstraint` is not implemented anywhere. | High-fps / show-existing-heavy streams exceed 300 headers/s. |
| `NumRefFrames <= MaxLevelRefFrames` | A.4 / p1109–1110 | **ABSENT** | `MaxLevelRefFrames` absent. `VideoConfig::num_ref_frames` is `1..16` via `assert` (`video_headers.cpp:93`). At 4K level 5.1 `MaxLevelRefFrames = 8` (DecodeCount 1). | `num_ref_frames = 16` at 4K → DPB exceeds the level's frame-buffer budget. |
| `LumaSampleCount <= FrameParsingTime * MaxDecodeRate` | A.4 / p1110 | **ABSENT** | `FrameParsingTime` needs `Removal[i+1] - Removal[i]` (BRT), which is never emitted; no decode-rate model. | Over-subscribed decode rate; a conformant level-5.1 decoder may not keep up. |
| `NumTiles <= Min(MaxTiles, Max(1, MaxTiles*120*FrameParsingTime))` | A.4 / p1110 | **ABSENT** | Same dependency on `FrameParsingTime`. | Insufficient tile parallelism above 4K60 (NOTE, p1111). |
| `CompressedSize <= Min((LumaSampleCount*PicSizeProfileFactor>>3)*1.25, (FrameParsingTime*MaxDecodeRate*PicSizeProfileFactor>>3)/MinCompBasis)` | A.4 / p1110 | **ABSENT** | No `CompressedSize` accounting; per-frame OBU byte totals are only printed by gates. | Burst frame exceeds the level's per-frame byte budget. |
| `FrameSymbolCount <= FrameParsingTime*MaxDecodeRate*PicSizeProfileFactor*(8/(9*MinCompBasis)+1/48)` | A.4 / p1110 | **ABSENT** | `FrameSymbolCount` (accumulated in `read_symbol`) has no encoder counterpart. | Symbol rate above what the level guarantees. |
| `MaxTileSizeInLumaSamples * NumFrameHeadersPerSec <= ASF*547 430 400/4` | A.4 / p1110 | **ABSENT** | No implementation. Vacuous only for levels ≤ 5.1 (NOTE, p1111); binding above that. | Above level 5.1, the parallelism guarantee is broken. |
| `MaxBitrate = MaxMbps × 1e6`, `MaxBufferSize = MaxBitrate × 1 s`, and the stream must stay inside them | A.4 / p1110–1111 | **ABSENT** | Rate control is CQP/CRF/VBR with an internal leaky bucket that explicitly "is not an HRD conformance implementation" and "does not claim decoder HRD conformance" (`rc/twopass/twopass.h:30-36`). The HRD planner takes `bit_rate`/`buffer_size_bits` from the caller (`hrd_buffer.h:84-86`), never from the level table. | A 4K stream at QP 90 can exceed 40 Mbit/s (Main, level 5.1) with no diagnostic. |
| `seq_tier = 1` may only be signalled for level ≥ 4.0 | A.4 NOTE / p1106 | **ABSENT** | `pipeline_config_valid()` accepts any `seq_tier` 0/1 (`pipeline.cpp:355-357`); `level_tile_limits()` returns tier-1 factors for every level (`tile_layout.cpp:120-125`). The M2 writer hardcodes `seq_tier = 0` (`video_headers.cpp:100`) and the still writers omit it (`single_picture_header_flag = 1` removes it, p1104), yet the planner is fed `seq_tier` (`pipeline.cpp:157-160`). | Plans are computed against a tier the emitted stream cannot express. |
| `seq_level_idx` 22–30 are Reserved; 31 = maximum parameters | A.4 Table A.7 / p1104–1105 | **PARTIAL** | Rejected for *planning*: `tile_layout.cpp:112-115` sets `reserved` for `22..30`. Header writers accept anything 0..31: `basic_headers.cpp:173` (`assert <= 31`), `pipeline.cpp:350-353`, `video_headers.cpp:99` (no check). | A reserved level value is emitted; decoder behaviour undefined. |

---

## 3. Alpha, spatial mapping and resampling (Annex D, p1115–1121)

Annex D is **informative** ("implementing it according to the description is
not mandatory for a conformant decoder", p1115), so D.2–D.5.2 impose no
encoder obligation. The *syntax-level* rules that would bind an encoder that
emits ATS/atlas payloads are in §5.9/§6.9 (p79, p292–293). The encoder emits
no atlas syntax: `docs/spec-notes.md:55` lists §5.5–5.17 (MSDO, LCR, atlas,
OPS, BRT) as "deliberately untouched", and no writer references `ats_*`.

| Constraint | §/page | Status | Evidence | Consequence of violation |
|---|---|---|---|---|
| Last atlas segment must not be an alpha segment (`ats_msi_alpha_segment_flag` inferred 0 for `i == num_atlas_segments_minus_1`) | §5.9.4 / p79; NOTE D.1 / p1116 | **ABSENT** (structurally vacuous) | No ATS writer. The syntax itself only codes the flag when `i != ats_msi_num_atlas_segments_minus_1` (`p79` loop guard), so the rule cannot be violated by a conforming writer. | None today; re-check if atlas support lands. |
| `ats_num_atlas_segments_minus_1 < MAX_NUM_ATLAS_SEGMENTS`; all `ats_input_stream_id` values unique | §6.9.6 / p293 | **ABSENT** | No `ats_`/`atlas` implementation in `src/` (only the `lcr_*_atlas_id_present_flag = 0` bits in `svc_framing.cpp:452-456`). | If added without checks: duplicate layer ids / over-count would be non-conforming. |
| D.2 chroma format determination (`subX`, `subY` from the composite's 4:2:0/4:2:2/4:4:4) | D.2 / p1116 | **ABSENT** | Not implemented. | n/a (decoder/composition side). |
| D.3 array initialization + D.3.1 background colour (BT.709 8-bit D′R/D′G/D′B, defaults 16) | D.3 / p1117 | **ABSENT** | Not implemented. Encoder writes no `ats_msi_background_*` (inferred 16). | n/a; encoder default matches the inference. |
| D.4 spatial mapping (resample to `ats_msi_segment_width/height`, clip at composite edge) | D.4 / p1118 | **ABSENT** | Not implemented. | Segments would need encoder-side placement metadata that is never written. |
| D.5 spatial mapping with alpha (per-sample `(alphaMax-alpha)*dst + alpha*src`, `Round2(temp, bitdepthAlpha)`, out-of-bounds alpha = `alphaMax`) | D.5 / p1119–1120 | **ABSENT** | Not implemented; no alpha plane path in the encoder (`docs/bitrate-features.md:161` lists "explicit and derived alpha" as unverified/optional). | Alpha-composited output cannot be produced or signalled. |
| D.5.1 frame resampling and D.5.2 monochrome (alpha) resampling | D.5.1–D.5.2 / p1120–1121 | **ABSENT** | Both sections are explicit placeholders ("outside the scope of this annex", p1120–1121). The only "resample" in the tree is the DIP 8×8→w×h step (`src/encoder/dip/dip.h:167`), unrelated to Annex D. | n/a. |

Conclusion for this section: **no encoder-binding alpha/resampling constraint
is exercised at all**, which is correct for the current single-stream product
but means any future atlas/alpha delivery feature starts from zero.

---

## 4. Frame timing, DFG and the decoder model (Annex E, p1122–1136)

Baseline: the emission path writes **no timing at all**.
`decoder_model_info_present_flag = 0` and
`seq_initial_display_delay_present_flag = 0`
(`src/encoder/m2/video_headers.cpp:126-127`; M0 has the same block absent,
`basic_headers.cpp:203`), no Content Interpretation OBU, no
`buffer_removal_timing_obu()`, `num_units_in_decoding_tick` never written.
A complete, tested HRD component exists (`src/encoder/rc/hrd/`) but its README
states it "is not wired into the root `CMakeLists.txt`, `src/main.cpp`, the
M1/M2 header writers or the frame loop"
(`src/encoder/rc/hrd/README.md:3-6`; grep confirms no caller outside its own
gate/tests, and `CMakeLists.txt` contains no `hrd` target).

| Constraint | §/page | Status | Evidence | Consequence of violation |
|---|---|---|---|---|
| Either operating mode needs `ci_timing_info_present_flag = 1` and the decoder-model flags; without them level conformance **cannot be verified** | E.4.1/E.4.2/E.4.4 / p1126–1128 | **ABSENT** | All flags are 0; no CI OBU is emitted by any writer (`build_content_interpretation_obu()` only called from `hrd_stream_gate.cpp`). E.4.4: "it is impossible to verify whether the bitstream satisfies the level constraints according to either of the decoder models." | Every emitted stream is unverifiable against its claimed level. Not a decode failure — a conformance-claim failure. |
| `seq_initial_display_delay_minus_1` / `ops_initial_display_delay_minus_1` and `InitialPresentationDelay` (E.5.7) | E.5.7 / p1131 | **ABSENT** | Flag 0 at `video_headers.cpp:126`; `SequenceTimingConfig` exists (`hrd.h:115-129`) but is unreachable. | The decoder must guess its startup delay; the level's presentation schedule is unspecified. |
| DFG = all OBUs from the end of the last OBU of the previous `ShowExistingFrame == 0` frame through the last OBU of the current one (metadata/CI/BRT included) | E.3 / p1125–1126 | **PARTIAL** | No DFG accounting in any writer. `hrd_stream_gate.cpp` iterates BRT size to a fixed point because "the BRT OBU itself is part of the DFG" (README p8); `validate_temporal_unit_order()` checks ordering only inside the unwired component. | If timing is ever enabled, a mis-grouped metadata/CI/BRT OBU shifts `CodedBits[i]` and `br_time` for that frame and every later RAP segment. |
| `FirstBitArrival[i] = max(LastBitArrival[i-1], LatestArrivalTime[i])`; `LatestArrivalTime[i] = ScheduledRemoval[i] - (EBD+DBD)/90000`; `LastBitArrival = First + CodedBits/BitRate` | E.5.1/E.5.2 / p1128–1129 | **PARTIAL** | Implemented exactly in `hrd_buffer.cpp:360-365` (integer units of 1/(90000·time_scale)); never called from an encoder path. | Wrong arrival model ⇒ wrong `br_time` ⇒ underflow/overflow at the decoder. |
| `ScheduledRemoval[i] >= LastBitArrival[i]` (DFG available at scheduled removal; strict mode) | E.7.1/E.7.4 / p1136–1137 | **PARTIAL** | `hrd_buffer.cpp:366-376` fails the plan in strict mode; low-delay mode defers removal to the first DecCT tick after the DFG is complete (`:376`). Unwired. | Decoder underflow. |
| `ScheduledRemovalTiming[i] = ScheduledRemovalTiming[PrevRap] + BufferRemovalTime[i]*DecCT` | E.5.4 / p1129–1130 | **PARTIAL** | `hrd_buffer.cpp:325-358`; `br_time` is coded `rg(4)`, so ≤ 511 ticks, with an explicit error otherwise (`:347-355`). Unwired. | Wrong removal schedule after a RAP. |
| `DecoderBufferDelay <= ceil(TimeDelta[i])` at each RAP where DBD is signalled | E.7.2 / p1137 | **PARTIAL** | `hrd_buffer.cpp:336-340` pushes the RAP later by `dbd_units` when needed. Unwired. | Non-conforming across random access points. |
| Smoothing buffer shall never overflow (`occupancy <= BufferSize`) | E.7.3 / p1137 | **PARTIAL** | Occupancy sweep at `hrd_buffer.cpp:449-517`; fails when peak > `buffer_size_bits`. `buffer_size_bits` comes from VBV config, **not** `MaxBufferSize × BitrateProfileFactor`. Unwired. | Buffer overflow; the decoder drops data. |
| Smoothing buffer shall never underflow when `LowDelayMode == 0` | E.7.4 / p1137 | **PARTIAL** | `hrd_buffer.cpp:366-373` (strict mode only). Unwired; the M2 stream sets no `low_delay_mode_flag` at all. | Underflow / late decode. |
| `ScheduledRemoval[i+1] - Removal[i] >= Max(TimeToDecode[i], 1/MaxNumFrameHeadersPerSec)` | E.7.5 / p1137 | **PARTIAL** | Only a proxy: at least one DecCT tick between actual removals (`hrd_buffer.cpp:378-384`). `TimeToDecode` is **not** modelled — `hrd_buffer.h:36-37`: "Presentation times, frame-buffer reuse and TimeToDecode (E.5.6/E.5.7, E.6) are not modelled". | Back-to-back frames with no decode margin; a level-conformant decoder may miss the deadline. |
| `TimeToDecode[i] = lumaSamples / MaxDecodeRate` (intra = W×H, inter = max frame, `DecodeCount = 2` for global-intrabc + in-loop filtering) | E.5.6 / p1131 | **ABSENT** | No `MaxDecodeRate` anywhere in `src/`; the `DecodeCount` reserve (and the `MaxLevelRefFrames - 1` note, p1110) is unimplemented. | Decode-time model is wrong by up to 2× for global-intrabc + loop-filter streams. |
| `PresentationInterval[j] >= Max(W*H*numOutputFramesInTU[j]/MaxDisplayRate, MinFrameTime)` with `MinFrameTime = MaxDecodeRate/(MaxNumFrameHeadersPerSec*MaxDisplayRate)` | E.7.6 / p1138 | **ABSENT** | No presentation-time or temporal-unit model; `numOutputFramesInTU` unimplemented. | Variable-rate / multi-layer streams can schedule two output frames in the same display slot. |
| `Removal[i] + TimeToDecode[i] <= PresentationTime[i]` (decode deadline) | E.7.7 / p1138 | **ABSENT** | Not modelled (`hrd_buffer.h:36-37`); corresponds to `DISPLAY_FRAME_LATE`, which the planner never evaluates. | Frame decoded too late to be displayed. |
| `DecoderBufferDelay != 0` and `<= 90000 * (BufferSize / BitRate)` in decoding-schedule mode | E.7.8 / p1138 | **PARTIAL** | `hrd_buffer.cpp:272-276` rejects `dbd == 0`; `:265-270` and `:292-296` reject `(DBD+EBD)*BitRate > BufferSize`, which implies the bound. Unwired. | Delay/rate pair that cannot fit the buffer. |
| `PresentationTime[j]` non-decreasing after each RAP (output order) | E.7.1 / p1136 | **ABSENT** | No check. Relevant now that B-frame gates exist (`src/encoder/inter/bframes/`); M2's `immediate_output_frame = 1` (`video_headers.cpp:236`) keeps the simple path monotone by construction. | Non-monotone presentation ⇒ conformant-decoder output order is undefined. |
| A suffix starting at a random access point must itself be decoder-model conformant (CLK; OLK + leading frames removed; RAS with LTRs present) | E.7.1 / p1136–1137 | **PARTIAL** | Only decodability is tested: `leading_gate.cpp:331-346` emits `.rap.ivf` / `.rap-no-leading.ivf` suffixed streams and `ras_gate.cpp` exists, but both only ask `dav2d` to decode them — **no buffer model is run on the suffix**. The planner requires `frames[0]` to be a RAP (`hrd_buffer.cpp:180-182`) but has no suffix test. | A stream that is conformant in full but not from its RAPs is not a conformant AV2 bitstream. |
| The decoder-model process must not raise `DECODE_FRAME_BUF_UNAVAILABLE` / `DECODE_EXISTING_FRAME_BUF_EMPTY` / `DISPLAY_FRAME_LATE` | E.6/E.7.9 / p1134–1138 | **ABSENT** | No frame-buffer/`BufferPool`/`VBI` simulation; the planner only covers the smoothing buffer. | Delivered bitstream can require more DPB slots than the level grants (see `MaxLevelRefFrames`). |
| Operating-mode selection: decoding schedule when BRT is present, else resource availability | E.4.3 / p1128 | **PARTIAL** | Only schedule mode is implemented (`hrd_buffer.h:38-39`); resource-availability `time_next_buffer_is_free()` (p1130–1131) is absent. | If BRT is ever omitted while timing is present, the model that would prove conformance does not exist. |

---

## 5. Encoder consequences / recommendations (Annex B–C, p1112–1114)

| Constraint | §/page | Status | Evidence | Consequence of violation |
|---|---|---|---|---|
| Length-delimited packing: `leb128(num_bytes_in_obu)` outside `open_bitstream_unit()` | Annex B.2 / p1112 | **WIRED** | `basic_headers.cpp` `append_leb128`/`make_obu`; `hrd.cpp` `build_*_obu()`; IVF length prefixes written by `src/io/ivf*`. | Decoder cannot delimit OBUs. |
| Processable-frame conditions: `primary_ref_frame == PRIMARY_REF_NONE` or a processed ref; `use_ref_frame_mvs == 0` or all refs processed; `RefOrderHint` written before use; `found_ref == 1` ⇒ processed; `frame_to_show_map_idx` processed | C.2/C.4 / p1113–1114 | **PARTIAL** (satisfied by construction, not checked) | `video_headers.cpp:247-252` (intra ⇒ `PRIMARY_REF_NONE`, `signal_primary_ref_frame = 0`), `:266` (`use_ref_frame_mvs = 0`), `:244` (`frame_size_override_flag = 0` ⇒ explicit size, no `found_ref`); keyframe `order_hint = 0`. No validator asserts these invariants. | Starting mid-stream (non-key) is undefined; C.2 conditions are an encoder responsibility. |
| Encoder must make a non-key-frame starting point processable ("the encoder needs to be careful") | C.4 / p1114 | **PARTIAL** | Same construction as above; `leading`/`ras` gates decode from a non-key point, but no check that every emitted frame satisfies C.2. | Output "corrupt" at a non-key start point even though the stream decodes from frame 0. |
| Decoder supports starting where the first TU has a sequence header and all frames are processable (recommendation) | C.3/C.5 / p1114 | **PARTIAL** | M2/M3 gates always start with TD+SEQ+key frame; no test covers a start at a later TU. | Untested error-resilience contract. |

---

## 6. Counts

| Status | Rows |
|---|---|
| **WIRED** | 3 |
| **PARTIAL** | 18 |
| **ABSENT** | 33 |
| **Total** | 54 |

Per annex:

| Area | WIRED | PARTIAL | ABSENT | Total |
|---|---|---|---|---|
| A.2/A.3 profiles | 2 | 1 | 3 | 6 |
| A.4 levels (+A.5) | 0 | 3 | 16 | 19 |
| Annex D alpha/resampling | 0 | 0 | 7 | 7 |
| Annex E timing/DFG/model/conformance | 0 | 11 | 7 | 18 |
| Annex B/C framing + recommendations | 1 | 3 | 0 | 4 |

The distribution is the headline: the encoder is conformant on **syntax
validity** (profile/chroma/bit-depth, tile geometry when the planner is used,
OBU framing) and has essentially **no coverage of the quantitative
conformance envelope** (size, rate, bitrate, header rate, ref frames) or of
the **decoder model** (nothing is signalled, so nothing can be verified).

---

## 7. Constraints most likely violated by current streams

Ordered by "will happen / is reachable today", with the exact trigger:

1. **`TileCols <= MaxTileCols` (level 5.1 ⇒ 8) on the ≥3-GPU path.**
   `default_layout_for_gpus()` returns `k16x2` for `gpu_count >= 3`
   (`pipeline.cpp:297-302`), the planner accepts 16 columns at level 5.1
   (`pipeline_test.cpp:277-290`, 32 tiles), and the bench plan documents
   16×2 as the default for 4 GPUs at 1080p and 4K
   (`bench/multi_gpu_plan/README.md:20,54,57`). The first 4-GPU stream is
   non-conforming unless the level is raised to 6.0+ or the preset changes.
   This is the highest-confidence violation and it is in the product plan,
   not a corner case.

2. **`MaxPicSize` / `MaxHSize` / `MaxVSize` for anything above ~4K.**
   Level 5.1 is hardcoded in every writer and `main.cpp:496` only enforces the
   `uint16` container limit. `4096x2304` (DCI 4K) is already over
   `MaxPicSize`; 5K is over both `MaxPicSize` and `MaxHSize`. Nothing in the
   CLI, the M0 path, the M1 path, or the M2 path notices.

3. **No decoder model at all ⇒ level conformance unverifiable for every
   stream.** This is systemic rather than a numeric violation, but E.4.4 makes
   it dispositive: without `ci_timing_info_present_flag` and
   `decoder_model_info_present_flag`, the claimed level cannot be checked by
   either operating mode. Today's streams (64×64 and 640×360 reference
   bitstreams, the M2/M3 gates, `video_gate.ivf`) are all in this state.

4. **Bitrate (`MaxMbps`) and buffer size (`MaxBufferSize`) unchecked.**
   At level 5.1 the Main-tier ceiling is 40 Mbit/s and the buffer is 40 Mbit.
   Rate control targets a caller-supplied bitrate with an internal leaky
   bucket that explicitly disclaims HRD conformance
   (`twopass.h:30-36`), and `BufferSize` is taken from VBV config rather than
   the level. A 4K stream at the default QP 90 can cross 40 Mbit/s silently.

5. **Single-tile tile-area / tile-width for large M1 stills.** The M1 writer
   emits one tile with no bound (`still_headers.cpp:64-75`); the only guard in
   M0/M10 is an `assert` that `-DNDEBUG` removes (`basic_headers.cpp:88`,
   `high_bitdepth_headers.cpp:74`, `build/CMakeCache.txt:53`). An 8K still therefore
   violates both `TileWidth <= 4096` and the tile-area bound while claiming
   level 5.1.

6. **`NumRefFrames <= MaxLevelRefFrames` at 4K.** `num_ref_frames` is only
   `assert`-bounded to 16 (`video_headers.cpp:93`); at 4K level 5.1 the level
   grants 8. Any 9–16-reference experiment violates the level.

7. **`seq_tier` consistency.** `pipeline_config_valid()` accepts `seq_tier = 1`
   (`pipeline.cpp:355-357`) and the planner then uses high-tier tile limits,
   while the still writers do not code `seq_tier` at all
   (`single_picture_header_flag = 1`). Plans and emitted stream can disagree
   about which tier's limits apply — low severity today because the tier-0 and
   tier-1 scaling factors are identical for levels ≤ 6.3 in width and ≤ 6.3 in
   area, but it is a latent mismatch.

Not at risk today: alpha/resampling (Annex D) and the atlas syntax rules —
nothing emits those, so those rows are ABSENT-but-dormant, not violations.

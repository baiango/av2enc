# Spec audit 11 — §7.21–§7.23: output, film grain, motion-field storage, reference update

Scope: AV2 v1.0.0 specification pages **560–579** (`§7.21`–`§7.23`), plus the
syntax clauses those processes consume (`§5.14`, `§5.18.10`, `§6.13`,
`§6.17.10`). Read-only audit of `/Users/arin/work/av2enc` at commit `ae7ffd8`.

Spec identity: `spec/AV2_Spec_v1.0.0.pdf`, AV2 Bitstream & Decoding Process
Specification, v1.0.0, 2026-05-28, commit `38f28e7`. Page numbers below are
physical PDF pages (`pypdf` page index = page − 1).

Status legend:

| Status | Meaning |
|---|---|
| **WIRED** | Production encode path (`src/main.cpp` → M0/M1, or the M2 frame writer) emits the syntax and uses the result in a decision. |
| **COMPONENT** | Standalone module emits/uses the syntax correctly and is tested (and usually `dav2d`-gated), but is not reachable from the production path. |
| **PARTIAL** | Some of the required syntax/logic exists, but a required piece is missing or only the trivial case is expressible. |
| **ABSENT** | No encoder-side implementation. |
| **DECODER-ONLY** | The normative decode-side process is implemented as a reference/oracle; the encoder cannot emit or use it. |

Production-path fact used throughout: `src/main.cpp` encodes **one still
picture** only (`Usage: av2enc <input.y4m> -o <output.ivf> ...`, `src/main.cpp:122`),
and `m2::encode_m2_sequence()` (`src/encoder/m2/m2_frame.cpp:929`) is only called
by gate/test binaries. The M2 stream is IPPP with `num_ref_frames = 2`,
`order_hint_bits = 4`, `monotonic_output_order_flag = 1`
(`src/encoder/m2/m2_headers.cpp:110`), one reference slot (0), and no
reordering.

---

## 1. §7.21 output process

| Process | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| Output process (`output_process`, `mixedOutput`, dispatch to grain) | §7.21.1 p.560–561 | **PARTIAL** | Production header writes `immediate_output_frame = 1` on every frame and, because `monotonic_output_order_flag = 1`, infers `implicit_output_frame = 0` (`src/encoder/m2/m2_headers.cpp:110,236-237`); `leading.cpp:250-268` and `tlayers_syntax.cpp:262-266` implement the general §5.18.2 form. SEF writer implements the alternate branch: `refresh_frame_flags = 0`, `immediate_output_frame = 1`, `frame_to_show_map_idx`, derive/explicit hint (`src/encoder/inter/sef/sef.cpp:190-207`, `sef.h:33-47`). `mixedOutput` semantics documented/tested in `sef.h:33-47` and `src/encoder/inter/sef/README.md` ("Output order and buffer retention"). | No encoder implementation of the output process itself (no OutY/OutU/OutV assembly, no crop, no film-grain dispatch). Production streams are monotonic, so only the trivial "output current frame immediately" case exists. `mixedOutput` is never exercised with a hint different from the reference's own `OrderHint` (the writer forces derive = 0 + same hint for the dav2d rule). |
| Intermediate output preparation (copy from `FrameStore` / current `LrFrame`, crop, `filmGrainPresent`) | §7.21.2 p.561–562 | **PARTIAL** | Every sequence writer hard-codes `seq_cropping_window_present_flag = 0`: `src/encoder/m1/m1_headers.cpp:100`, `m2_headers.cpp:125`, `hbd/hbd_headers.cpp:115`, `sb128/sb128_stream.cpp:452`, `sb256/sb256_stream.cpp:415`, `inter/leading/leading.cpp:113`, `inter/random_access/random_access.cpp:284`, `inter/tlayers/tlayers_syntax.cpp:135`. `filmGrainPresent` exists only in the standalone synthesis model (`filters/spec_tables/film_grain.h:44-75`), not in any header writer. | No crop-window signalling (`RefCropWidth/Height/Left/Top` can never be non-trivial), no `RefFilmGrainPresent`/`RefNumPlanes` slot bookkeeping. The encoder can express only "crop = frame, no grain". |
| Output successive frames | §7.21.3 p.562 | **COMPONENT** | `OutputSim::output_successive()` walks `hint+1, hint+2, …` over eligible slots (`src/encoder/inter/bframes/bframes.cpp:350-368`), called from `output_slot()`/`frame_buffers()`. Covered by `bframes_test` (11,893 checks per README) including 128-frame display-order replay (`bframes_test.cpp:639-642`). | Only in the B-frame planner simulation. Not called by SEF, leading, tlayers or M2. Single embedded layer only. No `RefOutputOrder`/layer term in `output_ordering()` (§7.21.3 p.564). |
| Output implicit output frame | §7.21.4 p.562–563 | **COMPONENT** | `OutputSim::eligible()` implements `RefImplicitOutputFrame && RefValid && !shown && RefOrderHint != RESTRICTED_OH` (`bframes.cpp:326-334`); slot metadata carries `implicit`/`immediate` (`bframes.cpp:303-312,756-761`). `tlayers` writes `implicit_output_frame = 1` for reordered frames with `monotonic = 0` (`tlayers_syntax.cpp:262-266`) and its gate decodes a reordered stream whose output must match display order (`tlayers_gate.cpp:542-560`). | Not usable from M2: M2 forces `monotonic_output_order_flag = 1`, which makes `implicit_output_frame` inferred 0. No production reordering path. |
| Flush implicit output frames | §7.21.5 p.563–564 | **PARTIAL** | `OutputSim::flush()` repeatedly emits the lowest-hint eligible slot (`bframes.cpp:420-437`), called once at the end of `simulate_decode()` (`bframes.cpp:922`). | The `olkLimit` input is not modelled: the spec calls `flush_implicit_output_frames(0)` at CVS start and `(1)` on the first regular picture after an OLK, excluding `RefOrderHint >= OlkTUOrderHint` (`§5.18.2 p.95`; `§7.21.5 p.564`). No flush at random-access boundaries in any module. |
| Output frame buffers | §7.21.6 p.564–565 | **COMPONENT** | `OutputSim::frame_buffers()` outputs lower-hint eligible frames first, then `ref_idx` (or the current frame when −1), then successive hints (`bframes.cpp:383-418`); invoked after every §7.23 refresh and for the current frame (`bframes.cpp:737-766`). SEF output path tested in `sef_gate` phases A/B/C (`sef_gate.cpp:1-20`) against pinned `dav2d`. | Component only. No `output_ordering(i)` multi-layer term (`RefOrderHint * (max_mlayer_id + 1) + RefMLayerId`); no `olkLimit`; no output process in production. |

## 2. §7.21.7 film grain synthesis

Film grain is a **decode-side output process**. AV2 v1.0.0 signals it through a
separate `OBU_FILM_GRAIN` (`§5.14 p.84-85`, `fgm_update_flags f(8)`,
`fgm_chroma_idc`, `film_grain_model()`), and the frame header carries
`film_grain_config()`: `apply_grain f(1)`, `fgm_id f(3)`, `grain_seed f(16)`
(`§5.18.10.1 p.141`). `film_grain_params_present` is a sequence-header bit
(`§5.4.1`). The encoder currently sets that bit to 0 everywhere.

| Process | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| Film grain general (RNG init, `GrainMin/Max`, call sequence) | §7.21.7.1 p.565 | **DECODER-ONLY** | `ApplyFilmGrain()` (`filters/spec_tables/film_grain.cpp:531-554`) sets `RandomRegister = grain_seed`, `GrainMin/Max`, then calls generate/scaling/add-noise. `film_grain_test: 611 checks passed` (run during this audit). | No encoder call site; `film_grain.cpp` is not in the root `CMakeLists.txt` (`spec_tables/README.md`: "Nothing here is wired into … `src/main.cpp`"). |
| Random number process | §7.21.7.2 p.565 | **DECODER-ONLY** | `RandomRegister::GetRandomNumber()` (`film_grain.cpp:16-40`) implements the 16-bit LFSR exactly. | Algorithm complete; decodable only. |
| Generate grain (white noise + AR templates, luma and chroma) | §7.21.7.3 p.566–567 | **DECODER-ONLY** | `GenerateGrain()` (`film_grain.cpp:57-162`): 82×73 luma, 44×38 / 82×73 chroma by `subX/subY`, `GrainMin/Max` clip, `ar_coeff_lag 0..3` raster template, chroma seeded `0xb524`/`0x49d8`, chroma AR includes the averaged luma-grain term with `Round2(luma, subX+subY)`. Randoms from the recovered `Gaussian_Sequence` table (`film_grain_tables.h`, pinned by SHA-256). | No model estimation/fitting, no syntax writer. |
| Scaling lookup initialization | §7.21.7.4 p.568 | **DECODER-ONLY** | `BuildScalingLut()` (`film_grain.cpp:165-218`): 256-entry piecewise-linear LUT, plane selection including `chroma_scaling_from_luma`, `get_x`/`get_y`, 16.16 delta interpolation. | Complete; unused by any encoder. |
| Add noise synthesis (stripes, overlap_flag, 16/32 blocks, chroma-from-luma, scaling, restricted range) | §7.21.7.5 p.569–572 | **DECODER-ONLY** | `AddNoise()` (`film_grain.cpp:221-475`): `film_grain_block_size ? 32 : 16`, horizontal overlap (27/17 and 17/27 for luma, 23/22 for chroma), vertical stripe blend, `scale_lut()` with `ScalingShift = grain_scaling_minus_8+8`, `chroma_scaling_from_luma` merged luma, `clip_to_restricted_range` min/max and `fg_mc_identity`. Exercised for 8/10-bit 4:0:0/4:2:0/4:2:2/4:4:4, seeds 0/1/0xffff, overlap on/off, 16/32 blocks (`film_grain_test.cpp:378-562`; 611 checks pass). | Complete synthesis oracle, but output-only and unreachable from a stream. The optional 2-line-buffer implementation noted on p.570 is not done (not required). |
| Film-grain syntax emission (model + config) | §5.14 p.84; §5.18.10 p.141; §6.13 p.302; §6.17.10 p.346 | **ABSENT** | `film_grain_params_present` is hard-coded 0 in every writer: `m0_headers.cpp:268`, `m1_headers.cpp:159`, `m2_headers.cpp:219`, `hbd/hbd_headers.cpp:171`, `sb128:506`, `sb256:471`, `leading.cpp:204`, `random_access.cpp:376`; `pipeline/tiled_still_gate.cpp:200`. `ObuType` has only `{sequence_header, temporal_delimiter, closed_loop_key, regular_tile_group}` (`src/bitstream/m0_headers.h:26-31`) — no `OBU_FILM_GRAIN`. No writer anywhere for `film_grain_obu()`, `film_grain_model()`, or `film_grain_config()`; the SEF writer explicitly rejects `film_grain_params_present` (`sef.cpp:127,188,201-208`, `sef.h:118-120`). `docs/spec-notes.md:59` lists the film grain OBU as deliberately untouched. | Whole signalling path: sequence bit, film grain OBU (8 slots, `fgm_chroma_idc`, `fgm_update_flags`), per-model syntax (§5.18.10.2: point curves, `ar_coeff_lag`, AR coefficients, `ar_coeff_shift`, `grain_scale_shift`, `cb_*`/`cr_*`, overlap/clip/block-size), per-frame `apply_grain`/`fgm_id`/`grain_seed`, and `save_grain_params`/`load_grain_model` slots. |
| Encoder-side grain model estimation / `fgm_id` selection | (encoder analysis; no normative clause) | **ABSENT** | No code outside synthesis. `docs/bitrate-features.md:285` records film grain as "M encoder analysis … U2 — unverified"; `docs/tool-roadmap.md:427` is likewise U2. | Denoising, point-curve estimation, AR least-squares fit, model rate/distortion selection, per-frame seed choice. |

## 3. §7.22 motion field motion vector storage

| Process | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| Motion-field MV storage (wedge allow-list, ref duplication, past/future ordering, `REFMVS_LIMIT`) | §7.22 p.572–575 | **PARTIAL** | `store_motion_field_entry()` (`inter/temporal/motion_field.cpp:1208-1292`): wedge mask counts `>60` / `<4` disable a list; drops MVs with `abs(mv) > REFMVS_LIMIT` (`kRefMvsLimit = 2047`, `temporal_types.h:67`); copies one list into the other when only one is present; applies the three-case `toSwitch` ordering; writes `MfRefFrames`/`MfMvs`. Tests in `temporal_test.cpp:364-411`; whole suite `checks: 8980, failures: 0` (run during this audit). | No block-level driver: caller must supply `refs`, `mvs`, `is_wedge`, `mask`, `i8/j8`; `bw4/bh4`, `n`, `refIdx0/1`, `isWedge`, `tipPred`, and the `Min(MiCols, …)` / `Min(MiRows, …)` clamps are not derived. `mvMethod > 0` (`MvDeltas` addition, p.573) not implemented; `tipPred`/`get_tip_cand()` (p.574-575) not implemented; warp/global-motion branch (`get_sub_block_warp_mv`) left to the caller. Not integrated: every sequence writer codes `enable_ref_frame_mvs = 0` (`m2_headers.cpp:155`, `leading.cpp:142`, `random_access.cpp:314`, `tlayers_syntax.cpp:181`, `longterm_headers.cpp:259`, `gme_gate.cpp:224`, `drl_stream.cpp:133`), so no `use_ref_frame_mvs` and no stored field is ever read by prediction. |
| `compression_mv` / `uncompression_mv` | §7.22 p.575–576 | **COMPONENT** | `temporal_math.h:89-106` implements both functions; compression applied at `motion_field.cpp:1287-1290`, decompression at `motion_field.cpp:342-343` when projecting a saved field. | Complete and tested, but only reachable from the same un-wired temporal component. |

## 4. §7.23 reference frame update

| Process | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| Slot refresh, `RefValid = (KEY/SWITCH) ? first : 1`, `RefOrderHint`, `RefCounter`, dims, base_q, t/m-layer | §7.23 p.576–577 | **COMPONENT** | `adv::refresh_reference_slots()` implements the `first` rule and increments `FrameCounter` (`inter/advanced/b_frame.cpp:61-90`); `sef::apply_refresh()` resets/validates slots and tracks shown/output flags (`sef.cpp:347-371`); `tlayers::first_refreshed_slot()` + key-slot invalidation (`tlayers.cpp:12-13,339,388-389,601`); `longterm_pool` §7.23 state machine incl. `RefLongTermId` (`longterm/README.md:116-126`). Production M2 does emit a correct trivial update (key ⇒ inferred `allFrames`; P ⇒ `refresh_frame_flags = 1`, ref slot 0; `m2_frame.cpp:1003,1041-1042`, `m2_headers.cpp:240,258`). | No general DPB: production hard-codes slot 0 and never recycles more than one slot (`num_ref_frames = 2` but only 1 valid after a P). `RefOutputOrder`/`RefMLayerId`/`RefTLayerId` multi-layer form absent from `bframes`; `RefFrameFiltersOn`, `RefFrameLrWienerNs`, `RefNumFilterClasses`, `RefDeltaQUAc/QVAc`, `RefNumTotalRefs` not stored. |
| Output eligibility checked inside refresh (`is_frame_eligible_for_output(i)` → §7.21.6) | §7.23 p.576 | **COMPONENT** | `simulate_decode()` calls output-eligible slots before storing the new frame, then refreshes, then outputs immediate/current (`bframes.cpp:737-766`). `sef::mark_shown()` marks a slot output by `output_process` (`sef.cpp:373-377`). | Component only. SEF/leading paths rely on the decoder gate rather than an encoder-side output log. |
| Frame store + `SavedMvs`, `SavedRefFrames`, `SavedGmParams`, `SavedSegmentIds` copy | §7.23 p.577 | **ABSENT** | No references to `SavedMvs`, `SavedRefFrames`, `SavedGmParams`, `SavedSegmentIds` in `src/` (grep empty). `gme_syntax.h:69` mentions `SavedGmParams` only as a syntax note; no storage. | Needed to use a stored frame's motion field (`§7.22`) or global motion for later temporal prediction; currently impossible because `enable_ref_frame_mvs = 0` and M2 has no GM. `RefFrameFiltersOn`/`RefFrameLrWienerNs`/`RefNumFilterClasses` snapshots also missing. |
| `save_cdfs(ctx)` / `load_cdfs(ctx)` reference contexts | §7.23 p.577–578 | **COMPONENT** | `avg_cdf_bank` implements per-slot `save_reference`/`load_reference` and `§7.4.4` blend (`src/encoder/entropy/avg_cdf/avg_cdf_bank.h:177-180`; `avg_cdf/README.md:16,49`). | Not used by M2 (`disable_cdf_update = true`, no primary-ref context load: `m2_headers.cpp:254-255,278`, `m2_frame.cpp:996`), and not used by the inter planners. The bank stores CDF arrays, not the full §7.23 frame context. |
| `save_grain_params` / `load_grain_params` and `save/load_ccso_params` | §7.23 p.577–578 | **ABSENT** | Film-grain save/load exists only as prose in `film_grain.h:12-13` and `spec_tables/README.md`; there is no `save_grain_params` implementation and no CCSO parameter store. Grain signalling itself is absent (see §7.21.7 row). | Both depend on features the encoder does not emit (`film_grain_params_present = 1`) or implement per slot (CCSO params). |
| Long-term references and OLK/RAS survival / first-Regular invalidation | §7.23 p.576–578; §7.4.4 | **COMPONENT** | `inter/longterm/longterm_pool.cpp` state machine implements `RefValid`/`RefLongTermId`/`RefOrderHint` update, the `first` rule, CLK invalidation and OLK survival; `longterm_headers.cpp` writes `long_term_frame_id_bits`, `long_term_id_plus_1`, `num_key_ref_frames`, `ref_long_term_id[]`; `longterm_gate.cpp` runs a 3-phase `dav2d --strict 1` gate. `random_access` adds SWITCH/RAS derivation and RAS refresh masks with its own gate. | Component only. M2 writes `long_term_frame_id_bits = 0` (`m2_headers.cpp:163`) and never emits OLK/SWITCH/RAS, so no long-term retention exists in production. |

---

## Counts

| Status | Count | Rows |
|---|---:|---|
| WIRED | **0** | — |
| COMPONENT | **8** | §7.21.3, §7.21.4, §7.21.6, §7.22 compression, §7.23 refresh, §7.23 output-on-refresh, §7.23 CDF contexts, §7.23 long-term/OLK |
| PARTIAL | **4** | §7.21.1, §7.21.2, §7.21.5, §7.22 storage |
| ABSENT | **4** | film-grain syntax, film-grain model estimation, §7.23 Saved* arrays, §7.23 grain/CCSO save-load |
| DECODER-ONLY | **5** | §7.21.7.1–§7.21.7.5 |
| **Total** | **21** | |

## Verdicts

**Can film grain be emitted? No.** No stream path can produce film grain. Every
sequence writer hard-codes `film_grain_params_present = 0`; `ObuType` in
`src/bitstream/m0_headers.h:26-31` has no `OBU_FILM_GRAIN`, so no film grain OBU
can even be constructed; no writer exists for `film_grain_model()` or
`film_grain_config()` (`apply_grain`, `fgm_id`, `grain_seed`); and there is no
encoder-side model estimation to populate a model. The 611-check synthesis
implementation in `src/encoder/filters/spec_tables/film_grain.cpp` is a
decode-side oracle only, is not in the root CMake build, has no call site
outside its unit test, and the SEF writer explicitly rejects
`film_grain_params_present == 1` (`sef.cpp:127,188`). A grain-bearing stream
therefore cannot be produced today.

**Is the output order correct (SEF / leading / implicit)? Where implemented,
yes; in production, only the trivial monotonic case exists.**

* **SEF**: the header layout, `derive_sef_order_hint` rule (derive = 1 only for
  a hidden, unshown frame; derive = 0 with the reference's own hint otherwise),
  `refresh_frame_flags = 0`, and §7.21.6 output routing are implemented and
  gated against pinned `dav2d` in three phases (`sef_gate.cpp:1-20`,
  `src/encoder/inter/sef/README.md` bit goldens, `sef_test` bit-exact OBUs). This is COMPONENT, not
  wired: `src/main.cpp` has no inter/SEF path.
* **Leading**: `leading.cpp:250-268` writes `immediate_output_frame` /
  `implicit_output_frame` exactly per §5.18.2 (OLK immediate inferred 0;
  `implicit` only when `immediate == 0` and `monotonic == 0`), validates the
  §6.17.2 rules, and is gated with a CLK + OLK + 3 leading + 3 regular stream,
  including RAP-start decode and output suppression (`leading_gate.cpp:1-30`).
* **Implicit output**: the §7.21.4/.5/.6 state machine in
  `bframes.cpp:300-441` implements eligible/implicit, successive-hint output,
  lower-hint-first frame-buffer output and flush, and the tlayers writer/gate
  proves a reordered stream (`monotonic_output_order_flag = 0`,
  `implicit_output_frame = 1`) decodes in display order with zero recon diffs
  (`tlayers_syntax.cpp:262`, `tlayers_gate.cpp:542-560`). The one functional gap
  is that flush ignores the spec's `olkLimit` / `OlkTUOrderHint` argument
  (§7.21.5 p.563-564; call sites §5.18.2 p.95), so RAP-boundary flushing is not
  modelled.
* **Production**: M2 sets `monotonic_output_order_flag = 1`, which infers
  `implicit_output_frame = 0`, and writes `immediate_output_frame = 1` on every
  frame (`m2_headers.cpp:110,236-237`). Output order is therefore trivially
  display = decode order, and correct — but no reordering, SEF or implicit
  output is reachable from the shipped still/M2 encoders. Wiring SEF, leading
  or B-frame output would require the integration steps already documented in
  `src/encoder/inter/sef/README.md`, `leading/M2_INTEGRATION.md` and
  `bframes/M2_INTEGRATION.md`.

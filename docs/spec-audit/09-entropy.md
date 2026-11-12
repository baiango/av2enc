# Spec audit 09 — entropy descriptors and the §8.2 symbol coder

Scope: AV2 v1.0.0 **§4.11 descriptors** (PDF pages 45–52) and **§8** (pages
579–609) for the entropy-relevant material: `f(n)`, `uvlc()`, `leb128()`,
`su(n)`, `ns(n)`, `rg(n)`, `L(n)`, `S()`, `NS(n)`; symbol-decoder state,
renormalisation and exit; CDF adaptation; average CDF; `disable_cdf_update`;
tile/frame CDF initialisation; trailing bits.

Read-only audit. No source, table, test or build file was modified; this report
is the only file created. PDF page index = printed spec page − 1 (pypdf).

Artifacts read (not trusted from docs):

| Area | Files |
|---|---|
| entropy core | `src/entropy/av2/av2_entropy.h`, `.cpp`, `av2_entropy_tables.h` |
| generic range coder | `src/entropy/range_coder.{h,cpp}` (not normative; see findings) |
| raw-bit I/O | `src/bitstream/bitwriter.{h,cpp}`, `src/bitstream/basic_headers.{h,cpp}` |
| M1 writer | `src/encoder/m1/still_syntax.cpp`, `still_headers.cpp`, `still_encoder.cpp`, `m1/tiles_stream/*`, `m1/qctx/*` |
| M2 writer | `src/encoder/m2/video_syntax.cpp`, `video_headers.cpp`, `video_frame.cpp`, `m2/post/tiled_frame.cpp` |
| CDF bookkeeping | `src/encoder/entropy/avg_cdf/*` |
| other descriptor homes | `src/encoder/rc/hrd/{hrd.cpp,bit_reader.h}`, `src/encoder/frame/mfh/mfh.cpp`, `src/encoder/syntax/small_header/*`, `src/encoder/palette/*` |
| CUDA mirror | `src/cuda/entropy.h`, `src/cuda/entropy.cu` |

Verification performed (all commands are read-only against the tree; build
output went to the approved temp directory):

- `src/entropy/av2/build_and_test.sh` → **116,478 checks pass**.
- `src/encoder/entropy/avg_cdf/build_and_test.sh` (unit + dav2d gate) →
  **555 checks pass**, 3 tiled 2-frame streams decode with dav2d `--strict 1`.
- `src/encoder/m1/partition_stream/build_and_test.sh` → adapter tests pass and
  six frozen/adaptive streams decode with dav2d (literal bits exercised).
- `tests/basic_encode_smoke.sh` → 4/4 cases pass (64×64/640×360, frozen and
  `--adaptive-cdf`).
- `av2enc --still --adaptive-cdf` on 64×64 produces a stream dav2d decodes.
- Frozen vs adaptive M0 output differ (1,138 B vs 1,094 B), so adaptation is
  genuinely applied, not ignored.
- Six descriptor encoders were re-implemented from the source and compared
  against spec-derived decoders in a throwaway script: `uvlc()` 0…65535 plus
  31/32-bit boundaries, `ns(n)` n=2…129 × every value, `NS(n)` (both
  hand-rolled copies), `su(n)` n=1…16 × every value, `rg(n)` n=1…9 × every
  representable value, `leb128()` 20k random 32-bit values → **0 mismatches**.
- `Prob_Inc[7][8]` and `Para_Adjustment_List[125][3]` were parsed out of the
  PDF and compared element-for-element with `av2_entropy_tables.h` → **exact**.

## Status legend

| Status | Meaning |
|---|---|
| **WIRED** | The rule is implemented and a real bitstream writer (M0/M1/M2 syntax writer, or the tile entropy coder) calls it. |
| **COMPONENT** | A correct implementation exists and is tested, but no production writer calls it. |
| **PARTIAL** | Some of the rule is implemented/wired and some is missing or only nominal. |
| **ABSENT** | No implementation of the rule anywhere in `src/`. |

---

## 1. §4.11 descriptors

| Rule/descriptor | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| `f(n)` — unsigned n-bit, MSB first | §4.11.2 p.46; §8.1 p.579 | **WIRED** | `BitWriter::put_bits` `src/bitstream/bitwriter.cpp:18-28`; readers `src/encoder/m1/deltaq/bitreader.h:42-51`, `src/encoder/frame/mfh/mfh.cpp:66-76`, `src/encoder/m1/filters_stream/bit_io.h:47-58`. Used for every header field (`basic_headers.cpp:177-199,308-316`; `still_headers.cpp:93-99,175-178`; `video_headers.cpp:278-290`). | No single shared `f(n)` API: `BitWriter` plus many ad-hoc `BitReader` classes (deltaq, mfh, filters_stream, hbd, random_access, leading, and test-local copies). No runtime range check (asserts only), but all call sites are compile-time-sized. |
| `uvlc()` | §4.11.3 p.46-47 | **WIRED** | `basic_headers.cpp:11-22`; 33 more independent definitions (`still_headers.cpp:13-24`, `video_headers.cpp:13-24`, and others). Used for `seq_header_id`, `chroma_format_idc`, `bit_depth_idc`, `cur_mfh_id`, `seq_header_id_in_frame_header` (`basic_headers.cpp:176-182,297-298`). | **34 independent definitions** in the tree. The `leadingZeros < 32` requirement is enforced only by `assert`; a release build asked to encode `0xFFFFFFFF` would emit 65 bits instead of rejecting. Nothing calls it with that value. Decoders cap at 32 leading zeros (`mfh.cpp:81-84`). |
| `leb128()` | §4.11.6 p.47-48 | **WIRED** | `append_leb128` `src/bitstream/basic_headers.cpp:125-143`; used for the Annex B.2 `num_bytes_in_obu` prefix at `:154` and by every OBU assembler. SVC variant `src/encoder/svc/svc_framing.cpp:182,304`. | Writing is complete and byte-aligned by construction (max 5 bytes for uint32, limit 8). **No production reader**: the in-OBU `leb128()` fields (`lcr_data_size`, `ops_data_size`, metadata sizes) are never emitted, and the only readers are test/gate helpers (`dcdeltaq_gate.cpp:70-83`). |
| `su(n)` | §4.11.7 p.48-49 | **COMPONENT** | `put_su`/`read_su` `src/encoder/frame/mfh/mfh.cpp:48-55,106-122`; `small_header.cpp:46-50,340-355`. Two's-complement semantics verified for all n=1…16. | **No wired call site.** Every emitted frame has `segmentation_enabled = 0` (`basic_headers.cpp:309`, `still_headers.cpp:176`, `video_headers.cpp:283`) and `delta_q_present = 0`, so `feature_value su(n)` (p.66) and `delta_q su(7)` (p.120) are never written. The implementations live in header modules that are not in `CMakeLists.txt` and not on any CLI path. `small_header::put_su` has UB at n=32 (`1U << 32`); `mfh` handles 32 explicitly. |
| `ns(n)` | §4.11.8 p.49 | **WIRED** | `basic_headers.cpp:25-40`; 25 more definitions (24 `void put_ns` copies plus the inline `m1/filters_stream/bit_io.h:107-119`). Used for `seq_max_bvp_drl_bits_minus_1 = ns(3)` at `basic_headers.cpp:230`, `still_headers.cpp:125`, `video_headers.cpp:166`, etc. Verified bit-exact for n=2…129 × every value. | 26 independent definitions, inconsistent validation: `basic_headers` asserts `n>=2`, some copies silently return for `n<=1`. Spec `ns(1)` is legal and emits zero bits; no caller uses it. Values are asserted `< n`, not checked at runtime. |
| `rg(n)` | §4.11.10 p.50 | **COMPONENT** | `put_rg` `src/encoder/rc/hrd/hrd.cpp:38-55`; `read_rg` `src/encoder/rc/hrd/bit_reader.h:90-113`. Used for `br_time rg(4)` and `ci_color_description_idc rg(2)` in HRD OPS/BRT OBUs. Verified inverse for n=1…9 × every representable value. | HRD is a standalone module: not in `CMakeLists.txt`, not called by M0/M1/M2 or the CLI. Writer rejects `n == 0`, although `rg(0)` is spec-legal (unary only with a zero-length remainder); no caller uses it. Returns an error for `q >= 32` / the `-1` case, which is correct for conformance. |
| `L(n)` — arithmetic literal | §4.11.11 p.50; §8.2.5 p.587 | **WIRED** | Encoder: `write_literal` `src/encoder/m1/still_syntax.cpp:441-448`, `src/encoder/m2/video_syntax.cpp:532-539`, `src/encoder/palette/palette.cpp:41-47` (10 independent copies exist in `src/`). Decoder: `av2_entropy.h:737-761`. Exercised by the partition dav2d gate (`literal_bits=1/2`) and the M2/palette writers. | No shared helper; 10 hand-rolled copies. Encoder-side `symbol_count` does not include literal bits, while `Decoder::read_literal` adds n to `symbol_count` (§8.2.5 FrameSymbolCount) — statistics only, no stream effect. |
| `S()` — arithmetic symbol | §4.11.12 p.50; §8.2.6 p.587-588 | **WIRED** | `av2::Encoder::encode_symbol` `av2_entropy.h:438-490`; `Decoder::decode_symbol` `:615-695`; all M1/M2 syntax symbols go through `Writer::sym` (`still_syntax.cpp:423-431`, `video_syntax.cpp:508-512`). | None for the descriptor. CDF-selection coverage is syntax-layer (§8.3) and out of scope here. `encode_symbol` correctly refuses a zero-probability symbol (`:469-475`). |
| `NS(n)` — arithmetic non-symmetric | §4.11.13 p.50-51 | **WIRED** | `write_ns_bits` `src/encoder/m2/video_syntax.cpp:1535-1547` (used for the `col_mv_index` suffix at `:1612-1614`); hand-rolled `encode_ns` in `src/encoder/palette/palette.cpp:538-552` (palette colour token at `:612`). Verified both against a spec decoder for n=2…129 × every value. | No entropy-layer helper; two independent copies. Neither call site is reachable from the shipped CLI (`encoder/m2/**` and `encoder/palette/**` are not in `CMakeLists.txt` and palette is not wired into M1 still), so NS is writer-wired but not CLI-reachable. |

Out-of-scope descriptors seen in passing: `svlc()` exists only in
`src/encoder/qm/qm.cpp:27-28`; `le(n)` is implemented ad hoc as `append_le`
(`m1/tiles_stream/tiles_stream.cpp:71`, `pipeline/tiled_still_gate.cpp:114`);
`tu(mx)` exists in `m1/filters_stream/bit_io.h:122-130`. None of them is in the
requested audit set.

---

## 2. §8.2 symbol-decoder state, renormalisation and termination

| Rule/descriptor | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| `get_position()` / byte-aligned `init_symbol` | §8.2.1 p.579; §8.2.2 note | **WIRED** | `Decoder::bit_pos` is the position (`av2_entropy.h:367-380,579-613`); `exit_symbol` uses it (`:763-804`). All writers init on a byte-aligned tile payload (`still_syntax.cpp:318-319`, `video_syntax.cpp:412-413`, `basic_encoder.cpp:497`). | The decoder does not assert that the *underlying OBU* position is byte aligned; it trusts the caller's tile span. Harmless for the current one-tile-group writers. |
| `init_symbol(sz)` state: `numBits=Min(8sz,15)`, `paddedBuf`, 15-bit inversion, `SymbolRange=1<<15`, `SymbolMaxBits=8sz-15` | §8.2.2 p.579-580 | **WIRED** | `av2_entropy.h:579-613` exactly; encoder counterpart `:416-436`. Tests cover sz=0/1/2, truncation, max-bits values (`av2_entropy_test.cpp:283-369`). | None. |
| §8.2.2 copy of every frame CDF array to its `Tile*` copy at tile start | §8.2.2 p.580-585 | **COMPONENT** | Exact implementation: `CdfBank::load_tile_cdfs()` `avg_cdf_bank.cpp:227-234`, storage model `avg_cdf_bank.h:75-101`. **No caller outside `src/encoder/entropy/avg_cdf/`** (verified by grep). Writers build one `CdfState` per `Writer` from defaults (`still_syntax.cpp:316-317`, `video_syntax.cpp:410-411`, `basic_encoder.cpp:493`). | The frame→tile copy is never invoked. It is currently stream-equivalent because every emitted frame starts from the §9.3 defaults and either is single-picture or signals `disable_cross_frame_cdf_init = 1`; a frame that carries CDFs between tiles/frames would need this. |
| `read_bool()`: `cur=range>>1`; `symbol=value<cur`; subtract `cur` when 0; `Clip3(0,1,maxBits)` new bit; `value=(value<<1)|(newData^1)`; `maxBits-1` | §8.2.3 p.585 | **WIRED** | `av2_entropy.h:697-735`; encoder inverse `:492-520`. Tests: 20-bit bool stream, all-zero padding, `maxBits` boundary (`av2_entropy_test.cpp:548-621`). | The published §8.2.3 step list omits a `SymbolRange` assignment; this implementation deliberately leaves range unchanged (documented at `av2_entropy.h:13-14` and `src/entropy/av2/README.md:169-196`). That resolution is consistent with the even-range invariant and is validated by dav2d. |
| `read_literal(n)`: n `read_bool()` calls, `FrameSymbolCount += n` | §8.2.5 p.587 | **WIRED** | `av2_entropy.h:737-761`; tests n=0…32 (`av2_entropy_test.cpp:592-621,1105-1107`). | Encoder `write_literal` does not mirror the count (see L(n) row). Stats only. |
| Boundary computation: `f`, `pp=((f>>7)<<4)+Prob_Inc[N-2][s]`, `cur=(((range>>8)*pp)>>7)<<3`, do/while on `value<cur` | §8.2.6 p.587 | **WIRED** | `compute_boundaries` `av2_entropy.h:157-181`; decoder loop `:642-657`; encoder reuses the same boundaries `:460-477`. Zero-width entries handled and tested (`av2_entropy_test.cpp:669-706`). | None. `cdf_is_valid` intentionally allows repeated boundaries (legal zero-probability symbol) and rejects only decreases. |
| Renormalisation: `newRange`, `newValue`, `bits=15-FloorLog2(newRange)`, `Clip3(0,bits,maxBits)`, `paddedData`, `mask`, XOR, `maxBits-=bits` | §8.2.6 p.588 | **WIRED** | `av2_entropy.h:659-687`. Boundaries are multiples of 8, so `bits<=12` and `range` stays in [32768,65536). Tested over a 200k-symbol stream (`av2_entropy_test.cpp:1011-1080`). | None. |
| CDF update: `timeInterval`, `rate = 3+ti+Min(FloorLog2(N),2)+Para[cdf[N-1]][ti]`, subtract/add half-open halves, `cdf[N] += (cdf[N]<32)` | §8.2.6 p.588 | **WIRED** | `adapt_cdf` `av2_entropy.h:185-204`; called from both directions only when enabled (`:485-487`, `:689-691`). Exact-value tests cover both `timeInterval` thresholds and N=2/3/4/8 (`av2_entropy_test.cpp:622-668`). | None. Zero-probability entries are handled; an encoder asked to code one returns `kInvalidSymbol` instead of corrupting state. |
| `cdf[N-1]` constant rate index and `cdf[N]` count cap 32 | §8.2.6 p.587-588 | **WIRED** | `cdf_is_valid` `av2_entropy.h:127-146`; `adapt_cdf` loop stops at `n-1`, count capped at `kCdfCountCap=32` (`:201-203`). | None. Tables are validated per symbol. |
| §9.2 `Prob_Inc[7][8]`, `Para_Adjustment_List[125][3]` | §9.2 p.621-622 | **WIRED** | `av2_entropy_tables.h:21-158`, generated with a pinned PDF SHA. Reparsed from the PDF in this audit: 7/7 and 125/125 rows identical. | None. |
| `exit_symbol()`: `maxBits >= -14`, trailing position, advance `Max(0,maxBits)`, byte alignment, marker bit 1, intervening zeros | §8.2.4 p.585-586 | **WIRED** | Decoder `av2_entropy.h:763-804`; encoder `:522-573` (`delta=(0x4000-low)&0x7fff`, carry, marker, pad). Tests: 8 manual patterns plus the `maxBits=-14` accept/reject pair (`av2_entropy_test.cpp:312-369`). dav2d accepts every emitted tile. | Encoder's final-code choice is one valid point in the final interval; AV2 only defines the decoder, so this is a free choice (documented). |
| `numLog2 = Min(3, FloorLog2(TileCols*TileRows))` | §8.2.4 p.586 | **COMPONENT** | `average_log2`/`average_tile_limit` `avg_cdf_bank.cpp:26-40`; tests at 1/2/3/4/8/9/64 tiles (`avg_cdf_test.cpp`, 555 checks). | Computed only inside the unused bank. No tile-group writer computes or uses it. |
| `avg_cdf(cdf, tilecdf, sz)` incl. tile-0 initialisation, per-element `(32768-tilecdf)>>numLog2`, count sum, and the `TileNum < 1<<numLog2` cutoff | §8.2.4 p.586-587 | **COMPONENT** | Two exact implementations: `av2::average_cdf` `av2_entropy.cpp:56-81` and `CdfBank::average_cdf` `avg_cdf_bank.cpp:42-66`; cutoff `:240-251`. Unit-tested against hand values and cross-checked between the two (`av2_entropy_test.cpp:986-1010`; bank 555 checks incl. pinned `15193,19866,56,15`). | **Not wired.** No writer calls either function; `M1/M2` never carry `Saved*Cdf` state. The dav2d avg-CDF gate (`gate_dav2d_avg_cdf.cpp`) deliberately freezes CDFs, so it validates only the `context_update_tile_id` omission, not averaged probabilities. |
| `copyCdf` on exit / `context_update_tile_id` | §8.2.4 p.586; §5.18.7.2 p.123 | **PARTIAL** | Header side is correct: the field is written as `cols_log2+rows_log2` bits only when `!enable_avg_cdf || !avg_cdf_type` (`m2/post/tiled_frame.cpp:357-366`), omitted for the inferred 1/1 single-picture headers (`m1/tiles_stream/tiles_stream.cpp:355-381`), absent for one tile (`basic_headers.cpp:95-101`). Bank side `CdfBank::save_tile_cdfs` `avg_cdf_bank.cpp:236-263`. | The `Saved*` copy is never performed by a writer. M2 writes a placeholder `context_update_tile_id = 0` (comment at `tiled_frame.cpp:361-365`). Semantically ignored by the decoder while `disable_cdf_update = 1`, so current streams are valid, but a real tile-selection policy is not implemented. |
| `FrameEndUpdateCdf()`: `frame = Saved`, `frame[N] = (3*Saved[N])>>2` | §7.5 p.395 | **COMPONENT** | `CdfBank::average_at_frame_end` `avg_cdf_bank.cpp:265-275`; tested (pinned final frame CDF and hash). | Never called from a writer; no frame-level CDF state exists in M1/M2. Unobservable for single-picture M0/M1 and for M2's frozen CDFs. |
| `load_cdfs(ctx)` / `blend_cdfs(ctx)` reference contexts; `blend_cdf` 7/8+1/8 arithmetic | §7.4.4 p.111; blend p.334 | **COMPONENT** | `CdfBank::save_reference`/`load_reference`/`blend_reference` `avg_cdf_bank.cpp:277-317`; `blend_cdf` `:68-85` matches `(32768 - ((32768-saved)+7*(32768-cur)+4)>>3)` and blends the count but not the rate entry. | Not wired, and no writer can emit a frame that loads/blends reference CDFs. Currently harmless: every emitted frame signals reset (`disable_cross_frame_cdf_init = 1` at `video_headers.cpp:255`, or `single_picture_header_flag = 1` for M0/M1). |
| `disable_cdf_update` gating | §8.2.6 p.588; header §5.18.2 | **WIRED** | Encoder/decoder flags `av2_entropy.h:229,379`, gates `:485-487,689-691`. Header bit matches the coder flag in every writer (`basic_headers.cpp:303`, `still_headers.cpp:172`, `video_headers.cpp:278`, `m2/post/tiled_frame.cpp:436`). CLI opt-in for M0/M1 (`main.cpp:268-269`, `still_encoder.cpp:1810`). Adaptive vs frozen M0 streams differ and both decode; M1 adaptive decodes; partition gate adaptive streams decode. | The *rule* is complete, but policy blocks the interesting cases: `VideoConfig.disable_cdf_update` is hard-coded `true` (`video_frame.cpp:403,996`); tiled M1 forces `adaptive_cdf=false` (`m1/tiles_stream/tiles_stream.cpp:527`); tiled M2 rejects adaptive multi-tile (`tiled_frame.cpp:379,650`). No multi-frame or multi-tile adaptive CDF stream can therefore be produced. |
| Tile/frame CDF initialisation (`init_coeff_cdfs`, `init_non_coeff_cdfs`, qctx slice, reset-on-reference rule) | §6.17.2 p.328-333; §7.4.4 p.111 | **PARTIAL** | `init_cdfs` copies all §9.3 defaults and selects the coefficient slice from raw `base_q_idx` (`still_syntax.cpp:216-300` with `qctx.h:51-56`; M2 equivalent `video_syntax.cpp:288-348,410-411`; M0 fixed slice 0 at `basic_encoder.cpp:368` and QP pinned to 90 at `:2088`). Slice boundaries 90/140/190 verified against p.333. | Only the "reset to defaults" branch is wired. `load_cdfs`/`blend_cdfs`/`Saved*` installs are absent from the writers (bank-only). This is currently conformant because all emitted frames take the reset branch, but cross-frame CDF continuity is unimplemented. |
| `trailing_bits(nbBits)` for OBUs, and the tile-group exception | §5.2.3 p.54; §5.2.1 p.53 | **WIRED** | Sequence/other OBUs: `put_trailing_bits` `basic_headers.cpp:54-57,283`, M1/M2 reuse it. Tile groups: `usedArith = is_tile_group()` skips §5.2.3, and §8.2.4 exit consumes the OBU trailing bits (p.586 note). Writers concatenate the tile payload with **no** extra trailing byte (`m1/tiles_stream/tiles_stream.cpp:434-446`, `basic_headers.cpp:341-349`, `m2/post/tiled_frame.cpp:692-710`). | None. This matches the spec's tile-group exception and is dav2d-validated at 1, 2 and 4 tiles. |
| Encoder inverse: interval commit, carry into emitted prefix, bit renormalisation, termination | §8.2.2–8.2.4 (inverse; spec defines decoder only) | **WIRED** | `av2_entropy.h:307-352` (carry/commit), `:438-490`, `:522-573`. 116,478 checks incl. 200k symbols and multi-tile lockstep; a CUDA mirror exists (`src/cuda/entropy.h:336-530`, not run here); every M0/M1/M2 gate stream I ran decodes. | `Writer` constructors ignore the `init_symbol` status (`still_syntax.cpp:319`, `video_syntax.cpp:413`, `basic_encoder.cpp:497`); cannot fail with the fixed 16 MiB tile buffer, but it is unchecked. Encoder `symbol_count` does not count literal bits (stats only). |
| CUDA entropy port | portability, not a spec clause | **COMPONENT** | `src/cuda/entropy.h:1-22` reuses `av2::compute_boundaries/adapt_cdf/cdf_is_valid/floor_log2` and mirrors the CPU state machine; kernel in `src/cuda/entropy.cu`; byte-exact comparison source in `src/cuda/entropy_test.cu` (not run in this audit: no CUDA on this macOS host). | Not wired into the CLI or M0/M1 pipeline; CPU entropy remains the shipped path. Shared helpers mean no rule fork. |

---

## 3. Findings and flags

### 3.1 No descriptor is absent

All nine requested descriptors have a correct encoder somewhere in `src/`.
**Zero ABSENT descriptors.** The material gaps are reachability and
bookkeeping, not bit layout:

1. **`su(n)` — COMPONENT, not wired.** Correct in `mfh.cpp` /
   `small_header.cpp`, but every emitted frame disables segmentation and
   delta-Q, so no `su(n)` ever reaches a bitstream. The module is not in
   `CMakeLists.txt`.
2. **`rg(n)` — COMPONENT, not wired.** Correct in the HRD module for
   `rg(2)`/`rg(4)`, but HRD OBUs are not produced by the encoder CLI and the
   module is not in `CMakeLists.txt`. `rg(0)` (spec-legal) is rejected.
3. **`NS(n)` — writer-wired, not CLI-reachable.** Correct in `video_syntax.cpp`
   (`col_mv_index` suffix) and `palette.cpp` (colour tokens), but M2/palette are
   not in the shipped binary.
4. **Descriptor robustness gaps (no current stream impact):** `uvlc()`
   rejects the illegal `0xFFFFFFFF` only via `assert`; `ns(n)` validates
   `value<n` with `assert`; 34 copies of `uvlc()`, 26 of `ns(n)`, 10 of `L(n)`.

### 3.2 CDF-update chain: the main partial area

The rules are individually implemented and tested, but the full §8.2.2 →
§8.2.4 → §7.5 → §7.4.4 chain is **not wired**:

| Rule | Status | Consequence today |
|---|---|---|
| §8.2.2 frame→`Tile*` copy | COMPONENT | Stream-equivalent: every frame resets to defaults. |
| §8.2.4 `copyCdf` / `Saved*` | PARTIAL | `context_update_tile_id` is written correctly but is a placeholder 0; `Saved*` never filled. The effect is nil while `disable_cdf_update=1`, because every tile's CDFs are identical. |
| §8.2.4 `avgCdf` | COMPONENT | Never averages; M2 sets `enable_avg_cdf=0`; M0/M1 infer 1/1 but are single-picture and either one tile or frozen. |
| §7.5 `frame_end_update_cdf` | COMPONENT | Never installs a frame CDF. |
| §7.4.4 `load_cdfs`/`blend_cdfs` | COMPONENT | No cross-frame CDF continuity or blending. Safe only because every emitted frame signals `disable_cross_frame_cdf_init=1`. |
| Tile/frame CDF init | PARTIAL | Defaults + qctx slice are wired; reference/saved branches are not. |

**Flag:** any change that turns on adaptive CDFs for a tiled M2/M1 frame whose
CDFs are supposed to survive a frame end, or that stops signalling
`disable_cross_frame_cdf_init=1`, will make the encoder diverge at the first
tile that must start from a non-default CDF. The CDF bank
(`src/encoder/entropy/avg_cdf/`) is the ready integration point
(`load_tile_cdfs` → per-tile CDF seed → `save_tile_cdfs` →
`average_at_frame_end`), but the syntax writers expose neither a seed parameter
nor a final-CDF export (`still_syntax.cpp:1492`, `video_syntax.cpp:1992` hold one
private `CdfState`).

### 3.3 Policy restrictions that hide the gap

- `VideoConfig.disable_cdf_update = true` at both M2 frame constructors
  (`video_frame.cpp:403,996`).
- Tiled M1 hard-codes `options.adaptive_cdf = false`
  (`m1/tiles_stream/tiles_stream.cpp:527`) and writes
  `disable_cdf_update = 1` (`:188`).
- Tiled M2 refuses multi-tile adaptive CDFs outright
  (`m2/post/tiled_frame.cpp:379,650`).
- Single-tile adaptive M0/M1 is the only adaptive path that is exercised
  end-to-end; it is conformant (dav2d), but it cannot exercise §8.2.2 copies,
  §8.2.4 averaging, §7.5 frame-end update or §7.4.4 reference blending.

### 3.4 Non-conformance / stale-doc observations

- **No spec non-conformance was found in the emitted streams.** The symbol
  coder matches §8.2 arithmetic, the descriptor encoders are bit-exact
  inverses, and dav2d accepts frozen and adaptive M0/M1 streams plus the
  frozen-CDF multi-tile avg-CDF gate.
- `src/entropy/av2/README.md:299-320` is stale: it says there is no external
  dav2d conformance and no CUDA kernel, but `tests/basic_encode_smoke.sh`,
  `m1/partition_stream/gate_dav2d.py` and `src/cuda/entropy.{h,cu}` now exist;
  the CPU gates pass, and the CUDA test was not run on this host. Do not cite
  that section.
- `src/entropy/range_coder.{h,cpp}` is **not** the AV2 coder (its own header
  says so). It is used only by `src/entropy/entropy_test.cpp`; no bitstream
  path consumes it. It must not be treated as §8.2 coverage.
- Comment mis-citations: `m1/tiles_stream/tiles_stream.cpp:27` and
  `pipeline/tiled_still_gate.cpp:70` say "§4.11.5 le(n), §4.11.6 uvlc"; `uvlc`
  is §4.11.3 and `leb128` is §4.11.6. `m1/deltaq/bitreader.h:9-11` claims to be
  the inverse for `uvlc()/su(n)` but implements neither.
- `CdfBank::average_cdf`'s `first_tile` flag and `av2::average_cdf`'s
  `tile_num` argument both correctly implement the spec, but they have
  duplicated arithmetic in two modules; the bank test cross-checks them, so
  drift is currently detected.

---

## 4. Counts

| Measure | Count |
|---|---:|
| Rules/descriptors audited | **30** |
| WIRED | **20** |
| COMPONENT | **8** |
| PARTIAL | **2** |
| ABSENT | **0** |

Breakdown:

| Area | Rows | WIRED | COMPONENT | PARTIAL | ABSENT |
|---|---:|---:|---:|---:|---:|
| §4.11 descriptors | 9 | 7 | 2 | 0 | 0 |
| §8.2 state machine + termination + encoder inverse | 10 | 10 | 0 | 0 | 0 |
| §8 CDF update / average / init / disable | 8 | 1 | 5 | 2 | 0 |
| §9.2 tables, OBU trailing bits, CUDA mirror | 3 | 2 | 1 | 0 | 0 |

Descriptors by status: `f(n)`, `uvlc()`, `leb128()`, `ns(n)`, `L(n)`, `S()`,
`NS(n)` = WIRED; `su(n)`, `rg(n)` = COMPONENT; none PARTIAL or ABSENT.

CDF-update rules by status: adaptation formula, count/rate cap and
`disable_cdf_update` gating = WIRED; `Tile*` copy, `avg_cdf`, `numLog2`,
`frame_end_update_cdf`, reference `load`/`blend` = COMPONENT; `copyCdf`
(`Saved*`) and the tile/frame init surface = PARTIAL. **No CDF rule is
ABSENT**, but six of the eight bookkeeping/init rules are not wired, and the
two PARTIAL rules are the integration points most likely to break first when
cross-frame adaptation is enabled.

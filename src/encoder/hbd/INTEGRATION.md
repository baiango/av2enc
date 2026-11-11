# Flipping M1/M2 to 10-bit — exact call sites and risks

This document is a precise work list for the M1/M2 owners.  It is written against
the tree as of 2026-11-12; line numbers drift, so the **symbol** is the primary
key.  Nothing in this list was edited by `src/encoder/hbd/`.

Suggested order: **(1) header bits → (2) sample containers → (3) transform /
dequant / recon → (4) entropy → (5) filters.**  The first stream should be
64x64, QP 90, DC intra, deblock off.  Compare against the `hbd` 0-LSB gate
before adding any tool.

`src/encoder/hbd/hbd_depth.*` is the source of truth for every numeric bound
below; do not re-derive them.

---

## M1 — headers

| Call site | Current | 10-bit change |
|---|---|---|
| `src/encoder/m1/m1_headers.h:62` `M1Config::base_q_idx` | `std::uint8_t` | `std::uint16_t` (MaxQ 303 does not fit a byte) |
| `src/encoder/m1/m1_headers.h` `M1Config` | no depth field | add `fmt::BitDepth depth = k10` or an explicit `bit_depth_idc` |
| `src/encoder/m1/m1_headers.cpp:85` | `put_uvlc(writer, 1); // bit_depth_idc = 1 => 8-bit` | `put_uvlc(writer, 0);` for 10-bit (`fmt::bit_depth_idc(depth)`) |
| `src/encoder/m1/m1_headers.cpp:175` | `writer.put_bits(c.base_q_idx, 8U)` | `writer.put_bits(c.base_q_idx, 9U)` (`fmt::base_q_idx_bits(depth)`) |
| `M1Config` validation (`m1_still.cpp:1471` for q=0) | `1..255` | `1 <= q <= fmt::max_quantizer(depth)` = 303; reject 12-bit |
| Reject 12-bit | not checked | `fmt::bit_depth_idc(k12) == -1` |

The sequence-header tool bits do not change: `chroma_format_idc = 0`,
`seq_profile_idc = 0`, and the tool disables are the same at both depths.
`header_bit_counts()` in `hbd` pins 81/23/24 bits at 64x64 for the 10-bit
M10-compatible tool set.

## M1 — options, sample containers, reconstruction

| Call site | Current | 10-bit change |
|---|---|---|
| `src/encoder/m1/m1_still.h:13-15` `StillPicture::y/u/v` | `span<const uint8_t>` | `span<const uint16_t>` (or a depth-generic view) |
| `src/encoder/m1/m1_still.h:22` `M1Options::base_q_idx` | `uint8_t` | `uint16_t` |
| `src/encoder/m1/m1_still.h:73-75` `M1Result::recon_*` | `vector<uint8_t>` | `vector<uint16_t>` |
| `src/encoder/m1/m1_still.cpp:73` `clip1(int)` | `clip3(0,255,v)` | `depth::clip1(v)` (1023) |
| `src/encoder/m1/m1_still.cpp:102` `Plane::samples` | `vector<uint8_t>` | `vector<uint16_t>` |
| `src/encoder/m1/m1_still.cpp:130-131` `TxPlan::pred/recon` | `vector<uint8_t>` | `vector<uint16_t>` |
| `src/encoder/m1/m1_still.cpp:410` `source_sample()` | returns `uint8_t` | returns `uint16_t` |
| `src/encoder/m1/m1_still.cpp:992` | `static_cast<uint8_t>(clip1(p + r))` | `static_cast<uint16_t>(depth::clip1(p + r))` |
| `src/encoder/m1/m1_still.cpp:196` `quant_step_` | `rc::kQlookupQ0To255[q]` | `depth::qlookup(q)` (same table, q up to 303) |
| `src/encoder/m1/m1_still.cpp:769-784` dequant | `txv::inverse_transform` / manual ±(1<<15) clip | `depth::clip_dequant`, then `hbd::inverse_block(..., 10)` |
| `src/encoder/m1/m1_still.cpp:223` recon init | 0 | unchanged (decoder's CurrFrame starts at 0) |
| M1 DC predictor neutral | literal `128` / `1 << (bd-1)` | `depth::sample_ranges().dc_bias` = 512 |
| `src/encoder/m1/m1_still.cpp:1498-1499` `above_level_` / `left_level_` | `uint8_t` | unchanged: `cul_level = min(4, ...)` saturates (verified line 1298) |

**Do not** call `txv::dequantize` / `txv::inverse_block` for 10-bit: both clip
at 8-bit bounds (`±(1<<15)` and `[-256,255]`).  Use
`hbd::dequantize` / `hbd::inverse_block(..., 10)`, or the depth wrappers here.
Residuals stay `int16_t`, dequant `int32_t`, `Transform_Shift`, `dq_denom`,
scan and EOB unchanged.

The intra-mode component (`src/encoder/intra_modes/`) is already
bit-depth-generic: `clip1(value, bit_depth)`, `mid = 1 << (bd-1)`, and the
edge fillers `mid ± 1`.  Only `PlaneView::data` is `uint8_t`
(`intra_modes.h:97`, `read_sample()` at `intra_modes.cpp:110`) — widen it.

## M1 — entropy

| Call site | Current | 10-bit change |
|---|---|---|
| `src/encoder/m1/m1_syntax.cpp:216,316` | `init_cdfs(cdfs, qctx::slice_for_base_q_idx(base_q_idx))` | already correct: slice from **raw** `base_q_idx`; no depth normalization |
| coefficient levels | escape path handles >255 | verified by M10 (`max|level|` 700–2500); add a 10-bit round-trip test |
| `above_level`/`left_level` | `min(4, cul_level+abs)` | unchanged |
| `base_q_idx` header field | 8 bits | 9 bits; the coder sees the same integer |

The integrated M1/M2 entropy is already qctx-aware (unlike the reused M10
writer), so q=91..303 works there once the header and quantizer are flipped.

## M2 — headers and frame

| Call site | Current | 10-bit change |
|---|---|---|
| `src/encoder/m2/m2_headers.h:25` `M2Config::base_q_idx` | `uint8_t` | `uint16_t` |
| `src/encoder/m2/m2_headers.cpp:102` | `put_uvlc(writer, 1)` | `put_uvlc(writer, 0)` |
| `src/encoder/m2/m2_headers.cpp:282` | `put_bits(c.base_q_idx, 8U)` | 9 bits |
| `src/encoder/m2/m2_frame.h:30-32` `M2InputFrame` | `span<const uint8_t>` | `span<const uint16_t>` |
| `src/encoder/m2/m2_frame.h:39-41` `M2ReconFrame` | `vector<uint8_t>` | `vector<uint16_t>` |
| `src/encoder/m2/m2_frame.h:45` `M2Options::base_q_idx` | `uint8_t` | `uint16_t` |
| `src/encoder/m2/m2_frame.cpp:42` `Plane::samples` | `vector<uint8_t>` | `vector<uint16_t>` |
| `src/encoder/m2/m2_frame.cpp:69` `Plan::recon` | `vector<uint8_t>` | `vector<uint16_t>` |
| `src/encoder/m2/m2_frame.cpp:84` `clip1` | 255 | `depth::clip1` |
| `src/encoder/m2/m2_frame.cpp:147` `frame_quant_step()` | `rc::kQlookupQ0To255` | `depth::qlookup` (q up to 303) |
| `src/encoder/m2/m2_frame.cpp:206-241` dequant/recon | manual ±(1<<15) + `txv::inverse_block` | `depth::clip_dequant` + `hbd::inverse_block(..., 10)` |
| `src/encoder/m2/m2_frame.cpp:117-127` intra DC fallback | `uint8_t` pred + `max_x/y` | `uint16_t`; the intra component handles 10-bit |
| `src/encoder/m2/m2_syntax.cpp:288,410` | already qctx-aware | no change beyond the header/quantizer |

M2 P-frame residual:

| Call site | Current | 10-bit change |
|---|---|---|
| `src/encoder/m2/residual/inter_residual.h:38,93,175-183` | `uint8_t` source/pred/recon spans | `uint16_t`; 10-bit dequant/inverse |
| `src/encoder/m2/residual/inter_coeff_syntax.h:114-126` | `uint8_t` context arrays + `tx_size/type` | context arrays stay byte-sized (saturation), tx metadata unchanged |

## Filters

| Call site | Current | 10-bit change / risk |
|---|---|---|
| `src/encoder/filters/deblock/deblock.cpp:55` | rejects `bit_depth != 8` | widen `DeblockPlane::data` / `DeblockCoreFrame::plane` to `uint16_t`, `clip1_8` → depth clip |
| `deblock_core.h:260` `kMaxQ` | 255 | 303 at 10-bit (`depth::max_q`) |
| `deblock_core.h:319-321` | q_ind = `lvl - 24*(bd-8)`, side `(t+(1<<(12-bd)))>>(13-bd)` | formula is already depth-aware; **verify** the 10-bit `side` |
| CDEF `cdef.h:54-55` / `cdef_core.h:343` | `coeffShift = bd - 8`; only 8 supported | `coeffShift = 2` at 10-bit; pri/sec strengths are shifted by 2 |
| CCSO `ccso_core.h:43-45,72` | `uint8_t` planes | not ready; keep CCSO off |
| restoration (Wiener/PC-Wiener) | 8-bit sample types | not ready; keep restoration off |
| M2 post `src/encoder/m2/post/loop_filters.h:105-108` | `uint8_t* plane_data` | 10-bit M2 must keep all loop-filter flags off (the checklist already says so) |

Deblock first gate: **off**.  When enabling it, use q where the 10-bit `side`
is non-zero.  At `base_q_idx = 90` the side is **0** at 10-bit
(`Side_Thresholds[90-48] = -16`, `(-16+4)>>3 = 0`), whereas 8-bit gives 1.
This is not a bug: the §7.17 q index is offset by `24*(BitDepth-8) = 48`.
Pick q >= 140 for a 10-bit deblock gate (`side = 4` at q=140).

## CfL

| Call site | Current | 10-bit change |
|---|---|---|
| `src/encoder/cfl/cfl_common.h:288,293` `Block::luma` / `chroma` | `const uint8_t*` / `uint8_t*` | `const uint16_t*` / `uint16_t*` |
| `src/encoder/cfl/cfl.h:49` `SourcePlane::plane` | `const uint8_t*` | `const uint16_t*` |
| `src/encoder/cfl/cfl.cpp:380,419` | casts to `uint8_t` | `uint16_t` |
| `Block::bit_depth` | 8 default | set 10; `clip1(value, bit_depth)` already correct |
| `compute_luma()` `cfl.cpp:248-252` | `8 << (bd-1)` = 4096, clamp `(8<<bd)-1` = 8191 | already correct; add a 10-bit test |

## Risks (ordered by how easy they are to miss)

1. **`Clip1` / `Clip3` bounds.** Every `255`, `128`, `±(1<<15)`, `±(1<<8)` and
   `uint8_t` sample container above is an 8-bit assumption.  At 10-bit the
   correct values are `1023`, `512`, `±(1<<17)`, `±(1<<10)` and `uint16_t`.
   The two `txv` functions must not be used for 10-bit at all.
2. **`MaxQ`.** `base_q_idx` is f(9) and MaxQ is 303.  A `uint8_t` config
   silently wraps 303 → 47; validate before the header write.
3. **`qlookup`.** The step table is depth-independent, but q climbs to 303.
   `qlookup(303) = 245760`; the dequant product `Abs(qc) * q2` should be
   computed in 64-bit (or masked in 32-bit only because 10-bit levels are
   small).  `hbd_still` uses int32 at q <= 90; an integrated path supporting
   303 should use int64 or prove the level bound.
4. **Coefficient-CDF selection.** §8.3.1 uses the **raw frame** `base_q_idx`,
   not a delta-Q/segment qindex and not the depth.  `m1_syntax`/`m2_syntax`
   already do this; keep it that way.
5. **Deblock q offset/side.** Subtract 48 from the side-table index; at q=90
   the 10-bit side is 0, so "deblock enabled but nothing filtered" is expected.
6. **DC bias / edge fills.** `1 << (BitDepth-1)` = 512; `AboveRow` = 511,
   `LeftCol` = 513, corner = 512.  Do not keep the 8-bit 127/129/128 constants
   in a 10-bit predictor.
7. **CfL luma units.** `L` and `lumaAvg` carry 3 fractional bits; the bias is
   `8 << (bd-1)` = 4096 and the clamp is 8191.  `scaledLuma` is
   `Round2Signed(alpha_q8*(L-lumaAvg), 11)` and the final write is `Clip1`.
8. **Residual conformance.** `Residual` must fit a signed integer with
   `1 + BitDepth` bits (`±(1<<10)` at 10-bit); `hbd::depth::residual_representable()`
   is the check.
9. **Float contraction.** Keep `-ffp-contract=off` (CPU) / `-fmad=false`
   (CUDA).  The forward transform is double-precision and must match the
   decoder's integer inverse.
10. **Init state.** Reconstructed planes start at 0, not `1 << (bd-1)`.  CfL
    reads reconstructed luma, so an initialised-to-512 plane shifts predictions.

## Seams owned by other agents

* `src/encoder/m10/m10_entropy.cpp` — slice-0 coefficient CDF only.  The
  `hbd` encode path is therefore limited to q <= 90.  Fixing it means adding a
  qctx argument to `init_cdfs()`; the four slices already exist in
  `src/encoder/m1/qctx/`.
* `src/encoder/m2/post/bit_depth_checklist.cpp` — already lists the 10-bit M2
  gaps; update its status only after the above lands.
* `src/main.cpp` / `CMakeLists.txt` — not touched.  The component has its own
  `build_and_test.sh`.

# AV2 v1.0.0 spec notes for M0

Working notes for the still-picture encoder. The *recipe* is
`docs/minimal-bitstream.md`; this file is the **map of the specification** plus the
normative pieces you must internalise before writing code.

Spec identity: `spec/AV2_Spec_v1.0.0.pdf`, AV2 Bitstream & Decoding Process
Specification, v1.0.0, 2026-05-28, commit `38f28e7`, 1169 pages, SHA-256
`e9916f091e4e83446aad6b4601641c5b292e569c144c4163b26a4497573b533f`.
HTML cross-reference: <https://av2.aomedia.org/v1.0.0/>.

A note that saves days: **the HTML version omits all numeric tables** in §9.3
(default CDFs) and §9.6 (transform kernels), replacing them with "is defined in the
`<name>.h` header file". The PDF contains the numbers. All generated tables in
`spec/tables/` come from the PDF. See `docs/open-questions.md` for the other
HTML/PDF differences (there is at least one missing algorithmic step in §7.15.4.1
in the HTML).

---

## 1. Section map

| Clause | Title | Why M0 needs it |
|---|---|---|
| §4.11 | Descriptors | `f(n)`, `uvlc()`, `le(n)`, `ns(n)`, `L(n)`, `leb128()` |
| §5.2 | OBU syntax | header bits, payload sizing, trailing bits, byte alignment |
| §5.4.1–5.4.10 | Sequence header | every tool on/off switch |
| §5.18.1–5.18.10 | Frame header | the 22 bits M0 writes, and why the rest is absent |
| §5.19 | Tile group OBU | `is_first_tile_group`, frame header placement |
| §5.20.1 | Tile group payload | tile size fields, `init_symbol`/`exit_symbol` |
| §5.20.2–5.20.5 | Tile/partition/block/mode | partition + mode syntax |
| §5.20.6 | Transform size | `TX_MODE_LARGEST` short-circuit |
| §5.20.7.23–.28 | Residual, coeffs, `read_quant` | coefficient coding |
| §5.20.8.2–.3 | Transform type | why luma codes none and chroma codes one |
| §5.20.9.1 | `is_inside` | tile-relative availability |
| §6.2 | OBU semantics | OBU type table |
| §6.4 | Sequence header semantics | `bit_depth_idc` inversion, level, still picture |
| §6.17 | Frame header semantics | field meanings |
| §6.18–6.19 | Tile group semantics | conformance requirements on tile order/sizes |
| §7.5 | Frame end CDF update | next-frame CDF state |
| §7.13.2.10 | DC intra prediction | the only prediction process M0 uses |
| §7.14.2 | `get_dc_quant` / `get_ac_quant` | qlookup |
| §7.14.3–.4 | reconstruct / dequant | dequant scaling |
| §7.15.2.1 | 1D inverse transform | kernel application + shifts |
| §7.15.4–.4.1 | 2D inverse transform | row/col passes, `Transform_Shift` |
| §8.2.2–8.2.6 | Symbol decoder | range decoder, CDF update |
| §8.2.3, §8.2.5 | `read_bool`, `read_literal` | raw bits share the range coder |
| §8.3.1–8.3.2 | CDF selection | every symbol→CDF mapping |
| §9.2 | Conversion tables | block/tx/scan/context tables |
| §9.3 | Default CDF tables | 180 arrays, PDF only |
| §9.6 | 1D transform tables | DCT/ADST kernels, PDF only |
| Annex A.2/A.4 | Profiles/levels | profile 0, level index mapping |
| Annex B | Length-delimited OBU format | `leb128()` length prefix |

Not needed for M0 (and deliberately untouched): §5.5–5.17 (MSDO, LCR, atlas, OPS,
BRT, QM OBU, film grain OBU, metadata), §5.18.5.1 (interpolation filter), §5.18.9
(global motion), §5.20.7.1–.22 and .29–.32 (inter prediction, CFL), §5.20.8.1/.4/.5
(palette), §5.20.10 (in-loop filters), §7.9–7.12 (motion fields/MV prediction),
§7.13.3+ (inter), §7.16–7.20 (filters), §7.21.7 (film grain), Annex C–G.

---

## 2. OBU layer

- Length-delimited stream: `leb128() num_bytes_in_obu; open_bitstream_unit(n)`
  (Annex B.2). `num_bytes_in_obu` counts the whole OBU including its 1-byte header
  (and extension bytes if `obu_header_extension_flag = 1`).
- `obuPayloadSize = sz - 1 - obu_header_extension_flag` (§5.2.1).
- Header: `obu_header_extension_flag f(1)`, `obu_type f(5)`, `obu_tlayer_id f(2)`,
  then optional `obu_mlayer_id f(3)`, `obu_xlayer_id f(5)` (§5.2.2).
- With the extension flag 0, `obu_xlayer_id` is inferred to
  `GLOBAL_XLAYER_ID` for `OBU_TEMPORAL_DELIMITER`/`OBU_MSDO`, else 0. So a TD OBU
  differs from other OBUs in its inferred layer.
- Every non-tile OBU payload ends with `trailing_bits()`: a 1 bit then zeros
  (§5.2.1, §5.2.3). Tile-group payloads end with the range coder's own trailing
  bits inside the last tile (§8.2.4).
- `byte_alignment()` writes only zero bits (§5.2.4) — there is no "alignment one
  bit" as in some other codecs.
- OBU types (§6.2.2): 1 seq header, 2 temporal delimiter, 4 closed-loop key,
  7 regular tile group, 25 padding, 26–31 reserved. Reserved OBUs must be ignored.

M0 order: TD → SEQ → CLK (§7.3.3). One frame only.

---

## 3. Sequence header — the parts that matter

Full field list with values is in `docs/minimal-bitstream.md` §3. Semantics:

- `single_picture_header_flag = 1` is the single biggest simplifier. It forces
  (§5.4.1, §5.4.6, §5.4.7, §5.4.8, §5.4.10):
  - `still_picture = 1`, `monotonic_output_order_flag = 1`,
  - `max_tlayer_id = max_mlayer_id = 0`, `SeqMaxMlayerCnt = 1`,
  - `seq_lcr_id = 0`,
  - `OrderHintBits = 0`, `NumRefFrames = 2`, `enable_global_motion = 0`,
  - `seq_force_screen_content_tools = SELECT_SCREEN_CONTENT_TOOLS`,
  - `enable_avg_cdf = 1`, `avg_cdf_type = 1`,
  - no `enable_inter_ddt`, no `choose_tcq_per_frame`,
  - `CdefOnSkipTxfm = CDEF_ON_SKIP_TXFM_ADAPTIVE`,
  - `gdf_frame_enable = 1` / `cdef_frame_enable = 1` / `ccso_frame_flag = 1` /
    `apply_grain = 1` **if** the corresponding `enable_*` is 1. Therefore set
    `enable_gdf = enable_cdef = enable_ccso = 0` and `film_grain_params_present = 0`,
    otherwise the frame header gains unconditional tool syntax.
- `bit_depth_idc`: **0 = 10-bit, 1 = 8-bit** (§6.4.1 table). Also sets
  `MaxQ = MAXQ_8_BITS = 255` for 8-bit.
- `chroma_format_idc`: 0 = 4:2:0, 1 = 4:0:0, 2 = 4:4:4, 3 = 4:2:2 (§6.4.1).
- `seq_level_idx` 0..31 maps to levels 2.0 … 8.3; 31 = "maximum parameters"
  (Annex A.4). Level 5.1 = index 7.
- `seq_tier` is only present when `seq_level_idx > 3 && !single_picture_header_flag`.
- Tile constraints are level-dependent only if `seq_level_idx != 31`
  (§5.18.7.3); with 31 the max tile size is the frame itself.
- `seq_tile_info_present_flag = 0` means "tile params may be signalled at frame
  level, or default to a single tile covering the frame" (§6.4.2). With M0's
  single tile, the frame-level `tile_params()` still codes
  `uniform_tile_spacing_flag` (1 bit) and then no increments.

Constants worth pinning (`spec/tables/constants.csv`): `DELTA_DCQUANT_BITS = 5`,
`DELTA_DCQUANT_MAX = 8`, `DELTA_DCQUANT_MIN = -23`, `MAXQ_8_BITS = 255`,
`MAX_TILE_WIDTH = 4096`, `MAX_TILE_AREA = 9437184`, `MAX_TILE_COLS/ROWS = 64`.

---

## 4. Frame header — the 22 bits and the state they set

Order verified against §5.18.1/§5.18.2; see `docs/minimal-bitstream.md` §4 for the
table. State that later code depends on:

- `SbSize = BLOCK_64X64` (`use_256x256_superblock = use_128x128_superblock = 0`,
  §5.18.2).
- `FrameIsIntra = 1`, `FrameType = KEY_FRAME`, `keyFrame = 1`.
- `frame_size_override_flag = 0` → `FrameWidth = max_frame_width_minus_1 + 1`.
- `primary_ref_frame = PRIMARY_REF_NONE` → `set_primary_ref_frame_and_ctx(1)`
  calls `init_non_coeff_cdfs()` + `setup_past_independence()`, then
  `init_coeff_cdfs()` (§5.18.2, §6.17.2).
- `CodedLossless` is computed per segment: 1 only if every qindex is 0 and all
  deltas are ≤ 0 (§5.18.2). `base_q_idx >= 1` keeps `CodedLossless = 0`.
- `tx_mode_select = 0` → `TxMode = TX_MODE_LARGEST` (§5.18.8.1).
- Deblocking always codes `apply_deblocking_filter[0]` and `[1]` when not
  lossless; `[2]`/`[3]` only if either luma flag is 1 (§5.18.5.2).
- `reduced_tx_set` is always 2 bits in the frame header (§5.18.2).

---

## 5. Tile group and payload

- `tile_group_obu()` (§5.19) carries the frame header when
  `is_first_tile_group = 1`. The frame header is **not** a separate OBU.
- `is_first_tile_group` and `SeenFrameHeader` must differ (§6.18).
- `tg_start`/`tg_end` only when `NumTiles > 1` and
  `tile_start_and_end_present_flag = 1`; width `TileColsLog2 + TileRowsLog2`.
- Each tile: optional `le(TileSizeBytes) tile_size_minus_1`, then
  `init_symbol(tileSize)`, `decode_tile()`, `exit_symbol()`.
- Last tile in a group: size implied by the remaining OBU bytes; includes the OBU
  trailing bits.
- `frame_end_update_cdf()` + `decode_frame_wrapup()` after `tg_end == NumTiles-1`
  (§5.20.1, §7.5, §7.2).

---

## 6. Entropy coder (the part that must be perfect)

### 6.1 Initialisation (§8.2.2)

```
numBits   = Min(sz * 8, 15)
buf       = f(numBits)                      # MSB-first
paddedBuf = buf << (15 - numBits)
SymbolValue = ((1 << 15) - 1) ^ paddedBuf
SymbolRange = 1 << 15
SymbolMaxBits = 8 * sz - 15
```

`SymbolMaxBits` may go negative; negative means zero padding bits were consumed.
A fresh copy of every CDF array is made into its `Tile*` names at this point.

### 6.2 Symbol decode (§8.2.6)

```
FrameSymbolCount++
cur = SymbolRange
symbol = -1
do {
    symbol++
    prev = cur
    f = (symbol == N-1) ? 0 : (1 << 15) - cdf[symbol]
    pp = ((f >> EC_PROB_SHIFT) << 4) + Prob_Inc[N-2][symbol]   # EC_PROB_SHIFT = 7
    cur = (((SymbolRange >> 8) * pp) >> 7) << 3
} while (SymbolValue < cur)

newRange = prev - cur
newValue = SymbolValue - cur
bits     = 15 - FloorLog2(newRange)
SymbolRange = newRange << bits
numBits  = Clip3(0, bits, SymbolMaxBits)
newData  = f(numBits)
paddedData = newData << (bits - numBits)
mask     = (1 << bits) - 1
SymbolValue = (newValue << bits) | (paddedData ^ mask)
SymbolMaxBits -= bits
```

`Prob_Inc[7][8]` is in `spec/tables/block_tables.json`.

### 6.3 Raw bits (§8.2.3, §8.2.5)

```
read_bool():
    cur = SymbolRange >> 1
    symbol = SymbolValue < cur
    if (symbol == 0) SymbolValue -= cur
    numBits = Clip3(0, 1, SymbolMaxBits)
    newData = f(numBits)
    SymbolValue = (SymbolValue << 1) | (newData ^ 1)
    SymbolMaxBits -= 1
read_literal(n): x = 0; repeat n: x = 2*x + read_bool()
```

`L(n)` uses `read_literal(n)`. This is why raw sign bits and `eob_extra` bits are
part of the same arithmetic stream.

### 6.4 CDF update (§8.2.6)

```
timeInterval = cdf[N] > 31 ? 2 : cdf[N] > 15 ? 1 : 0
rate = 3 + timeInterval + Min(FloorLog2(N), 2)
     + Para_Adjustment_List[cdf[N-1]][timeInterval]
for i in 0..N-2:
    cdf[i] += (i < symbol) ? -(cdf[i] >> rate)
                           : (((1 << 15) - cdf[i]) >> rate)
cdf[N] += (cdf[N] < 32)
```

- `cdf[N-1]` is the constant base rate (from the default table), `cdf[N]` the
  adaptation count (0..32).
- `Para_Adjustment_List[125][3]` is in `spec/tables/block_tables.json`.
- If `disable_cdf_update = 1`, skip the update block entirely.

### 6.5 Exit (§8.2.4)

- Conformance: `SymbolMaxBits >= -14`.
- `trailingBitPosition = get_position() - Min(15, SymbolMaxBits + 15)`.
- Advance by `Max(0, SymbolMaxBits)`.
- The bit at `trailingBitPosition` must be 1; bits between it and
  `paddingEndPosition` must be 0.
- The encoder must emit exactly that: a 1 then zeros, so that the decoder's
  `SymbolMaxBits` accounting lands on the right bit.

This is the trickiest part of writing (as opposed to reading) the format: the
encoder has to know how many renorm bits it "would have read" and place the
terminator accordingly. Practical approach: implement the decoder's bookkeeping in
the encoder (track `SymbolMaxBits`/bit position as if decoding) and verify by
decoding your own output.

### 6.6 CDF selection (§8.3.2)

Each `S()` symbol maps to a CDF array via §8.3.2. The entries M0 uses:

| Symbol | CDF | Extra indices |
|---|---|---|
| `do_split` | `TileDoSplitCdf[PlaneStart][ctx]` | `ctx = Partition_Size_Adjust[bSize]*4 + ctx1*2 + ctx2` |
| `do_square_split` | `TileDoSquareSplitCdf[PlaneStart][ctx]` | `ctx = (bSize==256x256?4:0)+left*2+above` |
| `y_mode_set` | `TileYModeSetCdf` | — |
| `y_mode_index` | `TileYModeIndexCdf[ctx]` | `ctx` from joint modes of above-left/left |
| `uv_mode` | `TileUVModeCflNotAllowedCdf[ctx]` | `ctx = is_directional_mode(YMode)` |
| `all_zero` | plane 0/1: `TileTxbSkipCdf[is_inter\|\|fsc_mode][txSzCtx][ctx]`; plane 2: `TileVTxbSkipCdf[ctx]` | plane 0 uses the `bw==w && bh==h → 0` shortcut; plane 1 `6+above+left`; plane 2 `above+left [+3] [+6 if EobU]` |
| `eob_pt_1024` | `TileEobPt1024Cdf[eobCtx]` | `eobCtx = 0` (intra luma) |
| `eob_pt_256` | `TileEobPt256Cdf[eobCtx]` | `eobCtx = 2` (chroma) |
| `eob_extra` | `TileEobExtraCdf` | — |
| `intra_tx_type` | `TileIntraTxTypeSet1Cdf[Tx_Size_Sqr[txSz]]` | only chroma 16x16 for M0 |
| `coeff_base_eob` | luma `TileCoeffBaseEobCdf[txSzCtx][ctx]`; chroma `TileCoeffBaseEobUvCdf[txSzCtx][ctx]` | `ctx = 0` at DC |
| `coeff_base` | luma `TileCoeffBaseCdf[txSzCtx][hfCtx][tcqState>>1]`; chroma `TileCoeffBaseUvCdf[txSzCtx][uvCtx]` | `isLf` selects `TileCoeffBaseLfCdf` |
| `coeff_br` | luma `TileCoeffBrCdf[ctx]` / `TileCoeffBrLfCdf[ctx]`; chroma `TileCoeffBrUvCdf[ctx]` | — |
| `dc_sign` | `TileDcSignCdf[ptype][isHidden][ctx]` | luma DC only |

Every coefficient CDF is first sliced by the quantiser context
`idx = base_q_idx<=90 ? 0 : <=140 ? 1 : <=190 ? 2 : 3` at
`init_coeff_cdfs()` (§6.17.2). §8.3.2 omits that leading index; see
`docs/open-questions.md`.

---

## 7. Intra DC prediction (§7.13.2.10)

```
if (haveLeft && haveAbove)  pred = Clip1(approx_divide(sum(LeftCol[0..h-1]) +
                                                       sum(AboveRow[0..w-1]), w+h))
else if (haveLeft)          pred = Round2(sum(LeftCol[0..h-1]), log2H)
else if (haveAbove)         pred = Round2(sum(AboveRow[0..w-1]), log2W)
else                        pred = 1 << (BitDepth-1)          # 128 for 8-bit
```

`approx_divide` uses `resolve_divisor()` and `Round2` (§7.13.2.11, §7.13.2.7).
`haveLeft`/`haveAbove` come from `is_inside()` (§5.20.9.1), which is
**tile-relative**, so the first block of a tile behaves as if it had no neighbours.

Reference sample arrays are built in §7.13.2.1 with a non-obvious default:
when a neighbour is missing but the other exists, the missing side is filled with
the *present* side's edge sample; when both are missing, `AboveRow` is filled with
`(1<<(BitDepth-1)) - 1 = 127` and `LeftCol` with `128`. DC prediction ignores these
values, but they matter for the availability-driven branches above — so implement
`haveLeft`/`haveAbove` exactly, not "both or neither".

Chroma DC with `w,h <= 32` uses the plain DC process, not the subsampled variant
(§7.13.2.1 dispatch).

---

## 8. Transform and quantisation

- `get_dc_quant`/`get_ac_quant` (§7.14.2) → `get_q(qindex, delta)` → `qlookup()`.
  `Ac_Qlookup[25]` is the only step table; larger indices are a
  shift-by-`(q-1)/24` extrapolation.
- `dequant()` (§7.14.4) clips to `±(1 << (7 + BitDepth))` and divides by
  `dqDenom = 1 << shift`, `shift = (pels > 256) + (pels > 1024)` plus 1 if
  `allow_tcq` applies (M0: no). 32x32 → shift 1; 16x16 → shift 0.
- `Adjusted_Tx_Size` (§9.2) is identity for all our sizes (used for 64-length
  transforms that are coded as 32 + duplication; not M0).
- 2D inverse transform (§7.15.4.1): row pass with `Transform_Shift[txSz][0]`,
  column pass with `[1]`; clip limits `±(1 << (BitDepth + 7))` for the row pass and
  `±(1 << BitDepth)` for the column pass; `Round2` at each stage.
- `Dct_Kernel32[j][i]`: row index `j` is the input coefficient, column index `i`
  the output sample. Do not transpose it by accident.
- The 1D DCT is a plain matrix multiply; there is no integer lifting butterfly in
  the normative text. That is good for correctness and bad for speed; a fast
  implementation must be bit-exact against the matrix form.

---

## 9. Things that bite

1. **`bit_depth_idc` is inverted** (1 = 8-bit).
2. **`uvlc()` writes lz zeros, then a 1, then lz payload bits** — not "1 then lz".
3. **`byte_alignment()` writes zeros**, no marker bit.
4. **The frame header is inside the tile-group OBU.**
5. **`TileTxbSkipCdf` has a hidden leading quantiser-context index** at init.
6. **Chroma codes an `intra_tx_type` even for DCT-only**; luma 32x32 does not.
7. **DC sign is adaptive for luma, raw `L(1)` for chroma and all AC.**
8. **The `& 0xFFFFFF` in dequant is a 24-bit mask**, applied before `Round2(...,3)`.
9. **`is_inside()` is tile-relative**, `is_inside_frame()` is not.
10. **`read_bool()` shares the range coder state with `read_symbol()`**, so `L(n)`
    cannot be written as a separate bitstream.
11. **`exit_symbol()` requires an exact trailing 1 + zeros pattern** derived from
    `SymbolMaxBits`; getting it wrong corrupts the last few coefficients or the
    next OBU's parsing.
12. **CDF tables exist only in the PDF.** Do not trust the HTML for §9.3/§9.6.
13. **HTML §7.15.4.1 omits the row-transform step** that the PDF has. Use the PDF.

---

## 10. Deliberate M0 omissions

These are *in scope to skip* for M0, not unknown:

- Any inter coding, TIP, BRU, RAS, SEF, multistream/LCR/Atlas/OPS.
- Multiple tiles, multiple tile groups, multiple frames.
- Loop filters: deblock, CDEF, GSF/GDF, CCSO, loop restoration, PC-Wiener.
- Film grain (synthesis and OBU).
- Palette, intra block copy, CfL, MHCCP, IBP, DIP, MRL, FSC, IST, DDT, TCQ,
  parity hiding, CCTX.
- Segmentation, delta-Q, quantiser matrices (built-in or user), `reduced_tx_set`
  other than 0, transform partitioning.
- 4:0:0 / 4:2:2 / 4:4:4, 10/12-bit, 256x256 superblocks, SDP, extended partitions.
- Encoder-side rate control, mode search, RDO. M0 is fixed-mode.
- Decoder model / timing / levels beyond writing a sane `seq_level_idx`.

Recorded here so a later reader does not mistake them for spec gaps.

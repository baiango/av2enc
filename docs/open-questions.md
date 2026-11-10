# Open questions, ambiguities, and HTML/PDF disagreements

Pinned spec: `spec/AV2_Spec_v1.0.0.pdf` (v1.0.0, 2026-05-28, commit `38f28e7`,
SHA-256 `e9916f09…b533f`). HTML: <https://av2.aomedia.org/v1.0.0/>.

Each item states the impact on M0 and what would settle it. "Blocker" means it can
stop a first valid stream; "risk" means a wrong guess produces a stream that parses
differently than intended.

---

## A. HTML vs PDF disagreements

### A1. §9.3 default CDF tables are absent from the HTML — BLOCKER (resolved)

The HTML §9.3 contains 180 paragraphs of the form
"`Default_Mv_Joint_Adaptive_Cdf` is defined in the
`default_mv_joint_adaptive_cdf.h` header file." There is **no numeric data**.
The PDF prints all 180 arrays in full.

Resolution: `tools/spec_extract/extract_tables.py` parses them from the PDF.
Generated: `spec/tables/default_cdf.json` (all 180) and
`spec/tables/m0_cdf.json` + `m0_cdf.h` (the 19 used by M0).
Verified structurally: every array's element count equals the product of its
declared dimensions, cumulative values are non-decreasing, adaptation counts are 0.

Same pattern for §9.2 (conversion tables): the HTML says "is defined in
`mi_width_log2.h`"; the PDF prints the numbers.

### A2. §9.6 1D transform tables are absent from the HTML — BLOCKER (resolved)

HTML: "`Dct_Kernel32` is defined in the `dct_kernel32.h` header file."
PDF: full 32x32 integer matrix (and 4/8/16, ADST, FDST, DDTX).
Resolution: parsed from the PDF into `spec/tables/dct_kernels.json` and
`spec/tables/dct32.h`.

### A3. §7.15.4.1 (2D matrix transform) is missing the row-transform step in HTML — BLOCKER (resolved)

HTML text goes from "txRowIn[j] is derived as follows…" straight to "The following
applies for j = 0..(w-1):" with the column transform, omitting the step that
produces `intermediate[i][j]`. The PDF (page 527/528) has:

```
The row transform is applied as follows:
  if Lossless: inverse WHT(txRowIn, shift=3) -> txRowOut
  else if rowType == IDT: inverse identity(txRowIn, get_identity_scale(log2W), w, rowShift, 0)
  else: 1d matrix transform(txRowIn, rowType, w, rowShift, 0) -> txRowOut
Set intermediate[i][j] = txRowOut[j] for j = 0..(w-1)
```

Resolution: implement from the PDF. Encoder reconstruction depends on this; a
missing row pass would produce a completely different residual.

### A4. HTML anchors/layout artifacts — cosmetic

The HTML `data-original-syntax` attribute injects `<p bs-line-number=NNNN>` tags at
a few table boundaries (e.g. §5.18.2 around `TipFrameMode`, §5.19 `tile_info`,
§5.20.3.2 `read_tx_partition`). These are rendering artifacts, not syntax. The
prose and pseudocode around them are consistent with the PDF.

### A5. PDF two-column interleaving — parser hazard (resolved)

Several §9.2 arrays have prose paragraphs injected mid-initializer
(`Tx_Size_Sqr_Up` is the worst: four "For a transform size t…" lines). The PDF text
extractor also emits page headers inside arrays. Both are handled by
`tools/spec_extract/specpdf.py` (page furniture removal + "keep only lines that are
pure C initializer" filter), and the dimension check in §A1 catches any silent
truncation. Anyone regenerating tables should keep that check.

---

## B. Spec-internal ambiguities

### B1. Coefficient CDFs lose their quantiser-context dimension in §8.3.2 — RISK

The default arrays are declared `[COEFF_CDF_Q_CTXS][…]` (e.g.
`Default_Coeff_Base_Cdf[4][5][20][2][5]`), but §8.3.2 writes the selection as
`TileCoeffBaseCdf[txSzCtx][hfCtx][(tcqState>>1)&1]` — three indices for a
five-dimensional array. Likewise `TileTxbSkipCdf[is_inter||fsc_mode][txSzCtx][ctx]`
for a `[4][2][5][10][3]` array, and `TileDcSignCdf[ptype][isHidden][ctx]` for
`[4][2][2][3][3]`.

The only consistent reading is that `init_coeff_cdfs()` (§6.17.2) slices the
leading `idx` dimension once:

```
idx = base_q_idx <= 90  ? 0 :
      base_q_idx <= 140 ? 1 :
      base_q_idx <= 190 ? 2 : 3
```

and the frame-level arrays are the 4-D remainder. **This is an inference from
consistent usage, not an explicit sentence.** Impact: if wrong, every coefficient
symbol is decoded with the wrong CDF.

Mitigation: `base_q_idx = 90` selects slice 0 for the first test stream; we can
then try 91 (slice 1) and see whether dav2d agrees — a cheap differential test.

### B2. `TileTxbSkipCdf` for plane 2 uses a differently-shaped table — RISK

Plane 2 (`V`) uses `TileVTxbSkipCdf` (`Default_V_Txb_Skip_Cdf[4][12][3]`), not
`TileTxbSkipCdf` (§8.3.2). Its context derivation is the same but has
`V_TXB_SKIP_CONTEXTS = 12` instead of 10, and adds `+6` when `EobU != 0` and `+3`
when the chroma block is larger than the transform block. Easy to miss; a
first-stream failure on V-only coefficients points here.

### B3. `TileIntraTxTypeSet1Cdf[Tx_Size_Sqr[txSz]]` uses a tx-size enum as an index — OK but confusing

`Default_Intra_Tx_Type_Set1_Cdf[3][8]` has 3 rows but `Tx_Size_Sqr[txSz]` for
16x16 is 2 (TX_16X16). Rows 0..2 correspond to TX_4X4/TX_8X8/TX_16X16. For M0
chroma this is row 2, 7 symbols (8 = N+1). Luma 32x32 never reaches this code
(`TX_SET_DCTONLY`). No ambiguity, but the indexing is by *square tx enum*, not by
log2.

### B4. `read_partition()`'s `bru_mode` early return — OK, but easy to misread

§5.20.3.2 contains `if ( bru_mode != BRU_ACTIVE ) return PARTITION_NONE;`. With
`use_bru = 0` and `BruTileActive = 0`, `decode_tile()` sets
`bru_mode = BRU_ACTIVE`, so the normal path is taken. If BRU is ever enabled,
a whole tile can become syntax-free. Not an M0 issue.

### B5. `partition_implied()`'s `bSize >= BLOCK_4X32` test is enum-order dependent — RISK

`if ( bSize == BLOCK_4X4 || bSize >= BLOCK_4X32 ) return (1, PARTITION_NONE);`
Only makes sense with the enum order of Table 6.22, where `BLOCK_4X32` is 25 and the
large square sizes are 9..18. Therefore `BLOCK_32X32` (9) and `BLOCK_64X64` (12) are
**not** implied and do code a partition. If you assume "large blocks are implied
NONE", you produce a stream with no partition bins where the decoder expects two.

Verified against the enum in `spec/tables/block_tables.json` (`enum_block_size`).

### B6. `still_picture` vs `single_picture_header_flag` — no ambiguity, just two flags

`single_picture_header_flag` *forces* `still_picture = 1` and removes syntax; it is
not read from the bitstream in that case (§5.4.1). A sequence header with
`single_picture_header_flag = 1` is a complete still-picture header.

### B7. `seq_level_idx = 31` semantics — POLICY, not ambiguity

§A.4: 31 = "maximum parameters"; "the decoder should examine the properties of the
bitstream". Legal, but shifts conformance checking to the decoder. M0 uses 7 (5.1).

### B8. `uvlc()` conformance note — no ambiguity

"leadingZeros is less than 32 when this function returns" (§4.11.3). Our values are
0 and 1, so `1` and `010` respectively.

### B9. `ns(n)` for `seq_max_bvp_drl_bits_minus_1` — no ambiguity

`MAX_REF_BV_STACK_SIZE = 4` → `ns(3)`; value 0 is the single bit `0` (§4.11.8).
Unused at M0 (`allow_intrabc = 0`) but still present in the sequence header.

### B10. `Transform_Shift` has 25 rows but is indexed by `txSz` — OK

Rows 0..24 follow `TX_SIZES_ALL`. M0 needs rows 2 (TX_16X16) and 3 (TX_32X32), both
`{6, 13}`. Verified in `spec/tables/block_tables.json`.

### B11. Dequant `& 0xFFFFFF` — unclear intent, clear operation — RISK (low)

§7.14.4: `dq = Round2(dqHigh & 0xFFFFFF, QUANT_TABLE_BITS)`. The mask truncates
`|qc| * q` to 24 bits before rounding. For 8-bit streams with `q ≤ 61440` (q=255)
and `|qc|` up to a few thousand, the product can exceed 2^24 (e.g. 61440 * 1000 =
6.1e7 > 1.6e7), so the mask is *reachable* and must be implemented literally.
No note explains it; do not "simplify" it away.

### B12. `read_quant()`'s Golomb-style escape — not reachable at M0 but documented

When `level >= maxLevel` the decoder reads a unary `q_length_bit` loop, possibly a
`golomb_length_bit` unary loop, then `coeff_rem`. §5.20.7.28. M0 should clamp
coefficients so this path is never taken (luma LF `maxLevel = 8`, chroma LF 5,
HF luma 6, HF chroma 5), or implement it exactly. Not an ambiguity, flagged as a
place where a too-large coefficient silently changes the bitstream.

---

## C. Decoder / tooling unknowns (dav2d `f9dafbc`)

### C1. dav2d command line — RESOLVED

Confirmed against the pinned binary (`tools/bin/dav2d-macos`, `--version` →
`f9dafbc`): the CLI is dav1d-compatible.

```
dav2d -q -i in.ivf -o out.y4m --threads 0
dav2d -q -i in.ivf --muxer md5 -o -
```

`tools/validate_dav2d.sh` uses exactly these. See `docs/validation.md`.

### C2. IVF fourcc — RESOLVED

The pinned dav2d decodes `testdata/sample_480x270_130f.ivf`, whose header reports
`fourcc AV02`. The fourcc is an IVF convention, not part of the AV2 spec; `AV02`
is what the working sample uses, so the M0 muxer should write `AV02`.

### C3. Multi-tile support — REPORTED OK, unverified

The project states dav2d `f9dafbc` decodes 1x1, 2x1, 2x2, 4x2 and 8x2. Not verified
here. M0 stays 1x1. See `docs/tiles.md` §9.

### C4. `disable_loopfilters_across_tiles` with filters on — REPORTED GAP

One public port's notes say dav2d ignores this flag in some filter paths. M0 has no
filters, so no impact. **TODO** before enabling filters with tiles.

### C5. `avg_cdf_type = 1` / per-tile CDF averaging — unverified

With `single_picture_header_flag = 1`, `avg_cdf_type = 1` is forced and
`context_update_tile_id` is absent. For a single frame nothing observable depends on
the averaging. If a second frame is added, `Saved*` arrays matter; whether dav2d
implements `TileNum < (1 << numLog2)` averaging exactly is unverified. **TODO**.

### C6. Level 31 acceptance — unverified

Legal per §A.4, but decoder policy. Not used by M0.

### C7. 10-bit and non-4:2:0 — out of scope

Public port notes report 10-bit bit-exact; 4:0:0/4:2:2/4:4:4 and resize were not
matched against the AOM reference decoder in that port at the time of writing.
M0 is 8-bit 4:2:0 only.

---

## D. Deliberate M0 omissions (not spec gaps)

Recorded so they are not mistaken for unknowns. Each is switched off by an explicit
sequence/frame-header field and can be enabled later:

- inter prediction, compound, TIP, BRU, RAS, SEF, MSDO/LCR/Atlas/OPS
- multiple tiles / tile groups / frames
- deblocking, CDEF, GDF, CCSO, loop restoration (incl. PC-Wiener), film grain
- palette, intra block copy (local and global), CfL, MHCCP, IBP, DIP, MRL,
  MRL-secondary, FSC, IST, DDT, TCQ, parity hiding, CCTX
- segmentation, delta-Q, quantiser matrices (built-in and user-defined),
  `reduced_tx_set != 0`, transform partitioning, SDP, extended/uneven partitions
- 256x256 and 128x128 superblocks, 4:0:0/4:2:2/4:4:4, 10/12-bit
- rate control, mode search, RDO (M0 is fixed DC + DCT32)

`docs/tiles.md` covers the first item on that list and is explicitly *not* an M0
scope change.

---

## E. Verification backlog

Ordered by value:

1. Decode `out.ivf` with dav2d; confirm one frame, correct size, non-zero output.
2. Compare our own reconstruction with dav2d's output **bit-exactly** (not just PSNR).
   Any mismatch isolates to dequant/inverse-transform first, prediction second.
3. Flip `disable_cdf_update` to 1 and re-decode. If that works and 0 fails, the bug
   is CDF adaptation (§8.2.6) or `exit_symbol()` (§8.2.4).
4. Encode a flat grey 64x64 frame (all DC, `eob=1`) and a random-noise frame
   (`eob>1`, `coeff_br`, `read_quant` escapes). These two bracket the coefficient
   coder.
5. Encode 256x256 and check that `all_zero`/partition contexts still match.
6. Only then enable `base_q_idx = 91` to test the CDF qctx slice inference (B1).

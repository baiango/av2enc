# Spec audit 12 — core tables (§9.1–§9.3)

Scope: AV2 v1.0.0 §9.1 General, §9.2 Conversion tables, §9.3 Default CDF
tables — PDF pages 609–742 (pypdf page index = spec page − 1).

Read-only audit. No generator, build, or test was run; the checked-in
artifacts were compared against the pinned PDF text. This report is the only
file created.

Artifacts audited:

| Artifact | Role |
|---|---|
| `spec/tables/block_tables.json` | generated §9.2 conversion / block / partition / TX tables |
| `spec/tables/scan.json` | generated `get_scan()` (TX_CLASS_2D) scans |
| `spec/tables/default_cdf.json` | all 180 §9.3 default CDF arrays |
| `spec/tables/m0_cdf.json`, `spec/tables/m0_cdf.h` | 19-array M0 CDF subset |
| `src/entropy/av2/av2_entropy_tables.h` | generated `Prob_Inc`, `Para_Adjustment_List` |
| generated C++ headers under `src/encoder/**` | the actual encoder bindings |

Definition used below: a table is **present** when it exists in one of the
generated artifacts above. A table is **used** when at least one non-test
encoder/entropy path consumes it (directly, or through a generated C++
header generated from the JSON). “Used” does **not** assert that the owning
tool is enabled in the production M0/M1/M2 bitstream; it only means the table
is reachable from a code path in this tree.

## §9.1 General

No tables in §9.1 (page 609). Nothing to audit.

## §9.2 Conversion tables (pages 609–622)

There are **36** top-level arrays in §9.2. `block_tables.json` contains 24 of
them. All 24 have the right dimensions and match the PDF element-for-element.
Twelve are absent from the generated set; eight of those are hand-coded
elsewhere in `src/`, three are inlined as literals/assumptions, and one
(`Mode_To_Txfm`) has no consumer at all.

| Table | §/page | Present? | Dimensions | Used by | Status/Gap |
|---|---|---:|---|---|---|
| `Mi_Width_Log2` | §9.2 p.609 | Yes | `[29]` | `m1/m1_tables.h` `kMiWidthLog2`; `m1/partition_stream`; `sb128`/`sb256`; partition geometry | OK |
| `Mi_Height_Log2` | §9.2 p.609 | Yes | `[29]` | `m1/m1_tables.h` `kMiHeightLog2`; partition/TX geometry | OK |
| `Num_4x4_Blocks_Wide` | §9.2 p.609 | Yes | `[29]` | indirect: `format/format_tables.h` `kBlockWidth = 4 ×` | OK, indirect binding (no standalone C++ table) |
| `Num_4x4_Blocks_High` | §9.2 p.609 | Yes | `[29]` | indirect: `format/format_tables.h` `kBlockHeight = 4 ×` | OK, indirect binding |
| `Size_Group` | §9.2 p.610 | Yes | `[29]` | `m2/advanced/advanced_cdf_tables.h` `kSizeGroup`; `cfl/mhccp` | OK |
| `Max_Tx_Size_Rect` | §9.2 p.610 | Yes | `[29]` | `m1/m1_tables.h` `kMaxTxSizeRect`; `m1/sub8`; `format` | OK |
| `Partition_Subsize` | §9.2 p.610 | Yes | `[10][29]` | `partition_ext/ext_tables_generated.h` `kPartitionSubsize`; `sb128`/`sb256`; `m1_syntax` | OK |
| `H_Partition_Midsize` | §9.2 p.612 | Yes | `[29]` | `partition_ext/ext_tables_generated.h` `kHPartitionMidsize` | OK |
| `Mode_To_Txfm` | §9.2 p.612 | **No** | `[14]` | none found | **Absent; no consumer in the tree** |
| `Palette_Color_Context` | §9.2 p.612 | **No** | `[9]` | hand copy `palette/palette_core.h` `kPaletteColorContext` | Absent from generated set; local copy is used |
| `Palette_Color_Hash_Multipliers` | §9.2 p.613 | **No** | `[3]` | inlined as `1, 2, 2` in `palette_core.h` | Absent; value inlined, comment only |
| `Mode_To_Angle` | §9.2 p.613 | Yes | `[13]` | `tx_secondary/tx_secondary_tables.h`; `cuda/intra_modes` | OK |
| `Dr_Intra_Derivative` | §9.2 p.613 | **No** | `[90]` | hand copy `cuda/intra_modes/intra_modes.h` macro | Absent from generated set; local copy is used |
| `Side_Thresholds` | §9.2 p.613 | **No** | `[296]` | hand copy `filters/deblock/deblock_core.h` `kSideThresholds` | Absent from generated set; local copy is used |
| `Tx_Size_Sqr` | §9.2 p.614 | Yes | `[25]` | `m1/m1_tables.h`; `tx_type`; `tcq/parity` | OK |
| `Q_First` | §9.2 p.614 | **No** | `[9]` | hand copy `filters/deblock/deblock_core.h` `kQFirst` | Absent from generated set; local copy is used |
| `W_Mult` | §9.2 p.614 | **No** | `[8]` | hand copy `filters/deblock/deblock_core.h` `kWMult` | Absent from generated set; local copy is used |
| `Q_Thresh_Mults` | §9.2 p.614 | **No** | `[8]` | hand copy `filters/deblock/deblock_core.h` `kQThreshMults` | Absent from generated set; local copy is used |
| `Tx_Size_Sqr_Up` | §9.2 p.614 | Yes | `[25]` | `m1/m1_tables.h`; `tx_type` | OK |
| `Tx_Width` | §9.2 p.615 | Yes | `[25]` | `tx_secondary`; `tx_variants`; `deblock`; `tcq/parity` | OK |
| `Tx_Height` | §9.2 p.615 | Yes | `[25]` | same consumers as `Tx_Width` | OK |
| `Tx_Width_Log2` | §9.2 p.615 | Yes | `[25]` | `tx_secondary`; `tx_variants`; eob multisize | OK |
| `Tx_Height_Log2` | §9.2 p.615 | Yes | `[25]` | same consumers as `Tx_Width_Log2` | OK |
| `Wedge_Bits` | §9.2 p.615 | **No** | `[29]` | hand copy `inter/advanced/tables/wedge_tables.h` `kWedgeBits` | Absent from generated set; local copy is used |
| `Sig_Ref_Diff_Offset` | §9.2 p.615 | Yes | `[3][5][2]` | only TX_CLASS_2D row hand-coded in `lossless.cpp` as `kSigRefDiff[5][2]`; m2 residual comments | **Present but only partially used; full table has no consumer** |
| `Adjusted_Tx_Size` | §9.2 p.616 | Yes | `[25]` | behavior computed inline in `format/chroma/chroma_residual.cpp` (`min(32, w/h)`) | **Present but table is not bound/consumed** |
| `Size_To_Tx_Part_Group_Lookup` | §9.2 p.616 | Yes | `[29]` | none found | **Present; unused** |
| `Size_To_Tx_Type_Group_Vert_And_Horz` | §9.2 p.616 | Yes | `[29]` | none found | **Present; unused** |
| `Size_To_Tx_Type_Group_Vert_Or_Horz` | §9.2 p.617 | Yes | `[29]` | none found | **Present; unused** |
| `Size_Class` | §9.2 p.617 | Yes | `[25]` | `tx_type/tx_type_tables.h`; `m1/m1_tables.h` | OK |
| `Md_Idx_To_Type` | §9.2 p.617 | Yes | `[4][13][7]` | `tx_type/tx_type_tables.h` | OK |
| `Gaussian_Sequence` | §9.2 p.618 | **No** | `[2048]` | hand copy `filters/spec_tables/film_grain_tables.h`; film-grain component only, not wired into M0/M1/M2 | Absent from generated set; local copy is used |
| `Para_Adjustment_List` | §9.2 p.621 | Yes | `[125][3]` | `src/entropy/av2/av2_entropy_tables.h`; consumed by `av2_entropy.h` | OK (also generated independently from the PDF) |
| `Prob_Inc` | §9.2 p.622 | Yes | `[7][8]` | `src/entropy/av2/av2_entropy_tables.h`; consumed by `av2_entropy.h` | OK (also generated independently from the PDF) |
| `Tile_Width_Scaling_Factor` | §9.2 p.622 | **No** | `[2][31]` | only a comment in `bitstream/m0_headers.cpp`; value 4 assumed | **Absent; not generated, not a real consumer** |
| `Tile_Area_Scaling_Factor` | §9.2 p.622 | **No** | `[2][31]` | only a comment in `bitstream/m0_headers.cpp`; value 4 assumed | **Absent; not generated, not a real consumer** |

### Absent §9.2 tables — disposition

| Absent table | Local equivalent | Consumer | Risk |
|---|---|---|---|
| `Mode_To_Txfm` | none | none | No normative binding; not used by the current encoder |
| `Palette_Color_Context` | `kPaletteColorContext` in `palette_core.h` | palette encoding | Hand copy, not regenerated |
| `Palette_Color_Hash_Multipliers` | literal `1, 2, 2` | `palette_color_context()` | Value inlined; table not auditable |
| `Dr_Intra_Derivative` | macro in `cuda/intra_modes/intra_modes.h` | intra directional prediction | Hand copy, not regenerated |
| `Side_Thresholds` | `kSideThresholds` in `deblock_core.h` | deblocking / HBD | Hand copy, not regenerated |
| `Q_First` | `kQFirst` in `deblock_core.h` | deblocking | Hand copy, not regenerated |
| `W_Mult` | `kWMult` in `deblock_core.h` | deblocking | Hand copy, not regenerated |
| `Q_Thresh_Mults` | `kQThreshMults` in `deblock_core.h` | deblocking | Hand copy, not regenerated |
| `Wedge_Bits` | `kWedgeBits` in `wedge_tables.h` | inter wedge gating | Hand copy, not regenerated |
| `Gaussian_Sequence` | `kGaussianSequence` in `film_grain_tables.h` | film grain | Hand copy, not regenerated |
| `Tile_Width_Scaling_Factor` | literal 4 | `m0_headers.cpp` comment | Not a table in code |
| `Tile_Area_Scaling_Factor` | literal 4 | `m0_headers.cpp` comment | Not a table in code |

Note: the hand-copied values that were checked (`Side_Thresholds`, `Q_First`,
`W_Mult`, `Q_Thresh_Mults`, `Wedge_Bits`, `Gaussian_Sequence`,
`Palette_Color_Context`, `Dr_Intra_Derivative`) currently equal the PDF. The
gap is provenance/regeneration, not a value error.

## Additional materialized tables in `block_tables.json`

These are present in the generated artifact but are **not §9.2** tables. They
are listed because they are covered by the same file and are sometimes cited
as §9.2.

| Table | §/page | Present? | Dimensions | Used by | Status/Gap |
|---|---|---:|---|---|---|
| `Subsampled_Size` | §5.20.7.26 | Yes | `[29][2][2]` | `format/format_tables.h` `kSubsampledSize`; chroma residual | OK; `clause` field incorrectly says `9.2` |
| `Partition_Size_Adjust` | §8.3.2 | Yes | `[29]` | `m1/m1_tables.h`; `partition_ext` | OK; `clause` field says `9.2`; `generate_m1_tables.py` hardcodes the values instead of reading the JSON |
| `Partition_Size_Adjust_Rect_Type` | §8.3.2 | Yes | `[29]` | `m1/m1_tables.h`; `partition_ext` | Same as above |
| `Transform_Shift` | §7.15.4.1 | Yes | `[25][2]` | hardcoded branch logic in `format/chroma/chroma_residual.cpp` | **Present but table is unused**; `clause` field says `9.2` |
| `Reordered_Y_Mode` | §5.20.5 | Yes | `[13]` | `m1` / `mrl` | OK (out of §9.1–§9.3) |
| `Default_Mode_List_Y` | §5.20.5 | Yes | `[56]` | `m1` / `mrl` | OK (out of §9.1–§9.3) |
| `Default_Mode_List_Uv` | §5.20.5 | Yes | `[13]` | `m1` / `mrl` | OK (out of §9.1–§9.3) |
| `enum_block_size` | §6.19.3.1 | Yes | `[29]` | generators (`extract_tables.py`, `generate_format_tables.py`) | OK |
| `enum_tx_size` | §6.19.6.1 | Yes | `[25]` | generators | OK |

## Scan tables

There is **no literal scan table in §9.1–§9.3**. The normative scan is the
`get_scan()` function in §5.20.7.30 (page 523). `spec/tables/scan.json` is
therefore derived from that algorithm, not parsed from a table. It contains
four TX_CLASS_2D square scans and all four are used.

| Table | §/page | Present? | Dimensions | Used by | Status/Gap |
|---|---|---:|---|---|---|
| `scan.4` (diagonal 4×4) | §5.20.7.30 | Yes | `[16]` | `tx_type/tx_type_tables.h` `kScan4`; lossless also has a matching hand copy | OK |
| `scan.8` (diagonal 8×8) | §5.20.7.30 | Yes | `[64]` | `tx_type/tx_type_tables.h` `kScan8`; lossless hand copy | OK |
| `scan.16` (diagonal 16×16) | §5.20.7.30 | Yes | `[256]` | `m0_tables.h` `kScan16`; `tx_type` `kScan16` | OK |
| `scan.32` (diagonal 32×32) | §5.20.7.30 | Yes | `[1024]` | `m0_tables.h` `kScan32`; `tx_type` `kScan32` | OK |

All four are permutations of `0..N-1` and match a literal implementation of
`get_scan()`. **Gap:** the generated set covers only square TX_CLASS_2D scans.
`get_scan()` also defines TX_CLASS_VERT/HORIZ and rectangular sizes
(`4x8 … 64x4`); `scan.json` has none of those. The current encoder paths only
consume the square 2D scans, so this is a coverage limitation, not a wrong
value. Secondary-transform `Stx_Scan_Order_4x4/8x8` (§7.15.3) are a separate
out-of-scope table set, present in `tx_secondary_tables.h` and used.

## §9.3 Default CDFs (pages 623–742)

**Completeness:** the PDF contains exactly **180** `Default_*_Cdf` arrays in
§9.3; `spec/tables/default_cdf.json` contains exactly the same 180 names.
Every array's element count equals the product of its declared dimensions
(`COEFF_CDF_Q_CTXS`, `BLOCK_SIZES`, etc. resolved through
`spec/tables/constants.json`). Examples of the large coefficient tables:

| CDF | Dimensions | Values | Product |
|---|---|---:|---:|
| `Default_Coeff_Base_Cdf` | `[4][5][20][2][5]` | 4000 | 4000 |
| `Default_Coeff_Base_Lf_Cdf` | `[4][5][33][2][7]` | 9240 | 9240 |
| `Default_Coeff_Base_Lf_Eob_Cdf` | `[4][5][4][6]` | 480 | 480 |
| `Default_Coeff_Base_Ph_Cdf` | `[4][5][5]` | 100 | 100 |
| `Default_Eob_Pt_1024_Cdf` | `[4][3][9]` | 108 | 108 |

All 180 arrays match the PDF-extracted values element-for-element; each row
has a non-decreasing cumulative part, a base-rate entry, and adaptation count
0. No CDF is truncated.

### CDF coverage by spec page

| Table group | §/page | Present? | Dimensions | Used by | Status/Gap |
|---|---|---:|---|---|---|
| Motion / BAWP / TIP / warp / JMVD / CWP / FSC | §9.3 pp.623–626 | Yes (39) | per-array, rows 3–9 | `m2/advanced/advanced_cdf_tables.h`, `m2/multiref/gate`, `m1/ibc_stream`, `tcq`, `lossless`, `tx_secondary` | 33 used, 6 unbound (below) |
| Intra mode / CFL / MRL / CCSO / IBC | §9.3 pp.627–628 | Yes (11) | per-array, rows 3–9 | `m0_cdf_tables.h`, `m1_cdf_tables.h`, `m1/mrl_stream`, `m1/ibc_stream`, `m1_syntax.cpp` | all used |
| Partition / TX partition / segment / skip / DRL | §9.3 pp.630–637 | Yes (11) | per-array, rows 3–9 | `m0_cdf_tables.h`, `partition_ext`, `m1/deltaq`, `inter/drl_reorder` | 7 used, 4 unbound |
| Inter compound / refs / precision / interp filter | §9.3 pp.638–641 | Yes (17) | per-array, rows 3–9 | `m2_cdf_tables.h`, `m2/multiref/gate`, `inter/drl_reorder`, `m1/ibc_stream` | 13 used, 4 unbound |
| Palette / delta-Q / TX type / wedge / warp / CFL / restoration | §9.3 pp.642–648 | Yes (49) | per-array, rows 3–9 | `palette/scc_cdf.h`, `m1/deltaq`, `tx_type`, `tx_secondary`, `m2/advanced`, `m2/residual`, `cfl/mhccp`, `wiener_cdfs.h` | all used |
| Coefficient base / EOB / IDTX / PH / LF; CFL MH; DIP | §9.3 pp.655–670 | Yes (20) | per-array, rows 3–9 | `m0_cdf_tables.h`, `m1/qctx/qctx_cdf_tables.h`, `cfl/mhccp`, `dip` | all used |
| Coefficient LF luma/chroma | §9.3 p.708 | Yes (2) | per-array, rows 7 | `m0_cdf_tables.h`, `m1/qctx` | all used |
| Coeff UV / DPCM / morph / SDP / TIP / long TX / CDEF / AMVD / MRL / lossless / GDF / BRU | §9.3 pp.732–742 | Yes (31) | per-array, rows 3–9 | `cuda/m1/cdef`, `lossless`, `m0_cdf_tables.h`, `m1_cdf_tables.h`, `m1/mrl_stream`, `m2_cdf_tables.h` | 24 used, 7 unbound |

“Rows 3–9” means the last dimension (`N+1`) is between 3 and 9; the exact
per-array shapes are in `default_cdf.json`.

### CDFs present but not referenced by any encoder path (21)

These 21 arrays exist in `default_cdf.json` with correct dimensions and
values, but no generated C++ binding and no non-test consumer was found. All
are listed with their exact spec page and declared shape.

| Table | §/page | Present? | Dimensions | Used by | Status/Gap |
|---|---|---:|---|---|---|
| `Default_Mv_Joint_Adaptive_Cdf` | §9.3 p.623 | Yes | `[5]` | none | **Unused** |
| `Default_Amvd_Indices_Cdf` | §9.3 p.623 | Yes | `[2][9]` | none | **Unused** |
| `Default_Tip_Mode_Cdf` | §9.3 p.625 | Yes | `[3][3]` | none (TIP mode is a raw bit) | **Unused** |
| `Default_Use_Refinemv_Cdf` | §9.3 p.625 | Yes | `[24][3]` | none (`enable_refinemv` is a raw bit) | **Unused** |
| `Default_Jmvd_Scale_Mode_Cdf` | §9.3 p.626 | Yes | `[6]` | none | **Unused** |
| `Default_Jmvd_Adaptive_Scale_Mode_Cdf` | §9.3 p.626 | Yes | `[4]` | none | **Unused** |
| `Default_Tx_Do_Partition_Cdf` | §9.3 p.634 | Yes | `[2][2][9][3]` | none | **Unused** |
| `Default_Tx_Partition_Type_Reduced_Cdf` | §9.3 p.634 | Yes | `[2][2][14][8]` | none | **Unused** |
| `Default_Tx_2or3_Partition_Type_Cdf` | §9.3 p.636 | Yes | `[2][2][2][3]` | none | **Unused** |
| `Default_Tx_Partition_Type_Cdf` | §9.3 p.636 | Yes | `[2][2][14][8]` | none | **Unused** |
| `Default_Use_Optflow_Cdf` | §9.3 p.641 | Yes | `[2][3]` | none found in an entropy path | **Unused** |
| `Default_Interp_Filter_Cdf` | §9.3 p.641 | Yes | `[16][4]` | none | **Unused** |
| `Default_Use_Most_Probable_Precision_Cdf` | §9.3 p.641 | Yes | `[3][3]` | none; `me2/README.md` explicitly says the default rows are absent from the tree | **Unused** |
| `Default_Pb_Mv_Precision_Cdf` | §9.3 p.641 | Yes | `[2][3][4]` | none; same `me2` gap | **Unused** |
| `Default_Morph_Pred_Cdf` | §9.3 p.739 | Yes | `[3][3]` | none | **Unused** |
| `Default_Region_Type_Cdf` | §9.3 p.739 | Yes | `[4][3]` | SDP plans `region_type` but no entropy binding | **Unused** |
| `Default_Tip_Pred_Mode_Cdf` | §9.3 p.740 | Yes | `[3]` | none | **Unused** |
| `Default_Use_Amvd_Cdf` | §9.3 p.741 | Yes | `[9][3][3]` | none | **Unused** |
| `Default_Lossless_Inter_Tx_Type_Cdf` | §9.3 p.742 | Yes | `[3]` | none | **Unused** |
| `Default_Use_Gdf_Cdf` | §9.3 p.742 | Yes | `[3]` | none | **Unused** |
| `Default_Bru_Mode_Cdf` | §9.3 p.742 | Yes | `[4]` | none found | **Unused** |

### `m0_cdf.json` / `m0_cdf.h` subset is stale, truncated, and unused

`spec/tables/m0_cdf.json` and `m0_cdf.h` contain **19** arrays and are described
in `spec/tables/README.md` as “the 19 arrays the M0 syntax path reads”. The
actual M0 generator, `src/encoder/generate_m0_cdf.py`, binds **27** arrays from
`default_cdf.json`:

- nine arrays used by M0 but missing from `m0_cdf.json`/`m0_cdf.h`:
  `Default_Rect_Type_Cdf`, `Default_Eob_Pt_64_Cdf`, `Default_Eob_Pt_128_Cdf`,
  `Default_Eob_Pt_512_Cdf`, `Default_Coeff_Base_Lf_Cdf`,
  `Default_Coeff_Base_Lf_Eob_Cdf`, `Default_Coeff_Base_Lf_Uv_Cdf`,
  `Default_Coeff_Base_Lf_Eob_Uv_Cdf`, `Default_Coeff_Br_Lf_Cdf`;
- one array in the subset not used by that M0 binding:
  `Default_Y_Mode_Offset_Cdf` (it is used by M1).

Neither `m0_cdf.json` nor `m0_cdf.h` is included by any build target or C++
source. They are stale duplicate artifacts and should not be used to judge M0
coverage.

## Value discrepancies vs the spec

**None found** in the generated tables:

- all 28 numeric arrays in `block_tables.json` match the PDF values and shapes;
- all 180 arrays in `default_cdf.json` match the PDF values and declared
  dimensions;
- `Prob_Inc` and `Para_Adjustment_List` in
  `src/entropy/av2/av2_entropy_tables.h` match both the PDF and
  `block_tables.json`;
- the four `scan.json` arrays match a literal `get_scan()` implementation and
  are permutations;
- C++ copies inspected in `m0_cdf_tables.h`, `m1_tables.h`,
  `tx_type_tables.h`, `format_tables.h`, `ext_tables_generated.h`,
  `tx_secondary_tables.h`, `qctx_cdf_tables.h`, `deblock_core.h`,
  `film_grain_tables.h`, and `wedge_tables.h` have no value differences.
  (`rdoq_cdf_tables.h` intentionally stores only the qctx-0 slice, hence its
  smaller element counts; that is not a truncation.)

The only parser warnings were prose lines interleaved into numeric arrays
(`Side_Thresholds`, `Tx_Size_Sqr_Up`, `Md_Idx_To_Type`, `Default_Use_Amvd_Cdf`,
`Tile_*_Scaling_Factor`); none changed the parsed element count.

## Additional findings not included in the 220

- `Transform_Shift` is present in `block_tables.json` but unused (behavior is
  hardcoded in `chroma_residual.cpp`); it is a §7.15.4.1 table, not §9.2.
- `m0_cdf.json`/`m0_cdf.h` are stale 19-array subsets; the real M0 binding uses
  27 arrays. Both files are unused by the build.
- `block_tables.json` labels `Subsampled_Size` (§5.20.7.26),
  `Partition_Size_Adjust` (§8.3.2), `Partition_Size_Adjust_Rect_Type`
  (§8.3.2), and `Transform_Shift` (§7.15.4.1) as clause `9.2`.
- `src/encoder/m1/generate_m1_tables.py` claims all inputs come from
  `spec/tables/*.json` but hardcodes `Partition_Size_Adjust` and
  `Partition_Size_Adjust_Rect_Type` rather than reading
  `block_tables.json`. Values currently match; regeneration provenance does
  not.

## Reproduction (read-only)

```sh
# 1. Extract §9.2 / §9.3 text and compare every JSON array to the PDF.
python3 - <<'PY'
import json, sys
sys.path.insert(0, "tools/spec_extract")
from specpdf import find_array, find_arrays, section, pdf_text
text = pdf_text()
sec = section(text, "9.3. Default CDF tables", "9.4. Quantizer matrix tables")
gen = json.load(open("spec/tables/default_cdf.json"))["arrays"]
bad = []
for arr in find_arrays(sec, "Default_"):
    if arr.values != gen[arr.name]["data"]:
        bad.append(arr.name)
print("CDF mismatches:", bad)
PY

# 2. Find the encoder binding for a table.
grep -rn --include='*.h' --include='*.cpp' --include='*.cu' -E \
  'kMiWidthLog2|kPartitionSubsize|kCoeffBaseCdf|kScan32' src | grep -v _test

# 3. Regenerate/check the generated table set (not run for this audit).
# python3 tools/spec_extract/build_all.py
# python3 src/entropy/av2/generate_tables.py
```

The comparison scripts only read `spec/tables/*.json` and the pinned PDF; the
generators were deliberately not run so no checked-in table could be
rewritten.

## Counts

| Measure | Count |
|---|---:|
| Total §9.1–§9.3 tables audited (`36 §9.2 + 4 scans + 180 CDFs`) | **220** |
| Present in generated artifacts | **208** |
| — §9.2 arrays present | 24 / 36 |
| — scan arrays present | 4 / 4 |
| — §9.3 CDF arrays present | 180 / 180 |
| Absent from generated artifacts | **12** (all §9.2) |
| Present but unused / only partially used by any encoder path | **26** (5 §9.2 + 21 CDFs) |
| Value mismatches against spec | **0** |
| Truncated generated artifacts | **1 set** (`m0_cdf.json`/`m0_cdf.h`: 19 of the 27 arrays used by M0) |

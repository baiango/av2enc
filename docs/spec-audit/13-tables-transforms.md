# Spec audit 13 — §9.4–§9.8 tables: QM, warp, transforms, IST, loop restoration

Scope: the numeric tables printed in AV2 v1.0.0 §9.4–§9.8, i.e. spec pages
742–1100. The brief said "pages 742–921"; 921 is only where **§9.8 Loop
restoration tables** starts. §9.8 runs through page 1100 (`Dip_Weights` ends
there, Annex A starts on 1101), so this audit covers the whole of §9.8 rather
than stopping at its heading.

Read from `spec/AV2_Spec_v1.0.0.pdf` (v1.0.0, 2026-05-28, commit `38f28e7`,
SHA-256 `e9916f09…b533f`). Page index = spec page − 1 with `pypdf`, as in the
rest of `tools/spec_extract/`. Code read from `src/encoder/**` and
`src/cuda/**`; no file outside `docs/spec-audit/13-tables-transforms.md` was
modified.

## What counts as a table here

Every array declaration that appears in §9.4–§9.8:

| Section | Tables |
|---|---|
| §9.4 Quantizer matrix tables (p742–840) | `Qm_Offset`, `Quantizer_Matrix` |
| §9.5 Warp filter tables (p841–844) | `Warped_Filters`, `Ext_Warped_Filters` |
| §9.6 1d transform tables (p845–849) | `Dct_Kernel4/8/16/32`, `Adst_Kernel4/8/16`, `Fdst_Kernel4/8/16`, `Ddtx_Kernel8/16` |
| §9.7 Secondary transform tables (p849–920) | `Ist_4x4_Kernel`, `Ist_8x8_Kernel`, `Stx_Scan_Map` |
| §9.8 Loop restoration tables (p921–1100) | `Gdf_Alpha`, `Gdf_Weight`, `Gdf_Bias`, `Gdf_Intra_Error`, `Gdf_Inter_Error`, `Pc_Wiener_Sub_Classify2`, `Pc_Wiener_Lut_To_Class`, `Pc_Wiener_Sub_Classify`, `Pc_Wiener_Filters`, `Dip_Weights` |

That is **29 arrays**. Film-grain `Gaussian_Sequence` is **not** in this range
(it is §9.2, p618) and is reported separately at the end.

## Legend

| Status | Meaning |
|---|---|
| **WIRED** | Read by a path that the `av2enc` encoder binary / CUDA pipeline executes, and which changes emitted syntax or reconstruction. |
| **COMPONENT** | Spec-complete implementation with tests, but no encode path calls it; the owning tool flag/syntax bit is hard-coded off. |
| **COMPONENT-GATE** | As COMPONENT, but the values are also exercised end-to-end by a dedicated decoder gate (the gate is not the production path). |
| **ABSENT** | No copy of the table anywhere in `src/`. |

"Unused" in the counts below means **not WIRED** (COMPONENT or COMPONENT-GATE).

## Inventory

| Table | §/page | Present? | Dimensions | Used by | Status/Gap |
|---|---|---|---|---|---|
| `Qm_Offset` | 9.4.3 / p743 | Yes — `src/encoder/qm/qm_constants.h:20` as `kQmOffset`; also `qm_tables.json` | `[TX_SIZES_ALL]` = `[25]` | `qm::qm_offset()` (`qm.cpp:182`), `default_matrix_value()`, `dcdeltaq/quant.cpp:159`; `qm_test`, `qm_gate.sh` | **COMPONENT-GATE**. Generated from the PDF with a pinned SHA-256 and shape/offset checks; 25/25 values match. Production header writers still code `using_qmatrix = 0` (`m1_headers.cpp:177`, `m0_headers.cpp:310`, `m2_headers.cpp:284`). |
| `Quantizer_Matrix` | 9.4.3 / p743–840 | Yes — `src/encoder/qm/qm_tables.h:15` as `kQuantizerMatrix`; `qm_tables.json` | `[15][2][3600]` = 108,000 | `qm::default_matrix_value()` (`qm.cpp:185-193`), `build_q2_matrix()`, `dcdeltaq/quant.cpp:169`, CUDA `src/cuda/qm/qm_gpu.cu` | **COMPONENT-GATE**. 108,000/108,000 values match the PDF. M1/M2 dequant still uses the fixed `528` step (`txv::dequantize`), so an encoded stream cannot reach this table. |
| `Warped_Filters` | 9.5 / p841–844 | Yes — `src/encoder/inter/advanced/tables/warp_filters.h:23` as `kWarpedFilters` | `[WARPEDPIXEL_PREC_SHIFTS*7+1][8]` = `[449][8]` = 3592 | `warp_common.h:377,403` (`block_warp`), `advanced_test.cpp` | **COMPONENT**. 3592/3592 values match. No M2 warp predictor is wired (`grep local_warp` in `m2_syntax.cpp`/`m2/rdo` is empty), and `inter/advanced/` is not in the core build. |
| `Ext_Warped_Filters` | 9.5 / p844 | Yes — `warp_filters.h:475` as `kExtWarpedFilters` | `[EXT_WARP_PHASES+1][EXT_WARP_TAPS]` = `[65][6]` = 390 | `warp_common.h:589,604,628,638` (`extend_warp`) | **COMPONENT**. 390/390 values match. Same unreachable-tool blocker as above. |
| `Dct_Kernel4` | 9.6 / p845 | Yes — `src/encoder/tx_variants/tx_tables.h:10`, `tx_tables_dev.cuh`, `src/encoder/m0_tables.h:98`, `spec/tables/dct_kernels.json` | `[4][4]` | M0 CPU `m0_still.cpp`; M1 CPU `txv::*`; CUDA `m1_tx.cu:11`, `hbd_tx.cu:9`, `tx_variants.cu:9` | **WIRED**. 16/16 values match. |
| `Dct_Kernel8` | 9.6 / p845 | Yes — `tx_tables.h:17`, `m0_tables.h`, `dct_kernels.json` | `[8][8]` | as above | **WIRED**. 64/64 values match. |
| `Dct_Kernel16` | 9.6 / p846 | Yes — `tx_tables.h:28`, `m0_tables.h`, `dct_kernels.json` | `[16][16]` | as above | **WIRED**. 256/256 values match. |
| `Dct_Kernel32` | 9.6 / p846 | Yes — `tx_tables.h:47`, `m0_tables.h`, `dct_kernels.json`, `spec/tables/dct32.h` | `[32][32]` | as above; M0 32x32, M1 all sizes | **WIRED**. 1024/1024 values match. `spec/tables/dct32.h` is an unused duplicate (see residue notes). |
| `Adst_Kernel4` | 9.6 / p848 | Yes — `tx_tables.h:82`, `tx_tables_dev.cuh` | `[4][4]` | `txv::tx_kernel(ADST,4)`; M1 `--m1-tx-type`; CUDA `m1_tx.cu`/`hbd_tx.cu` | **WIRED** but only when the TX_TYPE search selects a non-DCT type (`--m1-tx-type`; default is DCT_DCT). 16/16 values match. |
| `Adst_Kernel8` | 9.6 / p848 | Yes — `tx_tables.h:89`, dev copy | `[8][8]` | as above | **WIRED** (same flag). 64/64 values match. |
| `Adst_Kernel16` | 9.6 / p848 | Yes — `tx_tables.h:100`, dev copy | `[16][16]` | as above | **WIRED** (same flag). 256/256 values match. |
| `Fdst_Kernel4` | 9.6 / p848 | Yes — `tx_tables.h:119`, dev copy | `[4][4]` | `txv::tx_kernel(FDST,4)`; FLIPADST TX_TYPEs | **WIRED** (same flag). 16/16 values match. |
| `Fdst_Kernel8` | 9.6 / p848 | Yes — `tx_tables.h:126`, dev copy | `[8][8]` | as above | **WIRED** (same flag). 64/64 values match. |
| `Fdst_Kernel16` | 9.6 / p849 | Yes — `tx_tables.h:137`, dev copy | `[16][16]` | as above | **WIRED** (same flag). 256/256 values match. |
| `Ddtx_Kernel8` | 9.6 / p849 | Yes — `src/encoder/m2/ddt/ddt_tables.h:19` as `kDdtxKernel8` | `[8][8]` | `ddt_transform.cpp:180` (`ddt_kernel`); `ddt_test`; `m2/ddt/gate` (patched build) | **COMPONENT-GATE**. 64/64 values match. Hand-transcribed, not generated; header points at the two attachment headers. Production M2 writes `enable_inter_ddt = 0` (`m2_headers.cpp:195`) and `residual_decision.cpp:19`. |
| `Ddtx_Kernel16` | 9.6 / p849 | Yes — `ddt_tables.h:30` | `[16][16]` | as above | **COMPONENT-GATE**. 256/256 values match. Same blocker. |
| `Ist_4x4_Kernel` | 9.7 / p849–857 | Yes — `src/encoder/tx_secondary/tx_secondary_tables.h:86` | `[IST_SET_SIZE_4X4][STX_TYPES-1][IST_4X4_HEIGHT][IST_4X4_WIDTH]` = `[14][3][8][16]` = 5376 | `tx_secondary.cpp:299-314,373-382`; CUDA `tx_secondary_gpu.cu:24` | **COMPONENT**. 5376/5376 values match. Generated from SHA-256-pinned `ist_4x4_kernel.h`; an independent PDF parse agrees. No `sec_tx_type` syntax path exists (`enable_intra_ist`/`enable_inter_ist` hard-coded 0 at `m1_headers.cpp:132-133`). |
| `Ist_8x8_Kernel` | 9.7 / p858–918 | Yes — `tx_secondary_tables.h:537` | `[IST_SET_SIZE_8X8][STX_TYPES-1][IST_8X8_HEIGHT][IST_8X8_WIDTH]` = `[11][3][32][48]` = 50,688 | as above; CUDA `tx_secondary_gpu.cu:25` | **COMPONENT**. 50,688/50,688 values match. Same blocker. |
| `Stx_Scan_Map` | 9.7 / p919–920 | Yes — `tx_secondary_tables.h:1684` | `[IST_SET_SIZE_8X8][STX_TYPES-1][IST_8X8_WIDTH]` = `[11][3][48]` = 1584 | `tx_secondary.cpp:313,381`; CUDA `tx_secondary_gpu.cu:26` | **COMPONENT**. 1584/1584 values match. Same blocker. |
| `Gdf_Alpha` | 9.8 / p921–932 | Yes — `src/encoder/filters/spec_tables/gdf_tables.h:22` | `[6][6][22][4]` = 3168 | `spec_tables/gdf.cpp:142` (`ApplyGdfFilter`) | **COMPONENT**. 3168/3168 values match; attachment SHA-256 pinned and PDF re-checked. `enable_gdf` is hard-coded 0 at `m1_headers.cpp:147`, `m0_headers.cpp:257`, `m2_headers.cpp:209`; M2 `make_sequence_filter_config()` sets it false. |
| `Gdf_Weight` | 9.8 / p933–970 | Yes — `gdf_tables.h:903` | `[6][6][3][22][4]` = 9504 | `gdf.cpp:164` | **COMPONENT**. 9504/9504 values match. Same blocker. |
| `Gdf_Bias` | 9.8 / p971 | Yes — `gdf_tables.h:3584` | `[6][6][3]` = 108 | `gdf.cpp:171` | **COMPONENT**. 108/108 values match. Same blocker. |
| `Gdf_Intra_Error` | 9.8 / p971–1014 | Yes — `gdf_tables.h:3637` | `[6][4096]` = 24,576 | `gdf.cpp:176` | **COMPONENT**. 24,576/24,576 values match. Same blocker. |
| `Gdf_Inter_Error` | 9.8 / p1015–1067 | Yes — `gdf_tables.h:5190` | `[5][6][1000]` = 30,000 | `gdf.cpp:177` | **COMPONENT**. 30,000/30,000 values match. Same blocker. |
| `Pc_Wiener_Sub_Classify2` | 9.8 / p1068–1081 | Yes — `src/encoder/filters/spec_tables/pc_wiener_tables.h:625` | `[4][8][NUM_PC_WIENER_LUT_CLASSES]` = `[4][8][256]` = 8192 | only `spec_tables_test.cpp:98` (extent/consistency check) | **COMPONENT**, and not even used by the reference PC-Wiener filter — the class map is still caller-supplied. 8192/8192 values match. M1/M2 reject PC-Wiener outright (below). |
| `Pc_Wiener_Lut_To_Class` | 9.8 / p1082–1085 | Yes — `pc_wiener_tables.h:287` | `[4096]` | `pc_wiener.cpp:115` (`PcWienerClass`) | **COMPONENT**. 4096/4096 values match. Same blocker. |
| `Pc_Wiener_Sub_Classify` | 9.8 / p1086 | Yes — `pc_wiener_tables.h:548` | `[4][NUM_PC_WIENER_LUT_CLASSES]` = `[4][256]` = 1024 | `pc_wiener.cpp:130` (`PcWienerFilterBlock`) | **COMPONENT**. 1024/1024 values match. Same blocker. |
| `Pc_Wiener_Filters` | 9.8 / p1087–1094 | Yes — `pc_wiener_tables.h:18` | `[4][NUM_PC_WIENER_FILTERS][PC_WIENER_COEFFS]` = `[4][64][13]` = 3328 | `pc_wiener.cpp:137` | **COMPONENT**. 3328/3328 values match. Same blocker. |
| `Dip_Weights` | 9.8 / p1095–1100 | Yes — `src/encoder/dip/dip_weights.h:25`, body in `dip_weights_data.inc` | `[6][64][11]` = 4224 | `dip.cpp:213,518`; CUDA `src/cuda/dip/dip_device.cuh:23` | **COMPONENT**. 4224/4224 values match; content SHA-256 + sum pinned. `enable_dip` hard-coded 0 (`m1_headers.cpp:114`, `m0_headers.cpp:218`, `m2_headers.cpp:141`, `pipeline/tiled_still_gate.cpp:164`) and no DIP syntax path exists. |

## Counts

| Metric | Count |
|---|---:|
| Tables in §9.4–§9.8 | **29** |
| Present in `src/` | **29** |
| Absent | **0** |
| Wired to an encoder path | **10** (`Dct_Kernel4/8/16/32`, `Adst_Kernel4/8/16`, `Fdst_Kernel4/8/16`) |
| Recovered but not wired ("unused") | **19** |
| Value mismatches vs the PDF | **0** |

The 19 unused break down as: QM 2, warp 2, DDTX 2, IST 3, §9.8 loop restoration 10.

One caveat on "wired": the ADST and FDST kernels are only reached when the M1
intra TX_TYPE search is enabled (`--m1-tx-type`, `m1_still.cpp:818-866`);
without it the encoder is DCT_DCT and those 6 kernel tables are loaded but not
indexed at runtime. They are still a first-class, gate-covered encode path.

## Verification: every value, not a sample

The usual "spot check" is not enough for a table audit, so every array was
extracted from the pinned PDF with `tools/spec_extract/specpdf.find_array`,
and compared element-for-element against the checked-in table. (The only PDF
clean-up needed was collapsing the PDF's `- 1` spacing inside the warp arrays.)

| Tables | Source file compared | Values compared | Result |
|---|---|---:|---|
| `Qm_Offset`, `Quantizer_Matrix` | `qm_constants.h`, `qm_tables.h` | 25 + 108,000 | all equal |
| `Warped_Filters`, `Ext_Warped_Filters` | `warp_filters.h` | 3,592 + 390 | all equal |
| `Dct_Kernel4/8/16/32` | `tx_variants/tx_tables.h` | 16 + 64 + 256 + 1,024 | all equal |
| `Adst_Kernel4/8/16` | `tx_variants/tx_tables.h` | 16 + 64 + 256 | all equal |
| `Fdst_Kernel4/8/16` | `tx_variants/tx_tables.h` | 16 + 64 + 256 | all equal |
| `Ddtx_Kernel8/16` | `m2/ddt/ddt_tables.h` | 64 + 256 | all equal |
| `Ist_4x4_Kernel` | `tx_secondary_tables.h` | 5,376 | all equal |
| `Ist_8x8_Kernel` | `tx_secondary_tables.h` | 50,688 | all equal |
| `Stx_Scan_Map` | `tx_secondary_tables.h` | 1,584 | all equal |
| `Gdf_Alpha` | `gdf_tables.h` | 3,168 | all equal |
| `Gdf_Weight` | `gdf_tables.h` | 9,504 | all equal |
| `Gdf_Bias` | `gdf_tables.h` | 108 | all equal |
| `Gdf_Intra_Error` | `gdf_tables.h` | 24,576 | all equal |
| `Gdf_Inter_Error` | `gdf_tables.h` | 30,000 | all equal |
| `Pc_Wiener_Sub_Classify2` | `pc_wiener_tables.h` | 8,192 | all equal |
| `Pc_Wiener_Lut_To_Class` | `pc_wiener_tables.h` | 4,096 | all equal |
| `Pc_Wiener_Sub_Classify` | `pc_wiener_tables.h` | 1,024 | all equal |
| `Pc_Wiener_Filters` | `pc_wiener_tables.h` | 3,328 | all equal |
| `Dip_Weights` | `dip_weights_data.inc` | 4,224 | all equal |

Representative values printed in the PDF and found in the code:

| Table | PDF value | Code |
|---|---|---|
| `Qm_Offset` | p743 `0, 16, 80, 336, 336, 1360, …` | identical `kQmOffset` |
| `Quantizer_Matrix[0][0]` first 4x4 | p743 `32, 43, 73, 97, 43, 67, 94, 110, …` | identical |
| `Quantizer_Matrix[14][1]` first 4x4 | p834 `31, 31, 31, 31, 31, 31, 31, 31, …` | identical |
| `Warped_Filters[0]` | p841 `{127, 1, 0, 0, 0, 0, 0, 0}` | identical |
| `Warped_Filters[192]` | p841 zero phase `{0, 0, 0, 127, 1, 0, 0, 0}` | identical |
| `Ext_Warped_Filters[64]` | p844 `{0, 0, 0, 128, 0, 0}` | identical |
| `Dct_Kernel32[1][0]` | p846 `90` | identical |
| `Adst_Kernel4[0][0]` | p848 `18` | identical |
| `Fdst_Kernel4[0][0]` | p848 `89` | identical |
| `Ddtx_Kernel8[0]` | p849 `4, 7, 15, 33, 65, 98, 106, 80` | identical |
| `Ist_4x4_Kernel[0][0][0]` | p849 `102, -45, -53, -5, -13, -3, -3, 19, …` | identical |
| `Ist_8x8_Kernel[0][0][0]` | p858 `103, -44, -48, -9, -12, -6, …` | identical |
| `Stx_Scan_Map[0][0]` | p919 `0, 1, 2, 3, 4, 5, …` | identical |
| `Gdf_Alpha[0][0][2]` | p921 `{511, 386, 511, 386}` | identical |
| `Gdf_Bias[0][0]` | p971 `{1925, -13583, 11215}` | identical |
| `Gdf_Intra_Error[0][0]` | p971 `0` | identical |
| `Gdf_Inter_Error[0][0][0..7]` | p1015 `6, 5, 4, 3, 2, 3, 4, 5` (inter-distance 0) | identical |
| `Pc_Wiener_Filters[0][0]` | p1087 `73, 127, -20, -30, -38, -29, …` | identical |
| `Pc_Wiener_Lut_To_Class[0]` | p1082 `83` | identical |
| `Pc_Wiener_Sub_Classify[0][0]` | p1086 `7` | identical |
| `Pc_Wiener_Sub_Classify2[0][0][0]` | p1068 `0` | identical |
| `Dip_Weights[0][0]` | p1095 `-248, 690, 53, -26, 19, 413, 133, 3, 3, -13, -1` | identical |

Repo-level provenance gates also pass on this tree:

```
python3 src/encoder/qm/generate_qm_tables.py --check          -> "qm tables: up to date"
python3 src/encoder/tx_secondary/generate_tx_secondary_tables.py --check \
        --attachments "$IST_ATT"                              -> exit 0
python3 src/encoder/filters/spec_tables/generate_spec_tables.py --check
                                                              -> all four headers current
python3 src/encoder/filters/spec_tables/verify_pdf_tables.py  -> all 10 tables match the pinned PDF
python3 src/encoder/inter/advanced/tables/extract_spec_tables.py --check
                                                              -> "generated headers are current"
```

The three §9.7 attachment headers were re-downloaded and matched the pinned
SHA-256 values:

```
ist_4x4_kernel.h  9fafc6bc…85404
ist_8x8_kernel.h  65d17c0e…7fcc4
stx_scan_map.h    8825c77d…c7146
```

## Recovered but not wired

This is the dominant result: **19 of 29 tables are complete and correct but
cannot be reached by an encoded stream.** Grouped by the blocker:

1. **Quantizer matrices (2 tables).** `src/encoder/qm/` has the matrices, the
   §7.14.4 `q2` helper, the `setup_qm_params()` writer and the QM OBU writer,
   and `qm_gate.sh` shows both `dav2d` (default matrices) and `avmdec`
   (user-defined matrices) accept the output. But:
   * `m1_headers.cpp:177`, `m0_headers.cpp:310`, `m2_headers.cpp:284` write
     `using_qmatrix = 0`;
   * the M1/M2/CUDA dequant call sites still use the fixed `528` step
     (`tx_variants.cpp`, `m1_tx.cu:60`, `hbd_tx.cu:81`, …) instead of
     `qm::build_q2_matrix()`;
   * the only consumer of `qm::matrix_value()` is `src/encoder/quant/dcdeltaq/`
     (`quant.cpp:152-169`), which is itself a component and is not in
     `CMakeLists.txt`.
2. **Warp filters (2).** `inter/advanced/` is not compiled into `av2enc_core`
   or the M2 binary, and M2 emits no local/extend-warp syntax. The tables are
   exercised by `advanced_test.cpp` only.
3. **DDTX kernels (2).** `m2_headers.cpp:195` writes `enable_inter_ddt = 0` and
   `rdo/residual_decision.cpp:19` forces the decision flag false. A dedicated
   gate (`src/encoder/m2/ddt/gate/`) proves the inverse against `dav2d` using
   build-directory patches, so the kernels are known-good; production is not
   wired.
4. **IST kernels and scan (3).** `m1_headers.cpp:132-133` hard-codes both
   sequence bits to 0, `m1_syntax.cpp` has no `sec_tx_type`/
   `most_probable_stx_set` symbols, and the M1 `BlockSyntax` has no `MrlIndex`.
   The component cannot emit an IST stream without core edits; no decoder gate
   exists.
5. **GDF (5).** `spec_tables/gdf.cpp` implements the §7.20.5 reference, but
   `enable_gdf` is hard-coded 0 in every header writer, and the M2 restoration
   adapter sets `config.enable_gdf = false`. Worse, `restoration/README.md` and
   the M1/M2 code comments still say the GDF tables are unavailable.
6. **PC-Wiener (4).** `spec_tables/pc_wiener.cpp` implements §7.20.4, but
   `m2_restoration.cpp:60` forces `lr_disable_luma_pc_wiener = true` and
   `validate_no_pc_wiener()` rejects `RESTORE_PC_WIENER`. The rejection message
   (`m2_restoration.cpp:393-395`, `m1/filters_stream.cpp:1728-1731`) says the
   `pc_wiener_*` tables are "absent from the spec text" — that is false; they
   are §9.8 and are fully recovered in `spec_tables/`. Only
   `Pc_Wiener_Sub_Classify2` is not used by the reference filter itself; it is
   read only by `spec_tables_test.cpp`.
7. **DIP (1).** `src/encoder/dip/` is complete with CPU + CUDA references and
   tests, but `enable_dip` is hard-coded 0 in the sequence writers and no
   `dip_mode_info()` syntax path exists.

## Value mismatches

**None.** Every one of the 29 arrays matches the pinned PDF element-for-element
(260,227 values), and the §9.7 values also match the three official attachment
headers byte-for-byte.

There are, however, several **documentation/label mismatches** that make the
tree look less complete than it is. These are not table defects, but they are
why some blockers above are stale:

| Location | Claim | Reality |
|---|---|---|
| `src/encoder/dip/dip_weights.h:3` and `dip/README.md` | `Dip_Weights` is "§9.3 default CDF tables" | It is **§9.8**, p1095–1100, the last table before Annex A. |
| `src/encoder/filters/spec_tables/README.md` | "§9.8 and §9.9" | There is no §9.9 in the pinned PDF; §9.8 ends at p1100. |
| `src/encoder/tx_secondary/README.md` | "The §9.7 kernels are **not** … in the PDF body; the spec references three attachment headers" | The PDF prints all three arrays in full (p849–920). The attachment headers are still the right primary source, and the values agree. |
| `src/encoder/filters/restoration/README.md:18` | Film grain "**TODO — Gaussian table missing**" | `Gaussian_Sequence` is recovered in `spec_tables/film_grain_tables.h` and verified against the PDF. |
| `docs/open-questions.md` A2 | "`Dct_Kernel4/8/16`, ADST, FDST, DDTX" are "parsed into `spec/tables/dct_kernels.json` and `dct32.h`" | `dct_kernels.json` contains **only** `Dct_Kernel4/8/16/32`; ADST/FDST live in `tx_variants/tx_tables.h`, DDTX in `m2/ddt/ddt_tables.h`. |
| `src/encoder/filters/restoration/classes/generate_classes_tables.py` docstring and `wiener_tables.h:119` | PC-Wiener tables "are not part of the specification text" | They are §9.8 and recovered; the classes module still makes the caller supply them, so two copies of the recovery exist (see below). |
| `m2_restoration.cpp:393`, `m1/filters_stream.cpp:1728` | PC-Wiener tables "absent from the spec text" | Same as above; the guard is now unnecessary for data-availability reasons (the tool is still unwired for other reasons). |

## Residue and drift risks (no value error today)

* **DCT kernel duplication.** The same 4/8/16/32 kernels exist in four places:
  `src/encoder/m0_tables.h` (generated from `spec/tables/dct_kernels.json`),
  `src/encoder/tx_variants/tx_tables.h` + `tx_tables_dev.cuh` (generated from
  the PDF), `src/cuda/transform.cu` (hand-copied device arrays with a comment
  saying they were copied from `m0_tables.h`), and `spec/tables/dct32.h` (a
  generated 32x32-only header with no `#include` anywhere). All agree today;
  `transform_test.cu` compares every entry of the CUDA copy against the CPU
  reference. Consolidating the two generators would remove a future drift risk.
* **`spec/tables/dct32.h` is dead.** `grep -rn "dct32.h"` finds only the
  generator, the spec-table README and two docs; nothing includes it.
  `spec/tables/dct_kernels.json` is live (`generate_m0_tables.py` reads it).
* **Provenance strength is uneven.** QM pins the PDF SHA-256 and validates the
  informative §9.4.2 derivation; IST and the §9.8 recovery pin attachment
  SHA-256 and validate against the PDF; DIP pins a content digest and sum.
  The warp generator and `tx_variants/generate_tx_tables.py` parse the PDF but
  do not pin its hash or a content digest (`specpdf.py` only warns on a SHA
  mismatch), and `generate_tx_tables.py` has no `--check` mode. The warp
  generator does have `--check`. The DDTX tables are hand-transcribed (with a
  header comment and attachment cross-check) rather than generated.
* **Two recoveries of the PC-Wiener/loop-restoration world.**
  `src/encoder/filters/spec_tables/` has the full §9.8 datasets, while
  `src/encoder/filters/restoration/classes/` still hard-codes the prose tables
  (`Pc_Wiener_Normalizer`, `Mode_Weights`, `Mode_Offsets`, `Shuffled_Index`)
  and expects the class map from the caller. No numeric conflict, but a single
  owner would make the M1/M2 rejection messages easy to retire.

## Out-of-range table checked anyway

`Gaussian_Sequence[2048]` (film-grain grain-template RNG) is **§9.2
Conversion tables, p618** — not §9.4–§9.8.

* Present: `src/encoder/filters/spec_tables/film_grain_tables.h:16`
  (`kGaussianSequence`), from the SHA-pinned `raw/gaussian_sequence.h`.
* Dimensions: `[2048]`. All 2048 values match the PDF
  (`verify_pdf_tables.py` prints `ok Gaussian_Sequence 2048 values (PDF p618)`).
* Used by: `film_grain.cpp` only (`src/encoder/filters/spec_tables/`); film
  grain is output-only and not wired into M0/M1/M2.
* The stale "Gaussian table missing" note in `filters/restoration/README.md`
  should be removed when that module is next touched.

Supporting tables that are **printed in prose, not §9.x**, and therefore not
counted above: `Gdf_Coords`, `Pc_Wiener_Config`, `Pc_Wiener_Normalizer`,
`Mode_Weights`, `Mode_Offsets`, `Decode_Num_Filter_Classes` (transcribed in
`gdf.h` / `pc_wiener.h` / `wiener_tables.h`), and the §7.20.3
`Wiener_Ns_Config_Y/Uv` + `Wiener_Filters_420` tables used by the wired Wiener
NS filter. They are outside this audit's section range but were spot-checked
when tracing the §9.8 consumers.

## Bottom line

* **Data coverage is excellent:** 29/29 §9.4–§9.8 tables present, correct
  dimensions, generated from pinned normative artifacts, and zero value
  mismatches against the PDF.
* **Integration coverage is poor:** only the §9.6 DCT/ADST/FDST kernels (10
  tables) reach the encoder. QM, warp, DDTX, IST, GDF, PC-Wiener and DIP are
  all recovered-but-unwired, and several blockers still quote the obsolete
  "tables are missing" reason even though the tables are in the tree.
* The highest-value next step for this area is not more table extraction; it
  is wiring one tool per gate (QM is the cheapest: matrix + §7.14.4 helper +
  OBU writer already exist and `qm_gate.sh` reports green in its README), then
  retiring the stale availability comments so the next audit does not
  re-litigate them.

## Reproduction

```sh
# page-accurate extraction (pypdf; page index = spec page - 1)
python3 - <<'PY'
from pypdf import PdfReader
r = PdfReader("spec/AV2_Spec_v1.0.0.pdf")
print(r.pages[920].extract_text()[:80])   # spec page 921 -> §9.8
PY

# provenance / freshness gates
python3 src/encoder/qm/generate_qm_tables.py --check
python3 src/encoder/filters/spec_tables/generate_spec_tables.py --check
python3 src/encoder/filters/spec_tables/verify_pdf_tables.py
python3 src/encoder/inter/advanced/tables/extract_spec_tables.py --check
python3 src/encoder/tx_secondary/generate_tx_secondary_tables.py --check \
        --attachments <dir-with-the-three-ist-headers>

# full value diff used for the table above (no repo writes)
# parse each PDF array with tools/spec_extract/specpdf.find_array and compare
# the flattened values to the named C++ initializer.
```

# Generated spec tables

**Do not edit by hand.** Everything here is produced by
`python3 tools/spec_extract/build_all.py` from
`spec/AV2_Spec_v1.0.0.pdf` (SHA-256 `e9916f09…b533f`) and the cached HTML
spec. See `tools/spec_extract/README.md`.

| File | Contents | Clause |
|---|---|---|
| `constants.csv` / `constants.json` | all 338 named constants with evaluated values | §3 |
| `block_tables.json` | block/tx size tables, partition tables, scan/context tables, `Prob_Inc`, `Para_Adjustment_List`, `Transform_Shift`, `Md_Idx_To_Type`, intra mode lists | §9.2, §5.20 |
| `intra_modes.json` | `YMode` / `UVMode` enums (numbers), `Reordered_Y_Mode`, `Default_Mode_List_Y/Uv` | §6.19.5.5/.6 |
| `quant.json` | `Ac_Qlookup[25]`, `qlookup(q)` for q = 0..255, `MAXQ_8_BITS` | §7.14.2 |
| `dct_kernels.json` | `Dct_Kernel4/8/16/32` full matrices | §9.6 |
| `dct32.h` | `av2spec::kDctKernel32[32][32]` as int16 | §9.6 |
| `scan.json` | diagonal (TX_CLASS_2D) scan for 4/8/16/32, generated from the §5.20.7.30 algorithm | §5.20.7.30 |
| `default_cdf.json` | all 180 default CDF arrays (flat data + declared dims) | §9.3 |
| `basic_cdf.json` | the 19 CDF arrays the M0 path reads | §9.3 |
| `basic_cdf.h` | the same 19 as typed `constexpr` C++ arrays | §9.3 |
| `spec_sections.json` | heading id → title (navigation only) | — |

## CDF encoding reminder

Each CDF array has `N+1` entries per distribution:

```
cdf[0 .. N-2]   cumulative probabilities (scaled to 32768)
cdf[N-1]        constant base adaptation rate (index into Para_Adjustment_List)
cdf[N]          adaptation counter (0 at init, saturates at 32)
```

See §8.2.6. `Default_*_Cdf` arrays with a leading `COEFF_CDF_Q_CTXS` dimension are
sliced once at `init_coeff_cdfs()` using
`base_q_idx<=90 ? 0 : <=140 ? 1 : <=190 ? 2 : 3`.

## Regenerating

```
python3 -m pip install pypdf beautifulsoup4 lxml
python3 tools/spec_extract/build_all.py
```

`_cache/` holds the downloaded HTML and is git-ignored.

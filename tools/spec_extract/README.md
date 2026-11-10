# tools/spec_extract

Generators for the machine-readable tables in `spec/tables/`. Everything is parsed
from the official AV2 v1.0.0 artefacts; nothing is hand-typed.

```
python3 tools/spec_extract/build_all.py
```

Requires `python3` with `pypdf` and `beautifulsoup4`/`lxml`.

## Sources

| Source | Used for | Why |
|---|---|---|
| `spec/AV2_Spec_v1.0.0.pdf` (pinned, SHA-256 checked) | §9.3 default CDFs, §9.6 transform kernels, §9.2 conversion tables, §7.14 quantiser | the **only** place the numbers exist |
| `https://av2.aomedia.org/v1.0.0/index.html` (cached in `spec/tables/_cache/`) | §3 constant table, section map | the PDF prints it, but with line-wrapping that makes the value column ambiguous |

The HTML version of §9.2/§9.3/§9.6 says "is defined in `<name>.h`" instead of
printing the tables. See `docs/open-questions.md` §A.

## Scripts

| Script | Output |
|---|---|
| `specpdf.py` | library: PDF text cleaning (page furniture, interleaved prose), C-array parser |
| `extract_constants.py` | `constants.csv`, `constants.json` — 338 constants, expressions evaluated in dependency order |
| `extract_tables.py` | `block_tables.json`, `quant.json`, `dct_kernels.json`, `dct32.h`, `scan.json`, `default_cdf.json`, `m0_cdf.json`, `m0_cdf.h`, `intra_modes.json` |
| `extract_sections.py` | `spec_sections.json` — heading id → number/title (navigation aid only) |
| `build_all.py` | runs all three in order |

## Validation built into the generators

`extract_tables.py` fails hard unless:

- every `Default_*_Cdf` in §9.3 was found (expects ≥150; actual 180);
- every CDF's element count equals the product of its declared dimensions
  (`[COEFF_CDF_Q_CTXS]` etc. resolved through `constants.json`);
- each CDF row has non-decreasing cumulative values and an adaptation count of 0;
- spot checks pass: `Ac_Qlookup[0] == 64`, `Dct_Kernel32[1][0] == 90`,
  `Max_Tx_Size_Rect[BLOCK_32X32] == TX_32X32`,
  `Num_4x4_Blocks_Wide[BLOCK_64X64] == 16`, the 32x32 scan is a permutation of
  0..1023 with head `0, 32, 1`.

`m0_cdf.h` and `dct32.h` are plain C++20 headers and compile cleanly with
`-Wall -Wextra -Wpedantic` (verified with clang).

## Output sizes

Deliberately small: the whole `spec/tables/` is ~300 KB. `default_cdf.json`
(121 KB) holds all 180 CDFs because they are painful to re-derive; `m0_cdf.json`
and `m0_cdf.h` carry only the 19 arrays the M0 syntax path reads.

## Adding a table

1. Add the name to `NUMERIC_TABLES` (§9.2) or `M0_CDFS` (not needed — all CDFs are
   already extracted) in `extract_tables.py`.
2. Re-run `build_all.py`; the dimension check will tell you if the parse slipped.
3. If the array contains enum identifiers, extend `ENUMS` or `resolve_enum()`.

Do not edit files in `spec/tables/` by hand; they are overwritten.

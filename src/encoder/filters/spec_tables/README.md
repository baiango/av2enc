# Recovered normative tables: GDF, PC-Wiener, film grain

AV2 v1.0.0 (2026-05-28, spec commit `38f28e7`) references several tables that
are **not printed in the PDF or HTML text**.  §9.8 and §9.9 say only
"`Gdf_Alpha` is defined in the `gdf_alpha.h` header file", and so on.

This directory recovers those attachments byte-for-byte, pins their SHA-256,
generates the C++ headers from them, and provides scalar CPU references for
the three tools that need them.  Nothing here is wired into the M1/M2 encode
path, `src/main.cpp`, the root `CMakeLists.txt`, or the existing
`src/encoder/filters/restoration/` code.

## Finding

The official attachments are published with the specification:

- Web: `https://av2.aomedia.org/v1.0.0/attachments/<name>`
- Mirror: `AOMediaCodec/av2-spec`, path `v1.0.0/attachments/`,
  commit `42d091fe9f8ea4b526800d073b3e4c4da132967e`

Every file below was downloaded from both locations and compared byte-for-byte
(`cmp`); all ten matched.  The verbatim files are fetched on demand into
`raw/` and verified against the SHA-256 values pinned in the generator; `raw/`
is not redistributed with this repository (only the derived headers are).

**Important correction to the earlier blocker note.**  The pinned PDF
(`spec/AV2_Spec_v1.0.0.pdf`, SHA-256 `e9916f09…b533f`) *does* print the full
data in §9.8 and §9.9, even though the HTML rendering only says "defined in
the `<name>.h` header file".  All ten PDF blocks were parsed and compared
number-for-number against the attachments: they are identical.
`./verify_pdf_tables.py` reproduces that check (PDF pages are listed below).

| Spec name | File | SHA-256 | Extents (`[refDstIdx/qpIdx/...]`) | PDF page |
|---|---|---|---|---|
| `Gdf_Alpha` | `gdf_alpha.h` | `8286d7fe…4ecf3600` | `[6][6][22][4]` | 921 |
| `Gdf_Weight` | `gdf_weight.h` | `58b154df…7862f666` | `[6][6][3][22][4]` | 933 |
| `Gdf_Bias` | `gdf_bias.h` | `9c9790f9…18b2372` | `[6][6][3]` | 971 |
| `Gdf_Intra_Error` | `gdf_intra_error.h` | `5c921991…36f13e5` | `[6][4096]` | 971 |
| `Gdf_Inter_Error` | `gdf_inter_error.h` | `4ca252f2…c5376f` | `[5][6][1000]` | 1015 |
| `Pc_Wiener_Filters` | `pc_wiener_filters.h` | `d368e9d2…285393a63` | `[4][64][13]` | 1087 |
| `Pc_Wiener_Lut_To_Class` | `pc_wiener_lut_to_class.h` | `2b5a7364…8d664a46` | `[4096]` | 1082 |
| `Pc_Wiener_Sub_Classify` | `pc_wiener_sub_classify.h` | `2641737152…638479e` | `[4][256]` | 1086 |
| `Pc_Wiener_Sub_Classify2` | `pc_wiener_sub_classify2.h` | `ae1f16ce…0aae3362` | `[4][8][256]` | 1068 |
| `Gaussian_Sequence` | `gaussian_sequence.h` | `1380bf0a…c38e4ab0` | `[2048]` | 618 |

The full hashes are pinned in `generate_spec_tables.py` and in the generated
`spec_tables_provenance.h` (`kTableProvenance`).  The PDF double-check makes
the recovery independent of both the web mirror and the HTML build: three
sources agree.

**Correction to the earlier blocker note** in
`src/encoder/filters/restoration/README.md`: `Gdf_Alpha` is
`[6][6][22][4]`, not `[6][3][22][4]`.  The first index is `refDstIdx` (0 =
intra, 1..5 = inter distances), the second is `qpIdx`
(`Clip3(0,2,…) + gdf_pic_qc_idx`, therefore 0..5).

No hand-typed values are used.  `Gdf_Coords`, `Pc_Wiener_Config`,
`Pc_Wiener_Normalizer`, `Mode_Weights`, `Mode_Offsets`,
`Decode/Encode_Num_Filter_Classes` and the `quant.json` lookup are printed in
the specification text and are transcribed with clause citations in `gdf.h`
and `pc_wiener.h`; everything that was only in an attachment comes from the
files above.

## Generated headers

`generate_spec_tables.py` parses the raw initializer bodies, validates the
declared extents against the spec constants, and emits:

| Header | Contents |
|---|---|
| `gdf_tables.h` | `kGdfAlpha`, `kGdfWeight`, `kGdfBias`, `kGdfIntraError`, `kGdfInterError`, `kGdfCoords` |
| `pc_wiener_tables.h` | `kPcWienerFilters`, `kPcWienerLutToClass`, `kPcWienerSubClassify`, `kPcWienerSubClassify2`, `kAcQlookup[25]`, `kQlookup[352]` |
| `film_grain_tables.h` | `kGaussianSequence[2048]` |
| `spec_tables_provenance.h` | spec version, mirror commit, per-file SHA-256 |

```sh
python3 generate_spec_tables.py            # rewrite the headers
python3 generate_spec_tables.py --check    # verify raw hashes + no stale header
python3 verify_pdf_tables.py               # optional: cross-check the pinned PDF
```

`--check` is the provenance gate: it fails if a vendored attachment changes by
one byte or if a generated header is out of date.  `verify_pdf_tables.py` is a
second, independent gate: it extracts §9.8/§9.9 from the pinned PDF and
compares every number to `raw/`.  It skips cleanly when the PDF or `pypdf` is
unavailable; `build_and_test.sh` runs it automatically when the PDF exists.

## CPU references

| File | Clause | Notes |
|---|---|---|
| `luma_source.h/.cpp` | §7.20.2 | `get_source_sample(0,x,y)`: CDEF/CCSO inside the stripe, CurrFrame outside, two-line clamp |
| `gdf.h/.cpp` | §5.18.7.9, §6.17.7.5, §7.20.1, §7.20.5 | `GdfBlockTableIndices` + `ApplyGdfFilter` |
| `pc_wiener.h/.cpp` | §7.20.4 | class derivation, `skipFilter` classify-only mode, 25-tap filter |
| `film_grain.h/.cpp` | §5.18.10, §6.17.10, §7.21.7 | RNG, AR grain templates, scaling LUT, stripe blending |

Scope:

- GDF and PC-Wiener are luma-only, as the specification defines them.
- Film grain supports 4:0:0, 4:2:0, 4:2:2 and 4:4:4 with BitDepth 8, 10, 12.
- All arithmetic is integer and uses `Round2`/`Round2Signed`/`Clip3` from
  §4.8 (`spec_math.h`).
- Film grain is output-only and must not be used as a reconstruction oracle.

### Known spec-text discrepancy

§7.20.5 says `get_gdf_sample` "gets a sample from the current stripe with
reflection at the stripe end", but the normative pseudocode it prints is
`return get_luma_source_sample(x, y)`, which routes to §7.20.2's clamp +
`CurrFrame`/`CdefFrame` selection.  This implementation follows the
pseudocode (the same behaviour as the other tools), and keeps the unused
`stripe_end_y` argument for signature fidelity.  If a later errata changes
`get_gdf_sample` to reflect, only `luma_source.cpp` and the GDF oracle need to
change.

## Tests

```sh
./build_and_test.sh                 # verify tables, build, run all four suites
./build_and_test.sh --no-run
SANITIZE=1 ./build_and_test.sh      # ASan + UBSan
```

The test oracle in each `*_test.cpp` is a second, independent transcription of
the spec pseudocode.  It shares only the normative generated tables with the
implementation.

| Suite | Checks | Coverage |
|---|---|---|
| `spec_tables_test` | 84,885 | extents, index ranges, Gaussian multiple-of-4/range, qlookup recurrence, `Sub_Classify2` vs `Decode_Num_Filter_Classes` |
| `gdf_test` | 321 | all 6 `refDstIdx` × 6 `qpIdx` × 4 scales × BitDepth 8/10/12, stripe-boundary sourcing, block-split equivalence, output range, index derivation |
| `pc_wiener_test` | 1,595 | all four filter sets, BitDepth 8/10, classify-only vs filter, `BlockEndX`/`LrTxSkip` clamping, 4×4 vs block boundaries |
| `film_grain_test` | 611 | 8/10-bit 4:2:0/4:4:4/4:0:0, AR lag 0–3, grain_scale_shift 0–3, seeds 0/1/0xffff, 16/32 blocks, overlap on/off, restricted range, chroma-from-luma, validation |

Total randomized pixel comparisons per run: well over 1.5 million.

## Blocked items

None for GDF, PC-Wiener or film grain: every table those tools need is printed
in the pinned PDF **and** published as an official attachment, and the two
sources agree exactly.  The earlier claim that "§9.8 does not print these
tables" came from the HTML rendering, which only links the headers; the PDF
inlines them.  The exact external files are `gdf_alpha.h`, `gdf_weight.h`,
`gdf_bias.h`, `gdf_intra_error.h`, `gdf_inter_error.h`,
`pc_wiener_filters.h`, `pc_wiener_lut_to_class.h`,
`pc_wiener_sub_classify.h`, `pc_wiener_sub_classify2.h`,
`gaussian_sequence.h`, all vendored in `raw/`.

The one remaining gap is integration, not data: none of these filters is
wired into a bitstream yet, so no `dav2d` claim is made.  Per
`docs/negative-results.md`, do not claim quality gains before a minimal stream
passes `tools/validate_dav2d.sh`.

## Licensing / redistribution

These are the normative AV2 specification attachments, mirrored from
`AOMediaCodec/av2-spec` (see that repository's `LICENSE`).  They are vendored
verbatim with hashes exactly so the generated headers can be
reproducibly checked; do not edit `raw/`.

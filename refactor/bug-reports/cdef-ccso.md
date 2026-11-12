# CDEF/CCSO bug scout

| id | severity | class | title | repro | evidence | files | status |
|---|---|---|---|---|---|---|---|
| SCOUT-CDEF-01 | high | confirmed | IBC skip blocks are mis-handled by CDEF: `--still-ibc --still-cdef` emits a bitstream dav2d rejects, or decodes with a silent recon mismatch | Details 01 | depth 2 default strengths: encoder exit 0, `dav2d --strict 1` rc 1 (`Error decoding frame: Invalid argument`, no output); strengths 1: decode rc 0 but Y/U/V 45/181/292 samples differ (518 bytes, max 2); adding `--still-cdef-on-skip-txfm` or dropping `--still-cdef` is sample-exact; depths 3..8 decode with 2..12 differing bytes | `src/pipeline/still/still_encoder.cpp:4050-4055`, `src/pipeline/still/still_syntax.cpp:2554-2570` | new |
| SCOUT-CDEF-02 | high | confirmed | SB128/256 + Wiener/chroma-LR/PC-Wiener: silent recon desync; with CDEF added the encoder aborts on `use_bank_k` | `build/av2enc tests/data/basic_640x360_1f.y4m -o /tmp/w.ivf --still --still-sb128 --still-wiener --recon /tmp/w.y4m` | decoder rc 0, Y 25,319 bytes differ (max 107), U/V 0; chroma-LR U/V 6,047/5,442; PC-Wiener Y 5,836; `--still-sb128 --still-cdef --still-wiener` exits 1 `use_bank_k outside the current bank`; sb256 identical | `src/pipeline/still/still_encoder.cpp:4642,4652-4653,4690-4692`, `src/pipeline/still/still_headers.cpp:75-82` | duplicate (SCOUT-PARTITION-05, `bug-reports/partition.md:32`) |
| SCOUT-CDEF-03 | high | confirmed | SB128/256 + GDF: `gdf_unit_matches_sb_size` is written although §5.4.10 omits it for a non-64 sequence SB; dav2d rejects the sequence header | `build/av2enc tests/data/basic_640x360_1f.y4m -o /tmp/g.ivf --still --still-sb128 --still-gdf && tools/bin/dav2d-macos -q -i /tmp/g.ivf -o /tmp/g.y4m --threads 1` | `Error parsing sequence header` / `Error parsing OBU data` / `Error decoding frame: Invalid argument`, no output; sb256 same; GDF alone exact | `src/pipeline/still/still_headers.cpp:186-188`, `src/pipeline/still/still_encoder.cpp:4909-4912,4955-4957` | duplicate (SCOUT-PARTITION-04, `bug-reports/partition.md:31`) |
| SCOUT-CDEF-04 | low | doc | Docs still claim SB128/256 reject CDEF/CCSO and pin `CcsoLumaSizeLog2` 6, but the code signals 7 and the combos gate/decode | `build/av2enc tests/data/basic_640x360_1f.y4m -o /tmp/a.ivf --still --still-sb128 --still-cdef --still-ccso-edge --recon /tmp/a.y4m && tools/bin/dav2d-macos -q -i /tmp/a.ivf -o /tmp/a.y4m.dec --threads 1` | decode rc 0 and recon sample-exact; same for sb256/bo-only/strengths 2/3/5/7/8 | `docs/m1-features.md:328-335`, `docs/negative-results.md:121` vs `src/pipeline/still/still_encoder.cpp:4233-4235`, `src/codec/partition/sb128/product_gate.sh` | new |

## Details

### SCOUT-CDEF-01 — CDEF ignores the IBC `skip_flag`

Repro:
```sh
B=build/av2enc; D=tools/bin/dav2d-macos
# A: default strengths (CdefStrengths 4) -> decoder rejects the tile
$B testdata/src_640x360.y4m -o /tmp/ibc_cdef.ivf --still --still-depth 2 \
   --still-ibc --still-cdef
$D -q -i /tmp/ibc_cdef.ivf -o /tmp/ibc_cdef.dec.y4m --threads 1 --strict 1
# B: silent variant, one strength set -> no cdef_idx symbols at all
$B testdata/src_640x360.y4m -o /tmp/ibc_cdef1.ivf --still --still-depth 2 \
   --still-ibc --still-cdef --still-cdef-strengths 1 --recon /tmp/ibc.rec.y4m
$D -q -i /tmp/ibc_cdef1.ivf -o /tmp/ibc.dec.y4m --threads 1
# compare /tmp/ibc.rec.y4m /tmp/ibc.dec.y4m (Y4M plane lengths 640*360, 320*180)
```
Observed:
* A: encoder exits 0 (`ibc: 23 exact-copy block(s)`), dav2d exits 1 with
  `Error decoding frame: Invalid argument` / `No data decoded`, no Y4M.
  `--still-cdef-strengths 2` and `8` fail the same way.
* B: dav2d exits 0 but the encoder reconstruction and the decoder output differ:
  Y 45, U 181, V 292 samples (518 bytes; max |delta| 2).
* Silent variant on `tests/data/basic_640x360_1f.y4m` (default strengths):
  depth 2 decodes but differs by 9 bytes; depths 3/4/5/6/7/8 differ by
  2/9/12/7/7/7 bytes. No decoder error, so a recon check is the only signal.
* Controls: same command plus `--still-cdef-on-skip-txfm` is sample-exact
  (0/0/0); `--still-ibc` without `--still-cdef` is sample-exact.

Expected:
Either a sample-exact stream or a CLI rejection of `--still-cdef` with
`--still-ibc` (the CLI currently accepts the pair and encodes).

Likely cause:
IBC is the only still-path leaf that codes `skip_flag = 1`, but both CDEF
inputs assume intra `skip = 0`:
1. `still_encoder.cpp:4050-4055` fills `CdefMiInputs` with only `mi_cols`/
   `mi_rows`, so `still_cdef_build_mi_grids()` materialises an all-zero
   `Skips[]` grid (it documents "intra blocks have skip_flag == 0", which is
   false once IBC is enabled). The encoder therefore runs CDEF on blocks the
   decoder will copy.
2. `still_syntax.cpp:2554-2570` calls `cdef_encode_block_index()` for every
   non-CHROMA_PART leaf, including IBC leaves with `skip_flag = 1`.
   §5.20.10.1 `read_cdef()` returns immediately when
   `skip_flag && !cdef_on_skip_txfm_frame_enable`, so the decoder does not read
   that `cdef_idx` symbol (desync — case A) and also does not filter the block
   (recon mismatch — case B). With `--still-cdef-strengths 1` no index symbol
   is coded, so only the filter-side mismatch remains (case B).

`docs/m1-features.md:163-164` ("Intra blocks derive skip = 0, so the still
recon is unchanged by the frame bit") is only true without IBC; SCOUT-CDEF-01
is the counterexample.

### SCOUT-CDEF-02 — SB128/256 + loop restoration (duplicate)

Additional evidence beyond SCOUT-PARTITION-05, from this lane:
`--still-sb128 --still-cdef --still-wiener` (and sb256) aborts during encode
with `av2enc: error: use_bank_k outside the current bank`
(`src/codec/filters/loop_restoration/classes/wiener_syntax.cpp:1365-1372`);
`--still-sb128 --still-cdef --still-chroma-lr` decodes but mismatches. Both are
the same SB-vs-LR-unit-size root cause reported in
`bug-reports/partition.md:32`.

### SCOUT-CDEF-03 — SB128/256 + GDF (duplicate)

Same failure as SCOUT-PARTITION-04 (`bug-reports/partition.md:31`); reproduced
at 640x360 for both sb128 and sb256. The sequence row's
`gdf_unit_matches_sb_size` bit is written from the hardcoded
`c.gdf.sb_size == 64`, while §5.4.10 only codes that bit when
`get_seq_sb_size() == BLOCK_64X64`.

### SCOUT-CDEF-04 — stale CDEF/CCSO SB128 documentation

Repro:
```sh
build/av2enc tests/data/basic_640x360_1f.y4m -o /tmp/a.ivf --still \
  --still-sb128 --still-cdef --still-ccso-edge --recon /tmp/a.y4m
tools/bin/dav2d-macos -q -i /tmp/a.ivf -o /tmp/a.dec.y4m --threads 1   # rc 0
```
Observed: encode/decode succeed and `/tmp/a.y4m` == `/tmp/a.dec.y4m`
(sample-exact); `--still-sb256`, bo-only CCSO, and strengths 2/3/5/7/8 behave
the same.

Expected: `docs/m1-features.md:328-335` ("The still path rejects
`--still-sb128`/`--still-sb256` with CDEF and CCSO, the M1 header pins
`ccso_unit_matches_sb_size = 1` with unit size 6") and the `DEFERRED` row
`docs/negative-results.md:121` should describe the shipped behaviour:
`still_encoder.cpp:4233-4235` sends `CcsoLumaSizeLog2` 7 for sb128/sb256 and
`src/codec/partition/sb128/product_gate.sh` gates it PASS.

## Coverage (no fault found)

Main fixture `tests/data/basic_640x360_1f.y4m` plus the 64x64 baseline matrix,
decoder `--strict 1`, encoder recon vs decoded Y4M compared per plane:

* CDEF: strengths 1/2/3/4/5/7/8; sb64/128/256; depths 0..8; `--still-single`;
  `--still-cdef-on-skip-txfm`; `--no-deblock`; qp 1/90/255; `--adaptive-cdf`.
* CDEF combined with CCSO/edge-CCSO, Wiener, chroma-LR, PC-Wiener, GDF,
  per-segment-Q, delta-Q, palette, IBP, MRL, DIP, MHCCP, TCQ, IST, CCTX,
  TX-partition, IDTX, QM, ext-partitions, sub-8: sample-exact at the default
  64-SB geometry (the SB128/256 combinations are the duplicate rows above).
* CCSO: bands 0..6, band-only and edge-classified, sb64/128/256; per-unit
  `ccso_blk` grids (including partial bottom unit row at y=320..359); with and
  without CDEF, deblock, GDF, LR, segmentation: sample-exact.
* IBC + CCSO and IBC + edge-CCSO: sample-exact (CCSO does not read `skip`).
* `--tiles` still rejects CDEF/CCSO by name (checked in `src/main.cpp`).
* Pinned gate MD5s re-checked for `still_360_cdef_skip_txfm`,
  `still_360_cdef_ccso_wiener`, `still_360_cdef_ccso_edge`,
  `still_360_ccso_edge`, `still_360_cdef_ccso_lr_skip_txfm`: all match.

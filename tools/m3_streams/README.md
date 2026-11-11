# M3 per-tool stream gate

Minimal, reproducible **per-tool** AV2 streams on top of the M1 still-picture
baseline, each decoded by the pinned `dav2d --strict 1` and compared with the
encoder's own reconstruction sample for sample.

This directory owns only the gate. It does not change the encoder, `main.cpp`,
or `CMakeLists.txt`, and nothing here is committed.

## Result matrix

Fixture: `tests/data/m0_64x64_1f.y4m` — 64x64, 8-bit 4:2:0, one frame, 30/1 fps,
`base_q_idx = 90`. Every row enables exactly one tool on top of the `--m1`
baseline, except the baseline control.

| Tool | Status | Flags added to `--m1` | IVF bytes | IVF MD5 | Decoded YUV MD5 | Encoder recon |
|---|---|---|---:|---|---|---|
| M1 baseline (control) | **PASS** | *(none)* | 936 | `0ef9825062a25c4d950ee924113c6f89` | `41b30b8ff21d571a65fee34ebacdd6c4` | BIT-EXACT |
| CDEF | **PASS** | `--m1-cdef --m1-cdef-strengths 1` | 938 | `b795a6f998c7037c95b926d6afdfc89e` | `5db402680605e8fce3c7485e78bbf644` | BIT-EXACT |
| CCSO (band-offset) | **PASS** | `--m1-ccso --m1-ccso-bands 2` | 941 | `e6ce83aa4a7dc3afe300c0bf49e0c186` | `bdb839ee27860b1b6062875b0dcc4de0` | BIT-EXACT |
| RDOQ | **PASS** | `--m1-rdoq` | 794 | `a7ee3a0ac75781ce2a2c7390c3503019` | `4e2a0cf26d593d3c68263e486d4aafa3` | BIT-EXACT |
| Intra luma TX type | **PASS** | `--m1-tx-type` | 944 | `15681508a144eeea88bb8223185d75e9` | `6e75f8f91c4b7ffd86b737751ae1bfbb` | BIT-EXACT |
| Wiener NS (single class, `frame_filters_on=1`) | **BLOCKED** | — | — | — | — | no syntax path |
| Delta-Q (per-superblock qindex) | **BLOCKED** | — | — | — | — | no syntax path |
| Quantization matrices (QM) | **BLOCKED** | — | — | — | — | no syntax path |
| Segmentation | **BLOCKED** | — | — | — | — | no syntax path |

`PASS` means all of:

1. the IVF byte count and MD5 match the pin;
2. `dav2d --strict 1` decodes it without a compliance error;
3. every Y/U/V sample equals `M1Result.recon_y/u/v` from the same encode
   options — zero differing samples, not a PSNR threshold;
4. the IVF MD5 **and** the decoded pixel MD5 differ from the baseline, so the
   tool is actually coded and has an effect.

The decoded MD5 is over the raw Y+U+V plane bytes, not the Y4M container
header, so it does not depend on decoder metadata text.

The gate runs these checks itself; the table is regenerated into
`out/matrix.md` on every run. `m3_cases.json` is the authoritative pin list;
this table mirrors it.

## Exact settings

Pinned 2026-11-11 on macOS arm64 with the CPU reference build below and the
`f9dafbc` decoder. Encoder (CPU reference build):

```sh
cmake -S . -B build-m1 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DAV2ENC_ENABLE_CUDA=OFF -DAV2ENC_ALLOW_CPU_ONLY=ON
cmake --build build-m1 -j
```

All rows use:

```text
input       tests/data/m0_64x64_1f.y4m   (64x64, 8-bit 4:2:0, first frame)
output      <case>.ivf
arguments   --frames 1 --m1 <one tool flag from the matrix>
QP          base_q_idx = 90 (M1 pins it)
partition   M1 defaults: depth 4, min block 8, DC-safe luma search
filters     deblock on, CfL on, adaptive CDF off
```

Decoder, pinned at `f9dafbc` (`tools/dav2d/VERSION.md`):

```sh
dav2d -q -i <case>.ivf -o <case>.decoded.y4m --threads 1 --strict 1
```

`frame_bytes` from the encoder dump plus the 44-byte IVF container overhead
(32-byte file header + 12-byte frame header) must equal the IVF size. The gate
checks this too, so a CLI/dump option mismatch cannot silently "pass".

One case by hand (CDEF):

```sh
build-m1/av2enc tests/data/m0_64x64_1f.y4m -o /tmp/cdef.ivf --frames 1 \
  --m1 --m1-cdef --m1-cdef-strengths 1
tools/m3_streams/build/m1_dump -i tests/data/m0_64x64_1f.y4m \
  -o /tmp/cdef.dump --cdef --cdef-strengths 1
tools/m3_streams/compare_recon.py --ivf /tmp/cdef.ivf --dump /tmp/cdef.dump
```

### Why these are the smallest active streams

* **CDEF** — `CdefStrengths = 1` is the smallest `cdef_params()` form: no
  `cdef_idx` symbols. On this fixture the strength search prunes to one set, so
  `--m1-cdef-strengths 1..8` all produce the same 938-byte stream.
* **CCSO** — `max_band_log2 = 2` (4 bands) is the smallest setting that changes
  reconstruction here. `0` and `1` are valid 938-byte streams, but select zero
  offsets and decode identically to the baseline; they would not verify the
  filter, so the gate uses the smallest *active* form.
* **RDOQ** — same syntax as the baseline; only the trellis-quantized levels
  change. Default `--m1-rdoq-scale 0.125`.
* **TX type** — default `--m1-tx-type-scale 0.125`; luma only. Chroma stays
  `DCT_DCT` (`enable_chroma_dctonly = 1`), and identity types stay excluded
  because `enable_idtx_intra = 0`.

CDEF and CCSO are frame-level filters, so the tile payload stays at 870 bytes;
only the sequence/frame headers grow. RDOQ and TX type change tile bytes (728
and 878).

## Blocked tools — no integrated syntax

Nothing below is faked. The gate checks the source for the hardcoded-off flag
and the absent call sites. If any of those change, the row becomes
`STALE-BLOCK` and the gate fails: the tool then needs a real stream instead of
a blocker note.

### Wiener NS (single class, `frame_filters_on = 1`)

Single class means `num_filter_classes = 1`, so no §7.20.4 pixel-class map is
needed; the filter coefficients are still frame-level (`frame_filters_on = 1`).

**Missing:** `§5.4.10 sequence_filter_config() enable_restoration`,
`§5.18.7.11 lr_params()`, `§5.20.10.4 read_lr()` /
`§5.20.10.6 read_wienerns_filter()`.

The §7.20.3 pixel core exists (`src/encoder/filters/restoration/`), and
`src/encoder/m1/filters_stream/` contains `lr_write_params()` /
`lr_write_block()` syntax helpers. But neither is wired into the M1 writers:
`m1_headers.cpp` hardcodes `enable_restoration = 0` and has no `lr_params()`,
and `m1_syntax.cpp` has no `read_lr()` call. The existing M2 tiled writer,
`src/encoder/m2/post/tiled_frame.cpp`, rejects `enable_restoration` outright.
There is also no encoder decision path for a single-class frame filter. A
single-class stream therefore needs encoder integration, not a different CLI
flag.

### Delta-Q

**Missing:** `§5.18.7.8 delta_q_params()` (`delta_q_present`, `delta_q_res`)
and `§5.20.5.11 read_delta_qindex()` with per-superblock `CurrentQIndex`
bookkeeping.

`m1_headers.cpp` hardcodes `delta_q_present = 0`; `m1_syntax.cpp` never calls
`read_delta_qindex()`. Per-superblock qindex changes dequantization and
reconstruction, so reusing the baseline coefficients would produce wrong
pixels.

### Quantization matrices (QM)

**Missing:** `§5.18.6.2 setup_qm_params()` (`using_qmatrix`,
`pic_qm_num_minus_1`, `qm_y`, `qm_u`, `qm_v`), `§6.17.6.2` QM level selection,
and the dequantization-matrix path.

`m1_headers.cpp` hardcodes `using_qmatrix = 0`. There is no QM level table, no
per-segment QM selection, and the dequantizer does not apply a matrix. Flipping
the header bit alone would desynchronise or decode wrong pixels.

### Segmentation

**Missing:** `§5.18.7.1 segmentation_params()` and
`§5.20.5.7 intra_segment_id()` / `§5.20.5.8 read_segment_id()`.

`m1_headers.cpp` hardcodes `segmentation_enabled = 0`; the tile writer has no
segment-id symbol and no segment feature state. Segmentation changes
qindex/filter features per block, so it needs its own decision and syntax path.

## Files

```text
tools/m3_streams/
  gate.py            main runner: encode -> pin -> strict decode -> recon compare
  compare_recon.py   m1_dump vs decoded Y4M, exact per-sample comparison
  m3_cases.json      pinned manifest: flags, expected bytes/MD5s, blockers
  README.md          this report
  .gitignore         keeps build/ and out/ out of git
  build/             local m1_dump build (gitignored)
  out/               generated IVFs, decoded Y4Ms, results.tsv/json, matrix.md
```

## Run

```sh
# CPU reference build already present in build-m1/
tools/m3_streams/gate.py --dav2d tools/bin/dav2d-macos

# or on the Linux box with the x64 dav2d
tools/m3_streams/gate.py --bin build-m1/av2enc --dav2d tools/bin/dav2d

# reuse an existing build/m1_dump
tools/m3_streams/gate.py --no-build --out-dir /tmp/m3_streams
```

Exit codes: `0` all verified and all blockers still real; `1` a pin, decode,
recon, activation, or blocker check failed; `2` infrastructure (missing binary,
library, decoder, or build failure).

## Scope

This gate proves bitstream syntax and reconstruction correctness for one 64x64
still frame on the CPU M1 path. It makes no rate/quality claim, does not cover
the CUDA path, and does not claim that a passing tool improves compression.

### Coverage boundary

These are deliberately minimal streams: CDEF uses one strength set (no
`cdef_idx` symbols) and CCSO has one 64x64 unit (no left-unit context). The
larger multi-superblock variants live in `tools/m1_verify/feature_gate.sh`
(640x360, CDEF with 1 and 4 sets, CCSO, and combinations). This gate answers a
different question: does each tool, alone on top of the M1 baseline, produce a
stream that decodes and reconstructs exactly?

The decode here is stricter than `tools/validate_dav2d.sh`'s default: it fixes
`--threads 1` and adds `--strict 1`.

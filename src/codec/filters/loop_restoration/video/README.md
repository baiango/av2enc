# M2 loop restoration — CCSO + Wiener NS adapters

Scope: the M2-side glue for the in-loop restoration stage:

```
reconstruct -> deblock -> CDEF -> CCSO (§7.19) -> loop restoration (§7.20) -> output
```

The filter cores and the arithmetic are owned by
`src/encoder/filters/restoration/` and `src/cuda/restoration/`; the bitstream
symbols are owned by `src/encoder/m1/filters_stream/`.  This directory does not
duplicate either.  It provides:

* the M2 sequence/frame-header adapters and the M2 entry points for the
  per-superblock / per-unit tile symbols;
* per-unit decision plumbing and §7.20.4 class-map gating;
* a CPU filter chain adapter for the already-composed
  deblock/CDEF/CCSO frame (`src/encoder/m2/post/`);
* exact tool bit-cost measurement through the real writers;
* the component gates in `video_restoration_test.cpp`.

## API

Include `encoder/m2/restoration/video_restoration.h`.  Namespace
`av2enc::video::restoration`; `fs` abbreviates `av2enc::still::filters_stream`.

| Deliverable | Function |
|---|---|
| sequence §5.4.10 row | `make_sequence_filter_config`, `write_sequence_filter_config`, `read_sequence_filter_config` |
| frame `lr_params()` §5.18.7.11 | `write_lr_params`, `read_lr_params`, `make_lr_header_context` |
| frame `ccso_params()` §5.18.7.12 | `write_ccso_params`, `read_ccso_params`, `make_ccso_header_context` |
| tile `read_ccso()` §5.20.10.2 | `init_ccso_tile`, `write_read_ccso`, `read_ccso` |
| tile `read_lr()` §5.20.10.4 | `init_lr_tile`, `build_lr_decisions`, `write_read_lr`, `read_lr` |
| tile `read_lr_unit()` §5.20.10.5 | `write_read_lr_unit`, `read_lr_unit` |
| tile `read_wienerns_filter()` §5.20.10.6 | `write_read_wienerns_filter`, `read_wienerns_filter` |
| per-unit decisions | `build_lr_decisions`, `lr_block_unit_count` |
| class-map handling | `validate_class_map`, `make_single_class_map`, `validate_no_pc_wiener`, `validate_frame_filter_layout` |
| CPU filter chain | `apply_ccso_frame`, `apply_lr_frame`, `apply_restoration_frame`, `make_ccso_frame_params` |
| bit cost | `measure_restoration_cost` |

`VideoRestorationTools` is the sequence-level selection:

```cpp
VideoRestorationTools tools;
tools.ccso = true;                 // enable_ccso + 64x64 units
tools.wiener_ns = true;            // enable_restoration
tools.frame_filters = true;        // frame_filters_on instead of per-unit
tools.num_filter_classes = 1;      // gated single-class path
tools.chroma_lr = true;            // lr_tools_uv_present
```

Header rows are byte-identical to the M1 module because they call the same
`fs::write_*` functions with `single_picture_header_flag = false`.

## M2 stream integration call sites (core files, not edited here)

The M2 core writer has no restoration hook.  A minimal 2-frame stream with
CCSO/Wiener is **not possible without editing `src/encoder/m2/` core files**;
the exact edits are:

1. **Sequence header** — `src/encoder/m2/video_headers.cpp`,
   `write_sequence_header()` §5.4.10 block (the six filter bits around the
   `enable_restoration`/`enable_ccso` lines).  Replace that block with:

   ```cpp
   av2enc::video::restoration::write_sequence_filter_config(writer, tools, &error);
   ```

   The adapter writes `disable_loopfilters_across_tiles`, `enable_cdef` (0),
   `enable_gdf` (0), the LR tool-disable row, `enable_ccso`,
   `ccso_unit_matches_sb_size`, `cdef_on_skip_txfm_*` and `df_par_bits_minus_2`.

2. **Frame header** — same file, `write_frame_header_syntax()`:
   after the `apply_deblocking_filter[]` / `df_delta_q_present[]` bits, before
   `tx_mode_select`, insert (in this order):

   ```cpp
   // cdef_params() would go here; M2 core currently keeps CDEF off.
   av2enc::video::restoration::write_lr_params(writer, lr_params, lr_context, &error);
   av2enc::video::restoration::write_ccso_params(writer, ccso_params, ccso_context, &error);
   ```

   `lr_context.frame_is_intra` must be `h.keyframe`; frame-level filters are
   intra-only (see blockers).  `ccso_context.tile_start_or` is the OR of
   `MiColStarts[] | MiRowStarts[]` (0 for one tile).

3. **Tile, per superblock** — `src/encoder/m2/video_syntax.cpp`, `Writer::run()`
   in the superblock loop, **before** `write_node()`:

   ```cpp
   std::vector<fs::LrUnitDecision> decisions;
   av2enc::video::restoration::build_lr_decisions(lr_state, node.mi_row, node.mi_col,
                                       sb_mi, sb_mi, 0, 0, &decisions, &error);
   av2enc::video::restoration::write_read_lr(enc_, lr_state, node.mi_row, node.mi_col,
                                  sb_mi, sb_mi, 0, 0, decisions, &error);
   ```

   `read_lr()` is called once per superblock, before `read_partition()`
   (§5.20.10.4), not once per frame.  For a multi-tile frame pass that tile's
   `MiRowStart`/`MiColStart` instead of the two trailing zeros.

4. **Tile, per coding block** — `Writer::write_block()` (the leaf emitter),
   after the `read_cdef()` position and before `read_intra_y_mode()`:

   ```cpp
   const int blk[3] = {1, 1, 1};  // per-unit CCSO decisions
   av2enc::video::restoration::write_read_ccso(enc_, ccso_state, node.mi_row,
                                    node.mi_col, blk, &error);
   ```

   `write_read_ccso()` is a no-op off a CCSO unit origin, matching
   §5.20.10.2.

5. **Encoder decisions/reconstruction** — in `src/encoder/m2/video_frame.cpp` the
   reconstruction for each frame must be filtered after `run_video_post_filters()`
   using `apply_restoration_frame()`, and the filtered output (not the raw
   reconstruction) must become the reference plane.  This is the same
   curr/cdef/lr buffer discipline as M1: `curr` is post-deblock, `cdef` is
   post-CDEF/CCSO, `lr` is the output.

These call sites are intentionally listed rather than applied: the M2 core,
`video_headers.cpp`, `video_syntax.cpp`, `video_frame.cpp`, `main.cpp` and the root
`CMakeLists.txt` are outside this workstream.

## Component gates

```sh
./src/encoder/m2/restoration/build_and_test.sh
SANITIZE=1 ./src/encoder/m2/restoration/build_and_test.sh
WERROR=1   ./src/encoder/m2/restoration/build_and_test.sh
```

The gate covers:

* sequence filter row write/read round trip (key and P);
* `ccso_params()` band-offset-only and edge-classified forms, offset-index
  equality;
* `lr_params()` frame filters (coefficients) and per-unit mode;
* `read_ccso()` grid equality on a 128×80 MI plane;
* `read_lr()` unit-type and coefficient equality, including the direct
  `read_lr_unit()` / `read_wienerns_filter()` aliases;
* class-map gates (single-class accepted, multi-class without a map rejected,
  explicit map accepted, PC-Wiener rejected);
* two-frame syntax rows for key and P (P-frame CCSO codes the reuse flags);
* exact header + standalone symbol cost measurement;
* CPU adapter parity: adapter-driven CCSO+Wiener is byte-identical to direct
  `restoration::ccso_frame()` + `loop_restoration_frame()` calls.

Current result on this host (macOS, no CUDA toolchain):

```
PASS: 282 checks, 0 failures            (also under ASan+UBSan, -Werror)
```

## Blockers

### PC-Wiener (§7.20.4)

`pc_wiener_sub_classify2.h`, `pc_wiener_lut_to_class.h`,
`pc_wiener_sub_classify.h`, `pc_wiener_filters.h` are referenced by §9.8 but
are not printed in the spec text.  Hand-typing them is forbidden.  The adapters
reject `RESTORE_PC_WIENER` at the frame type and at every switchable unit.  No
bitstream with PC-Wiener can be produced or consumed.

### Adaptive frame-filter group search (§5.20.10.6)

The writer supports the deterministic **group-0** layout: all classes are in
the first group, `use_alt_group` is never coded, and each class `c` matches
the all-zero filter (0) or a previous class.  `validate_frame_filter_layout()`
checks this.  The encoder does not search group assignments or reference-bank
reuse, so it cannot claim the rate-optimal group layout.  This is a rate
limitation, not a syntax gap: the produced layout is valid.

### Multi-class class map

The bitstream signals only `num_filter_classes_idx`; the decoder derives
`SubclassLookup` from the missing §7.20.4 tables.  The single-class layout
(`num_filter_classes = 1`) needs no map and is the gated path.  Multi-class is
accepted only when the caller supplies a `mi_cols × mi_rows`
`filter_class_grid` and a 16-entry `subclass_lookup` (or the legacy
pre-looked-up `subclass_grid`); `validate_class_map()` enforces shape and
range.  The CPU/GPU core also accepts the 256-entry `FilterClass` range used by
`Pc_Wiener_Lut_To_Class`, but this adapter's `fs::WienerNsClassMap` is the
16-entry form; a 256-entry map cannot be produced without those tables and is
therefore not exposed.

## Quality at QP 90 vs no filters

Source/recon pairs: M1 QP90 encode, decoded with `dav2d`.  Measured on the real
deblock→CDEF chain: `--recon-pre` supplies the pre-CDEF frame for CCSO/§7.20
sourcing and `--recon` is the CDEF decode.  PSNR is the integer plane PSNR
from `restoration_quality_bench.cpp`; avg is sample-weighted over the 4:2:0
planes.  CCSO uses `max_band_log2 = 3` (72 offsets/plane); Wiener is a
single-class frame-level least-squares fit.  `both` is CCSO then Wiener in
normative order.

| Input | none Y/U/V (avg) | CCSO Y/U/V (avg) | Wiener Y/U/V (avg) | both Y/U/V (avg) |
|---|---|---|---|---|
| 64×64 | 42.58 / 45.80 / 45.12 (43.54) | **42.78 / 46.26 / 45.37 (43.80)** | 42.89 / 46.09 / 46.52 (44.02) | **42.75 / 47.39 / 46.92 (44.22)** |
| 640×360 | 45.24 / 52.47 / 50.68 (47.35) | **45.41 / 52.52 / 50.70 (47.47)** | 45.27 / 52.85 / 51.31 (47.54) | **45.41 / 52.81 / 51.32 (47.63)** |
| 256×256 | 47.40 / 51.46 / 50.01 (48.51) | **47.58 / 51.59 / 50.08 (48.67)** | 47.32 / 51.83 / 50.90 (48.67) | **47.48 / 51.92 / 50.92 (48.79)** |

Luma-only deltas vs none: CCSO +0.21 / +0.17 / +0.18 dB; both +0.17 / +0.18 /
+0.08 dB.  Wiener alone is roughly flat on luma (−0.08 dB at 256×256) but wins
chroma (+0.46 dB U at 64×64).  Against CCSO-only, adding the frame Wiener
changes luma by −0.03 / +0.00 / −0.10 dB; the average gain comes from chroma.
This is filter *potential*; selection should be per-unit RDO before enabling
Wiener by default.

## Bit cost

Two independent measurements:

1. **Real integrated CCSO stream (M1 QP90, `--still` vs `--still-ccso`):**

   | Input | baseline | CCSO | delta |
   |---|---:|---:|---:|
   | 64×64 | 936 B | 970 B | +34 B |
   | 640×360 | 13,355 B | 13,397 B | +42 B |
   | 256×256 | 5,336 B | 5,373 B | +37 B |

   This is the actual cost of the sequence flag, `ccso_params()` and the
   per-unit `read_ccso()` symbols in a gated stream.  The M1 default is
   band-offset-only with 64 bands; the fitted M2 row below uses edge classes
   with `max_band_log2 = 3`, so the two rows are not the same configuration.

2. **M2 syntax cost for the fitted tools** (`measure_restoration_cost`, exact
   header bits + standalone symbol bytes up to coder termination;
   `max_band_log2 = 3`, single-class frame Wiener):

   | Input | CCSO | Wiener NS | both |
   |---|---:|---:|---:|
   | 64×64 | 51 B | 28 B | 79 B |
   | 640×360 | 55 B | 33 B | 88 B |
   | 256×256 | 53 B | 25 B | 78 B |

   The header dominates (CCSO 339–379 bits; LR 149–196 bits); the symbol
   streams are 3–13 B.  A 936-byte 64×64 stream therefore cannot hide a ~79-byte
   filter header; 640×360 and 256×256 are the plausible targets.  This is not a
   BD-rate number: the M2 tile-routing edit above is required to place those
   symbols in a real stream.

## Files

```text
src/encoder/m2/restoration/
  video_restoration.h         M2 adapters, gates, cost API
  video_restoration.cpp       implementation (thin over filters_stream + cores)
  video_restoration_test.cpp  component gates
  build_and_test.sh        standalone CPU build/run (no root CMake changes)
  README.md                this file
```

## Verification status

* CPU adapters and gates: **PASS on this host** (282 checks, ASan/UBSan clean).
* Real deblock→CDEF→CCSO→Wiener CPU chain: **PASS** with
  `real_recon_oracle_test --pair` on 64×64 and 640×360 QP90 (see
  `src/cuda/m1/restoration/README.md`).
* GPU stage parity: the CUDA test is ready but **was not executed here**
  (macOS has no `nvcc`).  The changed CUDA translation units pass a clang
  `--cuda-host-only` syntax check against a stubbed `cuda_runtime.h`.  Run the
  real parity on the 3060 box with the command in the M1 restoration README.
* Full 2-frame M2 stream with CCSO/Wiener: **not produced**; the core tile
  writer needs the four call-site edits listed above.

# PC-Wiener stream syntax and encoder decision

Spec-only module for AV2 v1.0.0 luma PC-Wiener (`RESTORE_PC_WIENER`, §7.20.4)
and the frame-level pixel-classified Wiener mode that uses the recovered PC-Wiener
classifier and translated filter bank.

No production core file, CMake target or `src/main.cpp` entry is changed.  The
module is self-contained and reuses only read-only headers/tables:

* `encoder/filters/spec_tables/pc_wiener.{h,cpp}` — §7.20.4 classifier and the
  25-tap recovered `Pc_Wiener_Filters` bank;
* `encoder/filters/spec_tables/pc_wiener_tables.h` — the recovered
  `Pc_Wiener_Lut_To_Class`, `Pc_Wiener_Sub_Classify` and
  `Pc_Wiener_Sub_Classify2` tables;
* `encoder/filters/restoration/wiener_core.h` — §7.20.2 sample sourcing and the
  32-tap §7.20.3 Wiener NS filter, used to evaluate frame-class mode;
* `encoder/filters/restoration/classes/wiener_cdfs.h` — normative §9.3 CDFs.

## Normative syntax

### §5.4.10 `sequence_filter_config()`

`SequenceFilterConfig` / `WriteSequenceFilterConfig` / `ReadSequenceFilterConfig`
write and parse the complete filter row.  The PC-Wiener-relevant part is:

```
enable_restoration                    f(1)
if (enable_restoration) {
  lr_tools_disable[0][RESTORE_PC_WIENER]        f(1)
  lr_tools_disable[0][RESTORE_WIENER_NONSEP]    f(1)
  lr_tools_disable[1][RESTORE_PC_WIENER] = 1    (inferred, never coded)
  lr_tools_uv_present                           f(1)
  if (lr_tools_uv_present)
    lr_tools_disable[1][RESTORE_WIENER_NONSEP]  f(1)
  else
    lr_tools_disable[1][RESTORE_WIENER_NONSEP] =
        lr_tools_disable[0][RESTORE_WIENER_NONSEP]
}
```

The audit's `lr_tools_disable[0][RESTORE_PC_WIENER]` is now signallable as 0.

### §5.18.7.11 `lr_params()`

`LrParams`, `WriteLrParams`, `ReadLrParams` and `LrParamsBits` implement:

* `FrameRestorationType[plane]` through the `ns(n)` tool list, including
  `RESTORE_PC_WIENER` when `lr_tools_disable[0][RESTORE_PC_WIENER] == 0`;
* `frame_filters_on[plane]`, `temporal_pred_flag`, `rst_ref_pic_idx` bits
  (temporal prediction is rejected: no reference state is in this module);
* `num_filter_classes_idx` f(3) and
  `NumFilterClasses = Decode_Num_Filter_Classes[idx]`;
* `SubclassLookup = Pc_Wiener_Sub_Classify2[get_filter_set_index(base_q_idx)]
  [num_filter_classes_idx]` (256 entries) when luma frame filters are on;
* `lr_{luma,chroma}_use_{half,max,quarter}_size` and the final
  `LoopRestorationSize`;
* §5.20.10.6 frame filters: group prediction, match indices, merged flags,
  subsets, `uv_sym` and coefficients.

Important normative detail: when `FrameRestorationType[0] == RESTORE_PC_WIENER`
the spec does **not** code `frame_filters_on` (that branch is only for
`RESTORE_WIENER_NONSEP`/`RESTORE_SWITCHABLE`).  A writer that emits the bit would
desynchronise the decoder; `WriteLrParams` rejects that combination.

### §5.20.10.5 `read_lr_unit()`

`WriteUnit` / `ReadUnit` implement:

* `use_pc_wiener` S() for a `RESTORE_PC_WIENER` frame;
* `use_wiener_ns` S() for a `RESTORE_WIENER_NONSEP` frame;
* the `flex_restoration_type` loop for `RESTORE_SWITCHABLE`, where tool 0 is
  `RESTORE_NONE`, tool 1 is `RESTORE_PC_WIENER`, and "no flex symbol set" is
  `RESTORE_WIENER_NONSEP`;
* per-unit §5.20.10.6 `merged`, bank refs, subsets and coefficients for
  `RESTORE_WIENER_NONSEP` units when frame filters are off.

PC-Wiener carries **no** class, scale, subset or coefficient syntax.  The
decoder derives the class from reconstructed pixels and `base_q_idx`
(`Pc_Wiener_Lut_To_Class`, `Pc_Wiener_Sub_Classify`) and uses the fixed
`Pc_Wiener_Filters` bank.  The audit phrase "class/scale indices, subset and
coefficients" describes the frame-filter (§5.20.10.6) syntax, which is needed by
the frame-level PC-Wiener translation mode below, not by `RESTORE_PC_WIENER`.

### Frame-level pixel-classified mode

When `frame_filters_on[0] == 1` and `NumFilterClasses > 1`, §7.20.4 runs with
`skipFilter = 1` on each 4×4 block, then §7.20.3 filters with
`FrameLrWienerNs[0][SubclassLookup[FilterClass]]`.  The encoder can point a class
at a translated PC-Wiener filter:

```
get_translated_pc_wiener(m, j) =
  Clip3(Wiener_Ns_Taps_Min[0][j],
        Wiener_Ns_Taps_Min[0][j] + (1 << Wiener_Ns_Taps_K[0][j]) - 1,
        Pc_Wiener_Filters[0][Shuffled_Index[m]][j])
```

`match_index = NumFilterClasses + m`, `merged = 1` copies that translated set
without coefficient bits.  This path is only available when
`lr_tools_disable[0][RESTORE_PC_WIENER] == 0` (the PC-Wiener group in
`search_frame_filters`).

## Encoder decision

`DecidePerUnitPcWiener`:

1. builds a `PcWienerFrame` from `CurrFrame`/`CdefFrame` and runs the recovered
   §7.20.4 classifier on every 4×4 block;
2. applies the fixed 25-tap bank to every block and computes per-unit
   `SSE(identity)` vs `SSE(filtered)` against the target;
3. turns a unit on when `filtered_sse + lambda * 1 bit < identity_sse`;
4. writes `RESTORE_PC_WIENER` for luma and reports the exact header bits
   (`LrParamsBits`) and exact tile bits (a scratch §8.2 encoder including the
   §8.2.4 marker/padding).

`DecideFrameClassPcWiener`:

1. derives `FilterClass` on the luma MI grid;
2. maps it through `Pc_Wiener_Sub_Classify2[filter_set][num_classes_idx]`;
3. for each coefficient class, selects one of the 64 translated PC-Wiener
   filters by SSE (sampled search + exact winner) and sets `match_index` /
   `merged`;
4. reports exact header and tile bits.

Both decisions use `Pc_Wiener_Sub_Classify` only through the recovered
`spec_tables` classifier; nothing is invented.

## Tests

`build_and_test.sh` builds and runs `pc_wiener_stream_test.cpp`: **223 checks**.

Coverage:

* sequence row write/read and exact bit counts for restoration on/off and
  `lr_tools_uv_present`;
* `lr_params()` for `RESTORE_PC_WIENER`, frame-class mode, and switchable plus
  chroma Wiener NS;
* per-unit `use_pc_wiener`, switchable PC/NS/NONE, per-unit bank refs/merged,
  subsets, `uv_sym` and coefficients;
* frame-filter class-to-class references (`match_index = c` copies class
  `c-1`), including merged chains;
* independent reader: write → parse → re-write byte equality, plus field-by-field
  comparison;
* the recovered classifier, per-unit on/off decisions, frame-class translated
  matches, and `ApplyPerUnitPcWiener`/`ApplyFrameClassWiener`;
* pinned goldens:
  * sequence: `1050`
  * PC `lr_params`: `40`
  * frame-class `lr_params`: `bc3f7aedcb76acd8af5aad4a74a8d0ffff80` (137 bits)
  * switchable+chroma `lr_params`: `d886c1822000000000` (72 bits)
  * PC unit on: `c000`
  * switchable unit sequence: `c1ca9a951f50f26539e48f280000`

`SANITIZE=1` and `WERROR=1` both pass.

## Decoder gate

`gate/build_and_test.sh` vendors `src/encoder/m2/m2_syntax.cpp` into the build
directory with one §5.20.10.4 `read_lr()` hook (the M2 core is not edited),
builds a one-frame fixture, decodes a feature-off baseline with the pinned
`dav2d`, runs the decision on the decoded luma, rebuilds the same frame with the
LR syntax, and decodes with:

```
dav2d --strict 1 --threads 1
```

The gate compares every decoded luma sample with the encoder's own
`spec_tables` §7.20.4 output (per-unit mode) or `wiener_core.h` §7.20.3 output
(frame-class mode), and chroma with the feature-off frame.

### Result

The pinned `tools/bin/dav2d-macos` **accepts both luma PC-Wiener forms**:

| Case | Mode | Luma diffs | Chroma diffs | Result |
|---|---|---|---|---|
| 640×360 | per-unit PC-Wiener | 0 | 0 | PASS |
| 640×360 | frame-class (16) | 0 | 0 | PASS |
| 1280×704 | per-unit PC-Wiener | 0 | 0 | PASS |
| 1280×704 | frame-class (16) | 0 | 0 | PASS |

The gate does **not** need the decoder-limited fallback for luma PC-Wiener.
(If a future `dav2d` rejects it, the gate prints the decoder error, verifies
exact self-parse and feature-off decode invariance, and only exits 0 when those
pass.)

### 720p fixture limit

The vendored M2 fixture (not the PC-Wiener syntax) rejects `1280×720` with
`Error parsing frame header` before the LR stage; `1280×704` and `640×360`
pass.  The existing `lr_chroma` gate has the same `1280×720` failure, so this is
a fixture limitation, not a PC-Wiener syntax limitation.  The gate prints this
explicitly when a non-multiple-of-64 height fails to decode, and the 720p-class
measurement uses **1280×704**.  `1280×688` is also rejected by the fixture;
`1280×704` is the closest working 720p-class size.

### Measured quality and bit cost

Fixture: first frame of `testdata/shine_1280x720_60f.y4m`, downscaled; a
16×16-leaf DC-only M2 syntax fixture (blocky by construction).  PSNR-Y “vs
source” is therefore dominated by the fixture’s lost detail; the meaningful
filter number is “vs denoise target”, a 3×3-smoothed version of the baseline
reconstruction used as the encoder decision target.  CVVDP is measured on the
same source.

| Size | Mode | Units on | Header bits | Tile bits | IVF bytes (base→PC) | PSNR-Y vs source | PSNR-Y vs target | CVVDP (JOD) |
|---|---|---:|---:|---:|---|---|---|---|
| 640×360 | per-unit | 6 / 60 | 4 | 48 | 5326 → 5338 (+12) | 15.286 → 15.286 | 54.427 → 55.278 | 5.4438 → 5.4437 |
| 640×360 | frame-class 16 | 60 / 60 | 133 | 24 | 5327 → 5347 (+20) | 15.286 → 15.288 | 54.427 → 62.418 | 5.4438 → 5.4413 |
| 1280×704 | per-unit | 30 / 220 | 4 | 152 | 22304 → 22349 (+45) | 15.440 → 15.440 | 50.312 → 50.794 | 5.6888 → 5.6888 |
| 1280×704 | frame-class 16 | 220 / 220 | 131 | 24 | 22305 → 22338 (+33) | 15.440 → 15.442 | 50.312 → 58.762 | 5.6888 → 5.6877 |

The frame-class mode is much more effective on this fixture because it fits one
translated bank entry per class (up to 64 entries) and uses the classifier’s
256→16 mapping; the per-unit mode is limited to the single fixed bank selected
by `base_q_idx` and the unit-level on/off decision.  CVVDP changes are below
0.002 JOD, consistent with the fixture’s low absolute quality.

## API / call sites

Public header: `pc_wiener_stream.h`.

| Function | Use |
|---|---|
| `WriteSequenceFilterConfig` / `ReadSequenceFilterConfig` / `SequenceFilterConfigBits` | §5.4.10 |
| `WriteLrParams` / `ReadLrParams` / `LrParamsBits` | §5.18.7.11 |
| `InitTileState`, `WriteUnit` / `ReadUnit`, `UnitRangeFor`, `WriteBlock` / `ReadBlock` | §5.20.10.4–.6 |
| `BuildFilterClassGrid` | §7.20.4 classifier on a reconstruction |
| `DecidePerUnitPcWiener` / `DecideFrameClassPcWiener` | encoder decision + exact bit count |
| `ApplyPerUnitPcWiener` / `ApplyFrameClassWiener` | expected reconstruction for tests/gates |
| `TranslatedPcWiener`, `TranslatedPcWienerSet`, `PcWienerSubclassLookup`, `FilterSetIndex` | §5.20.10.6 translation and class map |

There are **no production call sites** yet: `m1_headers.cpp` and
`m2_headers.cpp` still write `enable_restoration = 0`, and the production
restoration core still rejects `RESTORE_PC_WIENER`.  The intended integration is:

1. replace the sequence filter row with `WriteSequenceFilterConfig`;
2. replace the frame `lr_params()` bytes with `WriteLrParams`;
3. call `WriteBlock` once per superblock at the §5.20.10.4 position (before
   `read_partition()`), exactly as the gate hook does;
4. feed `DecidePerUnitPcWiener` or `DecideFrameClassPcWiener` from the
   post-CDEF reconstruction and the source picture.

## Files

```
pc_wiener_stream.h/.cpp        syntax + decision + reconstruction helpers
pc_wiener_stream_test.cpp      223-check unit test with pinned goldens
build_and_test.sh              unit test build/run
gate/pc_wiener_gate.cpp        closed-loop dav2d gate + quality measurement
gate/vendor_m2_syntax_pc.py    build-only M2 tile writer hook
gate/build_and_test.sh         gate build/run
```

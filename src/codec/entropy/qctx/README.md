# Coefficient-CDF quantization contexts (`qctx`)

This module owns the four `COEFF_CDF_Q_CTXS` slices of the AV2 v1.0.0 §9.3
default coefficient CDFs.  It exists so the M1/M2 writers can stop pinning
`base_q_idx = 90` (slice 0) without hand-editing CDF values.

Everything is generated from `spec/tables/default_cdf.json`; nothing here is
hand-typed.

## Normative selection

§8.3.1 `init_coeff_cdfs()` selects one slice **per frame** from the raw frame
`base_q_idx`:

| raw `base_q_idx` | slice |
|---|---|
| `<= 90` | 0 |
| `91 .. 140` | 1 |
| `141 .. 190` | 2 |
| `>= 191` | 3 |

It is **not** derived from a tile qindex, delta-Q qindex, segmentation qindex,
or the per-block effective `qindex`.  Those change the quantizer, not the
frame-level CDF initialization.  The API therefore takes an `int base_q_idx`
and has no upper clamp (10-bit MaxQ 303 still selects slice 3).

## Files

| File | Kind | Contents |
|---|---|---|
| `generate_qctx_tables.py` | script | discovers every `COEFF_CDF_Q_CTXS` array in the pinned JSON, emits the header, and implements `--check` |
| `qctx_cdf_tables.h` | generated | 27 tables × 4 slices = 18,868 `uint16_t` values, plus per-slice typed views and descriptors |
| `qctx.h` | hand-written | slice selection, `CoeffCdfSlice` accessors, `copy_cdf` |
| `qctx_test.cpp` | test | selection boundaries, every slice/context CDF validity, slice-0 byte comparison against shipping tables, view/copy plumbing |
| `qctx_decode_gate.cpp` | test | normative §8.2 encode→decode round trip for every slice/context |
| `decode_gate.sh` | test | real `dav2d` end-to-end gate at QPs spanning slices 0–3 |
| `run_tests.sh` | driver | runs all of the above without touching `CMakeLists.txt` |

### Generated tables

The generator discovers arrays whose first declared dimension is
`COEFF_CDF_Q_CTXS` (currently 27), keeps the leading 4-slice axis, and emits
nested braces in the same style as `basic_cdf_tables.h` / `rdoq_cdf_tables.h`.
It also emits:

* `struct CoeffCdfSlice` — 27 pointer-to-inner-array fields named exactly like
  the M1/M2 `CdfState` members (`txb_skip`, `coeff_base`, `eob_pt_16`, …);
* `kCoeffCdfSlices[4]` — the four typed views;
* `kCoeffCdfTables[]` — descriptor list (`name`, slice-0 scalar pointer,
  scalar stride, symbol count) used by the exhaustive tests.

Five arrays are emitted for completeness even though the current M1/M2 syntax
does not use them yet: `Coeff_Base_Bob`, `Coeff_Base_Idtx`, `Coeff_Br_Idtx`,
`Coeff_Base_Ph`, `Idtx_Sign`.

## API

```cpp
#include "encoder/m1/qctx/qctx.h"

using namespace av2enc::still::qctx;

int slice = slice_for_base_q_idx(config.base_q_idx);   // 0..3
const CoeffCdfSlice& q = tables_for_base_q_idx(config.base_q_idx);

copy_cdf(cdfs.txb_skip,       *q.txb_skip);
copy_cdf(cdfs.coeff_base,     *q.coeff_base);
copy_cdf(cdfs.coeff_base_lf,  *q.coeff_base_lf);
copy_cdf(cdfs.dc_sign,        *q.dc_sign);
// ... every field name matches CdfState.
```

`q.field` is `const uint16_t (*)[inner...]`; `*q.field` is the slice-removed
array reference, so the existing `memcpy`-based `copy_array` in the writers
works unchanged.  `copy_cdf` is a compile-time shape-checked alternative.

Also available: `tables_for_slice(int)` (clamped), `table_slice(info, slice)`
(descriptor-driven, for tools/tests), and `kSliceCount`.

## Generate / verify

```sh
python3 src/encoder/m1/qctx/generate_qctx_tables.py          # write header
python3 src/encoder/m1/qctx/generate_qctx_tables.py --check  # gate
```

`--check` currently reports:

```
qctx table check OK: 27 tables, 4 slices, 18868 values
  slice 0 vs basic_cdf_tables.h: 20 tables compared
  slice 0 vs still_cdf_tables.h: no coefficient CDFs present (skipped)
  slice 0 vs rdoq_cdf_tables.h: 22 tables compared
  slice 0 vs syntax writers: 2 EOB tables compared
```

It compares numeric values (not formatting) and exits non-zero on any stale
header or mismatch.  `still_cdf_tables.h` genuinely contains no coefficient CDFs
(intra/filter CDFs only); the two EOB tables that the M1 writer historically
defined locally are checked in `still_syntax.cpp`/`video_syntax.cpp` and in the rdoq
header.

## Tests

```sh
src/encoder/m1/qctx/run_tests.sh
```

Results on the current tree:

```
qctx_test: PASS (337 checks)
qctx_decode_gate: PASS (3452 distributions, 15528 symbols)
qctx_decode_gate: PASS (7 streams, slices 0..3)
```

* `qctx_test` checks every boundary (`-1, 0, 89, 90, 91, 140, 141, 190, 191,
  255, 303`), every generated table, every slice and every context row with
  `av2::cdf_is_valid` (monotonic boundaries, rate index, adaptation counter),
  and compares slice 0 byte-for-byte (`memcmp`) with `basic_cdf_tables.h` (20
  tables) and `rdoq_cdf_tables.h` (22 tables, including `Eob_Pt_16/32`).
* `qctx_decode_gate` encodes and decodes every one of the 3,424
  table/slice/context distributions plus 28 selections through the public API,
  using the normative §8.2 decoder state machine.
* `decode_gate.sh` is the real gate: it builds nothing, but encodes the same
  64×64 still at `base_q_idx` 90, 91, 140, 141, 190, 191, 255 and requires the
  pinned `dav2d` to decode each stream.  It also requires the bitstreams to be
  distinct, so a silent QP clamp cannot pass.  It skips (exit 77) when no
  encoder/decoder/test media is available; `run_tests.sh` treats that as a
  clean skip.  Set `QCTX_SKIP_END_TO_END=1` to skip it.

The decoded streams at all seven QPs pass `tools/validate_dav2d.sh`.  The same
gate was also run on the 640×360 fixture at QP 90/91/141/191; all four streams
decoded (13,355 / 13,775 / 3,428 / 826 bytes).

## Core change required to stop pinning QP 90

### M1 syntax writer — already wired in the current working tree

`src/encoder/m1/still_syntax.cpp` now includes `encoder/m1/qctx/qctx.h` and does
exactly this (shown as the canonical change):

```cpp
void init_cdfs(CdfState& c, int qctx_slice) {
  const qctx::CoeffCdfSlice& q = qctx::tables_for_slice(qctx_slice);
  // ... non-coefficient CDFs unchanged ...
  copy_array(c.txb_skip, *q.txb_skip);
  // ... all 22 coefficient copies use *q.<same field name> ...
}

// Writer constructor:
init_cdfs(cdfs_,
          qctx::slice_for_base_q_idx(static_cast<int>(config_.base_q_idx)));
```

Slice 0 is byte-identical to the pre-change pinned tables (enforced by
`qctx_test` and `--check`).

### M1 still-picture path — already wired in the current working tree

For **real** QP > 90 streams the decision/reconstruction path must also use
`qlookup(base_q_idx)`, otherwise it still quantizes with the QP-90 step even
though the CDF slice is correct.  The current tree does this in
`src/encoder/m1/still_encoder.cpp`:

* replaces the `base_q_idx != 90` rejection with a `base_q_idx == 0` (lossless)
  rejection;
* uses `rc::kQlookupQ0To255[options.base_q_idx]` for the forward quantizer and
  the matching inverse step.

Without that, the encoder still emits decodable streams but they are not the
requested QP.  The `decode_gate.sh` uniqueness check catches an accidental
clamp.

### M2 syntax writer — still pinned (exact change)

`src/encoder/m2/video_syntax.cpp` still copies slice 0.  Apply the same pattern:

```cpp
#include "encoder/m1/qctx/qctx.h"

void init_cdfs(CdfState& c, int qctx_slice) {
  const qctx::CoeffCdfSlice& q = qctx::tables_for_slice(qctx_slice);
  // ... non-coefficient CDFs unchanged ...
  copy_array(c.txb_skip, *q.txb_skip);
  // ... all 22 coefficient copies use *q.<same field name> ...
}

// Writer constructor:
init_cdfs(cdfs_, qctx::slice_for_base_q_idx(config_.base_q_idx));
```

`VideoConfig::base_q_idx` already exists, so no header change is needed.

### M2 rate-control policy — still pinned

`src/encoder/m2/post/rc_adapter.cpp`:

```cpp
bool video_coeff_cdf_slice_available(int qctx) noexcept {
  return qctx >= 0 && qctx < av2enc::still::qctx::kSliceCount;  // was qctx == 0
}
```

and raise `VideoRcPolicy::max_base_q_idx` from
`kM2MaxCoeffCdfSlice0QIndex` (90) to the format MaxQ (`rc::kMaxQIndex`, 255 for
8-bit).  `bit_depth_checklist.cpp`'s “M2 ships coefficient-CDF slice 0 only”
diagnostic should then be removed.  `require_video_coeff_cdf_slice` can stay true
once all four slices exist.

### M1 GPU path — still pinned

`src/cuda/m1/still_gpu.cu` rejects `options.base_q_idx != 90` (line ~1031) and
`src/cuda/m1/still_gpu.h` documents “M1 pins 90”.  The GPU path funnels through
the same CPU syntax writer for assembly, so lifting that guard is sufficient
for bitstream correctness.  Keep `--still-gpu` frozen until the GPU quantizer uses
`qlookup(base_q_idx)` like `still_encoder.cpp`.

### RDO rate models — quality caveat, not a bitstream bug

`src/encoder/m1/rdo_decide/transform_stream.cpp` and
`src/encoder/rdoq/generate_rdoq_cdf.py` still use qctx-0 tables for candidate
bit estimates.  At QP > 90 the syntax writer is correct, but TX-type/RDOQ
decisions are ranked with the wrong static model.  To fix, pass the slice index
into those rate models and index the same `kCoeffCdfTables` rows.  This does
not affect decodability.

## No commit

These files are new and uncommitted, by design.  The shared CDF headers,
`main.cpp`, and `CMakeLists.txt` were not edited.

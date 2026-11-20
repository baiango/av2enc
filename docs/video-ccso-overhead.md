# CCSO parameter traffic (`--video-ccso-header-rd`)

`§5.18.7.12 ccso_params()` is the only per-frame filter row whose size is
content-dependent: it codes `2^ccso_max_band_log2` `tu(7)` offset symbols per
enabled plane. The nightly aggregate
(`--fast-decode --video-ccso-rd --video-ccso-bands 6 --video-ccso-lambda 22
--video-ccso-refit 2 --video-ccso-unit-bits 2`) pinned the band count at 64 and
the rate-aware fit priced only the *extra* bits of a non-zero band offset, so
the row was invisible to the decision that produced it. This document measures
the row, shows where the bytes go, and ships the opt-in fix.

Everything below is CPU-only, single tile, 8-bit 4:2:0, `--qp 110`, 60 frames of
`testdata/nasa_480x240_30fps_60f.y4m`, measured on the M4 host.

## 1. What the row actually costs

`AV2ENC_TRACE_CCSO=1` reports the exact bit split of every `ccso_params()` row
(the accounting is `filters_stream::CcsoHeaderBits`, pinned by
`filters_stream_test`), plus the change against the previous frame's table.

Pin (`--fast-decode`, MD5 `5e316100ce7fd709b4298ccfebfffccd`, 24,885 B):

| Part | Bits (60 frames) | Bytes | Per enabled frame |
|---|---:|---:|---:|
| `ccso_frame_flag` | 47 | 5.9 | 1.0 bits |
| `ccso_planes` + `reuse_ccso`/`sb_reuse_ccso` | 289 | 36.1 | 6.1 bits |
| `bo_only`/`scale_idx`/`quant_idx`/`ext_filter`/`edge_clf`/`max_band_log2` | 450 | 56.2 | 9.6 bits |
| `tu(7)` offset table | 7,642 | 955.2 | 162.6 bits |
| **row total** | **8,441** | **1,055.1** | **179.3 bits (22.4 B)** |

So the CCSO row is **1,055 B of the 24,885 B pin (4.2%)**, 17.6 B per frame
across the sequence: 47 of 60 frames enable luma CCSO (14 of them also enable
U, 14 V), the rest code the single zero `ccso_frame_flag`. An enabled frame
costs 78-376 bits; the busiest is 47 B.

That is *more* than the 431 B / 7.3 B-per-frame figure the earlier header-total
comparison suggested. The two measurements differ because the older A/B dropped
the `--video-ccso-*` flags, which also drops them from the `quality` preset —
and the preset's fallback fit (4 bands, distortion only) writes large tables
(`ccso_bits=52..79` per P frame, traced). Comparing two CCSO-carrying streams
that way charges the difference of two rows, not the row. The row size above is
accounted from the writer itself, inside the real frame header, so it is exact.

### Where the table bits go

* **62.8% of the table bits are the all-zero floor.** `tu(7)` codes index 0 as a
  single `0`, so a 64-band table costs ≥64 bits (8 B) *per enabled plane per
  frame* even when every offset is off. The pin's tables average 15.3 non-zero
  bands of 64; 8 of the 8-12 bytes the pin spends per plane buy nothing.
* Non-zero offsets average 15.3 bands, each 2-4 bits (`tu(7)` index 1..3).
* Frame to frame the table is *not* stable: over the 74 plane-frames with a
  previous table, 16 of 64 band indices changed on average and only 29 of 74
  kept the same `scale_idx`. The zeros are stable; everything the table pays
  for is re-chosen every frame.

## 2. Why the decision did not see it

The rate-aware fit (`prepare_ccso`, `--video-ccso-rd`) minimised

```text
J_band = coded SSE(band, idx) + lambda * tu7_extra_bits(idx)
```

The `lambda * extra` term is zero for index 0, so *every* band the fit leaves
off is a free bit, and the fixed part of the row (frame flag, plane flags,
`bo_only`/`scale`/`max_band`) was not priced at all. With `ccso_bands = 6`
pinned from the CLI, the floor was a constant the fit could not trade away:
8 B per plane on every frame, including the keyframe, whose 64-band table only
ever had 4 non-zero entries.

`--video-ccso-header-rd` (implies `--video-ccso-rd`) prices the whole row:

```text
J = sum_band [ SSE(band, idx) + lambda * tu7_extra_bits(idx) ]
    + lambda * (11 + 2^ccso_max_band_log2)
```

The 11 fixed bits are what turning one plane on costs in an inter frame:
`ccso_planes[0..2]` (3) + `reuse_ccso`/`sb_reuse_ccso` (2) + `bo_only` (1) +
`scale_idx` (2) + `max_band_log2` (3). `ccso_max_band_log2` becomes a *decision*
instead of a CLI constant: the fit evaluates every band count from 1 to
`--video-ccso-bands` on one residual histogram (the band statistics are
additive, so coarser bands aggregate out of the 64-bin histogram — one pass
over the plane, no extra pixel work), and drops the plane when the best table
does not repay its bits. The per-unit `ccso_blk` prune, the `scale_idx` sweep
and the two refit passes are unchanged.

## 3. The other two levers

**(b) reuse the previous frame's table (`reuse_ccso`).** §5.18.7.12 codes
`reuse_ccso` + `sb_reuse_ccso` + `f(CeilLog2(NumTotalRefs))` instead of the
table, which with a single reference is 2-3 bits instead of 9 + table bits.
Evaluated on the same histograms, the stale table's SSE penalty exceeded the
saved bits on *every* frame where it was probed (`ccso_offset_idx` tables are
re-chosen every frame, §1): the modelled cost of the previous frame's table was
223,751 vs 228,295 (SSE + lambda·bits) on one P frame, and 250,033 vs 251,038
on another where it loses even to coding no CCSO at all (249,769). The lever is real in the syntax but loses on
this content, so it is not implemented. A band-count-aware reuse could only win
where the table is already ≤2 B, where there is nothing to save.

**(c) fewer bands/units.** This is the shipped change: with the row priced, the
fit picks 2-64 bands per plane instead of 64:

| | pin | `--video-ccso-header-rd` |
|---|---:|---:|
| band counts chosen | 64 (all 47 plane-frames) | 2, 4, 8, 16, 32, 64 (25 still at 64) |
| enabled plane-frames (Y/U/V) | 47 / 14 / 14 | 36 / 5 / 3 |
| all-zero floor share of table bits | 62.8% | 44.9% |
| row bits | 8,441 | 4,798 |
| row bytes per frame | 17.59 | 10.00 |

## 4. Measured stream effect

Same QP, flag off → on: **24,885 B → 24,291 B (-594 B, -2.39%)**, i.e.
**9.9 B per frame** of the 60-frame stream. The traced row accounts for 455 B
(7.6 B/frame) and the tiles for the other 139 B: 20 fewer plane-frames code
`ccso_blk` symbols (luma 47→36, U 14→5, V 14→3), and the pruned tables change the
coded residual slightly. Encoding time is unchanged (user CPU 21.5-22.0 s per
60-frame encode either way on the shared host): the band search runs on the
64-bin residual histogram and adds no pixel pass, so the per-unit `ccso_blk`
decisions still dominate the stage.

Quality changes because the filter does, so the curves were measured with
`tools/quality.py` (CVVDP 0.5.7, `standard_fhd`, MPS) at QP 106-112:

| QP | pin B | pin kbps | pin CVVDP | flag-on B | on kbps | on CVVDP | on PSNR-Y |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 108 | 25,324 | 101.296 | 9.7305 | 24,794 | 99.176 | 9.7307 | 43.9501 |
| 109 | 25,098 | 100.392 | 9.7188 | 24,566 | 98.264 | 9.7343 | 43.9159 |
| **110** | **24,885** | **99.540** | **9.7177** | **24,291** | **97.164** | **9.7096** | **43.7948** |
| 111 | 24,633 | 98.532 | 9.7154 | 24,281 | 97.124 | 9.7034 | 43.8594 |
| 112 | 24,281 | 97.124 | 9.6981 | 23,538 | 94.152 | 9.6923 | 43.8644 |

Iso-rate (log-linear interpolation on the overlapping part of the curves):

| Rate | pin CVVDP | flag-on CVVDP | Delta |
|---:|---:|---:|---:|
| 97.5 kbps | 9.7027 | 9.7171 | **+0.0144** |
| 98.0 kbps | 9.7089 | 9.7284 | **+0.0195** |
| 98.5 kbps | 9.7150 | 9.7334 | **+0.0184** |
| 99.0 kbps | 9.7165 | 9.7314 | **+0.0149** |

Read that as **+0.015 to +0.020 JOD at iso-rate**, i.e. the saving is real but
small: it is the same order as the ±0.02 JOD QP-to-QP ripple documented in
`docs/video-nightly.md`, so the honest claim is "no worse, and the direction is
consistently positive on this clip". At fixed QP the flag is -0.14 dB PSNR-Y
for -2.4% rate, which is what the table pruning buys. The QP 106/107 points of
the flag-on curve are not usable for the comparison (the drift refresh moves
between QPs and they land off the curve), so the table starts at QP 108.

## 5. Where it is not worth it

* **The per-frame row saving does not survive at fixed QP as quality.** A
  2.4% rate cut on this clip is worth only +0.015-0.02 JOD because the loss is
  spread over every frame (see the oracle table in `docs/video-nightly.md`);
  the bytes are better spent on the keyframe. The flag is therefore an
  **opt-in**, not part of `--fast-decode`: the nightly pin, the `quality` preset
  and the default stream must stay byte-identical.
* **Reuse (b) does not pay** at this band count and content (§3).
* **The 64-band table is not always wrong.** 25 of 60 frames still choose 64
  bands: when the P frame has real band structure the fine table repays its
  8 B floor. A fixed `--video-ccso-bands 4` cuts the same 849 B at QP 110 but
  charges every frame the coarse table, and it cannot turn CCSO off on the
  frames where the whole tool is marginal.

## 6. Verification

* Flag off: pin byte-identical (24,885 B / `5e316100ce7fd709b4298ccfebfffccd`),
  and the other pins are untouched — `tools/gate_video_nightly.sh` passes.
* Flag on: `dav2d --strict 1` decodes **60/60** pictures and the visible
  encoder recon (10,368,000 B) equals the decoded payload byte for byte.
* `AV2ENC_TRACE_CCSO` shows the row shrinking 8,441 → 4,798 bits (43.2%) and the
  band count leaving `{64}`.
* `tools/gate_video_ccso_overhead.sh` pins all of the above; it is registered in
  `tools/run_gates.sh`.
* `filters_stream_test` pins the `CcsoHeaderBits` split (1 + 5 + 6 + 64 bits for
  a 64-band zero table with one enabled plane; 7 + 1 bits for a `tu(7)` escape).
* Rejections: `--video-ccso-header-rd` with `--video-gpu` (CPU-measured model,
  the CUDA stream is not calibrated against it), with `--still` (video-only) and
  with a 10-bit/4:4:4 core (the `--video-ccso-rd` contract).

## 7. Files

| File | Change |
|---|---|
| `src/codec/filters/stream/filters_stream.h/.cpp` | `CcsoHeaderBits` accounting out-parameter on `write_ccso_params()`; no bits change |
| `src/pipeline/video/post/video_loop_filters.h/.cpp` | `ccso_header_rd` option, `fit_ccso_plane_header_aware()`, `AV2ENC_TRACE_CCSO` |
| `src/main.cpp` | `--video-ccso-header-rd` (parse, help, plumbing, rejections) |
| `src/codec/filters/stream/filters_stream_test.cpp` | row accounting test |
| `tools/gate_video_ccso_overhead.sh`, `tools/run_gates.sh` | the gate |

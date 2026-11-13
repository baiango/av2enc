# Video 1-D intra transform types / `dc_sign` — investigation

Status: **2-D IDTX wired and decoder-gated; the one-axis 1-D types stay
rejected by name.** Normative source: AV2 v1.0.0 (`spec/AV2_Spec_v1.0.0.pdf`,
commit `38f28e7`) §5.4.8, §5.20.5.5, §5.20.7.27, §5.20.7.30, §7.15.4.1.
`dav2d` and `avmdec` are black-box validators.

## What the syntax needs

A 1-D (identity-axis) TX type — `V_DCT`, `H_DCT`, `V_ADST`, `H_ADST`,
`V_FLIPADST`, `H_FLIPADST` (TX_TYPE 10..15) — differs from the shipped
`TX_CLASS_2D` path in four places:

1. `enable_idtx_intra = 1` in §5.4.8 (the sequence header conditional is
   `enable_fsc ? (inferred 1) : enable_idtx_intra`).
2. `fsc_mode` in §5.20.5.5 for every luma coding block with
   Block_Width/Height <= 32, coded after the Y mode/angle delta.
3. §5.20.7.30 `get_scan()` uses the HORIZ (row-major) or VERT (column-major)
   scan, and `get_lf_limits()` changes the LF region (`col < 2` for HORIZ,
   `row < 2` for VERT, luma).
4. §5.20.7.27 selects `dc_sign_horz_vert` for the non-DC coefficient on the
   first row/column.  Its context is the constant 0 and its CDF is
   `TileDcSignCdf[ptype][isHidden][0]` — the same slot the existing
   `dc_sign` array exposes as `dc_sign[0][0][0]`.

## What is implemented

* §5.4.8 `enable_idtx_intra` is signalled by `--video-idtx`.
* The §5.20.5.5 `fsc_mode = 0` symbol is written for every <= 32x32 luma
  coding block (intra keyframe and intra leaves in P frames) when the bit is
  set.  The CDF is `Default_Fsc_Mode_Cdf` with context `[0][Fsc_Bsize_Groups]`.
* The TX search exposes IDTX (type 9, `TX_CLASS_2D`) only when
  `enable_idtx_intra = 1`.  Without the new filter `--video-tx-search` could
  select IDTX on WIDE_32/HIGH_32 transforms while the sequence header said
  `enable_idtx_intra = 0`; that stream decoded with the pinned dav2d but was
  non-conformant.  The filter is `txtsearch::Options::allow_idtx`.
* The syntax writer rejects TX_TYPE 10..15 loudly instead of coding a 2-D
  scan for a 1-D transform.

`--video-idtx` therefore gives a fully decoded 2-D identity path:
`video_idtx_gate.sh` selects 4 IDTX leaves in 4 frames (8 in 8), writes
`enable_idtx_intra = 1`, and every decoded sample equals `--video-recon`
under `dav2d --strict 1`.

## What stays rejected: `--video-idtx-1d`

The pinned dav2d `f9dafbc` **rejects negative H_DCT/V_DCT coefficients under
`--strict 1`**.  The component gate already pins this:

```
src/codec/intra/supplemental/build_and_test.sh      # builds gate_dc_sign_horz_vert
# hv_pos.ivf   sign 0, normative ctx 0   -> strict 1 decodes
# hv_neg.ivf   sign 1, normative ctx 0   -> strict 0 decodes, strict 1 fails
```

Exact repro (from a checkout root):

```
cd src/codec/intra/supplemental && ./build_and_test.sh   # if the gate links
# or build the gate directly against the product core:
c++ -std=c++20 -O2 -I ../../../src gate_dc_sign_horz_vert.cpp \
    dc_sign_horz_vert.cpp dc_intra_subsampled.cpp intra_filter_corner.cpp \
    save_subpu.cpp ../../../build/libav2enc_core.a -o /tmp/gate_hv
mkdir -p /tmp/hv && /tmp/gate_hv --out /tmp/hv
tools/bin/dav2d-macos -q -i /tmp/hv/hv_neg.ivf -o /tmp/n.y4m --strict 1
# -> Error decoding frame: Invalid argument / No data decoded
tools/bin/dav2d-macos -q -i /tmp/hv/hv_neg.ivf -o /tmp/n.y4m --strict 0
# -> decodes
```

`avmdec` also rejects the same stream (`Failed to decode tile data`), so AVM
is not an alternative gate for negative 1-D coefficients.  A product stream
cannot in general keep every 1-D coefficient non-negative, so the one-axis
types remain a named rejection:

```
av2enc clip.y4m -o out.ivf --video --video-idtx-1d
# av2enc: error: --video-idtx-1d is rejected: ... pinned dav2d f9dafbc rejects
# every negative H_DCT/V_DCT coefficient under --strict 1 ...
```

Revisit when a decoder is pinned that implements the 1-D coefficient signs;
then add the class scans, `get_lf_limits()` and `dc_sign_horz_vert` to
`video_syntax.cpp` and the 1-D candidates to `--video-tx-search`.

## CUDA parity note

The CPU search now filters IDTX unless `enable_idtx_intra` is set.  The CUDA
TX search (`src/cuda/pipeline/video/tx/video_tx.h`) has no `allow_idtx` field
yet; the device must adopt the same filter before `--video-gpu` can claim
bit-exact parity with the CPU TX-search streams.  `--video-idtx` is CPU-only
and `--video-gpu --video-idtx` is rejected by name.

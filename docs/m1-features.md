# M1 feature integration status (2026-11-11)

Integrator workstream: shared M1 core (`src/encoder/m1/m1_headers.*`,
`m1_syntax.*`, `m1_still.*`, `src/cuda/m1/m1_gpu.*`, `src/main.cpp`,
`CMakeLists.txt`).  Every tool below is **opt-in**; the default `--m1` and
`--m1-gpu` bitstreams are unchanged.

## Status

| # | Feature | Flag (default) | CPU | GPU | Gate |
|---:|---|---|---|---|---|
| 1 | CDEF (§7.18) | `--m1-cdef` (off) | exact | not ported | PASS |
| 2a | CCSO band-offset (§7.19) | `--m1-ccso` (off) | exact | not ported | PASS |
| 2b | Wiener NS (§7.20) | — | not integrated | — | — |
| 3 | RDOQ (§7.16/§8.3) | `--m1-rdoq` (off) | exact | not ported | PASS |
| 4 | Intra TX type | `--m1-tx-type` (off) | exact | not ported | PASS |
| 5 | Rate-aware intra | `--m1-rdo-intra` (off) | exact | not ported | PASS |
| 6 | Full UV intra | `--m1-uv-full` (off) | exact | compiled, unverified | PASS (CPU) |
| 7 | Extended partitions | — | not integrated | — | — |
| 8 | 10-bit | — | not started | — | — |

"exact" means: the encoder reconstruction matches the pinned `dav2d`
(f9dafbc) decode sample-for-sample on every gated case, and the IVF decodes
with `--strict 1`.  CPU builds are the reference; the GPU port is gated
separately on the 3060 box and has not been re-run here (no CUDA toolchain on
this host).

## Pinned gate

`tools/m1_verify/feature_gate.sh` (IVF MD5 + `dav2d --strict 1` + optional
`m1_dump`/`verify.py` recon compare).  All 22 cases pass:

| Case | IVF bytes | IVF MD5 |
|---|---:|---|
| 64x64 `--m1` (baseline) | 936 | `0ef9825062a25c4d950ee924113c6f89` |
| 64x64 CDEF | 938 | `b795a6f998c7037c95b926d6afdfc89e` |
| 64x64 RDOQ | 794 | `a7ee3a0ac75781ce2a2c7390c3503019` |
| 64x64 TX type | 944 | `15681508a144eeea88bb8223185d75e9` |
| 64x64 rate-aware intra | 965 | `45b4a209a8c3e889c91dfa201f26d2d4` |
| 64x64 full UV | 939 | `2ed7f1ca8d146be8345ed95fee65f9bf` |
| 64x64 CDEF+RDOQ+TX type | 817 | `aa885652b8c57ecc25b14b87c97862d8` |
| 64x64 CCSO | 970 | `980e5c2194091faf2347e3b05dfdea69` |
| 64x64 CDEF+CCSO | 972 | `a90c525e480e8ff52db3f47f77fff887` |
| 640x360 `--m1` (baseline) | 13,355 | `e61de628b0812b716b0efa7dcafcdd06` |
| 640x360 CDEF (1 set) | 13,358 | `dec10527a779e7664ea78e53def9dbf1` |
| 640x360 CDEF (4 sets) | 13,375 | `b37bed7eb785d0c6b2e07219e888ff89` |
| 640x360 RDOQ | 11,487 | `ac668d497012333b6aeecb82c2e70337` |
| 640x360 TX type | 13,388 | `21c39aec98507dba7c24e5d320e1bb3c` |
| 640x360 rate-aware intra | 13,152 | `211da270c5baf34f7bd40da019ffd84b` |
| 640x360 full UV | 13,368 | `400dd3ad02db9e0e3e366091a7e6962f` |
| 640x360 CDEF+RDOQ+TX type | 11,611 | `84ab3934c08e6c90b876d067bcbb3606` |
| 640x360 CCSO | 13,397 | `54e853e24b21bf1237b8b51a846b1c35` |
| 640x360 CDEF+CCSO | 13,415 | `a272550a72c387a8a4f8cc8d3186f8da` |
| 640x360 CCSO+RDOQ | 11,535 | `7fd7a2a0e078665614f8b6d2c5fe1b57` |

Quality (640x360, `dav2d` decode, PSNR from ffmpeg):

| Case | Bytes | PSNR-Y | PSNR-avg |
|---|---:|---:|---:|
| baseline | 13,355 | 45.089 | 46.368 |
| CDEF (4 sets) | 13,375 | 45.236 | 46.510 |
| TX type | 13,388 | 45.156 | 46.432 |
| CCSO | 13,397 | 45.126 | 46.403 |
| rate-aware intra (full intra) | 13,152 | 44.958 | 46.246 |
| RDOQ | 11,487 | 44.612 | 45.846 |
| CDEF+RDOQ+TX type+CCSO | 11,627 | 44.775 | 45.981 |

## Per-feature notes

### CDEF

* Sequence `enable_cdef=1`; frame `cdef_params()` after the four
  `df_delta_q_present` bits; `read_cdef()` at the start of every coding block.
* Strength search (`m1_cdef_plan`) runs after deblock using the original
  picture as the SSE reference.  `--m1-cdef-strengths N` selects 1..8; the
  search prunes unused sets, and `CdefStrengths == 1` codes no index symbols.
* Filtered planes replace the reconstruction and are the reported output.
* Verified at 64x64, 256x256, 640x360 and 1920x1080, with full-intra, no-deblock,
  no-CfL and adaptive-CDF variants.

### CCSO (band-offset-only)

* Sequence `enable_ccso=1` + `ccso_unit_matches_sb_size=1`; frame
  `ccso_params()` writes all three planes; `read_ccso()` codes one binary
  symbol per enabled plane at each 64x64 unit origin using the §8.3.1 left-unit
  context.
* Encoder decision: for each plane and band, choose the `Ccso_Offset` index
  that minimises SSE against the source over visible samples.  The band uses
  the **pre-CDEF** luma (`CurrFrame`), and the filter runs post-CDEF, in
  normative order.  `--m1-ccso-bands N` (0..6, default 6 = 64 bands).
* The edge-classified CCSO form and per-unit `ccso_blk=0` decisions are **not**
  implemented yet; only the band-offset form is signalled.
* Wiener NS is **not integrated**: its tile syntax
  (`read_wienerns_filter()` incl. `search_frame_filters()` group coding) has no
  reference implementation in `src/cuda/m1/restoration/`, and the component's
  README states no bitstream path exists.  Implementing it needs its own
  coefficient-fit and group-symbol writer before it can be gated.

### RDOQ

* Replaces scalar quantisation inside `compute_tx()`; the syntax writer and
  inverse transform consume the trellis levels, so no syntax changes.
* Rate model: `rdoq::av2::Model` (static qctx-0 CDFs).  Lambda:
  `rdo::lambda_sse_from_quant_step(1/quant_factor, sse_scale)`;
  `--m1-rdoq-scale` (default 0.125).  Calibration table is in
  `src/encoder/rdoq/README.md`; 0.125 was reused so the measured deltas match
  the component report (−14.1% tile bytes at 640x360 for −0.53 dB).
* If TX-type selection + RDOQ leave `eob <= 1`, the block falls back to
  `DCT_DCT` and is re-quantised, because the decoder infers DCT there.

### Intra TX type

* Luma only; chroma stays `DCT_DCT` (`enable_chroma_dctonly=1`).
* Sequence header keeps `enable_idtx_intra=0`, so identity 1D transforms
  (`IDTX`, `V/H_DCT`, `V/H_ADST`, `V/H_FLIPADST`) are filtered out of the
  candidate set even though the long-side sets list them.
* Observed while integrating: signalling `enable_idtx_intra=1` made `dav2d`
  reject otherwise-valid streams (64x64, no identity type selected).  That
  points at a sequence-header bit-position/open item outside this component;
  until it is understood, M1 keeps the flag 0 and excludes identity types.
* `--m1-tx-type-scale` is the SSE-domain lambda scale (default 0.125).

### Rate-aware intra

* `rdo::select_intra_mode()` over the existing 61-candidate luma search; each
  candidate's real quantised levels feed
  `intra_coefficient_rate_bits_from_levels()`.
* `--m1-rdo-intra` implies `--m1-full-intra`; `--m1-rdo-scale` is the
  `sse_scale` placeholder (`quant_step = txv::kQuantStep = 528`).
* Measured 640x360: 13,355 → 13,152 B at 0.125 (matches
  `src/encoder/rdo/experiment/README.md`), 13,149 B at 0.02.

### Full UV

* `--m1-uv-full` enables the directional/smooth/Paeth chroma search.  CPU is
  bit-exact on all gated sizes; GPU `chroma_decide` supports it
  (`M1ChromaOptions::enable_uv_full`) and `GpuOptions::uv_full` now plumbs it,
  but the GPU path was not rebuilt here.

## Not integrated

* **GPUs for features 1, 3, 4, 5.**  `encode_m1_gpu` rejects those option
  combinations with a clear error instead of silently ignoring them.  The CPU
  path is the reference; porting each stage needs the 3060 box.
* **Wiener NS** — see CCSO above.
* **Extended partitions** (`src/encoder/partition_ext/`) — the solver,
  `enable_ext_partitions` header flag, 10-value `read_partition`, and chroma
  ownership rules are a separate integration.  The component gate is clean,
  but no M1 syntax path exists yet.
* **10-bit** — headers, entropy contexts, transforms and reconstruction all
  change; deliberately last.

## Verification

```sh
# CPU reference build
cmake -S . -B build-m1 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DAV2ENC_ENABLE_CUDA=OFF -DAV2ENC_ALLOW_CPU_ONLY=ON
cmake --build build-m1 -j

# feature gate (pinned MD5 + dav2d --strict 1)
tools/m1_verify/feature_gate.sh --bin build-m1/av2enc

# with encoder-recon exactness
g++ -O2 -std=c++20 -Isrc tools/m1_verify/m1_dump.cpp \
  -o build-m1/m1_dump build-m1/libav2enc_core.a
tools/m1_verify/feature_gate.sh --bin build-m1/av2enc --dump build-m1/m1_dump

# ctest (includes the M0/M1 header + decoder smoke tests)
ctest --test-dir build-m1 --output-on-failure
```

The baseline `--m1` MD5s are unchanged from `AGENTS.md`:
64x64 `0ef9825062a25c4d950ee924113c6f89`,
640x360 `e61de628b0812b716b0efa7dcafcdd06`.

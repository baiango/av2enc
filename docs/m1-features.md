# M1 feature integration status (2026-11-11)

Integrator workstream: shared M1 core (`src/encoder/m1/still_headers.*`,
`still_syntax.*`, `still_encoder.*`, `src/cuda/m1/still_gpu.*`, `src/main.cpp`,
`CMakeLists.txt`).  Every tool below is **opt-in**; the default `--still` and
`--still-gpu` bitstreams are unchanged.

## Status

| # | Feature | Flag (default) | CPU | GPU | Gate |
|---:|---|---|---|---|---|
| 1 | CDEF (§7.18) | `--still-cdef` (off) | exact | not ported | PASS |
| 2a | CCSO band-offset (§7.19) | `--still-ccso` (off) | exact | not ported | PASS |
| 2a2 | CCSO edge-classified + per-unit skip | `--still-ccso-edge` (off) | exact | not ported | PASS |
| 2b | Wiener NS, luma (§7.20.3) | `--still-wiener` (off) | exact | not ported | PASS |
| 2c | Wiener NS, chroma (§7.20.3) | `--still-chroma-lr` (off) | exact | not ported | PASS |
| 2d | PC-Wiener, luma (§7.20.4) | `--still-pc-wiener` (off) | exact | not ported | PASS |
| 2e | GDF (§7.20.5) | `--still-gdf` (off) | exact | not ported | PASS |
| 2f | Film grain signalling (§5.14/§5.18.10) | `--still-grain`, `--still-grain-adaptive` (off) | recon exact, grain output | N/A (decoder) | PASS |
| 3 | RDOQ (§7.16/§8.3) | `--still-rdoq` (off) | exact | not ported | PASS |
| 4 | Intra TX type | `--still-tx-type` (off) | exact | not ported | PASS |
| 5 | Rate-aware intra | `--still-rdo-intra` (off) | exact | not ported | PASS |
| 6 | Full UV intra | `--still-uv-full` (off) | exact | rejected by name | PASS (CPU) |
| 7 | Extended partitions | — | not integrated | — | — |
| 8 | 10-bit | — | not started | — | — |
| 9 | Coeff-CDF qctx 1–3 (§8.3.1) | `--qp 1..255` (default 90) | exact | pinned 90 | PASS |
| 10 | Quantizer matrices (§5.18.6.2/§7.14.4) | `--still-qm` (off) | exact | not ported | PASS (module gate) |
| 11 | Intra secondary transform (§5.20.8.2/§7.15.3) | `--still-ist` (off) | exact | not ported | PASS (module gate) |
| 12 | DC/AC delta-Q (§5.4.8/§5.18.6.1) | `--still-dc-ac-delta-q` (off) | exact | not ported | PASS (module gate) |

"exact" means: the encoder reconstruction matches the pinned `dav2d`
(f9dafbc) decode sample-for-sample on every gated case, and the IVF decodes
with `--strict 1`.  CPU builds are the reference; the GPU port is gated
separately on the 3060 box and has not been re-run here (no CUDA toolchain on
this host).

## Pinned gate

`tools/still_verify/feature_gate.sh` (IVF MD5 + `dav2d --strict 1` + optional
`still_dump`/`verify.py` recon compare).  All 30 cases pass:

| Case | IVF bytes | IVF MD5 |
|---|---:|---|
| 64x64 `--still` (baseline) | 936 | `0ef9825062a25c4d950ee924113c6f89` |
| 64x64 CDEF | 938 | `b795a6f998c7037c95b926d6afdfc89e` |
| 64x64 RDOQ | 794 | `a7ee3a0ac75781ce2a2c7390c3503019` |
| 64x64 TX type | 944 | `15681508a144eeea88bb8223185d75e9` |
| 64x64 rate-aware intra | 965 | `45b4a209a8c3e889c91dfa201f26d2d4` |
| 64x64 full UV | 939 | `2ed7f1ca8d146be8345ed95fee65f9bf` |
| 64x64 CDEF+RDOQ+TX type | 817 | `aa885652b8c57ecc25b14b87c97862d8` |
| 64x64 CCSO | 970 | `980e5c2194091faf2347e3b05dfdea69` |
| 64x64 CDEF+CCSO | 972 | `a90c525e480e8ff52db3f47f77fff887` |
| 640x360 `--still` (baseline) | 13,355 | `e61de628b0812b716b0efa7dcafcdd06` |
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
| 64x64 QP 91 (qctx 1) | 982 | `f50e18c4e08eab3ca6437ed6283a9216` |
| 64x64 QP 141 (qctx 2) | 259 | `c47ea7a5e8f826addb8ef90c527c917a` |
| 64x64 QP 191 (qctx 3) | 105 | `3768deaaad9a67f7bece568d743993e5` |
| 64x64 QP 255 (qctx 3) | 73 | `d1e1ebebf2885e575c55508f8b7157d1` |
| 640x360 QP 91 (qctx 1) | 13,775 | `f5762980cda6c08ec82e131bac487a77` |
| 640x360 QP 141 (qctx 2) | 3,428 | `6db7ffba200e2b90075fcb826aaef3c1` |
| 640x360 QP 191 (qctx 3) | 826 | `df6894744772f7107cad88381ae82e2e` |
| 640x360 QP 255 (qctx 3) | 201 | `d0c61808ae8dd6e011046bbab34256af` |

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

### DC/AC delta-Q

* Sequence header writes the §5.4.8 delta group (`equal_ac_dc_q=0`, all three
  enables, base deltas 0) and the frame header writes §5.18.6.1
  `quantization_params()` from `dcdeltaq::search_frame_delta_q()`.  The
  default-off state writes exactly the historical 8 sequence bits and the
  `base_q_idx` byte, so pinned streams do not move.
* Forward quantisation and §7.14.4 dequant use `get_q(qindex, delta)` per
  coefficient: the DC quantizer at adjusted raster index 0, AC elsewhere.  The
  coefficient-CDF slice still comes from the raw `base_q_idx` (§8.3.1).
* **Deblocking interaction:** §7.17.6 chroma filter strength adds the effective
  UV AC delta (`DeltaQUAc/DeltaQVAc + BaseUVAcDeltaQ`).  The encoder must pass
  the same values to `DeblockOptions`; omitting them desyncs only the reported
  chroma reconstruction (the tile bytes are unchanged).  Fixed here; recorded
  in `docs/negative-results.md`.
* Measured at 640x360 QP 90: 13,200 B vs 13,355 B baseline (−1.2%) at
  PSNR-Y 45.084 vs 45.089 and avg 46.318 vs 46.368.  That is a measured RD
  trade-off from the component's proxy search, not a claim of a rate win.
* Gate: `src/codec/transform/delta_q/still_delta_q_gate.sh` (pinned MD5s,
  `dav2d --strict 1`, recon exactness at QP 90/141, IST interaction, and the
  `--still-qm` / `--still-rdoq` rejections).

### Intra secondary transform (IST)

* Sequence header `enable_intra_ist=1`, `enable_inter_ist=0`; `sec_tx_type`
  is coded for every IST-applicable luma block (including symbol 0), followed
  by `most_probable_stx_set` only when nonzero.  Chroma never enters the path.
* Applicability is the §5.20.8.2 predicate: luma, `TxType` DCT_DCT/ADST_ADST,
  `YMode != PAETH`, `eob != 1`, `eob <= eobLim` (8 / 20 / 32).  The encoder
  decides per block with the component's candidate enumeration and an
  analysis projection of the quantised coefficients (the spec defines only the
  decoder inverse).  The projection is applied to the sub-block the §9.7 scan
  map addresses; `forward_secondary_coeffs_kernel()` alone zeroes the whole
  block and is only valid for 4x4/8x8, so the still path does its own partial
  projection.
* Reconstruction runs dequant → §7.15.3 inverse → primary 2-D inverse.  The
  encoder recon is decoder-exact at 64x64/640x360, including `--still-qm`,
  `--still-tx-type` and `--still-rdoq` combinations.
* Rate: at 640x360 QP 90 the gate stream is 13,371 B vs 13,355 B baseline at
  the same PSNR — the analysis projection + static symbol costs do not find a
  gain on this fixture.  The gate is about *valid signalling and exact
  reconstruction*, not a BD-rate claim.  `--still-ist-scale` is the SSE-domain
  lambda (default 0.125).
* Gate: `src/codec/transform/secondary/still_ist_gate.sh` (pinned MD5s,
  `dav2d --strict 1`, recon compare, and the documented interactions).

### Quantizer matrices (QM)

* Sequence header is unchanged: the encoder signals only spec-defined default
  matrices (`using_qmatrix = 1`, `qm_y/u/v[0] = qm_level`, one segment), so no
  `quantizer_matrix_obu()` is emitted and `QmDataPresent[]` stays 0.  The
  pinned `dav2d` accepts default matrices but rejects QM OBUs.
* `--still-qm-level` (default 3, range 0..14) selects the §9.4.3 matrix; 15 is
  the "no QM" sentinel and is not selectable.  Forward quantisation scales the
  scalar `quant_factor` by `q / q2` with `q2 = Round2(q * m, 5)`; the §7.14.4
  dequant in reconstruction uses the same `q2`, so encoder recon stays
  decoder-exact (0 differing samples at 64x64/640x360, QP 90/91/141).
* `--still-qm --still-rdoq` is rejected by name: the RDOQ rate model assumes
  the scalar quantiser.  TX_TYPE search is compatible (the selector runs on
  its own candidate metric; the chosen type is re-quantised under QM).
* Gate: `src/codec/transform/qm/still_qm_gate.sh` (pinned IVF MD5s for
  `--still-qm` at 64x64/640x360 and QP 141, `dav2d --strict 1`, recon compare,
  and the RDOQ rejection).

### CDEF

* Sequence `enable_cdef=1`; frame `cdef_params()` after the four
  `df_delta_q_present` bits; `read_cdef()` at the start of every coding block.
* Strength search (`still_cdef_plan`) runs after deblock using the original
  picture as the SSE reference.  `--still-cdef-strengths N` selects 1..8; the
  search prunes unused sets, and `CdefStrengths == 1` codes no index symbols.
* Filtered planes replace the reconstruction and are the reported output.
* `--still-cdef-on-skip-txfm` (debug, requires `--still-cdef`) codes
  `cdef_on_skip_txfm_frame_enable = 1` in the ADAPTIVE row.  A single-picture
  header forces `CdefOnSkipTxfm = CDEF_ON_SKIP_TXFM_ADAPTIVE`, so the two
  sequence modes (`DISABLED`/`ALWAYS_ON`) are not codeable on the still path.
  They are selected with `VideoPostSequenceOptions::filters` on a
  non-single-picture M2 stream and are gated by
  `src/pipeline/video/post/build_and_test.sh`: three 2-frame tiled streams with
  CDEF enabled (`CdefStrengths == 1`, all-zero set) code each sequence mode and
  decode sample-exactly.  Intra blocks derive skip = 0, so the still recon is
  unchanged by the frame bit (only the header bit moves); pinned at 64x64
  `cf86fce480936dd5abb4ece5f4983ba4` and 640x360
  `8815c209c2b9e1838a7c56ebb4c3f8bd`.
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
  normative order.  `--still-ccso-bands N` (0..6, default 6 = 64 bands).
* `--still-ccso-edge` (implies `--still-ccso`) signals the edge-classified
  form: `ccso_bo_only = 0`, a searched `(scale_idx, quant_idx, ext_filter,
  edge_clf)` tuple, `ccso_max_band_log2 <= 3`, and a `(band, cls0, cls1)` offset
  table fitted by SSE.  It also writes a per-64x64-unit `ccso_blk` grid and
  clears units whose fitted offset does not reduce SSE (per-unit `ccso_blk=0`).
  The default band-offset path is unchanged, including its all-ones grid.
  §5.18.7.12 caps `max_band_log2` at 3 in the edge-classified form, so
  `--still-ccso-bands` is clamped to 3 there.
  Pinned edge streams: 64x64 `b83169960ce4e75c6ee5c1d933901923` (982 B);
  640x360 `f0f3fb9693d9cb5bf2f86cef305004bb` (13,414 B);
  640x360 CDEF+edge CCSO `a06c1765e37827cbb840ed018b154037` (13,433 B).
* Wiener NS and chroma LR are CPU-integrated (see the section below).  The CUDA
  restoration kernel is still not called: `--still-gpu` rejects the flags by
  name, and frame-level/multi-class `search_frame_filters()` coding remains
  unused (M1 signals per-unit single-class filters only).

### Wiener NS / chroma LR (§7.20.3)

* `--still-wiener` enables luma loop restoration; `--still-chroma-lr` enables
  the shared U/V planes.  Both are per-unit `RESTORE_WIENER_NONSEP` with
  `frame_filters_on = 0`, `LoopRestorationSize` 64 luma / 32 chroma (4:2:0).
  The search (`restoration::classes::wiener_decide`) fits coefficients against
  the source, distortion-only (`lambda_bits = 0`), and runs after CDEF/CCSO in
  normative order.
* The sequence header sets `enable_restoration = 1` and disables PC-Wiener;
  `lr_params()` is written between `cdef_params()` and `ccso_params()`.  Every
  superblock origin writes `read_lr()` for each enabled plane before
  `read_partition()`.  Disabled planes code no tool symbol when the only
  available tool is `RESTORE_NONE`.
* Reconstruction/output is `LrFrame`, i.e. the decoder's post-LR picture.
* New pinned streams (`tools/still_verify/loop_restoration_gate.sh`):

| Case | IVF bytes | IVF MD5 |
|---|---:|---|
| 64x64 luma Wiener | 941 | `f2c005d7873c9c493e1b41473c8f3a54` |
| 64x64 chroma LR | 952 | `0e291675117f627783606d4bf0b3ca43` |
| 640x360 luma Wiener | 13,552 | `4bbede8ad74b6552a2404e5c23431adb` |
| 640x360 chroma LR | 13,739 | `e6a83f2fec7bc4537b868cbbdc389504` |
| 640x360 luma+chroma | 13,936 | `a7a4c06967bd4704dbb72f4de8e79f91` |
| 640x360 CDEF+CCSO+LR | 14,001 | `2f4ffb0c0dc62d713f5a6a9f5e236240` |

The separate gate script leaves `feature_gate.sh`'s published 30-case list
untouched, so `tools/refactor/verify_baseline.sh --check` still passes.

### PC-Wiener (§7.20.4)

* `--still-pc-wiener` signals `FrameRestorationType[0] = RESTORE_PC_WIENER` and
  writes one `use_pc_wiener` symbol per 64x64 luma unit.  The recovered
  classifier/filter bank in `src/codec/filters/tables/pc_wiener.*` supplies the
  pixel filter; no coefficients or `frame_filters_on` bit are coded.
* The decision is distortion-only (`lambda = 0`) against the source and uses
  the §7.20.1 `LrTxSkip` map derived from the block stream
  (`skip_flag || eob == 0`; M1 has no block skip, so `eob == 0` per 4x4 luma
  cell).  Omitting it desynchronises the classifier for skipped residuals and
  was observed as 45 luma samples differing at 640x360.
* Applied after CDEF/CCSO; chroma is untouched by PC-Wiener itself.  Mutually
  exclusive with `--still-wiener` (both select plane 0's
  FrameRestorationType).
* Composes with chroma LR: plane 0 is `RESTORE_PC_WIENER` and planes 1/2 are
  `RESTORE_WIENER_NONSEP`, so `lr_params()` codes both rows and the tile writes
  `use_pc_wiener` for luma units then `use_wiener_ns` for chroma units.
  Composes with GDF too: GDF runs last and reads the post-CCSO `CdefFrame`,
  not the PC-Wiener output.
* Pinned streams: 64x64 `508ab8ec82f6b5fc0a1b6a1980eb0c50` (938 B);
  640x360 `5236b0dd917d7cd62832b9d79b88f007` (13,365 B);
  640x360 CDEF+CCSO+PC-Wiener `cffddedb6fac60b29d38e2caf5f9b6f4` (13,426 B).
  Combinations: 64x64 PC-Wiener+chroma `afb66bbe431d39f8a5f6867ef6a2b865`;
  640x360 CDEF+CCSO+PC-Wiener+chroma
  `d23bc36b9d85ec34e70ddce53baccdfe`; 640x360 GDF+PC-Wiener+chroma
  `ba08c8ef70953cf54de375ddc8b9cdd4`; 640x360 full chain
  `c706c3709a5ffc936f62f161e2986347`; 1080p PC-Wiener+chroma
  `6270eb72216b965d90584501da119680`.
* Pinned streams: 64x64 `508ab8ec82f6b5fc0a1b6a1980eb0c50` (938 B);
  640x360 `5236b0dd917d7cd62832b9d79b88f007` (13,365 B);
  640x360 CDEF+CCSO+PC-Wiener `cffddedb6fac60b29d38e2caf5f9b6f4` (13,426 B).

### GDF (§7.20.5)

* `--still-gdf` writes `enable_gdf = 1` and
  `gdf_unit_matches_sb_size = 1` in the sequence header, and the
  `gdf_params()` row between `deblocking_filter_params()` and `cdef_params()`.
  M1 signals `gdf_per_block = 0`, so every 64x64 unit is filtered and **no**
  `use_gdf` tile symbols are coded; the recovered `spec_tables/gdf.*` filter is
  applied in the loop-restoration stage against the source.
* §7.20 order: `LrFrame := CdefFrame`, then Wiener/PC-Wiener reads
  `CdefFrame` and writes `LrFrame`, then GDF (§7.20.1) reads `CdefFrame`
  inside the stripe / `CurrFrame` outside and rewrites luma `LrFrame`.  The
  encoder keeps the post-CCSO `CdefFrame` in a separate buffer, so GDF now
  composes with luma Wiener, PC-Wiener and chroma LR instead of overwriting
  their output.
* Pinned streams: 64x64 `1714218f7dab3dbaaa334885a38c2a88` (937 B);
  640x360 `b07c1327de42ec038e25dc7c74561984` (13,356 B);
  640x360 CDEF+CCSO+GDF `ebe9f970bd7a3a73e4b38ae8e3393354` (13,416 B).
  Combination pins (same gate): 64x64 GDF+Wiener
  `37569cd29b22dedb926d31fad17d4aa2`; 640x360 GDF+Wiener
  `fd67ddab058eb86485eb9cdb986ffbb2`; 640x360 GDF+chroma LR
  `636137a428463bf0a6ecf59375c09a5a`; 640x360 GDF+Wiener+chroma LR
  `a55b3f920a57d3872dd7a336d664a65f`; 640x360 CDEF+CCSO+GDF+Wiener+chroma LR
  `1dfb02e1ccc42d31e8176caae1ad0f9f`; 640x360 GDF+PC-Wiener
  `6baecfd7d4ed2bd6f2328358f394a7f8`; 1080p GDF+Wiener+chroma LR
  `a4cfd4cb671dd7f38ccbcdf1f76e58c5`.

### Film grain (§5.14 / §5.18.10)

* `--still-grain` sets `film_grain_params_present = 1`, inserts a
  `film_grain_obu()` (slot 0, a fixed two-point 4:2:0 model) before the coded
  frame OBU, and appends `film_grain_config()` to the frame header.  For a
  single-picture header `apply_grain` is inferred to 1, so only `fgm_id` (0)
  and `grain_seed` (0) are coded.
* Grain is a decoder-side output process: the encoder reconstruction and
  reference stay grain-free.  The gate checks recon exactness with
  `dav2d --filmgrain 0` and separately that `--filmgrain 1` produces a
  different (synthesized) output.
* Pinned streams: 64x64 `2c487ae7f9abb1f51d33c335dc07feac` (987 B);
  640x360 `b23601e81d0600debc5b6d84f91be11a` (13,407 B).
* `--still-grain-adaptive` (requires `--still-grain`) replaces the fixed model
  with the `GrainStreamPlanner`: the estimator fits nine luma scaling buckets
  and the AR taps from the reconstruction (`min_activity = 0`, because a
  single-picture header infers `apply_grain = 1`), picks an `fgm_id` slot and
  writes `grain_seed = GrainSeedForFrame(0)`.  A genuinely flat frame yields no
  model and the encoder then emits no film-grain OBU (output stays grain-free).
  Pinned: 64x64 `9db402e9d668c1143451e34d47b9b2aa` (999 B); 640x360
  `78dfef3a53b7b315fa9150ad915f1ef6` (13,421 B); 1080p
  `80f5de0949b8800c7cff48f3447455ab` (61,640 B).  Every case is recon-exact
  with `--filmgrain 0` and produces different samples with `--filmgrain 1`.
* The grain module's old `pipeline/grain/stream/` include paths were stale after
  the folder refactor; this commit fixes them so the sources build in the root
  target.

### Completion-wave filter gaps (recorded, not wired)

* **Temporal LR / cross-frame filter-bank reuse.**  Intra-frame coefficient
  bank reuse is implemented and used: `wiener_decide` fills
  `merged`/`use_bank_k`, `wiener_write_unit` codes the §5.20.10.6 `merged`
  flag and bank index, and the still gate exercises it (the `unit_merged` /
  `unit_bank_k` arrays in `StillLrPlane`).  Cross-frame temporal LR needs a
  non-single-picture stream with `frame_filters_on = 1`,
  `temporal_pred_flag = 1` and `rst_ref_pic_idx`, plus `RefLrWienerNs` slot
  state.  `filters_stream.cpp` deliberately rejects temporal frame-filter
  prediction, the M2 product video writer never enables restoration and has no
  per-superblock `read_lr()` hook, and the still path has one picture and no
  reference slots.  No stream can exercise it, so it stays recorded: land the
  M2 LR tile hook first, then a 2-frame `temporal_pred_flag` gate.
* **CCSO non-4:2:0 / 128–256 superblock unit sizes.**
  `CcsoLumaSizeLog2` 7/8 needs `use_128x128_superblock` /
  `use_256x256_superblock` (or tile-aligned 128/256-sample units).  The still
  path rejects `--still-sb128`/`--still-sb256` with CDEF and CCSO, the M1
  header pins `ccso_unit_matches_sb_size = 1` with unit size 6, and the filter
  cores validate 4:2:0.  Unit sizing for the larger superblocks is owned by
  `integration/partition2`; it must land with the CDEF/CCSO grids and a
  `dav2d` gate.  Chroma already shares the luma unit grid, and the U/V shared
  signalling (`lr_uv_present`, per-plane CCSO) is wired.

### RDOQ

* Replaces scalar quantisation inside `compute_tx()`; the syntax writer and
  inverse transform consume the trellis levels, so no syntax changes.
* Rate model: `rdoq::av2::Model` (static qctx-0 CDFs).  Lambda:
  `rdo::lambda_sse_from_quant_step(1/quant_factor, sse_scale)`;
  `--still-rdoq-scale` (default 0.125).  Calibration table is in
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
* `--still-tx-type-scale` is the SSE-domain lambda scale (default 0.125).

### Rate-aware intra

* `rdo::select_intra_mode()` over the existing 61-candidate luma search; each
  candidate's real quantised levels feed
  `intra_coefficient_rate_bits_from_levels()`.
* `--still-rdo-intra` implies `--still-full-intra`; `--still-rdo-scale` is the
  `sse_scale` placeholder (`quant_step = txv::kQuantStep = 528`).
* Measured 640x360: 13,355 → 13,152 B at 0.125 (matches
  `src/encoder/rdo/experiment/README.md`), 13,149 B at 0.02.

### Full UV

* `--still-uv-full` enables the directional/smooth/Paeth chroma search.  CPU is
  bit-exact on all gated sizes.
* The GPU `chroma_decide` kernel supports `enable_uv_full`, but the
  orchestrator never fills the chroma job's luma-mode/UVSmooth inputs
  (`y_mode`, `y_angle_delta`, `filter_type_above/left`,
  `num4_above_right/below_left`), so the GPU run cannot reproduce the CPU
  decision.  `--still-gpu --still-uv-full` is therefore **rejected by name** instead
  of running a mismatched search or crashing.

### Coefficient-CDF qctx slices (`--qp 1..255`, CPU)

* §8.3.1 `init_coeff_cdfs()` selects one of four `COEFF_CDF_Q_CTXS` slices from
  the **raw frame** `base_q_idx`: `<=90` slice 0, `<=140` slice 1, `<=190`
  slice 2, otherwise slice 3.
* `src/encoder/m1/qctx/` (generated from `spec/tables/default_cdf.json`) holds
  all four slices.  `still_syntax.cpp` installs the slice before any symbol is
  coded; slice 0 is byte-identical to the historical tables, so the pinned
  `--still` MD5s do not move.
* `base_q_idx > 90` also needs the matching quantiser.  `still_encoder.cpp` derives
  `qlookup(base_q_idx)` from the normative table (`src/encoder/rc/rc_tables.h`):
  the forward `TxParams::quant_factor` is scaled by `528 / qlookup` and the
  inverse dequant uses `Round2(level * qlookup, 3) / dq_denom`, matching
  §7.14.4.  At QP 90 both forms reduce to the previous code exactly.
* Gate coverage: one case at each slice's lower bound (91, 141, 191) plus 255,
  on both fixtures, with `--strict 1` and zero recon differences.  All CPU
  tools (CDEF, CCSO, RDOQ, TX type, rate-aware intra, full UV, adaptive CDF)
  were also checked in combination at QPs 91/141/191/255.
* The M2 syntax writer (`video_syntax.cpp`) uses the same slice selection, the M2
  RC adapter accepts the full 8-bit qindex range, and `video_frame.cpp` uses
  `qlookup(base_q_idx)` for forward quant and inverse dequant.  The 2-frame and
  8-frame gates pass at QP 90 (bytes unchanged) and QP 141/191 with zero recon
  differences.
* `--still-gpu` still pins 90: its TX kernel bakes `kQuantStep = 528` and its
  decision path was not verified for other QPs.
* The RDOQ rate model (`rdoq::av2::BlockInfo::qctx`) and the rdo_decide
  transform-stream model (`TransformStreamBlock::qctx`) select the same slice.
  At QP > 90 the ranking therefore uses the correct static CDFs; at QP 90 the
  tables are byte-identical to the previous qctx-0 set.

## Not integrated

* **GPUs for features 1, 3, 4, 5, 6 and the new filter flags.**
  `GpuOptions` carries every M3 flag and its decision knobs,
  `gpu_options_to_m1()` mirrors them into the internal CPU parity encode, and
  `gpu_m3_options_supported()` rejects each unported tool **by name**
  (`--still-cdef`, `--still-ccso`, `--still-ccso-edge`, `--still-wiener`,
  `--still-chroma-lr`, `--still-pc-wiener`, `--still-gdf`, `--still-grain`,
  `--still-rdoq`, `--still-tx-type`, `--still-rdo-intra`, `--still-uv-full`)
  with a "not ported" message.  No flag is
  accepted-and-ignored, and the gate classifies these as `SKIP-REJECTED`, not
  `FAIL`.  `src/cuda/m1/still_gpu_options_test.cpp` pins the mapping and the
  rejection set on the host (no CUDA needed).  The GPU kernels/adapters for the
  pieces exist (`src/encoder/m1/transform_stream/transform_stream_gpu.cu`,
  `src/cuda/m1/restoration/`, `src/cuda/cdef/`), but the orchestrator still has
  a fixed DCT_DCT path and no CDEF/CCSO/restoration stage call.  Wiring them
  must be gated on the 3060 box; this integration host has no CUDA toolchain
  or device, so the 112-case matrix was **not** re-run here.
* **Wiener NS** — the §7.20.3 core and `filters_stream` tile/header writers
  exist, and the coefficient-fit/class adapters are landing under
  `src/encoder/filters/restoration/classes/`.  The §7.20.4 multi-class path is
  still blocked on the `Pc_Wiener_Lut_To_Class`/`Pc_Wiener_Sub_Classify2`
  tables, which are not in the spec text.  No M1 call site exists yet, so the
  tool stays off.
* **Delta-Q / segmentation** — `src/encoder/m1/deltaq/` has the frame/tile
  syntax and `src/encoder/rc/` has the planning structures, but the M1 header
  still codes `segmentation_enabled = 0` / `delta_q_present = 0` and the
  coefficient path assumes one frame qindex.  Wiring it changes per-SB dequant
  and deblock state, so it needs its own gate.
* **QM** — `src/encoder/qm/` has the normative matrices, `setup_qm_params()`
  writer and §7.14.4 dequant helpers, but the header still codes
  `using_qmatrix = 0` and the M1 transform path does not apply a matrix.
* **Extended partitions** (`src/encoder/partition_ext/`) — the solver,
  `enable_ext_partitions` header flag, 10-value `read_partition`, and chroma
  ownership rules are a separate integration.  The component gate is clean,
  but no M1 syntax path exists yet.
* **Unified RDO loop** — `src/encoder/m1/rdo_loop/` is the policy half
  (candidate order, pruning, branch-and-bound, cost breakdown); its encoder
  adapter/experiment was not present at integration time, so no CLI flag is
  wired.
* **10-bit** — headers, entropy contexts, transforms and reconstruction all
  change; deliberately last.

## Verification

```sh
# CPU reference build
cmake -S . -B build-m1 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DAV2ENC_ENABLE_CUDA=OFF -DAV2ENC_ALLOW_CPU_ONLY=ON
cmake --build build-m1 -j

# feature gate (pinned MD5 + dav2d --strict 1)
tools/still_verify/feature_gate.sh --bin build-m1/av2enc

# with encoder-recon exactness
g++ -O2 -std=c++20 -Isrc tools/still_verify/still_dump.cpp \
  -o build-m1/still_dump build-m1/libav2enc_core.a
tools/still_verify/feature_gate.sh --bin build-m1/av2enc --dump build-m1/still_dump

# ctest (includes the M0/M1 header + decoder smoke tests)
ctest --test-dir build-m1 --output-on-failure
```

The baseline `--still` MD5s are unchanged from `AGENTS.md`:
64x64 `0ef9825062a25c4d950ee924113c6f89`,
640x360 `e61de628b0812b716b0efa7dcafcdd06`.

## Partition / superblock tools (branch `integration/partition`)

All rows are **opt-in, default off**; the default `--still` / `--still-gpu`
streams stay byte-identical.  The CUDA planner accepts 64x64 superblocks only,
so the new modes are CPU-only and `--still-gpu` rejects them by name.

| Tool | Flag (default) | CPU | GPU | Gate |
|---|---|---|---|---|
| 128x128 superblocks | `--still-sb128` (off) | exact | rejected by name | PASS |
| 256x256 sequence SBs | `--still-sb256` (off) | exact | rejected by name | PASS |
| Extended partitions | `--still-ext-partitions` (off) | exact | rejected by name | PASS |

* §5.4.3 now writes the superblock flag pair: `use_256x256_superblock`, then
  `use_128x128_superblock` only when the first is 0.  Default (both 0) is
  byte-identical.
* `write_tile_info()` derives the tile grid from the selected SB size
  (`sb_mi = 32` for 128), and `still_syntax` emits `do_square_split` between
  `do_split` and `rect_type`.  `PARTITION_SPLIT` exists only at 128x128, so the
  64x64 path never codes the new symbol.
* The tree solver enables `allow_split` at 128 SB and makes >64x64 leaves
  unattractive, because the writer codes at most one 64x64 leaf per plane
  (TX_MODE_LARGEST).  Only 128x128 roots are split four ways; the 256x256
  recursion is inter-only (`sb256` module).
* CfL: `still_encoder` now passes `geometry_.sb_size` into `Block::sb_top_y`
  instead of the hardcoded 64.  With a 128 SB the §7.13.5 above-window is
  clamped to the 128-row superblock; the old constant produced exact luma but
  wrong chroma (`U`/`V` mismatched).
* Rejected combinations: `--still-gpu`, `--still-single`, `--still-depth 0`.
  `--still-cdef` and `--still-ccso` are supported on the 128 SB grid since the
  completion wave (see below).
* Gate: `dav2d --strict 1` on 256x256, 640x360 and 1920x1080, plus
  `still_dump`/`verify.py` recon equality on all three.  IVF MD5s:
  256x256 `bd9987821168eccf6bf135a930992cd3`,
  640x360 `eef3856c7d549dfd7a4f63f52714d2e8`,
  1080p `f3e8f3ea5182b72f8a50cbc6108cbf09`.

* `--still-sb256` writes `use_256x256_superblock = 1` and **no**
  `use_128x128_superblock` bit.  §5.18.1 clamps an intra frame to
  `SbSize = 128x128`, so the tile uses the sb128 tree/grid (including the
  CDEF/CCSO sizing below).  The 256x256 *coding* recursion is inter-only and
  not exercised by a still picture.
* Gate: `dav2d --strict 1` plus `still_dump` equality at 256x256 and 640x360.
  IVF MD5s: 256x256 `59a76b8d915bfd8959d5b167c02d8ea0`,
  640x360 `6936545a38c647297e2f0a85407ab040`.  Negative control: clearing the
  flag with `src/codec/partition/sb256/patch_use256_flag.py` makes `dav2d`
  reject the stream ("Error parsing sequence header"), proving it is parsed.

* `--still-ext-partitions` writes `enable_ext_partitions = 1` plus the dependent
  `enable_uneven_4way_partitions = 1` bit and routes the partition tree through
  `src/codec/partition/ext/` (`partition_ext::decide_frame`) and
  `src/codec/partition/stream/` (`plan_node` / `emit_symbols` in
  `still_syntax`).  `BlockSyntax::allowed_mask` is now 16-bit so the 10-value
  spec enum fits.
* Search policy (completion wave): chroma ownership is carried per leaf, so the
  shared 8x16/16x8 middle block of `HORZ_3@8x32` / `VERT_3@32x8` and the first
  children of a 4-way split code correctly.  The ext search uses the 4-sample
  floor; `HORZ_3`/`VERT_3` need the split dimension >= 32, `HORZ_4*` >= 32 and
  `VERT_4*` >= 64 (vertical uneven 4-way with a 4-wide 1/8 child still
  desyncs at the bottom edge; see `docs/negative-results.md`).
* Rejected combinations: `--still-gpu` (no CUDA extended tree),
  `--still-single` (a depth-0 search would have to select a 3-way row),
  `--still-sub8` and `--still-sdp` (not wired together yet).
* Gate (completion wave): `src/codec/partition/ext/product_gate.sh` checks the
  trace selects `HORZ_3@8x32`, `VERT_3@32x8` and `HORZ_4A@32x32` on
  640x360 shine and requires sample-exact recon with deblock on and off.
  The IVF MD5s above are from the earlier wave and change with the 4-sample
  floor; use the gate, not the old hashes.
### Completion wave (branch `integration/partition2`)

Adds four opt-in, default-off tools; the pinned `--still` streams and
`refactor/baseline.json` are unchanged.

| Tool | Flag (default) | Gate | Status |
|---|---|---|---|
| Sub-8x8 leaves | `--still-sub8` (off) | `src/codec/partition/sub8/product_gate.sh` | PASS |
| SDP two-tree intra | `--still-sdp` (off) | `src/codec/partition/sdp/product_gate.sh` | PASS (forced NONE trees) |
| Shared-chroma ext rows | `--still-ext-partitions` (extended) | `src/codec/partition/ext/product_gate.sh` | PASS |
| CDEF/CCSO at 128 SB | `--still-sb128/256` + `--still-cdef/ccso` | `src/codec/partition/sb128/product_gate.sh` | PASS |

* **sub-8**: `BlockSyntax` carries `has_chroma`/`chroma_offset`/
  `chroma_bsize`/`chroma_mi_row`/`chroma_mi_col` (behind `kFlagHasChromaInfo` so
  the GPU/basic producers keep the leaf-derived path); the encoder derives
  ChromaMi from `partition::PartitionNode`; UV is coded only on the owner at the
  ChromaMi origin.  `--still-sub8` sets the floor to 4 and depth >= 8,
  CPU-only, rejected with `--still-single` / `--still-ext-partitions` /
  `--still-gpu`.
* **SDP**: the sequence header writes `enable_sdp`; `collect_sdp_node()` emits
  the luma root then the chroma root under a no-syntax split node; partition
  symbols use `TileDoSplitCdf[PlaneStart]` (a separate chroma-plane context
  grid) and `CHROMA_PART` leaves skip the luma syntax.  The gated configuration
  forces `PARTITION_NONE` on both trees (`sdp::decide_frame`); any other
  sub-tree partition is rejected explicitly.  CPU-only, 64-SB, exclusive with
  sb128/256, ext, sub-8, CDEF/CCSO and segmentation/delta-Q.
* **ext**: `collect_node_ext()` now carries `has_chroma`/`chroma_offset`/
  `ChromaMi`, and `write_node_ext()` feeds them to
  `partition_stream::plan_node()`.  Gate selects `HORZ_3@8x32`,
  `VERT_3@32x8` and `HORZ_4A@32x32` and matches all planes exactly.
* **sb filters**: CDEF uses `sb_width_log2 = 5` and CCSO
  `CcsoLumaSizeLog2 = 7` (unit grid included) for a 128 SB.  Negative controls:
  forcing 4/6 makes `dav2d` reject the stream.
* `--recon` now also dumps the plain `--still` reconstruction (needed by the
  gates); default streams are unchanged.

## Intra integration pass (`integration/intra`)

Second integration wave for the standalone `src/codec/intra/*` components.
Every flag is off by default; the pinned `--still` / `--still-gpu` streams and
the `refactor/baseline.json` pins are unchanged.  CPU only — no CUDA stage was
wired here.

| Tool | Flag (default) | CPU | Gate |
|---|---|---|---|
| IBP (§5.4.5, §7.13.2.7-.12) | `--still-ibp` (off) | exact | PASS |
| MRL (§5.20.5.5, §7.13.2.1/.7/.8) | `--still-mrl` (off) | exact | PASS |
| CfL explicit alpha (§5.20.7.32) | `--still-cfl-explicit` (off) | exact | PASS |
| CfL DS filters (§5.4.5) | `--still-cfl-ds N` (0) | exact | PASS |
| DIP (§5.20.7.9, §7.13.2.3-.6) | `--still-dip` (off) | exact | PASS |
| MHCCP (CFL_MULTI, §7.13.6-.8) | `--still-mhccp` (off) | exact | PASS |
| Palette (§5.20.8) | `--still-palette` (off) | exact | PASS |
| IntraBC (§5.20.5.3/.4) | `--still-ibc` (off) | exact | PASS |

"exact": `dav2d` (`f9dafbc`) `--strict 1 --threads 1` decodes the stream and the
decoded planes equal the encoder reconstruction sample for sample.  The gate is
`tools/still_verify/intra_gate.sh --bin <av2enc> --dump <still_recon_dump>`;
`still_recon_dump` is the test-only harness built from
`src/pipeline/still/stream/still_recon_dump.cpp`.

Interactions and constraints:

* The §5.18.3.3/.4 frame bits are written through
  `src/codec/intra/screen_content/scc_flags.cpp`, which validates the
  combination (`force_integer_mv` only with `allow_screen_content_tools`,
  region bits only with `allow_intrabc`, no `change_bvp_drl`) and derives
  `IsIntraBC`/`FrameMvPrecision`.  `--still-palette`/`--still-ibc` set
  `allow_screen_content_tools`; both require `--still`, reject `--still-gpu`,
  `--lossless`, multi-tile `--tiles`, and the 10-bit/chroma format writers.
  IBC additionally requires `force_integer_mv` (always set) and 64x64
  superblocks.

* `--still-mrl` implies the directional luma search (only directional modes can
  signal `mrl_index`).  It rejects `--still-ibp` (IBP is specified for
  MrlIndex = 0) and `--still-rdo-intra` (no MRL rate model).
* `--still-dip` is a luma DC alternative; the DC-only default path scores plain
  DC first and keeps DIP only when the §7.13.2.4 decision wins by SSE.
* `--still-palette` is luma-only (§5.20.8 has no chroma palette): the palette
  decision replaces the DC luma prediction and the block codes `eob = 0` on all
  three planes, while chroma keeps the normal DC/CfL path.  It is only
  considered for `MiSize >= BLOCK_8X8` blocks of at most 64x64 whose luma mode
  is DC, and it sets `allow_screen_content_tools` in the frame header.
  §5.20.8.1's palette cache is interleaved above+left and may repeat a colour;
  `split_cache_and_run()` consumes each palette colour at most once and
  `encode_palette_y()` re-derives the coded palette and fails if it differs
  from the analysis result (see `docs/negative-results.md`).
* `--still-mhccp` requires CfL (`--no-cfl` is rejected) and codes
  `read_cfl_alphas()` through the gated MHCCP adapter, which owns the
  `cfl_mhccp` / `cfl_mh_dir` CDFs.
* `--still-ibc` codes `use_intrabc` / `skip_flag` / `intrabc_info` with the
  exact §5.20.4.1 NPos/NPosBuf contexts and the §7.12.2 BVP stack rebuilt from
  the same neighbour map the decoder keeps.  Because `skip_flag = 1` codes no
  residual, the decision only selects exact copies (checked on Y/U/V with
  `ibc::copy_block`); the search is stack-first, then periodic offsets, then a
  `--still-ibc-search` bounded local window (`NEWMV` when the vector is not a
  stack entry).  It signals `allow_intrabc` + global region (`allow_local = 0`),
  requires `force_integer_mv`, and rejects `--still-sb128`/`--still-sb256`
  (the §5.20.5.1 delay/ring model is 64x64-only).  A skip block still resets the
  coefficient contexts (§5.20.4.1 `reset_block_context`) or later blocks derive
  different CDF contexts.  Palette and IBC combine: screen-content tools also
  make every eligible DC block code `palette_mode_info()` even in an
  IBC-only stream.
* `--still-cfl-ds N` is the explicit sequence-level `cfl_ds_filter_index`
  (0..3, 3 aliases 0).  Index 0 remains the default because it won on all
  measured content.

Pinned gate MD5s (CPU, 2026-11-12, `tools/still_verify/intra_gate.sh --pin`):

| Case | IVF MD5 |
|---|---|
| `ibp_64` | `b57b3e4b687d1cf23dfd43ce5699a753` |
| `ibp_360` | `36e07364bad2d3d5e93093a1d6354c92` |
| `ibp_fullintra_360` | `b2d97369af6afb937536647d8dbe46b9` |
| `mrl_64` | `ecde69a68a87af141a2b72a709106206` |
| `mrl_360` | `ed9b4eb8df2f83712791309889d60291` |
| `mrl_txtype_64` | `9f8115d473bd73bb453a3a4597575ee0` |
| `cfl_ds1_360` | `7dbfbba32cc8fa120e8a4c86878069d7` |
| `cfl_ds2_360` | `3ef10468f121ea24a02de583ab7c8d99` |
| `cfl_explicit_64` | `d75429a0f29c2bcba5aba7482c1fe37d` |
| `cfl_explicit_360` | `c967038d187224c528e6c93a03c1c0cd` |
| `dip_64` | `ae4a44295f63ba32415c04d6d3875aa8` |
| `dip_360` | `0b187ba0b25eaff077e922d0642dab45` |
| `mhccp_64` | `40e72a1843f87481179661ad4ff2025f` |
| `mhccp_360` | `29051001bfb04557b204c5eaba692019` |
| `palette_scc64` | `47bfd8db98aaa08df240e0fe674fe0e5` |
| `palette_scc128` | `daf0532f15fc4d37f5ead6938943c074` |
| `palette_scc128_dip` | `ec81d5d3b4dd69e656d9e40b3912c712` |
| `palette_scc128_depth5` | `7db8078929a6043313700ea4ecf6261d` |
| `ibc_384x128` | `dc5fca8c44f78e51850afe7bf1ac8983` |
| `ibc_384x128_palette` | `de0e3a238163e751a6ae6c6b5ef5b4f5` |
| `ibc_384x128_depth5` | `3ba3788644e5b6456c34a060da970752` |

Not wired in this pass:

* **DC-subsampled intra (§7.13.2.11)** — unreachable in the still path:
  §7.13.2 dispatches it only for a *chroma* DC block with a CfL UV mode whose
  w or h exceeds 32.  M1's largest coded block is 64x64 (chroma 32x32), and
  `--still-sb128` roots still split to 64x64 rather than 64x128/128x64, so no
  chroma block is large enough.  `still_encoder.cpp` rejects such a block
  rather than emitting a non-normative prediction.
* **`dc_sign_horz_vert` (§5.20.7.27)** — unreachable: it is selected only for
  1D TX classes, `enable_idtx_intra = 0` filters identity-axis intra TX types
  out of the candidate sets, and the pinned dav2d rejects negative 1D
  coefficients at `--strict 1`.  `still_syntax.cpp` fails loudly if a 1D intra
  TX type ever reaches the sign coder.  `intra_filter_corner` (§7.13.2.14) is
  already wired in `intra_modes.cpp` and needs no flag.
* **`src/codec/intra/supplemental` reference files** — `dc_intra_subsampled.cpp`,
  `dc_sign_horz_vert.cpp` and `save_subpu.cpp` stay standalone (their component
  gate compiles them directly).  The production counterparts are
  `intra_modes.cpp` (corner, already wired) and `still_syntax.cpp` (sign
  dispatch); `save_subpu` belongs to the inter/deblock path, outside still
  scope.

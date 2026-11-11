# Spec audit 15 — CLI reachability of the AV2 coding tools

Scope: cross-cutting. Master list of coding tools is the sequence-header
on/off grammar in AV2 v1.0.0 §5.4.1–§5.4.10 (spec PDF pages 54–65). The
question answered per row is **not** "is there code in the tree" but "can the
shipped `av2enc` CLI actually put this tool into a bitstream".

Audited artifact: `build/av2enc` (Mach-O arm64), built from
`build/CMakeCache.txt` with `AV2ENC_ENABLE_CUDA=OFF` and
`AV2ENC_ALLOW_CPU_ONLY=ON`. The CPU-only binary is the only `av2enc`
executable currently shipped in this repo. A CUDA/M1-GPU build would add a
second path; its extra restrictions are noted where relevant.

Code read: `src/main.cpp` (flag parser and option mapping), the M0 and M1
header writers, `src/encoder/m1/m1_still.cpp`, `src/cuda/m1/m1_gpu.h`/`.cu`,
`CMakeLists.txt`, and every source directory under `src/`. README/status text
was treated as a claim to verify, not as evidence.

## Legend

| Status | Meaning |
|---|---|
| **YES** | A CLI flag controls the tool, the flag reaches the encoder options, and the path is available in the shipped binary. |
| **YES\*** | The shipped path always emits the tool (fixed on, inferred on, or fixed to one value). There is no CLI toggle. |
| **IMPL** | A real implementation exists (component tests and/or a decoder gate), but no production code path writes the enabling syntax and no CLI flag reaches it. |
| **ABSENT** | No implementation of the tool anywhere in `src/`; the header writer hardcodes it off. |

`reachable` in the counts = **YES + YES\***. `implemented-but-unreachable` =
**IMPL**.

---

## 1. Tool reachability matrix

`(none)` in the CLI column means the sequence flag is fixed by the shipped
path. "M0 + M1" evidence is `src/bitstream/m0_headers.cpp` and
`src/encoder/m1/m1_headers.cpp`; option plumbing is `src/main.cpp`.

| Tool | sequence flag | CLI flag | reachable? | evidence |
|---|---|---|---|---|
| Single-picture header | `single_picture_header_flag` | (none; always 1) | YES* | m0_headers.cpp:178; m1_headers.cpp:82 |
| Film grain | `film_grain_params_present` | — | IMPL | m0_headers.cpp:268 / m1_headers.cpp:159 force 0; `encoder/filters/spec_tables/film_grain.cpp` + gate |
| Sequence-level tile config | `seq_tile_info_present_flag` | — | IMPL | m1_headers.cpp:157 force 0; multi-tile writer in `encoder/m1/tiles_stream` + `encoder/tiles` (gate only) |
| Tile-info change | `allow_tile_info_change` | — | ABSENT | no writer sets `seq_tile_info_present_flag = 1` |
| 256x256 superblock | `use_256x256_superblock` | — | IMPL | m1_headers.cpp:103 force 0; `encoder/sb256` (dav2d gate) |
| 128x128 superblock | `use_128x128_superblock` | — | IMPL | m1_headers.cpp:104 force 0; `encoder/sb128` (dav2d gate) |
| Semi-decoupled partitioning | `enable_sdp` | — | IMPL | m1_headers.cpp:105 force 0; `encoder/sdp` |
| Extended SDP | `enable_extended_sdp` | — | IMPL | single-picture forces 0; `encoder/sdp/sdp_syntax.cpp` |
| Extended partitions | `enable_ext_partitions` | — | IMPL | m1_headers.cpp:106 force 0; `encoder/partition_ext`, `m1/partition_stream` |
| Uneven 4-way partitions | `enable_uneven_4way_partitions` | — | IMPL | m1_headers.cpp:107 force 0; `encoder/partition_ext`, `m1/partition_stream` |
| PB aspect-ratio cap | `reduce_pb_aspect_ratio` | — | ABSENT | m1_headers.cpp:107 force 0; no implementation |
| Extended segmentation | `enable_ext_seg` | — | ABSENT | m1_headers.cpp:110 force 0; only 8-segment constants in `lossless`/`dcdeltaq` |
| Sequence segment info | `seq_seg_info_present_flag` | — | ABSENT | m1_headers.cpp:111 force 0; no `seg_info()` writer |
| DIP | `enable_dip` | — | IMPL | m1_headers.cpp:114 force 0; `encoder/dip`, `cuda/dip` |
| Intra edge filter | `enable_intra_edge_filter` | (none; M1 fixed on) | YES* | m1_headers.cpp:115 writes `c.enable_intra_edge_filter`; m1_still.cpp:1811-1814; env `AV2ENC_M1_NO_EDGE` is the only off switch |
| MRLS | `enable_mrls` | — | IMPL | m1_headers.cpp:116 force 0; `encoder/mrl`, `m1/mrl_stream`, `cuda/mrl` |
| CfL intra | `enable_cfl_intra` | `--m1` + `--no-cfl` | YES | m1_headers.cpp:117; m1_still.cpp:1814; main.cpp:283-286, 634 |
| CfL downsampling filter | `cfl_ds_filter_index` | (none; fixed 0) | YES* | m1_headers.cpp:118 writes 0; `cfl/ds_filters` implements other indices but is not linked |
| MHCCP | `enable_mhccp` | — | IMPL | m1_headers.cpp:119 force 0; `encoder/cfl/mhccp` |
| IBP | `enable_ibp` | — | IMPL | m1_headers.cpp:120 force 0; `encoder/ibp`, `cuda/ibp` |
| Motion-mode set | `seq_enabled_motion_modes[]` | — | IMPL | single-picture forces all 0; `encoder/inter/*`, `encoder/m2` |
| Six-parameter warp delta | `enable_six_param_warp_delta` | — | IMPL | `encoder/m2/advanced/advanced_syntax` |
| Masked compound | `enable_masked_compound` | — | IMPL | `encoder/m2/multiref/compound_predict` |
| Ref-frame MVs | `enable_ref_frame_mvs` | — | IMPL | `encoder/inter/temporal/motion_field` |
| Reduced ref-frame MVs | `reduced_ref_frame_mvs_mode` | — | IMPL | `encoder/inter/temporal/temporal_syntax` |
| Order hints | `OrderHintBits` | — | IMPL | single-picture forces 0; `inter/temporal`, `inter/random_access` |
| Ref MV bank | `enable_refmvbank` | — | IMPL | m1_headers.cpp:123 force 0; `inter/temporal/ref_mv_bank.h` |
| DRL reorder | `disable_drl_reorder` / `constrain_drl_reorder` | (none; fixed disabled) | IMPL | m1_headers.cpp:124 writes 1 (disabled); `inter/drl_reorder` never called |
| Explicit ref-frame map | `explicit_ref_frame_map` | — | IMPL | `inter/advanced/b_frame`, `inter/random_access` |
| Explicit ref count | `explicit_num_ref_frames` | — | IMPL | `inter/advanced`, `inter/random_access` |
| Long-term refs | `long_term_frame_id_bits` | — | IMPL | `encoder/inter/longterm` |
| DRL bit budget | `seq_max_drl_bits` / `allow_frame_max_drl_bits` | — | IMPL | `inter/drl_reorder`, `m2/multiref` |
| BVP DRL bit budget | `seq_max_bvp_drl_bits` / `allow_frame_max_bvp_drl_bits` | — | IMPL | m1_headers.cpp:125-126 write 0/false; `m1/ibc_stream`, `encoder/ibc` |
| Same-ref compound count | `num_same_ref_compound` | — | IMPL | `inter/random_access`, `inter/tlayers` |
| TIP | `enable_tip` | — | IMPL | `encoder/inter/temporal/tip.h` |
| TIP hole fill | `enable_tip_hole_fill` | — | IMPL | `encoder/inter/temporal/motion_field` |
| MV trajectory | `enable_mv_traj` | — | IMPL | `encoder/inter/temporal/motion_field` |
| BAWP | `enable_bawp` | — | IMPL | m1_headers.cpp:127 force 0; `inter/advanced/bawp.h`, `temporal_syntax` |
| CWP | `enable_cwp` | — | IMPL | `inter/advanced/compound`, `temporal_syntax` |
| Implicit masked build | `enable_imp_msk_bld` | — | IMPL | `inter/advanced/compound_common`, `temporal_syntax` |
| Sub-PU deblock | `enable_df_sub_pu` | — | IMPL | `inter/temporal/temporal_syntax` |
| TIP explicit QP | `enable_tip_explicit_qp` | — | IMPL | `inter/temporal/temporal_syntax` |
| OPFL refinement | `enable_opfl_refine` | — | IMPL | `inter/temporal/opfl.h` |
| RefineMV | `enable_refinemv` | — | IMPL | `inter/temporal/refinemv.cpp` |
| TIP RefineMV | `enable_tip_refinemv` | — | IMPL | `inter/temporal/temporal_syntax` |
| BRU | `enable_bru` | — | IMPL | `inter/random_access` |
| Adaptive MVD | `enable_adaptive_mvd` | — | IMPL | `inter/random_access` |
| MVD sign derivation | `enable_mvd_sign_derive` | — | IMPL | `inter/random_access` |
| Flexible MV resolution | `enable_flex_mvres` | — | IMPL | `inter/gme/gme_syntax` |
| Global motion | `enable_global_motion` | — | IMPL | single-picture forces 0; `encoder/inter/gme` |
| Short refresh frame flags | `enable_short_refresh_frame_flags` | — | IMPL | `inter/gme/gme_syntax`, `random_access` |
| Screen-content tools | `seq_force_screen_content_tools` | — | IMPL | single-picture forces SELECT; `encoder/palette`, `m1/palette_stream`, `cuda/palette` |
| Integer MV | `seq_force_integer_mv` | — | IMPL | single-picture forces SELECT; `m1/ibc_stream`, `palette` |
| FSC | `enable_fsc` | — | IMPL | m1_headers.cpp:130 force 0; `encoder/lossless` |
| IDTX intra | `enable_idtx_intra` | — | IMPL | m1_still.cpp:1816 hardcodes false; `tx_variants`/`tx_type` support IDTX but filter it out |
| Intra IST | `enable_intra_ist` | — | IMPL | m1_headers.cpp:132 force 0; `encoder/tx_secondary`, `cuda/tx_secondary` |
| Inter IST | `enable_inter_ist` | — | IMPL | m1_headers.cpp:133 force 0; `encoder/tx_secondary`, `cuda/tx_secondary` |
| Chroma DCT-only | `enable_chroma_dctonly` | (none; fixed 1) | YES* | m1_headers.cpp:134; m1_still.cpp:1815 |
| Inter DDT | `enable_inter_ddt` | — | IMPL | single-picture omits it; `encoder/m2/ddt` |
| Reduced TX partition set | `reduced_tx_part_set` | — | IMPL | m1_headers.cpp:135 force 0; `quant/dcdeltaq/syntax` (syntax only) |
| CCTX | `enable_cctx` | — | IMPL | m1_headers.cpp:136 force 0; `encoder/tcq/cctx` |
| TCQ | `enable_tcq` | — | IMPL | m1_headers.cpp:137 force 0; `encoder/tcq` |
| Per-frame TCQ | `choose_tcq_per_frame` | — | IMPL | `encoder/tcq/tcq_flags` |
| Parity hiding | `enable_parity_hiding` | — | IMPL | m1_headers.cpp:138 force 0; `encoder/tcq/parity` |
| Average CDF | `enable_avg_cdf` | (none; inferred 1) | YES* | single-picture infers 1; one tile makes it inert; `entropy/avg_cdf` unused |
| Average-CDF type | `avg_cdf_type` | (none; inferred 1) | YES* | single-picture infers 1; `entropy/avg_cdf` unused |
| Separate UV delta-Q | `separate_uv_delta_q` | — | IMPL | m1_headers.cpp:139 force 0; `quant/dcdeltaq` |
| Equal AC/DC quant | `equal_ac_dc_q` | (none; fixed 1) | YES* | m1_headers.cpp:140; m0_headers.cpp:248 |
| Y DC delta-Q | `base_y_dc_delta_q` / `y_dc_delta_q_enabled` | — | IMPL | m1_headers omits the fields; `quant/dcdeltaq` |
| UV DC delta-Q | `base_uv_dc_delta_q` / `uv_dc_delta_q_enabled` | — | IMPL | m1_headers omits the fields; `quant/dcdeltaq` |
| UV AC delta-Q | `base_uv_ac_delta_q` / `uv_ac_delta_q_enabled` | — | IMPL | m1_headers.cpp:141-142 writes 0/false; `quant/dcdeltaq` |
| Cross-tile loop filter disable | `disable_loopfilters_across_tiles` | — | ABSENT | m1_headers.cpp:145 force 0; single-tile only |
| CDEF | `enable_cdef` | `--m1` + `--m1-cdef` | YES | m1_headers.cpp:146; m1_still.cpp:1823-1830; main.cpp:358-361 |
| GDF | `enable_gdf` | — | IMPL | m1_headers.cpp:147 force 0; `encoder/filters/spec_tables/gdf` |
| GDF unit size match | `gdf_unit_matches_sb_size` | — | IMPL | only coded when `enable_gdf`; gdf component |
| Loop restoration | `enable_restoration` | — | IMPL | m1_headers.cpp:148 force 0; `encoder/filters/restoration`, `lr_chroma` |
| LR tool disable mask | `lr_tools_disable[][PC_WIENER/WIENER_NONSEP]` | — | IMPL | only coded with `enable_restoration`; `restoration/classes`, `lr_chroma` |
| LR UV presence | `lr_tools_uv_present` | — | IMPL | only coded with `enable_restoration`; `lr_chroma` |
| CCSO | `enable_ccso` | `--m1` + `--m1-ccso` | YES | m1_headers.cpp:150; m1_still.cpp:1799-1807; main.cpp:363-366 |
| CCSO unit size match | `ccso_unit_matches_sb_size` | (none; fixed 1 with CCSO) | YES* | m1_headers.cpp:151-153 |
| CDEF on skip TX | `CdefOnSkipTxfm` | (none; forced adaptive) | YES* | single-picture forces ADAPTIVE; per-frame bit from CDEF search, m1_headers.cpp:206 |
| Deblock parallelism | `df_par_bits_minus_2` | — | ABSENT | m1_headers.cpp:154 force 0; no implementation |

### Non-sequence CLI switches (for completeness)

These tools do not have a §5.4 sequence flag but are exposed by the same CLI.
They are outside the 82-row master count.

| Tool | CLI flag | reachable? | evidence |
|---|---|---|---|
| Deblocking filter (frame-level) | `--no-deblock` | YES | m1_still.cpp:1819-1822 -> m1_headers.cpp:182-197 |
| Adaptive CDF (§8.2.6 update) | `--adaptive-cdf` | YES | m1_still.cpp:1810; m0_still.cpp:498; av2_entropy.h:185 |
| Directional/smooth intra modes | `--m1-full-intra` | YES | main.cpp:639 sets `luma_dc_only = false`; m1_still.cpp:622 |
| Full chroma intra search | `--m1-uv-full` | YES | main.cpp:640; m1_still.cpp:1227 |
| Rate-aware luma intra decision | `--m1-rdo-intra` | YES | main.cpp:641; m1_still.cpp:653-710 |
| Full-trellis RDOQ | `--m1-rdoq` | YES | main.cpp:643; m1_still.cpp:900-927 |
| Intra luma TX_TYPE search | `--m1-tx-type` | YES | main.cpp:645; m1_still.cpp:1052 |
| Partition depth cap | `--m1-depth` | YES | main.cpp:651-653 |
| One-SB / DC-only debug gates | `--m1-single`, `--m1-dc` | YES (hidden) | parsed main.cpp:293-301 but absent from `--help` |
| Quantiser | `--qp` | YES, M1 only | main.cpp:533-543; M0 and M1-GPU pin 90, qp 0 rejected |
| Chroma format | (none) | fixed 4:2:0 | m1_headers.cpp:84; Y4M reader rejects non-C420 (`io/y4m.cpp:43-46,192-194`) |
| Bit depth | (none) | fixed 8-bit | m1_headers.cpp:85; no 10/12-bit path in the CLI |
| Multiple frames | (none) | ABSENT | `--frames` is parsed but only one frame is read (main.cpp:545-552) |
| M1 GPU backend | `--m1-gpu` | NO in shipped build | build is CPU-only; main.cpp:472-487 rejects the flag. A CUDA build gates it, but `m1_gpu.h:282-336` then rejects cdef/ccso/rdoq/tx_type/rdo-intra/uv-full |

---

## 2. Component inventory

"Production" column:

* **CPU** — linked into `av2enc_core` and reached from `src/main.cpp` (directly
  or through `m1_still`).
* **HDR** — header reached from production, implementation not linked.
* **CUDA** — compiled only in the CUDA/M1-GPU target (CMakeLists 129-203). Not
  in the shipped CPU binary.
* **no** — not in any production target.
* **dead** — linked into `av2enc_core` but no call site.

`tests?` = a test source exists (most are standalone `build_and_test.sh`; only
`bitwriter`, `ivf`, `y4m`, `m0_headers`, `m1_headers`, `m1_gpu_options` are in
top-level CTest). `gate?` = a decoder/conformance gate exists (dav2d or an
equivalent harness).

| Component dir | tests? | gate? | called from production? | evidence |
|---|---|---|---|---|
| src/bitstream | — | — | CPU | CMake 77-79; m0/m1 header+OBU writers |
| src/io | — | — | CPU | CMake 75-76; main.cpp Y4M/IVF |
| src/entropy | Y | — | no | root `range_coder.*` not in CMake; production uses `entropy/av2` |
| src/entropy/av2 | Y | — | CPU | CMake 98; `m0_still.cpp`/`m1_syntax.cpp` |
| src/profile | — | — | dead | `profiler.cpp` in CMake 99 but zero include/call sites |
| src/cpu | — | — | empty | empty directory (junk) |
| src/build-dcdeltaq | — | — | empty | empty directory (junk) |
| src/build-hdr | — | — | empty | empty directory (junk) |
| src/encoder | Y | Y | CPU | `m0_still.cpp` CMake 80; M0/M1 gate scripts |
| src/encoder/partition | Y | — | CPU | CMake 81; M1 partition tree |
| src/encoder/intra_modes | Y | — | CPU | CMake 82; M1 intra prediction |
| src/encoder/rdoq | Y | Y | CPU | CMake 83-84; `--m1-rdoq` |
| src/encoder/tx_type | Y | — | CPU | CMake 85; `--m1-tx-type` |
| src/encoder/tx_variants | Y | — | CPU | CMake 86-87; TX kernels/selection |
| src/encoder/cfl | Y | Y | CPU | CMake 88; CfL stage |
| src/encoder/filters/deblock | Y | — | CPU | CMake 89; M1 deblock |
| src/encoder/filters/cdef | Y | — | CPU | CMake 90; `--m1-cdef` |
| src/encoder/filters/restoration | Y | — | CPU | CMake 91; CCSO used, Wiener disabled by `enable_restoration=0` |
| src/encoder/rdo | Y | — | CPU | CMake 92-93; `rdo.cpp`, `intra_decision.cpp` |
| src/encoder/rc | Y | Y | HDR | only `rc_tables.h` is included (m1_still.cpp:24); `rc.cpp` not linked |
| src/encoder/rc/hrd | Y | Y | no | HRD/BRT not wired |
| src/encoder/rc/twopass | Y | — | no | two-pass planner not wired |
| src/encoder/m1 | Y | Y | CPU | CMake 94-96; `m1_still` is the `--m1` path |
| src/encoder/m1/qctx | Y | Y | CPU | included at m1_still.cpp:19; QP-slice CDFs |
| src/encoder/m1/stream | Y | — | CPU | CMake 97; plan->syntax bridge |
| src/encoder/m1/deltaq | Y | — | no | frame `delta_q_present=0`; not linked |
| src/encoder/m1/filters_stream | Y | — | no | M1 calls filters directly; stream wrapper unused |
| src/encoder/m1/ibc_stream | Y | Y | no | IBC disabled in sequence header |
| src/encoder/m1/mrl_stream | Y | Y | no | MRLS disabled in sequence header |
| src/encoder/m1/palette_stream | Y | Y | no | SCC/palette disabled |
| src/encoder/m1/partition_stream | Y | Y | no | M1 uses `encoder/partition`, not this stream writer |
| src/encoder/m1/rdo_decide | Y | — | no | M1 uses `encoder/rdo/intra_decision` instead |
| src/encoder/m1/rdo_loop | Y | — | no | separate RDO loop, not linked |
| src/encoder/m1/sub8 | Y | Y | no | `min_block_size=8` hardcoded; 4x4 path unused |
| src/encoder/m1/tiles_stream | Y | Y | no | M1 writes one tile in `m1_headers` |
| src/encoder/m1/transform_stream | Y | — | no | GPU .cu not in CMake GLOB (only `src/cuda/m1`) |
| src/encoder/sb128 | Y | Y | no | 128-SB decoder-verified gate only |
| src/encoder/sb256 | Y | Y | no | 256-SB decoder-verified gate only |
| src/encoder/dip | Y | — | no | DIP component + tests only |
| src/encoder/hbd | Y | Y | no | high-bit-depth encoder not wired |
| src/encoder/hdr | Y | Y | no | HDR metadata append tool, not encoder path |
| src/encoder/ibc | Y | Y | no | IBC disabled |
| src/encoder/ibp | Y | — | no | IBP disabled |
| src/encoder/lossless | Y | Y | no | qp 0 rejected by main.cpp:534-537 |
| src/encoder/m10 | Y | Y | no | 10-bit still encoder, separate CLI-less gate |
| src/encoder/mrl | Y | — | no | MRLS disabled |
| src/encoder/palette | Y | Y | no | SCC/palette disabled |
| src/encoder/partition_ext | Y | Y | no | extended partitions disabled |
| src/encoder/pipeline | Y | Y | no | decision-only composition of rc+tiles+multi_gpu |
| src/encoder/qm | Y | Y | no | user QM OBU not emitted |
| src/encoder/sdp | Y | — | no | SDP disabled |
| src/encoder/svc | Y | Y | no | SVC layers not wired |
| src/encoder/syntax/small_header | Y | Y | no | alternative header writer, not linked |
| src/encoder/tcq | Y | — | no | TCQ/CCTX component only |
| src/encoder/tcq/parity | Y | — | no | parity hiding component only |
| src/encoder/tiles | Y | — | no | multi-tile layout not used by M1 |
| src/encoder/tx_secondary | Y | — | no | IST component only |
| src/encoder/filters/lr_chroma | Y | Y | no | untracked new dir; loop-restoration chroma, not linked |
| src/encoder/filters/spec_tables | Y | Y | no | film grain + GDF tables/tools, not linked |
| src/encoder/filters/restoration/classes | Y | — | no | Wiener classes, not linked |
| src/encoder/cfl/mhccp | Y | — | no | MHCCP disabled |
| src/encoder/cfl/ds_filters | Y | Y | no | only hardcoded index 0; component not linked |
| src/encoder/cfl/alpha | Y | — | no | alpha-plane CfL, not linked |
| src/encoder/cfl/alpha/cuda_stub | — | — | dead | header-only stub |
| src/encoder/entropy/avg_cdf | Y | Y | no | one-tile path never averages |
| src/encoder/format | Y | Y | CUDA | CMake 146-151 adds `format.cpp` only with M1 GPU |
| src/encoder/format/chroma | Y | Y | no | 4:2:2/4:4:4/4:0:0 gate only |
| src/encoder/frame/mfh | Y | Y | no | multi-frame header not used (single frame) |
| src/encoder/quant/dcdeltaq | Y | Y | no | delta-Q component only |
| src/encoder/inter/advanced | Y | — | no | M2 inter tools, not linked |
| src/encoder/inter/bframes | Y | — | no | B-frames not wired |
| src/encoder/inter/drl_reorder | Y | Y | no | untracked new dir; M1 hardcodes reorder disabled |
| src/encoder/inter/gme | Y | Y | no | global motion not wired |
| src/encoder/inter/leading | Y | Y | no | leading pictures not wired |
| src/encoder/inter/longterm | Y | Y | no | long-term refs not wired |
| src/encoder/inter/mc | Y | — | no | MC component only |
| src/encoder/inter/me | Y | — | no | ME component only |
| src/encoder/inter/me2 | Y | — | no | second ME component only |
| src/encoder/inter/random_access | Y | Y | no | random-access group not wired |
| src/encoder/inter/sef | Y | Y | no | SEF not wired |
| src/encoder/inter/temporal | Y | — | no | TIP/OPFL/refine component only |
| src/encoder/inter/tlayers | Y | Y | no | temporal layers not wired |
| src/encoder/m2 | Y | Y | no | entire M2 encoder absent from CMake |
| src/encoder/m2/advanced | Y | — | no | M2 advanced inter tools |
| src/encoder/m2/cdef | Y | Y | no | M2 CDEF gate only |
| src/encoder/m2/ddt | Y | Y | no | inter DDT gate only |
| src/encoder/m2/deltaq | Y | Y | no | M2 delta-Q gate only |
| src/encoder/m2/merge | Y | — | no | M2 merge component |
| src/encoder/m2/multiref | Y | Y | no | multi-ref component |
| src/encoder/m2/post | Y | Y | no | M2 post-filters |
| src/encoder/m2/rdo | Y | — | no | M2 RDO |
| src/encoder/m2/residual | Y | Y | no | inter residual syntax |
| src/encoder/m2/restoration | Y | — | no | M2 restoration |
| src/encoder/metadata/content_interpretation | Y | Y | no | metadata OBU append tool |
| src/encoder/metadata/types | Y | Y | no | metadata types append tool |
| src/cuda | Y | Y | CUDA | `arena.cpp`/`context.cpp` CMake 117-119 only with CUDA |
| src/cuda/cdef | Y | — | CUDA | M1-GPU support source CMake 160 |
| src/cuda/cfl | Y | — | CUDA | support source CMake 156 |
| src/cuda/deblock | Y | — | CUDA | support source CMake 157 |
| src/cuda/dip | Y | — | no | GPU DIP not linked |
| src/cuda/format | Y | — | no | high-bit-depth GPU format not linked |
| src/cuda/ibc | Y | — | no | GPU IBC not linked |
| src/cuda/ibp | Y | — | no | GPU IBP not linked |
| src/cuda/inter/advanced | Y | — | no | GPU advanced inter not linked |
| src/cuda/inter/mc | Y | — | no | GPU MC not linked |
| src/cuda/inter/me | Y | — | no | GPU ME not linked |
| src/cuda/intra_modes | Y | — | CUDA | support source CMake 158 |
| src/cuda/m1 | Y | Y | CUDA | `m1_gpu.cu` GLOB; only with CUDA+entry point |
| src/cuda/m1/cdef | Y | Y | CPU+CUDA | `m1_cdef.cpp` CMake 103 (CPU); `.cu` GPU |
| src/cuda/m1/cfl | Y | — | CUDA | included by m1_gpu.cu:41 |
| src/cuda/m1/chroma | Y | — | CUDA | included by m1_gpu.cu:38 |
| src/cuda/m1/hbd | Y | Y | no | 10-bit GPU stage not referenced by m1_gpu.cu |
| src/cuda/m1/plan | Y | Y | HDR+CUDA | `plan.h` via m1_stream_bridge.h:28; `plan.cpp` only GPU GLOB |
| src/cuda/m1/recon | Y | — | CUDA | included by m1_gpu.cu:35 |
| src/cuda/m1/restoration | Y | Y | no | stage not referenced by m1_gpu.cu |
| src/cuda/m1/sched | Y | — | CUDA | included by m1_gpu.cu:44 |
| src/cuda/m1/search | Y | — | CUDA | included by m1_gpu.cu:29 |
| src/cuda/m1/tx | Y | — | CUDA | included by m1_gpu.cu:32 |
| src/cuda/m2 | Y | Y | no | GPU M2 not linked |
| src/cuda/mrl | Y | — | no | GPU MRLS not linked |
| src/cuda/palette | Y | — | no | GPU palette not linked |
| src/cuda/pipeline | — | — | no | no test, no gate, not linked |
| src/cuda/qm | Y | — | no | GPU QM not linked |
| src/cuda/rdoq | Y | — | no | GPU RDOQ batch not linked |
| src/cuda/restoration | Y | — | CUDA | support sources CMake 161-162 |
| src/cuda/scheduler | Y | Y | no | M0 scheduler component only |
| src/cuda/tx_secondary | Y | — | no | GPU IST not linked |
| src/cuda/tx_type | Y | — | no | GPU TX type not linked |
| src/cuda/tx_variants | Y | — | CUDA | support source CMake 159 |
| src/multi_gpu | Y | Y | no | planner only |
| src/multi_gpu/runtime | Y | Y | no | runtime backend not linked |

---

## 3. Counts

Master list (§5.4.1–§5.4.10 sequence flags): **82 tools/flags**.

| Category | Count | Rows |
|---|---:|---|
| Reachable (YES + YES*) | **12** | 3 CLI-controlled (`enable_cfl_intra`, `enable_cdef`, `enable_ccso`) + 9 fixed/inferred |
| Implemented-but-unreachable (IMPL) | **64** | every inter tool, SCC/palette/IBC, DIP/MRLS/MHCCP/IBP, SDP/ext partitions, 128/256 SB, TCQ/CCTX/parity/IST/IDTX/FSC, GDF/restoration/Wiener, delta-Q, film grain |
| Absent (ABSENT) | **6** | `allow_tile_info_change`, `reduce_pb_aspect_ratio`, `enable_ext_seg`, `seq_seg_info_present_flag`, `disable_loopfilters_across_tiles`, `df_par_bits_minus_2` |
| **Total** | **82** | |

Of the 12 "reachable", only **3** are actually controlled by a CLI flag:

1. `enable_cfl_intra` (`--no-cfl`, on by default with `--m1`)
2. `enable_cdef` (`--m1-cdef`)
3. `enable_ccso` (`--m1-ccso`)

The other 9 are fixed by the single-picture M1 path (always on, inferred on, or
fixed to one value) and cannot be changed without a rebuild or an environment
variable. `--m1-cdef`/`--m1-ccso` are silently ignored unless `--m1` is also
present (verified: identical IVF MD5 with and without the flags in M0 mode).

Component directories under `src/`: **131** inventoried (build directories,
`gate/`, `experiment/`, `tables/`, and test subdirectories are folded into
their parent component).

| Production status | Count |
|---|---:|
| CPU production (17 CPU + 1 CPU/CUDA + 2 header-only) | 20 |
| CUDA/M1-GPU-only | 15 |
| Dead/uncallable | 2 (`src/profile`, `cfl/alpha/cuda_stub`) |
| Empty directories | 3 |
| Not in any production target (tests/gates only) | 91 |
| **Total** | **131** |

Only two component tests (`m1_gpu_options_test` and `m1_feature_gate`) plus the
five top-level unit tests and three shell smokes are wired into CTest. The
other ~70 component `*_test.cpp` files are standalone scripts.

---

## 4. Dead code, duplication, and process findings

### Dead in the shipped binary

* `src/profile/profiler.cpp` is compiled into `av2enc_core` (CMakeLists:99) but
  no file includes `profiler.h`; `nm` shows the symbols, there are no callers.
* `src/entropy/range_coder.*` is a second entropy coder alongside the
  production `src/entropy/av2/av2_entropy.*`. Only `av2` is linked.
* `src/cuda/pipeline/pipeline_gpu.*` has no tests, no gate, and no CMake entry.
* `src/encoder/cfl/alpha/cuda_stub/` is an unused header-only stub.
* Empty directories: `src/cpu/`, `src/build-dcdeltaq/`, `src/build-hdr/`,
  `src/profile/build/` (build artifacts inside `src/`).
* `src/encoder/pipeline/` is a decision-only composition of `rc` + `tiles` +
  `multi_gpu`; none of its outputs reach the bitstream.
* Two untracked new trees, `src/encoder/filters/lr_chroma/` and
  `src/encoder/inter/drl_reorder/`, are gate-only and not in CMake.

### Parallel/duplicated implementations

* **M1 has two copies of the tool pipeline.** `m1_still.cpp` implements the
  search/syntax inline; `src/encoder/m1/{partition,transform,filters,mrl,
  palette,ibc,tiles,deltaq}_stream/` are separate writer implementations of
  the same tools, none linked. `rdo_decide` and `rdo_loop` duplicate
  `src/encoder/rdo/`.
* **CPU/GPU twins** for nearly every tool: `dip`, `mrl`, `ibp`, `ibc`,
  `palette`, `tx_type`, `tx_variants`, `tx_secondary`, `cdef`, `restoration`,
  `deblock`, `cfl`, `intra_modes`.
* **CDEF exists four times**: `encoder/filters/cdef` (production CPU),
  `cuda/cdef`, `cuda/m1/cdef` (production host half), `encoder/m2/cdef`.
* **Loop restoration exists six times**: `encoder/filters/restoration`
  (production, but only CCSO reachable), `restoration/classes`,
  `filters/lr_chroma`, `cuda/restoration`, `cuda/m1/restoration`,
  `encoder/m2/restoration`.
* **RDO exists four times**: `encoder/rdo` (production), `m1/rdo_decide`,
  `m1/rdo_loop`, `m2/rdo`.
* **Partitioning exists five times**: `encoder/partition` (production),
  `partition_ext`, `sdp`, `sb128`, `sb256`, plus `cuda/scheduler/m0_geometry.h`.
* **The whole M2 encoder** (`encoder/m2/**`, `cuda/m2`, the inter components)
  is absent from `CMakeLists.txt`. It is reachable only through per-component
  gates; the CLI has no M2 flag at all.

### Reachability warnings worth fixing

1. `--m1-*` flags are accepted with no `--m1` and do nothing. A typo (or an
   M0 invocation with tool flags) silently produces the M0 stream. Either
   imply `--m1` or reject.
2. `--m1-single` and `--m1-dc` are hidden from `--help` but parsed. Either
   document them as debug gates or remove them.
3. `--frames` claims to select a frame count but only one frame is ever read;
   values `> 1` are accepted and ignored.
4. `--m1-gpu` is advertised in `--help` although the shipped build cannot use
   it. In a CUDA build, `gpu_m3_options_supported()` (m1_gpu.h:282-336) also
   rejects cdef, ccso, rdoq, tx-type, rdo-intra and uv-full, so those flags
   have no GPU path even there.
5. `enable_idtx_intra` is hardcoded false at `m1_still.cpp:1816` while the TX
   type search silently filters IDTX candidates. If `--m1-tx-type` is meant to
   cover the full §5.4.8 set, the sequence flag and the candidate filter must
   be changed together.
6. `enable_avg_cdf = 1` / `avg_cdf_type = 1` are inferred by the single-picture
   header, but the encoder is one-tile, so the average-CDF component is dead
   weight. Multi-tile M1 (`tiles_stream`) is the only path that would exercise
   it and is not wired.

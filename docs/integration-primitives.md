# Integration primitives — findings and consolidation

Branch: `integration/primitives` · Base: `integration-base` · 2026-11-12

Scope: shared foundations only. No stream, header option, or CLI change; every
pinned stream is byte-identical (see "Gates"). This note records what is
shared, what is still duplicated, and how each duplicate is contained.

## 1. What changed

| Commit | Change |
|---|---|
| `ada2e96` | qctx CDF registry: regenerated header (provenance line only) and restored the `--check` cross-check, whose `src/encoder/...` paths had been dead since the folder move (it silently compared **0** tables). It now compares slice 0 against 20 tables in `basic_cdf_tables.h` and 22 in `rdoq_cdf_tables.h`. |
| `155632c` | New `src/spec_tables/spec_math.h`: one definition of §8.2 `kCdfScale`/`kMaxCdfSymbols`/`kCdfCountCap`/`kMaxAverageLog2` and the exact §8.2.4 `avg_cdf()` update. `av2::average_cdf()` and `avg_cdf::average_cdf()` now share it instead of carrying two copies of the arithmetic. |
| `341d860` | `cdf_continuity` no longer re-declares the 90/140/190 qctx thresholds or the selection function; it aliases `qctx::slice_for_base_q_idx()` (the registry). |
| `1e51124` | Default-CDF generators repaired: `generate_basic_cdf.py` had a wrong `ROOT` **and** a stale output path (`src/encoder/basic_cdf_tables.h`, i.e. it could not regenerate the file it owns). `generate_still_tables.py` and `generate_video_tables.py` had stale provenance lines. `--check` added to the basic and still generators; all regenerated headers are byte-identical apart from the path comment. |
| `283cb9b` | `generate_rc_tables.py` (qlookup) had a wrong `ROOT`; it could not read `spec/tables/quant.json`. Fixed; `rc_tables.h` regenerates identically apart from the path comment. |

No file moved, no option added, no CDF/quant/scan/DCT **value** changed.

## 2. Duplication inventory

"Agree?" means a value-by-value comparison was run during this audit.

| Primitive | Copies in tree | Agree? | Containment | Risk |
|---|---|---|---|---|
| Coefficient default CDFs (§9.3, `COEFF_CDF_Q_CTXS`) | `qctx_cdf_tables.h` (27×4 slices), `basic_cdf_tables.h` (20 slice-0), `rdoq_cdf_tables.h` (22 slice-0) | yes (20+22 tables, verified) | `generate_qctx_tables.py --check` now actually runs; slice 0 is compared against both shipping headers | Low. Regeneration is gated. Physical merge would touch the M1/M2 writers and rdoq; deferred. |
| Non-coefficient default CDFs | `kYModeSetCdf`, `kYModeIndexCdf`, `kUvModeCflNotAllowedCdf` exist in **both** `basic_cdf_tables.h` and `still_cdf_tables.h`; `video_cdf_tables.h` is disjoint | yes (3 arrays, 50 values) | both sides generated from `spec/tables/default_cdf.json`; `--check` added to both generators | Low. A real merge means switching `still_syntax.cpp`/`video_syntax.cpp` to one namespace; that is shared-file churn, deferred. |
| §8.2 CDF constants + `avg_cdf()` arithmetic | was `av2_entropy.h/.cpp` and `avg_cdf_bank.h/.cpp` | was identical | **consolidated** in `spec_tables/spec_math.h` | None now. |
| qctx slice thresholds/selection | was `qctx.h` and `cdf_continuity.h` | identical | **consolidated**: `cdf_continuity` aliases qctx | None now. |
| qlookup / quant step (§7.14.2) | `rc_tables.h` (`kAcQLookup[25]`, `kQlookupQ0To255[256]`), `wiener_qlookup.h` (same two arrays), `txv::kQuantStep = 528`, raw `528` in `basic_encoder.cpp`, `chroma_residual.h`, `cuda/transform.cu` | yes: both 256-entry tables equal; `qlookup(90) = 528`; all literals are 528 | both tables generated from `spec/tables/quant.json`; RC generator repaired | **Medium**: the `528` literals mean those paths are only correct at `base_q_idx = 90` (documented M0/M1 operating point), not drift. A single `spec_tables` table would require editing rate + filters generators and chroma/CUDA consumers — cross-workstream, deferred. |
| Scan tables (§5.20.7.30) | `tx_type_tables.h` (4/8/16/32), `basic_tables.h` (16/32), `lossless.cpp` local (4/8) | yes (all four sizes) | generated from `spec/tables/scan.json` (basic/lossless copies are embedded) | Low; types differ (`uint16_t` vs `int`). Consolidation needs generator changes in transform/entropy, deferred. |
| Transform kernels (§9.6) | `basic_tables.h`, `tx_variants/tx_tables.h`, `tx_variants/tx_tables_dev.cuh`, `cuda/transform.cu` device copies | yes (Dct4/8/16/32, 1360 values) | tx variants emits host+device from the PDF; M0/CUDA copies are legacy | Medium: four copies is the largest drift surface. End-to-end MD5 (`basic_64/360/1080p`) is the guard. Note: `src/codec/transform/variants/generate_tx_tables.py` is **broken after the move** (`REPO = parents[2]`, resolves to `src/`), so the two tx_variants headers are currently not regenerable — report to the transform workstream. |
| Bit I/O | `core/bitstream/bitwriter.*` (authoritative), `ibp::BitWriter` in `codec/intra/ibp/ibp.h/.cpp` | yes, same MSB-first semantics (ibp has no align/clear/take) | IBP is a standalone intra component; `TileBitWriter` is an entropy wrapper, not a duplicate | Low. Replace ibp's writer when the intra workstream wires IBP. |
| Error/status types | `av2::Status`, `dip::Status`, `drl::Status`, `merge::Status`, `warp::Status`, `mv_stack::Status`, `ddt::Status`, `cuda::Status`, dispatch-runtime `Status` | n/a — different domains | each module owns its status; no shared type | Low. A shared type would be a cross-cutting rewrite with silent mapping risk; not proposed. |
| `copy_cdf` | `av2::copy_cdf` (checked, flat), `qctx::copy_cdf` (constexpr template, nested), syntax `copy_array` | equivalent where used | kept | None worth changing. |

## 3. Gates (CPU-only, local)

```
cmake -S . -B build-int -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DAV2ENC_ENABLE_CUDA=OFF -DAV2ENC_ALLOW_CPU_ONLY=ON
cmake --build build-int -j3                 # clean
ctest --test-dir build-int                  # 10/10 passed
tools/refactor/verify_baseline.sh --check   # OK: 19 streams, ctest 10/10, feature gate 30
```

Module checks (standalone, not in ctest):

```
src/codec/entropy/qctx/run_tests.sh                    # generator --check + 337/3452 checks + 7 dav2d streams
WERROR=1 src/codec/entropy/av2/build_and_test.sh         # 116478 checks
WERROR=1 src/codec/entropy/avg_cdf/build_and_test.sh    # 555 checks + 3 dav2d tiled streams
WERROR=1 src/codec/entropy/cdf_continuity/build_and_test.sh  # 7979 checks + 2 dav2d streams
python3 src/pipeline/still/generate_basic_cdf.py --check
python3 src/pipeline/still/generate_still_tables.py --check
python3 src/codec/rate/rc/generate_rc_tables.py        # rewrites identically
python3 src/pipeline/video/generate_video_tables.py > /tmp/v.h  # byte-identical
```

No CUDA work was attempted locally. CUDA gates for the assistant: rebuild
`build_gpu_test.sh` on `sm_86` and run CUDA `ctest`; specifically
`src/codec/entropy/av2` header-only compile (it now includes
`spec_tables/spec_math.h`) and the transform-stream GPU test
(`src/codec/transform/stream/build_gpu_test.sh`) because the DCT copies are
unchanged but remain duplicated. No new pinned streams were added.

## 4. Remaining work (in priority order)

1. Report the broken `generate_tx_tables.py` path to the transform workstream
   (or fix it there); until then the tx_variants host/device tables cannot be
   regenerated.
2. Add a single `spec_tables` qlookup table once rate/filters consumers can be
   changed together; replace the `528` literals with `qlookup(base_q_idx)` in
   the same change (no earlier: those paths do not yet support other QPs).
3. When the M1/M2 writers next change, point `still_syntax.cpp` at the basic
   CDF namespace and delete the three duplicate intra-mode arrays from
   `still_cdf_tables.h`.
4. Replace `ibp::BitWriter` with `av2enc::BitWriter` when IBP is wired.
5. Update stale `src/encoder/...` references in the module READMEs
   (`qctx`, `avg_cdf`, `still`, `tx_variants`); they predate the move and
   currently send readers to nonexistent paths.

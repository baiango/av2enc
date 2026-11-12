# M3 GPU feature gate

CPU-vs-GPU gate for the M3 tools on the M1 still-picture path: **CDEF, CCSO
(band offset), RDOQ, intra TX type, rate-aware intra, full UV intra**.

```
Y4M ──av2enc --still      [features]──> cpu.ivf ──┐
                                                ├─ IVF frame-payload byte compare -> FAIL on any byte
Y4M ──av2enc --still-gpu  [features]──> gpu.ivf ──┤
        │                                       ├─ dav2d --strict 1 (both) -> Y/U/V plane compare -> FAIL on any sample
        │                                       └─ feature-effect probes (CPU + GPU)
        └─ Nsight Systems kernel count + stats/log staging check
```

A PASS means the GPU path produced **the same frame bytes as CPU M1 with the
same flags**, the stream decodes (`dav2d --strict 1`) to **zero differing Y/U/V
samples**, and Nsight Systems observed real CUDA kernels. It does **not** mean
the feature gives a rate win; this is a correctness gate.

A CPU fallback under `--still-gpu` would produce identical bytes and prove
nothing, so byte equality alone is never a pass. Rejected flags, CPU staging,
an ignored flag, or missing GPU evidence are **SKIP/FAIL, never PASS**.

## Files

| File | Purpose |
|---|---|
| `verify_m3_gpu.sh` | Single-case gate and `--matrix` mode. Feature mapping, CPU/GPU encodes, strict decode, byte/plane compare, feature-effect probes, determinism, memcheck/racecheck, Nsight attestation, PASS/FAIL/SKIP. |
| `ivf_compare.py` | IVF frame-payload byte comparison (no PSNR). |
| `plane_compare.py` | Bit-exact Y/U/V comparison of decoded Y4M (or `StillDUMP1`). |
| `remote.sh` | Syncs only `tools/m3_gpu` to the 3060 box and runs a case or the matrix. |
| `selftest.sh` | Mock-encoder self-test of the verdict logic (reject/staged/ignore/no-effect/unattested/pass/nondet/sanitizer). No CUDA needed. |
| `results/` | Local CPU-reference matrix output and pinned per-mode MD5s. |
| `RESULTS.md` | What has actually been run, with the exact commands and outcomes. |

## Verdicts

| Exit | Verdict | Meaning |
|---:|---|---|
| 0 | `PASS` | GPU attested by Nsight (default), feature changed both CPU and GPU bitstreams, frame bytes identical, both streams decode with `--strict 1`, Y/U/V planes identical. |
| 0 | `CPU-ONLY-PASS` | `--cpu-only`: CPU reference + strict decode only. **GPU not tested.** |
| 1 | `FAIL` | Byte/plane/decode difference, silent-ignore, GPU crash, nondeterminism, sanitizer error. |
| 2 | setup | Missing encoder/decoder/ffmpeg/input, bad arguments. |
| 3 | `SKIP` | Rejected flag, CPU staging, no-effect feature, no Nsight evidence, CPU-only build. **Never a pass.** |

`ALLOW_SKIP=1` maps SKIP to 0 for CI that only wants "no regression", while
still printing the verdict.

## Usage

```sh
# one M3 tool
tools/m3_gpu/verify_m3_gpu.sh --size 64x64 --features cdef

# a combination
tools/m3_gpu/verify_m3_gpu.sh --size 640x360 \
    --features cdef,ccso,rdoq,tx-type

# all six, determinism + memcheck
tools/m3_gpu/verify_m3_gpu.sh --size 1920x1080 --features all \
    --determinism --sanitize --keep

# CPU reference only (works on a non-CUDA host)
tools/m3_gpu/verify_m3_gpu.sh --cpu-only --size 640x360 --features all

# feature matrix: 18 modes x 64x64,640x360,1280x720,1920x1080
tools/m3_gpu/verify_m3_gpu.sh --matrix

# no-CUDA self-test of the verdict logic (mock encoder)
tools/m3_gpu/selftest.sh --encoder build-m1/av2enc

# subset
tools/m3_gpu/verify_m3_gpu.sh --matrix --sizes 64x64,640x360 \
    --modes baseline,cdef,ccso,rdoq,tx-type,rdo-intra,uv-full,all
```

From a laptop, using the prescribed tunnel:

```sh
AV2ENC_HOST='<user>@<host>' AV2ENC_PORT='<port>' \
    tools/m3_gpu/remote.sh verify --size 640x360 --features all

AV2ENC_HOST='<user>@<host>' AV2ENC_PORT='<port>' \
    tools/m3_gpu/remote.sh matrix --sizes 64x64,640x360,1280x720,1920x1080
```

`remote.sh` sends only `tools/m3_gpu` by default. `FULL_SYNC=1` uses
`tools/sync_remote.sh` when the encoder itself has to be rebuilt.

### Features

| `--features` name | Encoder flags | Notes |
|---|---|---|
| `cdef` | `--still-cdef` | 4 strength sets (default) |
| `cdef1` | `--still-cdef --still-cdef-strengths 1` | search-pruned single set |
| `ccso` | `--still-ccso` | band offset, default 64 bands |
| `ccso4` | `--still-ccso --still-ccso-bands 4` | 16 bands |
| `rdoq` | `--still-rdoq` | default sse_scale |
| `tx-type` | `--still-tx-type` | luma only |
| `rdo-intra` | `--still-rdo-intra` | implies full intra |
| `uv-full` | `--still-uv-full` | chroma directional search |
| `all` | all six | |

`--list-modes` prints every matrix mode (`baseline`, alone modes,
combinations, `all`).

### Matrix

18 modes × 4 native sizes = 72 cases by default:

```
baseline
cdef  cdef1  ccso  ccso4  rdoq  tx-type  rdo-intra  uv-full
cdef+ccso  cdef+rdoq  cdef+tx-type  rdoq+tx-type
cdef+ccso+rdoq  cdef+rdoq+tx-type  rdo-intra+uv-full
cdef+rdoq+tx-type+uv-full
all
```

Each case is an independent process, so one crash cannot poison the next.
Inputs are generated once per size as **native** testsrc2 Y4M (no upscaled
720p/1080p proxy). The matrix prints a progress line per case, every failure
with the exact byte offset / first sample, grouped SKIP reasons, and a
`-- feature readiness` table:

* `READY` — alone mode PASSes at every requested size.
* `NOT-READY` — with the observed class (`SKIP-REJECTED`, `IGNORED`,
  `BYTE-DIFF`, `SKIP-CPU-STAGED`, …).

`--matrix-budget SEC` (default 3600) stops starting new cases;
`--max-cases N` refuses an oversized matrix; `--timeout SEC` bounds each
command. Exit: all PASS → 0, any FAIL → 1, any SKIP → 3.

### Feature-effect probes (anti-vacuous-PASS)

Two extra checks run by default:

1. **CPU no-effect** — if `CPU(features) == CPU(no features)` byte-for-byte,
   the feature did not change the reference stream and the case is
   `SKIP-NO-EFFECT`. (Example: forcing a filter that makes no decision change
   at that size proves nothing.)
2. **GPU ignore probe** — the GPU is encoded with and without the feature
   flags. If the GPU bitstreams are identical while the CPU bitstreams differ,
   the GPU silently dropped the flag → `FAIL (IGNORED)`. This catches flags the
   CLI accepts but never plumbs into `GpuOptions` (see CCSO below).

`--no-noop-check` / `--no-ignore-probe` disable them (not recommended).

### Determinism

`--determinism` runs the GPU encode a second time (separate process, separate
IVF) and byte-compares payloads. A difference fails with the exact offset.
The second run must also be free of CPU-staging (and stats-attested when a
stats file is written).

### Compute Sanitizer

`--sanitize` replaces the normal GPU run with
`compute-sanitizer --tool memcheck --error-exitcode 99`; that run's IVF is the
one byte-gated and decoded.

* `ERROR SUMMARY: N errors`, `N > 0` → FAIL.
* `terminated before first instrumented API call` → SKIP (no CUDA work).
* no `ERROR SUMMARY` → FAIL (sanitizer did not complete).

Sanitizer output is not execution attestation. `--racecheck` adds
`--tool racecheck` at 64x64 only; any error **or warning** fails.

### Execution attestation

A PASS requires one of:

1. **Nsight Systems (default, required).** `nsys profile --trace=cuda` on the
   GPU command; `cuda_gpu_kern_sum` must report ≥1 kernel instance. The
   attestation run is redirected to `attest.ivf` so it cannot overwrite the
   gated artifact.
2. **Stats JSON** (`--attest stats`). `$AV2ENC_GPU_STATS` with
   `backend: gpu` or a positive kernel/launch count **and**
   `frame_bytes_from_cpu` not `true`. Use only where nsys is unavailable.
3. `--attest auto` uses stats when present, otherwise nsys.

The CLI currently does not write the stats file, so the CLI gate needs nsys.
`frame_bytes_from_cpu=1` (log) or `"frame_bytes_from_cpu": true` (stats) is
always `SKIP-CPU-STAGED`, even with identical bytes.

## Current GPU support (code-derived)

Read from the tree on 2026-11-13; docs may lag, the code and the gate output
win. This table is what the gate is expected to say, not a substitute for
running it on the CUDA box.

| Feature | CLI → `GpuOptions` | GPU implementation | Expected gate result |
|---|---|---|---|
| CDEF | `enable_cdef` plumbed in `src/main.cpp` | `still_cdef_filter` in `src/cuda/pipeline/still/still_gpu.cu` (§7.18) | `READY` or `BYTE-DIFF` (parity unverified without a device) |
| CCSO | `enable_ccso` / `ccso_bands_log2` / `ccso_edge` plumbed | `ccso_frame` in `src/cuda/pipeline/still/still_gpu.cu` | `READY` or `BYTE-DIFF` (parity unverified without a device) |
| Film grain | `enable_grain` plumbed; `--still-grain-adaptive` rejected by name | metadata-only (decoder-side synthesis), copied CPU `StillConfig` | not gated: the matrix has no `grain` mode |
| RDOQ | `rdoq` plumbed in `src/main.cpp` | rejected in `validate_options` | `SKIP-REJECTED` |
| TX type | `tx_type` plumbed in `src/main.cpp` | rejected in `validate_options` | `SKIP-REJECTED` |
| Rate-aware intra | `rdo_intra` plumbed in `src/main.cpp` | rejected in `validate_options` | `SKIP-REJECTED` |
| Full UV | `uv_full` plumbed in `src/main.cpp` | rejected in `gpu_m3_options_supported()` (chroma job lacks luma-mode inputs) | `SKIP-REJECTED` |

Non-default tuning flags are also not forwarded to the GPU: `--still-cdef-strengths`,
`--still-ccso-bands`, `--still-rdoq-scale`, `--still-tx-type-scale`, `--still-rdo-scale`.
If a feature is ported without forwarding its scale, the ignore probe and the
byte gate will catch the mismatch.

One latent plumbing inconsistency is hidden by the current rejections and must
be fixed before those features can pass:

* `main.cpp` sets GPU `luma_dc_only = !still_full_intra`, while the CPU path uses
  `!still_full_intra && !still_rdo_intra`. Once `--still-rdo-intra` is ported, the GPU
  would start from a DC-only luma search unless this is aligned.

## Limitations

* One frame is compared (M1/M3 still-picture scope).
* IVF timestamps are not part of the byte gate.
* `plane_compare.py` supports 8-bit 4:2:0 only.
* No GPU recon dump is exposed by the current CLI, so parity is checked on the
  decoded planes, not on `GpuResult::recon_*`.
* nsys attestation proves *some* CUDA kernel ran; it cannot prove those kernels
  produced the bytes. It catches the realistic accident (silent fallback that
  does no GPU work), not a deliberately deceptive shim.
* A matrix FAIL can come from the CPU reference. The gate always validates the
  CPU stream first and reports `CPU-REF-*` separately, so a CPU/geometry bug is
  not mistaken for GPU parity.
* The local host used for the CPU matrix has no CUDA; GPU verdicts are pending
  on the 3060 box (`RESULTS.md`).

## Hardware / build note

The target is Linux + CUDA, 8 GB Ampere `sm_86`. The dev box is a 12 GB 3060
proxy; see `docs/remote-env.md`. Do not claim 8 GB validation from the proxy.

# M3 GPU gate results

Snapshot: **2026-11-11**. Read `README.md` for the contract; this file records
what was actually executed.

## 1. Environment

| Item | Value |
|---|---|
| Host | macOS arm64 (Mac mini, Darwin 25.6.0) |
| CUDA | **not available** (`nvcc`, `nsys`, `compute-sanitizer` absent; no `build-cuda`) |
| Encoder | `build-m1/av2enc` (CPU-only reference build) |
| Decoder | `tools/bin/dav2d-macos` |
| Inputs | native `testsrc2` Y4M at each size (ffmpeg), not upscaled proxies |
| QP | 90 (M1 pins 90) |

Consequence: **CPU reference matrix is real; GPU verdicts are pending.** No row
below claims GPU execution on this host.

## 2. CPU reference matrix

Command:

```sh
tools/m3_gpu/verify_m3_gpu.sh --matrix \
    --encoder build-m1/av2enc --cpu-only \
    --matrix-budget 0 --workdir /tmp/m3_cpu_matrix
```

Result:

```
cases:    72 (18 modes x 4 sizes)
result:   71 CPU-ONLY-PASS, 0 FAIL, 1 SKIP
elapsed:  82s
```

Sizes: `64x64`, `640x360`, `1280x720`, `1920x1080`.
Modes: `baseline`, `cdef`, `cdef1`, `ccso`, `ccso4`, `rdoq`, `tx-type`,
`rdo-intra`, `uv-full`, `cdef+ccso`, `cdef+rdoq`, `cdef+tx-type`,
`rdoq+tx-type`, `cdef+ccso+rdoq`, `cdef+rdoq+tx-type`, `rdo-intra+uv-full`,
`cdef+rdoq+tx-type+uv-full`, `all`.

Every encoded CPU stream passed `dav2d --strict 1`. Every non-baseline mode
changed the CPU bitstream versus its no-feature baseline at every size, except
the one expected inert tuning case below.

The single SKIP is `cdef1` at 64x64: strengths 1 and 4 select the same CDEF
parameters on that content, so the two CPU bitstreams are identical
(`d6bb95f0…`). The gate refuses to call that a verified strength-parameter
pass (`SKIP-NO-EFFECT`); the `cdef1` mode is `PARTIAL`. This is intentional —
a GPU that dropped `--still-cdef-strengths` would otherwise look correct there.
At 640x360/720p/1080p the variants differ and `cdef1` proceeds normally.

Artifacts:

* `results/2026-11-11-cpu-matrix.tsv` — size, mode, class (72 rows).
* `results/2026-11-11-cpu-matrix-md5.tsv` — size, mode, IVF bytes, IVF MD5.
* `results/2026-11-11-cpu-matrix.log` — full matrix output.

### 64x64 (testsrc2, QP 90)

| Mode | Bytes | MD5 |
|---|---:|---|
| baseline | 2083 | `6e779725927dc3cf4fbe34922caf6655` |
| cdef | 2086 | `d6bb95f0f3bcd40c886ebd4d482e05ec` |
| cdef1 | 2086 | `d6bb95f0f3bcd40c886ebd4d482e05ec` |
| ccso | 2125 | `651c0c23cd01746c49ccc15a44f3c7ed` |
| ccso4 | 2094 | `333a7f3905bf393f091e683e1c149202` |
| rdoq | 1959 | `e81be47184fa393b4f199b58d8c1d7fa` |
| tx-type | 2093 | `542ebe5b54ed3003be48f766658160c9` |
| rdo-intra | 2024 | `fd433144704c892545c3218e17f61b76` |
| uv-full | 2049 | `c618bb2b3b088584f1b49355351ef5f3` |
| cdef+ccso | 2124 | `d575f61e47f375f0ec950ee454700d97` |
| cdef+rdoq+tx-type | 1981 | `8b62fde31ce7dc04f65e98279a711e10` |
| all | 1921 | `babee17a9b387166fc7b3a3e71146aa0` |

`cdef` and `cdef1` are identical at 64x64 because the strength search selects
the same set on this content; they diverge elsewhere (for example 720p:
`47340 B` vs `47289 B`). That is why the gate checks *bitstream change vs
baseline*, not "cdef1 differs from cdef".

### 1920x1080 (native)

| Mode | Bytes | MD5 |
|---|---:|---|
| baseline | 87972 | `c17a8cf285f699bf63b284ca22a3beed` |
| cdef | 88049 | `0ab7806558d124490ea83937595b30f3` |
| ccso | 88073 | `f1fbe974b173ce5b8e3a68bb29998e9d` |
| rdoq | 79925 | `f90fbc0b241c3facddc9ca4d185b9509` |
| tx-type | 88592 | `4cc76c8f194e0558dd7c16db7b016074` |
| uv-full | 80775 | `5487e5cc5ccb270e99e856924b40ce0b` |
| all | 71696 | `6b77f2f2d2ce60c905d8368b5af29daa` |

This confirms the CPU reference path exercises all six tools at native 1080p
with valid `dav2d --strict 1` streams. It says nothing about the GPU.

## 3. GPU gate on this host

Not run. There is no CUDA device or toolkit here, so the gate correctly
reports:

```sh
tools/m3_gpu/verify_m3_gpu.sh --encoder build-m1/av2enc \
    --size 64x64 --features cdef
# -> CLASS: SKIP-UNAVAILABLE
#    RESULT: SKIP (GPU path unavailable in this build/host)
#    exit 3
```

This is the required behaviour: a CPU-only build must never produce a GPU PASS.

## 4. Expected GPU outcome (code-derived, not measured)

From `src/main.cpp` and `src/cuda/m1/still_gpu.cu`:

| Feature | Expected class | Cause |
|---|---|---|
| CDEF | `SKIP-REJECTED` | `validate_options` rejects `enable_cdef` |
| RDOQ | `SKIP-REJECTED` | rejects `rdoq` |
| TX type | `SKIP-REJECTED` | rejects `tx_type` |
| Rate-aware intra | `SKIP-REJECTED` | rejects `rdo_intra` |
| CCSO | `FAIL (IGNORED)` | flag accepted by CLI, no `GpuOptions` field, silently dropped |
| Full UV | `READY` or `FAIL` | plumbed (`uv_full`) and implemented; parity unverified |

The matrix's `-- feature readiness` section is designed to print exactly this
split. Until it is run on the 3060, full UV is **not** claimed GPU-ready.

Latent issues to fix when porting (currently masked by the rejections):

* `main.cpp:590` sets GPU `luma_dc_only = !still_full_intra`, but the CPU path
  (`main.cpp:633`) uses `!still_full_intra && !still_rdo_intra`. A future
  `--still-rdo-intra` port would silently start from DC-only luma.
* `encode_still_gpu`'s internal CPU parity check (`to_cpu_options`,
  `still_gpu.cu:1000-1013`) does not carry CDEF/CCSO/RDOQ/TX-type/RDO. For CCSO,
  the GPU's own check therefore compares against a CPU encode *without* CCSO
  and can pass while the CLI CPU stream differs. The external flag/no-flag
  probe in `verify_m3_gpu.sh` is the check that catches this.

## 5. Harness self-tests (mock encoder)

To prove the gate cannot emit a false PASS, the classification paths are
driven by `selftest.sh` with a mock CLI and mock `nsys` /
`compute-sanitizer` (no CUDA needed):

```sh
tools/m3_gpu/selftest.sh --encoder build-m1/av2enc
# selftest: 11 passed, 0 failed
```

| Scenario | Observed exit | Class |
|---|---:|---|
| GPU rejects `--still-cdef` ("not ported") | 3 | `SKIP-REJECTED` |
| GPU writes CPU-reference bytes (`frame_bytes_from_cpu=1`) | 3 | `SKIP-CPU-STAGED` |
| GPU ignores `--still-ccso` (same bytes with/without flag) | 1 | `IGNORED` |
| Feature does not change CPU bytes | 3 | `SKIP-NO-EFFECT` |
| Identical bytes, nsys missing | 3 | `SKIP-UNATTESTED` |
| Identical bytes, nsys reports 7 kernels | 0 | `PASS` |
| `--attest stats`, `backend: gpu`, `frame_bytes_from_cpu: false` | 0 | `PASS` |
| `--attest stats`, `frame_bytes_from_cpu: true` | 3 | `SKIP-CPU-STAGED` |
| Second GPU encode differs (`--determinism`) | 1 | `NONDETERMINISTIC` |
| memcheck `ERROR SUMMARY: 3 errors` | 1 | `SANITIZER` |
| memcheck `ERROR SUMMARY: 0 errors` | 0 | `PASS` |

Matrix self-test (`baseline,cdef,ccso,uv-full` at 64x64/640x360, mock
rejecting CDEF and dropping CCSO) produced:

```
result:   4 PASS, 2 FAIL, 4 SKIP
GPU-READY FEATURES:     uv-full
NOT GPU-READY:          cdef (SKIP-REJECTED) ccso (IGNORED)
```

## 6. Reproduce on the CUDA box

```sh
# from the Mac
AV2ENC_HOST='<user>@<host>' AV2ENC_PORT='<port>' \
    tools/m3_gpu/remote.sh matrix \
        --sizes 64x64,640x360,1280x720,1920x1080

# determinism + memcheck at the small sizes
AV2ENC_HOST='<user>@<host>' AV2ENC_PORT='<port>' \
    tools/m3_gpu/remote.sh matrix \
        --sizes 64x64,640x360 --modes baseline,cdef,ccso,rdoq,tx-type,rdo-intra,uv-full,all \
        --determinism --sanitize

# on the box
tools/m3_gpu/verify_m3_gpu.sh --matrix --sizes 64x64,640x360,1280x720,1920x1080
```

The encoder must be rebuilt from the current tree (`build-cuda/av2enc`) and
must be linked to the same library as any local comparators. `remote.sh` only
syncs `tools/m3_gpu`; use `FULL_SYNC=1` when the encoder itself is stale.

## 7. Failing cases

* CPU reference: no parity/decoder failures. 71 `CPU-ONLY-PASS`, 1
  `SKIP-NO-EFFECT` (`cdef1` at 64x64, where strengths 1 and 4 coincide). The
  default matrix therefore exits 3 (unverified case) rather than 0; that is
  the strict verdict, not a failure.
* GPU: none measured yet because no CUDA host is attached to this session.
  The expected non-ready features are CDEF, CCSO, RDOQ, TX type and
  rate-aware intra; CCSO is expected to be `FAIL (IGNORED)` rather than a
  clean rejection because the CLI accepts a flag the GPU option struct does
  not carry.

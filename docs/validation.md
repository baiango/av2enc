# Validation gate — dav2d

Every av2enc output that is presented as an AV2 stream must decode with the
pinned **dav2d** build before it is accepted. This is a hard gate, not a smoke
test on the side.

## Decoder pin

| | |
|---|---|
| Version | `f9dafbc` (dav2d nightly, 2026-11-10) |
| Upstream | <https://code.videolan.org/videolan/dav2d> |
| Build | <https://github.com/EwoutH/AV2/releases/tag/dav2d-nightly-20261110-f9dafbc747> |
| Local macOS arm64 | `tools/bin/dav2d-macos` |
| Remote Linux x64 | `/workspace/av2enc/tools/bin/dav2d` |

The binaries are not committed. `tools/validate_dav2d.sh` searches, in order:

1. `$DAV2D`
2. `tools/bin/dav2d` (Linux/remote)
3. `tools/bin/dav2d-macos` (local macOS)

Do not modify `tools/bin/` or download a different decoder. If the encoder
moves to a newer bitstream snapshot, pin the matching dav2d commit and update
`tools/dav2d/VERSION.md` and this file together.

## Running the gate

```sh
# Known-good sanity check (decoder + validator + ffprobe)
tools/validate_dav2d.sh testdata/sample_480x270_130f.ivf

# Every real encoder output; pass the source for PSNR reporting
tools/validate_dav2d.sh out.ivf source.y4m
```

The input is a raw IVF file with FourCC `AV02`; no remuxing or wrapping step is
allowed. `out.ivf` must be passed to the validator exactly as written by the
encoder. The validator prints the decoder version, dimensions, pixel format,
decoded frame count, and (when a source is supplied) PSNR. It exits non-zero if
decoding fails or produces no output.

Useful environment variables:

| Variable | Meaning |
|---|---|
| `DAV2D` | Override the decoder path |
| `THREADS` | Decoder threads (`0` = auto) |
| `EXPECTED_MD5` | Fail when the decoded MD5 differs |
| `KEEP=1` | Keep the decoded Y4M and print its path |

On the remote box:

```sh
ssh -p <port> <user>@<host>
cd /workspace/av2enc
tools/validate_dav2d.sh out.ivf source.y4m
```

## CTest

`ctest --test-dir build --output-on-failure` includes `dav2d_smoke`. It runs
`tools/validate_dav2d.sh` on `testdata/sample_480x270_130f.ivf` when a dav2d
binary is present. If neither `tools/bin/dav2d` nor `tools/bin/dav2d-macos`
exists (and `DAV2D` is unset), the test exits with code 77 and CTest reports a
clean skip with an explanatory message.

The pinned sample decodes as:

```text
decoder: f9dafbc
width=480
height=270
pix_fmt=yuv420p
nb_read_frames=130
OK: dav2d decoded testdata/sample_480x270_130f.ivf
```

## Current encoder status

## Format gates

`src/formats/format_gate.sh` drives the opt-in optical-format still paths
through the product CLI and requires both a `dav2d --strict 1` decode and a
0-LSB encoder-reconstruction match, with pinned IVF/decoded MD5s in
`src/formats/format_goldens.txt`:

| Flag | Formats | base_q_idx |
|---|---|---|
| `--still --bit-depth 10 [--chroma ...]` | 10-bit 4:2:0 / 4:2:2 / 4:4:4 / 4:0:0 | 1..303 (qctx 0..3) |
| `--still --chroma 422\|444\|400` | 8-bit 4:2:2 / 4:4:4 / 4:0:0 | 90 |

The source Y4M must already be in the requested colorspace; the gate generates
real `C420p10`/`C422p10`/`C444p10`/`Cmono10`/`C422`/`C444`/`Cmono` files. A
missing decoder or fixture exits 77.

Every current encoder mode writes real AV2 syntax: Basic (the default), Still
(`--still`), Still GPU (`--still-gpu`), Video (`--video`), Lossless
(`--lossless`) and the still format paths (`--bit-depth 10`, `--chroma`).
`dav2d_smoke` validates the pinned sample and the validation harness; the
product gates (`tests/basic_encode_smoke.sh`, `tests/basic_edge_smoke.sh`, the
`tools/still_verify/` gates, and the `--video` / `--tiles` gates) encode through
the CLI and require `dav2d --strict 1`. There is no opt-out or "skip validation"
mode.

## Product gates in CTest

Wave 4 registers the product gates that were introduced during integration as
first-class CTest tests (`tests/gates/CMakeLists.txt`). The original ten tests
are unchanged; the default inventory is now 17 tests. Every gate:

- is invoked with an explicit `--bin $<TARGET_FILE:av2enc>`;
- uses the same decoder discovery as `tools/validate_dav2d.sh`
  (`$DAV2D`, then `tools/bin/dav2d`, then `tools/bin/dav2d-macos`);
- exits **77 only when a prerequisite is missing** (decoder, `python3`,
  gitignored `testdata/` fixture). Encoder, decode or reconstruction failures
  exit 1 and fail the test.

| CTest test | Script | Covers | Prerequisites |
|---|---|---|---|
| `gate_format` | `src/formats/format_gate.sh` | 10-bit 4:2:0 and 8-bit 4:2:2/4:4:4/4:0:0 stills; pinned IVF/decoded MD5s; recon exactness | dav2d, python3 |
| `gate_intra` | `tools/still_verify/intra_gate.sh` | IBP, MRL, CfL variants, DIP, MHCCP, default-off baseline regressions | dav2d, `still_recon_dump`, python3 |
| `gate_loop_restoration` | `tools/still_verify/loop_restoration_gate.sh` | Wiener, PC-Wiener, chroma LR, GDF, CCSO-edge, film grain; 1080p cases when `testdata/` exists | dav2d, `gate_still_dump`, python3; `testdata/` for HD |
| `gate_tiles` | `src/pipeline/tiles/stream/gate_cli_tiles.sh` | `--tiles` layouts, 1x1 byte-identity, default-stream pin, rejections | dav2d, python3; `testdata/` for 1080p |
| `gate_rc` | `src/pipeline/video/post/gate_cli_rc.sh` | `--rc cqp/crf/vbr`, default 720p pin, CRF monotonicity, VBR determinism | dav2d, python3, `testdata/` 720p |
| `gate_host_plan` | `src/pipeline/dispatch/gate_host_plan.sh` | `--gpu-plan` assignments, dry-run writes nothing, rejections | 360p fixture and `testdata/` 1080p (no decoder) |
| `gate_video_cli` | `src/pipeline/video/video_cli_gate.sh` | `--video` multi-frame decode and sample-exact reconstruction compare | dav2d, python3, `testdata/` 720p |

Skip behavior is deliberately granular:

- no decoder or `python3` → the whole gate skips (77);
- `gate_intra` without `still_recon_dump` → the runner reports a skip without
  invoking the gate; `gate_loop_restoration` still runs its pinned-MD5/decode
  checks when `gate_still_dump` is absent (recon exactness is then not checked);
- missing `testdata/` → `gate_rc`, `gate_host_plan` and `gate_video_cli` skip
  (77); `gate_tiles` and `gate_loop_restoration` still run their committed-fixture
  cases and skip only the 1080p part.

The fixtures under `tests/data/` are committed; `testdata/` and `tools/bin/` are
gitignored symlinks. `AV2ENC_TESTDATA=/path` overrides the `testdata/` root for
gate runs without touching the symlink.

Labels are `gate`, plus `gate-hd` (tiles, loop restoration) and `gate-video`
(rate control, video CLI). For example, `ctest -R '^gate_'` runs only the
product gates, and `ctest -LE gate-video` skips the 720p video gates.

## Gate runner

`tools/run_gates.sh` runs the CTest unit tests and every product gate in one
pass and prints a summary. It configures and builds nothing, so it works with a
CPU reference build and with a CUDA build:

```sh
# CPU reference build
tools/run_gates.sh --bin build-int/av2enc

# CUDA build on the 3060 box
tools/run_gates.sh --bin build-cuda/av2enc --jobs 8

# Fast pass: skip the 1080p/720p video gates and the 1080p cases
tools/run_gates.sh --bin build-int/av2enc --quick

# Gates only (no CTest), streaming each gate's output
tools/run_gates.sh --bin build-int/av2enc --no-ctest --verbose
```

| Option | Meaning |
|---|---|
| `--bin PATH` | av2enc binary (default: first of `build-int`, `build`, `build-cuda`, `build-gpu`, `refactor/build/cpu`) |
| `--build-dir DIR` | CTest build directory (default: the binary's directory) |
| `--dav2d PATH` | pinned decoder; an explicit bad path is an error |
| `--quick` | skip `gate_rc` and `gate_video_cli`, and pass `--quick` to the tiles/loop-restoration gates |
| `--no-ctest` | run only the product gates |
| `--jobs N` | CTest parallelism |
| `--verbose` | stream gate output and keep the log directory |

CTest runs with `-E '^gate_'` so the gates are not executed twice; the runner
invokes each gate itself to apply `--quick`, the explicit decoder path and the
per-gate summary. Exit status is 0 when every selected test passes or skips
(77), 1 on any real failure, 2 on a setup error.

## Acceptance rule

**No encoder change is accepted if dav2d fails.** An output that does not
decode, decodes the wrong number of frames, or has the wrong dimensions/pixel
format is a failed change. Bit-exact reconstruction against the decoder is the
next gate after decode correctness.

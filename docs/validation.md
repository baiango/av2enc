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

## Current scaffold status

The M0 CLI currently writes explicitly-marked placeholder IVF payloads; no AV2
syntax has been implemented yet, so that payload is container-valid but not
decodable. `dav2d_smoke` validates the pinned sample and the validation harness,
not the placeholder payload. Once the bitstream path emits real AV2 frames,
every encoder output must pass the gate; there is no opt-out or "skip
validation" mode.

## Acceptance rule

**No encoder change is accepted if dav2d fails.** An output that does not
decode, decodes the wrong number of frames, or has the wrong dimensions/pixel
format is a failed change. Bit-exact reconstruction against the decoder is the
next gate after decode correctness.

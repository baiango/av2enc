# dav2d validation decoder

Pinned decoder for the encoder gate:

- Binary: `f9dafbc` (dav2d nightly, 2026-11-10)
- Source: <https://code.videolan.org/videolan/dav2d>
- Build: <https://github.com/EwoutH/AV2/releases/tag/dav2d-nightly-20261110-f9dafbc747>
- Linux x64 binary on the Vast.ai box: `/workspace/av2enc/tools/bin/dav2d`
- macOS arm64 binary: `tools/bin/dav2d-macos` (local only)

The binary is not committed. `tools/validate_dav2d.sh` finds either name.

Every encoder output must pass:

```sh
tools/validate_dav2d.sh out.ivf [source.y4m]
```

For a regression hash:

```sh
EXPECTED_MD5=<hash> tools/validate_dav2d.sh out.ivf
```

AV2 is still moving. If the encoder targets a newer/older bitstream snapshot,
pin the matching dav2d commit and record it here.

#!/usr/bin/env python3
"""Patch a build-directory copy of src/pipeline/video/video_headers.cpp for the
transform-partition gate.

Two edits:
  1. include the gate hook,
  2. write tx_mode_select = 1 for non-key frames when the gate enables it.
     §5.18.8.1: tx_mode_select is a per-frame field; the keyframe keeps
     TX_MODE_LARGEST so its intra transforms need no partition symbols.

The repository source is never modified.  Anchors are asserted unique.
"""

from __future__ import annotations

import pathlib
import sys

INCLUDE_ANCHOR = '#include "core/bitstream/basic_headers.h"\n'
INCLUDE_ADD = (
    '#include "core/bitstream/basic_headers.h"\n'
    '#include "codec/transform/tx_partition/gate/tx_partition_gate_hook.h"\n'
)

TX_OLD = "  writer.put_bit(false);   // tx_mode_select = 0 -> TX_MODE_LARGEST\n"
TX_NEW = (
    "  // Gate build: §5.18.8.1 tx_mode_select is per frame.  The keyframe\n"
    "  // stays TX_MODE_LARGEST; the P frame enables TX_MODE_SELECT.\n"
    "  writer.put_bit(\n"
    "      !key &&\n"
    "      txpartition::gate::tx_mode_select_enabled());\n"
)


def replace_once(text: str, old: str, new: str, what: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{what}: anchor occurs {count} times, expected 1")
    return text.replace(old, new, 1)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_video_headers.py <video_headers.cpp>")
    path = pathlib.Path(sys.argv[1])
    text = path.read_text()
    text = replace_once(text, INCLUDE_ANCHOR, INCLUDE_ADD, "include")
    text = replace_once(text, TX_OLD, TX_NEW, "tx_mode_select")
    path.write_text(text)
    print(f"patched {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

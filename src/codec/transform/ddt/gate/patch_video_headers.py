#!/usr/bin/env python3
"""Rewrite a build-directory copy of video_headers.cpp for the DDT gate.

The only change is the §5.4.8 `enable_inter_ddt` bit: the core hard-codes 0,
the gate copy asks ddt::gate::enable_inter_ddt() (DDT_GATE_ENABLE=1).  This
lets one binary emit both the flag-0 stream (for the splice) and the flag-1
stream (for the decoder gate).

Usage:
  python3 patch_video_headers.py <src/encoder/m2/video_headers.cpp> <out.cpp>
"""
import sys

INCLUDE_ANCHOR = '#include "core/bitstream/basic_headers.h"\n'
INCLUDE_ADD = '#include "codec/transform/ddt/gate/ddt_gate_hook.h"\n'
OLD = "  writer.put_bit(false);  // enable_inter_ddt\n"
NEW = ("  // DDT gate patch: §5.4.8 enable_inter_ddt.\n"
       "  writer.put_bit(av2enc::video::ddt::gate::enable_inter_ddt());\n")


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    source, target = argv
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    if INCLUDE_ANCHOR not in text:
        raise SystemExit(f"anchor not found: {INCLUDE_ANCHOR!r}")
    text = text.replace(INCLUDE_ANCHOR, INCLUDE_ANCHOR + INCLUDE_ADD, 1)
    if OLD not in text:
        raise SystemExit("anchor not found: enable_inter_ddt bit")
    text = text.replace(OLD, NEW, 1)
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

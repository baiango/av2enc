#!/usr/bin/env python3
"""Rewrite a build-directory copy of m2_headers.cpp for the DDT gate.

The only change is the §5.4.8 `enable_inter_ddt` bit: the core hard-codes 0,
the gate copy asks ddt::gate::enable_inter_ddt() (DDT_GATE_ENABLE=1).  This
lets one binary emit both the flag-0 stream (for the splice) and the flag-1
stream (for the decoder gate).

Usage:
  python3 patch_m2_headers.py <src/encoder/m2/m2_headers.cpp> <out.cpp>
"""
import sys

INCLUDE_ANCHOR = '#include "bitstream/m0_headers.h"\n'
INCLUDE_ADD = '#include "encoder/m2/ddt/gate/ddt_gate_hook.h"\n'
OLD = "  writer.put_bit(false);  // enable_inter_ddt\n"
NEW = ("  // DDT gate patch: §5.4.8 enable_inter_ddt.\n"
       "  writer.put_bit(av2enc::m2::ddt::gate::enable_inter_ddt());\n")


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

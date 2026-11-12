#!/usr/bin/env python3
"""Rewrite a build-directory copy of inter_residual.cpp for the DDT gate.

Only one call site changes: the reconstruction inverse transform is routed
through ddt::gate::inverse_transform(), which uses the normative DDT kernels.
The forward transform, quantiser and decisions stay stock M2/residual code.

Usage:
  python3 patch_inter_residual.py <src/codec/transform/inter_residual/inter_residual.cpp> <out.cpp>
"""
import sys

INCLUDE_ANCHOR = '#include "codec/transform/inter_residual/inter_residual.h"\n'
INCLUDE_ADD = '#include "codec/transform/ddt/gate/ddt_gate_hook.h"\n'

OLD = """  txv::inverse_transform(out->levels.data(), params, dequant.data(),
                         recon_residual.data());
"""
NEW = """  // DDT gate patch: reconstruction uses the DDT-aware inverse.
  av2enc::video::ddt::gate::inverse_transform(out->levels.data(), params,
                                           dequant.data(),
                                           recon_residual.data());
"""


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
        raise SystemExit("anchor not found: txv::inverse_transform call")
    text = text.replace(OLD, NEW, 1)
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

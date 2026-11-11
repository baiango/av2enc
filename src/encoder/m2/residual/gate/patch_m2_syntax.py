#!/usr/bin/env python3
"""Produce the gate copy of src/encoder/m2/m2_syntax.cpp.

The component owns only src/encoder/m2/residual/.  To prove the inter-residual
syntax end-to-end without editing the M2 core, this script rewrites a *copy* of
the M2 syntax writer in the build directory so that `write_inter_tx_type()`
delegates the `inter_tx_type` symbols to this component instead of hard-coding
DCT_DCT.  Everything else is byte-for-byte the M2 writer.

The same three edits are the integration call sites documented in
docs/INTEGRATION.md; the script fails loudly if the anchors move.

Usage:
  python3 gate/patch_m2_syntax.py <input m2_syntax.cpp> <output.cpp>
  python3 gate/patch_m2_syntax.py --diff <output.cpp> > m2_syntax_inter.patch
"""
import sys

INCLUDE_ANCHOR = '#include "encoder/m2/merge/merge.h"\n'

HELPERS = '''
// ---------------------------------------------------------------------------
// Gate patch (src/encoder/m2/residual): inter transform-type signalling.
// Not part of the M2 core.  An integration would either replace the
// DCT_DCT-only writer below with residual::write_inter_tx_type() or pass the
// per-leaf TX type through BlockSyntax.
// ---------------------------------------------------------------------------
#include "encoder/m2/residual/inter_coeff_syntax.h"

namespace av2enc::m2::syntax {
namespace {

std::vector<std::uint8_t> g_gate_inter_tx_types;

bool gate_encode_symbol(void* context, std::uint16_t* cdf, int n, int symbol) {
  auto* encoder = static_cast<av2::Encoder*>(context);
  return encoder->encode_symbol(cdf, n, symbol) == av2::Status::kOk;
}

bool gate_encode_bool(void* context, bool bit) {
  auto* encoder = static_cast<av2::Encoder*>(context);
  return encoder->encode_bool(bit) == av2::Status::kOk;
}

}  // namespace

// Coding-order TX_TYPE list for the gate (one entry per non-skip inter leaf).
void m2_residual_gate_set_inter_tx_types(const std::uint8_t* types,
                                         int count) {
  g_gate_inter_tx_types.assign(types, types + (count > 0 ? count : 0));
}

}  // namespace av2enc::m2::syntax
'''

OLD_FUNCTION_START = (
    "  bool write_inter_tx_type(int tx_size, int eob, int coded_w, "
    "int coded_h) {\n")
NEW_FUNCTION = '''  bool write_inter_tx_type(int tx_size, int eob, int coded_w, int coded_h) {
    (void)coded_w;
    (void)coded_h;
    // Gate patch: take the decided type from the coding-order list instead of
    // always signalling DCT_DCT.  residual_cdfs_ is initialised next to
    // init_cdfs(cdfs_) in the constructor.
    int type = 0;  // txv::TxType::kDctDct
    if (gate_tx_cursor_ < g_gate_inter_tx_types.size()) {
      type = g_gate_inter_tx_types[gate_tx_cursor_];
    }
    ++gate_tx_cursor_;
    av2enc::m2::residual::SymbolSink sink;
    sink.context = &enc_;
    sink.encode_symbol = &gate_encode_symbol;
    sink.encode_bool = &gate_encode_bool;
    return av2enc::m2::residual::write_inter_tx_type(
        sink, residual_cdfs_, tx_size, type, eob, error_);
  }
'''

CDFS_ANCHOR = "  CdfState cdfs_{};\n"
CDFS_MEMBERS = '''  CdfState cdfs_{};
  // Gate patch: inter TX-type CDF state for residual::write_inter_tx_type().
  av2enc::m2::residual::InterResidualCdfs residual_cdfs_{};
  // Gate patch: coding-order cursor into the TX_TYPE list above.
  std::size_t gate_tx_cursor_ = 0;
'''

INIT_ANCHOR = "    init_cdfs(cdfs_);\n"
INIT_PATCH = '''    init_cdfs(cdfs_);
    av2enc::m2::residual::init_inter_residual_cdfs(residual_cdfs_);
'''


def patch(text):
    if INCLUDE_ANCHOR not in text:
        raise SystemExit(f"anchor not found: {INCLUDE_ANCHOR!r}")
    text = text.replace(INCLUDE_ANCHOR, INCLUDE_ANCHOR + HELPERS, 1)

    start = text.find(OLD_FUNCTION_START)
    if start < 0:
        raise SystemExit("anchor not found: write_inter_tx_type signature")
    # The function ends right before code_tx().
    end_marker = "\n  bool code_tx("
    end = text.find(end_marker, start)
    if end < 0:
        raise SystemExit("anchor not found: code_tx() after write_inter_tx_type")
    text = text[:start] + NEW_FUNCTION + text[end + 1:]

    if CDFS_ANCHOR not in text:
        raise SystemExit(f"anchor not found: {CDFS_ANCHOR!r}")
    text = text.replace(CDFS_ANCHOR, CDFS_MEMBERS, 1)

    if INIT_ANCHOR not in text:
        raise SystemExit(f"anchor not found: {INIT_ANCHOR!r}")
    text = text.replace(INIT_ANCHOR, INIT_PATCH, 1)
    return text


def main(argv):
    if len(argv) == 2 and argv[0] == "--diff":
        with open(argv[1], encoding="utf-8") as handle:
            sys.stdout.write(handle.read())
        return 0
    if len(argv) != 2:
        print(__doc__)
        return 2
    source, target = argv
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    patched = patch(text)
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(patched)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

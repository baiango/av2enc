#!/usr/bin/env python3
"""Produce the DDT gate copy of src/encoder/m2/video_syntax.cpp.

This is the same idea as src/codec/transform/inter_residual/gate/patch_video_syntax.py, but
kept in this component so the DDT gate does not depend on that script's
anchors.  It rewrites `write_inter_tx_type()` so the gate can signal non-DCT
inter TX types; everything else is byte-for-byte the M2 writer.

The three edits are the integration call sites documented in README.md.  The
script fails loudly if the anchors move.

Usage:
  python3 gate/patch_video_syntax.py <video_syntax.cpp> <out.cpp>
"""
import sys

INCLUDE_ANCHOR = '#include "codec/inter/merge/merge.h"\n'

HELPERS = '''
// ---------------------------------------------------------------------------
// DDT gate patch (src/codec/transform/ddt).  Not part of the M2 core.
// ---------------------------------------------------------------------------
#include "codec/transform/inter_residual/inter_coeff_syntax.h"

namespace av2enc::video::syntax {
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

void video_residual_gate_set_inter_tx_types(const std::uint8_t* types, int count) {
  g_gate_inter_tx_types.assign(types, types + (count > 0 ? count : 0));
}

}  // namespace av2enc::video::syntax
'''

OLD_FUNCTION_START = (
    "  bool write_inter_tx_type(int tx_size, int eob, int coded_w, "
    "int coded_h) {\n")
NEW_FUNCTION = '''  bool write_inter_tx_type(int tx_size, int eob, int coded_w, int coded_h) {
    (void)coded_w;
    (void)coded_h;
    // DDT gate patch: take the decided type from the coding-order list instead
    // of always signalling DCT_DCT.
    int type = 0;  // txv::TxType::kDctDct
    if (gate_tx_cursor_ < g_gate_inter_tx_types.size()) {
      type = g_gate_inter_tx_types[gate_tx_cursor_];
    }
    ++gate_tx_cursor_;
    av2enc::video::residual::SymbolSink sink;
    sink.context = &enc_;
    sink.encode_symbol = &gate_encode_symbol;
    sink.encode_bool = &gate_encode_bool;
    return av2enc::video::residual::write_inter_tx_type(
        sink, residual_cdfs_, tx_size, type, eob, error_);
  }
'''

CDFS_ANCHOR = "  CdfState cdfs_{};\n"
CDFS_MEMBERS = '''  CdfState cdfs_{};
  // DDT gate patch: inter TX-type CDF state for residual::write_inter_tx_type().
  av2enc::video::residual::InterResidualCdfs residual_cdfs_{};
  // DDT gate patch: coding-order cursor into the TX_TYPE list above.
  std::size_t gate_tx_cursor_ = 0;
'''

INIT_ANCHOR = """    init_cdfs(cdfs_,
              qctx::slice_for_base_q_idx(static_cast<int>(config_.base_q_idx)));
"""
INIT_PATCH = INIT_ANCHOR + """    av2enc::video::residual::init_inter_residual_cdfs(residual_cdfs_);
"""


def patch(text):
    if INCLUDE_ANCHOR not in text:
        raise SystemExit(f"anchor not found: {INCLUDE_ANCHOR!r}")
    text = text.replace(INCLUDE_ANCHOR, INCLUDE_ANCHOR + HELPERS, 1)

    start = text.find(OLD_FUNCTION_START)
    if start < 0:
        raise SystemExit("anchor not found: write_inter_tx_type signature")
    end_marker = "\n  bool code_tx("
    end = text.find(end_marker, start)
    if end < 0:
        raise SystemExit("anchor not found: code_tx() after write_inter_tx_type")
    text = text[:start] + NEW_FUNCTION + text[end + 1:]

    if CDFS_ANCHOR not in text:
        raise SystemExit(f"anchor not found: {CDFS_ANCHOR!r}")
    text = text.replace(CDFS_ANCHOR, CDFS_MEMBERS, 1)

    if INIT_ANCHOR not in text:
        raise SystemExit("anchor not found: init_cdfs(cdfs_, qctx...)")
    text = text.replace(INIT_ANCHOR, INIT_PATCH, 1)
    return text


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    source, target = argv
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(patch(text))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

#!/usr/bin/env python3
"""Patch a build-directory copy of src/pipeline/video/video_syntax.cpp for the
transform-partition gate.  The repository source is never modified.

The patch adds exactly three things to the Writer class:
  1. an include for the gate hook,
  2. a nested SymbolSink adapter plus txgate_code_luma(), which emits the
     §5.20.6.3 partition syntax and codes one code_tx() call per transform
     block, and
  3. one call-site swap in write_inter_residual(): the luma code_tx() call
     becomes txgate_code_luma().

Every anchor is asserted to occur exactly once.  Run as:
  patch_video_syntax.py <copy-of-video_syntax.cpp>
"""

from __future__ import annotations

import pathlib
import sys

INCLUDE_ANCHOR = '#include "codec/inter/merge/merge.h"\n'
INCLUDE_ADD = (
    '#include "codec/inter/merge/merge.h"\n'
    '#include "codec/transform/tx_partition/gate/tx_partition_gate_hook.h"\n'
)

MEMBER_ANCHOR = (
    "  // Inter residual syntax: transform coefficients for a non-skip inter "
    "leaf.\n"
)

MEMBER_CODE = r'''  // --- tx_partition gate hook (build-directory patched copy only) ---------
  struct TxGateSink final : txpartition::SymbolSink {
    Writer* writer = nullptr;
    bool symbol(std::uint16_t* cdf, int n, int value,
                std::string& error) override {
      if (!writer->sym(cdf, n, value)) {
        error = writer->error_;
        return false;
      }
      return true;
    }
  };

  bool txgate_code_luma(const BlockSyntax& node, std::size_t level_base,
                        int w, int h, std::size_t levels_end,
                        std::string& error) {
    (void)level_base;
    const auto* plan = txpartition::gate::active_split_plan();
    if (plan == nullptr || !plan->active ||
        plan->mi_row != node.mi_row || plan->mi_col != node.mi_col ||
        plan->mi_size != node.bsize) {
      return code_tx(node, 0, level_base, w, h, levels_end, true, error);
    }
    auto* cdfs = txpartition::gate::tx_partition_cdfs();
    txpartition::BlockTxContext ctx;
    ctx.mi_size = node.bsize;
    ctx.is_inter = true;
    ctx.fsc_mode = false;
    ctx.reduced_tx_part_set = false;
    TxGateSink sink;
    sink.writer = this;
    txpartition::SplitGeometry geom;
    if (!txpartition::write_tx_partition(
            sink, *cdfs, ctx,
            static_cast<txpartition::TxPartition>(plan->partition), &geom,
            error)) {
      return false;
    }
    if (geom.count != plan->count) {
      error = "gate split plan does not match the signalled partition";
      return false;
    }
    for (int i = 0; i < plan->count; ++i) {
      const auto& sub = plan->subs[i];
      BlockSyntax sub_node = node;
      sub_node.mi_row = sub.mi_row;
      sub_node.mi_col = sub.mi_col;
      sub_node.tx_size[0] = sub.tx_size;
      sub_node.eob[0] = sub.eob;
      sub_node.levels_offset = sub.levels_offset;
      if (!code_tx(sub_node, 0, sub.levels_offset, w, h, levels_end, true,
                   error)) {
        return false;
      }
    }
    return true;
  }

'''

LUMA_OLD = """    std::size_t level_base = node.levels_offset;
    if (!code_tx(node, 0, level_base, w, h, levels_end, true, error)) {
      return false;
    }
    level_base += node.eob[0];
"""

LUMA_NEW = """    std::size_t level_base = node.levels_offset;
    if (!txgate_code_luma(node, level_base, w, h, levels_end, error)) {
      return false;
    }
    level_base += node.eob[0];
"""


def replace_once(text: str, old: str, new: str, what: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{what}: anchor occurs {count} times, expected 1")
    return text.replace(old, new, 1)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_video_syntax.py <video_syntax.cpp>")
    path = pathlib.Path(sys.argv[1])
    text = path.read_text()
    text = replace_once(text, INCLUDE_ANCHOR, INCLUDE_ADD, "include")
    text = replace_once(text, MEMBER_ANCHOR, MEMBER_CODE + MEMBER_ANCHOR,
                        "member insertion")
    text = replace_once(text, LUMA_OLD, LUMA_NEW, "luma call site")
    path.write_text(text)
    print(f"patched {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Build-directory patch: add the §5.20.5.5 intra `fsc_mode` symbol.

The core M1 syntax writer never sets `enable_idtx_intra`, so it never codes
`fsc_mode`.  The moment a sequence header advertises `enable_idtx_intra = 1`,
`allow_fsc_intra()` becomes true for every intra luma block with
`Block_Width/Height <= FSC_MAX` and the decoder reads one `fsc_mode` symbol
there (§5.20.5.5).  Without it every tile is corrupt — this is why the pinned
dav2d rejected `enable_idtx_intra = 1` streams (docs/m1-features.md).

This patch is applied to a *copy* of `src/pipeline/still/still_syntax.cpp` in the
build directory, the same technique the DDT gate uses.  The core source is not
modified.  The symbol is always coded as 0 (FSC off); the coefficient path is
therefore unchanged.  The FSC=1 path (`coeff_base_bob`, `coeff_base_idtx`)
remains out of scope.

Usage: patch_still_syntax.py SRC DST
"""
from __future__ import annotations

import sys

FSC_GROUPS = (
    "0, 1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 6, 6, 6, 6, 6, "
    "6, 6, 6, 3, 3, 4, 4, 6, 6, 4, 4, 6, 6"
)

# Verbatim Default_Fsc_Mode_Cdf from spec/tables/default_cdf.json (§9.3),
# indented for insertion into the Writer's anonymous namespace.
FSC_CDF = """inline constexpr std::uint16_t kIdtxFscModeCdf[4][6][3] = {\n  {\n    {30503, 3, 0},\n    {31244, 3, 0},\n    {32254, 78, 0},\n    {32324, 93, 0},\n    {32582, 93, 0},\n    {32691, 93, 0}\n  },\n  {\n    {27437, 0, 0},\n    {27242, 90, 0},\n    {28040, 76, 0},\n    {27589, 76, 0},\n    {27234, 7, 0},\n    {23583, 62, 0}\n  },\n  {\n    {26068, 75, 0},\n    {22635, 75, 0},\n    {22069, 6, 0},\n    {19218, 32, 0},\n    {13701, 31, 0},\n    {4636, 38, 0}\n  },\n  {\n    {16384, 0, 0},\n    {16384, 0, 0},\n    {32016, 15, 0},\n    {32403, 93, 0},\n    {32583, 15, 0},\n    {32683, 76, 0}\n  }\n};"""


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    src, dst = sys.argv[1], sys.argv[2]
    text = open(src, encoding="utf-8").read()

    def sub(old: str, new: str) -> None:
        nonlocal text
        if old not in text:
            raise SystemExit(f"patch anchor not found: {old[:70]!r}")
        text = text.replace(old, new, 1)

    # 1. §9.3 FSC-mode CDF (copied from the spec table, so the patch does not
    # depend on another component's generated header) and the §9.2 block groups.
    sub(
        "constexpr int kIntraModesCount = 13;",
        "constexpr int kIntraModesCount = 13;\n"
        "// §9.3 Default_Fsc_Mode_Cdf[4][6][3].\n"
        f"{FSC_CDF}\n"
        "// §9.2 FSC block-size groups (kFscBsizeGroups).\n"
        f"constexpr int kIdtxFscGroups[29] = {{{FSC_GROUPS}}};",
    )

    # 2. CDF state.
    sub(
        "  std::uint16_t is_long_side_dct[2][3];",
        "  std::uint16_t is_long_side_dct[2][3];\n"
        "  std::uint16_t fsc_mode[4][6][3];",
    )
    sub(
        "  copy_array(c.is_long_side_dct, kIsLongSideDctCdf);",
        "  copy_array(c.is_long_side_dct, kIsLongSideDctCdf);\n"
        "  copy_array(c.fsc_mode, kIdtxFscModeCdf);",
    )

    # 3. Grid allocation.
    sub(
        "    mi_cols_ = partition::mi_cols(geometry_);\n"
        "    mi_rows_ = partition::mi_rows(geometry_);",
        "    mi_cols_ = partition::mi_cols(geometry_);\n"
        "    mi_rows_ = partition::mi_rows(geometry_);\n"
        "    fsc_modes_.assign(\n"
        "        static_cast<std::size_t>(mi_cols_) *\n"
        "            static_cast<std::size_t>(mi_rows_),\n"
        "        0);",
    )

    # 4. Emit fsc_mode after y_mode, before the chroma mode.
    sub(
        "    if (!code_y_mode(node)) {\n"
        "      error = error_;\n"
        "      return false;\n"
        "    }\n"
        "    if (!code_uv_mode(",
        "    if (!code_y_mode(node)) {\n"
        "      error = error_;\n"
        "      return false;\n"
        "    }\n"
        "    // [idtx_intra gate patch] §5.20.5.5: when enable_idtx_intra is set,\n"
        "    // allow_fsc_intra() codes fsc_mode for every intra luma block with\n"
        "    // Block_Width/Height <= FSC_MAX.  Always 0 here: FSC stays off and\n"
        "    // the coefficient path is unchanged.\n"
        "    if (config_.enable_idtx_intra && w <= 32 && h <= 32) {\n"
        "      const int fctx = fsc_context_for(r, c, w4, h4);\n"
        "      const int group = kIdtxFscGroups[static_cast<int>(size)];\n"
        "      if (!sym(&cdfs_.fsc_mode[fctx][group][0], 2, 0)) return false;\n"
        "      for (int yy = 0; yy < h4 && r + yy < mi_rows_; ++yy) {\n"
        "        for (int xx = 0; xx < w4 && c + xx < mi_cols_; ++xx) {\n"
        "          fsc_modes_[static_cast<std::size_t>(r + yy) *\n"
        "                         static_cast<std::size_t>(mi_cols_) +\n"
        "                     static_cast<std::size_t>(c + xx)] = 0;\n"
        "        }\n"
        "      }\n"
        "    }\n"
        "    if (!code_uv_mode(",
    )

    # 5. Context helper + grid member.
    sub(
        "  int eob_u_ = 0;\n  bool trace_ = false;",
        "  int eob_u_ = 0;\n"
        "  bool trace_ = false;\n"
        "  std::vector<std::uint8_t> fsc_modes_;\n"
        "  // §5.20.5.5 fsc_mode context: the first two available neighbours in\n"
        "  // this order, within the same 64x64 superblock.\n"
        "  int fsc_context_for(int r, int c, int w4, int h4) const {\n"
        "    int nnum = 0;\n"
        "    int sum = 0;\n"
        "    auto add = [&](int nr, int nc) {\n"
        "      if (nnum >= 2) return;\n"
        "      if (nr < 0 || nc < 0 || nr >= mi_rows_ || nc >= mi_cols_) return;\n"
        "      if ((r >> 4) != (nr >> 4)) return;\n"
        "      sum += fsc_modes_[static_cast<std::size_t>(nr) *\n"
        "                            static_cast<std::size_t>(mi_cols_) +\n"
        "                        static_cast<std::size_t>(nc)];\n"
        "      ++nnum;\n"
        "    };\n"
        "    add(r + h4 - 1, c - 1);\n"
        "    add(r - 1, c + w4 - 1);\n"
        "    add(r, c - 1);\n"
        "    add(r - 1, c);\n"
        "    return sum;\n"
        "  }",
    )

    open(dst, "w", encoding="utf-8").write(text)
    print(f"patched {src} -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

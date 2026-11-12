# Production integration (blocked, exact construction)

This component is deliberately self-contained. The shipped M1/M2 paths still
have **no IBC mode-info branch**, so `morph_pred` cannot be reached from
`src/main.cpp` without the edits below. They are listed as exact construction
and call sites; none is applied in this work.

The closest working example is `src/encoder/m1/ibc_stream/m1_ibc_stream.cpp`
(the IBC gate writer). It already writes `use_intrabc`, `skip_flag`,
`intrabc_info`, copies the IBC prediction and updates the neighbour grids.
`morph_pred` slots into that writer at three points.

## 1. Sequence and frame header bits

### `src/encoder/m1/m1_headers.h` — `M1Config`

```cpp
bool enable_bawp = false;                    // §5.4.6
bool allow_screen_content_tools = false;     // §5.18.2
bool allow_intrabc = false;                  // §5.18.2
bool allow_global_intrabc = true;
bool allow_local_intrabc = false;
```

### `src/encoder/m1/m1_headers.cpp:127`

```diff
-  writer.put_bit(false);  // enable_bawp
+  writer.put_bit(c.enable_bawp);
```

### `src/encoder/m1/m1_headers.cpp:170-171`

```diff
-  writer.put_bit(false);  // allow_screen_content_tools
-  writer.put_bit(false);  // allow_intrabc
+  writer.put_bit(c.allow_screen_content_tools);
+  if (c.allow_screen_content_tools) {
+    writer.put_bit(true);  // force_integer_mv
+  }
+  writer.put_bit(c.allow_intrabc);
+  if (c.allow_intrabc) {
+    writer.put_bit(c.allow_global_intrabc);
+    if (c.allow_global_intrabc) {
+      writer.put_bit(c.allow_local_intrabc);
+    }
+  }
```

`enable_bawp = 1` is what makes `morph_pred` signallable. In the
`single_picture_header_flag = 1` branch, `enable_imp_msk_bld` is inferred 0
and is not written; it has no effect on morph.

## 2. IBC stream writer: the three morph call sites

### Add per-tile state

```cpp
#include "encoder/inter/morph/morph_common.h"
#include "encoder/inter/morph/morph_syntax.h"

inter::morph::MorphCdfs morph_cdfs_;
std::vector<std::uint8_t> morph_preds_;  // mi_cols_ * mi_rows_, 4x4 grid
```

Initialize in the `TileWriter` constructor:

```cpp
inter::morph::init_morph_cdfs(morph_cdfs_);
morph_preds_.assign(static_cast<std::size_t>(mi_cols_) * mi_rows_, 0);
```

### A. Before writing the IBC block (validate, then code the symbol)

Immediately after `write_intrabc_info(...)` at
`m1_ibc_stream.cpp:433-436`:

```cpp
const int bw4 = std::max(1, bw / 4);
const int bh4 = std::max(1, bh / 4);
inter::morph::MorphPredGate gate;
gate.enable_bawp = config_.enable_bawp;
gate.allow_screen_content_tools = config_.allow_screen_content_tools;
gate.frame_is_intra = true;
gate.use_intrabc = true;
std::string morph_why;
const bool offset_valid = is_bv_offset_valid(
    frame_view(), ibc_config_, buffer_, block.mi_row, block.mi_col,
    block.bsize, block.bv, -1, -1, &morph_why);
if (block.morph_pred &&
    !inter::morph::morph_pred_allowed(gate, offset_valid, &morph_why)) {
  error = morph_why;
  return false;
}
const int morph_ctx = inter::morph::morph_symbol_context(
    morph_preds_.data(), mi_cols_, mi_rows_, block.mi_row, block.mi_col,
    bw4, bh4, 64);
if (!inter::morph::write_morph_pred(enc_, morph_cdfs_, morph_ctx,
                                    block.morph_pred, error)) {
  return false;
}
```

`block.morph_pred` is the encoder's decision. It must be 0 whenever the
prediction would not be applied (e.g. no valid `-1,-1` border); the check above
enforces that. The `§7.13.3.25` reference-box check is in
`morph_reference_geometry()`/`morph_reference_in_bounds()` and should also be
called before emitting the symbol if a plan can carry an unusual BV.

### B. After the IBC copy, apply the predictor

Immediately after `ibc_copy(...)` at `m1_ibc_stream.cpp:442`:

```cpp
if (block.morph_pred) {
  inter::morph::MorphContext mctx;
  mctx.mi_col = block.mi_col;
  mctx.mi_row = block.mi_row;
  mctx.mi_cols = mi_cols_;
  mctx.mi_rows = mi_rows_;
  mctx.avail_u = block.mi_row > 0;
  mctx.avail_l = block.mi_col > 0;
  inter::morph::MorphPlane plane{recon_y_.data(), config_.width};
  const inter::morph::MorphRect rect{x, y, bw, bh};
  if (!inter::morph::morph_predict(mctx, plane, rect, block.bv.row,
                                   block.bv.col)) {
    error = "morph prediction failed";
    return false;
  }
}
```

Morph is luma only; do not touch `recon_u_`/`recon_v_`.

### C. Update `MorphPreds`

In the existing per-4x4 update loop at `m1_ibc_stream.cpp:456-480`:

```cpp
morph_preds_[idx] = (is_ibc && block.morph_pred) ? 1 : 0;
```

This is `§5.20.4.1`: `MorphPreds[r+y][c+x] = use_intrabc && morph_pred`.
It must be updated for **every** block (0 for intra blocks), because the CDF
context sums the two `NPos` neighbours.

## 3. Production `m1_syntax.cpp` / `m1_still.cpp`

`m1_syntax.cpp` has no IBC branch and `BlockSyntax` has no IBC fields. To wire
the tool:

* add `use_intrabc`, `morph_pred`, the block vector and the IBC neighbour
  state to the block-result/syntax structs;
* in `Writer::write_block()` (`m1_syntax.cpp:1372`), before `code_y_mode`,
  emit `use_intrabc` with the §5.20.4.1 context, then
  `skip_flag`/`intrabc_info`/`morph_pred` exactly as in the gate writer;
* when `use_intrabc == 1`, skip `code_y_mode`/`code_uv_mode`/palette and code
  residual only when `skip_flag == 0`;
* in `m1_still.cpp`, reconstruct with `ibc::copy_block()` and then
  `inter::morph::morph_predict()` for the luma plane, in that order.

Without the IBC branch, any attempt to signal `morph_pred` is unreachable
because the symbol is read only from `read_intrabc_info()`.

## 4. CLI

`src/main.cpp` needs an opt-in switch (e.g. `--m1-ibc-morph`) that sets the
header fields and enables the IBC/morph decision. No CLI change is needed for
the component tests.

## What is already exact

* `morph_syntax.*` is the complete `S()` writer/reader and §8.3.2 context.
* `morph_common.h` is the complete §7.13.3.26 process.
* `morph_stream.cpp` is a working single-frame writer that proves the
  sequence/frame bits, the symbol, the context, the predictor and the decoder
  behavior end-to-end.
* `morph_test` pins all 22 IBC-legal block sizes × 4 availability states.

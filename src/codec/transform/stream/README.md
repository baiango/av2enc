# M1 transform-stream adapter (`src/encoder/m1/transform_stream/`)

Status: **component adapter.** It is not wired into `CMakeLists.txt` or
`src/main.cpp` by this agent, and it does not commit. It owns only this
directory. The M1 core can call it from the TX stage; the exact call sites are
below.

This is the single decision surface over the two verified M1 components:

| Component | What it supplies |
|---|---|
| `src/encoder/tx_type/` | legal intra TX sets, §5.20.8.2 symbols, static §9.3 type cost |
| `src/encoder/rdoq/` | full coefficient-level trellis with a pluggable rate model |
| `src/cuda/tx_type/` | bit-exact batch forward/inverse transforms per candidate |
| `src/cuda/rdoq/` | bit-exact batch trellis (caller-supplied rate table) |

One call consumes a coded residual, block geometry, the signalled intra mode,
a Lagrangian lambda and a rate model, and returns:

```
{ levels, eob, tx_type_symbol, cost }  (+ reconstruction, SSE, diagnostics)
```

## Files

| File | Contents |
|---|---|
| `transform_stream.h` / `.cpp` | CPU reference: candidate enumeration, TX-type symbol, RDOQ, selection |
| `transform_stream_gpu.h` / `.cu` | GPU batch path over the CUDA components |
| `transform_stream_test.cpp` | CPU tests + real-frame residuals |
| `transform_stream_gpu_test.cu` | per-candidate CPU-vs-GPU parity |
| `transform_stream_quality.cpp` | decoder-gated tile-byte / PSNR probe |
| `build_and_test.sh` | CPU test + quality probe (no CMake) |
| `build_gpu_test.sh` | CUDA parity test, run on the sm_86 box |

## API

```cpp
namespace av2enc::still::ts {

struct Request {
  const std::int16_t* residual_coded;  // coded size, row-major, tightly packed
  int coded_w, coded_h;                // 4..64
  int intra_mode, angle_delta;         // signalled mode; mapped internally
  int reduced_tx_set;                  // 0..3
  bool dct_only;                       // chroma / force DCT_DCT, no type syntax
  bool enable_idtx_intra = true;       // M1 passes false (§5.4.8)
  double lambda;                       // SSE-domain, coded residual/coeff units
  int plane;                           // 0 luma, 1 U, 2 V
  int txb_skip_context, dc_sign_context;
  int prev_level_cap = 2, level_cap = 0;
  bool allow_zero_to_one = false;
  const RateModelHooks* hooks = nullptr;
  std::span<const txtype::TxType> candidate_types;  // optional subset
};

struct Decision {
  bool valid, codable;
  txtype::TxType type;
  txv::TxParams params;                // chosen TX type + adjusted geometry
  TxTypeSymbol symbol;                 // exact §5.20.8.2 emission data
  int eob, nonzero;
  double sse;                          // coded-domain reconstruction SSE
  double coeff_rate_bits, type_bits, rate_bits, cost;
  double rdoq_distortion, rdoq_rate_bits, rdoq_cost;
  int max_level_searched; bool level_range_truncated;
  int candidates_considered;
  std::vector<std::int32_t> levels;    // scan order, size params.w*params.h
  std::vector<std::int16_t> recon_coded;
};

bool quantize_block(const Request&, Decision*, std::string* error = nullptr);
bool quantize_block_scalar(const Request&, Decision*, std::string* error = nullptr);
bool evaluate_candidates(const Request&, std::vector<Decision>*, std::string* error = nullptr);
TxTypeSymbol classify_tx_type(const txtype::TxTypeQuery&, txtype::TxType, int eob);
double default_lambda(const txv::TxParams&, double sse_scale = 0.125);
}
```

`levels` is **scan order** (the order the M1 writer consumes). `eob` is the
coded scan-position count. Trailing positions are zero.

### Selection order

Exact binary64, deterministic:

1. lowest `cost = sse + lambda * (coefficient/EOB bits + type bits)`,
2. lowest coded-domain `sse`,
3. lowest numeric `TX_TYPE` (DCT_DCT first).

`quantize_block_scalar` uses the same candidates, rate model and tie-break but
skips the trellis. `evaluate_candidates` returns the per-candidate evaluations
in symbol order; it is the CPU half of the GPU parity test.

### Lambda

Pass the same lambda on both sides of the decision:

```cpp
const auto params = *txv::make_tx_params(coded_w, coded_h);
const double quant_step = 1.0 / params.quant_factor;
const double lambda =
    rdo::lambda_sse_from_quant_step(quant_step, sse_scale);
// or: const double lambda = ts::default_lambda(params, sse_scale);
```

`sse_scale = 0.125` is the RDO-framework placeholder. Re-calibrate it at the
operating point. Do not pass the partition lambda or a QP lambda in a different
distortion domain. The current M1 tree has separate `rdoq_sse_scale` and
`tx_type_sse_scale` knobs (both default 0.125); the adapter uses one lambda for
both decisions, so pick the operating-point value and use it on both sides.

## §5.20.8.2 TX-type emission

`classify_tx_type(query, type, eob)` returns exactly what the writer must emit:

| `kind` | Condition | Symbol(s) | CDF | Cost |
|---|---|---|---|---|
| `kNone` | `eob <= 1`, `dct_only`, `reduced_tx_set == 2`, DCTONLY size | none | none | 0 |
| `kIntraSet1` | INTRA_1 (sqr_up <= 16x16) | one of 7 | `IntraTxTypeSet1Cdf[Tx_Size_Sqr]` | `-log2 P(symbol)` |
| `kIntraSet2` | INTRA_2 (`reduced_tx_set` 1/3) | one of 2 | `IntraTxTypeSet2Cdf[...]` | `-log2 P(symbol)` |
| `kLong32` | WIDE_32/HIGH_32 | flag + one of 4 | `IsLongSideDctCdf` then `IntraTxTypeLongCdf[...]` | both symbols |
| `kLong64` | WIDE_64/HIGH_64 | one of 4 | `IntraTxTypeLongCdf[...]` | `-log2 P(symbol)`; flag implied 1 |

`symbol` is -1 for `kNone`; `has_long_flag` is true only for `kLong32`;
`is_long_side_dct` is the signalled value (true for `kLong64`). The
`TxTypeSyntax` enum in `src/encoder/m1/still_syntax.cpp` classifies the same cases
for `reduced_tx_set = 0`; this adapter additionally supports INTRA_2 and
`enable_idtx_intra = 0`.

`candidate_codable(type, eob)` is enforced: a non-DCT winner whose RDOQ result
collapses to `eob <= 1` is not signallable and is dropped. `DCT_DCT` is in every
legal set, so a winner always exists.

### M1 rule: `enable_idtx_intra = 0`

M1 signals `enable_idtx_intra = 0` (§5.4.8), which forbids any 1D identity
transform. The verified TX-type component lists IDTX / V_DCT / H_DCT / V/H_ADST
/ V/H_FLIPADST in the long-32 sets. Pass `enable_idtx_intra = false` from M1;
the adapter filters those candidates before the decision and the GPU path
applies the same filter. Default is `true` (component full set).

## Rate model

The default is the AV2 §9.3 CDF model (`rdoq_av2_model.h`, static qctx 0,
`base_q_idx <= 90`), built per candidate from its scalar levels. It includes
`txb_skip`, EOB, `coeff_base` / `coeff_base_eob`, `coeff_br`, escape and sign
bits.

`RateModelHooks` replaces it:

```cpp
struct RateModelHooks {
  bool (*build)(void* user, const RateModelRequest&,
                rdoq::RateModel*, std::string* error);
  void (*release)(void* user, rdoq::RateModel*) noexcept;
  void* user;
  double (*type_bits)(void* user, const txtype::TxTypeQuery&,
                      const txtype::TxTypeCandidate&);
};
```

`build` is called once per candidate; `release` after the trellis. When
`release` is null, the hook owns the context for the adapter call. `type_bits`
replaces the static §9.3 symbol cost. For GPU parity the hook must be a pure
function of its arguments: the GPU path materialises the table by probing every
`(scan position, prev state, magnitude)` the trellis can query.

For M1, pass the **same** `txb_skip_context` and `dc_sign_context` the syntax
writer will derive from above/left cumulative levels. Passing 0 (as the current
in-tree RDOQ block does) is an approximation and changes the estimated bits.

## CPU / GPU parity

CPU tests (`build_and_test.sh`): **250,129 checks, 0 failures** and clean under
`SANITIZE=1` (adapter test).

GPU parity (`build_gpu_test.sh`) compares **every candidate**, not just the
winner:

* type, `eob`, nonzero, SSE, coefficient/type/total rate, cost;
* every level in scan order;
* every reconstructed residual sample;
* the §5.20.8.2 class/symbol/flag/bits;
* the winner and its cost; determinism across two launches.

The CUDA test covers synthetic mixed-size batches (including `enable_idtx_intra
= 0`, a candidate subset, `level_cap`, and a custom rate hook) plus real 64x64
and 640x360 luma blocks.

**This checkout has no `nvcc`**, so `build_gpu_test.sh` has not been run here.
To keep the glue honest, it was exercised with a host emulation of the four
CUDA entry points (CPU transform + CPU table-backed trellis + CPU inverse, with
a `malloc`/`memcpy` CUDA runtime shim). That emulation passes the same 6,156
checks, including the real frames, and is clean under ASan/UBSan. The real
kernels themselves are independently verified by `src/cuda/rdoq/` and
`src/cuda/tx_type/`. Run `build_gpu_test.sh` on the 3060 box before letting the
GPU result drive a bitstream.

## Integration notes

`cmake_fragment.txt` records the two source additions and the
`-ffp-contract=off` property the adapter needs. The M1 GPU target already
compiles every `.cu` with `--fmad=false`.

### CPU M1: `src/encoder/m1/still_encoder.cpp::compute_tx`

The current tree already has a TX-type chooser and an inline RDOQ pass in
`compute_tx` (function names are stable; line numbers move). Replace both with
one adapter call:

```cpp
av2enc::still::ts::Request req;
req.residual_coded = residual_coded.data();
req.coded_w = cw;
req.coded_h = ch;
req.intra_mode = y_mode;          // signalled YMode
req.angle_delta = y_angle_delta;
req.reduced_tx_set = 0;           // still_headers writes 0
req.dct_only = (plane != 0);      // chroma never signals a type
req.enable_idtx_intra = false;    // M1 §5.4.8
req.plane = plane;
req.txb_skip_context = skip_ctx;  // same contexts the writer derives
req.dc_sign_context = dc_ctx;
req.lambda = ts::default_lambda(*txv::make_tx_params(cw, ch),
                                options_.rdoq_sse_scale);

av2enc::still::ts::Decision d;
std::string error;
if (!av2enc::still::ts::quantize_block(req, &d, &error)) { ... }

plan.tx_size = d.params.tx_size;
plan.tx_type = static_cast<int>(d.type);
plan.adj_w = d.params.w;
plan.adj_h = d.params.h;
plan.eob = d.eob;
plan.levels.assign(static_cast<std::size_t>(d.params.w * d.params.h), 0);
std::vector<std::uint16_t> scan(static_cast<std::size_t>(d.params.w * d.params.h));
syntax::build_scan(d.params.w, d.params.h, scan);
for (int c = 0; c < d.eob; ++c) {
  plan.levels[scan[static_cast<std::size_t>(c)]] = d.levels[static_cast<std::size_t>(c)];
}
// plan.recon = clip1(pred + d.recon_coded), exactly as the current code does.
```

`d.levels` is scan order; `plan.levels` is raster. `append_levels` then writes
the scan-order levels unchanged.

### CfL ordering (mandatory)

`encode_block` chooses the luma mode, calls luma `compute_tx`, writes luma
recon, then calls `choose_uv_mode` and chroma `compute_tx`. The adapter must
run at the **luma** `compute_tx` and its `recon_coded` must be written to
`recon_[0]` **before** `choose_uv_mode`, because CfL reads the reconstructed
luma below/right of the current block. Chroma `compute_tx` passes
`dct_only = true` (and `plane = 1/2`), so it never signals a type.

### Syntax writer

`src/encoder/m1/still_syntax.h` already carries `BlockSyntax::tx_type[3]`, and
`still_syntax.cpp` already emits the symbol from `node.tx_type[0]` using
`txtype::tx_type_symbol` (the `TxTypeSyntax` switch). The adapter's
`Decision::type` feeds that field. If you are on a tree where the writer still
hardcodes symbol 0, the exact replacement is:

```cpp
const auto sym = ts::classify_tx_type(query, type, eob);  // query = same size/mode
switch (sym.kind) {
  case ts::TxTypeSyntaxKind::kNone: break;
  case ts::TxTypeSyntaxKind::kIntraSet1:
    sym(&cdfs_.intra_tx_type_set1[kTxSizeSqr[txSz]][0], 7, sym.symbol);
    break;
  case ts::TxTypeSyntaxKind::kIntraSet2:
    sym(&cdfs_.intra_tx_type_set2[kTxSizeSqr[txSz]][0], 2, sym.symbol);
    break;
  case ts::TxTypeSyntaxKind::kLong32:
    sym(&cdfs_.is_long_side_dct[0][0], 2, sym.is_long_side_dct ? 1 : 0);
    sym(&cdfs_.intra_tx_type_long[kTxSizeSqr[txSz]][0], 4, sym.symbol);
    break;
  case ts::TxTypeSyntaxKind::kLong64:
    sym(&cdfs_.intra_tx_type_long[kTxSizeSqr[txSz]][0], 4, sym.symbol);
    break;
}
```

The current `still_syntax.cpp` enum has no INTRA_2 case (M1 writes
`reduced_tx_set = 0`); add the `intra_tx_type_set2` CDF member if you enable
`reduced_tx_set` 1/3. Chroma never enters this switch.

### GPU M1

`src/cuda/m1/still_gpu.cu` currently rejects `--still-rdoq` / `--still-tx-type` (the
option gate near the `StillConfig` setup) and hardcodes `decision.tx_type_* = 0`.
The TX jobs are initialised in the `init_tx` lambda near the block loop
(`StillTxJob` is created with `TxType::kDctDct`). The adapter is the piece that
can turn those options on.

Call `gpu::quantize_batch` after partitioning and prediction, before writing
the stream:

```cpp
av2enc::still::ts::gpu::BatchOptions options;
options.allocator = arena_allocator;   // src/cuda/arena.*, no cudaMalloc
options.stream = stream;
options.table_budget_bytes = 256ull << 20;
av2enc::still::ts::gpu::quantize_batch(requests, nblocks, options,
                                    decisions, &error);
```

It returns the same `Decision` fields (host vectors). The device runs the
verified forward transforms, trellis and inverse transforms. The rate table is
materialised on the host from the same `RateModelHooks`/AV2 model because the
current `src/cuda/rdoq/` contract takes a caller-supplied table; a future
device-side model can replace that without changing this API.

The current entry point takes **host** residuals and returns host decisions.
For the fully device-resident M1 pipeline, either download the residual/EOB
buffers for this one decision per frame (correctness bring-up) or extend the
adapter with a device-residual overload plus a device rate model. The latter is
the only way to keep the whole frame GPU-resident, because the existing
`src/cuda/rdoq/` kernel deliberately has no device callback. Set
`decision.tx_type_y` from the luma `Decision::type`; keep U/V at 0.

`BatchOptions::table_budget_bytes` is a hard cap. If a batch exceeds it the
call fails before allocation with the required size; split the block batch
(the table is dominated by the largest coefficient bound, e.g. a uniform 64x64
DC block).

## Measurements (decoder-gated)

`transform_stream_quality` replays the M1 partition tree and TX geometry with
DC prediction from its **own reconstruction** (so the decoder sees the same
trajectory), three times: scalar, RDOQ-DCT, and RDOQ + TX types
(`enable_idtx_intra = 0`). It builds a real §8.2 stream with
`assemble_still_frame`, writes IVF with `--write-ivf`, and the pinned
`dav2d` (f9dafbc) decodes all three. **Decoded luma PSNR matches the reported
PSNR exactly.**

```sh
./build_and_test.sh
# or explicitly:
./transform_stream_quality --input tests/data/basic_640x360_1f.y4m \
  --tx-types --write-ivf /tmp/rq360
tools/validate_dav2d.sh /tmp/rq360.txtype.ivf tests/data/basic_640x360_1f.y4m
```

### 64x64 (`tests/data/basic_64x64_1f.y4m`, QP 90, 11 leaves)

| | scalar | RDOQ DCT | RDOQ + TX types |
|---|---:|---:|---:|
| tile bytes | 994 | 880 | 887 |
| frame bytes | 1015 | 901 | 908 |
| luma PSNR (decoded) | 42.488 | 41.380 | 41.634 |
| model bits | 6129.8 | 5087.2 | 5143.2 |
| RD cost | 87,874.9 | 80,682.6 | 79,973.3 |
| non-DCT wins | - | - | 8 / 11 |

### 640x360 (`tests/data/basic_640x360_1f.y4m`, QP 90, 218 leaves)

| | scalar | RDOQ DCT | RDOQ + TX types |
|---|---:|---:|---:|
| tile bytes | 15517 | 14031 | 13931 |
| frame bytes | 15540 | 14054 | 13954 |
| luma PSNR (decoded) | 45.081 | 44.600 | 44.629 |
| model bits | 95963.7 | 81970.7 | 81213.3 |
| RD cost | 1,394,193.2 | 1,306,924.0 | 1,296,833.1 |
| non-DCT wins | - | - | 56 / 218 |

TX-type histogram at 640x360: `0:162 1:19 2:12 3:6 4:13 5:1 7:2 8:3`.
Chroma is unchanged between variants (U 48.275 dB, V 49.486 dB).

**Reading the numbers.** With the fixed partition tree and DC mode:

* RDOQ saves **11.47% / 9.58%** of tile bytes for **-1.11 / -0.48 dB** luma.
* TX types add **-0.88% / -0.77%** RD cost over RDOQ-DCT. At 640x360 they also
  save another 100 tile bytes and gain +0.029 dB; at 64x64 they cost 7 bytes
  and gain +0.254 dB. Both are RD-positive at `sse_scale = 0.125`.
* This is a quantisation/TX-type measurement, not a full-encoder BD-rate:
  partition and mode decisions are held, CfL is disabled in the probe, and the
  reference M1 encoder's `base_q_idx = 90` stream is different (it uses the full
  mode/CfL search). Do not quote these as BD-rate.

## Limitations

* The GPU path builds rate tables on the host. Large batches need splitting;
  a device-side rate model is future work.
* The AV2 model is the static qctx-0 set (`base_q_idx <= 90`); contexts are
  frozen from the candidate's scalar quantisation (standard RDOQ
  approximation).
* No adaptive CDF, no lossless/FSC, no inter sets, no IST/DPCM.
* `candidate_codable` drops a non-DCT candidate whose RDOQ result is
  `eob <= 1`; it does not re-run a constrained trellis.
* The trellis minimises coefficient-domain distortion; the candidate comparison
  uses coded-domain SSE, matching `txtype`. For adjusted 64-point sizes the
  coded domain duplicates samples, so the RD balance follows the verified
  components' existing convention.
* The quality probe uses DC only and no deblocking, so its absolute tile bytes
  are not the M1 reference; only the same-tree deltas are meaningful.

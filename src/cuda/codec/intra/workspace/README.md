# Arena-backed intra workspace

One entry point for the AV2 intra tools: `modes`, MRL, IBP, DIP, CfL, palette,
IntraBC and the supplemental intra-full rules.  It exists so the still
orchestrator does not stage descriptor arrays by hand (the failure recorded in
`docs/negative-results.md`: a host `std::vector<Job>` passed to a `*_batch`
device API).

```
workspace/
  intra_workspace.h            API, Config, arena sizing, Status contract
  intra_workspace.cu           staging + tool dispatch
  intra_workspace_test.cu      real-GPU parity/contract test
  build_intra_workspace_test.sh
  README.md
```

Nothing outside `src/cuda/codec/intra/` is modified; the workspace is not wired
into `src/main.cpp`, the root `CMakeLists.txt`, or a still call site.

## API

```cpp
namespace av2enc::cuda::intra_workspace {

struct Config {
  int max_jobs = 256;              // descriptors per batch, every tool
  intra_full::Config supplemental; // forwarded DC/corner/chroma/tx/sign config
};

std::uint64_t staging_device_bytes(const Config&);
std::uint64_t persistent_device_bytes(const Config&);
std::uint64_t scratch_device_bytes(const Config&);
std::uint64_t required_device_bytes(const Config&);

class Workspace {
  Status init(Arena&, const Config& = {}, std::string* error = nullptr,
              std::uint64_t* required_out = nullptr);

  Status synchronize(cudaStream_t = nullptr);

  Status modes_predict(const intra_modes::IntraJob*, int, cudaStream_t = nullptr);
  Status mrl_predict(const mrl::MrlJob*, int, cudaStream_t = nullptr);
  Status ibp_predict(const ibp::IbpJob*, int, cudaStream_t = nullptr);
  Status dip_predict(const dip::DipJob*, int, cudaStream_t = nullptr);
  Status cfl_predict(const CflJob*, int, cudaStream_t = nullptr);
  Status cfl_predict_derived(const cfl::Block*, int* d_alpha_out, int,
                             cudaStream_t = nullptr);
  Status palette_apply(const PaletteDeviceBlock*,
                       const PaletteContextDevice* /* may be null */, int,
                       cudaStream_t = nullptr);
  Status ibc_copy(...);      // validated BV -> prediction planes
  Status ibc_sad(...);       // host BV array -> device SAD array
  Status ibc_validate(...);  // host BV array -> device validity bytes

  intra_full::Workspace& supplemental();  // DC-subsampled, corner, chroma type,
                                          // dc_sign, identity transforms
};
}
```

`init()` takes the caller's single `Arena` and allocates:

* one persistent descriptor-staging region (`max_jobs` × the largest job);
* the forwarded `intra_full::Workspace` persistent result arrays.

`required_device_bytes(config)` is exact: it is `staging + supplemental
persistent + supplemental scratch`, with every allocation rounded to
`Arena::kAlignment` exactly like `Arena::allocate_array`.  An undersized arena
returns `Status::kArenaExhausted`, reports the requirement through
`required_out` / `error`, and restores the arena bump pointer (no residue).  A
bad configuration returns `Status::kInvalidArgument` with requirement 0.

Every device byte comes from that arena; `tools/check_arena.sh` passes for the
tree.  The test's own input planes use `cudaMalloc` in the test translation unit
(permitted, not module code).

## Execution model

The tool methods are asynchronous: they upload host descriptors into the single
staging region and launch on the caller's stream.  Calls are serialized by the
stream, so a later upload is ordered after every earlier kernel that reads the
region.  Use one stream per workspace, keep the device buffers referenced by the
jobs alive until `synchronize()`, and do not interleave streams.

`palette_apply` stages the block descriptors, runs build-map + prediction, then
reuses the same region for the optional context descriptors (ordered after the
prediction kernels on the same stream).  The module's batch entry points use a
fixed 4x4x16x16 grid with `grid.z = count`, because palette blocks are at most
64x64.

The supplemental tools keep their module's synchronous batch semantics; call
them through `supplemental()` with the same arena.

## Build / test

```sh
cd src/cuda/codec/intra/workspace
./build_intra_workspace_test.sh                 # sm_86 (default)
ARCH=sm_120 ./build_intra_workspace_test.sh     # secondary target
./build_intra_workspace_test.sh --no-run
```

Measured on the RTX 3060 (`sm_86`, CUDA 13.0):

```
  modes: cases=370 samples=261072 mismatches=0
  modes edge-filter staging: cases=22 mismatches=0
  mrl: cases=525 samples=517440 mismatches=0
  ibp: cases=384 samples=352256 mismatches=0
  dip: cases=336 samples=319488 mismatches=0
  cfl explicit: cases=400 samples=136000 mismatches=0
  cfl derived: cases=400 samples=136000 mismatches=0
  palette: blocks=6 samples=5440 mismatches=0
  ibc: bvs=15 sad/validate=15 copy=320 mismatches=0
  arena: required=136448 persistent=132864 scratch=3584 capacity=1185024 peak=132864
intra_workspace_test: PASS (4379 checks, 0 failures)
```

* modes: CPU parity with the normative edge filter disabled (the CPU module's
  in-place filter is the documented cross-agent divergence, see
  `../modes/README.md`), plus workspace-vs-module equality with the filter on.
* MRL / IBP / DIP / CfL: CPU-reference parity through the workspace.
* palette: batch-vs-single-block equality (the single-block path already has
  the CPU parity suite in `../palette/`).
* IBC: workspace-vs-module equality for SAD, global validation and block copy.
* arena: `required = persistent + scratch`, exact init growth, flat `used()`
  across launches, one-byte-short `kArenaExhausted`, invalid config rejection.

## Still-orchestrator hand-off

The workspace is deliberately a thin dispatcher: it does not decide modes,
choose palette colours, or assemble syntax.  Integration is:

1. size the process arena from the device and `required_device_bytes(config)`;
2. `Workspace::init(arena, config)`;
3. per block/frame, build the host descriptor arrays the existing CPU planner
   already produces, call the matching method on the frame stream;
4. one `synchronize()` per frame;
5. call `supplemental()` for the DC-subsampled / dc_sign / identity-TX paths.

Remaining integration work is the same as the module READMEs: MRL/IBP/DIP/
palette/IBC/MHCCP are not wired into a stream, and several need syntax writers
or sequence-header flags before a `dav2d` gate is possible.

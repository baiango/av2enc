// CUDA intra features that were not covered by the M1 intra-modes component:
//
//   1. §7.13.2.11  DC intra prediction, **subsampled** variant.  stepW/stepH
//      become 2 when the extent exceeds 32; bit-depth agnostic (8/10/12).
//   2. §7.13.2.14  filter corner, 5/6/5 taps, Round2(., 4), guarded by
//      pAngle in (90, 180) and w + h >= 24.
//   3. All intra identity transform types through separable kernels:
//      IDTX, V_DCT, H_DCT, V_ADST, H_ADST, V_FLIPADST, H_FLIPADST.  The device
//      body is a bit-exact port of `src/encoder/tx_variants/tx_variants.cpp`
//      (`forward_block` / `inverse_block`) for those seven TX_TYPEs.
//   4. §5.20.7.29  chroma PlaneTxType derivation when
//      `enable_chroma_dctonly == 0`: Mode_To_Txfm[wide-angle-mapped UVMode]
//      filtered through `Tx_Type_In_Set_Intra[set]`, else DCT_DCT.
//   5. §5.20.7.27 / §8.3  `dc_sign` and `dc_sign_horz_vert` sign-element
//      selection plus the neighbour-derived `dc_sign` context.  The decision is
//      pure integer work; the shipped M1 writer never selects a 1-D intra TX
//      type, so this kernel is coverage for a future gate, not a wired path.
//
// This component owns only `src/cuda/intra_full/`.  It does not edit the core,
// the root CMakeLists, or `src/main.cpp`, and it is not wired into a stream.
//
// Memory contract (mandatory, project-wide)
// -----------------------------------------
// Every device byte comes from the single persistent `av2enc::cuda::Arena`.
// There is no cudaMalloc / cudaFree / cudaMallocHost in this module or in its
// launchers.  `required_device_bytes(config)` is the exact arena requirement of
// `Workspace::init()` for that configuration:
//
//   persistent : device result arrays, allocated once by init()
//   scratch    : per-launch descriptor staging, one `arena.mark()` .. `rewind()`
//
// `Workspace::init()` returns `Status::kArenaExhausted` (never a CUDA error and
// never a fallback allocation) when the arena is short, and reports the needed
// byte count through `required_out` / `error`.
//
// The batch entry points own their stream synchronization: they stage the host
// descriptor array in arena scratch, launch, copy requested results back, wait
// on the stream, and rewind the arena.  This makes `arena.mark()`/`rewind()`
// safe (the scratch is dead before it is reused) and keeps `used()` flat across
// back-to-back launches.  There is no hidden per-frame device sync in a kernel.
#ifndef AV2ENC_CUDA_INTRA_FULL_INTRA_FULL_H_
#define AV2ENC_CUDA_INTRA_FULL_INTRA_FULL_H_

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>

#include "cuda/arena.h"
#include "cuda/error.h"
#include "codec/transform/variants/tx_variants.h"

namespace av2enc::cuda::intra_full {

// ---------------------------------------------------------------------------
// §7.13.2.11 DC subsampled
// ---------------------------------------------------------------------------

// Inputs of §7.13.2.11.  `above`/`left` are device pointers to AboveRow[0] and
// LeftCol[0] from the §7.13.2.1 edge preparation.  The process is defined for
// 1..64 samples per axis; the real intra extents are 4..64 powers of two.
struct DcSubsampledDesc {
  const std::int16_t* above = nullptr;
  const std::int16_t* left = nullptr;
  int w = 0;
  int h = 0;
  int bit_depth = 8;  // 8, 10 or 12
  int have_above = 0;
  int have_left = 0;
};

// Trace of the process, matching av2enc::still::missing_bits::DcSubsampledTrace.
// step_w/step_h are 0 when the corresponding edge is unavailable.
struct DcResult {
  int step_w = 0;
  int step_h = 0;
  int sum = 0;
  int count = 0;
  int avg = 0;
};

// One DC prediction job.  `dst` is a device pointer; the whole block receives
// the same 16-bit prediction, so 10/12-bit values are not truncated.
struct DcJob {
  DcSubsampledDesc in;
  std::uint16_t* dst = nullptr;
  int dst_stride = 0;
};

// ---------------------------------------------------------------------------
// §7.13.2.14 filter corner
// ---------------------------------------------------------------------------

// `above` points at AboveRow[0] and `left` at LeftCol[0]; the corner slots
// above[-1] / left[-1] must be addressable (the spec's EdgeRow layout).
struct CornerJob {
  std::int16_t* above = nullptr;
  std::int16_t* left = nullptr;
  int w = 0;
  int h = 0;
  int p_angle = 0;
};

struct CornerResult {
  int applied = 0;  // 1 when the corner was rewritten
};

// ---------------------------------------------------------------------------
// Identity transform types (TX_TYPE 9..15)
// ---------------------------------------------------------------------------

// TX_TYPE values 9..15 are the intra identity types (§5.20.7.29 / §7.15.2.3).
inline bool is_intra_identity_type(int tx_type) noexcept {
  return tx_type >= 9 && tx_type <= 15;
}

// `row` is the 1D type applied across the width (dir 0), `col` down the height
// (dir 1).  Values are txv::Tx1dType (DCT=0, IDT=1, ADST=2, FDST=3).  Mirrors
// §7.15.4.1 Transform_1d_Type for IDTX and the six one-axis identity types.
// Returns false for the 2D DCT/ADST/FLIPADST types.
inline bool intra_identity_1d(int tx_type, int* row, int* col) noexcept {
  if (row == nullptr || col == nullptr) {
    return false;
  }
  switch (tx_type) {
    case 9:  // IDTX
      *row = 1;
      *col = 1;
      return true;
    case 10:  // V_DCT: identity across width, DCT down height
      *row = 1;
      *col = 0;
      return true;
    case 11:  // H_DCT: DCT across width, identity down height
      *row = 0;
      *col = 1;
      return true;
    case 12:  // V_ADST
      *row = 1;
      *col = 2;
      return true;
    case 13:  // H_ADST
      *row = 2;
      *col = 1;
      return true;
    case 14:  // V_FLIPADST
      *row = 1;
      *col = 3;
      return true;
    case 15:  // H_FLIPADST
      *row = 3;
      *col = 1;
      return true;
    default:
      return false;
  }
}

// One transform block.  `params` is a fully derived TxParams from the CPU
// reference (`av2enc::txv::make_tx_params`); offsets index the caller's
// residual/coefficient/level buffers, exactly like cuda::TxBlockJob.
struct TxJob {
  av2enc::txv::TxParams params;
  std::int32_t in_offset = 0;
  std::int32_t out_offset = 0;
};

// ---------------------------------------------------------------------------
// §5.20.7.29 chroma type derivation
// ---------------------------------------------------------------------------

struct ChromaQuery {
  int uv_mode = 0;               // 0..13 (13 = UV_CFL_PRED)
  int uv_angle_delta = 0;        // -3..3
  int coded_w = 0;               // 4..64, one of the 25 TX sizes
  int coded_h = 0;
  int plane = 1;                 // 1 = U, 2 = V (chroma derivation only)
  int enable_chroma_dctonly = 1;  // 0 enables the non-DCT derivation
};

struct ChromaResult {
  int tx_type = 0;   // TX_TYPE value
  int set = 0;       // §6.19.8.2 intra set value (0..6)
  int valid = 0;     // 0 for an unsupported size / mode / angle
};

// ---------------------------------------------------------------------------
// §5.20.7.27 / §8.3 dc_sign and dc_sign_horz_vert
// ---------------------------------------------------------------------------

// §5.20.7.29 get_tx_class(): which sign element a non-zero coefficient uses.
enum class SignTxClass : std::uint8_t {
  k2D = 0,
  kHoriz = 1,
  kVert = 2,
};

// §5.20.7.27 sign element for one coefficient position.
enum class SignElement : std::uint8_t {
  kDcSign = 0,
  kDcSignHorzVert = 1,
  kSignBit = 2,
};

struct SignQuery {
  int plane = 0;  // 0 = luma (the sign elements are luma-only)
  int row = 0;
  int col = 0;
  SignTxClass tx_class = SignTxClass::k2D;
  // Optional dc_sign context inputs.  They are read only when the selected
  // element is kDcSign; leave them null to get ctx 0 (above/left unavailable).
  const std::int8_t* above_dc = nullptr;
  const std::int8_t* left_dc = nullptr;
  int x4 = 0;
  int y4 = 0;
  int w4 = 0;
  int h4 = 0;
  int mi_cols = 0;
  int mi_rows = 0;
};

// `element` is the §5.20.7.27 selection; `ctx` is the §8.3 dc_sign context for
// kDcSign and the constant 0 for kDcSignHorzVert (and kSignBit, unused).
struct SignResult {
  std::uint8_t element = static_cast<std::uint8_t>(SignElement::kSignBit);
  std::uint8_t ctx = 0;
};

// §8.3 dc_sign context inputs.  `above_dc` is AboveDcContext[plane] indexed by
// x4, `left_dc` is LeftDcContext[plane] indexed by y4; either may be null.  The
// entries are the stored dcCategory values (0, 1 or 2).
struct SignContextJob {
  const std::int8_t* above_dc = nullptr;
  const std::int8_t* left_dc = nullptr;
  int x4 = 0;
  int y4 = 0;
  int w4 = 0;
  int h4 = 0;
  int mi_cols = 0;
  int mi_rows = 0;
};

// ---------------------------------------------------------------------------
// Configuration and arena sizing
// ---------------------------------------------------------------------------

// `max_*` bound the largest single launch.  A batch larger than the configured
// maximum is rejected with Status::kInvalidArgument.
struct Config {
  int max_dc_jobs = 64;
  int max_corner_jobs = 256;
  int max_chroma_jobs = 256;
  int max_tx_blocks = 64;
  int max_sign_jobs = 256;
};

namespace detail {

// Arena rounds every allocation up to its alignment (256 bytes by default).
constexpr std::uint64_t align_up_bytes(std::uint64_t value,
                                       std::uint64_t alignment) noexcept {
  return alignment == 0 ? value : ((value + alignment - 1) / alignment) * alignment;
}

constexpr int kMaxBatch = 1 << 20;

constexpr bool config_valid(const Config& c) noexcept {
  return c.max_dc_jobs >= 1 && c.max_dc_jobs <= kMaxBatch &&
         c.max_corner_jobs >= 1 && c.max_corner_jobs <= kMaxBatch &&
         c.max_chroma_jobs >= 1 && c.max_chroma_jobs <= kMaxBatch &&
         c.max_tx_blocks >= 1 && c.max_tx_blocks <= kMaxBatch &&
         c.max_sign_jobs >= 1 && c.max_sign_jobs <= kMaxBatch;
}

}  // namespace detail

// Persistent bytes (device result arrays) that init() keeps allocated.
inline std::uint64_t persistent_device_bytes(const Config& config) noexcept {
  if (!detail::config_valid(config)) {
    return 0;
  }
  const std::uint64_t align = Arena::kAlignment;
  return detail::align_up_bytes(sizeof(DcResult) *
                                    static_cast<std::uint64_t>(config.max_dc_jobs),
                                align) +
         detail::align_up_bytes(
             sizeof(CornerResult) *
                 static_cast<std::uint64_t>(config.max_corner_jobs),
             align) +
         detail::align_up_bytes(
             sizeof(ChromaResult) *
                 static_cast<std::uint64_t>(config.max_chroma_jobs),
             align) +
         detail::align_up_bytes(
             sizeof(SignResult) *
                 static_cast<std::uint64_t>(config.max_sign_jobs),
             align) +
         detail::align_up_bytes(
             sizeof(int) * static_cast<std::uint64_t>(config.max_sign_jobs),
             align);
}

// Largest per-launch scratch (descriptor staging) of any entry point.
inline std::uint64_t scratch_device_bytes(const Config& config) noexcept {
  if (!detail::config_valid(config)) {
    return 0;
  }
  const std::uint64_t align = Arena::kAlignment;
  const std::uint64_t dc =
      detail::align_up_bytes(sizeof(DcJob) *
                                 static_cast<std::uint64_t>(config.max_dc_jobs),
                             align);
  const std::uint64_t corner =
      detail::align_up_bytes(sizeof(CornerJob) *
                                 static_cast<std::uint64_t>(config.max_corner_jobs),
                             align);
  const std::uint64_t chroma =
      detail::align_up_bytes(sizeof(ChromaQuery) *
                                 static_cast<std::uint64_t>(config.max_chroma_jobs),
                             align);
  const std::uint64_t tx =
      detail::align_up_bytes(sizeof(TxJob) *
                                 static_cast<std::uint64_t>(config.max_tx_blocks),
                             align);
  const std::uint64_t sign_query =
      detail::align_up_bytes(
          sizeof(SignQuery) *
              static_cast<std::uint64_t>(config.max_sign_jobs),
          align);
  const std::uint64_t sign_context =
      detail::align_up_bytes(
          sizeof(SignContextJob) *
              static_cast<std::uint64_t>(config.max_sign_jobs),
          align);
  std::uint64_t largest = dc;
  if (corner > largest) largest = corner;
  if (chroma > largest) largest = chroma;
  if (tx > largest) largest = tx;
  if (sign_query > largest) largest = sign_query;
  if (sign_context > largest) largest = sign_context;
  return largest;
}

// Arena bytes `Workspace::init()` needs, relative to the arena mark at init.
// Includes the persistent result arrays and the largest per-launch descriptor
// staging array; every allocation is rounded to Arena::kAlignment exactly like
// Arena::allocate_array does.
inline std::uint64_t required_device_bytes(const Config& config) noexcept {
  return persistent_device_bytes(config) + scratch_device_bytes(config);
}

// ---------------------------------------------------------------------------
// Workspace: all device memory from one Arena
// ---------------------------------------------------------------------------

class Workspace {
 public:
  Workspace() = default;

  // Allocates the persistent result arrays exactly once and validates capacity.
  // On failure returns the reason; when it is kArenaExhausted the arena bump
  // pointer is restored and `*required_out` (when non-null) is the requirement.
  Status init(Arena& arena, const Config& config, std::string* error = nullptr,
              std::uint64_t* required_out = nullptr);

  bool initialized() const noexcept { return arena_ != nullptr; }
  const Config& config() const noexcept { return config_; }

  Arena* arena() noexcept { return arena_; }
  std::uint64_t persistent_bytes() const noexcept {
    return persistent_device_bytes(config_);
  }
  std::uint64_t scratch_bytes() const noexcept {
    return scratch_device_bytes(config_);
  }

  const DcResult* dc_results_device() const noexcept { return d_dc_results_; }
  const CornerResult* corner_results_device() const noexcept {
    return d_corner_results_;
  }
  const ChromaResult* chroma_results_device() const noexcept {
    return d_chroma_results_;
  }
  const SignResult* sign_results_device() const noexcept {
    return d_sign_results_;
  }
  const int* sign_context_results_device() const noexcept {
    return d_sign_context_results_;
  }

  // §7.13.2.11 batch.  `count <= config().max_dc_jobs`; `host_results` may be
  // null.  Synchronous (see the file-level contract).
  Status dc_subsampled_predict(const DcJob* host_jobs, int count,
                               DcResult* host_results = nullptr,
                               cudaStream_t stream = nullptr);

  // §7.13.2.14 batch.  Writes the corner into each job's above[-1]/left[-1].
  Status filter_corner(const CornerJob* host_jobs, int count,
                       CornerResult* host_results = nullptr,
                       cudaStream_t stream = nullptr);

  // §5.20.7.29 batch.  `host_results` must hold `count` entries.
  Status derive_chroma_type(const ChromaQuery* host_queries, int count,
                            ChromaResult* host_results,
                            cudaStream_t stream = nullptr);

  // §5.20.7.27 batch: sign element (and the dc_sign context) for `count`
  // coefficient positions.  `host_results` must hold `count` entries.
  Status select_sign_elements(const SignQuery* host_queries, int count,
                              SignResult* host_results,
                              cudaStream_t stream = nullptr);

  // §8.3 batch: neighbour-derived `dc_sign` context for `count` blocks.
  // `host_ctx` receives one int per job.
  Status dc_sign_contexts(const SignContextJob* host_jobs, int count,
                          int* host_ctx, cudaStream_t stream = nullptr);

  // Forward / inverse for the identity TX_TYPEs.  `residual` (int16, adjusted
  // size), `coeff` (double, adjusted size), `dequant` (int32, adjusted size)
  // and `coded` (int16, coded size) are caller-owned arena buffers.  Synchronous
  // because the descriptor scratch is rewound before returning.
  Status identity_forward(const TxJob* host_jobs, int count,
                          const std::int16_t* residual, double* coeff,
                          cudaStream_t stream = nullptr);
  Status identity_inverse(const TxJob* host_jobs, int count,
                          const std::int32_t* dequant, std::int16_t* coded,
                          cudaStream_t stream = nullptr);

 private:
  Arena* arena_ = nullptr;
  Config config_{};
  // Persistent device result arrays.
  DcResult* d_dc_results_ = nullptr;
  CornerResult* d_corner_results_ = nullptr;
  ChromaResult* d_chroma_results_ = nullptr;
  SignResult* d_sign_results_ = nullptr;
  int* d_sign_context_results_ = nullptr;
};

// ---------------------------------------------------------------------------
// Workspace::init (header-inline: the same allocation plan the emulation test
// checks against required_device_bytes())
// ---------------------------------------------------------------------------

inline Status Workspace::init(Arena& arena, const Config& config,
                              std::string* error,
                              std::uint64_t* required_out) {
  arena_ = nullptr;
  config_ = Config{};
  d_dc_results_ = nullptr;
  d_corner_results_ = nullptr;
  d_chroma_results_ = nullptr;
  d_sign_results_ = nullptr;
  d_sign_context_results_ = nullptr;

  if (!detail::config_valid(config)) {
    if (required_out != nullptr) *required_out = 0;
    if (error != nullptr) *error = "intra_full: invalid workspace configuration";
    return Status::kInvalidArgument;
  }

  const std::uint64_t need = required_device_bytes(config);
  if (required_out != nullptr) *required_out = need;
  if (!arena.initialized()) {
    if (error != nullptr) *error = "intra_full: arena is not initialized";
    return Status::kNotInitialized;
  }
  if (arena.remaining() < need) {
    if (error != nullptr) {
      *error = "intra_full: arena needs " + std::to_string(need) +
               " bytes, " + std::to_string(arena.remaining()) + " available";
    }
    return Status::kArenaExhausted;
  }

  const std::uint64_t mark = arena.mark();
  d_dc_results_ = arena.allocate_array<DcResult>(
      static_cast<std::size_t>(config.max_dc_jobs));
  d_corner_results_ = arena.allocate_array<CornerResult>(
      static_cast<std::size_t>(config.max_corner_jobs));
  d_chroma_results_ = arena.allocate_array<ChromaResult>(
      static_cast<std::size_t>(config.max_chroma_jobs));
  d_sign_results_ = arena.allocate_array<SignResult>(
      static_cast<std::size_t>(config.max_sign_jobs));
  d_sign_context_results_ = arena.allocate_array<int>(
      static_cast<std::size_t>(config.max_sign_jobs));
  if (d_dc_results_ == nullptr || d_corner_results_ == nullptr ||
      d_chroma_results_ == nullptr || d_sign_results_ == nullptr ||
      d_sign_context_results_ == nullptr) {
    arena.rewind(mark);
    d_dc_results_ = nullptr;
    d_corner_results_ = nullptr;
    d_chroma_results_ = nullptr;
    d_sign_results_ = nullptr;
    d_sign_context_results_ = nullptr;
    if (error != nullptr) {
      *error = "intra_full: arena exhausted after " + std::to_string(need) +
               " bytes";
    }
    return Status::kArenaExhausted;
  }

  config_ = config;
  arena_ = &arena;
  return Status::kOk;
}

}  // namespace av2enc::cuda::intra_full

#endif  // AV2ENC_CUDA_INTRA_FULL_INTRA_FULL_H_

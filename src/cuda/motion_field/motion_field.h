// AV2 v1.0.0 temporal motion field — CUDA launcher API.
//
// Scope:
//   §7.22    motion field motion vector storage (mvMethod 0/1, block to
//            8x8 subsampled grid, wedge allow-list, past/future ordering,
//            compression, trajectory gap fill)
//   §7.9     MV projection (get_mv_projection, clamp, sampled positions,
//            block-position checks, trajectory intersection)
//   §7.12.2.7/.8   temporal scan candidate generation
//   §7.12.2.19     the one-move-to-front stack sort
//   §7.12.2.21 / §5.20.2.2  reference-MV bank update / reset / MVP appends
//
// The normative integer arithmetic lives in `motion_field_device.cuh`, a
// `__host__ __device__` header shared by the kernels in `motion_field.cu`, the
// host-emulation parity test (`motion_field_emulation_test.cpp`) and the real
// GPU test (`motion_field_test.cu`).  Both paths run the *same* transcription,
// so parity comes from shared code, not two independent implementations.
//
// Memory contract (mandatory, one arena):
//   * every device buffer comes from the single persistent
//     av2enc::cuda::Arena; this module never calls cudaMalloc/cudaFree;
//   * required_device_bytes() reports the exact arena need of the persistent
//     buffers, including 256-byte alignment padding;
//   * allocate_checked() allocates once; an undersized arena returns
//     MfStatus::kArenaTooSmall and leaves the bump pointer untouched;
//   * launchers allocate nothing, copy nothing and never synchronise.  The
//     only copies are explicit upload_* setup calls;
//   * motion_field_mark()/motion_field_rewind() expose the arena watermark so
//     a caller can reuse a transient tail without growth.
//
// Resolution range: any MiCols/MiRows up to the configured maximum, including
// odd MI counts (the last 8x8 cell is clipped) and blocks that overhang the
// right/bottom edge.  No 8/10-bit dependency exists in this component: all
// values are integer MVs and order hints.

#ifndef AV2ENC_CUDA_MOTION_FIELD_MOTION_FIELD_H_
#define AV2ENC_CUDA_MOTION_FIELD_MOTION_FIELD_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include "cuda/arena.h"
#include "encoder/inter/temporal/temporal_types.h"

namespace av2enc::cuda::motion_field {

using inter::temporal::Mv;
using inter::temporal::MvPair;
using inter::temporal::RefMvBank;
using inter::temporal::RefMvBankEntry;
using inter::temporal::RefMvBankRowSource;
using inter::temporal::StackMv;

// Largest capacity accepted for any count-like configuration field.  2^26 is
// far above an 8K 4x4 grid (8.3e6) and keeps the byte math inside 64 bits.
inline constexpr int kMfMaxCount = 1 << 26;

// ---------------------------------------------------------------------------
// Views.  These mirror the CPU POD views in temporal_types.h but hold device
// pointers.  Their layouts are intentionally identical so a caller can build
// one from the other without copying.
// ---------------------------------------------------------------------------

// §7.9 motion field: [y8][x8] with mv[2], validity and MotionFieldOffset.
struct MfField {
  int w8 = 0;
  int h8 = 0;
  std::int32_t* mv = nullptr;      // [y8][x8][2]
  std::uint8_t* valid = nullptr;   // [y8][x8]
  std::int32_t* offset = nullptr;  // [y8][x8]
};

// §7.9.2 trajectory arrays.  `pos` is [k][rf][y8][x8][2] with k in 0..2.
struct MfTraj {
  int w8 = 0;
  int h8 = 0;
  int num_refs = 0;
  std::int32_t* mv = nullptr;         // [rf][y8][x8][2], clipped
  std::uint8_t* valid = nullptr;      // [rf][y8][x8]
  std::int16_t* pos = nullptr;        // [k][rf][y8][x8][2]
  std::uint8_t* pos_valid = nullptr;  // [k][rf][y8][x8]
};

// §7.22 storage output for one reference frame.
struct MfSavedField {
  int w8 = 0;
  int h8 = 0;
  std::int16_t* mv = nullptr;   // [y8][x8][list][comp], compressed
  std::int8_t* ref = nullptr;   // [y8][x8][list]
};

// ---------------------------------------------------------------------------
// §7.22 block storage input.  One descriptor per coded block.  The launcher
// fans the descriptors out to one thread per block; each thread writes only
// its own 8x8 cells, so block grids must be disjoint (a partition of the
// frame), exactly like the CPU driver's call order.
// ---------------------------------------------------------------------------

// Motion-vector source for mvMethod 0/1.  `mvs` is the absolute mode-info
// grid.  `refine_mvs` / `mv_deltas` are block-local arrays with `delta_stride4`
// 4x4 columns: entry (i4, j4, list, comp) lives at
// ((i4 * delta_stride4 + j4) * 4 + list * 2 + comp).
struct MfMvSource {
  const std::int32_t* mvs = nullptr;         // [mi_rows][mi_cols][2][2]
  const std::int32_t* refine_mvs = nullptr;  // block-local
  const std::int32_t* mv_deltas = nullptr;   // block-local
  int mi_cols = 0;
  int mi_rows = 0;
  int delta_stride4 = 0;
  int motion_mode = 0;       // >= LOCALWARP (2) selects the warp MV
  int force_integer_mv = 0;
  int is_global_mv_cand = 0;
  int use_refinemv = 0;
};

struct MfBlockStoreInput {
  int r = 0;  // MI row origin
  int c = 0;  // MI column origin
  int bw4 = 1;
  int bh4 = 1;
  int mi_cols = 0;
  int mi_rows = 0;
  int enable_ref_frame_mvs = 1;
  int mv_method = 0;  // §7.22 mvMethod: 0 = base MVs, 1 = add MvDeltas
  int refs[2] = {-1, -1};  // kRefNone
  int ref_order_hint[2] = {-1, -1};  // kRestrictedOrderHint
  int order_hint = 0;
  int is_wedge = 0;
  const std::uint8_t* mask = nullptr;  // 64x64 luma wedge mask
  int mask_stride = 0;
  // MV sources.
  MfMvSource src{};
  // mvMethod 0 special paths: the caller supplies the already-computed warp
  // and global MVs (this module does not duplicate §7.12.2.2 or the warp
  // estimator).  They are ignored unless the matching flag selects them.
  Mv warp_mv[2]{};
  Mv global_mv[2]{};
  // TIP prediction path (refs[0] == TIP_FRAME).
  int tip_pred = 0;
  int tip_frame_as_output = 0;
  int tip_global_wtd_index = 0;
  int closest_past = -1;  // kRefNone
  int closest_future = -1;  // kRefNone
  int tip_past_order_hint = -1;  // kRestrictedOrderHint
  int tip_future_order_hint = -1;
  int num_past_refs = 0;
  int num_future_refs = 0;
  // Embedded by value (device pointers inside); a pointer member would not be
  // valid after the descriptor is uploaded to the device.
  MfField field{};
  const int* mi_row_base = nullptr;  // [mi_rows][mi_cols]
  const int* mi_col_base = nullptr;
  const int* tip_sizes16 = nullptr;  // [mi_rows][mi_cols]
};

// ---------------------------------------------------------------------------
// §7.9 projection job.  One job is one entry of the CPU process list; jobs are
// applied in array order.  Serialising a job inside one thread is deliberate:
// the spec's replace rule is order-dependent when two source cells project to
// the same destination, so a cell-parallel kernel would not be bit-exact.
// ---------------------------------------------------------------------------

struct MfProjectionParams {
  int mi_cols = 0;
  int mi_rows = 0;
  int proj_step = 1;
  int sb_64x64 = 1;
  int enable_mv_traj = 0;
  int num_total_refs = 0;
  int order_hint = 0;  // OrderHint of the current frame
  int ref_order_hint[inter::temporal::kRefsPerFrame] = {
      -1, -1, -1, -1, -1, -1, -1};  // kRestrictedOrderHint
  int src = -1;             // frame index of the projected reference
  int src_order_hint = 0;   // OrderHints[src]
  int num_saved = 0;        // RefNumTotalRefs[src]
  int saved_order_hint[inter::temporal::kRefsPerFrame] = {
      -1, -1, -1, -1, -1, -1, -1};  // kRestrictedOrderHint
  int dst_sign = 1;
  int is_bwd = 0;
  int target_frame = -1;  // kRefNone
  MfSavedField saved{};
  MfField field{};
  MfTraj traj{};
};

// ---------------------------------------------------------------------------
// §7.12.2 temporal candidate + ref-MV bank block work item.
// ---------------------------------------------------------------------------

struct MfStackBlockInput {
  int mi_row = 0;
  int mi_col = 0;
  int mi_cols = 0;
  int mi_rows = 0;
  int bw4 = 1;
  int bh4 = 1;
  int tile_mi_col_start = 0;
  int tile_mi_col_end = 0;
  int tile_mi_row_start = 0;
  int tile_mi_row_end = 0;
  int ref0 = -1;  // kRefNone
  int ref1 = -1;
  int ref0_order_hint = -1;  // kRestrictedOrderHint
  int ref1_order_hint = -1;
  int order_hint = 0;
  int is_compound = 0;
  int use_traj = 0;
  int num_refs = 0;
  int use_temporal = 0;
  MfField field{};
  MfTraj traj{};  // embedded by value; zero pointers mean "unused"
  // §7.12.2.21 bank append.
  int use_refmvbank = 0;
  int use_intrabc = 0;
  int max_drl_bits_minus_1 = 0;
  // §7.12.2.19 sort over [0, num_nearest).
  int reorder = 0;
  int num_nearest = 0;
  // Input stack (from the earlier spatial scan).
  StackMv stack[inter::temporal::kMaxRefMvStackSize]{};
  int num_found = 0;
  int prune_count = 0;
};

// §7.12.2.21 update/reset operations, applied in array order by one thread.
struct MfBankOp {
  enum Kind : std::int32_t {
    kReset = 0,
    kUpdate = 1,
    kCount = 2,
  };
  std::int32_t kind = kUpdate;
  int mi_row = 0;
  int mi_col = 0;
  int bw4 = 1;
  int bh4 = 1;
  int sb_size4 = 16;
  int chroma_part = 0;
  int frame_is_intra = 0;
  int top_row = 0;
  int refs[2] = {-1, -1};  // kRefNone
  Mv mv[2]{};
  int cwp_idx = 8;  // kCwpEqual
  int from_within_sb = 0;
  int row_offset = 0;  // into the device RefMvBankRowSource array
  int row_mi_cols = 0;
};

// ---------------------------------------------------------------------------
// Configuration, statuses and buffers.
// ---------------------------------------------------------------------------

struct MfConfig {
  int mi_cols = 0;   // MiCols
  int mi_rows = 0;   // MiRows
  int num_refs = 0;  // trajectory reference count; 0 disables trajectory arrays
  int traj_fill_refs = 0;  // refs passed to §7.10.5; 0 = num_refs
  int proj_step = 1;  // ProjStep used by traj_fill (1 or 2)
  int sb_64x64 = 1;   // SbSize == BLOCK_64X64 used by traj_fill
  int max_blocks = 1;              // §7.22 descriptors per store launch
  int max_stack_blocks = 1;        // §7.12.2 merge-stack work items per launch
  int max_projection_jobs = 1;     // §7.9 jobs per call
  int max_bank_ops = 1;            // §7.12.2.21 operations per call
  int max_bank_rows = 1;           // RefMvBankRowSource entries (reset rows)
};

// Device bytes needed by MfConfig, including Arena::kAlignment padding.
// Returns 0 for an invalid configuration.
std::size_t required_device_bytes(const MfConfig& config) noexcept;

enum class MfStatus : int {
  kOk = 0,
  kInvalidArgument,
  kNotInitialized,
  kArenaTooSmall,
  kCudaError,
};

const char* mf_status_name(MfStatus status) noexcept;

struct MfDeviceBuffers {
  MfField field{};
  MfTraj traj{};
  MfSavedField saved{};
  int* saved_stamp = nullptr;  // [y8][x8] last-writer stamps for ordered storage
  MfBlockStoreInput* blocks = nullptr;
  MfStackBlockInput* stack_blocks = nullptr;
  StackMv* stack_out = nullptr;  // [max_stack_blocks][kMaxRefMvStackSize]
  int* stack_num_found = nullptr;
  int* stack_prune_count = nullptr;
  MfProjectionParams* projection_jobs = nullptr;
  RefMvBank* bank = nullptr;
  MfBankOp* bank_ops = nullptr;
  RefMvBankRowSource* bank_rows = nullptr;
  MfConfig config{};
};

// Allocates every persistent buffer once.  On kArenaTooSmall *required_bytes
// receives the exact need and the arena bump pointer is unchanged.
MfStatus allocate_checked(Arena& arena, const MfConfig& config,
                          MfDeviceBuffers* out,
                          std::size_t* required_bytes = nullptr,
                          std::string* error = nullptr);

// Arena watermark wrappers.  mark() is stable across calls; rewind() returns
// false for a mark newer than the current bump pointer.
std::uint64_t motion_field_mark(const Arena& arena) noexcept;
bool motion_field_rewind(Arena& arena, std::uint64_t mark) noexcept;

// Setup copies into the persistent buffers.  These are the only copies in the
// module and are never called by a launcher.
MfStatus upload_blocks(const MfDeviceBuffers& buffers,
                       const MfBlockStoreInput* host_blocks, int count,
                       cudaStream_t stream = nullptr,
                       std::string* error = nullptr);
MfStatus upload_stack_blocks(const MfDeviceBuffers& buffers,
                             const MfStackBlockInput* host_blocks, int count,
                             cudaStream_t stream = nullptr,
                             std::string* error = nullptr);
MfStatus upload_projection_jobs(const MfDeviceBuffers& buffers,
                                const MfProjectionParams* host_jobs, int count,
                                cudaStream_t stream = nullptr,
                                std::string* error = nullptr);
MfStatus upload_bank(const MfDeviceBuffers& buffers, const RefMvBank& bank,
                     cudaStream_t stream = nullptr,
                     std::string* error = nullptr);
MfStatus upload_bank_ops(const MfDeviceBuffers& buffers,
                         const MfBankOp* host_ops, int count,
                         const RefMvBankRowSource* host_rows, int row_count,
                         cudaStream_t stream = nullptr,
                         std::string* error = nullptr);

// Zero the persistent motion field (mv/valid/offset) and, when allocated, the
// trajectory arrays.  Matches the CPU driver's initialisation.
MfStatus zero_field(const MfDeviceBuffers& buffers,
                    cudaStream_t stream = nullptr,
                    std::string* error = nullptr);

// §7.22: writes `buffers.saved`.  Storage is two ordered passes (stamp then
// write, one thread per descriptor) because odd-origin 4x4 blocks can share an
// 8x8 cell; the highest descriptor index wins, matching CPU call order.
MfStatus store_blocks(const MfDeviceBuffers& buffers, int block_count,
                      cudaStream_t stream = nullptr,
                      std::string* error = nullptr);

// §7.9: applies `count` projection jobs in order, one thread per job.
MfStatus project(const MfDeviceBuffers& buffers, int job_count,
                 cudaStream_t stream = nullptr, std::string* error = nullptr);

// §7.9 trajectory gap fill (proj_step == 2 only; a no-op otherwise).
MfStatus traj_fill(const MfDeviceBuffers& buffers,
                   cudaStream_t stream = nullptr,
                   std::string* error = nullptr);

// §7.12.2.7/.8 temporal scan, §7.12.2.21 bank append, §7.12.2.19 sort:
// one thread per work item.  `bank` may be null when no item uses the bank.
MfStatus merge_stacks(const MfDeviceBuffers& buffers, int block_count,
                      cudaStream_t stream = nullptr,
                      std::string* error = nullptr);

// §5.20.2.2 / §7.12.2.21 reset/update/count operations, one thread total so
// the normative ring order is preserved.  Writes `buffers.bank`.
MfStatus apply_bank_ops(const MfDeviceBuffers& buffers, int op_count,
                        cudaStream_t stream = nullptr,
                        std::string* error = nullptr);

}  // namespace av2enc::cuda::motion_field

#endif  // AV2ENC_CUDA_MOTION_FIELD_MOTION_FIELD_H_

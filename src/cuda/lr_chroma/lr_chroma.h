// AV2 chroma loop restoration — CUDA pixel application.
//
// Scope
// -----
// Applies the per-unit chroma non-separable Wiener filter of §7.20.3 using the
// 18-coefficient chroma layout (§5.20.10.6): 12 chroma tap positions plus 12
// chroma-from-luma positions, subsets 0/1/2, `uv_sym` pair mirroring, and
// bank/merged filter selection.  Supports 8/10-bit, 4:2:0/4:2:2/4:4:4 plane
// geometry, LoopRestorationSize 32/64/128/256 chroma samples, and the
// §7.20.1 stripe/tile sample-sourcing rules.
//
// The per-unit decision stays on the CPU (`src/encoder/filters/lr_chroma/`);
// this module only expands the coded unit descriptors and filters pixels.
// `lr_chroma_device.cuh` is the shared arithmetic used by the kernels and by
// the host-emulation parity test.
//
// Memory contract (mandatory)
// ---------------------------
// * Every device byte comes from one caller-owned `av2enc::cuda::Arena`.
// * `required_device_bytes(config)` is the exact persistent footprint, with
//   every buffer rounded up to Arena::kAlignment (256 B).
// * `allocate_checked()` is the only allocating entry point.  On a short arena
//   it returns `kArenaTooSmall` (== `Status::kArenaExhausted`), reports the
//   exact requirement, leaves the output workspace empty, and rewinds the
//   arena so `used()` is unchanged.
// * The launchers (`resolve_units`, `apply`) allocate nothing, copy nothing
//   and never synchronise.  `upload_units` is a setup-time H2D copy.
// * One workspace is single-stream; serialise calls, as the Arena contract
//   already requires.
//
// Outputs never alias inputs: `apply` rejects `dst == cdef` for either chroma
// plane.  Reads come from CurrFrame/CdefFrame, never from LrFrame.

#ifndef AV2ENC_CUDA_LR_CHROMA_LR_CHROMA_H_
#define AV2ENC_CUDA_LR_CHROMA_LR_CHROMA_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include "cuda/arena.h"
#include "cuda/error.h"
#include "lr_chroma_types.h"

namespace av2enc::cuda::lr_chroma {

// Name used by this component for an undersized arena.  It is the runtime's
// existing Status value, so callers can compare against either name.
inline constexpr Status kArenaTooSmall = Status::kArenaExhausted;

// Exact persistent device bytes for `config`.  Returns 0 for an invalid
// configuration (non-positive MI grid, unsupported subsampling/bit depth/unit
// size, or a size_t overflow).  The result is the sum of 256-byte-aligned
// buffers: two unit-descriptor arrays, two expanded-coefficient arrays, two
// unit-active arrays and one bank-state array.
std::size_t required_device_bytes(const LrChromaConfig& config) noexcept;

// Convenience predicates used by validation and tests.
bool config_is_valid(const LrChromaConfig& config) noexcept;

// Explicit mark/rewind for a chroma-LR frame.  These are thin pass-throughs
// to Arena::mark()/Arena::rewind(); transient caller scratch (e.g. a device
// copy of a decision grid) can be released at the frame boundary.  The
// workspace itself is persistent and is only released by rewinding past its
// allocation mark.
inline std::uint64_t frame_mark(const Arena& arena) noexcept {
  return arena.mark();
}
inline bool frame_rewind(Arena& arena, std::uint64_t mark) noexcept {
  return arena.rewind(mark);
}

// Persistent per-frame workspace.  Does not own the arena; reset() only drops
// the pointers.
class LrChromaWorkspace {
 public:
  LrChromaWorkspace() = default;
  LrChromaWorkspace(const LrChromaWorkspace&) = delete;
  LrChromaWorkspace& operator=(const LrChromaWorkspace&) = delete;
  ~LrChromaWorkspace() = default;

  // See allocate_checked() below.
  Status create(Arena& arena, const LrChromaConfig& config,
                std::string* error = nullptr);

  void reset() noexcept;

  bool initialized() const noexcept { return units_[0] != nullptr; }
  const LrChromaConfig& config() const noexcept { return config_; }
  std::size_t required_bytes() const noexcept { return required_bytes_; }
  std::size_t persistent_bytes() const noexcept { return persistent_bytes_; }

  int unit_cols(int chroma) const noexcept { return unit_cols_[chroma]; }
  int unit_rows(int chroma) const noexcept { return unit_rows_[chroma]; }
  int unit_count(int chroma) const noexcept { return unit_count_[chroma]; }

  // Caller-writable device arrays (one descriptor per unit, row-major).
  LrChromaUnit* units(int chroma) noexcept { return units_[chroma]; }
  const LrChromaUnit* units(int chroma) const noexcept {
    return units_[chroma];
  }
  // Filled by resolve_units(): count*18 expanded coefficients per plane.
  const std::int16_t* resolved(int chroma) const noexcept {
    return resolved_[chroma];
  }
  // Filled by resolve_units(): one byte per unit (1 = filter, 0 = copy).
  const std::uint8_t* active(int chroma) const noexcept {
    return active_[chroma];
  }
  std::int16_t* resolved(int chroma) noexcept { return resolved_[chroma]; }
  std::uint8_t* active(int chroma) noexcept { return active_[chroma]; }
  // Two-entry device bank-state array (diagnostics/tests; resolve_units owns
  // the update).
  LrChromaBankState* bank_state() noexcept { return banks_; }
  const LrChromaBankState* bank_state() const noexcept { return banks_; }

 private:
  friend Status allocate_checked(Arena&, const LrChromaConfig&,
                                 LrChromaWorkspace*, std::size_t*,
                                 std::string*);
  void clear() noexcept;

  LrChromaConfig config_{};
  std::size_t required_bytes_ = 0;
  std::size_t persistent_bytes_ = 0;
  int unit_cols_[2] = {0, 0};
  int unit_rows_[2] = {0, 0};
  int unit_count_[2] = {0, 0};
  LrChromaUnit* units_[2] = {nullptr, nullptr};
  std::int16_t* resolved_[2] = {nullptr, nullptr};
  std::uint8_t* active_[2] = {nullptr, nullptr};
  LrChromaBankState* banks_ = nullptr;
};

// Allocate the workspace once from `arena`.  On success the arena's bump
// pointer has advanced by exactly required_device_bytes(config).
//
// On a short arena: returns kArenaTooSmall, writes required_device_bytes to
// *required_bytes when non-null, leaves *ws empty (reset), and rewinds the
// arena to its entry mark.  Never partially initialises the workspace.
Status allocate_checked(Arena& arena, const LrChromaConfig& config,
                        LrChromaWorkspace* ws,
                        std::size_t* required_bytes = nullptr,
                        std::string* error = nullptr);

// Setup-time host-to-device copy of `count == ws.unit_count(chroma)`
// descriptors.  `chroma` is 0 for U and 1 for V.  Zero count with a null host
// pointer is a no-op.  Does not synchronise.
Status upload_units(LrChromaWorkspace& ws, int chroma,
                    const LrChromaUnit* host_units, int count,
                    cudaStream_t stream = nullptr,
                    std::string* error = nullptr);

// Expand both planes' unit streams (§5.20.10.6) into the workspace's resolved
// coefficient and active arrays.  One tiny kernel; unit counts are a few
// thousand at most.  No allocation or synchronisation.
Status resolve_units(LrChromaWorkspace& ws, cudaStream_t stream = nullptr,
                     std::string* error = nullptr);

// Apply the resolved filters: copy CdefFrame -> LrFrame and filter units whose
// active flag is set.  One thread per 4x4 luma MI block per chroma plane.  The
// frame geometry/bit depth/subsampling/CfL index must match the workspace
// configuration.  The resolved arrays are read as-is; call resolve_units()
// first (on the same stream).
Status apply(const LrChromaWorkspace& ws, const LrChromaFrame& frame,
             cudaStream_t stream = nullptr, std::string* error = nullptr);

// Plane extents derived from a configuration (chroma_plane = 0 for U/V).
int plane_width(const LrChromaConfig& config) noexcept;
int plane_height(const LrChromaConfig& config) noexcept;
int unit_cols_for(const LrChromaConfig& config) noexcept;
int unit_rows_for(const LrChromaConfig& config) noexcept;

}  // namespace av2enc::cuda::lr_chroma

#endif  // AV2ENC_CUDA_LR_CHROMA_LR_CHROMA_H_

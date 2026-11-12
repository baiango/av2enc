// Host-side helpers for the CUDA lossless inter component: the pure §5.18.2 /
// §5.20.6.1 mirrors and the persistent arena workspace.
//
// This translation unit performs no device allocation.  `Workspace::init()`
// bump-allocates its persistent staging arrays from the caller's
// `av2enc::cuda::Arena` exactly once; every launcher afterwards only uploads
// descriptors and launches kernels.

#include "lossless_inter.h"

#include "lossless_inter_math.h"

#include <cstdint>
#include <string>

namespace av2enc::cuda::lossless_inter {
namespace {

std::uint64_t round_up(std::uint64_t value, std::uint64_t alignment) noexcept {
  const std::uint64_t mask = alignment - 1;
  return (value + mask) & ~mask;
}

template <typename T>
std::uint64_t array_bytes(int count) noexcept {
  return round_up(static_cast<std::uint64_t>(sizeof(T)) *
                      static_cast<std::uint64_t>(count),
                  static_cast<std::uint64_t>(Arena::kAlignment));
}

}  // namespace

FrameLossless derive_lossless(const FrameQuantInputs& inputs) {
  return li_derive_lossless(inputs);
}

bool find_tx_size(int coded_w, int coded_h, int* tx_w, int* tx_h) {
  if (tx_w == nullptr || tx_h == nullptr) return false;
  return li_find_tx_size(coded_w, coded_h, tx_w, tx_h);
}

TxSizeDecision select_tx_size(const TxSizeRequest& request) {
  return li_select_tx_size(request);
}

std::uint64_t required_device_bytes(const LaunchConfig& config) {
  if (config.max_jobs <= 0) return 0;
  std::uint64_t total = 0;
  // Order and sizes must match Workspace::init() exactly; the tests assert
  // that arena.used() after init equals this value.
  total += array_bytes<WhtJob>(config.max_jobs);
  total += array_bytes<ForwardJob>(config.max_jobs);
  total += array_bytes<InverseJob>(config.max_jobs);
  total += array_bytes<DpcmJob>(config.max_jobs);
  total += array_bytes<ReconJob>(config.max_jobs);
  total += array_bytes<TxSizeRequest>(config.max_jobs);
  total += array_bytes<TxSizeDecision>(config.max_jobs);
  total += array_bytes<FrameQuantInputs>(config.max_jobs);
  total += array_bytes<FrameLossless>(config.max_jobs);
  total += round_up(static_cast<std::uint64_t>(config.max_jobs),
                    static_cast<std::uint64_t>(Arena::kAlignment));
  return total;
}

void Workspace::reset() noexcept {
  wht_jobs = nullptr;
  forward_jobs = nullptr;
  inverse_jobs = nullptr;
  dpcm_jobs = nullptr;
  recon_jobs = nullptr;
  tx_requests = nullptr;
  tx_decisions = nullptr;
  quant_inputs = nullptr;
  lossless = nullptr;
  recon_status = nullptr;
  max_jobs = 0;
  max_w = 0;
  max_h = 0;
  bytes = 0;
}

Status Workspace::init(Arena& arena, const LaunchConfig& config,
                       std::uint64_t* required_bytes, std::string* error) {
  const std::uint64_t required = required_device_bytes(config);
  if (required_bytes != nullptr) *required_bytes = required;
  reset();

  if (config.max_jobs <= 0) {
    if (error != nullptr) *error = "lossless_inter workspace: max_jobs must be > 0";
    return Status::kInvalidArgument;
  }
  if (config.max_block_w <= 0 || config.max_block_h <= 0 ||
      config.max_block_w > kMaxTransformDim ||
      config.max_block_h > kMaxTransformDim) {
    if (error != nullptr) {
      *error = "lossless_inter workspace: block extent must be in [1, 32]";
    }
    return Status::kInvalidArgument;
  }
  if (!arena.initialized()) {
    if (error != nullptr) *error = "lossless_inter workspace: arena not initialized";
    return Status::kNotInitialized;
  }
  if (required > arena.remaining()) {
    if (error != nullptr) {
      *error = "lossless_inter workspace: need " + std::to_string(required) +
               " bytes, arena has " + std::to_string(arena.remaining());
    }
    return Status::kArenaExhausted;
  }

  const std::size_t jobs = static_cast<std::size_t>(config.max_jobs);
  const std::uint64_t mark = arena.mark();
  // Keep this order in sync with required_device_bytes().
  wht_jobs = arena.allocate_array<WhtJob>(jobs);
  forward_jobs = arena.allocate_array<ForwardJob>(jobs);
  inverse_jobs = arena.allocate_array<InverseJob>(jobs);
  dpcm_jobs = arena.allocate_array<DpcmJob>(jobs);
  recon_jobs = arena.allocate_array<ReconJob>(jobs);
  tx_requests = arena.allocate_array<TxSizeRequest>(jobs);
  tx_decisions = arena.allocate_array<TxSizeDecision>(jobs);
  quant_inputs = arena.allocate_array<FrameQuantInputs>(jobs);
  lossless = arena.allocate_array<FrameLossless>(jobs);
  recon_status = arena.allocate_array<std::uint8_t>(jobs);

  if (wht_jobs == nullptr || forward_jobs == nullptr ||
      inverse_jobs == nullptr || dpcm_jobs == nullptr ||
      recon_jobs == nullptr || tx_requests == nullptr ||
      tx_decisions == nullptr || quant_inputs == nullptr ||
      lossless == nullptr || recon_status == nullptr) {
    (void)arena.rewind(mark);
    reset();
    if (error != nullptr) {
      *error = "lossless_inter workspace: arena allocation failed after the "
               "size check";
    }
    return Status::kArenaExhausted;
  }

  max_jobs = config.max_jobs;
  max_w = config.max_block_w;
  max_h = config.max_block_h;
  bytes = required;
  return Status::kOk;
}

}  // namespace av2enc::cuda::lossless_inter

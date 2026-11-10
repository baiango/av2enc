// Status codes and CUDA error helpers for the av2enc CUDA runtime layer.
//
// The runtime layer never throws and never calls exit(). Functions return a
// Status; setup functions also accept an optional std::string* for a
// human-readable reason. Device code is unaffected: these are host-only
// helpers.
//
// cudaErrorNotReady is a normal result of a non-blocking stream query, so it
// maps to Status::kBusy rather than to a failure.
//
// Everything here is inline so the runtime has no extra translation unit.
#ifndef AV2ENC_CUDA_ERROR_H_
#define AV2ENC_CUDA_ERROR_H_

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <string>

namespace av2enc::cuda {

enum class Status : int {
  kOk = 0,
  kInvalidArgument,
  kNotInitialized,
  kAlreadyInitialized,
  kDeviceUnavailable,
  kOutOfMemory,
  kCapExceeded,
  kArenaExhausted,
  kBusy,
  kCudaError,
};

// Stable short names, suitable for logs. Never null.
inline const char* status_name(Status status) noexcept {
  switch (status) {
    case Status::kOk:
      return "ok";
    case Status::kInvalidArgument:
      return "invalid argument";
    case Status::kNotInitialized:
      return "not initialized";
    case Status::kAlreadyInitialized:
      return "already initialized";
    case Status::kDeviceUnavailable:
      return "device unavailable";
    case Status::kOutOfMemory:
      return "out of memory";
    case Status::kCapExceeded:
      return "VRAM cap exceeded";
    case Status::kArenaExhausted:
      return "arena exhausted";
    case Status::kBusy:
      return "busy";
    case Status::kCudaError:
      return "CUDA error";
  }
  return "unknown status";
}

// cudaGetErrorName() wrapper. Never null.
inline const char* cuda_error_name(cudaError_t error) noexcept {
  const char* name = cudaGetErrorName(error);
  return (name != nullptr) ? name : "cudaErrorUnknown";
}

inline bool is_ok(Status status) noexcept {
  return status == Status::kOk;
}

// Convert a CUDA runtime result to a Status and clear CUDA's sticky per-thread
// error state. Call this with the value returned by a CUDA call:
//
//   const Status s = status_from_cuda(cudaMemcpyAsync(...));
//
// cudaSuccess -> kOk, cudaErrorNotReady -> kBusy. Memory-allocation and
// device-selection failures get dedicated statuses so callers can react.
inline Status status_from_cuda(cudaError_t error) noexcept {
  if (error == cudaSuccess) {
    return Status::kOk;
  }
  (void)cudaGetLastError();  // clear the sticky per-thread error

  switch (error) {
    case cudaErrorMemoryAllocation:
      return Status::kOutOfMemory;
    case cudaErrorInvalidDevice:
    case cudaErrorNoDevice:
    case cudaErrorInsufficientDriver:
    case cudaErrorInitializationError:
      return Status::kDeviceUnavailable;
    case cudaErrorInvalidValue:
    case cudaErrorInvalidConfiguration:
    case cudaErrorInvalidResourceHandle:
    case cudaErrorLaunchOutOfResources:
      return Status::kInvalidArgument;
    case cudaErrorNotReady:
      return Status::kBusy;
    default:
      return Status::kCudaError;
  }
}

// Read and clear the sticky per-thread CUDA error. kOk when there is none.
// Use after a batch of asynchronous launches to catch launch-time failures.
inline Status last_error() noexcept {
  return status_from_cuda(cudaGetLastError());
}

// Format "kCudaError: invalid argument" (or just the status name when there is
// no CUDA detail) into buf, NUL-terminated. Returns buf. If buf is null or
// size is 0, returns buf unchanged.
inline const char* format_status(Status status, cudaError_t cuda_error, char* buf,
                                 std::size_t size) noexcept {
  if (buf == nullptr || size == 0) {
    return buf;
  }
  if (cuda_error == cudaSuccess) {
    std::snprintf(buf, size, "%s", status_name(status));
  } else {
    std::snprintf(buf, size, "%s: %s", status_name(status), cudaGetErrorString(cuda_error));
  }
  return buf;
}

}  // namespace av2enc::cuda

#endif  // AV2ENC_CUDA_ERROR_H_

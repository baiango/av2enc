// Status codes for the multi-GPU runtime layer.
//
// This header is deliberately CUDA-free so the CPU-only build and the mocked
// dispatch gate compile on a machine without nvcc. The CUDA backend translates
// cudaError_t into these values; nothing here includes cuda_runtime.h.
#pragma once

namespace av2enc::multigpu::runtime {

enum class Status : int {
  kOk = 0,
  kInvalidArgument,
  kNotInitialized,
  kAlreadyInitialized,
  kBusy,
  kDeviceUnavailable,
  kDeviceMismatch,
  kNoDevices,
  kOutOfMemory,
  kCapExceeded,
  kArenaExhausted,
  kPlanInvalid,
  kDispatchFailed,
  kFrameAborted,
  kBackendError,
};

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
    case Status::kBusy:
      return "busy";
    case Status::kDeviceUnavailable:
      return "device unavailable";
    case Status::kDeviceMismatch:
      return "device mismatch";
    case Status::kNoDevices:
      return "no devices";
    case Status::kOutOfMemory:
      return "out of memory";
    case Status::kCapExceeded:
      return "VRAM cap exceeded";
    case Status::kArenaExhausted:
      return "arena exhausted";
    case Status::kPlanInvalid:
      return "plan invalid";
    case Status::kDispatchFailed:
      return "dispatch failed";
    case Status::kFrameAborted:
      return "frame aborted";
    case Status::kBackendError:
      return "backend error";
  }
  return "unknown status";
}

inline bool is_ok(Status status) noexcept { return status == Status::kOk; }

}  // namespace av2enc::multigpu::runtime

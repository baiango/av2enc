// CUDA implementation of DeviceBackend for the multi-GPU runtime.
//
// This header is CUDA-free. Only cuda_backend.cpp includes cuda_runtime.h, and
// the runtime build script compiles it only when nvcc is available. On a
// CPU-only host the runtime links the mock backend instead.
#pragma once

#include <memory>
#include <string>

#include "device_backend.h"

namespace av2enc::multigpu::runtime {

// Returns nullptr and sets *error when the CUDA runtime cannot be initialized.
std::unique_ptr<DeviceBackend> make_cuda_backend(std::string* error = nullptr);

}  // namespace av2enc::multigpu::runtime

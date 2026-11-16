// Assembles a Backend from a GPU device's launchers (backend/opencl, backend/cuda).
#ifndef GUETZLI_BACKEND_GPU_GPU_BACKEND_H_
#define GUETZLI_BACKEND_GPU_GPU_BACKEND_H_

#include <memory>

#include "backend/gpu/gpu_block_zeroing.h"
#include "backend/gpu/gpu_comparator.h"
#include "backend/gpu/gpu_dct_ops.h"
#include "backend/gpu/gpu_kernels.h"
#include "core/encoder/backend.h"

namespace guetzli {

// Every part shares the device; it lives as long as the longest-lived part.
[[nodiscard]] inline Backend makeGpuBackend(BackendKind kind, const std::shared_ptr<GpuKernels>& device) {
  Backend backend;
  backend.kind = kind;
  backend.comparators = std::make_unique<GpuComparatorFactory>(device);
  backend.block_zeroing = std::make_unique<GpuBlockZeroingSearch>(device);
  backend.dct = std::make_unique<GpuDctOps>(device);
  return backend;
}

}  // namespace guetzli

#endif  // GUETZLI_BACKEND_GPU_GPU_BACKEND_H_

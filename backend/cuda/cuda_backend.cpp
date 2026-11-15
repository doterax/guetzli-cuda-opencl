#ifdef __USE_CUDA__

#include "backend/cuda/cuda_backend.h"

#include <memory>

#include "backend/gpu/gpu_dct_ops.h"
#include "clguetzli/cuguetzli.h"

namespace guetzli {

Backend makeCudaBackend() {
  Backend backend;
  backend.kind = BackendKind::Cuda;
  backend.dct = std::make_unique<GpuDctOps>(
      GpuDctKernels{cuCopyFromJpegComponent, cuApplyGlobalQuantization, cuComponentsToPixels});
  return backend;
}

}  // namespace guetzli

#endif  // __USE_CUDA__

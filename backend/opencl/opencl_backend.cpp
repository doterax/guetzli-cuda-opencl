#ifdef __USE_OPENCL__

#include "backend/opencl/opencl_backend.h"

#include <memory>

#include "backend/gpu/gpu_dct_ops.h"
#include "clguetzli/clguetzli.h"

namespace guetzli {

Backend makeOpenClBackend() {
  Backend backend;
  backend.kind = BackendKind::OpenCl;
  backend.dct = std::make_unique<GpuDctOps>(
      GpuDctKernels{clCopyFromJpegComponent, clApplyGlobalQuantization, clComponentsToPixels});
  return backend;
}

}  // namespace guetzli

#endif  // __USE_OPENCL__

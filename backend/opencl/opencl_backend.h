// The OpenCL backend (only defined when built with __USE_OPENCL__).
#ifndef GUETZLI_BACKEND_OPENCL_OPENCL_BACKEND_H_
#define GUETZLI_BACKEND_OPENCL_OPENCL_BACKEND_H_

#include "guetzli/backend.h"

namespace guetzli {

[[nodiscard]] Backend makeOpenClBackend();

}  // namespace guetzli

#endif  // GUETZLI_BACKEND_OPENCL_OPENCL_BACKEND_H_

// The CUDA backend (only defined when built with __USE_CUDA__).
#ifndef GUETZLI_BACKEND_CUDA_CUDA_BACKEND_H_
#define GUETZLI_BACKEND_CUDA_CUDA_BACKEND_H_

#include "guetzli/backend.h"

namespace guetzli {

[[nodiscard]] Backend makeCudaBackend();

}  // namespace guetzli

#endif  // GUETZLI_BACKEND_CUDA_CUDA_BACKEND_H_

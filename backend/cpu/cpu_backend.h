// The optimized CPU backend.
#ifndef GUETZLI_BACKEND_CPU_CPU_BACKEND_H_
#define GUETZLI_BACKEND_CPU_CPU_BACKEND_H_

#include "core/encoder/backend.h"

namespace guetzli {

[[nodiscard]] Backend makeCpuBackend();

}  // namespace guetzli

#endif  // GUETZLI_BACKEND_CPU_CPU_BACKEND_H_

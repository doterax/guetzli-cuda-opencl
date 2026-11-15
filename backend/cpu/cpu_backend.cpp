#include "backend/cpu/cpu_backend.h"

#include <memory>

namespace guetzli {

Backend makeCpuBackend() {
  Backend backend;
  backend.kind = BackendKind::Cpu;
  backend.dct = std::make_unique<CpuDctOps>();
  return backend;
}

}  // namespace guetzli

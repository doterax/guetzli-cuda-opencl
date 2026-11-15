#include "backend/cpu/cpu_backend.h"

#include <memory>

#include "backend/cpu/cpu_comparator.h"

namespace guetzli {

Backend makeCpuBackend() {
  Backend backend;
  backend.kind = BackendKind::Cpu;
  backend.comparators = std::make_unique<CpuComparatorFactory>();
  backend.block_zeroing = std::make_unique<CpuBlockZeroingSearch>();
  backend.dct = std::make_unique<CpuDctOps>();
  return backend;
}

}  // namespace guetzli

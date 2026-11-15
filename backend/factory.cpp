#include "backend/factory.h"

#include <memory>
#include <stdexcept>
#include <utility>

#include "backend/check/checking_comparator.h"
#include "backend/check/checking_dct_ops.h"
#include "backend/cpu/cpu_backend.h"
#include "backend/cuda/cuda_backend.h"
#include "backend/opencl/opencl_backend.h"
#include "clguetzli/gpu_error.h"
#include "clguetzli/gpu_options.h"
#ifdef __USE_OPENCL__
#include "clguetzli/ocl.h"
#endif
#ifdef __USE_CUDA__
#include "clguetzli/ocu.h"
#endif

namespace guetzli {

BackendProbes compiledInProbes() {
  BackendProbes probes;
#ifdef __USE_CUDA__
  probes.cuda = [] { return supportsCuda(); };
#endif
#ifdef __USE_OPENCL__
  probes.opencl = [] { return supportsOpenCl(); };
#endif
  return probes;
}

BackendKind resolveBackendKind(BackendKind requested, const BackendProbes& probes, ILogger& logger) {
  if (requested != BackendKind::Auto) return requested;

  logger.log(LogLevel::Info, "Autodetect:\n");
  if (probes.cuda) {
    logger.log(LogLevel::Info, "  Try CUDA...\n");
    if (probes.cuda()) {
      logger.log(LogLevel::Info, "  CUDA detected.\n");
      return BackendKind::Cuda;
    }
    logger.log(LogLevel::Info, "    CUDA is not supported.\n");
  }
  if (probes.opencl) {
    logger.log(LogLevel::Info, "  Try OpenCL...\n");
    if (probes.opencl()) {
      logger.log(LogLevel::Info, "  OpenCL detected.\n");
      return BackendKind::OpenCl;
    }
    logger.log(LogLevel::Info, "    OpenCL is not supported.\n");
  }
  logger.log(LogLevel::Info, "  Using optimized CPU implementation (--c).\n");
  return BackendKind::Cpu;
}

void configureGpuBackends(const BackendOptions& options, ILogger& logger) {
  GpuOptions gpu;
  gpu.device = options.device.value_or(-1);
  gpu.verbose = logger.enabled(LogLevel::Debug);
#ifdef __USE_OPENCL__
  configureOpenCl(gpu);
#endif
#ifdef __USE_CUDA__
  configureCuda(gpu);
#endif
  (void)gpu;
}

namespace {

Backend makeGpuBackend(BackendKind kind) {
  switch (kind) {
    case BackendKind::OpenCl:
#ifdef __USE_OPENCL__
      return makeOpenClBackend();
#else
      throw GpuError("OpenCL support is not compiled in");
#endif
    case BackendKind::Cuda:
#ifdef __USE_CUDA__
      return makeCudaBackend();
#else
      throw GpuError("CUDA support is not compiled in");
#endif
    case BackendKind::Cpu:
    case BackendKind::Auto:
      break;
  }
  throw std::logic_error("makeGpuBackend: not a GPU backend");
}

// Every capability runs on `tested` and on `reference`; the reference result is used.
Backend checkedAgainst(Backend tested, Backend reference, ILogger& logger) {
  Backend backend;
  backend.kind = tested.kind;
  backend.comparators = std::make_unique<CheckingComparatorFactory>(std::move(tested.comparators),
                                                                    std::move(reference.comparators), logger);
  backend.dct = std::make_unique<CheckingDctOps>(std::move(tested.dct), std::move(reference.dct), logger);
  return backend;
}

}  // namespace

Backend makeBackend(BackendKind kind, bool check_against_cpu, ILogger& logger) {
  if (kind == BackendKind::Auto) throw std::logic_error("makeBackend: resolve Auto first");
  if (kind == BackendKind::Cpu) return makeCpuBackend();
  Backend gpu = makeGpuBackend(kind);
  if (!check_against_cpu) return gpu;
  return checkedAgainst(std::move(gpu), makeCpuBackend(), logger);
}

}  // namespace guetzli

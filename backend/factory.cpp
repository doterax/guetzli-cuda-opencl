#include "backend/factory.h"

#include <memory>
#include <stdexcept>
#include <utility>

#include "backend/check/checking_block_zeroing.h"
#include "backend/check/checking_comparator.h"
#include "backend/check/checking_dct_ops.h"
#include "backend/cpu/cpu_backend.h"
#include "backend/gpu/gpu_backend.h"
#include "backend/gpu/gpu_options.h"
#include "backend/gpu_error.h"
#ifdef __USE_OPENCL__
#include "backend/opencl/opencl_context.h"
#include "backend/opencl/opencl_kernels.h"
#endif
#ifdef __USE_CUDA__
#include "backend/cuda/cuda_context.h"
#include "backend/cuda/cuda_kernels.h"
#endif

namespace guetzli {

struct GpuDevices::Devices {
#ifdef __USE_OPENCL__
  std::shared_ptr<OpenClKernels> opencl;
#endif
#ifdef __USE_CUDA__
  std::shared_ptr<CudaKernels> cuda;
#endif
};

GpuDevices::GpuDevices(const BackendOptions& options, ILogger& logger) : devices_(std::make_unique<Devices>()) {
  GpuOptions gpu;
  gpu.device = options.device.value_or(-1);
#ifdef __USE_OPENCL__
  devices_->opencl = std::make_shared<OpenClKernels>(gpu, logger);
#endif
#ifdef __USE_CUDA__
  devices_->cuda = std::make_shared<CudaKernels>(gpu, logger);
#endif
  (void)gpu;
  (void)logger;
}

GpuDevices::~GpuDevices() = default;

// Uses devices_ when a GPU backend is built.
// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
BackendProbes GpuDevices::probes() {
  BackendProbes probes;
#ifdef __USE_CUDA__
  probes.cuda = [device = devices_->cuda] { return device->probe(); };
#endif
#ifdef __USE_OPENCL__
  probes.opencl = [device = devices_->opencl] { return device->probe(); };
#endif
  return probes;
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
Backend GpuDevices::makeBackend(BackendKind kind) {
  switch (kind) {
    case BackendKind::OpenCl:
#ifdef __USE_OPENCL__
      return makeGpuBackend(BackendKind::OpenCl, devices_->opencl);
#else
      throw GpuError("OpenCL support is not compiled in");
#endif
    case BackendKind::Cuda:
#ifdef __USE_CUDA__
      return makeGpuBackend(BackendKind::Cuda, devices_->cuda);
#else
      throw GpuError("CUDA support is not compiled in");
#endif
    case BackendKind::Cpu:
    case BackendKind::Auto:
      break;
  }
  throw std::logic_error("GpuDevices::makeBackend: not a GPU backend");
}

BackendKind resolveBackendKind(BackendKind requested, const BackendProbes& probes, ILogger& logger) {
  if (requested != BackendKind::Auto) {
    return requested;
  }

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

namespace {

// Every capability runs on `tested` and on `reference`; the reference result is used.
Backend checkedAgainst(Backend tested, Backend reference, ILogger& logger) {
  Backend backend;
  backend.kind = tested.kind;
  backend.comparators = std::make_unique<CheckingComparatorFactory>(std::move(tested.comparators),
                                                                    std::move(reference.comparators), logger);
  backend.block_zeroing = std::make_unique<CheckingBlockZeroingSearch>(std::move(tested.block_zeroing),
                                                                       std::move(reference.block_zeroing), logger);
  backend.dct = std::make_unique<CheckingDctOps>(std::move(tested.dct), std::move(reference.dct), logger);
  return backend;
}

}  // namespace

Backend makeBackend(BackendKind kind, bool check_against_cpu, GpuDevices& devices, ILogger& logger) {
  if (kind == BackendKind::Auto) {
    throw std::logic_error("makeBackend: resolve Auto first");
  }
  if (kind == BackendKind::Cpu) {
    return makeCpuBackend();
  }
  Backend gpu = devices.makeBackend(kind);
  if (!check_against_cpu) {
    return gpu;
  }
  return checkedAgainst(std::move(gpu), makeCpuBackend(), logger);
}

void printGpuDevices() {
#ifdef __USE_CUDA__
  enumCudaDevices();
#endif
#ifdef __USE_OPENCL__
  enumOpenClDevices();
#endif
}

}  // namespace guetzli

// The only place that knows the concrete backends (docs/design/modernization.md §5.5).
#ifndef GUETZLI_BACKEND_FACTORY_H_
#define GUETZLI_BACKEND_FACTORY_H_

#include <functional>

#include "config/config.h"
#include "guetzli/backend.h"
#include "logging/logger.h"

namespace guetzli {

// Device probes used by BackendKind::Auto. An empty function means "not compiled in".
// Each probe initializes the device; a later makeBackend() for that kind reuses it.
struct BackendProbes {
  std::function<bool()> cuda;
  std::function<bool()> opencl;
};

// The probes of the backends compiled into this binary.
[[nodiscard]] BackendProbes compiledInProbes();

// Auto -> CUDA, then OpenCL, then CPU (logged as "Autodetect: ..."); any other kind is returned unchanged.
[[nodiscard]] BackendKind resolveBackendKind(BackendKind requested, const BackendProbes& probes, ILogger& logger);

// Passes --device / --verbose to the GPU layers. Initialization stays lazy: a small image never touches the GPU.
void configureGpuBackends(const BackendOptions& options, ILogger& logger);

// kind must not be Auto. A GPU kind that is not compiled in throws GpuError (so the caller falls back to the CPU).
// With check_against_cpu, every operation also runs on the CPU, mismatches are logged as "CHK ..." errors and the
// CPU result is used.
[[nodiscard]] Backend makeBackend(BackendKind kind, bool check_against_cpu, ILogger& logger);

}  // namespace guetzli

#endif  // GUETZLI_BACKEND_FACTORY_H_

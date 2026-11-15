// The only place that knows the concrete backends (docs/design/modernization.md §5.5).
#ifndef GUETZLI_BACKEND_FACTORY_H_
#define GUETZLI_BACKEND_FACTORY_H_

#include <functional>
#include <memory>

#include "config/config.h"
#include "guetzli/backend.h"
#include "logging/logger.h"

namespace guetzli {

// Device probes used by BackendKind::Auto. An empty function means "not compiled in".
struct BackendProbes {
  std::function<bool()> cuda;
  std::function<bool()> opencl;
};

// The GPU devices of one run (--device applies to whichever backend is chosen), owned by the caller and outliving
// every Backend made from it. A device is initialized on first use, so a small image never touches the GPU; a probe
// initializes it early, and backends made later reuse it. Released when this object is destroyed.
class GpuDevices {
 public:
  GpuDevices(const BackendOptions& options, ILogger& logger);
  ~GpuDevices();
  GpuDevices(const GpuDevices&) = delete;
  GpuDevices& operator=(const GpuDevices&) = delete;

  // The probes of the backends compiled into this binary, bound to these devices.
  [[nodiscard]] BackendProbes probes();

  // kind must be OpenCl or Cuda. Throws GpuError if that backend is not compiled in.
  [[nodiscard]] Backend makeBackend(BackendKind kind);

 private:
  struct Devices;
  std::unique_ptr<Devices> devices_;
};

// Auto -> CUDA, then OpenCL, then CPU (logged as "Autodetect: ..."); any other kind is returned unchanged.
[[nodiscard]] BackendKind resolveBackendKind(BackendKind requested, const BackendProbes& probes, ILogger& logger);

// kind must not be Auto. A GPU kind that is not compiled in throws GpuError (so the caller falls back to the CPU).
// With check_against_cpu, every operation also runs on the CPU, mismatches are logged as "CHK ..." errors and the
// CPU result is used.
[[nodiscard]] Backend makeBackend(BackendKind kind, bool check_against_cpu, GpuDevices& devices, ILogger& logger);

// Prints the CUDA and OpenCL devices with their indices (--version), to stdout.
void printGpuDevices();

}  // namespace guetzli

#endif  // GUETZLI_BACKEND_FACTORY_H_

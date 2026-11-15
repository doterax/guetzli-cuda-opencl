/*
* GPU backend error.
*
* Thrown by the OpenCL / CUDA layers when a device call fails (no device,
* kernel compile/load failure, out of device memory, launch failure, ...).
* main() catches it and re-runs the job on the CPU (MODE_CPU_OPT).
*/
#pragma once

#include <cstdlib>
#include <stdexcept>
#include <string>

class GpuError : public std::runtime_error {
 public:
  explicit GpuError(const std::string& what) : std::runtime_error(what) {}
};

// Test hook: GUETZLI_TEST_GPU_FAIL=1 makes GPU backend initialization fail,
// to exercise the CPU fallback path (see tests/regress/regress.sh).
inline void MaybeInjectGpuFailure(const char* backend) {
  const char* v = getenv("GUETZLI_TEST_GPU_FAIL");
  if (v && v[0] && v[0] != '0') {
    throw GpuError(std::string(backend) + ": failure injected by GUETZLI_TEST_GPU_FAIL");
  }
}

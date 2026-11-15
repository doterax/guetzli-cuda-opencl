// Runtime loading of the CUDA driver API.
#ifdef __USE_CUDA__

#include "backend/cuda/cuda_driver.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace guetzli {

namespace {

#ifdef _WIN32
using LibraryHandle = HMODULE;

LibraryHandle openLibrary() {
  return LoadLibraryA("nvcuda.dll");
}

void* symbol(LibraryHandle lib, const char* name) {
  return reinterpret_cast<void*>(GetProcAddress(lib, name));
}
#else
using LibraryHandle = void*;

LibraryHandle openLibrary() {
#if defined(__APPLE__)
  // macOS: CUDA driver is rarely available; try anyway
  return dlopen("libcuda.dylib", RTLD_LAZY);
#else
  LibraryHandle lib = dlopen("libcuda.so.1", RTLD_LAZY);
  if (!lib) lib = dlopen("libcuda.so", RTLD_LAZY);
  return lib;
#endif
}

void* symbol(LibraryHandle lib, const char* name) {
  return dlsym(lib, name);
}
#endif

// Some driver functions are versioned with a _v2 suffix in the library (cuda.h maps the plain name to it with a
// macro); the _v2 name is tried first. `exact`: the function has no versioned variant.
template <typename Fn>
bool resolve(LibraryHandle lib, Fn& fn, const char* name, bool exact, ILogger& logger) {
  void* p = nullptr;
  if (!exact) p = symbol(lib, (std::string(name) + "_v2").c_str());
  if (!p) p = symbol(lib, name);
  if (!p) {
    logger.log(LogLevel::Error, "CUDA dynload: failed to resolve %s\n", name);
    return false;
  }
  fn = reinterpret_cast<Fn>(p);
  return true;
}

}  // namespace

std::optional<CudaDriver> CudaDriver::load(ILogger& logger) {
  LibraryHandle lib = openLibrary();
  if (!lib) return std::nullopt;

  CudaDriver d;
  const bool ok =
      resolve(lib, d.init, "cuInit", true, logger) && resolve(lib, d.ctxCreate, "cuCtxCreate", false, logger) &&
      resolve(lib, d.ctxDestroy, "cuCtxDestroy", false, logger) &&
      resolve(lib, d.ctxSetCacheConfig, "cuCtxSetCacheConfig", true, logger) &&
      resolve(lib, d.deviceGetAttribute, "cuDeviceGetAttribute", true, logger) &&
      resolve(lib, d.deviceGetCount, "cuDeviceGetCount", true, logger) &&
      resolve(lib, d.deviceGet, "cuDeviceGet", false, logger) &&
      resolve(lib, d.deviceGetName, "cuDeviceGetName", true, logger) &&
      resolve(lib, d.memAlloc, "cuMemAlloc", false, logger) && resolve(lib, d.memFree, "cuMemFree", false, logger) &&
      resolve(lib, d.memcpyDtoH, "cuMemcpyDtoH", false, logger) &&
      resolve(lib, d.memcpyDtoHAsync, "cuMemcpyDtoHAsync", false, logger) &&
      resolve(lib, d.memcpyHtoDAsync, "cuMemcpyHtoDAsync", false, logger) &&
      resolve(lib, d.memcpyDtoD, "cuMemcpyDtoD", false, logger) &&
      resolve(lib, d.memcpyDtoDAsync, "cuMemcpyDtoDAsync", false, logger) &&
      resolve(lib, d.memsetD8Async, "cuMemsetD8Async", true, logger) &&
      resolve(lib, d.moduleLoadDataEx, "cuModuleLoadDataEx", true, logger) &&
      resolve(lib, d.moduleUnload, "cuModuleUnload", true, logger) &&
      resolve(lib, d.moduleGetFunction, "cuModuleGetFunction", true, logger) &&
      resolve(lib, d.streamCreate, "cuStreamCreate", false, logger) &&
      resolve(lib, d.streamSynchronize, "cuStreamSynchronize", true, logger) &&
      resolve(lib, d.streamDestroy, "cuStreamDestroy", false, logger) &&
      resolve(lib, d.launchKernel, "cuLaunchKernel", true, logger);
  if (!ok) return std::nullopt;
  return d;
}

}  // namespace guetzli

#endif  // __USE_CUDA__

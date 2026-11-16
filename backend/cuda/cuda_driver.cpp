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
#ifdef __APPLE__
  // macOS: CUDA driver is rarely available; try anyway
  return dlopen("libcuda.dylib", RTLD_LAZY);
#else
  LibraryHandle lib = dlopen("libcuda.so.1", RTLD_LAZY);
  if (lib == nullptr) {
    lib = dlopen("libcuda.so", RTLD_LAZY);
  }
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
  if (!exact) {
    p = symbol(lib, (std::string(name) + "_v2").c_str());
  }
  if (!p) {
    p = symbol(lib, name);
  }
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
  if (lib == nullptr) {
    return std::nullopt;
  }

  CudaDriver d;
  const bool ok =
      resolve(lib, d.init, "cuInit", true, logger) && resolve(lib, d.ctx_create, "cuCtxCreate", false, logger) &&
      resolve(lib, d.driver_get_version, "cuDriverGetVersion", true, logger) &&
      resolve(lib, d.ctx_destroy, "cuCtxDestroy", false, logger) &&
      resolve(lib, d.ctx_set_cache_config, "cuCtxSetCacheConfig", true, logger) &&
      resolve(lib, d.device_get_attribute, "cuDeviceGetAttribute", true, logger) &&
      resolve(lib, d.device_get_count, "cuDeviceGetCount", true, logger) &&
      resolve(lib, d.device_get, "cuDeviceGet", false, logger) &&
      resolve(lib, d.device_get_name, "cuDeviceGetName", true, logger) &&
      resolve(lib, d.mem_alloc, "cuMemAlloc", false, logger) && resolve(lib, d.mem_free, "cuMemFree", false, logger) &&
      resolve(lib, d.memcpy_dtoh, "cuMemcpyDtoH", false, logger) &&
      resolve(lib, d.memcpy_dtoh_async, "cuMemcpyDtoHAsync", false, logger) &&
      resolve(lib, d.memcpy_htod_async, "cuMemcpyHtoDAsync", false, logger) &&
      resolve(lib, d.memcpy_dtod, "cuMemcpyDtoD", false, logger) &&
      resolve(lib, d.memcpy_dtod_async, "cuMemcpyDtoDAsync", false, logger) &&
      resolve(lib, d.memset_d8_async, "cuMemsetD8Async", true, logger) &&
      resolve(lib, d.module_load_data_ex, "cuModuleLoadDataEx", true, logger) &&
      resolve(lib, d.module_unload, "cuModuleUnload", true, logger) &&
      resolve(lib, d.module_get_function, "cuModuleGetFunction", true, logger) &&
      resolve(lib, d.stream_create, "cuStreamCreate", false, logger) &&
      resolve(lib, d.stream_synchronize, "cuStreamSynchronize", true, logger) &&
      resolve(lib, d.stream_destroy, "cuStreamDestroy", false, logger) &&
      resolve(lib, d.launch_kernel, "cuLaunchKernel", true, logger);
  if (!ok) {
    return std::nullopt;
  }
  return d;
}

}  // namespace guetzli

#endif  // __USE_CUDA__

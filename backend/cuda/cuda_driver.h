// The CUDA driver API (nvcuda.dll / libcuda.so), loaded at runtime.
//
// This avoids a link-time dependency on the driver library, so the binary starts on systems without NVIDIA
// drivers (e.g. AMD-only machines) and falls back to OpenCL or the CPU. cuda.h is still needed for the types.
#ifndef GUETZLI_BACKEND_CUDA_CUDA_DRIVER_H_
#define GUETZLI_BACKEND_CUDA_CUDA_DRIVER_H_

#ifdef __USE_CUDA__

#include <cuda.h>

#include <optional>
#include <string>

#include "logging/logger.h"

namespace guetzli {

// The driver entry points this program uses. A plain value: copying it copies the function pointers.
struct CudaDriver {
  // Loads the driver library and resolves every entry point. Returns nullopt if the library is not installed (silent)
  // or an entry point is missing (logged). The library stays loaded for the rest of the process: unloading the
  // driver while it may still run threads is not safe.
  [[nodiscard]] static std::optional<CudaDriver> load(ILogger& logger);

  CUresult(CUDAAPI* init)(unsigned int Flags) = nullptr;
  CUresult(CUDAAPI* ctxCreate)(CUcontext* pctx, unsigned int flags, CUdevice dev) = nullptr;
  CUresult(CUDAAPI* ctxDestroy)(CUcontext ctx) = nullptr;
  CUresult(CUDAAPI* ctxSetCacheConfig)(CUfunc_cache config) = nullptr;
  CUresult(CUDAAPI* deviceGetAttribute)(int* pi, CUdevice_attribute attrib, CUdevice dev) = nullptr;
  CUresult(CUDAAPI* deviceGetCount)(int* count) = nullptr;
  CUresult(CUDAAPI* deviceGet)(CUdevice* device, int ordinal) = nullptr;
  CUresult(CUDAAPI* deviceGetName)(char* name, int len, CUdevice dev) = nullptr;
  CUresult(CUDAAPI* memAlloc)(CUdeviceptr* dptr, size_t bytesize) = nullptr;
  CUresult(CUDAAPI* memFree)(CUdeviceptr dptr) = nullptr;
  CUresult(CUDAAPI* memcpyDtoH)(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) = nullptr;
  CUresult(CUDAAPI* memcpyDtoHAsync)(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount,
                                     CUstream hStream) = nullptr;
  CUresult(CUDAAPI* memcpyHtoDAsync)(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount,
                                     CUstream hStream) = nullptr;
  CUresult(CUDAAPI* memcpyDtoD)(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) = nullptr;
  CUresult(CUDAAPI* memcpyDtoDAsync)(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount,
                                     CUstream hStream) = nullptr;
  CUresult(CUDAAPI* memsetD8Async)(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream) = nullptr;
  CUresult(CUDAAPI* moduleLoadDataEx)(CUmodule* module, const void* image, unsigned int numOptions,
                                      CUjit_option* options, void** optionValues) = nullptr;
  CUresult(CUDAAPI* moduleUnload)(CUmodule hmod) = nullptr;
  CUresult(CUDAAPI* moduleGetFunction)(CUfunction* hfunc, CUmodule hmod, const char* name) = nullptr;
  CUresult(CUDAAPI* streamCreate)(CUstream* phStream, unsigned int Flags) = nullptr;
  CUresult(CUDAAPI* streamSynchronize)(CUstream hStream) = nullptr;
  CUresult(CUDAAPI* streamDestroy)(CUstream hStream) = nullptr;
  CUresult(CUDAAPI* launchKernel)(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                  unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                  unsigned int sharedMemBytes, CUstream hStream, void** kernelParams,
                                  void** extra) = nullptr;
};

}  // namespace guetzli

#endif  // __USE_CUDA__

#endif  // GUETZLI_BACKEND_CUDA_CUDA_DRIVER_H_

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

  CUresult(CUDAAPI* init)(unsigned int flags) = nullptr;
  CUresult(CUDAAPI* ctx_create)(CUcontext* pctx, unsigned int flags, CUdevice dev) = nullptr;
  CUresult(CUDAAPI* ctx_destroy)(CUcontext ctx) = nullptr;
  CUresult(CUDAAPI* ctx_set_cache_config)(CUfunc_cache config) = nullptr;
  CUresult(CUDAAPI* device_get_attribute)(int* pi, CUdevice_attribute attrib, CUdevice dev) = nullptr;
  CUresult(CUDAAPI* device_get_count)(int* count) = nullptr;
  CUresult(CUDAAPI* device_get)(CUdevice* device, int ordinal) = nullptr;
  CUresult(CUDAAPI* device_get_name)(char* name, int len, CUdevice dev) = nullptr;
  CUresult(CUDAAPI* mem_alloc)(CUdeviceptr* dptr, size_t bytesize) = nullptr;
  CUresult(CUDAAPI* mem_free)(CUdeviceptr dptr) = nullptr;
  CUresult(CUDAAPI* memcpy_dtoh)(void* dst_host, CUdeviceptr src_device, size_t byte_count) = nullptr;
  CUresult(CUDAAPI* memcpy_dtoh_async)(void* dst_host, CUdeviceptr src_device, size_t byte_count,
                                       CUstream h_stream) = nullptr;
  CUresult(CUDAAPI* memcpy_htod_async)(CUdeviceptr dst_device, const void* src_host, size_t byte_count,
                                       CUstream h_stream) = nullptr;
  CUresult(CUDAAPI* memcpy_dtod)(CUdeviceptr dst_device, CUdeviceptr src_device, size_t byte_count) = nullptr;
  CUresult(CUDAAPI* memcpy_dtod_async)(CUdeviceptr dst_device, CUdeviceptr src_device, size_t byte_count,
                                       CUstream h_stream) = nullptr;
  CUresult(CUDAAPI* memset_d8_async)(CUdeviceptr dst_device, unsigned char uc, size_t n, CUstream h_stream) = nullptr;
  CUresult(CUDAAPI* module_load_data_ex)(CUmodule* module, const void* image, unsigned int num_options,
                                         CUjit_option* options, void** option_values) = nullptr;
  CUresult(CUDAAPI* module_unload)(CUmodule hmod) = nullptr;
  CUresult(CUDAAPI* module_get_function)(CUfunction* hfunc, CUmodule hmod, const char* name) = nullptr;
  CUresult(CUDAAPI* stream_create)(CUstream* ph_stream, unsigned int flags) = nullptr;
  CUresult(CUDAAPI* stream_synchronize)(CUstream h_stream) = nullptr;
  CUresult(CUDAAPI* stream_destroy)(CUstream h_stream) = nullptr;
  CUresult(CUDAAPI* launch_kernel)(CUfunction f, unsigned int grid_dim_x, unsigned int grid_dim_y,
                                   unsigned int grid_dim_z, unsigned int block_dim_x, unsigned int block_dim_y,
                                   unsigned int block_dim_z, unsigned int shared_mem_bytes, CUstream h_stream,
                                   void** kernel_params, void** extra) = nullptr;
};

}  // namespace guetzli

#endif  // __USE_CUDA__

#endif  // GUETZLI_BACKEND_CUDA_CUDA_DRIVER_H_

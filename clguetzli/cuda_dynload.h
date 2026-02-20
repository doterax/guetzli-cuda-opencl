/*
 * Dynamic loading of CUDA driver API (nvcuda.dll / libcuda.so)
 *
 * This avoids a hard link-time dependency on the CUDA driver library,
 * allowing the binary to start on systems without NVIDIA drivers
 * (e.g. AMD-only machines) and gracefully fall back to OpenCL or CPU.
 *
 * Usage:
 *   #include <cuda.h>          // still needed for type definitions
 *   #include "cuda_dynload.h"  // redefines API calls to function pointers
 *
 * Call cudaDynloadInit() once at startup. Returns true if nvcuda loaded.
 */
#pragma once

#ifdef __USE_CUDA__

#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --- Function pointer typedefs for every CUDA Driver API function we use --- */

typedef CUresult (CUDAAPI *PFN_cuInit)(unsigned int Flags);
typedef CUresult (CUDAAPI *PFN_cuCtxCreate)(CUcontext *pctx, unsigned int flags, CUdevice dev);
typedef CUresult (CUDAAPI *PFN_cuCtxDestroy)(CUcontext ctx);
typedef CUresult (CUDAAPI *PFN_cuCtxSetCacheConfig)(CUfunc_cache config);
typedef CUresult (CUDAAPI *PFN_cuDeviceGetAttribute)(int *pi, CUdevice_attribute attrib, CUdevice dev);
typedef CUresult (CUDAAPI *PFN_cuDeviceGetName)(char *name, int len, CUdevice dev);
typedef CUresult (CUDAAPI *PFN_cuMemAlloc)(CUdeviceptr *dptr, size_t bytesize);
typedef CUresult (CUDAAPI *PFN_cuMemFree)(CUdeviceptr dptr);
typedef CUresult (CUDAAPI *PFN_cuMemcpyDtoH)(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);
typedef CUresult (CUDAAPI *PFN_cuMemcpyDtoHAsync)(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemcpyHtoDAsync)(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemcpyDtoD)(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount);
typedef CUresult (CUDAAPI *PFN_cuMemcpyDtoDAsync)(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemsetD8Async)(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuModuleLoadDataEx)(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues);
typedef CUresult (CUDAAPI *PFN_cuModuleUnload)(CUmodule hmod);
typedef CUresult (CUDAAPI *PFN_cuModuleGetFunction)(CUfunction *hfunc, CUmodule hmod, const char *name);
typedef CUresult (CUDAAPI *PFN_cuStreamCreate)(CUstream *phStream, unsigned int Flags);
typedef CUresult (CUDAAPI *PFN_cuStreamSynchronize)(CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuLaunchKernel)(CUfunction f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, CUstream hStream,
    void **kernelParams, void **extra);

/* --- Extern function pointer declarations --- */

extern PFN_cuInit                 dynCuInit;
extern PFN_cuCtxCreate            dynCuCtxCreate;
extern PFN_cuCtxDestroy           dynCuCtxDestroy;
extern PFN_cuCtxSetCacheConfig    dynCuCtxSetCacheConfig;
extern PFN_cuDeviceGetAttribute   dynCuDeviceGetAttribute;
extern PFN_cuDeviceGetName        dynCuDeviceGetName;
extern PFN_cuMemAlloc             dynCuMemAlloc;
extern PFN_cuMemFree              dynCuMemFree;
extern PFN_cuMemcpyDtoH          dynCuMemcpyDtoH;
extern PFN_cuMemcpyDtoHAsync     dynCuMemcpyDtoHAsync;
extern PFN_cuMemcpyHtoDAsync     dynCuMemcpyHtoDAsync;
extern PFN_cuMemcpyDtoD          dynCuMemcpyDtoD;
extern PFN_cuMemcpyDtoDAsync     dynCuMemcpyDtoDAsync;
extern PFN_cuMemsetD8Async       dynCuMemsetD8Async;
extern PFN_cuModuleLoadDataEx    dynCuModuleLoadDataEx;
extern PFN_cuModuleUnload        dynCuModuleUnload;
extern PFN_cuModuleGetFunction   dynCuModuleGetFunction;
extern PFN_cuStreamCreate        dynCuStreamCreate;
extern PFN_cuStreamSynchronize   dynCuStreamSynchronize;
extern PFN_cuLaunchKernel        dynCuLaunchKernel;

/* --- Runtime loader --- */

/*
 * Attempt to load nvcuda.dll (Windows) or libcuda.so.1 (Linux/macOS).
 * Returns true if the library was loaded and all function pointers resolved.
 * Safe to call multiple times — subsequent calls return the cached result.
 */
bool cudaDynloadInit();

/* Returns true if cudaDynloadInit() has already succeeded. */
bool cudaDynloadAvailable();

#ifdef __cplusplus
}
#endif

/* --- Redirect standard CUDA API names to our function pointers --- */
/* This allows existing code to call cuInit(...) etc. without changes. */

#undef cuInit
#undef cuCtxCreate
#undef cuCtxDestroy
#undef cuCtxSetCacheConfig
#undef cuDeviceGetAttribute
#undef cuDeviceGetName
#undef cuMemAlloc
#undef cuMemFree
#undef cuMemcpyDtoH
#undef cuMemcpyDtoHAsync
#undef cuMemcpyHtoDAsync
#undef cuMemcpyDtoD
#undef cuMemcpyDtoDAsync
#undef cuMemsetD8Async
#undef cuModuleLoadDataEx
#undef cuModuleUnload
#undef cuModuleGetFunction
#undef cuStreamCreate
#undef cuStreamSynchronize
#undef cuLaunchKernel

#define cuInit                dynCuInit
#define cuCtxCreate           dynCuCtxCreate
#define cuCtxDestroy          dynCuCtxDestroy
#define cuCtxSetCacheConfig   dynCuCtxSetCacheConfig
#define cuDeviceGetAttribute  dynCuDeviceGetAttribute
#define cuDeviceGetName       dynCuDeviceGetName
#define cuMemAlloc            dynCuMemAlloc
#define cuMemFree             dynCuMemFree
#define cuMemcpyDtoH         dynCuMemcpyDtoH
#define cuMemcpyDtoHAsync    dynCuMemcpyDtoHAsync
#define cuMemcpyHtoDAsync    dynCuMemcpyHtoDAsync
#define cuMemcpyDtoD         dynCuMemcpyDtoD
#define cuMemcpyDtoDAsync    dynCuMemcpyDtoDAsync
#define cuMemsetD8Async      dynCuMemsetD8Async
#define cuModuleLoadDataEx   dynCuModuleLoadDataEx
#define cuModuleUnload       dynCuModuleUnload
#define cuModuleGetFunction  dynCuModuleGetFunction
#define cuStreamCreate       dynCuStreamCreate
#define cuStreamSynchronize  dynCuStreamSynchronize
#define cuLaunchKernel       dynCuLaunchKernel

#endif /* __USE_CUDA__ */

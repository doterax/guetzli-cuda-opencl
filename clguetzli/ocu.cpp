/*
* CUDA Manager
*
* Author: strongtu@tencent.com
*/
#include "ocu.h"

#ifdef __USE_CUDA__
#include <cstdio>
#include <cstdint>
#include <string>
#include <cuda.h>
#include "cuda_dynload.h"
#include "clguetzli/clguetzli_cu_ptx.h"
#include "clguetzli/clguetzli.h"
#include "lzodec.h"
#include "guetzli/gamma_correct.h"
#include <cstring>

bool supportsCuda() {
  if (!cudaDynloadInit()) return false;

  // Full initialization of the selected device (honors --device): context,
  // PTX JIT and kernel lookup. The result is reused by later getOcu() calls,
  // so the module is only JIT-compiled once.
  try {
    getOcu();
  } catch (const std::exception& e) {
    LogError("CUDA not usable: %s\n", e.what());
    return false;
  }
  return true;
}

void enumCudaDevices() {
  if (!cudaDynloadInit()) {
    fprintf(stdout, "CUDA:\n  (driver not available)\n\n");
    return;
  }

  CUresult err = cuInit(0);
  if (err != CUDA_SUCCESS) {
    fprintf(stdout, "CUDA:\n  (initialization failed)\n\n");
    return;
  }

  int deviceCount = 0;
  err = cuDeviceGetCount(&deviceCount);
  if (err != CUDA_SUCCESS || deviceCount == 0) {
    fprintf(stdout, "CUDA:\n  (no devices found)\n\n");
    return;
  }

  fprintf(stdout, "CUDA:\n");
  for (int i = 0; i < deviceCount; i++) {
    CUdevice dev;
    err = cuDeviceGet(&dev, i);
    if (err != CUDA_SUCCESS) continue;

    char name[256] = {0};
    cuDeviceGetName(name, sizeof(name), dev);
    fprintf(stdout, "  %d: %s\n", i, name);
  }
  fprintf(stdout, "\n");
}

namespace {
// Owned by the CUDA singleton until contexts become backend-owned objects (modernization phase 4).
GpuOptions& ocuOptions() {
  static GpuOptions options;
  return options;
}
}  // namespace

void configureCuda(const GpuOptions& options) {
  ocuOptions() = options;
}

ocu_args_d_t& getOcu(void) {
  static bool bInit = false;
  static ocu_args_d_t ocu;
  const GpuOptions& options = ocuOptions();

  if (bInit == true) return ocu;

  MaybeInjectGpuFailure("CUDA");

  // Ensure CUDA driver library is loaded (cudaDynloadInit is idempotent).
  // Without this, --cuda flag bypasses autoDetectBestMode() which normally
  // calls supportsCuda() -> cudaDynloadInit(), leaving function pointers
  // uninitialized and causing an access violation (0xC0000005).
  if (!cudaDynloadInit()) {
    throw GpuError("failed to load CUDA driver library");
  }

  // Any failure below throws GpuError; bInit is only set once everything succeeded.
  LOG_CU_RESULT(cuInit(0));
  CUdevice dev = 0;

  // Use user-specified device index if valid
  if (options.device >= 0) {
    int deviceCount = 0;
    LOG_CU_RESULT(cuDeviceGetCount(&deviceCount));
    if (options.device < deviceCount) {
      LOG_CU_RESULT(cuDeviceGet(&dev, options.device));
    } else {
      fprintf(stderr, "Warning: CUDA device %d not found (only %d devices available), using default device 0\n",
              options.device, deviceCount);
      LOG_CU_RESULT(cuDeviceGet(&dev, 0));
    }
  } else {
    LOG_CU_RESULT(cuDeviceGet(&dev, 0));
  }

  if (ocu.ctxt == NULL) {
    LOG_CU_RESULT(cuCtxCreate(&ocu.ctxt, CU_CTX_SCHED_AUTO, dev));
  }
  ocu.dev = dev;

  char name[1024] = {0};
  int proc_count = 0;
  int thread_count = 0;
  int cap_major = 0, cap_minor = 0;
  cuDeviceGetName(name, sizeof(name), dev);
  cuDeviceGetAttribute(&cap_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
  cuDeviceGetAttribute(&cap_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
  cuDeviceGetAttribute(&proc_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
  cuDeviceGetAttribute(&thread_count, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, dev);
  LogError("CUDA Adapter:%s Ver%d.%d MP %d MaxThread Per MP %d)\r\n", name, cap_major, cap_minor, proc_count,
           thread_count);
  if (!proc_count || !thread_count) {
    throw GpuError("CUDA device reports no multiprocessors");
  }

  if (ocu.mod == NULL) {
    LzoDec decompressed(clguetzli_cu64_lzo, sizeof(clguetzli_cu64_lzo));
    if (!decompressed.isValid()) {
      throw GpuError("failed to decompress embedded CUDA PTX");
    }
    // getData() is NUL-terminated, as cuModuleLoadDataEx expects for PTX text.
    const char* ptx = (const char*)decompressed.getData();

    CUjit_option jit_options[1];
    void* jit_optvals[1];
    jit_options[0] = CU_JIT_CACHE_MODE;
    jit_optvals[0] = (void*)(uintptr_t)CU_JIT_CACHE_OPTION_CA;
    LOG_CU_RESULT(cuModuleLoadDataEx(&ocu.mod, ptx, 1, jit_options, jit_optvals));
  }

  static const struct {
    KernelName id;
    const char* name;
  } kKernels[] = {
      {KERNEL_CONVOLUTIONX, "clConvolutionXEx"},
      {KERNEL_CONVOLUTIONY, "clConvolutionYEx"},
      {KERNEL_SQUARESAMPLE, "clSquareSampleEx"},
      {KERNEL_OPSINDYNAMICSIMAGE, "clOpsinDynamicsImageEx"},
      {KERNEL_MASKHIGHINTENSITYCHANGE, "clMaskHighIntensityChangeEx"},
      {KERNEL_EDGEDETECTOR, "clEdgeDetectorMapEx"},
      {KERNEL_BLOCKDIFFMAP, "clBlockDiffMapEx"},
      {KERNEL_EDGEDETECTORLOWFREQ, "clEdgeDetectorLowFreqEx"},
      {KERNEL_DIFFPRECOMPUTE, "clDiffPrecomputeEx"},
      {KERNEL_SCALEIMAGE, "clScaleImageEx"},
      {KERNEL_AVERAGE5X5, "clAverage5x5Ex"},
      {KERNEL_MINSQUAREVAL, "clMinSquareValEx"},
      {KERNEL_DOMASK, "clDoMaskEx"},
      {KERNEL_COMBINECHANNELS, "clCombineChannelsEx"},
      {KERNEL_UPSAMPLESQUAREROOT, "clUpsampleSquareRootEx"},
      {KERNEL_REMOVEBORDER, "clRemoveBorderEx"},
      {KERNEL_ADDBORDER, "clAddBorderEx"},
      {KERNEL_COMPUTEBLOCKZEROINGORDER, "clComputeBlockZeroingOrderEx"},
      {KERNEL_COPYFROMJPEGCOMPONENT, "clCopyFromJpegComponentEx"},
      {KERNEL_APPLYGLOBALQUANTIZATION, "clApplyGlobalQuantizationEx"},
      {KERNEL_COMPONENTSTOPIXELS, "clComponentsToPixels"},
      {KERNEL_COMPONENTSTOPIXELS_EX1, "clComponentsToPixelsEx1"},
      {KERNEL_COMPONENTSTOPIXELS_EX2, "clComponentsToPixelsEx2"},
      {KERNEL_COLORTRANSFORMYCBCRTORGB, "clColorTransformYCbCrToRGB"},
      {KERNEL_SRGBTOLINEAR, "clSrgbToLinearEx"},
      {KERNEL_CONVOLUTIONSCALE, "clConvolutionScaleEx"},
  };
  static_assert(sizeof(kKernels) / sizeof(kKernels[0]) == KERNEL_COUNT, "kernel table out of sync with KernelName");
  for (size_t i = 0; i < sizeof(kKernels) / sizeof(kKernels[0]); i++) {
    CUresult err = cuModuleGetFunction(&ocu.kernel[kKernels[i].id], ocu.mod, kKernels[i].name);
    if (err != CUDA_SUCCESS) {
      LogError("Error: cuModuleGetFunction(%s) returned %s.\n", kKernels[i].name, TranslateCUDAError(err));
      throw GpuError(std::string("CUDA kernel not found: ") + kKernels[i].name);
    }
  }

  cuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_SHARED);
  // cuCtxSetSharedMemConfig is deprecated and no longer needed on modern GPUs
  // as they have a fixed shared memory bank size

  if (ocu.commandQueue == NULL) {
    LOG_CU_RESULT(cuStreamCreate(&ocu.commandQueue, 0));
  }
  ocu.mem_pool.commandQueue = ocu.commandQueue;

  bInit = true;
  return ocu;
}

ocu_args_d_t::ocu_args_d_t() : dev(0), commandQueue(NULL), mod(NULL), ctxt(NULL), srgbLut(0) {
  for (int i = 0; i < 6; i++) maskLut[i] = 0;
}

ocu_args_d_t::~ocu_args_d_t() {
  if (ctxt == NULL) return;  // CUDA never initialized (driver entry points may be NULL)

  // Teardown order: free device memory, unload module, destroy stream, then context.
  for (std::map<unsigned int, cu_mem>::iterator it = blurWeights.begin(); it != blurWeights.end(); ++it)
    cuMemFree(it->second);
  for (int i = 0; i < 6; i++)
    if (maskLut[i]) cuMemFree(maskLut[i]);
  if (srgbLut) cuMemFree(srgbLut);
  for (std::map<std::vector<size_t>, cu_mem>::iterator it = convScales.begin(); it != convScales.end(); ++it)
    cuMemFree(it->second);
  mem_pool.drain();
  if (mod) cuModuleUnload(mod);
  if (commandQueue) cuStreamDestroy(commandQueue);
  cuCtxDestroy(ctxt);
}

cu_mem ocu_args_d_t::allocMem(size_t s, const void* init) {
  return mem_pool.allocMem(s, init);
}

void ocu_args_d_t::releaseMem(cu_mem mem) {
  mem_pool.releaseMem(mem);
}

cu_mem ocu_args_d_t::allocMemUninit(size_t s) {
  return mem_pool.allocMem(s, NULL, false);
}

ocu_channels ocu_args_d_t::allocMemChannelsUninit(size_t s) {
  ocu_channels img;
  for (int i = 0; i < 3; i++) {
    img.ch[i] = allocMemUninit(s);
  }
  return img;
}

cu_mem ocu_args_d_t::uploadConstant(const void* data, size_t s) {
  cu_mem mem = 0;
  CUresult err = cuMemAlloc(&mem, s);
  if (err == CUDA_ERROR_OUT_OF_MEMORY) {
    mem_pool.drain();
    err = cuMemAlloc(&mem, s);
  }
  LOG_CU_RESULT(err);
  // Synchronous: data may be a caller's local.
  err = cuMemcpyHtoDAsync(mem, data, s, commandQueue);
  if (err == CUDA_SUCCESS) err = cuStreamSynchronize(commandQueue);
  if (err != CUDA_SUCCESS) cuMemFree(mem);
  LOG_CU_RESULT(err);
  return mem;
}

cu_mem ocu_args_d_t::getBlurWeights(float sigma, const std::vector<float>& weights) {
  unsigned int key;
  memcpy(&key, &sigma, sizeof(key));
  std::map<unsigned int, cu_mem>::iterator it = blurWeights.find(key);
  if (it != blurWeights.end()) return it->second;

  cu_mem mem = uploadConstant(weights.data(), weights.size() * sizeof(float));
  blurWeights[key] = mem;
  return mem;
}

cu_mem ocu_args_d_t::getMaskLut(int i, const float* lut512) {
  if (!maskLut[i]) maskLut[i] = uploadConstant(lut512, 512 * sizeof(float));
  return maskLut[i];
}

cu_mem ocu_args_d_t::getSrgbToLinearLut() {
  if (srgbLut) return srgbLut;

  // Same values as OutputImage::ToLinearRGB (the double table cast to float).
  const double* lut = guetzli::Srgb8ToLinearTable();
  float table[256];
  for (int i = 0; i < 256; i++) table[i] = static_cast<float>(lut[i]);
  srgbLut = uploadConstant(table, sizeof(table));
  return srgbLut;
}

ocu_channels ocu_args_d_t::allocMemChannels(size_t s, const void* c0, const void* c1, const void* c2) {
  const void* c[3] = {c0, c1, c2};

  ocu_channels img;
  for (int i = 0; i < 3; i++) {
    img.ch[i] = allocMem(s, c[i]);
  }

  return img;
}

void ocu_args_d_t::releaseMemChannels(ocu_channels& rgb) {
  for (int i = 0; i < 3; i++) {
    releaseMem(rgb.ch[i]);
    rgb.ch[i] = NULL;
  }
}

const char* TranslateCUDAError(CUresult errorCode) {
  switch (errorCode) {
    case CUDA_SUCCESS:
      return "CUDA_SUCCESS";
    case CUDA_ERROR_INVALID_VALUE:
      return "CUDA_ERROR_INVALID_VALUE";
    case CUDA_ERROR_OUT_OF_MEMORY:
      return "CUDA_ERROR_OUT_OF_MEMORY";
    case CUDA_ERROR_NOT_INITIALIZED:
      return "CUDA_ERROR_NOT_INITIALIZED";
    case CUDA_ERROR_DEINITIALIZED:
      return "CUDA_ERROR_DEINITIALIZED";
    case CUDA_ERROR_PROFILER_DISABLED:
      return "CUDA_ERROR_PROFILER_DISABLED";
    case CUDA_ERROR_PROFILER_NOT_INITIALIZED:
      return "CUDA_ERROR_PROFILER_NOT_INITIALIZED";
    case CUDA_ERROR_PROFILER_ALREADY_STARTED:
      return "CUDA_ERROR_PROFILER_ALREADY_STARTED";
    case CUDA_ERROR_PROFILER_ALREADY_STOPPED:
      return "CUDA_ERROR_PROFILER_ALREADY_STOPPED";
    case CUDA_ERROR_NO_DEVICE:
      return "CUDA_ERROR_NO_DEVICE";
    case CUDA_ERROR_INVALID_DEVICE:
      return "CUDA_ERROR_INVALID_DEVICE";
    case CUDA_ERROR_INVALID_IMAGE:
      return "CUDA_ERROR_INVALID_IMAGE";
    case CUDA_ERROR_INVALID_CONTEXT:
      return "CUDA_ERROR_INVALID_CONTEXT";
    case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
      return "CUDA_ERROR_CONTEXT_ALREADY_CURRENT";
    case CUDA_ERROR_MAP_FAILED:
      return "CUDA_ERROR_MAP_FAILED";
    case CUDA_ERROR_UNMAP_FAILED:
      return "CUDA_ERROR_UNMAP_FAILED";
    case CUDA_ERROR_ARRAY_IS_MAPPED:
      return "CUDA_ERROR_ARRAY_IS_MAPPED";
    case CUDA_ERROR_ALREADY_MAPPED:
      return "CUDA_ERROR_ALREADY_MAPPED";
    case CUDA_ERROR_NO_BINARY_FOR_GPU:
      return "CUDA_ERROR_NO_BINARY_FOR_GPU";
    case CUDA_ERROR_ALREADY_ACQUIRED:
      return "CUDA_ERROR_ALREADY_ACQUIRED";
    case CUDA_ERROR_NOT_MAPPED:
      return "CUDA_ERROR_NOT_MAPPED";
    case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
      return "CUDA_ERROR_NOT_MAPPED_AS_ARRAY";
    case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
      return "CUDA_ERROR_NOT_MAPPED_AS_POINTER";
    case CUDA_ERROR_ECC_UNCORRECTABLE:
      return "CUDA_ERROR_ECC_UNCORRECTABLE";
    case CUDA_ERROR_UNSUPPORTED_LIMIT:
      return "CUDA_ERROR_UNSUPPORTED_LIMIT";
    case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
      return "CUDA_ERROR_CONTEXT_ALREADY_IN_USE";
    case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED:
      return "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED";
    case CUDA_ERROR_INVALID_PTX:
      return "CUDA_ERROR_INVALID_PTX";
    case CUDA_ERROR_INVALID_GRAPHICS_CONTEXT:
      return "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT";
    // case CUDA_ERROR_NVLINK_UNCORRECTABLE: return "CUDA_ERROR_NVLINK_UNCORRECTABLE";
    case CUDA_ERROR_INVALID_SOURCE:
      return "CUDA_ERROR_INVALID_SOURCE";
    case CUDA_ERROR_FILE_NOT_FOUND:
      return "CUDA_ERROR_FILE_NOT_FOUND";
    case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
      return "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND";
    case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
      return "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED";
    case CUDA_ERROR_OPERATING_SYSTEM:
      return "CUDA_ERROR_OPERATING_SYSTEM";
    case CUDA_ERROR_INVALID_HANDLE:
      return "CUDA_ERROR_INVALID_HANDLE";
    case CUDA_ERROR_NOT_FOUND:
      return "CUDA_ERROR_NOT_FOUND";
    case CUDA_ERROR_NOT_READY:
      return "CUDA_ERROR_NOT_READY";
    case CUDA_ERROR_ILLEGAL_ADDRESS:
      return "CUDA_ERROR_ILLEGAL_ADDRESS";
    case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
      return "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";
    case CUDA_ERROR_LAUNCH_TIMEOUT:
      return "CUDA_ERROR_LAUNCH_TIMEOUT";
    case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
      return "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING";
    case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
      return "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED";
    case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
      return "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED";
    case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
      return "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE";
    case CUDA_ERROR_CONTEXT_IS_DESTROYED:
      return "CUDA_ERROR_CONTEXT_IS_DESTROYED";
    case CUDA_ERROR_ASSERT:
      return "CUDA_ERROR_ASSERT";
    case CUDA_ERROR_TOO_MANY_PEERS:
      return "CUDA_ERROR_TOO_MANY_PEERS";
    case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
      return "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED";
    case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
      return "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED";
    case CUDA_ERROR_HARDWARE_STACK_ERROR:
      return "CUDA_ERROR_HARDWARE_STACK_ERROR";
    case CUDA_ERROR_ILLEGAL_INSTRUCTION:
      return "CUDA_ERROR_ILLEGAL_INSTRUCTION";
    case CUDA_ERROR_MISALIGNED_ADDRESS:
      return "CUDA_ERROR_MISALIGNED_ADDRESS";
    case CUDA_ERROR_INVALID_ADDRESS_SPACE:
      return "CUDA_ERROR_INVALID_ADDRESS_SPACE";
    case CUDA_ERROR_INVALID_PC:
      return "CUDA_ERROR_INVALID_PC";
    case CUDA_ERROR_LAUNCH_FAILED:
      return "CUDA_ERROR_LAUNCH_FAILED";
    case CUDA_ERROR_NOT_PERMITTED:
      return "CUDA_ERROR_NOT_PERMITTED";
    case CUDA_ERROR_NOT_SUPPORTED:
      return "CUDA_ERROR_NOT_SUPPORTED";
    case CUDA_ERROR_UNKNOWN:
      return "CUDA_ERROR_UNKNOWN";
    default:
      return "CUDA_ERROR_UNKNOWN";
  }
}
#endif
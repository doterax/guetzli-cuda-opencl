/*
* CUDA Manager
*
* Author: strongtu@tencent.com
*/
#pragma once

#ifdef __USE_CUDA__

#include <map>
#include <vector>
#include <cuda.h>
#include "cuda_dynload.h"
#include "ocl.h"
#include "cumem_pool.h"
#include "gpu_options.h"

// Evaluates e once; on failure logs and throws GpuError (main() falls back to CPU).
#define LOG_CU_RESULT(e)                                                                                       \
  do {                                                                                                         \
    const CUresult _cu_res = (e);                                                                              \
    if (CUDA_SUCCESS != _cu_res) {                                                                             \
      LogError("Error: %s:%d returned %s.\n", __FUNCTION__, __LINE__, TranslateCUDAError(_cu_res));            \
      throw GpuError(std::string("CUDA call failed in ") + __FUNCTION__ + ": " + TranslateCUDAError(_cu_res)); \
    }                                                                                                          \
  } while (0)

struct ocu_args_d_t;

const char* TranslateCUDAError(CUresult errorCode);

// Device selection for the (lazy) CUDA initialization; call before first use.
void configureCuda(const GpuOptions& options);
bool supportsCuda();
void enumCudaDevices();
ocu_args_d_t& getOcu(void);

struct ocu_args_d_t {
  ocu_args_d_t();
  ~ocu_args_d_t();

  cu_mem allocMem(size_t s, const void* init = NULL);
  void releaseMem(cu_mem mem);
  ocu_channels allocMemChannels(size_t s, const void* c0 = NULL, const void* c1 = NULL, const void* c2 = NULL);
  void releaseMemChannels(ocu_channels& rgb);
  // Contents undefined: only for buffers that are fully overwritten before being read.
  cu_mem allocMemUninit(size_t s);
  ocu_channels allocMemChannelsUninit(size_t s);

  // Device-resident constant tables, uploaded once and kept for the context's lifetime.
  cu_mem getBlurWeights(float sigma, const std::vector<float>& weights);
  cu_mem getMaskLut(int i, const float* lut512);  // six 512-entry DoMask LUTs
  cu_mem getSrgbToLinearLut();                    // float of guetzli::Srgb8ToLinearTable()
  // Convolution scale tables (clConvolutionScaleEx), keyed by {weights buffer, n, step,
  // border_ratio bits}; filled by cuguetzli.cpp, freed with the context.
  std::map<std::vector<size_t>, cu_mem> convScales;

  CUfunction kernel[KERNEL_COUNT];
  CUstream commandQueue;
  CUmodule mod;
  CUcontext ctxt;
  CUdevice dev;
  cu_mem_pool_t mem_pool;

 private:
  cu_mem uploadConstant(const void* data, size_t s);

  std::map<unsigned int, cu_mem> blurWeights;  // keyed by sigma's bit pattern
  cu_mem maskLut[6];
  cu_mem srgbLut;
};

#endif
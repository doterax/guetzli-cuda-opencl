/*
* OpenCL Manager
*
* Author: strongtu@tencent.com
*         ianhuang@tencent.com
*/
#pragma once

enum KernelName {
  KERNEL_CONVOLUTION = 0,
  KERNEL_CONVOLUTIONX,
  KERNEL_CONVOLUTIONY,
  KERNEL_SQUARESAMPLE,
  KERNEL_OPSINDYNAMICSIMAGE,
  KERNEL_MASKHIGHINTENSITYCHANGE,
  KERNEL_EDGEDETECTOR,
  KERNEL_BLOCKDIFFMAP,
  KERNEL_EDGEDETECTORLOWFREQ,
  KERNEL_DIFFPRECOMPUTE,
  KERNEL_SCALEIMAGE,
  KERNEL_AVERAGE5X5,
  KERNEL_MINSQUAREVAL,
  KERNEL_DOMASK,
  KERNEL_COMBINECHANNELS,
  KERNEL_UPSAMPLESQUAREROOT,
  KERNEL_REMOVEBORDER,
  KERNEL_ADDBORDER,
  KERNEL_COMPUTEBLOCKZEROINGORDER,
  KERNEL_COPYFROMJPEGCOMPONENT,
  KERNEL_APPLYGLOBALQUANTIZATION,
  KERNEL_COMPONENTSTOPIXELS,
  KERNEL_COMPONENTSTOPIXELS_EX1,
  KERNEL_COMPONENTSTOPIXELS_EX2,
  KERNEL_COLORTRANSFORMYCBCRTORGB,
  KERNEL_SRGBTOLINEAR,
  KERNEL_CONVOLUTIONSCALE,
  KERNEL_COUNT,
};

#include "utils.h"
#include "gpu_error.h"
#include "clguetzli.cl.h"

#ifdef __USE_OPENCL__
#include <map>
#include <vector>
#include "third_party/OpenCL/include/CL/cl.h"

// Evaluates e once; on failure logs and throws GpuError (main() falls back to CPU).
#define LOG_CL_RESULT(e)                                                                                           \
  do {                                                                                                             \
    const cl_int _cl_res = (e);                                                                                    \
    if (CL_SUCCESS != _cl_res) {                                                                                   \
      LogError("Error: %s:%d returned %s.\n", __FUNCTION__, __LINE__, TranslateOpenCLError(_cl_res));              \
      throw GpuError(std::string("OpenCL call failed in ") + __FUNCTION__ + ": " + TranslateOpenCLError(_cl_res)); \
    }                                                                                                              \
  } while (0)

struct ocl_args_d_t;
class Device;

const char* TranslateOpenCLError(cl_int errorCode);

bool supportsOpenCl();
void enumOpenClDevices();

ocl_args_d_t& getOcl();

struct ocl_args_d_t {
  ocl_args_d_t();
  ~ocl_args_d_t();

  // init != NULL: contents are uploaded from init. init == NULL: zero-filled.
  cl_mem allocMem(size_t s, const void* init = NULL);
  ocl_channels allocMemChannels(size_t s, const void* c0 = NULL, const void* c1 = NULL, const void* c2 = NULL);
  // Contents undefined: only for buffers that are fully overwritten before being read.
  cl_mem allocMemUninit(size_t s);
  ocl_channels allocMemChannelsUninit(size_t s);
  // Returns the buffer to the pool (buffers are reused by size; the in-order
  // queue orders any later use after the commands already enqueued on it).
  void releaseMem(cl_mem mem);
  void releaseMemChannels(ocl_channels& rgb);
  // Frees all idle pooled buffers.
  void drainPool();

  // Device-resident constant tables, uploaded once and kept for the context's lifetime.
  // Gaussian blur weights for sigma (computed by the caller on first use).
  cl_mem getBlurWeights(float sigma, const std::vector<float>& weights);
  // Six 512-entry DoMask LUTs (x, y, b, dc_x, dc_y, dc_b).
  cl_mem getMaskLut(int i, const float* lut512);
  // 256-entry sRGB8 -> linear table (float of guetzli::Srgb8ToLinearTable()).
  cl_mem getSrgbToLinearLut();
  // Convolution scale tables (clConvolutionScaleEx), keyed by {weights buffer, n, step,
  // border_ratio bits}; filled by clguetzli.cpp, freed with the context.
  std::map<std::vector<size_t>, cl_mem> convScales;

  // Regular OpenCL objects:
  cl_context context;              // hold the context handler
  cl_command_queue commandQueue;   // hold the commands-queue handler
  cl_program program;              // hold the program handler
  cl_kernel kernel[KERNEL_COUNT];  // hold the kernel handler
  bool isAmd;
  Device* device;

  int allocations;

 private:
  cl_mem createBuffer(size_t s);

  std::map<size_t, std::vector<cl_mem>> freeBuffers;  // idle pooled buffers by size
  std::map<cl_mem, size_t> bufferSizes;               // every pooled buffer (busy or idle) -> size
  std::map<unsigned int, cl_mem> blurWeights;         // keyed by sigma's bit pattern
  cl_mem maskLut[6];
  cl_mem srgbLut;
};

#endif

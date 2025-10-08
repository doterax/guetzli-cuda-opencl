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
	KERNEL_COUNT,
};

#include "utils.h"
#include "clguetzli.cl.h"

#ifdef __USE_OPENCL__

#include "third_party/OpenCL/include/CL/cl.h"

#define LOG_CL_RESULT(e)   if (CL_SUCCESS != (e)) { LogError("Error: %s:%d returned %s.\n", __FUNCTION__, __LINE__, TranslateOpenCLError((e))); std::abort();}

struct ocl_args_d_t;
class Device;

const char* TranslateOpenCLError(cl_int errorCode);

bool supportsOpenCl();

ocl_args_d_t& getOcl();

struct ocl_args_d_t
{
	ocl_args_d_t();
	~ocl_args_d_t();

	cl_mem allocMem(size_t s, const void *init = NULL);
	ocl_channels allocMemChannels(size_t s, const void *c0 = NULL, const void *c1 = NULL, const void *c2 = NULL);
    void releaseMemChannels(ocl_channels &rgb);

	// Regular OpenCL objects:
	cl_context       context;           // hold the context handler
	cl_command_queue commandQueue;      // hold the commands-queue handler
	cl_program       program;           // hold the program handler
	cl_kernel        kernel[KERNEL_COUNT];            // hold the kernel handler
	bool             isAmd;
	Device*          device;
};

#endif

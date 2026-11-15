/*
* CUDA Manager
*
* Author: strongtu@tencent.com
*/
#pragma once

#ifdef __USE_CUDA__

#include <cuda.h>
#include "cuda_dynload.h"
#include "ocl.h"
#include "cumem_pool.h"

// Evaluates e once; on failure logs and throws GpuError (main() falls back to CPU).
#define LOG_CU_RESULT(e) do { \
		const CUresult _cu_res = (e); \
		if (CUDA_SUCCESS != _cu_res) { \
			LogError("Error: %s:%d returned %s.\n", __FUNCTION__, __LINE__, TranslateCUDAError(_cu_res)); \
			throw GpuError(std::string("CUDA call failed in ") + __FUNCTION__ + ": " + TranslateCUDAError(_cu_res)); \
		} \
	} while (0)

struct ocu_args_d_t;

const char* TranslateCUDAError(CUresult errorCode);

bool supportsCuda();
void enumCudaDevices();
ocu_args_d_t& getOcu(void);

struct ocu_args_d_t
{
    ocu_args_d_t();
    ~ocu_args_d_t();

    cu_mem allocMem(size_t s, const void *init = NULL);
    void releaseMem(cu_mem mem);
    ocu_channels allocMemChannels(size_t s, const void *c0 = NULL, const void *c1 = NULL, const void *c2 = NULL);
    void releaseMemChannels(ocu_channels &rgb);

    CUfunction  kernel[KERNEL_COUNT];
    CUstream    commandQueue;
    CUmodule    mod;
    CUcontext   ctxt;
    CUdevice    dev;
    cu_mem_pool_t mem_pool;
};



#endif
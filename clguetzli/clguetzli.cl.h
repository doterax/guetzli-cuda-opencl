/*
* OpenCL/CUDA edition implementation of ButteraugliComparator.
*
* Author: strongtu@tencent.com
*         ianhuang@tencent.com
*         chriskzhou@tencent.com
*         stephendeng@tencent.com
*/
#ifndef __CLGUETZLI_CL_H__
#define __CLGUETZLI_CL_H__

#if defined(__USE_OPENCL__) || defined(__USE_CUDA__)

#if defined(__cplusplus) && !defined(__CUDACC__)

#ifdef __USE_OPENCL__
#include "third_party/OpenCL/include/CL/cl.h"
#endif

#ifdef __USE_CUDA__
#include "cuda.h"
typedef CUdeviceptr cu_mem;
#endif

#endif/*__cplusplus && !__CUDACC__*/

#define __USE_DOUBLE_AS_FLOAT__

#if defined(__cplusplus) && !defined(__CUDACC__)
    #define __kernel
    #define __private
    #define __global
    #define __constant
    #define __constant_ex
    #define __device__

    typedef unsigned char uchar;
    typedef unsigned short ushort;

    int get_global_id(int dim);
    int get_global_size(int dim);
    void set_global_id(int dim, int id);
    void set_global_size(int dim, int size);

    #ifdef __checkcl
        typedef union ocl_channels_t
        {
            struct
            {
                float * r;
                float * g;
                float * b;
            };
            union
            {
                float *ch[3];
            };
        }ocl_channels;

        typedef union ocu_channels_t
        {
            struct
            {
                float * r;
                float * g;
                float * b;
            };
            union
            {
                float *ch[3];
            };
        }ocu_channels;
    #else
#ifdef __USE_OPENCL__
        typedef union ocl_channels_t
        {
            struct
            {
                cl_mem r;
                cl_mem g;
                cl_mem b;
            };
            struct
            {
                cl_mem x;
                cl_mem y;
                cl_mem b_;
            };
            union
            {
                cl_mem ch[3];
            };
        }ocl_channels;
#endif
#ifdef __USE_CUDA__
        typedef union ocu_channels_t
        {
            struct
            {
                cu_mem r;
                cu_mem g;
                cu_mem b;
            };
            struct
            {
                cu_mem x;
                cu_mem y;
                cu_mem b_;
            };
            union
            {
                cu_mem ch[3];
            };
        }ocu_channels;
#endif
    #endif /*__checkcl*/
#endif /*__cplusplus && !__CUDACC__*/

#ifdef __OPENCL_VERSION__
    #define __constant_ex __constant
    #define __device__

#endif /*__OPENCL_VERSION__*/

#ifdef __CUDACC__
    #define __kernel    extern "C" __global__
    #define __private
    #define __global
    #define __constant  __constant__
    #define __constant_ex
    typedef unsigned char uchar;
    typedef unsigned short ushort;

    __device__ int get_global_id(int dim)
    {
        switch (dim)
        {
        case 0:  return blockIdx.x * blockDim.x + threadIdx.x;
        case 1:  return blockIdx.y * blockDim.y + threadIdx.y;
        default: return blockIdx.z * blockDim.z + threadIdx.z;
        }
    }

    __device__ int get_global_size(int dim)
    {
        switch(dim)
        {
        case 0: return gridDim.x * blockDim.x;
        case 1: return gridDim.y * blockDim.y;
        default: return gridDim.z * blockDim.z;
        }
    }
	
#endif /*__CUDACC__*/

    typedef short coeff_t;

    typedef struct __channel_info_t
    {
        int factor;
        int block_width;
        int block_height;
        __global const coeff_t *coeff;
        __global const ushort  *pixel;
    }channel_info;

#endif /*__CLGUETZLI_CL_H__*/

#endif // __USE_OPENCL__
/*
 * Host-side types shared with the kernels in kernels/guetzli.cl.
 *
 * Host code only: the kernel source carries its own copy of ChannelInfo.
 * The two struct layouts must stay identical (it is passed to kernels by value).
 *
 * Author: strongtu@tencent.com
 *         ianhuang@tencent.com
 *         chriskzhou@tencent.com
 *         stephendeng@tencent.com
 */
#ifndef __CLGUETZLI_CL_H__
#define __CLGUETZLI_CL_H__

#ifdef __USE_OPENCL__
#include "third_party/OpenCL/include/CL/cl.h"
#endif

#ifdef __USE_CUDA__
#include "cuda.h"
typedef CUdeviceptr cu_mem;
#endif

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef short coeff_t;

#ifdef __USE_OPENCL__
union OclChannels {
  struct {
    cl_mem r;
    cl_mem g;
    cl_mem b;
  };
  struct {
    cl_mem x;
    cl_mem y;
  };
  union {
    cl_mem ch[3];
  };
};
#endif

#ifdef __USE_CUDA__
union OcuChannels {
  struct {
    cu_mem r;
    cu_mem g;
    cu_mem b;
  };
  struct {
    cu_mem x;
    cu_mem y;
  };
  union {
    cu_mem ch[3];
  };
};
#endif

#pragma pack(push, 1)
struct ChannelInfo {
  int factor;
  int block_width;
  int block_height;
  const coeff_t* coeff;
  const ushort* pixel;
};
#pragma pack(pop)

#endif /*__CLGUETZLI_CL_H__*/

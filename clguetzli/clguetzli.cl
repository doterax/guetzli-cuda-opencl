#ifndef __CL_GUETZLI_CU__
#ifndef __USE_OPENCL__
#define __USE_OPENCL__
#endif
#endif
/*
 * OpenCL Kernels
 *
 * Author: strongtu@tencent.com
 *         ianhuang@tencent.com
 *         chriskzhou@tencent.com
 *         stephendeng@tencent.com
 */
#if defined(__USE_OPENCL__) || defined(__USE_CUDA__)

// #include  "clguetzli/clguetzli.cl.h"
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
#include "CL/cl.h"
#endif

#ifdef __USE_CUDA__
#include "cuda.h"
typedef CUdeviceptr cu_mem;
#endif

#endif /*__cplusplus && !__CUDACC__*/

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
		float *r;
		float *g;
		float *b;
	};
	union
	{
		float *ch[3];
	};
} ocl_channels;

typedef union ocu_channels_t
{
	struct
	{
		float *r;
		float *g;
		float *b;
	};
	union
	{
		float *ch[3];
	};
} ocu_channels;
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
} ocl_channels;
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
} ocu_channels;
#endif
#endif /*__checkcl*/
#endif /*__cplusplus && !__CUDACC__*/

#ifdef __OPENCL_VERSION__
#define __constant_ex __constant
#define __device__

#endif /*__OPENCL_VERSION__*/

#ifdef __CUDACC__
#define __kernel extern "C" __global__
#define __private
#define __global
#define __constant __constant__
#define __constant_ex
typedef unsigned char uchar;
typedef unsigned short ushort;

__device__ int get_global_id(int dim)
{
	switch (dim)
	{
	case 0:
		return blockIdx.x * blockDim.x + threadIdx.x;
	case 1:
		return blockIdx.y * blockDim.y + threadIdx.y;
	default:
		return blockIdx.z * blockDim.z + threadIdx.z;
	}
}

__device__ int get_global_size(int dim)
{
	switch (dim)
	{
	case 0:
		return gridDim.x * blockDim.x;
	case 1:
		return gridDim.y * blockDim.y;
	default:
		return gridDim.z * blockDim.z;
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
	__global const ushort *pixel;
} channel_info;

#endif /*__CLGUETZLI_CL_H__*/

#endif // __USE_OPENCL__


#define kBlockEdge 8
#define kBlockSize (kBlockEdge * kBlockEdge)
#define kDCTBlockSize (kBlockEdge * kBlockEdge)
#define kBlockEdgeHalf (kBlockEdge / 2)
#define kBlockHalf (kBlockEdge * kBlockEdgeHalf)
#define kComputeBlockSize (kBlockSize * 3)

// IntFloatPair: opencl version of output_order/input_order
typedef struct __IntFloatPair
{
	int idx;
	float err;
} IntFloatPair, DCTScoreData, CoeffData;

typedef struct __IntFloatPairList
{
	int size;
	__private IntFloatPair *pData;
} IntFloatPairList;

__device__ void XybToVals(
	__private const float x, __private const float y, __private const float z,
	__private float *valx, __private float *valy, __private float *valz);
__device__ float InterpolateClampNegative(__global const float *array, int size, float sx);
__device__ void XybDiffLowFreqSquaredAccumulate(float r0, float g0, float b0,
												float r1, float g1, float b1,
												float factor, float res[3]);
__device__ float DotProduct(__global const float u[3], const float v[3]);
__device__ void OpsinAbsorbance(__private const float in[3], __private float out[3]);
__device__ void RgbToXyb(const float r, const float g, const float b, __private float *valx, __private float *valy, __private float *valz);
__device__ float Gamma(float v);
__device__ void ButteraugliBlockDiff(__private float xyb0[3 * kBlockSize],
									 __private float xyb1[3 * kBlockSize],
									 float diff_xyb_dc[3],
									 float diff_xyb_ac[3],
									 float diff_xyb_edge_dc[3]);
__device__ void Butteraugli8x8CornerEdgeDetectorDiff(
	int pos_x,
	int pos_y,
	int xsize,
	int ysize,
	__global const float *r, __global const float *g, __global const float *b,
	__global const float *r2, __global const float *g2, __global const float *b2,
	__private float *diff_xyb);

__device__ int MakeInputOrderEx(const coeff_t block[3 * 8 * 8], const coeff_t orig_block[3 * 8 * 8], __private IntFloatPairList *input_order);

__device__ float Factor2(const channel_info mayout_channel[3],
						  __private const coeff_t *candidate_block,
						  const int block_x,
						  const int block_y,
						  __global const float *orig_image_batch,
						  __global const float *mask_scale,
						  const int image_width,
						  const int image_height);

__device__ float CompareBlockFactor1(const channel_info mayout_channel[3],
									  __private const coeff_t *candidate_block,
									  const int block_x,
									  const int block_y,
									  __global const float *orig_image_batch,
									  __global const float *mask_scale,
									  const int image_width,
									  const int image_height);

__device__ float CompareBlockFactor(const channel_info mayout_channel[3],
									 __private const coeff_t *candidate_block,
									 const int block_x,
									 const int block_y,
									 __global const float *orig_image_batch,
									 __global const float *mask_scale,
									 const int image_width,
									 const int image_height,
									 const int factor);

__device__ void floatcopy(__private float *dst, __private const float *src, const int size);
__device__ void coeffcopy(__private coeff_t *dst, const coeff_t *src, int size);
__device__ void coeffcopy_g(__private coeff_t *dst, __global const coeff_t *src, int size);
__device__ int list_erase(__private IntFloatPairList *list, const int idx);
__device__ int list_push_back(__private IntFloatPairList *list, int i, float f);
__device__ void CoeffToIDCT(__private const coeff_t block[8 * 8], uchar out[8 * 8]);
__device__ coeff_t Quantize(coeff_t raw_coeff, int quant);
__device__ bool QuantizeBlock(coeff_t block[kDCTBlockSize], __global const int q[kDCTBlockSize]);
__device__ void ColorTransformYCbCrToRGB(__global uchar pixel[3]);

__device__ float _abs_f(float val)
{
	return val >= 0 ? val : -val;
}

__device__ float _abs_d(float val)
{
	return val >= 0 ? val : -val;
}

__device__ float _min_f(float a, float b)
{
	return a < b ? a : b;
}

__device__ float _max_f(float a, float b)
{
	return a > b ? a : b;
}

__device__ float _min_d(float a, float b)
{
	return a < b ? a : b;
}

__device__ float _max_d(float a, float b)
{
	return a > b ? a : b;
}

__device__ int _min_i(int a, int b)
{
	return a < b ? a : b;
}

__device__ int _max_i(int a, int b)
{
	return a > b ? a : b;
}

__kernel void clConvolutionEx(
	__global float *__restrict__ result,
	__global const float *__restrict__ inp, const int xsize,
	__global const float *multipliers, const int len,
	const int xstep, const int offset, const float border_ratio)
{
	const int ox = get_global_id(0);
	const int y = get_global_id(1);

	// Early exit for out of bounds
	if (ox >= get_global_size(0) || y >= get_global_size(1))
		return;

	const int oxsize = get_global_size(0);
	const int ysize = get_global_size(1);

	const int x = ox * xstep;

	float weight_no_border = 0;
	for (int j = 0; j <= 2 * offset; j++)
	{
		weight_no_border += multipliers[j];
	}

	int minx = x < offset ? 0 : x - offset;
	int maxx = _min_i(xsize, x + len - offset);

	float weight = 0.0f;
	for (int j = minx; j < maxx; j++)
	{
		weight += multipliers[j - x + offset];
	}

	weight = (1.0f - border_ratio) * weight + border_ratio * weight_no_border;
	float scale = 1.0f / weight;

	float sum = 0.0f;
	for (int j = minx; j < maxx; j++)
	{
		sum += inp[y * xsize + j] * multipliers[j - x + offset];
	}

	result[ox * ysize + y] = sum * scale;
}

__kernel void clConvolutionXEx(
	__global float *__restrict__ result,
	const int xsize, const int ysize,
	__global const float *__restrict__ inp,
	__global const float *multipliers, const int len,
	const int step, const int offset, const float border_ratio)
{
	const int x = get_global_id(0) * step;
	const int y = get_global_id(1);

	if (x >= xsize || y >= ysize)
		return;

	if (x % step != 0)
		return;

	float weight_no_border = 0;
	for (int j = 0; j <= 2 * offset; j++)
	{
		weight_no_border += multipliers[j];
	}

	int minx = x < offset ? 0 : x - offset;
	int maxx = _min_i(xsize, x + len - offset);

	float weight = 0.0f;
	for (int j = minx; j < maxx; j++)
	{
		weight += multipliers[j - x + offset];
	}

	weight = (1.0f - border_ratio) * weight + border_ratio * weight_no_border;
	float scale = 1.0f / weight;

	float sum = 0.0f;
	for (int j = minx; j < maxx; j++)
	{
		sum += inp[y * xsize + j] * multipliers[j - x + offset];
	}

	result[y * xsize + x] = sum * scale;
}

__kernel void clConvolutionYEx(
	__global float *__restrict__ result,
	const int xsize, const int ysize,
	__global const float *__restrict__ inp,
	__global const float *multipliers, const int len,
	const int step, const int offset, const float border_ratio)
{
	const int x = get_global_id(0) * step;
	const int y = get_global_id(1) * step;

	if (x >= xsize || y >= ysize)
		return;
	if (x % step != 0)
		return;
	if (y % step != 0)
		return;

	float weight_no_border = 0;
	for (int j = 0; j <= 2 * offset; j++)
	{
		weight_no_border += multipliers[j];
	}

	int miny = y < offset ? 0 : y - offset;
	int maxy = _min_i(ysize, y + len - offset);

	float weight = 0.0f;
	for (int j = miny; j < maxy; j++)
	{
		weight += multipliers[j - y + offset];
	}

	weight = (1.0f - border_ratio) * weight + border_ratio * weight_no_border;
	float scale = 1.0f / weight;

	float sum = 0.0f;
	for (int j = miny; j < maxy; j++)
	{
		sum += inp[j * xsize + x] * multipliers[j - y + offset];
	}

	result[y * xsize + x] = sum * scale;
}

__kernel void clSquareSampleEx(
	__global float *__restrict__ result,
	const int xsize, const int ysize,
	__global const float *image,
	const int xstep, const int ystep)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	if (x >= xsize || y >= ysize)
		return;

	int x_sample = x - x % xstep;
	int y_sample = y - y % ystep;

	if (x_sample == x && y_sample == y)
		return;

	result[y * xsize + x] = image[y_sample * xsize + x_sample];
}

__kernel void clOpsinDynamicsImageEx(
	__global float *r,
	__global float *g,
	__global float *b,
	const int size,
	__global const float *r_blurred,
	__global const float *g_blurred,
	__global const float *b_blurred)
{
	const int i = get_global_id(0);
	if (i >= size)
		return;

	float pre[3] = {r_blurred[i], g_blurred[i], b_blurred[i]};
	float pre_mixed[3];
	OpsinAbsorbance(pre, pre_mixed);

	float sensitivity[3];
	sensitivity[0] = Gamma(pre_mixed[0]) / pre_mixed[0];
	sensitivity[1] = Gamma(pre_mixed[1]) / pre_mixed[1];
	sensitivity[2] = Gamma(pre_mixed[2]) / pre_mixed[2];

	float cur_rgb[3] = {r[i], g[i], b[i]};
	float cur_mixed[3];
	OpsinAbsorbance(cur_rgb, cur_mixed);
	cur_mixed[0] *= sensitivity[0];
	cur_mixed[1] *= sensitivity[1];
	cur_mixed[2] *= sensitivity[2];

	float x, y, z;
	RgbToXyb(cur_mixed[0], cur_mixed[1], cur_mixed[2], &x, &y, &z);
	r[i] = x;
	g[i] = y;
	b[i] = z;
}

__kernel void clMaskHighIntensityChangeEx(
	__global float *xyb0_x, __global float *xyb0_y, __global float *xyb0_b,
	const int xsize, const int ysize,
	__global float *xyb1_x, __global float *xyb1_y, __global float *xyb1_b,
	__global const float *c0_x, __global const float *c0_y, __global const float *c0_b,
	__global const float *c1_x, __global const float *c1_y, __global const float *c1_b)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	if (x >= xsize || y >= ysize)
		return;

	const int ix = y * xsize + x;
	const float ave[3] = {
		(c0_x[ix] + c1_x[ix]) * 0.5f,
		(c0_y[ix] + c1_y[ix]) * 0.5f,
		(c0_b[ix] + c1_b[ix]) * 0.5f,
	};
	float sqr_max_diff = -1;
	{
		int offset[4] = {-1, 1, -(int)(xsize), (int)(xsize)};
		int border[4] = {x == 0, x + 1 == xsize, y == 0, y + 1 == ysize};
		for (int dir = 0; dir < 4; ++dir)
		{
			if (border[dir])
			{
				continue;
			}
			const int ix2 = ix + offset[dir];
			float diff = 0.5f * (c0_y[ix2] + c1_y[ix2]) - ave[1];
			diff *= diff;
			if (sqr_max_diff < diff)
			{
				sqr_max_diff = diff;
			}
		}
	}
	const float kReductionX = 275.19165240059317f;
	const float kReductionY = 18599.41286306991f;
	const float kReductionZ = 410.8995306951065f;
	const float kChromaBalance = 106.95800948271017f;
	float chroma_scale = kChromaBalance / (ave[1] + kChromaBalance);

	const float mix[3] = {
		chroma_scale * kReductionX / (sqr_max_diff + kReductionX),
		kReductionY / (sqr_max_diff + kReductionY),
		chroma_scale * kReductionZ / (sqr_max_diff + kReductionZ),
	};
	// Interpolate lineraly between the average color and the actual
	// color -- to reduce the importance of this pixel.
	xyb0_x[ix] = (float)(mix[0] * c0_x[ix] + (1 - mix[0]) * ave[0]);
	xyb1_x[ix] = (float)(mix[0] * c1_x[ix] + (1 - mix[0]) * ave[0]);

	xyb0_y[ix] = (float)(mix[1] * c0_y[ix] + (1 - mix[1]) * ave[1]);
	xyb1_y[ix] = (float)(mix[1] * c1_y[ix] + (1 - mix[1]) * ave[1]);

	xyb0_b[ix] = (float)(mix[2] * c0_b[ix] + (1 - mix[2]) * ave[2]);
	xyb1_b[ix] = (float)(mix[2] * c1_b[ix] + (1 - mix[2]) * ave[2]);
}

__kernel void clEdgeDetectorMapEx(
	__global float *result,
	const int res_xsize, 
	const int res_ysize,
	__global const float *r, 
	__global const float *g, 
	__global const float *b,
	__global const float *r2, 
	__global const float *g2, 
	__global const float *b2,
	int xsize, 
	int ysize, 
	int step)
{
	const int res_x = get_global_id(0);
	const int res_y = get_global_id(1);

	if (res_x >= res_xsize || res_y >= res_ysize)
		return;

	int pos_x = res_x * step;
	int pos_y = res_y * step;

	if (pos_x >= xsize - (8 - step))
		return;
	if (pos_y >= ysize - (8 - step))
		return;

	pos_x = _min_i(pos_x, xsize - 8);
	pos_y = _min_i(pos_y, ysize - 8);

	float diff_xyb[3] = {0.0};
	Butteraugli8x8CornerEdgeDetectorDiff(pos_x, pos_y, xsize, ysize,
										 r, g, b,
										 r2, g2, b2,
										 &diff_xyb[0]);

	int idx = (res_y * res_xsize + res_x) * 3;
	result[idx] = diff_xyb[0];
	result[idx + 1] = diff_xyb[1];
	result[idx + 2] = diff_xyb[2];
}

__kernel void clBlockDiffMapEx(
	__global float *block_diff_dc, __global float *block_diff_ac,
	const int res_xsize, const int res_ysize,
	__global const float *r, __global const float *g, __global const float *b,
	__global const float *r2, __global const float *g2, __global const float *b2,
	int xsize, int ysize, int step)
{
	const int res_x = get_global_id(0);
	const int res_y = get_global_id(1);

	if (res_x >= res_xsize || res_y >= res_ysize)
		return;

	int pos_x = res_x * step;
	int pos_y = res_y * step;

	if ((pos_x + kBlockEdge - step - 1) >= xsize)
		return;
	if ((pos_y + kBlockEdge - step - 1) >= ysize)
		return;

	unsigned int res_ix = res_y * res_xsize + res_x;
	unsigned int offset = _min_i(pos_y, ysize - 8) * xsize + _min_i(pos_x, xsize - 8);

	__private float block0[3 * kBlockEdge * kBlockEdge];
	__private float block1[3 * kBlockEdge * kBlockEdge];

	__private float *block0_r = &block0[0];
	__private float *block0_g = &block0[kBlockEdge * kBlockEdge];
	__private float *block0_b = &block0[2 * kBlockEdge * kBlockEdge];

	__private float *block1_r = &block1[0];
	__private float *block1_g = &block1[kBlockEdge * kBlockEdge];
	__private float *block1_b = &block1[2 * kBlockEdge * kBlockEdge];

	for (int y = 0; y < kBlockEdge; y++)
	{
		for (int x = 0; x < kBlockEdge; x++)
		{
			block0_r[kBlockEdge * y + x] = r[offset + y * xsize + x];
			block0_g[kBlockEdge * y + x] = g[offset + y * xsize + x];
			block0_b[kBlockEdge * y + x] = b[offset + y * xsize + x];
			block1_r[kBlockEdge * y + x] = r2[offset + y * xsize + x];
			block1_g[kBlockEdge * y + x] = g2[offset + y * xsize + x];
			block1_b[kBlockEdge * y + x] = b2[offset + y * xsize + x];
		}
	}

	__private float diff_xyb_dc[3] = {0.0};
	__private float diff_xyb_ac[3] = {0.0};
	__private float diff_xyb_edge_dc[3] = {0.0};

	ButteraugliBlockDiff(block0, block1, diff_xyb_dc, diff_xyb_ac, diff_xyb_edge_dc);

	for (int i = 0; i < 3; ++i)
	{
		block_diff_dc[3 * res_ix + i] = diff_xyb_dc[i];
		block_diff_ac[3 * res_ix + i] = diff_xyb_ac[i];
	}
}

__kernel void clEdgeDetectorLowFreqEx(
	__global float *block_diff_ac,
	const int res_xsize, const int res_ysize,
	__global const float *r, __global const float *g, __global const float *b,
	__global const float *r2, __global const float *g2, __global const float *b2,
	int xsize, int ysize, int step_)
{
	const int res_x = get_global_id(0);
	const int res_y = get_global_id(1);

	if (res_x >= res_xsize || res_y >= res_ysize)
		return;

	const int step = 8;
	if (res_x < step / step_)
		return;

	int x = (res_x - (step / step_)) * step_;
	int y = res_y * step_;

	if (x + step >= xsize)
		return;
	if (y + step >= ysize)
		return;

	int ix = y * xsize + x;

	float diff[4][3];
	__global const float *blurred0[3] = {r, g, b};
	__global const float *blurred1[3] = {r2, g2, b2};

	for (int i = 0; i < 3; ++i)
	{
		int ix2 = ix + 8;
		diff[0][i] =
			((blurred1[i][ix] - blurred0[i][ix]) +
			 (blurred0[i][ix2] - blurred1[i][ix2]));
		ix2 = ix + 8 * xsize;
		diff[1][i] =
			((blurred1[i][ix] - blurred0[i][ix]) +
			 (blurred0[i][ix2] - blurred1[i][ix2]));
		ix2 = ix + 6 * xsize + 6;
		diff[2][i] =
			((blurred1[i][ix] - blurred0[i][ix]) +
			 (blurred0[i][ix2] - blurred1[i][ix2]));
		ix2 = ix + 6 * xsize - 6;
		diff[3][i] = x < step ? 0 : ((blurred1[i][ix] - blurred0[i][ix]) + (blurred0[i][ix2] - blurred1[i][ix2]));
	}
	float max_diff_xyb[3] = {0};
	for (int k = 0; k < 4; ++k)
	{
		float diff_xyb[3] = {0};
		XybDiffLowFreqSquaredAccumulate(diff[k][0], diff[k][1], diff[k][2],
										0, 0, 0, 1.0,
										diff_xyb);
		for (int i = 0; i < 3; ++i)
		{
			max_diff_xyb[i] = _max_f(max_diff_xyb[i], diff_xyb[i]);
		}
	}

	int res_ix = res_y * res_xsize + res_x;

	const float kMul = 10;

	block_diff_ac[res_ix * 3] += max_diff_xyb[0] * kMul;
	block_diff_ac[res_ix * 3 + 1] += max_diff_xyb[1] * kMul;
	block_diff_ac[res_ix * 3 + 2] += max_diff_xyb[2] * kMul;
}

__kernel void clDiffPrecomputeEx(
	__global float *mask_x, __global float *mask_y, __global float *mask_b,
	const int xsize, const int ysize,
	__global const float *xyb0_x, __global const float *xyb0_y, __global const float *xyb0_b,
	__global const float *xyb1_x, __global const float *xyb1_y, __global const float *xyb1_b)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	if (x >= xsize || y >= ysize)
		return;

	__private float valsh0[3] = {0.0};
	__private float valsv0[3] = {0.0};
	__private float valsh1[3] = {0.0};
	__private float valsv1[3] = {0.0};
	int ix2;

	int ix = x + xsize * y;
	if (x + 1 < xsize)
	{
		ix2 = ix + 1;
	}
	else
	{
		ix2 = ix - 1;
	}
	{
		__private float x0 = (xyb0_x[ix] - xyb0_x[ix2]);
		__private float y0 = (xyb0_y[ix] - xyb0_y[ix2]);
		__private float z0 = (xyb0_b[ix] - xyb0_b[ix2]);
		XybToVals(x0, y0, z0, &valsh0[0], &valsh0[1], &valsh0[2]);
		__private float x1 = (xyb1_x[ix] - xyb1_x[ix2]);
		__private float y1 = (xyb1_y[ix] - xyb1_y[ix2]);
		__private float z1 = (xyb1_b[ix] - xyb1_b[ix2]);
		XybToVals(x1, y1, z1, &valsh1[0], &valsh1[1], &valsh1[2]);
	}
	if (y + 1 < ysize)
	{
		ix2 = ix + xsize;
	}
	else
	{
		ix2 = ix - xsize;
	}
	{
		__private float x0 = (xyb0_x[ix] - xyb0_x[ix2]);
		__private float y0 = (xyb0_y[ix] - xyb0_y[ix2]);
		__private float z0 = (xyb0_b[ix] - xyb0_b[ix2]);
		XybToVals(x0, y0, z0, &valsv0[0], &valsv0[1], &valsv0[2]);
		__private float x1 = (xyb1_x[ix] - xyb1_x[ix2]);
		__private float y1 = (xyb1_y[ix] - xyb1_y[ix2]);
		__private float z1 = (xyb1_b[ix] - xyb1_b[ix2]);
		XybToVals(x1, y1, z1, &valsv1[0], &valsv1[1], &valsv1[2]);
	}

	__private float sup0 = _abs_d(valsh0[0]) + _abs_d(valsv0[0]);
	__private float sup1 = _abs_d(valsh1[0]) + _abs_d(valsv1[0]);
	__private float m = _min_d(sup0, sup1);
	mask_x[ix] = (float)(m);

	sup0 = _abs_d(valsh0[1]) + _abs_d(valsv0[1]);
	sup1 = _abs_d(valsh1[1]) + _abs_d(valsv1[1]);
	m = _min_d(sup0, sup1);
	mask_y[ix] = (float)(m);

	sup0 = _abs_d(valsh0[2]) + _abs_d(valsv0[2]);
	sup1 = _abs_d(valsh1[2]) + _abs_d(valsv1[2]);
	m = _min_d(sup0, sup1);
	mask_b[ix] = (float)(m);
}

__kernel void clScaleImageEx(__global float *img, const int size, float scale)
{
	const int i = get_global_id(0);
	if (i >= size)
		return;

	img[i] *= scale;
}

#define Average5x5_w 0.679144890667f
__constant float Average5x5_scale = 1.0f / (5.0f + 4 * Average5x5_w);
__kernel void clAverage5x5Ex(__global float *img, const int xsize, const int ysize, __global const float *img_org)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	if (x >= xsize || y >= ysize)
		return;

	const int row0 = y * xsize;
	if (x - 1 >= 0)
	{
		img[row0 + x] += img_org[row0 + x - 1];
	}
	if (x + 1 < xsize)
	{
		img[row0 + x] += img_org[row0 + x + 1];
	}

	if (y > 0)
	{
		const int rowd1 = row0 - xsize;
		if (x - 1 >= 0)
		{
			img[row0 + x] += img_org[rowd1 + x - 1] * Average5x5_w;
		}
		img[row0 + x] += img_org[rowd1 + x];
		if (x + 1 < xsize)
		{
			img[row0 + x] += img_org[rowd1 + x + 1] * Average5x5_w;
		}
	}

	if (y + 1 < ysize)
	{
		const int rowu1 = row0 + xsize;
		if (x - 1 >= 0)
		{
			img[row0 + x] += img_org[rowu1 + x - 1] * Average5x5_w;
		}
		img[row0 + x] += img_org[rowu1 + x];
		if (x + 1 < xsize)
		{
			img[row0 + x] += img_org[rowu1 + x + 1] * Average5x5_w;
		}
	}

	img[row0 + x] *= Average5x5_scale;
}

__kernel void clMinSquareValEx(__global float *result, const int xsize, const int ysize, __global const float *img, const int square_size, const int offset)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	if (x >= xsize || y >= ysize)
		return;

	int minH = offset > y ? 0 : y - offset;
	int maxH = _min_i(y + square_size - offset, ysize);

	int minW = offset > x ? 0 : x - offset;
	int maxW = _min_i(x + square_size - offset, xsize);

	float minValue = img[minH * xsize + minW];

	for (int j = minH; j < maxH; j++)
	{
		for (int i = minW; i < maxW; i++)
		{
			float tmp = img[j * xsize + i];
			if (tmp < minValue)
				minValue = tmp;
		}
	}

	result[y * xsize + x] = minValue;
}

__kernel void clDoMaskEx(
	__global float *mask_x, __global float *mask_y, __global float *mask_b,
	const int xsize, const int ysize,
	__global float *mask_dc_x, __global float *mask_dc_y, __global float *mask_dc_b,
	__global const float *lut_x, __global const float *lut_y, __global const float *lut_b,
	__global const float *lut_dc_x, __global const float *lut_dc_y, __global const float *lut_dc_b)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	if (x >= xsize || y >= ysize)
		return;

	const float w00 = 232.206464018f;
	const float w11 = 22.9455222245f;
	const float w22 = 503.962310606f;

	const unsigned int idx = y * xsize + x;
	const float s0 = mask_x[idx];
	const float s1 = mask_y[idx];
	const float s2 = mask_b[idx];
	const float p0 = w00 * s0;
	const float p1 = w11 * s1;
	const float p2 = w22 * s2;

	mask_x[idx] = (float)(InterpolateClampNegative(lut_x, 512, p0));
	mask_y[idx] = (float)(InterpolateClampNegative(lut_y, 512, p1));
	mask_b[idx] = (float)(InterpolateClampNegative(lut_b, 512, p2));
	mask_dc_x[idx] = (float)(InterpolateClampNegative(lut_dc_x, 512, p0));
	mask_dc_y[idx] = (float)(InterpolateClampNegative(lut_dc_y, 512, p1));
	mask_dc_b[idx] = (float)(InterpolateClampNegative(lut_dc_b, 512, p2));
}

__kernel void clCombineChannelsEx(
	__global float *result,
	__global const float *mask_x, __global const float *mask_y, __global const float *mask_b,
	__global const float *mask_dc_x, __global const float *mask_dc_y, __global const float *mask_dc_b,
	const int xsize, const int ysize,
	__global const float *block_diff_dc,
	__global const float *block_diff_ac,
	__global float *edge_detector_map,
	const int res_xsize,
	const int step)
{
	const int res_x = get_global_id(0) * step;
	const int res_y = get_global_id(1) * step;

	if (res_x + (8 - step) >= xsize || res_y + (8 - step) >= ysize)
		return;

	float mask[3];
	float dc_mask[3];
	mask[0] = mask_x[(res_y + 3) * xsize + (res_x + 3)];
	dc_mask[0] = mask_dc_x[(res_y + 3) * xsize + (res_x + 3)];

	mask[1] = mask_y[(res_y + 3) * xsize + (res_x + 3)];
	dc_mask[1] = mask_dc_y[(res_y + 3) * xsize + (res_x + 3)];

	mask[2] = mask_b[(res_y + 3) * xsize + (res_x + 3)];
	dc_mask[2] = mask_dc_b[(res_y + 3) * xsize + (res_x + 3)];

	unsigned int res_ix = (res_y * res_xsize + res_x) / step;
	result[res_ix] = (float)(DotProduct(&block_diff_dc[3 * res_ix], dc_mask) +
							 DotProduct(&block_diff_ac[3 * res_ix], mask) +
							 DotProduct(&edge_detector_map[3 * res_ix], mask));
}

__kernel void clUpsampleSquareRootEx(__global float *diffmap_out, __global const float *diffmap, const int xsize, const int ysize, const int step)
{
	const int res_x = get_global_id(0);
	const int res_y = get_global_id(1);

	const int res_xsize = get_global_size(0);
	const int res_ysize = get_global_size(1);

	const int pos_x = res_x * step;
	const int pos_y = res_y * step;

	if (pos_y + 8 - step >= ysize)
		return;
	if (pos_x + 8 - step >= xsize)
		return;

	int s2 = (8 - step) / 2;

	// Upsample and take square root.
	float orig_val = diffmap[res_y * res_xsize + res_x];

	const float kInitialSlope = 100;
	// TODO(b/29974893): Until that is fixed do not call sqrt on very small
	// numbers.
	float val = orig_val < (1.0 / (kInitialSlope * kInitialSlope))
					 ? kInitialSlope * orig_val
					 : sqrt(orig_val);

	for (unsigned int off_y = 0; off_y < step; ++off_y)
	{
		for (unsigned int off_x = 0; off_x < step; ++off_x)
		{
			diffmap_out[(pos_y + off_y + s2) * xsize + pos_x + off_x + s2] = val;
		}
	}
}

__kernel void clRemoveBorderEx(__global float *out, const int xsize, const int ysize, __global const float *in, int s, int s2)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	if (x >= xsize || y >= ysize)
		return;

	out[y * xsize + x] = in[(y + s2) * (xsize + s) + x + s2];
}

__kernel void clAddBorderEx(__global float *out, const int xsize, const int ysize, int s, int s2, __global const float *in)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	if (x >= xsize - s ||
		y >= ysize - s)
	{
		return;
	}

	const float mul1 = 24.8235314874f;
	out[(y + s2) * xsize + x + s2] += (float)(mul1)*in[y * (xsize - s) + x];
}

__kernel void clComputeBlockZeroingOrderEx(
	__global const coeff_t *orig_batch_0,	// Coeffs of Original image.
	__global const coeff_t *orig_batch_1,	// Coeffs of Original image.
	__global const coeff_t *orig_batch_2,	// Coeffs of Original image.
	__global const float *orig_image_batch, // pregamma of Original image..
	__global const float *mask_scale,		// mask_scale of Original image..
	const int block_xsize,
	const int block_ysize,
	const int image_width,
	const int image_height,

	__global const coeff_t *mayout_batch_0, // Coeffs of output image.
	__global const coeff_t *mayout_batch_1, // Coeffs of output image.
	__global const coeff_t *mayout_batch_2, // Coeffs of output image.
	__global const ushort *mayout_pixel_0,
	__global const ushort *mayout_pixel_1,
	__global const ushort *mayout_pixel_2,

	const channel_info mayout_channel_0,
	const channel_info mayout_channel_1,
	const channel_info mayout_channel_2,
	const int factor,	 // Current factor in computing.
	const int comp_mask, // Current channel in computing.
	const float BlockErrorLimit,
	__global CoeffData *output_order_list /*out*/)
{
	const int block_x = get_global_id(0);
	const int block_y = get_global_id(1);

	if (block_x >= block_xsize || block_y >= block_ysize)
		return;

	channel_info orig_channel[3];
	orig_channel[0].coeff = orig_batch_0;
	orig_channel[1].coeff = orig_batch_1;
	orig_channel[2].coeff = orig_batch_2;

	channel_info mayout_channel[3] = {mayout_channel_0, mayout_channel_1, mayout_channel_2};
	mayout_channel[0].coeff = mayout_batch_0;
	mayout_channel[1].coeff = mayout_batch_1;
	mayout_channel[2].coeff = mayout_batch_2;
	mayout_channel[0].pixel = mayout_pixel_0;
	mayout_channel[1].pixel = mayout_pixel_1;
	mayout_channel[2].pixel = mayout_pixel_2;

	int block_idx = 0;

	coeff_t mayout_block[kComputeBlockSize] = {0};
	coeff_t orig_block[kComputeBlockSize] = {0};

	for (int c = 0; c < 3; c++)
	{
		if (comp_mask & (1 << c))
		{
			block_idx = block_y * mayout_channel[c].block_width + block_x;
			coeffcopy_g(&mayout_block[c * kBlockSize],
						mayout_channel[c].coeff + block_idx * kBlockSize,
						kBlockSize);
			coeffcopy_g(&orig_block[c * kBlockSize],
						orig_channel[c].coeff + block_idx * kBlockSize,
						kBlockSize);
		}
	}

	__private DCTScoreData input_order_data[kComputeBlockSize];
	CoeffData output_order_data[kComputeBlockSize];

	__private IntFloatPairList input_order = {0, input_order_data};
	IntFloatPairList output_order = {0, output_order_data};

	int count = MakeInputOrderEx(mayout_block, orig_block, &input_order);

	while (input_order.size > 0)
	{
		float best_err = 1e17f;
		int best_i = 0;
		for (int i = 0; i < _min_i(3, input_order.size); i++)
		{
			const int idx = input_order.pData[i].idx;
			coeff_t old_coeff = mayout_block[idx];
			mayout_block[idx] = 0;

			float max_err = CompareBlockFactor(mayout_channel,
											   mayout_block,
											   block_x,
											   block_y,
											   orig_image_batch,
											   mask_scale,
											   image_width,
											   image_height,
											   factor);
			if (max_err < best_err)
			{
				best_err = max_err;
				best_i = i;
			}
			mayout_block[idx] = old_coeff;
		}

		if (best_err >= BlockErrorLimit)
		{ // The input_order is an ascent vector, break when best_err exceed the error limit.
			break;
		}
		int idx = input_order.pData[best_i].idx;
		mayout_block[idx] = 0;
		list_erase(&input_order, best_i);

		list_push_back(&output_order, idx, best_err);
	}

	float min_err = 1e10;
	for (int i = output_order.size - 1; i >= 0; --i)
	{
		min_err = _min_f(min_err, output_order.pData[i].err);
		output_order.pData[i].err = min_err;
	}

	__global CoeffData *output_block = output_order_list + block_idx * kComputeBlockSize;

	int out_count = 0;
	for (int i = 0; i < kComputeBlockSize && i < output_order.size; i++)
	{
		// err exceeding the limit is no need to continue.
		if (output_order.pData[i].err <= BlockErrorLimit)
		{
			output_block[out_count].idx = output_order.pData[i].idx;
			output_block[out_count].err = output_order.pData[i].err;
			out_count++;
		}
	}
}

__kernel void clCopyFromJpegComponentEx(
	__global coeff_t *output_batch, // Coeffs of output image.
	__global uchar *output_idct,

	__global const coeff_t *jpeg_batch, // Coeffs of Original image.

	__global const int *quant,

	const int jpeg_block_width,
	const int jpeg_block_height,
	const int output_block_width,
	const int output_block_height,
	const int output_width_,
	const int output_height_)
{
	const int block_x = get_global_id(0);
	const int block_y = get_global_id(1);

	if (block_x >= output_block_width || block_y >= output_block_height)
		return;

	int offset_src = (block_y * jpeg_block_width + block_x) * kDCTBlockSize;
	int offset_dst = (block_y * output_block_width + block_x) * kDCTBlockSize;

	__global const coeff_t *src_coeffs = &jpeg_batch[offset_src];

	coeff_t block[kDCTBlockSize];
	for (int i = 0; i < kDCTBlockSize; ++i)
	{
		block[i] = src_coeffs[i] * quant[i];
		output_batch[offset_dst + i] = block[i];
	}

	__global uchar *idct_g = &output_idct[offset_dst];
	uchar idct[kDCTBlockSize];
	CoeffToIDCT(block, idct);
	for (int i = 0; i < kDCTBlockSize; ++i)
	{
		idct_g[i] = idct[i];
	}
}

__kernel void clApplyGlobalQuantizationEx(
	__global coeff_t *output_batch,
	__global uchar *output_idct,
	__global uchar *output_bool,
	__global const int *q,
	const int block_width,
	const int block_height)
{
	const int block_x = get_global_id(0);
	const int block_y = get_global_id(1);

	if (block_x >= block_width || block_y >= block_height)
		return;

	int offset = (block_y * block_width + block_x) * kDCTBlockSize;
	int bool_offset = (block_y * block_width + block_x);

	__global coeff_t *src_coeffs = &output_batch[offset];

	coeff_t block[kDCTBlockSize];
	for (int i = 0; i < kDCTBlockSize; ++i)
	{
		block[i] = src_coeffs[i];
	}

	output_bool[bool_offset] = 0;
	if (QuantizeBlock(block, q))
	{
		output_bool[bool_offset] = 1;
		for (int i = 0; i < kDCTBlockSize; ++i)
		{
			src_coeffs[i] = block[i];
		}
		__global uchar *idct_g = &output_idct[offset];
		uchar idct[kDCTBlockSize];
		CoeffToIDCT(block, idct);
		for (int i = 0; i < kDCTBlockSize; ++i)
		{
			idct_g[i] = idct[i];
		}
	}
}

__kernel void clComponentsToPixels(
	__global uchar *out,
	const int xmin,
	const int ymin,
	const int xsize,
	const int ysize,
	const int stride,

	__global const ushort *pixels,
	const int width,
	const int height)
{
	const int block_x = get_global_id(0);
	const int block_y = get_global_id(1);

	const int yend1 = ymin + ysize;
	const int yend0 = _min_i(yend1, height);

	const int xend1 = xmin + xsize;
	const int xend0 = _min_i(xend1, width);

	int y = ymin + block_y;
	int x = xmin + block_x;

	if (x >= xend0 || y >= yend0)
		return;

	int px = y * width + x;

	__global uchar *out_dst = out + ((xend0 - xmin) * block_y + block_x) * stride;

	*out_dst = (uchar)((pixels[px] + 8 - (x & 1)) >> 4);
}

__kernel void clComponentsToPixelsEx1(
	__global uchar *out,
	const int xmin,
	const int ymin,
	const int xsize,
	const int ysize,
	const int stride,

	const int width,
	const int height)
{
	const int block_x = get_global_id(0);
	const int block_y = get_global_id(1);

	const int yend1 = ymin + ysize;
	const int yend0 = _min_i(yend1, height);

	const int xend1 = xmin + xsize;
	const int xend0 = _min_i(xend1, width);

	int x = xend0 + block_x;
	int y = ymin + block_y;

	if (x >= xend1 || y >= yend0)
		return;

	__global uchar *out_dst = out + ((xend0 - xmin) * block_y + block_x) * stride;
	__global uchar *out_src = out + ((xend0 - xmin) * block_y + xend0 - xmin - 1) * stride;

	*out_dst = *out_src;
	/*
	int y = ymin;
	for (; y < yend0; ++y) {
		const int xend1 = xmin + xsize;
		const int xend0 = std::min(xend1, width_);
		int x = xmin;
		int px = y * width_ + xmin;
		for (; x < xend0; ++x, ++px, out += stride) {}

		const int offset = -stride;
		for (; x < xend1; ++x) {
			*out = out[offset];
			out += stride;
		}
	}
	*/
}

__kernel void clComponentsToPixelsEx2(
	__global uchar *out,
	const int xmin,
	const int ymin,
	const int xsize,
	const int ysize,
	const int stride,

	const int width,
	const int height)
{
	const int block_x = get_global_id(0);
	const int block_y = get_global_id(1);

	const int yend1 = ymin + ysize;
	const int yend0 = _min_i(yend1, height);

	const int xend1 = xmin + xsize;
	const int xend0 = _min_i(xend1, width);

	int x = block_x;
	int y = yend0 + block_y;

	if (x >= xsize || y >= yend1)
		return;

	__global uchar *out_dst = out + (xsize * block_y + block_x) * stride;
	__global uchar *out_src = out + (xsize * (yend0 - ymin - 1) + block_x) * stride;

	*out_dst = *out_src;
	/*
	int y = ymin;
	for (; y < yend1; ++y) {
		const int offset = -stride * xsize;
		for (int x = 0; x < xsize; ++x) {
			*out = out[offset];
			out += stride;
		}
	}*/
}

__kernel void clColorTransformYCbCrToRGB(
	__global uchar *rgb)
{
	const int block_x = get_global_id(0);

	ColorTransformYCbCrToRGB(&rgb[block_x * 3]);
}

__device__ void Butteraugli8x8CornerEdgeDetectorDiff(
	int pos_x,
	int pos_y,
	int xsize,
	int ysize,
	__global const float *r, __global const float *g, __global const float *b,
	__global const float *r2, __global const float *g2, __global const float *b2,
	__private float *diff_xyb)
{
	int local_count = 0;
	float local_xyb[3] = {0};
	const float w = 0.711100840192f;

	int offset[4][2] = {{0, 0}, {0, 7}, {7, 0}, {7, 7}};
	int edgeSize = 3;

	for (int k = 0; k < 4; k++)
	{
		int x = pos_x + offset[k][0];
		int y = pos_y + offset[k][1];

		if (x >= edgeSize && x + edgeSize < xsize)
		{
			unsigned int ix = y * xsize + (x - edgeSize);
			unsigned int ix2 = ix + 2 * edgeSize;
			XybDiffLowFreqSquaredAccumulate(
				w * (r[ix] - r[ix2]),
				w * (g[ix] - g[ix2]),
				w * (b[ix] - b[ix2]),
				w * (r2[ix] - r2[ix2]),
				w * (g2[ix] - g2[ix2]),
				w * (b2[ix] - b2[ix2]),
				1.0, local_xyb);
			++local_count;
		}
		if (y >= edgeSize && y + edgeSize < ysize)
		{
			unsigned int ix = (y - edgeSize) * xsize + x;
			unsigned int ix2 = ix + 2 * edgeSize * xsize;
			XybDiffLowFreqSquaredAccumulate(
				w * (r[ix] - r[ix2]),
				w * (g[ix] - g[ix2]),
				w * (b[ix] - b[ix2]),
				w * (r2[ix] - r2[ix2]),
				w * (g2[ix] - g2[ix2]),
				w * (b2[ix] - b2[ix2]),
				1.0, local_xyb);
			++local_count;
		}
	}

	const float weight = 0.01617112696f;
	const float mul = weight * 8.0f / local_count;
	for (int i = 0; i < 3; ++i)
	{
		diff_xyb[i] += mul * local_xyb[i];
	}
}

__device__ float DotProduct(__global const float u[3], const float v[3])
{
	return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

__device__ float Interpolate(__constant_ex const float *array, const int size, const float sx)
{
	float ix = _abs_d(sx);

	int baseix = (int)(ix);
	float res;
	if (baseix >= size - 1)
	{
		res = array[size - 1];
	}
	else
	{
		float mix = ix - baseix;
		int nextix = baseix + 1;
		res = array[baseix] + mix * (array[nextix] - array[baseix]);
	}
	if (sx < 0)
		res = -res;
	return res;
}

#define XybToVals_off_x 11.38708334481672f
#define XybToVals_inc_x 14.550189611520716f
__constant float XybToVals_lut_x[21] = {
	0,
	XybToVals_off_x,
	XybToVals_off_x + 1 * XybToVals_inc_x,
	XybToVals_off_x + 2 * XybToVals_inc_x,
	XybToVals_off_x + 3 * XybToVals_inc_x,
	XybToVals_off_x + 4 * XybToVals_inc_x,
	XybToVals_off_x + 5 * XybToVals_inc_x,
	XybToVals_off_x + 6 * XybToVals_inc_x,
	XybToVals_off_x + 7 * XybToVals_inc_x,
	XybToVals_off_x + 8 * XybToVals_inc_x,
	XybToVals_off_x + 9 * XybToVals_inc_x,
	XybToVals_off_x + 10 * XybToVals_inc_x,
	XybToVals_off_x + 11 * XybToVals_inc_x,
	XybToVals_off_x + 12 * XybToVals_inc_x,
	XybToVals_off_x + 13 * XybToVals_inc_x,
	XybToVals_off_x + 14 * XybToVals_inc_x,
	XybToVals_off_x + 15 * XybToVals_inc_x,
	XybToVals_off_x + 16 * XybToVals_inc_x,
	XybToVals_off_x + 17 * XybToVals_inc_x,
	XybToVals_off_x + 18 * XybToVals_inc_x,
	XybToVals_off_x + 19 * XybToVals_inc_x,
};

#define XybToVals_off_y 1.4103373714040413f
#define XybToVals_inc_y 0.7084088867024f
__constant float XybToVals_lut_y[21] = {
	0,
	XybToVals_off_y,
	XybToVals_off_y + 1 * XybToVals_inc_y,
	XybToVals_off_y + 2 * XybToVals_inc_y,
	XybToVals_off_y + 3 * XybToVals_inc_y,
	XybToVals_off_y + 4 * XybToVals_inc_y,
	XybToVals_off_y + 5 * XybToVals_inc_y,
	XybToVals_off_y + 6 * XybToVals_inc_y,
	XybToVals_off_y + 7 * XybToVals_inc_y,
	XybToVals_off_y + 8 * XybToVals_inc_y,
	XybToVals_off_y + 9 * XybToVals_inc_y,
	XybToVals_off_y + 10 * XybToVals_inc_y,
	XybToVals_off_y + 11 * XybToVals_inc_y,
	XybToVals_off_y + 12 * XybToVals_inc_y,
	XybToVals_off_y + 13 * XybToVals_inc_y,
	XybToVals_off_y + 14 * XybToVals_inc_y,
	XybToVals_off_y + 15 * XybToVals_inc_y,
	XybToVals_off_y + 16 * XybToVals_inc_y,
	XybToVals_off_y + 17 * XybToVals_inc_y,
	XybToVals_off_y + 18 * XybToVals_inc_y,
	XybToVals_off_y + 19 * XybToVals_inc_y,
};

__device__ void XybToVals(
	__private const float x, __private const float y, __private const float z,
	__private float *valx, __private float *valy, __private float *valz)
{
	const float xmul = 0.758304045695f;
	const float ymul = 2.28148649801f;
	const float zmul = 1.87816926918f;

	*valx = Interpolate(&XybToVals_lut_x[0], 21, x * xmul);
	*valy = Interpolate(&XybToVals_lut_y[0], 21, y * ymul);
	*valz = zmul * z;
}

#define XybLowFreqToVals_inc 5.2511644570349185f
__constant float XybLowFreqToVals_lut[21] = {
	0,
	1 * XybLowFreqToVals_inc,
	2 * XybLowFreqToVals_inc,
	3 * XybLowFreqToVals_inc,
	4 * XybLowFreqToVals_inc,
	5 * XybLowFreqToVals_inc,
	6 * XybLowFreqToVals_inc,
	7 * XybLowFreqToVals_inc,
	8 * XybLowFreqToVals_inc,
	9 * XybLowFreqToVals_inc,
	10 * XybLowFreqToVals_inc,
	11 * XybLowFreqToVals_inc,
	12 * XybLowFreqToVals_inc,
	13 * XybLowFreqToVals_inc,
	14 * XybLowFreqToVals_inc,
	15 * XybLowFreqToVals_inc,
	16 * XybLowFreqToVals_inc,
	17 * XybLowFreqToVals_inc,
	18 * XybLowFreqToVals_inc,
	19 * XybLowFreqToVals_inc,
	20 * XybLowFreqToVals_inc,
};

__device__ void XybLowFreqToVals(__private const float x, __private const float y, __private const float z,
								 __private float *valx, __private float *valy, __private float *valz)
{
	const float xmul = 6.64482198135f;
	const float ymul = 0.837846224276f;
	const float zmul = 7.34905756986f;
	const float y_to_z_mul = 0.0812519812628f;

	float zz = z + y_to_z_mul * y;
	*valz = zz * zmul;
	*valx = x * xmul;
	*valy = Interpolate(&XybLowFreqToVals_lut[0], 21, y * ymul);
}

__device__ float InterpolateClampNegative(__global const float *array,
										   int size, float sx)
{
	if (sx < 0)
	{
		sx = 0;
	}
	float ix = _abs_d(sx);
	int baseix = (int)(ix);
	float res;
	if (baseix >= size - 1)
	{
		res = array[size - 1];
	}
	else
	{
		float mix = ix - baseix;
		int nextix = baseix + 1;
		res = array[baseix] + mix * (array[nextix] - array[baseix]);
	}
	return res;
}

__device__ void XybDiffLowFreqSquaredAccumulate(__private const float r0, __private const float g0, __private const float b0,
												__private const float r1, __private const float g1, __private const float b1,
												__private const float factor, __private float res[3])
{
	float valx0, valy0, valz0;
	float valx1, valy1, valz1;
	XybLowFreqToVals(r0, g0, b0, &valx0, &valy0, &valz0);
	if (r1 == 0.0 && g1 == 0.0 && b1 == 0.0)
	{
		// PROFILER_ZONE("XybDiff r1=g1=b1=0");
		res[0] += factor * valx0 * valx0;
		res[1] += factor * valy0 * valy0;
		res[2] += factor * valz0 * valz0;
		return;
	}
	XybLowFreqToVals(r1, g1, b1, &valx1, &valy1, &valz1);
	// Approximate the distance of the colors by their respective distances
	// to gray.
	float valx = valx0 - valx1;
	float valy = valy0 - valy1;
	float valz = valz0 - valz1;
	res[0] += factor * valx * valx;
	res[1] += factor * valy * valy;
	res[2] += factor * valz * valz;
}

typedef struct __Complex
{
	float real;
	float imag;
} Complex;

__constant float kSqrtHalf = 0.70710678118654752440084436210484903f;
__device__ void RealFFT8(__private const float *in, __private Complex *out)
{
	float t1, t2, t3, t5, t6, t7, t8;
	t8 = in[6];
	t5 = in[2] - t8;
	t8 += in[2];
	out[2].real = t8;
	out[6].imag = -t5;
	out[4].imag = t5;
	t8 = in[4];
	t3 = in[0] - t8;
	t8 += in[0];
	out[0].real = t8;
	out[4].real = t3;
	out[6].real = t3;
	t7 = in[5];
	t3 = in[1] - t7;
	t7 += in[1];
	out[1].real = t7;
	t8 = in[7];
	t5 = in[3] - t8;
	t8 += in[3];
	out[3].real = t8;
	t2 = -t5;
	t6 = t3 - t5;
	t8 = kSqrtHalf;
	t6 *= t8;
	out[5].real = out[4].real - t6;
	t1 = t3 + t5;
	t1 *= t8;
	out[5].imag = out[4].imag - t1;
	t6 += out[4].real;
	out[4].real = t6;
	t1 += out[4].imag;
	out[4].imag = t1;
	t5 = t2 - t3;
	t5 *= t8;
	out[7].imag = out[6].imag - t5;
	t2 += t3;
	t2 *= t8;
	out[7].real = out[6].real - t2;
	t2 += out[6].real;
	out[6].real = t2;
	t5 += out[6].imag;
	out[6].imag = t5;
	t5 = out[2].real;
	t1 = out[0].real - t5;
	t7 = out[3].real;
	t5 += out[0].real;
	t3 = out[1].real - t7;
	t7 += out[1].real;
	t8 = t5 + t7;
	out[0].real = t8;
	t5 -= t7;
	out[1].real = t5;
	out[2].imag = t3;
	out[3].imag = -t3;
	out[3].real = t1;
	out[2].real = t1;
	out[0].imag = 0;
	out[1].imag = 0;

	// Reorder to the correct output order.
	// TODO: Modify the above computation so that this is not needed.
	__private Complex tmp = out[2];
	out[2] = out[3];
	out[3] = out[5];
	out[5] = out[7];
	out[7] = out[4];
	out[4] = out[1];
	out[1] = out[6];
	out[6] = tmp;
}

__device__ void TransposeBlock(Complex data[kBlockSize])
{
	for (int i = 0; i < kBlockEdge; i++)
	{
		for (int j = 0; j < i; j++)
		{
			Complex tmp = data[kBlockEdge * i + j];
			data[kBlockEdge * i + j] = data[kBlockEdge * j + i];
			data[kBlockEdge * j + i] = tmp;
		}
	}
}

//  D. J. Bernstein's Fast Fourier Transform algorithm on 4 elements.
__device__ void FFT4(__private Complex *a)
{
	float t1, t2, t3, t4, t5, t6, t7, t8;
	t5 = a[2].real;
	t1 = a[0].real - t5;
	t7 = a[3].real;
	t5 += a[0].real;
	t3 = a[1].real - t7;
	t7 += a[1].real;
	t8 = t5 + t7;
	a[0].real = t8;
	t5 -= t7;
	a[1].real = t5;
	t6 = a[2].imag;
	t2 = a[0].imag - t6;
	t6 += a[0].imag;
	t5 = a[3].imag;
	a[2].imag = t2 + t3;
	t2 -= t3;
	a[3].imag = t2;
	t4 = a[1].imag - t5;
	a[3].real = t1 + t4;
	t1 -= t4;
	a[2].real = t1;
	t5 += a[1].imag;
	a[0].imag = t6 + t5;
	t6 -= t5;
	a[1].imag = t6;
}

//  D. J. Bernstein's Fast Fourier Transform algorithm on 8 elements.
__device__ void FFT8(__private Complex *a)
{
	const float kSqrtHalf = 0.70710678118654752440084436210484903f;
	float t1, t2, t3, t4, t5, t6, t7, t8;

	t7 = a[4].imag;
	t4 = a[0].imag - t7;
	t7 += a[0].imag;
	a[0].imag = t7;

	t8 = a[6].real;
	t5 = a[2].real - t8;
	t8 += a[2].real;
	a[2].real = t8;

	t7 = a[6].imag;
	a[6].imag = t4 - t5;
	t4 += t5;
	a[4].imag = t4;

	t6 = a[2].imag - t7;
	t7 += a[2].imag;
	a[2].imag = t7;

	t8 = a[4].real;
	t3 = a[0].real - t8;
	t8 += a[0].real;
	a[0].real = t8;

	a[4].real = t3 - t6;
	t3 += t6;
	a[6].real = t3;

	t7 = a[5].real;
	t3 = a[1].real - t7;
	t7 += a[1].real;
	a[1].real = t7;

	t8 = a[7].imag;
	t6 = a[3].imag - t8;
	t8 += a[3].imag;
	a[3].imag = t8;
	t1 = t3 - t6;
	t3 += t6;

	t7 = a[5].imag;
	t4 = a[1].imag - t7;
	t7 += a[1].imag;
	a[1].imag = t7;

	t8 = a[7].real;
	t5 = a[3].real - t8;
	t8 += a[3].real;
	a[3].real = t8;

	t2 = t4 - t5;
	t4 += t5;

	t6 = t1 - t4;
	t8 = kSqrtHalf;
	t6 *= t8;
	a[5].real = a[4].real - t6;
	t1 += t4;
	t1 *= t8;
	a[5].imag = a[4].imag - t1;
	t6 += a[4].real;
	a[4].real = t6;
	t1 += a[4].imag;
	a[4].imag = t1;

	t5 = t2 - t3;
	t5 *= t8;
	a[7].imag = a[6].imag - t5;
	t2 += t3;
	t2 *= t8;
	a[7].real = a[6].real - t2;
	t2 += a[6].real;
	a[6].real = t2;
	t5 += a[6].imag;
	a[6].imag = t5;

	FFT4(a);

	// Reorder to the correct output order.
	// TODO: Modify the above computation so that this is not needed.
	Complex tmp = a[2];
	a[2] = a[3];
	a[3] = a[5];
	a[5] = a[7];
	a[7] = a[4];
	a[4] = a[1];
	a[1] = a[6];
	a[6] = tmp;
}

__device__ float abssq(const Complex c)
{
	return c.real * c.real + c.imag * c.imag;
}

__device__ void ButteraugliFFTSquared(__private float block[kBlockSize])
{
	const float global_mul = 0.000064f;
	__private Complex block_c[kBlockSize];

	for (int y = 0; y < kBlockEdge; ++y)
	{
		__private const float *input_ptr = block + y * kBlockEdge;
		__private Complex *output_ptr = block_c + y * kBlockEdge;
		RealFFT8(input_ptr, output_ptr);
	}
	TransposeBlock(block_c);
	__private float r0[kBlockEdge];
	__private float r1[kBlockEdge];
	for (int x = 0; x < kBlockEdge; ++x)
	{
		r0[x] = block_c[x].real;
		r1[x] = block_c[kBlockHalf + x].real;
	}
	RealFFT8(r0, block_c);
	RealFFT8(r1, block_c + kBlockHalf);
	for (int y = 1; y < kBlockEdgeHalf; ++y)
	{
		__private Complex *output_ptr = block_c + y * kBlockEdge;
		FFT8(output_ptr);
	}
	for (int i = kBlockEdgeHalf; i < kBlockHalf + kBlockEdgeHalf + 1; ++i)
	{
		block[i] = abssq(block_c[i]);
		block[i] *= global_mul;
	}
}

__device__ float RemoveRangeAroundZero(float v, float range)
{
	if (v >= -range && v < range)
	{
		return 0;
	}
	if (v < 0)
	{
		return v + range;
	}
	else
	{
		return v - range;
	}
}

#define MakeHighFreqColorDiffDy_off 1.4103373714040413f
#define MakeHighFreqColorDiffDy_inc 0.7084088867024f
__constant float MakeHighFreqColorDiffDy_lut[21] = {
	0.0,
	MakeHighFreqColorDiffDy_off,
	MakeHighFreqColorDiffDy_off + 1 * MakeHighFreqColorDiffDy_inc,
	MakeHighFreqColorDiffDy_off + 2 * MakeHighFreqColorDiffDy_inc,
	MakeHighFreqColorDiffDy_off + 3 * MakeHighFreqColorDiffDy_inc,
	MakeHighFreqColorDiffDy_off + 4 * MakeHighFreqColorDiffDy_inc,
	MakeHighFreqColorDiffDy_off + 5 * MakeHighFreqColorDiffDy_inc,
	MakeHighFreqColorDiffDy_off + 6 * MakeHighFreqColorDiffDy_inc,
	MakeHighFreqColorDiffDy_off + 7 * MakeHighFreqColorDiffDy_inc,
	MakeHighFreqColorDiffDy_off + 8 * MakeHighFreqColorDiffDy_inc,
	MakeHighFreqColorDiffDy_off + 9 * MakeHighFreqColorDiffDy_inc,
	MakeHighFreqColorDiffDy_off + 10 * MakeHighFreqColorDiffDy_inc,
	MakeHighFreqColorDiffDy_off + 11 * MakeHighFreqColorDiffDy_inc,
	MakeHighFreqColorDiffDy_off + 12 * MakeHighFreqColorDiffDy_inc,
	MakeHighFreqColorDiffDy_off + 13 * MakeHighFreqColorDiffDy_inc,
	MakeHighFreqColorDiffDy_off + 14 * MakeHighFreqColorDiffDy_inc,
	MakeHighFreqColorDiffDy_off + 15 * MakeHighFreqColorDiffDy_inc,
	MakeHighFreqColorDiffDy_off + 16 * MakeHighFreqColorDiffDy_inc,
	MakeHighFreqColorDiffDy_off + 17 * MakeHighFreqColorDiffDy_inc,
	MakeHighFreqColorDiffDy_off + 18 * MakeHighFreqColorDiffDy_inc,
	MakeHighFreqColorDiffDy_off + 19 * MakeHighFreqColorDiffDy_inc,
};

__constant float csf8x8[kBlockHalf + kBlockEdgeHalf + 1] = {
	5.28270670524f,
	0.0f,
	0.0f,
	0.0f,
	0.3831134973f,
	0.676303603859f,
	3.58927792424f,
	18.6104367002f,
	18.6104367002f,
	3.09093131948f,
	1.0f,
	0.498250875965f,
	0.36198671102f,
	0.308982169883f,
	0.1312701920435f,
	2.37370549629f,
	3.58927792424f,
	1.0f,
	2.37370549629f,
	0.991205724152f,
	1.05178802919f,
	0.627264168628f,
	0.4f,
	0.1312701920435f,
	0.676303603859f,
	0.498250875965f,
	0.991205724152f,
	0.5f,
	0.3831134973f,
	0.349686450518f,
	0.627264168628f,
	0.308982169883f,
	0.3831134973f,
	0.36198671102f,
	1.05178802919f,
	0.3831134973f,
	0.12f,
};

// Computes 8x8 FFT of each channel of xyb0 and xyb1 and adds the total squared
// 3-dimensional xybdiff of the two blocks to diff_xyb_{dc,ac} and the average
// diff on the edges to diff_xyb_edge_dc.
__device__ void ButteraugliBlockDiff(__private float xyb0[3 * kBlockSize],
									 __private float xyb1[3 * kBlockSize],
									 __private float diff_xyb_dc[3],
									 __private float diff_xyb_ac[3],
									 __private float diff_xyb_edge_dc[3])
{

	__private float avgdiff_xyb[3] = {0.0};
	__private float avgdiff_edge[3][4] = {{0.0}};

	for (int i = 0; i < 3 * kBlockSize; ++i)
	{
		const float diff_xyb = xyb0[i] - xyb1[i];
		const int c = i / kBlockSize;
		avgdiff_xyb[c] += diff_xyb / kBlockSize;
		const int k = i % kBlockSize;
		const int kx = k % kBlockEdge;
		const int ky = k / kBlockEdge;
		const int h_edge_idx = ky == 0 ? 1 : ky == 7 ? 3
													 : -1;
		const int v_edge_idx = kx == 0 ? 0 : kx == 7 ? 2
													 : -1;
		if (h_edge_idx >= 0)
		{
			avgdiff_edge[c][h_edge_idx] += diff_xyb / kBlockEdge;
		}
		if (v_edge_idx >= 0)
		{
			avgdiff_edge[c][v_edge_idx] += diff_xyb / kBlockEdge;
		}
	}
	XybDiffLowFreqSquaredAccumulate(avgdiff_xyb[0],
									avgdiff_xyb[1],
									avgdiff_xyb[2],
									0, 0, 0, csf8x8[0],
									diff_xyb_dc);
	for (int i = 0; i < 4; ++i)
	{
		XybDiffLowFreqSquaredAccumulate(avgdiff_edge[0][i],
										avgdiff_edge[1][i],
										avgdiff_edge[2][i],
										0, 0, 0, csf8x8[0],
										diff_xyb_edge_dc);
	}

	__private float *xyb_avg = xyb0;
	__private float *xyb_halfdiff = xyb1;
	for (int i = 0; i < 3 * kBlockSize; ++i)
	{
		float avg = (xyb0[i] + xyb1[i]) / 2;
		float halfdiff = (xyb0[i] - xyb1[i]) / 2;
		xyb_avg[i] = avg;
		xyb_halfdiff[i] = halfdiff;
	}
	__private float *y_avg = &xyb_avg[kBlockSize];
	__private float *x_halfdiff_squared = &xyb_halfdiff[0];
	__private float *y_halfdiff = &xyb_halfdiff[kBlockSize];
	__private float *z_halfdiff_squared = &xyb_halfdiff[2 * kBlockSize];
	ButteraugliFFTSquared(y_avg);
	ButteraugliFFTSquared(x_halfdiff_squared);
	ButteraugliFFTSquared(y_halfdiff);
	ButteraugliFFTSquared(z_halfdiff_squared);

	const float xmul = 64.8f;
	const float ymul = 1.753123908348329f;
	const float ymul2 = 1.51983458269f;
	const float zmul = 2.4f;

	for (unsigned int i = kBlockEdgeHalf; i < kBlockHalf + kBlockEdgeHalf + 1; ++i)
	{
		float d = csf8x8[i];
		diff_xyb_ac[0] += d * xmul * x_halfdiff_squared[i];
		diff_xyb_ac[2] += d * zmul * z_halfdiff_squared[i];

		y_avg[i] = sqrt(y_avg[i]);
		y_halfdiff[i] = sqrt(y_halfdiff[i]);
		float y0 = y_avg[i] - y_halfdiff[i];
		float y1 = y_avg[i] + y_halfdiff[i];
		// Remove the impact of small absolute values.
		// This improves the behavior with flat noise.
		const float ylimit = 0.04f;
		y0 = RemoveRangeAroundZero(y0, ylimit);
		y1 = RemoveRangeAroundZero(y1, ylimit);
		if (y0 != y1)
		{
			float valy0 = Interpolate(&MakeHighFreqColorDiffDy_lut[0], 21, y0 * ymul2);
			float valy1 = Interpolate(&MakeHighFreqColorDiffDy_lut[0], 21, y1 * ymul2);
			float valy = ymul * (valy0 - valy1);
			diff_xyb_ac[1] += d * valy * valy;
		}
	}
}

__constant static float g_mix[12] = {
	0.348036746003f,
	0.577814843137f,
	0.0544556093735f,
	0.774145581713f,
	0.26922717275f,
	0.767247733938f,
	0.0366922708552f,
	0.920130265014f,
	0.0882062883536f,
	0.158581714673f,
	0.712857943858f,
	10.6524069248f,
};

__device__ void OpsinAbsorbance(__private const float in[3], __private float out[3])
{
	out[0] = g_mix[0] * in[0] + g_mix[1] * in[1] + g_mix[2] * in[2] + g_mix[3];
	out[1] = g_mix[4] * in[0] + g_mix[5] * in[1] + g_mix[6] * in[2] + g_mix[7];
	out[2] = g_mix[8] * in[0] + g_mix[9] * in[1] + g_mix[10] * in[2] + g_mix[11];
}

__device__ float EvaluatePolynomial(const float x, __constant_ex const float *coefficients, int n)
{
	float b1 = 0.0f;
	float b2 = 0.0f;

	for (int i = n - 1; i >= 0; i--)
	{
		if (i == 0)
		{
			const float x_b1 = x * b1;
			b1 = x_b1 - b2 + coefficients[0];
			break;
		}
		const float x_b1 = x * b1;
		const float t = (x_b1 + x_b1) - b2 + coefficients[i];
		b2 = b1;
		b1 = t;
	}

	return b1;
}

static __constant float g_gamma_p[5 + 1] = {
	881.979476556478289f,
	1496.058452015812463f,
	908.662212739659481f,
	373.566100223287378f,
	85.840860336314364f,
	6.683258861509244f,
};

static __constant float g_gamma_q[5 + 1] = {
	12.262350348616792f,
	20.557285797683576f,
	12.161463238367844f,
	4.711532733641639f,
	0.899112889751053f,
	0.035662329617191f,
};

__device__ float Gamma(float v)
{
	const float min_value = 0.770000000000000f;
	const float max_value = 274.579999999999984f;
	const float x01 = (v - min_value) / (max_value - min_value);
	const float xc = 2.0f * x01 - 1.0f;

	const float yp = EvaluatePolynomial(xc, g_gamma_p, 6);
	const float yq = EvaluatePolynomial(xc, g_gamma_q, 6);
	if (yq == 0.0)
		return 0.0f;
	return (float)(yp / yq);
}

__device__ void RgbToXyb(__private const float r, __private const float g, __private const float b, __private float *valx, __private float *valy, __private float *valz)
{
	const float a0 = 1.01611726948f;
	const float a1 = 0.982482243696f;
	const float a2 = 1.43571362627f;
	const float a3 = 0.896039849412f;
	*valx = a0 * r - a1 * g;
	*valy = a2 * r + a3 * g;
	*valz = b;
}

__device__ int list_push_back(__private IntFloatPairList *list, int i, float f)
{
	list->pData[list->size].idx = i;
	list->pData[list->size].err = f;
	return ++list->size;
}

__device__ int list_erase(__private IntFloatPairList *list, const int idx)
{
	for (int i = idx; i < list->size - 1; i++)
	{
		list->pData[i].idx = list->pData[i + 1].idx;
		list->pData[i].err = list->pData[i + 1].err;
	}
	return --list->size;
}

__device__ int SortInputOrder(__private DCTScoreData *input_order, int size)
{
	int i, j;
	DCTScoreData tmp;
	for (j = 1; j < size; j++)
	{
		tmp.idx = input_order[j].idx;
		tmp.err = input_order[j].err;

		i = j - 1;
		while (i >= 0 && input_order[i].err > tmp.err)
		{
			input_order[i + 1].idx = input_order[i].idx;
			input_order[i + 1].err = input_order[i].err;
			i--;
		}
		input_order[i + 1].idx = tmp.idx;
		input_order[i + 1].err = tmp.err;
	}
	return size;
}

__constant static float csf[192] = {
	0.0f,
	1.71014f,
	0.298711f,
	0.233709f,
	0.223126f,
	0.207072f,
	0.192775f,
	0.161201f,
	2.05807f,
	0.222927f,
	0.203406f,
	0.188465f,
	0.184668f,
	0.169993f,
	0.159142f,
	0.130155f,
	0.430518f,
	0.204939f,
	0.206655f,
	0.192231f,
	0.182941f,
	0.169455f,
	0.157599f,
	0.127153f,
	0.234757f,
	0.191098f,
	0.192698f,
	0.17425f,
	0.166503f,
	0.142154f,
	0.126182f,
	0.104196f,
	0.226117f,
	0.185373f,
	0.183825f,
	0.166643f,
	0.159414f,
	0.12636f,
	0.108696f,
	0.0911974f,
	0.207463f,
	0.171517f,
	0.170124f,
	0.141582f,
	0.126213f,
	0.103627f,
	0.0882436f,
	0.0751848f,
	0.196436f,
	0.161947f,
	0.159271f,
	0.126938f,
	0.109125f,
	0.0878027f,
	0.0749842f,
	0.0633859f,
	0.165232f,
	0.132905f,
	0.128679f,
	0.105766f,
	0.0906087f,
	0.0751544f,
	0.0641187f,
	0.0529921f,
	0.0f,
	0.147235f,
	0.11264f,
	0.0757892f,
	0.0493929f,
	0.0280663f,
	0.0075012f,
	-0.000945567f,
	0.149251f,
	0.0964806f,
	0.0786224f,
	0.05206f,
	0.0292758f,
	0.00353094f,
	-0.00277912f,
	-0.00404481f,
	0.115551f,
	0.0793142f,
	0.0623735f,
	0.0405019f,
	0.0152656f,
	-0.00145742f,
	-0.00370369f,
	-0.00375106f,
	0.0791547f,
	0.0537506f,
	0.0413634f,
	0.0193486f,
	0.000609066f,
	-0.00510923f,
	-0.0046452f,
	-0.00385187f,
	0.0544534f,
	0.0334066f,
	0.0153899f,
	0.000539088f,
	-0.00356085f,
	-0.00535661f,
	-0.00429145f,
	-0.00343131f,
	0.0356439f,
	0.00865645f,
	0.00165229f,
	-0.00425931f,
	-0.00507324f,
	-0.00459083f,
	-0.003703f,
	-0.00310327f,
	0.0121926f,
	-0.0009259f,
	-0.00330991f,
	-0.00499378f,
	-0.00437381f,
	-0.00377427f,
	-0.00311731f,
	-0.00255125f,
	-0.000320593f,
	-0.00426043f,
	-0.00416549f,
	-0.00419364f,
	-0.00365418f,
	-0.00317499f,
	-0.00255932f,
	-0.00217917f,
	0.0f,
	0.143471f,
	0.124336f,
	0.0947465f,
	0.0814066f,
	0.0686776f,
	0.0588122f,
	0.0374415f,
	0.146315f,
	0.105334f,
	0.0949415f,
	0.0784241f,
	0.0689064f,
	0.0588304f,
	0.0495961f,
	0.0202342f,
	0.123818f,
	0.0952654f,
	0.0860556f,
	0.0724158f,
	0.0628307f,
	0.0529965f,
	0.0353941f,
	0.00815821f,
	0.097054f,
	0.080422f,
	0.0731085f,
	0.0636154f,
	0.055606f,
	0.0384127f,
	0.0142879f,
	0.00105195f,
	0.0849312f,
	0.071115f,
	0.0631183f,
	0.0552972f,
	0.0369221f,
	0.00798314f,
	0.000716374f,
	-0.00200948f,
	0.0722298f,
	0.0599559f,
	0.054841f,
	0.0387529f,
	0.0107262f,
	0.000355315f,
	-0.00244803f,
	-0.00335222f,
	0.0635335f,
	0.0514196f,
	0.0406309f,
	0.0125833f,
	0.00151305f,
	-0.00140269f,
	-0.00362547f,
	-0.00337649f,
	0.0472024f,
	0.0198725f,
	0.0113437f,
	0.00266305f,
	-0.00137183f,
	-0.00354158f,
	-0.00341292f,
	-0.00290074f};

__constant static float bias[192] = {
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0};

__device__ int MakeInputOrder(__global const coeff_t *block, __global const coeff_t *orig_block, __private IntFloatPairList *input_order, const int block_size)
{
	int size = 0;
	for (int c = 0; c < 3; ++c)
	{
		for (int k = 1; k < block_size; ++k)
		{
			int idx = c * block_size + k;
			if (block[idx] != 0)
			{
				float score = _abs_f(orig_block[idx]) * csf[idx] + bias[idx];
				size = list_push_back(input_order, idx, score);
			}
		}
	}
	return SortInputOrder(input_order->pData, size);
}

__constant static int kIDCTMatrix[kDCTBlockSize] = {
	8192,
	11363,
	10703,
	9633,
	8192,
	6437,
	4433,
	2260,
	8192,
	9633,
	4433,
	-2259,
	-8192,
	-11362,
	-10704,
	-6436,
	8192,
	6437,
	-4433,
	-11362,
	-8192,
	2261,
	10704,
	9633,
	8192,
	2260,
	-10703,
	-6436,
	8192,
	9633,
	-4433,
	-11363,
	8192,
	-2260,
	-10703,
	6436,
	8192,
	-9633,
	-4433,
	11363,
	8192,
	-6437,
	-4433,
	11362,
	-8192,
	-2261,
	10704,
	-9633,
	8192,
	-9633,
	4433,
	2259,
	-8192,
	11362,
	-10704,
	6436,
	8192,
	-11363,
	10703,
	-9633,
	8192,
	-6437,
	4433,
	-2260,
};

// Computes out[x] = sum{kIDCTMatrix[8*x+u]*in[u*stride]; for u in [0..7]}
__device__ void Compute1dIDCT(__private const coeff_t *in, const int stride, int out[8])
{
	int tmp0, tmp1, tmp2, tmp3, tmp4;

	tmp1 = kIDCTMatrix[0] * in[0];
	out[0] = out[1] = out[2] = out[3] = out[4] = out[5] = out[6] = out[7] = tmp1;

	tmp0 = in[stride];
	tmp1 = kIDCTMatrix[1] * tmp0;
	tmp2 = kIDCTMatrix[9] * tmp0;
	tmp3 = kIDCTMatrix[17] * tmp0;
	tmp4 = kIDCTMatrix[25] * tmp0;
	out[0] += tmp1;
	out[1] += tmp2;
	out[2] += tmp3;
	out[3] += tmp4;
	out[4] -= tmp4;
	out[5] -= tmp3;
	out[6] -= tmp2;
	out[7] -= tmp1;

	tmp0 = in[2 * stride];
	tmp1 = kIDCTMatrix[2] * tmp0;
	tmp2 = kIDCTMatrix[10] * tmp0;
	out[0] += tmp1;
	out[1] += tmp2;
	out[2] -= tmp2;
	out[3] -= tmp1;
	out[4] -= tmp1;
	out[5] -= tmp2;
	out[6] += tmp2;
	out[7] += tmp1;

	tmp0 = in[3 * stride];
	tmp1 = kIDCTMatrix[3] * tmp0;
	tmp2 = kIDCTMatrix[11] * tmp0;
	tmp3 = kIDCTMatrix[19] * tmp0;
	tmp4 = kIDCTMatrix[27] * tmp0;
	out[0] += tmp1;
	out[1] += tmp2;
	out[2] += tmp3;
	out[3] += tmp4;
	out[4] -= tmp4;
	out[5] -= tmp3;
	out[6] -= tmp2;
	out[7] -= tmp1;

	tmp0 = in[4 * stride];
	tmp1 = kIDCTMatrix[4] * tmp0;
	out[0] += tmp1;
	out[1] -= tmp1;
	out[2] -= tmp1;
	out[3] += tmp1;
	out[4] += tmp1;
	out[5] -= tmp1;
	out[6] -= tmp1;
	out[7] += tmp1;

	tmp0 = in[5 * stride];
	tmp1 = kIDCTMatrix[5] * tmp0;
	tmp2 = kIDCTMatrix[13] * tmp0;
	tmp3 = kIDCTMatrix[21] * tmp0;
	tmp4 = kIDCTMatrix[29] * tmp0;
	out[0] += tmp1;
	out[1] += tmp2;
	out[2] += tmp3;
	out[3] += tmp4;
	out[4] -= tmp4;
	out[5] -= tmp3;
	out[6] -= tmp2;
	out[7] -= tmp1;

	tmp0 = in[6 * stride];
	tmp1 = kIDCTMatrix[6] * tmp0;
	tmp2 = kIDCTMatrix[14] * tmp0;
	out[0] += tmp1;
	out[1] += tmp2;
	out[2] -= tmp2;
	out[3] -= tmp1;
	out[4] -= tmp1;
	out[5] -= tmp2;
	out[6] += tmp2;
	out[7] += tmp1;

	tmp0 = in[7 * stride];
	tmp1 = kIDCTMatrix[7] * tmp0;
	tmp2 = kIDCTMatrix[15] * tmp0;
	tmp3 = kIDCTMatrix[23] * tmp0;
	tmp4 = kIDCTMatrix[31] * tmp0;
	out[0] += tmp1;
	out[1] += tmp2;
	out[2] += tmp3;
	out[3] += tmp4;
	out[4] -= tmp4;
	out[5] -= tmp3;
	out[6] -= tmp2;
	out[7] -= tmp1;
}

__device__ void CoeffToIDCT(__private const coeff_t block[8 * 8], uchar out[8 * 8])
{
	coeff_t colidcts[kDCTBlockSize];
	const int kColScale = 11;
	const int kColRound = 1 << (kColScale - 1);
	for (int x = 0; x < 8; ++x)
	{
		int colbuf[8] = {0};
		Compute1dIDCT(&block[x], 8, colbuf);
		for (int y = 0; y < 8; ++y)
		{
			colidcts[8 * y + x] = (colbuf[y] + kColRound) >> kColScale;
		}
	}
	const int kRowScale = 18;
	const int kRowRound = 257 << (kRowScale - 1); // includes offset by 128
	for (int y = 0; y < 8; ++y)
	{
		const int rowidx = 8 * y;
		int rowbuf[8] = {0};
		Compute1dIDCT(&colidcts[rowidx], 1, rowbuf);
		for (int x = 0; x < 8; ++x)
		{
			out[rowidx + x] = _max_i(0, _min_i(255, (rowbuf[x] + kRowRound) >> kRowScale));
		}
	}
}

__device__ void IDCTToPixel8x8(const uchar idct[8 * 8], ushort pixels_[8 * 8])
{
	const int block_x = 0;
	const int block_y = 0;
	const int width_ = 8;
	const int height_ = 8;

	for (int iy = 0; iy < 8; ++iy)
	{
		for (int ix = 0; ix < 8; ++ix)
		{
			int x = 8 * block_x + ix;
			int y = 8 * block_y + iy;
			if (x >= width_ || y >= height_)
				continue;
			int p = y * width_ + x;
			pixels_[p] = idct[8 * iy + ix] << 4;
		}
	}
}

__device__ void IDCTToPixel16x16(const uchar idct[8 * 8], ushort pixels_out[16 * 16], __global const ushort *pixel_orig, int block_x, int block_y, int width_, int height_)
{
	// Fill in the 10x10 pixel area in the subsampled image that will be the
	// basis of the upsampling. This area is enough to hold the 3x3 kernel of
	// the fancy upsampler around each pixel.
#define kSubsampledEdgeSize 10
	ushort subsampled[kSubsampledEdgeSize * kSubsampledEdgeSize];
	for (int j = 0; j < kSubsampledEdgeSize; ++j)
	{
		// The order we fill in the rows is:
		//   8 rows intersecting the block, row below, row above
		const int y0 = block_y * 16 + (j < 9 ? j * 2 : -2);
		for (int i = 0; i < kSubsampledEdgeSize; ++i)
		{
			// The order we fill in each row is:
			//   8 pixels within the block, left edge, right edge
			const int ix = ((j < 9 ? (j + 1) * kSubsampledEdgeSize : 0) +
							(i < 9 ? i + 1 : 0));
			const int x0 = block_x * 16 + (i < 9 ? i * 2 : -2);
			if (x0 < 0)
			{
				subsampled[ix] = subsampled[ix + 1];
			}
			else if (y0 < 0)
			{
				subsampled[ix] = subsampled[ix + kSubsampledEdgeSize];
			}
			else if (x0 >= width_)
			{
				subsampled[ix] = subsampled[ix - 1];
			}
			else if (y0 >= height_)
			{
				subsampled[ix] = subsampled[ix - kSubsampledEdgeSize];
			}
			else if (i < 8 && j < 8)
			{
				subsampled[ix] = idct[j * 8 + i] << 4;
			}
			else
			{
				// Reconstruct the subsampled pixels around the edge of the current
				// block by computing the inverse of the fancy upsampler.
				const int y1 = _max_i(y0 - 1, 0);
				const int x1 = _max_i(x0 - 1, 0);
				subsampled[ix] = (pixel_orig[y0 * width_ + x0] * 9 +
								  pixel_orig[y1 * width_ + x1] +
								  pixel_orig[y0 * width_ + x1] * -3 +
								  pixel_orig[y1 * width_ + x0] * -3) >>
								 2;
			}
		}
	}
	// Determine area to update.
	int xmin = block_x * 16; // std::max(block_x * 16 - 1, 0);
	int xmax = _min_i(block_x * 16 + 15, width_ - 1);
	int ymin = block_y * 16; // std::max(block_y * 16 - 1, 0);
	int ymax = _min_i(block_y * 16 + 15, height_ - 1);

	// Apply the fancy upsampler on the subsampled block.
	for (int y = ymin; y <= ymax; ++y)
	{
		const int y0 = ((y & ~1) / 2 - block_y * 8 + 1) * kSubsampledEdgeSize;
		const int dy = ((y & 1) * 2 - 1) * kSubsampledEdgeSize;
		for (int x = xmin; x <= xmax; ++x)
		{
			const int x0 = (x & ~1) / 2 - block_x * 8 + 1;
			const int dx = (x & 1) * 2 - 1;
			const int ix = x0 + y0;

			int out_x = x - xmin;
			int out_y = y - ymin;

			pixels_out[out_y * 16 + out_x] = (subsampled[ix] * 9 + subsampled[ix + dy] * 3 +
											  subsampled[ix + dx] * 3 + subsampled[ix + dx + dy]) >>
											 4;
		}
	}
}

// out = [YUVYUV....YUVYUV]
__device__ void PixelToYUV(ushort pixels_[8 * 8], __private uchar out[8 * 8], int xsize /* = 8*/, int ysize /* = 8*/)
{
	const int stride = 3;

	for (int y = 0; y < xsize; ++y)
	{
		for (int x = 0; x < ysize; ++x)
		{
			int px = y * xsize + x;
			*out = (uchar)((pixels_[px] + 8 - (x & 1)) >> 4);
			out += stride;
		}
	}
}

__constant static int kCrToRedTable[256] = {
	-179, -178, -177, -175, -174, -172, -171, -170, -168, -167, -165, -164,
	-163, -161, -160, -158, -157, -156, -154, -153, -151, -150, -149, -147,
	-146, -144, -143, -142, -140, -139, -137, -136, -135, -133, -132, -130,
	-129, -128, -126, -125, -123, -122, -121, -119, -118, -116, -115, -114,
	-112, -111, -109, -108, -107, -105, -104, -102, -101, -100, -98, -97,
	-95, -94, -93, -91, -90, -88, -87, -86, -84, -83, -81, -80,
	-79, -77, -76, -74, -73, -72, -70, -69, -67, -66, -64, -63,
	-62, -60, -59, -57, -56, -55, -53, -52, -50, -49, -48, -46,
	-45, -43, -42, -41, -39, -38, -36, -35, -34, -32, -31, -29,
	-28, -27, -25, -24, -22, -21, -20, -18, -17, -15, -14, -13,
	-11, -10, -8, -7, -6, -4, -3, -1, 0, 1, 3, 4,
	6, 7, 8, 10, 11, 13, 14, 15, 17, 18, 20, 21,
	22, 24, 25, 27, 28, 29, 31, 32, 34, 35, 36, 38,
	39, 41, 42, 43, 45, 46, 48, 49, 50, 52, 53, 55,
	56, 57, 59, 60, 62, 63, 64, 66, 67, 69, 70, 72,
	73, 74, 76, 77, 79, 80, 81, 83, 84, 86, 87, 88,
	90, 91, 93, 94, 95, 97, 98, 100, 101, 102, 104, 105,
	107, 108, 109, 111, 112, 114, 115, 116, 118, 119, 121, 122,
	123, 125, 126, 128, 129, 130, 132, 133, 135, 136, 137, 139,
	140, 142, 143, 144, 146, 147, 149, 150, 151, 153, 154, 156,
	157, 158, 160, 161, 163, 164, 165, 167, 168, 170, 171, 172,
	174, 175, 177, 178};

__constant static int kCbToBlueTable[256] = {
	-227,
	-225,
	-223,
	-222,
	-220,
	-218,
	-216,
	-214,
	-213,
	-211,
	-209,
	-207,
	-206,
	-204,
	-202,
	-200,
	-198,
	-197,
	-195,
	-193,
	-191,
	-190,
	-188,
	-186,
	-184,
	-183,
	-181,
	-179,
	-177,
	-175,
	-174,
	-172,
	-170,
	-168,
	-167,
	-165,
	-163,
	-161,
	-159,
	-158,
	-156,
	-154,
	-152,
	-151,
	-149,
	-147,
	-145,
	-144,
	-142,
	-140,
	-138,
	-136,
	-135,
	-133,
	-131,
	-129,
	-128,
	-126,
	-124,
	-122,
	-120,
	-119,
	-117,
	-115,
	-113,
	-112,
	-110,
	-108,
	-106,
	-105,
	-103,
	-101,
	-99,
	-97,
	-96,
	-94,
	-92,
	-90,
	-89,
	-87,
	-85,
	-83,
	-82,
	-80,
	-78,
	-76,
	-74,
	-73,
	-71,
	-69,
	-67,
	-66,
	-64,
	-62,
	-60,
	-58,
	-57,
	-55,
	-53,
	-51,
	-50,
	-48,
	-46,
	-44,
	-43,
	-41,
	-39,
	-37,
	-35,
	-34,
	-32,
	-30,
	-28,
	-27,
	-25,
	-23,
	-21,
	-19,
	-18,
	-16,
	-14,
	-12,
	-11,
	-9,
	-7,
	-5,
	-4,
	-2,
	0,
	2,
	4,
	5,
	7,
	9,
	11,
	12,
	14,
	16,
	18,
	19,
	21,
	23,
	25,
	27,
	28,
	30,
	32,
	34,
	35,
	37,
	39,
	41,
	43,
	44,
	46,
	48,
	50,
	51,
	53,
	55,
	57,
	58,
	60,
	62,
	64,
	66,
	67,
	69,
	71,
	73,
	74,
	76,
	78,
	80,
	82,
	83,
	85,
	87,
	89,
	90,
	92,
	94,
	96,
	97,
	99,
	101,
	103,
	105,
	106,
	108,
	110,
	112,
	113,
	115,
	117,
	119,
	120,
	122,
	124,
	126,
	128,
	129,
	131,
	133,
	135,
	136,
	138,
	140,
	142,
	144,
	145,
	147,
	149,
	151,
	152,
	154,
	156,
	158,
	159,
	161,
	163,
	165,
	167,
	168,
	170,
	172,
	174,
	175,
	177,
	179,
	181,
	183,
	184,
	186,
	188,
	190,
	191,
	193,
	195,
	197,
	198,
	200,
	202,
	204,
	206,
	207,
	209,
	211,
	213,
	214,
	216,
	218,
	220,
	222,
	223,
	225,
};

__constant static int kCrToGreenTable[256] = {
	5990656,
	5943854,
	5897052,
	5850250,
	5803448,
	5756646,
	5709844,
	5663042,
	5616240,
	5569438,
	5522636,
	5475834,
	5429032,
	5382230,
	5335428,
	5288626,
	5241824,
	5195022,
	5148220,
	5101418,
	5054616,
	5007814,
	4961012,
	4914210,
	4867408,
	4820606,
	4773804,
	4727002,
	4680200,
	4633398,
	4586596,
	4539794,
	4492992,
	4446190,
	4399388,
	4352586,
	4305784,
	4258982,
	4212180,
	4165378,
	4118576,
	4071774,
	4024972,
	3978170,
	3931368,
	3884566,
	3837764,
	3790962,
	3744160,
	3697358,
	3650556,
	3603754,
	3556952,
	3510150,
	3463348,
	3416546,
	3369744,
	3322942,
	3276140,
	3229338,
	3182536,
	3135734,
	3088932,
	3042130,
	2995328,
	2948526,
	2901724,
	2854922,
	2808120,
	2761318,
	2714516,
	2667714,
	2620912,
	2574110,
	2527308,
	2480506,
	2433704,
	2386902,
	2340100,
	2293298,
	2246496,
	2199694,
	2152892,
	2106090,
	2059288,
	2012486,
	1965684,
	1918882,
	1872080,
	1825278,
	1778476,
	1731674,
	1684872,
	1638070,
	1591268,
	1544466,
	1497664,
	1450862,
	1404060,
	1357258,
	1310456,
	1263654,
	1216852,
	1170050,
	1123248,
	1076446,
	1029644,
	982842,
	936040,
	889238,
	842436,
	795634,
	748832,
	702030,
	655228,
	608426,
	561624,
	514822,
	468020,
	421218,
	374416,
	327614,
	280812,
	234010,
	187208,
	140406,
	93604,
	46802,
	0,
	-46802,
	-93604,
	-140406,
	-187208,
	-234010,
	-280812,
	-327614,
	-374416,
	-421218,
	-468020,
	-514822,
	-561624,
	-608426,
	-655228,
	-702030,
	-748832,
	-795634,
	-842436,
	-889238,
	-936040,
	-982842,
	-1029644,
	-1076446,
	-1123248,
	-1170050,
	-1216852,
	-1263654,
	-1310456,
	-1357258,
	-1404060,
	-1450862,
	-1497664,
	-1544466,
	-1591268,
	-1638070,
	-1684872,
	-1731674,
	-1778476,
	-1825278,
	-1872080,
	-1918882,
	-1965684,
	-2012486,
	-2059288,
	-2106090,
	-2152892,
	-2199694,
	-2246496,
	-2293298,
	-2340100,
	-2386902,
	-2433704,
	-2480506,
	-2527308,
	-2574110,
	-2620912,
	-2667714,
	-2714516,
	-2761318,
	-2808120,
	-2854922,
	-2901724,
	-2948526,
	-2995328,
	-3042130,
	-3088932,
	-3135734,
	-3182536,
	-3229338,
	-3276140,
	-3322942,
	-3369744,
	-3416546,
	-3463348,
	-3510150,
	-3556952,
	-3603754,
	-3650556,
	-3697358,
	-3744160,
	-3790962,
	-3837764,
	-3884566,
	-3931368,
	-3978170,
	-4024972,
	-4071774,
	-4118576,
	-4165378,
	-4212180,
	-4258982,
	-4305784,
	-4352586,
	-4399388,
	-4446190,
	-4492992,
	-4539794,
	-4586596,
	-4633398,
	-4680200,
	-4727002,
	-4773804,
	-4820606,
	-4867408,
	-4914210,
	-4961012,
	-5007814,
	-5054616,
	-5101418,
	-5148220,
	-5195022,
	-5241824,
	-5288626,
	-5335428,
	-5382230,
	-5429032,
	-5475834,
	-5522636,
	-5569438,
	-5616240,
	-5663042,
	-5709844,
	-5756646,
	-5803448,
	-5850250,
	-5897052,
	-5943854,
};

__constant static int kCbToGreenTable[256] = {
	2919680,
	2897126,
	2874572,
	2852018,
	2829464,
	2806910,
	2784356,
	2761802,
	2739248,
	2716694,
	2694140,
	2671586,
	2649032,
	2626478,
	2603924,
	2581370,
	2558816,
	2536262,
	2513708,
	2491154,
	2468600,
	2446046,
	2423492,
	2400938,
	2378384,
	2355830,
	2333276,
	2310722,
	2288168,
	2265614,
	2243060,
	2220506,
	2197952,
	2175398,
	2152844,
	2130290,
	2107736,
	2085182,
	2062628,
	2040074,
	2017520,
	1994966,
	1972412,
	1949858,
	1927304,
	1904750,
	1882196,
	1859642,
	1837088,
	1814534,
	1791980,
	1769426,
	1746872,
	1724318,
	1701764,
	1679210,
	1656656,
	1634102,
	1611548,
	1588994,
	1566440,
	1543886,
	1521332,
	1498778,
	1476224,
	1453670,
	1431116,
	1408562,
	1386008,
	1363454,
	1340900,
	1318346,
	1295792,
	1273238,
	1250684,
	1228130,
	1205576,
	1183022,
	1160468,
	1137914,
	1115360,
	1092806,
	1070252,
	1047698,
	1025144,
	1002590,
	980036,
	957482,
	934928,
	912374,
	889820,
	867266,
	844712,
	822158,
	799604,
	777050,
	754496,
	731942,
	709388,
	686834,
	664280,
	641726,
	619172,
	596618,
	574064,
	551510,
	528956,
	506402,
	483848,
	461294,
	438740,
	416186,
	393632,
	371078,
	348524,
	325970,
	303416,
	280862,
	258308,
	235754,
	213200,
	190646,
	168092,
	145538,
	122984,
	100430,
	77876,
	55322,
	32768,
	10214,
	-12340,
	-34894,
	-57448,
	-80002,
	-102556,
	-125110,
	-147664,
	-170218,
	-192772,
	-215326,
	-237880,
	-260434,
	-282988,
	-305542,
	-328096,
	-350650,
	-373204,
	-395758,
	-418312,
	-440866,
	-463420,
	-485974,
	-508528,
	-531082,
	-553636,
	-576190,
	-598744,
	-621298,
	-643852,
	-666406,
	-688960,
	-711514,
	-734068,
	-756622,
	-779176,
	-801730,
	-824284,
	-846838,
	-869392,
	-891946,
	-914500,
	-937054,
	-959608,
	-982162,
	-1004716,
	-1027270,
	-1049824,
	-1072378,
	-1094932,
	-1117486,
	-1140040,
	-1162594,
	-1185148,
	-1207702,
	-1230256,
	-1252810,
	-1275364,
	-1297918,
	-1320472,
	-1343026,
	-1365580,
	-1388134,
	-1410688,
	-1433242,
	-1455796,
	-1478350,
	-1500904,
	-1523458,
	-1546012,
	-1568566,
	-1591120,
	-1613674,
	-1636228,
	-1658782,
	-1681336,
	-1703890,
	-1726444,
	-1748998,
	-1771552,
	-1794106,
	-1816660,
	-1839214,
	-1861768,
	-1884322,
	-1906876,
	-1929430,
	-1951984,
	-1974538,
	-1997092,
	-2019646,
	-2042200,
	-2064754,
	-2087308,
	-2109862,
	-2132416,
	-2154970,
	-2177524,
	-2200078,
	-2222632,
	-2245186,
	-2267740,
	-2290294,
	-2312848,
	-2335402,
	-2357956,
	-2380510,
	-2403064,
	-2425618,
	-2448172,
	-2470726,
	-2493280,
	-2515834,
	-2538388,
	-2560942,
	-2583496,
	-2606050,
	-2628604,
	-2651158,
	-2673712,
	-2696266,
	-2718820,
	-2741374,
	-2763928,
	-2786482,
	-2809036,
	-2831590,
};

__constant static uchar kRangeLimitLut[4 * 256] = {
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	1,
	2,
	3,
	4,
	5,
	6,
	7,
	8,
	9,
	10,
	11,
	12,
	13,
	14,
	15,
	16,
	17,
	18,
	19,
	20,
	21,
	22,
	23,
	24,
	25,
	26,
	27,
	28,
	29,
	30,
	31,
	32,
	33,
	34,
	35,
	36,
	37,
	38,
	39,
	40,
	41,
	42,
	43,
	44,
	45,
	46,
	47,
	48,
	49,
	50,
	51,
	52,
	53,
	54,
	55,
	56,
	57,
	58,
	59,
	60,
	61,
	62,
	63,
	64,
	65,
	66,
	67,
	68,
	69,
	70,
	71,
	72,
	73,
	74,
	75,
	76,
	77,
	78,
	79,
	80,
	81,
	82,
	83,
	84,
	85,
	86,
	87,
	88,
	89,
	90,
	91,
	92,
	93,
	94,
	95,
	96,
	97,
	98,
	99,
	100,
	101,
	102,
	103,
	104,
	105,
	106,
	107,
	108,
	109,
	110,
	111,
	112,
	113,
	114,
	115,
	116,
	117,
	118,
	119,
	120,
	121,
	122,
	123,
	124,
	125,
	126,
	127,
	128,
	129,
	130,
	131,
	132,
	133,
	134,
	135,
	136,
	137,
	138,
	139,
	140,
	141,
	142,
	143,
	144,
	145,
	146,
	147,
	148,
	149,
	150,
	151,
	152,
	153,
	154,
	155,
	156,
	157,
	158,
	159,
	160,
	161,
	162,
	163,
	164,
	165,
	166,
	167,
	168,
	169,
	170,
	171,
	172,
	173,
	174,
	175,
	176,
	177,
	178,
	179,
	180,
	181,
	182,
	183,
	184,
	185,
	186,
	187,
	188,
	189,
	190,
	191,
	192,
	193,
	194,
	195,
	196,
	197,
	198,
	199,
	200,
	201,
	202,
	203,
	204,
	205,
	206,
	207,
	208,
	209,
	210,
	211,
	212,
	213,
	214,
	215,
	216,
	217,
	218,
	219,
	220,
	221,
	222,
	223,
	224,
	225,
	226,
	227,
	228,
	229,
	230,
	231,
	232,
	233,
	234,
	235,
	236,
	237,
	238,
	239,
	240,
	241,
	242,
	243,
	244,
	245,
	246,
	247,
	248,
	249,
	250,
	251,
	252,
	253,
	254,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
	255,
};

__device__ void ColorTransformYCbCrToRGB(__global uchar pixel[3])
{
	__constant_ex uchar *kRangeLimit = kRangeLimitLut + 384;

	int y = pixel[0];
	int cb = pixel[1];
	int cr = pixel[2];
	pixel[0] = kRangeLimit[y + kCrToRedTable[cr]];
	pixel[1] = kRangeLimit[y +
						   ((kCrToGreenTable[cr] + kCbToGreenTable[cb]) >> 16)];
	pixel[2] = kRangeLimit[y + kCbToBlueTable[cb]];
}

__device__ void YUVToRGB(__private uchar pixelBlock[3 * 8 * 8], const int size /*= 8 * 8*/)
{
	__constant_ex uchar *kRangeLimit = kRangeLimitLut + 384;

	for (int i = 0; i < size; i++)
	{
		__private uchar *pixel = &pixelBlock[i * 3];

		int y = pixel[0];
		int cb = pixel[1];
		int cr = pixel[2];
		pixel[0] = kRangeLimit[y + kCrToRedTable[cr]];
		pixel[1] = kRangeLimit[y + ((kCrToGreenTable[cr] + kCbToGreenTable[cb]) >> 16)];
		pixel[2] = kRangeLimit[y + kCbToBlueTable[cb]];
	}
}

__constant static float kSrgb8ToLinearTable[256] = {
	0.000000f,
	0.077399f,
	0.154799f,
	0.232198f,
	0.309598f,
	0.386997f,
	0.464396f,
	0.541796f,
	0.619195f,
	0.696594f,
	0.773994f,
	0.853367f,
	0.937509f,
	1.026303f,
	1.119818f,
	1.218123f,
	1.321287f,
	1.429375f,
	1.542452f,
	1.660583f,
	1.783830f,
	1.912253f,
	2.045914f,
	2.184872f,
	2.329185f,
	2.478910f,
	2.634105f,
	2.794824f,
	2.961123f,
	3.133055f,
	3.310673f,
	3.494031f,
	3.683180f,
	3.878171f,
	4.079055f,
	4.285881f,
	4.498698f,
	4.717556f,
	4.942502f,
	5.173584f,
	5.410848f,
	5.654341f,
	5.904108f,
	6.160196f,
	6.422649f,
	6.691512f,
	6.966827f,
	7.248640f,
	7.536993f,
	7.831928f,
	8.133488f,
	8.441715f,
	8.756651f,
	9.078335f,
	9.406810f,
	9.742115f,
	10.084290f,
	10.433375f,
	10.789410f,
	11.152432f,
	11.522482f,
	11.899597f,
	12.283815f,
	12.675174f,
	13.073712f,
	13.479465f,
	13.892470f,
	14.312765f,
	14.740385f,
	15.175366f,
	15.617744f,
	16.067555f,
	16.524833f,
	16.989614f,
	17.461933f,
	17.941824f,
	18.429322f,
	18.924460f,
	19.427272f,
	19.937793f,
	20.456054f,
	20.982090f,
	21.515934f,
	22.057618f,
	22.607175f,
	23.164636f,
	23.730036f,
	24.303404f,
	24.884774f,
	25.474176f,
	26.071642f,
	26.677203f,
	27.290891f,
	27.912736f,
	28.542769f,
	29.181020f,
	29.827520f,
	30.482299f,
	31.145387f,
	31.816813f,
	32.496609f,
	33.184802f,
	33.881422f,
	34.586499f,
	35.300062f,
	36.022139f,
	36.752760f,
	37.491953f,
	38.239746f,
	38.996169f,
	39.761248f,
	40.535013f,
	41.317491f,
	42.108710f,
	42.908697f,
	43.717481f,
	44.535088f,
	45.361546f,
	46.196882f,
	47.041124f,
	47.894297f,
	48.756429f,
	49.627547f,
	50.507676f,
	51.396845f,
	52.295078f,
	53.202402f,
	54.118843f,
	55.044428f,
	55.979181f,
	56.923129f,
	57.876298f,
	58.838712f,
	59.810398f,
	60.791381f,
	61.781686f,
	62.781338f,
	63.790363f,
	64.808784f,
	65.836627f,
	66.873918f,
	67.920679f,
	68.976937f,
	70.042715f,
	71.118037f,
	72.202929f,
	73.297414f,
	74.401516f,
	75.515259f,
	76.638668f,
	77.771765f,
	78.914575f,
	80.067122f,
	81.229428f,
	82.401518f,
	83.583415f,
	84.775142f,
	85.976722f,
	87.188178f,
	88.409534f,
	89.640813f,
	90.882037f,
	92.133229f,
	93.394412f,
	94.665609f,
	95.946841f,
	97.238133f,
	98.539506f,
	99.850982f,
	101.172584f,
	102.504334f,
	103.846254f,
	105.198366f,
	106.560693f,
	107.933256f,
	109.316077f,
	110.709177f,
	112.112579f,
	113.526305f,
	114.950375f,
	116.384811f,
	117.829635f,
	119.284868f,
	120.750532f,
	122.226647f,
	123.713235f,
	125.210317f,
	126.717914f,
	128.236047f,
	129.764737f,
	131.304005f,
	132.853871f,
	134.414357f,
	135.985483f,
	137.567270f,
	139.159738f,
	140.762907f,
	142.376799f,
	144.001434f,
	145.636832f,
	147.283012f,
	148.939997f,
	150.607804f,
	152.286456f,
	153.975971f,
	155.676371f,
	157.387673f,
	159.109900f,
	160.843070f,
	162.587203f,
	164.342319f,
	166.108438f,
	167.885578f,
	169.673761f,
	171.473005f,
	173.283330f,
	175.104755f,
	176.937299f,
	178.780982f,
	180.635824f,
	182.501843f,
	184.379058f,
	186.267489f,
	188.167154f,
	190.078073f,
	192.000265f,
	193.933749f,
	195.878543f,
	197.834666f,
	199.802137f,
	201.780975f,
	203.771198f,
	205.772826f,
	207.785876f,
	209.810367f,
	211.846319f,
	213.893748f,
	215.952674f,
	218.023115f,
	220.105089f,
	222.198615f,
	224.303711f,
	226.420395f,
	228.548685f,
	230.688599f,
	232.840156f,
	235.003373f,
	237.178269f,
	239.364861f,
	241.563167f,
	243.773205f,
	245.994993f,
	248.228549f,
	250.473890f,
	252.731035f,
	255.000000f,
};

__device__ void YUVToImage(
	__private uchar yuv[3 * 8 * 8],
	__private float *r,
	__private float *g,
	__private float *b,
	const int xsize /* = 8*/,
	const int ysize /* = 8*/,
	const int inside_x /* = 8*/,
	const int inside_y /* = 8*/)
{
	YUVToRGB(yuv, xsize * ysize);

#define lut kSrgb8ToLinearTable
	//    const __constant float* lut = kSrgb8ToLinearTable;

	for (int i = 0; i < xsize * ysize; i++)
	{
		r[i] = lut[yuv[3 * i]];
		g[i] = lut[yuv[3 * i + 1]];
		b[i] = lut[yuv[3 * i + 2]];
	}
	for (int y = 0; y < inside_y; y++)
	{
		for (int x = inside_x; x < xsize; x++)
		{
			int idx = y * xsize + (inside_x - 1);
			r[y * xsize + x] = r[idx];
			g[y * xsize + x] = g[idx];
			b[y * xsize + x] = b[idx];
		}
	}
	for (int y = inside_y; y < ysize; y++)
	{
		for (int x = 0; x < xsize; x++)
		{
			int idx = (inside_y - 1) * xsize + x;
			r[y * xsize + x] = r[idx];
			g[y * xsize + x] = g[idx];
			b[y * xsize + x] = b[idx];
		}
	}
#undef lut
}

__device__ void BlockToImage(__private const coeff_t block[8 * 8 * 3], float r[8 * 8], float g[8 * 8], float b[8 * 8], int inside_x, int inside_y)
{
	uchar idct[3][8 * 8];
	CoeffToIDCT(&block[0], idct[0]);
	CoeffToIDCT(&block[8 * 8], idct[1]);
	CoeffToIDCT(&block[8 * 8 * 2], idct[2]);

	ushort pixels[3][8 * 8];
	IDCTToPixel8x8(idct[0], pixels[0]);
	IDCTToPixel8x8(idct[1], pixels[1]);
	IDCTToPixel8x8(idct[2], pixels[2]);

	uchar yuv[8 * 8 * 3];
	PixelToYUV(pixels[0], &yuv[0], 8, 8);
	PixelToYUV(pixels[1], &yuv[1], 8, 8);
	PixelToYUV(pixels[2], &yuv[2], 8, 8);

	YUVToRGB(yuv, 8 * 8);

	for (int i = 0; i < 8 * 8; i++)
	{
		r[i] = kSrgb8ToLinearTable[yuv[3 * i]];
		g[i] = kSrgb8ToLinearTable[yuv[3 * i + 1]];
		b[i] = kSrgb8ToLinearTable[yuv[3 * i + 2]];
	}
	for (int y = 0; y < inside_y; y++)
	{
		for (int x = inside_x; x < 8; x++)
		{
			int idx = y * 8 + (inside_x - 1);
			r[y * 8 + x] = r[idx];
			g[y * 8 + x] = g[idx];
			b[y * 8 + x] = b[idx];
		}
	}
	for (int y = inside_y; y < 8; y++)
	{
		for (int x = 0; x < 8; x++)
		{
			int idx = (inside_y - 1) * 8 + x;
			r[y * 8 + x] = r[idx];
			g[y * 8 + x] = g[idx];
			b[y * 8 + x] = b[idx];
		}
	}
}

__device__ void CoeffToYUV16x16(__private const coeff_t block[8 * 8], __private uchar *yuv, __global const ushort *pixel_orig, int block_x, int block_y, int width_, int height_)
{
	uchar idct[8 * 8];
	CoeffToIDCT(&block[0], &idct[0]);

	ushort pixels[16 * 16];
	IDCTToPixel16x16(idct, pixels, pixel_orig, block_x, block_y, width_, height_);

	PixelToYUV(pixels, yuv, 16, 16);
}

__device__ void CoeffToYUV8x8(__private const coeff_t block[8 * 8], __private uchar *yuv)
{
	uchar idct[8 * 8];
	CoeffToIDCT(&block[0], &idct[0]);

	ushort pixels[8 * 8];
	IDCTToPixel8x8(idct, pixels);

	PixelToYUV(pixels, yuv, 8, 8);
}

__device__ void CoeffToYUV16x16_g(__global const coeff_t block[8 * 8], __private uchar *yuv, __global const ushort *pixel_orig, int block_x, int block_y, int width_, int height_)
{
	coeff_t b[8 * 8];
	for (int i = 0; i < 8 * 8; i++)
	{
		b[i] = block[i];
	}
	CoeffToYUV16x16(b, yuv, pixel_orig, block_x, block_y, width_, height_);
}

__device__ void CoeffToYUV8x8_g(__global const coeff_t block[8 * 8], __private uchar *yuv)
{
	coeff_t b[8 * 8];
	for (int i = 0; i < 8 * 8; i++)
	{
		b[i] = block[i];
	}

	CoeffToYUV8x8(b, yuv);
}

__device__ void Copy8x8To16x16(const uchar yuv8x8[3 * 8 * 8], __private uchar yuv16x16[3 * 16 * 16], int off_x, int off_y)
{
	for (int y = 0; y < 8; y++)
	{
		for (int x = 0; x < 8; x++)
		{
			int idx = y * 8 + x;
			int idx16 = (y + off_y * 8) * 16 + (x + off_x * 8);
			yuv16x16[idx16 * 3] = yuv8x8[idx * 3];
		}
	}
}

__device__ void Copy16x16To8x8(const uchar yuv16x16[3 * 16 * 16], __private uchar yuv8x8[3 * 8 * 8], int off_x, int off_y)
{
	for (int y = 0; y < 8; y++)
	{
		for (int x = 0; x < 8; x++)
		{
			int idx = y * 8 + x;
			int idx16 = (y + off_y * 8) * 16 + (x + off_x * 8);
			yuv8x8[idx * 3] = yuv16x16[idx16 * 3];
		}
	}
}

__device__ void Copy16x16ToChannel(const float rgb16x16[3][16 * 16], float r[8 * 8], float g[8 * 8], float b[8 * 8], int off_x, int off_y)
{
	for (int y = 0; y < 8; y++)
	{
		for (int x = 0; x < 8; x++)
		{
			int idx = y * 8 + x;
			int idx16 = (y + off_y * 8) * 16 + (x + off_x * 8);
			r[idx] = rgb16x16[0][idx16];
			g[idx] = rgb16x16[1][idx16];
			b[idx] = rgb16x16[2][idx16];
		}
	}
}

__device__ void Convolution(
	const unsigned int xsize, const unsigned int ysize,
	const int xstep, const int len, const int offset,
	__private const float *multipliers,
	__private const float *inp,
	const float border_ratio,
	__private float *result)
{
	float weight_no_border = 0;

	for (unsigned int j = 0; j <= 2 * offset; ++j)
	{
		weight_no_border += multipliers[j];
	}
	for (unsigned int x = 0, ox = 0; x < xsize; x += xstep, ox++)
	{
		int minx = x < offset ? 0 : x - offset;
		int maxx = _min_i((int)xsize, (int)(x + len - offset)) - 1;
		float weight = 0.0f;
		for (int j = minx; j <= maxx; ++j)
		{
			weight += multipliers[j - x + offset];
		}
		// Interpolate linearly between the no-border scaling and border scaling.
		weight = (1.0f - border_ratio) * weight + border_ratio * weight_no_border;
		float scale = 1.0f / weight;
		for (unsigned int y = 0; y < ysize; ++y)
		{
			float sum = 0.0f;
			for (int j = minx; j <= maxx; ++j)
			{
				sum += inp[y * xsize + j] * multipliers[j - x + offset];
			}
			result[ox * ysize + y] = (float)(sum * scale);
		}
	}
}

__device__ void BlurEx(__private const float *r, const int xsize, const int ysize, const float kSigma, const float border_ratio, __private float *output)
{
	// const float sigma = 1.1;
	// float m = 2.25;  // Accuracy increases when m is increased.
	const float scaler = -0.41322314049586772f; // when sigma=1.1, scaler is -0.41322314049586772
	const int diff = 2;							// when sigma=1.1, diff's value is 2.
	const int expn_size = 5;					// when sigma=1.1, scaler is  5
	__private float expn[5] = {
		exp(scaler * (-diff) * (-diff)),
		exp(scaler * (-diff + 1) * (-diff + 1)),
		exp(scaler * (-diff + 2) * (-diff + 2)),
		exp(scaler * (-diff + 3) * (-diff + 3)),
		exp(scaler * (-diff + 4) * (-diff + 4))};
	const int xstep = 1; // when sigma=1.1, xstep is 1.
	const int ystep = xstep;

	int dxsize = (xsize + xstep - 1) / xstep;

	__private float tmp[8 * 8] = {0};
	Convolution(xsize, ysize, xstep, expn_size, diff, expn, r, border_ratio, tmp);
	Convolution(ysize, dxsize, ystep, expn_size, diff, expn, tmp,
				border_ratio, output);
}

__device__ void OpsinDynamicsImageBlock(__private float *r, __private float *g, __private float *b,
										__private const float *r_blurred, __private const float *g_blurred, __private const float *b_blurred,
										const int size)
{
	for (unsigned int i = 0; i < size; ++i)
	{
		float sensitivity[3];
		{
			// Calculate sensitivity[3] based on the smoothed image gamma derivative.
			__private float pre_rgb[3] = {r_blurred[i], g_blurred[i], b_blurred[i]};
			__private float pre_mixed[3];
			OpsinAbsorbance(pre_rgb, pre_mixed);
			sensitivity[0] = Gamma(pre_mixed[0]) / pre_mixed[0];
			sensitivity[1] = Gamma(pre_mixed[1]) / pre_mixed[1];
			sensitivity[2] = Gamma(pre_mixed[2]) / pre_mixed[2];
		}
		__private float cur_rgb[3] = {r[i], g[i], b[i]};
		__private float cur_mixed[3];
		OpsinAbsorbance(cur_rgb, cur_mixed);
		cur_mixed[0] *= sensitivity[0];
		cur_mixed[1] *= sensitivity[1];
		cur_mixed[2] *= sensitivity[2];
		float x, y, z;
		RgbToXyb(cur_mixed[0], cur_mixed[1], cur_mixed[2], &x, &y, &z);
		r[i] = (float)(x);
		g[i] = (float)(y);
		b[i] = (float)(z);
	}
}

__device__ void MaskHighIntensityChangeBlock(__private float *xyb0_x, __private float *xyb0_y, __private float *xyb0_b,
											 __private float *xyb1_x, __private float *xyb1_y, __private float *xyb1_b,
											 const __private float *c0_x, const __private float *c0_y, const __private float *c0_b,
											 const __private float *c1_x, const __private float *c1_y, const __private float *c1_b,
											 const int xsize, const int ysize)
{
	for (int x = 0; x < xsize; ++x)
	{
		for (int y = 0; y < ysize; ++y)
		{
			unsigned int ix = y * xsize + x;
			const float ave[3] = {
				(c0_x[ix] + c1_x[ix]) * 0.5f,
				(c0_y[ix] + c1_y[ix]) * 0.5f,
				(c0_b[ix] + c1_b[ix]) * 0.5f,
			};
			float sqr_max_diff = -1;
			{
				int offset[4] = {-1, 1, -(int)(xsize), (int)(xsize)};
				int border[4] = {x == 0, x + 1 == xsize, y == 0, y + 1 == ysize};
				for (int dir = 0; dir < 4; ++dir)
				{
					if (border[dir])
					{
						continue;
					}
					const int ix2 = ix + offset[dir];
					float diff = 0.5f * (c0_y[ix2] + c1_y[ix2]) - ave[1];
					diff *= diff;
					if (sqr_max_diff < diff)
					{
						sqr_max_diff = diff;
					}
				}
			}
			const float kReductionX = 275.19165240059317f;
			const float kReductionY = 18599.41286306991f;
			const float kReductionZ = 410.8995306951065f;
			const float kChromaBalance = 106.95800948271017f;
			float chroma_scale = kChromaBalance / (ave[1] + kChromaBalance);

			const float mix[3] = {
				chroma_scale * kReductionX / (sqr_max_diff + kReductionX),
				kReductionY / (sqr_max_diff + kReductionY),
				chroma_scale * kReductionZ / (sqr_max_diff + kReductionZ),
			};
			// Interpolate lineraly between the average color and the actual
			// color -- to reduce the importance of this pixel.
			xyb0_x[ix] = (float)(mix[0] * c0_x[ix] + (1 - mix[0]) * ave[0]);
			xyb1_x[ix] = (float)(mix[0] * c1_x[ix] + (1 - mix[0]) * ave[0]);

			xyb0_y[ix] = (float)(mix[1] * c0_y[ix] + (1 - mix[1]) * ave[1]);
			xyb1_y[ix] = (float)(mix[1] * c1_y[ix] + (1 - mix[1]) * ave[1]);

			xyb0_b[ix] = (float)(mix[2] * c0_b[ix] + (1 - mix[2]) * ave[2]);
			xyb1_b[ix] = (float)(mix[2] * c1_b[ix] + (1 - mix[2]) * ave[2]);
		}
	}
}

__device__ void floatcopy(__private float *dst, __private const float *src, const int size)
{
	for (int i = 0; i < size; i++)
	{
		dst[i] = src[i];
	}
}

__device__ void coeffcopy_g(__private coeff_t *dst, __global const coeff_t *src, int size)
{
	for (int i = 0; i < size; i++)
	{
		dst[i] = src[i];
	}
}

__device__ void coeffcopy(__private coeff_t *dst, const coeff_t *src, int size)
{
	for (int i = 0; i < size; i++)
	{
		dst[i] = src[i];
	}
}

__device__ void CalcOpsinDynamicsImage(__private float rgb[3][kDCTBlockSize])
{
	__private float rgb_blurred[3][kDCTBlockSize];
	for (int i = 0; i < 3; ++i)
	{
		BlurEx(rgb[i], 8, 8, 1.1f, 0, rgb_blurred[i]);
	}
	OpsinDynamicsImageBlock(rgb[0], rgb[1], rgb[2], rgb_blurred[0], rgb_blurred[1], rgb_blurred[2], kDCTBlockSize);
}

__device__ float ComputeImage8x8Block(__private float rgb0_c[3][kDCTBlockSize], __private float rgb1_c[3][kDCTBlockSize], const __global float *mask_scale_block)
{
	CalcOpsinDynamicsImage(rgb1_c);

	__private float rgb0[3][kDCTBlockSize];
	__private float rgb1[3][kDCTBlockSize];

	floatcopy(&rgb0[0][0], &rgb0_c[0][0], 3 * kDCTBlockSize);
	floatcopy(&rgb1[0][0], &rgb1_c[0][0], 3 * kDCTBlockSize);

	MaskHighIntensityChangeBlock(rgb0[0], rgb0[1], rgb0[2],
								 rgb1[0], rgb1[1], rgb1[2],
								 rgb0_c[0], rgb0_c[1], rgb0_c[2],
								 rgb1_c[0], rgb1_c[1], rgb1_c[2],
								 8, 8);

	float b0[3 * kDCTBlockSize];
	float b1[3 * kDCTBlockSize];
	for (int c = 0; c < 3; ++c)
	{
		for (int ix = 0; ix < kDCTBlockSize; ++ix)
		{
			b0[c * kDCTBlockSize + ix] = rgb0[c][ix];
			b1[c * kDCTBlockSize + ix] = rgb1[c][ix];
		}
	}

	__private float diff_xyz_dc[3] = {0.0};
	__private float diff_xyz_ac[3] = {0.0};
	__private float diff_xyz_edge_dc[3] = {0.0};
	ButteraugliBlockDiff(b0, b1, diff_xyz_dc, diff_xyz_ac, diff_xyz_edge_dc);

	float diff = 0.0f;
	float diff_edge = 0.0f;

	for (int c = 0; c < 3; ++c)
	{
		diff += diff_xyz_dc[c] * mask_scale_block[c];
		diff += diff_xyz_ac[c] * mask_scale_block[c];
		diff_edge += diff_xyz_edge_dc[c] * mask_scale_block[c];
	}
	const float kEdgeWeight = 0.05f;
	return sqrt((1 - kEdgeWeight) * diff + kEdgeWeight * diff_edge);
}

// return the count of Non-zero item
__device__ int MakeInputOrderEx(const coeff_t block[3 * 8 * 8], const coeff_t orig_block[3 * 8 * 8], __private IntFloatPairList *input_order)
{
	const int block_size = 64;
	int size = 0;
	for (int c = 0; c < 3; ++c)
	{
		for (int k = 1; k < block_size; ++k)
		{
			int idx = c * block_size + k;
			if (block[idx] != 0)
			{
				float score = _abs_f(orig_block[idx]) * csf[idx] + bias[idx];
				size = list_push_back(input_order, idx, score);
			}
		}
	}

	return SortInputOrder(input_order->pData, size);
}

__device__ int GetOrigBlock(float rgb0_c[3][kDCTBlockSize],
							const __global float *orig_image_batch,
							int width_, int height_,
							int block_x, int block_y,
							int factor,
							int off_x, int off_y)
{
	int block_xx = block_x * factor + off_x;
	int block_yy = block_y * factor + off_y;
	if (block_xx * 8 >= width_ || block_yy * 8 >= height_)
		return -1;

	const int block8_width = (width_ + 8 - 1) / 8;

	int block_ix = block_yy * block8_width + block_xx;

	__global const float *block_opsin = &orig_image_batch[block_ix * 3 * kDCTBlockSize];
	for (int i = 0; i < 3; ++i)
	{
		for (int k = 0; k < kDCTBlockSize; k++)
		{
			rgb0_c[i][k] = block_opsin[i * kDCTBlockSize + k];
		}
	}

	return block_ix;
}

__device__ float CompareBlockFactor1(const channel_info mayout_channel[3],
									  __private const coeff_t *candidate_block,
									  const int block_x,
									  const int block_y,
									  __global const float *orig_image_batch,
									  __global const float *mask_scale,
									  const int image_width,
									  const int image_height)
{
	__private const coeff_t *candidate_channel[3];
	for (int c = 0; c < 3; c++)
	{
		candidate_channel[c] = &candidate_block[c * 8 * 8];
	}

	uchar yuv16x16[3 * 16 * 16] = {0}; // factor 2 mode output image
	uchar yuv8x8[3 * 8 * 8] = {0};	   // factor 1 mode output image

	for (int c = 0; c < 3; c++)
	{
		if (mayout_channel[c].factor == 1)
		{
			__private const coeff_t *coeff_block = candidate_channel[c];
			CoeffToYUV8x8(coeff_block, &yuv8x8[c]);
		}
		else
		{
			int block_xx = block_x / mayout_channel[c].factor;
			int block_yy = block_y / mayout_channel[c].factor;
			int ix = block_x % mayout_channel[c].factor;
			;
			int iy = block_y % mayout_channel[c].factor;

			int block_16x16idx = block_yy * mayout_channel[c].block_width + block_xx;
			__global const coeff_t *coeff_block = mayout_channel[c].coeff + block_16x16idx * 8 * 8;

			CoeffToYUV16x16_g(coeff_block, &yuv16x16[c],
							  mayout_channel[c].pixel, block_xx, block_yy,
							  image_width,
							  image_height);

			// copy YUV16x16 corner to YUV8x8
			Copy16x16To8x8(&yuv16x16[c], &yuv8x8[c], ix, iy);
		}
	}

	{
		float rgb0_c[3][kDCTBlockSize];
		int block_8x8idx = GetOrigBlock(rgb0_c, orig_image_batch, image_width, image_height, block_x, block_y, 1, 0, 0);

		int inside_x = block_x * 8 + 8 > image_width ? image_width - block_x * 8 : 8;
		int inside_y = block_y * 8 + 8 > image_height ? image_height - block_y * 8 : 8;
		float rgb1_c[3][kDCTBlockSize];

		YUVToImage(yuv8x8, rgb1_c[0], rgb1_c[1], rgb1_c[2], 8, 8, inside_x, inside_y);

		return ComputeImage8x8Block(rgb0_c, rgb1_c, mask_scale + block_8x8idx * 3);
	}
}

__device__ float Factor2(const channel_info mayout_channel[3],
						  __private const coeff_t *candidate_block,
						  const int block_x,
						  const int block_y,
						  __global const float *orig_image_batch,
						  __global const float *mask_scale,
						  const int image_width,
						  const int image_height)
{
	const int factor = 2;
	__private const coeff_t *candidate_channel[3];
	for (int c = 0; c < 3; c++)
	{
		candidate_channel[c] = &candidate_block[c * 8 * 8];
	}

	uchar yuv16x16[3 * 16 * 16] = {0}; // factor 2 mode output image
	uchar yuv8x8[3 * 8 * 8] = {0};	   // factor 1 mode output image

	for (int c = 0; c < 3; c++)
	{
		if (mayout_channel[c].factor == 1)
		{
			for (int iy = 0; iy < factor; ++iy)
			{
				for (int ix = 0; ix < factor; ++ix)
				{
					int block_xx = block_x * factor + ix;
					int block_yy = block_y * factor + iy;

					/// if (ix != off_x || iy != off_y) continue;
					if (block_xx >= mayout_channel[c].block_width ||
						block_yy >= mayout_channel[c].block_height)
					{
						continue;
					}
					int block_8x8idx = block_yy * mayout_channel[c].block_width + block_xx;
					__global const coeff_t *coeff_block = mayout_channel[c].coeff + block_8x8idx * 8 * 8;
					CoeffToYUV8x8_g(coeff_block, &yuv8x8[c]);

					// copy YUV8x8 to YUV1616 corner
					Copy8x8To16x16(&yuv8x8[c], &yuv16x16[c], ix, iy);
				}
			}
		}
		else
		{
			__private const coeff_t *coeff_block = candidate_channel[c];
			CoeffToYUV16x16(coeff_block, &yuv16x16[c],
							mayout_channel[c].pixel, block_x, block_y,
							image_width,
							image_height);
		}
	}

	int inside_x = block_x * 16 + 16 > image_width ? image_width - block_x * 16 : 16;
	int inside_y = block_y * 16 + 16 > image_height ? image_height - block_y * 16 : 16;

	float rgb16x16[3][16 * 16];
	YUVToImage(yuv16x16, rgb16x16[0], rgb16x16[1], rgb16x16[2], 16, 16, inside_x, inside_y);

	float max_err = 0;
	for (int iy = 0; iy < factor; ++iy)
	{
		for (int ix = 0; ix < factor; ++ix)
		{
			int block_xx = block_x * factor + ix;
			int block_yy = block_y * factor + iy;

			if (block_xx * 8 >= image_width ||
				block_yy * 8 >= image_height)
			{
				continue;
			}

			float rgb0_c[3][kDCTBlockSize];
			int block_8x8idx = GetOrigBlock(rgb0_c, orig_image_batch, image_width, image_height, block_x, block_y, factor, ix, iy);

			float rgb1_c[3][kDCTBlockSize];
			Copy16x16ToChannel(rgb16x16, rgb1_c[0], rgb1_c[1], rgb1_c[2], ix, iy);
			float err = ComputeImage8x8Block(rgb0_c, rgb1_c, mask_scale + block_8x8idx * 3);
			max_err = _max_f(max_err, err);
		}
	}
	return max_err;
}

__device__ float CompareBlockFactor(const channel_info mayout_channel[3],
									 __private const coeff_t *candidate_block,
									 const int block_x,
									 const int block_y,
									 __global const float *orig_image_batch,
									 __global const float *mask_scale,
									 const int image_width,
									 const int image_height,
									 const int factor)
{
	__private const coeff_t *candidate_channel[3];
	for (int c = 0; c < 3; c++)
	{
		candidate_channel[c] = &candidate_block[c * 8 * 8];
	}

	uchar yuv16x16[3 * 16 * 16] = {0}; // factor 2 mode output image
	uchar yuv8x8[3 * 8 * 8] = {0};	   // factor 1 mode output image

	for (int c = 0; c < 3; c++)
	{
		if (mayout_channel[c].factor == 1)
		{
			if (factor == 1)
			{
				__private const coeff_t *coeff_block = candidate_channel[c];
				CoeffToYUV8x8(coeff_block, &yuv8x8[c]);
			}
			else
			{
				for (int iy = 0; iy < factor; ++iy)
				{
					for (int ix = 0; ix < factor; ++ix)
					{
						int block_xx = block_x * factor + ix;
						int block_yy = block_y * factor + iy;

						/// if (ix != off_x || iy != off_y) continue;
						if (block_xx >= mayout_channel[c].block_width ||
							block_yy >= mayout_channel[c].block_height)
						{
							continue;
						}
						int block_8x8idx = block_yy * mayout_channel[c].block_width + block_xx;
						__global const coeff_t *coeff_block = mayout_channel[c].coeff + block_8x8idx * 8 * 8;
						CoeffToYUV8x8_g(coeff_block, &yuv8x8[c]);

						// copy YUV8x8 to YUV1616 corner
						Copy8x8To16x16(&yuv8x8[c], &yuv16x16[c], ix, iy);
					}
				}
			}
		}
		else
		{
			if (factor == 1)
			{
				int block_xx = block_x / mayout_channel[c].factor;
				int block_yy = block_y / mayout_channel[c].factor;
				int ix = block_x % mayout_channel[c].factor;
				;
				int iy = block_y % mayout_channel[c].factor;

				int block_16x16idx = block_yy * mayout_channel[c].block_width + block_xx;
				__global const coeff_t *coeff_block = mayout_channel[c].coeff + block_16x16idx * 8 * 8;

				CoeffToYUV16x16_g(coeff_block, &yuv16x16[c],
								  mayout_channel[c].pixel, block_xx, block_yy,
								  image_width,
								  image_height);

				// copy YUV16x16 corner to YUV8x8
				Copy16x16To8x8(&yuv16x16[c], &yuv8x8[c], ix, iy);
			}
			else
			{
				__private const coeff_t *coeff_block = candidate_channel[c];
				CoeffToYUV16x16(coeff_block, &yuv16x16[c],
								mayout_channel[c].pixel, block_x, block_y,
								image_width,
								image_height);
			}
		}
	}

	if (factor == 1)
	{
		float rgb0_c[3][kDCTBlockSize];
		int block_8x8idx = GetOrigBlock(rgb0_c, orig_image_batch, image_width, image_height, block_x, block_y, factor, 0, 0);

		int inside_x = block_x * 8 + 8 > image_width ? image_width - block_x * 8 : 8;
		int inside_y = block_y * 8 + 8 > image_height ? image_height - block_y * 8 : 8;
		float rgb1_c[3][kDCTBlockSize];

		YUVToImage(yuv8x8, rgb1_c[0], rgb1_c[1], rgb1_c[2], 8, 8, inside_x, inside_y);

		return ComputeImage8x8Block(rgb0_c, rgb1_c, mask_scale + block_8x8idx * 3);
	}
	else
	{
		int inside_x = block_x * 16 + 16 > image_width ? image_width - block_x * 16 : 16;
		int inside_y = block_y * 16 + 16 > image_height ? image_height - block_y * 16 : 16;

		float rgb16x16[3][16 * 16];
		YUVToImage(yuv16x16, rgb16x16[0], rgb16x16[1], rgb16x16[2], 16, 16, inside_x, inside_y);

		float max_err = 0;
		for (int iy = 0; iy < factor; ++iy)
		{
			for (int ix = 0; ix < factor; ++ix)
			{
				int block_xx = block_x * factor + ix;
				int block_yy = block_y * factor + iy;

				if (block_xx * 8 >= image_width ||
					block_yy * 8 >= image_height)
				{
					continue;
				}

				float rgb0_c[3][kDCTBlockSize];
				int block_8x8idx = GetOrigBlock(rgb0_c, orig_image_batch, image_width, image_height, block_x, block_y, factor, ix, iy);

				float rgb1_c[3][kDCTBlockSize];
				Copy16x16ToChannel(rgb16x16, rgb1_c[0], rgb1_c[1], rgb1_c[2], ix, iy);
				float err = ComputeImage8x8Block(rgb0_c, rgb1_c, mask_scale + block_8x8idx * 3);
				max_err = _max_f(max_err, err);
			}
		}
		return max_err;
	}
}

__device__ coeff_t Quantize(coeff_t raw_coeff, int quant)
{
	const int r = raw_coeff % quant;
	const coeff_t delta =
		2 * r > quant ? quant - r : (-2) * r > quant ? -quant - r
													 : -r;
	return raw_coeff + delta;
}

__device__ bool QuantizeBlock(coeff_t block[kDCTBlockSize], __global const int q[kDCTBlockSize])
{
	bool changed = false;
	for (int k = 0; k < kDCTBlockSize; ++k)
	{
		coeff_t coeff = Quantize(block[k], q[k]);
		changed = changed || (coeff != block[k]);
		block[k] = coeff;
	}
	return changed;
}

#endif //__USE_OPENCL__

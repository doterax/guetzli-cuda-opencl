/*
 * OpenCL edition implementation of guetzli.
 *
 * Author: strongtu@tencent.com
 *         ianhuang@tencent.com
 *         chriskzhou@tencent.com
 *		  stephendeng@tencent.com
 */
#include "clguetzli.h"
#include <math.h>
#include <algorithm>
#include <vector>
#include <set>
#include "cl.hpp"
#define LOG
#define UTILITIES_FILE
#include "third_party/OpenCL-Wrapper/opencl.hpp"

static size_t next_perf_id = 0;
static size_t next_alloc_id = 0;
extern bool verbose_cl = true;

class Perf
{
private:
	size_t _id;
	Clock _clock;
	const char *_fn;

public:
	Perf(const char *fn) : _clock(), _fn(fn), _id(next_perf_id++)
	{
		if (verbose_cl)
			LogInfo("+ %s(%3d)\n", _fn, _id);
	}
	~Perf()
	{
		if (verbose_cl)
			LogInfo("- %s(%3d): %s ms\n", _fn, _id, to_string(_clock.stop() * 1000, 3).c_str());
	}
};

string mem_to_string(size_t mem) {
	if (mem < 1024)
		return alignr(8, mem);
	if (mem < 1024 * 1024)
		return alignr(4, mem/1024) + " KiB";
	if (mem < 1024 * 1024 * 1024)
		return alignr(4, mem / 1024 / 1024) + " MiB";
	if (mem < 1024l * 1024l * 1024l * 1024l)
		return alignr(4, mem / 1024 / 1024 / 1024) + " GiB";

	return alignr(10, mem);
}

class MemoryInfo {
	size_t _memInUse;
	size_t _maxMemInUse;
	size_t _objectsInUse;
	size_t _maxObjectsInUse;
	std::set<size_t> _granularity;
	std::set<size_t> _lastGranules;
public:
	MemoryInfo() : _memInUse(0), _maxMemInUse(0), _objectsInUse(0), _maxObjectsInUse(0) {}
	void Alloc(size_t size, const string& reason, size_t allocId) {
		_memInUse += size;
		++_objectsInUse;
		_granularity.insert(size);
		_lastGranules.insert(size);
		if (_memInUse > _maxMemInUse) _maxMemInUse = _memInUse;
		if (_objectsInUse > _maxObjectsInUse) _maxObjectsInUse = _objectsInUse;
		uint align = 8;
		if (verbose_cl) {
			string info = "alloc(#"+ alignr(5, allocId) + ") " + mem_to_string(size) + " used: " + mem_to_string(_memInUse) + ": " + reason + "\n";
			printf(info.c_str());
		}
	}
	void Free(size_t size, const string& reason, size_t allocId) {
		_memInUse -= size;
		--_objectsInUse;
		uint align = 8;
		if (verbose_cl) {
			string info = "free (#" + alignr(5, allocId) + ") " + mem_to_string(size) + " used: " + mem_to_string(_memInUse) + ": " + reason + "\n";
			printf(info.c_str());
		}

		if (_memInUse == 0) {
			string info = "Peak memory usage: " + mem_to_string(_maxMemInUse) + " Peak objects: " + alignl(5, _maxObjectsInUse) + "\n";
			printf(info.c_str());
			
			for (auto i = _granularity.begin(); i != _granularity.end(); ++i) {
				bool inLast = _lastGranules.find(*i) == _lastGranules.end();
				string marker = inLast ? " *" : "";
				string alloc = "- mem: " + mem_to_string(*i) + marker + "\n";
				printf(alloc.c_str());
			}
			_lastGranules.clear();
		}
	}
};

static MemoryInfo memoryInfo;

class TrackMemory {
	MemoryInfo& _info;
	string _reason;
	size_t _size;
	size_t _allocId;
public:
	TrackMemory(size_t size, const string& reason): _info(memoryInfo), _size(size), _reason(reason), _allocId(next_alloc_id++) {
		_info.Alloc(_size, _reason, _allocId);
	}
	~TrackMemory() {
		_info.Free(_size, _reason, _allocId);
	}
};

extern MATH_MODE g_mathMode = MODE_AUTO;

// Helper function to calculate optimal workgroup size for AMD GPUs
void calculateOptimalWorkgroupSize(size_t globalSize[2], size_t localSize[2], bool isAMD)
{
	if (isAMD)
	{
		// For AMD GPUs, prefer multiples of 64 (wavefront size)
		// But ensure the global size is divisible by local size
		localSize[0] = 64;
		localSize[1] = 1;

		// Adjust local size if global size is not divisible
		if (globalSize[0] % localSize[0] != 0)
		{
			// Find the largest divisor of globalSize[0] that is <= 64
			for (int i = 64; i >= 1; i--)
			{
				if (globalSize[0] % i == 0)
				{
					localSize[0] = i;
					break;
				}
			}
		}

		// Handle edge case where globalSize[0] is very small
		if (globalSize[0] < localSize[0])
		{
			localSize[0] = globalSize[0];
		}

		// Ensure we don't exceed maximum workgroup size
		if (localSize[0] > 256)
			localSize[0] = 256;
		if (localSize[1] > 256)
			localSize[1] = 256;

		// If we can't find a good divisor, let OpenCL choose
		if (localSize[0] == 0)
		{
			localSize[0] = 0;
			localSize[1] = 0;
		}
	}
	else
	{
		// For non-AMD GPUs, use NULL (let OpenCL choose)
		localSize[0] = 0;
		localSize[1] = 0;
	}
}
#ifdef __USE_OPENCL__

#ifdef __USE_DOUBLE_AS_FLOAT__
#define double float
#endif

void clOpsinDynamicsImage(float *r, float *g, float *b, const size_t xsize, const size_t ysize)
{
	size_t elements = xsize * ysize;
	size_t channel_size = elements * sizeof(float);

	Perf clk("clOpsinDynamicsImage");
	ocl_args_d_t &ocl = getOcl();
	TrackMemory rgb_m(channel_size * 3, "clOpsinDynamicsImage:rgb");
	ocl_channels rgb = ocl.allocMemChannels(channel_size, r, g, b);

	clOpsinDynamicsImageEx(rgb, xsize, ysize);

	clEnqueueReadBuffer(ocl.commandQueue, rgb.r, false, 0, channel_size, r, 0, NULL, NULL);
	clEnqueueReadBuffer(ocl.commandQueue, rgb.g, false, 0, channel_size, g, 0, NULL, NULL);
	clEnqueueReadBuffer(ocl.commandQueue, rgb.b, false, 0, channel_size, b, 0, NULL, NULL);
	clFinish(ocl.commandQueue);

	ocl.releaseMemChannels(rgb);
}

void clDiffmapOpsinDynamicsImage(
	float *result,
	const float *r, const float *g, const float *b,
	const float *r2, const float *g2, const float *b2,
	const size_t xsize, const size_t ysize,
	const size_t step)
{
	size_t channel_size = xsize * ysize * sizeof(float);

	Perf clk("clDiffmapOpsinDynamicsImage");
	ocl_args_d_t &ocl = getOcl();
	TrackMemory  xyb0_m(channel_size * 3, "clDiffmapOpsinDynamicsImage:xyb0");
	ocl_channels xyb0 = ocl.allocMemChannels(channel_size, r, g, b);

	TrackMemory  xyb1_m(channel_size * 3, "clDiffmapOpsinDynamicsImage:xyb1");
	ocl_channels xyb1 = ocl.allocMemChannels(channel_size, r2, g2, b2);

	TrackMemory  mem_result_m(channel_size, "clDiffmapOpsinDynamicsImage:mem_result");
	cl_mem mem_result = ocl.allocMem(channel_size, result);

	clDiffmapOpsinDynamicsImageEx(mem_result, xyb0, xyb1, xsize, ysize, step);

	clEnqueueReadBuffer(ocl.commandQueue, mem_result, false, 0, channel_size, result, 0, NULL, NULL);
	cl_int err = clFinish(ocl.commandQueue);

	ocl.releaseMemChannels(xyb1);
	ocl.releaseMemChannels(xyb0);

	clReleaseMemObject(mem_result);
}

void clComputeBlockZeroingOrder(
	guetzli::CoeffData *output_order_batch,
	const channel_info orig_channel[3],
	const float *orig_image_batch,
	const float *mask_scale,
	const int image_width,
	const int image_height,
	const channel_info mayout_channel[3],
	const int factor,
	const int comp_mask,
	const float BlockErrorLimit)
{
	// Validate factor to prevent division by zero
	if (factor <= 0)
	{
		LogInfo("Error: factor is %d, must be > 0. Skipping clComputeBlockZeroingOrder.\n", factor);
		return;
	}

	const int block8_width = (image_width + 8 - 1) / 8;
	const int block8_height = (image_height + 8 - 1) / 8;
	const int blockf_width = (image_width + 8 * factor - 1) / (8 * factor);
	const int blockf_height = (image_height + 8 * factor - 1) / (8 * factor);

	using namespace guetzli;

	Perf clk("clComputeBlockZeroingOrder");
	ocl_args_d_t &ocl = getOcl();

	cl_mem mem_orig_coeff[3];
	cl_mem mem_mayout_coeff[3];
	cl_mem mem_mayout_pixel[3];
	size_t total_orig_coeff_size = 0;
	size_t total_mayout_coeff_size = 0;
	size_t total_mayout_pixel_size = 0;
	for (int c = 0; c < 3; c++)
	{
		int block_count = orig_channel[c].block_width * orig_channel[c].block_height;
		size_t orig_coeff_size = block_count * sizeof(::coeff_t) * kDCTBlockSize;
		total_orig_coeff_size += orig_coeff_size;

		block_count = mayout_channel[c].block_width * mayout_channel[c].block_height;
		size_t mayout_coeff_size = block_count * sizeof(::coeff_t) * kDCTBlockSize;
		total_mayout_coeff_size += mayout_coeff_size;

		size_t mayout_pixel_size = image_width * image_height * sizeof(uint16_t);
		total_mayout_pixel_size += mayout_pixel_size;
	}
	TrackMemory mem_orig_coeff_m(total_orig_coeff_size, "clComputeBlockZeroingOrder:mem_orig_coeff");
	TrackMemory mem_mayout_coeff_m(total_mayout_coeff_size, "clComputeBlockZeroingOrder:mem_mayout_coeff");
	TrackMemory mem_mayout_pixel_m(total_mayout_pixel_size, "clComputeBlockZeroingOrder:mem_mayout_pixel");
	
	size_t orig_image_size = sizeof(float) * 3 * kDCTBlockSize * block8_width * block8_height;
	TrackMemory mem_orig_image_m(orig_image_size, "clComputeBlockZeroingOrder:mem_orig_image");
	cl_mem mem_orig_image = ocl.allocMem(orig_image_size, orig_image_batch);
	
	size_t mask_scale_size = sizeof(float) * 3 * block8_width * block8_height;
	TrackMemory mem_mask_scale_m(mask_scale_size, "clComputeBlockZeroingOrder:mem_mask_scale");
	cl_mem mem_mask_scale = ocl.allocMem(mask_scale_size, mask_scale);

	int output_order_batch_size = sizeof(CoeffData) * 3 * kDCTBlockSize * blockf_width * blockf_height;
	TrackMemory mem_output_order_batch_m(output_order_batch_size, "clComputeBlockZeroingOrder:mem_output_order_batch");
	cl_mem mem_output_order_batch = ocl.allocMem(output_order_batch_size, output_order_batch);

	for (int c = 0; c < 3; c++)
	{
		int block_count = orig_channel[c].block_width * orig_channel[c].block_height;
		size_t orig_coeff_size = block_count * sizeof(::coeff_t) * kDCTBlockSize;
		mem_orig_coeff[c] = ocl.allocMem(orig_coeff_size, orig_channel[c].coeff);

		block_count = mayout_channel[c].block_width * mayout_channel[c].block_height;
		size_t mayout_coeff_size = block_count * sizeof(::coeff_t) * kDCTBlockSize;
		mem_mayout_coeff[c] = ocl.allocMem(mayout_coeff_size, mayout_channel[c].coeff);

		size_t mayout_pixel_size = image_width * image_height * sizeof(uint16_t);
		mem_mayout_pixel[c] = ocl.allocMem(mayout_pixel_size, mayout_channel[c].pixel);
	}

	cl_kernel kernel = ocl.kernel[KERNEL_COMPUTEBLOCKZEROINGORDER];
	clSetKernelArgEx(kernel, &mem_orig_coeff[0], &mem_orig_coeff[1], &mem_orig_coeff[2],
					 &mem_orig_image, &mem_mask_scale,
					 &blockf_width, &blockf_height,
					 &image_width, &image_height,
					 &mem_mayout_coeff[0], &mem_mayout_coeff[1], &mem_mayout_coeff[2],
					 &mem_mayout_pixel[0], &mem_mayout_pixel[1], &mem_mayout_pixel[2],
					 &mayout_channel[0], &mayout_channel[1], &mayout_channel[2],
					 &factor,
					 &comp_mask,
					 &BlockErrorLimit,
					 &mem_output_order_batch);

	size_t globalWorkSize[2] = {blockf_width, blockf_height};
	size_t localWorkSize[2];
	calculateOptimalWorkgroupSize(globalWorkSize, localWorkSize, ocl.isAmd);

	cl_int err;
	if (localWorkSize[0] > 0 && localWorkSize[1] > 0)
	{
		if (verbose_cl)
			LogInfo("Launching kernel with workgroup size: [%zu, %zu], global size: [%zu, %zu]\n",
					localWorkSize[0], localWorkSize[1], globalWorkSize[0], globalWorkSize[1]);
		err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	}
	else
	{
		if (verbose_cl)
			LogInfo("Launching kernel with auto workgroup size, global size: [%zu, %zu]\n",
					globalWorkSize[0], globalWorkSize[1]);
		err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	}
	LOG_CL_RESULT(err);
	err = clFinish(ocl.commandQueue);
	LOG_CL_RESULT(err);

	clEnqueueReadBuffer(ocl.commandQueue, mem_output_order_batch, false, 0, output_order_batch_size, output_order_batch, 0, NULL, NULL);
	clFinish(ocl.commandQueue);

	for (int c = 0; c < 3; c++)
	{
		clReleaseMemObject(mem_orig_coeff[c]);
		clReleaseMemObject(mem_mayout_coeff[c]);
		clReleaseMemObject(mem_mayout_pixel[c]);
	}

	clReleaseMemObject(mem_orig_image);
	clReleaseMemObject(mem_mask_scale);
	clReleaseMemObject(mem_output_order_batch);
}

void clMask(
	float *mask_r, float *mask_g, float *mask_b,
	float *maskdc_r, float *maskdc_g, float *maskdc_b,
	const size_t xsize, const size_t ysize,
	const float *r, const float *g, const float *b,
	const float *r2, const float *g2, const float *b2)
{
	Perf clk("clMask");
	ocl_args_d_t &ocl = getOcl();

	size_t channel_size = xsize * ysize * sizeof(float);

	TrackMemory rgb_m(channel_size * 3, "clMask:rgb");
	ocl_channels rgb = ocl.allocMemChannels(channel_size, r, g, b);
	TrackMemory rgb2_m(channel_size * 3, "clMask:rgb2");
	ocl_channels rgb2 = ocl.allocMemChannels(channel_size, r2, g2, b2);
	TrackMemory mask_m(channel_size * 3, "clMask:mask");
	ocl_channels mask = ocl.allocMemChannels(channel_size);
	TrackMemory mask_dc_m(channel_size * 3, "clMask:mask_dc");
	ocl_channels mask_dc = ocl.allocMemChannels(channel_size);

	clMaskEx(mask, mask_dc, rgb, rgb2, xsize, ysize);

	clEnqueueReadBuffer(ocl.commandQueue, mask.r, false, 0, channel_size, mask_r, 0, NULL, NULL);
	clEnqueueReadBuffer(ocl.commandQueue, mask.g, false, 0, channel_size, mask_g, 0, NULL, NULL);
	clEnqueueReadBuffer(ocl.commandQueue, mask.b, false, 0, channel_size, mask_b, 0, NULL, NULL);
	clEnqueueReadBuffer(ocl.commandQueue, mask_dc.r, false, 0, channel_size, maskdc_r, 0, NULL, NULL);
	clEnqueueReadBuffer(ocl.commandQueue, mask_dc.g, false, 0, channel_size, maskdc_g, 0, NULL, NULL);
	clEnqueueReadBuffer(ocl.commandQueue, mask_dc.b, false, 0, channel_size, maskdc_b, 0, NULL, NULL);
	clFinish(ocl.commandQueue);

	ocl.releaseMemChannels(rgb);
	ocl.releaseMemChannels(rgb2);
	ocl.releaseMemChannels(mask);
	ocl.releaseMemChannels(mask_dc);
}

void clDiffmapOpsinDynamicsImageEx(
	cl_mem result,
	ocl_channels xyb0,
	ocl_channels xyb1,
	const size_t xsize, const size_t ysize,
	const size_t step)
{
	const size_t res_xsize = (xsize + step - 1) / step;
	const size_t res_ysize = (ysize + step - 1) / step;

	size_t channel_size = xsize * ysize * sizeof(float);
	size_t channel_step_size = res_xsize * res_ysize * sizeof(float);

	Perf clk("clDiffmapOpsinDynamicsImageEx");
	ocl_args_d_t &ocl = getOcl();

	size_t edge_detector_map_size = 3 * channel_step_size;
	TrackMemory edge_detector_map_m(edge_detector_map_size, "clDiffmapOpsinDynamicsImageEx:edge_detector_map");
	cl_mem edge_detector_map = ocl.allocMem(edge_detector_map_size);
	
	size_t block_diff_dc_size = 3 * channel_step_size;
	TrackMemory block_diff_dc_m(block_diff_dc_size, "clDiffmapOpsinDynamicsImageEx:block_diff_dc");
	cl_mem block_diff_dc = ocl.allocMem(block_diff_dc_size);
	
	size_t block_diff_ac_size = 3 * channel_step_size;
	TrackMemory block_diff_ac_m(block_diff_ac_size, "clDiffmapOpsinDynamicsImageEx:block_diff_ac");
	cl_mem block_diff_ac = ocl.allocMem(block_diff_ac_size);

	clMaskHighIntensityChangeEx(xyb0, xyb1, xsize, ysize);

	clEdgeDetectorMapEx(edge_detector_map, xyb0, xyb1, xsize, ysize, step);
	clBlockDiffMapEx(block_diff_dc, block_diff_ac, xyb0, xyb1, xsize, ysize, step);
	clEdgeDetectorLowFreqEx(block_diff_ac, xyb0, xyb1, xsize, ysize, step);
	{
		TrackMemory mask_m(channel_size * 3, "clDiffmapOpsinDynamicsImageEx:mask");
		ocl_channels mask = ocl.allocMemChannels(channel_size);
		TrackMemory mask_dc_m(channel_size * 3, "clDiffmapOpsinDynamicsImageEx:mask_dc");
		ocl_channels mask_dc = ocl.allocMemChannels(channel_size);
		clMaskEx(mask, mask_dc, xyb0, xyb1, xsize, ysize);
		clCombineChannelsEx(result, mask, mask_dc, xsize, ysize, block_diff_dc, block_diff_ac, edge_detector_map, res_xsize, step);

		ocl.releaseMemChannels(mask);
		ocl.releaseMemChannels(mask_dc);
	}

	clCalculateDiffmapEx(result, xsize, ysize, step);

	clReleaseMemObject(edge_detector_map);
	clReleaseMemObject(block_diff_dc);
	clReleaseMemObject(block_diff_ac);
}
void clConvolutionEx(
	cl_mem result /*out*/,
	const cl_mem inp, unsigned int xsize, unsigned int ysize,
	const cl_mem multipliers, unsigned int len,
	int xstep, int offset, float border_ratio)
{
	Perf clk("clConvolutionEx");
	ocl_args_d_t &ocl = getOcl();

	size_t oxsize = (xsize + xstep - 1) / xstep;

	cl_kernel kernel = ocl.kernel[KERNEL_CONVOLUTION];
	clSetKernelArgEx(kernel, &result, &inp, &xsize, &multipliers, &len, &xstep, &offset, &border_ratio);

	size_t globalWorkSize[2] = {oxsize, ysize};
	size_t localWorkSize[2];
	calculateOptimalWorkgroupSize(globalWorkSize, localWorkSize, ocl.isAmd);

	cl_int err;
	if (localWorkSize[0] > 0 && localWorkSize[1] > 0)
	{
		if (verbose_cl)
			LogInfo("Launching clConvolutionEx with workgroup size: [%zu, %zu], global size: [%zu, %zu]\n",
					localWorkSize[0], localWorkSize[1], globalWorkSize[0], globalWorkSize[1]);
		err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	}
	else
	{
		if (verbose_cl)
			LogInfo("Launching clConvolutionEx with auto workgroup size, global size: [%zu, %zu]\n",
					globalWorkSize[0], globalWorkSize[1]);
		err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	}
	LOG_CL_RESULT(err);
	err = clFinish(ocl.commandQueue);
	LOG_CL_RESULT(err);
}

void clConvolutionXEx(
	cl_mem result /*out*/,
	const cl_mem inp, size_t xsize, size_t ysize,
	const cl_mem multipliers, size_t len,
	int xstep, int offset, float border_ratio)
{
	Perf clk("clConvolutionXEx");
	ocl_args_d_t &ocl = getOcl();

	cl_kernel kernel = ocl.kernel[KERNEL_CONVOLUTIONX];
	clSetKernelArgEx(kernel, &result, &xsize, &ysize, &inp, &multipliers, &len, &xstep, &offset, &border_ratio);

	size_t x_count = (xsize + xstep - 1) / xstep;
	size_t globalWorkSize[2] = {x_count, ysize};
	size_t localWorkSize[2];
	calculateOptimalWorkgroupSize(globalWorkSize, localWorkSize, ocl.isAmd);

	cl_int err;
	if (localWorkSize[0] > 0 && localWorkSize[1] > 0)
	{
		if (verbose_cl)
			LogInfo("Launching clConvolutionXEx with workgroup size: [%zu, %zu], global size: [%zu, %zu]\n",
					localWorkSize[0], localWorkSize[1], globalWorkSize[0], globalWorkSize[1]);
		err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
		LOG_CL_RESULT(err);
	}
	else
	{
		if (verbose_cl)
			LogInfo("Launching clConvolutionXEx with auto workgroup size, global size: [%zu, %zu]\n",
					globalWorkSize[0], globalWorkSize[1]);
		err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
		LOG_CL_RESULT(err);
	}

	err = clFinish(ocl.commandQueue);
	LOG_CL_RESULT(err);
}

void clConvolutionYEx(
	cl_mem result /*out*/,
	const cl_mem inp, size_t xsize, size_t ysize,
	const cl_mem multipliers, size_t len,
	int xstep, int offset, float border_ratio)
{
	Perf clk("clConvolutionYEx");
	ocl_args_d_t &ocl = getOcl();

	cl_kernel kernel = ocl.kernel[KERNEL_CONVOLUTIONY];
	clSetKernelArgEx(kernel, &result, &xsize, &ysize, &inp, &multipliers, &len, &xstep, &offset, &border_ratio);

	size_t x_count = (xsize + xstep - 1) / xstep;
	size_t y_count = (ysize + xstep - 1) / xstep;
	size_t globalWorkSize[2] = {x_count, y_count};
	cl_int err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	LOG_CL_RESULT(err);
	err = clFinish(ocl.commandQueue);
	LOG_CL_RESULT(err);
}

void clSquareSampleEx(
	cl_mem result /*out*/,
	const cl_mem image, size_t xsize, size_t ysize,
	size_t xstep, size_t ystep)
{
	Perf clk("clSquareSampleEx");
	ocl_args_d_t &ocl = getOcl();

	cl_kernel kernel = ocl.kernel[KERNEL_SQUARESAMPLE];
	clSetKernelArgEx(kernel, &result, &xsize, &ysize, &image, &xstep, &ystep);

	size_t globalWorkSize[2] = {xsize, ysize};
	cl_int err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	LOG_CL_RESULT(err);
	err = clFinish(ocl.commandQueue);
	LOG_CL_RESULT(err);
}

void clBlurEx(cl_mem image /*out, opt*/, const size_t xsize, const size_t ysize,
			  const double sigma, const double border_ratio,
			  cl_mem result /*out, opt*/)
{
	Perf clk("clBlurEx");
	double m = 2.25; // Accuracy increases when m is increased.
	const double scaler = -1.0 / (2 * sigma * sigma);
	// For m = 9.0: exp(-scaler * diff * diff) < 2^ {-52}
	const int diff = std::max<int>(1, m * fabs(sigma));
	const int expn_size = 2 * diff + 1;
	std::vector<float> expn(expn_size);
	for (int i = -diff; i <= diff; ++i)
	{
		expn[i + diff] = static_cast<float>(exp(scaler * i * i));
	}

	result = result ? result : image;

	const int xstep = std::max<int>(1, int(sigma / 3));

	ocl_args_d_t &ocl = getOcl();

	TrackMemory temp_m(expn_size * sizeof(float), "clBlurEx:mem_expn");
	Memory<cl_float> mem_expn_o(*ocl.device, expn_size, 1, expn.data());
	const cl_mem mem_expn = mem_expn_o.get_cl_buffer().get();

	if (xstep > 1)
	{
		Memory<cl_float> m(*ocl.device, xsize * ysize);
		TrackMemory temp_m(xsize * ysize * 3, "clBlurEx:temp1");
		const cl_mem temp = m.get_cl_buffer().get();
		clConvolutionXEx(temp, image, xsize, ysize, mem_expn, expn_size, xstep, diff, border_ratio);
		clConvolutionYEx(result, temp, xsize, ysize, mem_expn, expn_size, xstep, diff, border_ratio);
		clSquareSampleEx(result, result, xsize, ysize, xstep, xstep);
	}
	else
	{
		Memory<cl_float> m(*ocl.device, xsize * ysize);
		TrackMemory temp_m(xsize * ysize * 3, "clBlurEx:temp2");
		const cl_mem temp = m.get_cl_buffer().get();
		clConvolutionXEx(temp, image, xsize, ysize, mem_expn, expn_size, xstep, diff, border_ratio);
		clConvolutionYEx(result, temp, xsize, ysize, mem_expn, expn_size, xstep, diff, border_ratio);
	}
}

void clOpsinDynamicsImageEx(ocl_channels &rgb, const size_t xsize, const size_t ysize)
{
	static const double kSigma = 1.1;
	const size_t size = xsize * ysize;
	size_t channel_size = size * sizeof(float);

	Perf clk("clOpsinDynamicsImageEx");
	ocl_args_d_t &ocl = getOcl();

	size_t blurred_size = size * sizeof(float);
	Memory<float> r_blurred(*ocl.device, size);
	TrackMemory r_blurred_m(blurred_size, "clOpsinDynamicsImageEx:r_blurred");
	Memory<float> g_blurred(*ocl.device, size);
	TrackMemory g_blurred_m(blurred_size, "clOpsinDynamicsImageEx:g_blurred");
	Memory<float> b_blurred(*ocl.device, size);
	TrackMemory b_blurred_m(blurred_size, "clOpsinDynamicsImageEx:b_blurred");

	if (verbose_cl)
		LogInfo("clOpsinDynamicsImageEx: blur.r\n");
	clBlurEx(rgb.r, xsize, ysize, kSigma, 0.0, r_blurred.get_cl_buffer().get());
	if (verbose_cl)
		LogInfo("clOpsinDynamicsImageEx: blur.g\n");
	clBlurEx(rgb.g, xsize, ysize, kSigma, 0.0, g_blurred.get_cl_buffer().get());
	if (verbose_cl)
		LogInfo("clOpsinDynamicsImageEx: blur.b\n");
	clBlurEx(rgb.b, xsize, ysize, kSigma, 0.0, b_blurred.get_cl_buffer().get());

	cl_kernel kernel = ocl.kernel[KERNEL_OPSINDYNAMICSIMAGE];
	if (verbose_cl)
		LogInfo("clOpsinDynamicsImageEx: clSetKernelArgEx\n");
	clSetKernelArgEx(kernel, &rgb.r, &rgb.g, &rgb.b, &size, &r_blurred.get_cl_buffer(), &g_blurred.get_cl_buffer(), &b_blurred.get_cl_buffer());

	// https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clGetKernelSuggestedLocalWorkSizeKHR.html
	size_t globalWorkSize[1] = {size};
	if (verbose_cl)
		LogInfo("clOpsinDynamicsImageEx: clEnqueueNDRangeKernel+\n");
	cl_int err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	if (verbose_cl)
		LogInfo("clOpsinDynamicsImageEx: clEnqueueNDRangeKernel-\n");
	LOG_CL_RESULT(err);
	if (verbose_cl)
		LogInfo("clOpsinDynamicsImageEx: clFinish+\n");
	err = clFinish(ocl.commandQueue);
	if (verbose_cl)
		LogInfo("clOpsinDynamicsImageEx: clFinish-\n");
	LOG_CL_RESULT(err);

	if (verbose_cl)
		LogInfo("clOpsinDynamicsImageEx: done\n");
}

void clMaskHighIntensityChangeEx(
	ocl_channels &xyb0 /*in,out*/,
	ocl_channels &xyb1 /*in,out*/,
	const size_t xsize, const size_t ysize)
{
	size_t channel_size = xsize * ysize * sizeof(float);

	Perf clk("clMaskHighIntensityChangeEx");
	ocl_args_d_t &ocl = getOcl();

	TrackMemory c0_m(channel_size * 3, "clMaskHighIntensityChangeEx:c0");
	ocl_channels c0 = ocl.allocMemChannels(channel_size);
	TrackMemory c1_m(channel_size * 3, "clMaskHighIntensityChangeEx:c1");
	ocl_channels c1 = ocl.allocMemChannels(channel_size);

	clEnqueueCopyBuffer(ocl.commandQueue, xyb0.r, c0.r, 0, 0, channel_size, 0, NULL, NULL);
	clEnqueueCopyBuffer(ocl.commandQueue, xyb0.g, c0.g, 0, 0, channel_size, 0, NULL, NULL);
	clEnqueueCopyBuffer(ocl.commandQueue, xyb0.b, c0.b, 0, 0, channel_size, 0, NULL, NULL);
	clEnqueueCopyBuffer(ocl.commandQueue, xyb1.r, c1.r, 0, 0, channel_size, 0, NULL, NULL);
	clEnqueueCopyBuffer(ocl.commandQueue, xyb1.g, c1.g, 0, 0, channel_size, 0, NULL, NULL);
	clEnqueueCopyBuffer(ocl.commandQueue, xyb1.b, c1.b, 0, 0, channel_size, 0, NULL, NULL);
	clFinish(ocl.commandQueue);

	cl_kernel kernel = ocl.kernel[KERNEL_MASKHIGHINTENSITYCHANGE];
	clSetKernelArgEx(kernel,
					 &xyb0.r, &xyb0.g, &xyb0.b,
					 &xsize, &ysize,
					 &xyb1.r, &xyb1.g, &xyb1.b,
					 &c0.r, &c0.g, &c0.b,
					 &c1.r, &c1.g, &c1.b);

	size_t globalWorkSize[2] = {xsize, ysize};
	cl_int err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	LOG_CL_RESULT(err);
	err = clFinish(ocl.commandQueue);
	LOG_CL_RESULT(err);

	ocl.releaseMemChannels(c0);
	ocl.releaseMemChannels(c1);
}

void clEdgeDetectorMapEx(
	cl_mem result /*out*/,
	const ocl_channels &rgb, const ocl_channels &rgb2,
	const size_t xsize, const size_t ysize, const size_t step)
{
	size_t channel_size = xsize * ysize * sizeof(float);

	Perf clk("clEdgeDetectorMapEx");
	ocl_args_d_t &ocl = getOcl();

	TrackMemory rgb_blured_m(channel_size * 3, "clEdgeDetectorMapEx:rgb_blured");
	ocl_channels rgb_blured = ocl.allocMemChannels(channel_size);
	TrackMemory rgb2_blured_m(channel_size * 3, "clEdgeDetectorMapEx:rgb2_blured");
	ocl_channels rgb2_blured = ocl.allocMemChannels(channel_size);

	static const double kSigma[3] = {1.5, 0.586, 0.4};

	for (int i = 0; i < 3; i++)
	{
		clBlurEx(rgb.ch[i], xsize, ysize, kSigma[i], 0.0, rgb_blured.ch[i]);
		clBlurEx(rgb2.ch[i], xsize, ysize, kSigma[i], 0.0, rgb2_blured.ch[i]);
	}

	const size_t res_xsize = (xsize + step - 1) / step;
	const size_t res_ysize = (ysize + step - 1) / step;

	cl_kernel kernel = ocl.kernel[KERNEL_EDGEDETECTOR];
	clSetKernelArgEx(kernel, &result,
					 &res_xsize, &res_ysize,
					 &rgb_blured.r, &rgb_blured.g, &rgb_blured.b,
					 &rgb2_blured.r, &rgb2_blured.g, &rgb2_blured.b,
					 &xsize, &ysize, &step);

	size_t globalWorkSize[2] = {res_xsize, res_ysize};
	cl_int err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	LOG_CL_RESULT(err);
	err = clFinish(ocl.commandQueue);
	LOG_CL_RESULT(err);

	ocl.releaseMemChannels(rgb_blured);
	ocl.releaseMemChannels(rgb2_blured);
}

void clBlockDiffMapEx(
	cl_mem block_diff_dc /*out*/,
	cl_mem block_diff_ac /*out*/,
	const ocl_channels &rgb, const ocl_channels &rgb2,
	const size_t xsize, const size_t ysize, const size_t step)
{
	Perf clk("clBlockDiffMapEx");
	ocl_args_d_t &ocl = getOcl();

	const size_t res_xsize = (xsize + step - 1) / step;
	const size_t res_ysize = (ysize + step - 1) / step;

	cl_kernel kernel = ocl.kernel[KERNEL_BLOCKDIFFMAP];
	clSetKernelArgEx(kernel, &block_diff_dc, &block_diff_ac,
					 &res_xsize, &res_ysize,
					 &rgb.r, &rgb.g, &rgb.b,
					 &rgb2.r, &rgb2.g, &rgb2.b,
					 &xsize, &ysize, &step);

	size_t globalWorkSize[2] = {res_xsize, res_ysize};
	cl_int err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	LOG_CL_RESULT(err);
	err = clFinish(ocl.commandQueue);
	LOG_CL_RESULT(err);
}

void clEdgeDetectorLowFreqEx(
	cl_mem block_diff_ac /*in,out*/,
	const ocl_channels &rgb, const ocl_channels &rgb2,
	const size_t xsize, const size_t ysize, const size_t step)
{
	size_t channel_size = xsize * ysize * sizeof(float);

	static const double kSigma = 14;
	Perf clk("clEdgeDetectorLowFreqEx");
	ocl_args_d_t &ocl = getOcl();
	TrackMemory rgb_blured_m(channel_size * 3, "clEdgeDetectorLowFreqEx:rgb_blured");
	ocl_channels rgb_blured = ocl.allocMemChannels(channel_size);
	TrackMemory rgb2_blured_m(channel_size * 3, "clEdgeDetectorLowFreqEx:rgb2_blured");
	ocl_channels rgb2_blured = ocl.allocMemChannels(channel_size);

	for (int i = 0; i < 3; i++)
	{
		clBlurEx(rgb.ch[i], xsize, ysize, kSigma, 0.0, rgb_blured.ch[i]);
		clBlurEx(rgb2.ch[i], xsize, ysize, kSigma, 0.0, rgb2_blured.ch[i]);
	}

	const size_t res_xsize = (xsize + step - 1) / step;
	const size_t res_ysize = (ysize + step - 1) / step;

	cl_kernel kernel = ocl.kernel[KERNEL_EDGEDETECTORLOWFREQ];
	clSetKernelArgEx(kernel, &block_diff_ac,
					 &res_xsize, &res_ysize,
					 &rgb_blured.r, &rgb_blured.g, &rgb_blured.b,
					 &rgb2_blured.r, &rgb2_blured.g, &rgb2_blured.b,
					 &xsize, &ysize, &step);

	size_t globalWorkSize[2] = {res_xsize, res_ysize};
	cl_int err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	LOG_CL_RESULT(err);
	err = clFinish(ocl.commandQueue);
	LOG_CL_RESULT(err);

	ocl.releaseMemChannels(rgb_blured);
	ocl.releaseMemChannels(rgb2_blured);
}

void clDiffPrecomputeEx(
	ocl_channels &mask /*out*/,
	const ocl_channels &xyb0, const ocl_channels &xyb1,
	const size_t xsize, const size_t ysize)
{
	Perf clk("clDiffPrecomputeEx");
	ocl_args_d_t &ocl = getOcl();

	cl_kernel kernel = ocl.kernel[KERNEL_DIFFPRECOMPUTE];
	clSetKernelArgEx(kernel, &mask.x, &mask.y, &mask.b,
					 &xsize, &ysize,
					 &xyb0.x, &xyb0.y, &xyb0.b,
					 &xyb1.x, &xyb1.y, &xyb1.b);

	size_t globalWorkSize[2] = {xsize, ysize};
	cl_int err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	LOG_CL_RESULT(err);
	err = clFinish(ocl.commandQueue);
	LOG_CL_RESULT(err);
}

void clScaleImageEx(cl_mem img /*in, out*/, size_t size, double w)
{
	Perf clk("clScaleImageEx");
	ocl_args_d_t &ocl = getOcl();
	float fw = w;

	cl_kernel kernel = ocl.kernel[KERNEL_SCALEIMAGE];
	clSetKernelArgEx(kernel, &img, &size, &fw);

	size_t globalWorkSize[1] = {size};
	cl_int err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	LOG_CL_RESULT(err);
	err = clFinish(ocl.commandQueue);
	LOG_CL_RESULT(err);
}

void clAverage5x5Ex(cl_mem img /*in,out*/, const size_t xsize, const size_t ysize)
{
	if (xsize < 4 || ysize < 4)
	{
		// TODO: Make this work for small dimensions as well.
		return;
	}

	Perf clk("clAverage5x5Ex");
	ocl_args_d_t &ocl = getOcl();

	size_t len = xsize * ysize * sizeof(float);
	TrackMemory img_org_m(len, "clAverage5x5Ex:img_org");
	cl_mem img_org = ocl.allocMem(len);

	clEnqueueCopyBuffer(ocl.commandQueue, img, img_org, 0, 0, len, 0, NULL, NULL);

	cl_kernel kernel = ocl.kernel[KERNEL_AVERAGE5X5];
	clSetKernelArgEx(kernel, &img, &xsize, &ysize, &img_org);

	size_t globalWorkSize[2] = {xsize, ysize};
	cl_int err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	LOG_CL_RESULT(err);
	err = clFinish(ocl.commandQueue);
	LOG_CL_RESULT(err);

	clReleaseMemObject(img_org);
}

void clMinSquareValEx(
	cl_mem img /*in,out*/,
	const size_t xsize, const size_t ysize,
	const size_t square_size, const size_t offset)
{
	Perf clk("clMinSquareValEx"); // possible candidat
	ocl_args_d_t &ocl = getOcl();

	size_t result_size = sizeof(cl_float) * xsize * ysize;
	TrackMemory result_m(result_size, "clMinSquareValEx:result");
	cl_mem result = ocl.allocMem(result_size);

	cl_kernel kernel = ocl.kernel[KERNEL_MINSQUAREVAL];
	clSetKernelArgEx(kernel, &result, &xsize, &ysize, &img, &square_size, &offset);

	size_t globalWorkSize[2] = {xsize, ysize};
	cl_int err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	LOG_CL_RESULT(err);
	err = clEnqueueCopyBuffer(ocl.commandQueue, result, img, 0, 0, sizeof(cl_float) * xsize * ysize, 0, NULL, NULL);
	LOG_CL_RESULT(err);
	err = clFinish(ocl.commandQueue);
	LOG_CL_RESULT(err);
	clReleaseMemObject(result);
}

static void MakeMask(double extmul, double extoff,
					 double mul, double offset,
					 double scaler, double *result)
{
	for (size_t i = 0; i < 512; ++i)
	{
		const double c = mul / ((0.01 * scaler * i) + offset);
		result[i] = 1.0 + extmul * (c + extoff);
		result[i] *= result[i];
	}
}

static const double kInternalGoodQualityThreshold = 14.921561160295326;
static const double kGlobalScale = 1.0 / kInternalGoodQualityThreshold;

void clDoMask(ocl_channels mask /*in, out*/, ocl_channels mask_dc /*in, out*/, size_t xsize, size_t ysize)
{
	Perf clk("clDoMask");
	ocl_args_d_t &ocl = getOcl();

	double extmul = 0.975741017749;
	double extoff = -4.25328244168;
	double offset = 0.454909521427;
	double scaler = 0.0738288224836;
	double mul = 20.8029176447;
	static double lut_x[512];
	static bool lutx_init = false;
	if (!lutx_init)
	{
		lutx_init = true;
		MakeMask(extmul, extoff, mul, offset, scaler, lut_x);
	}

	extmul = 0.373995618954;
	extoff = 1.5307267433;
	offset = 0.911952641929;
	scaler = 1.1731667845;
	mul = 16.2447033988;
	static double lut_y[512];
	static bool luty_init = false;
	if (!luty_init)
	{
		luty_init = true;
		MakeMask(extmul, extoff, mul, offset, scaler, lut_y);
	}

	extmul = 0.61582234137;
	extoff = -4.25376118646;
	offset = 1.05105070921;
	scaler = 0.47434643535;
	mul = 31.1444967089;
	static double lut_b[512];
	static bool lutb_init = false;
	if (!lutb_init)
	{
		lutb_init = true;
		MakeMask(extmul, extoff, mul, offset, scaler, lut_b);
	}

	extmul = 1.79116943438;
	extoff = -3.86797479189;
	offset = 0.670960225853;
	scaler = 0.486575865525;
	mul = 20.4563479139;
	static double lut_dcx[512];
	static bool lutdcx_init = false;
	if (!lutdcx_init)
	{
		lutdcx_init = true;
		MakeMask(extmul, extoff, mul, offset, scaler, lut_dcx);
	}

	extmul = 0.212223514236;
	extoff = -3.65647120524;
	offset = 1.73396799447;
	scaler = 0.170392660501;
	mul = 21.6566724788;
	static double lut_dcy[512];
	static bool lutdcy_init = false;
	if (!lutdcy_init)
	{
		lutdcy_init = true;
		MakeMask(extmul, extoff, mul, offset, scaler, lut_dcy);
	}

	extmul = 0.349376011816;
	extoff = -0.894711072781;
	offset = 0.901647926679;
	scaler = 0.380086095024;
	mul = 18.0373825149;
	static double lut_dcb[512];
	static bool lutdcb_init = false;
	if (!lutdcb_init)
	{
		lutdcb_init = true;
		MakeMask(extmul, extoff, mul, offset, scaler, lut_dcb);
	}

	size_t channel_size = 512 * sizeof(double);
	TrackMemory xyb_m(channel_size * 3, "clDoMask:xyb");
	ocl_channels xyb = ocl.allocMemChannels(channel_size, lut_x, lut_y, lut_b);
	TrackMemory xyb_dc_m(channel_size * 3, "clDoMask:xyb_dc");
	ocl_channels xyb_dc = ocl.allocMemChannels(channel_size, lut_dcx, lut_dcy, lut_dcb);

	cl_kernel kernel = ocl.kernel[KERNEL_DOMASK];
	clSetKernelArgEx(kernel, &mask.r, &mask.g, &mask.b,
					 &xsize, &ysize,
					 &mask_dc.r, &mask_dc.g, &mask_dc.b,
					 &xyb.x, &xyb.y, &xyb.b,
					 &xyb_dc.x, &xyb_dc.y, &xyb_dc.b);

	size_t globalWorkSize[2] = {xsize, ysize};
	cl_int err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	LOG_CL_RESULT(err);
	err = clFinish(ocl.commandQueue);
	LOG_CL_RESULT(err);

	ocl.releaseMemChannels(xyb);
	ocl.releaseMemChannels(xyb_dc);
}

void clMaskEx(
	ocl_channels mask /*out*/, ocl_channels mask_dc /*out*/,
	const ocl_channels &rgb, const ocl_channels &rgb2,
	const size_t xsize, const size_t ysize)
{
	clDiffPrecomputeEx(mask, rgb, rgb2, xsize, ysize);
	for (int i = 0; i < 3; i++)
	{
		clAverage5x5Ex(mask.ch[i], xsize, ysize);
		clMinSquareValEx(mask.ch[i], xsize, ysize, 4, 0);

		static const double sigma[3] = {
			9.65781083553,
			14.2644604355,
			4.53358927369,
		};

		clBlurEx(mask.ch[i], xsize, ysize, sigma[i], 0.0);
	}

	clDoMask(mask, mask_dc, xsize, ysize);

	for (int i = 0; i < 3; i++)
	{
		clScaleImageEx(mask.ch[i], xsize * ysize, kGlobalScale * kGlobalScale);
		clScaleImageEx(mask_dc.ch[i], xsize * ysize, kGlobalScale * kGlobalScale);
	}
}

void clCombineChannelsEx(
	cl_mem result /*out*/,
	const ocl_channels &mask,
	const ocl_channels &mask_dc,
	const size_t xsize, const size_t ysize,
	const cl_mem block_diff_dc,
	const cl_mem block_diff_ac,
	const cl_mem edge_detector_map,
	const size_t res_xsize,
	const size_t step)
{
	Perf clk("clCombineChannelsEx");
	ocl_args_d_t &ocl = getOcl();

	const size_t work_xsize = ((xsize - 8 + step) + step - 1) / step;
	const size_t work_ysize = ((ysize - 8 + step) + step - 1) / step;

	cl_kernel kernel = ocl.kernel[KERNEL_COMBINECHANNELS];
	clSetKernelArgEx(kernel, &result,
					 &mask.r, &mask.g, &mask.b,
					 &mask_dc.r, &mask_dc.g, &mask_dc.b,
					 &xsize, &ysize,
					 &block_diff_dc, &block_diff_ac,
					 &edge_detector_map,
					 &res_xsize,
					 &step);

	size_t globalWorkSize[2] = {work_xsize, work_ysize};
	cl_int err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	LOG_CL_RESULT(err);
	err = clFinish(ocl.commandQueue);
	LOG_CL_RESULT(err);
}

void clUpsampleSquareRootEx(cl_mem diffmap, const size_t xsize, const size_t ysize, const int step)
{
	Perf clk("clUpsampleSquareRootEx");
	ocl_args_d_t &ocl = getOcl();

	size_t diffmap_out_size = xsize * ysize * sizeof(float);
	TrackMemory diffmap_out_m(diffmap_out_size, "clUpsampleSquareRootEx:diffmap_out");
	cl_mem diffmap_out = ocl.allocMem(diffmap_out_size);

	cl_kernel kernel = ocl.kernel[KERNEL_UPSAMPLESQUAREROOT];
	clSetKernelArgEx(kernel, &diffmap_out, &diffmap, &xsize, &ysize, &step);

	const size_t res_xsize = (xsize + step - 1) / step;
	const size_t res_ysize = (ysize + step - 1) / step;

	size_t globalWorkSize[2] = {res_xsize, res_ysize};
	cl_int err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	LOG_CL_RESULT(err);
	err = clEnqueueCopyBuffer(ocl.commandQueue, diffmap_out, diffmap, 0, 0, xsize * ysize * sizeof(float), 0, NULL, NULL);
	LOG_CL_RESULT(err);
	err = clFinish(ocl.commandQueue);
	LOG_CL_RESULT(err);

	clReleaseMemObject(diffmap_out);
}

void clRemoveBorderEx(cl_mem out, const cl_mem in, const size_t xsize, const size_t ysize, const int step)
{
	Perf clk("clRemoveBorderEx");
	ocl_args_d_t &ocl = getOcl();

	cl_int cls = 8 - step;
	cl_int cls2 = (8 - step) / 2;

	int out_xsize = xsize - cls;
	int out_ysize = ysize - cls;

	cl_kernel kernel = ocl.kernel[KERNEL_REMOVEBORDER];
	clSetKernelArgEx(kernel, &out, &out_xsize, &out_ysize, &in, &cls, &cls2);

	size_t globalWorkSize[2] = {out_xsize, out_ysize};
	cl_int err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	LOG_CL_RESULT(err);
	err = clFinish(ocl.commandQueue);
	LOG_CL_RESULT(err);
}

void clAddBorderEx(cl_mem out, size_t xsize, size_t ysize, int step, cl_mem in)
{
	Perf clk("clAddBorderEx");
	ocl_args_d_t &ocl = getOcl();

	cl_int cls = 8 - step;
	cl_int cls2 = (8 - step) / 2;
	cl_kernel kernel = ocl.kernel[KERNEL_ADDBORDER];
	clSetKernelArgEx(kernel, &out, &xsize, &ysize, &cls, &cls2, &in);

	size_t globalWorkSize[2] = {xsize, ysize};
	cl_int err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	LOG_CL_RESULT(err);
	err = clFinish(ocl.commandQueue);
	LOG_CL_RESULT(err);
}

void clCalculateDiffmapEx(cl_mem diffmap /*in,out*/, const size_t xsize, const size_t ysize, const int step)
{
	Perf clk("clCalculateDiffmapEx");
	clUpsampleSquareRootEx(diffmap, xsize, ysize, step);

	static const double kSigma = 8.8510880283;
	static const double mul1 = 24.8235314874;
	static const double scale = 1.0 / (1.0 + mul1);

	const int s = 8 - step;
	int s2 = (8 - step) / 2;

	ocl_args_d_t &ocl = getOcl();
	size_t blurred_size = (xsize - s) * (ysize - s) * sizeof(float);
	TrackMemory blurred_m(blurred_size, "clCalculateDiffmapEx:blurred");
	cl_mem blurred = ocl.allocMem(blurred_size);
	clRemoveBorderEx(blurred, diffmap, xsize, ysize, step);

	static const double border_ratio = 0.03027655136;
	clBlurEx(blurred, xsize - s, ysize - s, kSigma, border_ratio);

	clAddBorderEx(diffmap, xsize, ysize, step, blurred);
	clScaleImageEx(diffmap, xsize * ysize, scale);

	clReleaseMemObject(blurred);
}

void clCopyFromJpegComponent(
	coeff_t *output_batch /*in,out*/,
	uint8_t *output_idct /*out*/,
	const coeff_t *jpeg_batch /*in*/,
	const int *quant,
	const int jpeg_block_width,
	const int jpeg_block_height,
	const int output_block_width,
	const int output_block_height,
	const int output_width,
	const int output_height)
{
	using namespace guetzli;

	Perf clk("clCopyFromJpegComponent");
	ocl_args_d_t &ocl = getOcl();

	int src_block_count = jpeg_block_width * jpeg_block_height;
	size_t src_coeff_size = src_block_count * sizeof(::coeff_t) * kDCTBlockSize;
	TrackMemory src_coeff_m(src_coeff_size, "clCopyFromJpegComponent:src_coeff");
	cl_mem src_coeff = ocl.allocMem(src_coeff_size, jpeg_batch);

	int dst_coeff_size = output_block_width * output_block_height * sizeof(::coeff_t) * kDCTBlockSize;
	TrackMemory dst_coeff_m(dst_coeff_size, "clCopyFromJpegComponent:dst_coeff");
	cl_mem dst_coeff = ocl.allocMem(dst_coeff_size, output_batch);

	int dst_idct_size = output_block_width * output_block_height * sizeof(uint8_t) * kDCTBlockSize;
	TrackMemory dst_idct_m(dst_idct_size, "clCopyFromJpegComponent:dst_idct");
	cl_mem dst_idct = ocl.allocMem(dst_idct_size, output_idct);

	int src_quant_size = kDCTBlockSize * sizeof(int);
	TrackMemory src_quant_m(src_quant_size, "clCopyFromJpegComponent:src_quant");
	cl_mem src_quant = ocl.allocMem(src_quant_size, quant);

	cl_kernel kernel = ocl.kernel[KERNEL_COPYFROMJPEGCOMPONENT];
	clSetKernelArgEx(kernel, &dst_coeff, &dst_idct,
					 &src_coeff, &src_quant, &jpeg_block_width, &jpeg_block_height,
					 &output_block_width, &output_block_height,
					 &output_width, &output_height);

	size_t globalWorkSize[2] = {output_block_width, output_block_height};
	cl_int err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	LOG_CL_RESULT(err);
	err = clFinish(ocl.commandQueue);
	LOG_CL_RESULT(err);

	err = clEnqueueReadBuffer(ocl.commandQueue, dst_coeff, false, 0, dst_coeff_size, output_batch, 0, NULL, NULL);
	err = clEnqueueReadBuffer(ocl.commandQueue, dst_idct, false, 0, dst_idct_size, output_idct, 0, NULL, NULL);
	err = clFinish(ocl.commandQueue);

	clReleaseMemObject(src_coeff);
	clReleaseMemObject(dst_coeff);
	clReleaseMemObject(dst_idct);
	clReleaseMemObject(src_quant);
}

void clApplyGlobalQuantization(
	coeff_t *output_batch /*in,out*/,
	uchar *output_idct /*out*/,
	uchar *output_bool /*out*/,
	const int *q /*in*/,
	const int block_width,
	const int block_height)
{
	using namespace guetzli;

	Perf clk("clApplyGlobalQuantization");
	ocl_args_d_t &ocl = getOcl();

	int dst_coeff_size = block_width * block_height * sizeof(::coeff_t) * kDCTBlockSize;
	TrackMemory dst_coeff_m(dst_coeff_size, "clApplyGlobalQuantization:dst_coeff");
	cl_mem dst_coeff = ocl.allocMem(dst_coeff_size, output_batch);

	int dst_idct_size = block_width * block_height * sizeof(uint8_t) * kDCTBlockSize;
	TrackMemory dst_idct_m(dst_idct_size, "clApplyGlobalQuantization:dst_idct");
	cl_mem dst_idct = ocl.allocMem(dst_idct_size, output_idct);

	int dst_bool_size = block_width * block_height * sizeof(uchar);
	TrackMemory dst_bool_m(dst_bool_size, "clApplyGlobalQuantization:dst_bool");
	cl_mem dst_bool = ocl.allocMem(dst_bool_size, output_bool);

	int src_q_size = kDCTBlockSize * sizeof(int);
	TrackMemory src_q_m(src_q_size, "clApplyGlobalQuantization:src_q");
	cl_mem src_q = ocl.allocMem(src_q_size, q);

	cl_kernel kernel = ocl.kernel[KERNEL_APPLYGLOBALQUANTIZATION];
	clSetKernelArgEx(kernel, &dst_coeff, &dst_idct,
					 &dst_bool, &src_q, &block_width, &block_height);

	size_t globalWorkSize[2] = {block_width, block_height};
	cl_int err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	LOG_CL_RESULT(err);
	err = clFinish(ocl.commandQueue);
	LOG_CL_RESULT(err);

	err = clEnqueueReadBuffer(ocl.commandQueue, dst_coeff, false, 0, dst_coeff_size, output_batch, 0, NULL, NULL);
	err = clEnqueueReadBuffer(ocl.commandQueue, dst_idct, false, 0, dst_idct_size, output_idct, 0, NULL, NULL);
	err = clEnqueueReadBuffer(ocl.commandQueue, dst_bool, false, 0, dst_bool_size, output_bool, 0, NULL, NULL);
	err = clFinish(ocl.commandQueue);

	clReleaseMemObject(dst_bool);
	clReleaseMemObject(dst_coeff);
	clReleaseMemObject(dst_idct);
	clReleaseMemObject(src_q);
}

void clComponentsToPixels(
	uchar *rgb, /*out*/
	const int xmin,
	const int ymin,
	const int xsize,
	const int ysize,
	const std::vector<guetzli::OutputImageComponent> &components /*in*/)
{
	using namespace guetzli;

	Perf clk("clComponentsToPixels");
	ocl_args_d_t &ocl = getOcl();

	const int stride = 3;

	int out_size = xsize * ysize * stride * sizeof(uchar);
	TrackMemory cl_out_m(out_size, "clComponentsToPixels:cl_out");
	cl_mem cl_out = ocl.allocMem(out_size, rgb);

	for (int c = 0; c < stride; ++c)
	{
		cl_int err;
		cl_buffer_region region;
		region.origin = c;
		region.size = out_size - c;
		cl_mem cl_out_offset = clCreateSubBuffer(cl_out, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);

		const int width = components[c].width();
		const int height = components[c].height();

		const int yend1 = ymin + ysize;
		const int yend0 = std::min(yend1, height);
		const int xend1 = xmin + xsize;
		const int xend0 = std::min(xend1, width);

		int pixels_size = width * height * sizeof(ushort);
		TrackMemory cl_pixels_m(pixels_size, "clComponentsToPixels:cl_pixels");
		cl_mem cl_pixels = ocl.allocMem(pixels_size, components[c].pixels());

		{
			cl_kernel kernel = ocl.kernel[KERNEL_COMPONENTSTOPIXELS];
			clSetKernelArgEx(kernel,
							 &cl_out_offset, &xmin, &ymin, &xsize, &ysize, &stride,
							 &cl_pixels, &width, &height);

			size_t globalWorkSize[2] = {xend0 - xmin, yend0 - ymin};
			cl_int err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
			LOG_CL_RESULT(err);
			err = clFinish(ocl.commandQueue);
			LOG_CL_RESULT(err);
		}

		clReleaseMemObject(cl_pixels);

		if (xend1 - xend0 > 0)
		{
			cl_kernel kernel = ocl.kernel[KERNEL_COMPONENTSTOPIXELS_EX1];
			clSetKernelArgEx(kernel,
							 &cl_out_offset, &xmin, &ymin, &xsize, &ysize, &stride,
							 &width, &height);

			size_t globalWorkSize[2] = {xend1 - xend0, yend0 - ymin};
			cl_int err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
			LOG_CL_RESULT(err);
			err = clFinish(ocl.commandQueue);
			LOG_CL_RESULT(err);
		}

		if (yend1 - yend0 > 0)
		{
			cl_kernel kernel = ocl.kernel[KERNEL_COMPONENTSTOPIXELS_EX2];
			clSetKernelArgEx(kernel,
							 &cl_out_offset, &xmin, &ymin, &xsize, &ysize, &stride,
							 &width, &height);

			size_t globalWorkSize[2] = {xsize, yend1 - yend0};
			cl_int err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
			LOG_CL_RESULT(err);
			err = clFinish(ocl.commandQueue);
			LOG_CL_RESULT(err);
		}

		clReleaseMemObject(cl_out_offset);
	}

	cl_kernel kernel = ocl.kernel[KERNEL_COLORTRANSFORMYCBCRTORGB];
	clSetKernelArgEx(kernel, &cl_out);

	size_t globalWorkSize[2] = {xsize * ysize, 1};
	cl_int err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	LOG_CL_RESULT(err);
	err = clFinish(ocl.commandQueue);
	LOG_CL_RESULT(err);

	clEnqueueReadBuffer(ocl.commandQueue, cl_out, CL_TRUE, 0, out_size, rgb, 0, NULL, NULL);
	clReleaseMemObject(cl_out);
}

#ifdef __USE_DOUBLE_AS_FLOAT__
#undef double
#endif

#endif
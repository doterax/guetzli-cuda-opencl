/*
 * OpenCL edition implementation of guetzli.
 *
 * Author: strongtu@tencent.com
 *         ianhuang@tencent.com
 *         chriskzhou@tencent.com
 *		  stephendeng@tencent.com
 */
#include <math.h>
#include <algorithm>
#include <vector>
#include <set>
#include <map>
#include <string.h>
#include "backend/opencl/opencl_kernels.h"
#include "backend/opencl/kernel_args.h"
#define LOG
#define UTILITIES_FILE
#include "third_party/OpenCL-Wrapper/opencl.hpp"

#ifdef __USE_OPENCL__
// Largest work-group the kernel can be launched with on our device (cached per kernel).
static size_t clKernelMaxWorkGroup(cl_kernel kernel) {
  static std::map<cl_kernel, size_t> cache;
  std::map<cl_kernel, size_t>::iterator it = cache.find(kernel);
  if (it != cache.end()) return it->second;

  cl_device_id dev = NULL;
  LOG_CL_RESULT(clGetCommandQueueInfo(getOcl().commandQueue, CL_QUEUE_DEVICE, sizeof(dev), &dev, NULL));
  size_t wg = 0;
  LOG_CL_RESULT(clGetKernelWorkGroupInfo(kernel, dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof(wg), &wg, NULL));
  if (wg == 0) wg = 1;
  cache[kernel] = wg;
  return wg;
}

static size_t roundUp(size_t v, size_t m) {
  return (v + m - 1) / m * m;
}

// Enqueues a w x h range padded up to whole lx x ly work-groups. Every kernel
// bounds-checks its global id, so padding threads just return. (Was: NULL local
// size, which lets drivers pick tiny groups for odd sizes -- a prime width gives
// groups of one -- or an AMD-only divisor search.)
static cl_int clEnqueueRange2D(cl_kernel kernel, size_t w, size_t h, size_t lx = 16, size_t ly = 16) {
  if (w == 0 || h == 0) return CL_SUCCESS;
  const size_t max_wg = clKernelMaxWorkGroup(kernel);
  while (lx * ly > max_wg && ly > 1) ly /= 2;
  while (lx * ly > max_wg && lx > 1) lx /= 2;
  // Short ranges (e.g. a few rows): trade group height for width.
  while (ly > 1 && ly / 2 >= h) {
    ly /= 2;
    lx *= 2;
  }
  size_t global[2] = {roundUp(w, lx), roundUp(h, ly)};
  size_t local[2] = {lx, ly};
  return clEnqueueNDRangeKernel(getOcl().commandQueue, kernel, 2, NULL, global, local, 0, NULL, NULL);
}

static cl_int clEnqueueRange1D(cl_kernel kernel, size_t n, size_t l = 256) {
  if (n == 0) return CL_SUCCESS;
  const size_t max_wg = clKernelMaxWorkGroup(kernel);
  while (l > max_wg && l > 1) l /= 2;
  size_t global[1] = {roundUp(n, l)};
  size_t local[1] = {l};
  return clEnqueueNDRangeKernel(getOcl().commandQueue, kernel, 1, NULL, global, local, 0, NULL, NULL);
}
#endif

#ifdef __USE_OPENCL__

void clOpsinDynamicsImage(float* r, float* g, float* b, const unsigned int xsize, const unsigned int ysize) {
  size_t elements = xsize * ysize;
  size_t channel_size = elements * sizeof(float);

  ocl_args_d_t& ocl = getOcl();

  ocl_channels rgb = ocl.allocMemChannels(channel_size, r, g, b);

  clOpsinDynamicsImageEx(rgb, xsize, ysize);

  LOG_CL_RESULT(clEnqueueReadBuffer(ocl.commandQueue, rgb.r, false, 0, channel_size, r, 0, NULL, NULL));
  LOG_CL_RESULT(clEnqueueReadBuffer(ocl.commandQueue, rgb.g, false, 0, channel_size, g, 0, NULL, NULL));
  LOG_CL_RESULT(clEnqueueReadBuffer(ocl.commandQueue, rgb.b, false, 0, channel_size, b, 0, NULL, NULL));
  LOG_CL_RESULT(clFinish(ocl.commandQueue));

  ocl.releaseMemChannels(rgb);
}

void clComputeBlockZeroingOrder(guetzli::CoeffData* output_order_batch, const channel_info orig_channel[3],
                                const float* orig_image_batch, const float* mask_scale, const int image_width,
                                const int image_height, const channel_info mayout_channel[3], const int factor,
                                const int comp_mask, const float BlockErrorLimit) {
  // Validate factor to prevent division by zero
  if (factor <= 0) {
    LogInfo("Error: factor is %d, must be > 0. Skipping clComputeBlockZeroingOrder.\n", factor);
    return;
  }

  const int block8_width = (image_width + 8 - 1) / 8;
  const int block8_height = (image_height + 8 - 1) / 8;
  const int blockf_width = (image_width + 8 * factor - 1) / (8 * factor);
  const int blockf_height = (image_height + 8 * factor - 1) / (8 * factor);

  using namespace guetzli;

  ocl_args_d_t& ocl = getOcl();

  cl_mem mem_orig_coeff[3];
  cl_mem mem_mayout_coeff[3];
  cl_mem mem_mayout_pixel[3];
  size_t total_orig_coeff_size = 0;
  size_t total_mayout_coeff_size = 0;
  size_t total_mayout_pixel_size = 0;
  for (int c = 0; c < 3; c++) {
    int block_count = orig_channel[c].block_width * orig_channel[c].block_height;
    size_t orig_coeff_size = block_count * sizeof(::coeff_t) * kDCTBlockSize;
    total_orig_coeff_size += orig_coeff_size;

    block_count = mayout_channel[c].block_width * mayout_channel[c].block_height;
    size_t mayout_coeff_size = block_count * sizeof(::coeff_t) * kDCTBlockSize;
    total_mayout_coeff_size += mayout_coeff_size;

    size_t mayout_pixel_size = image_width * image_height * sizeof(uint16_t);
    total_mayout_pixel_size += mayout_pixel_size;
  }

  size_t orig_image_size = sizeof(float) * 3 * kDCTBlockSize * block8_width * block8_height;
  cl_mem mem_orig_image = ocl.allocMem(orig_image_size, orig_image_batch);

  size_t mask_scale_size = sizeof(float) * 3 * block8_width * block8_height;
  cl_mem mem_mask_scale = ocl.allocMem(mask_scale_size, mask_scale);

  size_t output_order_batch_size = sizeof(CoeffData) * 3 * kDCTBlockSize * blockf_width * blockf_height;
  // Zero-filled (the kernel writes only each block's valid entries).
  cl_mem mem_output_order_batch = ocl.allocMem(output_order_batch_size);

  for (int c = 0; c < 3; c++) {
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
  clSetKernelArgEx(kernel, &mem_orig_coeff[0], &mem_orig_coeff[1], &mem_orig_coeff[2], &mem_orig_image, &mem_mask_scale,
                   &blockf_width, &blockf_height, &image_width, &image_height, &mem_mayout_coeff[0],
                   &mem_mayout_coeff[1], &mem_mayout_coeff[2], &mem_mayout_pixel[0], &mem_mayout_pixel[1],
                   &mem_mayout_pixel[2], &orig_channel[0].block_width, &orig_channel[1].block_width,
                   &orig_channel[2].block_width, &mayout_channel[0].factor, &mayout_channel[1].factor,
                   &mayout_channel[2].factor, &mayout_channel[0].block_width, &mayout_channel[1].block_width,
                   &mayout_channel[2].block_width, &mayout_channel[0].block_height, &mayout_channel[1].block_height,
                   &mayout_channel[2].block_height, &factor, &comp_mask, &BlockErrorLimit, &mem_output_order_batch);

  // Heavy per-thread state limits occupancy anyway: one-warp groups so small
  // images (few blocks) still spread across all compute units.
  cl_int err = clEnqueueRange2D(kernel, blockf_width, blockf_height, 8, 4);
  LOG_CL_RESULT(err);

  LOG_CL_RESULT(clEnqueueReadBuffer(ocl.commandQueue, mem_output_order_batch, false, 0, output_order_batch_size,
                                    output_order_batch, 0, NULL, NULL));
  LOG_CL_RESULT(clFinish(ocl.commandQueue));

  for (int c = 0; c < 3; c++) {
    ocl.releaseMem(mem_orig_coeff[c]);
    ocl.releaseMem(mem_mayout_coeff[c]);
    ocl.releaseMem(mem_mayout_pixel[c]);
  }

  ocl.releaseMem(mem_orig_image);
  ocl.releaseMem(mem_mask_scale);
  ocl.releaseMem(mem_output_order_batch);
}

void clMask(float* mask_r, float* mask_g, float* mask_b, float* maskdc_r, float* maskdc_g, float* maskdc_b,
            const unsigned int xsize, const unsigned int ysize, const float* r, const float* g, const float* b,
            const float* r2, const float* g2, const float* b2) {
  ocl_args_d_t& ocl = getOcl();

  size_t channel_size = xsize * ysize * sizeof(float);

  ocl_channels rgb = ocl.allocMemChannels(channel_size, r, g, b);
  ocl_channels rgb2 = ocl.allocMemChannels(channel_size, r2, g2, b2);
  ocl_channels mask = ocl.allocMemChannels(channel_size);
  ocl_channels mask_dc = ocl.allocMemChannels(channel_size);

  clMaskEx(mask, mask_dc, rgb, rgb2, xsize, ysize);

  LOG_CL_RESULT(clEnqueueReadBuffer(ocl.commandQueue, mask.r, false, 0, channel_size, mask_r, 0, NULL, NULL));
  LOG_CL_RESULT(clEnqueueReadBuffer(ocl.commandQueue, mask.g, false, 0, channel_size, mask_g, 0, NULL, NULL));
  LOG_CL_RESULT(clEnqueueReadBuffer(ocl.commandQueue, mask.b, false, 0, channel_size, mask_b, 0, NULL, NULL));
  LOG_CL_RESULT(clEnqueueReadBuffer(ocl.commandQueue, mask_dc.r, false, 0, channel_size, maskdc_r, 0, NULL, NULL));
  LOG_CL_RESULT(clEnqueueReadBuffer(ocl.commandQueue, mask_dc.g, false, 0, channel_size, maskdc_g, 0, NULL, NULL));
  LOG_CL_RESULT(clEnqueueReadBuffer(ocl.commandQueue, mask_dc.b, false, 0, channel_size, maskdc_b, 0, NULL, NULL));
  LOG_CL_RESULT(clFinish(ocl.commandQueue));

  ocl.releaseMemChannels(rgb);
  ocl.releaseMemChannels(rgb2);
  ocl.releaseMemChannels(mask);
  ocl.releaseMemChannels(mask_dc);
}

void clDiffmapOpsinDynamicsImageEx(cl_mem result, ocl_channels xyb0, ocl_channels xyb1, const unsigned int xsize,
                                   const unsigned int ysize, const unsigned int step) {
  const size_t res_xsize = (xsize + step - 1) / step;
  const size_t res_ysize = (ysize + step - 1) / step;

  size_t channel_size = xsize * ysize * sizeof(float);
  size_t channel_step_size = res_xsize * res_ysize * sizeof(float);

  ocl_args_d_t& ocl = getOcl();

  size_t edge_detector_map_size = 3 * channel_step_size;
  cl_mem edge_detector_map = ocl.allocMem(edge_detector_map_size);

  size_t block_diff_dc_size = 3 * channel_step_size;
  cl_mem block_diff_dc = ocl.allocMem(block_diff_dc_size);

  size_t block_diff_ac_size = 3 * channel_step_size;
  cl_mem block_diff_ac = ocl.allocMem(block_diff_ac_size);

  clMaskHighIntensityChangeEx(xyb0, xyb1, xsize, ysize);

  clEdgeDetectorMapEx(edge_detector_map, xyb0, xyb1, xsize, ysize, step);
  clBlockDiffMapEx(block_diff_dc, block_diff_ac, xyb0, xyb1, xsize, ysize, step);
  clEdgeDetectorLowFreqEx(block_diff_ac, xyb0, xyb1, xsize, ysize, step);
  {
    // Every pixel of mask (DiffPrecompute) and mask_dc (DoMask) is written.
    ocl_channels mask = ocl.allocMemChannelsUninit(channel_size);
    ocl_channels mask_dc = ocl.allocMemChannelsUninit(channel_size);
    clMaskEx(mask, mask_dc, xyb0, xyb1, xsize, ysize);
    clCombineChannelsEx(result, mask, mask_dc, xsize, ysize, block_diff_dc, block_diff_ac, edge_detector_map, res_xsize,
                        step);

    ocl.releaseMemChannels(mask);
    ocl.releaseMemChannels(mask_dc);
  }

  clCalculateDiffmapEx(result, xsize, ysize, step);

  ocl.releaseMem(edge_detector_map);
  ocl.releaseMem(block_diff_dc);
  ocl.releaseMem(block_diff_ac);
}
// Scale table for a convolution with the (constant, cached) weights buffer
// multipliers over n samples: clConvolutionScaleEx is run once per
// {multipliers, n, step, border_ratio} and the result kept on the device.
static cl_mem clGetConvolutionScale(const cl_mem multipliers, int len, int n, int step, int offset,
                                    float border_ratio) {
  ocl_args_d_t& ocl = getOcl();

  unsigned int br_bits;
  memcpy(&br_bits, &border_ratio, sizeof(br_bits));
  std::vector<size_t> key(4);
  key[0] = (size_t)multipliers;
  key[1] = (size_t)n;
  key[2] = (size_t)step;
  key[3] = (size_t)br_bits;
  std::map<std::vector<size_t>, cl_mem>::iterator it = ocl.convScales.find(key);
  if (it != ocl.convScales.end()) return it->second;

  const int count = (n + step - 1) / step;
  cl_int err = CL_SUCCESS;
  cl_mem scale = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, count * sizeof(float), NULL, &err);
  LOG_CL_RESULT(err);
  ocl.convScales[key] = scale;  // owned by the context from here on

  cl_kernel kernel = ocl.kernel[KERNEL_CONVOLUTIONSCALE];
  clSetKernelArgEx(kernel, &scale, &n, &multipliers, &len, &step, &offset, &border_ratio, &count);
  LOG_CL_RESULT(clEnqueueRange1D(kernel, count, 64));
  return scale;
}

void clConvolutionXEx(cl_mem result /*out*/, const cl_mem inp, unsigned int xsize, unsigned int ysize,
                      const cl_mem multipliers, unsigned int len, int xstep, int offset, float border_ratio) {
  ocl_args_d_t& ocl = getOcl();

  cl_mem scale = clGetConvolutionScale(multipliers, len, xsize, xstep, offset, border_ratio);

  cl_kernel kernel = ocl.kernel[KERNEL_CONVOLUTIONX];
  clSetKernelArgEx(kernel, &result, &xsize, &ysize, &inp, &multipliers, &len, &xstep, &offset, &scale);

  size_t x_count = (xsize + xstep - 1) / xstep;
  cl_int err = clEnqueueRange2D(kernel, x_count, ysize);
  LOG_CL_RESULT(err);
}

void clConvolutionYEx(cl_mem result /*out*/, const cl_mem inp, unsigned int xsize, unsigned int ysize,
                      const cl_mem multipliers, unsigned int len, int xstep, int offset, float border_ratio) {
  ocl_args_d_t& ocl = getOcl();

  cl_mem scale = clGetConvolutionScale(multipliers, len, ysize, xstep, offset, border_ratio);

  cl_kernel kernel = ocl.kernel[KERNEL_CONVOLUTIONY];
  clSetKernelArgEx(kernel, &result, &xsize, &ysize, &inp, &multipliers, &len, &xstep, &offset, &scale);

  size_t x_count = (xsize + xstep - 1) / xstep;
  size_t y_count = (ysize + xstep - 1) / xstep;
  cl_int err = clEnqueueRange2D(kernel, x_count, y_count);
  LOG_CL_RESULT(err);
}

void clSquareSampleEx(cl_mem result /*out*/, const cl_mem image, unsigned int xsize, unsigned int ysize,
                      unsigned int xstep, unsigned int ystep) {
  ocl_args_d_t& ocl = getOcl();

  cl_kernel kernel = ocl.kernel[KERNEL_SQUARESAMPLE];
  clSetKernelArgEx(kernel, &result, &xsize, &ysize, &image, &xstep, &ystep);

  cl_int err = clEnqueueRange2D(kernel, xsize, ysize);
  LOG_CL_RESULT(err);
}

void clBlurEx(cl_mem image, const unsigned int xsize, const unsigned int ysize, const float sigma,
              const float border_ratio, cl_mem result) {
  float m = 2.25;  // Accuracy increases when m is increased.
  const float scaler = -1.0 / (2 * sigma * sigma);
  // For m = 9.0: exp(-scaler * diff * diff) < 2^ {-52}
  const int diff = std::max<int>(1, m * fabs(sigma));
  const int expn_size = 2 * diff + 1;
  std::vector<float> expn(expn_size);
  for (int i = -diff; i <= diff; ++i) {
    expn[i + diff] = static_cast<float>(exp(scaler * i * i));
  }

  const int xstep = std::max<int>(1, int(sigma / 3));

  ocl_args_d_t& ocl = getOcl();

  // Weights depend only on sigma: uploaded once per sigma and kept on the device.
  const cl_mem mem_expn = ocl.getBlurWeights(sigma, expn);

  // Every element ConvolutionY reads is written by ConvolutionX: no zero-fill needed.
  cl_mem temp = ocl.allocMemUninit(xsize * ysize * sizeof(float));
  clConvolutionXEx(temp, image, xsize, ysize, mem_expn, expn_size, xstep, diff, border_ratio);
  clConvolutionYEx(result, temp, xsize, ysize, mem_expn, expn_size, xstep, diff, border_ratio);
  if (xstep > 1) {
    clSquareSampleEx(result, result, xsize, ysize, xstep, xstep);
  }
  ocl.releaseMem(temp);
}

void clOpsinDynamicsImageEx(ocl_channels& rgb, const unsigned int xsize, const unsigned int ysize) {
  static const float kSigma = 1.1;
  const size_t size = xsize * ysize;
  size_t channel_size = size * sizeof(float);

  ocl_args_d_t& ocl = getOcl();

  // Fully overwritten by the blurs.
  ocl_channels blurred = ocl.allocMemChannelsUninit(channel_size);

  clBlurEx(rgb.r, xsize, ysize, kSigma, 0.0, blurred.r);
  clBlurEx(rgb.g, xsize, ysize, kSigma, 0.0, blurred.g);
  clBlurEx(rgb.b, xsize, ysize, kSigma, 0.0, blurred.b);

  cl_kernel kernel = ocl.kernel[KERNEL_OPSINDYNAMICSIMAGE];
  const int size_arg = (int)size;
  clSetKernelArgEx(kernel, &rgb.r, &rgb.g, &rgb.b, &size_arg, &blurred.r, &blurred.g, &blurred.b);

  // https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clGetKernelSuggestedLocalWorkSizeKHR.html
  cl_int err = clEnqueueRange1D(kernel, size);
  LOG_CL_RESULT(err);

  ocl.releaseMemChannels(blurred);
}

void clMaskHighIntensityChangeEx(ocl_channels& xyb0 /*in,out*/, ocl_channels& xyb1 /*in,out*/, const unsigned int xsize,
                                 const unsigned int ysize) {
  size_t channel_size = xsize * ysize * sizeof(float);

  ocl_args_d_t& ocl = getOcl();

  ocl_channels c0 = ocl.allocMemChannelsUninit(channel_size);  // full copies below
  ocl_channels c1 = ocl.allocMemChannelsUninit(channel_size);

  LOG_CL_RESULT(clEnqueueCopyBuffer(ocl.commandQueue, xyb0.r, c0.r, 0, 0, channel_size, 0, NULL, NULL));
  LOG_CL_RESULT(clEnqueueCopyBuffer(ocl.commandQueue, xyb0.g, c0.g, 0, 0, channel_size, 0, NULL, NULL));
  LOG_CL_RESULT(clEnqueueCopyBuffer(ocl.commandQueue, xyb0.b, c0.b, 0, 0, channel_size, 0, NULL, NULL));
  LOG_CL_RESULT(clEnqueueCopyBuffer(ocl.commandQueue, xyb1.r, c1.r, 0, 0, channel_size, 0, NULL, NULL));
  LOG_CL_RESULT(clEnqueueCopyBuffer(ocl.commandQueue, xyb1.g, c1.g, 0, 0, channel_size, 0, NULL, NULL));
  LOG_CL_RESULT(clEnqueueCopyBuffer(ocl.commandQueue, xyb1.b, c1.b, 0, 0, channel_size, 0, NULL, NULL));

  cl_kernel kernel = ocl.kernel[KERNEL_MASKHIGHINTENSITYCHANGE];
  clSetKernelArgEx(kernel, &xyb0.r, &xyb0.g, &xyb0.b, &xsize, &ysize, &xyb1.r, &xyb1.g, &xyb1.b, &c0.r, &c0.g, &c0.b,
                   &c1.r, &c1.g, &c1.b);

  cl_int err = clEnqueueRange2D(kernel, xsize, ysize);
  LOG_CL_RESULT(err);

  ocl.releaseMemChannels(c0);
  ocl.releaseMemChannels(c1);
}

void clEdgeDetectorMapEx(cl_mem result /*out*/, const ocl_channels& rgb, const ocl_channels& rgb2,
                         const unsigned int xsize, const unsigned int ysize, const unsigned int step) {
  size_t channel_size = xsize * ysize * sizeof(float);

  ocl_args_d_t& ocl = getOcl();

  ocl_channels rgb_blured = ocl.allocMemChannelsUninit(channel_size);  // fully written by the blurs
  ocl_channels rgb2_blured = ocl.allocMemChannelsUninit(channel_size);

  static const float kSigma[3] = {1.5, 0.586, 0.4};

  for (int i = 0; i < 3; i++) {
    clBlurEx(rgb.ch[i], xsize, ysize, kSigma[i], 0.0, rgb_blured.ch[i]);
    clBlurEx(rgb2.ch[i], xsize, ysize, kSigma[i], 0.0, rgb2_blured.ch[i]);
  }

  const size_t res_xsize = (xsize + step - 1) / step;
  const size_t res_ysize = (ysize + step - 1) / step;

  cl_kernel kernel = ocl.kernel[KERNEL_EDGEDETECTOR];
  clSetKernelArgEx(kernel, &result, &res_xsize, &res_ysize, &rgb_blured.r, &rgb_blured.g, &rgb_blured.b, &rgb2_blured.r,
                   &rgb2_blured.g, &rgb2_blured.b, &xsize, &ysize, &step);

  cl_int err = clEnqueueRange2D(kernel, res_xsize, res_ysize);
  LOG_CL_RESULT(err);

  ocl.releaseMemChannels(rgb_blured);
  ocl.releaseMemChannels(rgb2_blured);
}

void clBlockDiffMapEx(cl_mem block_diff_dc /*out*/, cl_mem block_diff_ac /*out*/, const ocl_channels& rgb,
                      const ocl_channels& rgb2, const unsigned int xsize, const unsigned int ysize,
                      const unsigned int step) {
  ocl_args_d_t& ocl = getOcl();

  const size_t res_xsize = (xsize + step - 1) / step;
  const size_t res_ysize = (ysize + step - 1) / step;

  cl_kernel kernel = ocl.kernel[KERNEL_BLOCKDIFFMAP];
  clSetKernelArgEx(kernel, &block_diff_dc, &block_diff_ac, &res_xsize, &res_ysize, &rgb.r, &rgb.g, &rgb.b, &rgb2.r,
                   &rgb2.g, &rgb2.b, &xsize, &ysize, &step);

  cl_int err = clEnqueueRange2D(kernel, res_xsize, res_ysize);
  LOG_CL_RESULT(err);
}

void clEdgeDetectorLowFreqEx(cl_mem block_diff_ac /*in,out*/, const ocl_channels& rgb, const ocl_channels& rgb2,
                             const unsigned int xsize, const unsigned int ysize, const unsigned int step) {
  size_t channel_size = xsize * ysize * sizeof(float);

  static const float kSigma = 14;
  ocl_args_d_t& ocl = getOcl();
  ocl_channels rgb_blured = ocl.allocMemChannelsUninit(channel_size);  // fully written by the blurs
  ocl_channels rgb2_blured = ocl.allocMemChannelsUninit(channel_size);

  for (int i = 0; i < 3; i++) {
    clBlurEx(rgb.ch[i], xsize, ysize, kSigma, 0.0, rgb_blured.ch[i]);
    clBlurEx(rgb2.ch[i], xsize, ysize, kSigma, 0.0, rgb2_blured.ch[i]);
  }

  const size_t res_xsize = (xsize + step - 1) / step;
  const size_t res_ysize = (ysize + step - 1) / step;

  cl_kernel kernel = ocl.kernel[KERNEL_EDGEDETECTORLOWFREQ];
  clSetKernelArgEx(kernel, &block_diff_ac, &res_xsize, &res_ysize, &rgb_blured.r, &rgb_blured.g, &rgb_blured.b,
                   &rgb2_blured.r, &rgb2_blured.g, &rgb2_blured.b, &xsize, &ysize, &step);

  cl_int err = clEnqueueRange2D(kernel, res_xsize, res_ysize);
  LOG_CL_RESULT(err);

  ocl.releaseMemChannels(rgb_blured);
  ocl.releaseMemChannels(rgb2_blured);
}

void clDiffPrecomputeEx(ocl_channels& mask /*out*/, const ocl_channels& xyb0, const ocl_channels& xyb1,
                        const unsigned int xsize, const unsigned int ysize) {
  ocl_args_d_t& ocl = getOcl();

  cl_kernel kernel = ocl.kernel[KERNEL_DIFFPRECOMPUTE];
  clSetKernelArgEx(kernel, &mask.x, &mask.y, &mask.b, &xsize, &ysize, &xyb0.x, &xyb0.y, &xyb0.b, &xyb1.x, &xyb1.y,
                   &xyb1.b);

  cl_int err = clEnqueueRange2D(kernel, xsize, ysize);
  LOG_CL_RESULT(err);
}

void clScaleImageEx(cl_mem img /*in, out*/, unsigned int size, float w) {
  ocl_args_d_t& ocl = getOcl();

  cl_kernel kernel = ocl.kernel[KERNEL_SCALEIMAGE];
  clSetKernelArgEx(kernel, &img, &size, &w);

  cl_int err = clEnqueueRange1D(kernel, size);
  LOG_CL_RESULT(err);
}

void clAverage5x5Ex(cl_mem img /*in,out*/, const unsigned int xsize, const unsigned int ysize) {
  if (xsize < 4 || ysize < 4) {
    // TODO: Make this work for small dimensions as well.
    return;
  }

  ocl_args_d_t& ocl = getOcl();

  size_t len = xsize * ysize * sizeof(float);
  cl_mem img_org = ocl.allocMemUninit(len);  // full copy below

  LOG_CL_RESULT(clEnqueueCopyBuffer(ocl.commandQueue, img, img_org, 0, 0, len, 0, NULL, NULL));

  cl_kernel kernel = ocl.kernel[KERNEL_AVERAGE5X5];
  clSetKernelArgEx(kernel, &img, &xsize, &ysize, &img_org);

  cl_int err = clEnqueueRange2D(kernel, xsize, ysize);
  LOG_CL_RESULT(err);

  ocl.releaseMem(img_org);
}

void clMinSquareValEx(cl_mem img /*in,out*/, const unsigned int xsize, const unsigned int ysize,
                      const unsigned int square_size, const unsigned int offset) {
  ocl_args_d_t& ocl = getOcl();

  size_t result_size = sizeof(cl_float) * xsize * ysize;
  cl_mem result = ocl.allocMemUninit(result_size);  // every pixel written

  cl_kernel kernel = ocl.kernel[KERNEL_MINSQUAREVAL];
  clSetKernelArgEx(kernel, &result, &xsize, &ysize, &img, &square_size, &offset);

  cl_int err = clEnqueueRange2D(kernel, xsize, ysize);
  LOG_CL_RESULT(err);
  err = clEnqueueCopyBuffer(ocl.commandQueue, result, img, 0, 0, sizeof(cl_float) * xsize * ysize, 0, NULL, NULL);
  LOG_CL_RESULT(err);
  ocl.releaseMem(result);
}

static void MakeMask(float extmul, float extoff, float mul, float offset, float scaler, float* result) {
  for (size_t i = 0; i < 512; ++i) {
    const float c = mul / ((0.01 * scaler * i) + offset);
    result[i] = 1.0 + extmul * (c + extoff);
    result[i] *= result[i];
  }
}

static const float kInternalGoodQualityThreshold = 14.921561160295326;
static const float kGlobalScale = 1.0 / kInternalGoodQualityThreshold;

void clDoMask(ocl_channels mask /*in, out*/, ocl_channels mask_dc /*in, out*/, size_t xsize, size_t ysize) {
  ocl_args_d_t& ocl = getOcl();

  float extmul = 0.975741017749;
  float extoff = -4.25328244168;
  float offset = 0.454909521427;
  float scaler = 0.0738288224836;
  float mul = 20.8029176447;
  static float lut_x[512];
  static bool lutx_init = false;
  if (!lutx_init) {
    lutx_init = true;
    MakeMask(extmul, extoff, mul, offset, scaler, lut_x);
  }

  extmul = 0.373995618954;
  extoff = 1.5307267433;
  offset = 0.911952641929;
  scaler = 1.1731667845;
  mul = 16.2447033988;
  static float lut_y[512];
  static bool luty_init = false;
  if (!luty_init) {
    luty_init = true;
    MakeMask(extmul, extoff, mul, offset, scaler, lut_y);
  }

  extmul = 0.61582234137;
  extoff = -4.25376118646;
  offset = 1.05105070921;
  scaler = 0.47434643535;
  mul = 31.1444967089;
  static float lut_b[512];
  static bool lutb_init = false;
  if (!lutb_init) {
    lutb_init = true;
    MakeMask(extmul, extoff, mul, offset, scaler, lut_b);
  }

  extmul = 1.79116943438;
  extoff = -3.86797479189;
  offset = 0.670960225853;
  scaler = 0.486575865525;
  mul = 20.4563479139;
  static float lut_dcx[512];
  static bool lutdcx_init = false;
  if (!lutdcx_init) {
    lutdcx_init = true;
    MakeMask(extmul, extoff, mul, offset, scaler, lut_dcx);
  }

  extmul = 0.212223514236;
  extoff = -3.65647120524;
  offset = 1.73396799447;
  scaler = 0.170392660501;
  mul = 21.6566724788;
  static float lut_dcy[512];
  static bool lutdcy_init = false;
  if (!lutdcy_init) {
    lutdcy_init = true;
    MakeMask(extmul, extoff, mul, offset, scaler, lut_dcy);
  }

  extmul = 0.349376011816;
  extoff = -0.894711072781;
  offset = 0.901647926679;
  scaler = 0.380086095024;
  mul = 18.0373825149;
  static float lut_dcb[512];
  static bool lutdcb_init = false;
  if (!lutdcb_init) {
    lutdcb_init = true;
    MakeMask(extmul, extoff, mul, offset, scaler, lut_dcb);
  }

  // Constant LUTs: uploaded once, kept on the device.
  ocl_channels xyb, xyb_dc;
  xyb.ch[0] = ocl.getMaskLut(0, lut_x);
  xyb.ch[1] = ocl.getMaskLut(1, lut_y);
  xyb.ch[2] = ocl.getMaskLut(2, lut_b);
  xyb_dc.ch[0] = ocl.getMaskLut(3, lut_dcx);
  xyb_dc.ch[1] = ocl.getMaskLut(4, lut_dcy);
  xyb_dc.ch[2] = ocl.getMaskLut(5, lut_dcb);

  cl_kernel kernel = ocl.kernel[KERNEL_DOMASK];
  clSetKernelArgEx(kernel, &mask.r, &mask.g, &mask.b, &xsize, &ysize, &mask_dc.r, &mask_dc.g, &mask_dc.b, &xyb.x,
                   &xyb.y, &xyb.b, &xyb_dc.x, &xyb_dc.y, &xyb_dc.b);

  cl_int err = clEnqueueRange2D(kernel, xsize, ysize);
  LOG_CL_RESULT(err);
}

void clMaskEx(ocl_channels mask /*out*/, ocl_channels mask_dc /*out*/, const ocl_channels& rgb,
              const ocl_channels& rgb2, const unsigned int xsize, const unsigned int ysize) {
  clDiffPrecomputeEx(mask, rgb, rgb2, xsize, ysize);
  for (int i = 0; i < 3; i++) {
    clAverage5x5Ex(mask.ch[i], xsize, ysize);
    clMinSquareValEx(mask.ch[i], xsize, ysize, 4, 0);

    static const float sigma[3] = {
        9.65781083553,
        14.2644604355,
        4.53358927369,
    };

    clBlurEx(mask.ch[i], xsize, ysize, sigma[i], 0.0, mask.ch[i]);
  }

  clDoMask(mask, mask_dc, xsize, ysize);

  for (int i = 0; i < 3; i++) {
    clScaleImageEx(mask.ch[i], xsize * ysize, kGlobalScale * kGlobalScale);
    clScaleImageEx(mask_dc.ch[i], xsize * ysize, kGlobalScale * kGlobalScale);
  }
}

void clCombineChannelsEx(cl_mem result /*out*/, const ocl_channels& mask, const ocl_channels& mask_dc,
                         const unsigned int xsize, const unsigned int ysize, const cl_mem block_diff_dc,
                         const cl_mem block_diff_ac, const cl_mem edge_detector_map, const unsigned int res_xsize,
                         const unsigned int step) {
  ocl_args_d_t& ocl = getOcl();

  const size_t work_xsize = ((xsize - 8 + step) + step - 1) / step;
  const size_t work_ysize = ((ysize - 8 + step) + step - 1) / step;

  cl_kernel kernel = ocl.kernel[KERNEL_COMBINECHANNELS];
  clSetKernelArgEx(kernel, &result, &mask.r, &mask.g, &mask.b, &mask_dc.r, &mask_dc.g, &mask_dc.b, &xsize, &ysize,
                   &block_diff_dc, &block_diff_ac, &edge_detector_map, &res_xsize, &step);

  cl_int err = clEnqueueRange2D(kernel, work_xsize, work_ysize);
  LOG_CL_RESULT(err);
}

void clUpsampleSquareRootEx(cl_mem diffmap, const unsigned int xsize, const unsigned int ysize, const int step) {
  ocl_args_d_t& ocl = getOcl();

  size_t diffmap_out_size = xsize * ysize * sizeof(float);
  cl_mem diffmap_out = ocl.allocMem(diffmap_out_size);

  cl_kernel kernel = ocl.kernel[KERNEL_UPSAMPLESQUAREROOT];
  const size_t res_xsize = (xsize + step - 1) / step;
  const size_t res_ysize = (ysize + step - 1) / step;
  const int res_xsize_arg = (int)res_xsize, res_ysize_arg = (int)res_ysize;
  clSetKernelArgEx(kernel, &diffmap_out, &diffmap, &xsize, &ysize, &step, &res_xsize_arg, &res_ysize_arg);

  cl_int err = clEnqueueRange2D(kernel, res_xsize, res_ysize);
  LOG_CL_RESULT(err);
  err = clEnqueueCopyBuffer(ocl.commandQueue, diffmap_out, diffmap, 0, 0, xsize * ysize * sizeof(float), 0, NULL, NULL);
  LOG_CL_RESULT(err);

  ocl.releaseMem(diffmap_out);
}

void clRemoveBorderEx(cl_mem out, const cl_mem in, const unsigned int xsize, const unsigned int ysize, const int step) {
  ocl_args_d_t& ocl = getOcl();

  cl_int cls = 8 - step;
  cl_int cls2 = (8 - step) / 2;

  int out_xsize = xsize - cls;
  int out_ysize = ysize - cls;

  cl_kernel kernel = ocl.kernel[KERNEL_REMOVEBORDER];
  clSetKernelArgEx(kernel, &out, &out_xsize, &out_ysize, &in, &cls, &cls2);

  cl_int err = clEnqueueRange2D(kernel, out_xsize, out_ysize);
  LOG_CL_RESULT(err);
}

void clAddBorderEx(cl_mem out, const unsigned int xsize, const unsigned int ysize, const int step, const cl_mem in) {
  ocl_args_d_t& ocl = getOcl();

  cl_int cls = 8 - step;
  cl_int cls2 = (8 - step) / 2;
  cl_kernel kernel = ocl.kernel[KERNEL_ADDBORDER];
  clSetKernelArgEx(kernel, &out, &xsize, &ysize, &cls, &cls2, &in);

  cl_int err = clEnqueueRange2D(kernel, xsize, ysize);
  LOG_CL_RESULT(err);
}

void clCalculateDiffmapEx(cl_mem diffmap /*in,out*/, const unsigned int xsize, const unsigned int ysize,
                          const int step) {
  clUpsampleSquareRootEx(diffmap, xsize, ysize, step);

  static const float kSigma = 8.8510880283;
  static const float mul1 = 24.8235314874;
  static const float scale = 1.0 / (1.0 + mul1);

  const int s = 8 - step;
  const int s2 = (8 - step) / 2;

  ocl_args_d_t& ocl = getOcl();
  size_t blurred_size = (xsize - s) * (ysize - s) * sizeof(float);
  cl_mem blurred = ocl.allocMemUninit(blurred_size);  // RemoveBorder writes every element
  clRemoveBorderEx(blurred, diffmap, xsize, ysize, step);

  static const float border_ratio = 0.03027655136;
  clBlurEx(blurred, xsize - s, ysize - s, kSigma, border_ratio, blurred);

  clAddBorderEx(diffmap, xsize, ysize, step, blurred);
  clScaleImageEx(diffmap, xsize * ysize, scale);

  ocl.releaseMem(blurred);
}

void clCopyFromJpegComponent(coeff_t* output_batch /*in,out*/, uint8_t* output_idct /*out*/,
                             const coeff_t* jpeg_batch /*in*/, const int* quant, const int jpeg_block_width,
                             const int jpeg_block_height, const int output_block_width, const int output_block_height,
                             const int output_width, const int output_height) {
  using namespace guetzli;

  ocl_args_d_t& ocl = getOcl();

  int src_block_count = jpeg_block_width * jpeg_block_height;
  size_t src_coeff_size = src_block_count * sizeof(::coeff_t) * kDCTBlockSize;
  cl_mem src_coeff = ocl.allocMem(src_coeff_size, jpeg_batch);

  size_t dst_coeff_size = output_block_width * output_block_height * sizeof(::coeff_t) * kDCTBlockSize;
  cl_mem dst_coeff = ocl.allocMem(dst_coeff_size, output_batch);

  size_t dst_idct_size = output_block_width * output_block_height * sizeof(uint8_t) * kDCTBlockSize;
  cl_mem dst_idct = ocl.allocMem(dst_idct_size, output_idct);

  size_t src_quant_size = kDCTBlockSize * sizeof(int);
  cl_mem src_quant = ocl.allocMem(src_quant_size, quant);

  cl_kernel kernel = ocl.kernel[KERNEL_COPYFROMJPEGCOMPONENT];
  clSetKernelArgEx(kernel, &dst_coeff, &dst_idct, &src_coeff, &src_quant, &jpeg_block_width, &jpeg_block_height,
                   &output_block_width, &output_block_height, &output_width, &output_height);

  cl_int err = clEnqueueRange2D(kernel, output_block_width, output_block_height);
  LOG_CL_RESULT(err);

  err = clEnqueueReadBuffer(ocl.commandQueue, dst_coeff, false, 0, dst_coeff_size, output_batch, 0, NULL, NULL);
  err = clEnqueueReadBuffer(ocl.commandQueue, dst_idct, false, 0, dst_idct_size, output_idct, 0, NULL, NULL);
  err = clFinish(ocl.commandQueue);

  ocl.releaseMem(src_coeff);
  ocl.releaseMem(dst_coeff);
  ocl.releaseMem(dst_idct);
  ocl.releaseMem(src_quant);
}

void clApplyGlobalQuantization(coeff_t* output_batch /*in,out*/, uchar* output_idct /*out*/, uchar* output_bool /*out*/,
                               const int* q /*in*/, const int block_width, const int block_height) {
  using namespace guetzli;

  ocl_args_d_t& ocl = getOcl();

  size_t dst_coeff_size = block_width * block_height * sizeof(::coeff_t) * kDCTBlockSize;
  cl_mem dst_coeff = ocl.allocMem(dst_coeff_size, output_batch);

  size_t dst_idct_size = block_width * block_height * sizeof(uint8_t) * kDCTBlockSize;
  cl_mem dst_idct = ocl.allocMem(dst_idct_size, output_idct);

  size_t dst_bool_size = block_width * block_height * sizeof(uchar);
  cl_mem dst_bool = ocl.allocMem(dst_bool_size, output_bool);

  size_t src_q_size = kDCTBlockSize * sizeof(int);
  cl_mem src_q = ocl.allocMem(src_q_size, q);

  cl_kernel kernel = ocl.kernel[KERNEL_APPLYGLOBALQUANTIZATION];
  clSetKernelArgEx(kernel, &dst_coeff, &dst_idct, &dst_bool, &src_q, &block_width, &block_height);

  cl_int err = clEnqueueRange2D(kernel, block_width, block_height);
  LOG_CL_RESULT(err);

  err = clEnqueueReadBuffer(ocl.commandQueue, dst_coeff, false, 0, dst_coeff_size, output_batch, 0, NULL, NULL);
  err = clEnqueueReadBuffer(ocl.commandQueue, dst_idct, false, 0, dst_idct_size, output_idct, 0, NULL, NULL);
  err = clEnqueueReadBuffer(ocl.commandQueue, dst_bool, false, 0, dst_bool_size, output_bool, 0, NULL, NULL);
  err = clFinish(ocl.commandQueue);

  ocl.releaseMem(dst_bool);
  ocl.releaseMem(dst_coeff);
  ocl.releaseMem(dst_idct);
  ocl.releaseMem(src_q);
}

// Enqueues the component -> 8-bit pixel conversion of a region plus the YCbCr -> RGB
// transform into the device buffer out (xsize * ysize * 3 bytes, every byte written).
// pixels[c] holds the device copy of components[c].pixels().
static void clComponentsToPixelsDevice(cl_mem out, const cl_mem pixels[3], const int xmin, const int ymin,
                                       const int xsize, const int ysize,
                                       const guetzli::OutputImageComponent* const components[3]) {
  ocl_args_d_t& ocl = getOcl();

  for (int c = 0; c < 3; ++c) {
    const int width = components[c]->width();
    const int height = components[c]->height();

    const int yend1 = ymin + ysize;
    const int yend0 = std::min(yend1, height);
    const int xend1 = xmin + xsize;
    const int xend0 = std::min(xend1, width);

    {
      cl_kernel kernel = ocl.kernel[KERNEL_COMPONENTSTOPIXELS];
      clSetKernelArgEx(kernel, &out, &xmin, &ymin, &xsize, &ysize, &c, &pixels[c], &width, &height);

      cl_int err = clEnqueueRange2D(kernel, xend0 - xmin, yend0 - ymin);
      LOG_CL_RESULT(err);
    }

    if (xend1 - xend0 > 0) {
      cl_kernel kernel = ocl.kernel[KERNEL_COMPONENTSTOPIXELS_EX1];
      clSetKernelArgEx(kernel, &out, &xmin, &ymin, &xsize, &ysize, &c, &width, &height);

      cl_int err = clEnqueueRange2D(kernel, xend1 - xend0, yend0 - ymin);
      LOG_CL_RESULT(err);
    }

    if (yend1 - yend0 > 0) {
      cl_kernel kernel = ocl.kernel[KERNEL_COMPONENTSTOPIXELS_EX2];
      clSetKernelArgEx(kernel, &out, &xmin, &ymin, &xsize, &ysize, &c, &width, &height);

      cl_int err = clEnqueueRange2D(kernel, xsize, yend1 - yend0);
      LOG_CL_RESULT(err);
    }
  }

  cl_kernel kernel = ocl.kernel[KERNEL_COLORTRANSFORMYCBCRTORGB];
  const int pixel_count = xsize * ysize;
  clSetKernelArgEx(kernel, &out, &pixel_count);

  cl_int err = clEnqueueRange1D(kernel, pixel_count);
  LOG_CL_RESULT(err);
}

void clComponentsToPixels(uchar* rgb, /*out*/
                          const int xmin, const int ymin, const int xsize, const int ysize,
                          const std::vector<guetzli::OutputImageComponent>& components /*in*/) {
  using namespace guetzli;

  ocl_args_d_t& ocl = getOcl();

  const OutputImageComponent* comps[3] = {&components[0], &components[1], &components[2]};
  cl_mem pixels[3];
  for (int c = 0; c < 3; ++c) {
    const size_t pixels_size = (size_t)comps[c]->width() * comps[c]->height() * sizeof(ushort);
    pixels[c] = ocl.allocMem(pixels_size, comps[c]->pixels());
  }

  const size_t out_size = (size_t)xsize * ysize * 3 * sizeof(uchar);
  cl_mem cl_out = ocl.allocMemUninit(out_size);  // every byte is written

  clComponentsToPixelsDevice(cl_out, pixels, xmin, ymin, xsize, ysize, comps);

  LOG_CL_RESULT(clEnqueueReadBuffer(ocl.commandQueue, cl_out, CL_TRUE, 0, out_size, rgb, 0, NULL, NULL));
  ocl.releaseMem(cl_out);
  for (int c = 0; c < 3; ++c) ocl.releaseMem(pixels[c]);
}

// Per-image state of the OpenCL comparator's Compare() (backend/opencl).
struct ClCompareContext {
  int xsize;
  int ysize;
  ocl_channels ref;  // opsin-dynamics image of the original; never modified
};

ClCompareContext* clCompareCreate(const float* r, const float* g, const float* b, int xsize, int ysize) {
  ocl_args_d_t& ocl = getOcl();
  const size_t channel_size = (size_t)xsize * ysize * sizeof(float);

  ClCompareContext* ctx = new ClCompareContext;
  ctx->xsize = xsize;
  ctx->ysize = ysize;
  try {
    ctx->ref = ocl.allocMemChannels(channel_size, r, g, b);
  } catch (...) {
    delete ctx;
    throw;
  }
  return ctx;
}

void clCompareDestroy(ClCompareContext* ctx) {
  if (!ctx) return;
  try {
    getOcl().releaseMemChannels(ctx->ref);
  } catch (...) {
    // Called from destructors (also while unwinding a GpuError): never throw.
  }
  delete ctx;
}

void clCompare(ClCompareContext* ctx, const guetzli::OutputImage& img, int step, float* distmap /*out*/) {
  using namespace guetzli;

  ocl_args_d_t& ocl = getOcl();

  const int xsize = ctx->xsize;
  const int ysize = ctx->ysize;
  const int size = xsize * ysize;
  const size_t channel_size = (size_t)size * sizeof(float);

  // 1. Output image -> linear RGB, entirely on the device (was: download 8-bit RGB,
  //    CPU LUT, re-upload three float planes).
  const OutputImageComponent* comps[3] = {&img.component(0), &img.component(1), &img.component(2)};
  cl_mem pixels[3];
  for (int c = 0; c < 3; ++c) {
    const size_t pixels_size = (size_t)comps[c]->width() * comps[c]->height() * sizeof(ushort);
    pixels[c] = ocl.allocMemUninit(pixels_size);
    // Non-blocking: img outlives this call, and the blocking read at the end
    // completes all queued work before we return.
    LOG_CL_RESULT(
        clEnqueueWriteBuffer(ocl.commandQueue, pixels[c], CL_FALSE, 0, pixels_size, comps[c]->pixels(), 0, NULL, NULL));
  }

  cl_mem srgb = ocl.allocMemUninit((size_t)size * 3);
  clComponentsToPixelsDevice(srgb, pixels, 0, 0, xsize, ysize, comps);

  ocl_channels xyb1 = ocl.allocMemChannelsUninit(channel_size);
  {
    cl_mem lut = ocl.getSrgbToLinearLut();
    cl_kernel kernel = ocl.kernel[KERNEL_SRGBTOLINEAR];
    clSetKernelArgEx(kernel, &srgb, &xyb1.r, &xyb1.g, &xyb1.b, &lut, &size);
    LOG_CL_RESULT(clEnqueueRange1D(kernel, size));
  }
  ocl.releaseMem(srgb);
  for (int c = 0; c < 3; ++c) ocl.releaseMem(pixels[c]);

  // 2. Reference stays resident; the diffmap mutates its input, so work on a copy.
  ocl_channels xyb0 = ocl.allocMemChannelsUninit(channel_size);
  for (int c = 0; c < 3; ++c) {
    LOG_CL_RESULT(clEnqueueCopyBuffer(ocl.commandQueue, ctx->ref.ch[c], xyb0.ch[c], 0, 0, channel_size, 0, NULL, NULL));
  }

  // Zero-filled: CombineChannels / UpsampleSquareRoot leave the borders untouched.
  cl_mem mem_result = ocl.allocMem(channel_size);

  clOpsinDynamicsImageEx(xyb1, xsize, ysize);
  clDiffmapOpsinDynamicsImageEx(mem_result, xyb0, xyb1, xsize, ysize, step);

  LOG_CL_RESULT(clEnqueueReadBuffer(ocl.commandQueue, mem_result, CL_TRUE, 0, channel_size, distmap, 0, NULL, NULL));

  ocl.releaseMem(mem_result);
  ocl.releaseMemChannels(xyb0);
  ocl.releaseMemChannels(xyb1);
}

#endif

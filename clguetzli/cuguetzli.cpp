/*
* CUDA edition implementation of guetzli.
*
* Author: strongtu@tencent.com
*         ianhuang@tencent.com
*         chriskzhou@tencent.com
*		  stephendeng@tencent.com
*/
#include "cuguetzli.h"
#include <algorithm>
#include "ocu.h"

#ifdef __USE_CUDA__

#define cuFinish cuStreamSynchronize
#define BLOCK_SIZE 96
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define BLOCK_COUNT_X(size, x)    ((size + x - 1) / x)
#define BLOCK_COUNT_Y(size, y)    ((size + y - 1) / y)

void calcBlock(int width, int height, int &x, int &y)
{
	unsigned long long thread = 0;
	for (int i = BLOCK_SIZE, j = 1; i > 1; i /= 2, j *= 2)
	{
		unsigned long long temp = BLOCK_COUNT_X(width, i) * BLOCK_COUNT_Y(height, j) * i * j;
		if (thread > temp || thread == 0) {
			thread = temp;
			x = i;
			y = j;
		}
	}
}

void cuOpsinDynamicsImage(float *r, float *g, float *b, const unsigned int xsize, const unsigned int ysize)
{
    unsigned int channel_size = xsize * ysize * sizeof(float);

    ocu_args_d_t &ocu = getOcu();
    ocu_channels rgb = ocu.allocMemChannels(channel_size, r, g, b);

    cuOpsinDynamicsImageEx(rgb, xsize, ysize);

    cuMemcpyDtoHAsync(r, rgb.r, channel_size, ocu.commandQueue);
    cuMemcpyDtoHAsync(g, rgb.g, channel_size, ocu.commandQueue);
	cuMemcpyDtoHAsync(b, rgb.b, channel_size, ocu.commandQueue);
    cuFinish(ocu.commandQueue);

    ocu.releaseMemChannels(rgb);
}

void cuDiffmapOpsinDynamicsImage(
    float* result,
    const float* r, const float* g, const float* b,
    const float* r2, const float* g2, const float* b2,
    const unsigned int xsize, const unsigned int ysize,
    const unsigned int step)
{
    unsigned int channel_size = xsize * ysize * sizeof(float);

    ocu_args_d_t &ocu = getOcu();
    ocu_channels xyb0 = ocu.allocMemChannels(channel_size, r, g, b);
    ocu_channels xyb1 = ocu.allocMemChannels(channel_size, r2, g2, b2);

    cu_mem mem_result = ocu.allocMem(channel_size, result);

    cuDiffmapOpsinDynamicsImageEx(mem_result, xyb0, xyb1, xsize, ysize, step);

    cuMemcpyDtoH(result, mem_result, channel_size);

    ocu.releaseMemChannels(xyb1);
    ocu.releaseMemChannels(xyb0);

    ocu.releaseMem(mem_result);
}

void cuComputeBlockZeroingOrder(
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
    if (factor <= 0) {
        LogInfo("Error: factor is %d, must be > 0. Skipping cuComputeBlockZeroingOrder.\n", factor);
        return;
    }
    
    const int block8_width = (image_width + 8 - 1) / 8;
    const int block8_height = (image_height + 8 - 1) / 8;
    const int blockf_width = (image_width + 8 * factor - 1) / (8 * factor);
    const int blockf_height = (image_height + 8 * factor - 1) / (8 * factor);

    using namespace guetzli;

    ocu_args_d_t &ocu = getOcu();

    cu_mem mem_orig_coeff[3];
    cu_mem mem_mayout_coeff[3];
    cu_mem mem_mayout_pixel[3];
    for (int c = 0; c < 3; c++)
    {
        int block_count = orig_channel[c].block_width * orig_channel[c].block_height;
        mem_orig_coeff[c] = ocu.allocMem(block_count * sizeof(::coeff_t) * kDCTBlockSize, orig_channel[c].coeff);

        block_count = mayout_channel[c].block_width * mayout_channel[c].block_height;
        mem_mayout_coeff[c] = ocu.allocMem(block_count * sizeof(::coeff_t) * kDCTBlockSize, mayout_channel[c].coeff);

        mem_mayout_pixel[c] = ocu.allocMem(image_width * image_height * sizeof(uint16_t), mayout_channel[c].pixel);
    }
    cu_mem mem_orig_image = ocu.allocMem(sizeof(float) * 3 * kDCTBlockSize * block8_width * block8_height, orig_image_batch);
    cu_mem mem_mask_scale = ocu.allocMem(sizeof(float) * 3 * block8_width * block8_height, mask_scale);

    int output_order_batch_size = sizeof(CoeffData) * 3 * kDCTBlockSize * blockf_width * blockf_height;
    cu_mem mem_output_order_batch = ocu.allocMem(output_order_batch_size, output_order_batch);

    CUfunction kernel = ocu.kernel[KERNEL_COMPUTEBLOCKZEROINGORDER];
    const void *args[] = { &mem_orig_coeff[0], &mem_orig_coeff[1], &mem_orig_coeff[2],
        &mem_orig_image, &mem_mask_scale,
        &blockf_width, &blockf_height,
        &image_width, &image_height,
        &mem_mayout_coeff[0], &mem_mayout_coeff[1], &mem_mayout_coeff[2],
        &mem_mayout_pixel[0], &mem_mayout_pixel[1], &mem_mayout_pixel[2],
        &mayout_channel[0], &mayout_channel[1], &mayout_channel[2],
        &factor,
        &comp_mask,
        &BlockErrorLimit,
        &mem_output_order_batch };

	int x = BLOCK_SIZE_X, y = BLOCK_SIZE_Y;
	calcBlock(blockf_width, blockf_height, x, y);
    CUresult err = cuLaunchKernel(kernel,
        BLOCK_COUNT_X(blockf_width, x), BLOCK_COUNT_Y(blockf_height, y), 1,
        x, y, 1,
        0,
        ocu.commandQueue, (void**)args, NULL);
    LOG_CU_RESULT(err);

    err = cuFinish(ocu.commandQueue);
	LOG_CU_RESULT(err);

    cuMemcpyDtoH(output_order_batch, mem_output_order_batch, output_order_batch_size);

    for (int c = 0; c < 3; c++)
    {
        ocu.releaseMem(mem_orig_coeff[c]);
        ocu.releaseMem(mem_mayout_coeff[c]);
        ocu.releaseMem(mem_mayout_pixel[c]);
    }

    ocu.releaseMem(mem_orig_image);
    ocu.releaseMem(mem_mask_scale);
    ocu.releaseMem(mem_output_order_batch);
}

void cuMask(
    float* mask_r, float* mask_g, float* mask_b,
    float* maskdc_r, float* maskdc_g, float* maskdc_b,
    const unsigned int xsize, const unsigned int ysize,
    const float* r, const float* g, const float* b,
    const float* r2, const float* g2, const float* b2)
{
    ocu_args_d_t &ocu = getOcu();

    unsigned int channel_size = xsize * ysize * sizeof(float);

    ocu_channels rgb = ocu.allocMemChannels(channel_size, r, g, b);
    ocu_channels rgb2 = ocu.allocMemChannels(channel_size, r2, g2, b2);
    ocu_channels mask = ocu.allocMemChannels(channel_size);
    ocu_channels mask_dc = ocu.allocMemChannels(channel_size);

    cuMaskEx(mask, mask_dc, rgb, rgb2, xsize, ysize);

    cuMemcpyDtoHAsync(mask_r, mask.r, channel_size, ocu.commandQueue);
    cuMemcpyDtoHAsync(mask_g, mask.g, channel_size, ocu.commandQueue);
    cuMemcpyDtoHAsync(mask_b, mask.b, channel_size, ocu.commandQueue);
    cuMemcpyDtoHAsync(maskdc_r, mask_dc.r, channel_size, ocu.commandQueue);
    cuMemcpyDtoHAsync(maskdc_g, mask_dc.g, channel_size, ocu.commandQueue);
    cuMemcpyDtoHAsync(maskdc_b, mask_dc.b, channel_size, ocu.commandQueue);
    cuFinish(ocu.commandQueue);

    ocu.releaseMemChannels(rgb);
    ocu.releaseMemChannels(rgb2);
    ocu.releaseMemChannels(mask);
    ocu.releaseMemChannels(mask_dc);
}

void cuDiffmapOpsinDynamicsImageEx(
    cu_mem result,
    ocu_channels xyb0,
    ocu_channels xyb1,
    const unsigned int xsize, const unsigned int ysize,
    const unsigned int step)
{
    const unsigned int res_xsize = (xsize + step - 1) / step;
    const unsigned int res_ysize = (ysize + step - 1) / step;

    unsigned int channel_size = xsize * ysize * sizeof(float);
    unsigned int channel_step_size = res_xsize * res_ysize * sizeof(float);

    ocu_args_d_t &ocu = getOcu();
 
    cu_mem edge_detector_map = ocu.allocMem(3 * channel_step_size);
    cu_mem block_diff_dc = ocu.allocMem(3 * channel_step_size);
    cu_mem block_diff_ac = ocu.allocMem(3 * channel_step_size);

    cuMaskHighIntensityChangeEx(xyb0, xyb1, xsize, ysize);

    cuEdgeDetectorMapEx(edge_detector_map, xyb0, xyb1, xsize, ysize, step);
    cuBlockDiffMapEx(block_diff_dc, block_diff_ac, xyb0, xyb1, xsize, ysize, step);
    cuEdgeDetectorLowFreqEx(block_diff_ac, xyb0, xyb1, xsize, ysize, step);
    {
        ocu_channels mask = ocu.allocMemChannels(channel_size);
        ocu_channels mask_dc = ocu.allocMemChannels(channel_size);
        cuMaskEx(mask, mask_dc, xyb0, xyb1, xsize, ysize);
        cuCombineChannelsEx(result, mask, mask_dc, xsize, ysize, block_diff_dc, block_diff_ac, edge_detector_map, res_xsize, step);

        ocu.releaseMemChannels(mask);
        ocu.releaseMemChannels(mask_dc);
    }

    cuCalculateDiffmapEx(result, xsize, ysize, step);

    ocu.releaseMem(edge_detector_map);
    ocu.releaseMem(block_diff_dc);
    ocu.releaseMem(block_diff_ac);
}

void cuConvolutionEx(
    cu_mem result/*out*/,
    const cu_mem inp, unsigned int xsize, unsigned int ysize,
    const cu_mem multipliers, unsigned int len,
    int xstep, int offset, float border_ratio)
{
    ocu_args_d_t &ocu = getOcu();

    unsigned int oxsize = (xsize + xstep - 1) / xstep;

	CUfunction kernel = ocu.kernel[KERNEL_CONVOLUTION];
    const void *args[] = { &result, &inp, &xsize, &multipliers, &len, &xstep, &offset, &border_ratio };

    CUresult err = cuLaunchKernel(kernel,
        oxsize, ysize, 1,
        1, 1, 1,
        0,
        ocu.commandQueue, (void**)args, NULL);
	LOG_CU_RESULT(err);
    err = cuFinish(ocu.commandQueue);
	LOG_CU_RESULT(err);
}


void cuConvolutionXEx(
    cu_mem result/*out*/,
    const cu_mem inp, unsigned int xsize, unsigned int ysize,
    const cu_mem multipliers, unsigned int len,
    int xstep, int offset, float border_ratio)
{
    ocu_args_d_t &ocu = getOcu();

	CUfunction kernel = ocu.kernel[KERNEL_CONVOLUTIONX];
    const void *args[] = { &result, &xsize, &ysize, &inp, &multipliers, &len, &xstep, &offset, &border_ratio };

	unsigned int x_count = (xsize + xstep - 1) / xstep;
	int x = BLOCK_SIZE_X, y = BLOCK_SIZE_Y;
	calcBlock(x_count, ysize, x, y);
    CUresult err = cuLaunchKernel(kernel,
        BLOCK_COUNT_X(x_count, x), BLOCK_COUNT_Y(ysize, y), 1,
        x, y, 1,
        0,
        ocu.commandQueue, (void**)args, NULL);
	LOG_CU_RESULT(err);
    err = cuFinish(ocu.commandQueue);
	LOG_CU_RESULT(err);
}

void cuConvolutionYEx(
    cu_mem result/*out*/,
    const cu_mem inp, unsigned int xsize, unsigned int ysize,
    const cu_mem multipliers, unsigned int len,
    int xstep, int offset, float border_ratio)
{
    ocu_args_d_t &ocu = getOcu();

	CUfunction kernel = ocu.kernel[KERNEL_CONVOLUTIONY];
    const void *args[] = { &result, &xsize, &ysize, &inp, &multipliers, &len, &xstep, &offset, &border_ratio };

	unsigned int x_count = (xsize + xstep - 1) / xstep;
	unsigned int y_count = (ysize + xstep - 1) / xstep;
	int x = BLOCK_SIZE_X, y = BLOCK_SIZE_Y;
	calcBlock(x_count, y_count, x, y);
    CUresult err = cuLaunchKernel(kernel,
        BLOCK_COUNT_X(x_count, x), BLOCK_COUNT_Y(y_count, y), 1,
        x, y, 1,
        0,
        ocu.commandQueue, (void**)args, NULL);
	LOG_CU_RESULT(err);
    err = cuFinish(ocu.commandQueue);
	LOG_CU_RESULT(err);
}

void cuSquareSampleEx(
    cu_mem result/*out*/,
    const cu_mem image, unsigned int xsize, unsigned int ysize,
    unsigned int xstep, unsigned int ystep)
{
    ocu_args_d_t &ocu = getOcu();

	CUfunction kernel = ocu.kernel[KERNEL_SQUARESAMPLE];
    const void *args[] = { &result, &xsize, &ysize, &image, &xstep, &ystep };

	int x = BLOCK_SIZE_X, y = BLOCK_SIZE_Y;
	calcBlock(xsize, ysize, x, y);
    CUresult err = cuLaunchKernel(kernel,
        BLOCK_COUNT_X(xsize, x), BLOCK_COUNT_Y(ysize, y), 1,
        x, y, 1,
        0,
        ocu.commandQueue, (void**)args, NULL);
	LOG_CU_RESULT(err);
    err = cuFinish(ocu.commandQueue);
	LOG_CU_RESULT(err);
}

void cuBlurEx(cu_mem image/*out, opt*/, const unsigned int xsize, const unsigned int ysize,
    const float sigma, const float border_ratio,
    cu_mem result/*out, opt*/)
{
    float m = 2.25;  // Accuracy increases when m is increased.
    const float scaler = -1.0f / (2 * sigma * sigma);
    // For m = 9.0: exp(-scaler * diff * diff) < 2^ {-52}
    const int diff = std::max<int>(1, m * fabs(sigma));
    const int expn_size = 2 * diff + 1;
    std::vector<float> expn(expn_size);
    for (int i = -diff; i <= diff; ++i) {
        expn[i + diff] = static_cast<float>(exp(scaler * i * i));
    }

    const int xstep = std::max<int>(1, int(sigma / 3));

    ocu_args_d_t &ocu = getOcu();
    cu_mem mem_expn = ocu.allocMem(sizeof(float) * expn_size, expn.data());

    if (xstep > 1)
    {
        cu_mem m = ocu.allocMem(sizeof(float) * xsize * ysize);
        cuConvolutionXEx(m, image, xsize, ysize, mem_expn, expn_size, xstep, diff, border_ratio);
        cuConvolutionYEx(result ? result : image, m, xsize, ysize, mem_expn, expn_size, xstep, diff, border_ratio);
        cuSquareSampleEx(result ? result : image, result ? result : image, xsize, ysize, xstep, xstep);
        ocu.releaseMem(m);
    }
    else
    {
        cu_mem m = ocu.allocMem(sizeof(float) * xsize * ysize);
        cuConvolutionXEx(m, image, xsize, ysize, mem_expn, expn_size, xstep, diff, border_ratio);
        cuConvolutionYEx(result ? result : image, m, xsize, ysize, mem_expn, expn_size, xstep, diff, border_ratio);
        ocu.releaseMem(m);
    }

    ocu.releaseMem(mem_expn);
}

void cuOpsinDynamicsImageEx(ocu_channels &rgb, const unsigned int xsize, const unsigned int ysize)
{
    static const float kSigma = 1.1f;

    unsigned int channel_size = xsize * ysize * sizeof(float);

    ocu_args_d_t &ocu = getOcu();
    ocu_channels rgb_blurred = ocu.allocMemChannels(channel_size);

    const int size = xsize * ysize;

    cuBlurEx(rgb.r, xsize, ysize, kSigma, 0.0, rgb_blurred.r);
    cuBlurEx(rgb.g, xsize, ysize, kSigma, 0.0, rgb_blurred.g);
    cuBlurEx(rgb.b, xsize, ysize, kSigma, 0.0, rgb_blurred.b);

	CUfunction kernel = ocu.kernel[KERNEL_OPSINDYNAMICSIMAGE];
    const void *args[] = { &rgb.r, &rgb.g, &rgb.b, &size, &rgb_blurred.r, &rgb_blurred.g, &rgb_blurred.b };

    CUresult err = cuLaunchKernel(kernel,
		BLOCK_COUNT_X(size, BLOCK_SIZE), 1, 1,
        BLOCK_SIZE, 1, 1,
        //(size + 511) / 512, 1, 1,
        //512, 1, 1,
        0,
        ocu.commandQueue, (void**)args, NULL);
	LOG_CU_RESULT(err);
    err = cuFinish(ocu.commandQueue);
	LOG_CU_RESULT(err);

    ocu.releaseMemChannels(rgb_blurred);
}

void cuMaskHighIntensityChangeEx(
    ocu_channels &xyb0/*in,out*/,
    ocu_channels &xyb1/*in,out*/,
    const unsigned int xsize, const unsigned int ysize)
{
    unsigned int channel_size = xsize * ysize * sizeof(float);

    ocu_args_d_t &ocu = getOcu();

    ocu_channels c0 = ocu.allocMemChannels(channel_size);
    ocu_channels c1 = ocu.allocMemChannels(channel_size);

    cuMemcpyDtoDAsync(c0.r, xyb0.r, channel_size, ocu.commandQueue);
    cuMemcpyDtoDAsync(c0.g, xyb0.g, channel_size, ocu.commandQueue);
    cuMemcpyDtoDAsync(c0.b, xyb0.b, channel_size, ocu.commandQueue);
    cuMemcpyDtoDAsync(c1.r, xyb1.r, channel_size, ocu.commandQueue);
    cuMemcpyDtoDAsync(c1.g, xyb1.g, channel_size, ocu.commandQueue);
    cuMemcpyDtoDAsync(c1.b, xyb1.b, channel_size, ocu.commandQueue);
	cuFinish(ocu.commandQueue);

	CUfunction kernel = ocu.kernel[KERNEL_MASKHIGHINTENSITYCHANGE];
    const void *args[] = { 
		&xyb0.r, &xyb0.g, &xyb0.b,
        &xsize, &ysize,
        &xyb1.r, &xyb1.g, &xyb1.b,
        &c0.r, &c0.g, &c0.b,
        &c1.r, &c1.g, &c1.b };

	int x = BLOCK_SIZE_X, y = BLOCK_SIZE_Y;
	calcBlock(xsize, ysize, x, y);
    CUresult err = cuLaunchKernel(kernel,
        BLOCK_COUNT_X(xsize, x), BLOCK_COUNT_Y(ysize, y), 1,
        x, y, 1,
        0,
        ocu.commandQueue, (void**)args, NULL);
	LOG_CU_RESULT(err);
    err = cuFinish(ocu.commandQueue);
	LOG_CU_RESULT(err);

    ocu.releaseMemChannels(c0);
    ocu.releaseMemChannels(c1);
}

void cuEdgeDetectorMapEx(
    cu_mem result/*out*/,
    const ocu_channels &rgb, const ocu_channels &rgb2,
    const unsigned int xsize, const unsigned int ysize, const unsigned int step)
{
    unsigned int channel_size = xsize * ysize * sizeof(float);

    ocu_args_d_t &ocu = getOcu();

    ocu_channels rgb_blured = ocu.allocMemChannels(channel_size);
    ocu_channels rgb2_blured = ocu.allocMemChannels(channel_size);

    static const float kSigma[3] = { 1.5f, 0.586f, 0.4f };

    for (int i = 0; i < 3; i++)
    {
        cuBlurEx(rgb.ch[i], xsize, ysize, kSigma[i], 0.0, rgb_blured.ch[i]);
        cuBlurEx(rgb2.ch[i], xsize, ysize, kSigma[i], 0.0, rgb2_blured.ch[i]);
    }

    const unsigned int res_xsize = (xsize + step - 1) / step;
    const unsigned int res_ysize = (ysize + step - 1) / step;

	CUfunction kernel = ocu.kernel[KERNEL_EDGEDETECTOR];
    const void *args[] = { &result,
        &res_xsize, &res_ysize,
        &rgb_blured.r, &rgb_blured.g, &rgb_blured.b,
        &rgb2_blured.r, &rgb2_blured.g, &rgb2_blured.b,
        &xsize, &ysize, &step };

	int x = BLOCK_SIZE_X, y = BLOCK_SIZE_Y;
	calcBlock(res_xsize, res_ysize, x, y);
    CUresult err = cuLaunchKernel(kernel,
        BLOCK_COUNT_X(res_xsize, x), BLOCK_COUNT_Y(res_ysize, y), 1,
        x, y, 1,
        0,
        ocu.commandQueue, (void**)args, NULL);
	LOG_CU_RESULT(err);
    err = cuFinish(ocu.commandQueue);
	LOG_CU_RESULT(err);

    ocu.releaseMemChannels(rgb_blured);
    ocu.releaseMemChannels(rgb2_blured);
}

void cuBlockDiffMapEx(
    cu_mem block_diff_dc/*out*/,
    cu_mem block_diff_ac/*out*/,
    const ocu_channels &rgb, const ocu_channels &rgb2,
    const unsigned int xsize, const unsigned int ysize, const unsigned int step)
{
    ocu_args_d_t &ocu = getOcu();

    const unsigned int res_xsize = (xsize + step - 1) / step;
    const unsigned int res_ysize = (ysize + step - 1) / step;

	CUfunction kernel = ocu.kernel[KERNEL_BLOCKDIFFMAP];
    const void *args[] = { &block_diff_dc, &block_diff_ac,
        &res_xsize, &res_ysize,
        &rgb.r, &rgb.g, &rgb.b,
        &rgb2.r, &rgb2.g, &rgb2.b,
        &xsize, &ysize, &step };

	int x = BLOCK_SIZE_X, y = BLOCK_SIZE_Y;
	calcBlock(res_xsize, res_ysize, x, y);
    CUresult err = cuLaunchKernel(kernel,
        BLOCK_COUNT_X(res_xsize, x), BLOCK_COUNT_Y(res_ysize, y), 1,
        x, y, 1,
        0,
        ocu.commandQueue, (void**)args, NULL);
	LOG_CU_RESULT(err);
    err = cuFinish(ocu.commandQueue);
	LOG_CU_RESULT(err);
}

void cuEdgeDetectorLowFreqEx(
    cu_mem block_diff_ac/*in,out*/,
    const ocu_channels &rgb, const ocu_channels &rgb2,
    const unsigned int xsize, const unsigned int ysize, const unsigned int step)
{
    unsigned int channel_size = xsize * ysize * sizeof(float);

    static const float kSigma = 14;

    ocu_args_d_t &ocu = getOcu();
    ocu_channels rgb_blured = ocu.allocMemChannels(channel_size);
    ocu_channels rgb2_blured = ocu.allocMemChannels(channel_size);

    for (int i = 0; i < 3; i++)
    {
        cuBlurEx(rgb.ch[i], xsize, ysize, kSigma, 0.0, rgb_blured.ch[i]);
        cuBlurEx(rgb2.ch[i], xsize, ysize, kSigma, 0.0, rgb2_blured.ch[i]);
    }

    const unsigned int res_xsize = (xsize + step - 1) / step;
    const unsigned int res_ysize = (ysize + step - 1) / step;

	CUfunction kernel = ocu.kernel[KERNEL_EDGEDETECTORLOWFREQ];
    const void *args[] = { &block_diff_ac,
        &res_xsize, &res_ysize,
        &rgb_blured.r, &rgb_blured.g, &rgb_blured.b,
        &rgb2_blured.r, &rgb2_blured.g, &rgb2_blured.b,
        &xsize, &ysize, &step };

	int x = BLOCK_SIZE_X, y = BLOCK_SIZE_Y;
	calcBlock(res_xsize, res_ysize, x, y);
    CUresult err = cuLaunchKernel(kernel,
        BLOCK_COUNT_X(res_xsize, x), BLOCK_COUNT_Y(res_ysize, y), 1,
        x, y, 1,
        0,
        ocu.commandQueue, (void**)args, NULL);
	LOG_CU_RESULT(err);
    err = cuFinish(ocu.commandQueue);
	LOG_CU_RESULT(err);

    ocu.releaseMemChannels(rgb_blured);
    ocu.releaseMemChannels(rgb2_blured);
}

void cuDiffPrecomputeEx(
    ocu_channels &mask/*out*/,
    const ocu_channels &xyb0, const ocu_channels &xyb1,
    const unsigned int xsize, const unsigned int ysize)
{
    ocu_args_d_t &ocu = getOcu();

	CUfunction kernel = ocu.kernel[KERNEL_DIFFPRECOMPUTE];
    const void *args[] = { &mask.x, &mask.y, &mask.b,
        &xsize, &ysize,
        &xyb0.x, &xyb0.y, &xyb0.b,
        &xyb1.x, &xyb1.y, &xyb1.b };

	int x = BLOCK_SIZE_X, y = BLOCK_SIZE_Y;
	calcBlock(xsize, ysize, x, y);
    CUresult err = cuLaunchKernel(kernel,
        BLOCK_COUNT_X(xsize, x), BLOCK_COUNT_Y(ysize, y), 1,
        x, y, 1,
        0,
        ocu.commandQueue, (void**)args, NULL);
	LOG_CU_RESULT(err);
    err = cuFinish(ocu.commandQueue);
	LOG_CU_RESULT(err);
}

void cuScaleImageEx(cu_mem img/*in, out*/, unsigned int size, float w)
{
    ocu_args_d_t &ocu = getOcu();
    float fw = w;

	CUfunction kernel = ocu.kernel[KERNEL_SCALEIMAGE];
    const void *args[] = { &img, &size, &fw };

    CUresult err = cuLaunchKernel(kernel,
//        (size + BLOCK_SIZE_X * BLOCK_SIZE_Y - 1) / BLOCK_SIZE_X * BLOCK_SIZE_Y, 1, 1,
        (size + 511) / 512, 1, 1,
//        BLOCK_SIZE_X * BLOCK_SIZE_Y, 1, 1,
        512, 1, 1,
        0,
        ocu.commandQueue, (void**)args, NULL);
	LOG_CU_RESULT(err);
    err = cuFinish(ocu.commandQueue);
	LOG_CU_RESULT(err);
}

void cuAverage5x5Ex(cu_mem img/*in,out*/, const unsigned int xsize, const unsigned int ysize)
{
    if (xsize < 4 || ysize < 4) {
        // TODO: Make this work for small dimensions as well.
        return;
    }

    ocu_args_d_t &ocu = getOcu();

    unsigned int len = xsize * ysize * sizeof(float);
    cu_mem img_org = ocu.allocMem(len);

    cuMemcpyDtoD(img_org, img, len);

	CUfunction kernel = ocu.kernel[KERNEL_AVERAGE5X5];
    const void *args[] = { &img, &xsize, &ysize, &img_org };

	int x = BLOCK_SIZE_X, y = BLOCK_SIZE_Y;
	calcBlock(xsize, ysize, x, y);
    CUresult err = cuLaunchKernel(kernel,
        BLOCK_COUNT_X(xsize, x), BLOCK_COUNT_Y(ysize, y), 1,
        x, y, 1,
        0,
        ocu.commandQueue, (void**)args, NULL);
	LOG_CU_RESULT(err);
    err = cuFinish(ocu.commandQueue);
	LOG_CU_RESULT(err);

    ocu.releaseMem(img_org);
}

void cuMinSquareValEx(
    cu_mem img/*in,out*/,
    const unsigned int xsize, const unsigned int ysize,
    const unsigned int square_size, const unsigned int offset)
{
    ocu_args_d_t &ocu = getOcu();

    cu_mem result = ocu.allocMem(sizeof(float) * xsize * ysize);

	CUfunction kernel = ocu.kernel[KERNEL_MINSQUAREVAL];
    const void *args[] = { &result, &xsize, &ysize, &img, &square_size, &offset };

	int x = BLOCK_SIZE_X, y = BLOCK_SIZE_Y;
	calcBlock(xsize, ysize, x, y);
    CUresult err = cuLaunchKernel(kernel,
        BLOCK_COUNT_X(xsize, x), BLOCK_COUNT_Y(ysize, y), 1,
        x, y, 1,
        0,
        ocu.commandQueue, (void**)args, NULL);
	LOG_CU_RESULT(err);
    err = cuFinish(ocu.commandQueue);
	LOG_CU_RESULT(err);
    cuMemcpyDtoD(img, result, sizeof(float) * xsize * ysize);
    ocu.releaseMem(result);
}

static void MakeMask(float extmul, float extoff,
    float mul, float offset,
    float scaler, float *result)
{
    for (unsigned int i = 0; i < 512; ++i) {
        const float c = mul / ((0.01f * scaler * i) + offset);
        result[i] = 1.0f + extmul * (c + extoff);
        result[i] *= result[i];
    }
}

static const float kInternalGoodQualityThreshold = 14.921561160295326f;
static const float kGlobalScale = 1.0f / kInternalGoodQualityThreshold;

void cuDoMask(ocu_channels mask/*in, out*/, ocu_channels mask_dc/*in, out*/, unsigned int xsize, unsigned int ysize)
{
    ocu_args_d_t &ocu = getOcu();

    float extmul = 0.975741017749f;
    float extoff = -4.25328244168f;
    float offset = 0.454909521427f;
    float scaler = 0.0738288224836f;
    float mul = 20.8029176447f;
    static float lut_x[512];
    static bool lutx_init = false;
    if (!lutx_init)
    {
        lutx_init = true;
        MakeMask(extmul, extoff, mul, offset, scaler, lut_x);
    }

    extmul = 0.373995618954f;
    extoff = 1.5307267433f;
    offset = 0.911952641929f;
    scaler = 1.1731667845f;
    mul = 16.2447033988f;
    static float lut_y[512];
    static bool luty_init = false;
    if (!luty_init)
    {
        luty_init = true;
        MakeMask(extmul, extoff, mul, offset, scaler, lut_y);
    }

    extmul = 0.61582234137f;
    extoff = -4.25376118646f;
    offset = 1.05105070921f;
    scaler = 0.47434643535f;
    mul = 31.1444967089f;
    static float lut_b[512];
    static bool lutb_init = false;
    if (!lutb_init)
    {
        lutb_init = true;
        MakeMask(extmul, extoff, mul, offset, scaler, lut_b);
    }

    extmul = 1.79116943438f;
    extoff = -3.86797479189f;
    offset = 0.670960225853f;
    scaler = 0.486575865525f;
    mul = 20.4563479139f;
    static float lut_dcx[512];
    static bool lutdcx_init = false;
    if (!lutdcx_init)
    {
        lutdcx_init = true;
        MakeMask(extmul, extoff, mul, offset, scaler, lut_dcx);
    }

    extmul = 0.212223514236f;
    extoff = -3.65647120524f;
    offset = 1.73396799447f;
    scaler = 0.170392660501f;
    mul = 21.6566724788f;
    static float lut_dcy[512];
    static bool lutdcy_init = false;
    if (!lutdcy_init)
    {
        lutdcy_init = true;
        MakeMask(extmul, extoff, mul, offset, scaler, lut_dcy);
    }

    extmul = 0.349376011816f;
    extoff = -0.894711072781f;
    offset = 0.901647926679f;
    scaler = 0.380086095024f;
    mul = 18.0373825149f;
    static float lut_dcb[512];
    static bool lutdcb_init = false;
    if (!lutdcb_init)
    {
        lutdcb_init = true;
        MakeMask(extmul, extoff, mul, offset, scaler, lut_dcb);
    }

    unsigned int channel_size = 512 * sizeof(float);
    ocu_channels xyb = ocu.allocMemChannels(channel_size, lut_x, lut_y, lut_b);
    ocu_channels xyb_dc = ocu.allocMemChannels(channel_size, lut_dcx, lut_dcy, lut_dcb);

	CUfunction kernel = ocu.kernel[KERNEL_DOMASK];
    const void *args[] = { &mask.r, &mask.g, &mask.b,
        &xsize, &ysize,
        &mask_dc.r, &mask_dc.g, &mask_dc.b,
        &xyb.x, &xyb.y, &xyb.b,
        &xyb_dc.x, &xyb_dc.y, &xyb_dc.b };

	int x = BLOCK_SIZE_X, y = BLOCK_SIZE_Y;
	calcBlock(xsize, ysize, x, y);
    CUresult err = cuLaunchKernel(kernel,
        BLOCK_COUNT_X(xsize, x), BLOCK_COUNT_Y(ysize, y), 1,
        x, y, 1,
        0,
        ocu.commandQueue, (void**)args, NULL);
	LOG_CU_RESULT(err);
    err = cuFinish(ocu.commandQueue);
	LOG_CU_RESULT(err);

    ocu.releaseMemChannels(xyb);
    ocu.releaseMemChannels(xyb_dc);
}

void cuMaskEx(
    ocu_channels mask/*out*/, ocu_channels mask_dc/*out*/,
    const ocu_channels &rgb, const ocu_channels &rgb2,
    const unsigned int xsize, const unsigned int ysize)
{
    cuDiffPrecomputeEx(mask, rgb, rgb2, xsize, ysize);
    for (int i = 0; i < 3; i++)
    {
        cuAverage5x5Ex(mask.ch[i], xsize, ysize);
        cuMinSquareValEx(mask.ch[i], xsize, ysize, 4, 0);

        static const float sigma[3] = {
            9.65781083553f,
            14.2644604355f,
            4.53358927369f,
        };

        cuBlurEx(mask.ch[i], xsize, ysize, sigma[i], 0.0);
    }

    cuDoMask(mask, mask_dc, xsize, ysize);

    for (int i = 0; i < 3; i++)
    {
        cuScaleImageEx(mask.ch[i], xsize * ysize, kGlobalScale * kGlobalScale);
        cuScaleImageEx(mask_dc.ch[i], xsize * ysize, kGlobalScale * kGlobalScale);
    }
}

void cuCombineChannelsEx(
    cu_mem result /*out*/,
    const ocu_channels& mask,
    const ocu_channels& mask_dc,
    const unsigned int xsize, const unsigned int ysize,
    const cu_mem block_diff_dc,
    const cu_mem block_diff_ac,
    const cu_mem edge_detector_map,
    const unsigned int res_xsize,
    const unsigned int step)
{
    ocu_args_d_t& ocu = getOcu();

    // Work area: how many output pixels we need to compute
    const unsigned int work_xsize = ((xsize - 8 + step) + step - 1) / step;
    const unsigned int work_ysize = ((ysize - 8 + step) + step - 1) / step;

    // Block size: 128 threads per block (16x8)
    const int block_x = 16;
    const int block_y = 8;

    // Grid size: enough blocks to cover the work area
    const int grid_x = (work_xsize + block_x - 1) / block_x;
    const int grid_y = (work_ysize + block_y - 1) / block_y;

    CUfunction kernel = ocu.kernel[KERNEL_COMBINECHANNELS];
    const void* args[] = {
        &result,
        &mask.r, &mask.g, &mask.b,
        &mask_dc.r, &mask_dc.g, &mask_dc.b,
        &xsize, &ysize,
        &block_diff_dc, &block_diff_ac,
        &edge_detector_map,
        &res_xsize,
        &step
    };

    CUresult err = cuLaunchKernel(kernel,
        grid_x, grid_y, 1,         // gridDim
        block_x, block_y, 1,       // blockDim
        0,                         // sharedMemBytes
        ocu.commandQueue, (void**)args, nullptr);
    LOG_CU_RESULT(err);

    err = cuFinish(ocu.commandQueue);
    LOG_CU_RESULT(err);
}

void cuUpsampleSquareRootEx(cu_mem diffmap, const unsigned int xsize, const unsigned int ysize, const int step)
{
    ocu_args_d_t &ocu = getOcu();

    cu_mem diffmap_out = ocu.allocMem(xsize * ysize * sizeof(float));

	CUfunction kernel = ocu.kernel[KERNEL_UPSAMPLESQUAREROOT];
    const void *args[] = { &diffmap_out, &diffmap, &xsize, &ysize, &step };

    const unsigned int res_xsize = (xsize + step - 1) / step;
    const unsigned int res_ysize = (ysize + step - 1) / step;

    const int block_x = 16;
    const int block_y = 8;

    const int grid_x = (res_xsize + block_x - 1) / block_x;
    const int grid_y = (res_ysize + block_y - 1) / block_y;


    CUresult err = cuLaunchKernel(kernel,
		(grid_x), (grid_y), 1,
        block_x, block_y, 1,
        0,
        ocu.commandQueue, (void**)args, NULL);
	LOG_CU_RESULT(err);
    err = cuFinish(ocu.commandQueue);
	LOG_CU_RESULT(err);
    cuMemcpyDtoD(diffmap, diffmap_out, xsize * ysize * sizeof(float));

    ocu.releaseMem(diffmap_out);
}

void cuRemoveBorderEx(cu_mem out, const cu_mem in, const unsigned int xsize, const unsigned int ysize, const int step)
{
    ocu_args_d_t &ocu = getOcu();

    int cls = 8 - step;
    int cls2 = (8 - step) / 2;

    int out_xsize = xsize - cls;
    int out_ysize = ysize - cls;

	CUfunction kernel = ocu.kernel[KERNEL_REMOVEBORDER];
    const void *args[] = { &out, &out_xsize, &out_ysize, &in, &cls, &cls2 };

	int x = BLOCK_SIZE_X, y = BLOCK_SIZE_Y;
	calcBlock(out_xsize, out_ysize, x, y);
    CUresult err = cuLaunchKernel(kernel,
        BLOCK_COUNT_X(out_xsize, x), BLOCK_COUNT_Y(out_ysize, y), 1,
        x, y, 1,
        0,
        ocu.commandQueue, (void**)args, NULL);
	LOG_CU_RESULT(err);
    err = cuFinish(ocu.commandQueue);
	LOG_CU_RESULT(err);
}

void cuAddBorderEx(cu_mem out, unsigned int xsize, unsigned int ysize, int step, cu_mem in)
{
    ocu_args_d_t &ocu = getOcu();

    int cls = 8 - step;
    int cls2 = (8 - step) / 2;
	CUfunction kernel = ocu.kernel[KERNEL_ADDBORDER];
    const void *args[] = { &out, &xsize, &ysize, &cls, &cls2, &in };

	int x = BLOCK_SIZE_X, y = BLOCK_SIZE_Y;
	calcBlock(xsize, ysize, x, y);
    CUresult err = cuLaunchKernel(kernel,
        BLOCK_COUNT_X(xsize, x), BLOCK_COUNT_Y(ysize, y), 1,
        x, y, 1,
        0,
        ocu.commandQueue, (void**)args, NULL);
	LOG_CU_RESULT(err);
    err = cuFinish(ocu.commandQueue);
	LOG_CU_RESULT(err);
}

void cuCalculateDiffmapEx(cu_mem diffmap/*in,out*/, const unsigned int xsize, const unsigned int ysize, const int step)
{
    cuUpsampleSquareRootEx(diffmap, xsize, ysize, step);

    static const float kSigma = 8.8510880283;
    static const float mul1 = 24.8235314874;
    static const float scale = 1.0 / (1.0 + mul1);

    const int s = 8 - step;
    int s2 = (8 - step) / 2;

    ocu_args_d_t &ocu = getOcu();
    cu_mem blurred = ocu.allocMem((xsize - s) * (ysize - s) * sizeof(float));
    cuRemoveBorderEx(blurred, diffmap, xsize, ysize, step);

    static const float border_ratio = 0.03027655136f;
    cuBlurEx(blurred, xsize - s, ysize - s, kSigma, border_ratio);

    cuAddBorderEx(diffmap, xsize, ysize, step, blurred);
    cuScaleImageEx(diffmap, xsize * ysize, scale);

    ocu.releaseMem(blurred);
}

void cuCopyFromJpegComponent(
	coeff_t *output_batch/*in,out*/,
	uint8_t  *output_idct/*in,out*/,
	const coeff_t *jpeg_batch/*in*/,
	const int *quant,
	const int jpeg_block_width,
	const int jpeg_block_height,
	const int output_block_width,
	const int output_block_height,
	const int output_width,
	const int output_height)
{
	using namespace guetzli;

	ocu_args_d_t &ocu = getOcu();

	int src_block_count = jpeg_block_width * jpeg_block_height;
	cu_mem src_coeff = ocu.allocMem(src_block_count * sizeof(::coeff_t) * kDCTBlockSize, jpeg_batch);

	int dst_coeff_size = output_block_width * output_block_height * sizeof(::coeff_t) * kDCTBlockSize;
	cu_mem dst_coeff = ocu.allocMem(dst_coeff_size, output_batch);

	int dst_idct_size = output_block_width * output_block_height * sizeof(uint8_t) * kDCTBlockSize;
	cu_mem dst_idct = ocu.allocMem(dst_idct_size, output_idct);

	int src_quant_size = kDCTBlockSize * sizeof(int);
	cu_mem src_quant = ocu.allocMem(src_quant_size, quant);

	CUfunction kernel = ocu.kernel[KERNEL_COPYFROMJPEGCOMPONENT];
	const void *args[] = { &dst_coeff, &dst_idct,
		&src_coeff, &src_quant, &jpeg_block_width, &jpeg_block_height,
		&output_block_width, &output_block_height,
		&output_width, &output_height };

	int x = BLOCK_SIZE_X, y = BLOCK_SIZE_Y;
	calcBlock(output_block_width, output_block_height, x, y);
	CUresult err = cuLaunchKernel(kernel,
		BLOCK_COUNT_X(output_block_width, x), BLOCK_COUNT_Y(output_block_height, y), 1,
		x, y, 1,
		0,
		ocu.commandQueue, (void**)args, NULL);
	LOG_CU_RESULT(err);
	err = cuFinish(ocu.commandQueue);
	LOG_CU_RESULT(err);
	
	cuMemcpyDtoHAsync(output_batch, dst_coeff, dst_coeff_size, ocu.commandQueue);
	cuMemcpyDtoHAsync(output_idct, dst_idct, dst_idct_size, ocu.commandQueue);
	err = cuFinish(ocu.commandQueue);
	LOG_CU_RESULT(err);

	ocu.releaseMem(src_coeff);
	ocu.releaseMem(dst_coeff);
	ocu.releaseMem(dst_idct);
	ocu.releaseMem(src_quant);
}

void cuApplyGlobalQuantization(
	coeff_t *output_batch/*in,out*/,
	uchar  *output_idct/*out*/,
	uchar  *output_bool/*out*/,
	const int* q/*in*/,
	const int block_width,
	const int block_height)
{
	using namespace guetzli;

	ocu_args_d_t &ocu = getOcu();

	int dst_coeff_size = block_width * block_height * sizeof(::coeff_t) * kDCTBlockSize;
	cu_mem dst_coeff = ocu.allocMem(dst_coeff_size, output_batch);

	int dst_idct_size = block_width * block_height * sizeof(uint8_t)* kDCTBlockSize;
	cu_mem dst_idct = ocu.allocMem(dst_idct_size, output_idct);

	int dst_bool_size = block_width * block_height * sizeof(uchar);
	cu_mem dst_bool = ocu.allocMem(dst_bool_size, output_bool);

	int src_q_size = kDCTBlockSize * sizeof(int);
	cu_mem src_q = ocu.allocMem(src_q_size, q);

	CUfunction kernel = ocu.kernel[KERNEL_APPLYGLOBALQUANTIZATION];
	const void *args[] = { &dst_coeff, &dst_idct,
		&dst_bool, &src_q, &block_width, &block_height };

	int x = BLOCK_SIZE_X, y = BLOCK_SIZE_Y;
	calcBlock(block_width, block_height, x, y);
	CUresult err = cuLaunchKernel(kernel,
		BLOCK_COUNT_X(block_width, x), BLOCK_COUNT_Y(block_height, y), 1,
		x, y, 1,
		0,
		ocu.commandQueue, (void**)args, NULL);
	LOG_CU_RESULT(err);
	
	err = cuFinish(ocu.commandQueue);
	LOG_CU_RESULT(err);

	cuMemcpyDtoHAsync(output_batch, dst_coeff, dst_coeff_size, ocu.commandQueue);
	cuMemcpyDtoHAsync(output_idct, dst_idct, dst_idct_size, ocu.commandQueue);
	cuMemcpyDtoHAsync(output_bool, dst_bool, dst_bool_size, ocu.commandQueue);

	err = cuFinish(ocu.commandQueue);
	LOG_CU_RESULT(err);

	ocu.releaseMem(dst_bool);
	ocu.releaseMem(dst_coeff);
	ocu.releaseMem(dst_idct);
	ocu.releaseMem(src_q);
}

void cuComponentsToPixels(
	uchar *rgb,/*out*/
	const int xmin,
	const int ymin,
	const int xsize,
	const int ysize,
	const std::vector<guetzli::OutputImageComponent> &components /*in*/)
{
	using namespace guetzli;

	ocu_args_d_t &ocu = getOcu();

	const int stride = 3;

	int out_size = xsize * ysize * stride * sizeof(uchar);
	cu_mem cl_out = ocu.allocMem(out_size, rgb);

	for (int c = 0; c < stride; ++c)
	{
		cu_mem cl_out_offset = cl_out + c;

		const int width = components[c].width();
		const int height = components[c].height();

		const int yend1 = ymin + ysize;
		const int yend0 = std::min(yend1, height);
		const int xend1 = xmin + xsize;
		const int xend0 = std::min(xend1, width);

		int pixels_size = width * height * sizeof(ushort);
		cu_mem cl_pixels = ocu.allocMem(pixels_size, components[c].pixels());

		{
			CUfunction kernel = ocu.kernel[KERNEL_COMPONENTSTOPIXELS];
			const void *args[] = { &cl_out_offset, 
				&xmin, &ymin, &xsize, &ysize, &stride,
				&cl_pixels, &width, &height };

			int x = BLOCK_SIZE_X, y = BLOCK_SIZE_Y;
			calcBlock(xend0 - xmin, yend0 - ymin, x, y);
			CUresult err = cuLaunchKernel(kernel,
				BLOCK_COUNT_X(xend0 - xmin, x), BLOCK_COUNT_Y(yend0 - ymin, y), 1,
				x, y, 1,
				0,
				ocu.commandQueue, (void**)args, NULL);
			LOG_CU_RESULT(err);

			err = cuFinish(ocu.commandQueue);
			LOG_CU_RESULT(err);
		}

		ocu.releaseMem(cl_pixels);

		if (xend1 - xend0 > 0) {
			CUfunction kernel = ocu.kernel[KERNEL_COMPONENTSTOPIXELS_EX1];
			const void *args[] = { &cl_out_offset,
				&xmin, &ymin, &xsize, &ysize, &stride,
				&width, &height };

			CUresult err = cuLaunchKernel(kernel,
				xend1 - xend0, BLOCK_COUNT_Y(yend0 - ymin, BLOCK_SIZE), 1,
				1, BLOCK_SIZE, 1,
				0,
				ocu.commandQueue, (void**)args, NULL);
			LOG_CU_RESULT(err);

			err = cuFinish(ocu.commandQueue);
			LOG_CU_RESULT(err);
		}

		if (yend1 - yend0 > 0) {
			CUfunction kernel = ocu.kernel[KERNEL_COMPONENTSTOPIXELS_EX2];
			const void *args[] = { &cl_out_offset,
				&xmin, &ymin, &xsize, &ysize, &stride,
				&width, &height };

			CUresult err = cuLaunchKernel(kernel,
				BLOCK_COUNT_X(xsize, BLOCK_SIZE), yend1 - yend0, 1,
				BLOCK_SIZE, 1, 1,
				0,
				ocu.commandQueue, (void**)args, NULL);
			LOG_CU_RESULT(err);

			err = cuFinish(ocu.commandQueue);
			LOG_CU_RESULT(err);
		}
	}

	CUfunction kernel = ocu.kernel[KERNEL_COLORTRANSFORMYCBCRTORGB];
	const void *args[] = { &cl_out };

	CUresult err = cuLaunchKernel(kernel,
		BLOCK_COUNT_X(xsize * ysize, BLOCK_SIZE), 1, 1,
		BLOCK_SIZE, 1, 1,
		0,
		ocu.commandQueue, (void**)args, NULL);
	LOG_CU_RESULT(err);

	err = cuFinish(ocu.commandQueue);
	LOG_CU_RESULT(err);

	cuMemcpyDtoH(rgb, cl_out, out_size);
	ocu.releaseMem(cl_out);
}

#endif
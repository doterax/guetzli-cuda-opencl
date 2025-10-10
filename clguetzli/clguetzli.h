/*
* OpenCL edition implementation of guetzli.
*
* Author: strongtu@tencent.com
*         ianhuang@tencent.com
*         chriskzhou@tencent.com
*/
#pragma once
#include <vector>
#include "guetzli/processor.h"
#include "guetzli/butteraugli_comparator.h"
#include "ocl.h"
#include "clguetzli.cl.h"

#include "cuguetzli.h"

enum MATH_MODE
{
	MODE_CPU = 0,
	MODE_CPU_OPT,
	MODE_OPENCL,
	MODE_CUDA,
	MODE_CHECKCL,
	MODE_CHECKCUDA,
    MODE_AUTO,
};

extern MATH_MODE g_mathMode;

#ifdef __USE_OPENCL__

void clOpsinDynamicsImage(
    float *r, float *g, float *b, 
    const unsigned int xsize, const unsigned int ysize);

void clDiffmapOpsinDynamicsImage(
    float* result,
    const float* r,  const float* g,  const float* b,
    const float* r2, const float* g2, const float* b2,
    const unsigned int xsize, const unsigned int ysize,
    const unsigned int step);

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
    const float BlockErrorLimit);

void clMask(
    float* mask_r,   float* mask_g,   float* mask_b,
    float* maskdc_r, float* maskdc_g, float* maskdc_b, 
    const unsigned int xsize, const unsigned int ysize,
    const float* r,  const float* g,  const float* b,
    const float* r2, const float* g2, const float* b2);

void clDiffmapOpsinDynamicsImageEx(
    cl_mem result,
    ocl_channels xyb0,
    ocl_channels xyb1,
    const unsigned int xsize, const unsigned int ysize,
    const unsigned int step);


void clConvolutionEx(
    cl_mem result/*out*/,
    const cl_mem inp, unsigned int xsize, unsigned int ysize,
    const cl_mem multipliers, unsigned int len,
    int xstep, int offset, float border_ratio);

void clConvolutionXEx(
    cl_mem result/*out*/, 
    const cl_mem inp, unsigned int xsize, unsigned int ysize,
    const cl_mem multipliers, unsigned int len,
    int xstep, int offset, float border_ratio);

void clConvolutionYEx(
    cl_mem result/*out*/,
    const cl_mem inp, unsigned int xsize, unsigned int ysize,
    const cl_mem multipliers, unsigned int len,
    int xstep, int offset, float border_ratio);

void clSquareSampleEx(
    cl_mem result/*out*/,
    const cl_mem image, unsigned int xsize, unsigned int ysize,
    unsigned int xstep, unsigned int ystep);

void clBlurEx(cl_mem image/*out, opt*/, const unsigned int xsize, const unsigned int ysize,
    const float sigma, const float border_ratio,
    cl_mem result = nullptr/*out, opt*/);

void clOpsinDynamicsImageEx(ocl_channels &rgb, const unsigned int xsize, const unsigned int ysize);

void clMaskHighIntensityChangeEx(
    ocl_channels &xyb0/*in,out*/,
	ocl_channels &xyb1/*in,out*/,
	const unsigned int xsize, const unsigned int ysize);

void clEdgeDetectorMapEx(
    cl_mem result/*out*/,
    const ocl_channels &rgb, const ocl_channels &rgb2,
    const unsigned int xsize, const unsigned int ysize, const unsigned int step);

void clBlockDiffMapEx(
    cl_mem block_diff_dc/*out*/, 
    cl_mem block_diff_ac/*out*/,
    const ocl_channels &rgb, const ocl_channels &rgb2,
	const unsigned int xsize, const unsigned int ysize, const unsigned int step);

void clEdgeDetectorLowFreqEx(
    cl_mem block_diff_ac/*in,out*/,
    const ocl_channels &rgb, const ocl_channels &rgb2,
    const unsigned int xsize, const unsigned int ysize, const unsigned int step);

void clDiffPrecomputeEx(
    ocl_channels &mask/*out*/,
    const ocl_channels &xyb0, const ocl_channels &xyb1, 
    const unsigned int xsize, const unsigned int ysize);

void clScaleImageEx(cl_mem img/*in, out*/, unsigned int size, float w);

void clAverage5x5Ex(cl_mem img/*in,out*/, const unsigned int xsize, const unsigned int ysize);

void clMinSquareValEx(
    cl_mem img/*in,out*/, 
    const unsigned int xsize, const unsigned int ysize, 
    const unsigned int square_size, const unsigned int offset);

void clMaskEx(
    ocl_channels mask/*out*/, ocl_channels mask_dc/*out*/,
    const ocl_channels &rgb, const ocl_channels &rgb2,
	const unsigned int xsize, const unsigned int ysize);

void clCombineChannelsEx(
    cl_mem result/*out*/,
	const ocl_channels &mask,
	const ocl_channels &mask_dc,
    const unsigned int xsize, const unsigned int ysize,
	const cl_mem block_diff_dc,
	const cl_mem block_diff_ac,
	const cl_mem edge_detector_map,
	const unsigned int res_xsize,
	const unsigned int step);

void clUpsampleSquareRootEx(cl_mem diffmap, const unsigned int xsize, const unsigned int ysize, const int step);

void clRemoveBorderEx(cl_mem out, const cl_mem in, const unsigned int xsize, const unsigned int ysize, const int step);

void clAddBorderEx(cl_mem out, const unsigned int xsize, const unsigned int ysize, const int step, const cl_mem in);

void clCalculateDiffmapEx(cl_mem diffmap/*in,out*/, const unsigned int xsize, const unsigned int ysize, const int step);

void clCopyFromJpegComponent(
	coeff_t *output_batch/*in,out*/,
	uint8_t  *output_idct/*in,out*/,
	const coeff_t *jpeg_batch/*in*/,
	const int *quant,
	const int jpeg_block_width,
	const int jpeg_block_height,
	const int output_block_width,
	const int output_block_height,
	const int output_width,
	const int output_height);

void clApplyGlobalQuantization(
	coeff_t *output_batch/*in,out*/,
	uchar  *output_idct/*out*/,
	uchar  *output_bool/*out*/,
	const int* q/*in*/,
	const int block_width,
	const int block_height);

void clComponentsToPixels(
	uchar *rgb,/*out*/
	const int xmin,
	const int ymin,
	const int xsize,
	const int ysize,
	const std::vector<guetzli::OutputImageComponent> &components /*in*/);

#endif

#if defined(__USE_OPENCL__) || defined(__USE_CUDA__)

class guetzli::OutputImage;

namespace guetzli {

    class ButteraugliComparatorEx : public ButteraugliComparator
    {
    public:
        ButteraugliComparatorEx(const int width, const int height,
            const std::vector<uint8_t>* rgb,
            const float target_distance, ProcessStats* stats);

        void Compare(const OutputImage& img) override;
        void StartBlockComparisons() override;
        void FinishBlockComparisons() override;

        double CompareBlock(const OutputImage& img, int off_x, int off_y, const coeff_t* candidate_block, const int comp_mask) const override;
    public:
        std::vector<float> imgOpsinDynamicsBlockList;   // [RR..RRGG..GGBB..BB]:blockCount
        std::vector<float> imgMaskXyzScaleBlockList;    // [RGBRGB..RGBRGB]:blockCount
        std::vector<std::vector<float>> rgb_orig_opsin;
    };
}

#endif
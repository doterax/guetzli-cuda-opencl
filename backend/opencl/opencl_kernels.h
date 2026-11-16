// The OpenCL device of one run: kernel launchers for every pipeline stage (GpuKernels). The context is created on
// first use (or by probe()) and owned here.
//
// Author: strongtu@tencent.com
//         ianhuang@tencent.com
//         chriskzhou@tencent.com
#ifndef GUETZLI_BACKEND_OPENCL_OPENCL_KERNELS_H_
#define GUETZLI_BACKEND_OPENCL_OPENCL_KERNELS_H_

#ifdef __USE_OPENCL__

#include <memory>
#include <vector>

#include "backend/gpu/gpu_kernels.h"
#include "backend/gpu/gpu_options.h"
#include "backend/opencl/opencl_context.h"
#include "logging/logger.h"

namespace guetzli {

class OpenClKernels final : public GpuKernels {
 public:
  OpenClKernels(const GpuOptions& options, ILogger& logger);
  ~OpenClKernels() override;

  // Creates the context now (device selection, kernel compile). Logs the reason and returns false if OpenCL is not
  // usable.
  [[nodiscard]] bool probe();

  void opsinDynamicsImage(float* r, float* g, float* b, unsigned int xsize, unsigned int ysize) override;
  void mask(float* mask_r, float* mask_g, float* mask_b, float* maskdc_r, float* maskdc_g, float* maskdc_b,
            unsigned int xsize, unsigned int ysize, const float* r, const float* g, const float* b, const float* r2,
            const float* g2, const float* b2) override;
  std::unique_ptr<GpuCompareSession> startCompare(const float* r, const float* g, const float* b, int xsize,
                                                  int ysize) override;

  void copyFromJpegComponent(coeff_t* output_batch, uint8_t* output_idct, const coeff_t* jpeg_batch, const int* quant,
                             int jpeg_block_width, int jpeg_block_height, int output_block_width,
                             int output_block_height, int output_width, int output_height) override;
  void applyGlobalQuantization(coeff_t* output_batch, uint8_t* output_idct, uint8_t* output_bool, const int* q,
                               int block_width, int block_height) override;
  void componentsToPixels(uint8_t* rgb, int xmin, int ymin, int xsize, int ysize,
                          const std::vector<OutputImageComponent>& components) override;

  void computeBlockZeroingOrder(CoeffData* output_order_batch, const ChannelInfo orig_channel[3],
                                const float* orig_image_batch, const float* mask_scale, int image_width,
                                int image_height, const ChannelInfo mayout_channel[3], int factor, int comp_mask,
                                float block_error_limit) override;

 private:
  class CompareSession;

  OpenClContext& context();

  // Launchers of the butteraugli pipeline on device buffers.
  void enqueueRange2D(cl_kernel kernel, size_t w, size_t h, size_t lx = 16, size_t ly = 16);
  void enqueueRange1D(cl_kernel kernel, size_t n, size_t l = 256);
  void diffmapOpsinDynamicsImage(cl_mem result, OclChannels xyb0, OclChannels xyb1, unsigned int xsize,
                                 unsigned int ysize, unsigned int step);
  cl_mem convolutionScale(cl_mem multipliers, int len, int n, int step, int offset, float border_ratio);
  void convolutionX(cl_mem result, cl_mem inp, unsigned int xsize, unsigned int ysize, cl_mem multipliers,
                    unsigned int len, int xstep, int offset, float border_ratio);
  void convolutionY(cl_mem result, cl_mem inp, unsigned int xsize, unsigned int ysize, cl_mem multipliers,
                    unsigned int len, int xstep, int offset, float border_ratio);
  void squareSample(cl_mem result, cl_mem image, unsigned int xsize, unsigned int ysize, unsigned int xstep,
                    unsigned int ystep);
  void blur(cl_mem image, unsigned int xsize, unsigned int ysize, float sigma, float border_ratio, cl_mem result);
  void opsinDynamicsOnDevice(const OclChannels& rgb, unsigned int xsize, unsigned int ysize);
  void maskHighIntensityChange(const OclChannels& xyb0, const OclChannels& xyb1, unsigned int xsize,
                               unsigned int ysize);
  void edgeDetectorMap(cl_mem result, const OclChannels& rgb, const OclChannels& rgb2, unsigned int xsize,
                       unsigned int ysize, unsigned int step);
  void blockDiffMap(cl_mem block_diff_dc, cl_mem block_diff_ac, const OclChannels& rgb, const OclChannels& rgb2,
                    unsigned int xsize, unsigned int ysize, unsigned int step);
  void edgeDetectorLowFreq(cl_mem block_diff_ac, const OclChannels& rgb, const OclChannels& rgb2, unsigned int xsize,
                           unsigned int ysize, unsigned int step);
  void diffPrecompute(const OclChannels& mask, const OclChannels& xyb0, const OclChannels& xyb1, unsigned int xsize,
                      unsigned int ysize);
  void scaleImage(cl_mem img, unsigned int size, float w);
  void average5x5(cl_mem img, unsigned int xsize, unsigned int ysize);
  void minSquareVal(cl_mem img, unsigned int xsize, unsigned int ysize, unsigned int square_size, unsigned int offset);
  void doMask(OclChannels mask, OclChannels mask_dc, size_t xsize, size_t ysize);
  void maskOnDevice(OclChannels mask, OclChannels mask_dc, const OclChannels& rgb, const OclChannels& rgb2,
                    unsigned int xsize, unsigned int ysize);
  void combineChannels(cl_mem result, const OclChannels& mask, const OclChannels& mask_dc, unsigned int xsize,
                       unsigned int ysize, cl_mem block_diff_dc, cl_mem block_diff_ac, cl_mem edge_detector_map,
                       unsigned int res_xsize, unsigned int step);
  void upsampleSquareRoot(cl_mem diffmap, unsigned int xsize, unsigned int ysize, int step);
  void removeBorder(cl_mem out, cl_mem in, unsigned int xsize, unsigned int ysize, int step);
  void addBorder(cl_mem out, unsigned int xsize, unsigned int ysize, int step, cl_mem in);
  void calculateDiffmap(cl_mem diffmap, unsigned int xsize, unsigned int ysize, int step);
  void componentsToPixelsDevice(cl_mem out, const cl_mem pixels[3], int xmin, int ymin, int xsize, int ysize,
                                const OutputImageComponent* const components[3]);
  void compare(const OclChannels& ref, int xsize, int ysize, const OutputImage& img, int step, float* distmap);

  GpuOptions options_;
  ILogger& logger_;
  std::unique_ptr<OpenClContext> context_;
};

}  // namespace guetzli

#endif  // __USE_OPENCL__

#endif  // GUETZLI_BACKEND_OPENCL_OPENCL_KERNELS_H_

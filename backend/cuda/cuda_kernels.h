// The CUDA device of one run: kernel launchers for every pipeline stage (GpuKernels). The driver is loaded and the
// context created on first use (or by probe()) and owned here.
//
// Author: strongtu@tencent.com
//         ianhuang@tencent.com
//         chriskzhou@tencent.com
#ifndef GUETZLI_BACKEND_CUDA_CUDA_KERNELS_H_
#define GUETZLI_BACKEND_CUDA_CUDA_KERNELS_H_

#ifdef __USE_CUDA__

#include <memory>
#include <vector>

#include "backend/cuda/cuda_context.h"
#include "backend/cuda/cuda_driver.h"
#include "backend/gpu/gpu_kernels.h"
#include "backend/gpu/gpu_options.h"
#include "logging/logger.h"

namespace guetzli {

class CudaKernels final : public GpuKernels {
 public:
  CudaKernels(const GpuOptions& options, ILogger& logger);
  ~CudaKernels() override;
  CudaKernels(const CudaKernels&) = delete;
  CudaKernels& operator=(const CudaKernels&) = delete;
  CudaKernels(CudaKernels&&) = delete;
  CudaKernels& operator=(CudaKernels&&) = delete;

  // Loads the driver and creates the context now. Returns false if CUDA is not usable (logged, unless the driver is
  // simply not installed).
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

  CudaContext& context();
  void createContext(const CudaDriver& driver);

  // Launchers of the butteraugli pipeline on device buffers.
  void diffmapOpsinDynamicsImage(cu_mem result, OcuChannels xyb0, OcuChannels xyb1, unsigned int xsize,
                                 unsigned int ysize, unsigned int step);
  cu_mem convolutionScale(cu_mem multipliers, int len, int n, int step, int offset, float border_ratio);
  void convolutionX(cu_mem result, cu_mem inp, unsigned int xsize, unsigned int ysize, cu_mem multipliers,
                    unsigned int len, int xstep, int offset, float border_ratio);
  void convolutionY(cu_mem result, cu_mem inp, unsigned int xsize, unsigned int ysize, cu_mem multipliers,
                    unsigned int len, int xstep, int offset, float border_ratio);
  void squareSample(cu_mem result, cu_mem image, unsigned int xsize, unsigned int ysize, unsigned int xstep,
                    unsigned int ystep);
  // result == 0: in place.
  void blur(cu_mem image, unsigned int xsize, unsigned int ysize, float sigma, float border_ratio, cu_mem result = 0);
  void opsinDynamicsOnDevice(const OcuChannels& rgb, unsigned int xsize, unsigned int ysize);
  void maskHighIntensityChange(const OcuChannels& xyb0, const OcuChannels& xyb1, unsigned int xsize,
                               unsigned int ysize);
  void edgeDetectorMap(cu_mem result, const OcuChannels& rgb, const OcuChannels& rgb2, unsigned int xsize,
                       unsigned int ysize, unsigned int step);
  void blockDiffMap(cu_mem block_diff_dc, cu_mem block_diff_ac, const OcuChannels& rgb, const OcuChannels& rgb2,
                    unsigned int xsize, unsigned int ysize, unsigned int step);
  void edgeDetectorLowFreq(cu_mem block_diff_ac, const OcuChannels& rgb, const OcuChannels& rgb2, unsigned int xsize,
                           unsigned int ysize, unsigned int step);
  void diffPrecompute(const OcuChannels& mask, const OcuChannels& xyb0, const OcuChannels& xyb1, unsigned int xsize,
                      unsigned int ysize);
  void scaleImage(cu_mem img, unsigned int size, float w);
  void average5x5(cu_mem img, unsigned int xsize, unsigned int ysize);
  void minSquareVal(cu_mem img, unsigned int xsize, unsigned int ysize, unsigned int square_size, unsigned int offset);
  void doMask(OcuChannels mask, OcuChannels mask_dc, unsigned int xsize, unsigned int ysize);
  void maskOnDevice(OcuChannels mask, OcuChannels mask_dc, const OcuChannels& rgb, const OcuChannels& rgb2,
                    unsigned int xsize, unsigned int ysize);
  void combineChannels(cu_mem result, const OcuChannels& mask, const OcuChannels& mask_dc, unsigned int xsize,
                       unsigned int ysize, cu_mem block_diff_dc, cu_mem block_diff_ac, cu_mem edge_detector_map,
                       unsigned int res_xsize, unsigned int step);
  void upsampleSquareRoot(cu_mem diffmap, unsigned int xsize, unsigned int ysize, int step);
  void removeBorder(cu_mem out, cu_mem in, unsigned int xsize, unsigned int ysize, int step);
  void addBorder(cu_mem out, unsigned int xsize, unsigned int ysize, int step, cu_mem in);
  void calculateDiffmap(cu_mem diffmap, unsigned int xsize, unsigned int ysize, int step);
  void componentsToPixelsDevice(cu_mem out, const cu_mem pixels[3], int xmin, int ymin, int xsize, int ysize,
                                const OutputImageComponent* const components[3]);
  void compare(const OcuChannels& ref, int xsize, int ysize, const OutputImage& img, int step, float* distmap);

  GpuOptions options_;
  ILogger& logger_;
  std::unique_ptr<CudaContext> context_;
};

}  // namespace guetzli

#endif  // __USE_CUDA__

#endif  // GUETZLI_BACKEND_CUDA_CUDA_KERNELS_H_

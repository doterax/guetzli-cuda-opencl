// Butteraugli on a GPU. OpenCL and CUDA differ only in the kernel launchers and the device-resident
// Compare() state, which the leaf classes in backend/opencl and backend/cuda provide.
#ifndef GUETZLI_BACKEND_GPU_GPU_COMPARATOR_H_
#define GUETZLI_BACKEND_GPU_GPU_COMPARATOR_H_

#include <optional>
#include <vector>

#include "guetzli/butteraugli_comparator.h"

namespace guetzli {

// The cl*/cu* launchers for the image operations (clguetzli.h, cuguetzli.h).
struct GpuButteraugliKernels {
  void (*opsin_dynamics_image)(float* r, float* g, float* b, unsigned int xsize, unsigned int ysize);
  void (*mask)(float* mask_r, float* mask_g, float* mask_b, float* maskdc_r, float* maskdc_g, float* maskdc_b,
               unsigned int xsize, unsigned int ysize, const float* r, const float* g, const float* b, const float* r2,
               const float* g2, const float* b2);
};

// Images larger than 100x100 run on the device; smaller ones, and MaskHighIntensityChange, use the
// reference code in butteraugli.h.
class GpuButteraugliOps final : public ButteraugliOps {
 public:
  explicit GpuButteraugliOps(const GpuButteraugliKernels& kernels) : kernels_(kernels) {}

  void OpsinDynamicsImage(size_t xsize, size_t ysize, std::vector<std::vector<float>>& rgb) override;
  void Mask(const std::vector<std::vector<float>>& xyb0, const std::vector<std::vector<float>>& xyb1, size_t xsize,
            size_t ysize, std::vector<std::vector<float>>* mask, std::vector<std::vector<float>>* mask_dc) override;
  void MaskHighIntensityChange(size_t xsize, size_t ysize, const std::vector<std::vector<float>>& c0,
                               const std::vector<std::vector<float>>& c1, std::vector<std::vector<float>>& xyb0,
                               std::vector<std::vector<float>>& xyb1) override;

 private:
  GpuButteraugliKernels kernels_;
};

class GpuButteraugliComparator : public ButteraugliComparator {
 public:
  using ButteraugliComparator::ButteraugliComparator;

  // The original image stays on the device; the output image is converted to linear RGB there.
  void Compare(const OutputImage& img) final;
  // Also builds the per-block lists for the device block-zeroing search (once per image).
  void StartBlockComparisons() final;
  std::optional<BlockReferenceData> blockReferenceData() const final;

 protected:
  // Computes the distance map of img against rgb_orig_opsin_ (width_ * height_ values) on the device.
  virtual void compareOnDevice(const OutputImage& img, float* distmap) = 0;

 private:
  bool block_lists_ready_ = false;
  std::vector<float> opsin_blocks_;       // [RR..RRGG..GGBB..BB] per block
  std::vector<float> mask_scale_blocks_;  // [RGB] per block
};

}  // namespace guetzli

#endif  // GUETZLI_BACKEND_GPU_GPU_COMPARATOR_H_

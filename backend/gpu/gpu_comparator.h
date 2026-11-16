// Butteraugli on a GPU. OpenCL and CUDA differ only in the kernel launchers (GpuButteraugliKernels).
#ifndef GUETZLI_BACKEND_GPU_GPU_COMPARATOR_H_
#define GUETZLI_BACKEND_GPU_GPU_COMPARATOR_H_

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "backend/gpu/gpu_kernels.h"
#include "core/quality/butteraugli_comparator.h"

namespace guetzli {

// Images larger than 100x100 run on the device; smaller ones, and MaskHighIntensityChange, use the
// reference code in butteraugli.h.
class GpuButteraugliOps final : public ButteraugliOps {
 public:
  explicit GpuButteraugliOps(std::shared_ptr<GpuButteraugliKernels> kernels) : kernels_(std::move(kernels)) {}

  void opsinDynamicsImage(size_t xsize, size_t ysize, std::vector<std::vector<float>>& rgb) override;
  void mask(const std::vector<std::vector<float>>& xyb0, const std::vector<std::vector<float>>& xyb1, size_t xsize,
            size_t ysize, std::vector<std::vector<float>>* mask, std::vector<std::vector<float>>* mask_dc) override;
  void maskHighIntensityChange(size_t xsize, size_t ysize, const std::vector<std::vector<float>>& c0,
                               const std::vector<std::vector<float>>& c1, std::vector<std::vector<float>>& xyb0,
                               std::vector<std::vector<float>>& xyb1) override;

 private:
  std::shared_ptr<GpuButteraugliKernels> kernels_;
};

class GpuButteraugliComparator final : public ButteraugliComparator {
 public:
  GpuButteraugliComparator(int width, int height, const std::vector<uint8_t>* rgb, float target_distance,
                           ButteraugliOps& ops, std::shared_ptr<GpuButteraugliKernels> kernels);
  ~GpuButteraugliComparator() override;
  GpuButteraugliComparator(const GpuButteraugliComparator&) = delete;
  GpuButteraugliComparator& operator=(const GpuButteraugliComparator&) = delete;
  GpuButteraugliComparator(GpuButteraugliComparator&&) = delete;
  GpuButteraugliComparator& operator=(GpuButteraugliComparator&&) = delete;

  // The original image stays on the device; the output image is converted to linear RGB there.
  void compare(const OutputImage& img) override;
  // Also builds the per-block lists for the device block-zeroing search (once per image).
  void startBlockComparisons() override;
  [[nodiscard]] std::optional<BlockReferenceData> blockReferenceData() const override;

 private:
  std::shared_ptr<GpuButteraugliKernels> kernels_;
  std::unique_ptr<GpuCompareSession> session_;  // created on the first Compare()
  bool block_lists_ready_ = false;
  std::vector<float> opsin_blocks_;       // [RR..RRGG..GGBB..BB] per block
  std::vector<float> mask_scale_blocks_;  // [RGB] per block
};

class GpuComparatorFactory final : public IComparatorFactory {
 public:
  explicit GpuComparatorFactory(const std::shared_ptr<GpuButteraugliKernels>& kernels)
      : kernels_(kernels), ops_(kernels) {}

  std::unique_ptr<Comparator> create(int width, int height, const std::vector<uint8_t>* rgb,
                                     float target_distance) override;

 private:
  std::shared_ptr<GpuButteraugliKernels> kernels_;
  GpuButteraugliOps ops_;
};

}  // namespace guetzli

#endif  // GUETZLI_BACKEND_GPU_GPU_COMPARATOR_H_

#ifdef __USE_CUDA__

#include "backend/cuda/cuda_backend.h"

#include <memory>

#include "backend/gpu/gpu_block_zeroing.h"
#include "backend/gpu/gpu_comparator.h"
#include "backend/gpu/gpu_dct_ops.h"
#include "clguetzli/cuguetzli.h"

namespace guetzli {

namespace {

class CudaButteraugliComparator final : public GpuButteraugliComparator {
 public:
  using GpuButteraugliComparator::GpuButteraugliComparator;
  ~CudaButteraugliComparator() override { cuCompareDestroy(ctx_); }

 private:
  void compareOnDevice(const OutputImage& img, float* distmap) override {
    if (!ctx_) {
      ctx_ = cuCompareCreate(rgb_orig_opsin_[0].data(), rgb_orig_opsin_[1].data(), rgb_orig_opsin_[2].data(), width_,
                             height_);
    }
    cuCompare(ctx_, img, kButteraugliStep, distmap);
  }

  CuCompareContext* ctx_ = nullptr;
};

class CudaComparatorFactory final : public IComparatorFactory {
 public:
  std::unique_ptr<Comparator> create(int width, int height, const std::vector<uint8_t>* rgb, float target_distance,
                                     ProcessStats* stats) override {
    return std::make_unique<CudaButteraugliComparator>(width, height, rgb, target_distance, stats, ops_);
  }

 private:
  GpuButteraugliOps ops_{GpuButteraugliKernels{cuOpsinDynamicsImage, cuMask}};
};

}  // namespace

Backend makeCudaBackend() {
  Backend backend;
  backend.kind = BackendKind::Cuda;
  backend.comparators = std::make_unique<CudaComparatorFactory>();
  backend.block_zeroing = std::make_unique<GpuBlockZeroingSearch>(cuComputeBlockZeroingOrder);
  backend.dct = std::make_unique<GpuDctOps>(
      GpuDctKernels{cuCopyFromJpegComponent, cuApplyGlobalQuantization, cuComponentsToPixels});
  return backend;
}

}  // namespace guetzli

#endif  // __USE_CUDA__

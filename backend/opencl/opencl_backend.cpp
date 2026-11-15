#ifdef __USE_OPENCL__

#include "backend/opencl/opencl_backend.h"

#include <memory>

#include "backend/gpu/gpu_block_zeroing.h"
#include "backend/gpu/gpu_comparator.h"
#include "backend/gpu/gpu_dct_ops.h"
#include "clguetzli/clguetzli.h"

namespace guetzli {

namespace {

class OpenClButteraugliComparator final : public GpuButteraugliComparator {
 public:
  using GpuButteraugliComparator::GpuButteraugliComparator;
  ~OpenClButteraugliComparator() override { clCompareDestroy(ctx_); }

 private:
  void compareOnDevice(const OutputImage& img, float* distmap) override {
    if (!ctx_) {
      ctx_ = clCompareCreate(rgb_orig_opsin_[0].data(), rgb_orig_opsin_[1].data(), rgb_orig_opsin_[2].data(), width_,
                             height_);
    }
    clCompare(ctx_, img, kButteraugliStep, distmap);
  }

  ClCompareContext* ctx_ = nullptr;
};

class OpenClComparatorFactory final : public IComparatorFactory {
 public:
  std::unique_ptr<Comparator> create(int width, int height, const std::vector<uint8_t>* rgb, float target_distance,
                                     ProcessStats* stats) override {
    return std::make_unique<OpenClButteraugliComparator>(width, height, rgb, target_distance, stats, ops_);
  }

 private:
  GpuButteraugliOps ops_{GpuButteraugliKernels{clOpsinDynamicsImage, clMask}};
};

}  // namespace

Backend makeOpenClBackend() {
  Backend backend;
  backend.kind = BackendKind::OpenCl;
  backend.comparators = std::make_unique<OpenClComparatorFactory>();
  backend.block_zeroing = std::make_unique<GpuBlockZeroingSearch>(clComputeBlockZeroingOrder);
  backend.dct = std::make_unique<GpuDctOps>(
      GpuDctKernels{clCopyFromJpegComponent, clApplyGlobalQuantization, clComponentsToPixels});
  return backend;
}

}  // namespace guetzli

#endif  // __USE_OPENCL__

// Butteraugli on the CPU, optimized (float) implementation.
#ifndef GUETZLI_BACKEND_CPU_CPU_COMPARATOR_H_
#define GUETZLI_BACKEND_CPU_CPU_COMPARATOR_H_

#include <memory>
#include <vector>

#include "clguetzli/clbutter_comparator.h"
#include "guetzli/butteraugli_comparator.h"

namespace guetzli {

class CpuButteraugliOps final : public ButteraugliOps {
 public:
  void OpsinDynamicsImage(size_t xsize, size_t ysize, std::vector<std::vector<float>>& rgb) override;
  void Mask(const std::vector<std::vector<float>>& xyb0, const std::vector<std::vector<float>>& xyb1, size_t xsize,
            size_t ysize, std::vector<std::vector<float>>* mask, std::vector<std::vector<float>>* mask_dc) override;
  void MaskHighIntensityChange(size_t xsize, size_t ysize, const std::vector<std::vector<float>>& c0,
                               const std::vector<std::vector<float>>& c1, std::vector<std::vector<float>>& xyb0,
                               std::vector<std::vector<float>>& xyb1) override;
};

class CpuButteraugliComparator final : public ButteraugliComparator {
 public:
  CpuButteraugliComparator(int width, int height, const std::vector<uint8_t>* rgb, float target_distance,
                           ProcessStats* stats, ButteraugliOps& ops);

  void Compare(const OutputImage& img) override;

 private:
  ::butteraugli::clButteraugliComparator comparator_;
};

class CpuComparatorFactory final : public IComparatorFactory {
 public:
  std::unique_ptr<Comparator> create(int width, int height, const std::vector<uint8_t>* rgb, float target_distance,
                                     ProcessStats* stats) override;

 private:
  CpuButteraugliOps ops_;
};

}  // namespace guetzli

#endif  // GUETZLI_BACKEND_CPU_CPU_COMPARATOR_H_

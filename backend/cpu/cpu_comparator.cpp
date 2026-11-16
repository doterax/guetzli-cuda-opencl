#include "backend/cpu/cpu_comparator.h"

#include "core/quality/butteraugli.h"

namespace guetzli {

void CpuButteraugliOps::opsinDynamicsImage(size_t xsize, size_t ysize, std::vector<std::vector<float>>& rgb) {
  ::butteraugli::opsinDynamicsImageOpt(xsize, ysize, rgb);
}

void CpuButteraugliOps::mask(const std::vector<std::vector<float>>& xyb0, const std::vector<std::vector<float>>& xyb1,
                             size_t xsize, size_t ysize, std::vector<std::vector<float>>* mask,
                             std::vector<std::vector<float>>* mask_dc) {
  ::butteraugli::maskOpt(xyb0, xyb1, xsize, ysize, mask, mask_dc);
}

void CpuButteraugliOps::maskHighIntensityChange(size_t xsize, size_t ysize, const std::vector<std::vector<float>>& c0,
                                                const std::vector<std::vector<float>>& c1,
                                                std::vector<std::vector<float>>& xyb0,
                                                std::vector<std::vector<float>>& xyb1) {
  ::butteraugli::maskHighIntensityChangeOpt(xsize, ysize, c0, c1, xyb0, xyb1);
}

CpuButteraugliComparator::CpuButteraugliComparator(int width, int height, const std::vector<uint8_t>* rgb,
                                                   float target_distance, ProcessStats* stats, ButteraugliOps& ops)
    : ButteraugliComparator(width, height, rgb, target_distance, stats, ops),
      comparator_(width, height, kButteraugliStep) {}

void CpuButteraugliComparator::compare(const OutputImage& img) {
  std::vector<std::vector<float>> rgb0 = rgb_orig_opsin_;

  std::vector<std::vector<float>> rgb(3, std::vector<float>(width_ * height_));
  img.toLinearRgb(&rgb);
  ops_.opsinDynamicsImage(width_, height_, rgb);
  std::vector<float>().swap(distmap_);
  comparator_.diffmapOpsinDynamicsImageOpt(rgb0, rgb, distmap_);
  distance_ = ::butteraugli::butteraugliScoreFromDiffmap(distmap_);
}

std::unique_ptr<Comparator> CpuComparatorFactory::create(int width, int height, const std::vector<uint8_t>* rgb,
                                                         float target_distance, ProcessStats* stats) {
  return std::make_unique<CpuButteraugliComparator>(width, height, rgb, target_distance, stats, ops_);
}

}  // namespace guetzli

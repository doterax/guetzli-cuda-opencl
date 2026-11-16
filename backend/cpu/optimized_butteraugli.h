/*
* Optimized (float) CPU implementation of the butteraugli metric, written alongside the GPU kernels.
* Used by the CPU backend (backend/cpu).
*
* Author: strongtu@tencent.com
*         ianhuang@tencent.com
*         chriskzhou@tencent.com
*/
#pragma once
#include <vector>
#include "core/quality/butteraugli.h"

namespace butteraugli {

class OptimizedButteraugliComparator : public ButteraugliComparator {
 public:
  OptimizedButteraugliComparator(size_t xsize, size_t ysize, int step);

  // Optimized counterpart of DiffmapOpsinDynamicsImage(); xyb0 and xyb1 are mutated.
  virtual void diffmapOpsinDynamicsImageOpt(std::vector<std::vector<float>>& xyb0,
                                            std::vector<std::vector<float>>& xyb1, std::vector<float>& result);

  virtual void blockDiffMapOpt(const std::vector<std::vector<float>>& rgb0, const std::vector<std::vector<float>>& rgb1,
                               std::vector<float>* block_diff_dc, std::vector<float>* block_diff_ac);

  virtual void edgeDetectorMapOpt(const std::vector<std::vector<float>>& rgb0,
                                  const std::vector<std::vector<float>>& rgb1, std::vector<float>* edge_detector_map);

  virtual void edgeDetectorLowFreqOpt(const std::vector<std::vector<float>>& rgb0,
                                      const std::vector<std::vector<float>>& rgb1, std::vector<float>* block_diff_ac);

  virtual void combineChannelsOpt(const std::vector<std::vector<float>>& scale_xyb,
                                  const std::vector<std::vector<float>>& scale_xyb_dc,
                                  const std::vector<float>& block_diff_dc, const std::vector<float>& block_diff_ac,
                                  const std::vector<float>& edge_detector_map, std::vector<float>* result);
};

// Optimized counterparts of the butteraugli.h functions of the same name without "Opt".
void opsinDynamicsImageOpt(size_t xsize, size_t ysize, std::vector<std::vector<float>>& rgb);
void maskOpt(const std::vector<std::vector<float>>& xyb0, const std::vector<std::vector<float>>& xyb1, size_t xsize,
             size_t ysize, std::vector<std::vector<float>>* mask, std::vector<std::vector<float>>* mask_dc);
void maskHighIntensityChangeOpt(size_t xsize, size_t ysize, const std::vector<std::vector<float>>& c0,
                                const std::vector<std::vector<float>>& c1, std::vector<std::vector<float>>& xyb0,
                                std::vector<std::vector<float>>& xyb1);

}  // namespace butteraugli

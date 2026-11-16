/*
 * Copyright 2016 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef GUETZLI_BUTTERAUGLI_COMPARATOR_H_
#define GUETZLI_BUTTERAUGLI_COMPARATOR_H_

#include <cstddef>
#include <vector>

#include "core/quality/comparator.h"
#include "core/jpeg/jpeg_data.h"
#include "core/image/output_image.h"
#include "core/encoder/stats.h"

namespace guetzli {

constexpr int kButteraugliStep = 3;

// The butteraugli image operations whose implementation depends on the backend (optimized CPU code,
// GPU kernels, or the reference code in butteraugli.h). Owned by the backend; must outlive its comparators.
class ButteraugliOps {
 public:
  virtual ~ButteraugliOps() = default;

  virtual void opsinDynamicsImage(size_t xsize, size_t ysize, std::vector<std::vector<float>>& rgb) = 0;
  virtual void mask(const std::vector<std::vector<float>>& xyb0, const std::vector<std::vector<float>>& xyb1,
                    size_t xsize, size_t ysize, std::vector<std::vector<float>>* mask,
                    std::vector<std::vector<float>>* mask_dc) = 0;
  virtual void maskHighIntensityChange(size_t xsize, size_t ysize, const std::vector<std::vector<float>>& c0,
                                       const std::vector<std::vector<float>>& c1, std::vector<std::vector<float>>& xyb0,
                                       std::vector<std::vector<float>>& xyb1) = 0;
};

// Butteraugli comparator. The whole-image Compare() is implemented by each backend; everything else is
// shared and runs the image operations through the backend's ButteraugliOps.
class ButteraugliComparator : public Comparator {
 public:
  ButteraugliComparator(const int width, const int height, const std::vector<uint8_t>* rgb, const float target_distance,
                        ProcessStats* stats, ButteraugliOps& ops);

  void startBlockComparisons() override;
  void finishBlockComparisons() override;

  void switchBlock(int block_x, int block_y, int factor_x, int factor_y) override;

  double compareBlock(const OutputImage& img, int off_x, int off_y, const coeff_t* candidate_block,
                      const int comp_mask) const override;

  double scoreOutputSize(int size) const override;

  bool distanceOk(double target_mul) const override { return distance_ <= target_mul * target_distance_; }

  const std::vector<float> distmap() const override { return distmap_; }
  float distmapAggregate() const override { return distance_; }

  float blockErrorLimit() const override;

  void computeBlockErrorAdjustmentWeights(int direction, int max_block_dist, double target_mul, int factor_x,
                                          int factor_y, const std::vector<float>& distmap,
                                          std::vector<float>* block_weight) override;

 protected:
  const int width_;
  const int height_;
  const float target_distance_;
  const std::vector<uint8_t>& rgb_orig_;
  ButteraugliOps& ops_;
  int block_x_;
  int block_y_;
  int factor_x_;
  int factor_y_;
  // Opsin-dynamics image of the original (single-precision sRGB table): the reference of every Compare().
  std::vector<std::vector<float>> rgb_orig_opsin_;
  // Mask(original, original); computed by the first StartBlockComparisons() and reused.
  std::vector<std::vector<float>> mask_xyz_orig_;
  std::vector<std::vector<float>> mask_xyz_;  // valid between Start/FinishBlockComparisons()
  std::vector<std::vector<std::vector<float>>> per_block_pregamma_;
  float distance_;
  std::vector<float> distmap_;
  ProcessStats* stats_;
};

}  // namespace guetzli

#endif  // GUETZLI_BUTTERAUGLI_COMPARATOR_H_

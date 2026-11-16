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

#include "core/quality/butteraugli_comparator.h"

#include <algorithm>

#include "core/quality/butteraugli.h"
#include "core/quality/opsin_block.h"
#include "core/image/gamma_correct.h"
#include "core/quality/score.h"

namespace guetzli {

ButteraugliComparator::ButteraugliComparator(const int width, const int height, const std::vector<uint8_t>* rgb,
                                             const float target_distance, ButteraugliOps& ops)
    : width_(width), height_(height), target_distance_(target_distance), rgb_orig_(*rgb), ops_(ops) {
  rgb_orig_opsin_.resize(3);
  rgb_orig_opsin_[0].resize(width * height);
  rgb_orig_opsin_[1].resize(width * height);
  rgb_orig_opsin_[2].resize(width * height);

  const float* lut = srgb8ToLinearTableF();

  for (int c = 0; c < 3; ++c) {
    for (int y = 0, ix = 0; y < height_; ++y) {
      for (int x = 0; x < width_; ++x, ++ix) {
        rgb_orig_opsin_[c][ix] = lut[rgb_orig_[3 * ix + c]];
      }
    }
  }
  ops_.opsinDynamicsImage(width_, height_, rgb_orig_opsin_);
}

void ButteraugliComparator::startBlockComparisons() {
  // Mask(orig, orig) depends only on the original image: compute it once and
  // reuse it for every block-comparison round (SelectFrequencyMasking runs 2-3x).
  if (mask_xyz_orig_.empty()) {
    std::vector<std::vector<float>> dummy(3);
    ops_.mask(rgb_orig_opsin_, rgb_orig_opsin_, width_, height_, &mask_xyz_orig_, &dummy);
  }
  mask_xyz_ = mask_xyz_orig_;  // FinishBlockComparisons() clears mask_xyz_
}

void ButteraugliComparator::finishBlockComparisons() {
  mask_xyz_.clear();
}

void ButteraugliComparator::switchBlock(int block_x, int block_y, int factor_x, int factor_y) {
  block_x_ = block_x;
  block_y_ = block_y;
  factor_x_ = factor_x;
  factor_y_ = factor_y;
  per_block_pregamma_.resize(factor_x_ * factor_y_);
  const double* lut = srgb8ToLinearTable();
  for (int off_y = 0, bx = 0; off_y < factor_y_; ++off_y) {
    for (int off_x = 0; off_x < factor_x_; ++off_x, ++bx) {
      per_block_pregamma_[bx].resize(3, std::vector<float>(kDctBlockSize));
      int block_xx = block_x_ * factor_x_ + off_x;
      int block_yy = block_y_ * factor_y_ + off_y;
      for (int iy = 0, i = 0; iy < 8; ++iy) {
        for (int ix = 0; ix < 8; ++ix, ++i) {
          int x = std::min(8 * block_xx + ix, width_ - 1);
          int y = std::min(8 * block_yy + iy, height_ - 1);
          int px = y * width_ + x;
          for (int c = 0; c < 3; ++c) {
            per_block_pregamma_[bx][c][i] = lut[rgb_orig_[3 * px + c]];
          }
        }
      }
      ops_.opsinDynamicsImage(8, 8, per_block_pregamma_[bx]);
    }
  }
}

double ButteraugliComparator::compareBlock(const OutputImage& img, int off_x, int off_y, const coeff_t* candidate_block,
                                           const int comp_mask) const {
  int block_x = block_x_ * factor_x_ + off_x;
  int block_y = block_y_ * factor_y_ + off_y;
  int xmin = 8 * block_x;
  int ymin = 8 * block_y;
  int block_ix = off_y * factor_x_ + off_x;
  const std::vector<std::vector<float>>& rgb0_c = per_block_pregamma_[block_ix];

  std::vector<std::vector<float>> rgb1_c(3, std::vector<float>(kDctBlockSize));
  img.toLinearRgb(xmin, ymin, 8, 8, &rgb1_c);
  ops_.opsinDynamicsImage(8, 8, rgb1_c);

  std::vector<std::vector<float>> rgb0 = rgb0_c;
  std::vector<std::vector<float>> rgb1 = rgb1_c;

  ops_.maskHighIntensityChange(8, 8, rgb0_c, rgb1_c, rgb0, rgb1);

  double b0[3 * kDctBlockSize];
  double b1[3 * kDctBlockSize];
  for (int c = 0; c < 3; ++c) {
    for (int ix = 0; ix < kDctBlockSize; ++ix) {
      b0[c * kDctBlockSize + ix] = rgb0[c][ix];
      b1[c * kDctBlockSize + ix] = rgb1[c][ix];
    }
  }
  double diff_xyz_dc[3] = {0.0};
  double diff_xyz_ac[3] = {0.0};
  double diff_xyz_edge_dc[3] = {0.0};
  ::butteraugli::butteraugliBlockDiff(b0, b1, diff_xyz_dc, diff_xyz_ac, diff_xyz_edge_dc);

  double scale[3];
  for (int c = 0; c < 3; ++c) {
    scale[c] = mask_xyz_[c][ymin * width_ + xmin];
  }

  static const double kEdgeWeight = 0.05;

  double diff = 0.0;
  double diff_edge = 0.0;
  for (int c = 0; c < 3; ++c) {
    diff += diff_xyz_dc[c] * scale[c];
    diff += diff_xyz_ac[c] * scale[c];
    diff_edge += diff_xyz_edge_dc[c] * scale[c];
  }
  return std::sqrt((1 - kEdgeWeight) * diff + kEdgeWeight * diff_edge);
}

float ButteraugliComparator::blockErrorLimit() const {
  return target_distance_;
}

void ButteraugliComparator::computeBlockErrorAdjustmentWeights(int direction, int max_block_dist, double target_mul,
                                                               int factor_x, int factor_y,
                                                               const std::vector<float>& block_max8,
                                                               std::vector<float>* block_weight) {
  const double target_distance = target_distance_ * target_mul;
  const int sizex = 8 * factor_x;
  const int sizey = 8 * factor_y;
  const int block_width = (width_ + sizex - 1) / sizex;
  const int block_height = (height_ + sizey - 1) / sizey;
  // A factor_x x factor_y group of 8x8 blocks covers the same pixels as the sizex x sizey block.
  const int width8 = (width_ + 7) / 8;
  const int height8 = (height_ + 7) / 8;
  std::vector<float> max_dist_per_block(block_width * block_height);
  for (int block_y = 0; block_y < block_height; ++block_y) {
    for (int block_x = 0; block_x < block_width; ++block_x) {
      int block_ix = block_y * block_width + block_x;
      int x_max = std::min(width8, factor_x * (block_x + 1));
      int y_max = std::min(height8, factor_y * (block_y + 1));
      float max_dist = 0.0;
      for (int y = factor_y * block_y; y < y_max; ++y) {
        for (int x = factor_x * block_x; x < x_max; ++x) {
          max_dist = std::max(max_dist, block_max8[y * width8 + x]);
        }
      }
      max_dist_per_block[block_ix] = max_dist;
    }
  }
  for (int block_y = 0; block_y < block_height; ++block_y) {
    for (int block_x = 0; block_x < block_width; ++block_x) {
      int block_ix = block_y * block_width + block_x;
      auto max_local_dist = static_cast<float>(target_distance);
      int x_min = std::max(0, block_x - max_block_dist);
      int y_min = std::max(0, block_y - max_block_dist);
      int x_max = std::min(block_width, block_x + 1 + max_block_dist);
      int y_max = std::min(block_height, block_y + 1 + max_block_dist);
      for (int y = y_min; y < y_max; ++y) {
        for (int x = x_min; x < x_max; ++x) {
          max_local_dist = std::max(max_local_dist, max_dist_per_block[y * block_width + x]);
        }
      }
      if (direction > 0) {
        if (max_dist_per_block[block_ix] <= target_distance && max_local_dist <= 1.1 * target_distance) {
          (*block_weight)[block_ix] = 1.0;
        }
      } else {
        constexpr double kLocalMaxWeight = 0.5;
        if (max_dist_per_block[block_ix] <=
            (1 - kLocalMaxWeight) * target_distance + kLocalMaxWeight * max_local_dist) {
          continue;
        }
        for (int y = y_min; y < y_max; ++y) {
          for (int x = x_min; x < x_max; ++x) {
            int d = std::max(std::abs(y - block_y), std::abs(x - block_x));
            int ix = y * block_width + x;
            (*block_weight)[ix] = std::max<float>((*block_weight)[ix], 1.0f / (d + 1.0f));
          }
        }
      }
    }
  }
}

void ButteraugliComparator::setDistanceFromDistmap() {
  const int width8 = (width_ + 7) / 8;
  const int height8 = (height_ + 7) / 8;
  block_max8_.assign(static_cast<size_t>(width8) * height8, 0.0f);
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      float& m = block_max8_[((y / 8) * width8) + (x / 8)];
      m = std::max(m, distmap_[(y * width_) + x]);
    }
  }
  distance_ = ::butteraugli::butteraugliScoreFromDiffmap(distmap_);
}

double ButteraugliComparator::scoreOutputSize(int size) const {
  return scoreJpeg(distance_, size, target_distance_);
}

}  // namespace guetzli

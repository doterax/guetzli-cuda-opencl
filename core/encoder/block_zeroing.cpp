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

#include "core/encoder/block_zeroing.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <utility>

#include "core/util/narrow_cast.h"
#include "core/util/fast_log.h"
#include "core/jpeg/jpeg_data.h"

namespace guetzli {

namespace {

constexpr int kBlockSize = kBlockZeroingBlockSize;

// REQUIRES: block[c*64...(c*64+63)] is all zero if (comp_mask & (1<<c)) == 0.
void computeBlockZeroingOrder(const coeff_t block[kBlockSize], const coeff_t orig_block[kBlockSize], const int block_x,
                              const int block_y, const int factor_x, const int factor_y, const uint8_t comp_mask,
                              OutputImage* img, Comparator* comparator, int greedy_lookahead, bool new_zeroing_model,
                              std::vector<CoeffData>* output_order) {
  static const uint8_t kOldCsf[kDctBlockSize] = {
      10, 10, 20, 40, 60, 70, 80, 90, 10, 20, 30, 60, 70, 80, 90, 90, 20, 30, 60, 70, 80, 90,
      90, 90, 40, 60, 70, 80, 90, 90, 90, 90, 60, 70, 80, 90, 90, 90, 90, 90, 70, 80, 90, 90,
      90, 90, 90, 90, 80, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90,
  };
  static const double kWeight[3] = {1.0, 0.22, 0.20};
#include "core/encoder/order.inc"
  std::vector<std::pair<int, float>> input_order;
  for (int c = 0; c < 3; ++c) {
    if ((comp_mask & (1 << c)) == 0) {
      continue;
    }
    for (int k = 1; k < kDctBlockSize; ++k) {
      int idx = c * kDctBlockSize + k;
      if (block[idx] != 0) {
        float score;
        if (new_zeroing_model) {
          score = std::abs(orig_block[idx]) * kCsf[idx] + kBias[idx];
        } else {
          score =
              static_cast<float>((std::abs(orig_block[idx]) - kJpegZigZagOrder[k] / 64.0) * kWeight[c] / kOldCsf[k]);
        }
        input_order.emplace_back(idx, score);
      }
    }
  }
  std::sort(input_order.begin(), input_order.end(),
            [](const std::pair<int, float>& a, const std::pair<int, float>& b) { return a.second < b.second; });
  coeff_t processed_block[kBlockSize];
  memcpy(processed_block, block, sizeof(processed_block));
  comparator->switchBlock(block_x, block_y, factor_x, factor_y);
  while (!input_order.empty()) {
    float best_err = 1e17f;
    int best_i = 0;
    for (size_t i = 0; i < std::min<size_t>(greedy_lookahead, input_order.size()); ++i) {
      coeff_t candidate_block[kBlockSize];
      memcpy(candidate_block, processed_block, sizeof(candidate_block));
      const int idx = input_order[i].first;
      candidate_block[idx] = 0;
      for (int c = 0; c < 3; ++c) {
        if ((comp_mask & (1 << c)) != 0) {
          img->component(c).setCoeffBlock(block_x, block_y, &candidate_block[c * kDctBlockSize]);
        }
      }
      float max_err = 0;
      for (int iy = 0; iy < factor_y; ++iy) {
        for (int ix = 0; ix < factor_x; ++ix) {
          int block_xx = block_x * factor_x + ix;
          int block_yy = block_y * factor_y + iy;
          if (8 * block_xx < img->width() && 8 * block_yy < img->height()) {
            auto err = static_cast<float>(comparator->compareBlock(*img, ix, iy, candidate_block, comp_mask));
            max_err = std::max(max_err, err);
          }
        }
      }
      if (max_err < best_err) {
        best_err = max_err;
        best_i = i;
      }
    }
    int idx = input_order[best_i].first;
    processed_block[idx] = 0;
    input_order.erase(input_order.begin() + best_i);
    output_order->push_back({.idx = idx, .block_err = best_err});
    for (int c = 0; c < 3; ++c) {
      if ((comp_mask & (1 << c)) != 0) {
        img->component(c).setCoeffBlock(block_x, block_y, &processed_block[c * kDctBlockSize]);
      }
    }
    if (best_err >= comparator->blockErrorLimit()) {
      // The input_order is an ascent vector, break when best_err exceed the error limit.
      break;
    }
  }
  // Make the block error values monotonic.
  float min_err = 1e10;
  const int size = narrow_cast<int>(output_order->size());
  for (int i = size - 1; i >= 0; --i) {
    min_err = std::min(min_err, (*output_order)[i].block_err);
    (*output_order)[i].block_err = min_err;
  }
  // Cut off at the block error limit.
  size_t num = 0;
  while (num < output_order->size() && (*output_order)[num].block_err <= comparator->blockErrorLimit()) {
    ++num;
  }
  output_order->resize(num);
  // Restore *img to the same state as it was at the start of this function.
  for (int c = 0; c < 3; ++c) {
    if ((comp_mask & (1 << c)) != 0) {
      img->component(c).setCoeffBlock(block_x, block_y, &block[c * kDctBlockSize]);
    }
  }
}

}  // namespace

std::vector<CoeffData> CpuBlockZeroingSearch::run(const BlockZeroingInput& in) {
  OutputImage* img = &in.img;
  const JpegData& jpg = in.jpg;
  const uint8_t comp_mask = in.comp_mask;
  const int last_c = log2FloorNonZero(comp_mask);
  const int factor_x = img->component(last_c).factorX();
  const int factor_y = img->component(last_c).factorY();
  const int block_width = (img->width() + 8 * factor_x - 1) / (8 * factor_x);
  const int block_height = (img->height() + 8 * factor_y - 1) / (8 * factor_y);
  const int num_blocks = block_width * block_height;

  std::vector<CoeffData> output_order(num_blocks * kBlockSize);
  for (int block_y = 0, block_ix = 0; block_y < block_height; ++block_y) {
    for (int block_x = 0; block_x < block_width; ++block_x, ++block_ix) {
      coeff_t block[kBlockSize] = {0};
      coeff_t orig_block[kBlockSize] = {0};
      for (int c = 0; c < 3; ++c) {
        if ((comp_mask & (1 << c)) != 0) {
          assert(img->component(c).factor_x() == factor_x);
          assert(img->component(c).factor_y() == factor_y);
          img->component(c).getCoeffBlock(block_x, block_y, &block[c * kDctBlockSize]);
          const JpegComponent& comp = jpg.components[c];
          int jpg_block_ix = block_y * comp.width_in_blocks + block_x;
          memcpy(&orig_block[c * kDctBlockSize], &comp.coeffs[jpg_block_ix * kDctBlockSize],
                 kDctBlockSize * sizeof(orig_block[0]));
        }
      }

      std::vector<CoeffData> block_order;
      computeBlockZeroingOrder(block, orig_block, block_x, block_y, factor_x, factor_y, comp_mask, img, &in.comparator,
                               in.greedy_lookahead, in.new_zeroing_model, &block_order);

      CoeffData* p = &output_order[block_ix * kBlockSize];
      for (size_t i = 0; i < block_order.size(); i++) {
        p[i].idx = block_order[i].idx;
        p[i].block_err = block_order[i].block_err;
      }
    }
  }
  return output_order;
}

}  // namespace guetzli

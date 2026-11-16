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

#include "core/encoder/frequency_masking.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <utility>

#include "core/encoder/size_estimator.h"
#include "core/util/narrow_cast.h"
#include "core/encoder/debug_print.h"
#include "core/util/fast_log.h"
#include "core/jpeg/quantize.h"

namespace guetzli {

namespace {

// The block grid of the components in comp_mask (they all have the subsampling of the last one).
struct BlockGrid {
  int factor_x;
  int factor_y;
  int block_width;
  int block_height;
  int num_blocks;
};

BlockGrid blockGrid(const OutputImage& img, int last_c) {
  BlockGrid g;
  g.factor_x = img.component(last_c).factorX();
  g.factor_y = img.component(last_c).factorY();
  g.block_width = (img.width() + 8 * g.factor_x - 1) / (8 * g.factor_x);
  g.block_height = (img.height() + 8 * g.factor_y - 1) / (8 * g.factor_y);
  g.num_blocks = g.block_width * g.block_height;
  return g;
}

}  // namespace

ZeroingCandidates collectZeroingCandidates(std::span<const CoeffData> orders, int num_blocks, float block_error_limit) {
  ZeroingCandidates out;
  out.offsets.resize(num_blocks + 1);
  for (int block_ix = 0; block_ix < num_blocks; ++block_ix) {
    const CoeffData* p = &orders[static_cast<size_t>(block_ix) * kBlockZeroingBlockSize];
    out.offsets[block_ix] = narrow_cast<int>(out.coeffs.size());
    for (int i = 0; i < kBlockZeroingBlockSize; i++) {
      if (p[i].block_err > 0 && p[i].block_err <= block_error_limit) {
        out.coeffs.push_back(static_cast<uint8_t>(p[i].idx));
        out.errors.push_back(p[i].block_err);
      }
    }
  }
  out.offsets[num_blocks] = narrow_cast<int>(out.coeffs.size());
  return out;
}

void FrequencyMasking::run(const JpegData& jpg, OutputImage& img, const uint8_t comp_mask, const double target_mul,
                           bool stop_early) {
  const int last_c = log2FloorNonZero(comp_mask);
  if (static_cast<size_t>(last_c) >= jpg.components.size()) return;
  const BlockGrid grid = blockGrid(img, last_c);

  comparator_->startBlockComparisons();
  const BlockZeroingInput search_input{.jpg = jpg,
                                       .img = img,
                                       .comp_mask = comp_mask,
                                       .comparator = *comparator_,
                                       .greedy_lookahead = params_->zeroing_greedy_lookahead,
                                       .new_zeroing_model = params_->new_zeroing_model};
  const std::vector<CoeffData> orders = search_->run(search_input);
  const ZeroingCandidates candidates =
      collectZeroingCandidates(orders, grid.num_blocks, comparator_->blockErrorLimit());
  comparator_->finishBlockComparisons();

  if (candidates.coeffs.empty()) return;  // nothing to do
  adjust(jpg, img, comp_mask, target_mul, stop_early, candidates);
}

void FrequencyMasking::adjust(const JpegData& jpg, OutputImage& img, const uint8_t comp_mask, const double target_mul,
                              bool stop_early, const ZeroingCandidates& candidates) {
  const std::vector<int>& candidate_coeff_offsets = candidates.offsets;
  const std::vector<uint8_t>& candidate_coeffs = candidates.coeffs;
  const std::vector<float>& candidate_coeff_errors = candidates.errors;
  const int width = img.width();
  const int height = img.height();
  const BlockGrid grid = blockGrid(img, log2FloorNonZero(comp_mask));
  const int factor_x = grid.factor_x;
  const int factor_y = grid.factor_y;
  const int block_width = grid.block_width;
  const int block_height = grid.block_height;
  const int num_blocks = grid.num_blocks;

  JpegSizeEstimator size_estimator = [&] {
    JpegData jpg_out = jpg;
    img.saveToJpegData(&jpg_out);
    return JpegSizeEstimator(jpg_out, jpg.components.size(), params_->clear_metadata);
  }();
  const int base_size = static_cast<int>(size_estimator.estimate());
  int prev_size = base_size;

  std::vector<float> max_block_error(num_blocks);
  std::vector<int> last_indexes(num_blocks);

  bool first_up_iter = true;
  for (int direction : {1, -1}) {
    for (;;) {
      if (stop_early && direction == -1) {
        if (prev_size > 1.01 * selector_->best().size()) {
          // If we are down-adjusting the error, the output size will only keep
          // increasing.
          // TODO(user): Do this check always by comparing only the size
          // of the currently processed components.
          break;
        }
      }
      std::vector<std::pair<int, float>> global_order;
      int blocks_to_change;
      std::vector<float> block_weight;
      for (int rblock = 1; rblock <= 4; ++rblock) {
        block_weight = std::vector<float>(num_blocks);
        std::vector<float> distmap(width * height);
        if (!first_up_iter) {
          distmap = comparator_->distmap();
        }
        comparator_->computeBlockErrorAdjustmentWeights(direction, rblock, target_mul, factor_x, factor_y, distmap,
                                                        &block_weight);
        global_order.clear();
        blocks_to_change = 0;
        for (int block_y = 0, block_ix = 0; block_y < block_height; ++block_y) {
          for (int block_x = 0; block_x < block_width; ++block_x, ++block_ix) {
            const int last_index = last_indexes[block_ix];
            const int offset = candidate_coeff_offsets[block_ix];
            const int num_candidates = candidate_coeff_offsets[block_ix + 1] - offset;
            const float* candidate_errors = candidate_coeff_errors.data() + offset;
            const float max_err = max_block_error[block_ix];
            if (block_weight[block_ix] == 0) {
              continue;
            }
            if (direction > 0) {
              for (size_t i = last_index; std::cmp_less(i, num_candidates); ++i) {
                float val = ((candidate_errors[i] - max_err) / block_weight[block_ix]);
                global_order.emplace_back(block_ix, val);
              }
              blocks_to_change += (last_index < num_candidates ? 1 : 0);
            } else {
              for (int i = last_index - 1; i >= 0; --i) {
                float val = ((max_err - candidate_errors[i]) / block_weight[block_ix]);
                global_order.emplace_back(block_ix, val);
              }
              blocks_to_change += (last_index > 0 ? 1 : 0);
            }
          }
        }
        if (!global_order.empty()) {
          // If we found something to adjust with the current block adjustment
          // radius, we can stop and adjust the blocks we have.
          break;
        }
      }

      if (global_order.empty()) {
        break;
      }

      std::sort(global_order.begin(), global_order.end(),
                [](const std::pair<int, float>& a, const std::pair<int, float>& b) { return a.second < b.second; });

      double rel_size_delta = direction > 0 ? 0.01 : 0.0005;
      if (direction > 0 && comparator_->distanceOk(1.0)) {
        rel_size_delta = 0.05;
      }
      double min_size_delta = base_size * rel_size_delta;

      float coeffs_to_change_per_block = direction > 0 ? 2.0f : factor_x * factor_y * 0.2f;
      int min_coeffs_to_change = coeffs_to_change_per_block * blocks_to_change;

      if (first_up_iter) {
        const float limit = 0.75f * comparator_->blockErrorLimit();
        auto it = std::partition_point(global_order.begin(), global_order.end(),
                                       [=](const std::pair<int, float>& a) { return a.second < limit; });
        min_coeffs_to_change = std::max<int>(min_coeffs_to_change, it - global_order.begin());
        first_up_iter = false;
      }

      std::set<int> changed_blocks;
      float val_threshold = 0.0;
      int changed_coeffs = 0;
      int est_jpg_size = prev_size;
      for (size_t i = 0; i < global_order.size(); ++i) {
        const int block_ix = global_order[i].first;
        // Block coordinates in the grid of the components in comp_mask, which all have its subsampling.
        const int block_x = block_ix % block_width;
        const int block_y = block_ix / block_width;
        const int last_idx = last_indexes[block_ix];
        const uint8_t* block_candidates = candidate_coeffs.data() + candidate_coeff_offsets[block_ix];
        const int idx = block_candidates[last_idx + std::min(direction, 0)];
        const int c = idx / kDctBlockSize;
        const int k = idx % kDctBlockSize;
        const int* quant = img.component(c).quant();
        const JpegComponent& comp = jpg.components[c];
        assert(img.component(c).factor_x() == factor_x && img.component(c).factor_y() == factor_y);

        const int jpg_block_ix = block_y * comp.width_in_blocks + block_x;
        const int newval = direction > 0 ? 0 : quantize(comp.coeffs[jpg_block_ix * kDctBlockSize + k], quant[k]);
        coeff_t old_block[kDctBlockSize];
        img.component(c).getCoeffBlock(block_x, block_y, old_block);
        coeff_t block[kDctBlockSize];
        std::memcpy(block, old_block, sizeof(block));
        block[k] = newval;
        size_estimator.replaceBlock(c, old_block, block, quant);
        img.component(c).setCoeffBlock(block_x, block_y, block);
        last_indexes[block_ix] += direction;
        changed_blocks.insert(block_ix);
        val_threshold = global_order[i].second;
        ++changed_coeffs;
        constexpr int kEntropyCodeUpdateFreq = 10;
        if (i % kEntropyCodeUpdateFreq == 0) {
          size_estimator.refreshEntropyCodes();
        }
        est_jpg_size = static_cast<int>(size_estimator.estimate());
        if (changed_coeffs > min_coeffs_to_change && std::abs(est_jpg_size - prev_size) > min_size_delta) {
          break;
        }
      }
      size_t global_order_size = global_order.size();
      std::vector<std::pair<int, float>>().swap(global_order);

      for (int i = 0; i < num_blocks; ++i) {
        max_block_error[i] += block_weight[i] * val_threshold * direction;
      }

      ++stats_->counters[kNumItersCnt];
      ++stats_->counters[direction > 0 ? kNumItersUpCnt : kNumItersDownCnt];
      std::string encoded_jpg;
      {
        JpegData jpg_out = jpg;
        img.saveToJpegData(&jpg_out);
        encoded_jpg = selector_->serialize(jpg_out);
      }
      GUETZLI_LOG(stats_,
                  "Iter %2d: %s(%d) %s Coeffs[%d/%zd] "
                  "Blocks[%zd/%d/%d] ValThres[%.4f] Out[%7zd] EstErr[%.2f%%]",
                  stats_->counters[kNumItersCnt], img.frameTypeStr().c_str(), comp_mask, direction > 0 ? "up" : "down",
                  changed_coeffs, global_order_size, changed_blocks.size(), blocks_to_change, num_blocks, val_threshold,
                  encoded_jpg.size(), 100.0 - (100.0 * est_jpg_size) / encoded_jpg.size());
      comparator_->compare(img);
      selector_->offer(encoded_jpg, *comparator_);
      prev_size = est_jpg_size;
    }
  }
}

}  // namespace guetzli

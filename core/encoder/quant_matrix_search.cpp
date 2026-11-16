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

#include "core/encoder/quant_matrix_search.h"

#include <algorithm>
#include <cstring>
#include <string>

#include "core/encoder/debug_print.h"

namespace guetzli {

namespace {

constexpr int kQuantMatrixSize = 3 * kDctBlockSize;

double contrastSensitivity(int k) {
  return 1.0 / (1.0 + kJpegZigZagOrder[k] / 2.0);
}

}  // namespace

bool compareQuantData(const QuantData& a, const QuantData& b) {
  if (a.dist_ok && !b.dist_ok) {
    return true;
  }
  if (!a.dist_ok && b.dist_ok) {
    return false;
  }
  return a.jpg_size < b.jpg_size;
}

int compareQuantMatrices(const int* a, const int* b) {
  int i = 0;
  while (i < kQuantMatrixSize && a[i] == b[i]) {
    ++i;
  }
  if (i == kQuantMatrixSize) {
    return 0;
  }
  if (a[i] < b[i]) {
    for (++i; i < kQuantMatrixSize; ++i) {
      if (a[i] > b[i]) {
        return 2;
      }
    }
    return -1;
  }
  for (++i; i < kQuantMatrixSize; ++i) {
    if (a[i] < b[i]) {
      return 2;
    }
  }
  return 1;
}

double quantMatrixHeuristicScore(const int q[3][kDctBlockSize]) {
  double score = 0.0;
  for (int c = 0; c < 3; ++c) {
    for (int k = 0; k < kDctBlockSize; ++k) {
      score += 0.5 * (q[c][k] - 1.0) * contrastSensitivity(k);
    }
  }
  return score;
}

QuantMatrixGenerator::QuantMatrixGenerator(bool downsample) : downsample_(downsample) {
  for (int k = 0; k < kDctBlockSize; ++k) {
    total_csf_ += 3.0 * contrastSensitivity(k);
  }
}

bool QuantMatrixGenerator::getNext(int q[3][kDctBlockSize]) {
  // This loop should terminate by return. This 1000 iteration limit is just a
  // precaution.
  for (int iter = 0; iter < 1000; iter++) {
    double hscore = 0.0;
    if (hscore_b_ == -1.0) {
      if (hscore_a_ == -1.0) {
        hscore = downsample_ ? 0.0 : total_csf_;
      } else {
        if (hscore_a_ < 5.0 * total_csf_) {
          hscore = hscore_a_ + total_csf_;
        } else {
          hscore = 2 * (hscore_a_ + total_csf_);
        }
      }
      if (hscore > 100 * total_csf_) {
        // We could not find a quantization matrix that creates enough
        // butteraugli error. This can happen if all dct coefficients are
        // close to zero in the original image.
        return false;
      }
    } else if (hscore_b_ == 0.0) {
      return false;
    } else if (hscore_a_ == -1.0) {
      hscore = 0.0;
    } else {
      int lower_q[3][kDctBlockSize];
      int upper_q[3][kDctBlockSize];
      constexpr double kEps = 0.05;
      getQuantMatrixWithHeuristicScore((1 - kEps) * hscore_a_ + kEps * 0.5 * (hscore_a_ + hscore_b_), lower_q);
      getQuantMatrixWithHeuristicScore((1 - kEps) * hscore_b_ + kEps * 0.5 * (hscore_a_ + hscore_b_), upper_q);
      if (compareQuantMatrices(&lower_q[0][0], &upper_q[0][0]) == 0) {
        return false;
      }
      hscore = (hscore_a_ + hscore_b_) * 0.5;
    }
    getQuantMatrixWithHeuristicScore(hscore, q);
    bool retry = false;
    for (auto& quant : quants_) {
      if (compareQuantMatrices(&q[0][0], &quant.q[0][0]) == 0) {
        if (quant.dist_ok) {
          hscore_a_ = hscore;
        } else {
          hscore_b_ = hscore;
        }
        retry = true;
        break;
      }
    }
    if (!retry) {
      return true;
    }
  }
  return false;
}

void QuantMatrixGenerator::add(const QuantData& data) {
  quants_.push_back(data);
  double hscore = quantMatrixHeuristicScore(data.q);
  if (data.dist_ok) {
    hscore_a_ = std::max(hscore_a_, hscore);
  } else {
    hscore_b_ = hscore_b_ == -1.0 ? hscore : std::min(hscore_b_, hscore);
  }
}

void QuantMatrixGenerator::getQuantMatrixWithHeuristicScore(double score, int q[3][kDctBlockSize]) const {
  int level = static_cast<int>(score / total_csf_);
  score -= level * total_csf_;
  for (int k = kDctBlockSize - 1; k >= 0; --k) {
    for (int c = 0; c < 3; ++c) {
      q[c][kJpegNaturalOrder[k]] = 2 * level + (score > 0.0 ? 3 : 1);
    }
    score -= 3.0 * contrastSensitivity(kJpegNaturalOrder[k]);
  }
}

QuantData QuantMatrixSearch::tryMatrix(const JpegData& jpg, const float target_mul, int q[3][kDctBlockSize],
                                       OutputImage& img) {
  QuantData data{};
  std::memcpy(data.q, q, sizeof(data.q));
  img.copyFromJpegData(jpg);
  img.applyGlobalQuantization(data.q);
  std::string encoded_jpg;
  {
    JpegData jpg_out = jpg;
    img.saveToJpegData(&jpg_out);
    encoded_jpg = selector_->serialize(jpg_out);
  }
  GUETZLI_LOG(stats_, "Iter %2d: %s quantization matrix:\n", stats_->counters[kNumItersCnt] + 1,
              img.frameTypeStr().c_str());
  GUETZLI_LOG_QUANT(stats_, q);
  GUETZLI_LOG(stats_, "Iter %2d: %s GQ[%5.2f] Out[%7zd]", stats_->counters[kNumItersCnt] + 1,
              img.frameTypeStr().c_str(), quantMatrixHeuristicScore(q), encoded_jpg.size());
  ++stats_->counters[kNumItersCnt];
  comparator_->compare(img);
  data.dist_ok = comparator_->distanceOk(target_mul);
  data.jpg_size = encoded_jpg.size();
  selector_->offer(encoded_jpg, *comparator_);
  return data;
}

bool QuantMatrixSearch::run(const JpegData& jpg, const bool downsample, int best_q[3][kDctBlockSize],
                            OutputImage& img) {
  QuantMatrixGenerator qgen(downsample);
  // Don't try to go up to exactly the target distance when selecting a
  // quantization matrix, since we will need some slack to do the frequency
  // masking later.
  const float target_mul_high = 0.97f;
  const float target_mul_low = 0.95f;

  QuantData best = tryMatrix(jpg, target_mul_high, best_q, img);
  for (;;) {
    int q_next[3][kDctBlockSize];
    if (!qgen.getNext(q_next)) {
      break;
    }

    QuantData data = tryMatrix(jpg, target_mul_high, q_next, img);
    qgen.add(data);
    if (compareQuantData(data, best)) {
      best = data;
      if (data.dist_ok && !comparator_->distanceOk(target_mul_low)) {
        break;
      }
    }
  }

  std::memcpy(&best_q[0][0], &best.q[0][0], kQuantMatrixSize * sizeof(best_q[0][0]));
  GUETZLI_LOG(stats_, "\n%s selected quantization matrix:\n", downsample ? "YUV420" : "YUV444");
  GUETZLI_LOG_QUANT(stats_, best_q);
  return best.dist_ok;
}

}  // namespace guetzli

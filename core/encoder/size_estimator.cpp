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

#include "core/encoder/size_estimator.h"

#include <cstdlib>
#include <cstring>

#include "core/util/fast_log.h"

namespace guetzli {

namespace {

// HistogramEntropyCost() for a histogram whose symbols cost raw_bits: adds the escape-byte estimate.
size_t withEscapeBytes(size_t raw_bits) {
  return raw_bits + ((raw_bits * 3 + 512) >> 10);
}

// The bits HistogramEntropyCost() counts for one occurrence of symbol (code length + extra bits).
size_t symbolBits(const uint8_t* depths, int symbol) {
  return depths[symbol] + (symbol & 0xf);
}

}  // namespace

size_t computeEntropyCodes(const std::vector<JpegHistogram>& histograms, std::vector<uint8_t>* depths) {
  std::vector<JpegHistogram> clustered = histograms;
  size_t num = histograms.size();
  std::vector<int> indexes(histograms.size());
  std::vector<uint8_t> clustered_depths(histograms.size() * JpegHistogram::kSize);
  clusterHistograms(&clustered[0], &num, &indexes[0], &clustered_depths[0]);
  depths->resize(clustered_depths.size());
  for (size_t i = 0; i < histograms.size(); ++i) {
    std::memcpy(&(*depths)[i * JpegHistogram::kSize], &clustered_depths[indexes[i] * JpegHistogram::kSize],
                JpegHistogram::kSize);
  }
  size_t histogram_size = 0;
  for (size_t i = 0; i < num; ++i) {
    histogram_size += histogramHeaderCost(clustered[i]) / 8;
  }
  return histogram_size;
}

size_t entropyCodedDataSize(const std::vector<JpegHistogram>& histograms, const std::vector<uint8_t>& depths) {
  size_t numbits = 0;
  for (size_t i = 0; i < histograms.size(); ++i) {
    numbits += histogramEntropyCost(histograms[i], &depths[i * JpegHistogram::kSize]);
  }
  return (numbits + 7) / 8;
}

size_t estimateDcSize(const JpegData& jpg) {
  std::vector<JpegHistogram> histograms(jpg.components.size());
  buildDcHistograms(jpg, &histograms[0]);
  size_t num = histograms.size();
  std::vector<int> indexes(num);
  std::vector<uint8_t> depths(num * JpegHistogram::kSize);
  return clusterHistograms(&histograms[0], &num, &indexes[0], &depths[0]);
}

JpegSizeEstimator::JpegSizeEstimator(const JpegData& jpg, size_t num_histograms, bool clear_metadata)
    : histograms_(num_histograms), raw_bits_(num_histograms) {
  // Truncated to int as the original encoder did; the sum is the same for any real header.
  const int header_size = static_cast<int>(jpegHeaderSize(jpg, clear_metadata));
  const int dc_size = static_cast<int>(estimateDcSize(jpg));
  fixed_size_ = static_cast<size_t>(header_size) + static_cast<size_t>(dc_size);
  buildAcHistograms(jpg, &histograms_[0]);
  refreshEntropyCodes();
}

void JpegSizeEstimator::refreshEntropyCodes() {
  ac_histogram_size_ = computeEntropyCodes(histograms_, &depths_);
  recomputeBits();
}

void JpegSizeEstimator::recomputeBits() {
  total_bits_ = 0;
  for (size_t h = 0; h < histograms_.size(); ++h) {
    const uint8_t* depths = &depths_[h * JpegHistogram::kSize];
    size_t bits = 0;
    for (int i = 0; i + 1 < JpegHistogram::kSize; ++i) {
      // JpegHistogram::add() counts every symbol twice.
      bits += (histograms_[h].counts[i] / 2) * symbolBits(depths, i);
    }
    raw_bits_[h] = bits;
    total_bits_ += withEscapeBytes(bits);
  }
}

size_t JpegSizeEstimator::estimate() const {
  return fixed_size_ + ac_histogram_size_ + (total_bits_ + 7) / 8;
}

void JpegSizeEstimator::replaceBlock(int c, const coeff_t* old_block, const coeff_t* new_block, const int* quant) {
  addBlock(c, old_block, quant, -1);
  addBlock(c, new_block, quant, 1);
}

void JpegSizeEstimator::addSymbol(int c, int symbol, int weight) {
  histograms_[c].add(symbol, weight);
  const size_t bits = symbolBits(&depths_[c * JpegHistogram::kSize], symbol);
  const size_t old_raw = raw_bits_[c];
  const size_t new_raw = weight > 0 ? old_raw + weight * bits : old_raw - (-weight) * bits;
  raw_bits_[c] = new_raw;
  total_bits_ = total_bits_ - withEscapeBytes(old_raw) + withEscapeBytes(new_raw);
}

// The AC symbols of one block, as UpdateACHistogramForDCTBlock() counts them for the written file.
void JpegSizeEstimator::addBlock(int c, const coeff_t* coeffs, const int* quant, int weight) {
  int r = 0;
  for (int k = 1; k < 64; ++k) {
    const int k_nat = kJpegNaturalOrder[k];
    coeff_t coeff = coeffs[k_nat];
    if (coeff == 0) {
      r++;
      continue;
    }
    while (r > 15) {
      addSymbol(c, 0xf0, weight);
      r -= 16;
    }
    int nbits = log2FloorNonZero(std::abs(coeff / quant[k_nat])) + 1;
    int symbol = (r << 4) + nbits;
    addSymbol(c, symbol, weight);
    r = 0;
  }
  if (r > 0) {
    addSymbol(c, 0, weight);
  }
}

}  // namespace guetzli

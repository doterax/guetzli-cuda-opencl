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

// Functions for writing a JPEGData object into a jpeg byte stream.

#ifndef GUETZLI_JPEG_DATA_WRITER_H_
#define GUETZLI_JPEG_DATA_WRITER_H_

#include <stdint.h>
#include <string.h>
#include <utility>
#include <vector>

#include "core/jpeg/jpeg_data.h"
#include "core/jpeg/entropy_encode.h"

namespace guetzli {

// Function pointer type used to write len bytes into buf. Returns the
// number of bytes written or -1 on error.
using JPEGOutputHook = int (*)(void* data, const uint8_t* buf, size_t len);

// Output callback function with associated data.
struct JpegOutput {
  JpegOutput(JPEGOutputHook cb, void* data) : cb_(cb), data_(data) {}
  bool write(const uint8_t* buf, size_t len) const { return (len == 0) || (std::cmp_equal(cb_(data_, buf, len), len)); }

 private:
  JPEGOutputHook cb_;
  void* data_;
};

bool writeJpeg(const JpegData& jpg, bool strip_metadata, JpegOutput out);

struct HuffmanCodeTable {
  uint8_t depth[256];
  int code[256];
};

struct JpegHistogram {
  static const int kSize = kJpegHuffmanAlphabetSize + 1;

  JpegHistogram() { clear(); }
  void clear() {
    memset(counts, 0, sizeof(counts));
    counts[kSize - 1] = 1;
  }
  void add(int symbol) { counts[symbol] += 2; }
  void add(int symbol, int weight) { counts[symbol] += 2 * weight; }
  void addHistogram(const JpegHistogram& other) {
    for (int i = 0; i + 1 < kSize; ++i) {
      counts[i] += other.counts[i];
    }
    counts[kSize - 1] = 1;
  }
  [[nodiscard]] int numSymbols() const {
    int n = 0;
    for (int i = 0; i + 1 < kSize; ++i) {
      n += (counts[i] > 0 ? 1 : 0);
    }
    return n;
  }

  uint32_t counts[kSize]{};
};

void buildDcHistograms(const JpegData& jpg, JpegHistogram* histo);
void buildAcHistograms(const JpegData& jpg, JpegHistogram* histo);
size_t jpegHeaderSize(const JpegData& jpg, bool strip_metadata);

size_t histogramEntropyCost(const JpegHistogram& histo, const uint8_t depths[256]);
size_t histogramHeaderCost(const JpegHistogram& histo);

void updateAcHistogramForDctBlock(const coeff_t* coeffs, JpegHistogram* ac_histogram);

// hints (optional, 2 * *num entries) speed up repeated clustering of slowly changing histograms (HuffmanSortHint);
// the result does not depend on them.
size_t clusterHistograms(JpegHistogram* histo, size_t* num, int* histo_indexes, uint8_t* depths,
                         HuffmanSortHint* hints = nullptr);

}  // namespace guetzli

#endif  // GUETZLI_JPEG_DATA_WRITER_H_

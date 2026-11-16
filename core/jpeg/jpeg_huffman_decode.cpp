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

#include "core/jpeg/jpeg_huffman_decode.h"

#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cstring>

#include "core/jpeg/jpeg_data.h"

namespace guetzli {

// Returns the table width of the next 2nd level table, count is the histogram
// of bit lengths for the remaining symbols, len is the code length of the next
// processed symbol.
static inline int nextTableBitSize(const int* count, int len) {
  int left = 1 << (len - kJpegHuffmanRootTableBits);
  while (len < kJpegHuffmanMaxBitLength) {
    left -= count[len];
    if (left <= 0) {
      break;
    }
    ++len;
    left <<= 1;
  }
  return len - kJpegHuffmanRootTableBits;
}

int buildJpegHuffmanTable(const int* count_in, const int* symbols, HuffmanTableEntry* lut) {
  HuffmanTableEntry code;              // current table entry
  HuffmanTableEntry* table = nullptr;  // next available space in table
  int len = 0;                         // current code length
  int idx = 0;                         // symbol index
  int key = 0;                         // prefix code
  int reps = 0;                        // number of replicate key values in current table
  int low = 0;                         // low bits for current root entry
  int table_bits = 0;                  // key length of current table
  int table_size = 0;                  // size of current table
  int total_size = 0;                  // sum of root table size and 2nd level table sizes

  // Make a local copy of the input bit length histogram.
  int count[kJpegHuffmanMaxBitLength + 1] = {0};
  int total_count = 0;
  for (len = 1; len <= kJpegHuffmanMaxBitLength; ++len) {
    count[len] = count_in[len];
    total_count += count[len];
  }

  table = lut;
  table_bits = kJpegHuffmanRootTableBits;
  table_size = 1 << table_bits;
  total_size = table_size;

  // Special case code with only one value.
  if (total_count == 1) {
    code.bits = 0;
    code.value = symbols[0];
    for (key = 0; key < total_size; ++key) {
      table[key] = code;
    }
    return total_size;
  }

  // Fill in root table.
  key = 0;
  idx = 0;
  for (len = 1; len <= kJpegHuffmanRootTableBits; ++len) {
    for (; count[len] > 0; --count[len]) {
      code.bits = len;
      code.value = symbols[idx++];
      reps = 1 << (kJpegHuffmanRootTableBits - len);
      while ((reps--) != 0) {
        table[key++] = code;
      }
    }
  }

  // Fill in 2nd level tables and add pointers to root table.
  table += table_size;
  table_size = 0;
  low = 0;
  for (len = kJpegHuffmanRootTableBits + 1; len <= kJpegHuffmanMaxBitLength; ++len) {
    for (; count[len] > 0; --count[len]) {
      // Start a new sub-table if the previous one is full.
      if (low >= table_size) {
        table += table_size;
        table_bits = nextTableBitSize(count, len);
        table_size = 1 << table_bits;
        total_size += table_size;
        low = 0;
        lut[key].bits = table_bits + kJpegHuffmanRootTableBits;
        lut[key].value = (table - lut) - key;
        ++key;
      }
      code.bits = len - kJpegHuffmanRootTableBits;
      code.value = symbols[idx++];
      reps = 1 << (table_bits - code.bits);
      while ((reps--) != 0) {
        table[low++] = code;
      }
    }
  }

  return total_size;
}

}  // namespace guetzli

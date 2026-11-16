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

// Entropy encoding (Huffman) utilities.

#ifndef GUETZLI_ENTROPY_ENCODE_H_
#define GUETZLI_ENTROPY_ENCODE_H_

#include <stddef.h>
#include <stdint.h>

#include <vector>

namespace guetzli {

// A node of a Huffman tree.
struct HuffmanTree {
  HuffmanTree() = default;
  HuffmanTree(uint32_t count, int16_t left, int16_t right)
      : total_count(count), index_left(left), index_right_or_value(right) {}
  uint32_t total_count;
  int16_t index_left;
  int16_t index_right_or_value;
};

bool setDepth(int p, HuffmanTree* pool, uint8_t* depth, int max_depth);

// The sorted leaf orders (symbols) of the previous tree built with this hint, one per attempt of the depth-limit
// loop (each attempt clamps the counts differently; large images need several). A histogram that changed only a
// little since then sorts in near-linear time from it. Only affects speed: the order is a strict total order, so
// the tree is the same with or without a hint.
struct HuffmanSortHint {
  static constexpr int kMaxAttempts = 32;
  std::vector<int16_t> order[kMaxAttempts];
};

// This function will create a Huffman tree.
//
// The (data,length) contains the population counts.
// The tree_limit is the maximum bit depth of the Huffman codes.
//
// The depth contains the tree, i.e., how many bits are used for
// the symbol.
//
// The actual Huffman tree is constructed in the tree[] array, which has to
// be at least 2 * length + 1 long.
//
// See http://en.wikipedia.org/wiki/Huffman_coding
// hint (optional) is read and updated, see HuffmanSortHint.
void createHuffmanTree(const uint32_t* data, size_t length, int tree_limit, HuffmanTree* tree, uint8_t* depth,
                       HuffmanSortHint* hint = nullptr);

}  // namespace guetzli

#endif  // GUETZLI_ENTROPY_ENCODE_H_

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

#include "core/jpeg/jpeg_data_writer.h"

#include <assert.h>
#include <cstdlib>
#include <string.h>

#include "core/jpeg/entropy_encode.h"
#include "core/util/fast_log.h"
#include "core/jpeg/jpeg_bit_writer.h"

namespace guetzli {

namespace {

static const int kJpegPrecision = 8;

// Writes len bytes from buf, using the out callback.
inline bool jpegWrite(JpegOutput out, const uint8_t* buf, size_t len) {
  static const size_t kBlockSize = 1u << 30;
  size_t pos = 0;
  while (len - pos > kBlockSize) {
    if (!out.write(buf + pos, kBlockSize)) {
      return false;
    }
    pos += kBlockSize;
  }
  return out.write(buf + pos, len - pos);
}

// Writes a string using the out callback.
inline bool jpegWrite(JpegOutput out, const std::string& s) {
  const uint8_t* data = reinterpret_cast<const uint8_t*>(&s[0]);
  return jpegWrite(out, data, s.size());
}

bool encodeMetadata(const JpegData& jpg, bool strip_metadata, JpegOutput out) {
  if (strip_metadata) {
    const uint8_t app0_data[] = {
        0xff, 0xe0, 0x00, 0x10,        // APP0
        0x4a, 0x46, 0x49, 0x46, 0x00,  // 'JFIF'
        0x01, 0x01,                    // v1.01
        0x00, 0x00, 0x01, 0x00, 0x01,  // aspect ratio = 1:1
        0x00, 0x00                     // thumbnail width/height
    };
    return jpegWrite(out, app0_data, sizeof(app0_data));
  }
  bool ok = true;
  for (size_t i = 0; i < jpg.app_data.size(); ++i) {
    uint8_t data[1] = {0xff};
    ok = ok && jpegWrite(out, data, sizeof(data));
    ok = ok && jpegWrite(out, jpg.app_data[i]);
  }
  for (size_t i = 0; i < jpg.com_data.size(); ++i) {
    uint8_t data[2] = {0xff, 0xfe};
    ok = ok && jpegWrite(out, data, sizeof(data));
    ok = ok && jpegWrite(out, jpg.com_data[i]);
  }
  return ok;
}

bool encodeDqt(const std::vector<JpegQuantTable>& quant, JpegOutput out) {
  int marker_len = 2;
  for (size_t i = 0; i < quant.size(); ++i) {
    marker_len += 1 + (quant[i].precision ? 2 : 1) * kDctBlockSize;
  }
  std::vector<uint8_t> data(marker_len + 2);
  size_t pos = 0;
  data[pos++] = 0xff;
  data[pos++] = 0xdb;
  data[pos++] = marker_len >> 8;
  data[pos++] = marker_len & 0xff;
  for (size_t i = 0; i < quant.size(); ++i) {
    const JpegQuantTable& table = quant[i];
    data[pos++] = (table.precision << 4) + table.index;
    for (int k = 0; k < kDctBlockSize; ++k) {
      int val = table.values[kJpegNaturalOrder[k]];
      if (table.precision) {
        data[pos++] = val >> 8;
      }
      data[pos++] = val & 0xff;
    }
  }
  return jpegWrite(out, &data[0], pos);
}

bool encodeSof(const JpegData& jpg, JpegOutput out) {
  const size_t ncomps = jpg.components.size();
  const size_t marker_len = 8 + 3 * ncomps;
  std::vector<uint8_t> data(marker_len + 2);
  size_t pos = 0;
  data[pos++] = 0xff;
  data[pos++] = 0xc1;
  data[pos++] = static_cast<uint8_t>(marker_len >> 8);
  data[pos++] = marker_len & 0xff;
  data[pos++] = kJpegPrecision;
  data[pos++] = jpg.height >> 8;
  data[pos++] = jpg.height & 0xff;
  data[pos++] = jpg.width >> 8;
  data[pos++] = jpg.width & 0xff;
  data[pos++] = static_cast<uint8_t>(ncomps);
  for (size_t i = 0; i < ncomps; ++i) {
    data[pos++] = jpg.components[i].id;
    data[pos++] = ((jpg.components[i].h_samp_factor << 4) | (jpg.components[i].v_samp_factor));
    const size_t quant_idx = jpg.components[i].quant_idx;
    if (quant_idx >= jpg.quant.size()) {
      return false;
    }
    data[pos++] = jpg.quant[quant_idx].index;
  }
  return jpegWrite(out, &data[0], pos);
}

// Builds a JPEG-style huffman code from the given bit depths.
void buildHuffmanCode(uint8_t* depth, int* counts, int* values) {
  for (int i = 0; i < JpegHistogram::kSize; ++i) {
    if (depth[i] > 0) {
      ++counts[depth[i]];
    }
  }
  int offset[kJpegHuffmanMaxBitLength + 1] = {0};
  for (int i = 1; i <= kJpegHuffmanMaxBitLength; ++i) {
    offset[i] = offset[i - 1] + counts[i - 1];
  }
  for (int i = 0; i < JpegHistogram::kSize; ++i) {
    if (depth[i] > 0) {
      values[offset[depth[i]]++] = i;
    }
  }
}

void buildHuffmanCodeTable(const int* counts, const int* values, HuffmanCodeTable* table) {
  int huffcode[256];
  int huffsize[256];
  int p = 0;
  for (int l = 1; l <= kJpegHuffmanMaxBitLength; ++l) {
    int i = counts[l];
    while (i--) huffsize[p++] = l;
  }

  if (p == 0) return;

  huffsize[p - 1] = 0;
  int lastp = p - 1;

  int code = 0;
  int si = huffsize[0];
  p = 0;
  while (huffsize[p]) {
    while ((huffsize[p]) == si) {
      huffcode[p++] = code;
      code++;
    }
    code <<= 1;
    si++;
  }
  for (p = 0; p < lastp; p++) {
    int i = values[p];
    table->depth[i] = huffsize[p];
    table->code[i] = huffcode[p];
  }
}

}  // namespace

// Updates ac_histogram with the counts of the AC symbols that will be added by
// a sequential jpeg encoder for this block. Every symbol is counted twice so
// that we can add a fake symbol at the end with count 1 to be the last (least
// frequent) symbol with the all 1 code.
void updateAcHistogramForDctBlock(const coeff_t* coeffs, JpegHistogram* ac_histogram) {
  int r = 0;
  for (int k = 1; k < 64; ++k) {
    coeff_t coeff = coeffs[kJpegNaturalOrder[k]];
    if (coeff == 0) {
      r++;
      continue;
    }
    while (r > 15) {
      ac_histogram->add(0xf0);
      r -= 16;
    }
    int nbits = log2FloorNonZero(std::abs(coeff)) + 1;
    int symbol = (r << 4) + nbits;
    ac_histogram->add(symbol);
    r = 0;
  }
  if (r > 0) {
    ac_histogram->add(0);
  }
}

size_t histogramHeaderCost(const JpegHistogram& histo) {
  size_t header_bits = 17 * 8;
  for (int i = 0; i + 1 < JpegHistogram::kSize; ++i) {
    if (histo.counts[i] > 0) {
      header_bits += 8;
    }
  }
  return header_bits;
}

size_t histogramEntropyCost(const JpegHistogram& histo, const uint8_t depths[256]) {
  size_t bits = 0;
  for (int i = 0; i + 1 < JpegHistogram::kSize; ++i) {
    // JpegHistogram::add() counts every symbol twice, so we have to divide by
    // two here.
    bits += (histo.counts[i] / 2) * (depths[i] + (i & 0xf));
  }
  // Estimate escape byte rate to be 0.75/256.
  bits += (bits * 3 + 512) >> 10;
  return bits;
}

void buildDcHistograms(const JpegData& jpg, JpegHistogram* histo) {
  for (size_t i = 0; i < jpg.components.size(); ++i) {
    const JpegComponent& c = jpg.components[i];
    JpegHistogram* dc_histogram = &histo[i];
    coeff_t last_dc_coeff = 0;
    for (int mcu_y = 0; mcu_y < jpg.mcu_rows; ++mcu_y) {
      for (int mcu_x = 0; mcu_x < jpg.mcu_cols; ++mcu_x) {
        for (int iy = 0; iy < c.v_samp_factor; ++iy) {
          for (int ix = 0; ix < c.h_samp_factor; ++ix) {
            int block_y = mcu_y * c.v_samp_factor + iy;
            int block_x = mcu_x * c.h_samp_factor + ix;
            int block_idx = block_y * c.width_in_blocks + block_x;
            coeff_t dc_coeff = c.coeffs[block_idx << 6];
            int diff = std::abs(dc_coeff - last_dc_coeff);
            int nbits = log2Floor(diff) + 1;
            dc_histogram->add(nbits);
            last_dc_coeff = dc_coeff;
          }
        }
      }
    }
  }
}

void buildAcHistograms(const JpegData& jpg, JpegHistogram* histo) {
  for (size_t i = 0; i < jpg.components.size(); ++i) {
    const JpegComponent& c = jpg.components[i];
    JpegHistogram* ac_histogram = &histo[i];
    for (size_t j = 0; j < c.coeffs.size(); j += kDctBlockSize) {
      updateAcHistogramForDctBlock(&c.coeffs[j], ac_histogram);
    }
  }
}

// Size of everything except the Huffman codes and the entropy coded data.
size_t jpegHeaderSize(const JpegData& jpg, bool strip_metadata) {
  size_t num_bytes = 0;
  num_bytes += 2;  // SOI
  if (strip_metadata) {
    num_bytes += 18;  // APP0
  } else {
    for (size_t i = 0; i < jpg.app_data.size(); ++i) {
      num_bytes += 1 + jpg.app_data[i].size();
    }
    for (size_t i = 0; i < jpg.com_data.size(); ++i) {
      num_bytes += 2 + jpg.com_data[i].size();
    }
  }
  // DQT
  num_bytes += 4;
  for (size_t i = 0; i < jpg.quant.size(); ++i) {
    num_bytes += 1 + (jpg.quant[i].precision ? 2 : 1) * kDctBlockSize;
  }
  num_bytes += 10 + 3 * jpg.components.size();  // SOF
  num_bytes += 4;                               // DHT (w/o actual Huffman code data)
  num_bytes += 8 + 2 * jpg.components.size();   // SOS
  num_bytes += 2;                               // EOI
  num_bytes += jpg.tail_data.size();
  return num_bytes;
}

size_t clusterHistograms(JpegHistogram* histo, size_t* num, int* histo_indexes, uint8_t* depth) {
  memset(depth, 0, *num * JpegHistogram::kSize);
  size_t costs[kMaxComponents];
  for (size_t i = 0; i < *num; ++i) {
    histo_indexes[i] = i;
    std::vector<HuffmanTree> tree(2 * JpegHistogram::kSize + 1);
    createHuffmanTree(histo[i].counts, JpegHistogram::kSize, kJpegHuffmanMaxBitLength, &tree[0],
                      &depth[i * JpegHistogram::kSize]);
    costs[i] = (histogramHeaderCost(histo[i]) + histogramEntropyCost(histo[i], &depth[i * JpegHistogram::kSize]));
  }
  const size_t orig_num = *num;
  while (*num > 1) {
    size_t last = *num - 1;
    size_t second_last = *num - 2;
    JpegHistogram combined(histo[last]);
    combined.addHistogram(histo[second_last]);
    std::vector<HuffmanTree> tree(2 * JpegHistogram::kSize + 1);
    uint8_t depth_combined[JpegHistogram::kSize] = {0};
    createHuffmanTree(combined.counts, JpegHistogram::kSize, kJpegHuffmanMaxBitLength, &tree[0], depth_combined);
    size_t cost_combined = (histogramHeaderCost(combined) + histogramEntropyCost(combined, depth_combined));
    if (cost_combined < costs[last] + costs[second_last]) {
      histo[second_last] = combined;
      histo[last] = JpegHistogram();
      costs[second_last] = cost_combined;
      memcpy(&depth[second_last * JpegHistogram::kSize], depth_combined, sizeof(depth_combined));
      for (size_t i = 0; i < orig_num; ++i) {
        if (histo_indexes[i] == last) {
          histo_indexes[i] = second_last;
        }
      }
      --(*num);
    } else {
      break;
    }
  }
  size_t total_cost = 0;
  for (size_t i = 0; i < *num; ++i) {
    total_cost += costs[i];
  }
  return (total_cost + 7) / 8;
}

namespace {

// Writes DHT and SOS marker segments to out and fills in DC/AC Huffman tables
// for each component of the image.
bool buildAndEncodeHuffmanCodes(const JpegData& jpg, JpegOutput out, std::vector<HuffmanCodeTable>* dc_huff_tables,
                                std::vector<HuffmanCodeTable>* ac_huff_tables) {
  const int ncomps = jpg.components.size();
  dc_huff_tables->resize(ncomps);
  ac_huff_tables->resize(ncomps);

  // Build separate DC histograms for each component.
  std::vector<JpegHistogram> histograms(ncomps);
  buildDcHistograms(jpg, &histograms[0]);

  // Cluster DC histograms.
  size_t num_dc_histo = ncomps;
  int dc_histo_indexes[kMaxComponents];
  std::vector<uint8_t> depths(ncomps * JpegHistogram::kSize);
  clusterHistograms(&histograms[0], &num_dc_histo, dc_histo_indexes, &depths[0]);

  // Build separate AC histograms for each component.
  histograms.resize(num_dc_histo + ncomps);
  depths.resize((num_dc_histo + ncomps) * JpegHistogram::kSize);
  buildAcHistograms(jpg, &histograms[num_dc_histo]);

  // Cluster AC histograms.
  size_t num_ac_histo = ncomps;
  int ac_histo_indexes[kMaxComponents];
  clusterHistograms(&histograms[num_dc_histo], &num_ac_histo, ac_histo_indexes,
                    &depths[num_dc_histo * JpegHistogram::kSize]);

  // Compute DHT and SOS marker data sizes and start emitting DHT marker.
  int num_histo = num_dc_histo + num_ac_histo;
  histograms.resize(num_histo);
  int total_count = 0;
  for (size_t i = 0; i < histograms.size(); ++i) {
    total_count += histograms[i].numSymbols();
  }
  const size_t dht_marker_len = 2 + num_histo * (kJpegHuffmanMaxBitLength + 1) + total_count;
  const size_t sos_marker_len = 6 + 2 * ncomps;
  std::vector<uint8_t> data(dht_marker_len + sos_marker_len + 4);
  size_t pos = 0;
  data[pos++] = 0xff;
  data[pos++] = 0xc4;
  data[pos++] = static_cast<uint8_t>(dht_marker_len >> 8);
  data[pos++] = dht_marker_len & 0xff;

  // Compute Huffman codes for each histograms.
  for (int i = 0; i < num_histo; ++i) {
    const bool is_dc = static_cast<size_t>(i) < num_dc_histo;
    const int idx = is_dc ? i : i - num_dc_histo;
    int counts[kJpegHuffmanMaxBitLength + 1] = {0};
    int values[JpegHistogram::kSize] = {0};
    buildHuffmanCode(&depths[i * JpegHistogram::kSize], counts, values);
    HuffmanCodeTable table;
    for (int j = 0; j < 256; ++j) table.depth[j] = 255;
    buildHuffmanCodeTable(counts, values, &table);
    for (int c = 0; c < ncomps; ++c) {
      if (is_dc) {
        if (dc_histo_indexes[c] == idx) (*dc_huff_tables)[c] = table;
      } else {
        if (ac_histo_indexes[c] == idx) (*ac_huff_tables)[c] = table;
      }
    }
    int max_length = kJpegHuffmanMaxBitLength;
    while (max_length > 0 && counts[max_length] == 0) --max_length;
    --counts[max_length];
    int total_count = 0;
    for (int j = 0; j <= max_length; ++j) total_count += counts[j];
    data[pos++] = is_dc ? i : static_cast<uint8_t>(i - num_dc_histo + 0x10);
    for (size_t j = 1; j <= kJpegHuffmanMaxBitLength; ++j) {
      data[pos++] = counts[j];
    }
    for (int j = 0; j < total_count; ++j) {
      data[pos++] = values[j];
    }
  }

  // Emit SOS marker data.
  data[pos++] = 0xff;
  data[pos++] = 0xda;
  data[pos++] = static_cast<uint8_t>(sos_marker_len >> 8);
  data[pos++] = sos_marker_len & 0xff;
  data[pos++] = ncomps;
  for (int i = 0; i < ncomps; ++i) {
    data[pos++] = jpg.components[i].id;
    data[pos++] = (dc_histo_indexes[i] << 4) | ac_histo_indexes[i];
  }
  data[pos++] = 0;
  data[pos++] = 63;
  data[pos++] = 0;
  assert(pos == data.size());
  return jpegWrite(out, &data[0], data.size());
}

void encodeDctBlockSequential(const coeff_t* coeffs, const HuffmanCodeTable& dc_huff, const HuffmanCodeTable& ac_huff,
                              coeff_t* last_dc_coeff, BitWriter* bw) {
  coeff_t temp2;
  coeff_t temp;
  temp2 = coeffs[0];
  temp = temp2 - *last_dc_coeff;
  *last_dc_coeff = temp2;
  temp2 = temp;
  if (temp < 0) {
    temp = -temp;
    temp2--;
  }
  int nbits = log2Floor(temp) + 1;
  bw->writeBits(dc_huff.depth[nbits], dc_huff.code[nbits]);
  if (nbits > 0) {
    bw->writeBits(nbits, temp2 & ((1 << nbits) - 1));
  }
  int r = 0;
  for (int k = 1; k < 64; ++k) {
    if ((temp = coeffs[kJpegNaturalOrder[k]]) == 0) {
      r++;
      continue;
    }
    if (temp < 0) {
      temp = -temp;
      temp2 = ~temp;
    } else {
      temp2 = temp;
    }
    while (r > 15) {
      bw->writeBits(ac_huff.depth[0xf0], ac_huff.code[0xf0]);
      r -= 16;
    }
    int nbits = log2FloorNonZero(temp) + 1;
    int symbol = (r << 4) + nbits;
    bw->writeBits(ac_huff.depth[symbol], ac_huff.code[symbol]);
    bw->writeBits(nbits, temp2 & ((1 << nbits) - 1));
    r = 0;
  }
  if (r > 0) {
    bw->writeBits(ac_huff.depth[0], ac_huff.code[0]);
  }
}

bool encodeScan(const JpegData& jpg, const std::vector<HuffmanCodeTable>& dc_huff_table,
                const std::vector<HuffmanCodeTable>& ac_huff_table, JpegOutput out) {
  coeff_t last_dc_coeff[kMaxComponents] = {0};
  BitWriter bw(1 << 17);
  for (int mcu_y = 0; mcu_y < jpg.mcu_rows; ++mcu_y) {
    for (int mcu_x = 0; mcu_x < jpg.mcu_cols; ++mcu_x) {
      // Encode one MCU
      for (size_t i = 0; i < jpg.components.size(); ++i) {
        const JpegComponent& c = jpg.components[i];
        int nblocks_y = c.v_samp_factor;
        int nblocks_x = c.h_samp_factor;
        for (int iy = 0; iy < nblocks_y; ++iy) {
          for (int ix = 0; ix < nblocks_x; ++ix) {
            int block_y = mcu_y * nblocks_y + iy;
            int block_x = mcu_x * nblocks_x + ix;
            int block_idx = block_y * c.width_in_blocks + block_x;
            const coeff_t* coeffs = &c.coeffs[block_idx << 6];
            encodeDctBlockSequential(coeffs, dc_huff_table[i], ac_huff_table[i], &last_dc_coeff[i], &bw);
          }
        }
      }
      if (bw.pos > (1 << 16)) {
        if (!jpegWrite(out, bw.data.get(), bw.pos)) {
          return false;
        }
        bw.pos = 0;
      }
    }
  }
  bw.jumpToByteBoundary();
  return !bw.overflow && jpegWrite(out, bw.data.get(), bw.pos);
}

}  // namespace

bool writeJpeg(const JpegData& jpg, bool strip_metadata, JpegOutput out) {
  static const uint8_t kSoiMarker[2] = {0xff, 0xd8};
  static const uint8_t kEoiMarker[2] = {0xff, 0xd9};
  std::vector<HuffmanCodeTable> dc_codes;
  std::vector<HuffmanCodeTable> ac_codes;
  return (jpegWrite(out, kSoiMarker, sizeof(kSoiMarker)) && encodeMetadata(jpg, strip_metadata, out) &&
          encodeDqt(jpg.quant, out) && encodeSof(jpg, out) &&
          buildAndEncodeHuffmanCodes(jpg, out, &dc_codes, &ac_codes) && encodeScan(jpg, dc_codes, ac_codes, out) &&
          jpegWrite(out, kEoiMarker, sizeof(kEoiMarker)) && (strip_metadata || jpegWrite(out, jpg.tail_data)));
}

}  // namespace guetzli

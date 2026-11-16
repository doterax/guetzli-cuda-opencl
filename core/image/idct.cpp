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

// Integer implementation of the Inverse Discrete Cosine Transform (IDCT).

#include "core/image/idct.h"

#include <algorithm>
#include <cmath>

namespace guetzli {

// kIDCTMatrix[8*x+u] = alpha(u)*cos((2*x+1)*u*M_PI/16)*sqrt(2), with fixed 13
// bit precision, where alpha(0) = 1/sqrt(2) and alpha(u) = 1 for u > 0.
// Some coefficients are off by +-1 to mimick libjpeg's behaviour.
static const int kIdctMatrix[kDctBlockSize] = {
    8192,   11363,  10703,  9633,  8192,  6437,   4433,   2260,  8192,   9633,  4433,  -2259, -8192,
    -11362, -10704, -6436,  8192,  6437,  -4433,  -11362, -8192, 2261,   10704, 9633,  8192,  2260,
    -10703, -6436,  8192,   9633,  -4433, -11363, 8192,   -2260, -10703, 6436,  8192,  -9633, -4433,
    11363,  8192,   -6437,  -4433, 11362, -8192,  -2261,  10704, -9633,  8192,  -9633, 4433,  2259,
    -8192,  11362,  -10704, 6436,  8192,  -11363, 10703,  -9633, 8192,   -6437, 4433,  -2260,
};

// Computes out[x] = sum{kIDCTMatrix[8*x+u]*in[u*stride]; for u in [0..7]}
inline void compute1dIdct(const coeff_t* in, const int stride, int out[8]) {
  int tmp0 = 0;
  int tmp1 = 0;
  int tmp2 = 0;
  int tmp3 = 0;
  int tmp4 = 0;

  tmp1 = kIdctMatrix[0] * in[0];
  out[0] = out[1] = out[2] = out[3] = out[4] = out[5] = out[6] = out[7] = tmp1;

  tmp0 = in[stride];
  tmp1 = kIdctMatrix[1] * tmp0;
  tmp2 = kIdctMatrix[9] * tmp0;
  tmp3 = kIdctMatrix[17] * tmp0;
  tmp4 = kIdctMatrix[25] * tmp0;
  out[0] += tmp1;
  out[1] += tmp2;
  out[2] += tmp3;
  out[3] += tmp4;
  out[4] -= tmp4;
  out[5] -= tmp3;
  out[6] -= tmp2;
  out[7] -= tmp1;

  tmp0 = in[2 * stride];
  tmp1 = kIdctMatrix[2] * tmp0;
  tmp2 = kIdctMatrix[10] * tmp0;
  out[0] += tmp1;
  out[1] += tmp2;
  out[2] -= tmp2;
  out[3] -= tmp1;
  out[4] -= tmp1;
  out[5] -= tmp2;
  out[6] += tmp2;
  out[7] += tmp1;

  tmp0 = in[3 * stride];
  tmp1 = kIdctMatrix[3] * tmp0;
  tmp2 = kIdctMatrix[11] * tmp0;
  tmp3 = kIdctMatrix[19] * tmp0;
  tmp4 = kIdctMatrix[27] * tmp0;
  out[0] += tmp1;
  out[1] += tmp2;
  out[2] += tmp3;
  out[3] += tmp4;
  out[4] -= tmp4;
  out[5] -= tmp3;
  out[6] -= tmp2;
  out[7] -= tmp1;

  tmp0 = in[4 * stride];
  tmp1 = kIdctMatrix[4] * tmp0;
  out[0] += tmp1;
  out[1] -= tmp1;
  out[2] -= tmp1;
  out[3] += tmp1;
  out[4] += tmp1;
  out[5] -= tmp1;
  out[6] -= tmp1;
  out[7] += tmp1;

  tmp0 = in[5 * stride];
  tmp1 = kIdctMatrix[5] * tmp0;
  tmp2 = kIdctMatrix[13] * tmp0;
  tmp3 = kIdctMatrix[21] * tmp0;
  tmp4 = kIdctMatrix[29] * tmp0;
  out[0] += tmp1;
  out[1] += tmp2;
  out[2] += tmp3;
  out[3] += tmp4;
  out[4] -= tmp4;
  out[5] -= tmp3;
  out[6] -= tmp2;
  out[7] -= tmp1;

  tmp0 = in[6 * stride];
  tmp1 = kIdctMatrix[6] * tmp0;
  tmp2 = kIdctMatrix[14] * tmp0;
  out[0] += tmp1;
  out[1] += tmp2;
  out[2] -= tmp2;
  out[3] -= tmp1;
  out[4] -= tmp1;
  out[5] -= tmp2;
  out[6] += tmp2;
  out[7] += tmp1;

  tmp0 = in[7 * stride];
  tmp1 = kIdctMatrix[7] * tmp0;
  tmp2 = kIdctMatrix[15] * tmp0;
  tmp3 = kIdctMatrix[23] * tmp0;
  tmp4 = kIdctMatrix[31] * tmp0;
  out[0] += tmp1;
  out[1] += tmp2;
  out[2] += tmp3;
  out[3] += tmp4;
  out[4] -= tmp4;
  out[5] -= tmp3;
  out[6] -= tmp2;
  out[7] -= tmp1;
}

void computeBlockIdct(const coeff_t* block, uint8_t* out) {
  coeff_t colidcts[kDctBlockSize];
  const int col_scale = 11;
  const int col_round = 1 << (col_scale - 1);
  for (int x = 0; x < 8; ++x) {
    int colbuf[8] = {0};
    compute1dIdct(&block[x], 8, colbuf);
    for (int y = 0; y < 8; ++y) {
      colidcts[8 * y + x] = (colbuf[y] + col_round) >> col_scale;
    }
  }
  const int row_scale = 18;
  const int row_round = 257 << (row_scale - 1);  // includes offset by 128
  for (int y = 0; y < 8; ++y) {
    const int rowidx = 8 * y;
    int rowbuf[8] = {0};
    compute1dIdct(&colidcts[rowidx], 1, rowbuf);
    for (int x = 0; x < 8; ++x) {
      out[rowidx + x] = std::max(0, std::min(255, (rowbuf[x] + row_round) >> row_scale));
    }
  }
}

}  // namespace guetzli

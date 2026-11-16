#include <cstdlib>

#include "doctest.h"
#include "core/jpeg/jpeg_data.h"
#include "core/jpeg/quantize.h"

using guetzli::coeff_t;
using guetzli::kDctBlockSize;

TEST_CASE("Quantize rounds to the nearest multiple of the quantizer") {
  CHECK(guetzli::quantize(0, 7) == 0);
  CHECK(guetzli::quantize(14, 7) == 14);
  CHECK(guetzli::quantize(10, 7) == 7);
  CHECK(guetzli::quantize(11, 7) == 14);
  CHECK(guetzli::quantize(-10, 7) == -7);
  CHECK(guetzli::quantize(-11, 7) == -14);
  CHECK(guetzli::quantize(5, 1) == 5);
  for (int q = 1; q < 40; ++q) {
    for (int v = -300; v <= 300; ++v) {
      const coeff_t r = guetzli::quantize(static_cast<coeff_t>(v), q);
      CHECK(r % q == 0);
      CHECK(2 * std::abs(r - v) <= q);
    }
  }
}

TEST_CASE("QuantizeBlock reports whether anything changed") {
  int q[kDctBlockSize];
  coeff_t block[kDctBlockSize];
  for (int i = 0; i < kDctBlockSize; ++i) {
    q[i] = 4;
    block[i] = static_cast<coeff_t>(4 * (i - 32));
  }
  CHECK_FALSE(guetzli::quantizeBlock(block, q));
  block[5] += 3;
  CHECK(guetzli::quantizeBlock(block, q));
  CHECK(block[5] % 4 == 0);
}

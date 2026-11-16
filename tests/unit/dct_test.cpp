// Integer DCT/IDCT against the double-precision reference implementation.
//
// Conventions (see fdct.cc / dct_double.cc):
// - ComputeBlockDCT takes level-shifted samples in [-128, 127] and returns
//   coefficients scaled by 16 relative to ComputeBlockDCTDouble.
// - ComputeBlockIDCT takes dequantized JPEG coefficients (DC = 8 * mean of the
//   level-shifted block, the ComputeBlockDCTDouble scale) and returns pixels.
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <random>

#include "doctest.h"
#include "core/image/dct_double.h"
#include "core/image/fdct.h"
#include "core/image/idct.h"
#include "core/jpeg/jpeg_data.h"

using guetzli::coeff_t;
using guetzli::kDctBlockSize;

TEST_CASE("forward DCT of a flat block is (almost) DC only") {
  for (int v = -128; v <= 127; ++v) {
    coeff_t block[kDctBlockSize];
    for (short& i : block) i = static_cast<coeff_t>(v);
    guetzli::computeBlockDct(block);
    CHECK(std::abs(block[0] - 16 * 8 * v) <= 16);
    for (int i = 1; i < kDctBlockSize; ++i) CHECK(std::abs(block[i]) <= 3);  // < 0.2 unscaled
  }
}

TEST_CASE("forward integer DCT is 16x the double reference, within 0.25") {
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> px(-128, 127);
  for (int trial = 0; trial < 500; ++trial) {
    coeff_t block[kDctBlockSize];
    double ref[kDctBlockSize];
    for (int i = 0; i < kDctBlockSize; ++i) {
      block[i] = static_cast<coeff_t>(px(rng));
      ref[i] = block[i];
    }
    guetzli::computeBlockDct(block);
    guetzli::computeBlockDctDouble(ref);
    for (int i = 0; i < kDctBlockSize; ++i) CHECK(std::fabs(block[i] / 16.0 - ref[i]) <= 0.25);
  }
}

TEST_CASE("double DCT and IDCT are inverse") {
  std::mt19937 rng(3);
  std::uniform_real_distribution<double> px(-128.0, 127.0);
  double in[kDctBlockSize];
  double block[kDctBlockSize];
  for (int i = 0; i < kDctBlockSize; ++i) block[i] = in[i] = px(rng);
  guetzli::computeBlockDctDouble(block);
  guetzli::computeBlockIdctDouble(block);
  for (int i = 0; i < kDctBlockSize; ++i) CHECK(block[i] == doctest::Approx(in[i]).epsilon(1e-6));
}

TEST_CASE("IDCT of a DC-only block is flat") {
  coeff_t block[kDctBlockSize] = {0};
  block[0] = 8 * 40;  // mean +40 after the level shift of 128
  uint8_t out[kDctBlockSize];
  guetzli::computeBlockIdct(block, out);
  for (unsigned char& i : out) CHECK(i == 128 + 40);
}

TEST_CASE("integer IDCT inverts the rounded double DCT within 1") {
  std::mt19937 rng(7);
  std::uniform_int_distribution<int> px(0, 255);
  for (int trial = 0; trial < 500; ++trial) {
    uint8_t pixels[kDctBlockSize];
    double ref[kDctBlockSize];
    for (int i = 0; i < kDctBlockSize; ++i) {
      pixels[i] = static_cast<uint8_t>(px(rng));
      ref[i] = pixels[i] - 128.0;
    }
    guetzli::computeBlockDctDouble(ref);
    coeff_t coeffs[kDctBlockSize];
    for (int i = 0; i < kDctBlockSize; ++i) coeffs[i] = static_cast<coeff_t>(std::lround(ref[i]));
    uint8_t out[kDctBlockSize];
    guetzli::computeBlockIdct(coeffs, out);
    for (int i = 0; i < kDctBlockSize; ++i) CHECK(std::abs(out[i] - pixels[i]) <= 1);
  }
}

// Host port of the kernel helpers (core/quality/opsin_block.cpp).
#include <cmath>

#include "core/quality/opsin_block.h"
#include "doctest.h"
#include "core/image/gamma_correct.h"

TEST_CASE("float sRGB LUT matches the double LUT to float precision") {
  const float* f = guetzli::srgb8ToLinearTableF();
  const double* d = guetzli::srgb8ToLinearTable();
  CHECK(f[0] == 0.0f);
  for (int i = 1; i < 256; ++i) {
    CHECK(f[i] > f[i - 1]);
    CHECK(std::fabs(f[i] - d[i]) <= 1e-5 * d[i] + 1e-6);
  }
}

TEST_CASE("opsin dynamics of a flat block is flat") {
  const float* lut = guetzli::srgb8ToLinearTableF();
  float rgb[3][guetzli::kOpsinBlockSize];
  for (int i = 0; i < guetzli::kOpsinBlockSize; ++i) {
    rgb[0][i] = lut[200];
    rgb[1][i] = lut[120];
    rgb[2][i] = lut[60];
  }
  guetzli::opsinDynamicsBlock8x8(rgb);
  for (auto& c : rgb) {
    // Not std::isfinite: -ffast-math lets the compiler assume it is true. NaN/inf fail this bound.
    CHECK(std::fabs(c[0]) < 1e6f);
    for (int i = 1; i < guetzli::kOpsinBlockSize; ++i) {
      // Border handling of the 8x8 blur is exact only in real arithmetic;
      // float rounding leaves ~2e-4 relative ripple.
      CHECK(c[i] == doctest::Approx(c[0]).epsilon(1e-3));
    }
  }
}

TEST_CASE("opsin dynamics: neutral gray has a small X channel") {
  const float* lut = guetzli::srgb8ToLinearTableF();
  float rgb[3][guetzli::kOpsinBlockSize];
  for (auto& c : rgb) {
    for (float& i : c) {
      i = lut[128];
    }
  }
  guetzli::opsinDynamicsBlock8x8(rgb);
  CHECK(rgb[1][0] > 0.0f);
  CHECK(std::fabs(rgb[0][0]) < 0.1f * std::fabs(rgb[1][0]));
}

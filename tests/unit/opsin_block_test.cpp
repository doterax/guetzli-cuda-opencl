// Host port of the kernel helpers (guetzli/opsin_block.cc).
#include <cmath>

#include "guetzli/opsin_block.h"
#include "doctest.h"
#include "guetzli/gamma_correct.h"

TEST_CASE("float sRGB LUT matches the double LUT to float precision") {
  const float* f = guetzli::Srgb8ToLinearTableF();
  const double* d = guetzli::Srgb8ToLinearTable();
  CHECK(f[0] == 0.0f);
  for (int i = 1; i < 256; ++i) {
    CHECK(f[i] > f[i - 1]);
    CHECK(std::fabs(f[i] - d[i]) <= 1e-5 * d[i] + 1e-6);
  }
}

TEST_CASE("opsin dynamics of a flat block is flat") {
  const float* lut = guetzli::Srgb8ToLinearTableF();
  float rgb[3][guetzli::kOpsinBlockSize];
  for (int i = 0; i < guetzli::kOpsinBlockSize; ++i) {
    rgb[0][i] = lut[200];
    rgb[1][i] = lut[120];
    rgb[2][i] = lut[60];
  }
  guetzli::OpsinDynamicsBlock8x8(rgb);
  for (int c = 0; c < 3; ++c) {
    CHECK(std::isfinite(rgb[c][0]));
    for (int i = 1; i < guetzli::kOpsinBlockSize; ++i) {
      // Border handling of the 8x8 blur is exact only in real arithmetic;
      // float rounding leaves ~2e-4 relative ripple.
      CHECK(rgb[c][i] == doctest::Approx(rgb[c][0]).epsilon(1e-3));
    }
  }
}

TEST_CASE("opsin dynamics: neutral gray has a small X channel") {
  const float* lut = guetzli::Srgb8ToLinearTableF();
  float rgb[3][guetzli::kOpsinBlockSize];
  for (int c = 0; c < 3; ++c)
    for (int i = 0; i < guetzli::kOpsinBlockSize; ++i) rgb[c][i] = lut[128];
  guetzli::OpsinDynamicsBlock8x8(rgb);
  CHECK(rgb[1][0] > 0.0f);
  CHECK(std::fabs(rgb[0][0]) < 0.1f * std::fabs(rgb[1][0]));
}

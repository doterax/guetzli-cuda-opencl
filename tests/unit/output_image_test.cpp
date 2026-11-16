// Unit tests for OutputImage / OutputImageComponent (CPU DCT operations).
#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#include "doctest.h"
#include "core/jpeg/jpeg_data.h"
#include "core/image/jpeg_data_encoder.h"
#include "core/image/output_image.h"

using guetzli::coeff_t;
using guetzli::kDctBlockSize;

namespace {

std::vector<uint8_t> colorImage(int w, int h) {
  std::mt19937 rng(11);
  std::vector<uint8_t> rgb(3 * w * h);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      uint8_t* p = &rgb[3 * (y * w + x)];
      p[0] = static_cast<uint8_t>(x * 255 / w);
      p[1] = static_cast<uint8_t>(y * 255 / h);
      p[2] = static_cast<uint8_t>(rng() % 256);
    }
  }
  return rgb;
}

guetzli::JpegData encodeWithQuant(const std::vector<uint8_t>& rgb, int w, int h, int quant_value) {
  int quant[3 * kDctBlockSize];
  for (int& q : quant) {
    q = quant_value;
  }
  guetzli::JpegData jpg;
  REQUIRE(guetzli::encodeRgbToJpeg(rgb, w, h, quant, &jpg));
  return jpg;
}

}  // namespace

TEST_CASE("OutputImageComponent: a coefficient block round-trips and updates its pixels") {
  guetzli::OutputImageComponent comp(20, 12);
  CHECK(comp.widthInBlocks() == 3);
  CHECK(comp.heightInBlocks() == 2);
  coeff_t block[kDctBlockSize] = {};
  block[0] = 8 * 40;  // DC only: a flat block, 40 levels above mid-gray
  comp.setCoeffBlock(2, 1, block);
  coeff_t read[kDctBlockSize];
  comp.getCoeffBlock(2, 1, read);
  CHECK(std::equal(block, block + kDctBlockSize, read));

  uint8_t pixels[4];
  comp.toPixels(16, 8, 4, 1, pixels, 1);  // inside block (2, 1)
  for (uint8_t p : pixels) {
    CHECK(p == 168);
  }
  comp.toPixels(0, 0, 4, 1, pixels, 1);  // untouched block
  for (uint8_t p : pixels) {
    CHECK(p == 128);
  }
  CHECK_FALSE(comp.isAllZero());
}

TEST_CASE("OutputImage: JPEG coefficients round-trip through CopyFromJpegData / SaveToJpegData") {
  const int w = 35;
  const int h = 19;
  const guetzli::JpegData jpg = encodeWithQuant(colorImage(w, h), w, h, 3);
  guetzli::CpuDctOps dct;
  guetzli::OutputImage img(w, h, dct);
  img.copyFromJpegData(jpg);
  CHECK(img.frameTypeStr() == "f111111");
  for (int c = 0; c < 3; ++c) {
    CHECK(img.component(c).quant()[5] == 3);
  }

  guetzli::JpegData out;
  img.saveToJpegData(&out);
  REQUIRE(out.components.size() == 3);
  CHECK(out.is444());
  for (int c = 0; c < 3; ++c) {
    CHECK(out.components[c].coeffs == jpg.components[c].coeffs);
    CHECK(out.quant[out.components[c].quant_idx].values == jpg.quant[jpg.components[c].quant_idx].values);
  }
}

TEST_CASE("OutputImage: global quantization") {
  const int w = 24;
  const int h = 16;
  guetzli::CpuDctOps dct;
  guetzli::OutputImage img(w, h, dct);
  img.copyFromJpegData(encodeWithQuant(colorImage(w, h), w, h, 1));
  int q[3][kDctBlockSize];
  for (int c = 0; c < 3; ++c) {
    for (int k = 0; k < kDctBlockSize; ++k) {
      q[c][k] = 2 + (k + c) % 7;
    }
  }
  img.applyGlobalQuantization(q);
  for (int c = 0; c < 3; ++c) {
    const guetzli::OutputImageComponent& comp = img.component(c);
    CHECK(std::equal(&q[c][0], &q[c][0] + kDctBlockSize, comp.quant()));
    const coeff_t* coeffs = comp.coeffs();
    for (int i = 0; i < comp.widthInBlocks() * comp.heightInBlocks() * kDctBlockSize; ++i) {
      REQUIRE(coeffs[i] % q[c][i % kDctBlockSize] == 0);
    }
  }
}

TEST_CASE("OutputImage: downsampling to 4:2:0 and saving the result") {
  const int w = 33;
  const int h = 18;
  guetzli::CpuDctOps dct;
  guetzli::OutputImage img(w, h, dct);
  img.copyFromJpegData(encodeWithQuant(colorImage(w, h), w, h, 1));
  img.downsample(guetzli::OutputImage::DownsampleConfig{});
  CHECK(img.frameTypeStr() == "f112222");
  CHECK(img.component(1).widthInBlocks() == 3);
  CHECK(img.component(1).heightInBlocks() == 2);

  guetzli::JpegData out;
  img.saveToJpegData(&out);
  CHECK(out.max_h_samp_factor == 2);
  CHECK(out.max_v_samp_factor == 2);
  CHECK(out.is420());
  CHECK(out.components[0].width_in_blocks == 6);
  CHECK(out.components[1].width_in_blocks == 3);
}

TEST_CASE("OutputImage: an image without chroma is saved as grayscale") {
  const int w = 16;
  const int h = 16;
  std::vector<uint8_t> gray(3 * w * h);
  for (int i = 0; i < w * h; ++i) {
    gray[3 * i] = gray[3 * i + 1] = gray[3 * i + 2] = static_cast<uint8_t>(i);
  }
  guetzli::CpuDctOps dct;
  guetzli::OutputImage img(w, h, dct);
  img.copyFromJpegData(encodeWithQuant(gray, w, h, 1));
  REQUIRE(img.component(1).isAllZero());
  REQUIRE(img.component(2).isAllZero());
  guetzli::JpegData out;
  img.saveToJpegData(&out);
  CHECK(out.components.size() == 1);
}

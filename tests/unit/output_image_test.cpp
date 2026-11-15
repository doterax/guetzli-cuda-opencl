// Unit tests for OutputImage / OutputImageComponent (CPU DCT operations).
#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#include "doctest.h"
#include "guetzli/jpeg_data.h"
#include "guetzli/jpeg_data_encoder.h"
#include "guetzli/output_image.h"

using guetzli::coeff_t;
using guetzli::kDCTBlockSize;

namespace {

std::vector<uint8_t> ColorImage(int w, int h) {
  std::mt19937 rng(11);
  std::vector<uint8_t> rgb(3 * w * h);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      uint8_t* p = &rgb[3 * (y * w + x)];
      p[0] = static_cast<uint8_t>(x * 255 / w);
      p[1] = static_cast<uint8_t>(y * 255 / h);
      p[2] = static_cast<uint8_t>(rng() % 256);
    }
  return rgb;
}

guetzli::JPEGData EncodeWithQuant(const std::vector<uint8_t>& rgb, int w, int h, int quant_value) {
  int quant[3 * kDCTBlockSize];
  for (int& q : quant) q = quant_value;
  guetzli::JPEGData jpg;
  REQUIRE(guetzli::EncodeRGBToJpeg(rgb, w, h, quant, &jpg));
  return jpg;
}

}  // namespace

TEST_CASE("OutputImageComponent: a coefficient block round-trips and updates its pixels") {
  guetzli::OutputImageComponent comp(20, 12);
  CHECK(comp.width_in_blocks() == 3);
  CHECK(comp.height_in_blocks() == 2);
  coeff_t block[kDCTBlockSize] = {};
  block[0] = 8 * 40;  // DC only: a flat block, 40 levels above mid-gray
  comp.SetCoeffBlock(2, 1, block);
  coeff_t read[kDCTBlockSize];
  comp.GetCoeffBlock(2, 1, read);
  CHECK(std::equal(block, block + kDCTBlockSize, read));

  uint8_t pixels[4];
  comp.ToPixels(16, 8, 4, 1, pixels, 1);  // inside block (2, 1)
  for (uint8_t p : pixels) CHECK(p == 168);
  comp.ToPixels(0, 0, 4, 1, pixels, 1);  // untouched block
  for (uint8_t p : pixels) CHECK(p == 128);
  CHECK_FALSE(comp.IsAllZero());
}

TEST_CASE("OutputImage: JPEG coefficients round-trip through CopyFromJpegData / SaveToJpegData") {
  const int w = 35, h = 19;
  const guetzli::JPEGData jpg = EncodeWithQuant(ColorImage(w, h), w, h, 3);
  guetzli::CpuDctOps dct;
  guetzli::OutputImage img(w, h, dct);
  img.CopyFromJpegData(jpg);
  CHECK(img.FrameTypeStr() == "f111111");
  for (int c = 0; c < 3; ++c) CHECK(img.component(c).quant()[5] == 3);

  guetzli::JPEGData out;
  img.SaveToJpegData(&out);
  REQUIRE(out.components.size() == 3);
  CHECK(out.Is444());
  for (int c = 0; c < 3; ++c) {
    CHECK(out.components[c].coeffs == jpg.components[c].coeffs);
    CHECK(out.quant[out.components[c].quant_idx].values == jpg.quant[jpg.components[c].quant_idx].values);
  }
}

TEST_CASE("OutputImage: global quantization") {
  const int w = 24, h = 16;
  guetzli::CpuDctOps dct;
  guetzli::OutputImage img(w, h, dct);
  img.CopyFromJpegData(EncodeWithQuant(ColorImage(w, h), w, h, 1));
  int q[3][kDCTBlockSize];
  for (int c = 0; c < 3; ++c)
    for (int k = 0; k < kDCTBlockSize; ++k) q[c][k] = 2 + (k + c) % 7;
  img.ApplyGlobalQuantization(q);
  for (int c = 0; c < 3; ++c) {
    const guetzli::OutputImageComponent& comp = img.component(c);
    CHECK(std::equal(&q[c][0], &q[c][0] + kDCTBlockSize, comp.quant()));
    const coeff_t* coeffs = comp.coeffs();
    for (int i = 0; i < comp.width_in_blocks() * comp.height_in_blocks() * kDCTBlockSize; ++i) {
      REQUIRE(coeffs[i] % q[c][i % kDCTBlockSize] == 0);
    }
  }
}

TEST_CASE("OutputImage: downsampling to 4:2:0 and saving the result") {
  const int w = 33, h = 18;
  guetzli::CpuDctOps dct;
  guetzli::OutputImage img(w, h, dct);
  img.CopyFromJpegData(EncodeWithQuant(ColorImage(w, h), w, h, 1));
  img.Downsample(guetzli::OutputImage::DownsampleConfig{});
  CHECK(img.FrameTypeStr() == "f112222");
  CHECK(img.component(1).width_in_blocks() == 3);
  CHECK(img.component(1).height_in_blocks() == 2);

  guetzli::JPEGData out;
  img.SaveToJpegData(&out);
  CHECK(out.max_h_samp_factor == 2);
  CHECK(out.max_v_samp_factor == 2);
  CHECK(out.Is420());
  CHECK(out.components[0].width_in_blocks == 6);
  CHECK(out.components[1].width_in_blocks == 3);
}

TEST_CASE("OutputImage: an image without chroma is saved as grayscale") {
  const int w = 16, h = 16;
  std::vector<uint8_t> gray(3 * w * h);
  for (int i = 0; i < w * h; ++i) gray[3 * i] = gray[3 * i + 1] = gray[3 * i + 2] = static_cast<uint8_t>(i);
  guetzli::CpuDctOps dct;
  guetzli::OutputImage img(w, h, dct);
  img.CopyFromJpegData(EncodeWithQuant(gray, w, h, 1));
  REQUIRE(img.component(1).IsAllZero());
  REQUIRE(img.component(2).IsAllZero());
  guetzli::JPEGData out;
  img.SaveToJpegData(&out);
  CHECK(out.components.size() == 1);
}

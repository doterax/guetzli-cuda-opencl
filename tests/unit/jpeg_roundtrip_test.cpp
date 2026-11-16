// Encode RGB -> JPEGData -> bytes -> JPEGData -> RGB.
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "doctest.h"
#include "core/jpeg/jpeg_data.h"
#include "core/image/jpeg_data_decoder.h"
#include "core/image/jpeg_data_encoder.h"
#include "core/jpeg/jpeg_data_reader.h"
#include "core/jpeg/jpeg_data_writer.h"

namespace {

int AppendToString(void* data, const uint8_t* buf, size_t len) {
  static_cast<std::string*>(data)->append(reinterpret_cast<const char*>(buf), len);
  return static_cast<int>(len);
}

std::string Serialize(const guetzli::JPEGData& jpg) {
  std::string out;
  REQUIRE(guetzli::WriteJpeg(jpg, false, guetzli::JPEGOutput(AppendToString, &out)));
  return out;
}

std::vector<uint8_t> SmoothImage(int w, int h) {
  std::vector<uint8_t> rgb(3 * w * h);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      uint8_t* p = &rgb[3 * (y * w + x)];
      p[0] = static_cast<uint8_t>(x * 255 / (w - 1));
      p[1] = static_cast<uint8_t>(y * 255 / (h - 1));
      p[2] = static_cast<uint8_t>(128 + (x - y) / 4);
    }
  return rgb;
}

}  // namespace

TEST_CASE("encode, write, read and decode an odd-sized image") {
  const int w = 37, h = 21;
  const std::vector<uint8_t> rgb = SmoothImage(w, h);
  int quant[3 * guetzli::kDCTBlockSize];
  for (int i = 0; i < 3 * guetzli::kDCTBlockSize; ++i) quant[i] = 1;

  guetzli::JPEGData jpg;
  REQUIRE(guetzli::EncodeRGBToJpeg(rgb, w, h, quant, &jpg));
  const std::string bytes = Serialize(jpg);

  guetzli::JPEGData read;
  REQUIRE(guetzli::ReadJpeg(bytes, guetzli::JPEG_READ_ALL, &read));
  CHECK(read.width == w);
  CHECK(read.height == h);
  CHECK(read.components.size() == 3);

  guetzli::CpuDctOps dct;
  const std::vector<uint8_t> decoded = guetzli::DecodeJpegToRGB(read, dct);
  REQUIRE(decoded.size() == rgb.size());
  int max_err = 0;
  for (size_t i = 0; i < rgb.size(); ++i) max_err = std::max(max_err, std::abs(decoded[i] - rgb[i]));
  CHECK(max_err <= 4);  // quantizer 1: only color-transform and IDCT rounding
}

TEST_CASE("write(read(write(x))) is byte-stable") {
  const int w = 24, h = 16;
  guetzli::JPEGData jpg;
  REQUIRE(guetzli::EncodeRGBToJpeg(SmoothImage(w, h), w, h, &jpg));
  const std::string first = Serialize(jpg);

  guetzli::JPEGData read;
  REQUIRE(guetzli::ReadJpeg(first, guetzli::JPEG_READ_ALL, &read));
  CHECK(Serialize(read) == first);
}

TEST_CASE("truncated input is rejected") {
  const int w = 16, h = 16;
  guetzli::JPEGData jpg;
  REQUIRE(guetzli::EncodeRGBToJpeg(SmoothImage(w, h), w, h, &jpg));
  const std::string bytes = Serialize(jpg);
  guetzli::JPEGData read;
  CHECK_FALSE(guetzli::ReadJpeg(bytes.substr(0, bytes.size() / 2), guetzli::JPEG_READ_ALL, &read));
}

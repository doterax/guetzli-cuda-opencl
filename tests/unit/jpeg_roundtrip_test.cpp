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

int appendToString(void* data, const uint8_t* buf, size_t len) {
  static_cast<std::string*>(data)->append(reinterpret_cast<const char*>(buf), len);
  return static_cast<int>(len);
}

std::string serialize(const guetzli::JpegData& jpg) {
  std::string out;
  REQUIRE(guetzli::writeJpeg(jpg, false, guetzli::JpegOutput(appendToString, &out)));
  return out;
}

std::vector<uint8_t> smoothImage(int w, int h) {
  std::vector<uint8_t> rgb(3 * w * h);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      uint8_t* p = &rgb[3 * (y * w + x)];
      p[0] = static_cast<uint8_t>(x * 255 / (w - 1));
      p[1] = static_cast<uint8_t>(y * 255 / (h - 1));
      p[2] = static_cast<uint8_t>(128 + (x - y) / 4);
    }
  }
  return rgb;
}

}  // namespace

TEST_CASE("encode, write, read and decode an odd-sized image") {
  const int w = 37;
  const int h = 21;
  const std::vector<uint8_t> rgb = smoothImage(w, h);
  int quant[3 * guetzli::kDctBlockSize];
  for (int& i : quant) {
    i = 1;
  }

  guetzli::JpegData jpg;
  REQUIRE(guetzli::encodeRgbToJpeg(rgb, w, h, quant, &jpg));
  const std::string bytes = serialize(jpg);

  guetzli::JpegData read;
  REQUIRE(guetzli::readJpeg(bytes, guetzli::JPEG_READ_ALL, &read));
  CHECK(read.width == w);
  CHECK(read.height == h);
  CHECK(read.components.size() == 3);

  guetzli::CpuDctOps dct;
  const std::vector<uint8_t> decoded = guetzli::decodeJpegToRgb(read, dct);
  REQUIRE(decoded.size() == rgb.size());
  int max_err = 0;
  for (size_t i = 0; i < rgb.size(); ++i) {
    max_err = std::max(max_err, std::abs(decoded[i] - rgb[i]));
  }
  CHECK(max_err <= 4);  // quantizer 1: only color-transform and IDCT rounding
}

TEST_CASE("write(read(write(x))) is byte-stable") {
  const int w = 24;
  const int h = 16;
  guetzli::JpegData jpg;
  REQUIRE(guetzli::encodeRgbToJpeg(smoothImage(w, h), w, h, &jpg));
  const std::string first = serialize(jpg);

  guetzli::JpegData read;
  REQUIRE(guetzli::readJpeg(first, guetzli::JPEG_READ_ALL, &read));
  CHECK(serialize(read) == first);
}

TEST_CASE("truncated input is rejected") {
  const int w = 16;
  const int h = 16;
  guetzli::JpegData jpg;
  REQUIRE(guetzli::encodeRgbToJpeg(smoothImage(w, h), w, h, &jpg));
  const std::string bytes = serialize(jpg);
  guetzli::JpegData read;
  CHECK_FALSE(guetzli::readJpeg(bytes.substr(0, bytes.size() / 2), guetzli::JPEG_READ_ALL, &read));
}

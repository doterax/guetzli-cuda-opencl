// core/io: format detection, PNG decoding and alpha blending, file I/O.
#include <cstdio>
#include <string>

#include "core/io/image_io.h"
#include "doctest.h"

#ifndef GUETZLI_SOURCE_DIR
#error "GUETZLI_SOURCE_DIR must point to the source tree (set by CMake)"
#endif

namespace {

std::string sourceFile(const char* relative) {
  return std::string(GUETZLI_SOURCE_DIR) + "/" + relative;
}

}  // namespace

TEST_CASE("format detection") {
  const std::string png = guetzli::readFile(sourceFile("tests/regress/input/rgb_150x118.png"));
  const std::string jpg = guetzli::readFile(sourceFile("tests/regress/input/yuv444_136x120.jpg"));
  CHECK(guetzli::isPng(png));
  CHECK_FALSE(guetzli::isTiff(png));
  CHECK_FALSE(guetzli::isPng(jpg));
  CHECK_FALSE(guetzli::isTiff(jpg));
  CHECK(guetzli::isTiff(std::string("II*\0", 4)));
  CHECK(guetzli::isTiff(std::string("MM\0*", 4)));
  CHECK_FALSE(guetzli::isPng(""));
}

TEST_CASE("decodePng: RGB, grayscale and size") {
  const auto rgb = guetzli::decodePng(guetzli::readFile(sourceFile("tests/regress/input/rgb_odd_121x101.png")),
                                      guetzli::AlphaBackground::Black);
  REQUIRE(rgb);
  CHECK(rgb->width == 121);
  CHECK(rgb->height == 101);
  CHECK(rgb->rgb.size() == 3u * 121 * 101);

  const auto gray = guetzli::decodePng(guetzli::readFile(sourceFile("tests/regress/input/gray_odd_117x105.png")),
                                       guetzli::AlphaBackground::Black);
  REQUIRE(gray);
  for (size_t i = 0; i < gray->rgb.size(); i += 3) {
    CHECK(gray->rgb[i] == gray->rgb[i + 1]);
    CHECK(gray->rgb[i] == gray->rgb[i + 2]);
  }
}

TEST_CASE("decodePng: alpha is blended on black or white") {
  const std::string data = guetzli::readFile(sourceFile("tests/regress/input/rgba_alpha_128x112.png"));
  const auto black = guetzli::decodePng(data, guetzli::AlphaBackground::Black);
  const auto white = guetzli::decodePng(data, guetzli::AlphaBackground::White);
  REQUIRE(black);
  REQUIRE(white);
  REQUIRE(black->rgb.size() == white->rgb.size());
  bool any_different = false;
  for (size_t i = 0; i < black->rgb.size(); ++i) {
    CHECK(white->rgb[i] >= black->rgb[i]);  // white background can only make pixels lighter
    any_different |= white->rgb[i] != black->rgb[i];
  }
  CHECK(any_different);
}

TEST_CASE("decodePng rejects truncated data") {
  const std::string data = guetzli::readFile(sourceFile("tests/regress/input/rgb_150x118.png"));
  CHECK_FALSE(guetzli::decodePng(data.substr(0, data.size() / 2), guetzli::AlphaBackground::Black));
}

TEST_CASE("readFile / writeFile round trip and errors") {
  const std::string path = std::string(std::tmpnam(nullptr)) + ".guetzli_test";
  const std::string contents("binary\0data\r\n", 13);
  guetzli::writeFile(path, contents);
  CHECK(guetzli::readFile(path) == contents);
  std::remove(path.c_str());

  CHECK_THROWS_AS((void)guetzli::readFile(path), guetzli::IoError);
  CHECK_THROWS_AS(guetzli::writeFile(sourceFile("no/such/dir/out.jpg"), "x"), guetzli::IoError);
}

// core/io: format detection, PNG decoding and alpha blending, file I/O.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "core/io/image_io.h"
#include "doctest.h"
#include "test_support.h"

using guetzli_test::sourceFile;

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

TEST_CASE("decodeTiff matches decodePng for the same RGBA image, on black and on white") {
  // Same pixels saved as PNG and as TIFF (unassociated alpha, every alpha value 0..255).
  // libtiff returns premultiplied samples; the result must still equal the PNG compositing
  // up to rounding.
  guetzli::NullLogger logger;
  const std::string png = guetzli::readFile(sourceFile("tests/unit/data/rgba_ramp_64x16.png"));
  const std::string tif = guetzli::readFile(sourceFile("tests/unit/data/rgba_ramp_64x16.tif"));
  REQUIRE(guetzli::isTiff(tif));
  for (const guetzli::AlphaBackground bg : {guetzli::AlphaBackground::Black, guetzli::AlphaBackground::White}) {
    CAPTURE(bg == guetzli::AlphaBackground::White);
    const auto from_png = guetzli::decodePng(png, bg);
    const auto from_tif = guetzli::decodeTiff(tif, bg, logger);
    REQUIRE(from_png);
    REQUIRE(from_tif);
    CHECK(from_tif->width == 64);
    CHECK(from_tif->height == 16);
    REQUIRE(from_tif->rgb.size() == from_png->rgb.size());
    int max_diff = 0;
    for (size_t i = 0; i < from_png->rgb.size(); ++i) {
      max_diff = std::max(max_diff, std::abs(from_tif->rgb[i] - from_png->rgb[i]));
    }
    CHECK(max_diff <= 1);
  }
}

TEST_CASE("decodeTiff of an opaque image ignores the background") {
  guetzli::NullLogger logger;
  // The regression TIFF has partial alpha; check opaque pixels via the ramp's alpha == 255 column.
  const std::string tif = guetzli::readFile(sourceFile("tests/unit/data/rgba_ramp_64x16.tif"));
  const auto black = guetzli::decodeTiff(tif, guetzli::AlphaBackground::Black, logger);
  const auto white = guetzli::decodeTiff(tif, guetzli::AlphaBackground::White, logger);
  REQUIRE(black);
  REQUIRE(white);
  // alpha = (x * 4 + y) % 256 == 255 at x = 63, y = 3.
  const size_t i = 3 * (3 * 64 + 63);
  for (int c = 0; c < 3; ++c) CHECK(black->rgb[i + c] == white->rgb[i + c]);
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

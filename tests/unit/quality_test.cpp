#include "doctest.h"
#include "core/quality/quality.h"

TEST_CASE("butteraugli target decreases as quality increases") {
  double prev = guetzli::ButteraugliScoreForQuality(70);
  for (int q = 71; q <= 110; ++q) {
    const double cur = guetzli::ButteraugliScoreForQuality(q);
    CHECK(cur < prev);
    prev = cur;
  }
}

TEST_CASE("quality is clamped outside [70, 110]") {
  CHECK(guetzli::ButteraugliScoreForQuality(10) == guetzli::ButteraugliScoreForQuality(70));
  CHECK(guetzli::ButteraugliScoreForQuality(200) == guetzli::ButteraugliScoreForQuality(110));
}

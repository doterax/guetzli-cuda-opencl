#include "doctest.h"
#include "core/quality/quality.h"

TEST_CASE("butteraugli target decreases as quality increases") {
  double prev = guetzli::butteraugliScoreForQuality(70);
  for (int q = 71; q <= 110; ++q) {
    const double cur = guetzli::butteraugliScoreForQuality(q);
    CHECK(cur < prev);
    prev = cur;
  }
}

TEST_CASE("quality is clamped outside [70, 110]") {
  CHECK(guetzli::butteraugliScoreForQuality(10) == guetzli::butteraugliScoreForQuality(70));
  CHECK(guetzli::butteraugliScoreForQuality(200) == guetzli::butteraugliScoreForQuality(110));
}

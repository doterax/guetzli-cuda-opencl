// Unit tests for the encoder classes (core/encoder).
#include <algorithm>
#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "backend/cpu/cpu_backend.h"
#include "core/encoder/encoder.h"
#include "core/encoder/frequency_masking.h"
#include "core/encoder/output_selector.h"
#include "core/encoder/quant_matrix_search.h"
#include "core/encoder/size_estimator.h"
#include "doctest.h"
#include "core/image/jpeg_data_encoder.h"
#include "core/jpeg/jpeg_data_reader.h"
#include "core/image/output_image.h"
#include "core/jpeg/quantize.h"
#include "test_support.h"

using guetzli::coeff_t;
using guetzli::kDctBlockSize;
using guetzli::LogLevel;
using guetzli_test::RecordingLogger;

namespace {

std::vector<uint8_t> noisyImage(int w, int h, unsigned seed) {
  std::mt19937 rng(seed);
  std::vector<uint8_t> rgb(3 * w * h);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      uint8_t* p = &rgb[3 * (y * w + x)];
      const int noise = static_cast<int>(rng() % 48);
      p[0] = static_cast<uint8_t>((x * 200 / w + noise) % 256);
      p[1] = static_cast<uint8_t>((y * 180 / h + 2 * noise) % 256);
      p[2] = static_cast<uint8_t>((128 + (x - y) + noise) % 256);
    }
  }
  return rgb;
}

void fillQuant(int q[3][kDctBlockSize], int value) {
  for (int c = 0; c < 3; ++c) {
    for (int k = 0; k < kDctBlockSize; ++k) {
      q[c][k] = value;
    }
  }
}

// Scores an output by a configurable value (default: its size); every distance is OK, block errors up
// to 1 are acceptable, every block may be adjusted upwards and none downwards.
class ScoreComparator final : public guetzli::Comparator {
 public:
  void compare(const guetzli::OutputImage& /*img*/) override {}
  void startBlockComparisons() override {}
  void finishBlockComparisons() override {}
  void switchBlock(int /*block_x*/, int /*block_y*/, int /*factor_x*/, int /*factor_y*/) override {}
  double compareBlock(const guetzli::OutputImage& /*img*/, int /*off_x*/, int /*off_y*/,
                      const coeff_t* /*candidate_block*/, const int /*comp_mask*/) const override {
    return 0;
  }
  [[nodiscard]] double scoreOutputSize(int size) const override { return score >= 0 ? score : size; }
  [[nodiscard]] bool distanceOk(double /*target_mul*/) const override { return true; }
  [[nodiscard]] std::vector<float> distmap() const override { return {}; }
  [[nodiscard]] float distmapAggregate() const override { return 0; }
  [[nodiscard]] float blockErrorLimit() const override { return 1.0f; }
  void computeBlockErrorAdjustmentWeights(int direction, int /*max_block_dist*/, double /*target_mul*/,
                                          int /*factor_x*/, int /*factor_y*/, const std::vector<float>& /*distmap*/,
                                          std::vector<float>* block_weight) override {
    std::ranges::fill(*block_weight, direction > 0 ? 1.0f : 0.0f);
  }

  double score = -1;  // < 0: the output size
};

// Returns fixed block-zeroing orders.
class FixedSearch final : public guetzli::IBlockZeroingSearch {
 public:
  explicit FixedSearch(std::vector<guetzli::CoeffData> orders) : orders_(std::move(orders)) {}
  std::vector<guetzli::CoeffData> run(const guetzli::BlockZeroingInput& /*in*/) override { return orders_; }

 private:
  std::vector<guetzli::CoeffData> orders_;
};

}  // namespace

// ---------------------------------------------------------------------------------------------
// Size estimator
// ---------------------------------------------------------------------------------------------
TEST_CASE("JpegSizeEstimator: incremental updates match a full recomputation") {
  const int w = 61;
  const int h = 45;
  const std::vector<uint8_t> rgb = noisyImage(w, h, 1);
  int quant[3 * kDctBlockSize];
  for (int i = 0; i < 3 * kDctBlockSize; ++i) {
    quant[i] = 2 + i % 5;
  }
  guetzli::JpegData jpg;
  REQUIRE(guetzli::encodeRgbToJpeg(rgb, w, h, quant, &jpg));

  guetzli::CpuDctOps dct;
  guetzli::OutputImage img(w, h, dct);
  img.copyFromJpegData(jpg);
  auto snapshot = [&] {
    guetzli::JpegData out = jpg;
    img.saveToJpegData(&out);
    return out;
  };
  guetzli::JpegSizeEstimator estimator(snapshot(), 3, true);

  std::mt19937 rng(7);
  size_t fixed_part = estimator.estimate() - guetzli::entropyCodedDataSize(estimator.histograms(), estimator.depths());
  for (int step = 1; step <= 400; ++step) {
    const int c = static_cast<int>(rng() % 3);
    guetzli::OutputImageComponent& comp = img.component(c);
    const int bx = static_cast<int>(rng() % comp.widthInBlocks());
    const int by = static_cast<int>(rng() % comp.heightInBlocks());
    const int k = 1 + static_cast<int>(rng() % (kDctBlockSize - 1));
    coeff_t old_block[kDctBlockSize];
    comp.getCoeffBlock(bx, by, old_block);
    coeff_t block[kDctBlockSize];
    std::copy(old_block, old_block + kDctBlockSize, block);
    const int value = static_cast<int>(rng() % 200) - 100;
    block[k] = ((rng() % 2) != 0u) ? 0 : guetzli::quantize(static_cast<coeff_t>(value), comp.quant()[k]);
    estimator.replaceBlock(c, old_block, block, comp.quant());
    comp.setCoeffBlock(bx, by, block);

    // Between refreshes only the entropy-coded part changes, and it equals the full sum.
    CHECK(estimator.estimate() - guetzli::entropyCodedDataSize(estimator.histograms(), estimator.depths()) ==
          fixed_part);
    if (step % 10 == 0) {
      estimator.refreshEntropyCodes();
      // After a refresh the estimate equals one computed from scratch on the changed image.
      const guetzli::JpegSizeEstimator fresh(snapshot(), 3, true);
      REQUIRE(estimator.estimate() == fresh.estimate());
      for (size_t i = 0; i < fresh.histograms().size(); ++i) {
        for (int s = 0; s < guetzli::JpegHistogram::kSize; ++s) {
          REQUIRE(estimator.histograms()[i].counts[s] == fresh.histograms()[i].counts[s]);
        }
      }
      fixed_part = estimator.estimate() - guetzli::entropyCodedDataSize(estimator.histograms(), estimator.depths());
    }
  }
}

TEST_CASE("JpegSizeEstimator: extra histograms start empty") {
  const int w = 16;
  const int h = 16;
  guetzli::JpegData jpg;
  REQUIRE(guetzli::encodeRgbToJpeg(std::vector<uint8_t>(3 * w * h, 128), w, h, &jpg));
  jpg.components.resize(1);  // a grayscale file
  const guetzli::JpegSizeEstimator estimator(jpg, 3, true);
  REQUIRE(estimator.histograms().size() == 3);
  CHECK(estimator.histograms()[2].numSymbols() == 0);
  CHECK(estimator.estimate() > 0);
}

// ---------------------------------------------------------------------------------------------
// Quantization matrix search
// ---------------------------------------------------------------------------------------------
TEST_CASE("CompareQuantMatrices orders matrices coordinate-wise") {
  int a[3][kDctBlockSize];
  int b[3][kDctBlockSize];
  fillQuant(a, 4);
  fillQuant(b, 4);
  CHECK(guetzli::compareQuantMatrices(&a[0][0], &b[0][0]) == 0);
  b[2][63] = 5;
  CHECK(guetzli::compareQuantMatrices(&a[0][0], &b[0][0]) == -1);
  CHECK(guetzli::compareQuantMatrices(&b[0][0], &a[0][0]) == 1);
  a[0][0] = 5;
  CHECK(guetzli::compareQuantMatrices(&a[0][0], &b[0][0]) == 2);
}

TEST_CASE("QuantMatrixHeuristicScore is zero for the identity matrix and grows with the quantizers") {
  int q[3][kDctBlockSize];
  fillQuant(q, 1);
  CHECK(guetzli::quantMatrixHeuristicScore(q) == 0.0);
  fillQuant(q, 3);
  const double s3 = guetzli::quantMatrixHeuristicScore(q);
  fillQuant(q, 5);
  CHECK(guetzli::quantMatrixHeuristicScore(q) > s3);
  CHECK(s3 > 0.0);
}

TEST_CASE("CompareQuantData prefers results within the target, then smaller files") {
  guetzli::QuantData ok_big{};
  ok_big.dist_ok = true;
  ok_big.jpg_size = 1000;
  guetzli::QuantData ok_small = ok_big;
  ok_small.jpg_size = 900;
  guetzli::QuantData bad_tiny = ok_big;
  bad_tiny.dist_ok = false;
  bad_tiny.jpg_size = 10;
  CHECK(guetzli::compareQuantData(ok_small, ok_big));
  CHECK_FALSE(guetzli::compareQuantData(ok_big, ok_small));
  CHECK(guetzli::compareQuantData(ok_big, bad_tiny));
  CHECK_FALSE(guetzli::compareQuantData(bad_tiny, ok_big));
}

TEST_CASE("QuantMatrixGenerator: binary search between the coarsest good and finest bad matrix") {
  int q[3][kDctBlockSize];
  SUBCASE("4:2:0 starts at the identity matrix") {
    guetzli::QuantMatrixGenerator gen(true);
    REQUIRE(gen.getNext(q));
    for (auto& c : q) {
      for (int& k : c) {
        CHECK(k == 1);
      }
    }
  }
  SUBCASE("terminates and never proposes a matrix twice") {
    guetzli::QuantMatrixGenerator gen(false);
    std::vector<guetzli::QuantData> tried;
    int iterations = 0;
    while (gen.getNext(q)) {
      REQUIRE(++iterations < 100);
      guetzli::QuantData data{};
      std::copy(&q[0][0], &q[0][0] + 3 * kDctBlockSize, &data.q[0][0]);
      for (const guetzli::QuantData& t : tried) {
        CHECK(guetzli::compareQuantMatrices(&t.q[0][0], &data.q[0][0]) != 0);
      }
      // Pretend matrices up to a heuristic score of 30 stay within the target.
      data.dist_ok = guetzli::quantMatrixHeuristicScore(q) <= 30.0;
      data.jpg_size = 1;
      tried.push_back(data);
      gen.add(data);
    }
    CHECK(iterations > 2);
  }
}

// ---------------------------------------------------------------------------------------------
// Frequency masking and output selection
// ---------------------------------------------------------------------------------------------
TEST_CASE("CollectZeroingCandidates keeps the entries within the error limit, per block") {
  std::vector<guetzli::CoeffData> orders(2 * guetzli::kBlockZeroingBlockSize,
                                         guetzli::CoeffData{.idx = 0, .block_err = 0.0f});
  orders[0] = {.idx = 5, .block_err = 0.1f};
  orders[1] = {.idx = 6, .block_err = 0.5f};
  orders[2] = {.idx = 7, .block_err = 2.0f};  // above the limit
  orders[guetzli::kBlockZeroingBlockSize + 3] = {.idx = 130, .block_err = 0.2f};
  const guetzli::ZeroingCandidates c = guetzli::collectZeroingCandidates(orders, 2, 1.0f);
  CHECK(c.offsets == std::vector<int>{0, 2, 3});
  CHECK(c.coeffs == std::vector<uint8_t>{5, 6, 130});
  CHECK(c.errors == std::vector<float>{0.1f, 0.5f, 0.2f});
}

TEST_CASE("FrequencyMasking zeroes the candidate in its own block of a 4:2:0 chroma component") {
  const int w = 64;
  const int h = 48;  // chroma: 4x3 blocks of 16x16 pixels
  int ones[3 * kDctBlockSize];
  std::fill(ones, ones + 3 * kDctBlockSize, 1);  // as after RemoveOriginalQuantization
  guetzli::JpegData jpg;
  REQUIRE(guetzli::encodeRgbToJpeg(noisyImage(w, h, 13), w, h, ones, &jpg));
  guetzli::CpuDctOps dct;
  guetzli::OutputImage img(w, h, dct);
  img.copyFromJpegData(jpg);
  img.downsample(guetzli::OutputImage::DownsampleConfig{});
  img.saveToJpegData(&jpg);
  const guetzli::OutputImageComponent& cb = img.component(1);
  REQUIRE(cb.widthInBlocks() == 4);
  REQUIRE(cb.heightInBlocks() == 3);

  // One candidate: a nonzero AC coefficient of Cb block (3, 1).
  const int bx = 3;
  const int by = 1;
  coeff_t target[kDctBlockSize];
  cb.getCoeffBlock(bx, by, target);
  int k = 1;
  while (k < kDctBlockSize && target[k] == 0) {
    ++k;
  }
  REQUIRE(k < kDctBlockSize);
  std::vector<guetzli::CoeffData> orders(12 * guetzli::kBlockZeroingBlockSize,
                                         guetzli::CoeffData{.idx = 0, .block_err = 0.0f});
  orders[(by * 4 + bx) * guetzli::kBlockZeroingBlockSize] = {.idx = kDctBlockSize + k, .block_err = 0.5f};

  std::vector<coeff_t> before(cb.coeffs(), cb.coeffs() + 12 * kDctBlockSize);
  guetzli::ProcessStats stats;
  guetzli::OutputSelector selector(true, stats);
  ScoreComparator comparator;
  FixedSearch search(orders);
  const guetzli::EncoderParams params;
  guetzli::FrequencyMasking masking(params, search, comparator, selector, stats);
  masking.run(jpg, img, 6, 1.0, false);

  const std::vector<coeff_t> after(cb.coeffs(), cb.coeffs() + 12 * kDctBlockSize);
  CHECK(after[(by * 4 + bx) * kDctBlockSize + k] == 0);
  // Block (bx / 2, by / 2) is not touched.
  const int other = (by / 2) * 4 + bx / 2;
  CHECK(std::equal(&after[other * kDctBlockSize], &after[(other + 1) * kDctBlockSize], &before[other * kDctBlockSize]));
  // Nothing else changes either; in particular the blocks after (bx, by), which have no candidates.
  std::vector<coeff_t> expected = before;
  expected[(by * 4 + bx) * kDctBlockSize + k] = 0;
  CHECK(after == expected);
  CHECK(selector.hasOutput());
}

TEST_CASE("OutputSelector keeps the output with the lowest score") {
  guetzli::ProcessStats stats;
  guetzli::OutputSelector selector(true, stats);
  ScoreComparator comparator;
  CHECK_FALSE(selector.hasOutput());
  comparator.score = 5;
  selector.offer("first", comparator);
  CHECK(selector.best() == "first");
  comparator.score = 7;
  selector.offer("worse", comparator);
  CHECK(selector.best() == "first");
  comparator.score = 2;
  selector.offer("better", comparator);
  CHECK(selector.best() == "better");
  CHECK(selector.bestScore() == 2);
  CHECK(selector.takeBest() == "better");
}

TEST_CASE("OutputSelector::serialize writes a readable JPEG") {
  guetzli::ProcessStats stats;
  const guetzli::OutputSelector selector(true, stats);
  guetzli::JpegData jpg;
  REQUIRE(guetzli::encodeRgbToJpeg(noisyImage(20, 12, 3), 20, 12, &jpg));
  guetzli::JpegData read;
  REQUIRE(guetzli::readJpeg(selector.serialize(jpg), guetzli::JPEG_READ_ALL, &read));
  CHECK(read.width == 20);
  CHECK(read.height == 12);
}

// ---------------------------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------------------------
TEST_CASE("Encoder: encodes RGB pixels on the CPU backend") {
  guetzli::Backend backend = guetzli::makeCpuBackend();
  RecordingLogger logger(LogLevel::Debug);
  guetzli::EncoderParams params;
  params.butteraugli_target = 2.0f;
  guetzli::Encoder encoder(params, backend, logger);
  const int w = 40;
  const int h = 36;
  const std::optional<std::string> jpeg = encoder.encodeRgb(noisyImage(w, h, 5), w, h);
  REQUIRE(jpeg);
  guetzli::JpegData read;
  REQUIRE(guetzli::readJpeg(*jpeg, guetzli::JPEG_READ_ALL, &read));
  CHECK(read.width == w);
  CHECK(read.height == h);
  CHECK(encoder.stats().counters.at(guetzli::kNumItersCnt) > 0);
  const std::string trace = logger.text(LogLevel::Debug);
  CHECK(trace.find("Original Out[") != std::string::npos);
  CHECK(trace.find("selected quantization matrix") != std::string::npos);
  CHECK(logger.text(LogLevel::Error).empty());

  SUBCASE("re-encoding its own output works too") {
    CHECK(encoder.encodeJpeg(*jpeg));
  }
}

TEST_CASE("Encoder: images too small for Butteraugli are returned as encoded") {
  guetzli::Backend backend = guetzli::makeCpuBackend();
  RecordingLogger logger(LogLevel::Debug);
  guetzli::Encoder encoder(guetzli::EncoderParams{}, backend, logger);
  const std::optional<std::string> jpeg = encoder.encodeRgb(noisyImage(31, 40, 9), 31, 40);
  REQUIRE(jpeg);
  CHECK(logger.text(LogLevel::Debug).find("too small for Butteraugli") != std::string::npos);
}

TEST_CASE("Encoder: invalid JPEG data is reported") {
  guetzli::Backend backend = guetzli::makeCpuBackend();
  RecordingLogger logger;
  guetzli::Encoder encoder(guetzli::EncoderParams{}, backend, logger);
  CHECK_FALSE(encoder.encodeJpeg("\xFF\xD8 not a jpeg"));
  CHECK(logger.text(LogLevel::Error).find("Can't read jpg data") != std::string::npos);
}

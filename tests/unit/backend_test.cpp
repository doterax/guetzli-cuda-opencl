// Backend selection (fake probes) and the --check decorators (fake implementations).
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "backend/check/checking_block_zeroing.h"
#include "backend/check/checking_comparator.h"
#include "backend/check/checking_dct_ops.h"
#include "backend/factory.h"
#include "doctest.h"
#include "guetzli/output_image.h"
#include "test_support.h"

using guetzli::BackendKind;
using guetzli::coeff_t;
using guetzli::LogLevel;
using guetzli_test::RecordingLogger;

namespace {

struct ProbeLog {
  int cuda_calls = 0;
  int opencl_calls = 0;
};

guetzli::BackendProbes fakeProbes(ProbeLog& log, std::optional<bool> cuda, std::optional<bool> opencl) {
  guetzli::BackendProbes probes;
  if (cuda)
    probes.cuda = [&log, v = *cuda] {
      ++log.cuda_calls;
      return v;
    };
  if (opencl)
    probes.opencl = [&log, v = *opencl] {
      ++log.opencl_calls;
      return v;
    };
  return probes;
}

}  // namespace

TEST_CASE("an explicit backend is used as is, without probing") {
  for (BackendKind kind : {BackendKind::Cpu, BackendKind::OpenCl, BackendKind::Cuda}) {
    ProbeLog log;
    RecordingLogger logger;
    CHECK(guetzli::resolveBackendKind(kind, fakeProbes(log, true, true), logger) == kind);
    CHECK(log.cuda_calls == 0);
    CHECK(log.opencl_calls == 0);
    CHECK(logger.lines.empty());
  }
}

TEST_CASE("auto tries CUDA, then OpenCL, then falls back to the CPU") {
  struct Case {
    std::optional<bool> cuda, opencl;
    BackendKind expected;
    int cuda_calls, opencl_calls;
  };
  const Case cases[] = {
      {true, true, BackendKind::Cuda, 1, 0},
      {false, true, BackendKind::OpenCl, 1, 1},
      {false, false, BackendKind::Cpu, 1, 1},
      {std::nullopt, true, BackendKind::OpenCl, 0, 1},       // CUDA not compiled in
      {std::nullopt, std::nullopt, BackendKind::Cpu, 0, 0},  // no GPU backend compiled in
  };
  for (const Case& c : cases) {
    ProbeLog log;
    RecordingLogger logger;
    CHECK(guetzli::resolveBackendKind(BackendKind::Auto, fakeProbes(log, c.cuda, c.opencl), logger) == c.expected);
    CHECK(log.cuda_calls == c.cuda_calls);
    CHECK(log.opencl_calls == c.opencl_calls);
    const std::string info = logger.text(LogLevel::Info);
    CHECK(info.rfind("Autodetect:\n", 0) == 0);
    if (c.expected == BackendKind::Cpu) CHECK(info.find("Using optimized CPU implementation") != std::string::npos);
  }
}

TEST_CASE("makeBackend: the CPU backend provides every capability; Auto must be resolved first") {
  RecordingLogger logger;
  guetzli::GpuDevices devices(guetzli::BackendOptions{}, logger);  // CPU backends never touch them
  for (bool check : {false, true}) {
    const guetzli::Backend backend = guetzli::makeBackend(BackendKind::Cpu, check, devices, logger);
    CHECK(backend.kind == BackendKind::Cpu);
    CHECK(backend.comparators != nullptr);
    CHECK(backend.block_zeroing != nullptr);
    CHECK(backend.dct != nullptr);
  }
  CHECK_THROWS_AS((void)guetzli::makeBackend(BackendKind::Auto, false, devices, logger), std::logic_error);
}

// ---------------------------------------------------------------------------------------------
// Decorators
// ---------------------------------------------------------------------------------------------
namespace {

class FakeComparator final : public guetzli::Comparator {
 public:
  FakeComparator(std::vector<float> distmap, float distance, std::vector<std::string>& calls, const char* name)
      : distmap_(std::move(distmap)), distance_(distance), calls_(calls), name_(name) {}

  void Compare(const guetzli::OutputImage&) override { record("Compare"); }
  void StartBlockComparisons() override { record("Start"); }
  void FinishBlockComparisons() override { record("Finish"); }
  void SwitchBlock(int, int, int, int) override { record("SwitchBlock"); }
  double CompareBlock(const guetzli::OutputImage&, int, int, const coeff_t*, const int) const override {
    record("CompareBlock");
    return distance_;
  }
  double ScoreOutputSize(int size) const override { return size * distance_; }
  bool DistanceOK(double) const override { return true; }
  const std::vector<float> distmap() const override { return distmap_; }
  float distmap_aggregate() const override { return distance_; }
  float BlockErrorLimit() const override { return distance_; }
  void ComputeBlockErrorAdjustmentWeights(int, int, double, int, int, const std::vector<float>&,
                                          std::vector<float>*) override {}
  std::optional<guetzli::BlockReferenceData> blockReferenceData() const override {
    if (!provides_block_data) return std::nullopt;
    return guetzli::BlockReferenceData{distmap_, distmap_};
  }

  bool provides_block_data = false;

 private:
  void record(const char* what) const { calls_.push_back(std::string(name_) + "." + what); }

  std::vector<float> distmap_;
  float distance_;
  std::vector<std::string>& calls_;
  const char* name_;
};

int countChk(const RecordingLogger& logger) {
  int n = 0;
  for (const auto& [level, message] : logger.lines) {
    if (level == LogLevel::Error && message.rfind("CHK ", 0) == 0) ++n;
  }
  return n;
}

}  // namespace

TEST_CASE("CheckingComparator: runs both, keeps the reference result, reports differing distance maps") {
  guetzli::CpuDctOps dct;
  const guetzli::OutputImage img(8, 8, dct);

  struct Case {
    std::vector<float> tested;
    int expected_chk;
  };
  const Case cases[] = {
      {{1.0f, 2.0f, 3.0f}, 0},
      {{1.0f, 2.004f, 3.0f}, 0},  // within the tolerance
      {{1.0f, 2.5f, 3.0f}, 1},
  };
  for (const Case& c : cases) {
    std::vector<std::string> calls;
    auto tested = std::make_unique<FakeComparator>(c.tested, 9.0f, calls, "gpu");
    tested->provides_block_data = true;
    auto reference = std::make_unique<FakeComparator>(std::vector<float>{1.0f, 2.0f, 3.0f}, 9.0f, calls, "cpu");
    RecordingLogger logger;
    guetzli::CheckingComparator checking(std::move(tested), std::move(reference), logger);

    checking.Compare(img);
    CHECK(checking.distmap() == std::vector<float>{1.0f, 2.0f, 3.0f});
    CHECK(countChk(logger) == c.expected_chk);
    CHECK(calls == std::vector<std::string>{"gpu.Compare", "cpu.Compare"});

    // Per-block work runs on the reference only; block data comes from the tested comparator.
    calls.clear();
    checking.StartBlockComparisons();
    checking.SwitchBlock(0, 0, 1, 1);
    (void)checking.CompareBlock(img, 0, 0, nullptr, 7);
    checking.FinishBlockComparisons();
    CHECK(calls == std::vector<std::string>{"gpu.Start", "cpu.Start", "cpu.SwitchBlock", "cpu.CompareBlock",
                                            "gpu.Finish", "cpu.Finish"});
    CHECK(checking.blockReferenceData().has_value());
  }
}

namespace {

class FakeSearch final : public guetzli::IBlockZeroingSearch {
 public:
  explicit FakeSearch(std::vector<guetzli::CoeffData> result) : result_(std::move(result)) {}
  std::vector<guetzli::CoeffData> run(const guetzli::BlockZeroingInput&) override {
    ++runs;
    return result_;
  }
  int runs = 0;

 private:
  std::vector<guetzli::CoeffData> result_;
};

}  // namespace

TEST_CASE("CheckingBlockZeroingSearch: returns the reference orders and reports differences") {
  guetzli::CpuDctOps dct;
  guetzli::OutputImage img(8, 8, dct);
  guetzli::JPEGData jpg;
  std::vector<std::string> calls;
  FakeComparator comparator({}, 1.0f, calls, "c");
  const guetzli::BlockZeroingInput input{jpg, img, 7, comparator, 3, true};

  const std::vector<guetzli::CoeffData> reference_orders = {{5, 0.5f}, {9, 0.75f}};
  struct Case {
    std::vector<guetzli::CoeffData> tested;
    int expected_chk;
  };
  const Case cases[] = {
      {{{5, 0.5f}, {9, 0.75f}}, 0},
      {{{5, 0.5f}, {9, 0.7505f}}, 0},  // within 0.001
      {{{5, 0.5f}, {8, 0.75f}}, 1},    // different coefficient
      {{{5, 0.5f}}, 1},                // different size
  };
  for (const Case& c : cases) {
    auto tested = std::make_unique<FakeSearch>(c.tested);
    auto reference = std::make_unique<FakeSearch>(reference_orders);
    FakeSearch* tested_ptr = tested.get();
    FakeSearch* reference_ptr = reference.get();
    RecordingLogger logger;
    guetzli::CheckingBlockZeroingSearch checking(std::move(tested), std::move(reference), logger);
    const std::vector<guetzli::CoeffData> result = checking.run(input);
    REQUIRE(result.size() == reference_orders.size());
    CHECK(result[1].idx == 9);
    CHECK(tested_ptr->runs == 1);
    CHECK(reference_ptr->runs == 1);
    CHECK(countChk(logger) == c.expected_chk);
  }
}

namespace {

// CPU results, with one coefficient changed if `corrupt` is set; counts sRGB conversions.
class FakeDctOps final : public guetzli::IDctOps {
 public:
  explicit FakeDctOps(bool corrupt) : corrupt_(corrupt) {}
  void copyFromJpegComponent(guetzli::OutputImageComponent& c, const guetzli::JPEGComponent& comp, int factor_x,
                             int factor_y, const int* quant) override {
    cpu_.copyFromJpegComponent(c, comp, factor_x, factor_y, quant);
    if (corrupt_) {
      coeff_t block[guetzli::kDCTBlockSize];
      c.GetCoeffBlock(0, 0, block);
      block[1] += quant[1];
      c.SetCoeffBlock(0, 0, block);
    }
  }
  void applyGlobalQuantization(guetzli::OutputImageComponent& c, const int q[guetzli::kDCTBlockSize]) override {
    cpu_.applyGlobalQuantization(c, q);
  }
  void toSrgb(const guetzli::OutputImage& img, int xmin, int ymin, int xsize, int ysize,
              std::vector<uint8_t>& rgb) override {
    ++srgb_calls;
    cpu_.toSrgb(img, xmin, ymin, xsize, ysize, rgb);
  }
  int srgb_calls = 0;

 private:
  bool corrupt_;
  guetzli::CpuDctOps cpu_;
};

guetzli::JPEGComponent flatComponent(int blocks_x, int blocks_y) {
  guetzli::JPEGComponent comp;
  comp.width_in_blocks = blocks_x;
  comp.height_in_blocks = blocks_y;
  comp.coeffs.assign(static_cast<size_t>(blocks_x) * blocks_y * guetzli::kDCTBlockSize, 0);
  for (size_t i = 0; i < comp.coeffs.size(); i += guetzli::kDCTBlockSize) comp.coeffs[i] = 3;  // DC only
  return comp;
}

}  // namespace

TEST_CASE("CheckingDctOps: keeps the reference result; single blocks are converted by the reference only") {
  const int quant[guetzli::kDCTBlockSize] = {
      8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
      8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
  };
  const guetzli::JPEGComponent comp = flatComponent(2, 2);

  for (bool corrupt : {false, true}) {
    auto tested = std::make_unique<FakeDctOps>(corrupt);
    FakeDctOps* tested_ptr = tested.get();
    RecordingLogger logger;
    guetzli::CheckingDctOps checking(std::move(tested), std::make_unique<guetzli::CpuDctOps>(), logger);

    guetzli::OutputImage img(16, 16, checking);
    guetzli::OutputImageComponent& c = img.component(0);
    checking.copyFromJpegComponent(c, comp, 1, 1, quant);
    coeff_t block[guetzli::kDCTBlockSize];
    c.GetCoeffBlock(0, 0, block);
    CHECK(block[1] == 0);                          // the reference result, not the corrupted one
    CHECK(countChk(logger) == (corrupt ? 2 : 0));  // coefficients and pixels differ

    (void)img.ToSRGB(0, 0, 8, 8);
    CHECK(tested_ptr->srgb_calls == 0);
    (void)img.ToSRGB();
    CHECK(tested_ptr->srgb_calls == 1);
  }
}

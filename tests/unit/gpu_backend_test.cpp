// The shared GPU code in backend/gpu, with fake launchers instead of a device.
#include <memory>
#include <vector>

#include "backend/gpu/constant_cache.h"
#include "backend/gpu/gpu_backend.h"
#include "backend/gpu/gpu_kernels.h"
#include "doctest.h"
#include "core/image/output_image.h"

namespace {

// A fake device allocation: counts live instances.
struct FakeMemory {
  static inline int live = 0;
  int id;
  explicit FakeMemory(int i) : id(i) { ++live; }
  FakeMemory(FakeMemory&& other) noexcept : id(other.id) {
    other.id = -1;
    ++live;
  }
  FakeMemory(const FakeMemory&) = delete;
  FakeMemory& operator=(const FakeMemory&) = delete;
  FakeMemory& operator=(FakeMemory&&) = delete;
  ~FakeMemory() { --live; }
  [[nodiscard]] int get() const { return id; }
};

struct Calls {
  int start_compare = 0;
  int compare = 0;
  int opsin = 0;
  int mask = 0;
};

class FakeSession final : public guetzli::GpuCompareSession {
 public:
  explicit FakeSession(Calls& calls) : calls_(calls) {}
  // The map is 0.25 everywhere except pixel 5 (in 8x8 block 0), which is 0.75.
  void compare(const guetzli::OutputImage& img, int /*step*/, float* block_max8) override {
    ++calls_.compare;
    width_ = img.width();
    height_ = img.height();
    const size_t n8 = static_cast<size_t>((width_ + 7) / 8) * ((height_ + 7) / 8);
    for (size_t i = 0; i < n8; ++i) {
      block_max8[i] = (i == 0) ? 0.75f : 0.25f;
    }
  }
  void readDistmap(float* distmap) override {
    const size_t n = static_cast<size_t>(width_) * height_;
    for (size_t i = 0; i < n; ++i) {
      distmap[i] = (i == 5) ? 0.75f : 0.25f;
    }
  }

 private:
  Calls& calls_;
  int width_ = 0;
  int height_ = 0;
};

class FakeKernels final : public guetzli::GpuKernels {
 public:
  Calls calls;

  void opsinDynamicsImage(float* /*r*/, float* /*g*/, float* /*b*/, unsigned int /*xsize*/,
                          unsigned int /*ysize*/) override {
    ++calls.opsin;
  }
  void mask(float* /*mask_r*/, float* /*mask_g*/, float* /*mask_b*/, float* /*maskdc_r*/, float* /*maskdc_g*/,
            float* /*maskdc_b*/, unsigned int /*xsize*/, unsigned int /*ysize*/, const float* /*r*/, const float* /*g*/,
            const float* /*b*/, const float* /*r2*/, const float* /*g2*/, const float* /*b2*/) override {
    ++calls.mask;
  }
  std::unique_ptr<guetzli::GpuCompareSession> startCompare(const float* /*r*/, const float* /*g*/, const float* /*b*/,
                                                           int /*xsize*/, int /*ysize*/) override {
    ++calls.start_compare;
    return std::make_unique<FakeSession>(calls);
  }
  void copyFromJpegComponent(guetzli::coeff_t* /*output_batch*/, uint8_t* /*output_idct*/,
                             const guetzli::coeff_t* /*jpeg_batch*/, const int* /*quant*/, int /*jpeg_block_width*/,
                             int /*jpeg_block_height*/, int /*output_block_width*/, int /*output_block_height*/,
                             int /*output_width*/, int /*output_height*/) override {}
  void applyGlobalQuantization(guetzli::coeff_t* /*output_batch*/, uint8_t* /*output_idct*/, uint8_t* /*output_bool*/,
                               const int* /*q*/, int /*block_width*/, int /*block_height*/) override {}
  void componentsToPixels(uint8_t* /*rgb*/, int /*xmin*/, int /*ymin*/, int /*xsize*/, int /*ysize*/,
                          const std::vector<guetzli::OutputImageComponent>& /*components*/) override {}
  void computeBlockZeroingOrder(guetzli::CoeffData* /*output_order_batch*/, const ChannelInfo /*orig_channel*/[3],
                                const float* /*orig_image_batch*/, const float* /*mask_scale*/, int /*image_width*/,
                                int /*image_height*/, const ChannelInfo /*mayout_channel*/[3], int /*factor*/,
                                int /*comp_mask*/, float /*block_error_limit*/) override {}
};

}  // namespace

TEST_CASE("ConstantCache: each table is made once per key and released by clear()") {
  guetzli::ConstantCache<FakeMemory> cache;
  int made = 0;
  const auto make = [&made] { return FakeMemory(++made); };

  CHECK(cache.blurWeights(1.5f, make) == 1);
  CHECK(cache.blurWeights(1.5f, make) == 1);
  CHECK(cache.blurWeights(2.0f, make) == 2);
  CHECK(cache.maskLut(3, make) == 3);
  CHECK(cache.maskLut(3, make) == 3);
  CHECK(cache.srgbToLinear(make) == 4);
  CHECK(cache.srgbToLinear(make) == 4);
  CHECK(cache.convolutionScale({1, 2, 3, 4}, make) == 5);
  CHECK(cache.convolutionScale({1, 2, 3, 5}, make) == 6);
  CHECK(cache.convolutionScale({1, 2, 3, 4}, make) == 5);
  CHECK(made == 6);
  CHECK(FakeMemory::live == 6);

  cache.clear();
  CHECK(FakeMemory::live == 0);
  CHECK(cache.srgbToLinear(make) == 7);  // made again after clear()
  cache.clear();
}

TEST_CASE("GPU backend: one device behind every part; the comparator keeps one compare session per image") {
  auto device = std::make_shared<FakeKernels>();
  guetzli::Backend backend = guetzli::makeGpuBackend(guetzli::BackendKind::OpenCl, device);
  CHECK(backend.kind == guetzli::BackendKind::OpenCl);
  REQUIRE(backend.comparators != nullptr);
  CHECK(backend.block_zeroing != nullptr);
  CHECK(backend.dct != nullptr);
  CHECK(device.use_count() > 1);  // shared by the parts

  // 16x16: below the device threshold, so the image operations use the reference code.
  const int w = 16;
  const int h = 16;
  std::vector<uint8_t> rgb(static_cast<size_t>(w) * h * 3, 128);
  std::unique_ptr<guetzli::Comparator> comparator = backend.comparators->create(w, h, &rgb, 1.0f);
  CHECK(device->calls.opsin == 0);
  CHECK(device->calls.mask == 0);

  guetzli::CpuDctOps dct;
  guetzli::OutputImage img(w, h, dct);
  comparator->compare(img);
  comparator->compare(img);
  CHECK(device->calls.start_compare == 1);
  CHECK(device->calls.compare == 2);
  CHECK(comparator->distmap().size() == static_cast<size_t>(w) * h);
  CHECK(comparator->distmap()[5] == 0.75f);
  CHECK(comparator->distmapAggregate() > 0.0f);

  // The parts keep the device alive.
  comparator.reset();
  backend = guetzli::Backend{};
  CHECK(device.use_count() == 1);
}

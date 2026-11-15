// Device-resident constant tables of one GPU context (blur weights, DoMask LUTs, the sRGB LUT, convolution scales),
// created on first use and kept for the context's lifetime.
#ifndef GUETZLI_BACKEND_GPU_CONSTANT_CACHE_H_
#define GUETZLI_BACKEND_GPU_CONSTANT_CACHE_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <optional>
#include <utility>

namespace guetzli {

// Memory: a move-only owner of a device allocation with get(). Each lookup takes a `make` callable that creates the
// table if it is not cached yet.
template <class Memory>
class ConstantCache {
 public:
  using Handle = decltype(std::declval<const Memory&>().get());
  // Convolution scale tables are keyed by {weights buffer, sample count, step, border_ratio bits}.
  using ConvolutionKey = std::array<size_t, 4>;

  template <class Make>
  Handle blurWeights(float sigma, Make make) {
    uint32_t key;
    std::memcpy(&key, &sigma, sizeof(key));
    return lookup(blur_weights_, key, make);
  }

  // i = 0..5: x, y, b, dc_x, dc_y, dc_b.
  template <class Make>
  Handle maskLut(int i, Make make) {
    std::optional<Memory>& slot = mask_luts_.at(i);
    if (!slot) slot.emplace(make());
    return slot->get();
  }

  template <class Make>
  Handle srgbToLinear(Make make) {
    if (!srgb_to_linear_) srgb_to_linear_.emplace(make());
    return srgb_to_linear_->get();
  }

  template <class Make>
  Handle convolutionScale(const ConvolutionKey& key, Make make) {
    return lookup(convolution_scales_, key, make);
  }

  // Releases every table.
  void clear() noexcept {
    blur_weights_.clear();
    for (std::optional<Memory>& lut : mask_luts_) lut.reset();
    srgb_to_linear_.reset();
    convolution_scales_.clear();
  }

 private:
  template <class Map, class Key, class Make>
  static Handle lookup(Map& map, const Key& key, Make& make) {
    auto it = map.find(key);
    if (it == map.end()) it = map.emplace(key, make()).first;
    return it->second.get();
  }

  std::map<uint32_t, Memory> blur_weights_;  // keyed by sigma's bit pattern
  std::array<std::optional<Memory>, 6> mask_luts_;
  std::optional<Memory> srgb_to_linear_;
  std::map<ConvolutionKey, Memory> convolution_scales_;
};

}  // namespace guetzli

#endif  // GUETZLI_BACKEND_GPU_CONSTANT_CACHE_H_

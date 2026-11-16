// Checked narrowing conversion.
#ifndef GUETZLI_CORE_UTIL_NARROW_CAST_H_
#define GUETZLI_CORE_UTIL_NARROW_CAST_H_

#include <stdexcept>

namespace guetzli {

// Converts v to Target; throws std::runtime_error if the value does not survive the round trip.
template <class Target, class Source>
Target narrow_cast(Source v) {
  auto r = static_cast<Target>(v);
  if (static_cast<Source>(r) != v) {
    throw std::runtime_error("narrow_cast<>() failed");
  }
  return r;
}

}  // namespace guetzli

#endif  // GUETZLI_CORE_UTIL_NARROW_CAST_H_

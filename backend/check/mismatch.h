// Mismatch reporting shared by the --check decorators.
#ifndef GUETZLI_BACKEND_CHECK_MISMATCH_H_
#define GUETZLI_BACKEND_CHECK_MISMATCH_H_

#include <cmath>
#include <cstddef>

#include "logging/logger.h"

namespace guetzli {

// Counts elements that differ by more than tolerance and logs one "CHK <what> <part> <count>:<size>"
// error line if there are any. Returns the count.
template <typename T>
size_t reportMismatches(ILogger& logger, const char* what, const char* part, const T* reference, const T* tested,
                        size_t size, double tolerance = 0.0) {
  size_t count = 0;
  for (size_t i = 0; i < size; ++i) {
    if (std::fabs(static_cast<double>(reference[i]) - static_cast<double>(tested[i])) > tolerance) ++count;
  }
  if (count > 0) logger.log(LogLevel::Error, "CHK %s %s %zu:%zu\n", what, part, count, size);
  return count;
}

}  // namespace guetzli

#endif  // GUETZLI_BACKEND_CHECK_MISMATCH_H_

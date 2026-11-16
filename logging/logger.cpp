#include "logging/logger.h"

#include <cstdarg>
#include <vector>

namespace guetzli {

void ILogger::log(LogLevel level, const char* format, ...) {
  if (!enabled(level)) {
    return;
  }
  va_list args;  // NOLINT(cppcoreguidelines-init-variables): set by va_start; not a pointer on every ABI
  va_start(args, format);
  va_list copy;  // NOLINT(cppcoreguidelines-init-variables): set by va_copy
  va_copy(copy, args);
  const int n = std::vsnprintf(nullptr, 0, format, copy);
  va_end(copy);
  if (n > 0) {
    std::vector<char> buf(static_cast<size_t>(n) + 1);
    std::vsnprintf(buf.data(), buf.size(), format, args);
    write(level, std::string_view(buf.data(), static_cast<size_t>(n)));
  }
  va_end(args);
}

void ConsoleLogger::write(LogLevel level, std::string_view message) {
  if (!enabled(level)) {
    return;
  }
  std::FILE* stream = level == LogLevel::Info ? out_ : err_;
  std::fwrite(message.data(), 1, message.size(), stream);
}

}  // namespace guetzli

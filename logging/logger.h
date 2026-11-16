// Injected logging (docs/design/modernization.md §5.7).
#ifndef GUETZLI_LOGGING_LOGGER_H_
#define GUETZLI_LOGGING_LOGGER_H_

#include <cstdio>
#include <string>
#include <string_view>

namespace guetzli {

enum class LogLevel {
  Error,
  Warning,
  Info,   // normal progress messages
  Debug,  // --verbose trace
};

class ILogger {
 public:
  virtual ~ILogger() = default;

  [[nodiscard]] virtual bool enabled(LogLevel level) const = 0;
  // Writes message as-is (no newline is added).
  virtual void write(LogLevel level, std::string_view message) = 0;

  // printf-style convenience; formats only when the level is enabled.
#if defined(__GNUC__) || defined(__clang__)
  __attribute__((format(printf, 3, 4)))
#endif
  void log(LogLevel level, const char* format, ...);
};

// Info goes to stdout; errors, warnings and the debug trace go to stderr
// (the same streams the tool always used).
class ConsoleLogger final : public ILogger {
 public:
  explicit ConsoleLogger(LogLevel threshold, std::FILE* out = stdout, std::FILE* err = stderr)
      : threshold_(threshold), out_(out), err_(err) {}

  [[nodiscard]] bool enabled(LogLevel level) const override { return level <= threshold_; }
  void write(LogLevel level, std::string_view message) override;

 private:
  LogLevel threshold_;
  std::FILE* out_;
  std::FILE* err_;
};

class NullLogger final : public ILogger {
 public:
  [[nodiscard]] bool enabled(LogLevel /*level*/) const override { return false; }
  void write(LogLevel /*level*/, std::string_view /*message*/) override {}
};

}  // namespace guetzli

#endif  // GUETZLI_LOGGING_LOGGER_H_

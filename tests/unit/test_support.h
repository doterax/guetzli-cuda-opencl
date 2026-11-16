// Helpers shared by the unit tests.
#ifndef GUETZLI_TESTS_UNIT_TEST_SUPPORT_H_
#define GUETZLI_TESTS_UNIT_TEST_SUPPORT_H_

#include <atomic>
#include <filesystem>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "logging/logger.h"

#ifndef GUETZLI_SOURCE_DIR
#error "GUETZLI_SOURCE_DIR must point to the source tree (set by the build)"
#endif

namespace guetzli_test {

// A fresh path in the temp directory (not created): a random tag per run plus a counter.
inline std::string tempPath(std::string_view suffix) {
  static const unsigned int kRun = std::random_device{}();
  static std::atomic<int> counter{0};
  const std::string name =
      "guetzli_test_" + std::to_string(kRun) + "_" + std::to_string(counter++) + std::string(suffix);
  return (std::filesystem::temp_directory_path() / name).string();
}

// Absolute path of a file in the source tree, e.g. "tests/unit/data/rgb_16x16.png".
inline std::string sourceFile(const char* relative) {
  return std::string(GUETZLI_SOURCE_DIR) + "/" + relative;
}

class RecordingLogger final : public guetzli::ILogger {
 public:
  explicit RecordingLogger(guetzli::LogLevel threshold = guetzli::LogLevel::Info) : threshold_(threshold) {}
  [[nodiscard]] bool enabled(guetzli::LogLevel level) const override { return level <= threshold_; }
  void write(guetzli::LogLevel level, std::string_view message) override {
    lines.emplace_back(level, std::string(message));
  }

  // All messages of `level`, concatenated.
  [[nodiscard]] std::string text(guetzli::LogLevel level) const {
    std::string all;
    for (const auto& [l, m] : lines) {
      if (l == level) {
        all += m;
      }
    }
    return all;
  }

  std::vector<std::pair<guetzli::LogLevel, std::string>> lines;

 private:
  guetzli::LogLevel threshold_;
};

}  // namespace guetzli_test

#endif  // GUETZLI_TESTS_UNIT_TEST_SUPPORT_H_

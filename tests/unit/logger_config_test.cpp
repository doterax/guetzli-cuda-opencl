#include <cstdio>
#include <string>
#include <vector>

#include "config/config.h"
#include "doctest.h"
#include "logging/logger.h"
#include "test_support.h"

using guetzli_test::RecordingLogger;

TEST_CASE("ILogger::log formats and respects the level") {
  RecordingLogger logger(guetzli::LogLevel::Info);
  logger.log(guetzli::LogLevel::Info, "%s %d\n", "answer", 42);
  logger.log(guetzli::LogLevel::Debug, "hidden %d\n", 1);
  REQUIRE(logger.lines.size() == 1);
  CHECK(logger.lines[0].first == guetzli::LogLevel::Info);
  CHECK(logger.lines[0].second == "answer 42\n");

  // Longer than any fixed buffer.
  logger.log(guetzli::LogLevel::Error, "%s", std::string(5000, 'x').c_str());
  CHECK(logger.lines.back().second.size() == 5000);
}

TEST_CASE("ConsoleLogger: info to out, everything else to err, filtered by threshold") {
  // C stdio temp files, as ConsoleLogger takes FILE*.
  // NOLINTBEGIN(cppcoreguidelines-owning-memory)
  std::FILE* out = std::tmpfile();
  std::FILE* err = std::tmpfile();
  REQUIRE(out);
  REQUIRE(err);
  guetzli::ConsoleLogger logger(guetzli::LogLevel::Info, out, err);
  logger.write(guetzli::LogLevel::Info, "info;");
  logger.write(guetzli::LogLevel::Error, "error;");
  logger.write(guetzli::LogLevel::Warning, "warning;");
  logger.write(guetzli::LogLevel::Debug, "debug;");
  auto contents = [](std::FILE* f) {
    std::fseek(f, 0, SEEK_SET);
    std::string s;
    for (int c = 0; (c = std::fgetc(f)) != EOF;) {
      s.push_back(static_cast<char>(c));
    }
    return s;
  };
  CHECK(contents(out) == "info;");
  CHECK(contents(err) == "error;warning;");
  std::fclose(out);
  std::fclose(err);
  // NOLINTEND(cppcoreguidelines-owning-memory)
}

TEST_CASE("Config::validate") {
  guetzli::Config c;
  CHECK(c.validate().size() == 2);  // no input, no output
  c.io.input = "a.png";
  c.io.output = "b.jpg";
  CHECK(c.validate().empty());

  c.backend.check_against_cpu = true;
  CHECK(c.validate().size() == 1);  // --check needs a GPU backend
  c.backend.kind = guetzli::BackendKind::Cuda;
  CHECK(c.validate().empty());
}

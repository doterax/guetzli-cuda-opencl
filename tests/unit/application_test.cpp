// Application::run(): exit codes and messages (CPU backend only; CI has no GPU).
#include <cstdio>
#include <string>

#include "app/application.h"
#include "core/io/image_io.h"
#include "doctest.h"
#include "test_support.h"

using guetzli_test::RecordingLogger;
using guetzli_test::sourceFile;

namespace {

std::string tempPath(const char* suffix) {
  return std::string(std::tmpnam(nullptr)) + suffix;
}

guetzli::Config cpuConfig(const std::string& input, const std::string& output) {
  guetzli::Config c;
  c.backend.kind = guetzli::BackendKind::Cpu;
  c.io.input = input;
  c.io.output = output;
  return c;
}

int run(const guetzli::Config& config, RecordingLogger& logger) {
  return guetzli::Application(config, logger).run();
}

}  // namespace

TEST_CASE("encodes a PNG and writes a JPEG") {
  const std::string out = tempPath(".jpg");
  RecordingLogger logger;
  CHECK(run(cpuConfig(sourceFile("tests/unit/data/rgb_16x16.png"), out), logger) == guetzli::kExitSuccess);
  const std::string jpeg = guetzli::readFile(out);
  CHECK(guetzli::isJpeg(jpeg));
  CHECK(logger.text(guetzli::LogLevel::Error).empty());
  CHECK(logger.text(guetzli::LogLevel::Info).find("succeeded") != std::string::npos);
  std::remove(out.c_str());
}

TEST_CASE("input that is not PNG, TIFF or JPEG: exit 2 and one message") {
  const std::string in = tempPath(".bin");
  guetzli::writeFile(in, "definitely not an image");
  RecordingLogger logger;
  CHECK(run(cpuConfig(in, tempPath(".jpg")), logger) == guetzli::kExitUnsupportedInput);
  CHECK(logger.text(guetzli::LogLevel::Error).find("Unknown file format") != std::string::npos);
  CHECK(logger.lines.size() == 1);
  std::remove(in.c_str());
}

TEST_CASE("corrupt PNG or JPEG: exit 3, reported as such and not as an unknown format") {
  const std::string png = guetzli::readFile(sourceFile("tests/unit/data/rgb_16x16.png"));
  const struct {
    const char* name;
    std::string data;
    const char* message;
  } cases[] = {
      {.name = "png", .data = png.substr(0, png.size() / 2), .message = "Error reading PNG data"},
      {.name = "jpeg", .data = std::string("\xFF\xD8\xFF\xE0 garbage", 12), .message = "Error reading JPEG data"},
  };
  for (const auto& c : cases) {
    CAPTURE(c.name);
    const std::string in = tempPath(".in");
    guetzli::writeFile(in, c.data);
    RecordingLogger logger;
    CHECK(run(cpuConfig(in, tempPath(".jpg")), logger) == guetzli::kExitProcessingFailed);
    const std::string errors = logger.text(guetzli::LogLevel::Error);
    CHECK(errors.find(c.message) != std::string::npos);
    CHECK(errors.find("Unknown file format") == std::string::npos);
    std::remove(in.c_str());
  }
}

TEST_CASE("memory limit: exit 3 with a hint") {
  guetzli::Config config = cpuConfig(sourceFile("tests/unit/data/rgb_16x16.png"), tempPath(".jpg"));
  config.io.memlimit_mb = 50;  // below the 100 MB floor
  RecordingLogger logger;
  CHECK(run(config, logger) == guetzli::kExitProcessingFailed);
  CHECK(logger.text(guetzli::LogLevel::Error).find("--nomemlimit") != std::string::npos);
}

TEST_CASE("missing input or unwritable output: exit 1") {
  RecordingLogger missing;
  CHECK(run(cpuConfig(tempPath(".does_not_exist"), tempPath(".jpg")), missing) == guetzli::kExitUsageOrIoError);
  CHECK(missing.text(guetzli::LogLevel::Error).find("Can't open input file") != std::string::npos);

  RecordingLogger unwritable;
  CHECK(run(cpuConfig(sourceFile("tests/unit/data/rgb_16x16.png"), sourceFile("no/such/dir/out.jpg")), unwritable) ==
        guetzli::kExitUsageOrIoError);
  CHECK(unwritable.text(guetzli::LogLevel::Error).find("Can't open output file") != std::string::npos);
}

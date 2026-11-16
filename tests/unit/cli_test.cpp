// Command line -> Config, including the legacy flags (app/cli.cpp).
#include <sstream>
#include <string>
#include <vector>

#include "app/cli.h"
#include "doctest.h"

using guetzli::BackendKind;
using guetzli::BuildFeatures;
using guetzli::CliResult;

namespace {

constexpr BuildFeatures kAll{.opencl = true, .cuda = true};
constexpr BuildFeatures kCpuOnly{.opencl = false, .cuda = false};

struct Parsed {
  CliResult result;
  std::string out;
  std::string err;
};

Parsed parse(std::vector<std::string> args, BuildFeatures features = kAll) {
  args.insert(args.begin(), "guetzli");
  std::vector<const char*> argv;
  argv.reserve(args.size());
  for (const std::string& a : args) argv.push_back(a.c_str());
  std::ostringstream out;
  std::ostringstream err;
  Parsed p{.result = guetzli::parseCommandLine(static_cast<int>(argv.size()), argv.data(), out, err, features),
           .out = "",
           .err = ""};
  p.out = out.str();
  p.err = err.str();
  return p;
}

guetzli::Config runConfig(const Parsed& p) {
  REQUIRE_MESSAGE(p.result.action == CliResult::Action::Run, p.err);
  return p.result.config;
}

}  // namespace

TEST_CASE("defaults") {
  const Parsed p = parse({"in.png", "out.jpg"});
  const guetzli::Config& c = runConfig(p);
  CHECK(c.io.input == "in.png");
  CHECK(c.io.output == "out.jpg");
  CHECK(c.encoder.quality == 95);
  CHECK_FALSE(c.encoder.blend_on_white);
  CHECK(c.backend.kind == BackendKind::Auto);
  CHECK_FALSE(c.backend.check_against_cpu);
  CHECK_FALSE(c.backend.device.has_value());
  REQUIRE(c.io.memlimit_mb.has_value());
  CHECK(*c.io.memlimit_mb == 6000);
  CHECK(c.log.level == guetzli::LogLevel::Info);
}

TEST_CASE("encoder, io and log options") {
  const guetzli::Config& c =
      runConfig(parse({"--quality", "84", "--blend-on-white", "--verbose", "--memlimit", "1234", "a", "b"}));
  CHECK(c.encoder.quality == 84);
  CHECK(c.encoder.blend_on_white);
  CHECK(c.log.level == guetzli::LogLevel::Debug);
  CHECK(*c.io.memlimit_mb == 1234);
}

TEST_CASE("memory limit: --nomemlimit, -1 and the last one wins") {
  CHECK_FALSE(runConfig(parse({"--nomemlimit", "a", "b"})).io.memlimit_mb.has_value());
  CHECK_FALSE(runConfig(parse({"--memlimit", "-1", "a", "b"})).io.memlimit_mb.has_value());
  CHECK(*runConfig(parse({"--nomemlimit", "--memlimit", "500", "a", "b"})).io.memlimit_mb == 500);
  CHECK_FALSE(runConfig(parse({"--memlimit", "500", "--nomemlimit", "a", "b"})).io.memlimit_mb.has_value());

  const Parsed negative = parse({"--memlimit", "-5", "a", "b"});
  CHECK(negative.result.action == CliResult::Action::Exit);
  CHECK(negative.result.exit_code == 1);
}

TEST_CASE("canonical backend selection") {
  CHECK(runConfig(parse({"--backend", "cpu", "a", "b"})).backend.kind == BackendKind::Cpu);
  CHECK(runConfig(parse({"--backend", "opencl", "a", "b"})).backend.kind == BackendKind::OpenCl);
  CHECK(runConfig(parse({"--backend", "CUDA", "a", "b"})).backend.kind == BackendKind::Cuda);
  CHECK(runConfig(parse({"--backend", "auto", "a", "b"})).backend.kind == BackendKind::Auto);

  const guetzli::Config& check = runConfig(parse({"--check", "--backend", "opencl", "a", "b"}));
  CHECK(check.backend.kind == BackendKind::OpenCl);
  CHECK(check.backend.check_against_cpu);
}

TEST_CASE("legacy backend aliases") {
  struct Case {
    const char* flag;
    BackendKind kind;
    bool check;
  };
  const Case cases[] = {
      {.flag = "--c", .kind = BackendKind::Cpu, .check = false},
      {.flag = "--auto", .kind = BackendKind::Auto, .check = false},
      {.flag = "--opencl", .kind = BackendKind::OpenCl, .check = false},
      {.flag = "--checkcl", .kind = BackendKind::OpenCl, .check = true},
      {.flag = "--cuda", .kind = BackendKind::Cuda, .check = false},
      {.flag = "--checkcuda", .kind = BackendKind::Cuda, .check = true},
  };
  for (const Case& c : cases) {
    CAPTURE(c.flag);
    const guetzli::Config& config = runConfig(parse({c.flag, "a", "b"}));
    CHECK(config.backend.kind == c.kind);
    CHECK(config.backend.check_against_cpu == c.check);
  }
}

TEST_CASE("mode flags: the last one wins") {
  CHECK(runConfig(parse({"--opencl", "--c", "a", "b"})).backend.kind == BackendKind::Cpu);
  CHECK(runConfig(parse({"--c", "--cuda", "a", "b"})).backend.kind == BackendKind::Cuda);

  const guetzli::Config& c = runConfig(parse({"--checkcl", "--opencl", "a", "b"}));
  CHECK(c.backend.kind == BackendKind::OpenCl);
  CHECK_FALSE(c.backend.check_against_cpu);

  CHECK(runConfig(parse({"--cuda", "--backend", "cpu", "a", "b"})).backend.kind == BackendKind::Cpu);
  CHECK(runConfig(parse({"--backend", "cpu", "--opencl", "a", "b"})).backend.kind == BackendKind::OpenCl);
}

TEST_CASE("--device") {
  REQUIRE(runConfig(parse({"--device", "1", "a", "b"})).backend.device.has_value());
  CHECK(*runConfig(parse({"--device", "1", "a", "b"})).backend.device == 1);
  // A negative index always meant "default selection".
  CHECK_FALSE(runConfig(parse({"--device", "-1", "a", "b"})).backend.device.has_value());
}

TEST_CASE("-- ends the options, - is a valid file name") {
  const guetzli::Config& c = runConfig(parse({"--c", "--", "-", "--out.jpg"}));
  CHECK(c.io.input == "-");
  CHECK(c.io.output == "--out.jpg");
}

TEST_CASE("version") {
  CHECK(parse({"--version"}).result.action == CliResult::Action::ShowVersion);
  CHECK(parse({"--verbose", "--version"}).result.action == CliResult::Action::ShowVersion);
}

TEST_CASE("help goes to stdout with exit code 0") {
  const Parsed p = parse({"--help"});
  CHECK(p.result.action == CliResult::Action::Exit);
  CHECK(p.result.exit_code == 0);
  CHECK(p.out.find("--quality") != std::string::npos);
  CHECK(p.out.find("--checkcl") != std::string::npos);
  CHECK(p.err.empty());
}

TEST_CASE("usage errors: message and usage on stderr, exit code 1") {
  for (const std::vector<std::string>& args : std::vector<std::vector<std::string>>{
           {},                               // no files
           {"only_input.png"},               // no output
           {"--bogus", "a", "b"},            // unknown flag
           {"--quality", "high", "a", "b"},  // not a number
           {"--backend", "gpu", "a", "b"},   // unknown backend
           {"a", "b", "c"},                  // extra positional
           {"--check", "a", "b"},            // --check without a GPU backend
           {"--backend", "cpu", "--check", "a", "b"},
       }) {
    const Parsed p = parse(args);
    CAPTURE(p.err);
    CHECK(p.result.action == CliResult::Action::Exit);
    CHECK(p.result.exit_code == 1);
    CHECK(p.err.find("Error:") != std::string::npos);
    CHECK(p.err.find("--quality") != std::string::npos);  // usage follows the error
  }
}

TEST_CASE("GPU flags exist only when the backend is compiled in") {
  CHECK(parse({"--opencl", "a", "b"}, kCpuOnly).result.exit_code == 1);
  CHECK(parse({"--cuda", "a", "b"}, kCpuOnly).result.exit_code == 1);
  CHECK(parse({"--checkcl", "a", "b"}, kCpuOnly).result.exit_code == 1);
  CHECK(parse({"--backend", "opencl", "a", "b"}, kCpuOnly).result.exit_code == 1);
  CHECK(runConfig(parse({"--backend", "cpu", "a", "b"}, kCpuOnly)).backend.kind == BackendKind::Cpu);

  const BuildFeatures opencl_only{.opencl = true, .cuda = false};
  CHECK(runConfig(parse({"--opencl", "a", "b"}, opencl_only)).backend.kind == BackendKind::OpenCl);
  CHECK(parse({"--cuda", "a", "b"}, opencl_only).result.exit_code == 1);
}

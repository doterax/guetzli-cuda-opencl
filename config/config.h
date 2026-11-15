// Validated run configuration, filled by app/cli.cpp and passed down
// (docs/design/modernization.md §5.2). Value types only.
#ifndef GUETZLI_CONFIG_CONFIG_H_
#define GUETZLI_CONFIG_CONFIG_H_

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "logging/logger.h"

namespace guetzli {

enum class BackendKind {
  Auto,  // probe CUDA, then OpenCL, then CPU
  Cpu,   // optimized CPU implementation
  OpenCl,
  Cuda,
};

struct BackendOptions {
  BackendKind kind = BackendKind::Auto;
  std::optional<int> device;       // index within the chosen GPU backend
  bool check_against_cpu = false;  // run GPU and CPU, report mismatches (--check)
};

inline constexpr int kDefaultQuality = 95;
inline constexpr std::size_t kDefaultMemlimitMb = 6000;

struct EncoderOptions {
  int quality = kDefaultQuality;  // JPEG-like quality, mapped to a butteraugli target
  bool blend_on_white = false;    // alpha is blended on black unless set
};

struct IoOptions {
  std::filesystem::path input;                                  // "-" = stdin
  std::filesystem::path output;                                 // "-" = stdout
  std::optional<std::size_t> memlimit_mb = kDefaultMemlimitMb;  // nullopt = no limit
};

struct LogOptions {
  LogLevel level = LogLevel::Info;
};

struct Config {
  EncoderOptions encoder;
  BackendOptions backend;
  IoOptions io;
  LogOptions log;

  // Human-readable problems; empty means the configuration is usable.
  [[nodiscard]] std::vector<std::string> validate() const;
};

}  // namespace guetzli

#endif  // GUETZLI_CONFIG_CONFIG_H_

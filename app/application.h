// Runs one encode for a validated Config: backend setup, input decoding, encoding,
// CPU fallback on GPU failure, output. The composition root of the tool.
#ifndef GUETZLI_APP_APPLICATION_H_
#define GUETZLI_APP_APPLICATION_H_

#include <string>

#include "app/exit_codes.h"
#include "config/config.h"
#include "logging/logger.h"

namespace guetzli {

class Application {
 public:
  Application(const Config& config, ILogger& logger) : config_(config), logger_(logger) {}

  // Returns one of the kExit* codes.
  [[nodiscard]] int run();

 private:
  enum class Result { NotSupported, Failed, Success };

  void selectBackend();
  [[nodiscard]] Result encode(const std::string& in_data, std::string* out_data);
  [[nodiscard]] bool withinMemoryLimit(double pixels) const;
  void logMemoryLimitExceeded();

  const Config& config_;
  ILogger& logger_;
};

}  // namespace guetzli

#endif  // GUETZLI_APP_APPLICATION_H_

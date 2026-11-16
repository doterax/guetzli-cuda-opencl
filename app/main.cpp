// guetzli command-line tool: CLI11 -> Config -> Application::run().
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <new>

#include "app/application.h"
#include "app/cli.h"
#include "backend/factory.h"
#include "app/version.h"
#include "logging/logger.h"

namespace {

[[noreturn]] void exitOutOfMemory() {
  std::fprintf(stderr,
               "Unhandled exception. Most likely insufficient memory available.\n"
               "Make sure that there is 300MB/MPix of memory available.\n");
  std::exit(1);
}

// Reports the exception being handled (or none) and exits with code 1.
[[noreturn]] void reportUnhandledException() {
  try {
    if (std::exception_ptr e = std::current_exception()) {
      std::rethrow_exception(e);
    }
  } catch (const std::bad_alloc&) {
    exitOutOfMemory();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "Unhandled exception: %s\n", e.what());
    std::exit(1);
  } catch (...) {
    exitOutOfMemory();
  }
  exitOutOfMemory();
}

int showVersion() {
  std::fprintf(stdout, "Guetzli JPEG compressor (%s) - CUDA/OpenCL Optimized\n\n", guetzli::kVersion);
  guetzli::printGpuDevices();
  return 0;
}

int run(int argc, char** argv) {
  const guetzli::CliResult cli = guetzli::parseCommandLine(argc, argv, std::cout, std::cerr);
  std::cout.flush();
  std::cerr.flush();

  int exit_code = cli.exit_code;
  switch (cli.action) {
    case guetzli::CliResult::Action::ShowVersion:
      exit_code = showVersion();
      break;
    case guetzli::CliResult::Action::Run: {
      guetzli::ConsoleLogger logger(cli.config.log.level);
      exit_code = guetzli::Application(cli.config, logger).run();
      break;
    }
    case guetzli::CliResult::Action::Exit:
      break;
  }
  return exit_code;
}

}  // namespace

int main(int argc, char** argv) {
  std::set_terminate(reportUnhandledException);
  try {
    return run(argc, argv);
  } catch (...) {
    reportUnhandledException();
  }
}

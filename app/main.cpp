// guetzli command-line tool: CLI11 -> Config -> Application::run().
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <new>

#include "app/application.h"
#include "app/cli.h"
#include "backend/opencl/opencl_kernels.h"
#include "guetzli/version.h"
#include "logging/logger.h"
#ifdef __USE_GPERFTOOLS__
#include <google/profiler.h>
#endif

namespace {

void terminateHandler() {
  try {
    std::exception_ptr e = std::current_exception();
    if (e) std::rethrow_exception(e);
  } catch (const std::bad_alloc&) {
    // fall through to the out-of-memory message below
  } catch (const std::exception& e) {
    std::fprintf(stderr, "Unhandled exception: %s\n", e.what());
    std::exit(1);
  } catch (...) {
  }
  std::fprintf(stderr,
               "Unhandled exception. Most likely insufficient memory available.\n"
               "Make sure that there is 300MB/MPix of memory available.\n");
  std::exit(1);
}

int showVersion() {
  std::fprintf(stdout, "Guetzli JPEG compressor (%s) - CUDA/OpenCL Optimized\n\n", guetzli::kVersion);
#ifdef __USE_CUDA__
  enumCudaDevices();
#endif
#ifdef __USE_OPENCL__
  enumOpenClDevices();
#endif
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
#ifdef __USE_GPERFTOOLS__
  ProfilerStart("guetzli.prof");
#endif
  std::set_terminate(terminateHandler);

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

#ifdef __USE_GPERFTOOLS__
  ProfilerStop();
#endif
  return exit_code;
}

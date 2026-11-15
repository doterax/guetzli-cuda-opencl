// Command line -> Config (docs/design/modernization.md §5.2).
#ifndef GUETZLI_APP_CLI_H_
#define GUETZLI_APP_CLI_H_

#include <ostream>

#include "config/config.h"

namespace guetzli {

// Backends compiled into this binary; decides which flags exist.
struct BuildFeatures {
  bool opencl = false;
  bool cuda = false;
};

[[nodiscard]] BuildFeatures compiledFeatures();

struct CliResult {
  enum class Action {
    Run,          // encode with `config`
    ShowVersion,  // print version and enumerate devices
    Exit,         // help or error already printed; exit with `exit_code`
  };
  Action action = Action::Exit;
  int exit_code = 0;
  Config config;
};

// Parses argv. Help goes to `out`; errors and usage go to `err` (exit code 1, as before).
// Legacy flags (--c, --opencl, --cuda, --auto, --checkcl, --checkcuda, --nomemlimit) are kept;
// mode flags and --backend/--check are applied in command-line order, the last one wins.
[[nodiscard]] CliResult parseCommandLine(int argc, const char* const* argv, std::ostream& out, std::ostream& err,
                                         const BuildFeatures& features = compiledFeatures());

}  // namespace guetzli

#endif  // GUETZLI_APP_CLI_H_

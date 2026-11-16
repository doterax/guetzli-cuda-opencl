#include "app/cli.h"

#include <map>
#include <string>

#include "CLI11.hpp"
#include "app/exit_codes.h"
#include "app/version.h"

namespace guetzli {

BuildFeatures compiledFeatures() {
  BuildFeatures f;
#ifdef __USE_OPENCL__
  f.opencl = true;
#endif
#ifdef __USE_CUDA__
  f.cuda = true;
#endif
  return f;
}

namespace {

constexpr const char* kFooter =
    "Exit codes: 0 success, 1 usage or I/O error, 2 input is not PNG/TIFF/JPEG,\n"
    "3 input could not be processed (corrupt data, memory limit, encoder failure).\n"
    "\n"
    "Support this project: if you find this tool useful, please consider\n"
    "buying me a coffee: https://www.buymeacoffee.com/doterax\n"
    "Your support helps maintain and improve this project!";

std::string description() {
  return std::string("Guetzli JPEG compressor (") + kVersion +
         ") - CUDA/OpenCL Optimized\n"
         "Repository: https://github.com/doterax/guetzli-cuda-opencl\n"
         "Maintainer: https://github.com/doterax\n";
}

}  // namespace

CliResult parseCommandLine(int argc, const char* const* argv, std::ostream& out, std::ostream& err,
                           const BuildFeatures& features) {
  CliResult result;
  Config& config = result.config;

  CLI::App app(description(), "guetzli");
  app.footer(kFooter);
  app.get_formatter()->column_width(28);

  std::string input;
  std::string output;
  // Not required() so that --version works with other flags; Config::validate() reports missing files.
  app.add_option("input", input, "Input image (PNG, JPEG, TIFF; - = stdin)");
  app.add_option("output", output, "Output JPEG (- = stdout)");

  bool version = false;
  app.add_flag("--version", version, "Print the version and list CUDA/OpenCL devices");
  bool verbose = false;
  app.add_flag("--verbose", verbose, "Print a verbose trace of all attempts to standard error");
  app.add_option("--quality", config.encoder.quality, "Visual quality to aim for, as a JPEG quality value")
      ->default_val(kDefaultQuality);
  int memlimit = static_cast<int>(kDefaultMemlimitMb);
  CLI::Option* memlimit_opt =
      app.add_option("--memlimit", memlimit, "Memory limit in MB; guetzli fails if it cannot stay under it")
          ->default_val(kDefaultMemlimitMb);
  CLI::Option* nomemlimit_opt = app.add_flag("--nomemlimit", "Do not limit memory usage");
  app.add_flag("--blend-on-white", config.encoder.blend_on_white,
               "Blend pixels with transparency on a white background (default: black)");

  // Backend selection. Canonical: --backend/--check; the legacy flags remain as aliases.
  std::map<std::string, BackendKind> backends = {{"auto", BackendKind::Auto}, {"cpu", BackendKind::Cpu}};
  if (features.opencl) {
    backends["opencl"] = BackendKind::OpenCl;
  }
  if (features.cuda) {
    backends["cuda"] = BackendKind::Cuda;
  }
  std::vector<std::string> backend_names;
  backend_names.reserve(backends.size());
  for (const auto& [name, kind] : backends) {
    backend_names.push_back(name);
  }
  std::string backend_value;
  CLI::Option* backend_opt =
      app.add_option("--backend", backend_value, "Backend; auto tries CUDA, then OpenCL, then the CPU (default: auto)")
          ->transform(CLI::IsMember(backend_names, CLI::ignore_case))  // also normalizes the case
          ->type_name("NAME");
  CLI::Option* check_opt = app.add_flag("--check", "Run the GPU backend and the CPU one, report mismatches (slow)");
  int device = -1;
  CLI::Option* device_opt =
      app.add_option("--device", device, "Device index N within the selected backend (see --version)");

  std::map<CLI::Option*, std::pair<BackendKind, bool>> mode_flags;  // flag -> {kind, check}
  mode_flags[app.add_flag("--c", "Same as --backend cpu")] = {BackendKind::Cpu, false};
  mode_flags[app.add_flag("--auto", "Same as --backend auto")] = {BackendKind::Auto, false};
  if (features.opencl) {
    mode_flags[app.add_flag("--opencl", "Same as --backend opencl")] = {BackendKind::OpenCl, false};
    mode_flags[app.add_flag("--checkcl", "Same as --backend opencl --check")] = {BackendKind::OpenCl, true};
  }
  if (features.cuda) {
    mode_flags[app.add_flag("--cuda", "Same as --backend cuda")] = {BackendKind::Cuda, false};
    mode_flags[app.add_flag("--checkcuda", "Same as --backend cuda --check")] = {BackendKind::Cuda, true};
  }
  for (auto& [opt, mode] : mode_flags) {
    opt->group("Legacy aliases");
  }

  try {
    app.parse(argc, argv);
  } catch (const CLI::CallForHelp&) {
    out << app.help();
    result.exit_code = kExitSuccess;
    return result;
  } catch (const CLI::ParseError& e) {
    err << "Error: " << e.what() << "\n\n" << app.help();
    result.exit_code = kExitUsageOrIoError;
    return result;
  }

  if (version) {
    result.action = CliResult::Action::ShowVersion;
    return result;
  }

  // Order-dependent options: the last occurrence wins, as with the old hand-written parser.
  // Legacy mode flags set backend and check mode (--checkcl then --opencl = no check);
  // --backend sets only the backend and --check only the check mode, in any order.
  for (CLI::Option* opt : app.parse_order()) {
    if (auto it = mode_flags.find(opt); it != mode_flags.end()) {
      config.backend.kind = it->second.first;
      config.backend.check_against_cpu = it->second.second;
    } else if (opt == backend_opt) {
      config.backend.kind = backends.at(backend_value);
    } else if (opt == check_opt) {
      config.backend.check_against_cpu = true;
    } else if (opt == memlimit_opt) {
      if (memlimit == -1) {
        config.io.memlimit_mb.reset();  // -1 always meant "no limit"
      } else if (memlimit < 0) {
        err << "Error: --memlimit must not be negative (use --nomemlimit)\n";
        result.exit_code = kExitUsageOrIoError;
        return result;
      } else {
        config.io.memlimit_mb = static_cast<std::size_t>(memlimit);
      }
    } else if (opt == nomemlimit_opt) {
      config.io.memlimit_mb.reset();
    }
  }
  // A negative device index always meant "default device selection".
  if (device_opt->count() > 0 && device >= 0) {
    config.backend.device = device;
  }
  config.io.input = input;
  config.io.output = output;
  config.log.level = verbose ? LogLevel::Debug : LogLevel::Info;

  const std::vector<std::string> problems = config.validate();
  if (!problems.empty()) {
    for (const std::string& p : problems) {
      err << "Error: " << p << "\n";
    }
    err << "\n" << app.help();
    result.exit_code = kExitUsageOrIoError;
    return result;
  }
  result.action = CliResult::Action::Run;
  return result;
}

}  // namespace guetzli

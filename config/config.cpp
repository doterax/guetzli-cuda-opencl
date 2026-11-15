#include "config/config.h"

namespace guetzli {

std::vector<std::string> Config::validate() const {
  std::vector<std::string> errors;
  if (io.input.empty()) errors.emplace_back("no input file given");
  if (io.output.empty()) errors.emplace_back("no output file given");
  if (backend.check_against_cpu && backend.kind != BackendKind::OpenCl && backend.kind != BackendKind::Cuda) {
    errors.emplace_back("--check needs a GPU backend (--backend opencl or --backend cuda)");
  }
  return errors;
}

}  // namespace guetzli

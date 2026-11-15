// The backend the encoder runs on (docs/design/modernization.md §5.3): one implementation per
// capability. Built by backend/factory.h; the encoder only sees these interfaces, so it contains
// no backend checks. The interfaces live here, next to their consumer, so that the core never
// includes anything from backend/.
#ifndef GUETZLI_BACKEND_H_
#define GUETZLI_BACKEND_H_

#include <memory>

#include "config/config.h"
#include "guetzli/block_zeroing.h"
#include "guetzli/comparator.h"
#include "guetzli/dct_ops.h"

namespace guetzli {

struct Backend {
  BackendKind kind = BackendKind::Cpu;  // never Auto
  std::unique_ptr<IComparatorFactory> comparators;
  std::unique_ptr<IBlockZeroingSearch> block_zeroing;
  std::unique_ptr<IDctOps> dct;
};

}  // namespace guetzli

#endif  // GUETZLI_BACKEND_H_

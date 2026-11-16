// --check for IDctOps (docs/design/modernization.md §5.4): runs the tested (GPU) and the reference (CPU)
// implementation, reports differences as "CHK" errors and keeps the reference result.
#ifndef GUETZLI_BACKEND_CHECK_CHECKING_DCT_OPS_H_
#define GUETZLI_BACKEND_CHECK_CHECKING_DCT_OPS_H_

#include <memory>

#include "core/image/dct_ops.h"
#include "logging/logger.h"

namespace guetzli {

class CheckingDctOps final : public IDctOps {
 public:
  CheckingDctOps(std::unique_ptr<IDctOps> tested, std::unique_ptr<IDctOps> reference, ILogger& logger)
      : tested_(std::move(tested)), reference_(std::move(reference)), logger_(logger) {}

  void copyFromJpegComponent(OutputImageComponent& c, const JpegComponent& comp, int factor_x, int factor_y,
                             const int* quant) override;
  void applyGlobalQuantization(OutputImageComponent& c, const int q[kDctBlockSize]) override;
  // Windows of at most one 8x8 block (per-block searches convert one block at a time) use only the
  // reference: checking each of them would cost a device round trip per block.
  void toSrgb(const OutputImage& img, int xmin, int ymin, int xsize, int ysize, std::vector<uint8_t>& rgb) override;

 private:
  void compare(const char* what, const OutputImageComponent& reference, const OutputImageComponent& tested);

  std::unique_ptr<IDctOps> tested_;
  std::unique_ptr<IDctOps> reference_;
  ILogger& logger_;
};

}  // namespace guetzli

#endif  // GUETZLI_BACKEND_CHECK_CHECKING_DCT_OPS_H_

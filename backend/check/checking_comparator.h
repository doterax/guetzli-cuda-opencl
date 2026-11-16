// --check for comparators (docs/design/modernization.md §5.4): Compare() and StartBlockComparisons() run
// on the tested (GPU) and the reference (CPU) comparator; the distance maps are compared and the reference
// result is used. Per-block comparisons run on the reference only (no backend implements them on a device).
#ifndef GUETZLI_BACKEND_CHECK_CHECKING_COMPARATOR_H_
#define GUETZLI_BACKEND_CHECK_CHECKING_COMPARATOR_H_

#include <memory>
#include <optional>
#include <vector>

#include "core/quality/comparator.h"
#include "logging/logger.h"

namespace guetzli {

class CheckingComparator final : public Comparator {
 public:
  CheckingComparator(std::unique_ptr<Comparator> tested, std::unique_ptr<Comparator> reference, ILogger& logger)
      : tested_(std::move(tested)), reference_(std::move(reference)), logger_(logger) {}

  void compare(const OutputImage& img) override;
  void startBlockComparisons() override;
  void finishBlockComparisons() override;
  void switchBlock(int block_x, int block_y, int factor_x, int factor_y) override {
    reference_->switchBlock(block_x, block_y, factor_x, factor_y);
  }
  double compareBlock(const OutputImage& img, int off_x, int off_y, const coeff_t* candidate_block,
                      const int comp_mask) const override {
    return reference_->compareBlock(img, off_x, off_y, candidate_block, comp_mask);
  }
  [[nodiscard]] double scoreOutputSize(int size) const override { return reference_->scoreOutputSize(size); }
  [[nodiscard]] bool distanceOk(double target_mul) const override { return reference_->distanceOk(target_mul); }
  [[nodiscard]] std::vector<float> distmap() const override { return reference_->distmap(); }
  [[nodiscard]] float distmapAggregate() const override { return reference_->distmapAggregate(); }
  [[nodiscard]] float blockErrorLimit() const override { return reference_->blockErrorLimit(); }
  void computeBlockErrorAdjustmentWeights(int direction, int max_block_dist, double target_mul, int factor_x,
                                          int factor_y, const std::vector<float>& distmap,
                                          std::vector<float>* block_weight) override {
    reference_->computeBlockErrorAdjustmentWeights(direction, max_block_dist, target_mul, factor_x, factor_y, distmap,
                                                   block_weight);
  }
  // The tested backend's data: a checked block-zeroing search runs the tested search with it.
  [[nodiscard]] std::optional<BlockReferenceData> blockReferenceData() const override {
    return tested_->blockReferenceData();
  }

 private:
  std::unique_ptr<Comparator> tested_;
  std::unique_ptr<Comparator> reference_;
  ILogger& logger_;
};

class CheckingComparatorFactory final : public IComparatorFactory {
 public:
  CheckingComparatorFactory(std::unique_ptr<IComparatorFactory> tested, std::unique_ptr<IComparatorFactory> reference,
                            ILogger& logger)
      : tested_(std::move(tested)), reference_(std::move(reference)), logger_(logger) {}

  std::unique_ptr<Comparator> create(int width, int height, const std::vector<uint8_t>* rgb,
                                     float target_distance) override;

 private:
  std::unique_ptr<IComparatorFactory> tested_;
  std::unique_ptr<IComparatorFactory> reference_;
  ILogger& logger_;
};

}  // namespace guetzli

#endif  // GUETZLI_BACKEND_CHECK_CHECKING_COMPARATOR_H_

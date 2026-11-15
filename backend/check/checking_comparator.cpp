#include "backend/check/checking_comparator.h"

#include "backend/check/mismatch.h"

namespace guetzli {

namespace {
// GPU kernels and the optimized CPU code differ in the last bits.
constexpr double kDistmapTolerance = 1e-3;
}  // namespace

void CheckingComparator::Compare(const OutputImage& img) {
  tested_->Compare(img);
  reference_->Compare(img);
  const std::vector<float> reference = reference_->distmap();
  const std::vector<float> tested = tested_->distmap();
  if (tested.size() != reference.size()) {
    logger_.log(LogLevel::Error, "CHK Compare distmap size %zu:%zu\n", tested.size(), reference.size());
    return;
  }
  reportMismatches(logger_, "Compare", "distmap", reference.data(), tested.data(), reference.size(), kDistmapTolerance);
  const float distance[2] = {reference_->distmap_aggregate(), tested_->distmap_aggregate()};
  reportMismatches(logger_, "Compare", "distance", &distance[0], &distance[1], 1, kDistmapTolerance);
}

void CheckingComparator::StartBlockComparisons() {
  tested_->StartBlockComparisons();
  reference_->StartBlockComparisons();
}

void CheckingComparator::FinishBlockComparisons() {
  tested_->FinishBlockComparisons();
  reference_->FinishBlockComparisons();
}

std::unique_ptr<Comparator> CheckingComparatorFactory::create(int width, int height, const std::vector<uint8_t>* rgb,
                                                              float target_distance, ProcessStats* stats) {
  return std::make_unique<CheckingComparator>(tested_->create(width, height, rgb, target_distance, stats),
                                              reference_->create(width, height, rgb, target_distance, stats), logger_);
}

}  // namespace guetzli

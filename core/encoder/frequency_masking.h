// Frequency masking (the second stage of the encoder): zeroes coefficients block by block where the
// distance map allows it, and restores them where it does not.
#ifndef GUETZLI_CORE_ENCODER_FREQUENCY_MASKING_H_
#define GUETZLI_CORE_ENCODER_FREQUENCY_MASKING_H_

#include <cstdint>
#include <span>
#include <vector>

#include "core/encoder/encoder_params.h"
#include "core/encoder/output_selector.h"
#include "core/encoder/block_zeroing.h"
#include "core/quality/comparator.h"
#include "core/jpeg/jpeg_data.h"
#include "core/image/output_image.h"
#include "core/encoder/stats.h"

namespace guetzli {

// The coefficients a block-zeroing search found worth zeroing, per block in zeroing order.
struct ZeroingCandidates {
  std::vector<int> offsets;     // num_blocks + 1 entries: block i owns [offsets[i], offsets[i + 1])
  std::vector<uint8_t> coeffs;  // coefficient index within the block (c * kDCTBlockSize + k)
  std::vector<float> errors;    // block error after zeroing this and all previous candidates
};

// Keeps the entries of orders (kBlockZeroingBlockSize per block) with 0 < block_err <= block_error_limit.
[[nodiscard]] ZeroingCandidates CollectZeroingCandidates(std::span<const CoeffData> orders, int num_blocks,
                                                         float block_error_limit);

class FrequencyMasking {
 public:
  // All arguments must outlive this object.
  FrequencyMasking(const EncoderParams& params, IBlockZeroingSearch& search, Comparator& comparator,
                   OutputSelector& selector, ProcessStats& stats)
      : params_(&params), search_(&search), comparator_(&comparator), selector_(&selector), stats_(&stats) {}

  // Adjusts the components in comp_mask of img (all with the subsampling of the last one) towards the
  // distance target times target_mul. jpg holds the original coefficients (quantization removed).
  // Every step's result is offered to the selector. stop_early: give up the downward pass once the
  // estimated size exceeds the best output so far by 1%.
  void run(const JPEGData& jpg, OutputImage& img, uint8_t comp_mask, double target_mul, bool stop_early);

 private:
  void adjust(const JPEGData& jpg, OutputImage& img, uint8_t comp_mask, double target_mul, bool stop_early,
              const ZeroingCandidates& candidates);

  const EncoderParams* params_;
  IBlockZeroingSearch* search_;
  Comparator* comparator_;
  OutputSelector* selector_;
  ProcessStats* stats_;
};

}  // namespace guetzli

#endif  // GUETZLI_CORE_ENCODER_FREQUENCY_MASKING_H_

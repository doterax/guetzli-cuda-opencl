// Encoder settings (filled by the application from Config).
#ifndef GUETZLI_CORE_ENCODER_ENCODER_PARAMS_H_
#define GUETZLI_CORE_ENCODER_ENCODER_PARAMS_H_

namespace guetzli {

struct EncoderParams {
  float butteraugli_target = 1.0;
  bool clear_metadata = true;
  bool try_420 = false;
  bool force_420 = false;
  bool use_silver_screen = false;
  int zeroing_greedy_lookahead = 3;
  bool new_zeroing_model = true;
};

}  // namespace guetzli

#endif  // GUETZLI_CORE_ENCODER_ENCODER_PARAMS_H_

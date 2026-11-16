/*
 * Copyright 2016 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef GUETZLI_STATS_H_
#define GUETZLI_STATS_H_

namespace guetzli {

// Per-image encoder counters (the --verbose trace goes through the injected ILogger).
struct ProcessStats {
  int iterations = 0;       // quantization-matrix and frequency-masking iterations ("Iter" in the trace)
  int iterations_up = 0;    // frequency-masking iterations that raised the block error limits
  int iterations_down = 0;  // frequency-masking iterations that lowered them
};

}  // namespace guetzli

#endif  // GUETZLI_STATS_H_

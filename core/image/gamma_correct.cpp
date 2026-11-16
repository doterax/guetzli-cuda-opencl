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

#include "core/image/gamma_correct.h"

#include <array>
#include <cmath>

namespace guetzli {

std::array<double, 256> makeSrgb8ToLinearTable() {
  std::array<double, 256> table{};
  int i = 0;
  for (; i < 11; ++i) {
    table[i] = i / 12.92;
  }
  for (; i < 256; ++i) {
    table[i] = 255.0 * std::pow(((i / 255.0) + 0.055) / 1.055, 2.4);
  }
  return table;
}

const double* srgb8ToLinearTable() {
  static const std::array<double, 256> kSrgb8ToLinearTable = makeSrgb8ToLinearTable();
  return kSrgb8ToLinearTable.data();
}

}  // namespace guetzli

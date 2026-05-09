/*
 * Copyright 2021 The Modelbox Project Authors. All Rights Reserved.
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

#include "paddle_inference.h"

namespace modelbox {

Status PaddleInference::Init(const PaddleInferenceParams& /*p*/) {
  return {STATUS_NOTSUPPORT, "paddle: not implemented"};
}

Status PaddleInference::Infer(
    const std::vector<std::shared_ptr<Buffer>>& /*inputs*/,
    std::vector<std::shared_ptr<Buffer>>& /*outputs*/) {
  return {STATUS_NOTSUPPORT, "paddle: not implemented"};
}

}  // namespace modelbox

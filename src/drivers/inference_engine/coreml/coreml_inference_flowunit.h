/*
 * Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
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

#ifndef MODELBOX_COREML_INFERENCE_FLOWUNIT_H_
#define MODELBOX_COREML_INFERENCE_FLOWUNIT_H_

#include <modelbox/base/status.h>
#include <modelbox/data_context.h>
#include <modelbox/flowunit.h>

#include <memory>
#include <string>

#include "coreml_inference.h"

// Shared base for the apple_silicon CoreML flowunit driver. The current port
// only exposes one device subclass (apple_silicon), but the indirection
// matches the OpenVINO base so future Mac-only device variants (e.g. ANE-only)
// can reuse this without duplicating the inference logic.
class CoreMLInferenceFlowUnit : public modelbox::FlowUnit {
 public:
  explicit CoreMLInferenceFlowUnit(std::string target_device);
  ~CoreMLInferenceFlowUnit() override;

  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;
  modelbox::Status Close() override;
  modelbox::Status Process(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

 private:
  std::string target_device_;
  std::unique_ptr<CoreMLInference> infer_;
};

#endif  // MODELBOX_COREML_INFERENCE_FLOWUNIT_H_

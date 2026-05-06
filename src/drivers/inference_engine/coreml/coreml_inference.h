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

#ifndef MODELBOX_COREML_INFERENCE_H_
#define MODELBOX_COREML_INFERENCE_H_

#include <modelbox/base/configuration.h>
#include <modelbox/base/status.h>
#include <modelbox/data_context.h>
#include <modelbox/flowunit.h>

#include <memory>
#include <string>
#include <vector>

constexpr const char *INFERENCE_TYPE = "coreml";

struct CoreMLIOList {
  std::vector<std::string> input_name_list;
  std::vector<std::string> output_name_list;
  std::vector<std::string> input_type_list;
  std::vector<std::string> output_type_list;
};

// Forward-declare the implementation pimpl so this header stays C++-only.
// The .mm file holds the strong reference to the Objective-C MLModel object.
class CoreMLInferenceImpl;

class CoreMLInference {
 public:
  // target_device is accepted for symmetry with the OpenVINO base; Core ML
  // selects the compute unit at predict time via MLComputeUnitsAll, so this
  // field is informational only ("apple_silicon" / "cpu" / "gpu" / "ane").
  explicit CoreMLInference(std::string target_device);
  ~CoreMLInference();

  modelbox::Status Open(const std::shared_ptr<modelbox::Configuration> &opts,
                        std::shared_ptr<modelbox::FlowUnitDesc> flowunit_desc);

  modelbox::Status Infer(
      const std::shared_ptr<modelbox::DataContext> &data_ctx);

 private:
  modelbox::Status GetFlowUnitIO(
      const std::shared_ptr<modelbox::FlowUnitDesc> &flowunit_desc);

  std::string target_device_;
  std::string model_entry_;
  CoreMLIOList io_list_;
  std::unique_ptr<CoreMLInferenceImpl> impl_;
};

#endif  // MODELBOX_COREML_INFERENCE_H_

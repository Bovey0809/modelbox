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

#ifndef MODELBOX_OPENVINO_INFERENCE_H_
#define MODELBOX_OPENVINO_INFERENCE_H_

#include <modelbox/base/configuration.h>
#include <modelbox/base/status.h>
#include <modelbox/data_context.h>
#include <modelbox/flowunit.h>

#include <memory>
#include <string>
#include <vector>

#include "openvino/openvino.hpp"

constexpr const char *INFERENCE_TYPE = "openvino";

struct OpenVINOIOList {
  std::vector<std::string> input_name_list;
  std::vector<std::string> output_name_list;
  std::vector<std::string> input_type_list;
  std::vector<std::string> output_type_list;
};

class OpenVINOInference {
 public:
  // target_device is the OpenVINO device string ("CPU", "GPU", "AUTO", ...).
  explicit OpenVINOInference(std::string target_device);
  ~OpenVINOInference();

  modelbox::Status Open(const std::shared_ptr<modelbox::Configuration> &opts,
                        std::shared_ptr<modelbox::FlowUnitDesc> flowunit_desc);

  modelbox::Status Infer(
      const std::shared_ptr<modelbox::DataContext> &data_ctx);

 private:
  modelbox::Status GetFlowUnitIO(
      const std::shared_ptr<modelbox::FlowUnitDesc> &flowunit_desc);

  modelbox::Status SetInputs(
      const std::shared_ptr<modelbox::DataContext> &data_ctx);

  modelbox::Status WriteOutputs(
      const std::shared_ptr<modelbox::DataContext> &data_ctx);

  static modelbox::ModelBoxDataType OvElementToModelBox(ov::element::Type type);
  static size_t OvElementSize(ov::element::Type type);

  std::string target_device_;
  std::string model_entry_;
  // When set to "ultralytics_image" the model is wrapped with an OpenVINO
  // PrePostProcessor that accepts BGR uint8 NHWC frames at any resolution
  // and bakes resize / scale(/255) / NHWC->NCHW into the GPU graph itself.
  // Empty string = no built-in preprocess (caller must pre-shape the buffer).
  std::string preprocess_mode_;
  ov::Core core_;
  ov::CompiledModel compiled_model_;
  ov::InferRequest infer_request_;
  OpenVINOIOList io_list_;
};

#endif  // MODELBOX_OPENVINO_INFERENCE_H_

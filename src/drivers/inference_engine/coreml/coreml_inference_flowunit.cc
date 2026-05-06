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

#include "coreml_inference_flowunit.h"

#include <modelbox/base/log.h>

#include <memory>
#include <utility>

CoreMLInferenceFlowUnit::CoreMLInferenceFlowUnit(std::string target_device)
    : target_device_(std::move(target_device)) {}

CoreMLInferenceFlowUnit::~CoreMLInferenceFlowUnit() = default;

modelbox::Status CoreMLInferenceFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  infer_.reset(new CoreMLInference(target_device_));
  return infer_->Open(opts, GetFlowUnitDesc());
}

modelbox::Status CoreMLInferenceFlowUnit::Close() {
  infer_.reset();
  return modelbox::STATUS_OK;
}

modelbox::Status CoreMLInferenceFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  if (infer_ == nullptr) {
    return {modelbox::STATUS_FAULT, "coreml flowunit not opened"};
  }
  return infer_->Infer(data_ctx);
}

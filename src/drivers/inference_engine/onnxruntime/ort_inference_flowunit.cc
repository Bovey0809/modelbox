/*
 * Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include "ort_inference_flowunit.h"

#include <modelbox/base/log.h>

#include <utility>

OrtInferenceFlowUnit::OrtInferenceFlowUnit(std::string backend)
    : backend_(std::move(backend)) {}

OrtInferenceFlowUnit::~OrtInferenceFlowUnit() = default;

modelbox::Status OrtInferenceFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  infer_.reset(new OrtInference(backend_));
  return infer_->Open(opts, GetFlowUnitDesc());
}

modelbox::Status OrtInferenceFlowUnit::Close() {
  infer_.reset();
  return modelbox::STATUS_OK;
}

modelbox::Status OrtInferenceFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  if (infer_ == nullptr) {
    return {modelbox::STATUS_FAULT, "onnxruntime flowunit not opened"};
  }
  return infer_->Infer(data_ctx);
}

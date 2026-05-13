/*
 * Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#ifndef MODELBOX_ORT_INFERENCE_FLOWUNIT_H_
#define MODELBOX_ORT_INFERENCE_FLOWUNIT_H_

#include <modelbox/base/status.h>
#include <modelbox/data_context.h>
#include <modelbox/flowunit.h>

#include <memory>
#include <string>

#include "ort_inference.h"

// Shared base for per-device ORT flow units. Device subclasses pick a backend
// ("CPU"/"CUDA"/"CoreML"/...) via the constructor; everything else lives here.
class OrtInferenceFlowUnit : public modelbox::FlowUnit {
 public:
  explicit OrtInferenceFlowUnit(std::string backend);
  ~OrtInferenceFlowUnit() override;

  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;
  modelbox::Status Close() override;
  modelbox::Status Process(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

 private:
  std::string backend_;
  std::unique_ptr<OrtInference> infer_;
};

#endif  // MODELBOX_ORT_INFERENCE_FLOWUNIT_H_

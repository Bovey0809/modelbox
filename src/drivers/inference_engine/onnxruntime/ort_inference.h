/*
 * Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#ifndef MODELBOX_ORT_INFERENCE_H_
#define MODELBOX_ORT_INFERENCE_H_

#include <modelbox/base/configuration.h>
#include <modelbox/base/status.h>
#include <modelbox/data_context.h>
#include <modelbox/flowunit.h>

#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"

constexpr const char *INFERENCE_TYPE = "onnxruntime";

struct OrtIOList {
  std::vector<std::string> input_name_list;
  std::vector<std::string> output_name_list;
  std::vector<std::string> input_type_list;
  std::vector<std::string> output_type_list;
};

class OrtInference {
 public:
  // backend is the ORT execution provider tag ("CPU", "CUDA", "CoreML", "ROCM",
  // "DML"). Anything unknown falls back to the default CPU provider.
  explicit OrtInference(std::string backend);
  ~OrtInference();

  modelbox::Status Open(const std::shared_ptr<modelbox::Configuration> &opts,
                        std::shared_ptr<modelbox::FlowUnitDesc> flowunit_desc);

  modelbox::Status Infer(
      const std::shared_ptr<modelbox::DataContext> &data_ctx);

 private:
  modelbox::Status GetFlowUnitIO(
      const std::shared_ptr<modelbox::FlowUnitDesc> &flowunit_desc);

  modelbox::Status BuildSessionOptions(Ort::SessionOptions &options);

  modelbox::Status RunOnce(
      const std::shared_ptr<modelbox::DataContext> &data_ctx);

  static modelbox::ModelBoxDataType OrtTypeToModelBox(
      ONNXTensorElementDataType type);
  static size_t OrtTypeSize(ONNXTensorElementDataType type);

  std::string backend_;
  std::string model_entry_;
  Ort::Env env_;
  std::unique_ptr<Ort::Session> session_;
  Ort::AllocatorWithDefaultOptions allocator_;
  // Cached input/output names from the model (owned by allocator_).
  std::vector<Ort::AllocatedStringPtr> model_input_names_owned_;
  std::vector<Ort::AllocatedStringPtr> model_output_names_owned_;
  std::vector<const char *> model_input_names_;
  std::vector<const char *> model_output_names_;
  // Per-input static shapes (-1 means dynamic).
  std::vector<std::vector<int64_t>> model_input_shapes_;
  std::vector<ONNXTensorElementDataType> model_input_types_;
  std::vector<ONNXTensorElementDataType> model_output_types_;
  OrtIOList io_list_;
};

#endif  // MODELBOX_ORT_INFERENCE_H_

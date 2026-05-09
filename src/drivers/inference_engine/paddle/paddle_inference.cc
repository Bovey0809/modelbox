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

#include <modelbox/base/log.h>

#include <cstring>

#include "paddle_inference_api.h"

namespace modelbox {

Status PaddleInference::Init(const PaddleInferenceParams& p) {
  if (p.model_file.empty() || p.params_file.empty()) {
    return {STATUS_BADCONF, "paddle: model_file/params_file required"};
  }

  paddle_infer::Config cfg;
  cfg.SetModel(p.model_file, p.params_file);
  cfg.DisableGlogInfo();
  cfg.SwitchUseFeedFetchOps(false);
  cfg.SwitchSpecifyInputNames(true);
  cfg.EnableMemoryOptim();

  device_ = p.device;
  if (p.device == "gpu") {
    cfg.EnableUseGpu(256, p.gpu_id);
    if (p.enable_trt) {
      auto prec = paddle_infer::Config::Precision::kFloat32;
      if (p.trt_precision == "fp16") {
        prec = paddle_infer::Config::Precision::kHalf;
      } else if (p.trt_precision == "int8") {
        prec = paddle_infer::Config::Precision::kInt8;
      }
      cfg.EnableTensorRtEngine(
          static_cast<int64_t>(p.trt_workspace_mb) << 20, 1, 3, prec, false,
          false);
    }
  } else {
    cfg.DisableGpu();
    cfg.SetCpuMathLibraryNumThreads(p.cpu_threads);
    if (p.enable_mkldnn) {
      cfg.EnableMKLDNN();
    }
  }

  predictor_ = paddle_infer::CreatePredictor(cfg);
  if (!predictor_) {
    return {STATUS_FAULT, "paddle: CreatePredictor failed"};
  }

  input_names_ =
      p.input_names.empty() ? predictor_->GetInputNames() : p.input_names;
  output_names_ =
      p.output_names.empty() ? predictor_->GetOutputNames() : p.output_names;
  MBLOG_INFO << "paddle: loaded " << p.model_file
             << " inputs=" << input_names_.size()
             << " outputs=" << output_names_.size();
  return STATUS_OK;
}

Status PaddleInference::Infer(
    const std::vector<std::shared_ptr<Buffer>>& inputs,
    std::vector<std::shared_ptr<Buffer>>& outputs) {
  if (!predictor_) {
    return {STATUS_FAULT, "paddle: predictor not initialized"};
  }
  if (inputs.size() != input_names_.size()) {
    return {STATUS_FAULT, "paddle: input count mismatch"};
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto handle = predictor_->GetInputHandle(input_names_[i]);
    std::vector<size_t> shape_sz;
    inputs[i]->Get("shape", shape_sz);
    if (shape_sz.empty()) {
      return {STATUS_FAULT,
              "paddle: input " + input_names_[i] + " missing shape meta"};
    }
    std::vector<int> shape(shape_sz.begin(), shape_sz.end());
    handle->Reshape(shape);
    handle->CopyFromCpu(static_cast<const float*>(inputs[i]->ConstData()));
  }
  if (!predictor_->Run()) {
    return {STATUS_FAULT, "paddle: Run failed"};
  }

  outputs.resize(output_names_.size());
  for (size_t i = 0; i < output_names_.size(); ++i) {
    auto handle = predictor_->GetOutputHandle(output_names_[i]);
    auto shape = handle->shape();
    size_t numel = 1;
    for (auto d : shape) {
      numel *= static_cast<size_t>(d);
    }
    auto buf = std::make_shared<Buffer>();
    auto status = buf->Build(numel * sizeof(float));
    if (!status) {
      return {STATUS_FAULT, "paddle: Buffer::Build failed"};
    }
    handle->CopyToCpu(static_cast<float*>(buf->MutableData()));
    std::vector<size_t> shape_sz(shape.begin(), shape.end());
    buf->Set("shape", shape_sz);
    outputs[i] = buf;
  }
  return STATUS_OK;
}

}  // namespace modelbox

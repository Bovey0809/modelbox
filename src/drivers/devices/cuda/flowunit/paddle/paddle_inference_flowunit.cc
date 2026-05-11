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

#include "paddle_inference_flowunit.h"

#include <modelbox/base/log.h>
#include <modelbox/flowunit_api_helper.h>

#include <cstring>

#include "virtualdriver_inference.h"

modelbox::Status PaddleInferenceFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  auto unit_desc = std::dynamic_pointer_cast<VirtualInferenceFlowUnitDesc>(
      this->GetFlowUnitDesc());
  if (!unit_desc) {
    return {modelbox::STATUS_BADCONF,
            "paddle_inference: flowunit desc is not VirtualInferenceFlowUnitDesc"};
  }

  for (const auto &in : unit_desc->GetFlowUnitInput()) {
    in_ports_.push_back(in.GetPortName());
  }
  for (const auto &out : unit_desc->GetFlowUnitOutput()) {
    out_ports_.push_back(out.GetPortName());
  }
  if (in_ports_.empty() || out_ports_.empty()) {
    return {modelbox::STATUS_BADCONF,
            "paddle_inference: input/output ports missing in TOML"};
  }

  modelbox::PaddleInferenceParams p;
  p.model_file = opts->GetString("model_file", unit_desc->GetModelEntry());
  p.params_file = opts->GetString("params_file");
  p.device = "gpu";
  p.gpu_id = opts->GetInt32("gpu_id", 0);
  p.enable_trt = opts->GetBool("enable_trt", false);
  p.trt_workspace_mb = opts->GetInt32("trt_workspace_mb", 256);
  p.trt_precision = opts->GetString("trt_precision", "fp32");
  p.input_names = opts->GetStrings("input_name");
  p.output_names = opts->GetStrings("output_name");

  if (p.input_names.empty()) {
    p.input_names = in_ports_;
  }
  if (p.output_names.empty()) {
    p.output_names = out_ports_;
  }

  engine_.reset(new modelbox::PaddleInference());
  auto st = engine_->Init(p);
  if (!st) {
    return st;
  }
  MBLOG_INFO << "paddle_inference: ready, in_ports=" << in_ports_.size()
             << " out_ports=" << out_ports_.size();
  return modelbox::STATUS_OK;
}

modelbox::Status PaddleInferenceFlowUnit::Close() {
  engine_.reset();
  return modelbox::STATUS_OK;
}

modelbox::Status PaddleInferenceFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> ctx) {
  std::vector<std::shared_ptr<modelbox::Buffer>> ins;
  std::vector<std::shared_ptr<modelbox::Buffer>> outs;
  for (const auto &name : in_ports_) {
    auto port = ctx->Input(name);
    if (!port || port->Size() == 0) {
      return {modelbox::STATUS_FAULT,
              "paddle_inference: empty input port " + name};
    }
    ins.push_back(port->At(0));
  }

  auto st = engine_->Infer(ins, outs);
  if (!st) {
    return st;
  }
  if (outs.size() != out_ports_.size()) {
    return {modelbox::STATUS_FAULT,
            "paddle_inference: output count mismatch (got " +
                std::to_string(outs.size()) + ", expected " +
                std::to_string(out_ports_.size()) + ")"};
  }
  for (size_t i = 0; i < out_ports_.size(); ++i) {
    auto port = ctx->Output(out_ports_[i]);
    auto build = port->Build({outs[i]->GetBytes()});
    if (!build) {
      return build;
    }
    std::memcpy(port->At(0)->MutableData(), outs[i]->ConstData(),
                outs[i]->GetBytes());
    std::vector<size_t> shape;
    if (outs[i]->Get("shape", shape)) {
      port->At(0)->Set("shape", shape);
    }
  }
  return modelbox::STATUS_OK;
}

std::shared_ptr<modelbox::FlowUnit>
PaddleInferenceFlowUnitFactory::VirtualCreateFlowUnit(
    const std::string &unit_name, const std::string &unit_type,
    const std::string &virtual_type) {
  return std::make_shared<PaddleInferenceFlowUnit>();
}

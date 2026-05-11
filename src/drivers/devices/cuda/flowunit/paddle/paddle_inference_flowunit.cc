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
  // Merge model TOML's [config] (carried via unit_desc) with the per-instance
  // graph node attributes from opts. Pattern from mindspore_inference.cc.
  auto merged = std::make_shared<modelbox::Configuration>();
  merged->Add(*unit_desc->GetConfiguration());
  merged->Add(*opts);

  p.model_file = merged->GetString("config.model_file", unit_desc->GetModelEntry());
  p.params_file = merged->GetString("config.params_file");
  p.device = merged->GetString("config.runtime", "gpu");  // "gpu" | "cpu"
  p.gpu_id = merged->GetInt32("config.gpu_id", 0);
  p.enable_trt = merged->GetBool("config.enable_trt", false);
  p.trt_workspace_mb = merged->GetInt32("config.trt_workspace_mb", 256);
  p.trt_precision = merged->GetString("config.trt_precision", "fp32");
  p.input_names = merged->GetStrings("config.input_name");
  p.output_names = merged->GetStrings("config.output_name");

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
  // Paddle's CopyFromCpu/CopyToCpu expects host memory. Migrate every input
  // buffer to the host CPU device before handing it to Paddle.
  auto host_device =
      this->GetBindDevice()->GetDeviceManager()->GetDevice("cpu", "0");
  if (!host_device) {
    return {modelbox::STATUS_FAULT,
            "paddle_inference: cannot acquire cpu device"};
  }

  // All input ports must carry the same batch size (one buffer per frame).
  size_t batch = 0;
  for (const auto &name : in_ports_) {
    auto port = ctx->Input(name);
    if (!port) {
      return {modelbox::STATUS_FAULT,
              "paddle_inference: missing input port " + name};
    }
    if (batch == 0) {
      batch = port->Size();
    } else if (port->Size() != batch) {
      return {modelbox::STATUS_FAULT,
              "paddle_inference: input ports have mismatched batch sizes"};
    }
  }
  if (batch == 0) {
    return modelbox::STATUS_OK;
  }

  // Pre-size output ports to match batch.
  std::vector<std::shared_ptr<modelbox::BufferList>> out_ports(out_ports_.size());
  std::vector<std::vector<std::vector<uint8_t>>> out_blobs(out_ports_.size());
  std::vector<std::vector<std::vector<size_t>>> out_shapes(out_ports_.size());
  for (size_t i = 0; i < out_ports_.size(); ++i) {
    out_ports[i] = ctx->Output(out_ports_[i]);
    out_blobs[i].resize(batch);
    out_shapes[i].resize(batch);
  }

  for (size_t b = 0; b < batch; ++b) {
    std::vector<std::shared_ptr<modelbox::Buffer>> ins;
    std::vector<std::shared_ptr<modelbox::Buffer>> outs;
    for (const auto &name : in_ports_) {
      auto port = ctx->Input(name);
      auto buf = port->At(b);
      if (!buf->GetDevice() || buf->GetDevice()->GetType() != "cpu") {
        buf = buf->CopyTo(host_device);
      }
      ins.push_back(buf);
    }
    auto st = engine_->Infer(ins, outs, host_device);
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
      const auto bytes = outs[i]->GetBytes();
      out_blobs[i][b].resize(bytes);
      std::memcpy(out_blobs[i][b].data(), outs[i]->ConstData(), bytes);
      outs[i]->Get("shape", out_shapes[i][b]);
    }
  }

  for (size_t i = 0; i < out_ports_.size(); ++i) {
    std::vector<size_t> sizes(batch);
    size_t total = 0;
    for (size_t b = 0; b < batch; ++b) {
      sizes[b] = out_blobs[i][b].size();
      total += sizes[b];
    }
    std::vector<uint8_t> contig(total);
    size_t off = 0;
    for (size_t b = 0; b < batch; ++b) {
      std::memcpy(contig.data() + off, out_blobs[i][b].data(), sizes[b]);
      off += sizes[b];
    }
    auto build = out_ports[i]->BuildFromHost(sizes, contig.data(), total);
    if (!build) {
      return build;
    }
    for (size_t b = 0; b < batch; ++b) {
      if (!out_shapes[i][b].empty()) {
        out_ports[i]->At(b)->Set("shape", out_shapes[i][b]);
      }
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

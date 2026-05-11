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

#ifndef MODELBOX_FLOWUNIT_PADDLE_INFERENCE_CUDA_H_
#define MODELBOX_FLOWUNIT_PADDLE_INFERENCE_CUDA_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/buffer.h>
#include <modelbox/device/cuda/device_cuda.h>
#include <modelbox/flow.h>
#include <modelbox/flowunit.h>

#include <memory>
#include <string>
#include <vector>

#include "paddle_inference.h"

constexpr const char *FLOWUNIT_TYPE = "cuda";
constexpr const char *INFERENCE_TYPE = "paddle";

class PaddleInferenceFlowUnit : public modelbox::FlowUnit {
 public:
  PaddleInferenceFlowUnit() = default;
  ~PaddleInferenceFlowUnit() override = default;

  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;

  modelbox::Status Close() override;

  modelbox::Status Process(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

 private:
  std::unique_ptr<modelbox::PaddleInference> engine_;
  std::vector<std::string> in_ports_;
  std::vector<std::string> out_ports_;
};

class PaddleInferenceFlowUnitFactory : public modelbox::FlowUnitFactory {
 public:
  PaddleInferenceFlowUnitFactory() = default;
  ~PaddleInferenceFlowUnitFactory() override = default;

  std::shared_ptr<modelbox::FlowUnit> VirtualCreateFlowUnit(
      const std::string &unit_name, const std::string &unit_type,
      const std::string &virtual_type) override;

  std::string GetFlowUnitFactoryType() override { return FLOWUNIT_TYPE; }
  std::string GetVirtualType() override { return INFERENCE_TYPE; }

  std::map<std::string, std::shared_ptr<modelbox::FlowUnitDesc>> FlowUnitProbe()
      override {
    return std::map<std::string, std::shared_ptr<modelbox::FlowUnitDesc>>();
  }
};

#endif  // MODELBOX_FLOWUNIT_PADDLE_INFERENCE_CUDA_H_

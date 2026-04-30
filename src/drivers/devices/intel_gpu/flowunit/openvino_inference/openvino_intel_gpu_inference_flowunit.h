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

#ifndef MODELBOX_FLOWUNIT_INFERENCE_OPENVINO_INTEL_GPU_H_
#define MODELBOX_FLOWUNIT_INFERENCE_OPENVINO_INTEL_GPU_H_

#include <modelbox/flowunit.h>

#include "openvino_inference_flowunit.h"

constexpr const char *FLOWUNIT_TYPE = "intel_gpu";

class OpenVINOIntelGpuFlowUnit : public OpenVINOInferenceFlowUnit {
 public:
  // Map the modelbox device "intel_gpu" to OpenVINO's "GPU" device string,
  // which the OpenVINO clDNN / Level Zero plugin resolves to the Arc.
  OpenVINOIntelGpuFlowUnit() : OpenVINOInferenceFlowUnit("GPU") {}
  ~OpenVINOIntelGpuFlowUnit() override = default;
};

class OpenVINOIntelGpuFlowUnitFactory : public modelbox::FlowUnitFactory {
 public:
  OpenVINOIntelGpuFlowUnitFactory() = default;
  ~OpenVINOIntelGpuFlowUnitFactory() override = default;

  std::shared_ptr<modelbox::FlowUnit> VirtualCreateFlowUnit(
      const std::string &unit_name, const std::string &unit_type,
      const std::string &virtual_type) override;

  std::string GetFlowUnitFactoryType() override { return FLOWUNIT_TYPE; }
  std::string GetVirtualType() override { return INFERENCE_TYPE; }

  // Input buffers stay on the cpu device; OpenVINO's GPU plugin handles
  // the host->Arc upload internally via its Level Zero backend during
  // compile_model("GPU"). Mirrors the CUDA + TensorFlow flowunit pattern.
  std::string GetFlowUnitInputDeviceType() override { return "cpu"; }

  std::map<std::string, std::shared_ptr<modelbox::FlowUnitDesc>> FlowUnitProbe()
      override {
    return {};
  }
};

#endif  // MODELBOX_FLOWUNIT_INFERENCE_OPENVINO_INTEL_GPU_H_

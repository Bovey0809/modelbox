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

#ifndef MODELBOX_FLOWUNIT_INFERENCE_COREML_APPLE_SILICON_H_
#define MODELBOX_FLOWUNIT_INFERENCE_COREML_APPLE_SILICON_H_

#include <modelbox/flowunit.h>

#include "coreml_inference_flowunit.h"

constexpr const char *FLOWUNIT_TYPE = "apple_silicon";

class CoreMLAppleSiliconFlowUnit : public CoreMLInferenceFlowUnit {
 public:
  // CoreML's compute unit selection happens via MLModelConfiguration; the
  // target_device string is informational. We pass "apple_silicon" so logs
  // and error messages are unambiguous.
  CoreMLAppleSiliconFlowUnit() : CoreMLInferenceFlowUnit("apple_silicon") {}
  ~CoreMLAppleSiliconFlowUnit() override = default;
};

class CoreMLAppleSiliconFlowUnitFactory : public modelbox::FlowUnitFactory {
 public:
  CoreMLAppleSiliconFlowUnitFactory() = default;
  ~CoreMLAppleSiliconFlowUnitFactory() override = default;

  std::shared_ptr<modelbox::FlowUnit> VirtualCreateFlowUnit(
      const std::string &unit_name, const std::string &unit_type,
      const std::string &virtual_type) override;

  std::string GetFlowUnitFactoryType() override { return FLOWUNIT_TYPE; }
  std::string GetVirtualType() override { return INFERENCE_TYPE; }

  // Input buffers stay on the cpu device. CoreML internally moves the data
  // onto the chosen compute unit (CPU/GPU/ANE) at predict time. Mirrors the
  // intel_gpu OpenVINO flowunit input device contract.
  std::string GetFlowUnitInputDeviceType() override { return "cpu"; }

  std::map<std::string, std::shared_ptr<modelbox::FlowUnitDesc>>
  FlowUnitProbe() override {
    return {};
  }
};

#endif  // MODELBOX_FLOWUNIT_INFERENCE_COREML_APPLE_SILICON_H_

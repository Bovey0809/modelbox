/*
 * Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#ifndef MODELBOX_FLOWUNIT_INFERENCE_ONNXRUNTIME_CUDA_H_
#define MODELBOX_FLOWUNIT_INFERENCE_ONNXRUNTIME_CUDA_H_

#include <modelbox/flowunit.h>

#include "ort_inference_flowunit.h"

constexpr const char *FLOWUNIT_TYPE = "cuda";

class OrtCudaFlowUnit : public OrtInferenceFlowUnit {
 public:
  OrtCudaFlowUnit() : OrtInferenceFlowUnit("CUDA") {}
  ~OrtCudaFlowUnit() override = default;
};

class OrtCudaFlowUnitFactory : public modelbox::FlowUnitFactory {
 public:
  OrtCudaFlowUnitFactory() = default;
  ~OrtCudaFlowUnitFactory() override = default;

  std::shared_ptr<modelbox::FlowUnit> VirtualCreateFlowUnit(
      const std::string &unit_name, const std::string &unit_type,
      const std::string &virtual_type) override;

  std::string GetFlowUnitFactoryType() override { return FLOWUNIT_TYPE; }
  std::string GetVirtualType() override { return INFERENCE_TYPE; }

  std::map<std::string, std::shared_ptr<modelbox::FlowUnitDesc>> FlowUnitProbe()
      override {
    return {};
  }
};

#endif  // MODELBOX_FLOWUNIT_INFERENCE_ONNXRUNTIME_CUDA_H_

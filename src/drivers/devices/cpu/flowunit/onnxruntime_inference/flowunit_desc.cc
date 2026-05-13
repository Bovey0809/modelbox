/*
 * Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include <stdio.h>

#include <memory>

#include "modelbox/base/driver_api_helper.h"
#include "modelbox/base/status.h"
#include "modelbox/device/cpu/device_cpu.h"
#include "modelbox/flowunit.h"
#include "ort_cpu_inference_flowunit.h"

constexpr const char *FLOWUNIT_NAME = "onnxruntime_inference";
constexpr const char *FLOWUNIT_DESC = "A cpu ONNX Runtime inference flowunit";

std::shared_ptr<modelbox::DriverFactory> CreateDriverFactory() {
  std::shared_ptr<modelbox::DriverFactory> factory =
      std::make_shared<OrtCpuFlowUnitFactory>();
  return factory;
}

void DriverDescription(modelbox::DriverDesc *desc) {
  desc->SetName(FLOWUNIT_NAME);
  desc->SetClass(modelbox::DRIVER_CLASS_INFERENCE);
  desc->SetType(modelbox::DEVICE_TYPE);
  desc->SetDescription(FLOWUNIT_DESC);
  desc->SetNodelete(true);
}

modelbox::Status DriverInit() { return modelbox::STATUS_OK; }

void DriverFini() {}

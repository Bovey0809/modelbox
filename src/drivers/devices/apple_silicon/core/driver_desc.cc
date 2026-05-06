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

#include <stdio.h>

#include <memory>

#include "modelbox/base/driver_api_helper.h"
#include "modelbox/base/timer.h"
#include "modelbox/device/apple_silicon/device_apple_silicon.h"

namespace modelbox {

std::shared_ptr<Timer> kAppleSiliconDeviceTimer;

Timer *GetTimer() { return kAppleSiliconDeviceTimer.get(); }

}  // namespace modelbox

std::shared_ptr<modelbox::DriverFactory> CreateDriverFactory() {
  std::shared_ptr<modelbox::DriverFactory> factory =
      std::make_shared<modelbox::AppleSiliconFactory>();
  return factory;
}

void DriverDescription(modelbox::DriverDesc *desc) {
  desc->SetClass(modelbox::DRIVER_CLASS_DEVICE);
  desc->SetType(modelbox::DEVICE_TYPE);
  desc->SetName(modelbox::DEVICE_DRIVER_NAME);
  desc->SetDescription(modelbox::DEVICE_DRIVER_DESCRIPTION);
}

modelbox::Status DriverInit() {
  if (modelbox::kAppleSiliconDeviceTimer != nullptr) {
    return modelbox::STATUS_OK;
  }

  modelbox::kAppleSiliconDeviceTimer = std::make_shared<modelbox::Timer>();
  modelbox::kAppleSiliconDeviceTimer->SetName("AppleSilicon-Timer");
  modelbox::kAppleSiliconDeviceTimer->Start();
  return modelbox::STATUS_OK;
}

void DriverFini() {
  if (modelbox::kAppleSiliconDeviceTimer == nullptr) {
    return;
  }
  modelbox::kAppleSiliconDeviceTimer->Stop();
  modelbox::kAppleSiliconDeviceTimer = nullptr;
}

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

#ifndef MODELBOX_DEVICE_APPLE_SILICON_H_
#define MODELBOX_DEVICE_APPLE_SILICON_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/base/timer.h>
#include <modelbox/device/apple_silicon/apple_silicon_memory.h>
#include <modelbox/flow.h>

namespace modelbox {

constexpr const char *DEVICE_TYPE = "apple_silicon";
constexpr const char *DEVICE_DRIVER_NAME = "device-apple-silicon";
constexpr const char *DEVICE_DRIVER_DESCRIPTION =
    "Apple Silicon (M-series) device driver routing inference through Core ML "
    "(CPU + GPU + Neural Engine).";

class AppleSilicon : public Device {
 public:
  AppleSilicon(const std::shared_ptr<DeviceMemoryManager> &mem_mgr);
  ~AppleSilicon() override = default;
  std::string GetType() const override;

  Status DeviceExecute(const DevExecuteCallBack &fun, int32_t priority,
                       size_t count) override;
  bool NeedResourceNice() override;
};

class AppleSiliconFactory : public DeviceFactory {
 public:
  AppleSiliconFactory() = default;
  ~AppleSiliconFactory() override = default;

  std::map<std::string, std::shared_ptr<DeviceDesc>> DeviceProbe() override;
  std::string GetDeviceFactoryType() override;
  std::vector<std::string> GetDeviceList() override;
  std::shared_ptr<Device> CreateDevice(const std::string &device_id) override;
};

class AppleSiliconDesc : public DeviceDesc {
 public:
  AppleSiliconDesc() = default;
  ~AppleSiliconDesc() override = default;
};

}  // namespace modelbox

#endif  // MODELBOX_DEVICE_APPLE_SILICON_H_

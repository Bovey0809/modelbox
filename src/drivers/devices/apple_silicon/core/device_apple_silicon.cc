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

#include "modelbox/device/apple_silicon/device_apple_silicon.h"

#include <stdio.h>
#include <sys/sysctl.h>

#include "modelbox/base/log.h"
#include "modelbox/base/utils.h"

namespace modelbox {

AppleSilicon::AppleSilicon(const std::shared_ptr<DeviceMemoryManager> &mem_mgr)
    : Device(mem_mgr) {}

std::string AppleSilicon::GetType() const { return DEVICE_TYPE; }

Status AppleSilicon::DeviceExecute(const DevExecuteCallBack &fun,
                                   int32_t priority, size_t count) {
  if (0 == count) {
    return STATUS_OK;
  }

  for (size_t i = 0; i < count; ++i) {
    auto status = fun(i);
    if ((status != STATUS_OK) && (status != STATUS_CONTINUE)) {
      MBLOG_WARN << "apple_silicon executor func failed: " << status;
      return status;
    }
  }

  return STATUS_OK;
}

bool AppleSilicon::NeedResourceNice() { return true; }

std::map<std::string, std::shared_ptr<DeviceDesc>>
AppleSiliconFactory::DeviceProbe() {
  std::map<std::string, std::shared_ptr<DeviceDesc>> return_map;

  // Apple Silicon presents a single combined CPU + GPU + ANE compute target.
  // Core ML decides at predict time which unit to dispatch to (configurable
  // via MLComputeUnitsAll / CPUOnly / GPUOnly). We expose one device id "0".
  uint64_t total_bytes = 0;
  size_t total_size = sizeof(total_bytes);
  std::string mem_str = "unknown";
  if (sysctlbyname("hw.memsize", &total_bytes, &total_size, nullptr, 0) == 0) {
    mem_str = std::to_string(total_bytes);
  }

  auto device_desc = std::make_shared<AppleSiliconDesc>();
  device_desc->SetDeviceDesc(
      "Apple Silicon (M-series) — CPU + GPU + Neural Engine via Core ML.");
  device_desc->SetDeviceId("0");
  device_desc->SetDeviceMemory(mem_str);
  device_desc->SetDeviceType(DEVICE_TYPE);
  return_map.insert(std::make_pair("0", device_desc));
  return return_map;
}

std::string AppleSiliconFactory::GetDeviceFactoryType() { return DEVICE_TYPE; }

std::vector<std::string> AppleSiliconFactory::GetDeviceList() {
  return std::vector<std::string>{"0"};
}

std::shared_ptr<Device> AppleSiliconFactory::CreateDevice(
    const std::string &device_id) {
  auto mem_mgr = std::make_shared<AppleSiliconMemoryManager>(device_id);
  auto status = mem_mgr->Init();
  if (!status) {
    StatusError = status;
    return nullptr;
  }
  return std::make_shared<AppleSilicon>(mem_mgr);
}

}  // namespace modelbox

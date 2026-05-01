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

#include "modelbox/device/intel_gpu/device_intel_gpu.h"

#include <level_zero/ze_api.h>
#include <stdio.h>

#include "modelbox/base/log.h"
#include "modelbox/base/utils.h"

namespace modelbox {

IntelGpu::IntelGpu(const std::shared_ptr<DeviceMemoryManager> &mem_mgr)
    : Device(mem_mgr) {}

std::string IntelGpu::GetType() const { return DEVICE_TYPE; }

Status IntelGpu::DeviceExecute(const DevExecuteCallBack &fun, int32_t priority,
                               size_t count) {
  if (0 == count) {
    return STATUS_OK;
  }

  for (size_t i = 0; i < count; ++i) {
    auto status = fun(i);
    if ((status != STATUS_OK) && (status != STATUS_CONTINUE)) {
      MBLOG_WARN << "intel_gpu executor func failed: " << status;
      return status;
    }
  }

  return STATUS_OK;
}

bool IntelGpu::NeedResourceNice() { return true; }

std::vector<std::string> IntelGpuFactory::ProbeLevelZeroGpus() {
  std::vector<std::string> ids;

  // ZE_INIT_FLAG_GPU_ONLY restricts the loader to GPU drivers only.
  ze_result_t res = zeInit(ZE_INIT_FLAG_GPU_ONLY);
  if (res != ZE_RESULT_SUCCESS) {
    MBLOG_WARN << "zeInit failed, ze_result=" << res;
    return ids;
  }

  uint32_t driver_count = 0;
  res = zeDriverGet(&driver_count, nullptr);
  if (res != ZE_RESULT_SUCCESS || driver_count == 0) {
    MBLOG_WARN << "zeDriverGet found no Level Zero drivers, ze_result=" << res;
    return ids;
  }

  std::vector<ze_driver_handle_t> drivers(driver_count);
  res = zeDriverGet(&driver_count, drivers.data());
  if (res != ZE_RESULT_SUCCESS) {
    MBLOG_WARN << "zeDriverGet failed, ze_result=" << res;
    return ids;
  }

  size_t global_idx = 0;
  for (auto driver : drivers) {
    uint32_t device_count = 0;
    res = zeDeviceGet(driver, &device_count, nullptr);
    if (res != ZE_RESULT_SUCCESS || device_count == 0) {
      continue;
    }

    std::vector<ze_device_handle_t> devices(device_count);
    res = zeDeviceGet(driver, &device_count, devices.data());
    if (res != ZE_RESULT_SUCCESS) {
      continue;
    }

    for (auto dev : devices) {
      ze_device_properties_t props{};
      props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
      res = zeDeviceGetProperties(dev, &props);
      if (res != ZE_RESULT_SUCCESS) {
        continue;
      }
      if (props.type != ZE_DEVICE_TYPE_GPU) {
        continue;
      }
      ids.emplace_back(std::to_string(global_idx));
      ++global_idx;
    }
  }

  return ids;
}

std::map<std::string, std::shared_ptr<DeviceDesc>>
IntelGpuFactory::DeviceProbe() {
  std::map<std::string, std::shared_ptr<DeviceDesc>> return_map;
  auto device_list = ProbeLevelZeroGpus();
  for (const auto &id : device_list) {
    auto device_desc = std::make_shared<IntelGpuDesc>();
    device_desc->SetDeviceDesc("Intel GPU device (Arc / DG2 via Level Zero).");
    device_desc->SetDeviceId(id);
    // Memory size is reported as "unknown" until the Level Zero memory layer
    // is added; OpenVINO's GPU plugin manages device memory on its own for now.
    device_desc->SetDeviceMemory("unknown");
    device_desc->SetDeviceType(DEVICE_TYPE);
    return_map.insert(std::make_pair(id, device_desc));
  }
  return return_map;
}

std::string IntelGpuFactory::GetDeviceFactoryType() { return DEVICE_TYPE; }

std::vector<std::string> IntelGpuFactory::GetDeviceList() {
  return ProbeLevelZeroGpus();
}

std::shared_ptr<Device> IntelGpuFactory::CreateDevice(
    const std::string &device_id) {
  auto mem_mgr = std::make_shared<IntelGpuMemoryManager>(device_id);
  auto status = mem_mgr->Init();
  if (!status) {
    StatusError = status;
    return nullptr;
  }
  return std::make_shared<IntelGpu>(mem_mgr);
}

}  // namespace modelbox

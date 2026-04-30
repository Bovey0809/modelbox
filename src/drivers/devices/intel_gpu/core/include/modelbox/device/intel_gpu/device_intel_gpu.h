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

#ifndef MODELBOX_DEVICE_INTEL_GPU_H_
#define MODELBOX_DEVICE_INTEL_GPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/base/timer.h>
#include <modelbox/device/intel_gpu/intel_gpu_memory.h>
#include <modelbox/flow.h>

namespace modelbox {

constexpr const char *DEVICE_TYPE = "intel_gpu";
constexpr const char *DEVICE_DRIVER_NAME = "device-intel-gpu";
constexpr const char *DEVICE_DRIVER_DESCRIPTION =
    "An Intel GPU device driver (Arc / DG2 via Level Zero)";

class IntelGpu : public Device {
 public:
  IntelGpu(const std::shared_ptr<DeviceMemoryManager> &mem_mgr);
  ~IntelGpu() override = default;
  std::string GetType() const override;

  Status DeviceExecute(const DevExecuteCallBack &fun, int32_t priority,
                       size_t count) override;
  bool NeedResourceNice() override;
};

class IntelGpuFactory : public DeviceFactory {
 public:
  IntelGpuFactory() = default;
  ~IntelGpuFactory() override = default;

  std::map<std::string, std::shared_ptr<DeviceDesc>> DeviceProbe() override;
  std::string GetDeviceFactoryType() override;
  std::vector<std::string> GetDeviceList() override;
  std::shared_ptr<Device> CreateDevice(const std::string &device_id) override;

 private:
  // Enumerate Intel GPUs visible to Level Zero. Returns one entry per
  // ZE_DEVICE_TYPE_GPU device discovered across all drivers.
  std::vector<std::string> ProbeLevelZeroGpus();
};

class IntelGpuDesc : public DeviceDesc {
 public:
  IntelGpuDesc() = default;
  ~IntelGpuDesc() override = default;
};

}  // namespace modelbox

#endif  // MODELBOX_DEVICE_INTEL_GPU_H_

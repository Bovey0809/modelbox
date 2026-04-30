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

#ifndef MODELBOX_INTEL_GPU_MEMORY_H_
#define MODELBOX_INTEL_GPU_MEMORY_H_

#include <modelbox/base/device.h>
#include <modelbox/base/memory_pool.h>
#include <modelbox/base/status.h>
#include <modelbox/base/timer.h>

#include <mutex>

namespace modelbox {

// First-pass implementation: buffers are allocated in host memory. This works
// because the OpenVINO inference flowunit declares input device = "cpu" and
// OpenVINO performs the host->Arc upload internally during compile_model("GPU").
// A future commit can swap this for zeMemAllocDevice + zeCommandListAppend-
// MemoryCopy to enable zero-copy buffer sharing with the Level Zero driver.

class IntelGpuMemory : public DeviceMemory {
 public:
  IntelGpuMemory(const std::shared_ptr<Device> &device,
                 const std::shared_ptr<DeviceMemoryManager> &mem_mgr,
                 const std::shared_ptr<void> &device_mem_ptr, size_t size);
  ~IntelGpuMemory() override = default;
};

class IntelGpuMemoryManager;

class IntelGpuMemoryPool : public MemoryPoolBase {
 public:
  IntelGpuMemoryPool(IntelGpuMemoryManager *mem_manager);
  ~IntelGpuMemoryPool() override;
  Status Init();
  void *MemAlloc(size_t size) override;
  void MemFree(void *ptr) override;
  virtual void OnTimer();
  size_t CalSlabSize(size_t object_size) override;

 private:
  IntelGpuMemoryManager *mem_manager_;
  std::shared_ptr<TimerTask> flush_timer_;
};

class IntelGpuMemoryManager : public DeviceMemoryManager {
 public:
  IntelGpuMemoryManager(const std::string &device_id);
  ~IntelGpuMemoryManager() override;

  Status Init();

  std::shared_ptr<DeviceMemory> MakeDeviceMemory(
      const std::shared_ptr<Device> &device, std::shared_ptr<void> mem_ptr,
      size_t size) override;

  void *Malloc(size_t size, uint32_t mem_flags) override;
  std::shared_ptr<void> AllocSharedPtr(size_t size,
                                       uint32_t mem_flags) override;
  void Free(void *mem_ptr, uint32_t mem_flags) override;

  Status Copy(void *dest, size_t dest_size, const void *src_buffer,
              size_t src_size, DeviceMemoryCopyKind kind) override;

  Status DeviceMemoryCopy(
      const std::shared_ptr<DeviceMemory> &dest_memory, size_t dest_offset,
      const std::shared_ptr<const DeviceMemory> &src_memory, size_t src_offset,
      size_t src_size,
      DeviceMemoryCopyKind copy_kind = DeviceMemoryCopyKind::FromHost) override;

  Status GetDeviceMemUsage(size_t *free, size_t *total) const override;

 private:
  IntelGpuMemoryPool mem_pool_;
  std::mutex malloc_mtx_;
};

}  // namespace modelbox

#endif  // MODELBOX_INTEL_GPU_MEMORY_H_

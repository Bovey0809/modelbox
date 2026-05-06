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

#ifndef MODELBOX_APPLE_SILICON_MEMORY_H_
#define MODELBOX_APPLE_SILICON_MEMORY_H_

#include <modelbox/base/device.h>
#include <modelbox/base/memory_pool.h>
#include <modelbox/base/status.h>
#include <modelbox/base/timer.h>

#include <mutex>

namespace modelbox {

// Apple's M-series uses a unified memory architecture: CPU, GPU, and Neural
// Engine see one physical pool. This first-pass driver allocates regular
// host buffers and lets Core ML route them to the chosen compute unit at
// predict time. A future iteration can switch to IOSurface-backed memory or
// MLMultiArray.dataPointer-aware allocations to enable zero-copy.

class AppleSiliconMemory : public DeviceMemory {
 public:
  AppleSiliconMemory(const std::shared_ptr<Device> &device,
                     const std::shared_ptr<DeviceMemoryManager> &mem_mgr,
                     const std::shared_ptr<void> &device_mem_ptr, size_t size);
  ~AppleSiliconMemory() override = default;
};

class AppleSiliconMemoryManager;

class AppleSiliconMemoryPool : public MemoryPoolBase {
 public:
  AppleSiliconMemoryPool(AppleSiliconMemoryManager *mem_manager);
  ~AppleSiliconMemoryPool() override;
  Status Init();
  void *MemAlloc(size_t size) override;
  void MemFree(void *ptr) override;
  virtual void OnTimer();
  size_t CalSlabSize(size_t object_size) override;

 private:
  AppleSiliconMemoryManager *mem_manager_;
  std::shared_ptr<TimerTask> flush_timer_;
};

class AppleSiliconMemoryManager : public DeviceMemoryManager {
 public:
  AppleSiliconMemoryManager(const std::string &device_id);
  ~AppleSiliconMemoryManager() override;

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
  AppleSiliconMemoryPool mem_pool_;
  std::mutex malloc_mtx_;
};

}  // namespace modelbox

#endif  // MODELBOX_APPLE_SILICON_MEMORY_H_

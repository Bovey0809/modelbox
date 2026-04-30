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

#include "modelbox/device/intel_gpu/intel_gpu_memory.h"

#include <securec.h>

#include <linux/kernel.h>
#include <linux/unistd.h>
#include <sys/sysinfo.h>

#include "modelbox/base/log.h"
#include "modelbox/device/intel_gpu/device_intel_gpu.h"

namespace modelbox {

IntelGpuMemory::IntelGpuMemory(
    const std::shared_ptr<Device> &device,
    const std::shared_ptr<DeviceMemoryManager> &mem_mgr,
    const std::shared_ptr<void> &device_mem_ptr, size_t size)
    : DeviceMemory(device, mem_mgr, device_mem_ptr, size, true) {}

IntelGpuMemoryPool::IntelGpuMemoryPool(IntelGpuMemoryManager *mem_manager)
    : mem_manager_(mem_manager) {}

Status IntelGpuMemoryPool::Init() {
  auto status = InitSlabCache();
  if (!status) {
    return {status, "init intel_gpu mempool failed."};
  }
  return STATUS_OK;
}

IntelGpuMemoryPool::~IntelGpuMemoryPool() {
  if (flush_timer_) {
    flush_timer_->Stop();
    flush_timer_ = nullptr;
  }
}

void IntelGpuMemoryPool::OnTimer() {
  // TODO support config shrink time.
}

void *IntelGpuMemoryPool::MemAlloc(size_t size) {
  return mem_manager_->Malloc(size, 0);
}

void IntelGpuMemoryPool::MemFree(void *ptr) { mem_manager_->Free(ptr, 0); }

size_t IntelGpuMemoryPool::CalSlabSize(size_t object_size) {
  return object_size;
}

IntelGpuMemoryManager::IntelGpuMemoryManager(const std::string &device_id)
    : DeviceMemoryManager(device_id), mem_pool_(this) {}

IntelGpuMemoryManager::~IntelGpuMemoryManager() { mem_pool_.DestroySlabCache(); }

Status IntelGpuMemoryManager::Init() { return mem_pool_.Init(); }

std::shared_ptr<DeviceMemory> IntelGpuMemoryManager::MakeDeviceMemory(
    const std::shared_ptr<Device> &device, std::shared_ptr<void> mem_ptr,
    size_t size) {
  return std::make_shared<IntelGpuMemory>(device, shared_from_this(), mem_ptr,
                                          size);
}

void *IntelGpuMemoryManager::Malloc(size_t size, uint32_t mem_flags) {
  if (size == 0) {
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(malloc_mtx_);
  void *ptr = malloc(size);
  if (ptr == nullptr) {
    MBLOG_ERROR << "intel_gpu malloc failed, size=" << size;
  }
  return ptr;
}

std::shared_ptr<void> IntelGpuMemoryManager::AllocSharedPtr(
    size_t size, uint32_t mem_flags) {
  return mem_pool_.AllocSharedPtr(size);
}

void IntelGpuMemoryManager::Free(void *mem_ptr, uint32_t mem_flags) {
  std::lock_guard<std::mutex> lock(malloc_mtx_);
  if (mem_ptr != nullptr) {
    free(mem_ptr);
  }
}

Status IntelGpuMemoryManager::Copy(void *dest, size_t dest_size,
                                   const void *src_buffer, size_t src_size,
                                   DeviceMemoryCopyKind kind) {
  if (dest == nullptr || src_buffer == nullptr) {
    MBLOG_ERROR << "intel_gpu copy null ptr, dest=" << dest
                << " src=" << src_buffer;
    return STATUS_INVALID;
  }

  if (dest_size < src_size) {
    MBLOG_ERROR << "intel_gpu memcpy failed, dest size[" << dest_size
                << "] < src size[" << src_size << "]";
    return STATUS_RANGE;
  }

  int ret = memcpy_s(dest, dest_size, src_buffer, src_size);
  if (ret != EOK) {
    MBLOG_ERROR << "intel_gpu copy memcpy_s failed, ret=" << ret;
    return STATUS_FAULT;
  }

  return STATUS_SUCCESS;
}

Status IntelGpuMemoryManager::DeviceMemoryCopy(
    const std::shared_ptr<DeviceMemory> &dest_memory, size_t dest_offset,
    const std::shared_ptr<const DeviceMemory> &src_memory, size_t src_offset,
    size_t src_size, DeviceMemoryCopyKind copy_kind) {
  auto *dest_ptr = dest_memory->GetPtr<uint8_t>().get() + dest_offset;
  const auto *src_ptr = src_memory->GetConstPtr<uint8_t>().get() + src_offset;
  auto ret = memcpy_s(dest_ptr, src_size, src_ptr, src_size);
  if (EOK != ret) {
    MBLOG_ERROR << "intel_gpu DeviceMemoryCopy memcpy_s failed, ret=" << ret
                << " src size " << src_size;
    return STATUS_FAULT;
  }
  return STATUS_OK;
}

Status IntelGpuMemoryManager::GetDeviceMemUsage(size_t *free,
                                                size_t *total) const {
  // First-pass: report host RAM. Replace with zeDeviceGetMemoryProperties
  // when the Level Zero memory layer lands.
  struct sysinfo s_info;
  auto ret = sysinfo(&s_info);
  if (ret != 0) {
    return {STATUS_FAULT, "sysinfo failed"};
  }
  if (free != nullptr) {
    *free = s_info.freeram;
  }
  if (total != nullptr) {
    *total = s_info.totalram;
  }
  return STATUS_SUCCESS;
}

}  // namespace modelbox

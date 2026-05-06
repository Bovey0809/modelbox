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

#include "modelbox/device/apple_silicon/apple_silicon_memory.h"

#include <mach/mach.h>
#include <mach/mach_host.h>
#include <securec.h>
#include <sys/sysctl.h>

#include "modelbox/base/log.h"
#include "modelbox/device/apple_silicon/device_apple_silicon.h"

namespace modelbox {

AppleSiliconMemory::AppleSiliconMemory(
    const std::shared_ptr<Device> &device,
    const std::shared_ptr<DeviceMemoryManager> &mem_mgr,
    const std::shared_ptr<void> &device_mem_ptr, size_t size)
    : DeviceMemory(device, mem_mgr, device_mem_ptr, size, true) {}

AppleSiliconMemoryPool::AppleSiliconMemoryPool(
    AppleSiliconMemoryManager *mem_manager)
    : mem_manager_(mem_manager) {}

Status AppleSiliconMemoryPool::Init() {
  auto status = InitSlabCache();
  if (!status) {
    return {status, "init apple_silicon mempool failed."};
  }
  return STATUS_OK;
}

AppleSiliconMemoryPool::~AppleSiliconMemoryPool() {
  if (flush_timer_) {
    flush_timer_->Stop();
    flush_timer_ = nullptr;
  }
}

void AppleSiliconMemoryPool::OnTimer() {}

void *AppleSiliconMemoryPool::MemAlloc(size_t size) {
  return mem_manager_->Malloc(size, 0);
}

void AppleSiliconMemoryPool::MemFree(void *ptr) {
  mem_manager_->Free(ptr, 0);
}

size_t AppleSiliconMemoryPool::CalSlabSize(size_t object_size) {
  return object_size;
}

AppleSiliconMemoryManager::AppleSiliconMemoryManager(
    const std::string &device_id)
    : DeviceMemoryManager(device_id), mem_pool_(this) {}

AppleSiliconMemoryManager::~AppleSiliconMemoryManager() {
  mem_pool_.DestroySlabCache();
}

Status AppleSiliconMemoryManager::Init() { return mem_pool_.Init(); }

std::shared_ptr<DeviceMemory> AppleSiliconMemoryManager::MakeDeviceMemory(
    const std::shared_ptr<Device> &device, std::shared_ptr<void> mem_ptr,
    size_t size) {
  return std::make_shared<AppleSiliconMemory>(device, shared_from_this(),
                                              mem_ptr, size);
}

void *AppleSiliconMemoryManager::Malloc(size_t size, uint32_t mem_flags) {
  if (size == 0) {
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(malloc_mtx_);
  void *ptr = malloc(size);
  if (ptr == nullptr) {
    MBLOG_ERROR << "apple_silicon malloc failed, size=" << size;
  }
  return ptr;
}

std::shared_ptr<void> AppleSiliconMemoryManager::AllocSharedPtr(
    size_t size, uint32_t mem_flags) {
  return mem_pool_.AllocSharedPtr(size);
}

void AppleSiliconMemoryManager::Free(void *mem_ptr, uint32_t mem_flags) {
  std::lock_guard<std::mutex> lock(malloc_mtx_);
  if (mem_ptr != nullptr) {
    free(mem_ptr);
  }
}

Status AppleSiliconMemoryManager::Copy(void *dest, size_t dest_size,
                                       const void *src_buffer, size_t src_size,
                                       DeviceMemoryCopyKind kind) {
  if (dest == nullptr || src_buffer == nullptr) {
    MBLOG_ERROR << "apple_silicon copy null ptr, dest=" << dest
                << " src=" << src_buffer;
    return STATUS_INVALID;
  }

  if (dest_size < src_size) {
    MBLOG_ERROR << "apple_silicon memcpy failed, dest size[" << dest_size
                << "] < src size[" << src_size << "]";
    return STATUS_RANGE;
  }

  int ret = memcpy_s(dest, dest_size, src_buffer, src_size);
  if (ret != EOK) {
    MBLOG_ERROR << "apple_silicon copy memcpy_s failed, ret=" << ret;
    return STATUS_FAULT;
  }

  return STATUS_SUCCESS;
}

Status AppleSiliconMemoryManager::DeviceMemoryCopy(
    const std::shared_ptr<DeviceMemory> &dest_memory, size_t dest_offset,
    const std::shared_ptr<const DeviceMemory> &src_memory, size_t src_offset,
    size_t src_size, DeviceMemoryCopyKind copy_kind) {
  auto *dest_ptr = dest_memory->GetPtr<uint8_t>().get() + dest_offset;
  const auto *src_ptr = src_memory->GetConstPtr<uint8_t>().get() + src_offset;
  auto ret = memcpy_s(dest_ptr, src_size, src_ptr, src_size);
  if (EOK != ret) {
    MBLOG_ERROR << "apple_silicon DeviceMemoryCopy memcpy_s failed, ret=" << ret
                << " src size " << src_size;
    return STATUS_FAULT;
  }
  return STATUS_OK;
}

Status AppleSiliconMemoryManager::GetDeviceMemUsage(size_t *free,
                                                    size_t *total) const {
  // Unified memory: report system RAM. hw.memsize for total; free pages from
  // host_statistics64 for free.
  if (total != nullptr) {
    uint64_t total_bytes = 0;
    size_t total_size = sizeof(total_bytes);
    if (sysctlbyname("hw.memsize", &total_bytes, &total_size, nullptr, 0) !=
        0) {
      return {STATUS_FAULT, "sysctlbyname(hw.memsize) failed"};
    }
    *total = static_cast<size_t>(total_bytes);
  }

  if (free != nullptr) {
    vm_size_t page_size = 0;
    mach_port_t host = mach_host_self();
    if (host_page_size(host, &page_size) != KERN_SUCCESS) {
      page_size = 4096;
    }
    vm_statistics64_data_t vm_stat;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(host, HOST_VM_INFO64,
                          reinterpret_cast<host_info64_t>(&vm_stat),
                          &count) != KERN_SUCCESS) {
      *free = 0;
    } else {
      *free = static_cast<size_t>(vm_stat.free_count) *
              static_cast<size_t>(page_size);
    }
  }

  return STATUS_SUCCESS;
}

}  // namespace modelbox

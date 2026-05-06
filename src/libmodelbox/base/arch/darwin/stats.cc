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

#include "stats.h"

#include <ifaddrs.h>
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <net/if.h>
#include <net/if_dl.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <fstream>
#include <memory>
#include <thread>

#include "modelbox/base/log.h"
#include "modelbox/base/os.h"
#include "modelbox/base/status.h"
#include "securec.h"

namespace modelbox {

OSInfo *os = &DarwinOSInfo::GetInstance();

OSProcess::OSProcess() = default;
OSProcess::~OSProcess() = default;

OSThread::OSThread() = default;
OSThread::~OSThread() = default;

OSInfo::OSInfo() = default;
OSInfo::~OSInfo() = default;

DarwinOSProcess::DarwinOSProcess() = default;

DarwinOSProcess::~DarwinOSProcess() = default;

uint32_t DarwinOSProcess::GetPid() { return getpid(); }

uint32_t DarwinOSProcess::GetPPid() { return getppid(); }

int32_t DarwinOSProcess::GetThreadsNumber(uint32_t pid) { return 0; }

uint32_t DarwinOSProcess::GetMemorySize(uint32_t pid) { return 0; }

uint32_t DarwinOSProcess::GetMemorySHR(uint32_t pid) { return 0; }

uint32_t DarwinOSProcess::GetMemoryRSS(uint32_t pid) { return 0; }

std::vector<uint32_t> DarwinOSProcess::GetProcessTime(uint32_t pid) {
  std::vector<uint32_t> ss{0, 0};
  return ss;
}

std::vector<uint32_t> DarwinOSProcess::GetTotalTime(uint32_t pid) {
  std::vector<uint32_t> ss{0, 0};
  return ss;
}

DarwinOSThread::DarwinOSThread() = default;
DarwinOSThread::~DarwinOSThread() = default;

std::thread::id DarwinOSThread::GetTid() {
  return std::this_thread::get_id();
}

Status DarwinOSThread::SetName(const std::string &name) {
  // Darwin's pthread_setname_np takes a single arg and only renames the
  // calling thread; this matches stats.cc usage on Linux which always names
  // the running thread.
  pthread_setname_np(name.c_str());
  return STATUS_OK;
}

Status DarwinOSThread::SetThreadPriority(const std::thread::id &thread,
                                         int32_t priority) {
  return STATUS_OK;
}

Status DarwinOSThread::SetThreadLogicalCPUAffinity(
    const std::thread::id &thread, const std::vector<int16_t> &l_cpus) {
  return STATUS_OK;
}

Status DarwinOSThread::SetThreadPhysicalCPUAffinity(
    const std::thread::id &thread, const std::vector<int16_t> &p_cpus) {
  return STATUS_OK;
}

int32_t DarwinOSThread::GetThreadPriority(const std::thread::id &thread) {
  return 0;
}

DarwinOSInfo::DarwinOSInfo() {
  Process = std::make_shared<DarwinOSProcess>();
  Thread = std::make_shared<DarwinOSThread>();
}

DarwinOSInfo::~DarwinOSInfo() = default;

DarwinOSInfo &DarwinOSInfo::GetInstance() {
  static DarwinOSInfo info;
  return info;
}

Status DarwinOSInfo::GetMemoryUsage(size_t *free, size_t *total) {
  uint64_t total_bytes = 0;
  size_t total_size = sizeof(total_bytes);
  if (sysctlbyname("hw.memsize", &total_bytes, &total_size, nullptr, 0) != 0) {
    MBLOG_ERROR << "sysctlbyname(hw.memsize) failed: " << StrError(errno);
    return STATUS_FAULT;
  }
  if (total != nullptr) {
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
      return STATUS_OK;
    }
    *free = static_cast<size_t>(vm_stat.free_count) *
            static_cast<size_t>(page_size);
  }

  return STATUS_SUCCESS;
}

std::vector<uint32_t> DarwinOSInfo::GetCpuRunTime() {
  std::vector<uint32_t> ss{0, 0};
  return ss;
}

int32_t DarwinOSInfo::GetPhysicalCpuNumbers() {
  int32_t ncpu = 1;
  size_t len = sizeof(ncpu);
  if (sysctlbyname("hw.physicalcpu", &ncpu, &len, nullptr, 0) != 0) {
    return 1;
  }
  return ncpu;
}

int32_t DarwinOSInfo::GetLogicalCpuNumbers() {
  return static_cast<int32_t>(sysconf(_SC_NPROCESSORS_ONLN));
}

std::string DarwinOSInfo::GetSystemID() {
  // Darwin has no /etc/machine-id; fall back to uname machine+nodename like
  // the Linux implementation does when machine-id is unavailable.
  struct utsname buf;
  if (uname(&buf) != 0) {
    StatusError = {STATUS_FAULT, StrError(errno)};
    return "";
  }
  std::string result = buf.machine;
  result += buf.nodename;
  return result;
}

std::string DarwinOSInfo::GetMacAddress(const std::string &nic) {
  std::string mac;
  struct ifaddrs *ifap = nullptr;
  if (getifaddrs(&ifap) != 0) {
    StatusError = {STATUS_FAULT, "getifaddrs failed"};
    return mac;
  }
  Defer { freeifaddrs(ifap); };

  StatusError = {STATUS_NOTFOUND, "not found nic"};
  for (struct ifaddrs *cur = ifap; cur != nullptr; cur = cur->ifa_next) {
    if (cur->ifa_addr == nullptr || cur->ifa_addr->sa_family != AF_LINK) {
      continue;
    }
    if ((cur->ifa_flags & IFF_LOOPBACK) != 0) {
      continue;
    }
    if (nic.length() > 0 && nic != cur->ifa_name) {
      continue;
    }

    auto *sdl = reinterpret_cast<struct sockaddr_dl *>(cur->ifa_addr);
    if (sdl->sdl_alen != 6) {
      continue;
    }
    auto *bytes = reinterpret_cast<uint8_t *>(LLADDR(sdl));
    char tmp[64];
    int len = snprintf_s(
        tmp, sizeof(tmp), sizeof(tmp), "%.2x:%.2x:%.2x:%.2x:%.2x:%.2x",
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5]);
    if (len < 0) {
      continue;
    }
    mac = tmp;
    break;
  }

  return mac;
}

}  // namespace modelbox

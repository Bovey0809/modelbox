#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


cmake_minimum_required(VERSION 3.10)

if(NOT APPLE)
    return()
endif()

subdirlist(SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR} "CMakeLists.txt")

set(DEMO_APPLE_SILICON_YOLO_DIR "${MODELBOX_DEMO_INSTALL_DIR}/apple_silicon_yolo" CACHE INTERNAL "")
set(DEMO_APPLE_SILICON_YOLO_FLOWUNIT_DIR ${DEMO_APPLE_SILICON_YOLO_DIR}/flowunit CACHE INTERNAL "")
set(DEMO_APPLE_SILICON_YOLO_GRAPH_DIR ${DEMO_APPLE_SILICON_YOLO_DIR}/graph)

foreach(subdir ${SUBDIRS})
    add_subdirectory(${subdir})
endforeach()

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "Do not build in source directory!")
endif()

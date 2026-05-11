#
# Copyright 2021 The Modelbox Project Authors. All Rights Reserved.
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

set(UNIT_DEVICE "cuda")
set(UNIT_NAME "paddle")

project(modelbox-flowunit-${UNIT_DEVICE}-${UNIT_NAME})

if(NOT MODELBOX_PADDLE_FOUND OR NOT CUDA_FOUND)
    message(STATUS "Skip ${UNIT_NAME} cuda flowunit (MODELBOX_PADDLE_FOUND=${MODELBOX_PADDLE_FOUND}, CUDA_FOUND=${CUDA_FOUND})")
    return()
endif()

file(GLOB_RECURSE UNIT_SOURCE *.cpp *.cc *.c)
group_source_test_files(MODELBOX_UNIT_SOURCE MODELBOX_UNIT_TEST_SOURCE "_test.c*" ${UNIT_SOURCE})

include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${LIBMODELBOX_INCLUDE})
include_directories(${LIBMODELBOX_BASE_INCLUDE})
include_directories(${LIBMODELBOX_DEVICE_CUDA_INCLUDE})
include_directories(${LIBMODELBOX_VIRTUALDRIVER_INFERENCE_INCLUDE})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PADDLE_INFERENCE_INCLUDE})
include_directories(${MODELBOX_ENGINE_PADDLE_INCLUDE})

link_directories(${PADDLE_INFERENCE_LIB_DIR})

set(MODELBOX_UNIT_SHARED libmodelbox-unit-${UNIT_DEVICE}-${UNIT_NAME}-shared)
set(MODELBOX_UNIT_SOURCE_INCLUDE ${CMAKE_CURRENT_LIST_DIR})

cuda_add_library(${MODELBOX_UNIT_SHARED} SHARED ${MODELBOX_UNIT_SOURCE})

set_target_properties(${MODELBOX_UNIT_SHARED} PROPERTIES
    SOVERSION ${MODELBOX_VERSION_MAJOR}
    VERSION ${MODELBOX_VERSION_MAJOR}.${MODELBOX_VERSION_MINOR}.${MODELBOX_VERSION_PATCH}
)

target_link_libraries(${MODELBOX_UNIT_SHARED} pthread)
target_link_libraries(${MODELBOX_UNIT_SHARED} rt)
target_link_libraries(${MODELBOX_UNIT_SHARED} dl)
target_link_libraries(${MODELBOX_UNIT_SHARED} ${LIBMODELBOX_DEVICE_CUDA_SHARED})
target_link_libraries(${MODELBOX_UNIT_SHARED} ${LIBMODELBOX_SHARED})
target_link_libraries(${MODELBOX_UNIT_SHARED} ${LIBMODELBOX_VIRTUALDRIVER_INFERENCE_SHARED})
target_link_libraries(${MODELBOX_UNIT_SHARED} ${MODELBOX_ENGINE_PADDLE_TARGET})
target_link_libraries(${MODELBOX_UNIT_SHARED} paddle_inference)

set_target_properties(${MODELBOX_UNIT_SHARED} PROPERTIES OUTPUT_NAME "modelbox-unit-${UNIT_DEVICE}-${UNIT_NAME}")

install(TARGETS ${MODELBOX_UNIT_SHARED}
        COMPONENT cuda-device-flowunit
        RUNTIME DESTINATION ${CMAKE_INSTALL_FULL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
        OPTIONAL)

set(LIBMODELBOX_FLOWUNIT_PADDLE_CUDA_SHARED ${MODELBOX_UNIT_SHARED} CACHE INTERNAL "")
set(LIBMODELBOX_FLOWUNIT_PADDLE_CUDA_INCLUDE ${MODELBOX_UNIT_SOURCE_INCLUDE} CACHE INTERNAL "")
set(LIBMODELBOX_FLOWUNIT_PADDLE_SOURCES ${MODELBOX_UNIT_SOURCE} CACHE INTERNAL "")
set(LIBMODELBOX_FLOWUNIT_PADDLE_SO_PATH ${CMAKE_CURRENT_BINARY_DIR}/libmodelbox-unit-${UNIT_DEVICE}-${UNIT_NAME}.so CACHE INTERNAL "")

# driver test
list(APPEND DRIVER_UNIT_TEST_SOURCE ${MODELBOX_UNIT_TEST_SOURCE})
list(APPEND DRIVER_UNIT_TEST_TARGET ${MODELBOX_UNIT_SHARED})
set(DRIVER_UNIT_TEST_SOURCE ${DRIVER_UNIT_TEST_SOURCE} CACHE INTERNAL "")
set(DRIVER_UNIT_TEST_TARGET ${DRIVER_UNIT_TEST_TARGET} CACHE INTERNAL "")

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
set(SUBDIRECTORY_NAME paddle)
project(modelbox-inference-engine-${SUBDIRECTORY_NAME})

if(NOT DEFINED PaddleInference_DIR AND DEFINED ENV{PADDLE_INFERENCE_DIR})
    set(PaddleInference_DIR $ENV{PADDLE_INFERENCE_DIR})
endif()

if(NOT PaddleInference_DIR OR NOT EXISTS "${PaddleInference_DIR}/paddle/include/paddle_inference_api.h")
    message(STATUS "Paddle Inference SDK not found, disable paddle engine")
    return()
endif()

message(STATUS "MODELBOX_PADDLE_FOUND: using ${PaddleInference_DIR}")

set(MODELBOX_PADDLE_FOUND TRUE CACHE INTERNAL "")
set(PADDLE_INFERENCE_INCLUDE "${PaddleInference_DIR}/paddle/include" CACHE INTERNAL "")
set(PADDLE_INFERENCE_LIB_DIR "${PaddleInference_DIR}/paddle/lib" CACHE INTERNAL "")
set(PADDLE_INFERENCE_DIR "${PaddleInference_DIR}" CACHE INTERNAL "")

include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${LIBMODELBOX_INCLUDE})
include_directories(${LIBMODELBOX_BASE_INCLUDE})
include_directories(${PaddleInference_DIR}/paddle/include)

link_directories(${PaddleInference_DIR}/paddle/lib)

file(GLOB SRC paddle_inference.cc)
add_library(modelbox-engine-paddle STATIC ${SRC})
target_compile_options(modelbox-engine-paddle PRIVATE -fPIC)
target_link_libraries(modelbox-engine-paddle paddle_inference ${LIBMODELBOX_SHARED})

set(MODELBOX_ENGINE_PADDLE_TARGET modelbox-engine-paddle CACHE INTERNAL "")
set(MODELBOX_ENGINE_PADDLE_INCLUDE ${CMAKE_CURRENT_LIST_DIR} CACHE INTERNAL "")

#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
cmake_minimum_required(VERSION 3.10)

project(modelbox-inference-onnxruntime)

if(NOT ONNXRUNTIME_FOUND)
    message(STATUS "Not found onnxruntime, disable onnxruntime engine")
    return()
endif()

file(GLOB_RECURSE MODELBOX_UNIT_SOURCE *.cpp *.cc *.c)

include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${LIBMODELBOX_INCLUDE})
include_directories(${LIBMODELBOX_BASE_INCLUDE})
include_directories(${LIBMODELBOX_VIRTUALDRIVER_INFERENCE_INCLUDE})
include_directories(${ONNXRUNTIME_INCLUDE_DIR})

set(MODELBOX_UNIT_SHARED libmodelbox-engine-onnxruntime)
set(MODELBOX_UNIT_SOURCE_INCLUDE ${CMAKE_CURRENT_LIST_DIR})

add_library(${MODELBOX_UNIT_SHARED} SHARED ${MODELBOX_UNIT_SOURCE})

# ORT 1.16+ public headers use C++17 (noexcept in typedefs, no-message
# static_assert). Bump the standard for this target only; rest of modelbox
# stays at C++11.
set_target_properties(${MODELBOX_UNIT_SHARED} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    SOVERSION ${MODELBOX_VERSION_MAJOR}
    VERSION ${MODELBOX_VERSION_MAJOR}.${MODELBOX_VERSION_MINOR}.${MODELBOX_VERSION_PATCH}
)

target_link_libraries(${MODELBOX_UNIT_SHARED} pthread)
if(NOT APPLE)
    target_link_libraries(${MODELBOX_UNIT_SHARED} rt)
endif()
target_link_libraries(${MODELBOX_UNIT_SHARED} dl)
target_link_libraries(${MODELBOX_UNIT_SHARED} ${ONNXRUNTIME_LIBRARIES})
target_link_libraries(${MODELBOX_UNIT_SHARED} ${LIBMODELBOX_SHARED})
target_link_libraries(${MODELBOX_UNIT_SHARED} ${LIBMODELBOX_VIRTUALDRIVER_INFERENCE_SHARED})

set_target_properties(${MODELBOX_UNIT_SHARED} PROPERTIES OUTPUT_NAME "modelbox-engine-onnxruntime")

install(TARGETS ${MODELBOX_UNIT_SHARED}
        COMPONENT cpu-device-flowunit
        RUNTIME DESTINATION ${CMAKE_INSTALL_FULL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
        OPTIONAL
)

set(LIBMODELBOX_FLOWUNIT_INFERENCE_ONNXRUNTIME_SHARED ${MODELBOX_UNIT_SHARED} CACHE INTERNAL "")
set(LIBMODELBOX_FLOWUNIT_INFERENCE_ONNXRUNTIME_INCLUDE ${MODELBOX_UNIT_SOURCE_INCLUDE} CACHE INTERNAL "")
set(LIBMODELBOX_FLOWUNIT_INFERENCE_ONNXRUNTIME_SOURCES ${MODELBOX_UNIT_SOURCE} CACHE INTERNAL "")
set(LIBMODELBOX_FLOWUNIT_INFERENCE_ONNXRUNTIME_SO_PATH ${CMAKE_CURRENT_BINARY_DIR}/libmodelbox-engine-onnxruntime.so CACHE INTERNAL "")

#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#

cmake_minimum_required(VERSION 3.10)
set(UNIT_DEVICE "cpu")
set(UNIT_NAME "tracknet_frame_stacker")
project(modelbox-flowunit-${UNIT_DEVICE}-${UNIT_NAME})

set(UNIT_SOURCE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/${UNIT_NAME}_flowunit.cc)
set(UNIT_TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/${UNIT_NAME}_test.cc)

include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${LIBMODELBOX_INCLUDE})
include_directories(${LIBMODELBOX_BASE_INCLUDE})
include_directories(${LIBMODELBOX_DEVICE_CPU_INCLUDE})
include_directories(${HUAWEI_SECURE_C_INCLUDE_DIR})

set(MODELBOX_UNIT_SHARED modelbox-unit-${UNIT_DEVICE}-${UNIT_NAME}-shared)
add_library(${MODELBOX_UNIT_SHARED} SHARED ${UNIT_SOURCE_FILE})
set_target_properties(${MODELBOX_UNIT_SHARED} PROPERTIES
    SOVERSION ${MODELBOX_VERSION_MAJOR}
    VERSION ${MODELBOX_VERSION_MAJOR}.${MODELBOX_VERSION_MINOR}.${MODELBOX_VERSION_PATCH}
)
target_link_libraries(${MODELBOX_UNIT_SHARED}
    ${LIBMODELBOX_DEVICE_CPU_SHARED} ${LIBMODELBOX_SHARED} pthread dl)
set_target_properties(${MODELBOX_UNIT_SHARED} PROPERTIES
    OUTPUT_NAME "modelbox-unit-${UNIT_DEVICE}-${UNIT_NAME}")

install(TARGETS ${MODELBOX_UNIT_SHARED}
    COMPONENT demo
    LIBRARY DESTINATION ${DEMO_APPLE_SILICON_YOLO_FLOWUNIT_DIR}/${UNIT_NAME}
    OPTIONAL)

# Standalone gtest binary for the pure stacking helper.
set(UNIT_TEST_TARGET ${UNIT_NAME}_test)
add_executable(${UNIT_TEST_TARGET} ${UNIT_TEST_FILE} ${UNIT_SOURCE_FILE})
target_include_directories(${UNIT_TEST_TARGET} PRIVATE
    ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
target_link_libraries(${UNIT_TEST_TARGET}
    gtest_main gtest
    ${LIBMODELBOX_DEVICE_CPU_SHARED} ${LIBMODELBOX_SHARED} pthread dl)
set_target_properties(${UNIT_TEST_TARGET} PROPERTIES
    BUILD_WITH_INSTALL_RPATH OFF
    SKIP_BUILD_RPATH OFF
    BUILD_RPATH "${CMAKE_BINARY_DIR}/src/libmodelbox;${CMAKE_BINARY_DIR}/src/drivers/devices/cpu/core;${CMAKE_BINARY_DIR}/thirdparty/download/build/Huawei_Secure_C")
add_test(NAME ${UNIT_TEST_TARGET} COMMAND ${UNIT_TEST_TARGET})

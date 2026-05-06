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

project(modelbox-inference-coreml)

if (NOT APPLE OR NOT COREML_FOUND)
    message(STATUS "Not on Apple or coreml not found, disable coreml engine")
    return()
endif()

file(GLOB_RECURSE MODELBOX_UNIT_SOURCE *.cpp *.cc *.c *.mm)

# .mm sources are Objective-C++; tag them so clang invokes the ObjC frontend
# even though the project is otherwise plain C++. ARC keeps the Core ML
# Objective-C objects' lifetimes scoped to their enclosing C++ scope.
foreach(src ${MODELBOX_UNIT_SOURCE})
    if(${src} MATCHES "\\.mm$")
        set_source_files_properties(${src} PROPERTIES
            COMPILE_FLAGS "-fobjc-arc")
    endif()
endforeach()

include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${LIBMODELBOX_INCLUDE})
include_directories(${LIBMODELBOX_BASE_INCLUDE})
include_directories(${LIBMODELBOX_VIRTUALDRIVER_INFERENCE_INCLUDE})

set(MODELBOX_UNIT_SHARED libmodelbox-engine-coreml)
set(MODELBOX_UNIT_SOURCE_INCLUDE ${CMAKE_CURRENT_LIST_DIR})

add_library(${MODELBOX_UNIT_SHARED} SHARED ${MODELBOX_UNIT_SOURCE})

set_target_properties(${MODELBOX_UNIT_SHARED} PROPERTIES
    SOVERSION ${MODELBOX_VERSION_MAJOR}
    VERSION ${MODELBOX_VERSION_MAJOR}.${MODELBOX_VERSION_MINOR}.${MODELBOX_VERSION_PATCH}
)

target_link_libraries(${MODELBOX_UNIT_SHARED} pthread)
target_link_libraries(${MODELBOX_UNIT_SHARED} dl)
target_link_libraries(${MODELBOX_UNIT_SHARED} ${COREML_LIBRARIES})
target_link_libraries(${MODELBOX_UNIT_SHARED} ${LIBMODELBOX_SHARED})
target_link_libraries(${MODELBOX_UNIT_SHARED} ${LIBMODELBOX_VIRTUALDRIVER_INFERENCE_SHARED})

set_target_properties(${MODELBOX_UNIT_SHARED} PROPERTIES OUTPUT_NAME "modelbox-engine-coreml")

install(TARGETS ${MODELBOX_UNIT_SHARED}
        COMPONENT apple-silicon-device-flowunit
        RUNTIME DESTINATION ${CMAKE_INSTALL_FULL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
        OPTIONAL
        )

set(LIBMODELBOX_FLOWUNIT_INFERENCE_COREML_SHARED ${MODELBOX_UNIT_SHARED} CACHE INTERNAL "")
set(LIBMODELBOX_FLOWUNIT_INFERENCE_COREML_INCLUDE ${MODELBOX_UNIT_SOURCE_INCLUDE} CACHE INTERNAL "")
set(LIBMODELBOX_FLOWUNIT_INFERENCE_COREML_SOURCES ${MODELBOX_UNIT_SOURCE} CACHE INTERNAL "")
set(LIBMODELBOX_FLOWUNIT_INFERENCE_COREML_SO_PATH ${CMAKE_CURRENT_BINARY_DIR}/libmodelbox-engine-coreml.dylib CACHE INTERNAL "")

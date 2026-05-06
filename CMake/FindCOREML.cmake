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

# Locate Apple Core ML on macOS. The framework search uses Xcode's SDK so we
# don't need explicit hint paths. Linux/Windows hosts: COREML_FOUND stays FALSE.

if(APPLE)
    find_path(COREML_INCLUDE
        NAMES CoreML/CoreML.h
        PATHS ${CMAKE_OSX_SYSROOT}/System/Library/Frameworks/CoreML.framework/Headers
        NO_DEFAULT_PATH
    )
    if(NOT COREML_INCLUDE)
        # FRAMEWORK headers are addressed via -framework, so a successful
        # framework lookup is sufficient even when the explicit path is empty.
        find_library(COREML_FRAMEWORK NAMES CoreML)
        find_library(FOUNDATION_FRAMEWORK NAMES Foundation)
        if(COREML_FRAMEWORK AND FOUNDATION_FRAMEWORK)
            set(COREML_INCLUDE ${COREML_FRAMEWORK})
        endif()
    endif()
    mark_as_advanced(COREML_INCLUDE)

    find_library(COREML_LIBRARY NAMES CoreML)
    find_library(FOUNDATION_LIBRARY NAMES Foundation)
    mark_as_advanced(COREML_LIBRARY FOUNDATION_LIBRARY)
endif()

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(COREML
    REQUIRED_VARS COREML_LIBRARY FOUNDATION_LIBRARY)

if(COREML_FOUND)
    set(COREML_LIBRARIES "-framework CoreML" "-framework Foundation")
    set(COREML_INCLUDE_DIR ${COREML_INCLUDE})
endif()

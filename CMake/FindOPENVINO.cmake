set(HINTS_OPENVINO_PATH $ENV{OPENVINO_ROOT})

find_path(OPENVINO_INCLUDE
  NAMES openvino/openvino.hpp
  HINTS ${HINTS_OPENVINO_PATH}/runtime/include
        /opt/intel/openvino_2025/runtime/include
        /opt/intel/openvino_2024/runtime/include
        /opt/intel/openvino/runtime/include
        ${CMAKE_INSTALL_FULL_INCLUDEDIR}
        /usr/include
)
mark_as_advanced(OPENVINO_INCLUDE)

# Look for the library (sorted from most current/relevant entry to least).
find_library(OPENVINO_LIBRARY NAMES
    openvino
    HINTS ${HINTS_OPENVINO_PATH}/runtime/lib/intel64
          /opt/intel/openvino_2025/runtime/lib/intel64
          /opt/intel/openvino_2024/runtime/lib/intel64
          /opt/intel/openvino/runtime/lib/intel64
          ${CMAKE_INSTALL_FULL_LIBDIR}
          /usr/lib/x86_64-linux-gnu
)
mark_as_advanced(OPENVINO_LIBRARY)

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(OPENVINO
                                  REQUIRED_VARS OPENVINO_LIBRARY OPENVINO_INCLUDE
                                  VERSION_VAR OPENVINO_VERSION_STRING)

if(OPENVINO_FOUND)
  set(OPENVINO_LIBRARIES ${OPENVINO_LIBRARY})
  set(OPENVINO_INCLUDE_DIR ${OPENVINO_INCLUDE})
endif()

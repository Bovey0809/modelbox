set(HINTS_LEVEL_ZERO_PATH $ENV{LEVEL_ZERO_ROOT})

find_path(INTELGPU_INCLUDE
  NAMES level_zero/ze_api.h
  HINTS ${HINTS_LEVEL_ZERO_PATH}/include
        ${CMAKE_INSTALL_FULL_INCLUDEDIR}
        /usr/include
)
mark_as_advanced(INTELGPU_INCLUDE)

find_library(INTELGPU_LIBRARY NAMES
    ze_loader
    HINTS ${HINTS_LEVEL_ZERO_PATH}/lib
          ${CMAKE_INSTALL_FULL_LIBDIR}
          /usr/lib/x86_64-linux-gnu
)
mark_as_advanced(INTELGPU_LIBRARY)

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(INTELGPU
                                  REQUIRED_VARS INTELGPU_LIBRARY INTELGPU_INCLUDE
                                  VERSION_VAR INTELGPU_VERSION_STRING)

if(INTELGPU_FOUND)
  set(INTELGPU_LIBRARIES ${INTELGPU_LIBRARY})
  set(INTELGPU_INCLUDE_DIR ${INTELGPU_INCLUDE})
endif()

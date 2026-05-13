# Locate ONNX Runtime (https://onnxruntime.ai).
#
# Inputs:
#   ONNXRUNTIME_ROOT (env or var) — install prefix.
#
# Outputs:
#   ONNXRUNTIME_FOUND
#   ONNXRUNTIME_INCLUDE_DIR — directory containing onnxruntime_cxx_api.h
#   ONNXRUNTIME_LIBRARIES   — libonnxruntime[.so|.dylib]
set(_ort_hints
    ${ONNXRUNTIME_ROOT}
    $ENV{ONNXRUNTIME_ROOT}
    /opt/homebrew
    /opt/homebrew/opt/onnxruntime
    /opt/onnxruntime
    /usr/local/onnxruntime
    /usr/local
    /usr
)

find_path(ONNXRUNTIME_INCLUDE
    NAMES onnxruntime_cxx_api.h
    PATH_SUFFIXES include include/onnxruntime include/onnxruntime/core/session
    HINTS ${_ort_hints}
)
mark_as_advanced(ONNXRUNTIME_INCLUDE)

find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATH_SUFFIXES lib lib64
    HINTS ${_ort_hints}
)
mark_as_advanced(ONNXRUNTIME_LIBRARY)

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(ONNXRUNTIME
    REQUIRED_VARS ONNXRUNTIME_LIBRARY ONNXRUNTIME_INCLUDE)

if(ONNXRUNTIME_FOUND)
    set(ONNXRUNTIME_LIBRARIES ${ONNXRUNTIME_LIBRARY})
    set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_INCLUDE})
endif()

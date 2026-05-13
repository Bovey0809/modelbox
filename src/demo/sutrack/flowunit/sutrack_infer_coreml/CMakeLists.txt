#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
cmake_minimum_required(VERSION 3.10)

# CoreML is Apple-only; skip on Linux builds.
if(NOT APPLE)
    return()
endif()

set(FLOWUNIT_NAME "sutrack_infer_coreml")
set(FLOWUNIT_PATH ${CMAKE_CURRENT_BINARY_DIR}/${FLOWUNIT_NAME})
file(MAKE_DIRECTORY ${FLOWUNIT_PATH})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${FLOWUNIT_NAME}.toml
               ${FLOWUNIT_PATH}/${FLOWUNIT_NAME}.toml @ONLY)

set(SUTRACK_MLPACKAGE_SRC "${CMAKE_SOURCE_DIR}/assets/sutrack/sutrack_b224.mlpackage")
set(SUTRACK_MLPACKAGE_DST "${FLOWUNIT_PATH}/sutrack_b224.mlpackage")
set(SUTRACK_MLPACKAGE_STAMP_SRC "${SUTRACK_MLPACKAGE_SRC}/Manifest.json")
set(SUTRACK_MLPACKAGE_STAMP_DST "${SUTRACK_MLPACKAGE_DST}/Manifest.json")

if(EXISTS ${SUTRACK_MLPACKAGE_STAMP_SRC})
    add_custom_command(
        OUTPUT  ${SUTRACK_MLPACKAGE_STAMP_DST}
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${SUTRACK_MLPACKAGE_SRC} ${SUTRACK_MLPACKAGE_DST}
        DEPENDS ${SUTRACK_MLPACKAGE_STAMP_SRC}
        VERBATIM
    )
    add_custom_target(${FLOWUNIT_NAME}_model ALL DEPENDS ${SUTRACK_MLPACKAGE_STAMP_DST})
else()
    message(WARNING
        "sutrack_infer_coreml: missing .mlpackage at ${SUTRACK_MLPACKAGE_SRC}. "
        "Run models/sutrack/deploy/coreml/convert_torchscript_to_coreml.py before configuring.")
endif()

install(DIRECTORY ${FLOWUNIT_PATH}
    DESTINATION ${DEMO_SUTRACK_FLOWUNIT_DIR}
    COMPONENT demo USE_SOURCE_PERMISSIONS
)

#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
cmake_minimum_required(VERSION 3.10)

set(FLOWUNIT_NAME "sutrack_infer_ort")
set(FLOWUNIT_PATH ${CMAKE_CURRENT_BINARY_DIR}/${FLOWUNIT_NAME})
file(MAKE_DIRECTORY ${FLOWUNIT_PATH})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${FLOWUNIT_NAME}.toml
               ${FLOWUNIT_PATH}/${FLOWUNIT_NAME}.toml @ONLY)

set(SUTRACK_ONNX_SRC "${CMAKE_SOURCE_DIR}/assets/sutrack/sutrack_b224.onnx")
set(SUTRACK_ONNX_DST "${FLOWUNIT_PATH}/sutrack_b224.onnx")
if(EXISTS ${SUTRACK_ONNX_SRC})
    add_custom_command(
        OUTPUT  ${SUTRACK_ONNX_DST}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${SUTRACK_ONNX_SRC} ${SUTRACK_ONNX_DST}
        DEPENDS ${SUTRACK_ONNX_SRC}
        VERBATIM
    )
    add_custom_target(${FLOWUNIT_NAME}_model ALL DEPENDS ${SUTRACK_ONNX_DST})
else()
    message(WARNING "sutrack_infer_ort: missing ONNX at ${SUTRACK_ONNX_SRC}")
endif()

install(DIRECTORY ${FLOWUNIT_PATH}
    DESTINATION ${DEMO_SUTRACK_FLOWUNIT_DIR}
    COMPONENT demo
)

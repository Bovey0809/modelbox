#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
cmake_minimum_required(VERSION 3.10)

set(FLOWUNIT_NAME "hit_sound_infer_ort")
set(FLOWUNIT_PATH ${CMAKE_CURRENT_BINARY_DIR}/${FLOWUNIT_NAME})
file(MAKE_DIRECTORY ${FLOWUNIT_PATH})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${FLOWUNIT_NAME}.toml
               ${FLOWUNIT_PATH}/${FLOWUNIT_NAME}.toml @ONLY)

set(HIT_SOUND_ONNX_SRC "${CMAKE_SOURCE_DIR}/assets/hit_sound/best.onnx")
set(HIT_SOUND_ONNX_DST "${FLOWUNIT_PATH}/best.onnx")
if(EXISTS ${HIT_SOUND_ONNX_SRC})
    add_custom_command(
        OUTPUT ${HIT_SOUND_ONNX_DST}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${HIT_SOUND_ONNX_SRC} ${HIT_SOUND_ONNX_DST}
        DEPENDS ${HIT_SOUND_ONNX_SRC}
        VERBATIM
    )
    add_custom_target(${FLOWUNIT_NAME}_model ALL DEPENDS ${HIT_SOUND_ONNX_DST})
endif()

install(DIRECTORY ${FLOWUNIT_PATH}
    DESTINATION ${DEMO_HIT_SOUND_FLOWUNIT_DIR}
    COMPONENT demo
)

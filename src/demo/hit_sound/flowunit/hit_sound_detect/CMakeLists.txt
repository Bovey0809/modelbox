#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
cmake_minimum_required(VERSION 3.10)

set(FLOWUNIT_NAME "hit_sound_detect")
set(FLOWUNIT_PATH ${CMAKE_CURRENT_BINARY_DIR}/${FLOWUNIT_NAME})
file(MAKE_DIRECTORY ${FLOWUNIT_PATH})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${FLOWUNIT_NAME}.py
               ${FLOWUNIT_PATH}/${FLOWUNIT_NAME}.py @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/hit_sound_utils.py
               ${FLOWUNIT_PATH}/hit_sound_utils.py @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${FLOWUNIT_NAME}.toml
               ${FLOWUNIT_PATH}/${FLOWUNIT_NAME}.toml @ONLY)

# Stage the ONNX weights next to the python flowunit so the relative
# model_path = "best.onnx" resolves at runtime.
set(HIT_SOUND_ONNX_SRC "${CMAKE_SOURCE_DIR}/assets/hit_sound/best.onnx")
set(HIT_SOUND_ONNX_DST "${FLOWUNIT_PATH}/best.onnx")
if(EXISTS ${HIT_SOUND_ONNX_SRC})
    add_custom_command(
        OUTPUT  ${HIT_SOUND_ONNX_DST}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${HIT_SOUND_ONNX_SRC} ${HIT_SOUND_ONNX_DST}
        DEPENDS ${HIT_SOUND_ONNX_SRC}
        VERBATIM
    )
    add_custom_target(${FLOWUNIT_NAME}_model ALL DEPENDS ${HIT_SOUND_ONNX_DST})
else()
    message(WARNING
        "hit_sound_detect: missing ONNX at ${HIT_SOUND_ONNX_SRC}. "
        "Place tennis_audio_seg best.onnx there before building, or copy it "
        "into ${FLOWUNIT_PATH}/ manually before running the demo.")
endif()

install(DIRECTORY
    ${FLOWUNIT_PATH}
    DESTINATION ${DEMO_HIT_SOUND_FLOWUNIT_DIR}
    COMPONENT demo
)

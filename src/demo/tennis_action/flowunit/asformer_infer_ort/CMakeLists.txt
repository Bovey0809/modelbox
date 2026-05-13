#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
cmake_minimum_required(VERSION 3.10)

set(FLOWUNIT_NAME "asformer_infer_ort")
set(FLOWUNIT_PATH ${CMAKE_CURRENT_BINARY_DIR}/${FLOWUNIT_NAME})
file(MAKE_DIRECTORY ${FLOWUNIT_PATH})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${FLOWUNIT_NAME}.toml
               ${FLOWUNIT_PATH}/${FLOWUNIT_NAME}.toml @ONLY)

set(ASFORMER_ONNX_SRC "${CMAKE_SOURCE_DIR}/assets/tennis_action/asformer.onnx")
set(ASFORMER_ONNX_DST "${FLOWUNIT_PATH}/asformer.onnx")
if(EXISTS ${ASFORMER_ONNX_SRC})
    add_custom_command(
        OUTPUT ${ASFORMER_ONNX_DST}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${ASFORMER_ONNX_SRC} ${ASFORMER_ONNX_DST}
        DEPENDS ${ASFORMER_ONNX_SRC}
        VERBATIM
    )
    add_custom_target(${FLOWUNIT_NAME}_model ALL DEPENDS ${ASFORMER_ONNX_DST})
else()
    message(WARNING
        "${FLOWUNIT_NAME}: ${ASFORMER_ONNX_SRC} missing; graph will fail at "
        "open time. Run scripts/build_tennis_action_stub_onnx.py or drop a "
        "real model in.")
endif()

set(ASFORMER_META_SRC "${CMAKE_SOURCE_DIR}/assets/tennis_action/asformer_meta.json")
if(EXISTS ${ASFORMER_META_SRC})
    install(FILES ${ASFORMER_META_SRC}
        DESTINATION ${DEMO_TENNIS_ACTION_ASSETS_DIR}
        COMPONENT demo)
else()
    message(WARNING
        "${FLOWUNIT_NAME}: ${ASFORMER_META_SRC} missing; hit_fuse will fail "
        "at open() time. Generate via scripts/build_tennis_action_stub_onnx.py "
        "or drop a real meta.json in.")
endif()

install(DIRECTORY ${FLOWUNIT_PATH}
    DESTINATION ${DEMO_TENNIS_ACTION_FLOWUNIT_DIR}
    COMPONENT demo
)

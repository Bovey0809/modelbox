#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
cmake_minimum_required(VERSION 3.10)

set(FLOWUNIT_NAME "yolo_pose_detect_ort")
set(FLOWUNIT_PATH ${CMAKE_CURRENT_BINARY_DIR}/${FLOWUNIT_NAME})
file(MAKE_DIRECTORY ${FLOWUNIT_PATH})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${FLOWUNIT_NAME}.toml
               ${FLOWUNIT_PATH}/${FLOWUNIT_NAME}.toml @ONLY)

set(YOLO_POSE_ONNX_SRC "${CMAKE_SOURCE_DIR}/assets/tennis_action/yolov8n-pose.onnx")
set(YOLO_POSE_ONNX_DST "${FLOWUNIT_PATH}/yolov8n-pose.onnx")
if(EXISTS ${YOLO_POSE_ONNX_SRC})
    add_custom_command(
        OUTPUT ${YOLO_POSE_ONNX_DST}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${YOLO_POSE_ONNX_SRC} ${YOLO_POSE_ONNX_DST}
        DEPENDS ${YOLO_POSE_ONNX_SRC}
        VERBATIM
    )
    add_custom_target(${FLOWUNIT_NAME}_model ALL DEPENDS ${YOLO_POSE_ONNX_DST})
else()
    message(WARNING
        "${FLOWUNIT_NAME}: ${YOLO_POSE_ONNX_SRC} missing; cpu_stub graph will "
        "fail at open time. Run scripts/build_tennis_action_stub_onnx.py.")
endif()

install(DIRECTORY ${FLOWUNIT_PATH}
    DESTINATION ${DEMO_TENNIS_ACTION_FLOWUNIT_DIR}
    COMPONENT demo
)

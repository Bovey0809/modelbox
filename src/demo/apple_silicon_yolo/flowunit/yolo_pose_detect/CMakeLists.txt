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


cmake_minimum_required(VERSION 3.10)

set(FLOWUNIT_NAME "yolo_pose_detect")

set(FLOWUNIT_PATH ${CMAKE_CURRENT_BINARY_DIR}/${FLOWUNIT_NAME})
set(FLOWUNIT_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/${FLOWUNIT_NAME}.toml)
set(EXPORT_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/../../scripts/export_yolo_pose_coreml.py)
set(MODEL_DIR ${FLOWUNIT_PATH}/yolov8n-pose.mlpackage)
set(MODEL_STAMP ${MODEL_DIR}/Manifest.json)

file(MAKE_DIRECTORY ${FLOWUNIT_PATH})
configure_file(${FLOWUNIT_CONFIG} ${FLOWUNIT_PATH}/${FLOWUNIT_NAME}.toml @ONLY)

add_custom_command(
    OUTPUT ${MODEL_STAMP}
    COMMAND ${CMAKE_COMMAND} -E echo "Exporting yolov8n-pose.mlpackage..."
    COMMAND ${PYTHON_EXECUTABLE} ${EXPORT_SCRIPT} --out ${MODEL_DIR} --imgsz 640
    DEPENDS ${EXPORT_SCRIPT}
    VERBATIM
)
add_custom_target(${FLOWUNIT_NAME}_model ALL DEPENDS ${MODEL_STAMP})

install(DIRECTORY ${FLOWUNIT_PATH}
    DESTINATION ${DEMO_APPLE_SILICON_YOLO_FLOWUNIT_DIR}
    COMPONENT demo USE_SOURCE_PERMISSIONS)
